import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    balanced_accuracy_score, roc_curve
)

CSV_PATH = "/Folder Directory/UCSF-PDGM-metadata_v5.csv"
IMAGE_FOLDER = "preprocessed_g_96_501"

TDA_TRAIN_PATH = "X_train_fs.npy"
TDA_VAL_PATH   = "X_val_fs.npy"
TDA_TEST_PATH  = "X_test_fs.npy"

BEST_MODEL_PATH = "best_lfx3d_323.pth"
BATCH_SIZE = 2
EPOCHS = 50
LR = 1e-4
RANDOM_STATE = 42

TDA_DIM = 364
NUM_CLASSES = 2

cnn_embed_dim = 128
tda_embed_dim = 128
class_embed_dim1 = 128

patience1 = 3
patience2 = 12

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.benchmark = True

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

def normalize_ucsf_id(pid):
    num = str(pid).split("-")[-1].zfill(3)
    return f"UCSF-PDGM-{num}"

df = pd.read_csv(CSV_PATH)

df = df[df["WHO CNS Grade"].notna()]
df = df[df["WHO CNS Grade"].isin([2, 3, 4])]

df["label"] = df["WHO CNS Grade"].apply(lambda x: 1 if x == 4 else 0)
df["file_id"] = df["ID"].apply(normalize_ucsf_id)

print("Total usable patients:", len(df))
print(df["label"].value_counts())

X_temp, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=df["label"]
)

train_df, val_df = train_test_split(
    X_temp,
    test_size=0.125,
    random_state=RANDOM_STATE,
    stratify=X_temp["label"]
)

print("Train:", len(train_df))
print("Val  :", len(val_df))
print("Test :", len(test_df))

X_train_tda = np.load(TDA_TRAIN_PATH).astype(np.float32)
X_val_tda   = np.load(TDA_VAL_PATH).astype(np.float32)
X_test_tda  = np.load(TDA_TEST_PATH).astype(np.float32)

print("TDA train shape:", X_train_tda.shape)
print("TDA val shape  :", X_val_tda.shape)
print("TDA test shape :", X_test_tda.shape)

TDA_DIM = X_train_tda.shape[1]

tda_scaler = StandardScaler()

X_train_tda = tda_scaler.fit_transform(X_train_tda).astype(np.float32)
X_val_tda   = tda_scaler.transform(X_val_tda).astype(np.float32)
X_test_tda  = tda_scaler.transform(X_test_tda).astype(np.float32)

class UCSFX3DTDAFusionDataset(Dataset):

    def __init__(self, dataframe, image_folder, tda_features):
        self.df = dataframe.reset_index(drop=True)
        self.image_folder = image_folder
        self.tda_features = tda_features

        assert len(self.df) == len(self.tda_features), \
            f"Mismatch: df={len(self.df)}, tda={len(self.tda_features)}"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        pid = row["file_id"]
        label = int(row["label"])

        image_path = os.path.join(
            self.image_folder,
            f"{pid}.npy"
        )

        image = np.load(image_path).astype(np.float32)
        
        if image.shape[-1] == 4:
            image = np.transpose(image, (3, 0, 1, 2))

        image = torch.from_numpy(image)

        tda = torch.tensor(
            self.tda_features[idx],
            dtype=torch.float32
        )

        label = torch.tensor(label, dtype=torch.long)

        return image, tda, label

train_dataset = UCSFX3DTDAFusionDataset(
    train_df,
    IMAGE_FOLDER,
    X_train_tda
)

val_dataset = UCSFX3DTDAFusionDataset(
    val_df,
    IMAGE_FOLDER,
    X_val_tda
)

test_dataset = UCSFX3DTDAFusionDataset(
    test_df,
    IMAGE_FOLDER,
    X_test_tda
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=True
)

class X3DTDALateFusion(nn.Module):

    def __init__(self, tda_dim, num_classes=2):
        super().__init__()

        self.adapter = nn.Sequential(
            nn.Conv3d(4, 3, kernel_size=1, bias=False),
            nn.BatchNorm3d(3),
            nn.ReLU(inplace=True)
        )

        x3d_full = torch.hub.load(
            "facebookresearch/pytorchvideo",
            "x3d_m",
            pretrained=True
        )

        self.x3d_backbone = nn.Sequential(
            *list(x3d_full.blocks[:-1])
        )

        self.pool = nn.AdaptiveAvgPool3d(1)

        self.cnn_dim = 192

        self.cnn_projection = nn.Sequential(
            nn.Linear(192, cnn_embed_dim),
            nn.LayerNorm(cnn_embed_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.tda_projection = nn.Sequential(
            nn.Linear(tda_dim, tda_embed_dim),
            nn.LayerNorm(tda_embed_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        fusion_dim = cnn_embed_dim + tda_embed_dim

        print("X3D dim before projection :", 192)
        print("TDA dim before projection :", tda_dim)
        print("Fusion dim after projection:", fusion_dim)

        self.classifier = nn.Sequential(

            nn.Linear(fusion_dim, class_embed_dim1),
            nn.LayerNorm(class_embed_dim1),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(class_embed_dim1, num_classes)
        )

    def forward(self, image, tda):

        image = self.adapter(image)

        feat = self.x3d_backbone(image)

        feat = self.pool(feat)

        feat = torch.flatten(feat, 1)

        feat = self.cnn_projection(feat)      

        tda = self.tda_projection(tda)        

        fused = torch.cat([feat, tda], dim=1) 

        logits = self.classifier(fused)

        return logits

model = X3DTDALateFusion(
    tda_dim=TDA_DIM,
    num_classes=NUM_CLASSES
).to(DEVICE)

class_counts = train_df["label"].value_counts().sort_index().values

class_weights = 1.0 / (class_counts ** 0.10)
class_weights = class_weights / class_weights.mean()

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32
).to(DEVICE)

print("Class weights:", class_weights)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=1e-5
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=patience1
)

scaler = GradScaler(enabled=torch.cuda.is_available())

def evaluate(model, loader, threshold=0.5):

    model.eval()

    all_probs = []
    all_preds = []
    all_labels = []

    with torch.no_grad():

        for image, tda, y in loader:

            image = image.to(DEVICE, non_blocking=True)
            tda = tda.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            with autocast(
                device_type="cuda",
                enabled=torch.cuda.is_available()
            ):
                logits = model(image, tda)

            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = (probs > threshold).long()

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    auc = roc_auc_score(all_labels, all_probs)
    acc = accuracy_score(all_labels, all_preds)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(
        all_labels,
        all_preds
    ).ravel()

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        "auc": auc,
        "acc": acc,
        "f1": f1,
        "sensitivity": sens,
        "specificity": spec
    }

def get_probs_labels(model, loader):

    model.eval()

    probs_all = []
    labels_all = []

    with torch.no_grad():

        for image, tda, y in loader:

            image = image.to(DEVICE, non_blocking=True)
            tda = tda.to(DEVICE, non_blocking=True)

            with autocast(
                device_type="cuda",
                enabled=torch.cuda.is_available()
            ):
                logits = model(image, tda)

            probs = torch.softmax(logits, dim=1)[:, 1]

            probs_all.extend(probs.cpu().numpy())
            labels_all.extend(y.numpy())

    return np.array(probs_all), np.array(labels_all)

best_auc = 0
epochs_without_improvement = 0
min_delta = 0.001

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0

    for image, tda, y in tqdm(train_loader):

        image = image.to(DEVICE, non_blocking=True)
        tda = tda.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        with autocast(
            device_type="cuda",
            enabled=torch.cuda.is_available()
        ):
            logits = model(image, tda)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)

    val_metrics = evaluate(
        model,
        val_loader,
        threshold=0.5
    )

    scheduler.step(val_metrics["auc"])

    current_lr = optimizer.param_groups[0]["lr"]

    print("\n==")
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print("Train Loss :", round(train_loss, 4))
    print("Val AUC    :", round(val_metrics["auc"], 4))
    print("Val Acc    :", round(val_metrics["acc"], 4))
    print("Val F1     :", round(val_metrics["f1"], 4))
    print("Val Sens   :", round(val_metrics["sensitivity"], 4))
    print("Val Spec   :", round(val_metrics["specificity"], 4))
    print("LR         :", current_lr)

    if val_metrics["auc"] > best_auc + min_delta:

        best_auc = val_metrics["auc"]
        epochs_without_improvement = 0

        torch.save(
            model.state_dict(),
            BEST_MODEL_PATH
        )

        print("Best X3D+TDA fusion model saved.")

    else:

        epochs_without_improvement += 1

        print(
            f"No improvement for "
            f"{epochs_without_improvement} epoch(s)."
        )

    if epochs_without_improvement >= patience2:

        print("\nEarly stopping triggered.")
        break

model.load_state_dict(
    torch.load(
        BEST_MODEL_PATH,
        map_location=DEVICE
    )
)

val_probs, val_labels = get_probs_labels(
    model,
    val_loader
)

fpr, tpr, thresholds = roc_curve(
    val_labels,
    val_probs
)

j_scores = tpr - fpr
best_idx = np.argmax(j_scores)
best_threshold = thresholds[best_idx]

print("\nBest validation threshold:", round(best_threshold, 4))

val_auc = roc_auc_score(
    val_labels,
    val_probs
)

print("\nValidation AUC:", round(val_auc, 4))

for th in [0.5, 0.6, 0.7, 0.8]:

    preds = (val_probs > th).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        val_labels,
        preds
    ).ravel()

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0

    acc = accuracy_score(
        val_labels,
        preds
    )

    bal_acc = balanced_accuracy_score(
        val_labels,
        preds
    )

    f1 = f1_score(
        val_labels,
        preds,
        zero_division=0
    )

    print(
        f"Val Threshold={th:.1f} | "
        f"AUC={val_auc:.4f} | "
        f"Acc={acc:.4f} | "
        f"F1={f1:.4f} | "
        f"Sens={sens:.4f} | "
        f"Spec={spec:.4f}"
    )

test_metrics = evaluate(
    model,
    test_loader,
    threshold=best_threshold
)

test_probs, test_labels = get_probs_labels(
    model,
    test_loader
)

print("\nTest probability statistics:")
print("Prob min :", round(test_probs.min(), 4))
print("Prob max :", round(test_probs.max(), 4))
print("Prob mean:", round(test_probs.mean(), 4))

print(
    "Labels:",
    np.unique(
        test_labels,
        return_counts=True
    )
)

print(
    "Predictions:",
    np.unique(
        (test_probs > best_threshold).astype(int),
        return_counts=True
    )
)

print("\n FINAL TEST RESULTS")

for k, v in test_metrics.items():
    print(f"{k}: {round(v, 4)}")
