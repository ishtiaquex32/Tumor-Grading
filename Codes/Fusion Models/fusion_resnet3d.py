import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from monai.networks.nets import resnet18

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    balanced_accuracy_score, roc_curve
)

from tqdm import tqdm
from torch.amp import autocast, GradScaler

TSV_PATH = "/Folder Directory/UTSW_Glioma_Metadata.tsv"
IMAGE_TENSOR_PATH = "preprocessed_96_2"
BEST_MODEL_PATH = "lfr21.pth"
TDA_TRAIN_PATH = "X_train_fs2.npy"
TDA_VAL_PATH   = "X_val_fs2.npy"
TDA_TEST_PATH  = "X_test_fs2.npy"

BATCH_SIZE = 2
EPOCHS = 50
LR = 0.0001
RANDOM_STATE = 42
patience1 = 3
patience2 = 12
TDA_DIM = 540
IMG_EMBED_DIM = 128
TDA_EMBED_DIM = 128
FUSION_HIDDEN_DIM = 128
NUM_CLASSES = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.benchmark = True

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# ============================================================
# LOAD PREPROCESSED .NPY FILES
# ============================================================

IMAGE_FOLDER = "preprocessed_96_2"

image_dict = {}

npy_files = sorted([
    f for f in os.listdir(IMAGE_FOLDER)
    if f.endswith(".npy")
])

print("Total npy files:", len(npy_files))

for file_name in tqdm(npy_files):

    patient_id = file_name.replace(".npy", "")

    path = os.path.join(IMAGE_FOLDER, file_name)

    arr = np.load(path).astype(np.float32)

    if arr.shape[-1] == 4:
        arr = np.transpose(arr, (3,0,1,2))

    image_dict[patient_id] = arr

print("Loaded patients:", len(image_dict))


# ============================================================
# LOAD METADATA AND SPLIT
# ============================================================

df = pd.read_csv(TSV_PATH, sep="\t")
df["tensor_index"] = np.arange(len(df))
df = df[df["Tumor Grade"].notna()]
df = df[df["Tumor Grade"].isin([2,3,4])]

df["label"] = df["Tumor Grade"].apply(lambda x: 1 if x == 4 else 0)

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


# ============================================================
# LOAD SELECTED TDA FEATURES
# ============================================================

X_train_tda = np.load(TDA_TRAIN_PATH).astype(np.float32)
X_val_tda   = np.load(TDA_VAL_PATH).astype(np.float32)
X_test_tda  = np.load(TDA_TEST_PATH).astype(np.float32)

print("TDA train shape:", X_train_tda.shape)
print("TDA val shape  :", X_val_tda.shape)
print("TDA test shape :", X_test_tda.shape)

scaler = StandardScaler()

X_train_tda = scaler.fit_transform(X_train_tda).astype(np.float32)
X_val_tda   = scaler.transform(X_val_tda).astype(np.float32)
X_test_tda  = scaler.transform(X_test_tda).astype(np.float32)


# ============================================================
# DATASET
# ============================================================

class FusionDataset(Dataset):

    def __init__(self, dataframe, image_dict, tda_features):

        self.df = dataframe.reset_index(drop=True)

        self.image_dict = image_dict

        self.tda_features = tda_features

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        pid = row["Subject ID"]

        label = int(row["label"])

        image = self.image_dict[pid]

        tda = self.tda_features[idx]

        image = torch.tensor(
            image,
            dtype=torch.float32
        )

        tda = torch.tensor(
            tda,
            dtype=torch.float32
        )

        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return image, tda, label


train_dataset = FusionDataset(train_df, image_dict, X_train_tda)
val_dataset   = FusionDataset(val_df, image_dict, X_val_tda)
test_dataset  = FusionDataset(test_df, image_dict, X_test_tda)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    drop_last=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
)


# ============================================================
# LATE FUSION MODEL
# ============================================================

class ResNetTDALateFusion(nn.Module):

    def __init__(
        self,
        image_embed_dim=IMG_EMBED_DIM,
        tda_embed_dim=TDA_EMBED_DIM,
        tda_dim=TDA_DIM,
        fusion_hidden_dim=FUSION_HIDDEN_DIM,
        num_classes=2
    ):

        super().__init__()

        self.image_backbone = resnet18(
            spatial_dims=3,
            n_input_channels=4,
            num_classes=image_embed_dim
        )


        self.tda_branch = nn.Sequential(
            nn.Linear(tda_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            #nn.Linear(64, 32),
            #nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(image_embed_dim + tda_embed_dim, fusion_hidden_dim),
            nn.BatchNorm1d(fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(fusion_hidden_dim, num_classes)
        )

    def forward(self, image, tda):

        image_features = self.image_backbone(image)

        tda_features = self.tda_branch(tda)

        fused = torch.cat(
            [image_features, tda_features],
            dim=1
        )

        logits = self.classifier(fused)

        return logits


model = ResNetTDALateFusion(
    image_embed_dim=IMG_EMBED_DIM,
    tda_dim=TDA_DIM,
    tda_embed_dim=TDA_EMBED_DIM,
    fusion_hidden_dim=FUSION_HIDDEN_DIM,
    num_classes=NUM_CLASSES
).to(DEVICE)


# ============================================================
# LOSS, OPTIMIZER, SCHEDULER
# ============================================================

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

scaler_amp = GradScaler(enabled=torch.cuda.is_available())

# ============================================================
# EVALUATION
# ============================================================

def evaluate(model, loader, threshold=0.5):
    model.eval()

    all_probs, all_preds, all_labels = [], [], []

    with torch.no_grad():
        for image, tda, y in loader:

            image = image.to(DEVICE, non_blocking=True)
            tda = tda.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            with autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                logits = model(image, tda)

            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = (probs > threshold).long()

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()

    return {
        "auc": roc_auc_score(all_labels, all_probs),
        "acc": accuracy_score(all_labels, all_preds),
        "bal_acc": balanced_accuracy_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0,
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
    }

def get_probs_labels(model, loader):
    model.eval()

    probs_all, labels_all = [], []

    with torch.no_grad():
        for image, tda, y in loader:

            image = image.to(DEVICE, non_blocking=True)
            tda = tda.to(DEVICE, non_blocking=True)

            with autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                logits = model(image, tda)

            probs = torch.softmax(logits, dim=1)[:, 1]

            probs_all.extend(probs.cpu().numpy())
            labels_all.extend(y.numpy())

    return np.array(probs_all), np.array(labels_all)


# ============================================================
# TRAINING LOOP
# ============================================================

best_auc = 0
patience = patience2
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

        with autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            logits = model(image, tda)
            loss = criterion(logits, y)

        scaler_amp.scale(loss).backward()
        scaler_amp.step(optimizer)
        scaler_amp.update()

        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)

    val_metrics = evaluate(
        model,
        val_loader,
        threshold=0.5
    )

    scheduler.step(val_metrics["auc"])

    current_lr = optimizer.param_groups[0]["lr"]

    print("\n====================================")
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print("Train Loss :", round(train_loss, 4))
    print("Val AUC    :", round(val_metrics["auc"], 4))
    print("Val Acc    :", round(val_metrics["acc"], 4))
    print("Val F1     :", round(val_metrics["f1"], 4))
    print("Val Sens   :", round(val_metrics["sensitivity"], 4))
    print("Val Spec   :", round(val_metrics["specificity"], 4))
    print("LR         :", current_lr)
    print("====================================")

    if val_metrics["auc"] > best_auc + min_delta:

        best_auc = val_metrics["auc"]
        epochs_without_improvement = 0

        torch.save(
            model.state_dict(),
            BEST_MODEL_PATH
        )

        print("Best Resnet3D model saved.")

    else:

        epochs_without_improvement += 1

        print(
            f"No improvement for "
            f"{epochs_without_improvement} epoch(s)."
        )

    if epochs_without_improvement >= patience:

        print("\nEarly stopping triggered.")
        break

# ============================================================
# TEST EVALUATION
# ============================================================

model.load_state_dict(
    torch.load(
        BEST_MODEL_PATH,
        map_location=DEVICE
    )
)

# ------------------------------------------------------------
# STEP 1: GET VALIDATION PROBABILITIES
# ------------------------------------------------------------

val_probs, val_labels = get_probs_labels(
    model,
    val_loader
)

# ------------------------------------------------------------
# STEP 2: FIND BEST THRESHOLD FROM VALIDATION SET
# ------------------------------------------------------------

fpr, tpr, thresholds = roc_curve(
    val_labels,
    val_probs
)

j_scores = tpr - fpr

best_idx = np.argmax(j_scores)

best_threshold = thresholds[best_idx]

print("\nBest validation threshold:", round(best_threshold, 4))

# ------------------------------------------------------------
# STEP 3: APPLY SAME THRESHOLD TO TEST SET
# ------------------------------------------------------------

test_metrics = evaluate(
    model,
    test_loader,
    threshold=best_threshold
)

# ------------------------------------------------------------
# STEP 4: PRINT TEST PROBABILITY DISTRIBUTION
# ------------------------------------------------------------

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

# ------------------------------------------------------------
# STEP 5: FINAL TEST RESULTS
# ------------------------------------------------------------

print("\n========== FINAL TEST RESULTS ==========")

# Header row
print("\t".join(test_metrics.keys()))

# Value row
print("\t".join(f"{v:.5f}" for v in test_metrics.values()))

print("========================================")
