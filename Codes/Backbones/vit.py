import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

from monai.networks.nets import UNETR

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    balanced_accuracy_score, roc_curve
)


# ============================================================
# CONFIG
# ============================================================

TSV_PATH = "/Folder Directory/UTSW_Glioma_Metadata.tsv"
IMAGE_FOLDER = "preprocessed_96_2"
BEST_MODEL_PATH = "vit24.pth"
BATCH_SIZE = 2
EPOCHS = 50
LR = 1e-4
RANDOM_STATE = 42

patience1 = 3
patience2 = 12

IMG_SIZE = (96, 96, 96)
IN_CHANNELS = 4
NUM_CLASSES = 2

HIDDEN_SIZE = 768
FEATURE_SIZE = 16
MLP_DIM = 3072
NUM_HEADS = 12

alpha = 0.1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# ID NORMALIZATION
# ============================================================

def normalize_file_id(pid):
    return str(pid)


# ============================================================
# LOAD METADATA
# ============================================================

df = pd.read_csv(TSV_PATH, sep="\t")

df = df[df["Tumor Grade"].notna()]
df = df[df["Tumor Grade"].isin([2,3,4])]

df["label"] = df["Tumor Grade"].apply(lambda x: 1 if x == 4 else 0)
df["file_id"] = df["Subject ID"].apply(normalize_file_id)

print("Total usable patients:", len(df))
print(df["label"].value_counts())


# ============================================================
# SPLIT 70 / 10 / 20
# ============================================================

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
# DATASET
# ============================================================

class UCSFViTDataset(Dataset):

    def __init__(self, dataframe, image_folder):
        self.df = dataframe.reset_index(drop=True)
        self.image_folder = image_folder

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        pid = row["file_id"]
        label = int(row["label"])

        path = os.path.join(self.image_folder, f"{pid}.npy")

        image = np.load(path).astype(np.float32)

        # Accept [4,96,96,96] or [96,96,96,4]
        if image.shape[-1] == 4:
            image = np.transpose(image, (3, 0, 1, 2))

        image = torch.from_numpy(image)
        label = torch.tensor(label, dtype=torch.long)

        return image, label


train_loader = DataLoader(
    UCSFViTDataset(train_df, IMAGE_FOLDER),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=True
)

val_loader = DataLoader(
    UCSFViTDataset(val_df, IMAGE_FOLDER),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=True
)

test_loader = DataLoader(
    UCSFViTDataset(test_df, IMAGE_FOLDER),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=True
)


# ============================================================
# 3D ViT-UNet / UNETR CLASSIFIER
# ============================================================

class ViTUNETClassifier(nn.Module):

    def __init__(
        self,
        img_size=(96, 96, 96),
        in_channels=4,
        num_classes=2,
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12
    ):
        super().__init__()

        self.unetr = UNETR(
            in_channels=in_channels,
            out_channels=2,
            img_size=img_size,
            feature_size=feature_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_heads=num_heads,
            proj_type="conv",
            norm_name="instance",
            res_block=True,
            dropout_rate=0.1,
            spatial_dims=3
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):

        # Use ViT encoder only.
        # MONAI UNETR ViT returns:
        # tokens: [B, num_patches, hidden_size]
        # hidden_states_out: list of intermediate states
        tokens, hidden_states_out = self.unetr.vit(x)

        # Global average pooling over patch tokens
        pooled = tokens.mean(dim=1)

        logits = self.classifier(pooled)

        return logits


model = ViTUNETClassifier(
    img_size=IMG_SIZE,
    in_channels=IN_CHANNELS,
    num_classes=NUM_CLASSES,
    feature_size=FEATURE_SIZE,
    hidden_size=HIDDEN_SIZE,
    mlp_dim=MLP_DIM,
    num_heads=NUM_HEADS
).to(DEVICE)


# ============================================================
# LOSS / OPTIMIZER / SCHEDULER
# ============================================================

class_counts = train_df["label"].value_counts().sort_index().values

class_weights = 1.0 / (class_counts ** alpha)
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


# ============================================================
# EVALUATION
# ============================================================

def evaluate(model, loader, threshold=0.5):

    model.eval()

    all_probs = []
    all_preds = []
    all_labels = []

    with torch.no_grad():

        for x, y in loader:

            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            with autocast(
                device_type="cuda",
                enabled=torch.cuda.is_available()
            ):
                logits = model(x)

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
    f1 = f1_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        "auc": auc,
        "acc": acc,
        "bal_acc": bal_acc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "sensitivity": sens,
        "specificity": spec
    }


def get_probs_labels(model, loader):

    model.eval()

    probs_all = []
    labels_all = []

    with torch.no_grad():

        for x, y in loader:

            x = x.to(DEVICE, non_blocking=True)

            with autocast(
                device_type="cuda",
                enabled=torch.cuda.is_available()
            ):
                logits = model(x)

            probs = torch.softmax(logits, dim=1)[:, 1]

            probs_all.extend(probs.cpu().numpy())
            labels_all.extend(y.numpy())

    return np.array(probs_all), np.array(labels_all)


# ============================================================
# TRAINING LOOP
# ============================================================

best_auc = 0
epochs_without_improvement = 0
min_delta = 0.001

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0

    for x, y in tqdm(train_loader):

        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        with autocast(
            device_type="cuda",
            enabled=torch.cuda.is_available()
        ):
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)

    val_metrics = evaluate(model, val_loader, threshold=0.5)

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

        print("Best model saved.")

    else:

        epochs_without_improvement += 1

        print(
            f"No improvement for "
            f"{epochs_without_improvement} epoch(s)."
        )

    if epochs_without_improvement >= patience2:

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
# OPTIONAL: SEE HOW VALIDATION BEHAVES
# ------------------------------------------------------------

for th in [0.5, 0.6, 0.7, 0.8]:

    preds = (val_probs > th).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        val_labels,
        preds
    ).ravel()

    sens = tp / (tp + fn)
    spec = tn / (tn + fp)

    print(
        f"Val Threshold={th:.1f} | "
        f"Sens={sens:.3f} | "
        f"Spec={spec:.3f}"
    )

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
