import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from monai.networks.nets import resnet18
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    balanced_accuracy_score, roc_curve
)

from tqdm import tqdm
from torch.amp import autocast, GradScaler


# ============================================================
# CONFIG
# ============================================================

TSV_PATH = "/Folder Name/UTSW_Glioma_Metadata.tsv"
IMAGE_FOLDER = "preprocessed_96_2"

BATCH_SIZE = 2
EPOCHS = 50
LR = 0.0001
RANDOM_STATE = 42
patience1 = 3
patience2 = 12

EMBED_DIM = 128
NUM_CLASSES = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.benchmark = True

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# LOAD METADATA
# ============================================================

df = pd.read_csv(TSV_PATH, sep="\t")

df = df[df["Tumor Grade"].notna()]
df = df[df["Tumor Grade"].isin([2,3,4])]

df["label"] = df["Tumor Grade"].apply(lambda x: 1 if x == 4 else 0)

print("Total usable patients:", len(df))
print(df["label"].value_counts())


# ============================================================
# TRAIN / VAL / TEST SPLIT
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

class SharedBackboneMRIDataset(Dataset):

    def __init__(self, dataframe, image_folder):
        self.df = dataframe.reset_index(drop=True)
        self.image_folder = image_folder

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        pid = row["Subject ID"]
        label = int(row["label"])

        path = os.path.join(self.image_folder, f"{pid}.npy")

        image = np.load(path).astype(np.float32)

        # Expected shape either:
        # (4, 96, 96, 96) or (96, 96, 96, 4)
        if image.shape[-1] == 4:
            image = np.transpose(image, (3, 0, 1, 2))

        image = torch.from_numpy(image)

        label = torch.tensor(label, dtype=torch.long)

        return image, label


train_dataset = SharedBackboneMRIDataset(train_df, IMAGE_FOLDER)
val_dataset   = SharedBackboneMRIDataset(val_df, IMAGE_FOLDER)
test_dataset  = SharedBackboneMRIDataset(test_df, IMAGE_FOLDER)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=True,
    drop_last=True
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


# ============================================================
# MODEL
# ============================================================

model = resnet18(
    spatial_dims=3,
    n_input_channels=4,
    num_classes=2
)

model = model.to(DEVICE)


# ============================================================
# LOSS / OPTIMIZER / SCHEDULER
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

scaler = GradScaler(enabled=torch.cuda.is_available())

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
            "best_lf2_resnet18_tda.pth"
        )

        print("Best X3D model saved.")

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
        "best_lf2_resnet18_tda.pth",
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

for k, v in test_metrics.items():
    print(f"{k}: {round(v, 4)}")

print("========================================")