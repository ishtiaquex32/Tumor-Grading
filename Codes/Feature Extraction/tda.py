import os
import re
import glob
import numpy as np
import pandas as pd
import nibabel as nib
import gudhi as gd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from monai.networks.nets import resnet18

from scipy.ndimage import zoom
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)

from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier

# ============================================================
# CONFIG
# ============================================================

DATASET_DIR = ('/Folder Directory/UTSW-Glioma')
TSV_PATH = "/Folder Directory/UTSW_Glioma_Metadata.tsv"

df = pd.read_csv(TSV_PATH, sep="\t")
df = df[df["Tumor Grade"].notna()]

df["Tumor Grade"] = pd.to_numeric(
    df["Tumor Grade"],
    errors="coerce"
)
df = df[df["Tumor Grade"].notna()]

df = df[df["Tumor Grade"].isin([2,3,4])]

df["label"] = df["Tumor Grade"].apply(lambda x: 1 if x == 4 else 0)

print("Total usable patients:", len(df))

TARGET_SHAPE = (96, 96, 96)

CHANNELS = ["t1", "t1ce", "t2", "fl"]

BATCH_SIZE = 2
EPOCHS = 50
LR = 0.0001
MF = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MD = 6
alpha = 0.04
N_THRESHOLDS = 50
RANDOM_STATE = 42

# ============================================================
# HELPERS
# ============================================================

def normalize_volume(x):
    x = x.astype(np.float32)
    if np.std(x) == 0:
        return np.zeros_like(x)
    x = (x - np.mean(x)) / np.std(x)
    return x


def resize_volume(vol, target_shape=TARGET_SHAPE):
    factors = (
        target_shape[0] / vol.shape[0],
        target_shape[1] / vol.shape[1],
        target_shape[2] / vol.shape[2],
    )
    return zoom(vol, factors, order=1)


def load_nifti(path):
    return nib.load(path).get_fdata()

def intensity_transform(vol):

    vol = vol - vol.min()

    vol = np.log1p(vol)

    vol = np.clip(vol, 0, 3)

    return vol


def get_patient_folder(dataset_dir, patient_id):

    folder = os.path.join(
        dataset_dir,
        str(patient_id)
    )

    if os.path.exists(folder):
        return folder

    return None

def find_modality_file(folder, modality):

    modality_files = {
        "t1": "brain_t1_ants.nii.gz",
        "t1ce": "brain_t1ce_ants.nii.gz",
        "t2": "brain_t2_ants.nii.gz",
        "fl": "brain_fl_ants.nii.gz"
    }

    path = os.path.join(folder, modality_files[modality])

    if os.path.exists(path):
        return path

    return None

# ============================================================
# BETTI FEATURES
# ============================================================

def compute_persistence(volume):
    cc = gd.CubicalComplex(top_dimensional_cells=volume)
    cc.compute_persistence()

    return cc.persistence()

def betti_curves_from_persistence(persistence, thresholds, min_lifetime=0.02):
    b0 = np.zeros(len(thresholds))
    b1 = np.zeros(len(thresholds))
    b2 = np.zeros(len(thresholds))

    for dim, (birth, death) in persistence:

        if death == float('inf'):
            death = thresholds[-1]

        lifetime = death - birth

        if lifetime < min_lifetime:
            continue

        alive = (thresholds >= birth) & (thresholds < death)

        if dim == 0:
            b0[alive] += 1
        elif dim == 1:
            b1[alive] += 1
        elif dim == 2:
            b2[alive] += 1

    return b0, b1, b2

def normalize_curve(curve):

    if np.max(curve) == 0:
        return curve
    return curve / np.max(curve)


def betti_features_3d(volume, n_thresholds=50):

    vmin, vmax = np.min(volume), np.max(volume)
    thresholds = np.linspace(vmin, vmax, n_thresholds)

    persistence = compute_persistence(volume)

    b0, b1, b2 = betti_curves_from_persistence(
        persistence,
        thresholds,
        min_lifetime=0.02
    )

    b0 = normalize_curve(b0)
    b1 = normalize_curve(b1)
    b2 = normalize_curve(b2)

    return np.concatenate([b0, b1, b2]).astype(np.float32)

def extract_patient_features_from_npy(pid):

    npy_path = os.path.join(
        PREPROCESSED_DIR,
        f"{pid}.npy"
    )

    if not os.path.exists(npy_path):
        print("Missing preprocessed file:", pid)
        return None

    patient_tensor = np.load(
        npy_path
    ).astype(np.float32)

    all_feats = []

    for c in range(patient_tensor.shape[0]):

        vol = patient_tensor[c]

        feats = betti_features_3d(
            vol,
            n_thresholds=50
        )

        all_feats.append(feats)

    return np.concatenate(all_feats).astype(np.float32)

# ============================================================
# PREPROCESS AND SAVE AS .NPY
# ============================================================

PREPROCESSED_DIR = "./preprocessed_96_2"

os.makedirs(PREPROCESSED_DIR, exist_ok=True)


def preprocess_and_save(dataframe):

    for _, row in tqdm(dataframe.iterrows(), total=len(dataframe)):

        pid = row["Subject ID"]

        save_path = os.path.join(
            PREPROCESSED_DIR,
            f"{pid}.npy"
        )

        if os.path.exists(save_path):
            continue

        folder = os.path.join(DATASET_DIR, pid)

        if not os.path.exists(folder):
            print(f"Missing folder for {pid}")
            continue

        volumes = []

        try:

            for ch in CHANNELS:

                path = find_modality_file(folder, ch)

                if path is None:
                    raise ValueError(f"Missing {ch} for {pid}")

                vol = load_nifti(path)

                vol = normalize_volume(vol)
                vol = intensity_transform(vol)
                vol = resize_volume(vol, TARGET_SHAPE)

                volumes.append(vol)

            volumes = np.stack(volumes, axis=0).astype(np.float32)

            np.save(save_path, volumes)

        except Exception as e:

            print(f"Error processing {pid}")
            print(e)

preprocess_and_save(df)

# ============================================================
# FEATURE EXTRACTION FROM PREPROCESSED .NPY FILES
# ============================================================

X = []
y = []
patient_names = []

for _, row in tqdm(df.iterrows(), total=len(df)):

    pid = row["Subject ID"]

    feats = extract_patient_features_from_npy(pid)

    if feats is None:
        continue

    X.append(feats)
    y.append(row["label"])
    patient_names.append(pid)

X = np.array(X, dtype=np.float32)
y = np.array(y)

print("Feature shape:", X.shape)
print("Labels shape :", y.shape)

#%%
from sklearn.model_selection import train_test_split

# Step 1: Split out 20% test set
X_temp, X_test, y_temp, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Step 2: Split remaining 80% into 70% train and 10% val
# 10% out of total means 10/80 = 0.125 of remaining
X_train, X_val, y_train, y_val = train_test_split(
    X_temp,
    y_temp,
    test_size=0.125,   # 0.125 * 0.80 = 0.10 total
    random_state=42,
    stratify=y_temp
)

print("Train:", X_train.shape)
print("Val  :", X_val.shape)
print("Test :", X_test.shape)

# ============================================================
# CLASS IMBALANCE SETUP FOR XGBOOST
# ============================================================

neg_count = np.sum(y_train == 0)
pos_count = np.sum(y_train == 1)

scale_pos_weight = (neg_count / pos_count) ** alpha

print("Negative count:", neg_count)
print("Positive count:", pos_count)
print("scale_pos_weight:", scale_pos_weight)

# ============================================================
# XGBOOST BASELINE
# ============================================================
model = XGBClassifier(
    n_estimators=300,
    max_depth=MD,
    learning_rate=LR,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    random_state=RANDOM_STATE,
    tree_method="hist",
    device="cuda",
    scale_pos_weight=scale_pos_weight
)

model.fit(X_train, y_train)

# ============================================================
# VALIDATION PERFORMANCE
# ============================================================
val_prob = model.predict_proba(X_val)[:,1]
val_pred = (val_prob > 0.5).astype(int)

val_auc = roc_auc_score(y_val, val_prob)
print("\nValidation AUC:", val_auc)

importances = model.feature_importances_

useful = np.sum(importances > 0)
print("Useful Features:", useful, "/", len(importances))

selector = SelectFromModel(model, prefit=True, max_features=MF, threshold=-np.inf)

X_train_fs = selector.transform(X_train)
X_val_fs   = selector.transform(X_val)
X_test_fs  = selector.transform(X_test)

print("Selected features:", X_train_fs.shape[1])

# ============================================================
# SAVE SELECTED FEATURES
# ============================================================

np.save("X_train_fs2.npy", X_train_fs)
np.save("y_train_fs2.npy", y_train)

np.save("X_val_fs2.npy", X_val_fs)
np.save("y_val_fs2.npy", y_val)

np.save("X_test_fs2.npy", X_test_fs)
np.save("y_test_fs2.npy", y_test)

print("\nSelected feature matrices saved.")

# ============================================================
# RETRAIN WITH SELECTED FEATURES
# ============================================================
model_fs = XGBClassifier(
    n_estimators=200,
    max_depth=MD,
    learning_rate=LR,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    random_state=RANDOM_STATE,
    tree_method="hist",
    device="cuda",
    scale_pos_weight=scale_pos_weight
)

model_fs.fit(X_train_fs, y_train)

from sklearn.metrics import roc_curve

val_prob_fs = model_fs.predict_proba(X_val_fs)[:, 1]

fpr, tpr, thresholds = roc_curve(y_val, val_prob_fs)
j_scores = tpr - fpr
best_idx = np.argmax(j_scores)
best_threshold = thresholds[best_idx]

print("Best validation threshold:", best_threshold)

test_prob = model_fs.predict_proba(X_test_fs)[:, 1]
test_pred = (test_prob > best_threshold).astype(int)
auc  = roc_auc_score(y_test, test_prob)
acc  = accuracy_score(y_test, test_pred)
f1   = f1_score(y_test, test_pred)
prec = precision_score(y_test, test_pred)
rec  = recall_score(y_test, test_pred)

tn, fp, fn, tp = confusion_matrix(y_test, test_pred).ravel()

sens = tp / (tp + fn)
spec = tn / (tn + fp)

# ============================================================
# RESULTS
# ============================================================
print("\n========== TEST RESULTS ===========")
print("AUC         :", round(auc,4))
print("Accuracy    :", round(acc,4))
print("F1 Score    :", round(f1,4))
print("Sensitivity :", round(sens,4))
print("Specificity :", round(spec,4))
print("==============================================")

# ============================================================
# CHANNEL-WISE FEATURE ANALYSIS
# ============================================================
selected_idx = selector.get_support(indices=True)

channel_count = {
    "t1":0,
    "t1ce":0,
    "t2":0,
    "fl":0
}

for idx in selected_idx:
    if idx < 150:
        channel_count["t1"] += 1
    elif idx < 300:
        channel_count["t1ce"] += 1
    elif idx < 450:
        channel_count["t2"] += 1
    else:
        channel_count["fl"] += 1

print("\nSelected Features per Channel:")
for k,v in channel_count.items():
    print(k, ":", v)
