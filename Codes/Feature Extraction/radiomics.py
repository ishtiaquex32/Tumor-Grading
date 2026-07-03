import os
import glob
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torch.nn as nn
from tqdm import tqdm
from radiomics import featureextractor
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
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

DATASET_DIR = "/Folder Directory/UCSF-PDGM-v5"
CSV_PATH = "/Folder Directory/UCSF-PDGM-metadata_v5.csv"

CHANNELS = ["T1", "T1c", "T2", "FLAIR"]

RANDOM_STATE = 42

MAX_SELECTED_FEATURES = 100

USE_GPU_XGBOOST = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
 
# ============================================================
# HELPERS
# ============================================================

def get_patient_folder(dataset_dir, patient_id):
    num = patient_id.split("-")[-1]
    num = num.zfill(4)

    folder = os.path.join(
        dataset_dir,
        f"UCSF-PDGM-{num}_nifti"
    )

    if os.path.exists(folder):
        return folder

    return None


def find_modality_file(folder, modality):
    files = glob.glob(os.path.join(folder, f"*_{modality}.nii.gz"))

    if len(files) == 0:
        return None

    return files[0]


def find_tumor_mask(folder):

    files = glob.glob(
        os.path.join(
            folder,
            "*_tumor_segmentation.nii.gz"
        )
    )

    if len(files) == 0:
        return None

    return files[0]


# ============================================================
# RADIOMICS EXTRACTOR
# ============================================================
params = {
    "binWidth": 25,
    "resampledPixelSpacing": None,
    "interpolator": sitk.sitkBSpline,
    "normalize": True,
    "normalizeScale": 100,
    "removeOutliers": 3,
   # "force2D": True,
}

extractor = featureextractor.RadiomicsFeatureExtractor(**params)
extractor.disableAllFeatures()
extractor.enableFeatureClassByName("firstorder")
extractor.enableFeatureClassByName("glcm")
extractor.disableAllImageTypes()
extractor.enableImageTypeByName("Original")

print("Active Image Types:", list(extractor.enabledImagetypes.keys()))
print("Active Feature Classes:", list(extractor.enabledFeatures.keys()))


# ============================================================
# EXTRACT FEATURES FOR ONE PATIENT
# ============================================================

def extract_patient_radiomics(folder, patient_id):
    mask_path = find_tumor_mask(folder)

    if mask_path is None:
        print("Missing mask:", patient_id)
        return None

    all_features = {}

    for ch in CHANNELS:
        image_path = find_modality_file(folder, ch)

        if image_path is None:
            print("Missing channel:", patient_id, ch)
            return None

        try:
            result = extractor.execute(
                image_path,
                mask_path,
                label=2
            )

            for key, value in result.items():

                if key.startswith("diagnostics"):
                    continue

                try:
                    value = float(value)
                except:
                    continue

                feature_name = f"{ch}_{key}"
                all_features[feature_name] = value

        except Exception as e:
            print("Radiomics error:", patient_id, ch)
            print(e)
            return None

    return all_features


# ============================================================
# LOAD METADATA
# ============================================================

df = pd.read_csv(CSV_PATH)

df = df[df["WHO CNS Grade"].notna()]
df = df[df["WHO CNS Grade"].isin([2, 3, 4])]

df["label"] = df["WHO CNS Grade"].apply(
    lambda x: 1 if x == 4 else 0
)

print("Total usable patients:", len(df))
print(df["label"].value_counts())


# ============================================================
# SPLIT FIRST: 70 / 10 / 20
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
# FEATURE EXTRACTION
# ============================================================

def build_radiomics_dataframe(split_df, split_name):
    rows = []

    for _, row in tqdm(split_df.iterrows(), total=len(split_df)):
        pid = row["ID"]

        folder = get_patient_folder(DATASET_DIR, pid)

        if folder is None:
            print("Missing folder:", pid)
            continue

        feats = extract_patient_radiomics(folder, pid)

        if feats is None:
            continue

        feats["ID"] = pid
        feats["label"] = row["label"]

        rows.append(feats)

    out_df = pd.DataFrame(rows)

    print(f"{split_name} radiomics shape:", out_df.shape)

    return out_df


train_rad = build_radiomics_dataframe(train_df, "train")
val_rad = build_radiomics_dataframe(val_df, "val")
test_rad = build_radiomics_dataframe(test_df, "test")

# ============================================================
# ALIGN COMMON FEATURES
# ============================================================

feature_cols = [
    c for c in train_rad.columns
    if c not in ["ID", "label"]
]

feature_cols = [
    c for c in feature_cols
    if c in val_rad.columns and c in test_rad.columns
]

X_train = train_rad[feature_cols].values
y_train = train_rad["label"].values

X_val = val_rad[feature_cols].values
y_val = val_rad["label"].values

X_test = test_rad[feature_cols].values
y_test = test_rad["label"].values

print("Initial radiomics feature count:", X_train.shape[1])


# ============================================================
# IMPUTE + SCALE
# ============================================================

imputer = SimpleImputer(strategy="median")

X_train = imputer.fit_transform(X_train)
X_val = imputer.transform(X_val)
X_test = imputer.transform(X_test)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


# Save processed full features
np.save("X_train_radiomics_full.npy", X_train)
np.save("X_val_radiomics_full.npy", X_val)
np.save("X_test_radiomics_full.npy", X_test)

np.save("y_train_radiomics.npy", y_train)
np.save("y_val_radiomics.npy", y_val)
np.save("y_test_radiomics.npy", y_test)

pd.Series(feature_cols).to_csv(
    "radiomics_feature_names_full.csv",
    index=False
)


# ============================================================
# XGBOOST BASELINE
# ============================================================

xgb_params = dict(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.0001,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=RANDOM_STATE
)

if USE_GPU_XGBOOST:
    xgb_params.update({
        "tree_method": "hist",
        "device": "cuda"
    })

model = XGBClassifier(**xgb_params)

model.fit(X_train, y_train)

# ============================================================
# VALIDATION PERFORMANCE
# ============================================================

val_prob = model.predict_proba(X_val)[:, 1]
val_pred = (val_prob > 0.5).astype(int)

val_auc = roc_auc_score(y_val, val_prob)
val_acc = accuracy_score(y_val, val_pred)
val_f1 = f1_score(y_val, val_pred)

print("\n========== BASELINE VALIDATION ==========")
print("Val AUC :", round(val_auc, 4))
print("Val Acc :", round(val_acc, 4))
print("Val F1  :", round(val_f1, 4))
print("=========================================")


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importances = model.feature_importances_

print("\nFeature importance summary:")
print("Min importance   :", np.min(importances))
print("Median importance:", np.median(importances))
print("Max importance   :", np.max(importances))
print("Nonzero features :", np.sum(importances > 0), "/", len(importances))


# ============================================================
# FEATURE SELECTION
# ============================================================

selector = SelectFromModel(
    model,
    prefit=True,
    max_features=MAX_SELECTED_FEATURES,
    threshold=-np.inf
)

X_train_fs = selector.transform(X_train)
X_val_fs = selector.transform(X_val)
X_test_fs = selector.transform(X_test)

selected_idx = selector.get_support(indices=True)
selected_feature_names = [feature_cols[i] for i in selected_idx]

print("Selected features:", X_train_fs.shape[1])


# Save selected features
#np.save("X_train_radiomics_selected.npy", X_train_fs)
#np.save("X_val_radiomics_selected.npy", X_val_fs)
#np.save("X_test_radiomics_selected.npy", X_test_fs)

pd.Series(selected_feature_names).to_csv(
    "radiomics_selected_feature_names.csv",
    index=False
)


# ============================================================
# RETRAIN XGBOOST WITH SELECTED FEATURES
# ============================================================

model_fs = XGBClassifier(**xgb_params)

model_fs.fit(X_train_fs, y_train)


# ============================================================
# TEST EVALUATION
# ============================================================

from sklearn.metrics import roc_curve

val_prob_fs = model_fs.predict_proba(X_val_fs)[:, 1]

fpr, tpr, thresholds = roc_curve(y_val, val_prob_fs)
j_scores = tpr - fpr
best_idx = np.argmax(j_scores)
best_threshold = thresholds[best_idx]

print("Best validation threshold:", best_threshold)

test_prob = model_fs.predict_proba(X_test_fs)[:, 1]
#test_pred = (test_prob > 0.5).astype(int)
test_pred = (test_prob > best_threshold).astype(int)

auc = roc_auc_score(y_test, test_prob)
acc = accuracy_score(y_test, test_pred)
f1 = f1_score(y_test, test_pred)
prec = precision_score(y_test, test_pred)
rec = recall_score(y_test, test_pred)

tn, fp, fn, tp = confusion_matrix(y_test, test_pred).ravel()

sens = tp / (tp + fn)
spec = tn / (tn + fp)

print("\n========== TEST RESULTS: SELECTED RADIOMICS ==========")
print("AUC         :", round(auc, 4))
print("Accuracy    :", round(acc, 4))
print("F1 Score    :", round(f1, 4))
print("Sensitivity :", round(sens, 4))
print("Specificity :", round(spec, 4))
print("Precision   :", round(prec, 4))
print("Recall      :", round(rec, 4))
print("=====================================================")

selected_idx = selector.get_support(indices=True)

channel_count = {
    "T1":0,
    "T1c":0,
    "T2":0,
    "FLAIR":0
}

for idx in selected_idx:
    if idx < 150:
        channel_count["T1"] += 1
    elif idx < 300:
        channel_count["T1c"] += 1
    elif idx < 450:
        channel_count["T2"] += 1
    else:
        channel_count["FLAIR"] += 1

print("\nSelected Features per Channel:")
for k,v in channel_count.items():
    print(k, ":", v)
