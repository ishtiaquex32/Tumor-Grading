# Topological Deep Learning for Brain Tumor Grading

Official anonymous implementation accompanying our IEEE BIBM submission.

---

## Overview

This repository contains the implementation of our framework for binary glioma grading from multi-parametric MRI by integrating Topological Data Analysis (TDA) with multiple deep learning backbones through late feature fusion.

The framework consists of:

- Topological Data Analysis (TDA) feature extraction
- Radiomics feature extraction
- Four deep learning backbones
- Four TDA-based late fusion models

---

## Repository Structure

```
.
├── Codes/
│   ├── Feature Extraction/
│   │   ├── tda.py
│   │   └── radiomics.py
│   │
│   ├── Backbones/
│   │   ├── resnet3d.py
│   │   ├── x3d.py
│   │   ├── vit.py
│   │   └── swin.py
│   │
│   └── Fusion Models/
│       ├── fusion_resnet3d.py
│       ├── fusion_x3d.py
│       ├── fusion_vit.py
│       └── fusion_swin.py
│
├── configs/
│   ├── ucsf.yaml
│   ├── utsw.yaml
│   ├── training.yaml
│   └── model.yaml
│
├── LICENSE
└── requirements.txt
```

---

## Implemented Methods

### Feature Extraction

- Topological Data Analysis (Cubical Persistent Homology)
- Radiomics (PyRadiomics)

### Deep Learning Backbones

- 3D ResNet18
- X3D
- UNETR (Vision Transformer)
- SwinUNETR

### Fusion Models

- TDA + ResNet3D
- TDA + X3D
- TDA + ViT/UNETR
- TDA + SwinUNETR

---

## Installation

Clone the repository

```bash
git clone <anonymous_repository_url>

cd repository
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Dataset

The repository is designed for multi-parametric MRI data consisting of four MRI sequences:

- T1
- T1-contrast
- T2
- FLAIR

The code has been developed and evaluated on two independent glioma datasets:

- UCSF-PDGM
- UTSW Glioma

Users should update the corresponding dataset paths in the configuration files.

---

## Configuration

All experiment settings are located in the `configs/` directory.

```
configs/

├── ucsf.yaml
├── utsw.yaml
├── training.yaml
└── model.yaml
```

These files define:

- dataset locations
- preprocessing settings
- model hyperparameters
- optimizer settings
- training configuration

---

## Running the Code

### Topological Features

```bash
python Codes/tda.py
```

---

### Radiomics Features

```bash
python Codes/radiomics.py
```

---

### Train Individual Deep Learning Models

ResNet3D

```bash
python Codes/resnet3d.py
```

X3D

```bash
python Codes/x3d.py
```

UNETR

```bash
python Codes/vit.py
```

SwinUNETR

```bash
python Codes/swin.py
```

---

### Train Fusion Models

ResNet3D + TDA

```bash
python Codes/fusion_resnet3d.py
```

X3D + TDA

```bash
python Codes/fusion_x3d.py
```

UNETR + TDA

```bash
python Codes/fusion_vit.py
```

SwinUNETR + TDA

```bash
python Codes/fusion_swin.py
```

---

## Notes

- The provided scripts are intended to reproduce the experiments described in the accompanying manuscript.
- Dataset paths should be updated before running the code.
- Model checkpoints and extracted features are generated during execution and are not included in this repository.

---

## License

This repository is provided solely for anonymous peer review.

See the `LICENSE` file for details.
