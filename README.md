# Myocardial Infarction Detection Using ECG Signals: Experimental ML/DL Pipelines

## Abstract

This repository contains experimental machine learning and deep learning pipelines for detecting Myocardial Infarction (MI) from 12-lead Electrocardiogram (ECG) signals. The work explores multiple architectural paradigms including convolutional neural networks, attention mechanisms, transformer-based models, spectrogram representations, and ensemble strategies. Due to the research-oriented nature of this work, the repository includes both successful approaches and abandoned experiments, providing transparency into the iterative development process typical of thesis research. All experiments use the publicly available PTB-XL dataset with binary classification (MI vs Normal). This codebase is intended for academic research and benchmarking purposes; it is not validated for clinical use.

---

## Repository Structure

```
better-detection/
├── approaches/                    # All experimental pipelines
│   ├── LGS-NET/                   # [EXPERIMENTAL] Local-Global-Spectral Network
│   │   ├── code.py
│   │   └── results/report.txt
│   ├── cct/                       # [EXPERIMENTAL] Compact Convolutional Transformer
│   │   ├── code.py
│   │   └── results/report.txt
│   ├── ecg_base/                  # [STABLE] Baseline ResNet + LogisticRegression
│   │   ├── code.py
│   │   └── results/report.txt
│   ├── ecg_ome/                   # [EXPERIMENTAL] 5-Paradigm Orthogonal Model Ensemble
│   │   ├── code.py
│   │   └── results/report.txt
│   ├── ect_fast/                  # [STABLE] Fast Multi-Model Ensemble
│   │   ├── code.py
│   │   └── results/report.txt
│   ├── ect_n_titan/               # [EXPERIMENTAL] Combined ECT + TITAN (2 versions)
│   │   ├── code.v1.py
│   │   ├── code.v2.py
│   │   └── results/report.v1.txt
│   ├── ensemble/                  # [STABLE] 3-Model Voting Classifier
│   │   ├── code.py
│   │   └── results/report.txt
│   ├── resnet/                    # [STABLE] ResNet-1D + Multi-Head Attention
│   │   ├── code.py
│   │   └── results/
│   │       ├── report.txt
│   │       └── training_curves.png
│   ├── resnet-plus/               # [EXPERIMENTAL] Enhanced ResNet Variant
│   │   ├── code.py
│   │   └── results/report.txt
│   ├── sota/                      # [EXPERIMENTAL] InceptionTime (SOTA attempt)
│   │   ├── code.py
│   │   └── results/report.txt
│   └── titan/                     # [EXPERIMENTAL] 5-Method Deep Learning Ensemble
│       ├── code.py
│       └── results/titan_report.txt
├── ptbdb/                         # PTB-XL dataset subset (CSV files)
├── requirements.txt               # Python dependencies
├── .gitignore
└── README.md
```

### Approach Status Legend

| Status | Description |
|--------|-------------|
| **STABLE** | Validated approach with reproducible results |
| **EXPERIMENTAL** | Promising approach, results may vary |

---

## Methodologies Implemented

### Signal Preprocessing

All pipelines implement:
- **Z-score normalization**: Per-lead standardization `(signal - mean) / (std + 1e-8)`
- **Resampling**: 100 Hz (low-res) or 250 Hz (high-res from 500 Hz)
- **Bandpass filtering** (in `sota/`): 0.5-45 Hz Butterworth filter
- **Padding/Cropping**: Fixed 1000 samples (10 seconds at 100 Hz)

### Deep Learning Architectures

| Approach | Folder | Description | Status |
|----------|--------|-------------|--------|
| ResNet-1D + Multi-Head Attention | `resnet/` | Residual blocks with 4-head self-attention | Stable |
| Lead Attention + LogReg | `ecg_base/` | Channel attention for lead weighting | Stable |
| 3-Model Voting Ensemble | `ensemble/` | ResNet-Plus with label smoothing | Stable |
| Multi-Model Ensemble | `ect_fast/` | Transformer + HybridCNN + ResNet + XGBoost | Stable |
| LGS-NET | `LGS-NET/` | Dual-stream (time + frequency), local window attention | Experimental |
| CCT Transformer | `cct/` | Conv tokenizer + Transformer on spectrograms | Experimental |
| InceptionTime | `sota/` | Multi-scale inception modules at 250 Hz | Experimental |
| TITAN Pipeline | `titan/` | InceptionV3 + ResNet-SE + DenseNet + BiLSTM + XGBoost | Experimental |
| 5-Paradigm OME | `ecg_ome/` | XGBoost + Inception1D + SpatialAttn + SpectralNet + ROCKET | Experimental |
| ECT + TITAN | `ect_n_titan/` | Combined approach (2 versions) | Experimental |

### Classical ML Models

| Model | Usage |
|-------|-------|
| XGBoost | Statistical features / deep embeddings |
| Logistic Regression | Head classifier on CNN features |
| Ridge Classifier | ROCKET-lite features |

---

## Datasets

### PTB-XL Dataset

| Property | Value |
|----------|-------|
| Source | [PhysioNet PTB-XL](https://physionet.org/content/ptb-xl/1.0.1/) via KaggleHub |
| Total Records | 21,837 ECG recordings |
| Used Subset | MI vs Normal (balanced) |
| Sampling Rate | 100 Hz (LR) / 500 Hz (HR) |
| Duration | 10 seconds per recording |
| Leads | 12-lead standard ECG |

**Note**: Dataset is downloaded automatically via `kagglehub.dataset_download()` at runtime. The `ptbdb/` folder contains a preprocessed subset.

---

## Results Summary

| Approach | Folder | Accuracy | AUC | F1 |
|----------|--------|----------|-----|-----|
| **3-Model Ensemble** | `ensemble/` | 0.924 | **0.978** | 0.924 |
| LGS-NET | `LGS-NET/` | 0.920 | 0.975 | - |
| TITAN Ensemble | `titan/` | 0.910 | 0.975 | - |
| InceptionTime | `sota/` | 0.906 | 0.965 | 0.905 |
| ResNet + Attention | `resnet/` | 0.884 | 0.968 | 0.872 |
| 5-Paradigm OME | `ecg_ome/` | 0.877 | 0.943 | - |
| CCT Transformer | `cct/` | 0.858 | 0.934 | 0.860 |

**Best reproducible result**: 3-Model Ensemble (92.4% accuracy, 0.978 AUC)

---

## Reproducibility

### Requirements

| Package | Version |
|---------|---------|
| Python | 3.10+ |
| TensorFlow | 2.15+ |
| NumPy | 2.3.3 |
| Scikit-learn | 1.7.2 |

### Random Seeds

All scripts use `random_state=42` and `tf.random.set_seed(42)` for reproducibility.

### Hardware

- GPU: Mixed precision (FP16) enabled automatically
- Memory: ~8-16 GB RAM recommended

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run any approach
cd approaches/<approach_name>
python code.py
```

Results are saved to `./results/` within each approach folder.

**Note**: Scripts are not fully automated; hyperparameters are hardcoded.

---

## Known Issues and Limitations

### Data Leakage Risks
- Some scripts apply augmentation before train/val split
- No patient-level splitting enforced

### Dataset Limitations
- Single-center study (PTB-XL)
- Balanced via downsampling (selection bias possible)

### Clinical Validity
**NOT validated for clinical use.** Models have not been tested on external datasets.

---

## Research Status

| Property | Status |
|----------|--------|
| Purpose | M.Tech Thesis Research |
| Development | Experimental / Active |
| Clinical Validation | Not performed |

---

## Citation

```bibtex
@misc{mi-detection-ecg-2025,
  author = {Kholwal, Kanak},
  title  = {MI Detection Using ECG: Experimental ML/DL Pipelines},
  year   = {2025},
  note   = {M.Tech Thesis Work}
}
```

---

## Disclaimer

**MEDICAL**: This software is for research only. NOT validated for clinical use.

**ACADEMIC**: Results may not be reproducible across different configurations.
