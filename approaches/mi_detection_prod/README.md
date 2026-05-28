# MI Detection (PTB-XL) — Production Pipeline

PyTorch pipeline for binary myocardial-infarction detection on PTB-XL with strict
patient-wise splits, no synthetic oversampling, and a TITAN-style ensemble
(SimpleCNN + Inception + ResNet + XGBoost on penultimate embeddings).

## Layout

| File | Purpose |
|---|---|
| [config.py](config.py) | Seeds, hyperparameters, paths, `set_determinism()` |
| [data.py](data.py) | PTB-XL load (100 Hz), MI/NORM label, z-score per record, `GroupShuffleSplit` |
| [models.py](models.py) | `SimpleCNN`, `InceptionNet`, `ResNet1D` with dropout-free `.penultimate()` |
| [train.py](train.py) | Training loop, BCEWithLogits + `pos_weight`, early stop on val AUC |
| [evaluate.py](evaluate.py) | Embeddings, XGBoost, weighted ensemble, metrics, report |
| [run.py](run.py) | End-to-end entrypoint |

## Hard guarantees

- **Patient-wise split** via `sklearn.model_selection.GroupShuffleSplit` — disjoint patient sets asserted at runtime.
- **No SMOTE, no oversampling** — class imbalance handled via `BCEWithLogitsLoss(pos_weight=...)` for CNNs and `scale_pos_weight` for XGBoost.
- **Raw morphology** — only per-record z-score; no filtering, no augmentation.
- **100 Hz** — uses PTB-XL `records100/` files (no resampling).
- **Reproducible** — `set_determinism(42)` + `CUBLAS_WORKSPACE_CONFIG=:4096:8` + single-threaded XGBoost.
- **Embeddings computed with `model.eval()` + `torch.no_grad()`** and a dedicated dropout-free `.penultimate()` path → deterministic, no val/test inversion.

## Run locally

```bash
cd <repo-root>
pip install -r approaches/mi_detection_prod/requirements.txt
PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  python -m approaches.mi_detection_prod.run
```

Results land in `approaches/mi_detection_prod/results/`:
`checkpoints/`, `histories/`, `embeddings/`, `predictions/`,
`final_report.{txt,json}`, `metrics_table.md`, `confusion_matrix.png`, `xgb_model.json`, `config_snapshot.json`.

## Google Colab — minimal cell snippet

> Paste each block as its own Colab cell, in order.

**Cell 1 — set env vars BEFORE any import (critical for determinism):**
```python
import os
os.environ["PYTHONHASHSEED"] = "42"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
```

**Cell 2 — install:**
```python
!pip install -q kagglehub wfdb "xgboost>=2.0,<3.0" scikit-learn pandas matplotlib
# Torch is preinstalled on Colab runtimes; uncomment to pin:
# !pip install -q "torch>=2.2,<3.0"
```

**Cell 3 — fetch code:**
```python
!git clone https://github.com/kanakkholwal/better-detection.git
%cd better-detection
```

**Cell 4 — authenticate Kaggle (needed by kagglehub):**
```python
# Upload kaggle.json (Kaggle > Account > Create New API Token) when prompted.
from google.colab import files
files.upload()
!mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
```

**Cell 5 — run end-to-end:**
```python
!python -m approaches.mi_detection_prod.run
```

**Cell 6 — view report:**
```python
from pathlib import Path
print(Path("approaches/mi_detection_prod/results/final_report.txt").read_text())
```

Expected runtime on a single Colab T4 GPU: ~15–25 minutes after PTB-XL download (~1.7 GB, one-time).

## Post-hoc tuning (no retraining)

After `run.py` has produced `results/predictions/{val,test}_predictions.csv`:

```python
!python -m approaches.mi_detection_prod.tune
```

Grid-searches an accuracy-optimal threshold on val for each base model + ensemble,
and fits a logistic-regression meta-learner (5-fold OOF on val for threshold
tuning, full val for final fit). Outputs: `results/tuned_metrics_table.md`,
`results/tuned_report.json`.

## K-fold patient-wise ensemble (higher-lift path)

For the accuracy target, run the 5-fold ensemble instead of `run.py`. Test set is
held fixed (same patient-wise split as `run.py`); train+val are combined and
split 5 ways with `GroupKFold` over patient IDs.

```python
!python -m approaches.mi_detection_prod.kfold
```

Produces 15 CNN checkpoints (3 archs × 5 folds) + 5 XGBoost models, OOF
predictions on dev, fold-averaged predictions on test, and both
AUC-weighted-average and logistic-regression meta-learner ensembles with
accuracy-optimal thresholds.

Outputs: `results/kfold/kfold_report.{md,json}`,
`results/kfold/kfold_{test,dev_oof}_predictions.csv`, per-fold checkpoints.

Expected runtime: ~60–90 minutes on a T4 GPU.

## Targets

- **Test AUC (ensemble) ≥ 0.95**
- **Test Accuracy (ensemble) ≥ 0.95** (requires K-fold; single-pass run.py gets ~0.93)
- Per-model generalization gap `|val AUC − test AUC| < 0.03`
- If any single model reports test AUC > 0.99, treat as leakage and stop.
