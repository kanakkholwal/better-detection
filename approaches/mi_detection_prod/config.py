"""Central configuration and determinism setup for the MI-detection pipeline."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

SEED: int = 42

INPUT_LENGTH: int = 1000
NUM_LEADS: int = 12
SAMPLING_RATE: int = 100

TEST_FRAC: float = 0.15
VAL_FRAC: float = 0.15

BATCH_SIZE: int = 64
EPOCHS: int = 50
PATIENCE: int = 8
LR: float = 1e-3
WEIGHT_DECAY: float = 1e-4
REDUCE_LR_PATIENCE: int = 4
REDUCE_LR_FACTOR: float = 0.5
MIN_LR: float = 1e-6

EMBEDDING_DIM: int = 128

K_FOLDS: int = 5

XGB: dict = dict(
    n_estimators=600,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    device="cpu",
    eval_metric="auc",
    random_state=SEED,
    n_jobs=1,
)

_PKG_DIR = Path(__file__).resolve().parent
PATHS: dict = dict(
    dataset_slug="khyeh0719/ptb-xl-dataset",
    output_dir=str(_PKG_DIR / "results"),
    checkpoints_dir=str(_PKG_DIR / "results" / "checkpoints"),
    histories_dir=str(_PKG_DIR / "results" / "histories"),
    embeddings_dir=str(_PKG_DIR / "results" / "embeddings"),
    predictions_dir=str(_PKG_DIR / "results" / "predictions"),
    kfold_dir=str(_PKG_DIR / "results" / "kfold"),
)

MODEL_NAMES: tuple[str, ...] = ("SimpleCNN", "Inception", "ResNet")


def set_determinism(seed: int = SEED) -> None:
    """Seed every RNG source we touch and force deterministic kernels."""
    import torch  # lazy: tune.py doesn't need torch

    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
    except Exception:
        torch.use_deterministic_algorithms(True, warn_only=True)


def ensure_dirs() -> None:
    for key in ("output_dir", "checkpoints_dir", "histories_dir", "embeddings_dir", "predictions_dir", "kfold_dir"):
        Path(PATHS[key]).mkdir(parents=True, exist_ok=True)


def device():
    import torch  # lazy: tune.py doesn't need torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
