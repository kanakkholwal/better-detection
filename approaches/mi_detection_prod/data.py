"""PTB-XL data loading, labeling, z-score normalization, and patient-wise splitting.

No augmentation, no SMOTE, no oversampling. Raw morphology preserved; only per-record
z-score normalization is applied. All splits are patient-wise via GroupShuffleSplit.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight as sk_compute_class_weight
from torch.utils.data import Dataset

from .config import INPUT_LENGTH, NUM_LEADS, PATHS, SAMPLING_RATE


def locate_ptbxl() -> Tuple[str, str]:
    """Download PTB-XL via kagglehub and return (ptbxl_database.csv, data_root)."""
    import kagglehub

    root = kagglehub.dataset_download(PATHS["dataset_slug"])
    csv_path = None
    for dirpath, _, files in os.walk(root):
        if "ptbxl_database.csv" in files:
            csv_path = os.path.join(dirpath, "ptbxl_database.csv")
            data_root = dirpath
            break
    if csv_path is None:
        raise FileNotFoundError("ptbxl_database.csv not found under " + root)
    return csv_path, data_root


def load_metadata(csv_path: str, scp_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(csv_path, index_col="ecg_id")
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)
    df = df.sort_index()

    scp = pd.read_csv(scp_path)
    scp_code_col = scp.columns[0]
    scp[scp_code_col] = scp[scp_code_col].astype(str)
    scp = scp[scp["diagnostic"] == 1].copy()
    return df, scp


def filter_mi_normal(df: pd.DataFrame, scp: pd.DataFrame) -> pd.DataFrame:
    """Binary label: 1 = MI, 0 = NORM. Drop ambiguous / neither / both."""
    code_col = scp.columns[0]
    mi_codes = set(scp.loc[scp["diagnostic_class"] == "MI", code_col].astype(str))
    norm_codes = set(scp.loc[scp["diagnostic_class"] == "NORM", code_col].astype(str))

    def classify(codes: dict) -> int:
        keys = set(map(str, codes.keys()))
        has_mi = len(keys & mi_codes) > 0
        has_norm = len(keys & norm_codes) > 0
        if has_mi and not has_norm:
            return 1
        if has_norm and not has_mi:
            return 0
        return -1

    df = df.copy()
    df["label"] = df["scp_codes"].apply(classify)
    n_before = len(df)
    df = df[df["label"].isin([0, 1])].copy()
    n_after = len(df)

    print(
        f"[data] filter_mi_normal: kept {n_after}/{n_before} rows "
        f"(MI={int((df['label']==1).sum())}, NORM={int((df['label']==0).sum())}, "
        f"dropped ambiguous/other={n_before-n_after})"
    )
    return df


def _read_lr_signal(data_root: str, rel_path: str) -> np.ndarray:
    import wfdb

    full = os.path.join(data_root, rel_path)
    sig, meta = wfdb.rdsamp(full)
    if meta["fs"] != SAMPLING_RATE:
        raise ValueError(f"Expected {SAMPLING_RATE} Hz, got {meta['fs']} Hz for {rel_path}")
    if sig.shape[0] != INPUT_LENGTH or sig.shape[1] != NUM_LEADS:
        raise ValueError(
            f"Unexpected shape {sig.shape} for {rel_path}; need ({INPUT_LENGTH},{NUM_LEADS})"
        )
    return sig.astype(np.float32)


def load_signals_100hz(
    df: pd.DataFrame, data_root: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load all 12-lead ECGs at 100 Hz. Returns X (N,1000,12), y (N,), patient_ids (N,)."""
    if "filename_lr" not in df.columns:
        raise KeyError("filename_lr column missing from PTB-XL metadata")

    n = len(df)
    X = np.zeros((n, INPUT_LENGTH, NUM_LEADS), dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.int64)
    pids = df["patient_id"].to_numpy()

    rel_paths = df["filename_lr"].tolist()
    log_every = max(1, n // 20)
    for i, rel in enumerate(rel_paths):
        X[i] = _read_lr_signal(data_root, rel)
        if (i + 1) % log_every == 0 or i + 1 == n:
            print(f"[data] loaded {i+1}/{n} signals", flush=True)
    return X, y, pids


def zscore_per_record(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-record, per-lead z-score. Computed independently per sample → no leakage."""
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    return ((X - mean) / (std + eps)).astype(np.float32)


def patient_wise_split(
    groups: np.ndarray,
    test_frac: float,
    val_frac: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two-stage GroupShuffleSplit. Returns disjoint-patient train/val/test indices."""
    n = len(groups)
    all_idx = np.arange(n)

    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    trainval_idx, test_idx = next(gss1.split(all_idx, groups=groups))

    rel_val = val_frac / (1.0 - test_frac)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=rel_val, random_state=seed + 1)
    tr_rel, va_rel = next(gss2.split(trainval_idx, groups=groups[trainval_idx]))
    train_idx = trainval_idx[tr_rel]
    val_idx = trainval_idx[va_rel]

    tr_p = set(groups[train_idx].tolist())
    va_p = set(groups[val_idx].tolist())
    te_p = set(groups[test_idx].tolist())
    assert tr_p.isdisjoint(va_p), "train/val share patients"
    assert tr_p.isdisjoint(te_p), "train/test share patients"
    assert va_p.isdisjoint(te_p), "val/test share patients"

    print(
        f"[data] patient split: train={len(tr_p)} val={len(va_p)} test={len(te_p)} patients "
        f"| records train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}"
    )
    return train_idx, val_idx, test_idx


def compute_class_weight(y_train: np.ndarray) -> dict:
    classes = np.array([0, 1], dtype=np.int64)
    w = sk_compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    return {int(c): float(wc) for c, wc in zip(classes, w)}


class ECGDataset(Dataset):
    """Wraps (X, y). X shape (N, 1000, 12) float32; returns (12, 1000) tensors."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        if X.dtype != np.float32:
            X = X.astype(np.float32)
        self.X = np.transpose(X, (0, 2, 1))  # (N, 12, 1000)
        self.y = y.astype(np.float32)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, i: int):
        return (
            torch.from_numpy(self.X[i]),
            torch.tensor(self.y[i], dtype=torch.float32),
        )
