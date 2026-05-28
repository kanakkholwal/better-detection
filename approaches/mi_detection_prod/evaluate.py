"""Embedding extraction, XGBoost on embeddings, weighted ensemble, metric reporting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from .config import BATCH_SIZE, EMBEDDING_DIM, PATHS, XGB, device


def extract_embeddings(model, ds, batch_size: int = BATCH_SIZE) -> np.ndarray:
    """model.eval() + no_grad → dropout off, BN in eval. Dropout-free by construction."""
    import torch
    from torch.utils.data import DataLoader

    dev = device()
    model = model.to(dev)
    model.eval()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    out = np.zeros((len(ds), EMBEDDING_DIM), dtype=np.float32)
    i = 0
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(dev, non_blocking=True)
            emb = model.penultimate(xb).detach().cpu().numpy()
            out[i:i + emb.shape[0]] = emb
            i += emb.shape[0]
    return out


def predict_probs(model, ds, batch_size: int = BATCH_SIZE) -> np.ndarray:
    import torch
    from torch.utils.data import DataLoader

    dev = device()
    model = model.to(dev)
    model.eval()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    out = np.zeros(len(ds), dtype=np.float32)
    i = 0
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(dev, non_blocking=True)
            p = torch.sigmoid(model(xb)).detach().cpu().numpy()
            out[i:i + p.shape[0]] = p
            i += p.shape[0]
    return out


def train_xgb_on_embeddings(
    emb_train: np.ndarray, y_train: np.ndarray,
    emb_val: np.ndarray, y_val: np.ndarray,
    cfg: dict = XGB,
):
    from xgboost import XGBClassifier

    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale_pos_weight = (n_neg / max(n_pos, 1))
    print(f"[xgb] n_pos={n_pos} n_neg={n_neg} scale_pos_weight={scale_pos_weight:.3f}")

    params = dict(cfg)
    params["scale_pos_weight"] = scale_pos_weight
    params["early_stopping_rounds"] = 30

    clf = XGBClassifier(**params)
    clf.fit(emb_train, y_train, eval_set=[(emb_val, y_val)], verbose=False)
    return clf


def weighted_ensemble(
    probs_by_model: Dict[str, np.ndarray], val_aucs: Dict[str, float],
) -> Tuple[np.ndarray, Dict[str, float]]:
    names = list(probs_by_model.keys())
    aucs = np.array([max(val_aucs[n], 0.0) for n in names], dtype=np.float64)
    if aucs.sum() <= 0:
        weights = np.ones_like(aucs) / len(aucs)
    else:
        weights = aucs / aucs.sum()
    stacked = np.stack([probs_by_model[n] for n in names], axis=0)
    ens = (weights.reshape(-1, 1) * stacked).sum(axis=0)
    return ens.astype(np.float32), {n: float(w) for n, w in zip(names, weights)}


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "threshold": float(threshold),
    }


def _format_report_txt(per_model: dict, weights: dict, val_aucs: dict) -> str:
    lines = []
    lines.append("PTB-XL MI Detection — Final Report (patient-wise, no SMOTE)")
    lines.append("=" * 72)
    lines.append("")
    lines.append("Ensemble weights (softmax-free; w_i = val_auc_i / sum val_auc_j):")
    for n, w in weights.items():
        lines.append(f"  {n:<18} weight={w:.4f}  val_auc={val_aucs.get(n, float('nan')):.4f}")
    lines.append("")
    header = f"{'Model':<20}{'AUC':>8}{'Acc':>8}{'F1':>8}{'Sens':>8}{'Spec':>8}"
    lines.append(header)
    lines.append("-" * len(header))
    for name, m in per_model.items():
        lines.append(
            f"{name:<20}{m['auc']:>8.4f}{m['accuracy']:>8.4f}{m['f1']:>8.4f}"
            f"{m['sensitivity']:>8.4f}{m['specificity']:>8.4f}"
        )
    lines.append("")
    lines.append("Confusion matrices (test) [[TN, FP], [FN, TP]]:")
    for name, m in per_model.items():
        lines.append(f"  {name:<20} {m['confusion_matrix']}")
    return "\n".join(lines) + "\n"


def _format_metrics_md(per_model: dict) -> str:
    rows = ["| Model | AUC | Acc | F1 | Sens | Spec |", "|---|---:|---:|---:|---:|---:|"]
    for n, m in per_model.items():
        rows.append(
            f"| {n} | {m['auc']:.4f} | {m['accuracy']:.4f} | {m['f1']:.4f} | "
            f"{m['sensitivity']:.4f} | {m['specificity']:.4f} |"
        )
    return "\n".join(rows) + "\n"


def save_report(
    per_model_test: dict,
    per_model_val: dict,
    weights: dict,
    val_aucs: dict,
    out_dir: str,
    ensemble_test_cm_name: str = "Ensemble(TITAN)",
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    final = {
        "per_model_test": per_model_test,
        "per_model_val": per_model_val,
        "ensemble_weights": weights,
        "val_aucs": val_aucs,
    }
    with open(out / "final_report.json", "w") as f:
        json.dump(final, f, indent=2)

    with open(out / "final_report.txt", "w") as f:
        f.write(_format_report_txt(per_model_test, weights, val_aucs))

    with open(out / "metrics_table.md", "w") as f:
        f.write("## Test set (patient-wise)\n\n")
        f.write(_format_metrics_md(per_model_test))
        f.write("\n## Validation set\n\n")
        f.write(_format_metrics_md(per_model_val))

    _save_cm_plot(per_model_test.get(ensemble_test_cm_name, {}).get("confusion_matrix"), out / "confusion_matrix.png")


def _save_cm_plot(cm: list | None, path: Path) -> None:
    if cm is None:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        arr = np.array(cm)
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(arr, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Normal", "MI"])
        ax.set_yticklabels(["Normal", "MI"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Ensemble (TITAN) — Test CM")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, int(arr[i, j]), ha="center", va="center",
                        color="white" if arr[i, j] > arr.max() / 2 else "black")
        fig.colorbar(im, ax=ax, shrink=0.7)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"[report] CM plot skipped: {e}")
