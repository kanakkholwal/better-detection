"""Post-hoc threshold tuning + meta-learner stacking on saved predictions.

Reads `results/predictions/{val,test}_predictions.csv` (produced by run.py) and
produces a tuned report without retraining any base model.

Two complementary lifts:
  1. Per-model threshold tuning on val (max accuracy) applied to test.
  2. Meta-learner (LogisticRegression) stacked on the four base-model probabilities.
     Validation-fold honest probabilities for the meta-learner come from 5-fold CV
     within val (StratifiedKFold), so the threshold tuning for the meta-learner
     is not in-sample.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from .config import PATHS, SEED
from .evaluate import compute_metrics

BASE_MODELS: Tuple[str, ...] = ("SimpleCNN", "Inception", "ResNet", "XGB")


def _best_threshold(y: np.ndarray, p: np.ndarray, criterion: str = "accuracy") -> Tuple[float, float]:
    """Grid-search threshold on (y, p). Returns (threshold, best_score)."""
    thresholds = np.linspace(0.01, 0.99, 197)  # step 0.005
    best_t, best_s = 0.5, -np.inf
    for t in thresholds:
        yhat = (p >= t).astype(int)
        if criterion == "accuracy":
            s = float((yhat == y).mean())
        elif criterion == "f1":
            tp = int(((yhat == 1) & (y == 1)).sum())
            fp = int(((yhat == 1) & (y == 0)).sum())
            fn = int(((yhat == 0) & (y == 1)).sum())
            s = (2 * tp) / max(2 * tp + fp + fn, 1)
        elif criterion == "youden":
            tp = int(((yhat == 1) & (y == 1)).sum())
            tn = int(((yhat == 0) & (y == 0)).sum())
            fp = int(((yhat == 1) & (y == 0)).sum())
            fn = int(((yhat == 0) & (y == 1)).sum())
            sens = tp / max(tp + fn, 1)
            spec = tn / max(tn + fp, 1)
            s = sens + spec - 1.0
        else:
            raise ValueError(criterion)
        if s > best_s:
            best_s, best_t = s, float(t)
    return best_t, best_s


def _format_md_row(name: str, m: dict, threshold: float) -> str:
    return (
        f"| {name} | {threshold:.3f} | {m['auc']:.4f} | {m['accuracy']:.4f} | "
        f"{m['f1']:.4f} | {m['sensitivity']:.4f} | {m['specificity']:.4f} |"
    )


def main() -> None:
    pred_dir = Path(PATHS["predictions_dir"])
    out_dir = Path(PATHS["output_dir"])
    val = pd.read_csv(pred_dir / "val_predictions.csv")
    test = pd.read_csv(pred_dir / "test_predictions.csv")

    y_val = val["y"].to_numpy()
    y_test = test["y"].to_numpy()
    X_val = val[[f"p_{n}" for n in BASE_MODELS]].to_numpy()
    X_test = test[[f"p_{n}" for n in BASE_MODELS]].to_numpy()

    # ---------- 1) Per-model threshold tuning on val, applied to test ----------
    tuned_thresholds: Dict[str, float] = {}
    tuned_rows_test: Dict[str, dict] = {}

    cols_to_tune: List[Tuple[str, np.ndarray, np.ndarray]] = [
        (n, val[f"p_{n}"].to_numpy(), test[f"p_{n}"].to_numpy()) for n in BASE_MODELS
    ]
    cols_to_tune.append(("Ensemble(orig)", val["p_Ensemble"].to_numpy(), test["p_Ensemble"].to_numpy()))

    for name, p_v, p_t in cols_to_tune:
        t, s_val = _best_threshold(y_val, p_v, criterion="accuracy")
        tuned_thresholds[name] = t
        tuned_rows_test[name] = compute_metrics(y_test, p_t, threshold=t)
        print(
            f"[tune] {name:<18} t*={t:.3f}  val_acc*={s_val:.4f}  "
            f"test_acc={tuned_rows_test[name]['accuracy']:.4f}  "
            f"test_auc={tuned_rows_test[name]['auc']:.4f}"
        )

    # ---------- 2) Meta-learner (LogisticRegression) stacked on val probs ----------
    meta = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)

    # Honest val probabilities via 5-fold CV for threshold tuning
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    p_meta_val_oof = cross_val_predict(
        LogisticRegression(C=1.0, max_iter=2000, random_state=SEED),
        X_val, y_val, cv=cv, method="predict_proba", n_jobs=1,
    )[:, 1]

    # Final meta-LR trained on ALL val, applied to test
    meta.fit(X_val, y_val)
    p_meta_test = meta.predict_proba(X_test)[:, 1]
    meta_coef = {n: float(c) for n, c in zip(BASE_MODELS, meta.coef_.ravel())}
    meta_intercept = float(meta.intercept_.ravel()[0])

    # Threshold tuned on out-of-fold val (no leakage onto test)
    t_meta, s_meta_val = _best_threshold(y_val, p_meta_val_oof, criterion="accuracy")
    m_meta_test_acc = compute_metrics(y_test, p_meta_test, threshold=t_meta)
    tuned_thresholds["Meta-LR"] = t_meta
    tuned_rows_test["Meta-LR"] = m_meta_test_acc

    print(
        f"[tune] Meta-LR           t*={t_meta:.3f}  oof_val_acc*={s_meta_val:.4f}  "
        f"test_acc={m_meta_test_acc['accuracy']:.4f}  test_auc={m_meta_test_acc['auc']:.4f}"
    )
    print(f"[tune] Meta-LR coefficients: {meta_coef}  intercept={meta_intercept:.4f}")

    # ---------- Write tuned report ----------
    md_lines = [
        "## Tuned test metrics (thresholds chosen to maximize val accuracy)",
        "",
        "| Model | Threshold | AUC | Acc | F1 | Sens | Spec |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, m in tuned_rows_test.items():
        md_lines.append(_format_md_row(name, m, tuned_thresholds[name]))
    md_lines.extend(
        [
            "",
            "### Meta-LR weights (stack on base-model probabilities)",
            "",
            "| Base model | Coefficient |",
            "|---|---:|",
            *[f"| {n} | {c:+.4f} |" for n, c in meta_coef.items()],
            f"| _intercept_ | {meta_intercept:+.4f} |",
            "",
            "_Thresholds are picked on val (accuracy-maximizing grid search over "
            "[0.01, 0.99] step 0.005). Meta-LR threshold is tuned on 5-fold OOF "
            "val predictions to avoid in-sample bias; the final Meta-LR is fit on "
            "the full val set before scoring test._",
        ]
    )
    (out_dir / "tuned_metrics_table.md").write_text("\n".join(md_lines) + "\n")

    payload = {
        "tuned_thresholds": tuned_thresholds,
        "tuned_test_metrics": tuned_rows_test,
        "meta_learner": {
            "coefficients": meta_coef,
            "intercept": meta_intercept,
            "oof_val_accuracy_at_best_threshold": float(s_meta_val),
        },
    }
    (out_dir / "tuned_report.json").write_text(json.dumps(payload, indent=2))

    # ---------- Console summary ----------
    print("\n[tune] === SUMMARY (test set) ===")
    print(f"{'Model':<18}{'Thresh':>9}{'AUC':>9}{'Acc':>9}{'F1':>9}{'Sens':>9}{'Spec':>9}")
    for name, m in tuned_rows_test.items():
        print(
            f"{name:<18}{tuned_thresholds[name]:>9.3f}{m['auc']:>9.4f}"
            f"{m['accuracy']:>9.4f}{m['f1']:>9.4f}"
            f"{m['sensitivity']:>9.4f}{m['specificity']:>9.4f}"
        )

    best_name = max(tuned_rows_test, key=lambda k: tuned_rows_test[k]["accuracy"])
    best = tuned_rows_test[best_name]
    print(
        f"\n[tune] Best tuned-accuracy: {best_name} — acc={best['accuracy']:.4f}, "
        f"auc={best['auc']:.4f} @ threshold={tuned_thresholds[best_name]:.3f}"
    )


if __name__ == "__main__":
    main()
