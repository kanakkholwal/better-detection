"""Patient-wise 5-fold ensemble.

Holds the original test set fixed (same seed as run.py). Combines train+val into a
dev set, runs GroupKFold(5) on the dev set's patient IDs, and trains all three
CNNs + an XGBoost on Inception embeddings in each fold. Out-of-fold (OOF)
predictions on the dev set are used to train a logistic-regression meta-learner
and pick an operating threshold; test predictions are fold-averaged per
architecture before meta-learner scoring.

No SMOTE, no patient leakage (asserted), no augmentation, no retraining on test.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from .config import (
    K_FOLDS,
    MODEL_NAMES,
    PATHS,
    SEED,
    TEST_FRAC,
    VAL_FRAC,
    XGB,
    ensure_dirs,
    set_determinism,
)
from .data import (
    ECGDataset,
    compute_class_weight,
    filter_mi_normal,
    load_metadata,
    load_signals_100hz,
    locate_ptbxl,
    patient_wise_split,
    zscore_per_record,
)
from .evaluate import (
    compute_metrics,
    extract_embeddings,
    predict_probs,
    train_xgb_on_embeddings,
)
from .train import train_one_model
from .tune import _best_threshold

BASE_KEYS = ("SimpleCNN", "Inception", "ResNet", "XGB")


def _fmt_row(name: str, m: dict, threshold: float) -> str:
    return (
        f"| {name} | {threshold:.3f} | {m['auc']:.4f} | {m['accuracy']:.4f} | "
        f"{m['f1']:.4f} | {m['sensitivity']:.4f} | {m['specificity']:.4f} |"
    )


def main() -> None:
    set_determinism(SEED)
    ensure_dirs()

    # ---- load + split (keep test fixed, same as run.py) ----
    csv_path, data_root = locate_ptbxl()
    scp_path = os.path.join(data_root, "scp_statements.csv")
    df, scp = load_metadata(csv_path, scp_path)
    df = filter_mi_normal(df, scp)

    X, y, pids = load_signals_100hz(df, data_root)
    X = zscore_per_record(X)

    tr, va, te = patient_wise_split(pids, TEST_FRAC, VAL_FRAC, SEED)
    dev_idx = np.concatenate([tr, va])
    dev_idx.sort()  # deterministic order

    y_dev = y[dev_idx]
    pids_dev = pids[dev_idx]
    y_test = y[te]
    test_ds = ECGDataset(X[te], y_test)

    assert set(pids_dev.tolist()).isdisjoint(set(pids[te].tolist())), "test patient leakage"

    # ---- buffers ----
    dev_oof: Dict[str, np.ndarray] = {k: np.zeros(len(dev_idx), dtype=np.float32) for k in BASE_KEYS}
    test_sum: Dict[str, np.ndarray] = {k: np.zeros(len(te), dtype=np.float32) for k in BASE_KEYS}
    fold_val_aucs: list[dict] = []

    gkf = GroupKFold(n_splits=K_FOLDS)
    kfold_dir = Path(PATHS["kfold_dir"])

    for fold, (tr_rel, va_rel) in enumerate(gkf.split(dev_idx, y=y_dev, groups=pids_dev), start=1):
        train_idx = dev_idx[tr_rel]
        val_idx = dev_idx[va_rel]

        # patient disjointness within fold (sanity)
        tr_p = set(pids[train_idx].tolist())
        va_p = set(pids[val_idx].tolist())
        te_p = set(pids[te].tolist())
        assert tr_p.isdisjoint(va_p) and tr_p.isdisjoint(te_p) and va_p.isdisjoint(te_p), \
            f"patient leak in fold {fold}"

        print(
            f"\n[kfold] ==== fold {fold}/{K_FOLDS} | "
            f"train={len(train_idx)} val={len(val_idx)} test={len(te)} | "
            f"train_patients={len(tr_p)} val_patients={len(va_p)} ===="
        )

        train_ds = ECGDataset(X[train_idx], y[train_idx])
        val_ds = ECGDataset(X[val_idx], y[val_idx])
        cw = compute_class_weight(y[train_idx])

        # ---- CNNs ----
        fold_aucs = {}
        fold_models = {}
        for name in MODEL_NAMES:
            ckpt = str(kfold_dir / f"{name}_fold{fold}.pt")
            hist = str(kfold_dir / f"{name}_fold{fold}_history.json")
            model, meta = train_one_model(name, train_ds, val_ds, cw, ckpt, hist)
            fold_models[name] = model
            # OOF predictions for this fold's val partition
            p_val = predict_probs(model, val_ds)
            dev_oof[name][va_rel] = p_val
            p_test = predict_probs(model, test_ds)
            test_sum[name] += p_test
            fold_aucs[name] = float(roc_auc_score(y[val_idx], p_val))

        # ---- XGB on Inception embeddings ----
        inc = fold_models["Inception"]
        emb_tr = extract_embeddings(inc, train_ds)
        emb_va = extract_embeddings(inc, val_ds)
        emb_te = extract_embeddings(inc, test_ds)
        xgb_clf = train_xgb_on_embeddings(emb_tr, y[train_idx], emb_va, y[val_idx], XGB)
        p_val_xgb = xgb_clf.predict_proba(emb_va)[:, 1].astype(np.float32)
        p_test_xgb = xgb_clf.predict_proba(emb_te)[:, 1].astype(np.float32)
        dev_oof["XGB"][va_rel] = p_val_xgb
        test_sum["XGB"] += p_test_xgb
        fold_aucs["XGB"] = float(roc_auc_score(y[val_idx], p_val_xgb))

        xgb_clf.save_model(str(kfold_dir / f"xgb_fold{fold}.json"))
        fold_val_aucs.append(fold_aucs)
        print(f"[kfold] fold {fold} val AUCs: {fold_aucs}")

        # free GPU memory between folds
        del fold_models, inc, xgb_clf
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # ---- fold-averaged test predictions per architecture ----
    test_avg: Dict[str, np.ndarray] = {k: (test_sum[k] / K_FOLDS).astype(np.float32) for k in BASE_KEYS}

    # ---- metrics per architecture (default threshold 0.5) ----
    per_model_test_default = {k: compute_metrics(y_test, test_avg[k]) for k in BASE_KEYS}

    # ---- proportional-to-OOF-AUC ensemble ----
    mean_val_aucs = {
        k: float(roc_auc_score(y_dev, dev_oof[k])) for k in BASE_KEYS
    }
    w = np.array([max(mean_val_aucs[k], 0.0) for k in BASE_KEYS], dtype=np.float64)
    w = w / w.sum() if w.sum() > 0 else np.ones_like(w) / len(w)
    weights = {k: float(wi) for k, wi in zip(BASE_KEYS, w)}
    ens_dev = sum(w[i] * dev_oof[k] for i, k in enumerate(BASE_KEYS)).astype(np.float32)
    ens_test = sum(w[i] * test_avg[k] for i, k in enumerate(BASE_KEYS)).astype(np.float32)

    # ---- meta-learner on dev OOF → test ----
    X_dev = np.column_stack([dev_oof[k] for k in BASE_KEYS])
    X_test = np.column_stack([test_avg[k] for k in BASE_KEYS])
    meta = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)
    meta.fit(X_dev, y_dev)
    p_meta_dev = meta.predict_proba(X_dev)[:, 1].astype(np.float32)
    p_meta_test = meta.predict_proba(X_test)[:, 1].astype(np.float32)
    meta_coef = {n: float(c) for n, c in zip(BASE_KEYS, meta.coef_.ravel())}

    # ---- threshold tuning on dev (OOF) for every stream ----
    tuned_thresholds: Dict[str, float] = {}
    per_model_test_tuned: Dict[str, dict] = {}
    for k in BASE_KEYS:
        t, _ = _best_threshold(y_dev, dev_oof[k], criterion="accuracy")
        tuned_thresholds[k] = t
        per_model_test_tuned[k] = compute_metrics(y_test, test_avg[k], threshold=t)

    t_ens, _ = _best_threshold(y_dev, ens_dev, criterion="accuracy")
    tuned_thresholds["Ensemble(AUCw)"] = t_ens
    per_model_test_tuned["Ensemble(AUCw)"] = compute_metrics(y_test, ens_test, threshold=t_ens)

    t_meta, _ = _best_threshold(y_dev, p_meta_dev, criterion="accuracy")
    tuned_thresholds["Meta-LR"] = t_meta
    per_model_test_tuned["Meta-LR"] = compute_metrics(y_test, p_meta_test, threshold=t_meta)

    # ---- persist ----
    pred_df = pd.DataFrame({"y": y_test})
    for k in BASE_KEYS:
        pred_df[f"p_{k}"] = test_avg[k]
    pred_df["p_Ensemble_AUCw"] = ens_test
    pred_df["p_Meta_LR"] = p_meta_test
    pred_df.to_csv(kfold_dir / "kfold_test_predictions.csv", index=False)

    dev_pred_df = pd.DataFrame({"y": y_dev})
    for k in BASE_KEYS:
        dev_pred_df[f"p_{k}"] = dev_oof[k]
    dev_pred_df["p_Ensemble_AUCw"] = ens_dev
    dev_pred_df["p_Meta_LR"] = p_meta_dev
    dev_pred_df.to_csv(kfold_dir / "kfold_dev_oof_predictions.csv", index=False)

    payload = {
        "k_folds": K_FOLDS,
        "per_fold_val_aucs": fold_val_aucs,
        "oof_auc_on_dev": mean_val_aucs,
        "ensemble_weights": weights,
        "tuned_thresholds": tuned_thresholds,
        "per_model_test_default_threshold": per_model_test_default,
        "per_model_test_tuned": per_model_test_tuned,
        "meta_learner": {"coefficients": meta_coef, "intercept": float(meta.intercept_.ravel()[0])},
    }
    (kfold_dir / "kfold_report.json").write_text(json.dumps(payload, indent=2))

    # human-readable md
    lines = [
        f"# K-Fold ({K_FOLDS}) Patient-Wise Ensemble — Test Set",
        "",
        "## Threshold = 0.5 (raw averaged probabilities)",
        "",
        "| Model | AUC | Acc | F1 | Sens | Spec |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for k in BASE_KEYS:
        m = per_model_test_default[k]
        lines.append(
            f"| {k} | {m['auc']:.4f} | {m['accuracy']:.4f} | {m['f1']:.4f} | "
            f"{m['sensitivity']:.4f} | {m['specificity']:.4f} |"
        )
    lines.extend([
        "",
        "## Tuned thresholds (max accuracy on dev OOF)",
        "",
        "| Model | Threshold | AUC | Acc | F1 | Sens | Spec |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for k, m in per_model_test_tuned.items():
        lines.append(_fmt_row(k, m, tuned_thresholds[k]))
    lines.extend([
        "",
        "## Meta-LR coefficients",
        "",
        "| Base | Coef |",
        "|---|---:|",
        *[f"| {k} | {v:+.4f} |" for k, v in meta_coef.items()],
        f"| _intercept_ | {float(meta.intercept_.ravel()[0]):+.4f} |",
        "",
        f"_OOF AUC on dev set: " + ", ".join(f"{k}={mean_val_aucs[k]:.4f}" for k in BASE_KEYS) + "_",
    ])
    (kfold_dir / "kfold_report.md").write_text("\n".join(lines) + "\n")

    # console summary
    print("\n[kfold] === TEST SUMMARY ===")
    print(f"{'Model':<18}{'Thresh':>9}{'AUC':>9}{'Acc':>9}{'F1':>9}{'Sens':>9}{'Spec':>9}")
    for k, m in per_model_test_tuned.items():
        print(
            f"{k:<18}{tuned_thresholds[k]:>9.3f}{m['auc']:>9.4f}"
            f"{m['accuracy']:>9.4f}{m['f1']:>9.4f}"
            f"{m['sensitivity']:>9.4f}{m['specificity']:>9.4f}"
        )

    best_name = max(per_model_test_tuned, key=lambda k: per_model_test_tuned[k]["accuracy"])
    best = per_model_test_tuned[best_name]
    print(
        f"\n[kfold] Best accuracy: {best_name}  "
        f"acc={best['accuracy']:.4f}  auc={best['auc']:.4f}  "
        f"@ threshold={tuned_thresholds[best_name]:.3f}"
    )
    print(f"[kfold] Gate: acc ≥ 0.95 {'MET' if best['accuracy'] >= 0.95 else 'NOT MET'}; "
          f"auc ≥ 0.95 {'MET' if best['auc'] >= 0.95 else 'NOT MET'}.")


if __name__ == "__main__":
    main()
