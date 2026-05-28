"""End-to-end entrypoint: load PTB-XL, split patient-wise, train three CNNs + XGB, ensemble, report."""

from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .config import (
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
    save_report,
    train_xgb_on_embeddings,
    weighted_ensemble,
)
from .train import train_all_models


def _snapshot_env() -> dict:
    import torch

    snap = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda": torch.version.cuda if torch.cuda.is_available() else None,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "seed": SEED,
        "test_frac": TEST_FRAC,
        "val_frac": VAL_FRAC,
        "xgb": XGB,
    }
    return snap


def main() -> None:
    set_determinism(SEED)
    ensure_dirs()

    snap = _snapshot_env()
    with open(Path(PATHS["output_dir"]) / "config_snapshot.json", "w") as f:
        json.dump(snap, f, indent=2, default=str)
    print("[run] env snapshot:", json.dumps(snap, indent=2, default=str))

    # ---- data ----
    csv_path, data_root = locate_ptbxl()
    scp_path = os.path.join(data_root, "scp_statements.csv")
    df, scp = load_metadata(csv_path, scp_path)
    df = filter_mi_normal(df, scp)

    X, y, pids = load_signals_100hz(df, data_root)
    X = zscore_per_record(X)
    print(f"[run] X={X.shape} y={y.shape} unique_patients={len(np.unique(pids))}")

    tr, va, te = patient_wise_split(pids, TEST_FRAC, VAL_FRAC, SEED)
    cw = compute_class_weight(y[tr])
    print(f"[run] class_weight(train)={cw}")
    print(
        f"[run] class counts — train: "
        f"pos={int((y[tr]==1).sum())} neg={int((y[tr]==0).sum())} | "
        f"val: pos={int((y[va]==1).sum())} neg={int((y[va]==0).sum())} | "
        f"test: pos={int((y[te]==1).sum())} neg={int((y[te]==0).sum())}"
    )

    train_ds = ECGDataset(X[tr], y[tr])
    val_ds = ECGDataset(X[va], y[va])
    test_ds = ECGDataset(X[te], y[te])

    # ---- train three CNNs ----
    trained = train_all_models(train_ds, val_ds, cw)

    # ---- embeddings from Inception → XGBoost ----
    inc_model, inc_meta = trained["Inception"]
    emb_tr = extract_embeddings(inc_model, train_ds)
    emb_va = extract_embeddings(inc_model, val_ds)
    emb_te = extract_embeddings(inc_model, test_ds)

    emb_dir = Path(PATHS["embeddings_dir"])
    np.save(emb_dir / "inception_train.npy", emb_tr)
    np.save(emb_dir / "inception_val.npy", emb_va)
    np.save(emb_dir / "inception_test.npy", emb_te)

    xgb_clf = train_xgb_on_embeddings(emb_tr, y[tr], emb_va, y[va], XGB)
    xgb_clf.save_model(str(Path(PATHS["output_dir"]) / "xgb_model.json"))

    # ---- collect probabilities ----
    probs_val, probs_test, val_aucs = {}, {}, {}
    for name, (m, _) in trained.items():
        probs_val[name] = predict_probs(m, val_ds)
        probs_test[name] = predict_probs(m, test_ds)
        val_aucs[name] = float(roc_auc_score(y[va], probs_val[name]))

    probs_val["XGB"] = xgb_clf.predict_proba(emb_va)[:, 1].astype(np.float32)
    probs_test["XGB"] = xgb_clf.predict_proba(emb_te)[:, 1].astype(np.float32)
    val_aucs["XGB"] = float(roc_auc_score(y[va], probs_val["XGB"]))

    ens_val, weights = weighted_ensemble(probs_val, val_aucs)
    ens_test, _ = weighted_ensemble(probs_test, val_aucs)

    # ---- metrics per model + ensemble ----
    per_model_test = {n: compute_metrics(y[te], probs_test[n]) for n in probs_test}
    per_model_test["Ensemble(TITAN)"] = compute_metrics(y[te], ens_test)

    per_model_val = {n: compute_metrics(y[va], probs_val[n]) for n in probs_val}
    per_model_val["Ensemble(TITAN)"] = compute_metrics(y[va], ens_val)

    # ---- save predictions ----
    pred_dir = Path(PATHS["predictions_dir"])
    val_df = pd.DataFrame({"y": y[va]})
    test_df = pd.DataFrame({"y": y[te]})
    for n in probs_val:
        val_df[f"p_{n}"] = probs_val[n]
        test_df[f"p_{n}"] = probs_test[n]
    val_df["p_Ensemble"] = ens_val
    test_df["p_Ensemble"] = ens_test
    val_df.to_csv(pred_dir / "val_predictions.csv", index=False)
    test_df.to_csv(pred_dir / "test_predictions.csv", index=False)

    # ---- final report ----
    save_report(per_model_test, per_model_val, weights, val_aucs, PATHS["output_dir"])

    # ---- sanity gates ----
    print("\n[run] summary:")
    for n in per_model_test:
        v_auc = per_model_val[n]["auc"]
        t_auc = per_model_test[n]["auc"]
        gap = abs(v_auc - t_auc)
        flag = " !!" if t_auc > 0.99 else ("  overfit?" if gap > 0.03 else "")
        print(f"  {n:<20} val_auc={v_auc:.4f}  test_auc={t_auc:.4f}  gap={gap:.4f}{flag}")

    ens_auc = per_model_test["Ensemble(TITAN)"]["auc"]
    print(f"\n[run] Ensemble test AUC = {ens_auc:.4f} (target ≥ 0.95)")
    if ens_auc > 0.99:
        print("[run] WARNING: AUC > 0.99 — check for leakage as per spec.")


if __name__ == "__main__":
    main()
