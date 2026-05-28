"""Training loop, early stopping on val AUC, LR schedule, deterministic data loading."""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from .config import (
    BATCH_SIZE,
    EPOCHS,
    LR,
    MIN_LR,
    MODEL_NAMES,
    PATHS,
    PATIENCE,
    REDUCE_LR_FACTOR,
    REDUCE_LR_PATIENCE,
    SEED,
    WEIGHT_DECAY,
    device,
)
from .models import build_model


def _seed_worker(worker_id: int) -> None:
    seed = (torch.initial_seed() + worker_id) % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def make_loader(ds: Dataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    gen = torch.Generator()
    gen.manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        generator=gen if shuffle else None,
        worker_init_fn=_seed_worker,
    )


class EarlyStopping:
    """Monitor val AUC (higher is better). Restores best weights when triggered."""

    def __init__(self, patience: int, mode: str = "max"):
        self.patience = patience
        self.mode = mode
        self.best: float = -float("inf") if mode == "max" else float("inf")
        self.best_epoch: int = -1
        self.best_state: Dict[str, torch.Tensor] | None = None
        self.counter: int = 0
        self.stop: bool = False

    def _is_better(self, cur: float) -> bool:
        return cur > self.best if self.mode == "max" else cur < self.best

    def step(self, metric: float, model: nn.Module, epoch: int) -> bool:
        if self._is_better(metric):
            self.best = metric
            self.best_epoch = epoch
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop

    def restore(self, model: nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def _forward_collect(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, dev: torch.device):
    model.eval()
    losses, probs, targets = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(dev, non_blocking=True)
            yb = yb.to(dev, non_blocking=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            losses.append(loss.item() * xb.size(0))
            probs.append(torch.sigmoid(logits).detach().cpu().numpy())
            targets.append(yb.detach().cpu().numpy())
    y_prob = np.concatenate(probs)
    y_true = np.concatenate(targets)
    avg_loss = float(sum(losses) / len(y_true))
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = float("nan")
    f1 = float(f1_score(y_true, (y_prob >= 0.5).astype(int), zero_division=0))
    return avg_loss, auc, f1


def _train_one_epoch(
    model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module, dev: torch.device,
) -> float:
    model.train()
    total, n = 0.0, 0
    for xb, yb in loader:
        xb = xb.to(dev, non_blocking=True)
        yb = yb.to(dev, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        optimizer.step()
        total += loss.item() * xb.size(0)
        n += xb.size(0)
    return total / max(n, 1)


def _pos_weight_from_class_weight(class_weight: dict) -> torch.Tensor:
    # BCEWithLogitsLoss expects pos_weight = weight_for_positive / weight_for_negative
    pw = float(class_weight[1]) / float(class_weight[0])
    return torch.tensor([pw], dtype=torch.float32)


def train_one_model(
    name: str,
    train_ds: Dataset,
    val_ds: Dataset,
    class_weight: dict,
    ckpt_path: str,
    history_path: str,
) -> Tuple[nn.Module, Dict]:
    dev = device()
    model = build_model(name).to(dev)

    pw = _pos_weight_from_class_weight(class_weight).to(dev)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=REDUCE_LR_FACTOR, patience=REDUCE_LR_PATIENCE, min_lr=MIN_LR,
    )

    train_loader = make_loader(train_ds, BATCH_SIZE, shuffle=True, seed=SEED)
    val_loader = make_loader(val_ds, BATCH_SIZE, shuffle=False, seed=SEED)

    es = EarlyStopping(patience=PATIENCE, mode="max")
    history: Dict[str, list] = {"train_loss": [], "val_loss": [], "val_auc": [], "val_f1": [], "lr": []}

    print(f"\n[train] === {name} | pos_weight={pw.item():.3f} | device={dev} ===")
    for epoch in range(1, EPOCHS + 1):
        tr_loss = _train_one_epoch(model, train_loader, optimizer, loss_fn, dev)
        va_loss, va_auc, va_f1 = _forward_collect(model, val_loader, loss_fn, dev)
        lr_now = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["val_auc"].append(va_auc)
        history["val_f1"].append(va_f1)
        history["lr"].append(lr_now)

        print(
            f"[train] {name} epoch {epoch:02d}/{EPOCHS} "
            f"train_loss={tr_loss:.4f} val_loss={va_loss:.4f} "
            f"val_auc={va_auc:.4f} val_f1={va_f1:.4f} lr={lr_now:.2e}",
            flush=True,
        )

        scheduler.step(va_auc)
        if es.step(va_auc, model, epoch):
            print(f"[train] {name} early stop at epoch {epoch}, best epoch {es.best_epoch} val_auc={es.best:.4f}")
            break

    es.restore(model)
    Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)

    meta = {
        "name": name,
        "best_epoch": es.best_epoch,
        "best_val_auc": es.best,
        "history": history,
    }
    Path(history_path).parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "w") as f:
        json.dump(meta, f, indent=2)

    return model, meta


def train_all_models(
    train_ds: Dataset, val_ds: Dataset, class_weight: dict,
) -> Dict[str, Tuple[nn.Module, Dict]]:
    trained: Dict[str, Tuple[nn.Module, Dict]] = {}
    for name in MODEL_NAMES:
        ckpt = str(Path(PATHS["checkpoints_dir"]) / f"{name}.pt")
        hist = str(Path(PATHS["histories_dir"]) / f"{name}_history.json")
        model, meta = train_one_model(name, train_ds, val_ds, class_weight, ckpt, hist)
        trained[name] = (model, meta)
    return trained
