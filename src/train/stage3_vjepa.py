from __future__ import annotations

from datetime import datetime

import pandas as pd
import torch
from tqdm import tqdm

from src.config import (
    BATCH_SIZE,
    COMMA2K19_STAGE3_TRAIN_MANIFEST,
    COMMA2K19_STAGE3_VAL_MANIFEST,
    DEVICE,
    STAGE3_EPOCHS,
    STAGE3_HEAD_LR,
    STAGE3_TRAIN_SAMPLE_LIMIT,
    STAGE3_TRAIN_TEMPORAL_STRIDE,
    STAGE3_VAL_SAMPLE_LIMIT,
    STAGE3_VAL_TEMPORAL_STRIDE,
    STAGE3_VJEPA_CHECKPOINT,
    STAGE3_VJEPA_MODEL,
)
from src.datasets.comma2k19_stage3_vjepa import Comma2k19Stage3VJEPADataset
from src.models.stage3_vjepa import Stage3VJEPA
from src.train.stage3 import (
    _class_weights,
    _classification_metrics,
    _limited_dataset,
    _loader,
    _loss,
    _print_metrics_table,
    _selection_score,
    _validate_all,
)

HISTORY_KEYS = [
    "epoch", "arch", "dataset", "batch_size", "selection_metric_name", "selection_metric",
    "train_loss", "val_loss", "train_accel_loss", "train_steer_loss",
    "train_accel_accuracy", "val_accel_accuracy", "train_accel_macro_f1", "val_accel_macro_f1",
    "train_steer_accuracy", "val_steer_accuracy", "train_steer_macro_f1", "val_steer_macro_f1",
    "selection", "comma_val_accel_accuracy", "comma_val_accel_macro_f1", "comma_val_steer_accuracy",
    "comma_val_steer_macro_f1", "comma_selection",
]


def _datasets():
    if not COMMA2K19_STAGE3_TRAIN_MANIFEST.is_file():
        raise FileNotFoundError(f"missing comma2k19 Stage3 train manifest: {COMMA2K19_STAGE3_TRAIN_MANIFEST}")
    train, before, after = _limited_dataset(
        Comma2k19Stage3VJEPADataset(COMMA2K19_STAGE3_TRAIN_MANIFEST),
        STAGE3_TRAIN_TEMPORAL_STRIDE,
        STAGE3_TRAIN_SAMPLE_LIMIT,
    )
    summary = {"comma_train_before": before, "comma_train_after": after}
    val = {}
    if COMMA2K19_STAGE3_VAL_MANIFEST.is_file():
        ds, before, after = _limited_dataset(
            Comma2k19Stage3VJEPADataset(COMMA2K19_STAGE3_VAL_MANIFEST),
            STAGE3_VAL_TEMPORAL_STRIDE,
            STAGE3_VAL_SAMPLE_LIMIT,
        )
        val["comma2k19"] = ds
        summary.update(comma_val_before=before, comma_val_after=after)
    return train, val, summary


def _append_history(history: dict, epoch: int, train_loss: float, train_accel_loss: float, train_steer_loss: float, train_metrics: dict, metrics: dict, selection: float) -> None:
    overall = metrics["overall"]
    comma = metrics.get("comma2k19")
    row = {
        "epoch": epoch,
        "arch": "vjepa_vitl_224",
        "dataset": "comma2k19",
        "batch_size": BATCH_SIZE,
        "selection_metric_name": "overall.selection",
        "selection_metric": selection,
        "train_loss": train_loss,
        "val_loss": overall.get("loss", float("nan")),
        "train_accel_loss": train_accel_loss,
        "train_steer_loss": train_steer_loss,
        "train_accel_accuracy": train_metrics["accel"]["accuracy"],
        "train_accel_macro_f1": train_metrics["accel"]["macro_f1"],
        "train_steer_accuracy": train_metrics["steer"]["accuracy"],
        "train_steer_macro_f1": train_metrics["steer"]["macro_f1"],
        "val_accel_accuracy": overall["accel"]["accuracy"],
        "val_accel_macro_f1": overall["accel"]["macro_f1"],
        "val_steer_accuracy": overall["steer"]["accuracy"],
        "val_steer_macro_f1": overall["steer"]["macro_f1"],
        "selection": overall["selection"],
        "comma_val_accel_accuracy": comma["accel"]["accuracy"] if comma else float("nan"),
        "comma_val_accel_macro_f1": comma["accel"]["macro_f1"] if comma else float("nan"),
        "comma_val_steer_accuracy": comma["steer"]["accuracy"] if comma else float("nan"),
        "comma_val_steer_macro_f1": comma["steer"]["macro_f1"] if comma else float("nan"),
        "comma_selection": comma["selection"] if comma else float("nan"),
    }
    for key in history:
        history[key].append(row.get(key, float("nan")))


def _save_history(history: dict, run_id: str):
    history_dir = STAGE3_VJEPA_MODEL / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    path = history_dir / f"{run_id}_history.csv"
    pd.DataFrame(history).to_csv(path, index=False)
    return path


def _checkpoint_payload(model: Stage3VJEPA, epoch: int, train_loss: float, metrics: dict, history: dict) -> dict:
    return {
        "arch": "vjepa_vitl_224",
        "encoder_checkpoint": str(STAGE3_VJEPA_CHECKPOINT),
        "head": model.head.state_dict(),
        "epoch": epoch,
        "train_loss": train_loss,
        "dataset": "comma2k19",
        "selection_metric_name": "overall.selection",
        "selection_metric": float(metrics["overall"]["selection"]),
        "history": history,
    }


def fit_stage3_vjepa():
    if not STAGE3_VJEPA_CHECKPOINT.is_file():
        raise FileNotFoundError(f"missing V-JEPA checkpoint: {STAGE3_VJEPA_CHECKPOINT}")
    out = STAGE3_VJEPA_MODEL
    out.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    history = {k: [] for k in HISTORY_KEYS}

    train_dataset, val_datasets, _ = _datasets()
    print("=== Stage 3 V-JEPA Dataset ===")
    print("Dataset: comma2k19")
    print("Architecture: vjepa_vitl_224")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {sum(len(v) for v in val_datasets.values())}")
    print(f"Batch size: {BATCH_SIZE}")

    train_loader = _loader(train_dataset, shuffle=True)
    val_loaders = {name: _loader(ds, shuffle=False) for name, ds in val_datasets.items()}
    model = Stage3VJEPA(STAGE3_VJEPA_CHECKPOINT).to(DEVICE)
    opt = torch.optim.AdamW(model.head.parameters(), lr=STAGE3_HEAD_LR)
    accel_class_weights = _class_weights("accel")
    steer_class_weights = _class_weights("steer")
    best = -1.0
    best_metrics = None
    best_epoch = 0

    for epoch in range(STAGE3_EPOCHS):
        model.train()
        total_loss = total_accel_loss = total_steer_loss = 0.0
        train_accel_pred, train_accel_target, train_steer_pred, train_steer_target = [], [], [], []
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{STAGE3_EPOCHS} V-JEPA")
        for batch in progress:
            accel, steer = model(batch["video"].to(DEVICE, non_blocking=True))
            loss, loss_accel, loss_steer = _loss(accel, steer, batch, accel_class_weights, steer_class_weights)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += float(loss.detach().cpu())
            total_accel_loss += float(loss_accel.cpu())
            total_steer_loss += float(loss_steer.cpu())
            train_accel_pred.extend(accel.argmax(1).detach().cpu().tolist())
            train_steer_pred.extend(steer.argmax(1).detach().cpu().tolist())
            train_accel_target.extend(batch["accel_label"].tolist())
            train_steer_target.extend(batch["steer_label"].tolist())
            progress.set_postfix(loss=f"{float(loss.detach().cpu()):.4f}")

        steps = max(1, len(train_loader))
        train_loss = total_loss / steps
        train_accel_loss = total_accel_loss / steps
        train_steer_loss = total_steer_loss / steps
        train_metrics = {
            "accel": _classification_metrics(train_accel_pred, train_accel_target, 4),
            "steer": _classification_metrics(train_steer_pred, train_steer_target, 3),
        }
        train_metrics["selection"] = _selection_score(train_metrics["accel"]["macro_f1"], train_metrics["steer"]["macro_f1"])
        metrics = _validate_all(model, val_loaders, accel_class_weights, steer_class_weights) if val_loaders else {"overall": {"loss": float("nan"), "accel": {"accuracy": float("nan"), "macro_f1": float("nan")}, "steer": {"accuracy": float("nan"), "macro_f1": float("nan")}, "selection": float("nan")}}
        _print_metrics_table(metrics, prefix=f"epoch={epoch + 1} ")
        select = metrics["overall"]["selection"]
        _append_history(history, epoch + 1, train_loss, train_accel_loss, train_steer_loss, train_metrics, metrics, select)
        _save_history(history, run_id)
        if select > best:
            best = select
            best_epoch = epoch + 1
            best_metrics = metrics
            torch.save(_checkpoint_payload(model, epoch + 1, train_loss, metrics, history), out / "best.pt")

    if best_metrics:
        print(f"Best epoch: {best_epoch}")
        _print_metrics_table(best_metrics)
        print(f"selection={best:.5f}")