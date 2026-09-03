from __future__ import annotations

import pandas as pd
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from src.config import (
    BATCH_SIZE,
    COMMA2K19_STAGE3_TRAIN_MANIFEST,
    COMMA2K19_STAGE3_VAL_MANIFEST,
    DEVICE,
    EPOCHS,
    SEED,
    STAGE3_CLASS_WEIGHTS,
    STAGE3_LOSS_WEIGHTS,
    STAGE3_MODEL,
    STAGE3_NUM_WORKERS,
    STAGE3_RAW,
    STAGE3_TRAIN_TEMPORAL_STRIDE,
    STAGE3_VAL_TEMPORAL_STRIDE,
)
from src.datasets.comma2k19_stage3 import Comma2k19Stage3Dataset, Stage3DaconDataset
from src.models import Stage3MViT
from src.utils import set_seed

set_seed(SEED)


def _classification_metrics(pred: list[int], target: list[int], num_classes: int) -> dict:
    if not target:
        return {"accuracy": float("nan"), "macro_f1": float("nan"), "confusion_matrix": [], "prediction_distribution": []}
    pred_t = torch.tensor(pred)
    target_t = torch.tensor(target)
    matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for y, yhat in zip(target_t, pred_t):
        if 0 <= y < num_classes and 0 <= yhat < num_classes:
            matrix[y, yhat] += 1
    f1 = []
    for cls in range(num_classes):
        tp = matrix[cls, cls].item()
        fp = (matrix[:, cls].sum() - matrix[cls, cls]).item()
        fn = (matrix[cls, :].sum() - matrix[cls, cls]).item()
        denom = (2 * tp) + fp + fn
        f1.append((2 * tp) / denom if denom else 0.0)
    return {
        "accuracy": float((pred_t == target_t).float().mean()),
        "macro_f1": float(sum(f1) / num_classes),
        "confusion_matrix": matrix.tolist(),
        "prediction_distribution": torch.bincount(pred_t, minlength=num_classes).tolist(),
    }


def _stride_manifest(df: pd.DataFrame, stride: int) -> pd.DataFrame:
    if stride <= 1 or df.empty:
        return df.reset_index(drop=True)
    key = "segment_id" if "segment_id" in df.columns else "video_path"
    parts = [part.iloc[::stride] for _, part in df.groupby(key, sort=False)]
    return pd.concat(parts, ignore_index=True) if parts else df.reset_index(drop=True)


def _comma_dataset(path, stride: int):
    dataset = Comma2k19Stage3Dataset(path)
    before = len(dataset)
    dataset.df = _stride_manifest(dataset.df, stride)
    return dataset, before, len(dataset)


def _datasets():
    train_sets, val_sets = [], []
    summary = {
        "dacon_train": 0,
        "dacon_val": 0,
        "comma_train_before": 0,
        "comma_train_after": 0,
        "comma_val_before": 0,
        "comma_val_after": 0,
    }
    labels_path = STAGE3_RAW / "labels.csv"
    if labels_path.is_file():
        df = pd.read_csv(labels_path).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
        split = max(1, int(len(df) * 0.8)) if len(df) > 1 else len(df)
        train_sets.append(Stage3DaconDataset(df.iloc[:split]))
        summary["dacon_train"] = split
        if split < len(df):
            val_sets.append(Stage3DaconDataset(df.iloc[split:]))
            summary["dacon_val"] = len(df) - split
    if COMMA2K19_STAGE3_TRAIN_MANIFEST.is_file():
        dataset, before, after = _comma_dataset(COMMA2K19_STAGE3_TRAIN_MANIFEST, STAGE3_TRAIN_TEMPORAL_STRIDE)
        train_sets.append(dataset)
        summary["comma_train_before"] = before
        summary["comma_train_after"] = after
    if COMMA2K19_STAGE3_VAL_MANIFEST.is_file():
        dataset, before, after = _comma_dataset(COMMA2K19_STAGE3_VAL_MANIFEST, STAGE3_VAL_TEMPORAL_STRIDE)
        val_sets.append(dataset)
        summary["comma_val_before"] = before
        summary["comma_val_after"] = after
    if not train_sets:
        raise FileNotFoundError("No Stage3 training data found.")
    return ConcatDataset(train_sets), ConcatDataset(val_sets) if val_sets else None, summary


def _print_dataset_summary(train_dataset, val_dataset, summary: dict) -> None:
    print("=== Stage 3 Dataset ===")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset) if val_dataset else 0}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Train temporal stride: {STAGE3_TRAIN_TEMPORAL_STRIDE}")
    print(f"Validation temporal stride: {STAGE3_VAL_TEMPORAL_STRIDE}")
    print(f"DACON train samples: {summary['dacon_train']}")
    print(f"comma2k19 train samples: {summary['comma_train_after']} / {summary['comma_train_before']}")
    print(f"DACON val samples: {summary['dacon_val']}")
    print(f"comma2k19 val samples: {summary['comma_val_after']} / {summary['comma_val_before']}")


def _class_weights(name: str):
    values = STAGE3_CLASS_WEIGHTS.get(name)
    return torch.tensor(values, dtype=torch.float32, device=DEVICE) if values is not None else None


def _loss(accel, steer, batch, accel_weight=None, steer_weight=None):
    loss_accel = nn.functional.cross_entropy(accel, batch["accel_label"].to(DEVICE), weight=accel_weight)
    loss_steer = nn.functional.cross_entropy(steer, batch["steer_label"].to(DEVICE), weight=steer_weight)
    total = (STAGE3_LOSS_WEIGHTS["accel"] * loss_accel) + (STAGE3_LOSS_WEIGHTS["steer"] * loss_steer)
    return total, loss_accel.detach(), loss_steer.detach()


def _validate(model, loader):
    model.eval()
    accel_pred, accel_target, steer_pred, steer_target = [], [], [], []
    with torch.inference_mode():
        for batch in loader:
            accel, steer = model(batch["video"].to(DEVICE, non_blocking=True))
            accel_pred.extend(accel.argmax(1).cpu().tolist())
            steer_pred.extend(steer.argmax(1).cpu().tolist())
            accel_target.extend(batch["accel_label"].tolist())
            steer_target.extend(batch["steer_label"].tolist())
    accel_metrics = _classification_metrics(accel_pred, accel_target, 4)
    steer_metrics = _classification_metrics(steer_pred, steer_target, 3)
    return {"accel": accel_metrics, "steer": steer_metrics, "selection": (accel_metrics["macro_f1"] + steer_metrics["macro_f1"]) / 2}


def _loader(dataset, shuffle: bool):
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=STAGE3_NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=STAGE3_NUM_WORKERS > 0,
    )


def fit_stage3():
    out = STAGE3_MODEL
    out.mkdir(parents=True, exist_ok=True)

    train_dataset, val_dataset, summary = _datasets()
    _print_dataset_summary(train_dataset, val_dataset, summary)
    train_loader = _loader(train_dataset, shuffle=True)
    val_loader = _loader(val_dataset, shuffle=False) if val_dataset else None
    model = Stage3MViT().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), 1e-4)
    accel_class_weights = _class_weights("accel")
    steer_class_weights = _class_weights("steer")
    best = -1.0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = total_accel_loss = total_steer_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} Train")
        for batch in progress:
            accel, steer = model(batch["video"].to(DEVICE, non_blocking=True))
            loss, loss_accel, loss_steer = _loss(accel, steer, batch, accel_class_weights, steer_class_weights)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.detach().cpu())
            total_accel_loss += float(loss_accel.cpu())
            total_steer_loss += float(loss_steer.cpu())
            progress.set_postfix(loss=f"{float(loss.detach().cpu()):.4f}")

        steps = max(1, len(train_loader))
        train_loss = total_loss / steps
        train_accel_loss = total_accel_loss / steps
        train_steer_loss = total_steer_loss / steps
        print(
            f"[Stage 3] Epoch {epoch + 1}/{EPOCHS} | "
            f"train_loss={train_loss:.5f} | "
            f"train_accel_loss={train_accel_loss:.5f} | "
            f"train_steer_loss={train_steer_loss:.5f}"
        )

        if val_loader is None:
            torch.save({"model": model.state_dict(), "epoch": epoch + 1, "train_loss": train_loss}, out / "best.pt")
            continue
        metrics = _validate(model, val_loader)
        print(f"epoch={epoch + 1} val_accel_accuracy={metrics['accel']['accuracy']:.5f}")
        print(f"epoch={epoch + 1} val_accel_macro_f1={metrics['accel']['macro_f1']:.5f}")
        print(f"epoch={epoch + 1} val_accel_confusion_matrix={metrics['accel']['confusion_matrix']}")
        print(f"epoch={epoch + 1} val_accel_prediction_distribution={metrics['accel']['prediction_distribution']}")
        print(f"epoch={epoch + 1} val_steer_accuracy={metrics['steer']['accuracy']:.5f}")
        print(f"epoch={epoch + 1} val_steer_macro_f1={metrics['steer']['macro_f1']:.5f}")
        print(f"epoch={epoch + 1} val_steer_confusion_matrix={metrics['steer']['confusion_matrix']}")
        print(f"epoch={epoch + 1} val_steer_prediction_distribution={metrics['steer']['prediction_distribution']}")
        if metrics["selection"] > best:
            best = metrics["selection"]
            torch.save({"model": model.state_dict(), "epoch": epoch + 1, "metrics": metrics, "train_loss": train_loss}, out / "best.pt")
