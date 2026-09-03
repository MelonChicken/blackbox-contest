from __future__ import annotations

import pandas as pd
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

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
    STAGE3_RAW,
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


def _datasets():
    train_sets, val_sets = [], []
    labels_path = STAGE3_RAW / "labels.csv"
    if labels_path.is_file():
        df = pd.read_csv(labels_path).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
        split = max(1, int(len(df) * 0.8)) if len(df) > 1 else len(df)
        train_sets.append(Stage3DaconDataset(df.iloc[:split]))
        if split < len(df):
            val_sets.append(Stage3DaconDataset(df.iloc[split:]))
    if COMMA2K19_STAGE3_TRAIN_MANIFEST.is_file():
        train_sets.append(Comma2k19Stage3Dataset(COMMA2K19_STAGE3_TRAIN_MANIFEST))
    if COMMA2K19_STAGE3_VAL_MANIFEST.is_file():
        val_sets.append(Comma2k19Stage3Dataset(COMMA2K19_STAGE3_VAL_MANIFEST))
    if not train_sets:
        raise FileNotFoundError("No Stage3 training data found.")
    return ConcatDataset(train_sets), ConcatDataset(val_sets) if val_sets else None


def _class_weights(name: str):
    values = STAGE3_CLASS_WEIGHTS.get(name)
    return torch.tensor(values, dtype=torch.float32, device=DEVICE) if values is not None else None


def _loss(accel, steer, batch):
    loss_accel = nn.functional.cross_entropy(accel, batch["accel_label"].to(DEVICE), weight=_class_weights("accel"))
    loss_steer = nn.functional.cross_entropy(steer, batch["steer_label"].to(DEVICE), weight=_class_weights("steer"))
    return (STAGE3_LOSS_WEIGHTS["accel"] * loss_accel) + (STAGE3_LOSS_WEIGHTS["steer"] * loss_steer)


def _validate(model, loader):
    model.eval()
    accel_pred, accel_target, steer_pred, steer_target = [], [], [], []
    with torch.inference_mode():
        for batch in loader:
            accel, steer = model(batch["video"].to(DEVICE))
            accel_pred.extend(accel.argmax(1).cpu().tolist())
            steer_pred.extend(steer.argmax(1).cpu().tolist())
            accel_target.extend(batch["accel_label"].tolist())
            steer_target.extend(batch["steer_label"].tolist())
    accel_metrics = _classification_metrics(accel_pred, accel_target, 4)
    steer_metrics = _classification_metrics(steer_pred, steer_target, 3)
    return {"accel": accel_metrics, "steer": steer_metrics, "selection": (accel_metrics["macro_f1"] + steer_metrics["macro_f1"]) / 2}


def fit_stage3():
    out = STAGE3_MODEL
    out.mkdir(parents=True, exist_ok=True)

    train_dataset, val_dataset = _datasets()
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False) if val_dataset else None
    model = Stage3MViT().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), 1e-4)
    best = -1.0

    for epoch in range(EPOCHS):
        model.train()
        for batch in train_loader:
            accel, steer = model(batch["video"].to(DEVICE))
            loss = _loss(accel, steer, batch)
            opt.zero_grad()
            loss.backward()
            opt.step()

        if val_loader is None:
            torch.save({"model": model.state_dict()}, out / "best.pt")
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
            torch.save({"model": model.state_dict(), "metrics": metrics}, out / "best.pt")
