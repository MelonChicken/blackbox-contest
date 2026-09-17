from __future__ import annotations

from datetime import datetime

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

cv2.setNumThreads(0)

from src.config import (
    BATCH_SIZE,
    COMMA2K19_STAGE3_TRAIN_MANIFEST,
    COMMA2K19_STAGE3_VAL_MANIFEST,
    DEVICE,
    SEED,
    STAGE3_ARCH,
    STAGE3_CLASS_WEIGHTS,
    STAGE3_DATASET_MODE,
    STAGE3_EPOCHS,
    STAGE3_LOSS_WEIGHTS,
    STAGE3_MVIT_BACKBONE_LR,
    STAGE3_MVIT_PRETRAINED,
    STAGE3_MVIT_PRETRAINED_WEIGHTS,
    STAGE3_MVIT_PREPROCESS,
    STAGE3_MODEL,
    STAGE3_NUM_WORKERS,
    STAGE3_PREFETCH_FACTOR,
    STAGE3_SAMPLE_PROFILE,
    STAGE3_HEAD_LR,
    STAGE3_TRAIN_SAMPLE_LIMIT,
    STAGE3_TRAIN_TEMPORAL_STRIDE,
    STAGE3_VAL_SAMPLE_LIMIT,
    STAGE3_VAL_TEMPORAL_STRIDE,
)
from src.datasets.comma2k19_stage3 import ACCEL_TO_ID, STEER_TO_ID, Comma2k19Stage3Dataset
from src.models import Stage3MViT
from src.tools.stage3_comma_manifest import active_manifest_path
from src.utils import set_seed

set_seed(SEED)
SOURCE_REFS = {"comma2k19": 0.75}


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
    key = ["route_id", "segment_id"] if {"route_id", "segment_id"}.issubset(df.columns) else "segment_id" if "segment_id" in df.columns else "video_path"
    parts = [part.iloc[::stride] for _, part in df.groupby(key, sort=False)]
    return pd.concat(parts, ignore_index=True) if parts else df.reset_index(drop=True)


def _balanced_limit(df: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    if not limit or limit <= 0 or len(df) <= limit:
        return df.reset_index(drop=True)
    key = "segment_id" if "segment_id" in df.columns else None
    if key is None:
        return df.sample(n=limit, random_state=SEED).sort_index().reset_index(drop=True)
    per_group = max(1, limit // df[key].nunique() + 1)
    out = df.groupby(key, group_keys=False).sample(frac=1.0, random_state=SEED).groupby(key, group_keys=False).head(per_group)
    return out.head(limit).sort_index().reset_index(drop=True)


def _limited_dataset(dataset, stride: int, limit: int | None):
    before = len(dataset)
    dataset.df = _balanced_limit(_stride_manifest(dataset.df, stride), limit)
    return dataset, before, len(dataset)


def _label_id(value, mapping: dict[str, int]) -> int:
    return int(value) if not isinstance(value, str) else mapping[value]


def _comma_manifest_path(split: str):
    path = active_manifest_path(split)
    if path.is_file() and path.stat().st_size > 0:
        return path
    fallback = COMMA2K19_STAGE3_TRAIN_MANIFEST if split == "train" else COMMA2K19_STAGE3_VAL_MANIFEST
    return fallback if fallback.is_file() and fallback.stat().st_size > 0 else path


def _datasets():
    if STAGE3_DATASET_MODE != "comma_only":
        raise ValueError(f"Stage3 supports only comma_only, got: {STAGE3_DATASET_MODE}")
    train_manifest = _comma_manifest_path("train")
    if not train_manifest.is_file():
        raise FileNotFoundError(f"missing comma2k19 Stage3 train manifest: {train_manifest}")
    train, before, after = _limited_dataset(Comma2k19Stage3Dataset(train_manifest), STAGE3_TRAIN_TEMPORAL_STRIDE, STAGE3_TRAIN_SAMPLE_LIMIT)
    summary = {"train_sources": {"comma2k19": len(train)}, "val_sources": {}, "comma_train_before": before, "comma_train_after": after}
    val = {}
    val_manifest = _comma_manifest_path("val")
    if val_manifest.is_file():
        ds, before, after = _limited_dataset(Comma2k19Stage3Dataset(val_manifest), STAGE3_VAL_TEMPORAL_STRIDE, STAGE3_VAL_SAMPLE_LIMIT)
        val["comma2k19"] = ds
        summary.update(comma_val_before=before, comma_val_after=after)
        summary["val_sources"]["comma2k19"] = len(ds)
    return train, val, summary


def _build_stage3_model(pretrained: bool = True):
    return Stage3MViT(pretrained=pretrained)


def _labels_from_dataset(dataset):
    accel = [_label_id(v, ACCEL_TO_ID) for v in dataset.df.accel_label.tolist()]
    steer = [_label_id(v, STEER_TO_ID) for v in dataset.df.steer_label.tolist()]
    return accel, steer


def _count_names(values, names):
    counts = pd.Series(values).value_counts().sort_index() if values else pd.Series(dtype=int)
    total = max(1, len(values))
    return {names[i]: f"{int(counts.get(i, 0))} ({int(counts.get(i, 0)) / total:.4f})" for i in names}


def _print_one_distribution(name: str, dataset) -> None:
    accel_names = {v: k for k, v in ACCEL_TO_ID.items()}
    steer_names = {v: k for k, v in STEER_TO_ID.items()}
    accel, steer = _labels_from_dataset(dataset)
    print(f"[{name}]")
    print(f"samples: {len(dataset)}")
    print(f"accel: {_count_names(accel, accel_names)}")
    print(f"steer: {_count_names(steer, steer_names)}")
    if isinstance(dataset, Comma2k19Stage3Dataset) and dataset.cache_root is not None:
        key = ["route_id", "segment_id"] if {"route_id", "segment_id"}.issubset(dataset.df.columns) else "video_path"
        rows = [part.iloc[0] for _, part in dataset.df.groupby(key, sort=False)]
        cached = sum(dataset._cache_dir(row) is not None for row in rows)
        print(f"frame cache: {cached}/{len(rows)} segments cached")
        if rows:
            row = rows[0]
            video_path = dataset._video_path(str(row.video_path))
            expected = dataset._expected_cache_dir(row)
            print(f"frame cache sample: video={video_path} cache={expected} frames_csv={(expected / 'frames.csv').is_file() if expected else False}")


def _print_dataset_summary(train_dataset, val_datasets: dict[str, object], summary: dict) -> None:
    print("=== Stage 3 Dataset ===")
    print(f"Dataset mode: {STAGE3_DATASET_MODE}")
    print(f"Sample profile: {STAGE3_SAMPLE_PROFILE}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {sum(len(v) for v in val_datasets.values())}")
    print(f"Architecture: {STAGE3_ARCH}")
    print(f"MViT pretrained: {STAGE3_MVIT_PRETRAINED} ({STAGE3_MVIT_PRETRAINED_WEIGHTS})")
    print(f"MViT preprocessing: {STAGE3_MVIT_PREPROCESS} mean=[0.45, 0.45, 0.45] std=[0.225, 0.225, 0.225] resize=224 crop=224")
    print(f"Batch size: {BATCH_SIZE}")
    _print_one_distribution("comma2k19 train", train_dataset)
    for source, ds in val_datasets.items():
        _print_one_distribution(f"{source} val", ds)


def _class_weights(name: str):
    values = STAGE3_CLASS_WEIGHTS.get(name)
    return torch.tensor(values, dtype=torch.float32, device=DEVICE) if values is not None else None


def _model_outputs(model, batch):
    return model(batch["video"].to(DEVICE, non_blocking=True))


def _loss(accel, steer, batch, accel_weight=None, steer_weight=None):
    loss_accel = nn.functional.cross_entropy(accel, batch["accel_label"].to(DEVICE), weight=accel_weight)
    loss_steer = nn.functional.cross_entropy(steer, batch["steer_label"].to(DEVICE), weight=steer_weight)
    total = (STAGE3_LOSS_WEIGHTS["accel"] * loss_accel) + (STAGE3_LOSS_WEIGHTS["steer"] * loss_steer)
    return total, loss_accel.detach(), loss_steer.detach()


def _selection_score(accel_f1: float, steer_f1: float) -> float:
    accel_w = float(STAGE3_LOSS_WEIGHTS.get("accel", 1.0))
    steer_w = float(STAGE3_LOSS_WEIGHTS.get("steer", 1.0))
    return ((accel_w * accel_f1) + (steer_w * steer_f1)) / max(1e-12, accel_w + steer_w)


def _validate(model, loader, accel_weight=None, steer_weight=None):
    model.eval()
    accel_pred, accel_target, steer_pred, steer_target = [], [], [], []
    total_loss = 0.0
    with torch.inference_mode():
        for batch in loader:
            accel, steer = _model_outputs(model, batch)
            loss, _, _ = _loss(accel, steer, batch, accel_weight, steer_weight)
            total_loss += float(loss.cpu())
            accel_pred.extend(accel.argmax(1).cpu().tolist())
            steer_pred.extend(steer.argmax(1).cpu().tolist())
            accel_target.extend(batch["accel_label"].tolist())
            steer_target.extend(batch["steer_label"].tolist())
    accel_metrics = _classification_metrics(accel_pred, accel_target, 4)
    steer_metrics = _classification_metrics(steer_pred, steer_target, 3)
    return {"loss": total_loss / max(1, len(loader)), "accel": accel_metrics, "steer": steer_metrics, "selection": _selection_score(accel_metrics["macro_f1"], steer_metrics["macro_f1"])}


def _validate_all(model, loaders: dict[str, DataLoader], accel_weight=None, steer_weight=None) -> dict:
    metrics = {source: _validate(model, loader, accel_weight, steer_weight) for source, loader in loaders.items()}
    only = next(iter(metrics.values())) if metrics else {"selection": float("nan")}
    metrics["overall"] = only
    return metrics


def _stage3_collate(batch: list[dict]) -> dict:
    out = {
        "accel_label": torch.tensor([item["accel_label"] for item in batch], dtype=torch.long),
        "steer_label": torch.tensor([item["steer_label"] for item in batch], dtype=torch.long),
    }
    if all("video" in item for item in batch):
        out["video"] = torch.stack([item["video"] for item in batch])
    return out


def _loader(dataset, shuffle: bool):
    kwargs = {
        "batch_size": BATCH_SIZE,
        "shuffle": shuffle,
        "num_workers": STAGE3_NUM_WORKERS,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": STAGE3_NUM_WORKERS > 0,
        "collate_fn": _stage3_collate,
    }
    if STAGE3_NUM_WORKERS > 0:
        kwargs["prefetch_factor"] = STAGE3_PREFETCH_FACTOR
    return DataLoader(dataset, **kwargs)


def _param_count(model, trainable: bool) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad is trainable)


def _history_keys() -> list[str]:
    return [
        "epoch", "arch", "dataset_mode", "batch_size", "selection_metric_name", "selection_metric",
        "train_loss", "val_loss", "train_accel_loss", "train_steer_loss",
        "train_accel_accuracy", "val_accel_accuracy", "train_accel_macro_f1", "val_accel_macro_f1",
        "train_steer_accuracy", "val_steer_accuracy", "train_steer_macro_f1", "val_steer_macro_f1",
        "selection", "comma_val_accel_accuracy", "comma_val_accel_macro_f1", "comma_val_steer_accuracy",
        "comma_val_steer_macro_f1", "comma_selection",
    ]


def _append_history(history: dict, epoch: int, train_loss: float, train_accel_loss: float, train_steer_loss: float, train_metrics: dict, metrics: dict, selection_metric_name: str, selection_metric: float) -> None:
    overall = metrics["overall"]
    comma = metrics.get("comma2k19")
    row = {
        "epoch": epoch,
        "arch": STAGE3_ARCH,
        "dataset_mode": STAGE3_DATASET_MODE,
        "batch_size": BATCH_SIZE,
        "selection_metric_name": selection_metric_name,
        "selection_metric": selection_metric,
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


def _save_history(history: dict, run_id: str) -> tuple:
    history_dir = STAGE3_MODEL / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    history_path = history_dir / f"{run_id}_history.csv"
    loss_path = history_dir / f"{run_id}_loss.png"
    metrics_path = history_dir / f"{run_id}_metrics.png"
    df = pd.DataFrame(history)
    df.to_csv(history_path, index=False)
    plt.figure()
    for name in ("train_loss", "val_loss", "train_accel_loss", "train_steer_loss"):
        plt.plot(df["epoch"], df[name], marker="o", label=name)
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title(f"Stage3 Training Loss - {STAGE3_ARCH}"); plt.legend(); plt.grid(True)
    plt.savefig(loss_path, bbox_inches="tight"); plt.close()
    plt.figure()
    for name in ("train_accel_macro_f1", "val_accel_macro_f1", "train_steer_macro_f1", "val_steer_macro_f1", "selection", "comma_selection"):
        if name in df and not df[name].isna().all():
            plt.plot(df["epoch"], df[name], marker="o", label=name)
    plt.xlabel("Epoch"); plt.ylabel("Score"); plt.ylim(0, 1); plt.title(f"Stage3 Validation Metrics - {STAGE3_ARCH}"); plt.legend(); plt.grid(True)
    plt.savefig(metrics_path, bbox_inches="tight"); plt.close()
    return history_path, loss_path, metrics_path


def _checkpoint_payload(model, epoch: int, train_loss: float, metrics=None, history=None) -> dict:
    selection_metric = float(metrics["overall"]["selection"]) if metrics else float("nan")
    payload = {
        "model": model.state_dict(),
        "arch": STAGE3_ARCH,
        "epoch": epoch,
        "train_loss": train_loss,
        "dataset_mode": STAGE3_DATASET_MODE,
        "selection_metric_name": "overall.selection",
        "selection_metric": selection_metric,
        "sampling_hz": STAGE3_SAMPLING_HZ,
        "sampling_policy": STAGE3_SAMPLING_POLICY,
        "num_frames": STAGE3_NUM_FRAMES,
        "past_frames": STAGE3_PAST_FRAMES,
        "future_frames": STAGE3_FUTURE_FRAMES,
        "boundary_policy": STAGE3_BOUNDARY_POLICY,
        "output_hz": STAGE3_OUTPUT_HZ,
        "image_size": 224,
    }
    if metrics is not None:
        payload["metrics"] = metrics
    if history is not None:
        payload["history"] = history
    if hasattr(model, "model_config"):
        payload["model_config"] = model.model_config()
    return payload


def _print_metrics_table(metrics: dict, prefix: str = "") -> None:
    print(prefix + "Source      Accel F1   Steer F1   Weighted")
    print(prefix + "-------------------------------------------")
    for name, label in (("comma2k19", "comma val"), ("overall", "overall")):
        if name in metrics:
            m = metrics[name]
            print(prefix + f"{label:<11} {m['accel']['macro_f1']:.5f}    {m['steer']['macro_f1']:.5f}    {m['selection']:.5f}")


def fit_stage3():
    out = STAGE3_MODEL
    out.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    history = {k: [] for k in _history_keys()}
    history_path = loss_path = metrics_path = None
    train_dataset, val_datasets, summary = _datasets()
    _print_dataset_summary(train_dataset, val_datasets, summary)
    train_loader = _loader(train_dataset, shuffle=True)
    val_loaders = {name: _loader(ds, shuffle=False) for name, ds in val_datasets.items()}
    model = _build_stage3_model(pretrained=STAGE3_MVIT_PRETRAINED).to(DEVICE)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Total trainable parameters: {_param_count(model, True)}")
    print(f"Frozen parameters: {_param_count(model, False)}")
    opt = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": STAGE3_MVIT_BACKBONE_LR},
        {"params": list(model.accel.parameters()) + list(model.steer.parameters()), "lr": STAGE3_HEAD_LR},
    ])
    print(f"Optimizer LR: backbone={STAGE3_MVIT_BACKBONE_LR} head={STAGE3_HEAD_LR}")
    accel_class_weights = _class_weights("accel")
    steer_class_weights = _class_weights("steer")
    best = -1.0
    best_metrics = None
    best_epoch = 0

    for epoch in range(STAGE3_EPOCHS):
        model.train()
        total_loss = total_accel_loss = total_steer_loss = 0.0
        train_accel_pred, train_accel_target, train_steer_pred, train_steer_target = [], [], [], []
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{STAGE3_EPOCHS} Train")
        for batch in progress:
            accel, steer = _model_outputs(model, batch)
            loss, loss_accel, loss_steer = _loss(accel, steer, batch, accel_class_weights, steer_class_weights)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += float(loss.detach().cpu()); total_accel_loss += float(loss_accel.cpu()); total_steer_loss += float(loss_steer.cpu())
            train_accel_pred.extend(accel.argmax(1).detach().cpu().tolist())
            train_steer_pred.extend(steer.argmax(1).detach().cpu().tolist())
            train_accel_target.extend(batch["accel_label"].tolist())
            train_steer_target.extend(batch["steer_label"].tolist())
            progress.set_postfix(loss=f"{float(loss.detach().cpu()):.4f}")
        steps = max(1, len(train_loader))
        train_loss = total_loss / steps; train_accel_loss = total_accel_loss / steps; train_steer_loss = total_steer_loss / steps
        train_metrics = {
            "accel": _classification_metrics(train_accel_pred, train_accel_target, 4),
            "steer": _classification_metrics(train_steer_pred, train_steer_target, 3),
        }
        train_metrics["selection"] = _selection_score(train_metrics["accel"]["macro_f1"], train_metrics["steer"]["macro_f1"])
        print(f"[Stage 3] Epoch {epoch + 1}/{STAGE3_EPOCHS} | train_loss={train_loss:.5f} | train_accel_loss={train_accel_loss:.5f} | train_steer_loss={train_steer_loss:.5f}")
        metrics = _validate_all(model, val_loaders, accel_class_weights, steer_class_weights) if val_loaders else {"overall": {"loss": float("nan"), "accel": {"accuracy": float("nan"), "macro_f1": float("nan")}, "steer": {"accuracy": float("nan"), "macro_f1": float("nan")}, "selection": float("nan")}}
        _print_metrics_table(metrics, prefix=f"epoch={epoch + 1} ")
        select = metrics["overall"]["selection"]
        print(f"selection_metric_name=overall.selection selection_metric={select:.5f}")
        _append_history(history, epoch + 1, train_loss, train_accel_loss, train_steer_loss, train_metrics, metrics, "overall.selection", select)
        history_path, loss_path, metrics_path = _save_history(history, run_id)
        if select > best:
            best = select; best_epoch = epoch + 1; best_metrics = metrics
            torch.save(_checkpoint_payload(model, epoch + 1, train_loss, metrics, history), out / "best.pt")

    print(f"Stage3 history saved:\n{history_path}")
    print(f"Loss plot:\n{loss_path}")
    print(f"Metrics plot:\n{metrics_path}")
    if best_metrics:
        print(f"Best epoch: {best_epoch}")
        _print_metrics_table(best_metrics)
        for source, ref in SOURCE_REFS.items():
            if source in best_metrics:
                print(f"{source} selection delta vs reference: {best_metrics[source]['selection'] - ref:+.5f}")
        print(f"selection={best:.5f}")

