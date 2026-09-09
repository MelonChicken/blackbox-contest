from __future__ import annotations

from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    KITTI_STAGE3_TRAIN_MANIFEST,
    KITTI_STAGE3_VAL_MANIFEST,
    SEED,
    STAGE3_ARCH,
    STAGE3_CLASS_WEIGHTS,
    STAGE3_DATASET,
    STAGE3_EPOCHS,
    STAGE3_KITTI_TRAIN_SAMPLE_LIMIT,
    STAGE3_KITTI_VAL_SAMPLE_LIMIT,
    STAGE3_LOSS_WEIGHTS,
    STAGE3_MODEL,
    STAGE3_NUM_WORKERS,
    STAGE3_RAW,
    STAGE3_TARTANVO_FEATURE,
    STAGE3_TARTANVO_USE_FEATURE_CACHE,
    STAGE3_TRAIN_SAMPLE_LIMIT,
    STAGE3_TRAIN_TEMPORAL_STRIDE,
    STAGE3_VAL_SAMPLE_LIMIT,
    STAGE3_VAL_TEMPORAL_STRIDE,
)
from src.datasets.comma2k19_stage3 import ACCEL_TO_ID, STEER_TO_ID, Comma2k19Stage3Dataset, Stage3DaconDataset
from src.datasets.kitti_stage3 import KittiStage3Dataset
from src.datasets.stage3_tartanvo_pose import Stage3TartanFeatureDataset
from src.models import Stage3MViT, Stage3ResNetGRU, Stage3TartanVOGRU
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
    key = "segment_id" if "segment_id" in df.columns else "sequence_id" if "sequence_id" in df.columns else "video_path"
    parts = [part.iloc[::stride] for _, part in df.groupby(key, sort=False)]
    return pd.concat(parts, ignore_index=True) if parts else df.reset_index(drop=True)


def _balanced_limit(df: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    if not limit or limit <= 0 or len(df) <= limit:
        return df.reset_index(drop=True)
    key = "sequence_id" if "sequence_id" in df.columns else "segment_id" if "segment_id" in df.columns else None
    if key is None:
        return df.sample(n=limit, random_state=SEED).sort_index().reset_index(drop=True)
    per_group = max(1, limit // df[key].nunique() + 1)
    out = df.groupby(key, group_keys=False).sample(frac=1.0, random_state=SEED).groupby(key, group_keys=False).head(per_group)
    return out.head(limit).sort_index().reset_index(drop=True)


def _limited_dataset(dataset, stride: int, limit: int | None):
    before = len(dataset)
    dataset.df = _balanced_limit(_stride_manifest(dataset.df, stride), limit)
    return dataset, before, len(dataset)


def _build_stage3_model(pretrained: bool = True):
    if STAGE3_ARCH == "mvit":
        return Stage3MViT(pretrained=pretrained)
    if STAGE3_ARCH == "resnet18_gru":
        return Stage3ResNetGRU(pretrained=pretrained)
    if STAGE3_ARCH == "tartanvo_gru":
        return Stage3TartanVOGRU(load_pretrained=pretrained)
    raise ValueError(f"Unknown STAGE3_ARCH: {STAGE3_ARCH}")


def _datasets():
    if STAGE3_ARCH == "tartanvo_gru" and STAGE3_TARTANVO_USE_FEATURE_CACHE:
        return (
            Stage3TartanFeatureDataset("train", STAGE3_TARTANVO_FEATURE, dataset=STAGE3_DATASET),
            Stage3TartanFeatureDataset("val", STAGE3_TARTANVO_FEATURE, dataset=STAGE3_DATASET),
            {"cache": True},
        )
    train_sets, val_sets = [], []
    summary = {"dacon_train": 0, "dacon_val": 0}
    labels_path = STAGE3_RAW / "labels.csv"
    if STAGE3_DATASET == "comma2k19" and labels_path.is_file():
        df = pd.read_csv(labels_path).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
        split = max(1, int(len(df) * 0.8)) if len(df) > 1 else len(df)
        train_sets.append(Stage3DaconDataset(df.iloc[:split]))
        summary["dacon_train"] = split
        if split < len(df):
            val_sets.append(Stage3DaconDataset(df.iloc[split:]))
            summary["dacon_val"] = len(df) - split
    if STAGE3_DATASET == "kitti":
        if KITTI_STAGE3_TRAIN_MANIFEST.is_file():
            ds, before, after = _limited_dataset(KittiStage3Dataset(KITTI_STAGE3_TRAIN_MANIFEST), STAGE3_TRAIN_TEMPORAL_STRIDE, STAGE3_KITTI_TRAIN_SAMPLE_LIMIT)
            train_sets.append(ds); summary.update(kitti_train_before=before, kitti_train_after=after)
        if KITTI_STAGE3_VAL_MANIFEST.is_file():
            ds, before, after = _limited_dataset(KittiStage3Dataset(KITTI_STAGE3_VAL_MANIFEST), STAGE3_VAL_TEMPORAL_STRIDE, STAGE3_KITTI_VAL_SAMPLE_LIMIT)
            val_sets.append(ds); summary.update(kitti_val_before=before, kitti_val_after=after)
    elif STAGE3_DATASET == "comma2k19":
        if COMMA2K19_STAGE3_TRAIN_MANIFEST.is_file():
            ds, before, after = _limited_dataset(Comma2k19Stage3Dataset(COMMA2K19_STAGE3_TRAIN_MANIFEST), STAGE3_TRAIN_TEMPORAL_STRIDE, STAGE3_TRAIN_SAMPLE_LIMIT)
            train_sets.append(ds); summary.update(comma_train_before=before, comma_train_after=after)
        if COMMA2K19_STAGE3_VAL_MANIFEST.is_file():
            ds, before, after = _limited_dataset(Comma2k19Stage3Dataset(COMMA2K19_STAGE3_VAL_MANIFEST), STAGE3_VAL_TEMPORAL_STRIDE, STAGE3_VAL_SAMPLE_LIMIT)
            val_sets.append(ds); summary.update(comma_val_before=before, comma_val_after=after)
    else:
        raise ValueError(f"Unknown STAGE3_DATASET: {STAGE3_DATASET}")
    if not train_sets:
        raise FileNotFoundError("No Stage3 training data found.")
    return ConcatDataset(train_sets), ConcatDataset(val_sets) if val_sets else None, summary


def _labels_from_dataset(dataset):
    if hasattr(dataset, "df"):
        seq = dataset.df.sequence_id.astype(str).nunique() if "sequence_id" in dataset.df.columns else None
        return dataset.df.accel_label.astype(int).tolist(), dataset.df.steer_label.astype(int).tolist(), seq
    accel, steer, seqs = [], [], set()
    for ds in getattr(dataset, "datasets", []):
        a, s, n = _labels_from_dataset(ds)
        accel.extend(a); steer.extend(s)
        if n is not None and hasattr(ds, "df"):
            seqs.update(ds.df.sequence_id.astype(str).unique())
    return accel, steer, len(seqs) if seqs else None


def _count_names(values, names):
    counts = pd.Series(values).value_counts().sort_index() if values else pd.Series(dtype=int)
    total = max(1, len(values))
    return {names[i]: f"{int(counts.get(i, 0))} ({int(counts.get(i, 0)) / total:.4f})" for i in names}


def _print_distribution(train_dataset, val_dataset) -> None:
    accel_names = {v: k for k, v in ACCEL_TO_ID.items()}
    steer_names = {v: k for k, v in STEER_TO_ID.items()}
    for name, ds in (("train", train_dataset), ("val", val_dataset)):
        if ds is None:
            continue
        accel, steer, seq_count = _labels_from_dataset(ds)
        if seq_count is not None:
            print(f"{name} sequence count: {seq_count}")
        print(f"{name} accel distribution: {_count_names(accel, accel_names)}")
        print(f"{name} steer distribution: {_count_names(steer, steer_names)}")


def _print_dataset_summary(train_dataset, val_dataset, summary: dict) -> None:
    print("=== Stage 3 Dataset ===")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset) if val_dataset else 0}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Architecture: {STAGE3_ARCH}")
    print(f"Dataset source: {STAGE3_DATASET}")
    if summary.get("cache"):
        print("TartanVO feature cache: enabled")
        print("feature: [15, 1536]")
        _print_distribution(train_dataset, val_dataset)
        return
    print(f"Train temporal stride: {STAGE3_TRAIN_TEMPORAL_STRIDE}")
    print(f"Validation temporal stride: {STAGE3_VAL_TEMPORAL_STRIDE}")
    for key in sorted(summary):
        print(f"{key}: {summary[key]}")
    _print_distribution(train_dataset, val_dataset)


def _class_weights(name: str):
    values = STAGE3_CLASS_WEIGHTS.get(name)
    return torch.tensor(values, dtype=torch.float32, device=DEVICE) if values is not None else None


def _model_outputs(model, batch):
    if "feature" in batch:
        return model.forward_feature(batch["feature"].to(DEVICE, non_blocking=True))
    if "pose" in batch:
        return model.forward_pose(batch["pose"].to(DEVICE, non_blocking=True))
    video = batch["video"].to(DEVICE, non_blocking=True)
    intrinsics = batch.get("intrinsics")
    if intrinsics is not None and hasattr(model, "feature_sequence"):
        return model(video, intrinsics=intrinsics.to(DEVICE, non_blocking=True))
    return model(video)


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
            accel, steer = _model_outputs(model, batch)
            accel_pred.extend(accel.argmax(1).cpu().tolist())
            steer_pred.extend(steer.argmax(1).cpu().tolist())
            accel_target.extend(batch["accel_label"].tolist())
            steer_target.extend(batch["steer_label"].tolist())
    accel_metrics = _classification_metrics(accel_pred, accel_target, 4)
    steer_metrics = _classification_metrics(steer_pred, steer_target, 3)
    return {"accel": accel_metrics, "steer": steer_metrics, "selection": (accel_metrics["macro_f1"] + steer_metrics["macro_f1"]) / 2}


def _param_count(model, trainable: bool) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad is trainable)


def _save_history(history: dict, run_id: str) -> tuple:
    history_dir = STAGE3_MODEL / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    history_path = history_dir / f"{run_id}_history.csv"
    loss_path = history_dir / f"{run_id}_loss.png"
    metrics_path = history_dir / f"{run_id}_metrics.png"
    df = pd.DataFrame(history)
    df.to_csv(history_path, index=False)
    plt.figure()
    for name in ("train_loss", "train_accel_loss", "train_steer_loss"):
        plt.plot(df["epoch"], df[name], marker="o", label=name)
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title(f"Stage3 Training Loss - {STAGE3_ARCH}"); plt.legend(); plt.grid(True)
    plt.savefig(loss_path, bbox_inches="tight"); plt.close()
    plt.figure()
    for name in ("val_accel_macro_f1", "val_steer_macro_f1", "selection"):
        plt.plot(df["epoch"], df[name], marker="o", label=name)
    plt.xlabel("Epoch"); plt.ylabel("Score"); plt.ylim(0, 1); plt.title(f"Stage3 Validation Metrics - {STAGE3_ARCH}"); plt.legend(); plt.grid(True)
    plt.savefig(metrics_path, bbox_inches="tight"); plt.close()
    return history_path, loss_path, metrics_path


def _checkpoint_payload(model, epoch: int, train_loss: float, metrics=None, history=None) -> dict:
    payload = {"model": model.state_dict(), "arch": STAGE3_ARCH, "epoch": epoch, "train_loss": train_loss}
    if metrics is not None:
        payload["metrics"] = metrics
    if history is not None:
        payload["history"] = history
    if hasattr(model, "model_config"):
        payload["model_config"] = model.model_config()
    return payload


def _loader(dataset, shuffle: bool):
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=STAGE3_NUM_WORKERS, pin_memory=torch.cuda.is_available(), persistent_workers=STAGE3_NUM_WORKERS > 0)


def fit_stage3():
    out = STAGE3_MODEL
    out.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    history = {k: [] for k in ("epoch", "arch", "batch_size", "train_sample_limit", "val_sample_limit", "train_loss", "train_accel_loss", "train_steer_loss", "val_accel_accuracy", "val_accel_macro_f1", "val_steer_accuracy", "val_steer_macro_f1", "selection")}
    history_path = loss_path = metrics_path = None
    train_dataset, val_dataset, summary = _datasets()
    _print_dataset_summary(train_dataset, val_dataset, summary)
    train_loader = _loader(train_dataset, shuffle=True)
    val_loader = _loader(val_dataset, shuffle=False) if val_dataset else None
    model = _build_stage3_model(pretrained=True).to(DEVICE)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("Stage3 model has no trainable parameters.")
    print(f"Trainable parameters: {_param_count(model, True)}")
    print(f"Frozen parameters: {_param_count(model, False)}")
    if STAGE3_ARCH == "tartanvo_gru":
        print(f"TartanVO feature mode: {STAGE3_TARTANVO_FEATURE}")
        print(f"TartanVO pretrained loaded: {model.tartanvo.pretrained_loaded}")
    opt = torch.optim.AdamW(trainable_params, 1e-4)
    accel_class_weights = _class_weights("accel")
    steer_class_weights = _class_weights("steer")
    best = -1.0

    for epoch in range(STAGE3_EPOCHS):
        model.train()
        total_loss = total_accel_loss = total_steer_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{STAGE3_EPOCHS} Train")
        for batch in progress:
            accel, steer = _model_outputs(model, batch)
            loss, loss_accel, loss_steer = _loss(accel, steer, batch, accel_class_weights, steer_class_weights)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += float(loss.detach().cpu()); total_accel_loss += float(loss_accel.cpu()); total_steer_loss += float(loss_steer.cpu())
            progress.set_postfix(loss=f"{float(loss.detach().cpu()):.4f}")
        steps = max(1, len(train_loader))
        train_loss = total_loss / steps; train_accel_loss = total_accel_loss / steps; train_steer_loss = total_steer_loss / steps
        print(f"[Stage 3] Epoch {epoch + 1}/{STAGE3_EPOCHS} | train_loss={train_loss:.5f} | train_accel_loss={train_accel_loss:.5f} | train_steer_loss={train_steer_loss:.5f}")
        metrics = _validate(model, val_loader) if val_loader is not None else {"accel": {"accuracy": float("nan"), "macro_f1": float("nan"), "confusion_matrix": [], "prediction_distribution": []}, "steer": {"accuracy": float("nan"), "macro_f1": float("nan"), "confusion_matrix": [], "prediction_distribution": []}, "selection": float("nan")}
        print(f"epoch={epoch + 1} val_accel_accuracy={metrics['accel']['accuracy']:.5f}")
        print(f"epoch={epoch + 1} val_accel_macro_f1={metrics['accel']['macro_f1']:.5f}")
        print(f"epoch={epoch + 1} val_accel_confusion_matrix={metrics['accel']['confusion_matrix']}")
        print(f"epoch={epoch + 1} val_accel_prediction_distribution={metrics['accel']['prediction_distribution']}")
        print(f"epoch={epoch + 1} val_steer_accuracy={metrics['steer']['accuracy']:.5f}")
        print(f"epoch={epoch + 1} val_steer_macro_f1={metrics['steer']['macro_f1']:.5f}")
        print(f"epoch={epoch + 1} val_steer_confusion_matrix={metrics['steer']['confusion_matrix']}")
        print(f"epoch={epoch + 1} val_steer_prediction_distribution={metrics['steer']['prediction_distribution']}")
        print(f"epoch={epoch + 1} selection={metrics['selection']:.5f}")
        for k, v in {"epoch": epoch + 1, "arch": STAGE3_ARCH, "batch_size": BATCH_SIZE, "train_sample_limit": STAGE3_TRAIN_SAMPLE_LIMIT, "val_sample_limit": STAGE3_VAL_SAMPLE_LIMIT, "train_loss": train_loss, "train_accel_loss": train_accel_loss, "train_steer_loss": train_steer_loss, "val_accel_accuracy": metrics["accel"]["accuracy"], "val_accel_macro_f1": metrics["accel"]["macro_f1"], "val_steer_accuracy": metrics["steer"]["accuracy"], "val_steer_macro_f1": metrics["steer"]["macro_f1"], "selection": metrics["selection"]}.items():
            history[k].append(v)
        history_path, loss_path, metrics_path = _save_history(history, run_id)
        if val_loader is None or metrics["selection"] > best:
            best = metrics["selection"]
            torch.save(_checkpoint_payload(model, epoch + 1, train_loss, metrics, history), out / "best.pt")
    print(f"Stage3 history saved:\n{history_path}")
    print(f"Loss plot:\n{loss_path}")
    print(f"Metrics plot:\n{metrics_path}")
