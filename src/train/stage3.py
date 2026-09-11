from __future__ import annotations

from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Sampler
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
    STAGE3_COMMA_TRAIN_SAMPLE_LIMIT,
    STAGE3_COMMA_VAL_SAMPLE_LIMIT,
    STAGE3_DATASET,
    STAGE3_EPOCHS,
    STAGE3_KITTI_TRAIN_SAMPLE_LIMIT,
    STAGE3_KITTI_VAL_SAMPLE_LIMIT,
    STAGE3_LOSS_WEIGHTS,
    STAGE3_MODEL,
    STAGE3_NUM_WORKERS,
    STAGE3_NUSCENES_MANIFEST,
    STAGE3_NUSCENES_ROOT,
    STAGE3_NUSCENES_SAMPLE_LIMIT,
    STAGE3_NUSCENES_TRAIN_MANIFEST,
    STAGE3_NUSCENES_VAL_MANIFEST,
    STAGE3_RAW,
    STAGE3_SOURCE_BALANCED_SAMPLING,
    STAGE3_TARTANVO_FEATURE,
    STAGE3_TARTANVO_FEATURE_CACHE,
    STAGE3_TARTANVO_MODE,
    STAGE3_TARTANVO_USE_FEATURE_CACHE,
    STAGE3_TARTANVO_LR,
    STAGE3_HEAD_LR,
    STAGE3_TRAIN_SAMPLE_LIMIT,
    STAGE3_TRAIN_TEMPORAL_STRIDE,
    STAGE3_VAL_SAMPLE_LIMIT,
    STAGE3_VAL_TEMPORAL_STRIDE,
)
try:
    from src.config import STAGE3_DATASET_MODE
except ImportError:
    STAGE3_DATASET_MODE = STAGE3_DATASET
from src.datasets.comma2k19_stage3 import ACCEL_TO_ID, STEER_TO_ID, Comma2k19Stage3Dataset, Stage3DaconDataset
from src.datasets.kitti_stage3 import KittiStage3Dataset, _label_id
from src.datasets.nuscenes_stage3 import NuScenesStage3Dataset
from src.datasets.stage3_tartanvo_pose import Stage3MixedTartanFeatureDataset, Stage3TartanFeatureDataset
from src.models import Stage3MViT, Stage3ResNetGRU, Stage3TartanVOGRU
from src.utils import set_seed

set_seed(SEED)
SOURCE_REFS = {"comma2k19": 0.75, "kitti": 0.8887}


class SourceBalancedSampler(Sampler[int]):
    def __init__(self, dataset, seed: int = SEED):
        self.lengths = [len(ds) for ds in dataset.datasets]
        self.offsets = [0]
        for n in self.lengths[:-1]:
            self.offsets.append(self.offsets[-1] + n)
        self.samples_per_source = max(self.lengths) if self.lengths else 0
        self.seed = int(seed)

    def __len__(self) -> int:
        return self.samples_per_source * len(self.lengths)

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed + torch.initial_seed() % 100000)
        indices = []
        for offset, length in zip(self.offsets, self.lengths):
            if length <= 0:
                continue
            base = torch.randperm(length, generator=g).tolist()
            while len(base) < self.samples_per_source:
                base.extend(torch.randperm(length, generator=g).tolist())
            indices.extend(offset + j for j in base[:self.samples_per_source])
        order = torch.randperm(len(indices), generator=g).tolist()
        return iter([indices[j] for j in order])


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
    key = "segment_id" if "segment_id" in df.columns else "sequence_id" if "sequence_id" in df.columns else "scene" if "scene" in df.columns else "video_path"
    parts = [part.iloc[::stride] for _, part in df.groupby(key, sort=False)]
    return pd.concat(parts, ignore_index=True) if parts else df.reset_index(drop=True)


def _balanced_limit(df: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    if not limit or limit <= 0 or len(df) <= limit:
        return df.reset_index(drop=True)
    key = "sequence_id" if "sequence_id" in df.columns else "segment_id" if "segment_id" in df.columns else "scene" if "scene" in df.columns else None
    if key is None:
        return df.sample(n=limit, random_state=SEED).sort_index().reset_index(drop=True)
    per_group = max(1, limit // df[key].nunique() + 1)
    out = df.groupby(key, group_keys=False).sample(frac=1.0, random_state=SEED).groupby(key, group_keys=False).head(per_group)
    return out.head(limit).sort_index().reset_index(drop=True)


def _limited_dataset(dataset, stride: int, limit: int | None):
    before = len(dataset)
    dataset.df = _balanced_limit(_stride_manifest(dataset.df, stride), limit)
    return dataset, before, len(dataset)


def _source_limit(source: str, split: str) -> int | None:
    if source == "kitti":
        return STAGE3_KITTI_TRAIN_SAMPLE_LIMIT if split == "train" else STAGE3_KITTI_VAL_SAMPLE_LIMIT
    if STAGE3_DATASET_MODE == "mixed_features":
        return None
    if STAGE3_DATASET_MODE == "comma_only" and source == "comma2k19":
        return STAGE3_COMMA_TRAIN_SAMPLE_LIMIT if split == "train" else STAGE3_COMMA_VAL_SAMPLE_LIMIT
    if source == "nuscenes":
        return STAGE3_NUSCENES_SAMPLE_LIMIT
    if STAGE3_DATASET_MODE == "mixed":
        return STAGE3_COMMA_TRAIN_SAMPLE_LIMIT if split == "train" else STAGE3_COMMA_VAL_SAMPLE_LIMIT
    return STAGE3_TRAIN_SAMPLE_LIMIT if split == "train" else STAGE3_VAL_SAMPLE_LIMIT


def _feature_dataset(source: str, split: str):
    return Stage3TartanFeatureDataset(split, STAGE3_TARTANVO_FEATURE, root=STAGE3_TARTANVO_FEATURE_CACHE, dataset=source, limit=_source_limit(source, split))


def _available_feature_sources(split: str, sources: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(source for source in sources if (STAGE3_TARTANVO_FEATURE_CACHE / STAGE3_TARTANVO_FEATURE / source / f"{split}_index.csv").is_file())


def _feature_datasets():
    if STAGE3_DATASET_MODE == "mixed_features":
        train_sources = _available_feature_sources("train", ("comma2k19", "kitti", "nuscenes"))
        val_sources = _available_feature_sources("val", ("comma2k19", "kitti"))
        train = Stage3MixedTartanFeatureDataset("train", STAGE3_TARTANVO_FEATURE, root=STAGE3_TARTANVO_FEATURE_CACHE, sources=train_sources)
        val = {source: _feature_dataset(source, "val") for source in val_sources}
        return train, val, {"cache": True, "mixed": True, "train_sources": train.source_counts, "val_sources": {k: len(v) for k, v in val.items()}}
    source = "comma2k19" if STAGE3_DATASET_MODE == "comma_only" else STAGE3_DATASET_MODE
    train = _feature_dataset(source, "train")
    val = _feature_dataset(source, "val")
    return train, {source: val}, {"cache": True, "mixed": False, "train_sources": {source: len(train)}, "val_sources": {source: len(val)}}


def _raw_datasets():
    train_sets, val_sets = [], []
    train_sources, val_sources = {}, {}
    summary = {"dacon_train": 0, "dacon_val": 0, "train_sources": train_sources, "val_sources": val_sources}
    labels_paths = [STAGE3_RAW / "labels.csv", STAGE3_RAW / "stage3" / "labels.csv"]
    labels_path = next((path for path in labels_paths if path.is_file()), labels_paths[0])

    if STAGE3_DATASET_MODE in {"comma2k19", "mixed"} and labels_path.is_file():
        df = pd.read_csv(labels_path).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
        split = max(1, int(len(df) * 0.8)) if len(df) > 1 else len(df)
        video_root = labels_path.parent / "videos"
        ds = Stage3DaconDataset(df.iloc[:split], video_root=video_root)
        train_sets.append(ds); train_sources["DACON"] = len(ds); summary["dacon_train"] = len(ds)
        if split < len(df):
            ds = Stage3DaconDataset(df.iloc[split:], video_root=video_root)
            val_sets.append(("DACON", ds)); val_sources["DACON"] = len(ds); summary["dacon_val"] = len(ds)

    if STAGE3_DATASET_MODE in {"kitti", "mixed"}:
        if KITTI_STAGE3_TRAIN_MANIFEST.is_file():
            ds, before, after = _limited_dataset(KittiStage3Dataset(KITTI_STAGE3_TRAIN_MANIFEST), STAGE3_TRAIN_TEMPORAL_STRIDE, STAGE3_KITTI_TRAIN_SAMPLE_LIMIT)
            train_sets.append(ds); train_sources["KITTI"] = len(ds); summary.update(kitti_train_before=before, kitti_train_after=after)
        if KITTI_STAGE3_VAL_MANIFEST.is_file():
            ds, before, after = _limited_dataset(KittiStage3Dataset(KITTI_STAGE3_VAL_MANIFEST), STAGE3_VAL_TEMPORAL_STRIDE, STAGE3_KITTI_VAL_SAMPLE_LIMIT)
            val_sets.append(("KITTI", ds)); val_sources["KITTI"] = len(ds); summary.update(kitti_val_before=before, kitti_val_after=after)

    if STAGE3_DATASET_MODE == "nuscenes":
        if STAGE3_NUSCENES_MANIFEST.is_file():
            df = pd.read_csv(STAGE3_NUSCENES_MANIFEST)
            if "split" not in df.columns:
                df["split"] = "train"
            train_df = df[df.split == "train"].reset_index(drop=True)
            val_df = df[df.split == "val"].reset_index(drop=True)
            ds, before, after = _limited_dataset(NuScenesStage3Dataset(train_df, root=STAGE3_NUSCENES_ROOT), STAGE3_TRAIN_TEMPORAL_STRIDE, STAGE3_NUSCENES_SAMPLE_LIMIT)
            train_sets.append(ds); train_sources["nuScenes"] = len(ds); summary.update(nuscenes_train_before=before, nuscenes_train_after=after)
            if len(val_df):
                ds, before, after = _limited_dataset(NuScenesStage3Dataset(val_df, root=STAGE3_NUSCENES_ROOT), STAGE3_VAL_TEMPORAL_STRIDE, STAGE3_NUSCENES_SAMPLE_LIMIT)
                val_sets.append(("nuScenes", ds)); val_sources["nuScenes"] = len(ds); summary.update(nuscenes_val_before=before, nuscenes_val_after=after)
    elif STAGE3_DATASET_MODE == "mixed":
        if COMMA2K19_STAGE3_TRAIN_MANIFEST.is_file():
            ds, before, after = _limited_dataset(Comma2k19Stage3Dataset(COMMA2K19_STAGE3_TRAIN_MANIFEST), STAGE3_TRAIN_TEMPORAL_STRIDE, STAGE3_COMMA_TRAIN_SAMPLE_LIMIT)
            train_sets.append(ds); train_sources["comma2k19"] = len(ds); summary.update(comma_train_before=before, comma_train_after=after)
        if COMMA2K19_STAGE3_VAL_MANIFEST.is_file():
            ds, before, after = _limited_dataset(Comma2k19Stage3Dataset(COMMA2K19_STAGE3_VAL_MANIFEST), STAGE3_VAL_TEMPORAL_STRIDE, STAGE3_COMMA_VAL_SAMPLE_LIMIT)
            val_sets.append(("comma2k19", ds)); val_sources["comma2k19"] = len(ds); summary.update(comma_val_before=before, comma_val_after=after)
        if STAGE3_NUSCENES_TRAIN_MANIFEST.is_file():
            ds = NuScenesStage3Dataset(STAGE3_NUSCENES_TRAIN_MANIFEST, root=STAGE3_NUSCENES_ROOT)
            train_sets.append(ds); train_sources["nuScenes"] = len(ds); summary.update(nuscenes_train_before=len(ds), nuscenes_train_after=len(ds))
        summary["nuscenes_val_manifest"] = str(STAGE3_NUSCENES_VAL_MANIFEST)
    elif STAGE3_DATASET_MODE in {"comma2k19", "comma_only"}:
        if COMMA2K19_STAGE3_TRAIN_MANIFEST.is_file():
            ds, before, after = _limited_dataset(Comma2k19Stage3Dataset(COMMA2K19_STAGE3_TRAIN_MANIFEST), STAGE3_TRAIN_TEMPORAL_STRIDE, STAGE3_TRAIN_SAMPLE_LIMIT)
            train_sets.append(ds); train_sources["comma2k19"] = len(ds); summary.update(comma_train_before=before, comma_train_after=after)
        if COMMA2K19_STAGE3_VAL_MANIFEST.is_file():
            ds, before, after = _limited_dataset(Comma2k19Stage3Dataset(COMMA2K19_STAGE3_VAL_MANIFEST), STAGE3_VAL_TEMPORAL_STRIDE, STAGE3_VAL_SAMPLE_LIMIT)
            val_sets.append(("comma2k19", ds)); val_sources["comma2k19"] = len(ds); summary.update(comma_val_before=before, comma_val_after=after)
    elif STAGE3_DATASET_MODE != "kitti":
        raise ValueError(f"Unknown STAGE3_DATASET_MODE: {STAGE3_DATASET_MODE}")

    if not train_sets:
        raise FileNotFoundError("No Stage3 training data found.")
    train = ConcatDataset(train_sets)
    train.source_counts = dict(train_sources)
    return train, dict(val_sets), summary

def _datasets():
    if STAGE3_DATASET_MODE == "mixed_features":
        return _feature_datasets()
    if STAGE3_DATASET_MODE in {"mixed", "nuscenes"}:
        return _raw_datasets()
    if STAGE3_ARCH == "tartanvo_gru" and STAGE3_TARTANVO_MODE == "cached" and STAGE3_TARTANVO_USE_FEATURE_CACHE:
        return _feature_datasets()
    return _raw_datasets()


def _build_stage3_model(pretrained: bool = True):
    if STAGE3_ARCH == "mvit":
        return Stage3MViT(pretrained=pretrained)
    if STAGE3_ARCH == "resnet18_gru":
        return Stage3ResNetGRU(pretrained=pretrained)
    if STAGE3_ARCH == "tartanvo_gru":
        return Stage3TartanVOGRU(load_pretrained=pretrained)
    raise ValueError(f"Unknown STAGE3_ARCH: {STAGE3_ARCH}")


def _labels_from_dataset(dataset):
    if hasattr(dataset, "df"):
        accel = [_label_id(v, ACCEL_TO_ID) for v in dataset.df.accel_label.tolist()]
        steer = [_label_id(v, STEER_TO_ID) for v in dataset.df.steer_label.tolist()]
        return accel, steer
    accel, steer = [], []
    for ds in getattr(dataset, "datasets", []):
        a, s = _labels_from_dataset(ds)
        accel.extend(a); steer.extend(s)
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


def _print_dataset_summary(train_dataset, val_datasets: dict[str, object], summary: dict) -> None:
    print("=== Stage 3 Dataset ===")
    print(f"Dataset mode: {STAGE3_DATASET_MODE}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {sum(len(v) for v in val_datasets.values())}")
    print(f"Architecture: {STAGE3_ARCH}")
    print(f"TartanVO feature mode: {STAGE3_TARTANVO_MODE}/{STAGE3_TARTANVO_FEATURE}")
    print(f"TartanVO feature cache: {bool(summary.get('cache'))}")
    print(f"Batch size: {BATCH_SIZE}")
    if summary.get("cache"):
        print("feature: [15, 1536]")
    if hasattr(train_dataset, "source_counts"):
        print("=== Stage 3 Training Sources ===")
        for name in ("KITTI", "comma2k19", "nuScenes"):
            print(f"{name:<11}: {train_dataset.source_counts.get(name, 0)}")
        print(f"{'Total':<11}: {len(train_dataset)}")
        print(f"source counts: {train_dataset.source_counts}")
        for source, ds in getattr(train_dataset, "source_datasets", {}).items():
            _print_one_distribution(f"{source} train", ds)
        if hasattr(train_dataset, "comma"):
            _print_one_distribution("comma2k19 train", train_dataset.comma)
        if hasattr(train_dataset, "kitti"):
            _print_one_distribution("KITTI train", train_dataset.kitti)
        _print_one_distribution("mixed train", train_dataset)
    else:
        _print_one_distribution(f"{STAGE3_DATASET_MODE} train", train_dataset)
    for source, ds in val_datasets.items():
        _print_one_distribution(f"{source} val", ds)
    if len(val_datasets) > 1:
        _print_one_distribution("mixed val", ConcatDataset(list(val_datasets.values())))
    accel_names = {v: k for k, v in ACCEL_TO_ID.items()}
    steer_names = {v: k for k, v in STEER_TO_ID.items()}
    accel, steer = _labels_from_dataset(train_dataset)
    print("Accel:")
    for i in range(4):
        print(f"  {accel_names[i]} {accel.count(i)}")
    print("Steering:")
    for i in range(3):
        print(f"  {steer_names[i]} {steer.count(i)}")


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


def _validate_all(model, loaders: dict[str, DataLoader]) -> dict:
    metrics = {source: _validate(model, loader) for source, loader in loaders.items()}
    if len(loaders) > 1:
        metrics["overall"] = _validate(model, DataLoader(ConcatDataset([loader.dataset for loader in loaders.values()]), batch_size=BATCH_SIZE, shuffle=False, num_workers=0))
        metrics["mixed_source_selection"] = sum(metrics[s]["selection"] for s in loaders) / len(loaders)
    else:
        only = next(iter(metrics.values())) if metrics else {"selection": float("nan")}
        metrics["overall"] = only
        metrics["mixed_source_selection"] = only["selection"]
    return metrics


def _source_balanced_sampler(dataset):
    if STAGE3_DATASET_MODE in {"mixed", "mixed_features", "comma_only"} or not STAGE3_SOURCE_BALANCED_SAMPLING or not hasattr(dataset, "source_counts"):
        return None
    return SourceBalancedSampler(dataset)

def _stage3_collate(batch: list[dict]) -> dict:
    out = {
        "accel_label": torch.tensor([item["accel_label"] for item in batch], dtype=torch.long),
        "steer_label": torch.tensor([item["steer_label"] for item in batch], dtype=torch.long),
    }
    for key in ("feature", "pose", "video", "intrinsics"):
        if all(key in item for item in batch):
            out[key] = torch.stack([item[key] for item in batch])
    return out


def _loader(dataset, shuffle: bool):
    sampler = _source_balanced_sampler(dataset) if shuffle else None
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle and sampler is None, sampler=sampler, num_workers=STAGE3_NUM_WORKERS, pin_memory=torch.cuda.is_available(), persistent_workers=STAGE3_NUM_WORKERS > 0, collate_fn=_stage3_collate)


def _param_count(model, trainable: bool) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad is trainable)


def _history_keys() -> list[str]:
    keys = ["epoch", "arch", "dataset_mode", "batch_size", "train_loss", "train_accel_loss", "train_steer_loss", "val_accel_accuracy", "val_accel_macro_f1", "val_steer_accuracy", "val_steer_macro_f1", "selection"]
    for source, prefix in (("comma2k19", "comma"), ("kitti", "kitti"), ("nuscenes", "nuscenes")):
        keys += [f"{prefix}_val_accel_accuracy", f"{prefix}_val_accel_macro_f1", f"{prefix}_val_steer_accuracy", f"{prefix}_val_steer_macro_f1", f"{prefix}_selection"]
    keys.append("mixed_source_selection")
    return keys


def _append_history(history: dict, epoch: int, train_loss: float, train_accel_loss: float, train_steer_loss: float, metrics: dict) -> None:
    overall = metrics["overall"]
    row = {
        "epoch": epoch,
        "arch": STAGE3_ARCH,
        "dataset_mode": STAGE3_DATASET_MODE,
        "batch_size": BATCH_SIZE,
        "train_loss": train_loss,
        "train_accel_loss": train_accel_loss,
        "train_steer_loss": train_steer_loss,
        "val_accel_accuracy": overall["accel"]["accuracy"],
        "val_accel_macro_f1": overall["accel"]["macro_f1"],
        "val_steer_accuracy": overall["steer"]["accuracy"],
        "val_steer_macro_f1": overall["steer"]["macro_f1"],
        "selection": overall["selection"],
        "mixed_source_selection": metrics["mixed_source_selection"],
    }
    for source, prefix in (("comma2k19", "comma"), ("kitti", "kitti"), ("nuscenes", "nuscenes")):
        m = metrics.get(source)
        row[f"{prefix}_val_accel_accuracy"] = m["accel"]["accuracy"] if m else float("nan")
        row[f"{prefix}_val_accel_macro_f1"] = m["accel"]["macro_f1"] if m else float("nan")
        row[f"{prefix}_val_steer_accuracy"] = m["steer"]["accuracy"] if m else float("nan")
        row[f"{prefix}_val_steer_macro_f1"] = m["steer"]["macro_f1"] if m else float("nan")
        row[f"{prefix}_selection"] = m["selection"] if m else float("nan")
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
    for name in ("train_loss", "train_accel_loss", "train_steer_loss"):
        plt.plot(df["epoch"], df[name], marker="o", label=name)
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title(f"Stage3 Training Loss - {STAGE3_ARCH}"); plt.legend(); plt.grid(True)
    plt.savefig(loss_path, bbox_inches="tight"); plt.close()
    plt.figure()
    for name in ("val_accel_macro_f1", "val_steer_macro_f1", "selection", "mixed_source_selection", "comma_selection", "kitti_selection"):
        if name in df and not df[name].isna().all():
            plt.plot(df["epoch"], df[name], marker="o", label=name)
    plt.xlabel("Epoch"); plt.ylabel("Score"); plt.ylim(0, 1); plt.title(f"Stage3 Validation Metrics - {STAGE3_ARCH}"); plt.legend(); plt.grid(True)
    plt.savefig(metrics_path, bbox_inches="tight"); plt.close()
    return history_path, loss_path, metrics_path


def _checkpoint_payload(model, epoch: int, train_loss: float, metrics=None, history=None) -> dict:
    payload = {"model": model.state_dict(), "arch": STAGE3_ARCH, "epoch": epoch, "train_loss": train_loss, "dataset_mode": STAGE3_DATASET_MODE}
    if metrics is not None:
        payload["metrics"] = metrics
    if history is not None:
        payload["history"] = history
    if hasattr(model, "model_config"):
        payload["model_config"] = model.model_config()
    return payload


def _print_metrics_table(metrics: dict, prefix: str = "") -> None:
    print(prefix + "Source      Accel F1   Steer F1   Selection")
    print(prefix + "-------------------------------------------")
    for name, label in (("comma2k19", "comma val"), ("kitti", "KITTI val"), ("nuscenes", "nuScenes"), ("overall", "overall")):
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
    model = _build_stage3_model(pretrained=True).to(DEVICE)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("Stage3 model has no trainable parameters.")
    print(f"Trainable parameters: {_param_count(model, True)}")
    print(f"Frozen parameters: {_param_count(model, False)}")
    if STAGE3_ARCH == "tartanvo_gru":
        print(f"TartanVO feature mode: {STAGE3_TARTANVO_MODE}/{STAGE3_TARTANVO_FEATURE}")
        print(f"TartanVO unfreeze: {model.tartanvo_unfreeze}")
        print(f"feature dimension: {model.feature_dim}")
        print("sequence length: 15")
        print(f"TartanVO pretrained loaded: {model.tartanvo.pretrained_loaded}")
    if STAGE3_ARCH == "tartanvo_gru" and STAGE3_TARTANVO_MODE == "finetune":
        tartan_params = [p for p in model.tartanvo.parameters() if p.requires_grad]
        head_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("tartanvo.")]
        groups = []
        if tartan_params:
            groups.append({"params": tartan_params, "lr": STAGE3_TARTANVO_LR})
        if head_params:
            groups.append({"params": head_params, "lr": STAGE3_HEAD_LR})
        opt = torch.optim.AdamW(groups)
        print(f"Optimizer LR: tartanvo={STAGE3_TARTANVO_LR} head={STAGE3_HEAD_LR}")
    else:
        opt = torch.optim.AdamW(trainable_params, STAGE3_HEAD_LR)
    accel_class_weights = _class_weights("accel")
    steer_class_weights = _class_weights("steer")
    best = -1.0
    best_metrics = None
    best_epoch = 0

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
        metrics = _validate_all(model, val_loaders) if val_loaders else {"overall": {"accel": {"accuracy": float("nan"), "macro_f1": float("nan")}, "steer": {"accuracy": float("nan"), "macro_f1": float("nan")}, "selection": float("nan")}, "mixed_source_selection": float("nan")}
        _print_metrics_table(metrics, prefix=f"epoch={epoch + 1} ")
        _append_history(history, epoch + 1, train_loss, train_accel_loss, train_steer_loss, metrics)
        history_path, loss_path, metrics_path = _save_history(history, run_id)
        select = metrics["mixed_source_selection"] if STAGE3_DATASET_MODE in {"mixed", "mixed_features"} else metrics["overall"]["selection"]
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
        print(f"mixed_source_selection={best:.5f}")

