from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from src.config import (
    COMMA2K19_STAGE3_TRAIN_MANIFEST,
    COMMA2K19_STAGE3_VAL_MANIFEST,
    DEVICE,
    KITTI_STAGE3_TRAIN_MANIFEST,
    KITTI_STAGE3_VAL_MANIFEST,
    SEED,
    STAGE3_COMMA_TRAIN_SAMPLE_LIMIT,
    STAGE3_COMMA_VAL_SAMPLE_LIMIT,
    STAGE3_KITTI_TRAIN_SAMPLE_LIMIT,
    STAGE3_KITTI_VAL_SAMPLE_LIMIT,
    STAGE3_NUSCENES_ROOT,
    STAGE3_NUSCENES_TRAIN_MANIFEST,
    STAGE3_NUSCENES_VAL_MANIFEST,
    STAGE3_NUM_FRAMES,
    STAGE3_TARTANVO_FEATURE_CACHE,
    STAGE3_TRAIN_SAMPLE_LIMIT,
    STAGE3_TRAIN_TEMPORAL_STRIDE,
    STAGE3_VAL_SAMPLE_LIMIT,
    STAGE3_VAL_TEMPORAL_STRIDE,
    TARTANVO_CHECKPOINT,
)
from src.datasets.comma2k19_stage3 import ACCEL_TO_ID, STEER_TO_ID, Comma2k19Stage3Dataset
from src.datasets.kitti_stage3 import KittiStage3Dataset, _label_id
from src.datasets.nuscenes_stage3 import NuScenesStage3Dataset
from src.datasets.stage3_tartanvo_pose import stage3_tartanvo_feature_path, stage3_tartanvo_sample_key
from src.models import Stage3TartanVOGRU

CACHE_VERSION = 2


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
    return df.groupby(key, group_keys=False).sample(frac=1.0, random_state=SEED).groupby(key, group_keys=False).head(max(1, limit // df[key].nunique() + 1)).head(limit).sort_index().reset_index(drop=True)


def _split_config(dataset: str, split: str):
    if dataset == "kitti":
        if split == "train":
            return KITTI_STAGE3_TRAIN_MANIFEST, STAGE3_TRAIN_TEMPORAL_STRIDE, STAGE3_KITTI_TRAIN_SAMPLE_LIMIT
        if split == "val":
            return KITTI_STAGE3_VAL_MANIFEST, STAGE3_VAL_TEMPORAL_STRIDE, STAGE3_KITTI_VAL_SAMPLE_LIMIT
    if dataset == "comma2k19":
        if split == "train":
            return COMMA2K19_STAGE3_TRAIN_MANIFEST, STAGE3_TRAIN_TEMPORAL_STRIDE, STAGE3_COMMA_TRAIN_SAMPLE_LIMIT
        if split == "val":
            return COMMA2K19_STAGE3_VAL_MANIFEST, STAGE3_VAL_TEMPORAL_STRIDE, STAGE3_COMMA_VAL_SAMPLE_LIMIT
    if dataset == "nuscenes":
        return (STAGE3_NUSCENES_TRAIN_MANIFEST if split == "train" else STAGE3_NUSCENES_VAL_MANIFEST), 1, None
    raise ValueError(f"unknown dataset/split: {dataset}/{split}")



def _dataset(dataset: str, manifest: Path, split: str):
    if dataset == "kitti":
        return KittiStage3Dataset(manifest)
    if dataset == "comma2k19":
        return Comma2k19Stage3Dataset(manifest)
    if dataset == "nuscenes":
        return NuScenesStage3Dataset(manifest, root=STAGE3_NUSCENES_ROOT)
    raise ValueError(f"unknown dataset: {dataset}")


def _cache_base(root: Path, feature: str, dataset: str) -> Path:
    return root / feature / dataset


def _checkpoint_sha1(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _row_frame_index(row) -> int:
    return int(getattr(row, "video_frame_index", getattr(row, "frame_index", getattr(row, "frame_idx", -1))))


def _row_sequence_id(row):
    return getattr(row, "sequence_id", getattr(row, "scene", getattr(row, "ID", None)))


def cache_split(split: str, feature: str, root: Path = STAGE3_TARTANVO_FEATURE_CACHE, overwrite: bool = False, dataset: str = "comma2k19", batch_size: int = 4, max_samples: int | None = None) -> None:
    if feature not in {"pose", "latent"}:
        raise ValueError(f"unknown feature: {feature}")
    manifest, stride, limit = _split_config(dataset, split)
    ds = _dataset(dataset, manifest, split)
    ds.df = _balanced_limit(_stride_manifest(ds.df, stride), limit)
    if max_samples:
        ds.df = ds.df.head(max_samples).reset_index(drop=True)

    base = _cache_base(root, feature, dataset)
    out_dir = base / split
    out_dir.mkdir(parents=True, exist_ok=True)
    model = Stage3TartanVOGRU(load_pretrained=True, feature=feature).to(DEVICE).eval()
    rows = []
    pending = []

    def flush() -> None:
        if not pending:
            return
        items = [x[0] for x in pending]
        video = torch.stack([item["video"] for item in items]).to(DEVICE, non_blocking=True)
        intrinsics = torch.stack([item["intrinsics"] for item in items]).to(DEVICE, non_blocking=True) if all("intrinsics" in item for item in items) else None
        cached_batch = model.feature_sequence(video, intrinsics=intrinsics).cpu().to(torch.float32)
        for cached, (item, row, key, path, accel_label, steer_label) in zip(cached_batch, pending):
            payload = {
                "feature": cached,
                "feature_mode": feature,
                "feature_dim": int(cached.shape[-1]),
                "accel_label": accel_label,
                "steer_label": steer_label,
                "sample_key": key,
                "source": dataset,
                "video_frame_index": _row_frame_index(row),
            }

            for name in ("video_path", "image_path", "sequence_id", "scene", "timestamp", "frame_paths"):
                if name in item:
                    payload[name] = item[name]
                elif hasattr(row, name):
                    payload[name] = getattr(row, name)
            torch.save(payload, path)
        pending.clear()

    with torch.inference_mode():
        for i in tqdm(range(len(ds)), desc=f"cache {dataset} tartanvo {feature} {split}"):
            row = ds.df.iloc[i]
            key = stage3_tartanvo_sample_key(row)
            feature_path = stage3_tartanvo_feature_path(split, key)
            path = base / feature_path
            accel_label = _label_id(row.accel_label, ACCEL_TO_ID)
            steer_label = _label_id(row.steer_label, STEER_TO_ID)
            if overwrite or not path.exists():
                pending.append((ds[i], row, key, path, accel_label, steer_label))
                if len(pending) >= batch_size:
                    flush()
            out_row = {"sample_key": key, "feature_path": feature_path, "accel_label": accel_label, "steer_label": steer_label, "sequence_id": _row_sequence_id(row), "video_frame_index": _row_frame_index(row)}
            for name in ("alignment_version", "alignment_source", "target_timestamp", "video_pts_sec", "alignment_error_sec"):
                if hasattr(row, name):
                    out_row[name] = getattr(row, name)
            rows.append(out_row)
        flush()
    pd.DataFrame(rows).to_csv(base / f"{split}_index.csv", index=False)
    metadata = {
        "cache_version": CACHE_VERSION,
        "dataset": dataset,
        "feature_mode": feature,
        "feature_dim": model.feature_dim,
        "num_frames": STAGE3_NUM_FRAMES,
        "temporal_stride": stride,
        "sample_limit": limit,
        "max_samples": max_samples,
        "seed": SEED,
        "tartanvo_checkpoint": str(TARTANVO_CHECKPOINT),
        "tartanvo_checkpoint_name": TARTANVO_CHECKPOINT.name,
        "tartanvo_checkpoint_sha1": _checkpoint_sha1(TARTANVO_CHECKPOINT),
        "preprocessing": "Stage3 dataset video -> unnormalize S3_MEAN/S3_STD -> bilinear resize 448x640 -> TartanVO pairs",
        "temporal_length": STAGE3_NUM_FRAMES - 1,
        "split": split,
        "samples": len(rows),
        "manifest_alignment_version": str(ds.df.alignment_version.iloc[0]) if "alignment_version" in ds.df.columns and len(ds.df) else None,
    }
    (base / f"{split}_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

def main() -> None:
    parser = argparse.ArgumentParser(description="Cache frozen TartanVO features for Stage3.")
    parser.add_argument("--split", choices=["train", "val"], required=True)
    parser.add_argument("--feature", choices=["pose", "latent"], default="latent")
    parser.add_argument("--dataset", choices=["comma2k19", "kitti", "nuscenes"], default="comma2k19")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()
    cache_split(args.split, args.feature, overwrite=args.overwrite, dataset=args.dataset, batch_size=args.batch_size, max_samples=args.max_samples)


if __name__ == "__main__":
    main()

