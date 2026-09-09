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
    STAGE3_KITTI_TRAIN_SAMPLE_LIMIT,
    STAGE3_KITTI_VAL_SAMPLE_LIMIT,
    STAGE3_NUM_FRAMES,
    STAGE3_TARTANVO_FEATURE_CACHE,
    STAGE3_TRAIN_SAMPLE_LIMIT,
    STAGE3_TRAIN_TEMPORAL_STRIDE,
    STAGE3_VAL_SAMPLE_LIMIT,
    STAGE3_VAL_TEMPORAL_STRIDE,
    TARTANVO_CHECKPOINT,
)
from src.datasets.comma2k19_stage3 import Comma2k19Stage3Dataset
from src.datasets.kitti_stage3 import KittiStage3Dataset
from src.datasets.stage3_tartanvo_pose import stage3_tartanvo_sample_key
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
            return COMMA2K19_STAGE3_TRAIN_MANIFEST, STAGE3_TRAIN_TEMPORAL_STRIDE, STAGE3_TRAIN_SAMPLE_LIMIT
        if split == "val":
            return COMMA2K19_STAGE3_VAL_MANIFEST, STAGE3_VAL_TEMPORAL_STRIDE, STAGE3_VAL_SAMPLE_LIMIT
    raise ValueError(f"unknown dataset/split: {dataset}/{split}")


def _dataset(dataset: str, manifest: Path):
    if dataset == "kitti":
        return KittiStage3Dataset(manifest)
    if dataset == "comma2k19":
        return Comma2k19Stage3Dataset(manifest)
    raise ValueError(f"unknown dataset: {dataset}")


def _cache_base(root: Path, feature: str, dataset: str) -> Path:
    return root / feature / dataset if dataset != "comma2k19" else root / feature


def _checkpoint_sha1(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def cache_split(split: str, feature: str, root: Path = STAGE3_TARTANVO_FEATURE_CACHE, overwrite: bool = False, dataset: str = "comma2k19") -> None:
    if feature not in {"pose", "latent"}:
        raise ValueError(f"unknown feature: {feature}")
    manifest, stride, limit = _split_config(dataset, split)
    ds = _dataset(dataset, manifest)
    ds.df = _balanced_limit(_stride_manifest(ds.df, stride), limit)

    base = _cache_base(root, feature, dataset)
    out_dir = base / split
    out_dir.mkdir(parents=True, exist_ok=True)
    model = Stage3TartanVOGRU(load_pretrained=True, feature=feature).to(DEVICE).eval()
    rows = []
    with torch.inference_mode():
        for i in tqdm(range(len(ds)), desc=f"cache {dataset} tartanvo {feature} {split}"):
            row = ds.df.iloc[i]
            key = stage3_tartanvo_sample_key(row)
            path = out_dir / f"{key}.pt"
            accel_label = int(row.accel_label)
            steer_label = int(row.steer_label)
            if overwrite or not path.exists():
                item = ds[i]
                video = item["video"].unsqueeze(0).to(DEVICE, non_blocking=True)
                intrinsics = item.get("intrinsics")
                intrinsics = intrinsics.unsqueeze(0).to(DEVICE, non_blocking=True) if intrinsics is not None else None
                cached = model.feature_sequence(video, intrinsics=intrinsics).squeeze(0).cpu().to(torch.float32)
                payload = {
                    "feature": cached,
                    "feature_mode": feature,
                    "feature_dim": int(cached.shape[-1]),
                    "accel_label": accel_label,
                    "steer_label": steer_label,
                    "sample_key": key,
                    "frame_index": int(row.frame_index),
                }
                for name in ("video_path", "image_path", "sequence_id"):
                    if name in item:
                        payload[name] = item[name]
                    elif hasattr(row, name):
                        payload[name] = getattr(row, name)
                torch.save(payload, path)
            rows.append({"sample_key": key, "accel_label": accel_label, "steer_label": steer_label, "sequence_id": getattr(row, "sequence_id", None), "frame_index": int(row.frame_index)})
    pd.DataFrame(rows).to_csv(base / f"{split}_index.csv", index=False)
    metadata = {
        "cache_version": CACHE_VERSION,
        "dataset": dataset,
        "feature_mode": feature,
        "feature_dim": model.feature_dim,
        "num_frames": STAGE3_NUM_FRAMES,
        "temporal_stride": stride,
        "sample_limit": limit,
        "seed": SEED,
        "tartanvo_checkpoint": str(TARTANVO_CHECKPOINT),
        "tartanvo_checkpoint_name": TARTANVO_CHECKPOINT.name,
        "tartanvo_checkpoint_sha1": _checkpoint_sha1(TARTANVO_CHECKPOINT),
        "split": split,
        "samples": len(rows),
    }
    (base / f"{split}_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache frozen TartanVO features for Stage3.")
    parser.add_argument("--split", choices=["train", "val"], required=True)
    parser.add_argument("--feature", choices=["pose", "latent"], default="latent")
    parser.add_argument("--dataset", choices=["comma2k19", "kitti"], default="comma2k19")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    cache_split(args.split, args.feature, overwrite=args.overwrite, dataset=args.dataset)


if __name__ == "__main__":
    main()
