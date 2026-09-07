from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from src.config import (
    COMMA2K19_STAGE3_TRAIN_MANIFEST,
    COMMA2K19_STAGE3_VAL_MANIFEST,
    DEVICE,
    STAGE3_TARTANVO_FEATURE_CACHE,
    STAGE3_TRAIN_SAMPLE_LIMIT,
    STAGE3_TRAIN_TEMPORAL_STRIDE,
    STAGE3_VAL_SAMPLE_LIMIT,
    STAGE3_VAL_TEMPORAL_STRIDE,
)
from src.datasets.comma2k19_stage3 import Comma2k19Stage3Dataset
from src.datasets.stage3_tartanvo_pose import stage3_tartanvo_sample_key
from src.models import Stage3TartanVOGRU


def _stride_manifest(df: pd.DataFrame, stride: int) -> pd.DataFrame:
    if stride <= 1 or df.empty:
        return df.reset_index(drop=True)
    key = "segment_id" if "segment_id" in df.columns else "video_path"
    parts = [part.iloc[::stride] for _, part in df.groupby(key, sort=False)]
    return pd.concat(parts, ignore_index=True) if parts else df.reset_index(drop=True)


def _split_config(split: str):
    if split == "train":
        return COMMA2K19_STAGE3_TRAIN_MANIFEST, STAGE3_TRAIN_TEMPORAL_STRIDE, STAGE3_TRAIN_SAMPLE_LIMIT
    if split == "val":
        return COMMA2K19_STAGE3_VAL_MANIFEST, STAGE3_VAL_TEMPORAL_STRIDE, STAGE3_VAL_SAMPLE_LIMIT
    raise ValueError(f"unknown split: {split}")


def cache_split(split: str, root: Path = STAGE3_TARTANVO_FEATURE_CACHE, overwrite: bool = False) -> None:
    manifest, stride, limit = _split_config(split)
    dataset = Comma2k19Stage3Dataset(manifest)
    dataset.df = _stride_manifest(dataset.df, stride)
    if limit is not None and limit > 0:
        dataset.df = dataset.df.head(limit).reset_index(drop=True)

    out_dir = root / split
    out_dir.mkdir(parents=True, exist_ok=True)
    model = Stage3TartanVOGRU(load_pretrained=True).to(DEVICE).eval()
    rows = []
    with torch.inference_mode():
        for i in tqdm(range(len(dataset)), desc=f"cache tartanvo {split}"):
            row = dataset.df.iloc[i]
            key = stage3_tartanvo_sample_key(row)
            path = out_dir / f"{key}.pt"
            accel_label = int(row.accel_label)
            steer_label = int(row.steer_label)
            if overwrite or not path.exists():
                item = dataset[i]
                video = item["video"].unsqueeze(0).to(DEVICE, non_blocking=True)
                pose = model.pose_sequence(video).squeeze(0).cpu().to(torch.float32)
                torch.save({
                    "pose": pose,
                    "accel_label": accel_label,
                    "steer_label": steer_label,
                    "sample_key": key,
                    "video_path": item["video_path"],
                    "frame_index": int(row.frame_index),
                }, path)
            rows.append({"sample_key": key, "accel_label": accel_label, "steer_label": steer_label})
    pd.DataFrame(rows).to_csv(root / f"{split}_index.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache frozen TartanVO pose features for Stage3.")
    parser.add_argument("--split", choices=["train", "val"], required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    cache_split(args.split, overwrite=args.overwrite)


if __name__ == "__main__":
    main()