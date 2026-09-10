from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import STAGE3_NUSCENES_MANIFEST, STAGE3_NUSCENES_ROOT
from src.datasets.comma2k19_stage3 import ACCEL_TO_ID, STEER_TO_ID
from src.datasets.nuscenes_stage3 import NuScenesStage3Dataset


def _stats(name: str, values: pd.Series) -> None:
    arr = values.dropna().to_numpy(float)
    print(f"{name} min/mean/max: {arr.min():.6f} {arr.mean():.6f} {arr.max():.6f}")


def _dist(name: str, values: pd.Series, names: dict[int, str]) -> None:
    counts = values.astype(int).value_counts().sort_index()
    print(name)
    for key, label in names.items():
        print(f"  {label}: {int(counts.get(key, 0))}")


def smoke(manifest: Path, root: Path, samples: int) -> None:
    df = pd.read_csv(manifest)
    ds = NuScenesStage3Dataset(manifest, root=root)
    print(f"number of scenes: {df.scene.nunique()}")
    print(f"number of valid frames: {len(df)}")
    print(f"number of generated clips: {len(ds)}")
    for col in ("speed", "accel", "yaw_rate", "steering_raw", "steering_centered"):
        _stats(col, df[col])
    _dist("accel class distribution", df.accel_label, {v: k for k, v in ACCEL_TO_ID.items()})
    _dist("steering class distribution", df.steer_label, {v: k for k, v in STEER_TO_ID.items()})
    if len(ds) == 0:
        return
    rng = np.random.default_rng(42)
    for idx in rng.choice(len(ds), size=min(samples, len(ds)), replace=False):
        sample = ds[int(idx)]
        row = ds.df.iloc[int(idx)]
        clip = ds.clip_rows(row)
        ts = clip.timestamp.to_numpy(float)
        print(f"\nclip index: {int(idx)}")
        print("frame paths:", sample["frame_paths"])
        print("timestamps:", ts.astype(np.int64).tolist())
        print("timestamp intervals:", (np.diff(ts) / 1_000_000.0).round(4).tolist())
        print("accel values:", clip.accel.round(6).tolist())
        print("yaw_rate values:", clip.yaw_rate.round(6).tolist())
        print("final class labels:", {"accel_label": sample["accel_label"], "steer_label": sample["steer_label"]})
        print("image tensor shape:", tuple(sample["video"].shape))


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test nuScenes Stage3 manifest and dataset.")
    parser.add_argument("--manifest", type=Path, default=STAGE3_NUSCENES_MANIFEST)
    parser.add_argument("--root", type=Path, default=STAGE3_NUSCENES_ROOT)
    parser.add_argument("--samples", type=int, default=3)
    args = parser.parse_args()
    smoke(args.manifest, args.root, args.samples)


if __name__ == "__main__":
    main()