from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import torch

from src.config import (
    COMMA2K19_STAGE3_TRAIN_MANIFEST,
    DEVICE,
    STAGE3_TARTANVO_CACHE_PAIR_BATCH_SIZE,
    STAGE3_TARTANVO_FEATURE_CACHE,
)
from src.datasets.comma2k19_stage3 import Comma2k19Stage3Dataset
from src.datasets.stage3_tartanvo_pose import Stage3TartanFeatureDataset
from src.models import Stage3TartanVOGRU
from src.tools.cache_stage3_tartanvo_feature import cache_segment_split
from src.train.stage3 import _loader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--manifest", type=Path, default=COMMA2K19_STAGE3_TRAIN_MANIFEST)
    parser.add_argument("--cache-root", type=Path, default=STAGE3_TARTANVO_FEATURE_CACHE)
    parser.add_argument("--pair-batch-size", type=int, default=STAGE3_TARTANVO_CACHE_PAIR_BATCH_SIZE)
    parser.add_argument("--benchmark-segments", type=int, default=3)
    args = parser.parse_args()

    t0 = time.perf_counter()
    cache_segment_split(args.split, args.cache_root, pair_batch_size=args.pair_batch_size, max_segments=args.benchmark_segments)
    elapsed = time.perf_counter() - t0

    base = args.cache_root / "segment_latent" / "comma2k19"
    index = pd.read_csv(base / f"{args.split}_index.csv")
    first = index.iloc[0]
    item = torch.load(base / str(first.feature_path), map_location="cpu", weights_only=False)
    assert item["features"].shape == (int(item["num_frames"]) - 1, 1536)

    ds = Stage3TartanFeatureDataset(args.split, "latent", root=args.cache_root, dataset="comma2k19", limit=8)
    sample = ds[0]
    assert sample["feature"].shape == (15, 1536)

    raw = Comma2k19Stage3Dataset(args.manifest)
    raw.df = pd.DataFrame([ds.df.iloc[0]]).reset_index(drop=True)
    model = Stage3TartanVOGRU(load_pretrained=True, feature="latent").to(DEVICE).eval()
    batch = next(iter(_loader(ds, shuffle=False)))
    with torch.inference_mode():
        old = model.feature_sequence(raw[0]["video"].unsqueeze(0).to(DEVICE)).squeeze(0).cpu()
        accel, steer = model.forward_feature(batch["feature"].to(DEVICE))
    new = sample["feature"]
    diff = (old - new).abs()

    pairs = int(index.num_pairs.sum())
    size_mb = sum(p.stat().st_size for p in (base / args.split).glob("*.pt")) / 1024 / 1024
    print("segment smoke ok")
    print("window old: center-8..center+7 frames, clamped in legacy sample loader")
    print("window new: pair indices center-8:center+7, invalid samples excluded")
    print("segments", len(index), "pairs", pairs, "feature_dim", 1536)
    print("cache_size_mb", f"{size_mb:.2f}")
    print("max_abs_diff", float(diff.max()), "mean_abs_diff", float(diff.mean()))
    print("dataset_batch_shape", tuple(batch["feature"].shape))
    print("model_outputs", tuple(accel.shape), tuple(steer.shape))
    print("seconds_per_segment", f"{elapsed / max(1, len(index)):.3f}")
    print("seconds_per_pair", f"{elapsed / max(1, pairs):.5f}")
    print("pairs_per_second", f"{pairs / max(elapsed, 1e-9):.2f}")


if __name__ == "__main__":
    main()
