from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src import config as C
from src.datasets.stage3_tartanvo_pose import (
    Stage3TartanFeatureDataset,
    _limit_df,
    _row_frame_index,
    _stride_manifest,
    stage3_tartanvo_segment_key,
    stage3_tartanvo_window,
)
from src.tools.stage3_comma_manifest import active_manifest_path, active_subset_name, segment_cache_index_name

ACCEL = ["ACCELERATING", "DECELERATING", "CONSTANT", "STOPPED"]
STEER = ["LEFT", "STRAIGHT", "RIGHT"]


def _label_dist(df: pd.DataFrame, col: str, names: list[str]) -> dict[str, float]:
    values = df[col]
    if values.dtype == object:
        counts = values.astype(str).value_counts().to_dict()
        total = max(1, len(values))
        return {name: float(counts.get(name, 0) / total) for name in names}
    counts = values.astype(int).value_counts().to_dict()
    total = max(1, len(values))
    return {name: float(counts.get(i, 0) / total) for i, name in enumerate(names)}


def _print_delta(title: str, full: pd.DataFrame, sampled: pd.DataFrame) -> None:
    print(title)
    for col, names in (("accel_label", ACCEL), ("steer_label", STEER)):
        base = _label_dist(full, col, names)
        now = _label_dist(sampled, col, names)
        print(f"  {col}")
        for name in names:
            print(f"    {name:<12} full={base[name]:.4f} sampled={now[name]:.4f} delta={now[name] - base[name]:+.4f}")


def _segment_index(split: str) -> dict[str, dict]:
    subset = active_subset_name(split)
    path = C.STAGE3_TARTANVO_FEATURE_CACHE / "segment_latent" / "comma2k19" / segment_cache_index_name(split, subset)
    if not path.is_file():
        raise FileNotFoundError(f"missing segment cache index: {path}")
    index = pd.read_csv(path)
    return {str(row.segment_key): row._asdict() for row in index.itertuples(index=False)}


def _manifest_with_validity(split: str) -> pd.DataFrame:
    manifest = active_manifest_path(split)
    if not manifest.is_file():
        raise FileNotFoundError(f"missing active manifest: {manifest}")
    segment_index = _segment_index(split)
    df = pd.read_csv(manifest)
    df["segment_key"] = [stage3_tartanvo_segment_key(row) for row in df.itertuples(index=False)]
    df["cache_available"] = [str(key) in segment_index for key in df.segment_key]
    starts, ends, valid = [], [], []
    for row in df.itertuples(index=False):
        start, end = stage3_tartanvo_window(_row_frame_index(row))
        entry = segment_index.get(str(row.segment_key))
        starts.append(start)
        ends.append(end)
        valid.append(entry is not None and start >= 0 and end <= int(entry["num_pairs"]))
    df["tartanvo_window_start"] = starts
    df["tartanvo_window_end"] = ends
    df["tartanvo_valid"] = valid
    return df


def trace_split(split: str, stride: int, limit: int | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = _manifest_with_validity(split)
    strided = _stride_manifest(df, stride)
    after_cache = strided[strided.cache_available].reset_index(drop=True)
    after_boundary = after_cache[after_cache.tartanvo_valid].reset_index(drop=True)
    final = _limit_df(after_boundary, limit)
    before_stride_boundary = df[df.tartanvo_valid].reset_index(drop=True)

    print(f"[{split} Stage3 comma sample pipeline]")
    print(f"manifest: {active_manifest_path(split)}")
    print(f"full subset manifest rows: {len(df)}")
    print(f"cache availability before stride: {int(df.cache_available.sum())}")
    print(f"TartanVO boundary-valid before stride: {len(before_stride_boundary)}")
    print(f"after temporal stride ({stride}) [actual current order]: {len(strided)}")
    print(f"after cache availability: {len(after_cache)}")
    print(f"after boundary filter: {len(after_boundary)}")
    print(f"after sample limit ({limit}): {len(final)}")
    print(f"final dataset len: {len(final)}")
    print(f"final route count: {final.route_id.nunique() if 'route_id' in final else 'n/a'}")
    print(f"final segment count: {final.segment_key.nunique()}")
    _print_delta(f"[{split} class distribution delta vs boundary-valid full]", before_stride_boundary, final)
    return before_stride_boundary, final


def _collate_features(batch: list[dict]) -> torch.Tensor:
    return torch.stack([item["feature"] for item in batch])


def benchmark(train_stride: int, val_stride: int, train_limit: int | None, val_limit: int | None, batches: int, num_workers: int, shuffle: bool) -> None:
    import src.datasets.stage3_tartanvo_pose as dataset_module

    dataset_module.STAGE3_TRAIN_TEMPORAL_STRIDE = train_stride
    dataset_module.STAGE3_VAL_TEMPORAL_STRIDE = val_stride
    train = Stage3TartanFeatureDataset("train", C.STAGE3_TARTANVO_FEATURE, C.STAGE3_TARTANVO_FEATURE_CACHE, "comma2k19", train_limit)
    val = Stage3TartanFeatureDataset("val", C.STAGE3_TARTANVO_FEATURE, C.STAGE3_TARTANVO_FEATURE_CACHE, "comma2k19", val_limit)
    print("[dataset summary]")
    print(f"Train samples: {len(train)}")
    print(f"Val samples: {len(val)}")
    print(f"batch size: {C.BATCH_SIZE}")
    print(f"train iterations/epoch: {(len(train) + C.BATCH_SIZE - 1) // C.BATCH_SIZE}")
    print(f"val iterations: {(len(val) + C.BATCH_SIZE - 1) // C.BATCH_SIZE}")
    print(f"train segments: {train.df.segment_key.nunique()}")
    print(f"val segments: {val.df.segment_key.nunique()}")

    hit_miss = Counter()
    if num_workers == 0:
        original = train._load_segment

        def counted(segment_key: str):
            hit_miss["hit" if segment_key in train._segment_cache else "miss"] += 1
            return original(segment_key)

        train._load_segment = counted

    loader = DataLoader(train, batch_size=C.BATCH_SIZE, shuffle=shuffle, num_workers=num_workers, pin_memory=False, collate_fn=_collate_features)
    t0 = time.perf_counter()
    seen_batches = seen_samples = 0
    for features in loader:
        seen_batches += 1
        seen_samples += int(features.shape[0])
        if seen_batches >= batches:
            break
    elapsed = max(1e-9, time.perf_counter() - t0)
    print("[DataLoader benchmark: dataset loading only, no GPU forward]")
    print(f"requested batches: {batches}")
    print(f"actual batches: {seen_batches}")
    print(f"samples: {seen_samples}")
    print(f"seconds: {elapsed:.3f}")
    print(f"batches/sec: {seen_batches / elapsed:.2f}")
    print(f"samples/sec: {seen_samples / elapsed:.2f}")
    if num_workers == 0:
        total = max(1, hit_miss["hit"] + hit_miss["miss"])
        print(f"segment cache hit: {hit_miss['hit']} ({hit_miss['hit'] / total:.4f})")
        print(f"segment cache miss: {hit_miss['miss']} ({hit_miss['miss'] / total:.4f})")
    else:
        print("segment cache hit/miss: unavailable with worker subprocesses; rerun with --num-workers 0")


def _limit_arg(value: str) -> int | None:
    return None if value.lower() in {"none", "null", "0"} else int(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Stage3 comma2k19 cached TartanVO sampling without training.")
    parser.add_argument("--train-stride", type=int, default=C.STAGE3_TRAIN_TEMPORAL_STRIDE)
    parser.add_argument("--val-stride", type=int, default=C.STAGE3_VAL_TEMPORAL_STRIDE)
    parser.add_argument("--train-limit", type=_limit_arg, default=C.STAGE3_COMMA_TRAIN_SAMPLE_LIMIT)
    parser.add_argument("--val-limit", type=_limit_arg, default=C.STAGE3_COMMA_VAL_SAMPLE_LIMIT)
    parser.add_argument("--candidate", nargs=2, action="append", metavar=("TRAIN_STRIDE", "VAL_STRIDE"), help="Extra candidate stride pair to trace.")
    parser.add_argument("--benchmark-batches", type=int, default=300)
    parser.add_argument("--num-workers", type=int, default=C.STAGE3_NUM_WORKERS)
    parser.add_argument("--no-shuffle", action="store_true")
    args = parser.parse_args()

    print("[config]")
    for name in (
        "STAGE3_SAMPLE_PROFILE",
        "STAGE3_COMMA_SUBSET_MODE",
        "STAGE3_COMMA_TRAIN_SEGMENTS_PER_ROUTE",
        "STAGE3_COMMA_VAL_SEGMENTS_PER_ROUTE",
        "STAGE3_TRAIN_TEMPORAL_STRIDE",
        "STAGE3_VAL_TEMPORAL_STRIDE",
        "STAGE3_COMMA_TRAIN_SAMPLE_LIMIT",
        "STAGE3_COMMA_VAL_SAMPLE_LIMIT",
        "STAGE3_SEGMENT_FEATURE_LRU_SIZE",
        "STAGE3_TARTANVO_MODE",
        "STAGE3_TARTANVO_UNFREEZE",
        "STAGE3_TARTANVO_USE_FEATURE_CACHE",
    ):
        print(f"{name}: {getattr(C, name, None)}")
    print("TartanVO encoder forward in training: not used for cached feature batches; Stage3 train calls model.forward_feature(batch['feature']).")

    trace_split("train", args.train_stride, args.train_limit)
    trace_split("val", args.val_stride, args.val_limit)
    for item in args.candidate or []:
        train_stride, val_stride = map(int, item)
        print(f"[candidate train_stride={train_stride} val_stride={val_stride}]")
        _, train_final = trace_split("train", train_stride, None)
        _, val_final = trace_split("val", val_stride, None)
        print(f"candidate final train={len(train_final)} val={len(val_final)}")
    benchmark(args.train_stride, args.val_stride, args.train_limit, args.val_limit, args.benchmark_batches, args.num_workers, not args.no_shuffle)


if __name__ == "__main__":
    main()