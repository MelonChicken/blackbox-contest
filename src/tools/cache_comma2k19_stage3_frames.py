from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import (
    COMMA2K19_STAGE3_FRAME_CACHE,
    COMMA2K19_STAGE3_RAW,
    COMMA2K19_STAGE3_TRAIN_MANIFEST,
    DEVICE,
    STAGE3_FRAME_CACHE_JPEG_QUALITY,
    STAGE3_FRAME_CACHE_SIZE,
    STAGE3_NUM_FRAMES,
    STAGE3_TRAIN_TEMPORAL_STRIDE,
)
from src.datasets.comma2k19_stage3 import Comma2k19Stage3Dataset
from src.datasets.stage3_sampling import parse_clip_float_list, parse_clip_frame_indices
from src.models import Stage3MViT
from src.train.stage3 import _loss, _stride_manifest


def _resize_center_crop_bgr(frame: np.ndarray, size: int = STAGE3_FRAME_CACHE_SIZE) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = size / min(h, w)
    nh, nw = max(size, round(h * scale)), max(size, round(w * scale))
    frame = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    y, x = (nh - size) // 2, (nw - size) // 2
    return frame[y:y + size, x:x + size]


def _video_path(raw_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else raw_root / path


def _frame_index_column(df: pd.DataFrame) -> str:
    return "video_frame_index" if "video_frame_index" in df.columns else "frame_index"


def _timestamp_column(df: pd.DataFrame) -> str:
    return "target_timestamp" if "target_timestamp" in df.columns else "timestamp"


def _needed_frames(df: pd.DataFrame) -> np.ndarray:
    if "clip_frame_indices" not in df.columns:
        raise RuntimeError("manifest lacks clip_frame_indices; regenerate Stage3 manifest before caching")
    needed = set()
    for value in df["clip_frame_indices"]:
        needed.update(parse_clip_frame_indices(str(value), STAGE3_NUM_FRAMES))
    if not needed:
        raise RuntimeError("manifest contains no requested clip frames")
    return np.asarray(sorted(needed), dtype=np.int64)

def _timestamps(df: pd.DataFrame, needed: np.ndarray) -> np.ndarray:
    by_frame = {}
    if "clip_target_timestamps" in df.columns:
        for row in df.itertuples(index=False):
            indices = parse_clip_frame_indices(str(row.clip_frame_indices), STAGE3_NUM_FRAMES)
            timestamps = parse_clip_float_list(str(row.clip_target_timestamps), STAGE3_NUM_FRAMES, "clip_target_timestamps")
            for idx, ts in zip(indices, timestamps):
                by_frame.setdefault(int(idx), float(ts))
    frame_col = _frame_index_column(df)
    time_col = _timestamp_column(df)
    source = df.sort_values(frame_col).drop_duplicates(frame_col)
    fallback = dict(zip(source[frame_col].to_numpy(dtype=int).tolist(), source[time_col].to_numpy(dtype=float).tolist()))
    return np.asarray([by_frame.get(int(idx), fallback.get(int(idx), float("nan"))) for idx in needed], dtype=float)

def _cache_dir(cache_root: Path, group: pd.DataFrame, video: Path) -> Path:
    first = group.iloc[0]
    if "route_id" in group.columns and "segment_id" in group.columns:
        return cache_root / str(first.route_id) / str(first.segment_id)
    return cache_root / Path(str(first.video_path)).with_suffix("")


def cache_segment(group: pd.DataFrame, raw_root: Path, cache_root: Path, jpeg_quality: int) -> tuple[int, Path]:
    video = _video_path(raw_root, str(group.iloc[0].video_path))
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video}")

    reported_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    needed = _needed_frames(group)
    required_set = set(needed.tolist())
    max_required = int(needed[-1])
    out_dir = _cache_dir(cache_root, group, video)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    ts = dict(zip(needed.tolist(), _timestamps(group, needed).tolist()))
    saved = 0
    existing = 0
    decoded = 0
    idx = 0
    try:
        while idx <= max_required:
            ok, frame = cap.read()
            if not ok:
                break
            decoded += 1
            if idx in required_set:
                name = f"{idx:06d}.jpg"
                out_path = out_dir / name
                if out_path.is_file():
                    existing += 1
                else:
                    ok = cv2.imwrite(str(out_path), _resize_center_crop_bgr(frame), [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
                    if not ok:
                        raise ValueError(f"cannot write cached frame: {out_path}")
                    saved += 1
                rows.append({"original_frame_index": idx, "cached_path": name, "timestamp": float(ts[idx])})
            idx += 1
    finally:
        cap.release()

    missing = sorted(required_set - {int(row["original_frame_index"]) for row in rows})
    if saved + existing == 0:
        raise RuntimeError(f"no required frames decoded from {video}")
    pd.DataFrame(rows).to_csv(out_dir / "frames.csv", index=False)
    print(
        f"segment_cache_stats requested={len(needed)} decoded={decoded} saved={saved} existing={existing} "
        f"missing={len(missing)} reported_frame_count={reported_count}"
    )
    if missing:
        print(f"missing_required_frame_indices={missing[:20]}{'...' if len(missing) > 20 else ''}")
    return saved, out_dir


def cache_manifest(manifest: Path, raw_root: Path, cache_root: Path, stride: int, limit_segments: int | None, jpeg_quality: int) -> list[tuple[int, Path]]:
    df = _stride_manifest(pd.read_csv(manifest), stride)
    key = ["route_id", "segment_id"] if {"route_id", "segment_id"}.issubset(df.columns) else "segment_id" if "segment_id" in df.columns else "video_path"
    groups = [part for _, part in df.groupby(key, sort=False)]
    if limit_segments:
        groups = groups[:limit_segments]
    outputs = []
    for group in groups:
        try:
            saved, out_dir = cache_segment(group, raw_root, cache_root, jpeg_quality)
            outputs.append((saved, out_dir))
            print(f"cached {saved} frames -> {out_dir}")
        except Exception as exc:
            warnings.warn(f"skip cache segment {group.iloc[0].video_path}: {exc}")
    return outputs


def _take(dataset: Comma2k19Stage3Dataset, n: int) -> Comma2k19Stage3Dataset:
    dataset.df = dataset.df.head(n).reset_index(drop=True)
    return dataset


def benchmark(manifest: Path, raw_root: Path, cache_root: Path, samples: int, stride: int) -> None:
    hevc = Comma2k19Stage3Dataset(manifest, root=raw_root, cache_root=None)
    cached = Comma2k19Stage3Dataset(manifest, root=raw_root, cache_root=cache_root)
    hevc.df = _stride_manifest(hevc.df, stride)
    cached.df = _stride_manifest(cached.df, stride)
    hevc = _take(hevc, samples)
    cached = _take(cached, samples)

    def run(name: str, dataset) -> float:
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        start = time.perf_counter()
        count = 0
        for _ in loader:
            count += 1
        elapsed = time.perf_counter() - start
        print(f"{name}:")
        print(f"{count} samples = {elapsed:.3f} sec")
        print(f"samples/sec = {count / elapsed:.3f}")
        return elapsed

    hevc_time = run("HEVC", hevc)
    cached_time = run("Cached", cached)
    print(f"speedup = {hevc_time / cached_time:.2f} x")


def smoke(manifest: Path, raw_root: Path, cache_root: Path, stride: int) -> None:
    dataset = Comma2k19Stage3Dataset(manifest, root=raw_root, cache_root=cache_root)
    dataset.df = _stride_manifest(dataset.df, stride)
    dataset = _take(dataset, 1)
    batch = next(iter(DataLoader(dataset, batch_size=1, num_workers=0)))
    model = Stage3MViT(pretrained=False).to(DEVICE).eval()
    with torch.inference_mode():
        accel, steer = model(batch["video"].to(DEVICE))
        loss, accel_loss, steer_loss = _loss(accel, steer, batch)
    print("smoke ok")
    print("video_shape", tuple(batch["video"].shape))
    print("loss", float(loss.cpu()), float(accel_loss.cpu()), float(steer_loss.cpu()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=COMMA2K19_STAGE3_TRAIN_MANIFEST)
    parser.add_argument("--raw-root", type=Path, default=COMMA2K19_STAGE3_RAW)
    parser.add_argument("--cache-root", type=Path, default=COMMA2K19_STAGE3_FRAME_CACHE)
    parser.add_argument("--temporal-stride", type=int, default=STAGE3_TRAIN_TEMPORAL_STRIDE)
    parser.add_argument("--limit-segments", type=int, default=1)
    parser.add_argument("--jpeg-quality", type=int, default=STAGE3_FRAME_CACHE_JPEG_QUALITY)
    parser.add_argument("--benchmark-samples", type=int, default=100)
    parser.add_argument("--skip-benchmark", action="store_true")
    args = parser.parse_args()

    outputs = cache_manifest(args.manifest, args.raw_root, args.cache_root, args.temporal_stride, args.limit_segments, args.jpeg_quality)
    if not outputs:
        raise RuntimeError("no segments were cached")
    total_size = sum(p.stat().st_size for _, out_dir in outputs for p in out_dir.rglob("*") if p.is_file())
    print(f"cached_frames = {sum(saved for saved, _ in outputs)}")
    print(f"cache_size_mb = {total_size / 1024 / 1024:.2f}")
    smoke(args.manifest, args.raw_root, args.cache_root, args.temporal_stride)
    if not args.skip_benchmark:
        benchmark(args.manifest, args.raw_root, args.cache_root, args.benchmark_samples, args.temporal_stride)


if __name__ == "__main__":
    main()

