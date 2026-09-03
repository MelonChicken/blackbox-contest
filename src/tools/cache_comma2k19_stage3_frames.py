from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.config import (
    COMMA2K19_STAGE3_FRAME_CACHE,
    COMMA2K19_STAGE3_RAW,
    COMMA2K19_STAGE3_TRAIN_MANIFEST,
    DEVICE,
    STAGE3_FRAME_CACHE_JPEG_QUALITY,
    STAGE3_NUM_FRAMES,
    STAGE3_TRAIN_TEMPORAL_STRIDE,
)
from src.datasets.comma2k19_stage3 import Comma2k19Stage3Dataset
from src.models import Stage3MViT
from src.train.stage3 import _loss, _stride_manifest


def _video_path(raw_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else raw_root / path


def _clip_indices(center: int, total: int, frames: int = STAGE3_NUM_FRAMES) -> np.ndarray:
    return np.clip(int(center) - frames // 2 + np.arange(frames), 0, total - 1).astype(int)


def _needed_frames(df: pd.DataFrame, frame_count: int) -> np.ndarray:
    needed = set()
    for frame_index in df.frame_index.astype(int):
        needed.update(_clip_indices(frame_index, frame_count).tolist())
    return np.asarray(sorted(needed), dtype=np.int64)


def _timestamps(df: pd.DataFrame, needed: np.ndarray) -> np.ndarray:
    source = df.sort_values("frame_index").drop_duplicates("frame_index")
    return np.interp(needed, source.frame_index.to_numpy(dtype=float), source.timestamp.to_numpy(dtype=float))


def _cache_dir(cache_root: Path, group: pd.DataFrame, video: Path) -> Path:
    first = group.iloc[0]
    if "route_id" in group.columns and "segment_id" in group.columns:
        return cache_root / str(first.route_id) / str(first.segment_id)
    return cache_root / Path(str(first.video_path)).with_suffix("")


def cache_segment(group: pd.DataFrame, raw_root: Path, cache_root: Path, jpeg_quality: int) -> tuple[int, Path]:
    video = _video_path(raw_root, str(group.iloc[0].video_path))
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {video}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        raise ValueError(f"invalid frame count: {video}")

    needed = _needed_frames(group, frame_count)
    need_set = set(needed.tolist())
    out_dir = _cache_dir(cache_root, group, video)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    ts = dict(zip(needed.tolist(), _timestamps(group, needed).tolist()))
    saved = 0
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx in need_set:
                name = f"{idx:06d}.jpg"
                ok = cv2.imwrite(str(out_dir / name), frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
                if not ok:
                    raise ValueError(f"cannot write cached frame: {out_dir / name}")
                rows.append({"original_frame_index": idx, "cached_path": name, "timestamp": float(ts[idx])})
                saved += 1
            idx += 1
    finally:
        cap.release()

    pd.DataFrame(rows).to_csv(out_dir / "frames.csv", index=False)
    return saved, out_dir


def cache_manifest(manifest: Path, raw_root: Path, cache_root: Path, stride: int, limit_segments: int | None, jpeg_quality: int) -> list[tuple[int, Path]]:
    df = _stride_manifest(pd.read_csv(manifest), stride)
    key = "segment_id" if "segment_id" in df.columns else "video_path"
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
    model = Stage3MViT().to(DEVICE).eval()
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
