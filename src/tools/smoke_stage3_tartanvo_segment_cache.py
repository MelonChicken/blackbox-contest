from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from src.config import (
    COMMA2K19_STAGE3_RAW,
    COMMA2K19_STAGE3_TRAIN_MANIFEST,
    DEVICE,
    S3_MEAN,
    S3_STD,
    STAGE3_TARTANVO_CACHE_PAIR_BATCH_SIZE,
    STAGE3_TARTANVO_FEATURE_CACHE,
)
from src.datasets.comma2k19_stage3 import Comma2k19Stage3Dataset
from src.datasets.stage3_tartanvo_pose import Stage3TartanFeatureDataset
from src.models import Stage3TartanVOGRU
from src.tools.cache_stage3_tartanvo_feature import cache_segment_split
from src.train.stage3 import _loader
from src.utils import _crop_tensor, video_frames


def _video_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else COMMA2K19_STAGE3_RAW / path


def _legacy_indices(center: int, total: int) -> list[int]:
    return [min(max(center - 8 + i, 0), total - 1) for i in range(16)]


def _direct_clip(video_path: Path, indices: list[int]) -> torch.Tensor:
    frames = video_frames(video_path)
    return torch.stack([_crop_tensor(frames[i]) for i in indices], dim=1)


def _normalized(raw_clip: torch.Tensor) -> torch.Tensor:
    return (raw_clip - S3_MEAN[:, None, :, :]) / S3_STD[:, None, :, :]


def _vonet_inputs(model: Stage3TartanVOGRU, normalized_clip: torch.Tensor, pair: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    frames = (normalized_clip.unsqueeze(0).to(DEVICE) * model.s3_std + model.s3_mean).clamp(0.0, 1.0)
    b, c, t, h, w = frames.shape
    resized = F.interpolate(frames.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w), size=(model.tartanvo.height, model.tartanvo.width), mode="bilinear", align_corners=False)
    resized = resized.reshape(b, t, c, model.tartanvo.height, model.tartanvo.width)
    img1 = resized[:, pair].contiguous()
    img2 = resized[:, pair + 1].contiguous()
    intrinsic = model.tartanvo._intrinsic(1, img1.device, img1.dtype)
    return img1.cpu(), img2.cpu(), intrinsic.cpu()


def _describe_tensor(name: str, x: torch.Tensor) -> None:
    print(name, "shape", tuple(x.shape), "dtype", str(x.dtype), "min", float(x.min()), "max", float(x.max()), "mean", float(x.mean()))


def _direct_feature(model: Stage3TartanVOGRU, clip: torch.Tensor, pair_batch_size: int | None = None) -> torch.Tensor:
    video = _normalized(clip).unsqueeze(0).to(DEVICE)
    if pair_batch_size is None:
        return model.feature_sequence(video).squeeze(0).cpu()
    outs = []
    raw = (video * model.s3_std + model.s3_mean).clamp(0.0, 1.0).squeeze(0).cpu()
    for start in range(0, raw.shape[1] - 1, pair_batch_size):
        chunk = raw[:, start : min(start + pair_batch_size + 1, raw.shape[1])]
        norm = _normalized(chunk).unsqueeze(0).to(DEVICE)
        outs.append(model.feature_sequence(norm).squeeze(0).cpu())
    return torch.cat(outs, dim=0)


def _print_diff(name: str, a: torch.Tensor, b: torch.Tensor) -> None:
    d = (a - b).abs()
    print(name, "max_abs_diff", float(d.max()), "mean_abs_diff", float(d.mean()), "allclose", bool(torch.allclose(a, b, atol=1e-4, rtol=1e-4)))


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
    row = ds.df.iloc[0]
    center = int(row.video_frame_index)
    start, end = int(row.tartanvo_window_start), int(row.tartanvo_window_end)
    total = int(ds.segment_index[str(row.segment_key)]["num_frames"])
    frame_indices = _legacy_indices(center, total)
    pair_indices = list(range(start, end))
    assert frame_indices == list(range(center - 8, center + 8))
    assert pair_indices == list(range(center - 8, center + 7))

    video_path = _video_path(str(row.video_path))
    raw_direct = Comma2k19Stage3Dataset(args.manifest, cache_root=None)
    raw_direct.df = pd.DataFrame([row]).reset_index(drop=True)
    raw_default = Comma2k19Stage3Dataset(args.manifest)
    raw_default.df = pd.DataFrame([row]).reset_index(drop=True)
    default_cache_dir = raw_default._cache_dir(row)

    model = Stage3TartanVOGRU(load_pretrained=True, feature="latent").to(DEVICE).eval()
    batch = next(iter(_loader(ds, shuffle=False)))
    print("model_training", bool(model.training), "tartanvo_training", bool(model.tartanvo.training), "vonet_training", bool(model.tartanvo.vonet.training))

    with torch.inference_mode():
        direct_clip = raw_direct[0]["video"]
        default_clip = raw_default[0]["video"]
        segment_raw_clip = _direct_clip(video_path, frame_indices)
        segment_clip = _normalized(segment_raw_clip)
        direct_img1, direct_img2, direct_intrinsic = _vonet_inputs(model, direct_clip)
        segment_img1, segment_img2, segment_intrinsic = _vonet_inputs(model, segment_clip)
        legacy = model.feature_sequence(direct_clip.unsqueeze(0).to(DEVICE)).squeeze(0).cpu()
        segment_bs1 = _direct_feature(model, segment_raw_clip, 1)
        segment_bs16 = _direct_feature(model, segment_raw_clip, 16)
        accel, steer = model.forward_feature(batch["feature"].to(DEVICE))

    print("center_frame", center)
    print("legacy_frame_indices", frame_indices)
    print("legacy_pair_indices", list(range(frame_indices[0], frame_indices[-1])))
    print("segment_slice_start_end", start, end)
    print("segment_pair_indices", pair_indices)
    print("legacy_default_frame_source", "jpeg_frame_cache" if default_cache_dir else "direct_video")
    print("input_channel_order", "RGB")
    print("direct_input_normalized", True, "segment_input_normalized", True)
    _describe_tensor("direct_final_img1", direct_img1)
    _describe_tensor("segment_final_img1", segment_img1)
    _describe_tensor("direct_final_intrinsic", direct_intrinsic)
    _describe_tensor("segment_final_intrinsic", segment_intrinsic)
    if default_cache_dir:
        _print_diff("default_cached_normalized_clip_vs_direct_normalized_clip", default_clip, direct_clip)
    _print_diff("direct_normalized_clip_vs_segment_normalized_clip", direct_clip, segment_clip)
    _print_diff("direct_final_img1_vs_segment_final_img1", direct_img1, segment_img1)
    _print_diff("direct_final_img2_vs_segment_final_img2", direct_img2, segment_img2)
    _print_diff("direct_final_intrinsic_vs_segment_final_intrinsic", direct_intrinsic, segment_intrinsic)
    _print_diff("legacy_vs_segment_cache", legacy, sample["feature"])
    _print_diff("legacy_vs_segment_bs1", legacy, segment_bs1)
    _print_diff("legacy_vs_segment_bs16", legacy, segment_bs16)
    _print_diff("segment_bs1_vs_segment_bs16", segment_bs1, segment_bs16)
    for i, d in enumerate((legacy - sample["feature"]).abs()):
        print(f"pair {i:02d} mean_abs_diff {float(d.mean())} max_abs_diff {float(d.max())}")

    pairs = int(index.num_pairs.sum())
    size_mb = sum(p.stat().st_size for p in (base / args.split).glob("*.pt")) / 1024 / 1024
    print("segment smoke ok")
    print("window old: center-8..center+7 frames, clamped only in legacy boundary cases")
    print("window new: pair indices center-8:center+7, invalid samples excluded")
    print("segments", len(index), "pairs", pairs, "feature_dim", 1536)
    print("cache_size_mb", f"{size_mb:.2f}")
    print("dataset_batch_shape", tuple(batch["feature"].shape))
    print("model_outputs", tuple(accel.shape), tuple(steer.shape))
    print("seconds_per_segment", f"{elapsed / max(1, len(index)):.3f}")
    print("seconds_per_pair", f"{elapsed / max(1, pairs):.5f}")
    print("pairs_per_second", f"{pairs / max(elapsed, 1e-9):.2f}")


if __name__ == "__main__":
    main()
