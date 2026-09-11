from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import COMMA2K19_STAGE3_RAW, STAGE3_TARTANVO_FEATURE, STAGE3_TARTANVO_FEATURE_CACHE
from src.tools.build_comma2k19_stage3_manifest import ALIGNMENT_VERSION, _direct_video, _frame_time_arrays, _segment_dirs, _series, _video_timing


def _summary(name: str, t) -> dict:
    arr = np.asarray(t, dtype=float).squeeze()
    if len(arr) == 0:
        return {f"{name}_start": float("nan"), f"{name}_end": float("nan"), f"{name}_duration": 0.0, f"{name}_samples": 0}
    return {
        f"{name}_start": float(arr[0]),
        f"{name}_end": float(arr[-1]),
        f"{name}_duration": float(arr[-1] - arr[0]),
        f"{name}_samples": int(len(arr)),
    }


def _best_pose_times(segment: Path, decoded_frames: int):
    arrays = _frame_time_arrays(segment)
    exact = [(name, values) for name, values in arrays if len(values) == decoded_frames]
    if exact:
        return exact[0][0], exact[0][1]
    return (arrays[0] if arrays else ("missing", np.asarray([], dtype=float)))


def _cadence_summary(prefix: str, t) -> dict:
    arr = np.asarray(t, dtype=float).squeeze()
    if len(arr) < 2:
        return {f"{prefix}_diff_mean": float("nan"), f"{prefix}_diff_median": float("nan"), f"{prefix}_diff_p5": float("nan"), f"{prefix}_diff_p95": float("nan")}
    diff = np.diff(arr)
    return {
        f"{prefix}_diff_mean": float(np.mean(diff)),
        f"{prefix}_diff_median": float(np.median(diff)),
        f"{prefix}_diff_p5": float(np.percentile(diff, 5)),
        f"{prefix}_diff_p95": float(np.percentile(diff, 95)),
    }


def _segment_row(segment: Path) -> dict:
    video = _direct_video(segment)
    timing = _video_timing(video)
    speed_t, _ = _series(segment, "speed")
    steer_t, _ = _series(segment, "steering_angle")
    pose_source, pose_t = _best_pose_times(segment, timing.decoded_frame_count)
    row = {
        "segment": str(segment),
        "route_id": segment.parent.name,
        "segment_id": segment.name,
        "video_reported_fps": timing.reported_fps,
        "video_time_base": timing.time_base,
        "video_first_pts": float(timing.pts_sec[0]),
        "video_last_pts": float(timing.pts_sec[-1]),
        "video_duration": timing.duration,
        "video_decoded_frames": timing.decoded_frame_count,
        "pose_source": pose_source,
        "pose_matches_decoded_frames": len(pose_t) == timing.decoded_frame_count,
    }
    row.update(_summary("speed", speed_t))
    row.update(_summary("steering", steer_t))
    row.update(_summary("pose", pose_t))
    row.update(_cadence_summary("pose", pose_t))
    return row


def _print_invalid_cache_report() -> None:
    print("\n[comma2k19 TartanVO cache validity]")
    base = STAGE3_TARTANVO_FEATURE_CACHE / STAGE3_TARTANVO_FEATURE / "comma2k19"
    for split in ("train", "val"):
        index = base / f"{split}_index.csv"
        meta = base / f"{split}_metadata.json"
        if not index.is_file():
            print(f"{split}: missing cache index ({index})")
            continue
        reason = f"metadata lacks manifest_alignment_version={ALIGNMENT_VERSION}"
        try:
            import json
            data = json.loads(meta.read_text(encoding="utf-8")) if meta.is_file() else {}
            if data.get("manifest_alignment_version") == ALIGNMENT_VERSION:
                print(f"{split}: cache appears aligned ({index})")
            else:
                print(f"{split}: INVALID until regenerated - {reason}: {index}")
        except Exception as exc:
            print(f"{split}: INVALID until regenerated - cannot read metadata {meta}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit comma2k19 CAN/video timestamp synchronization for Stage3.")
    parser.add_argument("--raw-root", type=Path, default=COMMA2K19_STAGE3_RAW)
    parser.add_argument("--segments", type=int, default=5)
    args = parser.parse_args()

    rows = []
    for segment in _segment_dirs(args.raw_root)[: args.segments]:
        try:
            rows.append(_segment_row(segment))
        except Exception as exc:
            rows.append({"segment": str(segment), "error": str(exc)})
    if not rows:
        print(f"no comma2k19 segment directories found under {args.raw_root}")
    else:
        df = pd.DataFrame(rows)
        cols = [
            "segment", "video_duration", "speed_duration", "steering_duration", "pose_duration",
            "video_first_pts", "video_last_pts", "speed_start", "speed_end", "steering_start", "steering_end",
            "pose_start", "pose_end", "video_reported_fps", "video_decoded_frames", "speed_samples",
            "steering_samples", "pose_samples", "pose_source", "pose_matches_decoded_frames",
            "pose_diff_mean", "pose_diff_median", "pose_diff_p5", "pose_diff_p95", "error",
        ]
        cols = [c for c in cols if c in df.columns]
        print(df[cols].to_string(index=False))
    _print_invalid_cache_report()


if __name__ == "__main__":
    main()