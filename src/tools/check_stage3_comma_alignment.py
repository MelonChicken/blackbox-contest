from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pandas as pd

from src.tools.stage3_comma_manifest import abs_video_path, frame_index_series, read_manifest

CHECK_TIMES_SEC = (0.0, 10.0, 30.0, 59.0)


@dataclass(frozen=True)
class VideoPtsIndex:
    reported_fps: float
    decoded_frame_count: int
    pts_sec: np.ndarray


def _rate_to_float(rate) -> float:
    if rate is None:
        return float("nan")
    if isinstance(rate, Fraction):
        return float(rate)
    return float(rate)


def _video_pts_index(path: Path) -> VideoPtsIndex:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        reported_fps = _rate_to_float(stream.average_rate)
        time_base = stream.time_base
        pts = []
        for i, frame in enumerate(container.decode(stream)):
            if frame.pts is not None and time_base is not None:
                pts.append(float(frame.pts * time_base))
            elif frame.time is not None:
                pts.append(float(frame.time))
            elif reported_fps == reported_fps and reported_fps > 0:
                pts.append(i / reported_fps)
            else:
                pts.append(float(i))
    return VideoPtsIndex(reported_fps=reported_fps, decoded_frame_count=len(pts), pts_sec=np.asarray(pts, dtype=float))


def _nearest_index(values: np.ndarray, target: float) -> int:
    if len(values) == 0:
        raise ValueError("video decoded zero frames")
    return int(np.argmin(np.abs(values - target)))


def _manifest_time_column(seg: pd.DataFrame) -> str:
    return "target_timestamp" if "target_timestamp" in seg.columns else "timestamp"


def _manifest_row_near(seg: pd.DataFrame, desired_time: float):
    col = _manifest_time_column(seg)
    return seg.iloc[int(np.argmin(np.abs(seg[col].to_numpy(float) - desired_time)))]


def _sample_index(timestamp: float) -> int:
    return int(round(timestamp * 10.0))


def _print_check(video_path: Path, seg: pd.DataFrame, pts_index: VideoPtsIndex, desired_time: float) -> None:
    row = _manifest_row_near(seg, desired_time)
    relative_timestamp = float(row.timestamp)
    target_timestamp = float(getattr(row, "target_timestamp", relative_timestamp))
    dataset_frame = int(row.video_frame_index if "video_frame_index" in seg.columns else row.frame_index)
    nearest_frame = _nearest_index(pts_index.pts_sec, relative_timestamp)
    selected_pts = float(pts_index.pts_sec[dataset_frame]) if 0 <= dataset_frame < len(pts_index.pts_sec) else float("nan")
    selected_clock = float(getattr(row, "video_frame_timestamp", selected_pts))
    nearest_pts = float(pts_index.pts_sec[nearest_frame])
    manifest_error = float(getattr(row, "alignment_error_sec", selected_clock - target_timestamp))
    sample_index = _sample_index(relative_timestamp)
    print(f"sample_index={sample_index}")
    print(f"target_timestamp={target_timestamp:.6f}")
    print(f"relative_timestamp={relative_timestamp:.6f}")
    print(f"video_reported_fps={pts_index.reported_fps:.6f}")
    print(f"decoded_frame_count={pts_index.decoded_frame_count}")
    print(f"selected_frame_index={dataset_frame}")
    print(f"selected_frame_pts_sec={selected_pts:.6f}")
    print(f"selected_frame_timestamp={selected_clock:.6f}")
    print(f"time_error_sec={manifest_error:.6f}")
    print(f"nearest_by_pts_frame_index={nearest_frame}")
    print(f"nearest_by_pts_sec={nearest_pts:.6f}")
    print(f"nearest_time_error_sec={nearest_pts - target_timestamp:.6f}")
    print(f"dataset_selected_nearest_by_pts={dataset_frame == nearest_frame}")
    print(f"expected_if_20fps={int(round(relative_timestamp * 20.0))}")
    print(f"expected_if_25fps={int(round(relative_timestamp * 25.0))}")
    print(f"video_path={video_path}")
    print()


def _video_groups(max_videos: int):
    frames = []
    for split in ("train", "val"):
        try:
            df = read_manifest(split)
        except FileNotFoundError as exc:
            print(exc)
            continue
        if ("timestamp" not in df.columns and "target_timestamp" not in df.columns) or ("frame_index" not in df.columns and "video_frame_index" not in df.columns):
            print(f"{split} manifest missing timestamp/target_timestamp or frame index columns")
            continue
        df = df.copy()
        df["_split"] = split
        frames.append(df)
    if not frames:
        return []
    df = pd.concat(frames, ignore_index=True)
    groups = []
    for _, seg in df.groupby("video_path", sort=False):
        video_path = abs_video_path(seg.iloc[0])
        if video_path.is_file():
            groups.append((video_path, seg.sort_values("timestamp").reset_index(drop=True)))
        if len(groups) >= max_videos:
            break
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(description="Check comma2k19 Stage3 frame alignment from decoded frame PTS.")
    parser.add_argument("--max-videos", type=int, default=5)
    parser.add_argument("--times", type=float, nargs="*", default=list(CHECK_TIMES_SEC))
    args = parser.parse_args()

    groups = _video_groups(max(5, args.max_videos))
    if len(groups) < 5:
        print(f"warning: only {len(groups)} readable comma2k19 videos found; expected at least 5")
    for video_no, (video_path, seg) in enumerate(groups[: args.max_videos], start=1):
        print(f"=== video {video_no} ===")
        print(f"segment path: {Path(str(seg.iloc[0].video_path)).parent}")
        pts_index = _video_pts_index(video_path)
        for desired_time in args.times:
            _print_check(video_path, seg, pts_index, float(desired_time))


if __name__ == "__main__":
    main()