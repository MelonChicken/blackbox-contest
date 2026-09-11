from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np
import pandas as pd

from src.config import COMMA2K19_STAGE3_MANIFEST, COMMA2K19_STAGE3_RAW, STAGE3_ACCEL_LABEL_MODE, STAGE3_OUTPUT_HZ
from src.datasets.stage3_labels import ACCEL_NAMES, STEER_NAMES, derive_accel_label, derive_acceleration, derive_steer_label

VIDEO_EXT = {".hevc", ".mp4", ".mkv", ".avi", ".mov"}
ALIGNMENT_VERSION = "comma_frame_times_nearest_v1"
DEFAULT_MAX_ALIGNMENT_ERROR_SEC = 0.06


@dataclass(frozen=True)
class VideoTiming:
    reported_fps: float
    pts_sec: np.ndarray
    time_base: str

    @property
    def decoded_frame_count(self) -> int:
        return int(len(self.pts_sec))

    @property
    def duration(self) -> float:
        return float(self.pts_sec[-1] - self.pts_sec[0]) if len(self.pts_sec) else 0.0


@dataclass(frozen=True)
class AlignmentStats:
    candidates: int
    valid: int
    dropped_no_video_frame: int
    errors: tuple[float, ...]


def _load_array(path: Path) -> np.ndarray:
    return np.asarray(np.load(path, allow_pickle=False)).squeeze()


def _series_one(segment: Path, name: str) -> tuple[np.ndarray, np.ndarray] | None:
    base = segment / "processed_log" / "CAN" / name
    for t_name in ("t", "t.npy"):
        for v_name in ("value", "value.npy"):
            t_path, v_path = base / t_name, base / v_name
            if t_path.is_file() and v_path.is_file():
                return _load_array(t_path).astype(float), _load_array(v_path).astype(float)
    flat_t = next(segment.rglob(f"processed_log__CAN__{name}__t*"), None)
    flat_v = next(segment.rglob(f"processed_log__CAN__{name}__value*"), None)
    if flat_t and flat_v:
        return _load_array(flat_t).astype(float), _load_array(flat_v).astype(float)
    return None


def _series(segment: Path, name: str) -> tuple[np.ndarray, np.ndarray]:
    names = (name, "car_speed") if name == "speed" else (name,)
    for candidate in names:
        found = _series_one(segment, candidate)
        if found is not None:
            return found
    raise FileNotFoundError(f"missing processed_log/CAN/{name}/{{t,value}} under {segment}")


def _direct_video(segment: Path) -> Path:
    videos = sorted(p for p in segment.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXT)
    if not videos:
        raise FileNotFoundError("missing video file")
    return videos[0]


def _video_timing(video: Path) -> VideoTiming:
    pts = []
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate is not None else float("nan")
        time_base = str(stream.time_base)
        for i, frame in enumerate(container.decode(stream)):
            if frame.pts is not None and stream.time_base is not None:
                pts.append(float(frame.pts * stream.time_base))
            elif frame.time is not None:
                pts.append(float(frame.time))
            elif fps == fps and fps > 0:
                pts.append(i / fps)
            else:
                pts.append(float(i))
    if not pts:
        raise ValueError(f"video decoded zero frames: {video}")
    return VideoTiming(reported_fps=fps, pts_sec=np.asarray(pts, dtype=float), time_base=time_base)


def _frame_time_arrays(segment: Path) -> list[tuple[str, np.ndarray]]:
    rels = (
        "global_pose/frame_times",
        "global_pose/frame_times.npy",
        "global_pos/frame_times",
        "global_pos/frame_times.npy",
        "frame_times",
        "frame_times.npy",
    )
    out = []
    for rel in rels:
        path = segment / rel
        if path.is_file():
            out.append((rel, _load_array(path).astype(float)))
    for path in sorted(segment.rglob("*frame_times*.npy")):
        rel = str(path.relative_to(segment))
        if rel not in {name for name, _ in out}:
            out.append((rel, _load_array(path).astype(float)))
    return out


def _video_clock(segment: Path, timing: VideoTiming) -> tuple[np.ndarray, str]:
    for name, values in _frame_time_arrays(segment):
        values = np.asarray(values, dtype=float).squeeze()
        if len(values) == timing.decoded_frame_count:
            return values, name
    raise FileNotFoundError(
        f"no frame timestamp array with decoded frame count={timing.decoded_frame_count} under {segment}; "
        "refusing to use HEVC PTS as the synchronization clock"
    )



def _target_grid(start: float, end: float) -> np.ndarray:
    hz = float(STAGE3_OUTPUT_HZ)
    first = np.ceil(start * hz - 1e-9) / hz
    if first > end:
        return np.empty(0, dtype=float)
    return np.arange(first, end + 1e-9, 1.0 / hz, dtype=float)


def _rows(
    video_path: str,
    video_clock_t: np.ndarray,
    video_pts_sec: np.ndarray,
    speed_t,
    speed_v,
    steer_t,
    steer_v,
    invert_steering: bool,
    alignment_source: str,
    max_alignment_error_sec: float = DEFAULT_MAX_ALIGNMENT_ERROR_SEC,
) -> tuple[list[dict], AlignmentStats]:
    video_clock_t = np.asarray(video_clock_t, dtype=float).squeeze()
    video_pts_sec = np.asarray(video_pts_sec, dtype=float).squeeze()
    speed_t = np.asarray(speed_t, dtype=float).squeeze()
    steer_t = np.asarray(steer_t, dtype=float).squeeze()
    speed_v = np.asarray(speed_v, dtype=float).squeeze()
    steer_v = np.asarray(steer_v, dtype=float).squeeze()
    start = max(float(speed_t[0]), float(steer_t[0]))
    end = min(float(speed_t[-1]), float(steer_t[-1]))
    target_t = _target_grid(start, end)
    if len(target_t) == 0:
        return [], AlignmentStats(0, 0, 0, tuple())

    speed = np.interp(target_t, speed_t, speed_v)
    steering = np.interp(target_t, steer_t, steer_v)
    acceleration = derive_acceleration(speed, STAGE3_ACCEL_LABEL_MODE)
    accel_label = derive_accel_label(speed, acceleration)
    steer_label = derive_steer_label(steering, invert_steering)
    errors = []
    rows = []
    dropped = 0
    video_start = float(video_clock_t[0])
    for i, target in enumerate(target_t):
        if target < float(video_clock_t[0]) or target > float(video_clock_t[-1]):
            dropped += 1
            continue
        frame_index = int(np.argmin(np.abs(video_clock_t - target)))
        err = abs(float(video_clock_t[frame_index] - target))
        if err > max_alignment_error_sec:
            dropped += 1
            continue
        errors.append(err)
        rows.append(
            {
                "video_path": video_path,
                "sample_index": int(round((target - video_start) * STAGE3_OUTPUT_HZ)),
                "target_timestamp": float(target),
                "video_frame_index": frame_index,
                "video_frame_timestamp": float(video_clock_t[frame_index]),
                "video_pts_sec": float(video_pts_sec[frame_index]),
                "alignment_error_sec": err,
                "alignment_source": alignment_source,
                "alignment_version": ALIGNMENT_VERSION,
                "speed": float(speed[i]),
                "acceleration": float(acceleration[i]),
                "steering_angle": float(steering[i]),
                "accel_label": int(accel_label[i]),
                "steer_label": int(steer_label[i]),
            }
        )
    return rows, AlignmentStats(len(target_t), len(rows), dropped, tuple(errors))


def _segment_dirs(raw_root: Path) -> list[Path]:
    chunks = sorted(p for p in raw_root.glob("Chunk_*") if p.is_dir())
    segments = []
    for chunk in chunks:
        for route in sorted(p for p in chunk.iterdir() if p.is_dir()):
            for segment in sorted(p for p in route.iterdir() if p.is_dir()):
                if (segment / "video.hevc").is_file():
                    segments.append(segment)
    return segments


def _local_rows(segment: Path, raw_root: Path, invert_steering: bool) -> tuple[list[dict], AlignmentStats]:
    video = _direct_video(segment)
    timing = _video_timing(video)
    video_clock_t, alignment_source = _video_clock(segment, timing)
    speed_t, speed_v = _series(segment, "speed")
    steer_t, steer_v = _series(segment, "steering_angle")
    rel_video = video.relative_to(raw_root) if video.is_relative_to(raw_root) else video
    rows, stats = _rows(str(rel_video), video_clock_t, timing.pts_sec, speed_t, speed_v, steer_t, steer_v, invert_steering, alignment_source)
    route_id = segment.parent.name
    segment_id = segment.name
    for row in rows:
        row["route_id"] = route_id
        row["segment_id"] = segment_id
        row["source"] = "comma2k19"
        row["video_reported_fps"] = timing.reported_fps
        row["video_decoded_frames"] = timing.decoded_frame_count
        row["video_duration_sec"] = timing.duration
    return rows, stats


def _hf_video_path(row: dict, video_dir: Path) -> str:
    video = row["video"]
    if isinstance(video, dict) and video.get("path"):
        return str(Path(video["path"]))
    segment_id = row.get("segment_id", str(len(list(video_dir.glob("*.hevc")))))
    out = video_dir / f"{str(segment_id).replace('/', '__').replace('|', '_')}.hevc"
    data = video.get("bytes") if isinstance(video, dict) else video
    if data is None:
        raise ValueError("HF row video has neither path nor bytes")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    return str(out)


def _hf_rows(split: str, limit: int | None, video_dir: Path, invert_steering: bool) -> list[dict]:
    from datasets import Video, load_dataset

    ds = load_dataset("commaai/comma2k19", split=split)
    if "video" in ds.features:
        ds = ds.cast_column("video", Video(decode=False))
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    rows = []
    for row in ds:
        log = row["log"]
        video_path = _hf_video_path(row, video_dir)
        video = Path(video_path)
        timing = _video_timing(video)
        frame_t = np.asarray(log["global_pose__frame_times"], dtype=float)
        if len(frame_t) != timing.decoded_frame_count:
            raise ValueError(f"HF frame_times length {len(frame_t)} != decoded frames {timing.decoded_frame_count}: {video_path}")
        part, _ = _rows(
            video_path,
            frame_t,
            timing.pts_sec,
            log["processed_log__CAN__speed__t"],
            log["processed_log__CAN__speed__value"],
            log["processed_log__CAN__steering_angle__t"],
            log["processed_log__CAN__steering_angle__value"],
            invert_steering,
            "global_pose__frame_times",
        )
        rows.extend(part)
    return rows


def _alignment_summary(df: pd.DataFrame) -> None:
    if df.empty or "alignment_error_sec" not in df.columns:
        return
    err = df.alignment_error_sec.to_numpy(float)
    delta = df.video_frame_index.to_numpy(int) - (2 * df.sample_index.to_numpy(int))
    print(f"valid aligned samples: {len(df)}")
    print(f"max alignment error: {err.max():.6f}")
    print(f"mean alignment error: {err.mean():.6f}")
    print(f"p95 alignment error: {np.percentile(err, 95):.6f}")
    print(
        "actual_frame_index - 2*k: "
        f"min={delta.min()} p5={np.percentile(delta, 5):.1f} median={np.median(delta):.1f} "
        f"p95={np.percentile(delta, 95):.1f} max={delta.max()}"
    )


def _write_splits(df: pd.DataFrame, out_dir: Path, val_ratio: float) -> tuple[Path, Path]:
    if df.empty:
        raise RuntimeError("no valid comma2k19 samples were produced")

    split_key = df["route_id"].astype(str) if "route_id" in df.columns else df.video_path.astype(str)
    routes = sorted(split_key.unique())
    val_count = max(1, int(round(len(routes) * val_ratio))) if val_ratio > 0 and len(routes) > 1 else 0
    val_routes = set(routes[-val_count:]) if val_count else set()
    df["split"] = np.where(split_key.isin(val_routes), "val", "train")

    out_dir.mkdir(parents=True, exist_ok=True)
    schema = [
        "route_id", "segment_id", "video_path", "sample_index", "target_timestamp",
        "video_frame_index", "video_frame_timestamp", "alignment_error_sec", "alignment_version",
        "speed", "steering_angle", "acceleration", "accel_label", "steer_label", "source", "split",
    ]
    for col in schema:
        if col not in df.columns:
            df[col] = "" if col in {"route_id", "segment_id", "source", "split", "alignment_version", "video_path"} else np.nan
    df = df[schema]
    train_path, val_path = out_dir / "train.csv", out_dir / "val.csv"
    df[df.split == "train"].to_csv(train_path, index=False)
    df[df.split == "val"].to_csv(val_path, index=False)
    for split, part in df.groupby("split"):
        print(f"{split}: {len(part)} samples")
        print(f"{split} routes: {part.route_id.nunique() if 'route_id' in part.columns else 'N/A'}")
        print("accel:", {ACCEL_NAMES[k]: int(v) for k, v in part.accel_label.value_counts().sort_index().items()})
        print("steer:", {STEER_NAMES[k]: int(v) for k, v in part.steer_label.value_counts().sort_index().items()})
        _alignment_summary(part)
    return train_path, val_path


def build_manifest(raw_root: Path, out_dir: Path, val_ratio: float, limit: int | None, invert_steering: bool, hf_split: str | None) -> tuple[Path, Path]:
    if hf_split:
        rows = _hf_rows(hf_split, limit, out_dir.parent / "videos", invert_steering)
    else:
        segments = _segment_dirs(raw_root)
        if limit:
            segments = segments[:limit]
        if not segments:
            raise FileNotFoundError(f"no comma2k19 Chunk_* segment directories found under {raw_root}")
        rows = []
        total_candidates = total_valid = total_dropped = 0
        all_errors = []
        for segment in segments:
            try:
                part, stats = _local_rows(segment, raw_root, invert_steering)
                rows.extend(part)
                total_candidates += stats.candidates
                total_valid += stats.valid
                total_dropped += stats.dropped_no_video_frame
                all_errors.extend(stats.errors)
            except Exception as exc:
                warnings.warn(f"skip incomplete/unsynchronized segment {segment}: {exc}")
        print(f"candidate 10Hz samples: {total_candidates}")
        print(f"valid aligned samples: {total_valid}")
        print(f"dropped because no video frame: {total_dropped}")
        if all_errors:
            err = np.asarray(all_errors, dtype=float)
            print(f"max / mean / p95 alignment error: {err.max():.6f} / {err.mean():.6f} / {np.percentile(err, 95):.6f}")
            tmp = pd.DataFrame(rows)
            delta = tmp.video_frame_index.to_numpy(int) - (2 * tmp.sample_index.to_numpy(int))
            print(
                "actual_frame_index - 2*k distribution: "
                f"min={delta.min()} p5={np.percentile(delta, 5):.1f} median={np.median(delta):.1f} "
                f"p95={np.percentile(delta, 95):.1f} max={delta.max()}"
            )
    return _write_splits(pd.DataFrame(rows), out_dir, val_ratio)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=COMMA2K19_STAGE3_RAW)
    parser.add_argument("--out-dir", type=Path, default=COMMA2K19_STAGE3_MANIFEST)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--hf-split", default=None, help="Use HuggingFace load_dataset('commaai/comma2k19', split=...) instead of local Chunk_* files.")
    parser.add_argument("--invert-steering", action="store_true")
    args = parser.parse_args()
    build_manifest(args.raw_root, args.out_dir, args.val_ratio, args.limit, args.invert_steering, args.hf_split)


if __name__ == "__main__":
    main()