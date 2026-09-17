from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def build_centered_clip_indices(center_index: int, num_frames: int, clip_len: int = 16) -> list[int]:
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    start = int(center_index) - int(clip_len) // 2
    return [min(max(start + i, 0), int(num_frames) - 1) for i in range(int(clip_len))]


def build_centered_clip_timestamps(center_timestamp: float, sampling_hz: float, past_frames: int, future_frames: int) -> list[float]:
    if sampling_hz <= 0:
        raise ValueError("sampling_hz must be positive")
    offsets = range(-int(past_frames), int(future_frames) + 1)
    return [float(center_timestamp) + (i / float(sampling_hz)) for i in offsets]


def nearest_frame_indices_for_timestamps(
    frame_times,
    target_timestamps,
    max_alignment_error_sec: float | None = None,
    boundary_policy: str = "clamp",
) -> tuple[list[int], list[float]]:
    frame_times = np.asarray(frame_times, dtype=float).squeeze()
    targets = np.atleast_1d(np.asarray(target_timestamps, dtype=float).squeeze())
    if frame_times.ndim != 1 or len(frame_times) == 0:
        raise ValueError("frame_times must be a non-empty 1D sequence")
    if np.any(np.diff(frame_times) < 0):
        raise ValueError("frame_times must be sorted ascending")
    if boundary_policy not in {"clamp", "edge_clamp"}:
        raise ValueError(f"unsupported boundary_policy: {boundary_policy}")

    indices: list[int] = []
    errors: list[float] = []
    for target in targets:
        if target <= frame_times[0]:
            idx = 0
            in_range = target >= frame_times[0]
        elif target >= frame_times[-1]:
            idx = len(frame_times) - 1
            in_range = target <= frame_times[-1]
        else:
            right = int(np.searchsorted(frame_times, target, side="left"))
            left = right - 1
            idx = left if abs(frame_times[left] - target) <= abs(frame_times[right] - target) else right
            in_range = True
        err = abs(float(frame_times[idx] - target))
        if max_alignment_error_sec is not None and in_range and err > float(max_alignment_error_sec):
            raise ValueError(f"clip timestamp alignment error {err:.6f}s exceeds {float(max_alignment_error_sec):.6f}s")
        indices.append(int(idx))
        errors.append(err)
    return indices, errors


def build_timestamp_nearest_clip(
    frame_times,
    center_timestamp: float,
    sampling_hz: float,
    past_frames: int,
    future_frames: int,
    max_alignment_error_sec: float | None = None,
    boundary_policy: str = "clamp",
) -> dict[str, Any]:
    target_timestamps = build_centered_clip_timestamps(center_timestamp, sampling_hz, past_frames, future_frames)
    indices, errors = nearest_frame_indices_for_timestamps(
        frame_times,
        target_timestamps,
        max_alignment_error_sec=max_alignment_error_sec,
        boundary_policy=boundary_policy,
    )
    selected_times = np.asarray(frame_times, dtype=float)[indices]
    deltas = np.diff(selected_times)
    return {
        "clip_frame_indices": indices,
        "clip_target_timestamps": target_timestamps,
        "clip_alignment_errors_sec": errors,
        "clip_max_alignment_error_sec": max(errors) if errors else 0.0,
        "clip_selected_interval_min_sec": float(deltas.min()) if len(deltas) else 0.0,
        "clip_selected_interval_median_sec": float(np.median(deltas)) if len(deltas) else 0.0,
        "clip_selected_interval_max_sec": float(deltas.max()) if len(deltas) else 0.0,
    }


def json_dumps_compact(values) -> str:
    return json.dumps(list(values), separators=(",", ":"))


def parse_json_list(value: str, expected_len: int, item_type: type, name: str) -> list:
    try:
        data = json.loads(value)
    except Exception as exc:
        raise ValueError(f"invalid JSON in {name}: {value!r}") from exc
    if not isinstance(data, list) or len(data) != int(expected_len):
        raise ValueError(f"{name} must be a JSON list of length {expected_len}")
    out = []
    for item in data:
        if item_type is int:
            if not isinstance(item, int):
                raise ValueError(f"{name} contains non-int value: {item!r}")
            out.append(int(item))
        else:
            if not isinstance(item, (int, float)):
                raise ValueError(f"{name} contains non-number value: {item!r}")
            out.append(float(item))
    return out


def parse_clip_frame_indices(value: str, expected_len: int) -> list[int]:
    return parse_json_list(value, expected_len, int, "clip_frame_indices")


def parse_clip_float_list(value: str, expected_len: int, name: str) -> list[float]:
    return parse_json_list(value, expected_len, float, name)


def sampling_metadata(
    *,
    sampling_hz: float,
    sampling_policy: str,
    num_frames: int,
    past_frames: int,
    future_frames: int,
    boundary_policy: str,
    output_hz: float,
) -> dict[str, Any]:
    return {
        "sampling_hz": float(sampling_hz),
        "sampling_policy": str(sampling_policy),
        "num_frames": int(num_frames),
        "past_frames": int(past_frames),
        "future_frames": int(future_frames),
        "boundary_policy": str(boundary_policy),
        "output_hz": float(output_hz),
    }


def validate_sampling_metadata(metadata: dict[str, Any], expected: dict[str, Any]) -> None:
    for key, want in expected.items():
        if key not in metadata:
            raise RuntimeError(f"manifest metadata missing {key}")
        got = metadata[key]
        if isinstance(want, float):
            if abs(float(got) - want) > 1e-9:
                raise RuntimeError(f"manifest metadata {key} mismatch: expected {want}, got {got}")
        else:
            if got != want:
                raise RuntimeError(f"manifest metadata {key} mismatch: expected {want}, got {got}")


def read_manifest_metadata(manifest: str | Path) -> dict[str, Any]:
    path = Path(manifest).parent / "metadata.json"
    if not path.is_file():
        raise RuntimeError(f"missing Stage3 manifest metadata: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


