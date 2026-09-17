from __future__ import annotations


def build_centered_clip_indices(center_index: int, num_frames: int, clip_len: int = 16) -> list[int]:
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    start = int(center_index) - int(clip_len) // 2
    return [min(max(start + i, 0), int(num_frames) - 1) for i in range(int(clip_len))]
