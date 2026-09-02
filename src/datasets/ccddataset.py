from __future__ import annotations

from src.datasets.stage2_dataset import (
    IMAGE_EXTENSIONS,
    MISSING_LABEL,
    VIDEO_EXTENSIONS,
    VIDEOMAE_MEAN,
    VIDEOMAE_STD,
    Stage2Dataset,
    decode_video_rgb,
    frame_number,
    image_paths,
    original_frame_to_sample_index,
    preprocess_frames,
    preprocess_image_sequence,
    preprocess_video,
    sample_frame_indices,
    transform_videomae_frame,
)

# Compatibility aliases for older utility imports. New Stage2 training uses
# Stage2Dataset from src.datasets.stage2_dataset directly.
sample_uniform_indices = sample_frame_indices
preprocess_videomae_video = preprocess_video

__all__ = [
    "IMAGE_EXTENSIONS",
    "MISSING_LABEL",
    "VIDEO_EXTENSIONS",
    "VIDEOMAE_MEAN",
    "VIDEOMAE_STD",
    "Stage2Dataset",
    "decode_video_rgb",
    "frame_number",
    "image_paths",
    "original_frame_to_sample_index",
    "preprocess_frames",
    "preprocess_image_sequence",
    "preprocess_video",
    "preprocess_videomae_video",
    "sample_frame_indices",
    "sample_uniform_indices",
    "transform_videomae_frame",
]
