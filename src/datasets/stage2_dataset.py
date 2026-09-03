from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset


MISSING_LABEL = -1
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
VIDEOMAE_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
VIDEOMAE_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def frame_number(path: Path) -> int:
    match = re.search(r"(\d+)$", path.stem)
    return int(match.group(1)) if match else 0


def image_paths(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    return sorted(
        (p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS),
        key=frame_number,
    )


def decode_video_rgb(video_path: str | Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()

    if not frames:
        raise RuntimeError(f"No frames decoded from video: {video_path}")
    return np.stack(frames, axis=0)


def sample_frame_indices(frame_count: int, num_frames: int) -> np.ndarray:
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if frame_count == 1:
        return np.zeros(num_frames, dtype=np.int64)
    return np.rint(np.linspace(0, frame_count - 1, num_frames)).astype(np.int64)


def original_frame_to_sample_index(original_frame: int, sampled_indices: Iterable[int]) -> int:
    if original_frame < 0:
        return MISSING_LABEL
    indices = np.asarray(list(sampled_indices), dtype=np.int64)
    if indices.size == 0:
        return MISSING_LABEL
    return int(np.abs(indices - int(original_frame)).argmin())


def transform_videomae_frame(frame: np.ndarray | Image.Image, image_size: int = 224) -> torch.Tensor:
    image = frame if isinstance(frame, Image.Image) else Image.fromarray(frame)
    image = image.convert("RGB")
    width, height = image.size
    scale = image_size / min(width, height)
    resized = (round(height * scale), round(width * scale))
    tensor = TF.to_tensor(TF.resize(image, resized, antialias=True))
    tensor = TF.center_crop(tensor, [image_size, image_size])
    return (tensor - VIDEOMAE_MEAN) / VIDEOMAE_STD


def preprocess_frames(
    frames: np.ndarray,
    num_frames: int = 16,
    image_size: int = 224,
) -> tuple[torch.Tensor, np.ndarray]:
    sampled_indices = sample_frame_indices(len(frames), num_frames)
    clip = torch.stack(
        [transform_videomae_frame(frames[int(index)], image_size) for index in sampled_indices],
        dim=0,
    )
    return clip, sampled_indices


def preprocess_video(
    video_path: str | Path,
    num_frames: int = 16,
    image_size: int = 224,
) -> tuple[torch.Tensor, np.ndarray]:
    return preprocess_frames(decode_video_rgb(video_path), num_frames=num_frames, image_size=image_size)


def preprocess_image_sequence(
    paths: list[Path],
    num_frames: int = 16,
    image_size: int = 224,
) -> tuple[torch.Tensor, np.ndarray]:
    if not paths:
        raise RuntimeError("Stage2 image sequence is empty")
    sampled_positions = sample_frame_indices(len(paths), num_frames)
    sampled_paths = [paths[int(index)] for index in sampled_positions]
    clip = torch.stack(
        [
            transform_videomae_frame(np.asarray(Image.open(path).convert("RGB")), image_size)
            for path in sampled_paths
        ],
        dim=0,
    )
    original_frame_numbers = np.asarray([frame_number(path) for path in sampled_paths], dtype=np.int64)
    return clip, original_frame_numbers


def _label(row: pd.Series, name: str) -> int:
    value = row.get(name, MISSING_LABEL)
    if pd.isna(value):
        return MISSING_LABEL
    return int(value)


class Stage2Dataset(Dataset):
    def __init__(self, manifest_path: str | Path, num_frames: int = 16, image_size: int = 224, min_pseudo_label_confidence: float | None = None):
        self.manifest_path = Path(manifest_path)
        df = pd.read_csv(self.manifest_path)
        if min_pseudo_label_confidence is not None and "overall_confidence" in df.columns:
            df = df[df["overall_confidence"].fillna(0.0).astype(float) >= float(min_pseudo_label_confidence)].reset_index(drop=True)
        self.rows = df.to_dict("records")
        self.num_frames = num_frames
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = pd.Series(self.rows[index])
        video_path = Path(row["video_path"])
        video, sampled_indices = preprocess_video(video_path, self.num_frames, self.image_size)

        collision_frame = _label(row, "collision_frame")
        entry_frame = _label(row, "entry_frame")
        return {
            "video": video,
            "video_path": str(video_path),
            "sampled_indices": torch.as_tensor(sampled_indices, dtype=torch.long),
            "collision_frame": torch.tensor(collision_frame, dtype=torch.long),
            "entry_frame": torch.tensor(entry_frame, dtype=torch.long),
            "collision_index": torch.tensor(
                original_frame_to_sample_index(collision_frame, sampled_indices), dtype=torch.long
            ),
            "entry_index": torch.tensor(original_frame_to_sample_index(entry_frame, sampled_indices), dtype=torch.long),
            "direction": torch.tensor(_label(row, "direction"), dtype=torch.long),
            "avoidance": torch.tensor(_label(row, "avoidance"), dtype=torch.long),
        }
