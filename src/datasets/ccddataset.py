from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


VIDEOMAE_MEAN = torch.tensor(
    [0.485, 0.456, 0.406],
    dtype=torch.float32,
).view(3, 1, 1)

VIDEOMAE_STD = torch.tensor(
    [0.229, 0.224, 0.225],
    dtype=torch.float32,
).view(3, 1, 1)


def decode_video_rgb(
    path: Path,
) -> np.ndarray:
    cap = cv2.VideoCapture(
        str(path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open video: {path}"
        )

    frames = []

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        frames.append(frame)

    cap.release()

    if not frames:
        raise RuntimeError(
            f"No frames decoded: {path}"
        )

    return np.stack(
        frames,
        axis=0,
    )


def sample_uniform_indices(
    frame_count: int,
    num_frames: int,
) -> np.ndarray:
    indices = np.linspace(
        0,
        frame_count - 1,
        num_frames,
    )

    indices = np.round(
        indices
    ).astype(np.int64)

    indices = np.clip(
        indices,
        0,
        frame_count - 1,
    )

    return indices


def transform_videomae_frame(
    frame: np.ndarray,
    image_size: int = 224,
) -> torch.Tensor:
    x = torch.from_numpy(
        frame.copy()
    )

    x = x.permute(
        2, 0, 1
    )

    x = (
        x.float()
        / 255.0
    )

    _, h, w = x.shape

    scale = (
        image_size
        / min(h, w)
    )

    new_h = max(
        image_size,
        round(h * scale),
    )

    new_w = max(
        image_size,
        round(w * scale),
    )

    x = TF.resize(
        x,
        [new_h, new_w],
        antialias=True,
    )

    x = TF.center_crop(
        x,
        [
            image_size,
            image_size,
        ],
    )

    x = (
        x - VIDEOMAE_MEAN
    ) / VIDEOMAE_STD

    return x


def preprocess_videomae_video(
    path: Path,
    num_frames: int = 16,
    image_size: int = 224,
) -> tuple[torch.Tensor, np.ndarray]:
    frames = decode_video_rgb(
        path
    )

    sampled_indices = sample_uniform_indices(
        len(frames),
        num_frames,
    )

    sampled_frames = frames[
        sampled_indices
    ]

    clip = torch.stack(
        [
            transform_videomae_frame(
                frame,
                image_size=image_size,
            )
            for frame
            in sampled_frames
        ],
        dim=0,
    )

    return (
        clip,
        sampled_indices,
    )


class CCDStage2VideoMAEDataset(Dataset):
    """
    CCD Stage 2 dataset for VideoMAE.
    Hugging Face VideoMAE 입력은: [B, T, C, H, W]
    이므로 Dataset 하나의 출력은: [T, C, H, W]

    Returns:
        video:
            FloatTensor [T, C, H, W]

        collision_index:
            collision frame mapped to sampled video timestep

        collision_frame:
            original CCD frame index

        sampled_indices:
            original frame indices selected for VideoMAE
    """

    def __init__(
        self,
        manifest_path: str | Path,
        num_frames: int = 16,
        image_size: int = 224,
    ):
        self.manifest_path = Path(
            manifest_path
        )

        self.df = pd.read_csv(
            self.manifest_path,
            dtype={
                "video_id": str,
                "source_id": str,
            },
        )

        self.df["video_id"] = (
            self.df["video_id"]
            .astype(str)
            .str.zfill(6)
        )

        self.num_frames = num_frames
        self.image_size = image_size

        self.mean = VIDEOMAE_MEAN
        self.std = VIDEOMAE_STD

    def __len__(self) -> int:
        return len(self.df)

    # ========================================================
    # Decode
    # ========================================================

    def _decode_video(
        self,
        path: Path,
    ) -> np.ndarray:
        return decode_video_rgb(
            path
        )

    # ========================================================
    # Temporal sampling
    # ========================================================

    def _sample_indices(
        self,
        frame_count: int,
    ) -> np.ndarray:
        return sample_uniform_indices(
            frame_count,
            self.num_frames,
        )

    def _map_original_to_sample(
        self,
        original_frame: int,
        sampled_indices: np.ndarray,
    ) -> int:
        """
        Original CCD frame index -> sampled VideoMAE timestep.
        """

        distance = np.abs(
            sampled_indices
            - original_frame
        )

        return int(
            np.argmin(distance)
        )

    # ========================================================
    # Spatial transform
    # ========================================================

    def _transform_frame(
        self,
        frame: np.ndarray,
    ) -> torch.Tensor:
        return transform_videomae_frame(
            frame,
            image_size=self.image_size,
        )

    # ========================================================
    # Sample
    # ========================================================

    def __getitem__(
        self,
        index: int,
    ) -> dict:
        row = self.df.iloc[
            index
        ]

        video_path = Path(
            row["video_path"]
        )

        (
            clip,
            sampled_indices,
        ) = preprocess_videomae_video(
            video_path,
            num_frames=self.num_frames,
            image_size=self.image_size,
        )

        # [T, C, H, W]

        collision_frame = int(
            row["collision_frame"]
        )

        collision_index = (
            self._map_original_to_sample(
                collision_frame,
                sampled_indices,
            )
        )

        entry_side = int(
            row["entry_side"]
        )

        side_valid = bool(
            row["side_valid"]
        )

        side_weight = float(
            row["side_weight"]
        )

        return {
            "video": clip,

            "video_id": (
                row["video_id"]
            ),

            "sampled_indices": (
                torch.from_numpy(
                    sampled_indices
                ).long()
            ),

            "collision_frame": (
                torch.tensor(
                    collision_frame,
                    dtype=torch.long,
                )
            ),

            "collision_index": (
                torch.tensor(
                    collision_index,
                    dtype=torch.long,
                )
            ),

            "collision_valid": (
                torch.tensor(
                    bool(
                        row[
                            "collision_valid"
                        ]
                    ),
                    dtype=torch.bool,
                )
            ),

            "entry_side": (
                torch.tensor(
                    entry_side,
                    dtype=torch.long,
                )
            ),

            "side_valid": (
                torch.tensor(
                    side_valid,
                    dtype=torch.bool,
                )
            ),

            "side_weight": (
                torch.tensor(
                    side_weight,
                    dtype=torch.float32,
                )
            ),
        }