from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


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

        # VideoMAE pretrained preprocessing에 맞춤.
        self.mean = torch.tensor(
            [0.485, 0.456, 0.406],
            dtype=torch.float32,
        ).view(3, 1, 1)

        self.std = torch.tensor(
            [0.229, 0.224, 0.225],
            dtype=torch.float32,
        ).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.df)

    # ========================================================
    # Decode
    # ========================================================

    def _decode_video(
        self,
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

    # ========================================================
    # Temporal sampling
    # ========================================================

    def _sample_indices(
        self,
        frame_count: int,
    ) -> np.ndarray:
        """
        Uniform temporal sampling.

        CCD normally contains 50 frames.
        """

        if frame_count < self.num_frames:
            # 극단적인 예외 처리.
            indices = np.linspace(
                0,
                frame_count - 1,
                self.num_frames,
            )
        else:
            indices = np.linspace(
                0,
                frame_count - 1,
                self.num_frames,
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

        # Short side -> 224
        scale = (
            self.image_size
            / min(h, w)
        )

        new_h = max(
            self.image_size,
            round(h * scale),
        )

        new_w = max(
            self.image_size,
            round(w * scale),
        )

        x = TF.resize(
            x,
            [new_h, new_w],
            antialias=True,
        )

        # Center crop 224x224
        x = TF.center_crop(
            x,
            [
                self.image_size,
                self.image_size,
            ],
        )

        x = (
            x - self.mean
        ) / self.std

        return x

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

        frames = self._decode_video(
            video_path
        )

        sampled_indices = (
            self._sample_indices(
                len(frames)
            )
        )

        sampled_frames = frames[
            sampled_indices
        ]

        clip = torch.stack(
            [
                self._transform_frame(
                    frame
                )
                for frame
                in sampled_frames
            ],
            dim=0,
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