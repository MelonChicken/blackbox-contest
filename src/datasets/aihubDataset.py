from pathlib import Path

import cv2
import pandas as pd
import torch

from torch.utils.data import Dataset

from src.preprocessing.recapture import RecaptureTransform


class AIHubStage1Dataset(Dataset):
    """
    AI-Hub 교통사고 블랙박스 영상을 이용한
    Stage 1 ORIGINAL / RERECORDED Dataset.

    하나의 source video를 한 번만 decode한 뒤

        ORIGINAL   label = 0
        RERECORDED label = 1

    두 sample을 동시에 생성한다.

    __getitem__ 반환:

        clips:
            [2, C, T, H, W]

        labels:
            [2]

    기본값:

        C = 3
        T = 16
        H = 224
        W = 224
    """

    def __init__(
        self,
        root,
        manifest,
        frames=16,
        size=224,
        mean=(0.45, 0.45, 0.45),
        std=(0.225, 0.225, 0.225),
        train=True,
    ):
        # ------------------------------
        # 1. 기본 설정
        # ------------------------------

        self.root = Path(root)
        self.manifest = Path(manifest)

        self.frames = frames
        self.size = size
        self.train = train

        # ------------------------------
        # 2. Manifest 로드
        # ------------------------------

        self.df = pd.read_csv(
            self.manifest
        )

        if len(self.df) == 0:
            raise RuntimeError(
                f"Empty manifest: {self.manifest}"
            )

        if "video_path" not in self.df.columns:
            raise RuntimeError(
                "Manifest must contain "
                "'video_path' column."
            )

        # ------------------------------
        # 3. Normalization
        # ------------------------------

        self.mean = torch.as_tensor(
            mean,
            dtype=torch.float32,
        ).reshape(
            1,
            3,
            1,
            1,
        )

        self.std = torch.as_tensor(
            std,
            dtype=torch.float32,
        ).reshape(
            1,
            3,
            1,
            1,
        )

        # ------------------------------
        # 4. Synthetic recapture
        # ------------------------------

        self.recapture = RecaptureTransform()

    def __len__(self):
        """
        이제 Dataset의 index 하나는
        sample 하나가 아니라 source video 하나를 의미한다.

        따라서:

            len(dataset)
            =
            source video 수
        """

        return len(self.df)

    def _load_clip(
        self,
        video_path: Path,
    ):
        """
        전체 영상을 메모리에 올리지 않고
        균등한 위치의 self.frames개 frame만 읽는다.

        반환:
            Tensor [T, C, H, W]

        dtype:
            float32

        range:
            [0, 1]
        """

        cap = cv2.VideoCapture(
            str(video_path)
        )

        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open video: "
                f"{video_path}"
            )

        try:
            # --------------------------
            # 전체 frame 수
            # --------------------------

            total_frames = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_COUNT
                )
            )

            if total_frames <= 0:
                raise RuntimeError(
                    f"Invalid frame count: "
                    f"{video_path}"
                )

            # --------------------------
            # Sampling 위치 생성
            # --------------------------

            indices = torch.linspace(
                0,
                total_frames - 1,
                steps=self.frames,
            ).round().long().tolist()

            frames = []

            # --------------------------
            # 선택된 frame decode
            # --------------------------

            for frame_index in indices:

                cap.set(
                    cv2.CAP_PROP_POS_FRAMES,
                    frame_index,
                )

                success, frame = cap.read()

                if not success:
                    raise RuntimeError(
                        f"Could not decode frame "
                        f"{frame_index} "
                        f"from {video_path}"
                    )

                # ----------------------
                # BGR -> RGB
                # ----------------------

                frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB,
                )

                # ----------------------
                # Resize
                # ----------------------

                frame = cv2.resize(
                    frame,
                    (
                        self.size,
                        self.size,
                    ),
                    interpolation=(
                        cv2.INTER_LINEAR
                    ),
                )

                # ----------------------
                # numpy -> Tensor
                #
                # [H,W,C]
                # ->
                # [C,H,W]
                # ----------------------

                frame = torch.from_numpy(
                    frame
                ).permute(
                    2,
                    0,
                    1,
                ).contiguous()

                # uint8 [0,255]
                # ->
                # float32 [0,1]

                frame = (
                    frame.float()
                    / 255.0
                )

                frames.append(
                    frame
                )

            # --------------------------
            # [T,C,H,W]
            # --------------------------

            clip = torch.stack(
                frames,
                dim=0,
            )

            return clip

        finally:
            cap.release()

    def _normalize(
        self,
        clip: torch.Tensor,
    ):
        """
        clip:
            [T,C,H,W]
        """

        return (
            clip - self.mean
        ) / self.std

    def __getitem__(
        self,
        index,
    ):
        """
        Source video 하나를 한 번만 decode하고

        ORIGINAL
        RERECORDED

        두 sample을 동시에 반환한다.

        Returns
        -------
        clips:
            [2,C,T,H,W]

        labels:
            [2]
            [0,1]
        """

        # ------------------------------
        # 1. Source video
        # ------------------------------

        source_index = index

        row = self.df.iloc[
            source_index
        ]

        video_path = (
            self.root
            / row["video_path"]
        )

        if not video_path.exists():
            raise FileNotFoundError(
                f"Video not found: "
                f"{video_path}"
            )

        # ------------------------------
        # 2. Video decode
        #
        # 여기서 단 한 번만 수행
        # ------------------------------

        # [T,C,H,W]
        # range [0,1]

        clip = self._load_clip(
            video_path
        )

        # ------------------------------
        # 3. ORIGINAL
        # ------------------------------

        original = clip.clone()

        # ------------------------------
        # 4. RERECORDED
        # ------------------------------

        rerecorded = clip.clone()

        if self.train:

            # Training에서는 매 epoch마다
            # 새로운 synthetic artifact
            seed = None

        else:

            # Validation에서는 항상
            # 동일한 synthetic artifact
            seed = source_index

        rerecorded = self.recapture(
            rerecorded,
            seed=seed,
        )

        # ------------------------------
        # 5. Normalize
        # ------------------------------

        original = self._normalize(
            original
        )

        rerecorded = self._normalize(
            rerecorded
        )

        # ------------------------------
        # 6. MViT input shape
        #
        # [T,C,H,W]
        # ->
        # [C,T,H,W]
        # ------------------------------

        original = original.permute(
            1,
            0,
            2,
            3,
        ).contiguous()

        rerecorded = rerecorded.permute(
            1,
            0,
            2,
            3,
        ).contiguous()

        # ------------------------------
        # 7. Pair 생성
        # ------------------------------

        # [2,C,T,H,W]

        clips = torch.stack(
            [
                original,
                rerecorded,
            ],
            dim=0,
        )

        # ORIGINAL   = 0
        # RERECORDED = 1

        labels = torch.tensor(
            [0, 1],
            dtype=torch.long,
        )

        return (
            clips,
            labels,
        )