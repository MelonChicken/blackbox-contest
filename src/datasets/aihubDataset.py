from __future__ import annotations

import math
import random
from pathlib import Path

import cv2
import pandas as pd
import torch
import torchvision.transforms.functional as TF

from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode

from src.preprocessing.recapture import RecaptureTransform


class CommonVideoTransform:
    """
    ORIGINAL / RERECORDED 양쪽에 공통으로 나타날 수 있는
    일반적인 영상 품질 변화를 합성한다.

    이 transform은 재촬영 여부를 나타내는 feature가 아니다.

    따라서 학습 시 두 class에 동일한 확률 분포로 독립 적용하여

        blur -> RERECORDED
        resize degradation -> RERECORDED
        gamma shift -> RERECORDED

    와 같은 class shortcut을 학습하기 어렵게 한다.

    Input
    -----
    clip:
        float Tensor [T, C, H, W]
        range [0, 1]

    Output
    ------
    float Tensor [T, C, H, W]
        range [0, 1]
    """

    def __call__(
        self,
        clip: torch.Tensor,
        seed: int | None = None,
    ) -> torch.Tensor:
        rng = random.Random(seed)

        clip = clip.clone()

        # 일반적인 static gamma 변화
        if rng.random() < 0.35:
            clip = self._gamma(
                clip,
                rng,
            )

        # 일반적인 mild blur
        if rng.random() < 0.25:
            clip = self._blur(
                clip,
                rng,
            )

        # 일반적인 resize / resampling degradation
        if rng.random() < 0.25:
            clip = self._resample(
                clip,
                rng,
            )

        return clip.clamp(
            0.0,
            1.0,
        )

    @staticmethod
    def _gamma(
        clip: torch.Tensor,
        rng: random.Random,
    ) -> torch.Tensor:
        """
        일반적인 밝기 / camera response 차이를
        약한 gamma 변화로 근사한다.
        """

        gamma = rng.uniform(
            0.92,
            1.08,
        )

        return (
            clip
            .clamp(1e-6, 1.0)
            .pow(gamma)
        )

    @staticmethod
    def _blur(
        clip: torch.Tensor,
        rng: random.Random,
    ) -> torch.Tensor:
        """
        재촬영 여부와 무관하게 발생할 수 있는
        약한 blur를 적용한다.
        """

        sigma = rng.uniform(
            0.20,
            0.70,
        )

        return TF.gaussian_blur(
            clip,
            kernel_size=[
                3,
                3,
            ],
            sigma=[
                sigma,
                sigma,
            ],
        )

    @staticmethod
    def _resample(
        clip: torch.Tensor,
        rng: random.Random,
    ) -> torch.Tensor:
        """
        일반적인 resize / encoding 과정에서도 나타날 수 있는
        약한 resampling degradation을 적용한다.
        """

        _, _, h, w = clip.shape

        scale = rng.uniform(
            0.82,
            0.98,
        )

        small_h = max(
            32,
            int(h * scale),
        )

        small_w = max(
            32,
            int(w * scale),
        )

        clip = TF.resize(
            clip,
            [
                small_h,
                small_w,
            ],
            interpolation=(
                InterpolationMode.BILINEAR
            ),
            antialias=True,
        )

        clip = TF.resize(
            clip,
            [
                h,
                w,
            ],
            interpolation=(
                InterpolationMode.BILINEAR
            ),
            antialias=True,
        )

        return clip


class AIHubStage1Dataset(Dataset):
    """
    AI-Hub 교통사고 블랙박스 영상을 이용한
    Stage 1 ORIGINAL / RERECORDED Dataset.

    하나의 source video에서

        ORIGINAL   label = 0
        RERECORDED label = 1

    두 sample을 생성한다.

    핵심 원칙
    ---------
    ORIGINAL과 RERECORDED는 반드시 동일한 spatial preprocessing
    경로에서 출발한다.

        decoded source
              |
              v
        intermediate resize
             320
              |
         shared base clip
           /       \\
          /         \\
    ORIGINAL     RERECORDED
                    |
             RecaptureTransform
          \\         /
           \\       /
              |
        동일한 320 -> 224 resize
              |
       common augmentation
              |
          normalization

    이렇게 함으로써

        source -> 224

    와

        source -> 320 -> 224

    의 resize-path 차이가 label shortcut이 되는 것을 방지한다.

    학습 중에는 blur, ordinary resampling, static gamma처럼
    재촬영에 고유하지 않은 특징을 두 class에 동일한 분포로
    독립적으로 적용한다.

    Returns
    -------
    clips:
        Tensor [2, C, T, H, W]

    labels:
        Tensor [2]

        ORIGINAL   = 0
        RERECORDED = 1
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
        recapture_size=320,
    ):
        # ------------------------------
        # 1. 기본 설정
        # ------------------------------

        self.root = Path(root)
        self.manifest = Path(manifest)

        self.frames = frames
        self.size = size
        self.train = train

        # Synthetic recapture를 적용할
        # intermediate spatial resolution.
        #
        # ORIGINAL과 RERECORDED 모두
        # 동일한 intermediate resolution을 거친다.
        self.recapture_size = max(
            recapture_size,
            size,
        )

        # ------------------------------
        # 2. Manifest 로드
        # ------------------------------

        self.df = pd.read_csv(
            self.manifest
        )

        if len(self.df) == 0:
            raise RuntimeError(
                f"Empty manifest: "
                f"{self.manifest}"
            )

        if "video_path" not in self.df.columns:
            raise RuntimeError(
                "Manifest must contain "
                "'video_path' column."
            )

        # ------------------------------
        # 3. Normalization
        # ------------------------------
        #
        # clip shape이 normalize 시점에는
        # [T, C, H, W]이므로
        #
        # [1, 3, 1, 1]
        #
        # 형태로 broadcasting한다.
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
        # 4. Augmentation
        # ------------------------------
        #
        # Validation이 training과 완전히 동일한
        # synthetic parameter distribution만 평가하지 않도록
        # 별도의 holdout profile을 사용한다.
        #
        # 단, 이것은 실제 physical recapture validation을
        # 대체하지 않는다.
        # ------------------------------

        recapture_profile = (
            "train"
            if self.train
            else "holdout"
        )

        self.recapture = RecaptureTransform(
            profile=recapture_profile,
        )

        self.common = CommonVideoTransform()

    def __len__(self):
        """
        Dataset index 하나는 individual sample이 아니라
        source video 하나를 의미한다.

        source 하나에서

            ORIGINAL
            RERECORDED

        두 sample이 생성된다.
        """

        return len(self.df)

    @staticmethod
    def _frame_to_tensor(
        frame,
    ) -> torch.Tensor:
        """
        RGB numpy frame을 PyTorch Tensor로 변환한다.

        Input:
            uint8 [H, W, C]
            range [0, 255]

        Output:
            float32 [C, H, W]
            range [0, 1]
        """

        frame = torch.from_numpy(
            frame
        ).permute(
            2,
            0,
            1,
        ).contiguous()

        return (
            frame.float()
            / 255.0
        )

    def _load_clip(
        self,
        video_path: Path,
    ):
        """
        전체 영상을 메모리에 올리지 않고
        균등한 위치에서 self.frames개의 frame을 decode한다.

        ORIGINAL과 RERECORDED가 서로 다른 resize pipeline을
        갖지 않도록 하나의 shared intermediate clip만 생성한다.

        동일한 decoded frame은

            source
              |
              v
        recapture_size x recapture_size

        로 한 번 resize된 뒤 양 class가 공유한다.

        Returns
        -------
        base_clip:
            Tensor
            [T, C, recapture_size, recapture_size]

        timestamps:
            Tensor [T] 또는 None

            FPS metadata가 정상적이면 각 sampled frame의
            실제 timestamp(sec)를 반환한다.

            FPS metadata가 유효하지 않으면 None을 반환하며,
            RecaptureTransform 내부의 normalized-time fallback을
            사용한다.
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
            # FPS
            # --------------------------

            fps = float(
                cap.get(
                    cv2.CAP_PROP_FPS
                )
            )

            if (
                not math.isfinite(fps)
                or fps <= 1e-6
            ):
                fps = None

            # --------------------------
            # Sampling 위치
            # --------------------------
            #
            # Stage 1은 영상 전체의 재촬영 여부를
            # 판별하므로 영상 전체에서 균등하게
            # 16 frame을 선택한다.
            # --------------------------

            indices = (
                torch.linspace(
                    0,
                    total_frames - 1,
                    steps=self.frames,
                )
                .round()
                .long()
                .tolist()
            )

            frames = []

            # --------------------------
            # 선택 frame decode
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
                # Shared intermediate resize
                #
                # ORIGINAL / RERECORDED 모두
                # 동일한 representation에서 시작한다.
                # ----------------------

                frame = cv2.resize(
                    frame,
                    (
                        self.recapture_size,
                        self.recapture_size,
                    ),
                    interpolation=(
                        cv2.INTER_LINEAR
                    ),
                )

                frames.append(
                    self._frame_to_tensor(
                        frame
                    )
                )

            # --------------------------
            # [T, C, H, W]
            # --------------------------

            base_clip = torch.stack(
                frames,
                dim=0,
            )

            # --------------------------
            # 실제 sampled timestamps
            # --------------------------

            if fps is not None:

                timestamps = torch.tensor(
                    [
                        frame_index / fps
                        for frame_index in indices
                    ],
                    dtype=torch.float32,
                )

            else:

                timestamps = None

            return (
                base_clip,
                timestamps,
            )

        finally:
            cap.release()

    def _resize_to_model_size(
        self,
        clip: torch.Tensor,
    ) -> torch.Tensor:
        """
        Intermediate resolution의 clip을
        Stage 1 MViT input resolution으로 resize한다.

        이 함수는 ORIGINAL과 RERECORDED 양쪽에
        반드시 동일하게 사용한다.

        Input:
            [T, C, recapture_size, recapture_size]

        Output:
            [T, C, size, size]
        """

        return TF.resize(
            clip,
            [
                self.size,
                self.size,
            ],
            interpolation=(
                InterpolationMode.BILINEAR
            ),
            antialias=True,
        )

    def _normalize(
        self,
        clip: torch.Tensor,
    ) -> torch.Tensor:
        """
        Stage 1 MViT normalization.

        Input:
            [T, C, H, W]

        Output:
            [T, C, H, W]
        """

        return (
            clip
            - self.mean
        ) / self.std

    def __getitem__(
        self,
        index,
    ):
        """
        Source video 하나에서

            ORIGINAL
            RERECORDED

        두 sample을 생성한다.

        Returns
        -------
        clips:
            Tensor [2, C, T, H, W]

        labels:
            Tensor [2]
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
        # 2. Shared source clip
        # ------------------------------

        base_clip, timestamps = (
            self._load_clip(
                video_path
            )
        )

        # 두 class가 완전히 같은 intermediate
        # representation에서 출발한다.

        original = base_clip.clone()

        rerecorded = base_clip.clone()

        # ------------------------------
        # 3. Synthetic RERECORDED
        # ------------------------------

        if self.train:

            # Training에서는 sample이 호출될 때마다
            # 새로운 synthetic recapture condition을 생성한다.
            recapture_seed = (
                random.getrandbits(63)
            )

        else:

            # Validation에서는 같은 source video에 대해
            # 항상 동일한 synthetic recapture condition을
            # 재현할 수 있도록 deterministic seed를 사용한다.
            recapture_seed = (
                source_index
                + 10_000_000
            )

        rerecorded = self.recapture(
            rerecorded,
            seed=recapture_seed,
            timestamps=timestamps,
        )

        # ------------------------------
        # 4. Model input resolution
        # ------------------------------
        #
        # ORIGINAL:
        #     shared 320
        #         ->
        #     동일 resize
        #         ->
        #     224
        #
        # RERECORDED:
        #     shared 320
        #         ->
        #     RecaptureTransform
        #         ->
        #     동일 resize
        #         ->
        #     224
        #
        # 따라서 resize path 자체가
        # class label을 나타낼 수 없다.
        # ------------------------------

        original = self._resize_to_model_size(
            original
        )

        rerecorded = self._resize_to_model_size(
            rerecorded
        )

        # ------------------------------
        # 5. Common augmentation
        # ------------------------------
        #
        # blur / ordinary resampling / static gamma는
        # 실제 ORIGINAL에도 나타날 수 있다.
        #
        # 따라서 training에서는 두 class에
        # 동일한 확률 분포로 독립 적용한다.
        #
        # Validation에서는 적용하지 않아
        # deterministic evaluation을 유지한다.
        # ------------------------------

        if self.train:

            original_common_seed = (
                random.getrandbits(63)
            )

            rerecorded_common_seed = (
                random.getrandbits(63)
            )

            original = self.common(
                original,
                seed=original_common_seed,
            )

            rerecorded = self.common(
                rerecorded,
                seed=rerecorded_common_seed,
            )

        # ------------------------------
        # 6. Normalize
        # ------------------------------

        original = self._normalize(
            original
        )

        rerecorded = self._normalize(
            rerecorded
        )

        # ------------------------------
        # 7. MViT input shape
        #
        # [T, C, H, W]
        # ->
        # [C, T, H, W]
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
        # 8. Pair 생성
        # ------------------------------
        #
        # [2, C, T, H, W]
        # ------------------------------

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
            [
                0,
                1,
            ],
            dtype=torch.long,
        )

        return (
            clips,
            labels,
        )