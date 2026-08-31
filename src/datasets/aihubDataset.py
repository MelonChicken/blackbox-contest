from __future__ import annotations

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

    따라서 학습 시 두 class에 독립적으로 적용하여

        blur -> RERECORDED
        resize degradation -> RERECORDED
        gamma shift -> RERECORDED

    같은 shortcut을 학습하지 못하도록 한다.

    Input
    -----
    clip:
        float Tensor [T, C, H, W], range [0, 1]

    Output
    ------
    float Tensor [T, C, H, W], range [0, 1]
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

    source video 하나를 한 번 decode한 뒤

        ORIGINAL   label = 0
        RERECORDED label = 1

    두 sample을 생성한다.

    ORIGINAL은 실제 inference와 동일하게
    선택된 frame을 바로 model input size로 resize한다.

    RERECORDED는 더 큰 intermediate resolution에서
    synthetic screen-recapture process를 적용한 뒤
    model input size로 resize한다.

    학습 시에는 blur, resize degradation, static gamma처럼
    재촬영에 고유하지 않은 artifact를 두 class 모두에
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

        # 재촬영 simulation을 수행할
        # intermediate spatial resolution
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

        # train과 validation에서 재촬영 generator의
        # parameter distribution을 완전히 동일하게 두지 않는다.
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
        Dataset index 하나는 sample 하나가 아니라
        source video 하나를 의미한다.
        """

        return len(self.df)

    @staticmethod
    def _frame_to_tensor(
        frame,
    ) -> torch.Tensor:
        """
        RGB numpy frame [H, W, C], uint8
        ->
        Tensor [C, H, W], float32 [0, 1]
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
        균등한 위치의 frame만 decode한다.

        동일하게 decode된 source frame으로부터

            original_clip:
                [T, C, size, size]

            recapture_clip:
                [T, C, recapture_size, recapture_size]

        두 spatial representation을 만든다.

        또한 CAP_PROP_FPS가 정상적인 경우
        실제 sampled frame timestamp를 반환한다.

        Returns
        -------
        original_clip:
            Tensor [T, C, size, size]

        recapture_clip:
            Tensor [T, C, recapture_size, recapture_size]

        timestamps:
            Tensor [T] 또는 None
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
                not torch.isfinite(
                    torch.tensor(fps)
                )
                or fps <= 1e-6
            ):
                fps = None

            # --------------------------
            # Sampling 위치
            # --------------------------

            indices = torch.linspace(
                0,
                total_frames - 1,
                steps=self.frames,
            ).round().long().tolist()

            original_frames = []
            recapture_frames = []

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

                # BGR -> RGB
                frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB,
                )

                # ----------------------
                # ORIGINAL path
                #
                # 실제 inference와 동일하게
                # source frame -> 224 직접 resize
                # ----------------------

                original_frame = cv2.resize(
                    frame,
                    (
                        self.size,
                        self.size,
                    ),
                    interpolation=(
                        cv2.INTER_LINEAR
                    ),
                )

                original_frames.append(
                    self._frame_to_tensor(
                        original_frame
                    )
                )

                # ----------------------
                # RERECORDED simulation path
                #
                # 224보다 큰 spatial resolution에서
                # screen-camera artifact를 만든다.
                # ----------------------

                recapture_frame = cv2.resize(
                    frame,
                    (
                        self.recapture_size,
                        self.recapture_size,
                    ),
                    interpolation=(
                        cv2.INTER_LINEAR
                    ),
                )

                recapture_frames.append(
                    self._frame_to_tensor(
                        recapture_frame
                    )
                )

            original_clip = torch.stack(
                original_frames,
                dim=0,
            )

            recapture_clip = torch.stack(
                recapture_frames,
                dim=0,
            )

            # --------------------------
            # 실제 sampled timestamps
            # --------------------------

            if fps is not None:

                timestamps = torch.tensor(
                    [
                        frame_index / fps
                        for frame_index
                        in indices
                    ],
                    dtype=torch.float32,
                )

            else:

                # FPS metadata가 신뢰할 수 없는 경우
                # RecaptureTransform 내부의 normalized-time
                # fallback을 사용한다.
                timestamps = None

            return (
                original_clip,
                recapture_clip,
                timestamps,
            )

        finally:
            cap.release()

    def _resize_recaptured_clip(
        self,
        clip: torch.Tensor,
    ) -> torch.Tensor:
        """
        recapture simulation이 끝난 intermediate clip을
        model input size로 resize한다.

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
        clip:
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
        # 2. Source frame decode
        # ------------------------------

        (
            original,
            recapture_base,
            timestamps,
        ) = self._load_clip(
            video_path
        )

        # ------------------------------
        # 3. Synthetic RERECORDED
        # ------------------------------

        if self.train:

            # DataLoader worker의 Python RNG state를 이용해
            # 매 epoch / sample마다 새로운 seed를 생성한다.
            #
            # random.Random(None)을 직접 반복 생성하는 것보다
            # 전체 training seed 체계와 더 잘 맞는다.

            recapture_seed = random.getrandbits(
                63
            )

        else:

            # validation에서는 같은 source에 대해
            # 항상 동일한 synthetic recapture를 생성한다.
            recapture_seed = (
                source_index
                + 10_000_000
            )

        rerecorded = self.recapture(
            recapture_base,
            seed=recapture_seed,
            timestamps=timestamps,
        )

        # intermediate resolution
        # ->
        # model resolution
        rerecorded = (
            self._resize_recaptured_clip(
                rerecorded
            )
        )

        # ------------------------------
        # 4. Common augmentation
        # ------------------------------
        #
        # 일반 blur / resampling / static gamma는
        # RERECORDED만의 특징이 아니므로
        # 학습 중에는 두 class에 독립적으로 적용한다.
        #
        # validation에서는 실제 inference의 ORIGINAL
        # preprocessing을 유지하기 위해 적용하지 않는다.
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
        # 7. Pair 생성
        # ------------------------------

        clips = torch.stack(
            [
                original,
                rerecorded,
            ],
            dim=0,
        )

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