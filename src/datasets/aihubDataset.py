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

    하나의 source video로부터 두 sample을 만든다.

    ORIGINAL:
        label = 0
        원본 블랙박스 영상

    RERECORDED:
        label = 1
        원본 영상에 synthetic recapture augmentation 적용

    반환 shape:
        [C, T, H, W]

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

        # AI-Hub 데이터 root
        #
        # ex)
        # data/stage1/aihub597
        self.root = Path(root)

        # manifest CSV
        #
        # ex)
        # manifest/train.csv
        # manifest/val.csv
        self.manifest = Path(manifest)

        # MViT 입력 frame 수
        self.frames = frames

        # MViT 입력 spatial size
        self.size = size

        # Train / Validation 여부
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
        # 3. Normalization 설정
        # ------------------------------

        # torch.tensor()가 아니라
        # as_tensor()를 사용하여
        # mean/std가 이미 Tensor인 경우에도
        # 불필요한 copy warning이 발생하지 않게 한다.

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

        self.recapture = (
            RecaptureTransform()
        )


    def __len__(self):
        """
        원본 영상 하나에서

        ORIGINAL   1개
        RERECORDED 1개

        두 sample을 만든다.

        따라서 Dataset 길이는
        manifest의 source video 수 × 2.
        """

        return len(self.df) * 2


    def _load_clip(
        self,
        video_path: Path,
    ):
        """
        전체 영상을 메모리에 올리지 않고
        필요한 frame만 선택적으로 읽는다.

        기존 torchvision.io.read_video()는
        전체 영상을 decode한 뒤 메모리에 적재하기 때문에
        고해상도 블랙박스 영상에서 RAM 부족이 발생할 수 있다.

        이 함수에서는 OpenCV VideoCapture를 이용해
        전체 영상에서 균등하게 self.frames개의 위치를
        선택하여 필요한 frame만 읽는다.

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

            # 전체 영상을 self.frames개 구간으로
            # 균등하게 sampling한다.
            #
            # 예:
            #
            # 1000 frames
            # →
            # 0, 66, 133, ..., 999

            indices = torch.linspace(
                0,
                total_frames - 1,
                steps=self.frames,
            ).round().long().tolist()


            frames = []


            # --------------------------
            # 선택된 frame만 decode
            # --------------------------

            for frame_index in indices:

                # 필요한 frame 위치로 이동
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
                # BGR → RGB
                # ----------------------

                # OpenCV는 BGR 순서로 영상을 읽지만
                # PyTorch 모델은 RGB를 사용한다.

                frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB,
                )


                # ----------------------
                # Resize
                # ----------------------

                # 고해상도 frame을
                # 즉시 MViT 입력 크기로 줄인다.
                #
                # 따라서 원본 해상도 frame을
                # 메모리에 계속 보관하지 않는다.

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
                # numpy → Tensor
                # ----------------------

                # [H, W, C]
                # →
                # [C, H, W]

                frame = torch.from_numpy(
                    frame
                ).permute(
                    2,
                    0,
                    1,
                ).contiguous()


                # uint8 [0,255]
                # →
                # float32 [0,1]

                frame = (
                    frame.float()
                    / 255.0
                )


                frames.append(
                    frame
                )


            # --------------------------
            # Frame stack
            # --------------------------

            # list:
            # T × [C,H,W]
            #
            # →
            #
            # Tensor:
            # [T,C,H,W]

            clip = torch.stack(
                frames,
                dim=0,
            )

            return clip


        finally:
            # 오류가 발생하더라도
            # VideoCapture resource를 해제한다.
            cap.release()


    def _normalize(
        self,
        clip: torch.Tensor,
    ):
        """
        clip:
            [T,C,H,W]

        Stage 1 mean/std normalization.
        """

        return (
            clip - self.mean
        ) / self.std


    def __getitem__(
        self,
        index,
    ):
        """
        Dataset index를 source video와
        Stage 1 label로 변환한다.

        예:

        index = 0
            source_index = 0
            label = 0
            → video 0 ORIGINAL

        index = 1
            source_index = 0
            label = 1
            → video 0 RERECORDED

        index = 2
            source_index = 1
            label = 0
            → video 1 ORIGINAL

        index = 3
            source_index = 1
            label = 1
            → video 1 RERECORDED
        """

        # ------------------------------
        # 1. Source video 선택
        # ------------------------------

        source_index = (
            index // 2
        )


        # ------------------------------
        # 2. Label 결정
        # ------------------------------

        # 짝수 index
        # → ORIGINAL
        #
        # 홀수 index
        # → RERECORDED

        label = (
            index % 2
        )


        # ------------------------------
        # 3. Manifest row
        # ------------------------------

        row = self.df.iloc[
            source_index
        ]


        # ------------------------------
        # 4. 실제 video path
        # ------------------------------

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
        # 5. Video clip 로드
        # ------------------------------

        # 전체 영상을 읽지 않고
        # 필요한 16 frame만 읽는다.
        #
        # output:
        #
        # [T,C,H,W]
        #
        # ex)
        #
        # [16,3,224,224]

        clip = self._load_clip(
            video_path
        )


        # ------------------------------
        # 6. RERECORDED 생성
        # ------------------------------

        if label == 1:

            if self.train:
                # Training에서는 매번
                # 다른 synthetic artifact를 생성한다.
                seed = None

            else:
                # Validation에서는 동일 source에 대해
                # 항상 동일한 synthetic artifact를 사용한다.
                #
                # Validation Macro-F1이 augmentation
                # randomness로 흔들리는 것을 방지한다.
                seed = source_index


            clip = self.recapture(
                clip,
                seed=seed,
            )


        # ------------------------------
        # 7. Normalize
        # ------------------------------

        clip = self._normalize(
            clip
        )


        # ------------------------------
        # 8. MViT input shape
        # ------------------------------

        # 현재:
        #
        # [T,C,H,W]
        #
        # MViT 입력:
        #
        # [C,T,H,W]

        clip = clip.permute(
            1,
            0,
            2,
            3,
        ).contiguous()


        # ------------------------------
        # 9. 반환
        # ------------------------------

        return (
            clip,
            label,
        )