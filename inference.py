from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF

from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights
from torchvision.transforms import InterpolationMode


# ============================================================
# Configuration
# ============================================================

S1_MEAN = torch.tensor(
    [0.45, 0.45, 0.45]
)[:, None, None, None]

S1_STD = torch.tensor(
    [0.225, 0.225, 0.225]
)[:, None, None, None]


S3_MEAN = torch.tensor(
    [0.45, 0.45, 0.45]
)[:, None, None]

S3_STD = torch.tensor(
    [0.225, 0.225, 0.225]
)[:, None, None]


VIDEO_EXT = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".m4v",
    ".3gp",
    ".3gpp",
    ".wmv",
}


ACCEL = [
    "ACCELERATING",
    "DECELERATING",
    "CONSTANT",
    "STOPPED",
]


STEER = [
    "LEFT",
    "STRAIGHT",
    "RIGHT",
]


cv2.setNumThreads(1)


# ============================================================
# Common
# ============================================================

def _device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "DACON inference requires a CUDA GPU."
        )

    return torch.device("cuda")


def _video_paths(
    root: Path,
):
    return sorted(
        p
        for p in root.rglob("*")
        if (
            p.is_file()
            and p.suffix.lower() in VIDEO_EXT
        )
    )


# ============================================================
# Models
# ============================================================

class Stage1MViT(nn.Module):
    """
    Stage 1 ORIGINAL / RERECORDED
    binary classification model.
    """

    def __init__(self):
        super().__init__()

        # Evaluation environment에서는 internet을 사용할 수 없고
        # checkpoint가 전체 network state를 포함하므로
        # pretrained weights를 여기서 다시 불러올 필요가 없다.
        self.net = mvit_v2_s(
            weights=None
        )

        self.net.head[1] = nn.Linear(
            self.net.head[1].in_features,
            2,
        )

    def forward(
        self,
        x,
    ):
        return self.net(x)


class Stage2Temporal(nn.Module):
    """
    Stage 2:
    Accident timing and scene analysis.
    """

    def __init__(self):
        super().__init__()

        self.r = nn.GRU(
            input_size=512,
            hidden_size=192,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.15,
        )

        self.tc = nn.Linear(
            384,
            1,
        )

        self.te = nn.Linear(
            384,
            1,
        )

        self.scene = nn.Sequential(
            nn.Linear(
                768,
                192,
            ),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(
                192,
                4,
            ),
        )

    def logits(
        self,
        x,
    ):
        h, _ = self.r(x)

        collision = (
            self.tc(h)
            .squeeze(-1)
        )

        entry = (
            self.te(h)
            .squeeze(-1)
        )

        return (
            collision,
            entry,
            h,
        )

    def forward(
        self,
        x,
    ):
        collision, entry, h = (
            self.logits(x)
        )

        ci = collision.argmax(
            1
        )

        ei = entry.argmax(
            1
        )

        b = torch.arange(
            len(h),
            device=h.device,
        )

        scene = self.scene(
            torch.cat(
                [
                    h[b, ci],
                    h[b, ei],
                ],
                dim=1,
            )
        )

        return (
            ci,
            ei,
            scene,
        )


class Stage3MViT(nn.Module):
    """
    Stage 3:
    Vehicle acceleration and steering classification.
    """

    def __init__(self):
        super().__init__()

        self.backbone = mvit_v2_s(
            weights=MViT_V2_S_Weights.KINETICS400_V1
        )

        dim = (
            self.backbone
            .head[1]
            .in_features
        )

        self.backbone.head = (
            nn.Identity()
        )

        self.accel = nn.Linear(
            dim,
            4,
        )

        self.steer = nn.Linear(
            dim,
            3,
        )

    def forward(
        self,
        x,
    ):
        z = self.backbone(x)

        return (
            self.accel(z),
            self.steer(z),
        )


# ============================================================
# Stage 1
# ============================================================

def _clip_ids(
    path: Path,
    n: int,
    slot: int = 0,
    slots: int = 1,
):
    """
    AIHubStage1Dataset과 동일하게
    영상 전체에서 n개의 frame을 균등 sampling한다.
    """

    cap = cv2.VideoCapture(
        str(path)
    )

    if not cap.isOpened():
        raise ValueError(
            f"cannot open video: {path.name}"
        )

    try:
        total = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

    finally:
        cap.release()

    if total <= 0:
        raise ValueError(
            f"invalid frame count: {path.name}"
        )

    # Training Dataset과 동일하게
    # torch.linspace + round 사용
    return (
        torch.linspace(
            0,
            total - 1,
            steps=n,
        )
        .round()
        .long()
        .tolist()
    )


def _decode_stage1_clip(
    path: Path,
    size: int,
    recapture_size: int,
    frame_ids,
):
    """
    Stage 1 inference preprocessing.

    AIHubStage1Dataset의 validation ORIGINAL path와
    동일한 spatial preprocessing을 사용한다.

        video decode
            ↓
        BGR -> RGB
            ↓
        cv2 resize -> recapture_size (기본 320)
            ↓
        Tensor [T,C,H,W]
            ↓
        torchvision bilinear + antialias
            ↓
        model size (기본 224)
            ↓
        [C,T,H,W]
            ↓
        normalization
    """

    cap = cv2.VideoCapture(
        str(path)
    )

    if not cap.isOpened():
        raise ValueError(
            f"cannot open video: {path.name}"
        )

    frames = []

    try:
        for idx in frame_ids:

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                int(idx),
            )

            ok, bgr = cap.read()

            if (
                not ok
                or bgr is None
            ):
                continue

            # ------------------------------
            # BGR -> RGB
            # ------------------------------

            rgb = cv2.cvtColor(
                bgr,
                cv2.COLOR_BGR2RGB,
            )

            # ------------------------------
            # Shared intermediate resize
            #
            # Training / validation의 ORIGINAL과 동일
            # ------------------------------

            rgb = cv2.resize(
                rgb,
                (
                    recapture_size,
                    recapture_size,
                ),
                interpolation=(
                    cv2.INTER_LINEAR
                ),
            )

            # [H,W,C]
            # ->
            # [C,H,W]

            frame = (
                torch.from_numpy(
                    rgb
                )
                .permute(
                    2,
                    0,
                    1,
                )
                .contiguous()
                .float()
                / 255.0
            )

            frames.append(
                frame
            )

    finally:
        cap.release()

    if not frames:
        raise ValueError(
            f"cannot decode video: {path.name}"
        )

    # 일부 selected frame이 decode되지 않은 경우
    # 마지막 정상 frame으로 필요한 clip 길이를 채운다.
    while len(frames) < len(frame_ids):

        frames.append(
            frames[-1].clone()
        )

    # ------------------------------
    # [T,C,320,320]
    # ------------------------------

    clip = torch.stack(
        frames,
        dim=0,
    )

    # ------------------------------
    # Dataset._resize_to_model_size()
    # 와 동일한 resize
    #
    # [T,C,320,320]
    # ->
    # [T,C,224,224]
    # ------------------------------

    clip = TF.resize(
        clip,
        [
            size,
            size,
        ],
        interpolation=(
            InterpolationMode.BILINEAR
        ),
        antialias=True,
    )

    # ------------------------------
    # [T,C,H,W]
    # ->
    # [C,T,H,W]
    # ------------------------------

    x = clip.permute(
        1,
        0,
        2,
        3,
    ).contiguous()

    # ------------------------------
    # Normalize
    # ------------------------------

    return (
        x - S1_MEAN
    ) / S1_STD


class Stage1Clips(Dataset):
    """
    Stage 1 inference dataset.
    """

    def __init__(
        self,
        videos,
        slots,
        size,
        frames,
        recapture_size,
    ):
        self.videos = videos
        self.slots = slots
        self.size = size
        self.frames = frames
        self.recapture_size = (
            recapture_size
        )

    def __len__(self):
        return (
            len(self.videos)
            * self.slots
        )

    def __getitem__(
        self,
        index,
    ):
        video_index = (
            index
            // self.slots
        )

        slot = (
            index
            % self.slots
        )

        path = self.videos[
            video_index
        ]

        try:
            frame_ids = _clip_ids(
                path,
                self.frames,
                slot,
                self.slots,
            )

            x = _decode_stage1_clip(
                path=path,
                size=self.size,
                recapture_size=(
                    self.recapture_size
                ),
                frame_ids=frame_ids,
            )

            valid = 1

        except Exception:

            x = torch.zeros(
                3,
                self.frames,
                self.size,
                self.size,
            )

            valid = 0

        return (
            x,
            video_index,
            valid,
        )


def predict_stage1(
    data_dir,
    model_dir,
):
    device = _device()

    # ------------------------------
    # Checkpoint
    # ------------------------------

    checkpoint = torch.load(
        Path(model_dir)
        / "best.pt",
        map_location="cpu",
        weights_only=False,
    )

    size = int(
        checkpoint["size"]
    )

    frames = int(
        checkpoint["frames"]
    )

    # 새 checkpoint:
    #     recapture_size = 320
    #
    # 기존 checkpoint:
    #     key가 없으므로 size를 사용해
    #     기존 source -> 224 preprocessing과 호환
    recapture_size = int(
        checkpoint.get(
            "recapture_size",
            size,
        )
    )

    # ------------------------------
    # Model
    # ------------------------------

    model = Stage1MViT()

    model.net.load_state_dict(
        checkpoint["model"]
    )

    model.to(
        device
    ).eval()

    # ------------------------------
    # Videos
    # ------------------------------

    videos = _video_paths(
        Path(data_dir)
        / "videos"
    )

    slots = 1

    dataset = Stage1Clips(
        videos=videos,
        slots=slots,
        size=size,
        frames=frames,
        recapture_size=(
            recapture_size
        ),
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        num_workers=4,
        pin_memory=True,
    )

    scores = [
        []
        for _ in videos
    ]

    # ------------------------------
    # Inference
    # ------------------------------

    with torch.inference_mode():

        for (
            clips,
            video_indices,
            valid,
        ) in loader:

            clips = clips.to(
                device,
                non_blocking=True,
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):

                logits = model(
                    clips
                )

                prob = torch.softmax(
                    logits,
                    dim=1,
                )[:, 1]

            for (
                idx,
                value,
                ok,
            ) in zip(
                video_indices.tolist(),
                prob.float()
                .cpu()
                .tolist(),
                valid.tolist(),
            ):

                if ok:
                    scores[idx].append(
                        float(value)
                    )

    # ------------------------------
    # Prediction
    # ------------------------------

    rows = []

    for (
        path,
        values,
    ) in zip(
        videos,
        scores,
    ):

        # 정상적으로 decode된 prediction이 존재하면
        # probability 평균을 사용한다.
        #
        # 완전히 decode할 수 없는 영상의 fallback은
        # 현재 기존 정책을 유지한다.
        probability = (
            float(
                np.mean(values)
            )
            if values
            else 1.0
        )

        answer = (
            "RERECORDED"
            if probability >= 0.5
            else "ORIGINAL"
        )

        rows.append(
            {
                "ID":
                    path.stem,

                "answer":
                    answer,
            }
        )

    del model

    torch.cuda.empty_cache()

    return pd.DataFrame(
        rows,
        columns=[
            "ID",
            "answer",
        ],
    )


# ============================================================
# Stage 2
# ============================================================

class Stage2Frames(Dataset):
    def __init__(
        self,
        paths,
        transform,
    ):
        self.paths = paths
        self.transform = transform

    def __len__(
        self,
    ):
        return len(
            self.paths
        )

    def __getitem__(
        self,
        index,
    ):
        with Image.open(
            self.paths[index]
        ) as image:

            image = image.convert(
                "RGB"
            )

            return self.transform(
                image
            )


def _frame_number(
    path: Path,
):
    match = re.search(
        r"(\d+)$",
        path.stem,
    )

    return (
        int(
            match.group(1)
        )
        if match
        else 0
    )


def predict_stage2(
    data_dir,
    model_dir,
):
    device = _device()

    model_dir = Path(
        model_dir
    )

    transform = (
        ResNet18_Weights
        .IMAGENET1K_V1
        .transforms()
    )

    # ------------------------
    # ResNet18 backbone
    # ------------------------

    backbone = resnet18(
        weights=None
    )

    backbone_state = torch.load(
        model_dir
        / "resnet18-f37072fd.pth",
        map_location="cpu",
        weights_only=True,
    )

    backbone.load_state_dict(
        backbone_state
    )

    backbone.fc = nn.Identity()

    backbone.to(
        device
    ).eval()

    # ------------------------
    # Temporal model
    # ------------------------

    temporal = Stage2Temporal()

    temporal_checkpoint = (
        torch.load(
            model_dir
            / "best.pt",
            map_location="cpu",
            weights_only=False,
        )
    )

    temporal.load_state_dict(
        temporal_checkpoint[
            "model"
        ]
    )

    temporal.to(
        device
    ).eval()

    # ------------------------
    # Data
    # ------------------------

    image_root = (
        Path(data_dir)
        / "images"
    )

    folders = sorted(
        p
        for p in image_root.iterdir()
        if p.is_dir()
    )

    rows = []

    with torch.inference_mode():

        for folder in folders:

            paths = sorted(
                (
                    p
                    for p
                    in folder.iterdir()
                    if (
                        p.suffix.lower()
                        in {
                            ".jpg",
                            ".jpeg",
                            ".png",
                        }
                    )
                ),
                key=_frame_number,
            )

            if not paths:
                continue

            loader = DataLoader(
                Stage2Frames(
                    paths,
                    transform,
                ),
                batch_size=256,
                num_workers=6,
                pin_memory=True,
            )

            features = []

            for images in loader:

                images = images.to(
                    device,
                    non_blocking=True,
                )

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                ):

                    feature = backbone(
                        images
                    )

                features.append(
                    feature
                    .float()
                    .cpu()
                )

            sequence = (
                torch.cat(
                    features
                )
                [None]
                .to(device)
            )

            (
                collision_idx,
                entry_idx,
                scene,
            ) = temporal(
                sequence
            )

            frame_numbers = [
                _frame_number(
                    path
                )
                for path
                in paths
            ]

            collision_idx = int(
                collision_idx.item()
            )

            entry_idx = int(
                entry_idx.item()
            )

            evasion_space = int(
                scene[
                    :,
                    :2,
                ]
                .argmax(1)
                .item()
            )

            entry_side_idx = int(
                scene[
                    :,
                    2:,
                ]
                .argmax(1)
                .item()
            )

            entry_side = (
                "RIGHT"
                if entry_side_idx
                else "LEFT"
            )

            rows.append(
                {
                    "ID":
                        folder.name,

                    "collision_frame":
                        frame_numbers[
                            collision_idx
                        ],

                    "entry_frame":
                        frame_numbers[
                            entry_idx
                        ],

                    "evasion_space":
                        evasion_space,

                    "entry_side":
                        entry_side,
                }
            )

    del backbone
    del temporal

    torch.cuda.empty_cache()

    return pd.DataFrame(
        rows,
        columns=[
            "ID",
            "collision_frame",
            "entry_frame",
            "evasion_space",
            "entry_side",
        ],
    )


# ============================================================
# Stage 3
# ============================================================

def _stage3_frames(
    path: Path,
):
    capture = cv2.VideoCapture(
        str(path)
    )

    frames = []

    while True:

        ok, bgr = (
            capture.read()
        )

        if not ok:
            break

        rgb = cv2.cvtColor(
            bgr,
            cv2.COLOR_BGR2RGB,
        )

        image = Image.fromarray(
            rgb
        )

        width, height = (
            image.size
        )

        scale = (
            256
            / min(
                width,
                height,
            )
        )

        image = image.resize(
            (
                round(
                    width
                    * scale
                ),
                round(
                    height
                    * scale
                ),
            )
        )

        width, height = (
            image.size
        )

        x = (
            width
            - 224
        ) // 2

        y = (
            height
            - 224
        ) // 2

        image = image.crop(
            (
                x,
                y,
                x + 224,
                y + 224,
            )
        )

        frame = (
            torch.from_numpy(
                np.asarray(
                    image
                ).copy()
            )
            .permute(
                2,
                0,
                1,
            )
            .to(
                torch.uint8
            )
        )

        frames.append(
            frame
        )

    capture.release()

    if not frames:
        raise ValueError(
            f"cannot decode video: "
            f"{path.name}"
        )

    return torch.stack(
        frames
    )


def predict_stage3(
    data_dir,
    model_dir,
):
    device = _device()

    checkpoint = torch.load(
        Path(model_dir)
        / "best.pt",
        map_location="cpu",
        weights_only=False,
    )

    model = Stage3MViT()

    model.load_state_dict(
        checkpoint["model"]
    )

    model.to(
        device
    ).eval()

    videos = _video_paths(
        Path(data_dir)
        / "videos"
    )

    rows = []

    with torch.inference_mode():

        for path in videos:

            frames = (
                _stage3_frames(
                    path
                )
            )

            count = len(
                frames
            )

            centers = np.arange(
                count
            )

            accel_predictions = []
            steer_predictions = []

            for start in range(
                0,
                count,
                8,
            ):

                center = centers[
                    start:
                    start + 8
                ]

                indices = np.clip(
                    (
                        center[:, None]
                        - 8
                        + np.arange(
                            16
                        )[None, :]
                    ),
                    0,
                    count - 1,
                )

                indices = (
                    torch.from_numpy(
                        indices
                    )
                )

                clips = (
                    frames[
                        indices
                    ]
                    .permute(
                        0,
                        2,
                        1,
                        3,
                        4,
                    )
                    .float()
                    / 255.0
                )

                clips = (
                    clips
                    - S3_MEAN[
                        None,
                        :,
                        None,
                        :,
                        :,
                    ]
                ) / S3_STD[
                    None,
                    :,
                    None,
                    :,
                    :,
                ]

                clips = clips.to(
                    device,
                    non_blocking=True,
                )

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                ):

                    (
                        accel_logits,
                        steer_logits,
                    ) = model(
                        clips
                    )

                accel_predictions.extend(
                    accel_logits
                    .argmax(1)
                    .cpu()
                    .tolist()
                )

                steer_predictions.extend(
                    steer_logits
                    .argmax(1)
                    .cpu()
                    .tolist()
                )

            for (
                sample_index,
                (
                    accel,
                    steer,
                ),
            ) in enumerate(
                zip(
                    accel_predictions,
                    steer_predictions,
                )
            ):

                rows.append(
                    {
                        "ID":
                            path.stem,

                        "sample_index":
                            sample_index,

                        "accel_label":
                            ACCEL[
                                accel
                            ],

                        "steer_label":
                            STEER[
                                steer
                            ],
                    }
                )

    del model

    torch.cuda.empty_cache()

    return pd.DataFrame(
        rows,
        columns=[
            "ID",
            "sample_index",
            "accel_label",
            "steer_label",
        ],
    )