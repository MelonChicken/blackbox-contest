from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models.video import mvit_v2_s
from torchvision.transforms import InterpolationMode
from transformers import VideoMAEConfig, VideoMAEModel


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

IMAGE_EXT = {
    ".jpg",
    ".jpeg",
    ".png",
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
    if not root.exists():
        return []

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

        # Evaluation environment??????internet?????????????力?肉??
        # checkpoint????ル늉?? ?????諛몃마??network state??????????
        # pretrained weights?????????????살퓢??????⑥ル럯?????????諛몃마??????▲뀋?????쎛 ????嶺뚮ㅏ援??
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


class Stage3MViT(nn.Module):
    """
    Stage 3:
    Vehicle acceleration and steering classification.
    """

    def __init__(self):
        super().__init__()

        self.backbone = mvit_v2_s(
            weights=None
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
    AIHubStage1Dataset????????ㅻ쿋???????ㅻ깹??
    ?????紐껊괘???????諛몃마??????n????ル늉????frame?????歷?퉭留㏝걡???롫렓?sampling???꿔꺂?????
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

    # Training Dataset????????ㅻ쿋???????ㅻ깹??
    # torch.linspace + round ????
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

    AIHubStage1Dataset??validation ORIGINAL path??
    ??????ㅻ쿋???spatial preprocessing????????꿔꺂?????

        video decode
            ??
        BGR -> RGB
            ??
        cv2 resize -> recapture_size (????????320)
            ??
        Tensor [T,C,H,W]
            ??
        torchvision bilinear + antialias
            ??
        model size (????????224)
            ??
        [C,T,H,W]
            ??
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
            # Training / validation??ORIGINAL????????ㅻ쿋??
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

    # ??? selected frame??decode??? ??? ??棺堉?뤃????
    # ?饔낅떽?????????轅붽틓???????곷뼱?frame?????????????諛몃마???clip ???亦낃콛?????????굩????????
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
    # ?? ??????ㅻ쿋???resize
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

    # ??checkpoint:
    #     recapture_size = 320
    #
    # ????????checkpoint:
    #     key????ル늉?? ??????⑤뜤?嶺뚮Ŋ裕녺댆???몄뿭????size???????
    #     ????????source -> 224 preprocessing????轅붽틓??筌뚮챶夷??
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

        # ??轅붽틓???????곷뼱??????ㅻ쑋?꿔꺂??琉뷩궘??decode??prediction?????怨쀫뮡?????????
        # probability ????????????꿔꺂?????
        #
        # ?????諛몃마???decode????????嶺뚮ㅎ????????紐껊괘???fallback??
        # ?????諛몃마????????????轅붽틓?????????????꿔꺂?????
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

VIDEOMAE_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
VIDEOMAE_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


class Stage2VideoMAE(nn.Module):
    def __init__(self, model_config: dict):
        super().__init__()
        self.num_frames = int(model_config.get("num_frames", 16))
        self.image_size = int(model_config.get("image_size", 224))
        self.direction_classes = int(model_config.get("direction_classes", 2))
        self.avoidance_classes = int(model_config.get("avoidance_classes", 2))
        self.dropout = float(model_config.get("dropout", 0.2))
        hf_config = model_config.get("hf_config")
        if hf_config is None:
            raise KeyError("Stage 2 checkpoint model_config must contain hf_config.")
        self.backbone = VideoMAEModel(VideoMAEConfig.from_dict(hf_config))
        hidden_size = int(self.backbone.config.hidden_size)
        self.head_dropout = nn.Dropout(self.dropout)
        self.collision_head = nn.Linear(hidden_size, 1)
        self.entry_head = nn.Linear(hidden_size, 1)
        self.direction_head = nn.Linear(hidden_size, self.direction_classes)
        self.avoidance_head = nn.Linear(hidden_size, self.avoidance_classes)

    def forward(self, pixel_values: torch.Tensor) -> dict:
        outputs = self.backbone(pixel_values=pixel_values)
        tokens = outputs.last_hidden_state
        batch_size, token_count, hidden_size = tokens.shape
        temporal_count = max(1, self.num_frames // int(self.backbone.config.tubelet_size))
        spatial_count = max(1, token_count // temporal_count)
        temporal = tokens[:, : temporal_count * spatial_count].reshape(
            batch_size, temporal_count, spatial_count, hidden_size
        ).mean(dim=2)
        temporal = self.head_dropout(temporal)
        collision_logits = self.collision_head(temporal).squeeze(-1)
        entry_logits = self.entry_head(temporal).squeeze(-1)
        if collision_logits.shape[1] != self.num_frames:
            collision_logits = F.interpolate(
                collision_logits.unsqueeze(1), size=self.num_frames, mode="linear", align_corners=False
            ).squeeze(1)
            entry_logits = F.interpolate(
                entry_logits.unsqueeze(1), size=self.num_frames, mode="linear", align_corners=False
            ).squeeze(1)
        global_feature = self.head_dropout(temporal.mean(dim=1))
        return {
            "collision_logits": collision_logits,
            "entry_logits": entry_logits,
            "direction_logits": self.direction_head(global_feature),
            "avoidance_logits": self.avoidance_head(global_feature),
        }


class LegacyStage2VideoMAE(nn.Module):
    def __init__(self, model_config: dict):
        super().__init__()
        hf_config = model_config["hf_config"]
        self.encoder = VideoMAEModel(VideoMAEConfig.from_dict(hf_config))
        config = self.encoder.config
        self.hidden_size = int(config.hidden_size)
        self.patch_size = int(config.patch_size)
        self.tubelet_size = int(config.tubelet_size)
        self.image_size = int(config.image_size)
        self.num_frames = int(model_config.get("num_frames", 16))
        spatial_size = self.image_size // self.patch_size
        self.num_spatial_tokens = spatial_size * spatial_size
        self.num_temporal_tokens = self.num_frames // self.tubelet_size
        self.temporal_norm = nn.LayerNorm(self.hidden_size)
        self.collision_head = nn.Sequential(nn.Dropout(0.2), nn.Linear(self.hidden_size, 1))
        self.side_head = nn.Sequential(nn.LayerNorm(self.hidden_size), nn.Dropout(0.2), nn.Linear(self.hidden_size, 2))

    def forward(self, video: torch.Tensor) -> dict:
        outputs = self.encoder(pixel_values=video)
        batch_size = outputs.last_hidden_state.shape[0]
        temporal = outputs.last_hidden_state.reshape(
            batch_size, self.num_temporal_tokens, self.num_spatial_tokens, self.hidden_size
        ).mean(dim=2)
        temporal = self.temporal_norm(temporal)
        collision_logits = self.collision_head(temporal).squeeze(-1)
        collision_logits = F.interpolate(
            collision_logits.unsqueeze(1), size=self.num_frames, mode="linear", align_corners=False
        ).squeeze(1)
        return {"collision_logits": collision_logits, "side_logits": self.side_head(temporal.mean(dim=1))}


def _stage2_model_config_from_checkpoint(checkpoint: dict):
    config = checkpoint.get("model_config")
    if config is not None:
        return config
    config = checkpoint.get("videomae_config")
    if config is None:
        raise KeyError("Stage2 checkpoint contains neither 'model_config' nor 'videomae_config'.")
    return {
        "num_frames": checkpoint.get("num_frames", config.get("num_frames", 16)),
        "image_size": config.get("image_size", 224),
        "hf_config": config,
    }


def _frame_number(path: Path):
    match = re.search(r"(\d+)$", path.stem)
    return int(match.group(1)) if match else 0


def _stage2_image_paths(folder: Path):
    if not folder.exists():
        return []
    return sorted((p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXT), key=_frame_number)


def _stage2_sequence_folders(data_dir):
    root = Path(data_dir)
    image_root = root / "images"
    if image_root.exists():
        folders = sorted(p for p in image_root.iterdir() if p.is_dir() and _stage2_image_paths(p))
        if folders:
            return folders
        if _stage2_image_paths(image_root):
            return [image_root]
    folders = sorted(p for p in root.iterdir() if p.is_dir() and _stage2_image_paths(p)) if root.exists() else []
    if folders:
        return folders
    return [root] if _stage2_image_paths(root) else []


def sample_frame_indices(frame_count: int, num_frames: int) -> np.ndarray:
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if frame_count == 1:
        return np.zeros(num_frames, dtype=np.int64)
    return np.rint(np.linspace(0, frame_count - 1, num_frames)).astype(np.int64)


def transform_videomae_frame(frame: np.ndarray, image_size: int = 224) -> torch.Tensor:
    image = Image.fromarray(frame).convert("RGB")
    width, height = image.size
    scale = image_size / min(width, height)
    resized = (round(height * scale), round(width * scale))
    x = TF.to_tensor(TF.resize(image, resized, antialias=True))
    x = TF.center_crop(x, [image_size, image_size])
    return (x - VIDEOMAE_MEAN) / VIDEOMAE_STD


def preprocess_videomae_images(paths, num_frames: int = 16, image_size: int = 224):
    if not paths:
        raise RuntimeError("Stage 2 image sequence is empty")
    sampled_positions = sample_frame_indices(len(paths), num_frames)
    frames = []
    frame_numbers = []
    for index in sampled_positions:
        path = paths[int(index)]
        with Image.open(path) as image:
            frame = np.asarray(image.convert("RGB"))
        frames.append(transform_videomae_frame(frame, image_size=image_size))
        frame_numbers.append(_frame_number(path))
    return torch.stack(frames, dim=0), np.asarray(frame_numbers, dtype=np.int64)


def _resolve_stage2_checkpoint(model_dir) -> Path:
    checkpoint_path = Path(model_dir) / "best.pt"
    if checkpoint_path.is_file():
        return checkpoint_path
    raise FileNotFoundError(f"Stage 2 checkpoint not found: {checkpoint_path}")


def _load_stage2_videomae(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    config = _stage2_model_config_from_checkpoint(checkpoint)
    model = LegacyStage2VideoMAE(config) if any(key.startswith("encoder.") for key in state_dict) else Stage2VideoMAE(config)
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    return model


def _predict_stage2_clip(model: Stage2VideoMAE, clip: torch.Tensor, sampled_frames: np.ndarray, device: torch.device):
    video = clip.unsqueeze(0).to(device, non_blocking=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        outputs = model(video)
    collision_idx = int(outputs["collision_logits"].argmax(dim=1).item())
    side_logits = outputs["direction_logits"] if "direction_logits" in outputs else outputs["side_logits"]
    direction_idx = int(side_logits.argmax(dim=1).item())
    collision_frame = int(sampled_frames[collision_idx])
    return {
        "collision_frame": collision_frame,
        "entry_frame": collision_frame,
        "evasion_space": 0,
        "entry_side": "RIGHT" if direction_idx == 1 else "LEFT",
    }


def predict_stage2(data_dir, model_dir):
    device = _device()
    model = _load_stage2_videomae(_resolve_stage2_checkpoint(model_dir), device)
    rows = []
    with torch.inference_mode():
        for folder in _stage2_sequence_folders(data_dir):
            paths = _stage2_image_paths(folder)
            if not paths:
                continue
            clip, sampled_frames = preprocess_videomae_images(paths, num_frames=model.num_frames, image_size=model.image_size)
            rows.append({"ID": folder.name, **_predict_stage2_clip(model, clip, sampled_frames, device)})
    del model
    torch.cuda.empty_cache()
    if not rows:
        raise RuntimeError(f"No Stage 2 image folders found under: {data_dir}")
    return pd.DataFrame(rows, columns=["ID", "collision_frame", "entry_frame", "evasion_space", "entry_side"])
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





