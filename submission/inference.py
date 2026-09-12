# -*- coding: latin-1 -*-
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
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from model.stage1 import Stage1MViT
from model.stage2 import LegacyStage2VideoMAE, Stage2VideoMAE
from model.stage3 import Stage3MViT, Stage3ResNetGRU, Stage3TartanVOGRU


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

STAGE3_ENSEMBLE = True
STAGE3_MVIT_CHECKPOINT = "mvit_best.pt"
STAGE3_RESNET_GRU_CHECKPOINT = "resnet_gru_best.pt"


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

def _tartanvo_config(checkpoint: dict) -> dict:
    config = dict(checkpoint.get("model_config") or {})
    state = checkpoint.get("model", {})
    input_size = state.get("gru.weight_ih_l0")
    if input_size is not None:
        config.setdefault("tartanvo_feature", "pose" if input_size.shape[1] == 6 else "latent")
    if "feature_norm.weight" in state:
        config.setdefault("tartanvo_feature_norm", "layernorm")
    if "pose_norm.weight" in state:
        config.setdefault("tartanvo_pose_norm", "layernorm")
    return config


def _stage3_checkpoint_path(model_dir) -> Path:
    root = Path(model_dir)
    tartan = root / "tartanvo_best.pt"
    return tartan if tartan.is_file() else root / "best.pt"


def _stage3_model(arch: str, checkpoint: dict | None = None):
    if arch == "mvit":
        return Stage3MViT()
    if arch == "resnet18_gru":
        return Stage3ResNetGRU()
    if arch == "tartanvo_gru":
        return Stage3TartanVOGRU(load_pretrained=False, model_config=_tartanvo_config(checkpoint or {}))
    raise ValueError(f"Unknown Stage3 arch: {arch}")


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
    AIHubStage1Dataset??????????�퓢?????????�뭐??
    ?????筌뤾?�愿????????�쏅챶留??????n??????�뒌????frame?????????�??�괌???濡ル??sampling???轅붽??????
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

    # Training Dataset??????????�퓢?????????�뭐??
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
    ????????�퓢???spatial preprocessing????????轅붽??????

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
            # Training / validation??ORIGINAL??????????�퓢??
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

    # ??? selected frame??decode??? ??? ??汝뷴??�????
    # ??�붾굝??????????饔낅??????????�룸??frame??????????????�쏅챶留???clip ?????굛肄??????????????????
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
    # ?? ????????�퓢???resize
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
    #     key??????�뒌?? ????????�쑄??�ル??��?�뺣?????紐꾨?????size???????
    #     ????????source -> 224 preprocessing????饔낅????嶺뚮??��?�ㅇ??
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

        # ??饔낅??????????�룸?????????�몝?轅붽???筌뚮??��??decode??prediction???????�???????????
        # probability ????????????轅붽??????
        #
        # ??????�쏅챶留???decode?????????�ル??????????筌뤾?�愿???fallback??
        # ??????�쏅챶留????????????饔낅???????????????轅붽??????
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

        height, width = rgb.shape[:2]

        scale = 224 / min(
            height,
            width,
        )

        height = max(
            224,
            round(height * scale),
        )

        width = max(
            224,
            round(width * scale),
        )

        rgb = cv2.resize(
            rgb,
            (
                width,
                height,
            ),
            interpolation=cv2.INTER_AREA,
        )

        x = (
            width
            - 224
        ) // 2

        y = (
            height
            - 224
        ) // 2

        rgb = rgb[
            y:y + 224,
            x:x + 224,
        ]

        frame = (
            torch.from_numpy(
                rgb.copy()
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


def _load_stage3_checkpoint(model_dir, filename, expected_arch):
    checkpoint = torch.load(
        Path(model_dir) / filename,
        map_location="cpu",
        weights_only=False,
    )
    arch = checkpoint.get("arch") or "mvit"
    if arch != expected_arch:
        raise ValueError(f"{filename} arch mismatch: expected {expected_arch}, got {arch}")
    model = _stage3_model(arch, checkpoint)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model

def predict_stage3(
    data_dir,
    model_dir,
):
    device = _device()

    checkpoint = torch.load(
        _stage3_checkpoint_path(model_dir),
        map_location="cpu",
        weights_only=False,
    )
    arch = checkpoint.get("arch") or "mvit"
    use_ensemble = STAGE3_ENSEMBLE and arch != "tartanvo_gru"

    if use_ensemble:
        mvit_model = _load_stage3_checkpoint(model_dir, STAGE3_MVIT_CHECKPOINT, "mvit").to(device).eval()
        resnet_model = _load_stage3_checkpoint(model_dir, STAGE3_RESNET_GRU_CHECKPOINT, "resnet18_gru").to(device).eval()
        model = None
    else:
        model = _stage3_model(arch, checkpoint)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.to(device).eval()

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
            window_batch = 1 if arch == "tartanvo_gru" else 8

            for start in range(
                0,
                count,
                window_batch,
            ):

                center = centers[
                    start:
                    start + window_batch
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

                    if use_ensemble:
                        accel_logits, _ = mvit_model(clips)
                        _, steer_logits = resnet_model(clips)
                    else:
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





