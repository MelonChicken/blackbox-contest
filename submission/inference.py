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

STAGE2_VIDEOMAE_CONFIG = {
    "image_size": 224,
    "patch_size": 16,
    "num_channels": 3,
    "num_frames": 16,
    "tubelet_size": 2,
    "hidden_size": 384,
    "num_hidden_layers": 12,
    "num_attention_heads": 6,
    "intermediate_size": 1536,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.0,
    "attention_probs_dropout_prob": 0.0,
    "initializer_range": 0.02,
    "layer_norm_eps": 1e-12,
    "qkv_bias": True,
}

class Stage2VideoMAE(nn.Module):
    def __init__(self, config_dict: dict, num_input_frames: int = 16, dropout: float = 0.2):
        super().__init__()
        self.encoder = VideoMAEModel(VideoMAEConfig.from_dict(config_dict))
        config = self.encoder.config
        self.hidden_size = config.hidden_size
        self.patch_size = config.patch_size
        self.tubelet_size = config.tubelet_size
        self.image_size = config.image_size
        self.num_input_frames = num_input_frames
        spatial_size = self.image_size // self.patch_size
        self.num_spatial_tokens = spatial_size * spatial_size
        self.num_temporal_tokens = num_input_frames // self.tubelet_size
        self.temporal_norm = nn.LayerNorm(self.hidden_size)
        self.collision_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.hidden_size, 1))
        self.side_head = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, 2),
        )

    def _temporal_features(self, hidden: torch.Tensor) -> torch.Tensor:
        batch_size = hidden.shape[0]
        expected_tokens = self.num_temporal_tokens * self.num_spatial_tokens
        if hidden.shape[1] != expected_tokens:
            raise RuntimeError(
                "Unexpected VideoMAE token count: "
                f"got={hidden.shape[1]}, expected={expected_tokens}"
            )
        hidden = hidden.reshape(
            batch_size,
            self.num_temporal_tokens,
            self.num_spatial_tokens,
            self.hidden_size,
        )
        return hidden.mean(dim=2)

    def forward(self, video: torch.Tensor) -> dict:
        outputs = self.encoder(pixel_values=video)
        temporal = self.temporal_norm(self._temporal_features(outputs.last_hidden_state))
        collision_logits = self.collision_head(temporal).squeeze(-1)
        collision_logits = F.interpolate(
            collision_logits.unsqueeze(1),
            size=self.num_input_frames,
            mode="linear",
            align_corners=False,
        ).squeeze(1)
        global_feature = temporal.mean(dim=1)
        return {
            "collision_logits": collision_logits,
            "side_logits": self.side_head(global_feature),
            "temporal_features": temporal,
        }


def infer_videomae_config_from_state_dict(state_dict: dict, num_frames: int = 16) -> dict:
    projection = state_dict["encoder.embeddings.patch_embeddings.projection.weight"]
    hidden_size = int(projection.shape[0])
    num_channels = int(projection.shape[1])
    tubelet_size = int(projection.shape[2])
    patch_size = int(projection.shape[3])
    layer_indices = [
        int(key.split(".")[3])
        for key in state_dict
        if key.startswith("encoder.encoder.layer.")
    ]
    if not layer_indices:
        raise RuntimeError("Cannot infer VideoMAE config: encoder layers are missing from checkpoint.")
    intermediate_size = int(state_dict["encoder.encoder.layer.0.intermediate.dense.weight"].shape[0])
    num_attention_heads = {384: 6, 768: 12, 1024: 16}.get(hidden_size)
    if num_attention_heads is None:
        for candidate in range(16, 0, -1):
            if hidden_size % candidate == 0:
                num_attention_heads = candidate
                break
    return VideoMAEConfig(
        image_size=224,
        patch_size=patch_size,
        num_channels=num_channels,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        hidden_size=hidden_size,
        num_hidden_layers=max(layer_indices) + 1,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        qkv_bias=True,
    ).to_dict()


def _frame_number(path: Path):
    match = re.search(r"(\d+)$", path.stem)
    return int(match.group(1)) if match else 0


def _stage2_image_paths(folder: Path):
    if not folder.exists():
        return []
    return sorted(
        (p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXT),
        key=_frame_number,
    )


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


def transform_videomae_frame(frame: np.ndarray, image_size: int = 224) -> torch.Tensor:
    x = torch.from_numpy(frame.copy()).permute(2, 0, 1).float() / 255.0
    _, h, w = x.shape
    scale = image_size / min(h, w)
    new_h = max(image_size, round(h * scale))
    new_w = max(image_size, round(w * scale))
    x = TF.resize(x, [new_h, new_w], antialias=True)
    x = TF.center_crop(x, [image_size, image_size])
    return (x - VIDEOMAE_MEAN) / VIDEOMAE_STD


def sample_uniform_indices(frame_count: int, num_frames: int) -> np.ndarray:
    indices = np.linspace(0, frame_count - 1, num_frames)
    indices = np.round(indices).astype(np.int64)
    return np.clip(indices, 0, frame_count - 1)


def preprocess_videomae_images(paths, num_frames: int = 16, image_size: int = 224):
    sampled_indices = sample_uniform_indices(len(paths), num_frames)
    frames = []
    frame_numbers = []
    for index in sampled_indices:
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

    raise FileNotFoundError(
        f"Stage 2 VideoMAE checkpoint not found: {checkpoint_path}"
    )




def _patch_videomae_attention_biases(model: nn.Module) -> None:
    layers = getattr(model.encoder.encoder, "layer", [])
    for layer in layers:
        attention = layer.attention.attention
        if not hasattr(attention, "q_bias") or attention.q_bias is None:
            continue
        if not hasattr(attention, "k_bias"):
            attention.register_parameter("k_bias", nn.Parameter(torch.zeros_like(attention.q_bias)))

        def _forward_with_key_bias(self, hidden_states, head_mask=None):
            batch_size, _, _ = hidden_states.shape
            keys = F.linear(hidden_states, self.key.weight, self.k_bias)
            values = F.linear(hidden_states, self.value.weight, self.v_bias)
            queries = F.linear(hidden_states, self.query.weight, self.q_bias)

            key_layer = keys.view(batch_size, -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
            value_layer = values.view(batch_size, -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
            query_layer = queries.view(batch_size, -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)

            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) * self.scaling
            attention_probs = F.softmax(attention_scores, dim=-1)
            attention_probs = F.dropout(attention_probs, p=self.dropout_prob, training=self.training)
            if head_mask is not None:
                attention_probs = attention_probs * head_mask

            context_layer = torch.matmul(attention_probs, value_layer)
            context_layer = context_layer.transpose(1, 2).contiguous()
            context_layer = context_layer.view(batch_size, -1, self.all_head_size)
            return context_layer, attention_probs

        attention.forward = _forward_with_key_bias.__get__(attention, attention.__class__)


def _adapt_stage2_state_dict(model: nn.Module, state_dict: dict) -> dict:
    model_state = model.state_dict()
    adapted = {}
    for key, value in state_dict.items():
        mapped_key = key
        if key.endswith(".attention.attention.query.bias"):
            mapped_key = key[: -len("query.bias")] + "q_bias"
        elif key.endswith(".attention.attention.key.bias"):
            mapped_key = key[: -len("key.bias")] + "k_bias"
        elif key.endswith(".attention.attention.value.bias"):
            mapped_key = key[: -len("value.bias")] + "v_bias"

        if mapped_key in model_state:
            adapted[mapped_key] = value
        elif key in model_state:
            adapted[key] = value
    return adapted

def _load_compatible_state_dict(model: nn.Module, state_dict: dict) -> None:
    model_state = model.state_dict()
    adapted_state = _adapt_stage2_state_dict(model, state_dict)
    compatible = {
        key: value
        for key, value
        in adapted_state.items()
        if key in model_state and tuple(model_state[key].shape) == tuple(value.shape)
    }

    total_tensors = len(state_dict)
    total_params = sum(value.numel() for value in state_dict.values() if hasattr(value, "numel"))
    compatible_params = sum(value.numel() for value in compatible.values())

    if len(compatible) != total_tensors or compatible_params != total_params:
        missing = [key for key in model_state if key not in compatible]
        unexpected = [key for key in state_dict if key not in _adapt_stage2_state_dict(model, {key: state_dict[key]})]
        shape_mismatch = [
            (key, tuple(value.shape), tuple(model_state[key].shape))
            for key, value
            in adapted_state.items()
            if key in model_state and tuple(model_state[key].shape) != tuple(value.shape)
        ]
        raise RuntimeError(
            "Incomplete Stage 2 VideoMAE checkpoint load: "
            f"compatible_tensors={len(compatible)}/{total_tensors}, "
            f"compatible_params={compatible_params}/{total_params}, "
            f"missing={missing[:20]}, unexpected={unexpected[:20]}, "
            f"shape_mismatch={shape_mismatch[:20]}"
        )

    model_state.update(compatible)
    model.load_state_dict(model_state, strict=True)

def _load_stage2_videomae(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise KeyError("Stage 2 checkpoint must contain 'model_state_dict'.")
    state_dict = checkpoint["model_state_dict"]
    num_frames = int(checkpoint.get("num_frames", 16))
    config_dict = checkpoint.get("videomae_config", STAGE2_VIDEOMAE_CONFIG)
    model = Stage2VideoMAE(config_dict=config_dict, num_input_frames=num_frames)
    _patch_videomae_attention_biases(model)
    _load_compatible_state_dict(model, state_dict)
    model.to(device).eval()
    return model, num_frames


def predict_stage2(data_dir, model_dir):
    device = _device()
    checkpoint_path = _resolve_stage2_checkpoint(model_dir)
    model, num_frames = _load_stage2_videomae(checkpoint_path, device)
    rows = []
    with torch.inference_mode():
        for folder in _stage2_sequence_folders(data_dir):
            paths = _stage2_image_paths(folder)
            if not paths:
                continue
            clip, sampled_frames = preprocess_videomae_images(paths, num_frames=num_frames, image_size=224)
            video = clip.unsqueeze(0).to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(video)
            pred_sample_idx = int(outputs["collision_logits"].argmax(dim=1).item())
            side_idx = int(outputs["side_logits"].argmax(dim=1).item())
            rows.append(
                {
                    "ID": folder.name,
                    "collision_frame": int(sampled_frames[pred_sample_idx]),
                    "entry_frame": int(sampled_frames[pred_sample_idx]),
                    "evasion_space": 0,
                    "entry_side": "RIGHT" if side_idx else "LEFT",
                }
            )
    del model
    torch.cuda.empty_cache()
    return pd.DataFrame(
        rows,
        columns=["ID", "collision_frame", "entry_frame", "evasion_space", "entry_side"],
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





