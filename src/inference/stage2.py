from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.config import STAGE2_MODEL, STAGE2_VIDEOMAE_CHECKPOINT
from src.datasets.ccddataset import (
    preprocess_videomae_video,
    sample_uniform_indices,
    transform_videomae_frame,
)
from src.inference.stage1 import _video_paths
from src.models.stage2 import (
    Stage2VideoMAE,
    infer_videomae_config_from_state_dict,
)


IMAGE_EXT = {".jpg", ".jpeg", ".png"}
PLACEHOLDER_ENTRY_FRAME = pd.NA
PLACEHOLDER_EVASION_SPACE = pd.NA


def _device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "DACON inference requires a CUDA GPU."
        )

    return torch.device("cuda")


def _frame_number(path: Path) -> int:
    match = re.search(r"(\d+)$", path.stem)
    return int(match.group(1)) if match else 0


def _stage2_image_paths(folder: Path) -> list[Path]:
    if not folder.exists():
        return []

    return sorted(
        (
            p
            for p
            in folder.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXT
        ),
        key=_frame_number,
    )


def _stage2_sequence_folders(data_dir) -> list[Path]:
    root = Path(data_dir)
    image_root = root / "images"

    if image_root.exists():
        folders = sorted(
            p
            for p
            in image_root.iterdir()
            if p.is_dir() and _stage2_image_paths(p)
        )
        if folders:
            return folders

        if _stage2_image_paths(image_root):
            return [image_root]

    if root.exists():
        folders = sorted(
            p
            for p
            in root.iterdir()
            if p.is_dir() and _stage2_image_paths(p)
        )
        if folders:
            return folders

    return [root] if _stage2_image_paths(root) else []


def preprocess_videomae_images(
    paths: list[Path],
    num_frames: int = 16,
    image_size: int = 224,
) -> tuple[torch.Tensor, np.ndarray]:
    sampled_indices = sample_uniform_indices(
        len(paths),
        num_frames,
    )

    frames = []
    frame_numbers = []

    for index in sampled_indices:
        path = paths[int(index)]
        with Image.open(path) as image:
            frame = np.asarray(image.convert("RGB"))

        frames.append(
            transform_videomae_frame(
                frame,
                image_size=image_size,
            )
        )
        frame_numbers.append(
            _frame_number(path)
        )

    return (
        torch.stack(frames, dim=0),
        np.asarray(frame_numbers, dtype=np.int64),
    )


def _resolve_checkpoint_path(model_dir=None) -> Path:
    if model_dir is None:
        candidates = [
            STAGE2_VIDEOMAE_CHECKPOINT,
            STAGE2_MODEL / "best.pt",
        ]
    else:
        candidates = [
            Path(model_dir) / "best.pt",
        ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "Stage 2 VideoMAE checkpoint not found. "
        f"Checked: {', '.join(str(path) for path in candidates)}"
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

def _load_stage2_videomae(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[Stage2VideoMAE, int]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "Stage 2 checkpoint must contain 'model_state_dict'."
        )

    state_dict = checkpoint["model_state_dict"]
    num_frames = int(checkpoint.get("num_frames", 16))
    config_dict = checkpoint.get("videomae_config", STAGE2_VIDEOMAE_CONFIG)

    model = Stage2VideoMAE(
        config_dict=config_dict,
        num_input_frames=num_frames,
    )

    _load_compatible_state_dict(model, state_dict)
    model.to(device).eval()

    return model, num_frames


def _predict_clip(
    model: Stage2VideoMAE,
    clip: torch.Tensor,
    sampled_frames: np.ndarray,
    device: torch.device,
) -> tuple[int, str]:
    video = clip.unsqueeze(0).to(
        device,
        non_blocking=True,
    )

    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
    ):
        outputs = model(video)

    pred_sample_idx = int(
        outputs["collision_logits"]
        .argmax(dim=1)
        .item()
    )
    side_idx = int(
        outputs["side_logits"]
        .argmax(dim=1)
        .item()
    )

    return (
        int(sampled_frames[pred_sample_idx]),
        "RIGHT" if side_idx else "LEFT",
    )


def predict_stage2(
    data_dir,
    model_dir=None,
):
    device = _device()
    checkpoint_path = _resolve_checkpoint_path(model_dir)
    model, num_frames = _load_stage2_videomae(
        checkpoint_path,
        device,
    )

    rows = []

    with torch.inference_mode():
        folders = _stage2_sequence_folders(data_dir)

        for folder in folders:
            paths = _stage2_image_paths(folder)
            if not paths:
                continue

            clip, sampled_frames = preprocess_videomae_images(
                paths,
                num_frames=num_frames,
                image_size=224,
            )
            collision_frame, entry_side = _predict_clip(
                model,
                clip,
                sampled_frames,
                device,
            )

            rows.append(
                {
                    "ID": folder.name,
                    "collision_frame": collision_frame,
                    "entry_frame": PLACEHOLDER_ENTRY_FRAME,
                    "evasion_space": PLACEHOLDER_EVASION_SPACE,
                    "entry_side": entry_side,
                }
            )

        if not rows:
            videos = _video_paths(
                Path(data_dir) / "videos"
            )

            for path in videos:
                clip, sampled_indices = preprocess_videomae_video(
                    path,
                    num_frames=num_frames,
                    image_size=224,
                )
                collision_frame, entry_side = _predict_clip(
                    model,
                    clip,
                    sampled_indices,
                    device,
                )

                rows.append(
                    {
                        "ID": path.stem,
                        "collision_frame": collision_frame,
                        "entry_frame": PLACEHOLDER_ENTRY_FRAME,
                        "evasion_space": PLACEHOLDER_EVASION_SPACE,
                        "entry_side": entry_side,
                    }
                )

    del model
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



