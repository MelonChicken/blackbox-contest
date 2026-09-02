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

    model.load_state_dict(state_dict)
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



