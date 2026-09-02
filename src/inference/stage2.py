from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from src.config import STAGE2_VIDEOMAE_CHECKPOINT
from src.datasets.ccddataset import preprocess_videomae_video
from src.inference.stage1 import _video_paths
from src.models.stage2 import (
    Stage2VideoMAE,
    infer_videomae_config_from_state_dict,
)


PLACEHOLDER_ENTRY_FRAME = pd.NA
PLACEHOLDER_EVASION_SPACE = pd.NA
PLACEHOLDER_ENTRY_SIDE = "TODO"


def _device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "DACON inference requires a CUDA GPU."
        )

    return torch.device("cuda")


def _resolve_checkpoint_path(
    model_dir,
) -> Path:
    if model_dir is None:
        return STAGE2_VIDEOMAE_CHECKPOINT

    model_path = Path(
        model_dir
    )

    candidates = [
        model_path,
        model_path / "videomae" / "best.pt",
        model_path / "best.pt",
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

    state_dict = checkpoint[
        "model_state_dict"
    ]

    num_frames = int(
        checkpoint.get(
            "num_frames",
            16,
        )
    )

    config_dict = checkpoint.get(
        "videomae_config"
    )

    if config_dict is None:
        config_dict = infer_videomae_config_from_state_dict(
            state_dict,
            num_frames=num_frames,
        )

    model = Stage2VideoMAE(
        config_dict=config_dict,
        num_input_frames=num_frames,
    )

    model.load_state_dict(
        state_dict
    )

    model.to(
        device
    ).eval()

    return (
        model,
        num_frames,
    )


def predict_stage2(
    data_dir,
    model_dir=None,
):
    device = _device()

    checkpoint_path = _resolve_checkpoint_path(
        model_dir
    )

    (
        model,
        num_frames,
    ) = _load_stage2_videomae(
        checkpoint_path,
        device,
    )

    videos = _video_paths(
        Path(data_dir) / "videos"
    )

    rows = []

    with torch.inference_mode():
        for path in videos:
            (
                clip,
                sampled_indices,
            ) = preprocess_videomae_video(
                path,
                num_frames=num_frames,
                image_size=224,
            )

            video = clip.unsqueeze(0).to(
                device,
                non_blocking=True,
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):
                outputs = model(
                    video
                )

            pred_sample_idx = int(
                outputs["collision_logits"]
                .argmax(dim=1)
                .item()
            )

            collision_frame = int(
                sampled_indices[
                    pred_sample_idx
                ]
            )

            rows.append(
                {
                    "ID": path.stem,
                    "collision_frame": collision_frame,
                    "entry_frame": PLACEHOLDER_ENTRY_FRAME,
                    "evasion_space": PLACEHOLDER_EVASION_SPACE,
                    "entry_side": PLACEHOLDER_ENTRY_SIDE,
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
