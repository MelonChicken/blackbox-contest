from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from src.config import STAGE2_CHECKPOINT, STAGE2_MODEL
from src.datasets.stage2_dataset import image_paths, preprocess_image_sequence, preprocess_video
from src.inference.stage1 import _video_paths
from src.models.stage2_videomae import Stage2VideoMAE


OUTPUT_COLUMNS = ["ID", "collision_frame", "entry_frame", "evasion_space", "entry_side"]


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _sequence_folders(data_dir: str | Path) -> list[Path]:
    root = Path(data_dir)
    image_root = root / "images"
    if image_root.exists():
        folders = sorted(p for p in image_root.iterdir() if p.is_dir() and image_paths(p))
        if folders:
            return folders
        if image_paths(image_root):
            return [image_root]
    if root.exists():
        folders = sorted(p for p in root.iterdir() if p.is_dir() and image_paths(p))
        if folders:
            return folders
    return [root] if root.exists() and image_paths(root) else []


def _resolve_checkpoint_path(model_dir: str | Path | None = None) -> Path:
    candidates = [Path(model_dir) / "best.pt"] if model_dir is not None else [STAGE2_CHECKPOINT, STAGE2_MODEL / "best.pt"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Stage2 checkpoint not found. Checked: {', '.join(str(path) for path in candidates)}")


def load_stage2_model(checkpoint_path: str | Path, device: torch.device | None = None) -> Stage2VideoMAE:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_config" not in checkpoint:
        raise KeyError("Stage2 checkpoint must contain model_config. Retrain Stage2 with src.train.stage2.fit_stage2().")
    model = Stage2VideoMAE.from_config(checkpoint["model_config"], use_pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device or _device()).eval()
    return model


@torch.inference_mode()
def run_stage2_clip(
    clip: torch.Tensor,
    sampled_frames,
    model: Stage2VideoMAE,
    device: torch.device,
) -> dict[str, int | str]:
    outputs = model(clip.unsqueeze(0).to(device, non_blocking=True))
    collision_idx = int(outputs["collision_logits"].argmax(dim=1).item())
    entry_idx = int(outputs["entry_logits"].argmax(dim=1).item())
    direction_idx = int(outputs["direction_logits"].argmax(dim=1).item())
    avoidance_idx = int(outputs["avoidance_logits"].argmax(dim=1).item())
    return {
        "collision_frame": int(sampled_frames[collision_idx]),
        "entry_frame": int(sampled_frames[entry_idx]),
        "evasion_space": avoidance_idx,
        "entry_side": "RIGHT" if direction_idx == 1 else "LEFT",
    }


def predict_stage2(data_dir, model_dir=None) -> pd.DataFrame:
    device = _device()
    checkpoint_path = _resolve_checkpoint_path(model_dir)
    model = load_stage2_model(checkpoint_path, device)
    num_frames = model.num_frames
    image_size = model.image_size

    rows = []
    for folder in _sequence_folders(data_dir):
        paths = image_paths(folder)
        clip, sampled_frames = preprocess_image_sequence(paths, num_frames=num_frames, image_size=image_size)
        result = run_stage2_clip(clip, sampled_frames, model, device)
        rows.append({"ID": folder.name, **result})

    if not rows:
        for path in _video_paths(Path(data_dir) / "videos"):
            clip, sampled_frames = preprocess_video(path, num_frames=num_frames, image_size=image_size)
            result = run_stage2_clip(clip, sampled_frames, model, device)
            rows.append({"ID": path.stem, **result})

    if not rows:
        raise RuntimeError(f"No Stage2 image folders or videos found under: {data_dir}")

    if device.type == "cuda":
        torch.cuda.empty_cache()
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
