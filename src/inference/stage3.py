from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.datasets.stage3_sampling import build_centered_clip_indices
from src.inference.stage1 import _video_paths
from src.models.stage3_vjepa import Stage3VJEPA

ACCEL = ["ACCELERATING", "DECELERATING", "CONSTANT", "STOPPED"]
STEER = ["LEFT", "STRAIGHT", "RIGHT"]
VJEPA_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None]
VJEPA_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None]


def _device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("DACON inference requires a CUDA GPU.")
    return torch.device("cuda")


def _stage3_frames(path: Path):
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, bgr = capture.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        width, height = image.size
        scale = 224 / min(width, height)
        image = image.resize((max(224, round(width * scale)), max(224, round(height * scale))))
        width, height = image.size
        x, y = (width - 224) // 2, (height - 224) // 2
        image = image.crop((x, y, x + 224, y + 224))
        frames.append(torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).to(torch.uint8))
    capture.release()
    if not frames:
        raise ValueError(f"cannot decode video: {path.name}")
    return torch.stack(frames)


def _stage3_checkpoint_path(model_dir) -> Path:
    root = Path(model_dir)
    candidates = [root / "vjepa" / "best.pt", root / "best.pt"]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"missing Stage3 V-JEPA checkpoint under: {root}")


def _stage3_encoder_path(model_dir, checkpoint: dict) -> Path:
    root = Path(model_dir)
    filename = checkpoint.get("encoder_filename")
    candidates = []
    if filename:
        candidates.append(root / filename)
    candidates.extend([
        root / "vitl16.pth.tar",
        root / "vitl16.pth",
        root.parent / "vitl16.pth.tar",
        root.parent / "vitl16.pth",
    ])
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"missing Stage3 V-JEPA encoder under: {root}")


def _stage3_frame_sampling(checkpoint: dict) -> tuple[int, list[int]]:
    source_num_frames = int(checkpoint.get("source_num_frames", 16))
    positions = [int(value) for value in checkpoint.get("frame_positions", range(source_num_frames))]
    if source_num_frames <= 0 or not positions or min(positions) < 0 or max(positions) >= source_num_frames:
        raise ValueError("invalid Stage3 frame sampling metadata")
    return source_num_frames, positions


def predict_stage3(data_dir, model_dir):
    device = _device()
    checkpoint = torch.load(_stage3_checkpoint_path(model_dir), map_location="cpu", weights_only=True)
    arch = checkpoint.get("arch")
    if arch != "vjepa_vitl_224":
        raise ValueError(f"Stage3 inference requires vjepa_vitl_224, got: {arch}")
    model = Stage3VJEPA(_stage3_encoder_path(model_dir, checkpoint))
    model.head.load_state_dict(checkpoint["head"], strict=True)
    model.to(device).eval()
    source_num_frames, frame_positions = _stage3_frame_sampling(checkpoint)
    videos = _video_paths(Path(data_dir) / "videos")
    rows = []
    with torch.inference_mode():
        for path in videos:
            frames = _stage3_frames(path)
            count = len(frames)
            accel_predictions, steer_predictions = [], []
            for start in range(0, count, 1):
                center = np.arange(count)[start : start + 1]
                indices = np.asarray(
                    [
                        np.asarray(build_centered_clip_indices(int(c), count, source_num_frames))[frame_positions]
                        for c in center
                    ],
                    dtype=np.int64,
                )
                clips = frames[torch.from_numpy(indices)].permute(0, 2, 1, 3, 4).float() / 255.0
                clips = (clips - VJEPA_MEAN[None, :, None, :, :]) / VJEPA_STD[None, :, None, :, :]
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    accel_logits, steer_logits = model(clips.to(device, non_blocking=True))
                accel_predictions.extend(accel_logits.argmax(1).cpu().tolist())
                steer_predictions.extend(steer_logits.argmax(1).cpu().tolist())
            for sample_index, (accel, steer) in enumerate(zip(accel_predictions, steer_predictions)):
                rows.append({"ID": path.stem, "sample_index": sample_index, "accel_label": ACCEL[accel], "steer_label": STEER[steer]})
    del model
    torch.cuda.empty_cache()
    return pd.DataFrame(rows, columns=["ID", "sample_index", "accel_label", "steer_label"])
