from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.config import S3_MEAN, S3_STD
from src.datasets.stage3_sampling import build_centered_clip_indices
from src.inference.stage1 import _video_paths
from src.models.stage3 import Stage3MViT

ACCEL = ["ACCELERATING", "DECELERATING", "CONSTANT", "STOPPED"]
STEER = ["LEFT", "STRAIGHT", "RIGHT"]


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
    return Path(model_dir) / "best.pt"


def predict_stage3(data_dir, model_dir):
    device = _device()
    checkpoint = torch.load(_stage3_checkpoint_path(model_dir), map_location="cpu", weights_only=True)
    arch = checkpoint.get("arch", "mvit_v2_s")
    if arch not in {"mvit_v2_s", "mvit"}:
        raise ValueError(f"Stage3 inference supports only mvit_v2_s checkpoints, got: {arch}")
    model = Stage3MViT(pretrained=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    videos = _video_paths(Path(data_dir) / "videos")
    rows = []
    with torch.inference_mode():
        for path in videos:
            frames = _stage3_frames(path)
            count = len(frames)
            accel_predictions, steer_predictions = [], []
            for start in range(0, count, 8):
                center = np.arange(count)[start : start + 8]
                indices = np.asarray([build_centered_clip_indices(int(c), count, 16) for c in center], dtype=np.int64)
                clips = frames[torch.from_numpy(indices)].permute(0, 2, 1, 3, 4).float() / 255.0
                clips = (clips - S3_MEAN[None, :, None, :, :]) / S3_STD[None, :, None, :, :]
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    accel_logits, steer_logits = model(clips.to(device, non_blocking=True))
                accel_predictions.extend(accel_logits.argmax(1).cpu().tolist())
                steer_predictions.extend(steer_logits.argmax(1).cpu().tolist())
            for sample_index, (accel, steer) in enumerate(zip(accel_predictions, steer_predictions)):
                rows.append({"ID": path.stem, "sample_index": sample_index, "accel_label": ACCEL[accel], "steer_label": STEER[steer]})
    del model
    torch.cuda.empty_cache()
    return pd.DataFrame(rows, columns=["ID", "sample_index", "accel_label", "steer_label"])