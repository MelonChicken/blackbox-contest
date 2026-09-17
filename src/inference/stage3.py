from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.config import S3_MEAN, S3_STD
from src.datasets.stage3_sampling import build_centered_clip_indices
from src.inference.stage1 import _video_paths
from src.models.stage3 import Stage3MViT, Stage3ResNetGRU, Stage3TartanVOGRU
from src.models.stage3_vjepa2 import Stage3VJEPA2Frozen


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


def _tartanvo_config(checkpoint: dict) -> dict:
    config = dict(checkpoint.get("model_config") or {})
    state = checkpoint.get("model", {})
    input_size = state.get("gru.weight_ih_l0")
    if input_size is not None:
        config.setdefault("tartanvo_feature", "pose" if input_size.shape[1] == 6 else "latent")
    config.setdefault("tartanvo_feature_norm", config.get("tartanvo_pose_norm", "none"))
    return config


def _stage3_state_dict(checkpoint: dict) -> dict:
    state = dict(checkpoint["model"])
    if "pose_norm.weight" in state and "feature_norm.weight" not in state:
        state["feature_norm.weight"] = state.pop("pose_norm.weight")
        state["feature_norm.bias"] = state.pop("pose_norm.bias")
    return state


def _stage3_checkpoint_path(model_dir) -> Path:
    root = Path(model_dir)
    best = root / "best.pt"
    tartan = root / "tartanvo_best.pt"
    return best if best.is_file() else tartan


def _stage3_model(arch: str, checkpoint: dict | None = None):
    if arch in {"mvit_v2_s", "mvit"}:
        return Stage3MViT(pretrained=False)
    if arch == "resnet18_gru":
        return Stage3ResNetGRU(pretrained=False)
    if arch == "vjepa2_1_vitb_frozen":
        config = dict((checkpoint or {}).get("model_config") or {})
        required = ("backbone_repo", "backbone_name", "probe_type", "input_size", "num_frames")
        missing = [key for key in required if key not in config]
        if missing:
            raise KeyError(f"V-JEPA Stage3 checkpoint missing model_config keys: {missing}")
        return Stage3VJEPA2Frozen(
            repo_dir=config["backbone_repo"],
            checkpoint=None,
            backbone_name=config["backbone_name"],
            probe_type=config["probe_type"],
            input_size=int(config["input_size"]),
            num_frames=int(config["num_frames"]),
            freeze_backbone=bool(config.get("backbone_frozen", True)),
            probe_dim=int(config.get("probe_dim", 384)),
            probe_heads=int(config.get("probe_heads", 6)),
            dropout=float(config.get("dropout", 0.2)),
        )
    if arch == "tartanvo_gru":
        config = _tartanvo_config(checkpoint or {})
        return Stage3TartanVOGRU(
            load_pretrained=False,
            feature=config.get("tartanvo_feature", "pose"),
            feature_norm=config.get("tartanvo_feature_norm", "none"),
            hidden_size=int(config.get("gru_hidden_size", 256)),
            num_layers=int(config.get("gru_num_layers", 1)),
            dropout=float(config.get("dropout", 0.2)),
            height=int(config.get("tartanvo_height", 448)),
            width=int(config.get("tartanvo_width", 640)),
        )
    raise ValueError(f"Unknown Stage3 arch: {arch}")


def predict_stage3(data_dir, model_dir):
    device = _device()
    checkpoint = torch.load(_stage3_checkpoint_path(model_dir), map_location="cpu", weights_only=True)
    arch = checkpoint.get("arch", "mvit_v2_s")
    model = _stage3_model(arch, checkpoint)
    model.load_state_dict(_stage3_state_dict(checkpoint), strict=True)
    model.to(device).eval()
    videos = _video_paths(Path(data_dir) / "videos")
    rows = []
    with torch.inference_mode():
        for path in videos:
            frames = _stage3_frames(path)
            count = len(frames)
            centers = np.arange(count)
            accel_predictions, steer_predictions = [], []
            window_batch = 1 if arch == "tartanvo_gru" else 8
            for start in range(0, count, window_batch):
                center = centers[start : start + window_batch]
                indices = np.asarray([build_centered_clip_indices(int(c), count, 16) for c in center], dtype=np.int64)
                clips = frames[torch.from_numpy(indices)].permute(0, 2, 1, 3, 4).float() / 255.0
                clips = (clips - S3_MEAN[None, :, None, :, :]) / S3_STD[None, :, None, :, :]
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    accel_logits, steer_logits = model(clips.to(device, non_blocking=True))
                accel_predictions.extend(accel_logits.argmax(1).cpu().tolist())
                steer_predictions.extend(steer_logits.argmax(1).cpu().tolist())
            for sample_index, (accel, steer) in enumerate(zip(accel_predictions, steer_predictions)):
                rows.append(
                    {
                        "ID": path.stem,
                        "sample_index": sample_index,
                        "accel_label": ACCEL[accel],
                        "steer_label": STEER[steer],
                    }
                )
    del model
    torch.cuda.empty_cache()
    return pd.DataFrame(rows, columns=["ID", "sample_index", "accel_label", "steer_label"])



