import re
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18

from src.models.stage2 import Stage2Temporal


def _device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("DACON inference requires a CUDA GPU.")
    return torch.device("cuda")


class Stage2Frames(Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB"))


def _frame_number(path: Path):
    match = re.search(r"(\d+)$", path.stem)
    return int(match.group(1)) if match else 0


def predict_stage2(data_dir, model_dir):
    device = _device()
    model_dir = Path(model_dir)
    transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
    backbone = resnet18(weights=None)
    backbone.load_state_dict(torch.load(model_dir / "resnet18-f37072fd.pth", map_location="cpu", weights_only=True))
    backbone.fc = nn.Identity()
    backbone.to(device).eval()
    temporal = Stage2Temporal()
    temporal.load_state_dict(torch.load(model_dir / "best.pt", map_location="cpu", weights_only=False)["model"])
    temporal.to(device).eval()

    image_root = Path(data_dir) / "images"
    folders = sorted(p for p in image_root.iterdir() if p.is_dir())
    rows = []
    with torch.inference_mode():
        for folder in folders:
            paths = sorted(
                (p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}),
                key=_frame_number,
            )
            if not paths:
                continue
            loader = DataLoader(Stage2Frames(paths, transform), batch_size=256, num_workers=6, pin_memory=True)
            features = []
            for images in loader:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    features.append(backbone(images.to(device, non_blocking=True)).float().cpu())
            sequence = torch.cat(features)[None].to(device)
            collision_idx, entry_idx, scene = temporal(sequence)
            frame_numbers = [_frame_number(path) for path in paths]
            rows.append(
                {
                    "ID": folder.name,
                    "collision_frame": frame_numbers[int(collision_idx)],
                    "entry_frame": frame_numbers[int(entry_idx)],
                    "evasion_space": int(scene[:, :2].argmax(1)),
                    "entry_side": "RIGHT" if int(scene[:, 2:].argmax(1)) else "LEFT",
                }
            )
    del backbone, temporal
    torch.cuda.empty_cache()
    return pd.DataFrame(
        rows, columns=["ID", "collision_frame", "entry_frame", "evasion_space", "entry_side"]
    )
