from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from .network import VONet


class TartanVOEncoder(nn.Module):
    pose_std = torch.tensor([0.13, 0.13, 0.13, 0.013, 0.013, 0.013], dtype=torch.float32)

    def __init__(self, checkpoint: str | Path | None = None, load_pretrained: bool = True, height: int = 448, width: int = 640):
        super().__init__()
        self.vonet = VONet()
        self.height = int(height)
        self.width = int(width)
        self.pretrained_loaded = False
        if load_pretrained:
            if checkpoint is None or not Path(checkpoint).is_file():
                raise FileNotFoundError(f"TartanVO checkpoint not found: {checkpoint}")
            self.load_pretrained(checkpoint)
        for p in self.vonet.parameters():
            p.requires_grad = False
        self.vonet.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.vonet.eval()
        return self

    def load_pretrained(self, checkpoint: str | Path) -> None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = state.get("state_dict", state) if isinstance(state, dict) else state
        model_state = self.vonet.state_dict()
        cleaned = {}
        for key, value in state.items():
            name = key[7:] if key.startswith("module.") else key
            if name in model_state and model_state[name].shape == value.shape:
                cleaned[name] = value
        if not cleaned:
            raise RuntimeError(f"No compatible TartanVO weights found in {checkpoint}")
        missing, unexpected = self.vonet.load_state_dict(cleaned, strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected TartanVO checkpoint keys: {unexpected[:5]}")
        self.pretrained_loaded = True
        self.loaded_key_count = len(cleaned)
        self.missing_key_count = len(missing)

    def _intrinsic(self, batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        h, w = self.height, self.width
        fx, fy, cx, cy = float(w), float(w), w / 2.0, h / 2.0
        ys, xs = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
        intrinsic = torch.stack(((xs.float() - cx + 0.5) / fx, (ys.float() - cy + 0.5) / fy), dim=0)
        intrinsic = F.interpolate(intrinsic.unsqueeze(0), size=(h // 4, w // 4), mode="bilinear", align_corners=False).to(dtype)
        return intrinsic.repeat(batch, 1, 1, 1)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        b, c, t, _, _ = frames.shape
        if c != 3 or t < 2:
            raise ValueError(f"expected [B,3,T,H,W] with T>=2, got {frames.shape}")
        # ponytail: Stage3 data is already 224 center-cropped; 448x640 resize only preserves FC compatibility, not lost FOV.
        resized = F.interpolate(frames.permute(0, 2, 1, 3, 4).reshape(b * t, c, frames.shape[-2], frames.shape[-1]), size=(self.height, self.width), mode="bilinear", align_corners=False)
        resized = resized.reshape(b, t, c, self.height, self.width)
        img1 = resized[:, :-1].reshape(b * (t - 1), c, self.height, self.width).contiguous()
        img2 = resized[:, 1:].reshape(b * (t - 1), c, self.height, self.width).contiguous()
        intrinsic = self._intrinsic(img1.shape[0], img1.device, img1.dtype)
        with torch.no_grad():
            _, pose = self.vonet([img1, img2, intrinsic])
            pose = pose * self.pose_std.to(device=pose.device, dtype=pose.dtype)
        return pose.reshape(b, t - 1, 6)