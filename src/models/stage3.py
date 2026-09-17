from __future__ import annotations

from torch import nn
from torchvision.models.video import MViT_V2_S_Weights, mvit_v2_s

from src.config import STAGE3_MVIT_PREPROCESS, STAGE3_MVIT_PRETRAINED_WEIGHTS, STAGE3_NUM_FRAMES


class Stage3MViT(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = MViT_V2_S_Weights.KINETICS400_V1 if pretrained else None
        self.backbone = mvit_v2_s(weights=weights)
        dim = self.backbone.head[1].in_features
        self.backbone.head = nn.Identity()
        self.accel = nn.Linear(dim, 4)
        self.steer = nn.Linear(dim, 3)

    def forward(self, x):
        z = self.backbone(x)
        return self.accel(z), self.steer(z)

    def model_config(self) -> dict:
        return {
            "num_frames": STAGE3_NUM_FRAMES,
            "image_size": 224,
            "feature_dim": self.accel.in_features,
            "preprocess": STAGE3_MVIT_PREPROCESS,
            "pretrained_weights": STAGE3_MVIT_PRETRAINED_WEIGHTS,
        }