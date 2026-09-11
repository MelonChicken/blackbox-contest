from __future__ import annotations

from torch import nn
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models.video import MViT_V2_S_Weights, mvit_v2_s

from src.config import S3_MEAN, S3_STD, STAGE3_TARTANVO_FEATURE, STAGE3_TARTANVO_HEIGHT, STAGE3_TARTANVO_WIDTH, TARTANVO_CHECKPOINT
try:
    from src.config import STAGE3_TARTANVO_FEATURE_NORM
except ImportError:
    from src.config import STAGE3_TARTANVO_POSE_NORM as STAGE3_TARTANVO_FEATURE_NORM
try:
    from src.config import STAGE3_TARTANVO_MODE, STAGE3_TARTANVO_UNFREEZE
except ImportError:
    STAGE3_TARTANVO_MODE, STAGE3_TARTANVO_UNFREEZE = "cached", "none"
from src.models.tartanvo import TartanVOEncoder


class Stage3MViT(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.backbone = mvit_v2_s(weights=MViT_V2_S_Weights.KINETICS400_V1 if pretrained else None)
        dim = self.backbone.head[1].in_features
        self.backbone.head = nn.Identity()
        self.accel = nn.Linear(dim, 4)
        self.steer = nn.Linear(dim, 3)

    def forward(self, x):
        z = self.backbone(x)
        return self.accel(z), self.steer(z)


class Stage3ResNetGRU(nn.Module):
    def __init__(self, pretrained: bool = False, hidden_size: int = 256, num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.gru = nn.GRU(feature_dim, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.accel = nn.Linear(hidden_size, 4)
        self.steer = nn.Linear(hidden_size, 3)

    def forward(self, x):
        b, c, t, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        features = self.backbone(x).reshape(b, t, -1)
        temporal, _ = self.gru(features)
        z = self.dropout(temporal.mean(dim=1))
        return self.accel(z), self.steer(z)


class Stage3TartanVOGRU(nn.Module):
    def __init__(
        self,
        load_pretrained: bool = True,
        checkpoint=TARTANVO_CHECKPOINT,
        feature: str = STAGE3_TARTANVO_FEATURE,
        hidden_size: int = 256,
        num_layers: int = 1,
        dropout: float = 0.2,
        feature_norm: str = STAGE3_TARTANVO_FEATURE_NORM,
        pose_norm: str | None = None,
        height: int = STAGE3_TARTANVO_HEIGHT,
        width: int = STAGE3_TARTANVO_WIDTH,
    ):
        super().__init__()
        if feature not in {"pose", "latent"}:
            raise ValueError(f"Unknown TartanVO feature: {feature}")
        self.feature = feature
        self.feature_dim = 6 if feature == "pose" else TartanVOEncoder.latent_dim
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.dropout_value = float(dropout)
        self.tartanvo_mode = STAGE3_TARTANVO_MODE
        self.tartanvo_unfreeze = STAGE3_TARTANVO_UNFREEZE
        feature_norm = pose_norm if pose_norm is not None else feature_norm
        if feature_norm not in {"none", "layernorm"}:
            raise ValueError(f"Unknown TartanVO feature norm: {feature_norm}")
        self.feature_norm_name = feature_norm
        self.feature_norm = nn.Identity() if feature_norm == "none" else nn.LayerNorm(self.feature_dim)
        self.tartanvo = TartanVOEncoder(
            checkpoint=checkpoint,
            load_pretrained=load_pretrained,
            height=height,
            width=width,
            trainable=self.tartanvo_mode == "finetune",
        )
        self.gru = nn.GRU(self.feature_dim, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.accel = nn.Linear(hidden_size, 4)
        self.steer = nn.Linear(hidden_size, 3)
        self.register_buffer("s3_mean", S3_MEAN[:, None, :, :].clone())
        self.register_buffer("s3_std", S3_STD[:, None, :, :].clone())
        self.apply_tartanvo_unfreeze(self.tartanvo_unfreeze)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.tartanvo_mode == "cached":
            self.tartanvo.eval()
        return self

    def apply_tartanvo_unfreeze(self, mode: str) -> None:
        if mode not in {"none", "last_block", "last_2_blocks"}:
            raise ValueError(f"unknown STAGE3_TARTANVO_UNFREEZE: {mode}")
        for p in self.tartanvo.parameters():
            p.requires_grad = False
        if self.tartanvo_mode != "finetune" or mode == "none":
            self.tartanvo.trainable = False
            return
        self.tartanvo.trainable = True
        modules = [self.tartanvo.vonet.flowPoseNet.layer5]
        if mode == "last_2_blocks":
            modules.insert(0, self.tartanvo.vonet.flowPoseNet.layer4)
        for module in modules:
            for p in module.parameters():
                p.requires_grad = True

    def feature_sequence(self, x, intrinsics=None):
        frames = (x * self.s3_std + self.s3_mean).clamp(0.0, 1.0)
        return self.tartanvo(frames, feature=self.feature, intrinsics=intrinsics)

    def pose_sequence(self, x, intrinsics=None):
        frames = (x * self.s3_std + self.s3_mean).clamp(0.0, 1.0)
        return self.tartanvo(frames, feature="pose", intrinsics=intrinsics)

    def forward_feature(self, features):
        temporal, _ = self.gru(self.feature_norm(features))
        z = self.dropout(temporal.mean(dim=1))
        return self.accel(z), self.steer(z)

    def forward_pose(self, poses):
        return self.forward_feature(poses)

    def forward(self, x, intrinsics=None):
        return self.forward_feature(self.feature_sequence(x, intrinsics=intrinsics))

    def model_config(self) -> dict:
        return {
            "tartanvo_feature": self.feature,
            "tartanvo_feature_dim": self.feature_dim,
            "gru_hidden_size": self.hidden_size,
            "gru_num_layers": self.num_layers,
            "dropout": self.dropout_value,
            "tartanvo_feature_norm": self.feature_norm_name,
            "tartanvo_pose_norm": self.feature_norm_name,
            "tartanvo_height": self.tartanvo.height,
            "tartanvo_width": self.tartanvo.width,
            "tartanvo_mode": self.tartanvo_mode,
            "tartanvo_unfreeze": self.tartanvo_unfreeze,
        }