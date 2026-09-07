# 1. Configuration

# 1.1. 관련 패키지 임포트
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights

from src.config import S3_MEAN, S3_STD, STAGE3_TARTANVO_FEATURE, STAGE3_TARTANVO_HEIGHT, STAGE3_TARTANVO_POSE_NORM, STAGE3_TARTANVO_WIDTH, TARTANVO_CHECKPOINT
from src.models.tartanvo import TartanVOEncoder


class Stage3MViT(nn.Module):
    """
    Stage 3: 차량 거동 특성 분석 모델
    """

    def __init__(self, pretrained: bool = True):
        # nn.Module의 초기화를 수행
        super().__init__()

        # Torchvision의 MViTv2-Small 모델을 생성
        self.backbone = mvit_v2_s(weights=MViT_V2_S_Weights.KINETICS400_V1 if pretrained else None)

        # MViT가 최종적으로 생성하는 representation의 dimension 확인
        dim = self.backbone.head[1].in_features

        # classification head 제거
        # 이제 class logits이 아니라 video feature vector가 나온다.
        self.backbone.head = nn.Identity()
        # acceleration 범주 4가지의 classifier 설정
        # ACCELERATING, DECELERATING, CONSTANT, STOPPED
        self.accel = nn.Linear(dim, 4)
        # 조향 범주 3가지 classifier 설정
        # LEFT, STRAIGHT, RIGHT
        self.steer = nn.Linear(dim, 3)

    def forward(self, x):
        # 영상 하나를 하나의 feature vector로 압축
        z = self.backbone(x)
        # 동일한 z를 두 classifier가 공유
        # accel_logits, steer_logits가 나온다.
        return self.accel(z), self.steer(z)

class Stage3ResNetGRU(nn.Module):
    def __init__(
        self,
        pretrained: bool = False,
        hidden_size: int = 256,
        num_layers: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
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
        pose_norm: str = STAGE3_TARTANVO_POSE_NORM,
        height: int = STAGE3_TARTANVO_HEIGHT,
        width: int = STAGE3_TARTANVO_WIDTH,
    ):
        super().__init__()
        if feature != "pose":
            raise ValueError(f"Stage3TartanVOGRU only supports pose feature, got: {feature}")
        self.feature = feature
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.dropout_value = float(dropout)
        if pose_norm not in {"none", "layernorm"}:
            raise ValueError(f"Unknown TartanVO pose norm: {pose_norm}")
        self.pose_norm_name = pose_norm
        self.pose_norm = nn.Identity() if pose_norm == "none" else nn.LayerNorm(6)
        self.tartanvo = TartanVOEncoder(checkpoint=checkpoint, load_pretrained=load_pretrained, height=height, width=width)
        self.gru = nn.GRU(input_size=6, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.accel = nn.Linear(hidden_size, 4)
        self.steer = nn.Linear(hidden_size, 3)
        self.register_buffer("s3_mean", S3_MEAN[:, None, :, :].clone())
        self.register_buffer("s3_std", S3_STD[:, None, :, :].clone())

    def train(self, mode: bool = True):
        super().train(mode)
        self.tartanvo.eval()
        return self

    def pose_sequence(self, x):
        frames = (x * self.s3_std + self.s3_mean).clamp(0.0, 1.0)
        return self.tartanvo(frames)

    def forward_pose(self, poses):
        temporal, _ = self.gru(self.pose_norm(poses))
        z = self.dropout(temporal.mean(dim=1))
        return self.accel(z), self.steer(z)

    def forward(self, x):
        return self.forward_pose(self.pose_sequence(x))

    def model_config(self) -> dict:
        return {
            "tartanvo_feature": self.feature,
            "gru_hidden_size": self.hidden_size,
            "gru_num_layers": self.num_layers,
            "dropout": self.dropout_value,
            "tartanvo_pose_norm": self.pose_norm_name,
            "tartanvo_height": self.tartanvo.height,
            "tartanvo_width": self.tartanvo.width,
        }