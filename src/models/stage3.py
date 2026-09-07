# 1. Configuration

# 1.1. 관련 패키지 임포트
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights


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
