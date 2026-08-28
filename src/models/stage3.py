# 1. Configuration

# 1.1. 관련 패키지 임포트
from torch import nn
from torchvision.models.video import mvit_v2_s


class Stage3MViT(nn.Module):
    """
    Stage 3: 차량 거동 특성 분석 모델
    """

    def __init__(self):
        # nn.Module의 초기화를 수행
        super().__init__()

        # Torchvision의 MViTv2-Small 모델을 생성
        self.backbone = mvit_v2_s(weights=None)

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
