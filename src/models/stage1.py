# 1. Configuration

# 1.1. 관련 패키지 임포트
from torch import nn
from torchvision.models.video import mvit_v2_s


class Stage1MViT(nn.Module):
    """
    영상 전체를 하나의 2-class 문제로 분류하는 모델 클래스
    Stage 1의 ORIGINAL / RERECORDED 이진 분류 모델
    """

    def __init__(self):
        # nn.Module의 초기화를 수행
        super().__init__()
        # Torchvision의 MViTv2-Small 모델을 생성
        self.net = mvit_v2_s(weights=None)
        # 기본 MViT의 classification head 변경 (우리는 이진 분류를 원하니 out_features를 2로 조정한다)
        self.net.head[1] = nn.Linear(self.net.head[1].in_features, 2)

    # 입력 x를 MViT에 그대로 넣으면 두 클래스에 대한 logit이 나온다. 이후에 추론에서 softmax를 통해 probability롷 전환한다.
    def forward(self, x): return self.net(x)
