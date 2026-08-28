# 1. Configuration

# 1.1. 관련 패키지 임포트
import torch
from torch import nn



class Stage2Temporal(nn.Module):
    """
    입력은 이미 다른 모델에서 추출한 frame-wise feature sequence가 된다.
    입력 feature dimension은 512이다.
    Stage 2: 사고 주요시점·상황 분석 모델
    """

    def __init__(self):
        # nn.Module의 초기화를 수행
        super().__init__()

        # 각 frame feature가 512차원 input_size = 512
        # GRU의 hidden representation 크기는 192 hidden_size = 192
        # GRU layer를 두 층 num_layers = 2
        # 입력 shape이 [B, T, feature] 가 되도록 한다. batch_first=True
        # 양방향 GRU로 설정. 이를 통해 representation의 수가 192x2=384가 된다. bidirectional=True
        # GRU layer 사이에 dropout을 적용 dropout=0.15
        self.r = nn.GRU(512, 192, 2, batch_first=True, bidirectional=True, dropout=0.15)

        # collision head 확인. 각 timestep의 384차원 feature를 scalar 하나로 줄인다.
        # 즉 각 frame마다 "이 frame이 collision frame일 가능성이 얼마나 높은가?"에 대응되는 logit 하나를 생성
        self.tc = nn.Linear(384, 1)
        # entry head 확인. 각 frame마다 하나의 score를 만들지만 이번에는 entry frame을 찾기 위한 것
        self.te = nn.Linear(384, 1)
        # scene classifier. 각 클래스별로 입력으로 받는 feature에 대하여 scene logit을 계산한다.
        self.scene = nn.Sequential(nn.Linear(768, 192), nn.ReLU(), nn.Dropout(0.2), nn.Linear(192, 4))

    def logits(self, x):
        # 입력에 대해 GRU를 통과시킨다.
        # 입력: x = [B,T,512]
        # 출력: h = [B,T,384]
        h, _ = self.r(x)
        # collision score : tc를 통해 확인 [B, T]
        # entry score : te를 통해 확인 [B, T]
        return self.tc(h).squeeze(-1), self.te(h).squeeze(-1), h

    def forward(self, x):
        # logit을 계산한다.
        collision, entry, h = self.logits(x)
        # collision frame과 entry frame 선택
        ci, ei = collision.argmax(1), entry.argmax(1)
        # 각 batch sample에서 선택한 frame의 hidden state를 가져온다.
        b = torch.arange(len(h), device=h.device)
        # 두 feature를 concatenate 한 뒤 self.scene()에 넣는다.
        # predicted collision index
        # predicted entry index
        # scene classification logits
        # 위의 세가지를 최종적으로 제출한다.
        return ci, ei, self.scene(torch.cat([h[b, ci], h[b, ei]], 1))

