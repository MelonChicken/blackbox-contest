# 1. Configuration

# 1.1. 관련 패키지 임포트
import pandas as pd, torch
from torch import nn

# 다른 파일에서 모델, 함수, configuration 로드
from src.models import Stage1MViT
from src.utils import clip, set_seed
from src.config import (
    DATA,
    MODEL,
    DEVICE,
    EPOCHS,
    S1_MEAN,
    S1_STD, SEED,
)

# 랜덤 시드 고정
set_seed(SEED)

# stage 1을 위한 data경로 확인
STAGE1_DATA = DATA / 'stage1' / 'dlc2021'

def fit_stage1():
    """
    Stage 1 모델을 학습하는 함수
    :return:
    """
    # 학습된 모델을 저장할 폴더 생성
    out=MODEL/'stage1'
    out.mkdir(parents=True,exist_ok=True)

    # Stage 1 라벨 CSV을 읽는다.
    df=pd.read_csv(STAGE1_DATA /'labels.csv')

    # Stage 1 MViT 모델을 만들고 활용 가능한 디바이스로 보낸다.
    model=Stage1MViT().to(DEVICE)
    
    # AdamW optimizer를 사용한다.
    opt=torch.optim.AdamW(model.parameters(),1e-4)
    
    # 설정한 epochs만큼 데이터셋에 대한 학습 진행
    for _ in range(EPOCHS):
        # 모델을 training mode로 전환
        model.train()

        # 그 다음이 실제 데이터 반복하는데 순서를 섞는다.
        # itertuples()는 dataframe의 각 행을 tuple-like 객체로 반환
        for r in df.sample(frac=1,random_state=SEED).itertuples():
            # 영상 하나를 불러와서 16-frame clip 추출
            x,_=clip(DATA/'stage1'/r.path,16)
            # Stage 1 정규화 (사전에 정해놓은 값 활용)
            x=(x-S1_MEAN)/S1_STD

            # label을 숫자로 변환
            y=torch.tensor([0 if r.label=='ORIGINAL' else 1],device=DEVICE)

            # forward pass
            # x[None]으로 앞에 차원을 하나 추가 (Batch dimension을 위함)
            loss=nn.functional.cross_entropy(model(x[None].to(DEVICE)),y)
            # 이전 iteration에서 남아 있는 gradient를 초기화
            opt.zero_grad()
            # 역전파를 수행
            loss.backward()
            # 계산된 gradient를 이용해 실제 model weight를 업데이트
            opt.step()

    # 실제 inference.py는 래퍼가 아닌 mvit_v2_s 본체에 직접 로드한다.
    torch.save({'model':model.net.state_dict(),'size':224,'frames':16},out/'best.pt')