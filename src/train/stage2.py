# 1. Configuration

# 1.1. 관련 패키지 임포트
import pandas as pd, torch
from PIL import Image
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights
from src.config import (
    DATA,
    MODEL,
    DEVICE,
    EPOCHS,
    SEED,
)
# 1.2. 관련 파일에서 필요한 함수 로드
from src.models import Stage2Temporal
from src.utils import set_seed, video_frames

# 랜덤 시드 고정
set_seed(SEED)

def _resnet_backbone():
    """
    Stage 2에서 각 프레임의 512차원 feature를 추출할 ResNet18 준비하는 함수
    :return: 
    """
    # pre-train된 weight 활용 가능한지 확인
    try: model=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    except Exception:
        print('경고: ImageNet 가중치를 받지 못해 weights=None으로 진행합니다.')
        # 안되면 그냥 없이 진행
        model=resnet18(weights=None)
    return model

def fit_stage2():
    """
    Stage 2 모델을 학습하는 함수
    :return:
    """
    # 학습된 모델을 저장할 폴더 생성
    out=MODEL/'stage2'
    out.mkdir(parents=True,exist_ok=True)

    # Stage 2 라벨 CSV을 읽는다.
    df=pd.read_csv(DATA/'stage2/labels.csv')
    # ResNet18 backbone 생성
    backbone=_resnet_backbone()

    # ResNet weight 저장
    torch.save(backbone.state_dict(),out/'resnet18-f37072fd.pth')

    # ResNet classifier 제거
    backbone.fc=nn.Identity()
    # GPU로 이동 및 inference mode 설정
    backbone.to(DEVICE).eval()
    # ImageNet preprocessing 준비
    transform=ResNet18_Weights.IMAGENET1K_V1.transforms()
    # feature sequence를 저장할 리스트
    sequences=[]
    with torch.inference_mode():
        for r in df.itertuples():
            frames=video_frames(DATA/'stage2'/r.path); batches=[]
            for start in range(0,len(frames),64):
                x=torch.stack([transform(Image.fromarray(a)) for a in frames[start:start+64]]).to(DEVICE)
                batches.append(backbone(x).float().cpu())
            sequences.append((torch.cat(batches),min(int(r.t_collision),len(frames)-1)))
    temporal=Stage2Temporal().to(DEVICE); opt=torch.optim.AdamW(temporal.parameters(),2e-4)
    for _ in range(max(1,EPOCHS)):
        temporal.train()
        for seq,target in sequences:
            collision,_,_=temporal.logits(seq[None].to(DEVICE))
            loss=nn.functional.cross_entropy(collision,torch.tensor([target],device=DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()
    # 공개 CCD 5건에는 충돌 구간만 공식 주석이 있어 나머지 헤드는 구조 확인용이다.
    torch.save({'model':temporal.state_dict()},out/'best.pt')