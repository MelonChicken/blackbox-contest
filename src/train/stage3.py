# 1. Configuration

# 1.1. 관련 패키지 임포트
import pandas as pd, torch
from torch import nn
from src.config import (
    DATA,
    MODEL,
    DEVICE,
    EPOCHS,
    SEED, S3_MEAN, S3_STD,
)
from src.models import Stage3MViT
from src.utils import set_seed, clip

# 랜덤 시드 고정
set_seed(SEED)


def fit_stage3():
    out=MODEL/'stage3'; out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(DATA/'stage3/labels.csv')
    amap={'ACCELERATING':0,'DECELERATING':1,'CONSTANT':2,'STOPPED':3}
    smap={'LEFT':0,'STRAIGHT':1,'RIGHT':2}
    model=Stage3MViT().to(DEVICE); opt=torch.optim.AdamW(model.parameters(),1e-4)
    for _ in range(EPOCHS):
        model.train()
        for r in df.itertuples():
            x,_=clip(DATA/'stage3/videos'/f'{r.ID}.mp4',16,int(r.frame_index))
            x=(x-S3_MEAN[:,None,:,:])/S3_STD[:,None,:,:]
            a,s=model(x[None].to(DEVICE))
            loss=nn.functional.cross_entropy(a,torch.tensor([amap[r.accel_label]],device=DEVICE))
            loss+=nn.functional.cross_entropy(s,torch.tensor([smap[r.steer_label]],device=DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()
    torch.save({'model':model.state_dict()},out/'best.pt')