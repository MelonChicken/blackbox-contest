from pathlib import Path
import os
import torch

# 사전설정

## 경로 설정
ROOT = Path.cwd()
DATA = ROOT / 'data'
MODEL = ROOT / 'model'

## 디바이스 설정
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 10

## 입력 크기 설정
SIZE = 224

## 정규화 상수 준비
### 각 Stage별 영상 정규화용 평균과 표준편차
S1_MEAN = torch.tensor([0.45, 0.45, 0.45])[:, None, None, None]
S1_STD = torch.tensor([0.225, 0.225, 0.225])[:, None, None, None]
S3_MEAN = torch.tensor([0.45, 0.45, 0.45])[:, None, None]
S3_STD = torch.tensor([0.225, 0.225, 0.225])[:, None, None]

## 난수 고정
SEED = 42
