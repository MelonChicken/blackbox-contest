# 1. Configuration
# 1.1. 관련 패키지 임포트
from pathlib import Path
import cv2, numpy as np, torch
from torch import Tensor
import random

# 함수 리턴 타입 표시를 위한 임포트
from typing import List

# 2. Implementation
# 2.1. Helper Functions

def video_frames(path) -> List:
    """
    영상 파일 하나를 열어서 모든 프레임을 메모리에 읽어오는 함수

    :param path: `Path`, 메모리에 읽어오려는 영상의 주소
    :return: `List<np.ndarray>`, 읽어들인 모든 프레임을 담은 리스트. 각 array는 [H, W, 3]의 크기를 갖는다.
    """
    # OpenCV로 영상 파일을 연다.
    cap = cv2.VideoCapture(str(path))

    # 리턴할 프레임 목록을 저장하는 리스트

    out = []

    while True:
        # 프레임 하나를 읽는다.
        ok, bgr = cap.read()

        # 정상적으로 읽지 못한 케이스
        if not ok: break

        # BGR 형태의 프레임을 RGB 형태로 바꾸고 리스트에 추가한다.
        out.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    # 읽기가 끝
    cap.release()

    # 만약 out에 하나도 없다면
    if not out: raise ValueError(f'cannot decode: {path}')


    return out


def _crop_tensor(rgb: np.ndarray, size=224) -> torch.Tensor:
    """
    프레임 하나를 모델 입력용 224×224 tensor로 바꾸는 함수

    :param rgb: `np.ndarray`, `[H, W, 3]`의 크기를 가진 입력 프레임
    :param size: `int`, `default=224`, crop할 사이즈. aspect ratio를 유지하며 해당 사이즈로 size by size로 crop된다.

    :return:
    """
    # 원본 크기 확인
    h, w = rgb.shape[:2]

    # 짧은 변을 224로 resize
    scale = size / min(h, w)

    # 실제로 aspect ratio를 유지한채로 짧은 변이 224가 되게끔 하는 nh, nw 계산
    nh, nw = max(size, round(h * scale)), max(size, round(w * scale))

    # 영상 비율을 유지한 상태에서 짧은 변이 224가 되게끔 resize
    rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)

    # center crop
    y, x = (nh - size) // 2, (nw - size) // 2

    # NumPy array를 PyTorch tensor로 변환
    # .permute(2,0,1)를 통해 tensor에서 요구하는 [C, H, W]를 맞춘다.
    # /225를 통해 0~255 → 0~1로 min-max scaling을 진행한다. (숫자의 단위를 통일)
    return torch.from_numpy(rgb[y:y + size, x:x + size].copy()).permute(2, 0, 1).float() / 255


def clip(path: Path, n=16, center=None) -> tuple[Tensor, int]:
    """
    영상 하나에서 n개 프레임을 선택해서 MViT 입력용 video tensor를 만드는 함수
    총체적으로 다른 함수들을 호출하고 encoder에 입력될 값들을 정리한다.

    :param path: `Path`, encoder 용 입력값으로 처리할 영상의 주소
    :param n: `int`, `default=16`, 선택할 프레임의 수
    :param center: `int`, `default=None`, 영상에서 집중적으로 다룰 구간. center를 중심으로 n개의 프레임을 선택하게 된다. 선택하지 않는 경우
                    영상 전체에서 띄엄띄엄 n개의 frame을 선택하게 된다.
    :return:
    """
    # 영상으로부터 모든 프레임을 읽는다.
    frames = video_frames(path)

    # 전체 개수 확인
    total = len(frames)

    # 만약 특정 시점에서 집중적으로 뽑고 싶지 않다면
    if center is None:
        # 영상 전체 구간에서 균등하게 n개의 frame의 idx를 뽑기
        idx = np.linspace(0, total - 1, n).round().astype(int)
    # 특정 시점에서 뽑고 싶다면
    else:
        # 해당 구간에서 idx n개를 뽑는다
        idx = np.clip(center - n // 2 + np.arange(n), 0, total - 1)

    # 실제로 앞에서 선택한 idx의 frame을 torch.Tensor로 변환한다.
    # 그 뒤에 x라는 변수에 dimension 1 위치에 쌓는다. [C,T,H,W] 꼴이 된다.
    x = torch.stack([_crop_tensor(frames[int(i)]) for i in idx], 1)

    return x, total

def set_seed(seed=42):
    """
    디바이스의 랜덤 시드 설정
    :param seed:
    :return:
    """
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)