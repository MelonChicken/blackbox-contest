# 1. Configuration

# 1.1. 관련 패키지 임포트
import torch

from torch import nn
from torch.utils.data import DataLoader

# 대회 Stage 1 평가 지표인 Macro-F1을 계산하기 위한 함수
from sklearn.metrics import f1_score
from collections import Counter
from sklearn.metrics import confusion_matrix

# AI-Hub 블랙박스 영상을 Stage 1 학습 데이터로 제공하는 Dataset
from src.datasets.aihubDataset import AIHubStage1Dataset

# Stage 1 모델 로드
from src.models import Stage1MViT

# 랜덤 시드 고정을 위한 함수
from src.utils import set_seed

# 공통 configuration 로드
from src.config import (
    DATA,
    MODEL,
    DEVICE,
    EPOCHS,
    S1_MEAN,
    S1_STD,
    SEED,
)


# 랜덤 시드를 고정하여
# 동일한 조건에서 실험을 재현할 수 있도록 한다.
set_seed(SEED)


# --------------------------------------------------
# Stage 1 AI-Hub 데이터 경로
# --------------------------------------------------

STAGE1_DATA = (
    DATA
    / "stage1"
    / "aihub597"
)

TRAIN_MANIFEST = (
    STAGE1_DATA
    / "manifest"
    / "train.csv"
)

VAL_MANIFEST = (
    STAGE1_DATA
    / "manifest"
    / "val.csv"
)

def fit_stage1():
    """
    AI-Hub 교통사고 블랙박스 영상을 이용하여

    ORIGINAL:
        AI-Hub 원본 영상

    RERECORDED:
        원본 영상에 synthetic recapture augmentation 적용

    두 클래스를 분류하는 Stage 1 모델을 학습한다.
    """

    # ------------------------------
    # 1. 모델 저장 경로 생성
    # ------------------------------

    out = MODEL / "stage1"

    out.mkdir(
        parents=True,
        exist_ok=True,
    )


    # ------------------------------
    # 2. AI-Hub Dataset 생성
    # ------------------------------

    train_dataset = AIHubStage1Dataset(
        root=STAGE1_DATA,
        manifest=TRAIN_MANIFEST,

        # MViT 입력 frame 수
        frames=16,

        # 입력 spatial size
        size=224,

        mean=S1_MEAN,
        std=S1_STD,

        # Train에서는 매 epoch마다
        # 다른 synthetic recapture artifact를 생성한다.
        train=True,
    )


    val_dataset = AIHubStage1Dataset(
        root=STAGE1_DATA,
        manifest=VAL_MANIFEST,

        frames=16,
        size=224,

        mean=S1_MEAN,
        std=S1_STD,

        # Validation에서는 동일한 source video에 대해
        # 항상 같은 synthetic artifact를 생성한다.
        train=False,
    )


    # Dataset은 source video 하나당
    #
    # ORIGINAL   1개
    # RERECORDED 1개
    #
    # 를 생성하므로 실제 sample 수는
    # manifest 영상 개수의 2배가 된다.

    print("=== Stage 1 Dataset ===")

    print(
        f"Train source videos: "
        f"{len(train_dataset) // 2}"
    )

    print(
        f"Train samples: "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation source videos: "
        f"{len(val_dataset) // 2}"
    )

    print(
        f"Validation samples: "
        f"{len(val_dataset)}"
    )


    # ------------------------------
    # 3. DataLoader 생성
    # ------------------------------

    train_loader = DataLoader(
        train_dataset,

        batch_size=2,

        shuffle=True,

        num_workers=4,

        pin_memory=True,
    )


    val_loader = DataLoader(
        val_dataset,

        batch_size=2,

        shuffle=False,

        num_workers=4,

        pin_memory=True,
    )


    # ------------------------------
    # 4. Stage 1 모델 생성
    # ------------------------------

    # Kinetics-400으로 사전학습된
    # MViTv2-S 기반 Stage 1 모델
    model = Stage1MViT()

    model = model.to(DEVICE)


    # ------------------------------
    # 5. Optimizer 설정
    # ------------------------------

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
    )


    # ------------------------------
    # 6. Best validation score
    # ------------------------------

    best_f1 = -1.0


    # ------------------------------
    # 7. Epoch 단위 학습
    # ------------------------------

    for epoch in range(EPOCHS):

        # ==========================
        # Training
        # ==========================

        model.train()

        running_loss = 0.0
        train_count = 0


        for x, y in train_loader:

            # x:
            # [B, C, T, H, W]
            #
            # ex.
            # [2, 3, 16, 224, 224]

            x = x.to(
                DEVICE,
                non_blocking=True,
            )

            y = y.to(
                DEVICE,
                non_blocking=True,
            )


            # Gradient 초기화
            opt.zero_grad()


            # Forward
            logits = model(x)


            # Cross Entropy
            loss = nn.functional.cross_entropy(
                logits,
                y,
            )


            # Backpropagation
            loss.backward()


            # Weight update
            opt.step()


            batch_size = y.size(0)

            running_loss += (
                loss.item()
                * batch_size
            )

            train_count += batch_size


        # ------------------------------
        # Train Loss
        # ------------------------------

        avg_loss = (
            running_loss
            / train_count
        )


        # ==========================
        # Validation
        # ==========================

        model.eval()

        y_true = []
        y_pred = []


        with torch.no_grad():

            for x, y in val_loader:

                x = x.to(
                    DEVICE,
                    non_blocking=True,
                )


                # Forward
                logits = model(x)


                # Prediction
                pred = logits.argmax(
                    dim=1
                )


                y_true.extend(
                    y.tolist()
                )


                y_pred.extend(
                    pred.cpu().tolist()
                )


        # ------------------------------
        # Validation Macro-F1
        # ------------------------------

        macro_f1 = f1_score(
            y_true,
            y_pred,
            average="macro",
        )

        print(
            "True:",
            Counter(y_true),
        )

        print(
            "Pred:",
            Counter(y_pred),
        )

        print(
            "Confusion Matrix:"
        )

        print(
            confusion_matrix(
                y_true,
                y_pred,
            )
        )

        # ------------------------------
        # Epoch 결과
        # ------------------------------

        print(
            f"[Stage 1] "
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"loss={avg_loss:.4f} | "
            f"val_macro_f1={macro_f1:.4f}"
        )


        # ------------------------------
        # Best checkpoint 저장
        # ------------------------------

        if macro_f1 > best_f1:

            best_f1 = macro_f1


            # inference.py에서는
            # Stage1MViT wrapper가 아니라
            # mvit_v2_s 본체에 직접 weight를
            # load하므로 model.net을 저장한다.
            torch.save(
                {
                    "model": model.net.state_dict(),

                    "size": 224,

                    "frames": 16,

                    "val_macro_f1": macro_f1,
                },

                out / "best.pt",
            )


            print(
                f"[Stage 1] "
                f"Best model updated: "
                f"{best_f1:.4f}"
            )


    # ------------------------------
    # 8. 학습 종료
    # ------------------------------

    print()

    print(
        f"[Stage 1] "
        f"Best Validation Macro-F1: "
        f"{best_f1:.4f}"
    )