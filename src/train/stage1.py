# 1. Configuration

# 1.1. 라이브러리와 학습 구성 불러오기
from collections import Counter

import torch
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader, Subset

from sklearn.metrics import (
    f1_score,
    confusion_matrix,
)

# AI-Hub Stage 1 ORIGINAL/RERECORDED 학습용 Dataset
from src.datasets.aihubDataset import AIHubStage1Dataset

# Stage 1 모델
from src.models import Stage1MViT

# 재현 가능한 실험을 위한 seed 설정 유틸리티
from src.utils import set_seed

# 프로젝트 configuration 불러오기
from src.config import (
    STAGE1_MODEL,
    AIHUB_STAGE1_RAW,
    AIHUB_STAGE1_MANIFEST,
    DEVICE,
    EPOCHS,
    S1_MEAN,
    S1_STD,
    SEED,
    TRAIN_SOURCE_LIMIT,
    BATCH_SIZE,
)


# 재현 가능한 학습을 위해 난수 seed를 고정한다.
# DataLoader worker와 torch 연산의 무작위성을 동일하게 맞춘다.
set_seed(SEED)


# --------------------------------------------------
# Stage 1 AI-Hub 학습 경로 설정
# --------------------------------------------------

STAGE1_RAW = AIHUB_STAGE1_RAW

STAGE1_MANIFEST = AIHUB_STAGE1_MANIFEST

TRAIN_MANIFEST = (
    STAGE1_MANIFEST
    / "train.csv"
)

VAL_MANIFEST = (
    STAGE1_MANIFEST
    / "val.csv"
)


def fit_stage1():
    """
    AI-Hub Stage 1 ORIGINAL/RERECORDED 학습 코드.

    ORIGINAL:
        AI-Hub 원본 영상 샘플.

    RERECORDED:
        원본 영상에 synthetic recapture augmentation을 적용한 샘플.

    ORIGINAL과 synthetic RERECORDED를 구분하도록 Stage 1 모델을 학습한다.

    Dataset index 하나는 source video 하나를 의미한다.

        ORIGINAL
        RERECORDED

    source 하나에서 ORIGINAL과 synthetic RERECORDED 2개 sample을 생성한다.
    """

    # ------------------------------
    # 1. 모델 저장 경로 생성
    # ------------------------------

    out = STAGE1_MODEL

    out.mkdir(
        parents=True,
        exist_ok=True,
    )


    # ------------------------------
    # 2. AI-Hub Dataset 생성
    # ------------------------------

    train_dataset = AIHubStage1Dataset(
        root=STAGE1_RAW,
        manifest=TRAIN_MANIFEST,

        frames=16,
        size=224,
        recapture_size=320,

        mean=S1_MEAN,
        std=S1_STD,

        train=True,
    )

    val_dataset = AIHubStage1Dataset(
        root=STAGE1_RAW,
        manifest=VAL_MANIFEST,

        frames=16,
        size=224,
        recapture_size=320,

        mean=S1_MEAN,
        std=S1_STD,

        train=False,
    )


    # ------------------------------
    # 3. Train source video 수 제한
    # ------------------------------
    #
    # AIHubStage1Dataset에서
    #
    # len(dataset)
    #
    # len(dataset)은 source video 개수를 의미한다.
    #
    # Dataset item 하나는
    #
    # ORIGINAL
    # RERECORDED
    #
    # 두 sample을 함께 반환한다.
    # ------------------------------

    num_train_sources = len(train_dataset)

    if (
        TRAIN_SOURCE_LIMIT is not None
        and num_train_sources > TRAIN_SOURCE_LIMIT
    ):

        generator = torch.Generator()
        generator.manual_seed(SEED)

        # source video 단위로 재현 가능하게 샘플링한다.
        selected_sources = torch.randperm(
            num_train_sources,
            generator=generator,
        )[:TRAIN_SOURCE_LIMIT].tolist()

        train_dataset = Subset(
            train_dataset,
            selected_sources,
        )


    # ------------------------------
    # 4. Dataset 규모 확인
    # ------------------------------

    print("=== Stage 1 Dataset ===")

    print(
        f"Train source videos: "
        f"{len(train_dataset)}"
    )

    print(
        f"Train samples: "
        f"{len(train_dataset) * 2}"
    )

    print(
        f"Validation source videos: "
        f"{len(val_dataset)}"
    )

    print(
        f"Validation samples: "
        f"{len(val_dataset) * 2}"
    )

    print(
        f"Source batch size: "
        f"{BATCH_SIZE}"
    )

    print(
        f"Effective MViT batch size: "
        f"{BATCH_SIZE * 2}"
    )


    # ------------------------------
    # 5. DataLoader 생성
    # ------------------------------
    #
    # 여기서 batch_size는
    #
    # 실제 clip sample 개수가 아니라
    # source video 개수이다.
    #
    # BATCH_SIZE = 2라면
    #
    # source 2개
    # x
    # ORIGINAL/RERECORDED 2개
    #
    # 실제 MViT batch = 4
    #
    # BATCH_SIZE는 source batch size이며
    # 실제 MViT batch는 BATCH_SIZE * 2이다.
    # DataLoader는 [B,2,C,T,H,W]와 [B,2]를 반환한다.
    # 학습 루프에서 각각 [B*2,C,T,H,W], [B*2]로 펼친다.
    # ------------------------------

    train_loader = DataLoader(
        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        num_workers=1,

        pin_memory=True,

        persistent_workers=True,

        prefetch_factor=1,
    )


    val_loader = DataLoader(
        val_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=1,

        pin_memory=True,

        persistent_workers=True,

        prefetch_factor=1,
    )


    # ------------------------------
    # 6. Stage 1 모델 생성
    # ------------------------------

    # Kinetics-400 pretrained MViTv2-S 기반 Stage 1 모델
    # ORIGINAL/RERECORDED 2-class 분류를 수행한다.
    model = Stage1MViT()

    model = model.to(DEVICE)


    # ------------------------------
    # 7. Optimizer 설정
    # ------------------------------

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
    )


    # ------------------------------
    # 8. Best validation score
    # ------------------------------

    best_f1 = -1.0


    # ------------------------------
    # 9. Epoch 학습 루프
    # ------------------------------

    for epoch in range(EPOCHS):

        # ==========================
        # Training
        # ==========================

        model.train()

        running_loss = 0.0
        train_count = 0


        for x, y in tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{EPOCHS} Train",
        ):

            # Dataset item 하나의 x/y shape
            #
            # x:
            # [2, C, T, H, W]
            #
            # y:
            # [2]
            #
            # DataLoader batch shape
            #
            # x:
            # [B, 2, C, T, H, W]
            #
            # y:
            # [B, 2]
            #
            # MViT 입력 shape으로 펼친다.
            #
            # [B, 2, C, T, H, W]
            # ->
            # [B*2, C, T, H, W]

            x = x.flatten(
                0,
                1,
            )

            # [B, 2]
            # ->
            # [B*2]

            y = y.flatten()


            # ----------------------
            # GPU 전송
            # ----------------------

            x = x.to(
                DEVICE,
                non_blocking=True,
            )

            y = y.to(
                DEVICE,
                non_blocking=True,
            )


            # ----------------------
            # Gradient 초기화
            # ----------------------

            opt.zero_grad()


            # ----------------------
            # Forward
            # ----------------------

            logits = model(x)


            # ----------------------
            # Cross Entropy
            # ----------------------

            loss = nn.functional.cross_entropy(
                logits,
                y,
            )


            # ----------------------
            # Backpropagation
            # ----------------------

            loss.backward()


            # ----------------------
            # Weight update
            # ----------------------

            opt.step()


            # ----------------------
            # Train loss 누적
            # ----------------------

            # flatten 이후 y 길이가 실제 clip sample 개수이다.
            # source batch가 아니라 MViT 입력 batch 기준으로 누적한다.
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

                # [B,2,C,T,H,W]
                # ->
                # [B*2,C,T,H,W]

                x = x.flatten(
                    0,
                    1,
                )

                # [B,2]
                # ->
                # [B*2]

                y = y.flatten()


                # ------------------
                # GPU 전송
                # ------------------

                x = x.to(
                    DEVICE,
                    non_blocking=True,
                )


                # ------------------
                # Forward
                # ------------------

                logits = model(x)


                # ------------------
                # Prediction
                # ------------------

                pred = logits.argmax(
                    dim=1
                )


                # y는 CPU tensor이므로 그대로 list로 변환하고,
                # pred는 CPU로 옮긴 뒤 list로 변환한다.

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


        # ------------------------------
        # Validation Macro-F1, prediction distribution, confusion matrix
        # ------------------------------

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
        # Epoch 결과 출력
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


            # inference.py에서는 Stage1MViT wrapper가 아니라
            # 내부 mvit_v2_s 네트워크 weight를 직접 load하므로
            # best checkpoint에는 model.net.state_dict(), size, frames,
            # val_macro_f1를 저장한다.

            torch.save(
                {
                    "model":
                        model.net.state_dict(),

                    "size":
                        224,

                    "frames":
                        16,

                    "recapture_size":
                        320,

                    "val_macro_f1":
                        macro_f1,
                },

                out / "best.pt",
            )


            print(
                f"[Stage 1] "
                f"Best model updated: "
                f"{best_f1:.4f}"
            )


    # ------------------------------
    # 10. 학습 종료
    # ------------------------------

    print()

    print(
        f"[Stage 1] "
        f"Best Validation Macro-F1: "
        f"{best_f1:.4f}"
    )
