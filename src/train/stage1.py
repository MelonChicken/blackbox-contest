# 1. Configuration

# 1.1. ?온?????텕筌왖 ?袁る７??
from collections import Counter

import torch
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader, Subset

from sklearn.metrics import (
    f1_score,
    confusion_matrix,
)

# AI-Hub ?됰뗀?볢쳸類ㅻ뮞 ?怨멸맒??Stage 1 ??덈뮸 ?怨쀬뵠?怨뺤쨮 ??볥궗??롫뮉 Dataset
from src.datasets.aihubDataset import AIHubStage1Dataset

# Stage 1 筌뤴뫀??嚥≪뮆諭?
from src.models import Stage1MViT

# ??뺣쑁 ??뺣굡 ?⑥쥙????袁る립 ??λ땾
from src.utils import set_seed

# ?⑤벏??configuration 嚥≪뮆諭?
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


# ??뺣쑁 ??뺣굡???⑥쥙???뤿연
# ??덉뵬??鈺곌퀗援?癒?퐣 ??쎈퓮???????????덈즲嚥???뺣뼄.
set_seed(SEED)


# --------------------------------------------------
# Stage 1 AI-Hub ?怨쀬뵠??野껋럥以?
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
    AI-Hub ?대???????됰뗀?볢쳸類ㅻ뮞 ?怨멸맒????곸뒠??뤿연

    ORIGINAL:
        AI-Hub ?癒?궚 ?怨멸맒

    RERECORDED:
        ?癒?궚 ?怨멸맒??synthetic recapture augmentation ?怨몄뒠

    ???????? ?브쑬履??롫뮉 Stage 1 筌뤴뫀?????덈뮸??뺣뼄.

    Dataset?? source video ??롪돌????甕곕뜄彛?decode????

        ORIGINAL
        RERECORDED

    ??sample????덈뻻????밴쉐??뺣뼄.
    """

    # ------------------------------
    # 1. 筌뤴뫀??????野껋럥以???밴쉐
    # ------------------------------

    out = STAGE1_MODEL

    out.mkdir(
        parents=True,
        exist_ok=True,
    )


    # ------------------------------
    # 2. AI-Hub Dataset ??밴쉐
    # ------------------------------

    train_dataset = AIHubStage1Dataset(
        root=STAGE1_RAW,
        manifest=TRAIN_MANIFEST,

        frames=16,
        size=224,

        mean=S1_MEAN,
        std=S1_STD,

        train=True,
    )


    val_dataset = AIHubStage1Dataset(
        root=STAGE1_RAW,
        manifest=VAL_MANIFEST,

        frames=16,
        size=224,

        mean=S1_MEAN,
        std=S1_STD,

        train=False,
    )


    # ------------------------------
    # 3. Train source video ??쀫립
    # ------------------------------
    #
    # ??덉쨮??Dataset?癒?퐣??
    #
    # len(dataset)
    #
    # ?癒?퍥揶쎛 source video ??륁뵠??
    #
    # Dataset item ??롪돌??
    #
    # ORIGINAL
    # RERECORDED
    #
    # ??sample????ｍ뜞 獄쏆꼹???뺣뼄.
    # ------------------------------

    num_train_sources = len(train_dataset)

    if (
        TRAIN_SOURCE_LIMIT is not None
        and num_train_sources > TRAIN_SOURCE_LIMIT
    ):

        generator = torch.Generator()
        generator.manual_seed(SEED)

        # source video 疫꿸퀣? ??뺣쑁 ?醫뤾문
        selected_sources = torch.randperm(
            num_train_sources,
            generator=generator,
        )[:TRAIN_SOURCE_LIMIT].tolist()

        train_dataset = Subset(
            train_dataset,
            selected_sources,
        )


    # ------------------------------
    # 4. Dataset ?類ｋ궖 ?곗뮆??
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
    # 5. DataLoader ??밴쉐
    # ------------------------------
    #
    # 筌띲끉??餓λ쵐??
    #
    # batch_size????곸젫 clip ??? ?袁⑤빍??
    # source video ??롫뼄.
    #
    # BATCH_SIZE = 2??겹늺
    #
    # source 2揶?
    # ??
    # ORIGINAL/RERECORDED 2揶?
    #
    # ??쇱젫 GPU batch = 4
    #
    # num_workers=4 / prefetch=2??
    # ?袁⑹삺 ??뺤쒔 ?怨뱀넺?癒?퐣????쇰뻻 筌롫뗀?덄뵳?CPU
    # ?얜챷?ｅ첎? 獄쏆뮇源??揶쎛?關苑????됱몵沃샕嚥?
    # ?怨쀪퐨 癰귣똻??怨몄몵嚥???뽰삂??뺣뼄.
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
    # 6. Stage 1 筌뤴뫀????밴쉐
    # ------------------------------

    # Kinetics-400??곗쨮 ?????덈뮸??
    # MViTv2-S 疫꿸퀡而?Stage 1 筌뤴뫀??
    model = Stage1MViT()

    model = model.to(DEVICE)


    # ------------------------------
    # 7. Optimizer ??쇱젟
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
    # 9. Epoch ??μ맄 ??덈뮸
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

            # Dataset ??롪돌??獄쏆꼹??
            #
            # x:
            # [2, C, T, H, W]
            #
            # y:
            # [2]
            #
            # DataLoader ??꾩뜎:
            #
            # x:
            # [B, 2, C, T, H, W]
            #
            # y:
            # [B, 2]
            #
            # MViT???節딅┛ ?袁る퉸:
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
            # GPU ??猷?
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
            # Gradient ?λ뜃由??
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
            # Train loss 筌욌쵌??
            # ----------------------

            # flatten ??꾩뜎???嚥?
            # ??쇱젫 clip 揶쏆뮇??
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
                # GPU ??猷?
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


                # y???袁⑹춦 CPU????됱몵沃샕嚥?
                # 獄쏅뗀以?list嚥?癰궰??묐립??

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
        # Validation ?怨멸쉭 野껉퀗??
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
        # Epoch 野껉퀗??
        # ------------------------------

        print(
            f"[Stage 1] "
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"loss={avg_loss:.4f} | "
            f"val_macro_f1={macro_f1:.4f}"
        )


        # ------------------------------
        # Best checkpoint ????
        # ------------------------------

        if macro_f1 > best_f1:

            best_f1 = macro_f1


            # inference.py?癒?퐣??
            # Stage1MViT wrapper揶쎛 ?袁⑤빍??
            # mvit_v2_s 癰귣챷猿??筌욊낯??weight??
            # load???嚥?model.net?????館釉??

            torch.save(
                {
                    "model":
                        model.net.state_dict(),

                    "size":
                        224,

                    "frames":
                        16,

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
    # 10. ??덈뮸 ?ル굝利?
    # ------------------------------

    print()

    print(
        f"[Stage 1] "
        f"Best Validation Macro-F1: "
        f"{best_f1:.4f}"
    )
