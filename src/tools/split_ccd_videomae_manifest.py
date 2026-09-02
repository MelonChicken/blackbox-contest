from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.config import CCD_STAGE2_MANIFEST


INPUT_PATH = (
    CCD_STAGE2_MANIFEST
    / "videomae"
    / "all.csv"
)

OUTPUT_DIR = (
    CCD_STAGE2_MANIFEST
    / "videomae"
)

TRAIN_PATH = OUTPUT_DIR / "train.csv"
VAL_PATH = OUTPUT_DIR / "val.csv"

SEED = 42
VAL_SIZE = 0.15


def main() -> None:
    df = pd.read_csv(
        INPUT_PATH,
        dtype={
            "video_id": str,
            "source_id": str,
        },
    )

    df["video_id"] = (
        df["video_id"]
        .astype(str)
        .str.zfill(6)
    )

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=VAL_SIZE,
        random_state=SEED,
    )

    train_idx, val_idx = next(
        splitter.split(
            df,
            groups=df["source_id"],
        )
    )

    train_df = (
        df.iloc[train_idx]
        .reset_index(drop=True)
    )

    val_df = (
        df.iloc[val_idx]
        .reset_index(drop=True)
    )

    train_df.to_csv(
        TRAIN_PATH,
        index=False,
    )

    val_df.to_csv(
        VAL_PATH,
        index=False,
    )

    train_sources = set(
        train_df["source_id"]
    )

    val_sources = set(
        val_df["source_id"]
    )

    overlap = (
        train_sources
        & val_sources
    )

    print(
        "=== CCD VideoMAE Split ==="
    )

    print(
        f"Train videos: {len(train_df)}"
    )

    print(
        f"Val videos: {len(val_df)}"
    )

    print(
        f"Train sources: "
        f"{len(train_sources)}"
    )

    print(
        f"Val sources: "
        f"{len(val_sources)}"
    )

    print(
        f"Source overlap: "
        f"{len(overlap)}"
    )

    print()

    print(
        "Train labels:"
    )

    print(
        f"  collision: "
        f"{train_df['collision_valid'].sum()}"
    )

    print(
        f"  side: "
        f"{train_df['side_valid'].sum()}"
    )

    print()

    print(
        "Validation labels:"
    )

    print(
        f"  collision: "
        f"{val_df['collision_valid'].sum()}"
    )

    print(
        f"  side: "
        f"{val_df['side_valid'].sum()}"
    )

    print()

    print(
        f"Saved: {TRAIN_PATH}"
    )

    print(
        f"Saved: {VAL_PATH}"
    )


if __name__ == "__main__":
    main()