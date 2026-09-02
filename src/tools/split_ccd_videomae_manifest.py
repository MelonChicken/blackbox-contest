from __future__ import annotations

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.config import STAGE2_ALL_MANIFEST, STAGE2_MANIFEST, STAGE2_TRAIN_MANIFEST, STAGE2_VAL_MANIFEST


INPUT_PATH = STAGE2_ALL_MANIFEST
OUTPUT_DIR = STAGE2_MANIFEST
TRAIN_PATH = STAGE2_TRAIN_MANIFEST
VAL_PATH = STAGE2_VAL_MANIFEST
SEED = 42
VAL_SIZE = 0.15


def _count_valid(df: pd.DataFrame, column: str) -> int:
    return int((df[column].fillna(-1).astype(int) >= 0).sum())


def main() -> None:
    df = pd.read_csv(INPUT_PATH, dtype={"video_id": str, "source_id": str})
    if "video_id" in df.columns:
        df["video_id"] = df["video_id"].astype(str).str.zfill(6)

    groups = df["source_id"] if "source_id" in df.columns else df.get("video_id", df.index)
    splitter = GroupShuffleSplit(n_splits=1, test_size=VAL_SIZE, random_state=SEED)
    train_idx, val_idx = next(splitter.split(df, groups=groups))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    train_df.to_csv(TRAIN_PATH, index=False)
    val_df.to_csv(VAL_PATH, index=False)

    train_groups = set(groups.iloc[train_idx].astype(str)) if hasattr(groups, "iloc") else set()
    val_groups = set(groups.iloc[val_idx].astype(str)) if hasattr(groups, "iloc") else set()
    print("=== Stage2 Manifest Split ===")
    print(f"Train rows: {len(train_df)}")
    print(f"Val rows: {len(val_df)}")
    print(f"Group overlap: {len(train_groups & val_groups)}")
    print("Train labels:")
    print(f"  collision: {_count_valid(train_df, 'collision_frame')}")
    print(f"  entry: {_count_valid(train_df, 'entry_frame')}")
    print(f"  direction: {_count_valid(train_df, 'direction')}")
    print(f"  avoidance: {_count_valid(train_df, 'avoidance')}")
    print("Validation labels:")
    print(f"  collision: {_count_valid(val_df, 'collision_frame')}")
    print(f"  entry: {_count_valid(val_df, 'entry_frame')}")
    print(f"  direction: {_count_valid(val_df, 'direction')}")
    print(f"  avoidance: {_count_valid(val_df, 'avoidance')}")
    print(f"Saved: {TRAIN_PATH}")
    print(f"Saved: {VAL_PATH}")


if __name__ == "__main__":
    main()
