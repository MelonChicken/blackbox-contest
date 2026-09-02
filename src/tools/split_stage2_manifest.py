from __future__ import annotations

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.config import STAGE2_ALL_MANIFEST, STAGE2_MANIFEST, STAGE2_TRAIN_MANIFEST, STAGE2_VAL_MANIFEST

SEED = 42
VAL_SIZE = 0.15


def _count_valid(df: pd.DataFrame, column: str) -> int:
    return int((df[column].fillna(-1).astype(int) >= 0).sum())


def main() -> None:
    df = pd.read_csv(STAGE2_ALL_MANIFEST, dtype={"video_id": str, "source_id": str})
    groups = df["source_id"] if "source_id" in df.columns else df.get("video_id", df.index)
    train_idx, val_idx = next(GroupShuffleSplit(n_splits=1, test_size=VAL_SIZE, random_state=SEED).split(df, groups=groups))

    STAGE2_MANIFEST.mkdir(parents=True, exist_ok=True)
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    train_df.to_csv(STAGE2_TRAIN_MANIFEST, index=False)
    val_df.to_csv(STAGE2_VAL_MANIFEST, index=False)

    print("=== Stage2 Manifest Split ===")
    print(f"Train rows: {len(train_df)}")
    print(f"Val rows: {len(val_df)}")
    for name in ("collision_frame", "entry_frame", "direction", "avoidance"):
        print(f"{name}: train={_count_valid(train_df, name)} val={_count_valid(val_df, name)}")
    print(f"Saved: {STAGE2_TRAIN_MANIFEST}")
    print(f"Saved: {STAGE2_VAL_MANIFEST}")


if __name__ == "__main__":
    main()
