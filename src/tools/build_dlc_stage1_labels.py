import pandas as pd
from pathlib import Path

from src.config import DLC_STAGE1_PROCESSED, DLC_STAGE1_RAW

DLC_STAGE1_DATA = DLC_STAGE1_RAW

VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".avi",
    ".m4v",
}


def find_videos(root: Path):
    return sorted(
        p
        for p in root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in VIDEO_EXTENSIONS
    )


def main():
    rows = []

    original_root = DLC_STAGE1_DATA / "original" / "clips_video"
    for path in find_videos(original_root):
        rows.append(
            {
                "path": str(path.relative_to(DLC_STAGE1_DATA)),
                "label": 0,
            }
        )

    recaptured_root = DLC_STAGE1_DATA / "recaptured" / "clips_video"
    for path in find_videos(recaptured_root):
        rows.append(
            {
                "path": str(path.relative_to(DLC_STAGE1_DATA)),
                "label": 1,
            }
        )

    df = pd.DataFrame(rows)

    print(df["label"].value_counts())
    print(df.head())

    output = DLC_STAGE1_PROCESSED / "labels.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)

    print(f"Saved: {output}")
    print(f"Total videos: {len(df)}")


if __name__ == "__main__":
    main()
