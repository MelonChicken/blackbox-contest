from pathlib import Path
import pandas as pd

from src.config import DATA

DATA = DATA / 'stage1' / 'dlc2021'

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

    # ORIGINAL = 0
    original_root = DATA / "original" / "clips_video"

    for path in find_videos(original_root):
        rows.append(
            {
                "path": str(path.relative_to(DATA)),
                "label": 0,
            }
        )

    # RERECORDED = 1
    recaptured_root = DATA / "recaptured" / "clips_video"

    for path in find_videos(recaptured_root):
        rows.append(
            {
                "path": str(path.relative_to(DATA)),
                "label": 1,
            }
        )

    df = pd.DataFrame(rows)

    print(df["label"].value_counts())
    print(df.head())

    output = DATA / "labels.csv"
    df.to_csv(output, index=False)

    print(f"Saved: {output}")
    print(f"Total videos: {len(df)}")


if __name__ == "__main__":
    main()