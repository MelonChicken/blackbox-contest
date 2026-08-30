import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import AIHUB_STAGE1_MANIFEST, AIHUB_STAGE1_RAW, PROJECT_ROOT

# --------------------------------------------------
# Path
# --------------------------------------------------

RAW_ROOT = AIHUB_STAGE1_RAW
MANIFEST_ROOT = AIHUB_STAGE1_MANIFEST
VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
}


def detect_split(path: Path):
    """
    AI-Hub??怨듭떇 Training / Validation 援ъ“瑜?
    洹몃?濡??ъ슜?쒕떎.

    1.Training   -> train
    2.Validation -> val
    """

    path_text = str(path).lower()

    if "1.training" in path_text:
        return "train"

    if "2.validation" in path_text:
        return "val"

    return None


def load_metadata(
    json_path: Path,
):
    """
    AI-Hub label JSON?먯꽌
    video metadata瑜??쎈뒗??
    """

    with json_path.open(
        "r",
        encoding="utf-8-sig",
    ) as f:
        data = json.load(f)

    video = data["video"]

    return {
        "filming_way": (
            video.get("filming_way")
        ),

        "video_point_of_view": (
            video.get(
                "video_point_of_view"
            )
        ),
    }


def main():

    print("=== Paths ===")
    print(
        f"PROJECT_ROOT: {PROJECT_ROOT}"
    )
    print(
        f"RAW_ROOT:     {RAW_ROOT}"
    )
    print()

    if not RAW_ROOT.exists():
        raise FileNotFoundError(
            f"Raw directory not found: "
            f"{RAW_ROOT}"
        )

    MANIFEST_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )


    # --------------------------------------------------
    # 1. JSON ?먯깋
    # --------------------------------------------------

    json_paths = list(
        RAW_ROOT.rglob("*.json")
    )

    print(
        f"JSON files: {len(json_paths)}"
    )


    # 媛숈? stem??JSON??李얘린 ?꾪븳 index
    json_index = {}

    for path in json_paths:

        # ?뱀떆 媛숈? ?뚯씪紐낆씠 ?щ윭 split???덉쓣 寃쎌슦瑜??鍮꾪빐
        # split??媛숈씠 key濡??ъ슜
        split = detect_split(path)

        if split is None:
            continue

        json_index[
            (
                split,
                path.stem,
            )
        ] = path


    # --------------------------------------------------
    # 2. Video ?먯깋
    # --------------------------------------------------

    video_paths = [
        path
        for path in RAW_ROOT.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower()
            in VIDEO_EXTENSIONS
        )
    ]

    print(
        f"Video files: {len(video_paths)}"
    )


    # --------------------------------------------------
    # 3. MP4 ??JSON matching
    # --------------------------------------------------

    records = []

    unmatched = 0
    cctv = 0
    blackbox = 0

    for video_path in video_paths:

        split = detect_split(
            video_path
        )

        if split is None:
            continue


        json_path = json_index.get(
            (
                split,
                video_path.stem,
            )
        )


        if json_path is None:

            unmatched += 1

            continue


        try:

            metadata = load_metadata(
                json_path
            )

        except Exception as e:

            print(
                f"[WARN] JSON read failed: "
                f"{json_path} | {e}"
            )

            continue


        # ------------------------------------------
        # 釉붾옓諛뺤뒪留??ъ슜
        # ------------------------------------------

        filming_way = (
            metadata["filming_way"]
        )

        if filming_way != "bb":

            cctv += 1

            continue


        blackbox += 1


        records.append(
            {
                "source_id": (
                    video_path.stem
                ),

                "video_path": str(
                    video_path.relative_to(
                        RAW_ROOT
                    )
                ),

                "label_path": str(
                    json_path.relative_to(
                        RAW_ROOT
                    )
                ),

                "split": split,

                "filming_way": (
                    filming_way
                ),

                "point_of_view": (
                    metadata[
                        "video_point_of_view"
                    ]
                ),
            }
        )


    # --------------------------------------------------
    # 4. DataFrame ?앹꽦
    # --------------------------------------------------

    df = pd.DataFrame(
        records
    )

    if len(df) == 0:
        raise RuntimeError(
            "No blackbox videos found."
        )


    # --------------------------------------------------
    # 5. AI-Hub 怨듭떇 split ?ъ슜
    # --------------------------------------------------

    train_df = (
        df[
            df["split"] == "train"
        ]
        .reset_index(drop=True)
    )

    val_df = (
        df[
            df["split"] == "val"
        ]
        .reset_index(drop=True)
    )


    # --------------------------------------------------
    # 6. Manifest ???
    # --------------------------------------------------

    train_df.to_csv(
        MANIFEST_ROOT
        / "train.csv",

        index=False,
    )

    val_df.to_csv(
        MANIFEST_ROOT
        / "val.csv",

        index=False,
    )


    # --------------------------------------------------
    # 7. 寃곌낵 異쒕젰
    # --------------------------------------------------

    print()
    print(
        "=== AI-Hub Stage 1 Manifest ==="
    )

    print(
        f"Blackbox: {blackbox}"
    )

    print(
        f"CCTV excluded: {cctv}"
    )

    print(
        f"Unmatched: {unmatched}"
    )

    print()

    print(
        f"Train source videos: "
        f"{len(train_df)}"
    )

    print(
        f"Validation source videos: "
        f"{len(val_df)}"
    )

    print()


    if len(train_df) > 0:

        print(
            "Train point-of-view:"
        )

        print(
            train_df[
                "point_of_view"
            ].value_counts(
                dropna=False
            )
        )


    print()


    if len(val_df) > 0:

        print(
            "Validation point-of-view:"
        )

        print(
            val_df[
                "point_of_view"
            ].value_counts(
                dropna=False
            )
        )


if __name__ == "__main__":
    main()
