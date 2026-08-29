import json
from pathlib import Path

import pandas as pd


# --------------------------------------------------
# Path
# --------------------------------------------------

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

ROOT = (
    PROJECT_ROOT
    / "data"
    / "stage1"
    / "aihub597"
)

RAW_ROOT = ROOT / "raw"

MANIFEST_ROOT = (
    ROOT
    / "manifest"
)

VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
}


def detect_split(path: Path):
    """
    AI-Hub의 공식 Training / Validation 구조를
    그대로 사용한다.

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
    AI-Hub label JSON에서
    video metadata를 읽는다.
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
    # 1. JSON 탐색
    # --------------------------------------------------

    json_paths = list(
        RAW_ROOT.rglob("*.json")
    )

    print(
        f"JSON files: {len(json_paths)}"
    )


    # 같은 stem의 JSON을 찾기 위한 index
    json_index = {}

    for path in json_paths:

        # 혹시 같은 파일명이 여러 split에 있을 경우를 대비해
        # split도 같이 key로 사용
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
    # 2. Video 탐색
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
    # 3. MP4 ↔ JSON matching
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
        # 블랙박스만 사용
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
                        ROOT
                    )
                ),

                "label_path": str(
                    json_path.relative_to(
                        ROOT
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
    # 4. DataFrame 생성
    # --------------------------------------------------

    df = pd.DataFrame(
        records
    )

    if len(df) == 0:
        raise RuntimeError(
            "No blackbox videos found."
        )


    # --------------------------------------------------
    # 5. AI-Hub 공식 split 사용
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
    # 6. Manifest 저장
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
    # 7. 결과 출력
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