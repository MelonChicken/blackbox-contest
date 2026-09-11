from __future__ import annotations

from pathlib import Path

from src.config import STAGE3_OUTPUT_HZ
from src.datasets.comma2k19_stage3 import Comma2k19Stage3Dataset
from src.tools.stage3_comma_debug import abs_video_path, cap_info, feature_index, read_manifest


def _print_rows(split: str, max_segments: int = 3) -> None:
    df = read_manifest(split)
    ds = Comma2k19Stage3Dataset(__import__('src.config').config.COMMA2K19_STAGE3_TRAIN_MANIFEST if split == 'train' else __import__('src.config').config.COMMA2K19_STAGE3_VAL_MANIFEST)
    print(f"[{split} dataset]")
    key = "segment_id" if "segment_id" in df.columns else "video_path"
    for _, seg in list(df.groupby(key, sort=False))[:max_segments]:
        for idx in [0, min(100, len(seg) - 1), len(seg) - 1]:
            row = seg.iloc[int(idx)]
            sample_index = int(round(float(row.timestamp) * STAGE3_OUTPUT_HZ))
            expected_frame = 2 * sample_index
            actual_frame = int(row.frame_index)
            target_time = sample_index / STAGE3_OUTPUT_HZ
            actual_time = float(row.timestamp)
            video = abs_video_path(row)
            fps, total = cap_info(video) if video.is_file() else (20.0, int(seg.frame_index.max()) + 1)
            print("segment path", Path(str(row.video_path)).parent)
            print("original fps", fps)
            print("total original frames", total)
            print("expected 10Hz samples", total // 2)
            print(f"sample_index={sample_index}")
            print(f"target_time={target_time:.3f}")
            print(f"expected_original_frame={expected_frame}")
            print(f"actual_center_frame_used_by_dataset={actual_frame}")
            print(f"actual frame timestamp={actual_time:.3f}")
            print(f"time error={actual_time - target_time:.6f}")
            item = ds[df.index.get_loc(row.name)] if row.name in df.index else None
            if item:
                print(f"dataset_item_frame_index={item['frame_index']}")
            print()


def _print_cache(split: str) -> None:
    idx = feature_index(split)
    print(f"[{split} cache]")
    if idx is None:
        print("cache index: missing")
        return
    for row in idx.head(5).itertuples(index=False):
        sample_index = int(round(float(getattr(row, 'timestamp', 0.0)) * STAGE3_OUTPUT_HZ)) if hasattr(row, 'timestamp') else None
        expected = 2 * sample_index if sample_index is not None else 'timestamp missing'
        print(f"sample_key={row.sample_key} frame_index={getattr(row, 'frame_index', 'missing')} expected_frame={expected}")


def main() -> None:
    for split in ("train", "val"):
        try:
            _print_rows(split)
            _print_cache(split)
        except FileNotFoundError as exc:
            print(exc)


if __name__ == "__main__":
    main()