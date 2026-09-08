from __future__ import annotations

import argparse

from src.tools.cache_stage3_tartanvo_feature import cache_split


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache frozen TartanVO pose features for Stage3.")
    parser.add_argument("--split", choices=["train", "val"], required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    cache_split(args.split, "pose", overwrite=args.overwrite)


if __name__ == "__main__":
    main()
