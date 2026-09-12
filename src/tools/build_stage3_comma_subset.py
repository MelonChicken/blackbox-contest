from __future__ import annotations

import argparse

from src.config import SEED, STAGE3_COMMA_TRAIN_SEGMENTS_PER_ROUTE, STAGE3_COMMA_VAL_SEGMENTS_PER_ROUTE
from src.tools.stage3_comma_manifest import build_route_balanced_subsets, print_subset_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build route-balanced comma2k19 Stage 3 subset manifests.")
    parser.add_argument("--train-segments-per-route", type=int, default=STAGE3_COMMA_TRAIN_SEGMENTS_PER_ROUTE)
    parser.add_argument("--val-segments-per-route", type=int, default=STAGE3_COMMA_VAL_SEGMENTS_PER_ROUTE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--seconds-per-segment", type=float)
    parser.add_argument("--mb-per-segment", type=float)
    args = parser.parse_args()
    metadata = build_route_balanced_subsets(args.train_segments_per_route, args.val_segments_per_route, args.seed)
    print_subset_summary(metadata, args.seconds_per_segment, args.mb_per_segment)


if __name__ == "__main__":
    main()
