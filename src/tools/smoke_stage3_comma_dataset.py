from __future__ import annotations

from src.train.stage3 import _datasets, _print_dataset_summary
from src.tools.stage3_comma_debug import print_route_overlap, print_sample_count_audit


def main() -> None:
    try:
        train, val, summary = _datasets()
        _print_dataset_summary(train, val, summary)
        print_route_overlap()
        for split in ("train", "val"):
            print_sample_count_audit(split)
        print("comma dataset smoke passed")
    except FileNotFoundError as exc:
        print(exc)


if __name__ == "__main__":
    main()