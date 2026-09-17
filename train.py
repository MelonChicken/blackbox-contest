from __future__ import annotations

import argparse

from src.train import fit_stage1, fit_stage2, fit_stage3


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the canonical training entrypoint for one stage.")
    parser.add_argument("stage", nargs="?", choices=("stage1", "stage2", "stage3"), default="stage3")
    args = parser.parse_args()
    if args.stage == "stage1":
        fit_stage1()
    elif args.stage == "stage2":
        fit_stage2()
    else:
        fit_stage3()


if __name__ == "__main__":
    main()
