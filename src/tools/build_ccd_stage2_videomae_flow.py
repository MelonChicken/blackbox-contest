from __future__ import annotations

from collections.abc import Callable

from src.tools import build_ccd_collision_candidates
from src.tools import build_ccd_stage2_manifest
from src.tools import build_ccd_vehicles_tracks
from src.tools import build_ccd_videomae_manifest
from src.tools import split_ccd_videomae_manifest


Step = tuple[str, Callable[[], None]]


def build_ccd_stage2_videomae_flow(
    include_tracking: bool = True,
) -> None:
    steps: list[Step] = [
        (
            "base manifest",
            build_ccd_stage2_manifest.main,
        ),
    ]

    if include_tracking:
        steps.extend(
            [
                (
                    "vehicle tracks",
                    build_ccd_vehicles_tracks.main,
                ),
                (
                    "collision candidates",
                    build_ccd_collision_candidates.main,
                ),
            ]
        )

    steps.extend(
        [
            (
                "VideoMAE manifest",
                build_ccd_videomae_manifest.main,
            ),
            (
                "VideoMAE train/val split",
                split_ccd_videomae_manifest.main,
            ),
        ]
    )

    for index, (name, run_step) in enumerate(
        steps,
        start=1,
    ):
        print(
            f"[{index}/{len(steps)}] {name}"
        )

        run_step()

        print()


def main() -> None:
    build_ccd_stage2_videomae_flow()


if __name__ == "__main__":
    main()
