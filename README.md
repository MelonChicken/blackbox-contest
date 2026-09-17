# DACON Blackbox Video Pipeline

This repository keeps one active submission pipeline per stage.

## Active Models

- Stage 1: MViT-v2-S recapture classifier
- Stage 2: VideoMAE collision and direction model
- Stage 3: MViT-v2-S comma2k19-only acceleration and steering model

## Data Flow

```text
manifest -> frame cache or dataset images -> train -> checkpoint -> submission inference
```

Stage 1 and Stage 2 currently read dataset inputs directly, so their cache command is a no-op wrapper. Stage 3 uses a comma2k19 JPEG frame cache.

## Canonical Commands

```bash
python -m src.stage1.manifest
python -m src.stage1.cache
python -m src.stage1.train

python -m src.stage2.manifest
python -m src.stage2.cache
python -m src.stage2.train

python -m src.stage3.manifest
python -m src.stage3.cache
python -m src.stage3.train
```

Root training wrapper:

```bash
python train.py stage1
python train.py stage2
python train.py stage3
```

## Checkpoints

Default checkpoint roots are under `model/` unless `DACON_MODEL_ROOT` is set.

- Stage 1: `model/stage1/best.pt`
- Stage 2: `model/stage2/best.pt`
- Stage 3: `model/stage3/best.pt`

Stage 3 checkpoints store MViT weights under `model` plus sampling metadata: `sampling_hz`, `num_frames`, `past_frames`, `future_frames`, `clip_duration_sec`, `sampling_policy`, `boundary_policy`, `image_size`, `arch`, and `dataset_mode`.

## Submission

```bash
python -m src.tools.build_submission
```

Submission inference lives in `submission/inference.py` and loads the three stage checkpoints from `submission/model/stage*/best.pt`.

## Stage 3 Sampling

Stage 3 training and cache generation use timestamp-nearest sampling aligned to the submission cadence:

- `sampling_hz = 10.0`
- `num_frames = 16`
- offsets: `[-8, -7, ..., 6, 7]`
- coverage: `1.5 sec`
- center: manifest `target_timestamp`
- frame choice: nearest source frame timestamp
- boundary policy: `clamp`
- image size: `224`

Older Stage 3 checkpoints trained with consecutive source frames are legacy-sampling checkpoints and should be retrained for this sampler.
