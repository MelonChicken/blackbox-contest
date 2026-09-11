# Stage 3 Data Pipeline

Canonical Stage 3 data flow is comma2k19-only:

```text
raw comma2k19
-> manifest
-> labels
-> TartanVO cache
-> dataset
-> train / inference
```

## Raw Data

Expected root:

```text
data/raw/comma2k19/
```

Local comma2k19 segments are discovered under `Chunk_*/*/*/video.hevc`.

## Manifest Build

Canonical builder:

```bash
python -m src.tools.audit_stage3_comma_sync
python -m src.tools.build_comma2k19_stage3_manifest
python -m src.tools.check_stage3_comma_alignment
```

The manifest builder aligns absolute CAN target timestamps to decoded frame indices using `global_pose/frame_times` as the synchronization clock. The `frame_times` length must match the decoded frame count; HEVC PTS is decoded only as diagnostic metadata and is not used as the canonical clock. Targets outside the `frame_times` range are dropped, not clamped.

Outputs:

```text
data/processed/stage3/manifest/train.csv
data/processed/stage3/manifest/val.csv
data/processed/stage3/manifest/metadata.json  # planned summary location
```

Canonical schema:

| column | meaning |
|---|---|
| `route_id` | comma2k19 route id |
| `segment_id` | route segment id |
| `video_path` | path to source video, relative to raw root when possible |
| `sample_index` | 10 Hz sample index relative to the aligned video start |
| `target_timestamp` | target CAN/video-clock timestamp used for labels and frame lookup |
| `video_frame_index` | selected decoded video frame index nearest to `target_timestamp` |
| `video_frame_timestamp` | timestamp of selected video frame in the synchronization clock |
| `alignment_error_sec` | `abs(video_frame_timestamp - target_timestamp)` |
| `alignment_version` | manifest alignment version, currently `comma_frame_times_nearest_v1` |
| `speed` | speed interpolated at `target_timestamp` |
| `steering_angle` | steering angle interpolated at `target_timestamp` |
| `acceleration` | derived acceleration at `target_timestamp` |
| `accel_label` | acceleration class id |
| `steer_label` | steering class id |
| `source` | dataset source, currently `comma2k19` |
| `split` | `train` or `val` |

Legacy manifests with `frame_index` are read as a fallback, but new manifests write `video_frame_index` only.

## Labels

Canonical label module:

```text
src/datasets/stage3_labels.py
```

Public API:

```python
derive_acceleration_current(...)
derive_acceleration_window_regression(...)
derive_acceleration(...)
derive_accel_label(...)
derive_steer_label(...)
```

Manifest build, label audit, and datasets should use this module rather than duplicating thresholds or label rules.

Label audit:

```bash
python -m src.tools.audit_stage3_comma_labels
```

## TartanVO Cache

Canonical cache builder:

```bash
python -m src.tools.cache_stage3_tartanvo_feature --split train --feature latent
python -m src.tools.cache_stage3_tartanvo_feature --split val --feature latent
```

The cache builder reads the canonical manifest frame fields and records `manifest_alignment_version` in cache metadata. comma2k19 caches without `manifest_alignment_version=comma_frame_times_nearest_v1` are treated as invalid and must be regenerated.

Target structure:

```text
data/processed/stage3/tartanvo_features/
  latent/
    comma2k19/
      train_index.csv
      val_index.csv
      train_metadata.json
      val_metadata.json
```

## Dataset And Training

Dataset smoke tests:

```bash
python -m src.tools.smoke_stage3_comma_dataset
python -m src.tools.smoke_stage3_tartanvo_finetune
```

Training:

```bash
python train.py
```

Current default config is `STAGE3_DATASET_MODE = "comma_only"`.

## Processed Directory

Target processed layout:

```text
data/processed/stage3/
  manifest/
    train.csv
    val.csv
    metadata.json
  tartanvo_features/
    latent/
      comma2k19/
        train_index.csv
        val_index.csv
        train_metadata.json
        val_metadata.json
  audits/
    alignment.csv
    label_audit.csv
```

Long cache regeneration and long training should only run after sync, manifest, and label audits pass.