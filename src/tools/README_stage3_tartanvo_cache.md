# Stage3 TartanVO Pose Cache

This cache is training-only. Submission keeps the online `tartanvo_gru` path and loads the full `best.pt` state_dict.

Run on the data server after comma2k19 manifests and videos are available:

```powershell
uv run python -m src.tools.cache_stage3_tartanvo_pose --split train
uv run python -m src.tools.cache_stage3_tartanvo_pose --split val
uv run python train.py
```

Required config values in `src/config.py`:

```python
STAGE3_ARCH = "tartanvo_gru"
STAGE3_TARTANVO_USE_FEATURE_CACHE = True
STAGE3_TARTANVO_FEATURE = "pose"
STAGE3_TARTANVO_POSE_NORM = "none"
```

Cache layout:

```text
data/processed/stage3/tartanvo_pose_features/
    train/<sample_key>.pt
    val/<sample_key>.pt
    train_index.csv
    val_index.csv
```

Each `.pt` stores `pose [15,6]`, labels, `sample_key`, `video_path`, and `frame_index`.