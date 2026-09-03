# comma2k19 Stage 3 manifest

Expected local raw layout:

- `/data/raw/comma2k19/Chunk_*/*/*/video.hevc`
- `/data/raw/comma2k19/Chunk_*/*/*/processed_log/CAN/speed/{t,value}` or `car_speed/{t,value}`
- `/data/raw/comma2k19/Chunk_*/*/*/processed_log/CAN/steering_angle/{t,value}`
- `/data/raw/comma2k19/Chunk_*/*/*/global_pose/frame_times`

Each directory matched by `Chunk_*/*/*` is treated as one driving segment. Missing video/state files produce a warning and the segment is skipped.

HuggingFace path is also supported:

`python -m src.tools.build_comma2k19_stage3_manifest --hf-split demo`

`processed_log` timestamps and `global_pose__frame_times` are boot-time seconds in comma2k19. If local `frame_times` is missing, the builder falls back to OpenCV FPS and segment-relative video time, then shifts CAN timestamps to the same relative start.

Speed is linearly interpolated to 10 Hz. Acceleration is `np.gradient` over a 5-sample moving average. Steering angle uses comma2k19 CAN `steering_angle` in degrees. Default mapping treats positive steering angle as LEFT; pass `--invert-steering` if your visual check shows the opposite sign.
