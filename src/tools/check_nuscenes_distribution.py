from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.nuscenes import NuScenes


DATA_ROOT = Path(r"..\..\data\raw\stage3\nuscenes")
VERSION = "v1.0-mini"

# nuScenes load
nusc = NuScenes(
    version=VERSION,
    dataroot=str(DATA_ROOT),
    verbose=True,
)

nusc_can = NuScenesCanBus(
    dataroot=str(DATA_ROOT),
)


def get_cam_front_frames(scene):
    """
    한 scene의 CAM_FRONT 전체 프레임을 반환합니다.
    samples + sweeps 모두 포함됩니다.
    """
    sample = nusc.get(
        "sample",
        scene["first_sample_token"],
    )

    token = sample["data"]["CAM_FRONT"]

    frames = []

    while token:
        sd = nusc.get("sample_data", token)

        frames.append(
            {
                "timestamp": sd["timestamp"],
                "filename": sd["filename"],
                "is_key_frame": sd["is_key_frame"],
            }
        )

        token = sd["next"]

    return frames


def interpolate_scene(scene):
    scene_name = scene["name"]

    frames = get_cam_front_frames(scene)

    # CAN
    try:
        pose = nusc_can.get_messages(
            scene_name,
            "pose",
        )

        steer = nusc_can.get_messages(
            scene_name,
            "steeranglefeedback",
        )

    except Exception as e:
        print(
            f"[SKIP] {scene_name}: "
            f"CAN loading failed: {e}"
        )
        return []

    if len(pose) == 0 or len(steer) == 0:
        print(
            f"[SKIP] {scene_name}: "
            f"pose={len(pose)}, steer={len(steer)}"
        )
        return []

    # --------------------------------
    # CAN arrays
    # --------------------------------

    pose_ts = np.asarray(
        [x["utime"] for x in pose],
        dtype=np.float64,
    )

    speed = np.asarray(
        [x["vel"][0] for x in pose],
        dtype=np.float64,
    )

    accel = np.asarray(
        [x["accel"][0] for x in pose],
        dtype=np.float64,
    )

    yaw_rate = np.asarray(
        [x["rotation_rate"][2] for x in pose],
        dtype=np.float64,
    )

    steer_ts = np.asarray(
        [x["utime"] for x in steer],
        dtype=np.float64,
    )

    steering = np.asarray(
        [x["value"] for x in steer],
        dtype=np.float64,
    )

    # --------------------------------
    # CAN이 실제로 존재하는 공통 timestamp 범위
    # --------------------------------

    valid_start = max(
        pose_ts[0],
        steer_ts[0],
    )

    valid_end = min(
        pose_ts[-1],
        steer_ts[-1],
    )

    rows = []

    for frame_idx, frame in enumerate(frames):
        t = frame["timestamp"]

        # extrapolation 방지
        if t < valid_start or t > valid_end:
            continue

        speed_t = np.interp(
            t,
            pose_ts,
            speed,
        )

        accel_t = np.interp(
            t,
            pose_ts,
            accel,
        )

        yaw_rate_t = np.interp(
            t,
            pose_ts,
            yaw_rate,
        )

        steering_t = np.interp(
            t,
            steer_ts,
            steering,
        )

        rows.append(
            {
                "scene": scene_name,
                "description": scene["description"],
                "frame_idx": frame_idx,
                "timestamp": int(t),
                "frame_path": str(
                    DATA_ROOT / frame["filename"]
                ),
                "is_key_frame": frame["is_key_frame"],
                "speed": speed_t,
                "accel": accel_t,
                "steering": steering_t,
                "yaw_rate": yaw_rate_t,
            }
        )

    print(
        f"{scene_name}: "
        f"CAM={len(frames)}, "
        f"valid={len(rows)}, "
        f"pose={len(pose)}, "
        f"steer={len(steer)}"
    )

    return rows


# ====================================
# Build dataframe
# ====================================

all_rows = []

for scene in nusc.scene:
    rows = interpolate_scene(scene)
    all_rows.extend(rows)

df = pd.DataFrame(all_rows)

print()
print("================================")
print("nuScenes Mini Summary")
print("================================")

print(f"Scenes          : {df['scene'].nunique()}")
print(f"Frames          : {len(df)}")

print()
print("Speed")
print(df["speed"].describe())

print()
print("Acceleration")
print(df["accel"].describe())

print()
print("Steering")
print(df["steering"].describe())

print()
print("Yaw rate")
print(df["yaw_rate"].describe())


# ====================================
# Useful motion statistics
# ====================================

print()
print("================================")
print("Motion Statistics")
print("================================")

print(
    "Stopped (<0.5 m/s):",
    (df["speed"].abs() < 0.5).mean(),
)

print(
    "Low speed (<5 m/s):",
    (df["speed"].abs() < 5.0).mean(),
)

print(
    "Accel > 0.5:",
    (df["accel"] > 0.5).mean(),
)

print(
    "Brake < -0.5:",
    (df["accel"] < -0.5).mean(),
)

print(
    "|steering| > 0.1:",
    (df["steering"].abs() > 0.1).mean(),
)

print(
    "|yaw_rate| > 0.1:",
    (df["yaw_rate"].abs() > 0.1).mean(),
)


# ====================================
# Scene-level summary
# ====================================

scene_summary = (
    df.groupby("scene")
    .agg(
        frames=("frame_idx", "count"),
        speed_mean=("speed", "mean"),
        speed_max=("speed", "max"),
        accel_std=("accel", "std"),
        steering_std=("steering", "std"),
        steering_abs_max=(
            "steering",
            lambda x: x.abs().max(),
        ),
        yaw_rate_abs_max=(
            "yaw_rate",
            lambda x: x.abs().max(),
        ),
    )
)

print()
print("================================")
print("Scene Summary")
print("================================")

print(scene_summary)


# ====================================
# Save manifest
# ====================================

OUTPUT_DIR = Path(
    "./data/processed/stage3/nuscenes"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

csv_path = OUTPUT_DIR / "nuscenes_mini_manifest.csv"

df.to_csv(
    csv_path,
    index=False,
)

print()
print(f"Manifest saved: {csv_path}")


# ====================================
# Histograms
# ====================================

plot_columns = [
    ("speed", "Speed [m/s]"),
    ("accel", "Longitudinal acceleration [m/s²]"),
    ("steering", "Steering"),
    ("yaw_rate", "Yaw rate [rad/s]"),
]

for column, title in plot_columns:
    plt.figure(figsize=(8, 5))

    plt.hist(
        df[column].dropna(),
        bins=60,
    )

    plt.title(title)
    plt.xlabel(title)
    plt.ylabel("Frame count")

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR /
        f"{column}_hist.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.show()


# ====================================
# Time-series per scene
# ====================================

for scene_name, scene_df in df.groupby("scene"):
    scene_df = scene_df.sort_values(
        "timestamp"
    ).copy()

    t0 = scene_df["timestamp"].iloc[0]

    # microseconds → seconds
    scene_df["time_sec"] = (
        scene_df["timestamp"] - t0
    ) / 1_000_000

    # steering
    plt.figure(figsize=(10, 4))

    plt.plot(
        scene_df["time_sec"],
        scene_df["steering"],
    )

    plt.xlabel("Time [s]")
    plt.ylabel("Steering")
    plt.title(
        f"{scene_name} - Steering"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR /
        f"{scene_name}_steering.png",
        dpi=150,
    )

    plt.close()

    # acceleration
    plt.figure(figsize=(10, 4))

    plt.plot(
        scene_df["time_sec"],
        scene_df["accel"],
    )

    plt.xlabel("Time [s]")
    plt.ylabel("Acceleration [m/s²]")
    plt.title(
        f"{scene_name} - Acceleration"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR /
        f"{scene_name}_accel.png",
        dpi=150,
    )

    plt.close()


print()
print("Done.")

scene_stats = (
    df.groupby("scene")
    .apply(
        lambda x: pd.Series({
            "speed_mean": x["speed"].mean(),
            "steer_mean": x["steering"].mean(),
            "steer_std": x["steering"].std(),
            "yaw_mean": x["yaw_rate"].mean(),
            "yaw_std": x["yaw_rate"].std(),
            "steer_yaw_corr":
                x["steering"].corr(x["yaw_rate"]),
        })
    )
)

print(scene_stats.round(4))


stopped = df[df["speed"].abs() < 0.5]

print(
    stopped.groupby("scene")[
        ["steering", "yaw_rate"]
    ].agg(["mean", "std"])
)