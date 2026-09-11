from __future__ import annotations

import numpy as np

from src.config import (
    STAGE3_ACCEL_LABEL_MODE,
    STAGE3_ACCEL_THRESHOLD,
    STAGE3_ACCEL_WINDOW_SECONDS,
    STAGE3_DECEL_THRESHOLD,
    STAGE3_OUTPUT_HZ,
    STAGE3_STEER_POSITIVE_IS,
    STAGE3_STEER_THRESHOLD_DEG,
    STAGE3_STOP_SPEED_THRESHOLD,
)

ACCEL_TO_ID = {"ACCELERATING": 0, "DECELERATING": 1, "CONSTANT": 2, "STOPPED": 3}
STEER_TO_ID = {"LEFT": 0, "STRAIGHT": 1, "RIGHT": 2}
ACCEL_NAMES = {v: k for k, v in ACCEL_TO_ID.items()}
STEER_NAMES = {v: k for k, v in STEER_TO_ID.items()}


def _smooth(values: np.ndarray, width: int = 5) -> np.ndarray:
    values = np.asarray(values, dtype=float).squeeze()
    if len(values) < width:
        return values
    return np.convolve(values, np.ones(width, dtype=float) / width, mode="same")


def derive_acceleration_current(speed: np.ndarray, hz: float = STAGE3_OUTPUT_HZ, smooth_width: int = 5) -> np.ndarray:
    return np.gradient(_smooth(speed, smooth_width), 1.0 / hz)


def derive_acceleration_window_regression(speed: np.ndarray, hz: float = STAGE3_OUTPUT_HZ, half_window_seconds: float = STAGE3_ACCEL_WINDOW_SECONDS) -> np.ndarray:
    speed = np.asarray(speed, dtype=float).squeeze()
    times = np.arange(len(speed), dtype=float) / hz
    out = np.zeros(len(speed), dtype=float)
    for i, t in enumerate(times):
        mask = np.abs(times - t) <= half_window_seconds
        out[i] = float(np.polyfit(times[mask] - t, speed[mask], 1)[0]) if mask.sum() >= 2 else 0.0
    return out


def derive_acceleration(speed: np.ndarray, mode: str = STAGE3_ACCEL_LABEL_MODE) -> np.ndarray:
    if mode == "current":
        return derive_acceleration_current(speed)
    if mode == "window_regression":
        return derive_acceleration_window_regression(speed)
    raise ValueError(f"unknown STAGE3_ACCEL_LABEL_MODE: {mode}")


def derive_accel_label(
    speed: np.ndarray,
    acceleration: np.ndarray,
    accel_threshold: float = STAGE3_ACCEL_THRESHOLD,
    decel_threshold: float = STAGE3_DECEL_THRESHOLD,
    stopped_speed_threshold: float = STAGE3_STOP_SPEED_THRESHOLD,
) -> np.ndarray:
    speed = np.asarray(speed, dtype=float).squeeze()
    acceleration = np.asarray(acceleration, dtype=float).squeeze()
    label = np.full(len(speed), ACCEL_TO_ID["CONSTANT"], dtype=np.int64)
    label[acceleration > accel_threshold] = ACCEL_TO_ID["ACCELERATING"]
    label[acceleration < -decel_threshold] = ACCEL_TO_ID["DECELERATING"]
    label[speed < stopped_speed_threshold] = ACCEL_TO_ID["STOPPED"]
    return label


def derive_steer_label(
    steering_angle: np.ndarray,
    invert_steering: bool = False,
    threshold_deg: float = STAGE3_STEER_THRESHOLD_DEG,
) -> np.ndarray:
    steering = np.asarray(steering_angle, dtype=float).squeeze()
    steering = -steering if invert_steering else steering
    label = np.full(len(steering), STEER_TO_ID["STRAIGHT"], dtype=np.int64)
    if STAGE3_STEER_POSITIVE_IS == "LEFT":
        label[steering > threshold_deg] = STEER_TO_ID["LEFT"]
        label[steering < -threshold_deg] = STEER_TO_ID["RIGHT"]
    elif STAGE3_STEER_POSITIVE_IS == "RIGHT":
        label[steering > threshold_deg] = STEER_TO_ID["RIGHT"]
        label[steering < -threshold_deg] = STEER_TO_ID["LEFT"]
    else:
        raise ValueError(f"unknown STAGE3_STEER_POSITIVE_IS: {STAGE3_STEER_POSITIVE_IS}")
    return label