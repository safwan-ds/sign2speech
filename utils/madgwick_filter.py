"""Madgwick orientation filter for IMU sensor data.

Implements the Madgwick filter for estimating orientation (quaternion) from
accelerometer and gyroscope readings, as well as helper functions for
quaternion arithmetic and batch processing of sensor sequences.
"""

from dataclasses import dataclass
from dataclasses import field

import numpy as np

from utils.normalization import get_feature_names


def _normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return (quaternion / norm).astype(np.float32)


def _quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = left
    w2, x2, y2, z2 = right
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float32,
    )


@dataclass
class MadgwickFilter:
    """IMU-only Madgwick orientation filter that emits [w, x, y, z]."""

    beta: float = 0.1
    sample_period: float = 1.0 / 100.0
    quaternion: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    )

    def update_imu(
        self,
        accel: np.ndarray,
        gyro: np.ndarray,
        *,
        gyro_degrees: bool = False,
    ) -> np.ndarray:
        accel = np.asarray(accel, dtype=np.float32)
        gyro = np.asarray(gyro, dtype=np.float32)
        if accel.shape != (3,) or gyro.shape != (3,):
            raise ValueError("accel and gyro must be 3-element vectors")

        if gyro_degrees:
            gyro = np.deg2rad(gyro).astype(np.float32)

        accel_norm = float(np.linalg.norm(accel))
        if accel_norm <= 1e-12:
            return self.quaternion.copy()
        accel = accel / accel_norm

        q1, q2, q3, q4 = self.quaternion
        ax, ay, az = accel

        objective = np.array(
            [
                2.0 * (q2 * q4 - q1 * q3) - ax,
                2.0 * (q1 * q2 + q3 * q4) - ay,
                2.0 * (0.5 - q2 * q2 - q3 * q3) - az,
            ],
            dtype=np.float32,
        )
        jacobian = np.array(
            [
                [-2.0 * q3, 2.0 * q4, -2.0 * q1, 2.0 * q2],
                [2.0 * q2, 2.0 * q1, 2.0 * q4, 2.0 * q3],
                [0.0, -4.0 * q2, -4.0 * q3, 0.0],
            ],
            dtype=np.float32,
        )
        step = jacobian.T @ objective
        step_norm = float(np.linalg.norm(step))
        if step_norm > 1e-12:
            step = step / step_norm

        q_dot = 0.5 * _quaternion_multiply(
            self.quaternion,
            np.array([0.0, gyro[0], gyro[1], gyro[2]], dtype=np.float32),
        ) - self.beta * step

        self.quaternion = _normalize_quaternion(
            self.quaternion + q_dot * float(self.sample_period)
        )
        return self.quaternion.copy()


def compute_madgwick_quaternions(
    sequence: np.ndarray,
    *,
    feature_names: list[str] | None = None,
    beta: float = 0.1,
    sample_rate_hz: float = 100.0,
    gyro_degrees: bool = False,
) -> np.ndarray:
    """Convert raw accel/gyro columns from a sequence into quaternion features."""
    arr = np.asarray(sequence, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("sequence must be a 2D array")

    names = feature_names or get_feature_names()
    name_to_idx = {name: idx for idx, name in enumerate(names)}
    required = ["accelX", "accelY", "accelZ", "gyroX", "gyroY", "gyroZ"]
    missing = [name for name in required if name not in name_to_idx]
    if missing:
        raise ValueError("Missing IMU feature(s): " + ", ".join(missing))

    sample_period = 1.0 / max(float(sample_rate_hz), 1e-6)
    filter_state = MadgwickFilter(beta=beta, sample_period=sample_period)
    quaternions = np.empty((len(arr), 4), dtype=np.float32)

    for idx, row in enumerate(arr):
        accel = np.array([row[name_to_idx[name]] for name in required[:3]], dtype=np.float32)
        gyro = np.array([row[name_to_idx[name]] for name in required[3:]], dtype=np.float32)
        quaternions[idx] = filter_state.update_imu(
            accel,
            gyro,
            gyro_degrees=gyro_degrees,
        )

    return quaternions.astype(np.float32)
