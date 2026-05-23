"""
Data processing and normalization utilities for Sign2Speech project
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from config.config import (
    FLEX_SENSOR_RANGES,
    FLEX_SENSOR_DEFAULT_RANGE,
    MIN_ACCEL_VALUE,
    MAX_ACCEL_VALUE,
    MIN_GYRO_VALUE,
    MAX_GYRO_VALUE,
    NORM_MIN,
    NORM_MAX,
    NUM_FLEX_SENSORS,
    DETECT_GESTURE_MOTION,
    MOTION_THRESHOLD,
    MOTION_DETECTION_MIN_DURATION,
    MOTION_DETECTION_SMOOTHING_WINDOW,
    SEQUENCE_OVERLAP,
    MOTION_PADDING_RATIO,
)

logger = logging.getLogger(__name__)

QUATERNION_FEATURE_NAMES = ["quatW", "quatX", "quatY", "quatZ"]


def convert_to_snake_case(label: str) -> str:
    """Convert gesture label to snake case (spaces to underscores)."""
    return label.replace(" ", "_")


def normalize_value(name: str, value: float) -> float | None:
    """
    Normalize sensor value to configured range (default 0.0 - 1.0)

    Args:
        name: Sensor variable name (e.g., 'flex0', 'accelX', 'gyroY')
        value: Raw sensor value

    Returns:
        float | None: Normalized value in range [NORM_MIN, NORM_MAX], or None if sensor name is unknown
    """
    if name.startswith("flex"):
        # Per-sensor normalization for flex sensors
        sensor_idx = int(name[4:])  # Extract index from "flexN"
        min_val, max_val = FLEX_SENSOR_RANGES.get(sensor_idx, FLEX_SENSOR_DEFAULT_RANGE)
        value = max(min_val, min(max_val, value))  # Clip
        normalized = (value - min_val) / (max_val - min_val)
    elif name.startswith("accel"):
        # Accelerometer normalization
        value = max(MIN_ACCEL_VALUE, min(MAX_ACCEL_VALUE, value))  # Clip
        normalized = (value - MIN_ACCEL_VALUE) / (MAX_ACCEL_VALUE - MIN_ACCEL_VALUE)
    elif name.startswith("gyro"):
        # Gyroscope normalization
        value = max(MIN_GYRO_VALUE, min(MAX_GYRO_VALUE, value))  # Clip
        normalized = (value - MIN_GYRO_VALUE) / (MAX_GYRO_VALUE - MIN_GYRO_VALUE)
    else:
        return None

    # Scale to NORM_MIN - NORM_MAX range
    return normalized * (NORM_MAX - NORM_MIN) + NORM_MIN


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize all sensor data in a DataFrame and create normalized columns

    Args:
        df: pandas DataFrame with sensor columns

    Returns:
        pandas DataFrame: DataFrame with additional normalized columns
    """
    df = df.copy()

    # Normalize flex sensors
    for i in range(NUM_FLEX_SENSORS):
        raw_col = f"flex{i}" if f"flex{i}" in df.columns else f"sensor{i}"
        norm_col = f"flex{i}_norm"

        if raw_col in df.columns:
            min_val, max_val = FLEX_SENSOR_RANGES.get(i, (0, 1023))
            # Clip values to range, then normalize to 0-1
            df[norm_col] = df[raw_col].clip(min_val, max_val)
            df[norm_col] = (df[norm_col] - min_val) / (max_val - min_val)
            df[norm_col] = df[norm_col] * (NORM_MAX - NORM_MIN) + NORM_MIN

    # Normalize accelerometer data
    for axis in ["X", "Y", "Z"]:
        raw_col = f"accel{axis}"
        norm_col = f"accel{axis}_norm"

        if raw_col in df.columns:
            df[norm_col] = df[raw_col].clip(MIN_ACCEL_VALUE, MAX_ACCEL_VALUE)
            df[norm_col] = (df[norm_col] - MIN_ACCEL_VALUE) / (
                MAX_ACCEL_VALUE - MIN_ACCEL_VALUE
            )
            df[norm_col] = df[norm_col] * (NORM_MAX - NORM_MIN) + NORM_MIN

    # Normalize gyroscope data
    for axis in ["X", "Y", "Z"]:
        raw_col = f"gyro{axis}"
        norm_col = f"gyro{axis}_norm"

        if raw_col in df.columns:
            df[norm_col] = df[raw_col].clip(MIN_GYRO_VALUE, MAX_GYRO_VALUE)
            df[norm_col] = (df[norm_col] - MIN_GYRO_VALUE) / (
                MAX_GYRO_VALUE - MIN_GYRO_VALUE
            )
            df[norm_col] = df[norm_col] * (NORM_MAX - NORM_MIN) + NORM_MIN

    return df


def extract_normalized_features(df: pd.DataFrame):
    """
    Extract normalized sensor values as a feature array

    Args:
        df: pandas DataFrame with normalized sensor columns

    Returns:
        numpy.ndarray: Feature array of shape (n_samples, n_features)
    """
    expected_cols: list[str] = []

    # Flex sensors
    for i in range(NUM_FLEX_SENSORS):
        expected_cols.append(f"flex{i}_norm")

    # IMU data
    expected_cols.extend(
        [
            "accelX_norm",
            "accelY_norm",
            "accelZ_norm",
            "gyroX_norm",
            "gyroY_norm",
            "gyroZ_norm",
        ]
    )

    missing_cols = [col for col in expected_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            "Missing expected normalized columns: " + ", ".join(missing_cols)
        )

    feature_cols = expected_cols

    # Extract values as numpy array
    features = df[feature_cols].values.astype(np.float32)
    return features


def detect_motion_boundaries(
    sequence: np.ndarray,
    threshold: float = MOTION_THRESHOLD,
    min_duration: int = MOTION_DETECTION_MIN_DURATION,
):
    """
    Detect start and end of gesture motion based on sensor activity

    Args:
        sequence: numpy array of shape (n_samples, n_features)
        threshold: Motion threshold (std dev of normalized features)
        min_duration: Minimum samples required to consider as motion

    Returns:
        tuple: (start_idx, end_idx) of active gesture region, or None if no motion
    """
    # Calculate motion energy as sum of absolute differences
    motion = np.abs(np.diff(sequence, axis=0))
    motion_energy = np.mean(motion, axis=1)

    # Smooth motion signal
    window = MOTION_DETECTION_SMOOTHING_WINDOW
    if len(motion_energy) >= window:
        motion_smooth = np.convolve(
            motion_energy, np.ones(window) / window, mode="same"
        )
    else:
        motion_smooth = motion_energy

    # Find samples above threshold
    active = motion_smooth > threshold

    if not np.any(active):
        return None

    # Find continuous active regions
    changes = np.diff(active.astype(int))
    starts = np.where(changes == 1)[0] + 1
    ends = np.where(changes == -1)[0] + 1

    if active[0]:
        starts = np.insert(starts, 0, 0)
    if active[-1]:
        ends = np.append(ends, len(active))

    # Filter regions by minimum duration
    valid_regions: list[tuple[int, int]] = []
    for start, end in zip(starts, ends):
        if end - start >= min_duration:
            valid_regions.append((int(start), int(end)))

    if not valid_regions:
        return None

    # Return the largest/longest motion region
    longest_region = max(valid_regions, key=lambda x: x[1] - x[0])
    return longest_region


def segment_sequences(
    df: pd.DataFrame,
    sequence_length: int,
    overlap: float = SEQUENCE_OVERLAP,
    detect_motion: bool = DETECT_GESTURE_MOTION,
    motion_threshold: float = MOTION_THRESHOLD,
) -> list[np.ndarray]:
    """
    Segment continuous sensor data into overlapping sequences for LSTM

    Args:
        df: pandas DataFrame with normalized sensor data
        sequence_length: Number of timesteps per sequence
        overlap: Overlap ratio between consecutive sequences (0-1)
        detect_motion: If True, only segment from active gesture regions

    Returns:
        list: List of numpy arrays, each of shape (sequence_length, num_features)
    """
    # First normalize if not already done
    if f"flex0_norm" not in df.columns:
        df = normalize_dataframe(df)

    sequence = extract_normalized_features(df)

    if len(sequence) < sequence_length:
        logger.warning(f"Sequence too short ({len(sequence)} < {sequence_length})")
        return []

    sequences: list[np.ndarray] = []
    step_size = int(sequence_length * (1 - overlap))

    # Detect active gesture region if requested
    if detect_motion:
        motion_region = detect_motion_boundaries(sequence, threshold=motion_threshold)
        if motion_region:
            start_idx, end_idx = motion_region
            # Add some padding around detected motion
            padding = int(sequence_length * MOTION_PADDING_RATIO)
            start_idx = max(0, start_idx - padding)
            end_idx = min(len(sequence), end_idx + padding)
            trimmed = sequence[start_idx:end_idx]
            # Fall back to full sequence if trimming made it too short
            if len(trimmed) >= sequence_length:
                sequence = trimmed
            else:
                logger.warning(
                    f"Motion-trimmed region too short "
                    f"({len(trimmed)} < {sequence_length}), using full sequence"
                )
        # If no motion detected, still continue with full sequence but log it
        else:
            logger.warning("No clear gesture motion detected")

    for i in range(0, len(sequence) - sequence_length + 1, step_size):
        seq = sequence[i : i + sequence_length]
        sequences.append(seq)

    return sequences


def get_feature_names():
    """
    Get list of feature names in the expected order

    Returns:
        list: List of feature names
    """
    features: list[str] = []

    # Flex sensors
    for i in range(NUM_FLEX_SENSORS):
        features.append(f"flex{i}")

    # IMU sensors
    features.extend(
        [
            "accelX",
            "accelY",
            "accelZ",
            "gyroX",
            "gyroY",
            "gyroZ",
        ]
    )

    return features


def compute_velocity(sequence: np.ndarray) -> np.ndarray:
    """
    Compute first-order derivatives (velocity) of sensor values

    Args:
        sequence: numpy array of shape (n_samples, n_features)

    Returns:
        velocity: numpy array of shape (n_samples, n_features)
    """
    # Pad the beginning with zeros to maintain sequence length
    velocity = np.diff(sequence, axis=0, prepend=sequence[0:1])
    return velocity.astype(np.float32)


def compute_acceleration(sequence: np.ndarray) -> np.ndarray:
    """
    Compute second-order derivatives (acceleration) of sensor values

    Args:
        sequence: numpy array of shape (n_samples, n_features)

    Returns:
        acceleration: numpy array of shape (n_samples, n_features)
    """
    velocity = compute_velocity(sequence)
    acceleration = np.diff(velocity, axis=0, prepend=velocity[0:1])
    return acceleration.astype(np.float32)


def compute_rolling_statistics(sequence: np.ndarray, window_size: int = 5):
    """
    Compute rolling window statistics (mean, std, min, max)

    Args:
        sequence: numpy array of shape (n_samples, n_features)
        window_size: Size of rolling window

    Returns:
        stats: Dictionary containing rolling statistics
    """
    n_samples, n_features = sequence.shape
    stats: dict[str, np.ndarray] = {}

    # Vectorized rolling window using numpy stride tricks
    from numpy.lib.stride_tricks import sliding_window_view

    # Pad the beginning to maintain output size
    pad = np.repeat(sequence[:1], window_size - 1, axis=0)
    padded = np.concatenate([pad, sequence], axis=0)

    # Create rolling windows: (n_samples, n_features, window_size)
    windows = sliding_window_view(padded, window_size, axis=0)

    stats["mean"] = np.mean(windows, axis=2).astype(np.float32)
    stats["std"] = np.std(windows, axis=2).astype(np.float32)
    stats["min"] = np.min(windows, axis=2).astype(np.float32)
    stats["max"] = np.max(windows, axis=2).astype(np.float32)

    return stats


def _euclidean_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Distance function shared by fastdtw and the local fallback."""
    return float(np.linalg.norm(np.asarray(left, dtype=float) - np.asarray(right, dtype=float)))


def _exact_dtw_path(
    sequence: np.ndarray,
    template: np.ndarray,
) -> tuple[float, list[tuple[int, int]]]:
    """Small, dependency-free DTW fallback for short inference sequences."""
    n_seq, n_template = len(sequence), len(template)
    if n_seq == 0 or n_template == 0:
        return float("inf"), []

    costs = np.full((n_seq + 1, n_template + 1), np.inf, dtype=np.float64)
    costs[0, 0] = 0.0
    backtrack: dict[tuple[int, int], tuple[int, int]] = {}

    for i in range(1, n_seq + 1):
        for j in range(1, n_template + 1):
            candidates = (
                (costs[i - 1, j], (i - 1, j)),
                (costs[i, j - 1], (i, j - 1)),
                (costs[i - 1, j - 1], (i - 1, j - 1)),
            )
            best_cost, best_prev = min(candidates, key=lambda item: item[0])
            costs[i, j] = (
                best_cost + _euclidean_distance(sequence[i - 1], template[j - 1])
            )
            backtrack[(i, j)] = best_prev

    path: list[tuple[int, int]] = []
    cursor = (n_seq, n_template)
    while cursor != (0, 0):
        i, j = cursor
        if i > 0 and j > 0:
            path.append((i - 1, j - 1))
        cursor = backtrack.get(cursor, (0, 0))
    path.reverse()
    return float(costs[n_seq, n_template]), path


def _dtw_path(
    sequence: np.ndarray,
    template: np.ndarray,
    radius: int = 5,
) -> tuple[float, list[tuple[int, int]]]:
    """Return DTW distance/path, preferring fastdtw when available."""
    try:
        from fastdtw import fastdtw  # type: ignore

        distance, path = fastdtw(
            sequence,
            template,
            radius=max(1, int(radius)),
            dist=_euclidean_distance,
        )
        return float(distance), [(int(i), int(j)) for i, j in path]
    except Exception:
        return _exact_dtw_path(sequence, template)


def resample_sequence(sequence: np.ndarray, target_length: int) -> np.ndarray:
    """Linearly resample a sequence along time to a target length."""
    arr = np.asarray(sequence, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("sequence must be a 2D array")
    if target_length <= 0:
        raise ValueError("target_length must be positive")
    if len(arr) == target_length:
        return arr.astype(np.float32, copy=True)
    if len(arr) == 0:
        return np.zeros((target_length, arr.shape[1]), dtype=np.float32)
    if len(arr) == 1:
        return np.repeat(arr, target_length, axis=0).astype(np.float32)

    source_x = np.linspace(0.0, 1.0, len(arr), dtype=np.float32)
    target_x = np.linspace(0.0, 1.0, target_length, dtype=np.float32)
    columns = [
        np.interp(target_x, source_x, arr[:, feature_idx])
        for feature_idx in range(arr.shape[1])
    ]
    return np.stack(columns, axis=1).astype(np.float32)


def align_sequence_to_template(
    sequence: np.ndarray,
    template: np.ndarray,
    *,
    radius: int = 5,
    target_length: int | None = None,
) -> tuple[np.ndarray, float]:
    """DTW-align an incoming sequence to a class template timeline."""
    seq = np.asarray(sequence, dtype=np.float32)
    tmpl = np.asarray(template, dtype=np.float32)
    if seq.ndim != 2 or tmpl.ndim != 2:
        raise ValueError("sequence and template must be 2D arrays")
    if seq.shape[1] != tmpl.shape[1]:
        raise ValueError(
            f"Feature mismatch: sequence has {seq.shape[1]}, template has {tmpl.shape[1]}"
        )

    distance, path = _dtw_path(seq, tmpl, radius=radius)
    output_length = int(target_length or len(tmpl))
    if not path:
        return resample_sequence(seq, output_length), distance

    grouped: list[list[np.ndarray]] = [[] for _ in range(len(tmpl))]
    for seq_idx, tmpl_idx in path:
        if 0 <= seq_idx < len(seq) and 0 <= tmpl_idx < len(tmpl):
            grouped[tmpl_idx].append(seq[seq_idx])

    aligned = np.empty_like(tmpl, dtype=np.float32)
    for tmpl_idx, rows in enumerate(grouped):
        if rows:
            aligned[tmpl_idx] = np.mean(np.stack(rows, axis=0), axis=0)
        else:
            nearest_idx = int(round(tmpl_idx * (len(seq) - 1) / max(1, len(tmpl) - 1)))
            aligned[tmpl_idx] = seq[nearest_idx]

    if len(aligned) != output_length:
        aligned = resample_sequence(aligned, output_length)
    return aligned.astype(np.float32), distance


def align_sequence_to_templates(
    sequence: np.ndarray,
    templates: dict[str, np.ndarray],
    *,
    radius: int = 5,
    target_length: int | None = None,
) -> tuple[np.ndarray, str | None, float | None]:
    """Align a sequence to the nearest class template and return the aligned copy."""
    if not templates:
        return np.asarray(sequence, dtype=np.float32), None, None

    best_label: str | None = None
    best_distance = float("inf")
    best_aligned: np.ndarray | None = None

    for label, template in templates.items():
        tmpl = np.asarray(template, dtype=np.float32)
        if tmpl.ndim != 2 or tmpl.shape[1] != np.asarray(sequence).shape[1]:
            continue
        aligned, distance = align_sequence_to_template(
            sequence,
            tmpl,
            radius=radius,
            target_length=target_length,
        )
        if distance < best_distance:
            best_label = str(label)
            best_distance = float(distance)
            best_aligned = aligned

    if best_aligned is None:
        return np.asarray(sequence, dtype=np.float32), None, None
    return best_aligned, best_label, best_distance


def load_dtw_templates(path_or_dir: str | Path | None) -> dict[str, np.ndarray]:
    """Load class templates from ``dtw_templates.npz`` when present."""
    if path_or_dir is None:
        return {}

    path = Path(path_or_dir)
    if path.is_dir():
        path = path / "dtw_templates.npz"
    if not path.exists():
        return {}

    try:
        payload = np.load(path, allow_pickle=True)
        if "templates" in payload.files:
            templates = np.asarray(payload["templates"], dtype=np.float32)
            labels = None
            if "classes" in payload.files:
                labels = payload["classes"]
            elif "labels" in payload.files:
                labels = payload["labels"]
            if labels is None:
                labels = [f"class_{idx}" for idx in range(len(templates))]
            return {
                str(label): np.asarray(template, dtype=np.float32)
                for label, template in zip(labels, templates)
                if np.asarray(template).ndim == 2
            }
        return {
            str(key): np.asarray(payload[key], dtype=np.float32)
            for key in payload.files
            if np.asarray(payload[key]).ndim == 2
        }
    except Exception as exc:
        logger.warning("Could not load DTW templates from %s: %s", path, exc)
        return {}


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


def extract_enhanced_features(
    sequence: np.ndarray,
    include_derivatives: bool = True,
    include_stats: bool = True,
    stats_window: int = 5,
):
    """
    Extract enhanced features from raw sensor sequence

    Args:
        sequence: numpy array of shape (n_samples, n_features)
        include_derivatives: Include velocity and acceleration features
        include_stats: Include rolling statistics
        stats_window: Window size for rolling statistics

    Returns:
        enhanced_features: numpy array with all features combined
    """
    feature_list = [sequence]

    if include_derivatives:
        velocity = compute_velocity(sequence)
        acceleration = compute_acceleration(sequence)
        feature_list.extend([velocity, acceleration])

    if include_stats:
        stats = compute_rolling_statistics(sequence, window_size=stats_window)
        feature_list.extend([stats["mean"], stats["std"]])

    # Concatenate all features
    enhanced_features = np.concatenate(feature_list, axis=1)

    return enhanced_features.astype(np.float32)


def compute_magnitude(sequence: np.ndarray, feature_indices: list[int]) -> np.ndarray:
    """
    Compute magnitude of multi-axis sensors (e.g., accelerometer, gyroscope)

    Args:
        sequence: numpy array of shape (n_samples, n_features)
        feature_indices: List of indices for X, Y, Z axes

    Returns:
        magnitude: numpy array of shape (n_samples, 1)
    """
    if len(feature_indices) < 2:
        raise ValueError("Need at least 2 axes to compute magnitude")

    magnitude = np.sqrt(
        np.sum(sequence[:, feature_indices] ** 2, axis=1, keepdims=True)
    )
    return magnitude.astype(np.float32)


def extract_frequency_features(sequence: np.ndarray, sampling_rate: int = 100):
    """
    Extract frequency domain features using FFT

    Args:
        sequence: numpy array of shape (n_samples, n_features)
        sampling_rate: Sampling rate in Hz

    Returns:
        freq_features: Dictionary containing frequency features
    """
    from scipy.fft import fft, fftfreq  # type: ignore

    n_samples, n_features = sequence.shape
    freq_features: dict[str, float] = {}

    for i in range(n_features):
        # Compute FFT
        fft_vals = fft(sequence[:, i])
        fft_freq: np.ndarray = fftfreq(n_samples, 1 / sampling_rate)  # type: ignore

        # Only keep positive frequencies
        positive_freq_idx = fft_freq > 0  # type: ignore
        fft_vals_filtered = fft_vals[positive_freq_idx]  # type: ignore
        fft_vals_abs = np.abs(fft_vals_filtered)  # type: ignore
        fft_freq = fft_freq[positive_freq_idx]  # type: ignore

        # Compute dominant frequency
        dominant_freq_idx = np.argmax(fft_vals_abs)
        dominant_freq = fft_freq[dominant_freq_idx]  # type: ignore

        # Compute spectral energy
        spectral_energy = np.sum(fft_vals_abs**2)

        # Compute spectral entropy
        normalized_spectrum = fft_vals_abs / np.sum(fft_vals_abs)
        spectral_entropy = -np.sum(
            normalized_spectrum * np.log(normalized_spectrum + 1e-10)
        )

        freq_features[f"feature_{i}_dominant_freq"] = dominant_freq
        freq_features[f"feature_{i}_spectral_energy"] = spectral_energy
        freq_features[f"feature_{i}_spectral_entropy"] = spectral_entropy

    return freq_features


def segment_sequences_with_enhanced_features(
    df: pd.DataFrame,
    sequence_length: int,
    overlap: float = 0.5,
    detect_motion: bool = DETECT_GESTURE_MOTION,
    motion_threshold: float = MOTION_THRESHOLD,
    use_enhanced_features: bool = False,
    include_derivatives: bool = True,
    include_stats: bool = False,
) -> list[np.ndarray]:
    """
    Segment continuous sensor data with optional enhanced features

    Args:
        df: pandas DataFrame with normalized sensor data
        sequence_length: Number of timesteps per sequence
        overlap: Overlap ratio between consecutive sequences (0-1)
        detect_motion: If True, only segment from active gesture regions
        motion_threshold: Motion detection threshold
        use_enhanced_features: If True, add velocity, acceleration, stats
        include_derivatives: Include velocity and acceleration
        include_stats: Include rolling statistics

    Returns:
        list: List of numpy arrays with enhanced features
    """
    # First normalize if not already done
    if f"flex0_norm" not in df.columns:
        df = normalize_dataframe(df)

    sequence = extract_normalized_features(df)

    # Add enhanced features if requested
    if use_enhanced_features:
        sequence = extract_enhanced_features(
            sequence,
            include_derivatives=include_derivatives,
            include_stats=include_stats,
        )

    if len(sequence) < sequence_length:
        logger.warning(f"Sequence too short ({len(sequence)} < {sequence_length})")
        return []

    sequences: list[np.ndarray] = []
    step_size = int(sequence_length * (1 - overlap))

    # Detect active gesture region if requested
    if detect_motion:
        # Use only base features for motion detection
        base_sequence = extract_normalized_features(df)
        motion_region = detect_motion_boundaries(
            base_sequence, threshold=motion_threshold
        )
        if motion_region:
            start_idx, end_idx = motion_region
            # Add some padding around detected motion
            padding = int(sequence_length * MOTION_PADDING_RATIO)
            start_idx = max(0, start_idx - padding)
            end_idx = min(len(sequence), end_idx + padding)
            trimmed = sequence[start_idx:end_idx]
            # Fall back to full sequence if trimming made it too short
            if len(trimmed) >= sequence_length:
                sequence = trimmed
            else:
                logger.warning(
                    f"Motion-trimmed region too short "
                    f"({len(trimmed)} < {sequence_length}), using full sequence"
                )
        else:
            logger.warning("No clear gesture motion detected")

    for i in range(0, len(sequence) - sequence_length + 1, step_size):
        seq = sequence[i : i + sequence_length]
        sequences.append(seq)

    return sequences
