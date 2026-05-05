"""
Data processing and normalization utilities for Sign Language Glove project
"""

import numpy as np
import pandas as pd
import logging

from config import (
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
    NORMALIZE_YAW_ROTATION,
    ACCEL_X_IDX,
    ACCEL_Y_IDX,
    GYRO_X_IDX,
    GYRO_Y_IDX,
)

logger = logging.getLogger(__name__)


def convert_to_snake_case(label: str) -> str:
    """Convert gesture label to snake case (spaces to underscores)."""
    return label.replace(" ", "_")


def normalize_yaw_rotation(
    sequence: np.ndarray,
    accel_x_idx: int = ACCEL_X_IDX,
    accel_y_idx: int = ACCEL_Y_IDX,
    gyro_x_idx: int = GYRO_X_IDX,
    gyro_y_idx: int = GYRO_Y_IDX,
) -> np.ndarray:
    """
    Remove the yaw (vertical-axis rotation) component from IMU data.

    Estimates the user's facing direction from the mean horizontal acceleration
    of the sequence and counter-rotates accelX/Y and gyroX/Y into a canonical
    frame.  This makes gesture features invariant to which direction the user
    is facing when performing a gesture.

    Args:
        sequence: numpy array of shape (n_samples, n_features)
        accel_x_idx: Index of accelX in the feature vector
        accel_y_idx: Index of accelY in the feature vector
        gyro_x_idx: Index of gyroX in the feature vector
        gyro_y_idx: Index of gyroY in the feature vector

    Returns:
        Yaw-normalized sequence of the same shape and dtype
    """
    n_features = sequence.shape[1]
    if n_features <= max(accel_x_idx, accel_y_idx, gyro_x_idx, gyro_y_idx):
        return sequence

    # Compute the mean horizontal acceleration to estimate facing direction
    mean_ax = np.mean(sequence[:, accel_x_idx])
    mean_ay = np.mean(sequence[:, accel_y_idx])

    # Counter-rotation angle that cancels out the mean yaw
    yaw = np.arctan2(mean_ay, mean_ax)
    cos_yaw = np.cos(-yaw)
    sin_yaw = np.sin(-yaw)

    result = sequence.copy()

    # Rotate accelerometer XY
    ax = sequence[:, accel_x_idx]
    ay = sequence[:, accel_y_idx]
    result[:, accel_x_idx] = cos_yaw * ax - sin_yaw * ay
    result[:, accel_y_idx] = sin_yaw * ax + cos_yaw * ay

    # Rotate gyroscope XY
    gx = sequence[:, gyro_x_idx]
    gy = sequence[:, gyro_y_idx]
    result[:, gyro_x_idx] = cos_yaw * gx - sin_yaw * gy
    result[:, gyro_y_idx] = sin_yaw * gx + cos_yaw * gy

    return result.astype(np.float32)


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

    # Apply yaw normalization to make features invariant to facing direction
    if NORMALIZE_YAW_ROTATION:
        sequence = normalize_yaw_rotation(sequence)

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

    # Apply yaw normalization on base features before any feature engineering,
    # so that derived features (velocity, rolling stats) are also yaw-invariant
    if NORMALIZE_YAW_ROTATION:
        sequence = normalize_yaw_rotation(sequence)

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
