"""Normalization utilities for sensor data.

Provides functions for normalizing raw sensor values (flex, accelerometer,
gyroscope) into a standard range, normalising whole DataFrames, extracting
normalised feature arrays, and querying feature names.
"""

import logging

import numpy as np
import pandas as pd

from config.architecture import architecture

logger = logging.getLogger(__name__)


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
        min_val, max_val = architecture.hardware.flex_sensor_ranges.get(sensor_idx, architecture.hardware.flex_sensor_default_range)
        value = max(min_val, min(max_val, value))  # Clip
        normalized = (value - min_val) / (max_val - min_val)
    elif name.startswith("accel"):
        # Accelerometer normalization
        value = max(architecture.hardware.min_accel_value, min(architecture.hardware.max_accel_value, value))  # Clip
        normalized = (value - architecture.hardware.min_accel_value) / (architecture.hardware.max_accel_value - architecture.hardware.min_accel_value)
    elif name.startswith("gyro"):
        # Gyroscope normalization
        value = max(architecture.hardware.min_gyro_value, min(architecture.hardware.max_gyro_value, value))  # Clip
        normalized = (value - architecture.hardware.min_gyro_value) / (architecture.hardware.max_gyro_value - architecture.hardware.min_gyro_value)
    else:
        return None

    # Scale to NORM_MIN - NORM_MAX range
    return normalized * (architecture.normalization.norm_max - architecture.normalization.norm_min) + architecture.normalization.norm_min


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
    for i in range(architecture.hardware.num_flex_sensors):
        raw_col = f"flex{i}" if f"flex{i}" in df.columns else f"sensor{i}"
        norm_col = f"flex{i}_norm"

        if raw_col in df.columns:
            min_val, max_val = architecture.hardware.flex_sensor_ranges.get(i, (0, 1023))
            # Clip values to range, then normalize to 0-1
            df[norm_col] = df[raw_col].clip(min_val, max_val)
            df[norm_col] = (df[norm_col] - min_val) / (max_val - min_val)
            df[norm_col] = df[norm_col] * (architecture.normalization.norm_max - architecture.normalization.norm_min) + architecture.normalization.norm_min

    # Normalize accelerometer data
    for axis in ["X", "Y", "Z"]:
        raw_col = f"accel{axis}"
        norm_col = f"accel{axis}_norm"

        if raw_col in df.columns:
            df[norm_col] = df[raw_col].clip(architecture.hardware.min_accel_value, architecture.hardware.max_accel_value)
            df[norm_col] = (df[norm_col] - architecture.hardware.min_accel_value) / (
                architecture.hardware.max_accel_value - architecture.hardware.min_accel_value
            )
            df[norm_col] = df[norm_col] * (architecture.normalization.norm_max - architecture.normalization.norm_min) + architecture.normalization.norm_min

    # Normalize gyroscope data
    for axis in ["X", "Y", "Z"]:
        raw_col = f"gyro{axis}"
        norm_col = f"gyro{axis}_norm"

        if raw_col in df.columns:
            df[norm_col] = df[raw_col].clip(architecture.hardware.min_gyro_value, architecture.hardware.max_gyro_value)
            df[norm_col] = (df[norm_col] - architecture.hardware.min_gyro_value) / (
                architecture.hardware.max_gyro_value - architecture.hardware.min_gyro_value
            )
            df[norm_col] = df[norm_col] * (architecture.normalization.norm_max - architecture.normalization.norm_min) + architecture.normalization.norm_min

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
    for i in range(architecture.hardware.num_flex_sensors):
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


def get_feature_names():
    """
    Get list of feature names in the expected order

    Returns:
        list: List of feature names
    """
    features: list[str] = []

    # Flex sensors
    for i in range(architecture.hardware.num_flex_sensors):
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
