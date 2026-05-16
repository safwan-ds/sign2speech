import glob
import logging
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    LOGS_DIR,
    PROCESSED_DIR,
    TEST_DATA_DIR,
    SEQUENCE_LENGTH,
    USE_TEST_SPLIT,
    TEST_DATA_SPLIT_PERCENTAGE,
    RANDOM_STATE,
    DETECT_GESTURE_MOTION,
    USE_ENHANCED_FEATURES,
    INCLUDE_VELOCITY,
    INCLUDE_ACCELERATION,
    INCLUDE_ROLLING_STATS,
    setup_logging,
)


from utils.data_utils import (
    normalize_dataframe,
    segment_sequences,
    segment_sequences_with_enhanced_features,
)
from utils.recording_utils import SENSOR_COLUMNS
from utils.serial_utils import FLEX_SENSOR_NAMES, build_flex_zero_warning

logger = logging.getLogger(__name__)


def load_log_file(filepath: str) -> pd.DataFrame | None:
    """Load a single CSV log file"""
    try:
        df = pd.read_csv(filepath)
        missing_columns = [col for col in SENSOR_COLUMNS if col not in df.columns]
        if missing_columns:
            logger.error(
                "Error loading %s: missing required sensor columns: %s",
                filepath,
                ", ".join(missing_columns),
            )
            return None

        numeric_df = df.loc[:, SENSOR_COLUMNS].copy()
        for column in SENSOR_COLUMNS:
            numeric_df[column] = pd.to_numeric(numeric_df[column], errors="coerce")

        before_rows = len(numeric_df)
        numeric_df = numeric_df.dropna(axis=0, how="any").reset_index(drop=True)
        dropped_rows = before_rows - len(numeric_df)
        if dropped_rows:
            logger.warning(
                "Dropped %s non-numeric row(s) from %s before serialization",
                dropped_rows,
                os.path.basename(filepath),
            )

        if numeric_df.empty:
            logger.error(f"Error loading {filepath}: no valid numeric sensor rows")
            return None

        zero_sensors = tuple(
            name for name in FLEX_SENSOR_NAMES if (numeric_df[name] == 0.0).any()
        )
        if zero_sensors:
            logger.warning(
                "%s in %s",
                build_flex_zero_warning(zero_sensors),
                os.path.basename(filepath),
            )

        logger.info(f"Loaded: {os.path.basename(filepath)} ({len(numeric_df)} samples)")
        return numeric_df
    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}")
        return None


def load_all_logs() -> dict[str, list[pd.DataFrame]]:
    """Load all CSV files from gesture subfolders in the logs directory"""
    if not os.path.exists(LOGS_DIR):
        logger.error(f"Logs directory not found: {LOGS_DIR}")
        return {}

    gestures_data: dict[str, list[pd.DataFrame]] = {}

    for gesture_folder in os.listdir(LOGS_DIR):
        gesture_path = os.path.join(LOGS_DIR, gesture_folder)

        if not os.path.isdir(gesture_path):
            continue

        csv_files = glob.glob(os.path.join(gesture_path, "*.csv"))

        if not csv_files:
            logger.warning(
                f"Gesture folder '{gesture_folder}' exists but contains no CSV files — skipping"
            )
            continue

        logger.info(
            f"Found {len(csv_files)} log file(s) for gesture: '{gesture_folder}'"
        )
        gestures_data[gesture_folder] = []

        for file in csv_files:
            df = load_log_file(file)
            if df is not None:

                df = normalize_dataframe(df)
                gestures_data[gesture_folder].append(df)

    if not gestures_data:
        logger.error(f"No log files found in {LOGS_DIR}")

    return gestures_data


def clear_previous_sequence_files() -> None:
    """Remove previously generated sequence files before regenerating data."""
    for output_dir in (PROCESSED_DIR, TEST_DATA_DIR):
        if not os.path.exists(output_dir):
            continue

        removed_files = 0
        for file_path in glob.glob(os.path.join(output_dir, "sequences_*.npz")):
            try:
                os.remove(file_path)
                removed_files += 1
                logger.info("Removed stale sequence file: %s", file_path)
            except OSError as exc:
                logger.warning("Could not remove %s: %s", file_path, exc)

        if removed_files:
            logger.info(
                "Cleared %s old sequence file(s) from %s", removed_files, output_dir
            )


def prepare_lstm_dataset(
    dataframes: list[pd.DataFrame],
    gesture_label: str,
    sequence_length: int = SEQUENCE_LENGTH,
):
    """
    Prepare dataset for LSTM training with optional enhanced features

    Returns:
        X: numpy array of shape (num_sequences, sequence_length, num_features)
        y: list of labels
    """
    all_sequences: list[np.ndarray] = []
    all_labels: list[str] = []

    for df in dataframes:
        if USE_ENHANCED_FEATURES:
            sequences = segment_sequences_with_enhanced_features(
                df,
                sequence_length,
                use_enhanced_features=True,
                include_derivatives=INCLUDE_VELOCITY or INCLUDE_ACCELERATION,
                include_stats=INCLUDE_ROLLING_STATS,
            )
        else:
            sequences = segment_sequences(df, sequence_length)

        for seq in sequences:
            all_sequences.append(seq)
            all_labels.append(gesture_label)

    if not all_sequences:
        return None, None

    X = np.array(all_sequences, dtype=np.float32)
    y = np.array(all_labels)

    return X, y


def save_processed_data_lstm(
    X: np.ndarray, y: np.ndarray, gesture_label: str, output_dir: str = PROCESSED_DIR
):
    """Save processed sequences for LSTM

    Args:
        X: Feature sequences
        y: Labels
        gesture_label: Name of the gesture
        output_dir: Directory to save to (default: PROCESSED_DIR)
    """
    os.makedirs(output_dir, exist_ok=True)

    filename = f"sequences_{gesture_label}.npz"
    filepath = os.path.join(output_dir, filename)

    np.savez_compressed(filepath, X=X, y=y)
    logger.info(f"Saved sequences to: {filepath}")
    logger.info(f"  Shape: X={X.shape}, y={y.shape}")

    return filepath


def main():
    """Main processing pipeline for LSTM data"""
    setup_logging("process_data")

    logger.info("LSTM DATA PROCESSING")
    clear_previous_sequence_files()

    gestures_data = load_all_logs()

    if not gestures_data:
        logger.error("No data to process!")
        return

    logger.info(f"Found {len(gestures_data)} gesture(s):")
    total_samples = 0
    for gesture_label, dataframes in gestures_data.items():
        sample_count = sum(len(df) for df in dataframes)
        total_samples += sample_count
        logger.info(
            f"  - {gesture_label}: {len(dataframes)} file(s), {sample_count} samples"
        )

    logger.info(f"Total samples: {total_samples}")
    if USE_TEST_SPLIT:
        logger.info(
            f"Data split: ~{(1-TEST_DATA_SPLIT_PERCENTAGE)*100:.0f}% training, ~{TEST_DATA_SPLIT_PERCENTAGE*100:.0f}% test (at recording-file level)"
        )
    else:
        logger.info("Test split: DISABLED — all data will be used for training")

    if DETECT_GESTURE_MOTION:
        logger.info("Motion detection: ENABLED")
        logger.info("  - Only active gesture regions will be segmented")
        logger.info("  - Idle periods before/after gestures will be excluded")
    else:
        logger.info("Motion detection: DISABLED")
        logger.info("  - Full recording will be segmented into fixed-length windows")

    if USE_ENHANCED_FEATURES:
        logger.info("Enhanced features: ENABLED")
        features_list = ["Base sensor values"]
        if INCLUDE_VELOCITY:
            features_list.append("Velocity (1st derivative)")
        if INCLUDE_ACCELERATION:
            features_list.append("Acceleration (2nd derivative)")
        if INCLUDE_ROLLING_STATS:
            features_list.append("Rolling statistics")
        logger.info("  - Features: " + ", ".join(features_list))
    else:
        logger.info("Enhanced features: DISABLED")
        logger.info("  - Using only base sensor values")

    processed_count = 0
    test_data_available = False

    for gesture_label, dataframes in gestures_data.items():
        logger.info(f"Processing gesture: '{gesture_label}'")

        n_files = len(dataframes)
        rng = np.random.RandomState(RANDOM_STATE)
        shuffled_indices = rng.permutation(n_files)

        if not USE_TEST_SPLIT:

            train_dfs = dataframes
            test_dfs = []
        elif n_files >= 2:
            split_idx = max(1, int(n_files * (1 - TEST_DATA_SPLIT_PERCENTAGE)))

            if split_idx >= n_files:
                split_idx = n_files - 1
            train_dfs = [dataframes[i] for i in shuffled_indices[:split_idx]]
            test_dfs = [dataframes[i] for i in shuffled_indices[split_idx:]]
        else:

            train_dfs = dataframes
            test_dfs = []

        logger.info(
            f"  File-level split: {len(train_dfs)} train, {len(test_dfs)} test recordings"
        )

        X_train, y_train = prepare_lstm_dataset(train_dfs, gesture_label)

        if X_train is not None and y_train is not None:
            logger.info(f"Training: {len(X_train)} sequences, shape: {X_train.shape}")
            logger.info(f"Features per timestep: {X_train.shape[2]}")
            save_processed_data_lstm(X_train, y_train, gesture_label, PROCESSED_DIR)
            processed_count += 1
        else:
            logger.warning(f"Could not create training sequences for '{gesture_label}'")

        if test_dfs:
            X_test, y_test = prepare_lstm_dataset(test_dfs, gesture_label)
            if X_test is not None and y_test is not None:
                logger.info(f"Test: {len(X_test)} sequences")
                save_processed_data_lstm(X_test, y_test, gesture_label, TEST_DATA_DIR)
                test_data_available = True
            else:
                logger.warning(f"Could not create test sequences for '{gesture_label}'")

    logger.info("PROCESSING COMPLETE")
    logger.info(f"Processed {processed_count}/{len(gestures_data)} gesture(s)")

    if test_data_available:
        logger.info(f"Training data saved to: {PROCESSED_DIR}")
        logger.info(f"Test data saved to:     {TEST_DATA_DIR}")
    else:
        logger.info(f"Data saved to: {PROCESSED_DIR}")

    logger.info("Next steps:")
    logger.info("1. Collect more data for different gestures (if needed)")
    logger.info("2. Run train_model.py to train the model")
    logger.info("3. Use predict.py for real-time recognition")


if __name__ == "__main__":
    main()
