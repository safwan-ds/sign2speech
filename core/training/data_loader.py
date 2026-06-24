"""Data loading utilities for training sequences"""

import glob
import logging
import os

import numpy as np

from config.architecture import architecture
from config.config import PROCESSED_DIR
from config.config import TEST_DATA_DIR

logger = logging.getLogger(__name__)


def load_processed_sequences():
    """Load training sequences from processed directory and optionally load separate test set"""
    npz_files = glob.glob(os.path.join(PROCESSED_DIR, "sequences_*.npz"))

    if not npz_files:
        logger.error(f"No processed sequences found in {PROCESSED_DIR}")
        logger.error("Please run process_data.py first to generate sequences.")
        return None, None, None, None

    logger.info(f"Found {len(npz_files)} training sequence file(s)")

    all_X = []
    all_y = []

    for file in npz_files:
        data = np.load(file)
        X = data["X"]
        y = data["y"]

        all_X.append(X)
        all_y.append(y)

        gesture = os.path.basename(file).replace("sequences_", "").replace(".npz", "")
        logger.info(f"  {gesture}: {len(X)} sequences")

    # Concatenate training data
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)

    # Try to load separate test dataset (only if USE_TEST_SPLIT is enabled and TEST_SIZE > 0)
    test_X = None
    test_y = None

    if not architecture.training.use_test_split:
        logger.info("USE_TEST_SPLIT is False - skipping separate test set loading")
        logger.info("All data will be used for training.")
        return X, y, None, None

    if architecture.training.test_size > 0 and os.path.exists(TEST_DATA_DIR):
        test_npz_files = glob.glob(os.path.join(TEST_DATA_DIR, "sequences_*.npz"))
        if test_npz_files:
            logger.info(f"Found {len(test_npz_files)} separate test sequence file(s)")
            test_all_X = []
            test_all_y = []

            for file in test_npz_files:
                data = np.load(file)
                test_X_part = data["X"]
                test_y_part = data["y"]

                test_all_X.append(test_X_part)
                test_all_y.append(test_y_part)

                gesture = (
                    os.path.basename(file).replace("sequences_", "").replace(".npz", "")
                )
                logger.info(f"  {gesture}: {len(test_X_part)} sequences")

            test_X = np.concatenate(test_all_X, axis=0)
            test_y = np.concatenate(test_all_y, axis=0)
            logger.info(f"Total test sequences: {len(test_X)}")
        else:
            logger.info(f"No test sequences found in {TEST_DATA_DIR}")
            logger.info("Will use train-test split instead.")
    elif architecture.training.test_size == 0:
        logger.info("TEST_SIZE is 0 - skipping separate test set loading")
        logger.info("All data will be used for training without validation split.")
    else:
        logger.info(f"Test data directory not found: {TEST_DATA_DIR}")
        logger.info("Will use train-test split instead.")

    return X, y, test_X, test_y
