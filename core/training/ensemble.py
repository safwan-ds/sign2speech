"""Ensemble model training utilities"""

import sys
import os
import numpy as np
import torch
from datetime import datetime
import logging

# Add parent directory to path for config imports
sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ),
)
from config import (
    MODELS_DIR,
    RANDOM_STATE,
    ENSEMBLE_SIZE,
    LSTM_UNITS,
    LSTM_LAYERS,
    DROPOUT_RATE,
    EPOCHS,
    BATCH_SIZE,
    MODEL_TYPE,
    USE_BIDIRECTIONAL,
    USE_ATTENTION,
    USE_BATCH_NORM,
    USE_AUGMENTATION,
    USE_WEIGHTED_LOSS,
    USE_LABEL_SMOOTHING,
    USE_COSINE_ANNEALING,
    LEARNING_RATE,
    WEIGHT_DECAY,
)

from .model_training import train_lstm_model
from .model_evaluation import evaluate_lstm_model
from .model_utils import save_lstm_model

logger = logging.getLogger(__name__)


def train_ensemble_models(X, y, test_X, test_y):
    """Train ensemble of models with different random seeds for diversity.

    Each model uses a different random seed so it learns slightly different
    patterns, making the ensemble more robust than any single model.

    Args:
        X: Training features
        y: Training labels
        test_X: Test features (optional)
        test_y: Test labels (optional)

    Returns:
        Tuple of (ensemble_models, label_encoder, mean, std)
    """
    logger.info(f"TRAINING ENSEMBLE OF {ENSEMBLE_SIZE} MODELS")

    all_accuracies = []
    ensemble_models = []
    label_encoder = None
    mean = None
    std = None

    # Create a shared folder for the whole ensemble run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(MODELS_DIR, f"ensemble_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)

    for ensemble_idx in range(ENSEMBLE_SIZE):
        logger.info(f"Training Model {ensemble_idx + 1}/{ENSEMBLE_SIZE}")

        # Use a different random seed for each ensemble member
        # so each model learns different patterns from the data
        ensemble_seed = RANDOM_STATE + ensemble_idx
        np.random.seed(ensemble_seed)
        torch.manual_seed(ensemble_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(ensemble_seed)

        # Train individual model
        (
            model,
            label_encoder,
            X_val,
            y_val,
            history,
            mean,
            std,
            test_X_norm,
            test_y_norm,
        ) = train_lstm_model(X, y, test_X, test_y)

        # Evaluate model
        val_accuracy, test_accuracy = evaluate_lstm_model(
            model,
            label_encoder,
            X_val,
            y_val,
            history,
            test_X_norm,
            test_y_norm,
            model_dir=model_dir,
        )

        all_accuracies.append(val_accuracy)
        ensemble_models.append(
            (model, label_encoder, mean, std, val_accuracy, test_accuracy)
        )

        # Save individual ensemble model
        metadata = {
            "num_classes": len(label_encoder.classes_),
            "classes": ", ".join(label_encoder.classes_),
            "val_accuracy": f"{val_accuracy:.4f}",
            "ensemble_idx": ensemble_idx + 1,
            "ensemble_size": ENSEMBLE_SIZE,
            "random_seed": ensemble_seed,
        }

        if test_accuracy is not None:
            metadata["test_accuracy"] = f"{test_accuracy:.4f}"

        metadata.update(
            {
                "total_sequences": len(X),
                "sequence_length": X.shape[1],
                "num_features": X.shape[2],
                "lstm_units": LSTM_UNITS,
                "lstm_layers": LSTM_LAYERS,
                "dropout_rate": DROPOUT_RATE,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "model_type": MODEL_TYPE,
                "bidirectional": USE_BIDIRECTIONAL,
                "attention": USE_ATTENTION,
                "batch_norm": USE_BATCH_NORM,
                "data_augmentation": USE_AUGMENTATION,
                "weighted_loss": USE_WEIGHTED_LOSS,
                "label_smoothing": USE_LABEL_SMOOTHING,
                "cosine_annealing": USE_COSINE_ANNEALING,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
            }
        )

        save_lstm_model(
            model,
            label_encoder,
            mean,
            std,
            metadata,
            ensemble_idx=ensemble_idx,
            model_dir=model_dir,
        )

    # Print ensemble summary
    logger.info("ENSEMBLE SUMMARY")
    for idx, acc in enumerate(all_accuracies):
        logger.info(f"  Model {idx + 1} validation accuracy: {acc:.4f}")
    logger.info(f"  Average accuracy: {np.mean(all_accuracies):.4f}")
    logger.info(f"  Std deviation:    {np.std(all_accuracies):.4f}")

    return ensemble_models, label_encoder, mean, std
