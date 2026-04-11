"""Main training orchestration module

This module coordinates the training pipeline by importing and using functions
from the training package:
- training.data_loader: Data loading utilities
- training.model_training: Core training logic
- training.model_evaluation: Evaluation utilities
- training.model_utils: Model persistence (saving/loading)
- training.ensemble: Ensemble training
"""

import sys
import os
from datetime import datetime
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import USE_ENSEMBLE, MODELS_DIR, setup_logging

logger = logging.getLogger(__name__)

from core.training.data_loader import load_processed_sequences
from core.training.model_training import train_lstm_model
from core.training.model_evaluation import evaluate_lstm_model
from core.training.model_utils import save_lstm_model
from core.training.ensemble import train_ensemble_models


def main():
    """Main training pipeline for LSTM"""
    setup_logging("train_model")

    logger.info("LSTM MODEL TRAINING")

    # Load training and test sequences
    X, y, test_X, test_y = load_processed_sequences()

    if X is None:
        return

    # Choose training mode: single model or ensemble
    if USE_ENSEMBLE:
        ensemble_models, label_encoder, mean, std = train_ensemble_models(
            X, y, test_X, test_y
        )
        logger.info("ENSEMBLE TRAINING COMPLETE!")
        logger.info(f"Trained {len(ensemble_models)} models for ensemble predictions")
    else:
        # Train single model
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

        # Create a dedicated folder for this training run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = os.path.join(MODELS_DIR, f"lstm_{timestamp}")
        os.makedirs(model_dir, exist_ok=True)

        # Evaluate model (results saved into model_dir/evaluation/)
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

        # Save model files into the same folder
        metadata = {
            "num_classes": len(label_encoder.classes_),
            "classes": ", ".join(label_encoder.classes_),
            "val_accuracy": f"{val_accuracy:.4f}",
        }

        if test_accuracy is not None:
            metadata["test_accuracy"] = f"{test_accuracy:.4f}"

        metadata.update(
            {
                "total_sequences": len(X),
                "sequence_length": X.shape[1],
                "num_features": X.shape[2],
            }
        )

        save_lstm_model(model, label_encoder, mean, std, metadata, model_dir=model_dir)

        logger.info("TRAINING COMPLETE!")


if __name__ == "__main__":
    main()
