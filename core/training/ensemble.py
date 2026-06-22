"""Ensemble model training utilities"""

import logging
import os
from datetime import datetime

import numpy as np
import torch

from config.architecture import architecture
from config.config import MODELS_DIR
from .model_evaluation import evaluate_lstm_model
from .model_training import train_lstm_model
from .model_utils import save_lstm_model

logger = logging.getLogger(__name__)


def train_ensemble_models(X, y, test_X, test_y, epoch_callback=None, cancel_event=None):
    """Train ensemble of models with different random seeds for diversity.

    Each model uses a different random seed so it learns slightly different
    patterns, making the ensemble more robust than any single model.

    Args:
        X: Training features
        y: Training labels
        test_X: Test features (optional)
        test_y: Test labels (optional)
        epoch_callback: Optional callback for progress reporting
        cancel_event: Optional threading.Event to abort training

    Returns:
        Tuple of (ensemble_models, label_encoder, mean, std)
    """
    logger.info(f"TRAINING ENSEMBLE OF {architecture.training.ensemble_size} MODELS")

    all_accuracies = []
    ensemble_models = []
    label_encoder = None
    mean = None
    std = None

    # Create a shared folder for the whole ensemble run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(MODELS_DIR, f"ensemble_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)

    total_ensemble_epochs = architecture.training.ensemble_size * architecture.training.epochs

    for ensemble_idx in range(architecture.training.ensemble_size):
        if cancel_event is not None and cancel_event.is_set():
            logger.info("Ensemble training cancelled between models.")
            break

        logger.info(f"Training Model {ensemble_idx + 1}/{architecture.training.ensemble_size}")

        # Use a different random seed for each ensemble member
        # so each model learns different patterns from the data
        ensemble_seed = architecture.training.random_state + ensemble_idx
        np.random.seed(ensemble_seed)
        torch.manual_seed(ensemble_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(ensemble_seed)

        def wrapped_callback(epoch, total, t_loss, t_acc, v_loss, v_acc, lr):
            if epoch_callback:
                # Offset epoch to show global progress across the whole ensemble
                global_epoch = ensemble_idx * architecture.training.epochs + epoch
                epoch_callback(
                    global_epoch,
                    total_ensemble_epochs,
                    t_loss,
                    t_acc,
                    v_loss,
                    v_acc,
                    lr,
                )

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
        ) = train_lstm_model(
            X,
            y,
            test_X,
            test_y,
            epoch_callback=wrapped_callback,
            cancel_event=cancel_event,
        )

        if model is None:  # Cancelled
            logger.info(f"Model {ensemble_idx + 1} training cancelled.")
            break

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
            "ensemble_size": architecture.training.ensemble_size,
            "random_seed": ensemble_seed,
        }

        if test_accuracy is not None:
            metadata["test_accuracy"] = f"{test_accuracy:.4f}"

        metadata.update(
            {
                "total_sequences": len(X),
                "sequence_length": X.shape[1],
                "num_features": X.shape[2],
                "lstm_units": architecture.model.lstm_units,
                "lstm_layers": architecture.model.lstm_layers,
                "dropout_rate": architecture.model.dropout_rate,
                "epochs": architecture.training.epochs,
                "batch_size": architecture.training.batch_size,
                "model_type": architecture.model.model_type,
                "bidirectional": architecture.model.use_bidirectional,
                "attention": architecture.model.use_attention,
                "batch_norm": architecture.model.use_batch_norm,
                "data_augmentation": architecture.augmentation.use_augmentation,
                "weighted_loss": architecture.training.use_weighted_loss,
                "label_smoothing": architecture.training.use_label_smoothing,
                "cosine_annealing": architecture.training.use_cosine_annealing,
                "learning_rate": architecture.training.learning_rate,
                "weight_decay": architecture.training.weight_decay,
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
