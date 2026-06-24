"""Reusable training pipeline orchestration shared by GUI and CLI wrappers."""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from typing import Literal

from config.architecture import architecture
from config.config import MODELS_DIR
from config.config import setup_logging

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TrainingPipelineResult:
    """Normalized training pipeline output for GUI/CLI callers."""

    status: Literal["completed", "cancelled", "failed"]
    message: str | None = None
    model_dir: str | None = None
    is_ensemble: bool = False
    metadata: dict[str, object] = field(default_factory=dict)


def _latest_model_dir(models_dir: str, prefix: str) -> str | None:
    candidates: list[tuple[float, str]] = []
    if not os.path.isdir(models_dir):
        return None
    for name in os.listdir(models_dir):
        if not name.startswith(prefix):
            continue
        path = os.path.join(models_dir, name)
        if os.path.isdir(path):
            candidates.append((os.path.getmtime(path), path))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def run_training_pipeline(
    *,
    logger_instance: logging.Logger | None = None,
    epoch_callback: Callable[[int, int, float, float, float, float, float], None]
    | None = None,
    cancel_event: threading.Event | None = None,
    n_epochs: int | None = None,
    learning_rate: float | None = None,
    batch_size: int | None = None,
    patience: int | None = None,
    use_ensemble: bool | None = None,
    models_dir: str = MODELS_DIR,
) -> TrainingPipelineResult:
    """Run the full processed-sequences → trained model pipeline."""
    active_logger = logger_instance or logger

    # Lazy imports keep module import lightweight for unit tests.
    from core.training.data_loader import load_processed_sequences
    from core.training.ensemble import train_ensemble_models
    from core.training.model_evaluation import evaluate_lstm_model
    from core.training.model_training import train_lstm_model
    from core.training.model_utils import save_lstm_model

    active_logger.info("LSTM MODEL TRAINING")

    eff_epochs = n_epochs if n_epochs is not None else architecture.training.epochs
    eff_lr = learning_rate if learning_rate is not None else architecture.training.learning_rate
    eff_batch = batch_size if batch_size is not None else architecture.training.batch_size
    eff_patience = patience if patience is not None else architecture.training.early_stopping_patience
    eff_use_ensemble = use_ensemble if use_ensemble is not None else architecture.training.use_ensemble

    X, y, test_X, test_y = load_processed_sequences()
    if X is None:
        return TrainingPipelineResult(
            status="failed",
            message="No processed sequences found. Run Processing first.",
        )

    if eff_use_ensemble:
        result_ensemble = train_ensemble_models(
            X, y, test_X, test_y, epoch_callback=epoch_callback, cancel_event=cancel_event
        )
        if result_ensemble[0] is None or len(result_ensemble[0]) == 0:
            return TrainingPipelineResult(status="cancelled")

        ensemble_models, label_encoder, mean, std = result_ensemble
        model_dir = _latest_model_dir(models_dir, "ensemble_")
        return TrainingPipelineResult(
            status="completed",
            model_dir=model_dir,
            is_ensemble=True,
            metadata={
                "ensemble_models": len(ensemble_models),
                "num_classes": len(label_encoder.classes_) if label_encoder is not None else 0,
                "has_normalization": bool(mean is not None and std is not None),
            },
        )

    result = train_lstm_model(
        X,
        y,
        test_X,
        test_y,
        epoch_callback=epoch_callback,
        cancel_event=cancel_event,
        n_epochs=eff_epochs,
        learning_rate=eff_lr,
        batch_size=eff_batch,
        patience=eff_patience,
    )
    if result[0] is None:
        return TrainingPipelineResult(status="cancelled")

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
    ) = result

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(models_dir, f"lstm_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)

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

    metadata: dict[str, object] = {
        "num_classes": len(label_encoder.classes_),
        "classes": ", ".join(label_encoder.classes_),
        "val_accuracy": f"{val_accuracy:.4f}",
        "total_sequences": len(X),
        "sequence_length": X.shape[1],
        "num_features": X.shape[2],
        "epochs_run": eff_epochs,
        "learning_rate": eff_lr,
        "batch_size": eff_batch,
        "early_stopping_patience": eff_patience,
    }
    if test_accuracy is not None:
        metadata["test_accuracy"] = f"{test_accuracy:.4f}"

    save_lstm_model(model, label_encoder, mean, std, metadata, model_dir=model_dir)
    return TrainingPipelineResult(
        status="completed",
        model_dir=model_dir,
        metadata=metadata,
    )


def main(*, configure_logging: bool = True) -> None:
    """CLI entry-point for training pipeline."""
    if configure_logging:
        setup_logging("train_model")
    result = run_training_pipeline(logger_instance=logger)
    if result.status == "cancelled":
        logger.info("TRAINING CANCELLED")
        return
    if result.status == "failed":
        logger.error("%s", result.message or "Training failed")
        return
    if result.model_dir:
        logger.info("MODEL_DIR=%s", result.model_dir)
    if result.is_ensemble:
        logger.info("ENSEMBLE TRAINING COMPLETE!")
    else:
        logger.info("TRAINING COMPLETE!")
