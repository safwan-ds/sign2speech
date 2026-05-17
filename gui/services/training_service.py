"""Background training service for the dataset manager GUI.

Orchestrates the full processed-sequences → trained model pipeline in a daemon
thread, exposes a :meth:`cancel` method so the GUI can abort training between
epochs, and emits typed events to the shared ``event_queue``.

Event types emitted
-------------------
``train_started``
    Training has begun. No extra payload.
``train_epoch``
    ``epoch: int, total: int, train_loss: float, train_acc: float,
    val_loss: float, val_acc: float, lr: float`` – per-epoch metrics.
``train_model_dir``
    ``model_dir: str`` – absolute path to the newly saved model directory.
``train_completed``
    Training finished successfully. No extra payload.
``train_cancelled``
    Training was cancelled by the user. No extra payload.
``train_failed``
    ``message: str`` – human-readable error message on failure.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from queue import Queue

from config.config import (
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    LEARNING_RATE,
    MODELS_DIR,
    USE_ENSEMBLE,
)
from core.training.data_loader import load_processed_sequences
from core.training.ensemble import train_ensemble_models
from core.training.model_evaluation import evaluate_lstm_model
from core.training.model_training import train_lstm_model
from core.training.model_utils import save_lstm_model


@dataclass(slots=True)
class TrainingOverrides:
    """Optional runtime hyper-parameter overrides exposed in the GUI Train tab."""

    epochs: int | None = None
    learning_rate: float | None = None
    batch_size: int | None = None
    early_stopping_patience: int | None = None


class TrainingService:
    """Run the training pipeline in a background thread with cancellation support."""

    def __init__(
        self,
        logger: logging.Logger,
        event_queue: Queue[dict],
    ) -> None:
        self._logger = logger
        self._event_queue = event_queue
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._cancel_event = threading.Event()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    def start(self, overrides: TrainingOverrides | None = None) -> bool:
        """Start training. Returns *False* when already running."""
        with self._lock:
            if self.is_running:
                return False
            self._cancel_event.clear()
            self._thread = threading.Thread(
                target=self._run,
                args=(overrides or TrainingOverrides(),),
                daemon=True,
                name="TrainingService",
            )
            self._thread.start()
            return True

    def cancel(self) -> None:
        """Request cancellation. Training stops cleanly after the current epoch."""
        self._cancel_event.set()
        self._logger.info("[train] Cancellation requested")

    # ------------------------------------------------------------------
    # Internal implementation
    # ------------------------------------------------------------------

    def _emit(self, event_type: str, **payload: object) -> None:
        self._event_queue.put({"type": event_type, **payload})

    def _epoch_callback(
        self,
        epoch: int,
        total: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        lr: float,
    ) -> None:
        self._emit(
            "train_epoch",
            epoch=epoch,
            total=total,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
            lr=lr,
        )

    def _run(self, overrides: TrainingOverrides) -> None:
        try:
            self._emit("train_started")
            self._logger.info("[train] Starting training pipeline")

            # Log effective hyper-parameters
            eff_epochs = overrides.epochs if overrides.epochs is not None else EPOCHS
            eff_lr = (
                overrides.learning_rate
                if overrides.learning_rate is not None
                else LEARNING_RATE
            )
            eff_batch = (
                overrides.batch_size
                if overrides.batch_size is not None
                else BATCH_SIZE
            )
            eff_patience = (
                overrides.early_stopping_patience
                if overrides.early_stopping_patience is not None
                else EARLY_STOPPING_PATIENCE
            )
            self._logger.info(
                "[train] Overrides — epochs=%d, lr=%.6f, batch=%d, patience=%d",
                eff_epochs,
                eff_lr,
                eff_batch,
                eff_patience,
            )

            # Load processed sequences
            X, y, test_X, test_y = load_processed_sequences()
            if X is None:
                self._emit("train_failed", message="No processed sequences found. Run Processing first.")
                return

            if USE_ENSEMBLE:
                ensemble_models, label_encoder, mean, std = train_ensemble_models(
                    X, y, test_X, test_y
                )
                self._logger.info("[train] ENSEMBLE TRAINING COMPLETE!")
                self._emit("train_completed")
                return

            # Single model training
            result = train_lstm_model(
                X,
                y,
                test_X,
                test_y,
                epoch_callback=self._epoch_callback,
                cancel_event=self._cancel_event,
                n_epochs=eff_epochs,
                learning_rate=eff_lr,
                batch_size=eff_batch,
                patience=eff_patience,
            )

            # Cancellation: train_lstm_model returns a tuple of Nones
            if result[0] is None:
                self._emit("train_cancelled")
                self._logger.info("[train] Training cancelled")
                return

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

            # Create model directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_dir = Path(MODELS_DIR) / f"lstm_{timestamp}"
            model_dir.mkdir(parents=True, exist_ok=True)

            # Evaluate
            val_accuracy, test_accuracy = evaluate_lstm_model(
                model,
                label_encoder,
                X_val,
                y_val,
                history,
                test_X_norm,
                test_y_norm,
                model_dir=str(model_dir),
            )

            # Persist model
            metadata: dict = {
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

            save_lstm_model(model, label_encoder, mean, std, metadata, model_dir=str(model_dir))

            self._emit("train_model_dir", model_dir=str(model_dir))
            self._emit("train_completed")
            self._logger.info("[train] Training complete — model saved to %s", model_dir)

        except Exception as exc:
            self._logger.exception("[train] Training pipeline failed: %s", exc)
            self._emit("train_failed", message=str(exc))
        finally:
            with self._lock:
                self._thread = None
