"""Model evaluation utilities"""

import json
import logging
import os
from datetime import datetime

import numpy as np
import torch

from config.config import MODELS_DIR
from utils.evaluation import (
    comprehensive_evaluation,
    analyze_misclassifications,
    plot_training_history,
    save_evaluation_summary,
)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger(__name__)


def evaluate_lstm_model(
    model,
    label_encoder,
    X_test,
    y_test,
    history=None,
    separate_test_X=None,
    separate_test_y=None,
    model_dir=None,
):
    """Evaluate trained model with comprehensive metrics and visualizations

    Args:
        model: Trained model
        label_encoder: Label encoder for class names
        X_test: Validation set features (from train-test split)
        y_test: Validation set labels (from train-test split)
        history: Training history dictionary
        separate_test_X: Optional separate test set features with all gestures
        separate_test_y: Optional separate test set labels with all gestures
        model_dir: Parent model directory. If provided, evaluation artifacts
                   are saved into ``model_dir/evaluation/``.

    Returns:
        Tuple of (val_accuracy, test_accuracy)
    """
    # Create evaluation directory inside the model folder (or standalone)
    if model_dir is not None:
        eval_dir = os.path.join(model_dir, "evaluation")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        eval_dir = os.path.join(MODELS_DIR, f"evaluation_{timestamp}")
    os.makedirs(eval_dir, exist_ok=True)

    logger.info(f"Saving evaluation results to: {eval_dir}")

    # Evaluate validation set with comprehensive metrics
    val_metrics = comprehensive_evaluation(
        model,
        X_test,
        y_test,
        label_encoder.classes_,
        device,
        save_dir=eval_dir,
        dataset_name="validation",
    )

    # Analyze misclassifications
    analyze_misclassifications(
        y_test, np.asarray(val_metrics["predictions"]), label_encoder.classes_, top_n=5
    )

    # Diagnostic warning for perfect accuracy
    if val_metrics["accuracy"] == 1.0:
        logger.warning(f"Perfect validation accuracy (1.0000) detected!")

    # Evaluate on separate test set if provided
    test_metrics = None
    if separate_test_X is not None and separate_test_y is not None:
        test_metrics = comprehensive_evaluation(
            model,
            separate_test_X,
            separate_test_y,
            label_encoder.classes_,
            device,
            save_dir=eval_dir,
            dataset_name="holdout_test",
        )

        # Analyze misclassifications for test set
        analyze_misclassifications(
            separate_test_y,
            np.asarray(test_metrics["predictions"]),
            label_encoder.classes_,
            top_n=5,
        )

        # Diagnostic warning for perfect accuracy on holdout test
        if test_metrics["accuracy"] == 1.0:
            logger.warning(f"Perfect holdout test accuracy (1.0000)")

    # Plot training history if provided
    if history:
        history_path = os.path.join(eval_dir, "training_history.pdf")
        plot_training_history(history, save_path=history_path)

        # Save evaluation summary
        summary_path = os.path.join(eval_dir, "evaluation_summary.txt")
        save_evaluation_summary(val_metrics, history, summary_path)

    # Save metrics to JSON for GUI display
    gui_metrics: dict = {
        "eval_dir": eval_dir,
        "validation": {
            "accuracy": float(val_metrics["accuracy"]),  # type: ignore
            "precision": float(val_metrics["precision"]),  # type: ignore
            "recall": float(val_metrics["recall"]),  # type: ignore
            "f1_score": float(val_metrics["f1_score"]),  # type: ignore
            "confusion_matrix": np.asarray(val_metrics["confusion_matrix"]).tolist(),  # type: ignore
            "class_names": list(label_encoder.classes_),
            "transition_metrics": val_metrics.get("transition_metrics", {}),
        },
    }

    if test_metrics:
        gui_metrics["test"] = {
            "accuracy": float(test_metrics["accuracy"]),  # type: ignore
            "precision": float(test_metrics["precision"]),  # type: ignore
            "recall": float(test_metrics["recall"]),  # type: ignore
            "f1_score": float(test_metrics["f1_score"]),  # type: ignore
            "confusion_matrix": np.asarray(test_metrics["confusion_matrix"]).tolist(),  # type: ignore
            "class_names": list(label_encoder.classes_),
            "transition_metrics": test_metrics.get("transition_metrics", {}),
        }

    per_class_thresholds = val_metrics.get("per_class_thresholds", {})
    if isinstance(per_class_thresholds, dict) and per_class_thresholds:
        thresholds_path = os.path.join(eval_dir, "per_class_thresholds.json")
        with open(thresholds_path, "w", encoding="utf-8") as f:
            json.dump(per_class_thresholds, f, indent=2)
        logger.info(f"Per-class thresholds saved to: {thresholds_path}")

    metrics_json_path = os.path.join(eval_dir, "metrics.json")
    with open(metrics_json_path, "w") as f:
        json.dump(gui_metrics, f, indent=2)
    logger.info(f"Metrics JSON saved to: {metrics_json_path}")

    return val_metrics["accuracy"], test_metrics["accuracy"] if test_metrics else None
