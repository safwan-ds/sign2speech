"""Model evaluation utilities"""

import os
import sys
import numpy as np
from datetime import datetime
import torch
import logging

# Add parent directory to path for config imports
sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ),
)
from config import MODELS_DIR
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

    return val_metrics["accuracy"], test_metrics["accuracy"] if test_metrics else None
