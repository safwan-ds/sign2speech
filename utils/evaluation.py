"""
Enhanced evaluation utilities for model performance analysis.

Core evaluation metrics and analysis functions. Plotting functions are
re-exported from ``utils.evaluation_plot`` (matplotlib) and
``utils.evaluation_plot_qtgraphs`` (QtGraphs backend).
"""

import logging
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,  # type: ignore
    confusion_matrix,
    precision_recall_fscore_support,  # type: ignore
)

from config.config import EVALUATION_CLASS_WEIGHT_EPSILON

logger = logging.getLogger(__name__)


def compute_class_weights(y: np.ndarray, num_classes: int):
    """
    Compute class weights for imbalanced datasets

    Args:
        y: Label array
        num_classes: Number of classes

    Returns:
        class_weights: Tensor of class weights
    """
    unique, counts = np.unique(y, return_counts=True)
    class_counts = np.zeros(num_classes)

    for cls, count in zip(unique, counts):
        class_counts[cls] = count

    # Compute weights (inverse frequency)
    total = np.sum(class_counts)
    class_weights = total / (
        num_classes * class_counts + EVALUATION_CLASS_WEIGHT_EPSILON
    )

    # Normalize weights
    class_weights = class_weights / np.sum(class_weights) * num_classes

    return torch.FloatTensor(class_weights)


def evaluate_transition_regions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    boundary_window: int = 2,
    top_n: int = 5,
) -> dict[str, object]:
    """Compute transition-focused metrics around true class boundaries."""
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    if y_true_arr.size == 0 or y_true_arr.shape != y_pred_arr.shape:
        return {
            "boundary_window": int(boundary_window),
            "boundary_frame_count": 0,
            "boundary_accuracy": 0.0,
            "boundary_error_rate": 0.0,
            "top_boundary_confusions": [],
        }

    # Indices immediately after each true class boundary.
    transition_points = np.where(y_true_arr[1:] != y_true_arr[:-1])[0] + 1
    boundary_mask = np.zeros(y_true_arr.shape[0], dtype=bool)
    for idx in transition_points:
        left = max(0, int(idx) - int(boundary_window))
        right = min(y_true_arr.shape[0], int(idx) + int(boundary_window) + 1)
        boundary_mask[left:right] = True

    boundary_count = int(np.sum(boundary_mask))
    if boundary_count == 0:
        return {
            "boundary_window": int(boundary_window),
            "boundary_frame_count": 0,
            "boundary_accuracy": 0.0,
            "boundary_error_rate": 0.0,
            "top_boundary_confusions": [],
        }

    boundary_true = y_true_arr[boundary_mask]
    boundary_pred = y_pred_arr[boundary_mask]
    boundary_acc = float(np.mean(boundary_true == boundary_pred))

    confusion_counts: dict[tuple[int, int], int] = {}
    for true_idx, pred_idx in zip(boundary_true.tolist(), boundary_pred.tolist()):
        if int(true_idx) == int(pred_idx):
            continue
        key = (int(true_idx), int(pred_idx))
        confusion_counts[key] = confusion_counts.get(key, 0) + 1

    ranked = sorted(confusion_counts.items(), key=lambda item: item[1], reverse=True)
    top_limit = max(0, int(top_n))
    top_confusions = [
        {
            "true": (
                class_names[true_idx]
                if 0 <= true_idx < len(class_names)
                else str(true_idx)
            ),
            "predicted": (
                class_names[pred_idx]
                if 0 <= pred_idx < len(class_names)
                else str(pred_idx)
            ),
            "count": int(count),
        }
        for (true_idx, pred_idx), count in ranked[:top_limit]
    ]

    return {
        "boundary_window": int(boundary_window),
        "boundary_frame_count": boundary_count,
        "boundary_accuracy": boundary_acc,
        "boundary_error_rate": float(1.0 - boundary_acc),
        "top_boundary_confusions": top_confusions,
    }


def derive_per_class_thresholds(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    class_names: list[str],
    min_samples: int = 5,
) -> dict[str, dict[str, float | int]]:
    """Derive class-wise confidence/gap thresholds from correct predictions."""
    probs = np.asarray(y_pred_probs, dtype=float)
    y_true_arr = np.asarray(y_true, dtype=int)

    if probs.ndim != 2 or probs.shape[0] == 0:
        return {}

    y_pred = np.argmax(probs, axis=1).astype(int)
    sorted_probs = np.sort(probs, axis=1)
    top1 = sorted_probs[:, -1]
    top2 = sorted_probs[:, -2] if sorted_probs.shape[1] > 1 else top1.copy()
    gaps = top1 - top2
    correct_mask = y_pred == y_true_arr

    fallback_conf = (
        float(np.percentile(top1[correct_mask], 25))
        if np.any(correct_mask)
        else float(np.percentile(top1, 50))
    )
    fallback_gap = (
        float(np.percentile(gaps[correct_mask], 25))
        if np.any(correct_mask)
        else float(np.percentile(gaps, 50))
    )

    thresholds: dict[str, dict[str, float | int]] = {}
    for class_idx, class_name in enumerate(class_names):
        class_mask = (y_true_arr == class_idx) & correct_mask
        sample_count = int(np.sum(class_mask))
        if sample_count >= max(1, int(min_samples)):
            conf_thr = float(np.percentile(top1[class_mask], 25))
            gap_thr = float(np.percentile(gaps[class_mask], 25))
        else:
            conf_thr = fallback_conf
            gap_thr = fallback_gap

        thresholds[str(class_name)] = {
            "confidence": float(np.clip(conf_thr, 0.0, 1.0)),
            "gap": float(np.clip(gap_thr, 0.0, 1.0)),
            "samples": sample_count,
        }

    return thresholds


def comprehensive_evaluation(
    model: torch.nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list[str],
    device: torch.device,
    save_dir: str | None = None,
    dataset_name: str = "test",
):
    """
    Perform comprehensive evaluation with visualizations

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        class_names: List of class names
        device: PyTorch device
        save_dir: Directory to save visualizations
        dataset_name: Name of the dataset (for file naming)

    Returns:
        metrics: Dictionary containing all evaluation metrics
    """
    # Ensure save directory exists
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Convert to tensors
    X_test_tensor = torch.FloatTensor(X_test).to(device)

    # Get predictions
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        y_pred_probs = torch.softmax(outputs, dim=1).cpu().numpy()
        y_pred = torch.argmax(outputs, dim=1).cpu().numpy()

    # Compute basic metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted"
    )

    # Build aligned labels so class_names and confusion matrix stay consistent
    # even when the val/test set doesn't contain every class.
    all_labels = list(range(len(class_names)))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=all_labels)

    # Print results
    logger.info(f"{dataset_name.upper()} SET EVALUATION")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision (weighted): {precision:.4f}")
    logger.info(f"Recall (weighted): {recall:.4f}")
    logger.info(f"F1-Score (weighted): {f1:.4f}")

    logger.info("Classification Report")
    logger.info("\n" + classification_report(y_test, y_pred, labels=all_labels, target_names=class_names, zero_division=0))  # type: ignore

    transition_metrics = evaluate_transition_regions(
        y_true=np.asarray(y_test),
        y_pred=np.asarray(y_pred),
        class_names=class_names,
        boundary_window=2,
        top_n=5,
    )
    per_class_thresholds = derive_per_class_thresholds(
        y_true=np.asarray(y_test),
        y_pred_probs=np.asarray(y_pred_probs),
        class_names=class_names,
        min_samples=5,
    )

    # Create visualizations
    if save_dir:
        # Confusion matrix
        cm_path = os.path.join(save_dir, f"confusion_matrix_{dataset_name}.pdf")
        plot_confusion_matrix(cm, class_names, save_path=cm_path, normalize=False)

        cm_norm_path = os.path.join(
            save_dir, f"confusion_matrix_{dataset_name}_normalized.pdf"
        )
        plot_confusion_matrix(cm, class_names, save_path=cm_norm_path, normalize=True)

        # ROC curves
        roc_path = os.path.join(save_dir, f"roc_curves_{dataset_name}.pdf")
        try:
            roc_auc = plot_roc_curves(
                y_test, y_pred_probs, class_names, save_path=roc_path
            )
        except Exception as e:
            logger.warning(f"Could not compute ROC curves: {e}")
            roc_auc = None

        # Per-class metrics
        metrics_path = os.path.join(save_dir, f"per_class_metrics_{dataset_name}.pdf")
        plot_per_class_metrics(y_test, y_pred, class_names, save_path=metrics_path)

        transition_report_path = os.path.join(
            save_dir, f"transition_metrics_{dataset_name}.txt"
        )
        with open(transition_report_path, "w", encoding="utf-8") as report:
            report.write("TRANSITION-FOCUSED METRICS\n")
            report.write("-" * 60 + "\n")
            report.write(
                f"Boundary window: ±{transition_metrics['boundary_window']} frames\n"
            )
            report.write(
                f"Boundary frames: {transition_metrics['boundary_frame_count']}\n"
            )
            report.write(
                f"Boundary accuracy: {transition_metrics['boundary_accuracy']:.4f}\n"
            )
            report.write(
                f"Boundary error rate: {transition_metrics['boundary_error_rate']:.4f}\n\n"
            )
            report.write("Top boundary confusions:\n")
            for entry in transition_metrics["top_boundary_confusions"]:
                report.write(
                    f"- {entry['true']} -> {entry['predicted']}: {entry['count']}\n"
                )
    else:
        roc_auc = None

    # Compile metrics
    metrics: dict[str, float | np.ndarray | dict[int, float] | None] = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": cm,
        "predictions": y_pred,
        "probabilities": y_pred_probs,
        "roc_auc": roc_auc,
        "transition_metrics": transition_metrics,
        "per_class_thresholds": per_class_thresholds,
    }

    return metrics


def analyze_misclassifications(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str], top_n: int = 5
):
    """
    Analyze most common misclassifications

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        top_n: Number of top misclassifications to display
    """
    all_labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)

    # Find misclassifications (off-diagonal elements)
    misclassifications: list[tuple[str, str, int, float]] = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                misclassifications.append(
                    (
                        class_names[i],
                        class_names[j],
                        cm[i, j],
                        cm[i, j] / np.sum(cm[i]) if np.sum(cm[i]) > 0 else 0,
                    )
                )

    # Sort by count
    misclassifications.sort(key=lambda x: x[2], reverse=True)

    logger.info(f"Top {top_n} Misclassifications")
    logger.info(
        f"{'True Class':<15} {'Predicted As':<15} {'Count':>8} {'% of True':>10}"
    )

    for i, (true_cls, pred_cls, count, percentage) in enumerate(
        misclassifications[:top_n]
    ):
        logger.info(f"{true_cls:<15} {pred_cls:<15} {count:>8} {percentage*100:>9.1f}%")


def save_evaluation_summary(
    metrics: dict[str, float | np.ndarray | dict[int, float] | None],
    history: dict[str, list[float]] | None,
    save_path: str,
):
    """
    Save evaluation summary to text file

    Args:
        metrics: Evaluation metrics dictionary
        history: Training history dictionary
        save_path: Path to save summary
    """
    with open(save_path, "w") as f:
        f.write("MODEL EVALUATION SUMMARY\n")

        f.write("Performance Metrics:\n")
        f.write("-" * 60 + "\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall: {metrics['recall']:.4f}\n")
        f.write(f"F1-Score: {metrics['f1_score']:.4f}\n\n")

        if history:
            f.write("Training Summary:\n")
            f.write("-" * 60 + "\n")
            f.write(f"Final Training Loss: {history['train_loss'][-1]:.4f}\n")
            f.write(f"Final Validation Loss: {history['val_loss'][-1]:.4f}\n")
            f.write(f"Final Training Accuracy: {history['train_acc'][-1]:.2f}%\n")
            f.write(f"Final Validation Accuracy: {history['val_acc'][-1]:.2f}%\n")
            f.write(f"Best Validation Accuracy: {max(history['val_acc']):.2f}%\n")
            f.write(f"Total Epochs: {len(history['train_loss'])}\n")

    logger.info(f"Evaluation summary saved to: {save_path}")


# ---------------------------------------------------------------------------
# Re-exports so that ``from utils.evaluation import plot_*`` still works.
# ---------------------------------------------------------------------------
from utils.evaluation_plot import (  # noqa: E402, F401
    plot_confusion_matrix,
    plot_per_class_metrics,
    plot_roc_curves,
    plot_training_history,
)
