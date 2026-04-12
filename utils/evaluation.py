"""
Enhanced evaluation utilities for model performance analysis
Includes ROC curves, visualization, and detailed metrics
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,  # type: ignore
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,  # type: ignore
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize
import torch
import os
import logging

from config import (
    EVALUATION_CLASS_WEIGHT_EPSILON,
    EVALUATION_DPI,
    CONFUSION_MATRIX_FIGSIZE,
    ROC_CURVE_FIGSIZE,
)

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


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    save_path: str | None = None,
    normalize: bool = False,
):
    """
    Plot confusion matrix with visualization

    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save figure
        normalize: Whether to normalize the confusion matrix
    """
    if normalize:
        cm = cm.astype("float") / (
            cm.sum(axis=1)[:, np.newaxis] + EVALUATION_CLASS_WEIGHT_EPSILON
        )
        fmt = ".2f"
        title = "Normalized Confusion Matrix"
    else:
        fmt = "d"
        title = "Confusion Matrix"

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
    )
    plt.title(title, fontsize=16, pad=20)
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="pdf", dpi=EVALUATION_DPI, bbox_inches="tight")
        logger.info(f"Confusion matrix saved to: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_roc_curves(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    class_names: list[str],
    save_path: str | None = None,
):
    """
    Plot ROC curves for multi-class classification

    Args:
        y_true: True labels
        y_pred_probs: Predicted probabilities (num_samples, num_classes)
        class_names: List of class names
        save_path: Path to save figure
    """
    num_classes = len(class_names)

    # Binarize labels for ROC computation
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
    if num_classes == 2:
        # For binary classification, label_binarize returns shape (n, 1)
        # We need to add the complement column
        y_true_bin_array = np.array(y_true_bin)
        y_true_bin = np.hstack([1 - y_true_bin_array, y_true_bin_array])  # type: ignore

    # Compute ROC curve and AUC for each class
    fpr: dict[int, np.ndarray] = dict()
    tpr: dict[int, np.ndarray] = dict()
    roc_auc: dict[int, float] = dict()

    plt.figure(figsize=(10, 8))

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])  # type: ignore
        roc_auc[i] = float(auc(fpr[i], tpr[i]))

        plt.plot(
            fpr[i],
            tpr[i],
            lw=2,
            label=f"{class_names[i]} (AUC = {roc_auc[i]:.3f})",
        )

    # Plot diagonal line
    plt.plot([0, 1], [0, 1], "k--", lw=2, label="Random Classifier")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curves - Multi-Class Classification", fontsize=16, pad=20)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="pdf", dpi=EVALUATION_DPI, bbox_inches="tight")
        logger.info(f"ROC curves saved to: {save_path}")
    else:
        plt.show()

    plt.close()

    return roc_auc


def plot_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    save_path: str | None = None,
):
    """
    Plot per-class precision, recall, and F1-score

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save figure
    """
    all_labels = list(range(len(class_names)))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=all_labels, average=None, zero_division=0
    )

    x = np.arange(len(class_names))
    width = 0.25

    _, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width, precision, width, label="Precision", color="skyblue")
    bars2 = ax.bar(x, recall, width, label="Recall", color="lightgreen")
    bars3 = ax.bar(x + width, f1, width, label="F1-Score", color="lightcoral")

    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Per-Class Performance Metrics", fontsize=16, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="pdf", dpi=EVALUATION_DPI, bbox_inches="tight")
        logger.info(f"Per-class metrics saved to: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_training_history(history: dict, save_path: str | None = None):
    """
    Plot training and validation loss/accuracy curves

    Args:
        history: Dictionary containing training history
        save_path: Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot loss
    ax1.plot(history["train_loss"], label="Training Loss", linewidth=2)
    ax1.plot(history["val_loss"], label="Validation Loss", linewidth=2)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Training and Validation Loss", fontsize=14)
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Plot accuracy
    ax2.plot(history["train_acc"], label="Training Accuracy", linewidth=2)
    ax2.plot(history["val_acc"], label="Validation Accuracy", linewidth=2)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Accuracy (%)", fontsize=12)
    ax2.set_title("Training and Validation Accuracy", fontsize=14)
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="pdf", dpi=EVALUATION_DPI, bbox_inches="tight")
        logger.info(f"Training history saved to: {save_path}")
    else:
        plt.show()

    plt.close()


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
