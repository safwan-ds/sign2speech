"""
Matplotlib-based plotting functions for model evaluation.

Provides ROC curves, confusion matrix, per-class metrics,
and training history visualizations. Falls back to matplotlib
when the QtGraphs backend is unavailable or disabled.
"""

import logging

import matplotlib
import numpy as np

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_fscore_support  # type: ignore
from sklearn.metrics import roc_curve
from sklearn.preprocessing import label_binarize

from config.architecture import architecture
from utils.evaluation_plot_qtgraphs import _QTGRAPHS_IMPORT_ERROR
from utils.evaluation_plot_qtgraphs import EVALUATION_PLOT_BACKEND
from utils.evaluation_plot_qtgraphs import QTGRAPHS_AVAILABLE
from utils.evaluation_plot_qtgraphs import _plot_per_class_metrics_qtgraphs
from utils.evaluation_plot_qtgraphs import _plot_roc_curves_qtgraphs
from utils.evaluation_plot_qtgraphs import _qtgraphs_backend_enabled

logger = logging.getLogger(__name__)


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
            cm.sum(axis=1)[:, np.newaxis] + architecture.evaluation.evaluation_class_weight_epsilon
        )
        fmt = ".2f"
        title = "Normalized Confusion Matrix"
    else:
        fmt = "d"
        title = "Confusion Matrix"

    # Dynamic scaling: 0.4 inches per class, minimum 10 inches
    num_classes = len(class_names)
    fig_size = max(10, int(num_classes * 0.4))

    plt.figure(figsize=(fig_size, fig_size * 0.8))

    # Disable annotations if too many classes to prevent text overlap
    show_annotations = num_classes <= 25
    annot_kwargs = {"size": max(6, 12 - (num_classes // 5))}

    sns.heatmap(
        cm,
        annot=show_annotations,
        annot_kws=annot_kwargs if show_annotations else None,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
    )
    plt.title(title, fontsize=16, pad=20)
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)

    # Adjust label rotation for dense grids
    plt.xticks(rotation=90, ha="center", fontsize=max(6, 10 - (num_classes // 10)))
    plt.yticks(rotation=0, fontsize=max(6, 10 - (num_classes // 10)))
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="pdf", dpi=architecture.evaluation.evaluation_dpi, bbox_inches="tight")
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

    Returns:
        roc_auc: Dictionary mapping class index to AUC value
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

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])  # type: ignore
        roc_auc[i] = float(auc(fpr[i], tpr[i]))

    if save_path and _qtgraphs_backend_enabled():
        try:
            _plot_roc_curves_qtgraphs(fpr, tpr, class_names, roc_auc, save_path)
            logger.info(f"ROC curves saved to: {save_path} (QtGraphs)")
            return roc_auc
        except Exception as exc:
            logger.warning(
                "QtGraphs ROC plotting failed, falling back to matplotlib: %s",
                exc,
            )

    if save_path and not QTGRAPHS_AVAILABLE and EVALUATION_PLOT_BACKEND != "matplotlib":
        logger.debug(
            "QtGraphs unavailable (%s). Using matplotlib for ROC export.",
            _QTGRAPHS_IMPORT_ERROR,
        )

    plt.figure(figsize=(12, 8))  # Slightly wider for external legend

    for i in range(num_classes):
        plt.plot(
            fpr[i],
            tpr[i],
            lw=1.5,  # Thinner lines for density
            alpha=0.8,  # Slight transparency
            label=f"{class_names[i]} (AUC = {roc_auc[i]:.3f})",
        )

    plt.plot([0, 1], [0, 1], "k--", lw=2, label="Random Classifier")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curves - Multi-Class Classification", fontsize=16, pad=20)

    # External multi-column legend handling
    if num_classes > 15:
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=8, ncol=2)
    else:
        plt.legend(loc="lower right", fontsize=10)

    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="pdf", dpi=architecture.evaluation.evaluation_dpi, bbox_inches="tight")
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

    if save_path and _qtgraphs_backend_enabled():
        try:
            _plot_per_class_metrics_qtgraphs(
                class_names=class_names,
                precision=np.asarray(precision, dtype=float),
                recall=np.asarray(recall, dtype=float),
                f1=np.asarray(f1, dtype=float),
                save_path=save_path,
            )
            logger.info(f"Per-class metrics saved to: {save_path} (QtGraphs)")
            return
        except Exception as exc:
            logger.warning(
                "QtGraphs per-class metrics plotting failed, falling back to matplotlib: %s",
                exc,
            )

    if save_path and not QTGRAPHS_AVAILABLE and EVALUATION_PLOT_BACKEND != "matplotlib":
        logger.debug(
            "QtGraphs unavailable (%s). Using matplotlib for per-class metrics export.",
            _QTGRAPHS_IMPORT_ERROR,
        )

    num_classes = len(class_names)
    x = np.arange(num_classes)
    width = 0.25

    # Dynamic width: 0.5 inches per class, minimum 12 inches
    fig_width = max(12, int(num_classes * 0.5))
    _, ax = plt.subplots(figsize=(fig_width, 6))

    bars1 = ax.bar(x - width, precision, width, label="Precision", color="skyblue")
    bars2 = ax.bar(x, recall, width, label="Recall", color="lightgreen")
    bars3 = ax.bar(x + width, f1, width, label="F1-Score", color="lightcoral")

    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Per-Class Performance Metrics", fontsize=16, pad=20)
    ax.set_xticks(x)

    # Vertical rotation for dense x-axis labels
    ax.set_xticklabels(
        class_names, rotation=90, ha="center", fontsize=max(6, 10 - (num_classes // 10))
    )

    # Move legend outside
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)

    # Only add value labels on bars if the grid is sparse enough
    if num_classes <= 20:
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
                    rotation=90,
                )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="pdf", dpi=architecture.evaluation.evaluation_dpi, bbox_inches="tight")
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
        plt.savefig(save_path, format="pdf", dpi=architecture.evaluation.evaluation_dpi, bbox_inches="tight")
        logger.info(f"Training history saved to: {save_path}")
    else:
        plt.show()

    plt.close()
