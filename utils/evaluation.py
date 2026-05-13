"""
Enhanced evaluation utilities for model performance analysis
Includes ROC curves, visualization, and detailed metrics
"""

import matplotlib
import numpy as np

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
import urllib.parse
from pathlib import Path

from config import (
    EVALUATION_CLASS_WEIGHT_EPSILON,
    EVALUATION_DPI,
)

logger = logging.getLogger(__name__)

EVALUATION_PLOT_BACKEND = (
    os.getenv("EVALUATION_PLOT_BACKEND", "matplotlib").strip().lower()
)

_QTGRAPHS_IMPORT_ERROR: Exception | None = None
try:
    from PySide6.QtCore import QMarginsF, QPointF, QSizeF, QUrl
    from PySide6.QtGui import (
        QColor,
        QImage,
        QPainter,
        QPageLayout,
        QPageSize,
        QPdfWriter,
    )
    from PySide6.QtGraphs import (
        QBarCategoryAxis,
        QBarSeries,
        QBarSet,
        QGraphsTheme,
        QLineSeries,
        QValueAxis,
    )
    from PySide6.QtQuickWidgets import QQuickWidget
    from PySide6.QtWidgets import QApplication

    QTGRAPHS_AVAILABLE = True
except Exception as exc:  # pragma: no cover - environment dependent
    QTGRAPHS_AVAILABLE = False
    _QTGRAPHS_IMPORT_ERROR = exc


def _qtgraphs_backend_enabled() -> bool:
    if EVALUATION_PLOT_BACKEND in {"qtgraphs", "qtgraphs-first"}:
        return QTGRAPHS_AVAILABLE and QApplication.instance() is not None
    return False


def _ensure_qapplication() -> QApplication:
    app = QApplication.instance()
    if app is None:
        raise RuntimeError("QtGraphs export requires an active QApplication instance")
    return app


def _render_qtgraphs_widget_to_file(
    widget: QQuickWidget,
    save_path: str,
    dpi: int = EVALUATION_DPI,
) -> None:
    app = _ensure_qapplication()
    widget.show()
    app.processEvents()
    image = widget.grabFramebuffer()
    widget.hide()

    if image.isNull():
        raise RuntimeError("QtGraphs returned an empty framebuffer")

    output_path = Path(save_path)
    if output_path.suffix.lower() == ".pdf":
        writer = QPdfWriter(str(output_path))
        writer.setResolution(dpi)

        width_inches = max(1.0, image.width() / float(dpi))
        height_inches = max(1.0, image.height() / float(dpi))
        writer.setPageSize(
            QPageSize(QSizeF(width_inches, height_inches), QPageSize.Unit.Inch)
        )
        writer.setPageMargins(QMarginsF(0, 0, 0, 0), QPageLayout.Unit.Millimeter)

        painter = QPainter(writer)
        painter.drawImage(0, 0, image)
        painter.end()
    else:
        image.save(str(output_path))

    widget.deleteLater()


def _create_qtgraphs_line_widget(
    series_count: int, width: int, height: int
) -> QQuickWidget:
    widget = QQuickWidget()
    widget.setResizeMode(QQuickWidget.ResizeMode.SizeRootObjectToView)
    widget.resize(width, height)

    series_defs = ",\n".join(
        f'        LineSeries {{ objectName: "series{idx}"; width: 2.0 }}'
        for idx in range(series_count)
    )
    qml = f"""import QtQuick 2.15
import QtGraphs

GraphsView {{
    objectName: "graphRoot"
    antialiasing: true
    clipPlotArea: true
    shadowVisible: false
    marginTop: 12
    marginBottom: 28
    marginLeft: 42
    marginRight: 16
    axisX: ValueAxis {{ objectName: "axisX"; min: 0; max: 1 }}
    axisY: ValueAxis {{ objectName: "axisY"; min: 0; max: 1 }}
    seriesList: [
{series_defs}
    ]
}}
"""

    qml_url = QUrl("data:text/plain," + urllib.parse.quote(qml))
    widget.setSource(qml_url)

    errors = widget.errors()
    if errors:
        joined = "; ".join(error.toString() for error in errors)
        raise RuntimeError(f"QtGraphs QML load failed: {joined}")

    return widget


def _create_qtgraphs_bar_widget(width: int, height: int) -> QQuickWidget:
    widget = QQuickWidget()
    widget.setResizeMode(QQuickWidget.ResizeMode.SizeRootObjectToView)
    widget.resize(width, height)

    qml = """import QtQuick 2.15
import QtGraphs

GraphsView {
    objectName: "graphRoot"
    antialiasing: true
    clipPlotArea: true
    shadowVisible: false
    marginTop: 12
    marginBottom: 40
    marginLeft: 46
    marginRight: 18
    axisX: BarCategoryAxis { objectName: "axisX" }
    axisY: ValueAxis { objectName: "axisY"; min: 0; max: 1.1 }
    seriesList: [ BarSeries { objectName: "bars" } ]
}
"""

    qml_url = QUrl("data:text/plain," + urllib.parse.quote(qml))
    widget.setSource(qml_url)

    errors = widget.errors()
    if errors:
        joined = "; ".join(error.toString() for error in errors)
        raise RuntimeError(f"QtGraphs QML load failed: {joined}")

    return widget


def _apply_qtgraphs_theme(root) -> None:
    theme = root.property("theme")
    if isinstance(theme, QGraphsTheme):
        theme.setBackgroundVisible(True)
        theme.setBackgroundColor(QColor("#ffffff"))
        theme.setPlotAreaBackgroundVisible(True)
        theme.setPlotAreaBackgroundColor(QColor("#ffffff"))
        theme.setGridVisible(True)
        theme.setLabelTextColor(QColor("#111827"))
        theme.setLabelBackgroundVisible(False)
        theme.setLabelBorderVisible(False)


def _plot_roc_curves_qtgraphs(
    fpr: dict[int, np.ndarray],
    tpr: dict[int, np.ndarray],
    class_names: list[str],
    roc_auc: dict[int, float],
    save_path: str,
) -> None:
    color_cycle = [
        "#2563eb",
        "#d97706",
        "#059669",
        "#dc2626",
        "#7c3aed",
        "#0891b2",
        "#b45309",
    ]

    widget = _create_qtgraphs_line_widget(
        series_count=len(class_names) + 1,
        width=1200,
        height=900,
    )
    root = widget.rootObject()
    if root is None:
        raise RuntimeError("QtGraphs ROC root object unavailable")

    _apply_qtgraphs_theme(root)

    axis_x = root.findChild(QValueAxis, "axisX")
    axis_y = root.findChild(QValueAxis, "axisY")
    if axis_x is None or axis_y is None:
        raise RuntimeError("QtGraphs ROC axis objects unavailable")

    axis_x.setRange(0.0, 1.0)
    axis_x.setTickInterval(0.2)
    axis_x.setLabelFormat("%.1f")
    axis_y.setRange(0.0, 1.05)
    axis_y.setTickInterval(0.2)
    axis_y.setLabelFormat("%.1f")

    for idx, class_name in enumerate(class_names):
        series = root.findChild(QLineSeries, f"series{idx}")
        if series is None:
            raise RuntimeError(f"QtGraphs ROC series missing: series{idx}")
        series.setName(f"{class_name} (AUC={roc_auc[idx]:.3f})")
        series.setColor(QColor(color_cycle[idx % len(color_cycle)]))

        for x_val, y_val in zip(fpr[idx].tolist(), tpr[idx].tolist()):
            series.append(QPointF(float(x_val), float(y_val)))

    random_series = root.findChild(QLineSeries, f"series{len(class_names)}")
    if random_series is None:
        raise RuntimeError("QtGraphs ROC random baseline series unavailable")
    random_series.setName("Random Classifier")
    random_series.setColor(QColor("#6b7280"))
    random_series.append(QPointF(0.0, 0.0))
    random_series.append(QPointF(1.0, 1.0))

    _render_qtgraphs_widget_to_file(widget, save_path)


def _plot_per_class_metrics_qtgraphs(
    class_names: list[str],
    precision: np.ndarray,
    recall: np.ndarray,
    f1: np.ndarray,
    save_path: str,
) -> None:
    widget = _create_qtgraphs_bar_widget(width=1400, height=840)
    root = widget.rootObject()
    if root is None:
        raise RuntimeError("QtGraphs metrics root object unavailable")

    _apply_qtgraphs_theme(root)

    axis_x = root.findChild(QBarCategoryAxis, "axisX")
    axis_y = root.findChild(QValueAxis, "axisY")
    bars = root.findChild(QBarSeries, "bars")

    if axis_x is None or axis_y is None or bars is None:
        raise RuntimeError("QtGraphs metrics objects unavailable")

    axis_x.setCategories([str(name) for name in class_names])
    axis_y.setRange(0.0, 1.1)
    axis_y.setTickInterval(0.2)
    axis_y.setLabelFormat("%.1f")

    precision_set = QBarSet("Precision")
    precision_set.append([float(v) for v in precision.tolist()])
    precision_set.setColor(QColor("#38bdf8"))

    recall_set = QBarSet("Recall")
    recall_set.append([float(v) for v in recall.tolist()])
    recall_set.setColor(QColor("#4ade80"))

    f1_set = QBarSet("F1-Score")
    f1_set.append([float(v) for v in f1.tolist()])
    f1_set.setColor(QColor("#f87171"))

    bars.append(precision_set)
    bars.append(recall_set)
    bars.append(f1_set)

    _render_qtgraphs_widget_to_file(widget, save_path)


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

    plt.figure(figsize=(10, 8))

    for i in range(num_classes):

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
    top_confusions = [
        {
            "true": class_names[true_idx]
            if 0 <= true_idx < len(class_names)
            else str(true_idx),
            "predicted": class_names[pred_idx]
            if 0 <= pred_idx < len(class_names)
            else str(pred_idx),
            "count": int(count),
        }
        for (true_idx, pred_idx), count in ranked[: max(1, int(top_n))]
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
    top2 = sorted_probs[:, -2] if sorted_probs.shape[1] > 1 else np.zeros_like(top1)
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
