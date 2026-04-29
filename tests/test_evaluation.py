"""Unit tests for compute_class_weights in evaluation module."""

from __future__ import annotations

import numpy as np
import pytest

# evaluation.py uses a QApplication return-type annotation; if PySide6 is
# absent the module raises a NameError at import time. Guard with try/except
# and skip the entire file when that happens.
try:
    import torch
    from utils.evaluation import compute_class_weights
    _evaluation_available = True
except Exception:
    _evaluation_available = False
    torch = None  # type: ignore[assignment]

pytestmark = pytest.mark.skipif(
    not _evaluation_available,
    reason="evaluation module requires PySide6 (QApplication annotation)",
)


class TestComputeClassWeights:
    def test_returns_tensor_of_correct_length(self) -> None:
        y = np.array([0, 1, 2, 0, 1, 2])
        weights = compute_class_weights(y, num_classes=3)
        assert isinstance(weights, torch.Tensor)
        assert len(weights) == 3

    def test_balanced_dataset_has_equal_weights(self) -> None:
        y = np.array([0, 1, 2, 0, 1, 2])
        weights = compute_class_weights(y, num_classes=3)
        # With equal class distribution, weights should be approximately equal
        assert torch.allclose(weights, weights[0].expand_as(weights), atol=1e-4)

    def test_imbalanced_minority_class_gets_higher_weight(self) -> None:
        # class 0 appears many times, class 1 appears rarely
        y = np.array([0] * 10 + [1] * 2)
        weights = compute_class_weights(y, num_classes=2)
        # Minority class (1) should have higher weight
        assert weights[1] > weights[0]

    def test_all_weights_are_positive(self) -> None:
        y = np.array([0, 0, 1, 2, 2, 2])
        weights = compute_class_weights(y, num_classes=3)
        assert (weights > 0).all()

    def test_weights_sum_to_num_classes(self) -> None:
        y = np.array([0, 1, 2, 0, 1, 2])
        num_classes = 3
        weights = compute_class_weights(y, num_classes=num_classes)
        assert float(weights.sum()) == pytest.approx(num_classes, abs=1e-4)
