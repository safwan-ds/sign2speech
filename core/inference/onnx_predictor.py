"""Standalone ONNX Runtime inference backend.

Extracts ONNX model loading and inference logic into a reusable class
that can be used independently of LSTMGesturePredictor.
"""

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


class ONNXBackend:
    """ONNX Runtime inference backend.

    Handles provider selection, session creation, and forward-pass
    inference.  ``predict()`` returns raw probabilities so the caller
    can apply any post-processing (e.g. ``_probability_result``).
    """

    def __init__(
        self,
        model_path: str,
        device: torch.device,
        norm_mean: np.ndarray | None = None,
    ) -> None:
        """Load an ONNX model and prepare it for inference.

        Args:
            model_path: Path to the ``.onnx`` file.
            device: Target torch device (used to place output tensors).
            norm_mean: Optional normalization mean array, used only as a
                fallback when the ONNX model metadata does not contain a
                static feature dimension.
        """
        try:
            import onnxruntime as ort  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required to load .onnx gesture models"
            ) from exc

        available = set(ort.get_available_providers())
        providers: list[str] = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in available
            else ["CPUExecutionProvider"]
        )
        self._session = ort.InferenceSession(model_path, providers=providers)

        input_meta = self._session.get_inputs()[0]
        self._onnx_input_name: str = input_meta.name
        self._input_size: int = self._resolve_input_size(input_meta, norm_mean)
        self._device = device

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def input_size(self) -> int:
        """Number of features the ONNX model expects per timestep."""
        return self._input_size

    def predict(self, sequence_tensor: torch.Tensor) -> torch.Tensor:
        """Run ONNX inference on a pre-processed input tensor.

        Args:
            sequence_tensor: Input tensor shaped ``(1, seq_len, input_size)``
                already on the correct device.

        Returns:
            Probabilities tensor shaped ``(1, num_classes)`` on the target
            device.
        """
        input_array = sequence_tensor.detach().cpu().numpy().astype(np.float32)
        outputs = self._session.run(
            None, {self._onnx_input_name: input_array}
        )
        logits = torch.from_numpy(
            np.asarray(outputs[0], dtype=np.float32)
        ).to(self._device)
        return torch.softmax(logits, dim=1)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_input_size(
        input_meta: object,
        norm_mean: np.ndarray | None,
    ) -> int:
        """Extract feature dimension from ONNX metadata or norm_mean."""
        input_shape = list(input_meta.shape)  # type: ignore[union-attr]
        feature_dim = input_shape[-1] if input_shape else None
        if isinstance(feature_dim, int) and feature_dim > 0:
            return feature_dim
        if norm_mean is not None:
            return int(len(norm_mean))
        raise ValueError(
            "Could not infer ONNX input feature size. Export with a static "
            "feature dimension or provide matching normalization.npz."
        )
