"""Model loading adapter for existing predictor implementation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from core.inference import gesture_predictor as predictor_module
from core.inference.gesture_predictor import LSTMGesturePredictor


@dataclass(slots=True)
class ModelMetadata:
    """Metadata shown in the model details panel."""

    classes: list[str]
    sequence_length: int
    input_shape: str
    loaded_at: str
    model_dir: Path


class ModelService:
    """Load and hold predictor state for GUI consumption."""

    def __init__(self) -> None:
        self.predictor: LSTMGesturePredictor | None = None
        self.metadata: ModelMetadata | None = None

    def load(self, model_dir: Path, use_ensemble: bool = False) -> ModelMetadata:
        """Load predictor assets from a model directory."""
        model_path = model_dir / "model.pth"
        encoder_path = model_dir / "encoder.npy"
        norm_path = model_dir / "normalization.npz"

        if not use_ensemble:
            if not model_path.exists() or not encoder_path.exists():
                raise FileNotFoundError(
                    "Selected model directory must contain model.pth and encoder.npy"
                )
        else:
            if not encoder_path.exists():
                raise FileNotFoundError(
                    "Selected model directory must contain encoder.npy for ensemble"
                )

        predictor_module.NORM_PATH = str(norm_path)
        self.predictor = LSTMGesturePredictor(
            model_path=str(model_path),
            encoder_path=str(encoder_path),
            use_ensemble=use_ensemble,
        )

        classes = [str(label) for label in self.predictor.classes.tolist()]
        sample_size = len(self.predictor.expected_features)
        self.metadata = ModelMetadata(
            classes=classes,
            sequence_length=self.predictor.sequence_length,
            input_shape=f"({self.predictor.sequence_length}, {sample_size})",
            loaded_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            model_dir=model_dir,
        )
        return self.metadata

    def require_predictor(self) -> LSTMGesturePredictor:
        """Return predictor instance or raise if model is not loaded."""
        if self.predictor is None:
            raise RuntimeError("Model is not loaded")
        return self.predictor
