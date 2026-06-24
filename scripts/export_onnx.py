"""Export a trained Sign2Speech LSTM checkpoint to ONNX and optional INT8."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from config.architecture import architecture
from core.inference.gesture_predictor import MODEL_PATH
from core.models.model_factory import build_model_from_checkpoint
from core.models.model_factory import load_model_checkpoint

logger = logging.getLogger(__name__)


class NpzCalibrationDataReader:
    """ONNX Runtime calibration reader backed by an ``.npz`` file with ``X``."""

    def __init__(
        self,
        npz_path: Path,
        input_name: str,
        *,
        batch_size: int = 16,
    ) -> None:
        payload = np.load(npz_path)
        if "X" not in payload:
            raise ValueError(f"{npz_path} must contain an X array")
        self._input_name = input_name
        self._x = np.asarray(payload["X"], dtype=np.float32)
        self._batch_size = max(1, int(batch_size))
        self._offset = 0

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self._offset >= len(self._x):
            return None
        batch = self._x[self._offset : self._offset + self._batch_size]
        self._offset += self._batch_size
        return {self._input_name: batch.astype(np.float32, copy=False)}


def _default_output_path(model_path: Path) -> Path:
    return model_path.with_suffix(".onnx")


def export_onnx(
    model_path: Path,
    output_path: Path,
    *,
    sequence_length: int = architecture.training.sequence_length,
    opset: int = 17,
) -> tuple[Path, str]:
    # Resolve num_classes — checkpoint takes priority, encoder is fallback
    _, arch_params, input_size, inferred_num_classes = load_model_checkpoint(
        str(model_path), torch.device("cpu")
    )
    num_classes = inferred_num_classes or int(arch_params.get("num_classes", 0))
    if num_classes <= 0:
        encoder_path = model_path.with_name("encoder.npy")
        if not encoder_path.exists():
            raise ValueError("Could not infer num_classes and encoder.npy is missing")
        num_classes = len(np.load(encoder_path, allow_pickle=True))

    model, _, _, _ = build_model_from_checkpoint(
        str(model_path),
        torch.device("cpu"),
        encoder_num_classes=num_classes,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, sequence_length, input_size, dtype=torch.float32)
    input_name = "input"
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=[input_name],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=opset,
    )
    return output_path, input_name


def quantize_static_int8(
    onnx_path: Path,
    quantized_path: Path,
    calibration_npz: Path,
    input_name: str,
    *,
    batch_size: int = 16,
) -> Path:
    try:
        from onnxruntime.quantization import QuantFormat  # type: ignore
        from onnxruntime.quantization import QuantType  # type: ignore
        from onnxruntime.quantization import quantize_static  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "onnxruntime is required for static INT8 quantization"
        ) from exc

    reader = NpzCalibrationDataReader(
        calibration_npz,
        input_name,
        batch_size=batch_size,
    )
    quantized_path.parent.mkdir(parents=True, exist_ok=True)
    quantize_static(
        model_input=str(onnx_path),
        model_output=str(quantized_path),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
    )
    return quantized_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path(MODEL_PATH))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--sequence-length", type=int, default=architecture.training.sequence_length)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--calibration-npz", type=Path, default=None)
    parser.add_argument("--quantized-output", type=Path, default=None)
    parser.add_argument("--calibration-batch-size", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    args = parse_args()
    output = args.output or _default_output_path(args.model)
    onnx_path, input_name = export_onnx(
        args.model,
        output,
        sequence_length=args.sequence_length,
        opset=args.opset,
    )
    logger.info("Exported ONNX model: %s", onnx_path)

    if args.calibration_npz is None:
        logger.info("Skipping INT8 quantization: --calibration-npz not provided")
        return

    quantized_output = args.quantized_output or onnx_path.with_name(
        f"{onnx_path.stem}.int8.onnx"
    )
    quantized_path = quantize_static_int8(
        onnx_path,
        quantized_output,
        args.calibration_npz,
        input_name,
        batch_size=args.calibration_batch_size,
    )
    logger.info("Exported INT8 ONNX model: %s", quantized_path)


if __name__ == "__main__":
    main()
