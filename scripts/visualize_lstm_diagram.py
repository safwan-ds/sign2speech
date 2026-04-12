"""
Generate a PDF architecture diagram for the project's LSTM model using torchviz.
"""

import argparse
import os
import sys
import logging
import numpy as np
import torch
from torchviz import make_dot

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import (
    MODELS_DIR,
    SEQUENCE_LENGTH,
    LSTM_UNITS,
    LSTM_LAYERS,
    DROPOUT_RATE,
    MODEL_TYPE,
    USE_BIDIRECTIONAL,
    USE_ATTENTION,
    USE_BATCH_NORM,
)
from core.models.lstm_model import build_lstm_model

logger = logging.getLogger(__name__)


DEFAULT_MODEL_PATH = os.path.join(MODELS_DIR, "latest", "model.pth")
DEFAULT_ENCODER_PATH = os.path.join(MODELS_DIR, "latest", "encoder.npy")
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "graphs")
DEFAULT_OUTPUT_BASENAME = "lstm_architecture"


def _infer_input_size(state_dict: dict[str, torch.Tensor]) -> int | None:
    """Infer input size from a saved state dict when possible."""
    # Check input_proj first - it contains the original input size
    if "input_proj.weight" in state_dict:
        return state_dict["input_proj.weight"].shape[1]
    if "lstm.weight_ih_l0" in state_dict:
        return state_dict["lstm.weight_ih_l0"].shape[1]
    return None


def _load_num_classes(encoder_path: str) -> int:
    classes = np.load(encoder_path, allow_pickle=True)
    return len(classes)


def _build_model(input_size: int, num_classes: int) -> torch.nn.Module:
    return build_lstm_model(
        input_size=input_size,
        num_classes=num_classes,
        hidden_size=LSTM_UNITS,
        num_layers=LSTM_LAYERS,
        dropout_rate=DROPOUT_RATE,
        device=torch.device("cpu"),
        model_type=MODEL_TYPE,
        bidirectional=USE_BIDIRECTIONAL,
        use_attention=USE_ATTENTION,
        use_batch_norm=USE_BATCH_NORM,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a torchviz PDF diagram for the LSTM model."
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help="Path to the trained model state dict (.pth).",
    )
    parser.add_argument(
        "--encoder-path",
        default=DEFAULT_ENCODER_PATH,
        help="Path to the label encoder (.npy).",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write the PDF diagram.",
    )
    parser.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT_BASENAME,
        help="Base name for the output PDF (no extension).",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=None,
        help="Override input size if it cannot be inferred.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path}")
    if not os.path.exists(args.encoder_path):
        raise FileNotFoundError(f"Encoder not found: {args.encoder_path}")

    state_dict = torch.load(args.model_path, map_location="cpu")

    # Handle both new format (dict with 'state_dict' + 'arch_params') and legacy
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    input_size = args.input_size or _infer_input_size(state_dict)
    if input_size is None:
        raise ValueError(
            "Could not infer input size. Provide --input-size to continue."
        )

    num_classes = _load_num_classes(args.encoder_path)
    model = _build_model(input_size, num_classes)
    model.load_state_dict(state_dict)
    model.eval()

    dummy_input = torch.zeros(1, SEQUENCE_LENGTH, input_size)
    output = model(dummy_input)

    params = dict(model.named_parameters())
    dot = make_dot(output, params=params)
    dot.attr(rankdir="LR")

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_name)
    dot.render(output_path, format="pdf", cleanup=True)

    logger.info(f"Saved diagram to {output_path}.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
