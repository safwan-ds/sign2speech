"""Model factory for building LSTM models from saved checkpoints.

Provides :func:`load_model_checkpoint` to extract raw checkpoint data and
:func:`build_model_from_checkpoint` that resolves architecture parameters,
builds the model, loads the state dict, and puts it in eval mode.

Usage::

    model, arch_params, input_size, inferred_num_classes = (
        build_model_from_checkpoint(
            "models/latest/model.pth",
            torch.device("cuda"),
            encoder_num_classes=10,
            hidden_size=64,
            num_layers=2,
            dropout_rate=0.4,
            model_type="enhanced",
            bidirectional=True,
            use_attention=True,
            use_batch_norm=True,
            use_cnn=False,
        )
    )
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from core.models.lstm_model import build_lstm_model


def load_model_checkpoint(
    path: str,
    device: torch.device,
) -> tuple[dict, dict, int, int | None]:
    """Load a model checkpoint with backward compatibility.

    Handles both the new format (a dict with ``state_dict`` + ``arch_params``
    keys) and the legacy format (bare state dict).

    Returns:
        ``(state_dict, arch_params, input_size, num_classes)`` where
        *num_classes* is the raw value inferred from the checkpoint (may be
        ``None`` if it could not be determined).
    """
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        # Older PyTorch without weights_only parameter
        checkpoint = torch.load(path, map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        arch_params = checkpoint.get("arch_params", {})
    else:
        state_dict = checkpoint
        arch_params = {}

    # Infer input size -----------------------------------------------------------
    # Check input_proj first (advanced model projects input before LSTM)
    if "input_size" in arch_params:
        input_size = int(arch_params["input_size"])
    elif "input_proj.weight" in state_dict:
        input_size = int(state_dict["input_proj.weight"].shape[1])
    elif "lstm.weight_ih_l0" in state_dict:
        input_size = int(state_dict["lstm.weight_ih_l0"].shape[1])
    else:
        raise ValueError(f"Cannot infer input size from model: {path}")

    # Infer num_classes ---------------------------------------------------------
    num_classes: int | None = arch_params.get("num_classes")  # type: ignore[assignment]
    if num_classes is None:
        # Try to find last layer output size
        # Check classifier (Transformer)
        classifier_weights = [
            k
            for k in state_dict.keys()
            if k.startswith("classifier.") and k.endswith(".weight")
        ]
        if classifier_weights:
            # Sort by index (classifier.N.weight)
            classifier_weights.sort(key=lambda x: int(x.split(".")[1]), reverse=True)
            num_classes = int(state_dict[classifier_weights[0]].shape[0])
        elif "fc_out.weight" in state_dict:
            num_classes = int(state_dict["fc_out.weight"].shape[0])
        elif "fc.weight" in state_dict:
            num_classes = int(state_dict["fc.weight"].shape[0])

    return state_dict, arch_params, input_size, num_classes


def build_model_from_checkpoint(
    path: str,
    device: torch.device,
    encoder_num_classes: int | None = None,
    **config_defaults: Any,
) -> tuple[nn.Module, dict, int, int | None]:
    """Load a checkpoint and build/eval the corresponding model.

    Loads the checkpoint, resolves architecture parameters (checkpoint values
    take priority over *config_defaults*), builds the LSTM model via
    :func:`~core.models.lstm_model.build_lstm_model`, loads the state dict,
    and sets the model to evaluation mode.

    num_classes resolution priority
        ``inferred_num_classes`` (from checkpoint) >
        ``arch_params["num_classes"]`` >
        ``encoder_num_classes`` (the *encoder_num_classes* argument)

    Args:
        path: Path to the ``.pth`` checkpoint file.
        device: Target device for the model (e.g. ``torch.device("cpu")``).
        encoder_num_classes:
            Fallback number of classes from the label encoder.  Used when the
            checkpoint does not carry num_classes information.
        **config_defaults:
            Default architecture parameter values.  Supported keys are:

            - ``hidden_size`` (default ``64``)
            - ``num_layers`` (default ``2``)
            - ``dropout_rate`` (default ``0.4``)
            - ``model_type`` (default ``"enhanced"``)
            - ``bidirectional`` (default ``True``)
            - ``use_attention`` (default ``True``)
            - ``use_batch_norm`` (default ``True``)
            - ``use_cnn`` (default ``False``)

    Returns:
        ``(model, arch_params, input_size, inferred_num_classes)`` where
        *inferred_num_classes* is the raw value from the checkpoint (before
        resolution), and *input_size*/*arch_params* are the raw checkpoint
        values.

    Raises:
        ValueError: If num_classes cannot be resolved from any source.
    """
    state_dict, arch_params, input_size, inferred_num_classes = load_model_checkpoint(
        path, device
    )

    # Resolve num_classes: checkpoint > arch_params > encoder fallback
    resolved_num_classes = (
        inferred_num_classes
        or arch_params.get("num_classes")
        or encoder_num_classes
    )
    if resolved_num_classes is None:
        raise ValueError(
            f"Cannot determine number of classes for checkpoint {path!r}. "
            "The checkpoint has no num_classes metadata and no "
            "encoder_num_classes was provided."
        )

    model = build_lstm_model(
        input_size=input_size,
        num_classes=int(resolved_num_classes),
        hidden_size=arch_params.get(
            "hidden_size", config_defaults.get("hidden_size", 64)
        ),
        num_layers=arch_params.get(
            "num_layers", config_defaults.get("num_layers", 2)
        ),
        dropout_rate=arch_params.get(
            "dropout_rate", config_defaults.get("dropout_rate", 0.4)
        ),
        device=device,
        model_type=arch_params.get(
            "model_type", config_defaults.get("model_type", "enhanced")
        ),
        bidirectional=arch_params.get(
            "bidirectional", config_defaults.get("bidirectional", True)
        ),
        use_attention=arch_params.get(
            "use_attention", config_defaults.get("use_attention", True)
        ),
        use_batch_norm=arch_params.get(
            "use_batch_norm", config_defaults.get("use_batch_norm", True)
        ),
        use_cnn=arch_params.get(
            "use_cnn", config_defaults.get("use_cnn", False)
        ),
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model, arch_params, input_size, inferred_num_classes
