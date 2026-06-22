"""Model persistence utilities (saving/loading)"""

import logging
import os
from datetime import datetime

import numpy as np
import torch

from config.architecture import architecture
from config.config import MODELS_DIR

logger = logging.getLogger(__name__)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _extract_arch_params(model: torch.nn.Module) -> dict:
    """Extract architecture parameters from a model for checkpoint reproducibility."""
    params: dict[str, object] = {}

    # Common attributes
    for attr in (
        "hidden_size",
        "num_layers",
        "bidirectional",
        "use_attention",
        "use_batch_norm",
        "use_cnn",
    ):
        if hasattr(model, attr):
            params[attr] = getattr(model, attr)

    # Infer model type from architecture
    if hasattr(model, "multihead_attn"):
        params["model_type"] = "advanced"
    elif getattr(model, "use_attention", False) or getattr(
        model, "use_batch_norm", False
    ):
        params["model_type"] = "enhanced"
    else:
        params["model_type"] = "basic"

    # Input size from weights
    # Check input_proj first (advanced model projects input before LSTM)
    state_dict = model.state_dict()
    if "input_proj.weight" in state_dict:
        params["input_size"] = int(state_dict["input_proj.weight"].shape[1])
    elif "lstm.weight_ih_l0" in state_dict:
        params["input_size"] = int(state_dict["lstm.weight_ih_l0"].shape[1])

    # Num classes from output layer
    if "fc_out.weight" in state_dict:
        params["num_classes"] = int(state_dict["fc_out.weight"].shape[0])
    elif "classifier.8.weight" in state_dict:
        params["num_classes"] = int(state_dict["classifier.8.weight"].shape[0])

    # Dropout rate
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            params["dropout_rate"] = module.p
            break

    return params


def save_lstm_model(
    model, label_encoder, mean, std, metadata, ensemble_idx=None, model_dir=None
):
    """Save trained LSTM model into a dedicated subfolder.

    Files are saved into ``models/lstm_<timestamp>/`` (or the given *model_dir*).
    A ``models/latest/`` folder is always kept as a copy of the most recent run
    so that inference scripts can load from a fixed path.

    Args:
        model: Trained model
        label_encoder: Label encoder for classes
        mean: Normalization mean
        std: Normalization std
        metadata: Metadata dictionary
        ensemble_idx: Index for ensemble model (None for single model)
        model_dir: Explicit directory to save into (created if needed).
                   When *None* a timestamped folder is generated automatically.

    Returns:
        Path to the model directory
    """
    # Create timestamped subfolder inside MODELS_DIR
    if model_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = (
            f"ensemble_{timestamp}" if ensemble_idx is not None else f"lstm_{timestamp}"
        )
        model_dir = os.path.join(MODELS_DIR, folder_name)
    os.makedirs(model_dir, exist_ok=True)

    # Build checkpoint with architecture params for reproducible loading
    checkpoint = {
        "state_dict": model.state_dict(),
        "arch_params": _extract_arch_params(model),
    }

    # Decide file names (ensemble members get an index suffix)
    suffix = f"_{ensemble_idx}" if ensemble_idx is not None else ""

    model_path = os.path.join(model_dir, f"model{suffix}.pth")
    encoder_path = os.path.join(model_dir, "encoder.npy")
    norm_path = os.path.join(model_dir, "normalization.npz")
    metadata_path = os.path.join(model_dir, f"metadata{suffix}.txt")

    # Save model (PyTorch)
    torch.save(checkpoint, model_path)

    # Save label encoder & normalization (shared across ensemble members)
    np.save(encoder_path, label_encoder.classes_)
    np.savez(norm_path, mean=mean, std=std)

    # Mirror into models/latest/ for quick inference access
    latest_dir = os.path.join(MODELS_DIR, "latest")
    
    # Clear latest directory if this is a single model or the first ensemble member
    # to avoid mixing stale files from previous different run types (e.g. ensemble vs single)
    if ensemble_idx is None or ensemble_idx == 0:
        if os.path.exists(latest_dir):
            try:
                import shutil
                shutil.rmtree(latest_dir)
            except Exception as e:
                logger.warning(f"Could not clear latest directory: {e}")
    
    os.makedirs(latest_dir, exist_ok=True)

    latest_model_name = (
        f"model{suffix}.pth" if ensemble_idx is not None else "model.pth"
    )
    torch.save(checkpoint, os.path.join(latest_dir, latest_model_name))
    np.save(os.path.join(latest_dir, "encoder.npy"), label_encoder.classes_)
    np.savez(os.path.join(latest_dir, "normalization.npz"), mean=mean, std=std)

    # Save metadata
    timestamp_str = os.path.basename(model_dir)
    with open(metadata_path, "w") as f:
        f.write(f"PyTorch LSTM Model Metadata\n")
        f.write(f"{'=' * 60}\n")
        f.write(f"Folder: {timestamp_str}\n")
        if ensemble_idx is not None:
            f.write(f"Ensemble: Model {ensemble_idx + 1} of {architecture.training.ensemble_size}\n")
        f.write(f"Model Type: PyTorch LSTM\n")
        f.write(f"Device: {device}\n")
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")

    logger.info(
        f"MODEL SAVED {f'(Ensemble {ensemble_idx + 1}/{architecture.training.ensemble_size})' if ensemble_idx is not None else ''}"
    )
    logger.info(f"Folder:        {model_dir}")
    logger.info(f"Model:         {model_path}")
    logger.info(f"Encoder:       {encoder_path}")
    logger.info(f"Normalization: {norm_path}")
    logger.info(f"Metadata:      {metadata_path}")
    logger.info(f"Latest:        {latest_dir}")

    return model_dir
