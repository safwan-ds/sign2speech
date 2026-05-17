import importlib.util
import logging
import os
import time
from collections import deque

import numpy as np
import torch
from core.models.lstm_model import build_lstm_model

from config.config import (
    MODELS_DIR,
    SEQUENCE_LENGTH,
    PREDICTION_INTERVAL,
    LSTM_UNITS,
    LSTM_LAYERS,
    DROPOUT_RATE,
    MODEL_TYPE,
    USE_BIDIRECTIONAL,
    USE_ATTENTION,
    USE_BATCH_NORM,
    USE_ENHANCED_FEATURES,
    INCLUDE_VELOCITY,
    INCLUDE_ACCELERATION,
    INCLUDE_ROLLING_STATS,
    ROLLING_WINDOW_SIZE,
    USE_ENSEMBLE,
    ENSEMBLE_SIZE,
)
from utils.data_utils import (
    normalize_value,
    compute_velocity,
    compute_acceleration,
    compute_rolling_statistics,
)

logger = logging.getLogger(__name__)

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model paths – new layout keeps everything inside models/latest/
MODEL_PATH = os.path.join(MODELS_DIR, "latest", "model.pth")
ENCODER_PATH = os.path.join(MODELS_DIR, "latest", "encoder.npy")
NORM_PATH = os.path.join(MODELS_DIR, "latest", "normalization.npz")


def _load_model_checkpoint(path: str) -> tuple[dict, dict, int]:
    """Load model checkpoint with backward compatibility.

    Handles both the new format (dict with 'state_dict' + 'arch_params')
    and legacy format (bare state_dict).

    Returns:
        (state_dict, arch_params, input_size)
    """
    try:
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    except TypeError:
        # Older PyTorch without weights_only parameter
        checkpoint = torch.load(path, map_location=DEVICE)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        arch_params = checkpoint.get("arch_params", {})
    else:
        state_dict = checkpoint
        arch_params = {}

    # Infer input size
    # Check input_proj first (advanced model projects input before LSTM)
    if "input_size" in arch_params:
        input_size = int(arch_params["input_size"])
    elif "input_proj.weight" in state_dict:
        input_size = int(state_dict["input_proj.weight"].shape[1])
    elif "lstm.weight_ih_l0" in state_dict:
        input_size = int(state_dict["lstm.weight_ih_l0"].shape[1])
    else:
        raise ValueError(f"Cannot infer input size from model: {path}")

    return state_dict, arch_params, input_size


class LSTMGesturePredictor:
    """Real-time gesture prediction using LSTM"""

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        encoder_path: str = ENCODER_PATH,
        sequence_length: int = SEQUENCE_LENGTH,
        use_ensemble: bool | None = None,
    ):
        self.sequence_length = sequence_length
        self.buffer: deque[list[float]] = deque(maxlen=sequence_length)
        self.use_enhanced_features = USE_ENHANCED_FEATURES
        self.include_velocity = INCLUDE_VELOCITY
        self.include_acceleration = INCLUDE_ACCELERATION
        self.include_rolling_stats = INCLUDE_ROLLING_STATS
        self.rolling_window_size = ROLLING_WINDOW_SIZE
        self.use_ensemble = use_ensemble if use_ensemble is not None else USE_ENSEMBLE

        self.norm_mean = None
        self.norm_std = None
        # GPU-resident copies for the inference hot path; avoids NumPy↔tensor
        # round-trips on every predict() call.
        self._norm_mean_t: torch.Tensor | None = None
        self._norm_std_t: torch.Tensor | None = None
        if os.path.exists(NORM_PATH):
            try:
                norm_data = np.load(NORM_PATH)
                self.norm_mean = norm_data["mean"].astype(np.float32)
                self.norm_std = norm_data["std"].astype(np.float32)
                self.norm_std[self.norm_std == 0] = 1.0
            except Exception as e:
                logger.warning(f"Warning: Could not load normalization data: {e}")

        logger.info("Loading LSTM model...")
        self.classes = np.load(encoder_path, allow_pickle=True)
        num_classes = len(self.classes)

        model_dir = os.path.dirname(model_path)

        if self.use_ensemble:
            # Load ensemble models
            self.ensemble_models = []
            input_size = None
            for ensemble_idx in range(ENSEMBLE_SIZE):
                ensemble_model_path = os.path.join(
                    model_dir, f"model_{ensemble_idx}.pth"
                )
                if not os.path.exists(ensemble_model_path):
                    logger.warning(
                        f"Warning: Ensemble model {ensemble_idx} not found at {ensemble_model_path}"
                    )
                    continue

                state_dict, arch_params, input_size = _load_model_checkpoint(
                    ensemble_model_path
                )
                self.expected_features = self._select_features(input_size)

                resolved_type = arch_params.get("model_type", MODEL_TYPE)
                model = build_lstm_model(
                    input_size=input_size,
                    num_classes=num_classes,
                    hidden_size=arch_params.get("hidden_size", LSTM_UNITS),
                    num_layers=arch_params.get("num_layers", LSTM_LAYERS),
                    dropout_rate=arch_params.get("dropout_rate", DROPOUT_RATE),
                    device=DEVICE,
                    model_type=resolved_type,
                    bidirectional=arch_params.get("bidirectional", USE_BIDIRECTIONAL),
                    use_attention=arch_params.get("use_attention", USE_ATTENTION),
                    use_batch_norm=arch_params.get("use_batch_norm", USE_BATCH_NORM),
                )
                model.load_state_dict(state_dict)
                model.eval()
                self.ensemble_models.append(model)

            if not self.ensemble_models:
                raise ValueError(
                    "No ensemble models could be loaded. Train ensemble models first."
                )

            logger.info(f"Loaded {len(self.ensemble_models)} ensemble models")
            if len(self.ensemble_models) != ENSEMBLE_SIZE:
                logger.warning(
                    f"Warning: Expected {ENSEMBLE_SIZE} models but loaded {len(self.ensemble_models)}"
                )
        else:
            # Load single model
            state_dict, arch_params, input_size = _load_model_checkpoint(model_path)
            self.expected_features = self._select_features(input_size)

            resolved_type = arch_params.get("model_type", MODEL_TYPE)
            self.model = build_lstm_model(
                input_size=input_size,
                num_classes=num_classes,
                hidden_size=arch_params.get("hidden_size", LSTM_UNITS),
                num_layers=arch_params.get("num_layers", LSTM_LAYERS),
                dropout_rate=arch_params.get("dropout_rate", DROPOUT_RATE),
                device=DEVICE,
                model_type=resolved_type,
                bidirectional=arch_params.get("bidirectional", USE_BIDIRECTIONAL),
                use_attention=arch_params.get("use_attention", USE_ATTENTION),
                use_batch_norm=arch_params.get("use_batch_norm", USE_BATCH_NORM),
            )

            self.model.load_state_dict(state_dict)
            self.model.eval()

        # Validate normalization dimensions match model input
        if self.norm_mean is not None and len(self.norm_mean) != input_size:
            logger.warning(
                f"Warning: Normalization size ({len(self.norm_mean)}) does not match "
                f"model input size ({input_size}). Skipping z-score normalization."
            )
            self.norm_mean = None
            self.norm_std = None

        # Move normalization stats to the inference device once, so the hot path
        # can stay in tensor land (no per-prediction NumPy subtract/divide).
        if self.norm_mean is not None and self.norm_std is not None:
            self._norm_mean_t = torch.from_numpy(self.norm_mean).to(DEVICE)
            self._norm_std_t = torch.from_numpy(self.norm_std).to(DEVICE)

        # Compile the forward graph for steady-state latency reduction. First
        # call pays a ~3-10s warmup cost, so we trigger it eagerly with a dummy
        # input. torch.compile is a no-op on PyTorch builds without it.
        self._maybe_compile_models(input_size)

        logger.info(f"Model loaded: {len(self.classes)} classes")
        logger.info(f"Classes: {', '.join(self.classes)}")
        logger.info(f"Model type: {MODEL_TYPE}")
        logger.info(f"Sequence length: {sequence_length}")
        logger.info(f"Input size: {input_size} features")
        logger.info(f"Base sensors: {', '.join(self.expected_features)}")
        if self.use_enhanced_features:
            features_list = ["Base"]
            if self.include_velocity:
                features_list.append("Velocity")
            if self.include_acceleration:
                features_list.append("Acceleration")
            if self.include_rolling_stats:
                features_list.append("Rolling Stats")
            logger.info(f"Enhanced features: {' + '.join(features_list)}")
        if self.use_ensemble:
            logger.info(f"Using ensemble of {len(self.ensemble_models)} models")
        logger.info(f"Device: {DEVICE}")

        # Start at zero so the first full buffer can be predicted immediately.
        self.last_prediction_time = 0.0

    def _maybe_compile_models(self, input_size: int) -> None:
        flag = os.environ.get("SIGN2SPEECH_TORCH_COMPILE", "1").strip().lower()
        if flag in {"0", "false", "no", "off"}:
            return

        compile_fn = getattr(torch, "compile", None)
        if compile_fn is None:
            return

        if importlib.util.find_spec("triton") is None:
            logger.info("torch.compile skipped (Triton not available)")
            return

        try:
            if self.use_ensemble:
                self.ensemble_models = [
                    compile_fn(m, mode="reduce-overhead") for m in self.ensemble_models
                ]
            else:
                self.model = compile_fn(self.model, mode="reduce-overhead")

            # Warmup: run one forward pass so the compile cost is paid up-front
            # rather than on the first user gesture.
            dummy = torch.zeros(
                (1, self.sequence_length, input_size),
                dtype=torch.float32,
                device=DEVICE,
            )
            with torch.no_grad():
                if self.use_ensemble:
                    for m in self.ensemble_models:
                        m(dummy)
                else:
                    self.model(dummy)
            logger.info("torch.compile warmup complete")
        except Exception as e:
            logger.warning(
                f"torch.compile unavailable or failed; running uncompiled: {e}"
            )

    def _select_features(self, input_size: int) -> list[str]:
        base_features = [
            "flex0",
            "flex1",
            "flex2",
            "flex3",
            "flex4",
            "accelX",
            "accelY",
            "accelZ",
            "gyroX",
            "gyroY",
            "gyroZ",
        ]
        n_base = len(base_features)  # 11

        # Determine expected size based on enabled enhanced features
        # base(11) + velocity(11) + acceleration(11) + rolling_mean(11) + rolling_std(11)
        expected = n_base
        if self.include_velocity:
            expected += n_base
        if self.include_acceleration:
            expected += n_base
        if self.include_rolling_stats:
            expected += n_base * 2  # mean + std

        if input_size == expected:
            return base_features
        if input_size == 33:  # base + vel + accel (no rolling)
            return base_features
        if input_size == 22:  # base + vel or base + accel
            return base_features
        if input_size == 11:
            return base_features
        if input_size == 6:
            return ["accelX", "accelY", "accelZ", "gyroX", "gyroY", "gyroZ"]
        if input_size == 5:
            return ["flex0", "flex1", "flex2", "flex3", "flex4"]
        raise ValueError(
            f"Unsupported input size {input_size}. Expected one of: 5, 6, 11, 22, 33, {expected}.\n"
            "Model was likely trained with enhanced features. "
            "Check USE_ENHANCED_FEATURES / INCLUDE_ROLLING_STATS in config.py"
        )

    def add_sensor_dict(self, sensor_dict: dict[str, float]):
        """Add a complete sensor reading (all 11 values at once)"""
        normalized_sample = {}
        for name in self.expected_features:
            if name not in sensor_dict:
                return
            normalized = normalize_value(name, sensor_dict[name])
            if normalized is None:
                return
            normalized_sample[name] = normalized

        feature_vector = np.array(
            [normalized_sample[feat] for feat in self.expected_features],
            dtype=np.float32,
        )

        self.buffer.append(feature_vector.tolist())

    def can_predict(self):
        current_time = time.time()
        has_data = len(self.buffer) == self.sequence_length
        time_elapsed = (current_time - self.last_prediction_time) >= PREDICTION_INTERVAL

        return has_data and time_elapsed

    def predict(self):
        if not self.can_predict():
            return None, None, None, None

        sequence = np.array(list(self.buffer), dtype=np.float32)

        if self.use_enhanced_features and (
            self.include_velocity
            or self.include_acceleration
            or self.include_rolling_stats
        ):
            enhanced_sequence = [sequence]

            if self.include_velocity:
                velocity = compute_velocity(sequence)
                enhanced_sequence.append(velocity)

            if self.include_acceleration:
                acceleration = compute_acceleration(sequence)
                enhanced_sequence.append(acceleration)

            if self.include_rolling_stats:
                stats = compute_rolling_statistics(
                    sequence, window_size=self.rolling_window_size
                )
                enhanced_sequence.append(stats["mean"])
                enhanced_sequence.append(stats["std"])

            sequence = np.concatenate(enhanced_sequence, axis=1)

        sequence_tensor = (
            torch.from_numpy(np.ascontiguousarray(sequence)).unsqueeze(0).to(DEVICE)
        )
        if self._norm_mean_t is not None:
            sequence_tensor = (sequence_tensor - self._norm_mean_t) / self._norm_std_t

        if self.use_ensemble:
            # Ensemble prediction: average probabilities across all models
            ensemble_probabilities = None

            with torch.no_grad():
                for model in self.ensemble_models:
                    outputs = model(sequence_tensor)
                    probabilities = torch.softmax(outputs, dim=1)

                    if ensemble_probabilities is None:
                        ensemble_probabilities = probabilities.clone()
                    else:
                        ensemble_probabilities += probabilities

            # Average the probabilities
            ensemble_probabilities /= len(self.ensemble_models)
            confidence, predicted_class_idx = torch.max(ensemble_probabilities, 1)

            predicted_class_idx = predicted_class_idx.item()
            confidence = confidence.item()

            # Calculate confidence gap (difference between top and second-best)
            sorted_probs, _ = torch.sort(ensemble_probabilities[0], descending=True)
            confidence_gap = (sorted_probs[0] - sorted_probs[1]).item()

            # Get all probabilities as dict
            all_probs = {
                self.classes[i]: ensemble_probabilities[0][i].item()
                for i in range(len(self.classes))
            }
        else:
            # Single model prediction
            with torch.no_grad():
                outputs = self.model(sequence_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted_class_idx = torch.max(probabilities, 1)

                predicted_class_idx = predicted_class_idx.item()
                confidence = confidence.item()

                # Calculate confidence gap (difference between top and second-best)
                sorted_probs, _ = torch.sort(probabilities[0], descending=True)
                confidence_gap = (sorted_probs[0] - sorted_probs[1]).item()

                # Get all probabilities as dict
                all_probs = {
                    self.classes[i]: probabilities[0][i].item()
                    for i in range(len(self.classes))
                }

        predicted_gesture = self.classes[predicted_class_idx]

        self.last_prediction_time = time.time()

        return predicted_gesture, confidence, confidence_gap, all_probs
