import importlib.util
import logging
import os
import threading
import time
from collections import deque

import numpy as np
import torch

from config.architecture import architecture
from config.config import MODELS_DIR
from core.inference.onnx_predictor import ONNXBackend
from core.models.model_factory import build_model_from_checkpoint
from utils.data_utils import QUATERNION_FEATURE_NAMES
from utils.data_utils import MadgwickFilter
from utils.data_utils import align_sequence_to_templates
from utils.data_utils import compute_acceleration
from utils.data_utils import compute_rolling_statistics
from utils.data_utils import compute_velocity
from utils.data_utils import load_dtw_templates
from utils.data_utils import normalize_value
from utils.data_utils import resample_sequence

try:
    PREDICTION_CLASS_THRESHOLDS = architecture.prediction.prediction_class_thresholds
except AttributeError:  # pragma: no cover - backward compatibility for old configs
    PREDICTION_CLASS_THRESHOLDS = {}

logger = logging.getLogger(__name__)

# Model paths – new layout keeps everything inside models/latest/
MODEL_PATH = os.path.join(MODELS_DIR, "latest", "model.pth")
ENCODER_PATH = os.path.join(MODELS_DIR, "latest", "encoder.npy")
NORM_PATH = os.path.join(MODELS_DIR, "latest", "normalization.npz")

_SENSOR_NAME_ALIASES = {
    "accelx": "accelX",
    "accel_x": "accelX",
    "accelerometer_x": "accelX",
    "accely": "accelY",
    "accel_y": "accelY",
    "accelerometer_y": "accelY",
    "accelz": "accelZ",
    "accel_z": "accelZ",
    "accelerometer_z": "accelZ",
    "gyrox": "gyroX",
    "gyro_x": "gyroX",
    "gyroscope_x": "gyroX",
    "gyroy": "gyroY",
    "gyro_y": "gyroY",
    "gyroscope_y": "gyroY",
    "gyroz": "gyroZ",
    "gyro_z": "gyroZ",
    "gyroscope_z": "gyroZ",
    "quatw": "quatW",
    "quat_w": "quatW",
    "quaternion_w": "quatW",
    "quatx": "quatX",
    "quat_x": "quatX",
    "quaternion_x": "quatX",
    "quaty": "quatY",
    "quat_y": "quatY",
    "quaternion_y": "quatY",
    "quatz": "quatZ",
    "quat_z": "quatZ",
    "quaternion_z": "quatZ",
}


def _canonical_sensor_name(name: str) -> str:
    """Map common snake_case/lowercase sensor keys to internal camelCase names."""
    key = name.strip().replace("-", "_").replace(" ", "_")
    lowered = key.lower()
    if lowered in _SENSOR_NAME_ALIASES:
        return _SENSOR_NAME_ALIASES[lowered]
    if lowered.startswith("flex_") and lowered[5:].isdigit():
        return f"flex{int(lowered[5:])}"
    if lowered.startswith("sensor_") and lowered[7:].isdigit():
        return f"flex{int(lowered[7:])}"
    if lowered.startswith("flex") and lowered[4:].isdigit():
        return f"flex{int(lowered[4:])}"
    return key


def _normalize_class_thresholds(
    thresholds: object,
) -> dict[str, float]:
    if not isinstance(thresholds, dict):
        return {}
    normalized: dict[str, float] = {}
    for key, value in thresholds.items():
        try:
            normalized[str(key).strip().upper()] = float(value)
        except (TypeError, ValueError):
            continue
    return normalized


class LSTMGesturePredictor:
    """Real-time gesture prediction using LSTM"""

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        encoder_path: str = ENCODER_PATH,
        sequence_length: int = architecture.training.sequence_length,
        use_ensemble: bool | None = None,
        device: torch.device | None = None,
    ):
        self._device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.sequence_length = sequence_length
        self.buffer: deque[list[float]] = deque(maxlen=sequence_length)
        self._lock = threading.Lock()
        self.use_enhanced_features = architecture.model.use_enhanced_features
        self.include_velocity = architecture.model.include_velocity
        self.include_acceleration = architecture.model.include_acceleration
        self.include_rolling_stats = architecture.model.include_rolling_stats
        self.rolling_window_size = architecture.model.rolling_window_size
        self.use_ensemble = use_ensemble if use_ensemble is not None else architecture.training.use_ensemble
        self._class_confidence_thresholds = _normalize_class_thresholds(
            PREDICTION_CLASS_THRESHOLDS
        )
        # Create Madgwick filter only when enabled via config to allow disabling
        # the yaw/orientation normalization for latency-sensitive deployments.
        self._madgwick_filter = MadgwickFilter() if architecture.motion_detection.enable_madgwick else None
        self._dtw_templates: dict[str, np.ndarray] = {}
        self._last_dtw_template: str | None = None
        self._last_dtw_distance: float | None = None
        self.use_onnx = str(model_path).lower().endswith(".onnx")
        self._onnx_backend: ONNXBackend | None = None

        self.norm_mean = None
        self.norm_std = None
        # GPU-resident copies for the inference hot path; avoids NumPy↔tensor
        # round-trips on every predict() call.
        self._norm_mean_t: torch.Tensor | None = None
        self._norm_std_t: torch.Tensor | None = None

        model_dir = os.path.dirname(os.path.abspath(model_path))
        encoder_candidate = os.path.join(model_dir, "encoder.npy")
        if os.path.exists(encoder_candidate):
            encoder_path = encoder_candidate

        norm_path = os.path.join(model_dir, "normalization.npz")
        if not os.path.exists(norm_path):
            norm_path = NORM_PATH

        if os.path.exists(norm_path):
            try:
                norm_data = np.load(norm_path)
                self.norm_mean = norm_data["mean"].astype(np.float32)
                self.norm_std = norm_data["std"].astype(np.float32)
                self.norm_std[self.norm_std == 0] = 1.0
            except Exception as e:
                logger.warning(f"Warning: Could not load normalization data: {e}")

        logger.info("Loading LSTM model...")
        self.classes = np.load(encoder_path, allow_pickle=True)
        num_classes = len(self.classes)

        if self.use_onnx:
            self.use_ensemble = False
            self._onnx_backend = ONNXBackend(
                model_path, self._device, norm_mean=self.norm_mean
            )
            input_size = self._onnx_backend.input_size
            self.expected_features = self._select_features(input_size)
        elif self.use_ensemble:
            # Load ensemble models
            self.ensemble_models = []

            # Auto-detect ensemble models in the directory
            ensemble_idx = 0
            while True:
                ensemble_model_path = os.path.join(
                    model_dir, f"model_{ensemble_idx}.pth"
                )
                if not os.path.exists(ensemble_model_path):
                    break

                model, arch_params, input_size, inferred_num_classes = (
                    build_model_from_checkpoint(
                        ensemble_model_path,
                        self._device,
                        encoder_num_classes=num_classes,
                        hidden_size=architecture.model.lstm_units,
                        num_layers=architecture.model.lstm_layers,
                        dropout_rate=architecture.model.dropout_rate,
                        model_type=architecture.model.model_type,
                        bidirectional=architecture.model.use_bidirectional,
                        use_attention=architecture.model.use_attention,
                        use_batch_norm=architecture.model.use_batch_norm,
                        use_cnn=False,
                    )
                )
                self.expected_features = self._select_features(input_size)

                # Priority: inferred_num_classes > arch_params > encoder count
                resolved_num_classes = (
                    inferred_num_classes or arch_params.get("num_classes") or num_classes
                )

                if resolved_num_classes != num_classes:
                    logger.warning(
                        f"Warning: Model {ensemble_idx} expects {resolved_num_classes} classes but "
                        f"encoder has {num_classes}. Adjusting classes list to {resolved_num_classes}."
                    )
                    if ensemble_idx == 0:
                        if resolved_num_classes < num_classes:
                            self.classes = self.classes[:resolved_num_classes]
                        else:
                            extra = resolved_num_classes - num_classes
                            new_names = [f"unknown_{i}" for i in range(extra)]
                            self.classes = np.concatenate([self.classes, new_names])
                        # Update num_classes for subsequent ensemble members check
                        num_classes = len(self.classes)

                self.ensemble_models.append(model)
                ensemble_idx += 1

            if not self.ensemble_models:
                raise ValueError(
                    f"No ensemble models (model_0.pth, etc.) found in {model_dir}. Train ensemble models first."
                )

            logger.info(f"Loaded {len(self.ensemble_models)} ensemble models")
        else:
            # Load single model — single call to factory returns model + metadata
            self.model, arch_params, input_size, inferred_num_classes = (
                build_model_from_checkpoint(
                    model_path,
                    self._device,
                    encoder_num_classes=num_classes,
                    hidden_size=architecture.model.lstm_units,
                    num_layers=architecture.model.lstm_layers,
                    dropout_rate=architecture.model.dropout_rate,
                    model_type=architecture.model.model_type,
                    bidirectional=architecture.model.use_bidirectional,
                    use_attention=architecture.model.use_attention,
                    use_batch_norm=architecture.model.use_batch_norm,
                    use_cnn=False,
                )
            )
            self.expected_features = self._select_features(input_size)

            # Priority: inferred_num_classes > arch_params > encoder count
            resolved_num_classes = (
                inferred_num_classes or arch_params.get("num_classes") or num_classes
            )

            if resolved_num_classes != num_classes:
                logger.warning(
                    f"Warning: Model expects {resolved_num_classes} classes but "
                    f"encoder has {num_classes}. Adjusting classes list to {resolved_num_classes}."
                )
                if resolved_num_classes < num_classes:
                    self.classes = self.classes[:resolved_num_classes]
                else:
                    extra = resolved_num_classes - num_classes
                    new_names = [f"unknown_{i}" for i in range(extra)]
                    self.classes = np.concatenate([self.classes, new_names])

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
            self._norm_mean_t = torch.from_numpy(self.norm_mean).to(self._device)
            self._norm_std_t = torch.clamp(
                torch.from_numpy(self.norm_std).to(self._device),
                min=1e-6,
            )

        self._dtw_templates = load_dtw_templates(model_dir)
        if self._dtw_templates:
            logger.info("Loaded %d DTW class template(s)", len(self._dtw_templates))

        # Compile the forward graph for steady-state latency reduction. First
        # call pays a ~3-10s warmup cost, so we trigger it eagerly with a dummy
        # input. torch.compile is a no-op on PyTorch builds without it.
        if not self.use_onnx:
            self._maybe_compile_models(input_size)

        logger.info(f"Model loaded: {len(self.classes)} classes")
        logger.info(f"Classes: {', '.join(self.classes)}")
        logger.info(f"Model type: {architecture.model.model_type}")
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
        if self.use_onnx:
            logger.info("Using ONNX Runtime inference")
        logger.info(f"Device: {self._device}")

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
                device=self._device,
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
        sensor_features = [
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
        quaternion_features = sensor_features + QUATERNION_FEATURE_NAMES

        def enhanced_size(feature_count: int) -> int:
            expected_size = feature_count
            if self.include_velocity:
                expected_size += feature_count
            if self.include_acceleration:
                expected_size += feature_count
            if self.include_rolling_stats:
                expected_size += feature_count * 2
            return expected_size

        # Determine expected size based on enabled enhanced features
        # base + velocity + acceleration + rolling_mean + rolling_std
        expected = enhanced_size(len(sensor_features))
        expected_with_quat = enhanced_size(len(quaternion_features))

        if input_size == expected_with_quat:
            return quaternion_features
        if input_size == expected:
            return sensor_features
        if input_size == 75:  # 15 * (base + vel + accel + rolling mean/std)
            return quaternion_features
        if input_size == 45:  # 15 * (base + two enhanced groups)
            return quaternion_features
        if input_size == 30:  # 15 * (base + one enhanced group)
            return quaternion_features
        if input_size == 15:
            return quaternion_features
        if input_size == 33:  # base + vel + accel (no rolling)
            return sensor_features
        if input_size == 22:  # base + vel or base + accel
            return sensor_features
        if input_size == 11:
            return sensor_features
        if input_size == 6:
            return ["accelX", "accelY", "accelZ", "gyroX", "gyroY", "gyroZ"]
        if input_size == 5:
            return ["flex0", "flex1", "flex2", "flex3", "flex4"]
        if input_size == 4:
            return QUATERNION_FEATURE_NAMES
        raise ValueError(
            f"Unsupported input size {input_size}. Expected one of: "
            f"4, 5, 6, 11, 15, 22, 30, 33, 45, 75, {expected}, {expected_with_quat}.\n"
            "Model was likely trained with enhanced features. "
            "Check architecture.model.use_enhanced_features / architecture.model.include_rolling_stats in config.py"
        )

    def _sanitize_sensor_dict(self, sensor_dict: dict[str, float]) -> dict[str, float]:
        sanitized: dict[str, float] = {}
        for raw_name, raw_value in sensor_dict.items():
            try:
                sanitized[_canonical_sensor_name(str(raw_name))] = float(raw_value)
            except (TypeError, ValueError):
                continue
        return sanitized

    def _add_quaternion_features(self, sensor_dict: dict[str, float]) -> None:
        if not any(name in self.expected_features for name in QUATERNION_FEATURE_NAMES):
            return
        if all(name in sensor_dict for name in QUATERNION_FEATURE_NAMES):
            return

        # If the Madgwick filter is disabled, skip quaternion computation entirely.
        if self._madgwick_filter is None:
            return

        required = ["accelX", "accelY", "accelZ", "gyroX", "gyroY", "gyroZ"]
        if any(name not in sensor_dict for name in required):
            return

        accel = np.array([sensor_dict[name] for name in required[:3]], dtype=np.float32)
        gyro_raw = np.array([sensor_dict[name] for name in required[3:]], dtype=np.float32)
        gyro_dps = gyro_raw / max(float(architecture.hardware.max_gyro_value), 1.0) * 2000.0
        quaternion = self._madgwick_filter.update_imu(
            accel,
            gyro_dps,
            gyro_degrees=True,
        )
        for name, value in zip(QUATERNION_FEATURE_NAMES, quaternion):
            sensor_dict.setdefault(name, float(value))

    def _normalize_feature(self, name: str, value: float) -> float | None:
        if name in QUATERNION_FEATURE_NAMES:
            return float(np.clip(value, -1.0, 1.0))
        return normalize_value(name, value)

    def add_sensor_dict(self, sensor_dict: dict[str, float]):
        """Add a complete sensor reading (all 11 values at once)"""
        sanitized = self._sanitize_sensor_dict(sensor_dict)

        with self._lock:
            self._add_quaternion_features(sanitized)
            normalized_sample = {}
            for name in self.expected_features:
                if name not in sanitized:
                    return
                normalized = self._normalize_feature(name, sanitized[name])
                if normalized is None:
                    return
                normalized_sample[name] = normalized

            feature_vector = np.array(
                [normalized_sample[feat] for feat in self.expected_features],
                dtype=np.float32,
            )

            self.buffer.append(feature_vector.tolist())

    def _can_predict_unlocked(self):
        current_time = time.time()
        has_data = len(self.buffer) == self.sequence_length
        time_elapsed = (current_time - self.last_prediction_time) >= architecture.prediction.prediction_interval

        return has_data and time_elapsed

    def can_predict(self):
        with self._lock:
            return self._can_predict_unlocked()

    def _apply_dtw_alignment(self, sequence: np.ndarray) -> np.ndarray:
        if not self._dtw_templates:
            return sequence

        aligned, label, distance = align_sequence_to_templates(
            sequence,
            self._dtw_templates,
            target_length=self.sequence_length,
        )
        self._last_dtw_template = label
        self._last_dtw_distance = distance
        if aligned.shape[0] != self.sequence_length:
            aligned = resample_sequence(aligned, self.sequence_length)
        return aligned.astype(np.float32)

    def _confidence_threshold_for(self, gesture: str) -> float:
        thresholds = self._class_confidence_thresholds
        if not thresholds:
            return 0.0
        key = str(gesture).strip().upper()
        return float(
            thresholds.get(
                key,
                thresholds.get("DEFAULT", thresholds.get("*", architecture.prediction.confidence_threshold)),
            )
        )

    def _probability_result(
        self,
        probabilities: torch.Tensor,
    ) -> tuple[int, float, float, dict[str, float]]:
        confidence_t, predicted_class_idx_t = torch.max(probabilities, 1)
        predicted_class_idx = int(predicted_class_idx_t.item())
        confidence = float(confidence_t.item())
        sorted_probs, _ = torch.sort(probabilities[0], descending=True)
        if len(sorted_probs) > 1:
            confidence_gap = float((sorted_probs[0] - sorted_probs[1]).item())
        else:
            confidence_gap = float(sorted_probs[0].item())
        all_probs = {
            str(self.classes[i]): float(probabilities[0][i].item())
            for i in range(len(self.classes))
        }
        return predicted_class_idx, confidence, confidence_gap, all_probs

    def _predict_onnx(
        self,
        sequence_tensor: torch.Tensor,
    ) -> tuple[int, float, float, dict[str, float]]:
        if self._onnx_backend is None:
            raise RuntimeError("ONNX backend is not initialized")
        probabilities = self._onnx_backend.predict(sequence_tensor)
        return self._probability_result(probabilities)

    def predict(self):
        """Predict on the current buffer and emit timing diagnostics for profiling."""
        t0 = time.perf_counter()
        with self._lock:
            if not self._can_predict_unlocked():
                return None, None, None, None
            sequence = np.array(list(self.buffer), dtype=np.float32)

        t_after_lock = time.perf_counter()
        t_enhance = 0.0
        t_dtw = 0.0
        t_tensor = 0.0
        t_infer = 0.0
        t_post = 0.0

        # Enhanced features (velocity/accel/rolling)
        if self.use_enhanced_features and (
            self.include_velocity
            or self.include_acceleration
            or self.include_rolling_stats
        ):
            t_a = time.perf_counter()
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
            t_enhance = time.perf_counter() - t_a

        # DTW alignment
        t_b = time.perf_counter()
        sequence = self._apply_dtw_alignment(sequence)
        t_dtw = time.perf_counter() - t_b

        # Convert to tensor and normalize
        t_c = time.perf_counter()
        sequence_tensor = (
            torch.from_numpy(np.ascontiguousarray(sequence)).unsqueeze(0).to(self._device)
        )
        if self._norm_mean_t is not None:
            sequence_tensor = (sequence_tensor - self._norm_mean_t) / self._norm_std_t
        t_tensor = time.perf_counter() - t_c

        # Inference (ONNX / Ensemble / Single model)
        t_d = time.perf_counter()
        if self.use_onnx:
            predicted_class_idx, confidence, confidence_gap, all_probs = (
                self._predict_onnx(sequence_tensor)
            )
        elif self.use_ensemble:
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
            predicted_class_idx, confidence, confidence_gap, all_probs = (
                self._probability_result(ensemble_probabilities)
            )
        else:
            # Single model prediction
            with torch.no_grad():
                outputs = self.model(sequence_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class_idx, confidence, confidence_gap, all_probs = (
                    self._probability_result(probabilities)
                )
        t_infer = time.perf_counter() - t_d

        # Post-processing
        t_e = time.perf_counter()
        predicted_gesture = str(self.classes[predicted_class_idx])
        confidence_threshold = self._confidence_threshold_for(predicted_gesture)

        with self._lock:
            self.last_prediction_time = time.time()
        t_post = time.perf_counter() - t_e

        # Diagnostics
        total = time.perf_counter() - t0
        logger.debug(
            "Prediction timing (ms): total=%.2f, lock=%.2f, enhance=%.2f, dtw=%.2f, tensor=%.2f, infer=%.2f, post=%.2f",
            total * 1000.0,
            (t_after_lock - t0) * 1000.0,
            t_enhance * 1000.0,
            t_dtw * 1000.0,
            t_tensor * 1000.0,
            t_infer * 1000.0,
            t_post * 1000.0,
        )

        if confidence < confidence_threshold:
            logger.debug(
                "Filtered %s: confidence %.3f < class threshold %.3f",
                predicted_gesture,
                confidence,
                confidence_threshold,
            )
            return None, confidence, confidence_gap, all_probs

        return predicted_gesture, confidence, confidence_gap, all_probs
