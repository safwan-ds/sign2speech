"""
Architecture configuration loader for Sign2Speech.

Provides a frozen dataclass hierarchy loaded from ``architecture.yaml``,
with fail-fast validation on import.  The module-level ``architecture``
singleton is the single source of truth for all architecture constants.

Usage::

    from config.architecture import architecture

    print(architecture.model.lstm_units)   # 64
    print(architecture.training.batch_size)  # 32
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from dataclasses import field
from typing import Any

import yaml

# ── Path resolution ──────────────────────────────────────────────────────────
_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_YAML_PATH = os.path.join(_CONFIG_DIR, "architecture.yaml")


# ═════════════════════════════════════════════════════════════════════════════
# Nested frozen dataclasses — one per YAML top-level section
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class HardwareConfig:
    """Serial port, flex sensor, and IMU hardware constants."""

    com_port: str = "COM9"
    baud_rate: int = 115200
    timeout: int = 3
    serial_connection_delay: int = 2
    num_flex_sensors: int = 5
    num_imu_axes: int = 6
    expected_sensor_count: int = 11
    flex_sensor_ranges: dict[int, list[int]] = field(
        default_factory=lambda: {i: [25, 300] for i in range(5)}
    )
    flex_sensor_calibration: dict[int, list[int]] = field(
        default_factory=lambda: {0: [28, 224], 1: [56, 293], 2: [39, 240], 3: [53, 239], 4: [44, 261]}
    )
    flex_sensor_default_range: list[int] = field(default_factory=lambda: [0, 1023])
    min_accel_value: int = -32768
    max_accel_value: int = 32767
    min_gyro_value: int = -32768
    max_gyro_value: int = 32767

    def __post_init__(self) -> None:
        for idx, rng in self.flex_sensor_ranges.items():
            if not isinstance(idx, int):
                raise ValueError(
                    f"flex_sensor_ranges keys must be int, got {type(idx).__name__}"
                )
            if len(rng) != 2:
                raise ValueError(
                    f"flex_sensor_ranges[{idx}] must have exactly 2 values, got {len(rng)}"
                )
        if self.num_flex_sensors <= 0:
            raise ValueError("num_flex_sensors must be positive")
        if self.num_imu_axes <= 0:
            raise ValueError("num_imu_axes must be positive")
        if self.baud_rate <= 0:
            raise ValueError("baud_rate must be positive")


@dataclass
class MotionDetectionConfig:
    """Motion / gesture-detection parameters."""

    detect_gesture_motion: bool = True
    motion_threshold: float = 0.02
    motion_detection_min_duration: int = 5
    motion_detection_smoothing_window: int = 2
    sequence_overlap: float = 0.1
    motion_padding_ratio: float = 0.2
    enable_madgwick: bool = True

    def __post_init__(self) -> None:
        if self.motion_threshold < 0:
            raise ValueError("motion_threshold must be non-negative")
        if self.motion_detection_min_duration <= 0:
            raise ValueError("motion_detection_min_duration must be positive")


@dataclass
class ModelConfig:
    """LSTM / neural-network architecture parameters."""

    lstm_units: int = 64
    lstm_layers: int = 2
    dropout_rate: float = 0.4
    model_type: str = "enhanced"
    use_bidirectional: bool = True
    use_attention: bool = True
    use_batch_norm: bool = True
    use_enhanced_features: bool = False
    include_velocity: bool = True
    include_acceleration: bool = True
    include_rolling_stats: bool = True
    rolling_window_size: int = 5

    def __post_init__(self) -> None:
        if self.lstm_units <= 0:
            raise ValueError("lstm_units must be positive")
        if self.lstm_layers <= 0:
            raise ValueError("lstm_layers must be positive")
        if not 0.0 <= self.dropout_rate <= 1.0:
            raise ValueError("dropout_rate must be between 0 and 1")
        if self.rolling_window_size <= 0:
            raise ValueError("rolling_window_size must be positive")


@dataclass
class TrainingConfig:
    """Training hyper-parameters."""

    batch_size: int = 32
    epochs: int = 150
    learning_rate: float = 0.0005
    sequence_length: int = 20
    weight_decay: float = 0.0005
    use_weighted_loss: bool = True
    use_label_smoothing: bool = True
    label_smoothing_factor: float = 0.1
    use_cosine_annealing: bool = True
    cosine_t_0: int = 10
    cosine_t_mult: int = 2
    cosine_eta_min: float = 0.000001
    lr_plateau_factor: float = 0.5
    lr_plateau_patience: int = 5
    lr_plateau_min: float = 0.000001
    use_warmup: bool = True
    warmup_epochs: int = 5
    warmup_start_factor: float = 0.1
    early_stopping_patience: int = 15
    min_delta: float = 0.0001
    gradient_clip_value: float = 1.0
    min_validation_samples_per_class: int = 5
    use_ensemble: bool = False
    ensemble_size: int = 3
    random_state: int = 42
    use_test_split: bool = False
    test_size: float = 0.1
    test_data_split_percentage: float = 0.1
    min_stratify_samples: int = 2
    default_validation_size: float = 0.1

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if not 0 < self.learning_rate < 1:
            raise ValueError("learning_rate must be between 0 and 1")
        if not 0.0 <= self.min_delta <= 1.0:
            raise ValueError("min_delta must be between 0 and 1")
        if self.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be positive")


@dataclass
class AugmentationConfig:
    """Data-augmentation parameters."""

    use_augmentation: bool = True
    augmentation_factor: int = 2
    augmentation_prob: float = 0.7
    num_augmentations_per_sample: int = 2
    time_warp_sigma: float = 0.2
    time_warp_knot: int = 4
    magnitude_warp_sigma: float = 0.2
    magnitude_warp_knot: int = 4
    noise_level: float = 0.01
    scale_range: list[float] = field(default_factory=lambda: [0.9, 1.1])
    time_shift_range: float = 0.1
    rotation_max_angle: int = 10

    def __post_init__(self) -> None:
        if len(self.scale_range) != 2:
            raise ValueError("scale_range must have exactly 2 values")
        if self.scale_range[0] >= self.scale_range[1]:
            raise ValueError("scale_range[0] must be < scale_range[1]")
        if not 0.0 <= self.augmentation_prob <= 1.0:
            raise ValueError("augmentation_prob must be between 0 and 1")
        if self.augmentation_factor <= 0:
            raise ValueError("augmentation_factor must be positive")


@dataclass
class PredictionConfig:
    """Real-time gesture prediction parameters."""

    prediction_interval: float = 0.08
    prediction_motion_threshold: int = 1000
    confidence_threshold: float = 0.74
    prediction_class_thresholds: dict[str, float] = field(
        default_factory=lambda: {"DEFAULT": 0.74}
    )
    prediction_consensus_frames: int = 5
    prediction_avg_motion_threshold: int = 600
    prediction_motion_variance_min: int = 150
    prediction_significant_motion_min_ratio: float = 0.35
    prediction_min_confidence_gap: float = 0.15
    prediction_debug_mode: bool = False
    min_consecutive_rest: int = 5
    min_gestures_for_llm: int = 2
    prediction_switch_consensus_frames: int = 3
    prediction_initial_consensus_frames: int = 2
    prediction_keep_last_stable_frames: int = 2
    prediction_uncertain_token: str = "UNKNOWN"
    prediction_rest_weight: float = 0.75
    enable_sequence_decoder: bool = False
    sequence_decoder_switch_penalty: float = 0.0
    sequence_decoder_rest_switch_penalty: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if self.prediction_interval <= 0:
            raise ValueError("prediction_interval must be positive")
        if len(self.prediction_class_thresholds) == 0:
            raise ValueError("prediction_class_thresholds must not be empty")


@dataclass
class NormalizationConfig:
    """Sensor data normalization range."""

    norm_min: float = 0.0
    norm_max: float = 1.0

    def __post_init__(self) -> None:
        if self.norm_min >= self.norm_max:
            raise ValueError("norm_min must be < norm_max")


@dataclass
class GeneralConfig:
    """General feature toggles not fitting other categories."""

    use_tts: bool = True


@dataclass
class LLMConfig:
    """Large Language Model configuration (local Qwen + remote fallback)."""

    use_qwen_llm: bool = True
    qwen_model_filename: str = "qwen2.5-7b-instruct-q6_k-00001-of-00002.gguf"
    qwen_n_ctx: int = 2048
    qwen_n_gpu_layers: int = -1
    qwen_n_batch: int = 512
    qwen_force_gpu: bool = True
    qwen_max_tokens: int = 64
    qwen_inference_temperature: float = 0.25
    llm_backend: str = "local"
    llm_remote_url: str = "https://api.deepseek.com"
    llm_remote_api_key: str = ""
    llm_remote_model: str = "deepseek-v4-flash"
    llm_remote_timeout: float = 15.0
    llm_remote_format: str = "chat"
    llm_remote_max_tokens: int = 1024

    def __post_init__(self) -> None:
        valid_backends = ("local", "remote")
        if self.llm_backend not in valid_backends:
            raise ValueError(
                f"llm_backend must be one of {valid_backends}, got {self.llm_backend!r}"
            )
        valid_formats = ("chat", "completions")
        if self.llm_remote_format not in valid_formats:
            raise ValueError(
                f"llm_remote_format must be one of {valid_formats}, "
                f"got {self.llm_remote_format!r}"
            )


@dataclass
class GUIConfig:
    """GUI window / layout constants."""

    keyboard_debounce_delay: float = 0.3
    keyboard_poll_interval: float = 0.05
    gui_min_width: int = 900
    gui_min_height: int = 600
    gui_padding: int = 20
    gui_small_padding: int = 10
    gui_font_size: int = 12
    gui_title_font_size: int = 14
    gui_review_batch_size: int = 4
    gui_plot_rows_calc: int = 14
    gui_plot_height_multiplier: int = 3
    gui_plot_hspace: float = 0.45
    gui_plot_wspace: float = 0.30
    gui_plot_top: float = 0.95
    gui_plot_bottom: float = 0.05
    gui_thread_sleep: float = 0.01
    data_manager_window_width: int = 1560
    data_manager_window_height: int = 940
    data_manager_min_width: int = 1220
    data_manager_min_height: int = 760
    gestures_editor_dialog_width: int = 600
    gestures_editor_dialog_height: int = 420
    default_ui_language: str = "tr"
    supported_ui_languages: list[str] = field(default_factory=lambda: ["tr", "en"])

    def __post_init__(self) -> None:
        if self.gui_min_width <= 0:
            raise ValueError("gui_min_width must be positive")
        if self.gui_min_height <= 0:
            raise ValueError("gui_min_height must be positive")
        if len(self.supported_ui_languages) == 0:
            raise ValueError("supported_ui_languages must not be empty")


@dataclass
class EvaluationConfig:
    """Evaluation / metrics plot constants."""

    evaluation_dpi: int = 300
    confusion_matrix_figsize: list[int] = field(default_factory=lambda: [10, 8])
    roc_curve_figsize: list[int] = field(default_factory=lambda: [10, 8])
    evaluation_class_weight_epsilon: float = 0.000001

    def __post_init__(self) -> None:
        if len(self.confusion_matrix_figsize) != 2:
            raise ValueError("confusion_matrix_figsize must have exactly 2 values")
        if len(self.roc_curve_figsize) != 2:
            raise ValueError("roc_curve_figsize must have exactly 2 values")


@dataclass
class PlotConfig:
    """General-purpose matplotlib plot constants."""

    plot_figure_width: int = 12
    plot_figure_height: int = 10
    plot_num_rows: int = 3
    plot_num_cols: int = 1
    plot_font_size: int = 14
    plot_marker_size: int = 2
    plot_grid_alpha: float = 0.3


# ═════════════════════════════════════════════════════════════════════════════
# Top-level container
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class ArchitectureConfig:
    """Root configuration — one attribute per YAML section."""

    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    motion_detection: MotionDetectionConfig = field(default_factory=MotionDetectionConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    prediction: PredictionConfig = field(default_factory=PredictionConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    general: GeneralConfig = field(default_factory=GeneralConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    gui: GUIConfig = field(default_factory=GUIConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    plot: PlotConfig = field(default_factory=PlotConfig)


# ── Section-name → dataclass-type mapping ───────────────────────────────────
_SECTION_MAP: dict[str, Any] = {
    "hardware": HardwareConfig,
    "motion_detection": MotionDetectionConfig,
    "model": ModelConfig,
    "training": TrainingConfig,
    "augmentation": AugmentationConfig,
    "prediction": PredictionConfig,
    "normalization": NormalizationConfig,
    "general": GeneralConfig,
    "llm": LLMConfig,
    "gui": GUIConfig,
    "evaluation": EvaluationConfig,
    "plot": PlotConfig,
}


# ═════════════════════════════════════════════════════════════════════════════
# Public loader
# ═════════════════════════════════════════════════════════════════════════════


def load_architecture(yaml_path: str | None = None) -> ArchitectureConfig:
    """Parse *yaml_path* (default: ``config/architecture.yaml``) and return a
    validated :class:`ArchitectureConfig` instance.

    Raises
        FileNotFoundError – if the YAML file does not exist.
        ValueError        – if the YAML is malformed or contains unknown keys.
    """
    if yaml_path is None:
        yaml_path = _DEFAULT_YAML_PATH

    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(
            f"Architecture YAML not found: {yaml_path}. "
            f"Ensure config/architecture.yaml exists."
        )

    with open(yaml_path, encoding="utf-8") as fh:
        raw: dict = yaml.safe_load(fh)

    if not isinstance(raw, dict):
        raise ValueError(
            f"Architecture YAML must contain a top-level mapping, "
            f"got {type(raw).__name__}"
        )

    # Build each section from the YAML dict (or fall back to defaults)
    kwargs: dict[str, Any] = {}
    for section_name, section_cls in _SECTION_MAP.items():
        if section_name in raw:
            section_data = raw[section_name]
            if not isinstance(section_data, dict):
                raise ValueError(
                    f"Section {section_name!r} must be a mapping, "
                    f"got {type(section_data).__name__}"
                )
            kwargs[section_name] = section_cls(**section_data)
        else:
            # Missing section — use all defaults
            kwargs[section_name] = section_cls()

    # Warn about unknown top-level keys
    unknown = set(raw) - set(_SECTION_MAP)
    if unknown:
        raise ValueError(
            f"Unknown section(s) in architecture YAML: {sorted(unknown)}"
        )

    return ArchitectureConfig(**kwargs)


# ═════════════════════════════════════════════════════════════════════════════
# Module-level singleton – loaded once at import time
# ═════════════════════════════════════════════════════════════════════════════
architecture: ArchitectureConfig = load_architecture()
