"""
Shared configuration and constants for Sign2Speech project

Architecture constants are loaded from ``architecture.yaml`` via the
:mod:`config.architecture` module.  Runtime-derived paths and
environment-variable-based values are computed here.
"""

import logging
import os
from datetime import datetime
from pathlib import Path

from config.architecture import architecture

# ---------------------------------------------------------------------------
# Load .env file (if present) — idempotent, deferred until init_config()
# ---------------------------------------------------------------------------
_config_initialized: bool = False
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


def init_config() -> None:
    """Load ``.env`` file into ``os.environ`` (idempotent, called once).

    Safe to call multiple times — subsequent calls are no-ops.  Called
    automatically at module import to preserve backward compatibility.
    """
    global _config_initialized
    if _config_initialized:
        return
    _config_initialized = True

    if not _ENV_PATH.is_file():
        return
    with open(_ENV_PATH, encoding="utf-8") as _fh:
        for _line in _fh:
            _line = _line.strip()
            if not _line or _line.startswith("#") or "=" not in _line:
                continue
            key, _, val = _line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val


# Backward-compatible: run at import time so existing consumers don't break.
init_config()

# ===========================================================================
# Architecture constants — re-exported from architecture.yaml
# ===========================================================================

# -- hardware ---------------------------------------------------------------
COM_PORT = os.environ.get("SIGN2SPEECH_COM_PORT", architecture.hardware.com_port)
BAUD_RATE = architecture.hardware.baud_rate
TIMEOUT = architecture.hardware.timeout
SERIAL_CONNECTION_DELAY = architecture.hardware.serial_connection_delay
FLEX_SENSOR_RANGES = {
    k: tuple(v) for k, v in architecture.hardware.flex_sensor_ranges.items()
}
FLEX_SENSOR_DEFAULT_RANGE = tuple(architecture.hardware.flex_sensor_default_range)
MIN_ACCEL_VALUE = architecture.hardware.min_accel_value
MAX_ACCEL_VALUE = architecture.hardware.max_accel_value
MIN_GYRO_VALUE = architecture.hardware.min_gyro_value
MAX_GYRO_VALUE = architecture.hardware.max_gyro_value
NUM_FLEX_SENSORS = architecture.hardware.num_flex_sensors
NUM_IMU_AXES = architecture.hardware.num_imu_axes
EXPECTED_SENSOR_COUNT = architecture.hardware.expected_sensor_count

# -- sensor-index aliases (computed from NUM_FLEX_SENSORS) ------------------
ACCEL_X_IDX = NUM_FLEX_SENSORS
ACCEL_Y_IDX = NUM_FLEX_SENSORS + 1
ACCEL_Z_IDX = NUM_FLEX_SENSORS + 2
GYRO_X_IDX = NUM_FLEX_SENSORS + 3
GYRO_Y_IDX = NUM_FLEX_SENSORS + 4
GYRO_Z_IDX = NUM_FLEX_SENSORS + 5

# -- motion_detection -------------------------------------------------------
DETECT_GESTURE_MOTION = architecture.motion_detection.detect_gesture_motion
MOTION_THRESHOLD = architecture.motion_detection.motion_threshold
MOTION_DETECTION_MIN_DURATION = architecture.motion_detection.motion_detection_min_duration
MOTION_DETECTION_SMOOTHING_WINDOW = architecture.motion_detection.motion_detection_smoothing_window
SEQUENCE_OVERLAP = architecture.motion_detection.sequence_overlap
MOTION_PADDING_RATIO = architecture.motion_detection.motion_padding_ratio
ENABLE_MADGWICK = architecture.motion_detection.enable_madgwick

# -- model ------------------------------------------------------------------
LSTM_UNITS = architecture.model.lstm_units
LSTM_LAYERS = architecture.model.lstm_layers
DROPOUT_RATE = architecture.model.dropout_rate
MODEL_TYPE = architecture.model.model_type
USE_BIDIRECTIONAL = architecture.model.use_bidirectional
USE_ATTENTION = architecture.model.use_attention
USE_BATCH_NORM = architecture.model.use_batch_norm
USE_ENHANCED_FEATURES = architecture.model.use_enhanced_features
INCLUDE_VELOCITY = architecture.model.include_velocity
INCLUDE_ACCELERATION = architecture.model.include_acceleration
INCLUDE_ROLLING_STATS = architecture.model.include_rolling_stats
ROLLING_WINDOW_SIZE = architecture.model.rolling_window_size

# -- training ---------------------------------------------------------------
BATCH_SIZE = architecture.training.batch_size
EPOCHS = architecture.training.epochs
LEARNING_RATE = architecture.training.learning_rate
SEQUENCE_LENGTH = architecture.training.sequence_length
WEIGHT_DECAY = architecture.training.weight_decay
USE_WEIGHTED_LOSS = architecture.training.use_weighted_loss
USE_LABEL_SMOOTHING = architecture.training.use_label_smoothing
LABEL_SMOOTHING_FACTOR = architecture.training.label_smoothing_factor
USE_COSINE_ANNEALING = architecture.training.use_cosine_annealing
COSINE_T_0 = architecture.training.cosine_t_0
COSINE_T_MULT = architecture.training.cosine_t_mult
COSINE_ETA_MIN = architecture.training.cosine_eta_min
LR_PLATEAU_FACTOR = architecture.training.lr_plateau_factor
LR_PLATEAU_PATIENCE = architecture.training.lr_plateau_patience
LR_PLATEAU_MIN = architecture.training.lr_plateau_min
USE_WARMUP = architecture.training.use_warmup
WARMUP_EPOCHS = architecture.training.warmup_epochs
WARMUP_START_FACTOR = architecture.training.warmup_start_factor
EARLY_STOPPING_PATIENCE = architecture.training.early_stopping_patience
MIN_DELTA = architecture.training.min_delta
GRADIENT_CLIP_VALUE = architecture.training.gradient_clip_value
MIN_VALIDATION_SAMPLES_PER_CLASS = architecture.training.min_validation_samples_per_class
USE_ENSEMBLE = architecture.training.use_ensemble
ENSEMBLE_SIZE = architecture.training.ensemble_size
RANDOM_STATE = architecture.training.random_state
USE_TEST_SPLIT = architecture.training.use_test_split
USE_AUGMENTATION = architecture.augmentation.use_augmentation
AUGMENTATION_FACTOR = architecture.augmentation.augmentation_factor
AUGMENTATION_PROB = architecture.augmentation.augmentation_prob
NUM_AUGMENTATIONS_PER_SAMPLE = architecture.augmentation.num_augmentations_per_sample
TIME_WARP_SIGMA = architecture.augmentation.time_warp_sigma
TIME_WARP_KNOT = architecture.augmentation.time_warp_knot
MAGNITUDE_WARP_SIGMA = architecture.augmentation.magnitude_warp_sigma
MAGNITUDE_WARP_KNOT = architecture.augmentation.magnitude_warp_knot
NOISE_LEVEL = architecture.augmentation.noise_level
SCALE_RANGE = tuple(architecture.augmentation.scale_range)
TIME_SHIFT_RANGE = architecture.augmentation.time_shift_range
ROTATION_MAX_ANGLE = architecture.augmentation.rotation_max_angle

# -- prediction -------------------------------------------------------------
PREDICTION_INTERVAL = architecture.prediction.prediction_interval
PREDICTION_MOTION_THRESHOLD = architecture.prediction.prediction_motion_threshold
CONFIDENCE_THRESHOLD = architecture.prediction.confidence_threshold
PREDICTION_CLASS_THRESHOLDS: dict[str, float] = dict(
    architecture.prediction.prediction_class_thresholds
)
PREDICTION_CONSENSUS_FRAMES = architecture.prediction.prediction_consensus_frames
PREDICTION_AVG_MOTION_THRESHOLD = architecture.prediction.prediction_avg_motion_threshold
PREDICTION_MOTION_VARIANCE_MIN = architecture.prediction.prediction_motion_variance_min
PREDICTION_SIGNIFICANT_MOTION_MIN_RATIO = (
    architecture.prediction.prediction_significant_motion_min_ratio
)
PREDICTION_MIN_CONFIDENCE_GAP = architecture.prediction.prediction_min_confidence_gap
PREDICTION_DEBUG_MODE = architecture.prediction.prediction_debug_mode
MIN_CONSECUTIVE_REST = architecture.prediction.min_consecutive_rest
MIN_GESTURES_FOR_LLM = architecture.prediction.min_gestures_for_llm
PREDICTION_SWITCH_CONSENSUS_FRAMES = (
    architecture.prediction.prediction_switch_consensus_frames
)
PREDICTION_INITIAL_CONSENSUS_FRAMES = (
    architecture.prediction.prediction_initial_consensus_frames
)
PREDICTION_KEEP_LAST_STABLE_FRAMES = (
    architecture.prediction.prediction_keep_last_stable_frames
)
PREDICTION_UNCERTAIN_TOKEN = architecture.prediction.prediction_uncertain_token
PREDICTION_REST_WEIGHT = architecture.prediction.prediction_rest_weight
ENABLE_SEQUENCE_DECODER = architecture.prediction.enable_sequence_decoder
SEQUENCE_DECODER_SWITCH_PENALTY = architecture.prediction.sequence_decoder_switch_penalty
SEQUENCE_DECODER_REST_SWITCH_PENALTY = (
    architecture.prediction.sequence_decoder_rest_switch_penalty
)

# -- normalization ----------------------------------------------------------
NORM_MIN = architecture.normalization.norm_min
NORM_MAX = architecture.normalization.norm_max

# -- general feature toggles -------------------------------------------------
USE_TTS = architecture.general.use_tts

# -- llm --------------------------------------------------------------------
USE_QWEN_LLM = architecture.llm.use_qwen_llm
QWEN_MODEL_FILENAME = architecture.llm.qwen_model_filename
QWEN_N_CTX = architecture.llm.qwen_n_ctx
QWEN_N_GPU_LAYERS = architecture.llm.qwen_n_gpu_layers
QWEN_N_BATCH = architecture.llm.qwen_n_batch
QWEN_FORCE_GPU = architecture.llm.qwen_force_gpu
QWEN_MAX_TOKENS = architecture.llm.qwen_max_tokens
QWEN_INFERENCE_TEMPERATURE = architecture.llm.qwen_inference_temperature

# -- plot -------------------------------------------------------------------
PLOT_FIGURE_WIDTH = architecture.plot.plot_figure_width
PLOT_FIGURE_HEIGHT = architecture.plot.plot_figure_height
PLOT_NUM_ROWS = architecture.plot.plot_num_rows
PLOT_NUM_COLS = architecture.plot.plot_num_cols
PLOT_FONT_SIZE = architecture.plot.plot_font_size
PLOT_MARKER_SIZE = architecture.plot.plot_marker_size
PLOT_GRID_ALPHA = architecture.plot.plot_grid_alpha

# -- gui --------------------------------------------------------------------
KEYBOARD_DEBOUNCE_DELAY = architecture.gui.keyboard_debounce_delay
KEYBOARD_POLL_INTERVAL = architecture.gui.keyboard_poll_interval
GUI_MIN_WIDTH = architecture.gui.gui_min_width
GUI_MIN_HEIGHT = architecture.gui.gui_min_height
GUI_PADDING = architecture.gui.gui_padding
GUI_SMALL_PADDING = architecture.gui.gui_small_padding
GUI_FONT_SIZE = architecture.gui.gui_font_size
GUI_TITLE_FONT_SIZE = architecture.gui.gui_title_font_size
GUI_REVIEW_BATCH_SIZE = architecture.gui.gui_review_batch_size
GUI_PLOT_ROWS_CALC = architecture.gui.gui_plot_rows_calc
GUI_PLOT_HEIGHT_MULTIPLIER = architecture.gui.gui_plot_height_multiplier
GUI_PLOT_HSPACE = architecture.gui.gui_plot_hspace
GUI_PLOT_WSPACE = architecture.gui.gui_plot_wspace
GUI_PLOT_TOP = architecture.gui.gui_plot_top
GUI_PLOT_BOTTOM = architecture.gui.gui_plot_bottom
GUI_THREAD_SLEEP = architecture.gui.gui_thread_sleep
DATA_MANAGER_WINDOW_WIDTH = architecture.gui.data_manager_window_width
DATA_MANAGER_WINDOW_HEIGHT = architecture.gui.data_manager_window_height
DATA_MANAGER_MIN_WIDTH = architecture.gui.data_manager_min_width
DATA_MANAGER_MIN_HEIGHT = architecture.gui.data_manager_min_height
GESTURES_EDITOR_DIALOG_WIDTH = architecture.gui.gestures_editor_dialog_width
GESTURES_EDITOR_DIALOG_HEIGHT = architecture.gui.gestures_editor_dialog_height
DEFAULT_UI_LANGUAGE = architecture.gui.default_ui_language
SUPPORTED_UI_LANGUAGES = tuple(architecture.gui.supported_ui_languages)

# -- evaluation -------------------------------------------------------------
EVALUATION_DPI = architecture.evaluation.evaluation_dpi
CONFUSION_MATRIX_FIGSIZE = tuple(architecture.evaluation.confusion_matrix_figsize)
ROC_CURVE_FIGSIZE = tuple(architecture.evaluation.roc_curve_figsize)
EVALUATION_CLASS_WEIGHT_EPSILON = architecture.evaluation.evaluation_class_weight_epsilon

# ===========================================================================
# Runtime-derived paths (computed at import time)
# ===========================================================================
CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CONFIG_DIR)
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
TEST_DATA_DIR = os.path.join(BASE_DIR, "data", "test")
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_OUTPUT_DIR = os.path.join(BASE_DIR, "logs")

QWEN_MODEL_PATH = os.path.join(MODELS_DIR, "llm", QWEN_MODEL_FILENAME)

MIN_FLEX_VALUE = min(v[0] for v in FLEX_SENSOR_RANGES.values())
MAX_FLEX_VALUE = max(v[1] for v in FLEX_SENSOR_RANGES.values())

# ===========================================================================
# Environment-variable-derived values (loaded at import time)
# ===========================================================================

# ── Remote LLM backend (OpenAI-compatible) ─────────────────────────────────
LLM_BACKEND = os.environ.get("LLM_BACKEND", architecture.llm.llm_backend)
LLM_REMOTE_URL = os.environ.get("LLM_REMOTE_URL", architecture.llm.llm_remote_url)
LLM_REMOTE_API_KEY = os.environ.get(
    "LLM_REMOTE_API_KEY",
    os.environ.get("DEEPSEEK_API_KEY", architecture.llm.llm_remote_api_key),
)
LLM_REMOTE_MODEL = os.environ.get(
    "LLM_REMOTE_MODEL", architecture.llm.llm_remote_model
)
LLM_REMOTE_TIMEOUT = architecture.llm.llm_remote_timeout
LLM_REMOTE_FORMAT = os.environ.get(
    "LLM_REMOTE_FORMAT", architecture.llm.llm_remote_format
)
LLM_REMOTE_MAX_TOKENS = int(
    os.environ.get("LLM_REMOTE_MAX_TOKENS", str(architecture.llm.llm_remote_max_tokens))
)

# ===========================================================================
# Logging constants (depend on the ``logging`` module)
# ===========================================================================
CONSOLE_LOG_LEVEL = logging.INFO
FILE_LOG_LEVEL = logging.DEBUG
CONSOLE_LOG_FORMAT = "%(levelname)s - %(message)s"
FILE_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


# ===========================================================================
# Logging setup helper
# ===========================================================================
def setup_logging(script_name: str | None = None) -> None:
    """
    Configure logging with dual handlers:
    - Console handler: Clean output without timestamps
    - File handler: Detailed logs with timestamps

    Args:
        script_name: Optional name for the log file (e.g., 'train', 'predict')
    """
    # Create logs directory if it doesn't exist
    os.makedirs(LOGS_OUTPUT_DIR, exist_ok=True)

    # Generate log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if script_name:
        log_filename = f"{script_name}_{timestamp}.log"
    else:
        log_filename = f"{timestamp}.log"
    log_filepath = os.path.join(LOGS_OUTPUT_DIR, log_filename)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    root_logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(CONSOLE_LOG_LEVEL)
    console_formatter = logging.Formatter(CONSOLE_LOG_FORMAT)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    file_handler = logging.handlers.RotatingFileHandler(
        log_filepath,
        maxBytes=2 * 1024 * 1024,  # 2 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(FILE_LOG_LEVEL)
    file_formatter = logging.Formatter(FILE_LOG_FORMAT, datefmt=DATE_FORMAT)
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # Silence matplotlib internal loggers (font manager, backends, pyplot)
    for mpl_logger in ("matplotlib", "matplotlib.font_manager", "matplotlib.pyplot"):
        lg = logging.getLogger(mpl_logger)
        lg.setLevel(logging.WARNING)
        lg.propagate = False

    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized - Log file: {log_filepath}")
