"""
Shared configuration and constants for Sign2Speech project
"""

import os
from pathlib import Path

# --- Load .env file (if present) -------------------------------------------
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
if _ENV_PATH.is_file():
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

COM_PORT = os.environ.get("SIGN2SPEECH_COM_PORT", "COM9")
BAUD_RATE = 115200
TIMEOUT = 3
SERIAL_CONNECTION_DELAY = 2

FLEX_SENSOR_RANGES = {
    0: (25, 300),
    1: (25, 300),
    2: (25, 300),
    3: (25, 300),
    4: (25, 300),
}

FLEX_SENSOR_DEFAULT_RANGE = (0, 1023)

MIN_ACCEL_VALUE = -32768
MAX_ACCEL_VALUE = 32767

MIN_GYRO_VALUE = -32768
MAX_GYRO_VALUE = 32767

NUM_FLEX_SENSORS = 5
NUM_IMU_AXES = 6
EXPECTED_SENSOR_COUNT = 11

ACCEL_X_IDX = NUM_FLEX_SENSORS
ACCEL_Y_IDX = NUM_FLEX_SENSORS + 1
ACCEL_Z_IDX = NUM_FLEX_SENSORS + 2
GYRO_X_IDX = NUM_FLEX_SENSORS + 3
GYRO_Y_IDX = NUM_FLEX_SENSORS + 4
GYRO_Z_IDX = NUM_FLEX_SENSORS + 5

DETECT_GESTURE_MOTION = True
MOTION_THRESHOLD = 0.02
MOTION_DETECTION_MIN_DURATION = 5
MOTION_DETECTION_SMOOTHING_WINDOW = 2
SEQUENCE_OVERLAP = 0.1
MOTION_PADDING_RATIO = 0.2

LSTM_UNITS = 64
LSTM_LAYERS = 2
DROPOUT_RATE = 0.4
BATCH_SIZE = 32
EPOCHS = 150
LEARNING_RATE = 0.0005
SEQUENCE_LENGTH = 20

MODEL_TYPE = "enhanced"
USE_BIDIRECTIONAL = True
USE_ATTENTION = True
USE_BATCH_NORM = True
WEIGHT_DECAY = 5e-4

USE_AUGMENTATION = True
AUGMENTATION_FACTOR = 2
AUGMENTATION_PROB = 0.7
NUM_AUGMENTATIONS_PER_SAMPLE = 2
TIME_WARP_SIGMA = 0.2
TIME_WARP_KNOT = 4
MAGNITUDE_WARP_SIGMA = 0.2
MAGNITUDE_WARP_KNOT = 4
NOISE_LEVEL = 0.01
SCALE_RANGE = (0.9, 1.1)
TIME_SHIFT_RANGE = 0.1
ROTATION_MAX_ANGLE = 10

USE_ENHANCED_FEATURES = False
INCLUDE_VELOCITY = True
INCLUDE_ACCELERATION = True
INCLUDE_ROLLING_STATS = True
ROLLING_WINDOW_SIZE = 5

ENABLE_MADGWICK = True

USE_WEIGHTED_LOSS = True
USE_LABEL_SMOOTHING = True
LABEL_SMOOTHING_FACTOR = 0.1

USE_COSINE_ANNEALING = True
COSINE_T_0 = 10
COSINE_T_MULT = 2
COSINE_ETA_MIN = 1e-6
LR_PLATEAU_FACTOR = 0.5
LR_PLATEAU_PATIENCE = 5
LR_PLATEAU_MIN = 1e-6

USE_WARMUP = True
WARMUP_EPOCHS = 5
WARMUP_START_FACTOR = 0.1

EARLY_STOPPING_PATIENCE = 15
MIN_DELTA = 1e-4

GRADIENT_CLIP_VALUE = 1.0

MIN_VALIDATION_SAMPLES_PER_CLASS = 5

USE_ENSEMBLE = False
ENSEMBLE_SIZE = 3

PREDICTION_INTERVAL = 0.08

RANDOM_STATE = 42
USE_TEST_SPLIT = False
TEST_SIZE = 0.1
TEST_DATA_SPLIT_PERCENTAGE = 0.1
MIN_STRATIFY_SAMPLES = 2
DEFAULT_VALIDATION_SIZE = 0.1

CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CONFIG_DIR)
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
TEST_DATA_DIR = os.path.join(BASE_DIR, "data", "test")
MODELS_DIR = os.path.join(BASE_DIR, "models")

PREDICTION_MOTION_THRESHOLD = 1000
CONFIDENCE_THRESHOLD = 0.74
PREDICTION_CLASS_THRESHOLDS: dict[str, float] = {"DEFAULT": CONFIDENCE_THRESHOLD}
PREDICTION_CONSENSUS_FRAMES = 5
PREDICTION_AVG_MOTION_THRESHOLD = 600
PREDICTION_MOTION_VARIANCE_MIN = 150
PREDICTION_SIGNIFICANT_MOTION_MIN_RATIO = 0.35
PREDICTION_MIN_CONFIDENCE_GAP = 0.15
PREDICTION_DEBUG_MODE = False
MIN_CONSECUTIVE_REST = 5
MIN_GESTURES_FOR_LLM = 2
PREDICTION_SWITCH_CONSENSUS_FRAMES = 3
PREDICTION_INITIAL_CONSENSUS_FRAMES = 2
PREDICTION_KEEP_LAST_STABLE_FRAMES = 2
PREDICTION_UNCERTAIN_TOKEN = "UNKNOWN"
PREDICTION_REST_WEIGHT = 0.75
ENABLE_SEQUENCE_DECODER = False
SEQUENCE_DECODER_SWITCH_PENALTY = 0.0
SEQUENCE_DECODER_REST_SWITCH_PENALTY = 0.0

NORM_MIN = 0.0
NORM_MAX = 1.0

MIN_FLEX_VALUE = min(v[0] for v in FLEX_SENSOR_RANGES.values())
MAX_FLEX_VALUE = max(v[1] for v in FLEX_SENSOR_RANGES.values())

USE_TTS = True

USE_QWEN_LLM = True
QWEN_MODEL_FILENAME = "qwen2.5-7b-instruct-q6_k-00001-of-00002.gguf"
QWEN_MODEL_PATH = os.path.join(MODELS_DIR, "llm", QWEN_MODEL_FILENAME)
QWEN_N_CTX = 2048
QWEN_N_GPU_LAYERS = -1
QWEN_N_BATCH = 512
QWEN_FORCE_GPU = True
QWEN_MAX_TOKENS = 64
QWEN_INFERENCE_TEMPERATURE = 0.25

# ── Remote LLM backend (OpenAI-compatible) ─────────────────────────────────
LLM_BACKEND = os.environ.get("LLM_BACKEND", "local")  # "local" or "remote"
LLM_REMOTE_URL = os.environ.get("LLM_REMOTE_URL", "https://api.deepseek.com")
LLM_REMOTE_API_KEY = os.environ.get(
    "LLM_REMOTE_API_KEY", os.environ.get("DEEPSEEK_API_KEY", "")
)
LLM_REMOTE_MODEL = os.environ.get("LLM_REMOTE_MODEL", "deepseek-v4-flash")
LLM_REMOTE_TIMEOUT = 15.0
LLM_REMOTE_FORMAT = os.environ.get(
    "LLM_REMOTE_FORMAT", "chat"
)  # "chat" (OpenAI/Nous) or "completions" (legacy)
LLM_REMOTE_MAX_TOKENS = int(os.environ.get("LLM_REMOTE_MAX_TOKENS", "1024"))

PLOT_FIGURE_WIDTH = 12
PLOT_FIGURE_HEIGHT = 10
PLOT_NUM_ROWS = 3
PLOT_NUM_COLS = 1
PLOT_FONT_SIZE = 14
PLOT_MARKER_SIZE = 2
PLOT_GRID_ALPHA = 0.3

KEYBOARD_DEBOUNCE_DELAY = 0.3
KEYBOARD_POLL_INTERVAL = 0.05

GUI_MIN_WIDTH = 900
GUI_MIN_HEIGHT = 600
GUI_PADDING = 20
GUI_SMALL_PADDING = 10
GUI_FONT_SIZE = 12
GUI_TITLE_FONT_SIZE = 14
GUI_REVIEW_BATCH_SIZE = 4
GUI_PLOT_ROWS_CALC = 14
GUI_PLOT_HEIGHT_MULTIPLIER = 3
GUI_PLOT_HSPACE = 0.45
GUI_PLOT_WSPACE = 0.30
GUI_PLOT_TOP = 0.95
GUI_PLOT_BOTTOM = 0.05
GUI_THREAD_SLEEP = 0.01

# Data Manager window dimensions
DATA_MANAGER_WINDOW_WIDTH = 1560
DATA_MANAGER_WINDOW_HEIGHT = 940
DATA_MANAGER_MIN_WIDTH = 1220
DATA_MANAGER_MIN_HEIGHT = 760

# Gestures Editor dialog dimensions
GESTURES_EDITOR_DIALOG_WIDTH = 600
GESTURES_EDITOR_DIALOG_HEIGHT = 420

DEFAULT_UI_LANGUAGE = "tr"
SUPPORTED_UI_LANGUAGES = ("tr", "en")

EVALUATION_DPI = 300
CONFUSION_MATRIX_FIGSIZE = (10, 8)
ROC_CURVE_FIGSIZE = (10, 8)
EVALUATION_CLASS_WEIGHT_EPSILON = 1e-6


import logging
from datetime import datetime

LOGS_OUTPUT_DIR = os.path.join(BASE_DIR, "logs")

CONSOLE_LOG_LEVEL = logging.INFO
FILE_LOG_LEVEL = logging.DEBUG

CONSOLE_LOG_FORMAT = "%(levelname)s - %(message)s"
FILE_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


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

    file_handler = logging.FileHandler(log_filepath, encoding="utf-8")
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
