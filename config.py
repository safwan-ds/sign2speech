"""
Shared configuration and constants for Sign Language Glove project
"""

import os


COM_PORT = "COM9"
BAUD_RATE = 115200
TIMEOUT = 1
SERIAL_CONNECTION_DELAY = 2


FLEX_SENSOR_RANGES = {
    0: (28, 224),
    1: (56, 293),
    2: (39, 240),
    3: (53, 239),
    4: (44, 261),
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
MOTION_DETECTION_SMOOTHING_WINDOW = 3
SEQUENCE_OVERLAP = 0.5
MOTION_PADDING_RATIO = 0.2


LSTM_UNITS = 64
LSTM_LAYERS = 2
DROPOUT_RATE = 0.4
BATCH_SIZE = 32
EPOCHS = 150
LEARNING_RATE = 0.0005
SEQUENCE_LENGTH = 30


MODEL_TYPE = "advanced"
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


USE_ENHANCED_FEATURES = True
INCLUDE_VELOCITY = True
INCLUDE_ACCELERATION = True
INCLUDE_ROLLING_STATS = True
ROLLING_WINDOW_SIZE = 5

NORMALIZE_YAW_ROTATION = True
# When True, each sequence is rotated in the horizontal (accelX/Y, gyroX/Y) plane
# to cancel the user's facing direction before training and inference.  This makes
# gesture recognition invariant to which way the user is facing.


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
USE_TEST_SPLIT = False  # If False, all data is used for training (no holdout test set)
TEST_SIZE = 0.1
TEST_DATA_SPLIT_PERCENTAGE = 0.15
MIN_STRATIFY_SAMPLES = 2
DEFAULT_VALIDATION_SIZE = 0.1


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
TEST_DATA_DIR = os.path.join(BASE_DIR, "data", "test")
MODELS_DIR = os.path.join(BASE_DIR, "models")


PREDICTION_MOTION_THRESHOLD = 1000
CONFIDENCE_THRESHOLD = 0.72
PREDICTION_MIN_CONFIDENCE = CONFIDENCE_THRESHOLD
PREDICTION_CONSENSUS_FRAMES = 2
PREDICTION_AVG_MOTION_THRESHOLD = 600
PREDICTION_MOTION_VARIANCE_MIN = 150
PREDICTION_SIGNIFICANT_MOTION_MIN_RATIO = 0.35
PREDICTION_MIN_CONFIDENCE_GAP = 0.15
PREDICTION_DEBUG_MODE = False
MIN_CONSECUTIVE_REST = 5
MIN_GESTURES_FOR_LLM = 2


NORM_MIN = 0.0
NORM_MAX = 1.0


MIN_FLEX_VALUE = min(v[0] for v in FLEX_SENSOR_RANGES.values())
MAX_FLEX_VALUE = max(v[1] for v in FLEX_SENSOR_RANGES.values())


USE_QWEN_LLM = True
QWEN_MODEL_FILENAME = "qwen2.5-3b-instruct-q4_k_m.gguf"
QWEN_MODEL_PATH = os.path.join(MODELS_DIR, QWEN_MODEL_FILENAME)
QWEN_N_CTX = 512
QWEN_N_GPU_LAYERS = -1
QWEN_N_BATCH = 512
QWEN_FORCE_GPU = True
QWEN_MAX_TOKENS = 32
QWEN_INFERENCE_TEMPERATURE = 0.1


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

# GUI language defaults
DEFAULT_UI_LANGUAGE = "tr"
SUPPORTED_UI_LANGUAGES = ("tr", "en")


EVALUATION_DPI = 300
CONFUSION_MATRIX_FIGSIZE = (10, 8)
ROC_CURVE_FIGSIZE = (10, 8)
EVALUATION_CLASS_WEIGHT_EPSILON = 1e-6


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
import logging
from datetime import datetime

# Logging directories
LOGS_OUTPUT_DIR = os.path.join(BASE_DIR, "logs")

# Logging levels
CONSOLE_LOG_LEVEL = logging.INFO
FILE_LOG_LEVEL = logging.DEBUG

# Log file naming

# Format strings
# Console: Clean output without timestamp
CONSOLE_LOG_FORMAT = "%(levelname)s - %(message)s"
# File: Detailed output with timestamp
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

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture all levels

    # Clear any existing handlers
    root_logger.handlers.clear()

    # Console handler - clean output without timestamp
    console_handler = logging.StreamHandler()
    console_handler.setLevel(CONSOLE_LOG_LEVEL)
    console_formatter = logging.Formatter(CONSOLE_LOG_FORMAT)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler - detailed output with timestamp
    file_handler = logging.FileHandler(log_filepath, encoding="utf-8")
    file_handler.setLevel(FILE_LOG_LEVEL)
    file_formatter = logging.Formatter(FILE_LOG_FORMAT, datefmt=DATE_FORMAT)
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # Log the initialization
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized - Log file: {log_filepath}")
