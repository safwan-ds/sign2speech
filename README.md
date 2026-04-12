# Sign Language Glove - Real-time Gesture Recognition

A production-grade system for real-time sign language gesture recognition using a smart glove with flex sensors and IMU data. Leverages deep learning (LSTM with attention) and optional LLM refinement for accurate, context-aware translation.

## 🌟 Features

- **Real-time Gesture Recognition**: 11+ gestures (REST, hello, goodbye, numbers, etc.) via LSTM-based inference
- **Smart Glove Hardware**: 5 flex sensors + IMU (accelerometer/gyroscope) data streaming over serial
- **Advanced ML Pipeline**:
  - LSTM with bidirectional, attention, and batch normalization layers
  - Data augmentation (time warping, magnitude warping, noise injection)
  - Weighted loss and label smoothing for imbalanced classes
  - Cosine annealing + learning rate plateau scheduling
  - Ensemble training support
- **Production GUI (PySide6)**: Non-blocking threaded pipeline with real-time prediction cards, confidence bars, and sentence assembly
- **Startup Quality-of-Life**: Automatically loads `models/latest` on launch
- **Smart Port UX**: Lists all serial ports, auto-selects CH340 when available, and validates input stream with startup timeout
- **Optional LLM Refinement**: QWEN 2.5 model integration for contextual sentence generation
- **Comprehensive Logging**: Structured logs with levels, file rotation, and GUI log viewer
- **Model Management**: Multiple model versions, easy model selection, automatic metadata tracking

## 📦 What's Inside

```text
sign_language_glove/
├── core/
│   ├── inference/          # Real-time prediction pipeline
│   ├── models/             # LSTM architecture definitions
│   └── training/           # Training orchestration & evaluation
├── gui/                    # Production PySide6 dashboard
├── scripts/                # CLI entry points (training, prediction, data processing)
├── data/
│   ├── raw/                # Raw sensor recordings per gesture
│   ├── processed/          # Normalized sequences (.npz format)
│   └── test/               # Holdout test set
├── models/                 # Trained model weights & encoders
├── utils/                  # Shared utilities (data, augmentation, plotting, serial)
├── config/                 # Configuration files (gestures, templates)
├── tests/                  # Unit tests (pytest)
└── docs/                   # API and architecture documentation
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Virtual environment (venv, conda, etc.)
- Smart glove hardware (or recorded data for training)

### Installation

```bash
# Clone repository
cd sign_language_glove

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Or use pyproject.toml
pip install -e .
```

### Running the GUI

```bash
python -m gui.main
# or
python scripts/run_gui.py
```

**Keyboard Shortcuts:**

- `Ctrl+S`: Start/Stop real-time stream
- `Ctrl+L`: Clear sentence
- `Ctrl+E`: Export sentence text

**Startup Behavior:**

- GUI auto-loads the latest model from `models/latest`
- Port list shows device names and one COM identifier (example: `USB-SERIAL CH340 (COM9)`)
- If a selected port does not send valid glove data after start, stream auto-stops with a warning

### Training a Model

```bash
python scripts/train_model.py
```

This will:

1. Load processed sequences from `data/processed/`
2. Train an LSTM model with current config
3. Save model to `models/lstm_<timestamp>/` with metadata

### Processing Raw Data

```bash
python scripts/process_data.py
```

Normalizes flex sensor and IMU data into fixed-length sequences ready for training.

### Making Predictions

```bash
python scripts/predict.py
```

Batch prediction on test data with confidence scores and confusion matrix.

## ⚙️ Configuration

All configuration is centralized in `config.py` and can be overridden via environment variables (see `.env.example`).

### Key Parameters

**Hardware:**

- `COM_PORT`: Serial port (e.g., "COM9", "/dev/ttyUSB0")
- `BAUD_RATE`: Serial baud rate (115200)
- `FLEX_SENSOR_RANGES`: Calibrated min/max per flex sensor

**Model Architecture:**

- `LSTM_UNITS`, `LSTM_LAYERS`: Model size/depth
- `DROPOUT_RATE`: Regularization
- `SEQUENCE_LENGTH`: Fixed input sequence length (30 timesteps)
- `BATCH_SIZE`, `EPOCHS`: Training hyperparameters

**Data Augmentation:**

- `USE_AUGMENTATION`: Enable/disable
- `TIME_WARP_SIGMA`, `MAGNITUDE_WARP_SIGMA`: Augmentation strength
- `SCALE_RANGE`: Sensor scaling variation

**Motion Detection:**

- `MOTION_THRESHOLD`: Acceleration threshold to detect active gesture
- `MOTION_DETECTION_MIN_DURATION`: Minimum frames to trigger prediction

**LLM Integration:**

- `USE_QWEN_LLM`: Enable sentence refinement
- `QWEN_N_GPU_LAYERS`: GPU acceleration (-1 for full GPU)

See [config.py](config.py) for all ~100 configurable parameters.

## 📊 Data Format

### Raw Data Structure

```text
data/raw/
├── hello/
│   ├── sample_001.csv
│   ├── sample_002.csv
│   └── ...
├── thank_you/
│   └── ...
└── [other gestures]/
```

Each CSV has columns: `flex_0, flex_1, flex_2, flex_3, flex_4, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z`

### Processed Data

Stored as `.npz` files (NumPy compressed):

```python
import numpy as np
data = np.load('data/processed/sequences_hello.npz')
sequences = data['sequences']  # Shape: (num_samples, 30, 11)
labels = data['labels']        # Shape: (num_samples,)
```

## 🧪 Testing

Run pytest from project root:

```bash
pytest tests/
pytest tests/test_data_utils.py -v              # Specific test file
pytest tests/ -k "augmentation" --tb=short      # Filter tests
```

## 📝 API Documentation

See [docs/API.md](docs/API.md) for detailed module interfaces.

### Core Modules

| Module                             | Purpose                             |
| ---------------------------------- | ----------------------------------- |
| `core.inference.gesture_predictor` | Real-time prediction from sequences |
| `core.models.lstm_model`           | LSTM architecture factory           |
| `core.training.model_training`     | Training loop with validation       |
| `core.training.data_loader`        | Load/split datasets                 |
| `utils.data_utils`                 | Normalization, windowing, sampling  |
| `utils.augmentation`               | Data augmentation pipelines         |
| `utils.serial_utils`               | Serial communication with glove     |
| `gui.services.model_service`       | Model loading/metadata              |

## 🏗️ Architecture

### Training Pipeline

```text
Raw Data (CSV)
  → Normalization (flex/IMU scaling)
  → Windowing (30-step sequences)
  → Augmentation (optional)
  → LSTM Training
  → Validation & Early Stopping
  → Model Checkpointing
```

### Inference Pipeline

```text
Serial Stream (glove hardware)
  → Motion Detection (threshold-based activation)
  → Sequence Buffering (30-step window)
  → LSTM Prediction per window
  → Smoothing & Debouncing
  → LLM Refinement (optional)
  → GUI Display
```

### GUI Architecture

- **Non-blocking I/O**: Separate thread for serial + model inference
- **Queue-based communication**: Thread-safe data passing
- **Qt timers + queue polling**: Responsive UI updates
- **Log streaming**: Real-time log viewer with file rotation

## 🔧 Development Workflow

1. **Branch naming**: `feature/xyz`, `fix/xyz`, `docs/xyz`
2. **Testing**: Write tests before committing to `main`
3. **Commits**: Use conventional format (`feat:`, `fix:`, `docs:`, `chore:`)
4. **Config changes**: Update `config.py` and `.env.example` together

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## 📄 License

[Add license information]

## 👥 Contributors

- Your Name

## 📞 Support

For issues, questions, or feature requests, open a GitHub issue or contact [maintainer email].

---

**Last Updated**: April 2026  
**Python Version**: 3.11+  
**Framework**: PyTorch 2.0+, CustomTkinter
