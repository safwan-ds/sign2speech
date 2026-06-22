"""Service adapters for GUI package."""

from .data_processing_service import DataProcessingService
from .llm_service import LLMResultEvent, LLMRequest, LLMService
from .logging_service import QueueLogHandler, configure_gui_logger
from .model_service import ModelMetadata, ModelService
from .recording_service import (
    LIVE_PREVIEW_ROW_LIMIT,
    RECORD_PROGRESS_INTERVAL_SECONDS,
    RecordingConfig,
    RecordingService,
)
from .sample_review_service import SampleRecord, SampleReviewService
from .script_runner import ScriptRunner
from .serial_service import SerialSettings, SerialService
from .stream_service import (
    GestureLockState,
    SequenceDecoder,
    StreamConfig,
    StreamWorker,
    TransitionHysteresis,
    calculate_motion_magnitude,
    validate_motion_consistency,
)
from .training_service import TrainingOverrides, TrainingService
from .tts_service import TTSRequest, TTSService

__all__ = [
    "DataProcessingService",
    "LLMResultEvent",
    "LLMRequest",
    "LLMService",
    "QueueLogHandler",
    "configure_gui_logger",
    "ModelMetadata",
    "ModelService",
    "LIVE_PREVIEW_ROW_LIMIT",
    "RECORD_PROGRESS_INTERVAL_SECONDS",
    "RecordingConfig",
    "RecordingService",
    "SampleRecord",
    "SampleReviewService",
    "ScriptRunner",
    "SerialSettings",
    "SerialService",
    "GestureLockState",
    "SequenceDecoder",
    "StreamConfig",
    "StreamWorker",
    "TransitionHysteresis",
    "calculate_motion_magnitude",
    "validate_motion_consistency",
    "TrainingOverrides",
    "TrainingService",
    "TTSRequest",
    "TTSService",
]
