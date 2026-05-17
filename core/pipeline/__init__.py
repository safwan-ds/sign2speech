"""Core pipeline modules for data processing and model training."""

from .data_processor import main as run_data_processing_pipeline
from .training_pipeline import main as run_training_pipeline

__all__ = ["run_data_processing_pipeline", "run_training_pipeline"]
