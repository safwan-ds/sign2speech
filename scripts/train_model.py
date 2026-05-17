"""Deprecated CLI wrapper for the reusable training pipeline."""

from __future__ import annotations

import logging

from config.config import setup_logging
from core.pipeline.training_pipeline import main as run_training_pipeline

logger = logging.getLogger(__name__)


def main() -> None:
    """Run model training via the shared pipeline module."""
    setup_logging("train_model")
    logger.warning(
        "scripts/train_model.py is deprecated; use core.pipeline.training_pipeline instead."
    )
    run_training_pipeline(configure_logging=False)


if __name__ == "__main__":
    main()
