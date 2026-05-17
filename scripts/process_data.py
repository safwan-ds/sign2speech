"""Deprecated CLI wrapper for the reusable data processing pipeline."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import setup_logging
from core.pipeline.data_processor import main as run_data_processing_pipeline

logger = logging.getLogger(__name__)


def main() -> None:
    """Run processing via the shared pipeline module."""
    setup_logging("process_data")
    logger.warning(
        "scripts/process_data.py is deprecated; use core.pipeline.data_processor instead."
    )
    run_data_processing_pipeline(configure_logging=False)


if __name__ == "__main__":
    main()
