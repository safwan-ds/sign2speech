"""Logging setup for GUI and queue forwarding."""

from __future__ import annotations

import logging
from logging import LogRecord
from logging.handlers import RotatingFileHandler
from pathlib import Path
from queue import Queue
from datetime import datetime

from gui.utils.formatting import now_stamp


class QueueLogHandler(logging.Handler):
    """Forward log messages to the GUI event queue."""

    def __init__(self, target_queue: Queue[dict]) -> None:
        super().__init__()
        self.target_queue = target_queue

    def emit(self, record: LogRecord) -> None:
        self.target_queue.put(
            {
                "type": "log",
                "level": record.levelname,
                "timestamp": datetime.fromtimestamp(record.created).strftime(
                    "%H:%M:%S"
                ),
                "source": record.name,
                "message": record.getMessage(),
            }
        )


def configure_gui_logger(log_dir: Path, event_queue: Queue[dict]) -> logging.Logger:
    """Create a GUI logger with file and queue handlers."""
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("gui_app")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    log_file = log_dir / f"gui_{now_stamp()}.log"

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=2 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    queue_handler = QueueLogHandler(event_queue)
    queue_handler.setLevel(logging.INFO)
    queue_handler.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(file_handler)
    logger.addHandler(queue_handler)
    logger.info("GUI logger initialized at %s", log_file)
    return logger
