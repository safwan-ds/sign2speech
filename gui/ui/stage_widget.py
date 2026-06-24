"""Small helper widget that visualizes a pipeline stage.

Contains an independent progress bar, status and metrics text and two
actionable buttons (Retry / Skip) that map to backend control methods.
"""

from __future__ import annotations

from PySide6.QtWidgets import QGroupBox
from PySide6.QtWidgets import QHBoxLayout
from PySide6.QtWidgets import QLabel
from PySide6.QtWidgets import QProgressBar
from PySide6.QtWidgets import QPushButton
from PySide6.QtWidgets import QVBoxLayout
from PySide6.QtWidgets import QWidget


class StageWidget(QGroupBox):
    """Small helper widget that visualizes a pipeline stage.

    Contains an independent progress bar, status and metrics text and two
    actionable buttons (Retry / Skip) that map to backend control methods.
    """

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(title, parent)
        self.setObjectName(f"stage_{title}")
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(6, 6, 6, 6)
        self._layout.setSpacing(6)

        self.status_label = QLabel("Idle")
        self._layout.addWidget(self.status_label)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self._layout.addWidget(self.progress)

        self.metrics_label = QLabel("")
        self.metrics_label.setWordWrap(True)
        self._layout.addWidget(self.metrics_label)

        btn_row = QHBoxLayout()
        self.retry_btn = QPushButton("Retry")
        self.skip_btn = QPushButton("Skip")
        btn_row.addWidget(self.retry_btn)
        btn_row.addWidget(self.skip_btn)
        btn_row.addStretch(1)
        self._layout.addLayout(btn_row)

    def set_started(self, message: str | None = None) -> None:
        """Set stage to busy state."""
        self.status_label.setText(message or "Running")
        self.progress.setRange(0, 0)  # busy

    def set_progress(self, done: int, total: int) -> None:
        """Update progress display."""
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(max(0, min(total, done)))
        self.status_label.setText(f"{done}/{total}")

    def set_completed(self) -> None:
        """Mark stage as completed."""
        self.status_label.setText("Completed")
        self.progress.setRange(0, 100)
        self.progress.setValue(100)

    def set_failed(self, message: str | None = None) -> None:
        """Mark stage as failed."""
        self.status_label.setText(f"Failed: {message or 'error'}")
        self.progress.setRange(0, 100)
        self.progress.setValue(0)

    def set_skipped(self) -> None:
        """Mark stage as skipped."""
        self.status_label.setText("Skipped")
        self.progress.setRange(0, 100)
        self.progress.setValue(0)

    def set_metrics(self, metrics: dict | None) -> None:
        """Display key-value metrics on the stage widget."""
        if not metrics:
            self.metrics_label.setText("")
            return
        lines = [f"{k}: {v}" for k, v in metrics.items()]
        self.metrics_label.setText("; ".join(lines))
