"""Left, center, and right panel builders for the dashboard window."""

# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportGeneralTypeIssues=false

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class AppWindowPanelsMixin:
    """Build the left, center, and right panels of the dashboard."""

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(8)

        self.control_center_label = QLabel("")
        self.control_center_label.setObjectName("panelTitle")
        layout.addWidget(self.control_center_label)

        self.runtime_label = QLabel("")
        self.connection_label = QLabel("")
        self.model_status_label = QLabel("")
        layout.addWidget(self.runtime_label)
        layout.addWidget(self.connection_label)
        layout.addWidget(self.model_status_label)

        self.start_stop_btn = QPushButton("")
        self.start_stop_btn.clicked.connect(self.toggle_stream)
        self.start_stop_btn.setToolTip(self._t("tooltip_stream_toggle"))
        layout.addWidget(self.start_stop_btn)

        self.clear_btn = QPushButton("")
        self.clear_btn.clicked.connect(self.clear_sentence)
        self.clear_btn.setToolTip(self._t("tooltip_clear_sentence"))
        layout.addWidget(self.clear_btn)

        self.copy_btn = QPushButton("")
        self.copy_btn.clicked.connect(self.copy_sentence)
        layout.addWidget(self.copy_btn)

        self.export_btn = QPushButton("")
        self.export_btn.clicked.connect(self.export_sentence_text)
        self.export_btn.setToolTip(self._t("tooltip_export_sentence"))
        layout.addWidget(self.export_btn)

        layout.addStretch(1)
        return panel

    def _build_center_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)

        self.prediction_card = QLabel(self._t("rest_label"))
        self.prediction_card.setObjectName("predictionCard")
        self.prediction_card.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.prediction_card)

        self.confidence_bar = QProgressBar()
        self.confidence_bar.setRange(0, 100)
        self.confidence_bar.setValue(0)
        layout.addWidget(self.confidence_bar)

        stats = QHBoxLayout()
        self.confidence_label = QLabel("")
        self.word_count_label = QLabel("")
        stats.addWidget(self.confidence_label)
        stats.addStretch(1)
        stats.addWidget(self.word_count_label)
        layout.addLayout(stats)

        sentence_container = QWidget()
        sentence_layout = QVBoxLayout(sentence_container)
        sentence_layout.setContentsMargins(0, 0, 0, 0)
        sentence_layout.setSpacing(6)

        sentence_header = QHBoxLayout()
        self.sentence_header_label = QLabel("")
        sentence_header.addWidget(self.sentence_header_label)
        sentence_header.addStretch(1)

        self.sentence_remove_last_btn = QPushButton("")
        self.sentence_remove_last_btn.clicked.connect(self.remove_last_word)
        sentence_header.addWidget(self.sentence_remove_last_btn)

        self.sentence_clear_btn = QPushButton("")
        self.sentence_clear_btn.clicked.connect(self.clear_sentence)
        sentence_header.addWidget(self.sentence_clear_btn)

        self.sentence_copy_btn = QPushButton("")
        self.sentence_copy_btn.clicked.connect(self.copy_sentence)
        sentence_header.addWidget(self.sentence_copy_btn)
        sentence_layout.addLayout(sentence_header)

        self.sentence_box = QTextEdit()
        self.sentence_box.setReadOnly(True)
        self.sentence_box.setPlainText("")
        sentence_layout.addWidget(self.sentence_box)

        refined_container = QWidget()
        refined_layout = QVBoxLayout(refined_container)
        refined_layout.setContentsMargins(0, 0, 0, 0)
        refined_layout.setSpacing(6)

        refined_header = QHBoxLayout()
        self.refined_header_label = QLabel("")
        refined_header.addWidget(self.refined_header_label)
        refined_header.addStretch(1)

        self.refined_refine_btn = QPushButton("")
        self.refined_refine_btn.clicked.connect(self.request_llm_refinement)
        refined_header.addWidget(self.refined_refine_btn)

        self.refined_copy_btn = QPushButton("")
        self.refined_copy_btn.clicked.connect(self.copy_refined)
        refined_header.addWidget(self.refined_copy_btn)
        refined_layout.addLayout(refined_header)

        self.llm_progress_label = QLabel("")
        refined_layout.addWidget(self.llm_progress_label)

        self.llm_backend_label = QLabel("")
        refined_layout.addWidget(self.llm_backend_label)

        self.refined_box = QTextEdit()
        self.refined_box.setReadOnly(True)
        self.refined_box.setPlainText("")
        refined_layout.addWidget(self.refined_box)

        layout.addWidget(sentence_container, stretch=1)
        layout.addWidget(refined_container, stretch=1)

        history_container = QWidget()
        history_layout = QVBoxLayout(history_container)
        history_layout.setContentsMargins(0, 0, 0, 0)
        history_layout.setSpacing(6)
        self.history_label = QLabel("")
        history_layout.addWidget(self.history_label)

        self.history_table = QTableWidget(0, 3)
        self.history_table.setHorizontalHeaderLabels(["", "", ""])
        self.history_table.verticalHeader().setVisible(False)
        self.history_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.history_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self.history_table.horizontalHeader().setStretchLastSection(True)
        history_layout.addWidget(self.history_table)

        layout.addWidget(history_container, stretch=1)

        return panel

    def _build_right_panel(self) -> QWidget:
        self.right_tabs = QTabWidget()
        self.right_tabs.setObjectName("rightTabs")
        self.settings_tab = self._build_settings_tab()
        self.logs_tab = self._build_logs_tab()
        self.right_tabs.addTab(self.settings_tab, "")
        self.right_tabs.addTab(self.logs_tab, "")
        return self.right_tabs
