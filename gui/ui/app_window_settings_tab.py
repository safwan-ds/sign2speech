"""Settings and logs tab builders for the dashboard window."""

# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportGeneralTypeIssues=false

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QCheckBox
from PySide6.QtWidgets import QComboBox
from PySide6.QtWidgets import QGridLayout
from PySide6.QtWidgets import QGroupBox
from PySide6.QtWidgets import QHBoxLayout
from PySide6.QtWidgets import QLabel
from PySide6.QtWidgets import QLineEdit
from PySide6.QtWidgets import QListWidget
from PySide6.QtWidgets import QPlainTextEdit
from PySide6.QtWidgets import QProgressBar
from PySide6.QtWidgets import QPushButton
from PySide6.QtWidgets import QScrollArea
from PySide6.QtWidgets import QSlider
from PySide6.QtWidgets import QVBoxLayout
from PySide6.QtWidgets import QWidget

from config.architecture import architecture


class AppWindowSettingsTabMixin:
    """Build the settings tab and logs tab for the right panel."""

    def _build_settings_tab(self) -> QWidget:
        tab = QScrollArea()
        tab.setObjectName("settingsTab")
        tab.setWidgetResizable(True)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(10)

        self.model_group = QGroupBox("")
        model_layout = QVBoxLayout(self.model_group)
        model_layout.setSpacing(8)

        self.model_picker_label = QLabel("")
        model_layout.addWidget(self.model_picker_label)

        picker_row = QHBoxLayout()
        self.model_dir_combo = QComboBox()
        self.model_dir_combo.currentIndexChanged.connect(
            self._on_model_selection_changed
        )
        picker_row.addWidget(self.model_dir_combo, stretch=1)

        self.refresh_models_btn = QPushButton("")
        self.refresh_models_btn.clicked.connect(self.refresh_model_dirs)
        picker_row.addWidget(self.refresh_models_btn)
        model_layout.addLayout(picker_row)

        self.model_path_label = QLabel("")
        model_layout.addWidget(self.model_path_label)

        self.model_path_edit = QLineEdit(self.model_dir)
        self.model_path_edit.setReadOnly(True)
        model_layout.addWidget(self.model_path_edit)

        model_btns = QHBoxLayout()
        self.model_browse_btn = QPushButton("")
        self.model_browse_btn.clicked.connect(self.select_model_dir)
        self.latest_btn = QPushButton("")
        self.latest_btn.clicked.connect(self.use_latest_model)
        self.load_btn = QPushButton("")
        self.load_btn.clicked.connect(self.load_model_async)
        model_btns.addWidget(self.model_browse_btn)
        model_btns.addWidget(self.latest_btn)
        model_btns.addWidget(self.load_btn)
        model_layout.addLayout(model_btns)

        self.model_load_progress = QProgressBar()
        self.model_load_progress.setObjectName("modelLoadProgress")
        self.model_load_progress.setRange(0, 0)
        self.model_load_progress.setTextVisible(False)
        self.model_load_progress.setVisible(False)
        self.model_load_progress.setFixedHeight(4)
        model_layout.addWidget(self.model_load_progress)

        model_stats = QGridLayout()
        model_stats.setHorizontalSpacing(6)
        model_stats.setVerticalSpacing(6)
        model_stats.setColumnStretch(0, 1)
        model_stats.setColumnStretch(1, 1)

        self.model_classes_card = QWidget()
        self.model_classes_card.setObjectName("modelMetricCard")
        classes_card_layout = QVBoxLayout(self.model_classes_card)
        classes_card_layout.setContentsMargins(10, 8, 10, 8)
        classes_card_layout.setSpacing(2)
        self.model_classes_stat_title = QLabel("")
        self.model_classes_stat_title.setObjectName("modelMetricLabel")
        self.model_classes_stat_title.setWordWrap(True)
        self.model_classes_value = QLabel("0")
        self.model_classes_value.setObjectName("modelMetricValue")
        self.model_classes_value.setMinimumHeight(20)
        classes_card_layout.addWidget(self.model_classes_stat_title)
        classes_card_layout.addWidget(self.model_classes_value)
        model_stats.addWidget(self.model_classes_card, 0, 0)

        self.model_sequence_card = QWidget()
        self.model_sequence_card.setObjectName("modelMetricCard")
        sequence_card_layout = QVBoxLayout(self.model_sequence_card)
        sequence_card_layout.setContentsMargins(10, 8, 10, 8)
        sequence_card_layout.setSpacing(2)
        self.model_sequence_stat_title = QLabel("")
        self.model_sequence_stat_title.setObjectName("modelMetricLabel")
        self.model_sequence_stat_title.setWordWrap(True)
        self.model_sequence_value = QLabel("--")
        self.model_sequence_value.setObjectName("modelMetricValue")
        self.model_sequence_value.setMinimumHeight(20)
        sequence_card_layout.addWidget(self.model_sequence_stat_title)
        sequence_card_layout.addWidget(self.model_sequence_value)
        model_stats.addWidget(self.model_sequence_card, 0, 1)

        self.model_input_card = QWidget()
        self.model_input_card.setObjectName("modelMetricCard")
        input_card_layout = QVBoxLayout(self.model_input_card)
        input_card_layout.setContentsMargins(10, 8, 10, 8)
        input_card_layout.setSpacing(2)
        self.model_input_stat_title = QLabel("")
        self.model_input_stat_title.setObjectName("modelMetricLabel")
        self.model_input_stat_title.setWordWrap(True)
        self.model_input_value = QLabel("--")
        self.model_input_value.setObjectName("modelMetricValue")
        self.model_input_value.setMinimumHeight(20)
        input_card_layout.addWidget(self.model_input_stat_title)
        input_card_layout.addWidget(self.model_input_value)
        model_stats.addWidget(self.model_input_card, 1, 0)

        self.model_loaded_card = QWidget()
        self.model_loaded_card.setObjectName("modelMetricCard")
        loaded_card_layout = QVBoxLayout(self.model_loaded_card)
        loaded_card_layout.setContentsMargins(10, 8, 10, 8)
        loaded_card_layout.setSpacing(2)
        self.model_loaded_stat_title = QLabel("")
        self.model_loaded_stat_title.setObjectName("modelMetricLabel")
        self.model_loaded_stat_title.setWordWrap(True)
        self.model_loaded_value = QLabel("--")
        self.model_loaded_value.setObjectName("modelMetricValue")
        self.model_loaded_value.setMinimumHeight(20)
        loaded_card_layout.addWidget(self.model_loaded_stat_title)
        loaded_card_layout.addWidget(self.model_loaded_value)
        model_stats.addWidget(self.model_loaded_card, 1, 1)

        model_layout.addLayout(model_stats)

        self.model_classes_header = QLabel("")
        model_layout.addWidget(self.model_classes_header)

        self.model_class_filter = QLineEdit()
        self.model_class_filter.textChanged.connect(self._filter_model_classes)
        model_layout.addWidget(self.model_class_filter)

        self.model_classes_list = QListWidget()
        self.model_classes_list.setObjectName("modelClassesList")
        self.model_classes_list.setAlternatingRowColors(True)
        self.model_classes_list.setMinimumHeight(120)
        model_layout.addWidget(self.model_classes_list, stretch=1)

        layout.addWidget(self.model_group)

        self.settings_group = QGroupBox("")
        settings_layout = QGridLayout(self.settings_group)

        self.serial_port_label = QLabel("")
        settings_layout.addWidget(self.serial_port_label, 0, 0)
        self.port_combo = QComboBox()
        self.port_combo.currentTextChanged.connect(
            lambda _text: self._refresh_action_states()
        )
        settings_layout.addWidget(self.port_combo, 1, 0, 1, 2)

        self.refresh_ports_btn = QPushButton("")
        self.refresh_ports_btn.clicked.connect(self.refresh_ports)
        settings_layout.addWidget(self.refresh_ports_btn, 2, 0, 1, 2)

        self.baud_rate_label = QLabel("")
        settings_layout.addWidget(self.baud_rate_label, 3, 0)
        self.baud_edit = QLineEdit(str(architecture.hardware.baud_rate))
        settings_layout.addWidget(self.baud_edit, 4, 0, 1, 2)

        self.threshold_label = QLabel("")
        settings_layout.addWidget(self.threshold_label, 5, 0, 1, 2)
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(30, 99)
        self.threshold_slider.setValue(int(architecture.prediction.confidence_threshold * 100))
        self.threshold_slider.valueChanged.connect(self._on_threshold_change)
        settings_layout.addWidget(self.threshold_slider, 6, 0, 1, 2)

        self.smoothing_label = QLabel("")
        settings_layout.addWidget(self.smoothing_label, 7, 0, 1, 2)
        self.smoothing_slider = QSlider(Qt.Orientation.Horizontal)
        self.smoothing_slider.setRange(1, 12)
        self.smoothing_slider.valueChanged.connect(self._on_smoothing_change)
        self.smoothing_slider.setValue(architecture.motion_detection.motion_detection_smoothing_window)
        settings_layout.addWidget(self.smoothing_slider, 8, 0, 1, 2)

        self.llm_checkbox = QCheckBox("")
        self.llm_checkbox.stateChanged.connect(self._on_llm_changed)
        self.llm_checkbox.setChecked(architecture.llm.use_qwen_llm)
        settings_layout.addWidget(self.llm_checkbox, 9, 0, 1, 2)

        from config.config import LLM_BACKEND as _DEFAULT_LLM_BE

        self.llm_backend_label = QLabel("")
        settings_layout.addWidget(self.llm_backend_label, 10, 0)
        self.llm_backend_combo = QComboBox()
        self.llm_backend_combo.addItem("", "local")
        self.llm_backend_combo.addItem("", "remote")
        idx = self.llm_backend_combo.findData(_DEFAULT_LLM_BE)
        self.llm_backend_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.llm_backend_combo.currentIndexChanged.connect(self._on_llm_backend_changed)
        settings_layout.addWidget(self.llm_backend_combo, 11, 0)

        self.tts_mode_label = QLabel("")
        self.tts_mode_combo = QComboBox()
        self.tts_status_label = QLabel("")
        self.tts_status_value_label = QLabel("")
        self.tts_checkbox = QCheckBox("")
        self.tts_checkbox.stateChanged.connect(self._on_tts_changed)
        self.tts_checkbox.setChecked(architecture.general.use_tts)
        settings_layout.addWidget(self.tts_checkbox, 12, 0, 1, 2)

        self.ensemble_checkbox = QCheckBox("")
        self.ensemble_checkbox.stateChanged.connect(self._on_ensemble_changed)
        self.ensemble_checkbox.setChecked(architecture.training.use_ensemble)
        settings_layout.addWidget(self.ensemble_checkbox, 13, 0, 1, 2)

        settings_layout.addWidget(self.tts_mode_label, 14, 0, 1, 2)
        self.tts_mode_combo.addItem("", "instant")
        self.tts_mode_combo.addItem("", "llm")
        self.tts_mode_combo.addItem("", "hybrid")
        self.tts_mode_combo.currentIndexChanged.connect(self._on_tts_mode_changed)
        self.tts_mode_combo.setCurrentIndex(2)
        settings_layout.addWidget(self.tts_mode_combo, 15, 0, 1, 2)

        settings_layout.addWidget(self.tts_status_label, 16, 0, 1, 2)
        settings_layout.addWidget(self.tts_status_value_label, 17, 0, 1, 2)

        layout.addWidget(self.settings_group)
        self.refresh_model_dirs()
        layout.addStretch(1)
        tab.setWidget(content)
        return tab

    def _build_logs_tab(self) -> QWidget:
        tab = QWidget()
        tab.setObjectName("logsTab")
        layout = QVBoxLayout(tab)

        controls = QHBoxLayout()
        self.clear_logs_btn = QPushButton("")
        self.clear_logs_btn.clicked.connect(self._clear_logs_view)
        self.export_logs_btn = QPushButton("")
        self.export_logs_btn.clicked.connect(self._export_logs_view)
        controls.addWidget(self.clear_logs_btn)
        controls.addWidget(self.export_logs_btn)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        layout.addWidget(self.log_box)
        return tab
