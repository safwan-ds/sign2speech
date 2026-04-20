"""Centralized theming utilities for all GUI windows."""

from __future__ import annotations

from PySide6.QtGui import QColor

DEFAULT_THEME = "dark"

_PALETTES: dict[str, dict[str, str]] = {
    "light": {
        "bg": "#f6f7fb",
        "panel": "#ffffff",
        "text": "#1f2937",
        "subtext": "#5a6777",
        "input_bg": "#ffffff",
        "input_border": "#d1d5db",
        "button_bg": "#f3f4f6",
        "button_hover": "#e5e7eb",
        "group_border": "#d1d5db",
        "prediction_bg": "#eef2ff",
        "prediction_text": "#13264d",
        "status_info_bg": "#e9f5ec",
        "status_info_text": "#12381f",
        "status_warning_bg": "#fff4e5",
        "status_warning_text": "#7d5200",
        "status_error_bg": "#fdecea",
        "status_error_text": "#8a1c1c",
        "badge_connected_bg": "#e7f6ea",
        "badge_connected_text": "#1e6c35",
        "badge_disconnected_bg": "#f4ecec",
        "badge_disconnected_text": "#7d1f1f",
        "badge_loading_bg": "#e8f0ff",
        "badge_loading_text": "#1f4a86",
        "confusion_text_low": "#0f172a",
        "confusion_text_high": "#f8fafc",
    },
    "dark": {
        "bg": "#1f2329",
        "panel": "#2b3139",
        "text": "#e5e7eb",
        "subtext": "#b3bcc9",
        "input_bg": "#242a31",
        "input_border": "#3d4651",
        "button_bg": "#353d48",
        "button_hover": "#46505d",
        "group_border": "#4b5563",
        "prediction_bg": "#2f3540",
        "prediction_text": "#f3f4f6",
        "status_info_bg": "#e9f5ec",
        "status_info_text": "#12381f",
        "status_warning_bg": "#fff4e5",
        "status_warning_text": "#7d5200",
        "status_error_bg": "#fdecea",
        "status_error_text": "#8a1c1c",
        "badge_connected_bg": "#e7f6ea",
        "badge_connected_text": "#1e6c35",
        "badge_disconnected_bg": "#f4ecec",
        "badge_disconnected_text": "#7d1f1f",
        "badge_loading_bg": "#e8f0ff",
        "badge_loading_text": "#1f4a86",
        "confusion_text_low": "#0f172a",
        "confusion_text_high": "#f8fafc",
    },
}

_PLOT_LINE_COLORS: dict[str, str] = {
    "line_a": "#63b3ed",
    "line_b": "#f6ad55",
    "line_c": "#68d391",
    "line_d": "#f687b3",
    "line_e": "#f6e05e",
}

_CONFUSION_LOW_RGB = (227, 242, 253)
_CONFUSION_HIGH_RGB = (13, 71, 161)


def normalize_theme(theme: str | None) -> str:
    """Return a known theme name, falling back to dark."""
    if isinstance(theme, str) and theme in _PALETTES:
        return theme
    return DEFAULT_THEME


def get_palette(theme: str | None = None) -> dict[str, str]:
    """Return a copy of the selected GUI color palette."""
    resolved = normalize_theme(theme)
    return dict(_PALETTES[resolved])


def get_plot_palette(theme: str | None = None) -> dict[str, str]:
    """Return plot colors derived from the selected theme palette."""
    palette = get_palette(theme)
    result = {
        "figure_bg": palette["panel"],
        "axes_bg": palette["input_bg"],
        "text": palette["text"],
        "grid": palette["group_border"],
        "spine": palette["input_border"],
    }
    result.update(_PLOT_LINE_COLORS)
    return result


def _box_style(background: str, text: str, padding: str = "8px 10px") -> str:
    return (
        f"padding: {padding}; border-radius: 6px; "
        f"background: {background}; color: {text};"
    )


def get_status_banner_style(level: str, theme: str | None = None) -> str:
    """Return stylesheet for status banner severity."""
    palette = get_palette(theme)
    normalized = str(level).upper()
    if normalized == "ERROR":
        return _box_style(palette["status_error_bg"], palette["status_error_text"])
    if normalized == "WARNING":
        return _box_style(
            palette["status_warning_bg"],
            palette["status_warning_text"],
        )
    return _box_style(palette["status_info_bg"], palette["status_info_text"])


def get_connection_badge_style(
    connected: bool,
    theme: str | None = None,
) -> str:
    """Return stylesheet for device connection badge."""
    palette = get_palette(theme)
    if connected:
        return _box_style(
            palette["badge_connected_bg"],
            palette["badge_connected_text"],
            padding="6px 8px",
        )
    return _box_style(
        palette["badge_disconnected_bg"],
        palette["badge_disconnected_text"],
        padding="6px 8px",
    )


def get_model_badge_style(state: str, theme: str | None = None) -> str:
    """Return stylesheet for model lifecycle badge state."""
    palette = get_palette(theme)
    normalized = str(state).lower()
    if normalized == "loading":
        return _box_style(
            palette["badge_loading_bg"],
            palette["badge_loading_text"],
            padding="6px 8px",
        )
    if normalized == "ready":
        return _box_style(
            palette["badge_connected_bg"],
            palette["badge_connected_text"],
            padding="6px 8px",
        )
    if normalized == "error":
        return _box_style(
            palette["status_error_bg"],
            palette["status_error_text"],
            padding="6px 8px",
        )
    return _box_style(
        palette["status_warning_bg"],
        palette["status_warning_text"],
        padding="6px 8px",
    )


def build_dashboard_stylesheet(theme: str | None = None) -> str:
    """Build the dashboard window stylesheet for a given theme."""
    palette = get_palette(theme)
    status_info_style = _box_style(
        palette["status_info_bg"],
        palette["status_info_text"],
    )
    return f"""
            QMainWindow, QWidget#centralRoot {{ background: {palette['bg']}; }}
            QWidget {{ color: {palette['text']}; }}
            QWidget#centralRoot {{ background: {palette['bg']}; }}
            QTabWidget::pane, QGroupBox, QPlainTextEdit, QTextEdit, QTableWidget, QStatusBar {{ background: {palette['panel']}; }}
            QTabWidget#rightTabs {{ background: {palette['panel']}; }}
            QScrollArea#settingsTab, QWidget#logsTab {{ background: {palette['panel']}; }}
            QScrollArea#settingsTab {{ border: none; }}
            QScrollArea#settingsTab > QWidget > QWidget {{ background: {palette['panel']}; }}
            QTabWidget#rightTabs QGroupBox {{ background: {palette['panel']}; }}
            QTabBar::tab {{
                background: {palette['button_bg']};
                border: 1px solid {palette['input_border']};
                border-bottom: none;
                padding: 6px 12px;
                margin-right: 2px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
            }}
            QTabBar::tab:selected {{
                background: {palette['panel']};
                color: {palette['text']};
            }}
            QLineEdit, QComboBox, QTextEdit, QPlainTextEdit, QTableWidget {{
                background: {palette['input_bg']};
                border: 1px solid {palette['input_border']};
                border-radius: 6px;
                padding: 4px;
            }}
            QHeaderView::section {{
                background: {palette['button_bg']};
                color: {palette['text']};
                border: 1px solid {palette['input_border']};
                padding: 4px;
            }}
            QPushButton {{
                background: {palette['button_bg']};
                border: 1px solid {palette['input_border']};
                border-radius: 6px;
                padding: 6px 10px;
            }}
            QPushButton:hover {{ background: {palette['button_hover']}; }}
            QLabel#title {{ font-size: 30px; font-weight: 700; }}
            QLabel#subtitle {{ color: {palette['subtext']}; }}
            QLabel#statusInfo {{ {status_info_style} }}
            QLabel#predictionCard {{
                font-size: 64px;
                font-weight: 700;
                border-radius: 12px;
                padding: 20px;
                background: {palette['prediction_bg']};
                color: {palette['prediction_text']};
            }}
            QLabel#panelTitle {{ font-size: 20px; font-weight: 700; }}
            QWidget#modelMetricCard {{
                background: {palette['input_bg']};
                border: 1px solid {palette['input_border']};
                border-radius: 8px;
                min-height: 76px;
            }}
            QLabel#modelMetricLabel {{ color: {palette['subtext']}; font-size: 11px; }}
            QLabel#modelMetricValue {{ font-size: 16px; font-weight: 700; }}
            QListWidget#modelClassesList {{
                background: {palette['input_bg']};
                border: 1px solid {palette['input_border']};
                border-radius: 8px;
            }}
            QGroupBox {{ font-weight: 600; margin-top: 8px; border: 1px solid {palette['group_border']}; border-radius: 8px; }}
            QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 3px; }}
            """


def build_data_manager_stylesheet(theme: str | None = None) -> str:
    """Build the data manager window stylesheet for a given theme."""
    palette = get_palette(theme)
    status_info_style = _box_style(
        palette["status_info_bg"],
        palette["status_info_text"],
    )
    return f"""
            QMainWindow, QWidget#centralRoot {{ background: {palette['bg']}; }}
            QWidget {{ color: {palette['text']}; }}
            QTabWidget::pane, QGroupBox, QPlainTextEdit, QTableWidget, QStatusBar {{ background: {palette['panel']}; }}
            QTabBar::tab {{
                background: {palette['button_bg']};
                border: 1px solid {palette['input_border']};
                border-bottom: none;
                padding: 6px 12px;
                margin-right: 2px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
            }}
            QTabBar::tab:selected {{
                background: {palette['panel']};
                color: {palette['text']};
            }}
            QLineEdit, QComboBox, QPlainTextEdit, QTableWidget, QSpinBox, QDoubleSpinBox {{
                background: {palette['input_bg']};
                border: 1px solid {palette['input_border']};
                border-radius: 6px;
                padding: 4px;
            }}
            QHeaderView::section {{
                background: {palette['button_bg']};
                color: {palette['text']};
                border: 1px solid {palette['input_border']};
                padding: 4px;
            }}
            QPushButton {{
                background: {palette['button_bg']};
                border: 1px solid {palette['input_border']};
                border-radius: 6px;
                padding: 6px 10px;
            }}
            QPushButton:hover {{ background: {palette['button_hover']}; }}
            QLabel#title {{ font-size: 30px; font-weight: 700; }}
            QLabel#subtitle {{ color: {palette['subtext']}; }}
            QLabel#statusInfo {{ {status_info_style} }}
            QLabel#panelTitle {{ font-size: 20px; font-weight: 700; }}
            QGroupBox {{ font-weight: 600; margin-top: 8px; border: 1px solid {palette['group_border']}; border-radius: 8px; }}
            QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 3px; }}
            """


def get_confusion_cell_color(ratio: float, theme: str | None = None) -> QColor:
    """Return confusion-matrix cell color for normalized value in [0, 1]."""
    del theme
    value = max(0.0, min(1.0, ratio))
    red = int(
        _CONFUSION_LOW_RGB[0] + (_CONFUSION_HIGH_RGB[0] - _CONFUSION_LOW_RGB[0]) * value
    )
    green = int(
        _CONFUSION_LOW_RGB[1] + (_CONFUSION_HIGH_RGB[1] - _CONFUSION_LOW_RGB[1]) * value
    )
    blue = int(
        _CONFUSION_LOW_RGB[2] + (_CONFUSION_HIGH_RGB[2] - _CONFUSION_LOW_RGB[2]) * value
    )
    return QColor(red, green, blue)


def get_confusion_text_color(ratio: float, theme: str | None = None) -> QColor:
    """Return readable confusion-matrix text color for a given normalized value."""
    palette = get_palette(theme)
    if ratio >= 0.55:
        return QColor(palette["confusion_text_high"])
    return QColor(palette["confusion_text_low"])
