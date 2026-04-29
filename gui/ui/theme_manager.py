"""Centralized theming utilities for all GUI windows."""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtGui import QColor

_THEME_JSON_PATH = Path(__file__).with_name("custom_widgets_style.json")

_REQUIRED_PALETTE_KEYS = {
    "bg",
    "panel",
    "text",
    "accent",
    "accent_text",
    "subtext",
    "input_bg",
    "input_border",
    "button_bg",
    "button_hover",
    "group_border",
    "prediction_bg",
    "prediction_text",
    "status_info_bg",
    "status_info_text",
    "status_warning_bg",
    "status_warning_text",
    "status_error_bg",
    "status_error_text",
    "badge_connected_bg",
    "badge_connected_text",
    "badge_disconnected_bg",
    "badge_disconnected_text",
    "badge_loading_bg",
    "badge_loading_text",
    "confusion_text_low",
    "confusion_text_high",
}

_REQUIRED_PLOT_LINE_KEYS = {
    "line_a",
    "line_b",
    "line_c",
    "line_d",
    "line_e",
}


def _as_string_map(value: object, section: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise RuntimeError(f"Theme JSON section '{section}' must be an object.")
    result: dict[str, str] = {}
    for key, item in value.items():
        result[str(key)] = str(item)
    return result


def _as_rgb_triplet(value: object, section: str) -> tuple[int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise RuntimeError(
            f"Theme JSON section '{section}' must be a 3-item list like [r, g, b]."
        )
    try:
        red = max(0, min(255, int(value[0])))
        green = max(0, min(255, int(value[1])))
        blue = max(0, min(255, int(value[2])))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Theme JSON section '{section}' must contain integer RGB values."
        ) from exc
    return red, green, blue


def _load_theme_config() -> tuple[
    str,
    dict[str, dict[str, str]],
    dict[str, str],
    tuple[int, int, int],
    tuple[int, int, int],
]:
    try:
        raw = json.loads(_THEME_JSON_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load theme configuration from '{_THEME_JSON_PATH}'."
        ) from exc

    if not isinstance(raw, dict):
        raise RuntimeError("Theme JSON root must be an object.")

    app_theme = raw.get("AppThemeConfig")
    if not isinstance(app_theme, dict):
        raise RuntimeError("Theme JSON is missing required object 'AppThemeConfig'.")

    default_theme = str(app_theme.get("default_theme", "dark")).strip().lower()

    raw_palettes = app_theme.get("palettes")
    if not isinstance(raw_palettes, dict):
        raise RuntimeError(
            "Theme JSON is missing required object 'AppThemeConfig.palettes'."
        )

    palettes: dict[str, dict[str, str]] = {}
    for theme_name in ("dark", "light"):
        palette = _as_string_map(
            raw_palettes.get(theme_name),
            f"AppThemeConfig.palettes.{theme_name}",
        )
        missing = sorted(_REQUIRED_PALETTE_KEYS.difference(palette.keys()))
        if missing:
            raise RuntimeError(
                "Theme JSON palette "
                f"'AppThemeConfig.palettes.{theme_name}' is missing keys: {', '.join(missing)}"
            )
        palettes[theme_name] = palette

    if default_theme not in palettes:
        default_theme = "dark"

    plot_line_colors = _as_string_map(
        app_theme.get("plot_line_colors"),
        "AppThemeConfig.plot_line_colors",
    )
    missing_plot_keys = sorted(
        _REQUIRED_PLOT_LINE_KEYS.difference(plot_line_colors.keys())
    )
    if missing_plot_keys:
        raise RuntimeError(
            "Theme JSON section 'AppThemeConfig.plot_line_colors' "
            f"is missing keys: {', '.join(missing_plot_keys)}"
        )

    confusion_gradient = app_theme.get("confusion_gradient")
    if not isinstance(confusion_gradient, dict):
        raise RuntimeError(
            "Theme JSON is missing required object 'AppThemeConfig.confusion_gradient'."
        )

    low_rgb = _as_rgb_triplet(
        confusion_gradient.get("low_rgb"),
        "AppThemeConfig.confusion_gradient.low_rgb",
    )
    high_rgb = _as_rgb_triplet(
        confusion_gradient.get("high_rgb"),
        "AppThemeConfig.confusion_gradient.high_rgb",
    )

    return default_theme, palettes, plot_line_colors, low_rgb, high_rgb


DEFAULT_THEME, _PALETTES, _PLOT_LINE_COLORS, _CONFUSION_LOW_RGB, _CONFUSION_HIGH_RGB = (
    _load_theme_config()
)


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
                border: 1px solid {palette['accent']};
                border-bottom: 2px solid {palette['accent']};
            }}
            QLineEdit, QComboBox, QTextEdit, QPlainTextEdit, QTableWidget {{
                background: {palette['input_bg']};
                border: 1px solid {palette['input_border']};
                border-radius: 6px;
                padding: 4px;
            }}
            QLineEdit:focus, QComboBox:focus, QTextEdit:focus, QPlainTextEdit:focus, QTableWidget:focus {{
                border: 1px solid {palette['accent']};
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
            QPushButton:hover {{
                background: {palette['accent']};
                color: {palette['accent_text']};
                border: 1px solid {palette['accent']};
            }}
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
            QProgressBar {{
                background: {palette['input_bg']};
                border: 1px solid {palette['input_border']};
                border-radius: 4px;
                text-align: center;
            }}
            QProgressBar::chunk {{
                background-color: {palette['accent']};
                border-radius: 4px;
            }}
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
                border: 1px solid {palette['accent']};
                border-bottom: 2px solid {palette['accent']};
            }}
            QLineEdit, QComboBox, QPlainTextEdit, QTableWidget, QSpinBox, QDoubleSpinBox {{
                background: {palette['input_bg']};
                border: 1px solid {palette['input_border']};
                border-radius: 6px;
                padding: 4px;
            }}
            QLineEdit:focus, QComboBox:focus, QPlainTextEdit:focus, QTableWidget:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
                border: 1px solid {palette['accent']};
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
            QPushButton:hover {{
                background: {palette['accent']};
                color: {palette['accent_text']};
                border: 1px solid {palette['accent']};
            }}
            QLabel#title {{ font-size: 30px; font-weight: 700; }}
            QLabel#subtitle {{ color: {palette['subtext']}; }}
            QLabel#statusInfo {{ {status_info_style} }}
            QLabel#panelTitle {{ font-size: 20px; font-weight: 700; }}
            QGroupBox {{ font-weight: 600; margin-top: 8px; border: 1px solid {palette['group_border']}; border-radius: 8px; }}
            QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 3px; }}
            QProgressBar {{
                background: {palette['input_bg']};
                border: 1px solid {palette['input_border']};
                border-radius: 4px;
                text-align: center;
            }}
            QProgressBar::chunk {{
                background-color: {palette['accent']};
                border-radius: 4px;
            }}
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
