"""Unit tests for centralized GUI theme manager helpers."""

from PySide6.QtGui import QColor

from gui.ui.theme_manager import (
    DEFAULT_THEME,
    build_dashboard_stylesheet,
    build_data_manager_stylesheet,
    get_confusion_cell_color,
    get_confusion_text_color,
    get_connection_badge_style,
    get_model_badge_style,
    get_palette,
    get_plot_palette,
    get_status_banner_style,
    normalize_theme,
)


REQUIRED_PALETTE_KEYS = {
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
}


PLOT_KEYS = {
    "figure_bg",
    "axes_bg",
    "text",
    "grid",
    "spine",
    "line_a",
    "line_b",
    "line_c",
    "line_d",
    "line_e",
}


def test_normalize_theme_falls_back_to_default() -> None:
    assert normalize_theme("unknown") == DEFAULT_THEME
    assert normalize_theme(None) == DEFAULT_THEME


def test_get_palette_contains_required_keys_for_both_themes() -> None:
    for theme in ("dark", "light"):
        palette = get_palette(theme)
        assert REQUIRED_PALETTE_KEYS.issubset(palette.keys())


def test_plot_palette_contains_expected_keys() -> None:
    plot_palette = get_plot_palette("dark")
    assert PLOT_KEYS.issubset(plot_palette.keys())


def test_core_theme_values_match_24_modern_palette() -> None:
    expected = {
        "dark": {
            "bg": "#21272a",
            "text": "#fefefe",
            "accent": "#fba43b",
        },
        "light": {
            "bg": "#ffffff",
            "text": "#010000",
            "accent": "#26bae3",
        },
    }

    for theme, expected_values in expected.items():
        palette = get_palette(theme)
        for key, expected_value in expected_values.items():
            assert palette[key].lower() == expected_value.lower()


def test_status_banner_styles_include_expected_theme_colors() -> None:
    palette = get_palette("dark")

    error_style = get_status_banner_style("ERROR", "dark")
    warning_style = get_status_banner_style("WARNING", "dark")
    info_style = get_status_banner_style("INFO", "dark")

    assert palette["status_error_bg"] in error_style
    assert palette["status_warning_bg"] in warning_style
    assert palette["status_info_bg"] in info_style


def test_badge_styles_are_resolved_by_state() -> None:
    dark = get_palette("dark")

    connected = get_connection_badge_style(True, "dark")
    disconnected = get_connection_badge_style(False, "dark")
    loading = get_model_badge_style("loading", "dark")
    idle_fallback = get_model_badge_style("unknown", "dark")

    assert dark["badge_connected_bg"] in connected
    assert dark["badge_disconnected_bg"] in disconnected
    assert dark["badge_loading_bg"] in loading
    assert dark["status_warning_bg"] in idle_fallback


def test_confusion_cell_color_clamps_ratio_bounds() -> None:
    low = get_confusion_cell_color(-0.5)
    zero = get_confusion_cell_color(0.0)
    high = get_confusion_cell_color(1.2)
    one = get_confusion_cell_color(1.0)

    assert isinstance(low, QColor)
    assert isinstance(high, QColor)
    assert low.getRgb() == zero.getRgb()
    assert high.getRgb() == one.getRgb()


def test_confusion_text_color_switches_at_threshold() -> None:
    dark = get_palette("dark")

    low = get_confusion_text_color(0.54, "dark")
    high = get_confusion_text_color(0.55, "dark")

    assert low.name().lower() == QColor(dark["confusion_text_low"]).name().lower()
    assert high.name().lower() == QColor(dark["confusion_text_high"]).name().lower()


def test_stylesheet_builders_emit_key_selectors() -> None:
    dashboard_qss = build_dashboard_stylesheet("dark")
    manager_qss = build_data_manager_stylesheet("dark")

    assert "QMainWindow" in dashboard_qss
    assert "QLabel#predictionCard" in dashboard_qss
    assert "QMainWindow" in manager_qss
    assert "QTabBar::tab" in manager_qss
