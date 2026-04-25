"""CustomWidgets integration helpers for runtime JSON style loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject


class _UiProxy:
    """Expose child widgets by objectName for CustomWidgets style lookup."""


def _build_ui_proxy(window: Any) -> _UiProxy:
    proxy = _UiProxy()
    try:
        children = window.findChildren(QObject)
    except Exception:
        return proxy

    for child in children:
        name = child.objectName()
        if not isinstance(name, str):
            continue
        key = name.strip()
        if not key:
            continue
        try:
            setattr(proxy, key, child)
        except Exception:
            continue
    return proxy


def _ensure_theme_engine(window: Any) -> bool:
    if hasattr(window, "themeEngine"):
        return True
    try:
        from Custom_Widgets.QCustomTheme import QCustomTheme

        window.themeEngine = QCustomTheme()
        return True
    except Exception:
        return False


def _try_load_json_style_current(
    loader: Any,
    window: Any,
    json_file: str,
) -> bool:
    try:
        loader(window, jsonFiles=[json_file])
        return True
    except Exception:
        return False


def _try_load_json_style_legacy(
    loader: Any,
    window: Any,
    ui_proxy: Any,
    json_file: str,
) -> bool:
    payloads = ({json_file}, [json_file], (json_file,))

    for payload in payloads:
        try:
            loader(window, ui_proxy, jsonFiles=payload)
            return True
        except TypeError:
            continue
        except Exception:
            return False

    for payload in payloads:
        try:
            loader(window, ui_proxy, payload)
            return True
        except TypeError:
            continue
        except Exception:
            return False

    return False


def apply_custom_widgets_theme(
    window: Any,
    project_root: Path | str,
    json_path: Path | str | None = None,
) -> bool:
    """Apply CustomWidgets JSON style to a window.

    Returns True when loading succeeds and False when import, file lookup,
    or style loading fails.
    """
    try:
        from Custom_Widgets.JSonStyles import loadJsonStyle
    except Exception:
        try:
            from Custom_Widgets.Widgets import loadJsonStyle
        except Exception:
            return False

    try:
        root = Path(project_root)
        style_path = (
            Path(json_path)
            if json_path is not None
            else root / "gui" / "ui" / "custom_widgets_style.json"
        )
        style_path = style_path.resolve()
        if not style_path.is_file():
            return False

        if not _ensure_theme_engine(window):
            return False

        proxy = _build_ui_proxy(window)
        if not hasattr(window, "ui"):
            try:
                window.ui = proxy
            except Exception:
                pass

        style_file = str(style_path)
        if _try_load_json_style_current(loadJsonStyle, window, style_file):
            return True
        return _try_load_json_style_legacy(loadJsonStyle, window, proxy, style_file)
    except Exception:
        return False
