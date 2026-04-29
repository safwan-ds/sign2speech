"""Test-only compatibility shims for optional GUI dependencies."""

from __future__ import annotations

import sys
import types


def _install_pyside6_stub() -> None:
    qt_module = types.ModuleType("PySide6")
    qtgui_module = types.ModuleType("PySide6.QtGui")
    qtwidgets_module = types.ModuleType("PySide6.QtWidgets")

    class QColor:
        def __init__(self, *args: object) -> None:
            if len(args) == 1 and isinstance(args[0], str):
                value = args[0].strip()
                if not value.startswith("#"):
                    value = f"#{value}"
                if len(value) == 4:
                    value = "#" + "".join(ch * 2 for ch in value[1:])
                self._value = value.lower()
            elif len(args) == 3:
                red, green, blue = (max(0, min(255, int(part))) for part in args)
                self._value = f"#{red:02x}{green:02x}{blue:02x}"
            else:
                raise TypeError("QColor expects '#RRGGBB' or three RGB integers")

        def getRgb(self) -> tuple[int, int, int, int]:
            red = int(self._value[1:3], 16)
            green = int(self._value[3:5], 16)
            blue = int(self._value[5:7], 16)
            return red, green, blue, 255

        def name(self) -> str:
            return self._value

    class QCloseEvent:
        def accept(self) -> None:
            return None

    class QTableWidgetItem:
        def __init__(self, text: str = "") -> None:
            self._text = str(text)

        def text(self) -> str:
            return self._text

    qtgui_module.QColor = QColor
    qtgui_module.QCloseEvent = QCloseEvent
    qtwidgets_module.QTableWidgetItem = QTableWidgetItem
    qt_module.QtGui = qtgui_module
    qt_module.QtWidgets = qtwidgets_module

    sys.modules.setdefault("PySide6", qt_module)
    sys.modules.setdefault("PySide6.QtGui", qtgui_module)
    sys.modules.setdefault("PySide6.QtWidgets", qtwidgets_module)


try:
    from PySide6.QtGui import QColor as _QColor  # noqa: F401
except Exception:
    _install_pyside6_stub()
