"""Dialog to view and edit the gestures list stored in config/gestures.json."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from config.config import (
    GESTURES_EDITOR_DIALOG_HEIGHT,
    GESTURES_EDITOR_DIALOG_WIDTH,
)
from utils.recording_utils import load_gestures, save_gestures


class GesturesEditorDialog(QDialog):
    """Dialog to view and edit the gestures list stored in config/gestures.json."""

    def __init__(self, project_root: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.project_root = project_root
        self.setWindowTitle("Manage Gestures")
        self.resize(GESTURES_EDITOR_DIALOG_WIDTH, GESTURES_EDITOR_DIALOG_HEIGHT)

        layout = QVBoxLayout(self)

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Name", "Translation"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(
            QTableWidget.EditTrigger.DoubleClicked
            | QTableWidget.EditTrigger.SelectedClicked
            | QTableWidget.EditTrigger.EditKeyPressed
        )
        layout.addWidget(self.table, stretch=1)

        btn_row = QHBoxLayout()
        self.add_btn = QPushButton("Add")
        self.add_btn.clicked.connect(self._add_row)
        btn_row.addWidget(self.add_btn)

        self.delete_btn = QPushButton("Delete Selected")
        self.delete_btn.clicked.connect(self._delete_selected)
        btn_row.addWidget(self.delete_btn)

        btn_row.addStretch(1)

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.Save | QDialogButtonBox.Cancel
        )
        self.button_box.accepted.connect(self._on_save)
        self.button_box.rejected.connect(self.reject)
        btn_row.addWidget(self.button_box)

        layout.addLayout(btn_row)

        self._load()

    def _load(self) -> None:
        """Load gesture entries into the table."""
        self.table.setRowCount(0)
        entries = []
        try:
            entries = load_gestures(self.project_root)
        except Exception:
            entries = []

        for entry in entries:
            name = entry.get("name", "") if isinstance(entry, dict) else ""
            trans = entry.get("translation", "") if isinstance(entry, dict) else ""
            self._insert_row(str(name), str(trans))

    def _insert_row(self, name: str = "", translation: str = "") -> None:
        """Insert a row into the table."""
        row = self.table.rowCount()
        self.table.insertRow(row)
        name_item = QTableWidgetItem(name)
        trans_item = QTableWidgetItem(translation)
        self.table.setItem(row, 0, name_item)
        self.table.setItem(row, 1, trans_item)

    def _add_row(self) -> None:
        """Add a new empty row and start editing."""
        self._insert_row()
        # Start editing the name cell of the new row
        new_row = self.table.rowCount() - 1
        self.table.editItem(self.table.item(new_row, 0))

    def _delete_selected(self) -> None:
        """Delete currently selected rows."""
        rows = sorted({idx.row() for idx in self.table.selectedIndexes()}, reverse=True)
        if not rows:
            return
        for r in rows:
            self.table.removeRow(r)

    def _on_save(self) -> None:
        """Validate and save the gesture entries."""
        # Collect entries
        entries: list[dict[str, str]] = []
        seen: set[str] = set()
        for row in range(self.table.rowCount()):
            name_item = self.table.item(row, 0)
            trans_item = self.table.item(row, 1)
            name = (name_item.text() if name_item else "").strip()
            translation = (trans_item.text() if trans_item else "").strip()
            if not name:
                QMessageBox.warning(
                    self,
                    "Invalid Entry",
                    f"Row {row+1} has empty name. Please provide a name or delete the row.",
                )
                return
            key = name.lower()
            if key in seen:
                QMessageBox.warning(
                    self,
                    "Duplicate Entry",
                    f"Duplicate gesture name: '{name}'. Names must be unique.",
                )
                return
            seen.add(key)
            entries.append({"name": name, "translation": translation})

        try:
            save_gestures(self.project_root, entries)
        except Exception as exc:
            QMessageBox.critical(self, "Save Failed", f"Could not save gestures: {exc}")
            return

        QMessageBox.information(self, "Saved", "Gestures saved successfully.")
        self.accept()
