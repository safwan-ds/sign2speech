"""Sample discovery and quarantine helpers for the data manager GUI."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import shutil

import pandas as pd


@dataclass(slots=True)
class SampleRecord:
    """Metadata for one recorded CSV sample."""

    gesture: str
    csv_path: Path
    meta_path: Path | None
    row_count: int
    orientation: str
    recorded_at: str
    source: str

    @property
    def file_name(self) -> str:
        return self.csv_path.name


class SampleReviewService:
    """Manage raw sample listing and quarantine lifecycle."""

    QUARANTINE_DIR_NAME = "_quarantine"

    def __init__(self, raw_root: Path) -> None:
        self.raw_root = raw_root
        self.quarantine_root = raw_root / self.QUARANTINE_DIR_NAME

    def list_samples(self, include_quarantine: bool = False) -> list[SampleRecord]:
        records: list[SampleRecord] = []
        records.extend(self._collect_from_root(self.raw_root, source="raw"))
        if include_quarantine:
            records.extend(
                self._collect_from_root(self.quarantine_root, source="quarantine")
            )

        records.sort(key=lambda item: item.csv_path.name.lower(), reverse=True)
        return records

    def quarantine_sample(self, sample_path: Path) -> Path:
        """Move a raw sample and sidecar metadata into quarantine."""
        sample_path = sample_path.resolve()
        if self._is_quarantined(sample_path):
            return sample_path

        relative = sample_path.relative_to(self.raw_root)
        target = self.quarantine_root / relative
        target = self._ensure_unique_path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        moved_csv = Path(shutil.move(str(sample_path), str(target)))

        src_meta = sample_path.with_suffix(".meta.json")
        if src_meta.exists():
            dst_meta = moved_csv.with_suffix(".meta.json")
            dst_meta.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src_meta), str(dst_meta))

        return moved_csv

    def restore_sample(self, sample_path: Path) -> Path:
        """Restore a quarantined sample back into raw gesture folder."""
        sample_path = sample_path.resolve()
        if not self._is_quarantined(sample_path):
            return sample_path

        relative = sample_path.relative_to(self.quarantine_root)
        target = self.raw_root / relative
        target = self._ensure_unique_path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        moved_csv = Path(shutil.move(str(sample_path), str(target)))

        src_meta = sample_path.with_suffix(".meta.json")
        if src_meta.exists():
            dst_meta = moved_csv.with_suffix(".meta.json")
            dst_meta.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src_meta), str(dst_meta))

        return moved_csv

    def _collect_from_root(self, root: Path, source: str) -> list[SampleRecord]:
        if not root.exists():
            return []

        records: list[SampleRecord] = []
        for csv_path in root.rglob("*.csv"):
            if csv_path.parent == root:
                continue
            if self.QUARANTINE_DIR_NAME in csv_path.parts and source != "quarantine":
                continue

            gesture = self._gesture_name(csv_path, source)
            meta_path = csv_path.with_suffix(".meta.json")
            orientation = "unspecified"
            recorded_at = "-"
            if meta_path.exists():
                try:
                    with meta_path.open("r", encoding="utf-8") as handle:
                        payload = json.load(handle)
                    orientation = str(payload.get("orientation", orientation))
                    recorded_at = str(payload.get("recorded_at", recorded_at))
                except Exception:
                    pass

            row_count = self._safe_row_count(csv_path)
            records.append(
                SampleRecord(
                    gesture=gesture,
                    csv_path=csv_path,
                    meta_path=meta_path if meta_path.exists() else None,
                    row_count=row_count,
                    orientation=orientation,
                    recorded_at=recorded_at,
                    source=source,
                )
            )
        return records

    def _gesture_name(self, csv_path: Path, source: str) -> str:
        if source == "quarantine":
            try:
                return csv_path.relative_to(self.quarantine_root).parts[0]
            except Exception:
                return csv_path.parent.name
        try:
            return csv_path.relative_to(self.raw_root).parts[0]
        except Exception:
            return csv_path.parent.name

    @staticmethod
    def _safe_row_count(csv_path: Path) -> int:
        try:
            return int(pd.read_csv(csv_path).shape[0])
        except Exception:
            return 0

    def _is_quarantined(self, sample_path: Path) -> bool:
        try:
            sample_path.relative_to(self.quarantine_root)
            return True
        except ValueError:
            return False

    @staticmethod
    def _ensure_unique_path(path: Path) -> Path:
        if not path.exists():
            return path

        stem = path.stem
        suffix = path.suffix
        parent = path.parent
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        candidate = parent / f"{stem}_{timestamp}{suffix}"
        index = 1
        while candidate.exists():
            candidate = parent / f"{stem}_{timestamp}_{index}{suffix}"
            index += 1
        return candidate
