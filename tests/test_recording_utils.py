"""Unit tests for recording_utils module."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from utils.recording_utils import (
    CSV_COLUMNS,
    sanitize_gesture_label,
    gesture_output_dir,
    build_recording_metadata_path,
    save_rows_to_csv,
    save_recording_metadata,
    load_gesture_names,
    count_csv_samples,
)


class TestSanitizeGestureLabel:
    def test_strips_whitespace(self):
        assert sanitize_gesture_label("  hello  ") == "hello"

    def test_replaces_spaces_with_underscores(self):
        assert sanitize_gesture_label("hello world") == "hello_world"

    def test_removes_special_characters(self):
        assert sanitize_gesture_label("hello!@#$%^&*()") == "hello"

    def test_preserves_hyphens(self):
        assert sanitize_gesture_label("thumb-up") == "thumb-up"

    def test_preserves_alphanumeric(self):
        assert sanitize_gesture_label("Gesture123") == "Gesture123"

    def test_collapses_multiple_underscores(self):
        assert sanitize_gesture_label("hello   world") == "hello_world"

    def test_empty_label_returns_gesture(self):
        assert sanitize_gesture_label("") == "gesture"

    def test_only_special_chars_returns_gesture(self):
        assert sanitize_gesture_label("!@#") == "gesture"


class TestGestureOutputDir:
    def test_returns_path_under_base_dir(self, tmp_path):
        result = gesture_output_dir("hello", base_dir=str(tmp_path))
        assert result == tmp_path / "hello"

    def test_returns_path_object(self, tmp_path):
        result = gesture_output_dir("ok", base_dir=str(tmp_path))
        assert isinstance(result, Path)


class TestBuildRecordingMetadataPath:
    def test_returns_meta_json_sidecar(self, tmp_path):
        csv_path = tmp_path / "hello_20260101_120000.csv"
        meta_path = build_recording_metadata_path(csv_path)
        assert meta_path.suffix == ".json"
        assert meta_path.stem == "hello_20260101_120000.meta"
        assert meta_path.parent == csv_path.parent

    def test_accepts_string_path(self, tmp_path):
        csv_path = str(tmp_path / "gesture.csv")
        meta_path = build_recording_metadata_path(csv_path)
        assert isinstance(meta_path, Path)


class TestSaveRowsToCsv:
    def _make_row(self, t_ms: int = 0) -> dict:
        return {
            "t_ms": t_ms,
            "flex0": 100,
            "flex1": 110,
            "flex2": 120,
            "flex3": 130,
            "flex4": 140,
            "accelX": 1.0,
            "accelY": 2.0,
            "accelZ": 9.8,
            "gyroX": 0.1,
            "gyroY": 0.2,
            "gyroZ": 0.3,
        }

    def test_creates_file_with_correct_header(self, tmp_path):
        target = tmp_path / "gesture" / "hello.csv"
        rows = [self._make_row(0), self._make_row(10)]
        save_rows_to_csv(target, rows)
        assert target.exists()
        with target.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            assert reader.fieldnames == CSV_COLUMNS

    def test_creates_parent_directories(self, tmp_path):
        target = tmp_path / "a" / "b" / "c" / "test.csv"
        save_rows_to_csv(target, [self._make_row()])
        assert target.exists()

    def test_written_row_count_matches_input(self, tmp_path):
        target = tmp_path / "out.csv"
        rows = [self._make_row(i) for i in range(5)]
        save_rows_to_csv(target, rows)
        with target.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            assert sum(1 for _ in reader) == 5

    def test_returns_path_object(self, tmp_path):
        target = tmp_path / "ret.csv"
        result = save_rows_to_csv(target, [self._make_row()])
        assert isinstance(result, Path)
        assert result == target


class TestSaveRecordingMetadata:
    def test_written_json_matches_input(self, tmp_path):
        meta_path = tmp_path / "gesture" / "hello.meta.json"
        payload = {"orientation": "palm_up", "recorded_at": "2026-04-18T12:00:00"}
        save_recording_metadata(meta_path, payload)
        loaded = json.loads(meta_path.read_text(encoding="utf-8"))
        assert loaded == payload

    def test_creates_parent_directories(self, tmp_path):
        meta_path = tmp_path / "a" / "b" / "meta.json"
        save_recording_metadata(meta_path, {"key": "val"})
        assert meta_path.exists()

    def test_returns_path_object(self, tmp_path):
        meta_path = tmp_path / "meta.json"
        result = save_recording_metadata(meta_path, {})
        assert isinstance(result, Path)


class TestLoadGestureNames:
    def test_loads_names_from_file(self, tmp_path):
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "gestures.txt").write_text(
            "hello - merhaba\nme - ben\nthank_you - teşekkürler\n",
            encoding="utf-8",
        )
        names = load_gesture_names(tmp_path)
        assert names == ["hello", "me", "thank_you"]

    def test_skips_empty_lines(self, tmp_path):
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "gestures.txt").write_text(
            "hello - merhaba\n\nme - ben\n", encoding="utf-8"
        )
        names = load_gesture_names(tmp_path)
        assert names == ["hello", "me"]

    def test_returns_empty_list_when_file_missing(self, tmp_path):
        names = load_gesture_names(tmp_path)
        assert names == []

    def test_handles_lines_without_translation_separator(self, tmp_path):
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "gestures.txt").write_text(
            "REST\nhello - merhaba\n", encoding="utf-8"
        )
        names = load_gesture_names(tmp_path)
        assert "REST" in names
        assert "hello" in names


class TestCountCsvSamples:
    def test_returns_zero_for_missing_directory(self, tmp_path):
        count = count_csv_samples("nonexistent", base_dir=str(tmp_path))
        assert count == 0

    def test_counts_only_csv_files(self, tmp_path):
        gesture_dir = tmp_path / "hello"
        gesture_dir.mkdir()
        (gesture_dir / "hello_1.csv").write_text("data", encoding="utf-8")
        (gesture_dir / "hello_2.csv").write_text("data", encoding="utf-8")
        (gesture_dir / "notes.txt").write_text("not csv", encoding="utf-8")
        count = count_csv_samples("hello", base_dir=str(tmp_path))
        assert count == 2

    def test_returns_zero_for_empty_directory(self, tmp_path):
        gesture_dir = tmp_path / "empty"
        gesture_dir.mkdir()
        count = count_csv_samples("empty", base_dir=str(tmp_path))
        assert count == 0
