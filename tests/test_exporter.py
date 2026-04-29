"""Unit tests for gui/utils/exporter.py."""

from __future__ import annotations

from pathlib import Path

from gui.utils.exporter import export_sentence


def test_export_sentence_creates_file(tmp_path: Path) -> None:
    export_dir = tmp_path / "exports"
    path = export_sentence("hello world", export_dir)
    assert path.exists()


def test_export_sentence_creates_export_dir(tmp_path: Path) -> None:
    export_dir = tmp_path / "new" / "nested" / "dir"
    export_sentence("hello", export_dir)
    assert export_dir.is_dir()


def test_export_sentence_file_has_txt_extension(tmp_path: Path) -> None:
    path = export_sentence("test", tmp_path)
    assert path.suffix == ".txt"


def test_export_sentence_content_matches(tmp_path: Path) -> None:
    path = export_sentence("hello world", tmp_path)
    content = path.read_text(encoding="utf-8")
    assert content.strip() == "hello world"


def test_export_sentence_strips_leading_trailing_whitespace(tmp_path: Path) -> None:
    path = export_sentence("  spaces  ", tmp_path)
    content = path.read_text(encoding="utf-8")
    assert content.strip() == "spaces"


def test_export_sentence_filename_starts_with_prefix(tmp_path: Path) -> None:
    path = export_sentence("hi", tmp_path)
    assert path.name.startswith("sentence_")


def test_export_sentence_returns_path_object(tmp_path: Path) -> None:
    result = export_sentence("text", tmp_path)
    assert isinstance(result, Path)
