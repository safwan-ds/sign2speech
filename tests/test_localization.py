"""Unit tests for gui/ui/localization.py."""

from __future__ import annotations

from pathlib import Path

from gui.ui.localization import _load_language_file, LOCALIZATION


class TestLoadLanguageFile:
    def test_loads_valid_json(self, tmp_path: Path) -> None:
        lang_file = tmp_path / "en.json"
        lang_file.write_text('{"title": "Sign2Speech", "ready": "Ready"}', encoding="utf-8")
        result = _load_language_file(lang_file)
        assert result["title"] == "Sign2Speech"
        assert result["ready"] == "Ready"

    def test_returns_empty_dict_for_missing_file(self, tmp_path: Path) -> None:
        result = _load_language_file(tmp_path / "missing.json")
        assert result == {}

    def test_returns_empty_dict_for_non_dict_json(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "bad.json"
        bad_file.write_text('["not", "a", "dict"]', encoding="utf-8")
        result = _load_language_file(bad_file)
        assert result == {}

    def test_all_values_are_strings(self, tmp_path: Path) -> None:
        lang_file = tmp_path / "mixed.json"
        lang_file.write_text('{"count": 42, "label": "hello"}', encoding="utf-8")
        result = _load_language_file(lang_file)
        for v in result.values():
            assert isinstance(v, str)


class TestLocalizationDict:
    def test_contains_tr_and_en_keys(self) -> None:
        assert "tr" in LOCALIZATION
        assert "en" in LOCALIZATION

    def test_tr_locale_is_dict(self) -> None:
        assert isinstance(LOCALIZATION["tr"], dict)

    def test_en_locale_is_dict(self) -> None:
        assert isinstance(LOCALIZATION["en"], dict)

    def test_tr_locale_has_title_key(self) -> None:
        assert "title" in LOCALIZATION["tr"]

    def test_en_locale_has_title_key(self) -> None:
        assert "title" in LOCALIZATION["en"]

    def test_both_locales_have_same_keys(self) -> None:
        tr_keys = set(LOCALIZATION["tr"].keys())
        en_keys = set(LOCALIZATION["en"].keys())
        assert tr_keys == en_keys
