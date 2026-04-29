from __future__ import annotations

from pathlib import Path

import pandas as pd

from gui.services.sample_review_service import SampleReviewService


def _write_sample(base: Path, gesture: str, stem: str) -> Path:
    target = base / gesture / f"{stem}.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "flex0": [1, 2],
            "flex1": [3, 4],
            "flex2": [5, 6],
            "flex3": [7, 8],
            "flex4": [9, 10],
            "accelX": [11, 12],
            "accelY": [13, 14],
            "accelZ": [15, 16],
            "gyroX": [17, 18],
            "gyroY": [19, 20],
            "gyroZ": [21, 22],
        }
    )
    frame.to_csv(target, index=False)
    target.with_suffix(".meta.json").write_text(
        '{"orientation":"palm_up","recorded_at":"2026-04-18T12:00:00"}',
        encoding="utf-8",
    )
    return target


def test_list_samples_includes_raw_and_optional_quarantine(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    service = SampleReviewService(raw_root)

    sample = _write_sample(raw_root, "hello", "hello_1")
    service.quarantine_sample(sample)

    raw_records = service.list_samples(include_quarantine=False)
    assert len(raw_records) == 0

    all_records = service.list_samples(include_quarantine=True)
    assert len(all_records) == 1
    assert all_records[0].source == "quarantine"
    assert all_records[0].gesture == "hello"


def test_quarantine_and_restore_move_meta_sidecar(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    service = SampleReviewService(raw_root)

    sample = _write_sample(raw_root, "ok", "ok_1")
    meta = sample.with_suffix(".meta.json")

    quarantined = service.quarantine_sample(sample)
    assert quarantined.exists()
    assert not sample.exists()
    assert not meta.exists()
    assert quarantined.with_suffix(".meta.json").exists()

    restored = service.restore_sample(quarantined)
    assert restored.exists()
    assert restored.with_suffix(".meta.json").exists()
    assert not quarantined.exists()


def test_quarantine_without_meta_sidecar(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    service = SampleReviewService(raw_root)

    # Write sample but remove the meta file
    sample = _write_sample(raw_root, "ok", "ok_nometa")
    sample.with_suffix(".meta.json").unlink()

    quarantined = service.quarantine_sample(sample)
    assert quarantined.exists()
    assert not quarantined.with_suffix(".meta.json").exists()


def test_quarantine_already_quarantined_is_idempotent(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    service = SampleReviewService(raw_root)

    sample = _write_sample(raw_root, "ok", "ok_2")
    quarantined = service.quarantine_sample(sample)
    # Calling again on already-quarantined path should return it unchanged
    result = service.quarantine_sample(quarantined)
    assert result == quarantined


def test_restore_non_quarantined_is_idempotent(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    service = SampleReviewService(raw_root)

    sample = _write_sample(raw_root, "ok", "ok_3")
    # Not quarantined – restore should return path unchanged
    result = service.restore_sample(sample)
    assert result == sample.resolve()


def test_list_samples_returns_raw_records(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    service = SampleReviewService(raw_root)

    _write_sample(raw_root, "hello", "hello_1")
    _write_sample(raw_root, "hello", "hello_2")

    records = service.list_samples(include_quarantine=False)
    assert len(records) == 2
    assert all(r.source == "raw" for r in records)
    assert all(r.gesture == "hello" for r in records)


def test_sample_record_without_meta_has_default_orientation(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    service = SampleReviewService(raw_root)

    sample = _write_sample(raw_root, "hello", "hello_nometa")
    sample.with_suffix(".meta.json").unlink()

    records = service.list_samples()
    assert len(records) == 1
    assert records[0].orientation == "unspecified"
    assert records[0].meta_path is None
