"""Record new gesture samples from serial and save CSV files into data/raw."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    BAUD_RATE,
    COM_PORT,
    LOGS_DIR,
    SERIAL_CONNECTION_DELAY,
    TIMEOUT,
    setup_logging,
)
from utils.serial_utils import connect_serial, parse_sensor_data
from utils.recording_utils import (
    save_rows_to_csv,
    sanitize_gesture_label,
    build_recording_file_path,
)

logger = logging.getLogger(__name__)


def _record_one_sample(
    ser,
    output_dir: str,
    gesture_label: str,
    duration_seconds: float,
    min_rows: int,
) -> str | None:
    rows: list[dict[str, float | int]] = []

    # Remove stale lines before each capture window.
    ser.reset_input_buffer()
    start = time.perf_counter()

    while (time.perf_counter() - start) < duration_seconds:
        try:
            raw_line = ser.readline().decode("utf-8", errors="ignore").strip()
        except Exception:
            continue

        parsed = parse_sensor_data(raw_line)
        if parsed is None:
            continue

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        row: dict[str, float | int] = {"t_ms": elapsed_ms}
        row.update(parsed)
        rows.append(row)

    if len(rows) < min_rows:
        logger.warning(
            "Sample discarded: only %s valid rows captured (min required: %s)",
            len(rows),
            min_rows,
        )
        return None

    file_path = build_recording_file_path(gesture_label, base_dir=output_dir)
    return str(save_rows_to_csv(file_path, rows))


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record gesture samples into data/raw")
    parser.add_argument("--gesture", required=True, help="Gesture label/folder name")
    parser.add_argument("--port", default=COM_PORT, help="Serial COM port")
    parser.add_argument("--baud", type=int, default=BAUD_RATE, help="Serial baud rate")
    parser.add_argument(
        "--duration",
        type=float,
        default=4.0,
        help="Duration in seconds for each sample",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=3,
        help="Number of samples to record",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=1.0,
        help="Pause in seconds between samples",
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=20,
        help="Minimum valid rows required to save a sample",
    )
    return parser


def main() -> None:
    setup_logging("record_samples")
    parser = _build_arg_parser()
    args = parser.parse_args()

    gesture_label = sanitize_gesture_label(args.gesture)
    if args.count < 1:
        raise ValueError("--count must be at least 1")
    if args.duration <= 0:
        raise ValueError("--duration must be greater than 0")
    if args.pause < 0:
        raise ValueError("--pause must be >= 0")

    output_dir = os.path.join(LOGS_DIR, gesture_label)
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Recording gesture: %s", gesture_label)
    logger.info("Output directory: %s", output_dir)
    logger.info(
        "Config -> port=%s, baud=%s, count=%s, duration=%.2fs, pause=%.2fs",
        args.port,
        args.baud,
        args.count,
        args.duration,
        args.pause,
    )

    ser = connect_serial(args.port, args.baud, timeout=TIMEOUT)
    try:
        time.sleep(SERIAL_CONNECTION_DELAY)
        saved = 0

        for i in range(args.count):
            logger.info("Prepare gesture now: sample %s/%s", i + 1, args.count)
            path = _record_one_sample(
                ser,
                output_dir=output_dir,
                gesture_label=gesture_label,
                duration_seconds=args.duration,
                min_rows=args.min_rows,
            )
            if path:
                saved += 1
                logger.info("Saved: %s", path)

            if i < args.count - 1 and args.pause > 0:
                logger.info("Pause %.2fs before next sample", args.pause)
                time.sleep(args.pause)

        logger.info("Recording complete. Saved %s/%s samples.", saved, args.count)
    finally:
        try:
            ser.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
