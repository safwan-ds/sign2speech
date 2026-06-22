import csv
import logging
import os
import sys
import threading
import time

import keyboard
import serial

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.architecture import architecture
from config.config import (
    COM_PORT,
    RAW_DATA_DIR,
    setup_logging,
)

from utils.serial_utils import (
    ContinuousWarningBeeper,
    FlexZeroWarningMonitor,
    parse_sensor_data,
    select_serial_port,
)
from utils.plotting import plot_recording
from utils.recording_utils import (
    SENSOR_COLUMNS,
    build_recording_file_path,
    count_csv_samples,
    load_gesture_names,
    sanitize_gesture_label,
)

logger = logging.getLogger(__name__)

_toggle_event = threading.Event()
_quit_event = threading.Event()
_input_event = threading.Event()


def clear_input_buffer() -> None:
    """Clear any pending console input (Windows-safe no-op elsewhere)."""
    try:
        import msvcrt
    except ImportError:
        return

    while msvcrt.kbhit():
        msvcrt.getwch()


def prompt_input(prompt: str) -> str:
    """Read input while pausing hotkeys in the background listener."""
    _input_event.set()
    clear_input_buffer()
    try:
        return input(prompt)
    finally:
        clear_input_buffer()
        _input_event.clear()


def keyboard_listener():
    """Monitors keystrokes in background."""
    while not _quit_event.is_set():
        if _input_event.is_set():
            time.sleep(0.05)
            continue
        if keyboard.is_pressed("s"):
            _toggle_event.set()
            time.sleep(architecture.gui.keyboard_debounce_delay)
        elif keyboard.is_pressed("esc"):
            _quit_event.set()
            break
        time.sleep(0.05)


def main():
    setup_logging("log")

    if not os.path.exists(RAW_DATA_DIR):
        os.makedirs(RAW_DATA_DIR)

    logger.info(" GESTURE DATA RECORDER (THREAD-SAFE)")

    try:
        selected_port = select_serial_port(COM_PORT)
        if not selected_port:
            logger.error("No serial ports detected.")
            return

        logger.info(f"Connecting to {selected_port}...")
        ser = serial.Serial(selected_port, architecture.hardware.baud_rate, timeout=architecture.hardware.timeout)
        time.sleep(architecture.hardware.serial_connection_delay)
        logger.info("Connected!")
    except Exception as e:
        logger.error(f"Error connecting: {e}")
        return

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parsed_gestures = load_gesture_names(project_root)
    if not parsed_gestures:
        logger.error("No gestures found in config/gestures.txt")
        return

    gesture_counts: dict[str, int] = {}
    for gesture_name in parsed_gestures:
        gesture_counts[gesture_name] = count_csv_samples(
            gesture_name, base_dir=RAW_DATA_DIR
        )

    logger.info("Available gestures:")
    for i, gesture_name in enumerate(parsed_gestures):
        count = gesture_counts[gesture_name]
        logger.info(f"  {i}: {gesture_name} [{count} samples]")

    gesture_label = None
    while gesture_label is None:
        try:
            num = int(
                prompt_input(f"Enter gesture number (0-{len(parsed_gestures)-1}): ")
            )
            if 0 <= num < len(parsed_gestures):
                gesture_label = sanitize_gesture_label(parsed_gestures[num])
            else:
                logger.warning(
                    f"Invalid number. Please enter 0-{len(parsed_gestures)-1}."
                )
        except ValueError:
            logger.warning("Please enter a valid number.")

    is_recording = False
    t = threading.Thread(target=keyboard_listener)
    t.daemon = True
    t.start()

    logger.info(f"Target Gesture: [{gesture_label}]")
    logger.info("Controls:")
    logger.info("  [S] START Recording / STOP Recording")
    logger.info("  [ESC] QUIT")

    record_start_time = None
    filename = None
    csv_file = None
    csv_writer = None
    warning_beeper = ContinuousWarningBeeper()
    flex_zero_monitor = FlexZeroWarningMonitor(logger)

    gesture_dir = os.path.join(RAW_DATA_DIR, gesture_label)
    try:
        while not _quit_event.is_set():
            if _toggle_event.is_set():
                _toggle_event.clear()
                if not is_recording:
                    if not os.path.exists(gesture_dir):
                        os.makedirs(gesture_dir)

                    filename = str(
                        build_recording_file_path(gesture_label, base_dir=RAW_DATA_DIR)
                    )
                    csv_file = open(filename, "w", newline="")
                    csv_writer = csv.writer(csv_file)

                    record_start_time = time.perf_counter()

                    header = ["t_ms", *SENSOR_COLUMNS]
                    csv_writer.writerow(header)

                    logger.info(f"RECORDING STARTED: {filename}")
                    is_recording = True
                else:
                    if csv_file:
                        csv_file.close()
                        csv_file = None
                        csv_writer = None

                    print()
                    logger.info("RECORDING STOPPED. File Saved.")
                    is_recording = False
                    record_start_time = None

                    if filename:
                        if plot_recording(filename):
                            decision = (
                                prompt_input(
                                    "Plot saved. Do you want to keep this recording? (y/n): "
                                )
                                .strip()
                                .lower()
                            )
                            while decision not in ("y", "n"):
                                decision = (
                                    prompt_input(
                                        "Please enter 'y' to keep or 'n' to delete: "
                                    )
                                    .strip()
                                    .lower()
                                )

                            if decision == "y":
                                logger.info("Recording kept.")
                            if decision == "n":
                                os.remove(filename)
                                logger.info("Recording deleted.")

                            logger.info(
                                f"{gesture_label} samples: {len(os.listdir(gesture_dir))}"
                            )

            try:
                if ser.in_waiting:
                    line = ser.readline().decode("utf-8", errors="ignore")
                    sensor_dict = parse_sensor_data(line)

                    if sensor_dict:
                        zero_sensors = flex_zero_monitor.check(sensor_dict)
                        if zero_sensors:
                            warning_beeper.start()
                        else:
                            warning_beeper.stop()
                        if (
                            is_recording
                            and csv_writer
                            and record_start_time is not None
                        ):
                            t_ms = int((time.perf_counter() - record_start_time) * 1000)
                            row = [t_ms] + [sensor_dict[k] for k in SENSOR_COLUMNS]
                            csv_writer.writerow(row)
                            print(".", end="", flush=True)
            except (serial.SerialException, OSError) as e:
                logger.error(f"Serial connection lost: {e}")
                break
            except Exception as e:
                logger.error(f"Error reading line: {e}")

    except KeyboardInterrupt:
        pass
    finally:
        warning_beeper.stop()
        if csv_file:
            csv_file.close()
        ser.close()
        logger.info("Exiting.")


if __name__ == "__main__":
    main()
