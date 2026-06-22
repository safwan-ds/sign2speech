import logging
import sys
from collections import deque

import pandas as pd
import serial
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget

from config.architecture import architecture
from gui.ui.trace_preview_widget import TracePreviewWidget
from utils.serial_utils import (
    detect_glove_ports,
    select_serial_port,
    connect_serial,
    FlexZeroWarningMonitor,
    ContinuousWarningBeeper,
    build_flex_zero_warning,
    parse_sensor_data,
)

logger = logging.getLogger(__name__)


class SerialDebugWindow(QMainWindow):
    def __init__(self, ser):
        super().__init__()
        self.default_window_title = "Real-time Sensor Debug"
        self.alert_window_title = "Real-time Sensor Debug - ZERO SENSOR ALERT"
        self.setWindowTitle(self.default_window_title)
        self.resize(1000, 800)

        self.ser = ser
        self.max_points = 100
        self.data_history = deque(maxlen=self.max_points)
        self.flex_zero_monitor = FlexZeroWarningMonitor(logger)
        self.zero_feedback_active = False
        self.last_zero_message = ""
        self._warning_beeper = ContinuousWarningBeeper()

        # UI Setup
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.trace_widget = TracePreviewWidget(self)
        layout.addWidget(self.trace_widget)
        self.statusBar().showMessage("Monitoring sensor stream...")

        # Timer for reading serial data
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_data)
        self.timer.start(10)  # ~100Hz update attempt

    def _show_zero_feedback(self, warning_message: str):
        if not self.zero_feedback_active:
            self._warning_beeper.start()
            self.zero_feedback_active = True
            self.setWindowTitle(self.alert_window_title)
            self.statusBar().setStyleSheet(
                "QStatusBar { background-color: #5c1f1f; color: #ffe8e8; font-weight: bold; }"
            )

        if warning_message != self.last_zero_message:
            self.statusBar().showMessage(warning_message)
            self.last_zero_message = warning_message

    def _clear_zero_feedback(self):
        if not self.zero_feedback_active:
            return
        self.zero_feedback_active = False
        self.last_zero_message = ""
        self._warning_beeper.stop()
        self.setWindowTitle(self.default_window_title)
        self.statusBar().setStyleSheet("")

    def update_data(self):
        if not self.ser or not self.ser.is_open:
            return

        try:
            while self.ser.in_waiting > 0:
                line = self.ser.readline().decode("utf-8", errors="ignore").strip()
                if not line:
                    continue

                parsed_data = parse_sensor_data(line)
                if parsed_data:
                    zero_sensors = self.flex_zero_monitor.check(parsed_data)
                    if zero_sensors:
                        self._show_zero_feedback(build_flex_zero_warning(zero_sensors))
                    else:
                        self._clear_zero_feedback()
                    self.data_history.append(parsed_data)

                    # Update plot
                    df = pd.DataFrame(list(self.data_history))
                    self.trace_widget.plot_dataframe(df)
                else:
                    sys.stdout.write(f"\rRaw: {line[:50]}...")
                    sys.stdout.flush()
        except (serial.SerialException, OSError) as e:
            print(f"\nSerial connection lost: {e}")
            self.timer.stop()
        except Exception as e:
            print(f"\nError reading serial: {e}")

    def closeEvent(self, event):
        self._warning_beeper.stop()
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("\nSerial connection closed.")
        event.accept()


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Try to select a port automatically
    selected_port = select_serial_port()

    # If no port found via description matching, try probing
    if not selected_port:
        glove_ports = detect_glove_ports()
        if glove_ports:
            selected_port = glove_ports[0]

    if not selected_port:
        print("Error: No serial ports detected. Please check your connection.")
        return

    print(f"Connecting to {selected_port} at {architecture.hardware.baud_rate} baud...")

    try:
        ser = connect_serial(selected_port, architecture.hardware.baud_rate)
        print("Connected. Opening GUI...")

        app = QApplication(sys.argv)
        window = SerialDebugWindow(ser)
        window.show()
        sys.exit(app.exec())

    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()
