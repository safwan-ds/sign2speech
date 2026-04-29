import sys
from collections import deque

import pandas as pd
import serial
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget

from config import (
    BAUD_RATE,
)
from gui.ui.trace_preview_widget import TracePreviewWidget
from utils.serial_utils import (
    detect_glove_ports,
    select_serial_port,
    connect_serial,
    parse_sensor_data,
)


class SerialDebugWindow(QMainWindow):
    def __init__(self, ser):
        super().__init__()
        self.setWindowTitle("Real-time Sensor Debug")
        self.resize(1000, 800)

        self.ser = ser
        self.max_points = 100
        self.data_history = deque(maxlen=self.max_points)

        # UI Setup
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.trace_widget = TracePreviewWidget(self)
        layout.addWidget(self.trace_widget)

        # Timer for reading serial data
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_data)
        self.timer.start(10)  # ~100Hz update attempt

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
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("\nSerial connection closed.")
        event.accept()


def main():
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

    print(f"Connecting to {selected_port} at {BAUD_RATE} baud...")

    try:
        ser = connect_serial(selected_port, BAUD_RATE)
        print("Connected. Opening GUI...")

        app = QApplication(sys.argv)
        window = SerialDebugWindow(ser)
        window.show()
        sys.exit(app.exec())

    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()
