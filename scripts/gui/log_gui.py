import serial
import time
import csv
import os
import threading
from datetime import datetime
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import matplotlib.pyplot as plt
import sys

# Import config
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from config import (
    COM_PORT,
    BAUD_RATE,
    TIMEOUT,
    LOGS_DIR,
    SERIAL_CONNECTION_DELAY,
)

# Import utilities
from utils.serial_utils import parse_sensor_data, select_serial_port
from utils.plotting import plot_recording
from utils.data_utils import convert_to_snake_case


class GestureLoggerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Gesture Data Logger")
        self.root.geometry("700x600")

        # Setup logs directory
        if not os.path.exists(LOGS_DIR):
            os.makedirs(LOGS_DIR)

        # State variables
        self.ser = None
        self.is_recording = False
        self.csv_writer = None
        self.csv_file = None
        self.record_start_time = None
        self.current_filename = None
        self.gesture_label = None
        self.expected_sensors = [f"flex{i}" for i in range(5)] + [
            "accelX",
            "accelY",
            "accelZ",
            "gyroX",
            "gyroY",
            "gyroZ",
        ]

        # Thread control
        self.running = True
        self.serial_thread = None

        # Load gestures
        self.gestures = self.load_gestures()

        # Create UI
        self.create_widgets()

        # Connect to serial port
        self.connect_serial()

        # Start serial reading thread
        self.start_serial_thread()

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def load_gestures(self):
        """Load gestures from gestures.txt"""
        try:
            gestures_path = os.path.join(
                os.path.dirname(
                    os.path.dirname(
                        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    )
                ),
                "gestures.txt",
            )
            with open(gestures_path, "r") as f:
                gesture_list = [line.strip() for line in f.readlines() if line.strip()]

            parsed_gestures = []
            for line in gesture_list:
                if " - " in line:
                    gesture_name = line.split(" - ")[0].strip()
                else:
                    gesture_name = line.strip()
                parsed_gestures.append((gesture_name, line))

            return parsed_gestures
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load gestures.txt: {e}")
            return []

    def create_widgets(self):
        """Create GUI widgets"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

        # Connection status
        ttk.Label(main_frame, text="Connection Status:").grid(
            row=0, column=0, sticky=tk.W, pady=5
        )
        self.status_label = ttk.Label(main_frame, text="Disconnected", foreground="red")
        self.status_label.grid(row=0, column=1, sticky=tk.W, pady=5)

        # Gesture selection
        ttk.Label(main_frame, text="Select Gesture:").grid(
            row=1, column=0, sticky=tk.W, pady=5
        )

        gesture_frame = ttk.Frame(main_frame)
        gesture_frame.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5)

        self.gesture_var = tk.StringVar()
        self.gesture_combo = ttk.Combobox(
            gesture_frame, textvariable=self.gesture_var, width=40, state="readonly"
        )

        # Add gesture counts
        gesture_display = []
        for gesture_name, full_line in self.gestures:
            gesture_dir = os.path.join(LOGS_DIR, convert_to_snake_case(gesture_name))
            count = 0
            if os.path.exists(gesture_dir):
                count = len([f for f in os.listdir(gesture_dir) if f.endswith(".csv")])
            gesture_display.append(f"{full_line} [{count} samples]")

        self.gesture_combo["values"] = gesture_display
        if gesture_display:
            self.gesture_combo.current(0)
        self.gesture_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Recording status
        ttk.Label(main_frame, text="Recording Status:").grid(
            row=2, column=0, sticky=tk.W, pady=5
        )
        self.recording_label = ttk.Label(
            main_frame, text="Not Recording", foreground="gray"
        )
        self.recording_label.grid(row=2, column=1, sticky=tk.W, pady=5)

        # Current file
        ttk.Label(main_frame, text="Current File:").grid(
            row=3, column=0, sticky=tk.W, pady=5
        )
        self.file_label = ttk.Label(main_frame, text="None", foreground="gray")
        self.file_label.grid(row=3, column=1, sticky=tk.W, pady=5)

        # Sample count
        ttk.Label(main_frame, text="Samples Recorded:").grid(
            row=4, column=0, sticky=tk.W, pady=5
        )
        self.sample_label = ttk.Label(main_frame, text="0", foreground="gray")
        self.sample_label.grid(row=4, column=1, sticky=tk.W, pady=5)

        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=5, column=0, columnspan=2, pady=20)

        self.record_button = ttk.Button(
            button_frame,
            text="Start Recording",
            command=self.toggle_recording,
            width=20,
        )
        self.record_button.pack(side=tk.LEFT, padx=5)

        self.plot_button = ttk.Button(
            button_frame,
            text="Plot Last Recording",
            command=self.plot_last_recording,
            width=20,
        )
        self.plot_button.pack(side=tk.LEFT, padx=5)
        self.plot_button.config(state=tk.DISABLED)

        # Sensor data display
        ttk.Label(main_frame, text="Live Sensor Data:").grid(
            row=6, column=0, columnspan=2, sticky=tk.W, pady=(10, 5)
        )

        self.sensor_text = scrolledtext.ScrolledText(
            main_frame, height=15, width=80, state=tk.DISABLED
        )
        self.sensor_text.grid(
            row=7, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5
        )

        # Configure grid weights for expansion
        main_frame.rowconfigure(7, weight=1)

        # Sample counter
        self.sample_count = 0

    def connect_serial(self):
        """Connect to serial port"""
        try:
            selected_port = select_serial_port(COM_PORT)
            if not selected_port:
                self.update_status("No serial ports detected", "red")
                return

            self.ser = serial.Serial(selected_port, BAUD_RATE, timeout=TIMEOUT)
            time.sleep(SERIAL_CONNECTION_DELAY)  # Allow Arduino reset
            self.update_status(f"Connected to {selected_port}", "green")
        except Exception as e:
            self.update_status(f"Connection error: {e}", "red")
            messagebox.showerror("Connection Error", f"Failed to connect: {e}")

    def update_status(self, text, color):
        """Update connection status label"""
        self.status_label.config(text=text, foreground=color)

    def toggle_recording(self):
        """Start or stop recording"""
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        """Start recording sensor data"""
        if not self.ser or not self.ser.is_open:
            messagebox.showerror("Error", "Serial port not connected")
            return

        # Get selected gesture
        selected_idx = self.gesture_combo.current()
        if selected_idx < 0:
            messagebox.showerror("Error", "Please select a gesture")
            return

        gesture_name, _ = self.gestures[selected_idx]
        self.gesture_label = convert_to_snake_case(gesture_name)

        # Create gesture-specific folder
        gesture_dir = os.path.join(LOGS_DIR, self.gesture_label)
        if not os.path.exists(gesture_dir):
            os.makedirs(gesture_dir)

        # Create CSV file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_filename = os.path.join(
            gesture_dir, f"{self.gesture_label}_{timestamp}.csv"
        )
        self.csv_file = open(self.current_filename, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)

        # Write header
        header = ["t_ms"] + self.expected_sensors
        self.csv_writer.writerow(header)

        # Start recording
        self.record_start_time = time.perf_counter()
        self.is_recording = True
        self.sample_count = 0

        # Update UI
        self.recording_label.config(text="Recording...", foreground="red")
        self.record_button.config(text="Stop Recording")
        self.file_label.config(
            text=os.path.basename(self.current_filename), foreground="blue"
        )
        self.sample_label.config(text="0", foreground="blue")
        self.gesture_combo.config(state=tk.DISABLED)

        self.log_message(f"Started recording: {self.gesture_label}")

    def stop_recording(self):
        """Stop recording sensor data"""
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None

        self.is_recording = False
        self.record_start_time = None

        # Update UI
        self.recording_label.config(text="Not Recording", foreground="gray")
        self.record_button.config(text="Start Recording")
        self.gesture_combo.config(state="readonly")
        self.plot_button.config(state=tk.NORMAL)

        self.log_message(
            f"Stopped recording. Saved {self.sample_count} samples to: {os.path.basename(self.current_filename)}"
        )

        # Update gesture count
        self.update_gesture_counts()

    def update_gesture_counts(self):
        """Update gesture sample counts in dropdown"""
        gesture_display = []
        for gesture_name, full_line in self.gestures:
            gesture_dir = os.path.join(LOGS_DIR, convert_to_snake_case(gesture_name))
            count = 0
            if os.path.exists(gesture_dir):
                count = len([f for f in os.listdir(gesture_dir) if f.endswith(".csv")])
            gesture_display.append(f"{full_line} [{count} samples]")

        current_idx = self.gesture_combo.current()
        self.gesture_combo["values"] = gesture_display
        self.gesture_combo.current(current_idx)

    def plot_last_recording(self):
        """Plot the last recorded file"""
        if self.current_filename and os.path.exists(self.current_filename):
            plot_recording(self.current_filename)
        else:
            messagebox.showwarning("Warning", "No recording available to plot")

    def start_serial_thread(self):
        """Start background thread for reading serial data"""
        self.serial_thread = threading.Thread(
            target=self.serial_reader_loop, daemon=True
        )
        self.serial_thread.start()

    def serial_reader_loop(self):
        """Background loop to read serial data"""
        while self.running:
            if self.ser and self.ser.is_open and self.ser.in_waiting:
                try:
                    line = self.ser.readline().decode("utf-8", errors="ignore")
                    sensor_dict = parse_sensor_data(line)

                    if sensor_dict:
                        # Write to CSV if recording
                        if (
                            self.is_recording
                            and self.csv_writer
                            and self.record_start_time is not None
                        ):
                            t_ms = int(
                                (time.perf_counter() - self.record_start_time) * 1000
                            )
                            row = [t_ms] + [
                                sensor_dict[k] for k in self.expected_sensors
                            ]
                            self.csv_writer.writerow(row)
                            self.sample_count += 1

                            # Update sample count in UI (every 10 samples to reduce overhead)
                            if self.sample_count % 10 == 0:
                                self.root.after(
                                    0,
                                    lambda: self.sample_label.config(
                                        text=str(self.sample_count)
                                    ),
                                )

                        # Display sensor data
                        self.root.after(
                            0, lambda s=sensor_dict: self.display_sensor_data(s)
                        )

                except Exception as e:
                    self.root.after(0, lambda: self.log_message(f"Error reading: {e}"))

            time.sleep(0.01)  # Small delay to prevent CPU spinning

    def display_sensor_data(self, sensor_dict):
        """Display sensor data in text widget"""
        # Format sensor data
        flex_values = [f"F{i}:{sensor_dict[f'flex{i}']:4.0f}" for i in range(5)]
        accel_values = [
            f"{k}:{sensor_dict[k]:6.0f}" for k in ["accelX", "accelY", "accelZ"]
        ]
        gyro_values = [
            f"{k}:{sensor_dict[k]:6.0f}" for k in ["gyroX", "gyroY", "gyroZ"]
        ]

        text = f"Flex: {' '.join(flex_values)} | Accel: {' '.join(accel_values)} | Gyro: {' '.join(gyro_values)}\n"

        self.sensor_text.config(state=tk.NORMAL)
        self.sensor_text.insert(tk.END, text)
        self.sensor_text.see(tk.END)

        # Keep only last 100 lines
        lines = int(self.sensor_text.index("end-1c").split(".")[0])
        if lines > 100:
            self.sensor_text.delete("1.0", "2.0")

        self.sensor_text.config(state=tk.DISABLED)

    def log_message(self, message):
        """Log a message to the sensor text widget"""
        self.sensor_text.config(state=tk.NORMAL)
        self.sensor_text.insert(tk.END, f"\n>>> {message}\n\n")
        self.sensor_text.see(tk.END)
        self.sensor_text.config(state=tk.DISABLED)

    def on_close(self):
        """Handle window close event"""
        if self.is_recording:
            if messagebox.askokcancel(
                "Recording in Progress",
                "Recording is in progress. Stop recording and quit?",
            ):
                self.stop_recording()
            else:
                return

        self.running = False
        if self.serial_thread:
            self.serial_thread.join(timeout=1)
        if self.ser and self.ser.is_open:
            self.ser.close()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = GestureLoggerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
