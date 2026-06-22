"""Shared plotting utilities for Sign2Speech project"""

import csv
import logging
import os

import matplotlib.pyplot as plt

from config.architecture import architecture

logger = logging.getLogger(__name__)


def plot_recording(filename: str, title: str = "", show: bool = True) -> bool:
    """Plot sensor data from a CSV recording.

    Args:
        filename: Path to CSV file
        title: Optional title prefix
        show: If True, call plt.show() after plotting

    Returns:
        True if plot was created successfully
    """
    try:
        data: dict[str, list[float]] = {}
        times: list[float] = []

        with open(filename, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                times.append(float(row["t_ms"]))
                for key in row:
                    if key != "t_ms":
                        if key not in data:
                            data[key] = []
                        data[key].append(float(row[key]))

        if not data:
            logger.error(f"No data in {filename}")
            return False

        # Create subplots for different sensor types
        fig, axes = plt.subplots(architecture.plot.plot_num_rows, architecture.plot.plot_num_cols, figsize=(architecture.plot.plot_figure_width, architecture.plot.plot_figure_height))  # type: ignore
        try:
            manager = fig.canvas.manager
            if manager and hasattr(manager, "window"):
                manager.window.state("zoomed")  # type: ignore
        except Exception:
            pass  # Non-interactive backend or unsupported window manager
        display_title = (
            f"{title}\n{os.path.basename(filename)}"
            if title
            else f"Recording: {os.path.basename(filename)}"
        )
        fig.suptitle(display_title, fontsize=architecture.plot.plot_font_size)  # type: ignore

        # Plot flex sensors
        ax = axes[0]
        for i in range(architecture.hardware.num_flex_sensors):
            key = f"flex{i}"
            if key in data:
                ax.plot(
                    times, data[key], label=key, marker="o", markersize=architecture.plot.plot_marker_size
                )
        ax.set_ylabel("Flex Sensor Values")
        ax.set_xlabel("Time (ms)")
        ax.legend()
        ax.grid(True, alpha=architecture.plot.plot_grid_alpha)
        ax.set_title("Flex Sensors")

        # Plot accelerometer
        ax = axes[1]
        for key in ["accelX", "accelY", "accelZ"]:
            if key in data:
                ax.plot(
                    times, data[key], label=key, marker="o", markersize=architecture.plot.plot_marker_size
                )
        ax.set_ylabel("Acceleration (16-bit raw)")
        ax.set_xlabel("Time (ms)")
        ax.legend()
        ax.grid(True, alpha=architecture.plot.plot_grid_alpha)
        ax.set_title("Accelerometer")

        # Plot gyroscope
        ax = axes[2]
        for key in ["gyroX", "gyroY", "gyroZ"]:
            if key in data:
                ax.plot(
                    times, data[key], label=key, marker="o", markersize=architecture.plot.plot_marker_size
                )
        ax.set_ylabel("Angular Velocity (16-bit raw)")
        ax.set_xlabel("Time (ms)")
        ax.legend()
        ax.grid(True, alpha=architecture.plot.plot_grid_alpha)
        ax.set_title("Gyroscope")

        plt.tight_layout()

        if show:
            plt.show()  # type: ignore

        return True

    except Exception as e:
        logger.error(f"Error plotting: {e}")
        return False
