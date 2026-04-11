"""GUI application for reviewing and cleaning gesture log files."""

import os
import csv
import sys
import tkinter as tk
from tkinter import ttk, messagebox

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from config import LOGS_DIR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GESTURES_PATH = os.path.join(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ),
    "gestures.txt",
)


def load_gesture_names() -> list[str]:
    """Load available gesture names from gestures.txt."""
    gesture_names: list[str] = []
    if os.path.exists(GESTURES_PATH):
        with open(GESTURES_PATH, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                gesture_names.append(
                    line.split(" - ")[0].strip() if " - " in line else line
                )
    return gesture_names


def read_csv_data(filepath: str) -> tuple[list[float], dict[str, list[float]]]:
    """Read a CSV log file and return (times, data_dict)."""
    data: dict[str, list[float]] = {}
    times: list[float] = []
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row["t_ms"]))
            for key in row:
                if key != "t_ms":
                    data.setdefault(key, []).append(float(row[key]))
    return times, data


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

BATCH_SIZE = 4


class ReviewApp(tk.Tk):
    """Tkinter application for reviewing and cleaning log files."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Log File Reviewer & Cleaner")
        self.state("zoomed")
        self.minsize(900, 600)

        self.gestures = load_gesture_names()
        self.selected_gesture: str = ""
        self.gesture_dir: str = ""
        self.log_files: list[str] = []
        self.current_batch: int = 0
        self.total_batches: int = 0

        # file path -> True means marked for deletion
        self.marked_for_deletion: dict[str, bool] = {}
        # checkbutton variables for the current batch
        self.check_vars: list[tk.BooleanVar] = []

        self._build_gesture_selector()

    # ---- Gesture selection screen -----------------------------------------

    def _build_gesture_selector(self) -> None:
        """Show the gesture selection screen."""
        self._clear()

        frame = ttk.Frame(self, padding=20)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text="Select a gesture to review", font=("", 14, "bold")).pack(
            pady=(0, 12)
        )

        list_frame = ttk.Frame(frame)
        list_frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.gesture_listbox = tk.Listbox(
            list_frame,
            font=("Consolas", 12),
            yscrollcommand=scrollbar.set,
            activestyle="none",
        )
        self.gesture_listbox.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.gesture_listbox.yview)

        for gesture in self.gestures:
            gesture_dir = os.path.join(LOGS_DIR, gesture)
            count = 0
            if os.path.isdir(gesture_dir):
                count = len([f for f in os.listdir(gesture_dir) if f.endswith(".csv")])
            self.gesture_listbox.insert(tk.END, f"  {gesture}  ({count} samples)")

        self.gesture_listbox.bind(
            "<Double-Button-1>", lambda _: self._on_gesture_selected()
        )

        btn_frame = ttk.Frame(frame)
        btn_frame.pack(pady=12)
        ttk.Button(btn_frame, text="Open", command=self._on_gesture_selected).pack()

    def _on_gesture_selected(self) -> None:
        sel = self.gesture_listbox.curselection()
        if not sel:
            messagebox.showwarning("No selection", "Please select a gesture first.")
            return

        self.selected_gesture = self.gestures[sel[0]]
        self.gesture_dir = os.path.join(LOGS_DIR, self.selected_gesture)
        if not os.path.isdir(self.gesture_dir):
            messagebox.showerror("Error", f"Directory not found:\n{self.gesture_dir}")
            return

        self.log_files = sorted(
            [f for f in os.listdir(self.gesture_dir) if f.endswith(".csv")]
        )
        if not self.log_files:
            messagebox.showinfo("Empty", f"No CSV files for '{self.selected_gesture}'.")
            return

        self.total_batches = (len(self.log_files) + BATCH_SIZE - 1) // BATCH_SIZE
        self.current_batch = 0
        self.marked_for_deletion = {}
        self._build_review_screen()

    # ---- Review screen ----------------------------------------------------

    def _build_review_screen(self) -> None:
        """Build (or rebuild) the batch review screen."""
        self._clear()

        # ---- Top bar ----
        top = ttk.Frame(self, padding=(10, 6))
        top.pack(fill=tk.X)

        ttk.Button(
            top, text="\u2190 Gestures", command=self._build_gesture_selector
        ).pack(side=tk.LEFT)

        self.batch_label = ttk.Label(top, font=("", 12, "bold"))
        self.batch_label.pack(side=tk.LEFT, padx=20)

        ttk.Button(top, text="Delete Marked Files", command=self._confirm_delete).pack(
            side=tk.RIGHT
        )

        self.marked_count_label = ttk.Label(top, foreground="red")
        self.marked_count_label.pack(side=tk.RIGHT, padx=10)

        # ---- Navigation bar ----
        nav = ttk.Frame(self, padding=(10, 4))
        nav.pack(fill=tk.X)

        self.prev_btn = ttk.Button(
            nav, text="\u25c0 Previous", command=self._prev_batch
        )
        self.prev_btn.pack(side=tk.LEFT)

        self.next_btn = ttk.Button(nav, text="Next \u25b6", command=self._next_batch)
        self.next_btn.pack(side=tk.RIGHT)

        # ---- Content (plots + checkboxes) ----
        self.content_frame = ttk.Frame(self)
        self.content_frame.pack(fill=tk.BOTH, expand=True)

        self._show_batch()

    def _show_batch(self) -> None:
        """Render the current batch of plots."""
        # Destroy previous canvas if any
        for child in self.content_frame.winfo_children():
            child.destroy()

        start = self.current_batch * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(self.log_files))
        batch = self.log_files[start:end]

        # Update labels / buttons
        self.batch_label.config(
            text=f"{self.selected_gesture}  —  Batch {self.current_batch + 1}/{self.total_batches}  "
            f"(files {start + 1}–{end} of {len(self.log_files)})"
        )
        self.prev_btn.config(state=tk.NORMAL if self.current_batch > 0 else tk.DISABLED)
        self.next_btn.config(
            state=(
                tk.NORMAL
                if self.current_batch < self.total_batches - 1
                else tk.DISABLED
            )
        )
        self._update_marked_count()

        # ---- Matplotlib figure ----
        nrows = len(batch)
        fig, axes = plt.subplots(nrows, 3, figsize=(14, 3 * nrows), squeeze=False)
        fig.subplots_adjust(hspace=0.45, wspace=0.30, top=0.95, bottom=0.05)

        self.check_vars = []

        for row_idx, filename in enumerate(batch):
            filepath = os.path.join(self.gesture_dir, filename)
            try:
                times, data = read_csv_data(filepath)
            except Exception:
                for col in range(3):
                    axes[row_idx][col].text(
                        0.5,
                        0.5,
                        "Error reading file",
                        ha="center",
                        va="center",
                        transform=axes[row_idx][col].transAxes,
                    )
                self.check_vars.append(
                    tk.BooleanVar(value=self.marked_for_deletion.get(filepath, False))
                )
                continue

            # Flex sensors
            ax = axes[row_idx][0]
            for i in range(5):
                key = f"flex{i}"
                if key in data:
                    ax.plot(times, data[key], linewidth=0.8)
            ax.set_ylabel("Flex", fontsize=8)
            ax.set_title(filename, fontsize=8, fontweight="bold")
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)

            # Accelerometer
            ax = axes[row_idx][1]
            for key in ("accelX", "accelY", "accelZ"):
                if key in data:
                    ax.plot(times, data[key], linewidth=0.8)
            ax.set_ylabel("Accel", fontsize=8)
            ax.set_title("Accelerometer", fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)

            # Gyroscope
            ax = axes[row_idx][2]
            for key in ("gyroX", "gyroY", "gyroZ"):
                if key in data:
                    ax.plot(times, data[key], linewidth=0.8)
            ax.set_ylabel("Gyro", fontsize=8)
            ax.set_title("Gyroscope", fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)

            # Checkbox variable – restore previous state if user revisits batch
            var = tk.BooleanVar(value=self.marked_for_deletion.get(filepath, False))
            self.check_vars.append(var)

        # Scrollable area -------
        outer = ttk.Frame(self.content_frame)
        outer.pack(fill=tk.BOTH, expand=True)

        vscroll = ttk.Scrollbar(outer, orient=tk.VERTICAL)
        vscroll.pack(side=tk.RIGHT, fill=tk.Y)

        self._scroll_canvas = tk.Canvas(outer, yscrollcommand=vscroll.set)
        self._scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vscroll.config(command=self._scroll_canvas.yview)

        inner = ttk.Frame(self._scroll_canvas)

        # Embed matplotlib figure inside inner frame (correct parent)
        canvas = FigureCanvasTkAgg(fig, master=inner)
        canvas.draw()
        plot_widget = canvas.get_tk_widget()
        # Give the plot widget an explicit pixel height so it doesn't collapse
        plot_height = int(fig.get_figheight() * fig.dpi)
        plot_widget.config(height=plot_height)
        plot_widget.pack(fill=tk.X, expand=False)

        # Checkboxes row
        cb_frame = ttk.Frame(inner, padding=(10, 6))
        cb_frame.pack(fill=tk.X)
        ttk.Label(cb_frame, text="Mark for deletion:", font=("", 10, "bold")).pack(
            side=tk.LEFT, padx=(0, 10)
        )
        for idx, filename in enumerate(batch):
            filepath = os.path.join(self.gesture_dir, filename)
            cb = ttk.Checkbutton(
                cb_frame,
                text=filename,
                variable=self.check_vars[idx],
                command=lambda fp=filepath, v=self.check_vars[
                    idx
                ]: self._on_check_toggled(fp, v),
            )
            cb.pack(side=tk.LEFT, padx=6)

        # Place inner frame inside scrollable canvas
        self._scroll_canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.update_idletasks()
        self._scroll_canvas.config(scrollregion=self._scroll_canvas.bbox("all"))
        self._scroll_canvas.bind_all(
            "<MouseWheel>",
            lambda e: self._scroll_canvas.yview_scroll(int(-e.delta / 120), "units"),
        )

        # keep a reference so the figure isn't garbage-collected
        self._current_fig = fig

    # ---- Callbacks --------------------------------------------------------

    def _on_check_toggled(self, filepath: str, var: tk.BooleanVar) -> None:
        if var.get():
            self.marked_for_deletion[filepath] = True
        else:
            self.marked_for_deletion.pop(filepath, None)
        self._update_marked_count()

    def _update_marked_count(self) -> None:
        n = sum(1 for v in self.marked_for_deletion.values() if v)
        self.marked_count_label.config(
            text=f"{n} file(s) marked for deletion" if n else ""
        )

    def _prev_batch(self) -> None:
        if self.current_batch > 0:
            self.current_batch -= 1
            self._show_batch()

    def _next_batch(self) -> None:
        if self.current_batch < self.total_batches - 1:
            self.current_batch += 1
            self._show_batch()

    def _confirm_delete(self) -> None:
        to_delete = [fp for fp, v in self.marked_for_deletion.items() if v]
        if not to_delete:
            messagebox.showinfo(
                "Nothing to delete", "No files are marked for deletion."
            )
            return

        names = "\n".join(os.path.basename(f) for f in to_delete)
        if not messagebox.askyesno(
            "Confirm deletion",
            f"Delete {len(to_delete)} file(s)?\n\n{names}",
        ):
            return

        deleted = 0
        errors: list[str] = []
        for fp in to_delete:
            try:
                os.remove(fp)
                deleted += 1
            except Exception as e:
                errors.append(f"{os.path.basename(fp)}: {e}")

        # Remove deleted entries
        for fp in to_delete:
            self.marked_for_deletion.pop(fp, None)

        # Refresh file list
        self.log_files = sorted(
            [f for f in os.listdir(self.gesture_dir) if f.endswith(".csv")]
        )
        self.total_batches = max(
            1, (len(self.log_files) + BATCH_SIZE - 1) // BATCH_SIZE
        )
        if self.current_batch >= self.total_batches:
            self.current_batch = self.total_batches - 1

        msg = f"Deleted {deleted} file(s)."
        if errors:
            msg += "\n\nErrors:\n" + "\n".join(errors)
        messagebox.showinfo("Done", msg)

        if self.log_files:
            self._show_batch()
        else:
            messagebox.showinfo("Empty", "No more files for this gesture.")
            self._build_gesture_selector()

    # ---- Utilities --------------------------------------------------------

    def _clear(self) -> None:
        """Destroy all child widgets."""
        for child in self.winfo_children():
            child.destroy()
        # Close any open matplotlib figures
        plt.close("all")


def main() -> None:
    app = ReviewApp()
    app.mainloop()


if __name__ == "__main__":
    main()
