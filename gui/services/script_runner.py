"""Run long scripts in background and stream output to UI queue."""

from __future__ import annotations

import logging
import subprocess
import sys
import threading
from pathlib import Path


class ScriptRunner:
    """Execute project scripts without blocking Tk mainloop."""

    def __init__(self, project_root: Path, logger: logging.Logger) -> None:
        self._project_root = project_root
        self._logger = logger
        self._active_lock = threading.Lock()

    @staticmethod
    def _level_for_line(line: str) -> int:
        text = line.upper()
        if "ERROR" in text or "EXCEPTION" in text or "TRACEBACK" in text:
            return logging.ERROR
        if "WARNING" in text or "WARN" in text:
            return logging.WARNING
        return logging.INFO

    def run_script(self, script_rel_path: str, args: list[str] | None = None) -> bool:
        """Start script in background and return False when already busy."""
        if not self._active_lock.acquire(blocking=False):
            return False

        script_path = self._project_root / script_rel_path
        cli_args = args or []

        def _runner() -> None:
            try:
                process = subprocess.Popen(
                    [sys.executable, str(script_path), *cli_args],
                    cwd=str(self._project_root),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                if cli_args:
                    self._logger.info(
                        "[script] Running %s %s",
                        script_rel_path,
                        " ".join(cli_args),
                    )
                else:
                    self._logger.info("[script] Running %s", script_rel_path)
                assert process.stdout is not None
                for line in process.stdout:
                    stripped = line.rstrip()
                    if not stripped:
                        continue
                    level = self._level_for_line(stripped)
                    self._logger.log(level, "[script:%s] %s", script_rel_path, stripped)

                code = process.wait()
                if code == 0:
                    self._logger.info(
                        "[script] %s exited with code %s", script_rel_path, code
                    )
                else:
                    self._logger.error(
                        "[script] %s exited with code %s", script_rel_path, code
                    )
            except Exception as exc:
                self._logger.exception("Script failed for %s: %s", script_rel_path, exc)
            finally:
                self._active_lock.release()

        threading.Thread(target=_runner, daemon=True).start()
        return True
