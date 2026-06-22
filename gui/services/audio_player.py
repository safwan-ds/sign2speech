"""Low-level audio playback for TTS using Windows MCI.

Provides the platform-specific MCI audio player and the main playback
function used by TTSService. This module has no dependency on TTSService
or any other module in gui.services.
"""

from __future__ import annotations

import ctypes
import logging
import platform
import sys
import threading
import time
import uuid
from collections.abc import Callable


def _winmm_mci() -> Callable | None:
    """Return the Windows MCI ``mciSendStringW`` callable, or None elsewhere."""
    if platform.system() != "Windows":
        return None
    try:
        winmm = ctypes.WinDLL("winmm")
    except (OSError, AttributeError):  # pragma: no cover - non-Windows
        return None
    fn = winmm.mciSendStringW
    fn.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_wchar_p,
        ctypes.c_uint,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_uint
    return fn


_MCI_SEND: Callable | None = _winmm_mci()


def play_audio_file(
    audio_path: str,
    logger: logging.Logger,
    stop_event: threading.Event,
    mci_lock: threading.Lock,
    set_status: Callable[[str, str, str], None],
) -> None:
    """Play an MP3 file using Windows MCI (native, no subprocess).

    Parameters
    ----------
    audio_path : str
        Path to the audio file to play.
    logger : logging.Logger
        Logger instance for debug/error output.
    stop_event : threading.Event
        Event that signals playback should be interrupted.
    mci_lock : threading.Lock
        Lock serializing MCI commands (winmm is not thread-safe).
    set_status : Callable[[str, str, str], None]
        Callback to report status changes (state, backend, message).
    """
    mci_send = _MCI_SEND
    if mci_send is None:
        # Non-Windows fallback: best-effort, will not block.
        logger.error(
            "Audio playback unsupported on this platform: %s", sys.platform
        )
        set_status("error", "edge", "Audio playback unsupported")
        return

    # Each utterance gets a unique alias so concurrent calls don't clash.
    alias = f"sgglove_{uuid.uuid4().hex[:12]}"
    # MCI does not handle paths with spaces unless quoted; alias must not be quoted.
    open_cmd = f'open "{audio_path}" type mpegvideo alias {alias}'

    def _send(cmd: str, return_buf: bool = False) -> tuple[int, str]:
        buf = ctypes.create_unicode_buffer(128) if return_buf else None
        err = mci_send(cmd, buf, 128 if buf else 0, None)
        return err, (buf.value if buf else "")

    with mci_lock:
        err, _ = _send(open_cmd)
        if err != 0:
            logger.error("MCI open failed (%d) for %s", err, audio_path)
            set_status("error", "edge", f"Playback open failed ({err})")
            return

        try:
            # Query duration so we can wait the right amount.
            err, length_str = _send(f"status {alias} length", return_buf=True)
            length_ms = 0
            if err == 0 and length_str:
                try:
                    length_ms = int(length_str)
                except ValueError:
                    length_ms = 0

            err, _ = _send(f"play {alias}")
            if err != 0:
                logger.error("MCI play failed (%d)", err)
                set_status("error", "edge", f"Playback failed ({err})")
                return

            logger.debug("MCI playing: %s (%d ms)", audio_path, length_ms)

            # Sleep slightly longer than the clip to ensure clean tail.
            # If length is unknown, 5s is a safe fallback for most sentences.
            wait_s = (length_ms / 1000.0) + 0.15 if length_ms > 0 else 5.0

            # Sleep in small increments to remain responsive to stop events.
            end_time = time.time() + wait_s
            while time.time() < end_time and not stop_event.is_set():
                time.sleep(0.1)
        finally:
            _send(f"close {alias}")
