"""Non-blocking text-to-speech service using Edge neural voices."""

from __future__ import annotations

import asyncio
import logging
import os
import queue
import subprocess
import tempfile
import threading
from pathlib import Path
from dataclasses import dataclass

import edge_tts


@dataclass(slots=True)
class TTSRequest:
    """Request payload for TTS queue."""

    text: str
    language: str


class TTSService:
    """Queue-based non-blocking service for text-to-speech synthesis."""

    _VOICE_MAP: dict[str, tuple[str, ...]] = {
        "tr": ("tr-TR-AhmetNeural", "tr-TR-EmelNeural"),
        "en": ("en-US-AriaNeural", "en-US-GuyNeural"),
    }

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._logger = logger or logging.getLogger(__name__)
        self._requests: queue.Queue[TTSRequest | None] = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._warned_language_fallback: set[str] = set()
        self._worker.start()

    def _voice_candidates(self, language: str) -> tuple[str, ...]:
        requested = language if language in self._VOICE_MAP else "en"
        if requested != language and language not in self._warned_language_fallback:
            self._warned_language_fallback.add(language)
            self._logger.warning(
                "Unsupported TTS language '%s'. Falling back to English.", language
            )
        return self._VOICE_MAP[requested]

    @staticmethod
    async def _synthesize_to_file(text: str, voice: str, audio_path: str) -> None:
        communicate = edge_tts.Communicate(text=text, voice=voice)
        await communicate.save(audio_path)

    @staticmethod
    def _play_audio_file(audio_path: str) -> None:
        audio_uri = Path(audio_path).resolve().as_uri()
        script = (
            "Add-Type -AssemblyName PresentationCore;"
            "$player = New-Object System.Windows.Media.MediaPlayer;"
            f"$player.Open([System.Uri]'{audio_uri}');"
            "$player.Play();"
            "while (-not $player.NaturalDuration.HasTimeSpan) { Start-Sleep -Milliseconds 50 };"
            "Start-Sleep -Milliseconds ([int]$player.NaturalDuration.TimeSpan.TotalMilliseconds + 120);"
            "$player.Stop();"
            "$player.Close();"
            "$player = $null"
        )
        subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                script,
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    def _speak_with_voice(self, text: str, voice: str) -> None:
        fd, audio_path = tempfile.mkstemp(prefix="sign_glove_tts_", suffix=".mp3")
        os.close(fd)
        try:
            asyncio.run(
                self._synthesize_to_file(text=text, voice=voice, audio_path=audio_path)
            )
            self._play_audio_file(audio_path)
        finally:
            try:
                os.remove(audio_path)
            except OSError:
                pass

    def _handle_request(self, request: TTSRequest) -> None:
        candidates = self._voice_candidates(request.language)
        last_error: Exception | None = None

        for voice in candidates:
            try:
                self._logger.debug("Trying TTS voice: %s", voice)
                self._speak_with_voice(request.text, voice)
                return
            except Exception as exc:
                last_error = exc
                self._logger.warning("TTS voice '%s' failed: %s", voice, exc)

        self._logger.error(
            "All candidate TTS voices failed for language '%s': %s",
            request.language,
            last_error,
        )

    def _run(self) -> None:
        """Worker thread main loop."""
        try:
            while not self._stop_event.is_set():
                try:
                    # Wait for request with timeout
                    request = self._requests.get(timeout=0.5)
                    if request is None:
                        break
                    self._handle_request(request)

                except queue.Empty:
                    continue
                except Exception as e:
                    self._logger.error(f"Error in TTS worker: {e}")

        except Exception as e:
            self._logger.error(f"TTS worker crashed: {e}")

    def speak(self, text: str, language: str = "tr") -> None:
        """Queue text for speech synthesis (non-blocking)."""
        text = str(text).strip()
        if not text:
            return

        # Non-blocking put with maxsize=1 drops old requests if new one arrives
        try:
            self._requests.put_nowait(TTSRequest(text=text, language=language))
        except queue.Full:
            # Queue is full, drop old request and queue new one
            try:
                self._requests.get_nowait()
                self._requests.put_nowait(TTSRequest(text=text, language=language))
            except queue.Empty:
                pass

    def stop(self) -> None:
        """Stop the TTS service."""
        self._stop_event.set()
        try:
            self._requests.put_nowait(None)
        except queue.Full:
            pass
        self._worker.join(timeout=2.0)
