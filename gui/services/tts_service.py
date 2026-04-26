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

try:
    import pyttsx3
except Exception:  # pragma: no cover - dependency may be missing at runtime
    pyttsx3 = None


@dataclass(slots=True)
class TTSRequest:
    """Request payload for TTS queue."""

    text: str
    language: str
    backend: str = "local"


class TTSService:
    """Queue-based non-blocking service for text-to-speech synthesis."""

    _VOICE_MAP: dict[str, tuple[str, ...]] = {
        "tr": ("tr-TR-AhmetNeural", "tr-TR-EmelNeural"),
        "en": ("en-US-AriaNeural", "en-US-GuyNeural"),
    }

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._logger = logger or logging.getLogger(__name__)
        self._local_requests: queue.Queue[TTSRequest | None] = queue.Queue(maxsize=6)
        self._edge_requests: queue.Queue[TTSRequest | None] = queue.Queue(maxsize=2)
        self._stop_event = threading.Event()
        self._local_worker = threading.Thread(target=self._local_tts_loop, daemon=True)
        self._edge_worker = threading.Thread(target=self._edge_tts_loop, daemon=True)
        self._warned_language_fallback: set[str] = set()
        self._local_worker.start()
        self._edge_worker.start()

    @staticmethod
    def _language_key(language: str) -> str:
        normalized = str(language).strip().lower()
        if normalized.startswith("tr"):
            return "tr"
        return "en"

    def _voice_candidates(self, language: str) -> tuple[str, ...]:
        normalized = str(language).strip().lower()
        requested = self._language_key(language)
        if (
            requested == "en"
            and not normalized.startswith("en")
            and not normalized.startswith("tr")
            and language not in self._warned_language_fallback
        ):
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

    def _create_local_engine(self):
        if pyttsx3 is None:
            raise RuntimeError("pyttsx3 is not available")
        engine = pyttsx3.init()
        # Slightly faster speaking rate for token-level responsiveness.
        engine.setProperty("rate", 185)
        return engine

    def _choose_local_voice_id(self, engine, language: str) -> str | None:
        target = self._language_key(language)

        voices_obj = engine.getProperty("voices")
        voices = list(voices_obj) if isinstance(voices_obj, (list, tuple)) else []
        selected_id = None
        for voice in voices:
            name = str(getattr(voice, "name", "")).lower()
            voice_id = str(getattr(voice, "id", "")).lower()
            langs = getattr(voice, "languages", [])
            langs_text = " ".join(str(item).lower() for item in langs)
            blob = f"{name} {voice_id} {langs_text}"

            if target == "tr" and (
                "turkish" in blob or "turkce" in blob or "tr-" in blob or " tr" in blob
            ):
                selected_id = getattr(voice, "id", None)
                break

            if target == "en" and ("english" in blob or "en-" in blob or " en" in blob):
                selected_id = getattr(voice, "id", None)
                break

        if selected_id:
            return str(selected_id)
        return None

    def _set_local_voice(
        self,
        engine,
        language: str,
        voice_cache: dict[str, str | None],
    ) -> bool:
        target = self._language_key(language)
        if target not in voice_cache:
            voice_cache[target] = self._choose_local_voice_id(engine, language)

        selected_id = voice_cache[target]
        if selected_id:
            engine.setProperty("voice", selected_id)
            return True
        return False

    def _local_tts_loop(self) -> None:
        """Dedicated local TTS loop.

        pyttsx3/SAPI is initialized exactly once in this thread and reused.
        """
        if pyttsx3 is None:
            self._logger.error("Local TTS unavailable: pyttsx3 is not installed")
            return

        voice_cache: dict[str, str | None] = {}
        engine = self._create_local_engine()

        try:
            while not self._stop_event.is_set():
                try:
                    request = self._local_requests.get(timeout=0.5)
                except queue.Empty:
                    continue

                if request is None:
                    break

                try:
                    voice_selected = self._set_local_voice(
                        engine,
                        request.language,
                        voice_cache,
                    )
                    if (
                        self._language_key(request.language) == "tr"
                        and not voice_selected
                    ):
                        self._logger.error("No local Turkish voice found for local TTS")
                        continue

                    engine.say(request.text)
                    engine.runAndWait()
                except Exception as exc:
                    self._logger.error("Local TTS failed: %s", exc)
                    try:
                        engine.stop()
                    except Exception:
                        pass
                    engine = self._create_local_engine()
        except Exception as exc:
            self._logger.error("Local TTS worker crashed: %s", exc)
        finally:
            try:
                engine.stop()
            except Exception:
                pass

    def _handle_edge_request(self, request: TTSRequest) -> None:

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

    def _edge_tts_loop(self) -> None:
        """Dedicated Edge TTS loop for refined/LLM output when requested."""
        try:
            while not self._stop_event.is_set():
                try:
                    request = self._edge_requests.get(timeout=0.5)
                    if request is None:
                        break
                    self._handle_edge_request(request)

                except queue.Empty:
                    continue
                except Exception as e:
                    self._logger.error(f"Error in Edge TTS worker: {e}")

        except Exception as e:
            self._logger.error(f"Edge TTS worker crashed: {e}")

    def speak(self, text: str, language: str = "tr", backend: str = "local") -> None:
        """Queue text for speech synthesis (non-blocking)."""
        text = str(text).strip()
        if not text:
            return

        backend = str(backend).strip().lower()
        if backend not in {"edge", "local"}:
            backend = "local"

        target_queue = (
            self._edge_requests if backend == "edge" else self._local_requests
        )

        # Non-blocking put with bounded queue; if full, drop oldest and keep latest.
        try:
            target_queue.put_nowait(
                TTSRequest(text=text, language=language, backend=backend)
            )
        except queue.Full:
            # Queue is full, drop old request and queue new one.
            try:
                target_queue.get_nowait()
                target_queue.put_nowait(
                    TTSRequest(text=text, language=language, backend=backend)
                )
            except queue.Empty:
                pass

    def stop(self) -> None:
        """Stop the TTS service."""
        self._stop_event.set()
        try:
            self._local_requests.put_nowait(None)
        except queue.Full:
            try:
                self._local_requests.get_nowait()
                self._local_requests.put_nowait(None)
            except queue.Empty:
                pass

        try:
            self._edge_requests.put_nowait(None)
        except queue.Full:
            try:
                self._edge_requests.get_nowait()
                self._edge_requests.put_nowait(None)
            except queue.Empty:
                pass

        self._local_worker.join(timeout=2.0)
        self._edge_worker.join(timeout=2.0)
