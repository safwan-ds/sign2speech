"""Non-blocking text-to-speech service using Edge neural voices.

Audio playback uses Windows MCI directly via ctypes to avoid spawning a fresh
PowerShell process per utterance (which adds 300-500ms of latency on every
spoken word). The Edge worker keeps a persistent asyncio event loop so that
edge-tts can reuse its HTTPS session across requests.
"""

from __future__ import annotations

import asyncio
import ctypes
import logging
import os
import platform
import queue
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass

import diskcache
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


def _winmm_mci():
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


_MCI_SEND = _winmm_mci()


class TTSService:
    """Queue-based non-blocking service for text-to-speech synthesis."""

    _VOICE_MAP: dict[str, tuple[str, ...]] = {
        "tr": ("tr-TR-AhmetNeural", "tr-TR-EmelNeural"),
        "en": ("en-US-AriaNeural", "en-US-GuyNeural"),
    }

    def __init__(
        self,
        logger: logging.Logger | None = None,
        event_queue: queue.Queue[dict] | None = None,
        cache_dir: str | None = None,
    ) -> None:
        self._logger = logger or logging.getLogger(__name__)
        self._event_queue = event_queue

        # Initialize diskcache for Edge TTS audio
        if cache_dir is None:
            from config.config import BASE_DIR

            cache_dir = os.path.join(BASE_DIR, "data", "cache", "tts")
        os.makedirs(cache_dir, exist_ok=True)
        self._cache = diskcache.Cache(
            cache_dir, size_limit=100 * 1024 * 1024
        )  # 100MB limit

        self._local_requests: queue.Queue[TTSRequest | None] = queue.Queue(maxsize=6)
        self._edge_requests: queue.Queue[TTSRequest | None] = queue.Queue(maxsize=2)
        self._stop_event = threading.Event()
        self._local_worker = threading.Thread(target=self._local_tts_loop, daemon=True)
        self._edge_worker = threading.Thread(target=self._edge_tts_loop, daemon=True)
        self._warned_language_fallback: set[str] = set()
        self._warned_local_edge_fallback: set[str] = set()
        self._mci_lock = threading.Lock()
        self._status_lock = threading.Lock()
        self._last_status_emitted: tuple[str, str, str] | None = None
        self._local_worker.start()
        self._edge_worker.start()
        self._set_status("waiting", "local", "")

    def _emit_status(self, state: str, backend: str, message: str = "") -> None:
        if self._event_queue is None:
            return
        payload = {
            "type": "tts_status",
            "state": state,
            "backend": backend,
            "message": message,
        }
        try:
            self._event_queue.put_nowait(payload)
        except queue.Full:
            pass

    def _set_status(self, state: str, backend: str, message: str = "") -> None:
        key = (state, backend, message)
        with self._status_lock:
            if self._last_status_emitted == key:
                return
            self._last_status_emitted = key
        self._emit_status(state=state, backend=backend, message=message)

    def _set_post_request_status(
        self, target_queue: queue.Queue[TTSRequest | None], backend: str
    ) -> None:
        if target_queue.empty():
            self._set_status("waiting", backend, "")
        else:
            self._set_status("working", backend, "")

    def _fallback_local_request_to_edge(self, request: TTSRequest, reason: str) -> None:
        """Route a failed local request to Edge TTS as a best-effort fallback."""
        warn_key = f"{self._language_key(request.language)}:{reason}"
        if warn_key not in self._warned_local_edge_fallback:
            self._warned_local_edge_fallback.add(warn_key)
            self._logger.warning(
                "Local TTS unavailable (%s). Falling back to Edge backend.", reason
            )
        self._set_status(
            "working",
            "edge",
            f"Local unavailable ({reason}), using Edge",
        )
        self.speak(request.text, request.language, backend="edge")

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

    def _play_audio_file(self, audio_path: str) -> None:
        """Play an MP3 file using Windows MCI (native, no subprocess)."""
        mci_send = _MCI_SEND
        if mci_send is None:
            # Non-Windows fallback: best-effort, will not block.
            self._logger.error(
                "Audio playback unsupported on this platform: %s", sys.platform
            )
            self._set_status("error", "edge", "Audio playback unsupported")
            return

        # Each utterance gets a unique alias so concurrent calls don't clash.
        alias = f"sgglove_{uuid.uuid4().hex[:12]}"
        # MCI does not handle paths with spaces unless quoted; alias must not be quoted.
        open_cmd = f'open "{audio_path}" type mpegvideo alias {alias}'

        def _send(cmd: str, return_buf: bool = False) -> tuple[int, str]:
            buf = ctypes.create_unicode_buffer(128) if return_buf else None
            err = mci_send(cmd, buf, 128 if buf else 0, None)
            return err, (buf.value if buf else "")

        with self._mci_lock:
            err, _ = _send(open_cmd)
            if err != 0:
                self._logger.error("MCI open failed (%d) for %s", err, audio_path)
                self._set_status("error", "edge", f"Playback open failed ({err})")
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
                    self._logger.error("MCI play failed (%d)", err)
                    self._set_status("error", "edge", f"Playback failed ({err})")
                    return

                self._logger.debug("MCI playing: %s (%d ms)", audio_path, length_ms)

                # Sleep slightly longer than the clip to ensure clean tail.
                # If length is unknown, 5s is a safe fallback for most sentences.
                wait_s = (length_ms / 1000.0) + 0.15 if length_ms > 0 else 5.0

                # Sleep in small increments to remain responsive to stop events.
                end_time = time.time() + wait_s
                while time.time() < end_time and not self._stop_event.is_set():
                    time.sleep(0.1)
            finally:
                _send(f"close {alias}")

    def _speak_with_voice(
        self,
        text: str,
        voice: str,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        # Check cache first
        cache_key = f"{voice}_{text}"
        cached_audio = self._cache.get(cache_key)

        if cached_audio:
            self._logger.debug("TTS Cache hit: %s", text)
            fd, audio_path = tempfile.mkstemp(prefix="sign_glove_tts_", suffix=".mp3")
            os.close(fd)
            try:
                with open(audio_path, "wb") as f:
                    f.write(cached_audio)
                self._play_audio_file(audio_path)
            finally:
                try:
                    os.remove(audio_path)
                except OSError:
                    pass
            return

        fd, audio_path = tempfile.mkstemp(prefix="sign_glove_tts_", suffix=".mp3")
        os.close(fd)
        try:
            # Reuse the worker thread's persistent event loop instead of
            # spinning up a new one (and a new HTTPS session) per utterance.
            future = asyncio.run_coroutine_threadsafe(
                self._synthesize_to_file(text=text, voice=voice, audio_path=audio_path),
                loop,
            )
            future.result()

            # Store in cache
            try:
                with open(audio_path, "rb") as f:
                    self._cache.set(cache_key, f.read())
            except Exception as e:
                self._logger.warning("Failed to cache TTS audio: %s", e)

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
        A silent warmup utterance is run at startup to pay SAPI's first-call
        latency before any user-visible request arrives.
        """
        if pyttsx3 is None:
            self._logger.error("Local TTS unavailable: pyttsx3 is not installed")
            self._set_status("error", "local", "pyttsx3 is not installed")
            return

        voice_cache: dict[str, str | None] = {}
        try:
            engine = self._create_local_engine()
        except Exception as exc:
            self._logger.error("Local TTS engine init failed: %s", exc)
            self._set_status("error", "local", f"Local engine init failed: {exc}")
            return

        # Warmup: a single empty/space utterance forces SAPI to initialize.
        # This trims ~300-700ms off the first real word.
        try:
            engine.say(" ")
            engine.runAndWait()
        except Exception as exc:
            self._logger.warning("Local TTS warmup failed: %s", exc)

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
                        self._fallback_local_request_to_edge(
                            request,
                            reason="no Turkish local voice",
                        )
                        continue

                    engine.say(request.text)
                    engine.runAndWait()
                    self._set_post_request_status(self._local_requests, "local")
                except Exception as exc:
                    self._logger.error("Local TTS failed: %s", exc)
                    self._set_status("error", "local", f"Local TTS failed: {exc}")
                    self._fallback_local_request_to_edge(
                        request,
                        reason="local backend runtime error",
                    )
                    try:
                        engine.stop()
                    except Exception:
                        pass
                    try:
                        engine = self._create_local_engine()
                    except Exception as exc2:
                        self._logger.error("Local TTS engine recovery failed: %s", exc2)
                        self._set_status(
                            "error",
                            "local",
                            f"Local recovery failed: {exc2}",
                        )
                        return
        except Exception as exc:
            self._logger.error("Local TTS worker crashed: %s", exc)
            self._set_status("error", "local", f"Local worker crashed: {exc}")
        finally:
            try:
                engine.stop()
            except Exception:
                pass

    def _handle_edge_request(
        self,
        request: TTSRequest,
        loop: asyncio.AbstractEventLoop,
    ) -> bool:
        candidates = self._voice_candidates(request.language)
        last_error: Exception | None = None

        for voice in candidates:
            try:
                self._logger.debug("Trying TTS voice: %s", voice)
                self._speak_with_voice(request.text, voice, loop)
                return True
            except Exception as exc:
                last_error = exc
                self._logger.warning("TTS voice '%s' failed: %s", voice, exc)

        self._logger.error(
            "All candidate TTS voices failed for language '%s': %s",
            request.language,
            last_error,
        )
        self._set_status("error", "edge", f"All Edge voices failed: {last_error}")
        return False

    def _edge_tts_loop(self) -> None:
        """Dedicated Edge TTS loop with a persistent asyncio event loop.

        Reusing one loop across requests avoids paying loop-setup cost (and the
        edge-tts/aiohttp connection setup cost) on every spoken sentence.
        """
        loop = asyncio.new_event_loop()
        loop_thread = threading.Thread(
            target=self._run_event_loop, args=(loop,), daemon=True
        )
        loop_thread.start()

        try:
            while not self._stop_event.is_set():
                try:
                    request = self._edge_requests.get(timeout=0.5)
                except queue.Empty:
                    continue

                if request is None:
                    break

                try:
                    success = self._handle_edge_request(request, loop)
                    if success:
                        self._set_post_request_status(self._edge_requests, "edge")
                except Exception as exc:
                    self._logger.error("Edge TTS request failed: %s", exc)
                    self._set_status("error", "edge", f"Edge request failed: {exc}")
        except Exception as exc:
            self._logger.error("Edge TTS worker crashed: %s", exc)
            self._set_status("error", "edge", f"Edge worker crashed: {exc}")
        finally:
            loop.call_soon_threadsafe(loop.stop)
            loop_thread.join(timeout=1.0)
            try:
                loop.close()
            except Exception:
                pass

    @staticmethod
    def _run_event_loop(loop: asyncio.AbstractEventLoop) -> None:
        asyncio.set_event_loop(loop)
        loop.run_forever()

    def speak(self, text: str, language: str = "tr", backend: str = "local") -> None:
        """Queue text for speech synthesis (non-blocking)."""
        text = str(text).strip()
        if not text:
            return

        backend = str(backend).strip().lower()
        if backend not in {"edge", "local"}:
            backend = "local"

        if backend == "local" and not self._local_worker.is_alive():
            warn_key = "local-worker-unavailable"
            if warn_key not in self._warned_local_edge_fallback:
                self._warned_local_edge_fallback.add(warn_key)
                self._logger.warning(
                    "Local TTS worker unavailable. Falling back to Edge backend."
                )
            backend = "edge"

        self._set_status("working", backend, "")

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
            self._cache.close()
        except Exception:
            pass
        self._set_status("waiting", "local", "")
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
