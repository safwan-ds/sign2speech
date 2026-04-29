"""Non-blocking real-time stream worker for prediction pipeline."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from queue import Queue
from typing import TYPE_CHECKING

from config import MIN_CONSECUTIVE_REST, MIN_GESTURES_FOR_LLM
from core.inference.gesture_translations import (
    load_gesture_translations,
    translate_gesture,
    translate_gestures,
)
from gui.services.serial_service import SerialService, SerialSettings
from gui.utils.formatting import now_hms
from gui.utils.smoothing import PredictionSmoother, SentenceAssembler

if TYPE_CHECKING:
    from gui.services.model_service import ModelService


@dataclass(slots=True)
class StreamConfig:
    """Runtime settings for prediction stream."""

    serial_settings: SerialSettings
    confidence_threshold: float
    smoothing_window: int
    language: str = "tr"


@dataclass(slots=True)
class GestureLockState:
    """Track the currently emitted gesture and suppress duplicate static holds."""

    locked_token: str | None = None
    locked_at: float | None = None
    last_transition_at: float | None = None

    def observe(
        self, token: str, confidence: float, confidence_threshold: float
    ) -> tuple[bool, bool, str]:
        """Return (should_emit, is_rest, normalized_token)."""
        normalized = token.strip().upper()
        is_rest = normalized == "REST"
        confident = confidence >= confidence_threshold
        now = time.monotonic()

        if is_rest:
            self.locked_token = None
            self.locked_at = None
            self.last_transition_at = now
            return False, True, normalized

        if not confident:
            return False, False, normalized

        if self.locked_token is None or self.locked_token != normalized:
            self.locked_token = normalized
            self.locked_at = now
            self.last_transition_at = now
            return True, False, normalized

        return False, False, normalized


class StreamWorker(threading.Thread):
    """Thread that reads serial data, predicts, smooths, and emits queue events."""

    def __init__(
        self,
        model_service: "ModelService",
        serial_service: SerialService,
        event_queue: Queue[dict],
        logger: logging.Logger,
        config: StreamConfig,
    ) -> None:
        super().__init__(daemon=True)
        self._model_service = model_service
        self._serial_service = serial_service
        self._event_queue = event_queue
        self._logger = logger
        self._config = config
        self._stop_event = threading.Event()
        self._smoother = PredictionSmoother(config.smoothing_window)
        self._sentence = SentenceAssembler()
        self._translations = load_gesture_translations()
        self._lock_state = GestureLockState()

    def stop(self) -> None:
        """Request worker shutdown."""
        self._stop_event.set()

    def run(self) -> None:
        """Worker lifecycle entrypoint."""
        try:
            predictor = self._model_service.require_predictor()
            opened = self._serial_service.connect(self._config.serial_settings)
            if opened:
                self._logger.info(
                    "Connected to %s @ %s",
                    self._config.serial_settings.port,
                    self._config.serial_settings.baud_rate,
                )
            else:
                self._logger.info(
                    "Reusing open serial connection on %s @ %s",
                    self._config.serial_settings.port,
                    self._config.serial_settings.baud_rate,
                )
            self._event_queue.put({"type": "connected", "value": True})

            collected_gestures: list[tuple[str, float]] = []
            consecutive_rest_frames = 0
            stream_input_emitted = False
            stream_started_emitted = False
            last_llm_text: str | None = None

            while not self._stop_event.is_set():
                # read_sensor_row() blocks on the SerialService queue until a
                # parsed frame is available or the timeout expires, so no extra
                # sleep is needed here.
                sensor = self._serial_service.read_sensor_row(timeout=0.2)
                if sensor is None:
                    continue

                if not stream_input_emitted:
                    self._event_queue.put(
                        {
                            "type": "stream_input_detected",
                            "timestamp": now_hms(),
                        }
                    )
                    stream_input_emitted = True

                predictor.add_sensor_dict(sensor)
                if not predictor.can_predict():
                    continue

                gesture, confidence, _, _ = predictor.predict()
                if gesture is None or confidence is None:
                    continue

                if not stream_started_emitted:
                    self._event_queue.put(
                        {
                            "type": "stream_started",
                            "timestamp": now_hms(),
                        }
                    )
                    stream_started_emitted = True

                smooth_token = self._smoother.update(str(gesture))
                is_confident = confidence >= self._config.confidence_threshold
                display_token = (
                    smooth_token
                    if is_confident
                    else ("Belirsiz" if self._config.language == "tr" else "Uncertain")
                )

                should_emit, is_rest, _ = self._lock_state.observe(
                    smooth_token,
                    confidence,
                    self._config.confidence_threshold,
                )

                self._event_queue.put(
                    {
                        "type": "prediction",
                        "timestamp": now_hms(),
                        "gesture": display_token,
                        "raw_gesture": smooth_token,
                        "confidence": confidence,
                    }
                )

                if should_emit and not is_rest:
                    translated = translate_gesture(
                        smooth_token,
                        self._translations,
                        target_language=self._config.language,
                    )
                    if self._sentence.try_append(smooth_token):
                        self._event_queue.put(
                            {
                                "type": "sentence",
                                "token": translated,
                                "sentence": self._sentence.text(),
                            }
                        )

                if is_rest:
                    consecutive_rest_frames += 1

                    if (
                        consecutive_rest_frames == MIN_CONSECUTIVE_REST
                        and len(collected_gestures) >= MIN_GESTURES_FOR_LLM
                    ):
                        gesture_names = [name for name, _ in collected_gestures]
                        translated_gestures = translate_gestures(
                            gesture_names,
                            self._translations,
                            target_language=self._config.language,
                        )
                        gesture_text = " ".join(translated_gestures)

                        # Debounce: skip refinement if the gesture sequence
                        # hasn't changed since the last LLM call.
                        if gesture_text != last_llm_text:
                            self._event_queue.put(
                                {
                                    "type": "llm_request",
                                    "text": gesture_text,
                                }
                            )
                            last_llm_text = gesture_text
                            self._logger.info("LLM refinement request queued")
                        else:
                            self._logger.debug(
                                "LLM refinement skipped (duplicate sequence)"
                            )

                        collected_gestures.clear()
                        consecutive_rest_frames = 0
                elif should_emit and is_confident:
                    consecutive_rest_frames = 0

                    collected_gestures.append((smooth_token, confidence))
        except Exception as exc:  # pragma: no cover - guarded by UI notifications
            self._logger.exception("Stream worker failed")
            self._event_queue.put(
                {
                    "type": "error",
                    "message": f"Streaming error: {exc}",
                }
            )
        finally:
            self._event_queue.put({"type": "connected", "value": False})
            self._event_queue.put({"type": "stopped"})
            self._logger.info("Streaming stopped")
