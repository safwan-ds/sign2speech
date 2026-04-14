"""Non-blocking real-time stream worker for prediction pipeline."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from queue import Queue

from config import MIN_CONSECUTIVE_REST, MIN_GESTURES_FOR_LLM
from core.inference.gesture_translations import (
    load_gesture_translations,
    translate_gesture,
    translate_gestures,
)
from gui.services.model_service import ModelService
from gui.services.serial_service import SerialService, SerialSettings
from gui.utils.formatting import now_hms
from gui.utils.smoothing import PredictionSmoother, SentenceAssembler


@dataclass(slots=True)
class StreamConfig:
    """Runtime settings for prediction stream."""

    serial_settings: SerialSettings
    confidence_threshold: float
    smoothing_window: int
    language: str = "tr"


class StreamWorker(threading.Thread):
    """Thread that reads serial data, predicts, smooths, and emits queue events."""

    def __init__(
        self,
        model_service: ModelService,
        event_queue: Queue[dict],
        logger: logging.Logger,
        config: StreamConfig,
    ) -> None:
        super().__init__(daemon=True)
        self._model_service = model_service
        self._event_queue = event_queue
        self._logger = logger
        self._config = config
        self._stop_event = threading.Event()
        self._serial_service = SerialService()
        self._smoother = PredictionSmoother(config.smoothing_window)
        self._sentence = SentenceAssembler()
        self._translations = load_gesture_translations()

    def stop(self) -> None:
        """Request worker shutdown."""
        self._stop_event.set()

    def run(self) -> None:
        """Worker lifecycle entrypoint."""
        try:
            predictor = self._model_service.require_predictor()
            self._serial_service.connect(self._config.serial_settings)
            self._event_queue.put({"type": "connected", "value": True})
            self._logger.info(
                "Connected to %s @ %s",
                self._config.serial_settings.port,
                self._config.serial_settings.baud_rate,
            )

            last_gesture = None
            last_added_gesture = None
            collected_gestures: list[tuple[str, float]] = []
            consecutive_rest_frames = 0
            stream_input_emitted = False
            stream_started_emitted = False

            while not self._stop_event.is_set():
                sensor = self._serial_service.read_sensor_row()
                if sensor is None:
                    time.sleep(0.01)
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
                display_token = smooth_token
                if confidence < self._config.confidence_threshold:
                    display_token = (
                        "Belirsiz" if self._config.language == "tr" else "Uncertain"
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

                stable_token = smooth_token.upper()
                if (
                    display_token not in {"Belirsiz", "Uncertain"}
                    and stable_token != "REST"
                    and self._sentence.try_append(smooth_token)
                ):
                    translated = translate_gesture(
                        smooth_token,
                        self._translations,
                        target_language=self._config.language,
                    )
                    self._event_queue.put(
                        {
                            "type": "sentence",
                            "token": translated,
                            "sentence": self._sentence.text(),
                        }
                    )

                is_rest = stable_token == "REST"
                is_new = smooth_token != last_gesture

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

                        self._event_queue.put(
                            {
                                "type": "llm_request",
                                "text": gesture_text,
                            }
                        )
                        self._logger.info("LLM refinement request queued")

                        collected_gestures.clear()
                        last_added_gesture = None
                        consecutive_rest_frames = 0
                elif is_new and display_token not in {"Belirsiz", "Uncertain"}:
                    consecutive_rest_frames = 0

                    if smooth_token != last_added_gesture:
                        collected_gestures.append((smooth_token, confidence))
                        last_added_gesture = smooth_token

                last_gesture = smooth_token
        except Exception as exc:  # pragma: no cover - guarded by UI notifications
            self._logger.exception("Stream worker failed")
            self._event_queue.put(
                {
                    "type": "error",
                    "message": f"Streaming error: {exc}",
                }
            )
        finally:
            self._serial_service.disconnect()
            self._event_queue.put({"type": "connected", "value": False})
            self._event_queue.put({"type": "stopped"})
            self._logger.info("Streaming stopped")
