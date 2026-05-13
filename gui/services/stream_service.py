"""Non-blocking real-time stream worker for prediction pipeline."""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from typing import TYPE_CHECKING

import numpy as np

from config import (
    ENABLE_SEQUENCE_DECODER,
    MIN_CONSECUTIVE_REST,
    MIN_GESTURES_FOR_LLM,
    PREDICTION_AVG_MOTION_THRESHOLD,
    PREDICTION_CONSENSUS_FRAMES,
    PREDICTION_INITIAL_CONSENSUS_FRAMES,
    PREDICTION_KEEP_LAST_STABLE_FRAMES,
    PREDICTION_MIN_CONFIDENCE_GAP,
    PREDICTION_MOTION_THRESHOLD,
    PREDICTION_MOTION_VARIANCE_MIN,
    PREDICTION_REST_WEIGHT,
    PREDICTION_SIGNIFICANT_MOTION_MIN_RATIO,
    PREDICTION_SWITCH_CONSENSUS_FRAMES,
    PREDICTION_UNCERTAIN_TOKEN,
    SEQUENCE_DECODER_REST_SWITCH_PENALTY,
    SEQUENCE_DECODER_SWITCH_PENALTY,
    SEQUENCE_LENGTH,
)
from core.inference.gesture_translations import (
    load_gesture_translations,
    translate_gesture,
    translate_gestures,
)
from gui.services.serial_service import SerialService, SerialSettings
from gui.utils.formatting import now_hms
from gui.utils.smoothing import PredictionSmoother, SentenceAssembler
from utils.serial_utils import ContinuousWarningBeeper, FlexZeroWarningMonitor

if TYPE_CHECKING:
    from gui.services.model_service import ModelService


_SIGNIFICANT_MOTION_MIN_FRAMES = int(
    SEQUENCE_LENGTH * PREDICTION_SIGNIFICANT_MOTION_MIN_RATIO
)


def calculate_motion_magnitude(sensor_dict: dict[str, float]) -> float:
    """Calculate combined accel + gyro motion magnitude from sensor data."""
    return math.hypot(
        sensor_dict.get("accelX", 0.0),
        sensor_dict.get("accelY", 0.0),
        sensor_dict.get("accelZ", 0.0),
    ) + math.hypot(
        sensor_dict.get("gyroX", 0.0),
        sensor_dict.get("gyroY", 0.0),
        sensor_dict.get("gyroZ", 0.0),
    )


def validate_motion_consistency(motion_samples: deque[float]) -> bool:
    """Validate that motion remains meaningful over the prediction window."""
    if len(motion_samples) < SEQUENCE_LENGTH // 2:
        return False

    motion_array = np.asarray(motion_samples, dtype=float)
    avg_motion = float(np.mean(motion_array))
    if avg_motion < PREDICTION_AVG_MOTION_THRESHOLD:
        return False

    motion_variance = float(np.var(motion_array))
    if motion_variance < PREDICTION_MOTION_VARIANCE_MIN:
        return False

    significant_motion_frames = int(np.sum(motion_array > PREDICTION_MOTION_THRESHOLD))
    return significant_motion_frames >= _SIGNIFICANT_MOTION_MIN_FRAMES


def _confidence_gap_for_token(probabilities: dict[str, float], token: str) -> float:
    token_prob = float(probabilities.get(token, 0.0))
    rival_prob = max(
        (float(value) for key, value in probabilities.items() if key != token),
        default=0.0,
    )
    return max(0.0, token_prob - rival_prob)


def _load_per_class_thresholds(model_dir: Path | None) -> dict[str, dict[str, float]]:
    """Load class thresholds emitted by evaluation artifacts when available."""
    if model_dir is None:
        return {}

    candidate_paths = (
        model_dir / "evaluation" / "per_class_thresholds.json",
        model_dir / "per_class_thresholds.json",
    )
    for candidate in candidate_paths:
        if not candidate.exists():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return {
                    str(k).upper(): {
                        "confidence": float(v.get("confidence", 0.0)),
                        "gap": float(v.get("gap", 0.0)),
                    }
                    for k, v in payload.items()
                    if isinstance(v, dict)
                }
        except Exception:
            continue
    return {}


@dataclass(slots=True)
class TransitionHysteresis:
    """Asymmetric transition policy with uncertain fallback."""

    initial_consensus_frames: int = max(
        1, PREDICTION_INITIAL_CONSENSUS_FRAMES, PREDICTION_CONSENSUS_FRAMES
    )
    switch_consensus_frames: int = PREDICTION_SWITCH_CONSENSUS_FRAMES
    keep_last_stable_frames: int = PREDICTION_KEEP_LAST_STABLE_FRAMES
    uncertain_token: str = PREDICTION_UNCERTAIN_TOKEN
    stable_token: str | None = None
    candidate_token: str | None = None
    candidate_count: int = 0
    invalid_streak: int = 0

    def resolve(self, token: str, *, valid: bool, is_rest: bool) -> str:
        normalized = token.strip().upper()
        if is_rest:
            self.stable_token = None
            self.candidate_token = None
            self.candidate_count = 0
            self.invalid_streak = 0
            return "REST"

        if not valid:
            self.invalid_streak += 1
            self.candidate_token = None
            self.candidate_count = 0
            if (
                self.stable_token is not None
                and self.invalid_streak <= self.keep_last_stable_frames
            ):
                return self.stable_token
            return self.uncertain_token

        self.invalid_streak = 0

        if self.stable_token is None:
            if self.candidate_token == normalized:
                self.candidate_count += 1
            else:
                self.candidate_token = normalized
                self.candidate_count = 1
            if self.candidate_count >= max(1, self.initial_consensus_frames):
                self.stable_token = normalized
                self.candidate_token = None
                self.candidate_count = 0
                return normalized
            return self.uncertain_token

        if normalized == self.stable_token:
            self.candidate_token = None
            self.candidate_count = 0
            return self.stable_token

        if self.candidate_token == normalized:
            self.candidate_count += 1
        else:
            self.candidate_token = normalized
            self.candidate_count = 1

        if self.candidate_count >= max(1, self.switch_consensus_frames):
            self.stable_token = normalized
            self.candidate_token = None
            self.candidate_count = 0
            return normalized

        return self.stable_token


class SequenceDecoder:
    """Online Viterbi-like decoder with transition penalties."""

    def __init__(
        self,
        classes: list[str],
        *,
        enabled: bool = ENABLE_SEQUENCE_DECODER,
        switch_penalty: float = SEQUENCE_DECODER_SWITCH_PENALTY,
        rest_switch_penalty: float = SEQUENCE_DECODER_REST_SWITCH_PENALTY,
    ) -> None:
        self.enabled = enabled
        self.classes = [str(c) for c in classes]
        self.switch_penalty = max(0.0, float(switch_penalty))
        self.rest_switch_penalty = max(0.0, float(rest_switch_penalty))
        self._scores: dict[str, float] | None = None

    def _transition_penalty(self, prev_cls: str, next_cls: str) -> float:
        if prev_cls == next_cls:
            return 0.0
        if "REST" in {prev_cls.upper(), next_cls.upper()}:
            return -self.rest_switch_penalty
        return -self.switch_penalty

    def decode(self, probabilities: dict[str, float], fallback: str) -> str:
        if not probabilities:
            return fallback
        if not self.enabled:
            return max(probabilities.items(), key=lambda item: float(item[1]))[0]

        epsilon = 1e-8
        if self._scores is None:
            self._scores = {
                cls: math.log(max(epsilon, float(probabilities.get(cls, epsilon))))
                for cls in self.classes
            }
        else:
            next_scores: dict[str, float] = {}
            for next_cls in self.classes:
                emit = math.log(max(epsilon, float(probabilities.get(next_cls, epsilon))))
                best_prev = max(
                    (
                        self._scores.get(prev_cls, -1e9)
                        + self._transition_penalty(prev_cls, next_cls)
                    )
                    for prev_cls in self.classes
                )
                next_scores[next_cls] = best_prev + emit
            self._scores = next_scores

        return max(self._scores.items(), key=lambda item: item[1])[0]


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
        self._smoother = PredictionSmoother(
            config.smoothing_window, rest_weight=PREDICTION_REST_WEIGHT
        )
        self._sentence = SentenceAssembler()
        self._translations = load_gesture_translations()
        self._lock_state = GestureLockState()
        self._transition_hysteresis = TransitionHysteresis()
        self._warning_beeper = ContinuousWarningBeeper()
        model_dir = getattr(getattr(model_service, "metadata", None), "model_dir", None)
        self._per_class_thresholds = _load_per_class_thresholds(model_dir)
        self._decoder: SequenceDecoder | None = None
        self._flex_zero_monitor = FlexZeroWarningMonitor(
            logger,
            min_consecutive_samples=2,
            emit=lambda message: self._event_queue.put(
                {"type": "warning", "message": message}
            ),
        )

    def _thresholds_for(self, token: str) -> tuple[float, float]:
        per_class = self._per_class_thresholds.get(token.strip().upper())
        conf_threshold = self._config.confidence_threshold
        gap_threshold = PREDICTION_MIN_CONFIDENCE_GAP
        if per_class:
            conf_threshold = max(conf_threshold, float(per_class.get("confidence", 0.0)))
            gap_threshold = max(gap_threshold, float(per_class.get("gap", 0.0)))
        return conf_threshold, gap_threshold

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
            motion_samples: deque[float] = deque(maxlen=SEQUENCE_LENGTH)

            predictor_classes = getattr(predictor, "classes", [])
            if hasattr(predictor_classes, "tolist"):
                class_list = [str(c) for c in predictor_classes.tolist()]
            else:
                class_list = [str(c) for c in predictor_classes]
            self._decoder = SequenceDecoder(class_list or ["REST"])

            while not self._stop_event.is_set():
                # read_sensor_row() blocks on the SerialService queue until a
                # parsed frame is available or the timeout expires, so no extra
                # sleep is needed here.
                sensor = self._serial_service.read_sensor_row(timeout=0.2)
                if sensor is None:
                    continue

                zero_sensors = self._flex_zero_monitor.check(sensor)
                if zero_sensors:
                    self._warning_beeper.start()
                else:
                    self._warning_beeper.stop()

                motion_samples.append(calculate_motion_magnitude(sensor))

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

                gesture, confidence, confidence_gap, all_probs = predictor.predict()
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

                decoded_token = (
                    self._decoder.decode(all_probs or {}, str(gesture))
                    if self._decoder is not None
                    else str(gesture)
                )
                decoded_is_rest = decoded_token.strip().upper() == "REST"
                smooth_token = self._smoother.update(
                    decoded_token,
                    confidence=float(confidence),
                    is_rest=decoded_is_rest,
                )
                probabilities = {
                    str(key): float(value) for key, value in (all_probs or {}).items()
                }
                effective_confidence = float(
                    probabilities.get(smooth_token, float(confidence))
                )
                effective_gap = (
                    _confidence_gap_for_token(probabilities, smooth_token)
                    if probabilities
                    else float(confidence_gap or 0.0)
                )
                is_rest = smooth_token.strip().upper() == "REST"
                conf_threshold, gap_threshold = self._thresholds_for(smooth_token)
                is_confident = effective_confidence >= conf_threshold
                has_gap = effective_gap >= gap_threshold
                motion_ok = True if is_rest else validate_motion_consistency(motion_samples)
                filtered_token = self._transition_hysteresis.resolve(
                    smooth_token,
                    valid=(is_confident and has_gap and motion_ok),
                    is_rest=is_rest,
                )
                filtered_is_uncertain = (
                    filtered_token.strip().upper() == PREDICTION_UNCERTAIN_TOKEN
                )
                filtered_is_rest = filtered_token.strip().upper() == "REST"
                display_token = (
                    filtered_token
                    if not filtered_is_uncertain
                    else ("Belirsiz" if self._config.language == "tr" else "Uncertain")
                )

                if filtered_is_uncertain:
                    should_emit = False
                else:
                    should_emit, filtered_is_rest, _ = self._lock_state.observe(
                        filtered_token,
                        effective_confidence,
                        conf_threshold,
                    )

                self._event_queue.put(
                    {
                        "type": "prediction",
                        "timestamp": now_hms(),
                        "gesture": display_token,
                        "raw_gesture": filtered_token,
                        "confidence": effective_confidence,
                    }
                )

                if should_emit and not filtered_is_rest and not filtered_is_uncertain:
                    translated = translate_gesture(
                        filtered_token,
                        self._translations,
                        target_language=self._config.language,
                    )
                    if self._sentence.try_append(filtered_token):
                        self._event_queue.put(
                            {
                                "type": "sentence",
                                "token": translated,
                                "sentence": self._sentence.text(),
                            }
                        )

                if filtered_is_rest:
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

                    collected_gestures.append((filtered_token, effective_confidence))
        except Exception as exc:  # pragma: no cover - guarded by UI notifications
            self._logger.exception("Stream worker failed")
            self._event_queue.put(
                {
                    "type": "error",
                    "message": f"Streaming error: {exc}",
                }
            )
        finally:
            self._warning_beeper.stop()
            self._event_queue.put({"type": "connected", "value": False})
            self._event_queue.put({"type": "stopped"})
            self._logger.info("Streaming stopped")
