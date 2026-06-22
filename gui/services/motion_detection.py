"""Standalone motion-detection utilities for the prediction pipeline."""

from __future__ import annotations

import json
import math
from collections import deque
from pathlib import Path

import numpy as np

from config.config import (
    PREDICTION_AVG_MOTION_THRESHOLD,
    PREDICTION_MOTION_THRESHOLD,
    PREDICTION_MOTION_VARIANCE_MIN,
    PREDICTION_SIGNIFICANT_MOTION_MIN_RATIO,
    SEQUENCE_LENGTH,
)


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


def _extract_class_list(predictor: object) -> list[str]:
    predictor_classes = getattr(predictor, "classes", [])
    if hasattr(predictor_classes, "tolist"):
        return [str(c) for c in predictor_classes.tolist()]
    return [str(c) for c in predictor_classes]
