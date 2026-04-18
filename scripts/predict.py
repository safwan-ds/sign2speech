import math
import os
import sys
import time
import threading
from collections import deque
from typing import Sequence
import logging

import serial
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    COM_PORT,
    BAUD_RATE,
    TIMEOUT,
    SEQUENCE_LENGTH,
    CONFIDENCE_THRESHOLD,
    PREDICTION_MOTION_THRESHOLD,
    PREDICTION_CONSENSUS_FRAMES,
    PREDICTION_AVG_MOTION_THRESHOLD,
    PREDICTION_MOTION_VARIANCE_MIN,
    PREDICTION_SIGNIFICANT_MOTION_MIN_RATIO,
    PREDICTION_MIN_CONFIDENCE_GAP,
    PREDICTION_DEBUG_MODE,
    SERIAL_CONNECTION_DELAY,
    MIN_CONSECUTIVE_REST,
    MIN_GESTURES_FOR_LLM,
    setup_logging,
)
from utils.serial_utils import parse_sensor_data
from core.inference.gesture_predictor import (
    LSTMGesturePredictor,
    MODEL_PATH,
    ENCODER_PATH,
)
from core.inference.gesture_translations import (
    load_gesture_translations,
    translate_gestures,
)
from utils.llm_utils import load_qwen_model, generate_turkish_reply

logger = logging.getLogger(__name__)

# Precompute constant for motion validation
_SIGNIFICANT_MOTION_MIN_FRAMES = int(
    SEQUENCE_LENGTH * PREDICTION_SIGNIFICANT_MOTION_MIN_RATIO
)


def calculate_motion_magnitude(sensor_dict: dict) -> float:
    """Calculate combined accel + gyro motion magnitude from sensor data."""
    return math.hypot(
        sensor_dict.get("accelX", 0),
        sensor_dict.get("accelY", 0),
        sensor_dict.get("accelZ", 0),
    ) + math.hypot(
        sensor_dict.get("gyroX", 0),
        sensor_dict.get("gyroY", 0),
        sensor_dict.get("gyroZ", 0),
    )


def validate_motion_consistency(motion_samples: Sequence[float]) -> bool:
    """
    Validate that motion is consistent and meaningful throughout the sequence.
    Returns True if motion characteristics support a gesture prediction.
    """
    if len(motion_samples) < SEQUENCE_LENGTH // 2:
        return False

    motion_array = np.array(motion_samples)

    # Check 1: Average motion
    avg_motion = np.mean(motion_array)
    if avg_motion < PREDICTION_AVG_MOTION_THRESHOLD:
        if PREDICTION_DEBUG_MODE:
            logger.debug(
                f"Motion check: avg_motion={avg_motion:.1f} < threshold={PREDICTION_AVG_MOTION_THRESHOLD}"
            )
        return False

    # Check 2: Motion variance (motion should vary, not be constant)
    motion_variance = np.var(motion_array)
    if motion_variance < PREDICTION_MOTION_VARIANCE_MIN:
        if PREDICTION_DEBUG_MODE:
            logger.debug(
                f"Motion check: variance={motion_variance:.1f} < threshold={PREDICTION_MOTION_VARIANCE_MIN}"
            )
        return False

    # Check 3: Motion should not be just noise - require peaks
    # At least some frames should have significant motion
    significant_motion_frames = np.sum(motion_array > PREDICTION_MOTION_THRESHOLD)
    if significant_motion_frames < _SIGNIFICANT_MOTION_MIN_FRAMES:
        if PREDICTION_DEBUG_MODE:
            logger.debug(
                f"Motion check: significant_frames={significant_motion_frames} < threshold={_SIGNIFICANT_MOTION_MIN_FRAMES}"
            )
        return False

    if PREDICTION_DEBUG_MODE:
        logger.debug(
            f"Motion validation passed: avg={avg_motion:.1f}, var={motion_variance:.1f}, peaks={significant_motion_frames}"
        )

    return True


def main():
    """Main real-time prediction loop"""
    setup_logging("predict")

    logger.info("LSTM REAL-TIME GESTURE RECOGNITION")

    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model not found at {MODEL_PATH}")
        logger.error("Please train a model using train_model.py first")
        return

    if not os.path.exists(ENCODER_PATH):
        logger.error(f"Encoder not found at {ENCODER_PATH}")
        return

    # Initialize predictor
    try:
        predictor = LSTMGesturePredictor(MODEL_PATH, ENCODER_PATH)
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return

    translations = load_gesture_translations()

    llm = load_qwen_model()
    if llm is not None:
        from config import QWEN_N_GPU_LAYERS

        logger.info(f"Qwen GGUF loaded with GPU offload (layers: {QWEN_N_GPU_LAYERS})")

    # Motion detection settings
    logger.info("MOTION DETECTION (Rest vs Gesture Classification)")
    logger.info(
        f"Instantaneous Motion Threshold: {PREDICTION_MOTION_THRESHOLD} (peak motion)"
    )
    logger.info(
        f"Average Motion Threshold: {PREDICTION_AVG_MOTION_THRESHOLD} (sustained motion)"
    )
    logger.info(
        f"Motion Variance Min: {PREDICTION_MOTION_VARIANCE_MIN} (motion variation)"
    )
    logger.info(f"Min Confidence for Gestures: {CONFIDENCE_THRESHOLD:.1%}")
    logger.info(
        f"Min Confidence Gap: {PREDICTION_MIN_CONFIDENCE_GAP:.1%} (certainty margin)"
    )
    logger.info(f"Temporal Stability: {PREDICTION_CONSENSUS_FRAMES} consecutive frames")
    logger.info(
        "\nValidates:\n"
        "  - Peak motion detected\n"
        "  - Sustained motion throughout sequence\n"
        "  - Motion variation (not static pose)\n"
        "  - Model confidence above threshold\n"
        "  - Clear winner (confidence gap check)\n"
        "  - Consistent predictions over time\n"
    )

    # Connect to serial port
    logger.info(f"Connecting to {COM_PORT}...")
    try:
        ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=TIMEOUT)
        time.sleep(SERIAL_CONNECTION_DELAY)
        logger.info("Connected!")
    except Exception as e:
        logger.error(f"Could not connect to {COM_PORT}")
        logger.error(f"Details: {e}")
        return

    logger.info("COLLECTING DATA... (Press Ctrl+C to stop)")
    logger.info(f"Buffering {SEQUENCE_LENGTH} samples before first prediction...")

    last_gesture = None
    last_added_gesture = None
    collected_gestures: list[tuple[str, float]] = []
    motion_samples: deque[float] = deque(maxlen=SEQUENCE_LENGTH)
    prediction_history: deque = deque(maxlen=PREDICTION_CONSENSUS_FRAMES)
    consecutive_rest_frames = 0
    llm_busy = False  # Guard against overlapping LLM calls

    try:
        while True:
            if ser.in_waiting > 0:
                line = ser.readline().decode("utf-8", errors="ignore")
                sensor_dict = parse_sensor_data(line)

                if sensor_dict:  # Got complete reading in one line
                    # Calculate motion magnitude
                    motion = calculate_motion_magnitude(sensor_dict)
                    motion_samples.append(motion)
                    predictor.add_sensor_dict(sensor_dict)

                    # Try to predict
                    if predictor.can_predict():
                        gesture, confidence, confidence_gap, all_probs = (
                            predictor.predict()
                        )

                        # Add to prediction history for temporal stability
                        prediction_history.append((gesture, confidence, confidence_gap))

                        if gesture:
                            # Show all probabilities in debug mode
                            if PREDICTION_DEBUG_MODE and all_probs:
                                sorted_probs = sorted(
                                    all_probs.items(), key=lambda x: x[1], reverse=True
                                )
                                probs_str = ", ".join(
                                    [
                                        f"{name}: {prob:.1%}"
                                        for name, prob in sorted_probs
                                    ]
                                )
                                logger.debug(f"All predictions: {probs_str}")

                            # Multi-level validation for non-REST gestures
                            is_rest = gesture.upper() == "REST"
                            original_gesture = gesture  # Store original prediction

                            if not is_rest:
                                # Cheapest checks first, short-circuit on failure
                                valid = (
                                    confidence is not None
                                    and confidence_gap is not None
                                    and confidence >= CONFIDENCE_THRESHOLD
                                    and confidence_gap >= PREDICTION_MIN_CONFIDENCE_GAP
                                )

                                if PREDICTION_DEBUG_MODE and not valid:
                                    if (
                                        confidence is not None
                                        and confidence < CONFIDENCE_THRESHOLD
                                    ):
                                        logger.debug(
                                            f"Filtered {original_gesture}: confidence {confidence:.2%} < {CONFIDENCE_THRESHOLD:.2%}"
                                        )
                                    elif (
                                        confidence_gap is not None
                                        and confidence_gap
                                        < PREDICTION_MIN_CONFIDENCE_GAP
                                    ):
                                        logger.debug(
                                            f"Filtered {original_gesture}: confidence gap {confidence_gap:.2%} < {PREDICTION_MIN_CONFIDENCE_GAP:.2%}"
                                        )

                                # Check temporal consensus (consistent predictions)
                                if (
                                    valid
                                    and len(prediction_history)
                                    >= PREDICTION_CONSENSUS_FRAMES
                                ):
                                    valid = all(
                                        g == gesture for g, _, _ in prediction_history
                                    )
                                    if PREDICTION_DEBUG_MODE and not valid:
                                        logger.debug(
                                            f"Filtered {original_gesture}: failed temporal consensus"
                                        )

                                # ALWAYS check motion for non-REST gestures
                                if valid:
                                    valid = validate_motion_consistency(motion_samples)
                                    if PREDICTION_DEBUG_MODE and not valid:
                                        logger.debug(
                                            f"Filtered {original_gesture}: failed motion validation"
                                        )

                                if not valid:
                                    # Force to REST when gesture validation fails
                                    gesture = "REST"
                                    confidence = 1.0
                                    is_rest = True

                            # Re-check after validation
                            is_rest = gesture.upper() == "REST"
                            is_new = gesture != last_gesture

                            if is_rest:
                                consecutive_rest_frames += 1

                                # Trigger QWEN exactly once after sustained REST
                                if (
                                    consecutive_rest_frames == MIN_CONSECUTIVE_REST
                                    and len(collected_gestures) >= MIN_GESTURES_FOR_LLM
                                    and not llm_busy
                                ):
                                    logger.info("REST (confidence: 1.00)")

                                    gesture_names = [
                                        name for name, _ in collected_gestures
                                    ]
                                    translated_gestures = translate_gestures(
                                        gesture_names, translations
                                    )
                                    gesture_text = " ".join(translated_gestures)

                                    if llm is not None:

                                        def _run_llm(
                                            text: str = gesture_text,
                                        ) -> None:
                                            nonlocal llm_busy
                                            try:
                                                reply = generate_turkish_reply(
                                                    llm, text
                                                )
                                                if reply:
                                                    logger.info(f"QWEN: {reply}")
                                            finally:
                                                llm_busy = False

                                        llm_busy = True
                                        threading.Thread(
                                            target=_run_llm, daemon=True
                                        ).start()

                                    collected_gestures.clear()
                                    last_added_gesture = None
                                    consecutive_rest_frames = 0
                            elif is_new and confidence is not None:
                                # New gesture detected (not REST) - reset REST counter
                                consecutive_rest_frames = 0

                                # Only add if it's different from the last gesture that was added
                                if gesture != last_added_gesture:
                                    # Print gesture immediately
                                    gesture_display = gesture.replace("_", " ")
                                    logger.info(f"{gesture_display} ({confidence:.2%})")
                                    collected_gestures.append((gesture, confidence))
                                    last_added_gesture = gesture

                            last_gesture = gesture

    except KeyboardInterrupt:
        logger.info("\n\nStopping...")
    except Exception as e:
        logger.error(f"\nError: {e}")
    finally:
        if "ser" in locals() and ser.is_open:
            ser.close()
            logger.info("Closed connection")


if __name__ == "__main__":
    main()
