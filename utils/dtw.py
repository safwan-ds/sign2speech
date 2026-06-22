"""DTW (Dynamic Time Warping) alignment utilities for gesture sequence matching.

Provides functions for computing DTW distances and paths, resampling
sequences, aligning sequences to class templates, and loading/saving
DTW template files.
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _euclidean_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Distance function shared by fastdtw and the local fallback."""
    return float(np.linalg.norm(np.asarray(left, dtype=float) - np.asarray(right, dtype=float)))


def _exact_dtw_path(
    sequence: np.ndarray,
    template: np.ndarray,
) -> tuple[float, list[tuple[int, int]]]:
    """Small, dependency-free DTW fallback for short inference sequences."""
    n_seq, n_template = len(sequence), len(template)
    if n_seq == 0 or n_template == 0:
        return float("inf"), []

    costs = np.full((n_seq + 1, n_template + 1), np.inf, dtype=np.float64)
    costs[0, 0] = 0.0
    backtrack: dict[tuple[int, int], tuple[int, int]] = {}

    for i in range(1, n_seq + 1):
        for j in range(1, n_template + 1):
            candidates = (
                (costs[i - 1, j], (i - 1, j)),
                (costs[i, j - 1], (i, j - 1)),
                (costs[i - 1, j - 1], (i - 1, j - 1)),
            )
            best_cost, best_prev = min(candidates, key=lambda item: item[0])
            costs[i, j] = (
                best_cost + _euclidean_distance(sequence[i - 1], template[j - 1])
            )
            backtrack[(i, j)] = best_prev

    path: list[tuple[int, int]] = []
    cursor = (n_seq, n_template)
    while cursor != (0, 0):
        i, j = cursor
        if i > 0 and j > 0:
            path.append((i - 1, j - 1))
        cursor = backtrack.get(cursor, (0, 0))
    path.reverse()
    return float(costs[n_seq, n_template]), path


def _dtw_path(
    sequence: np.ndarray,
    template: np.ndarray,
    radius: int = 5,
) -> tuple[float, list[tuple[int, int]]]:
    """Return DTW distance/path, preferring fastdtw when available."""
    try:
        from fastdtw import fastdtw  # type: ignore

        distance, path = fastdtw(
            sequence,
            template,
            radius=max(1, int(radius)),
            dist=_euclidean_distance,
        )
        return float(distance), [(int(i), int(j)) for i, j in path]
    except Exception:
        return _exact_dtw_path(sequence, template)


def resample_sequence(sequence: np.ndarray, target_length: int) -> np.ndarray:
    """Linearly resample a sequence along time to a target length."""
    arr = np.asarray(sequence, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("sequence must be a 2D array")
    if target_length <= 0:
        raise ValueError("target_length must be positive")
    if len(arr) == target_length:
        return arr.astype(np.float32, copy=True)
    if len(arr) == 0:
        return np.zeros((target_length, arr.shape[1]), dtype=np.float32)
    if len(arr) == 1:
        return np.repeat(arr, target_length, axis=0).astype(np.float32)

    source_x = np.linspace(0.0, 1.0, len(arr), dtype=np.float32)
    target_x = np.linspace(0.0, 1.0, target_length, dtype=np.float32)
    columns = [
        np.interp(target_x, source_x, arr[:, feature_idx])
        for feature_idx in range(arr.shape[1])
    ]
    return np.stack(columns, axis=1).astype(np.float32)


def align_sequence_to_template(
    sequence: np.ndarray,
    template: np.ndarray,
    *,
    radius: int = 5,
    target_length: int | None = None,
) -> tuple[np.ndarray, float]:
    """DTW-align an incoming sequence to a class template timeline."""
    seq = np.asarray(sequence, dtype=np.float32)
    tmpl = np.asarray(template, dtype=np.float32)
    if seq.ndim != 2 or tmpl.ndim != 2:
        raise ValueError("sequence and template must be 2D arrays")
    if seq.shape[1] != tmpl.shape[1]:
        raise ValueError(
            f"Feature mismatch: sequence has {seq.shape[1]}, template has {tmpl.shape[1]}"
        )

    distance, path = _dtw_path(seq, tmpl, radius=radius)
    output_length = int(target_length or len(tmpl))
    if not path:
        return resample_sequence(seq, output_length), distance

    grouped: list[list[np.ndarray]] = [[] for _ in range(len(tmpl))]
    for seq_idx, tmpl_idx in path:
        if 0 <= seq_idx < len(seq) and 0 <= tmpl_idx < len(tmpl):
            grouped[tmpl_idx].append(seq[seq_idx])

    aligned = np.empty_like(tmpl, dtype=np.float32)
    for tmpl_idx, rows in enumerate(grouped):
        if rows:
            aligned[tmpl_idx] = np.mean(np.stack(rows, axis=0), axis=0)
        else:
            nearest_idx = int(round(tmpl_idx * (len(seq) - 1) / max(1, len(tmpl) - 1)))
            aligned[tmpl_idx] = seq[nearest_idx]

    if len(aligned) != output_length:
        aligned = resample_sequence(aligned, output_length)
    return aligned.astype(np.float32), distance


def align_sequence_to_templates(
    sequence: np.ndarray,
    templates: dict[str, np.ndarray],
    *,
    radius: int = 5,
    target_length: int | None = None,
) -> tuple[np.ndarray, str | None, float | None]:
    """Align a sequence to the nearest class template and return the aligned copy."""
    if not templates:
        return np.asarray(sequence, dtype=np.float32), None, None

    best_label: str | None = None
    best_distance = float("inf")
    best_aligned: np.ndarray | None = None

    for label, template in templates.items():
        tmpl = np.asarray(template, dtype=np.float32)
        if tmpl.ndim != 2 or tmpl.shape[1] != np.asarray(sequence).shape[1]:
            continue
        aligned, distance = align_sequence_to_template(
            sequence,
            tmpl,
            radius=radius,
            target_length=target_length,
        )
        if distance < best_distance:
            best_label = str(label)
            best_distance = float(distance)
            best_aligned = aligned

    if best_aligned is None:
        return np.asarray(sequence, dtype=np.float32), None, None
    return best_aligned, best_label, best_distance


def load_dtw_templates(path_or_dir: str | Path | None) -> dict[str, np.ndarray]:
    """Load class templates from ``dtw_templates.npz`` when present."""
    if path_or_dir is None:
        return {}

    path = Path(path_or_dir)
    if path.is_dir():
        path = path / "dtw_templates.npz"
    if not path.exists():
        return {}

    try:
        payload = np.load(path, allow_pickle=True)
        if "templates" in payload.files:
            templates = np.asarray(payload["templates"], dtype=np.float32)
            labels = None
            if "classes" in payload.files:
                labels = payload["classes"]
            elif "labels" in payload.files:
                labels = payload["labels"]
            if labels is None:
                labels = [f"class_{idx}" for idx in range(len(templates))]
            return {
                str(label): np.asarray(template, dtype=np.float32)
                for label, template in zip(labels, templates)
                if np.asarray(template).ndim == 2
            }
        return {
            str(key): np.asarray(payload[key], dtype=np.float32)
            for key in payload.files
            if np.asarray(payload[key]).ndim == 2
        }
    except Exception as exc:
        logger.warning("Could not load DTW templates from %s: %s", path, exc)
        return {}
