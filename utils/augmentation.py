"""
Data augmentation utilities for time series sensor data
Implements various augmentation techniques to improve model generalization
"""

import numpy as np
import logging
from typing import Callable
from scipy.interpolate import interp1d  # type: ignore
from scipy.ndimage import gaussian_filter1d  # type: ignore

from config import (
    ACCEL_X_IDX,
    ACCEL_Y_IDX,
    GYRO_X_IDX,
    GYRO_Y_IDX,
    TIME_WARP_KNOT,
    MAGNITUDE_WARP_KNOT,
    NOISE_LEVEL,
    TIME_WARP_SIGMA,
    MAGNITUDE_WARP_SIGMA,
    SCALE_RANGE,
    TIME_SHIFT_RANGE,
    ROTATION_MAX_ANGLE,
    AUGMENTATION_PROB,
    NUM_AUGMENTATIONS_PER_SAMPLE,
)

logger = logging.getLogger(__name__)


class TimeSeriesAugmenter:
    """
    Augmentation techniques for time series sensor data
    """

    def __init__(self, random_state: int | None = None):
        """
        Initialize augmenter

        Args:
            random_state: Random seed for reproducibility
        """
        self.rng = np.random.RandomState(random_state)

    def time_warp(
        self, sequence: np.ndarray, sigma: float = 0.2, knot: int = TIME_WARP_KNOT
    ):
        """
        Apply time warping to sequence

        Args:
            sequence: Input sequence (seq_len, num_features)
            sigma: Warping strength
            knot: Number of warping points

        Returns:
            Warped sequence
        """
        seq_len = sequence.shape[0]

        # Generate random warping curve
        time_warp_points = self.rng.randn(knot + 2) * sigma
        time_warp_points[0] = 0
        time_warp_points[-1] = 0

        # Create smooth warping function
        orig_steps = np.linspace(0, seq_len - 1, num=knot + 2)  # type: ignore
        warp_steps = orig_steps + time_warp_points * (seq_len / knot)  # type: ignore
        warp_steps = np.clip(warp_steps, 0, seq_len - 1)  # type: ignore

        # Interpolate
        warped_sequence = np.zeros_like(sequence)
        for i in range(sequence.shape[1]):
            f = interp1d(
                warp_steps,  # type: ignore
                sequence[orig_steps.astype(int), i],  # type: ignore
                kind="cubic",
                fill_value="extrapolate",  # type: ignore
            )
            warped_sequence[:, i] = f(np.arange(seq_len))  # type: ignore

        return warped_sequence.astype(np.float32)

    def magnitude_warp(
        self, sequence: np.ndarray, sigma: float = 0.2, knot: int = MAGNITUDE_WARP_KNOT
    ):
        """
        Apply magnitude warping to sequence

        Args:
            sequence: Input sequence (seq_len, num_features)
            sigma: Warping strength
            knot: Number of warping points

        Returns:
            Magnitude warped sequence
        """
        seq_len = sequence.shape[0]

        # Generate smooth magnitude curve
        warp_curve: np.ndarray = self.rng.randn(knot) * sigma
        warp_curve = np.interp(  # type: ignore
            np.arange(seq_len).astype(np.float64), np.linspace(0, seq_len - 1, num=knot), warp_curve  # type: ignore
        )
        warp_curve = 1 + gaussian_filter1d(warp_curve, sigma=seq_len / knot)  # type: ignore

        # Apply magnitude warping
        warped_sequence = sequence * warp_curve[:, np.newaxis]

        return warped_sequence.astype(np.float32)

    def add_noise(self, sequence: np.ndarray, noise_level: float = 0.01):
        """
        Add Gaussian noise to sequence

        Args:
            sequence: Input sequence (seq_len, num_features)
            noise_level: Standard deviation of noise

        Returns:
            Noisy sequence
        """
        noise = self.rng.randn(*sequence.shape) * noise_level
        noisy_sequence = sequence + noise

        return noisy_sequence.astype(np.float32)

    def scale(
        self, sequence: np.ndarray, scale_range: tuple[float, float] = (0.9, 1.1)
    ):
        """
        Scale sequence by random factor

        Args:
            sequence: Input sequence (seq_len, num_features)
            scale_range: Min and max scaling factors

        Returns:
            Scaled sequence
        """
        scale_factor = self.rng.uniform(scale_range[0], scale_range[1])
        scaled_sequence = sequence * scale_factor

        return scaled_sequence.astype(np.float32)

    def time_shift(self, sequence: np.ndarray, shift_range: float = 0.1):
        """
        Shift sequence in time

        Args:
            sequence: Input sequence (seq_len, num_features)
            shift_range: Max fraction of sequence length to shift

        Returns:
            Time-shifted sequence
        """
        seq_len = sequence.shape[0]
        max_shift = int(seq_len * shift_range)
        shift = self.rng.randint(-max_shift, max_shift + 1)

        if shift == 0:
            return sequence

        shifted_sequence = np.zeros_like(sequence)

        if shift > 0:
            shifted_sequence[shift:] = sequence[:-shift]
            # Pad with first value
            shifted_sequence[:shift] = sequence[0]
        else:
            shifted_sequence[:shift] = sequence[-shift:]
            # Pad with last value
            shifted_sequence[shift:] = sequence[-1]

        return shifted_sequence.astype(np.float32)

    def rotation(
        self,
        sequence: np.ndarray,
        max_angle: float = 15,
        accel_xy: tuple[int, int] = (ACCEL_X_IDX, ACCEL_Y_IDX),
        gyro_xy: tuple[int, int] = (GYRO_X_IDX, GYRO_Y_IDX),
    ):
        """
        Apply random rotation to accelerometer/gyroscope data

        Args:
            sequence: Input sequence (seq_len, num_features)
            max_angle: Maximum rotation angle in degrees
            accel_xy: Tuple of (accelX_index, accelY_index) in feature vector
            gyro_xy: Tuple of (gyroX_index, gyroY_index) in feature vector

        Returns:
            Rotated sequence (only affects IMU features)
        """
        rotated_sequence = sequence.copy()

        ax_idx, ay_idx = accel_xy
        gx_idx, gy_idx = gyro_xy
        n_features = sequence.shape[1]

        if n_features <= max(ax_idx, ay_idx):
            return rotated_sequence

        angle = self.rng.uniform(-max_angle, max_angle)
        angle_rad = np.radians(angle)

        # Simple 2D rotation in XY plane for accel and gyro
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        # Rotate accelerometer XY
        accel_x = sequence[:, ax_idx]
        accel_y = sequence[:, ay_idx]
        rotated_sequence[:, ax_idx] = cos_a * accel_x - sin_a * accel_y
        rotated_sequence[:, ay_idx] = sin_a * accel_x + cos_a * accel_y

        # Rotate gyroscope XY
        if n_features > max(gx_idx, gy_idx):
            gyro_x = sequence[:, gx_idx]
            gyro_y = sequence[:, gy_idx]
            rotated_sequence[:, gx_idx] = cos_a * gyro_x - sin_a * gyro_y
            rotated_sequence[:, gy_idx] = sin_a * gyro_x + cos_a * gyro_y

        return rotated_sequence.astype(np.float32)

    def mixup(self, sequence1: np.ndarray, sequence2: np.ndarray, alpha: float = 0.2):
        """
        Apply mixup augmentation between two sequences

        Args:
            sequence1: First sequence (seq_len, num_features)
            sequence2: Second sequence (seq_len, num_features)
            alpha: Mixup parameter

        Returns:
            Mixed sequence
        """
        lam = self.rng.beta(alpha, alpha)
        mixed_sequence = lam * sequence1 + (1 - lam) * sequence2

        return mixed_sequence.astype(np.float32), lam

    def cutout(self, sequence: np.ndarray, num_holes: int = 1, hole_size: int = 5):
        """
        Apply cutout (zeroing random time windows)

        Args:
            sequence: Input sequence (seq_len, num_features)
            num_holes: Number of cutout regions
            hole_size: Size of each cutout region

        Returns:
            Sequence with cutout applied
        """
        seq_len = sequence.shape[0]
        cutout_sequence = sequence.copy()

        for _ in range(num_holes):
            start = self.rng.randint(0, max(1, seq_len - hole_size))
            end = min(start + hole_size, seq_len)
            cutout_sequence[start:end] = 0

        return cutout_sequence.astype(np.float32)

    def augment_batch(
        self,
        sequences: np.ndarray,
        augmentation_prob: float = AUGMENTATION_PROB,
        num_augmentations: int = NUM_AUGMENTATIONS_PER_SAMPLE,
    ) -> np.ndarray:
        """
        Apply random augmentations to a batch of sequences

        Args:
            sequences: Batch of sequences (batch_size, seq_len, num_features)
            augmentation_prob: Probability of applying each augmentation
            num_augmentations: Number of augmentation techniques to apply per sequence

        Returns:
            Augmented batch
        """
        augmented_sequences: list[np.ndarray] = []

        for seq in sequences:
            # Randomly select augmentations
            augmentations: list[Callable[[np.ndarray], np.ndarray]] = []

            if self.rng.random() < augmentation_prob:
                augmentations.append(
                    lambda x: self.add_noise(x, noise_level=NOISE_LEVEL)
                )

            if self.rng.random() < augmentation_prob:
                augmentations.append(lambda x: self.time_warp(x, sigma=TIME_WARP_SIGMA))

            if self.rng.random() < augmentation_prob:
                augmentations.append(
                    lambda x: self.magnitude_warp(x, sigma=MAGNITUDE_WARP_SIGMA)
                )

            if self.rng.random() < augmentation_prob:
                augmentations.append(lambda x: self.scale(x, scale_range=SCALE_RANGE))

            if self.rng.random() < augmentation_prob:
                augmentations.append(
                    lambda x: self.time_shift(x, shift_range=TIME_SHIFT_RANGE)
                )

            if self.rng.random() < augmentation_prob:
                augmentations.append(
                    lambda x: self.rotation(x, max_angle=ROTATION_MAX_ANGLE)
                )

            # Apply selected augmentations
            augmented_seq = seq.copy()
            if augmentations:
                indices = self.rng.choice(
                    len(augmentations),
                    size=min(num_augmentations, len(augmentations)),
                    replace=False,
                )
                selected: list[Callable[[np.ndarray], np.ndarray]] = [
                    augmentations[i] for i in indices
                ]
                for aug in selected:
                    augmented_seq = aug(augmented_seq)

            augmented_sequences.append(augmented_seq)

        return np.array(augmented_sequences)


def create_augmented_dataset(
    X: np.ndarray,
    y: np.ndarray,
    augmentation_factor: int = 2,
    random_state: int | None = None,
):
    """
    Create augmented dataset by applying augmentations to original data

    Args:
        X: Original sequences (num_samples, seq_len, num_features)
        y: Original labels (num_samples,)
        augmentation_factor: How many augmented versions to create per sample
        random_state: Random seed

    Returns:
        X_augmented: Combined original + augmented data
        y_augmented: Combined labels
    """
    augmenter = TimeSeriesAugmenter(random_state=random_state)
    rng = np.random.RandomState(random_state)

    X_augmented_list = [X]
    y_augmented_list = [y]

    for _ in range(augmentation_factor):
        X_aug = augmenter.augment_batch(
            X,
            augmentation_prob=AUGMENTATION_PROB,
            num_augmentations=NUM_AUGMENTATIONS_PER_SAMPLE,
        )
        X_augmented_list.append(X_aug)
        y_augmented_list.append(y)

    # Mixup augmentation: blend same-class pairs for smoother decision boundaries
    mixup_X: list[np.ndarray] = []
    mixup_y: list = []
    unique_classes = np.unique(y)
    for cls in unique_classes:
        cls_idx = np.where(y == cls)[0]
        if len(cls_idx) < 2:
            continue
        n_mixup = max(1, len(cls_idx) // 4)  # ~25% extra from mixup
        for _ in range(n_mixup):
            i, j = rng.choice(cls_idx, size=2, replace=False)
            mixed, _ = augmenter.mixup(X[i], X[j])
            mixup_X.append(mixed)
            mixup_y.append(cls)
    if mixup_X:
        X_augmented_list.append(np.array(mixup_X))
        y_augmented_list.append(np.array(mixup_y))

    X_augmented = np.concatenate(X_augmented_list, axis=0)
    y_augmented = np.concatenate(y_augmented_list, axis=0)

    # Shuffle
    indices = rng.permutation(len(X_augmented))
    X_augmented = X_augmented[indices]
    y_augmented = y_augmented[indices]

    logger.info(f"Dataset augmented: {len(X)} -> {len(X_augmented)} samples")

    return X_augmented, y_augmented
