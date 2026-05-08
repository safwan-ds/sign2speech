"""Unit tests for augmentation module"""

import numpy as np

from utils.augmentation import TimeSeriesAugmenter, create_augmented_dataset


class TestTimeSeriesAugmenter:
    def setup_method(self):
        self.augmenter = TimeSeriesAugmenter(random_state=42)
        self.sequence = np.random.RandomState(42).rand(30, 11).astype(np.float32)

    def test_add_noise_shape(self):
        result = self.augmenter.add_noise(self.sequence)
        assert result.shape == self.sequence.shape
        assert result.dtype == np.float32

    def test_add_noise_changes_values(self):
        result = self.augmenter.add_noise(self.sequence)
        # Noise should alter values (not identical to input)
        assert not np.array_equal(result, self.sequence)
        # With small noise_level, values should remain close to original
        assert np.allclose(result, self.sequence, atol=0.1)

    def test_time_warp_shape(self):
        result = self.augmenter.time_warp(self.sequence)
        assert result.shape == self.sequence.shape

    def test_magnitude_warp_shape(self):
        result = self.augmenter.magnitude_warp(self.sequence)
        assert result.shape == self.sequence.shape

    def test_scale_shape(self):
        result = self.augmenter.scale(self.sequence)
        assert result.shape == self.sequence.shape

    def test_time_shift_shape(self):
        result = self.augmenter.time_shift(self.sequence)
        assert result.shape == self.sequence.shape

    def test_rotation_shape(self):
        result = self.augmenter.rotation(self.sequence)
        assert result.shape == self.sequence.shape

    def test_mixup_shape(self):
        seq2 = np.random.RandomState(99).rand(30, 11).astype(np.float32)
        result, lam = self.augmenter.mixup(self.sequence, seq2)
        assert result.shape == self.sequence.shape
        assert 0.0 <= lam <= 1.0

    def test_cutout_shape(self):
        result = self.augmenter.cutout(self.sequence)
        assert result.shape == self.sequence.shape

    def test_cutout_has_zeros(self):
        result = self.augmenter.cutout(self.sequence, num_holes=3, hole_size=5)
        assert np.any(result == 0)

    def test_augment_batch(self):
        batch = np.random.RandomState(42).rand(5, 30, 11).astype(np.float32)
        result = self.augmenter.augment_batch(batch)
        assert result.shape == batch.shape


class TestCreateAugmentedDataset:
    def test_augmentation_increases_size(self):
        X = np.random.RandomState(42).rand(10, 30, 11).astype(np.float32)
        y = np.array(["a"] * 5 + ["b"] * 5)
        X_aug, y_aug = create_augmented_dataset(
            X, y, augmentation_factor=2, random_state=42
        )
        assert len(X_aug) > len(X)
        assert len(X_aug) == len(y_aug)

    def test_labels_preserved(self):
        X = np.random.RandomState(42).rand(10, 30, 11).astype(np.float32)
        y = np.array(["a"] * 5 + ["b"] * 5)
        X_aug, y_aug = create_augmented_dataset(
            X, y, augmentation_factor=1, random_state=42
        )
        assert set(np.unique(y_aug)) == {"a", "b"}
