"""Core LSTM model training logic"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import logging

# Add parent directory to path for config imports
sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ),
)
from config import (
    RANDOM_STATE,
    USE_TEST_SPLIT,
    TEST_SIZE,
    LSTM_UNITS,
    LSTM_LAYERS,
    DROPOUT_RATE,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    USE_AUGMENTATION,
    AUGMENTATION_FACTOR,
    USE_WEIGHTED_LOSS,
    USE_LABEL_SMOOTHING,
    LABEL_SMOOTHING_FACTOR,
    USE_COSINE_ANNEALING,
    COSINE_T_0,
    COSINE_T_MULT,
    COSINE_ETA_MIN,
    LR_PLATEAU_FACTOR,
    LR_PLATEAU_PATIENCE,
    LR_PLATEAU_MIN,
    USE_WARMUP,
    WARMUP_EPOCHS,
    WARMUP_START_FACTOR,
    EARLY_STOPPING_PATIENCE,
    MIN_DELTA,
    GRADIENT_CLIP_VALUE,
    MODEL_TYPE,
    USE_BIDIRECTIONAL,
    USE_ATTENTION,
    USE_BATCH_NORM,
    MIN_STRATIFY_SAMPLES,
    DEFAULT_VALIDATION_SIZE,
    MIN_VALIDATION_SAMPLES_PER_CLASS,
)

from core.models.lstm_model import build_lstm_model
from utils.augmentation import create_augmented_dataset
from utils.evaluation import compute_class_weights

logger = logging.getLogger(__name__)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_data_split(X, y, label_encoder, separate_test_X=None, separate_test_y=None):
    """Prepare train-validation split and normalize data"""
    # Check if stratification is possible (need at least 2 samples per class)
    unique, counts = np.unique(y, return_counts=True)
    min_samples = min(counts)
    use_stratify = min_samples >= MIN_STRATIFY_SAMPLES

    if not use_stratify:
        logger.warning(f"Some classes have only {min_samples} sample(s).")
        logger.warning("Using regular split instead of stratified split.")

    # Split data - always create a proper validation split to avoid evaluating on training data
    if not USE_TEST_SPLIT:
        # No holdout — use a small validation split only for early stopping
        val_size = DEFAULT_VALIDATION_SIZE
        logger.info(
            f"USE_TEST_SPLIT is False. Using {val_size*100:.0f}% validation split (no holdout test set)"
        )
    elif TEST_SIZE <= 0.0 or (separate_test_X is not None and TEST_SIZE < 0.1):
        # Use default validation size
        val_size = DEFAULT_VALIDATION_SIZE
        logger.info(
            f"TEST_SIZE is small/zero. Using {val_size*100:.0f}% for validation split"
        )
        if separate_test_X is not None:
            logger.info("Will also evaluate on separate holdout test set")
    else:
        val_size = TEST_SIZE

    # Always do a stratified random split (not sequential!) to avoid data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=val_size,
        random_state=RANDOM_STATE,
        stratify=y if use_stratify else None,
    )

    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Validation samples: {len(X_test)}")

    # Warn if validation set is too small (less than configured samples per class on average)
    if len(X_test) < len(label_encoder.classes_) * MIN_VALIDATION_SAMPLES_PER_CLASS:
        logger.warning(f"Validation set is VERY SMALL ({len(X_test)} samples)!")

    # Check for NaN or Inf values
    if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
        logger.warning("NaN or Inf values detected in training data!")
        logger.warning("Replacing NaN/Inf with 0...")
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    if separate_test_X is not None:
        if np.any(np.isnan(separate_test_X)) or np.any(np.isinf(separate_test_X)):
            logger.warning("NaN or Inf values detected in separate test data!")
            separate_test_X = np.nan_to_num(
                separate_test_X, nan=0.0, posinf=0.0, neginf=0.0
            )

    # Normalize data (per feature across all sequences)
    logger.info("Normalizing data...")
    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])

    # Calculate mean and std from training data
    mean = np.mean(X_train_reshaped, axis=0)
    std = np.std(X_train_reshaped, axis=0)
    std[std == 0] = 1  # Avoid division by zero

    # Normalize both train and test
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    logger.info(
        f"Data range after normalization: [{X_train.min():.3f}, {X_train.max():.3f}]"
    )

    # Normalize separate test set if provided
    if separate_test_X is not None:
        separate_test_X = (separate_test_X - mean) / std

    return X_train, X_test, y_train, y_test, mean, std, separate_test_X


def train_lstm_model(X, y, separate_test_X=None, separate_test_y=None):
    """Train LSTM model

    Args:
        X: Training features
        y: Training labels
        separate_test_X: Optional separate test set features (completely unseen during training)
        separate_test_y: Optional separate test set labels

    Returns:
        Tuple of (model, label_encoder, X_test, y_test, history, mean, std, separate_test_X, separate_test_y)
    """
    logger.info("TRAINING LSTM MODEL")
    logger.info(f"Using device: {device}")

    # Encode labels - fit on all available labels to handle all gestures
    label_encoder = LabelEncoder()

    # If we have a separate test set, fit on all labels (train + test)
    # so the encoder knows about all possible gestures
    if separate_test_y is not None:
        all_labels = np.concatenate([y, separate_test_y])
        label_encoder.fit(all_labels)
        y_encoded = label_encoder.transform(y)
        separate_test_y_encoded = label_encoder.transform(separate_test_y)
        logger.info(
            f"Fit label encoder on {len(np.unique(all_labels))} unique gestures (train + test)"
        )
    else:
        y_encoded = label_encoder.fit_transform(y)
        separate_test_y_encoded = None

    num_classes = len(label_encoder.classes_)
    logger.info(f"Classes: {list(label_encoder.classes_)}")
    logger.info(f"Total training sequences: {len(X)}")
    if separate_test_X is not None and separate_test_y is not None:
        logger.info(f"Total test sequences: {len(separate_test_X)}")
    logger.info(f"Sequence shape: {X.shape}")

    # Check class distribution (original data, before augmentation)
    unique, counts = np.unique(y_encoded, return_counts=True)
    class_distribution = dict(zip(unique, counts))
    logger.info(f"Class distribution (original):")
    for idx, count in class_distribution.items():
        logger.info(f"  {label_encoder.classes_[idx]}: {count} samples")

    # Prepare data split BEFORE augmentation to prevent data leakage
    # (augmented copies of a sample must not appear in both train and validation)
    X_train, X_test, y_train, y_test, mean, std, separate_test_X = prepare_data_split(
        X, y_encoded, label_encoder, separate_test_X, separate_test_y
    )

    # Update separate test y with encoded labels
    if separate_test_y_encoded is not None:
        separate_test_y = separate_test_y_encoded

    # Apply data augmentation ONLY to training data (after split)
    if USE_AUGMENTATION and AUGMENTATION_FACTOR > 0:
        logger.info(
            f"Applying data augmentation to training set (factor={AUGMENTATION_FACTOR})..."
        )
        X_train, y_train = create_augmented_dataset(
            X_train,
            np.asarray(y_train),
            augmentation_factor=AUGMENTATION_FACTOR,
            random_state=RANDOM_STATE,
        )

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.LongTensor(y_test).to(device)

    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Build model
    input_size = X.shape[2]  # num_features
    model = build_lstm_model(
        input_size,
        num_classes,
        LSTM_UNITS,
        LSTM_LAYERS,
        DROPOUT_RATE,
        device=device,
        model_type=MODEL_TYPE,
        bidirectional=USE_BIDIRECTIONAL,
        use_attention=USE_ATTENTION,
        use_batch_norm=USE_BATCH_NORM,
    )

    logger.info("MODEL ARCHITECTURE")
    logger.info(str(model))

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Loss and optimizer
    if USE_WEIGHTED_LOSS:
        class_weights = compute_class_weights(y_train, num_classes).to(device)
        logger.info(f"Using weighted loss with class weights:")
        for i, (cls, weight) in enumerate(zip(label_encoder.classes_, class_weights)):
            logger.info(f"  {cls}: {weight:.3f}")
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=LABEL_SMOOTHING_FACTOR if USE_LABEL_SMOOTHING else 0.0,
        )
    else:
        criterion = nn.CrossEntropyLoss(
            label_smoothing=LABEL_SMOOTHING_FACTOR if USE_LABEL_SMOOTHING else 0.0
        )

    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # Learning rate scheduler (with optional warmup chained via SequentialLR)
    warmup_scheduler = None
    if USE_COSINE_ANNEALING:
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=COSINE_T_0, T_mult=COSINE_T_MULT, eta_min=COSINE_ETA_MIN
        )
        if USE_WARMUP:
            warmup_sched = optim.lr_scheduler.LinearLR(
                optimizer, start_factor=WARMUP_START_FACTOR, total_iters=WARMUP_EPOCHS
            )
            scheduler = optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_sched, cosine_scheduler],
                milestones=[WARMUP_EPOCHS],
            )
            logger.info(
                f"Using Cosine Annealing (T_0={COSINE_T_0}, T_mult={COSINE_T_MULT}) "
                f"with warmup for {WARMUP_EPOCHS} epochs"
            )
        else:
            scheduler = cosine_scheduler
            logger.info(
                f"Using Cosine Annealing with Warm Restarts (T_0={COSINE_T_0}, T_mult={COSINE_T_MULT})"
            )
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=LR_PLATEAU_FACTOR,
            patience=LR_PLATEAU_PATIENCE,
            min_lr=LR_PLATEAU_MIN,
        )
        if USE_WARMUP:
            warmup_scheduler = optim.lr_scheduler.LinearLR(
                optimizer, start_factor=WARMUP_START_FACTOR, total_iters=WARMUP_EPOCHS
            )
            logger.info(f"Using ReduceLROnPlateau + warmup for {WARMUP_EPOCHS} epochs")
        else:
            logger.info("Using ReduceLROnPlateau scheduler")

    # Training loop
    logger.info("TRAINING")

    best_val_acc = 0.0
    patience_counter = 0
    patience = EARLY_STOPPING_PATIENCE
    best_model_state = None
    best_epoch = 0

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=GRADIENT_CLIP_VALUE
            )

            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss /= len(test_loader)
        val_acc = 100 * val_correct / val_total

        # Save history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Update learning rate
        if USE_COSINE_ANNEALING:
            scheduler.step()
        elif warmup_scheduler is not None and epoch < WARMUP_EPOCHS:
            warmup_scheduler.step()
        else:
            scheduler.step(val_loss)  # type: ignore
        current_lr = optimizer.param_groups[0]["lr"]

        # Print progress
        logger.info(
            f"Epoch {epoch+1}/{EPOCHS} - "
            f"Loss: {train_loss:.4f} - Acc: {train_acc:.2f}% - "
            f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2f}% - "
            f"LR: {current_lr:.6f}"
        )

        # Early stopping based on validation accuracy (with MIN_DELTA threshold)
        if val_acc > best_val_acc + MIN_DELTA:
            best_val_acc = val_acc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            best_epoch = epoch + 1
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                logger.info(
                    f"Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch}"
                )
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                break

    history = {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "train_acc": train_accs,
        "val_acc": val_accs,
    }

    return (
        model,
        label_encoder,
        X_test,
        y_test,
        history,
        mean,
        std,
        separate_test_X,
        separate_test_y,
    )
