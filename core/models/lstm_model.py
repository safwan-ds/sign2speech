"""
PyTorch LSTM model architecture for Sign Language Glove gesture recognition
Enhanced with attention mechanism, bidirectional layers, and residual connections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any


class AttentionLayer(nn.Module):
    """
    Self-attention mechanism for sequence data
    """

    def __init__(self, hidden_size: int):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, lstm_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply attention to LSTM outputs

        Args:
            lstm_out: LSTM output (batch_size, seq_len, hidden_size)

        Returns:
            context: Weighted context vector (batch_size, hidden_size)
            attention_weights: Attention weights (batch_size, seq_len)
        """
        # Calculate attention scores
        attention_scores = self.attention(lstm_out)  # (batch, seq_len, 1)
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch, seq_len, 1)

        # Apply attention weights
        context = torch.sum(attention_weights * lstm_out, dim=1)  # (batch, hidden_size)

        return context, attention_weights.squeeze(-1)


class LSTMModel(nn.Module):
    """
    PyTorch LSTM model for gesture recognition

    Architecture:
        - Bidirectional LSTM layers with dropout
        - Self-attention mechanism
        - Batch normalization
        - Residual connections in FC layers
        - Output layer for classification
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        dropout_rate: float,
        bidirectional: bool = True,
        use_attention: bool = True,
        use_batch_norm: bool = True,
        use_cnn: bool = False,
    ):
        """
        Initialize Enhanced LSTM model

        Args:
            input_size: Number of features per timestep
            hidden_size: Number of LSTM hidden units
            num_layers: Number of LSTM layers
            num_classes: Number of gesture classes to predict
            dropout_rate: Dropout rate for regularization
            bidirectional: Use bidirectional LSTM
            use_attention: Use attention mechanism
            use_batch_norm: Use batch normalization
        """
        super(LSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.use_batch_norm = use_batch_norm
        self.use_cnn = use_cnn

        # Calculate effective hidden size (doubled if bidirectional)
        self.num_directions = 2 if bidirectional else 1
        lstm_output_size = hidden_size * self.num_directions

        if use_cnn:
            self.conv1 = nn.Conv1d(
                in_channels=input_size,
                out_channels=input_size,
                kernel_size=3,
                padding=1,
            )
            self.conv_bn = nn.BatchNorm1d(input_size)
            self.conv_pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
            self.conv_dropout = nn.Dropout(dropout_rate)

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        # Attention mechanism
        if use_attention:
            self.attention = AttentionLayer(lstm_output_size)

        # Batch normalization
        if use_batch_norm:
            self.bn1 = nn.BatchNorm1d(lstm_output_size)
            self.bn2 = nn.BatchNorm1d(128)
            self.bn3 = nn.BatchNorm1d(64)

        # Fully connected layers with residual connections
        self.fc1 = nn.Linear(lstm_output_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc_out = nn.Linear(32, num_classes)

        # Projection layers for residual connections
        self.proj1 = (
            nn.Linear(lstm_output_size, 128) if lstm_output_size != 128 else None
        )
        self.proj2 = nn.Linear(128, 64)

        # Dropout and activation
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

    def _apply_temporal_cnn(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.conv_bn(x)
        x = self.relu(x)
        x = self.conv_pool(x)
        x = self.conv_dropout(x)
        return x.transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        if self.use_cnn:
            x = self._apply_temporal_cnn(x)

        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size * num_directions)

        # Apply attention or use last output
        if self.use_attention:
            out, _ = self.attention(lstm_out)  # (batch, hidden_size * num_directions)
        else:
            out = lstm_out[:, -1, :]  # Take last output

        # Apply batch normalization if enabled
        if self.use_batch_norm:
            out = self.bn1(out)

        # First FC layer with residual connection
        identity = self.proj1(out) if self.proj1 is not None else out
        out = self.fc1(out)
        if self.use_batch_norm:
            out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = out + identity  # Residual connection

        # Second FC layer with residual connection
        identity = self.proj2(out)
        out = self.fc2(out)
        if self.use_batch_norm:
            out = self.bn3(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = out + identity  # Residual connection

        # Third FC layer
        out = self.relu(self.fc3(out))
        out = self.dropout(out)

        # Output layer
        out = self.fc_out(out)

        return out


class TransformerLSTMModel(nn.Module):
    """
    Advanced LSTM model with transformer-style architecture.
    Includes multi-head attention, layer normalization, and feed-forward network.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        dropout_rate: float,
        bidirectional: bool = True,
        use_cnn: bool = False,
    ):
        super(TransformerLSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.use_cnn = use_cnn

        if use_cnn:
            self.conv1 = nn.Conv1d(
                in_channels=input_size,
                out_channels=input_size,
                kernel_size=3,
                padding=1,
            )
            self.conv_bn = nn.BatchNorm1d(input_size)
            self.conv_pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
            self.conv_dropout = nn.Dropout(dropout_rate)

        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.input_ln = nn.LayerNorm(hidden_size)

        # Bidirectional LSTM layers
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        lstm_output_size = hidden_size * self.num_directions

        # Multi-head attention
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=lstm_output_size,
            num_heads=4,
            dropout=dropout_rate,
            batch_first=True,
        )

        # Layer normalization
        self.ln1 = nn.LayerNorm(lstm_output_size)
        self.ln2 = nn.LayerNorm(lstm_output_size)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(lstm_output_size * 2, lstm_output_size),
            nn.Dropout(dropout_rate),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes),
        )

    def _apply_temporal_cnn(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.conv_bn(x)
        x = F.gelu(x)
        x = self.conv_pool(x)
        x = self.conv_dropout(x)
        return x.transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with transformer-style architecture

        Args:
            x: Input tensor (batch_size, seq_len, input_size)

        Returns:
            Output tensor (batch_size, num_classes)
        """
        if self.use_cnn:
            x = self._apply_temporal_cnn(x)

        # Input projection
        x = self.input_proj(x)
        x = self.input_ln(x)

        # LSTM
        lstm_out, _ = self.lstm(x)
        lstm_out = self.ln1(lstm_out)

        # Multi-head attention with residual
        attn_out, _ = self.multihead_attn(lstm_out, lstm_out, lstm_out)
        x = lstm_out + attn_out  # Residual connection

        # Feed-forward with residual
        x = self.ln2(x)
        ffn_out = self.ffn(x)
        x = x + ffn_out  # Residual connection

        # Global average pooling
        x = torch.mean(x, dim=1)

        # Classification
        out = self.classifier(x)

        return out


def build_lstm_model(
    input_size: int,
    num_classes: int,
    hidden_size: int,
    num_layers: int,
    dropout_rate: float,
    device: torch.device | None = None,
    model_type: str = "enhanced",
    **kwargs: Any,
) -> nn.Module:
    """
    Build and return LSTM model

    Args:
        input_size: Number of features per timestep
        num_classes: Number of gesture classes
        hidden_size: Number of LSTM hidden units
        num_layers: Number of LSTM layers
        dropout_rate: Dropout rate for regularization
        device: PyTorch device (cuda/cpu), auto-detected if None
        model_type: Model type ('basic', 'enhanced', 'advanced')
        **kwargs: Additional model arguments

    Returns:
        LSTMModel: Initialized model on the specified device
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_type == "advanced":
        model = TransformerLSTMModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            bidirectional=kwargs.get("bidirectional", True),
            use_cnn=kwargs.get("use_cnn", False),
        )
    elif model_type == "enhanced":
        model = LSTMModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            bidirectional=kwargs.get("bidirectional", True),
            use_attention=kwargs.get("use_attention", True),
            use_batch_norm=kwargs.get("use_batch_norm", True),
            use_cnn=kwargs.get("use_cnn", False),
        )
    else:  # basic
        model = LSTMModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            bidirectional=False,
            use_attention=False,
            use_batch_norm=False,
            use_cnn=kwargs.get("use_cnn", False),
        )

    return model.to(device)
