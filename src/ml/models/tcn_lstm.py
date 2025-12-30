"""
TCN-LSTM-Attention Model
=========================

Hybrid deep learning architecture combining:
1. Temporal Convolutional Network (TCN): Parallel processing, dilated convolutions
2. LSTM: Sequential memory, long-term dependencies
3. Multi-Head Attention: Focus on important time steps

Research shows TCN-LSTM hybrid outperforms standalone models.
Attention mechanism further improves by 10-15%.

Sources:
- MDPI 2024: Attention-LSTM variants
- Springer 2024: TCN-LSTM hybrid for stock prediction
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class CausalConv1d(nn.Module):
    """
    Causal convolution layer.
    Ensures no information leakage from future to past.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1
    ):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation
        )

    def forward(self, x):
        # x: (batch, channels, sequence)
        out = self.conv(x)
        # Remove future padding
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class TCNBlock(nn.Module):
    """
    Temporal Convolutional Network block with residual connection.

    Uses dilated causal convolutions to capture long-range dependencies
    while maintaining causality (no future information leakage).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.2
    ):
        super().__init__()

        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Residual connection
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

        self.layer_norm1 = nn.LayerNorm(out_channels)
        self.layer_norm2 = nn.LayerNorm(out_channels)

    def forward(self, x):
        # x: (batch, channels, sequence)
        residual = self.residual(x)

        out = self.conv1(x)
        out = out.transpose(1, 2)  # (batch, seq, channels) for LayerNorm
        out = self.layer_norm1(out)
        out = out.transpose(1, 2)  # Back to (batch, channels, seq)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = out.transpose(1, 2)
        out = self.layer_norm2(out)
        out = out.transpose(1, 2)
        out = self.relu(out)
        out = self.dropout(out)

        return self.relu(out + residual)


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention for focusing on important time steps.

    Research shows 4 attention heads achieve optimal accuracy
    for stock prediction tasks.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = np.sqrt(self.d_k)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = query.size(0)

        # Linear projections
        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        output = self.W_o(context)

        return output, attention_weights


class TCNLSTMAttention(nn.Module):
    """
    Hybrid TCN-LSTM model with Multi-Head Attention.

    Architecture:
    1. TCN: Extracts multi-scale temporal features using dilated convolutions
    2. LSTM: Captures long-term sequential dependencies
    3. Attention: Focuses on important time steps
    4. Output: Binary classification (up/down) with probability

    Parameters:
    -----------
    input_size : int
        Number of input features per time step
    hidden_size : int
        Size of hidden layers
    num_layers : int
        Number of LSTM layers
    tcn_channels : list
        Channel sizes for TCN layers
    kernel_size : int
        Kernel size for TCN convolutions
    n_heads : int
        Number of attention heads
    dropout : float
        Dropout probability
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        tcn_channels: list = None,
        kernel_size: int = 3,
        n_heads: int = 4,
        dropout: float = 0.2
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        if tcn_channels is None:
            tcn_channels = [64, 64, 128]

        # TCN layers with increasing dilation
        self.tcn_layers = nn.ModuleList()
        in_channels = input_size

        for i, out_channels in enumerate(tcn_channels):
            dilation = 2 ** i  # Exponential dilation: 1, 2, 4, 8, ...
            self.tcn_layers.append(
                TCNBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout
                )
            )
            in_channels = out_channels

        # LSTM for sequential processing
        self.lstm = nn.LSTM(
            input_size=tcn_channels[-1],
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )

        # Multi-Head Attention
        self.attention = MultiHeadAttention(
            d_model=hidden_size,
            n_heads=n_heads,
            dropout=dropout
        )

        # Layer norm after attention
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )

        # Store attention weights for interpretability
        self._attention_weights = None

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, sequence, features)
            return_attention: If True, also return attention weights

        Returns:
            Prediction probability (batch, 1) or tuple with attention
        """
        batch_size, seq_len, _ = x.shape

        # TCN: expects (batch, channels, sequence)
        x = x.transpose(1, 2)

        for tcn_layer in self.tcn_layers:
            x = tcn_layer(x)

        # Back to (batch, sequence, channels)
        x = x.transpose(1, 2)

        # LSTM
        lstm_out, _ = self.lstm(x)

        # Self-attention
        attended, attention_weights = self.attention(lstm_out, lstm_out, lstm_out)
        self._attention_weights = attention_weights

        # Residual + LayerNorm
        x = self.layer_norm(lstm_out + attended)

        # Use last time step for prediction
        x = x[:, -1, :]

        # Output
        output = self.fc(x)

        if return_attention:
            return output, attention_weights

        return output

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Get attention weights from last forward pass."""
        return self._attention_weights

    @property
    def device(self) -> torch.device:
        """Get device of model parameters."""
        return next(self.parameters()).device


class SimpleLSTM(nn.Module):
    """
    Simple LSTM model for comparison/fallback.

    Lighter weight than TCN-LSTM-Attention.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        return self.fc(last_out)
