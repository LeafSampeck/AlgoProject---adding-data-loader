"""
Building blocks shared by ALL three models.

Why this file exists: Experiment 3 (ablation) only means something if the three
models differ in exactly one place. So the CNN front end, positional encoding,
encoder block wrapper, pooling and classification head live here, and each
model only chooses which attention module (none / softmax / linear) to plug in.

Person 1: please build hybrid_model.py from CNNBackbone + EncoderBlock +
ClassificationHead below and pass your LinearSelfAttention as `attention`.
Your attention module only needs: forward(x: (B, T, d)) -> (B, T, d).
"""

import math

import torch
import torch.nn as nn


class CNNBackbone(nn.Module):
    """1D CNN over time. (B, T, F) -> (B, T, d_model).

    'same' padding keeps the sequence length T unchanged, so the attention layer
    downstream sees the full history length (needed for Experiment 2).
    """

    def __init__(self, in_features=40, d_model=64, kernel_size=3, n_layers=2, dropout=0.1):
        super().__init__()
        assert kernel_size % 2 == 1, "use an odd kernel so 'same' padding is symmetric"
        layers = []
        ch_in = in_features
        for _ in range(n_layers):
            layers += [
                nn.Conv1d(ch_in, d_model, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            ch_in = d_model
        self.net = nn.Sequential(*layers)

    def forward(self, x):                 # x: (B, T, F)
        x = x.transpose(1, 2)             # (B, F, T) for Conv1d
        x = self.net(x)                   # (B, d, T)
        return x.transpose(1, 2)          # (B, T, d)


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sin/cos encoding. Computed on the fly so any T works (50 ... 2000+)."""

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, x):                 # x: (B, T, d)
        T, d = x.size(1), self.d_model
        pos = torch.arange(T, device=x.device, dtype=x.dtype).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2, device=x.device, dtype=x.dtype)
                        * (-math.log(10000.0) / d))
        pe = torch.zeros(T, d, device=x.device, dtype=x.dtype)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].size(1)])
        return x + pe.unsqueeze(0)


class EncoderBlock(nn.Module):
    """Pre-norm Transformer encoder block with a pluggable attention module.

    x = x + Dropout(Attn(LN(x)))
    x = x + Dropout(FFN(LN(x)))
    """

    def __init__(self, attention: nn.Module, d_model=64, ffn_mult=2, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = attention
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult * d_model, d_model),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.drop1(self.attn(self.norm1(x)))
        x = x + self.drop2(self.ffn(self.norm2(x)))
        return x


class ClassificationHead(nn.Module):
    """Mean-pool over time, then LayerNorm -> Dropout -> Linear. (B, T, d) -> (B, C)."""

    def __init__(self, d_model=64, num_classes=3, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = x.mean(dim=1)                 # (B, d)
        return self.fc(self.drop(self.norm(x)))


def count_parameters(model: nn.Module, trainable_only=True) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)