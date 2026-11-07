
"""
Baseline 2: CNN + standard (softmax) multi-head self-attention.

Input  : (batch, seq_len, 40)
Output : logits (batch, 3)

The attention is written out explicitly:
    A = softmax(Q K^T / sqrt(d_head))     -> a (T x T) matrix per head
    out = A V
so the O(T^2) time and memory cost is real and measurable in Experiment 2.

IMPORTANT for Experiment 2: keep use_sdpa=False when benchmarking.
torch's F.scaled_dot_product_attention may dispatch to FlashAttention /
memory-efficient kernels on GPU, which never materialize the T x T matrix and
would hide the quadratic memory cost. Linear attention (Person 1) is also
written in plain PyTorch, so the comparison is like-for-like. If you want, run
use_sdpa=True as an extra "optimized softmax attention" row and label it.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common import (CNNBackbone, ClassificationHead, EncoderBlock,
                           SinusoidalPositionalEncoding)


class SoftmaxSelfAttention(nn.Module):
    """Multi-head scaled dot-product self-attention. (B, T, d) -> (B, T, d)."""

    def __init__(self, d_model=64, n_heads=4, dropout=0.1, use_sdpa=False):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.dropout_p = dropout
        self.use_sdpa = use_sdpa
        self.last_attn = None  # set when store_attn=True, for optional attention plots
        self.store_attn = False

    def forward(self, x):
        B, T, d = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)           # each (B, H, T, d_head)

        if self.use_sdpa:
            out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout_p if self.training else 0.0)
        else:
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)   # (B, H, T, T)
            attn = scores.softmax(dim=-1)
            if self.store_attn:
                self.last_attn = attn.detach()
            out = self.attn_drop(attn) @ v                                # (B, H, T, d_head)

        out = out.transpose(1, 2).reshape(B, T, d)    # (B, T, d)
        return self.out(out)


class CNNStandardAttention(nn.Module):
    def __init__(self, in_features=40, d_model=64, cnn_kernel_size=3, cnn_layers=2,
                 n_heads=4, n_attn_layers=1, ffn_mult=2, num_classes=3, dropout=0.1,
                 use_pos_encoding=True, use_sdpa=False, **_unused):
        super().__init__()
        self.backbone = CNNBackbone(in_features, d_model, cnn_kernel_size, cnn_layers, dropout)
        self.pos = SinusoidalPositionalEncoding(d_model) if use_pos_encoding else nn.Identity()
        self.encoder = nn.Sequential(*[
            EncoderBlock(SoftmaxSelfAttention(d_model, n_heads, dropout, use_sdpa),
                         d_model, ffn_mult, dropout)
            for _ in range(n_attn_layers)
        ])
        self.head = ClassificationHead(d_model, num_classes, dropout)

    def forward(self, x):                 # (B, T, 40)
        h = self.backbone(x)              # (B, T, d)
        h = self.pos(h)
        h = self.encoder(h)               # (B, T, d)
        return self.head(h)               # (B, 3)