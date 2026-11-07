"""
Baseline 1: lightweight CNN (no attention).

Input  : (batch, seq_len, 40)
Output : logits (batch, 3)

Identical to the other two models except the attention block is removed, so it
isolates what attention adds (Experiment 3, row 1).
"""

import torch.nn as nn

from models.common import CNNBackbone, ClassificationHead


class CNNBaseline(nn.Module):
    def __init__(self, in_features=40, d_model=64, cnn_kernel_size=3, cnn_layers=2,
                 num_classes=3, dropout=0.1, **_unused):
        super().__init__()
        self.backbone = CNNBackbone(in_features, d_model, cnn_kernel_size, cnn_layers, dropout)
        self.head = ClassificationHead(d_model, num_classes, dropout)

    def forward(self, x):                 # (B, T, 40)
        h = self.backbone(x)              # (B, T, d)
        return self.head(h)               # (B, 3)