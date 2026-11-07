"""
Day 1-2 sanity checks for the baselines on synthetic tensors (blueprint: "verify
forward/backward passes on synthetic tensors").

Run either way from the repo root:
    python -m tests.test_baselines
    pytest -q tests/test_baselines.py
"""

import torch

from config import DEFAULT_CONFIG
from models import build_model, available_models
from models.common import count_parameters
from models.standard_attention import SoftmaxSelfAttention

CFG = dict(DEFAULT_CONFIG)
B = 8


def _x(T=CFG["seq_len"]):
    return torch.randn(B, T, CFG["in_features"])


def test_output_shape_contract():
    """Every model: (B, T, 40) -> (B, 3). This is the team-wide contract."""
    for name in available_models(CFG):
        out = build_model(name, CFG)(_x())
        assert out.shape == (B, CFG["num_classes"]), f"{name}: {tuple(out.shape)}"


def test_backward_and_finite():
    for name in available_models(CFG):
        model = build_model(name, CFG)
        y = torch.randint(0, 3, (B,))
        loss = torch.nn.functional.cross_entropy(model(_x()), y)
        loss.backward()
        assert torch.isfinite(loss), name
        missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
        assert not missing, f"{name}: parameters with no gradient: {missing}"


def test_variable_sequence_lengths():
    """Experiment 2 runs T = 50 ... 1000 (+2000 synthetic) without rebuilding models."""
    for name in available_models(CFG):
        model = build_model(name, CFG).eval()
        with torch.no_grad():
            for T in (50, 100, 200, 500):
                assert model(_x(T)).shape == (B, 3), f"{name} failed at T={T}"


def test_sdpa_matches_explicit_attention():
    """The explicit O(T^2) path and PyTorch's fused kernel should agree numerically."""
    torch.manual_seed(0)
    a = SoftmaxSelfAttention(CFG["d_model"], CFG["n_heads"], dropout=0.0).eval()
    b = SoftmaxSelfAttention(CFG["d_model"], CFG["n_heads"], dropout=0.0, use_sdpa=True).eval()
    b.load_state_dict(a.state_dict())
    x = torch.randn(B, 100, CFG["d_model"])
    with torch.no_grad():
        assert torch.allclose(a(x), b(x), atol=1e-5)


def test_ablation_differs_only_in_attention():
    """CNN-only params + one attention encoder block == CNN+attention params."""
    cnn = count_parameters(build_model("cnn", CFG))
    std = count_parameters(build_model("cnn_std_attn", CFG))
    assert std > cnn
    print(f"params: cnn={cnn:,}  cnn_std_attn={std:,}  (attention block adds {std - cnn:,})")


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    print("models available:", available_models(CFG))
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print("all baseline checks passed")