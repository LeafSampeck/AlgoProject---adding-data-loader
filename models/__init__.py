"""
Model registry. Every script builds models through build_model(name, config)
so all three models get exactly the same hyperparameters.

Names:
    "cnn"           -> CNNBaseline                 (Person 2, models/cnn_baseline.py)
    "cnn_std_attn"  -> CNNStandardAttention        (Person 2, models/standard_attention.py)
    "cnn_lin_attn"  -> HybridCNNLinearAttention    (Person 1, models/hybrid_model.py)
"""

MODEL_NAMES = ["cnn", "cnn_std_attn", "cnn_lin_attn"]


def build_model(name, config):
    cfg = dict(config)
    if name == "cnn":
        from models.cnn_baseline import CNNBaseline
        return CNNBaseline(**cfg)
    if name == "cnn_std_attn":
        from models.standard_attention import CNNStandardAttention
        return CNNStandardAttention(**cfg)
    if name == "cnn_lin_attn":
        try:
            from models.hybrid_model import HybridCNNLinearAttention
        except ImportError as e:
            raise ImportError(
                "models/hybrid_model.py (Person 1) is not in the repo yet, or does not "
                "define HybridCNNLinearAttention. Expected constructor: "
                "HybridCNNLinearAttention(**config) with the same keys as DEFAULT_CONFIG."
            ) from e
        return HybridCNNLinearAttention(**cfg)
    raise ValueError(f"unknown model name {name!r}; choose from {MODEL_NAMES}")


def available_models(config):
    """Names of models that can currently be built (skips Person 1's if absent)."""
    names = []
    for n in MODEL_NAMES:
        try:
            build_model(n, config)
            names.append(n)
        except ImportError:
            pass
    return names
