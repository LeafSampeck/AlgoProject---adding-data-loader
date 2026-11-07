"""
Shared experiment configuration.

Every model (CNN, CNN + standard attention, CNN + linear attention) MUST be
built and trained from this one config so the comparison is fair. If the team
changes a value, change it here only, and record the final values in the paper
(Section IV / V).
"""

DEFAULT_CONFIG = {
    # ---- data contract (agreed with Person 3) ----
    "in_features": 40,        # 10 levels x (bid price, bid vol, ask price, ask vol)
    "seq_len": 100,           # history window T
    "num_classes": 3,         # Up / Stationary / Down
    "horizon_k": 10,          # FI-2010 prediction horizon (one of 10, 20, 30, 50, 100)

    # ---- shared architecture (agreed with Person 1) ----
    "d_model": 64,            # CNN output channels == attention width
    "cnn_kernel_size": 3,
    "cnn_layers": 2,
    "n_heads": 4,
    "n_attn_layers": 1,
    "ffn_mult": 2,            # FFN hidden = ffn_mult * d_model
    "dropout": 0.1,
    "use_pos_encoding": True, # sinusoidal; works for any sequence length

    # ---- shared training settings (Person 1's train.py should read these) ----
    "seed": 42,
    "batch_size": 64,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 30,
    "early_stop_patience": 5,
    "use_class_weights": True,  # FI-2010 is imbalanced (Stationary dominates at small k)
}

# FI-2010 ships labels as 1, 2, 3. We map them to 0, 1, 2 in this order.
# Person 3: confirm this mapping in fi2010_loader.py.
CLASS_NAMES = ["Up", "Stationary", "Down"]

# Display names used in tables and figures.
MODEL_DISPLAY_NAMES = {
    "cnn": "CNN",
    "cnn_std_attn": "CNN + standard attention",
    "cnn_lin_attn": "CNN + linear attention",
}