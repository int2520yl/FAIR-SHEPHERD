"""FAIR-SHEPHERD experiment configurations."""

SEEDS = (42, 43, 44, 45, 46)

TRAINING = {
    "optimizer": "Adam",
    "learning_rate": 0.001,
    "epochs": 40,
    "batch_size": 64,
    "vcs_weight": 0.1,
    "gradpen_structural_weight": 0.1,
    "gradpen_k": 10.0,
    "monotower_weight": 0.1,
}

__all__ = ["SEEDS", "TRAINING"]
