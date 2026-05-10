"""Per-sample uncentered NMSE, matching SPARC's reference loss.

Reference: https://github.com/AtlasAnalyticsLab/SPARC/blob/main/sparc/loss.py
(itself adapted from OpenAI's sparse_autoencoder).
"""

import torch


def normalized_mean_squared_error(
    reconstruction: torch.Tensor,
    original_input: torch.Tensor,
) -> torch.Tensor:
    """Per-sample uncentered NMSE.

    For each row in the batch, computes ‖r − t‖² / ‖t‖² (where ‖·‖² is the
    mean squared element, equivalent to ‖·‖₂² up to a constant factor that
    cancels). Returns the mean across the batch.
    """
    return (
        ((reconstruction - original_input) ** 2).mean(dim=1)
        / (original_input ** 2).mean(dim=1).clamp(min=1e-8)
    ).mean()


def normalized_L1_loss(
    latent_activations: torch.Tensor,
    original_input: torch.Tensor,
) -> torch.Tensor:
    """Per-sample L1 norm of latents normalized by ‖input‖₂. Logged, not optimized in top-k SAEs."""
    return (latent_activations.abs().sum(dim=1) / original_input.norm(dim=1).clamp(min=1e-8)).mean()
