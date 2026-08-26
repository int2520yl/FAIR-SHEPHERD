"""Differentiable FAIR-SHEPHERD training objectives."""

from __future__ import annotations

from typing import Any, Hashable, Mapping, Sequence

import numpy as np

from .structural import ProxyPair, StructuralPolicy


def gradpen(
    task_loss: Any,
    gradients: Any,
    inputs: Any,
    levels: Sequence[Hashable],
    proxy_groups: Mapping[str, Sequence[Hashable]],
    policy: StructuralPolicy,
    proxy_pairs: Sequence[ProxyPair],
    *,
    structural_weight: float,
    k: float,
    vcs_weight: float,
    zero_tolerance: float = 1e-8,
) -> Any:
    """Add the GradPen structural and VCS penalties to task_loss."""

    torch, functional = _torch()
    _training_inputs(task_loss, gradients, inputs, policy, zero_tolerance, torch)
    weights = _weights(structural_weight, vcs_weight)
    k = float(k)
    if not np.isfinite(k) or k <= 0:
        raise ValueError("k must be finite and positive.")
    pairs = tuple(proxy_pairs)
    if not pairs or len({pair.name for pair in pairs}) != len(pairs):
        raise ValueError("At least one uniquely named proxy pair is required.")
    masks = _torch_level_masks(levels, policy, len(gradients), gradients.device, torch)
    group_masks = {}
    for pair in pairs:
        if pair.group_key not in proxy_groups:
            raise ValueError(f"Missing proxy group array: {pair.group_key}")
        group_masks[pair.group_key] = _label_values(
            proxy_groups[pair.group_key], len(gradients)
        )

    vcs_loss = _vcs_loss(gradients, masks, policy, zero_tolerance, torch)
    hls_loss = gradients.new_zeros(())
    projection_loss = gradients.new_zeros(())
    vulnerable = policy.vulnerable_indices.tolist()
    for level, level_mask in masks.items():
        rows = gradients[level_mask][:, vulnerable]
        for pair in pairs:
            labels = group_masks[pair.group_key]
            mask_a = level_mask & torch.as_tensor(
                labels == pair.group_a, dtype=torch.bool, device=gradients.device
            )
            mask_b = level_mask & torch.as_tensor(
                labels == pair.group_b, dtype=torch.bool, device=gradients.device
            )
            if not mask_a.any() or not mask_b.any():
                continue
            direction = (
                inputs[mask_a][:, vulnerable].mean(0) - inputs[mask_b][:, vulnerable].mean(0)
            ).detach()
            norm = torch.linalg.vector_norm(direction)
            if not torch.isfinite(norm) or norm.detach().item() <= zero_tolerance:
                continue
            expanded = direction.unsqueeze(0).expand_as(rows)
            hls_loss = hls_loss + functional.cosine_similarity(
                rows, expanded, dim=1, eps=zero_tolerance
            ).abs().mean()
            unit = direction / norm
            projection_loss = projection_loss + (rows @ unit).abs().mean()
    structural_loss = hls_loss + projection_loss / k
    return task_loss + weights[0] * structural_loss + weights[1] * vcs_loss


def monotower(
    task_loss: Any,
    gradients: Any,
    levels: Sequence[Hashable],
    policy: StructuralPolicy,
    harmful_signs: Mapping[str, int],
    *,
    monotonicity_weight: float,
    vcs_weight: float,
    zero_tolerance: float = 1e-8,
) -> Any:
    """Add VCS and one-sided harmful coordinate-gradient penalties to task_loss."""

    torch, _ = _torch()
    _training_inputs(task_loss, gradients, gradients, policy, zero_tolerance, torch)
    weights = _weights(monotonicity_weight, vcs_weight)
    if not harmful_signs:
        raise ValueError("harmful_signs cannot be empty.")
    unknown = set(harmful_signs) - set(policy.vulnerable_features)
    if unknown:
        raise ValueError(f"harmful_signs contains non-vulnerable features: {sorted(unknown)}")
    for name, sign in harmful_signs.items():
        if isinstance(sign, bool) or sign not in (-1, 1):
            raise ValueError(f"The harmful sign for {name!r} must be -1 or +1.")

    masks = _torch_level_masks(levels, policy, len(gradients), gradients.device, torch)
    indices = [policy.feature_names.index(name) for name in harmful_signs]
    signs = gradients.new_tensor(list(harmful_signs.values()))
    mono_loss = torch.relu(gradients[:, indices] * signs).mean()
    vcs_loss = _vcs_loss(gradients, masks, policy, zero_tolerance, torch)
    return task_loss + weights[0] * mono_loss + weights[1] * vcs_loss


def _vcs_loss(gradients, masks, policy, tolerance, torch):
    normative = policy.normative_indices.tolist()
    loss = gradients.new_zeros(())
    for left, right in zip(policy.level_order[:-1], policy.level_order[1:]):
        if not masks[left].any() or not masks[right].any():
            continue
        a = gradients[masks[left]][:, normative].mean(0)
        b = gradients[masks[right]][:, normative].mean(0)
        cosine = torch.dot(a / (torch.norm(a) + tolerance), b / (torch.norm(b) + tolerance))
        loss = loss + 1.0 - cosine
    return loss


def _training_inputs(task_loss, gradients, inputs, policy, tolerance, torch):
    if not torch.is_tensor(task_loss) or task_loss.ndim != 0:
        raise ValueError("task_loss must be a scalar PyTorch tensor.")
    if not torch.is_tensor(gradients) or not torch.is_tensor(inputs):
        raise ValueError("gradients and inputs must be PyTorch tensors.")
    if gradients.ndim != 2 or inputs.shape != gradients.shape:
        raise ValueError("gradients and inputs must be equal-shape two-dimensional tensors.")
    if gradients.shape[1] != len(policy.feature_names):
        raise ValueError("The tensor feature dimension must match the policy.")
    if gradients.device != inputs.device or task_loss.device != gradients.device:
        raise ValueError("task_loss, gradients, and inputs must be on the same device.")
    if tolerance <= 0:
        raise ValueError("zero_tolerance must be positive.")


def _torch_level_masks(levels, policy, length, device, torch):
    values = _label_values(levels, length)
    masks = {level: np.asarray(values == level, dtype=bool) for level in policy.level_order}
    if np.any(np.sum(np.stack(tuple(masks.values())), axis=0) != 1):
        raise ValueError("levels must assign every row to exactly one declared level.")
    return {level: torch.as_tensor(mask, dtype=torch.bool, device=device) for level, mask in masks.items()}


def _label_values(values, length):
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    array = np.asarray(values, dtype=object)
    if array.ndim != 1 or len(array) != length:
        raise ValueError("Labels must contain one value per row.")
    return array


def _weights(*values):
    weights = tuple(float(value) for value in values)
    if not np.isfinite(weights).all() or any(value < 0 for value in weights):
        raise ValueError("Penalty weights must be finite and non-negative.")
    return weights


def _torch():
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as error:
        raise ImportError("Install fair-shepherd-metrics[methods] to use training methods.") from error
    return torch, functional
