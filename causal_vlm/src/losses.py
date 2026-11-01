"""Weighted causal objective reduced over all eligible forward pairs."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def hard_negative_mask(probabilities, targets, ratio=0.20):
    """Inputs are vectors of eligible pairs, never the full adjacency matrix.

    Retain all positives and floor(ratio * negatives), with at least one negative.
    """
    if not 0 < ratio <= 1:
        raise ValueError("Hard-negative ratio must be in (0, 1]")
    selected = targets == 1
    negative_indices = torch.nonzero(targets == 0, as_tuple=False).flatten()
    if negative_indices.numel():
        k = max(1, int(ratio * negative_indices.numel()))
        hardest = probabilities.detach()[negative_indices].topk(k).indices
        selected = selected.clone()
        selected[negative_indices[hardest]] = True
    return selected


def causal_loss(prediction, target, hnm_ratio=0.20, false_negative_weight=10.0):
    if prediction.ndim != 2 or prediction.shape[0] != prediction.shape[1] or target.shape != prediction.shape:
        raise ValueError("Prediction and target must be equally sized square matrices")
    if not torch.isfinite(prediction).all() or ((prediction < 0) | (prediction > 1)).any():
        raise ValueError("Predictions must be finite probabilities")
    if ((target != 0) & (target != 1)).any():
        raise ValueError("Targets must be binary")
    eligible = torch.ones_like(target, dtype=torch.bool).triu(1)
    if target[~eligible].any():
        raise ValueError("Targets must contain only forward edges")
    p, g = prediction[eligible].float(), target[eligible].float()
    if p.numel() == 0:
        return prediction.sum() * 0.0
    n_pos = g.sum()
    n_neg = g.numel() - n_pos
    # Positive weighting is unused without positives and equals zero without negatives.
    positive_weight = n_neg / n_pos if n_pos > 0 else g.new_tensor(1.0)
    selected = hard_negative_mask(p, g, hnm_ratio)
    weights = torch.where(g == 1, positive_weight, torch.ones_like(g))
    weights = torch.where((g == 1) & (p.detach() < 0.5), weights * false_negative_weight, weights)
    # CUDA autocast rejects probability-space BCE even when inputs are float32.
    # Compute the probability-space objective and its gradients in float32.
    with torch.autocast(device_type=p.device.type, enabled=False):
        bce = F.binary_cross_entropy(p, g, reduction="none")
    return (weights * selected * bce).sum() / p.numel()
