"""Depth losses and metrics independent of DBH geometry."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def silog_loss(
    prediction_m: torch.Tensor,
    target_m: torch.Tensor,
    valid: torch.Tensor,
    lambda_scale: float = 0.85,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Scale-invariant log loss used for dense valid-pixel supervision."""

    keep = valid.bool() & torch.isfinite(prediction_m) & torch.isfinite(target_m)
    keep &= (prediction_m > eps) & (target_m > eps)
    if not keep.any():
        return prediction_m.sum() * 0.0
    delta = torch.log(prediction_m[keep]) - torch.log(target_m[keep])
    variance = delta.square().mean() - lambda_scale * delta.mean().square()
    return torch.sqrt(variance.clamp_min(0.0) + eps)


def robust_log_depth_loss(
    prediction_m: torch.Tensor,
    target_m: torch.Tensor,
    valid: torch.Tensor,
    beta: float = 0.1,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Smooth-L1 loss on log depth, robust to temporal projection outliers."""

    if beta <= 0:
        raise ValueError("beta must be positive")
    keep = valid.bool() & torch.isfinite(target_m) & (target_m > eps)
    if not keep.any():
        return prediction_m.sum() * 0.0
    prediction = prediction_m[keep].float().clamp_min(eps)
    target = target_m[keep].float()
    delta = torch.log(prediction) - torch.log(target)
    return F.smooth_l1_loss(delta, torch.zeros_like(delta), beta=beta)


def depth_metrics(prediction_m: np.ndarray, target_m: np.ndarray, valid: np.ndarray) -> dict:
    pred = np.asarray(prediction_m, dtype=np.float64)
    target = np.asarray(target_m, dtype=np.float64)
    # ``&=`` must not mutate a caller-owned boolean mask.  The evaluator
    # reuses the same ground-truth support for every compared method.
    keep = np.asarray(valid, dtype=bool).copy()
    keep &= np.isfinite(pred) & np.isfinite(target) & (pred > 0) & (target > 0)
    if not keep.any():
        return {"pixels": 0, "abs_rel": None, "rmse_m": None, "silog": None}
    pred, target = pred[keep], target[keep]
    delta = pred - target
    log_delta = np.log(pred) - np.log(target)
    silog = np.sqrt(max(0.0, np.mean(np.square(log_delta)) - np.mean(log_delta) ** 2))
    return {
        "pixels": int(keep.sum()),
        "abs_rel": float(np.mean(np.abs(delta) / target)),
        "rmse_m": float(np.sqrt(np.mean(np.square(delta)))),
        "silog": float(silog),
    }
