"""Construct the official PriorDA v1.1 three-channel condition tensor."""
from __future__ import annotations

import numpy as np
import torch

from .alignment import AlignmentConfig, align_disparity, spatially_subsample_prior


CONDITION_ORDER = ("sparse_mask", "global_aligned", "knn_aligned")


def depth_to_disparity(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    disparity = np.zeros_like(depth)
    valid = np.isfinite(depth) & (depth > 0)
    disparity[valid] = 1.0 / depth[valid]
    return disparity


def disparity_to_depth(disparity: np.ndarray) -> np.ndarray:
    disparity = np.asarray(disparity, dtype=np.float32)
    depth = np.zeros_like(disparity)
    valid = np.isfinite(disparity) & (disparity > 0)
    depth[valid] = 1.0 / disparity[valid]
    return depth


def prior_affine(depth: np.ndarray, valid: np.ndarray) -> tuple[float, float]:
    values = np.asarray(depth, dtype=np.float32)[valid]
    values = values[np.isfinite(values) & (values > 0)]
    if values.size < 2:
        raise ValueError("at least two positive prior depths are required")
    minimum = float(values.min())
    span = float(values.max() - minimum)
    if not np.isfinite(span) or span <= np.finfo(np.float32).eps:
        raise ValueError("prior depth range is degenerate")
    return minimum, span


def _metric_depth_condition(depth: np.ndarray, minimum: float, span: float) -> np.ndarray:
    normalized_depth = (np.asarray(depth, dtype=np.float32) - minimum) / span
    return depth_to_disparity(normalized_depth)


def build_conditions(
    relative_disparity: np.ndarray,
    prior_depth_m: np.ndarray,
    prior_valid: np.ndarray | None = None,
    config: AlignmentConfig | None = None,
) -> tuple[np.ndarray, tuple[float, float], dict]:
    """Build ``[sparse mask, global condition, KNN condition]``.

    The ordering is the published v1.1 ordering and matches
    ``prior_depth_anything_vitb_1_1.pth``.  No semantic/DBH mask is accepted by
    this function, preventing measurement geometry from entering the model.
    """

    config = config or AlignmentConfig()
    predicted = np.asarray(relative_disparity, dtype=np.float32)
    prior = np.asarray(prior_depth_m, dtype=np.float32)
    if prior_valid is None:
        valid = np.isfinite(prior) & (prior > 0)
    else:
        valid = np.asarray(prior_valid, dtype=bool) & np.isfinite(prior) & (prior > 0)
    valid &= np.isfinite(predicted) & (predicted > 0)
    if prior.shape != predicted.shape or valid.shape != predicted.shape:
        raise ValueError("relative disparity and metric prior must share shape [H,W]")

    selected = spatially_subsample_prior(valid, config.max_prior_points)
    minimum, span = prior_affine(prior, selected)
    prior_disparity = depth_to_disparity(prior)
    global_disp, knn_disp, diagnostics = align_disparity(
        predicted, prior_disparity, selected, config
    )
    global_depth = disparity_to_depth(global_disp)
    knn_depth = disparity_to_depth(knn_disp)
    condition = np.stack(
        [
            selected.astype(np.float32),
            _metric_depth_condition(global_depth, minimum, span),
            _metric_depth_condition(knn_depth, minimum, span),
        ],
        axis=0,
    ).astype(np.float32)
    condition[~np.isfinite(condition)] = 0.0
    diagnostics.update(
        {
            "condition_order": list(CONDITION_ORDER),
            "prior_min_m": minimum,
            "prior_span_m": span,
            "input_prior_points": int(valid.sum()),
        }
    )
    return condition, (minimum, span), diagnostics


def decode_metric_depth(
    normalized_disparity: np.ndarray,
    affine: tuple[float, float],
) -> np.ndarray:
    """Apply the same de-normalization used by the official implementation."""

    minimum, span = affine
    normalized_depth = disparity_to_depth(normalized_disparity)
    return (minimum + span * normalized_depth).astype(np.float32)


def decode_metric_depth_torch(
    normalized_disparity: torch.Tensor,
    affine: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Differentiable batched counterpart of :func:`decode_metric_depth`.

    ``affine`` is ``[B, 2]`` and contains the metric minimum and span derived
    exclusively from the input prior. Non-positive output disparity follows
    the released PriorDA implementation and decodes to normalized depth zero.
    """

    if normalized_disparity.ndim not in {3, 4}:
        raise ValueError("normalized_disparity must have shape [B,H,W] or [B,1,H,W]")
    if affine.ndim != 2 or affine.shape[1] != 2:
        raise ValueError("affine must have shape [B,2]")
    if normalized_disparity.shape[0] != affine.shape[0]:
        raise ValueError("batch size mismatch between prediction and affine")

    # Metric inversion is kept in float32 even under AMP. In float16, the
    # reciprocal of the 1e-6 floor overflows before ``where`` masks it out,
    # which poisons otherwise finite gradients at ReLU-zero output pixels.
    working_disparity = normalized_disparity.float()
    working_affine = affine.float()
    view_shape = (affine.shape[0],) + (1,) * (normalized_disparity.ndim - 1)
    minimum = working_affine[:, 0].reshape(view_shape)
    span = working_affine[:, 1].reshape(view_shape)
    safe_disparity = working_disparity.clamp_min(eps)
    normalized_depth = torch.where(
        working_disparity > eps,
        safe_disparity.reciprocal(),
        torch.zeros_like(working_disparity),
    )
    return minimum + span * normalized_depth
