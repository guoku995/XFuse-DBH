"""Coarse disparity alignment used by Prior Depth Anything.

The implementation follows the paper's global affine fit and K=5 spatial KNN
affine fit.  The only model-side change is optional Huber/MAD weighting of the
LiDAR anchor residuals.  Its scale is estimated from each input, so no
dataset- or DBH-specific rejection threshold is required.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.spatial import cKDTree


EPS = 1e-12
HUBER_C = 1.345
MAD_TO_SIGMA = 1.4826


@dataclass(frozen=True)
class AlignmentConfig:
    """Configuration shared by training and inference."""

    k: int = 5
    robust: bool = True
    irls_steps: int = 3
    query_chunk: int = 65536
    max_prior_points: int = 0
    max_knn_support_points: int = 20000
    # The released PriorDA implementation uses inverse Euclidean distance
    # weights.  The paper's displayed equation uses an inverse square.  Keep
    # this explicit so a run cannot silently change the alignment rule.
    distance_power: float = 1.0

    def validate(self) -> None:
        if self.k < 2:
            raise ValueError("KNN affine alignment requires k >= 2")
        if self.irls_steps < 0:
            raise ValueError("irls_steps must be non-negative")
        if self.query_chunk < 1:
            raise ValueError("query_chunk must be positive")
        if self.max_prior_points < 0:
            raise ValueError("max_prior_points must be non-negative")
        if self.max_knn_support_points < 0:
            raise ValueError("max_knn_support_points must be non-negative")
        if not np.isfinite(self.distance_power) or self.distance_power < 0:
            raise ValueError("distance_power must be a finite non-negative number")

    def to_dict(self) -> dict:
        return asdict(self)


def _mad_scale(values: np.ndarray, axis=None, keepdims: bool = False) -> np.ndarray:
    median = np.median(values, axis=axis, keepdims=True)
    mad = np.median(np.abs(values - median), axis=axis, keepdims=keepdims)
    return MAD_TO_SIGMA * mad


def _huber_weights(residual: np.ndarray, scale: np.ndarray | float) -> np.ndarray:
    cutoff = HUBER_C * np.maximum(scale, EPS)
    magnitude = np.abs(residual)
    return np.where(magnitude <= cutoff, 1.0, cutoff / np.maximum(magnitude, EPS))


def _weighted_affine(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve y ~= scale*x + shift along the final dimension."""

    weight_sum = weights.sum(axis=-1)
    safe_sum = np.maximum(weight_sum, EPS)
    mean_x = (weights * x).sum(axis=-1) / safe_sum
    mean_y = (weights * y).sum(axis=-1) / safe_sum
    centered_x = x - mean_x[..., None]
    centered_y = y - mean_y[..., None]
    variance_x = (weights * np.square(centered_x)).sum(axis=-1)
    covariance = (weights * centered_x * centered_y).sum(axis=-1)
    scale = covariance / np.maximum(variance_x, EPS)
    shift = mean_y - scale * mean_x

    reference = safe_sum * np.maximum((weights * np.square(x)).sum(axis=-1) / safe_sum, 1.0)
    stable = (
        np.isfinite(scale)
        & np.isfinite(shift)
        & (weight_sum > EPS)
        & (variance_x > np.finfo(np.float64).eps * reference)
    )
    return scale, shift, stable


def fit_global_affine(
    predicted_disparity: np.ndarray,
    prior_disparity: np.ndarray,
    valid: np.ndarray,
    robust: bool = True,
    irls_steps: int = 3,
) -> tuple[float, float, dict]:
    """Fit metric disparity = scale * relative disparity + shift."""

    x = np.asarray(predicted_disparity[valid], dtype=np.float64)
    y = np.asarray(prior_disparity[valid], dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size < 2:
        raise ValueError("at least two finite prior points are required")

    weights = np.ones_like(x)
    scale, shift, stable = _weighted_affine(x[None], y[None], weights[None])
    if not bool(stable[0]):
        raise ValueError("global affine fit is rank deficient")
    for _ in range(irls_steps if robust else 0):
        residual = y - (scale[0] * x + shift[0])
        sigma = float(_mad_scale(residual))
        if sigma <= EPS:
            break
        weights = _huber_weights(residual, sigma)
        scale, shift, stable = _weighted_affine(x[None], y[None], weights[None])
        if not bool(stable[0]):
            raise ValueError("robust global affine fit became rank deficient")

    residual = y - (scale[0] * x + shift[0])
    sigma = float(_mad_scale(residual))
    diagnostics = {
        "points": int(x.size),
        "scale": float(scale[0]),
        "shift": float(shift[0]),
        "residual_mad_sigma": sigma,
    }
    return float(scale[0]), float(shift[0]), diagnostics


def spatially_subsample_prior(valid: np.ndarray, max_points: int) -> np.ndarray:
    """Deterministically thin a prior while retaining raster-wide coverage."""

    valid = np.asarray(valid, dtype=bool)
    flat = np.flatnonzero(valid.ravel())
    if max_points <= 0 or flat.size <= max_points:
        return valid.copy()
    positions = np.linspace(0, flat.size - 1, max_points, dtype=np.int64)
    selected = np.zeros(valid.size, dtype=bool)
    selected[flat[positions]] = True
    return selected.reshape(valid.shape)


def align_disparity(
    predicted_disparity: np.ndarray,
    prior_disparity: np.ndarray,
    prior_valid: np.ndarray,
    config: AlignmentConfig,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return globally and locally aligned dense metric-disparity maps."""

    config.validate()
    predicted = np.asarray(predicted_disparity, dtype=np.float32)
    prior = np.asarray(prior_disparity, dtype=np.float32)
    valid = np.asarray(prior_valid, dtype=bool)
    if predicted.ndim != 2 or prior.shape != predicted.shape or valid.shape != predicted.shape:
        raise ValueError("predicted disparity, prior disparity and mask must share shape [H,W]")
    valid &= np.isfinite(predicted) & np.isfinite(prior) & (predicted > 0) & (prior > 0)
    if int(valid.sum()) < config.k:
        raise ValueError(f"need at least {config.k} valid prior points, found {int(valid.sum())}")

    global_scale, global_shift, global_diagnostics = fit_global_affine(
        predicted, prior, valid, robust=config.robust, irls_steps=config.irls_steps
    )
    global_map = global_scale * predicted.astype(np.float64) + global_shift

    support = spatially_subsample_prior(valid, config.max_knn_support_points)
    support_y, support_x = np.nonzero(support)
    support_xy = np.column_stack((support_y, support_x)).astype(np.float64)
    support_pred = predicted[support].astype(np.float64)
    support_prior = prior[support].astype(np.float64)
    global_residual = support_prior - (global_scale * support_pred + global_shift)
    if config.robust:
        residual_scale = float(_mad_scale(global_residual))
        anchor_weights = _huber_weights(global_residual, residual_scale)
    else:
        residual_scale = float(_mad_scale(global_residual))
        anchor_weights = np.ones_like(global_residual)

    tree = cKDTree(support_xy)
    height, width = predicted.shape
    local_map = np.empty(predicted.size, dtype=np.float32)
    fallback_count = 0
    k = min(config.k, support_xy.shape[0])

    for start in range(0, predicted.size, config.query_chunk):
        stop = min(start + config.query_chunk, predicted.size)
        flat = np.arange(start, stop, dtype=np.int64)
        query_xy = np.column_stack((flat // width, flat % width))
        distance, neighbor = tree.query(query_xy, k=k, workers=-1)
        if k == 1:
            distance = distance[:, None]
            neighbor = neighbor[:, None]

        x = support_pred[neighbor]
        y = support_prior[neighbor]
        weights = 1.0 / np.maximum(
            np.power(np.asarray(distance, dtype=np.float64), config.distance_power), 1.0
        )
        weights *= anchor_weights[neighbor]
        scale, shift, stable = _weighted_affine(x, y, weights)

        if config.robust and config.irls_steps > 0:
            for _ in range(config.irls_steps):
                residual = y - (scale[:, None] * x + shift[:, None])
                sigma = _mad_scale(residual, axis=1, keepdims=True)
                robust_weights = _huber_weights(residual, sigma)
                updated_scale, updated_shift, updated_stable = _weighted_affine(
                    x, y, weights * robust_weights
                )
                replace = stable & updated_stable
                scale = np.where(replace, updated_scale, scale)
                shift = np.where(replace, updated_shift, shift)
                stable &= updated_stable

        query_pred = predicted.ravel()[flat].astype(np.float64)
        fitted = scale * query_pred + shift
        stable &= np.isfinite(fitted) & (fitted > 0) & (scale > 0)
        global_fitted = global_scale * query_pred + global_shift
        fitted = np.where(stable, fitted, global_fitted)
        fallback_count += int((~stable).sum())
        local_map[start:stop] = fitted.astype(np.float32)

    local_map = local_map.reshape(predicted.shape)
    global_map = global_map.astype(np.float32)
    global_map[valid] = prior[valid]
    local_map[valid] = prior[valid]
    diagnostics = {
        "method": "prior_da_knn_huber_mad" if config.robust else "prior_da_knn",
        "valid_prior_points": int(valid.sum()),
        "knn_support_points": int(support.sum()),
        "local_fallback_pixels": fallback_count,
        "anchor_residual_mad_sigma": residual_scale,
        "global": global_diagnostics,
    }
    return global_map, local_map, diagnostics
