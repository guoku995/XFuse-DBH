"""Leakage-free confidence targets for target-frame-excluded temporal depth."""
from __future__ import annotations

import numpy as np
import torch
from scipy.spatial import cKDTree

from .data import TemporalPriorDataset


def reprojection_compatibility(
    prior_m: np.ndarray,
    prior_valid: np.ndarray,
    temporal_m: np.ndarray,
    temporal_valid: np.ndarray,
    neighbors: int = 4,
    spatial_scale_px: float = 24.0,
    max_distance_px: float = 96.0,
    log_depth_scale: float = 0.15,
    minimum_weight: float = 0.05,
) -> np.ndarray:
    """Score temporal supervision by agreement with current-frame LiDAR.

    This is a training-only target quality score.  It uses no segmentation,
    DBH metadata, or learned KNN map: target-frame-excluded temporal points are
    trusted only when a nearby raw LiDAR anchor supports the same metric depth.
    """

    prior_m = np.asarray(prior_m, dtype=np.float32)
    temporal_m = np.asarray(temporal_m, dtype=np.float32)
    prior_valid = np.asarray(prior_valid, dtype=bool)
    temporal_valid = np.asarray(temporal_valid, dtype=bool)
    if (
        prior_m.shape != temporal_m.shape
        or prior_valid.shape != prior_m.shape
        or temporal_valid.shape != temporal_m.shape
    ):
        raise ValueError("all depth and validity maps must share shape [H,W]")
    weight = np.zeros_like(temporal_m, dtype=np.float32)
    anchors_yx = np.column_stack(np.nonzero(prior_valid & np.isfinite(prior_m) & (prior_m > 0)))
    targets_yx = np.column_stack(np.nonzero(temporal_valid & np.isfinite(temporal_m) & (temporal_m > 0)))
    if anchors_yx.size == 0 or targets_yx.size == 0:
        return weight
    k = min(max(1, int(neighbors)), anchors_yx.shape[0])
    distance, neighbor = cKDTree(anchors_yx.astype(np.float32)).query(targets_yx.astype(np.float32), k=k)
    if k == 1:
        distance, neighbor = distance[:, None], neighbor[:, None]
    neighbor_yx = anchors_yx[neighbor]
    anchor_depth = prior_m[neighbor_yx[..., 0], neighbor_yx[..., 1]]
    target_depth = temporal_m[targets_yx[:, 0], targets_yx[:, 1]][:, None]
    log_error = np.abs(np.log(target_depth / np.maximum(anchor_depth, 1e-6)))
    compatibility = np.exp(-distance / spatial_scale_px - log_error / log_depth_scale)
    best = compatibility.max(axis=1)
    nearest = distance.min(axis=1)
    best[nearest > max_distance_px] = 0.0
    best[best < minimum_weight] = 0.0
    weight[targets_yx[:, 0], targets_yx[:, 1]] = best.astype(np.float32)
    return weight


class ReliableTemporalPriorDataset(TemporalPriorDataset):
    """Temporal dataset that adds a reprojection-compatible supervision weight."""

    def __init__(
        self,
        *args,
        reliability_neighbors: int = 4,
        reliability_spatial_scale_px: float = 24.0,
        reliability_max_distance_px: float = 96.0,
        reliability_log_depth_scale: float = 0.15,
        reliability_minimum_weight: float = 0.05,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.reliability_neighbors = int(reliability_neighbors)
        self.reliability_spatial_scale_px = float(reliability_spatial_scale_px)
        self.reliability_max_distance_px = float(reliability_max_distance_px)
        self.reliability_log_depth_scale = float(reliability_log_depth_scale)
        self.reliability_minimum_weight = float(reliability_minimum_weight)

    def __getitem__(self, index: int) -> dict:
        sample = super().__getitem__(index)
        weight = reprojection_compatibility(
            sample["prior_m"].squeeze(0).numpy(),
            sample["prior_valid"].squeeze(0).numpy(),
            sample["target_m"].squeeze(0).numpy(),
            sample["valid"].squeeze(0).numpy(),
            neighbors=self.reliability_neighbors,
            spatial_scale_px=self.reliability_spatial_scale_px,
            max_distance_px=self.reliability_max_distance_px,
            log_depth_scale=self.reliability_log_depth_scale,
            minimum_weight=self.reliability_minimum_weight,
        )
        sample["temporal_weight"] = torch.from_numpy(weight)[None]
        sample["temporal_weight_sum"] = torch.tensor(float(weight.sum()), dtype=torch.float32)
        return sample
