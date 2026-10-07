"""Single-frame LiDAR-prior inference pipeline."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from depth_anything_v2 import build_backbone

from .alignment import AlignmentConfig
from .conditions import build_conditions, decode_metric_depth
from .model import PriorLidarModel


class PriorLidarPipeline:
    """Fuse one RGB frame and one projected raw LiDAR depth map."""

    def __init__(
        self,
        conditioned_checkpoint: Path,
        geometric_checkpoint: Path,
        device: str | None = None,
        conditioned_size: str = "vitb",
        geometric_size: str = "vitl",
        alignment: AlignmentConfig | None = None,
        training_checkpoint: bool = False,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.alignment = alignment or AlignmentConfig()
        if training_checkpoint:
            # Training checkpoints carry the adapter configuration.  Build the
            # matching topology before loading so the single-frame deployment
            # path accepts both the released PriorDA weights and residual runs.
            payload = torch.load(conditioned_checkpoint, map_location="cpu", weights_only=False)
            model_config = payload.get("model_config", {}) if isinstance(payload, dict) else {}
            residual_adapter = bool(model_config.get("residual_adapter", False))
            max_log_correction = float(model_config.get("residual_max_log", 0.05))
            self.model = PriorLidarModel(
                conditioned_size,
                residual_adapter=residual_adapter,
                max_log_correction=max_log_correction,
            )
            self.checkpoint = self.model.load_training_checkpoint(conditioned_checkpoint)
        else:
            self.model = PriorLidarModel(conditioned_size)
            self.checkpoint = self.model.load_official(conditioned_checkpoint)
        self.model.to(self.device).eval()

        self.geometric = build_backbone(depth_size=geometric_size)
        state = torch.load(geometric_checkpoint, map_location="cpu", weights_only=False)
        self.geometric.load_state_dict(state, strict=True)
        self.geometric.to(self.device).eval()
        for parameter in self.geometric.parameters():
            parameter.requires_grad = False

    @torch.inference_mode()
    def geometric_disparity(self, image_bgr: np.ndarray) -> np.ndarray:
        image = torch.from_numpy(np.ascontiguousarray(image_bgr)).permute(2, 0, 1)[None]
        disparity = self.geometric(image.to(self.device), 518, device=self.device)
        return disparity.squeeze().float().cpu().numpy().astype(np.float32)

    @torch.inference_mode()
    def infer(
        self,
        image_bgr: np.ndarray,
        prior_depth_m: np.ndarray,
        prior_valid: np.ndarray | None = None,
        relative_disparity: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict]:
        if image_bgr.shape[:2] != prior_depth_m.shape:
            raise ValueError("RGB and projected LiDAR depth must have the same resolution")
        if relative_disparity is None:
            relative_disparity = self.geometric_disparity(image_bgr)
        condition, affine, alignment_debug = build_conditions(
            relative_disparity, prior_depth_m, prior_valid, self.alignment
        )
        image = torch.from_numpy(np.ascontiguousarray(image_bgr)).permute(2, 0, 1)[None]
        condition_tensor = torch.from_numpy(condition)[None]
        output = self.model(image.to(self.device), condition_tensor.to(self.device), self.device)
        depth = decode_metric_depth(output.squeeze().float().cpu().numpy(), affine)
        return depth, {
            "alignment": alignment_debug,
            "affine": {"minimum_m": affine[0], "span_m": affine[1]},
            "raw_writeback": False,
            "single_frame": True,
        }
