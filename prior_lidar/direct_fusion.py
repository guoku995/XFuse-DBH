"""Direct multi-scale RGB-LiDAR disparity decoder.

Unlike an output residual adapter, this model never receives an official
PriorDA disparity as an input and never multiplies a base prediction by a
learned correction.  It starts from the released conditioned DINO/DPT weights,
injects a separately encoded LiDAR pyramid into all four DPT scales, and the
DPT decoder directly emits the final disparity map.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from depth_anything_v2 import build_backbone


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def lidar_pyramid_evidence(condition: torch.Tensor) -> torch.Tensor:
    """Return KNN-free sparse LiDAR evidence at the input resolution.

    The released conditioned encoder still consumes the published three
    channels for checkpoint compatibility.  The learned LiDAR stream receives
    only a raw-anchor mask, globally aligned depth, and three support radii.
    This keeps the new fusion path independent of the fixed KNN propagation.
    """

    safe = torch.nan_to_num(condition.float(), nan=0.0, posinf=0.0, neginf=0.0)
    sparse = safe[:, :1].clamp(0.0, 1.0)
    global_depth = safe[:, 1:2]
    peak = global_depth.amax(dim=(-2, -1), keepdim=True).clamp_min(1.0)
    global_depth = (global_depth / peak).clamp(0.0, 1.0)
    support_5 = F.max_pool2d(sparse, kernel_size=5, stride=1, padding=2)
    support_17 = F.max_pool2d(sparse, kernel_size=17, stride=1, padding=8)
    support_65 = F.max_pool2d(sparse, kernel_size=65, stride=1, padding=32)
    return torch.cat((sparse, global_depth, support_5, support_17, support_65), dim=1)


class BidirectionalCrossFusion(nn.Module):
    """Fuse one RGB feature scale with its learned LiDAR feature scale.

    RGB queries select a local LiDAR neighborhood, while a second branch uses
    RGB context to update LiDAR features before the direct fusion projection.
    The last projection is zero-initialized so loading the released checkpoint
    preserves its output exactly before training.
    """

    def __init__(self, channels: int, evidence_channels: int = 5, hidden_channels: int = 48) -> None:
        super().__init__()
        if hidden_channels % 8:
            raise ValueError("hidden_channels must be divisible by 8")
        self.evidence_encoder = nn.Sequential(
            nn.Conv2d(evidence_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.GELU(),
        )
        self.rgb_query = nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False)
        self.lidar_key = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, bias=False)
        self.lidar_value = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, bias=False)
        self.rgb_value = nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False)
        self.lidar_update = nn.Sequential(
            nn.Conv2d(2 * hidden_channels, hidden_channels, kernel_size=1),
            nn.GroupNorm(8, hidden_channels),
            nn.GELU(),
        )
        self.direct_fusion = nn.Sequential(
            nn.Conv2d(channels + 2 * hidden_channels, channels, kernel_size=1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        # The four fusion blocks are an exact identity before learning.  This
        # is a feature-space initialization condition, not an output residual.
        nn.init.zeros_(self.direct_fusion[-1].weight)
        nn.init.zeros_(self.direct_fusion[-1].bias)

    def forward(self, rgb: torch.Tensor, evidence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        height, width = rgb.shape[-2:]
        evidence = F.interpolate(evidence, (height, width), mode="bilinear", align_corners=False)
        lidar = self.evidence_encoder(evidence)
        query = self.rgb_query(rgb)
        key = self.lidar_key(lidar)
        value = self.lidar_value(lidar)
        batch, channels, _, _ = query.shape
        key_windows = F.unfold(key, kernel_size=3, padding=1).reshape(batch, channels, 9, height, width)
        value_windows = F.unfold(value, kernel_size=3, padding=1).reshape(batch, channels, 9, height, width)
        score = (query.unsqueeze(2) * key_windows).sum(dim=1) / math.sqrt(float(channels))
        support_windows = F.unfold(evidence[:, :1], kernel_size=3, padding=1).reshape(batch, 9, height, width)
        score = score + torch.log(support_windows.clamp_min(1e-4))
        attention = score.softmax(dim=1)
        rgb_to_lidar = (attention.unsqueeze(1) * value_windows).sum(dim=2)
        lidar_to_rgb = self.lidar_update(torch.cat((rgb_to_lidar, self.rgb_value(rgb)), dim=1))
        support = F.max_pool2d(evidence[:, :1], kernel_size=3, stride=1, padding=1)
        fused = self.direct_fusion(torch.cat((rgb, rgb_to_lidar, lidar_to_rgb), dim=1))
        return rgb + support * fused, fused


class DirectFusionDisparityModel(nn.Module):
    """Prior-initialized dual-stream decoder with a direct disparity output."""

    variant = "priorda_v1_1_dual_stream_direct_disparity"
    condition_channels = 3

    def __init__(
        self,
        size: str = "vitb",
        fusion_channels: int = 48,
        cross_fusion: bool = True,
        evidence: bool = True,
        fusion_scales: str = "all",
    ) -> None:
        super().__init__()
        if size not in {"vits", "vitb"}:
            raise ValueError("released PriorDA fine-stage sizes are vits and vitb")
        if fusion_scales not in {"all", "finest", "coarsest"}:
            raise ValueError("fusion_scales must be 'all', 'finest' or 'coarsest'")
        self.size = size
        self.fusion_channels = int(fusion_channels)

        self.cross_fusion_enabled = bool(cross_fusion)
        self.evidence_enabled = bool(evidence)
        self.fusion_scales = fusion_scales
        self.network = build_backbone(depth_size=size, encoder_cond_dim=self.condition_channels)
        self.network.construct_aux_layers()
        decoder_channels = 64 if size == "vits" else 128
        self.cross_fusion = nn.ModuleList(
            [BidirectionalCrossFusion(decoder_channels, hidden_channels=fusion_channels) for _ in range(4)]
        )


    def _active_fusion_scales(self) -> set[int]:
        if self.fusion_scales == "all":
            return {0, 1, 2, 3}
        if self.fusion_scales == "finest":
            return {0}
        return {3}

    @property
    def model_variant(self) -> str:
        return self.variant

    def train(self, mode: bool = True):  # type: ignore[override]
        super().train(mode)
        self.network.pretrained.eval()
        return self

    def load_official(self, checkpoint: Path, strict: bool = True) -> dict:
        checkpoint = Path(checkpoint)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = payload.get("model", payload)
        state = {key.replace("module.", "", 1): value for key, value in state.items()}
        missing, unexpected = self.network.load_state_dict(state, strict=strict)
        return {
            "path": str(checkpoint),
            "sha256": sha256_file(checkpoint),
            "missing": list(missing),
            "unexpected": list(unexpected),
            "source_epoch": payload.get("epoch") if isinstance(payload, dict) else None,
            "source_step": payload.get("step") if isinstance(payload, dict) else None,
        }

    def configure_trainable(self, train_decoder: bool = False) -> dict[str, int]:
        for parameter in self.network.pretrained.parameters():
            parameter.requires_grad = False
        for parameter in self.network.depth_head.parameters():
            parameter.requires_grad = bool(train_decoder)
        for parameter in self.cross_fusion.parameters():
            parameter.requires_grad = True
        return {
            "frozen_encoder": sum(parameter.numel() for parameter in self.network.pretrained.parameters()),
            "direct_decoder": sum(parameter.numel() for parameter in self.network.depth_head.parameters() if parameter.requires_grad),
            "dual_stream_fusion": sum(parameter.numel() for parameter in self.cross_fusion.parameters()),
        }

    def parameter_groups(self, fusion_lr: float, decoder_lr: float, weight_decay: float) -> list[dict]:
        fusion = [parameter for parameter in self.cross_fusion.parameters() if parameter.requires_grad]
        decoder = [parameter for parameter in self.network.depth_head.parameters() if parameter.requires_grad]
        groups = [{"params": fusion, "lr": fusion_lr, "weight_decay": weight_decay, "name": "dual_stream_fusion"}]
        if decoder:
            groups.append({"params": decoder, "lr": decoder_lr, "weight_decay": weight_decay, "name": "direct_decoder"})
        return groups

    def _encode(
        self, image_bgr: torch.Tensor, condition: torch.Tensor, device: str
    ) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], tuple[int, int, int, int]]:
        image, original_shape = self.network.raw2input(image_bgr, 518, device)
        resized_h, resized_w = image.shape[-2:]
        patch_h, patch_w = resized_h // 14, resized_w // 14
        conditioned = F.interpolate(condition, (resized_h, resized_w), mode="bilinear", align_corners=True)
        with torch.no_grad():
            features = self.network.pretrained.get_intermediate_layers(
                image,
                self.network.intermediate_layer_idx[self.network.encoder],
                return_class_token=True,
                condition=conditioned,
            )
        return features, (patch_h, patch_w, *original_shape)

    def _decode_direct(
        self,
        out_features: list[tuple[torch.Tensor, torch.Tensor]],
        evidence: torch.Tensor,
        patch_h: int,
        patch_w: int,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        head = self.network.depth_head
        projected = []
        for index, feature in enumerate(out_features):
            if head.use_clstoken:
                tokens, class_token = feature[0], feature[1]
                readout = class_token.unsqueeze(1).expand_as(tokens)
                tokens = head.readout_projects[index](torch.cat((tokens, readout), dim=-1))
            else:
                tokens = feature[0]
            tokens = tokens.permute(0, 2, 1).reshape(tokens.shape[0], tokens.shape[-1], patch_h, patch_w)
            projected.append(head.resize_layers[index](head.projects[index](tokens)))

        active = self._active_fusion_scales() if self.cross_fusion_enabled else set()
        if not self.evidence_enabled:
            evidence = torch.zeros_like(evidence)
        layers, deltas = [], []
        for index, feature in enumerate(projected):
            visual = getattr(head.scratch, f"layer{index + 1}_rn")(feature) #visual是不同尺度的RGBD特征
            if index in active:
                fused, delta = self.cross_fusion[index](visual, evidence) #cross_fusion列表有4个融合模块
            else:
                fused, delta = visual, torch.zeros_like(visual[:, :1])
            layers.append(fused)
            deltas.append(delta)
        path_4 = head.scratch.refinenet4(layers[3], size=layers[2].shape[2:])
        path_3 = head.scratch.refinenet3(path_4, layers[2], size=layers[1].shape[2:])
        path_2 = head.scratch.refinenet2(path_3, layers[1], size=layers[0].shape[2:])
        path_1 = head.scratch.refinenet1(path_2, layers[0])
        output = head.scratch.output_conv1(path_1)
        output = F.interpolate(output, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        return head.scratch.output_conv2(output), deltas

    def forward(
        self,
        image_bgr: torch.Tensor,
        condition: torch.Tensor,
        device: str,
        return_aux: bool = False,
        zero_evidence: bool | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if image_bgr.dtype != torch.uint8 or image_bgr.ndim != 4 or image_bgr.shape[1] != 3:
            raise ValueError("image_bgr must be uint8 [B,3,H,W]")
        if condition.ndim != 4 or condition.shape[1] != self.condition_channels:
            raise ValueError("condition must be float [B,3,H,W]")
        features, shape = self._encode(image_bgr, condition, device)
        patch_h, patch_w, original_h, original_w = shape
        evidence = lidar_pyramid_evidence(condition)
        if zero_evidence is False or (zero_evidence is None and not self.evidence_enabled):
            evidence = torch.zeros_like(evidence)
        logits, fusion_deltas = self._decode_direct(features, evidence, patch_h, patch_w)
        disparity = F.relu(logits)
        disparity = F.interpolate(disparity, (original_h, original_w), mode="bilinear", align_corners=True)
        if not return_aux:
            return disparity
        fusion_energy = torch.stack([delta.square().mean() for delta in fusion_deltas]).mean()
        return disparity, {"fusion_energy": fusion_energy}
