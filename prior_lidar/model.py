"""Thin wrapper around the official PriorDA conditioned DPT model."""
from __future__ import annotations

import hashlib
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


class BoundedResidualAdapter(nn.Module):
    """Small zero-initialized log-depth adapter placed after official PriorDA.

    The adapter is deliberately residual and bounded.  With zero weights the
    complete model is bit-for-bit the released PriorDA model; during training
    it can learn a modest domain correction without replacing the DPT decoder
    or changing the single-frame interface.
    """

    def __init__(self, condition_channels: int, max_log_correction: float = 0.05) -> None:
        super().__init__()
        if max_log_correction <= 0 or not torch.isfinite(torch.tensor(max_log_correction)):
            raise ValueError("max_log_correction must be positive and finite")
        self.max_log_correction = float(max_log_correction)
        # RGB, three PriorDA conditions, and the official normalized disparity.
        input_channels = 3 + condition_channels + 1
        self.body = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
        )
        self.output = nn.Conv2d(32, 1, kernel_size=1)
        # Zero initialization is the key compatibility guarantee.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        image_bgr: torch.Tensor,
        condition: torch.Tensor,
        base_disparity: torch.Tensor,
    ) -> torch.Tensor:
        height, width = base_disparity.shape[-2:]
        image = image_bgr.float().div(255.0)
        # Conditions are already normalized by the coarse stage, but their
        # reciprocal representation can contain very large values near zero.
        # A per-channel finite peak normalization keeps the tiny adapter
        # numerically stable without changing the official condition tensor.
        safe_condition = torch.nan_to_num(condition.float(), nan=0.0, posinf=0.0, neginf=0.0)
        peak = safe_condition.amax(dim=(-2, -1), keepdim=True).clamp_min(1.0)
        safe_condition = (safe_condition / peak).clamp(0.0, 1.0)
        features = torch.cat((image, safe_condition, base_disparity.float().clamp(0.0, 1000.0)), dim=1)
        # Work at quarter resolution; the full-resolution interpolation keeps
        # the deployment cost close to the unmodified model.
        reduced_h = max(1, (height + 3) // 4)
        reduced_w = max(1, (width + 3) // 4)
        features = F.interpolate(features, (reduced_h, reduced_w), mode="bilinear", align_corners=False)
        correction = self.body(features)
        correction = self.output(correction)
        correction = F.interpolate(correction, (height, width), mode="bilinear", align_corners=False)
        return self.max_log_correction * torch.tanh(correction)


class PriorLidarModel(nn.Module):
    """Official PriorDA fine-stage topology with an optional residual adapter."""

    variant = "priorda_v1_1_lidar"
    condition_channels = 3

    def __init__(
        self,
        size: str = "vitb",
        residual_adapter: bool = False,
        max_log_correction: float = 0.05,
    ) -> None:
        super().__init__()
        if size not in {"vits", "vitb"}:
            raise ValueError("released PriorDA fine-stage sizes are vits and vitb")
        self.size = size
        self.residual_adapter_enabled = bool(residual_adapter)
        self.max_log_correction = float(max_log_correction)
        self.network = build_backbone(depth_size=size, encoder_cond_dim=self.condition_channels)
        self.network.construct_aux_layers()
        self.residual_adapter = (
            BoundedResidualAdapter(self.condition_channels, max_log_correction)
            if self.residual_adapter_enabled else None
        )

    @property
    def model_variant(self) -> str:
        return f"{self.variant}_residual" if self.residual_adapter_enabled else self.variant

    def forward(
        self,
        image_bgr: torch.Tensor,
        condition: torch.Tensor,
        device: str,
        return_base: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if image_bgr.dtype != torch.uint8 or image_bgr.ndim != 4 or image_bgr.shape[1] != 3:
            raise ValueError("image_bgr must be uint8 [B,3,H,W]")
        if condition.ndim != 4 or condition.shape[1] != self.condition_channels:
            raise ValueError("condition must be float [B,3,H,W]")
        base = self.network(image_bgr, 518, condition=condition, device=device)
        if self.residual_adapter is None:
            return (base, base) if return_base else base
        correction = self.residual_adapter(image_bgr, condition, base)
        # A positive log-depth correction corresponds to lower disparity.
        output = torch.where(
            base > 1e-6,
            base * torch.exp(-correction),
            base,
        )
        return (output, base) if return_base else output

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

    def configure_trainable(
        self,
        encoder_blocks: int = -1,
        decoder_mode: str = "full",
    ) -> dict[str, int]:
        """Configure the paper's encoder/decoder learning-rate groups.

        ``encoder_blocks=-1`` trains the full encoder as in the paper.  Zero
        freezes the RGB encoder except its condition projection; a positive
        value trains that many tail transformer blocks.
        """

        if decoder_mode not in {"full", "adapter", "head", "residual"}:
            raise ValueError("decoder_mode must be one of: full, adapter, head, residual")
        for parameter in self.network.parameters():
            parameter.requires_grad = False
        if decoder_mode in {"full", "head"}:
            for parameter in self.network.depth_head.parameters():
                parameter.requires_grad = True
        if decoder_mode != "residual":
            for parameter in self.network.pretrained.patch_embed.alpha_proj.parameters():
                parameter.requires_grad = True

        if decoder_mode == "residual":
            if self.residual_adapter is None:
                raise ValueError("decoder_mode='residual' requires residual_adapter=True")
            for parameter in self.residual_adapter.parameters():
                parameter.requires_grad = True

        # The residual variant is intentionally an adapter-only update.  This
        # keeps its zero-initialized path a faithful PriorDA checkpoint and
        # prevents a caller's generic encoder setting from silently turning it
        # into a large domain fine-tune.
        if decoder_mode != "residual":
            blocks = self.network.pretrained.blocks
            if encoder_blocks < 0:
                for parameter in self.network.pretrained.parameters():
                    parameter.requires_grad = True
            elif encoder_blocks > 0:
                for block in blocks[-encoder_blocks:]:
                    for parameter in block.parameters():
                        parameter.requires_grad = True
                for parameter in self.network.pretrained.norm.parameters():
                    parameter.requires_grad = True

        decoder = 0
        encoder = 0
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith("network.pretrained.") and "patch_embed.alpha_proj" not in name:
                encoder += parameter.numel()
            else:
                decoder += parameter.numel()
        return {"encoder": encoder, "decoder_and_condition": decoder}

    def parameter_groups(self) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
        encoder: list[nn.Parameter] = []
        decoder: list[nn.Parameter] = []
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith("network.pretrained.") and "patch_embed.alpha_proj" not in name:
                encoder.append(parameter)
            else:
                decoder.append(parameter)
        return encoder, decoder

    def load_training_checkpoint(self, checkpoint: Path, strict: bool = True) -> dict:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = payload.get("model", payload)
        missing, unexpected = self.load_state_dict(state, strict=strict)
        if strict and (missing or unexpected):
            raise RuntimeError(f"checkpoint mismatch: missing={missing}, unexpected={unexpected}")
        return payload
