#!/usr/bin/env python3
"""Train the direct RGB--LiDAR disparity fusion decoder on temporal data."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prior_lidar.alignment import AlignmentConfig
from prior_lidar.conditions import decode_metric_depth_torch
from prior_lidar.data import read_temporal_manifest
from prior_lidar.direct_fusion import DirectFusionDisparityModel
from prior_lidar.model import PriorLidarModel
from prior_lidar.temporal_reliability import ReliableTemporalPriorDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / "data/temporal_lidar_cache_m")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/prior_lidar_direct_fusion_s05_v1")
    parser.add_argument("--init", type=Path, default=ROOT / "checkpoints/prior_depth_anything_vitb_1_1.pth")
    parser.add_argument("--size", choices=("vits", "vitb"), default="vitb")
    parser.add_argument("--train-sequences", default="03")
    parser.add_argument("--val-sequences", default="06")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--fusion-channels", type=int, default=48)
    parser.add_argument("--cross-fusion", action=argparse.BooleanOptionalAction, default=True,
                        help="ablation: disable the RGB-LiDAR cross-attention branch entirely")
    parser.add_argument("--evidence", action=argparse.BooleanOptionalAction, default=True,
                        help="ablation: keep the branch but feed it zero LiDAR evidence")
    parser.add_argument("--fusion-scales", choices=("all", "finest", "coarsest"), default="all",
                        help="ablation: how many DPT scales receive a fusion injection")
    parser.add_argument("--fusion-lr", type=float, default=5e-5)
    parser.add_argument("--decoder-lr", type=float, default=2e-6)
    parser.add_argument("--train-decoder", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--teacher-weight", type=float, default=0.15)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--keep-aspect-ratio", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--alignment-k", type=int, default=5)
    parser.add_argument("--alignment-distance-power", type=float, default=1.0)
    parser.add_argument("--max-knn-support-points", type=int, default=20000)
    parser.add_argument("--robust-alignment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--frozen-mde-size", choices=("vits", "vitb", "vitl"), default="vitl")
    parser.add_argument("--log-huber-beta", type=float, default=0.1)
    parser.add_argument("--anchor-weight", type=float, default=0.75)
    parser.add_argument("--reliability-neighbors", type=int, default=4)
    parser.add_argument("--reliability-spatial-scale-px", type=float, default=24.0)
    parser.add_argument("--reliability-max-distance-px", type=float, default=96.0)
    parser.add_argument("--reliability-log-depth-scale", type=float, default=0.15)
    parser.add_argument("--reliability-minimum-weight", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    return parser.parse_args()


def split_csv(value: str) -> set[str]:
    return {part.strip().zfill(2) for part in value.split(",") if part.strip()}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def weighted_log_loss(prediction: torch.Tensor, target: torch.Tensor, weight: torch.Tensor, beta: float) -> torch.Tensor:
    keep = torch.isfinite(prediction) & torch.isfinite(target) & torch.isfinite(weight)
    keep &= (prediction > 1e-6) & (target > 1e-6) & (weight > 0)
    if not keep.any():
        return prediction.sum() * 0.0
    delta = torch.log(prediction[keep]) - torch.log(target[keep])
    loss = F.smooth_l1_loss(delta, torch.zeros_like(delta), beta=beta, reduction="none")
    selected_weight = weight[keep].float()
    return (loss * selected_weight).sum() / selected_weight.sum().clamp_min(1e-6)


class WeightedMetrics:
    def __init__(self) -> None:
        self.weight = self.absrel = self.squared = self.log = self.log_squared = 0.0

    def update(self, prediction: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> None:
        keep = torch.isfinite(prediction) & torch.isfinite(target) & torch.isfinite(weight)
        keep &= (prediction > 0) & (target > 0) & (weight > 0)
        if not keep.any():
            return
        pred, truth, selected = prediction[keep].double(), target[keep].double(), weight[keep].double()
        delta = pred - truth
        log_delta = torch.log(pred) - torch.log(truth)
        self.weight += float(selected.sum())
        self.absrel += float((selected * delta.abs() / truth).sum())
        self.squared += float((selected * delta.square()).sum())
        self.log += float((selected * log_delta).sum())
        self.log_squared += float((selected * log_delta.square()).sum())

    def result(self) -> dict:
        if self.weight <= 0:
            return {"weight_sum": 0.0, "abs_rel": None, "rmse_m": None, "silog": None}
        mean_log = self.log / self.weight
        return {
            "weight_sum": self.weight,
            "abs_rel": self.absrel / self.weight,
            "rmse_m": math.sqrt(self.squared / self.weight),
            "silog": math.sqrt(max(0.0, self.log_squared / self.weight - mean_log * mean_log)),
        }


def autocast_context(args: argparse.Namespace):
    return torch.autocast(
        device_type="cuda", dtype=torch.bfloat16,
        enabled=args.precision == "bf16" and args.device.startswith("cuda"),
    )


@torch.inference_mode()
def validate(model: DirectFusionDisparityModel, loader: DataLoader, args: argparse.Namespace) -> dict:
    model.eval()
    trusted, all_temporal, anchors = WeightedMetrics(), WeightedMetrics(), WeightedMetrics()
    fusion_energy = 0.0
    batches = 0
    for batch in loader:
        image = batch["image"].to(args.device, non_blocking=True)
        condition = batch["condition"].to(args.device, non_blocking=True)
        affine = batch["affine"].to(args.device, non_blocking=True)
        target = batch["target_m"].to(args.device, non_blocking=True)
        temporal_weight = batch["temporal_weight"].to(args.device, non_blocking=True)
        valid = batch["valid"].to(args.device, non_blocking=True).float()
        prior = batch["prior_m"].to(args.device, non_blocking=True)
        prior_valid = batch["prior_valid"].to(args.device, non_blocking=True).float()
        with autocast_context(args):
            disparity, aux = model(image, condition, args.device, return_aux=True)
            prediction = decode_metric_depth_torch(disparity, affine)
        trusted.update(prediction, target, temporal_weight)
        all_temporal.update(prediction, target, valid)
        anchors.update(prediction, prior, prior_valid)
        fusion_energy += float(aux["fusion_energy"])
        batches += 1
    trusted_result, anchor_result = trusted.result(), anchors.result()
    if trusted_result["abs_rel"] is None or anchor_result["abs_rel"] is None:
        raise ValueError("validation has no reliable temporal or anchor support")
    return {
        "selection_score": trusted_result["abs_rel"] + trusted_result["silog"] + 0.2 * anchor_result["abs_rel"],
        "trusted_temporal": trusted_result,
        "all_temporal": all_temporal.result(),
        "input_lidar_anchors": anchor_result,
        "mean_fusion_energy": fusion_energy / max(1, batches),
    }


def checkpoint_payload(model, epoch, validation, args, alignment, initialization, trainable):
    return {
        "model": model.state_dict(), "epoch": epoch, "val": validation,
        "model_config": {
            "variant": model.model_variant, "size": model.size, "fusion_channels": args.fusion_channels,
            "cross_fusion_enabled": bool(model.cross_fusion_enabled),
            "evidence_enabled": bool(model.evidence_enabled),
            "fusion_scales": model.fusion_scales,
            "condition_channels": 3,
            "condition_order": ["sparse_mask", "global_aligned", "knn_aligned"],
            "learned_evidence_order": ["sparse_mask", "global_aligned", "support_5", "support_17", "support_65"],
            "alignment": alignment.to_dict(), "frozen_mde_size": args.frozen_mde_size,
            "direct_disparity_output": True, "output_residual": False, "single_frame": True,
        },
        "training_config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "initialization": initialization, "trainable_parameters": trainable,
    }


def main() -> None:
    args = parse_args()
    if args.run_dir.exists():
        raise FileExistsError(f"run directory already exists: {args.run_dir}")
    if args.epochs < 1 or args.repeats < 1:
        raise ValueError("epochs and repeats must be positive")
    if args.teacher_weight < 0:
        raise ValueError("teacher weight must be non-negative")
    train_sequences, val_sequences = split_csv(args.train_sequences), split_csv(args.val_sequences)
    if not train_sequences or not val_sequences or train_sequences & val_sequences:
        raise ValueError("training and validation sequences must be non-empty and disjoint")
    manifest = read_temporal_manifest(args.cache, expected_frozen_mde_size=args.frozen_mde_size, require_frozen_mde=True)
    seed_everything(args.seed)
    args.run_dir.mkdir(parents=True)
    alignment = AlignmentConfig(
        k=args.alignment_k, robust=args.robust_alignment, irls_steps=3 if args.robust_alignment else 0,
        max_prior_points=0, max_knn_support_points=args.max_knn_support_points,
        distance_power=args.alignment_distance_power,
    )
    reliability_kwargs = {
        "reliability_neighbors": args.reliability_neighbors,
        "reliability_spatial_scale_px": args.reliability_spatial_scale_px,
        "reliability_max_distance_px": args.reliability_max_distance_px,
        "reliability_log_depth_scale": args.reliability_log_depth_scale,
        "reliability_minimum_weight": args.reliability_minimum_weight,
    }
    train_set = ReliableTemporalPriorDataset(
        args.cache, train_sequences, alignment, training=True, seed=args.seed, repeats=args.repeats,
        crop_size=518, frozen_mde_size=args.frozen_mde_size, keep_aspect_ratio=args.keep_aspect_ratio,
        **reliability_kwargs,
    )
    val_set = ReliableTemporalPriorDataset(
        args.cache, val_sequences, alignment, training=False, seed=args.seed + 1, repeats=1,
        crop_size=None, frozen_mde_size=args.frozen_mde_size, keep_aspect_ratio=args.keep_aspect_ratio,
        **reliability_kwargs,
    )
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=args.workers,
                              generator=torch.Generator().manual_seed(args.seed), persistent_workers=False)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=args.workers, persistent_workers=False)
    model = DirectFusionDisparityModel(args.size, args.fusion_channels,
                                       cross_fusion=args.cross_fusion, evidence=args.evidence,
                                       fusion_scales=args.fusion_scales)
    initialization = model.load_official(args.init, strict=True)
    trainable = model.configure_trainable(args.train_decoder)
    model.to(args.device)
    teacher = None
    if args.teacher_weight > 0:
        teacher = PriorLidarModel(args.size).to(args.device).eval()
        teacher_info = teacher.load_official(args.init, strict=True)
        if teacher_info["missing"] or teacher_info["unexpected"]:
            raise RuntimeError(f"official teacher checkpoint did not load strictly: {teacher_info}")
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
    parameter_groups = model.parameter_groups(args.fusion_lr, args.decoder_lr, args.weight_decay)
    optimizer = torch.optim.AdamW(parameter_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    provenance = {
        "cache_manifest": {key: manifest[key] for key in ("schema", "depth_unit", "sample_count", "sequences", "target_frame_excluded", "dbh_labels_read", "trunk_masks_read")},
        "training_sequences": sorted(train_sequences), "validation_sequences": sorted(val_sequences),
        "selection_metric": "S6 reprojection-compatible temporal AbsRel + SILog + raw-anchor AbsRel",
        "loss_support": "reprojection-compatible target-excluded temporal pixels plus current-frame raw LiDAR anchors",
        "architecture": "frozen conditioned DINO, four bidirectional RGB-LiDAR cross-fusion scales, direct disparity DPT decoder",
        "direct_disparity_output": True, "output_residual": False, "learned_branch_uses_knn": False,
        "teacher": "training-only frozen official PriorDA consistency" if teacher is not None else "disabled",
        "encoder_frozen": True, "dbh_labels_used_for_training_or_selection": False,
        "frozen_mde_size": args.frozen_mde_size, "reliability": reliability_kwargs,
    }
    atomic_json(args.run_dir / "provenance.json", provenance)
    initial = validate(model, val_loader, args)
    history = [{"epoch": 0, "train": None, "seconds": 0.0, "val": initial}]
    initial_payload = checkpoint_payload(model, 0, initial, args, alignment, initialization, trainable)
    torch.save(initial_payload, args.run_dir / "initial.pt")
    torch.save(initial_payload, args.run_dir / "best.pt")
    best_score = initial["selection_score"]
    atomic_json(args.run_dir / "history.json", history)
    print(f"epoch=000 val={json.dumps(initial, sort_keys=True)}", flush=True)
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        train_set.set_epoch(epoch)
        model.train()
        losses = []
        for batch in train_loader:
            image = batch["image"].to(args.device, non_blocking=True)
            condition = batch["condition"].to(args.device, non_blocking=True)
            affine = batch["affine"].to(args.device, non_blocking=True)
            target = batch["target_m"].to(args.device, non_blocking=True)
            temporal_weight = batch["temporal_weight"].to(args.device, non_blocking=True)
            prior, prior_valid = batch["prior_m"].to(args.device, non_blocking=True), batch["prior_valid"].to(args.device, non_blocking=True).float()
            valid = batch["valid"].to(args.device, non_blocking=True).float()
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(args):
                disparity, aux = model(image, condition, args.device, return_aux=True)
                prediction = decode_metric_depth_torch(disparity, affine)
                temporal_loss = weighted_log_loss(prediction, target, temporal_weight, args.log_huber_beta)
                anchor_loss = weighted_log_loss(prediction, prior, prior_valid, args.log_huber_beta)
                teacher_loss = prediction.sum() * 0.0
                if teacher is not None:
                    with torch.inference_mode():
                        teacher_disp = teacher(image, condition, args.device)
                        teacher_depth = decode_metric_depth_torch(teacher_disp, affine)
                    teacher_loss = weighted_log_loss(prediction, teacher_depth, valid, args.log_huber_beta)
                loss = temporal_loss + args.anchor_weight * anchor_loss + args.teacher_weight * teacher_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([parameter for parameter in model.parameters() if parameter.requires_grad], args.grad_clip)
            optimizer.step()
            losses.append([float(loss.detach()), float(temporal_loss.detach()), float(anchor_loss.detach()), float(teacher_loss.detach()), float(aux["fusion_energy"].detach())])
        scheduler.step()
        validation = validate(model, val_loader, args)
        means = np.asarray(losses, dtype=np.float64).mean(axis=0)
        record = {"epoch": epoch, "train": {"total": float(means[0]), "temporal_compatible_log": float(means[1]), "anchor_log": float(means[2]), "teacher_log": float(means[3]), "fusion_energy": float(means[4])}, "seconds": round(time.time() - started, 2), "learning_rates": [group["lr"] for group in optimizer.param_groups], "val": validation}
        history.append(record)
        payload = checkpoint_payload(model, epoch, validation, args, alignment, initialization, trainable)
        torch.save(payload, args.run_dir / "last.pt")
        if validation["selection_score"] < best_score:
            best_score = validation["selection_score"]
            torch.save(payload, args.run_dir / "best.pt")
        atomic_json(args.run_dir / "history.json", history)
        print(f"epoch={epoch:03d} train={json.dumps(record['train'], sort_keys=True)} val={json.dumps(validation, sort_keys=True)} seconds={record['seconds']}", flush=True)


if __name__ == "__main__":
    main()
