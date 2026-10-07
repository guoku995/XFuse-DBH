#!/usr/bin/env python3
"""Evaluate direct RGB--LiDAR disparity fusion."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prior_lidar.alignment import AlignmentConfig
from prior_lidar.conditions import build_conditions, decode_metric_depth
from prior_lidar.direct_fusion import DirectFusionDisparityModel, sha256_file
from tools.eval_prior_lidar import (
    build_fixed_geometry, depth_metrics, load_samples, measure_dbh,
    model_from_official, read_calibration, read_eval_manifest, read_tree_labels,
    sequence_from_name, summarize_dbh, summarize_depth,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / "data/dbh_cache_tree50")
    parser.add_argument("--labels", type=Path, default=ROOT / "data/sequences/tree_50.txt")
    parser.add_argument("--prior-checkpoint", type=Path, default=ROOT / "checkpoints/prior_depth_anything_vitb_1_1.pth")
    parser.add_argument("--trained-checkpoint", type=Path, default=ROOT / "runs/prior_lidar_direct_fusion_s05_v1/best.pt")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/eval_prior_lidar_drect_fusion_s05_v1.json")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--frozen-mde-size", choices=("vits", "vitb", "vitl"), default="vitl")
    parser.add_argument("--alignment-k", type=int, default=5)
    parser.add_argument("--alignment-distance-power", type=float, default=1.0)
    parser.add_argument("--max-knn-support-points", type=int, default=20000)
    parser.add_argument("--robust-alignment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--expect-samples", type=int, default=0)
    parser.add_argument("--require-labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allowed-sequences", default="")
    parser.add_argument("--filter-by-labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--zero-evidence", action=argparse.BooleanOptionalAction, default=False,
                        help="ablation: zero the LiDAR evidence at inference (keeps the trained weights)")
    parser.add_argument("--no-cross-fusion", action=argparse.BooleanOptionalAction, default=False,
                        help="ablation: skip the cross-attention branch at inference (keeps the trained weights)")
    parser.add_argument("--seed", type=int, default=20260910)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = read_tree_labels(args.labels)
    allowed = {part.strip().zfill(2) for part in str(getattr(args, "allowed_sequences", "")).split(",") if part.strip()}
    manifest = read_eval_manifest(args.cache, args.frozen_mde_size)
    samples = load_samples(args.cache, labels, args.frozen_mde_size,
                           require_labels=bool(getattr(args, "require_labels", True)),
                           expect_samples=int(getattr(args, "expect_samples", 0)),
                           allowed_sequences=allowed or None,
                           skip_unlabelled=bool(getattr(args, "filter_by_labels", True)))
    alignment = AlignmentConfig(
        k=args.alignment_k, robust=args.robust_alignment, irls_steps=3 if args.robust_alignment else 0,
        max_prior_points=0, max_knn_support_points=args.max_knn_support_points,
        distance_power=args.alignment_distance_power,
    )
    payload = torch.load(args.trained_checkpoint, map_location="cpu", weights_only=False)
    config = payload["model_config"]
    model = DirectFusionDisparityModel(
        str(config["size"]), int(config.get("fusion_channels", 48)),
        cross_fusion=bool(config.get("cross_fusion_enabled", True)),
        evidence=bool(config.get("evidence_enabled", True)),
        fusion_scales=str(config.get("fusion_scales", "all")),
    )
    model.load_state_dict(payload["model"], strict=True)
    model = model.to(args.device).eval()
    official = model_from_official(args.prior_checkpoint, args.device, size=str(config["size"]))
    rows, depth_rows = [], []
    for index, (_path, arrays, metadata) in enumerate(samples, start=1):
        image = arrays["img_bgr"]
        raw, raw_valid = arrays["prior_m"].astype(np.float32), arrays["prior_valid"] > 0
        condition, affine, alignment_debug = build_conditions(arrays["disp"].astype(np.float32), raw, raw_valid, alignment)
        sequence = sequence_from_name(metadata["name"])
        intrinsics, _transform, _distortion = read_calibration(sequence)
        geometry = build_fixed_geometry(raw, raw_valid, arrays["mask"] > 0, float(intrinsics[0, 0]), float(intrinsics[1, 1]))
        image_tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)[None].to(args.device)
        condition_tensor = torch.from_numpy(np.ascontiguousarray(condition))[None].to(args.device)
        with torch.inference_mode():
            if getattr(args, "zero_evidence", False) or getattr(args, "no_cross_fusion", False):
                model.cross_fusion_enabled = not getattr(args, "no_cross_fusion", False)
                model.evidence_enabled = not getattr(args, "zero_evidence", False)
            trained_depth = decode_metric_depth(model(image_tensor, condition_tensor, args.device).squeeze().float().cpu().numpy(), affine)
            prior_depth = decode_metric_depth(official(image_tensor, condition_tensor, args.device).squeeze().float().cpu().numpy(), affine)
        depths = {"raw": raw, "coarse_global": decode_metric_depth(condition[1], affine), "coarse_knn": decode_metric_depth(condition[2], affine), "prior": prior_depth, "trained": trained_depth}
        target_dbh = labels[(str(metadata["name"]).strip(), str(metadata["description"]).strip())]
        entry = {"index": int(metadata["index"]), "name": metadata["name"], "description": metadata["description"], "sequence": sequence, "gt_dbh_cm": float(target_dbh), "geometry": geometry, "alignment": alignment_debug, "methods": {}}
        for name, depth in depths.items():
            prediction, detail = measure_dbh(depth, geometry)
            entry["methods"][name] = {"status": "ok" if prediction is not None else "no_valid_slice_depth", "pred_dbh_cm": prediction, "error_cm": None if prediction is None else float(prediction - target_dbh), "detail": detail}
        rows.append(entry)
        depth_rows.append({"index": int(metadata["index"]), "name": metadata["name"], "sequence": sequence, "depth_metrics": {name: depth_metrics(depth, arrays["gt_m"], arrays["gt_valid"] > 0) for name, depth in depths.items()}})
        print(f"[{index:02d}] {metadata['name']} prior={entry['methods']['prior']['pred_dbh_cm']:.2f} trained={entry['methods']['trained']['pred_dbh_cm']:.2f}", flush=True)
    rng = np.random.default_rng(args.seed)
    names = ("raw", "coarse_global", "coarse_knn", "prior", "trained")
    dbh_summary = {name: summarize_dbh(rows, name, args.bootstrap, rng) for name in names}
    depth_summary = {name: summarize_depth(depth_rows, name, args.bootstrap, rng) for name in names}
    common = [row for row in rows if row["methods"]["prior"]["status"] == "ok" and row["methods"]["trained"]["status"] == "ok"]
    prior_abs = np.asarray([abs(row["methods"]["prior"]["error_cm"]) for row in common])
    trained_abs = np.asarray([abs(row["methods"]["trained"]["error_cm"]) for row in common])
    result = {
        "protocol": {"name": "prior_lidar_transparent_v1", "labels": str(args.labels), "cache": str(args.cache),
                     "test_sequences": sorted({str(m.get("sequence") or sequence_from_name(str(m["name"]))) for _p, _a, m in samples}), "sample_count": len(samples), "target_height_m": 1.3, "slice_half_height_m": 0.05, "base_quantile": 0.25, "geometry_source": "raw LiDAR prior + fixed P2 calibration + shared cached mask", "model_used_for_geometry": False, "raw_writeback": False, "test_calibration": False, "model_fallback": False, "alignment": alignment.to_dict(), "frozen_mde_size": args.frozen_mde_size, "evaluation_manifest": manifest},
        "checkpoints": {"prior": str(args.prior_checkpoint), "prior_sha256": sha256_file(args.prior_checkpoint), "trained": str(args.trained_checkpoint), "trained_sha256": sha256_file(args.trained_checkpoint), "trained_info": {"model_config": config, "training_config": payload.get("training_config"), "epoch": payload.get("epoch"), "val": payload.get("val")}},
        "dbh_summary": dbh_summary, "depth_summary": depth_summary,
        "paired_dbh": {"trained_minus_prior": {"n": len(common), "mae_delta_cm": float(trained_abs.mean() - prior_abs.mean()), "improved_count": int((trained_abs < prior_abs).sum()), "worsened_count": int((trained_abs > prior_abs).sum()), "unchanged_count": int((trained_abs == prior_abs).sum())}},
        "rows": rows, "depth_rows": depth_rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"dbh_summary": dbh_summary, "depth_summary": depth_summary, "paired_dbh": result["paired_dbh"]}, indent=2))


if __name__ == "__main__":
    main()
