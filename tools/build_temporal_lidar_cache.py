#!/usr/bin/env python3
"""Rebuild target-excluded temporal LiDAR supervision in camera coordinates.

The source cache supplies only the frozen RGB/MDE inputs and the chosen frame
list.  Neighbor LiDAR is reprojected from the original sequences with the
correct KITTI camera-pose convention.  DBH labels, measurement rows, and trunk
masks are neither read nor written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prior_lidar.temporal import (
    TEMPORAL_CACHE_SCHEMA,
    aggregate_neighbor_depth,
    overlap_diagnostics,
)
from prior_lidar.alignment import AlignmentConfig
from prior_lidar.conditions import build_conditions


REQUIRED_SOURCE_ARRAYS = {"img_bgr", "prior_m", "prior_valid", "disp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-cache",
        type=Path,
        default=ROOT / "错误尝试/dataset/temporal_lidar_cache_s3s6",
        help="Cache used only for its frozen image, MDE prediction, and frame selection.",
    )
    parser.add_argument("--out", type=Path, default=ROOT / "data/temporal_lidar_cache_m_5_6")
    parser.add_argument("--sequences", default="05,06")
    parser.add_argument("--min-depth-m", type=float, default=0.1)
    parser.add_argument("--max-depth-m", type=float, default=120.0)
    parser.add_argument("--frozen-mde-size", choices=("vits", "vitb", "vitl"), default="vitl")
    parser.add_argument("--alignment-k", type=int, default=5)
    parser.add_argument("--alignment-distance-power", type=float, default=1.0)
    parser.add_argument("--robust-alignment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-knn-support-points", type=int, default=20000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    sequences = {part.strip().zfill(2) for part in args.sequences.split(",") if part.strip()}
    if not sequences:
        raise ValueError("at least one sequence is required")
    if args.min_depth_m <= 0 or args.max_depth_m <= args.min_depth_m:
        raise ValueError("invalid metric depth range")
    if args.alignment_k < 2 or args.alignment_distance_power < 0 or not np.isfinite(args.alignment_distance_power):
        raise ValueError("invalid alignment configuration")
    paths = sorted(args.source_cache.glob("seq*_*.npz"))
    if not paths:
        raise FileNotFoundError(f"no source samples in {args.source_cache}")
    if args.out.exists() and any(args.out.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output directory is not empty: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    alignment = AlignmentConfig(
        k=args.alignment_k,
        robust=args.robust_alignment,
        irls_steps=3 if args.robust_alignment else 0,
        max_prior_points=0,
        max_knn_support_points=args.max_knn_support_points,
        distance_power=args.alignment_distance_power,
    )

    records = []
    for index, source_path in enumerate(paths, start=1):
        metadata_path = source_path.with_suffix(".json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        sequence = str(metadata.get("sequence", "")).zfill(2)
        if sequence not in sequences:
            continue
        frame = int(metadata.get("frame", -1))
        neighbors = [int(value) for value in metadata.get("neighbors", [])]
        if metadata.get("target_excluded") is not True or frame in neighbors:
            raise ValueError(f"target-frame leakage in {source_path.name}")
        if not neighbors:
            raise ValueError(f"no temporal neighbors in {source_path.name}")
        with np.load(source_path, allow_pickle=False) as source:
            missing = REQUIRED_SOURCE_ARRAYS - set(source.files)
            if missing:
                raise KeyError(f"{source_path.name} is missing {sorted(missing)}")
            arrays = {key: source[key] for key in REQUIRED_SOURCE_ARRAYS}
        image = arrays["img_bgr"]
        prior = arrays["prior_m"].astype(np.float32)
        prior_valid = arrays["prior_valid"] > 0
        if image.shape[:2] != prior.shape or arrays["disp"].shape != prior.shape:
            raise ValueError(f"shape mismatch in {source_path.name}")

        if not (ROOT / "data/sequences" / sequence / "velodyne" / f"{frame:06d}.bin").is_file():
            # The source cache may schedule frames that the recorded sequence does
            # not contain; those samples simply cannot be built.
            print(f"[skip] {source_path.stem}: frame {frame} is not recorded in sequence {sequence}", flush=True)
            continue
        gt, gt_valid = aggregate_neighbor_depth(
            ROOT / "data/sequences" / sequence,
            frame,
            neighbors,
            prior.shape,
            min_depth_m=args.min_depth_m,
            max_depth_m=args.max_depth_m,
        )
        condition, affine, alignment_diagnostics = build_conditions(
            arrays["disp"].astype(np.float32), prior, prior_valid, alignment
        )
        diagnostics = overlap_diagnostics(gt, gt_valid, prior, prior_valid)
        destination = args.out / source_path.name
        temporary = destination.with_suffix(".npz.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                img_bgr=image,
                prior_m=prior,
                prior_valid=prior_valid.astype(np.uint8),
                disp=arrays["disp"].astype(np.float32),
                gt_m=gt,
                gt_valid=gt_valid.astype(np.uint8),
                condition=condition,
                affine=np.asarray(affine, dtype=np.float32),
            )
        os.replace(temporary, destination)
        output_metadata = {
            "schema": TEMPORAL_CACHE_SCHEMA,
            "sequence": sequence,
            "frame": frame,
            "neighbors": neighbors,
            "target_excluded": True,
            "pose_convention": "world_from_camera; target_camera_from_source_lidar=inv(Pt)@Ps@Tr",
            "depth_unit": "m",
            "raw_valid_pixels": int(prior_valid.sum()),
            "temporal_valid_pixels": int(gt_valid.sum()),
            "overlap": diagnostics,
            "alignment": alignment.to_dict(),
            "alignment_diagnostics": alignment_diagnostics,
            "frozen_mde_size": metadata.get("frozen_mde_size", metadata.get("mde_size", args.frozen_mde_size)),
            "source_cache_file": source_path.name,
            "source_cache_sha256": sha256_file(source_path),
            "dbh_labels_read": False,
            "trunk_mask_read": False,
        }
        write_json_atomic(destination.with_suffix(".json"), output_metadata)
        records.append({"file": destination.name, **output_metadata})
        print(
            f"[{len(records):02d}] {destination.stem}: gt={int(gt_valid.sum())} "
            f"overlap={diagnostics['overlap_points']} "
            f"median_abs={diagnostics['median_abs_error_m']}",
            flush=True,
        )

    if not records:
        raise ValueError(f"no samples matched sequences {sorted(sequences)}")
    manifest = {
        "schema": TEMPORAL_CACHE_SCHEMA,
        "depth_unit": "m",
        "sample_count": len(records),
        "sequences": sorted({record["sequence"] for record in records}),
        "target_frame_excluded": True,
        "dbh_labels_read": False,
        "trunk_masks_read": False,
        "frozen_mde_size": args.frozen_mde_size,
        "source_cache": str(args.source_cache),
        "alignment": alignment.to_dict(),
        "records": records,
    }
    write_json_atomic(args.out / "manifest.json", manifest)
    print(json.dumps({key: manifest[key] for key in ("schema", "sample_count", "sequences")}, indent=2))


if __name__ == "__main__":
    main()
