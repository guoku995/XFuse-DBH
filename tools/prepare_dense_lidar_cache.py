#!/usr/bin/env python3
"""Strip the 40-tree cache to DBH-independent dense-depth training inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prior_lidar.alignment import AlignmentConfig
from prior_lidar.conditions import build_conditions
from prior_lidar.data import DEPTH_CACHE_SCHEMA


REQUIRED = {"img_bgr", "prior_m", "prior_valid", "disp", "gt_m", "gt_valid"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sequence_and_frame(name: str) -> tuple[str, int]:
    match = re.fullmatch(r"S(\d+)-(\d+)", name.strip())
    if not match:
        raise ValueError(f"invalid sample name: {name!r}")
    return f"{int(match.group(1)):02d}", int(match.group(2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/lidar_cache")
    parser.add_argument("--out", type=Path, default=ROOT / "data/lidar_depth_cache_m")
    parser.add_argument("--frozen-mde-size", choices=("vits", "vitb", "vitl"), default="vitl")
    parser.add_argument("--alignment-k", type=int, default=5)
    parser.add_argument("--alignment-distance-power", type=float, default=1.0)
    parser.add_argument("--robust-alignment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-knn-support-points", type=int, default=20000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.alignment_k < 2 or args.alignment_distance_power < 0 or not np.isfinite(args.alignment_distance_power):
        raise ValueError("invalid alignment configuration")
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
    for source_path in sorted(args.source.glob("idx*.npz")):
        metadata_path = source_path.with_suffix(".json")
        if not metadata_path.is_file():
            raise FileNotFoundError(f"metadata missing for {source_path.name}")
        source_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        sequence, frame = sequence_and_frame(str(source_metadata["name"]))
        with np.load(source_path, allow_pickle=False) as source:
            missing = REQUIRED - set(source.files)
            if missing:
                raise KeyError(f"{source_path.name} is missing {sorted(missing)}")
            arrays = {key: source[key] for key in REQUIRED}
        image = arrays["img_bgr"]
        prior = arrays["prior_m"].astype(np.float32)
        prior_valid = arrays["prior_valid"] > 0
        gt = arrays["gt_m"].astype(np.float32)
        gt_valid = arrays["gt_valid"] > 0
        relative = arrays["disp"].astype(np.float32)
        if image.shape[:2] != prior.shape or gt.shape != prior.shape or relative.shape != prior.shape:
            raise ValueError(f"shape mismatch in {source_path.name}")
        condition, affine, alignment_diagnostics = build_conditions(
            relative, prior, prior_valid, alignment
        )
        destination = args.out / source_path.name
        temporary = destination.with_suffix(".npz.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                img_bgr=image,
                prior_m=prior,
                prior_valid=prior_valid.astype(np.uint8),
                disp=relative,
                gt_m=np.where(gt_valid, gt, 0.0).astype(np.float32),
                gt_valid=gt_valid.astype(np.uint8),
                condition=condition,
                affine=np.asarray(affine, dtype=np.float32),
            )
        os.replace(temporary, destination)
        output_metadata = {
            "schema": DEPTH_CACHE_SCHEMA,
            "sequence": sequence,
            "frame": frame,
            "depth_unit": "m",
            "alignment": alignment.to_dict(),
            "alignment_diagnostics": alignment_diagnostics,
            "raw_valid_pixels": int(prior_valid.sum()),
            "gt_valid_pixels": int(gt_valid.sum()),
            "source_cache_file": source_path.name,
            "source_cache_sha256": sha256_file(source_path),
            "dbh_labels_read": False,
            "trunk_mask_read": False,
            "frozen_mde_size": args.frozen_mde_size,
        }
        atomic_json(destination.with_suffix(".json"), output_metadata)
        records.append({"file": destination.name, **output_metadata})
        print(f"[{len(records):02d}] {destination.stem}: gt={int(gt_valid.sum())}", flush=True)

    if not records:
        raise ValueError(f"no samples found in {args.source}")
    manifest = {
        "schema": DEPTH_CACHE_SCHEMA,
        "depth_unit": "m",
        "sample_count": len(records),
        "sequences": sorted({record["sequence"] for record in records}),
        "dbh_labels_read": False,
        "trunk_masks_read": False,
        "frozen_mde_size": args.frozen_mde_size,
        "source_cache": str(args.source),
        "alignment": alignment.to_dict(),
        "records": records,
    }
    atomic_json(args.out / "manifest.json", manifest)
    print(json.dumps({key: manifest[key] for key in ("schema", "sample_count", "sequences")}, indent=2))


if __name__ == "__main__":
    main()
