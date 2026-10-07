#!/usr/bin/env python3
"""Evaluate PriorDA-LiDAR without test-time calibration or tuned DBH fallbacks.

The evaluator keeps the measurement geometry identical for every depth source.
It is deliberately independent of ``DBH_13hight.py``: labels are read from
``tree_62.txt``, the target slice is located from the raw LiDAR prior and the
fixed camera calibration, and missing depth is reported as a failure rather
than replaced by another source.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prior_lidar.alignment import AlignmentConfig
from prior_lidar.conditions import build_conditions, decode_metric_depth
from prior_lidar.data import SUPPORTED_MDE_SIZES
from prior_lidar.metrics import depth_metrics
from prior_lidar.model import PriorLidarModel, sha256_file


TEST_SEQUENCES = {"00", "01", "02", "04", "05"}
MODEL_METHODS = ("prior", "trained")
EVAL_CACHE_SCHEMA = "prior_lidar_labeled_eval_v1"
#: Caches produced by ``tools/build_lidar_measurement_cache.py``.  The measurement
#: protocol is identical; only the provenance of mask/depth differs.
TRAIN_CACHE_SCHEMA = "prior_lidar_labeled_train_v1"
ACCEPTED_CACHE_SCHEMAS = (EVAL_CACHE_SCHEMA, TRAIN_CACHE_SCHEMA)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / "data/lidar_cache")
    parser.add_argument("--labels", type=Path, default=ROOT / "data/sequences/tree_62.txt")
    parser.add_argument("--prior-checkpoint", type=Path, default=ROOT / "checkpoints/prior_depth_anything_vitb_1_1.pth")
    parser.add_argument("--trained-checkpoint", type=Path, default=ROOT / "runs/prior_lidar_vkitti_v1/best.pt")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/eval_prior_lidar.json")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-prior-points", type=int, default=0,
                        help="Deterministic input-prior thinning; 0 keeps every raw LiDAR pixel.")
    parser.add_argument("--max-knn-support-points", type=int, default=20000)
    parser.add_argument("--alignment-k", type=int, default=5)
    parser.add_argument("--alignment-distance-power", type=float, default=1.0)
    parser.add_argument("--robust-alignment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--frozen-mde-size", choices=("vits", "vitb", "vitl"), default="vitl")
    parser.add_argument(
        "--allow-checkpoint-alignment-mismatch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow protocol-sensitivity runs with a checkpoint trained using another alignment.",
    )
    parser.add_argument("--slice-half-height-m", type=float, default=0.05,
                        help="Physical half-thickness of the DBH slice, in meters.")
    parser.add_argument("--base-quantile", type=float, default=0.25,
                        help="Lower mask-row fraction used to estimate the raw base depth.")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--expect-samples", type=int, default=0,
                        help="required number of samples in the cache (0 = accept any)")
    parser.add_argument("--require-labels", action=argparse.BooleanOptionalAction, default=True,
                        help="require a tree_62 label for every cached sample")
    parser.add_argument("--allowed-sequences", default="",
                        help="comma separated sequences accepted in the cache (default: any)")
    parser.add_argument("--filter-by-labels", action=argparse.BooleanOptionalAction, default=True,
                        help="evaluate only the samples that the label file covers (default on)")
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--prediction-cache", type=Path, default=None,
                        help="Optional directory for reusable per-model depth maps.")
    return parser.parse_args()


def read_tree_labels(path: Path) -> dict[tuple[str, str], float]:
    labels: dict[tuple[str, str], float] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        expected = {"name", "description", "DBH"}
        if not expected.issubset(reader.fieldnames or []):
            raise ValueError(f"{path} must contain tab-separated columns {sorted(expected)}")
        for row in reader:
            # label files may be CRLF; a stray "\r" would silently break every
            # (name, description) lookup
            clean = lambda text: str(text).strip().strip("\r\n").strip()  # noqa: E731
            key = (clean(row["name"]), clean(row["description"]))
            value = clean(row["DBH"])
            if not value:
                continue
            if key in labels:
                raise ValueError(f"duplicate tree label: {key}")
            labels[key] = float(value)
    return labels


def cache_sample_paths(cache: Path) -> list[Path]:
    return sorted(Path(cache).glob("idx*.npz"))


def read_eval_manifest(
    cache: Path,
    expected_frozen_mde_size: str | None = None,
    required_schema: tuple[str, ...] = ACCEPTED_CACHE_SCHEMAS,
) -> dict:
    """Validate a measurement cache manifest against its own contents.

    Earlier versions hard-coded the schema, a 40-sample count and the exact five
    evaluation sequences, which made every other labelled cache unusable.  The
    checks below instead verify that the manifest *agrees with the files on disk*,
    so any correctly built cache can be evaluated.
    """

    path = Path(cache) / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"cache manifest is required: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    schema = manifest.get("schema")
    if schema not in required_schema:
        raise ValueError(f"unsupported cache schema in {path}: {schema!r}")
    if manifest.get("depth_unit") != "m":
        raise ValueError(f"cache must use meters: {path}")

    files = cache_sample_paths(cache)
    if not files:
        raise FileNotFoundError(f"no idx*.npz samples found in {cache}")
    declared = manifest.get("sample_count")
    if declared is not None and int(declared) != len(files):
        raise ValueError(f"manifest sample_count {declared} does not match {len(files)} files in {cache}")

    frozen_size = manifest.get("frozen_mde_size")
    if frozen_size not in SUPPORTED_MDE_SIZES:
        raise ValueError(f"cache must declare a supported frozen_mde_size: {path}")
    if expected_frozen_mde_size is not None and frozen_size != expected_frozen_mde_size:
        raise ValueError(
            f"frozen MDE mismatch for {path}: cache={frozen_size!r}, "
            f"requested={expected_frozen_mde_size!r}"
        )

    declared_sequences = {str(value).zfill(2) for value in manifest.get("sequences", [])}
    if declared_sequences:
        present = set()
        for sample in files:
            metadata_path = sample.with_suffix(".json")
            if metadata_path.is_file():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                present.add(str(metadata.get("sequence") or sequence_from_name(str(metadata["name"]))))
        missing = present - declared_sequences
        if missing:
            raise ValueError(f"manifest sequences {sorted(declared_sequences)} miss {sorted(missing)} from {cache}")
    return manifest


def load_samples(
    cache: Path,
    labels: dict[tuple[str, str], float] | None = None,
    expected_frozen_mde_size: str | None = None,
    require_labels: bool = True,
    expect_samples: int = 0,
    allowed_sequences: set[str] | None = None,
    skip_unlabelled: bool = False,
) -> list[tuple[Path, dict, dict]]:
    """Load a measurement cache.

    ``labels`` is used when the caller has a label file; a sample missing from it
    falls back to the ``dbh_cm`` recorded in its own metadata, and ``require_labels``
    decides whether a completely unlabelled sample is an error.  ``expect_samples``
    and ``allowed_sequences`` default to "no restriction", so caches of any size and
    sequence set can be evaluated.
    """

    labels = labels or {}
    if expected_frozen_mde_size is not None or cache.is_dir():
        read_eval_manifest(cache, expected_frozen_mde_size)
    samples = []
    for path in cache_sample_paths(cache):
        metadata_path = path.with_suffix(".json")
        if not metadata_path.is_file():
            raise FileNotFoundError(f"metadata missing for {path.name}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata_mde = metadata.get("frozen_mde_size")
        if expected_frozen_mde_size is not None and metadata_mde not in (None, expected_frozen_mde_size):
            raise ValueError(f"sample/cache MDE mismatch in {path.name}: {metadata_mde!r}")
        name = str(metadata.get("name", ""))
        description = str(metadata.get("description", ""))
        sequence = str(metadata.get("sequence") or sequence_from_name(name))
        if allowed_sequences is not None and sequence not in allowed_sequences:
            raise ValueError(f"sequence {sequence} not allowed for {path.name}")
        key = (name.strip(), description.strip())
        if key in labels:
            metadata["dbh_cm"] = labels[key]
        elif require_labels:
            # A label file may deliberately cover only part of a cache (for example
            # after removing low-quality samples).  Skipping is opt-in so an
            # accidental mismatch is still reported loudly.
            if skip_unlabelled:
                continue
            raise ValueError(f"no tree label for {name!r}, {description!r}")
        elif metadata.get("dbh_cm") in (None, ""):
            raise ValueError(f"no label and no metadata dbh_cm for {path.name}")
        else:
            metadata["dbh_cm"] = float(metadata["dbh_cm"])
        with np.load(path, allow_pickle=False) as source:
            required = {"img_bgr", "prior_m", "prior_valid", "disp", "mask", "gt_m", "gt_valid"}
            missing = required - set(source.files)
            if missing:
                raise KeyError(f"{path.name} is missing {sorted(missing)}")
            arrays = {key: source[key] for key in required}
        samples.append((path, arrays, metadata))
    if expect_samples and len(samples) != expect_samples:
        raise ValueError(f"expected {expect_samples} samples, found {len(samples)} in {cache}")
    return samples


def sequence_from_name(name: str) -> str:
    match = re.fullmatch(r"S(\d+)-\d+", name.strip())
    if not match:
        raise ValueError(f"invalid tree sequence name: {name}")
    return f"{int(match.group(1)):02d}"


def frame_from_name(name: str) -> int:
    match = re.fullmatch(r"S\d+-(\d+)", name.strip())
    if not match:
        raise ValueError(f"invalid tree frame name: {name}")
    return int(match.group(1))


def read_calibration(sequence: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = {}
    path = ROOT / "data/sequences" / sequence / "calib.txt"
    for line in path.read_text(encoding="utf-8").splitlines():
        key, value = line.split(":", 1)
        values[key] = np.asarray([float(item) for item in value.split()], dtype=np.float64)
    if "P2" not in values or "Tr" not in values:
        raise ValueError(f"P2/Tr calibration missing from {path}")
    projection = values["P2"].reshape(3, 4)
    transform = values["Tr"].reshape(3, 4)
    intrinsics = projection[:, :3]
    distortion = np.asarray([0.0910, -0.2054, 0.0, 0.0, 0.0], dtype=np.float64)
    return intrinsics, transform, distortion


def raw_mask_hash(raw: np.ndarray, mask: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(raw, dtype=np.float32).tobytes())
    digest.update(np.ascontiguousarray(mask, dtype=np.uint8).tobytes())
    return digest.hexdigest()


def contiguous_segments(values: np.ndarray) -> list[tuple[int, int]]:
    if values.size == 0:
        return []
    values = np.asarray(values, dtype=np.int64)
    breaks = np.flatnonzero(np.diff(values) > 1)
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks, values.size - 1]
    return [(int(values[start]), int(values[end])) for start, end in zip(starts, ends)]


def robust_median(values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        return None
    return float(np.median(values))


def build_fixed_geometry(
    raw: np.ndarray,
    raw_valid: np.ndarray,
    mask: np.ndarray,
    fx: float,
    fy: float,
    target_height_m: float = 1.3,
    slice_half_height_m: float = 0.05,
    base_quantile: float = 0.25,
) -> dict:
    """Create model-independent DBH geometry from raw depth and the mask.

    The only distances are the physical target height (1.3 m) and a fixed
    5-cm slice thickness. The raw prior determines the visible trunk bottom
    and base range; model predictions never affect the selected rows/segment.
    """

    mask = np.asarray(mask, dtype=bool)
    raw = np.asarray(raw, dtype=np.float32)
    raw_valid = np.asarray(raw_valid, dtype=bool)
    if mask.shape != raw.shape or raw_valid.shape != raw.shape:
        raise ValueError("raw, raw_valid, and mask must share shape")
    rows = np.flatnonzero(mask.any(axis=1))
    if rows.size == 0:
        raise ValueError("empty trunk mask")
    top, bottom = int(rows.min()), int(rows.max())
    row_span = max(1, bottom - top)
    # Use a fixed lower fraction of the visible mask to estimate the base
    # range. This is an input-only statistic, not a test-label-derived gate.
    base_start = int(math.floor(bottom - row_span * base_quantile))
    base_pixels = raw_valid & mask & (np.indices(mask.shape)[0] >= base_start)
    base_depth = robust_median(raw[base_pixels])
    if base_depth is None:
        raise ValueError("raw LiDAR has no valid depth in the lower trunk range")
    raw_row = float(bottom) - target_height_m * float(fy) / base_depth
    if raw_row < top or raw_row > bottom:
        raise ValueError(f"1.3 m target is outside visible mask: row={raw_row:.2f}, range={top}:{bottom}")
    half_rows = max(1, int(math.ceil(slice_half_height_m * float(fy) / base_depth)))
    row_start = max(top, int(math.floor(raw_row - half_rows)))
    row_end = min(bottom, int(math.ceil(raw_row + half_rows)))

    # Estimate the trunk center from the lower mask, then choose the segment
    # nearest that center on each target row. No width threshold or fallback
    # is applied; rows without a segment simply contribute no measurement.
    lower_mask_rows = np.flatnonzero(mask.any(axis=1))
    lower_cut = int(math.floor(bottom - row_span * 0.35))
    lower_centers = []
    for row in lower_mask_rows[lower_mask_rows >= lower_cut]:
        xs = np.flatnonzero(mask[row])
        if xs.size:
            lower_centers.append(float(xs[0] + xs[-1]) / 2.0)
    if not lower_centers:
        raise ValueError("cannot establish lower trunk center")
    center_u = float(np.median(lower_centers))
    row_segments: dict[int, tuple[int, int]] = {}
    for row in range(row_start, row_end + 1):
        segments = contiguous_segments(np.flatnonzero(mask[row]))
        if segments:
            row_segments[row] = min(
                segments,
                key=lambda segment: abs((segment[0] + segment[1]) / 2.0 - center_u),
            )
    if not row_segments:
        raise ValueError("no trunk mask segment in the physical DBH slice")
    return {
        "target_height_m": float(target_height_m),
        "slice_half_height_m": float(slice_half_height_m),
        "base_quantile": float(base_quantile),
        "mask_top": top,
        "mask_bottom": bottom,
        "base_start_row": base_start,
        "base_depth_m": base_depth,
        "target_row_raw": raw_row,
        "slice_row_start": row_start,
        "slice_row_end": row_end,
        "trunk_center_u": center_u,
        "row_segments": {str(row): [left, right] for row, (left, right) in row_segments.items()},
        "fx": float(fx),
        "fy": float(fy),
    }


def measure_dbh(depth: np.ndarray, geometry: dict) -> tuple[float | None, dict]:
    depth = np.asarray(depth, dtype=np.float32)
    widths = []
    support = 0
    row_details = []
    for row_text, segment in geometry["row_segments"].items():
        row = int(row_text)
        left, right = int(segment[0]), int(segment[1])
        values = depth[row, left:right + 1]
        valid = np.isfinite(values) & (values > 0)
        z = robust_median(values[valid])
        if z is None or right <= left:
            continue
        # Pinhole horizontal extent. The same fixed pixel segment is used for
        # every method; only the depth values can change the physical width.
        width_m = float(right - left) * z / float(geometry["fx"])
        widths.append(width_m)
        support += int(valid.sum())
        row_details.append({"row": row, "pixel_width": right - left, "depth_m": z, "width_m": width_m})
    if not widths:
        return None, {"depth_pixels": support, "rows": row_details}
    width_m = float(np.median(np.asarray(widths, dtype=np.float64)))
    return width_m * 100.0, {
        "depth_pixels": support,
        "rows": row_details,
        "width_median_m": width_m,
    }


def model_from_official(checkpoint: Path, device: str, size: str = "vitb") -> PriorLidarModel:
    model = PriorLidarModel(size)
    info = model.load_official(checkpoint, strict=True)
    if info["missing"] or info["unexpected"]:
        raise RuntimeError(f"official checkpoint did not load strictly: {info}")
    return model.to(device).eval()


def _canonical_alignment(value: dict | None) -> dict | None:
    if value is None:
        return None
    result = dict(value)
    # Checkpoints written before distance_power was made explicit used the
    # released implementation's inverse-distance weighting.
    result.setdefault("distance_power", 1.0)
    return result


def model_from_training(
    checkpoint: Path,
    device: str,
    expected_frozen_mde_size: str | None = None,
    expected_alignment: AlignmentConfig | None = None,
    allow_alignment_mismatch: bool = False,
) -> tuple[PriorLidarModel, dict]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = payload.get("model_config") or {}
    variant = config.get("variant")
    if variant not in {"priorda_v1_1_lidar", "priorda_v1_1_lidar_residual"}:
        raise ValueError(f"unsupported trained model variant: {variant}")
    if config.get("condition_order") != ["sparse_mask", "global_aligned", "knn_aligned"]:
        raise ValueError("trained checkpoint does not declare official v1.1 condition order")
    provenance = payload.get("training_config") or {}
    checkpoint_mde = config.get("frozen_mde_size", provenance.get("frozen_mde_size"))
    if expected_frozen_mde_size is not None and checkpoint_mde != expected_frozen_mde_size:
        raise ValueError(
            f"trained checkpoint frozen MDE mismatch: checkpoint={checkpoint_mde!r}, "
            f"requested={expected_frozen_mde_size!r}"
        )
    checkpoint_alignment = _canonical_alignment(config.get("alignment"))
    alignment_match = None
    if expected_alignment is not None:
        alignment_match = checkpoint_alignment == _canonical_alignment(expected_alignment.to_dict())
        if not alignment_match and not allow_alignment_mismatch:
            raise ValueError(
                "trained checkpoint alignment does not match evaluation protocol; "
                "pass --allow-checkpoint-alignment-mismatch only for an explicit sensitivity run"
            )
    residual = variant.endswith("_residual") or bool(config.get("residual_adapter"))
    model = PriorLidarModel(
        str(config.get("size", "vitb")),
        residual_adapter=residual,
        max_log_correction=float(config.get("residual_max_log", 0.05)),
    )
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device).eval(), {
        "model_config": config,
        "training_config": provenance,
        "epoch": payload.get("epoch"),
        "val": payload.get("val"),
        "checkpoint_sha256": sha256_file(checkpoint),
        "frozen_mde_size": checkpoint_mde,
        "checkpoint_alignment": checkpoint_alignment,
        "alignment_match": alignment_match,
    }


@torch.inference_mode()
def infer_model_depth(
    model: PriorLidarModel,
    image: np.ndarray,
    condition: np.ndarray,
    affine: tuple[float, float],
    device: str,
) -> np.ndarray:
    image_tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)[None]
    condition_tensor = torch.from_numpy(np.ascontiguousarray(condition))[None]
    disparity = model(image_tensor.to(device), condition_tensor.to(device), device)
    depth = decode_metric_depth(disparity.squeeze().float().cpu().numpy(), affine)
    if not np.isfinite(depth).all():
        raise ValueError("model produced non-finite depth")
    return depth.astype(np.float32)


def prediction_cache_path(
    cache_dir: Path | None,
    method: str,
    sample_path: Path,
    checkpoint_id: str | None = None,
) -> Path | None:
    if cache_dir is None:
        return None
    cache_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{checkpoint_id[:16]}" if checkpoint_id else ""
    return cache_dir / f"{method}_{sample_path.stem}{suffix}.npy"


def load_or_infer(
    method: str,
    model: PriorLidarModel,
    image: np.ndarray,
    condition: np.ndarray,
    affine: tuple[float, float],
    device: str,
    path: Path,
    cache_dir: Path | None,
    checkpoint_id: str | None = None,
) -> np.ndarray:
    cached = prediction_cache_path(cache_dir, method, path, checkpoint_id)
    if cached is not None and cached.is_file():
        depth = np.load(cached, allow_pickle=False)
        if depth.shape == image.shape[:2] and np.isfinite(depth).all():
            return depth.astype(np.float32)
    depth = infer_model_depth(model, image, condition, affine, device)
    if cached is not None:
        temporary = cached.with_suffix(".npy.tmp")
        with temporary.open("wb") as handle:
            np.save(handle, depth)
        temporary.replace(cached)
    return depth


def bootstrap_mean(values: np.ndarray, count: int, rng: np.random.Generator) -> dict | None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    if count <= 0:
        return {"mean": float(values.mean()), "low": None, "high": None, "samples": int(values.size)}
    indices = rng.integers(0, values.size, size=(count, values.size))
    means = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "low": float(np.quantile(means, 0.025)),
        "high": float(np.quantile(means, 0.975)),
        "samples": int(values.size),
        "bootstrap": int(count),
    }


def summarize_dbh(rows: list[dict], method: str, bootstrap: int, rng: np.random.Generator) -> dict:
    measured = [row for row in rows if row["methods"][method]["status"] == "ok"]
    errors = np.asarray([row["methods"][method]["error_cm"] for row in measured], dtype=np.float64)
    abs_errors = np.abs(errors)
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in measured:
        grouped[row["sequence"]].append(abs(float(row["methods"][method]["error_cm"])))
    return {
        "n_total": len(rows),
        "n_measured": len(measured),
        "failure_rate": 1.0 - len(measured) / max(1, len(rows)),
        "mae_cm": bootstrap_mean(abs_errors, bootstrap, rng),
        "rmse_cm": float(np.sqrt(np.mean(errors ** 2))) if errors.size else None,
        "median_abs_error_cm": float(np.median(abs_errors)) if errors.size else None,
        "p95_abs_error_cm": float(np.quantile(abs_errors, 0.95)) if errors.size else None,
        "per_sequence_mae_cm": {
            sequence: float(np.mean(values)) for sequence, values in sorted(grouped.items())
        },
    }


def summarize_depth(depth_rows: list[dict], method: str, bootstrap: int, rng: np.random.Generator) -> dict:
    metrics = [row["depth_metrics"][method] for row in depth_rows]
    valid = [item for item in metrics if item["pixels"] > 0]
    # The aggregate metric is weighted by valid LiDAR pixels, while the CI is
    # a tree-level bootstrap so one dense frame cannot dominate uncertainty.
    pixels = sum(item["pixels"] for item in valid)
    absrel = np.asarray([item["abs_rel"] for item in valid], dtype=np.float64)
    rmses = np.asarray([item["rmse_m"] for item in valid], dtype=np.float64)
    silogs = np.asarray([item["silog"] for item in valid], dtype=np.float64)
    return {
        "frames": len(valid),
        "pixels": pixels,
        "mean_frame_abs_rel": bootstrap_mean(absrel, bootstrap, rng),
        "mean_frame_rmse_m": bootstrap_mean(rmses, bootstrap, rng),
        "mean_frame_silog": bootstrap_mean(silogs, bootstrap, rng),
    }


def main() -> None:
    args = parse_args()
    if args.max_prior_points < 0 or args.max_knn_support_points < 0:
        raise ValueError("prior point limits must be non-negative")
    if args.alignment_k < 2:
        raise ValueError("--alignment-k must be at least 2")
    if args.alignment_distance_power < 0 or not np.isfinite(args.alignment_distance_power):
        raise ValueError("--alignment-distance-power must be finite and non-negative")
    if not 0.0 < args.base_quantile <= 1.0:
        raise ValueError("base quantile must be in (0, 1]")
    if args.slice_half_height_m <= 0:
        raise ValueError("slice half height must be positive")
    labels = read_tree_labels(args.labels)
    allowed = {part.strip().zfill(2) for part in args.allowed_sequences.split(",") if part.strip()}
    eval_manifest = read_eval_manifest(args.cache, args.frozen_mde_size)
    samples = load_samples(args.cache, labels, args.frozen_mde_size,
                           require_labels=args.require_labels, expect_samples=args.expect_samples,
                           allowed_sequences=allowed or None,
                           skip_unlabelled=args.filter_by_labels)
    alignment = AlignmentConfig(
        k=args.alignment_k,
        robust=args.robust_alignment,
        irls_steps=3 if args.robust_alignment else 0,
        max_prior_points=args.max_prior_points,
        max_knn_support_points=args.max_knn_support_points,
        distance_power=args.alignment_distance_power,
    )
    device = args.device
    official_id = sha256_file(args.prior_checkpoint)
    trained, trained_info = model_from_training(
        args.trained_checkpoint,
        device,
        expected_frozen_mde_size=args.frozen_mde_size,
        expected_alignment=alignment,
        allow_alignment_mismatch=args.allow_checkpoint_alignment_mismatch,
    )
    official = model_from_official(args.prior_checkpoint, device)
    trained_id = trained_info["checkpoint_sha256"]

    rows: list[dict] = []
    depth_rows: list[dict] = []
    for index, (path, arrays, metadata) in enumerate(samples, start=1):
        image = arrays["img_bgr"]
        raw = arrays["prior_m"].astype(np.float32)
        raw_valid = arrays["prior_valid"] > 0
        relative = arrays["disp"].astype(np.float32)
        mask = arrays["mask"] > 0
        gt = arrays["gt_m"].astype(np.float32)
        gt_valid = arrays["gt_valid"] > 0
        condition, affine, alignment_debug = build_conditions(relative, raw, raw_valid, alignment)
        sequence = sequence_from_name(metadata["name"])
        intrinsics, _transform, _distortion = read_calibration(sequence)
        geometry = build_fixed_geometry(
            raw,
            raw_valid,
            mask,
            fx=float(intrinsics[0, 0]),
            fy=float(intrinsics[1, 1]),
            slice_half_height_m=args.slice_half_height_m,
            base_quantile=args.base_quantile,
        )
        gt_dbh = labels[(str(metadata["name"]).strip(), str(metadata["description"]).strip())]
        depths: dict[str, np.ndarray] = {"raw": raw}
        depths["coarse_global"] = decode_metric_depth(condition[1], affine)
        depths["coarse_knn"] = decode_metric_depth(condition[2], affine)
        depths["prior"] = load_or_infer(
            "prior", official, image, condition, affine, device, path,
            args.prediction_cache, official_id,
        )
        depths["trained"] = load_or_infer(
            "trained", trained, image, condition, affine, device, path,
            args.prediction_cache, trained_id,
        )

        entry = {
            "index": int(metadata["index"]),
            "name": metadata["name"],
            "description": metadata["description"],
            "sequence": sequence,
            "frame": frame_from_name(metadata["name"]),
            "gt_dbh_cm": float(gt_dbh),
            "geometry": geometry,
            "alignment": alignment_debug,
            "methods": {},
        }
        for method, depth in depths.items():
            prediction, detail = measure_dbh(depth, geometry)
            method_entry = {
                "status": "ok" if prediction is not None else "no_valid_slice_depth",
                "pred_dbh_cm": prediction,
                "error_cm": None if prediction is None else float(prediction - gt_dbh),
                "detail": detail,
            }
            entry["methods"][method] = method_entry
        rows.append(entry)

        depth_entry = {
            "index": int(metadata["index"]),
            "name": metadata["name"],
            "sequence": sequence,
            "depth_metrics": {},
        }
        for method, depth in depths.items():
            depth_entry["depth_metrics"][method] = depth_metrics(depth, gt, gt_valid)
        depth_rows.append(depth_entry)
        print(
            f"[{index:02d}/40] {metadata['name']} "
            + " ".join(
                f"{method}={entry['methods'][method]['pred_dbh_cm']:.2f}"
                if entry["methods"][method]["pred_dbh_cm"] is not None else f"{method}=FAIL"
                for method in ("raw", "coarse_knn", "prior", "trained")
            ),
            flush=True,
        )

    rng = np.random.default_rng(args.seed)
    dbh_summary = {
        method: summarize_dbh(rows, method, args.bootstrap, rng)
        for method in ("raw", "coarse_global", "coarse_knn", "prior", "trained")
    }
    depth_summary = {
        method: summarize_depth(depth_rows, method, args.bootstrap, rng)
        for method in ("raw", "coarse_global", "coarse_knn", "prior", "trained")
    }
    paired = {}
    for left, right in (("prior", "trained"), ("raw", "prior"), ("coarse_knn", "trained")):
        common = [
            row for row in rows
            if row["methods"][left]["status"] == "ok" and row["methods"][right]["status"] == "ok"
        ]
        if common:
            left_abs = np.asarray([abs(row["methods"][left]["error_cm"]) for row in common])
            right_abs = np.asarray([abs(row["methods"][right]["error_cm"]) for row in common])
            paired[f"{right}_minus_{left}"] = {
                "n": len(common),
                "mae_delta_cm": float(right_abs.mean() - left_abs.mean()),
                "improved_count": int(np.sum(right_abs < left_abs)),
                "worsened_count": int(np.sum(right_abs > left_abs)),
                "unchanged_count": int(np.sum(right_abs == left_abs)),
            }
    output = {
        "protocol": {
            "name": "prior_lidar_transparent_v1",
            "labels": str(args.labels),
            "cache": str(args.cache),
            "test_sequences": sorted({str(metadata.get("sequence") or sequence_from_name(str(metadata["name"]))) for _p, _a, metadata in samples}),
            "sample_count": len(samples),
            "target_height_m": 1.3,
            "slice_half_height_m": args.slice_half_height_m,
            "base_quantile": args.base_quantile,
            "geometry_source": "raw LiDAR prior + fixed P2 calibration + shared cached mask",
            "model_used_for_geometry": False,
            "raw_writeback": False,
            "test_calibration": False,
            "model_fallback": False,
            "depth_metrics_support": "all valid gt_m pixels, no trunk-mask restriction",
            "alignment": alignment.to_dict(),
            "frozen_mde_size": args.frozen_mde_size,
            "evaluation_manifest": eval_manifest,
            "checkpoint_alignment_match": trained_info.get("alignment_match"),
            "checkpoint_alignment_override": args.allow_checkpoint_alignment_mismatch,
        },
        "checkpoints": {
            "prior": str(args.prior_checkpoint),
            "prior_sha256": official_id,
            "trained": str(args.trained_checkpoint),
            "trained_sha256": trained_id,
            "trained_info": trained_info,
        },
        "dbh_summary": dbh_summary,
        "depth_summary": depth_summary,
        "paired_dbh": paired,
        "rows": rows,
        "depth_rows": depth_rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"dbh_summary": dbh_summary, "depth_summary": depth_summary, "paired_dbh": paired}, indent=2))


if __name__ == "__main__":
    main()
