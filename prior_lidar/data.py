"""Explicit-unit vKITTI data pipeline for PriorDA-LiDAR training."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .alignment import AlignmentConfig
from .conditions import build_conditions
from .temporal import TEMPORAL_CACHE_SCHEMA


CACHE_SCHEMA = "prior_lidar_vkitti_meters_v1"
DEPTH_CACHE_SCHEMA = "prior_lidar_dense_depth_v1"
MODEL_SIZE = 518
SUPPORTED_MDE_SIZES = ("vits", "vitb", "vitl")
PAPER_PATTERNS = ("sparse", "lowres", "missing")
TEMPORAL_REQUIRED_ARRAYS = {
    "img_bgr",
    "prior_m",
    "prior_valid",
    "disp",
    "gt_m",
    "gt_valid",
    "condition",
    "affine",
}


def read_cache_manifest(
    cache_dir: Path,
    expected_frozen_mde_size: str | None = None,
    require_frozen_mde: bool = False,
) -> dict:
    path = Path(cache_dir) / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is required; legacy vKITTI caches with implicit units are rejected"
        )
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != CACHE_SCHEMA or manifest.get("depth_unit") != "m":
        raise ValueError(f"unsupported or non-metric vKITTI cache manifest: {path}")
    frozen_size = manifest.get("frozen_mde_size")
    if require_frozen_mde and frozen_size not in SUPPORTED_MDE_SIZES:
        raise ValueError(
            f"{path} must declare frozen_mde_size in {SUPPORTED_MDE_SIZES}; "
            "an implicit MDE/cache pairing is unsafe"
        )
    if expected_frozen_mde_size is not None and frozen_size != expected_frozen_mde_size:
        raise ValueError(
            f"frozen MDE mismatch for {path}: cache={frozen_size!r}, "
            f"requested={expected_frozen_mde_size!r}"
        )
    return manifest


def cache_paths(
    cache_dir: Path,
    scenes: set[str],
    expected_frozen_mde_size: str | None = None,
) -> list[Path]:
    manifest = read_cache_manifest(
        cache_dir,
        expected_frozen_mde_size=expected_frozen_mde_size,
        require_frozen_mde=expected_frozen_mde_size is not None,
    )
    manifest_size = manifest.get("frozen_mde_size")
    paths: list[Path] = []
    for path in sorted(Path(cache_dir).glob("*.npz")):
        metadata_path = path.with_suffix(".json")
        if not metadata_path.is_file():
            raise FileNotFoundError(f"metadata missing for {path.name}")
        metadata = json.loads(metadata_path.read_text())
        if expected_frozen_mde_size is not None and metadata.get("frozen_mde_size") != manifest_size:
            raise ValueError(f"sample/cache MDE provenance mismatch in {path.name}")
        if metadata.get("scene") in scenes:
            paths.append(path)
    if not paths:
        raise ValueError(f"no cache samples for scenes {sorted(scenes)}")
    return paths


class VkittiPriorDataset(Dataset):
    """vKITTI samples with paper prior patterns and deployment-shaped inputs."""

    def __init__(
        self,
        cache_dir: Path,
        scenes: set[str],
        alignment: AlignmentConfig,
        training: bool,
        seed: int = 0,
        repeats: int = 1,
        patterns: tuple[str, ...] = PAPER_PATTERNS,
        validation_prior_points: int = 400,
        noise_std_relative: float = 0.005,
        outlier_fraction: float = 0.005,
        frozen_mde_size: str | None = None,
        keep_aspect_ratio: bool = True,
    ) -> None:
        unknown = set(patterns) - set(PAPER_PATTERNS)
        if not patterns or unknown:
            raise ValueError(f"unknown prior patterns: {sorted(unknown)}")
        self.paths = cache_paths(Path(cache_dir), scenes, frozen_mde_size)
        self.alignment = alignment
        self.training = bool(training)
        self.seed = int(seed)
        self.epoch = 0
        self.repeats = max(1, int(repeats))
        self.patterns = tuple(patterns)
        self.validation_prior_points = int(validation_prior_points)
        self.noise_std_relative = float(noise_std_relative) if training else 0.0
        self.outlier_fraction = float(outlier_fraction) if training else 0.0
        self.frozen_mde_size = frozen_mde_size
        self.keep_aspect_ratio = bool(keep_aspect_ratio)
        if self.validation_prior_points < alignment.k:
            raise ValueError("validation_prior_points must be at least K")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.paths) * self.repeats

    def _rng(self, index: int) -> np.random.Generator:
        sequence = np.random.SeedSequence([self.seed, self.epoch, int(index)])
        return np.random.default_rng(sequence)

    @staticmethod
    def _square_crop(arr: np.ndarray, x0: int, y0: int, side: int) -> np.ndarray:
        return np.ascontiguousarray(arr[y0:y0 + side, x0:x0 + side])

    @staticmethod
    def _model_shape(height: int, width: int) -> tuple[int, int]:
        """Match PriorDA's lower-bound, aspect-preserving resize exactly."""

        scale = max(MODEL_SIZE / float(height), MODEL_SIZE / float(width))
        resized_h = max(MODEL_SIZE, int(np.round(scale * height / 14.0)) * 14)
        resized_w = max(MODEL_SIZE, int(np.round(scale * width / 14.0)) * 14)
        return resized_h, resized_w

    def _resize_sample(self, arrays: dict, rng: np.random.Generator) -> tuple[np.ndarray, ...]:
        image = arrays["img_bgr"]
        gt = arrays["gt_m"].astype(np.float32)
        valid = arrays["gt_valid"] > 0
        disparity = arrays["disp"].astype(np.float32)
        height, width = gt.shape
        if self.keep_aspect_ratio:
            # The deployment path feeds the complete frame to raw2input().
            # Keeping this geometry during training avoids teaching the model
            # a square-image distortion that never occurs at inference.
            output_h, output_w = self._model_shape(height, width)
        else:
            side = min(height, width)
            if self.training:
                x0 = int(rng.integers(0, width - side + 1))
                y0 = int(rng.integers(0, height - side + 1))
            else:
                x0 = (width - side) // 2
                y0 = (height - side) // 2
            image = self._square_crop(image, x0, y0, side)
            gt = self._square_crop(gt, x0, y0, side)
            valid = self._square_crop(valid, x0, y0, side)
            disparity = self._square_crop(disparity, x0, y0, side)
            output_h = output_w = MODEL_SIZE

        image = cv2.resize(image, (output_w, output_h), interpolation=cv2.INTER_CUBIC)
        gt = cv2.resize(gt, (output_w, output_h), interpolation=cv2.INTER_NEAREST)
        valid = cv2.resize(valid.astype(np.uint8), (output_w, output_h), interpolation=cv2.INTER_NEAREST) > 0
        disparity = cv2.resize(disparity, (output_w, output_h), interpolation=cv2.INTER_LINEAR)
        if self.training and bool(rng.integers(0, 2)):
            image = image[:, ::-1]
            gt = gt[:, ::-1]
            valid = valid[:, ::-1]
            disparity = disparity[:, ::-1]
        return tuple(np.ascontiguousarray(x) for x in (image, gt, valid, disparity))

    def _make_prior(
        self,
        gt: np.ndarray,
        valid: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        pattern = str(rng.choice(self.patterns)) if self.training else "sparse"
        prior = gt.copy()
        prior_valid = valid.copy()
        if pattern == "sparse":
            candidates = np.flatnonzero(valid.ravel())
            if self.training:
                count = int(rng.integers(100, min(2000, candidates.size) + 1))
            else:
                count = min(self.validation_prior_points, candidates.size)
            chosen = rng.choice(candidates, size=count, replace=False)
            prior_valid = np.zeros(valid.size, dtype=bool)
            prior_valid[chosen] = True
            prior_valid = prior_valid.reshape(valid.shape)
        elif pattern == "lowres":
            height, width = gt.shape
            small_h = max(1, int(round(height / 8)))
            small_w = max(1, int(round(width / 8)))
            small_depth = cv2.resize(gt, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
            small_valid = cv2.resize(valid.astype(np.uint8), (small_w, small_h), interpolation=cv2.INTER_NEAREST)
            prior = cv2.resize(small_depth, (width, height), interpolation=cv2.INTER_NEAREST)
            prior_valid = cv2.resize(small_valid, (width, height), interpolation=cv2.INTER_NEAREST) > 0
        elif pattern == "missing":
            side = min(160, *gt.shape)
            x0 = int(rng.integers(0, gt.shape[1] - side + 1))
            y0 = int(rng.integers(0, gt.shape[0] - side + 1))
            prior_valid[y0:y0 + side, x0:x0 + side] = False

        prior_valid &= np.isfinite(prior) & (prior > 0)
        if self.noise_std_relative > 0:
            noise = rng.normal(0.0, self.noise_std_relative, size=prior.shape).astype(np.float32)
            prior[prior_valid] *= 1.0 + noise[prior_valid]
        if self.outlier_fraction > 0 and prior_valid.any():
            candidates = np.flatnonzero(prior_valid.ravel())
            count = int(round(candidates.size * self.outlier_fraction))
            if count:
                chosen = rng.choice(candidates, size=count, replace=False)
                factors = np.exp(rng.normal(0.0, 0.25, size=count)).astype(np.float32)
                prior.ravel()[chosen] *= factors
        prior[~prior_valid] = 0.0
        return prior.astype(np.float32), prior_valid, pattern

    def __getitem__(self, index: int) -> dict:
        path = self.paths[index % len(self.paths)]
        rng = self._rng(index)
        with np.load(path) as source:
            arrays = {key: source[key] for key in source.files}
        image, gt, valid, disparity = self._resize_sample(arrays, rng)
        prior, prior_valid, pattern = self._make_prior(gt, valid, rng)
        condition, affine, diagnostics = build_conditions(
            disparity, prior, prior_valid, self.alignment
        )
        metadata = json.loads(path.with_suffix(".json").read_text())
        return {
            "image": torch.from_numpy(image.copy()).permute(2, 0, 1).to(torch.uint8),
            "condition": torch.from_numpy(condition),
            "target_m": torch.from_numpy(gt.copy())[None],
            "valid": torch.from_numpy(valid.copy())[None],
            "affine": torch.tensor(affine, dtype=torch.float32),
            "pattern": pattern,
            "sample": path.stem,
            "scene": metadata["scene"],
            "support_points": diagnostics["knn_support_points"],
        }


def read_temporal_manifest(
    cache_dir: Path,
    expected_frozen_mde_size: str | None = None,
    require_frozen_mde: bool = False,
) -> dict:
    path = Path(cache_dir) / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} is required")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema": TEMPORAL_CACHE_SCHEMA,
        "depth_unit": "m",
        "target_frame_excluded": True,
        "dbh_labels_read": False,
        "trunk_masks_read": False,
    }
    mismatched = {key: (manifest.get(key), expected) for key, expected in required.items()
                  if manifest.get(key) != expected}
    if mismatched:
        raise ValueError(f"unsafe temporal cache manifest {path}: {mismatched}")
    frozen_size = manifest.get("frozen_mde_size")
    if require_frozen_mde and frozen_size not in SUPPORTED_MDE_SIZES:
        raise ValueError(f"{path} must declare a supported frozen_mde_size")
    if expected_frozen_mde_size is not None and frozen_size != expected_frozen_mde_size:
        raise ValueError(
            f"frozen MDE mismatch for {path}: cache={frozen_size!r}, "
            f"requested={expected_frozen_mde_size!r}"
        )
    return manifest


def temporal_cache_paths(
    cache_dir: Path,
    sequences: set[str],
    expected_frozen_mde_size: str | None = None,
) -> list[Path]:
    manifest = read_temporal_manifest(
        cache_dir,
        expected_frozen_mde_size=expected_frozen_mde_size,
        require_frozen_mde=expected_frozen_mde_size is not None,
    )
    paths: list[Path] = []
    for record in manifest.get("records", []):
        sequence = str(record.get("sequence", "")).zfill(2)
        if sequence not in sequences:
            continue
        if record.get("target_excluded") is not True:
            raise ValueError(f"target-frame leakage declared for {record.get('file')}")
        path = Path(cache_dir) / str(record["file"])
        metadata_path = path.with_suffix(".json")
        if not path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"cache sample or metadata missing for {path.name}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if expected_frozen_mde_size is not None and metadata.get("frozen_mde_size") != manifest.get("frozen_mde_size"):
            raise ValueError(f"sample/cache MDE provenance mismatch in {path.name}")
        forbidden = {"dbh_cm", "measurement_row"} & set(metadata)
        if forbidden or metadata.get("trunk_mask_read") is not False:
            raise ValueError(f"DBH-derived metadata in temporal sample {path.name}: {sorted(forbidden)}")
        paths.append(path)
    if not paths:
        raise ValueError(f"no temporal samples for sequences {sorted(sequences)}")
    return sorted(paths)


class TemporalPriorDataset(Dataset):
    """Full-depth crops from target-frame-excluded temporal LiDAR.

    The dataset deliberately has no mask argument or mask access. Every valid
    temporal point in the sampled crop contributes to the depth loss.
    """

    def __init__(
        self,
        cache_dir: Path,
        sequences: set[str],
        alignment: AlignmentConfig,
        training: bool,
        seed: int = 0,
        repeats: int = 1,
        crop_size: int | None = MODEL_SIZE,
        frozen_mde_size: str | None = None,
        keep_aspect_ratio: bool = False,
    ) -> None:
        self.paths = temporal_cache_paths(Path(cache_dir), sequences, frozen_mde_size)
        self.alignment = alignment
        self.training = bool(training)
        self.seed = int(seed)
        self.epoch = 0
        self.repeats = max(1, int(repeats))
        self.crop_size = None if crop_size is None else int(crop_size)
        self.frozen_mde_size = frozen_mde_size
        self.keep_aspect_ratio = bool(keep_aspect_ratio)
        if self.crop_size is not None and self.crop_size < 14:
            raise ValueError("crop size must be at least one 14-pixel patch")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.paths) * self.repeats

    def _rng(self, index: int) -> np.random.Generator:
        return np.random.default_rng(
            np.random.SeedSequence([self.seed, self.epoch, int(index)])
        )

    @staticmethod
    def _load(path: Path) -> dict[str, np.ndarray]:
        with np.load(path, allow_pickle=False) as source:
            missing = TEMPORAL_REQUIRED_ARRAYS - set(source.files)
            if missing:
                raise KeyError(f"{path.name} is missing {sorted(missing)}")
            if "mask" in source.files:
                raise ValueError(f"{path.name} contains a forbidden training mask")
            return {key: source[key] for key in TEMPORAL_REQUIRED_ARRAYS}

    def _bounds(
        self, shape: tuple[int, int], rng: np.random.Generator
    ) -> tuple[int, int, int, int]:
        height, width = shape
        if self.crop_size is None:
            return 0, height, 0, width
        side = min(self.crop_size, height, width)
        if self.training:
            y0 = int(rng.integers(0, height - side + 1))
            x0 = int(rng.integers(0, width - side + 1))
        else:
            y0 = (height - side) // 2
            x0 = (width - side) // 2
        return y0, y0 + side, x0, x0 + side

    def __getitem__(self, index: int) -> dict:
        path = self.paths[index % len(self.paths)]
        rng = self._rng(index)
        arrays = self._load(path)
        shape = arrays["prior_m"].shape
        if arrays["img_bgr"].shape[:2] != shape:
            raise ValueError(f"RGB/depth shape mismatch in {path.name}")
        if self.keep_aspect_ratio:
            y0, y1, x0, x1 = 0, shape[0], 0, shape[1]
        else:
            y0, y1, x0, x1 = self._bounds(shape, rng)
        flip = self.training and bool(rng.integers(0, 2))

        def crop(array: np.ndarray) -> np.ndarray:
            value = array[y0:y1, x0:x1]
            if flip:
                value = value[:, ::-1]
            return np.ascontiguousarray(value)

        image = crop(arrays["img_bgr"])
        prior = crop(arrays["prior_m"]).astype(np.float32)
        prior_valid = crop(arrays["prior_valid"]) > 0
        target = crop(arrays["gt_m"]).astype(np.float32)
        target_valid = crop(arrays["gt_valid"]) > 0
        target_valid &= np.isfinite(target) & (target > 0)
        condition = arrays["condition"][:, y0:y1, x0:x1]
        if flip:
            condition = condition[:, :, ::-1]
        condition = np.ascontiguousarray(condition, dtype=np.float32)
        affine_values = np.asarray(arrays["affine"], dtype=np.float32).reshape(-1)
        if affine_values.size != 2:
            raise ValueError(f"invalid affine metadata in {path.name}")
        affine = (float(affine_values[0]), float(affine_values[1]))
        metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        cached_alignment = metadata.get("alignment")
        if cached_alignment != self.alignment.to_dict():
            raise ValueError(f"alignment mismatch for cached conditions in {path.name}")
        if self.keep_aspect_ratio:
            out_h, out_w = VkittiPriorDataset._model_shape(y1 - y0, x1 - x0)
            image = cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
            prior = cv2.resize(prior, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
            prior_valid = cv2.resize(prior_valid.astype(np.uint8), (out_w, out_h), interpolation=cv2.INTER_NEAREST) > 0
            target = cv2.resize(target, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
            target_valid = cv2.resize(target_valid.astype(np.uint8), (out_w, out_h), interpolation=cv2.INTER_NEAREST) > 0
            condition = np.stack([
                cv2.resize(channel, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
                for channel in condition
            ], axis=0).astype(np.float32)
            crop_info = torch.tensor([y0, y1, x0, x1], dtype=torch.int32)
        else:
            crop_info = torch.tensor([y0, y1, x0, x1], dtype=torch.int32)
        return {
            "image": torch.from_numpy(image).permute(2, 0, 1).to(torch.uint8),
            "condition": torch.from_numpy(condition),
            "target_m": torch.from_numpy(target)[None],
            "valid": torch.from_numpy(target_valid)[None],
            "prior_m": torch.from_numpy(prior)[None],
            "prior_valid": torch.from_numpy(prior_valid.copy())[None],
            "affine": torch.tensor(affine, dtype=torch.float32),
            "sample": path.stem,
            "sequence": str(metadata["sequence"]).zfill(2),
            "frame": int(metadata["frame"]),
            "crop": crop_info,
            "support_points": int(metadata["alignment_diagnostics"]["knn_support_points"]),
        }


def read_dense_manifest(
    cache_dir: Path,
    expected_frozen_mde_size: str | None = None,
    require_frozen_mde: bool = False,
) -> dict:
    path = Path(cache_dir) / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} is required")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema": DEPTH_CACHE_SCHEMA,
        "depth_unit": "m",
        "dbh_labels_read": False,
        "trunk_masks_read": False,
    }
    mismatched = {
        key: (manifest.get(key), expected)
        for key, expected in required.items()
        if manifest.get(key) != expected
    }
    if mismatched:
        raise ValueError(f"unsafe dense-depth manifest {path}: {mismatched}")
    frozen_size = manifest.get("frozen_mde_size")
    if require_frozen_mde and frozen_size not in SUPPORTED_MDE_SIZES:
        raise ValueError(f"{path} must declare a supported frozen_mde_size")
    if expected_frozen_mde_size is not None and frozen_size != expected_frozen_mde_size:
        raise ValueError(
            f"frozen MDE mismatch for {path}: cache={frozen_size!r}, "
            f"requested={expected_frozen_mde_size!r}"
        )
    return manifest


def dense_cache_paths(
    cache_dir: Path,
    sequences: set[str],
    expected_frozen_mde_size: str | None = None,
) -> list[Path]:
    manifest = read_dense_manifest(
        cache_dir,
        expected_frozen_mde_size=expected_frozen_mde_size,
        require_frozen_mde=expected_frozen_mde_size is not None,
    )
    paths: list[Path] = []
    for record in manifest.get("records", []):
        sequence = str(record.get("sequence", "")).zfill(2)
        if sequence not in sequences:
            continue
        path = Path(cache_dir) / str(record["file"])
        metadata_path = path.with_suffix(".json")
        if not path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"cache sample or metadata missing for {path.name}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if expected_frozen_mde_size is not None and metadata.get("frozen_mde_size") != manifest.get("frozen_mde_size"):
            raise ValueError(f"sample/cache MDE provenance mismatch in {path.name}")
        forbidden = {"dbh_cm", "measurement_row", "mask", "description"} & set(metadata)
        if forbidden or metadata.get("trunk_mask_read") is not False:
            raise ValueError(f"DBH/ROI metadata in dense sample {path.name}: {sorted(forbidden)}")
        paths.append(path)
    if not paths:
        raise ValueError(f"no dense-depth samples for sequences {sorted(sequences)}")
    return sorted(paths)


class DensePriorDataset(TemporalPriorDataset):
    """Dataset for the permitted 40-tree dense-depth cache.

    It inherits the crop and condition handling but uses a distinct manifest
    schema.  No cached segmentation or annotation text is accessed.
    """

    def __init__(
        self,
        cache_dir: Path,
        sequences: set[str],
        alignment: AlignmentConfig,
        training: bool,
        seed: int = 0,
        repeats: int = 1,
        crop_size: int | None = MODEL_SIZE,
        frozen_mde_size: str | None = None,
        keep_aspect_ratio: bool = False,
    ) -> None:
        self.paths = dense_cache_paths(Path(cache_dir), sequences, frozen_mde_size)
        self.alignment = alignment
        self.training = bool(training)
        self.seed = int(seed)
        self.epoch = 0
        self.repeats = max(1, int(repeats))
        self.crop_size = None if crop_size is None else int(crop_size)
        self.frozen_mde_size = frozen_mde_size
        self.keep_aspect_ratio = bool(keep_aspect_ratio)
        if self.crop_size is not None and self.crop_size < 14:
            raise ValueError("crop size must be at least one 14-pixel patch")
