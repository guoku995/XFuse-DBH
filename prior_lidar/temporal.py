"""Target-frame-excluded temporal LiDAR projection utilities."""
from __future__ import annotations

from pathlib import Path

import numpy as np


TEMPORAL_CACHE_SCHEMA = "prior_lidar_temporal_camera_pose_v1"


def as_homogeneous(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape == (4, 4):
        return matrix.copy()
    if matrix.shape != (3, 4):
        raise ValueError("pose/extrinsic matrix must have shape [3,4] or [4,4]")
    result = np.eye(4, dtype=np.float64)
    result[:3] = matrix
    return result


def read_calibration(path: Path) -> tuple[np.ndarray, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, text = line.split(":", 1)
        values[key.strip()] = np.fromstring(text, sep=" ", dtype=np.float64)
    if values.get("P2", np.empty(0)).size != 12:
        raise ValueError(f"P2 is missing or malformed in {path}")
    if values.get("Tr", np.empty(0)).size != 12:
        raise ValueError(f"Tr is missing or malformed in {path}")
    return values["P2"].reshape(3, 4), as_homogeneous(values["Tr"].reshape(3, 4))


def camera_from_source_lidar(
    world_from_target_camera: np.ndarray,
    world_from_source_camera: np.ndarray,
    camera_from_lidar: np.ndarray,
) -> np.ndarray:
    """Map a source LiDAR point into the target camera coordinate system.

    KITTI ``poses.txt`` stores camera poses, not LiDAR poses.  Applying its
    relative transform directly to LiDAR points omits the calibrated sensor
    extrinsic and creates a systematically misregistered temporal target.
    """

    target_pose = as_homogeneous(world_from_target_camera)
    source_pose = as_homogeneous(world_from_source_camera)
    extrinsic = as_homogeneous(camera_from_lidar)
    return np.linalg.inv(target_pose) @ source_pose @ extrinsic


def project_depth_zbuffer(
    points_xyz: np.ndarray,
    target_camera_from_source_lidar: np.ndarray,
    projection: np.ndarray,
    shape: tuple[int, int],
    min_depth_m: float = 0.1,
    max_depth_m: float = 120.0,
) -> np.ndarray:
    """Project source points and retain the nearest target-camera depth."""

    points = np.asarray(points_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_xyz must have shape [N,3]")
    height, width = shape
    points_h = np.column_stack((points, np.ones(points.shape[0], dtype=np.float64)))
    camera = points_h @ as_homogeneous(target_camera_from_source_lidar).T
    image_h = camera @ np.asarray(projection, dtype=np.float64).T
    depth = camera[:, 2]
    finite = np.isfinite(depth) & np.isfinite(image_h).all(axis=1)
    finite &= (depth >= min_depth_m) & (depth <= max_depth_m) & (image_h[:, 2] > 0)

    u = np.full(points.shape[0], -1, dtype=np.int32)
    v = np.full(points.shape[0], -1, dtype=np.int32)
    u[finite] = np.rint(image_h[finite, 0] / image_h[finite, 2]).astype(np.int32)
    v[finite] = np.rint(image_h[finite, 1] / image_h[finite, 2]).astype(np.int32)
    visible = finite & (u >= 0) & (u < width) & (v >= 0) & (v < height)

    zbuffer = np.full((height, width), np.inf, dtype=np.float32)
    if visible.any():
        flat = v[visible].astype(np.int64) * width + u[visible].astype(np.int64)
        np.minimum.at(zbuffer.ravel(), flat, depth[visible].astype(np.float32))
    return zbuffer


def aggregate_neighbor_depth(
    sequence_dir: Path,
    target_frame: int,
    neighbor_frames: list[int],
    shape: tuple[int, int],
    min_depth_m: float = 0.1,
    max_depth_m: float = 120.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build target-frame-excluded depth from neighboring LiDAR scans."""

    if target_frame in neighbor_frames:
        raise ValueError("target frame must not be used as temporal supervision")
    sequence_dir = Path(sequence_dir)
    projection, camera_from_lidar = read_calibration(sequence_dir / "calib.txt")
    raw_poses = np.loadtxt(sequence_dir / "poses.txt", dtype=np.float64).reshape(-1, 3, 4)
    if target_frame >= len(raw_poses):
        raise IndexError(f"target pose {target_frame} is unavailable")
    poses = [as_homogeneous(pose) for pose in raw_poses]
    target = np.full(shape, np.inf, dtype=np.float32)
    used = 0
    for source_frame in neighbor_frames:
        lidar_path = sequence_dir / "velodyne" / f"{source_frame:06d}.bin"
        # A scheduled neighbour can be missing from disk or can sit past the last
        # recorded pose.  That just means it cannot contribute supervision; it is
        # not a reason to abort the whole cache build.
        if source_frame < 0 or source_frame >= len(poses) or not lidar_path.is_file():
            continue
        points = np.fromfile(lidar_path, dtype=np.float32)
        if points.size == 0 or points.size % 4:
            continue
        used += 1
        transform = camera_from_source_lidar(
            poses[target_frame], poses[source_frame], camera_from_lidar
        )
        projected = project_depth_zbuffer(
            points.reshape(-1, 4)[:, :3],
            transform,
            projection,
            shape,
            min_depth_m=min_depth_m,
            max_depth_m=max_depth_m,
        )
        target = np.minimum(target, projected)
    if used == 0:
        raise ValueError(f"no usable temporal neighbour for frame {target_frame} in {sequence_dir}")
    valid = np.isfinite(target) & (target > 0)
    return np.where(valid, target, 0.0).astype(np.float32), valid


def overlap_diagnostics(
    temporal_depth_m: np.ndarray,
    temporal_valid: np.ndarray,
    raw_depth_m: np.ndarray,
    raw_valid: np.ndarray,
) -> dict[str, float | int | None]:
    overlap = np.asarray(temporal_valid, dtype=bool) & np.asarray(raw_valid, dtype=bool)
    overlap &= np.isfinite(temporal_depth_m) & np.isfinite(raw_depth_m)
    overlap &= (temporal_depth_m > 0) & (raw_depth_m > 0)
    if not overlap.any():
        return {
            "overlap_points": 0,
            "median_abs_error_m": None,
            "median_abs_log_error": None,
            "within_10_percent": None,
        }
    delta = np.abs(temporal_depth_m[overlap] - raw_depth_m[overlap])
    log_delta = np.abs(np.log(temporal_depth_m[overlap] / raw_depth_m[overlap]))
    return {
        "overlap_points": int(overlap.sum()),
        "median_abs_error_m": float(np.median(delta)),
        "median_abs_log_error": float(np.median(log_delta)),
        "within_10_percent": float(np.mean(log_delta <= np.log(1.1))),
    }
