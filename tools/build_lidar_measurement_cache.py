#!/usr/bin/env python3
"""Build the DBH measurement cache for a set of labelled trees.

The original all-in-one script (``DBH_13hight.py``) generated the trunk mask with
VLM + SAM and then derived the measurement row with the DBH protocol.  Its cache
builder was lost, so this script rebuilds the same cache schema from the pieces
that still exist on disk:

* ``mask``        <- the binary ``*_mask.png`` files saved by ``DBH_13hight.py``
                     (prompt and overlay visualizations are ignored)
* ``prior_m``     <- single-frame Velodyne projection with the cached ``P2``/``Tr``
                     calibration (verified to the float16 storage precision of the
                     existing evaluation cache)
* ``prior_valid`` <- projection support mask
* ``disp``        <- frozen Depth Anything V2 relative disparity, same as the
                     evaluation cache
* ``gt_m``        <- the sequence's own depth PNG converted to metres, exactly as
                     the evaluation cache stores it
* ``gt_valid``    <- its validity mask

``measurement_row`` is recorded from the current protocol in ``eval_prior_lidar.py``
so the training samples are generated under the same geometry as the evaluation.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prior_lidar.metrics import depth_metrics
from depth_anything_v2 import build_backbone
from tools.eval_prior_lidar import build_fixed_geometry, read_calibration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=ROOT / "data/sequences/tree_45.txt")
    parser.add_argument("--masks", type=Path, default=ROOT / "outputs/dbh_lidar_batch_results_masks")
    parser.add_argument("--sequences-root", type=Path, default=ROOT / "data/sequences")
    parser.add_argument("--out", type=Path, default=ROOT / "data/dbh_cache_tree50")
    parser.add_argument("--start-line", type=int, default=2,
                        help="1-based first label line to keep, counting the header line as line 1")
    parser.add_argument("--end-line", type=int, default=0, help="1-based last label line (0 = to the end)")
    parser.add_argument("--mask-offset", type=int, default=1,
                        help="mask file number = label line number - this offset (tree ordinal)")
    parser.add_argument("--mask-line-offset", type=int, default=0,
                        help="extra shift applied on top of --mask-offset when locating the mask")
    parser.add_argument("--index-base", default="sequential",
                        help="'sequential' numbers the emitted samples from 1 (idx001, idx002, ...); "
                             "a number starts from that value; 'mask' keeps the VLM+SAM mask numbering")
    parser.add_argument("--mde-size", choices=("vits", "vitb", "vitl"), default="vitl")
    parser.add_argument("--mde-checkpoint", type=Path, default=ROOT / "checkpoints/depth_anything_v2_vitl.pth")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-mde", action=argparse.BooleanOptionalAction, default=False,
                        help="store a zero disparity map instead of running the frozen MDE (debug only)")
    return parser.parse_args()


def parse_name(name: str) -> tuple[int, int]:
    match = re.fullmatch(r"S(\d+)-(\d+)", name.strip())
    if match is None:
        raise ValueError(f"unsupported sample name {name!r}; expected S<seq>-<frame>")
    return int(match.group(1)), int(match.group(2))


def read_labels(path: Path) -> list[tuple[str, str, float]]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").strip().splitlines()):
        if line_no == 0:
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            raise ValueError(f"bad label line: {line!r}")
        rows.append((parts[0].strip(), parts[1].strip(), float(parts[2])))
    return rows


def sanitize_filename(value: str) -> str:
    """Match the filename normalization used by ``DBH_13hight.py``."""

    return re.sub(r"[^0-9a-zA-Z]+", "_", value).strip("_")


def find_mask(masks_dir: Path, mask_number: int, name: str, description: str) -> Path | None:
    """Locate the mask belonging to one labelled tree.

    ``DBH_13hight.py`` writes a binary mask beside prompt and overlay images.  A
    labelled frame can contain several trunks, so all of the sample ordinal,
    frame name, and description must agree before a file is accepted.
    """

    stem = f"{name}_{sanitize_filename(description)}"
    expected = masks_dir / f"{mask_number:03d}_{stem}_mask.png"
    if expected.is_file():
        return expected

    # Permit a cache to be built from a subset manifest, whose sample ordinal
    # differs from tree_62.txt, but never treat prompt/overlay images as masks.
    matches = sorted(masks_dir.glob(f"*_{stem}_mask.png"))
    return matches[0] if len(matches) == 1 else None


def project_lidar(sequence_dir: Path, frame: int, intr: np.ndarray, transform: np.ndarray,
                  height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    points = np.fromfile(sequence_dir / "velodyne" / f"{frame:06d}.bin", dtype=np.float32).reshape(-1, 4)[:, :3]
    homogeneous = np.concatenate([points, np.ones((points.shape[0], 1), np.float32)], axis=1)
    camera = (transform @ homogeneous.T).T[:, :3]
    # The original cache kept returns just in front of the camera: the nearest
    # valid projected point is around 0.16 m, and the affine range depends on it.
    keep = camera[:, 2] > 0.1
    camera = camera[keep]
    projected = (intr @ camera.T).T
    u = projected[:, 0] / projected[:, 2]
    v = projected[:, 1] / projected[:, 2]
    z = camera[:, 2]
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u, v, z = u[inside], v[inside], z[inside]
    depth = np.zeros((height, width), np.float32)
    valid = np.zeros((height, width), np.uint8)
    iu = np.round(u).astype(int).clip(0, width - 1)
    iv = np.round(v).astype(int).clip(0, height - 1)
    # Plain scatter: later points overwrite earlier ones.  This reproduces the
    # original cache bit-for-bit on the evaluation frames, whereas ordering by
    # depth changes which return wins on overlapping pixels and therefore changes
    # the derived affine range.
    depth[iv, iu] = z.astype(np.float32)
    valid[iv, iu] = 1
    return depth, valid


def main() -> None:
    args = parse_args()
    labels = read_labels(args.labels)
    # label line numbers: the header is line 1, so tree ordinal = line - mask_offset
    end = args.end_line if args.end_line > 0 else len(labels) + 1
    selected = [(line, labels[line - 2]) for line in range(args.start_line, end + 1)]
    if args.limit:
        selected = selected[: args.limit]
    args.out.mkdir(parents=True, exist_ok=True)

    backbone = None
    if not args.skip_mde:
        if not args.mde_checkpoint.is_file():
            raise FileNotFoundError(f"frozen MDE checkpoint not found: {args.mde_checkpoint}")
        backbone = build_backbone(depth_size=args.mde_size, encoder_cond_dim=-1).to(args.device).eval()
        payload = torch.load(args.mde_checkpoint, map_location="cpu", weights_only=False)
        state = payload.get("model", payload)
        state = {key.replace("module.", "", 1): value for key, value in state.items()}
        missing, unexpected = backbone.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"MDE load mismatch: missing={missing[:4]} unexpected={unexpected[:4]}")

    records = []
    started = time.time()
    if str(args.index_base) == "mask":
        number_for_line = lambda line: line - args.mask_offset  # noqa: E731
    elif str(args.index_base) == "sequential":
        number_for_line = None
    else:
        base = int(args.index_base)
        number_for_line = lambda line: base + (line - args.start_line)  # noqa: E731
    for offset, (line, (name, description, dbh_cm)) in enumerate(selected):
        index = offset + 1 if number_for_line is None else number_for_line(line)
        sequence_id, frame = parse_name(name)
        sequence = f"{sequence_id:02d}"
        sequence_dir = args.sequences_root / sequence
        mask_path = find_mask(
            args.masks, line - args.mask_offset + args.mask_line_offset, name, description
        )
        if mask_path is None:
            print(f"[skip] idx{index:03d} {name}: no saved mask", flush=True)
            continue
        color_path = sequence_dir / "image_2" / f"{frame:06d}.png"
        depth_path = sequence_dir / "depth" / f"{frame:06d}.png"
        image = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"[skip] idx{index:03d} {name}: missing colour image", flush=True)
            continue
        height, width = image.shape[:2]
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"[skip] idx{index:03d} {name}: unreadable mask", flush=True)
            continue
        if mask.shape != (height, width):
            print(f"[skip] idx{index:03d} {name}: mask shape {mask.shape} != image {image.shape[:2]}", flush=True)
            continue
        raw_depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if raw_depth is None:
            print(f"[skip] idx{index:03d} {name}: missing depth image", flush=True)
            continue
        intrinsics, transform, _distortion = read_calibration(sequence)
        prior, prior_valid = project_lidar(sequence_dir, frame, intrinsics, transform, height, width)
        gt_m = (raw_depth.astype(np.float32) / 1000.0)
        gt_valid = (raw_depth > 0).astype(np.uint8)
        gt_m[gt_valid == 0] = 0.0
        if backbone is not None:
            image_tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)[None].to(args.device)
            with torch.inference_mode():
                disparity = backbone(image_tensor, 518, condition=None, device=args.device)
            disparity = disparity.squeeze().float().cpu().numpy()
        else:
            disparity = np.zeros((height, width), np.float32)
        try:
            geometry = build_fixed_geometry(
                prior, prior_valid > 0, mask > 0, float(intrinsics[0, 0]), float(intrinsics[1, 1])
            )
        except ValueError as error:
            print(f"[skip] idx{index:03d} {name}: geometry failed: {error}", flush=True)
            continue
        stem = f"idx{index:03d}_{name}_{sanitize_filename(description)}"
        np.savez_compressed(
            args.out / f"{stem}.npz",
            img_bgr=image, prior_m=prior, prior_valid=prior_valid, disp=disparity,
            mask=(mask > 0).astype(np.uint8), gt_m=gt_m, gt_valid=gt_valid,
        )
        metadata = {
            "index": int(index), "name": name, "description": description, "dbh_cm": f"{dbh_cm:.3f}",
            "measurement_row": str(int(round(geometry["target_row_raw"]))),
            "label_line": int(line), "tree_number": int(index),
            "color_path": str(color_path.relative_to(ROOT)), "depth_path": str(depth_path.relative_to(ROOT)),
            "mask_path": str(mask_path.resolve()),
            "sequence": sequence, "frame": int(frame),
            "frozen_mde_size": args.mde_size,
            "source": "rebuilt from saved VLM+SAM masks, Velodyne projection and the sequence depth PNG",
            "dbh_labels_read": True, "trunk_masks_read": True, "purpose": "training",
        }
        (args.out / f"{stem}.json").write_text(json.dumps(metadata, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        metrics = depth_metrics(gt_m, gt_m, gt_valid > 0)
        records.append({
            "file": f"{stem}.npz", "index": int(index), "label_line": int(line), "name": name, "sequence": sequence,
            "dbh_cm": dbh_cm, "measurement_row": metadata["measurement_row"],
            "mask_pixels": int((mask > 0).sum()), "prior_pixels": int(prior_valid.sum()),
            "gt_valid_pixels": int(gt_valid.sum()), "mask_pixels_in_slice": int(sum(
                seg[1] - seg[0] + 1 for seg in geometry["row_segments"].values()
            )),
            "base_depth_m": geometry["base_depth_m"], "target_row_raw": geometry["target_row_raw"],
        })
        print(f"[{len(records):02d}] idx{index:03d} {name} row={metadata['measurement_row']} "
              f"mask_px={records[-1]['mask_pixels']} prior_px={records[-1]['prior_pixels']} "
              f"gt_px={records[-1]['gt_valid_pixels']} baseZ={geometry['base_depth_m']:.2f}", flush=True)

    manifest = {
        "schema": "prior_lidar_labeled_train_v1",
        "depth_unit": "m",
        "sample_count": len(records),
        "purpose": "training",
        "label_source": str(args.labels),
        "label_line_range": [args.start_line, end],
        "mask_offset": args.mask_offset, "index_base": str(args.index_base),
        "mask_source": str(args.masks), "mask_line_offset": args.mask_line_offset,
        "frozen_mde_size": args.mde_size,
        "dbh_labels_read": True,
        "trunk_masks_read": True,
        "arrays": {
            "img_bgr": "uint8 BGR [H,W,3]",
            "prior_m": "float32 projected single-frame LiDAR depth in meters [H,W]",
            "prior_valid": "uint8 valid projected LiDAR mask [H,W]",
            "disp": "float32 frozen Depth Anything V2 relative disparity [H,W]",
            "mask": "uint8 VLM+SAM trunk mask [H,W]",
            "gt_m": "float32 sequence depth in meters [H,W]",
            "gt_valid": "uint8 its validity mask [H,W]",
        },
        "measurement_row_source": "current protocol in tools/eval_prior_lidar.py (build_fixed_geometry)",
        "records": records,
    }
    temporary = args.out / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, args.out / "manifest.json")
    print(json.dumps({"out": str(args.out), "samples": len(records), "seconds": round(time.time() - started, 1)}, indent=2))


if __name__ == "__main__":
    main()
