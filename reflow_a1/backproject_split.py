"""Canonical point aggregation and static/dynamic PLY export."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image

from .coarse_align import backproject_depth_to_world
from .dataset_scene import SceneDataError, SceneDatadirDataset
from .export_ply import write_ply


def _resize_mask(mask: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    if mask.shape[:2] == shape:
        return mask.astype(bool)
    img = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    img = img.resize((shape[1], shape[0]), Image.Resampling.NEAREST)
    return np.asarray(img) > 127


def _resize_rgb(rgb: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    if rgb.shape[:2] == shape:
        out = rgb
    else:
        if rgb.dtype != np.uint8:
            rgb_u8 = (np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        else:
            rgb_u8 = rgb
        img = Image.fromarray(rgb_u8, mode="RGB")
        img = img.resize((shape[1], shape[0]), Image.Resampling.BILINEAR)
        out = np.asarray(img)
    if out.dtype == np.uint8:
        return out.astype(np.float32) / 255.0
    return np.clip(out.astype(np.float32), 0.0, 1.0)


def _resize_float(arr: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    if arr.shape[:2] == shape:
        return arr.astype(np.float32)
    img = Image.fromarray(arr.astype(np.float32), mode="F")
    img = img.resize((shape[1], shape[0]), Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.float32)


def _load_scene_point_normalization(scene_root: Path) -> Optional[Dict[str, Any]]:
    scene_json_path = scene_root / "scene.json"
    if not scene_json_path.exists():
        return None
    with scene_json_path.open("r", encoding="utf-8") as f:
        scene_json = json.load(f)
    if "center" not in scene_json:
        return None
    scale = scene_json.get("scale", scene_json.get("scene_to_metric"))
    if scale is None:
        return None
    center = np.asarray(scene_json["center"], dtype=np.float32)
    if center.shape != (3,):
        return None
    return {
        "center": center,
        "scale": float(scale),
        "scene_json": str(scene_json_path),
    }


def _normalize_scene_points(points: np.ndarray, normalization: Dict[str, Any]) -> np.ndarray:
    center = np.asarray(normalization["center"], dtype=np.float32)
    scale = float(normalization["scale"])
    return ((points.astype(np.float32) - center[None, :]) * scale).astype(np.float32)


def _export_pidg_normalized_clouds(
    output_dir: Path,
    dataset: SceneDatadirDataset,
    clouds: Dict[str, Any],
    export_canonical_ply: bool,
    export_static_dynamic_ply: bool,
) -> Optional[Dict[str, Any]]:
    normalization = _load_scene_point_normalization(dataset.scene_root)
    if normalization is None:
        return None

    exported: Dict[str, str] = {}
    if export_canonical_ply:
        path = output_dir / "canonical_complete_pidg_normalized.ply"
        write_ply(path, _normalize_scene_points(clouds["canonical_points"], normalization), clouds["canonical_colors"])
        exported["canonical_complete_pidg_normalized"] = str(path)
    if export_static_dynamic_ply:
        static_path = output_dir / "static_complete_pidg_normalized.ply"
        dynamic_path = output_dir / "dynamic_complete_pidg_normalized.ply"
        write_ply(static_path, _normalize_scene_points(clouds["static_points"], normalization), clouds["static_colors"])
        write_ply(dynamic_path, _normalize_scene_points(clouds["dynamic_points"], normalization), clouds["dynamic_colors"])
        exported["static_complete_pidg_normalized"] = str(static_path)
        exported["dynamic_complete_pidg_normalized"] = str(dynamic_path)

    return {
        "scene_json": normalization["scene_json"],
        "scale": float(normalization["scale"]),
        "center": np.asarray(normalization["center"], dtype=float).tolist(),
        "files": exported,
    }


def _voxel_keys(points: np.ndarray, voxel: float) -> np.ndarray:
    origin = np.nanmin(points, axis=0)
    return np.floor((points - origin) / voxel).astype(np.int64)


def _voxel_downsample(
    points: np.ndarray,
    colors: Optional[np.ndarray],
    voxel: float,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if voxel <= 0 or len(points) == 0:
        return points, colors
    keys = _voxel_keys(points, voxel)
    _, keep = np.unique(keys, axis=0, return_index=True)
    keep = np.sort(keep)
    return points[keep], None if colors is None else colors[keep]


def _voxel_count(points: np.ndarray, voxel: float) -> int:
    if voxel <= 0 or len(points) == 0:
        return int(len(points))
    keys = _voxel_keys(points, voxel)
    return int(np.unique(keys, axis=0).shape[0])


def _adaptive_voxel_downsample(
    points: np.ndarray,
    colors: Optional[np.ndarray],
    target_points: Optional[int],
    rng_seed: int = 1234,
    sample_points: int = 200000,
    max_full_passes: int = 6,
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    """Voxel-thin a large candidate pool to a target budget.

    A single hand-picked voxel size is fragile across scenes because object
    scale and point density vary a lot. We estimate a voxel size from a random
    sample, then only do a few full-cloud passes until the result is at or
    below the requested point budget.
    """

    info: Dict[str, Any] = {
        "method": "adaptive_voxel",
        "target_points": int(target_points) if target_points else None,
        "input_points": int(len(points)),
        "output_points": int(len(points)),
        "voxel_size": None,
        "full_passes": 0,
        "fallback_random": False,
    }
    if target_points is None or target_points <= 0 or len(points) <= target_points:
        info["method"] = "none" if target_points is None or target_points <= 0 else "not_needed"
        return points, colors, info

    finite = np.isfinite(points).all(axis=1)
    if not np.all(finite):
        points = points[finite]
        colors = None if colors is None else colors[finite]
        info["input_points_after_finite_filter"] = int(len(points))
    if len(points) <= target_points:
        info["output_points"] = int(len(points))
        info["method"] = "not_needed_after_finite_filter"
        return points, colors, info

    rng = np.random.default_rng(rng_seed)
    if len(points) > sample_points:
        sample_idx = rng.choice(len(points), size=sample_points, replace=False)
        sample = points[sample_idx]
    else:
        sample = points

    bbox = np.nanmax(sample, axis=0) - np.nanmin(sample, axis=0)
    extent = float(np.nanmax(bbox))
    if not np.isfinite(extent) or extent <= 0:
        keep = rng.choice(len(points), size=target_points, replace=False)
        keep.sort()
        info.update(
            {
                "method": "random_degenerate_extent",
                "output_points": int(len(keep)),
                "fallback_random": True,
            }
        )
        return points[keep], None if colors is None else colors[keep], info

    sample_target = max(1, int(math.ceil(target_points * len(sample) / len(points))))
    lo = max(extent * 1e-8, 1e-12)
    hi = max(extent, lo * 2.0)
    while _voxel_count(sample, hi) > sample_target and hi < extent * 1e6:
        hi *= 2.0

    for _ in range(18):
        mid = (lo + hi) * 0.5
        if _voxel_count(sample, mid) > sample_target:
            lo = mid
        else:
            hi = mid

    voxel = float(hi)
    lower_voxel = 0.0
    upper_voxel: Optional[float] = None
    best_under: Optional[Tuple[float, np.ndarray, Optional[np.ndarray]]] = None
    best_over: Optional[Tuple[float, np.ndarray, Optional[np.ndarray]]] = None
    out_points = points
    out_colors = colors

    for pass_idx in range(max_full_passes):
        out_points, out_colors = _voxel_downsample(points, colors, voxel)
        count = len(out_points)
        info["full_passes"] = pass_idx + 1
        info["voxel_size"] = float(voxel)
        info["output_points"] = int(count)

        if count <= target_points:
            best_under = (voxel, out_points, out_colors)
            if count >= int(0.85 * target_points):
                return out_points, out_colors, info
            upper_voxel = voxel
            if lower_voxel > 0:
                voxel = 0.5 * (lower_voxel + upper_voxel)
            else:
                voxel *= 0.5
        else:
            best_over = (voxel, out_points, out_colors)
            lower_voxel = voxel
            if upper_voxel is not None:
                voxel = 0.5 * (lower_voxel + upper_voxel)
            else:
                # Surface-like clouds often scale closer to 1 / voxel^2 than
                # 1 / voxel^3, so sqrt gives a safer correction than cbrt.
                voxel *= max(1.08, math.sqrt(count / float(target_points)))

    if best_under is not None:
        voxel, out_points, out_colors = best_under
        info["voxel_size"] = float(voxel)
        info["output_points"] = int(len(out_points))
        return out_points, out_colors, info

    if best_over is not None:
        voxel, out_points, out_colors = best_over
        info["voxel_size"] = float(voxel)

    if len(out_points) > target_points:
        keep = rng.choice(len(out_points), size=target_points, replace=False)
        keep.sort()
        out_points = out_points[keep]
        out_colors = None if out_colors is None else out_colors[keep]
        info["fallback_random"] = True
        info["output_points"] = int(len(out_points))
    return out_points, out_colors, info


def _auto_voxel_size(points: np.ndarray, divisor: float = 1200.0, floor: float = 1e-6) -> float:
    if len(points) == 0:
        return float(floor)
    mins = np.nanmin(points, axis=0)
    maxs = np.nanmax(points, axis=0)
    extent = float(np.max(maxs - mins))
    if not np.isfinite(extent) or extent <= 0:
        return float(floor)
    return float(max(extent / max(divisor, 1.0), floor))


def _multiview_depth_consistency_filter(
    points: np.ndarray,
    colors: np.ndarray,
    weights: np.ndarray,
    origins: np.ndarray,
    source_frame_indices: np.ndarray,
    frame_views: list[Dict[str, Any]],
    neighbor_radius: int = 2,
    abs_depth_error: float = 0.02,
    rel_depth_error: float = 0.03,
    min_consistent_views: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    info: Dict[str, Any] = {
        "enabled": bool(neighbor_radius > 0 and len(points) > 0),
        "neighbor_radius": int(max(neighbor_radius, 0)),
        "abs_depth_error": float(abs_depth_error),
        "rel_depth_error": float(rel_depth_error),
        "min_consistent_views": int(max(min_consistent_views, 0)),
        "input_points": int(len(points)),
        "output_points": int(len(points)),
        "kept_ratio": 1.0,
        "checked_points": 0,
        "removed_points": 0,
    }
    if len(points) == 0 or neighbor_radius <= 0:
        return points, colors, weights, origins, source_frame_indices, info

    neighbor_radius = int(max(neighbor_radius, 0))
    min_consistent_views = int(max(min_consistent_views, 0))
    abs_depth_error = float(max(abs_depth_error, 0.0))
    rel_depth_error = float(max(rel_depth_error, 0.0))

    keep = np.zeros(len(points), dtype=bool)
    checked_total = 0

    for src_idx in range(len(frame_views)):
        src_mask = source_frame_indices == src_idx
        if not np.any(src_mask):
            continue
        local_ids = np.flatnonzero(src_mask)
        local_pts = points[local_ids]
        checked = np.zeros(len(local_ids), dtype=np.int16)
        consistent = np.zeros(len(local_ids), dtype=np.int16)

        start = max(0, src_idx - neighbor_radius)
        end = min(len(frame_views), src_idx + neighbor_radius + 1)
        for nbr_idx in range(start, end):
            if nbr_idx == src_idx:
                continue
            view = frame_views[nbr_idx]
            K = view["K"]
            T_cw = view["T_cw"]
            depth = view["depth"]
            valid = view["valid"]
            static = view["static"]
            h, w = depth.shape[:2]

            pts_cam = local_pts @ T_cw[:3, :3].T + T_cw[:3, 3]
            z = pts_cam[:, 2]
            in_front = z > 1e-6
            if not np.any(in_front):
                continue

            x = pts_cam[:, 0] / np.maximum(z, 1e-12)
            y = pts_cam[:, 1] / np.maximum(z, 1e-12)
            u = K[0, 0] * x + K[0, 2]
            v = K[1, 1] * y + K[1, 2]
            ix = np.rint(u).astype(np.int32)
            iy = np.rint(v).astype(np.int32)

            in_bounds = in_front & (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
            if not np.any(in_bounds):
                continue

            sel = np.flatnonzero(in_bounds)
            d_ref = depth[iy[sel], ix[sel]]
            view_ok = valid[iy[sel], ix[sel]] & static[iy[sel], ix[sel]]
            view_ok &= np.isfinite(d_ref) & (d_ref > 0)
            if not np.any(view_ok):
                continue

            checked_sel = sel[view_ok]
            checked[checked_sel] += 1
            d_ref_ok = d_ref[view_ok]
            z_ok = z[checked_sel]
            err = np.abs(d_ref_ok - z_ok)
            tol = np.maximum(abs_depth_error, rel_depth_error * np.maximum(d_ref_ok, z_ok))
            pass_mask = err <= tol
            if np.any(pass_mask):
                consistent[checked_sel[pass_mask]] += 1

        local_keep = (checked == 0) | (consistent >= min_consistent_views)
        keep[local_ids] = local_keep
        checked_total += int(np.count_nonzero(checked > 0))

    if not np.any(keep):
        keep[:] = True
        info["reason"] = "all_points_rejected_fallback_keep_all"

    info["checked_points"] = int(checked_total)
    info["output_points"] = int(np.count_nonzero(keep))
    info["removed_points"] = int(len(keep) - np.count_nonzero(keep))
    info["kept_ratio"] = float(info["output_points"] / max(info["input_points"], 1))
    return (
        points[keep],
        colors[keep],
        weights[keep],
        origins[keep],
        source_frame_indices[keep],
        info,
    )


def _ray_voxel_front_surface_filter(
    points: np.ndarray,
    colors: np.ndarray,
    weights: np.ndarray,
    origins: np.ndarray,
    source_frame_indices: np.ndarray,
    voxel_size: float = 0.0,
    azimuth_bins: int = 24,
    elevation_bins: int = 12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    info: Dict[str, Any] = {
        "enabled": bool(len(points) > 0),
        "input_points": int(len(points)),
        "output_points": int(len(points)),
        "voxel_size": None,
        "azimuth_bins": int(max(azimuth_bins, 1)),
        "elevation_bins": int(max(elevation_bins, 1)),
        "removed_points": 0,
    }
    if len(points) == 0:
        return points, colors, weights, origins, source_frame_indices, info

    azimuth_bins = int(max(azimuth_bins, 1))
    elevation_bins = int(max(elevation_bins, 1))
    voxel_size = float(voxel_size)
    if voxel_size <= 0:
        voxel_size = _auto_voxel_size(points, divisor=1400.0)
    info["voxel_size"] = float(voxel_size)

    rays = points - origins
    ray_depth = np.linalg.norm(rays, axis=1).astype(np.float32)
    valid_depth = ray_depth > 1e-8
    if not np.all(valid_depth):
        points = points[valid_depth]
        colors = colors[valid_depth]
        weights = weights[valid_depth]
        origins = origins[valid_depth]
        source_frame_indices = source_frame_indices[valid_depth]
        rays = rays[valid_depth]
        ray_depth = ray_depth[valid_depth]
    if len(points) == 0:
        info["output_points"] = 0
        info["removed_points"] = info["input_points"]
        return points, colors, weights, origins, source_frame_indices, info

    ray_dir = rays / np.maximum(ray_depth[:, None], 1e-8)
    azimuth = np.arctan2(ray_dir[:, 1], ray_dir[:, 0])
    planar = np.linalg.norm(ray_dir[:, :2], axis=1)
    elevation = np.arctan2(ray_dir[:, 2], np.maximum(planar, 1e-8))
    az_bin = np.clip(((azimuth + np.pi) * (azimuth_bins / (2.0 * np.pi))).astype(np.int32), 0, azimuth_bins - 1)
    el_bin = np.clip(((elevation + 0.5 * np.pi) * (elevation_bins / np.pi)).astype(np.int32), 0, elevation_bins - 1)

    voxel_key = _voxel_keys(points, voxel_size)
    combo = np.column_stack([voxel_key, az_bin[:, None], el_bin[:, None]]).astype(np.int64)
    order = np.lexsort((ray_depth, combo[:, 4], combo[:, 3], combo[:, 2], combo[:, 1], combo[:, 0]))
    sorted_combo = combo[order]
    keep_sorted = np.ones(len(order), dtype=bool)
    keep_sorted[1:] = np.any(sorted_combo[1:] != sorted_combo[:-1], axis=1)
    keep_idx = order[keep_sorted]
    keep_idx.sort()

    info["output_points"] = int(len(keep_idx))
    info["removed_points"] = int(info["input_points"] - len(keep_idx))
    return (
        points[keep_idx],
        colors[keep_idx],
        weights[keep_idx],
        origins[keep_idx],
        source_frame_indices[keep_idx],
        info,
    )


def _voxel_single_surface_aggregate(
    points: np.ndarray,
    colors: np.ndarray,
    weights: np.ndarray,
    voxel_size: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    info: Dict[str, Any] = {
        "enabled": bool(len(points) > 0),
        "input_points": int(len(points)),
        "output_points": int(len(points)),
        "voxel_size": None,
        "method": "weighted_mean",
    }
    if len(points) == 0:
        return points, colors, weights, info

    voxel_size = float(voxel_size)
    if voxel_size <= 0:
        voxel_size = _auto_voxel_size(points, divisor=1000.0)
    info["voxel_size"] = float(voxel_size)

    keys = _voxel_keys(points, voxel_size)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    n_vox = int(inv.max()) + 1 if len(inv) > 0 else 0
    if n_vox <= 0:
        return points, colors, weights, info

    w = np.clip(weights.astype(np.float64), 1e-6, None)
    wsum = np.zeros(n_vox, dtype=np.float64)
    np.add.at(wsum, inv, w)

    p_sum = np.zeros((n_vox, 3), dtype=np.float64)
    np.add.at(p_sum[:, 0], inv, points[:, 0] * w)
    np.add.at(p_sum[:, 1], inv, points[:, 1] * w)
    np.add.at(p_sum[:, 2], inv, points[:, 2] * w)
    out_points = (p_sum / np.maximum(wsum[:, None], 1e-12)).astype(np.float32)

    c_sum = np.zeros((n_vox, 3), dtype=np.float64)
    np.add.at(c_sum[:, 0], inv, colors[:, 0] * w)
    np.add.at(c_sum[:, 1], inv, colors[:, 1] * w)
    np.add.at(c_sum[:, 2], inv, colors[:, 2] * w)
    out_colors = (c_sum / np.maximum(wsum[:, None], 1e-12)).astype(np.float32)
    out_colors = np.clip(out_colors, 0.0, 1.0)

    counts = np.bincount(inv, minlength=n_vox).astype(np.float64)
    out_weights = (wsum / np.maximum(counts, 1.0)).astype(np.float32)

    info["output_points"] = int(len(out_points))
    info["removed_points"] = int(info["input_points"] - len(out_points))
    return out_points, out_colors, out_weights, info


def _voxel_keep_max_weight(
    points: np.ndarray,
    colors: np.ndarray,
    weights: np.ndarray,
    voxel_size: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    info: Dict[str, Any] = {
        "enabled": bool(len(points) > 0),
        "input_points": int(len(points)),
        "output_points": int(len(points)),
        "voxel_size": None,
        "method": "max_weight_per_voxel",
        "removed_points": 0,
    }
    if len(points) == 0:
        return points, colors, weights, info

    voxel_size = float(voxel_size)
    if voxel_size <= 0:
        voxel_size = _auto_voxel_size(points, divisor=1200.0)
    info["voxel_size"] = float(voxel_size)

    keys = _voxel_keys(points, voxel_size)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    w_safe = np.where(np.isfinite(weights), weights, -np.inf).astype(np.float32)
    tie_idx = np.arange(len(points), dtype=np.int64)
    order = np.lexsort((tie_idx, -w_safe, inv))
    sorted_inv = inv[order]
    keep_sorted = np.ones(len(order), dtype=bool)
    keep_sorted[1:] = sorted_inv[1:] != sorted_inv[:-1]
    keep_idx = np.sort(order[keep_sorted])

    out_points = points[keep_idx]
    out_colors = colors[keep_idx]
    out_weights = weights[keep_idx]
    info["output_points"] = int(len(out_points))
    info["removed_points"] = int(info["input_points"] - info["output_points"])
    return out_points, out_colors, out_weights, info


def _points_from_depth_pose(frame: Dict[str, Any], pose: Optional[np.ndarray] = None) -> np.ndarray:
    if frame["depth"] is None:
        raise SceneDataError(
            f"{frame['frame_id']}: no reliable dataset depth is available to synthesize missing pointmaps. "
            "The normal path should use MonST3R alignment pointmaps instead."
        )
    cam = frame["camera_dict"]
    T_wc = cam["T_wc"] if pose is None else pose
    return backproject_depth_to_world(frame["depth"], cam["K"], T_wc)


def _color_diversity_score(colors: np.ndarray, bins: int = 8) -> float:
    if colors.size == 0 or len(colors) < 2:
        return 0.0
    arr = np.clip(colors.reshape(-1, 3).astype(np.float32), 0.0, 1.0)
    std_score = float(np.clip(np.mean(np.std(arr, axis=0)) / 0.35, 0.0, 1.0))
    quant = np.clip((arr * bins).astype(np.int32), 0, bins - 1)
    ids = quant[:, 0] * bins * bins + quant[:, 1] * bins + quant[:, 2]
    counts = np.bincount(ids, minlength=bins**3).astype(np.float64)
    probs = counts[counts > 0] / float(len(ids))
    entropy_denom = math.log(float(min(bins**3, max(len(ids), 2))))
    entropy = 0.0 if entropy_denom <= 0 else float(-(probs * np.log(probs)).sum() / entropy_denom)
    return float(np.clip(0.5 * std_score + 0.5 * entropy, 0.0, 1.0))


def _select_dynamic_reference(
    dynamic_frames: list[Dict[str, Any]],
    coverage_weight: float = 0.6,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    if not dynamic_frames:
        empty = np.empty((0, 3), dtype=np.float32)
        return empty, empty, {
            "mode": "best_reference",
            "selected_frame_id": None,
            "reason": "no_dynamic_points",
            "frame_stats": [],
        }

    max_ratio = max(float(frame["stats"]["dynamic_mask_ratio"]) for frame in dynamic_frames)
    max_diversity = max(float(frame["stats"]["dynamic_color_diversity"]) for frame in dynamic_frames)
    best_idx = 0
    best_score = -1.0
    stats = []
    for idx, frame in enumerate(dynamic_frames):
        frame_stats = dict(frame["stats"])
        coverage_norm = 0.0 if max_ratio <= 0 else frame_stats["dynamic_mask_ratio"] / max_ratio
        diversity_norm = 0.0 if max_diversity <= 0 else frame_stats["dynamic_color_diversity"] / max_diversity
        score = coverage_weight * coverage_norm + (1.0 - coverage_weight) * diversity_norm
        frame_stats["dynamic_reference_score"] = float(score)
        frame_stats["dynamic_mask_ratio_normalized"] = float(coverage_norm)
        frame_stats["dynamic_color_diversity_normalized"] = float(diversity_norm)
        stats.append(frame_stats)
        if score > best_score:
            best_idx = idx
            best_score = score

    selected = dynamic_frames[best_idx]
    best_coverage = max(stats, key=lambda item: item["dynamic_mask_ratio"])
    best_diversity = max(stats, key=lambda item: item["dynamic_color_diversity"])
    metadata = {
        "mode": "best_reference",
        "selected_frame_id": selected["frame_id"],
        "selected_score": float(best_score),
        "coverage_weight": float(coverage_weight),
        "score_formula": "coverage_weight * normalized_dynamic_mask_ratio + (1 - coverage_weight) * normalized_dynamic_color_diversity",
        "best_coverage_frame_id": best_coverage["frame_id"],
        "best_color_diversity_frame_id": best_diversity["frame_id"],
        "frame_stats": stats,
    }
    return selected["points"], selected["colors"], metadata


def _project_rotation_and_scale(matrix: np.ndarray) -> Tuple[np.ndarray, float]:
    U, singular, Vt = np.linalg.svd(matrix.astype(np.float64))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1.0
        R = U @ Vt
    scale = float(np.mean(singular))
    return R.astype(np.float32), scale


def _camera_json_with_pose(raw_camera: Dict[str, Any], pose: np.ndarray) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    pose = np.asarray(pose, dtype=np.float32)
    R, scale = _project_rotation_and_scale(pose[:3, :3])
    T_wc = np.eye(4, dtype=np.float32)
    T_wc[:3, :3] = R
    T_wc[:3, 3] = pose[:3, 3]
    T_cw = np.linalg.inv(T_wc).astype(np.float32)

    out = dict(raw_camera)
    has_orientation_position = "orientation" in out or "position" in out
    if has_orientation_position or not any(
        key in out
        for key in (
            "T_wc",
            "c2w",
            "cam2world",
            "camera_to_world",
            "transform_matrix",
            "T_cw",
            "w2c",
            "world_to_camera",
            "extrinsic",
            "extrinsics",
        )
    ):
        # Nerfies/HyperNeRF-style JSON expects COLMAP world-to-camera rotation
        # in `orientation`, while `position` is the camera center in world space.
        out["orientation"] = T_cw[:3, :3].tolist()
        out["position"] = T_wc[:3, 3].tolist()

    for key in ("T_wc", "c2w", "cam2world", "camera_to_world", "transform_matrix"):
        if key in out:
            out[key] = T_wc.tolist()
    for key in ("T_cw", "w2c", "world_to_camera", "extrinsic", "extrinsics"):
        if key in out:
            out[key] = T_cw.tolist()

    # Add canonical pose matrices even when the source camera used the Nvidia
    # orientation/position format. Downstream readers can still ignore them.
    out.setdefault("T_wc", T_wc.tolist())
    out.setdefault("T_cw", T_cw.tolist())
    meta = {
        "position": T_wc[:3, 3].astype(float).tolist(),
        "rotation_scale_removed": scale,
    }
    return out, meta


def export_optimized_cameras(
    dataset: SceneDatadirDataset,
    alignment_state: Dict[str, Any],
    output_dir: str | Path,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    camera_dir = output_dir / "optimized_camera"
    camera_dir.mkdir(parents=True, exist_ok=True)

    index = []
    for frame_id in alignment_state["frame_ids"]:
        frame = dataset.get_frame_by_id(frame_id)
        record = dataset.get_record_by_id(frame_id)
        pose = alignment_state.get("poses", {}).get(frame_id)
        if pose is None:
            pose = frame["camera_dict"]["T_wc"]
        camera_json, meta = _camera_json_with_pose(frame["camera_dict"]["raw"], np.asarray(pose))
        out_path = camera_dir / record.camera_path.name
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(camera_json, f, indent=2)
        index.append(
            {
                "frame_id": frame_id,
                "camera_path": str(out_path),
                "source_camera_path": str(record.camera_path),
                **meta,
            }
        )

    index_path = output_dir / "optimized_camera_index.json"
    with index_path.open("w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    return {
        "optimized_camera_dir": str(camera_dir),
        "optimized_camera_index": str(index_path),
        "num_optimized_cameras": len(index),
    }


def export_source_cameras(
    dataset: SceneDatadirDataset,
    alignment_state: Dict[str, Any],
    output_dir: str | Path,
    dir_name: str = "source_camera",
) -> Dict[str, Any]:
    """Export the original scene cameras for downstream PIDG-style training.

    The source JSON is intentionally written unchanged. HyperNeRF/PIDG readers
    apply their own resolution ratio and scene normalization, so exporting the
    raw camera file avoids mixing MonST3R's optimized gauge into downstream
    camera loading.
    """

    output_dir = Path(output_dir)
    camera_dir = output_dir / dir_name
    camera_dir.mkdir(parents=True, exist_ok=True)

    index = []
    for frame_id in alignment_state["frame_ids"]:
        frame = dataset.get_frame_by_id(frame_id)
        record = dataset.get_record_by_id(frame_id)
        out_path = camera_dir / record.camera_path.name
        shutil.copy2(record.camera_path, out_path)
        index.append(
            {
                "frame_id": frame_id,
                "camera_path": str(out_path),
                "source_camera_path": str(record.camera_path),
                "position": np.asarray(frame["camera_dict"]["T_wc"][:3, 3], dtype=float).tolist(),
                "source_copy": True,
            }
        )

    index_path = output_dir / f"{dir_name}_index.json"
    with index_path.open("w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    return {
        "source_camera_dir": str(camera_dir),
        "source_camera_index": str(index_path),
        "num_source_cameras": len(index),
    }


def _estimate_similarity(src: np.ndarray, dst: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    if len(src) < 3 or len(dst) < 3:
        return 1.0, np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
    src = src.astype(np.float64)
    dst = dst.astype(np.float64)
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    x = src - mu_src
    y = dst - mu_dst
    cov = (y.T @ x) / max(len(src), 1)
    U, singular, Vt = np.linalg.svd(cov)
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[-1, -1] = -1.0
    R = U @ D @ Vt
    var_src = np.sum(x * x) / max(len(src), 1)
    scale = float(np.sum(singular * np.diag(D)) / max(var_src, 1e-12))
    t = mu_dst - scale * (R @ mu_src)
    return scale, R.astype(np.float32), t.astype(np.float32)


def _set_equal_3d_axes(ax: Any, points: np.ndarray) -> None:
    mins = np.nanmin(points, axis=0)
    maxs = np.nanmax(points, axis=0)
    center = (mins + maxs) * 0.5
    radius = float(np.max(maxs - mins) * 0.5)
    if not np.isfinite(radius) or radius <= 0:
        radius = 1.0
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def visualize_camera_trajectories(
    dataset: SceneDatadirDataset,
    alignment_state: Dict[str, Any],
    output_dir: str | Path,
) -> Dict[str, Any]:
    frame_ids = list(alignment_state["frame_ids"])
    original = []
    optimized = []
    kept_ids = []
    for frame_id in frame_ids:
        pose = alignment_state.get("poses", {}).get(frame_id)
        if pose is None:
            continue
        frame = dataset.get_frame_by_id(frame_id)
        original.append(frame["camera_dict"]["T_wc"][:3, 3])
        optimized.append(np.asarray(pose, dtype=np.float32)[:3, 3])
        kept_ids.append(frame_id)

    output_dir = Path(output_dir)
    plot_path = output_dir / "camera_trajectory_comparison.png"
    meta_path = output_dir / "camera_trajectory_comparison.json"
    if not original:
        metadata = {
            "trajectory_plot": None,
            "reason": "no_camera_poses",
            "frame_ids": [],
        }
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        return metadata

    original_arr = np.asarray(original, dtype=np.float32)
    optimized_arr = np.asarray(optimized, dtype=np.float32)
    scale, R, t = _estimate_similarity(original_arr, optimized_arr)
    original_aligned = (scale * (original_arr @ R.T) + t).astype(np.float32)
    combined = np.concatenate([original_aligned, optimized_arr], axis=0)
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Failed to render 3D camera trajectory plot. "
            "matplotlib with Agg backend is required."
        ) from exc

    fig = plt.figure(figsize=(12, 9), dpi=140)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        original_aligned[:, 0],
        original_aligned[:, 1],
        original_aligned[:, 2],
        c="#3066be",
        s=26,
        alpha=0.9,
        marker="o",
        label="original (Sim(3)-aligned)",
    )
    ax.scatter(
        optimized_arr[:, 0],
        optimized_arr[:, 1],
        optimized_arr[:, 2],
        c="#d65b2d",
        s=26,
        alpha=0.9,
        marker="o",
        label="optimized",
    )

    # Start/end markers as points only (no trajectory lines).
    ax.scatter(
        [original_aligned[0, 0], optimized_arr[0, 0]],
        [original_aligned[0, 1], optimized_arr[0, 1]],
        [original_aligned[0, 2], optimized_arr[0, 2]],
        c="#1c8c4a",
        s=70,
        marker="o",
        edgecolors="black",
        linewidths=0.6,
        label="start",
    )
    ax.scatter(
        [original_aligned[-1, 0], optimized_arr[-1, 0]],
        [original_aligned[-1, 1], optimized_arr[-1, 1]],
        [original_aligned[-1, 2], optimized_arr[-1, 2]],
        c="#a02424",
        s=70,
        marker="o",
        edgecolors="black",
        linewidths=0.6,
        label="end",
    )

    _set_equal_3d_axes(ax, combined)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Camera position comparison in 3D (points only)")
    ax.view_init(elev=24, azim=-58)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)

    metadata = {
        "trajectory_plot": str(plot_path),
        "metadata_path": str(meta_path),
        "frame_ids": kept_ids,
        "num_frames": len(kept_ids),
        "original_aligned_to_optimized_for_plot": True,
        "alignment_scale": float(scale),
        "alignment_rotation": R.tolist(),
        "alignment_translation": t.tolist(),
        "plot_coordinate_system": "3d",
        "plot_style": "points_only",
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return metadata


def split_static_dynamic_points(
    dataset: SceneDatadirDataset,
    alignment_state: Dict[str, Any],
    min_confidence: float = 3.0,
    max_points_per_frame: Optional[int] = None,
    voxel_downsample: float = 0.0,
    static_target_points: Optional[int] = 2_000_000,
    dynamic_reference_mode: str = "best_reference",
    static_mv_neighbor_radius: int = 2,
    static_mv_abs_depth_error: float = 0.02,
    static_mv_rel_depth_error: float = 0.03,
    static_mv_min_consistent_views: int = 1,
    static_front_surface_voxel: float = 0.0,
    static_front_ray_azimuth_bins: int = 24,
    static_front_ray_elevation_bins: int = 12,
    static_surface_aggregate_voxel: float = 0.0,
    fine_voxel_keep_max_conf: bool = False,
    fine_voxel_size: float = 0.0,
    rng_seed: int = 1234,
) -> Dict[str, Any]:
    """Aggregate canonical points and split them by segmentation masks.

    ReFlow uses separate static/dynamic modeling later, so A.1 keeps both
    `P3D,stat` and `P3D,dyn` instead of discarding dynamic regions. Static
    regions are pooled from every frame in canonical/world coordinates, then
    adaptively voxel-thinned. Dynamic initialization defaults to the single
    frame with the best coverage/color-diversity tradeoff.
    """

    rng = np.random.default_rng(rng_seed)
    static_points = []
    static_colors = []
    static_weights = []
    static_origins = []
    static_source_indices = []
    dynamic_frames = []
    frame_views: list[Dict[str, Any]] = []

    frame_ids = list(alignment_state["frame_ids"])
    for frame_seq_idx, frame_id in enumerate(frame_ids):
        frame = dataset.get_frame_by_id(frame_id)
        pose_wc = np.asarray(
            alignment_state.get("poses", {}).get(frame_id, frame["camera_dict"]["T_wc"]),
            dtype=np.float32,
        )
        T_cw = np.linalg.inv(pose_wc).astype(np.float32)
        pointmap = alignment_state.get("global_pointmaps", {}).get(frame_id)
        if pointmap is None:
            pointmap = _points_from_depth_pose(frame, pose_wc)
        pts = np.asarray(pointmap, dtype=np.float32)
        h, w = pts.shape[:2]

        colors = alignment_state.get("colors", {}).get(frame_id)
        if colors is None:
            colors = frame["rgb"]
        colors = _resize_rgb(np.asarray(colors), (h, w))

        dyn_mask = alignment_state.get("dynamic_masks", {}).get(frame_id)
        if dyn_mask is None:
            dyn_mask = frame["dynamic_mask"]
        dyn_mask = _resize_mask(np.asarray(dyn_mask), (h, w))

        valid = np.isfinite(pts).all(axis=-1)
        depth = alignment_state.get("depths", {}).get(frame_id)
        if depth is not None:
            depth = _resize_float(np.asarray(depth), (h, w))
        else:
            depth = (pts @ T_cw[:3, :3].T + T_cw[:3, 3]).astype(np.float32)[..., 2]
        valid &= np.isfinite(depth) & (depth > 0)

        valid_mask = alignment_state.get("valid_masks", {}).get(frame_id)
        if valid_mask is not None:
            valid &= _resize_mask(np.asarray(valid_mask), (h, w))

        conf_map = None
        conf = alignment_state.get("confidence", {}).get(frame_id)
        if conf is not None:
            conf_map = _resize_float(np.asarray(conf), (h, w))
            if min_confidence > 0:
                valid &= conf_map >= min_confidence

        K_view = alignment_state.get("intrinsics", {}).get(frame_id)
        if K_view is None:
            K_view = np.asarray(frame["camera_dict"]["K"], dtype=np.float32).copy()
            src_h = int(frame["camera_dict"].get("height", h))
            src_w = int(frame["camera_dict"].get("width", w))
            if src_h > 0 and src_w > 0 and (src_h != h or src_w != w):
                sx = float(w) / float(src_w)
                sy = float(h) / float(src_h)
                K_view[0, 0] *= sx
                K_view[0, 2] *= sx
                K_view[1, 1] *= sy
                K_view[1, 2] *= sy
        else:
            K_view = np.asarray(K_view, dtype=np.float32)
        frame_views.append(
            {
                "K": K_view,
                "T_cw": T_cw,
                "depth": depth.astype(np.float32),
                "valid": valid.astype(bool),
                "static": (~dyn_mask).astype(bool),
            }
        )

        valid_idx = np.flatnonzero(valid.reshape(-1))
        if max_points_per_frame is not None and max_points_per_frame > 0 and len(valid_idx) > max_points_per_frame:
            valid_idx = rng.choice(valid_idx, size=max_points_per_frame, replace=False)

        pts_flat = pts.reshape(-1, 3)[valid_idx]
        colors_flat = colors.reshape(-1, 3)[valid_idx]
        dyn_flat = dyn_mask.reshape(-1)[valid_idx]
        conf_flat = (
            conf_map.reshape(-1)[valid_idx].astype(np.float32)
            if conf_map is not None
            else np.ones(len(valid_idx), dtype=np.float32)
        )
        dynamic_valid_mask = valid & dyn_mask
        dynamic_colors_for_score = colors[dynamic_valid_mask]

        static_mask = ~dyn_flat
        if np.any(static_mask):
            n_static = int(np.count_nonzero(static_mask))
            static_points.append(pts_flat[static_mask])
            static_colors.append(colors_flat[static_mask])
            static_weights.append(np.clip(conf_flat[static_mask], 1e-3, None))
            static_origins.append(np.repeat(pose_wc[:3, 3][None].astype(np.float32), n_static, axis=0))
            static_source_indices.append(np.full((n_static,), frame_seq_idx, dtype=np.int32))
        if np.any(dyn_flat):
            dynamic_frames.append(
                {
                    "frame_id": frame_id,
                    "points": pts_flat[dyn_flat],
                    "colors": colors_flat[dyn_flat],
                    "stats": {
                        "frame_id": frame_id,
                        "total_pixels": int(h * w),
                        "valid_pixels": int(np.count_nonzero(valid)),
                        "dynamic_mask_pixels": int(np.count_nonzero(dyn_mask)),
                        "dynamic_valid_pixels": int(np.count_nonzero(dynamic_valid_mask)),
                        "dynamic_mask_ratio": float(np.count_nonzero(dyn_mask) / max(h * w, 1)),
                        "dynamic_valid_ratio": float(np.count_nonzero(dynamic_valid_mask) / max(h * w, 1)),
                        "dynamic_color_diversity": _color_diversity_score(dynamic_colors_for_score),
                    },
                }
            )

    def cat_or_empty(chunks: list[np.ndarray], shape: Tuple[int, ...], dtype: np.dtype) -> np.ndarray:
        return np.concatenate(chunks, axis=0) if chunks else np.empty(shape, dtype=dtype)

    stat_pts = cat_or_empty(static_points, (0, 3), np.float32)
    stat_col = cat_or_empty(static_colors, (0, 3), np.float32)
    stat_w = cat_or_empty(static_weights, (0,), np.float32)
    stat_origin = cat_or_empty(static_origins, (0, 3), np.float32)
    stat_src = cat_or_empty(static_source_indices, (0,), np.int32)

    (
        stat_pts,
        stat_col,
        stat_w,
        stat_origin,
        stat_src,
        static_multiview_filter,
    ) = _multiview_depth_consistency_filter(
        stat_pts,
        stat_col,
        stat_w,
        stat_origin,
        stat_src,
        frame_views=frame_views,
        neighbor_radius=static_mv_neighbor_radius,
        abs_depth_error=static_mv_abs_depth_error,
        rel_depth_error=static_mv_rel_depth_error,
        min_consistent_views=static_mv_min_consistent_views,
    )
    (
        stat_pts,
        stat_col,
        stat_w,
        stat_origin,
        stat_src,
        static_front_surface_filter,
    ) = _ray_voxel_front_surface_filter(
        stat_pts,
        stat_col,
        stat_w,
        stat_origin,
        stat_src,
        voxel_size=static_front_surface_voxel,
        azimuth_bins=static_front_ray_azimuth_bins,
        elevation_bins=static_front_ray_elevation_bins,
    )
    stat_pts, stat_col, stat_w, static_surface_aggregation = _voxel_single_surface_aggregate(
        stat_pts,
        stat_col,
        stat_w,
        voxel_size=static_surface_aggregate_voxel,
    )
    if fine_voxel_keep_max_conf:
        stat_pts, stat_col, stat_w, static_voxel_max_conf_filter = _voxel_keep_max_weight(
            stat_pts,
            stat_col,
            stat_w,
            voxel_size=fine_voxel_size,
        )
    else:
        static_voxel_max_conf_filter = {
            "enabled": False,
            "input_points": int(len(stat_pts)),
            "output_points": int(len(stat_pts)),
            "voxel_size": None,
            "method": "max_weight_per_voxel",
            "removed_points": 0,
        }

    if dynamic_reference_mode == "all_frames":
        dyn_pts = cat_or_empty([frame["points"] for frame in dynamic_frames], (0, 3), np.float32)
        dyn_col = cat_or_empty([frame["colors"] for frame in dynamic_frames], (0, 3), np.float32)
        dynamic_reference = {
            "mode": "all_frames",
            "selected_frame_id": None,
            "frame_stats": [frame["stats"] for frame in dynamic_frames],
        }
    elif dynamic_reference_mode == "best_reference":
        dyn_pts, dyn_col, dynamic_reference = _select_dynamic_reference(dynamic_frames)
    else:
        raise ValueError(f"Unknown dynamic_reference_mode: {dynamic_reference_mode}")

    static_downsample = {
        "method": "none",
        "target_points": int(static_target_points) if static_target_points else None,
        "input_points": int(len(stat_pts)),
        "output_points": int(len(stat_pts)),
        "voxel_size": None,
        "full_passes": 0,
        "fallback_random": False,
    }
    if voxel_downsample > 0:
        stat_pts, stat_col = _voxel_downsample(stat_pts, stat_col, voxel_downsample)
        static_downsample.update(
            {
                "method": "fixed_voxel",
                "voxel_size": float(voxel_downsample),
                "output_points": int(len(stat_pts)),
            }
        )
    if static_target_points is not None and static_target_points > 0:
        stat_pts, stat_col, adaptive_info = _adaptive_voxel_downsample(
            stat_pts,
            stat_col,
            target_points=static_target_points,
            rng_seed=rng_seed,
        )
        if adaptive_info["method"] != "not_needed" or static_downsample["method"] == "none":
            static_downsample = adaptive_info
        else:
            static_downsample["output_points"] = int(len(stat_pts))

    dyn_pts, dyn_col = _voxel_downsample(dyn_pts, dyn_col, voxel_downsample)
    canonical_pts = np.concatenate([stat_pts, dyn_pts], axis=0)
    canonical_col = np.concatenate([stat_col, dyn_col], axis=0)
    return {
        "static_points": stat_pts,
        "static_colors": stat_col,
        "dynamic_points": dyn_pts,
        "dynamic_colors": dyn_col,
        "canonical_points": canonical_pts,
        "canonical_colors": canonical_col,
        "static_multiview_filter": static_multiview_filter,
        "static_front_surface_filter": static_front_surface_filter,
        "static_surface_aggregation": static_surface_aggregation,
        "static_voxel_max_conf_filter": static_voxel_max_conf_filter,
        "static_downsample": static_downsample,
        "dynamic_reference": dynamic_reference,
    }


def export_reflow_a1_pointclouds(
    dataset: SceneDatadirDataset,
    alignment_state: Dict[str, Any],
    output_dir: str | Path,
    clip_len: int,
    num_clips: int,
    num_keyframes: int,
    image_size: int,
    use_camera_anchor: bool,
    pair_cache_stats: Dict[str, int],
    camera_anchor_mode: Optional[str] = None,
    export_canonical_ply: bool = True,
    export_static_dynamic_ply: bool = True,
    min_confidence: float = 3.0,
    max_points_per_frame: Optional[int] = None,
    voxel_downsample: float = 0.0,
    static_target_points: Optional[int] = 2_000_000,
    dynamic_reference_mode: str = "best_reference",
    static_mv_neighbor_radius: int = 2,
    static_mv_abs_depth_error: float = 0.02,
    static_mv_rel_depth_error: float = 0.03,
    static_mv_min_consistent_views: int = 1,
    static_front_surface_voxel: float = 0.0,
    static_front_ray_azimuth_bins: int = 24,
    static_front_ray_elevation_bins: int = 12,
    static_surface_aggregate_voxel: float = 0.0,
    fine_voxel_keep_max_conf: bool = False,
    fine_voxel_size: float = 0.0,
    camera_export_mode: str = "optimized",
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_camera_anchor_mode = str(
        camera_anchor_mode
        or alignment_state.get("camera_anchor_mode")
        or ("init" if use_camera_anchor else "none")
    )

    camera_export_mode = str(camera_export_mode).lower()
    if camera_export_mode not in ("optimized", "source", "both"):
        raise ValueError(
            "camera_export_mode must be one of 'optimized', 'source', or 'both', "
            f"got {camera_export_mode!r}"
        )
    optimized_camera_export = None
    source_camera_export = None
    if camera_export_mode in ("optimized", "both"):
        optimized_camera_export = export_optimized_cameras(dataset, alignment_state, output_dir)
    if camera_export_mode in ("source", "both"):
        source_camera_export = export_source_cameras(dataset, alignment_state, output_dir)

    pidg_camera_source = "optimized" if camera_export_mode == "optimized" else "source"
    pidg_camera_export = optimized_camera_export if pidg_camera_source == "optimized" else source_camera_export
    trajectory_export = visualize_camera_trajectories(dataset, alignment_state, output_dir)
    clouds = split_static_dynamic_points(
        dataset,
        alignment_state,
        min_confidence=min_confidence,
        max_points_per_frame=max_points_per_frame,
        voxel_downsample=voxel_downsample,
        static_target_points=static_target_points,
        dynamic_reference_mode=dynamic_reference_mode,
        static_mv_neighbor_radius=static_mv_neighbor_radius,
        static_mv_abs_depth_error=static_mv_abs_depth_error,
        static_mv_rel_depth_error=static_mv_rel_depth_error,
        static_mv_min_consistent_views=static_mv_min_consistent_views,
        static_front_surface_voxel=static_front_surface_voxel,
        static_front_ray_azimuth_bins=static_front_ray_azimuth_bins,
        static_front_ray_elevation_bins=static_front_ray_elevation_bins,
        static_surface_aggregate_voxel=static_surface_aggregate_voxel,
        fine_voxel_keep_max_conf=fine_voxel_keep_max_conf,
        fine_voxel_size=fine_voxel_size,
    )

    if export_canonical_ply:
        write_ply(output_dir / "canonical_complete.ply", clouds["canonical_points"], clouds["canonical_colors"])
    if export_static_dynamic_ply:
        write_ply(output_dir / "static_complete.ply", clouds["static_points"], clouds["static_colors"])
        write_ply(output_dir / "dynamic_complete.ply", clouds["dynamic_points"], clouds["dynamic_colors"])
    pidg_normalized_export = _export_pidg_normalized_clouds(
        output_dir=output_dir,
        dataset=dataset,
        clouds=clouds,
        export_canonical_ply=export_canonical_ply,
        export_static_dynamic_ply=export_static_dynamic_ply,
    )

    summary = {
        "scene_root": str(dataset.scene_root),
        "num_train_frames": len(alignment_state["frame_ids"]),
        "num_clips": int(num_clips),
        "num_keyframes": int(num_keyframes),
        "num_points_total": int(len(clouds["canonical_points"])),
        "num_points_static": int(len(clouds["static_points"])),
        "num_points_dynamic": int(len(clouds["dynamic_points"])),
        "clip_len": int(clip_len),
        "image_resize": int(image_size),
        "use_camera_anchor": bool(use_camera_anchor),
        "camera_anchor_mode": resolved_camera_anchor_mode,
        "poses_fixed_to_source": bool(alignment_state.get("poses_fixed_to_source", False)),
        "pair_cache_hits": int(pair_cache_stats.get("hits", 0)),
        "pair_cache_misses": int(pair_cache_stats.get("misses", 0)),
        "pair_cache_writes": int(pair_cache_stats.get("writes", 0)),
        "pair_cache_invalid": int(pair_cache_stats.get("invalid", 0)),
        "min_confidence": float(min_confidence),
        "voxel_downsample": float(voxel_downsample),
        "static_target_points": int(static_target_points) if static_target_points else None,
        "static_multiview_filter": clouds["static_multiview_filter"],
        "static_front_surface_filter": clouds["static_front_surface_filter"],
        "static_surface_aggregation": clouds["static_surface_aggregation"],
        "static_voxel_max_conf_filter": clouds["static_voxel_max_conf_filter"],
        "static_downsample": clouds["static_downsample"],
        "fine_voxel_keep_max_conf": bool(fine_voxel_keep_max_conf),
        "fine_voxel_size": float(fine_voxel_size),
        "dynamic_reference_mode": dynamic_reference_mode,
        "dynamic_reference": {
            key: value
            for key, value in clouds["dynamic_reference"].items()
            if key != "frame_stats"
        },
        "camera_export_mode": camera_export_mode,
        "pidg_camera_source": pidg_camera_source,
        "pidg_camera_dir": (
            pidg_camera_export["optimized_camera_dir"]
            if pidg_camera_source == "optimized"
            else pidg_camera_export["source_camera_dir"]
        ),
        "pidg_camera_index": (
            pidg_camera_export["optimized_camera_index"]
            if pidg_camera_source == "optimized"
            else pidg_camera_export["source_camera_index"]
        ),
        "optimized_camera_dir": None
        if optimized_camera_export is None
        else optimized_camera_export["optimized_camera_dir"],
        "optimized_camera_index": None
        if optimized_camera_export is None
        else optimized_camera_export["optimized_camera_index"],
        "source_camera_dir": None
        if source_camera_export is None
        else source_camera_export["source_camera_dir"],
        "source_camera_index": None
        if source_camera_export is None
        else source_camera_export["source_camera_index"],
        "camera_trajectory_plot": trajectory_export.get("trajectory_plot"),
        "camera_trajectory_metadata": trajectory_export.get("metadata_path"),
        "max_points_per_frame": int(max_points_per_frame) if max_points_per_frame else None,
        "source_camera_alignment": alignment_state.get("source_camera_alignment"),
        "pidg_normalized_pointclouds": pidg_normalized_export,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with (output_dir / "dynamic_reference_stats.json").open("w", encoding="utf-8") as f:
        json.dump(clouds["dynamic_reference"], f, indent=2)
    return summary
