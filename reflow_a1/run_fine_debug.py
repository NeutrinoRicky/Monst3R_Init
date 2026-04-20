"""Run coarse + overlap fine alignment and export global-registry debug artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from reflow_a1.backproject_split import export_optimized_cameras, visualize_camera_trajectories
    from reflow_a1.coarse_align import run_coarse_alignment
    from reflow_a1.dataset_scene import SceneDatadirDataset
    from reflow_a1.export_ply import write_ply
    from reflow_a1.fine_align import merge_clip_states, run_fine_alignment
    from reflow_a1.pair_infer import PairwiseInferencer
    from reflow_a1.pair_sampler import (
        PairGraph,
        build_coarse_pairs,
        build_fine_pairs_for_clip,
        build_reflow_a1_pair_graph,
    )
    from reflow_a1.run_coarse_debug import (
        _ensure_writable_evo_home,
        _maybe_cap,
        export_coarse_keyframe_pointcloud,
        export_coarse_pose_errors,
    )
else:
    from .backproject_split import export_optimized_cameras, visualize_camera_trajectories
    from .coarse_align import run_coarse_alignment
    from .dataset_scene import SceneDatadirDataset
    from .export_ply import write_ply
    from .fine_align import merge_clip_states, run_fine_alignment
    from .pair_infer import PairwiseInferencer
    from .pair_sampler import (
        PairGraph,
        build_coarse_pairs,
        build_fine_pairs_for_clip,
        build_reflow_a1_pair_graph,
    )
    from .run_coarse_debug import (
        _ensure_writable_evo_home,
        _maybe_cap,
        export_coarse_keyframe_pointcloud,
        export_coarse_pose_errors,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Debug runner: coarse + overlap fine global-registry alignment")
    parser.add_argument("--scene_root", required=True, type=str, help="Path to scene_datadir")
    parser.add_argument("--split", default="train", type=str, help="dataset.json split prefix")
    parser.add_argument("--clip_len", default=10, type=int, help="Number of train frames per clip")
    parser.add_argument(
        "--first_clip_index",
        default=0,
        type=int,
        help="First overlap clip index to fine-optimize (default: 0, supports negative index)",
    )
    parser.add_argument(
        "--overlap_frames",
        default=2,
        type=int,
        help="Number of frames shared by adjacent fine debug clips (0 keeps the old non-overlap graph)",
    )
    parser.add_argument(
        "--num_fine_debug_clips",
        default=2,
        type=int,
        help="Number of sequential fine clips to debug/update into the global registry",
    )
    parser.add_argument(
        "--coarse_max_offset",
        default=2,
        type=int,
        help="Maximum coarse keyframe edge offset (2 means i->i+1 and i->i+2; 3 adds i->i+3)",
    )
    parser.add_argument(
        "--fine_max_offset",
        default=4,
        type=int,
        help="Maximum intra-clip fine edge offset (4 adds i->i+1..i+4 inside each clip)",
    )
    parser.add_argument("--image_size", default=512, type=int, choices=[224, 512], help="MonST3R input size")
    parser.add_argument("--weights", default=None, type=str, help="Optional local MonST3R checkpoint path")
    parser.add_argument(
        "--model_name",
        default="Junyi42/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt",
        type=str,
        help="Hugging Face model id if --weights/default checkpoint is unavailable",
    )
    parser.add_argument("--device", default=None, type=str, help="Torch device, defaults to cuda when available")
    parser.add_argument("--batch_size", default=8, type=int, help="Pairwise inference batch size")

    parser.add_argument("--niter_coarse", default=1200, type=int, help="Coarse global alignment iterations")
    parser.add_argument("--schedule", default="linear", type=str, help="Coarse-stage MonST3R LR schedule")
    parser.add_argument("--lr", default=0.005, type=float, help="Coarse-stage alignment learning rate")
    parser.add_argument("--niter_fine", default=300, type=int, help="Fine clip alignment iterations")
    parser.add_argument(
        "--schedule_fine",
        default=None,
        type=str,
        help="Fine-stage MonST3R LR schedule (defaults to --schedule)",
    )
    parser.add_argument(
        "--lr_fine",
        default=None,
        type=float,
        help="Fine-stage alignment learning rate (defaults to --lr)",
    )
    parser.add_argument(
        "--fine_temporal_smoothing_weight",
        default=0.0,
        type=float,
        help="Fine-stage temporal camera smoothing loss weight",
    )
    parser.add_argument(
        "--fine_camera_pose_prior_weight",
        default=0.05,
        type=float,
        help="Fine-stage camera pose prior loss weight vs source cameras",
    )
    parser.add_argument(
        "--fine_camera_intrinsics_prior_weight",
        default=0.02,
        type=float,
        help="Fine-stage camera intrinsics prior loss weight vs source cameras",
    )
    parser.add_argument(
        "--fine_camera_pose_prior_translation_weight",
        default=1.0,
        type=float,
        help="Translation multiplier inside fine pose-prior loss term",
    )
    parser.add_argument(
        "--fine_camera_anchor_mode",
        default="fixed",
        choices=["none", "init", "fixed"],
        help="Fine-stage camera anchor mode; fixed is recommended for stable debug registry updates",
    )
    parser.add_argument(
        "--fine_coarse_pointmap_teacher_weight",
        default=0.1,
        type=float,
        help="Fine-stage coarse/canonical pointmap teacher loss weight for keyframe and overlap anchors",
    )
    parser.add_argument(
        "--fine_coarse_depth_teacher_weight",
        default=0.05,
        type=float,
        help="Fine-stage coarse/canonical depth teacher loss weight for keyframe and overlap anchors",
    )
    parser.add_argument(
        "--fine_coarse_teacher_min_confidence",
        default=3.0,
        type=float,
        help="Minimum anchor confidence for coarse/canonical geometry teacher pixels",
    )

    parser.add_argument("--force_recompute_pairs", action="store_true", help="Ignore pair cache and recompute")
    parser.add_argument("--max_frames", default=None, type=int, help="Optional debugging cap on split frames")
    parser.add_argument(
        "--min_confidence",
        default=3.0,
        type=float,
        help="Confidence threshold used when exporting debug PLYs",
    )
    parser.add_argument(
        "--max_points_per_frame",
        default=200000,
        type=int,
        help="Optional random cap per frame for debug PLY export (<=0 disables)",
    )
    parser.add_argument(
        "--dynamic_min_confidence",
        default=5.0,
        type=float,
        help="Stricter confidence threshold for dynamic-buffer point extraction",
    )
    parser.add_argument(
        "--coarse_voxel_keep_max_conf",
        action="store_true",
        help="Voxelize coarse debug cloud and keep max-confidence point per voxel",
    )
    parser.add_argument(
        "--coarse_voxel_size",
        default=0.0,
        type=float,
        help="Voxel size for --coarse_voxel_keep_max_conf (<=0 auto)",
    )
    parser.add_argument(
        "--fine_voxel_keep_max_conf",
        action="store_true",
        help="Kept for CLI compatibility; global static registry uses --global_static_voxel_size",
    )
    parser.add_argument(
        "--fine_voxel_size",
        default=0.0,
        type=float,
        help="Compatibility voxel size; used as global static voxel size only if --global_static_voxel_size is unset",
    )
    parser.add_argument(
        "--global_static_voxel_size",
        default=0.0,
        type=float,
        help="Voxel size for the debug GlobalStaticMap fusion pass (<=0 auto from coarse/static points)",
    )
    parser.add_argument("--seed", default=1234, type=int, help="Random seed for debug point sampling")
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help="Defaults to scene_root/monst3r_reflow_a1_fine_debug",
    )
    parser.add_argument("--dry_run", action="store_true", help="Only load data and build pair graph")
    return parser


def _resolve_clip_index(num_clips: int, clip_index: int) -> int:
    if num_clips <= 0:
        raise RuntimeError("No clips found; cannot run fine debugging")
    idx = int(clip_index)
    if idx < 0:
        idx += num_clips
    if idx < 0 or idx >= num_clips:
        raise ValueError(f"first_clip_index={clip_index} is out of range for {num_clips} clip(s)")
    return idx


def _safe_trajectory_export(
    dataset: SceneDatadirDataset,
    alignment_state: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    try:
        return visualize_camera_trajectories(dataset, alignment_state, output_dir)
    except Exception as exc:  # pragma: no cover
        return {"trajectory_plot": None, "error": str(exc)}


def _split_into_overlapping_clips(
    frame_ids: Sequence[str],
    clip_len: int,
    overlap_frames: int,
) -> List[List[str]]:
    if clip_len <= 0:
        raise ValueError(f"clip_len must be positive, got {clip_len}")
    overlap_frames = max(int(overlap_frames), 0)
    if overlap_frames <= 0:
        return [list(frame_ids[i : i + clip_len]) for i in range(0, len(frame_ids), clip_len)]
    if overlap_frames >= clip_len:
        raise ValueError(
            f"overlap_frames must be smaller than clip_len; got overlap_frames={overlap_frames}, clip_len={clip_len}"
        )

    stride = clip_len - overlap_frames
    clips: List[List[str]] = []
    start = 0
    frame_ids = list(frame_ids)
    while start < len(frame_ids):
        clips.append(list(frame_ids[start : start + clip_len]))
        if start + clip_len >= len(frame_ids):
            break
        start += stride
    return clips


def _build_fine_debug_pair_graph(
    frame_ids: Sequence[str],
    clip_len: int,
    overlap_frames: int,
    coarse_max_offset: int,
    fine_max_offset: int,
) -> PairGraph:
    if int(overlap_frames) <= 0:
        return build_reflow_a1_pair_graph(
            frame_ids,
            clip_len=clip_len,
            coarse_max_offset=coarse_max_offset,
            fine_max_offset=fine_max_offset,
        )

    clips = _split_into_overlapping_clips(
        frame_ids,
        clip_len=clip_len,
        overlap_frames=overlap_frames,
    )
    keyframes = [clip[0] for clip in clips if clip]
    coarse_pairs = build_coarse_pairs(keyframes, coarse_max_offset=coarse_max_offset)
    fine_pairs_per_clip = [build_fine_pairs_for_clip(clip, fine_max_offset=fine_max_offset) for clip in clips]
    return PairGraph(
        clips=clips,
        keyframes=keyframes,
        coarse_pairs=coarse_pairs,
        fine_pairs_per_clip=fine_pairs_per_clip,
    )


def _resolve_debug_clip_indices(num_clips: int, first_clip_index: int, num_debug_clips: int) -> List[int]:
    start = _resolve_clip_index(num_clips, first_clip_index)
    count = max(int(num_debug_clips), 1)
    return list(range(start, min(start + count, num_clips)))


def _build_clip_overlap_graph(clips: Sequence[Sequence[str]]) -> Dict[str, Any]:
    adjacent_pairs: List[Dict[str, Any]] = []
    for idx in range(max(len(clips) - 1, 0)):
        shared = sorted(set(clips[idx]).intersection(clips[idx + 1]))
        adjacent_pairs.append(
            {
                "clip_index_a": int(idx),
                "clip_index_b": int(idx + 1),
                "shared_frame_ids": shared,
                "num_shared_frames": int(len(shared)),
                "shared_anchor_ids_sample": [],
                "num_shared_anchor_ids": 0,
            }
        )
    return {
        "adjacent_clip_pairs": adjacent_pairs,
        "num_adjacent_pairs": int(len(adjacent_pairs)),
        "num_pairs_with_shared_frames": int(sum(1 for item in adjacent_pairs if item["num_shared_frames"] > 0)),
    }


def _stable_frame_seed(base_seed: int, frame_id: str, salt: int = 0) -> int:
    value = int(base_seed) + int(salt) * 1_000_003
    for idx, char in enumerate(str(frame_id)):
        value += (idx + 1) * ord(char)
    return int(value % (2**32 - 1))


def _resize_nearest(arr: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.shape[:2] == shape:
        return arr
    if arr.ndim < 2:
        return arr
    src_h, src_w = arr.shape[:2]
    if src_h <= 0 or src_w <= 0:
        return arr
    yy = np.clip(np.round(np.linspace(0, src_h - 1, shape[0])).astype(np.int64), 0, src_h - 1)
    xx = np.clip(np.round(np.linspace(0, src_w - 1, shape[1])).astype(np.int64), 0, src_w - 1)
    return arr[yy[:, None], xx[None, :]]


def _as_float_colors(colors: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    colors = _resize_nearest(np.asarray(colors), shape)
    colors = np.asarray(colors)
    if colors.ndim == 2:
        colors = np.repeat(colors[..., None], 3, axis=-1)
    colors = colors[..., :3].astype(np.float32)
    finite = colors[np.isfinite(colors)]
    if colors.dtype == np.uint8 or (finite.size > 0 and float(np.max(finite)) > 1.001):
        colors = colors / 255.0
    return np.clip(colors, 0.0, 1.0)


def _as_bool_mask(mask: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    mask = np.asarray(mask)
    while mask.ndim > 2 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    while mask.ndim > 2 and mask.shape[0] == 1:
        mask = mask[0]
    mask = _resize_nearest(mask, shape)
    return mask.astype(bool)


def _as_float_map(values: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    values = np.asarray(values)
    while values.ndim > 2 and values.shape[-1] == 1:
        values = values[..., 0]
    while values.ndim > 2 and values.shape[0] == 1:
        values = values[0]
    values = _resize_nearest(values, shape)
    return values.astype(np.float32)


def _pose_error_vs_source(
    dataset: SceneDatadirDataset,
    alignment_state: Dict[str, Any],
    frame_id: str,
) -> Tuple[float, float]:
    pose = alignment_state.get("poses", {}).get(frame_id)
    if pose is None:
        return 0.0, 0.0
    pose = np.asarray(pose, dtype=np.float32)
    pose_src = np.asarray(dataset.get_frame_by_id(frame_id)["camera_dict"]["T_wc"], dtype=np.float32)
    t_err = float(np.linalg.norm(pose[:3, 3] - pose_src[:3, 3]))
    r_delta = pose_src[:3, :3].T @ pose[:3, :3]
    cos_theta = float(np.clip((np.trace(r_delta) - 1.0) * 0.5, -1.0, 1.0))
    r_err = float(np.degrees(np.arccos(cos_theta)))
    return t_err, r_err


def _extract_frame_observations(
    dataset: SceneDatadirDataset,
    alignment_state: Dict[str, Any],
    frame_id: str,
    min_confidence: float,
    dynamic_min_confidence: float,
    max_points_per_frame: int | None,
    seed: int,
) -> Dict[str, Any]:
    pointmap = alignment_state.get("global_pointmaps", {}).get(frame_id)
    if pointmap is None:
        return {
            "frame_id": frame_id,
            "points_total": 0,
            "points_valid": 0,
            "points_sampled": 0,
            "pose_translation_error_vs_source": 0.0,
            "pose_rotation_error_vs_source_deg": 0.0,
            "static_points": np.empty((0, 3), dtype=np.float32),
            "static_colors": np.empty((0, 3), dtype=np.float32),
            "static_confidence": np.empty((0,), dtype=np.float32),
            "dynamic_points": np.empty((0, 3), dtype=np.float32),
            "dynamic_colors": np.empty((0, 3), dtype=np.float32),
            "dynamic_confidence": np.empty((0,), dtype=np.float32),
        }

    frame = dataset.get_frame_by_id(frame_id)
    pts_hw = np.asarray(pointmap, dtype=np.float32)
    h, w = pts_hw.shape[:2]
    shape = (h, w)
    pts = pts_hw.reshape(-1, 3)
    valid = np.isfinite(pts_hw).all(axis=-1)

    valid_mask = alignment_state.get("valid_masks", {}).get(frame_id)
    if valid_mask is not None:
        valid &= _as_bool_mask(np.asarray(valid_mask), shape)

    conf_map = None
    conf = alignment_state.get("confidence", {}).get(frame_id)
    if conf is not None:
        conf_map = _as_float_map(np.asarray(conf), shape)
        max_conf = float(np.nanmax(conf_map)) if np.isfinite(conf_map).any() else 0.0
        if max_conf <= 1.0 and min_confidence > 1.0:
            valid &= conf_map > 0.5
        elif min_confidence > 0:
            valid &= conf_map >= float(min_confidence)

    dyn_mask = alignment_state.get("dynamic_masks", {}).get(frame_id)
    if dyn_mask is None:
        dyn_mask = frame["dynamic_mask"]
    dyn_mask = _as_bool_mask(np.asarray(dyn_mask), shape)

    colors = alignment_state.get("colors", {}).get(frame_id)
    if colors is None:
        colors = frame["rgb"]
    colors_hw = _as_float_colors(np.asarray(colors), shape)

    valid_idx = np.flatnonzero(valid.reshape(-1))
    points_valid = int(len(valid_idx))
    if max_points_per_frame is not None and max_points_per_frame > 0 and len(valid_idx) > max_points_per_frame:
        rng = np.random.default_rng(_stable_frame_seed(seed, frame_id))
        valid_idx = np.sort(rng.choice(valid_idx, size=int(max_points_per_frame), replace=False))

    dyn_flat = dyn_mask.reshape(-1)[valid_idx]
    pts_keep = pts[valid_idx].astype(np.float32)
    colors_keep = colors_hw.reshape(-1, 3)[valid_idx].astype(np.float32)
    if conf_map is None:
        conf_keep = np.ones((len(valid_idx),), dtype=np.float32)
    else:
        conf_keep = conf_map.reshape(-1)[valid_idx].astype(np.float32)

    static_sel = ~dyn_flat
    dynamic_sel = dyn_flat & (conf_keep >= float(dynamic_min_confidence))
    pose_translation_error, pose_rotation_error = _pose_error_vs_source(dataset, alignment_state, frame_id)
    return {
        "frame_id": frame_id,
        "points_total": int(len(pts)),
        "points_valid": points_valid,
        "points_sampled": int(len(valid_idx)),
        "pose_translation_error_vs_source": float(pose_translation_error),
        "pose_rotation_error_vs_source_deg": float(pose_rotation_error),
        "static_points": pts_keep[static_sel],
        "static_colors": colors_keep[static_sel],
        "static_confidence": conf_keep[static_sel],
        "dynamic_points": pts_keep[dynamic_sel],
        "dynamic_colors": colors_keep[dynamic_sel],
        "dynamic_confidence": conf_keep[dynamic_sel],
    }


def _auto_voxel_size(points: np.ndarray, divisor: float = 900.0, floor: float = 1e-6) -> float:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if len(points) == 0:
        return float(floor)
    mins = np.nanmin(points, axis=0)
    maxs = np.nanmax(points, axis=0)
    extent = float(np.max(maxs - mins))
    if not np.isfinite(extent) or extent <= 0:
        return float(floor)
    return float(max(extent / max(divisor, 1.0), floor))


def _estimate_static_voxel_size(
    dataset: SceneDatadirDataset,
    coarse_state: Dict[str, Any],
    requested_voxel_size: float,
    min_confidence: float,
    dynamic_min_confidence: float,
    max_points_per_frame: int | None,
    seed: int,
) -> float:
    requested_voxel_size = float(requested_voxel_size)
    if requested_voxel_size > 0:
        return requested_voxel_size

    chunks: List[np.ndarray] = []
    for frame_id in coarse_state.get("frame_ids", []):
        obs = _extract_frame_observations(
            dataset=dataset,
            alignment_state=coarse_state,
            frame_id=frame_id,
            min_confidence=min_confidence,
            dynamic_min_confidence=dynamic_min_confidence,
            max_points_per_frame=max_points_per_frame,
            seed=seed,
        )
        if len(obs["static_points"]) > 0:
            chunks.append(obs["static_points"])
    if not chunks:
        return 1e-3
    return _auto_voxel_size(np.concatenate(chunks, axis=0))


def _voxel_reduce_max_confidence(
    points: np.ndarray,
    colors: np.ndarray,
    confidence: np.ndarray,
    voxel_size: float,
    origin: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    if len(points) == 0:
        return (
            np.empty((0, 3), dtype=np.int64),
            points,
            colors,
            confidence,
            0,
        )
    keys = np.floor((points - origin[None, :]) / float(voxel_size)).astype(np.int64)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    conf_safe = np.where(np.isfinite(confidence), confidence, -np.inf).astype(np.float32)
    order = np.lexsort((np.arange(len(points), dtype=np.int64), -conf_safe, inv))
    sorted_inv = inv[order]
    keep_sorted = np.ones(len(order), dtype=bool)
    keep_sorted[1:] = sorted_inv[1:] != sorted_inv[:-1]
    keep = np.sort(order[keep_sorted])
    return keys[keep], points[keep], colors[keep], confidence[keep], int(len(points) - len(keep))


class DebugGlobalStaticMap:
    """Small voxel-hash static map for validating the new fine-stage data flow."""

    def __init__(self, voxel_size: float):
        self.voxel_size = float(voxel_size)
        self.origin: np.ndarray | None = None
        self.records: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
        self.update_log: List[Dict[str, Any]] = []

    def update(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        confidence: np.ndarray,
        frame_id: str,
        clip_index: int | None,
        source_type: str,
    ) -> Dict[str, Any]:
        points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        colors = np.asarray(colors, dtype=np.float32).reshape(-1, 3)
        confidence = np.asarray(confidence, dtype=np.float32).reshape(-1)
        if len(points) == 0:
            meta = {
                "frame_id": frame_id,
                "clip_index": None if clip_index is None else int(clip_index),
                "source_type": source_type,
                "input_points": 0,
                "candidate_voxels": 0,
                "new_anchors": 0,
                "fused_anchors": 0,
                "suppressed_by_voxel": 0,
            }
            self.update_log.append(meta)
            return meta

        valid = np.isfinite(points).all(axis=1)
        points = points[valid]
        colors = colors[valid]
        confidence = confidence[valid]
        if self.origin is None and len(points) > 0:
            self.origin = np.nanmin(points, axis=0).astype(np.float32)
        if self.origin is None:
            self.origin = np.zeros((3,), dtype=np.float32)

        keys, pts, cols, conf, suppressed = _voxel_reduce_max_confidence(
            points=points,
            colors=colors,
            confidence=confidence,
            voxel_size=self.voxel_size,
            origin=self.origin,
        )
        new_anchors = 0
        fused_anchors = 0
        for key_arr, point, color, conf_value in zip(keys, pts, cols, conf):
            key = tuple(int(v) for v in key_arr)
            weight = float(max(conf_value, 1e-3))
            record = self.records.get(key)
            if record is None:
                record = {
                    "anchor_id": f"anchor_{len(self.records):08d}",
                    "point": point.astype(np.float64),
                    "color": color.astype(np.float64),
                    "weight": weight,
                    "confidence": float(conf_value),
                    "observations": 1,
                    "frame_ids": {frame_id},
                    "clip_indices": set() if clip_index is None else {int(clip_index)},
                    "source_types": {source_type},
                }
                self.records[key] = record
                new_anchors += 1
                continue

            prev_weight = float(record["weight"])
            total_weight = prev_weight + weight
            alpha = weight / max(total_weight, 1e-6)
            record["point"] = (1.0 - alpha) * record["point"] + alpha * point.astype(np.float64)
            record["color"] = (1.0 - alpha) * record["color"] + alpha * color.astype(np.float64)
            record["weight"] = total_weight
            record["confidence"] = float(max(float(record["confidence"]), float(conf_value)))
            record["observations"] = int(record["observations"]) + 1
            record["frame_ids"].add(frame_id)
            if clip_index is not None:
                record["clip_indices"].add(int(clip_index))
            record["source_types"].add(source_type)
            fused_anchors += 1

        meta = {
            "frame_id": frame_id,
            "clip_index": None if clip_index is None else int(clip_index),
            "source_type": source_type,
            "input_points": int(len(points)),
            "candidate_voxels": int(len(keys)),
            "new_anchors": int(new_anchors),
            "fused_anchors": int(fused_anchors),
            "suppressed_by_voxel": int(suppressed),
        }
        self.update_log.append(meta)
        return meta

    def update_from_state(
        self,
        dataset: SceneDatadirDataset,
        alignment_state: Dict[str, Any],
        frame_ids: Sequence[str],
        clip_index: int | None,
        source_type: str,
        min_confidence: float,
        dynamic_min_confidence: float,
        max_points_per_frame: int | None,
        seed: int,
    ) -> Dict[str, Any]:
        per_frame = []
        dynamic_points_seen = 0
        for frame_id in frame_ids:
            obs = _extract_frame_observations(
                dataset=dataset,
                alignment_state=alignment_state,
                frame_id=frame_id,
                min_confidence=min_confidence,
                dynamic_min_confidence=dynamic_min_confidence,
                max_points_per_frame=max_points_per_frame,
                seed=seed,
            )
            meta = self.update(
                points=obs["static_points"],
                colors=obs["static_colors"],
                confidence=obs["static_confidence"],
                frame_id=frame_id,
                clip_index=clip_index,
                source_type=source_type,
            )
            dynamic_points_seen += int(len(obs["dynamic_points"]))
            per_frame.append(
                {
                    **meta,
                    "points_total": int(obs["points_total"]),
                    "points_valid": int(obs["points_valid"]),
                    "points_sampled": int(obs["points_sampled"]),
                    "dynamic_points_buffered_elsewhere": int(len(obs["dynamic_points"])),
                }
            )
        return {
            "source_type": source_type,
            "clip_index": None if clip_index is None else int(clip_index),
            "num_frames": int(len(frame_ids)),
            "static_input_points": int(sum(item["input_points"] for item in per_frame)),
            "dynamic_points_seen": int(dynamic_points_seen),
            "new_anchors": int(sum(item["new_anchors"] for item in per_frame)),
            "fused_anchors": int(sum(item["fused_anchors"] for item in per_frame)),
            "suppressed_by_voxel": int(sum(item["suppressed_by_voxel"] for item in per_frame)),
            "per_frame": per_frame,
        }

    def arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.records:
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)
        records = sorted(self.records.values(), key=lambda item: item["anchor_id"])
        points = np.stack([np.asarray(item["point"], dtype=np.float32) for item in records], axis=0)
        colors = np.stack([np.asarray(item["color"], dtype=np.float32) for item in records], axis=0)
        return points, colors

    def shared_anchor_sample(self, clip_a: int, clip_b: int, limit: int = 128) -> Dict[str, Any]:
        anchor_ids = []
        for record in sorted(self.records.values(), key=lambda item: item["anchor_id"]):
            clips = record["clip_indices"]
            if int(clip_a) in clips and int(clip_b) in clips:
                anchor_ids.append(record["anchor_id"])
        return {
            "num_shared_anchor_ids": int(len(anchor_ids)),
            "shared_anchor_ids_sample": anchor_ids[:limit],
        }

    def export(self, out_path: Path) -> Dict[str, Any]:
        points, colors = self.arrays()
        write_ply(out_path, points, colors)
        return {
            "path": str(out_path),
            "num_anchors": int(len(points)),
            "voxel_size": float(self.voxel_size),
            "origin": None if self.origin is None else np.asarray(self.origin, dtype=float).tolist(),
        }

    def summary(self) -> Dict[str, Any]:
        source_counts: Dict[str, int] = {}
        observations = []
        visible_frames = []
        visible_clips = []
        for record in self.records.values():
            observations.append(int(record["observations"]))
            visible_frames.append(len(record["frame_ids"]))
            visible_clips.append(len(record["clip_indices"]))
            for source_type in record["source_types"]:
                source_counts[source_type] = source_counts.get(source_type, 0) + 1
        return {
            "num_anchors": int(len(self.records)),
            "voxel_size": float(self.voxel_size),
            "origin": None if self.origin is None else np.asarray(self.origin, dtype=float).tolist(),
            "anchors_by_source_type": source_counts,
            "mean_observations_per_anchor": float(np.mean(observations)) if observations else 0.0,
            "mean_visible_frames_per_anchor": float(np.mean(visible_frames)) if visible_frames else 0.0,
            "mean_visible_fine_clips_per_anchor": float(np.mean(visible_clips)) if visible_clips else 0.0,
        }


class DebugGlobalDynamicBuffer:
    """Frame-indexed dynamic buffer with overlap dedupe by pose-consistency score."""

    def __init__(self) -> None:
        self.frame_records: Dict[str, Dict[str, Any]] = {}
        self.clip_to_frames: Dict[int, List[str]] = {}
        self.update_log: List[Dict[str, Any]] = []

    def _should_replace_frame_record(self, prev: Dict[str, Any], cur: Dict[str, Any]) -> bool:
        prev_score = float(prev["pose_translation_error_vs_source"]) + 0.05 * float(
            prev["pose_rotation_error_vs_source_deg"]
        )
        cur_score = float(cur["pose_translation_error_vs_source"]) + 0.05 * float(
            cur["pose_rotation_error_vs_source_deg"]
        )
        if cur_score < prev_score - 1e-6:
            return True
        if abs(cur_score - prev_score) <= 1e-6:
            if int(cur["dynamic_points"].shape[0]) > int(prev["dynamic_points"].shape[0]):
                return True
            if int(cur.get("clip_index", -1)) > int(prev.get("clip_index", -1)):
                return True
        return False

    def update_from_state(
        self,
        dataset: SceneDatadirDataset,
        alignment_state: Dict[str, Any],
        frame_ids: Sequence[str],
        clip_index: int,
        min_confidence: float,
        dynamic_min_confidence: float,
        max_points_per_frame: int | None,
        seed: int,
    ) -> Dict[str, Any]:
        clip_index = int(clip_index)
        self.clip_to_frames[clip_index] = list(frame_ids)
        per_frame_updates = []
        kept_new_frames = 0
        replaced_frames = 0
        skipped_frames = 0

        for frame_id in frame_ids:
            obs = _extract_frame_observations(
                dataset=dataset,
                alignment_state=alignment_state,
                frame_id=frame_id,
                min_confidence=min_confidence,
                dynamic_min_confidence=dynamic_min_confidence,
                max_points_per_frame=max_points_per_frame,
                seed=seed,
            )
            cur = {
                **obs,
                "clip_index": clip_index,
                "frame_id": frame_id,
            }
            prev = self.frame_records.get(frame_id)
            action = "keep_new"
            if prev is None:
                self.frame_records[frame_id] = cur
                kept_new_frames += 1
            else:
                if self._should_replace_frame_record(prev, cur):
                    self.frame_records[frame_id] = cur
                    action = "replace_prev"
                    replaced_frames += 1
                else:
                    action = "skip_worse"
                    skipped_frames += 1

            per_frame_updates.append(
                {
                    "frame_id": frame_id,
                    "clip_index": clip_index,
                    "action": action,
                    "dynamic_points": int(cur["dynamic_points"].shape[0]),
                    "pose_translation_error_vs_source": float(cur["pose_translation_error_vs_source"]),
                    "pose_rotation_error_vs_source_deg": float(cur["pose_rotation_error_vs_source_deg"]),
                    "points_total": int(cur["points_total"]),
                    "points_valid": int(cur["points_valid"]),
                    "points_sampled": int(cur["points_sampled"]),
                }
            )

        selected_points = int(sum(item["dynamic_points"] for item in per_frame_updates if item["action"] != "skip_worse"))
        meta = {
            "clip_index": clip_index,
            "num_frames": int(len(frame_ids)),
            "selected_or_replaced_frames": int(kept_new_frames + replaced_frames),
            "kept_new_frames": int(kept_new_frames),
            "replaced_frames": int(replaced_frames),
            "skipped_frames": int(skipped_frames),
            "dynamic_points_selected_from_this_update": int(selected_points),
            "per_frame": per_frame_updates,
        }
        self.update_log.append(meta)
        return meta

    def arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.frame_records:
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)
        points = []
        colors = []
        for frame_id in sorted(self.frame_records.keys()):
            record = self.frame_records[frame_id]
            pts = np.asarray(record["dynamic_points"], dtype=np.float32).reshape(-1, 3)
            cols = np.asarray(record["dynamic_colors"], dtype=np.float32).reshape(-1, 3)
            if len(pts) == 0:
                continue
            points.append(pts)
            colors.append(cols)
        if not points:
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)
        return np.concatenate(points, axis=0), np.concatenate(colors, axis=0)

    def export(self, out_path: Path) -> Dict[str, Any]:
        points, colors = self.arrays()
        write_ply(out_path, points, colors)
        return {
            "path": str(out_path),
            "num_points": int(len(points)),
            "organization": "per_frame_best_candidate_by_pose_consistency",
        }

    def summary(self) -> Dict[str, Any]:
        points, _ = self.arrays()
        per_frame = []
        per_clip: Dict[int, int] = {}
        for frame_id in sorted(self.frame_records.keys()):
            record = self.frame_records[frame_id]
            clip_idx = int(record["clip_index"])
            count = int(np.asarray(record["dynamic_points"]).shape[0])
            per_clip[clip_idx] = per_clip.get(clip_idx, 0) + count
            per_frame.append(
                {
                    "frame_id": frame_id,
                    "clip_index": clip_idx,
                    "num_dynamic_points": count,
                    "pose_translation_error_vs_source": float(record["pose_translation_error_vs_source"]),
                    "pose_rotation_error_vs_source_deg": float(record["pose_rotation_error_vs_source_deg"]),
                }
            )

        per_clip_rows = [
            {
                "clip_index": int(clip_idx),
                "num_dynamic_points": int(num_points),
                "frame_ids": self.clip_to_frames.get(int(clip_idx), []),
            }
            for clip_idx, num_points in sorted(per_clip.items())
        ]
        return {
            "num_dynamic_points": int(len(points)),
            "num_frames_with_dynamic_points": int(sum(1 for item in per_frame if item["num_dynamic_points"] > 0)),
            "num_clips_with_dynamic_points": int(sum(1 for item in per_clip_rows if item["num_dynamic_points"] > 0)),
            "selection_policy": "keep_best_pose_consistency_per_frame_then_export",
            "per_frame": per_frame,
            "per_clip": per_clip_rows,
        }


def _inject_shared_anchor_counts(
    overlap_graph: Dict[str, Any],
    static_map: DebugGlobalStaticMap,
) -> Dict[str, Any]:
    out = json.loads(json.dumps(overlap_graph))
    for item in out.get("adjacent_clip_pairs", []):
        shared = static_map.shared_anchor_sample(item["clip_index_a"], item["clip_index_b"])
        item.update(shared)
    out["num_pairs_with_shared_anchors"] = int(
        sum(1 for item in out.get("adjacent_clip_pairs", []) if item.get("num_shared_anchor_ids", 0) > 0)
    )
    return out


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    scene_root = Path(args.scene_root).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else scene_root / "monst3r_reflow_a1_fine_debug"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    evo_fallback = _ensure_writable_evo_home(scene_root)
    if evo_fallback is not None:
        print(f"[Init] HOME is read-only; using writable HOME={evo_fallback}")
    keyframe_lock_policy = "post_sim3=disabled, coarse_init+teacher_geometry=enabled"
    print(
        "[Init] Camera policy: coarse=fixed_to_source_colmap(intrinsics+extrinsics), "
        f"fine_anchor_mode={args.fine_camera_anchor_mode} + {keyframe_lock_policy}"
    )

    dataset = SceneDatadirDataset(
        scene_root,
        split=args.split,
        dynamic_label_mode="nonzero_is_dynamic",
        dynamic_label=None,
    )
    frame_ids = _maybe_cap(dataset.get_train_frame_ids(), args.max_frames)
    if not frame_ids:
        raise RuntimeError(f"No frame ids found for split {args.split}")

    desc = dataset.describe()
    pair_graph = _build_fine_debug_pair_graph(
        frame_ids,
        clip_len=args.clip_len,
        overlap_frames=args.overlap_frames,
        coarse_max_offset=args.coarse_max_offset,
        fine_max_offset=args.fine_max_offset,
    )
    clip_index = _resolve_clip_index(len(pair_graph.clips), args.first_clip_index)
    debug_clip_indices = _resolve_debug_clip_indices(
        len(pair_graph.clips),
        clip_index,
        args.num_fine_debug_clips,
    )
    debug_clips = [list(pair_graph.clips[idx]) for idx in debug_clip_indices]
    debug_fine_pairs = [list(pair_graph.fine_pairs_per_clip[idx]) for idx in debug_clip_indices]
    selected_clip = debug_clips[0]
    selected_fine_pairs = debug_fine_pairs[0]
    overlap_graph = _build_clip_overlap_graph(pair_graph.clips)

    pair_graph_json = output_dir / "fine_debug_pair_graph.json"
    with pair_graph_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                **pair_graph.as_dict(),
                "num_frames_processed": len(frame_ids),
                "num_clips": len(pair_graph.clips),
                "num_keyframes": len(pair_graph.keyframes),
                "num_coarse_pairs": len(pair_graph.coarse_pairs),
                "num_fine_pairs_total": sum(len(pairs) for pairs in pair_graph.fine_pairs_per_clip),
                "coarse_max_offset": int(args.coarse_max_offset),
                "fine_max_offset": int(args.fine_max_offset),
                "overlap_frames": int(args.overlap_frames),
                "num_fine_debug_clips": int(len(debug_clip_indices)),
                "debug_clip_indices": debug_clip_indices,
                "debug_clip_frame_ids": debug_clips,
                "debug_clip_num_fine_pairs": [int(len(pairs)) for pairs in debug_fine_pairs],
                "clip_overlap_graph": overlap_graph,
                "selected_clip_index": int(clip_index),
                "selected_clip_frame_ids": selected_clip,
                "selected_clip_num_fine_pairs": int(len(selected_fine_pairs)),
            },
            f,
            indent=2,
        )
    print(
        f"[Phase 1-2] Dataset frames={desc['num_frames']} processed={len(frame_ids)} "
        f"clips={len(pair_graph.clips)} keyframes={len(pair_graph.keyframes)} "
        f"coarse_pairs={len(pair_graph.coarse_pairs)} selected_clip={clip_index} "
        f"debug_clips={debug_clip_indices} selected_clip_frames={len(selected_clip)} "
        f"selected_fine_pairs={len(selected_fine_pairs)} "
        f"(coarse_max_offset={args.coarse_max_offset}, fine_max_offset={args.fine_max_offset}, "
        f"overlap_frames={args.overlap_frames})"
    )
    print(f"[Phase 2] Wrote pair graph: {pair_graph_json}")
    if args.dry_run:
        print("[Phase 2] Dry run complete; no MonST3R alignment was executed")
        return 0

    inferencer = PairwiseInferencer(
        scene_root=dataset.scene_root,
        image_size=args.image_size,
        weights=args.weights,
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
        verbose=True,
    )
    print("[Phase 3] Pair inference cache is ready")

    coarse_state = run_coarse_alignment(
        dataset,
        keyframes=pair_graph.keyframes,
        coarse_pairs=pair_graph.coarse_pairs,
        inferencer=inferencer,
        niter=args.niter_coarse,
        schedule=args.schedule,
        lr=args.lr,
        force_recompute_pairs=args.force_recompute_pairs,
        verbose=True,
    )
    print(
        f"[Phase 4] Coarse alignment done: {len(coarse_state.get('frame_ids', []))} keyframe(s), "
        f"{coarse_state.get('num_pairs', len(pair_graph.coarse_pairs))} pair(s)"
    )

    coarse_out_dir = output_dir / "coarse"
    coarse_out_dir.mkdir(parents=True, exist_ok=True)
    coarse_ply_path = coarse_out_dir / "coarse_keyframes.ply"
    coarse_ply_meta = export_coarse_keyframe_pointcloud(
        coarse_state=coarse_state,
        out_path=coarse_ply_path,
        min_confidence=float(args.min_confidence),
        max_points_per_frame=None
        if args.max_points_per_frame is not None and args.max_points_per_frame <= 0
        else args.max_points_per_frame,
        seed=int(args.seed),
        voxel_keep_max_conf=bool(args.coarse_voxel_keep_max_conf),
        voxel_size=float(args.coarse_voxel_size),
    )
    coarse_pose_error_json = coarse_out_dir / "coarse_pose_error.json"
    coarse_pose_error_csv = coarse_out_dir / "coarse_pose_error.csv"
    coarse_pose_error_meta = export_coarse_pose_errors(
        dataset=dataset,
        coarse_state=coarse_state,
        out_json_path=coarse_pose_error_json,
        out_csv_path=coarse_pose_error_csv,
    )
    coarse_camera_export = export_optimized_cameras(dataset, coarse_state, coarse_out_dir)
    coarse_trajectory_meta = _safe_trajectory_export(dataset, coarse_state, coarse_out_dir)
    print(
        "[Phase 4] Coarse pose-vs-source summary: "
        f"raw_t_mean={coarse_pose_error_meta['translation_error_l2_raw']['mean']:.6g}, "
        f"raw_r_mean={coarse_pose_error_meta['rotation_error_deg_raw']['mean']:.6g} deg"
    )

    max_points_per_frame = (
        None
        if args.max_points_per_frame is not None and args.max_points_per_frame <= 0
        else args.max_points_per_frame
    )
    global_registry_dir = output_dir / "fine_global_registry"
    global_registry_dir.mkdir(parents=True, exist_ok=True)
    requested_global_static_voxel = float(args.global_static_voxel_size)
    if requested_global_static_voxel <= 0 and bool(args.fine_voxel_keep_max_conf) and float(args.fine_voxel_size) > 0:
        requested_global_static_voxel = float(args.fine_voxel_size)
    global_static_voxel_size = _estimate_static_voxel_size(
        dataset=dataset,
        coarse_state=coarse_state,
        requested_voxel_size=requested_global_static_voxel,
        min_confidence=float(args.min_confidence),
        dynamic_min_confidence=float(args.dynamic_min_confidence),
        max_points_per_frame=max_points_per_frame,
        seed=int(args.seed),
    )
    static_map = DebugGlobalStaticMap(voxel_size=global_static_voxel_size)
    dynamic_buffer = DebugGlobalDynamicBuffer()
    coarse_static_update_meta = static_map.update_from_state(
        dataset=dataset,
        alignment_state=coarse_state,
        frame_ids=coarse_state.get("frame_ids", []),
        clip_index=None,
        source_type="coarse_keyframe_static_seed",
        min_confidence=float(args.min_confidence),
        dynamic_min_confidence=float(args.dynamic_min_confidence),
        max_points_per_frame=max_points_per_frame,
        seed=int(args.seed),
    )
    print(
        "[Phase 4.5] Initialized GlobalStaticMap from coarse keyframes: "
        f"anchors={static_map.summary()['num_anchors']} voxel={global_static_voxel_size:.6g}"
    )

    fine_schedule = args.schedule if args.schedule_fine is None else args.schedule_fine
    fine_lr = args.lr if args.lr_fine is None else args.lr_fine
    print(
        f"[Phase 5] Running sequential overlap fine debug clips {debug_clip_indices} "
        f"niter={args.niter_fine}, schedule={fine_schedule}, lr={fine_lr}, "
        f"camera_mode={args.fine_camera_anchor_mode}, "
        f"temporal_smooth={args.fine_temporal_smoothing_weight}, "
        f"pose_prior={args.fine_camera_pose_prior_weight}, "
        f"intr_prior={args.fine_camera_intrinsics_prior_weight}, "
        f"pose_prior_t={args.fine_camera_pose_prior_translation_weight}, "
        f"coarse_point_teacher={args.fine_coarse_pointmap_teacher_weight}, "
        f"coarse_depth_teacher={args.fine_coarse_depth_teacher_weight}, "
        f"teacher_min_conf={args.fine_coarse_teacher_min_confidence})"
    )

    fine_clip_states: List[Dict[str, Any]] = []
    fine_clip_update_meta: List[Dict[str, Any]] = []
    for local_order, graph_clip_index in enumerate(debug_clip_indices):
        clip = debug_clips[local_order]
        pairs = debug_fine_pairs[local_order]
        print(
            f"[Phase 5] Fine clip {local_order + 1}/{len(debug_clip_indices)} "
            f"(graph_clip={graph_clip_index}, frames={len(clip)}, pairs={len(pairs)})"
        )
        clip_state = run_fine_alignment(
            dataset,
            clips=[clip],
            fine_pairs_per_clip=[pairs],
            coarse_state=coarse_state,
            inferencer=inferencer,
            niter=args.niter_fine,
            schedule=fine_schedule,
            lr=fine_lr,
            temporal_smoothing_weight=args.fine_temporal_smoothing_weight,
            camera_pose_prior_weight=args.fine_camera_pose_prior_weight,
            camera_intrinsics_prior_weight=args.fine_camera_intrinsics_prior_weight,
            camera_pose_prior_translation_weight=args.fine_camera_pose_prior_translation_weight,
            camera_anchor_mode=args.fine_camera_anchor_mode,
            previous_aligned_state=fine_clip_states[-1] if fine_clip_states else None,
            coarse_pointmap_teacher_weight=args.fine_coarse_pointmap_teacher_weight,
            coarse_depth_teacher_weight=args.fine_coarse_depth_teacher_weight,
            coarse_teacher_min_confidence=args.fine_coarse_teacher_min_confidence,
            force_recompute_pairs=args.force_recompute_pairs,
            verbose=True,
        )
        clip_state["debug_graph_clip_index"] = int(graph_clip_index)
        clip_state["debug_clip_frame_ids"] = list(clip)
        fine_clip_states.append(clip_state)

        static_update_meta = static_map.update_from_state(
            dataset=dataset,
            alignment_state=clip_state,
            frame_ids=clip,
            clip_index=int(graph_clip_index),
            source_type="fine_clip_static_update",
            min_confidence=float(args.min_confidence),
            dynamic_min_confidence=float(args.dynamic_min_confidence),
            max_points_per_frame=max_points_per_frame,
            seed=int(args.seed) + local_order + 1,
        )
        dynamic_update_meta = dynamic_buffer.update_from_state(
            dataset=dataset,
            alignment_state=clip_state,
            frame_ids=clip,
            clip_index=int(graph_clip_index),
            min_confidence=float(args.min_confidence),
            dynamic_min_confidence=float(args.dynamic_min_confidence),
            max_points_per_frame=max_points_per_frame,
            seed=int(args.seed) + local_order + 1,
        )
        fine_clip_update_meta.append(
            {
                "graph_clip_index": int(graph_clip_index),
                "frame_ids": list(clip),
                "num_fine_pairs": int(len(pairs)),
                "alignment_loss": clip_state.get("alignment_loss"),
                "fine_coarse_initialization": clip_state.get("fine_coarse_initialization", {}),
                "coarse_geometry_teacher": clip_state.get("coarse_geometry_teacher", {}),
                "static_update": static_update_meta,
                "dynamic_update": dynamic_update_meta,
            }
        )
        print(
            "[Phase 5] Updated global buffers: "
            f"static_anchors={static_map.summary()['num_anchors']} "
            f"(new={static_update_meta['new_anchors']}, fused={static_update_meta['fused_anchors']}), "
            f"dynamic_points_selected={dynamic_update_meta['dynamic_points_selected_from_this_update']}, "
            f"dynamic_replaced_frames={dynamic_update_meta['replaced_frames']}"
        )

    fine_state = merge_clip_states(fine_clip_states)
    fine_state["debug_graph_clip_indices"] = debug_clip_indices
    fine_state["debug_clip_frame_ids"] = debug_clips
    fine_state["keyframe_pose_lock"] = {
        "enabled": False,
        "camera_source": "coarse_initialization_plus_teacher_loss",
        "reason": "post_alignment_disabled_no_sim3_or_rigid_keyframe_retarget",
        "clips": [dict(state.get("keyframe_pose_lock", {})) for state in fine_clip_states],
    }
    fine_state["global_static_map_summary"] = static_map.summary()
    fine_state["global_dynamic_buffer_summary"] = dynamic_buffer.summary()
    print(f"[Phase 5] Fine alignment done: {len(fine_state.get('frame_ids', []))} unique frame(s)")

    fine_out_dir = output_dir / "fine_debug_state"
    fine_out_dir.mkdir(parents=True, exist_ok=True)
    fine_pose_error_json = fine_out_dir / "fine_debug_pose_error.json"
    fine_pose_error_csv = fine_out_dir / "fine_debug_pose_error.csv"
    fine_pose_error_meta = export_coarse_pose_errors(
        dataset=dataset,
        coarse_state=fine_state,
        out_json_path=fine_pose_error_json,
        out_csv_path=fine_pose_error_csv,
    )
    fine_camera_export = export_optimized_cameras(dataset, fine_state, fine_out_dir)
    fine_trajectory_meta = _safe_trajectory_export(dataset, fine_state, fine_out_dir)
    print(
        "[Phase 5] Fine(debug window) pose-vs-source summary: "
        f"raw_t_mean={fine_pose_error_meta['translation_error_l2_raw']['mean']:.6g}, "
        f"raw_r_mean={fine_pose_error_meta['rotation_error_deg_raw']['mean']:.6g} deg"
    )

    static_map_ply_meta = static_map.export(global_registry_dir / "global_static_map.ply")
    dynamic_buffer_ply_meta = dynamic_buffer.export(global_registry_dir / "global_dynamic_buffer.ply")
    static_points, static_colors = static_map.arrays()
    dynamic_points, dynamic_colors = dynamic_buffer.arrays()
    canonical_preview_path = global_registry_dir / "global_canonical_debug_preview.ply"
    write_ply(
        canonical_preview_path,
        np.concatenate([static_points, dynamic_points], axis=0),
        np.concatenate([static_colors, dynamic_colors], axis=0),
    )
    overlap_graph_with_anchors = _inject_shared_anchor_counts(overlap_graph, static_map)
    overlap_graph_json = global_registry_dir / "fine_debug_clip_overlap_graph.json"
    with overlap_graph_json.open("w", encoding="utf-8") as f:
        json.dump(overlap_graph_with_anchors, f, indent=2)

    global_registry_summary = {
        "global_coarse_state": {
            "keyframe_ids": list(coarse_state.get("frame_ids", [])),
            "num_keyframes": int(len(coarse_state.get("frame_ids", []))),
            "num_coarse_pairs": int(coarse_state.get("num_pairs", len(pair_graph.coarse_pairs))),
            "scene_transform": coarse_state.get("source_camera_alignment"),
            "static_masks_keyframes": "dynamic_masks inverted from dataset/alignment_state",
            "dynamic_masks_keyframes": "stored per frame in coarse_state['dynamic_masks']",
        },
        "global_canonical_registry": {
            "static_map": {
                **static_map.summary(),
                "ply": static_map_ply_meta,
            },
            "global_anchor_set": {
                **static_map.summary(),
                "anchor_id_policy": "voxel-hash anchors generated as anchor_XXXXXXXX",
            },
            "coarse_static_seed_update": coarse_static_update_meta,
            "fine_static_updates": [item["static_update"] for item in fine_clip_update_meta],
        },
        "global_dynamic_buffer": {
            **dynamic_buffer.summary(),
            "ply": dynamic_buffer_ply_meta,
            "fine_dynamic_updates": [item["dynamic_update"] for item in fine_clip_update_meta],
        },
        "clip_overlap_graph": {
            **overlap_graph_with_anchors,
            "json": str(overlap_graph_json),
        },
        "canonical_debug_preview": {
            "path": str(canonical_preview_path),
            "num_static_points": int(len(static_points)),
            "num_dynamic_points": int(len(dynamic_points)),
            "note": "debug preview only; static comes from GlobalStaticMap, dynamic from GlobalDynamicBuffer",
        },
    }
    global_registry_summary_json = global_registry_dir / "global_registry_summary.json"
    with global_registry_summary_json.open("w", encoding="utf-8") as f:
        json.dump(global_registry_summary, f, indent=2)
    print(
        "[Phase 5.5] Exported global registry debug artifacts: "
        f"static={static_map_ply_meta['num_anchors']} anchors, "
        f"dynamic={dynamic_buffer_ply_meta['num_points']} points"
    )

    summary = {
        "scene_root": str(scene_root),
        "split": args.split,
        "clip_len": int(args.clip_len),
        "coarse_max_offset": int(args.coarse_max_offset),
        "fine_max_offset": int(args.fine_max_offset),
        "overlap_frames": int(args.overlap_frames),
        "num_frames_processed": int(len(frame_ids)),
        "num_clips": int(len(pair_graph.clips)),
        "num_keyframes": int(len(pair_graph.keyframes)),
        "num_coarse_pairs": int(len(pair_graph.coarse_pairs)),
        "num_fine_pairs_total": int(sum(len(pairs) for pairs in pair_graph.fine_pairs_per_clip)),
        "selected_clip_index": int(clip_index),
        "selected_clip_frame_ids": selected_clip,
        "selected_clip_num_fine_pairs": int(len(selected_fine_pairs)),
        "debug_clip_indices": debug_clip_indices,
        "debug_clip_frame_ids": debug_clips,
        "debug_clip_num_fine_pairs": [int(len(pairs)) for pairs in debug_fine_pairs],
        "pair_graph_json": str(pair_graph_json),
        "coarse_hyperparams": {
            "image_size": int(args.image_size),
            "batch_size": int(args.batch_size),
            "niter_coarse": int(args.niter_coarse),
            "schedule": str(args.schedule),
            "lr": float(args.lr),
            "coarse_voxel_keep_max_conf": bool(args.coarse_voxel_keep_max_conf),
            "coarse_voxel_size": float(args.coarse_voxel_size),
            "camera_policy": "fixed_source_colmap_intrinsics_extrinsics",
            "static_masked_loss": bool(coarse_state.get("static_only_loss", False)),
        },
        "fine_hyperparams": {
            "niter_fine": int(args.niter_fine),
            "schedule_fine": str(fine_schedule),
            "lr_fine": float(fine_lr),
            "fine_camera_anchor_mode": str(args.fine_camera_anchor_mode),
            "fine_temporal_smoothing_weight": float(args.fine_temporal_smoothing_weight),
            "fine_camera_pose_prior_weight": float(args.fine_camera_pose_prior_weight),
            "fine_camera_intrinsics_prior_weight": float(args.fine_camera_intrinsics_prior_weight),
            "fine_camera_pose_prior_translation_weight": float(args.fine_camera_pose_prior_translation_weight),
            "fine_coarse_pointmap_teacher_weight": float(args.fine_coarse_pointmap_teacher_weight),
            "fine_coarse_depth_teacher_weight": float(args.fine_coarse_depth_teacher_weight),
            "fine_coarse_teacher_min_confidence": float(args.fine_coarse_teacher_min_confidence),
            "dynamic_min_confidence": float(args.dynamic_min_confidence),
            "fine_voxel_keep_max_conf": bool(args.fine_voxel_keep_max_conf),
            "fine_voxel_size": float(args.fine_voxel_size),
            "global_static_voxel_size": float(global_static_voxel_size),
            "camera_policy": (
                f"fine_camera_anchor_mode={args.fine_camera_anchor_mode}, "
                "post_sim3_disabled, coarse_keyframe_and_overlap_init"
            ),
            "global_registry_policy": "sequential_fine_static_fusion_plus_per_frame_best_dynamic_buffer",
        },
        "coarse": {
            "output_dir": str(coarse_out_dir),
            "alignment_loss": coarse_state.get("alignment_loss"),
            "pair_cache_stats": coarse_state.get("pair_cache", inferencer.cache_stats()),
            "coarse_keyframes_ply": coarse_ply_meta,
            "coarse_pose_error": {
                "json": str(coarse_pose_error_json),
                "csv": str(coarse_pose_error_csv),
                "translation_error_l2_raw": coarse_pose_error_meta["translation_error_l2_raw"],
                "rotation_error_deg_raw": coarse_pose_error_meta["rotation_error_deg_raw"],
                "sim3_alignment": coarse_pose_error_meta.get("sim3_alignment", {}),
            },
            "optimized_camera_dir": coarse_camera_export.get("optimized_camera_dir"),
            "optimized_camera_index": coarse_camera_export.get("optimized_camera_index"),
            "camera_trajectory_plot": coarse_trajectory_meta.get("trajectory_plot"),
        },
        "fine_debug_state": {
            "output_dir": str(fine_out_dir),
            "alignment_loss": fine_state.get("alignment_loss"),
            "camera_anchor_mode": fine_state.get("camera_anchor_mode"),
            "poses_fixed_to_source": bool(fine_state.get("poses_fixed_to_source", False)),
            "poses_fixed_to_initialization": bool(fine_state.get("poses_fixed_to_initialization", False)),
            "keyframe_pose_lock": fine_state.get("keyframe_pose_lock", {}),
            "debug_clip_updates": fine_clip_update_meta,
            "pose_error_vs_source_colmap": {
                "json": str(fine_pose_error_json),
                "csv": str(fine_pose_error_csv),
                "translation_error_l2_raw": fine_pose_error_meta["translation_error_l2_raw"],
                "rotation_error_deg_raw": fine_pose_error_meta["rotation_error_deg_raw"],
                "sim3_alignment": fine_pose_error_meta.get("sim3_alignment", {}),
            },
            "optimized_camera_dir": fine_camera_export.get("optimized_camera_dir"),
            "optimized_camera_index": fine_camera_export.get("optimized_camera_index"),
            "camera_trajectory_plot": fine_trajectory_meta.get("trajectory_plot"),
        },
        "global_registry": {
            "output_dir": str(global_registry_dir),
            "summary_json": str(global_registry_summary_json),
            "static_map_ply": static_map_ply_meta,
            "dynamic_buffer_ply": dynamic_buffer_ply_meta,
            "canonical_debug_preview": str(canonical_preview_path),
            "clip_overlap_graph_json": str(overlap_graph_json),
            "static_map": static_map.summary(),
            "dynamic_buffer": dynamic_buffer.summary(),
        },
    }
    summary_path = output_dir / "fine_debug_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[Done] Fine debug summary: {summary_path}")
    print(
        f"[Done] Fine clip camera-vs-source comparison: "
        f"{fine_pose_error_json} and {fine_pose_error_csv}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
