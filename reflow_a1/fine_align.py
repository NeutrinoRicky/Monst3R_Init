"""Clip-wise fine alignment for ReFlow A.1."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .coarse_align import run_monst3r_alignment
from .dataset_scene import SceneDatadirDataset
from .pair_infer import PairwiseInferencer

Pair = Tuple[str, str]


def _apply_transform_to_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    shape = points.shape
    flat = points.reshape(-1, 3)
    homog = np.concatenate([flat, np.ones((flat.shape[0], 1), dtype=np.float32)], axis=1)
    out = homog @ transform.T
    return out[:, :3].reshape(shape).astype(np.float32)


def estimate_similarity_from_pointmaps(
    src: np.ndarray,
    dst: np.ndarray,
    src_mask: np.ndarray | None = None,
    dst_mask: np.ndarray | None = None,
    max_points: int = 20000,
) -> np.ndarray:
    valid = np.isfinite(src).all(axis=-1) & np.isfinite(dst).all(axis=-1)
    if src_mask is not None and src_mask.shape == valid.shape:
        valid &= src_mask
    if dst_mask is not None and dst_mask.shape == valid.shape:
        valid &= dst_mask

    src_flat = src[valid].astype(np.float64)
    dst_flat = dst[valid].astype(np.float64)
    if src_flat.shape[0] < 8:
        raise ValueError("Not enough overlapping keyframe points for similarity alignment")

    if src_flat.shape[0] > max_points:
        rng = np.random.default_rng(17)
        keep = rng.choice(src_flat.shape[0], size=max_points, replace=False)
        src_flat = src_flat[keep]
        dst_flat = dst_flat[keep]

    mu_src = src_flat.mean(axis=0)
    mu_dst = dst_flat.mean(axis=0)
    x = src_flat - mu_src
    y = dst_flat - mu_dst
    cov = (y.T @ x) / src_flat.shape[0]
    U, singular, Vt = np.linalg.svd(cov)
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[-1, -1] = -1.0
    R = U @ D @ Vt
    var_src = np.sum(x * x) / src_flat.shape[0]
    scale = float(np.sum(singular * np.diag(D)) / max(var_src, 1e-12))
    t = mu_dst - scale * (R @ mu_src)

    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = (scale * R).astype(np.float32)
    transform[:3, 3] = t.astype(np.float32)
    return transform


def pose_transform_from_keyframe(local_state: Dict[str, Any], coarse_state: Dict[str, Any], keyframe: str) -> np.ndarray:
    local_pose = local_state["poses"][keyframe]
    coarse_pose = coarse_state["poses"][keyframe]
    return (coarse_pose @ np.linalg.inv(local_pose)).astype(np.float32)


def transform_alignment_state(state: Dict[str, Any], transform: np.ndarray) -> Dict[str, Any]:
    out = dict(state)
    out["global_pointmaps"] = {
        frame_id: _apply_transform_to_points(points, transform)
        for frame_id, points in state["global_pointmaps"].items()
    }
    out["poses"] = {
        frame_id: (transform @ pose).astype(np.float32)
        for frame_id, pose in state["poses"].items()
    }
    return out


def align_clip_to_coarse(
    local_state: Dict[str, Any],
    coarse_state: Dict[str, Any],
    keyframe: str,
) -> Dict[str, Any]:
    """Deprecated no-op: fine clips must now optimize directly in coarse space.

    The old implementation estimated a post-hoc Sim(3) from fine keyframe
    pointmaps to coarse keyframe pointmaps. That hid local scale drift instead
    of preventing it, so the fine stage now receives coarse initialization and
    teacher geometry before optimization.
    """

    out = dict(local_state)
    out["clip_to_coarse_transform"] = np.eye(4, dtype=np.float32)
    out["post_alignment"] = {
        "sim3_enabled": False,
        "rigid_keyframe_lock_enabled": False,
        "keyframe": keyframe,
        "reason": "disabled_fine_optimizes_directly_in_coarse_space",
    }
    return out


def enforce_keyframe_pose_to_coarse(
    aligned_state: Dict[str, Any],
    coarse_state: Dict[str, Any],
    keyframe: str,
) -> Dict[str, Any]:
    """Apply one rigid correction so the clip keyframe pose matches coarse exactly."""

    local_pose = aligned_state.get("poses", {}).get(keyframe)
    coarse_pose = coarse_state.get("poses", {}).get(keyframe)
    if local_pose is None or coarse_pose is None:
        out = dict(aligned_state)
        out["keyframe_pose_lock"] = {
            "enabled": False,
            "keyframe": keyframe,
            "reason": "missing_keyframe_pose",
        }
        return out

    local_pose = np.asarray(local_pose, dtype=np.float32)
    coarse_pose = np.asarray(coarse_pose, dtype=np.float32)
    correction = (coarse_pose @ np.linalg.inv(local_pose)).astype(np.float32)
    corrected = transform_alignment_state(aligned_state, correction)

    if "clip_to_coarse_transform" in aligned_state:
        prev = np.asarray(aligned_state["clip_to_coarse_transform"], dtype=np.float32)
        corrected["clip_to_coarse_transform"] = (correction @ prev).astype(np.float32)
    corrected["poses"][keyframe] = coarse_pose
    corrected["keyframe_pose_lock"] = {
        "enabled": True,
        "keyframe": keyframe,
        "camera_source": "coarse_fixed_source_colmap",
    }
    return corrected


def _state_has_frame_value(state: Optional[Mapping[str, Any]], key: str, frame_id: str) -> bool:
    value = None if state is None else state.get(key, {}).get(frame_id)
    return value is not None


def _copy_array(value: Any, dtype: Any = np.float32) -> np.ndarray:
    return np.asarray(value, dtype=dtype).copy()


def _source_camera(dataset: SceneDatadirDataset, frame_id: str) -> Tuple[np.ndarray, np.ndarray]:
    cam = dataset.get_frame_by_id(frame_id)["camera_dict"]
    return cam["T_wc"].astype(np.float32), cam["K"].astype(np.float32)


def _depth_from_world_pointmap(pointmap: np.ndarray, pose: np.ndarray) -> np.ndarray:
    pts = np.asarray(pointmap, dtype=np.float32)
    shape = pts.shape[:2]
    flat = pts.reshape(-1, 3)
    homog = np.concatenate([flat, np.ones((flat.shape[0], 1), dtype=np.float32)], axis=1)
    cam = homog @ np.linalg.inv(np.asarray(pose, dtype=np.float32)).T
    return cam[:, 2].reshape(shape).astype(np.float32)


def _state_depth_or_derived(state: Mapping[str, Any], frame_id: str, pose: np.ndarray) -> Optional[np.ndarray]:
    depth = state.get("depths", {}).get(frame_id)
    if depth is not None:
        arr = np.asarray(depth, dtype=np.float32)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        if arr.ndim == 2:
            return arr.copy()
    pointmap = state.get("global_pointmaps", {}).get(frame_id)
    if pointmap is None:
        return None
    return _depth_from_world_pointmap(np.asarray(pointmap, dtype=np.float32), pose)


def _teacher_mask_for_frame(
    state: Mapping[str, Any],
    frame_id: str,
    min_confidence: float,
) -> Optional[np.ndarray]:
    pointmap = state.get("global_pointmaps", {}).get(frame_id)
    depth = state.get("depths", {}).get(frame_id)
    mask: Optional[np.ndarray] = None
    if pointmap is not None:
        pts = np.asarray(pointmap, dtype=np.float32)
        if pts.ndim == 3 and pts.shape[-1] == 3:
            mask = np.isfinite(pts).all(axis=-1)
    if depth is not None:
        dep = np.asarray(depth, dtype=np.float32)
        if dep.ndim == 3 and dep.shape[-1] == 1:
            dep = dep[..., 0]
        if dep.ndim == 2:
            dep_mask = np.isfinite(dep) & (dep > 0)
            mask = dep_mask if mask is None else (mask & dep_mask if dep_mask.shape == mask.shape else mask)
    valid = state.get("valid_masks", {}).get(frame_id)
    if valid is not None:
        valid_arr = np.asarray(valid).astype(bool)
        if mask is None:
            mask = valid_arr
        elif valid_arr.shape == mask.shape:
            mask &= valid_arr
    conf = state.get("confidence", {}).get(frame_id)
    if conf is not None and mask is not None:
        conf_arr = np.asarray(conf, dtype=np.float32)
        if conf_arr.shape == mask.shape:
            mask &= conf_arr >= float(min_confidence)
    dyn = state.get("dynamic_masks", {}).get(frame_id)
    if dyn is not None and mask is not None:
        dyn_arr = np.asarray(dyn).astype(bool)
        if dyn_arr.shape == mask.shape:
            mask &= ~dyn_arr
    if mask is None:
        return None
    return mask.astype(bool)


def build_fine_clip_initialization(
    dataset: SceneDatadirDataset,
    clip: Sequence[str],
    coarse_state: Mapping[str, Any],
    previous_aligned_state: Optional[Mapping[str, Any]] = None,
    teacher_min_confidence: float = 3.0,
) -> Dict[str, Any]:
    """Build the coarse-space initialization/teacher package for one fine clip."""

    camera_init_poses: Dict[str, np.ndarray] = {}
    camera_init_intrinsics: Dict[str, np.ndarray] = {}
    depth_initialization: Dict[str, np.ndarray] = {}
    pointmap_teacher: Dict[str, np.ndarray] = {}
    depth_teacher: Dict[str, np.ndarray] = {}
    geometry_teacher_mask: Dict[str, np.ndarray] = {}
    frame_sources: Dict[str, Dict[str, Any]] = {}

    for frame_id in clip:
        source_pose, source_K = _source_camera(dataset, frame_id)
        anchor_state: Optional[Mapping[str, Any]] = None
        anchor_source = "source_camera"
        if (
            _state_has_frame_value(coarse_state, "poses", frame_id)
            or _state_has_frame_value(coarse_state, "global_pointmaps", frame_id)
        ):
            anchor_state = coarse_state
            anchor_source = "coarse_keyframe"
        elif previous_aligned_state is not None and (
            _state_has_frame_value(previous_aligned_state, "poses", frame_id)
            or _state_has_frame_value(previous_aligned_state, "global_pointmaps", frame_id)
        ):
            anchor_state = previous_aligned_state
            anchor_source = "previous_clip_overlap"

        pose = source_pose
        intrinsics = source_K
        if anchor_state is not None:
            if _state_has_frame_value(anchor_state, "poses", frame_id):
                pose = _copy_array(anchor_state["poses"][frame_id])
            if _state_has_frame_value(anchor_state, "intrinsics", frame_id):
                intrinsics = _copy_array(anchor_state["intrinsics"][frame_id])

            pointmap = anchor_state.get("global_pointmaps", {}).get(frame_id)
            if pointmap is not None:
                pointmap_teacher[frame_id] = _copy_array(pointmap)
            depth = _state_depth_or_derived(anchor_state, frame_id, pose)
            if depth is not None:
                depth_initialization[frame_id] = depth
                depth_teacher[frame_id] = depth.copy()
            mask = _teacher_mask_for_frame(anchor_state, frame_id, teacher_min_confidence)
            if mask is not None:
                geometry_teacher_mask[frame_id] = mask

        camera_init_poses[frame_id] = pose
        camera_init_intrinsics[frame_id] = intrinsics
        frame_sources[frame_id] = {
            "pose_source": anchor_source,
            "geometry_source": anchor_source if frame_id in pointmap_teacher or frame_id in depth_teacher else "none",
            "has_pointmap_teacher": frame_id in pointmap_teacher,
            "has_depth_teacher": frame_id in depth_teacher,
            "has_teacher_mask": frame_id in geometry_teacher_mask,
        }

    meta = {
        "enabled": True,
        "policy": "coarse_keyframe_then_previous_overlap_then_source_camera",
        "teacher_min_confidence": float(teacher_min_confidence),
        "num_frames": int(len(clip)),
        "num_pose_init": int(len(camera_init_poses)),
        "num_intrinsics_init": int(len(camera_init_intrinsics)),
        "num_depth_init": int(len(depth_initialization)),
        "num_pointmap_teacher_frames": int(len(pointmap_teacher)),
        "num_depth_teacher_frames": int(len(depth_teacher)),
        "num_teacher_mask_frames": int(len(geometry_teacher_mask)),
        "coarse_anchor_frame_ids": [
            frame_id for frame_id, item in frame_sources.items() if item["pose_source"] == "coarse_keyframe"
        ],
        "previous_overlap_anchor_frame_ids": [
            frame_id for frame_id, item in frame_sources.items() if item["pose_source"] == "previous_clip_overlap"
        ],
        "source_camera_frame_ids": [
            frame_id for frame_id, item in frame_sources.items() if item["pose_source"] == "source_camera"
        ],
        "frames": frame_sources,
    }

    return {
        "camera_init_poses": camera_init_poses,
        "camera_init_intrinsics": camera_init_intrinsics,
        "depth_initialization": depth_initialization,
        "pointmap_teacher": pointmap_teacher,
        "depth_teacher": depth_teacher,
        "geometry_teacher_mask": geometry_teacher_mask,
        "meta": meta,
    }


def merge_clip_states(clip_states: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {
        "frame_ids": [],
        "global_pointmaps": {},
        "poses": {},
        "intrinsics": {},
        "depths": {},
        "confidence": {},
        "colors": {},
        "dynamic_masks": {},
        "valid_masks": {},
        "alignment_loss": None,
        "source": "monst3r_clip_fine_alignment",
        "clips": [],
    }
    losses = []
    for state in clip_states:
        merged["clips"].append(state["frame_ids"])
        for frame_id in state["frame_ids"]:
            if frame_id in merged["global_pointmaps"]:
                continue
            merged["frame_ids"].append(frame_id)
            for key in (
                "global_pointmaps",
                "poses",
                "intrinsics",
                "depths",
                "confidence",
                "colors",
                "dynamic_masks",
                "valid_masks",
            ):
                merged[key][frame_id] = state[key][frame_id]
        if state.get("alignment_loss") is not None:
            losses.append(float(state["alignment_loss"]))
    if losses:
        merged["alignment_loss"] = float(np.mean(losses))
    anchor_modes = [state.get("camera_anchor_mode") for state in clip_states if state.get("camera_anchor_mode") is not None]
    if anchor_modes:
        merged["camera_anchor_mode"] = anchor_modes[0] if len(set(anchor_modes)) == 1 else "mixed"
    merged["use_camera_anchor"] = any(bool(state.get("use_camera_anchor")) for state in clip_states)
    merged["poses_fixed_to_source"] = bool(clip_states) and all(
        bool(state.get("poses_fixed_to_source")) for state in clip_states
    )
    merged["poses_fixed_to_initialization"] = bool(clip_states) and all(
        bool(state.get("poses_fixed_to_initialization")) for state in clip_states
    )
    return merged


def run_fine_alignment(
    dataset: SceneDatadirDataset,
    clips: Sequence[Sequence[str]],
    fine_pairs_per_clip: Sequence[Sequence[Pair]],
    coarse_state: Dict[str, Any],
    inferencer: PairwiseInferencer,
    niter: int = 300,
    schedule: str = "linear",
    lr: float = 0.01,
    temporal_smoothing_weight: float = 0.0,
    camera_pose_prior_weight: float = 0.0,
    camera_intrinsics_prior_weight: float = 0.0,
    camera_pose_prior_translation_weight: float = 1.0,
    camera_anchor_mode: str = "init",
    previous_aligned_state: Optional[Dict[str, Any]] = None,
    coarse_pointmap_teacher_weight: float = 0.0,
    coarse_depth_teacher_weight: float = 0.0,
    coarse_teacher_min_confidence: float = 3.0,
    force_recompute_pairs: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run local clip alignments directly in coarse canonical space."""

    # Fine stage policy applies to the coarse/overlap/source camera package:
    # - init: initialize from those cameras, then optimize poses
    # - fixed: keep those cameras fixed
    # - none: do not use known-pose initialization
    fine_camera_mode = str(camera_anchor_mode).lower()
    if fine_camera_mode not in ("none", "init", "fixed"):
        raise ValueError(
            "camera_anchor_mode must be one of 'none', 'init', or 'fixed', "
            f"got {camera_anchor_mode!r}"
        )
    clip_states: List[Dict[str, Any]] = []
    for clip_idx, clip in enumerate(clips):
        clip = list(clip)
        if len(clip) == 0:
            continue
        pairs = list(fine_pairs_per_clip[clip_idx])
        if verbose:
            print(f"[Phase 5] Fine alignment clip {clip_idx + 1}/{len(clips)}: {len(clip)} frame(s), {len(pairs)} pair(s)")
        previous_for_clip = clip_states[-1] if clip_states else previous_aligned_state
        fine_init = build_fine_clip_initialization(
            dataset=dataset,
            clip=clip,
            coarse_state=coarse_state,
            previous_aligned_state=previous_for_clip,
            teacher_min_confidence=coarse_teacher_min_confidence,
        )
        if verbose:
            meta = fine_init["meta"]
            print(
                "[Phase 5] Fine clip coarse init: "
                f"coarse_anchors={len(meta['coarse_anchor_frame_ids'])}, "
                f"previous_overlap_anchors={len(meta['previous_overlap_anchor_frame_ids'])}, "
                f"pointmap_teacher_frames={meta['num_pointmap_teacher_frames']}, "
                f"depth_teacher_frames={meta['num_depth_teacher_frames']}"
            )

        local_state = run_monst3r_alignment(
            dataset=dataset,
            frame_ids=clip,
            pairs=pairs,
            inferencer=inferencer,
            niter=niter,
            schedule=schedule,
            lr=lr,
            temporal_smoothing_weight=temporal_smoothing_weight,
            camera_pose_prior_weight=camera_pose_prior_weight,
            camera_intrinsics_prior_weight=camera_intrinsics_prior_weight,
            camera_pose_prior_translation_weight=camera_pose_prior_translation_weight,
            use_camera_anchor=True,
            camera_anchor_mode=fine_camera_mode,
            static_only_loss=False,
            camera_init_poses=fine_init["camera_init_poses"],
            camera_init_intrinsics=fine_init["camera_init_intrinsics"],
            depth_initialization=fine_init["depth_initialization"],
            coarse_pointmap_teacher=fine_init["pointmap_teacher"],
            coarse_depth_teacher=fine_init["depth_teacher"],
            coarse_geometry_teacher_mask=fine_init["geometry_teacher_mask"],
            coarse_pointmap_teacher_weight=coarse_pointmap_teacher_weight,
            coarse_depth_teacher_weight=coarse_depth_teacher_weight,
            force_recompute_pairs=force_recompute_pairs,
            verbose=verbose,
        )
        aligned_state = dict(local_state)
        aligned_state["clip_to_coarse_transform"] = np.eye(4, dtype=np.float32)
        aligned_state["post_alignment"] = {
            "sim3_enabled": False,
            "rigid_keyframe_lock_enabled": False,
            "keyframe": clip[0],
            "reason": "disabled_fine_optimizes_directly_in_coarse_space",
        }
        aligned_state["keyframe_pose_lock"] = {
            "enabled": False,
            "keyframe": clip[0],
            "reason": "disabled_post_alignment_coarse_pose_used_as_initialization",
            "pose_source": fine_init["meta"]["frames"].get(clip[0], {}).get("pose_source"),
        }
        aligned_state["fine_coarse_initialization"] = fine_init["meta"]
        clip_states.append(aligned_state)

    merged = merge_clip_states(clip_states)
    merged["camera_anchor_mode"] = fine_camera_mode
    merged["use_camera_anchor"] = fine_camera_mode != "none"
    merged["poses_fixed_to_initialization"] = fine_camera_mode == "fixed"
    merged["temporal_smoothing_weight"] = float(temporal_smoothing_weight)
    merged["camera_pose_prior_weight"] = float(camera_pose_prior_weight)
    merged["camera_intrinsics_prior_weight"] = float(camera_intrinsics_prior_weight)
    merged["camera_pose_prior_translation_weight"] = float(camera_pose_prior_translation_weight)
    merged["coarse_pointmap_teacher_weight"] = float(coarse_pointmap_teacher_weight)
    merged["coarse_depth_teacher_weight"] = float(coarse_depth_teacher_weight)
    merged["coarse_teacher_min_confidence"] = float(coarse_teacher_min_confidence)
    merged["keyframe_pose_lock"] = {
        "enabled": False,
        "camera_source": "coarse_initialization_plus_teacher_loss",
        "reason": "post_alignment_disabled_no_sim3_or_rigid_keyframe_retarget",
        "clips": [dict(s.get("keyframe_pose_lock", {})) for s in clip_states],
    }
    merged["post_alignment"] = {
        "sim3_enabled": False,
        "rigid_keyframe_lock_enabled": False,
        "reason": "fine_clips_are_initialized_and_constrained_in_coarse_space",
    }
    merged["fine_coarse_initialization"] = [dict(s.get("fine_coarse_initialization", {})) for s in clip_states]
    return merged
