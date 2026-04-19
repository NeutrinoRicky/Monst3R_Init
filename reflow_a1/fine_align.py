"""Clip-wise fine alignment for ReFlow A.1."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .coarse_align import run_monst3r_alignment, state_from_dataset_depth
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
    if keyframe not in coarse_state.get("global_pointmaps", {}):
        return local_state
    try:
        transform = estimate_similarity_from_pointmaps(
            local_state["global_pointmaps"][keyframe],
            coarse_state["global_pointmaps"][keyframe],
            src_mask=local_state.get("valid_masks", {}).get(keyframe),
            dst_mask=coarse_state.get("valid_masks", {}).get(keyframe),
        )
    except Exception:
        transform = pose_transform_from_keyframe(local_state, coarse_state, keyframe)
    aligned = transform_alignment_state(local_state, transform)
    aligned["clip_to_coarse_transform"] = transform
    return aligned


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
    force_recompute_pairs: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run local clip alignments and place every clip in coarse canonical space."""

    # Fine stage policy can be:
    # - init: source-initialized cameras, optimize poses/intrinsics
    # - fixed: keep source cameras fixed
    # - none: no source-camera anchoring
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
        if len(clip) == 1 or len(pairs) == 0:
            local_state = state_from_dataset_depth(dataset, clip)
        else:
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
                force_recompute_pairs=force_recompute_pairs,
                verbose=verbose,
            )
        # The keyframe coarse result fixes the clip gauge; local frames then
        # fill in the canonical space around that keyframe.
        aligned_state = align_clip_to_coarse(local_state, coarse_state, clip[0])
        aligned_state = enforce_keyframe_pose_to_coarse(aligned_state, coarse_state, clip[0])
        clip_states.append(aligned_state)

    merged = merge_clip_states(clip_states)
    merged["camera_anchor_mode"] = fine_camera_mode
    merged["use_camera_anchor"] = fine_camera_mode != "none"
    merged["poses_fixed_to_source"] = fine_camera_mode == "fixed"
    merged["temporal_smoothing_weight"] = float(temporal_smoothing_weight)
    merged["camera_pose_prior_weight"] = float(camera_pose_prior_weight)
    merged["camera_intrinsics_prior_weight"] = float(camera_intrinsics_prior_weight)
    merged["camera_pose_prior_translation_weight"] = float(camera_pose_prior_translation_weight)
    merged["keyframe_pose_lock"] = {
        "enabled": bool(clip_states) and all(bool(s.get("keyframe_pose_lock", {}).get("enabled")) for s in clip_states),
        "camera_source": "coarse_fixed_source_colmap",
        "clips": [dict(s.get("keyframe_pose_lock", {})) for s in clip_states],
    }
    return merged
