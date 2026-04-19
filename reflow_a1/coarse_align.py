"""Coarse keyframe alignment for ReFlow A.1 style construction."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .dataset_scene import SceneDataError, SceneDatadirDataset
from .monst3r_bridge import ensure_monst3r_imports
from .pair_infer import PairwiseInferencer

Pair = Tuple[str, str]
CameraAnchorMode = str


def _to_numpy(value: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (list, tuple)):
        return [_to_numpy(v) for v in value]
    return value


def _as_bool_mask_2d(mask: Any) -> Optional[np.ndarray]:
    if mask is None:
        return None
    arr = np.asarray(_to_numpy(mask))
    # Common dynamic-mask layouts here are [1,H,W] or [H,W,1].
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2:
        return None
    return arr.astype(bool)


def _apply_static_mask_to_pair_confidence(
    batch: Any,
    inferencer: PairwiseInferencer,
    neutral_conf: float = 1.0,
) -> Dict[str, int]:
    """Downweight dynamic pixels in pairwise confidence so coarse loss is static-focused.

    In Dust3R's default conf transform (`log`), `conf=1.0` gives zero weight.
    """

    masked_a = 0
    masked_b = 0
    changed_pairs = 0
    for result in batch.pair_results:
        frame_a, frame_b = result["pair"]
        conf_a = np.asarray(result["Ca_in_a"], dtype=np.float32)
        conf_b = np.asarray(result["Cb_in_a"], dtype=np.float32)
        dyn_a = _as_bool_mask_2d(batch.views_by_id.get(frame_a, {}).get("dynamic_mask"))
        dyn_b = _as_bool_mask_2d(batch.views_by_id.get(frame_b, {}).get("dynamic_mask"))

        changed = False
        if dyn_a is not None and dyn_a.shape == conf_a.shape:
            count = int(np.count_nonzero(dyn_a))
            if count > 0:
                conf_a = conf_a.copy()
                conf_a[dyn_a] = float(neutral_conf)
                result["Ca_in_a"] = conf_a
                masked_a += count
                changed = True
        if dyn_b is not None and dyn_b.shape == conf_b.shape:
            count = int(np.count_nonzero(dyn_b))
            if count > 0:
                conf_b = conf_b.copy()
                conf_b[dyn_b] = float(neutral_conf)
                result["Cb_in_a"] = conf_b
                masked_b += count
                changed = True
        if changed:
            changed_pairs += 1

    # Rebuild collated tensors from the masked pair results.
    batch.output = inferencer._assemble_dust3r_output(batch.views_by_id, batch.pair_results)
    return {
        "enabled": True,
        "neutral_conf": float(neutral_conf),
        "changed_pairs": int(changed_pairs),
        "masked_pixels_view1": int(masked_a),
        "masked_pixels_view2": int(masked_b),
    }


def backproject_depth_to_world(depth: np.ndarray, K: np.ndarray, T_wc: np.ndarray) -> np.ndarray:
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    h, w = depth.shape
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    z = depth.astype(np.float32)
    x = (xx - K[0, 2]) / K[0, 0] * z
    y = (yy - K[1, 2]) / K[1, 1] * z
    pts_cam = np.stack([x, y, z], axis=-1)
    pts_h = np.concatenate([pts_cam, np.ones((h, w, 1), dtype=np.float32)], axis=-1)
    pts_world = pts_h @ T_wc.T
    return pts_world[..., :3].astype(np.float32)


def state_from_dataset_depth(dataset: SceneDatadirDataset, frame_ids: Sequence[str]) -> Dict[str, Any]:
    """Fallback state for singleton clips.

    The normal A.1 path goes through MonST3R pairwise pointmaps. This fallback is
    only for degenerate one-frame graphs where no pairwise geometry exists.
    """

    pointmaps: Dict[str, np.ndarray] = {}
    poses: Dict[str, np.ndarray] = {}
    intrinsics: Dict[str, np.ndarray] = {}
    depths: Dict[str, np.ndarray] = {}
    confidence: Dict[str, np.ndarray] = {}
    colors: Dict[str, np.ndarray] = {}
    dynamic_masks: Dict[str, np.ndarray] = {}
    valid_masks: Dict[str, np.ndarray] = {}

    for frame_id in frame_ids:
        frame = dataset.get_frame_by_id(frame_id)
        if frame["depth"] is None:
            raise SceneDataError(
                f"{frame_id}: no reliable dataset depth is available for the singleton/no-pair fallback. "
                "Use at least two frames per clip/keyframe alignment so MonST3R can estimate pointmaps, "
                "or provide a trusted depth/ directory."
            )
        cam = frame["camera_dict"]
        pts = backproject_depth_to_world(frame["depth"], cam["K"], cam["T_wc"])
        valid = np.isfinite(pts).all(axis=-1) & np.isfinite(frame["depth"]) & (frame["depth"] > 0)
        pointmaps[frame_id] = pts
        poses[frame_id] = cam["T_wc"].astype(np.float32)
        intrinsics[frame_id] = cam["K"].astype(np.float32)
        depths[frame_id] = frame["depth"].astype(np.float32)
        confidence[frame_id] = valid.astype(np.float32)
        colors[frame_id] = frame["rgb"].astype(np.float32) / 255.0
        dynamic_masks[frame_id] = frame["dynamic_mask"].astype(bool)
        valid_masks[frame_id] = valid

    return {
        "frame_ids": list(frame_ids),
        "global_pointmaps": pointmaps,
        "poses": poses,
        "intrinsics": intrinsics,
        "depths": depths,
        "confidence": confidence,
        "colors": colors,
        "dynamic_masks": dynamic_masks,
        "valid_masks": valid_masks,
        "alignment_loss": None,
        "source": "dataset_depth_singleton_fallback",
    }


def resolve_camera_anchor_mode(use_camera_anchor: bool = False, camera_anchor_mode: Optional[str] = None) -> CameraAnchorMode:
    """Resolve the legacy boolean anchor flag into an explicit pose mode."""

    mode = "auto" if camera_anchor_mode is None else str(camera_anchor_mode).lower()
    if mode == "auto":
        return "init" if use_camera_anchor else "none"
    if mode not in ("none", "init", "fixed"):
        raise ValueError(
            "camera_anchor_mode must be one of 'auto', 'none', 'init', or 'fixed', "
            f"got {camera_anchor_mode!r}"
        )
    return mode


def _known_camera_inputs(batch: Any, frame_ids: Sequence[str]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    intrinsics = []
    poses = []
    for frame_id in frame_ids:
        view = batch.views_by_id[frame_id]
        intrinsics.append(np.asarray(_to_numpy(view["camera_intrinsics"][0]), dtype=np.float32))
        poses.append(np.asarray(_to_numpy(view["camera_pose"][0]), dtype=np.float32))
    return intrinsics, poses


def _preset_source_intrinsics(scene: Any, known_intrinsics: Sequence[np.ndarray]) -> None:
    # Avoid numpy scalar assignment surprises inside optimizer._set_focal by
    # passing native Python floats for focal initialization.
    known_focals = [float(K.diagonal()[:2].mean()) for K in known_intrinsics]
    known_pp = [np.asarray(K[:2, 2], dtype=np.float32) for K in known_intrinsics]
    scene.preset_focal(known_focals)
    scene.preset_principal_point(known_pp)


def _compute_fixed_pose_alignment(
    scene: Any,
    known_intrinsics: Sequence[np.ndarray],
    known_poses: Sequence[np.ndarray],
    niter: int,
    schedule: str,
    lr: float,
) -> Any:
    """Run MonST3R point/depth optimization while keeping image poses fixed."""

    import dust3r.cloud_opt.init_im_poses as init_fun

    _preset_source_intrinsics(scene, known_intrinsics)
    scene.preset_pose(known_poses, requires_grad=False)
    init_fun.init_from_known_poses(scene, min_conf_thr=scene.min_conf_thr)
    return scene.compute_global_alignment(init=None, niter=int(niter), schedule=schedule, lr=lr)


def extract_alignment_state(scene: Any, frame_ids: Sequence[str], loss: Optional[Any] = None) -> Dict[str, Any]:
    pts3d = _to_numpy(scene.get_pts3d())
    poses = _to_numpy(scene.get_im_poses())
    intrinsics = _to_numpy(scene.get_intrinsics())
    depths = _to_numpy(scene.get_depthmaps())
    masks = _to_numpy(scene.get_masks())
    raw_conf = _to_numpy(list(scene.im_conf))
    colors = _to_numpy(scene.imgs) if scene.imgs is not None else [None] * len(frame_ids)
    dyn = _to_numpy(scene.dynamic_masks) if scene.dynamic_masks is not None else [None] * len(frame_ids)

    state = {
        "frame_ids": list(frame_ids),
        "global_pointmaps": {},
        "poses": {},
        "intrinsics": {},
        "depths": {},
        "confidence": {},
        "colors": {},
        "dynamic_masks": {},
        "valid_masks": {},
        "alignment_loss": float(_to_numpy(loss)) if loss is not None and np.isfinite(_to_numpy(loss)).all() else None,
        "source": "monst3r_global_alignment",
    }
    for idx, frame_id in enumerate(frame_ids):
        state["global_pointmaps"][frame_id] = np.asarray(pts3d[idx], dtype=np.float32)
        state["poses"][frame_id] = np.asarray(poses[idx], dtype=np.float32)
        state["intrinsics"][frame_id] = np.asarray(intrinsics[idx], dtype=np.float32)
        state["depths"][frame_id] = np.asarray(depths[idx], dtype=np.float32)
        state["confidence"][frame_id] = np.asarray(raw_conf[idx], dtype=np.float32)
        state["colors"][frame_id] = None if colors[idx] is None else np.asarray(colors[idx], dtype=np.float32)
        state["dynamic_masks"][frame_id] = None if dyn[idx] is None else np.asarray(dyn[idx]).astype(bool)
        state["valid_masks"][frame_id] = np.asarray(masks[idx]).astype(bool)
    return state


def run_monst3r_alignment(
    dataset: SceneDatadirDataset,
    frame_ids: Sequence[str],
    pairs: Sequence[Pair],
    inferencer: PairwiseInferencer,
    niter: int = 300,
    schedule: str = "linear",
    lr: float = 0.01,
    temporal_smoothing_weight: float = 0.0,
    camera_pose_prior_weight: float = 0.0,
    camera_intrinsics_prior_weight: float = 0.0,
    camera_pose_prior_translation_weight: float = 1.0,
    use_camera_anchor: bool = False,
    camera_anchor_mode: Optional[str] = None,
    static_only_loss: bool = False,
    force_recompute_pairs: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    frame_ids = list(frame_ids)
    pairs = list(pairs)
    resolved_camera_anchor_mode = resolve_camera_anchor_mode(use_camera_anchor, camera_anchor_mode)
    if len(frame_ids) == 0:
        raise ValueError("Cannot align an empty frame set")
    if len(frame_ids) == 1 or len(pairs) == 0:
        state = state_from_dataset_depth(dataset, frame_ids)
        state["use_camera_anchor"] = resolved_camera_anchor_mode != "none"
        state["camera_anchor_mode"] = resolved_camera_anchor_mode
        state["poses_fixed_to_source"] = resolved_camera_anchor_mode == "fixed"
        state["static_only_loss"] = bool(static_only_loss)
        return state

    ensure_monst3r_imports()
    from dust3r.cloud_opt import GlobalAlignerMode, global_aligner

    batch = inferencer.infer_pairs(dataset, frame_ids, pairs, force_recompute=force_recompute_pairs)
    static_mask_meta = None
    if static_only_loss:
        static_mask_meta = _apply_static_mask_to_pair_confidence(batch, inferencer)
    scene = global_aligner(
        batch.output,
        device=inferencer.device,
        mode=GlobalAlignerMode.PointCloudOptimizer,
        verbose=verbose,
        shared_focal=False,
        optimize_pp=True,
        flow_loss_weight=0.0,
        temporal_smoothing_weight=float(temporal_smoothing_weight),
        translation_weight=0.1,
        camera_pose_prior_weight=float(camera_pose_prior_weight),
        camera_intrinsics_prior_weight=float(camera_intrinsics_prior_weight),
        camera_pose_prior_translation_weight=float(camera_pose_prior_translation_weight),
        num_total_iter=max(int(niter), 1),
    )

    known_intrinsics: List[np.ndarray] = []
    known_poses: List[np.ndarray] = []
    init = "mst"
    if resolved_camera_anchor_mode in ("init", "fixed"):
        known_intrinsics, known_poses = _known_camera_inputs(batch, frame_ids)
    if resolved_camera_anchor_mode == "init":
        _preset_source_intrinsics(scene, known_intrinsics)
        init = "known_poses"
        loss = scene.compute_global_alignment(init=init, niter=int(niter), schedule=schedule, lr=lr)
    elif resolved_camera_anchor_mode == "fixed":
        loss = _compute_fixed_pose_alignment(
            scene=scene,
            known_intrinsics=known_intrinsics,
            known_poses=known_poses,
            niter=niter,
            schedule=schedule,
            lr=lr,
        )
    else:
        loss = scene.compute_global_alignment(init=init, niter=int(niter), schedule=schedule, lr=lr)

    state = extract_alignment_state(scene, frame_ids, loss=loss)
    if resolved_camera_anchor_mode == "fixed":
        # Keep exported/compared poses byte-for-byte tied to the source camera
        # convention instead of a quaternion round-trip inside the optimizer.
        state["poses"] = {
            frame_id: dataset.get_frame_by_id(frame_id)["camera_dict"]["T_wc"].astype(np.float32)
            for frame_id in frame_ids
        }
        state["intrinsics"] = {
            frame_id: dataset.get_frame_by_id(frame_id)["camera_dict"]["K"].astype(np.float32)
            for frame_id in frame_ids
        }
    state["pair_cache"] = batch.stats.as_dict()
    state["num_pairs"] = len(pairs)
    state["use_camera_anchor"] = resolved_camera_anchor_mode != "none"
    state["camera_anchor_mode"] = resolved_camera_anchor_mode
    state["poses_fixed_to_source"] = resolved_camera_anchor_mode == "fixed"
    state["static_only_loss"] = bool(static_only_loss)
    state["temporal_smoothing_weight"] = float(temporal_smoothing_weight)
    state["camera_pose_prior_weight"] = float(camera_pose_prior_weight)
    state["camera_intrinsics_prior_weight"] = float(camera_intrinsics_prior_weight)
    state["camera_pose_prior_translation_weight"] = float(camera_pose_prior_translation_weight)
    if static_mask_meta is not None:
        state["static_loss_masking"] = static_mask_meta
    return state


def run_coarse_alignment(
    dataset: SceneDatadirDataset,
    keyframes: Sequence[str],
    coarse_pairs: Sequence[Pair],
    inferencer: PairwiseInferencer,
    niter: int = 300,
    schedule: str = "linear",
    lr: float = 0.01,
    force_recompute_pairs: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Align the keyframe graph into a coarse canonical skeleton."""

    # This is the ReFlow A.1 keyframe-level coarse stage: it gives a global
    # skeleton before each clip fills in local temporal detail.
    return run_monst3r_alignment(
        dataset=dataset,
        frame_ids=list(keyframes),
        pairs=list(coarse_pairs),
        inferencer=inferencer,
        niter=niter,
        schedule=schedule,
        lr=lr,
        use_camera_anchor=True,
        camera_anchor_mode="fixed",
        static_only_loss=True,
        force_recompute_pairs=force_recompute_pairs,
        verbose=verbose,
    )
