"""Command line entry point for ReFlow A.1 style canonical construction."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

if __package__ in (None, ""):
    # Allow `python run_reflow_a1.py` from inside reflow_a1/. The normal package
    # entry remains `python -m reflow_a1.run_reflow_a1` from the MonST3R root.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from reflow_a1.backproject_split import export_reflow_a1_pointclouds
    from reflow_a1.coarse_align import run_coarse_alignment
    from reflow_a1.dataset_scene import SceneDatadirDataset
    from reflow_a1.fine_align import run_fine_alignment
    from reflow_a1.pair_infer import PairwiseInferencer
    from reflow_a1.pair_sampler import build_reflow_a1_pair_graph
else:
    from .backproject_split import export_reflow_a1_pointclouds
    from .coarse_align import run_coarse_alignment
    from .dataset_scene import SceneDatadirDataset
    from .fine_align import run_fine_alignment
    from .pair_infer import PairwiseInferencer
    from .pair_sampler import build_reflow_a1_pair_graph


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ReFlow A.1 style initialization using MonST3R")
    parser.add_argument("--scene_root", required=True, type=str, help="Path to scene_datadir")
    parser.add_argument("--split", default="train", type=str, help="dataset.json split prefix")
    parser.add_argument("--clip_len", default=10, type=int, help="Number of train frames per clip")
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
    parser.add_argument("--niter_fine", default=300, type=int, help="Fine clip alignment iterations")
    parser.add_argument("--schedule", default="linear", type=str, help="Fine-stage MonST3R optimizer LR schedule")
    parser.add_argument(
        "--schedule_coarse",
        default=None,
        type=str,
        help="Coarse-stage MonST3R optimizer LR schedule (defaults to --schedule)",
    )
    parser.add_argument("--lr", default=0.01, type=float, help="Alignment learning rate")
    parser.add_argument(
        "--lr_coarse",
        default=None,
        type=float,
        help="Coarse-stage alignment learning rate (coarse only, defaults to --lr when omitted)",
    )
    parser.add_argument(
        "--fine_temporal_smoothing_weight",
        default=0.0,
        type=float,
        help="Fine-stage temporal camera smoothing loss weight",
    )
    parser.add_argument(
        "--fine_camera_pose_prior_weight",
        default=0.0,
        type=float,
        help="Fine-stage camera pose prior loss weight vs source cameras",
    )
    parser.add_argument(
        "--fine_camera_intrinsics_prior_weight",
        default=0.0,
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
        "--camera_export_mode",
        default="optimized",
        choices=["optimized", "source", "both"],
        help=(
            "Camera JSON export mode. 'optimized' preserves the original behavior; "
            "'source' exports original scene cameras for PIDG; 'both' writes both but "
            "marks source cameras as the PIDG camera set."
        ),
    )
    parser.set_defaults(align_to_source_cameras=True)
    parser.add_argument(
        "--align_to_source_cameras",
        dest="align_to_source_cameras",
        action="store_true",
        help="Sim(3)-align optimized MonST3R cameras/pointmaps back to the source camera coordinate system before export",
    )
    parser.add_argument(
        "--no_align_to_source_cameras",
        dest="align_to_source_cameras",
        action="store_false",
        help="Export raw MonST3R canonical coordinates without post-aligning to source cameras",
    )
    parser.add_argument("--force_recompute_pairs", action="store_true", help="Ignore pair cache and recompute")
    parser.add_argument("--dynamic_label_mode", default="nonzero_is_dynamic", choices=["nonzero_is_dynamic", "zero_is_dynamic", "label_is_dynamic"])
    parser.add_argument("--dynamic_label", default=None, type=int, help="Label id for label_is_dynamic mode")
    parser.add_argument("--voxel_downsample", default=0.0, type=float, help="Voxel size for final point-cloud thinning")
    parser.add_argument(
        "--static_target_points",
        default=2_000_000,
        type=int,
        help="Target static point count for adaptive voxel downsampling; set <=0 to disable",
    )
    parser.add_argument(
        "--dynamic_reference_mode",
        default="best_reference",
        choices=["best_reference", "all_frames"],
        help="Use the best dynamic reference frame by mask coverage/color diversity, or aggregate all dynamic frames",
    )
    parser.add_argument(
        "--static_mv_neighbor_radius",
        default=2,
        type=int,
        help="Neighbor radius for static multi-view depth consistency filtering",
    )
    parser.add_argument(
        "--static_mv_abs_depth_error",
        default=0.02,
        type=float,
        help="Absolute depth error threshold for static multi-view consistency",
    )
    parser.add_argument(
        "--static_mv_rel_depth_error",
        default=0.03,
        type=float,
        help="Relative depth error threshold for static multi-view consistency",
    )
    parser.add_argument(
        "--static_mv_min_consistent_views",
        default=1,
        type=int,
        help="Minimum neighbor-view consistency votes to keep a static point",
    )
    parser.add_argument(
        "--static_front_surface_voxel",
        default=0.0,
        type=float,
        help="Voxel size for ray/voxel front-surface filtering of static points (<=0 uses auto)",
    )
    parser.add_argument(
        "--static_front_ray_azimuth_bins",
        default=24,
        type=int,
        help="Azimuth bins for ray quantization in front-surface filtering",
    )
    parser.add_argument(
        "--static_front_ray_elevation_bins",
        default=12,
        type=int,
        help="Elevation bins for ray quantization in front-surface filtering",
    )
    parser.add_argument(
        "--static_surface_aggregate_voxel",
        default=0.0,
        type=float,
        help="Voxel size for weighted single-surface static aggregation (<=0 uses auto)",
    )
    parser.add_argument(
        "--fine_voxel_keep_max_conf",
        "--coarse_voxel_keep_max_conf",
        dest="fine_voxel_keep_max_conf",
        action="store_true",
        help=(
            "After static aggregation in fine export, voxelize static points and keep only "
            "the max-confidence point per voxel (dynamic points are unchanged)"
        ),
    )
    parser.add_argument(
        "--fine_voxel_size",
        "--coarse_voxel_size",
        dest="fine_voxel_size",
        default=0.0,
        type=float,
        help="Voxel size for --fine_voxel_keep_max_conf (<=0 uses automatic size)",
    )
    parser.add_argument("--max_points_per_frame", default=None, type=int, help="Random cap before frame aggregation")
    parser.add_argument("--min_confidence", default=3.0, type=float, help="MonST3R confidence threshold for export")
    parser.add_argument("--output_dir", default=None, type=str, help="Defaults to scene_root/monst3r_reflow_a1_outputs")
    parser.add_argument("--max_frames", default=None, type=int, help="Optional debugging cap on split frames")
    parser.add_argument("--dry_run", action="store_true", help="Only load data and build the hierarchical pair graph")
    parser.set_defaults(export_canonical_ply=True, export_static_dynamic_ply=True)
    parser.add_argument("--export_canonical_ply", dest="export_canonical_ply", action="store_true")
    parser.add_argument("--no_export_canonical_ply", dest="export_canonical_ply", action="store_false")
    parser.add_argument("--export_static_dynamic_ply", dest="export_static_dynamic_ply", action="store_true")
    parser.add_argument("--no_export_static_dynamic_ply", dest="export_static_dynamic_ply", action="store_false")
    return parser


def _maybe_cap(frame_ids: Sequence[str], max_frames: int | None) -> list[str]:
    frame_ids = list(frame_ids)
    if max_frames is not None and max_frames > 0:
        return frame_ids[:max_frames]
    return frame_ids


def _ensure_writable_evo_home(scene_root: Path) -> Path | None:
    """Ensure evo can create ~/.evo even on read-only HOME mounts."""
    try:
        default_evo = Path.home() / ".evo"
        default_evo.mkdir(parents=True, exist_ok=True)
        return None
    except OSError:
        fallback_home = scene_root / "monst3r_reflow_a1_cache" / "home"
        fallback_home.mkdir(parents=True, exist_ok=True)
        os.environ["HOME"] = str(fallback_home)
        return fallback_home


def _estimate_similarity(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Return Sim(3) mapping `src` points to `dst` points."""

    if len(src) < 3 or len(dst) < 3:
        return 1.0, np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
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


def _apply_similarity_to_pointmap(points: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    shape = points.shape
    flat = points.reshape(-1, 3).astype(np.float32)
    out = scale * (flat @ rotation.T) + translation
    return out.reshape(shape).astype(np.float32)


def _apply_similarity_to_pose(pose: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float32)
    out = np.eye(4, dtype=np.float32)
    out[:3, :3] = rotation @ pose[:3, :3]
    out[:3, 3] = scale * (rotation @ pose[:3, 3]) + translation
    return out


def align_state_to_source_cameras(dataset: SceneDatadirDataset, state: Dict[str, Any]) -> Dict[str, Any]:
    """Place optimized MonST3R pointmaps back in the source camera coordinate system."""

    frame_ids = []
    optimized_centers = []
    source_centers = []
    for frame_id in state.get("frame_ids", []):
        pose = state.get("poses", {}).get(frame_id)
        if pose is None:
            continue
        frame = dataset.get_frame_by_id(frame_id)
        frame_ids.append(frame_id)
        optimized_centers.append(np.asarray(pose, dtype=np.float32)[:3, 3])
        source_centers.append(np.asarray(frame["camera_dict"]["T_wc"], dtype=np.float32)[:3, 3])

    out = dict(state)
    if len(frame_ids) < 3:
        out["source_camera_alignment"] = {
            "enabled": False,
            "reason": "need_at_least_3_camera_centers",
            "num_matched_cameras": int(len(frame_ids)),
        }
        return out

    optimized_arr = np.asarray(optimized_centers, dtype=np.float32)
    source_arr = np.asarray(source_centers, dtype=np.float32)
    scale, rotation, translation = _estimate_similarity(optimized_arr, source_arr)
    aligned_centers = (scale * (optimized_arr @ rotation.T) + translation).astype(np.float32)
    center_errors = np.linalg.norm(aligned_centers - source_arr, axis=1)

    out["global_pointmaps"] = {
        frame_id: _apply_similarity_to_pointmap(points, scale, rotation, translation)
        for frame_id, points in state.get("global_pointmaps", {}).items()
    }
    out["poses"] = {
        frame_id: _apply_similarity_to_pose(pose, scale, rotation, translation)
        for frame_id, pose in state.get("poses", {}).items()
    }
    out["source_camera_alignment"] = {
        "enabled": True,
        "num_matched_cameras": int(len(frame_ids)),
        "scale": float(scale),
        "rotation": rotation.astype(float).tolist(),
        "translation": translation.astype(float).tolist(),
        "camera_center_error_mean": float(np.mean(center_errors)),
        "camera_center_error_median": float(np.median(center_errors)),
        "camera_center_error_max": float(np.max(center_errors)),
    }
    return out


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    scene_root = Path(args.scene_root).expanduser().resolve()
    evo_fallback = _ensure_writable_evo_home(scene_root)
    if evo_fallback is not None:
        print(f"[Init] HOME is read-only; using writable HOME={evo_fallback}")
    print(
        "[Init] Camera policy: coarse=fixed_to_source_colmap(intrinsics+extrinsics), "
        "fine=source_initialized_but_optimized(intrinsics+extrinsics)"
    )

    dataset = SceneDatadirDataset(
        scene_root,
        split=args.split,
        dynamic_label_mode=args.dynamic_label_mode,
        dynamic_label=args.dynamic_label,
    )
    frame_ids = _maybe_cap(dataset.get_train_frame_ids(), args.max_frames)
    if not frame_ids:
        raise RuntimeError(f"No frame ids found for split {args.split}")

    desc = dataset.describe()
    print(
        f"[Phase 1] Loaded dataset: {desc['num_frames']} split frame(s), "
        f"processing {len(frame_ids)} frame(s)"
    )
    print(f"[Phase 1] train_ids head={frame_ids[:3]} tail={frame_ids[-3:]}")
    if desc["first_shapes"] is not None:
        print(f"[Phase 1] First frame shapes: {desc['first_shapes']}")

    pair_graph = build_reflow_a1_pair_graph(
        frame_ids,
        clip_len=args.clip_len,
        coarse_max_offset=args.coarse_max_offset,
        fine_max_offset=args.fine_max_offset,
    )
    print(
        f"[Phase 2] Built {len(pair_graph.clips)} clips / {len(pair_graph.keyframes)} keyframes / "
        f"{len(pair_graph.coarse_pairs)} coarse pairs / "
        f"{sum(len(p) for p in pair_graph.fine_pairs_per_clip)} fine pairs "
        f"(coarse_max_offset={args.coarse_max_offset}, fine_max_offset={args.fine_max_offset})"
    )
    if args.dry_run:
        print("[Phase 2] Dry run complete; no MonST3R inference or export was executed")
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

    coarse_schedule = args.schedule if args.schedule_coarse is None else args.schedule_coarse
    coarse_lr = args.lr if args.lr_coarse is None else args.lr_coarse
    print(
        "[Phase 4] Running coarse keyframe alignment "
        f"(niter={args.niter_coarse}, schedule={coarse_schedule}, lr={coarse_lr})"
    )
    coarse_state = run_coarse_alignment(
        dataset,
        keyframes=pair_graph.keyframes,
        coarse_pairs=pair_graph.coarse_pairs,
        inferencer=inferencer,
        niter=args.niter_coarse,
        schedule=coarse_schedule,
        lr=coarse_lr,
        force_recompute_pairs=args.force_recompute_pairs,
        verbose=True,
    )
    print(
        f"[Phase 4] Coarse alignment done: {len(coarse_state['frame_ids'])} keyframe(s), "
        f"{coarse_state.get('num_pairs', len(pair_graph.coarse_pairs))} pair(s)"
    )

    print(
        "[Phase 5] Running clip-wise fine alignment "
        f"(temporal_smooth={args.fine_temporal_smoothing_weight}, "
        f"pose_prior={args.fine_camera_pose_prior_weight}, "
        f"intr_prior={args.fine_camera_intrinsics_prior_weight}, "
        f"pose_prior_t={args.fine_camera_pose_prior_translation_weight})"
    )
    fine_state = run_fine_alignment(
        dataset,
        clips=pair_graph.clips,
        fine_pairs_per_clip=pair_graph.fine_pairs_per_clip,
        coarse_state=coarse_state,
        inferencer=inferencer,
        niter=args.niter_fine,
        schedule=args.schedule,
        lr=args.lr,
        temporal_smoothing_weight=args.fine_temporal_smoothing_weight,
        camera_pose_prior_weight=args.fine_camera_pose_prior_weight,
        camera_intrinsics_prior_weight=args.fine_camera_intrinsics_prior_weight,
        camera_pose_prior_translation_weight=args.fine_camera_pose_prior_translation_weight,
        force_recompute_pairs=args.force_recompute_pairs,
        verbose=True,
    )
    print(f"[Phase 5] Fine alignment done: {len(fine_state['frame_ids'])} frame(s)")

    if args.align_to_source_cameras:
        fine_state = align_state_to_source_cameras(dataset, fine_state)
        align_meta = fine_state.get("source_camera_alignment", {})
        if align_meta.get("enabled"):
            print(
                "[Phase 5] Aligned MonST3R state back to source camera coordinates: "
                f"scale={align_meta['scale']:.6g}, "
                f"mean_center_error={align_meta['camera_center_error_mean']:.6g}, "
                f"max_center_error={align_meta['camera_center_error_max']:.6g}"
            )
        else:
            print(f"[Phase 5] Source-camera alignment skipped: {align_meta.get('reason', 'unknown reason')}")

    output_dir = Path(args.output_dir) if args.output_dir else dataset.scene_root / "monst3r_reflow_a1_outputs"
    summary = export_reflow_a1_pointclouds(
        dataset=dataset,
        alignment_state=fine_state,
        output_dir=output_dir,
        clip_len=args.clip_len,
        num_clips=len(pair_graph.clips),
        num_keyframes=len(pair_graph.keyframes),
        image_size=args.image_size,
        use_camera_anchor=True,
        camera_anchor_mode="coarse_fixed_fine_optimize",
        pair_cache_stats=inferencer.cache_stats(),
        export_canonical_ply=args.export_canonical_ply,
        export_static_dynamic_ply=args.export_static_dynamic_ply,
        min_confidence=args.min_confidence,
        max_points_per_frame=args.max_points_per_frame,
        voxel_downsample=args.voxel_downsample,
        static_target_points=args.static_target_points,
        dynamic_reference_mode=args.dynamic_reference_mode,
        static_mv_neighbor_radius=args.static_mv_neighbor_radius,
        static_mv_abs_depth_error=args.static_mv_abs_depth_error,
        static_mv_rel_depth_error=args.static_mv_rel_depth_error,
        static_mv_min_consistent_views=args.static_mv_min_consistent_views,
        static_front_surface_voxel=args.static_front_surface_voxel,
        static_front_ray_azimuth_bins=args.static_front_ray_azimuth_bins,
        static_front_ray_elevation_bins=args.static_front_ray_elevation_bins,
        static_surface_aggregate_voxel=args.static_surface_aggregate_voxel,
        fine_voxel_keep_max_conf=args.fine_voxel_keep_max_conf,
        fine_voxel_size=args.fine_voxel_size,
        camera_export_mode=args.camera_export_mode,
    )
    print(
        f"[Phase 6] Exported static/dynamic point clouds to {output_dir} "
        f"(static={summary['num_points_static']}, dynamic={summary['num_points_dynamic']})"
    )
    if summary.get("optimized_camera_dir") is not None:
        print(f"[Phase 6] Optimized cameras: {summary['optimized_camera_dir']}")
    if summary.get("source_camera_dir") is not None:
        print(f"[Phase 6] Source cameras for PIDG: {summary['source_camera_dir']}")
    print(
        f"[Phase 6] PIDG camera source: {summary['pidg_camera_source']} "
        f"({summary['pidg_camera_dir']})"
    )
    print(f"[Phase 6] Camera trajectory comparison: {summary['camera_trajectory_plot']}")
    print(f"[Phase 6] Summary: {output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
