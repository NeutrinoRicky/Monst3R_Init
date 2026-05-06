"""Run ReFlow A.1 coarse alignment and export coarse + dynamic initialization artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from reflow_a1.backproject_split import (
        _color_diversity_score,
        _resize_float,
        _resize_mask,
        _resize_rgb,
        _select_dynamic_reference,
        _voxel_downsample,
        export_optimized_cameras,
        visualize_camera_trajectories,
    )
    from reflow_a1.coarse_align import backproject_depth_to_world, run_coarse_alignment
    from reflow_a1.dataset_scene import SceneDatadirDataset
    from reflow_a1.export_ply import write_ply
    from reflow_a1.pair_infer import PairwiseInferencer
    from reflow_a1.pair_sampler import build_reflow_a1_pair_graph
    from reflow_a1.run_coarse_debug import (
        _ensure_writable_evo_home,
        _maybe_cap,
        export_coarse_keyframe_pointcloud,
        export_coarse_pose_errors,
    )
else:
    from .backproject_split import (
        _color_diversity_score,
        _resize_float,
        _resize_mask,
        _resize_rgb,
        _select_dynamic_reference,
        _voxel_downsample,
        export_optimized_cameras,
        visualize_camera_trajectories,
    )
    from .coarse_align import backproject_depth_to_world, run_coarse_alignment
    from .dataset_scene import SceneDatadirDataset
    from .export_ply import write_ply
    from .pair_infer import PairwiseInferencer
    from .pair_sampler import build_reflow_a1_pair_graph
    from .run_coarse_debug import (
        _ensure_writable_evo_home,
        _maybe_cap,
        export_coarse_keyframe_pointcloud,
        export_coarse_pose_errors,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Coarse runner with best-frame dynamic initialization export")
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
        help="Maximum intra-clip fine edge offset used when exporting pair graph metadata",
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
    parser.add_argument("--schedule", default="linear", type=str, help="MonST3R optimizer LR schedule")
    parser.add_argument("--lr", default=0.005, type=float, help="Alignment learning rate")
    parser.add_argument("--force_recompute_pairs", action="store_true", help="Ignore pair cache and recompute")
    parser.add_argument("--max_frames", default=None, type=int, help="Optional debugging cap on split frames")
    parser.add_argument(
        "--min_confidence",
        default=3.0,
        type=float,
        help="Confidence threshold used when exporting coarse_keyframes.ply and dynamic_complete.ply",
    )
    parser.add_argument(
        "--max_points_per_frame",
        default=200000,
        type=int,
        help="Optional random cap per keyframe for exported PLYs (<=0 disables)",
    )
    parser.add_argument(
        "--dynamic_reference_mode",
        default="best_reference",
        choices=["best_reference", "all_frames"],
        help="Use the best dynamic reference keyframe or aggregate all coarse keyframe dynamic points",
    )
    parser.add_argument(
        "--dynamic_voxel_downsample",
        default=0.0,
        type=float,
        help="Optional voxel size for dynamic_complete.ply (<=0 disables)",
    )
    parser.add_argument(
        "--coarse_voxel_keep_max_conf",
        action="store_true",
        help="After merging coarse keyframe points, voxelize and keep only the highest-confidence point per voxel",
    )
    parser.add_argument(
        "--coarse_voxel_size",
        default=0.0,
        type=float,
        help="Voxel size for --coarse_voxel_keep_max_conf (<=0 uses automatic size)",
    )
    parser.add_argument("--seed", default=1234, type=int, help="Random seed for coarse point sampling")
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help="Defaults to scene_root/monst3r_reflow_a1_coarse_all",
    )
    parser.add_argument("--dry_run", action="store_true", help="Only load data and build the coarse keyframe graph")
    return parser


def _points_from_state_or_depth(
    frame: Dict[str, Any],
    alignment_state: Dict[str, Any],
    frame_id: str,
) -> np.ndarray:
    pointmap = alignment_state.get("global_pointmaps", {}).get(frame_id)
    if pointmap is not None:
        return np.asarray(pointmap, dtype=np.float32)

    if frame["depth"] is None:
        raise RuntimeError(
            f"{frame_id}: no coarse pointmap is available and dataset depth is missing, "
            "so dynamic initialization cannot be synthesized for this frame."
        )
    pose_wc = np.asarray(
        alignment_state.get("poses", {}).get(frame_id, frame["camera_dict"]["T_wc"]),
        dtype=np.float32,
    )
    return backproject_depth_to_world(frame["depth"], frame["camera_dict"]["K"], pose_wc)


def _confidence_valid_mask(conf_map: np.ndarray, min_confidence: float) -> np.ndarray:
    finite = conf_map[np.isfinite(conf_map)]
    max_conf = float(np.max(finite)) if finite.size else 0.0
    if max_conf <= 1.0 and min_confidence > 1.0:
        return conf_map > 0.5
    return conf_map >= float(min_confidence)


def _collect_dynamic_frames(
    dataset: SceneDatadirDataset,
    alignment_state: Dict[str, Any],
    min_confidence: float,
    max_points_per_frame: Optional[int],
    rng: np.random.Generator,
    use_alignment_valid_mask: bool,
    use_alignment_confidence: bool,
) -> list[Dict[str, Any]]:
    dynamic_frames = []

    for frame_id in alignment_state.get("frame_ids", []):
        frame = dataset.get_frame_by_id(frame_id)
        pts = _points_from_state_or_depth(frame, alignment_state, frame_id)
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
        if use_alignment_valid_mask:
            valid_mask = alignment_state.get("valid_masks", {}).get(frame_id)
            if valid_mask is not None:
                valid &= _resize_mask(np.asarray(valid_mask), (h, w))

        if use_alignment_confidence:
            conf = alignment_state.get("confidence", {}).get(frame_id)
            if conf is not None:
                conf_map = _resize_float(np.asarray(conf), (h, w))
                valid &= _confidence_valid_mask(conf_map, min_confidence)

        dynamic_valid_mask = valid & dyn_mask
        dynamic_colors_for_score = colors[dynamic_valid_mask]
        keep_idx = np.flatnonzero(dynamic_valid_mask.reshape(-1))
        if max_points_per_frame is not None and max_points_per_frame > 0 and len(keep_idx) > max_points_per_frame:
            keep_idx = np.sort(rng.choice(keep_idx, size=int(max_points_per_frame), replace=False))

        if len(keep_idx) == 0:
            continue

        dynamic_frames.append(
            {
                "frame_id": frame_id,
                "points": pts.reshape(-1, 3)[keep_idx].astype(np.float32),
                "colors": colors.reshape(-1, 3)[keep_idx].astype(np.float32),
                "stats": {
                    "frame_id": frame_id,
                    "total_pixels": int(h * w),
                    "valid_pixels": int(np.count_nonzero(valid)),
                    "dynamic_mask_pixels": int(np.count_nonzero(dyn_mask)),
                    "dynamic_valid_pixels": int(np.count_nonzero(dynamic_valid_mask)),
                    "dynamic_mask_ratio": float(np.count_nonzero(dyn_mask) / max(h * w, 1)),
                    "dynamic_valid_ratio": float(np.count_nonzero(dynamic_valid_mask) / max(h * w, 1)),
                    "dynamic_color_diversity": _color_diversity_score(dynamic_colors_for_score),
                    "used_alignment_valid_mask": bool(use_alignment_valid_mask),
                    "used_alignment_confidence": bool(use_alignment_confidence),
                },
            }
        )

    return dynamic_frames


def export_dynamic_reference_pointcloud(
    dataset: SceneDatadirDataset,
    alignment_state: Dict[str, Any],
    out_path: Path,
    min_confidence: float = 3.0,
    max_points_per_frame: Optional[int] = None,
    dynamic_reference_mode: str = "best_reference",
    voxel_downsample: float = 0.0,
    seed: int = 1234,
) -> Dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    strict_frames = _collect_dynamic_frames(
        dataset=dataset,
        alignment_state=alignment_state,
        min_confidence=min_confidence,
        max_points_per_frame=max_points_per_frame,
        rng=rng,
        use_alignment_valid_mask=True,
        use_alignment_confidence=True,
    )
    relaxed_static_only_fallback = False
    filter_mode = "alignment_valid_mask_and_confidence"
    dynamic_frames = strict_frames
    if not dynamic_frames and bool(alignment_state.get("static_only_loss", False)):
        # Coarse alignment deliberately zero-weights dynamic regions, so its
        # confidence-derived valid mask is static-only. For dynamic
        # initialization we therefore fall back to finite pointmap geometry.
        dynamic_frames = _collect_dynamic_frames(
            dataset=dataset,
            alignment_state=alignment_state,
            min_confidence=min_confidence,
            max_points_per_frame=max_points_per_frame,
            rng=rng,
            use_alignment_valid_mask=False,
            use_alignment_confidence=False,
        )
        relaxed_static_only_fallback = bool(dynamic_frames)
        filter_mode = "finite_points_only_after_static_only_loss"

    if dynamic_reference_mode == "all_frames":
        if dynamic_frames:
            dyn_pts = np.concatenate([frame["points"] for frame in dynamic_frames], axis=0)
            dyn_col = np.concatenate([frame["colors"] for frame in dynamic_frames], axis=0)
        else:
            dyn_pts = np.empty((0, 3), dtype=np.float32)
            dyn_col = np.empty((0, 3), dtype=np.float32)
        dynamic_reference = {
            "mode": "all_frames",
            "selected_frame_id": None,
            "reason": None if dynamic_frames else "no_dynamic_points",
            "frame_stats": [frame["stats"] for frame in dynamic_frames],
            "filter_mode": filter_mode,
            "static_only_loss": bool(alignment_state.get("static_only_loss", False)),
            "relaxed_static_only_fallback": relaxed_static_only_fallback,
        }
    elif dynamic_reference_mode == "best_reference":
        dyn_pts, dyn_col, dynamic_reference = _select_dynamic_reference(dynamic_frames)
        dynamic_reference["filter_mode"] = filter_mode
        dynamic_reference["static_only_loss"] = bool(alignment_state.get("static_only_loss", False))
        dynamic_reference["relaxed_static_only_fallback"] = relaxed_static_only_fallback
    else:
        raise ValueError(f"Unknown dynamic_reference_mode: {dynamic_reference_mode}")

    if voxel_downsample > 0:
        dyn_pts, dyn_col = _voxel_downsample(dyn_pts, dyn_col, float(voxel_downsample))

    write_ply(out_path, dyn_pts, dyn_col)
    return {
        "path": str(out_path),
        "num_points": int(len(dyn_pts)),
        "dynamic_reference_mode": str(dynamic_reference_mode),
        "dynamic_voxel_downsample": float(voxel_downsample),
        "selected_frame_id": dynamic_reference.get("selected_frame_id"),
        "reference": dynamic_reference,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    scene_root = Path(args.scene_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else scene_root / "monst3r_reflow_a1_coarse_all"
    output_dir.mkdir(parents=True, exist_ok=True)

    evo_fallback = _ensure_writable_evo_home(scene_root)
    if evo_fallback is not None:
        print(f"[Init] HOME is read-only; using writable HOME={evo_fallback}")
    print("[Init] Coarse camera policy: fixed_to_source_colmap(intrinsics+extrinsics)")

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
    pair_graph = build_reflow_a1_pair_graph(
        frame_ids,
        clip_len=args.clip_len,
        coarse_max_offset=args.coarse_max_offset,
        fine_max_offset=args.fine_max_offset,
    )
    pair_graph_json = output_dir / "coarse_pair_graph.json"
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
            },
            f,
            indent=2,
        )
    print(
        f"[Phase 1-2] Dataset frames={desc['num_frames']} processed={len(frame_ids)} "
        f"clips={len(pair_graph.clips)} keyframes={len(pair_graph.keyframes)} "
        f"coarse_pairs={len(pair_graph.coarse_pairs)} coarse_max_offset={args.coarse_max_offset} "
        f"fine_max_offset={args.fine_max_offset}"
    )
    print(f"[Phase 2] Wrote pair graph: {pair_graph_json}")
    if args.dry_run:
        print("[Phase 2] Dry run complete; coarse alignment was not executed")
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

    coarse_ply_path = output_dir / "coarse_keyframes.ply"
    coarse_ply_meta = export_coarse_keyframe_pointcloud(
        coarse_state=coarse_state,
        out_path=coarse_ply_path,
        min_confidence=float(args.min_confidence),
        max_points_per_frame=None if args.max_points_per_frame is not None and args.max_points_per_frame <= 0 else args.max_points_per_frame,
        seed=int(args.seed),
        voxel_keep_max_conf=bool(args.coarse_voxel_keep_max_conf),
        voxel_size=float(args.coarse_voxel_size),
    )
    voxel_meta = coarse_ply_meta.get("voxel_max_conf_filter", {})
    if voxel_meta.get("enabled"):
        print(
            f"[Phase 4] Exported coarse keyframe cloud: {coarse_ply_path} "
            f"({coarse_ply_meta['num_points']} points, voxel_keep_max_conf=True, "
            f"voxel={voxel_meta.get('voxel_size'):.6g}, removed={voxel_meta.get('removed_points', 0)})"
        )
    else:
        print(
            f"[Phase 4] Exported coarse keyframe cloud: {coarse_ply_path} "
            f"({coarse_ply_meta['num_points']} points)"
        )

    dynamic_ply_path = output_dir / "dynamic_complete.ply"
    dynamic_meta = export_dynamic_reference_pointcloud(
        dataset=dataset,
        alignment_state=coarse_state,
        out_path=dynamic_ply_path,
        min_confidence=float(args.min_confidence),
        max_points_per_frame=None if args.max_points_per_frame is not None and args.max_points_per_frame <= 0 else args.max_points_per_frame,
        dynamic_reference_mode=str(args.dynamic_reference_mode),
        voxel_downsample=float(args.dynamic_voxel_downsample),
        seed=int(args.seed),
    )
    dynamic_stats_path = output_dir / "dynamic_reference_stats.json"
    with dynamic_stats_path.open("w", encoding="utf-8") as f:
        json.dump(dynamic_meta["reference"], f, indent=2)
    selected_frame = dynamic_meta.get("selected_frame_id")
    if selected_frame is None:
        print(
            f"[Phase 5] Exported dynamic initialization: {dynamic_ply_path} "
            f"({dynamic_meta['num_points']} points, mode={args.dynamic_reference_mode})"
        )
    else:
        print(
            f"[Phase 5] Exported dynamic initialization: {dynamic_ply_path} "
            f"({dynamic_meta['num_points']} points, selected_frame={selected_frame})"
        )

    pose_error_json = output_dir / "coarse_pose_error.json"
    pose_error_csv = output_dir / "coarse_pose_error.csv"
    pose_error_meta = export_coarse_pose_errors(
        dataset=dataset,
        coarse_state=coarse_state,
        out_json_path=pose_error_json,
        out_csv_path=pose_error_csv,
    )
    print(
        "[Phase 5] Pose error summary: "
        f"raw_t_mean={pose_error_meta['translation_error_l2_raw']['mean']:.6g}, "
        f"raw_r_mean={pose_error_meta['rotation_error_deg_raw']['mean']:.6g} deg"
    )

    camera_export = export_optimized_cameras(dataset, coarse_state, output_dir)
    print(f"[Phase 5] Exported keyframe cameras: {camera_export['optimized_camera_dir']}")
    trajectory_meta: Dict[str, Any]
    try:
        trajectory_meta = visualize_camera_trajectories(dataset, coarse_state, output_dir)
        print(f"[Phase 5] Camera trajectory plot: {trajectory_meta.get('trajectory_plot')}")
    except Exception as exc:
        trajectory_meta = {"trajectory_plot": None, "error": str(exc)}
        print(f"[Phase 5] Camera trajectory plot skipped: {exc}")

    summary = {
        "scene_root": str(scene_root),
        "split": args.split,
        "clip_len": int(args.clip_len),
        "coarse_max_offset": int(args.coarse_max_offset),
        "fine_max_offset": int(args.fine_max_offset),
        "num_frames_processed": int(len(frame_ids)),
        "num_clips": int(len(pair_graph.clips)),
        "num_keyframes": int(len(pair_graph.keyframes)),
        "num_coarse_pairs": int(len(pair_graph.coarse_pairs)),
        "keyframes": list(pair_graph.keyframes),
        "coarse_hyperparams": {
            "image_size": int(args.image_size),
            "batch_size": int(args.batch_size),
            "coarse_max_offset": int(args.coarse_max_offset),
            "niter_coarse": int(args.niter_coarse),
            "schedule": str(args.schedule),
            "lr": float(args.lr),
            "coarse_voxel_keep_max_conf": bool(args.coarse_voxel_keep_max_conf),
            "coarse_voxel_size": float(args.coarse_voxel_size),
            "camera_policy": "fixed_source_colmap_intrinsics_extrinsics",
            "static_masked_loss": bool(coarse_state.get("static_only_loss", False)),
        },
        "dynamic_hyperparams": {
            "dynamic_reference_mode": str(args.dynamic_reference_mode),
            "dynamic_voxel_downsample": float(args.dynamic_voxel_downsample),
            "min_confidence": float(args.min_confidence),
            "max_points_per_frame": None
            if args.max_points_per_frame is not None and args.max_points_per_frame <= 0
            else int(args.max_points_per_frame),
        },
        "pair_cache_stats": coarse_state.get("pair_cache", inferencer.cache_stats()),
        "alignment_loss": coarse_state.get("alignment_loss"),
        "coarse_keyframes_ply": coarse_ply_meta,
        "dynamic_complete_ply": {
            "path": dynamic_meta["path"],
            "num_points": int(dynamic_meta["num_points"]),
            "dynamic_reference_mode": dynamic_meta["dynamic_reference_mode"],
            "dynamic_voxel_downsample": dynamic_meta["dynamic_voxel_downsample"],
        },
        "dynamic_reference": {
            key: value
            for key, value in dynamic_meta["reference"].items()
            if key != "frame_stats"
        },
        "dynamic_reference_stats": str(dynamic_stats_path),
        "coarse_pose_error": {
            "json": str(pose_error_json),
            "csv": str(pose_error_csv),
            "translation_error_l2_raw": pose_error_meta["translation_error_l2_raw"],
            "rotation_error_deg_raw": pose_error_meta["rotation_error_deg_raw"],
            "sim3_alignment": pose_error_meta.get("sim3_alignment", {}),
        },
        "optimized_camera_dir": camera_export.get("optimized_camera_dir"),
        "optimized_camera_index": camera_export.get("optimized_camera_index"),
        "camera_trajectory_plot": trajectory_meta.get("trajectory_plot"),
    }
    summary_path = output_dir / "coarse_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[Done] Coarse-all summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
