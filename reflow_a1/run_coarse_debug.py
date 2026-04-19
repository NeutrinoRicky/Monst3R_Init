"""Run only ReFlow A.1 coarse keyframe alignment and export debug artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from reflow_a1.backproject_split import export_optimized_cameras, visualize_camera_trajectories
    from reflow_a1.coarse_align import run_coarse_alignment
    from reflow_a1.dataset_scene import SceneDatadirDataset
    from reflow_a1.export_ply import write_ply
    from reflow_a1.pair_infer import PairwiseInferencer
    from reflow_a1.pair_sampler import build_reflow_a1_pair_graph
else:
    from .backproject_split import export_optimized_cameras, visualize_camera_trajectories
    from .coarse_align import run_coarse_alignment
    from .dataset_scene import SceneDatadirDataset
    from .export_ply import write_ply
    from .pair_infer import PairwiseInferencer
    from .pair_sampler import build_reflow_a1_pair_graph


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Debug-only runner for ReFlow A.1 coarse alignment")
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
        help="Confidence threshold used when exporting coarse_keyframes.ply",
    )
    parser.add_argument(
        "--max_points_per_frame",
        default=200000,
        type=int,
        help="Optional random cap per keyframe for coarse_keyframes.ply (<=0 disables)",
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
        help="Defaults to scene_root/monst3r_reflow_a1_coarse_debug",
    )
    parser.add_argument("--dry_run", action="store_true", help="Only load data and build the coarse keyframe graph")
    return parser


def _maybe_cap(frame_ids: Sequence[str], max_frames: int | None) -> list[str]:
    frame_ids = list(frame_ids)
    if max_frames is not None and max_frames > 0:
        return frame_ids[:max_frames]
    return frame_ids


def _ensure_writable_evo_home(scene_root: Path) -> Path | None:
    try:
        default_evo = Path.home() / ".evo"
        default_evo.mkdir(parents=True, exist_ok=True)
        return None
    except OSError:
        fallback_home = scene_root / "monst3r_reflow_a1_cache" / "home"
        fallback_home.mkdir(parents=True, exist_ok=True)
        os.environ["HOME"] = str(fallback_home)
        return fallback_home


def _project_rotation(matrix: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(np.asarray(matrix, dtype=np.float64))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1.0
        R = U @ Vt
    return R.astype(np.float32)


def _rotation_error_deg(R_ref: np.ndarray, R_new: np.ndarray) -> float:
    dR = _project_rotation(R_ref).T @ _project_rotation(R_new)
    cos_theta = float(np.clip((np.trace(dR) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_theta)))


def _estimate_similarity(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
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


def _stats(values: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return {"mean": 0.0, "median": 0.0, "max": 0.0, "min": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "max": float(np.max(arr)),
        "min": float(np.min(arr)),
    }


def _as_uint8_colors(colors: np.ndarray, count: int) -> np.ndarray:
    arr = np.asarray(colors).reshape(-1, 3)
    if len(arr) != count:
        raise ValueError(f"Color count mismatch: expected {count}, got {len(arr)}")
    if np.issubdtype(arr.dtype, np.floating):
        finite = arr[np.isfinite(arr)]
        max_value = float(np.max(finite)) if finite.size else 0.0
        if max_value <= 1.001:
            arr = arr * 255.0
    return np.clip(arr, 0.0, 255.0).astype(np.uint8)


def _auto_voxel_size(points: np.ndarray, divisor: float = 1200.0, floor: float = 1e-6) -> float:
    if len(points) == 0:
        return float(floor)
    mins = np.nanmin(points, axis=0)
    maxs = np.nanmax(points, axis=0)
    extent = float(np.max(maxs - mins))
    if not np.isfinite(extent) or extent <= 0:
        return float(floor)
    return float(max(extent / max(divisor, 1.0), floor))


def _voxel_keep_max_confidence(
    points: np.ndarray,
    colors: np.ndarray,
    confidence: np.ndarray,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    info: Dict[str, Any] = {
        "enabled": bool(len(points) > 0),
        "input_points": int(len(points)),
        "output_points": int(len(points)),
        "voxel_size": None,
        "method": "max_confidence_per_voxel",
        "removed_points": 0,
    }
    if len(points) == 0:
        return points, colors, confidence, info

    voxel_size = float(voxel_size)
    if voxel_size <= 0:
        voxel_size = _auto_voxel_size(points)
    info["voxel_size"] = float(voxel_size)

    origin = np.nanmin(points, axis=0)
    keys = np.floor((points - origin) / voxel_size).astype(np.int64)
    _, inv = np.unique(keys, axis=0, return_inverse=True)

    conf_safe = np.where(np.isfinite(confidence), confidence, -np.inf).astype(np.float32)
    tie_idx = np.arange(len(points), dtype=np.int64)
    order = np.lexsort((tie_idx, -conf_safe, inv))
    sorted_inv = inv[order]
    keep_sorted = np.ones(len(order), dtype=bool)
    keep_sorted[1:] = sorted_inv[1:] != sorted_inv[:-1]
    keep_idx = np.sort(order[keep_sorted])

    out_points = points[keep_idx]
    out_colors = colors[keep_idx]
    out_conf = confidence[keep_idx]
    info["output_points"] = int(len(out_points))
    info["removed_points"] = int(info["input_points"] - info["output_points"])
    return out_points, out_colors, out_conf, info


def export_coarse_keyframe_pointcloud(
    coarse_state: Dict[str, Any],
    out_path: Path,
    min_confidence: float = 3.0,
    max_points_per_frame: int | None = 200000,
    seed: int = 1234,
    voxel_keep_max_conf: bool = False,
    voxel_size: float = 0.0,
) -> Dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    all_points: list[np.ndarray] = []
    all_colors: list[np.ndarray] = []
    all_conf: list[np.ndarray] = []
    per_frame_stats: list[Dict[str, Any]] = []

    for frame_id in coarse_state.get("frame_ids", []):
        pts = coarse_state.get("global_pointmaps", {}).get(frame_id)
        if pts is None:
            continue
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
        n_total = len(pts)
        if n_total == 0:
            continue

        valid = np.isfinite(pts).all(axis=1)
        conf_arr = None
        conf = coarse_state.get("confidence", {}).get(frame_id)
        if conf is not None:
            conf_arr = np.asarray(conf, dtype=np.float32).reshape(-1)
            if len(conf_arr) == n_total:
                max_conf = float(np.nanmax(conf_arr)) if np.isfinite(conf_arr).any() else 0.0
                if max_conf <= 1.0 and min_confidence > 1.0:
                    valid &= conf_arr > 0.5
                else:
                    valid &= conf_arr >= float(min_confidence)
        valid_mask = coarse_state.get("valid_masks", {}).get(frame_id)
        if valid_mask is not None:
            vm = np.asarray(valid_mask).reshape(-1)
            if len(vm) == n_total:
                valid &= vm.astype(bool)

        keep_idx = np.flatnonzero(valid)
        n_valid = int(len(keep_idx))
        if n_valid == 0:
            per_frame_stats.append(
                {"frame_id": frame_id, "points_total": n_total, "points_valid": 0, "points_kept": 0}
            )
            continue

        if max_points_per_frame is not None and max_points_per_frame > 0 and n_valid > max_points_per_frame:
            keep_idx = np.sort(rng.choice(keep_idx, size=int(max_points_per_frame), replace=False))

        keep_pts = pts[keep_idx]
        cols = coarse_state.get("colors", {}).get(frame_id)
        if cols is None:
            keep_cols = np.full((len(keep_pts), 3), 200, dtype=np.uint8)
        else:
            cols_u8 = _as_uint8_colors(np.asarray(cols).reshape(-1, 3), n_total)
            keep_cols = cols_u8[keep_idx]
        if conf_arr is not None and len(conf_arr) == n_total:
            keep_conf = conf_arr[keep_idx].astype(np.float32)
        else:
            keep_conf = np.ones((len(keep_pts),), dtype=np.float32)

        all_points.append(keep_pts.astype(np.float32))
        all_colors.append(keep_cols.astype(np.uint8))
        all_conf.append(keep_conf)
        per_frame_stats.append(
            {
                "frame_id": frame_id,
                "points_total": n_total,
                "points_valid": n_valid,
                "points_kept": int(len(keep_pts)),
            }
        )

    if all_points:
        points = np.concatenate(all_points, axis=0)
        colors = np.concatenate(all_colors, axis=0)
        confidence = np.concatenate(all_conf, axis=0)
    else:
        points = np.empty((0, 3), dtype=np.float32)
        colors = np.empty((0, 3), dtype=np.uint8)
        confidence = np.empty((0,), dtype=np.float32)

    voxel_meta: Dict[str, Any] = {"enabled": False}
    if voxel_keep_max_conf:
        points, colors, confidence, voxel_meta = _voxel_keep_max_confidence(
            points=points,
            colors=colors,
            confidence=confidence,
            voxel_size=float(voxel_size),
        )

    write_ply(out_path, points, colors)
    return {
        "path": str(out_path),
        "num_points": int(len(points)),
        "num_keyframes_with_points": int(sum(1 for s in per_frame_stats if s["points_kept"] > 0)),
        "voxel_max_conf_filter": voxel_meta,
        "per_frame": per_frame_stats,
    }


def export_coarse_pose_errors(
    dataset: SceneDatadirDataset,
    coarse_state: Dict[str, Any],
    out_json_path: Path,
    out_csv_path: Path,
) -> Dict[str, Any]:
    rows: list[Dict[str, Any]] = []
    source_centers: list[np.ndarray] = []
    optimized_centers: list[np.ndarray] = []

    for frame_id in coarse_state.get("frame_ids", []):
        pose_opt = coarse_state.get("poses", {}).get(frame_id)
        if pose_opt is None:
            continue
        pose_opt = np.asarray(pose_opt, dtype=np.float32)
        pose_src = np.asarray(dataset.get_frame_by_id(frame_id)["camera_dict"]["T_wc"], dtype=np.float32)

        center_src = pose_src[:3, 3]
        center_opt = pose_opt[:3, 3]
        source_centers.append(center_src)
        optimized_centers.append(center_opt)

        rows.append(
            {
                "frame_id": frame_id,
                "translation_error_l2_raw": float(np.linalg.norm(center_opt - center_src)),
                "rotation_error_deg_raw": _rotation_error_deg(pose_src[:3, :3], pose_opt[:3, :3]),
                "source_position": center_src.astype(float).tolist(),
                "optimized_position": center_opt.astype(float).tolist(),
            }
        )

    sim3_meta: Dict[str, Any] = {"enabled": False}
    if len(rows) >= 3:
        src_arr = np.asarray(source_centers, dtype=np.float32)
        opt_arr = np.asarray(optimized_centers, dtype=np.float32)
        scale, rotation, translation = _estimate_similarity(opt_arr, src_arr)
        aligned = (scale * (opt_arr @ rotation.T) + translation).astype(np.float32)
        sim3_errors = np.linalg.norm(aligned - src_arr, axis=1)
        for row, err in zip(rows, sim3_errors):
            row["camera_center_error_sim3"] = float(err)
        sim3_meta = {
            "enabled": True,
            "scale": float(scale),
            "rotation": rotation.astype(float).tolist(),
            "translation": translation.astype(float).tolist(),
            "camera_center_error_sim3": _stats(sim3_errors),
        }
    else:
        for row in rows:
            row["camera_center_error_sim3"] = None

    translation_raw = np.asarray([row["translation_error_l2_raw"] for row in rows], dtype=np.float32)
    rotation_raw = np.asarray([row["rotation_error_deg_raw"] for row in rows], dtype=np.float32)

    summary = {
        "num_compared_keyframes": int(len(rows)),
        "camera_anchor_mode": coarse_state.get("camera_anchor_mode"),
        "poses_fixed_to_source": bool(coarse_state.get("poses_fixed_to_source")),
        "translation_error_l2_raw": _stats(translation_raw),
        "rotation_error_deg_raw": _stats(rotation_raw),
        "sim3_alignment": sim3_meta,
        "per_frame": rows,
    }
    with out_json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with out_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame_id",
                "translation_error_l2_raw",
                "rotation_error_deg_raw",
                "camera_center_error_sim3",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "frame_id": row["frame_id"],
                    "translation_error_l2_raw": row["translation_error_l2_raw"],
                    "rotation_error_deg_raw": row["rotation_error_deg_raw"],
                    "camera_center_error_sim3": row["camera_center_error_sim3"],
                }
            )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    scene_root = Path(args.scene_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else scene_root / "monst3r_reflow_a1_coarse_debug"
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

    pose_error_json = output_dir / "coarse_pose_error.json"
    pose_error_csv = output_dir / "coarse_pose_error.csv"
    pose_error_meta = export_coarse_pose_errors(
        dataset=dataset,
        coarse_state=coarse_state,
        out_json_path=pose_error_json,
        out_csv_path=pose_error_csv,
    )
    print(
        "[Phase 4] Pose error summary: "
        f"raw_t_mean={pose_error_meta['translation_error_l2_raw']['mean']:.6g}, "
        f"raw_r_mean={pose_error_meta['rotation_error_deg_raw']['mean']:.6g} deg"
    )

    camera_export = export_optimized_cameras(dataset, coarse_state, output_dir)
    print(f"[Phase 4] Exported keyframe cameras: {camera_export['optimized_camera_dir']}")
    trajectory_meta: Dict[str, Any]
    try:
        trajectory_meta = visualize_camera_trajectories(dataset, coarse_state, output_dir)
        print(f"[Phase 4] Camera trajectory plot: {trajectory_meta.get('trajectory_plot')}")
    except Exception as exc:
        trajectory_meta = {"trajectory_plot": None, "error": str(exc)}
        print(f"[Phase 4] Camera trajectory plot skipped: {exc}")

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
        "pair_cache_stats": coarse_state.get("pair_cache", inferencer.cache_stats()),
        "alignment_loss": coarse_state.get("alignment_loss"),
        "coarse_keyframes_ply": coarse_ply_meta,
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
    print(f"[Done] Coarse debug summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
