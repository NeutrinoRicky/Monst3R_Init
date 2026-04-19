"""Compare original and optimized camera extrinsics frame-by-frame.

Outputs:
1) per-frame extrinsic errors (JSON + CSV)
2) one figure with two 3D subplots:
   - left: original extrinsics
   - right: optimized extrinsics
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from reflow_a1.dataset_scene import parse_camera_json
else:
    from .dataset_scene import parse_camera_json


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


def _ordered_filenames(optimized_dir: Path, optimized_index: Path | None) -> List[str]:
    if optimized_index is not None and optimized_index.exists():
        with optimized_index.open("r", encoding="utf-8") as f:
            data = json.load(f)
        names = []
        for item in data:
            path = item.get("camera_path")
            if path is None:
                continue
            names.append(Path(path).name)
        if names:
            return names
    return sorted(p.name for p in optimized_dir.glob("*.json"))


def _resolve_dirs(args: argparse.Namespace) -> Tuple[Path, Path, Path]:
    scene_root = Path(args.scene_root).expanduser().resolve() if args.scene_root else None

    if args.original_camera_dir is not None:
        original_dir = Path(args.original_camera_dir).expanduser().resolve()
    elif scene_root is not None:
        original_dir = scene_root / "camera"
    else:
        raise ValueError("Need --original_camera_dir, or provide --scene_root.")

    if args.optimized_camera_dir is not None:
        optimized_dir = Path(args.optimized_camera_dir).expanduser().resolve()
    elif scene_root is not None:
        optimized_dir = scene_root / "monst3r_reflow_a1_outputs" / "optimized_camera"
    else:
        raise ValueError("Need --optimized_camera_dir, or provide --scene_root.")

    if args.output_dir is not None:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        output_dir = optimized_dir.parent
    return original_dir, optimized_dir, output_dir


def _compute_stats(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "max": 0.0, "min": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "max": float(np.max(arr)),
        "min": float(np.min(arr)),
    }


def _plot_split_extrinsics(
    plot_path: Path,
    centers_orig: np.ndarray,
    centers_opt: np.ndarray,
    rots_orig: np.ndarray,
    rots_opt: np.ndarray,
    axis_stride: int = 8,
    axis_scale: float = 0.0,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib with Agg backend is required for plotting.") from exc

    axis_stride = max(1, int(axis_stride))
    all_centers = np.concatenate([centers_orig, centers_opt], axis=0)
    extent = float(np.max(np.nanmax(all_centers, axis=0) - np.nanmin(all_centers, axis=0)))
    if not np.isfinite(extent) or extent <= 0:
        extent = 1.0
    if axis_scale <= 0:
        axis_scale = extent * 0.035

    fig = plt.figure(figsize=(16, 7), dpi=150)
    ax_left = fig.add_subplot(1, 2, 1, projection="3d")
    ax_right = fig.add_subplot(1, 2, 2, projection="3d")

    color_values = np.linspace(0.0, 1.0, len(centers_orig), dtype=np.float32)
    cmap = "viridis"

    def draw_subplot(ax: Any, centers: np.ndarray, rots: np.ndarray, title: str) -> None:
        scatter = ax.scatter(
            centers[:, 0],
            centers[:, 1],
            centers[:, 2],
            c=color_values,
            cmap=cmap,
            s=22,
            alpha=0.95,
            marker="o",
        )
        ax.scatter(
            [centers[0, 0]],
            [centers[0, 1]],
            [centers[0, 2]],
            c="#1c8c4a",
            s=70,
            marker="o",
            edgecolors="black",
            linewidths=0.5,
            label="start",
        )
        ax.scatter(
            [centers[-1, 0]],
            [centers[-1, 1]],
            [centers[-1, 2]],
            c="#a02424",
            s=70,
            marker="o",
            edgecolors="black",
            linewidths=0.5,
            label="end",
        )
        for idx in range(0, len(centers), axis_stride):
            forward = _project_rotation(rots[idx])[:, 2]
            norm = np.linalg.norm(forward)
            if norm <= 1e-8:
                continue
            forward = forward / norm
            ax.quiver(
                centers[idx, 0],
                centers[idx, 1],
                centers[idx, 2],
                forward[0],
                forward[1],
                forward[2],
                length=axis_scale,
                color="black",
                linewidth=0.7,
                arrow_length_ratio=0.25,
            )
        _set_equal_3d_axes(ax, all_centers)
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.view_init(elev=24, azim=-58)
        ax.legend(loc="upper left")
        return scatter

    scatter_left = draw_subplot(ax_left, centers_orig, rots_orig, "Original Camera Extrinsics")
    draw_subplot(ax_right, centers_opt, rots_opt, "Optimized Camera Extrinsics")
    cbar = fig.colorbar(scatter_left, ax=[ax_left, ax_right], shrink=0.82, pad=0.04)
    cbar.set_label("Frame Order (normalized)")
    fig.suptitle("Camera Extrinsics: Original vs Optimized (separate 3D subplots)", y=0.98)
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare camera extrinsics and visualize side-by-side 3D subplots.")
    parser.add_argument("--scene_root", type=str, default=None, help="Scene root (used to infer default camera dirs).")
    parser.add_argument("--original_camera_dir", type=str, default=None, help="Directory with original camera json files.")
    parser.add_argument("--optimized_camera_dir", type=str, default=None, help="Directory with optimized camera json files.")
    parser.add_argument(
        "--optimized_index",
        type=str,
        default=None,
        help="Optional optimized_camera_index.json to preserve frame order.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for error reports and plot (default: parent of optimized_camera_dir).",
    )
    parser.add_argument("--axis_stride", type=int, default=8, help="Draw orientation arrow every N frames.")
    parser.add_argument("--axis_scale", type=float, default=0.0, help="Arrow length in world units (<=0 auto).")
    parser.add_argument("--topk", type=int, default=10, help="Top-K worst frames to include in summary.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    original_dir, optimized_dir, output_dir = _resolve_dirs(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not original_dir.exists():
        raise FileNotFoundError(f"Original camera dir does not exist: {original_dir}")
    if not optimized_dir.exists():
        raise FileNotFoundError(f"Optimized camera dir does not exist: {optimized_dir}")

    optimized_index = Path(args.optimized_index).expanduser().resolve() if args.optimized_index else None
    ordered = _ordered_filenames(optimized_dir, optimized_index)
    if not ordered:
        raise RuntimeError(f"No camera json files found in optimized dir: {optimized_dir}")

    records: List[Dict[str, Any]] = []
    centers_orig = []
    centers_opt = []
    rots_orig = []
    rots_opt = []
    missing_original = []

    for frame_idx, name in enumerate(ordered):
        opt_path = optimized_dir / name
        orig_path = original_dir / name
        if not opt_path.exists():
            continue
        if not orig_path.exists():
            missing_original.append(name)
            continue

        cam_orig = parse_camera_json(orig_path)
        cam_opt = parse_camera_json(opt_path)
        T_orig = np.asarray(cam_orig["T_wc"], dtype=np.float32)
        T_opt = np.asarray(cam_opt["T_wc"], dtype=np.float32)
        R_orig = _project_rotation(T_orig[:3, :3])
        R_opt = _project_rotation(T_opt[:3, :3])
        t_orig = T_orig[:3, 3]
        t_opt = T_opt[:3, 3]

        translation_error_l2 = float(np.linalg.norm(t_opt - t_orig))
        rotation_error_deg = _rotation_error_deg(R_orig, R_opt)
        extrinsic_frobenius = float(np.linalg.norm(T_opt - T_orig))

        records.append(
            {
                "frame_index": int(frame_idx),
                "camera_file": name,
                "translation_error_l2": translation_error_l2,
                "rotation_error_deg": rotation_error_deg,
                "extrinsic_frobenius": extrinsic_frobenius,
                "original_position": t_orig.astype(float).tolist(),
                "optimized_position": t_opt.astype(float).tolist(),
            }
        )
        centers_orig.append(t_orig.astype(np.float32))
        centers_opt.append(t_opt.astype(np.float32))
        rots_orig.append(R_orig.astype(np.float32))
        rots_opt.append(R_opt.astype(np.float32))

    if not records:
        raise RuntimeError(
            "No comparable camera files found between original and optimized dirs. "
            f"original={original_dir}, optimized={optimized_dir}"
        )

    centers_orig_arr = np.stack(centers_orig, axis=0)
    centers_opt_arr = np.stack(centers_opt, axis=0)
    rots_orig_arr = np.stack(rots_orig, axis=0)
    rots_opt_arr = np.stack(rots_opt, axis=0)

    plot_path = output_dir / "camera_extrinsics_split_plot.png"
    _plot_split_extrinsics(
        plot_path=plot_path,
        centers_orig=centers_orig_arr,
        centers_opt=centers_opt_arr,
        rots_orig=rots_orig_arr,
        rots_opt=rots_opt_arr,
        axis_stride=args.axis_stride,
        axis_scale=args.axis_scale,
    )

    trans = [r["translation_error_l2"] for r in records]
    rot = [r["rotation_error_deg"] for r in records]
    fro = [r["extrinsic_frobenius"] for r in records]
    records_sorted = sorted(records, key=lambda r: (r["translation_error_l2"], r["rotation_error_deg"]), reverse=True)
    topk = max(1, int(args.topk))
    summary = {
        "num_compared_frames": int(len(records)),
        "missing_original_files": missing_original,
        "translation_error_l2": _compute_stats(trans),
        "rotation_error_deg": _compute_stats(rot),
        "extrinsic_frobenius": _compute_stats(fro),
        "topk_worst_frames": records_sorted[:topk],
        "plot_path": str(plot_path),
    }

    json_path = output_dir / "camera_extrinsics_error.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_frame": records}, f, indent=2)

    csv_path = output_dir / "camera_extrinsics_error.csv"
    fields = [
        "frame_index",
        "camera_file",
        "translation_error_l2",
        "rotation_error_deg",
        "extrinsic_frobenius",
        "original_position",
        "optimized_position",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in records:
            row_out = dict(row)
            row_out["original_position"] = " ".join(f"{v:.8f}" for v in row["original_position"])
            row_out["optimized_position"] = " ".join(f"{v:.8f}" for v in row["optimized_position"])
            writer.writerow(row_out)

    print(f"[Done] Compared {len(records)} frame(s)")
    print(f"[Done] Error JSON: {json_path}")
    print(f"[Done] Error CSV: {csv_path}")
    print(f"[Done] Split 3D plot: {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
