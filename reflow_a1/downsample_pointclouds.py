"""Batch downsample ASCII PLY point clouds in ReFlow A.1 outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from reflow_a1.backproject_split import _adaptive_voxel_downsample
    from reflow_a1.export_ply import write_ply
else:
    from .backproject_split import _adaptive_voxel_downsample
    from .export_ply import write_ply


def read_ascii_ply(path: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    with path.open("r", encoding="utf-8") as f:
        first = f.readline().strip()
        if first != "ply":
            raise ValueError(f"{path}: not a PLY file")

        fmt = f.readline().strip()
        if not fmt.startswith("format ascii"):
            raise ValueError(f"{path}: only ASCII PLY is supported")

        vertex_count = None
        in_vertex = False
        prop_names: list[str] = []
        header_lines = 2
        for line in f:
            header_lines += 1
            text = line.strip()
            if text.startswith("element "):
                tokens = text.split()
                in_vertex = len(tokens) >= 3 and tokens[1] == "vertex"
                if in_vertex:
                    vertex_count = int(tokens[2])
                    prop_names = []
            elif in_vertex and text.startswith("property "):
                prop_names.append(text.split()[-1])
            elif text == "end_header":
                break

    if vertex_count is None:
        raise ValueError(f"{path}: missing vertex element")
    if not prop_names:
        raise ValueError(f"{path}: missing vertex properties")

    data = np.loadtxt(path, skiprows=header_lines, max_rows=vertex_count, ndmin=2)
    if data.shape[1] < len(prop_names):
        raise ValueError(f"{path}: malformed vertex rows")

    needed = ("x", "y", "z")
    if not all(name in prop_names for name in needed):
        raise ValueError(f"{path}: vertex properties must include x/y/z")

    ix = prop_names.index("x")
    iy = prop_names.index("y")
    iz = prop_names.index("z")
    points = data[:, [ix, iy, iz]].astype(np.float32)

    colors = None
    if all(name in prop_names for name in ("red", "green", "blue")):
        ir = prop_names.index("red")
        ig = prop_names.index("green")
        ib = prop_names.index("blue")
        colors = np.clip(data[:, [ir, ig, ib]], 0, 255).astype(np.uint8)
    return points, colors


def voxel_downsample(
    points: np.ndarray,
    colors: Optional[np.ndarray],
    voxel_size: float,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if voxel_size <= 0:
        raise ValueError(f"voxel_size must be > 0, got {voxel_size}")
    if len(points) == 0:
        return points, colors
    keys = np.floor(points / voxel_size).astype(np.int64)
    _, keep = np.unique(keys, axis=0, return_index=True)
    keep = np.sort(keep)
    out_points = points[keep]
    out_colors = None if colors is None else colors[keep]
    return out_points, out_colors


def random_downsample(
    points: np.ndarray,
    colors: Optional[np.ndarray],
    target_points: int,
    seed: int,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if target_points <= 0:
        return points[:0], None if colors is None else colors[:0]
    if target_points >= len(points):
        return points, colors
    rng = np.random.default_rng(seed)
    keep = rng.choice(len(points), size=target_points, replace=False)
    keep.sort()
    out_points = points[keep]
    out_colors = None if colors is None else colors[keep]
    return out_points, out_colors


def compute_target_points(n_points: int, ratio: Optional[float], max_points: Optional[int]) -> Optional[int]:
    target = None
    if ratio is not None:
        if ratio <= 0 or ratio > 1:
            raise ValueError(f"ratio must be in (0, 1], got {ratio}")
        target = max(1, int(round(n_points * ratio)))
    if max_points is not None:
        target = max_points if target is None else min(target, max_points)
    return target


def process_file(
    ply_path: Path,
    out_path: Path,
    method: str,
    voxel_size: float,
    ratio: Optional[float],
    max_points: Optional[int],
    seed: int,
) -> Tuple[int, int]:
    points, colors = read_ascii_ply(ply_path)
    n_before = len(points)

    target = compute_target_points(n_before, ratio, max_points)
    if method == "adaptive_voxel":
        if target is None:
            raise ValueError("adaptive_voxel mode needs --ratio or --max_points")
        points, colors, info = _adaptive_voxel_downsample(points, colors, target_points=target, rng_seed=seed)
        print(
            f"  adaptive_voxel: target={target}, voxel={info.get('voxel_size')}, "
            f"passes={info.get('full_passes')}, fallback_random={info.get('fallback_random')}"
        )
    elif method == "voxel":
        points, colors = voxel_downsample(points, colors, voxel_size)
        if target is not None:
            points, colors = random_downsample(points, colors, target, seed)
    else:
        if target is None:
            raise ValueError("random mode needs --ratio or --max_points")
        points, colors = random_downsample(points, colors, target, seed)

    write_ply(out_path, points, colors)
    return n_before, len(points)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Downsample ReFlow A.1 output point clouds")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/mnt/store/fd/project/dataset/Nvidia_monocular/Balloon1/monst3r_reflow_a1_outputs",
        help="Directory containing .ply files",
    )
    parser.add_argument("--pattern", type=str, default="*.ply", help="Glob pattern for input files")
    parser.add_argument("--output_dir", type=str, default=None, help="Defaults to input_dir")
    parser.add_argument("--suffix", type=str, default="_downsampled", help="Output filename suffix")
    parser.add_argument("--method", type=str, choices=["voxel", "adaptive_voxel", "random"], default="voxel")
    parser.add_argument("--voxel_size", type=float, default=0.01, help="Voxel size for voxel mode")
    parser.add_argument("--ratio", type=float, default=None, help="Optional keep ratio in (0,1]")
    parser.add_argument("--max_points", type=int, default=None, help="Optional max point count")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite original files")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = input_dir if args.output_dir is None else Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob(args.pattern))
    if not files:
        raise SystemExit(f"No files matched: {input_dir}/{args.pattern}")

    total_before = 0
    total_after = 0
    for ply_path in files:
        if args.overwrite:
            out_path = ply_path
        else:
            out_name = f"{ply_path.stem}{args.suffix}{ply_path.suffix}"
            out_path = output_dir / out_name

        n_before, n_after = process_file(
            ply_path=ply_path,
            out_path=out_path,
            method=args.method,
            voxel_size=args.voxel_size,
            ratio=args.ratio,
            max_points=args.max_points,
            seed=args.seed,
        )
        total_before += n_before
        total_after += n_after
        keep = 100.0 * n_after / max(n_before, 1)
        print(f"{ply_path.name}: {n_before} -> {n_after} ({keep:.2f}%) => {out_path}")

    keep_all = 100.0 * total_after / max(total_before, 1)
    print(f"Total: {total_before} -> {total_after} ({keep_all:.2f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
