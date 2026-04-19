"""Merge two PLY point clouds with optional simple scale/center alignment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from reflow_a1.export_ply import write_ply
else:
    from .export_ply import write_ply


_PLY_SCALAR_TYPES: dict[str, np.dtype] = {
    "char": np.dtype(np.int8),
    "uchar": np.dtype(np.uint8),
    "int8": np.dtype(np.int8),
    "uint8": np.dtype(np.uint8),
    "short": np.dtype(np.int16),
    "ushort": np.dtype(np.uint16),
    "int16": np.dtype(np.int16),
    "uint16": np.dtype(np.uint16),
    "int": np.dtype(np.int32),
    "uint": np.dtype(np.uint32),
    "int32": np.dtype(np.int32),
    "uint32": np.dtype(np.uint32),
    "float": np.dtype(np.float32),
    "float32": np.dtype(np.float32),
    "double": np.dtype(np.float64),
    "float64": np.dtype(np.float64),
}


def _parse_ply_header(path: Path) -> Tuple[str, int, list[Tuple[str, str]], int, int]:
    with path.open("rb") as f:
        first = f.readline().decode("latin1", errors="replace").strip()
        if first != "ply":
            raise ValueError(f"{path}: not a PLY file")

        fmt: Optional[str] = None
        vertex_count: Optional[int] = None
        in_vertex = False
        vertex_props: list[Tuple[str, str]] = []
        header_lines = 1

        for line in f:
            header_lines += 1
            text = line.decode("latin1", errors="replace").strip()
            if not text:
                continue
            tokens = text.split()
            if tokens[0] == "format" and len(tokens) >= 2:
                fmt = tokens[1]
            elif tokens[0] == "element" and len(tokens) >= 3:
                in_vertex = tokens[1] == "vertex"
                if in_vertex:
                    vertex_count = int(tokens[2])
                    vertex_props = []
            elif tokens[0] == "property" and in_vertex:
                if len(tokens) >= 3 and tokens[1] == "list":
                    raise ValueError(f"{path}: list property in vertex is not supported")
                if len(tokens) < 3:
                    raise ValueError(f"{path}: malformed property line: {text}")
                vertex_props.append((tokens[1], tokens[2]))
            elif tokens[0] == "end_header":
                break
        else:
            raise ValueError(f"{path}: missing end_header")

        data_start = f.tell()

    if fmt is None:
        raise ValueError(f"{path}: missing PLY format")
    if vertex_count is None:
        raise ValueError(f"{path}: missing vertex element")
    if not vertex_props:
        raise ValueError(f"{path}: missing vertex properties")
    return fmt, vertex_count, vertex_props, header_lines, data_start


def _dtype_from_ply_type(type_name: str, endian: str) -> np.dtype:
    base = _PLY_SCALAR_TYPES.get(type_name)
    if base is None:
        raise ValueError(f"Unsupported PLY scalar type: {type_name}")
    if base.itemsize == 1:
        return base
    return base.newbyteorder(endian)


def _to_uint8_colors(raw: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw)
    if raw.size == 0:
        return np.empty((0, 3), dtype=np.uint8)
    if np.issubdtype(raw.dtype, np.floating):
        finite = raw[np.isfinite(raw)]
        max_value = float(np.max(finite)) if finite.size else 0.0
        if max_value <= 1.001:
            raw = raw * 255.0
    return np.clip(raw, 0.0, 255.0).astype(np.uint8)


def read_ply_points_colors(path: Path) -> Tuple[np.ndarray, Optional[np.ndarray], str]:
    fmt, vertex_count, vertex_props, header_lines, data_start = _parse_ply_header(path)
    prop_names = [name for _, name in vertex_props]

    if not all(name in prop_names for name in ("x", "y", "z")):
        raise ValueError(f"{path}: vertex properties must include x/y/z")

    if fmt == "ascii":
        data = np.loadtxt(path, skiprows=header_lines, max_rows=vertex_count, ndmin=2)
        if data.shape[1] < len(prop_names):
            raise ValueError(f"{path}: malformed ASCII vertex rows")

        ix, iy, iz = prop_names.index("x"), prop_names.index("y"), prop_names.index("z")
        points = data[:, [ix, iy, iz]].astype(np.float32)

        colors = None
        if all(name in prop_names for name in ("red", "green", "blue")):
            ir, ig, ib = prop_names.index("red"), prop_names.index("green"), prop_names.index("blue")
            colors = _to_uint8_colors(data[:, [ir, ig, ib]])
        return points, colors, fmt

    if fmt == "binary_little_endian":
        dtype_fields = [(name, _dtype_from_ply_type(type_name, "<")) for type_name, name in vertex_props]
        vertex_dtype = np.dtype(dtype_fields)

        with path.open("rb") as f:
            f.seek(data_start)
            data = np.fromfile(f, dtype=vertex_dtype, count=vertex_count)
        if len(data) != vertex_count:
            raise ValueError(f"{path}: expected {vertex_count} vertices, got {len(data)}")

        points = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float32)
        colors = None
        if all(name in prop_names for name in ("red", "green", "blue")):
            colors = _to_uint8_colors(np.stack([data["red"], data["green"], data["blue"]], axis=1))
        return points, colors, fmt

    raise ValueError(f"{path}: unsupported PLY format '{fmt}' (supported: ascii, binary_little_endian)")


def _bbox_center(points: np.ndarray) -> np.ndarray:
    return 0.5 * (points.min(axis=0) + points.max(axis=0))


def _bbox_diag(points: np.ndarray) -> float:
    return float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))


def transform_points(
    points_to_transform: np.ndarray,
    reference_points: np.ndarray,
    manual_scale: float,
    manual_translate: np.ndarray,
    auto_scale_bbox: bool,
    align_centers: bool,
) -> Tuple[np.ndarray, float, np.ndarray]:
    if manual_scale == 0.0:
        raise ValueError("manual scale cannot be 0")

    effective_scale = float(manual_scale)
    if auto_scale_bbox:
        ref_diag = _bbox_diag(reference_points)
        src_diag = _bbox_diag(points_to_transform)
        if src_diag < 1e-12:
            raise ValueError("cannot auto-scale: source cloud has near-zero bbox diagonal")
        effective_scale *= ref_diag / src_diag

    transformed = points_to_transform * effective_scale
    auto_shift = np.zeros(3, dtype=np.float32)
    if align_centers:
        auto_shift = (_bbox_center(reference_points) - _bbox_center(transformed)).astype(np.float32)
        transformed = transformed + auto_shift

    transformed = transformed + manual_translate.reshape(1, 3)
    return transformed.astype(np.float32), effective_scale, auto_shift


def ensure_colors(colors: Optional[np.ndarray], count: int, fallback_rgb: np.ndarray) -> np.ndarray:
    if colors is None:
        rgb = np.asarray(fallback_rgb, dtype=np.uint8).reshape(1, 3)
        return np.repeat(rgb, count, axis=0)
    colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    if len(colors) != count:
        raise ValueError(f"Color count mismatch: expected {count}, got {len(colors)}")
    return colors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge two PLY point clouds")
    parser.add_argument("ply_a", type=str, help="First PLY file")
    parser.add_argument("ply_b", type=str, help="Second PLY file")
    parser.add_argument("--output", type=str, required=True, help="Output merged PLY path")
    parser.add_argument(
        "--transform_cloud",
        type=str,
        choices=["none", "a", "b"],
        default="none",
        help="Which cloud to transform before merging",
    )
    parser.add_argument("--scale", type=float, default=1.0, help="Manual scale for transformed cloud")
    parser.add_argument(
        "--translate",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("TX", "TY", "TZ"),
        help="Manual translation for transformed cloud",
    )
    parser.add_argument(
        "--auto_scale_bbox",
        action="store_true",
        help="Multiply scale by bbox diagonal ratio (reference/transform cloud)",
    )
    parser.add_argument(
        "--align_centers",
        action="store_true",
        help="Align bbox centers of the reference and transformed cloud",
    )
    parser.add_argument(
        "--fallback_color",
        type=int,
        nargs=3,
        default=(200, 200, 200),
        metavar=("R", "G", "B"),
        help="Color for points when a cloud has no RGB",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    ply_a = Path(args.ply_a).expanduser().resolve()
    ply_b = Path(args.ply_b).expanduser().resolve()
    out_path = Path(args.output).expanduser().resolve()
    fallback_color = np.asarray(args.fallback_color, dtype=np.uint8).reshape(3)
    manual_translate = np.asarray(args.translate, dtype=np.float32).reshape(3)

    points_a, colors_a, fmt_a = read_ply_points_colors(ply_a)
    points_b, colors_b, fmt_b = read_ply_points_colors(ply_b)

    if args.transform_cloud == "a":
        points_a, eff_scale, auto_shift = transform_points(
            points_to_transform=points_a,
            reference_points=points_b,
            manual_scale=args.scale,
            manual_translate=manual_translate,
            auto_scale_bbox=args.auto_scale_bbox,
            align_centers=args.align_centers,
        )
    elif args.transform_cloud == "b":
        points_b, eff_scale, auto_shift = transform_points(
            points_to_transform=points_b,
            reference_points=points_a,
            manual_scale=args.scale,
            manual_translate=manual_translate,
            auto_scale_bbox=args.auto_scale_bbox,
            align_centers=args.align_centers,
        )
    else:
        eff_scale = 1.0
        auto_shift = np.zeros(3, dtype=np.float32)

    colors_a_u8 = ensure_colors(colors_a, len(points_a), fallback_color)
    colors_b_u8 = ensure_colors(colors_b, len(points_b), fallback_color)

    merged_points = np.concatenate([points_a, points_b], axis=0)
    merged_colors = np.concatenate([colors_a_u8, colors_b_u8], axis=0)
    write_ply(out_path, merged_points, merged_colors)

    print(f"A: {ply_a} ({fmt_a}), points={len(points_a)}")
    print(f"B: {ply_b} ({fmt_b}), points={len(points_b)}")
    if args.transform_cloud != "none":
        print(
            f"Transform cloud={args.transform_cloud}, effective_scale={eff_scale:.6g}, "
            f"auto_shift={auto_shift.tolist()}, manual_translate={manual_translate.tolist()}"
        )
    print(f"Merged points: {len(merged_points)}")
    print(f"Output: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
