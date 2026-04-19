"""Minimal ASCII PLY writer for ReFlow A.1 point clouds."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def _as_uint8_colors(colors: Optional[np.ndarray], n: int) -> np.ndarray:
    if colors is None:
        return np.full((n, 3), 200, dtype=np.uint8)
    colors = np.asarray(colors)
    if colors.size == 0:
        return np.empty((0, 3), dtype=np.uint8)
    if colors.dtype == np.uint8:
        return colors.reshape(-1, 3)
    colors = np.clip(colors.reshape(-1, 3), 0.0, 1.0)
    return (colors * 255.0 + 0.5).astype(np.uint8)


def write_ply(path: str | Path, points: np.ndarray, colors: Optional[np.ndarray] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    colors_u8 = _as_uint8_colors(colors, len(points))
    if len(colors_u8) != len(points):
        raise ValueError(f"Color count {len(colors_u8)} does not match point count {len(points)}")

    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points, colors_u8):
            f.write(f"{p[0]:.7g} {p[1]:.7g} {p[2]:.7g} {int(c[0])} {int(c[1])} {int(c[2])}\n")

