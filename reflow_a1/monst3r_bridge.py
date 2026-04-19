"""Small import helpers for calling the local MonST3R/Dust3R modules."""

from __future__ import annotations

import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_monst3r_imports() -> Path:
    root = repo_root()
    extra_paths = [
        root,
        root / "third_party" / "sam2",
    ]
    for path in reversed(extra_paths):
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)
    return root


def default_weights_path() -> Path:
    return repo_root() / "checkpoints" / "MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth"

