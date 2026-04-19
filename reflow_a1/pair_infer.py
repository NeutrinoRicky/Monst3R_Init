"""Pairwise MonST3R inference with an on-disk cache.

This module is the engineering bridge between ReFlow A.1's custom hierarchical
pair graph and MonST3R's existing pairwise pointmap regression. It stores each
directed pair as `pair_<id_a>__<id_b>.npz` so coarse and fine phases can share
expensive predictions.
"""

from __future__ import annotations

import copy
import ctypes
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from PIL.ImageOps import exif_transpose

from .dataset_scene import SceneDatadirDataset
from .monst3r_bridge import default_weights_path, ensure_monst3r_imports

Pair = Tuple[str, str]


def _ensure_ijit_shim(shim_path: Path) -> None:
    if shim_path.exists():
        return
    source_path = shim_path.with_name("ijit_stub.c")
    if not source_path.exists():
        raise FileNotFoundError(f"Missing iJIT shim source: {source_path}")
    cmd = [
        "gcc",
        "-shared",
        "-fPIC",
        str(source_path),
        "-o",
        str(shim_path),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0 or not shim_path.exists():
        detail = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(
            "Failed to build iJIT shim with gcc. "
            f"Command: {' '.join(cmd)}\n{detail}"
        )


def _import_torch():
    try:
        import torch  # type: ignore

        return torch
    except ImportError as exc:
        msg = str(exc)
        if "iJIT_" not in msg:
            raise
        # Fallback for environments missing dynamic ITT runtime. If a local
        # shim exists, preload it globally and retry torch import once.
        shim_path = Path(__file__).resolve().with_name("libittnotify.so")
        try:
            _ensure_ijit_shim(shim_path)
        except Exception as build_exc:
            raise ImportError(
                f"{msg}\n\nTorch import failed due to missing iJIT runtime symbols. "
                f"Automatic shim setup failed: {build_exc}"
            ) from exc
        ctypes.CDLL(str(shim_path), mode=ctypes.RTLD_GLOBAL)
        import sys

        for key in list(sys.modules.keys()):
            if key == "torch" or key.startswith("torch."):
                sys.modules.pop(key, None)
        import torch  # type: ignore

        return torch


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    writes: int = 0
    invalid: int = 0

    def as_dict(self) -> Dict[str, int]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
            "invalid": self.invalid,
        }


@dataclass
class PairBatch:
    output: Dict[str, Any]
    views_by_id: Dict[str, Dict[str, Any]]
    pair_results: List[Dict[str, Any]]
    stats: CacheStats = field(default_factory=CacheStats)


def _safe_id(frame_id: str) -> str:
    return frame_id.replace("/", "_").replace("\\", "_")


def _to_numpy(value: Any) -> np.ndarray:
    torch = _import_torch()

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _clone_view(view: Mapping[str, Any]) -> Dict[str, Any]:
    torch = _import_torch()

    cloned: Dict[str, Any] = {}
    for key, value in view.items():
        if isinstance(value, torch.Tensor):
            cloned[key] = value.clone()
        elif isinstance(value, np.ndarray):
            cloned[key] = value.copy()
        else:
            cloned[key] = copy.deepcopy(value)
    return cloned


def _resize_crop_plan(width: int, height: int, size: int, square_ok: bool = False, crop: bool = True) -> Dict[str, Any]:
    if size == 224:
        long_edge = round(size * max(width / height, height / width))
    else:
        long_edge = size
    scale = long_edge / max(width, height)
    new_w = int(round(width * scale))
    new_h = int(round(height * scale))

    if size == 224:
        cx, cy = new_w // 2, new_h // 2
        half = min(cx, cy)
        box = (cx - half, cy - half, cx + half, cy + half)
        final_size = (box[2] - box[0], box[3] - box[1])
    else:
        cx, cy = new_w // 2, new_h // 2
        halfw = ((2 * cx) // 16) * 8
        halfh = ((2 * cy) // 16) * 8
        if not square_ok and new_w == new_h:
            halfh = int(round(3 * halfw / 4))
        if crop:
            box = (cx - halfw, cy - halfh, cx + halfw, cy + halfh)
            final_size = (box[2] - box[0], box[3] - box[1])
        else:
            box = (0, 0, new_w, new_h)
            final_size = (2 * halfw, 2 * halfh)

    return {
        "scale_x": new_w / width,
        "scale_y": new_h / height,
        "resized_size": (new_w, new_h),
        "crop_box": tuple(int(v) for v in box),
        "final_size": tuple(int(v) for v in final_size),
        "crop": crop,
    }


def _apply_intrinsics_plan(K: np.ndarray, plan: Mapping[str, Any]) -> np.ndarray:
    out = K.astype(np.float32).copy()
    out[0, :] *= float(plan["scale_x"])
    out[1, :] *= float(plan["scale_y"])
    left, top, _, _ = plan["crop_box"]
    out[0, 2] -= left
    out[1, 2] -= top
    return out


def preprocess_frame_for_monst3r(
    frame: Mapping[str, Any],
    local_idx: int,
    image_size: int,
    crop: bool = True,
    square_ok: bool = False,
) -> Dict[str, Any]:
    """Match MonST3R image preprocessing while keeping camera K in sync."""

    torch = _import_torch()

    ensure_monst3r_imports()
    from dust3r.utils.image import ImgNorm, ToTensor

    image = exif_transpose(Image.open(frame["rgb_path"])).convert("RGB")
    width, height = image.size
    plan = _resize_crop_plan(width, height, image_size, square_ok=square_ok, crop=crop)
    resized_size = plan["resized_size"]
    crop_box = plan["crop_box"]

    if max(width, height) > image_size:
        image_interp = Image.Resampling.LANCZOS
    else:
        image_interp = Image.Resampling.BICUBIC
    image = image.resize(resized_size, image_interp)
    if crop:
        image = image.crop(crop_box)
    else:
        image = image.resize(plan["final_size"], Image.Resampling.LANCZOS)

    dynamic_mask = Image.fromarray(frame["dynamic_mask"].astype(np.uint8) * 255, mode="L")
    dynamic_mask = dynamic_mask.resize(resized_size, Image.Resampling.NEAREST)
    if crop:
        dynamic_mask = dynamic_mask.crop(crop_box)
    else:
        dynamic_mask = dynamic_mask.resize(plan["final_size"], Image.Resampling.NEAREST)

    K = _apply_intrinsics_plan(frame["camera_dict"]["K"], plan)
    true_shape = np.int32([image.size[::-1]])
    img_tensor = ImgNorm(image)[None]
    raw_mask = ToTensor(image)[None].sum(1) > 0.01
    dyn_tensor = ToTensor(dynamic_mask)[None].sum(1) > 0.5

    return {
        "img": img_tensor,
        "true_shape": true_shape,
        "idx": int(local_idx),
        "instance": str(frame["rgb_path"]),
        "mask": raw_mask,
        "dynamic_mask": dyn_tensor,
        "camera_pose": torch.from_numpy(frame["camera_dict"]["T_wc"][None].astype(np.float32)),
        "camera_intrinsics": torch.from_numpy(K[None].astype(np.float32)),
        "orig_shape": np.int32([[height, width]]),
    }


def _collate_dicts(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    torch = _import_torch()

    ensure_monst3r_imports()
    from dust3r.utils.device import collate_with_cat

    tensor_shapes = []
    for item in items:
        for value in item.values():
            if isinstance(value, (torch.Tensor, np.ndarray)):
                tensor_shapes.append(tuple(value.shape))
                break
    lists = len(set(tensor_shapes)) > 1
    return collate_with_cat(items, lists=lists)


class PairwiseInferencer:
    def __init__(
        self,
        scene_root: str | Path,
        image_size: int = 512,
        weights: Optional[str | Path] = None,
        model_name: str = "Junyi42/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt",
        device: Optional[str] = None,
        batch_size: int = 8,
        crop: bool = True,
        square_ok: bool = False,
        verbose: bool = True,
    ) -> None:
        torch = _import_torch()

        self.scene_root = Path(scene_root).expanduser().resolve()
        self.image_size = image_size
        self.weights = Path(weights).expanduser().resolve() if weights else None
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.crop = crop
        self.square_ok = square_ok
        self.verbose = verbose
        self.cache_dir = self.scene_root / "monst3r_reflow_a1_cache" / "pairs"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.stats = CacheStats()
        self._model = None

    def cache_stats(self) -> Dict[str, int]:
        return self.stats.as_dict()

    def build_views(
        self,
        dataset: SceneDatadirDataset,
        frame_ids: Sequence[str],
    ) -> Dict[str, Dict[str, Any]]:
        views: Dict[str, Dict[str, Any]] = {}
        for idx, frame_id in enumerate(frame_ids):
            frame = dataset.get_frame_by_id(frame_id)
            views[frame_id] = preprocess_frame_for_monst3r(
                frame,
                idx,
                self.image_size,
                crop=self.crop,
                square_ok=self.square_ok,
            )
        return views

    def infer_pair(
        self,
        dataset: SceneDatadirDataset,
        pair: Pair,
        force_recompute: bool = False,
    ) -> Dict[str, Any]:
        batch = self.infer_pairs(dataset, list(pair), [pair], force_recompute=force_recompute)
        return batch.pair_results[0]

    def infer_pairs(
        self,
        dataset: SceneDatadirDataset,
        frame_ids: Sequence[str],
        pairs: Sequence[Pair],
        force_recompute: bool = False,
    ) -> PairBatch:
        frame_ids = list(frame_ids)
        pairs = list(pairs)
        views_by_id = self.build_views(dataset, frame_ids)
        local_stats = CacheStats()

        pair_results: Dict[Pair, Dict[str, Any]] = {}
        missing: List[Pair] = []
        for pair in pairs:
            cached = None if force_recompute else self._load_pair_cache(pair)
            if cached is None:
                missing.append(pair)
                self.stats.misses += 1
                local_stats.misses += 1
            else:
                pair_results[pair] = cached
                self.stats.hits += 1
                local_stats.hits += 1

        if missing:
            self._infer_and_cache_missing(views_by_id, missing, pair_results)
            local_stats.writes += len(missing)

        ordered_results = [pair_results[pair] for pair in pairs]
        output = self._assemble_dust3r_output(views_by_id, ordered_results)
        return PairBatch(output=output, views_by_id=views_by_id, pair_results=ordered_results, stats=local_stats)

    def _model_for_inference(self):
        if self._model is not None:
            return self._model

        ensure_monst3r_imports()
        from dust3r.model import AsymmetricCroCo3DStereo

        weights = self.weights
        if weights is None:
            candidate = default_weights_path()
            weights = candidate if candidate.exists() else None
        if weights is not None:
            if not weights.exists():
                raise FileNotFoundError(
                    "The --weights path does not exist:\n"
                    f"  {weights}\n"
                    "Please provide a valid local checkpoint file."
                )
            if not weights.is_file():
                raise FileNotFoundError(
                    "The --weights path is not a file:\n"
                    f"  {weights}\n"
                    "Please provide a valid .pth checkpoint file."
                )
        source = str(weights) if weights is not None else self.model_name
        if self.verbose:
            print(f"[Phase 3] Loading MonST3R model from {source}")
        try:
            model = AsymmetricCroCo3DStereo.from_pretrained(source).to(self.device)
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            if weights is not None:
                hint = (
                    "This path was treated as a local checkpoint. "
                    "Please verify the file is readable and compatible."
                )
            else:
                hint = (
                    "No local checkpoint was selected; source is treated as a Hugging Face model id. "
                    "If this machine is offline, pass --weights with a local .pth file."
                )
            raise RuntimeError(
                "Failed to load geometry model checkpoint. "
                f"source={source}. "
                f"underlying_error={detail}. "
                f"{hint}"
            ) from exc
        model.eval()
        self._model = model
        return model

    def _infer_and_cache_missing(
        self,
        views_by_id: Mapping[str, Dict[str, Any]],
        missing: Sequence[Pair],
        pair_results: Dict[Pair, Dict[str, Any]],
    ) -> None:
        ensure_monst3r_imports()
        from dust3r.inference import inference

        model = self._model_for_inference()
        pair_views = [(_clone_view(views_by_id[a]), _clone_view(views_by_id[b])) for a, b in missing]
        if self.verbose:
            print(f"[Phase 3] Pair inference: computing {len(missing)} cache miss(es)")
        output = inference(pair_views, model, self.device, batch_size=self.batch_size, verbose=self.verbose)

        for n, pair in enumerate(missing):
            result = self._result_from_output(output, n, pair)
            self._save_pair_cache(result)
            pair_results[pair] = result
            self.stats.writes += 1

    def _result_from_output(self, output: Mapping[str, Any], n: int, pair: Pair) -> Dict[str, Any]:
        view1, view2, pred1, pred2 = output["view1"], output["view2"], output["pred1"], output["pred2"]
        return {
            "pair": tuple(pair),
            "Xa_in_a": _to_numpy(pred1["pts3d"][n]).astype(np.float32),
            "Xb_in_a": _to_numpy(pred2["pts3d_in_other_view"][n]).astype(np.float32),
            "Ca_in_a": _to_numpy(pred1["conf"][n]).astype(np.float32),
            "Cb_in_a": _to_numpy(pred2["conf"][n]).astype(np.float32),
            "meta": {
                "image_size": int(self.image_size),
                "crop": bool(self.crop),
                "true_shape_a": _to_numpy(view1["true_shape"][n]).astype(np.int32).tolist(),
                "true_shape_b": _to_numpy(view2["true_shape"][n]).astype(np.int32).tolist(),
                "orig_shape_a": _to_numpy(view1["orig_shape"][n]).astype(np.int32).tolist()
                if "orig_shape" in view1
                else None,
                "orig_shape_b": _to_numpy(view2["orig_shape"][n]).astype(np.int32).tolist()
                if "orig_shape" in view2
                else None,
            },
        }

    def _pair_cache_path(self, pair: Pair) -> Path:
        a, b = pair
        return self.cache_dir / f"pair_{_safe_id(a)}__{_safe_id(b)}.npz"

    def _load_pair_cache(self, pair: Pair) -> Optional[Dict[str, Any]]:
        path = self._pair_cache_path(pair)
        if not path.exists():
            return None
        try:
            data = np.load(path, allow_pickle=True)
            meta = data["meta"].item()
            if int(meta.get("image_size", -1)) != int(self.image_size) or bool(meta.get("crop", True)) != bool(self.crop):
                self.stats.invalid += 1
                return None
            return {
                "pair": tuple(data["pair"].tolist()),
                "Xa_in_a": data["Xa_in_a"].astype(np.float32),
                "Xb_in_a": data["Xb_in_a"].astype(np.float32),
                "Ca_in_a": data["Ca_in_a"].astype(np.float32),
                "Cb_in_a": data["Cb_in_a"].astype(np.float32),
                "meta": meta,
            }
        except Exception:
            self.stats.invalid += 1
            return None

    def _save_pair_cache(self, result: Mapping[str, Any]) -> None:
        path = self._pair_cache_path(tuple(result["pair"]))
        np.savez_compressed(
            path,
            pair=np.asarray(result["pair"]),
            Xa_in_a=result["Xa_in_a"],
            Xb_in_a=result["Xb_in_a"],
            Ca_in_a=result["Ca_in_a"],
            Cb_in_a=result["Cb_in_a"],
            meta=np.asarray(result["meta"], dtype=object),
        )

    def _assemble_dust3r_output(
        self,
        views_by_id: Mapping[str, Dict[str, Any]],
        ordered_results: Sequence[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        torch = _import_torch()

        view1_items: List[Dict[str, Any]] = []
        view2_items: List[Dict[str, Any]] = []
        pred1_items: List[Dict[str, Any]] = []
        pred2_items: List[Dict[str, Any]] = []

        for result in ordered_results:
            a, b = result["pair"]
            view1_items.append(_clone_view(views_by_id[a]))
            view2_items.append(_clone_view(views_by_id[b]))
            pred1_items.append(
                {
                    "pts3d": torch.from_numpy(np.asarray(result["Xa_in_a"], dtype=np.float32))[None],
                    "conf": torch.from_numpy(np.asarray(result["Ca_in_a"], dtype=np.float32))[None],
                }
            )
            pred2_items.append(
                {
                    "pts3d_in_other_view": torch.from_numpy(np.asarray(result["Xb_in_a"], dtype=np.float32))[None],
                    "conf": torch.from_numpy(np.asarray(result["Cb_in_a"], dtype=np.float32))[None],
                }
            )

        return {
            "view1": _collate_dicts(view1_items),
            "view2": _collate_dicts(view2_items),
            "pred1": _collate_dicts(pred1_items),
            "pred2": _collate_dicts(pred2_items),
        }
