"""Scene-datadir reader for the ReFlow A.1 initialization pipeline.

The dataset keeps the original scene layout intact and exposes only the
`dataset.json` split requested by the A.1 construction. Segmentation is treated
as the direct dynamic-region source: by default non-zero labels are dynamic.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from PIL import Image


class SceneDataError(RuntimeError):
    """Raised when a scene directory cannot be interpreted unambiguously."""


IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")
ARRAY_SUFFIXES = (".npy", ".npz")

_COLMAP_CAMERA_MODEL_NUM_PARAMS = {
    0: 3,
    1: 4,
    2: 4,
    3: 5,
    4: 8,
    5: 8,
    6: 12,
    7: 5,
    8: 4,
    9: 5,
    10: 12,
}

_COLMAP_CAMERA_MODEL_NAMES = {
    0: "SIMPLE_PINHOLE",
    1: "PINHOLE",
    2: "SIMPLE_RADIAL",
    3: "RADIAL",
    4: "OPENCV",
    5: "OPENCV_FISHEYE",
    6: "FULL_OPENCV",
    7: "FOV",
    8: "SIMPLE_RADIAL_FISHEYE",
    9: "RADIAL_FISHEYE",
    10: "THIN_PRISM_FISHEYE",
}


@dataclass(frozen=True)
class FrameRecord:
    frame_id: str
    rgb_path: Path
    depth_path: Optional[Path]
    seg_path: Optional[Path]
    camera_path: Optional[Path]
    track_path: Optional[Path]
    camera_dict: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class ColmapCamera:
    id: int
    model: str
    width: int
    height: int
    params: np.ndarray


@dataclass(frozen=True)
class ColmapImage:
    id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str


def _read_next_bytes(fid, num_bytes: int, format_char_sequence: str, endian_character: str = "<") -> tuple[Any, ...]:
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def _read_colmap_cameras_text(path: Path) -> Dict[int, ColmapCamera]:
    cameras: Dict[int, ColmapCamera] = {}
    with path.open("r", encoding="utf-8") as fid:
        for line in fid:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            elems = line.split()
            camera_id = int(elems[0])
            cameras[camera_id] = ColmapCamera(
                id=camera_id,
                model=elems[1],
                width=int(elems[2]),
                height=int(elems[3]),
                params=np.array(tuple(map(float, elems[4:])), dtype=np.float32),
            )
    return cameras


def _read_colmap_images_text(path: Path) -> Dict[int, ColmapImage]:
    images: Dict[int, ColmapImage] = {}
    with path.open("r", encoding="utf-8") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            elems = line.split()
            image_id = int(elems[0])
            images[image_id] = ColmapImage(
                id=image_id,
                qvec=np.array(tuple(map(float, elems[1:5])), dtype=np.float32),
                tvec=np.array(tuple(map(float, elems[5:8])), dtype=np.float32),
                camera_id=int(elems[8]),
                name=elems[9],
            )
            fid.readline()
    return images


def _read_colmap_cameras_binary(path: Path) -> Dict[int, ColmapCamera]:
    cameras: Dict[int, ColmapCamera] = {}
    with path.open("rb") as fid:
        num_cameras = _read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = _read_next_bytes(fid, 24, "iiQQ")
            camera_id = int(camera_properties[0])
            model_id = int(camera_properties[1])
            num_params = _COLMAP_CAMERA_MODEL_NUM_PARAMS[model_id]
            params = _read_next_bytes(fid, 8 * num_params, "d" * num_params)
            cameras[camera_id] = ColmapCamera(
                id=camera_id,
                model=_COLMAP_CAMERA_MODEL_NAMES[model_id],
                width=int(camera_properties[2]),
                height=int(camera_properties[3]),
                params=np.asarray(params, dtype=np.float32),
            )
    return cameras


def _read_colmap_images_binary(path: Path) -> Dict[int, ColmapImage]:
    images: Dict[int, ColmapImage] = {}
    with path.open("rb") as fid:
        num_images = _read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_images):
            props = _read_next_bytes(fid, 64, "idddddddi")
            image_id = int(props[0])
            qvec = np.asarray(props[1:5], dtype=np.float32)
            tvec = np.asarray(props[5:8], dtype=np.float32)
            camera_id = int(props[8])
            chars: List[str] = []
            current_char = _read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":
                chars.append(current_char.decode("utf-8"))
                current_char = _read_next_bytes(fid, 1, "c")[0]
            num_points2d = _read_next_bytes(fid, 8, "Q")[0]
            if num_points2d > 0:
                fid.seek(24 * num_points2d, 1)
            images[image_id] = ColmapImage(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name="".join(chars),
            )
    return images


def _as_float_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape != shape:
        raise SceneDataError(f"Expected {name} to have shape {shape}, got {arr.shape}")
    return arr


def _first_present(mapping: Dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _first_present_item(mapping: Dict[str, Any], keys: Iterable[str]) -> tuple[Optional[str], Any]:
    for key in keys:
        if key in mapping:
            return key, mapping[key]
    return None, None


def _homogeneous_from_rt(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = rotation.astype(np.float32)
    mat[:3, 3] = translation.astype(np.float32)
    return mat


def _qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    qvec = np.asarray(qvec, dtype=np.float32)
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ],
        dtype=np.float32,
    )


def _colmap_camera_to_K(camera: ColmapCamera) -> np.ndarray:
    params = np.asarray(camera.params, dtype=np.float32)
    model = camera.model.upper()
    if model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE", "FOV"}:
        fx = fy = float(params[0])
        cx = float(params[1])
        cy = float(params[2])
    elif model in {"PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV", "THIN_PRISM_FISHEYE"}:
        fx = float(params[0])
        fy = float(params[1])
        cx = float(params[2])
        cy = float(params[3])
    else:
        raise SceneDataError(f"Unsupported COLMAP camera model: {camera.model}")
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


def _colmap_image_to_camera_dict(image: ColmapImage, camera: ColmapCamera) -> Dict[str, Any]:
    R_cw = _qvec2rotmat(image.qvec)
    t_cw = np.asarray(image.tvec, dtype=np.float32)
    T_cw = _homogeneous_from_rt(R_cw, t_cw)
    T_wc = np.linalg.inv(T_cw).astype(np.float32)
    K = _colmap_camera_to_K(camera)
    return {
        "K": K,
        "T_wc": T_wc,
        "T_cw": T_cw.astype(np.float32),
        "width": int(camera.width),
        "height": int(camera.height),
        "fx": float(K[0, 0]),
        "fy": float(K[1, 1]),
        "cx": float(K[0, 2]),
        "cy": float(K[1, 2]),
        "raw": {
            "image_id": int(image.id),
            "camera_id": int(camera.id),
            "image_name": image.name,
            "qvec": np.asarray(image.qvec, dtype=np.float32),
            "tvec": np.asarray(image.tvec, dtype=np.float32),
            "camera_model": camera.model,
            "camera_params": np.asarray(camera.params, dtype=np.float32),
        },
    }


def parse_camera_json(camera_path: str | Path) -> Dict[str, Any]:
    """Parse common camera-json variants into a single camera convention.

    The returned pose convention is camera-to-world (`T_wc`) plus its inverse
    (`T_cw`). Nerfies/HyperNeRF-style camera JSONs use `orientation` as COLMAP's
    world-to-camera rotation (`R_cw`) and `position` as the camera center (`C`).
    """

    path = Path(camera_path)
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    k_value = _first_present(
        raw,
        (
            "K",
            "intrinsics",
            "camera_intrinsics",
            "intrinsic",
            "camera_matrix",
        ),
    )
    width = _first_present(raw, ("width", "w", "image_width"))
    height = _first_present(raw, ("height", "h", "image_height"))

    if k_value is not None:
        K = _as_float_array(k_value, (3, 3), "camera intrinsics")
        fx = float(K[0, 0])
        fy = float(K[1, 1])
        cx = float(K[0, 2])
        cy = float(K[1, 2])
    else:
        focal = _first_present(raw, ("focal_length", "focal", "fl_x", "fx"))
        fx_value = _first_present(raw, ("fx", "fl_x"))
        fy_value = _first_present(raw, ("fy", "fl_y"))
        principal = _first_present(raw, ("principal_point", "pp"))
        cx_value = _first_present(raw, ("cx", "principal_x"))
        cy_value = _first_present(raw, ("cy", "principal_y"))
        skew = float(raw.get("skew", 0.0))

        if fx_value is None:
            fx_value = focal
        if fy_value is None:
            aspect = float(raw.get("pixel_aspect_ratio", 1.0))
            fy_value = None if focal is None else float(focal) * aspect
        if principal is not None:
            cx_value, cy_value = principal[:2]

        if fx_value is None or fy_value is None or cx_value is None or cy_value is None:
            raise SceneDataError(
                f"Camera {path} is missing intrinsics. Need a 3x3 K or fx/fy/cx/cy "
                "style fields."
            )

        fx = float(fx_value)
        fy = float(fy_value)
        cx = float(cx_value)
        cy = float(cy_value)
        K = np.array([[fx, skew, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)

    image_size = _first_present(raw, ("image_size", "size", "resolution"))
    if image_size is not None:
        width = int(image_size[0])
        height = int(image_size[1])
    if width is None or height is None:
        width = int(round(2 * cx))
        height = int(round(2 * cy))

    twc_value = _first_present(
        raw,
        (
            "T_wc",
            "c2w",
            "cam2world",
            "camera_to_world",
            "transform_matrix",
        ),
    )
    tcw_value = _first_present(raw, ("T_cw", "w2c", "world_to_camera", "extrinsic", "extrinsics"))

    if twc_value is not None:
        T_wc = _as_float_array(twc_value, (4, 4), "T_wc")
    elif tcw_value is not None:
        T_cw = _as_float_array(tcw_value, (4, 4), "T_cw")
        T_wc = np.linalg.inv(T_cw).astype(np.float32)
    else:
        rotation_key, rotation = _first_present_item(raw, ("orientation", "rotation", "R", "camera_rotation"))
        position = _first_present(raw, ("position", "translation", "t", "camera_position"))
        if rotation is None or position is None:
            raise SceneDataError(
                f"Camera {path} is missing pose. Need T_wc/T_cw or orientation + position."
            )
        R = _as_float_array(rotation, (3, 3), "orientation")
        t = _as_float_array(position, (3,), "position")
        if rotation_key == "orientation":
            T_wc = _homogeneous_from_rt(R.T, t)
        else:
            T_wc = _homogeneous_from_rt(R, t)

    T_cw = np.linalg.inv(T_wc).astype(np.float32)

    return {
        "K": K.astype(np.float32),
        "T_wc": T_wc.astype(np.float32),
        "T_cw": T_cw,
        "width": int(width),
        "height": int(height),
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "raw": raw,
    }


def dynamic_mask_from_segmentation(
    segmentation: np.ndarray,
    mode: str = "nonzero_is_dynamic",
    dynamic_label: Optional[int] = None,
) -> np.ndarray:
    seg = np.asarray(segmentation)
    if seg.ndim == 3 and seg.shape[-1] == 1:
        seg = seg[..., 0]
    elif seg.ndim == 3 and mode in ("nonzero_is_dynamic", "zero_is_dynamic"):
        mask = np.any(seg != 0, axis=-1)
        return mask if mode == "nonzero_is_dynamic" else ~mask
    if mode == "nonzero_is_dynamic":
        return seg != 0
    if mode == "zero_is_dynamic":
        return seg == 0
    if mode == "label_is_dynamic":
        if dynamic_label is None:
            raise SceneDataError("--dynamic_label is required when mode=label_is_dynamic")
        return seg == dynamic_label
    raise SceneDataError(f"Unknown dynamic label mode: {mode}")


class SceneDatadirDataset:
    """Reader for a single scene directory using dataset.json train/val ids."""

    def __init__(
        self,
        scene_root: str | Path,
        split: str = "train",
        dynamic_label_mode: str = "nonzero_is_dynamic",
        dynamic_label: Optional[int] = None,
    ) -> None:
        self.scene_root = Path(scene_root).expanduser().resolve()
        self.split = split
        self.dynamic_label_mode = dynamic_label_mode
        self.dynamic_label = dynamic_label

        if not self.scene_root.exists():
            raise SceneDataError(f"Scene root does not exist: {self.scene_root}")

        self.dataset_json_path = self.scene_root / "dataset.json"
        self.dataset_json = self._read_dataset_json()
        self.frame_ids = self._read_split_ids(split)
        self.frames = [self._build_record(frame_id) for frame_id in self.frame_ids]
        self._by_id = {record.frame_id: record for record in self.frames}

    def __len__(self) -> int:
        return len(self.frames)

    def get_train_frame_ids(self) -> List[str]:
        return list(self.frame_ids)

    def get_frame_by_id(self, frame_id: str) -> Dict[str, Any]:
        try:
            record = self._by_id[frame_id]
        except KeyError as exc:
            raise SceneDataError(f"Unknown frame id for split {self.split}: {frame_id}") from exc
        return self._load_record(record)

    def get_frame(self, idx: int) -> Dict[str, Any]:
        return self._load_record(self.frames[idx])

    def get_record_by_id(self, frame_id: str) -> FrameRecord:
        try:
            return self._by_id[frame_id]
        except KeyError as exc:
            raise SceneDataError(f"Unknown frame id for split {self.split}: {frame_id}") from exc

    def describe(self, sample_count: int = 3) -> Dict[str, Any]:
        head = self.frame_ids[:sample_count]
        tail = self.frame_ids[-sample_count:] if len(self.frame_ids) > sample_count else []
        samples = [self.get_frame_by_id(fid) for fid in head[:1]] if head else []
        first_shape = None
        if samples:
            sample = samples[0]
            first_shape = {
                "rgb": list(sample["rgb"].shape),
                "depth": None if sample["depth"] is None else list(sample["depth"].shape),
                "segmentation": list(sample["segmentation"].shape),
                "dynamic_mask": list(sample["dynamic_mask"].shape),
            }
        return {
            "scene_root": str(self.scene_root),
            "split": self.split,
            "num_frames": len(self),
            "frame_ids_head": head,
            "frame_ids_tail": tail,
            "first_shapes": first_shape,
        }

    def _read_dataset_json(self) -> Dict[str, Any]:
        if not self.dataset_json_path.exists():
            raise SceneDataError(f"Missing dataset.json: {self.dataset_json_path}")
        with self.dataset_json_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _read_split_ids(self, split: str) -> List[str]:
        key = f"{split}_ids"
        ids = self.dataset_json.get(key)
        if ids is None and split == "all":
            ids = self.dataset_json.get("ids")
        if ids is None:
            split_path = self.scene_root / "splits" / f"{split}.json"
            if split_path.exists():
                with split_path.open("r", encoding="utf-8") as f:
                    split_data = json.load(f)
                ids = split_data.get("ids", split_data.get(key, split_data))
        if ids is None:
            raise SceneDataError(f"Could not find split ids for split={split!r}")
        return [str(frame_id) for frame_id in ids]

    def _resolve_existing(self, relative_candidates: Iterable[str], required: bool = True) -> Optional[Path]:
        tried = []
        for rel in relative_candidates:
            tried.append(rel)
            path = self.scene_root / rel
            if path.exists():
                return path
        if required:
            raise SceneDataError(
                "Missing scene asset. Tried: "
                + ", ".join(str(self.scene_root / rel) for rel in tried)
            )
        return None

    def _existing_dirs(self, relative_candidates: Iterable[str]) -> List[str]:
        out = []
        seen = set()
        for rel in relative_candidates:
            if rel in seen:
                continue
            seen.add(rel)
            if (self.scene_root / rel).is_dir():
                out.append(rel)
        return out

    def _rgb_dirs(self) -> List[str]:
        split_specific = []
        rgb_root = self.scene_root / "rgb"
        preferred_split = f"2x_{self.split}_ids"
        if (rgb_root / preferred_split).is_dir():
            split_specific.append(f"rgb/{preferred_split}")
        if rgb_root.is_dir():
            for path in sorted(rgb_root.glob(f"*_{self.split}_ids")):
                rel = f"rgb/{path.name}"
                if rel not in split_specific:
                    split_specific.append(rel)

        # HyperNeRF's masks live at 2x resolution, so prefer rgb/2x and let
        # camera intrinsics be scaled to the loaded image size.
        generic = ["rgb/2x", "rgb/1x", "rgb/4x", "rgb/8x", "rgb/16x", "rgb"]
        return self._existing_dirs([*split_specific, *generic])

    def _candidate_paths(
        self,
        directories: Iterable[str],
        stems: Iterable[str],
        suffixes: Iterable[str],
    ) -> Iterable[str]:
        for directory in directories:
            for stem in stems:
                for suffix in suffixes:
                    yield f"{directory}/{stem}{suffix}"

    def _build_record(self, frame_id: str) -> FrameRecord:
        stem_candidates = [frame_id]
        if not frame_id.startswith("C_"):
            stem_candidates.append(f"C_{frame_id}")

        suffix = frame_id.split("_")[-1]
        rgb = self._resolve_existing(self._candidate_paths(self._rgb_dirs(), stem_candidates, IMAGE_SUFFIXES))
        depth_dirs = self._existing_dirs(
            (
                "depth/1x",
                f"depth/2x_{self.split}_ids",
                "depth/2x",
                "depth",
            )
        )
        depth = self._resolve_existing(
            self._candidate_paths(depth_dirs, stem_candidates, (*ARRAY_SUFFIXES, *IMAGE_SUFFIXES)),
            required=False,
        )
        seg_dirs = self._existing_dirs(
            (
                "mask",
                "masks",
                "segmentation/2x",
                "segmentation/1x",
                f"segmentation/2x_{self.split}_ids",
                "segmentation",
            )
        )
        seg = self._resolve_existing(self._candidate_paths(seg_dirs, stem_candidates, (*ARRAY_SUFFIXES, *IMAGE_SUFFIXES)))
        camera = self._resolve_existing(f"camera/{stem}.json" for stem in stem_candidates)
        track = self._resolve_existing(
            (
                f"tracks/1x/track_{frame_id}.npy",
                f"tracks/1x/track_{suffix}.npy",
                f"tracks/1x/track_C_{suffix}.npy",
            ),
            required=False,
        )
        return FrameRecord(frame_id, rgb, depth, seg, camera, track)

    def _load_array_or_image(self, path: Path, image_mode: Optional[str] = None) -> np.ndarray:
        suffix = path.suffix.lower()
        if suffix == ".npy":
            return np.load(path)
        if suffix == ".npz":
            data = np.load(path)
            preferred_keys = ("arr_0", "depth", "segmentation", "mask")
            for key in preferred_keys:
                if key in data:
                    return data[key]
            if len(data.files) == 1:
                return data[data.files[0]]
            raise SceneDataError(f"Cannot choose an array from multi-entry npz: {path}")
        if suffix in IMAGE_SUFFIXES:
            image = Image.open(path)
            if image_mode is not None:
                image = image.convert(image_mode)
            return np.asarray(image)
        raise SceneDataError(f"Unsupported scene asset format: {path}")

    def _load_depth(self, path: Path) -> np.ndarray:
        depth = self._load_array_or_image(path).astype(np.float32)
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        return depth

    def _load_segmentation(self, path: Path) -> np.ndarray:
        suffix = path.suffix.lower()
        if suffix in (".jpg", ".jpeg"):
            # HyperNeRF masks are stored as JPEGs, so tiny compression values
            # around binary edges should not become separate dynamic labels.
            return (self._load_array_or_image(path, image_mode="L") > 127).astype(np.uint8)

        segmentation = self._load_array_or_image(path)
        if suffix in IMAGE_SUFFIXES and segmentation.ndim == 3:
            if segmentation.shape[-1] == 4:
                segmentation = segmentation[..., :3]
            if segmentation.shape[-1] == 3 and np.all(segmentation[..., 0] == segmentation[..., 1]) and np.all(
                segmentation[..., 0] == segmentation[..., 2]
            ):
                segmentation = segmentation[..., 0]
            else:
                segmentation = self._load_array_or_image(path, image_mode="L")
        if segmentation.ndim == 3 and segmentation.shape[-1] == 1:
            segmentation = segmentation[..., 0]
        return segmentation

    def _scale_camera_to_image(self, camera: Dict[str, Any], width: int, height: int, frame_id: str) -> Dict[str, Any]:
        cam_w = int(camera["width"])
        cam_h = int(camera["height"])
        if (cam_w, cam_h) == (width, height):
            return camera
        if cam_w <= 0 or cam_h <= 0:
            raise SceneDataError(f"{frame_id}: invalid camera image size {(cam_w, cam_h)}")

        sx = float(width) / float(cam_w)
        sy = float(height) / float(cam_h)
        rel_scale_delta = abs(sx - sy) / max(abs(sx), abs(sy), 1e-8)
        if rel_scale_delta > 0.02:
            raise SceneDataError(
                f"{frame_id}: camera image size {(cam_w, cam_h)} does not match RGB size {(width, height)} "
                "and is not a near-uniform scale"
            )

        scaled = dict(camera)
        K = np.asarray(camera["K"], dtype=np.float32).copy()
        K[0, :] *= sx
        K[1, :] *= sy
        scaled["K"] = K
        scaled["width"] = int(width)
        scaled["height"] = int(height)
        scaled["fx"] = float(K[0, 0])
        scaled["fy"] = float(K[1, 1])
        scaled["cx"] = float(K[0, 2])
        scaled["cy"] = float(K[1, 2])
        scaled["source_width"] = cam_w
        scaled["source_height"] = cam_h
        scaled["image_scale_x"] = sx
        scaled["image_scale_y"] = sy
        return scaled

    def _load_record(self, record: FrameRecord) -> Dict[str, Any]:
        rgb = np.asarray(Image.open(record.rgb_path).convert("RGB"))
        depth = None if record.depth_path is None else self._load_depth(record.depth_path)
        if record.seg_path is None:
            segmentation = np.zeros(rgb.shape[:2], dtype=np.uint8)
        else:
            segmentation = self._load_segmentation(record.seg_path)
        if record.camera_dict is not None:
            camera = record.camera_dict
        elif record.camera_path is not None:
            camera = parse_camera_json(record.camera_path)
        else:
            raise SceneDataError(f"{record.frame_id}: record is missing both camera_path and camera_dict")
        camera = self._scale_camera_to_image(camera, rgb.shape[1], rgb.shape[0], record.frame_id)
        dynamic_mask = dynamic_mask_from_segmentation(
            segmentation,
            mode=self.dynamic_label_mode,
            dynamic_label=self.dynamic_label,
        )

        h, w = rgb.shape[:2]
        for name, arr in (("depth", depth), ("segmentation", segmentation), ("dynamic_mask", dynamic_mask)):
            if arr is None:
                continue
            if arr.shape[:2] != (h, w):
                raise SceneDataError(
                    f"{record.frame_id}: RGB shape {(h, w)} does not match {name} shape {arr.shape[:2]}"
                )
        track = np.load(record.track_path) if record.track_path is not None else None
        return {
            "frame_id": record.frame_id,
            "rgb_path": str(record.rgb_path),
            "depth_path": str(record.depth_path) if record.depth_path is not None else None,
            "seg_path": str(record.seg_path) if record.seg_path is not None else None,
            "camera_path": str(record.camera_path) if record.camera_path is not None else None,
            "track_path": str(record.track_path) if record.track_path is not None else None,
            "rgb": rgb,
            "depth": depth,
            "segmentation": segmentation,
            "dynamic_mask": dynamic_mask.astype(bool),
            "has_dynamic_mask": bool(record.seg_path is not None),
            "static_mask": ~dynamic_mask.astype(bool),
            "camera_dict": camera,
            "track": track,
        }


class ColmapSparseSceneDataset(SceneDatadirDataset):
    """In-memory scene reader for COLMAP sparse reconstructions plus image folders."""

    def __init__(
        self,
        scene_root: str | Path,
        split: str = "train",
        dynamic_label_mode: str = "nonzero_is_dynamic",
        dynamic_label: Optional[int] = None,
    ) -> None:
        self.scene_root = Path(scene_root).expanduser().resolve()
        self.split = split
        self.dynamic_label_mode = dynamic_label_mode
        self.dynamic_label = dynamic_label

        if not self.scene_root.exists():
            raise SceneDataError(f"Scene root does not exist: {self.scene_root}")

        self.dataset_json_path = self.scene_root / "dataset.json"
        self.dataset_json = {}
        self.colmap_root = self._resolve_colmap_root()
        self.colmap_cameras = self._read_colmap_cameras()
        self.colmap_images = self._read_colmap_images()
        self._images_by_frame_id = self._index_colmap_images()
        self.frame_ids = self._read_split_ids(split)
        self.frames = [self._build_record(frame_id) for frame_id in self.frame_ids]
        self._by_id = {record.frame_id: record for record in self.frames}

    def _resolve_colmap_root(self) -> Path:
        candidates = (
            self.scene_root / "sparse" / "0",
            self.scene_root / "colmap" / "sparse" / "0",
        )
        for candidate in candidates:
            if candidate.is_dir():
                return candidate
        raise SceneDataError(
            f"Could not find COLMAP sparse model under {self.scene_root}. "
            "Expected sparse/0 or colmap/sparse/0."
        )

    def _read_colmap_cameras(self) -> Dict[int, ColmapCamera]:
        bin_path = self.colmap_root / "cameras.bin"
        txt_path = self.colmap_root / "cameras.txt"
        if bin_path.exists():
            return _read_colmap_cameras_binary(bin_path)
        if txt_path.exists():
            return _read_colmap_cameras_text(txt_path)
        raise SceneDataError(f"Missing COLMAP cameras file under {self.colmap_root}")

    def _read_colmap_images(self) -> Dict[int, ColmapImage]:
        bin_path = self.colmap_root / "images.bin"
        txt_path = self.colmap_root / "images.txt"
        if bin_path.exists():
            return _read_colmap_images_binary(bin_path)
        if txt_path.exists():
            return _read_colmap_images_text(txt_path)
        raise SceneDataError(f"Missing COLMAP images file under {self.colmap_root}")

    def _index_colmap_images(self) -> Dict[str, ColmapImage]:
        indexed: Dict[str, ColmapImage] = {}
        for image in self.colmap_images.values():
            frame_id = Path(image.name).stem
            if frame_id in indexed:
                raise SceneDataError(f"Duplicate COLMAP frame stem {frame_id!r} in {self.colmap_root}")
            indexed[frame_id] = image
        return indexed

    def _read_split_ids(self, split: str) -> List[str]:
        if split not in {"train", "all"}:
            raise SceneDataError(
                f"COLMAP fallback scene {self.scene_root} has no split metadata for split={split!r}. "
                "Use --scene_split train or all."
            )
        return sorted(self._images_by_frame_id.keys(), key=self._frame_sort_key)

    @staticmethod
    def _frame_sort_key(frame_id: str) -> tuple[int, int | str, str]:
        stem = Path(frame_id).stem
        if stem.isdigit():
            return (0, int(stem), frame_id)
        return (1, stem, frame_id)

    def _colmap_rgb_dirs(self) -> List[str]:
        return self._existing_dirs(("images_2", "images", "rgb/2x", "rgb/1x", "rgb"))

    def _colmap_seg_dirs(self) -> List[str]:
        return self._existing_dirs(("mask", "masks", "segmentation/2x", "segmentation/1x", "segmentation"))

    def _build_record(self, frame_id: str) -> FrameRecord:
        try:
            image = self._images_by_frame_id[frame_id]
        except KeyError as exc:
            raise SceneDataError(f"Unknown COLMAP frame id {frame_id!r}") from exc

        image_name = image.name
        image_stem = Path(image_name).stem
        rgb_dirs = self._colmap_rgb_dirs()
        if not rgb_dirs:
            raise SceneDataError(f"Could not find an RGB image directory under {self.scene_root}")
        rgb = self._resolve_existing(
            [f"{directory}/{image_name}" for directory in rgb_dirs]
            + list(self._candidate_paths(rgb_dirs, (image_stem,), IMAGE_SUFFIXES))
        )

        seg_dirs = self._colmap_seg_dirs()
        seg = None
        if seg_dirs:
            seg = self._resolve_existing(
                [f"{directory}/{image_name}" for directory in seg_dirs]
                + list(self._candidate_paths(seg_dirs, (image_stem,), (*ARRAY_SUFFIXES, *IMAGE_SUFFIXES))),
                required=False,
            )

        camera = self.colmap_cameras.get(image.camera_id)
        if camera is None:
            raise SceneDataError(f"Missing COLMAP camera {image.camera_id} referenced by image {image_name}")
        camera_dict = _colmap_image_to_camera_dict(image, camera)
        return FrameRecord(
            frame_id=frame_id,
            rgb_path=rgb,
            depth_path=None,
            seg_path=seg,
            camera_path=None,
            track_path=None,
            camera_dict=camera_dict,
        )


def create_scene_dataset(
    scene_root: str | Path,
    split: str = "train",
    dynamic_label_mode: str = "nonzero_is_dynamic",
    dynamic_label: Optional[int] = None,
) -> SceneDatadirDataset:
    scene_root = Path(scene_root).expanduser().resolve()
    if (scene_root / "dataset.json").exists():
        return SceneDatadirDataset(
            scene_root=scene_root,
            split=split,
            dynamic_label_mode=dynamic_label_mode,
            dynamic_label=dynamic_label,
        )
    if (scene_root / "sparse" / "0").is_dir() or (scene_root / "colmap" / "sparse" / "0").is_dir():
        return ColmapSparseSceneDataset(
            scene_root=scene_root,
            split=split,
            dynamic_label_mode=dynamic_label_mode,
            dynamic_label=dynamic_label,
        )
    raise SceneDataError(
        f"Could not interpret scene root {scene_root}. "
        "Expected dataset.json or a COLMAP sparse model under sparse/0."
    )
