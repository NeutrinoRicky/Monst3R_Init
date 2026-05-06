# --------------------------------------------------------
# gradio demo
# --------------------------------------------------------

import argparse
import json
import math
import gradio
import os
from pathlib import Path
import torch
import numpy as np
import tempfile
import functools
import copy

from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images, load_prev_video_results, rgb, enlarge_seg_masks
from dust3r.utils.device import to_numpy
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.utils.viz_demo import convert_scene_output_to_glb, get_dynamic_mask_from_pairviewer
from reflow_a1.backproject_split import (
    _adaptive_voxel_downsample,
    _color_diversity_score,
    _resize_float,
    _resize_mask,
    _resize_rgb,
    _select_dynamic_reference,
    _voxel_keep_max_weight,
)
from reflow_a1.coarse_align import extract_alignment_state
from reflow_a1.dataset_scene import create_scene_dataset
from reflow_a1.export_ply import write_ply
from reflow_a1.monst3r_bridge import default_weights_path
from reflow_a1.pair_infer import preprocess_frame_for_monst3r
import matplotlib.pyplot as pl
import cv2

pl.ion()
torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12

STATIC_POINTCLOUD_MAX_POINTS = 20000
DYNAMIC_POINTCLOUD_MAX_POINTS = 10000


def default_demo_weights_path():
    return str(default_weights_path())


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser_url = parser.add_mutually_exclusive_group()
    parser_url.add_argument("--local_network", action='store_true', default=False,
                            help="make app accessible on local network: address will be set to 0.0.0.0")
    parser_url.add_argument("--server_name", type=str, default=None, help="server url, default is 127.0.0.1")
    parser.add_argument("--image_size", type=int, default=512, choices=[512, 224], help="image size")
    parser.add_argument("--server_port", type=int, help=("will start gradio app on this port (if available). "
                                                         "If None, will search for an available port starting at 7860."),
                        default=None)
    parser.add_argument("--weights", type=str, help="path to the model weights", default=default_demo_weights_path())
    parser.add_argument("--model_name", type=str, default='Junyi42/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt', help="model name")
    parser.add_argument("--device", type=str, default='cuda', help="pytorch device")
    parser.add_argument("--output_dir", type=str, default='./demo_tmp', help="value for tempfile.tempdir")
    parser.add_argument("--prev_output_dir", type=str, default=None, help="previous output dir")
    parser.add_argument("--prev_output_index", type=int, default=None, help="previous output video index")
    parser.add_argument("--silent", action='store_true', default=False,
                        help="silence logs")
    parser.add_argument("--input_dir", type=str, help="Path to input images directory", default=None)
    parser.add_argument("--scene_root", type=str, default=None,
                        help="Optional scene root. Supports scene_datadir/HyperNeRF layouts or COLMAP sparse scenes under sparse/0; when set, demo loads source cameras directly.")
    parser.add_argument("--scene_split", type=str, default="train",
                        help="dataset.json split prefix to use when --scene_root is set")
    parser.add_argument("--scene_camera_mode", type=str, default="fixed", choices=["none", "init", "fixed"],
                        help="How to use source cameras with --scene_root: ignore them, use as initialization, or keep them frozen")
    parser.add_argument("--scene_keyframe_stride", type=int, default=5,
                        help="Use every Nth frame from the chosen scene split as the optimization keyframes")
    parser.add_argument("--pointcloud_min_confidence", type=float, default=3.0,
                        help="Confidence threshold used for static point-cloud export")
    parser.add_argument("--dynamic_pointcloud_min_confidence", type=float, default=-1.0,
                        help="Confidence threshold used for dynamic point-cloud export. Negative values reuse min_conf_thr on init_conf maps; 0 disables dynamic confidence filtering.")
    parser.add_argument("--static_voxel_size", type=float, default=0.0,
                        help="Voxel size for static point-cloud export; <=0 selects an automatic size")
    parser.add_argument("--seq_name", type=str, help="Sequence name for evaluation", default='NULL')
    parser.add_argument('--use_gt_davis_masks', action='store_true', default=False, help='Use ground truth masks for DAVIS')
    parser.add_argument('--not_batchify', action='store_true', default=False, help='Use non batchify mode for global optimization')
    parser.add_argument('--real_time', action='store_true', default=False, help='Realtime mode')
    parser.add_argument('--window_wise', action='store_true', default=False, help='Use window wise mode for optimization')
    parser.add_argument('--window_size', type=int, default=100, help='Window size')
    parser.add_argument('--window_overlap_ratio', type=float, default=0.5, help='Window overlap ratio')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for inference')

    parser.add_argument('--fps', type=int, default=0, help='FPS for video processing')
    parser.add_argument('--num_frames', type=int, default=200, help='Maximum number of frames for video processing')
    
    # Add "share" argument if you want to make the demo accessible on the public internet
    parser.add_argument("--share", action='store_true', default=False, help="Share the demo")
    return parser

def get_3D_model_from_scene(outdir, silent, scene, min_conf_thr=3, as_pointcloud=False, mask_sky=False,
                            clean_depth=False, transparent_cams=False, cam_size=0.05, show_cam=True, save_name=None, thr_for_init_conf=True):
    """
    extract 3D_model (glb file) from a reconstructed scene
    """
    if scene is None:
        return None
    # post processes
    if clean_depth:
        scene = scene.clean_pointcloud()
    if mask_sky:
        scene = scene.mask_sky()

    # get optimized values from scene
    rgbimg = scene.imgs
    focals = scene.get_focals().cpu()
    cams2world = scene.get_im_poses().cpu()
    # 3D pointcloud from depthmap, poses and intrinsics
    pts3d = to_numpy(scene.get_pts3d(raw_pts=True))
    scene.min_conf_thr = min_conf_thr
    scene.thr_for_init_conf = thr_for_init_conf
    msk = to_numpy(scene.get_masks())
    cmap = pl.get_cmap('viridis')
    cam_color = [cmap(i/len(rgbimg))[:3] for i in range(len(rgbimg))]
    cam_color = [(255*c[0], 255*c[1], 255*c[2]) for c in cam_color]
    return convert_scene_output_to_glb(outdir, rgbimg, pts3d, msk, focals, cams2world, as_pointcloud=as_pointcloud,
                                        transparent_cams=transparent_cams, cam_size=cam_size, show_cam=show_cam, silent=silent, save_name=save_name,
                                        cam_color=cam_color)


def resolve_seq_name(args, seq_name):
    if seq_name != "NULL":
        return seq_name
    if args.scene_root:
        return Path(args.scene_root).resolve().name
    if args.input_dir:
        return Path(args.input_dir).resolve().stem
    return seq_name


def load_scene_views(args, image_size, silent, num_frames):
    dataset = create_scene_dataset(args.scene_root, split=args.scene_split)
    frame_ids = dataset.get_train_frame_ids() if args.scene_split == "train" else list(dataset.frame_ids)
    stride = max(int(args.scene_keyframe_stride), 1)
    frame_ids = frame_ids[::stride]
    if num_frames is not None and num_frames > 0:
        frame_ids = frame_ids[:num_frames]

    imgs = []
    for idx, frame_id in enumerate(frame_ids):
        frame = dataset.get_frame_by_id(frame_id)
        imgs.append(
            preprocess_frame_for_monst3r(
                frame,
                idx,
                image_size,
                crop=True,
                square_ok=False,
            )
        )
    if not silent:
        print(
            f'>> Loaded {len(imgs)} keyframes from scene_datadir {args.scene_root} '
            f'(split={args.scene_split}, stride={stride})'
        )
    return dataset, imgs, frame_ids


def init_scene_with_known_cameras(scene, imgs, camera_mode):
    if camera_mode == "none":
        return

    import dust3r.cloud_opt.init_im_poses as init_fun

    known_intrinsics = [img["camera_intrinsics"][0].detach().cpu().float() for img in imgs]
    known_poses = [img["camera_pose"][0].detach().cpu().float() for img in imgs]
    scene.preset_intrinsics(known_intrinsics)
    scene.preset_pose(known_poses, requires_grad=(camera_mode == "init"))
    init_fun.init_from_known_poses(scene, min_conf_thr=scene.min_conf_thr)


def _cap_point_cloud(points, colors, target_points, rng_seed, label):
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    colors_arr = None if colors is None else np.asarray(colors).reshape(-1, 3)
    target_points = int(target_points)
    info = {
        "label": str(label),
        "target_points": int(target_points),
        "input_points": int(len(points)),
        "output_points": int(len(points)),
        "method": "not_needed",
    }

    if target_points < 0 or len(points) <= target_points:
        return points, colors_arr, info
    if target_points == 0:
        info.update(
            {
                "method": "drop_all",
                "output_points": 0,
            }
        )
        empty_points = np.empty((0, 3), dtype=np.float32)
        empty_colors = None if colors_arr is None else np.empty((0, 3), dtype=colors_arr.dtype)
        return empty_points, empty_colors, info

    capped_points, capped_colors, adaptive_info = _adaptive_voxel_downsample(
        points,
        colors_arr,
        target_points=target_points,
        rng_seed=int(rng_seed),
    )
    adaptive_info["label"] = str(label)
    return capped_points, capped_colors, adaptive_info


def _apply_per_cloud_point_caps(
    static_pts,
    static_col,
    dynamic_pts,
    dynamic_col,
    static_max_points,
    dynamic_max_points,
):
    static_pts, static_col, static_meta = _cap_point_cloud(
        static_pts,
        static_col,
        static_max_points,
        rng_seed=2024,
        label="static",
    )
    dynamic_pts, dynamic_col, dynamic_meta = _cap_point_cloud(
        dynamic_pts,
        dynamic_col,
        dynamic_max_points,
        rng_seed=2025,
        label="dynamic",
    )
    meta = {
        "strategy": "independent_per_cloud_caps",
        "caps": {
            "static": int(static_max_points),
            "dynamic": int(dynamic_max_points),
        },
        "static_downsample": static_meta,
        "dynamic_downsample": dynamic_meta,
        "output_points": {
            "static": int(len(static_pts)),
            "dynamic": int(len(dynamic_pts)),
            "combined": int(len(static_pts) + len(dynamic_pts)),
        },
    }
    return static_pts, static_col, dynamic_pts, dynamic_col, meta


def _frame_mask_to_numpy(mask):
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    mask = np.asarray(mask)
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    elif mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    return mask.astype(bool)


def _frame_rgb_to_numpy(img_entry):
    img_tensor = img_entry.get("img")
    if img_tensor is None:
        return None
    img_rgb = rgb(img_tensor)
    if isinstance(img_rgb, list):
        img_rgb = img_rgb[0]
    img_rgb = np.asarray(img_rgb)
    if img_rgb.ndim == 4 and img_rgb.shape[0] == 1:
        img_rgb = img_rgb[0]
    return np.clip(img_rgb.astype(np.float32), 0.0, 1.0)


def build_input_frame_records(imgs):
    frame_ids = []
    frame_records = {}
    for idx, img_entry in enumerate(imgs):
        frame_id = f"{idx:06d}"
        frame_ids.append(frame_id)
        frame_records[frame_id] = {
            "frame_id": frame_id,
            "instance": img_entry.get("instance"),
            "rgb": _frame_rgb_to_numpy(img_entry),
            "dynamic_mask": _frame_mask_to_numpy(img_entry.get("dynamic_mask")),
        }
    return frame_ids, frame_records


def export_pointclouds_from_records(args, scene, frame_ids, frame_records, save_folder, silent):
    alignment_state = extract_alignment_state(scene, frame_ids)

    static_points = []
    static_colors = []
    static_weights = []
    dynamic_frames = []

    static_pointcloud_min_confidence = float(args.pointcloud_min_confidence)
    dynamic_pointcloud_min_confidence = float(args.dynamic_pointcloud_min_confidence)
    if dynamic_pointcloud_min_confidence < 0:
        dynamic_pointcloud_min_confidence = float(getattr(scene, "min_conf_thr", 0.0))

    dynamic_confidence_source = "none"
    for frame_id in frame_ids:
        frame = frame_records[frame_id]
        pts = np.asarray(alignment_state["global_pointmaps"][frame_id], dtype=np.float32)
        h, w = pts.shape[:2]

        colors = alignment_state["colors"].get(frame_id)
        if colors is None:
            colors = frame["rgb"]
        colors = _resize_rgb(np.asarray(colors), (h, w))

        dyn_mask = alignment_state["dynamic_masks"].get(frame_id)
        if dyn_mask is None:
            dyn_mask = frame["dynamic_mask"]
        dyn_mask = _resize_mask(np.asarray(dyn_mask), (h, w))

        base_valid = np.isfinite(pts).all(axis=-1)
        depth_map = alignment_state["depths"].get(frame_id)
        if depth_map is not None:
            depth_map = _resize_float(np.asarray(depth_map), (h, w))
            base_valid &= np.isfinite(depth_map) & (depth_map > 0)

        static_valid = base_valid.copy()
        valid_mask = alignment_state["valid_masks"].get(frame_id)
        if valid_mask is not None:
            static_valid &= _resize_mask(np.asarray(valid_mask), (h, w))

        conf_map = alignment_state["confidence"].get(frame_id)
        if conf_map is not None:
            conf_map = _resize_float(np.asarray(conf_map), (h, w))
        else:
            conf_map = None
        if conf_map is not None and static_pointcloud_min_confidence > 0:
            static_valid &= conf_map >= static_pointcloud_min_confidence

        static_valid_idx = np.flatnonzero(static_valid.reshape(-1))
        if static_valid_idx.size > 0:
            pts_static = pts.reshape(-1, 3)[static_valid_idx]
            colors_static = colors.reshape(-1, 3)[static_valid_idx]
            dyn_static = dyn_mask.reshape(-1)[static_valid_idx]
            if conf_map is None:
                conf_static = np.ones((static_valid_idx.size,), dtype=np.float32)
            else:
                conf_static = conf_map.reshape(-1)[static_valid_idx].astype(np.float32)

            static_sel = ~dyn_static
            if np.any(static_sel):
                static_points.append(pts_static[static_sel])
                static_colors.append(colors_static[static_sel])
                static_weights.append(np.clip(conf_static[static_sel], 1e-3, None))

        dynamic_conf_map = alignment_state.get("init_confidence", {}).get(frame_id)
        if dynamic_conf_map is not None:
            dynamic_conf_map = _resize_float(np.asarray(dynamic_conf_map), (h, w))
            dynamic_confidence_source = "init_confidence"
        elif conf_map is not None:
            dynamic_conf_map = conf_map
            if dynamic_confidence_source == "none":
                dynamic_confidence_source = "optimized_confidence"
        else:
            dynamic_conf_map = None

        dynamic_valid_mask = base_valid & dyn_mask
        if dynamic_conf_map is not None and dynamic_pointcloud_min_confidence > 0:
            dynamic_valid_mask &= dynamic_conf_map >= dynamic_pointcloud_min_confidence

        dynamic_colors_for_score = colors[dynamic_valid_mask]
        dynamic_valid_idx = np.flatnonzero(dynamic_valid_mask.reshape(-1))
        if dynamic_valid_idx.size > 0:
            dynamic_frames.append(
                {
                    "frame_id": frame_id,
                    "points": pts.reshape(-1, 3)[dynamic_valid_idx],
                    "colors": colors.reshape(-1, 3)[dynamic_valid_idx],
                    "stats": {
                        "frame_id": frame_id,
                        "total_pixels": int(h * w),
                        "static_valid_pixels": int(np.count_nonzero(static_valid)),
                        "dynamic_mask_pixels": int(np.count_nonzero(dyn_mask)),
                        "dynamic_static_valid_pixels": int(np.count_nonzero(static_valid & dyn_mask)),
                        "dynamic_valid_pixels": int(np.count_nonzero(dynamic_valid_mask)),
                        "dynamic_mask_ratio": float(np.count_nonzero(dyn_mask) / max(h * w, 1)),
                        "dynamic_valid_ratio": float(np.count_nonzero(dynamic_valid_mask) / max(h * w, 1)),
                        "dynamic_mask_keep_ratio": float(
                            np.count_nonzero(dynamic_valid_mask) / max(np.count_nonzero(dyn_mask), 1)
                        ),
                        "dynamic_color_diversity": _color_diversity_score(dynamic_colors_for_score),
                    },
                }
            )

    if static_points:
        static_pts = np.concatenate(static_points, axis=0)
        static_col = np.concatenate(static_colors, axis=0)
        static_w = np.concatenate(static_weights, axis=0)
    else:
        static_pts = np.empty((0, 3), dtype=np.float32)
        static_col = np.empty((0, 3), dtype=np.float32)
        static_w = np.empty((0,), dtype=np.float32)

    static_pts, static_col, static_w, static_voxel_meta = _voxel_keep_max_weight(
        static_pts,
        static_col,
        static_w,
        voxel_size=float(args.static_voxel_size),
    )
    dynamic_pts, dynamic_col, dynamic_reference = _select_dynamic_reference(dynamic_frames)
    pre_budget_counts = {
        "static": int(len(static_pts)),
        "dynamic": int(len(dynamic_pts)),
        "combined": int(len(static_pts) + len(dynamic_pts)),
    }
    static_pts, static_col, dynamic_pts, dynamic_col, point_budget_meta = _apply_per_cloud_point_caps(
        static_pts,
        static_col,
        dynamic_pts,
        dynamic_col,
        static_max_points=int(STATIC_POINTCLOUD_MAX_POINTS),
        dynamic_max_points=int(DYNAMIC_POINTCLOUD_MAX_POINTS),
    )

    combined_pts = np.concatenate([static_pts, dynamic_pts], axis=0)
    combined_col = np.concatenate([static_col, dynamic_col], axis=0)

    static_path = Path(save_folder) / "static_complete.ply"
    dynamic_path = Path(save_folder) / "dynamic_complete.ply"
    combined_path = Path(save_folder) / "combined_complete.ply"
    write_ply(static_path, static_pts, static_col)
    write_ply(dynamic_path, dynamic_pts, dynamic_col)
    write_ply(combined_path, combined_pts, combined_col)

    export_summary = {
        "frame_ids": list(frame_ids),
        "num_keyframes": int(len(frame_ids)),
        "pointcloud_min_confidence": float(static_pointcloud_min_confidence),
        "static_pointcloud_min_confidence": float(static_pointcloud_min_confidence),
        "dynamic_pointcloud_min_confidence": float(dynamic_pointcloud_min_confidence),
        "pointcloud_caps": {
            "static_max_points": int(STATIC_POINTCLOUD_MAX_POINTS),
            "dynamic_max_points": int(DYNAMIC_POINTCLOUD_MAX_POINTS),
        },
        "dynamic_confidence_source": dynamic_confidence_source,
        "static_voxel_filter": static_voxel_meta,
        "point_budget": point_budget_meta,
        "dynamic_reference": {
            key: value
            for key, value in dynamic_reference.items()
            if key != "frame_stats"
        },
        "files": {
            "static_complete": str(static_path),
            "dynamic_complete": str(dynamic_path),
            "combined_complete": str(combined_path),
        },
        "num_points_before_budget": pre_budget_counts,
        "num_points": {
            "static": int(len(static_pts)),
            "dynamic": int(len(dynamic_pts)),
            "combined": int(len(combined_pts)),
        },
    }
    with (Path(save_folder) / "pointcloud_summary.json").open("w", encoding="utf-8") as f:
        json.dump(export_summary, f, indent=2)
    with (Path(save_folder) / "dynamic_reference_stats.json").open("w", encoding="utf-8") as f:
        json.dump(dynamic_reference, f, indent=2)

    if not silent:
        print(
            f'>> Exported point clouds: static={len(static_pts)}, '
            f'dynamic={len(dynamic_pts)}, combined={len(combined_pts)}'
        )

    return export_summary


def export_scene_root_pointclouds(args, scene, dataset, frame_ids, save_folder, silent):
    frame_records = {frame_id: dataset.get_frame_by_id(frame_id) for frame_id in frame_ids}
    return export_pointclouds_from_records(args, scene, frame_ids, frame_records, save_folder, silent)


def get_reconstructed_scene(args, outdir, model, device, silent, image_size, filelist, schedule, niter, min_conf_thr,
                            as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size, show_cam, scenegraph_type, winsize, refid, 
                            seq_name, new_model_weights, temporal_smoothing_weight, translation_weight, shared_focal, 
                            flow_loss_weight, flow_loss_start_iter, flow_loss_threshold, use_gt_mask, fps, num_frames):
    """
    from a list of images, run dust3r inference, global aligner.
    then run get_3D_model_from_scene
    """
    translation_weight = float(translation_weight)
    if new_model_weights != args.weights:
        model = AsymmetricCroCo3DStereo.from_pretrained(new_model_weights).to(device)
    model.eval()
    seq_name = resolve_seq_name(args, seq_name)
    if use_gt_mask and seq_name != "NULL":
        dynamic_mask_path = f'data/davis/DAVIS/masked_images/480p/{seq_name}'
    else:
        dynamic_mask_path = None

    frame_ids = None
    frame_records = None
    dataset = None
    using_scene_root = bool(args.scene_root)
    input_frame_stride = max(int(args.scene_keyframe_stride), 1) if args.input_dir is not None else 1
    if using_scene_root:
        if args.window_wise:
            raise ValueError("--scene_root currently supports non-window-wise optimization only")
        dataset, imgs, frame_ids = load_scene_views(args, image_size, silent, num_frames)
        prev_video_results = None
    elif args.window_wise and args.prev_output_dir is not None:
        prev_num_frames = int(args.window_size * args.window_overlap_ratio)
        prev_video_results = load_prev_video_results(args.prev_output_dir, num_frames=prev_num_frames, index=args.prev_output_index)
        imgs = load_images(filelist, size=image_size, verbose=not silent, dynamic_mask_root=dynamic_mask_path,
                           fps=fps, num_frames=num_frames, frame_stride=input_frame_stride, imgs=prev_video_results['imgs'])
    else:
        prev_video_results = None
        imgs = load_images(filelist, size=image_size, verbose=not silent, dynamic_mask_root=dynamic_mask_path,
                           fps=fps, num_frames=num_frames, frame_stride=input_frame_stride)
    if not using_scene_root:
        frame_ids, frame_records = build_input_frame_records(imgs)
        
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1
    if scenegraph_type == "swin" or scenegraph_type == "swinstride" or scenegraph_type == "swin2stride":
        scenegraph_type = scenegraph_type + "-" + str(winsize) + "-noncyclic"
    elif scenegraph_type == "oneref":
        scenegraph_type = scenegraph_type + "-" + str(refid)

    pairs = make_pairs(imgs, scene_graph=scenegraph_type, prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=args.batch_size, verbose=not silent)
    # TODO YYJ del model
    if len(imgs) > 2:
        mode = GlobalAlignerMode.PointCloudOptimizer
        use_known_cameras = using_scene_root and args.scene_camera_mode in ("init", "fixed")
        use_supplied_masks = all(bool(img.get("has_dynamic_mask", False)) for img in imgs)
        use_self_mask = (not use_gt_mask) and (not use_supplied_masks)
        sam2_mask_refine = not use_supplied_masks
        scene = global_aligner(output, device=device, mode=mode, verbose=not silent, shared_focal=(False if use_known_cameras else shared_focal),
                               optimize_pp=use_known_cameras, temporal_smoothing_weight=temporal_smoothing_weight, translation_weight=translation_weight,
                               flow_loss_weight=flow_loss_weight, flow_loss_start_epoch=flow_loss_start_iter, flow_loss_thre=flow_loss_threshold,
                               use_self_mask=use_self_mask, sam2_mask_refine=sam2_mask_refine,
                               num_total_iter=niter, empty_cache=len(imgs) > 72, batchify=not (args.not_batchify or args.window_wise),
                               window_wise=args.window_wise, window_size=args.window_size, window_overlap_ratio=args.window_overlap_ratio,
                               prev_video_results=prev_video_results)
        if use_known_cameras:
            init_scene_with_known_cameras(scene, imgs, args.scene_camera_mode)
    else:
        mode = GlobalAlignerMode.PairViewer
        scene = global_aligner(output, device=device, mode=mode, verbose=not silent)
    lr = 0.01

    if mode == GlobalAlignerMode.PointCloudOptimizer:
        if args.window_wise:
            scene.compute_window_wise_alignment(init='mst', niter=niter, schedule=schedule, lr=lr)
        else:
            init_mode = None if (using_scene_root and args.scene_camera_mode in ("init", "fixed")) else 'mst'
            scene.compute_global_alignment(init=init_mode, niter=niter, schedule=schedule, lr=lr)

    if args.window_wise and args.prev_output_dir is not None:
        scene.clean_prev_results()
        
    save_folder = f'{args.output_dir}/{seq_name}'  #default is 'demo_tmp/NULL'
    os.makedirs(save_folder, exist_ok=True)
    outfile = get_3D_model_from_scene(save_folder, silent, scene, min_conf_thr, as_pointcloud, mask_sky,
                            clean_depth, transparent_cams, cam_size, show_cam)

    poses = scene.save_tum_poses(f'{save_folder}/pred_traj.txt')
    K = scene.save_intrinsics(f'{save_folder}/pred_intrinsics.txt')
    depth_maps = scene.save_depth_maps(save_folder)
    dynamic_masks = scene.save_dynamic_masks(save_folder)
    conf = scene.save_conf_maps(save_folder)
    init_conf = scene.save_init_conf_maps(save_folder)
    rgbs = scene.save_rgb_imgs(save_folder)
    enlarge_seg_masks(save_folder, kernel_size=5 if use_gt_mask else 3) 
    if frame_ids is not None:
        with open(f'{save_folder}/frame_ids.txt', 'w', encoding='utf-8') as f:
            for frame_id in frame_ids:
                f.write(f'{frame_id}\n')
    can_export_split_clouds = frame_ids is not None and len(frame_ids) == len(scene.imgs)
    if using_scene_root and dataset is not None and can_export_split_clouds:
        scene.min_conf_thr = min_conf_thr
        export_scene_root_pointclouds(
            args=args,
            scene=scene,
            dataset=dataset,
            frame_ids=frame_ids,
            save_folder=save_folder,
            silent=silent,
        )
    elif frame_records is not None and can_export_split_clouds:
        scene.min_conf_thr = min_conf_thr
        export_pointclouds_from_records(
            args=args,
            scene=scene,
            frame_ids=frame_ids,
            frame_records=frame_records,
            save_folder=save_folder,
            silent=silent,
        )

    # also return rgb, depth and confidence imgs
    # depth is normalized with the max value for all images
    # we apply the jet colormap on the confidence maps
    rgbimg = scene.imgs
    depths = to_numpy(scene.get_depthmaps())
    confs = to_numpy([c for c in scene.im_conf])
    init_confs = to_numpy([c for c in scene.init_conf_maps])
    cmap = pl.get_cmap('jet')
    depths_max = max([d.max() for d in depths])
    depths = [cmap(d/depths_max) for d in depths]
    confs_max = max([d.max() for d in confs])
    confs = [cmap(d/confs_max) for d in confs]
    init_confs_max = max([d.max() for d in init_confs])
    init_confs = [cmap(d/init_confs_max) for d in init_confs]

    imgs = []
    for i in range(len(rgbimg)):
        imgs.append(rgbimg[i])
        imgs.append(rgb(depths[i]))
        imgs.append(rgb(confs[i]))
        imgs.append(rgb(init_confs[i]))

    # if two images, and the shape is same, we can compute the dynamic mask
    if len(rgbimg) == 2 and rgbimg[0].shape == rgbimg[1].shape:
        motion_mask_thre = 0.35
        error_map = get_dynamic_mask_from_pairviewer(scene, both_directions=True, output_dir=args.output_dir, motion_mask_thre=motion_mask_thre)
        # imgs.append(rgb(error_map))
        # apply threshold on the error map
        normalized_error_map = (error_map - error_map.min()) / (error_map.max() - error_map.min())
        error_map_max = normalized_error_map.max()
        error_map = cmap(normalized_error_map/error_map_max)
        imgs.append(rgb(error_map))
        binary_error_map = (normalized_error_map > motion_mask_thre).astype(np.uint8)
        imgs.append(rgb(binary_error_map*255))

    return scene, outfile, imgs


def set_scenegraph_options(inputfiles, winsize, refid, scenegraph_type):
    # if inputfiles[0] is a video, set the num_files to 200
    if inputfiles is not None and len(inputfiles) == 1 and inputfiles[0].name.endswith(('.mp4', '.avi', '.mov', '.MP4', '.AVI', '.MOV')):
        num_files = 200
    else:
        num_files = len(inputfiles) if inputfiles is not None else 1
    max_winsize = max(1, math.ceil((num_files-1)/2))
    if scenegraph_type == "swin" or scenegraph_type == "swin2stride" or scenegraph_type == "swinstride":
        winsize = gradio.Slider(label="Scene Graph: Window Size", value=min(max_winsize,5),
                                minimum=1, maximum=max_winsize, step=1, visible=True)
        refid = gradio.Slider(label="Scene Graph: Id", value=0, minimum=0,
                              maximum=num_files-1, step=1, visible=False)
    elif scenegraph_type == "oneref":
        winsize = gradio.Slider(label="Scene Graph: Window Size", value=max_winsize,
                                minimum=1, maximum=max_winsize, step=1, visible=False)
        refid = gradio.Slider(label="Scene Graph: Id", value=0, minimum=0,
                              maximum=num_files-1, step=1, visible=True)
    else:
        winsize = gradio.Slider(label="Scene Graph: Window Size", value=max_winsize,
                                minimum=1, maximum=max_winsize, step=1, visible=False)
        refid = gradio.Slider(label="Scene Graph: Id", value=0, minimum=0,
                              maximum=num_files-1, step=1, visible=False)
    return winsize, refid


def get_reconstructed_scene_realtime(args, model, device, silent, image_size, filelist, scenegraph_type, refid, seq_name, fps, num_frames):
    """
    from a list of images, run dust3r inference, global aligner.
    then run get_3D_model_from_scene
    """
    model.eval()
    input_frame_stride = max(int(args.scene_keyframe_stride), 1) if args.input_dir is not None else 1
    imgs = load_images(filelist, size=image_size, verbose=not silent, fps=fps, num_frames=num_frames,
                       frame_stride=input_frame_stride)
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1
    
    if scenegraph_type == "oneref":
        scenegraph_type = scenegraph_type + "-" + str(refid)
    elif scenegraph_type == "oneref_mid":
        scenegraph_type = "oneref-" + str(len(imgs) // 2)
    else:
        raise ValueError(f"Unknown scenegraph type for realtime mode: {scenegraph_type}")
    
    pairs = make_pairs(imgs, scene_graph=scenegraph_type, prefilter=None, symmetrize=False)
    output = inference(pairs, model, device, batch_size=args.batch_size, verbose=not silent)

    save_folder = f'{args.output_dir}/{seq_name}'  #default is 'demo_tmp/NULL'
    os.makedirs(save_folder, exist_ok=True)


    view1, view2, pred1, pred2 = output['view1'], output['view2'], output['pred1'], output['pred2']
    pts1 = pred1['pts3d'].detach().cpu().numpy()
    pts2 = pred2['pts3d_in_other_view'].detach().cpu().numpy()
    for batch_idx in range(len(view1['img'])):
        colors1 = rgb(view1['img'][batch_idx])
        colors2 = rgb(view2['img'][batch_idx])
        xyzrgb1 = np.concatenate([pts1[batch_idx], colors1], axis=-1)   #(H, W, 6)
        xyzrgb2 = np.concatenate([pts2[batch_idx], colors2], axis=-1)
        np.save(save_folder + '/pts3d1_p' + str(batch_idx) + '.npy', xyzrgb1)
        np.save(save_folder + '/pts3d2_p' + str(batch_idx) + '.npy', xyzrgb2)

        conf1 = pred1['conf'][batch_idx].detach().cpu().numpy()
        conf2 = pred2['conf'][batch_idx].detach().cpu().numpy()
        np.save(save_folder + '/conf1_p' + str(batch_idx) + '.npy', conf1)
        np.save(save_folder + '/conf2_p' + str(batch_idx) + '.npy', conf2)

        # save the imgs of two views
        img1_rgb = cv2.cvtColor(colors1 * 255, cv2.COLOR_BGR2RGB)
        img2_rgb = cv2.cvtColor(colors2 * 255, cv2.COLOR_BGR2RGB)
        cv2.imwrite(save_folder + '/img1_p' + str(batch_idx) + '.png', img1_rgb)
        cv2.imwrite(save_folder + '/img2_p' + str(batch_idx) + '.png', img2_rgb)

    return save_folder


def main_demo(tmpdirname, model, device, image_size, server_name, server_port, silent=False, args=None):
    recon_fun = functools.partial(get_reconstructed_scene, args, tmpdirname, model, device, silent, image_size)
    model_from_scene_fun = functools.partial(get_3D_model_from_scene, tmpdirname, silent)
    with gradio.Blocks(css=""".gradio-container {margin: 0 !important; min-width: 100%};""", title="MonST3R Demo") as demo:
        # scene state is save so that you can change conf_thr, cam_size... without rerunning the inference
        scene = gradio.State(None)
        gradio.HTML(f'<h2 style="text-align: center;">MonST3R Demo</h2>')
        with gradio.Column():
            inputfiles = gradio.File(file_count="multiple")
            with gradio.Row():
                schedule = gradio.Dropdown(["linear", "cosine"],
                                           value='linear', label="schedule", info="For global alignment!")
                niter = gradio.Number(value=300, precision=0, minimum=0, maximum=5000,
                                      label="num_iterations", info="For global alignment!")
                seq_name = gradio.Textbox(label="Sequence Name", placeholder="NULL", value=args.seq_name, info="For evaluation")
                scenegraph_type = gradio.Dropdown(["complete", "swin", "oneref", "swinstride", "swin2stride"],
                                                  value='swinstride', label="Scenegraph",
                                                  info="Define how to make pairs",
                                                  interactive=True)
                winsize = gradio.Slider(label="Scene Graph: Window Size", value=5,
                                        minimum=1, maximum=1, step=1, visible=False)
                refid = gradio.Slider(label="Scene Graph: Id", value=0, minimum=0, maximum=0, step=1, visible=False)

            run_btn = gradio.Button("Run")

            with gradio.Row():
                # adjust the confidence thresholdx
                min_conf_thr = gradio.Slider(label="min_conf_thr", value=1.1, minimum=0.0, maximum=20, step=0.01)
                # adjust the camera size in the output pointcloud
                cam_size = gradio.Slider(label="cam_size", value=0.05, minimum=0.001, maximum=0.1, step=0.001)
                # adjust the temporal smoothing weight
                temporal_smoothing_weight = gradio.Slider(label="temporal_smoothing_weight", value=0.01, minimum=0.0, maximum=0.1, step=0.001)
                # add translation weight
                translation_weight = gradio.Textbox(label="translation_weight", placeholder="1.0", value="1.0", info="For evaluation")
                # change to another model
                new_model_weights = gradio.Textbox(label="New Model", placeholder=args.weights, value=args.weights, info="Path to updated model weights")
            with gradio.Row():
                as_pointcloud = gradio.Checkbox(value=True, label="As pointcloud")
                # two post process implemented
                mask_sky = gradio.Checkbox(value=False, label="Mask sky")
                clean_depth = gradio.Checkbox(value=True, label="Clean-up depthmaps")
                transparent_cams = gradio.Checkbox(value=False, label="Transparent cameras")
                # not to show camera
                show_cam = gradio.Checkbox(value=True, label="Show Camera")
                shared_focal = gradio.Checkbox(value=True, label="Shared Focal Length")
                use_davis_gt_mask = gradio.Checkbox(value=False, label="Use GT Mask (DAVIS)")
            with gradio.Row():
                flow_loss_weight = gradio.Slider(label="Flow Loss Weight", value=0.01, minimum=0.0, maximum=1.0, step=0.001)
                flow_loss_start_iter = gradio.Slider(label="Flow Loss Start Iter", value=0.1, minimum=0, maximum=1, step=0.01)
                flow_loss_threshold = gradio.Slider(label="Flow Loss Threshold", value=25, minimum=0, maximum=100, step=1)
                # for video processing
                fps = gradio.Slider(label="FPS", value=0, minimum=0, maximum=60, step=1)
                num_frames = gradio.Slider(label="Num Frames", value=100, minimum=0, maximum=200, step=1)

            outmodel = gradio.Model3D()
            outgallery = gradio.Gallery(label='rgb,depth,confidence, init_conf', columns=4, height="100%")

            # events
            scenegraph_type.change(set_scenegraph_options,
                                   inputs=[inputfiles, winsize, refid, scenegraph_type],
                                   outputs=[winsize, refid])
            inputfiles.change(set_scenegraph_options,
                              inputs=[inputfiles, winsize, refid, scenegraph_type],
                              outputs=[winsize, refid])
            run_btn.click(fn=recon_fun,
                          inputs=[inputfiles, schedule, niter, min_conf_thr, as_pointcloud,
                                  mask_sky, clean_depth, transparent_cams, cam_size, show_cam,
                                  scenegraph_type, winsize, refid, seq_name, new_model_weights, 
                                  temporal_smoothing_weight, translation_weight, shared_focal, 
                                  flow_loss_weight, flow_loss_start_iter, flow_loss_threshold, use_davis_gt_mask,
                                  fps, num_frames],
                          outputs=[scene, outmodel, outgallery])
            min_conf_thr.release(fn=model_from_scene_fun,
                                 inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                         clean_depth, transparent_cams, cam_size, show_cam],
                                 outputs=outmodel)
            cam_size.change(fn=model_from_scene_fun,
                            inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                    clean_depth, transparent_cams, cam_size, show_cam],
                            outputs=outmodel)
            as_pointcloud.change(fn=model_from_scene_fun,
                                 inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                         clean_depth, transparent_cams, cam_size, show_cam],
                                 outputs=outmodel)
            mask_sky.change(fn=model_from_scene_fun,
                            inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                    clean_depth, transparent_cams, cam_size, show_cam],
                            outputs=outmodel)
            clean_depth.change(fn=model_from_scene_fun,
                               inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                       clean_depth, transparent_cams, cam_size, show_cam],
                               outputs=outmodel)
            transparent_cams.change(model_from_scene_fun,
                                    inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                            clean_depth, transparent_cams, cam_size, show_cam],
                                    outputs=outmodel)
    demo.launch(share=args.share, server_name=server_name, server_port=server_port)


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    if args.output_dir is not None:
        tmp_path = args.output_dir
        os.makedirs(tmp_path, exist_ok=True)
        tempfile.tempdir = tmp_path

    if args.server_name is not None:
        server_name = args.server_name
    else:
        server_name = '0.0.0.0' if args.local_network else '127.0.0.1'

    if args.weights is not None and os.path.exists(args.weights):
        weights_path = args.weights
    else:
        weights_path = args.model_name

    model = AsymmetricCroCo3DStereo.from_pretrained(weights_path).to(args.device)

    # Use the provided output_dir or create a temporary directory
    tmpdirname = args.output_dir if args.output_dir is not None else tempfile.mkdtemp(suffix='monst3r_gradio_demo')

    if not args.silent:
        print('Outputting stuff in', tmpdirname)

    if args.input_dir is not None or args.scene_root is not None:
        # Process images in the input directory with default parameters
        if args.scene_root is not None:
            input_files = None
        elif os.path.isdir(args.input_dir):    # input_dir is a directory of images
            input_files = args.input_dir
        else:   # input_dir is a video
            input_files = [args.input_dir]

        if args.real_time:
            if args.scene_root is not None:
                raise ValueError("--real_time does not currently support --scene_root")
            recon_fun = functools.partial(get_reconstructed_scene_realtime, args, model, args.device, args.silent, args.image_size)
            outfile = recon_fun(
                filelist=input_files,
                scenegraph_type='oneref_mid',
                refid=0,
                seq_name=args.seq_name,
                fps=args.fps,
                num_frames=args.num_frames,
            )
        else:
            recon_fun = functools.partial(get_reconstructed_scene, args, tmpdirname, model, args.device, args.silent, args.image_size)
            # Call the function with default parameters
            scene, outfile, imgs = recon_fun(
                filelist=input_files,
                schedule='linear',
                niter=300,
                min_conf_thr=1.1,
                as_pointcloud=True,
                mask_sky=False,
                clean_depth=True,
                transparent_cams=False,
                cam_size=0.05,
                show_cam=True,
                scenegraph_type='swinstride',
                winsize=5,
                refid=0,
                seq_name=args.seq_name,
                new_model_weights=args.weights,
                temporal_smoothing_weight=0.01,
                translation_weight='1.0',
                shared_focal=True,
                flow_loss_weight=0.01,
                flow_loss_start_iter=0.1,
                flow_loss_threshold=25,
                use_gt_mask=args.use_gt_davis_masks,
                fps=args.fps,
                num_frames=args.num_frames,
            )
        print(f"Processing completed. Output saved in {tmpdirname}/{resolve_seq_name(args, args.seq_name)}")
    else:
        # Launch Gradio demo
        main_demo(tmpdirname, model, args.device, args.image_size, server_name, args.server_port, silent=args.silent, args=args)
