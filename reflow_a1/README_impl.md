# ReFlow A.1 Initialization in MonST3R

This folder implements a minimal ReFlow Appendix A.1 style canonical-space construction pipeline without changing the original MonST3R demo or training code.

## What This Module Does

1. Reads `scene_datadir/dataset.json` and processes only the requested split ids, usually `train_ids`.
2. Builds a hierarchical video graph:
   - non-overlapping clips,
   - first frame of each clip as keyframe,
   - adjacent keyframe coarse pairs,
   - keyframe-to-clip and adjacent-frame fine pairs.
3. Runs MonST3R pairwise pointmap inference through `dust3r.inference.inference`.
4. Reuses `dust3r.cloud_opt.global_aligner` for coarse keyframe alignment and clip-wise fine alignment.
5. Places every fine clip back into the coarse canonical frame using its keyframe.
6. Exports optimized cameras to `optimized_camera/*.json`, keeping the same
   per-frame JSON style as the source `camera/` folder, and writes
   `camera_trajectory_comparison.png` for before/after trajectory inspection.
7. Splits the final canonical points by segmentation:
   - `segmentation != 0` is dynamic by default,
   - `segmentation == 0` is static.
8. Builds dynamic initialization from the best reference frame, selected by
   dynamic-mask coverage and dynamic-region color diversity.
9. Builds static initialization from every frame's static-mask points after
   they are already in canonical/world coordinates, then uses adaptive voxel
   downsampling to hit a stable point budget.
10. Before static export, applies three anti-overlap stages:
    - multi-view depth-consistency filtering,
    - ray/voxel front-surface filtering,
    - weighted single-surface voxel aggregation.
11. Exports `canonical_complete.ply`, `static_complete.ply`,
    `dynamic_complete.ply`, `summary.json`, and `dynamic_reference_stats.json`.

## Why The Graph Is Hierarchical

A full video pair graph is O(N^2), expensive, and often unreliable for far-apart frames with weak co-visibility. ReFlow A.1 uses coarse-to-fine construction instead: keyframe coarse alignment gives the global skeleton, and clip fine alignment fills local temporal detail.

## Camera Anchoring

Current camera policy is fixed by implementation (no CLI switch):

- **Coarse stage**: camera intrinsics + extrinsics are fixed to source COLMAP/scene JSON.
- **Fine stage**: camera intrinsics + extrinsics are initialized from source cameras and then optimized.
- **Coarse loss**: dynamic-mask regions are down-weighted so global alignment focuses on static geometry.

The final optimized poses are written under:

```text
<output_dir>/optimized_camera/
<output_dir>/optimized_camera_index.json
<output_dir>/camera_trajectory_comparison.png
```

`camera_trajectory_comparison.png` draws the original trajectory and optimized
trajectory in one view. For visualization only, the original trajectory is
Sim(3)-aligned to the optimized trajectory so that unanchored MonST3R gauges
are still easy to compare.

For downstream PIDG training on COLMAP/HyperNeRF scenes, prefer keeping the
source camera JSONs as the training cameras instead of using MonST3R-optimized
poses:

```bash
--camera_export_mode source
```

Camera export modes:

- `optimized`: original behavior; writes `optimized_camera/` and marks it as the
  PIDG camera set.
- `source`: writes `source_camera/` by copying the original scene `camera/*.json`
  files and marks those source cameras as the PIDG camera set.
- `both`: writes both `optimized_camera/` and `source_camera/`, but marks
  `source_camera/` as the PIDG camera set.

The selected downstream camera directory is recorded in `summary.json` as:

```text
pidg_camera_source
pidg_camera_dir
pidg_camera_index
```

For a clearer split view and per-frame extrinsic error report, run:

```bash
python compare_camera_extrinsics.py \
  --scene_root /mnt/store/fd/project/dataset/Nvidia_monocular/Balloon1
```

This writes:

- `camera_extrinsics_error.json` (summary + per-frame errors)
- `camera_extrinsics_error.csv` (tabular errors)
- `camera_extrinsics_split_plot.png` (one figure with two 3D subplots: original vs optimized)

## Point Selection

Dynamic point initialization defaults to `--dynamic_reference_mode best_reference`.
For each frame, the exporter computes:

- dynamic mask coverage,
- valid dynamic point ratio,
- color diversity inside the dynamic region.

The selected frame maximizes a weighted coverage/color-diversity score, and
only that frame's dynamic points seed `dynamic_complete.ply`. Use
`--dynamic_reference_mode all_frames` to recover the older behavior.

Static points are aggregated from every frame's static mask after the points
are already in canonical/world coordinates. The default
`--static_target_points 2000000` runs adaptive voxel downsampling on that
candidate pool. Set `--static_target_points 0` to disable it, or additionally
set `--voxel_downsample <size>` when you want a fixed pre-voxel pass.

To reduce multi-layer background overlap, static export now runs:

- `--static_mv_neighbor_radius`, `--static_mv_abs_depth_error`,
  `--static_mv_rel_depth_error`, `--static_mv_min_consistent_views`
  for multi-view depth consistency.
- `--static_front_surface_voxel`, `--static_front_ray_azimuth_bins`,
  `--static_front_ray_elevation_bins` for ray/voxel front-surface retention.
- `--static_surface_aggregate_voxel` for weighted single-surface voxel fusion.

## Accepted Simplifications

- Coarse graph connects keyframes with offsets `1..coarse_max_offset`.
- Default `coarse_max_offset=2` means coarse edges `(i, i+1)` and `(i, i+2)`.
- Set `--coarse_max_offset 3` to additionally include `(i, i+3)` long-range edges.
- If there are `K` keyframes, coarse pair count is `sum_{d=1..M}(K-d)` where `M=coarse_max_offset`.
- Fine graph is clip-local and now supports intra-clip sliding-window offsets `1..fine_max_offset`.
- Default `fine_max_offset=4` means each clip adds `i->i+1, i+2, i+3, i+4` (when valid), plus keyframe-to-all edges.
- Tracks are read by the dataset but not used yet.
- Singleton clips fall back to depth+camera backprojection because no pairwise MonST3R graph exists for one frame.
- This implements A.1 only; it does not implement ReFlow A.2 tri-planes, 4DGS, or flow-matching self-correction.

## Example

From this folder, `/mnt/store/fd/project/DynamicReconstruction/monst3r/reflow_a1`:

现在只需要运行run_reflow_a1.py即可
```bash
CUDA_VISIBLE_DEVICES=2 python run_reflow_a1.py \
  --scene_root /mnt/store/fd/project/dataset/Nvidia_monocular/Balloon1 \
  --split train \
  --clip_len 10 \
  --coarse_max_offset 2 \
  --image_size 512 \
  --niter_coarse 1200 \
  --schedule_coarse linear \
  --lr_coarse 0.005 \
  --static_target_points 2000000 \
  --dynamic_reference_mode best_reference \
  --weights /mnt/store/fd/project/DynamicReconstruction/monst3r/ckpt/model.safetensors

CUDA_VISIBLE_DEVICES=2 python run_reflow_a1.py \
  --scene_root /mnt/store/fd/project/dataset/HyperNeRF/vrig/broom-single \
  --split train \
  --clip_len 10 \
  --coarse_max_offset 2 \
  --image_size 512 \
  --niter_coarse 1200 \
  --schedule_coarse linear \
  --lr_coarse 0.005 \
  --static_target_points 50000 \
  --dynamic_reference_mode best_reference \
  --weights /mnt/store/fd/project/DynamicReconstruction/monst3r/ckpt/model.safetensors \
  --output_dir /mnt/store/fd/project/dataset/HyperNeRF/vrig/broom-single/monst3r_reflow_a1_outputs_aligned

CUDA_VISIBLE_DEVICES=3 python run_reflow_a1.py \
  --scene_root /mnt/store/fd/project/dataset/HyperNeRF/vrig/broom-single \
  --split train \
  --clip_len 10 \
  --coarse_max_offset 2 \
  --image_size 512 \
  --niter_coarse 1200 \
  --schedule_coarse linear \
  --lr_coarse 0.005 \
  --camera_export_mode source \
  --static_target_points 50000 \
  --dynamic_reference_mode best_reference \
  --weights /mnt/store/fd/project/DynamicReconstruction/monst3r/ckpt/model.safetensors \
  --output_dir /mnt/store/fd/project/dataset/HyperNeRF/vrig/broom-single/monst3r_reflow_a1_outputs_sourcecam_fixed


CUDA_VISIBLE_DEVICES=2 python compare_camera_extrinsics.py \
  --scene_root /mnt/store/fd/project/dataset/Nvidia_monocular/Balloon1

CUDA_VISIBLE_DEVICES=2 python downsample_pointclouds.py \
  --input_dir /mnt/store/fd/project/dataset/Nvidia_monocular/Balloon1/monst3r_reflow_a1_outputs \
  --method adaptive_voxel \
  --max_points 2000000

# Aggressive coarse + static-only fine export voxel max-conf filtering
CUDA_VISIBLE_DEVICES=3 python run_reflow_a1.py \
  --scene_root /mnt/store/fd/project/dataset/HyperNeRF/vrig/broom-single \
  --split train \
  --clip_len 10 \
  --coarse_max_offset 9 \
  --niter_coarse 1500 \
  --niter_fine 1500 \
  --lr_coarse 0.05 \
  --lr 0.05 \
  --coarse_voxel_keep_max_conf \
  --fine_voxel_keep_max_conf \
  --fine_voxel_size 0.5 \
  --coarse_voxel_size 0.5 \
  --weights /mnt/store/fd/project/DynamicReconstruction/monst3r/ckpt/model.safetensors \
  --output_dir /mnt/store/fd/project/dataset/HyperNeRF/vrig/broom-single/monst3r_reflow_a1_outputs_new
```

--dynamic_reference_mode best_reference：只用最佳动态参考帧初始化动态点云。
--dynamic_reference_mode all_frames：恢复旧行为，所有帧动态点一起聚合。
--static_target_points 2000000：静态点云自适应 voxel 到约 200 万以内。
--static_target_points 0：关闭静态 adaptive voxel。
--voxel_downsample <size>：仍可作为固定 voxel 预处理使用。
--lr_coarse 不写时会回退使用 --lr（便于沿用 coarse_debug 的命令习惯）。
--fine_voxel_keep_max_conf/--coarse_voxel_keep_max_conf：仅作用于 fine 导出的 static 点。
--fine_voxel_size/--coarse_voxel_size：上面开关对应的体素大小（<=0 自动估计）。
--fine_temporal_smoothing_weight：fine 阶段相机时序平滑 loss 权重。
--fine_camera_pose_prior_weight：fine 阶段相机外参贴近 source 相机的 loss 权重。
--fine_camera_intrinsics_prior_weight：fine 阶段相机内参贴近 source 相机的 loss 权重。
--fine_camera_pose_prior_translation_weight：外参 prior 中平移项相对旋转项的倍率。
--fine_max_offset：fine clip 内部滑动窗口最大 offset（默认 4）。

示例（fine 相机平滑 + 相机贴近 source 约束）：

```bash
--fine_temporal_smoothing_weight 0.01 \
--fine_camera_pose_prior_weight 0.05 \
--fine_camera_intrinsics_prior_weight 0.02 \
--fine_camera_pose_prior_translation_weight 1.0
```

## HyperNeRF / PIDG Coordinate Notes

HyperNeRF/Nerfies camera JSON uses:

- `orientation`: COLMAP-style world-to-camera rotation (`R_cw`)
- `position`: camera center in world coordinates

The reader converts this to MonST3R's camera-to-world pose internally. During
export it writes `orientation` back as `R_cw`, so PIDG-style readers that consume
`orientation`/`position` can read the optimized cameras correctly.

MonST3R/DUSt3R reconstruction still has a free Sim(3) gauge. By default,
`run_reflow_a1.py` now applies a final Sim(3) alignment from optimized camera
centers back to the source camera coordinate system before exporting cameras and
point clouds. Disable this only if you explicitly want the raw MonST3R canonical
frame:

```bash
--no_align_to_source_cameras
```

For PIDG initialization, use the exported `*_pidg_normalized.ply` files rather
than the raw PLY files. They apply the same `(points - scene.center) * scene.scale`
normalization that PIDG applies to HyperNeRF camera positions. A common choice is
to use `static_complete_pidg_normalized.ply` as the initialization point cloud.

Or from `/mnt/store/fd/project/DynamicReconstruction`:

```bash
CUDA_VISIBLE_DEVICES=2 python -m monst3r.reflow_a1.run_reflow_a1 \
  --scene_root /mnt/store/fd/project/dataset/Nvidia_monocular/Balloon1 \
  --split train \
  --clip_len 10 \
  --image_size 512 \
  --static_target_points 2000000 \
  --dynamic_reference_mode best_reference \
  --weights /mnt/store/fd/project/DynamicReconstruction/monst3r/ckpt/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth
```

For a no-model data/pair-graph check:

```bash
python -m monst3r.reflow_a1.run_reflow_a1 \
  --scene_root /mnt/store/fd/project/dataset/Nvidia_monocular/Balloon1 \
  --dry_run
```

For coarse-stage-only debugging (no fine/export split), run:

```bash
CUDA_VISIBLE_DEVICES=3 python run_coarse_debug.py \
  --scene_root /mnt/store/fd/project/dataset/HyperNeRF/vrig/broom-single \
  --clip_len 10 \
  --coarse_max_offset 3 \
  --weights /mnt/store/fd/project/DynamicReconstruction/monst3r/ckpt/model.safetensors \
  --output_dir /mnt/store/fd/project/dataset/HyperNeRF/vrig/broom-single/monst3r_reflow_a1_coarse_debug_offset3 \
  --niter_coarse 1500 \
  --lr 0.05

CUDA_VISIBLE_DEVICES=3 python run_coarse_debug.py \
  --scene_root /mnt/store/fd/project/dataset/HyperNeRF/vrig/broom-single \
  --clip_len 10 \
  --coarse_max_offset 9 \
  --weights /mnt/store/fd/project/DynamicReconstruction/monst3r/ckpt/model.safetensors \
  --output_dir /mnt/store/fd/project/dataset/HyperNeRF/vrig/broom-single/monst3r_reflow_a1_coarse_debug_offset_filter_new \
  --niter_coarse 1500 \
  --lr 0.05 \
  --coarse_voxel_keep_max_conf \
  --coarse_voxel_size 0.5
```

Optional post-export voxel filtering for `coarse_keyframes.ply`:

```bash
--coarse_voxel_keep_max_conf --coarse_voxel_size 0.01
```

This keeps one point per voxel, selected by the highest confidence.
Set `--coarse_voxel_size <= 0` to auto-estimate voxel size.

This writes:

- `coarse_pair_graph.json` (clips/keyframes/coarse-pair edges)
- `coarse_keyframes.ply` (merged keyframe pointmaps for quick geometry check)
- `coarse_pose_error.json` and `coarse_pose_error.csv` (source-vs-coarse pose error)
- `optimized_camera/` + `optimized_camera_index.json`
- `camera_trajectory_comparison.png`
- `coarse_summary.json`

For step-by-step validation with only `coarse + first clip fine`, run:

```bash
CUDA_VISIBLE_DEVICES=3 python run_fine_debug.py \
  --scene_root /mnt/store/fd/project/dataset/HyperNeRF/vrig/broom-single \
  --clip_len 10 \
  --coarse_max_offset 9 \
  --overlap_frames 4 \
  --num_fine_debug_clips 2 \
  --weights /mnt/store/fd/project/DynamicReconstruction/monst3r/ckpt/model.safetensors \
  --niter_coarse 1500 \
  --lr 0.05 \
  --niter_fine 500 \
  --lr_fine 0.02 \
  --fine_camera_anchor_mode fixed \
  --dynamic_min_confidence 5.0 \
  --output_dir /mnt/store/fd/project/dataset/HyperNeRF/vrig/broom-single/monst3r_reflow_a1_fine_debug
```

This writes:

- `fine_debug_pair_graph.json` (full graph + selected clip index/frame ids)
- `coarse/` (coarse-only debug artifacts, same style as `run_coarse_debug.py`)
- `fine_first_clip/fine_first_clip_points.ply` (first-clip fine merged points)
- `fine_first_clip/fine_first_clip_pose_error.json` and `.csv`
  (fine first-clip optimized cameras vs source COLMAP cameras)
- `fine_first_clip/optimized_camera/` + `optimized_camera_index.json`
- `fine_first_clip/camera_trajectory_comparison.png`
- `fine_debug_summary.json`

## Downsample Output PLY

If `monst3r_reflow_a1_outputs` point clouds are too dense, you can batch downsample:

```bash
cd /mnt/store/fd/project/DynamicReconstruction/monst3r/reflow_a1
python downsample_pointclouds.py \
  --input_dir /mnt/store/fd/project/dataset/Nvidia_monocular/Balloon1/monst3r_reflow_a1_outputs \
  --method adaptive_voxel \
  --max_points 150000
```

This writes new files with suffix `_downsampled.ply` next to the originals.

git push -u origin main