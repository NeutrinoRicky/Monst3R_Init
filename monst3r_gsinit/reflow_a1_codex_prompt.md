# Codex任务说明：在 MonST3R 仓库中实现 ReFlow A.1 风格的 Complete Canonical Space Construction

## 0. 任务目标

请在 **MonST3R 仓库内的独立子文件夹/mnt/store/fd/project/DynamicReconstruction/monst3r/monst3r_gsinit**，实现一个最小但完整的 **ReFlow A.1 风格初始化管线**，目标是：

1. 读取给定场景目录 `scene_datadir`
2. 使用 `dataset.json` 里的 `train_ids` 作为训练/处理帧集合
3. 基于 MonST3R 的 pairwise pointmap / 全局对齐能力，实现 **A.1 的基本 coarse-to-fine canonical space construction**
4. 利用给定的 `segmentation`，把最终 canonical 3D 点云拆成：
   - `static_complete.ply`
   - `dynamic_complete.ply`
   - 可选：`canonical_complete.ply`
5. 实现时要把理论意图和代码入口写清楚，方便后续继续扩展到 ReFlow 后续模块

**注意**：
- 这里只做 **ReFlow Appendix A.1 对应的“基础初始化流程”**，不要做 A.2 之后的 tri-plane / 4DGS / self-correction flow matching。
- 重点是：**利用 MonST3R 把静态和动态区域都尽量补全到 canonical point cloud 中**，而不是仅恢复相机或仅恢复深度。
- 优先保证：
  - 代码结构清楚
  - 数据流清楚
  - 输出点云正确
  - 便于后续继续接 ReFlow 风格静动态分离建模

---

## 1. 新建子文件夹建议

请在 MonST3R 仓库根目录下新建：

```text
monst3r/reflow_a1/
```

建议内部结构如下：

```text
monst3r/reflow_a1/
├── __init__.py
├── dataset_scene.py          # 读取 scene_datadir、train_ids、camera/rgb/depth/segmentation/tracks
├── pair_sampler.py           # clip划分、keyframe选择、pair构建
├── pair_infer.py             # 调用MonST3R已有pairwise推理接口，输出pointmap/confidence
├── coarse_align.py           # keyframe级 coarse alignment
├── fine_align.py             # clip内部 fine alignment
├── backproject_split.py      # 用depth+camera反投影，拆成static/dynamic点云
├── export_ply.py             # ply导出
├── run_reflow_a1.py          # 命令行入口
└── README_impl.md            # 可选，实现说明
```

如果 MonST3R 仓库已有更合适的 `demo/`、`datasets/`、`utils/` 结构可复用，可以适度调用，但 **不要把本任务逻辑散落到仓库各处**；尽量把本任务实现集中放在 `reflow_a1/` 下。

---

## 2. 输入数据格式

当前单场景目录格式固定为：

```text
scene_datadir
├── camera
│   ├── C_XXXXX.json
│   └── ...
├── rgb
│   └── 1x
│       ├── C_XXXXX.png
│       └── ...
├── depth
│   └── 1x
│       ├── C_XXXXX.npy
│       └── ...
├── segmentation
│   └── 1x
│       ├── C_XXXXX.npy
│       └── ...
├── tracks
│   └── 1x
│       ├── track_XXXXX.npy
│       └── ...
├── splits
│   ├── train.json
│   └── val.json
├── scene.json
└── dataset.json
```
你可以参考/mnt/store/fd/project/dataset/Nvidia_monocular/Balloon1这里作为一个验证.

### 2.1 必须支持的读取规则

请实现一个 `SceneDatadirDataset`（名字可改，但职责要一致），要求：

1. 读取 `scene_datadir/dataset.json`
2. 从其中解析 `train_ids`
3. 只使用 `train_ids` 对应的帧作为本次 A.1 处理对象
4. 对每个 frame id，读取：
   - RGB: `rgb/1x/C_XXXXX.png`
   - Depth: `depth/1x/C_XXXXX.npy`
   - Segmentation: `segmentation/1x/C_XXXXX.npy`
   - Camera: `camera/C_XXXXX.json`
   - Track: 若存在则读取 `tracks/1x/track_XXXXX.npy`，若不存在不能报错
5. 统一返回一个按时间排序的 frame 列表，每项至少包含：
   - `frame_id`
   - `rgb_path`
   - `depth_path`
   - `seg_path`
   - `camera_path`
   - `track_path or None`
   - `rgb`（按需懒加载或预加载）
   - `depth`
   - `segmentation`
   - `camera_dict`

### 2.2 对 segmentation 的处理要求

这里的 `segmentation` 要作为 **动态区域掩码的直接来源**。

你需要假设：
- `segmentation` 中 **非零区域 = dynamic**
- `segmentation == 0` = static

如果该假设在代码中需要兼容其他编码形式，请：
- 在命令行参数中提供一个可选的 `--dynamic_label_mode`
- 默认使用 `nonzero_is_dynamic`

最终统一得到每帧：

```python
dynamic_mask: H x W  bool
static_mask = ~dynamic_mask
```

---

## 3. 理论目标：要实现的不是普通 MonST3R 推理，而是 ReFlow A.1 的基本版本

这一节非常重要。你实现时请始终对齐下面的理论逻辑。

### 3.1 ReFlow A.1 的核心需求

ReFlow A.1 的 Complete Canonical Space Construction，本质是：

1. 用 geometry foundation model（这里就是 MonST3R）
   对图像对回归 pairwise pointmap
2. 不直接做全视频 O(N^2) 全连边
3. 而是做 **coarse-to-fine hierarchical alignment**：
   - 先把整段视频切成 clips
   - 每个 clip 选 keyframe
   - 先对 keyframes 全局 coarse alignment
   - 再在每个 clip 内做 fine alignment
4. 得到一个更完整、更一致的 canonical 几何
5. 再根据动态掩码把 3D 点拆成 static / dynamic 两部分

所以，本任务不是“简单把每一帧深度反投影拼一起”，而是：
**必须经过 MonST3R pairwise 几何 + 分层对齐 + 再反投影拆分。**

### 3.2 MonST3R 在这里扮演什么角色

MonST3R / DUSt3R 系列的关键是：

- 对一对图像输出对应的 pointmap
- 两张 pointmap 在同一个参考相机坐标系中表达
- 再通过全局 alignment 把多对 pairwise 预测拼成统一世界/全局几何

因此，你在实现时，应当尽量复用 MonST3R 仓库中已有的：

- pairwise inference 入口
- pointmap/confidence 输出结构
- global alignment / scene graph / Lalign 对齐逻辑
- camera / focal / depth 的重参数化与优化逻辑

**不要自己重写一套完全不同的 SfM。**
本任务是“在 MonST3R 现有能力上，组织出 ReFlow A.1 的数据流和层级对齐策略”。

---

## 4. 告诉 Codex 去哪里找 MonST3R 里的计算细节

因为仓库版本可能不同，**不要硬编码假定某个具体文件名一定存在**。请先在仓库内搜索以下关键词，定位已有实现，再复用其中代码：

### 4.1 必搜关键词
在 MonST3R 仓库中全文搜索以下关键词：

```text
global alignment
Lalign
pointmap
confidence
pairwise
intrinsics
focal
pose
depthmap
PnP
sliding window
```

### 4.2 优先复用的功能类型

请优先找到并复用这些“功能”，而不是执着于某个固定文件路径：

1. **pairwise model inference**
   - 输入两张图
   - 输出两张 pointmap + confidence
2. **global pointmap alignment**
   - 多 pair 图上的全局优化
   - 通常会包含相机位姿 / 深度 / 内参 / 尺度 / 刚体变换等变量
3. **pointmap -> depth / camera / pose 相关工具**
4. **点云导出工具**
5. **图像预处理 / resize / tensor 归一化流程**

### 4.3 若 MonST3R 仓库有 demo / notebook / inference script

如果仓库里已有：
- 多帧视频推理 demo
- 两两配对推理 demo
- 全局 alignment demo

请优先复用其内部函数，而不是复制粘贴大量脚本逻辑。

### 4.4 如果找不到完全匹配的现成入口

允许你做一个轻量封装层，把 MonST3R 最底层 pairwise forward 和已有 alignment 模块包起来，
但要满足：
- 不改坏原仓库行为
- 新逻辑集中在 `reflow_a1/`
- 注释中写明“理论对应 ReFlow A.1 / 工程基于 MonST3R existing pairwise + alignment”

---

## 5. 具体实现阶段（请严格按阶段推进）

# Phase 1：数据读取与标准化

实现 `dataset_scene.py`

### 5.1 需要完成的功能

1. 读取 `scene_datadir/dataset.json`
2. 解析 `train_ids`
3. 根据 `train_ids` 构建处理帧序列
4. 对每个 `frame_id`，解析并返回：
   - RGB
   - depth
   - dynamic mask
   - camera json
   - 可选 tracks
5. 提供如下接口：

```python
class SceneDatadirDataset:
    def __init__(self, scene_root: str, split: str = "train"):
        ...
    def __len__(self): ...
    def get_frame(self, idx: int) -> dict: ...
    def get_frame_by_id(self, frame_id: str) -> dict: ...
    def get_train_frame_ids(self) -> list[str]: ...
```

### 5.2 camera json 读取要求

由于不同数据源的 camera json 字段名可能略有差异，请写一个稳健解析函数，最终统一成：

```python
{
    "K": np.ndarray shape (3, 3),         # 如果可直接解析
    "T_wc": np.ndarray shape (4, 4),      # camera-to-world
    "T_cw": np.ndarray shape (4, 4),      # world-to-camera
    "width": int,
    "height": int,
    "fx": float,
    "fy": float,
    "cx": float,
    "cy": float,
}
```

如果 json 原始字段不同，请做兼容映射；如果缺失某些项，要明确抛出易读错误。

### 5.3 输出检查

Phase 1 结束后，至少要能打印：

- 帧总数
- train_ids 前后几个样例
- 每帧 RGB/depth/seg/camera 是否成功对应
- 图像尺寸 / 深度尺寸 / mask 尺寸是否一致

---

# Phase 2：构建 ReFlow A.1 所需的层级 pair 图

实现 `pair_sampler.py`

### 6.1 clip 划分

给定训练帧序列 \(\{I_i\}_{i=1}^N\)，按固定 clip 长度 `clip_len` 划分成 K 个非重叠 clips：

```python
clips = [
    frame_ids[0:clip_len],
    frame_ids[clip_len:2*clip_len],
    ...
]
```

默认：
- `clip_len = 10`
- 最后一个 clip 不足也保留

### 6.2 keyframe 选择

每个 clip 的 **第一个 frame** 作为 keyframe，与 ReFlow A.1 保持一致：

```python
keyframes = [clip[0] for clip in clips]
```

### 6.3 coarse graph（keyframe graph）

在 keyframes 上构建 coarse graph。先实现最小可用版本：

- 相邻 keyframe 连边：\((k, k+1)\)
- 可选再加少量跨 clip 跳连边：\((k, k+2)\)

输出例如：
```python
coarse_pairs = [(kf0, kf1), (kf1, kf2), ...]
```

### 6.4 fine graph（clip 内局部图）

每个 clip 内部构建 local pairs，最小建议：

- 当前 clip 的 keyframe 与 clip 内所有其他帧连边
- 再加相邻时刻连边

例如 clip = `[f0, f1, f2, f3]`：
```python
[(f0, f1), (f0, f2), (f0, f3), (f1, f2), (f2, f3)]
```

### 6.5 目标

最终返回：
```python
{
    "clips": ...,
    "keyframes": ...,
    "coarse_pairs": ...,
    "fine_pairs_per_clip": ...,
}
```

这一步就是 ReFlow A.1 里的 **hierarchical coarse-to-fine pairing scaffold**。

---

# Phase 3：调用 MonST3R 做 pairwise 几何推理

实现 `pair_infer.py`

### 7.1 目标

给定一对 frame id `(a, b)`：

- 读取两帧 RGB
- 使用 MonST3R 现有推理能力
- 输出 pairwise 几何结果

### 7.2 期望输出结构

请统一封装成：

```python
{
    "pair": (id_a, id_b),
    "Xa_in_a": ...,        # a帧 pointmap in camera-a coordinates
    "Xb_in_a": ...,        # b帧 pointmap in camera-a coordinates
    "Ca_in_a": ...,        # confidence for a
    "Cb_in_a": ...,        # confidence for b
    "meta": {
        "image_size": ...,
        "orig_size": ...,
    }
}
```

如果 MonST3R 现有接口的命名不同，可以在封装层内做适配，但对外统一成这个格式。

### 7.3 缓存机制

pairwise 推理会较慢，请实现磁盘缓存：

```text
scene_datadir/monst3r_reflow_a1_cache/pairs/
    pair_<id_a>__<id_b>.npz
```

首次计算后缓存，后续直接读取。

### 7.4 注意事项

- 保持 MonST3R 原始预处理方式（resize / normalize / tensor layout）
- 不要私自改模型输入尺度策略，除非写成命令行参数
- 若仓库已有 pairwise cache 机制，可复用

---

# Phase 4：实现 coarse alignment（keyframe级）

实现 `coarse_align.py`

### 8.1 理论要求

这一步对应 ReFlow A.1 的：
- 在 keyframe 图 \(G_K\) 上建立 coarse globally consistent canonical structure

你的实现不需要发明新的优化器，而是：
**复用 MonST3R 现有 global alignment 逻辑，作用在 keyframe coarse_pairs 上。**

### 8.2 输入

- `keyframes`
- `coarse_pairs` 的 pairwise 结果缓存
- 可选：每个 keyframe 的 camera json 初值（如果 MonST3R 对齐支持外参初始化，可作为 anchor 或 warm-start）

### 8.3 输出

至少输出一个 `coarse_state`，包含：

```python
{
    "keyframe_ids": [...],
    "global_pointmaps": {...},   # 每个keyframe在统一canonical/world中的pointmap
    "poses": {...},              # 若可得
    "intrinsics": {...},         # 若可得
    "depths": {...},             # 若可得
}
```

### 8.4 工程要求

请优先使用 MonST3R 原有全局对齐函数。若需要新写一个轻量 wrapper，请满足：

- 输入：pairwise pointmaps + confidence + 图结构
- 输出：全局一致的 keyframe 几何
- 保持与原 MonST3R 对齐变量命名尽量一致

### 8.5 外参锚定，非常重要，必须实现

作用：
- 在 coarse alignment 中把 JSON 读出的相机外参作为初始化
- 或对全局坐标 gauge 做轻量锚定，减少漂移

注意：
- 只是作为初始化 / 正则 / gauge fixing

---

# Phase 5：实现 fine alignment（clip内细化）

实现 `fine_align.py`

### 9.1 目标

对每个 clip，在 coarse keyframe 对齐结果基础上，把 clip 内所有帧纳入统一 canonical 空间。

### 9.2 输入

- `clips`
- `fine_pairs_per_clip`
- 对应的 pairwise cache
- `coarse_state`

### 9.3 策略

每个 clip 单独处理：

1. 取该 clip 的 keyframe 作为参考
2. 用 `coarse_state` 中该 keyframe 的全局结果做初始化
3. 对 clip 内所有帧做局部对齐
4. 输出每帧在 canonical/world 下的 pointmap 或深度+pose 结果

### 9.4 输出

```python
{
    "frame_ids": [...],
    "global_pointmaps": {...},   # 所有train帧
    "poses": {...},
    "intrinsics": {...},
    "depths": {...},
}
```

### 9.5 要求

- 若 MonST3R 原生支持多帧/窗口式全局优化，优先复用
- 若不方便，允许“按 clip 独立优化 + keyframe世界坐标对齐”的方式实现基础版本
- 但最终所有帧必须落到同一个 canonical/world 坐标系中

---

# Phase 6：反投影、聚合、静动态拆分，导出点云

实现 `backproject_split.py` + `export_ply.py`

### 10.1 目标

用 fine alignment 结果导出最终 canonical 点云，并拆成静态/动态两部分。

### 10.2 两种允许的数据来源

优先级如下：

#### 方案 A（优先）
若 fine alignment 已直接得到每帧在统一 canonical/world 下的 pointmap：
- 直接使用该 pointmap

#### 方案 B
若 fine alignment 给的是 `depth + pose + intrinsics`：
- 用每帧像素反投影得到 3D 点
- 再变换到统一 canonical/world 坐标系

标准反投影：

```python
X_cam(u,v) = D(u,v) * K^{-1} [u, v, 1]^T
X_world = T_wc @ homog(X_cam)
```

### 10.3 掩码拆分

对每帧每个像素对应的 3D 点：

- `dynamic_mask[u,v] == True` -> 加入 dynamic 点集
- 否则加入 static 点集

### 10.4 过滤建议

为了提升输出质量，请加以下轻量过滤：

1. confidence 过滤（若 pointmap 对应 confidence 可取）
2. depth > 0 过滤
3. 去除 NaN / inf
4. 可选随机下采样或体素下采样
5. 可选颜色附着（从 RGB 取色）

### 10.5 输出文件

在如下目录导出：

```text
scene_datadir/monst3r_reflow_a1_outputs/
    canonical_complete.ply
    static_complete.ply
    dynamic_complete.ply
    summary.json
```

`summary.json` 至少包含：
- scene_root
- num_train_frames
- num_clips
- num_keyframes
- num_points_total
- num_points_static
- num_points_dynamic
- clip_len
- image_resize
- use_camera_anchor
- pair_cache_hits / misses

---

## 6. 命令行入口

实现：

```bash
python -m monst3r.reflow_a1.run_reflow_a1 \
    --scene_root /path/to/scene_datadir \
    --split train \
    --clip_len 10 \
    --image_size 512 \
    --use_camera_anchor
```

### 11.1 推荐参数

- `--scene_root`
- `--split`
- `--clip_len`
- `--image_size`
- `--use_camera_anchor`
- `--force_recompute_pairs`
- `--export_canonical_ply`
- `--export_static_dynamic_ply`
- `--dynamic_label_mode`
- `--voxel_downsample`
- `--max_points_per_frame`

### 11.2 运行日志必须清楚打印

每个阶段打印：
- [Phase 1] Loaded dataset ...
- [Phase 2] Built K clips / M coarse pairs / ...
- [Phase 3] Pair inference ...
- [Phase 4] Coarse alignment done
- [Phase 5] Fine alignment done
- [Phase 6] Exported static/dynamic point clouds ...

不要只打印含糊日志。

---

## 7. 与 ReFlow A.1 对齐时必须满足的“理论解释”要求

请在代码注释里明确写出以下几点，帮助后续阅读者理解：

### 12.1 为什么不是直接全视频两两配对
因为全连边是 \(O(N^2)\)，对视频太贵，而且远距离视角共视不足。

### 12.2 为什么要分 keyframe coarse + clip fine
- keyframe coarse 负责全局骨架一致性
- clip fine 负责局部时序补全与细化
- 这就是 ReFlow A.1 的 coarse-to-fine canonical construction

### 12.3 为什么最后要按 segmentation 拆 static / dynamic
因为 ReFlow 后续要对静态和动态做不同建模与不同 motion supervision，
所以 A.1 初始化阶段就要保留：
- `P3D,stat`
- `P3D,dyn`

### 12.4 为什么这里仍然以 MonST3R 为底层几何模块
因为 ReFlow A.1 的 “geometry foundation model + hierarchical alignment” 可以直接落到 MonST3R 的：
- pairwise pointmap regression
- confidence-aware alignment
- multi-frame global consistency

上面。

---

## 8. 最低交付标准（必须做到）

本任务完成后，至少必须满足以下结果：

1. 能从 `dataset.json` 正确读取 `train_ids`
2. 能基于这些帧构建 clips / keyframes / coarse_pairs / fine_pairs
3. 能调用 MonST3R 对图像对输出 pointmap + confidence
4. 能完成一个基础可运行的 coarse-to-fine alignment 流程
5. 能导出：
   - `static_complete.ply`
   - `dynamic_complete.ply`
6. 代码中对每一阶段职责有清楚注释
7. 不破坏 MonST3R 原始主流程

---

## 9. 可以接受的简化，但必须说明

以下简化是允许的，但要在代码注释或 README_impl.md 中写明：

1. **coarse graph 只连相邻 keyframes**
   - 可以，属于基础版
2. **fine alignment 只做 clip 内局部图**
   - 可以
3. **dynamic mask 直接来自 segmentation 非零区域**
   - 可以
4. **tracks 暂时不参与 A.1**
   - 可以读取但先不用
5. **不做额外 motion mask refinement**
   - 可以
6. **不复现 ReFlow 论文里所有全局优化细节**
   - 可以，但必须保留分层 A.1 主结构

---

## 10. 不要做的事情

1. 不要在本任务里实现 A.2 tri-plane / static-dynamic Gaussian decoder
2. 不要接 4DGS 训练
3. 不要引入外部光流模型
4. 不要把 segmentation 改成复杂的 learned motion mask
5. 不要把本任务做成只导出每帧局部点云、却没有统一 canonical 空间
6. 不要把代码写成一次性 notebook 风格脚本

---

## 11. 你最终应该提交的内容（对代码仓库而言）

至少应包含：

- `monst3r/reflow_a1/` 全部实现
- 一个可运行的 `run_reflow_a1.py`
- 必要注释
- 若有需要，额外补一个简短 `README_impl.md`

但请保持改动尽量聚焦，不要大面积重构原仓库。

---

## 12. 给 Codex 的执行优先级

请按以下顺序工作，不要一开始就把所有文件同时改乱：

1. 先实现 `dataset_scene.py`
2. 再实现 `pair_sampler.py`
3. 再接 `pair_infer.py`
4. 先跑通 keyframe coarse alignment
5. 再补 clip fine alignment
6. 最后实现 point cloud split + ply export
7. 最后统一整理命令行入口

每阶段完成后，优先保证“能跑通、接口稳定、输出可检查”。

---

## 13. 补充说明：关于输出“静态和动态都找全”的真实含义

这里的“找全”不是数学上绝对完整，而是工程上尽量实现：

- 静态区域：利用多帧/多 keyframe 融合提升覆盖率
- 动态区域：不要像传统 COLMAP 初始化那样直接丢掉
- 最终 dynamic 点云应来自参与 canonical construction 后的动态区域 3D 点聚合，而不是只取单帧 mask 贴图伪造

因此，动态点云至少应满足：
- 来源于 train_ids 全序列
- 在统一 canonical/world 坐标中聚合
- 不只是单帧的局部快照

---

## 14. 最后提醒

这个任务的本质是：

**在 MonST3R 仓库里，做一个“ReFlow A.1 风格的最小可运行初始化子模块”。**

不是重新训练 MonST3R，
不是复现完整 ReFlow，
而是把：

- 数据读取
- 分层 pair 组织
- MonST3R pairwise 几何
- coarse-to-fine alignment
- static/dynamic point cloud export

这条链条打通。

请优先保证结构清晰、接口合理、便于继续扩展。
