# 面向 Codex 的实现提纲：将 ReFlow A.1 的 Fine 阶段改为“局部细化但始终受全局 canonical 约束”

## 文档目标

本提纲用于指导对现有 ReFlow A.1 风格初始化代码进行结构性改造，使 Fine 阶段不再是“每个 clip 各自优化、导出、最后再 merge”的后处理式流程，而改为：

- 在 **coarse 阶段已经建立的全局 canonical / keyframe 参考系** 下进行局部 refinement；
- Fine 阶段所有 clip 都持续写入同一个全局 canonical 表示；
- 静态背景在 Fine 过程中就被统一融合，而不是最后依赖 merge heuristic 收尾；
- 动态区域不直接并入单一静态 canonical，而是进入单独的时序缓冲结构，为后续 A.2 的 static/dynamic disentanglement 做准备。

该提纲只描述工程结构、模块职责、损失设计与数据流，不包含源码实现。

---

# 一、数据结构

## 1.1 全局状态结构

Fine 阶段开始前，默认 coarse 阶段已经输出如下全局状态：

### A. `GlobalCoarseState`
用于保存 coarse 阶段的全局参考系与 keyframe 对齐结果。

建议包含：

- `keyframe_ids`
  - coarse 阶段选中的 keyframe 索引列表。
- `Kpose`
  - keyframe 的相机外参。
- `Kintr`
  - keyframe 的相机内参。
- `Kdepth` / `Kpointmap`
  - keyframe 的深度图或 pointmap。
- `static_masks_keyframes`
  - keyframe 对应的静态区域 mask。
- `dynamic_masks_keyframes`
  - keyframe 对应的动态区域 mask。
- `coarse_confidence`
  - coarse 阶段中各 keyframe/像素/点的置信度。
- `scene_transform`
  - 统一 canonical 坐标系到训练坐标系的变换（若有）。

### B. `GlobalCanonicalRegistry`
用于管理整个 Fine 阶段持续维护的统一 canonical 表示。

建议分成两个子模块：

- `GlobalStaticMap`
  - 保存静态主表面。
  - 可选实现形式：voxel-hash、surfel-map、point-bucket、稀疏 TSDF 风格容器。
- `GlobalAnchorSet`
  - 保存跨 clip 共享的静态锚点。
  - 每个锚点应有全局唯一 ID。
  - 每个锚点应记录：
    - 3D 位置
    - 法向（可选）
    - 颜色统计（可选）
    - 置信度
    - 可见的 frame/keyframe 列表
    - 来源类型（COLMAP 稀疏点 / coarse pointmap / MonST3R 高置信静态点）

### C. `GlobalDynamicBuffer`
用于保存动态区域的时间相关候选几何，而不是直接写进静态 canonical。

建议包含：

- `per_frame_dynamic_points`
  - 每帧动态候选点。
- `per_clip_dynamic_points`
  - 每个 clip 聚合后的动态候选点。
- `dynamic_confidence`
  - 动态点置信度。
- `dynamic_visibility`
  - 动态点被哪些帧看到。
- `dynamic_reference_candidates`
  - 供后续选取 reference frame 的候选信息。
- `track_ids`（若后续使用轨迹）
  - 可选，用于维护跨帧动态点关联。

---

## 1.2 clip 级状态结构

每个 clip 在 Fine 阶段不应拥有一份将被最终导出的独立 canonical，而应只拥有“局部优化状态”。

建议定义：`ClipFineState`

包含：

- `clip_id`
- `frame_ids`
- `local_pairs`
  - clip 内参与对齐的帧对。
- `init_pose`
  - 从 coarse / 邻近 keyframe 初始化得到的局部 pose。
- `init_intr`
  - 初始化的 intrinsics。
- `init_depth` / `init_pointmaps`
  - 初始化深度或 pointmap。
- `static_masks`
- `dynamic_masks`
- `confidence_maps`
- `overlap_with_prev_clip`
  - 与前一 clip 的重叠帧。
- `overlap_with_next_clip`
  - 与后一 clip 的重叠帧。
- `visible_global_anchors`
  - 本 clip 可见的全局静态锚点 ID 列表。
- `fine_variables`
  - Fine 阶段真正要更新的局部变量。

注意：

- `ClipFineState` 的输出不应是“导出的点云文件”；
- `ClipFineState` 的输出应是：
  - 更新后的局部相机/深度变量；
  - 新增或修正后的静态观测；
  - 新增或修正后的动态候选；
  - 对全局容器的更新操作。

---

## 1.3 重叠关系结构

为支持 cross-clip consistency，建议显式维护：`ClipOverlapGraph`

包含：

- `adjacent_clip_pairs`
- `shared_frame_ids`
- `shared_anchor_ids`
- `shared_static_regions`
- `shared_dynamic_regions`（可选）

其作用是：

- 避免 Fine 只在 clip 内自洽；
- 把跨 clip 的 seam 和 overlap 显式建模；
- 为损失函数中的 overlap consistency 提供索引。

---

# 二、损失函数

## 2.1 总体设计原则

Fine 阶段建议不再仅使用“clip 内局部对齐损失”，而应由五类损失共同组成：

1. `L_local_align`：clip 内局部几何一致性
2. `L_anchor`：对 coarse keyframe/global canonical 的锚定约束
3. `L_shared_anchor`：跨 clip 的共享静态锚点一致性
4. `L_overlap_consistency`：overlap 帧/区域的一致性约束
5. `L_static_fusion_regularization`：写入全局静态表面时的主表面约束

动态区域另行进入 buffer，不直接用静态融合损失约束。

---

## 2.2 `L_local_align`：clip 内局部对齐损失

### 目标

保持论文 A.1 Fine 阶段的基本形式：

- 在 clip 内 temporally close 的帧对之间建立 pointmap/depth 一致性；
- 强化局部共视下的几何对齐质量；
- 为缺失区域和非 keyframe 细节补全提供观测。

### 输入

- clip 内帧对的 pointmap / depth / confidence
- 局部 pose / intrinsics
- 静态 mask / 动态 mask

### 建议实现要点

- 静态区域权重大于动态区域；
- 动态区域可选择：
  - 降权参与；
  - 或在 Fine 的静态融合部分中暂时剔除，只进入动态候选缓冲；
- 仅在高 confidence 的局部对应上累积几何误差。

### 输出作用

- 提升 clip 内局部点图质量；
- 但不能单独决定最终全局 canonical。

---

## 2.3 `L_anchor`：全局 coarse 锚定损失

### 目标

这是当前实现最需要新增的部分之一。

它的作用不是完全冻结 clip 内变量，而是保证 clip 内 refinement **始终围绕 coarse 已建立的全局参考系收敛**，避免局部 gauge drift。

### 约束对象

- 局部 pose 不能偏离 coarse / keyframe 初始化过多；
- 局部 intrinsics 不能漂移；
- 静态主表面不能显著偏离 coarse 静态骨架。

### 建议形式

可对以下量加约束：

- pose 平移偏差
- pose 旋转偏差
- intrinsics 偏差
- 静态深度 / 静态点位置对 coarse 初始化的偏差

### 使用策略

- 对接近 keyframe 的帧施加强锚定；
- 对远离 keyframe 的帧施加较弱锚定；
- 对动态 mask 区域不做强几何锚定。

### 目的

- 避免 clip 自己收敛成一套局部 canonical；
- 避免 Fine 阶段把 global consistency 推给最后的 merge。

---

## 2.4 `L_shared_anchor`：共享静态锚点一致性损失

### 目标

该项用于显式建立跨 clip 的几何耦合。

### 核心思想

若 clip A 和 clip B 都观测到同一批静态锚点，则两者对这些锚点的重建应一致。

### 约束对象

- 同一全局静态锚点在不同 clip 中恢复得到的 3D 位置
- 或这些锚点的投影残差
- 或锚点附近局部 patch 的几何一致性

### 适用范围

- 优先只用于静态区域；
- 动态区域不建议用该项做强约束。

### 作用

- 减少静态 seam；
- 防止不同 clip 在背景上各自长出一层表面；
- 提升跨 clip 的世界坐标一致性。

---

## 2.5 `L_overlap_consistency`：重叠区域一致性损失

### 目标

若相邻 clips 共享一部分帧或时间区间，则应强制这些 overlap 区域的解释一致。

### 可约束内容

- overlap 帧的 depth / pointmap 一致性
- overlap 静态区域的表面位置一致性
- overlap 区域的法向一致性
- overlap 区域的可见性一致性
- 相机连续性约束

### 作用

- 直接削弱 boundary seam；
- 防止 clip A 和 clip B 在接缝处各自收敛到不同局部极小值；
- 使 merge 从“硬拼接”变为“软一致”。

---

## 2.6 `L_static_fusion_regularization`：静态融合正则

### 目标

该项用于约束向 `GlobalStaticMap` 写入时的行为，避免静态背景在 Fine 阶段逐 clip 叠层。

### 核心思想

当前 clip 提供的是对全局静态主表面的**观测更新**，而不是独立生成一层新的静态表面。

### 可包含的规则

- 若当前观测与已有静态主表面位置接近，则执行融合更新，而不是新建表面；
- 若同一 camera ray / voxel 中已存在更高置信主表面，则新观测应被抑制；
- 若新旧观测冲突明显，则仅保留高置信且更满足多视图一致性的那一个。

### 作用

- 避免静态重复表面；
- 避免后续 GS 初始化时出现背景 layering；
- 把“merge”提前到 Fine 优化过程中隐式完成。

---

# 三、clip overlap

## 3.1 为什么必须引入 overlap

当前“无重叠 clip + 各自优化 + 最后 merge”的结构中，跨 clip 基本没有直接约束，因此：

- static seam 难以消除；
- dynamic canonical 容易重复；
- 最终效果过度依赖 merge heuristic。

引入 overlap 的目标是让相邻 clips 在优化阶段就共享一部分观测与约束。

---

## 3.2 建议的 clip 切分方式

### 当前问题

若 clip 完全不重叠：

- clip 与 clip 之间只通过 coarse 阶段的弱锚定相连；
- local fine 得不到 seam 区域的显式约束。

### 建议改法

使用重叠切分：

- 相邻 clips 共享 20%～30% 的帧；
- 或至少共享 2～4 帧边界帧；
- 对于复杂动态片段，可适当增加 overlap 比例。

### 切分示意

例如：

- clip 1: frames 0–15
- clip 2: frames 12–27
- clip 3: frames 24–39

这样可保证：

- 边界处有共享观测；
- 可直接构造 overlap consistency loss；
- static anchors 可以自然跨 clip 传递。

---

## 3.3 overlap 的工程用途

### A. 用于共享静态锚点

在 overlap 帧里提取静态高置信 anchor，作为 clip 间的 shared anchors。

### B. 用于 seam 诊断

通过 overlap 区域中的几何差异直接检测：

- 静态错层
- 局部尺度漂移
- 深度偏置

### C. 用于更新全局容器

overlap 区域是最适合观察“当前 clip 是否与已写入的全局静态表面一致”的地方。

---

## 3.4 overlap 的调度建议

### 顺序更新

- 按 clip 时间顺序更新；
- 新 clip 优化前先读取前一 clip 已写入的 `GlobalStaticMap` 与 `GlobalAnchorSet`。

### 交替 refinement

若资源允许，可做二轮 refinement：

- 第一轮：顺序写入全局 canonical
- 第二轮：回头利用更完整的全局锚点重新细化前面 clips

这样更接近“共享全局 canonical 上的局部 refinement”。

---

# 四、global static fusion

## 4.1 设计目标

静态背景不应被每个 clip 重复恢复成多层壳，而应在 Fine 阶段持续融合成统一主表面。

### 原则

- Fine 阶段对静态区域的输出是“对全局静态表面的观测更新”；
- 不是“clip 自己生成一份最终静态点云”。

---

## 4.2 `GlobalStaticMap` 的建议职责

### 必须支持

- 查询某个位置附近是否已存在静态主表面；
- 根据新观测更新已有表面；
- 抑制重复表面写入；
- 维护置信度、法向、颜色等统计；
- 提供 anchor 导出给后续 clips。

### 可选实现方向

- voxel-hash + voxel 内主表面代表点
- surfel map
- point bucket + radius-based fusion
- 稀疏 TSDF 风格容器

对当前任务而言，重点不是选哪种容器，而是要明确：

> `GlobalStaticMap` 必须是 Fine 阶段优化循环中的“活状态”，而不是最后 merge 的输入缓存。

---

## 4.3 静态写入流程建议

每个 clip Fine 完成一个优化 step 或一个 clip 结束后，对静态区域执行：

1. 从当前 clip 的高置信静态点中提取候选更新；
2. 查询这些点在 `GlobalStaticMap` 中是否已有对应主表面；
3. 按置信度、多视图一致性、局部几何冲突情况决定：
   - 更新已有表面；
   - 或新增主表面；
   - 或丢弃冲突点；
4. 将被确认的稳定静态点加入 `GlobalAnchorSet`；
5. 将冲突严重或低置信的点标记为待观察，不直接写入主表面。

---

## 4.4 静态融合中的关键规则

### 规则 A：主表面优先

同一局部空间中只保留一个静态主表面解释，避免 layering。

### 规则 B：高置信优先

当新旧表面冲突时，优先保留：

- 置信度更高
- 可见帧更多
- 与 coarse/global anchors 更一致
- 与邻近 clip overlap 更一致

### 规则 C：边界谨慎更新

靠近 dynamic/static 边界的区域容易污染，应：

- 提高写入阈值；
- 或仅在多帧稳定一致时写入主表面。

### 规则 D：静态共享而非重复导出

后续 clip 若再次观测到同一静态表面，优先更新已有项，而非新增新点。

---

## 4.5 为什么这一步对 GS 初始化尤其重要

若静态背景在 Fine 阶段没有被统一融合，而只是最后简单 merge，则极易出现：

- 多层墙面 / 地面 /桌面
- 半透明叠层
- 深度排序不稳
- densify / prune 被背景错层带偏

因此，global static fusion 不是附加功能，而应作为 Fine 阶段的核心结构。

---

# 五、dynamic buffer

## 5.1 设计目标

动态区域不能像静态区域那样直接被融合成单一 canonical 主表面。

否则会出现：

- 不同时间的动态形态被错误平均；
- 多个 clip 对同一动态物体的不同解释被焊接到一起；
- 后续 A.2 的动态 Gaussian 初始化反而更乱。

因此，动态区域应单独进入 `GlobalDynamicBuffer`。

---

## 5.2 `GlobalDynamicBuffer` 的职责

### A. 保留时序信息

动态候选必须按时间、帧、clip 或轨迹组织，而不是统一无脑合并。

### B. 保留多候选解释

对于动态区域，允许多个时间片存在不同几何状态，等待后续 reference selection 或 motion-aware initialization 处理。

### C. 服务于后续 A.2 初始化

在进入 static/dynamic 分离建模时，`GlobalDynamicBuffer` 应支持：

- 选取动态区域 reference frame
- 汇总某个动态物体的高 coverage 时刻
- 估计动态点出现频率与可见性
- 为 dynamic Gaussian 初始化提供点云候选

---

## 5.3 动态写入流程建议

对每个 clip：

1. 根据 dynamic mask 提取动态候选点；
2. 保留对应时间戳 / frame id / clip id；
3. 记录置信度、来源帧、局部可见性；
4. 不写入 `GlobalStaticMap`；
5. 若后续有轨迹或对应关系，可建立 track-level grouping；
6. 在 clip 间 overlap 里仅做轻量时序一致性检查，不强行做静态式融合。

---

## 5.4 动态缓冲中的可选组织方式

### 方式 A：按帧组织

适合最简单实现。

- `frame_id -> dynamic point candidates`

优点：简单直接。  
缺点：跨帧关联弱。

### 方式 B：按 clip 组织

- `clip_id -> aggregated dynamic candidates`

优点：便于局部处理。  
缺点：clip 间重复较多。

### 方式 C：按轨迹 / 实例组织

- `track_id / object_id -> time-indexed dynamic geometry`

优点：最适合后续动态建模。  
缺点：实现成本最高。

当前建议：

- 第一版先按帧 + 按 clip 双层组织；
- 后续若加入 tracks，再升级为轨迹组织。

---

## 5.5 动态 buffer 与全局锚点的关系

动态区域不直接进入静态锚点集，但可以：

- 在边界区域引用静态 anchor 作为局部参考；
- 在后续 reference frame selection 中，借助静态背景的稳定坐标来确定动态点的相对位置。

换言之：

- 动态不并入静态 canonical；
- 但动态仍然生活在 coarse/global canonical 定义的统一坐标系中。

---

# 六、建议的整体数据流

## 阶段 0：coarse 输出

得到：

- `GlobalCoarseState`
- 初始 `GlobalAnchorSet`
- 初始 `GlobalStaticMap`（可由 coarse keyframes 静态区域构建）

## 阶段 1：构建 overlap clips

- 使用重叠切分替代完全独立切分；
- 建立 `ClipOverlapGraph`。

## 阶段 2：逐 clip Fine

每个 clip：

1. 读取 coarse/global 初始化；
2. 查询可见静态 anchors；
3. 优化 `L_local_align + L_anchor + L_shared_anchor + L_overlap_consistency`；
4. 将静态高置信候选写入 `GlobalStaticMap`；
5. 将动态候选写入 `GlobalDynamicBuffer`；
6. 更新 `GlobalAnchorSet`。

## 阶段 3：可选第二轮 refinement

- 使用已经更完整的 `GlobalStaticMap` 和 `GlobalAnchorSet` 再跑一遍 clips；
- 专门修复 seam 与 overlap 不一致。

## 阶段 4：输出给 A.2 / 4DGS 初始化

- 从 `GlobalStaticMap` 导出统一静态点云；
- 从 `GlobalDynamicBuffer` 选择动态 reference / 聚合动态候选；
- 不再使用“clip 导出点云后最后 merge”的流程。

---

# 七、对 Codex 的实现要求总结

请严格按以下原则修改 Fine 阶段：

1. **不要再让每个 clip 导出独立的最终 canonical 点云。**
2. **Fine 必须始终在 coarse 已建立的全局 canonical 参考下进行。**
3. **必须引入跨 clip 的共享静态锚点机制。**
4. **必须引入 overlap clips，而不是完全独立切分。**
5. **静态背景必须写入统一的 `GlobalStaticMap`，而不是最后再 merge。**
6. **动态区域必须进入独立 `GlobalDynamicBuffer`，不能直接做静态式融合。**
7. **merge 不再作为最后的核心步骤，而应被前移为 Fine 优化过程中的隐式融合与一致性维护。**

---

# 八、最终预期效果

完成上述改造后，Fine 阶段应从当前的：

- 局部 self-contained 优化
- 最后再 merge
- 结果高度依赖 merge heuristic

转变为：

- 在统一 canonical 中做局部 refinement
- 静态背景持续全局融合
- 动态区域按时间缓存
- overlap 区域在优化期就被约束一致
- 输出直接服务于 A.2 的 static/dynamic 分离初始化

其核心目标不是“让每个 clip 单独更好”，而是：

> **让所有 clip 对同一个 global canonical 的贡献在优化过程中就被统一，而不是留到最后再靠后处理拼接。**
