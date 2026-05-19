# 嵌套 Cluster — GQA 支持设计 note

> 状态：设计已定稿，未实施。GQA kernel 已写好
> （`kernels/flash_attention_gqa_min.py`），作为端到端验证用例。

## 1. 背景与目标

GQA（grouped-query attention）里 KV head 数 < Q head 数：`group_size`
个 Q head 共享一个 KV head，`kv_head_count = head_count // group_size`。

Q head `by` 映射到 KV head 用 **`by % kv_head_count`**（取模，不是
floordiv）：

* PLENA ISA **没有整数除法 op**，只有取模 → `by // group_size` 无法
  lower 到硬件。
* `%` 给出 *interleaved* 的 group 布局：Q head
  `h, h+kv_head_count, h+2*kv_head_count, ...` 共享 KV head `h`。
* 调用方需把 Q head 在 HBM 里按 interleaved-by-KV-head 排布。

这要求 lane 轴 `by` 被切成 **两层嵌套 cluster**，而当前体系只支持
**单层 cluster**。

### 目标层级

`by`（head_count 个）被嵌套切分，产出三个轴：

| 轴       | 角色                              | kind      |
|----------|-----------------------------------|-----------|
| `by_o`   | 外层 cluster 的 grid（number）    | BLOCK_IDX |
| `by_i_o` | 外层 cluster 的 phase；同时是内层 cluster 的 grid（number） | CLUSTER |
| `by_i_i` | 内层 cluster 的 phase             | CLUSTER   |

`by_i_o` 的"双重身份"是**循环嵌套位置性**的（它在 `by_i_o`-cluster
体内、`by_i_i`-cluster 体外），不编码进 kind —— 它和 `by_i_i` 都是
CLUSTER。

### op 的层级归属（GQA kernel）

| op                          | 嵌套在        | 原因 |
|-----------------------------|---------------|------|
| K/V DMA（`by % kv_head_count`） | 只被 `by_o`（外层）| KV head 只随外层变，group 内共享 |
| Q DMA / BTMM（`Q@K^T`）     | 被 `by_i_o`（外层 cluster）| Q 随完整 `by`，要覆盖整个 group |
| per-lane softmax / 标量 op  | 被 `by_i_i`（内层）| 每个 Q head 各自的 softmax 状态 |

`Q_sh` 的 s 维：`4 → 8`（hardware_lane_count × 第二层 cluster）。

## 2. 核心设计决定

| 决定 | 选择 | 理由 |
|------|------|------|
| `BufferDef.cluster_dim` | **不动**，保持单值 `int` | 走"乘积单维"，砍掉 `to_plena` 等 40+/70+ 处级联改动 — 最大的难度来源被绕过 |
| buffer growth | **A：乘积单维** | `Q_sh [64,16] → [8,64,16]`，`cluster_dim=0`。两层之分不进 buffer 形状 |
| `by_i_o` kind | CLUSTER | 双重身份靠循环位置表达，不进对象状态 |
| op 层级归属判定 | **async_wrap** 算，不是 fuse | async_wrap 跑在 view 之前，看到的索引是干净的 `by % N`；它本就管同步域 |
| fuse 角色 | 只做结构折叠 + 提升 | 不做语义推断，只读 async_wrap 标好的位图 |

**"cluster 列表"是这个方案的本质** —— 把到处隐含的单层 cluster 变成
一等公民的有序列表：

* 编译期配置：`cluster_counts` 每条目变成 `List[int]`（外→内）。
* 运行结构：fuse 的 `cluster_stack` / `MultiLaneOp.cluster_axis_names`。
* op 归属位图：`Async.trivial_levels` / `MultiLaneOp` 上的同名字段。

只有 `cluster_dim` 不列表化（故意）。

## 3. 改动方案 — 四节

依赖顺序：IR 字段 → split → async_wrap → view → fuse。

### 节 0：IR 字段

`ir.py`：

* `Async` 新增 `trivial_levels: List[bool]`（外→内，每层 cluster 一个
  bool；`True` = 该层对这个 op trivial）。
* `MultiLaneOp` 新增同名字段（从 `Async` 透传）。
* `MidFunc.cluster_counts` 语义变更：`List[int]` → `List[List[int]]`
  （每个 lane 轴一个内层列表）。

### 节 1：split — 嵌套切分

文件：`passes/split.py`

* **`cluster_counts` 嵌套化**：每条目 `int → List[int]`（外→内）。
  GQA：`lane_axes=["by"]`, `cluster_counts=[[kv_head_count, group_size]]`。
  单层 kernel 写 `[[4]]`（或保留 `int` 自动包成单元素 list 兼容）。
* **`_split_once(axis, count)`**：抽出一个纯函数，把一个轴切成
  `(number, phase)` 对，**不依赖** `lane_axes` / kind 检查。
* **递归切**：`_split_or_walk_parallel` 命中 lane 轴时，拿到该轴的
  count 列表，对列表 `fold`：第一次输入用户的 BLOCK_IDX 轴，后续每次
  输入上一次产出的 phase 轴。`splittable_kinds` 闸门**只在最外层入口
  判一次**；内层对 phase 轴的递归调用直接走 `_split_once`，不过 kind
  检查（因为 phase 轴是 CLUSTER，不在 splittable_kinds 里）。
  嵌套结构：`by_o.body=[by_i_o]`, `by_i_o.body=[by_i_i]`,
  `by_i_i.body=inner_body`。
* **buffer growth**：现有 line 376-383 的乘积逻辑**几乎不动**，只需
  flatten 一层嵌套列表：

  ```python
  cluster_total = 1
  for axis_counts in cluster_counts:      # axis_counts = [kv_head_count, group_size]
      for c in axis_counts:
          cluster_total *= c
  grown[buf.name] = _grow_buffer(buf, cluster_total)
  ```

  `_grow_buffer` 本身一行不改：仍 `shape=[cluster]+shape`,
  `cluster_dim=0`。`Q_sh → [head_count,64,16]`。

### 节 2：fuse — 嵌套 + DMA 提升

文件：`passes/fuse.py`

* **`cluster_stack` 递归 push — 零改动**。`_walk` 每进一层 CLUSTER 就
  `cluster_stack + [_ClusterAxis(...)]`；嵌套后走到内层体内自然是
  `[by_i_o_axis, by_i_i_axis]` 两项。
* **`_fuse_async` 读 `trivial_levels`**：不再自己扫索引反推 trivial。
  直接读 `Async.trivial_levels`，透传到 `MultiLaneOp`。`MultiLaneOp`
  仍带**全部层** `cluster_axis_names` + `trivial_levels` 位图。
  `dim_map` 维持 `[cdim]*n_axes`（buffer 走 A，两层物理维都是 0）。
* **提升**：`_walk` 处理 CLUSTER 节点时，收集 body 里"内层全 trivial"
  的 MultiLaneOp，移到该 cluster 轴的**兄弟位置（前面）**。K/V DMA
  因此只被 `by_i_o` 包着。
  **关键约束**：K/V DMA 在 `for kv_block` 内，提升出内层 cluster 但
  **不能**提出 `kv_block` 循环（每个 kv block 的 K/V 不同）。提升落点
  是「最内的非 trivial 包裹」—— 内层 cluster 之外、`kv_block` 之内。
* **纪律**：fuse **不碰索引**。识别 `by % N` 只在 async_wrap 做一次
  （趁索引干净）。`_match_lane_composite` 维持原状只认 lane 形式。
  不要在 fuse 里 match view 改写过的脏 mod 表达式。

### 节 3：async_wrap — 标每层 trivial 位图

文件：`passes/async_wrap.py`

* **`_walk` 携带 cluster 轴列表**：`in_cluster: bool` →
  `cluster_axes: List[ParallelAxis]`（外→内的 CLUSTER 轴）。进 CLUSTER
  节点时 append。
* **包 Async 时算位图**：async_wrap **不碰索引**（现状 line 25-27），
  所以此刻 K/V DMA 的索引还是干净的 `by % kv_head_count`，Q/BTMM 是
  裸 `by`。对每个 can_async op、对 `cluster_axes` 每一层 `ax`：
  * 索引里 `by` 出现在 `mod(by, N)` 内 → 只依赖外层 → 内层 trivial。
  * 索引里 `by` 裸用 / 其它形态 → 两层都非 trivial。
  * 产出 `trivial_levels`：K/V DMA = `[False, True]`,
    BTMM/softmax = `[False, False]`。
* `trivial_levels` 写到 `Async` 上，随后透传到 `MultiLaneOp`（fuse）。

### 节 4：view — ctx 栈化

文件：`passes/view.py`

**这是收尾里最实的一块** —— view 不是零改动。

* 现状 `_walk`（line 451）进 CLUSTER 节点造 `new_ctx` **替换**传入的
  `ctx`（line 505）。嵌套两层时外层 ctx 被丢弃 → 出错。
* **`_ClusterCtx` 单值 → ctx 栈**：`List[_ClusterCtx]`（外→内）。进
  CLUSTER 节点时 append 而非替换。
* **`_rewrite_lane_ref`**（line 211）：`new_indices = [ctx.phase_var] +
  indices` 只 prepend 一个 phase。改成 prepend 栈里**所有层**的
  phase（non-global buffer `Q_sh` 已 grow 成 `[8,64,16]`，需两个 phase
  索引才对得上 rank）。
* **`_subst_lane_var`**（line 122）：只认 `ctx.original_var` 一个，把
  `by` 换成单层 `phase+number*count`。改成把 `by` 换成**两层合成式**。
  递归不挑 op（line 134-138），`by % N` 的 `mod` 节点会被下钻，里面的
  `by` 被替换 —— 这点本就成立。
* `trivial_levels` 位图在 view **之前**就由 async_wrap 定死，view 改的
  是索引的值、不碰位图 —— 位图安全。

## 4. 改动总览表

| 节 | pass / 文件 | 改动量 | 说明 |
|----|-------------|--------|------|
| 0 | `ir.py` | 小 | 2 个新字段 + `cluster_counts` 类型 |
| 1 | `split.py` | 中 | `_split_once` + fold 递归；乘积 flatten 一层 |
| 3 | `async_wrap.py` | 中 | `_walk` 带轴列表；算 `trivial_levels` |
| 2 | `fuse.py` | 中 | `cluster_stack` 零改；`_fuse_async` 读位图 + 提升落点判断 |
| 4 | `view.py` | 中 | `_ClusterCtx` 栈化；多层 prepend；两层合成替换 |

**不动**：`BufferDef.cluster_dim`（保持单值 int），`to_plena.py`
（70+ 处读 `cluster_dim`），`_grow_buffer`（一行不改）。

## 5. 风险点

* **空 cluster body**：提升 K/V DMA 出内层后，若某 cluster body 所有 op
  都被提走 → body 空。`distribute_cluster` / `to_plena` 应 graceful
  容忍空 `ParallelAxis.body`（生成空循环或跳过），不报 unhandled。
  **GQA kernel 实际跑不到**（softmax 链一定留在最内层），标记为风险
  但不阻塞主线。
* **`trivial_levels` 透传**：位图从 async_wrap 生成、穿过 view、到 fuse
  消费 —— 每个 pass 重建节点时都要记得透传，中间不能丢。
* **fuse 提升落点**：「提出内层 cluster 但不出 `kv_block` 循环」是方案
  里最容易写错的一处，需要细心。

## 6. 难度评估

中等偏上，不是大重构。约 **3–5 天实现 + 2–3 天调试**（嵌套 cluster
第一次跑，大概率有 rank / 索引对不齐要排）。

最值钱的判断：走"乘积单维"绕开 `cluster_dim` 级联，把"40+ 处改动"
降到"4 个 pass 局部改 + 1 个 IR 字段"。

GQA kernel 已就绪，是现成的端到端验证用例。

## 7. 验证

`kernels/flash_attention_gqa_min.py`：

* `make_flash_attention_gqa_min(head_count=8, group_size=2, ...)` →
  `kv_head_count=4`。
* `group_size=1` 退化成纯 MHA，应与 `flash_attention_min` 输出一致 —
  回归保护。
