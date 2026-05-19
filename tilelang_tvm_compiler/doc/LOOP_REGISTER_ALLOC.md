# 循环变量寄存器分配 — HLIR liveness pass 设计 note

> 状态：设计定稿，未实施。

## 1. 背景与问题

当前 GP 寄存器分配**全部在 emit 阶段边走边分**，散在三处：

* `isa_pass._emit_for` — 循环的 `gp_loop`（硬件计数器）+ idx。
* `expr_materializer` — 物化 `tir.PrimExpr` 的临时 GP。
* `isa_emitter.emit_*`（40+ 个）— 每条指令的 scratch GP。

三者**抢同一个 16-GP 池**（gp0 保留），靠 `pin_gp` / auto-spill 临时协调。后果：

* **Bug 2 类**：caller（如 `_resolve_offset`）对 materializer 返回的寄存器
  做 `pin/unpin/free`，而该寄存器可能是 per-op idx 缓存拥有的 →
  双重 pin、提前 unpin、潜在 double-free。根因是「长生命周期的循环
  变量寄存器」和「短生命周期的临时值」在同一个裸池里没有边界。
* **深嵌套 GP 耗尽**：每层 C_LOOP 占 `gp_loop`(+idx)，五层嵌套 +
  matmul 的 7 个 scratch → 15 个 GP 不够，`RegisterExhausted`。

## 2. 为什么不做「一个 pass 全包」

HLIR liveness **看不到** emit 阶段的临时 GP 需求：

* `expr_materializer` 物化一个 PrimExpr 树要几个临时 GP，取决于树形
  和遍历顺序 —— 是 emit 时才定的。
* `emit_matmul_general` 内部 7 个 scratch、`emit_row_operation` 5 个 …
  这些是「一个 HLIR leaf op 降成几十条 ISA」时内部的事，HLIR 上不可见。

所以 HLIR 上能精确分析的，只有**显式写出来的循环变量**（`for` op 的
`loop_var`）—— 它的 def / use / kill 在 HLIR 上一清二楚。

## 3. 方案 — 分层

| 层 | 谁分配 | 何时 |
|----|--------|------|
| **循环变量寄存器**（`gp_loop` + idx，每层 C_LOOP 一组） | 新的 **HLIR liveness pass** —— 全局算活跃区间，提前定死 GP 号 | address_alloc 之后、isa_pass 之前 |
| **op 内部临时值**（matmul scratch、materializer 中间值） | emit 阶段局部分配器 —— **不变**，但 GP 池被缩小 | isa_pass 内，照旧 |

**关键机制**：`RegisterAllocator.__init__` 已有 `gp_reserved` 参数
（register_alloc.py:76）。liveness pass 算出「同一时刻最多 N 个循环
变量寄存器活跃」，并给每个循环变量定一个具体 GP 号；这些 GP 号进
`gp_reserved`。emit 阶段的 `allocate_gp` 从此**拿不到**这些 GP —— 循环
变量寄存器和临时值物理隔离，不再同池抢。

`_emit_for` 不再 `allocate_gp` 循环寄存器，改成**读 pass 标在 op 上的
GP 号**。emit 阶段的临时分配只在「剩余 GP」里跑。

## 4. liveness 分析

HLIR 是线性 op 流（`HLIRModule.ops: List[Op]`），`for` op 带 `body`
子列表。循环变量的生命周期由结构决定，不需要数据流不动点迭代：

* **def**：`for` op 进入处。
* **use**：body（含嵌套）里任何 `scalar_args` / region `starts` /
  `BufferElement.indices` 的 PrimExpr 树中出现该 `loop_var`。
* **kill**：`for` op 的 body 结束。

所以「活跃区间」= 循环的词法嵌套范围。**同一时刻活跃的循环变量数
= 当前嵌套深度**。算法：

1. 递归遍历 `mod.ops`，维护一个「当前嵌套的 `for` 栈」。
2. 进入一个 `for` → 它的循环变量在栈上，需要：
   * 1 个 `gp_loop`（C_LOOP 硬件计数器，恒占 GP）。
   * idx：若该 `for` 是 `unroll` → idx 是编译期常量，**0 GP**
     （已实现，`_emit_for` unroll 分支绑 `IntImm`）；
     若是 `serial`（C_LOOP）→ idx 需要 1 个寄存器位
     （GP 或 IntRAM slot —— 见第 6 节）。
3. 退出 `for` → 释放。
4. 全程峰值 = 需要预留的循环寄存器总数。
5. 按嵌套深度给每层一组固定 GP 号（外层先分，线性扫描即可 —— 区间
   是严格嵌套的，没有部分重叠，不需要图着色）。

「严格嵌套、无部分重叠」是这个分析能简单的根本原因：循环区间要么
包含、要么不相交，永远不会交叉。线性扫描 / 栈深度就够，不需要通用
寄存器分配器。

## 5. 改动清单

| 文件 | 改动 |
|------|------|
| 新建 `loop_register_alloc.py` | liveness pass：递归遍历，算每层 `for` 的循环寄存器，把 GP 号写到 `op.annotations["loop_gp"]`（`gp_loop` + 可选 idx_gp）。算出预留集合。 |
| `hlir.py` | `Op.annotations` 约定新键 `loop_gp` —— 无需改结构。 |
| `pipeline.py` | address_alloc 之后插入 `loop_register_alloc.run(mod)`；它返回预留 GP 集合，传给 `IsaEmitterPass` 构造的 `RegisterAllocator(gp_reserved=...)`。 |
| `isa_pass._emit_for` | 不再 `allocate_gp`/`claim_idx_slot` 循环寄存器，改读 `op.annotations["loop_gp"]`。`pin` 仍做（防 emit 临时分配器误用），但 GP 号是 pass 给的。 |
| `expr_materializer` / `emit_*` | **不动** —— 它们的 `allocate_gp` 照旧，只是池子小了。 |

不动：`register_alloc.py` 的核心（只多用 `gp_reserved`）、所有 mid_ir
pass、`address_alloc`、`dead_buffer_elim`、`loop_interchange` /
`fuse_adjacent_loops`。

## 6. 待定 — idx 放 GP 还是 IntRAM

`serial` 循环的 idx，两个选择：

* **idx 进预留 GP**：body 里用 idx 零 `S_LD_INT`（materializer 走
  `binding is int` 快路径）。代价：每层 serial 多预留 1 个 GP。
* **idx 进 IntRAM**：每层只预留 `gp_loop` 1 个 GP，idx 走 `S_LD_INT`
  重载（已有 per-op 缓存把同一 op 内的重载压成 1 次）。

liveness pass 算出峰值嵌套深度后，可以**动态决定**：深度浅 → idx 全
进 GP（最快）；深度深到 GP 不够 → 外层 idx 落 IntRAM、内层进 GP。这
正是「全局视野」带来的好处 —— emit 阶段做不到，pass 能。

第一版可以先简单：idx 一律 IntRAM（保守、不会 GP 耗尽），把「按深度
混合」留作 pass 内的后续优化。

## 7. 这样为什么根治 Bug 2

Bug 2 的根源：循环变量寄存器和 materializer 临时值在同一个裸 GP 池，
caller 不知道手里的寄存器是不是别人（idx 缓存）拥有的，乱 `pin/free`。

分层后：循环变量寄存器在 `gp_reserved` 里，emit 阶段的 `allocate_gp`
**物理上拿不到它们**。materializer / `_resolve_offset` 操作的永远是
「临时池」里的寄存器，那些确实是 caller 拥有、可以自由 pin/free 的。
两类寄存器物理隔离 → 不存在「谁拥有」的歧义 → Bug 2 不可能发生。

per-op idx 缓存（已实现）那个 `owns_register=False` 的别扭设计，分层
后也可以简化 —— idx 寄存器是预留的，本来就不该被 caller 的 `release`
碰。

## 8. 风险

* idx 全进 IntRAM（第一版保守选择）→ loadback 仍在，但 per-op 缓存
  已把同-op 重复压掉。可接受，后续按第 6 节优化。
* emit 临时池变小：预留掉循环寄存器后，深嵌套 kernel 留给 emit 的 GP
  可能不足 → `emit_matmul_general` 的 7 个 scratch 可能不够。liveness
  pass 应在算出预留集合时**检查** `15 - 预留数 >= 单个 op 最大 scratch
  需求`，不满足就报错（明确，而非 emit 阶段 auto-spill 到崩）。
