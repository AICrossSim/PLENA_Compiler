# SPMD Lane-Group Rewrite — Design

Status: design-only, 2026-05-11. Not yet implemented.

Replaces the four lane-fusion graph passes (`split_lane_groups`,
`lift_lane_groups`, `allocate_group_memory`, `expand_buffers`) with
**three early TIR passes** that resolve lane fusion before
`lift_from_raw_primfunc` ever runs:

1. `classify_lane_use` — scan op annotations + how `by` is used,
   tag each buffer with its lane-fusion role.
2. `expand_lane_grid` — for tagged buffers only, add a LANE outer
   dim and wrap per-lane work in a serial loop.
3. `infer_lane_layout` — pick where the lane axis sits in each
   buffer (BSHD vs BHSD) and rewrite shape + indices accordingly.

The split is intentional: each pass touches a different aspect of the
IR and can be unit-tested independently. They share state through
buffer attributes set by step 1.

---

## 1. Why

Today's `allocate_group_memory` + `expand_buffers` encode lane fusion
as a **buffer-shape decision**: a kernel-author 2D buffer
`alloc_shared((rows, hlen))` is silently rewritten into a 4D
`(B, rows, lane, hlen)` (COL_PACK) or `(B, lane, rows, hlen)`
(ROW_STACK), with the choice driven by ~8 heuristics. Every downstream
op then has to be lane-aware.

Three problems:

1. **Source code lies.** The kernel writer sees 2D, the compiler sees
   4D, the ASM consumer sees yet another shape. Every layer needs to
   re-derive what's true.
2. **Heuristics are opaque.** `_resolve_row_at_coords`'s if-else chain
   in `isa_pass.py` is a faithful read of where the buffer ended up —
   but you have to trace through 4 passes to know why.
3. **Lane fusion bugs are silent.** Wrong COL_PACK vs ROW_STACK pick
   = mis-laid buffer = numerically wrong ASM, no compile error.

The new model treats lane fusion as **buffer dimensionality + a serial
loop**: every `T.alloc_shared((rows, hlen))` becomes a `(LANE, rows,
hlen)` buffer (one extra outermost dim), the grid `by` axis is
replaced by an explicit `for lane in serial(LANE)`, and a separate
small pass decides where the lane dimension actually sits in each
buffer (BSHD vs BHSD) by inspecting how the buffer is used.

---

## 2. Where the new passes live

```
T.prim_func (tilelang DSL output)
   ↓ inline_let_stmts                      ← unchanged
   ↓ lower_compound_fp_stores              ← unchanged
   ↓ classify_lane_use      ★ NEW          ← tag each buffer with lane-fusion role
   ↓ expand_lane_grid       ★ NEW          ← tagged buffers gain a LANE outer dim; lane loop wraps per-lane work
   ↓ infer_lane_layout      ★ NEW          ← move lane dim to its physical position; rewrite indices
   ↓ lift_from_raw_primfunc                ← unchanged
[Graph]                                    ← graph_passes never see lane fusion
   ↓ graph_annotate_grid                   ← simplified (no lane handling)
   ↓ graph_annotate_sync                   ← deleted (no async/sync distinction left)
   ↓ split_lane_groups                     ← deleted
   ↓ lift_lane_groups                      ← deleted
   ↓ fuse_elementwise                      ← kept, simplified
   ↓ scope_inference                       ← kept, simplified
[graph_pipeline.materialize]
   ↓ allocate_group_memory.analyze         ← deleted
   ↓ expand_buffers.expand                 ← deleted
   ↓ lower_fp_row_patterns                 ← kept
```

**Net:** 4 graph passes deleted, 2 graph passes simplified, 3 new TIR
passes added. Estimated line delta: −1500, +600.

---

## 3. The three new passes

### 3.0 `classify_lane_use` — tag buffers by lane-fusion role

**Why first.** `expand_lane_grid` can't blindly add a LANE dim to
every buffer in the kernel — only buffers that participate in lane
fusion need it. Whether a buffer participates depends on **how the
ops that touch it are annotated**, not on the buffer's shape. So we
need one walk over the function body first, before any rewriting,
to label each buffer.

**Inputs the classifier looks at:**

| Op site                                                       | Role assigned to operand buffers |
|---------------------------------------------------------------|----------------------------------|
| `T.gemm(A, B, C)` under `T.attr(0, KIND, "btmm")`             | A: btmm_lhs, B: btmm_rhs, C: btmm_out |
| `T.gemm(A, B, C)` with no KIND attr                           | A: per_head_lhs, B: per_head_rhs, C: per_head_out |
| `T.copy(hbm_slice, dst)` where the HBM slice indexes `by`     | dst: lane_dma_dst                |
| `T.copy(hbm_slice, dst)` with no `by` in the slice            | dst: single_lane (no LANE dim)   |
| `T.serial(N) + T.Parallel(M)` over a tagged buffer            | propagates the tag to the body's loads/stores |
| Anything else                                                 | scalar — keep on outer attribute table |

**Output:** every `tir.Buffer` (param or alloc) gains one of:

- `lane_aware = True` (gets LANE outer dim in step 3.1)
  - sub-tag picks layout in step 3.2: `col_pack` / `bhsd` / `single`
- `lane_aware = False` (untouched in steps 3.1 and 3.2)

Stored as `buffer.var`-keyed attributes the next two passes read.

**Implementation:** one `stmt_functor.post_order_visit` walk. The
classification rules above each become one `isinstance` check + one
attribute write. ~80 lines.

**This is where we keep "implicit conventions inferred from `by`
use"** rather than forcing kernel authors to rewrite all 7 kernels
with explicit `T.async_copy` / `T.lane_parallel` macros. Today's 8
buffer-shape heuristics are gone; what stays is "if the op site says
btmm or uses `by`, lane fusion applies".

### 3.1 `expand_lane_grid` — pure structural rewrite

**Input:** a `tir.PrimFunc` post-`classify_lane_use`. Every buffer
has its `lane_aware` flag set. `LANE = MLEN / btmm_hlen` (= 4 today).
The kernel author marks the lane axis with
`T.func_attr({"plena.lane_axis": "by"})`.

**Output:** the same `tir.PrimFunc` with:

- The lane-axis grid var **erased**. If extent == LANE, no surrounding
  loop. If extent is a multiple of LANE, wrap with
  `for by_outer in T.serial(extent // LANE)`.
- Every buffer with `lane_aware = True` rewritten from
  `T.alloc_*((shape...))` to `T.alloc_*((LANE,) + shape)` — one extra
  outermost dim. Buffers with `lane_aware = False` are untouched.
  No new macro, no name mangling: `Q_sh` stays `Q_sh`, just shape
  `(LANE, rows, hlen)` instead of `(rows, hlen)`.
- Every reference to a lane-aware buffer indexed accordingly:
  - **Sync ops** (DMA, BTMM, V_*, etc.) that consume the buffer
    whole: the surface call stays `T.copy(...)` / `T.gemm(...)`.
    Codegen later sees a 3D buffer where dim 0 == LANE and emits
    the multi-lane HW instruction. No `*_multi` op kind.
  - **Per-lane work** (row_*, fp_*, per-head matmul on a
    `per_head_lhs`-tagged buffer): wrapped in
    `for lane in T.serial(LANE)` and indexed `Q_sh[lane, ...]`. The
    `lane` var is a regular serial loop var; no lane fusion semantics
    attached.

**That's the entire pass.** It does not pick layouts, does not insert
reshape ops, does not touch what scope buffers live in. Pure
structural: grid → buffer dim + loop.

**Worked example.** Input slice of `flash_attention_min` (LANE=4,
head_count=4):

```python
with T.Kernel(num_q_blocks, head_count) as (q_block, by):
    Q_sh   = T.alloc_shared((rows, hlen), "f16")
    K_sh   = T.alloc_shared((rows, hlen), "f16")
    S_loc  = T.alloc_fragment((rows, MLEN), "f16")
    M_OLD  = T.alloc_fragment((rows,), "f16")

    T.copy(Q_hbm[0, q_block*rows, by, 0], Q_sh)
    T.copy(K_hbm[0, kv_block*rows, by, 0], K_sh)
    with T.attr(0, KIND, "btmm"):
        T.gemm(Q_sh, K_sh, S_loc, transpose_B=True)

    for row in T.serial(rows):
        M_OLD[row] = M_INIT[row]
```

After `expand_lane_grid`:

```python
# `by` erased; head_count == LANE so no by_outer loop
Q_sh   = T.alloc_shared((4, rows, hlen), "f16")     # +outer dim
K_sh   = T.alloc_shared((4, rows, hlen), "f16")
S_loc  = T.alloc_fragment((4, rows, MLEN), "f16")
M_OLD  = T.alloc_fragment((4, rows), "f16")

T.copy(Q_hbm[0, q_block*rows, 0:4, 0], Q_sh)        # 3D dst
T.copy(K_hbm[0, kv_block*rows, 0:4, 0], K_sh)
with T.attr(0, KIND, "btmm"):
    T.gemm(Q_sh, K_sh, S_loc, transpose_B=True)     # all 3D

for lane in T.serial(4):                            # ← was implicit lane fusion
    for row in T.serial(rows):
        M_OLD[lane, row] = M_INIT[lane, row]
```

### 3.2 `infer_lane_layout` — choose where the lane dim sits

After `expand_lane_grid`, every lane-aware buffer has its lane dim at
position 0. But the **physical** layout (which 7D slot the lane axis
occupies in VRAM/MRAM) depends on how the buffer is consumed. Two
flavors today:

- **COL_PACK:** lane axis in the H slot of BSHD. Used for VRAM tiles
  that BTMM reads as LHS or that per-row VRAM↔FPRAM ops walk.
- **ROW_STACK / BHSD:** lane axis in the H slot but ahead of S. Used
  for BTMM outputs (`S_loc`) where each lane writes a full
  (rows, MLEN) slab and per-head matmul consumes one lane's slab as
  its LHS.

In the new model, "BHSD vs BSHD" reduces to **which dim of the
buffer's shape carries the lane index**. Always lane-dim-at-0 for
COL_PACK; lane-dim-at-1 for BHSD; etc.

**This pass:**

1. For each `lane_aware = True` buffer, **read its role tag from
   step 3.0** and map role → layout:
   - `btmm_lhs`, `btmm_rhs`, `lane_dma_dst`, `per_head_out`,
     fp/row state → COL_PACK (lane at outer dim 0)
   - `btmm_out`, `per_head_lhs` → BHSD (lane at dim 1)
   - Conflicting roles → error (a buffer can't be both BTMM output
     and BTMM LHS; if classification disagrees it's a kernel-source
     issue, not a pass issue)
2. **Permute the buffer's shape** so the lane axis sits where the
   chosen layout wants it. e.g. `S_loc: (4, rows, MLEN) → (rows, 4,
   MLEN)`.
3. **Update every load / store** of that buffer to permute its index
   tuple correspondingly. e.g. `S_loc[lane, row, col] →
   S_loc[row, lane, col]`.

No reshape ops in the IR — we rewrite shapes and indices in place.
The pass is one walk + one rewrite, no fixpoint, no cross-buffer
flow analysis.

**Worked example continuing from §3.1.** Suppose `S_loc` is BTMM output
+ per-head matmul LHS → both votes for BHSD. Then:

```python
S_loc = T.alloc_fragment((rows, 4, MLEN), "f16")    # ← lane moved to dim 1

with T.attr(0, KIND, "btmm"):
    T.gemm(Q_sh, K_sh, S_loc, transpose_B=True)     # codegen reads buf shape

for lane in T.serial(4):
    for row in T.serial(rows):
        M_OLD[lane, row] = ... S_loc[row, lane, ...] ...
```

`Q_sh`, `K_sh` stay `(4, rows, hlen)` — no consumer wanted them
anywhere else.

---

## 4. What graph_passes / codegen need to know

Graph IR sees normal 3D buffers. The lane axis is **just a dim**; it
doesn't get a special label. Codegen distinguishes single-lane from
multi-lane purely by buffer shape:

- DMA / BTMM / V_* called on a buffer whose shape matches a known
  3D-multi-lane pattern (dim that equals LANE is at the layout's H
  slot per `plena.layout`) → emit the multi-lane HW instruction
  with `lane_count=LANE`.
- DMA / BTMM / V_* called on a 2D buffer (or a 3D buffer whose
  outer dim is *not* LANE — single-tile broadcast case) → emit the
  single-lane HW instruction.

This is one if-else in each codegen handler, replacing the entire
`_resolve_row_at_coords` chain.

**Per-lane row/fp ops** stay exactly as they are — they receive a
single-lane address (the result of indexing a 3D buffer with the
`lane` loop var) and emit one per-lane HW instruction per `lane`
iteration. The legacy `lower_fp_row_patterns` pass handles the
`for lane in serial(4)` exactly like any other serial loop.

---

## 5. What gets deleted, what gets kept

### 5.1 Deleted

- `frontend/passes/graph_passes/split_lane_groups.py`
- `frontend/passes/graph_passes/lift_lane_groups.py`
- `frontend/passes/graph_passes/allocate_group_memory.py`
- `frontend/passes/graph_passes/expand_buffers.py`
- `frontend/passes/graph_passes/annotate_sync.py`
- The lane-handling branches in `_resolve_row_at_coords` (in
  `isa_pass.py`).

### 5.2 Simplified

- `frontend/passes/graph_passes/annotate_grid.py` — only annotates
  non-lane grid axes (`q_block` etc).
- `frontend/passes/graph_passes/fuse_elementwise.py` — no longer
  needs to special-case lane-group regions.
- `frontend/passes/graph_passes/scope_inference.py` — buffer scope
  comes straight from the kernel's `alloc_*` call; no cross-buffer
  inference.
- `codegen.py` — single-lane vs multi-lane is one shape check per
  intrinsic, no lookup table.
- All 7 kernels in `tilelang_tvm_compiler/kernels/` — one new line
  each: `T.func_attr({"plena.lane_axis": "by"})`. Buffer alloc
  shapes stay literally the same — `expand_lane_grid` adds the
  outer dim.

### 5.3 Untouched

- `intrinsics.py` (no new op kinds — multi-lane is implicit in
  buffer shape)
- `isa_emitter.py` / `isa_pass.py` (modulo the simplification above)
- `expr_materializer.py`, `register_alloc.py`, `address_alloc.py`
- Everything under `transactional_emulator/`

---

## 6. Risks / open questions

1. **`expand_lane_grid` and `T.copy`'s slice arg.** Today
   `T.copy(K_hbm[0, kv*rows, by, 0], K_sh)` uses `by` as an HBM
   index. After erasure we need to turn it into a **range** indexing
   `0:LANE` over that axis. That's the easy case (lane axis indexes
   directly into a tensor dim). If the kernel does arithmetic on
   `by` beyond a bare reference, the pass should refuse and ask the
   author to refactor — not try to be clever.

2. **`infer_lane_layout` conflict resolution.** A buffer used as
   both BTMM LHS (wants COL_PACK) and BTMM output (wants BHSD) is
   structurally impossible — it's not the same buffer. But what
   about a buffer used as BTMM input AND per-head matmul input? This
   does happen (S_loc → P @ V). The classification table in §3.2
   needs to be exhaustive against the 7 kernels; we'll discover any
   gap during step-by-step migration.

3. **Layout permutation cost.** Rewriting every load/store to
   permute its index tuple touches many sites in long kernels. It's
   mechanical but easy to bug. Mitigation: write a small
   `permute_index(buf_var, perm)` helper and use it everywhere; one
   permutation table per buffer.

4. **`q_block` is not a lane axis.** It's a sequential block index
   over Q tiles. The new pass only erases the var marked
   `plena.lane_axis`; everything else (including unmarked grid axes)
   stays as-is, lowered to `T.serial` by the existing TVM pipeline.

5. **head_count > LANE.** When `head_count = 8, LANE = 4`:
   `for by_outer in T.serial(2)` wraps the body. The lane buffers
   are reused across by_outer iterations — matches today.

6. **migration order.** Old and new pipelines coexist on
   `compile_func` per kernel, gated by
   `T.func_attr({"plena.use_spmd": True})`. flash_attention_min
   first; the rest follow once HLIR diff is byte-clean.

---

## 7. Implementation order

| Step | Deliverable | Estimate |
|---|---|---|
| 1 | `classify_lane_use.py` — role-tagging walk with the 6 rules in §3.0 | 0.5 day |
| 2 | `expand_lane_grid.py` for flash_attention_min only — reads tags from step 1, adds LANE dim + lane loop | 1.5 days |
| 3 | `infer_lane_layout.py` — reads tags from step 1, permutes shapes + indices | 1 day |
| 4 | Codegen: shape-driven multi-lane vs single-lane dispatch in `_resolve_row_at_coords` (and equivalents) | 1 day |
| 5 | flash_attention_min source: add `plena.lane_axis` + `plena.use_spmd` attrs; verify HLIR matches today's output byte-for-byte | 1 day |
| 6 | Simplify `annotate_grid` / `fuse_elementwise` / `scope_inference` to remove lane handling | 1 day |
| 7 | Delete the four graph passes + their tests + dead branches in `_resolve_row_at_coords` | 0.5 day |
| 8 | Port the other 6 kernels (mostly: add the func attrs; extend the role table in step 1 if any new pattern shows up) | 2 days |
| 9 | Delete the legacy `compile_func` branch (`expand_lane_buffers=True`) | 0.5 day |

Total: **~9 days** of focused work. Three small single-purpose
passes beat one big multi-walk pass for readability and
testability: classify alone can be unit-tested by checking the role
tags it sets, expand alone can be tested by feeding it a hand-tagged
IR, and infer alone can be tested similarly. The IR stays in vanilla
TIR throughout — no new macros, no `*_multi` op kinds, no
contiguous-backing tricks.
