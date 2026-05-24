# Plan: linear_min via BTMM (non-async) + split matmul / materialize

## Goal

Make `linear_min` compute its MLEN×MLEN×MLEN GEMM through the **BTMM**
systolic path instead of the current per-head `plena.matmul` (M_MM_WO)
path, exploiting the hardware accumulator so the K-loop accumulates in
`hm_accum` and only drains once.

This is a **second, distinct BTMM path** from flash-attention's Q@K^T:
- flash btmm: packed-head, **async**, wrapped in `MultiLaneOp`, lane_count>1.
- linear btmm: **non-async**, NOT multilane-fused, plain MLEN×MLEN tiles.

## Hardware flow (confirmed from main.rs)

1. **M_BTMM** (`btmm`, main.rs:413-486): computes `vec @ mat^T * scale` and
   **accumulates** into `hm_accum[lane_count, mlen, mlen]` (`+=`, line 485).
   So repeated BTMM over the K-loop accumulates in hardware — no software K
   add needed.
2. **M_BMM_WO** (`bmm_wo`, main.rs:623-647): drains `hm_accum` to VRAM, then
   **zeroes `hm_accum`** (line 639). One drain per output tile.
3. The drained tile(s) are then **added into the destination** (the BLEN /
   lane tiles accumulate into dst via v_add) — only needed if more than one
   accumulator tile maps to one logical dst, or to fold bias / wider-N.

Key consequence: the K-loop's software `C_loc += SCR_loc` (current
linear_min.py:165-167 / 228-230) **disappears** — hardware `hm_accum +=`
replaces it. The single drain happens after the last BTMM.

## User's "split into two ops" model

- **matmul (btmm)**: emit M_BTMM into the accumulator. No writeback.
- **materialize**: emit M_BMM_WO (drain) + the add-into-dst, emitted **only
  right before the shared-memory result is consumed as a src for the last
  time** (i.e. after the K-loop, before C_loc/C_sh is read for the C->HBM
  copy). Deferring the drain past intermediate BTMMs is what lets the
  hardware accumulator do the K accumulation.

## End-to-end change map (frontend → backend; mostly additive)

### 0. Kernel source — `kernels/linear_min.py`
- Wrap the gemm in the new KIND: `with T.attr(0, KIND, "btmm_mm"):
  T.gemm(A_sh, B_sh, C_loc, transpose_B=True)` writing **directly into
  `C_loc`** (the accumulating dst), dropping `SCR_loc` and the manual
  `C_loc += SCR_loc` K-add loop and the `C_loc = 0` seed (hardware accum +
  BMM_WO zero handles it).
- `KIND` import from `gemm_macros` (already exists; value `"plena.gemm_kind"`).
- Bias path unchanged (still a tile add after the drain).

### 1. New frontend pass — `frontend/passes/split_btmm_materialize.py` (NEW)
Runs in pipeline stmt-prep (after inline_let, near fission_vector_chains).
- Recognize a `T.gemm` under KIND `"btmm_mm"`.
- Rewrite the gemm-in-K-loop into:
  - **per k_block**: a `btmm_mm` compute op (accumulate-only, no dst write).
  - **after the K-loop, before C_loc's first read as a src**: a
    `materialize` op that drains the accumulator into C_loc (BMM_WO) and
    adds the tiles into the logical dst.
- "last src" detection: find the first statement after the gemm's enclosing
  K-loop that READS the gemm's dst buffer (the `T.copy(C_loc, C_sh)`), and
  insert materialize immediately before it. (For the minimal single-drain
  case this is just "after the K-loop".)
- NOTE: simplest first cut — since hardware already accumulates, the pass
  may only need to (a) tag the btmm op as compute-only and (b) emit a single
  materialize after the K-loop. The deferral rule generalizes later.

### 2. mid_ir IR — `frontend/mid_ir/ir.py`
- Add `kind="btmm_mm"` as a recognized Gemm kind (no struct change; it's a
  string). Optionally add a new mid_ir node `Materialize` (dst BufferRef +
  tile_count) OR represent materialize as an Elementwise/RawStore variant.
  **Decision needed (see open Q1).**

### 3. fold — `frontend/mid_ir/passes/fold.py`
- `_fold_gemm` already carries `kind` through from the KIND attr
  (fold.py ~1025). `"btmm_mm"` flows through unchanged → `Gemm(kind="btmm_mm")`.
- If a `Materialize` node is used, add folding for the new intrinsic the
  split pass emits.

### 4. mark — `frontend/mid_ir/passes/mark.py` (`_mark_gemm`, line 90)
- Currently: `is_btmm = op.kind == "btmm"` → `marker=BTMM, can_async=True`.
- Add: `"btmm_mm"` → a NEW marker (e.g. `Marker.BTMM_MM`) with
  **`can_async=False`** (non-async, not multilane-fused). The compute op
  stays per-tile, runs in program order inside the K-loop.

### 5. split / async / distribute — `frontend/mid_ir/passes/`
- async_wrap: only wraps `can_async=True`, so btmm_mm (False) is NOT wrapped
  in Async — correct (non-async).
- split/distribute: btmm_mm is MLEN×MLEN (lane_count==1 for linear), so the
  cluster/lane handling is trivial. Confirm it isn't forced into a per-lane
  loop the way `overwrite` is (we want one M_BTMM per k_block, not a
  per-lane matmul).

### 6. to_plena — `frontend/mid_ir/passes/to_plena.py`
- Add a lowering for `Gemm(kind="btmm_mm")` → HLIR `Op(kind="btmm_mm")`
  (compute-only: a_region VRAM, b_region MRAM, c-accumulator implied; NO c
  writeback region in the compute op).
- Add a lowering for the `Materialize` node → HLIR `Op(kind="bmm_wo")`
  (drain: dst VramRegion + tile_count) followed by the tile add(s).
- Model on `_lower_multi_lane_btmm` (to_plena.py:1266) but **without**
  MultiLaneOp wrapping and without the bundled writeback (the existing one
  emits BTMM+BMM_WO together; we split them).

### 7. pre_isa_pass_v2 — `pre_isa_pass_v2.py`
- Dispatch table: add `"btmm_mm"` → emit M_BTMM only; `"bmm_wo"` → emit
  M_BMM_WO + the add-into-dst.
- Reuse existing `emit_btmm` / `emit_btmm_wo` (isa_emitter.py:425-463); split
  their call sites so compute and drain are separate HLIR ops.

### 8. isa_pass.py (legacy) — mirror for completeness
- `_emit_btmm` (isa_pass.py:1793) currently emits BTMM+BMM_WO together. Add
  `_emit_btmm_mm` (compute only) and `_emit_bmm_wo` (drain only) for the new
  kinds, leaving the existing fused `_emit_btmm` for flash's path untouched.

### 9. intrinsics.py
- Register `plena.btmm_mm_at` (compute) and `plena.bmm_wo_at` (drain) specs
  mirroring `plena.btmm`.

## What does NOT change
- flash-attention's existing `"btmm"` path (async, packed-head) — fully
  untouched; new kind keeps it isolated.
- The M_MM / M_MM_WO matmul path stays for kernels that still use plain
  `T.gemm` overwrite (flash's P@V).

## Resolved decisions

- **Q1 (materialize representation) — RESOLVED**: materialize is LIGHTWEIGHT;
  no heavy new mid_ir node touched by fold/mark/split. The btmm compute op
  flows through the normal Gemm path; the **drain is inserted by the new
  frontend pass** as a separate lightweight op right before the btmm dst is
  first read as a src. Lowers to a small HLIR op kind (`bmm_wo`); no new
  mid_ir struct churn across passes.

- **Q4 (B → MRAM) — RESOLVED, NO KERNEL WORK**: `_infer_scope_overrides`
  (to_plena.py:542-564) already overrides ANY `Gemm`'s B operand to MRAM
  (kind-agnostic) and DMA auto-picks `dma_h2m`. `btmm_mm` is still a `Gemm`,
  so B goes to MRAM for free. Kernel keeps plain `T.copy(B_hbm, B_sh)`.

- **drain placement — RESOLVED**: the new frontend pass auto-inserts the
  drain right before the btmm dst's first read-as-src (after the K-loop,
  before `T.copy(C_loc, C_sh)`). Kernel only writes the btmm gemm; no hand-
  written materialize. This lets the hardware accumulator do the K
  accumulation (multiple BTMM compute, one drain).

## Still to confirm during impl (low risk)
- **Q2**: `transpose_B=True` mat-transpose under BTMM matches linear's A@B^T
  (B is (N,K)). flash btmm also uses transpose_B; same MRAM tile-transpose
  expected.
- **Q3**: lane_count==1 for linear (no head packing) → `hm_accum` drains a
  single MLEN×MLEN tile, so "add BLEN tiles into dst" is a single copy/no-op
  in the minimal case. Multi-tile add only matters if packing is added later.
- **Q4**: Does B need to be in MRAM (BTMM reads mat from MRAM) vs the current
  linear copying B into shared/VRAM? The DMA for B must target MRAM
  (dma_h2m) like flash's K, not dma_h2v. Kernel `T.copy(B_hbm, B_sh)` may
  need B_sh to be an MRAM-scoped buffer.

## Suggested implementation order
1. Resolve Q1-Q4.
2. Kernel source rewrite (linear_min.py) + new KIND.
3. mark.py new marker (smallest change, unblocks routing).
4. to_plena lowering for btmm_mm compute + bmm_wo drain.
5. pre_isa_pass_v2 dispatch + reuse emitters.
6. The split_btmm_materialize frontend pass (or, minimal first cut: have the
   kernel emit the two ops and skip the auto-split pass until the simple
   case works).
7. intrinsics + legacy isa_pass mirror.
8. Test linear_min numeric (1×1×1, then m/n/k>1).
