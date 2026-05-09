# Pipeline Architecture

End-to-end walkthrough of how `tilelang_tvm_compiler` lowers a user-written
`@T.prim_func` to PLENA ISA, with notes on each pass's responsibilities,
inter-pass dependencies, and known structural gaps.

---

## 1. Overview

```
@T.prim_func (user's tilelang kernel)
        │
        │  Frontend pipeline (10 stmt-rewriting passes + lift_to_blocks
        │                     + graph_pipeline back end, all operate on TIR)
        ▼
TIR with plena.* extern calls
(plena.matmul / plena.mv / plena.btmm / plena.zero_v / plena.v_add /
 plena.dma_h2v_slice / plena.copy_v_to_v / plena.row_load_v_to_fp / …)
        │
        │  PlenaCodegen.lower_to_hlir()
        │  (NOTE: a method on PlenaCodegen — does NOT relate to the
        │   deleted frontend pass that used to share this name.)
        ▼
HLIRModule (buffers + linear ops list)
        │
        │  AddressAllocationPass
        ▼
HLIR with concrete addresses on every buffer
        │
        │  IsaEmitterPass
        ▼
ISA text (the final .asm)
```

**Core principles (established in v1):**

1. **User-facing surface is tilelang DSL only** — `T.gemm` / `T.copy` /
   `T.Parallel` / `T.alloc_*`. `plena.*` is a compiler-internal IR
   namespace; kernel authors must not write it directly.
2. **Per-head offsets are auto-injected** — the user writes
   `T.gemm(buf, buf, buf)` without spelling out `by * stride`; the
   compiler infers each operand's lane-axis stride from its post-expansion
   shape.
3. **The `KIND` table has exactly two values** — `"btmm"` (head-fused)
   and `"overwrite"` (everything else). The lowering picks
   `plena.matmul` vs `plena.mv` (or `plena.btmm` vs `plena.btmv`)
   automatically based on the LHS row count.
4. **`fuse_elementwise` plus the idempotent lane-marking inside
   `KIND="overwrite"`** subsume four separate-looking use cases under one
   KIND: per-head matmul, per-head mv, whole-buffer DMA-driven matmul, and
   fragment-only output accumulation.

---

## 2. Frontend pipeline — 12 passes

Listed in execution order from `frontend/pipeline.py`.

### 2.1 `inline_let_stmts` — TIR housekeeping
- **What it does:** inlines `let x = expr in body` LetStmts (substitutes
  `expr` for `x` inside `body`).
- **Why:** the tilelang frontend occasionally emits these, and folding
  them up front lets later passes match expression patterns reliably.
- **Scope:** pure IR cleanup, no semantic change.

### 2.2 `lower_compound_fp_stores` — `arr[i] += x` → `arr[i] = arr[i] + x`
- **What it does:** rewrites compound assignments (which are a separate
  IR node) into explicit read-modify-write.
- **Why:** the downstream `fuse_elementwise` matches
  `dst[i] = lhs[i] + rhs[i]` style BinOp patterns; compound stores would
  fall through.
- **Scope:** predicate — only fires for kernels that contain compound
  stores.

### 2.3 `annotate_gemm_kind` — attach KIND attr to every `T.gemm`
- **What it does:** scans every `T.gemm`. If the user wrapped it in
  `with T.attr(0, KIND, ...)`, captures that kind; otherwise applies the
  default `"overwrite"`. Every gemm ends up wrapped in
  `tir.AttrStmt(plena.gemm_kind, kind)`.
- **Valid kinds (post-v1, only two):**
  - `"btmm"` — head-fused; lowers to plena.btmm / plena.btmv.
  - `"overwrite"` — everything else; lowers to plena.matmul / plena.mv.

### 2.4 `annotate_group` — find lane-fusion candidate axes
- **What it does:** walks `T.Kernel` head dims and `T.Parallel(N)` loops.
  Wraps each candidate for-loop in
  `tir.AttrStmt(plena.group, value=N)`. The `value=N` is the axis's
  logical width.
- **Role:** this attr is the "signpost" for lane fusion — it tells
  `split_lane_groups` and the eventual `graph_pipeline` back end which
  for-loops are lane candidates.

### 2.5 `annotate_sync` — mark sync sites
- **What it does:** wraps the following ops in
  `tir.AttrStmt(plena.sync, …)`:
  - HBM↔local `T.copy` (DMA)
  - vram↔fpram `T.copy` (lowers to S_MAP_*_*)
  - vram↔vram `T.copy` (V_ADD_VF f0=0)
  - `T.gemm` under KIND=btmm
  - already-fused `plena.zero_v` / `plena.v_*` extern calls
- **Sync site semantics:** "one HW instruction that fires across all
  lanes simultaneously." Downstream passes (`split_lane_groups`,
  `graph_pipeline`) use this to decide which ops hoist OUTSIDE the
  per-lane for-loop (one multi-lane invocation) and which stay INSIDE
  (per-lane serial loop).

> **Tech debt (see § 5):** this pass straddles tile-DSL (`T.copy` /
> `T.gemm`) and lowered `plena.*` extern forms, recognising both. Adding
> a new op requires touching both branches; missing one is a silent bug
> source.

### 2.6 `split_lane_groups` — split head axis into outer × inner
- **What it does:** for every for-loop with a `plena.group` attr:
  - If `extent == lane_count` (default 4): leave alone — already
    lane-fusion-eligible.
  - If `extent == k * lane_count` (k > 1): split into

    ```
    for v_outer in range(k):
        plena.group(k):
            for v_inner in range(lane_count):
                plena.group(lane_count):    ← marker for lane fusion
                    body[v → v_outer * lane_count + v_inner]
    ```
- **Important details:**
  - Body uses of the original `v` are substituted with the compound
    `v_outer * lane_count + v_inner` (`_VarSubst`).
  - The inner `plena.group(lane_count)` AttrStmt is what
    `graph_pipeline` later uses to identify the lane for. (It used to
    get consumed mid-walk by the old segmenter; the graph back end reads
    it once during lane-group extraction and never mutates it.)
  - The inner `Var`'s name is `f"{original_name}_i"` (e.g. `by_i`).

### 2.7 `fuse_elementwise` — `T.Parallel` patterns → `plena.v_*`
- **What it does:** matches three patterns and rewrites them in-place:
  - **Single-loop binary:**
    `for i in T.Parallel(N): dst[..., i] = a[..., i] OP b[..., i]`
    → `plena.v_<op>(a, b, dst)` (currently OP ∈ {`+` → plena.v_add}).
  - **Single-loop zero fill:**
    `for i in T.Parallel(N): dst[..., i] = 0`
    → `plena.zero_v(dst)`.
  - **Nested:**
    `for r in T.serial(R): for c in T.Parallel(C): dst[r, c] = …`
    → folded into a single whole-buffer `plena.v_*` / `plena.zero_v`.
- **Why the nested fold matters:** with lane fusion, the two loops
  together iterate `R * C * lane_count` elements — exactly the
  post-expansion buffer size. The whole-buffer HW path covers that in a
  single invocation. Leaving the outer `T.serial(R)` would re-execute
  the same whole-buffer op `R` times: wasted cycles for `zero_v`, an
  R-fold accumulation bug for `v_add`.
- **Restriction:** only fires for ops that are inherently whole-buffer
  (zero_v, v_*); per-head ops with offsets (matmul, mv) keep their
  surrounding for-loops.

### 2.8 `scope_inference` — assign storage scope to every buffer
- **What it does:** walks all buffers; based on declaration form
  (`T.alloc_shared` / `T.alloc_fragment` / function parameter) and
  usage context, assigns one of `hbm` / `vram` / `mram` / `fpram`.
- **Output:** `BufferScopeMap` (dict: buffer name → scope).
- **Used by:** `allocate_group_memory` (lane-axis labelling) and
  `graph_pipeline` (T.copy variant selection via the helpers in
  `frontend/passes/lower_to_hlir.py`).

### 2.9 `allocate_group_memory` — expand buffer shapes with a lane axis
- **What it does:** walks lane-group bodies, decides each buffer's
  lane-axis role, then rewrites the IR — buffer shapes get expanded and
  buffer accesses get the lane var inserted.
- **Three lane-axis modes:**
  - **COL_PACK** `(rows, last) → (1, rows, lane_count, last)` — each
    lane occupies its own `last`-wide column slice. Typical: `V_sh`,
    `PV_loc`, `O_loc`. Flat row stride = `lane_count * last` (= MLEN).
  - **ROW_STACK** `(rows, last) → (1, lane_count, rows, last)` — each
    lane occupies its own row block. Typical: btmm output `S_loc`, mv
    LHS. Flat row stride = `last`.
  - **FP_LANE** `(N,) → (lane_count, N)` — FPRAM scalar slot stacked
    across lanes. Typical: M_OLD / M_CURR / SCALE / online-softmax state.
- **Decision sources, by op type:**
  - `T.copy` HBM→local → mark local as COL_PACK.
  - `T.copy` vram↔fpram → mark fpram fragment as FP_LANE.
  - `T.gemm` KIND=btmm → LHS+RHS = COL_PACK, DST = ROW_STACK.
  - `T.gemm` KIND=overwrite → **idempotent**: skip operands already
    marked by surrounding ops; otherwise mark LHS=ROW_STACK,
    RHS+DST=COL_PACK.
  - Already-lowered `plena.*` extern → key off the op name (legacy path
    used by hand-written kernels).
- **Why the idempotent rule:** the legacy
  `_matmul_in_lane_group_kernel` test expects KIND=overwrite to be
  "neutral" (lane labels driven by the surrounding DMAs);
  flash_attention_min's `PV_loc` is fragment-only and has no surrounding
  marker. The "mark only if unmarked" rule satisfies both.

### 2.10 `lower_fp_row_patterns` — FPRAM↔VRAM row-level pattern recognition
- **What it does:** detects specific FPRAM↔VRAM row-element transfer
  patterns (`for i: vram[..., i] = fpram[i]` and friends) and lowers
  them to `plena.row_load_v_to_fp` / `plena.row_store_fp_to_v`.
- **Relationship to `graph_pipeline`'s `_lower_copy` helper:** the
  helper handles buffer-to-buffer wholesale transfers; this pass
  complements it by catching row-element-level rewrite patterns
  upstream so they reach the back end as already-lowered
  `plena.row_*` extern calls.

### 2.11 `lift_to_blocks` — wrap each op as its own BlockRealize

The post-rewrite IR has `tilelang_root` as a single coarse block holding
a flat SeqStmt of all ops (see § 1 overview). `lift_to_blocks` walks
this SeqStmt and wraps each op stmt in its own ``BlockRealize`` with
explicit ``reads`` / ``writes`` extracted from the op's region
arguments. ``plena.sync`` / ``plena.gemm_kind`` AttrStmt wrappers around
each op move INTO the new inner block as ``block.annotations[key] = value``.

Non-op stmts (For loops, nested SeqStmts, raw AttrStmts that aren't
plena.*) pass through unchanged — they're structural wrappers, not
graph nodes. After this pass, lane-group bodies look like:

```
For by in range(4):
  AttrStmt(plena.group, 4):
    BlockRealize "tilelang_root":
      alloc_buffers: [...]
      SeqStmt:
        BlockRealize "op_0" annotations={plena.sync: "..."}:
          reads(...) writes(...)
          body: Evaluate(Call(tl.tileop.copy, ...))
        BlockRealize "op_1" annotations={plena.gemm_kind: "btmm",
                                          plena.sync: "..."}:
          ...
```

The lifted IR is well-formed TIR (`verify_well_formed = True`) and TVM's
`tir.Schedule` API can `get_block` / `get_consumers` on it. The graph
back end consumes this form.

### 2.12 `graph_pipeline` — graph-IR back end

Replaces what used to be a recursive stmt walker (`lower_to_hlir._lower_body`
+ `_segment_lane_for` + `_do_segment` trio). The graph back end:

#### Step 1 — extract a graph

Walk the lifted IR. For each lane-group nest
``For(lane_var) → AttrStmt(plena.group, lane_count) → BlockRealize("tilelang_root")``,
extract a ``LaneGroup``:

```python
@dataclass
class LaneGroup:
    lane_var: tir.Var
    lane_count: int
    nodes: List[GraphNode | tir.Stmt]   # mixed list
    alloc_buffers: List[tir.Buffer]
```

Lifted op-blocks become ``GraphNode`` (kind, op_call, annotations);
non-op stmts (nested For loops, etc.) pass through as raw ``tir.Stmt``
elements — they participate in per-lane wrapping but carry no
graph-level metadata.

#### Step 2 — partition by sync barrier + per-lane affinity

Walk ``LaneGroup.nodes`` linearly:
- A ``GraphNode`` is **sync** if it has a ``plena.sync`` annotation OR
  its op-call is one of the inherently-sync externs (``plena.dma_*`` /
  ``plena.btmm`` / ``plena.btmv`` / ``plena.zero_v`` / ``plena.v_*`` /
  ``plena.copy_v_to_v`` / ``plena.row_*_v_to_fp`` /
  ``plena.row_store_fp_to_v``). Sync nodes emit ONCE outside any for-by,
  with the lane var substituted to 0 (``in_sync=True``) so the op
  becomes a single multi-lane HW instruction.
- Everything else is **per-lane**: it accumulates into a contiguous
  run, which gets wrapped in a ``for-by(0..lane_count)`` loop. The
  for-by uses ``UNROLLED`` if any node in the run is a ``plena.matmul``
  (mirrors the rule in the old segmenter), else ``SERIAL``.

#### Step 3 — emit plena.* extern stmts

Each ``GraphNode`` lowers via the helper functions kept in
``frontend/passes/lower_to_hlir.py``:

| Input | Selector | Output |
|-------|----------|--------|
| `tl.tileop.copy(src, dst)` | scope HBM→vram | `plena.dma_h2v_slice` |
| `tl.tileop.copy(src, dst)` | scope HBM→mram | `plena.dma_h2m_slice` |
| `tl.tileop.copy(src, dst)` | scope vram→HBM | `plena.dma_v2h_slice` |
| `tl.tileop.copy(src, dst)` | scope vram↔vram | `plena.copy_v_to_v` |
| `tl.tileop.copy(src, dst)` | scope vram↔fpram | `plena.row_load_v_to_fp` / `plena.row_store_fp_to_v` |
| `tl.tileop.gemm_py` | KIND=btmm, LHS rows=1 | `plena.btmv` |
| `tl.tileop.gemm_py` | KIND=btmm, LHS rows>1 | `plena.btmm` |
| `tl.tileop.gemm_py` | KIND=overwrite, LHS rows=1 | `plena.mv` |
| `tl.tileop.gemm_py` | KIND=overwrite, LHS rows>1 | `plena.matmul` |
| Already-lowered `tir.call_extern("plena.*")` | — | passthrough |

Per-lane offsets are auto-injected (`_auto_lane_offset`) from each
buffer's lane-axis stride. `dst_row_stride` is auto-computed:
COL_PACK ⇒ `lane_count * last_dim`, ROW_STACK / unexpanded ⇒ `last_dim`.

The lane-offset projection that used to be a separate stmt-rewrite
(`_project_matmul_offsets_to_lane`) is folded into per-lane node
lowering: when ``in_sync=True``, lane-var occurrences in offset
expressions get substituted with 0; per-lane lowering keeps them
referencing the lane var directly so the surrounding for-by drives the
iteration.

#### Why this is better than the old walker

The old `lower_to_hlir._lower_body` interleaved four concerns: (A) tile
DSL → plena translation, (B) lane-fusion segmentation, (C) lane-offset
projection, (D) attribute stripping. Adding a new op kind required
changes in scattered call paths; the order in which AttrStmts were
stripped was load-bearing (and consumed `plena.group` mid-walk, which
is why a v2 attempt to extract concern (B) into its own pass failed).

The graph back end separates these:
- ``lift_to_blocks`` is the ONLY place that sees raw stmt structure
- the graph back end works on a list of ``GraphNode``s, each carrying
  its op-call and a metadata dict
- (B) becomes a list partition; (C) is per-lane vs. in_sync lowering;
  (D) is reading ``block.annotations``

Adding a new sync/per-lane plena op = registering it in
``INHERENTLY_SYNC_EXTERNS`` (or `PER_LANE_UNROLLED_EXTERNS`) + adding
a lower function. No change to the partitioner or walker.

---

## 3. Backend — three stages
(Not part of `frontend/pipeline.py`, but the same `compile_kernel`
flow; see `tilelang_tvm_compiler/pipeline.py`.)

### 3.1 `PlenaCodegen.lower_to_hlir()` ([codegen.py](codegen.py))
TIR → HLIR data structure.
- `_collect_param_buffers` + `_collect_alloc_buffers` walk every buffer
  into `_buffers` (keyed by `tir.Var`, i.e. `buffer.data`).
- `_walk_stmt` / `_walk_evaluate` rewrite each `plena.*` extern call to
  `_hlir.Op(kind="<name without 'plena.' prefix>", buffer_args=[...], scalar_args=[...])`.
- For-loops become `_hlir.Op(kind="for", body=[...])` nests.
- Output is `_hlir.HLIRModule(name, buffers, ops)`.

### 3.2 `AddressAllocationPass` ([address_alloc.py](address_alloc.py))
Assigns each buffer a concrete address in declaration order:
- HBM: from `0`, advance by buffer size.
- VRAM: from `0`, advance by buffer size (each row is MLEN-wide).
- MRAM: tile-aligned allocation.
- FPRAM: from `FPRAM_USER_BASE = 32`, advance by buffer size.

The address is written back into `Buffer.addr`.

### 3.3 `IsaEmitterPass` ([isa_pass.py](isa_pass.py))
HLIR → ISA text. Every op kind has a corresponding `_emit_*` method
(`_emit_v_add`, `_emit_matmul`, `_emit_btmm`, etc.). A
`symbol_table: Dict[tir.Var, int]` tracks loop var → GP register
bindings, and `ExprMaterializer` lowers dynamic `PrimExpr`s into
chains of ISA arithmetic instructions.

---

## 4. End-to-end trace: a P @ V `T.gemm`

```python
# User writes:
with T.Kernel(1, head_count) as (_, by):
    ...
    T.gemm(S_loc, V_sh, PV_loc)        # default KIND=overwrite
```

| Step | Pass | What changes in the IR |
|------|------|------------------------|
| 1 | `annotate_gemm_kind` | Wrap the `T.gemm` in `AttrStmt(plena.gemm_kind, "overwrite")`. |
| 2 | `annotate_group` | Wrap the `head_count` axis in `AttrStmt(plena.group, head_count)`. |
| 3 | `annotate_sync` | overwrite is not a sync site — skipped. |
| 4 | `split_lane_groups` | If `head_count > lane_count`, split into `by_outer × by_inner`. |
| 5 | `scope_inference` | Resolve `S_loc` / `V_sh` / `PV_loc` scopes. |
| 6 | `allocate_group_memory` | `S_loc` → ROW_STACK `(1, lane_count, 1, MLEN)`; `V_sh` / `PV_loc` → COL_PACK. |
| 7 | `lift_to_blocks` | Wrap the gemm Evaluate in its own BlockRealize, hoist `plena.gemm_kind` annotation onto `block.annotations`. |
| 8 | `graph_pipeline` (extract) | Recognise the surrounding `for by` + `plena.group(4)` + `tilelang_root` as a LaneGroup; the gemm becomes a non-sync GraphNode. |
| 9 | `graph_pipeline` (partition) | mv is per-lane (no `plena.sync` annotation, op is `tl.tileop.gemm_py` not in INHERENTLY_SYNC_EXTERNS); accumulates into a per-lane run; surrounding `plena.v_add` is sync and hoists out. |
| 10 | `graph_pipeline` (lower) | Calls `_lower_gemm`: KIND=overwrite + LHS rows=1 ⇒ pick `plena.mv`; per-lane `_auto_lane_offset` from `by`; per-lane run wrapped in `for by in range(lane_count)`. |
| 11 | `PlenaCodegen` | `plena.mv` → `Op(kind="mv", scalar_args=[by*64, by*16, by*16])`. |
| 12 | `AddressAllocationPass` | Concrete addresses for `S_loc` / `V_sh` / `PV_loc`. |
| 13 | `IsaEmitterPass` | Emit `M_MV` × tile_count + `M_MV_WO` writeback. |

---

## 5. Known gaps (ranked by severity)

### 5.1 ~~`lower_to_hlir` couples three concerns~~ — RESOLVED
The old `lower_to_hlir.run` interleaved (A) tile→plena translation,
(B) lane-fusion segmentation, (C) lane-offset projection, and (D)
attribute stripping in one recursive stmt walker. Adding a new op
required changes scattered across the call paths; `_segment_lane_for`
consumed the `plena.group(lane_count)` AttrStmt mid-walk, which is why
v2's attempt to factor (C) into a standalone post-pass failed.

**Resolution:** the back end has been replaced by `lift_to_blocks` +
`graph_pipeline` (see § 2.11–2.12). Each concern now has a clear home:

  * `lift_to_blocks` is the only pass that sees raw stmt structure;
    it wraps each op as its own BlockRealize, pulling plena.* AttrStmts
    into `block.annotations`.
  * `graph_pipeline` extracts a list of `GraphNode | tir.Stmt` from
    each lane group, partitions it on sync boundaries (concern B), and
    emits stmts via per-op `_lower_copy` / `_lower_gemm` helpers
    (concern A); per-lane vs. in_sync lowering naturally handles
    concern C; reading `block.annotations` handles D.
  * The `plena.group` AttrStmt is read once during lane-group extraction
    and never mutated, so it stays available for any future pass that
    wants to reason about lane structure.

Adding a new sync/per-lane plena op is now: add a lower fn + register
the op name in `INHERENTLY_SYNC_EXTERNS` (or
`PER_LANE_UNROLLED_EXTERNS`). No partitioner / walker changes.

### 5.2 ~~`annotate_sync` straddles two IR levels (dual handling)~~ — DOWNGRADED
The pass still inspects both tile-DSL forms (`T.copy` / `T.gemm`) and
lowered `plena.*` extern calls. The dual handling continues to be a
small papercut for adding new ops, but it's no longer load-bearing for
correctness now that the back end is decoupled — the graph back end
treats `plena.sync` annotations and `INHERENTLY_SYNC_EXTERNS` as a
unified "is this node a sync barrier?" predicate.

**Possible cleanup (optional):** narrow `annotate_sync` to look only at
tile-DSL forms (since pre-lowered plena.* externs are now classified by
the back end's intrinsic table). ~30 LoC, no urgency.

### 5.3 `fuse_elementwise` only supports `+`, `-`, `*`, `0` ★
Division and other ops (`/`, `exp`, `relu`, …) and non-zero constant
fills have no fuse rule. Add new ones by registering the corresponding
backend intrinsic + extending `fuse_elementwise._OP_TO_INTRIN`. ~20
LoC each.

> Resolved (partial): `+` (plena.v_add), `-` (plena.v_sub), `*`
> (plena.v_mul), and `0`-fill (plena.zero_v) are all supported.
> Backend's `emit_tile_binary` already routes to V_ADD_VV / V_SUB_VV /
> V_MUL_VV; the `_emit_v_binary` dispatch in `isa_pass.py` is shared
> across `_emit_v_add` / `_emit_v_sub` / `_emit_v_mul`.

### 5.4 `KIND="add"` is reserved but not yet implemented ★
`C += A @ B` — the most common attention-tail accumulation pattern.
The kind-name and the scratch-attr key are both reserved (kernel
authors can already write `with T.attr(0, KIND, "add"): T.gemm(...)`
without a "unknown kind" parser error), but the lowering raises
`NotImplementedError` to make the gap explicit. For now write the
two ops manually:

```python
scratch = T.alloc_fragment((rows, hlen), "float16")
T.gemm(A, B, scratch)                     # KIND=overwrite (default)
for r in T.serial(rows):
    for c in T.Parallel(C):
        dst[r, c] = dst[r, c] + scratch[r, c]   # auto-fuses to plena.v_add
```

**Planned implementation** (when prioritised):
1. `_lower_gemm` for `kind="add"` reads the scratch buffer's `tir.Var`
   from a surrounding `T.attr(scratch.data, "plena.gemm_scratch", 0)`
   AttrStmt.
2. Emit `plena.matmul(A, B, scratch, …)` (same offset / stride logic
   as `kind="overwrite"`).
3. Emit `plena.v_add(C, scratch, C)`.
4. Wrap both in a `tir.SeqStmt`.

Kernel author handles the scratch alloc explicitly — no inline
`tir.Allocate`, no codegen / address_alloc changes. ~30 LoC in
`_lower_gemm` once we wire it through.

### 5.5 ~~`[:, col]` slice form is unsupported~~ — CLOSED (TIR-level block)
The "natural" column-wise expression
`for col in T.Parallel(hlen): O[:, col] = O[:, col] + PV[:, col]` is
**not implementable** — it's blocked at the TVM TIR layer, not just
in our `fuse_elementwise`. Probed behaviour (with current tilelang +
TVM):

| Form | Result |
|------|--------|
| `dst[:, col] = …` | Tilelang parses but `assign_slice` lowering crashes (`(stop − start)` on `None`). |
| `dst[0:4, col] = …` | Rejected by TVM IR builder: *"Only the last index of a buffer access may be a vector type."* |
| `dst[0:4, 0:16] = …` | Same TIR-level rejection. |
| `dst[row, 0:16] = …` | ✓ Works — slice on the **last** dim is the only allowed vector form. |

The "all rows, single column" semantics (`[:, col]`) is fundamentally
unrepresentable in TIR — TIR's SIMD model assumes the inner-most dim
is the only one that can carry a vector. No desugar pass can reach
across that.

The viable last-dim-slice form (`for row in T.Parallel: dst[row, 0:C] = …`)
saves only one line vs. the explicit nested form already supported by
`_try_fuse_nested`, so we don't add a desugar rule for it either.
Stick with the explicit form:

```python
for row in T.serial(rows):
    for col in T.Parallel(C):
        dst[row, col] = lhs[row, col] + rhs[row, col]   # auto-fuses
```

### 5.6 ~~No single source of truth for buffer addresses~~ — RESOLVED
A real bug we hit: the FPRAM-address mismatch in flash_decode_min. The
addresses reported by `make_flash_*_min`'s `constants` dict and the
addresses actually assigned by `AddressAllocationPass` were computed
independently — testbench used the dict, kernel ran on TVM's, and the
two drifted by 64 words. Every write was "valid"; symptom was
head-1/2 numerical drift while heads 0/3 looked fine.

**Resolution:** the compiler CLI gained a `--dump-buffer-addrs <path>`
flag that writes the post-`AddressAllocationPass` table as JSON
(`{name: {scope, address, shape, dtype}}`). Testbenches read that JSON
to drive FPRAM preload offsets / VRAM comparison row indices, instead
of mirroring the allocation rule by hand. The hand-rolled
`_slot_addresses` / `_slot_bases` helpers and all `*_ADDR` fields in
the kernel factory's `constants` dict have been deleted from
`flash_attention_min` and `flash_decode_min` (their HLIR is the only
truth). When new kernels are added, follow the same pattern — never
re-introduce hand-rolled address mirrors.

### 5.7 `forbid_plena_extern` is opt-in, not default ★
Some unit tests intentionally write `T.call_extern("plena.fp_copy_at", …)`
to exercise specific intrinsics' lowering paths, so the sanity check
cannot be default-on. Consequence: a new kernel author who falls back
to `plena.*` extern won't get warned.

**Fix:** route tests through a bypass flag, default-on the sanity
check. ~20 LoC + test setUp edits.

### 5.8 Test coverage is uneven ★
115 frontend tests sounds like a lot, but most are per-pass unit tests.
End-to-end **behavioural** tests (compile + simulator + golden compare)
exist only for `tvm_flash_attention_min` and `tvm_flash_decode_min`.
New KINDs / new op-fusion rules have a narrow regression net.

**Fix:** add more small e2e kernels (mm64, single-layer LayerNorm,
single-layer RoPE, …), each driving the full pipeline + simulator.

### 5.9 `lower_compound_fp_stores` is a hot-fix-shaped pass ★
The tilelang frontend occasionally produces compound stores
(`arr[i] += x`); the triggering condition isn't documented. This pass
just splits them. If tilelang upstream changes, the pass may need to
extend or vanish.

**Fix:** document the trigger conditions on the pass docstring; or
push back on the frontend to never produce compound stores.

---

## 6. Already cleaned up (delivered in v1)

- ✓ User-facing surface is tilelang DSL only —
  `flash_attention_min` / `flash_decode_min` contain zero `plena.*`
  externs.
- ✓ KIND table converged to two-active + one-reserved (btmm / overwrite
  + add reserved-but-not-implemented); matmul-vs-mv split is
  compiler-internal.
- ✓ Per-head offsets auto-injected.
- ✓ `dst_row_stride` auto-computed (correct for both COL_PACK and
  ROW_STACK).
- ✓ KIND=overwrite's idempotent lane marking subsumes both
  DMA-driven matmul and fragment-only matmul use cases.
- ✓ `fuse_elementwise` nested-fold rule (so zero / v_add aren't
  redundantly run by an outer serial loop).
- ✓ `fuse_elementwise` recognises `+` / `-` / `*` (→ plena.v_add /
  plena.v_sub / plena.v_mul) and `0`-fill (→ plena.zero_v).
- ✓ Buffer addresses single-source-of-truth via the compiler's
  `--dump-buffer-addrs` JSON; hand-rolled `*_ADDR` mirrors removed
  from the flash kernel factories.
- ✓ ASM byte-identical to the legacy hand-written `plena.*` extern path
  for flash_decode_min; semantically equivalent (op counts match) for
  flash_attention_min.
- ✓ `forbid_plena_extern` opt-in sanity check available.

---

## 7. Recommended next steps (by priority)

1. **§ 5.4 — finish KIND="add" lowering** — interface and scratch-attr
   key are reserved; ~30 LoC in `_lower_gemm` to wire it through.
2. **§ 5.8 — e2e tests** — cheapest insurance per LoC.
3. **§ 5.2 — narrow `annotate_sync` to tile-DSL only** — minor
   cleanup, ~30 LoC, no longer load-bearing now that the graph back
   end has its own sync classification table.
4. **§ 5.7 / 5.9** — minor cleanup, do as time allows.

(§ 5.1 closed: graph back end (`lift_to_blocks` + `graph_pipeline`)
now separates the four concerns the old walker conflated.)
(§ 5.5 closed: blocked at TIR layer, not actionable.)
(§ 5.6 closed: addresses single-source-of-truth via `--dump-buffer-addrs`.)
