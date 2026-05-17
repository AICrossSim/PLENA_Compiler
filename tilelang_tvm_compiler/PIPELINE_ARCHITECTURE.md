# Pipeline Architecture

End-to-end walkthrough of how `tilelang_tvm_compiler` lowers a user-written
`@T.prim_func` to PLENA ISA, with notes on each pass's responsibilities,
inter-pass dependencies, and known gaps.

---

## 1. Overview

```
@T.prim_func (user's tilelang DSL kernel)
        │
        │  Frontend (frontend/pipeline.py)
        │   1. Stmt prep: inline_let_stmts, lower_compound_fp_stores
        │   2. lift_from_raw_primfunc (TIR → graph_ir.Graph)
        │   3. Graph-IR passes (annotate, lower, fuse, ...)
        │   4. materialize_to_primfunc (Graph → TIR)
        │   5. _rewrite_buffer_scopes (shared.dyn → vram, etc.)
        ▼
TIR with plena.* extern calls only
(plena.matmul / plena.mv / plena.btmm / plena.zero_v / plena.v_add /
 plena.dma_h2v_slice / plena.copy_v_to_v / plena.row_load_v_to_fp / …)
        │
        │  Backend (pipeline.compile_kernel)
        │   PlenaCodegen.lower_to_hlir()
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

**Architectural principles:**

1. **All semantic structure work lives on the Graph IR.** Per-op metadata
   (sync, gemm kind, lane layout) is stored as `attrs` on `GraphNode` /
   `BufferNode` / `NestedForGroup` / `ForRoot` — not as `T.attr(...)`
   AttrStmts in a TIR tree. Passes are pure `Graph → Graph` functions.
2. **User-facing surface is tilelang DSL only** — `T.gemm` / `T.copy` /
   `T.Parallel` / `T.alloc_*` / `T.attr(0, "plena.gemm_kind", ...)`.
   `plena.*` is a compiler-internal IR namespace; kernel authors must
   not write it directly.
3. **Per-head offsets are auto-injected.** The user writes
   `T.gemm(buf, buf, buf)`; the compiler infers each operand's lane-axis
   stride from its post-expansion shape (`expand_buffers`).
4. **Buffer-shape decisions happen at materialize time, not mid-pipeline.**
   This is the key architectural shift versus the old stmt-walker
   pipeline: graph optimizations run on un-expanded (logical 2D) shapes;
   the lane axis is added once, at the boundary into TIR. Future
   optimizations that want to change buffer shape (double-buffering,
   dead-temp elim, etc.) are unblocked.

---

## 2. Frontend pipeline

The full chain from `frontend/pipeline.py:compile_func`:

```
TIR (user)
  │  inline_let_stmts                  (stmt walker, IR cleanup)
  │  lower_compound_fp_stores          (stmt walker, expression-level)
  │  lift_from_raw_primfunc            ← into the Graph IR
  ▼
Graph
  │  graph_passes.annotate_grid        (ATTR_GROUP_EXTENT)
  │  graph_passes.annotate_sync        (ATTR_IS_SYNC)
  │  graph_passes.split_lane_groups    (extent>lane → outer × inner)
  │  graph_passes.lift_lane_groups     (ForRoot → LaneGroup)
  │  graph_passes.fuse_elementwise     (T.Parallel → plena.v_*)
  │  graph_passes.scope_inference      (BufferScopeMap)
  │  graph_pipeline.materialize_to_primfunc(expand_lane_buffers=True):
  │      graph_passes.allocate_group_memory.analyze   (ATTR_LANE_LAYOUT)
  │      graph_passes.expand_buffers.expand           (rebuild tir.Buffer)
  │      graph_passes.lower_fp_row_patterns           (fp_*_at / row_*_at)
  │      _partition_and_materialize                    (curtain bundle)
  ▼
TIR (with plena.* externs, lane-expanded buffers, tilelang scopes)
  │  _rewrite_buffer_scopes            (shared.dyn → vram, etc.)
  ▼
TIR (fully lowered, physical scopes — backend input)
```

### 2.1 Stmt prep (pre-graph)

#### `inline_let_stmts`
Inlines `let x = expr in body` LetStmts. Pure IR cleanup, no semantic
change. Kept on stmt level because it's pre-everything-else and trivial.

#### `lower_compound_fp_stores`
Rewrites `arr[i] = a*b + c*d` (compound RHS) into a sequence of single-op
stores using auto-allocated `__tmp_fp_*` temporaries. **Must run before
lift** because it operates on TIR expression trees; once lift folds ops
into call form, the RHS expression structure is no longer accessible.

### 2.2 Lift to Graph IR (`lift_from_raw.py`)

`lift_from_raw_primfunc(func) → Graph`. Single shot. Translates:

| TIR construct | Graph IR |
|---|---|
| `T.launch_thread("blockIdx.x", N>1)` | `ForRoot(loop_var, extent=N)` |
| `threadIdx.*` or `blockIdx.*` extent==1 | dropped (degenerate) |
| `tilelang_root` BlockRealize body | `NodeRoot(items=[...])` |
| `Evaluate(Call(tl.tileop.copy/gemm_py/reduce))` | `GraphNode(op_call, reads, writes)` |
| `Evaluate(Call(tir.call_extern("plena.*")))` | `GraphNode(op_call)` (already-lowered passthrough) |
| `tir.For` (any kind) | `NestedForGroup(loop_var, kind, items)` |
| `BufferStore` / `LetStmt` / `IfThenElse` | `RawStmt(stmt)` (escape hatch) |
| `with T.attr(0, "plena.gemm_kind", "btmm"): T.gemm(...)` | `GraphNode(attrs={ATTR_GEMM_KIND: "btmm"})` (the AttrStmt is absorbed) |

Concurrently, `_collect_buffers` walks every `Block.alloc_buffers` and
`func.buffer_map` and builds `Graph.buffer_nodes: dict[str, BufferNode]`
(name → node with `shape`, `dtype`, `declared_scope`, `data_var`).

Each `GraphNode.reads / .writes` is a list of `BufferAccess(buffer_name,
starts, extents)` — references the BufferNode by name, not by direct
`tir.Buffer` reference. Layout rewrites in `expand_buffers` flow through
the BufferNode without per-region mutation.

### 2.3 Graph-IR passes (shape-agnostic phase)

Each pass takes a `Graph` and returns a `Graph`. None of them change
buffer shapes — those decisions are deferred to materialize.

#### `annotate_grid` (`graph_passes/annotate_grid.py`)
Sets `ATTR_GROUP_EXTENT` on every `ForRoot` (which came from a
`blockIdx.* > 1` binding) and on every `NestedForGroup` whose `kind ==
PARALLEL` (came from `T.Parallel`). Also rewrites the parallel kind to
SERIAL (PLENA hardware is single-threaded; the group annotation is what
signals "iterations are lane-fusion-eligible" to downstream passes).

#### `annotate_sync` (`graph_passes/annotate_sync.py`)
Sets `ATTR_IS_SYNC` on every GraphNode. True iff:
- HBM ↔ local-buffer DMA copy
- VRAM ↔ FPRAM rank-1 copy (S_MAP_*_*)
- VRAM ↔ VRAM copy (V_ADD_VF f0=0)
- gemm with `ATTR_GEMM_KIND == "btmm"`
- already-lowered `plena.*` extern in `INHERENTLY_SYNC_EXTERNS`

The classification table is in [graph_pipeline.py](frontend/passes/graph_pipeline.py#L52)
(`INHERENTLY_SYNC_EXTERNS` / `PER_LANE_UNROLLED_EXTERNS`).

> **Important invariant:** any pass that *creates* new GraphNodes must
> set `ATTR_IS_SYNC` itself if the new node is in
> `INHERENTLY_SYNC_EXTERNS`. `annotate_sync` runs once and never sees
> later-created nodes. `fuse_elementwise` sets it on the
> `plena.zero_v` / `plena.v_*` it creates — see [bug fix](#bug-history)
> for the consequence of forgetting.

#### `split_lane_groups` (`graph_passes/split_lane_groups.py`)
For every `ForRoot` / `NestedForGroup` carrying `ATTR_GROUP_EXTENT > lane_count`
where the body recursively contains a sync GraphNode that references the
loop var: rewrite as `outer(extent/lane_count) × inner(lane_count,
ATTR_IS_LANE_FOR=True)`. The body's references to `v` get substituted
with `v_outer * lane_count + v_inner` via `_GraphVarSubst`, which walks:
- every `GraphNode.op_call` (TIR Call, recursively)
- every `BufferAccess.starts` / `.extents`
- every `RawStmt.stmt` (recursively, via inlined `_StmtVarSubst`)
- every nested `NestedForGroup.min` / `.extent`

#### `lift_lane_groups` (`graph_passes/lift_lane_groups.py`)
ForRoots / inner-of-pair NestedForGroups carrying `ATTR_GROUP_EXTENT ==
lane_count` (or `ATTR_IS_LANE_FOR=True`) get upgraded to `LaneGroup`
nodes — the explicit container for "one lane fusion bundle" that the
materialize-time partitioner identifies as the curtain-bundle scope.

Without this upgrade, materialize would emit each lane group as a plain
`tir.For` with no per-lane partitioning — all ops would run inside the
for-by, including the sync ones (wrong).

#### `fuse_elementwise` (`graph_passes/fuse_elementwise.py`)
Pattern-matches `NestedForGroup(ATTR_GROUP_EXTENT, items=[RawStmt(BufferStore)])`
forms and replaces them with single GraphNodes:

| Pattern | Replacement |
|---|---|
| `for i: dst[..., i] = a[..., i] + b[..., i]` | `plena.v_add(a, b, dst)` |
| `for i: dst[..., i] = a[..., i] - b[..., i]` | `plena.v_sub(a, b, dst)` |
| `for i: dst[..., i] = a[..., i] * b[..., i]` | `plena.v_mul(a, b, dst)` |
| `for i: dst[..., i] = 0` | `plena.zero_v(dst)` |

Plus the **nested fold**: outer serial-for wrapping a single fused
whole-buffer op → drops the outer for entirely (since `plena.v_*` and
`plena.zero_v` are inherently whole-buffer; running them N times is
wrong).

The created GraphNodes are tagged `ATTR_IS_SYNC=True` because they're
all in `INHERENTLY_SYNC_EXTERNS`.

#### `scope_inference` (`graph_passes/scope_inference.py`)
Owns `BufferScopeMap = dict[str, str]` and `ScopeInferenceError`. Walks
every GraphNode and infers each buffer's physical scope:

| Declared scope + usage | Resolved physical scope |
|---|---|
| `func.buffer_map` (param) | `hbm` |
| `global.<phys>` | as-declared |
| `shared.dyn`, used as gemm RHS / matmul-arg[1] | `mram` |
| Other `shared.dyn` | `vram` |
| `local.fragment`, used as `plena.fp_*_at` / `row_*_at` operand, OR rank-1, OR T.reduce dst | `fpram` |
| Other `local.fragment` | `vram` |

### 2.4 Materialize (shape-aware phase + lowering)

`materialize_to_primfunc(graph, scopes, expand_lane_buffers=True)` —
defined in [graph_pipeline.py](frontend/passes/graph_pipeline.py).
Runs three more graph passes that need final shape decisions, then
walks the graph to emit TIR.

#### `allocate_group_memory.analyze`
Classifies every buffer touched by lane-fused ops. Sets
`BufferNode.attrs[ATTR_LANE_LAYOUT]` to one of:

| Layout | Pre-expansion shape | Post-expansion shape | Triggered by |
|---|---|---|---|
| `col_pack` | `(rows, last)` | `(1, rows, lane_count, last)` | BTMM args[0,1]; non-btmm matmul args[1,2]; HBM↔local DMA local side; plena.v_*/matmul/row_* trailing Var args |
| `row_stack` | `(rows, last)` | `(1, lane_count, rows, last)` | BTMM args[2]; non-btmm matmul args[0] |
| `fp_lane` | `(N,)` | `(lane_count, N)` | VRAM↔FPRAM rank-1 copy fpram side; plena.fp_*_at / row_*_at FP operands; BufferStore to FPRAM |

Conflict rules: `row_stack` wins over `col_pack` (BTMM output's BHSD
layout dictates); `fp_lane` doesn't mix with the other two.
`global.*`-scoped buffers are skipped (user already wrote the physical
shape).

Also writes `BufferNode.attrs[ATTR_LANE_VAR]` (string name of the lane
var that this buffer's lane axis substitutes for).

#### `expand_buffers.expand`
For every BufferNode with `ATTR_LANE_LAYOUT`, rebuilds a fresh
`tir.Buffer` with the expanded shape, then walks the whole graph
rewriting:
- every `GraphNode.op_call` arg referencing the old buffer (BufferLoad
  inside `tl.tileop.region`, trailing `buf.data` Var args, etc.) →
  swap to new buffer + fold lane index into the indices via
  `_StmtRewriter._fold_lane`
- every `BufferAccess(name, starts, extents)` → starts get `_fold_lane`
  applied, extents get a unit slot inserted at the lane axis
- every `RawStmt(stmt)` → `_StmtRewriter.visit(stmt)` does the same
  swap+fold for any BufferLoad/BufferStore/Var inside

Index folding rules (mirrors the buffer-shape changes):

| Mode | Pre-fold indices | Post-fold indices |
|---|---|---|
| `col_pack` | `[r, c]` | `[0, r, lane_var, c]` |
| `row_stack` | `[r, c]` | `[0, lane_var, r, c]` |
| `fp_lane` | `[r]` | `[lane_var, r]` |

Critical detail: `lane_var` is the **actual** `tir.Var` from the
surrounding ForRoot/LaneGroup, not a fresh same-named Var. TIR resolves
vars by identity; using a synthetic var produces "unbound symbol"
errors at codegen time.

#### `lower_fp_row_patterns`
Must run **after** expand because the row-parallel pattern matcher
requires the 4D-expanded buffer shape (matches legacy stmt-walker
ordering). Three pattern families:

| Source | Replacement |
|---|---|
| `RawStmt(BufferStore)` to FPRAM buffer | GraphNode `plena.fp_zero_at` / `fp_copy_at` / `fp_add_at` / `fp_sub_at` / `fp_mul_at` / `fp_exp_at` / `fp_reci_at` |
| `NestedForGroup(ATTR_GROUP_EXTENT, items=[RawStmt(BufferStore)])` on VRAM | GraphNode `plena.row_exp_at` / `row_sub_fp_at` / `row_mul_fp_at` |
| `GraphNode(tl.tileop.reduce)` with VRAM src + FPRAM dst | `RawStmt(For row in N: Evaluate(plena.row_reduce_max_at / row_reduce_sum_at))` (escape hatch — the per-row for has no graph-IR analogue) |

#### `_partition_and_materialize` (curtain bundle)
Walks the (now expanded + lowered) graph and emits TIR.

For a `LaneGroup`: scan items, partition at sync boundaries:
- **Sync GraphNode**: flush any accumulated per-lane run, emit op once
  with `in_sync=True` (no surrounding for-by — it's a multi-lane HW
  instruction).
- **Per-lane GraphNode**: accumulate into the current per-lane run.
- **NestedForGroup with no inner sync**: accumulate as opaque per-lane
  block.
- **NestedForGroup with inner sync**: flush per-lane run, recurse into
  body, wrap result in `tir.For(loop_var)`.
- **RawStmt**: accumulate as per-lane.

When the per-lane run flushes, it gets wrapped in
`for(lane_var, range(lane_count))` — `UNROLLED` kind if any item is in
`PER_LANE_UNROLLED_EXTERNS` (currently just `plena.matmul`), else
`SERIAL`.

For a `NodeRoot` (no lane fusion, e.g. mm64-style): items emit as a
plain stmt sequence with `lane_var=None`.

For `ForRoot`: recursively materialize the body, then wrap in `tir.For`.

Each `GraphNode._lower_node` delegates to
[`lower_to_hlir._lower_copy / _lower_gemm`](frontend/passes/lower_to_hlir.py)
for the actual `tl.tileop.copy → plena.dma_*` and `tl.tileop.gemm_py →
plena.btmm/matmul/mv` translation. Already-lowered `tir.call_extern`
nodes pass through unchanged.

### 2.5 Final scope rewrite (post-graph)

`_rewrite_buffer_scopes` (in `lower_to_hlir.py`, despite the misleading
filename — see § 5.4): substitutes every `shared.dyn` /
`local.fragment` declared scope with the resolved physical scope from
`scopes`. Rebuilds `tir.Buffer` objects so backend codegen reads
`buf.scope() ∈ {hbm, vram, mram, fpram, global.*}` directly.

This step is intentionally outside the graph layer — graph passes use
declared (tilelang) scopes; the codegen-facing physical scope rename
is the boundary into backend.

---

## 3. Backend — three stages

(Not part of `frontend/pipeline.py`; same `compile_kernel` flow; see
[`tilelang_tvm_compiler/pipeline.py`](pipeline.py).)

### 3.1 `PlenaCodegen.lower_to_hlir()` ([codegen.py](codegen.py))
TIR → HLIR data structure.
- `_collect_param_buffers` + `_collect_alloc_buffers` walk every buffer
  into `_buffers` (keyed by `tir.Var`, i.e. `buffer.data`).
- `_walk_stmt` / `_walk_evaluate` rewrite each `plena.*` extern call to
  `_hlir.Op(kind="<name without 'plena.' prefix>", buffer_args=[...],
  scalar_args=[...])`.
- For-loops become `_hlir.Op(kind="for", body=[...])` nests.
- Output is `_hlir.HLIRModule(name, buffers, ops)`.

### 3.2 `AddressAllocationPass` ([address_alloc.py](address_alloc.py))
Assigns each buffer a concrete address in declaration order:
- HBM: from `0`, advance by buffer size.
- VRAM: from `0`, advance by buffer size (each row is MLEN-wide).
- MRAM: tile-aligned allocation.
- FPRAM: from `FPRAM_USER_BASE = 32`, advance by buffer size.

### 3.3 `IsaEmitterPass` ([isa_pass.py](isa_pass.py))
HLIR → ISA text. Each op kind has a `_emit_*` method. A
`symbol_table: Dict[tir.Var, int]` tracks loop var → GP register
bindings; `ExprMaterializer` lowers dynamic `PrimExpr`s into chains of
ISA arithmetic instructions.

---

## 4. End-to-end trace: a P @ V `T.gemm`

```python
# User writes:
with T.Kernel(1, head_count) as (_, by):
    ...
    T.gemm(S_loc, V_sh, PV_loc)        # default KIND="overwrite"
```

| Step | Pass | What changes |
|------|------|---|
| 1 | `inline_let_stmts`, `lower_compound_fp_stores` | TIR cleanup; no change to this gemm. |
| 2 | `lift_from_raw_primfunc` | `T.gemm(...)` → `GraphNode(op_call=tl.tileop.gemm_py(...), reads, writes)`. The blockIdx.y binding becomes a `ForRoot(by, extent=head_count)`. |
| 3 | `annotate_grid` | `ForRoot(by).attrs[ATTR_GROUP_EXTENT] = head_count`. |
| 4 | `annotate_sync` | gemm has no `ATTR_GEMM_KIND="btmm"`, so `attrs[ATTR_IS_SYNC] = False`. |
| 5 | `split_lane_groups` | If `head_count > lane_count`, split the ForRoot into `by_outer × by_inner`; vars in op_call args / BufferAccess get rewritten. |
| 6 | `lift_lane_groups` | Inner ForRoot (extent=lane_count, ATTR_IS_LANE_FOR=True) → `LaneGroup(lane_var=by_inner, lane_count=4, items=[...])`. |
| 7 | `scope_inference` | `S_loc`/`V_sh`/`PV_loc` resolve (S_loc → vram, V_sh → mram, PV_loc → vram). |
| 8 | `materialize`: `allocate_group_memory.analyze` | Non-btmm gemm: `S_loc → row_stack`, `V_sh → col_pack`, `PV_loc → col_pack` (only if untouched by other ops). |
| 9 | `materialize`: `expand_buffers.expand` | `S_loc.shape: (64, 64) → (1, lane_count, 64, 64)`; `V_sh / PV_loc: (64, 16) → (1, 64, lane_count, 16)`. Op_call indices get `_fold_lane` applied. |
| 10 | `materialize`: `_partition_and_materialize` | gemm is per-lane (not sync). Accumulates into per-lane run wrapped in `for(by_inner)`. |
| 11 | `_lower_node → _lower_gemm` | KIND=overwrite + LHS rows=1 ⇒ `plena.mv`; per-lane offsets auto from `by_inner * stride`. |
| 12 | `_rewrite_buffer_scopes` | `shared.dyn` → `mram` for V_sh; `local.fragment` → `vram` for S_loc / PV_loc. |
| 13 | `PlenaCodegen` | `plena.mv` → `Op(kind="mv", scalar_args=[by*64, by*16, by*16])`. |
| 14 | `AddressAllocationPass` | Concrete addresses for `S_loc` / `V_sh` / `PV_loc`. |
| 15 | `IsaEmitterPass` | Emit `M_MV` × tile_count + `M_MV_WO` writeback. |

---

## 5. Known gaps

### 5.1 Loop-scheduling layer not started
Graph IR has structure for it (`ForRoot.attrs`, `NestedForGroup.attrs`,
`Graph.buffer_nodes` whose shape can be mutated mid-pipeline) but no
pass actually does loop optimization. The migration plan §"Phase D"
covers what's planned:

- **Cross-iter DMA merge**: combine DMAs of consecutive K/V tiles into
  a single bigger DMA. Reduces per-tile DMA setup cost.
- **Double-buffering / prefetch**: alloc `K_sh_alt`, prefetch tile `k+1`
  while computing on tile `k`. Hides DMA latency.

Both depend on hardware capability info we haven't pinned down (max
single-DMA element count, MRAM/VRAM capacity headroom, DMA stride
support — see `MIGRATION_PLAN.md` §"Open questions").

### 5.2 RawStmt is the graph layer's blind spot
`RawStmt` wraps a TIR subtree the lift can't classify (`IfThenElse`,
LetStmt-leftovers, BufferStore inside non-lane-eligible serial fors,
T.reduce that lower_fp_row_patterns turned into `for row: row_reduce_*`).
Graph passes can do mechanical var-subst and buffer-replace inside it
(`_StmtVarSubst`, `_StmtRewriter.visit`) but cannot reason about its
control flow.

If/when we add proper graph-IR nodes for `IfThenElse` and `Reduce`,
RawStmt usage shrinks; until then it's an escape hatch for "this shape
is rare, just pass it through".

### 5.3 `fuse_elementwise` op set
Currently: `+` `-` `*` `0`-fill. No `/`, no `exp`, no `relu`, no
non-zero const fill. Adding new ones requires a backend intrinsic plus
extending `fuse_elementwise._OP_TO_INTRIN`. ~20 LoC each.

### 5.4 `KIND="add"` reserved but not wired
`C += A @ B` in one gemm. Recognised by the kind parser (no error) but
`_lower_gemm` raises `NotImplementedError`. Workaround:

```python
scratch = T.alloc_fragment((rows, hlen), "float16")
T.gemm(A, B, scratch)                           # KIND=overwrite
for r in T.serial(rows):
    for c in T.Parallel(C):
        dst[r, c] = dst[r, c] + scratch[r, c]   # auto-fuses to plena.v_add
```

### 5.5 `lower_to_hlir.py` is a misleading filename
Despite the name, [frontend/passes/lower_to_hlir.py](frontend/passes/lower_to_hlir.py)
is **not** the TIR → HLIR backend — that's `PlenaCodegen.lower_to_hlir()`
in `codegen.py`, despite sharing the method name. The file holds three
unrelated frontend op-level helpers:

- `_lower_copy`: `tl.tileop.copy → plena.dma_*`
- `_lower_gemm`: `tl.tileop.gemm_py → plena.btmm/matmul/mv`
- `_rewrite_buffer_scopes`: tilelang scope → physical scope (used as
  the very last frontend step)

Renaming this file to e.g. `op_lowering.py` or `dsl_lowering.py` would
remove a recurring confusion source. Not done yet because it's a
renames-everywhere change.

### 5.6 `forbid_plena_extern` is opt-in
A sanity check that asserts a kernel uses only tilelang DSL (no
`plena.*` extern calls). Currently kernel authors must invoke it
manually before calling `compile_func`. Could be wired in by default
under a flag.

---

## 6. Bug history (lessons learned the hard way) {#bug-history}

### `fuse_elementwise` missed `ATTR_IS_SYNC` on created nodes
**Symptom**: flash_attention_min e2e numerics off by exactly `lane_count`
(simulated ≈ 4× golden) for the largest output magnitudes.

**Root cause**: `fuse_elementwise` runs **after** `annotate_sync`. When
it created new `plena.zero_v` / `plena.v_add` GraphNodes, it left
`attrs={}`. `annotate_sync` had already finished and didn't see them.
The materialize-time partitioner saw `ATTR_IS_SYNC=False` (default) and
emitted these `INHERENTLY_SYNC_EXTERNS` ops *inside* the per-lane
`for(by_i)`, running each `O += PV` four times instead of once.

**Fix**: `fuse_elementwise` now sets `attrs={ATTR_IS_SYNC: True}` on
every new node. Same invariant applies to any future pass that creates
inherently-sync ops.

**General principle**: any pass that creates new GraphNodes is
responsible for setting their attrs correctly — `annotate_*` passes
don't re-run.

---

## 7. Phase status

- ✅ **Phase A**: graph IR types, lift_from_raw, walker helpers
- ✅ **Phase B**: incremental graph-layer passes (annotate_gemm_kind,
  annotate_sync, scope_inference initially as proof of concept)
- ✅ **Phase C.1**: write all graph-layer passes; switch pipeline to
  graph path
- ✅ **Phase C.2**: delete legacy stmt-walker passes, frontend_legacy/,
  6 stmt-walker test files; only graph path remains
- ⏳ **Phase D**: loop-scheduling layer (DMA merge, prefetch / double-buffer)
  — not started; depends on HW capability info

See [`MIGRATION_PLAN.md`](MIGRATION_PLAN.md) for the original migration
plan and open questions.
