# Migration Plan: All-Graph-Layer Frontend

This document captured the target architecture for the frontend after
fully migrating from the original "stmt-walker chain + thin graph layer"
to "minimal stmt prep + full graph layer". Each section below describes
the target form, the migration steps, and the rationale for the design
choices.

**Status: Phases A / B / C.1 / C.2 complete.** The all-graph-layer
pipeline is the only path; legacy stmt-walker passes and
`frontend_legacy/` have been deleted. See `PIPELINE_ARCHITECTURE.md`
for the architecture as it stands. **Phase D (loop scheduling — DMA
merge, prefetch / double-buffer) is the next milestone, not started**;
it depends on hardware capability data that is still TBD (see § Open
questions).

> **Reading guide.** This document is now mostly retrospective — the
> "current pipeline" / "stmt walker" passages below describe the
> pre-migration state for context. For what the pipeline *currently*
> does, read `PIPELINE_ARCHITECTURE.md`. The Phase-D plan and the
> still-relevant Open Questions at the bottom are the only forward-
> looking parts.

---

## Why we're doing this

Current frontend is a chain of stmt-walker passes that communicate via
`T.attr(0, "plena.*", ...)` AttrStmts. Each pass re-walks the IR and
mutates it. Adding a new analysis means another walker; adding a new
fusion rule means coordinating attr ordering; adding new ops requires
multiple touchpoints scattered across passes.

In the graph layer, each op is a `GraphNode` with `attrs` — passes read
and write attrs directly. `reads` / `writes` are filled at lift time
and live on the node, so any pass can do data-flow analysis without
re-walking stmt trees. New analyses are pure functions on `Graph`. New
fusion rules are pattern match + node replace.

Architectural insight (key): **buffer layout decisions belong AT the
end of graph optimization, not as a separate stmt pass run before it.**
If `allocate_group_memory` runs before the graph layer, every graph
optimization that wants to change buffer shape (double-buffering for
prefetch, eliminating dead temps, etc) is locked out. The plan moves
buffer-shape allocation into `materialize`, where it has full
visibility of the post-optimization graph.

---

## Target pipeline shape

```
                     [STMT STAGE — minimal]
@T.prim_func
   │
   ▼
inline_let_stmts                    # TIR housekeeping (LetStmt → subst)
lower_compound_fp_stores            # arr[i]=a*b+c*d → temp → temp → out
   │
   ▼
                     [LIFT TO GRAPH]
lift_from_raw_primfunc              # raw PrimFunc → Graph
                                    # (one shot — no longer post-stmt-walker
                                    #  preprocessing required)
   │
   ▼
                     [GRAPH STAGE — analysis / annotation, no side effects]
graph.annotate_grid                 # grid bindings → ForNode attrs
                                    # (replaces stmt-walker annotate_group)
graph.annotate_sync                 # ATTR_IS_SYNC on each GraphNode
                                    # (current stub already runs)
graph.scope_inference               # BufferNode.physical_scope
                                    # (current implementation already exists,
                                    #  not wired)
graph.annotate_gemm_kind            # ATTR_GEMM_KIND on gemm nodes
                                    # (already done by lift_from_raw via
                                    #  KIND AttrStmt absorption)
   │
   ▼
                     [GRAPH STAGE — pattern fusion / canonicalisation]
graph.fuse_elementwise              # for+BufferStore patterns → plena.v_*
                                    # (replaces stmt fuse_elementwise)
graph.lower_fp_row_patterns         # FPRAM row-element idioms → plena.fp_*_at
                                    # (replaces stmt lower_fp_row_patterns)
   │
   ▼
                     [GRAPH STAGE — schedule transforms]
graph.split_lane_groups             # head_count > lane_count axis split
                                    # (replaces stmt split_lane_groups; now
                                    #  operates on ForNode + lane-attribute
                                    #  rather than for-rewriting + var subst)
   │
   ▼ (future)
                     [GRAPH STAGE — real fusion (Phase D)]
graph.dma_merge                     # cross-iteration DMA combining
                                    # (depends on HW capability data:
                                    #  max single-DMA size, etc)
graph.prefetch                      # double-buffer K/V across kv_block
                                    # (depends on HW: NO async DMA in PLENA,
                                    #  so this would be reorder-only, not
                                    #  overlap)
   │
   ▼
                     [MATERIALIZE]
graph.materialize:
  1. Resolve each BufferNode's final physical layout:
       - apply ATTR_LANE_LAYOUT (col_pack / row_stack / fp_lane) →
         expand shape with lane dimension;
       - check `global.*` scopes (no lane expansion);
       - account for any layout decisions from real fusion passes
         (e.g. double-buffer doubles the lane-axis extent).
  2. Build new tir.Buffer objects with finalized shapes.
  3. Walk graph items, lower each GraphNode to a tir.Stmt
     (delegates to existing lower_to_hlir helpers _lower_copy /
     _lower_gemm; their input is per-op call info, no IR walking).
  4. Wrap per-lane runs in `for(lane_var)` (the current
     graph_pipeline._partition_and_materialize logic moves here
     intact — it's already operating on the graph, no rewrite needed).
  5. Emit final tir.PrimFunc.

   │
   ▼
              [BACKEND — unchanged from today]
PlenaCodegen.lower_to_hlir() (TIR → HLIR)
AddressAllocationPass
IsaEmitterPass
   │
   ▼
.asm
```

---

## graph_ir extensions needed

**Already done (Phase A)**:
* `BufferNode(name, shape, dtype, declared_scope, physical_scope, data_var, attrs)`
* `ForNode(loop_var, min, extent, kind, thread_binding, body_items, attrs)`
* Attr keys: `ATTR_IS_SYNC`, `ATTR_GEMM_KIND`, `ATTR_GROUP_EXTENT`,
  `ATTR_IS_LANE_FOR`, `ATTR_LANE_LAYOUT`, `LAYOUT_COL_PACK`,
  `LAYOUT_ROW_STACK`, `LAYOUT_FP_LANE`.

**To add (Phase C.1)**:
* `Graph.buffer_nodes: dict[str, BufferNode]` — every alloc'd buffer
  AND every param buffer becomes a BufferNode. GraphNode.reads/writes
  reference these by name (not by tir.BufferRegion directly), so layout
  changes propagate without rewriting reads/writes.
* `GraphNode.reads/writes` semantics shift: today `tir.BufferRegion`
  carrying a `tir.Buffer` reference. Tomorrow: `BufferAccess(buffer_name,
  starts, extents)` referencing `Graph.buffer_nodes[buffer_name]`.

**Open**: do we keep the current `LaneGroup` / `NestedForGroup` /
`NodeRoot` / `ForRoot` distinction, or unify into a single recursive
`ForNode + items` representation? Probably unify — once
`graph.split_lane_groups` operates on ForNode directly, the
distinction adds little value.

---

## Per-pass migration notes

### inline_let_stmts (stmt stage, unchanged)
Pure TIR housekeeping. No reason to move.

### lower_compound_fp_stores (stmt stage, unchanged for now)
Decomposes compound FP store RHS into sequential single-op stores plus
auto-allocated `__tmp_fp_*` buffers. Operates on `tir.BufferStore`
trees with `.value` arithmetic — best done while we still have
expression-level IR, before lift collapses ops into op-call form.
Future improvement: this could happen in graph layer too, but the
benefit is minor.

### annotate_gemm_kind (DONE — removed from pipeline)
User writes `with T.attr(0, KIND, "btmm"): T.gemm(...)`; the AttrStmt
sits in raw IR. `lift_from_raw_primfunc._items_from_stmt` peels KIND
AttrStmts and writes `ATTR_GEMM_KIND` directly on the gemm GraphNode.
No-KIND gemm sites default to `"overwrite"` at lower time (in
`graph_pipeline._lower_node`).

### annotate_group → graph.annotate_grid
Stmt walker today identifies grid bindings (`thread_extent` AttrStmts
+ `T.Parallel` for-loops) and wraps them in
`for v: T.attr(0, "plena.group", N): ...`.

In the graph layer this becomes: walk every `ForNode`, set
`ATTR_GROUP_EXTENT` based on whether the for came from a grid binding
or a `T.Parallel` (lift_from_raw already records this). No IR rewrite
— just attr setting.

`lift_from_raw` currently produces `ForRoot` for grid bindings. Phase
C.1 should record grid-axis extent on the `ForRoot` directly, and
graph passes consume that.

### annotate_sync → graph.annotate_sync (DONE — `graph_passes/annotate_sync.py`)
Already exists. Sets `ATTR_IS_SYNC` on each GraphNode by inspecting
op kind, op-call args, ATTR_GEMM_KIND. Currently runs alongside the
stmt-walker version (which is still required because
`split_lane_groups` reads stmt-walker's `plena.sync` AttrStmts).

In Phase C, after `split_lane_groups` migrates, the stmt-walker
version goes away; `graph.annotate_sync` is the only path.

### split_lane_groups → graph.split_lane_groups
Stmt walker today: when a grid axis has extent > lane_count and is
sync-eligible, splits the for into outer × inner with var subst.

In graph layer:
1. Find ForRoots/ForNodes whose extent > lane_count and whose body
   contains a sync GraphNode (via ATTR_IS_SYNC, walking the items list).
2. Replace that ForNode with two new ForNodes: outer (extent =
   original / lane_count) wrapping inner (extent = lane_count, with
   `ATTR_IS_LANE_FOR=True`).
3. Walk the body items, substituting every reference to the original
   loop_var with `outer_var * lane_count + inner_var`. This walks
   `op_call.args` (region starts) and any RawStmt expressions.

The var subst step is the hardest part — affects every reads/writes
region and every op_call argument. Solution: a graph-wide expr
rewriter, similar to `_VarSubst` in the stmt walker.

### fuse_elementwise → graph.fuse_elementwise
Stmt walker today: matches `for i: AttrStmt(plena.group): BufferStore`
patterns and rewrites the whole for to a single `plena.v_*` extern
call.

In graph layer, the equivalent pattern is `NestedForGroup(items=[
RawStmt(BufferStore)])` (because lift_from_raw wraps BufferStores in
RawStmt) inside a LaneGroup or another NestedForGroup with matching
extent. Pass walks items, finds the pattern, replaces the whole
NestedForGroup with a single GraphNode(plena.v_*).

The "nested fold" rule (outer T.serial(R) wrapping a fuse target →
single whole-buffer op) becomes: when the outer NestedForGroup's body
is a single fused GraphNode with whole-buffer semantics, drop the
outer for entirely.

### lower_fp_row_patterns → graph.lower_fp_row_patterns
Largely the same as fuse_elementwise but with more pattern variants
(plena.fp_copy_at / fp_add_at / fp_exp_at / row_reduce_max_at / etc).
Uses BufferNode.physical_scope to determine FPRAM-residency of operands.

### scope_inference → graph.scope_inference (DONE — `graph_passes/scope_inference.py`)
Already exists, equivalent to stmt-walker version. Will start being
used once allocate_group_memory and the row-pattern lower also live in
the graph layer (so the dict is consumed by graph code, not stmt code).

### allocate_group_memory → INTEGRATED INTO MATERIALIZE
**Key architectural shift**: `allocate_group_memory` is no longer a
pass that runs in the middle of the pipeline. Its work — assigning
each buffer a lane layout (col_pack / row_stack / fp_lane) and
expanding shape — happens in `materialize`, after all graph
optimization is done.

Why: graph optimizations may want to change buffer shape (e.g. a
double-buffering pass doubles the lane-axis extent for K_sh; a
dead-temp-elim pass removes a temp entirely). If shape is baked in
before optimization, those transforms are blocked. Move shape decisions
to materialize, after optimization stabilizes.

Mechanism in materialize:
1. Walk graph nodes; for each op, infer the lane-layout role of each
   operand (from op kind + ATTR_GEMM_KIND, same rules as today's
   stmt-walker). Set `ATTR_LANE_LAYOUT` on each BufferNode.
2. Apply layout to BufferNode.shape: col_pack →
   `(..., orig_last) → (1, ..., lane_count, orig_last)`; row_stack →
   `(orig_first, ...) → (1, lane_count, orig_first, ...)`; fp_lane →
   `(N,) → (lane_count, N)`.
3. Build tir.Buffer objects from final BufferNode state.
4. Lowering each op call uses the final shapes for offset computation
   (existing `_auto_lane_offset` / `_dst_row_stride` logic in
   lower_to_hlir applies, just driven by BufferNode instead of
   tir.Buffer).

### lower_to_hlir helpers (kept as op-level lowering library)
`_lower_copy` / `_lower_gemm` / `_rewrite_buffer_scopes` and their
expr / region helpers are pure op-level translation — they take a Call
and emit a tir.Call for plena.* extern. They don't need to migrate;
materialize calls them per node. Already wired this way today.

---

## Verification strategy

Single source of truth for correctness: HLIR diff against `backup_20260509`.

For each migration step in Phase C:
1. Run all 7 working kernels through both old and new pipelines.
2. Compare HLIR ops + buffer tables.
3. Byte-identical → step lands.
4. Diverges → fix before merging.

Unit tests (`tests/`) provide a faster signal but cover less. They are
necessary but not sufficient — HLIR diff is the real verifier.

`/tmp/hlir_diff_tool.py` and `/tmp/hlir_diff_e2e_args.py` are existing
scripts that do the diff. Keep them through migration.

---

## Phases

* **A** ✅ — graph_ir extended, lift_from_raw exists, walker helpers exist.
* **B** ✅ — annotate_gemm_kind absorbed by lift; graph scope_inference and
  graph annotate_sync written and used.
* **C.1** ✅ — graph passes for annotate_grid / split_lane_groups /
  lift_lane_groups / fuse_elementwise / lower_fp_row_patterns written.
  allocate_group_memory split into `analyze` (graph pass) +
  `expand_buffers.expand` (run inside materialize). Pipeline rewired
  to graph-only path with a temporary fallback flag.
* **C.2** ✅ — fallback flag and legacy stmt-walker pipeline deleted.
  All 9 stmt-walker passes (annotate_group, annotate_sync,
  annotate_gemm_kind, split_lane_groups, fuse_elementwise,
  scope_inference, allocate_group_memory, lower_fp_row_patterns) +
  lift_to_blocks + lift_to_graph removed. `frontend_legacy/`
  directory removed. 6 stmt-walker test files removed.
* **D** ⏳ — loop-scheduling layer (NOT STARTED). Real new fusion:
  * **DMA merge** across iterations: combine consecutive K/V DMAs into
    one bigger transfer.
  * **Prefetch / double-buffer**: alloc `K_sh_alt`, prefetch tile k+1
    while computing on k.

  Requires confirmed answers to the HW capability questions below.

---

## Open questions

### Still relevant

**HW capabilities for Phase D**:
* Maximum single H_PREFETCH_V / H_PREFETCH_M element count?
* Total MRAM / VRAM capacity (so we know how much we can grow K_sh /
  V_sh for cross-kv_block merge)?
* Are H_PREFETCH variants stride-aware? (Affects whether N rows of
  K from `K_hbm[kv*64..(kv+1)*64]` can fold into one DMA.)

### Resolved during C.1 / C.2

* ~~Should `Graph.buffer_nodes` index by `tir.Var` (data) or by
  string name?~~ — **By name.** `BufferAccess(buffer_name, starts,
  extents)` references it; works across pass boundaries even if
  underlying `tir.Var` identity churns.
* ~~NestedForGroup vs ForNode unification?~~ — **Not done; not needed.**
  We added `attrs: dict` to NestedForGroup and ForRoot directly. The
  separate ForNode dataclass in graph_ir.py is now unused
  forward-looking infrastructure (no consumer). Can be removed in a
  later cleanup or repurposed if Phase D wants a different
  NestedForGroup shape.

---

## Phase C.2 cleanup notes (for future archaeologists)

What was deleted vs. inlined into graph_passes:

| Old stmt-walker | Replaced by |
|---|---|
| `annotate_group.py` | `graph_passes/annotate_grid.py` (+ `_VarSubst` inlined into `graph_passes/split_lane_groups.py` as `_StmtVarSubst`) |
| `annotate_sync.py` | `graph_passes/annotate_sync.py` |
| `annotate_gemm_kind.py` | absorbed by `lift_from_raw` (KIND_KEY constant inlined there) |
| `split_lane_groups.py` | `graph_passes/split_lane_groups.py` (operates on graph items + does the var substitution there) |
| `fuse_elementwise.py` | `graph_passes/fuse_elementwise.py` (sets ATTR_IS_SYNC on created nodes — see PIPELINE_ARCHITECTURE.md § Bug history) |
| `scope_inference.py` | `graph_passes/scope_inference.py` (also owns `BufferScopeMap` / `ScopeInferenceError` types) |
| `allocate_group_memory.py` | split into `graph_passes/allocate_group_memory.py:analyze` (pure analysis) + `graph_passes/expand_buffers.py:expand` (the rewriter; runs inside materialize). `_expand_buffer` and `_Rewriter` inlined into expand_buffers as `_StmtRewriter`. |
| `lower_fp_row_patterns.py` | `graph_passes/lower_fp_row_patterns.py` (runs inside materialize, AFTER expand_buffers, because pattern matchers need the 4D-expanded shape) |
| `lift_to_blocks.py` + `lift_to_graph.py` | `lift_from_raw.py` (single-shot; no intermediate block form needed) |

`graph_pipeline.run()` (the backwards-compat wrapper) deleted; only
`materialize_to_primfunc(graph, scopes, expand_lane_buffers=True)`
remains. `frontend/pipeline.py:compile_func` is now a single
straight-line call sequence with no fallback.
