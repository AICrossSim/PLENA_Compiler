"""Phase-1 frontend pipeline: tilelang IRModule → PLENA-flavored TIR.

The pipeline is built around two abstractions:

  * a *group* — a lane-fusion-eligible iteration domain. Every grid
    axis matching the hardware lane count, and every ``T.Parallel``
    iterator, is annotated as a group via
    ``ATTR_GROUP_EXTENT`` on its ForRoot / NestedForGroup.
  * a *sync site* — every DMA copy and every ``kind="btmm"`` gemm is
    marked with ``ATTR_IS_SYNC = True`` on its GraphNode. These are
    the points at which per-thread work fuses into one multi-lane
    hardware op.

Pipeline order:

    1. inline_let_stmts          — TIR housekeeping (LetStmt → subst)
    2. lower_compound_fp_stores  — arr[i] = a*b + c*d → temp → temp → out
    3. lift_from_raw_primfunc    — raw PrimFunc → :class:`Graph`
    4. graph_passes.annotate_grid       — set ATTR_GROUP_EXTENT
    5. graph_passes.annotate_sync       — set ATTR_IS_SYNC
    6. graph_passes.split_lane_groups   — split extent>lane axes
    7. graph_passes.lift_lane_groups    — ForRoot → LaneGroup upgrade
    8. graph_passes.fuse_elementwise    — T.Parallel → plena.v_*
    9. graph_passes.scope_inference     — buffer_name → physical scope
   10. graph_pipeline.materialize_to_primfunc, with expand_lane_buffers=True:
         a. graph_passes.allocate_group_memory.analyze    — set ATTR_LANE_LAYOUT
         b. graph_passes.expand_buffers.expand            — rebuild tir.Buffers
         c. graph_passes.lower_fp_row_patterns            — fp_*_at / row_*_at
         d. partition + materialize → tir.PrimFunc
   11. _rewrite_buffer_scopes   — shared.dyn → vram, etc, for codegen

Each pass lives under ``frontend/passes/`` (top-level for the stmt-prep
helpers + IR module + materializer) or ``frontend/passes/graph_passes/``
(for everything that operates on the :class:`graph_ir.Graph`).
"""

from __future__ import annotations

import tvm
from tvm import tir

from ..pipeline import PlenaTarget
from .passes import inline_let_stmts, lower_compound_fp_stores
from .passes.lift_from_raw import lift_from_raw_primfunc
from .passes.lower_to_hlir import _rewrite_buffer_scopes
from .passes import graph_pipeline
from .passes.graph_passes import (
    annotate_grid as graph_annotate_grid,
    annotate_sync as graph_annotate_sync,
    split_lane_groups as graph_split_lane_groups,
    lift_lane_groups as graph_lift_lane_groups,
    fuse_elementwise as graph_fuse_elementwise,
    scope_inference as graph_scope_inference,
)
# Opt-in sanity check; not invoked from compile_func by default.
# Kernels that want to enforce "tilelang DSL only" can call
# forbid_plena_extern.run(prim_func) before passing to compile_func.
from .passes import forbid_plena_extern  # noqa: F401


def compile_func(func: tir.PrimFunc,
                 target: PlenaTarget | None = None) -> tir.PrimFunc:
    """Run the Phase-1 passes in order. Returns a fully-lowered PrimFunc."""
    if target is None:
        target = PlenaTarget()
    sync_width = target.mlen // target.btmm_hlen

    # ---- minimal stmt prep ----
    func = inline_let_stmts.run(func)
    func = lower_compound_fp_stores.run(func)

    # ---- lift to graph ----
    graph = lift_from_raw_primfunc(func)

    # ---- graph-layer passes ----
    graph = graph_annotate_grid.run(graph)
    graph = graph_annotate_sync.run(graph)
    graph = graph_split_lane_groups.run(graph, lane_count=sync_width)
    # Upgrade lane-fusion-eligible ForRoots into LaneGroups so the
    # materialize-time partitioner does the curtain-bundle algorithm.
    graph = graph_lift_lane_groups.run(graph, lane_count=sync_width)
    graph = graph_fuse_elementwise.run(graph)
    scopes = graph_scope_inference.infer(graph)

    # ---- materialize ----
    # materialize_to_primfunc(expand_lane_buffers=True) internally runs
    # allocate_group_memory.analyze + expand_buffers.expand +
    # lower_fp_row_patterns just before lowering each op.
    out = graph_pipeline.materialize_to_primfunc(
        graph, scopes,
        lane_count=sync_width,
        target_mlen=target.mlen,
        target_hlen=target.btmm_hlen,
        expand_lane_buffers=True,
    )

    # ---- final scope rewrite ----
    # Turn ``shared.dyn`` / ``local.fragment`` buffers into their
    # resolved physical scopes (vram / mram / fpram) so codegen can
    # read ``buf.scope()`` directly.
    new_body = _rewrite_buffer_scopes(out.body, scopes)
    return tir.PrimFunc(
        params=out.params, body=new_body,
        ret_type=out.ret_type, buffer_map=out.buffer_map,
        attrs=out.attrs,
    )


def compile_to_tir_text(func: tir.PrimFunc, name: str = "kernel",
                        target: PlenaTarget | None = None) -> str:
    """Lower and serialise to TVMScript text."""
    lowered = compile_func(func, target=target)
    mod = tvm.IRModule({name: lowered})
    return mod.script()


__all__ = ["PlenaTarget", "compile_func", "compile_to_tir_text"]
