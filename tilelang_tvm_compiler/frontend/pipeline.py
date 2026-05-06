"""Phase-1 frontend pipeline: tilelang IRModule -> PLENA-flavored TIR.

The pipeline is built around an explicit *group* abstraction:

  * Every grid axis with extent matching the hardware lane count, and every
    `T.Parallel` iterator, is annotated as a group via
    ``T.attr(0, "plena.group", extent=N)``.
  * Every DMA copy and every ``kind="btmm"`` gemm is wrapped in implicit
    ``T.attr(0, "plena.sync", ...)`` markers — these are the points at
    which per-thread work fuses into one multi-lane hardware op.
  * Shared / fragment buffers used inside a group are expanded (last-dim
    multiplied by the group extent) so the post-fusion HW ops have
    enough storage.
  * The final ``lower_to_hlir`` pass walks the annotated IR and emits
    ``plena.*`` extern calls. Inside a group it does not unroll the
    underlying for-loop; instead, sync-bordered DMA / BTMM ops fold all
    iterations into a single multi-lane hardware op.

Pipeline order:

    1. annotate_gemm_kind     -- ensure every gemm carries `plena.gemm_kind`
                                  (default 'overwrite').
    2. annotate_group         -- detect group-eligible axes, wrap with
                                  `plena.group` AttrStmts.
    3. annotate_sync          -- insert implicit `plena.sync` markers
                                  around DMA copies and `kind=btmm` gemms.
    4. scope_inference (slim) -- map shared.dyn / local.fragment to PLENA
                                  storage scopes.
    5. allocate_group_memory  -- expand buffer last-dim by group extent
                                  for buffers used inside a group.
    6. fuse_elementwise       -- collapse per-thread elementwise ops in
                                  T.Parallel groups into single vector ops.
    7. lower_to_hlir          -- emit plena.* extern calls.

Each pass is in its own file under `frontend/passes/`. They are wired
here in order; passes 2-7 are work-in-progress.
"""

from __future__ import annotations

import tvm
from tvm import tir

from ..pipeline import PlenaTarget
from .passes import (
    inline_let_stmts, lower_compound_fp_stores,
    annotate_gemm_kind, annotate_group, annotate_sync, split_lane_groups,
    scope_inference, allocate_group_memory, lower_fp_row_patterns,
    fuse_elementwise, lower_to_hlir,
)
# Opt-in sanity check; not invoked from compile_func by default.
# Kernels that want to enforce "tilelang DSL only" can call
# forbid_plena_extern.run(prim_func) before passing to compile_func.
from .passes import forbid_plena_extern  # noqa: F401


def compile_func(func: tir.PrimFunc,
                 target: PlenaTarget | None = None) -> tir.PrimFunc:
    """Run the Phase-1 passes in order. Returns a fully-lowered PrimFunc.

    The pipeline is being rebuilt around the group abstraction; passes
    not yet implemented are skipped (their absence from the pipeline is
    intentional — a kernel that needs them will surface a downstream
    error rather than silently miscompile).
    """
    if target is None:
        target = PlenaTarget()
    sync_width = target.mlen // target.btmm_hlen

    func = inline_let_stmts.run(func)
    func = lower_compound_fp_stores.run(func)
    func = annotate_gemm_kind.run(func)
    func = annotate_group.run(func)
    func = annotate_sync.run(func, sync_width=sync_width)
    func = split_lane_groups.run(func, lane_count=sync_width)
    # Fuse T.Parallel elementwise patterns into plena.v_* / plena.zero_v
    # BEFORE allocate_group_memory walks the IR — that way the resulting
    # extern calls (rather than the raw T.Parallel forms) feed into
    # allocate's lane-axis discovery logic, so kernels written without
    # any plena.* extern still get their O_loc / PV_loc / etc. expanded.
    func = fuse_elementwise.run(func)
    scopes = scope_inference.infer(func)
    func = allocate_group_memory.run(func, scopes,
                                      lane_count=sync_width)
    func = lower_fp_row_patterns.run(func, scopes)
    func = lower_to_hlir.run(func, scopes,
                              lane_count=sync_width,
                              target_mlen=target.mlen,
                              target_hlen=target.btmm_hlen)
    return func


def compile_to_tir_text(func: tir.PrimFunc, name: str = "kernel",
                        target: PlenaTarget | None = None) -> str:
    """Lower and serialise to TVMScript text."""
    lowered = compile_func(func, target=target)
    mod = tvm.IRModule({name: lowered})
    return mod.script()


__all__ = ["PlenaTarget", "compile_func", "compile_to_tir_text"]
