"""Map tilelang storage scopes to PLENA storage scopes.

Returns a ``BufferScopeMap`` — a plain ``dict[str, str]`` from buffer name
to one of ``{"hbm", "mram", "vram", "fpram"}``.

Rules (slim version, sufficient for the matmul/btmm path):

  * Every ``T.match_buffer`` param → ``"hbm"``.
  * A ``shared.dyn`` buffer that ever appears as the RHS (arg[1]) of a
    ``tl.tileop.gemm_py`` call → ``"mram"``.  PLENA's MM hardware reads
    its right-hand operand from MRAM; other shared buffers stay in VRAM.
  * Every other ``shared.dyn`` buffer → ``"vram"``.
  * A ``local.fragment`` buffer that is referenced via BufferLoad at an
    FP-scalar operand position of ``plena.fp_*_at`` / ``plena.row_*_at``
    → ``"fpram"``.
  * Every other ``local.fragment`` buffer → ``"vram"``  (gemm
    accumulators and per-thread fragments live in VRAM today).
  * Buffers with any other declared scope are not yet supported and the
    pass raises ``ScopeInferenceError`` — this surfaces the problem
    early rather than silently miscompiling.

This pass does **not** mutate the IR. It walks once to collect uses and
returns the map. Downstream passes (``allocate_group_memory``,
``lower_to_hlir``) consume the map to either rewrite buffer scopes or
make code-emission decisions.
"""

from __future__ import annotations

from typing import Dict

from tvm import tir


_TILEOP_GEMM = "tl.tileop.gemm_py"
_TILEOP_REGION = "tl.tileop.region"
_TILEOP_REDUCE = "tl.tileop.reduce"


_FP_EXTERN_POSITIONS = {
    "plena.fp_copy_at": (0, 1),
    "plena.fp_add_at": (0, 1, 2),
    "plena.fp_sub_at": (0, 1, 2),
    "plena.fp_mul_at": (0, 1, 2),
    "plena.fp_max_at": (0, 1, 2),
    "plena.fp_exp_at": (0, 1),
    "plena.fp_reci_at": (0, 1),
    "plena.fp_sqrt_at": (0, 1),
    "plena.row_reduce_max_at": (1,),
    "plena.row_reduce_sum_at": (1,),
    "plena.row_sub_fp_at": (1,),
    "plena.row_mul_fp_at": (1,),
    "plena.row_add_fp_at": (1,),
}


# Public alias for clarity at call sites.
BufferScopeMap = Dict[str, str]


class ScopeInferenceError(RuntimeError):
    pass


def _region_buffer_name(call):
    """Return the name of the buffer wrapped by a `T.region(...)` call,
    or None if the argument isn't a region call we can read."""
    if not isinstance(call, tir.Call):
        return None
    if call.op.name != _TILEOP_REGION:
        return None
    load = call.args[0]
    if not isinstance(load, tir.BufferLoad):
        return None
    return load.buffer.name


def _region_buffer(call):
    if not isinstance(call, tir.Call):
        return None
    if call.op.name != _TILEOP_REGION:
        return None
    load = call.args[0]
    if not isinstance(load, tir.BufferLoad):
        return None
    return load.buffer


def _mark_rank1_fragment_loads(expr, out: set):
    if isinstance(expr, tir.BufferLoad):
        if len(expr.buffer.shape) == 1:
            out.add(expr.buffer.name)
        for i in expr.indices:
            _mark_rank1_fragment_loads(i, out)
        return
    if isinstance(expr, tir.Call):
        for a in expr.args:
            _mark_rank1_fragment_loads(a, out)
        return
    if hasattr(expr, "a") and hasattr(expr, "b"):
        _mark_rank1_fragment_loads(expr.a, out)
        _mark_rank1_fragment_loads(expr.b, out)
        return
    if hasattr(expr, "value"):
        _mark_rank1_fragment_loads(expr.value, out)


def _walk_collect_uses(stmt, mram_names: set, fpram_names: set):
    """Walk the IR and record every buffer that appears as gemm arg[1]
    in `mram_names` (passed by reference)."""
    if isinstance(stmt, tir.SeqStmt):
        for c in stmt.seq:
            _walk_collect_uses(c, mram_names, fpram_names)
        return
    if isinstance(stmt, tir.BlockRealize):
        _walk_collect_uses(stmt.block, mram_names, fpram_names)
        return
    if isinstance(stmt, tir.Block):
        _walk_collect_uses(stmt.body, mram_names, fpram_names)
        if stmt.init is not None:
            _walk_collect_uses(stmt.init, mram_names, fpram_names)
        return
    if isinstance(stmt, (tir.AttrStmt, tir.LetStmt, tir.For)):
        _walk_collect_uses(stmt.body, mram_names, fpram_names)
        return
    if isinstance(stmt, tir.IfThenElse):
        _walk_collect_uses(stmt.then_case, mram_names, fpram_names)
        if stmt.else_case is not None:
            _walk_collect_uses(stmt.else_case, mram_names, fpram_names)
        return
    if isinstance(stmt, tir.Evaluate):
        v = stmt.value
        if isinstance(v, tir.Call) and v.op.name == _TILEOP_GEMM:
            rhs_name = _region_buffer_name(v.args[1])
            if rhs_name is not None:
                mram_names.add(rhs_name)
        elif isinstance(v, tir.Call) and v.op.name == _TILEOP_REDUCE:
            dst = _region_buffer(v.args[1]) if len(v.args) >= 2 else None
            if dst is not None and len(dst.shape) == 1:
                fpram_names.add(dst.name)
        # Already-lowered plena.matmul (or plena.btmm) call_externs:
        # the RHS buffer (B operand) must live in MRAM. Without picking
        # these up we'd treat a buffer that's only used as a manual
        # matmul RHS as plain VRAM and fail scope verification.
        elif (isinstance(v, tir.Call) and v.op.name == "tir.call_extern"
              and v.args and isinstance(v.args[0], tir.StringImm)
              and v.args[0].value in ("plena.matmul", "plena.btmm",
                                      "plena.mv", "plena.btmv")):
            # call layout in v.args:
            #   [0] StringImm("plena.matmul" / "plena.btmm")
            #   [1] A.data  (LHS)
            #   [2] B.data  (RHS — MRAM)
            #   [3] C.data  (DST)
            #   [4..] scalar args
            rhs_var = v.args[2] if len(v.args) >= 3 else None
            if isinstance(rhs_var, tir.Var):
                mram_names.add(rhs_var)
        elif (isinstance(v, tir.Call) and v.op.name == "tir.call_extern"
              and v.args and isinstance(v.args[0], tir.StringImm)):
            name = v.args[0].value
            positions = _FP_EXTERN_POSITIONS.get(name, ())
            raw_args = list(v.args[1:])
            for pos in positions:
                if pos >= len(raw_args):
                    continue
                arg = raw_args[pos]
                if isinstance(arg, tir.BufferLoad):
                    fpram_names.add(arg.buffer.name)
        return
    if isinstance(stmt, tir.BufferStore):
        if len(stmt.buffer.shape) == 1:
            fpram_names.add(stmt.buffer.name)
        _mark_rank1_fragment_loads(stmt.value, fpram_names)
        return


def _alloc_buffers(stmt, out: list):
    """Recursively collect every Buffer declared via Block.alloc_buffers."""
    if isinstance(stmt, tir.SeqStmt):
        for c in stmt.seq:
            _alloc_buffers(c, out)
        return
    if isinstance(stmt, tir.BlockRealize):
        _alloc_buffers(stmt.block, out)
        return
    if isinstance(stmt, tir.Block):
        out.extend(stmt.alloc_buffers)
        _alloc_buffers(stmt.body, out)
        return
    if isinstance(stmt, (tir.AttrStmt, tir.LetStmt, tir.For)):
        _alloc_buffers(stmt.body, out)
        return
    if isinstance(stmt, tir.IfThenElse):
        _alloc_buffers(stmt.then_case, out)
        if stmt.else_case is not None:
            _alloc_buffers(stmt.else_case, out)
        return


def _assign_scope(buf: tir.Buffer, mram_names: set, fpram_names: set) -> str:
    declared = buf.scope() if callable(getattr(buf, "scope", None)) else "global"
    if declared == "shared.dyn":
        return "mram" if buf.name in mram_names else "vram"
    if declared == "local.fragment":
        # Rank-1 fragments are FPRAM by convention (lane-stacked scalar
        # scratch). Even if a fragment never participates in FP-scalar
        # arithmetic — e.g. it only appears as the source of T.copy(fp,
        # shared) for an explicit FP→V materialization — it still wants
        # to live in FPRAM so allocate_group_memory's FP-LANE expansion
        # applies. Higher-rank fragments default to VRAM (gemm
        # accumulators, P@V intermediates), unless usage promotes them.
        if buf.name in fpram_names or len(buf.shape) == 1:
            return "fpram"
        return "vram"
    raise ScopeInferenceError(
        f"buffer {buf.name!r} has unsupported declared scope {declared!r}; "
        f"slim scope_inference handles only shared.dyn and local.fragment"
    )


def _resolve_var_names(mram_set: set, allocs: list) -> set:
    """Some matmul RHS detection paths add a `tir.Var` (the buffer's
    `data` handle) to the mram set instead of a name string — those come
    from already-lowered `plena.matmul`/`plena.btmm` extern calls. Map
    them back to buffer names here so `_assign_scope` (which keys by
    name) can look them up uniformly."""
    var_to_name = {buf.data: buf.name for buf in allocs}
    out: set = set()
    for x in mram_set:
        if isinstance(x, str):
            out.add(x)
        elif isinstance(x, tir.Var) and x in var_to_name:
            out.add(var_to_name[x])
    return out


def infer(func: tir.PrimFunc) -> BufferScopeMap:
    """Return a name→scope map covering every buffer in the function."""
    scopes: BufferScopeMap = {}

    # 1. HBM buffers come from func.buffer_map (T.match_buffer params).
    for buf in func.buffer_map.values():
        scopes[buf.name] = "hbm"

    # 2. Walk the IR once, find every shared.dyn buffer used as gemm RHS
    # and every local.fragment used as an FP scalar scratch buffer.
    mram_names: set = set()
    fpram_names: set = set()
    _walk_collect_uses(func.body, mram_names, fpram_names)

    # 3. Walk allocations and assign scopes.
    allocs: list = []
    _alloc_buffers(func.body, allocs)
    mram_names = _resolve_var_names(mram_names, allocs)
    for buf in allocs:
        scopes[buf.name] = _assign_scope(buf, mram_names, fpram_names)

    return scopes


__all__ = ["infer", "BufferScopeMap", "ScopeInferenceError"]
