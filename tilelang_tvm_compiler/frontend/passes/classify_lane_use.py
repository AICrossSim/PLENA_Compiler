"""Tag every buffer in a raw PrimFunc with its lane-fusion role.

Why this pass exists
--------------------

The SPMD-rewrite pipeline (see ``compiler/SPMD_REWRITE.md``) replaces
the four lane-fusion graph passes with three small early TIR passes::

    classify_lane_use   ← this file. Read-only, sets attributes only.
    expand_lane_grid    ← needs the tags to know which buffers get a
                          LANE outer dim and which stay 2D / 1D.
    infer_lane_layout   ← needs the tags to know whether each buffer
                          wants COL_PACK (lane at dim 0) or BHSD
                          (lane at dim 1).

This pass walks the function body once, looks at the **op call sites**
that touch each buffer, and assigns one role per buffer. The two
downstream passes consume the role table and never re-derive it.

Whether a buffer participates in lane fusion is a function of *how the
ops that touch it are annotated*, not of the buffer's shape. That's
the entire reason classification has to come first — ``expand_lane_grid``
can't blindly add a LANE dim to every alloc.

Recognised op forms in the raw TIR (post-tilelang-lower, pre-PLENA-lift)
-------------------------------------------------------------------------

The pass runs after ``inline_let_stmts`` + ``lower_compound_fp_stores``
and before ``lift_from_raw_primfunc``. tilelang's ``T.gemm`` / ``T.copy``
have already been lowered into ``tir.call_extern`` shapes:

    T.gemm(A, B, C, ...)  →  Evaluate(call_extern("tl.tileop.gemm_py",
                                                  region(A), region(B),
                                                  region(C), ...))
    T.copy(src, dst)      →  Evaluate(call_extern("tl.tileop.copy",
                                                  region(src), region(dst)))

A ``with T.attr(0, KIND_KEY, "btmm"): T.gemm(...)`` adds an outer
``AttrStmt(attr_key="plena.gemm_kind", value=StringImm("btmm"))``
around the gemm Evaluate. ``classify_lane_use`` reads the attr the
same way ``lift_from_raw`` does.

The ``T.Kernel`` grid bindings appear as ``AttrStmt(thread_extent,
IterVar(thread_tag="blockIdx.x"|"blockIdx.y"))`` near the function
body's root. The kernel marks one of these as the lane axis with
``T.func_attr({"plena.lane_axis": "by"})``; the pass picks it up to
detect "this T.copy uses ``by`` in its HBM slice → lane fusion DMA".

Output
------

Returns ``(func, classification)`` where ``classification`` is a dict
``buffer_name -> BufferRole``:

    BufferRole.role : str  (see ROLE_* constants)
    BufferRole.lane_aware : bool

The PrimFunc itself is returned **unchanged** (read-only pass).
``expand_lane_grid`` and ``infer_lane_layout`` take ``classification``
as an extra argument; we don't try to round-trip the data through TIR
attributes since we're going to do that work ourselves anyway.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import tvm
from tvm import tir


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# AttrStmt key set by ``with T.attr(0, KIND, "btmm"): T.gemm(...)`` —
# matches gemm_macros.KIND and lift_from_raw.KIND_KEY. Duplicated here
# to keep this pass importable without dragging in either of those.
KIND_KEY = "plena.gemm_kind"

# func_attr key set by ``T.func_attr({"plena.lane_axis": "by"})``.
LANE_AXIS_FUNC_ATTR = "plena.lane_axis"

# tilelang-lowered op names we recognise as PLENA-relevant.
_TILEOP_COPY = "tl.tileop.copy"
_TILEOP_GEMM = "tl.tileop.gemm_py"
_TILEOP_REGION = "tl.tileop.region"

# Roles. See SPMD_REWRITE.md §3.0 / §3.2 for the full table.
ROLE_NONE          = "none"            # not lane-aware (single-tile / scalar / param)
ROLE_BTMM_LHS      = "btmm_lhs"        # COL_PACK (lane at dim 0)
ROLE_BTMM_RHS      = "btmm_rhs"        # COL_PACK
ROLE_BTMM_OUT      = "btmm_out"        # BHSD (lane at dim 1)
ROLE_PER_HEAD_LHS  = "per_head_lhs"    # BHSD
ROLE_PER_HEAD_RHS  = "per_head_rhs"    # COL_PACK
ROLE_PER_HEAD_OUT  = "per_head_out"    # COL_PACK
ROLE_LANE_DMA_DST  = "lane_dma_dst"    # COL_PACK (DMA fed by an HBM slice indexed by `by`)


# Roles that imply the buffer needs a LANE outer dim.
_LANE_AWARE_ROLES: Set[str] = {
    ROLE_BTMM_LHS, ROLE_BTMM_RHS, ROLE_BTMM_OUT,
    ROLE_PER_HEAD_LHS, ROLE_PER_HEAD_RHS, ROLE_PER_HEAD_OUT,
    ROLE_LANE_DMA_DST,
}


class ClassifyLaneUseError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Role table entry
# ---------------------------------------------------------------------------


@dataclass
class BufferRole:
    """One classification record per buffer."""
    role: str
    # Set of evidence sites that contributed (op kind names). Useful for
    # error messages when conflicting roles are assigned.
    evidence: Tuple[str, ...] = ()

    @property
    def lane_aware(self) -> bool:
        return self.role in _LANE_AWARE_ROLES


# ---------------------------------------------------------------------------
# Lane-axis var detection
# ---------------------------------------------------------------------------


def _read_lane_axis_name(func: tir.PrimFunc) -> Optional[str]:
    """Return the kernel-author-declared lane axis name, or None.

    Reads ``T.func_attr({"plena.lane_axis": "by"})`` from the function's
    attrs. Returns the bare string (e.g. ``"by"``); the body walker
    later matches it against grid IterVar names.
    """
    if func.attrs is None:
        return None
    if LANE_AXIS_FUNC_ATTR not in func.attrs:
        return None
    raw = func.attrs[LANE_AXIS_FUNC_ATTR]
    if raw is None:
        return None
    if isinstance(raw, tir.StringImm):
        return str(raw.value)
    return str(raw)


def _collect_lane_var(func: tir.PrimFunc, axis_name: str) -> Optional[tir.Var]:
    """Find the ``tir.Var`` bound to the named grid axis.

    Walks ``AttrStmt(thread_extent, IterVar(...))`` chains at the
    function root. Matches by ``IterVar.var.name``.
    """
    found: List[tir.Var] = []

    def visit(stmt):
        if stmt is None:
            return
        if isinstance(stmt, tir.AttrStmt):
            if (stmt.attr_key == "thread_extent"
                    and isinstance(stmt.node, tir.IterVar)
                    and stmt.node.var.name == axis_name):
                found.append(stmt.node.var)
            visit(stmt.body)
            return
        if isinstance(stmt, tir.SeqStmt):
            for c in stmt.seq:
                visit(c)
            return
        if isinstance(stmt, tir.BlockRealize):
            visit(stmt.block.body)
            if stmt.block.init is not None:
                visit(stmt.block.init)
            return
        if isinstance(stmt, (tir.For, tir.LetStmt)):
            visit(stmt.body)
            return
        if isinstance(stmt, tir.IfThenElse):
            visit(stmt.then_case)
            if stmt.else_case is not None:
                visit(stmt.else_case)
            return

    visit(func.body)
    if not found:
        return None
    if len(found) > 1:
        raise ClassifyLaneUseError(
            f"lane axis {axis_name!r} bound more than once "
            f"({len(found)} thread_extent sites); kernel is malformed"
        )
    return found[0]


# ---------------------------------------------------------------------------
# Expression scan: does this PrimExpr reference a given Var?
# ---------------------------------------------------------------------------


def _expr_uses_var(expr, var: tir.Var) -> bool:
    """True if ``expr`` (a tir.PrimExpr) syntactically references ``var``.

    Uses tvm's post_order_visit since PrimExprs can be arbitrary trees.
    """
    if expr is None:
        return False
    seen = [False]

    def cb(node):
        if isinstance(node, tir.Var) and node.same_as(var):
            seen[0] = True

    from tvm.tir import stmt_functor
    stmt_functor.post_order_visit(expr, cb)
    return seen[0]


# ---------------------------------------------------------------------------
# Region call → (buffer_name, starts) extractor
# ---------------------------------------------------------------------------


def _call_kind(call: tir.Call) -> Optional[str]:
    """Return the logical op name of a call.

    Handles two encodings of the same call:
      * Direct Op:    ``Call(op=Op("tl.tileop.gemm_py"), args=[...])``
      * call_extern:  ``Call(op=Op("tir.call_extern"),
                             args=[StringImm("tl.tileop.gemm_py"), ...])``

    The first is what tilelang produces in real lowering; the second is
    convenient for tests that don't load tilelang (so the
    ``tl.tileop.*`` Ops aren't registered). Returns ``None`` if the
    call doesn't look like either.
    """
    if not isinstance(call, tir.Call):
        return None
    op_name = getattr(call.op, "name", "")
    if op_name and not op_name.startswith("tir."):
        return op_name
    if op_name == "tir.call_extern" and call.args:
        head = call.args[0]
        if isinstance(head, tir.StringImm):
            return str(head.value)
    return None


def _call_args(call: tir.Call) -> List:
    """Strip the leading StringImm name for call_extern; pass through
    otherwise. Mirrors what ``codegen.py`` does."""
    op_name = getattr(call.op, "name", "")
    if op_name == "tir.call_extern" and call.args:
        return list(call.args[1:])
    return list(call.args)


def _region_buffer_and_starts(call: tir.Call) -> Optional[Tuple[str, List]]:
    """``tl.tileop.region(BufferLoad(buf, [starts]), ...)`` → (name, starts).

    Returns None for anything we don't recognise. The starts list is
    the BufferLoad's indices — what we need to ask "did this index use
    the lane var?".
    """
    if not isinstance(call, tir.Call):
        return None
    if _call_kind(call) != _TILEOP_REGION:
        return None
    args = _call_args(call)
    if not args:
        return None
    load = args[0]
    if not isinstance(load, tir.BufferLoad):
        return None
    return load.buffer.name, list(load.indices)


# ---------------------------------------------------------------------------
# The classifier
# ---------------------------------------------------------------------------


class _Classifier:
    def __init__(self, func: tir.PrimFunc):
        self.func = func
        self.lane_axis_name = _read_lane_axis_name(func)
        self.lane_var = (
            _collect_lane_var(func, self.lane_axis_name)
            if self.lane_axis_name is not None else None
        )
        # Buffer-name → BufferRole. Defaults to ROLE_NONE; we only
        # promote when an op site demands it. Param + alloc names
        # both keyed here.
        self.roles: Dict[str, BufferRole] = {}
        self._seed_param_buffers()

    # -- seeding --------------------------------------------------------

    def _seed_param_buffers(self) -> None:
        for buf in self.func.buffer_map.values():
            self.roles.setdefault(buf.name, BufferRole(role=ROLE_NONE))

    # -- public ---------------------------------------------------------

    def run(self) -> Dict[str, BufferRole]:
        # Walk the body, picking up alloc'd buffers and op call sites.
        self._visit_stmt(self.func.body, current_kind=None)
        return self.roles

    # -- assignment helpers --------------------------------------------

    def _assign(self, buf_name: str, role: str, evidence: str) -> None:
        existing = self.roles.get(buf_name)
        if existing is None or existing.role == ROLE_NONE:
            self.roles[buf_name] = BufferRole(
                role=role, evidence=(evidence,),
            )
            return
        if existing.role == role:
            self.roles[buf_name] = BufferRole(
                role=role,
                evidence=existing.evidence + (evidence,),
            )
            return
        # Conflict — same buffer wants two different roles.
        # Some role pairs are layout-compatible (both COL_PACK, both BHSD).
        # We tolerate those; the layout vote in infer_lane_layout will
        # break ties. Only structurally-incompatible pairs raise.
        if _layouts_compatible(existing.role, role):
            # Keep the existing role; record the additional evidence.
            self.roles[buf_name] = BufferRole(
                role=existing.role,
                evidence=existing.evidence + (evidence,),
            )
            return
        raise ClassifyLaneUseError(
            f"buffer {buf_name!r} has conflicting lane-fusion roles: "
            f"{existing.role} (from {existing.evidence}) "
            f"vs {role} (from {evidence}). "
            f"This means the same buffer is used as e.g. both a BTMM "
            f"output (BHSD) and a BTMM LHS (COL_PACK), which is not "
            f"physically representable. Refactor the kernel to use two "
            f"separate buffers."
        )

    # -- traversal ------------------------------------------------------

    def _visit_stmt(self, stmt, current_kind: Optional[str]) -> None:
        if stmt is None:
            return
        if isinstance(stmt, tir.SeqStmt):
            for c in stmt.seq:
                self._visit_stmt(c, current_kind)
            return
        if isinstance(stmt, tir.BlockRealize):
            self._visit_stmt(stmt.block.body, current_kind)
            if stmt.block.init is not None:
                self._visit_stmt(stmt.block.init, current_kind)
            return
        if isinstance(stmt, tir.AttrStmt):
            # Capture KIND_KEY so the Evaluate inside knows it's a btmm.
            if stmt.attr_key == KIND_KEY:
                v = stmt.value
                kind = v.value if isinstance(v, tir.StringImm) else str(v)
                self._visit_stmt(stmt.body, current_kind=kind)
                return
            self._visit_stmt(stmt.body, current_kind)
            return
        if isinstance(stmt, tir.For):
            self._visit_stmt(stmt.body, current_kind)
            return
        if isinstance(stmt, tir.LetStmt):
            self._visit_stmt(stmt.body, current_kind)
            return
        if isinstance(stmt, tir.IfThenElse):
            self._visit_stmt(stmt.then_case, current_kind)
            if stmt.else_case is not None:
                self._visit_stmt(stmt.else_case, current_kind)
            return
        if isinstance(stmt, tir.Allocate):
            # Allocate doesn't itself express a role — wait for an op
            # site to touch the buffer.
            self._visit_stmt(stmt.body, current_kind)
            return
        if isinstance(stmt, tir.Evaluate):
            self._visit_evaluate(stmt, current_kind)
            return
        # tir.BufferStore (per-element ops, e.g. fp scalar updates):
        # buffer is stored to, but with no extern call we can't tell
        # what role to assign. The kernel's lane loop (added later by
        # expand_lane_grid) will index into it; for now leave it as
        # ROLE_NONE and let downstream propagation pick it up.
        if isinstance(stmt, tir.BufferStore):
            return
        # Anything else: don't crash, just don't assign anything.

    def _visit_evaluate(self, ev: tir.Evaluate,
                        current_kind: Optional[str]) -> None:
        val = ev.value
        if not isinstance(val, tir.Call):
            return
        kind = _call_kind(val)
        if kind == _TILEOP_GEMM:
            self._classify_gemm(val, current_kind)
            return
        if kind == _TILEOP_COPY:
            self._classify_copy(val)
            return
        # Other extern calls (already-lowered plena.* builtins, or
        # tilelang reduce, etc.): skip. Reduce ops carry their roles
        # via the buffer that feeds them; the gemm/copy walkers
        # already covered the producers.

    def _classify_gemm(self, call: tir.Call,
                       current_kind: Optional[str]) -> None:
        """``tl.tileop.gemm_py(region(A), region(B), region(C), ...)``."""
        args = _call_args(call)
        if len(args) < 3:
            return
        a = _region_buffer_and_starts(args[0])
        b = _region_buffer_and_starts(args[1])
        c = _region_buffer_and_starts(args[2])
        if a is None or b is None or c is None:
            return
        if current_kind == "btmm":
            self._assign(a[0], ROLE_BTMM_LHS, evidence="gemm[btmm].A")
            self._assign(b[0], ROLE_BTMM_RHS, evidence="gemm[btmm].B")
            self._assign(c[0], ROLE_BTMM_OUT, evidence="gemm[btmm].C")
            return
        # Default kind = "overwrite" — per-head matmul.
        self._assign(a[0], ROLE_PER_HEAD_LHS, evidence="gemm.A")
        self._assign(b[0], ROLE_PER_HEAD_RHS, evidence="gemm.B")
        self._assign(c[0], ROLE_PER_HEAD_OUT, evidence="gemm.C")

    def _classify_copy(self, call: tir.Call) -> None:
        """``tl.tileop.copy(region(src), region(dst))``.

        If the src region's starts use the lane var → DMA pulls
        per-lane data → dst is a lane-aware buffer.
        If the dst region's starts use the lane var → DMA writes
        per-lane data → src is a lane-aware buffer.
        Neither references the lane var → single-lane copy.
        """
        args = _call_args(call)
        if len(args) < 2:
            return
        src = _region_buffer_and_starts(args[0])
        dst = _region_buffer_and_starts(args[1])
        if src is None or dst is None:
            return
        src_uses_lane = self._any_index_uses_lane(src[1])
        dst_uses_lane = self._any_index_uses_lane(dst[1])
        if src_uses_lane and not dst_uses_lane:
            self._assign(dst[0], ROLE_LANE_DMA_DST, evidence="copy.dst")
            return
        if dst_uses_lane and not src_uses_lane:
            self._assign(src[0], ROLE_LANE_DMA_DST, evidence="copy.src")
            return
        # Both or neither: single-lane copy. Nothing to assign.

    def _any_index_uses_lane(self, indices) -> bool:
        if self.lane_var is None:
            return False
        for idx in indices:
            if _expr_uses_var(idx, self.lane_var):
                return True
        return False


# ---------------------------------------------------------------------------
# Layout compatibility for conflict resolution
# ---------------------------------------------------------------------------

_COL_PACK_ROLES: Set[str] = {
    ROLE_BTMM_LHS, ROLE_BTMM_RHS,
    ROLE_PER_HEAD_RHS, ROLE_PER_HEAD_OUT,
    ROLE_LANE_DMA_DST,
}
_BHSD_ROLES: Set[str] = {
    ROLE_BTMM_OUT, ROLE_PER_HEAD_LHS,
}


def _layouts_compatible(role_a: str, role_b: str) -> bool:
    """True when two roles map to the same physical layout class."""
    if role_a in _COL_PACK_ROLES and role_b in _COL_PACK_ROLES:
        return True
    if role_a in _BHSD_ROLES and role_b in _BHSD_ROLES:
        return True
    return False


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


def run(func: tir.PrimFunc) -> Tuple[tir.PrimFunc, Dict[str, BufferRole]]:
    """Tag every buffer with its lane-fusion role.

    Returns the (unchanged) PrimFunc and a name → BufferRole dict.
    The caller passes the dict on to ``expand_lane_grid`` and
    ``infer_lane_layout``.
    """
    return func, _Classifier(func).run()


__all__ = [
    "run",
    "BufferRole",
    "ClassifyLaneUseError",
    "ROLE_NONE",
    "ROLE_BTMM_LHS",
    "ROLE_BTMM_RHS",
    "ROLE_BTMM_OUT",
    "ROLE_PER_HEAD_LHS",
    "ROLE_PER_HEAD_RHS",
    "ROLE_PER_HEAD_OUT",
    "ROLE_LANE_DMA_DST",
    "KIND_KEY",
    "LANE_AXIS_FUNC_ATTR",
]
