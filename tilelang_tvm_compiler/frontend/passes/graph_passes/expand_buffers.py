"""Materialize-time helper: expand each tagged BufferNode's
``tir.Buffer`` and rewrite every reference in the graph (op_calls,
BufferAccess regions, RawStmt TIR) to use the expanded buffer with the
lane axis folded into indices.

This is the *expansion* half of the legacy stmt-walker
``frontend/passes/allocate_group_memory.py``. The *analysis* half lives
in :mod:`graph_passes.allocate_group_memory` and runs as a graph pass;
this module runs at materialize time, after all other graph
optimizations.

Why split analysis (graph) from expansion (materialize)
-------------------------------------------------------
Per the migration plan: buffer-shape decisions live AT the end of
graph optimization, not in the middle. Optimizations that change
buffer shape (future double-buffering / dead-temp-elim) need to run
on un-expanded shapes; expansion happens once at materialize, where
it has full visibility of the post-optimization graph.

What this module does
---------------------
1. Build ``name → expanded tir.Buffer`` mapping for every BufferNode
   that carries ``ATTR_LANE_LAYOUT``. Reuses the legacy
   ``_expand_buffer`` helper for the actual shape rewrite.
2. Walk the graph, returning a NEW graph where:
   * every ``GraphNode.op_call`` has its inner ``BufferLoad`` /
     ``BufferRegion`` references rewritten to the expanded buffer with
     lane-folded indices;
   * every ``BufferAccess`` carries the expanded shape's starts /
     extents (same fold rules as op_call indices);
   * every ``RawStmt`` has its underlying TIR rewritten via the legacy
     ``_Rewriter`` (so BufferStore/BufferLoad inside RawStmts also pick
     up the expansion).
3. Replace ``LaneGroup.alloc_buffers`` / ``NodeRoot.alloc_buffers`` /
   ``Graph.buffer_map`` with the expanded ``tir.Buffer`` objects.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import tvm
from tvm import tir

from .... import scope as _scope
from ..graph_ir import (
    Graph, GraphNode, LaneGroup, NestedForGroup, NodeRoot, ForRoot, RootItem,
    RawStmt, BufferAccess, BufferNode,
    ATTR_LANE_LAYOUT, LAYOUT_COL_PACK, LAYOUT_ROW_STACK, LAYOUT_FP_LANE,
)


# ---------------------------------------------------------------------------
# Buffer expansion + stmt rewriter (inlined from the legacy stmt-walker
# ``allocate_group_memory`` module). These are the actual mechanics that
# turn a per-lane 2D buffer into a 4D lane-expanded buffer and rewrite
# every BufferLoad / BufferStore reference to it.
# ---------------------------------------------------------------------------

# Layout mode strings used in the (lane_expr, factor, mode) info tuple
# below. Same values as the public ``LAYOUT_*`` constants in graph_ir,
# kept duplicated as locals because the legacy `_Rewriter` checks
# `mode == FP_LANE` etc by string identity.
COL_PACK = "col_pack"
ROW_STACK = "row_stack"
FP_LANE = "fp_lane"
# Non-lane-fused 2D VRAM/MRAM buffer that still needs canonical 4D BSHD
# shape so downstream passes see one shape rank. This is the catch-all
# mode for buffers that aren't touched by a sync op (BTMM / lane-fused
# T.copy) but whose users (row_*_at, fp_at, DMA slice) expect 4D BSHD.
# Shape transformation: ``(rows, cols) → (1, rows, 1, cols)``;
# index fold: ``[r, c] → [0, r, 0, c]``.
BSHD_LIFT = "bshd_lift"


class _ExpandBuffersError(RuntimeError):
    pass


def _expand_buffer(buf: tir.Buffer, factor: int, mode: str) -> tir.Buffer:
    """Expand a per-lane buffer to a multi-lane buffer, in canonical BSHD.

      * COL_PACK:  ``(rows, last) → (1, rows, lane_count, last)`` — H axis
        carries the lane (narrow-D packing within an mlen-row).
      * ROW_STACK: ``(rows, mlen) → (lane_count, rows, 1, mlen)`` — B axis
        carries the lane (each lane's full tile stacked vertically in
        VRAM, matching the BMM_WO write pattern
        ``base + (j*mlen + i)*mlen``).
      * FP_LANE:   ``(N,) → (lane_count, N)``.

    Both VRAM/MRAM modes produce a 4D BSHD shape — isa_pass / address_alloc
    / lower_fp_row_patterns only ever see one layout family.
    """
    shape = list(buf.shape)
    one = tir.IntImm("int32", 1)
    lane_imm = tir.IntImm("int32", int(factor))
    if mode == FP_LANE:
        if len(shape) != 1:
            raise _ExpandBuffersError(
                f"buffer {buf.name!r}: FPRAM lane expansion expects rank-1 "
                f"pre-shape; got rank {len(shape)} ({shape})"
            )
        new_shape = [lane_imm, shape[0]]
    elif len(shape) != 2:
        raise _ExpandBuffersError(
            f"buffer {buf.name!r}: expansion only supports 2D pre-shapes "
            f"for VRAM/MRAM roles; got rank {len(shape)} ({shape})"
        )
    else:
        rows, last = shape
        if mode == COL_PACK:
            new_shape = [one, rows, lane_imm, last]
        elif mode == ROW_STACK:
            new_shape = [lane_imm, rows, one, last]
        elif mode == BSHD_LIFT:
            # No lane fusion — just lift 2D (rows, cols) into the
            # canonical (B=1, S=rows, H=1, D=cols) BSHD slot. Downstream
            # passes (address_alloc, isa_pass) only see 4D BSHD.
            new_shape = [one, rows, one, last]
        else:
            raise _ExpandBuffersError(f"unknown mode {mode!r}")
    declared_scope = buf.scope() if callable(getattr(buf, "scope", None)) else "global"
    new_data = tir.Var(buf.data.name, tvm.ir.PointerType(
        tvm.ir.PrimType(buf.dtype), declared_scope,
    ))
    return tir.decl_buffer(
        shape=new_shape, dtype=buf.dtype, name=buf.name,
        data=new_data, scope=declared_scope,
    )


class _StmtRewriter:
    """Rewrite a TIR Stmt subtree, swapping every reference to a tagged
    buffer for its expanded version and folding the lane axis into
    indices. Used directly on RawStmt-wrapped TIR; also used as the
    expression rewriter for op_call and BufferAccess in the graph
    walker below."""

    def __init__(self, info: Dict[str, Tuple["tir.PrimExpr", int, str]],
                 lane_count: int):
        self.info = info
        self.lane_count = lane_count
        self.name_to_new: Dict[str, tir.Buffer] = {}
        self.var_to_new: Dict[tir.Var, tir.Var] = {}

    def visit(self, n):
        if isinstance(n, tir.SeqStmt):
            return tir.SeqStmt([self.visit(c) for c in n.seq])
        if isinstance(n, tir.BlockRealize):
            return tir.BlockRealize(
                iter_values=[self.visit_expr(v) for v in n.iter_values],
                predicate=self.visit_expr(n.predicate),
                block=self.visit(n.block),
            )
        if isinstance(n, tir.Block):
            new_allocs = [self.name_to_new.get(b.name, b)
                          for b in n.alloc_buffers]
            return tir.Block(
                iter_vars=n.iter_vars, reads=n.reads, writes=n.writes,
                name_hint=n.name_hint, body=self.visit(n.body),
                init=self.visit(n.init) if n.init is not None else None,
                alloc_buffers=new_allocs,
                match_buffers=n.match_buffers, annotations=n.annotations,
            )
        if isinstance(n, tir.AttrStmt):
            return tir.AttrStmt(
                n.node, n.attr_key,
                self.visit_expr(n.value), self.visit(n.body),
            )
        if isinstance(n, tir.For):
            return tir.For(
                n.loop_var, self.visit_expr(n.min), self.visit_expr(n.extent),
                n.kind, self.visit(n.body), n.thread_binding, n.annotations,
            )
        if isinstance(n, tir.LetStmt):
            return tir.LetStmt(n.var, self.visit_expr(n.value), self.visit(n.body))
        if isinstance(n, tir.IfThenElse):
            return tir.IfThenElse(
                self.visit_expr(n.condition),
                self.visit(n.then_case),
                self.visit(n.else_case) if n.else_case is not None else None,
            )
        if isinstance(n, tir.Evaluate):
            return tir.Evaluate(self.visit_expr(n.value))
        if isinstance(n, tir.BufferStore):
            return self.visit_expr(n)
        return n

    def _fold_lane(self, indices, buf_name):
        """Lift 2D per-lane indices to 4D BSHD, inserting the lane axis.

          COL_PACK  2D [r, c] → 4D [0,  r,  by, c]   (H carries lane)
          ROW_STACK 2D [r, c] → 4D [by, r,  0,  c]   (B carries lane)
          FP_LANE   1D [r]    → 2D [by, r]

        Already-folded indices (idempotent re-walk) are left untouched.
        """
        if buf_name not in self.info or not indices:
            return indices
        lane_expr, _factor, mode = self.info[buf_name]
        if mode == FP_LANE:
            if len(indices) == 2:
                return list(indices)
            if len(indices) != 1:
                raise _ExpandBuffersError(
                    f"buffer {buf_name!r} access has rank {len(indices)}; "
                    f"_fold_lane expects pre-expansion rank 1 for fpram"
                )
            return [lane_expr, indices[0]]
        if len(indices) == 4:
            return list(indices)
        if len(indices) != 2:
            raise _ExpandBuffersError(
                f"buffer {buf_name!r} access has rank {len(indices)}; "
                f"_fold_lane expects pre-expansion rank 2"
            )
        zero_dtype = getattr(lane_expr, "dtype", "int32")
        zero = tir.IntImm(zero_dtype, 0)
        r, c = indices
        if mode == COL_PACK:
            return [zero, r, lane_expr, c]
        if mode == BSHD_LIFT:
            # No lane axis to fold — just insert unit B and H dims.
            return [zero, r, zero, c]
        # ROW_STACK: lane lives in B axis.
        return [lane_expr, r, zero, c]

    def visit_expr(self, e):
        if isinstance(e, tir.Var):
            return self.var_to_new.get(e, e)
        if isinstance(e, tir.BufferLoad):
            new_buf = self.name_to_new.get(e.buffer.name, e.buffer)
            indices = [self.visit_expr(i) for i in e.indices]
            indices = self._fold_lane(indices, e.buffer.name)
            return tir.BufferLoad(new_buf, indices)
        if isinstance(e, tir.BufferStore):
            new_buf = self.name_to_new.get(e.buffer.name, e.buffer)
            indices = [self.visit_expr(i) for i in e.indices]
            indices = self._fold_lane(indices, e.buffer.name)
            return tir.BufferStore(new_buf, self.visit_expr(e.value), indices)
        if isinstance(e, tir.Call):
            return tir.Call(e.dtype, e.op, [self.visit_expr(a) for a in e.args])
        if isinstance(e, tir.Cast):
            return type(e)(e.dtype, self.visit_expr(e.value))
        if hasattr(e, "a") and hasattr(e, "b"):
            return type(e)(self.visit_expr(e.a), self.visit_expr(e.b))
        return e


# ---------------------------------------------------------------------------
# Build the (name → expanded tir.Buffer) map and the matching info dict
# ---------------------------------------------------------------------------

# Map from graph-IR layout names to legacy stmt-walker mode strings.
_LAYOUT_TO_MODE = {
    LAYOUT_COL_PACK: COL_PACK,
    LAYOUT_ROW_STACK: ROW_STACK,
    LAYOUT_FP_LANE: FP_LANE,
}


def _collect_alloc_buffers_with_buffers(graph: Graph) -> Dict[str, tir.Buffer]:
    """Collect every alloc'd / param tir.Buffer into a name → buffer
    dict. Used to look up the original tir.Buffer when expanding."""
    out: Dict[str, tir.Buffer] = {}

    for buf in graph.buffer_map.values():
        out[buf.name] = buf

    def walk(root: RootItem):
        if isinstance(root, LaneGroup):
            for buf in root.alloc_buffers:
                out[buf.name] = buf
            return
        if isinstance(root, NodeRoot):
            for buf in root.alloc_buffers:
                out[buf.name] = buf
            return
        if isinstance(root, ForRoot):
            walk(root.body)
            return

    walk(graph.root)
    return out


def _collect_lane_vars(graph: Graph) -> Dict[str, tir.Var]:
    """Walk every for-node in the graph; return a ``name → tir.Var``
    map of every loop_var. Used so we can recover the actual ``tir.Var``
    that ``ATTR_LANE_VAR`` (a string name) refers to.

    The legacy ``_Rewriter._fold_lane`` inserts the lane var into folded
    indices using object identity; if we synthesise a fresh Var with the
    same name we'd produce indices that reference an unbound symbol
    (different Var object than the for's loop_var). Grab the real one."""
    out: Dict[str, tir.Var] = {}

    def visit_items(items):
        for it in items:
            if isinstance(it, NestedForGroup):
                if it.loop_var is not None:
                    out.setdefault(it.loop_var.name, it.loop_var)
                visit_items(it.items)

    def visit_root(root):
        if isinstance(root, ForRoot):
            if root.loop_var is not None:
                out.setdefault(root.loop_var.name, root.loop_var)
            visit_root(root.body)
            return
        if isinstance(root, LaneGroup):
            if root.lane_var is not None:
                out.setdefault(root.lane_var.name, root.lane_var)
            visit_items(root.items)
            return
        if isinstance(root, NodeRoot):
            visit_items(root.items)
            return

    visit_root(graph.root)
    return out


def _build_expansion(graph: Graph,
                     lane_count: int,
                     scopes: Optional[Dict[str, str]] = None,
                     ) -> Tuple[Dict[str, tir.Buffer], Dict[str, tuple]]:
    """Return (name → expanded tir.Buffer, name → (lane_expr, factor, mode))
    suitable for feeding into the legacy ``_Rewriter``.

    Two passes over the buffers:

      1. **lane-fused** — every BufferNode that ``g_alloc.analyze`` tagged
         with ``ATTR_LANE_LAYOUT`` (COL_PACK / ROW_STACK / FP_LANE). Mode
         comes from the layout tag, lane var from ``ATTR_LANE_VAR``.

      2. **non-lane-fused 2D BSHD lift** — every remaining 2D VRAM/MRAM
         alloc that wasn't picked up above. These buffers don't carry a
         lane axis but still need their shape promoted to 4D BSHD so the
         backend (address_alloc, isa_pass) sees one shape rank. Falls
         under :data:`BSHD_LIFT` mode; index fold inserts unit B/H dims.

    ``global.*`` scoped buffers are skipped from BSHD_LIFT — those are a
    user-facing escape hatch where the kernel author chose the explicit
    2D semantic (e.g. ``Q_cache(head_count, hlen)`` in flash_decode_min);
    auto-lifting them would assign the wrong layout role.
    """
    name_to_buf = _collect_alloc_buffers_with_buffers(graph)
    expanded: Dict[str, tir.Buffer] = {}
    info: Dict[str, tuple] = {}
    lane_vars = _collect_lane_vars(graph)

    for name, bn in graph.buffer_nodes.items():
        layout = bn.attrs.get(ATTR_LANE_LAYOUT)
        if layout is None:
            continue
        mode = _LAYOUT_TO_MODE[layout]
        lane_var_name = bn.attrs.get("lane_var_name")
        # Recover the actual tir.Var (not a synthetic same-named one)
        # so folded indices reference the correct symbol — the for-loop
        # the lane var is bound by emits the same Var object.
        lane_expr = lane_vars.get(lane_var_name)
        if lane_expr is None:
            # Shouldn't happen if analyze() saw this lane var; defensive.
            lane_expr = tir.Var(lane_var_name, "int32")
        old_buf = name_to_buf.get(name)
        if old_buf is None:
            continue
        new_buf = _expand_buffer(old_buf, lane_count, mode)
        expanded[name] = new_buf
        info[name] = (lane_expr, lane_count, mode)

    # Second pass: BSHD-lift remaining 2D VRAM/MRAM allocs that weren't
    # picked up by the lane-fusion pass above. Buffer scopes at this
    # point are still the user-facing ``shared.dyn`` / ``local.fragment``
    # tags (the final scope rewrite to ``vram`` / ``mram`` happens after
    # materialize), so we consult ``scopes`` (the result of
    # scope_inference) to decide eligibility.
    for name, old_buf in name_to_buf.items():
        if name in expanded:
            continue
        if len(old_buf.shape) != 2:
            continue
        declared_scope = (
            old_buf.scope() if callable(getattr(old_buf, "scope", None))
            else "global"
        )
        if _scope.is_global_scope(declared_scope):
            continue
        resolved_scope = None
        if scopes is not None:
            resolved_scope = scopes.get(name)
        if resolved_scope is None:
            continue
        if _scope.is_global_scope(resolved_scope):
            continue
        phys = _scope.physical_scope(resolved_scope)
        if phys not in (_scope.VRAM, _scope.MRAM):
            continue
        # BSHD_LIFT mode: no lane var needed. Pass a constant 0 so the
        # _fold_lane path's BSHD_LIFT branch can still read the lane_expr
        # without raising.
        zero_expr = tir.IntImm("int32", 0)
        new_buf = _expand_buffer(old_buf, 1, BSHD_LIFT)
        expanded[name] = new_buf
        info[name] = (zero_expr, 1, BSHD_LIFT)

    return expanded, info


# ---------------------------------------------------------------------------
# Stmt rewriter (delegates to the legacy _StmtRewriter for BufferLoad /
# BufferStore / Call / Var rewriting). The legacy class already handles
# the index fold and the data-Var substitution we need.
# ---------------------------------------------------------------------------

def _rewrite_call(call: tir.Call, rw: _StmtRewriter) -> tir.Call:
    """Rewrite a tir.Call (op_call) via the legacy stmt rewriter.
    ``visit_expr`` already handles tir.Call recursively."""
    return rw.visit_expr(call)


def _rewrite_access(access: BufferAccess,
                    rw: _StmtRewriter,
                    expanded: Dict[str, tir.Buffer]) -> BufferAccess:
    """Expand a BufferAccess to the new buffer's rank, folding the lane
    axis the same way ``_fold_lane`` does for BufferLoad indices."""
    name = access.buffer_name
    if name not in expanded:
        # Untouched buffer; just rewrite each PrimExpr in starts/extents
        # (their .data Vars stay the same, but a child Var ref may need
        # substitution if it referenced a renamed buffer's data var —
        # rare but defensive).
        return BufferAccess(
            buffer_name=name,
            starts=[rw.visit_expr(s) for s in access.starts],
            extents=[rw.visit_expr(e) for e in access.extents],
        )
    new_starts = [rw.visit_expr(s) for s in access.starts]
    new_extents = [rw.visit_expr(e) for e in access.extents]
    new_starts = rw._fold_lane(new_starts, name)
    # For extents, the lane axis becomes 1 (single lane covered per
    # access). The other axes carry their original extents in the new
    # rank's slots — same shape transformation as `_fold_lane` but
    # with extent-1 in the lane slot.
    new_extents = _fold_extents(new_extents, name, rw)
    return BufferAccess(
        buffer_name=name, starts=new_starts, extents=new_extents,
    )


def _fold_extents(extents, buf_name: str, rw: _StmtRewriter):
    """Mirror of ``_Rewriter._fold_lane`` for extents — the lane slot
    gets a unit extent (the access touches one lane at a time)."""
    if buf_name not in rw.info or not extents:
        return list(extents)
    _lane_expr, _factor, mode = rw.info[buf_name]
    one = tir.IntImm("int32", 1)
    if mode == FP_LANE:
        if len(extents) == 2:
            return list(extents)
        if len(extents) == 1:
            return [one, extents[0]]
        return list(extents)
    if len(extents) == 4:
        return list(extents)
    if len(extents) != 2:
        return list(extents)
    r, c = extents
    if mode == COL_PACK:
        return [one, r, one, c]
    if mode == BSHD_LIFT:
        # No lane axis — extents are just (rows, cols) in the S+D slot.
        return [one, r, one, c]
    return [one, one, r, c]


# ---------------------------------------------------------------------------
# Walk graph and rewrite
# ---------------------------------------------------------------------------

def _rewrite_items(items, rw: _StmtRewriter,
                   expanded: Dict[str, tir.Buffer]):
    out = []
    for it in items:
        if isinstance(it, GraphNode):
            new_call = _rewrite_call(it.op_call, rw)
            out.append(GraphNode(
                name=it.name, op_call=new_call, attrs=dict(it.attrs),
                reads=[_rewrite_access(a, rw, expanded) for a in it.reads],
                writes=[_rewrite_access(a, rw, expanded) for a in it.writes],
            ))
        elif isinstance(it, NestedForGroup):
            out.append(NestedForGroup(
                loop_var=it.loop_var,
                min=rw.visit_expr(it.min),
                extent=rw.visit_expr(it.extent),
                kind=it.kind, thread_binding=it.thread_binding,
                annotations=it.annotations,
                items=_rewrite_items(it.items, rw, expanded),
                attrs=dict(it.attrs),
            ))
        elif isinstance(it, RawStmt):
            out.append(RawStmt(
                name=it.name,
                stmt=rw.visit(it.stmt),
            ))
        else:
            out.append(it)
    return out


def _rewrite_root(root: RootItem, rw: _StmtRewriter,
                  expanded: Dict[str, tir.Buffer]) -> RootItem:
    if isinstance(root, ForRoot):
        return ForRoot(
            loop_var=root.loop_var,
            min=rw.visit_expr(root.min),
            extent=rw.visit_expr(root.extent),
            kind=root.kind, thread_binding=root.thread_binding,
            annotations=root.annotations,
            body=_rewrite_root(root.body, rw, expanded),
            attrs=dict(root.attrs),
        )
    if isinstance(root, LaneGroup):
        return LaneGroup(
            lane_var=root.lane_var, lane_count=root.lane_count,
            items=_rewrite_items(root.items, rw, expanded),
            alloc_buffers=[expanded.get(b.name, b) for b in root.alloc_buffers],
        )
    if isinstance(root, NodeRoot):
        return NodeRoot(
            items=_rewrite_items(root.items, rw, expanded),
            alloc_buffers=[expanded.get(b.name, b) for b in root.alloc_buffers],
        )
    return root


def _rewrite_buffer_map(buffer_map: Dict[tir.Var, tir.Buffer],
                        expanded: Dict[str, tir.Buffer],
                        rw: _StmtRewriter
                        ) -> Dict[tir.Var, tir.Buffer]:
    """Replace any param buffer that got expanded. The Var key changes
    too because ``_expand_buffer`` minted a fresh tir.Var for the new
    buffer's data handle, so the old ``buf.data`` is no longer the
    canonical handle — but the param list (PrimFunc.params) still
    references the old Var. We keep the old Var as the key (params
    don't change) and just point it at the new buffer. The data-Var
    substitution inside the rewriter (``rw.var_to_new``) handles call
    args that reference the OLD data Var — they get redirected to the
    new one. For buffer_map we want the parameter binding intact, so
    keep the old key.
    """
    out: Dict[tir.Var, tir.Buffer] = {}
    for k, buf in buffer_map.items():
        new_buf = expanded.get(buf.name, buf)
        if new_buf is not buf:
            # Bind the original param var to a fresh buffer that
            # uses the original param Var as data (so PrimFunc
            # signature stays consistent). Rebuild via decl_buffer.
            from tvm import tir as _tir
            out[k] = _tir.decl_buffer(
                shape=new_buf.shape, dtype=new_buf.dtype,
                name=new_buf.name, data=k,
                scope=k.type_annotation.storage_scope
                if hasattr(k.type_annotation, "storage_scope") else "global",
            )
        else:
            out[k] = buf
    return out


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

def expand(graph: Graph,
           lane_count: int = 4,
           scopes: Optional[Dict[str, str]] = None) -> Graph:
    """Expand every BufferNode tagged with ``ATTR_LANE_LAYOUT`` and
    rewrite the graph to use the expanded buffers.

    When ``scopes`` is provided, additionally BSHD-lift any remaining 2D
    VRAM/MRAM allocs that the lane-fusion pass didn't touch — see
    :func:`_build_expansion`.

    Returns a NEW Graph. ``buffer_nodes`` is preserved as-is (passes
    that consumed ATTR_LANE_LAYOUT may want to read it).
    """
    expanded, info = _build_expansion(graph, lane_count, scopes=scopes)
    if not expanded:
        return graph

    rw = _StmtRewriter(info, lane_count)
    # Pre-populate name_to_new / var_to_new so the StmtRewriter's
    # rewrite paths see the expanded buffers immediately. The legacy
    # `_Rewriter._expand` lazily builds these via `_expand_buffer`;
    # we already did the expansion, so just install the mapping
    # directly.
    for name, new_buf in expanded.items():
        rw.name_to_new[name] = new_buf
        # Map old data Var → new data Var. Pull old var from any
        # alloc_buffer / buffer_map entry sharing this name.
        old_buf = _find_old_buffer(graph, name)
        if old_buf is not None and old_buf.data is not new_buf.data:
            rw.var_to_new[old_buf.data] = new_buf.data

    new_root = _rewrite_root(graph.root, rw, expanded)
    new_buffer_map = _rewrite_buffer_map(graph.buffer_map, expanded, rw)

    return Graph(
        root=new_root,
        params=graph.params,
        buffer_map=new_buffer_map,
        ret_type=graph.ret_type,
        attrs=graph.attrs,
        buffer_nodes=graph.buffer_nodes,
    )


def _find_old_buffer(graph: Graph, name: str) -> Optional[tir.Buffer]:
    for buf in graph.buffer_map.values():
        if buf.name == name:
            return buf

    def walk(root):
        if isinstance(root, LaneGroup):
            for buf in root.alloc_buffers:
                if buf.name == name:
                    return buf
            return None
        if isinstance(root, NodeRoot):
            for buf in root.alloc_buffers:
                if buf.name == name:
                    return buf
            return None
        if isinstance(root, ForRoot):
            return walk(root.body)
        return None

    return walk(graph.root)


__all__ = ["expand"]
