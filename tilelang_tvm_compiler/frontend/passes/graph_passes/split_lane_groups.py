"""Graph pass: split a lane-fusion-eligible group axis whose extent
exceeds ``lane_count`` into ``outer × lane_count``.

Graph-IR replacement for the legacy stmt-walker
``frontend/passes/split_lane_groups.py``. Equivalent split semantics,
but operating on graph items (ForRoot / NestedForGroup) rather than
rewriting `tir.For` + `T.attr(plena.group)` pairs.

When does the split fire?
-------------------------
A ForRoot / NestedForGroup is a split candidate iff:
  * it carries ``attrs[ATTR_GROUP_EXTENT] = N`` (set by annotate_grid);
  * ``N > lane_count`` and ``N % lane_count == 0``;
  * the body (recursively) contains a sync GraphNode (``ATTR_IS_SYNC``)
    whose ``op_call`` references the for's ``loop_var``.

When all three hold, the for is replaced with::

    NestedForGroup(loop_var=v_outer, extent=N/lane_count,
                   attrs={ATTR_GROUP_EXTENT: N/lane_count},
                   items=[NestedForGroup(loop_var=v_inner, extent=lane_count,
                                         attrs={ATTR_GROUP_EXTENT: lane_count,
                                                ATTR_IS_LANE_FOR: True},
                                         items=<body with v→v_outer*lane+v_inner>)])

(or a ``ForRoot`` outermost if the original was a ForRoot)

Graph items below the split — every GraphNode's ``op_call.args``, every
``BufferAccess.starts`` / ``extents``, every nested NestedForGroup's
``min`` / ``extent``, every RawStmt's underlying tir Stmt — get the
substitution ``v → v_outer*lane_count + v_inner`` applied.

Pre-conditions
--------------
* :func:`annotate_grid.run` has populated ``ATTR_GROUP_EXTENT``.
* :func:`annotate_sync.run` has populated ``ATTR_IS_SYNC``.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Set, Union

from tvm import tir

from ..graph_ir import (
    Graph, GraphNode, LaneGroup, NestedForGroup, NodeRoot, ForRoot, RootItem,
    RawStmt, BufferAccess,
    ATTR_GROUP_EXTENT, ATTR_IS_LANE_FOR, ATTR_IS_SYNC,
)


class SplitLaneGroupError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# TIR var substitution (recursively rewrite Stmt and Expr trees,
# replacing every occurrence of a Var with its mapped expression).
# Inlined from the legacy stmt-walker ``annotate_group._VarSubst`` —
# only consumer is the graph-layer _GraphVarSubst below.
# ---------------------------------------------------------------------------

class _StmtVarSubst:
    def __init__(self, sub: Dict[tir.Var, "tir.PrimExpr"]):
        self.sub = sub
        self.sub_by_name = {v.name: e for v, e in sub.items()}

    def _lookup(self, var: tir.Var):
        if var in self.sub:
            return self.sub[var]
        return self.sub_by_name.get(var.name, var)

    def run(self, node):
        return self._visit(node)

    def _visit(self, n):
        if isinstance(n, tir.SeqStmt):
            return tir.SeqStmt([self._visit(c) for c in n.seq])
        if isinstance(n, tir.BlockRealize):
            return tir.BlockRealize(
                iter_values=[self._visit(v) for v in n.iter_values],
                predicate=self._visit(n.predicate),
                block=self._visit(n.block),
            )
        if isinstance(n, tir.Block):
            return tir.Block(
                iter_vars=n.iter_vars, reads=n.reads, writes=n.writes,
                name_hint=n.name_hint, body=self._visit(n.body),
                init=self._visit(n.init) if n.init is not None else None,
                alloc_buffers=n.alloc_buffers,
                match_buffers=n.match_buffers, annotations=n.annotations,
            )
        if isinstance(n, tir.AttrStmt):
            return tir.AttrStmt(n.node, n.attr_key,
                                self._visit(n.value), self._visit(n.body))
        if isinstance(n, tir.For):
            return tir.For(
                n.loop_var, self._visit(n.min), self._visit(n.extent),
                n.kind, self._visit(n.body), n.thread_binding, n.annotations,
            )
        if isinstance(n, tir.Evaluate):
            return tir.Evaluate(self._visit(n.value))
        if isinstance(n, tir.IfThenElse):
            return tir.IfThenElse(
                self._visit(n.condition),
                self._visit(n.then_case),
                self._visit(n.else_case) if n.else_case is not None else None,
            )
        if isinstance(n, tir.LetStmt):
            return tir.LetStmt(n.var, self._visit(n.value), self._visit(n.body))
        if isinstance(n, tir.BufferStore):
            return tir.BufferStore(
                n.buffer, self._visit(n.value),
                [self._visit(i) for i in n.indices],
            )
        if isinstance(n, tir.BufferLoad):
            return tir.BufferLoad(
                n.buffer, [self._visit(i) for i in n.indices],
            )
        if isinstance(n, tir.Call):
            return tir.Call(n.dtype, n.op, [self._visit(a) for a in n.args])
        if isinstance(n, tir.Var):
            return self._lookup(n)
        if isinstance(n, (tir.IntImm, tir.FloatImm, tir.StringImm)):
            return n
        # Common arithmetic: tir.Add/Sub/Mul/FloorDiv/FloorMod/Min/Max all
        # have (a, b). Reconstruct via the same constructor.
        if hasattr(n, "a") and hasattr(n, "b"):
            return type(n)(self._visit(n.a), self._visit(n.b))
        return n


# ---------------------------------------------------------------------------
# Free-var collection over graph items
# ---------------------------------------------------------------------------

def _collect_used_var_names_in_expr(expr: "tir.PrimExpr", out: Set[str]) -> None:
    """Recurse a TIR PrimExpr / Stmt subtree, adding every Var name into
    ``out``."""
    if expr is None:
        return
    if isinstance(expr, tir.Var):
        out.add(expr.name)
        return
    if isinstance(expr, (tir.IntImm, tir.FloatImm, tir.StringImm)):
        return
    if isinstance(expr, tir.BufferLoad):
        for i in expr.indices:
            _collect_used_var_names_in_expr(i, out)
        return
    if isinstance(expr, tir.BufferStore):
        _collect_used_var_names_in_expr(expr.value, out)
        for i in expr.indices:
            _collect_used_var_names_in_expr(i, out)
        return
    if isinstance(expr, tir.Call):
        for a in expr.args:
            _collect_used_var_names_in_expr(a, out)
        return
    # Generic Add/Mul/...: recurse via children.
    for attr in ("a", "b", "value", "condition", "true_value", "false_value"):
        child = getattr(expr, attr, None)
        if child is not None:
            _collect_used_var_names_in_expr(child, out)


def _collect_used_var_names_in_stmt(stmt: "tir.Stmt", out: Set[str]) -> None:
    if stmt is None:
        return
    if isinstance(stmt, tir.SeqStmt):
        for c in stmt.seq:
            _collect_used_var_names_in_stmt(c, out)
        return
    if isinstance(stmt, tir.AttrStmt):
        _collect_used_var_names_in_expr(stmt.value, out)
        _collect_used_var_names_in_stmt(stmt.body, out)
        return
    if isinstance(stmt, tir.For):
        _collect_used_var_names_in_expr(stmt.min, out)
        _collect_used_var_names_in_expr(stmt.extent, out)
        _collect_used_var_names_in_stmt(stmt.body, out)
        return
    if isinstance(stmt, tir.Evaluate):
        _collect_used_var_names_in_expr(stmt.value, out)
        return
    if isinstance(stmt, tir.IfThenElse):
        _collect_used_var_names_in_expr(stmt.condition, out)
        _collect_used_var_names_in_stmt(stmt.then_case, out)
        if stmt.else_case is not None:
            _collect_used_var_names_in_stmt(stmt.else_case, out)
        return
    if isinstance(stmt, tir.LetStmt):
        _collect_used_var_names_in_expr(stmt.value, out)
        _collect_used_var_names_in_stmt(stmt.body, out)
        return
    if isinstance(stmt, tir.BufferStore):
        _collect_used_var_names_in_expr(stmt, out)
        return
    if isinstance(stmt, tir.BlockRealize):
        for v in stmt.iter_values:
            _collect_used_var_names_in_expr(v, out)
        _collect_used_var_names_in_stmt(stmt.block.body, out)
        return


def _collect_used_var_names_in_access(access: BufferAccess, out: Set[str]) -> None:
    for s in access.starts:
        _collect_used_var_names_in_expr(s, out)
    for e in access.extents:
        _collect_used_var_names_in_expr(e, out)


def _collect_used_var_names_in_node(node: GraphNode, out: Set[str]) -> None:
    _collect_used_var_names_in_expr(node.op_call, out)
    for a in node.reads:
        _collect_used_var_names_in_access(a, out)
    for a in node.writes:
        _collect_used_var_names_in_access(a, out)


def _collect_used_var_names_in_items(items, out: Set[str]) -> None:
    for it in items:
        if isinstance(it, GraphNode):
            _collect_used_var_names_in_node(it, out)
        elif isinstance(it, NestedForGroup):
            _collect_used_var_names_in_expr(it.min, out)
            _collect_used_var_names_in_expr(it.extent, out)
            _collect_used_var_names_in_items(it.items, out)
        elif isinstance(it, RawStmt):
            _collect_used_var_names_in_stmt(it.stmt, out)


# ---------------------------------------------------------------------------
# "Does any sync GraphNode below reference var_name?"
# ---------------------------------------------------------------------------

def _sync_uses_var_in_items(items, var_name: str) -> bool:
    for it in items:
        if isinstance(it, GraphNode):
            if it.attrs.get(ATTR_IS_SYNC):
                used: Set[str] = set()
                _collect_used_var_names_in_node(it, used)
                if var_name in used:
                    return True
        elif isinstance(it, NestedForGroup):
            if _sync_uses_var_in_items(it.items, var_name):
                return True
    return False


# ---------------------------------------------------------------------------
# Var substitution over graph items
# ---------------------------------------------------------------------------

class _GraphVarSubst:
    """Apply ``var → expr`` substitution across a graph subtree.

    Reuses the existing stmt-walker ``_VarSubst`` to handle TIR PrimExpr
    / Stmt; wraps it in graph-item recursion."""

    def __init__(self, sub: Dict[tir.Var, "tir.PrimExpr"]):
        self._stmt_subst = _StmtVarSubst(sub)

    def _expr(self, e):
        if e is None:
            return None
        return self._stmt_subst.run(e)

    def _access(self, a: BufferAccess) -> BufferAccess:
        return BufferAccess(
            buffer_name=a.buffer_name,
            starts=[self._expr(s) for s in a.starts],
            extents=[self._expr(e) for e in a.extents],
        )

    def _node(self, n: GraphNode) -> GraphNode:
        new_call = self._expr(n.op_call)
        return GraphNode(
            name=n.name,
            op_call=new_call,
            attrs=dict(n.attrs),
            reads=[self._access(a) for a in n.reads],
            writes=[self._access(a) for a in n.writes],
        )

    def _raw(self, r: RawStmt) -> RawStmt:
        return RawStmt(name=r.name, stmt=self._stmt_subst.run(r.stmt))

    def items(self, items):
        out = []
        for it in items:
            if isinstance(it, GraphNode):
                out.append(self._node(it))
            elif isinstance(it, NestedForGroup):
                out.append(NestedForGroup(
                    loop_var=it.loop_var,
                    min=self._expr(it.min),
                    extent=self._expr(it.extent),
                    kind=it.kind,
                    thread_binding=it.thread_binding,
                    annotations=it.annotations,
                    items=self.items(it.items),
                    attrs=dict(it.attrs),
                ))
            elif isinstance(it, RawStmt):
                out.append(self._raw(it))
            else:
                out.append(it)
        return out


# ---------------------------------------------------------------------------
# The split itself
# ---------------------------------------------------------------------------

def _split_into_pair(loop_var: tir.Var,
                     N: int,
                     lane_count: int,
                     body_items) -> NestedForGroup:
    """Build the inner ``outer_for(NestedForGroup) × inner_for(NestedForGroup)``
    nesting that replaces a single split-target for. Caller decides whether
    the result is a NestedForGroup (interior) or wrapped in a ForRoot (root)."""
    if N % lane_count != 0:
        raise SplitLaneGroupError(
            f"group extent {N} not divisible by lane_count {lane_count}"
        )
    outer_extent = N // lane_count

    v_outer = tir.Var(f"{loop_var.name}_o", loop_var.dtype)
    v_inner = tir.Var(f"{loop_var.name}_i", loop_var.dtype)
    new_v_expr = v_outer * tir.IntImm(loop_var.dtype, lane_count) + v_inner

    rewritten = _GraphVarSubst({loop_var: new_v_expr}).items(body_items)

    inner = NestedForGroup(
        loop_var=v_inner,
        min=tir.IntImm(loop_var.dtype, 0),
        extent=tir.IntImm(loop_var.dtype, lane_count),
        kind=tir.ForKind.SERIAL,
        thread_binding=None,
        annotations=None,
        items=rewritten,
        attrs={
            ATTR_GROUP_EXTENT: lane_count,
            ATTR_IS_LANE_FOR: True,
        },
    )
    outer = NestedForGroup(
        loop_var=v_outer,
        min=tir.IntImm(loop_var.dtype, 0),
        extent=tir.IntImm(loop_var.dtype, outer_extent),
        kind=tir.ForKind.SERIAL,
        thread_binding=None,
        annotations=None,
        items=[inner],
        attrs={ATTR_GROUP_EXTENT: outer_extent},
    )
    return outer


# ---------------------------------------------------------------------------
# Walker
# ---------------------------------------------------------------------------

def _walk_items(items, lane_count: int):
    """Walk a list of items, splitting any candidate NestedForGroup."""
    out = []
    for it in items:
        if isinstance(it, NestedForGroup):
            # Recurse into the body first (deepest splits fire first;
            # also handles double-nested splits).
            new_inner = _walk_items(it.items, lane_count)
            it = NestedForGroup(
                loop_var=it.loop_var, min=it.min, extent=it.extent,
                kind=it.kind, thread_binding=it.thread_binding,
                annotations=it.annotations, items=new_inner,
                attrs=dict(it.attrs),
            )
            split = _maybe_split_nested(it, lane_count)
            out.append(split if split is not None else it)
        else:
            out.append(it)
    return out


def _maybe_split_nested(forgrp: NestedForGroup, lane_count: int):
    """Return a split replacement NestedForGroup if forgrp qualifies,
    else None."""
    N = forgrp.attrs.get(ATTR_GROUP_EXTENT)
    if N is None:
        return None
    # Already split? Inner-of-pair carries ATTR_IS_LANE_FOR.
    if forgrp.attrs.get(ATTR_IS_LANE_FOR):
        return None
    if not isinstance(N, int):
        return None
    if N <= lane_count or N % lane_count != 0:
        return None
    if not _sync_uses_var_in_items(forgrp.items, forgrp.loop_var.name):
        return None
    return _split_into_pair(forgrp.loop_var, N, lane_count, forgrp.items)


def _walk_root(root: RootItem, lane_count: int) -> RootItem:
    if isinstance(root, ForRoot):
        new_body = _walk_root(root.body, lane_count)
        # Try to split the ForRoot itself.
        N = root.attrs.get(ATTR_GROUP_EXTENT)
        if (isinstance(N, int) and not root.attrs.get(ATTR_IS_LANE_FOR)
                and N > lane_count and N % lane_count == 0):
            # Reach into new_body's items if it became a LaneGroup/NodeRoot
            # (our split needs the body items, not a wrapping root). For
            # ForRoot the body is a RootItem, not items list. We synthesise
            # an items list with the new_body.
            #
            # But sync detection has to look INSIDE the new_body's
            # graph-items. Use a wrapper.
            items_for_sync_check = _root_to_items_for_sync(new_body)
            if _sync_uses_var_in_items(items_for_sync_check, root.loop_var.name):
                pair = _split_into_pair(
                    root.loop_var, N, lane_count, items_for_sync_check,
                )
                # `pair` is a NestedForGroup outer wrapping the inner.
                # The original ForRoot wrapped a RootItem (LaneGroup /
                # NodeRoot / ForRoot). After splitting we still want a
                # RootItem on the outside; rebuild as ForRoot(outer_for) →
                # ForRoot(inner_for) → original RootItem-without-its-items.
                #
                # But our current root types don't let us easily replace
                # "the inner items of a LaneGroup/NodeRoot" cleanly. The
                # cleanest move: unwrap the new_body to its items+kind,
                # rebuild as a chain of ForRoots, then re-wrap with a
                # NodeRoot/LaneGroup carrying the (now-rewritten) items.
                return _rebuild_root_with_split(
                    pair, new_body,
                )
        return ForRoot(
            loop_var=root.loop_var, min=root.min, extent=root.extent,
            kind=root.kind, thread_binding=root.thread_binding,
            annotations=root.annotations, body=new_body,
            attrs=dict(root.attrs),
        )
    if isinstance(root, LaneGroup):
        return LaneGroup(
            lane_var=root.lane_var, lane_count=root.lane_count,
            items=_walk_items(root.items, lane_count),
            alloc_buffers=list(root.alloc_buffers),
        )
    if isinstance(root, NodeRoot):
        return NodeRoot(
            items=_walk_items(root.items, lane_count),
            alloc_buffers=list(root.alloc_buffers),
        )
    return root


def _root_to_items_for_sync(root: RootItem):
    """Project a RootItem's body into a flat items list for sync-var
    detection. Doesn't materialise — only used as input to
    _sync_uses_var_in_items."""
    if isinstance(root, LaneGroup):
        return root.items
    if isinstance(root, NodeRoot):
        return root.items
    if isinstance(root, ForRoot):
        # Wrap the inner ForRoot as a single NestedForGroup-equivalent.
        # _sync_uses_var_in_items only inspects items recursively; a
        # NestedForGroup wrapper with a single item (the body's items)
        # is enough.
        nested = NestedForGroup(
            loop_var=root.loop_var, min=root.min, extent=root.extent,
            kind=root.kind, thread_binding=root.thread_binding,
            annotations=root.annotations,
            items=_root_to_items_for_sync(root.body),
            attrs=dict(root.attrs),
        )
        return [nested]
    return []


def _rebuild_root_with_split(pair: NestedForGroup, original_body: RootItem) -> RootItem:
    """The original tree was ``ForRoot(loop_var=v) → original_body``. The
    split produced a NestedForGroup pair (outer × inner) that replaces
    the for. The leaf items of the pair are the rewritten items pulled
    from ``original_body``; we now re-wrap them in original_body's leaf
    container (LaneGroup / NodeRoot)."""
    # Pull the rewritten items out of pair (they live at pair.items[0].items).
    inner = pair.items[0]
    rewritten_items = inner.items
    # Replace inner's items with the original's inner-most container's
    # items wrapping. We need to materialise as (ForRoot outer) → (ForRoot inner) → leaf.
    # Build leaf container:
    if isinstance(original_body, LaneGroup):
        leaf = LaneGroup(
            lane_var=original_body.lane_var,
            lane_count=original_body.lane_count,
            items=rewritten_items,
            alloc_buffers=list(original_body.alloc_buffers),
        )
    elif isinstance(original_body, NodeRoot):
        leaf = NodeRoot(
            items=rewritten_items,
            alloc_buffers=list(original_body.alloc_buffers),
        )
    elif isinstance(original_body, ForRoot):
        # Nested ForRoot — preserve as-is but with rewritten subtree.
        # This shouldn't fire in practice (lift_from_raw chains ForRoots
        # only for grid bindings; the inner one would have been split
        # separately). Fall back to NodeRoot(items=) carrying the
        # rewritten items as opaque pass-through.
        leaf = NodeRoot(items=rewritten_items, alloc_buffers=[])
    else:
        leaf = NodeRoot(items=rewritten_items, alloc_buffers=[])

    # Build the inner ForRoot (lane-fusion-eligible).
    inner_root = ForRoot(
        loop_var=inner.loop_var,
        min=inner.min, extent=inner.extent,
        kind=inner.kind, thread_binding=inner.thread_binding,
        annotations=inner.annotations,
        body=leaf,
        attrs=dict(inner.attrs),
    )
    # Build the outer ForRoot.
    outer_root = ForRoot(
        loop_var=pair.loop_var,
        min=pair.min, extent=pair.extent,
        kind=pair.kind, thread_binding=pair.thread_binding,
        annotations=pair.annotations,
        body=inner_root,
        attrs=dict(pair.attrs),
    )
    return outer_root


def run(graph: Graph, lane_count: int = 4) -> Graph:
    """Split lane-fusion-eligible groups whose extent exceeds ``lane_count``.

    Returns a NEW Graph with the rewritten root; ``buffer_nodes`` /
    ``buffer_map`` etc are shared with the input.
    """
    if lane_count <= 0:
        raise SplitLaneGroupError(
            f"lane_count must be positive; got {lane_count}"
        )
    new_root = _walk_root(graph.root, lane_count)
    return Graph(
        root=new_root,
        params=graph.params,
        buffer_map=graph.buffer_map,
        ret_type=graph.ret_type,
        attrs=graph.attrs,
        buffer_nodes=graph.buffer_nodes,
    )


__all__ = ["run", "SplitLaneGroupError"]
