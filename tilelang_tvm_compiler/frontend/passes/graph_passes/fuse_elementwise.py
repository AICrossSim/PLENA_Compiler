"""Graph pass: fuse parallel-group elementwise patterns into single
``plena.v_*`` / ``plena.zero_v`` GraphNodes.

Graph-IR replacement for the legacy stmt-walker
``frontend/passes/fuse_elementwise.py``. Equivalent fusion semantics,
but instead of rewriting the stmt tree we replace a NestedForGroup
(post-``annotate_grid``) with a single GraphNode.

Pre-condition
-------------
Run after :func:`annotate_grid.run` — fusion targets are NestedForGroups
that carry ``attrs[ATTR_GROUP_EXTENT] == extent`` (i.e. came from a
``T.Parallel`` for-loop).

Patterns
--------
Binary elementwise::

    NestedForGroup(loop_var=i, extent=N, attrs={ATTR_GROUP_EXTENT: N},
                   items=[RawStmt(BufferStore(dst, lhs[..,i] OP rhs[..,i]))])
        → GraphNode("plena.v_<op>", call_extern("plena.v_<op>",
                                                lhs.data, rhs.data, dst.data))

Constant fill (only ``= 0`` lowers — HW lacks a generic fill)::

    NestedForGroup(loop_var=i, extent=N, attrs={ATTR_GROUP_EXTENT: N},
                   items=[RawStmt(BufferStore(dst, IntImm/FloatImm(0)))])
        → GraphNode("plena.zero_v", call_extern("plena.zero_v", dst.data))

Nested fold (outer serial-for wrapping a single fuse target whose HW op
is whole-buffer — drop the outer for entirely)::

    NestedForGroup(loop_var=r, kind=SERIAL,
                   items=[<single fused GraphNode>])
        → <the fused GraphNode>

Non-matching NestedForGroups are left as-is — fusion is opportunistic.
"""

from __future__ import annotations

from typing import Optional

import tvm
from tvm import tir

from ..graph_ir import (
    Graph, GraphNode, LaneGroup, NestedForGroup, NodeRoot, ForRoot, RootItem,
    RawStmt, BufferAccess,
    ATTR_GROUP_EXTENT, ATTR_IS_SYNC,
)


# Map TIR binary-op node type → plena vector intrinsic name.
_OP_TO_INTRIN = {
    tir.Add: "plena.v_add",
    tir.Sub: "plena.v_sub",
    tir.Mul: "plena.v_mul",
}


# Already-fused whole-buffer ops; the nested-fold rule drops outer
# serial for-loops around these.
_WHOLE_BUFFER_FUSED_OPS = ("plena.zero_v", "plena.v_add", "plena.v_sub",
                           "plena.v_mul")


def _make_call(name: str, args: list) -> tir.Call:
    extern_op = tvm.ir.Op.get("tir.call_extern")
    return tir.Call("handle", extern_op, [tir.StringImm(name), *args])


def _is_lane_var_indexed(load: tir.BufferLoad, lane_var_name: str) -> bool:
    if not load.indices:
        return False
    last = load.indices[-1]
    return isinstance(last, tir.Var) and last.name == lane_var_name


def _full_access(buf: tir.Buffer) -> BufferAccess:
    return BufferAccess(
        buffer_name=buf.name,
        starts=[tir.IntImm("int32", 0) for _ in buf.shape],
        extents=list(buf.shape),
    )


def _try_fuse_for(forgrp: NestedForGroup) -> Optional[GraphNode]:
    """If ``forgrp`` is a single-store NestedForGroup matching the
    elementwise pattern, return the replacement GraphNode (else None)."""
    if forgrp.attrs.get(ATTR_GROUP_EXTENT) is None:
        return None
    extent = forgrp.attrs[ATTR_GROUP_EXTENT]
    if not isinstance(forgrp.extent, tir.IntImm):
        return None
    if int(forgrp.extent.value) != int(extent):
        return None
    if len(forgrp.items) != 1:
        return None
    item = forgrp.items[0]
    if not isinstance(item, RawStmt):
        return None
    store = item.stmt
    if not isinstance(store, tir.BufferStore):
        return None

    lane_var_name = forgrp.loop_var.name
    if not store.indices or not isinstance(store.indices[-1], tir.Var):
        return None
    if store.indices[-1].name != lane_var_name:
        return None

    expr = store.value

    # Constant fill — only ``= 0`` lowers (plena.zero_v).
    if isinstance(expr, (tir.IntImm, tir.FloatImm)):
        if float(expr.value) != 0.0:
            return None
        call = _make_call("plena.zero_v", [store.buffer.data])
        # plena.zero_v is in INHERENTLY_SYNC_EXTERNS — must be marked
        # sync so the materialize-time partitioner emits it OUTSIDE the
        # lane-for, not inside (which would re-zero the buffer once per
        # lane and corrupt downstream accumulation).
        return GraphNode(
            name=f"zero_v_{store.buffer.name}",
            op_call=call,
            attrs={ATTR_IS_SYNC: True},
            reads=[],
            writes=[_full_access(store.buffer)],
        )

    # Binary elementwise — Add / Sub / Mul.
    intrin_name = _OP_TO_INTRIN.get(type(expr))
    if intrin_name is None:
        return None
    if not isinstance(expr.a, tir.BufferLoad) or not isinstance(expr.b, tir.BufferLoad):
        return None
    if not _is_lane_var_indexed(expr.a, lane_var_name):
        return None
    if not _is_lane_var_indexed(expr.b, lane_var_name):
        return None

    call = _make_call(intrin_name, [
        expr.a.buffer.data,
        expr.b.buffer.data,
        store.buffer.data,
    ])
    short = intrin_name.replace("plena.", "")
    # plena.v_add / v_sub / v_mul are in INHERENTLY_SYNC_EXTERNS — see
    # zero_v above; same reasoning applies.
    return GraphNode(
        name=f"{short}_{store.buffer.name}",
        op_call=call,
        attrs={ATTR_IS_SYNC: True},
        reads=[_full_access(expr.a.buffer), _full_access(expr.b.buffer)],
        writes=[_full_access(store.buffer)],
    )


def _is_whole_buffer_fused(node: GraphNode) -> bool:
    """``node`` is a fused whole-buffer op produced by _try_fuse_for."""
    call = node.op_call
    if call.op.name != "tir.call_extern":
        return False
    if not call.args or not isinstance(call.args[0], tir.StringImm):
        return False
    return call.args[0].value in _WHOLE_BUFFER_FUSED_OPS


def _try_fold_nested(forgrp: NestedForGroup) -> Optional[GraphNode]:
    """Outer serial for wrapping a single fused whole-buffer op → drop
    the outer for. Mirrors stmt-walker `_try_fuse_nested`."""
    if forgrp.kind != tir.ForKind.SERIAL:
        return None
    if forgrp.attrs.get(ATTR_GROUP_EXTENT) is not None:
        # This for is itself a parallel-group; don't fold here, the
        # inner fuse handles it.
        return None
    if len(forgrp.items) != 1:
        return None
    inner = forgrp.items[0]
    if not isinstance(inner, GraphNode):
        return None
    if not _is_whole_buffer_fused(inner):
        return None
    return inner


def _fuse_items(items):
    """Walk a list of items; return a new list with fusion applied where
    possible. Recurses into nested for-groups."""
    out = []
    for item in items:
        if isinstance(item, NestedForGroup):
            # Recurse first so inner fuses can fire.
            item = NestedForGroup(
                loop_var=item.loop_var, min=item.min, extent=item.extent,
                kind=item.kind, thread_binding=item.thread_binding,
                annotations=item.annotations,
                items=_fuse_items(item.items),
                attrs=dict(item.attrs),
            )
            # First try outer-fold, then single-loop fuse.
            folded = _try_fold_nested(item)
            if folded is not None:
                out.append(folded)
                continue
            fused = _try_fuse_for(item)
            if fused is not None:
                out.append(fused)
                continue
            out.append(item)
        else:
            out.append(item)
    return out


def _fuse_root(root: RootItem) -> RootItem:
    if isinstance(root, ForRoot):
        return ForRoot(
            loop_var=root.loop_var, min=root.min, extent=root.extent,
            kind=root.kind, thread_binding=root.thread_binding,
            annotations=root.annotations, body=_fuse_root(root.body),
            attrs=dict(root.attrs),
        )
    if isinstance(root, LaneGroup):
        return LaneGroup(
            lane_var=root.lane_var, lane_count=root.lane_count,
            items=_fuse_items(root.items),
            alloc_buffers=list(root.alloc_buffers),
        )
    if isinstance(root, NodeRoot):
        return NodeRoot(
            items=_fuse_items(root.items),
            alloc_buffers=list(root.alloc_buffers),
        )
    return root


def run(graph: Graph) -> Graph:
    """Fuse elementwise patterns. Returns a NEW Graph (the root tree is
    rebuilt; ``buffer_nodes`` / ``buffer_map`` etc are shared)."""
    new_root = _fuse_root(graph.root)
    return Graph(
        root=new_root,
        params=graph.params,
        buffer_map=graph.buffer_map,
        ret_type=graph.ret_type,
        attrs=graph.attrs,
        buffer_nodes=graph.buffer_nodes,
    )


__all__ = ["run"]
