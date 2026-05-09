"""Graph pass: annotate every lane-fusion-eligible for-loop with
``ATTR_GROUP_EXTENT``.

Graph-IR replacement for the legacy stmt-walker
``frontend/passes/annotate_group.py``. Equivalent semantics, but instead
of rewriting the stmt tree (wrapping the for in
``T.attr(0, "plena.group", N)``) it just sets a graph attr that
downstream passes consume.

What gets annotated
-------------------
* Every :class:`ForRoot` — these came from ``blockIdx.* > 1`` grid
  bindings in ``lift_from_raw`` (threadIdx and blockIdx==1 are dropped
  upstream). The grid axis extent goes into
  ``forroot.attrs[ATTR_GROUP_EXTENT]``.
* Every :class:`NestedForGroup` whose ``kind == PARALLEL`` — these came
  from ``T.Parallel`` for-loops. The pass also rewrites the kind to
  SERIAL (PLENA HW is single-threaded; the group annotation is what
  signals "iterations are fusion-eligible" to downstream passes).

The legacy stmt-walker also did a "drop blockIdx==1" / "subst threadIdx
to 0" rewrite on the IR. ``lift_from_raw._lift_root`` already does the
equivalent (it skips the AttrStmt and recurses into the body without
creating a ForRoot), so this pass doesn't need to repeat it.
"""

from __future__ import annotations

from tvm import tir

from ..graph_ir import (
    Graph, GraphNode, LaneGroup, NestedForGroup, NodeRoot, ForRoot, RootItem,
    RawStmt, ATTR_GROUP_EXTENT,
)


class AnnotateGridError(RuntimeError):
    pass


def _extent_int(extent: "tir.PrimExpr") -> int:
    if not isinstance(extent, tir.IntImm):
        raise AnnotateGridError(
            f"grid / parallel for has non-constant extent {extent!r}; "
            f"groups require compile-time extent"
        )
    return int(extent.value)


def _annotate_items(items) -> None:
    for item in items:
        if isinstance(item, NestedForGroup):
            if item.kind == tir.ForKind.PARALLEL:
                item.attrs[ATTR_GROUP_EXTENT] = _extent_int(item.extent)
                item.kind = tir.ForKind.SERIAL
            _annotate_items(item.items)
        # GraphNode / RawStmt: nothing to do.


def _annotate_root(root: RootItem) -> None:
    if isinstance(root, ForRoot):
        # ForRoots in the lift-from-raw graph correspond to blockIdx > 1
        # grid bindings, all of which are lane-fusion-eligible.
        root.attrs[ATTR_GROUP_EXTENT] = _extent_int(root.extent)
        _annotate_root(root.body)
        return
    if isinstance(root, LaneGroup):
        _annotate_items(root.items)
        return
    if isinstance(root, NodeRoot):
        _annotate_items(root.items)
        return


def run(graph: Graph) -> Graph:
    """Set ``attrs[ATTR_GROUP_EXTENT]`` on every grid / T.Parallel for in
    the graph. In-place mutation; also returns the graph for chaining."""
    _annotate_root(graph.root)
    return graph


__all__ = ["run", "AnnotateGridError"]
