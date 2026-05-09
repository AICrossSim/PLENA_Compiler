"""Graph pass: upgrade lane-fusion-eligible ForRoots into LaneGroups.

The legacy ``lift_to_graph`` matched the canonical
    For(loop_var, extent=lane_count) → AttrStmt(plena.group, lane_count) →
    BlockRealize("tilelang_root", body=...)
shape and produced a :class:`LaneGroup` directly. ``lift_from_raw``
produces ForRoots instead (it doesn't have the post-stmt-walker
plena.group annotation to key off of). After ``annotate_grid`` +
``split_lane_groups`` have run, the lane-fusion-eligible for-nodes are:

  * a :class:`ForRoot` with ``attrs[ATTR_GROUP_EXTENT] == lane_count``
    (an unsplit grid axis whose extent already equals lane_count); OR
  * a :class:`ForRoot` with ``attrs[ATTR_IS_LANE_FOR]`` set (the inner-
    of-pair ForRoot produced by split_lane_groups).

This pass walks the graph; when it finds such a ForRoot wrapping a
``NodeRoot``, it replaces the pair with a :class:`LaneGroup` carrying
the same items. Downstream ``graph_pipeline._partition_and_materialize``
then knows to do the curtain-bundle algorithm (sync ops fold across
lanes; per-lane runs wrap in a for-by).
"""

from __future__ import annotations

from typing import List

from tvm import tir

from ..graph_ir import (
    Graph, GraphNode, LaneGroup, NestedForGroup, NodeRoot, ForRoot, RootItem,
    RawStmt,
    ATTR_GROUP_EXTENT, ATTR_IS_LANE_FOR,
)


def _is_lane_for(root: ForRoot, lane_count: int) -> bool:
    if root.attrs.get(ATTR_IS_LANE_FOR):
        return True
    if root.attrs.get(ATTR_GROUP_EXTENT) == lane_count:
        return True
    return False


def _upgrade(root: RootItem, lane_count: int) -> RootItem:
    if isinstance(root, ForRoot):
        # Recurse first.
        new_body = _upgrade(root.body, lane_count)
        # Upgrade if this ForRoot is lane-fusion-eligible AND its body is
        # a NodeRoot/LaneGroup carrying graph items.
        if _is_lane_for(root, lane_count):
            if isinstance(new_body, NodeRoot):
                return LaneGroup(
                    lane_var=root.loop_var,
                    lane_count=lane_count,
                    items=new_body.items,
                    alloc_buffers=list(new_body.alloc_buffers),
                )
            # If the body is already a LaneGroup, the inner-of-pair
            # split case: keep it as the LaneGroup and wrap the outer
            # ForRoot. (Outer carries ATTR_GROUP_EXTENT > lane_count;
            # we don't upgrade it.)
        return ForRoot(
            loop_var=root.loop_var, min=root.min, extent=root.extent,
            kind=root.kind, thread_binding=root.thread_binding,
            annotations=root.annotations, body=new_body,
            attrs=dict(root.attrs),
        )
    return root


def run(graph: Graph, lane_count: int = 4) -> Graph:
    """Walk the graph; replace lane-fusion-eligible ForRoot wrapping
    NodeRoot pairs with LaneGroup. Returns a NEW Graph; the underlying
    items are shared with the input."""
    new_root = _upgrade(graph.root, lane_count)
    return Graph(
        root=new_root,
        params=graph.params,
        buffer_map=graph.buffer_map,
        ret_type=graph.ret_type,
        attrs=graph.attrs,
        buffer_nodes=graph.buffer_nodes,
    )


__all__ = ["run"]
