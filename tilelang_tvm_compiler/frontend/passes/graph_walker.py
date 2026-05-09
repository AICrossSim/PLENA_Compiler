"""Graph traversal helpers — used by graph-layer passes (R2 onward).

These helpers let a pass walk the Graph item tree without re-implementing
the recursive descent into LaneGroup / NestedForGroup / ForRoot bodies.
Each helper returns a generator of (item, parent_items_list, index)
so callers can both inspect and (if they want) mutate items in place.

Why this lives here instead of on Graph itself:
  * Keeps graph_ir.py purely declarative (just dataclasses).
  * Multiple traversal strategies (visit nodes only / visit for-nodes
    only / pre-order / post-order) without ballooning the dataclass
    surface.
"""

from __future__ import annotations

from typing import Callable, Iterator, List, Tuple

from .graph_ir import (
    Graph, GraphNode, NestedForGroup, LaneGroup, NodeRoot, ForRoot,
    RawStmt, RootItem,
)


def walk_root(graph: Graph) -> Iterator[Tuple[object, str]]:
    """Yield each item in the graph, paired with a label describing
    where it sits ("root" / "lane_group" / "nested_for" / "for_root")."""
    yield from _walk_root_item(graph.root, "root")


def _walk_root_item(item: RootItem, label: str) -> Iterator[Tuple[object, str]]:
    yield item, label
    if isinstance(item, ForRoot):
        yield from _walk_root_item(item.body, "for_root.body")
    elif isinstance(item, LaneGroup):
        for child in item.items:
            yield from _walk_item(child, "lane_group")
    elif isinstance(item, NodeRoot):
        for child in item.items:
            yield from _walk_item(child, "node_root")


def _walk_item(item, parent_label: str) -> Iterator[Tuple[object, str]]:
    yield item, parent_label
    if isinstance(item, NestedForGroup):
        for child in item.items:
            yield from _walk_item(child, "nested_for")


def walk_graph_nodes(graph: Graph) -> Iterator[GraphNode]:
    """Yield every GraphNode in the graph (recursively, in source order)."""
    for item, _ in walk_root(graph):
        if isinstance(item, GraphNode):
            yield item


def walk_nested_fors(graph: Graph) -> Iterator[NestedForGroup]:
    """Yield every NestedForGroup in the graph."""
    for item, _ in walk_root(graph):
        if isinstance(item, NestedForGroup):
            yield item


def find_nodes_where(graph: Graph,
                    predicate: Callable[[GraphNode], bool]) -> List[GraphNode]:
    """Return all GraphNodes for which ``predicate`` is true."""
    return [n for n in walk_graph_nodes(graph) if predicate(n)]


def transform_items_in_place(items: list,
                             transform: Callable[[object], object]) -> None:
    """Apply ``transform`` to each item in a flat item list in place.

    ``transform`` returns either the same item (no change) or a
    replacement. To remove an item, return None and the helper drops it.

    Used by pattern-matching passes (fuse_elementwise / lower_fp_row_patterns)
    to swap RawStmt patterns for GraphNode replacements without copying
    the surrounding structure.
    """
    out = []
    for it in items:
        new = transform(it)
        if new is None:
            continue
        out.append(new)
    items[:] = out


def transform_all_item_lists(graph: Graph,
                              transform: Callable[[object], object]) -> None:
    """Apply ``transform`` to every leaf item list (LaneGroup.items,
    NodeRoot.items, NestedForGroup.items) in the graph, in place.

    ``transform`` is called once per item. Returning None drops the item;
    returning a different object replaces it; returning the same object
    leaves it.
    """
    def visit_root(item: RootItem):
        if isinstance(item, ForRoot):
            visit_root(item.body)
            return
        if isinstance(item, LaneGroup):
            transform_items_in_place(item.items, transform)
            for child in item.items:
                if isinstance(child, NestedForGroup):
                    visit_nested(child)
            return
        if isinstance(item, NodeRoot):
            transform_items_in_place(item.items, transform)
            for child in item.items:
                if isinstance(child, NestedForGroup):
                    visit_nested(child)
            return

    def visit_nested(nfg: NestedForGroup):
        transform_items_in_place(nfg.items, transform)
        for child in nfg.items:
            if isinstance(child, NestedForGroup):
                visit_nested(child)

    visit_root(graph.root)


__all__ = [
    "walk_root", "walk_graph_nodes", "walk_nested_fors",
    "find_nodes_where",
    "transform_items_in_place", "transform_all_item_lists",
]
