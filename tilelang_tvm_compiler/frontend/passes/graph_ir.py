"""Graph IR — the data model the back-end and migrated frontend passes
all operate on.

Why a graph IR (vs. the old stmt-walker style)
----------------------------------------------
The frontend used to be a chain of stmt-walking passes that communicated
by stuffing AttrStmts onto the IR (``plena.sync`` / ``plena.group`` /
``plena.gemm_kind``) and re-reading them in the next walker. That style
makes per-op metadata "extrinsic" (parasitic on the stmt structure):
adding a new analysis means another walker, and the order in which a
walker peels AttrStmts is load-bearing.

In the graph IR each op is a :class:`GraphNode` with ``attrs`` — passes
read / write attrs directly on the node. ``reads`` / ``writes`` are
extracted at lift time (from the underlying ``BlockRealize`` or the
op's region arguments) and live on the node, so any pass can do
data-flow analysis without re-walking stmt trees.

Core types
----------
* :class:`GraphNode`  — a single op (a ``tl.tileop.*`` or a lowered
  ``tir.call_extern("plena.*", ...)`` call). Carries op_call, attrs,
  reads, writes.
* :class:`NestedForGroup` — a temporal for-loop sitting inside a lane
  group (e.g. ``for kv_block``). Body is again a list of items; the
  same sync-vs-per-lane partitioning applies recursively.
* :class:`LaneGroup` — the top-level lane fusion unit (one
  ``for lane_var in range(lane_count) × plena.group(lane_count) ×
  tilelang_root`` nest). Holds alloc'd buffers and the ordered item
  list.
* :class:`Graph` — the top-level Graph object, holds the PrimFunc
  signature data needed for materialization (params, buffer_map, attrs)
  plus a list of LaneGroup / outer-for / GraphNode items at the
  function root.

Passes operate on ``Graph`` end-to-end. ``compile_func`` calls
``lift_to_graph`` once at the top and ``materialize_to_primfunc`` once
at the end; everything in between is a chain of ``GraphPass`` objects
that take ``Graph`` and return ``Graph``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from tvm import tir


# ---------------------------------------------------------------------------
# Per-op attribute keys (graph-level metadata — replaces stmt AttrStmts)
# ---------------------------------------------------------------------------

# Set by annotate_sync_pass (or the lift-time fallback for already-fused
# plena.* externs). True iff this op is a multi-lane HW instruction
# that must fold OUTSIDE the per-lane for-by.
ATTR_IS_SYNC = "is_sync"

# Set by annotate_gemm_kind (eventually graph-level). One of "btmm",
# "overwrite", "add" (reserved). Determines the lower path.
ATTR_GEMM_KIND = "gemm_kind"


# ---------------------------------------------------------------------------
# For-node attribute keys
# ---------------------------------------------------------------------------

# Set by annotate_group_pass on a ForNode. The original lane-fusion-
# eligible extent (== axis logical width) — even after split_lane_groups
# rewrites the for to outer × inner, the inner-extent for-node carries
# this. Replaces the stmt-walker `T.attr(0, "plena.group", N)` AttrStmt.
ATTR_GROUP_EXTENT = "group_extent"

# Set by split_lane_groups_pass on the inner for-node of a head split
# (head_count > lane_count → outer × inner). True iff this is the
# lane-fusion for (its loop_var is the lane var).
ATTR_IS_LANE_FOR = "is_lane_for"


# ---------------------------------------------------------------------------
# Buffer-node attribute keys
# ---------------------------------------------------------------------------

# Set by allocate_group_memory_pass. One of "col_pack", "row_stack",
# "fp_lane", or absent (== unexpanded). Drives the buffer's lane-axis
# layout — eventually allocate_group_memory's stmt-rewriting work
# (changing buffer.shape and rewriting indices) becomes "set this attr,
# materialize uses it to compute the physical shape and rewrite refs".
ATTR_LANE_LAYOUT = "lane_layout"

LAYOUT_COL_PACK = "col_pack"     # (rows, last) → (1, rows, lane_count, last)
LAYOUT_ROW_STACK = "row_stack"   # (rows, last) → (1, lane_count, rows, last)
LAYOUT_FP_LANE = "fp_lane"       # (N,) → (lane_count, N)


# ---------------------------------------------------------------------------
# (R1 forward-looking) Buffer + For node types
# ---------------------------------------------------------------------------
#
# These are used by R2-R5 graph-layer passes to make buffer scope /
# layout / for-loop split into first-class graph operations (rather
# than stmt-level rewrites). Not consumed yet — current pipeline still
# operates on tir.Buffer / tir.For directly via NestedForGroup, LaneGroup,
# GraphNode.reads/writes.
#
# Migration plan:
#   R2: annotate_sync / annotate_gemm_kind populate node.attrs only —
#       no new types yet.
#   R3: fuse_elementwise / lower_fp_row_patterns produce GraphNodes
#       from RawStmt patterns. No new types.
#   R4: annotate_group / split_lane_groups operate on ForNode-typed
#       graph items (replacing NestedForGroup's anonymous tir.Var with
#       a richer ForNode that carries ATTR_GROUP_EXTENT / ATTR_IS_LANE_FOR).
#   R5: allocate_group_memory / scope_inference operate on BufferNode
#       (replacing the implicit tir.Buffer references in
#       GraphNode.reads/writes with explicit BufferNode references —
#       allows attr-driven shape / scope rewriting without mutating
#       the underlying tir.Buffer).


@dataclass
class BufferNode:
    """A buffer represented as a graph-layer node, NOT just a tir.Buffer
    reference.

    The graph-layer view of a buffer carries:
      * ``name`` — stable identifier used by passes / debug dumps.
      * ``shape`` — the **logical** shape used by the graph (mutable).
        ``allocate_group_memory_pass`` extends this by lane_count when
        flagging a buffer as col_pack / row_stack; ``materialize`` reads
        this to build the final tir.Buffer.
      * ``dtype`` — element type.
      * ``declared_scope`` — what the user wrote (``shared.dyn`` /
        ``local.fragment`` / ``global.vram`` / etc — pre-inference).
      * ``physical_scope`` — resolved scope (one of ``vram`` /
        ``mram`` / ``fpram`` / ``hbm`` / ``global.<phys>``). Filled by
        ``scope_inference_pass``. None until then.
      * ``data_var`` — the underlying tir.Var data handle. Preserved
        across the graph so users / op_call args still resolve.
      * ``attrs`` — free-form metadata (e.g. ATTR_LANE_LAYOUT).

    materialize_to_primfunc rebuilds a fresh ``tir.Buffer`` from these
    fields. Passes that change shape / scope just mutate this dataclass;
    no need to reconstruct downstream.
    """
    name: str
    shape: List["tir.PrimExpr"]
    dtype: str
    declared_scope: str
    physical_scope: Optional[str] = None
    data_var: Optional["tir.Var"] = None
    attrs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BufferAccess:
    """A read or write of a contiguous region of a graph-layer buffer.

    Replaces ``tir.BufferRegion`` on ``GraphNode.reads/writes``: stores a
    buffer **name** (resolved via ``Graph.buffer_nodes[name]``) plus the
    per-axis ``starts`` / ``extents`` PrimExprs. Decoupling reads/writes
    from a baked-in ``tir.Buffer`` reference lets buffer-shape rewrites
    (e.g. lane-axis expansion in materialize) propagate without having
    to mutate every BufferRegion in the graph.

    ``starts`` and ``extents`` MUST match the rank of the BufferNode's
    *current* shape (graph passes may rewrite expressions, but they must
    keep this invariant).
    """
    buffer_name: str
    starts: List["tir.PrimExpr"] = field(default_factory=list)
    extents: List["tir.PrimExpr"] = field(default_factory=list)


@dataclass
class ForNode:
    """A for-loop represented as a graph-layer node.

    Carries:
      * ``loop_var``, ``min``, ``extent``, ``kind`` — same as tir.For.
      * ``thread_binding`` — preserved from tir.For (most fors don't
        have one).
      * ``body_items`` — recursive item list (graph nodes / nested fors
        / raw stmts) that the for wraps.
      * ``attrs`` — graph metadata (ATTR_GROUP_EXTENT / ATTR_IS_LANE_FOR).

    R4 (graph-layer split_lane_groups + annotate_group) operates on
    these. Today the NestedForGroup type plays a similar role and the
    two will converge once R4 lands; for now ForNode is forward-looking
    infrastructure that materialize doesn't read.
    """
    loop_var: "tir.Var"
    min: "tir.PrimExpr"
    extent: "tir.PrimExpr"
    kind: "tir.ForKind"
    thread_binding: Optional["tir.IterVar"] = None
    body_items: List[Any] = field(default_factory=list)
    attrs: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# IR types
# ---------------------------------------------------------------------------

@dataclass
class RawStmt:
    """A raw stmt that doesn't fit the GraphNode shape (e.g. a
    BufferStore that wasn't fused into a plena.* extern, a LetStmt).
    It passes through the graph unchanged — graph passes treat it as
    opaque per-lane work and materialization emits the underlying
    ``stmt`` verbatim. This is an escape hatch for shapes the lift
    can't classify yet."""
    name: str
    stmt: "tir.Stmt"


@dataclass
class GraphNode:
    """A single op in the graph.

    Attributes
    ----------
    name : str
        Stable identifier ("op_0", "btmm_0", ...) used for debugging
        and graph-pass diffing.
    op_call : tir.Call
        The underlying ``tl.tileop.*`` (pre-lower) or
        ``tir.call_extern("plena.*", ...)`` (already-lowered) call.
        Materialization emits this directly (or lowers it via the
        helpers in ``lower_to_hlir.py``).
    attrs : dict
        Mutable, free-form metadata. Passes read and write keys here
        (e.g. ``ATTR_IS_SYNC``, ``ATTR_GEMM_KIND``).
    reads, writes : list of BufferAccess
        Data-flow info — what buffers this op reads / writes, with
        per-axis ranges. Filled at lift time. Each entry references a
        ``Graph.buffer_nodes[buffer_name]`` BufferNode (so layout
        rewrites in materialize don't require mutating reads/writes).
        Used by dependency analysis (sync classification, reorder
        safety, etc).
    """
    name: str
    op_call: tir.Call
    attrs: Dict[str, Any] = field(default_factory=dict)
    reads: List["BufferAccess"] = field(default_factory=list)
    writes: List["BufferAccess"] = field(default_factory=list)


@dataclass
class NestedForGroup:
    """A temporal for-loop sitting inside a lane group (e.g.
    ``for kv_block in range(num_kv_blocks)``). Its ``loop_var`` is NOT
    the lane var — it's a serial outer iteration whose body itself
    contains a mix of GraphNode and (further) NestedForGroup items.
    The same sync-vs-per-lane partitioning applies recursively to
    these inner items.

    ``attrs`` is graph-layer metadata (e.g. ATTR_GROUP_EXTENT set by
    annotate_grid_pass on T.Parallel-derived for-loops, ATTR_IS_LANE_FOR
    set by split_lane_groups_pass on the inner-of-split fors)."""
    loop_var: tir.Var
    min: "tir.PrimExpr"
    extent: "tir.PrimExpr"
    kind: tir.ForKind
    thread_binding: Optional[tir.IterVar]
    annotations: Optional[Dict[str, Any]]
    items: List[Union["GraphNode", "NestedForGroup", "RawStmt"]]
    attrs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LaneGroup:
    """A lane-fusion unit. Corresponds to one
    ``for lane_var in range(lane_count) × plena.group(lane_count) ×
    tilelang_root`` nest in the lifted IR."""
    lane_var: tir.Var
    lane_count: int
    items: List[Union[GraphNode, NestedForGroup, RawStmt]]
    alloc_buffers: List[tir.Buffer] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Top-level Graph
# ---------------------------------------------------------------------------

# Item types that can sit at the function root, OUTSIDE any LaneGroup.
# These are typically:
#   * outer kernel-grid for-loops not picked up as a lane-group entry
#     (e.g. q_block / by_o)
#   * AttrStmts that wrap nothing graph-relevant (rare)
#   * raw stmts the lift pass left as-is
#
# A LaneGroup is the only "graph-rich" thing — when a ForRoot wraps a
# LaneGroup we materialize the LaneGroup recursively and then wrap in
# the For. A NodeRoot is for kernels with no lane fusion at all (mm64).
@dataclass
class ForRoot:
    """An outer for-loop wrapping a LaneGroup or another ForRoot.

    ``attrs`` is graph-layer metadata (e.g. ATTR_GROUP_EXTENT — set by
    annotate_grid_pass when the ForRoot was peeled from a blockIdx
    binding with extent > 1; signals "this axis is lane-fusion-eligible
    if extent matches lane_count")."""
    loop_var: tir.Var
    min: "tir.PrimExpr"
    extent: "tir.PrimExpr"
    kind: tir.ForKind
    thread_binding: Optional[tir.IterVar]
    annotations: Optional[Dict[str, Any]]
    body: "RootItem"
    attrs: Dict[str, Any] = field(default_factory=dict)


# A function root is one of: a LaneGroup (tilelang_root has lane fusion),
# a NodeRoot (no lane fusion, ops sit directly under tilelang_root), or
# a ForRoot wrapping one of these (outer kernel-grid for-loops).
@dataclass
class NodeRoot:
    """A no-lane-fusion root: ops directly under tilelang_root.
    Used by kernels like mm64 with `T.Kernel(1)` that collapsed."""
    items: List[Union[GraphNode, NestedForGroup, RawStmt]]
    alloc_buffers: List[tir.Buffer] = field(default_factory=list)


RootItem = Union[LaneGroup, NodeRoot, ForRoot]


@dataclass
class Graph:
    """The whole-kernel graph.

    The root is a single :class:`RootItem`. The PrimFunc shell info
    (params, buffer_map, ret_type, attrs) is stashed alongside so
    materialize can rebuild the PrimFunc later.

    ``buffer_nodes`` is the graph-layer buffer table: every alloc'd
    buffer AND every param buffer has an entry, indexed by name. Graph
    passes mutate ``BufferNode.shape`` / ``physical_scope`` /
    ``attrs[ATTR_LANE_LAYOUT]`` here; ``GraphNode.reads/writes`` carry
    only the ``buffer_name`` (resolved via this dict), so rewrites
    propagate to all uses without per-region mutation.
    """
    root: RootItem

    # PrimFunc shell — preserved verbatim through graph passes; used
    # by materialize.
    params: List[tir.Var]
    buffer_map: Dict[tir.Var, tir.Buffer]
    ret_type: Any
    attrs: Any

    # Graph-layer buffer table. Empty {} for graphs produced before the
    # buffer-node migration (legacy lift_to_graph used to leave this
    # unfilled); current lifts (lift_from_raw_primfunc, lift_to_graph)
    # populate it.
    buffer_nodes: Dict[str, "BufferNode"] = field(default_factory=dict)


__all__ = [
    # Item types (current graph IR — used by graph_pipeline)
    "GraphNode", "NestedForGroup", "LaneGroup", "RawStmt",
    "ForRoot", "NodeRoot", "RootItem", "Graph",
    # Per-op attr keys
    "ATTR_IS_SYNC", "ATTR_GEMM_KIND",
    # For-node attr keys (R4-forward)
    "ATTR_GROUP_EXTENT", "ATTR_IS_LANE_FOR",
    # Buffer-node attr keys (R5-forward)
    "ATTR_LANE_LAYOUT",
    "LAYOUT_COL_PACK", "LAYOUT_ROW_STACK", "LAYOUT_FP_LANE",
    # Forward-looking node types (R4 / R5)
    "BufferNode", "BufferAccess", "ForNode",
]
