"""Graph pass: classify every GraphNode as sync or per-lane, store in
``node.attrs[ATTR_IS_SYNC]``.

This is the graph-IR replacement for the legacy stmt-walker
``frontend/passes/annotate_sync.py``. Equivalent classification rules,
but operating on graph nodes (with reads/writes already populated)
rather than stmt patterns.

Sync rules
----------
A GraphNode is marked sync iff one of:
  * it's a ``tl.tileop.copy`` between HBM and a local buffer (DMA);
  * it's a ``tl.tileop.copy`` between vram and a rank-1 fpram fragment
    (row_v_to_fp / row_fp_to_v — HW S_MAP_*_* covers MLEN = lane_count
    × hlen elements in one instruction);
  * it's a ``tl.tileop.copy`` between two local non-fpram buffers
    (vram↔vram "tensor cache" — one V_ADD_VF row covers MLEN);
  * it's a ``tl.tileop.gemm_py`` with ``ATTR_GEMM_KIND == "btmm"``;
  * it's an already-lowered plena.* extern in
    ``INHERENTLY_SYNC_EXTERNS``.

Buffer scope source
-------------------
The pass takes a ``hbm_names`` set (PrimFunc parameter names — these
buffers live in HBM) and reads the underlying ``tir.Buffer.scope()``
for everything else. We don't need the full ``BufferScopeMap`` (that's
the resolved physical scope after scope_inference); we only need the
*declared* tilelang scope (``shared.dyn`` / ``local.fragment`` /
HBM-via-param), which is what the original annotate_sync also looked
at.
"""

from __future__ import annotations

from typing import Set

from tvm import tir

from ..graph_ir import (
    Graph, GraphNode, LaneGroup, NestedForGroup, NodeRoot, ForRoot, RootItem,
    ATTR_IS_SYNC, ATTR_GEMM_KIND,
)
from ..graph_pipeline import INHERENTLY_SYNC_EXTERNS


_TILEOP_COPY = "tl.tileop.copy"
_TILEOP_GEMM = "tl.tileop.gemm_py"
_TILEOP_REGION = "tl.tileop.region"


def _region_buffer(call: "tir.Call"):
    """Pull the underlying tir.Buffer out of a ``tl.tileop.region(...)``
    call's args[0] (a BufferLoad)."""
    if not isinstance(call, tir.Call) or call.op.name != _TILEOP_REGION:
        return None
    load = call.args[0]
    if not isinstance(load, tir.BufferLoad):
        return None
    return load.buffer


def _copy_endpoints(call: tir.Call):
    """For a ``tl.tileop.copy(src_region, dst_region)`` call, return
    (src_buf, dst_buf). Either may be None if the region arg isn't
    parsable (defensive — shouldn't happen for well-formed input)."""
    if call.op.name != _TILEOP_COPY:
        return (None, None)
    return (_region_buffer(call.args[0]), _region_buffer(call.args[1]))


def _is_hbm(buf, hbm_names: Set[str]) -> bool:
    return buf is not None and buf.name in hbm_names


def _is_fpram_fragment(buf) -> bool:
    """A rank-1 ``local.fragment`` buffer maps to FPRAM."""
    if buf is None:
        return False
    declared = buf.scope() if callable(getattr(buf, "scope", None)) else "global"
    if declared != "local.fragment":
        return False
    if len(buf.shape) != 1:
        return False
    return True


def _classify_copy_sync(node: GraphNode, hbm_names: Set[str]) -> bool:
    """Apply the four ``T.copy``-related sync rules. Returns True if
    this node is sync."""
    src, dst = _copy_endpoints(node.op_call)
    src_hbm = _is_hbm(src, hbm_names)
    dst_hbm = _is_hbm(dst, hbm_names)
    if src_hbm ^ dst_hbm:
        return True  # DMA
    src_fp = _is_fpram_fragment(src)
    dst_fp = _is_fpram_fragment(dst)
    if src_fp ^ dst_fp:
        return True  # row_v_to_fp / fp_to_v
    if (src is not None and dst is not None
            and not src_hbm and not dst_hbm
            and not src_fp and not dst_fp):
        return True  # vram↔vram copy_v_to_v
    return False


def _is_inherently_sync_extern(call: tir.Call) -> bool:
    if call.op.name != "tir.call_extern":
        return False
    name_arg = call.args[0]
    if not isinstance(name_arg, tir.StringImm):
        return False
    return name_arg.value in INHERENTLY_SYNC_EXTERNS


def _classify_node(node: GraphNode, hbm_names: Set[str]) -> bool:
    """Return True iff this graph node is a sync site."""
    op_name = node.op_call.op.name
    if op_name == _TILEOP_COPY:
        return _classify_copy_sync(node, hbm_names)
    if op_name == _TILEOP_GEMM:
        return node.attrs.get(ATTR_GEMM_KIND) == "btmm"
    if op_name == "tir.call_extern":
        return _is_inherently_sync_extern(node.op_call)
    return False


# ---------------------------------------------------------------------------
# Walker over Graph (does NOT recurse into the tir IR — only into our
# graph-layer dataclasses).
# ---------------------------------------------------------------------------

def _annotate_items(items, hbm_names: Set[str]) -> None:
    for item in items:
        if isinstance(item, GraphNode):
            item.attrs[ATTR_IS_SYNC] = _classify_node(item, hbm_names)
        elif isinstance(item, NestedForGroup):
            _annotate_items(item.items, hbm_names)
        # RawStmt: never sync — it's per-lane opaque work, no attrs to set.


def _annotate_root(root: RootItem, hbm_names: Set[str]) -> None:
    if isinstance(root, LaneGroup):
        _annotate_items(root.items, hbm_names)
    elif isinstance(root, NodeRoot):
        _annotate_items(root.items, hbm_names)
    elif isinstance(root, ForRoot):
        _annotate_root(root.body, hbm_names)


def run(graph: Graph) -> Graph:
    """Annotate every GraphNode in the graph with
    ``attrs[ATTR_IS_SYNC] = bool``. In-place mutation; also returns the
    graph so callers can chain."""
    hbm_names = {buf.name for buf in graph.buffer_map.values()}
    _annotate_root(graph.root, hbm_names)
    return graph


__all__ = ["run"]
