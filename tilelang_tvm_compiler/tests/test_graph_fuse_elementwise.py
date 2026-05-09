"""Tests for the graph-layer ``fuse_elementwise`` pass.

Equivalent semantics to the legacy stmt-walker
``fuse_elementwise``, but operating on a :class:`graph_ir.Graph`
post-``annotate_grid``. Fusion replaces a NestedForGroup with a single
``plena.v_*`` / ``plena.zero_v`` GraphNode.
"""

from __future__ import annotations

from tvm import tir

import tilelang_tvm_compiler  # bootstrap TVM 0.23
import tilelang.language as T

from tilelang_tvm_compiler.frontend.passes.lift_from_raw import (
    lift_from_raw_primfunc,
)
from tilelang_tvm_compiler.frontend.passes.graph_passes import (
    annotate_grid, fuse_elementwise,
)
from tilelang_tvm_compiler.frontend.passes.graph_ir import (
    Graph, GraphNode, ForRoot, NestedForGroup, LaneGroup, NodeRoot,
)


def _walk_graph_nodes(graph: Graph):
    out = []

    def visit_items(items):
        for it in items:
            if isinstance(it, GraphNode):
                out.append(it)
            elif isinstance(it, NestedForGroup):
                visit_items(it.items)

    def visit_root(root):
        if isinstance(root, ForRoot):
            visit_root(root.body)
            return
        if isinstance(root, (LaneGroup, NodeRoot)):
            visit_items(root.items)

    visit_root(graph.root)
    return out


def _has_extern_call(graph: Graph, name: str) -> bool:
    for n in _walk_graph_nodes(graph):
        call = n.op_call
        if (call.op.name == "tir.call_extern"
                and isinstance(call.args[0], tir.StringImm)
                and call.args[0].value == name):
            return True
    return False


def _count_parallel_for(graph: Graph) -> int:
    """Count NestedForGroups still carrying ATTR_GROUP_EXTENT (i.e.
    ones that didn't fuse)."""
    from tilelang_tvm_compiler.frontend.passes.graph_ir import ATTR_GROUP_EXTENT
    n = 0

    def visit_items(items):
        nonlocal n
        for it in items:
            if isinstance(it, NestedForGroup):
                if it.attrs.get(ATTR_GROUP_EXTENT) is not None:
                    n += 1
                visit_items(it.items)

    def visit_root(root):
        if isinstance(root, ForRoot):
            visit_root(root.body)
            return
        if isinstance(root, (LaneGroup, NodeRoot)):
            visit_items(root.items)

    visit_root(graph.root)
    return n


def _add_kernel():
    @T.prim_func
    def k(
        A: T.Tensor((1, 64, 1, 64), "float16"),
        B: T.Tensor((1, 64, 1, 64), "float16"),
        C: T.Tensor((1, 64, 1, 64), "float16"),
    ):
        with T.Kernel(1, threads=128) as bx:
            A_sh = T.alloc_shared((64,), "float16")
            B_sh = T.alloc_shared((64,), "float16")
            C_sh = T.alloc_shared((64,), "float16")
            T.copy(A[0, 0, 0, 0], A_sh)
            T.copy(B[0, 0, 0, 0], B_sh)
            for i in T.Parallel(64):
                C_sh[i] = A_sh[i] + B_sh[i]
            T.copy(C_sh, C[0, 0, 0, 0])
    return k


def _no_parallel_kernel():
    @T.prim_func
    def k(
        A: T.Tensor((1, 64, 1, 64), "float16"),
        B: T.Tensor((1, 64, 1, 64), "float16"),
        C: T.Tensor((1, 64, 1, 64), "float16"),
    ):
        with T.Kernel(1, threads=128) as bx:
            A_sh = T.alloc_shared((64,), "float16")
            B_sh = T.alloc_shared((64,), "float16")
            C_sh = T.alloc_shared((64,), "float16")
            T.copy(A[0, 0, 0, 0], A_sh)
            T.copy(B[0, 0, 0, 0], B_sh)
            for i in T.serial(64):
                C_sh[i] = A_sh[i] + B_sh[i]
            T.copy(C_sh, C[0, 0, 0, 0])
    return k


def _zero_kernel():
    @T.prim_func
    def k(C: T.Tensor((1, 64, 1, 64), "float16")):
        with T.Kernel(1, threads=128) as bx:
            C_sh = T.alloc_shared((64,), "float16")
            for i in T.Parallel(64):
                C_sh[i] = T.float16(0.0)
            T.copy(C_sh, C[0, 0, 0, 0])
    return k


def _pipeline(kernel_factory):
    g = lift_from_raw_primfunc(kernel_factory())
    g = annotate_grid.run(g)
    g = fuse_elementwise.run(g)
    return g


def test_parallel_add_fuses_to_v_add():
    g = _pipeline(_add_kernel)
    assert _has_extern_call(g, "plena.v_add")
    assert _count_parallel_for(g) == 0


def test_serial_loop_is_not_fused():
    g = _pipeline(_no_parallel_kernel)
    assert not _has_extern_call(g, "plena.v_add")
    # The serial for-loop should still be a NestedForGroup item (no
    # parallel-group attr; that's fine).
    nodes = _walk_graph_nodes(g)
    extern_names = [n.op_call.args[0].value for n in nodes
                    if n.op_call.op.name == "tir.call_extern"
                    and isinstance(n.op_call.args[0], tir.StringImm)]
    assert "plena.v_add" not in extern_names


def test_parallel_zero_fuses_to_zero_v():
    g = _pipeline(_zero_kernel)
    assert _has_extern_call(g, "plena.zero_v")
    assert _count_parallel_for(g) == 0


def test_idempotent():
    g = _pipeline(_add_kernel)
    g_twice = fuse_elementwise.run(g)
    assert _has_extern_call(g_twice, "plena.v_add")


if __name__ == "__main__":
    test_parallel_add_fuses_to_v_add()
    test_serial_loop_is_not_fused()
    test_parallel_zero_fuses_to_zero_v()
    test_idempotent()
    print("graph fuse_elementwise tests passed")
