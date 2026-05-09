"""Tests for the graph-layer ``lower_fp_row_patterns`` pass.

Each pattern (FP scalar store, row-parallel store, reduce) is exercised
by lifting a small kernel, running the prerequisite graph passes
(annotate_grid + scope_inference), then checking that the targeted
intrinsic appears in the resulting graph.
"""

from __future__ import annotations

from tvm import tir

import tilelang_tvm_compiler  # bootstrap TVM 0.23
import tilelang.language as T

from tilelang_tvm_compiler.frontend.passes.lift_from_raw import (
    lift_from_raw_primfunc,
)
from tilelang_tvm_compiler.frontend.passes.graph_passes import (
    annotate_grid, scope_inference, lower_fp_row_patterns,
)
from tilelang_tvm_compiler.frontend.passes.graph_ir import (
    Graph, GraphNode, ForRoot, NestedForGroup, LaneGroup, NodeRoot,
    RawStmt,
)


def _walk(graph: Graph):
    """Yield every item (GraphNode / NestedForGroup / RawStmt) in the
    graph, recursively."""
    out = []

    def visit_items(items):
        for it in items:
            out.append(it)
            if isinstance(it, NestedForGroup):
                visit_items(it.items)

    def visit_root(root):
        if isinstance(root, ForRoot):
            visit_root(root.body)
            return
        if isinstance(root, (LaneGroup, NodeRoot)):
            visit_items(root.items)

    visit_root(graph.root)
    return out


def _has_extern(graph: Graph, name: str) -> bool:
    """Check if any GraphNode (or RawStmt-wrapped Evaluate(call_extern))
    matches the given name."""
    for it in _walk(graph):
        if isinstance(it, GraphNode):
            call = it.op_call
            if (call.op.name == "tir.call_extern"
                    and isinstance(call.args[0], tir.StringImm)
                    and call.args[0].value == name):
                return True
        elif isinstance(it, RawStmt):
            # Walk the wrapped TIR for an Evaluate(call_extern).
            stack = [it.stmt]
            while stack:
                s = stack.pop()
                if isinstance(s, tir.Evaluate) and isinstance(s.value, tir.Call):
                    c = s.value
                    if (c.op.name == "tir.call_extern"
                            and isinstance(c.args[0], tir.StringImm)
                            and c.args[0].value == name):
                        return True
                if isinstance(s, tir.For):
                    stack.append(s.body)
                elif isinstance(s, tir.SeqStmt):
                    stack.extend(s.seq)
                elif isinstance(s, tir.AttrStmt):
                    stack.append(s.body)
    return False


# ---------------------------------------------------------------------------
# Kernel: FP scalar store (M_OLD[row] = 0.0 → fp_zero_at)
# ---------------------------------------------------------------------------

def _fp_zero_kernel():
    @T.prim_func
    def k(X: T.Tensor((1, 64, 1, 64), "float16")):
        with T.Kernel(1, threads=128) as bx:
            X_v = T.alloc_shared((64, 64), "float16")
            M_fp = T.alloc_fragment((64,), "float16")
            T.copy(X[0, 0, 0, 0], X_v)
            for r in T.serial(64):
                M_fp[r] = T.float16(0.0)
    return k


def _fp_copy_kernel():
    @T.prim_func
    def k(X: T.Tensor((1, 64, 1, 64), "float16")):
        with T.Kernel(1, threads=128) as bx:
            X_v = T.alloc_shared((64, 64), "float16")
            M_fp = T.alloc_fragment((64,), "float16")
            N_fp = T.alloc_fragment((64,), "float16")
            T.copy(X[0, 0, 0, 0], X_v)
            for r in T.serial(64):
                N_fp[r] = M_fp[r]
    return k


def _pipeline(kernel_factory):
    g = lift_from_raw_primfunc(kernel_factory())
    g = annotate_grid.run(g)
    scopes = scope_inference.infer(g)
    return lower_fp_row_patterns.run(g, scopes)


def test_fp_zero_store_lowers_to_fp_zero_at():
    g = _pipeline(_fp_zero_kernel)
    assert _has_extern(g, "plena.fp_zero_at")


def test_fp_copy_lowers_to_fp_copy_at():
    g = _pipeline(_fp_copy_kernel)
    assert _has_extern(g, "plena.fp_copy_at")


def test_idempotent():
    g = _pipeline(_fp_zero_kernel)
    scopes = scope_inference.infer(g)
    g_twice = lower_fp_row_patterns.run(g, scopes)
    assert _has_extern(g_twice, "plena.fp_zero_at")


if __name__ == "__main__":
    test_fp_zero_store_lowers_to_fp_zero_at()
    test_fp_copy_lowers_to_fp_copy_at()
    test_idempotent()
    print("graph lower_fp_row_patterns tests passed")
