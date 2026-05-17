"""Legacy graph-IR frontend pipeline — removed.

The mid_ir pipeline in ``frontend/mid_ir/passes/`` is the only active
lowering chain now. The graph-IR layer (``frontend/passes/graph_*``,
``classify_lane_use``, ``expand_lane_grid``, ``infer_lane_layout``,
``fuse_elementwise``, ``lower_to_hlir``, ``lift_from_raw``,
``forbid_plena_extern``) has been deleted in full.

This stub stays so that ``import``-error sites surface a clear message
instead of ``ModuleNotFoundError`` for callers that haven't been
migrated yet (e.g. ``kernels/conv2d_min.py``). Migrate them to the
mid_ir pipeline (``tilelang_tvm_compiler.pipeline.compile_kernel``).
"""


def compile_func(*_args, **_kwargs):
    raise RuntimeError(
        "frontend.pipeline.compile_func has been removed. Use "
        "tilelang_tvm_compiler.pipeline.compile_kernel instead."
    )
