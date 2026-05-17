"""tilelang -> PLENA-flavored TIR frontend (legacy).

The legacy graph-IR pipeline lives in ``frontend/passes/`` and used to
expose ``compile_func`` as the public entry. The active pipeline now
runs through ``frontend/mid_ir/`` instead, so this package's __init__
intentionally does NOT eagerly import ``compile_func`` — that would
cause a circular import when ``..pipeline`` (the new top-level
compile_kernel) imports ``frontend/passes/inline_let_stmts``,
``lower_compound_fp_stores``, and ``frontend/mid_ir/passes/...``.

Callers that still need ``compile_func`` for some legacy reason can
import it directly: ``from .pipeline import compile_func``.
"""
