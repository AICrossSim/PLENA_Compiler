"""Compiler passes for tilelang_plena_compiler.

Each pass module exposes a single `run(mod) -> mod` function. Passes are
intentionally independent so they can be unit-tested in isolation; the
main `pipeline.py` strings them together.
"""
