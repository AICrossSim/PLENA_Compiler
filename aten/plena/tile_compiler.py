"""Compatibility alias for the former TileCompiler memory-state class."""

from __future__ import annotations

from compiler.aten.plena.memory_state import MemoryStateMixin

TileCompiler = MemoryStateMixin

__all__ = ["MemoryStateMixin", "TileCompiler"]
