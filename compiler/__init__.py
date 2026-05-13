"""Compatibility namespace for legacy ``compiler.*`` imports.

PLENA_Compiler packages live at the repository root (``aten``, ``generator``,
``asm_templates``, ...), but existing code imports them through ``compiler``.
Keep that import path local to this submodule instead of resolving to the
simulator sibling directory.
"""

from pathlib import Path

__path__ = [str(Path(__file__).resolve().parent.parent)]

