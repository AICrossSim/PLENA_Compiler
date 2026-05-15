"""Compatibility namespace for legacy ``compiler.*`` imports.

PLENA_Compiler packages live at the repository root (``aten``, ``generator``,
``asm_templates``, ...), but existing code imports them through ``compiler``.
Keep that import path local to this submodule instead of resolving to the
simulator sibling directory.

Also includes PLENA_Tools for ``compiler.sim_env_utils`` imports.
"""

from pathlib import Path

_compiler_root = Path(__file__).resolve().parent.parent
_tools_root = _compiler_root.parent / "PLENA_Tools"

__path__ = [str(_compiler_root), str(_tools_root)]

