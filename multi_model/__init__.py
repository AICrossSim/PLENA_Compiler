from __future__ import annotations

import importlib
import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
INPUTS_DIR = PACKAGE_DIR / "inputs"
OUTPUTS_DIR = PACKAGE_DIR / "outputs"
KERNEL_COMPILERS_PKG_DIR = PACKAGE_DIR / "kernel_compilers"


def _ensure_project_root_on_path() -> None:
    project_root = str(PROJECT_ROOT)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


def _install_legacy_alias(alias: str, target: str) -> None:
    if alias in sys.modules:
        return
    sys.modules[alias] = importlib.import_module(target)


_ensure_project_root_on_path()
_install_legacy_alias("kernel_compilers", "multi_model.kernel_compilers")
_install_legacy_alias("plena", "multi_model.kernel_compilers.plena")

__all__ = [
    "INPUTS_DIR",
    "KERNEL_COMPILERS_PKG_DIR",
    "OUTPUTS_DIR",
    "PACKAGE_DIR",
    "PROJECT_ROOT",
]
