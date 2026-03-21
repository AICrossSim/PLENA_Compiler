"""
PLENA multi-model compiler package.

This package groups the parser, codegen, kernel compiler, and reporting helpers
used to trace models and lower them into PLENA-oriented assembly.

Common callable APIs exposed from the package root:

- `VLMModelParser`
  Main parser class for model loading, pseudo-execution tracing, call-tree
  flattening, and model-info extraction.
- `template_qwen3_vl_inputs(processor, image_path, text=...)`
  Helper for building processor-backed Qwen3-VL style inputs.
- `VLMCodegenEnvironment`
  Hardware, scheduler, template, and type-registry container for codegen.
- `VLMAssemblyGenerator`
  Assembly generator that dispatches traced nodes to registered lowering
  handlers.
- `vlm_codegen(nodes, model_info, ...)`
  Convenience wrapper for generating PLENA assembly from traced nodes.
- `analyse_trace_utilization(...)`
  Compute and activation-memory utilization analysis based on traced execution.
- `render_markdown_report(report, ...)`
  Markdown rendering helper for utilization reports.
- `PLENA_BACKEND`
  Backend helper used by tracing / compilation flows.

Recommended usage from inside this package:

- Import public APIs from `multi_model` when you want a stable entrypoint, for
  example:
  `from multi_model import VLMModelParser, vlm_codegen, analyse_trace_utilization`
- Prefer package-relative imports or `multi_model.*` absolute imports inside
  leaf modules instead of manually modifying `sys.path`.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
INPUTS_DIR = PACKAGE_DIR / "inputs"
OUTPUTS_DIR = PACKAGE_DIR / "outputs"
KERNEL_COMPILERS_PKG_DIR = PACKAGE_DIR / "kernel_compilers"

_PUBLIC_API = {
    "analyse_trace_utilization": ("multi_model.utilization_report", "analyse_trace_utilization"),
    "PLENA_BACKEND": ("multi_model.plena_backend", "PLENA_BACKEND"),
    "render_markdown_report": ("multi_model.utilization_report", "render_markdown_report"),
    "template_qwen3_vl_inputs": ("multi_model.vlm_parser", "template_qwen3_vl_inputs"),
    "VLMAssemblyGenerator": ("multi_model.vlm_codegen_generator", "VLMAssemblyGenerator"),
    "VLMCodegenEnvironment": ("multi_model.vlm_codegen_env", "VLMCodegenEnvironment"),
    "VLMModelParser": ("multi_model.vlm_parser", "VLMModelParser"),
    "vlm_codegen": ("multi_model.vlm_codegen", "vlm_codegen"),
}


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


def __getattr__(name: str):
    if name in _PUBLIC_API:
        module_name, attr_name = _PUBLIC_API[name]
        module = importlib.import_module(module_name)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "INPUTS_DIR",
    "KERNEL_COMPILERS_PKG_DIR",
    "OUTPUTS_DIR",
    "PACKAGE_DIR",
    "PROJECT_ROOT",
    "analyse_trace_utilization",
    "PLENA_BACKEND",
    "render_markdown_report",
    "template_qwen3_vl_inputs",
    "VLMAssemblyGenerator",
    "VLMCodegenEnvironment",
    "VLMModelParser",
    "vlm_codegen",
]
