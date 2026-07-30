"""Static guards on the MoE ``@stage=`` attribution contract.

Stage markers are *sticky and authoritative*: once a program emits any marker,
the emulator disables its legacy substring rules and bills every subsequent
instruction to the most recent marker. Two consequences drive the tests here.

1. A helper called from inside a marked region inherits the enclosing marker.
   So an emitter reused across stages must be told which one it is serving, and
   a **default** for that parameter is a silent wrong answer rather than a
   missing one. This is not hypothetical: the shared-expert sigmoid gate reused
   ``moe_materialize_route_weights_for_active_rows_v0``, inherited its
   ``expert_route_weight`` default, and misattributed 999 instructions while
   every total still added up and every test stayed green.

2. A misattributed stage does not change instruction counts, cycle totals or
   numerical results. Nothing downstream fails. The only place the mistake is
   visible is the source line that omitted the argument -- so that is where it
   has to be caught.

Both tests read the source with :mod:`ast` rather than importing the modules and
using :mod:`inspect`. Not to avoid the dependency chain -- ``aten/__init__.py``
pulls in the op registry at collection time regardless -- but because parsing
covers things introspection cannot:

- a newly added module under ``aten/plena`` is checked the moment it lands, with
  nobody having to remember to register it;
- ``_DEPRECATED_METHOD_ALIASES`` rebinds emitters onto the mixin via ``setattr``,
  so a signature-walk over class attributes sees the same function twice and the
  declaration site not at all;
- literal ``stage=`` *arguments* are visible, which a signature walk cannot see.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

PLENA_DIR = pathlib.Path(__file__).resolve().parents[1] / "plena"
ROUTED_MOE_SOURCE = PLENA_DIR / "program_routed_moe.py"


def _python_sources() -> list[pathlib.Path]:
    sources = sorted(PLENA_DIR.rglob("*.py"))
    assert sources, f"no Python sources found under {PLENA_DIR}; the guard would pass vacuously"
    return sources


def _functions(path: pathlib.Path):
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node


def _declared_moe_stages() -> set[str]:
    """Parse ``MOE_STAGES`` out of ``program_routed_moe.py``."""
    tree = ast.parse(ROUTED_MOE_SOURCE.read_text(), filename=str(ROUTED_MOE_SOURCE))
    for node in ast.walk(tree):
        target = getattr(node, "target", None)
        if isinstance(node, ast.AnnAssign) and isinstance(target, ast.Name) and target.id == "MOE_STAGES":
            stages = {
                literal.value
                for literal in ast.walk(node.value)
                if isinstance(literal, ast.Constant) and isinstance(literal.value, str)
            }
            assert stages, "parsed an empty MOE_STAGES; the literal shape changed"
            return stages
    pytest.fail(f"{ROUTED_MOE_SOURCE} no longer declares MOE_STAGES")


def test_stage_parameters_have_no_default() -> None:
    """An emitter that takes ``stage`` must force the caller to name one.

    Scoped to "has a ``stage`` parameter" rather than "is reused across >= 2
    stages", because the two coincide by construction: a helper serving exactly
    one stage hardcodes its marker and never takes the parameter at all. Taking
    it *is* the declaration that the emitter is stage-polymorphic.
    """
    offenders: list[str] = []
    for path in _python_sources():
        for func in _functions(path):
            args = func.args
            defaulted = [
                (arg, default)
                for arg, default in zip(args.kwonlyargs, args.kw_defaults)
                if arg.arg == "stage" and default is not None
            ]
            # Positional `stage` params: line them up with their trailing defaults.
            positional = args.posonlyargs + args.args
            padding = len(positional) - len(args.defaults)
            defaulted += [
                (arg, args.defaults[index - padding])
                for index, arg in enumerate(positional)
                if arg.arg == "stage" and index >= padding
            ]
            for arg, default in defaulted:
                offenders.append(f"{path.name}:{func.lineno} {func.name}(stage={ast.unparse(default)})")

    assert not offenders, (
        "these emitters give `stage` a default, so a caller that forgets it is "
        "silently billed to the wrong stage instead of failing:\n  "
        + "\n  ".join(offenders)
    )


def test_stage_arguments_are_declared_moe_stages() -> None:
    """Every literal ``stage=`` argument must name a real stage.

    ``moe_stage_marker`` already rejects unknown names, but only on the paths a
    test actually emits. A typo on a rarely-exercised branch would otherwise
    reach the emulator, land in ``unresolved_stage_markers``, and leave that
    region inheriting the previous marker.
    """
    declared = _declared_moe_stages()
    offenders: list[str] = []
    for path in _python_sources():
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for keyword in node.keywords:
                if keyword.arg != "stage" or not isinstance(keyword.value, ast.Constant):
                    continue
                if keyword.value.value not in declared:
                    offenders.append(f"{path.name}:{node.lineno} stage={keyword.value.value!r}")

    assert not offenders, (
        "these call sites pass a stage name that is not in MOE_STAGES:\n  " + "\n  ".join(offenders)
    )


def test_guard_would_catch_a_defaulted_stage_parameter() -> None:
    """The lint above is only worth having if it can fail. Prove it does.

    Without this, a refactor that broke the AST walk (a renamed field, a missed
    argument category) would leave both tests passing over zero findings.
    """
    module = ast.parse(
        "def emit(self, x, *, policy: str = 'p', stage: str = 'gather', name: str) -> None: ...\n"
        "def emit_positional(self, stage: str = 'gather') -> None: ...\n"
        "def emit_ok(self, x, *, stage: str, name: str) -> None: ...\n"
    )
    found = []
    for func in (node for node in ast.walk(module) if isinstance(node, ast.FunctionDef)):
        args = func.args
        hits = [
            arg
            for arg, default in zip(args.kwonlyargs, args.kw_defaults)
            if arg.arg == "stage" and default is not None
        ]
        positional = args.posonlyargs + args.args
        padding = len(positional) - len(args.defaults)
        hits += [
            arg for index, arg in enumerate(positional) if arg.arg == "stage" and index >= padding
        ]
        if hits:
            found.append(func.name)

    assert found == ["emit", "emit_positional"], (
        f"the AST walk no longer detects defaulted stage parameters, so "
        f"test_stage_parameters_have_no_default cannot fail; detected {found}"
    )
