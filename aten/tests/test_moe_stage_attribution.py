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

TESTS_DIR = pathlib.Path(__file__).resolve().parent
CI_WORKFLOW = pathlib.Path(__file__).resolve().parents[2] / ".github" / "workflows" / "ci.yml"

#: Test files in this directory that no CI job runs yet.
#:
#: Every one of them imports torch and some want a real checkpoint, so wiring
#: them up is its own piece of work. Pinning the set is what stops a *newly
#: added* unwired file from hiding among them.
_UNWIRED_TESTS = frozenset(
    {
        "test_bf16_numerical_stability.py",
        "test_gpt_oss_moe_assertions.py",
        "test_gpt_oss_moe_reference.py",
        "test_plena_compiler.py",
        "test_quantization_ablation.py",
    }
)


def _python_sources() -> list[pathlib.Path]:
    sources = sorted(PLENA_DIR.rglob("*.py"))
    assert sources, f"no Python sources found under {PLENA_DIR}; the guard would pass vacuously"
    return sources


def _functions(path: pathlib.Path):
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node


def _callee_name(func: ast.expr) -> str | None:
    """The bare name a call resolves to, for ``f(...)`` and ``obj.f(...)`` alike."""
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _is_moe_callee(name: str) -> bool:
    """Whether *name*'s ``stage=`` argument means a MoE attribution stage.

    ``stage`` is not a reserved word. The attention emitters take one too --
    ``qkt_multiply(stage="decode")`` selects prefill vs decode and has nothing
    to do with MoE attribution -- so matching every ``stage=`` keyword in the
    tree reports those as unknown stage names. Match on the callee instead.

    ``moe_stage_marker`` is covered by the ``moe_`` prefix.
    """
    return name.startswith(("moe_", "gpt_oss_"))


def _moe_stage_arguments(tree: ast.AST):
    """Yield ``(callee, keyword)`` for every literal ``stage=`` on a MoE callee."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        callee = _callee_name(node.func)
        if callee is None or not _is_moe_callee(callee):
            continue
        for keyword in node.keywords:
            if keyword.arg == "stage" and isinstance(keyword.value, ast.Constant):
                yield callee, keyword


#: Callables that pre-bind arguments, turning a required parameter into a
#: supplied one at the binding site rather than in a signature.
_PARTIAL_BINDERS = frozenset({"partial", "partialmethod"})


def _stage_defaulting_lambdas(tree: ast.AST):
    """Yield lambdas that give ``stage`` a default.

    :func:`_functions` walks ``def`` and ``async def`` only. A lambda carries
    the same ``arguments`` node and the same failure mode, and nothing was
    looking at it.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Lambda) and _defaulted_stage_params(node):
            yield node


def _stage_binding_calls(tree: ast.AST):
    """Yield ``functools.partial``-style calls that pre-bind ``stage``.

    ``partial(emit, stage="gather")`` hands back a callable whose ``stage`` is
    already supplied. Every call through it omits the argument and is billed to
    whatever the binding site chose -- a default by another route, with exactly
    the consequence the required parameter exists to prevent.
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if _callee_name(node.func) in _PARTIAL_BINDERS and any(
            keyword.arg == "stage" for keyword in node.keywords
        ):
            yield node


def _stage_injecting_decorators(tree: ast.AST):
    """Yield ``(func, decorator)`` for decorators handed a ``stage=`` argument.

    Catches the declared shape, ``@with_stage(stage="gather")``. A decorator
    that injects a stage without naming it in its own call -- reading it from a
    closure, a registry or an attribute -- is not detectable without resolving
    what the decorator does, which is well beyond a source lint.

    TODO: if such a decorator is ever introduced, the check has to become a
    convention (an explicit allowlist of stage-injecting decorators) rather than
    a deeper analysis.
    """
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Call) and any(
                keyword.arg == "stage" for keyword in decorator.keywords
            ):
                yield node, decorator


def _defaulted_stage_params(func) -> list[tuple[ast.arg, ast.expr]]:
    """Every ``stage`` parameter of *func* that carries a default, with it.

    Keyword-only defaults sit in ``kw_defaults`` positionally aligned with
    ``kwonlyargs`` (``None`` where there is no default). Positional defaults sit
    in ``defaults``, right-aligned against ``posonlyargs + args``, hence the
    padding.
    """
    args = func.args
    defaulted = [
        (arg, default)
        for arg, default in zip(args.kwonlyargs, args.kw_defaults)
        if arg.arg == "stage" and default is not None
    ]
    positional = args.posonlyargs + args.args
    padding = len(positional) - len(args.defaults)
    defaulted += [
        (arg, args.defaults[index - padding])
        for index, arg in enumerate(positional)
        if arg.arg == "stage" and index >= padding
    ]
    return defaulted


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
            for _arg, default in _defaulted_stage_params(func):
                offenders.append(f"{path.name}:{func.lineno} {func.name}(stage={ast.unparse(default)})")

    assert not offenders, (
        "these emitters give `stage` a default, so a caller that forgets it is "
        "silently billed to the wrong stage instead of failing:\n  "
        + "\n  ".join(offenders)
    )


def test_stage_defaults_are_not_reintroduced_indirectly() -> None:
    """A ``def`` signature is not the only way to supply ``stage``.

    :func:`test_stage_parameters_have_no_default` walks ``def`` and ``async
    def``. Three constructs get a stage in without one, each leaving call sites
    that never name it -- the exact condition the required parameter exists to
    prevent, reached by a route the lint did not look at.
    """
    offenders: list[str] = []
    for path in _python_sources():
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in _stage_defaulting_lambdas(tree):
            offenders.append(f"{path.name}:{node.lineno} lambda with a defaulted `stage`")
        for node in _stage_binding_calls(tree):
            offenders.append(
                f"{path.name}:{node.lineno} {_callee_name(node.func)}(..., stage=...) pre-binds `stage`"
            )
        for func, decorator in _stage_injecting_decorators(tree):
            offenders.append(
                f"{path.name}:{decorator.lineno} @{_callee_name(decorator.func)}(stage=...) on {func.name}"
            )

    assert not offenders, (
        "these supply `stage` without a signature default, so callers still never "
        "have to name one:\n  " + "\n  ".join(offenders)
    )


def test_guard_would_catch_an_indirectly_supplied_stage(tmp_path: pathlib.Path) -> None:
    """Prove the three indirect checks can fail.

    Same reasoning as the signature self-check: a lint that has never fired is
    indistinguishable from one that cannot.
    """
    tree = ast.parse(
        "emit_gather = lambda rows, *, stage='gather': None\n"
        "emit_clean = lambda rows, *, stage: None\n"
        "bound = functools.partial(emit, stage='gather')\n"
        "bound_method = partialmethod(emit, stage='gather')\n"
        "bound_clean = functools.partial(emit, rows=[0])\n"
        "@with_stage(stage='gather')\n"
        "def decorated(self) -> None: ...\n"
        "@some_other_decorator(name='x')\n"
        "def undecorated(self) -> None: ...\n"
    )

    lambdas = [node.lineno for node in _stage_defaulting_lambdas(tree)]
    bindings = [_callee_name(node.func) for node in _stage_binding_calls(tree)]
    decorated = [func.name for func, _decorator in _stage_injecting_decorators(tree)]

    assert lambdas == [1], f"defaulted-stage lambdas not detected; found {lambdas}"
    assert bindings == ["partial", "partialmethod"], f"partial bindings not detected; found {bindings}"
    assert decorated == ["decorated"], f"stage-injecting decorators not detected; found {decorated}"


def test_stage_arguments_are_declared_moe_stages() -> None:
    """Every literal ``stage=`` argument must name a real stage.

    ``moe_stage_marker`` already rejects unknown names, but only on the paths a
    test actually emits. A typo on a rarely-exercised branch would otherwise
    reach the emulator, land in ``unresolved_stage_markers``, and leave that
    region inheriting the previous marker.

    Scoped to MoE callees by :func:`_is_moe_callee`, so an unrelated emitter
    that happens to take a ``stage`` argument is not reported as a bad stage
    name.
    """
    declared = _declared_moe_stages()
    offenders: list[str] = []
    for path in _python_sources():
        tree = ast.parse(path.read_text(), filename=str(path))
        for callee, keyword in _moe_stage_arguments(tree):
            if keyword.value.value not in declared:
                offenders.append(
                    f"{path.name}:{keyword.value.lineno} {callee}(stage={keyword.value.value!r})"
                )

    assert not offenders, (
        "these call sites pass a stage name that is not in MOE_STAGES:\n  " + "\n  ".join(offenders)
    )


def test_non_moe_stage_arguments_are_not_flagged() -> None:
    """``stage=`` on an unrelated emitter is not a MoE stage name.

    The attention path calls ``qkt_multiply(stage="decode")``, where ``stage``
    selects prefill vs decode. A matcher keyed on the argument name alone
    reports that as an unknown MoE stage -- a false positive on correct code,
    which is how a lint gets suppressed and then ignored.
    """
    tree = ast.parse(
        'qkt_multiply(d=16, stage="decode", mlen=4)\n'
        'attention_softmax(stage="attn_input")\n'
        'builder.flash_attention(stage="prefill")\n'
        'moe_expert_activation_v0(builder, stage="expert_activation")\n'
    )
    matched = [(callee, keyword.value.value) for callee, keyword in _moe_stage_arguments(tree)]

    assert matched == [("moe_expert_activation_v0", "expert_activation")], (
        "the stage-argument matcher is not scoped to MoE callees; it picked up "
        f"{matched}"
    )


def test_every_test_file_here_is_wired_into_ci() -> None:
    """A guard no job runs is worth exactly nothing.

    These lints sat in the tree with no workflow invoking them, which is
    indistinguishable from never having written them. This catches the next
    instance: a new file under ``aten/tests`` has to be named by a workflow, or
    else declared unwired on purpose.

    Matched as a substring of the workflow text rather than by parsing the job
    graph. A file named in a commented-out step would count as covered, which is
    the one false negative -- worth it to keep this readable and dependency-free.
    """
    workflow = CI_WORKFLOW.read_text()
    present = {path.name for path in TESTS_DIR.glob("test_*.py")}
    assert present, f"no test files found under {TESTS_DIR}; this guard would pass vacuously"

    unwired = sorted(name for name in present if name not in workflow and name not in _UNWIRED_TESTS)
    assert not unwired, (
        "these test files are in the tree but no CI job runs them, so they cannot "
        f"fail anything:\n  " + "\n  ".join(unwired) + f"\nadd a step to {CI_WORKFLOW.name} "
        "or, if that is deliberate, add them to _UNWIRED_TESTS with a reason"
    )

    stale = sorted(name for name in _UNWIRED_TESTS if name not in present or name in workflow)
    assert not stale, (
        "_UNWIRED_TESTS names files that are now wired into CI or no longer exist; "
        f"drop them so the exemption list keeps meaning something:\n  " + "\n  ".join(stale)
    )


def test_guard_would_catch_a_defaulted_stage_parameter(tmp_path: pathlib.Path) -> None:
    """The lint above is only worth having if it can fail. Prove it does.

    Without this, a refactor that broke the AST walk (a renamed field, a missed
    argument category) would leave both tests passing over zero findings.

    Driven through ``_functions`` and ``_defaulted_stage_params`` -- the same
    two helpers the real lint uses -- against a fixture written to disk. A
    self-check that reimplements the walk it is checking proves only that the
    copy agrees with itself, which is exactly the bug it is meant to catch.
    """
    fixture = tmp_path / "fixture.py"
    fixture.write_text(
        "def emit(self, x, *, policy: str = 'p', stage: str = 'gather', name: str) -> None: ...\n"
        "def emit_positional(self, stage: str = 'gather') -> None: ...\n"
        "async def emit_async(self, *, stage: str = 'gather') -> None: ...\n"
        "def emit_ok(self, x, *, stage: str, name: str) -> None: ...\n"
        "async def emit_async_ok(self, *, stage: str) -> None: ...\n"
        "def emit_no_stage(self, x: str = 'y') -> None: ...\n"
    )

    found = sorted(func.name for func in _functions(fixture) if _defaulted_stage_params(func))

    assert found == ["emit", "emit_async", "emit_positional"], (
        f"the AST walk no longer detects defaulted stage parameters, so "
        f"test_stage_parameters_have_no_default cannot fail; detected {found}"
    )
