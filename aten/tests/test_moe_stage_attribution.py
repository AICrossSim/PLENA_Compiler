"""Static guards on the MoE ``@stage=`` attribution contract.

Stage markers are *sticky and authoritative*: once a program emits any marker,
the emulator disables its legacy substring rules and bills every subsequent
instruction to the most recent marker. Two consequences drive the tests here.

1. A helper called from inside a marked region inherits the enclosing marker.
   So an emitter reused across stages must be told which one it is serving, and
   a **default** for that parameter is a silent wrong answer rather than a
   missing one.

2. A misattributed stage does not change instruction counts, cycle totals or
   numerical results. Nothing downstream fails. The only place the mistake is
   visible is the source line that omitted the argument -- so that is where it
   has to be caught.

These guards read the source with :mod:`ast` rather than importing the modules
and using :mod:`inspect`. ``_DEPRECATED_METHOD_ALIASES`` rebinds emitters onto
the mixin via ``setattr``, so a signature walk over class attributes sees the
same function twice and the declaration site not at all -- and literal
``stage=`` *arguments*, which the stage-name guard is built on, are invisible
to it.
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

    ``stage`` is not a reserved word -- the attention emitters take one too --
    so matching every ``stage=`` keyword in the tree reports those as unknown
    stage names.

    ``moe_stage_marker`` is covered by the ``moe_`` prefix.
    """
    return name.startswith(("moe_", "gpt_oss_"))


#: Callees taking the stage name as their *first positional* argument.
#:
#: ``moe_stage_marker(stage, detail)`` is the function that actually emits a
#: marker, and its call sites under ``aten/plena`` pass the stage positionally,
#: so a keyword-only matcher would not see any of them.
_POSITIONAL_STAGE_CALLEES = frozenset({"moe_stage_marker"})


def _moe_stage_arguments(tree: ast.AST):
    """Yield ``(callee, constant)`` for every literal stage name on a MoE callee.

    Covers both spellings: a ``stage=`` keyword argument, and the first
    positional argument of the callees in :data:`_POSITIONAL_STAGE_CALLEES`.
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        callee = _callee_name(node.func)
        if callee is None or not _is_moe_callee(callee):
            continue
        if callee in _POSITIONAL_STAGE_CALLEES and node.args:
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                yield callee, first
        for keyword in node.keywords:
            if keyword.arg == "stage" and isinstance(keyword.value, ast.Constant):
                yield callee, keyword.value


#: Callables that pre-bind arguments, turning a required parameter into a
#: supplied one at the binding site rather than in a signature.
_PARTIAL_BINDERS = frozenset({"partial", "partialmethod"})


def _stage_defaulting_lambdas(tree: ast.AST):
    """Yield lambdas that give ``stage`` a default.

    :func:`_functions` walks ``def`` and ``async def`` only. A lambda carries
    the same ``arguments`` node and the same failure mode.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Lambda) and _defaulted_stage_params(node):
            yield node


def _stage_binding_calls(tree: ast.AST):
    """Yield ``functools.partial``-style calls that pre-bind ``stage``.

    ``partial(emit, stage="gather")`` hands back a callable whose ``stage`` is
    already supplied, so every call through it is billed to whatever the
    binding site chose.
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


#: Containers a ``MOE_STAGES`` declaration may evaluate to.
#:
#: A ``dict`` is in the list because iterating one yields its keys, which is the
#: sensible reading of ``{"gather": "why it exists"}``. ``str`` and ``bytes`` are
#: deliberately absent: both iterate, so a scalar would otherwise parse as a set
#: of one-character "stage names" rather than failing.
_STAGE_CONTAINERS = (set, frozenset, list, tuple, dict)


def _module_level_binding(tree: ast.Module, name: str) -> ast.expr | None:
    """The expression bound to *name* at module scope, or ``None``.

    Module scope only, over ``tree.body`` rather than :func:`ast.walk`. A
    same-named local -- ``MOE_STAGES`` rebound inside a function or a class body
    -- is a different binding, and resolving to it means the guard checks call
    sites against a vocabulary the module never exports.
    """
    for node in tree.body:
        if isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == name and node.value is not None:
                return node.value
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return node.value
    return None


def _literal_stages(node: ast.expr) -> set[str]:
    """Evaluate a stage-set expression, raising on any shape it cannot prove.

    Handles the three forms a set-of-names declaration takes -- a literal, a
    ``frozenset()``/``set()`` around one, and ``|`` unions of those -- and
    refuses everything else. Refusing is the whole point: a partial set is worse
    than no set, because every caller of this reports "not a declared stage" by
    *absence*, so a name the parser dropped becomes a false offender and a name
    it invented silently stops being checked.
    """
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _literal_stages(node.left) | _literal_stages(node.right)
    if isinstance(node, ast.Call):
        callee = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", None)
        if callee in ("frozenset", "set") and not node.keywords:
            if not node.args:
                return set()
            if len(node.args) == 1:
                return _literal_stages(node.args[0])
        raise ValueError(f"unsupported call in MOE_STAGES: {ast.unparse(node)}")
    try:
        value = ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError) as exc:
        raise ValueError(f"MOE_STAGES is not a literal: {ast.unparse(node)} ({exc})") from exc
    if not isinstance(value, _STAGE_CONTAINERS):
        raise ValueError(f"MOE_STAGES is {type(value).__name__}, not a collection of stage names")
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"non-string stage name in MOE_STAGES: {item!r}")
    return set(value)


def _parse_moe_stages(source: str) -> set[str]:
    """The stage vocabulary *source* declares, or :exc:`ValueError`.

    The emulator has to recover the same vocabulary from the same file to check
    its ``StageKind`` variants against it (``MOE_STAGES_EXTRACTOR`` in
    ``transactional_emulator/src/stage_profile.rs``). This accepts and rejects
    the same shapes, so a declaration that satisfies one repo satisfies both.
    """
    binding = _module_level_binding(ast.parse(source), "MOE_STAGES")
    if binding is None:
        raise ValueError("no module-level MOE_STAGES assignment")
    return _literal_stages(binding)


def _declared_moe_stages() -> set[str]:
    """Parse ``MOE_STAGES`` out of ``program_routed_moe.py``."""
    try:
        stages = _parse_moe_stages(ROUTED_MOE_SOURCE.read_text())
    except ValueError as exc:
        pytest.fail(f"cannot recover MOE_STAGES from {ROUTED_MOE_SOURCE}: {exc}")
    assert stages, f"{ROUTED_MOE_SOURCE} declares an empty MOE_STAGES; the guard would pass vacuously"
    return stages


def test_stage_parameters_have_no_default() -> None:
    """An emitter that takes ``stage`` must force the caller to name one.

    Scoped to "has a ``stage`` parameter": a helper serving exactly one stage
    hardcodes its marker and never takes the parameter at all, so taking it *is*
    the declaration that the emitter is stage-polymorphic.
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
    that never name it.
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
    """
    declared = _declared_moe_stages()
    offenders: list[str] = []
    for path in _python_sources():
        tree = ast.parse(path.read_text(), filename=str(path))
        for callee, constant in _moe_stage_arguments(tree):
            if constant.value not in declared:
                offenders.append(f"{path.name}:{constant.lineno} {callee}({constant.value!r})")

    assert not offenders, (
        "these call sites pass a stage name that is not in MOE_STAGES:\n  " + "\n  ".join(offenders)
    )


def test_moe_stages_parser_handles_adversarial_declaration_shapes() -> None:
    """The declaration may be respelled without weakening the guard.

    ``MOE_STAGES`` is the vocabulary every stage-name check is measured against,
    so the parser is their single point of failure. Nothing forces the
    declaration to keep today's exact shape -- dropping the annotation, unioning
    two halves, spelling the members over a list are all ordinary edits.

    The list is the emulator's, from ``moe_stages_extractor_handles_adversarial_
    literal_shapes`` in ``transactional_emulator/src/stage_profile.rs``. Both
    repos parse the same file for the same set; pinning the same shapes is what
    keeps them from disagreeing about what it says.
    """
    for label, source, expected in [
        (
            "single-quoted names",
            "MOE_STAGES = {'gather', 'router_topk'}\n",
            {"gather", "router_topk"},
        ),
        (
            "a comment containing a closing brace",
            'MOE_STAGES = {\n    "gather",  # closes with } here\n    "router_topk",\n}\n',
            {"gather", "router_topk"},
        ),
        (
            "a union of frozensets",
            'MOE_STAGES = frozenset({"gather"}) | frozenset({"router_topk"})\n',
            {"gather", "router_topk"},
        ),
        (
            "a union of bare set literals",
            'MOE_STAGES = {"gather"} | {"router_topk"} | {"scatter_combine"}\n',
            {"gather", "router_topk", "scatter_combine"},
        ),
        (
            "the current shape: annotated, multiline, trailing comma",
            'MOE_STAGES: frozenset[str] = frozenset(\n    {\n        "gather",\n        "router_topk",\n    }\n)\n',
            {"gather", "router_topk"},
        ),
        (
            "a frozenset over a list",
            'MOE_STAGES = frozenset(["gather", "router_topk"])\n',
            {"gather", "router_topk"},
        ),
        (
            "a docstring mentioning MOE_STAGES before the declaration",
            '"""Validated against MOE_STAGES = {"not", "real"}."""\n\nMOE_STAGES = {"gather"}\n',
            {"gather"},
        ),
        (
            "a decoy dict holding a brace in a string",
            '_NOTES = {"a": "}"}\nMOE_STAGES = {"gather", "router_topk"}\n',
            {"gather", "router_topk"},
        ),
        (
            "a same-named local inside a function",
            'def f():\n    MOE_STAGES = {"wrong"}\n    return MOE_STAGES\n\nMOE_STAGES = {"gather"}\n',
            {"gather"},
        ),
        (
            "implicit string concatenation inside the literal",
            'MOE_STAGES = {"expert_" "projection"}\n',
            {"expert_projection"},
        ),
    ]:
        assert _parse_moe_stages(source) == expected, f"the parser mis-read {label}: {source!r}"


def test_moe_stages_parser_rejects_shapes_it_cannot_prove() -> None:
    """A shape the parser cannot evaluate must fail, never return a partial set.

    This is what keeps the vocabulary honest. A scraped set -- every string
    constant under the assignment -- is neither sound nor complete, and both
    directions are silent:

    - too many names (a removed stage still named in a ``-`` operand, the losing
      branch of a conditional, prose sitting next to a key) leaves that name
      accepted at every call site, so the typo the guard exists to catch passes;
    - too few, or names that were never stages at all (an argument to a helper
      call, the literal prefix of an f-string), makes correct call sites read as
      offenders, which is how a lint gets suppressed.

    Shared with the emulator's ``moe_stages_extractor_rejects_shapes_it_cannot_
    prove``: both repos must reject the same shapes.
    """
    for label, source in [
        ("no declaration at all", 'OTHER = {"gather"}\n'),
        ("an annotation with no value", "MOE_STAGES: frozenset[str]\n"),
        ("a set comprehension", 'MOE_STAGES = {s for s in ("a", "b")}\n'),
        ("a call the parser cannot evaluate", 'MOE_STAGES = _load_stages("stages.json")\n'),
        ("a non-string member", 'MOE_STAGES = {"gather", 7}\n'),
        ("a scalar", 'MOE_STAGES = "gather"\n'),
        # Annotated respellings, which reach _module_level_binding's AnnAssign branch.
        (
            "an annotated call, whose argument is not a stage name",
            'MOE_STAGES: frozenset[str] = _load_stages("stages.json")\n',
        ),
        (
            "an annotated non-string member, silently dropped",
            'MOE_STAGES: frozenset[str] = frozenset({"gather", 7})\n',
        ),
        (
            "a stage retired with a set difference, still named in the operand",
            'MOE_STAGES: frozenset[str] = frozenset({"gather", "legacy"}) - frozenset({"legacy"})\n',
        ),
        (
            "a conditional, whose losing branch is not declared",
            'MOE_STAGES: frozenset[str] = frozenset({"gather"}) if _FLAG else frozenset({"legacy"})\n',
        ),
        (
            "an f-string member, whose literal prefix is not a stage name",
            'MOE_STAGES: frozenset[str] = frozenset({f"expert_{_KIND}"})\n',
        ),
        (
            "a lookup into a table the parser cannot see",
            'MOE_STAGES: frozenset[str] = frozenset(_TABLE["moe"])\n',
        ),
        (
            "only a function-local binding",
            'def f():\n    MOE_STAGES: frozenset[str] = frozenset({"wrong"})\n',
        ),
        (
            "only a class-body binding",
            'class Stages:\n    MOE_STAGES: frozenset[str] = frozenset({"wrong"})\n',
        ),
    ]:
        with pytest.raises(ValueError):
            stages = _parse_moe_stages(source)
            pytest.fail(
                f"the parser accepted {label} and answered {sorted(stages)!r}, so it can "
                f"return a vocabulary the module does not declare: {source!r}"
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
        'moe_stage_marker("gather", f"detail {x}")\n'
    )
    matched = [(callee, constant.value) for callee, constant in _moe_stage_arguments(tree)]

    assert sorted(matched) == [
        ("moe_expert_activation_v0", "expert_activation"),
        ("moe_stage_marker", "gather"),
    ], f"the stage-argument matcher is wrong; it picked up {matched}"


    typo = ast.parse('moe_stage_marker("gathr", "detail")\n')
    assert [c.value for _callee, c in _moe_stage_arguments(typo)] == ["gathr"], (
        "a positional stage name is invisible to the matcher, so the lint cannot "
        "see the construct that actually emits a marker"
    )


def test_every_test_file_here_is_wired_into_ci() -> None:
    """A guard no job runs is worth exactly nothing.

    A new file under ``aten/tests`` has to be named by a workflow, or else
    declared unwired on purpose.

    Matched as a substring of the workflow text rather than by parsing the job
    graph, so a file named in a commented-out step would count as covered.
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
