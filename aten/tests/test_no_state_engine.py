"""The static path carries no descriptor machinery, and must not acquire any.

PLENA is a **statically scheduled** accelerator: the compiler decides every
address at compile time. A descriptor fetched from HBM at runtime re-decides
something the compiler already decided, and a residency cache defers a decision
the compiler is not allowed to defer. Neither belongs on this path, and this file
fails if either reappears.

What the recurrent kernels need instead is two ordinary arithmetic/data-move
opcodes -- `V_SOFTPLUS_V` (0x3D) and `S_MAP_FP_V` (0x3E) -- plus one
model-independent Matrix-tile opcode family, `L_TILE` (0x3F). `V_FMA_VF`
is only a readable assembler alias for the existing `V_MUL_VF` opcode's
funct1[3] accumulate mode. None fetches a state descriptor or names Mamba/KDA.

There is nothing to delete: this branch was cut from `a4b3e7de` on main, which
never had any of it. So this file is not a cleanup, it is the guard that keeps
the property. A grep is a check that ran once; a test is a check that keeps
running.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

#: The vocabulary of a run-time state engine: two module names and two opcode
#: names that only exist if the machine fetches its own work descriptors. None of
#: them may appear outside the documents that explain why they do not.
_FORBIDDEN = ("X_STATE", "L_SCATTER_M", "StateDescriptor", "state_engine")

#: Paths whose *purpose* is to record the decision not to build one.
_ALLOWED_PREFIXES = (
    "docs/superpowers/",  # the plan and the progress log
    "doc/static_path_measurements.md",
    "aten/plena/program_kda_layer.py",
    "aten/plena/hybrid_compile_report.py",
    "aten/tests/test_lstream_packet_lowering.py",
    "aten/tests/test_no_state_engine.py",
)

_SEARCH_SUFFIXES = (".rs", ".py", ".svh", ".json", ".toml")


def _repo_root() -> Path:
    """Return either a Simulator checkout or a standalone Compiler clone."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "transactional_emulator").is_dir():
            return parent
        if (parent / "aten").is_dir() and (parent / "doc" / "operation.svh").is_file():
            return parent
    raise AssertionError("test is not inside a PLENA Simulator or Compiler checkout")


def _compiler_root(root: Path) -> Path:
    submodule = root / "PLENA_Compiler"
    return submodule if (submodule / "doc" / "operation.svh").is_file() else root


def _tracked_files(root: Path) -> list[Path]:
    out: list[Path] = []
    repos = (
        (root, root / "PLENA_Compiler")
        if (root / "transactional_emulator").is_dir()
        else (root,)
    )
    for repo in repos:
        if not (repo / ".git").exists():
            continue
        listing = subprocess.run(
            ["git", "-C", str(repo), "ls-files"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()
        out.extend(repo / name for name in listing)
    return out


def test_no_descriptor_machinery_anywhere():
    """No opcode, module or type from the descriptor design.

    `kda_state_engine_step` and `KdaRecurrentState` in the CPU reference are *function
    and dataclass names*, not the instruction -- they are the boundary the
    reference defines, and the static lowering targets exactly that boundary.
    The pattern below is word-anchored so it does not catch them.
    """
    root = _repo_root()
    pattern = re.compile(r"\b(" + "|".join(_FORBIDDEN) + r")\b")
    hits: list[str] = []
    for path in _tracked_files(root):
        rel = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
        rel = rel.replace("PLENA_Compiler/", "", 1)
        if rel.startswith(_ALLOWED_PREFIXES) or path.suffix not in _SEARCH_SUFFIXES:
            continue
        try:
            text = path.read_text(errors="ignore")
        except OSError:  # pragma: no cover - unreadable file
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if pattern.search(line):
                hits.append(f"{rel}:{lineno}: {line.strip()[:90]}")
    assert not hits, "descriptor machinery has come back:\n  " + "\n  ".join(hits[:20])


def test_the_state_engine_directories_do_not_exist():
    root = _repo_root()
    compiler = _compiler_root(root)
    for gone in (
        root / "transactional_emulator" / "src" / "state_engine",
        compiler / "aten" / "state",
        compiler / "spec",
    ):
        assert not gone.exists(), f"{gone} exists; the static path does not need it"


def test_extension_opcode_ownership_is_explicit_and_conflict_free():
    """Shared Expert and static recurrent work occupy disjoint encodings."""
    root = _repo_root()
    svh = (_compiler_root(root) / "doc" / "operation.svh").read_text()
    defined = {
        int(m.group(2), 16): m.group(1)
        for m in re.finditer(r"(\w+)\s*=\s*6'h([0-9A-Fa-f]+)", svh)
    }
    expected = {
        0x39: "C_ROUTE_BEGIN",
        0x3A: "C_ROUTE_LOOP_START",
        0x3B: "C_ROUTE_LOOP_END",
        0x3C: "V_ROUTE_MUL",
        0x3D: "V_SOFTPLUS_V",
        0x3E: "S_MAP_FP_V",
        0x3F: "L_TILE",
    }
    for opcode, name in expected.items():
        assert defined.get(opcode) == name, (
            f"{name} should hold 0x{opcode:02X}; Shared Expert and L-Compute "
            "must not independently claim the same physical opcode"
        )
    assert max(defined) == 0x3F
    assert "V_FMA_VF" not in defined.values(), "FMA must remain a V_MUL_VF mode"


#: What `origin/main` stops at -- `C_SET_TOPK_REG`. Everything past it is this
#: branch's, and there are four.
MAIN_LAST_OPCODE = 0x38

#: Two ordinary operations plus one model-independent address mode. The routed
#: MoE opcodes at 0x39..0x3C are owned by the Shared Expert work.
OPCODES_ADDED_HERE = {0x3D, 0x3E, 0x3F}


def test_static_recurrent_path_uses_exactly_three_physical_opcodes():
    """Pin the exact ISA delta and keep model-specific state machinery out.

    `origin/main` stops at `C_SET_TOPK_REG` (0x38). Shared Expert owns
    0x39..0x3C. This work owns two ordinary operations at 0x3D..0x3E and one
    general address mode at 0x3F; FMA is a mode of V_MUL_VF, not an opcode.

    `L_TILE` is the only physical layout opcode. Its CFG/EXEC forms configure
    and walk a compiler-owned Matrix view; they do not fetch descriptors, hold
    a queue, or manage residency. The historical `L_CFG` research form shares
    this opcode but is excluded from the frozen Matrix-SRAM RTL candidate.
    """
    root = _repo_root()
    svh = (_compiler_root(root) / "doc" / "operation.svh").read_text()
    codes = {
        int(m.group(1), 16) for m in re.finditer(r"\w+\s*=\s*6'h([0-9A-Fa-f]+)", svh)
    }
    route_reserved = {0x39, 0x3A, 0x3B, 0x3C}
    added_by_this_work = {c for c in codes if c > MAIN_LAST_OPCODE} - route_reserved
    assert added_by_this_work == OPCODES_ADDED_HERE, (
        f"opcodes past {hex(MAIN_LAST_OPCODE)}: "
        f"{sorted(hex(c) for c in added_by_this_work)}; this work adds exactly "
        "three -- V_SOFTPLUS_V 0x3D, S_MAP_FP_V 0x3E, and L_TILE 0x3F; "
        "V_FMA_VF must not spend an opcode"
    )
