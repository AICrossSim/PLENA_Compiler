"""The static path carries no descriptor machinery, and must not acquire any.

PLENA is a **statically scheduled** accelerator: the compiler decides every
address at compile time. A descriptor fetched from HBM at runtime re-decides
something the compiler already decided, and a residency cache defers a decision
the compiler is not allowed to defer. Neither belongs on this path, and this file
fails if either reappears.

What the recurrent kernels need instead is three ordinary instructions --
`V_SOFTPLUS_V` (0x39), `S_MAP_FP_V` (0x3A) and `V_FMA_VF` (0x3B) -- each a
fixed-function ALU or move op with its operands named in the instruction word.
Everything else -- the state layout, the per-head streaming, the chunked
prefill -- is compiler work.

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

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

#: The vocabulary of a run-time state engine: two module names and two opcode
#: names that only exist if the machine fetches its own work descriptors. None of
#: them may appear outside the documents that explain why they do not.
_FORBIDDEN = ("X_STATE", "L_SCATTER_M", "StateDescriptor", "state_engine")

#: Paths whose *purpose* is to record the decision not to build one.
_ALLOWED_PREFIXES = (
    "docs/superpowers/",          # the plan and the progress log
    "doc/static_path_measurements.md",
    "aten/tests/test_no_state_engine.py",
)

_SEARCH_SUFFIXES = (".rs", ".py", ".svh", ".json", ".toml")


def _repo_root() -> Path:
    """The Simulator checkout, which contains the Compiler submodule."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "transactional_emulator").is_dir():
            return parent
    pytest.skip("not inside a Simulator checkout")


def _tracked_files(root: Path) -> list[Path]:
    out: list[Path] = []
    for repo in (root, root / "PLENA_Compiler"):
        if not (repo / ".git").exists():
            continue
        listing = subprocess.run(
            ["git", "-C", str(repo), "ls-files"],
            capture_output=True, text=True, check=True,
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
    assert not hits, (
        "descriptor machinery has come back:\n  " + "\n  ".join(hits[:20])
    )


def test_the_state_engine_directories_do_not_exist():
    root = _repo_root()
    for gone in (
        root / "transactional_emulator" / "src" / "state_engine",
        root / "PLENA_Compiler" / "aten" / "state",
        root / "PLENA_Compiler" / "spec",
    ):
        assert not gone.exists(), f"{gone} exists; the static path does not need it"


def test_the_descriptor_opcodes_are_still_free():
    """0x3D and 0x3F stay undefined.

    They are the slots a descriptor-driven state instruction and a
    descriptor-driven scatter would take. The compiler's opcode table must not
    define them and the emulator asserts 0x3F decodes to Invalid, so filling
    either needs a written argument rather than a free slot.
    """
    root = _repo_root()
    svh = (root / "PLENA_Compiler" / "doc" / "operation.svh").read_text()
    defined = {
        int(m.group(2), 16): m.group(1)
        for m in re.finditer(r"(\w+)\s*=\s*6'h([0-9A-Fa-f]+)", svh)
    }
    # 0x3D and 0x3F are the slots a descriptor-driven state instruction and a
    # descriptor-driven scatter would naturally take. Held free deliberately:
    # a free slot is not an argument for filling it.
    for opcode in (0x3D, 0x3F):
        assert opcode not in defined, (
            f"0x{opcode:02X} is defined as {defined[opcode]}. This path carries no "
            f"descriptor machinery; adding an opcode here needs an argument in "
            f"docs/superpowers/plans/, not just a free slot"
        )
    for opcode, name in ((0x39, "V_SOFTPLUS_V"), (0x3A, "S_MAP_FP_V"), (0x3B, "V_FMA_VF")):
        assert defined.get(opcode) == name, (
            f"{name} should hold 0x{opcode:02X} -- these three are what this work "
            f"adds to the ISA, and every one is an ordinary fixed-function op"
        )
    assert max(defined) == 0x3B, (
        f"the highest opcode is 0x{max(defined):02X}; adding another needs an "
        f"argument, not just a free slot"
    )


def test_exactly_one_opcode_was_added():
    """The whole design rests on this. If a second one appears, the claim that
    KDA and Mamba need no new mechanism is no longer true and the plan needs
    reopening."""
    root = _repo_root()
    svh = (root / "PLENA_Compiler" / "doc" / "operation.svh").read_text()
    codes = {int(m.group(1), 16)
             for m in re.finditer(r"\w+\s*=\s*6'h([0-9A-Fa-f]+)", svh)}
    # PLENA_RTL stops at 0x34; 0x35..0x3A predate this work (V_MAX_VF, V_MIN_VF,
    # V_TOPK, C_SET_TOPK_REG, V_SOFTPLUS_V, S_MAP_FP_V).
    added_by_this_work = {c for c in codes if c > 0x3A}
    assert added_by_this_work == {0x3B}, (
        f"opcodes past 0x3A: {sorted(hex(c) for c in added_by_this_work)}; this "
        f"work added exactly one, V_FMA_VF"
    )
