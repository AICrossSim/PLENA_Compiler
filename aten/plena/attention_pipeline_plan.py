"""Timing-aware instruction ordering for packed-GQA row microkernels.

The planner is intentionally compiler-only: it does not estimate end-to-end
latency and it does not replace the CostEmitter scoreboard.  It uses the
measured RTL-v3 ready latency and initiation interval only to choose an issue
order among independent rows.  Instructions belonging to one row remain in
their original order, which preserves the arithmetic sequence and makes the
    optimized lowering suitable for bitwise A/B checks.  RTL-v4 extends the
    same artifact schema with compact-stat operations, so the scheduling subset
    remains backward compatible.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_GQA_TIMING_ENV = "PLENA_GQA_TIMING_CALIBRATION"


def _default_timing_candidates() -> tuple[Path, ...]:
    source = Path(__file__).resolve()
    return (
        Path.cwd() / "transactional_emulator/calibration/rtl_opcode_timing_v4.json",
        Path.cwd() / "transactional_emulator/calibration/rtl_opcode_timing_v3.json",
        source.parents[3]
        / "transactional_emulator/calibration/rtl_opcode_timing_v4.json",
        source.parents[3]
        / "transactional_emulator/calibration/rtl_opcode_timing_v3.json",
    )


@dataclass(frozen=True)
class GQATimingProfile:
    """The small timing subset needed to order independent row operations."""

    path: Path
    sha256: str
    rob_depth: int
    fp_register_count: int
    vector_ii: int
    vector_shift_ii: int
    scalar_ready: dict[str, int]
    scalar_ii: dict[str, int]
    vector_ready: dict[str, int]

    @classmethod
    def load(cls, path: str | Path | None = None) -> "GQATimingProfile":
        requested = path or os.environ.get(DEFAULT_GQA_TIMING_ENV)
        candidates = (Path(requested),) if requested else _default_timing_candidates()
        selected = next((candidate for candidate in candidates if candidate.is_file()), None)
        if selected is None:
            searched = ", ".join(str(candidate) for candidate in candidates)
            raise FileNotFoundError(
                "row-interleaved-v1 requires an RTL-v3-or-later timing artifact; "
                f"searched: {searched}. Set {DEFAULT_GQA_TIMING_ENV} or pass "
                "gqa_timing_calibration explicitly."
            )

        raw_bytes = selected.read_bytes()
        data = json.loads(raw_bytes)
        schema_version = int(data.get("schema_version", 0))
        model = str(data.get("model", ""))
        if schema_version < 3 or not any(
            marker in model for marker in ("rtl_v3", "rtl_v4")
        ):
            raise ValueError(
                "GQA pipeline scheduling requires a compatible RTL-v3-or-later "
                f"artifact, got {selected}"
            )
        vector = data["vector"]
        scalar = data["scalar"]
        scalar_ready = {
            "load": int(scalar.get("sram_cycles", 1)),
            "move": int(scalar["fp_move_ready_cycles"]),
            "add": int(scalar["fp_add_ready_cycles"]),
            "sub": int(scalar["fp_sub_ready_cycles"]),
            "max": int(scalar["fp_max_ready_cycles"]),
            "mul": int(scalar["fp_mul_ready_cycles"]),
            "exp": int(scalar["fp_exp_ready_cycles"]),
            "reciprocal": int(scalar["fp_reciprocal_ready_cycles"]),
            "store": int(scalar.get("sram_cycles", 1)),
        }
        scalar_ii = {
            name: int(scalar.get(f"fp_{name}_initiation_interval_cycles", 1))
            for name in ("move", "add", "sub", "max", "mul", "exp", "reciprocal")
        }
        scalar_ii.update({"load": 1, "store": 1})
        vector_ready = {
            "mul_vf": int(vector["mul_vf_cycles"]),
            "sub_vf": int(vector["sub_vf_cycles"]),
            "exp": int(vector["exp_cycles"]),
            "add_vv": int(vector["add_vv_cycles"]),
            "shift": int(vector.get("shift_conservative_cycles", 13)),
            # Single-segment reduction latency depends on segment width.  The
            # compiler supplies the exact value when constructing each op.
            "reduction": 1,
        }
        return cls(
            path=selected.resolve(),
            sha256=sha256(raw_bytes).hexdigest(),
            rob_depth=int(scalar["rob_depth"]),
            fp_register_count=int(scalar["register_count"]),
            vector_ii=int(vector["initiation_interval_cycles"]),
            vector_shift_ii=int(vector.get("shift_initiation_interval_cycles", 1)),
            scalar_ready=scalar_ready,
            scalar_ii=scalar_ii,
            vector_ready=vector_ready,
        )

    def reduction_latency(self, *, kind: str, segment_width: int) -> int:
        if segment_width <= 0:
            raise ValueError(f"segment_width must be positive, got {segment_width}")
        levels = max(0, (segment_width - 1).bit_length())
        if kind == "sum":
            return int(self.vector_ready_cycles("reduce_sum", levels))
        if kind == "max":
            return int(self.vector_ready_cycles("reduce_max", levels))
        raise ValueError(f"unsupported reduction kind {kind!r}")

    def vector_ready_cycles(self, opcode: str, levels: int = 0) -> int:
        data = json.loads(self.path.read_text())["vector"]
        if opcode == "reduce_sum":
            return int(data["reduce_sum_base_cycles"]) + levels * int(
                data["reduce_sum_per_level_cycles"]
            )
        if opcode == "reduce_max":
            return int(data["reduce_max_base_cycles"]) + levels * int(
                data["reduce_max_per_level_cycles"]
            )
        return self.vector_ready[opcode]


@dataclass(frozen=True)
class RowPipelineOp:
    """One instruction in a row-local dependency chain."""

    text: str
    resource: str
    latency: int
    initiation_interval: int = 1
    is_blocking_reduction: bool = False


def interleave_row_chains(
    chains: Sequence[Sequence[RowPipelineOp]],
) -> tuple[str, ...]:
    """Deterministically interleave independent ordered row chains.

    This is a compact list scheduler, not a hardware simulator.  Each row has
    one conservative dependency-ready timestamp.  Resource availability is
    used to avoid issuing a blocking reduction while another row has ready
    scalar work.  The exact stalls and ROB retirement remain the responsibility
    of the rtl-v3 scoreboard used by the emulator and CostEmitter.
    """

    if not chains:
        return ()
    positions = [0] * len(chains)
    row_ready = [0] * len(chains)
    resource_ready: dict[str, int] = {}
    issue_cycle = 0
    output: list[str] = []

    while True:
        candidates: list[tuple[int, int, int, int, RowPipelineOp]] = []
        for row, chain in enumerate(chains):
            pos = positions[row]
            if pos >= len(chain):
                continue
            op = chain[pos]
            earliest = max(
                issue_cycle,
                row_ready[row],
                resource_ready.get(op.resource, 0),
                resource_ready.get("vector_block", 0)
                if op.resource.startswith("vector")
                else 0,
            )
            # At equal readiness prefer scalar/control work, then the oldest
            # row. This prevents a ready scalar continuation from sitting
            # behind a new long, single-segment reduction.
            blocking_rank = 1 if op.is_blocking_reduction else 0
            candidates.append((earliest, blocking_rank, pos, row, op))
        if not candidates:
            break

        earliest, _blocking_rank, _position, row, op = min(
            candidates, key=lambda item: item[:4]
        )
        issue_cycle = earliest
        output.append(op.text)
        positions[row] += 1
        row_ready[row] = issue_cycle + max(1, op.latency)
        resource_ready[op.resource] = issue_cycle + max(1, op.initiation_interval)
        if op.is_blocking_reduction:
            resource_ready["vector_block"] = issue_cycle + max(1, op.latency)
        issue_cycle += 1

    return tuple(output)


def arithmetic_opcodes(lines: Iterable[str]) -> tuple[str, ...]:
    """Return arithmetic opcodes for parity diagnostics."""

    prefixes = ("V_", "S_ADD_FP", "S_SUB_FP", "S_MUL_FP", "S_MAX_FP", "S_EXP_FP", "S_RECI_FP")
    return tuple(
        line.split()[0]
        for line in lines
        if line and not line.startswith(";") and line.startswith(prefixes)
    )


__all__ = [
    "DEFAULT_GQA_TIMING_ENV",
    "GQATimingProfile",
    "RowPipelineOp",
    "arithmetic_opcodes",
    "interleave_row_chains",
]
