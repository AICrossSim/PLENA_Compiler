"""Loop-attached affine address-generation helpers.

The AGU pass is deliberately conservative.  It only removes an in-place
``S_ADDI_INT`` chain when that chain is the final use of the destination GP
register in one hardware-loop iteration.  Everything else remains on the
legacy scalar path.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import re
from typing import Iterable


AGU_MODE_LEGACY = "legacy"
AGU_MODE_LOOP_V1 = "loop-agu-v1"
AGU_MAX_STREAMS = 6
AGU_MANTISSA_BITS = 17
AGU_SHIFT_BITS = 5
AGU_IMMEDIATE_BITS = AGU_MANTISSA_BITS + AGU_SHIFT_BITS
AGU_IMMEDIATE_MASK = (1 << AGU_IMMEDIATE_BITS) - 1
AGU_MAX_LOOP_DEPTH = 4
AGU_REPEAT_MAX_PERIOD = 64

_GP_RE = re.compile(r"^gp([0-9]|1[0-5])$")


def encode_agu_stride(stride: int) -> int | None:
    """Return the canonical 22-bit AGU stride encoding, or ``None``.

    The low 17 bits are a signed two's-complement mantissa and the high five
    bits are a left shift.  The smallest exact shift is selected so equivalent
    strides have one stable encoding.
    """

    lower = -(1 << (AGU_MANTISSA_BITS - 1))
    upper = (1 << (AGU_MANTISSA_BITS - 1)) - 1
    for shift in range(1 << AGU_SHIFT_BITS):
        scale = 1 << shift
        if stride % scale:
            continue
        mantissa = stride // scale
        if lower <= mantissa <= upper:
            return (shift << AGU_MANTISSA_BITS) | (
                mantissa & ((1 << AGU_MANTISSA_BITS) - 1)
            )
    return None


def decode_agu_stride(encoded: int) -> int:
    """Decode one 22-bit compressed AGU stride."""

    if not 0 <= encoded <= AGU_IMMEDIATE_MASK:
        raise ValueError(f"AGU immediate must be 22-bit, got {encoded}")
    shift = encoded >> AGU_MANTISSA_BITS
    raw = encoded & ((1 << AGU_MANTISSA_BITS) - 1)
    sign = 1 << (AGU_MANTISSA_BITS - 1)
    mantissa = raw - (1 << AGU_MANTISSA_BITS) if raw & sign else raw
    return mantissa << shift


@dataclass
class AguStats:
    agu_mode: str = AGU_MODE_LEGACY
    agu_loop_count: int = 0
    agu_stream_count_histogram: Counter[int] = field(default_factory=Counter)
    agu_affine_updates_elided: int = 0
    agu_large_immediate_chunks_elided: int = 0
    dynamic_loop_end_elided: int = 0
    agu_setup_instruction_count: int = 0
    agu_projected_cycle_savings: int = 0
    agu_refolded_loop_count: int = 0
    agu_refolded_instruction_count: int = 0
    agu_residual_s_addi: int = 0
    agu_fallback_reasons: Counter[str] = field(default_factory=Counter)

    def as_dict(self) -> dict[str, object]:
        return {
            "agu_mode": self.agu_mode,
            "agu_loop_count": self.agu_loop_count,
            "agu_stream_count_histogram": {
                str(key): value
                for key, value in sorted(self.agu_stream_count_histogram.items())
            },
            "agu_affine_updates_elided": self.agu_affine_updates_elided,
            "agu_large_immediate_chunks_elided": (
                self.agu_large_immediate_chunks_elided
            ),
            "dynamic_loop_end_elided": self.dynamic_loop_end_elided,
            "agu_setup_instruction_count": self.agu_setup_instruction_count,
            "agu_projected_cycle_savings": self.agu_projected_cycle_savings,
            "agu_refolded_loop_count": self.agu_refolded_loop_count,
            "agu_refolded_instruction_count": (
                self.agu_refolded_instruction_count
            ),
            "agu_residual_s_addi": self.agu_residual_s_addi,
            "agu_fallback_reasons": dict(sorted(self.agu_fallback_reasons.items())),
        }


@dataclass
class _Instruction:
    text: str
    opcode: str
    args: tuple[str, ...]

    @classmethod
    def parse(cls, line: str) -> "_Instruction | None":
        stripped = line.strip()
        if not stripped or stripped.startswith((";", "//")):
            return None
        code = stripped.split("//", 1)[0].split(";", 1)[0].strip()
        if not code:
            return None
        opcode, separator, tail = code.partition(" ")
        args = (
            tuple(value.strip() for value in tail.split(","))
            if separator
            else ()
        )
        return cls(text=stripped, opcode=opcode, args=args)

    def mentions(self, register: str) -> bool:
        return register in self.args


@dataclass
class _Loop:
    start: _Instruction
    body: list["_Node"]
    end: _Instruction


_Node = str | _Instruction | _Loop


def _parse_program(text: str) -> list[_Node]:
    root: list[_Node] = []
    stack: list[tuple[_Instruction, list[_Node]]] = []

    def append(node: _Node) -> None:
        (stack[-1][1] if stack else root).append(node)

    for raw in text.splitlines():
        instruction = _Instruction.parse(raw)
        if instruction is None:
            append(raw)
            continue
        if instruction.opcode == "C_LOOP_START":
            stack.append((instruction, []))
            continue
        if instruction.opcode == "C_LOOP_END":
            if not stack:
                raise ValueError("unmatched C_LOOP_END in AGU optimizer")
            start, body = stack.pop()
            expected = start.args[0] if start.args else ""
            actual = instruction.args[0] if instruction.args else ""
            if expected != actual:
                raise ValueError(
                    f"loop register mismatch: start={expected!r}, end={actual!r}"
                )
            append(_Loop(start=start, body=body, end=instruction))
            continue
        append(instruction)
    if stack:
        raise ValueError("unterminated C_LOOP_START in AGU optimizer")
    return root


def _render_nodes(nodes: Iterable[_Node]) -> list[str]:
    rendered: list[str] = []
    for node in nodes:
        if isinstance(node, str):
            rendered.append(node)
        elif isinstance(node, _Instruction):
            rendered.append(node.text)
        else:
            rendered.append(node.start.text)
            rendered.extend(_render_nodes(node.body))
            rendered.append(node.end.text)
    return rendered


def _static_instruction_count(nodes: Iterable[_Node]) -> int:
    count = 0
    for node in nodes:
        if isinstance(node, _Instruction):
            count += 1
        elif isinstance(node, _Loop):
            count += 2 + _static_instruction_count(node.body)
    return count


def _loop_events(nodes: Iterable[_Node]) -> list[tuple[_Instruction, bool]]:
    """Return body instructions and whether each belongs to this loop level."""

    result: list[tuple[_Instruction, bool]] = []
    for node in nodes:
        if isinstance(node, _Instruction):
            result.append((node, True))
        elif isinstance(node, _Loop):
            result.append((node.start, False))
            result.extend(
                (instruction, False)
                for instruction, _ in _loop_events(node.body)
            )
            result.append((node.end, False))
    return result


def _parse_loop_count(start: _Instruction) -> int:
    if len(start.args) != 2:
        raise ValueError(f"invalid C_LOOP_START operands: {start.text}")
    return int(start.args[1], 0)


def _candidate_groups(loop: _Loop) -> list[tuple[str, int, list[_Instruction]]]:
    events = _loop_events(loop.body)
    flat = [instruction for instruction, _ in events]
    groups: list[tuple[str, int, list[_Instruction]]] = []
    index = 0
    while index < len(events):
        instruction, direct = events[index]
        args = instruction.args
        if not (
            direct
            and instruction.opcode == "S_ADDI_INT"
            and len(args) == 3
            and args[0] == args[1]
            and _GP_RE.match(args[0])
            and args[0] != "gp0"
        ):
            index += 1
            continue
        register = args[0]
        chain: list[_Instruction] = []
        stride = 0
        while index < len(events):
            current, current_direct = events[index]
            if not (
                current_direct
                and current.opcode == "S_ADDI_INT"
                and len(current.args) == 3
                and current.args[0] == register
                and current.args[1] == register
            ):
                break
            chain.append(current)
            stride += int(current.args[2], 0)
            index += 1
        groups.append((register, stride, chain))

    valid: list[tuple[str, int, list[_Instruction]]] = []
    for register, stride, chain in groups:
        if stride == 0 or encode_agu_stride(stride) is None:
            continue
        chain_ids = {id(item) for item in chain}
        last_chain = max(flat.index(item) for item in chain)
        if any(
            instruction.mentions(register)
            for instruction in flat[last_chain + 1 :]
        ):
            continue
        valid.append((register, stride, chain))
    return valid


_GP_DESTINATION_OPS = frozenset(
    {
        "S_ADD_INT",
        "S_ADDI_INT",
        "S_SUB_INT",
        "S_MUL_INT",
        "S_LUI_INT",
        "S_LD_INT",
    }
)


def _gp_accesses(instruction: _Instruction) -> tuple[set[str], set[str]]:
    """Return conservative GP reads and writes for one assembly instruction.

    Most PLENA operands named ``gp*`` are SRAM addresses and are therefore
    reads even when encoded in the ISA ``rd`` field. Only Scalar integer
    operations produce an architectural GP value.
    """

    gp_args = [arg for arg in instruction.args if _GP_RE.match(arg)]
    if not gp_args:
        return set(), set()
    if instruction.opcode in _GP_DESTINATION_OPS:
        writes = {instruction.args[0]} if _GP_RE.match(instruction.args[0]) else set()
        reads = {
            arg
            for arg in instruction.args[1:]
            if _GP_RE.match(arg)
        }
        return reads, writes
    return set(gp_args), set()


@dataclass(frozen=True)
class _AguCandidate:
    register: str
    stride: int
    chain: tuple[_Instruction, ...]

    @property
    def setup_instruction_count(self) -> int:
        return 1


def _instruction_signature(instruction: _Instruction) -> tuple[str, tuple[str, ...]]:
    return instruction.opcode, instruction.args


def _best_exact_repeat(
    instructions: list[_Instruction],
) -> tuple[int, int, int] | None:
    """Find the highest-benefit exact repeated microkernel.

    The returned tuple is ``(start, period, repeat_count)``.  Only a block
    whose final address updates are legal AGU streams is eligible.  This keeps
    the transformation semantic: it reconstructs a loop the compiler had
    statically unrolled, rather than treating arbitrary repeated arithmetic as
    affine address generation.
    """

    signatures = tuple(
        _instruction_signature(item) for item in instructions
    )
    removable_cache: dict[
        tuple[tuple[str, tuple[str, ...]], ...],
        tuple[int, int],
    ] = {}
    best: tuple[int, int, int] | None = None
    best_savings = 0
    instruction_count = len(instructions)
    max_global_period = min(
        AGU_REPEAT_MAX_PERIOD,
        instruction_count // 2,
    )
    for period in range(1, max_global_period + 1):
        compare_limit = instruction_count - period
        compare_index = 0
        while compare_index < compare_limit:
            if signatures[compare_index] != signatures[
                compare_index + period
            ]:
                compare_index += 1
                continue
            run_start = compare_index
            compare_index += 1
            while (
                compare_index < compare_limit
                and signatures[compare_index]
                == signatures[compare_index + period]
            ):
                compare_index += 1
            run_end = compare_index
            if run_end - run_start < period:
                continue
            # Starts separated by one full period have the same candidate
            # body and strictly fewer repetitions. Only the first start for
            # each phase can therefore improve the optimum.
            last_candidate = min(
                run_start + period,
                run_end - period + 1,
            )
            for start in range(run_start, last_candidate):
                repeat_count = 1 + (run_end - start) // period
                if repeat_count < 2:
                    continue
                block_signatures = signatures[start : start + period]
                cached = removable_cache.get(block_signatures)
                if cached is None:
                    probe = _Loop(
                        start=_Instruction(
                            text=(
                                f"C_LOOP_START gp0, {repeat_count}"
                            ),
                            opcode="C_LOOP_START",
                            args=("gp0", str(repeat_count)),
                        ),
                        body=list(
                            instructions[start : start + period]
                        ),
                        end=_Instruction(
                            text="C_LOOP_END gp0",
                            opcode="C_LOOP_END",
                            args=("gp0",),
                        ),
                    )
                    candidates = _candidate_groups(probe)[
                        :AGU_MAX_STREAMS
                    ]
                    cached = (
                        sum(
                            len(chain)
                            for _, _, chain in candidates
                        ),
                        len(candidates),
                    )
                    removable_cache[block_signatures] = cached
                removed, candidate_count = cached
                # One bind per stream plus LOOP_LEN and LOOP_START_AGU execute
                # once. The static marker is never dispatched.
                savings = (
                    repeat_count * removed
                    - (candidate_count + 2)
                )
                candidate = (start, period, repeat_count)
                if (
                    savings > best_savings
                    or (
                        savings == best_savings
                        and savings > 0
                        and best is not None
                        and candidate[:2] < best[:2]
                    )
                ):
                    best_savings = savings
                    best = candidate
    return best


def _refold_exact_repeats(
    nodes: list[_Node],
    stats: AguStats,
    *,
    depth: int = 0,
) -> list[_Node]:
    """Reconstruct exact inner loops hidden by static microkernel unrolling."""

    recursively_folded: list[_Node] = []
    for node in nodes:
        if isinstance(node, _Loop):
            node.body = _refold_exact_repeats(
                node.body,
                stats,
                depth=depth + 1,
            )
        recursively_folded.append(node)
    if depth >= AGU_MAX_LOOP_DEPTH:
        return recursively_folded

    result: list[_Node] = []
    index = 0
    while index < len(recursively_folded):
        if not isinstance(recursively_folded[index], _Instruction):
            result.append(recursively_folded[index])
            index += 1
            continue
        end = index
        while end < len(recursively_folded) and isinstance(
            recursively_folded[end], _Instruction
        ):
            end += 1
        run = [
            item
            for item in recursively_folded[index:end]
            if isinstance(item, _Instruction)
        ]
        candidate = _best_exact_repeat(run)
        if candidate is None:
            result.extend(run)
            index = end
            continue
        start, period, repeat_count = candidate
        result.extend(
            _refold_exact_repeats(
                list(run[:start]),
                stats,
                depth=depth,
            )
        )
        body = list(run[start : start + period])
        result.append(
            _Loop(
                start=_Instruction(
                    text=f"C_LOOP_START gp0, {repeat_count}",
                    opcode="C_LOOP_START",
                    args=("gp0", str(repeat_count)),
                ),
                body=body,
                end=_Instruction(
                    text="C_LOOP_END gp0",
                    opcode="C_LOOP_END",
                    args=("gp0",),
                ),
            )
        )
        stats.agu_refolded_loop_count += 1
        stats.agu_refolded_instruction_count += (
            repeat_count * period - period
        )
        result.extend(
            _refold_exact_repeats(
                list(run[start + repeat_count * period :]),
                stats,
                depth=depth,
            )
        )
        index = end
    return result


def _remove_instruction_ids(nodes: list[_Node], removed: set[int]) -> list[_Node]:
    result: list[_Node] = []
    for node in nodes:
        if isinstance(node, _Instruction) and id(node) in removed:
            continue
        if isinstance(node, _Loop):
            node.body = _remove_instruction_ids(node.body, removed)
        result.append(node)
    return result


def _optimize_nodes(
    nodes: list[_Node],
    stats: AguStats,
    *,
    mode: str,
) -> list[_Node]:
    result: list[_Node] = []
    for node in nodes:
        if not isinstance(node, _Loop):
            result.append(node)
            continue
        node.body = _optimize_nodes(node.body, stats, mode=mode)
        count = _parse_loop_count(node.start)
        boundary_candidates = [
            _AguCandidate(
                register=register,
                stride=stride,
                chain=tuple(chain),
            )
            for register, stride, chain in _candidate_groups(node)
        ]
        candidates = sorted(
            boundary_candidates,
            key=lambda candidate: (
                -(
                    count * len(candidate.chain)
                    - candidate.setup_instruction_count
                ),
                candidate.register,
            ),
        )[:AGU_MAX_STREAMS]
        removed_per_iteration = sum(len(item.chain) for item in candidates)
        setup = sum(item.setup_instruction_count for item in candidates) + 1
        projected = count * (removed_per_iteration + 1) - setup
        if projected <= 0:
            stats.agu_fallback_reasons["not_profitable"] += 1
            result.append(node)
            continue

        removed = {id(item) for candidate in candidates for item in candidate.chain}
        node.body = _remove_instruction_ids(node.body, removed)
        body_words = _static_instruction_count(node.body)
        if not 0 < body_words < (1 << AGU_IMMEDIATE_BITS):
            stats.agu_fallback_reasons["body_length_out_of_range"] += 1
            result.append(node)
            continue

        for candidate in candidates:
            encoded = encode_agu_stride(candidate.stride)
            assert encoded is not None
            result.append(
                _Instruction(
                    text=f"C_AGU_BIND {candidate.register}, {encoded}",
                    opcode="C_AGU_BIND",
                    args=(candidate.register, str(encoded)),
                )
            )
            stats.agu_affine_updates_elided += count
            stats.agu_large_immediate_chunks_elided += count * (
                len(candidate.chain) - 1
            )
        result.append(
            _Instruction(
                text=f"C_AGU_LOOP_LEN {body_words}",
                opcode="C_AGU_LOOP_LEN",
                args=(str(body_words),),
            )
        )
        node.start = _Instruction(
            text=f"C_LOOP_START_AGU {node.start.args[0]}, {count}",
            opcode="C_LOOP_START_AGU",
            args=(node.start.args[0], str(count)),
        )
        result.append(node)
        stats.agu_loop_count += 1
        stats.agu_stream_count_histogram[len(candidates)] += 1
        stats.dynamic_loop_end_elided += count
        stats.agu_setup_instruction_count += setup
        stats.agu_projected_cycle_savings += projected
    return result


def optimize_agu_assembly(
    text: str,
    *,
    mode: str = AGU_MODE_LOOP_V1,
) -> tuple[str, dict[str, object]]:
    """Optimize one rendered assembly program for loop-attached AGU execution."""

    if mode not in {AGU_MODE_LEGACY, AGU_MODE_LOOP_V1}:
        raise ValueError(f"unsupported address_generation_mode={mode!r}")
    stats = AguStats(agu_mode=mode)
    if mode == AGU_MODE_LEGACY:
        return text, stats.as_dict()
    nodes = _parse_program(text)
    nodes = _refold_exact_repeats(nodes, stats)
    nodes = _optimize_nodes(nodes, stats, mode=mode)
    stats.agu_residual_s_addi = sum(
        line.strip().startswith("S_ADDI_INT")
        for line in _render_nodes(nodes)
    )
    lines = _render_nodes(nodes)
    return "\n".join(lines) + ("\n" if text.endswith("\n") else ""), stats.as_dict()


__all__ = [
    "AGU_MAX_STREAMS",
    "AGU_MODE_LEGACY",
    "AGU_MODE_LOOP_V1",
    "decode_agu_stride",
    "encode_agu_stride",
    "optimize_agu_assembly",
]
