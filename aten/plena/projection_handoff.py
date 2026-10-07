"""Static SRAM retention of produced rows using only ordinary Vector copies.

This is a Compiler lifetime pass, not a hardware cache. HBM stores remain in
the program; matching later reads may use explicitly retained SRAM payloads.
There is no host tensor inspection and no zero-cost data forwarding.
"""

from bisect import bisect_right
from collections import defaultdict
import re

from compiler.asm_templates._imm import load_large_int
from compiler.aten.plena.mview import MatrixViewShape


def retain_vector_handoffs(assembly, *, first_row=48, slots=10, projection_stages=()):
    """Return (assembly, report) for a full program with initially zero GP regs.

    Caller must reserve ``[first_row, first_row+slots)`` in every producer and
    consumer. All explicit Vector accesses are checked, as are M_MM.P's
    strided requests. L_TILE's inherited BF16 tree occupies rows 0..7.
    Loops are interpreted for address checking but remain textually unchanged;
    retention is disabled inside loops and invalidated across either boundary.
    Unsupported instructions fail closed. Partial HBM writes invalidate every
    overlapping retained row. The pass never removes a store, changes layout,
    or substitutes a host-computed value.

    Optional ``projection_stages`` identifies exact @operator/ operator names.
    During those stages retention is disabled and the pool belongs to the
    projection input cache. Payloads are invalidated at either phase boundary;
    no SRAM capacity is simultaneously assigned to both uses. All names must
    appear in the marked program, otherwise the transformation is rejected.
    """
    if (
        type(first_row) is not int
        or type(slots) is not int
        or not (8 <= first_row < first_row + slots <= 64)
    ):
        raise ValueError(
            "handoff rows must be disjoint from tree rows 0..7 and fit Vector SRAM"
        )
    lines = assembly.splitlines()
    projection_stages = frozenset(projection_stages)
    if any(not isinstance(name, str) or not name for name in projection_stages):
        raise ValueError("projection stage names must be nonempty strings")
    projection_lines, phase_barriers, markers, seen = [], set(), set(), set()
    projecting = False
    for index, raw in enumerate(lines):
        marker = re.fullmatch(r"\s*;\s*@?operator=([^\s]+)\s*", raw)
        if marker:
            markers.add(index)
            name = marker.group(1)
            seen.add(name)
            next_projection = name in projection_stages
            if next_projection != projecting:
                phase_barriers.add(index)
            projecting = next_projection
        projection_lines.append(projecting)
    if not projection_stages <= seen:
        raise ValueError("projection stage names missing from operator markers")
    parsed = []
    for raw in lines:
        clean = raw.split(";", 1)[0].split("//", 1)[0].strip()
        parts = clean.split(maxsplit=1)
        parsed.append(
            (parts[0], [x.strip() for x in parts[1].split(",")])
            if len(parts) == 2
            else (clean, [])
        )
    stack, ends = [], {}
    for index, (op, args) in enumerate(parsed):
        if index in markers and stack:
            raise ValueError("operator boundaries must be outside static loops")
        if op == "C_LOOP_START":
            if len(args) != 2 or int(args[1], 0) < 1:
                raise ValueError("handoff needs bounded positive static loops")
            stack.append(index)
        elif op == "C_LOOP_END":
            if not stack:
                raise ValueError("unmatched C_LOOP_END")
            ends[stack.pop()] = index
    if stack:
        raise ValueError("unterminated C_LOOP_START")

    gp, views = [0] * 16, {}
    events = {index: ("projection_barrier",) for index in phase_barriers}
    reads = defaultdict(list)
    examined = 0
    in_projection = False
    low, high = first_row * 2048, (first_row + slots) * 2048

    def reg(text):
        if (
            not text.startswith("gp")
            or not text[2:].isdigit()
            or not 0 <= int(text[2:]) < 16
        ):
            raise ValueError(f"expected GP register, got {text}")
        return int(text[2:])

    def value(text):
        return gp[reg(text)]

    def check(base, size=2048):
        if base < 0 or base + size > 64 * 2048:
            raise ValueError("Vector access exceeds SRAM capacity")
        if not in_projection and base < high and low < base + size:
            raise ValueError("program aliases reserved handoff rows")

    def row(text):
        base = value(text)
        if base % 2048:
            raise ValueError("ordinary Vector operation requires row alignment")
        check(base)

    def analyze(begin, end, depth=0):
        nonlocal examined, in_projection
        index = begin
        while index < end:
            examined += 1
            if examined > 5_000_000:
                raise ValueError(
                    "handoff address analysis exceeds bounded instruction budget"
                )
            op, args = parsed[index]
            in_projection = projection_lines[index]
            if not op:
                index += 1
                continue
            if op == "C_LOOP_START":
                finish = ends[index]
                counter = reg(args[0])
                if counter == 0 or parsed[finish][1] != [args[0]]:
                    raise ValueError("handoff requires a matched nonzero loop register")
                if depth == 0 and not in_projection:
                    events[index] = ("barrier",)
                    events[finish] = ("barrier",)
                for remaining in range(int(args[1], 0), 0, -1):
                    gp[counter] = remaining
                    analyze(index + 1, finish, depth + 1)
                    if gp[counter] != remaining:
                        raise ValueError(
                            "handoff cannot analyze a body that changes its loop counter"
                        )
                gp[counter] = 0
                index = finish + 1
                continue
            if op == "S_ADDI_INT":
                gp[reg(args[0])] = (value(args[1]) + int(args[2], 0)) & 0xFFFFFFFF
            elif op == "S_LUI_INT":
                gp[reg(args[0])] = (int(args[1], 0) << 12) & 0xFFFFFFFF
            elif op in ("S_ADD_INT", "S_SUB_INT", "S_MUL_INT"):
                a, b = value(args[1]), value(args[2])
                gp[reg(args[0])] = {
                    "S_ADD_INT": lambda: a + b,
                    "S_SUB_INT": lambda: a - b,
                    "S_MUL_INT": lambda: a * b,
                }[op]() & 0xFFFFFFFF
            elif op == "L_TILE_CFG":
                views[int(args[0], 0)] = MatrixViewShape.unpack(value(args[1]))
            elif op in ("H_PREFETCH_V", "H_STORE_V"):
                if len(args) != 5 or args[2:] != ["a0", "0", "2"]:
                    raise ValueError(
                        "handoff supports contiguous BF16 Vector DMA with a0=0"
                    )
                row(args[0])
                if depth == 0 and not in_projection:
                    address = value(args[1])
                    event = (
                        "read" if op == "H_PREFETCH_V" else "store",
                        address,
                        4096,
                        args[0],
                        args[1],
                        tuple(gp),
                    )
                    events[index] = event
                    if op == "H_PREFETCH_V":
                        reads[address].append(index)
            elif op in ("H_PREFETCH_V.MV", "H_STORE_V.MV"):
                if len(args) != 6 or args[2:5] != ["a0", "0", "2"]:
                    raise ValueError("handoff supports contiguous BF16 Matrix-view DMA")
                shape = views[int(args[5], 0)]
                if op == "H_STORE_V.MV" and depth == 0 and not in_projection:
                    events[index] = (
                        "invalidate",
                        value(args[1]),
                        shape.rows * shape.cols * shape.tile_count * 2,
                    )
            elif op == "M_MM.P":
                cfg = value(args[3])
                if cfg >> 18:
                    raise ValueError("reserved projection configuration bits")
                shape = views[int(args[4], 0)]
                for request in range((cfg & 3) + 1):
                    check(value(args[0]) + request * ((cfg >> 10) & 255) * 32, 32)
                    check(
                        value(args[2]) + request * ((cfg >> 2) & 255) * 256, shape.rows
                    )
            elif op == "M_MV":
                row(args[2])
            elif op == "M_MV_WO":
                check(value(args[0]) + int(args[1], 0), 32)
            elif op in ("V_RED_SUM", "V_RED_MAX"):
                row(args[1])
            elif op in (
                "V_SHFT_V",
                "V_ADD_VF",
                "V_SUB_VF",
                "V_MUL_VF",
                "V_FMA_VF",
                "V_EXP_V",
                "V_RECI_V",
                "V_SOFTPLUS_V",
                "V_MAX_VF",
                "V_MIN_VF",
            ):
                row(args[0])
                row(args[1])
            elif op in ("V_ADD_VV", "V_SUB_VV", "V_MUL_VV"):
                row(args[0])
                row(args[1])
                row(args[2])
            elif op in (
                "L_TILE_CCFG",
                "L_TILE_EXEC",
                "S_SUB_FP",
                "S_ADD_FP",
                "S_MUL_FP",
                "S_LD_FP",
                "S_ST_FP",
                "S_SQRT_FP",
                "S_RECI_FP",
                "S_EXP_FP",
                "S_MV_FP",
            ):
                pass
            else:
                raise ValueError(f"handoff cannot prove effects of {op}")
            if gp[0] != 0:
                raise ValueError("handoff requires immutable zero gp0")
            index += 1

    analyze(0, len(lines))
    live, output = {}, []
    report = dict(
        reserved_first_row=first_row,
        reserved_rows=slots,
        reserved_bytes=slots * 4096,
        retains=0,
        bypassed_reads=0,
        copy_instructions=0,
        hbm_read_bytes_saved=0,
        hbm_write_bytes_saved=0,
        peak_live_rows=0,
        loop_barriers=0,
        projection_phase_barriers=0,
        projection_stages=sorted(projection_stages),
        simultaneous_projection_reserved_bytes=0 if projection_stages else slots * 4096,
        semantics="static producer-row retention; stores preserved; ordinary Vector copies",
    )
    barriers = sorted(
        index
        for index, event in events.items()
        if event[0] in ("barrier", "projection_barrier")
    )

    def next_read(address, index):
        positions = reads.get(address, ())
        where = bisect_right(positions, index)
        result = positions[where] if where < len(positions) else float("inf")
        boundary = bisect_right(barriers, index)
        if boundary < len(barriers) and barriers[boundary] < result:
            return float("inf")
        return result

    def copy(event, slot, retain):
        _, _, _, destination, hbmreg, registers = event
        forbidden = reg(destination)
        scratch = (
            reg(hbmreg)
            if reg(hbmreg) != forbidden and reg(hbmreg) != 0
            else next(r for r in range(1, 16) if r != forbidden)
        )
        instructions = load_large_int(scratch, slot * 2048)
        instructions.append(
            f"V_SHFT_V gp{scratch}, {destination}, gp0"
            if retain
            else f"V_SHFT_V {destination}, gp{scratch}, gp0"
        )
        instructions.extend(load_large_int(scratch, registers[scratch]))
        report["copy_instructions"] += 1
        return instructions

    for index, raw in enumerate(lines):
        event = events.get(index)
        if event is None:
            output.append(raw)
            continue
        kind = event[0]
        if kind in ("barrier", "projection_barrier"):
            live.clear()
            report[
                "loop_barriers" if kind == "barrier" else "projection_phase_barriers"
            ] += 1
            output.append(raw)
            continue
        _, address, count, *_ = event
        if kind in ("store", "invalidate"):
            live = {
                a: slot
                for a, slot in live.items()
                if not (a < address + count and address < a + 4096)
            }
            output.append(raw)
            if kind != "store" or next_read(address, index) == float("inf"):
                continue
            free = sorted(set(range(first_row, first_row + slots)) - set(live.values()))
            if free:
                slot = free[0]
            else:
                victim = max(live, key=lambda a: (next_read(a, index), a))
                if next_read(address, index) >= next_read(victim, index):
                    continue
                slot = live.pop(victim)
            live[address] = slot
            output.extend(copy(event, slot, True))
            report["retains"] += 1
            report["peak_live_rows"] = max(report["peak_live_rows"], len(live))
        elif address in live:
            output.extend(copy(event, live[address], False))
            report["bypassed_reads"] += 1
            report["hbm_read_bytes_saved"] += 4096
        else:
            output.append(raw)
    return "\n".join(output) + "\n", report
