from __future__ import annotations

import math
from collections import Counter

import pytest

from compiler.aten.cost_emitter import (
    CompositeSink,
    CostSink,
    RawAsmCostError,
    ScheduleRepeat,
)
from compiler.aten.isa_builder import DmaTransfer, IsaBuilder, RepeatAxis, gp
from compiler.aten.plena import PlenaCompiler


def _asm_histogram(asm: str) -> tuple[Counter[str], Counter[str]]:
    static = Counter()
    dynamic = Counter()
    loop_stack: list[int] = []
    for raw_line in asm.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(";"):
            continue
        opcode = line.split(maxsplit=1)[0]
        multiplier = math.prod(loop_stack)
        static[opcode] += 1
        dynamic[opcode] += multiplier
        if opcode == "C_LOOP_START":
            loop_stack.append(int(line.rsplit(",", 1)[1]))
        elif opcode == "C_LOOP_END":
            loop_stack.pop()
    assert not loop_stack
    return static, dynamic


def test_symbolic_render_preserves_loop_syntax() -> None:
    body = IsaBuilder().instr("M_MM", gp(1), gp(2))
    asm = IsaBuilder().comment("loop").hardware_loop(gp(3), 4, body)

    assert asm.render() == (
        "; loop\n"
        "C_LOOP_START gp3, 4\n"
        "M_MM gp1, gp2\n"
        "C_LOOP_END gp3\n"
    )


def test_cost_sink_counts_compile_time_and_hardware_repeats() -> None:
    loop_body = IsaBuilder().instr("M_MM", gp(1), gp(2))
    repeated = IsaBuilder().instr("V_ADD_VV", gp(1), gp(2), gp(3), 0)
    repeated.hardware_loop(gp(4), 3, loop_body)
    program = IsaBuilder().instr("S_ADD_INT", gp(1), gp(2), gp(3)).repeat(2, repeated)

    sink = CostSink()
    sink.emit(program)
    trace = sink.trace

    assert trace.static_opcodes == {
        "S_ADD_INT": 1,
        "V_ADD_VV": 2,
        "C_LOOP_START": 2,
        "M_MM": 2,
        "C_LOOP_END": 2,
    }
    assert trace.dynamic_opcodes == {
        "S_ADD_INT": 1,
        "V_ADD_VV": 2,
        "C_LOOP_START": 2,
        "M_MM": 6,
        "C_LOOP_END": 6,
    }


def test_large_immediate_is_legalized_before_costing() -> None:
    sink = CostSink()
    sink.emit(IsaBuilder().instr("S_ADDI_INT", gp(5), gp(0), 300_000))

    assert sink.trace.static_opcodes == {"S_LUI_INT": 1, "S_ADDI_INT": 1}
    assert sink.trace.dynamic_opcodes == sink.trace.static_opcodes


def test_dma_metadata_inherits_stage_and_repeat() -> None:
    transfer = DmaTransfer(
        opcode="H_PREFETCH_M",
        direction="read",
        precision="weight",
        element_base=4096,
        scale_base=8192,
        dim=128,
        amount=128,
        stride=128,
    )
    body = IsaBuilder().dma_instr("H_PREFETCH_M", gp(1), gp(2), gp(3), 0, 0, dma=transfer)
    program = IsaBuilder().stage(
        "q_projection",
        IsaBuilder().repeat(
            8,
            body,
            axis=RepeatAxis("column", 8, element_base_delta=16_384, scale_base_delta=2_048),
        ),
    )

    sink = CostSink()
    sink.emit(program)

    assert sink.trace.dynamic_opcodes["H_PREFETCH_M"] == 8
    assert len(sink.trace.memory_events) == 1
    event = sink.trace.memory_events[0]
    assert event.stage == "q_projection"
    assert event.multiplicity == 8
    assert event.enclosing_axes[0].name == "column"
    assert event.stream_index == 0
    serialized = sink.trace.to_dict()
    assert serialized["schema_version"] == 4
    assert serialized["schedule_fidelity"] == "ordered_compressed"
    assert serialized["compressed_memory_events"][0]["geometry_fidelity"] == "exact"
    assert serialized["compressed_memory_events"][0]["stream_instruction_count"] == 8
    assert serialized["compressed_memory_events"][0]["precision_role"] == "weight"
    assert serialized["compressed_memory_events"][0]["memory_object"] == "anonymous"
    assert serialized["compressed_memory_events"][0]["logical_stride"] == 128
    assert serialized["compressed_memory_events"][0]["repeat_axes"][0][
        "logical_element_delta"
    ] == 16_384


def test_dma_event_can_bind_multiple_ordered_schedule_sites() -> None:
    """One compressed stream may cover several statically unrolled sites."""
    instruction = IsaBuilder().instr(
        "H_PREFETCH_M", gp(1), gp(2), "a1", 1, 0
    )
    program = IsaBuilder().stage(
        "layer/ffn",
        IsaBuilder()
        .repeat(3, instruction, axis=RepeatAxis("k_tile_a", 3))
        .repeat(3, instruction, axis=RepeatAxis("k_tile_b", 3)),
    )
    transfer = DmaTransfer(
        opcode="H_PREFETCH_M",
        direction="read",
        precision="weight",
        element_base=0,
        scale_base=4_096,
        dim=64,
        amount=64,
        stride=64,
    )
    sink = CostSink()
    sink.emit(program)
    stream_index = sink.add_memory_event(
        transfer=transfer,
        multiplicity=6,
        stage="layer/ffn",
        axes=(RepeatAxis("all_k_tiles", 6),),
    )

    trace = sink.finish()
    first, second = trace.schedule.children
    assert isinstance(first, ScheduleRepeat)
    assert isinstance(second, ScheduleRepeat)
    assert first.body.children[0].memory_stream_index == stream_index
    assert second.body.children[0].memory_stream_index == stream_index
    assert trace.dynamic_opcodes["H_PREFETCH_M"] == 6


def test_cost_sink_rejects_incomplete_dma_repeat_axes() -> None:
    sink = CostSink()
    transfer = DmaTransfer(
        opcode="H_PREFETCH_V",
        direction="read",
        precision="activation",
        element_base=0,
        scale_base=4096,
        dim=64,
        amount=64,
        stride=64,
    )

    with pytest.raises(ValueError, match="multiply to 4, expected 8"):
        sink.add_memory_event(
            transfer=transfer,
            multiplicity=8,
            axes=(RepeatAxis("batch", 4),),
        )


def test_cost_sink_rejects_raw_instruction_text() -> None:
    with pytest.raises(RawAsmCostError, match="unstructured ASM"):
        CostSink().emit(IsaBuilder().raw("M_MM gp1, gp2"))


def test_compatibility_raw_parser_counts_hardware_loops() -> None:
    sink = CostSink(strict_raw=False)
    sink.emit(
        IsaBuilder().raw(
            "; old template\n"
            "C_LOOP_START gp1, 3\n"
            "M_MM gp2, gp3\n"
            "C_LOOP_END gp1\n"
        )
    )

    assert sink.finish().static_opcodes == {"C_LOOP_START": 1, "M_MM": 1, "C_LOOP_END": 1}
    assert sink.trace.dynamic_opcodes == {"C_LOOP_START": 1, "M_MM": 3, "C_LOOP_END": 3}


def test_composite_sink_renders_and_counts_same_program() -> None:
    program = IsaBuilder().instr("S_ADD_FP", "f1", "f2", "f3")
    sink = CompositeSink()

    rendered = sink.emit(program)

    assert rendered == "S_ADD_FP f1, f2, f3\n"
    assert sink.asm.getvalue() == rendered
    assert sink.cost.trace.dynamic_opcodes == {"S_ADD_FP": 1}


def test_schedule_ir_preserves_nested_repeat_order() -> None:
    body = IsaBuilder().instr("V_ADD_VV", gp(1), gp(2), gp(3), 0)
    program = IsaBuilder().repeat(3, body, axis=RepeatAxis("vector", 3))
    sink = CostSink()
    sink.emit(program)

    serialized = sink.finish().to_dict()["compressed_schedule"]
    repeat = serialized["children"][0]
    assert repeat["type"] == "repeat"
    assert repeat["name"] == "vector"
    assert repeat["count"] == 3
    assert repeat["body"]["children"][0]["opcode"] == "V_ADD_VV"


def test_schedule_ir_recovers_typed_hardware_loop_markers() -> None:
    program = IsaBuilder()
    program.instr("C_LOOP_START", gp(7), 3)
    program.instr("V_ADD_VV", gp(1), gp(2), gp(3), 0)
    program.instr("C_LOOP_END", gp(7))
    sink = CostSink()
    sink.emit(program)

    trace = sink.finish()
    serialized = trace.to_dict()
    assert serialized["schedule_fidelity"] == "ordered_compressed"
    assert trace.dynamic_opcodes == {
        "C_LOOP_START": 1,
        "V_ADD_VV": 3,
        "C_LOOP_END": 3,
    }
    start, repeat = serialized["compressed_schedule"]["children"]
    assert start["opcode"] == "C_LOOP_START"
    assert repeat["type"] == "repeat"
    assert repeat["repeat_kind"] == "hardware_loop"
    assert repeat["count"] == 3
    assert [child["opcode"] for child in repeat["body"]["children"]] == [
        "V_ADD_VV",
        "C_LOOP_END",
    ]


def test_counts_only_summary_marks_schedule_unavailable() -> None:
    sink = CostSink()
    sink.add_counts(
        static_opcodes={"M_MM": 1},
        dynamic_opcodes={"M_MM": 64},
        stage="layer/ffn",
    )

    trace = sink.finish().to_dict()
    assert trace["schedule_fidelity"] == "unavailable"
    assert trace["schedule_unavailable_reasons"] == {
        "counts_only_kernel_summary": 1
    }
    assert trace["schedule_coverage"] == {
        "ordered_dynamic_instructions": 0,
        "unavailable_dynamic_instructions": 64,
        "ordered_fraction": 0.0,
        "unavailable_by_reason": {"counts_only_kernel_summary": 64},
    }


def test_both_mode_preserves_regular_asm_and_matches_asm_histogram() -> None:
    def lower(emission_mode: str) -> PlenaCompiler:
        compiler = PlenaCompiler(
            mlen=64,
            blen=64,
            hbm_v_prefetch_amount=64,
            hbm_v_writeback_amount=64,
            emission_mode=emission_mode,
            cost_strict_raw=True,
        )
        source = compiler.input("source", (64, 64), hbm_addr=300_000)
        loaded = compiler.load_batch(source)
        compiler.store(loaded, name="stored")
        return compiler

    asm_only = lower("asm").compile()
    both = lower("both")
    both_asm = both.compile()
    trace = both.compile_cost_trace()
    static, dynamic = _asm_histogram(both_asm)

    assert both_asm == asm_only
    assert trace.static_opcodes == static
    assert trace.dynamic_opcodes == dynamic
