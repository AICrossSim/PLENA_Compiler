from collections import Counter

import pytest

from compiler.aten.isa_builder import (
    ActiveDimensions,
    DmaTransfer,
    Instr,
    IsaBuilder,
    RepeatAxis,
    SramActivity,
    gp,
)
from compiler.aten.program_sink import AsmSink, SymbolicCostSink


def test_asm_sink_matches_builder_render_byte_for_byte():
    body = IsaBuilder().instr("V_ADD_VV", gp(1), gp(2), gp(3), 0)
    builder = (
        IsaBuilder()
        .comment("fixture")
        .stage("decoder/attention", IsaBuilder().hardware_loop(gp(7), 3, body))
    )

    sink = AsmSink()
    sink.consume(builder.finalized())

    assert sink.render() == builder.render()


def test_symbolic_sink_counts_hardware_loops_without_expansion():
    body = IsaBuilder().instr(
        "V_MUL_VV",
        gp(1),
        gp(2),
        gp(3),
        0,
        active=ActiveDimensions(lanes=32, total_lanes=64),
        sram=(SramActivity("vector", "read", accesses=2),),
    )
    schedule = IsaBuilder().stage(
        "decoder/ffn",
        IsaBuilder().hardware_loop(
            gp(6),
            1_000_000,
            body,
            axis=RepeatAxis.from_mapping("row", 1_000_000, {"vram": 64}),
        ),
    )
    sink = SymbolicCostSink(compiler_hash="fixture")
    sink.consume(schedule.finalized())
    trace = sink.finish()

    assert trace.dynamic_opcode_counts == Counter(
        {"C_LOOP_START": 1, "V_MUL_VV": 1_000_000, "C_LOOP_END": 1_000_000}
    )
    assert trace.metadata["materialized_dynamic_instructions"] == 0
    assert trace.metadata["materialized_schedule_leaves"] == 3


def test_legacy_loop_text_is_refolded_before_cost_collection():
    schedule = IsaBuilder().stage(
        "decoder/legacy",
        IsaBuilder().raw(
            "C_LOOP_START gp6, 7\n"
            "V_ADD_VV gp1, gp2, gp3, 0\n"
            "C_LOOP_END gp6\n"
        ),
    )
    sink = SymbolicCostSink()
    # final_sequence is normally called by IsaEmitMixin; exercise that path by
    # emitting the raw body as a standalone final schedule.
    from compiler.aten.isa_builder import final_sequence

    sink.consume(final_sequence(schedule))
    assert sink.finish().dynamic_opcode_counts == Counter(
        {"C_LOOP_START": 1, "V_ADD_VV": 7, "C_LOOP_END": 7}
    )


def test_dma_metadata_fails_closed_without_a_stage():
    dma = DmaTransfer(
        opcode="H_PREFETCH_M",
        direction="read",
        role="weight",
        element_base_bytes=0,
        scale_base_bytes=4096,
        dim=64,
        amount=4,
        stride_bytes=4096,
    )
    schedule = IsaBuilder().instr("H_PREFETCH_M", gp(1), gp(2), "a0", 1, 0, dma=dma)
    sink = SymbolicCostSink()
    with pytest.raises(ValueError, match="no stage ownership"):
        sink.consume(schedule.finalized())


def test_unknown_opcode_fails_closed():
    schedule = IsaBuilder().stage("decoder/test", IsaBuilder().instr("X_MAGIC"))
    sink = SymbolicCostSink()
    with pytest.raises(ValueError, match="unknown final-schedule opcode"):
        sink.consume(schedule.finalized())


def test_hbm_opcode_without_dma_geometry_fails_closed():
    schedule = IsaBuilder().stage(
        "decoder/test",
        IsaBuilder().instr("H_PREFETCH_M", gp(1), gp(2), "a0", 1, 0),
    )
    sink = SymbolicCostSink()
    sink.consume(schedule.finalized())
    with pytest.raises(ValueError, match="incomplete final-schedule DMA coverage"):
        sink.finish()


def test_summary_template_replay_normalizes_enclosing_repeat():
    dma = DmaTransfer(
        opcode="H_PREFETCH_M",
        direction="read",
        role="weight",
        element_base_bytes=0,
        scale_base_bytes=4096,
        dim=64,
        amount=4,
        stride_bytes=4096,
    )
    layer_axis = RepeatAxis.from_mapping("layer", 3, {"hbm": 8192})
    sink = SymbolicCostSink(granularity="affine-block-summary-v1")
    sink.begin_stage("decoder/layer")
    sink.begin_repeat(3, layer_axis, "model-layer")
    sink.begin_template(("projection",))
    sink.emit_instruction(Instr("M_MM"))
    sink.emit_instruction(Instr("H_PREFETCH_M", dma=dma))
    sink.end_template(("projection",))
    assert sink.replay_template(("projection",), count=4)
    sink.end_repeat(3, layer_axis, "model-layer")
    sink.end_stage("decoder/layer")

    trace = sink.finish()
    assert trace.dynamic_opcode_counts == Counter(
        {"M_MM": 15, "H_PREFETCH_M": 15}
    )
    assert sum(event.multiplicity for event in trace.dma_events) == 15
    assert all(event.repeat_axes == (layer_axis,) for event in trace.dma_events)
