from __future__ import annotations

from dataclasses import replace

from assembler.assembly_to_binary import AssemblyToBinary
from assembler.parser import parse_asm_file
from aten.kda.scheduler import KdaScheduleConfig, KimiK3KdaScheduler
from aten.mamba.scheduler import MambaScheduleConfig, Nemotron3MambaScheduler, SchedulePhase
from aten.state.isa_lowering import (
    KdaLayerMemoryMap,
    lower_kda_trace_to_existing_isa,
    lower_mamba_trace_to_existing_isa,
)


def test_real_nemotron_mamba_trace_lowers_to_existing_isa(tmp_path) -> None:
    trace = Nemotron3MambaScheduler(
        MambaScheduleConfig(phase=SchedulePhase.DECODE, decode_tokens=1)
    ).build()
    program = lower_mamba_trace_to_existing_isa(trace)
    assembly = program.assembly

    assert program.to_dict()["descriptor_count"] == 23
    assert program.instruction_count > 0
    for opcode in (
        "H_PREFETCH_M",
        "M_MM",
        "M_MM_WO",
        "V_EXP_V",
        "V_MUL_VV",
        "L_SCATTER_M",
        "X_STATE",
    ):
        assert opcode in assembly
    assert sum(event.operation == "IN_PROJECTION" for event in program.events) == 23
    assert sum(event.operation == "OUT_PROJECTION" for event in program.events) == 23
    assert sum(event.operation == "GATED_GROUP_RMSNORM" for event in program.events) == 23
    assert assembly.count("L_SCATTER_M") == 23
    assert program.to_dict()["layout_descriptor_count"] == 23

    path = tmp_path / "nemotron_mamba.asm"
    path.write_text(assembly)
    instructions = parse_asm_file(str(path))
    assembler = AssemblyToBinary("doc/operation.svh", "doc/configuration.svh")
    words = [assembler._convert_to_binary(instruction) for instruction in instructions]
    assert len(words) == program.instruction_count
    assert all(0 <= word < 1 << 32 for word in words)


def test_physical_memory_map_fits_the_configured_vector_sram() -> None:
    config = MambaScheduleConfig(
        phase=SchedulePhase.PREFILL,
        sequence_length=128,
    )
    program = lower_mamba_trace_to_existing_isa(
        Nemotron3MambaScheduler(config).build()
    )
    memories = [event.memory for event in program.events if event.memory is not None]
    assert memories
    assert all(
        memory.normalization_scratch_vram_addr + 64 <= config.vector_sram_elements
        for memory in memories
    )
    assert program.fpram_constants == ((1, 1.0e-5), (2, 1.0 / 512.0), (5, 1.0))


def test_real_kda_layer_lowers_all_official_projections_to_existing_isa(tmp_path) -> None:
    full = KimiK3KdaScheduler(
        KdaScheduleConfig(phase=SchedulePhase.DECODE, decode_tokens=1)
    ).build()
    # Physical lowering is layer-local. Keeping one real-shape layer makes the
    # assembler test fast without replacing any dimensions with a toy shape.
    events = tuple(
        event
        for event in full.events
        if event.layer_id == 0 or (event.layer_id is None and event.operation == "FENCE")
    )
    trace = replace(full, events=events)
    program = lower_kda_trace_to_existing_isa(trace)
    assembly = program.assembly

    assert program.to_dict()["contract"] == "plena-kda-existing-isa-v1"
    assert program.to_dict()["descriptor_count"] == 1
    assert program.instruction_count < 10_000
    assert assembly.count("rolled per-head KDA RMSNorm") == 1
    for stage in (
        "KDA_Q_PROJECTION",
        "KDA_K_PROJECTION",
        "KDA_V_PROJECTION",
        "KDA_DECAY_LOW_RANK_PROJECTION",
        "KDA_DECAY_EXPANSION_PROJECTION",
        "KDA_BETA_PROJECTION_PADDED_96_TO_128",
        "KDA_OUTPUT_GATE_PROJECTION",
        "KDA_OUT_PROJECTION",
    ):
        assert f"stage={stage}" in assembly
    for opcode in (
        "H_PREFETCH_M",
        "M_MM",
        "M_MM_WO",
        "V_RECI_V",
        "V_MUL_VV",
        "L_SCATTER_M",
        "X_STATE",
    ):
        assert opcode in assembly
    assert assembly.count("L_SCATTER_M") == 1
    assert program.to_dict()["layout_descriptor_count"] == 1

    memories = [event.memory for event in program.events if isinstance(event.memory, KdaLayerMemoryMap)]
    assert memories
    memory = memories[0]
    config = trace.config
    assert memory.normalization_scratch_vram_addr + 64 <= config.vector_sram_elements
    assert memory.hidden_vram_addr != memory.hidden_scratch_vram_addr
    assert memory.low_rank_vram_addr != memory.low_rank_scratch_vram_addr
    assert len(
        {
            memory.q_weight_hbm_addr,
            memory.k_weight_hbm_addr,
            memory.v_weight_hbm_addr,
            memory.decay_a_weight_hbm_addr,
            memory.decay_b_weight_hbm_addr,
            memory.beta_weight_hbm_addr,
            memory.gate_weight_hbm_addr,
            memory.output_weight_hbm_addr,
        }
    ) == 8
    # The learned KDA RMSNorm scale is stored as plain BF16.  It must use the
    # Vector KV precision selector (funct=1), not the default MX activation
    # selector that decodes the same bytes as zeros.
    assert "H_PREFETCH_V gp3, gp1, a2, 0, 1, 0" in assembly

    path = tmp_path / "kimi_k3_kda_layer.asm"
    path.write_text(assembly)
    instructions = parse_asm_file(str(path))
    assembler = AssemblyToBinary("doc/operation.svh", "doc/configuration.svh")
    words = [assembler._convert_to_binary(instruction) for instruction in instructions]
    assert len(words) == program.instruction_count
    assert all(0 <= word < 1 << 32 for word in words)


def test_compact_kda_keeps_the_same_dataflow_without_hardcoded_real_shapes() -> None:
    config = KdaScheduleConfig(
        phase=SchedulePhase.DECODE,
        decode_tokens=1,
        matrix_input_features=64,
        kda_layer_ids=(0,),
        kda_num_heads=1,
        projection_weight_hbm_base=0x10_0000,
        projection_weight_layer_stride=0x20_0000,
        projection_weight_offsets=tuple(index * 0x20_000 for index in range(9)),
    )
    trace = KimiK3KdaScheduler(config).build()
    program = lower_kda_trace_to_existing_isa(trace)
    assembly = program.assembly

    assert program.to_dict()["descriptor_count"] == 1
    assert "shape=64x128" in assembly
    assert "shape=128x64" in assembly
    assert "stage=KDA_BETA_PROJECTION_PADDED_TO_TILE" in assembly
    memories = [
        event.memory
        for event in program.events
        if isinstance(event.memory, KdaLayerMemoryMap)
    ]
    assert memories
    assert memories[0].q_weight_hbm_addr == config.projection_weight_hbm_base
    assert memories[0].norm_weight_hbm_addr == (
        config.projection_weight_hbm_base + config.projection_weight_offsets[-1]
    )


def test_compact_one_layer_mamba_hbm_layout_is_executable_sized() -> None:
    config = MambaScheduleConfig(
        phase=SchedulePhase.DECODE,
        decode_tokens=1,
        matrix_input_features=64,
        mamba_layer_ids=(0,),
        mamba_hbm_arena_base=0x10000,
    )
    trace = Nemotron3MambaScheduler(config).build()
    program = lower_mamba_trace_to_existing_isa(
        trace,
        descriptor_base=0,
        mlen=64,
        blen=4,
        vlen=64,
    )
    memories = {event.memory for event in program.events if event.memory is not None}

    assert trace.state_layers == (0,)
    assert trace.count("STEP") == 1
    assert len(memories) == 1
    memory = memories.pop()
    assert memory.input_projection_weight_hbm_addr < 16 * 1024 * 1024
    assert memory.output_projection_weight_hbm_addr < 16 * 1024 * 1024
    descriptor = next(event.descriptor for event in trace.events if event.descriptor)
    assert descriptor.state_hbm_addr < 16 * 1024 * 1024
    assert descriptor.payload.conv_weight_addr < 16 * 1024 * 1024
