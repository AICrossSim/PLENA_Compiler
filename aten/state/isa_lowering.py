"""Lower Mamba Matrix/Vector service events to existing PLENA assembly."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, replace

from asm_templates import preload_act_asm, preload_addr_reg_asm, rms_norm_asm, silu_asm
from aten.isa_builder import IsaBuilder, addr, gp
from aten.mamba.scheduler import (
    MambaHbmLayout,
    MambaScheduleConfig,
    ScheduleTrace,
    TraceEvent,
)
from aten.state.contract import KdaPayload, Mamba2Payload, StateSubop, decode_instruction
from aten.state.layout_contract import decode_layout_instruction
from aten.state.lowering import (
    LoweredLayoutCommand,
    LoweredStateCommand,
    lower_state_trace,
)


@dataclass(frozen=True)
class MambaLayerMemoryMap:
    layer_id: int
    hidden_vram_addr: int
    normalization_scratch_vram_addr: int
    input_projection_weight_hbm_addr: int
    output_projection_weight_hbm_addr: int
    norm_weight_hbm_addr: int
    input_projection_scratch_vram_addr: int
    output_projection_scratch_vram_addr: int


@dataclass(frozen=True)
class KdaLayerMemoryMap:
    """Physical storage assigned to one KDA chunk.

    Addresses are Vector-SRAM element addresses. Matrix outputs retain PLENA's
    existing blocked layout: one MLEN feature tile contains all padded BLEN
    rows before the next feature tile.
    """

    layer_id: int
    hidden_vram_addr: int
    hidden_scratch_vram_addr: int
    normalization_scratch_vram_addr: int
    low_rank_vram_addr: int
    low_rank_scratch_vram_addr: int
    norm_weight_vram_addr: int
    q_weight_hbm_addr: int
    k_weight_hbm_addr: int
    v_weight_hbm_addr: int
    decay_a_weight_hbm_addr: int
    decay_b_weight_hbm_addr: int
    beta_weight_hbm_addr: int
    gate_weight_hbm_addr: int
    output_weight_hbm_addr: int
    norm_weight_hbm_addr: int
    projection_scratch_vram_addr: int
    gate_scratch_vram_addr: int
    output_projection_scratch_vram_addr: int


@dataclass(frozen=True)
class LoweredIsaEvent:
    event_index: int
    resource: str
    operation: str
    instruction_count: int
    assembly: str
    memory: MambaLayerMemoryMap | KdaLayerMemoryMap | None = None

    def to_dict(self, *, include_assembly: bool = False) -> dict[str, object]:
        result: dict[str, object] = {
            "event_index": self.event_index,
            "resource": self.resource,
            "operation": self.operation,
            "instruction_count": self.instruction_count,
            "assembly_sha256": hashlib.sha256(self.assembly.encode()).hexdigest(),
        }
        if self.memory is not None:
            result["memory"] = asdict(self.memory)
        if include_assembly:
            result["assembly"] = self.assembly
        return result


@dataclass(frozen=True)
class LoweredMambaIsaProgram:
    descriptor_base: int
    descriptor_image: bytes
    events: tuple[LoweredIsaEvent, ...]
    fpram_constants: tuple[tuple[int, float], ...]
    layout_descriptor_base: int = 0
    layout_descriptor_image: bytes = b""
    contract: str = "plena-mamba-existing-isa-v1"

    @property
    def assembly(self) -> str:
        return "".join(event.assembly for event in self.events)

    @property
    def instruction_count(self) -> int:
        return sum(event.instruction_count for event in self.events)

    def to_dict(self, *, include_assembly: bool = False) -> dict[str, object]:
        assembly = self.assembly
        return {
            "contract": self.contract,
            "descriptor_base": self.descriptor_base,
            "descriptor_count": len(self.descriptor_image) // 256,
            "descriptor_image_sha256": hashlib.sha256(self.descriptor_image).hexdigest(),
            "layout_descriptor_base": self.layout_descriptor_base,
            "layout_descriptor_count": len(self.layout_descriptor_image) // 256,
            "layout_descriptor_image_sha256": hashlib.sha256(
                self.layout_descriptor_image
            ).hexdigest(),
            "instruction_count": self.instruction_count,
            "assembly_sha256": hashlib.sha256(assembly.encode()).hexdigest(),
            "fpram_constants": [
                {"address": address, "value": value}
                for address, value in self.fpram_constants
            ],
            "events": [
                event.to_dict(include_assembly=include_assembly) for event in self.events
            ],
        }


def lower_mamba_trace_to_existing_isa(
    trace: ScheduleTrace,
    *,
    descriptor_base: int = 0x7000_0000,
    mlen: int = 64,
    blen: int = 4,
    vlen: int = 64,
    mram_tile_capacity: int = 4,
) -> LoweredMambaIsaProgram:
    if trace.model_name != "nemotron3":
        raise ValueError("existing-ISA lowering currently supports Nemotron Mamba traces")
    if mlen % blen or mlen != vlen:
        raise ValueError("current physical lowering requires MLEN=VLEN and MLEN divisible by BLEN")
    if trace.config.physical_row_tile_size != blen:
        raise ValueError("trace physical_row_tile_size must match lowering BLEN")
    lowered_state = lower_state_trace(trace, descriptor_base=descriptor_base)
    state_commands = {command.event_index: command for command in lowered_state.commands}
    layout_commands = {
        command.event_index: command for command in lowered_state.layout_commands
    }
    events = []
    for event in trace.events:
        assembly = ""
        memory = None
        if event.index in layout_commands:
            assembly = _layout_command_asm(layout_commands[event.index])
        elif event.index in state_commands:
            assembly = _state_command_asm(state_commands[event.index])
        elif event.operation in {"IN_PROJECTION", "OUT_PROJECTION", "GATED_GROUP_RMSNORM"}:
            if event.descriptor is None or not isinstance(event.descriptor.payload, Mamba2Payload):
                raise ValueError(f"{event.operation} has no Mamba descriptor")
            memory = _memory_map(event, trace.config, blen=blen, vlen=vlen)
            if event.operation == "IN_PROJECTION":
                assembly = _projection_asm(
                    event,
                    memory,
                    input_features=trace.config.matrix_input_features,
                    output_features=event.descriptor.input_token_stride,
                    input_vram_addr=memory.hidden_vram_addr,
                    output_vram_addr=event.descriptor.input_vram_addr,
                    scratch_vram_addr=memory.input_projection_scratch_vram_addr,
                    weight_hbm_addr=memory.input_projection_weight_hbm_addr,
                    mlen=mlen,
                    blen=blen,
                    mram_tile_capacity=mram_tile_capacity,
                )
            elif event.operation == "OUT_PROJECTION":
                assembly = _projection_asm(
                    event,
                    memory,
                    input_features=event.descriptor.output_token_stride,
                    output_features=trace.config.matrix_input_features,
                    input_vram_addr=event.descriptor.output_vram_addr,
                    output_vram_addr=memory.hidden_vram_addr,
                    scratch_vram_addr=memory.output_projection_scratch_vram_addr,
                    weight_hbm_addr=memory.output_projection_weight_hbm_addr,
                    mlen=mlen,
                    blen=blen,
                    mram_tile_capacity=mram_tile_capacity,
                )
            else:
                assembly = _gated_group_rmsnorm_asm(
                    event,
                    memory,
                    vlen=vlen,
                    blen=blen,
                )
        if assembly:
            events.append(
                LoweredIsaEvent(
                    event.index,
                    event.resource.value,
                    event.operation,
                    _instruction_count(assembly),
                    assembly,
                    memory,
                )
            )
    return LoweredMambaIsaProgram(
        descriptor_base=descriptor_base,
        descriptor_image=lowered_state.descriptor_image,
        events=tuple(events),
        fpram_constants=((1, 1.0e-5), (2, 1.0 / 512.0), (5, 1.0)),
        layout_descriptor_base=lowered_state.layout_descriptor_base,
        layout_descriptor_image=lowered_state.layout_descriptor_image,
    )


def lower_state_trace_to_existing_isa(
    trace: ScheduleTrace,
    **kwargs: int,
) -> LoweredMambaIsaProgram:
    """Dispatch a common X_STATE trace to the matching physical lowerer."""

    if trace.model_name == "nemotron3":
        return lower_mamba_trace_to_existing_isa(trace, **kwargs)
    if trace.model_name == "kimi_k3":
        return lower_kda_trace_to_existing_isa(trace, **kwargs)
    raise ValueError(f"existing-ISA lowering does not support model {trace.model_name!r}")


def lower_kda_trace_to_existing_isa(
    trace: ScheduleTrace,
    *,
    descriptor_base: int = 0x7000_0000,
    mlen: int = 64,
    blen: int = 4,
    vlen: int = 64,
    mram_tile_capacity: int = 4,
) -> LoweredMambaIsaProgram:
    """Lower one or more Kimi K3 KDA chunks to existing Matrix/Vector ISA.

    KDA uses eight independent official projections. They are deliberately
    emitted as separate Matrix programs; treating them as packed QKV would
    contradict the pinned Kimi implementation and would make FIFO readiness
    incorrect.
    """

    if trace.model_name != "kimi_k3":
        raise ValueError("KDA lowering requires a kimi_k3 trace")
    if mlen % blen or mlen != vlen:
        raise ValueError("current physical lowering requires MLEN=VLEN and MLEN divisible by BLEN")
    if trace.config.physical_row_tile_size != blen:
        raise ValueError("trace physical_row_tile_size must match lowering BLEN")
    lowered_state = lower_state_trace(trace, descriptor_base=descriptor_base)
    state_commands = {command.event_index: command for command in lowered_state.commands}
    layout_commands = {
        command.event_index: command for command in lowered_state.layout_commands
    }
    events: list[LoweredIsaEvent] = []
    for event in trace.events:
        assembly = ""
        memory = None
        if event.index in layout_commands:
            assembly = _layout_command_asm(layout_commands[event.index])
        elif event.index in state_commands:
            assembly = _state_command_asm(state_commands[event.index])
        elif event.operation in {
            "KDA_QKV_PROJECTION",
            "KDA_DECAY_BETA_PROJECTION",
            "KDA_OUTPUT_GATE_PROJECTION",
            "KDA_OUTPUT_GATE_RMSNORM",
            "KDA_OUT_PROJECTION",
        }:
            if event.descriptor is None or not isinstance(event.descriptor.payload, KdaPayload):
                raise ValueError(f"{event.operation} has no KDA descriptor")
            memory = _kda_memory_map(event, trace.config, blen=blen, vlen=vlen)
            assembly = _kda_event_asm(
                event,
                memory,
                matrix_input_features=trace.config.matrix_input_features,
                mlen=mlen,
                blen=blen,
                vlen=vlen,
                mram_tile_capacity=mram_tile_capacity,
            )
        if assembly:
            events.append(
                LoweredIsaEvent(
                    event.index,
                    event.resource.value,
                    event.operation,
                    _instruction_count(assembly),
                    assembly,
                    memory,
                )
            )
    return LoweredMambaIsaProgram(
        descriptor_base=descriptor_base,
        descriptor_image=lowered_state.descriptor_image,
        events=tuple(events),
        fpram_constants=(
            (1, 1.0e-5),
            (2, 1.0 / 512.0),
            (3, 1.0 / 128.0),
            (5, 1.0),
        ),
        layout_descriptor_base=lowered_state.layout_descriptor_base,
        layout_descriptor_image=lowered_state.layout_descriptor_image,
        contract="plena-kda-existing-isa-v1",
    )


def _memory_map(
    event: TraceEvent,
    config: MambaScheduleConfig,
    *,
    blen: int,
    vlen: int,
) -> MambaLayerMemoryMap:
    descriptor = event.descriptor
    assert descriptor is not None
    projection_buffer = descriptor.input_token_stride * descriptor.chunk_size
    scan_buffer = descriptor.output_token_stride * descriptor.chunk_size
    projection_slot = descriptor.input_vram_addr // projection_buffer
    scan_base = 2 * projection_buffer
    scan_slot = (descriptor.output_vram_addr - scan_base) // scan_buffer
    if projection_slot not in (0, 1) or scan_slot not in (0, 1):
        raise ValueError("descriptor does not use the expected double-buffer layout")
    hidden_vram_addr = scan_base + 2 * scan_buffer
    hidden_reserved_rows = math.ceil(descriptor.chunk_size / blen) * blen
    normalization_scratch = hidden_vram_addr + hidden_reserved_rows * config.matrix_input_features
    if normalization_scratch + vlen > config.vector_sram_elements:
        raise ValueError("physical Mamba lowering exceeds configured Vector SRAM")
    layer_id = descriptor.layer_id
    layer_ordinal = config.mamba_layer_ids.index(layer_id)
    hbm = MambaHbmLayout.build(config)
    return MambaLayerMemoryMap(
        layer_id=layer_id,
        hidden_vram_addr=hidden_vram_addr,
        normalization_scratch_vram_addr=normalization_scratch,
        input_projection_weight_hbm_addr=hbm.address(
            "input_projection", ordinal=layer_ordinal
        ),
        output_projection_weight_hbm_addr=hbm.address(
            "output_projection", ordinal=layer_ordinal
        ),
        norm_weight_hbm_addr=hbm.address("norm_weight", ordinal=layer_ordinal),
        input_projection_scratch_vram_addr=(1 - projection_slot) * projection_buffer,
        output_projection_scratch_vram_addr=scan_base + (1 - scan_slot) * scan_buffer,
    )


def _kda_memory_map(
    event: TraceEvent,
    config: MambaScheduleConfig,
    *,
    blen: int,
    vlen: int,
) -> KdaLayerMemoryMap:
    descriptor = event.descriptor
    assert descriptor is not None and isinstance(descriptor.payload, KdaPayload)
    payload = descriptor.payload
    input_buffer = descriptor.input_token_stride * descriptor.chunk_size
    output_buffer = descriptor.output_token_stride * descriptor.chunk_size
    projection_slot = descriptor.input_vram_addr // input_buffer
    output_base = 2 * input_buffer
    output_slot = (descriptor.output_vram_addr - output_base) // output_buffer
    if projection_slot not in (0, 1) or output_slot not in (0, 1):
        raise ValueError("KDA descriptor does not use the expected double-buffer layout")

    gate_base = output_base + 2 * output_buffer
    rows = math.ceil(descriptor.chunk_size / blen) * blen
    hidden_buffer = rows * config.matrix_input_features
    hidden_base = gate_base + 2 * output_buffer
    low_rank_buffer = rows * payload.key_dim
    low_rank_base = hidden_base + 2 * hidden_buffer
    norm_weight_vram = low_rank_base + 2 * low_rank_buffer
    normalization_scratch = norm_weight_vram + payload.key_dim
    if normalization_scratch + vlen > config.vector_sram_elements:
        raise ValueError(
            "physical KDA lowering exceeds configured Vector SRAM: "
            f"needs {normalization_scratch + vlen}, has {config.vector_sram_elements} elements"
        )

    # HBM addresses are element addresses, matching H_PREFETCH_M's existing
    # convention. The official defaults retain one 512-Mi-element window per
    # layer; compact numerical tests can pack the same tensors near address 0.
    layer_ordinal = config.kda_layer_ids.index(descriptor.layer_id)
    weight_base = (
        config.projection_weight_hbm_base
        + layer_ordinal * config.projection_weight_layer_stride
    )
    (
        q_offset,
        k_offset,
        v_offset,
        gate_offset,
        output_offset,
        decay_a_offset,
        decay_b_offset,
        beta_offset,
        norm_offset,
    ) = config.projection_weight_offsets
    return KdaLayerMemoryMap(
        layer_id=descriptor.layer_id,
        # The projection FIFO is double-buffered; the model hidden state is not.
        # Alternating hidden with projection_slot disconnects adjacent layers
        # because non-KDA blocks own one canonical hidden tensor. Keep hidden at
        # slot 0 and reserve slot 1 strictly as projection/output scratch.
        hidden_vram_addr=hidden_base,
        hidden_scratch_vram_addr=hidden_base + hidden_buffer,
        normalization_scratch_vram_addr=normalization_scratch,
        low_rank_vram_addr=low_rank_base + projection_slot * low_rank_buffer,
        low_rank_scratch_vram_addr=low_rank_base + (1 - projection_slot) * low_rank_buffer,
        norm_weight_vram_addr=norm_weight_vram,
        q_weight_hbm_addr=weight_base + q_offset,
        k_weight_hbm_addr=weight_base + k_offset,
        v_weight_hbm_addr=weight_base + v_offset,
        gate_weight_hbm_addr=weight_base + gate_offset,
        output_weight_hbm_addr=weight_base + output_offset,
        decay_a_weight_hbm_addr=weight_base + decay_a_offset,
        decay_b_weight_hbm_addr=weight_base + decay_b_offset,
        beta_weight_hbm_addr=weight_base + beta_offset,
        norm_weight_hbm_addr=weight_base + norm_offset,
        projection_scratch_vram_addr=(1 - projection_slot) * input_buffer,
        gate_scratch_vram_addr=gate_base + (1 - output_slot) * output_buffer,
        output_projection_scratch_vram_addr=hidden_base + hidden_buffer,
    )


def _kda_event_asm(
    event: TraceEvent,
    memory: KdaLayerMemoryMap,
    *,
    matrix_input_features: int,
    mlen: int,
    blen: int,
    vlen: int,
    mram_tile_capacity: int,
) -> str:
    descriptor = event.descriptor
    assert descriptor is not None and isinstance(descriptor.payload, KdaPayload)
    payload = descriptor.payload
    rows = descriptor.batch_size * descriptor.chunk_size
    hidden = memory.hidden_vram_addr
    projected = descriptor.input_vram_addr

    def projection(
        operation: str,
        *,
        input_features: int,
        output_features: int,
        input_addr: int,
        output_addr: int,
        scratch_addr: int,
        weight_addr: int,
    ) -> str:
        return _projection_asm(
            replace(event, operation=operation),
            memory,
            input_features=input_features,
            output_features=output_features,
            input_vram_addr=input_addr,
            output_vram_addr=output_addr,
            scratch_vram_addr=scratch_addr,
            weight_hbm_addr=weight_addr,
            mlen=mlen,
            blen=blen,
            mram_tile_capacity=mram_tile_capacity,
        )

    if event.operation == "KDA_QKV_PROJECTION":
        qk_features = descriptor.num_heads * payload.key_dim
        value_features = descriptor.num_heads * payload.value_dim
        return "".join(
            projection(
                f"KDA_{name.upper()}_PROJECTION",
                input_features=matrix_input_features,
                output_features=output_features,
                input_addr=hidden,
                output_addr=projected + offset * rows,
                scratch_addr=memory.projection_scratch_vram_addr + offset * rows,
                weight_addr=weight,
            )
            for name, offset, output_features, weight in (
                ("q", payload.q_offset, qk_features, memory.q_weight_hbm_addr),
                ("k", payload.k_offset, qk_features, memory.k_weight_hbm_addr),
                ("v", payload.v_offset, value_features, memory.v_weight_hbm_addr),
            )
        )
    if event.operation == "KDA_DECAY_BETA_PROJECTION":
        # beta has 96 live outputs. It is padded to two MLEN tiles so the
        # existing Matrix service can write it without a tail opcode; the 32
        # padding values occupy the descriptor stride's explicit tail.
        return "".join(
            (
                projection(
                    "KDA_DECAY_LOW_RANK_PROJECTION",
                    input_features=matrix_input_features,
                    output_features=payload.key_dim,
                    input_addr=hidden,
                    output_addr=memory.low_rank_vram_addr,
                    scratch_addr=memory.low_rank_scratch_vram_addr,
                    weight_addr=memory.decay_a_weight_hbm_addr,
                ),
                projection(
                    "KDA_DECAY_EXPANSION_PROJECTION",
                    input_features=payload.key_dim,
                    output_features=descriptor.num_heads * payload.key_dim,
                    input_addr=memory.low_rank_vram_addr,
                    output_addr=projected + payload.decay_offset * rows,
                    scratch_addr=memory.projection_scratch_vram_addr,
                    weight_addr=memory.decay_b_weight_hbm_addr,
                ),
                projection(
                    (
                        "KDA_BETA_PROJECTION_PADDED_96_TO_128"
                        if descriptor.num_heads == 96 and mlen == 64
                        else "KDA_BETA_PROJECTION_PADDED_TO_TILE"
                    ),
                    input_features=matrix_input_features,
                    output_features=(
                        math.ceil(descriptor.num_heads / mlen) * mlen
                    ),
                    input_addr=hidden,
                    output_addr=projected + payload.beta_offset * rows,
                    scratch_addr=memory.projection_scratch_vram_addr + payload.beta_offset * rows,
                    weight_addr=memory.beta_weight_hbm_addr,
                ),
            )
        )
    if event.operation == "KDA_OUTPUT_GATE_PROJECTION":
        assert event.aux_vram_addr is not None
        return projection(
            event.operation,
            input_features=matrix_input_features,
            output_features=descriptor.num_heads * payload.value_dim,
            input_addr=hidden,
            output_addr=event.aux_vram_addr,
            scratch_addr=memory.gate_scratch_vram_addr,
            weight_addr=memory.gate_weight_hbm_addr,
        )
    if event.operation == "KDA_OUTPUT_GATE_RMSNORM":
        return _kda_gated_rmsnorm_asm(event, memory, vlen=vlen, blen=blen)
    if event.operation == "KDA_OUT_PROJECTION":
        return projection(
            event.operation,
            input_features=descriptor.num_heads * payload.value_dim,
            output_features=matrix_input_features,
            input_addr=descriptor.output_vram_addr,
            output_addr=hidden,
            scratch_addr=memory.output_projection_scratch_vram_addr,
            weight_addr=memory.output_weight_hbm_addr,
        )
    raise ValueError(f"unsupported KDA physical event {event.operation}")


def _projection_asm(
    event: TraceEvent,
    memory: MambaLayerMemoryMap | KdaLayerMemoryMap,
    *,
    input_features: int,
    output_features: int,
    input_vram_addr: int,
    output_vram_addr: int,
    scratch_vram_addr: int,
    weight_hbm_addr: int,
    mlen: int,
    blen: int,
    mram_tile_capacity: int,
) -> str:
    descriptor = event.descriptor
    assert descriptor is not None
    rows = descriptor.batch_size * descriptor.chunk_size
    for name, value, divisor in (
        ("input_features", input_features, mlen),
        ("output_features", output_features, mlen),
        ("rows", rows, blen),
    ):
        if value % divisor:
            raise ValueError(f"{name}={value} must be divisible by {divisor}")
    input_tiles = input_features // mlen
    output_blocks = output_features // mlen
    activation_blocks = rows // blen
    chunks = tuple(
        (start, min(mram_tile_capacity, input_tiles - start))
        for start in range(0, input_tiles, mram_tile_capacity)
    )
    asm = IsaBuilder().comment(
        f"stage={event.operation} layer={descriptor.layer_id} rows={rows} "
        f"shape={input_features}x{output_features}"
    )
    if not 0 <= weight_hbm_addr < 1 << 64:
        raise ValueError("projection weight address must fit the 64-bit HBM register")
    asm.instr("S_ADDI_INT", gp(14), gp(0), weight_hbm_addr >> 32)
    asm.instr("S_ADDI_INT", gp(15), gp(0), weight_hbm_addr & 0xFFFF_FFFF)
    asm.instr("C_SET_ADDR_REG", addr(1), gp(14), gp(15))
    asm.instr("S_ADDI_INT", gp(15), gp(0), input_features * output_features)
    asm.instr("C_SET_SCALE_REG", gp(15))
    asm.instr("S_ADDI_INT", gp(15), gp(0), output_features)
    asm.instr("C_SET_STRIDE_REG", gp(15))

    for chunk_index, (k_start, k_count) in enumerate(chunks):
        target = output_vram_addr if chunk_index == 0 else scratch_vram_addr
        asm.comment(f"K chunk {chunk_index}: tile_start={k_start}, tile_count={k_count}")
        asm.instr("S_ADDI_INT", gp(10), gp(0), k_start * mlen * output_features)
        asm.instr("S_ADDI_INT", gp(11), gp(0), target)
        asm.instr("S_ADDI_INT", gp(12), gp(0), input_vram_addr + k_start * mlen * rows)
        asm.instr("C_LOOP_START", gp(5), output_blocks)
        asm.instr("S_ADDI_INT", gp(1), gp(10), 0)
        asm.instr("S_ADDI_INT", gp(2), gp(0), 0)
        asm.instr("C_LOOP_START", gp(6), k_count)
        asm.instr("H_PREFETCH_M", gp(2), gp(1), addr(1), 1, 0)
        asm.instr("S_ADDI_INT", gp(2), gp(2), mlen * mlen)
        asm.instr("S_ADDI_INT", gp(1), gp(1), mlen * output_features)
        asm.instr("C_LOOP_END", gp(6))
        asm.instr("S_ADDI_INT", gp(14), gp(0), 0)
        asm.instr("S_ADDI_INT", gp(4), gp(11), 0)
        asm.instr("C_LOOP_START", gp(7), mlen // blen)
        asm.instr("S_ADDI_INT", gp(13), gp(12), 0)
        asm.instr("S_ADDI_INT", gp(9), gp(4), 0)
        asm.instr("C_LOOP_START", gp(8), activation_blocks)
        asm.instr("S_ADDI_INT", gp(3), gp(13), 0)
        asm.instr("S_ADDI_INT", gp(2), gp(14), 0)
        asm.instr("C_LOOP_START", gp(6), k_count)
        asm.instr("M_MM", 0, gp(2), gp(3))
        asm.instr("S_ADDI_INT", gp(2), gp(2), mlen * mlen)
        asm.instr("S_ADDI_INT", gp(3), gp(3), mlen * rows)
        asm.instr("C_LOOP_END", gp(6))
        asm.instr("M_MM_WO", gp(9), gp(0), 0)
        asm.instr("S_ADDI_INT", gp(13), gp(13), mlen * blen)
        asm.instr("S_ADDI_INT", gp(9), gp(9), blen * mlen)
        asm.instr("C_LOOP_END", gp(8))
        asm.instr("S_ADDI_INT", gp(14), gp(14), blen * mlen)
        asm.instr("S_ADDI_INT", gp(4), gp(4), blen)
        asm.instr("C_LOOP_END", gp(7))
        asm.instr("S_ADDI_INT", gp(10), gp(10), mlen)
        asm.instr("S_ADDI_INT", gp(11), gp(11), mlen * rows)
        asm.instr("C_LOOP_END", gp(5))
        if chunk_index:
            vectors = rows * output_features // mlen
            asm.comment(f"accumulate K chunk {chunk_index}")
            asm.instr("S_ADDI_INT", gp(1), gp(0), output_vram_addr)
            asm.instr("S_ADDI_INT", gp(2), gp(0), scratch_vram_addr)
            asm.instr("C_LOOP_START", gp(5), vectors)
            asm.instr("V_ADD_VV", gp(1), gp(1), gp(2), 0)
            asm.instr("S_ADDI_INT", gp(1), gp(1), mlen)
            asm.instr("S_ADDI_INT", gp(2), gp(2), mlen)
            asm.instr("C_LOOP_END", gp(5))
    return asm.render()


def _gated_group_rmsnorm_asm(
    event: TraceEvent,
    memory: MambaLayerMemoryMap,
    *,
    vlen: int,
    blen: int,
) -> str:
    descriptor = event.descriptor
    assert descriptor is not None and isinstance(descriptor.payload, Mamba2Payload)
    payload = descriptor.payload
    rows = descriptor.batch_size * descriptor.chunk_size
    value_base = descriptor.output_vram_addr
    gate_base = descriptor.input_vram_addr
    d_inner = descriptor.num_heads * payload.head_dim
    group_size = d_inner // payload.groups
    parts = [_comment(event, f"gate -> group RMSNorm, rows={rows}, groups={payload.groups}")]
    parts.append(
        silu_asm(
            const_one_fp_address=5,
            alive_registers=[1, 2, 3],
            activation_base_address=gate_base,
            scratchpad_base_address=memory.normalization_scratch_vram_addr,
            vlen=vlen,
            batch_size=rows,
            hidden_dim=d_inner,
        )
    )
    multiply = IsaBuilder().comment("value *= silu(gate)")
    multiply.instr("S_ADDI_INT", gp(1), gp(0), value_base)
    multiply.instr("S_ADDI_INT", gp(2), gp(0), gate_base)
    multiply.instr("C_LOOP_START", gp(3), rows * d_inner // vlen)
    multiply.instr("V_MUL_VV", gp(1), gp(1), gp(2), 0)
    multiply.instr("S_ADDI_INT", gp(1), gp(1), vlen)
    multiply.instr("S_ADDI_INT", gp(2), gp(2), vlen)
    multiply.instr("C_LOOP_END", gp(3))
    parts.append(multiply.render())
    for group in range(payload.groups):
        parts.append(
            rms_norm_asm(
                _eps_offset=1,
                reci_hid_offset=2,
                alive_registers=[1, 2, 3, 4],
                activation_base_address=value_base + group * group_size * rows,
                scratchpad_base_address=memory.normalization_scratch_vram_addr,
                vlen=vlen,
                batch_size=rows,
                hidden_dim=group_size,
                unroll=False,
            )
        )
    parts.append(
        preload_addr_reg_asm(
            addr_reg_to_set=[2],
            available_registers=[1, 2],
            addr_reg_val=[memory.norm_weight_hbm_addr],
        )
    )
    parts.append(
        preload_act_asm(
            vlen=vlen,
            preload_len=4,
            batch=1,
            hidden_size=d_inner,
            act_vram_offset=gate_base,
            alive_registers=[1, 2, 3, 4, 5],
            activation_offset_reg=2,
            storage_precision=2,
            hbm_precision=1,
        )
    )
    weight = IsaBuilder().comment("apply RMSNorm weight to every token")
    weight.instr("S_ADDI_INT", gp(4), gp(0), value_base)
    weight.instr("C_LOOP_START", gp(5), rows)
    weight.instr("S_ADDI_INT", gp(1), gp(4), 0)
    weight.instr("S_ADDI_INT", gp(2), gp(0), gate_base)
    weight.instr("C_LOOP_START", gp(3), d_inner // vlen)
    weight.instr("V_MUL_VV", gp(1), gp(1), gp(2), 0)
    weight.instr("S_ADDI_INT", gp(1), gp(1), vlen * rows)
    weight.instr("S_ADDI_INT", gp(2), gp(2), vlen)
    weight.instr("C_LOOP_END", gp(3))
    weight.instr("S_ADDI_INT", gp(4), gp(4), vlen)
    weight.instr("C_LOOP_END", gp(5))
    parts.append(weight.render())
    return "".join(parts)


def _kda_gated_rmsnorm_asm(
    event: TraceEvent,
    memory: KdaLayerMemoryMap,
    *,
    vlen: int,
    blen: int,
) -> str:
    """Emit official KDA ``RMSNorm(value) * sigmoid(gate)`` per head."""

    descriptor = event.descriptor
    assert descriptor is not None and isinstance(descriptor.payload, KdaPayload)
    assert event.aux_vram_addr is not None
    payload = descriptor.payload
    rows = descriptor.batch_size * descriptor.chunk_size
    value_base = descriptor.output_vram_addr
    gate_base = event.aux_vram_addr
    projection = descriptor.num_heads * payload.value_dim
    parts = [_comment(event, f"per-head RMSNorm then sigmoid gate, rows={rows}")]

    if payload.value_dim != 2 * vlen:
        raise ValueError("compact KDA head RMSNorm currently requires value_dim=2*VLEN")
    norm = IsaBuilder().comment("rolled per-head KDA RMSNorm")
    norm.instr("S_ADDI_INT", gp(1), gp(0), value_base)
    norm.instr("S_ADDI_INT", gp(4), gp(0), memory.normalization_scratch_vram_addr)
    norm.instr("S_LD_FP", "f1", gp(0), 1)
    norm.instr("S_LD_FP", "f3", gp(0), 3)
    norm.instr("C_LOOP_START", gp(5), descriptor.num_heads)
    norm.instr("S_ADDI_INT", gp(2), gp(1), 0)
    norm.instr("C_LOOP_START", gp(6), rows)
    norm.instr("S_ADD_FP", "f2", "f0", "f0")
    norm.instr("V_MUL_VV", gp(4), gp(2), gp(2), 0)
    norm.instr("V_RED_SUM", "f2", gp(4))
    norm.instr("S_ADDI_INT", gp(3), gp(2), vlen * rows)
    norm.instr("V_MUL_VV", gp(4), gp(3), gp(3), 0)
    norm.instr("V_RED_SUM", "f2", gp(4))
    norm.instr("S_MUL_FP", "f2", "f2", "f3")
    norm.instr("S_ADD_FP", "f2", "f2", "f1")
    norm.instr("S_SQRT_FP", "f2", "f2")
    norm.instr("S_RECI_FP", "f2", "f2")
    for _ in range(4):
        norm.instr("S_ADDI_INT", gp(0), gp(0), 0)
    norm.instr("V_MUL_VF", gp(2), gp(2), "f2", 0)
    norm.instr("S_ADDI_INT", gp(0), gp(0), 0)
    norm.instr("V_MUL_VF", gp(3), gp(3), "f2", 0)
    norm.instr("S_ADDI_INT", gp(2), gp(2), vlen)
    norm.instr("C_LOOP_END", gp(6))
    norm.instr("S_ADDI_INT", gp(1), gp(1), payload.value_dim * rows)
    norm.instr("C_LOOP_END", gp(5))
    parts.append(norm.render())

    # FusedRMSNormGated owns one learned 128-value scale shared by all heads.
    norm_address = IsaBuilder().comment("load 64-bit KDA norm-weight HBM address")
    norm_address.instr("S_ADDI_INT", gp(1), gp(0), memory.norm_weight_hbm_addr >> 32)
    norm_address.instr(
        "S_ADDI_INT", gp(2), gp(0), memory.norm_weight_hbm_addr & 0xFFFF_FFFF
    )
    norm_address.instr("C_SET_ADDR_REG", addr(2), gp(1), gp(2))
    parts.append(norm_address.render())
    parts.append(
        preload_act_asm(
            vlen=vlen,
            preload_len=2,
            batch=1,
            hidden_size=payload.value_dim,
            act_vram_offset=memory.norm_weight_vram_addr,
            alive_registers=[1, 2, 3, 4, 5],
            activation_offset_reg=2,
            storage_precision=2,
            hbm_precision=1,
        )
    )
    weight = IsaBuilder().comment("apply shared KDA RMSNorm weight")
    weight.instr("S_ADDI_INT", gp(1), gp(0), value_base)
    weight.instr("C_LOOP_START", gp(5), descriptor.num_heads)
    weight.instr("S_ADDI_INT", gp(2), gp(0), memory.norm_weight_vram_addr)
    weight.instr("C_LOOP_START", gp(4), payload.value_dim // vlen)
    weight.instr("S_ADDI_INT", gp(3), gp(1), 0)
    weight.instr("C_LOOP_START", gp(6), rows)
    weight.instr("V_MUL_VV", gp(3), gp(3), gp(2), 0)
    weight.instr("S_ADDI_INT", gp(3), gp(3), vlen)
    weight.instr("C_LOOP_END", gp(6))
    weight.instr("S_ADDI_INT", gp(1), gp(1), rows * vlen)
    weight.instr("S_ADDI_INT", gp(2), gp(2), vlen)
    weight.instr("C_LOOP_END", gp(4))
    weight.instr("C_LOOP_END", gp(5))
    parts.append(weight.render())

    # Compute sigmoid(gate) into one scratch vector and immediately multiply
    # the matching value vector. Unlike SiLU, the original gate itself is not
    # multiplied into the result.
    gate = IsaBuilder().comment("value *= sigmoid(output_gate)")
    gate.instr("S_ADDI_INT", gp(1), gp(0), value_base)
    gate.instr("S_ADDI_INT", gp(2), gp(0), gate_base)
    gate.instr("S_ADDI_INT", gp(3), gp(0), memory.normalization_scratch_vram_addr)
    gate.instr("S_LD_FP", "f1", gp(0), 5)
    gate.instr("C_LOOP_START", gp(4), rows * projection // vlen)
    gate.instr("V_SUB_VF", gp(3), gp(2), "f0", 0, 1)
    gate.instr("V_EXP_V", gp(3), gp(3), 0)
    gate.instr("V_ADD_VF", gp(3), gp(3), "f1", 0)
    gate.instr("V_RECI_V", gp(3), gp(3), 0)
    gate.instr("V_MUL_VV", gp(1), gp(1), gp(3), 0)
    gate.instr("S_ADDI_INT", gp(1), gp(1), vlen)
    gate.instr("S_ADDI_INT", gp(2), gp(2), vlen)
    gate.instr("C_LOOP_END", gp(4))
    parts.append(gate.render())
    return "".join(parts)


def _state_command_asm(command: LoweredStateCommand) -> str:
    fields = decode_instruction(command.instruction_word)
    subop = fields["subop"]
    assert isinstance(subop, StateSubop)
    asm = IsaBuilder().comment(f"stage={command.operation} event={command.event_index}")
    if subop == StateSubop.FENCE:
        asm.instr("X_STATE", gp(0), gp(0), addr(0), int(fields["queue_id"]), int(subop))
        return asm.render()
    writes = {(write.register_class, write.register_index): write.value for write in command.register_writes}
    context_gp = int(fields["context_gp"])
    descriptor_gp = int(fields["descriptor_offset_gp"])
    descriptor_hbm = int(fields["descriptor_hbm_reg"])
    asm.instr("S_ADDI_INT", gp(context_gp), gp(0), writes[("gp", context_gp)])
    asm.instr("S_ADDI_INT", gp(descriptor_gp), gp(0), writes[("gp", descriptor_gp)])
    asm.instr("S_ADDI_INT", gp(3), gp(0), writes[("hbm", descriptor_hbm)])
    asm.instr("C_SET_ADDR_REG", addr(descriptor_hbm), gp(0), gp(3))
    asm.instr(
        "X_STATE",
        gp(context_gp),
        gp(descriptor_gp),
        addr(descriptor_hbm),
        int(fields["queue_id"]),
        int(subop),
    )
    return asm.render()


def _layout_command_asm(command: LoweredLayoutCommand) -> str:
    fields = decode_layout_instruction(command.instruction_word)
    writes = {
        (write.register_class, write.register_index): write.value
        for write in command.register_writes
    }
    context_gp = int(fields["context_gp"])
    descriptor_gp = int(fields["descriptor_offset_gp"])
    descriptor_hbm = int(fields["descriptor_hbm_reg"])
    asm = IsaBuilder().comment(
        f"stage={command.operation} event={command.event_index} "
        f"layout={command.descriptor.mode.name}"
    )
    asm.instr("S_ADDI_INT", gp(context_gp), gp(0), writes[("gp", context_gp)])
    asm.instr(
        "S_ADDI_INT", gp(descriptor_gp), gp(0), writes[("gp", descriptor_gp)]
    )
    asm.instr("S_ADDI_INT", gp(3), gp(0), writes[("hbm", descriptor_hbm)])
    asm.instr("C_SET_ADDR_REG", addr(descriptor_hbm), gp(0), gp(3))
    asm.instr(
        "L_SCATTER_M",
        gp(context_gp),
        gp(descriptor_gp),
        addr(descriptor_hbm),
        int(fields["buffer_id"]),
        int(fields["mode"]),
    )
    return asm.render()


def _comment(event: TraceEvent, text: str) -> str:
    return IsaBuilder().comment(
        f"stage={event.operation} event={event.index} layer={event.layer_id}: {text}"
    ).render()


def _instruction_count(assembly: str) -> int:
    return sum(
        bool(line) and not line.startswith(";") and not line.startswith("//")
        for line in (raw.strip() for raw in assembly.splitlines())
    )
