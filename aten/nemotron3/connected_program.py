"""Connected 52-layer Nemotron 3 single-token decode program."""

from __future__ import annotations

import math
from collections import defaultdict

from aten.mamba.scheduler import (
    MambaScheduleConfig,
    Nemotron3MambaScheduler,
    SchedulePhase,
)
from aten.nemotron3.blocks import (
    NemotronAttentionShape,
    NemotronAttentionWeights,
    NemotronMoeConstants,
    NemotronMoeShape,
    NemotronMoeWeights,
    emit_nemotron_attention_block,
    emit_nemotron_moe_block,
)
from aten.nemotron3.scheduler import (
    HybridLayerType,
    Nemotron3Architecture,
    SYMBOL_TO_LAYER,
)
from aten.plena import (
    FullModelProgram,
    PlenaCompiler,
    SymbolicHbmBinding,
    VRAMMatrixVar,
    assert_registers_are_free,
    reserve_expert_weight_table,
)
from aten.plena.program_routed_moe import moe_stage_marker
from aten.state import Mamba2Payload, PrecisionCode
from aten.state.isa_lowering import MambaLayerMemoryMap


def _instruction_count(assembly: str) -> int:
    return sum(
        1
        for line in assembly.splitlines()
        if line.strip() and not line.strip().startswith(";")
    )


def build_connected_nemotron3_program(
    config: MambaScheduleConfig,
    *,
    architecture: Nemotron3Architecture | None = None,
    context_length: int | None = None,
    mlen: int | None = None,
    blen: int = 4,
):
    """Emit all 52 official block types with real producer-consumer edges."""
    from aten.nemotron3.program import _mamba_assembly_by_layer

    if config.phase != SchedulePhase.DECODE:
        raise ValueError("full-model lowering currently emits the decode program")
    arch = architecture or Nemotron3Architecture()
    if config.matrix_input_features != arch.hidden_size:
        raise ValueError("Mamba and Nemotron hidden widths must match")
    if mlen is None:
        mlen = 64
    aligned_widths = (
        arch.hidden_size,
        arch.attention_head_dim,
        arch.attention_heads * arch.attention_head_dim,
        arch.kv_heads * arch.attention_head_dim,
        arch.moe_intermediate_size,
        arch.shared_intermediate_size,
    )
    if any(width % mlen for width in aligned_widths):
        raise ValueError(
            f"mlen {mlen} does not tile the connected Nemotron widths {aligned_widths}"
        )
    if context_length is None:
        context_length = 1
    if context_length != 1:
        raise ValueError(
            "context_length exceeds the connected Nemotron decode capability: "
            "persistent multi-token GQA K/V cache append/read is not implemented"
        )

    mamba_assembly, mamba_program = _mamba_assembly_by_layer(
        config, mlen=mlen, blen=blen
    )
    memories = {
        event.memory
        for event in mamba_program.events
        if isinstance(event.memory, MambaLayerMemoryMap)
    }
    hidden_addresses = {memory.hidden_vram_addr for memory in memories}
    if len(hidden_addresses) != 1:
        raise ValueError("all Mamba layers must share one canonical hidden address")
    hidden_address = hidden_addresses.pop()
    workspace_end = max(
        memory.normalization_scratch_vram_addr + mlen for memory in memories
    )

    prog = PlenaCompiler(mlen=mlen, blen=blen, compact_matrix_loops=True)
    prog.hlen = 16
    prog.vram_allocator._vmm.mark_used(
        0, workspace_end, name="MAMBA_PHYSICAL_WORKSPACE"
    )
    fixed_hidden = prog.load_batch(
        prog.input(
            "hidden",
            shape=(1, arch.hidden_size),
            physical_shape=(blen, arch.hidden_size),
            prestaged_vram_addr=hidden_address,
        ),
        name="hidden",
    )

    # Existing Mamba descriptors use sparse fixed addresses up to 0x18... .
    # Keep compiler-owned weights above that range.  This full program is a
    # structural/machine-code artifact; compact Rust images use a separate
    # packed one-layer contract instead of allocating this sparse span.
    prog._next_hbm_addr = 0x1_9000_0000
    symbolic_bindings: list[SymbolicHbmBinding] = []

    def bind(
        *,
        name: str,
        hbm_addr: int,
        byte_size: int,
        logical_shape: tuple[int, ...],
        physical_shape: tuple[int, ...] | None = None,
        storage_format: str,
        layout: str = "row_major",
        source: str = "checkpoint_parameter",
        layer_id: int | None = None,
        metadata: tuple[tuple[str, int | float | str], ...] = (),
    ) -> None:
        symbolic_bindings.append(
            SymbolicHbmBinding(
                name=name,
                hbm_addr=hbm_addr,
                byte_size=byte_size,
                logical_shape=logical_shape,
                physical_shape=physical_shape or logical_shape,
                storage_format=storage_format,
                layout=layout,
                source=source,
                layer_id=layer_id,
                metadata=metadata,
            )
        )

    memory_by_layer = {memory.layer_id: memory for memory in memories}
    descriptor_by_layer = {}
    for event in Nemotron3MambaScheduler(config).build().events:
        if event.descriptor is not None:
            descriptor_by_layer[event.descriptor.layer_id] = event.descriptor
    for layer_id in sorted(memory_by_layer):
        memory = memory_by_layer[layer_id]
        descriptor = descriptor_by_layer[layer_id]
        payload = descriptor.payload
        if not isinstance(payload, Mamba2Payload):
            raise TypeError(f"Mamba layer {layer_id} has a non-Mamba descriptor")
        d_inner = descriptor.num_heads * payload.head_dim
        conv_channels = d_inner + 2 * payload.groups * payload.state_dim
        for suffix, address, rows, cols in (
            (
                "input_projection",
                memory.input_projection_weight_hbm_addr,
                config.matrix_input_features,
                descriptor.input_token_stride,
            ),
            (
                "output_projection",
                memory.output_projection_weight_hbm_addr,
                descriptor.output_token_stride,
                config.matrix_input_features,
            ),
        ):
            bind(
                name=f"mamba.layer{layer_id}.{suffix}",
                hbm_addr=address,
                byte_size=prog.hbm_tensor_size(rows * cols),
                logical_shape=(rows, cols),
                storage_format="plena_matrix_weight",
                layer_id=layer_id,
                metadata=(("real_data_ratio", prog.real_data_ratio),),
            )
        bind(
            name=f"mamba.layer{layer_id}.group_norm_weight",
            hbm_addr=memory.norm_weight_hbm_addr,
            byte_size=d_inner * 2,
            logical_shape=(d_inner,),
            storage_format="bf16_le",
            layer_id=layer_id,
        )
        parameter_specs = (
            (
                "conv_weight",
                payload.conv_weight_addr,
                conv_channels * payload.conv_kernel,
                (conv_channels, payload.conv_kernel),
            ),
            ("conv_bias", payload.conv_bias_addr, conv_channels, (conv_channels,)),
            (
                "a_log",
                payload.a_log_addr,
                descriptor.num_heads,
                (descriptor.num_heads,),
            ),
            (
                "dt_bias",
                payload.dt_bias_addr,
                descriptor.num_heads,
                (descriptor.num_heads,),
            ),
            (
                "d_skip",
                payload.d_skip_addr,
                descriptor.num_heads,
                (descriptor.num_heads,),
            ),
        )
        for suffix, address, elements, shape in parameter_specs:
            bind(
                name=f"mamba.layer{layer_id}.{suffix}",
                hbm_addr=address,
                byte_size=elements * descriptor.parameter_precision.element_bytes,
                logical_shape=shape,
                storage_format=f"state_{descriptor.parameter_precision.name.lower()}",
                layer_id=layer_id,
            )
        if descriptor.parameter_precision == PrecisionCode.MX8_B128:
            scale_bytes = (
                conv_channels * math.ceil(payload.conv_kernel / 128)
                + math.ceil(conv_channels / 128)
                + 3 * math.ceil(descriptor.num_heads / 128)
            )
            bind(
                name=f"mamba.layer{layer_id}.parameter_scales",
                hbm_addr=payload.parameter_scale_addr,
                byte_size=scale_bytes,
                logical_shape=(scale_bytes,),
                storage_format="mx8_b128_scale_stream",
                layer_id=layer_id,
            )

    prog.fp_var("zero", 1)
    mamba_eps = prog.fp_var("mamba_eps", 1)
    mamba_state_reciprocal = prog.fp_var("mamba_state_reciprocal", 1)
    prog.fp_var("mamba_reserved3", 1)
    prog.fp_var("mamba_reserved4", 1)
    prog.fp_var("mamba_one", 1)
    zero_row = prog.fp_var("zero_row", mlen)
    block_eps = prog.fp_var("block_eps", 1)
    block_reciprocal = prog.fp_var("block_reciprocal", 1)
    routed_scale = prog.fp_var("routed_scale", arch.experts_per_token)
    fpram_preload = [0.0] * (
        max(
            zero_row.address + zero_row.size,
            block_eps.address + block_eps.size,
            block_reciprocal.address + block_reciprocal.size,
            routed_scale.address + routed_scale.size,
        )
    )
    fpram_preload[mamba_eps.address] = 1.0e-5
    fpram_preload[mamba_state_reciprocal.address] = 1.0 / 512.0
    fpram_preload[5] = 1.0
    fpram_preload[block_eps.address] = 1.0e-5
    fpram_preload[block_reciprocal.address] = 1.0 / arch.hidden_size
    for index in range(routed_scale.size):
        fpram_preload[routed_scale.address + index] = 2.5
    moe_constants = NemotronMoeConstants(
        zero_row=zero_row,
        routed_scale=routed_scale,
    )

    stage_counts: dict[str, int] = defaultdict(int)
    prog.emit_comment(moe_stage_marker("non_moe", "connected Nemotron pre-MoE region"))

    def measured(stage: str, emit):
        before = len(prog._code_chunks)
        result = emit()
        stage_counts[stage] += _instruction_count("".join(prog._code_chunks[before:]))
        return result

    current_layer_id: int | None = None

    def weight(name: str, rows: int, cols: int, *, bf16: bool = False):
        var = prog.input(
            name,
            shape=(rows, cols),
            physical_shape=(rows, cols),
            real_data_ratio=2.0 if bf16 else None,
        )
        bind(
            name=name,
            hbm_addr=var.hbm_addr,
            byte_size=var.hbm_size,
            logical_shape=var.shape,
            physical_shape=var.physical_shape,
            storage_format="bf16_le" if bf16 else "plena_matrix_weight",
            layer_id=current_layer_id,
            metadata=(("real_data_ratio", 2.0 if bf16 else prog.real_data_ratio),),
        )
        return var

    def load_bf16_vector(name: str, width: int) -> VRAMMatrixVar:
        return prog.load_batch(
            weight(name, blen, width, bf16=True),
            name=name,
            storage_precision=2,
            hbm_precision=1,
        )

    def load_bf16_router_vector(name: str, width: int) -> VRAMMatrixVar:
        blocks = math.ceil(width / mlen)
        physical_rows = math.ceil(blocks / blen) * blen
        var = prog.input(
            name,
            shape=(blocks, mlen),
            physical_shape=(physical_rows, mlen),
            real_data_ratio=2.0,
        )
        bind(
            name=name,
            hbm_addr=var.hbm_addr,
            byte_size=var.hbm_size,
            logical_shape=(width,),
            physical_shape=var.physical_shape,
            storage_format="bf16_le",
            layer_id=current_layer_id,
            metadata=(("real_data_ratio", 2.0),),
        )
        return prog.load_batch(
            var,
            name=name,
            storage_precision=2,
            hbm_precision=1,
        )

    def expert_table(name: str, *, rows: int, cols: int):
        table = reserve_expert_weight_table(
            prog,
            name=name,
            num_experts=arch.routed_experts,
            rows=rows,
            cols=cols,
        )
        assert table.tile_group_stride is not None
        row_tiles = math.ceil(rows / mlen)
        col_tiles = math.ceil(cols / mlen)
        bind(
            name=name,
            hbm_addr=table.base,
            byte_size=row_tiles * col_tiles * table.tile_group_stride,
            logical_shape=(arch.routed_experts, rows, cols),
            physical_shape=(arch.routed_experts, rows, cols),
            storage_format="plena_matrix_weight",
            layout="expert_tile_major",
            layer_id=current_layer_id,
            metadata=(
                ("num_experts", arch.routed_experts),
                ("per_expert_stride", table.stride),
                ("tile_group_stride", table.tile_group_stride),
                ("mlen", mlen),
            ),
        )
        return table

    current = fixed_hidden
    for layer_id, symbol in enumerate(arch.pattern):
        current_layer_id = layer_id
        layer_type = SYMBOL_TO_LAYER[symbol]
        prog.emit_comment(
            f"==== connected Nemotron layer {layer_id} ({layer_type.value}) ===="
        )
        residual = prog.vram_copy(current, name=f"layer{layer_id}_residual", num_rows=1)
        normalized = prog.vram_copy(
            current, name=f"layer{layer_id}_normalized", num_rows=1
        )
        norm_weight = load_bf16_vector(f"layer{layer_id}_norm_weight", arch.hidden_size)
        measured(
            "block_rms_norm",
            lambda: (
                prog.rms_norm(
                    normalized,
                    eps_offset=block_eps.address,
                    reci_hid_offset=block_reciprocal.address,
                ),
                prog.vram_mul(normalized, norm_weight, num_rows=1),
            ),
        )
        prog.free_tensor(norm_weight)

        if layer_type is HybridLayerType.MAMBA:
            prog.vram_copy_region(
                fixed_hidden,
                normalized,
                num_rows=1,
                num_cols=arch.hidden_size,
            )
            assembly = mamba_assembly.get(layer_id)
            if assembly is None:
                raise ValueError(f"no Mamba assembly was lowered for layer {layer_id}")
            measured("mamba_mixer", lambda assembly=assembly: prog.emit(assembly))
            mixer_out = fixed_hidden
        elif layer_type is HybridLayerType.ATTENTION:
            mixer_out = measured(
                "attention",
                lambda: emit_nemotron_attention_block(
                    prog,
                    normalized,
                    shape=NemotronAttentionShape(
                        hidden=arch.hidden_size,
                        query_heads=arch.attention_heads,
                        kv_heads=arch.kv_heads,
                        head_dim=arch.attention_head_dim,
                    ),
                    weights=NemotronAttentionWeights(
                        q=weight(
                            f"attn_q_{layer_id}",
                            arch.hidden_size,
                            arch.attention_heads * arch.attention_head_dim,
                        ),
                        k=weight(
                            f"attn_k_{layer_id}",
                            arch.hidden_size,
                            arch.kv_heads * arch.attention_head_dim,
                        ),
                        v=weight(
                            f"attn_v_{layer_id}",
                            arch.hidden_size,
                            arch.kv_heads * arch.attention_head_dim,
                        ),
                        out=weight(
                            f"attn_out_{layer_id}",
                            arch.attention_heads * arch.attention_head_dim,
                            arch.hidden_size,
                        ),
                    ),
                    rows=1,
                    name=f"layer{layer_id}_attention",
                ),
            )
        else:
            correction = load_bf16_router_vector(
                f"moe_correction_{layer_id}", arch.routed_experts
            )
            mixer_out = measured(
                "moe",
                lambda: emit_nemotron_moe_block(
                    prog,
                    normalized,
                    shape=NemotronMoeShape(
                        hidden=arch.hidden_size,
                        intermediate=arch.moe_intermediate_size,
                        shared_intermediate=arch.shared_intermediate_size,
                        num_experts=arch.routed_experts,
                        top_k=arch.experts_per_token,
                    ),
                    weights=NemotronMoeWeights(
                        router=weight(
                            f"moe_router_{layer_id}",
                            arch.hidden_size,
                            arch.routed_experts,
                            bf16=True,
                        ),
                        routed_up=expert_table(
                            name=f"moe_expert_up_{layer_id}",
                            rows=arch.hidden_size,
                            cols=arch.moe_intermediate_size,
                        ),
                        routed_down=expert_table(
                            name=f"moe_expert_down_{layer_id}",
                            rows=arch.moe_intermediate_size,
                            cols=arch.hidden_size,
                        ),
                        shared_up=weight(
                            f"moe_shared_up_{layer_id}",
                            arch.hidden_size,
                            arch.shared_intermediate_size,
                        ),
                        shared_down=weight(
                            f"moe_shared_down_{layer_id}",
                            arch.shared_intermediate_size,
                            arch.hidden_size,
                        ),
                    ),
                    correction_bias=correction,
                    constants=moe_constants,
                    rows=1,
                    name=f"layer{layer_id}_moe",
                ),
            )
            prog.free_tensor(correction)

        prog.free_tensor(normalized)
        prog.vram_add(residual, mixer_out, num_rows=1)
        if mixer_out is not fixed_hidden:
            prog.free_tensor(mixer_out)
        if current is not fixed_hidden:
            prog.free_tensor(current)
        current = residual
        assert_registers_are_free(
            prog, f"connected Nemotron layer {layer_id} ({layer_type.value})"
        )

    current_layer_id = None
    final_norm_weight = load_bf16_vector("final_norm_weight", arch.hidden_size)
    measured(
        "final_rms_norm",
        lambda: (
            prog.rms_norm(
                current,
                eps_offset=block_eps.address,
                reci_hid_offset=block_reciprocal.address,
            ),
            prog.vram_mul(current, final_norm_weight, num_rows=1),
        ),
    )
    prog.free_tensor(final_norm_weight)

    assembly = prog.compile()
    layer_types = [SYMBOL_TO_LAYER[symbol].value for symbol in arch.pattern]
    return FullModelProgram(
        model="nemotron3",
        phase=config.phase.value,
        layer_counts={
            layer_type.value: layer_types.count(layer_type.value)
            for layer_type in HybridLayerType
        },
        assembly=assembly,
        instruction_count=_instruction_count(assembly),
        descriptor_base=mamba_program.descriptor_base,
        descriptor_image=mamba_program.descriptor_image,
        layout_descriptor_base=mamba_program.layout_descriptor_base,
        layout_descriptor_image=mamba_program.layout_descriptor_image,
        stage_instruction_counts=dict(stage_counts),
        fpram_preload=tuple(fpram_preload),
        output_vram_addr=prog.get_vram_addr(current.name),
        hbm_size=prog._next_hbm_addr,
        symbolic_hbm_bindings=tuple(symbolic_bindings),
    )


__all__ = ["build_connected_nemotron3_program"]
