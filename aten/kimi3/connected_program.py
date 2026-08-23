"""Connected whole-backbone Kimi K3 decode program construction.

Unlike the earlier instruction-coverage builder, every stage consumes the
``VRAMMatrixVar`` returned by its producer.  KDA keeps its fixed physical hidden
address, so AttnRes output is explicitly copied into that address before the
X_STATE block and the resulting mixer output is accumulated into the prefix.
"""

from __future__ import annotations

import math
from collections import defaultdict

from aten.kda.scheduler import KdaScheduleConfig, KimiK3KdaScheduler
from aten.kimi3.blocks import (
    AttnResConstants,
    KimiLatentMoeConstants,
    KimiLatentMoeShape,
    KimiLatentMoeWeights,
    MlaBlockShape,
    MlaBlockWeights,
    MlaNormConstants,
    emit_kimi_attn_res,
    emit_kimi_dense_ffn_residual_block,
    emit_kimi_latent_moe_residual_block,
    emit_mla_residual_block,
)
from aten.kimi3.scheduler import KimiK3Architecture
from aten.mamba.scheduler import SchedulePhase
from aten.plena import (
    FullModelProgram,
    PlenaCompiler,
    SymbolicHbmBinding,
    VRAMMatrixVar,
    assert_registers_are_free,
    reserve_expert_weight_table,
)
from compiler.aten.plena.program_routed_moe import KimiSituFPConstants
from aten.state import KdaPayload, PrecisionCode
from aten.state.isa_lowering import KdaLayerMemoryMap


def _instruction_count(assembly: str) -> int:
    return sum(
        1
        for line in assembly.splitlines()
        if line.strip() and not line.strip().startswith(";")
    )


def _allocate_constants(
    prog: PlenaCompiler,
    arch: KimiK3Architecture,
) -> tuple[MlaNormConstants, KimiLatentMoeConstants, tuple[float, ...]]:
    zero = prog.fp_var("zero", 1)
    kda_eps = prog.fp_var("kda_eps", 1)
    kda_state_reciprocal = prog.fp_var("kda_state_reciprocal", 1)
    kda_value_reciprocal = prog.fp_var("kda_value_reciprocal", 1)
    prog.fp_var("kda_reserved", 1)
    one = prog.fp_var("one", prog.blen)
    neg_one = prog.fp_var("neg_one", prog.blen)
    beta = prog.fp_var("beta", prog.blen)
    neg_two_beta = prog.fp_var("neg_two_beta", prog.blen)
    linear_beta = prog.fp_var("linear_beta", prog.blen)
    neg_two_linear_beta = prog.fp_var("neg_two_linear_beta", prog.blen)
    zero_row = prog.fp_var("zero_row", prog.mlen)
    hidden_eps = prog.fp_var("hidden_eps", 1)
    hidden_reciprocal = prog.fp_var("hidden_reciprocal", 1)
    q_eps = prog.fp_var("q_eps", 1)
    q_reciprocal = prog.fp_var("q_reciprocal", 1)
    kv_eps = prog.fp_var("kv_eps", 1)
    kv_reciprocal = prog.fp_var("kv_reciprocal", 1)
    routed_eps = prog.fp_var("routed_eps", 1)
    routed_reciprocal = prog.fp_var("routed_reciprocal", 1)

    variables = (
        zero,
        kda_eps,
        kda_state_reciprocal,
        kda_value_reciprocal,
        one,
        neg_one,
        beta,
        neg_two_beta,
        linear_beta,
        neg_two_linear_beta,
        zero_row,
        hidden_eps,
        hidden_reciprocal,
        q_eps,
        q_reciprocal,
        kv_eps,
        kv_reciprocal,
        routed_eps,
        routed_reciprocal,
    )
    preload = [0.0] * max(var.address + var.size for var in variables)

    def fill(var, value: float) -> None:
        for index in range(var.size):
            preload[var.address + index] = value

    fill(kda_eps, 1.0e-5)
    fill(kda_state_reciprocal, 1.0 / 512.0)
    fill(kda_value_reciprocal, 1.0 / arch.kda_head_dim)
    fill(one, 1.0)
    fill(neg_one, -1.0)
    fill(beta, 4.0)
    fill(neg_two_beta, -0.5)
    fill(linear_beta, 25.0)
    fill(neg_two_linear_beta, -0.08)
    for var in (hidden_eps, q_eps, kv_eps, routed_eps):
        fill(var, 1.0e-5)
    fill(hidden_reciprocal, 1.0 / arch.hidden_size)
    fill(q_reciprocal, 1.0 / arch.q_lora_rank)
    fill(kv_reciprocal, 1.0 / arch.kv_lora_rank)
    fill(routed_reciprocal, 1.0 / arch.routed_expert_hidden_size)

    mla = MlaNormConstants(
        input_eps=hidden_eps.address,
        input_reciprocal_hidden=hidden_reciprocal.address,
        q_eps=q_eps.address,
        q_reciprocal_hidden=q_reciprocal.address,
        kv_eps=kv_eps.address,
        kv_reciprocal_hidden=kv_reciprocal.address,
        gate_one=one,
        gate_neg_one=neg_one,
    )
    moe = KimiLatentMoeConstants(
        situ=KimiSituFPConstants(
            zero=zero,
            one=one,
            neg_one=neg_one,
            beta=beta,
            neg_two_over_beta=neg_two_beta,
            linear_beta=linear_beta,
            neg_two_over_linear_beta=neg_two_linear_beta,
        ),
        zero_row=zero_row,
        norm_eps=hidden_eps.address,
        norm_reciprocal_hidden=hidden_reciprocal.address,
        routed_norm_eps=routed_eps.address,
        routed_norm_reciprocal_hidden=routed_reciprocal.address,
    )
    return mla, moe, tuple(preload)


def build_connected_kimi_k3_program(
    config: KdaScheduleConfig,
    *,
    architecture: KimiK3Architecture | None = None,
    mlen: int = 64,
    blen: int = 4,
    context_length: int | None = None,
    heads: int | None = None,
    allow_unbounded_static_expansion: bool = False,
) -> FullModelProgram:
    """Emit all 93 layers with real producer-consumer and AttnRes ownership."""
    from aten.kimi3.program import MlaWidths, _kda_assembly_by_layer, mla_layer_ids

    if config.phase != SchedulePhase.DECODE:
        raise ValueError("full-model lowering currently emits the decode program")
    arch = architecture or KimiK3Architecture()
    if config.matrix_input_features != arch.hidden_size:
        raise ValueError("KDA and Kimi hidden widths must match")
    if tuple(config.kda_layer_ids) != arch.kda_layers:
        raise ValueError("KDA schedule layer IDs must match the Kimi architecture")
    if config.kda_num_heads != arch.attention_heads:
        raise ValueError("KDA schedule heads must match the Kimi architecture")
    if (
        config.kda_key_dim != arch.kda_head_dim
        or config.kda_value_dim != arch.kda_head_dim
    ):
        raise ValueError("KDA state dimensions must match the Kimi architecture")
    if context_length is None:
        context_length = 1
    if context_length != 1:
        raise ValueError(
            "context_length exceeds the full 93-layer connected Kimi builder's "
            "single-token limit; "
            "persistent compressed multi-token MLA append/reconstruct is implemented "
            "and Rust-verified by the standalone MLA block path"
        )

    widths = MlaWidths.from_architecture(arch)
    if heads is not None:
        widths = MlaWidths(**{**widths.__dict__, "heads": heads})
    unaligned = widths.unaligned(mlen)
    if unaligned:
        raise ValueError(f"mlen {mlen} does not tile these MLA widths: {unaligned}")

    kda_assembly, kda_program = _kda_assembly_by_layer(config, mlen=mlen, blen=blen)
    memories = {
        event.memory
        for event in kda_program.events
        if isinstance(event.memory, KdaLayerMemoryMap)
    }
    hidden_addresses = {memory.hidden_vram_addr for memory in memories}
    if len(hidden_addresses) != 1:
        raise ValueError("all KDA layers must share one canonical hidden address")
    hidden_address = hidden_addresses.pop()
    workspace_end = max(
        memory.normalization_scratch_vram_addr + mlen for memory in memories
    )

    prog = PlenaCompiler(
        mlen=mlen,
        blen=blen,
        compact_matrix_loops=not allow_unbounded_static_expansion,
    )
    prog.hlen = 16
    prog.vram_allocator._vmm.mark_used(0, workspace_end, name="KDA_PHYSICAL_WORKSPACE")
    hidden = prog.load_batch(
        prog.input(
            "hidden",
            shape=(1, arch.hidden_size),
            physical_shape=(blen, arch.hidden_size),
            prestaged_vram_addr=hidden_address,
        ),
        name="hidden",
    )
    fixed_hbm_end = max(
        kda_program.descriptor_base + len(kda_program.descriptor_image),
        kda_program.layout_descriptor_base + len(kda_program.layout_descriptor_image),
        config.projection_weight_hbm_base
        + len(config.kda_layer_ids) * config.projection_weight_layer_stride,
    )
    prog._next_hbm_addr = ((fixed_hbm_end + mlen - 1) // mlen) * mlen
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
    # The lowered ISA events intentionally retain only executable assembly and
    # memory maps. Read parameter addresses from the same scheduler trace that
    # produced those events instead of duplicating the descriptor ABI here.
    descriptor_by_layer = {}
    for event in KimiK3KdaScheduler(config).build().events:
        if event.descriptor is not None:
            descriptor_by_layer[event.descriptor.layer_id] = event.descriptor
    qk_features = arch.attention_heads * arch.kda_head_dim
    value_features = arch.attention_heads * arch.kda_head_dim
    for layer_id in sorted(memory_by_layer):
        memory = memory_by_layer[layer_id]
        descriptor = descriptor_by_layer[layer_id]
        payload = descriptor.payload
        if not isinstance(payload, KdaPayload):
            raise TypeError(f"KDA layer {layer_id} has a non-KDA descriptor")
        projection_specs = (
            ("q_proj", memory.q_weight_hbm_addr, arch.hidden_size, qk_features),
            ("k_proj", memory.k_weight_hbm_addr, arch.hidden_size, qk_features),
            ("v_proj", memory.v_weight_hbm_addr, arch.hidden_size, value_features),
            (
                "gate_proj",
                memory.gate_weight_hbm_addr,
                arch.hidden_size,
                value_features,
            ),
            (
                "out_proj",
                memory.output_weight_hbm_addr,
                value_features,
                arch.hidden_size,
            ),
            (
                "decay_a_proj",
                memory.decay_a_weight_hbm_addr,
                arch.hidden_size,
                payload.key_dim,
            ),
            (
                "decay_b_proj",
                memory.decay_b_weight_hbm_addr,
                payload.key_dim,
                qk_features,
            ),
            (
                "beta_proj",
                memory.beta_weight_hbm_addr,
                arch.hidden_size,
                math.ceil(arch.attention_heads / mlen) * mlen,
            ),
        )
        for suffix, address, rows, cols in projection_specs:
            bind(
                name=f"kda.layer{layer_id}.{suffix}",
                hbm_addr=address,
                byte_size=prog.hbm_tensor_size(rows * cols),
                logical_shape=(rows, cols),
                storage_format="plena_matrix_weight",
                layer_id=layer_id,
                metadata=(("real_data_ratio", prog.real_data_ratio),),
            )
        bind(
            name=f"kda.layer{layer_id}.group_norm_weight",
            hbm_addr=memory.norm_weight_hbm_addr,
            byte_size=value_features * 2,
            logical_shape=(value_features,),
            storage_format="bf16_le",
            layer_id=layer_id,
        )

        parameter_elements = (
            (
                "q_conv_weight",
                payload.q_conv_weight_addr,
                qk_features * payload.conv_kernel,
                (qk_features, payload.conv_kernel),
            ),
            (
                "k_conv_weight",
                payload.k_conv_weight_addr,
                qk_features * payload.conv_kernel,
                (qk_features, payload.conv_kernel),
            ),
            (
                "v_conv_weight",
                payload.v_conv_weight_addr,
                value_features * payload.conv_kernel,
                (value_features, payload.conv_kernel),
            ),
            ("q_conv_bias", payload.q_conv_bias_addr, qk_features, (qk_features,)),
            ("k_conv_bias", payload.k_conv_bias_addr, qk_features, (qk_features,)),
            (
                "v_conv_bias",
                payload.v_conv_bias_addr,
                value_features,
                (value_features,),
            ),
            (
                "a_log",
                payload.a_log_addr,
                arch.attention_heads,
                (arch.attention_heads,),
            ),
            ("dt_bias", payload.dt_bias_addr, qk_features, (qk_features,)),
        )
        for suffix, address, elements, shape in parameter_elements:
            bind(
                name=f"kda.layer{layer_id}.{suffix}",
                hbm_addr=address,
                byte_size=elements * descriptor.parameter_precision.element_bytes,
                logical_shape=shape,
                storage_format=f"state_{descriptor.parameter_precision.name.lower()}",
                layer_id=layer_id,
            )
        if descriptor.parameter_precision == PrecisionCode.MX8_B128:
            scale_bytes = (
                qk_features
                + qk_features
                + value_features
                + math.ceil(qk_features / 128)
                + math.ceil(qk_features / 128)
                + math.ceil(value_features / 128)
                + math.ceil(arch.attention_heads / 128)
                + arch.attention_heads
            )
            bind(
                name=f"kda.layer{layer_id}.parameter_scales",
                hbm_addr=payload.parameter_scale_addr,
                byte_size=scale_bytes,
                logical_shape=(scale_bytes,),
                storage_format="mx8_b128_scale_stream",
                layer_id=layer_id,
            )
    mla_constants, moe_constants, fpram_preload = _allocate_constants(prog, arch)
    attnres_constants = AttnResConstants(
        eps=mla_constants.input_eps,
        reciprocal_hidden=mla_constants.input_reciprocal_hidden,
    )
    stage_counts: dict[str, int] = defaultdict(int)
    current_layer_id: int | None = None

    def measured(stage: str, emit):
        # Counting the complete multi-million-line program around every stage is
        # quadratic.  EmitMixin stores append-only chunks, so count only what
        # this call appended.
        before = len(prog._code_chunks)
        result = emit()
        stage_counts[stage] += _instruction_count("".join(prog._code_chunks[before:]))
        return result

    def weight(
        name: str,
        rows: int,
        cols: int,
        *,
        bf16: bool = False,
        source: str = "checkpoint_parameter",
        layer_id: int | None = None,
    ):
        var = prog.input(
            name,
            shape=(rows, cols),
            physical_shape=(rows, cols),
            real_data_ratio=2.0 if bf16 else None,
        )
        resolved_layer_id = current_layer_id if layer_id is None else layer_id
        bind(
            name=name,
            hbm_addr=var.hbm_addr,
            byte_size=var.hbm_size,
            logical_shape=var.shape,
            physical_shape=var.physical_shape,
            storage_format="bf16_le" if bf16 else "plena_matrix_weight",
            source=source,
            layer_id=resolved_layer_id,
            metadata=(("real_data_ratio", 2.0 if bf16 else prog.real_data_ratio),),
        )
        return var

    def load_bf16_vector(
        name: str,
        width: int,
        *,
        source: str = "checkpoint_parameter",
        layer_id: int | None = None,
    ) -> VRAMMatrixVar:
        return prog.load_batch(
            weight(
                name,
                blen,
                width,
                bf16=True,
                source=source,
                layer_id=layer_id,
            ),
            name=name,
            storage_precision=2,
            hbm_precision=1,
        )

    def load_bf16_router_vector(
        name: str,
        width: int,
        *,
        source: str = "checkpoint_parameter",
        layer_id: int | None = None,
    ) -> VRAMMatrixVar:
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
            source=source,
            layer_id=current_layer_id if layer_id is None else layer_id,
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
            num_experts=arch.num_experts,
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
            logical_shape=(arch.num_experts, rows, cols),
            physical_shape=(arch.num_experts, rows, cols),
            storage_format="plena_matrix_weight",
            layout="expert_tile_major",
            layer_id=current_layer_id,
            metadata=(
                ("num_experts", arch.num_experts),
                ("per_expert_stride", table.stride),
                ("tile_group_stride", table.tile_group_stride),
                ("mlen", mlen),
            ),
        )
        return table

    correction = load_bf16_router_vector("moe_correction_bias", arch.num_experts)
    cos = load_bf16_vector(
        "rope_cos",
        arch.qk_rope_head_dim,
        source="runtime_generated",
    )
    sin = load_bf16_vector(
        "rope_sin",
        arch.qk_rope_head_dim,
        source="runtime_generated",
    )
    prefix = hidden
    block_residuals: list[VRAMMatrixVar] = []
    mla_layers = set(mla_layer_ids(arch.num_layers))

    for layer_id in range(arch.num_layers):
        current_layer_id = layer_id
        kind = "mla" if layer_id in mla_layers else "kda"
        prog.emit_comment(f"==== connected layer {layer_id} ({kind}) ====")
        if layer_id % arch.attn_res_block_size == 0:
            block_residuals.append(
                prog.vram_copy(
                    prefix,
                    name=f"attnres_block_snapshot_{layer_id // arch.attn_res_block_size}",
                    num_rows=1,
                )
            )

        mixer_score = load_bf16_vector(
            f"attnres_mixer_score_{layer_id}", arch.hidden_size
        )
        mixer_input = measured(
            "attn_res_before_mixer",
            lambda: emit_kimi_attn_res(
                prog,
                tuple(block_residuals),
                prefix,
                score_weight=mixer_score,
                constants=attnres_constants,
                rows=1,
                name=f"layer{layer_id}_attnres_mixer",
            ),
        )
        prog.free_tensor(mixer_score)

        # X_STATE writes its result to the canonical hidden address in place.
        # Preserve the unmodified prefix before a KDA layer can overwrite that
        # address; otherwise layer 0 computes mixer_out + mixer_out instead of
        # prefix + mixer_out.
        prefix_after_mixer = prog.vram_copy(
            prefix, name=f"layer{layer_id}_prefix_after_mixer", num_rows=1
        )

        if kind == "kda":
            kda_norm_weight = load_bf16_vector(
                f"kda_input_norm_{layer_id}", arch.hidden_size
            )
            normalized = prog.vram_copy(
                mixer_input, name=f"layer{layer_id}_kda_norm", num_rows=1
            )
            prog.rms_norm(
                normalized,
                eps_offset=mla_constants.input_eps,
                reci_hid_offset=mla_constants.input_reciprocal_hidden,
            )
            prog.vram_mul(normalized, kda_norm_weight, num_rows=1)
            prog.vram_copy_region(
                hidden, normalized, num_rows=1, num_cols=arch.hidden_size
            )
            prog.free_tensor(normalized)
            prog.free_tensor(kda_norm_weight)
            assembly = kda_assembly.get(layer_id)
            if assembly is None:
                raise ValueError(f"no KDA assembly was lowered for layer {layer_id}")
            measured("kda_mixer", lambda assembly=assembly: prog.emit(assembly))
            mixer_out = hidden
        else:
            input_norm = load_bf16_vector(
                f"mla_input_norm_{layer_id}", arch.hidden_size
            )
            q_norm = load_bf16_vector(f"mla_q_norm_{layer_id}", arch.q_lora_rank)
            kv_norm = load_bf16_vector(f"mla_kv_norm_{layer_id}", arch.kv_lora_rank)
            mla_weights = MlaBlockWeights(
                q_a=weight(f"mla_q_a_{layer_id}", arch.hidden_size, arch.q_lora_rank),
                q_b=weight(f"mla_q_b_{layer_id}", arch.q_lora_rank, widths.q_b_out),
                kv_a=weight(f"mla_kv_a_{layer_id}", arch.hidden_size, widths.kv_a_out),
                kv_b=weight(f"mla_kv_b_{layer_id}", arch.kv_lora_rank, widths.kv_b_out),
                out=weight(f"mla_out_{layer_id}", widths.attn_out, arch.hidden_size),
                q_rope_rotate=weight(
                    f"mla_q_rope_rotate_{layer_id}",
                    arch.qk_rope_head_dim,
                    arch.qk_rope_head_dim,
                    bf16=True,
                    source="runtime_generated",
                ),
                k_rope_rotate=weight(
                    f"mla_k_rope_rotate_{layer_id}",
                    arch.qk_rope_head_dim,
                    arch.qk_rope_head_dim,
                    bf16=True,
                    source="runtime_generated",
                ),
                gate=weight(f"mla_gate_{layer_id}", arch.hidden_size, widths.attn_out),
            )
            mixer_out = measured(
                "mla_mixer",
                lambda: emit_mla_residual_block(
                    prog,
                    mixer_input,
                    shape=MlaBlockShape(
                        hidden=arch.hidden_size,
                        q_lora=arch.q_lora_rank,
                        kv_lora=arch.kv_lora_rank,
                        qk_nope=arch.qk_nope_head_dim,
                        qk_rope=arch.qk_rope_head_dim,
                        v_head=arch.v_head_dim,
                        heads=widths.heads,
                    ),
                    weights=mla_weights,
                    cos=cos,
                    sin=sin,
                    norms=mla_constants,
                    input_norm_weight=input_norm,
                    q_norm_weight=q_norm,
                    kv_norm_weight=kv_norm,
                    rows=1,
                    name=f"layer{layer_id}_mla",
                    add_residual=False,
                ),
            )
            for value in (input_norm, q_norm, kv_norm):
                prog.free_tensor(value)
        prog.free_tensor(mixer_input)

        prog.vram_add(prefix_after_mixer, mixer_out, num_rows=1)
        if mixer_out is not hidden:
            prog.free_tensor(mixer_out)

        ffn_score = load_bf16_vector(f"attnres_ffn_score_{layer_id}", arch.hidden_size)
        ffn_input = measured(
            "attn_res_before_ffn",
            lambda: emit_kimi_attn_res(
                prog,
                tuple(block_residuals),
                prefix_after_mixer,
                score_weight=ffn_score,
                constants=attnres_constants,
                rows=1,
                name=f"layer{layer_id}_attnres_ffn",
            ),
        )
        prog.free_tensor(ffn_score)

        ffn_norm = load_bf16_vector(f"ffn_input_norm_{layer_id}", arch.hidden_size)
        if layer_id == 0:
            ffn_out = measured(
                "dense_situ_ffn",
                lambda: emit_kimi_dense_ffn_residual_block(
                    prog,
                    ffn_input,
                    weights=(
                        weight(
                            "dense_gate", arch.hidden_size, arch.dense_intermediate_size
                        ),
                        weight(
                            "dense_up", arch.hidden_size, arch.dense_intermediate_size
                        ),
                        weight(
                            "dense_down", arch.dense_intermediate_size, arch.hidden_size
                        ),
                    ),
                    intermediate=arch.dense_intermediate_size,
                    constants=moe_constants,
                    input_norm_weight=ffn_norm,
                    rows=1,
                    name="layer0_dense_ffn",
                    add_residual=False,
                ),
            )
        else:
            routed_norm = load_bf16_vector(
                f"routed_norm_{layer_id}", arch.routed_expert_hidden_size
            )
            shared_intermediate = arch.moe_intermediate_size * arch.shared_experts
            moe_weights = KimiLatentMoeWeights(
                router=weight(
                    f"moe_router_{layer_id}",
                    arch.hidden_size,
                    arch.num_experts,
                    bf16=True,
                ),
                routed_down=weight(
                    f"moe_routed_down_{layer_id}",
                    arch.hidden_size,
                    arch.routed_expert_hidden_size,
                ),
                routed_up=weight(
                    f"moe_routed_up_{layer_id}",
                    arch.routed_expert_hidden_size,
                    arch.hidden_size,
                ),
                routed_gate=expert_table(
                    name=f"moe_expert_gate_{layer_id}",
                    rows=arch.routed_expert_hidden_size,
                    cols=arch.moe_intermediate_size,
                ),
                routed_up_expert=expert_table(
                    name=f"moe_expert_up_{layer_id}",
                    rows=arch.routed_expert_hidden_size,
                    cols=arch.moe_intermediate_size,
                ),
                routed_down_expert=expert_table(
                    name=f"moe_expert_down_{layer_id}",
                    rows=arch.moe_intermediate_size,
                    cols=arch.routed_expert_hidden_size,
                ),
                shared=(
                    weight(
                        f"moe_shared_gate_{layer_id}",
                        arch.hidden_size,
                        shared_intermediate,
                    ),
                    weight(
                        f"moe_shared_up_{layer_id}",
                        arch.hidden_size,
                        shared_intermediate,
                    ),
                    weight(
                        f"moe_shared_down_{layer_id}",
                        shared_intermediate,
                        arch.hidden_size,
                    ),
                ),
            )
            ffn_out = measured(
                "latent_moe",
                lambda: emit_kimi_latent_moe_residual_block(
                    prog,
                    ffn_input,
                    shape=KimiLatentMoeShape(
                        hidden=arch.hidden_size,
                        routed_hidden=arch.routed_expert_hidden_size,
                        intermediate=arch.moe_intermediate_size,
                        shared_intermediate=shared_intermediate,
                        num_experts=arch.num_experts,
                        top_k=arch.experts_per_token,
                    ),
                    weights=moe_weights,
                    correction_bias=correction,
                    constants=moe_constants,
                    input_norm_weight=ffn_norm,
                    routed_norm_weight=routed_norm,
                    rows=1,
                    name=f"layer{layer_id}_latent_moe",
                    add_residual=False,
                ),
            )
            prog.free_tensor(routed_norm)
        prog.free_tensor(ffn_norm)
        prog.free_tensor(ffn_input)

        new_prefix = prog.vram_copy(
            prefix_after_mixer,
            name=f"layer{layer_id}_prefix_after_ffn",
            num_rows=1,
        )
        prog.vram_add(new_prefix, ffn_out, num_rows=1)
        prog.free_tensor(ffn_out)
        if prefix is not hidden:
            prog.free_tensor(prefix)
        prog.free_tensor(prefix_after_mixer)
        prefix = new_prefix
        assert_registers_are_free(prog, f"connected Kimi layer {layer_id}")

    current_layer_id = None
    final_score = load_bf16_vector("attnres_output_score", arch.hidden_size)
    output = measured(
        "output_attn_res",
        lambda: emit_kimi_attn_res(
            prog,
            tuple(block_residuals),
            prefix,
            score_weight=final_score,
            constants=attnres_constants,
            rows=1,
            name="output_attnres",
        ),
    )
    prog.free_tensor(final_score)
    final_norm_weight = load_bf16_vector("final_norm_weight", arch.hidden_size)
    measured(
        "final_rms_norm",
        lambda: (
            prog.rms_norm(
                output,
                eps_offset=mla_constants.input_eps,
                reci_hid_offset=mla_constants.input_reciprocal_hidden,
            ),
            prog.vram_mul(output, final_norm_weight, num_rows=1),
        ),
    )
    prog.free_tensor(final_norm_weight)

    assembly = prog.compile()
    return FullModelProgram(
        model="kimi_k3",
        phase=config.phase.value,
        layer_counts={
            "kda": arch.num_layers - len(mla_layers),
            "mla": len(mla_layers),
            "latent_moe": arch.num_layers - 1,
            "dense_ffn": 1,
        },
        assembly=assembly,
        instruction_count=_instruction_count(assembly),
        descriptor_base=kda_program.descriptor_base,
        descriptor_image=kda_program.descriptor_image,
        layout_descriptor_base=kda_program.layout_descriptor_base,
        layout_descriptor_image=kda_program.layout_descriptor_image,
        stage_instruction_counts=dict(stage_counts),
        fpram_preload=fpram_preload,
        output_vram_addr=prog.get_vram_addr(output.name),
        hbm_size=prog._next_hbm_addr,
        symbolic_hbm_bindings=tuple(symbolic_bindings),
    )


__all__ = ["build_connected_kimi_k3_program"]
