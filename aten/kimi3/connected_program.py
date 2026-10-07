"""Connected whole-backbone Kimi K3 decode program construction.

Unlike the earlier instruction-coverage builder, every stage consumes the
``VRAMMatrixVar`` returned by its producer.  KDA keeps its fixed physical hidden
address, so AttnRes output is explicitly copied into that address before the
X_STATE block and the resulting mixer output is accumulated into the prefix.
"""

from __future__ import annotations

import math
from collections import defaultdict

from aten.kda.scheduler import KdaScheduleConfig
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
    VRAMMatrixVar,
    assert_registers_are_free,
    reserve_expert_weight_table,
)
from compiler.aten.plena.program_routed_moe import KimiSituFPConstants
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
    if config.kda_key_dim != arch.kda_head_dim or config.kda_value_dim != arch.kda_head_dim:
        raise ValueError("KDA state dimensions must match the Kimi architecture")
    if context_length is None:
        context_length = 1
    if context_length != 1:
        raise ValueError(
            "context_length exceeds the connected Kimi decode capability: "
            "it currently executes the new token only; "
            "persistent multi-token MLA cache append is not implemented"
        )

    widths = MlaWidths.from_architecture(arch)
    if heads is not None:
        widths = MlaWidths(**{**widths.__dict__, "heads": heads})
    unaligned = widths.unaligned(mlen)
    if unaligned:
        raise ValueError(f"mlen {mlen} does not tile these MLA widths: {unaligned}")

    static_mla_heads = len(mla_layer_ids(arch.num_layers)) * widths.heads
    if not allow_unbounded_static_expansion:
        raise NotImplementedError(
            "full connected Kimi lowering requires compact Matrix tile loops "
            "and a looped MLA head body: a measured one-head diagnostic still "
            "emitted 100,221,916 instructions (3,739,264,558 assembly bytes), "
            "took 7m10s, and peaked at 24.1 GiB RSS; the requested configuration "
            f"would also statically expand {static_mla_heads} MLA head bodies. "
            "Routed Top-K is already looped; pass "
            "allow_unbounded_static_expansion=True only for compiler-scaling "
            "diagnostics, not deployable binaries"
        )

    kda_assembly, kda_program = _kda_assembly_by_layer(
        config, mlen=mlen, blen=blen
    )
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

    prog = PlenaCompiler(mlen=mlen, blen=blen)
    prog.hlen = 16
    prog.vram_allocator._vmm.mark_used(
        0, workspace_end, name="KDA_PHYSICAL_WORKSPACE"
    )
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
        kda_program.layout_descriptor_base
        + len(kda_program.layout_descriptor_image),
        config.projection_weight_hbm_base
        + len(config.kda_layer_ids) * config.projection_weight_layer_stride,
    )
    prog._next_hbm_addr = ((fixed_hbm_end + mlen - 1) // mlen) * mlen
    mla_constants, moe_constants, fpram_preload = _allocate_constants(prog, arch)
    attnres_constants = AttnResConstants(
        eps=mla_constants.input_eps,
        reciprocal_hidden=mla_constants.input_reciprocal_hidden,
    )
    stage_counts: dict[str, int] = defaultdict(int)

    def measured(stage: str, emit):
        # Counting the complete multi-million-line program around every stage is
        # quadratic.  EmitMixin stores append-only chunks, so count only what
        # this call appended.
        before = len(prog._code_chunks)
        result = emit()
        stage_counts[stage] += _instruction_count("".join(prog._code_chunks[before:]))
        return result

    def weight(name: str, rows: int, cols: int, *, bf16: bool = False):
        return prog.input(
            name,
            shape=(rows, cols),
            physical_shape=(rows, cols),
            real_data_ratio=2.0 if bf16 else None,
        )

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
        return prog.load_batch(
            prog.input(
                name,
                shape=(blocks, mlen),
                physical_shape=(physical_rows, mlen),
                real_data_ratio=2.0,
            ),
            name=name,
            storage_precision=2,
            hbm_precision=1,
        )

    correction = load_bf16_router_vector(
        "moe_correction_bias", arch.num_experts
    )
    cos = load_bf16_vector("rope_cos", arch.qk_rope_head_dim)
    sin = load_bf16_vector("rope_sin", arch.qk_rope_head_dim)
    prefix = hidden
    block_residuals: list[VRAMMatrixVar] = []
    mla_layers = set(mla_layer_ids(arch.num_layers))

    for layer_id in range(arch.num_layers):
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
            q_norm = load_bf16_vector(
                f"mla_q_norm_{layer_id}", arch.q_lora_rank
            )
            kv_norm = load_bf16_vector(
                f"mla_kv_norm_{layer_id}", arch.kv_lora_rank
            )
            mla_weights = MlaBlockWeights(
                q_a=weight(
                    f"mla_q_a_{layer_id}", arch.hidden_size, arch.q_lora_rank
                ),
                q_b=weight(
                    f"mla_q_b_{layer_id}", arch.q_lora_rank, widths.q_b_out
                ),
                kv_a=weight(
                    f"mla_kv_a_{layer_id}", arch.hidden_size, widths.kv_a_out
                ),
                kv_b=weight(
                    f"mla_kv_b_{layer_id}", arch.kv_lora_rank, widths.kv_b_out
                ),
                out=weight(
                    f"mla_out_{layer_id}", widths.attn_out, arch.hidden_size
                ),
                q_rope_rotate=weight(
                    f"mla_q_rope_rotate_{layer_id}",
                    arch.qk_rope_head_dim,
                    arch.qk_rope_head_dim,
                    bf16=True,
                ),
                k_rope_rotate=weight(
                    f"mla_k_rope_rotate_{layer_id}",
                    arch.qk_rope_head_dim,
                    arch.qk_rope_head_dim,
                    bf16=True,
                ),
                gate=weight(
                    f"mla_gate_{layer_id}", arch.hidden_size, widths.attn_out
                ),
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

        ffn_score = load_bf16_vector(
            f"attnres_ffn_score_{layer_id}", arch.hidden_size
        )
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

        ffn_norm = load_bf16_vector(
            f"ffn_input_norm_{layer_id}", arch.hidden_size
        )
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
                routed_gate=reserve_expert_weight_table(
                    prog,
                    name=f"moe_expert_gate_{layer_id}",
                    num_experts=arch.num_experts,
                    rows=arch.routed_expert_hidden_size,
                    cols=arch.moe_intermediate_size,
                ),
                routed_up_expert=reserve_expert_weight_table(
                    prog,
                    name=f"moe_expert_up_{layer_id}",
                    num_experts=arch.num_experts,
                    rows=arch.routed_expert_hidden_size,
                    cols=arch.moe_intermediate_size,
                ),
                routed_down_expert=reserve_expert_weight_table(
                    prog,
                    name=f"moe_expert_down_{layer_id}",
                    num_experts=arch.num_experts,
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
    )


__all__ = ["build_connected_kimi_k3_program"]
