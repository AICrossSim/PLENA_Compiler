"""Connected Kimi K3 decoder blocks.

These emitters return the actual VRAM tensor produced by the preceding stage.
They deliberately do not accept pre-staged stand-ins for intermediates: doing
so would let instruction-count tests pass while silently disconnecting the
model dataflow.
"""

from __future__ import annotations

from dataclasses import dataclass

from compiler.aten.isa_builder import IsaBuilder, fp, gp
from compiler.aten.plena import (
    ExpertWeightTable,
    FPVar,
    InputVar,
    PlenaCompiler,
    VRAMMatrixVar,
    reserve_expert_weight_table,
)
from compiler.aten.plena.program_routed_moe import (
    ExpertWeights,
    KimiSituFPConstants,
    moe_end_marker,
)


@dataclass(frozen=True)
class MlaBlockShape:
    hidden: int
    q_lora: int
    kv_lora: int
    qk_nope: int
    qk_rope: int
    v_head: int
    heads: int

    @property
    def qk_head(self) -> int:
        return self.qk_nope + self.qk_rope

    @property
    def q_width(self) -> int:
        return self.heads * self.qk_head

    @property
    def kv_a_width(self) -> int:
        return self.kv_lora + self.qk_rope

    @property
    def kv_b_head(self) -> int:
        return self.qk_nope + self.v_head

    @property
    def kv_b_width(self) -> int:
        return self.heads * self.kv_b_head

    @property
    def attention_width(self) -> int:
        return self.heads * self.v_head

    def validate(self, mlen: int) -> None:
        fields = {
            "hidden": self.hidden,
            "q_lora": self.q_lora,
            "kv_lora": self.kv_lora,
            "qk_nope": self.qk_nope,
            "qk_rope": self.qk_rope,
            "v_head": self.v_head,
            "q_width": self.q_width,
            "kv_a_width": self.kv_a_width,
            "kv_b_width": self.kv_b_width,
            "attention_width": self.attention_width,
        }
        invalid = {name: width for name, width in fields.items() if width <= 0 or width % mlen}
        if invalid:
            raise ValueError(f"MLA widths must be positive MLEN multiples (MLEN={mlen}): {invalid}")


@dataclass(frozen=True)
class MlaBlockWeights:
    q_a: InputVar
    q_b: InputVar
    kv_a: InputVar
    kv_b: InputVar
    out: InputVar
    q_rope_rotate: InputVar
    k_rope_rotate: InputVar
    gate: InputVar | None = None


@dataclass(frozen=True)
class MlaNormConstants:
    input_eps: int
    input_reciprocal_hidden: int
    q_eps: int
    q_reciprocal_hidden: int
    kv_eps: int
    kv_reciprocal_hidden: int
    gate_one: FPVar | None = None
    gate_neg_one: FPVar | None = None


@dataclass(frozen=True)
class AttnResConstants:
    eps: int
    reciprocal_hidden: int


@dataclass(frozen=True)
class KimiLatentMoeShape:
    hidden: int
    routed_hidden: int
    intermediate: int
    shared_intermediate: int
    num_experts: int
    top_k: int

    def validate(self, mlen: int) -> None:
        widths = {
            "hidden": self.hidden,
            "routed_hidden": self.routed_hidden,
            "intermediate": self.intermediate,
            "shared_intermediate": self.shared_intermediate,
        }
        invalid = {name: width for name, width in widths.items() if width <= 0 or width % mlen}
        if invalid:
            raise ValueError(f"Kimi latent-MoE widths must be MLEN multiples: {invalid}")
        if not 0 < self.top_k <= self.num_experts:
            raise ValueError(
                f"top_k={self.top_k} must be in [1, num_experts={self.num_experts}]"
            )


@dataclass(frozen=True)
class KimiLatentMoeWeights:
    router: InputVar
    routed_down: InputVar
    routed_up: InputVar
    routed_gate: ExpertWeightTable
    routed_up_expert: ExpertWeightTable
    routed_down_expert: ExpertWeightTable
    shared: ExpertWeights


@dataclass(frozen=True)
class KimiLatentMoeConstants:
    situ: KimiSituFPConstants
    zero_row: FPVar
    norm_eps: int
    norm_reciprocal_hidden: int
    routed_norm_eps: int
    routed_norm_reciprocal_hidden: int


def emit_kimi_dense_ffn_residual_block(
    prog: PlenaCompiler,
    hidden: VRAMMatrixVar,
    *,
    weights: ExpertWeights,
    intermediate: int,
    constants: KimiLatentMoeConstants,
    input_norm_weight: VRAMMatrixVar | None = None,
    rows: int = 1,
    name: str = "kimi_dense_ffn",
    add_residual: bool = True,
) -> VRAMMatrixVar:
    """Emit Kimi's dense SiTU FFN and optional ordinary residual."""
    if intermediate <= 0 or intermediate % prog.mlen:
        raise ValueError(
            f"{name}: intermediate={intermediate} must be a positive MLEN multiple"
        )
    if rows < 1 or rows > hidden.shape[0]:
        raise ValueError(f"{name}: rows={rows} outside hidden rows={hidden.shape[0]}")
    residual = (
        prog.vram_copy(hidden, name=f"{name}_residual", num_rows=rows)
        if add_residual
        else None
    )
    ffn_input = prog.vram_copy(hidden, name=f"{name}_input", num_rows=rows)
    prog.rms_norm(
        ffn_input,
        eps_offset=constants.norm_eps,
        reci_hid_offset=constants.norm_reciprocal_hidden,
    )
    if input_norm_weight is not None:
        prog.vram_mul(ffn_input, input_norm_weight, num_rows=rows)
    output = prog.moe_shared_expert_v0(
        ffn_input,
        weights,
        rows=rows,
        intermediate=intermediate,
        constants=constants.situ,
        zero_row=constants.zero_row,
        activation_policy="kimi_situ",
        policy_name="kimi_k3",
        name=name,
    )
    if residual is not None:
        prog.vram_add(output, residual, num_rows=rows)
        prog.free_tensor(residual)
    prog.free_tensor(ffn_input)
    return output


def _view_columns(
    prog: PlenaCompiler,
    source: VRAMMatrixVar,
    *,
    name: str,
    col_offset: int,
    width: int,
) -> VRAMMatrixVar:
    if col_offset % prog.mlen or width % prog.mlen:
        raise ValueError(
            f"{name}: column view must be MLEN-aligned, got offset={col_offset}, width={width}"
        )
    if col_offset + width > source.shape[1]:
        raise ValueError(
            f"{name}: view [{col_offset}, {col_offset + width}) exceeds {source.name} width={source.shape[1]}"
        )
    rows = source.shape[0]
    physical_rows = source.physical_shape[0]
    base = prog.get_vram_addr(source.name) + (col_offset // prog.mlen) * physical_rows * prog.mlen
    return prog.alloc_at(
        name,
        rows=rows,
        cols=width,
        vram_addr=base,
        physical_shape=(physical_rows, width),
    )


def _emit_depth_softmax(
    prog: PlenaCompiler,
    scores: FPVar,
    probabilities: FPVar,
    *,
    rows: int,
    depth: int,
) -> None:
    """Stable softmax over the depth axis of token-major FPRAM scores."""
    gp_scores, gp_probs = prog.register_allocator.allocate_gp(2)
    fp_max, fp_value, fp_sum, fp_inverse = prog.allocate_fp_reg(4)
    try:
        asm = IsaBuilder().comment(
            f"Kimi AttnRes depth softmax: rows={rows}, depth={depth}"
        )
        for row in range(rows):
            score_base = scores.address + row * depth
            probability_base = probabilities.address + row * depth
            asm.instr("S_ADDI_INT", gp(gp_scores), gp(0), score_base)
            asm.instr("S_ADDI_INT", gp(gp_probs), gp(0), probability_base)
            asm.instr("S_LD_FP", fp(fp_max), gp(gp_scores), 0)
            for index in range(1, depth):
                asm.instr("S_LD_FP", fp(fp_value), gp(gp_scores), index)
                asm.instr("S_MAX_FP", fp(fp_max), fp(fp_max), fp(fp_value))

            asm.instr("S_ADD_FP", fp(fp_sum), fp(0), fp(0))
            for index in range(depth):
                asm.instr("S_LD_FP", fp(fp_value), gp(gp_scores), index)
                asm.instr("S_SUB_FP", fp(fp_value), fp(fp_value), fp(fp_max))
                asm.instr("S_EXP_FP", fp(fp_value), fp(fp_value))
                asm.instr("S_ST_FP", fp(fp_value), gp(gp_probs), index)
                asm.instr("S_ADD_FP", fp(fp_sum), fp(fp_sum), fp(fp_value))
            asm.instr("S_RECI_FP", fp(fp_inverse), fp(fp_sum))
            for index in range(depth):
                asm.instr("S_LD_FP", fp(fp_value), gp(gp_probs), index)
                asm.instr("S_MUL_FP", fp(fp_value), fp(fp_value), fp(fp_inverse))
                asm.instr("S_ST_FP", fp(fp_value), gp(gp_probs), index)
        prog.emit(asm.render())
    finally:
        prog.free_fp_reg([fp_max, fp_value, fp_sum, fp_inverse])
        prog.register_allocator.free_gp([gp_scores, gp_probs])


def emit_kimi_attn_res(
    prog: PlenaCompiler,
    block_residuals: tuple[VRAMMatrixVar, ...],
    prefix_sum: VRAMMatrixVar,
    *,
    score_weight: VRAMMatrixVar,
    constants: AttnResConstants,
    rows: int = 1,
    name: str = "kimi_attn_res",
) -> VRAMMatrixVar:
    """Emit official block AttnRes over prior block snapshots plus prefix.

    ``score_weight`` is the compile-time product of the official RMSNorm weight
    and the layer's one-row projection weight. Keeping that fold outside the
    runtime preserves the exact formula while avoiding one redundant vector
    multiply on every saved depth candidate.
    """
    candidates = (*block_residuals, prefix_sum)
    if not candidates:
        raise ValueError(f"{name}: AttnRes needs at least the current prefix")
    hidden = prefix_sum.shape[1]
    if hidden % prog.mlen:
        raise ValueError(f"{name}: hidden={hidden} must be divisible by MLEN={prog.mlen}")
    if rows < 1 or rows > prefix_sum.shape[0]:
        raise ValueError(f"{name}: rows={rows} outside prefix rows={prefix_sum.shape[0]}")
    for candidate in candidates:
        if candidate.shape != prefix_sum.shape or candidate.physical_shape != prefix_sum.physical_shape:
            raise ValueError(
                f"{name}: all candidates must match prefix shape/layout "
                f"{prefix_sum.shape}/{prefix_sum.physical_shape}, got "
                f"{candidate.shape}/{candidate.physical_shape}"
            )
    if score_weight.shape[0] < 1 or score_weight.shape[1] < hidden:
        raise ValueError(
            f"{name}: score weight must cover (1, {hidden}), got {score_weight.shape}"
        )

    depth = len(candidates)
    if rows * depth * 2 > 1024:
        raise ValueError(
            f"{name}: scores and probabilities need {rows * depth * 2} FPRAM values; "
            "decode AttnRes must fit the 1024-value FPRAM"
        )
    scores = prog.fp_var(f"{name}_scores", size=rows * depth)
    probabilities = prog.fp_var(f"{name}_probabilities", size=rows * depth)
    for candidate_index, candidate in enumerate(candidates):
        normalized = prog.vram_copy(
            candidate,
            name=f"{name}_candidate{candidate_index}_norm",
            num_rows=rows,
        )
        prog.rms_norm(
            normalized,
            eps_offset=constants.eps,
            reci_hid_offset=constants.reciprocal_hidden,
        )
        logits = prog.moe_router_logits_bf16_v0(
            normalized,
            score_weight,
            rows=rows,
            hidden=hidden,
            num_experts=1,
            policy_name="kimi_attn_res",
            name=f"{name}_candidate{candidate_index}_score",
        )
        for row in range(rows):
            prog.tile_row_sum(
                scores,
                logits,
                row_idx=row,
                target_offset=row * depth + candidate_index,
            )
        prog.free_tensor(normalized)
        prog.free_tensor(logits)

    _emit_depth_softmax(
        prog,
        scores,
        probabilities,
        rows=rows,
        depth=depth,
    )

    result = None
    col_blocks = hidden // prog.mlen
    for candidate_index, candidate in enumerate(candidates):
        weighted = prog.vram_copy(
            candidate,
            name=f"{name}_candidate{candidate_index}_weighted",
            num_rows=rows,
        )
        for col_block in range(col_blocks):
            for row in range(rows):
                prog.tile_row_mul_fp(
                    weighted,
                    probabilities,
                    row_idx=row,
                    fpram_offset=row * depth + candidate_index,
                    tile_col_idx=col_block,
                )
        if result is None:
            result = weighted
        else:
            prog.vram_add(result, weighted, num_rows=rows)
            prog.free_tensor(weighted)

    prog.free_fp_var(scores)
    prog.free_fp_var(probabilities)
    assert result is not None
    return result


def emit_mla_residual_block(
    prog: PlenaCompiler,
    hidden: VRAMMatrixVar,
    *,
    shape: MlaBlockShape,
    weights: MlaBlockWeights,
    cos: VRAMMatrixVar,
    sin: VRAMMatrixVar,
    norms: MlaNormConstants,
    input_norm_weight: VRAMMatrixVar | None = None,
    q_norm_weight: VRAMMatrixVar | None = None,
    kv_norm_weight: VRAMMatrixVar | None = None,
    rows: int = 1,
    name: str = "kimi_mla",
    add_residual: bool = True,
) -> VRAMMatrixVar:
    """Emit pre-norm MLA and optionally add the ordinary residual.

    The decode correctness path currently covers the new token only
    (``kv_seq_len == rows``).  K/V are reconstructed from the compressed
    latent and then written to an HBM scratch consumed by the existing
    attention engine.  Persistent multi-token compressed-cache append is a
    separate system feature and is intentionally not claimed here.
    """
    shape.validate(prog.mlen)
    if rows < 1 or rows > hidden.shape[0]:
        raise ValueError(f"{name}: rows={rows} outside hidden rows={hidden.shape[0]}")
    if hidden.shape[1] != shape.hidden:
        raise ValueError(
            f"{name}: hidden width={hidden.shape[1]} does not match MLA hidden={shape.hidden}"
        )
    if cos.shape[1] != shape.qk_rope or sin.shape[1] != shape.qk_rope:
        raise ValueError(
            f"{name}: RoPE cos/sin widths must be {shape.qk_rope}, got {cos.shape[1]}/{sin.shape[1]}"
        )

    prog.emit(f"; {moe_end_marker(f'{name} non-MoE region')}\n")

    residual = (
        prog.vram_copy(hidden, name=f"{name}_residual", num_rows=rows)
        if add_residual
        else None
    )
    mixer_input = prog.vram_copy(hidden, name=f"{name}_input", num_rows=rows)
    prog.rms_norm(
        mixer_input,
        eps_offset=norms.input_eps,
        reci_hid_offset=norms.input_reciprocal_hidden,
    )
    if input_norm_weight is not None:
        prog.vram_mul(mixer_input, input_norm_weight, num_rows=rows)

    q_latent = prog.linear_projection(
        mixer_input,
        weights.q_a,
        name=f"{name}_q_a",
        physical_shape=(hidden.physical_shape[0], shape.q_lora),
    )
    prog.rms_norm(
        q_latent,
        eps_offset=norms.q_eps,
        reci_hid_offset=norms.q_reciprocal_hidden,
    )
    if q_norm_weight is not None:
        prog.vram_mul(q_latent, q_norm_weight, num_rows=rows)
    compressed_kv = prog.linear_projection(
        mixer_input,
        weights.kv_a,
        name=f"{name}_kv_a",
        physical_shape=(hidden.physical_shape[0], shape.kv_a_width),
    )
    kv_latent = _view_columns(
        prog,
        compressed_kv,
        name=f"{name}_kv_latent",
        col_offset=0,
        width=shape.kv_lora,
    )
    k_rope = _view_columns(
        prog,
        compressed_kv,
        name=f"{name}_k_rope",
        col_offset=shape.kv_lora,
        width=shape.qk_rope,
    )
    prog.rms_norm(
        kv_latent,
        eps_offset=norms.kv_eps,
        reci_hid_offset=norms.kv_reciprocal_hidden,
    )
    if kv_norm_weight is not None:
        prog.vram_mul(kv_latent, kv_norm_weight, num_rows=rows)
    k_rope_rot = prog.linear_projection_bf16(
        k_rope,
        weights.k_rope_rotate,
        name=f"{name}_k_rope_rot",
        physical_shape=k_rope.physical_shape,
    )
    prog.rope(k_rope, k_rope_rot, cos, sin)
    prog.free_tensor(k_rope_rot)

    attention = prog.alloc(
        f"{name}_attention",
        rows=hidden.shape[0],
        cols=shape.attention_width,
        strict=False,
        physical_shape=(hidden.physical_shape[0], shape.attention_width),
    )
    # The online-softmax primitive always consumes one complete MLEN-row Q/K/V
    # tile even for decode(seq_len=1).  The connected model otherwise carries
    # only BLEN physical rows, which is sufficient for projections but not for
    # the attention SRAM interface.  Pad only the per-head scratch tensors;
    # the logical row count and the final mixer output remain unchanged.
    attention_tile_rows = max(prog.mlen, hidden.physical_shape[0])
    for head in range(shape.heads):
        q_head = prog.linear_projection_slice(
            q_latent,
            weights.q_b,
            output_col_offset=head * shape.qk_head,
            output_features=shape.qk_head,
            name=f"{name}_q_b_head{head}",
            physical_shape=(attention_tile_rows, shape.qk_head),
        )
        q_rope = _view_columns(
            prog,
            q_head,
            name=f"{name}_q_rope{head}",
            col_offset=shape.qk_nope,
            width=shape.qk_rope,
        )
        q_rope_rot = prog.linear_projection_bf16(
            q_rope,
            weights.q_rope_rotate,
            name=f"{name}_q_rope_rot{head}",
            physical_shape=q_rope.physical_shape,
        )
        prog.rope(q_rope, q_rope_rot, cos, sin)
        prog.free_tensor(q_rope_rot)

        kv_head = prog.linear_projection_slice(
            kv_latent,
            weights.kv_b,
            output_col_offset=head * shape.kv_b_head,
            output_features=shape.kv_b_head,
            name=f"{name}_kv_b_head{head}",
            physical_shape=(attention_tile_rows, shape.kv_b_head),
        )
        k_head = prog.alloc(
            f"{name}_k_head{head}",
            rows=hidden.shape[0],
            cols=shape.qk_head,
            strict=False,
            physical_shape=(attention_tile_rows, shape.qk_head),
        )
        prog.vram_copy_region(
            k_head,
            kv_head,
            num_rows=rows,
            num_cols=shape.qk_nope,
        )
        prog.vram_copy_region(
            k_head,
            k_rope,
            num_rows=rows,
            num_cols=shape.qk_rope,
            dst_col_offset=shape.qk_nope,
        )
        v_head = _view_columns(
            prog,
            kv_head,
            name=f"{name}_v_head{head}",
            col_offset=shape.qk_nope,
            width=shape.v_head,
        )

        # The current attention primitive consumes K/V from HBM.  These stores
        # are a real producer-consumer edge, not pre-staged placeholders.
        k_hbm = prog.store(k_head, name=f"{name}_k_scratch{head}")
        v_hbm = prog.store(v_head, name=f"{name}_v_scratch{head}")
        head_out = prog.flash_attention(
            q_head,
            k_hbm,
            v_hbm,
            scale=shape.qk_head**-0.5,
            batch_size=1,
            seq_len=rows,
            kv_seq_len=rows,
        )
        prog.vram_copy_region(
            attention,
            head_out,
            num_rows=rows,
            num_cols=shape.v_head,
            dst_col_offset=head * shape.v_head,
        )
        prog.free_tensor(q_head)
        prog.free_tensor(kv_head)
        prog.free_tensor(k_head)
        prog.free_tensor(head_out)

    if weights.gate is not None:
        if norms.gate_one is None or norms.gate_neg_one is None:
            raise ValueError(f"{name}: output gate requires gate_one and gate_neg_one constants")
        gate = prog.linear_projection(
            mixer_input,
            weights.gate,
            name=f"{name}_gate",
            physical_shape=(hidden.physical_shape[0], shape.attention_width),
        )
        for col_block in range(shape.attention_width // prog.mlen):
            prog.tile_row_mul_fp(
                gate,
                norms.gate_neg_one,
                rows=list(range(rows)),
                tile_col_idx=col_block,
            )
            prog.tile_row_exp(gate, rows=list(range(rows)), tile_col_idx=col_block)
            prog.tile_row_add_fp(
                gate,
                norms.gate_one,
                rows=list(range(rows)),
                tile_col_idx=col_block,
            )
            prog.tile_row_reci(gate, rows=list(range(rows)), tile_col_idx=col_block)
        prog.vram_mul(attention, gate, num_rows=rows)

    mixer_out = prog.linear_projection(
        attention,
        weights.out,
        name=f"{name}_out",
        physical_shape=hidden.physical_shape,
    )
    if residual is not None:
        prog.vram_add(mixer_out, residual, num_rows=rows)
    for temporary in (
        mixer_input,
        q_latent,
        compressed_kv,
        attention,
    ):
        prog.free_tensor(temporary)
    if residual is not None:
        prog.free_tensor(residual)
    if weights.gate is not None:
        prog.free_tensor(gate)
    return mixer_out


def emit_kimi_latent_moe_residual_block(
    prog: PlenaCompiler,
    hidden: VRAMMatrixVar,
    *,
    shape: KimiLatentMoeShape,
    weights: KimiLatentMoeWeights,
    correction_bias: VRAMMatrixVar,
    constants: KimiLatentMoeConstants,
    input_norm_weight: VRAMMatrixVar | None = None,
    routed_norm_weight: VRAMMatrixVar | None = None,
    rows: int = 1,
    int_sram_base: int = 0,
    name: str = "kimi_latent_moe",
    add_residual: bool = True,
    loop_topk: bool = True,
) -> VRAMMatrixVar:
    """Emit Kimi's exact latent routed/shared MoE and optional residual."""
    shape.validate(prog.mlen)
    if hidden.shape[1] != shape.hidden:
        raise ValueError(
            f"{name}: hidden width={hidden.shape[1]} does not match {shape.hidden}"
        )
    if rows < 1 or rows > hidden.shape[0]:
        raise ValueError(f"{name}: rows={rows} outside hidden rows={hidden.shape[0]}")
    tables = (
        weights.routed_gate,
        weights.routed_up_expert,
        weights.routed_down_expert,
    )
    if any(table.num_experts != shape.num_experts for table in tables):
        raise ValueError("all routed expert tables must match shape.num_experts")

    prog.emit(f"; {moe_end_marker(f'{name} prelude')}\n")

    residual = (
        prog.vram_copy(hidden, name=f"{name}_residual", num_rows=rows)
        if add_residual
        else None
    )
    moe_input = prog.vram_copy(hidden, name=f"{name}_input", num_rows=rows)
    prog.rms_norm(
        moe_input,
        eps_offset=constants.norm_eps,
        reci_hid_offset=constants.norm_reciprocal_hidden,
    )
    if input_norm_weight is not None:
        prog.vram_mul(moe_input, input_norm_weight, num_rows=rows)

    logits = prog.qwen3_router_logits_matrix_bf16_rowpacked_v0(
        moe_input,
        weights.router,
        rows=rows,
        hidden=shape.hidden,
        num_experts=shape.num_experts,
        name=f"{name}_router",
    )
    topk_weights = prog.fp_var(f"{name}_topk_weights", size=rows * shape.top_k)
    for token_idx in range(rows):
        prog.moe_router_select_v0(
            logits,
            token_idx=token_idx,
            weights_fp_base=topk_weights.address + token_idx * shape.top_k,
            indices_int_base=int_sram_base + token_idx * shape.top_k,
            num_experts=shape.num_experts,
            top_k=shape.top_k,
            route_weight_mode="sigmoid_normalized",
            correction_bias=correction_bias,
            policy_name="kimi_k3",
            name=f"{name}_token{token_idx}",
        )

    routed_input = prog.linear_projection(
        moe_input,
        weights.routed_down,
        name=f"{name}_latent_down",
        physical_shape=(hidden.physical_shape[0], shape.routed_hidden),
    )
    routed_accumulator = prog.alloc(
        f"{name}_routed_accumulator",
        rows=hidden.shape[0],
        cols=shape.routed_hidden,
        strict=False,
        physical_shape=(hidden.physical_shape[0], shape.routed_hidden),
    )
    prog.moe_true_zero_vram_rows_v0(
        routed_accumulator,
        rows=list(range(rows)),
        hidden=shape.routed_hidden,
        zero_row=constants.zero_row,
        policy_name="kimi_k3",
        stage="accumulator_init",
        name=f"{name}_routed_zero",
    )

    route_scratch = prog.fp_var(f"{name}_route_scratch", size=prog.mlen)
    table_bases = tuple(table.base for table in tables)
    table_strides = tuple(table.stride for table in tables)
    tile_group_strides = tuple(table.tile_group_stride for table in tables)
    templates: ExpertWeights = tuple(table.template for table in tables)  # type: ignore[assignment]
    if loop_topk and rows != 1:
        raise NotImplementedError(
            "looped Kimi Top-K currently supports single-token decode only"
        )

    if loop_topk:
        gathered = prog.moe_gather_token_rows_from_vram_v0(
            routed_input,
            token_indices=[0],
            hidden=shape.routed_hidden,
            zero_row=constants.zero_row,
            policy_name="kimi_k3",
            name=f"{name}_loop_gather",
        )
        pair_gp, loop_gp = prog.register_allocator.allocate_gp(2)
        try:
            loop_start = IsaBuilder().comment(
                f"[kimi_k3] loop Top-{shape.top_k} routed expert pairs"
            )
            loop_start.instr("S_ADDI_INT", gp(pair_gp), gp(0), 0)
            loop_start.instr("C_LOOP_START", gp(loop_gp), shape.top_k)
            prog.emit(loop_start.render())
            expert_out = prog.moe_dynamic_expert_pair_v0(
                gathered,
                templates,
                weight_table_bases=table_bases,
                weight_table_strides=table_strides,
                weight_tile_group_strides=tile_group_strides,
                num_experts=shape.num_experts,
                expert_indices_int_base=int_sram_base,
                weights_fp_base=topk_weights.address,
                pair_idx=0,
                pair_index_gp=pair_gp,
                bias_tables=None,
                rows=prog.blen,
                intermediate=shape.intermediate,
                constants=constants.situ,
                zero_row=constants.zero_row,
                route_fp_scratch=route_scratch,
                policy_name="kimi_k3",
                activation_policy="kimi_situ",
                name=f"{name}_loop_pair",
            )
            prog.moe_scatter_add_active_rows_v0(
                routed_accumulator,
                expert_out,
                token_indices=[0],
                active_rows=[0],
                hidden=shape.routed_hidden,
                policy_name="kimi_k3",
                name=f"{name}_loop_scatter",
            )
            loop_end = IsaBuilder()
            loop_end.instr("S_ADDI_INT", gp(pair_gp), gp(pair_gp), 1)
            loop_end.instr("C_LOOP_END", gp(loop_gp))
            prog.emit(loop_end.render())
        finally:
            prog.register_allocator.free_gp([pair_gp, loop_gp])
        prog.free_tensor(gathered)
        prog.free_tensor(expert_out)
    else:
        for pair_idx in range(rows * shape.top_k):
            token_idx = pair_idx // shape.top_k
            gathered = prog.moe_gather_token_rows_from_vram_v0(
                routed_input,
                token_indices=[token_idx],
                hidden=shape.routed_hidden,
                zero_row=constants.zero_row,
                policy_name="kimi_k3",
                name=f"{name}_pair{pair_idx}_gather",
            )
            expert_out = prog.moe_dynamic_expert_pair_v0(
                gathered,
                templates,
                weight_table_bases=table_bases,
                weight_table_strides=table_strides,
                weight_tile_group_strides=tile_group_strides,
                num_experts=shape.num_experts,
                expert_indices_int_base=int_sram_base,
                weights_fp_base=topk_weights.address,
                pair_idx=pair_idx,
                bias_tables=None,
                rows=prog.blen,
                intermediate=shape.intermediate,
                constants=constants.situ,
                zero_row=constants.zero_row,
                route_fp_scratch=route_scratch,
                policy_name="kimi_k3",
                activation_policy="kimi_situ",
                name=f"{name}_pair{pair_idx}",
            )
            prog.moe_scatter_add_active_rows_v0(
                routed_accumulator,
                expert_out,
                token_indices=[token_idx],
                active_rows=[0],
                hidden=shape.routed_hidden,
                policy_name="kimi_k3",
                name=f"{name}_pair{pair_idx}_scatter",
            )
            prog.free_tensor(gathered)
            prog.free_tensor(expert_out)

    prog.rms_norm(
        routed_accumulator,
        eps_offset=constants.routed_norm_eps,
        reci_hid_offset=constants.routed_norm_reciprocal_hidden,
    )
    if routed_norm_weight is not None:
        prog.vram_mul(routed_accumulator, routed_norm_weight, num_rows=rows)
    routed_out = prog.linear_projection(
        routed_accumulator,
        weights.routed_up,
        name=f"{name}_latent_up",
        physical_shape=hidden.physical_shape,
    )

    shared_out = prog.moe_shared_expert_v0(
        moe_input,
        weights.shared,
        rows=rows,
        intermediate=shape.shared_intermediate,
        constants=constants.situ,
        zero_row=constants.zero_row,
        activation_policy="kimi_situ",
        policy_name="kimi_k3",
        name=f"{name}_shared",
    )
    prog.moe_combine_shared_and_routed_v0(
        routed_out,
        shared_out,
        rows=rows,
        policy_name="kimi_k3",
        name=f"{name}_combine",
    )
    if residual is not None:
        prog.vram_add(routed_out, residual, num_rows=rows)
    for temporary in (
        moe_input,
        logits,
        routed_input,
        routed_accumulator,
        shared_out,
    ):
        prog.free_tensor(temporary)
    if residual is not None:
        prog.free_tensor(residual)
    prog.free_fp_var(topk_weights)
    prog.free_fp_var(route_scratch)
    prog.emit(f"; {moe_end_marker(f'{name} complete')}\n")
    return routed_out


__all__ = [
    "AttnResConstants",
    "MlaBlockShape",
    "MlaBlockWeights",
    "MlaNormConstants",
    "ExpertWeightTable",
    "KimiLatentMoeShape",
    "KimiLatentMoeWeights",
    "KimiLatentMoeConstants",
    "emit_mla_residual_block",
    "emit_kimi_attn_res",
    "emit_kimi_dense_ffn_residual_block",
    "emit_kimi_latent_moe_residual_block",
    "reserve_expert_weight_table",
]
