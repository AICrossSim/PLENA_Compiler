"""Connected Nemotron 3 Attention and MoE decoder blocks.

The emitters follow the pinned ``modeling_nemotron_h.py`` implementation:

* GQA projects 32 query heads and two K/V heads, then reuses each K/V head for
  16 query heads.
* Routed and shared experts are two-matrix MLPs with ``relu(x) ** 2``.  They are
  not SwiGLU/SiTU experts and therefore cannot reuse Kimi's three-matrix path.
* Router weights are ``sigmoid(selected) / sum * 2.5`` while correction bias is
  used for expert selection only.
"""

from __future__ import annotations

from dataclasses import dataclass

from compiler.aten.plena import (
    DecodeCacheTensor,
    ExpertWeightTable,
    FPVar,
    InputVar,
    PlenaCompiler,
    VRAMMatrixVar,
    allocate_decode_cache_tensor,
)
from compiler.aten.plena.program_routed_moe import moe_end_marker, moe_stage_marker


@dataclass(frozen=True)
class NemotronAttentionShape:
    hidden: int
    query_heads: int
    kv_heads: int
    head_dim: int

    @property
    def q_width(self) -> int:
        return self.query_heads * self.head_dim

    @property
    def kv_width(self) -> int:
        return self.kv_heads * self.head_dim

    def validate(self, mlen: int) -> None:
        if self.query_heads <= 0 or self.kv_heads <= 0:
            raise ValueError("attention head counts must be positive")
        if self.query_heads % self.kv_heads:
            raise ValueError("query_heads must be divisible by kv_heads")
        widths = (self.hidden, self.head_dim, self.q_width, self.kv_width)
        if any(width <= 0 or width % mlen for width in widths):
            raise ValueError(
                f"Nemotron attention widths must be positive MLEN multiples: {widths}"
            )


@dataclass(frozen=True)
class NemotronAttentionWeights:
    q: InputVar
    k: InputVar
    v: InputVar
    out: InputVar


@dataclass(frozen=True)
class NemotronGqaDecodeCache:
    """Persistent per-KV-head K/V tensors for incremental GQA decode."""

    keys: tuple[DecodeCacheTensor, ...]
    values: tuple[DecodeCacheTensor, ...]
    max_tokens: int

    @property
    def persistent_bytes(self) -> int:
        return sum(cache.byte_capacity for cache in (*self.keys, *self.values))

    @property
    def backings(self) -> tuple[InputVar, ...]:
        return tuple(cache.backing for cache in (*self.keys, *self.values))


def allocate_nemotron_gqa_decode_cache(
    prog: PlenaCompiler,
    *,
    shape: NemotronAttentionShape,
    max_tokens: int,
    name: str = "nemotron_gqa_cache",
) -> NemotronGqaDecodeCache:
    """Allocate BF16 row-major cache tensors for all Nemotron K/V heads."""
    shape.validate(prog.mlen)
    keys = tuple(
        allocate_decode_cache_tensor(
            prog,
            name=f"{name}_k_head{head}",
            max_tokens=max_tokens,
            width=shape.head_dim,
        )
        for head in range(shape.kv_heads)
    )
    values = tuple(
        allocate_decode_cache_tensor(
            prog,
            name=f"{name}_v_head{head}",
            max_tokens=max_tokens,
            width=shape.head_dim,
        )
        for head in range(shape.kv_heads)
    )
    return NemotronGqaDecodeCache(
        keys=keys,
        values=values,
        max_tokens=max_tokens,
    )


@dataclass(frozen=True)
class NemotronMoeShape:
    hidden: int
    intermediate: int
    shared_intermediate: int
    num_experts: int
    top_k: int
    routed_scale: float = 2.5

    def validate(self, mlen: int) -> None:
        widths = (self.hidden, self.intermediate, self.shared_intermediate)
        if any(width <= 0 or width % mlen for width in widths):
            raise ValueError(
                f"Nemotron MoE widths must be positive MLEN multiples: {widths}"
            )
        if not 0 < self.top_k <= self.num_experts:
            raise ValueError(
                f"top_k={self.top_k} must be in [1, num_experts={self.num_experts}]"
            )
        if self.routed_scale <= 0:
            raise ValueError("routed_scale must be positive")


@dataclass(frozen=True)
class NemotronMoeWeights:
    router: InputVar
    routed_up: ExpertWeightTable
    routed_down: ExpertWeightTable
    shared_up: InputVar
    shared_down: InputVar


@dataclass(frozen=True)
class NemotronMoeConstants:
    zero_row: FPVar
    routed_scale: FPVar


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
            f"{name}: column view must be MLEN-aligned, got {col_offset}/{width}"
        )
    if col_offset + width > source.shape[1]:
        raise ValueError(f"{name}: view exceeds {source.name} width={source.shape[1]}")
    physical_rows = source.physical_shape[0]
    base = (
        prog.get_vram_addr(source.name)
        + (col_offset // prog.mlen) * physical_rows * prog.mlen
    )
    return prog.alloc_at(
        name,
        rows=source.shape[0],
        cols=width,
        vram_addr=base,
        physical_shape=(physical_rows, width),
    )


def emit_nemotron_attention_block(
    prog: PlenaCompiler,
    hidden: VRAMMatrixVar,
    *,
    shape: NemotronAttentionShape,
    weights: NemotronAttentionWeights,
    rows: int = 1,
    name: str = "nemotron_attention",
    cache: NemotronGqaDecodeCache | None = None,
    token_index: int | None = None,
    causal: bool = False,
) -> VRAMMatrixVar:
    """Emit connected GQA and return the mixer output.

    With ``cache=None`` this preserves the scratch path.  With a cache, decode
    appends one row while prefill appends prompt chunks at their global token
    offsets. The attention mask shifts its diagonal for every later chunk.
    """
    shape.validate(prog.mlen)
    if rows < 1 or rows > hidden.shape[0]:
        raise ValueError(f"{name}: rows={rows} outside hidden rows={hidden.shape[0]}")
    if hidden.shape[1] != shape.hidden:
        raise ValueError(
            f"{name}: hidden width={hidden.shape[1]} does not match {shape.hidden}"
        )
    if (cache is None) != (token_index is None):
        raise ValueError(f"{name}: cache and token_index must be provided together")
    if cache is not None:
        if len(cache.keys) != shape.kv_heads or len(cache.values) != shape.kv_heads:
            raise ValueError(f"{name}: cache head count does not match shape.kv_heads")
        if token_index + rows > cache.max_tokens:
            raise ValueError(
                f"{name}: token range [{token_index}, {token_index + rows}) "
                f"exceeds cache capacity={cache.max_tokens}"
            )

    prog.emit(f"; {moe_end_marker(f'{name} non-MoE region')}\n")
    projection_rows = max(prog.mlen, hidden.physical_shape[0])
    q_all = prog.linear_projection(
        hidden,
        weights.q,
        name=f"{name}_q",
        physical_shape=(projection_rows, shape.q_width),
    )
    k_all = prog.linear_projection(
        hidden,
        weights.k,
        name=f"{name}_k",
        physical_shape=(projection_rows, shape.kv_width),
    )
    v_all = prog.linear_projection(
        hidden,
        weights.v,
        name=f"{name}_v",
        physical_shape=(projection_rows, shape.kv_width),
    )
    attention = prog.alloc(
        f"{name}_heads",
        rows=hidden.shape[0],
        cols=shape.q_width,
        strict=False,
        physical_shape=(projection_rows, shape.q_width),
    )

    kv_inputs: list[tuple[InputVar, InputVar]] = []
    for kv_head in range(shape.kv_heads):
        k_head = _view_columns(
            prog,
            k_all,
            name=f"{name}_k_head{kv_head}",
            col_offset=kv_head * shape.head_dim,
            width=shape.head_dim,
        )
        v_head = _view_columns(
            prog,
            v_all,
            name=f"{name}_v_head{kv_head}",
            col_offset=kv_head * shape.head_dim,
            width=shape.head_dim,
        )
        if cache is None:
            kv_inputs.append(
                (
                    prog.store(k_head, name=f"{name}_k_scratch{kv_head}"),
                    prog.store(v_head, name=f"{name}_v_scratch{kv_head}"),
                )
            )
        else:
            cache.keys[kv_head].append_rows(
                prog,
                k_head,
                token_index=token_index,
                rows=rows,
                name=f"{name}_k_append_head{kv_head}",
            )
            cache.values[kv_head].append_rows(
                prog,
                v_head,
                token_index=token_index,
                rows=rows,
                name=f"{name}_v_append_head{kv_head}",
            )
            kv_inputs.append(
                (
                    cache.keys[kv_head].prefix(token_index + rows),
                    cache.values[kv_head].prefix(token_index + rows),
                )
            )
        prog.free_tensor(k_head)
        prog.free_tensor(v_head)

    heads_per_kv = shape.query_heads // shape.kv_heads
    for q_head_index in range(shape.query_heads):
        q_head = _view_columns(
            prog,
            q_all,
            name=f"{name}_q_head{q_head_index}",
            col_offset=q_head_index * shape.head_dim,
            width=shape.head_dim,
        )
        k_input, v_input = kv_inputs[q_head_index // heads_per_kv]
        head_out = prog.flash_attention(
            q_head,
            k_input,
            v_input,
            scale=shape.head_dim**-0.5,
            causal_mask=causal,
            batch_size=1,
            seq_len=rows,
            kv_seq_len=rows if cache is None else token_index + rows,
            k_matrix_precision="weights" if cache is None else "keyvalue",
            v_matrix_precision="weights" if cache is None else "keyvalue",
            k_hbm_element_bytes=1 if cache is None else 2,
            v_hbm_element_bytes=1 if cache is None else 2,
        )
        prog.vram_copy_region(
            attention,
            head_out,
            num_rows=rows,
            num_cols=shape.head_dim,
            dst_col_offset=q_head_index * shape.head_dim,
        )
        prog.free_tensor(q_head)
        prog.free_tensor(head_out)

    output = prog.linear_projection(
        attention,
        weights.out,
        name=f"{name}_out",
        physical_shape=hidden.physical_shape,
    )
    for temporary in (q_all, k_all, v_all, attention):
        prog.free_tensor(temporary)
    return output


def _relu2_in_place(
    prog: PlenaCompiler,
    value: VRAMMatrixVar,
    *,
    zero: FPVar,
    rows: int,
    width: int,
    stage: str,
    name: str,
) -> None:
    prog.emit(
        f"; {moe_stage_marker(stage, f'[nemotron3] relu2 {name}: rows={rows}')}\n"
    )
    for col_block in range(width // prog.mlen):
        prog.tile_row_max_fp(
            value,
            zero,
            rows=list(range(rows)),
            tile_col_idx=col_block,
        )
    prog.vram_mul(value, value, num_rows=rows)


def _dynamic_relu2_expert_pair(
    prog: PlenaCompiler,
    gathered: VRAMMatrixVar,
    *,
    weights: NemotronMoeWeights,
    shape: NemotronMoeShape,
    constants: NemotronMoeConstants,
    pair_idx: int,
    int_sram_base: int,
    weights_fp_base: int,
    route_scratch: FPVar,
    name: str,
) -> VRAMMatrixVar:
    projection_rows = max(prog.mlen, gathered.physical_shape[0])
    up = prog.moe_dynamic_linear_projection_v0(
        gathered,
        weights.routed_up.template,
        expert_indices_int_base=int_sram_base,
        pair_idx=pair_idx,
        table_base=weights.routed_up.base,
        per_expert_stride=weights.routed_up.stride,
        num_experts=weights.routed_up.num_experts,
        tile_group_stride=weights.routed_up.tile_group_stride,
        name=f"{name}_up",
        physical_shape=(projection_rows, shape.intermediate),
    )
    _relu2_in_place(
        prog,
        up,
        zero=constants.zero_row,
        rows=prog.blen,
        width=shape.intermediate,
        stage="expert_activation",
        name=name,
    )
    output = prog.moe_dynamic_linear_projection_v0(
        up,
        weights.routed_down.template,
        expert_indices_int_base=int_sram_base,
        pair_idx=pair_idx,
        table_base=weights.routed_down.base,
        per_expert_stride=weights.routed_down.stride,
        num_experts=weights.routed_down.num_experts,
        tile_group_stride=weights.routed_down.tile_group_stride,
        name=f"{name}_down",
        physical_shape=(projection_rows, shape.hidden),
    )
    route = prog.moe_materialize_topk_route_weight_v0(
        weights_fp_base=weights_fp_base,
        pair_idx=pair_idx,
        rows=prog.blen,
        hidden=shape.hidden,
        zero_row=constants.zero_row,
        fp_scratch=route_scratch,
        policy_name="nemotron3",
        name=f"{name}_route",
    )
    prog.emit(
        f"; {moe_stage_marker('expert_route_weight', f'[nemotron3] apply {name}')}\n"
    )
    prog.vram_mul(output, route, num_rows=prog.blen)
    prog.free_tensor(up)
    prog.free_tensor(route)
    return output


def emit_nemotron_moe_block(
    prog: PlenaCompiler,
    hidden: VRAMMatrixVar,
    *,
    shape: NemotronMoeShape,
    weights: NemotronMoeWeights,
    correction_bias: VRAMMatrixVar,
    constants: NemotronMoeConstants,
    rows: int = 1,
    int_sram_base: int = 0,
    name: str = "nemotron_moe",
) -> VRAMMatrixVar:
    """Emit official routed experts plus unweighted shared ReLU2 expert."""
    shape.validate(prog.mlen)
    if hidden.shape[1] != shape.hidden:
        raise ValueError(
            f"{name}: hidden width={hidden.shape[1]} does not match {shape.hidden}"
        )
    if rows < 1 or rows > hidden.shape[0]:
        raise ValueError(f"{name}: rows={rows} outside hidden rows={hidden.shape[0]}")
    for table in (weights.routed_up, weights.routed_down):
        if table.num_experts != shape.num_experts:
            raise ValueError("routed expert tables must match shape.num_experts")

    logits = prog.qwen3_router_logits_matrix_bf16_rowpacked_v0(
        hidden,
        weights.router,
        rows=rows,
        hidden=shape.hidden,
        num_experts=shape.num_experts,
        name=f"{name}_router",
    )
    topk_weights = prog.fp_var(f"{name}_topk_weights", rows * shape.top_k)
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
            policy_name="nemotron3",
            name=f"{name}_token{token_idx}",
        )
    if constants.routed_scale.size < shape.top_k:
        raise ValueError("routed_scale must provide one value per selected expert")
    for token_idx in range(rows):
        route_offset = token_idx * shape.top_k
        prog.fpvar_mul_region(
            topk_weights,
            constants.routed_scale,
            topk_weights,
            count=shape.top_k,
            src1_offset=route_offset,
            dst_offset=route_offset,
        )

    accumulator = prog.alloc(
        f"{name}_routed_accumulator",
        rows=hidden.shape[0],
        cols=shape.hidden,
        strict=False,
        physical_shape=hidden.physical_shape,
    )
    prog.moe_true_zero_vram_rows_v0(
        accumulator,
        rows=list(range(rows)),
        hidden=shape.hidden,
        zero_row=constants.zero_row,
        policy_name="nemotron3",
        stage="accumulator_init",
        name=f"{name}_zero",
    )
    route_scratch = prog.fp_var(f"{name}_route_scratch", prog.mlen)
    for pair_idx in range(rows * shape.top_k):
        token_idx = pair_idx // shape.top_k
        gathered = prog.moe_gather_token_rows_from_vram_v0(
            hidden,
            token_indices=[token_idx],
            hidden=shape.hidden,
            zero_row=constants.zero_row,
            policy_name="nemotron3",
            name=f"{name}_pair{pair_idx}_gather",
        )
        expert_out = _dynamic_relu2_expert_pair(
            prog,
            gathered,
            weights=weights,
            shape=shape,
            constants=constants,
            pair_idx=pair_idx,
            int_sram_base=int_sram_base,
            weights_fp_base=topk_weights.address,
            route_scratch=route_scratch,
            name=f"{name}_pair{pair_idx}",
        )
        prog.moe_scatter_add_active_rows_v0(
            accumulator,
            expert_out,
            token_indices=[token_idx],
            active_rows=[0],
            hidden=shape.hidden,
            policy_name="nemotron3",
            name=f"{name}_pair{pair_idx}_scatter",
        )
        prog.free_tensor(gathered)
        prog.free_tensor(expert_out)

    prog.emit(
        f"; {moe_stage_marker('shared_expert_projection', f'[nemotron3] {name} shared up')}\n"
    )
    shared = prog.linear_projection(
        hidden,
        weights.shared_up,
        name=f"{name}_shared_up",
        physical_shape=(hidden.physical_shape[0], shape.shared_intermediate),
    )
    _relu2_in_place(
        prog,
        shared,
        zero=constants.zero_row,
        rows=rows,
        width=shape.shared_intermediate,
        stage="shared_expert_activation",
        name=f"{name}_shared",
    )
    prog.emit(
        f"; {moe_stage_marker('shared_expert_projection', f'[nemotron3] {name} shared down')}\n"
    )
    shared_out = prog.linear_projection(
        shared,
        weights.shared_down,
        name=f"{name}_shared_down",
        physical_shape=hidden.physical_shape,
    )
    prog.moe_combine_shared_and_routed_v0(
        accumulator,
        shared_out,
        rows=rows,
        policy_name="nemotron3",
        name=f"{name}_combine",
    )

    for temporary in (logits, shared, shared_out):
        prog.free_tensor(temporary)
    prog.free_fp_var(topk_weights)
    prog.free_fp_var(route_scratch)
    prog.emit(f"; {moe_end_marker(f'{name} complete')}\n")
    return accumulator


__all__ = [
    "NemotronAttentionShape",
    "NemotronAttentionWeights",
    "NemotronMoeConstants",
    "NemotronMoeShape",
    "NemotronMoeWeights",
    "emit_nemotron_attention_block",
    "emit_nemotron_moe_block",
]
