"""Shape-only frontends that meter the production compiler lowering.

No model weights are loaded.  Inputs carry only logical/physical shape and HBM
ownership, then the ordinary :class:`PlenaCompiler` methods emit the final ISA
and its symbolic CostTrace together.  This keeps workload construction outside
the timing model while preserving the compiler as the sole source of opcode
and DMA work.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from pathlib import Path
import json
from typing import Any

import compiler.aten.ops as ops
from compiler.aten.model_extract import ModelConfig
from compiler.aten.ops.registry import Backend, OpRegistry
from compiler.aten.plena import PlenaCompiler
from compiler.aten.plena.cost_kernels import (
    projection_hbm_base_cost_summary,
    router_topk_cost_summary,
)
from compiler.aten.plena_frontend import (
    AttentionHeadPacking,
    LayerInputVars,
    _emit_attention_block,
    _emit_ffn_block,
    _emit_packed_attention_block,
)
from compiler.aten.program_sink import (
    COST_TRACE_GRANULARITIES,
    COST_TRACE_GRANULARITY_DETAILED,
    COST_TRACE_GRANULARITY_SUMMARY,
    CostTrace,
)


def _ceil(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _value(mapping: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return default


@dataclass(frozen=True)
class DecoderModelSpec:
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    num_hidden_layers: int = 1
    model_type: str = "qwen3"
    rms_norm_eps: float = 1e-5
    num_experts: int = 0
    experts_per_token: int = 0
    moe_intermediate_size: int | None = None

    @classmethod
    def load(cls, value: "DecoderModelSpec | Mapping[str, Any] | str | Path") -> "DecoderModelSpec":
        if isinstance(value, cls):
            return value
        if isinstance(value, (str, Path)):
            with Path(value).open() as handle:
                value = json.load(handle)
        if not isinstance(value, Mapping):
            raise TypeError("model specification must be a mapping or JSON path")
        hidden = int(_value(value, "hidden_size"))
        heads = int(_value(value, "num_attention_heads", "num_heads"))
        kv_heads = int(_value(value, "num_key_value_heads", "num_kv_heads", default=heads))
        return cls(
            hidden_size=hidden,
            intermediate_size=int(_value(value, "intermediate_size", default=4 * hidden)),
            num_attention_heads=heads,
            num_key_value_heads=kv_heads,
            head_dim=int(_value(value, "head_dim", default=hidden // heads)),
            num_hidden_layers=int(_value(value, "num_hidden_layers", "num_layers", default=1)),
            model_type=str(_value(value, "model_type", default="qwen3")),
            rms_norm_eps=float(_value(value, "rms_norm_eps", default=1e-5)),
            num_experts=int(_value(value, "num_experts", default=0) or 0),
            experts_per_token=int(
                _value(value, "num_experts_per_tok", "experts_per_token", default=0) or 0
            ),
            moe_intermediate_size=(
                None
                if _value(value, "moe_intermediate_size", "moe_inter_dim") is None
                else int(_value(value, "moe_intermediate_size", "moe_inter_dim"))
            ),
        )

    def validate(self) -> None:
        for name in (
            "hidden_size",
            "intermediate_size",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "num_hidden_layers",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")


@dataclass(frozen=True)
class CompilerHardwareSpec:
    mlen: int
    blen: int
    mram_tile_capacity: int = 4
    hlen: int | None = None
    broadcast_amount: int | None = None
    attention_head_packing: bool = False
    hbm_v_prefetch_amount: int = 4
    hbm_v_writeback_amount: int = 4

    def validate(self) -> None:
        if self.mlen <= 0 or self.blen <= 0 or self.mram_tile_capacity <= 0:
            raise ValueError("MLEN, BLEN and MRAM tile capacity must be positive")
        if self.mlen % self.blen:
            raise ValueError("MLEN must be divisible by BLEN")
        if self.attention_head_packing:
            if self.hlen is None or self.broadcast_amount is None:
                raise ValueError("packed attention requires HLEN and broadcast_amount")
            if self.hlen * self.broadcast_amount != self.mlen:
                raise ValueError("packed attention requires HLEN*broadcast_amount == MLEN")


@dataclass(frozen=True)
class RoutingHistogram:
    """Explicit routed-token counts, one entry per expert."""

    routes_per_expert: tuple[int, ...]
    token_count: int
    top_k: int
    label: str = "explicit"

    def __post_init__(self) -> None:
        if self.token_count <= 0 or self.top_k <= 0:
            raise ValueError("token_count and top_k must be positive")
        if not self.routes_per_expert or any(count < 0 for count in self.routes_per_expert):
            raise ValueError("routes_per_expert must be a non-empty nonnegative histogram")
        expected = self.token_count * self.top_k
        if sum(self.routes_per_expert) != expected:
            raise ValueError(
                f"routing histogram has {sum(self.routes_per_expert)} routes, expected {expected}"
            )

    @classmethod
    def balanced(cls, *, token_count: int, top_k: int, num_experts: int) -> "RoutingHistogram":
        total = token_count * top_k
        base, remainder = divmod(total, num_experts)
        return cls(
            routes_per_expert=tuple(base + (expert < remainder) for expert in range(num_experts)),
            token_count=token_count,
            top_k=top_k,
            label="balanced-fixture",
        )

    @classmethod
    def skewed(
        cls,
        *,
        token_count: int,
        top_k: int,
        num_experts: int,
        hot_fraction: float = 0.5,
    ) -> "RoutingHistogram":
        if not 0 < hot_fraction < 1:
            raise ValueError("hot_fraction must lie in (0, 1)")
        total = token_count * top_k
        hot = int(total * hot_fraction)
        tail = total - hot
        base, remainder = divmod(tail, max(1, num_experts - 1))
        values = [hot]
        values.extend(base + (idx < remainder) for idx in range(num_experts - 1))
        return cls(tuple(values), token_count, top_k, label="skewed-fixture")


@dataclass(frozen=True)
class CompilerTraceResult:
    trace: CostTrace
    assembly: str | None
    metadata: dict[str, Any]


def _model_config(spec: DecoderModelSpec) -> ModelConfig:
    return ModelConfig(
        hidden_size=spec.hidden_size,
        inter_dim=spec.intermediate_size,
        num_heads=spec.num_attention_heads,
        num_kv_heads=spec.num_key_value_heads,
        head_dim=spec.head_dim,
        eps=spec.rms_norm_eps,
        rope_theta=10_000.0,
        vocab_size=None,
        model_type=spec.model_type,
    )


def _register_dense_layer_inputs(
    program: PlenaCompiler,
    model: DecoderModelSpec,
    *,
    padded_hidden: int,
    padded_intermediate: int,
    padded_head_dim: int,
    total_q_dim: int,
) -> LayerInputVars:
    return LayerInputVars(
        w_q=program.input(
            "W_q_0", (padded_hidden, total_q_dim), memory_role="weight"
        ),
        w_o=program.input(
            "W_o_0", (total_q_dim, padded_hidden), memory_role="weight"
        ),
        w_k_heads=[
            program.input(
                f"W_k_0_h{head}",
                (padded_hidden, padded_head_dim),
                memory_role="weight",
            )
            for head in range(model.num_key_value_heads)
        ],
        w_v_heads=[
            program.input(
                f"W_v_0_h{head}",
                (padded_hidden, padded_head_dim),
                memory_role="weight",
            )
            for head in range(model.num_key_value_heads)
        ],
        w_gate=program.input(
            "W_gate_0", (padded_hidden, padded_intermediate), memory_role="weight"
        ),
        w_up=program.input(
            "W_up_0", (padded_hidden, padded_intermediate), memory_role="weight"
        ),
        w_down=program.input(
            "W_down_0", (padded_intermediate, padded_hidden), memory_role="weight"
        ),
    )


def compile_dense_decoder_trace(
    model_config: DecoderModelSpec | Mapping[str, Any] | str | Path,
    hardware: CompilerHardwareSpec,
    *,
    seq_len: int,
    batch_size: int = 1,
    num_layers: int | None = None,
    compiler_hash: str = "unknown",
    include_assembly: bool = False,
    cost_trace_granularity: str = COST_TRACE_GRANULARITY_DETAILED,
) -> CompilerTraceResult:
    """Lower one representative dense layer through the production compiler."""
    model = DecoderModelSpec.load(model_config)
    model.validate()
    hardware.validate()
    if seq_len <= 0 or batch_size <= 0:
        raise ValueError("seq_len and batch_size must be positive")
    if cost_trace_granularity not in COST_TRACE_GRANULARITIES:
        raise ValueError(
            f"unsupported cost_trace_granularity {cost_trace_granularity!r}; "
            f"expected one of {COST_TRACE_GRANULARITIES}"
        )
    if include_assembly and cost_trace_granularity == COST_TRACE_GRANULARITY_SUMMARY:
        raise ValueError("summary cost tracing cannot materialize full assembly")
    layers = model.num_hidden_layers if num_layers is None else int(num_layers)
    if layers <= 0:
        raise ValueError("num_layers must be positive")

    OpRegistry.load().set_backend(Backend.PLENA)
    mlen, blen = hardware.mlen, hardware.blen
    padded_seq = _ceil(seq_len, mlen if seq_len < mlen else blen)
    rows_per_batch = padded_seq if batch_size == 1 else _ceil(max(mlen, padded_seq), mlen)
    physical_rows = batch_size * rows_per_batch
    padded_hidden = _ceil(model.hidden_size, mlen)
    padded_intermediate = _ceil(model.intermediate_size, mlen)
    padded_head_dim = _ceil(model.head_dim, mlen)
    ratio = model.num_attention_heads // model.num_key_value_heads

    head_packing: AttentionHeadPacking | None = None
    if hardware.attention_head_packing:
        assert hardware.hlen is not None and hardware.broadcast_amount is not None
        if seq_len > mlen:
            raise ValueError("main packed-GQA lowering does not support sequence-tiled attention")
        if model.head_dim > hardware.hlen or ratio > hardware.broadcast_amount:
            raise ValueError("model heads do not fit the requested main packed-GQA layout")
        head_packing = AttentionHeadPacking(
            enabled=True,
            hlen=hardware.hlen,
            broadcast_amount=hardware.broadcast_amount,
            head_slot_dim=hardware.hlen,
            group_width=mlen,
            total_q_dim=model.num_key_value_heads * mlen,
        )
    total_q_dim = (
        head_packing.total_q_dim
        if head_packing is not None
        else model.num_attention_heads * padded_head_dim
    )

    program = PlenaCompiler(
        mlen=mlen,
        blen=blen,
        mram_tile_capacity=hardware.mram_tile_capacity,
        hbm_v_prefetch_amount=hardware.hbm_v_prefetch_amount,
        hbm_v_writeback_amount=hardware.hbm_v_writeback_amount,
        cost_trace=True,
        compiler_hash=compiler_hash,
        cost_trace_granularity=cost_trace_granularity,
        emit_assembly=include_assembly,
    )
    layer_inputs = _register_dense_layer_inputs(
        program,
        model,
        padded_hidden=padded_hidden,
        padded_intermediate=padded_intermediate,
        padded_head_dim=padded_head_dim,
        total_q_dim=total_q_dim,
    )
    x_input = program.input(
        "X", (physical_rows, padded_hidden), memory_role="activation"
    )
    pos_input = program.input(
        "POS", (physical_rows, padded_hidden), memory_role="activation"
    )
    rope_width = head_packing.group_width if head_packing is not None else padded_head_dim
    r_input = program.input("R_rope", (rope_width, rope_width), memory_role="weight")
    cos_input = program.input("COS", (physical_rows, rope_width), memory_role="activation")
    sin_input = program.input("SIN", (physical_rows, rope_width), memory_role="activation")
    mask_input = program.input("causal_mask", (mlen, mlen), memory_role="activation")

    with program.cost_stage("global/setup"):
        cos = program.load_batch(cos_input, name="COS")
        sin = program.load_batch(sin_input, name="SIN")
        causal_mask = program.load_batch(mask_input, name="CAUSAL_MASK")
        current = program.load_batch(x_input, name="X")
        pos = program.load_batch(pos_input, name="POS")
        ops.embedding_add(program, current, pos)

    scratch = program.alloc(
        "residual_scratch",
        physical_rows,
        padded_hidden,
        strict=False,
        physical_shape=(physical_rows, padded_hidden),
    )
    scale = 1.0 / math.sqrt(model.head_dim)
    with program.cost_stage("decoder/layer/attention"):
        if head_packing is not None:
            current = _emit_packed_attention_block(
                program,
                current,
                layer_inputs,
                (r_input, cos, sin),
                causal_mask,
                scratch,
                scale,
                0,
                padded_seq,
                model.head_dim,
                model.num_key_value_heads,
                ratio,
                head_packing,
                batch_size=batch_size,
                rows_per_batch=rows_per_batch,
                active_seq_len_per_batch=seq_len,
            )
        else:
            current = _emit_attention_block(
                program,
                current,
                layer_inputs,
                (r_input, cos, sin),
                causal_mask,
                scratch,
                scale,
                0,
                padded_seq,
                padded_head_dim,
                total_q_dim,
                model.num_attention_heads,
                model.num_key_value_heads,
                ratio,
                batch_size=batch_size,
                rows_per_batch=rows_per_batch,
                active_seq_len_per_batch=seq_len,
            )
    with program.cost_stage("decoder/layer/ffn"):
        current = _emit_ffn_block(program, current, layer_inputs, scratch)
    with program.cost_stage("decoder/final_norm"):
        ops.rms_norm(program, current, eps_offset=3, reci_hid_offset=4)

    trace = program.get_cost_trace(
        frontend="shape-only-dense-v1",
        model_type=model.model_type,
        representative_layer_count=1,
        requested_layer_count=layers,
        layer_scaling_required=layers != 1,
        seq_len=seq_len,
        batch_size=batch_size,
        materialized_block_pair_count=(
            0 if cost_trace_granularity == COST_TRACE_GRANULARITY_SUMMARY else None
        ),
        hardware={
            "mlen": mlen,
            "blen": blen,
            "mram_tile_capacity": hardware.mram_tile_capacity,
        },
    )
    metadata = {
        "model": model,
        "hardware": hardware,
        "physical_rows": physical_rows,
        "rows_per_batch": rows_per_batch,
        "padded_hidden": padded_hidden,
        "padded_intermediate": padded_intermediate,
        "padded_head_dim": padded_head_dim,
        "requested_layers": layers,
        "trace_layer_semantics": "one representative layer plus global setup/final norm",
    }
    return CompilerTraceResult(
        trace=trace,
        assembly=program.compile() if include_assembly else None,
        metadata=metadata,
    )


def compile_routed_moe_trace(
    model_config: DecoderModelSpec | Mapping[str, Any] | str | Path,
    hardware: CompilerHardwareSpec,
    routing: RoutingHistogram,
    *,
    compiler_hash: str = "unknown",
    include_assembly: bool = False,
    max_detailed_routes: int = 4096,
    cost_trace_granularity: str = COST_TRACE_GRANULARITY_DETAILED,
) -> CompilerTraceResult:
    """Meter main's routed expert/gather/scatter path for an explicit histogram.

    Detailed mode materializes small validation routes. Summary mode chunks
    routes to the fixed main FPRAM capacity and lowers each distinct bucket
    shape once. Neither mode invents a runtime routing distribution.
    """
    model = DecoderModelSpec.load(model_config)
    model.validate()
    hardware.validate()
    if cost_trace_granularity not in COST_TRACE_GRANULARITIES:
        raise ValueError(
            f"unsupported cost_trace_granularity {cost_trace_granularity!r}; "
            f"expected one of {COST_TRACE_GRANULARITIES}"
        )
    if include_assembly and cost_trace_granularity == COST_TRACE_GRANULARITY_SUMMARY:
        raise ValueError("summary cost tracing cannot materialize full assembly")
    if model.num_experts <= 0 or model.moe_intermediate_size is None:
        raise ValueError("routed-MoE tracing requires num_experts and moe_intermediate_size")
    if len(routing.routes_per_expert) != model.num_experts:
        raise ValueError("routing histogram expert count does not match the model")
    if routing.top_k != model.experts_per_token:
        raise ValueError("routing histogram top_k does not match the model")
    if (
        cost_trace_granularity == COST_TRACE_GRANULARITY_DETAILED
        and sum(routing.routes_per_expert) > max_detailed_routes
    ):
        raise ValueError("routing histogram is too large for detailed mode; use summary mode")

    OpRegistry.load().set_backend(Backend.PLENA)
    mlen, blen = hardware.mlen, hardware.blen
    hidden = _ceil(model.hidden_size, mlen)
    intermediate = _ceil(model.moe_intermediate_size, mlen)
    physical_tokens = max(blen, _ceil(routing.token_count, blen))
    program = PlenaCompiler(
        mlen=mlen,
        blen=blen,
        mram_tile_capacity=hardware.mram_tile_capacity,
        hbm_v_prefetch_amount=hardware.hbm_v_prefetch_amount,
        hbm_v_writeback_amount=hardware.hbm_v_writeback_amount,
        cost_trace=True,
        compiler_hash=compiler_hash,
        cost_trace_granularity=cost_trace_granularity,
        emit_assembly=include_assembly,
    )
    x_input = program.input(
        "moe_x", (physical_tokens, hidden), memory_role="activation"
    )
    router_weight = program.input(
        "router_weight", (hidden, model.num_experts),
        physical_shape=(hidden, _ceil(model.num_experts, mlen)),
        memory_role="weight",
    )
    expert_weights = [
        (
            program.input(f"expert{expert}_gate", (hidden, intermediate), memory_role="weight"),
            program.input(f"expert{expert}_up", (hidden, intermediate), memory_role="weight"),
            program.input(f"expert{expert}_down", (intermediate, hidden), memory_role="weight"),
        )
        for expert in range(model.num_experts)
    ]
    with program.cost_stage("decoder/moe/input"):
        x = program.load_batch(x_input, name="moe_x")
    max_expert_rows = max(1, max(routing.routes_per_expert) * blen)
    summary_mode = cost_trace_granularity == COST_TRACE_GRANULARITY_SUMMARY
    if summary_mode:
        # Production histograms can exceed the fixed main FPRAM.  V_TOPK uses
        # the true-zero row as reusable scratch; every later true-zero helper
        # rewrites that row before consuming it. Expert activation state is
        # allocated after the router and chunked to the remaining capacity.
        zero_row = program.fp_var("moe_zero_row", size=mlen)
        one = neg_one = None
    else:
        zero_row = program.fp_var("moe_zero_row", size=mlen)
        one = program.fp_var("moe_one", size=max_expert_rows)
        neg_one = program.fp_var("moe_neg_one", size=max_expert_rows)
    with program.cost_stage("decoder/moe/router"):
        logits = program.qwen3_router_logits_matrix_bf16_rowpacked_v0(
            x,
            router_weight,
            rows=routing.token_count,
            hidden=hidden,
            num_experts=model.num_experts,
            mram_tile_capacity=hardware.mram_tile_capacity,
            name="router_logits",
        )
        weights = (
            None
            if summary_mode
            else program.fp_var(
                "router_topk_weights",
                size=routing.token_count * routing.top_k,
            )
        )
        weights_fp_base = zero_row.address if summary_mode else weights.address
        indices = 0
        if summary_mode:
            expert_blocks = math.ceil(model.num_experts / mlen)
            topk_summary = router_topk_cost_summary(
                token_count=routing.token_count,
                top_k=routing.top_k,
                weights_fp_base=weights_fp_base,
                indices_int_base=indices,
                logits_base=program._vram_matrix_row_addr(logits, 0, 0),
                logits_token_stride=expert_blocks * mlen,
                weights_stride=0,
                indices_stride=0,
            )
            program.emit_cost_opcode_counts(
                topk_summary.opcodes,
                provenance="main-router-topk-v0",
            )
        else:
            for token in range(routing.token_count):
                program.gpt_oss_router_topk_softmax_v0(
                    logits,
                    token_idx=token,
                    weights_fp_base=weights_fp_base + token * routing.top_k,
                    indices_int_base=indices + token * routing.top_k,
                    num_experts=model.num_experts,
                    top_k=routing.top_k,
                    name=f"token{token}",
                )

    if summary_mode:
        fpram_capacity = program.fpram_allocator.total_size
        if mlen >= fpram_capacity:
            raise MemoryError(
                f"routed MoE requires a {mlen}-entry true-zero row in "
                f"main's {fpram_capacity}-entry FPRAM"
            )
        max_constant_rows = ((fpram_capacity - mlen) // (2 * blen)) * blen
        if max_constant_rows <= 0:
            raise MemoryError("main FPRAM cannot hold routed activation constants")
        max_constant_rows = min(max_expert_rows, max_constant_rows)
        one = program.fp_var("moe_one", size=max_constant_rows)
        neg_one = program.fp_var("moe_neg_one", size=max_constant_rows)
    else:
        max_constant_rows = max_expert_rows
    assert zero_row is not None and one is not None and neg_one is not None
    constants = (zero_row, zero_row, zero_row, one, neg_one)

    accumulator = program.alloc(
        "moe_accumulator",
        routing.token_count,
        hidden,
        strict=False,
        physical_shape=(physical_tokens, hidden),
    )
    with program.cost_stage("decoder/moe/combine_init"):
        if cost_trace_granularity == COST_TRACE_GRANULARITY_SUMMARY:
            program.gpt_oss_true_zero_vram_rows_summary_v0(
                accumulator,
                row_start=0,
                row_step=1,
                row_count=physical_tokens,
                hidden=hidden,
                zero_row=zero_row,
                name="accumulator_zero",
            )
        else:
            program.gpt_oss_true_zero_vram_rows_v0(
                accumulator,
                rows=range(physical_tokens),
                hidden=hidden,
                zero_row=zero_row,
                name="accumulator_zero",
            )

    def lower_expert(
        expert: int,
        route_count: int,
        route_start: int,
        token_indices: list[int] | None,
        chunk_index: int | None,
    ) -> None:
        expert_label = (
            f"expert{expert}"
            if chunk_index is None
            else f"expert{expert}_chunk{chunk_index}"
        )
        with program.cost_stage("decoder/moe/dispatch"):
            if cost_trace_granularity == COST_TRACE_GRANULARITY_SUMMARY:
                gathered = program.gpt_oss_gather_token_rows_from_vram_summary_v0(
                    x,
                    route_start=route_start,
                    route_count=route_count,
                    token_count=routing.token_count,
                    hidden=hidden,
                    zero_row=zero_row,
                    name=f"{expert_label}_gather",
                )
            else:
                assert token_indices is not None
                gathered = program.gpt_oss_gather_token_rows_from_vram_v0(
                    x,
                    token_indices=token_indices,
                    hidden=hidden,
                    zero_row=zero_row,
                    name=f"{expert_label}_gather",
                )
        projection_rows = max(mlen, route_count * blen)
        with program.cost_stage("decoder/moe/expert"):
            gate = program.linear_projection(
                gathered,
                expert_weights[expert][0],
                name=f"{expert_label}_gate_out",
                physical_shape=(projection_rows, intermediate),
            )
            up = program.linear_projection(
                gathered,
                expert_weights[expert][1],
                name=f"{expert_label}_up_out",
                physical_shape=(projection_rows, intermediate),
            )
            program.free_tensor(gathered)
            hidden_out = program.moe_expert_activation_v0(
                gate,
                up,
                rows=route_count * blen,
                intermediate=intermediate,
                constants=constants,
                activation_policy="standard_swiglu",
                name=expert_label,
            )
            expert_out = program.linear_projection(
                hidden_out,
                expert_weights[expert][2],
                name=f"{expert_label}_down_out",
                physical_shape=(projection_rows, hidden),
            )
            program.free_tensor(hidden_out)
            route_scale = program.alloc(
                f"{expert_label}_route_scale",
                projection_rows,
                hidden,
                strict=False,
                physical_shape=(projection_rows, hidden),
            )
            program.vram_mul(expert_out, route_scale, num_rows=route_count * blen)
            program.free_tensor(route_scale)
        with program.cost_stage("decoder/moe/combine"):
            if cost_trace_granularity == COST_TRACE_GRANULARITY_SUMMARY:
                program.gpt_oss_scatter_add_active_rows_summary_v0(
                    accumulator,
                    expert_out,
                    route_start=route_start,
                    route_count=route_count,
                    token_count=routing.token_count,
                    hidden=hidden,
                    name=f"{expert_label}_scatter",
                )
            else:
                assert token_indices is not None
                program.gpt_oss_scatter_add_active_rows_v0(
                    accumulator,
                    expert_out,
                    token_indices=token_indices,
                    active_rows=[route * blen for route in range(route_count)],
                    hidden=hidden,
                    name=f"{expert_label}_scatter",
                )
        program.free_tensor(expert_out)

    route_cursor = 0
    expert_chunk_count = 0
    max_routes_per_chunk = max(1, max_constant_rows // blen)
    expert_templates: dict[tuple[object, ...], tuple[int, int]] = {}

    def emit_expert_projection_base_cost(expert: int, route_count: int) -> None:
        projection_rows = max(mlen, route_count * blen)
        row_blocks = math.ceil(projection_rows / mlen)
        hidden_tiles = hidden // mlen
        intermediate_tiles = intermediate // mlen
        hidden_chunks = math.ceil(hidden_tiles / hardware.mram_tile_capacity)
        intermediate_chunks = math.ceil(
            intermediate_tiles / hardware.mram_tile_capacity
        )
        gate, up, down = expert_weights[expert]
        summary = projection_hbm_base_cost_summary(
            [
                (
                    program.get_hbm_layout(gate.name).hbm_base_addr,
                    row_blocks * intermediate_tiles * hidden_chunks,
                ),
                (
                    program.get_hbm_layout(up.name).hbm_base_addr,
                    row_blocks * intermediate_tiles * hidden_chunks,
                ),
                (
                    program.get_hbm_layout(down.name).hbm_base_addr,
                    row_blocks * hidden_tiles * intermediate_chunks,
                ),
            ]
        )
        with program.cost_stage("decoder/moe/expert"):
            program.emit_cost_opcode_counts(
                summary.opcodes,
                provenance="main-routed-expert-hbm-base-v0",
            )

    for expert, route_count in enumerate(routing.routes_per_expert):
        remaining = route_count
        chunk_index = 0
        while remaining:
            chunk_routes = (
                min(remaining, max_routes_per_chunk)
                if summary_mode
                else remaining
            )
            token_indices: list[int] | None = None
            if not summary_mode:
                token_indices = [
                    (route_cursor + route) % routing.token_count
                    for route in range(chunk_routes)
                ]
            emitted_chunk_index = (
                chunk_index if summary_mode and route_count > chunk_routes else None
            )
            if summary_mode:
                emit_expert_projection_base_cost(expert, chunk_routes)
                template_key = (
                    "routed-expert-chunk-v2",
                    chunk_routes,
                    hidden,
                    intermediate,
                    mlen,
                    blen,
                )
                representative = expert_templates.get(template_key)
                if representative is None:
                    program._suppress_summary_projection_hbm_base = True
                    try:
                        with program.cost_summary_template(template_key):
                            lower_expert(
                                expert,
                                chunk_routes,
                                route_cursor,
                                token_indices,
                                emitted_chunk_index,
                            )
                    finally:
                        program._suppress_summary_projection_hbm_base = False
                    representative_base = program.get_hbm_layout(
                        expert_weights[expert][0].name
                    ).hbm_base_addr
                    expert_templates[template_key] = (expert, representative_base)
                else:
                    _representative_expert, representative_base = representative
                    replay_base = program.get_hbm_layout(
                        expert_weights[expert][0].name
                    ).hbm_base_addr
                    if not program.replay_cost_summary_template(
                        template_key,
                        dma_address_delta_bytes=replay_base - representative_base,
                    ):
                        raise RuntimeError(
                            f"missing routed expert template {template_key!r}"
                        )
            else:
                lower_expert(
                    expert,
                    chunk_routes,
                    route_cursor,
                    token_indices,
                    emitted_chunk_index,
                )
            route_cursor += chunk_routes
            remaining -= chunk_routes
            chunk_index += 1
            expert_chunk_count += 1

    trace = program.get_cost_trace(
        frontend=(
            "shape-only-routed-moe-summary-v1"
            if summary_mode
            else "shape-only-routed-moe-detailed-v1"
        ),
        model_type=model.model_type,
        routing_label=routing.label,
        routing_histogram=list(routing.routes_per_expert),
        route_object_count=(
            0
            if cost_trace_granularity == COST_TRACE_GRANULARITY_SUMMARY
            else sum(routing.routes_per_expert)
        ),
        expanded_route_equivalent=sum(routing.routes_per_expert),
        expert_template_count=(
            len(expert_templates) if summary_mode else 0
        ),
        expert_chunk_count=expert_chunk_count,
        max_routes_per_chunk=max_routes_per_chunk,
        route_storage_mode=(
            "streamed-histogram-scratch"
            if summary_mode
            else "materialized-main-fpram"
        ),
        routing_semantics="explicit-expert-count-histogram",
        routing_order_semantics=(
            "affine-symbolic-routes"
            if summary_mode
            else "deterministic-validation-fixture"
        ),
        route_weight_semantics="shape-only-main-lowering",
        summary_fidelity=(
            "exact-algebraic-final-schedule"
            if summary_mode
            else "ordered-final-schedule"
        ),
    )
    return CompilerTraceResult(
        trace=trace,
        assembly=program.compile() if include_assembly else None,
        metadata={
            "model": model,
            "hardware": hardware,
            "routing": routing,
            "trace_mode": (
                "summary"
                if cost_trace_granularity == COST_TRACE_GRANULARITY_SUMMARY
                else "detailed"
            ),
        },
    )


__all__ = [
    "CompilerHardwareSpec",
    "CompilerTraceResult",
    "DecoderModelSpec",
    "RoutingHistogram",
    "compile_dense_decoder_trace",
    "compile_routed_moe_trace",
]
