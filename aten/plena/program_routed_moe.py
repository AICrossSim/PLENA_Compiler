"""Runtime-routed Qwen3-MoE program-builder substrate."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict

from compiler.aten.isa_builder import IsaBuilder, addr as areg, fp, gp
from compiler.aten.plena.packed_kv import PackedKVLayout
from compiler.aten.plena.vars import FPVar, InputVar, VRAMMatrixVar
from compiler.aten.qwen3_moe_runtime import (
    QWEN3_FULL_LAYER_DATAFLOW_STAGES,
    QWEN3_ROUTE_F32_SRAM_ENTRIES,
    QWEN3_ROUTE_SCORES_PER_TOKEN,
    QWEN3_MOE_RUNTIME_STAGES,
    Qwen3RawBf16ExpertBankLayout,
    RoutedMoeDecodePlan,
)

ExpertWeights = tuple[InputVar, InputVar, InputVar]
QwenSwiGLUConstants = tuple[FPVar, FPVar, FPVar]
MOE_STAGE_MARKER_PREFIX = "@stage="
MOE_END_STAGE = "non_moe"
MOE_STAGES = frozenset(
    (
        *QWEN3_MOE_RUNTIME_STAGES,
        "attention_residual",
        "post_attention_rmsnorm",
        "moe_residual",
        MOE_END_STAGE,
    )
)
QWEN_ROUTE_F32_SRAM_DEPTH = QWEN3_ROUTE_F32_SRAM_ENTRIES
QWEN_INT_SRAM_DEPTH = 1024


def moe_stage_marker(stage: str, detail: str = "") -> str:
    if stage not in MOE_STAGES:
        raise ValueError(f"unknown MoE stage {stage!r}")
    return f"{MOE_STAGE_MARKER_PREFIX}{stage}" + (
        f" {detail}" if detail else ""
    )


class ProgramRoutedMoeMixin:
    """Device-top-k and dynamic-expert lowering used by Qwen3 decode."""

    @staticmethod
    def _require_bf16_router_vector_format(vector_format: str) -> None:
        if str(vector_format).upper() != "BF16":
            raise RuntimeError(
                "executable Qwen3 router lowering requires global vector "
                "format BF16; this ISA has no per-stage precision switch, so "
                "narrow-vector router plans are analytic-only"
            )

    def _vram_matrix_row_addr(
        self, matrix: VRAMMatrixVar, row_idx: int, tile_col_idx: int = 0
    ) -> int:
        row_block = row_idx // self.mlen
        row_in_block = row_idx % self.mlen
        return (
            self.get_vram_tile_addr(matrix.name, row_block, tile_col_idx)
            + row_in_block * self.mlen
        )

    def emit_qwen3_moe_decode_contract(self, plan: RoutedMoeDecodePlan) -> None:
        """Emit auditable markers for every required layer and stage."""

        if len(plan.layers) != 48:
            raise ValueError("Qwen3-MoE contract must contain all 48 layers")
        frontend = plan.full_frontend_contract_receipt()
        self.emit_comment(
            "@moe_contract scope=compiler_substrate_only "
            "compiler_pipeline_valid=false emulator_valid=false"
        )
        self.emit_comment(
            "@qwen3_frontend_contract scope=contract_only layers=48 "
            "order=attention_then_runtime_moe expert_banks=128 "
            "dense_fallback=false executable=false sha256="
            f"{frontend['contract_sha256']}"
        )
        self.emit_comment(
            "@qwen3_frontend_order stages="
            + ">".join(QWEN3_FULL_LAYER_DATAFLOW_STAGES)
        )
        for layer in plan.layers:
            self.emit_comment(
                f"@qwen3_frontend_layer layer={layer.layer_index} "
                "attention_contract=native_decode "
                "moe_contract=runtime_topk expert_banks=128 topk=8 "
                f"expected_assignments={layer.expected_assignments} "
                "contract_only=true"
            )
            for stage in layer.stages:
                self.emit_comment(
                    moe_stage_marker(
                        stage,
                        f"layer={layer.layer_index} expected_assignments="
                        f"{layer.expected_assignments}",
                    )
                )
        self.emit_comment(moe_stage_marker(MOE_END_STAGE, "decode stack complete"))

    def qwen3_router_logits_bf16_v0(
        self,
        x: VRAMMatrixVar,
        router_weight_rows: VRAMMatrixVar,
        *,
        rows: int,
        hidden: int,
        num_experts: int = 128,
        router_vector_format: str,
        name: str = "qwen3_router_logits",
    ) -> VRAMMatrixVar:
        """Emit exact BF16 router GEMV followed by one BF16 logits cast."""

        self._require_bf16_router_vector_format(router_vector_format)
        if num_experts != 128:
            raise ValueError("exact Qwen3 router GEMV requires 128 experts")
        if hidden not in (64, 2048):
            raise ValueError("router GEMV supports hidden=64 validation or hidden=2048 target")
        if hidden % self.mlen:
            raise ValueError("router hidden size must be divisible by MLEN")
        if rows <= 0 or rows > x.shape[0]:
            raise ValueError(f"invalid router rows={rows} for x={x.shape}")
        if tuple(router_weight_rows.shape) != (num_experts, hidden):
            raise ValueError(
                "router BF16 weight matrix must have logical shape "
                f"({num_experts}, {hidden}), got {router_weight_rows.shape}"
            )
        if tuple(router_weight_rows.physical_shape) != (num_experts, hidden):
            raise ValueError(
                "router GEMV requires physical router layout "
                f"({num_experts}, {hidden}), got {router_weight_rows.physical_shape}"
            )
        expert_blocks = math.ceil(num_experts / self.mlen)
        logical_rows = rows if expert_blocks == 1 else rows * expert_blocks
        logits = self.alloc(
            name,
            rows=logical_rows,
            cols=num_experts if expert_blocks == 1 else self.mlen,
            strict=False,
            physical_shape=(
                max(self.blen, math.ceil(logical_rows / self.blen) * self.blen),
                self.mlen,
            ),
        )
        contiguous_x = self.alloc(
            f"{name}_contiguous_x",
            rows=1,
            cols=hidden,
            strict=False,
            physical_shape=(self.blen, hidden),
        )
        gp_src, gp_x, gp_w, gp_out = self._reg.allocate_gp(4)
        try:
            asm = IsaBuilder().comment(
                moe_stage_marker(
                    "router_bf16",
                    f"rows={rows} hidden={hidden} experts={num_experts}",
                )
            )
            asm.comment(
                "@moe_router_linear opcode=V_ROUTER_LINEAR_BF16 "
                "input=BF16 weights=BF16 output=BF16 "
                "transformers_5_5_fixture_parity=true cast_count=1 "
                "timing=structural_uncalibrated rtl_valid=false"
            )
            asm.instr(
                "S_ADDI_INT",
                gp(gp_w),
                gp(0),
                self._vram_matrix_row_addr(router_weight_rows, 0),
            )
            policy = 0 if hidden == 64 else 1
            for token_idx in range(rows):
                for col_block in range(hidden // self.mlen):
                    asm.instr(
                        "S_ADDI_INT",
                        gp(gp_x),
                        gp(0),
                        self._vram_matrix_row_addr(
                            contiguous_x, 0, col_block
                        ),
                    )
                    asm.instr(
                        "S_ADDI_INT",
                        gp(gp_src),
                        gp(0),
                        self._vram_matrix_row_addr(x, token_idx, col_block),
                    )
                    asm.instr("V_ADD_VF", gp(gp_x), gp(gp_src), fp(0), 0)
                asm.instr(
                    "S_ADDI_INT",
                    gp(gp_x),
                    gp(0),
                    self._vram_matrix_row_addr(contiguous_x, 0),
                )
                asm.instr(
                    "S_ADDI_INT",
                    gp(gp_out),
                    gp(0),
                    self._vram_matrix_row_addr(
                        logits, token_idx * expert_blocks
                    ),
                )
                asm.instr(
                    "V_ROUTER_LINEAR_BF16",
                    gp(gp_out),
                    gp(gp_x),
                    gp(gp_w),
                    policy,
                )
            self._emit(asm)
        finally:
            self._reg.free_gp([gp_src, gp_x, gp_w, gp_out])
        self.free_tensor(contiguous_x)
        return logits

    def qwen3_moe_router_topk8_v0(
        self,
        logits: VRAMMatrixVar,
        *,
        token_idx: int,
        route_f32_base: int,
        indices_int_base: int,
    ) -> None:
        """Write top-8 scores to FP32 route SRAM and IDs to integer SRAM."""

        expert_blocks = math.ceil(128 / self.mlen)
        required_rows = (token_idx + 1) * expert_blocks
        if token_idx < 0 or logits.shape[0] < required_rows:
            raise ValueError("router logits do not cover the requested token")
        if route_f32_base < 0 or route_f32_base + 8 > QWEN_ROUTE_F32_SRAM_DEPTH:
            raise ValueError("top-k output exceeds FP32 route SRAM")
        gp_weights, gp_logits, gp_indices = self._reg.allocate_gp(3)
        try:
            asm = IsaBuilder().comment(
                moe_stage_marker("topk8_runtime", f"token={token_idx}")
            )
            asm.comment(
                "@moe_route_isa scores=FP32 storage=route_sram "
                "timing=structural_uncalibrated rtl_valid=false"
            )
            asm.instr("S_ADDI_INT", gp(gp_weights), gp(0), route_f32_base)
            asm.instr(
                "S_ADDI_INT",
                gp(gp_logits),
                gp(0),
                self._vram_matrix_row_addr(
                    logits, token_idx * expert_blocks
                ),
            )
            asm.instr("S_ADDI_INT", gp(gp_indices), gp(0), indices_int_base)
            # rmask=1 is the architecture-defined 128-way/top-8 policy.
            asm.instr(
                "V_TOPK", gp(gp_weights), gp(gp_logits), gp(gp_indices), 1
            )
            self._emit(asm)
        finally:
            self._reg.free_gp([gp_weights, gp_logits, gp_indices])

    def _emit_runtime_expert_base(
        self,
        asm: IsaBuilder,
        *,
        indices_int_base: int,
        pair_idx: int,
        table_base: int,
        expert_stride: int,
        addr_reg: int,
        registers: Sequence[int],
        projection: str,
    ) -> None:
        if expert_stride <= 0:
            raise ValueError("expert HBM stride must be positive")
        gp_table, gp_expert, gp_stride, gp_offset, gp_base = registers
        asm.comment(
            moe_stage_marker(
                "dispatch_runtime_expert_id",
                f"pair={pair_idx} projection={projection}",
            )
        )
        asm.instr("S_ADDI_INT", gp(gp_table), gp(0), indices_int_base)
        asm.instr("S_LD_INT", gp(gp_expert), gp(gp_table), pair_idx)
        asm.instr("S_ADDI_INT", gp(gp_stride), gp(0), expert_stride)
        asm.instr("S_MUL_INT", gp(gp_offset), gp(gp_expert), gp(gp_stride))
        asm.instr("S_ADDI_INT", gp(gp_base), gp(0), table_base)
        asm.instr("S_ADD_INT", gp(gp_base), gp(gp_base), gp(gp_offset))
        asm.instr("C_SET_ADDR_REG", areg(addr_reg), gp(0), gp(gp_base))

    def _qwen3_dynamic_load_weight_col_v0(
        self,
        *,
        weight_template: InputVar,
        col_idx: int,
        indices_int_base: int,
        pair_idx: int,
        table_base: int,
        expert_stride: int,
        projection: str,
        k_block_start: int = 0,
        k_block_count: int | None = None,
    ) -> None:
        self._ensure_hbm_sub_matrix_registered(weight_template)
        layout = self.get_hbm_layout(weight_template.name)
        count = (
            layout.num_row_blocks
            if k_block_count is None
            else k_block_count
        )
        mram_name = (
            f"qwen3_{projection}_pair{pair_idx}_col{col_idx}_k{k_block_start}"
        )
        mram_addr = self.mram_allocator.allocate(
            mram_name, count * self.mlen * self.mlen
        )
        registers = self._reg.allocate_gp(8)
        gp_table, gp_expert, gp_expert_stride, gp_offset, gp_base, gp_scale, gp_stride, gp_mram = registers
        addr_reg = self._reg.allocate_addr(1)[0]
        try:
            asm = IsaBuilder().comment(
                moe_stage_marker(
                    f"expert_{projection}",
                    f"dynamic prefetch pair={pair_idx} col={col_idx}",
                )
            )
            self._emit_runtime_expert_base(
                asm,
                indices_int_base=indices_int_base,
                pair_idx=pair_idx,
                table_base=table_base,
                expert_stride=expert_stride,
                addr_reg=addr_reg,
                registers=(
                    gp_table,
                    gp_expert,
                    gp_expert_stride,
                    gp_offset,
                    gp_base,
                ),
                projection=projection,
            )
            self._emit_hbm_prefetch_setup(asm, layout, gp_scale, gp_stride)
            self._emit_hbm_subblock_sequence(
                asm,
                layout,
                (
                    (row_idx, col_idx)
                    for row_idx in range(k_block_start, k_block_start + count)
                ),
                mram_addr,
                addr_reg,
                gp_scale,
                gp_mram,
            )
            self._emit(asm)
        finally:
            self._reg.free_gp(registers)
            self._reg.free_addr([addr_reg])

    def qwen3_dynamic_projection_v0(
        self,
        x: VRAMMatrixVar,
        weight_template: InputVar,
        *,
        indices_int_base: int,
        pair_idx: int,
        table_base: int,
        expert_stride: int,
        projection: str,
        name: str,
    ) -> VRAMMatrixVar:
        """Project through a runtime-selected expert weight bank."""

        if projection not in {"gate", "up", "down"}:
            raise ValueError(f"invalid expert projection {projection!r}")
        x = self._require_var(x, VRAMMatrixVar, "x")
        weight_template = self._require_var(
            weight_template, InputVar, "weight_template"
        )
        physical_rows = max(self.mlen, x.physical_shape[0])
        output = self.alloc(
            name,
            rows=x.shape[0],
            cols=weight_template.shape[1],
            strict=False,
            physical_shape=(physical_rows, weight_template.physical_shape[1]),
        )
        physical_k = max(x.physical_shape[1], weight_template.physical_shape[0])
        num_k_tiles = math.ceil(physical_k / self.mlen)
        num_col_blocks = math.ceil(weight_template.physical_shape[1] / self.mlen)
        max_k_tiles = self.mram_tile_capacity
        temp = None
        for chunk_idx, k_start in enumerate(range(0, num_k_tiles, max_k_tiles)):
            k_count = min(max_k_tiles, num_k_tiles - k_start)
            for col_idx in range(num_col_blocks):
                super().reset_mram()
                self._qwen3_dynamic_load_weight_col_v0(
                    weight_template=weight_template,
                    col_idx=col_idx,
                    indices_int_base=indices_int_base,
                    pair_idx=pair_idx,
                    table_base=table_base,
                    expert_stride=expert_stride,
                    projection=projection,
                    k_block_start=k_start,
                    k_block_count=k_count,
                )
                target = output
                target_col = col_idx
                if chunk_idx:
                    if temp is None:
                        temp = self.alloc(
                            f"{name}_temp",
                            self.mlen,
                            self.mlen,
                            strict=False,
                            physical_shape=(self.mlen, self.mlen),
                        )
                    target = temp
                    target_col = 0
                self.emit_comment(
                    moe_stage_marker(
                        f"expert_{projection}",
                        f"pair={pair_idx} col={col_idx} k={k_start}:{k_count}",
                    )
                )
                super().vram_sub_projection_to(
                    vram_mat_name=x.name,
                    vram_row_idx=0,
                    mram_mat_name=weight_template.name,
                    mram_col_idx=col_idx,
                    target_matrix=target.name,
                    target_row_idx=0,
                    target_col_idx=target_col,
                    k_block_start=k_start,
                    k_block_count=k_count,
                )
                if chunk_idx:
                    self.vram_block_add_to(
                        output,
                        0,
                        col_idx,
                        temp,
                        0,
                        0,
                        output,
                        0,
                        col_idx,
                    )
        if temp is not None:
            self.free_tensor(temp)
        return output

    def qwen3_swiglu_v0(
        self,
        gate: VRAMMatrixVar,
        up: VRAMMatrixVar,
        *,
        rows: int,
        intermediate: int,
        constants: QwenSwiGLUConstants,
        name: str,
    ) -> VRAMMatrixVar:
        """Apply standard SiLU(gate)*up at the routed expert boundary."""

        zero, one, neg_one = constants
        if zero.address != 0:
            raise ValueError("Qwen SwiGLU requires the architectural f0 zero")
        if one.size < rows or neg_one.size < rows:
            raise ValueError("Qwen SwiGLU constants do not cover active rows")
        sigmoid = self.alloc(
            f"{name}_sigmoid",
            rows=rows,
            cols=intermediate,
            physical_shape=(max(self.mlen, gate.physical_shape[0]), intermediate),
            strict=False,
        )
        self.emit_comment(moe_stage_marker("expert_swiglu", name))
        self.vram_fill_zero(sigmoid, rows=range(rows))
        self.vram_add(sigmoid, gate, num_rows=rows)
        for col_block in range(math.ceil(intermediate / self.mlen)):
            self.tile_row_mul_fp(
                sigmoid, neg_one, rows=range(rows), tile_col_idx=col_block
            )
            self.tile_row_exp(sigmoid, rows=range(rows), tile_col_idx=col_block)
            self.tile_row_add_fp(
                sigmoid, one, rows=range(rows), tile_col_idx=col_block
            )
            self.tile_row_reci(sigmoid, rows=range(rows), tile_col_idx=col_block)
        self.vram_mul(gate, sigmoid, num_rows=rows)
        self.vram_mul(up, gate, num_rows=rows)
        self.free_tensor(sigmoid)
        return up

    def _qwen3_true_zero_rows_v0(
        self,
        matrix: VRAMMatrixVar,
        *,
        rows: Sequence[int],
        hidden: int,
        zero_row: FPVar,
        stage: str,
    ) -> None:
        if hidden % self.mlen:
            raise ValueError("zero-row width must be divisible by MLEN")
        row_list = [int(row) for row in rows]
        if row_list and (
            min(row_list) < 0 or max(row_list) >= matrix.physical_shape[0]
        ):
            raise ValueError("zero row exceeds the physical matrix")
        gp_fp, gp_dst, gp_loop = self._reg.allocate_gp(3)
        try:
            asm = IsaBuilder().comment(
                moe_stage_marker(stage, f"zero rows={row_list}")
            )
            asm.instr("S_ADDI_INT", gp(gp_fp), gp(0), zero_row.address)
            asm.instr("C_LOOP_START", gp(gp_loop), self.mlen)
            asm.instr("S_ST_FP", fp(0), gp(gp_fp), 0)
            asm.instr("S_ADDI_INT", gp(gp_fp), gp(gp_fp), 1)
            asm.instr("C_LOOP_END", gp(gp_loop))
            asm.instr("S_ADDI_INT", gp(gp_fp), gp(0), zero_row.address)
            for row_idx in row_list:
                for col_block in range(hidden // self.mlen):
                    asm.instr(
                        "S_ADDI_INT",
                        gp(gp_dst),
                        gp(0),
                        self._vram_matrix_row_addr(
                            matrix, row_idx, col_block
                        ),
                    )
                    asm.instr("S_MAP_V_FP", gp(gp_dst), gp(gp_fp), 0)
            self._emit(asm)
        finally:
            self._reg.free_gp([gp_fp, gp_dst, gp_loop])

    def qwen3_gather_pair_v0(
        self,
        x: VRAMMatrixVar,
        *,
        token_idx: int,
        hidden: int,
        zero_row: FPVar,
        name: str,
    ) -> VRAMMatrixVar:
        """Gather one known token row; only the expert id remains runtime data."""

        if not 0 <= token_idx < x.shape[0]:
            raise ValueError("token row is outside the decode batch")
        pair = self.alloc(
            name,
            rows=1,
            cols=hidden,
            strict=False,
            physical_shape=(self.blen, hidden),
        )
        self._qwen3_true_zero_rows_v0(
            pair,
            rows=range(self.blen),
            hidden=hidden,
            zero_row=zero_row,
            stage="dispatch_runtime_expert_id",
        )
        gp_dst, gp_src = self._reg.allocate_gp(2)
        try:
            asm = IsaBuilder().comment(
                moe_stage_marker(
                    "dispatch_runtime_expert_id", f"gather token={token_idx}"
                )
            )
            for col_block in range(hidden // self.mlen):
                asm.instr(
                    "S_ADDI_INT",
                    gp(gp_dst),
                    gp(0),
                    self._vram_matrix_row_addr(pair, 0, col_block),
                )
                asm.instr(
                    "S_ADDI_INT",
                    gp(gp_src),
                    gp(0),
                    self._vram_matrix_row_addr(x, token_idx, col_block),
                )
                # pair was zeroed, so ADD is an exact copy in BF16 VRAM.
                asm.instr("V_ADD_VV", gp(gp_dst), gp(gp_dst), gp(gp_src), 0)
            self._emit(asm)
        finally:
            self._reg.free_gp([gp_dst, gp_src])
        return pair

    def qwen3_apply_route_f32_v0(
        self,
        output: VRAMMatrixVar,
        *,
        route_f32_base: int,
        pair_idx: int,
        hidden: int,
    ) -> None:
        """Scale one expert output with its FP32 score, then cast once."""

        if tuple(output.shape) != (1, hidden):
            raise ValueError(
                f"route scaling expects output shape (1, {hidden}), got {output.shape}"
            )
        score_address = route_f32_base + pair_idx
        if not 0 <= score_address < QWEN_ROUTE_F32_SRAM_DEPTH:
            raise ValueError("route score address exceeds FP32 route SRAM")
        if hidden % self.mlen:
            raise ValueError("route-scaled output width must be divisible by MLEN")
        gp_dst, gp_src, gp_route = self._reg.allocate_gp(3)
        try:
            asm = IsaBuilder().comment(
                moe_stage_marker(
                    "route_weight",
                    f"pair={pair_idx} score=FP32 output_cast=BF16",
                )
            )
            asm.comment(
                "@moe_route_isa opcode=V_MUL_ROUTE_F32 "
                "timing=structural_uncalibrated rtl_valid=false"
            )
            asm.instr("S_ADDI_INT", gp(gp_route), gp(0), score_address)
            for col_block in range(hidden // self.mlen):
                address = self._vram_matrix_row_addr(output, 0, col_block)
                asm.instr(
                    "S_ADDI_INT",
                    gp(gp_dst),
                    gp(0),
                    address,
                )
                asm.instr(
                    "S_ADDI_INT", gp(gp_src), gp(0), address
                )
                asm.instr(
                    "V_MUL_ROUTE_F32",
                    gp(gp_dst),
                    gp(gp_src),
                    gp(gp_route),
                    0,
                )
            self._emit(asm)
        finally:
            self._reg.free_gp([gp_dst, gp_src, gp_route])

    def qwen3_rmsnorm_bf16_v0(
        self,
        source: VRAMMatrixVar,
        weight: VRAMMatrixVar,
        *,
        hidden: int,
        name: str,
        stage: str = "post_attention_rmsnorm",
    ) -> VRAMMatrixVar:
        """Emit exact Transformers 5.5 post-attention Qwen3 RMSNorm."""

        if hidden not in (64, 2048):
            raise ValueError("exact Qwen3 RMSNorm supports hidden=64 or hidden=2048")
        if tuple(source.shape) != (1, hidden):
            raise ValueError(f"Qwen3 RMSNorm source must have shape (1, {hidden})")
        if tuple(weight.shape) != (1, hidden):
            raise ValueError(f"Qwen3 RMSNorm weight must have shape (1, {hidden})")
        if source.physical_shape[0] != self.blen or weight.physical_shape[0] != self.blen:
            raise ValueError("Qwen3 RMSNorm requires BLEN-strided VRAM rows")
        output = self.alloc(
            name,
            rows=1,
            cols=hidden,
            strict=False,
            physical_shape=(self.blen, hidden),
        )
        gp_out, gp_source, gp_weight = self._reg.allocate_gp(3)
        try:
            policy = 0 if hidden == 64 else 1
            if stage == "post_attention_rmsnorm":
                marker = moe_stage_marker(stage, f"hidden={hidden}")
            elif stage in {"attention_rmsnorm", "q_rmsnorm", "k_rmsnorm"}:
                marker = f"@qwen3_stage={stage} hidden={hidden}"
            else:
                raise ValueError(f"unsupported Qwen3 RMSNorm stage {stage!r}")
            asm = IsaBuilder().comment(marker)
            asm.comment(
                "@qwen3_rmsnorm opcode=V_QWEN3_RMSNORM_BF16 "
                "variance=FP32 rsqrt=FP32 normalized_cast=BF16 affine=BF16 "
                "eps=1e-6 timing=structural_uncalibrated rtl_valid=false"
            )
            asm.instr(
                "S_ADDI_INT", gp(gp_out), gp(0), self._vram_matrix_row_addr(output, 0)
            )
            asm.instr(
                "S_ADDI_INT", gp(gp_source), gp(0), self._vram_matrix_row_addr(source, 0)
            )
            asm.instr(
                "S_ADDI_INT", gp(gp_weight), gp(0), self._vram_matrix_row_addr(weight, 0)
            )
            asm.instr(
                "V_QWEN3_RMSNORM_BF16",
                gp(gp_out),
                gp(gp_source),
                gp(gp_weight),
                policy,
            )
            self._emit(asm)
        finally:
            self._reg.free_gp([gp_out, gp_source, gp_weight])
        return output

    def _qwen3_copy_bf16_row_v0(
        self,
        source: VRAMMatrixVar,
        *,
        hidden: int,
        physical_rows: int,
        name: str,
    ) -> VRAMMatrixVar:
        """Copy one BF16 row by overwrite, independent of destination contents."""

        if tuple(source.shape) != (1, hidden) or hidden % self.mlen:
            raise ValueError("Qwen3 BF16 row copy requires one MLEN-tiled row")
        if physical_rows < 1 or physical_rows % self.blen:
            raise ValueError("Qwen3 BF16 row-copy padding must be BLEN aligned")
        target = self.alloc(
            name,
            rows=1,
            cols=hidden,
            strict=False,
            physical_shape=(physical_rows, hidden),
        )
        gp_dst, gp_src = self._reg.allocate_gp(2)
        try:
            asm = IsaBuilder().comment(
                "@qwen3_bf16_row_copy overwrite=true nan_independent=true"
            )
            for col_block in range(hidden // self.mlen):
                asm.instr(
                    "S_ADDI_INT",
                    gp(gp_dst),
                    gp(0),
                    self._vram_matrix_row_addr(target, 0, col_block),
                )
                asm.instr(
                    "S_ADDI_INT",
                    gp(gp_src),
                    gp(0),
                    self._vram_matrix_row_addr(source, 0, col_block),
                )
                asm.instr("V_ADD_VF", gp(gp_dst), gp(gp_src), fp(0), 0)
            self._emit(asm)
        finally:
            self._reg.free_gp([gp_dst, gp_src])
        return target

    def qwen3_exact_expert_combine_bf16_v0(
        self,
        source: VRAMMatrixVar,
        *,
        descriptor_layout: Qwen3RawBf16ExpertBankLayout,
        route_f32_base: int,
        indices_int_base: int,
        hidden: int,
        intermediate: int,
        name: str,
    ) -> VRAMMatrixVar:
        """Emit exact dynamic fused-expert execution from raw-BF16 HBM."""

        if route_f32_base != indices_int_base:
            raise ValueError(
                "exact expert-combine ABI requires a common FP32-route/INT-ID base"
            )
        if route_f32_base < 0 or route_f32_base + 8 > QWEN_ROUTE_F32_SRAM_DEPTH:
            raise ValueError("exact expert-combine route base exceeds SRAM")
        if route_f32_base + 8 > QWEN_INT_SRAM_DEPTH:
            raise ValueError("exact expert-combine ID base exceeds SRAM")
        if (descriptor_layout.hidden, descriptor_layout.intermediate) != (
            hidden,
            intermediate,
        ):
            raise ValueError("expert descriptor geometry does not match the transaction")
        if descriptor_layout.expert_count != 128:
            raise ValueError("exact expert-combine requires 128 expert banks")
        if descriptor_layout.descriptor_base > 0xFFFF_FFFF:
            raise ValueError("expert descriptor base exceeds the current 32-bit lowering ABI")
        if tuple(source.shape) != (1, hidden) or source.physical_shape[0] != self.blen:
            raise ValueError("exact expert-combine source must be one BLEN-strided row")
        output = self.alloc(
            name,
            rows=1,
            cols=hidden,
            strict=False,
            physical_shape=(self.blen, hidden),
        )
        gp_out, gp_source, gp_pairs, gp_descriptor = self._reg.allocate_gp(4)
        addr_reg = self._reg.allocate_addr(1)[0]
        try:
            asm = IsaBuilder().comment(
                moe_stage_marker(
                    "dispatch_runtime_expert_id",
                    "exact fused expert banks; ascending expert combine",
                )
            )
            asm.comment(
                "@qwen3_expert_combine opcode=V_QWEN3_EXPERT_COMBINE_BF16 "
                "hbm=raw_BF16 descriptor=Q3MOEBF1 topk=8 "
                "order=ascending_expert_id route_multiply=FP32 output_cast=BF16 "
                "timing=structural_uncalibrated rtl_valid=false"
            )
            asm.instr(
                "S_ADDI_INT",
                gp(gp_descriptor),
                gp(0),
                descriptor_layout.descriptor_base,
            )
            asm.instr(
                "C_SET_ADDR_REG", areg(addr_reg), gp(0), gp(gp_descriptor)
            )
            asm.instr(
                "S_ADDI_INT", gp(gp_out), gp(0), self._vram_matrix_row_addr(output, 0)
            )
            asm.instr(
                "S_ADDI_INT", gp(gp_source), gp(0), self._vram_matrix_row_addr(source, 0)
            )
            asm.instr("S_ADDI_INT", gp(gp_pairs), gp(0), route_f32_base)
            asm.instr(
                "V_QWEN3_EXPERT_COMBINE_BF16",
                gp(gp_out),
                gp(gp_source),
                gp(gp_pairs),
                addr_reg,
            )
            self._emit(asm)
        finally:
            self._reg.free_gp([gp_out, gp_source, gp_pairs, gp_descriptor])
            self._reg.free_addr([addr_reg])
        return output

    def qwen3_post_attention_moe_tail_transaction_v0(
        self,
        attention_output: VRAMMatrixVar,
        layer_residual: VRAMMatrixVar,
        post_attention_norm_weight: VRAMMatrixVar,
        router_weight_rows: VRAMMatrixVar,
        *,
        descriptor_layout: Qwen3RawBf16ExpertBankLayout,
        layer_index: int,
        route_f32_base: int = 0,
        indices_int_base: int = 0,
        hidden: int = 64,
        intermediate: int = 64,
        router_vector_format: str = "BF16",
        name: str = "qwen3_moe_tail",
        retain_intermediates: bool = False,
    ) -> tuple[VRAMMatrixVar, dict[str, object]]:
        """Lower one batch-1 post-attention residual/RMS/MoE transaction."""

        self._require_bf16_router_vector_format(router_vector_format)
        if not 0 <= layer_index < 48:
            raise ValueError("Qwen3 target layer index must be in [0, 48)")
        for label, tensor in (
            ("attention_output", attention_output),
            ("layer_residual", layer_residual),
        ):
            if tuple(tensor.shape) != (1, hidden):
                raise ValueError(f"{label} must have shape (1, {hidden})")
            if tensor.physical_shape[0] != self.blen:
                raise ValueError(f"{label} must use a BLEN-strided VRAM row")

        def allocate_copy(label: str, source: VRAMMatrixVar) -> VRAMMatrixVar:
            return self._qwen3_copy_bf16_row_v0(
                source,
                hidden=hidden,
                physical_rows=self.blen,
                name=label,
            )

        post_attention = allocate_copy(
            f"{name}_l{layer_index}_attention_residual", attention_output
        )
        self.vram_add(post_attention, layer_residual, num_rows=1)
        moe_residual = allocate_copy(
            f"{name}_l{layer_index}_moe_residual", post_attention
        )
        normalized = self.qwen3_rmsnorm_bf16_v0(
            post_attention,
            post_attention_norm_weight,
            hidden=hidden,
            name=f"{name}_l{layer_index}_normalized",
        )
        logits = self.qwen3_router_logits_bf16_v0(
            normalized,
            router_weight_rows,
            rows=1,
            hidden=hidden,
            num_experts=128,
            router_vector_format=router_vector_format,
            name=f"{name}_l{layer_index}_router",
        )
        self.qwen3_moe_router_topk8_v0(
            logits,
            token_idx=0,
            route_f32_base=route_f32_base,
            indices_int_base=indices_int_base,
        )
        combined = self.qwen3_exact_expert_combine_bf16_v0(
            normalized,
            descriptor_layout=descriptor_layout,
            route_f32_base=route_f32_base,
            indices_int_base=indices_int_base,
            hidden=hidden,
            intermediate=intermediate,
            name=f"{name}_l{layer_index}_combined",
        )
        output = allocate_copy(
            f"{name}_l{layer_index}_output", combined
        )
        self.vram_add(output, moe_residual, num_rows=1)

        def vram_receipt(tensor: VRAMMatrixVar) -> dict[str, object]:
            base = self.get_vram_addr(tensor.name)
            elements = tensor.physical_shape[0] * tensor.physical_shape[1]
            return {
                "base_element_address": base,
                "end_element_address_exclusive": base + elements,
                "logical_shape": list(tensor.shape),
                "physical_shape": list(tensor.physical_shape),
                "dtype": "BF16",
                "layout": "column-block-major BLEN-strided rows",
            }

        receipt: dict[str, object] = {
            "schema_version": 1,
            "scope": "post_attention_moe_tail",
            "layer_index": layer_index,
            "batch_size": 1,
            "packedkv_attention_prefix_executed": False,
            "packedkv_input_contract": (
                "attention output projection and layer residual are pre-staged BF16 VRAM"
            ),
            "stages_executed": [
                "attention_residual",
                "post_attention_rmsnorm",
                "router_bf16",
                "topk8_runtime",
                "dynamic_fused_experts",
                "fp32_route_combine",
                "moe_residual",
            ],
            "expert_hbm": descriptor_layout.as_dict(),
            "vram": {
                "attention_output": vram_receipt(attention_output),
                "layer_residual": vram_receipt(layer_residual),
                "post_attention_norm_weight": vram_receipt(
                    post_attention_norm_weight
                ),
                "router_weight_rows": vram_receipt(router_weight_rows),
                "attention_residual": vram_receipt(post_attention),
                "moe_residual": vram_receipt(moe_residual),
                "normalized": vram_receipt(normalized),
                "router_logits": vram_receipt(logits),
                "expert_combined": vram_receipt(combined),
                "output": vram_receipt(output),
            },
            "route_f32_base": route_f32_base,
            "indices_int_base": indices_int_base,
            "assignment_count": 8,
            "compiler_lowering_valid": True,
            "emulator_transaction_parity": False,
            "rtl_valid": False,
            "timing_calibrated": False,
            "publication_rankable": False,
        }
        if not retain_intermediates:
            self.free_tensor(post_attention)
            self.free_tensor(moe_residual)
            self.free_tensor(normalized)
            self.free_tensor(logits)
            self.free_tensor(combined)
        return output, receipt

    def qwen3_tiny_single_layer_decode_transaction_v0(
        self,
        layer_input: VRAMMatrixVar,
        attention_norm_weight: VRAMMatrixVar,
        q_norm_weight: VRAMMatrixVar,
        k_norm_weight: VRAMMatrixVar,
        rope_cos: VRAMMatrixVar,
        rope_sin: VRAMMatrixVar,
        post_attention_norm_weight: VRAMMatrixVar,
        router_weight_rows: VRAMMatrixVar,
        q_weight: InputVar,
        k_weight: InputVar,
        v_weight: InputVar,
        rotate_half_weight: InputVar,
        o_weight: InputVar,
        k_cache: InputVar,
        v_cache: InputVar,
        *,
        packed_layout: PackedKVLayout,
        descriptor_layout: Qwen3RawBf16ExpertBankLayout,
        cache_position: int = 3,
        layer_index: int = 0,
        route_f32_base: int = 0,
        indices_int_base: int = 0,
        name: str = "qwen3_tiny_single_layer_decode",
    ) -> tuple[VRAMMatrixVar, dict[str, object]]:
        """Lower the sealed hidden-64 q_len=1 attention plus routed-MoE proof.

        This is intentionally a validation transaction, not a target-model
        frontend.  Every precision and physical-layout transition is explicit,
        and all semantic intermediates remain live for an emulator dump.
        """

        if (self.mlen, self.blen, self.hlen, self.broadcast_amount) != (
            64,
            4,
            64,
            1,
        ):
            raise ValueError(
                "tiny Qwen3 decode transaction requires "
                "MLEN=64, BLEN=4, HLEN=64, BROADCAST_AMOUNT=1"
            )
        if isinstance(cache_position, bool) or not isinstance(cache_position, int):
            raise TypeError("cache_position must be an integer")
        if cache_position != 3:
            raise ValueError("sealed tiny Qwen3 decode fixture requires cache_position=3")
        if not isinstance(packed_layout, PackedKVLayout):
            raise TypeError("packed_layout must be a PackedKVLayout")
        expected_packed = PackedKVLayout(
            kv_heads=1,
            head_dim=64,
            mlen=64,
            block_size=8,
            element_bits=8,
            scale_bits=8,
        )
        if packed_layout != expected_packed:
            raise ValueError(
                "sealed tiny Qwen3 decode fixture requires one 64-wide "
                "MXFP8 PackedKV head"
            )
        if (descriptor_layout.hidden, descriptor_layout.intermediate) != (64, 64):
            raise ValueError("tiny Qwen3 decode expert bank must be hidden=intermediate=64")

        bf16_rows = (
            ("layer_input", layer_input, (1, 64), (4, 64)),
            ("attention_norm_weight", attention_norm_weight, (1, 64), (4, 64)),
            ("q_norm_weight", q_norm_weight, (1, 64), (4, 64)),
            ("k_norm_weight", k_norm_weight, (1, 64), (4, 64)),
            ("rope_cos", rope_cos, (1, 64), (4, 64)),
            ("rope_sin", rope_sin, (1, 64), (4, 64)),
            (
                "post_attention_norm_weight",
                post_attention_norm_weight,
                (1, 64),
                (4, 64),
            ),
            ("router_weight_rows", router_weight_rows, (128, 64), (128, 64)),
        )
        for label, tensor, logical, physical in bf16_rows:
            if not isinstance(tensor, VRAMMatrixVar):
                raise TypeError(f"{label} must be a VRAMMatrixVar")
            if tuple(tensor.shape) != logical or tuple(tensor.physical_shape) != physical:
                raise ValueError(
                    f"{label} must have logical/physical shapes {logical}/{physical}"
                )

        mx_tensors = (
            ("q_weight", q_weight, "weight"),
            ("k_weight", k_weight, "weight"),
            ("v_weight", v_weight, "weight"),
            ("rotate_half_weight", rotate_half_weight, "weight"),
            ("o_weight", o_weight, "weight"),
            ("k_cache", k_cache, "key"),
            ("v_cache", v_cache, "value"),
        )
        hbm_ranges: list[tuple[str, int, int]] = []
        for label, tensor, role in mx_tensors:
            if not isinstance(tensor, InputVar):
                raise TypeError(f"{label} must be an InputVar")
            if tuple(tensor.physical_shape) != (64, 64):
                raise ValueError(f"{label} must have physical shape (64, 64)")
            if tuple(tensor.shape) != (64, 64):
                raise ValueError(f"{label} must have logical shape (64, 64)")
            layout = self.get_hbm_layout(tensor.name)
            if (
                layout.precision_role != role
                or layout.hbm_element_width != 8
                or layout.hbm_block_size != 8
                or layout.hbm_scale_width != 8
            ):
                raise ValueError(
                    f"{label} must use role={role} MXFP8 E4M3/block8/scale8 storage"
                )
            hbm_ranges.append(
                (label, layout.hbm_base_addr, layout.hbm_base_addr + layout.hbm_size)
            )
        for label, start, end in hbm_ranges:
            if start < descriptor_layout.end_address and end > descriptor_layout.descriptor_base:
                raise ValueError(
                    f"expert descriptor/image overlaps HBM tensor {label}: "
                    f"[{start},{end})"
                )

        def allocate_padded_copy(
            label: str, source: VRAMMatrixVar
        ) -> VRAMMatrixVar:
            return self._qwen3_copy_bf16_row_v0(
                source,
                hidden=64,
                physical_rows=64,
                name=label,
            )

        def blen_view(label: str, source: VRAMMatrixVar) -> VRAMMatrixVar:
            return self.alloc_at(
                label,
                1,
                64,
                self.get_vram_addr(source.name),
                physical_shape=(4, 64),
            )

        self.emit_comment(
            "@qwen3_tiny_single_layer_decode schema=1 q_len=1 hidden=64 "
            "kv_heads=1 cache_position=3 scope=validation_only"
        )
        attention_normalized = self.qwen3_rmsnorm_bf16_v0(
            layer_input,
            attention_norm_weight,
            hidden=64,
            name=f"{name}_attention_normalized",
            stage="attention_rmsnorm",
        )
        attention_normalized_padded = allocate_padded_copy(
            f"{name}_attention_normalized_padded", attention_normalized
        )
        q_projected = self.linear_projection(
            attention_normalized_padded,
            q_weight,
            name=f"{name}_q_projected",
            physical_shape=(64, 64),
        )
        k_projected = self.linear_projection(
            attention_normalized_padded,
            k_weight,
            name=f"{name}_k_projected",
            physical_shape=(64, 64),
        )
        v_projected = self.linear_projection(
            attention_normalized_padded,
            v_weight,
            name=f"{name}_v_projected",
            physical_shape=(64, 64),
        )
        q_projected_view = blen_view(f"{name}_q_projected_blen", q_projected)
        k_projected_view = blen_view(f"{name}_k_projected_blen", k_projected)
        q_normalized = self.qwen3_rmsnorm_bf16_v0(
            q_projected_view,
            q_norm_weight,
            hidden=64,
            name=f"{name}_q_normalized",
            stage="q_rmsnorm",
        )
        k_normalized = self.qwen3_rmsnorm_bf16_v0(
            k_projected_view,
            k_norm_weight,
            hidden=64,
            name=f"{name}_k_normalized",
            stage="k_rmsnorm",
        )
        q_normalized_padded = allocate_padded_copy(
            f"{name}_q_normalized_padded", q_normalized
        )
        k_normalized_padded = allocate_padded_copy(
            f"{name}_k_normalized_padded", k_normalized
        )
        q_rotated = self.linear_projection(
            q_normalized_padded,
            rotate_half_weight,
            name=f"{name}_q_rotate_half",
            physical_shape=(64, 64),
        )
        k_rotated = self.linear_projection(
            k_normalized_padded,
            rotate_half_weight,
            name=f"{name}_k_rotate_half",
            physical_shape=(64, 64),
        )
        self.rope(q_normalized_padded, q_rotated, rope_cos, rope_sin)
        self.rope(k_normalized_padded, k_rotated, rope_cos, rope_sin)
        q_rope = q_normalized_padded
        k_rope = k_normalized_padded

        k_append = self.append_packed_kv_batch(
            k_rope,
            k_cache,
            cache_position=cache_position,
            batch_size=1,
            source_rows_per_batch=64,
            cache_rows_per_batch=64,
            packed_layout=packed_layout,
            role="key",
        )
        v_append = self.append_packed_kv_batch(
            v_projected,
            v_cache,
            cache_position=cache_position,
            batch_size=1,
            source_rows_per_batch=64,
            cache_rows_per_batch=64,
            packed_layout=packed_layout,
            role="value",
        )

        attention_output = self.alloc(
            f"{name}_attention_output",
            rows=1,
            cols=64,
            strict=False,
            physical_shape=(64, 64),
        )
        self.vram_fill_zero(attention_output)
        scratch = self.alloc(
            f"{name}_attention_scratch",
            rows=128,
            cols=64,
            strict=True,
            physical_shape=(128, 64),
        )
        self.flash_attention_packed_cache(
            q_rope,
            k_cache,
            v_cache,
            num_kv_heads=1,
            group_heads=1,
            head_slot_dim=64,
            output_base_address=self.get_vram_addr(attention_output.name),
            scratch_base_address=self.get_vram_addr(scratch.name),
            broadcast_amount=1,
            scale=0.125,
            causal_mask=False,
            valid_cols=cache_position + 1,
            cache_position=cache_position,
            batch_size=1,
            rows_per_batch=64,
            query_rows_per_batch=1,
            cache_rows_per_batch=64,
            kv_head_reuse=False,
        )
        o_projected = self.linear_projection(
            attention_output,
            o_weight,
            name=f"{name}_o_projected",
            physical_shape=(64, 64),
        )
        o_projected_view = blen_view(f"{name}_o_projected_blen", o_projected)
        output, tail_receipt = self.qwen3_post_attention_moe_tail_transaction_v0(
            o_projected_view,
            layer_input,
            post_attention_norm_weight,
            router_weight_rows,
            descriptor_layout=descriptor_layout,
            layer_index=layer_index,
            route_f32_base=route_f32_base,
            indices_int_base=indices_int_base,
            hidden=64,
            intermediate=64,
            router_vector_format="BF16",
            name=f"{name}_tail",
            retain_intermediates=True,
        )

        def vram_receipt(tensor: VRAMMatrixVar) -> dict[str, object]:
            base = self.get_vram_addr(tensor.name)
            elements = tensor.physical_shape[0] * tensor.physical_shape[1]
            return {
                "base_element_address": base,
                "end_element_address_exclusive": base + elements,
                "logical_shape": list(tensor.shape),
                "physical_shape": list(tensor.physical_shape),
                "dtype": "BF16",
                "layout": "column-block-major MLEN-vector rows",
            }

        boundaries = {
            "layer_input": vram_receipt(layer_input),
            "attention_normalized": vram_receipt(attention_normalized),
            "attention_normalized_padded": vram_receipt(
                attention_normalized_padded
            ),
            "q_projected": vram_receipt(q_projected),
            "k_projected": vram_receipt(k_projected),
            "v_projected": vram_receipt(v_projected),
            "q_normalized": vram_receipt(q_normalized),
            "k_normalized": vram_receipt(k_normalized),
            "q_rope": vram_receipt(q_rope),
            "k_rope": vram_receipt(k_rope),
            "attention_output": vram_receipt(attention_output),
            "o_projected": vram_receipt(o_projected),
            "attention_residual": tail_receipt["vram"]["attention_residual"],
            "post_attention_normalized": tail_receipt["vram"]["normalized"],
            "router_logits": tail_receipt["vram"]["router_logits"],
            "expert_combined": tail_receipt["vram"]["expert_combined"],
            "output": tail_receipt["vram"]["output"],
        }
        receipt: dict[str, object] = {
            "schema_version": 1,
            "scope": "tiny_single_layer_q_len1_decode_validation",
            "layer_index": layer_index,
            "batch_size": 1,
            "hidden": 64,
            "intermediate": 64,
            "cache_position": cache_position,
            "valid_cache_tokens": cache_position + 1,
            "packedkv_attention_prefix_executed": True,
            "packedkv": asdict(packed_layout),
            "packedkv_layout_id": packed_layout.layout_id,
            "append": {
                "key": [asdict(address) for address in k_append],
                "value": [asdict(address) for address in v_append],
            },
            "stages_executed": [
                "attention_rmsnorm",
                "qkv_projection",
                "qk_rmsnorm",
                "rope",
                "kv_cache_append",
                "packedkv_decode_attention",
                "attention_output_projection",
                *tail_receipt["stages_executed"],
            ],
            "vram_boundaries": boundaries,
            "tail": tail_receipt,
            "route_f32_base": route_f32_base,
            "indices_int_base": indices_int_base,
            "expected_assignment_count": 8,
            "compiler_lowering_valid": True,
            "compiler_generated_binary_emulator_parity": False,
            "tiny_single_layer_transaction_parity": False,
            "full_model_compiler_valid": False,
            "target_geometry_valid": False,
            "rtl_valid": False,
            "timing_calibrated": False,
            "publication_rankable": False,
        }
        return output, receipt

    def qwen3_scatter_pair_v0(
        self,
        accumulator: VRAMMatrixVar,
        pair_output: VRAMMatrixVar,
        *,
        token_idx: int,
        hidden: int,
        pair_idx: int,
    ) -> None:
        gp_dst, gp_src = self._reg.allocate_gp(2)
        try:
            asm = IsaBuilder().comment(
                moe_stage_marker(
                    "scatter_combine", f"pair={pair_idx} token={token_idx}"
                )
            )
            for col_block in range(hidden // self.mlen):
                asm.instr(
                    "S_ADDI_INT",
                    gp(gp_dst),
                    gp(0),
                    self._vram_matrix_row_addr(
                        accumulator, token_idx, col_block
                    ),
                )
                asm.instr(
                    "S_ADDI_INT",
                    gp(gp_src),
                    gp(0),
                    self._vram_matrix_row_addr(pair_output, 0, col_block),
                )
                asm.instr("V_ADD_VV", gp(gp_dst), gp(gp_dst), gp(gp_src), 0)
            self._emit(asm)
        finally:
            self._reg.free_gp([gp_dst, gp_src])

    def qwen3_dynamic_expert_pair_v0(
        self,
        x: VRAMMatrixVar,
        weights: ExpertWeights,
        *,
        weight_table_bases: tuple[int, int, int],
        weight_table_strides: tuple[int, int, int],
        indices_int_base: int,
        route_f32_base: int,
        pair_idx: int,
        intermediate: int,
        constants: QwenSwiGLUConstants,
        zero_row: FPVar,
        name: str,
    ) -> VRAMMatrixVar:
        """Run gate/up/SwiGLU/down for one runtime-selected expert."""

        w_gate, w_up, w_down = weights
        gate = self.qwen3_dynamic_projection_v0(
            x,
            w_gate,
            indices_int_base=indices_int_base,
            pair_idx=pair_idx,
            table_base=weight_table_bases[0],
            expert_stride=weight_table_strides[0],
            projection="gate",
            name=f"{name}_gate",
        )
        up = self.qwen3_dynamic_projection_v0(
            x,
            w_up,
            indices_int_base=indices_int_base,
            pair_idx=pair_idx,
            table_base=weight_table_bases[1],
            expert_stride=weight_table_strides[1],
            projection="up",
            name=f"{name}_up",
        )
        hidden = self.qwen3_swiglu_v0(
            gate,
            up,
            rows=1,
            intermediate=intermediate,
            constants=constants,
            name=name,
        )
        self.free_tensor(gate)
        output = self.qwen3_dynamic_projection_v0(
            hidden,
            w_down,
            indices_int_base=indices_int_base,
            pair_idx=pair_idx,
            table_base=weight_table_bases[2],
            expert_stride=weight_table_strides[2],
            projection="down",
            name=f"{name}_down",
        )
        self.free_tensor(hidden)
        self.qwen3_apply_route_f32_v0(
            output,
            route_f32_base=route_f32_base,
            pair_idx=pair_idx,
            hidden=w_down.shape[1],
        )
        return output

    def qwen3_moe_decode_layer_v0(
        self,
        x: VRAMMatrixVar,
        router_weight_rows: VRAMMatrixVar,
        expert_weight_templates: ExpertWeights,
        *,
        layer_index: int,
        batch_size: int,
        weight_table_bases: tuple[int, int, int],
        weight_table_strides: tuple[int, int, int],
        expert_table_counts: tuple[int, int, int],
        route_f32_base: int,
        indices_int_base: int,
        constants: QwenSwiGLUConstants,
        zero_row: FPVar,
        router_vector_format: str,
        hidden: int = 2048,
        intermediate: int = 768,
        name: str = "qwen3_moe",
    ) -> tuple[VRAMMatrixVar, dict[str, int | bool]]:
        """Lower one complete runtime-routed decode MoE layer."""

        if not 0 <= layer_index < 48:
            raise ValueError("Qwen3 target layer index must be in [0, 48)")
        if batch_size != x.shape[0]:
            raise ValueError("decode batch does not match the input rows")
        if tuple(x.shape) != (batch_size, hidden):
            raise ValueError(
                f"decode input must have shape ({batch_size}, {hidden}), got {x.shape}"
            )
        expected_shapes = (
            (hidden, intermediate),
            (hidden, intermediate),
            (intermediate, hidden),
        )
        actual_shapes = tuple(tuple(weight.shape) for weight in expert_weight_templates)
        if actual_shapes != expected_shapes:
            raise ValueError(
                f"expert templates have shapes {actual_shapes}, expected {expected_shapes}"
            )
        if any(base < 0 for base in weight_table_bases):
            raise ValueError("expert table bases must be non-negative")
        if any(stride <= 0 for stride in weight_table_strides):
            raise ValueError("expert table strides must be positive")
        if expert_table_counts != (128, 128, 128):
            raise ValueError(
                "runtime routing requires 128 addressable gate/up/down experts"
            )
        required_route_entries = batch_size * QWEN3_ROUTE_SCORES_PER_TOKEN
        if (
            route_f32_base < 0
            or route_f32_base + required_route_entries
            > QWEN_ROUTE_F32_SRAM_DEPTH
        ):
            raise ValueError(
                "whole-batch route scores exceed FP32 route SRAM; "
                "token-serial reuse is not implemented"
            )
        if (
            indices_int_base < 0
            or indices_int_base + required_route_entries > QWEN_INT_SRAM_DEPTH
        ):
            raise ValueError(
                "whole-batch expert IDs exceed integer SRAM; "
                "token-serial reuse is not implemented"
            )
        logits = self.qwen3_router_logits_bf16_v0(
            x,
            router_weight_rows,
            rows=batch_size,
            hidden=hidden,
            num_experts=128,
            router_vector_format=router_vector_format,
            name=f"{name}_l{layer_index}_router",
        )
        for token_idx in range(batch_size):
            self.qwen3_moe_router_topk8_v0(
                logits,
                token_idx=token_idx,
                route_f32_base=route_f32_base + token_idx * 8,
                indices_int_base=indices_int_base + token_idx * 8,
            )
        accumulator = self.alloc(
            f"{name}_l{layer_index}_combine",
            rows=batch_size,
            cols=hidden,
            strict=False,
            physical_shape=(
                max(self.blen, math.ceil(batch_size / self.blen) * self.blen),
                hidden,
            ),
        )
        self._qwen3_true_zero_rows_v0(
            accumulator,
            rows=range(accumulator.physical_shape[0]),
            hidden=hidden,
            zero_row=zero_row,
            stage="scatter_combine",
        )
        assignment_count = 0
        for token_idx in range(batch_size):
            for rank in range(8):
                pair_idx = token_idx * 8 + rank
                pair = self.qwen3_gather_pair_v0(
                    x,
                    token_idx=token_idx,
                    hidden=hidden,
                    zero_row=zero_row,
                    name=f"{name}_l{layer_index}_pair{pair_idx}_x",
                )
                output = self.qwen3_dynamic_expert_pair_v0(
                    pair,
                    expert_weight_templates,
                    weight_table_bases=weight_table_bases,
                    weight_table_strides=weight_table_strides,
                    indices_int_base=indices_int_base,
                    route_f32_base=route_f32_base,
                    pair_idx=pair_idx,
                    intermediate=intermediate,
                    constants=constants,
                    zero_row=zero_row,
                    name=f"{name}_l{layer_index}_pair{pair_idx}",
                )
                self.qwen3_scatter_pair_v0(
                    accumulator,
                    output,
                    token_idx=token_idx,
                    hidden=hidden,
                    pair_idx=pair_idx,
                )
                assignment_count += 1
                self.free_tensor(pair)
                self.free_tensor(output)
        if assignment_count != batch_size * 8:
            raise RuntimeError("compiler route conservation invariant failed")
        self.free_tensor(logits)
        return accumulator, {
            "layer_index": layer_index,
            "expected_assignments": batch_size * 8,
            "emitted_assignments": assignment_count,
            "router_linear_transformers_fixture_parity": True,
            "complete_runtime_moe_layer_lowering": True,
            "layer_emulator_transaction_parity": False,
        }


__all__ = [
    "MOE_END_STAGE",
    "MOE_STAGES",
    "MOE_STAGE_MARKER_PREFIX",
    "QWEN_INT_SRAM_DEPTH",
    "QWEN_ROUTE_F32_SRAM_DEPTH",
    "ProgramRoutedMoeMixin",
    "moe_stage_marker",
]
