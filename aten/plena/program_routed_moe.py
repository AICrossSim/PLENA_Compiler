"""Routed-MoE v0 program-builder helpers."""

from __future__ import annotations

import math
from collections.abc import Sequence

from compiler.aten.isa_builder import IsaBuilder, addr as areg, fp, gp
from compiler.aten.plena.vars import FPVar, InputVar, VRAMMatrixVar

GptOssFPConstants = tuple[FPVar, FPVar, FPVar, FPVar, FPVar]
ExpertWeights = tuple[InputVar, InputVar, InputVar]
ExpertBiases = tuple[VRAMMatrixVar | None, VRAMMatrixVar | None, VRAMMatrixVar | None]


class ProgramRoutedMoeMixin:
    """Routed-MoE v0 emit helpers used by GPT-OSS and Qwen bring-up.

    The helpers intentionally keep routing policy explicit: GPT-OSS uses
    high-precision router logits plus softmax-after-topk; Qwen adapters reuse
    the same substrate with 128-way/top-8 selection. Expert gate/up projections
    are split instead of using packed even/odd gate_up because current
    vector-scalar min/max ops apply to whole rows, not alternating lanes.
    """

    def _validate_gpt_oss_constants(self, constants: GptOssFPConstants, rows: int) -> None:
        zero, limit_pos, limit_neg, one, neg_alpha = constants
        if zero.address != 0:
            raise ValueError("vram_fill_zero assumes FPRAM f0 is preloaded with zero")
        for var in (limit_pos, limit_neg, one, neg_alpha):
            if var.size < rows:
                raise ValueError(f"FPVar {var.name} size={var.size} is smaller than rows={rows}")

    def _validate_standard_swiglu_constants(self, constants: GptOssFPConstants, rows: int) -> None:
        zero, _unused_pos, _unused_neg, one, neg_one = constants
        if zero.address != 0:
            raise ValueError("vram_fill_zero assumes FPRAM f0 is preloaded with zero")
        for var in (one, neg_one):
            if var.size < rows:
                raise ValueError(f"FPVar {var.name} size={var.size} is smaller than rows={rows}")

    def _vram_matrix_row_addr(self, matrix: VRAMMatrixVar, row_idx: int, tile_col_idx: int = 0) -> int:
        row_block = row_idx // self.mlen
        row_in_block = row_idx % self.mlen
        return self.get_vram_tile_addr(matrix.name, row_block, tile_col_idx) + row_in_block * self.mlen

    def gpt_oss_router_logits_bf16_v0(
        self,
        x: VRAMMatrixVar,
        router_weight_rows: VRAMMatrixVar,
        *,
        rows: int,
        hidden: int,
        num_experts: int,
        name: str = "gpt_oss_router_logits",
    ) -> VRAMMatrixVar:
        """Emit high-precision GPT-OSS router logits using BF16 vector dot products.

        This path intentionally avoids ``linear_projection`` and all HBM/MX
        prefetch machinery.  ``x`` and ``router_weight_rows`` must already be
        resident in BF16 VRAM.  The router is the only GPT-OSS MoE v0 path that
        uses this non-MX lowering; expert projections continue to use the MXFP8
        matrix path.

        ``router_weight_rows`` is laid out as ``[num_experts, hidden]`` so every
        expert's hidden vector can be multiplied row-wise against a token row.
        """
        if hidden % self.mlen != 0:
            raise ValueError(f"router hidden={hidden} must be divisible by MLEN={self.mlen}")
        if rows > x.shape[0]:
            raise ValueError(f"router rows={rows} exceeds x rows={x.shape[0]}")
        if hidden > x.shape[1]:
            raise ValueError(f"router hidden={hidden} exceeds x width={x.shape[1]}")
        if num_experts > router_weight_rows.shape[0] or hidden > router_weight_rows.shape[1]:
            raise ValueError(
                "router_weight_rows must have shape at least "
                f"({num_experts}, {hidden}), got {router_weight_rows.shape}"
            )

        expert_blocks = math.ceil(num_experts / self.mlen)
        logical_logit_rows = rows if expert_blocks == 1 else rows * expert_blocks
        logical_logit_cols = num_experts if expert_blocks == 1 else self.mlen
        physical_rows = max(self.blen, math.ceil(logical_logit_rows / self.blen) * self.blen)
        logits = self.alloc(
            name,
            rows=logical_logit_rows,
            cols=logical_logit_cols,
            strict=False,
            physical_shape=(physical_rows, self.mlen),
        )
        scratch = self.alloc(
            f"{name}_dot_scratch",
            rows=1,
            cols=self.mlen,
            strict=False,
            physical_shape=(1, self.mlen),
        )
        fp_scratch = self.fp_var(f"{name}_fp_scratch", size=expert_blocks * self.mlen)

        scratch_addr = self.get_vram_addr(scratch.name)
        k_blocks = hidden // self.mlen
        gp_x, gp_w, gp_scratch, gp_fp, gp_out, gp_loop = self._reg.allocate_gp(6)
        fp_acc = self.allocate_fp_reg(1)[0]
        try:
            asm = IsaBuilder().comment(
                f"GPT-OSS router BF16 vector-dot logits: rows={rows}, hidden={hidden}, experts={num_experts}"
            )
            asm.instr("S_ADDI_INT", gp(gp_scratch), gp(0), scratch_addr)
            fp_base = fp_scratch.address

            # Clear the FPRAM logits scratch once: positions [0, num_experts) are
            # overwritten by every token's per-expert S_ST_FP, and the padding
            # positions [num_experts, expert_blocks*mlen) are never written, so a
            # single clear before the token loop keeps them zero for all tokens.
            asm.comment("Clear FPRAM logits scratch (once, loop-invariant across tokens)")
            asm.instr("S_ADDI_INT", gp(gp_fp), gp(0), fp_base)
            asm.instr("C_LOOP_START", gp(gp_loop), expert_blocks * self.mlen)
            asm.instr("S_ST_FP", fp(0), gp(gp_fp), 0)
            asm.instr("S_ADDI_INT", gp(gp_fp), gp(gp_fp), 1)
            asm.instr("C_LOOP_END", gp(gp_loop))

            for token_idx in range(rows):
                for expert_idx in range(num_experts):
                    x_addr = self._vram_matrix_row_addr(x, token_idx, 0)
                    w_addr = self._vram_matrix_row_addr(router_weight_rows, expert_idx, 0)
                    x_step = x.physical_shape[0] * self.mlen
                    w_step = router_weight_rows.physical_shape[0] * self.mlen

                    asm.comment(f"Router dot token {token_idx}, expert {expert_idx}")
                    asm.instr("S_ADD_FP", fp(fp_acc), fp(0), fp(0))
                    asm.instr("S_ADDI_INT", gp(gp_x), gp(0), x_addr)
                    asm.instr("S_ADDI_INT", gp(gp_w), gp(0), w_addr)
                    asm.instr("C_LOOP_START", gp(gp_loop), k_blocks)
                    asm.instr("V_MUL_VV", gp(gp_scratch), gp(gp_x), gp(gp_w), 0)
                    asm.instr("V_RED_SUM", fp(fp_acc), gp(gp_scratch), 0, 0)
                    asm.instr("S_ADDI_INT", gp(gp_x), gp(gp_x), x_step)
                    asm.instr("S_ADDI_INT", gp(gp_w), gp(gp_w), w_step)
                    asm.instr("C_LOOP_END", gp(gp_loop))
                    asm.instr("S_ADDI_INT", gp(gp_fp), gp(0), fp_base + expert_idx)
                    asm.instr("S_ST_FP", fp(fp_acc), gp(gp_fp), 0)

                asm.comment(f"Router token {token_idx}: map FPRAM logits scratch to contiguous VRAM rows")
                for expert_block in range(expert_blocks):
                    out_row = token_idx if expert_blocks == 1 else token_idx * expert_blocks + expert_block
                    out_addr = self._vram_matrix_row_addr(logits, out_row, 0)
                    asm.instr("S_ADDI_INT", gp(gp_out), gp(0), out_addr)
                    asm.instr("S_ADDI_INT", gp(gp_fp), gp(0), fp_base + expert_block * self.mlen)
                    asm.instr("S_MAP_V_FP", gp(gp_out), gp(gp_fp), 0)
            self._emit(asm)
        finally:
            self.free_fp_reg([fp_acc])
            self._reg.free_gp([gp_x, gp_w, gp_scratch, gp_fp, gp_out, gp_loop])

        return logits

    def _pack_router_logits_token_major(
        self,
        matrix_logits: VRAMMatrixVar,
        *,
        rows: int,
        num_experts: int,
        expert_blocks: int,
        logical_logit_rows: int,
        physical_logit_rows: int,
        name: str,
        label: str,
    ) -> VRAMMatrixVar:
        """Pack a ``[rows, experts]`` logits tensor into the token-major V_TOPK ABI.

        Row layout: ``token0/block0, token0/block1, ..., token1/block0, ...``.
        When ``expert_blocks == 1`` the matrix tensor already matches the ABI and
        is returned unchanged; otherwise a new packed tensor is emitted and the
        source ``matrix_logits`` is freed. ``label`` only tags the emitted comment.
        """
        if expert_blocks == 1:
            return matrix_logits

        packed_logits = self.alloc(
            name,
            rows=logical_logit_rows,
            cols=self.mlen,
            strict=False,
            physical_shape=(physical_logit_rows, self.mlen),
        )

        gp_dst, gp_src = self._reg.allocate_gp(2)
        try:
            asm = IsaBuilder().comment(
                f"Qwen router {label} logits pack: rows={rows}, experts={num_experts}, blocks={expert_blocks}"
            )
            for token_idx in range(rows):
                for expert_block in range(expert_blocks):
                    src_addr = self._vram_matrix_row_addr(matrix_logits, token_idx, expert_block)
                    dst_row = token_idx * expert_blocks + expert_block
                    dst_addr = self._vram_matrix_row_addr(packed_logits, dst_row, 0)
                    asm.instr("S_ADDI_INT", gp(gp_dst), gp(0), dst_addr)
                    asm.instr("S_ADDI_INT", gp(gp_src), gp(0), src_addr)
                    asm.instr("V_ADD_VF", gp(gp_dst), gp(gp_src), fp(0), 0)
            self._emit(asm)
        finally:
            self._reg.free_gp([gp_dst, gp_src])

        self.free_tensor(matrix_logits)
        return packed_logits

    def qwen3_router_logits_matrix_bf16_rowpacked_v0(
        self,
        x: VRAMMatrixVar,
        router_weight_matrix: InputVar,
        *,
        rows: int,
        hidden: int,
        num_experts: int,
        mram_tile_capacity: int = 4,
        stream_k_accum: bool = True,
        name: str = "qwen3_router_logits_matrix",
    ) -> VRAMMatrixVar:
        """Emit Qwen router logits through the BF16 matrix path in V_TOPK layout.

        The matrix machine naturally produces a ``[rows, experts]`` tensor.  The
        existing V_TOPK ABI for 128 experts expects each token to occupy
        contiguous MLEN-wide rows: ``token0/block0, token0/block1, token1/...``.
        This helper keeps the numerically better matrix accumulation while
        packing the result into that token-major row layout.
        """
        if hidden % self.mlen != 0:
            raise ValueError(f"router hidden={hidden} must be divisible by MLEN={self.mlen}")
        if rows > x.shape[0]:
            raise ValueError(f"router rows={rows} exceeds x rows={x.shape[0]}")
        if hidden > x.shape[1]:
            raise ValueError(f"router hidden={hidden} exceeds x width={x.shape[1]}")
        if router_weight_matrix.shape[0] < hidden or router_weight_matrix.shape[1] < num_experts:
            raise ValueError(
                "router_weight_matrix must have shape at least "
                f"({hidden}, {num_experts}), got {router_weight_matrix.shape}"
            )
        if mram_tile_capacity <= 0:
            raise ValueError(f"mram_tile_capacity must be > 0, got {mram_tile_capacity}")

        expert_blocks = math.ceil(num_experts / self.mlen)
        physical_rows = max(self.blen, math.ceil(rows / self.blen) * self.blen)
        physical_experts = expert_blocks * self.mlen
        logical_logit_rows = rows if expert_blocks == 1 else rows * expert_blocks
        physical_logit_rows = max(self.blen, math.ceil(logical_logit_rows / self.blen) * self.blen)

        old_capacity = self.mram_tile_capacity
        self.mram_tile_capacity = mram_tile_capacity
        try:
            if stream_k_accum:
                matrix_logits = self.linear_projection_bf16_stream_k_accum(
                    x,
                    router_weight_matrix,
                    name=f"{name}_matrix",
                    physical_shape=(physical_rows, physical_experts),
                    max_k_tiles=mram_tile_capacity,
                )
            else:
                matrix_logits = self.linear_projection_bf16(
                    x,
                    router_weight_matrix,
                    name=f"{name}_matrix",
                    physical_shape=(physical_rows, physical_experts),
                )
        finally:
            self.mram_tile_capacity = old_capacity

        return self._pack_router_logits_token_major(
            matrix_logits,
            rows=rows,
            num_experts=num_experts,
            expert_blocks=expert_blocks,
            logical_logit_rows=logical_logit_rows,
            physical_logit_rows=physical_logit_rows,
            name=name,
            label="matrix",
        )

    def qwen3_router_logits_packed_skinny_bf16_rowpacked_v0(
        self,
        x: VRAMMatrixVar,
        router_weight_packed_skinny: InputVar,
        *,
        rows: int,
        hidden: int,
        num_experts: int,
        k_tiles_per_packed_tile: int = 8,
        name: str = "qwen3_router_logits_packed_skinny",
    ) -> VRAMMatrixVar:
        """Emit Qwen router logits from a packed-skinny BF16 HBM table.

        This is the integration form of the packed-skinny router probe: the
        weight table packs several skinny K slices into one full MRAM tile, so
        cap8-equivalent accumulation can be expressed under the existing cap4
        MRAM contract.  The result is returned in the existing V_TOPK token-
        major ABI when ``num_experts`` spans multiple MLEN rows.
        """
        if hidden % self.mlen != 0:
            raise ValueError(f"router hidden={hidden} must be divisible by MLEN={self.mlen}")
        if rows > x.shape[0]:
            raise ValueError(f"router rows={rows} exceeds x rows={x.shape[0]}")
        if rows > self.mlen:
            raise NotImplementedError(
                "packed-skinny Qwen router currently supports one sequence row-block; "
                f"got rows={rows}, MLEN={self.mlen}"
            )
        if hidden > x.shape[1]:
            raise ValueError(f"router hidden={hidden} exceeds x width={x.shape[1]}")
        if k_tiles_per_packed_tile <= 0:
            raise ValueError(f"k_tiles_per_packed_tile must be > 0, got {k_tiles_per_packed_tile}")

        expert_blocks = math.ceil(num_experts / self.mlen)
        physical_rows = max(self.blen, math.ceil(rows / self.blen) * self.blen)
        physical_experts = expert_blocks * self.mlen
        logical_logit_rows = rows if expert_blocks == 1 else rows * expert_blocks
        physical_logit_rows = max(self.blen, math.ceil(logical_logit_rows / self.blen) * self.blen)
        tiles_per_mlen = self.mlen // self.blen

        required_col_blocks = expert_blocks * math.ceil(self.mlen / self.blen)
        if router_weight_packed_skinny.physical_shape[1] < required_col_blocks * self.mlen:
            raise ValueError(
                "router_weight_packed_skinny physical width is too small for "
                f"{expert_blocks} output blocks: got {router_weight_packed_skinny.physical_shape}"
            )

        matrix_logits = self.alloc(
            f"{name}_matrix",
            rows=rows,
            cols=num_experts,
            strict=False,
            physical_shape=(physical_rows, physical_experts),
        )
        for expert_block in range(expert_blocks):
            self.vram_sub_projection_packed_skinny_stream_k_accum_to(
                x,
                0,
                router_weight_packed_skinny,
                expert_block * tiles_per_mlen,
                matrix_logits,
                0,
                expert_block,
                max_k_tiles_per_packed_tile=k_tiles_per_packed_tile,
                matrix_precision="keyvalue",
                set_scale=False,
                hbm_element_bytes=2,
            )

        return self._pack_router_logits_token_major(
            matrix_logits,
            rows=rows,
            num_experts=num_experts,
            expert_blocks=expert_blocks,
            logical_logit_rows=logical_logit_rows,
            physical_logit_rows=physical_logit_rows,
            name=name,
            label="packed-skinny",
        )

    def gpt_oss_router_topk_softmax_v0(
        self,
        logits: VRAMMatrixVar,
        *,
        token_idx: int,
        weights_fp_base: int,
        indices_int_base: int,
        num_experts: int = 32,
        top_k: int = 4,
        name: str = "gpt_oss_router_topk",
    ) -> None:
        """Emit V_TOPK for one router-logit row.

        V_TOPK v0 reads one BF16 router-logit row from VRAM, performs a
        linear-scan top-k with low-index tie break, stores the selected expert
        ids to INT SRAM, and stores the softmax-over-selected weights to FP
        SRAM.  The instruction intentionally keeps router/top-k on the BF16
        path and does not touch MX scale state.
        """
        if token_idx < 0 or token_idx >= logits.shape[0]:
            raise ValueError(f"token_idx={token_idx} outside logits rows={logits.shape[0]}")
        expert_blocks = math.ceil(num_experts / self.mlen)
        if expert_blocks == 1 and logits.shape[1] < num_experts:
            raise ValueError(f"V_TOPK expects at least {num_experts} logits, got {logits.shape[1]}")
        if expert_blocks > 1 and logits.shape[1] < self.mlen:
            raise ValueError(f"V_TOPK expects MLEN-wide logit rows, got width={logits.shape[1]}")
        required_rows = token_idx + 1 if expert_blocks == 1 else (token_idx + 1) * expert_blocks
        if logits.shape[0] < required_rows:
            raise ValueError(
                f"V_TOPK expects token {token_idx} to occupy {expert_blocks} contiguous logit rows, "
                f"got logits shape={logits.shape}"
            )
        policy_rmask = {(32, 4): 0, (128, 8): 1}.get((num_experts, top_k))
        if policy_rmask is None:
            raise NotImplementedError(f"V_TOPK policy unsupported for num_experts={num_experts}, top_k={top_k}")

        gp_weights, gp_logits, gp_indices = self._reg.allocate_gp(3)
        try:
            asm = IsaBuilder().comment(
                f"Routed-MoE V_TOPK {name}: token={token_idx}, experts={num_experts}, top_k={top_k}, "
                f"weights_fp={weights_fp_base}, indices_int={indices_int_base}"
            )
            asm.instr("S_ADDI_INT", gp(gp_weights), gp(0), weights_fp_base)
            asm.instr(
                "S_ADDI_INT",
                gp(gp_logits),
                gp(0),
                self._vram_matrix_row_addr(logits, token_idx if expert_blocks == 1 else token_idx * expert_blocks, 0),
            )
            asm.instr("S_ADDI_INT", gp(gp_indices), gp(0), indices_int_base)
            asm.instr("V_TOPK", gp(gp_weights), gp(gp_logits), gp(gp_indices), policy_rmask)
            self._emit(asm)
        finally:
            self._reg.free_gp([gp_weights, gp_logits, gp_indices])

    def _emit_expert_id_to_weight_base_v0(
        self,
        asm: IsaBuilder,
        *,
        expert_indices_int_base: int,
        pair_idx: int,
        table_base: int,
        per_expert_stride: int,
        addr_reg: int,
        gp_table: int,
        gp_expert: int,
        gp_stride: int,
        gp_offset: int,
        gp_base: int,
        name: str,
    ) -> None:
        """Emit the shared true-expert-id -> HBM-base address calculation."""
        if per_expert_stride <= 0:
            raise ValueError(f"{name}: per_expert_stride must be positive, got {per_expert_stride}")
        asm.comment(
            f"{name}: expert_id_to_weight_base pair={pair_idx}, "
            f"table_base={table_base}, stride={per_expert_stride}"
        )
        asm.instr("S_ADDI_INT", gp(gp_table), gp(0), expert_indices_int_base)
        asm.instr("S_LD_INT", gp(gp_expert), gp(gp_table), pair_idx)
        asm.instr("S_ADDI_INT", gp(gp_stride), gp(0), per_expert_stride)
        asm.instr("S_MUL_INT", gp(gp_offset), gp(gp_expert), gp(gp_stride))
        asm.instr("S_ADDI_INT", gp(gp_base), gp(0), table_base)
        asm.instr("S_ADD_INT", gp(gp_base), gp(gp_base), gp(gp_offset))
        asm.instr("C_SET_ADDR_REG", areg(addr_reg), gp(0), gp(gp_base))

    def _emit_expert_id_to_weight_base_table_v0(
        self,
        asm: IsaBuilder,
        *,
        expert_indices_int_base: int,
        expert_base_table_int_base: int,
        pair_idx: int,
        addr_reg: int,
        gp_table: int,
        gp_expert: int,
        gp_base: int,
        name: str,
    ) -> None:
        """Emit expert-id -> HBM-base lookup through an IntSRAM base table."""
        asm.comment(
            f"{name}: expert_id_to_weight_base_table pair={pair_idx}, "
            f"base_table_int={expert_base_table_int_base}"
        )
        asm.instr("S_ADDI_INT", gp(gp_table), gp(0), expert_indices_int_base)
        asm.instr("S_LD_INT", gp(gp_expert), gp(gp_table), pair_idx)
        asm.instr("S_LD_INT", gp(gp_base), gp(gp_expert), expert_base_table_int_base)
        asm.instr("C_SET_ADDR_REG", areg(addr_reg), gp(0), gp(gp_base))

    def gpt_oss_expert_id_to_weight_base_v0(
        self,
        *,
        expert_indices_int_base: int,
        pair_idx: int,
        table_base: int,
        per_expert_stride: int,
        addr_reg: int,
        name: str = "gpt_oss_expert_id_to_weight_base",
    ) -> None:
        """Public helper for Step6: set ``addr_reg`` to true expert HBM base.

        ``expert_indices_int_base[pair_idx]`` must contain a true GPT-OSS expert
        id in ``[0, 31]``.  This helper is the only supported address contract for
        dynamic expert weights; callers must not remap to host-compressed expert
        indices.
        """
        gp_table, gp_expert, gp_stride, gp_offset, gp_base = self._reg.allocate_gp(5)
        try:
            asm = IsaBuilder()
            self._emit_expert_id_to_weight_base_v0(
                asm,
                expert_indices_int_base=expert_indices_int_base,
                pair_idx=pair_idx,
                table_base=table_base,
                per_expert_stride=per_expert_stride,
                addr_reg=addr_reg,
                gp_table=gp_table,
                gp_expert=gp_expert,
                gp_stride=gp_stride,
                gp_offset=gp_offset,
                gp_base=gp_base,
                name=name,
            )
            self._emit(asm)
        finally:
            self._reg.free_gp([gp_table, gp_expert, gp_stride, gp_offset, gp_base])

    def _gpt_oss_dynamic_load_sub_matrix_col_v0(
        self,
        *,
        weight_template: InputVar,
        col_idx: int,
        expert_indices_int_base: int,
        pair_idx: int,
        table_base: int,
        per_expert_stride: int,
        expert_base_table_int_base: int | None = None,
        mram_start_addr: int | None = None,
        k_block_start: int = 0,
        k_block_count: int | None = None,
        name: str = "gpt_oss_dynamic_weight_load",
    ) -> None:
        """Load one weight column tile using runtime true expert id addressing."""
        self._ensure_hbm_sub_matrix_registered(weight_template)
        layout = self.get_hbm_layout(weight_template.name)
        num_row_blocks = layout.num_row_blocks
        block_size = self.mlen * self.mlen
        effective_count = k_block_count if k_block_count is not None else num_row_blocks
        if mram_start_addr is None:
            mram_start_addr = self.mram_allocator.allocate(
                f"{name}_{weight_template.name}_pair{pair_idx}_col{col_idx}",
                effective_count * block_size,
            )

        gp_table, gp_expert, gp_expert_stride, gp_expert_offset, gp_base, gp_scale, gp_stride, gp_mram = (
            self._reg.allocate_gp(8)
        )
        addr_reg = self._reg.allocate_addr(1)[0]
        try:
            asm = IsaBuilder().comment(
                f"GPT-OSS dynamic HBM weight prefetch: template={weight_template.name}, "
                f"pair={pair_idx}, col={col_idx}"
            )
            if expert_base_table_int_base is None:
                self._emit_expert_id_to_weight_base_v0(
                    asm,
                    expert_indices_int_base=expert_indices_int_base,
                    pair_idx=pair_idx,
                    table_base=table_base,
                    per_expert_stride=per_expert_stride,
                    addr_reg=addr_reg,
                    gp_table=gp_table,
                    gp_expert=gp_expert,
                    gp_stride=gp_expert_stride,
                    gp_offset=gp_expert_offset,
                    gp_base=gp_base,
                    name=name,
                )
            else:
                self._emit_expert_id_to_weight_base_table_v0(
                    asm,
                    expert_indices_int_base=expert_indices_int_base,
                    expert_base_table_int_base=expert_base_table_int_base,
                    pair_idx=pair_idx,
                    addr_reg=addr_reg,
                    gp_table=gp_table,
                    gp_expert=gp_expert,
                    gp_base=gp_base,
                    name=name,
                )
            self._emit_hbm_prefetch_setup(asm, layout, gp_scale, gp_stride)
            self._emit_hbm_subblock_sequence(
                asm,
                layout,
                ((row_idx, col_idx) for row_idx in range(k_block_start, k_block_start + effective_count)),
                mram_start_addr,
                addr_reg,
                gp_scale,
                gp_mram,
            )
            self._emit(asm)
        finally:
            self._reg.free_gp(
                [gp_table, gp_expert, gp_expert_stride, gp_expert_offset, gp_base, gp_scale, gp_stride, gp_mram]
            )
            self._reg.free_addr([addr_reg])

    def gpt_oss_dynamic_vram_sub_projection_to_v0(
        self,
        vram_matrix: VRAMMatrixVar,
        vram_row_idx: int,
        weight_template: InputVar,
        weight_col_idx: int,
        target: VRAMMatrixVar,
        target_row_idx: int,
        target_col_idx: int,
        *,
        expert_indices_int_base: int,
        pair_idx: int,
        table_base: int,
        per_expert_stride: int,
        expert_base_table_int_base: int | None = None,
        auto_reset_mram: bool = True,
        k_block_start: int = 0,
        k_block_count: int | None = None,
        name: str = "gpt_oss_dynamic_projection",
    ) -> None:
        """Projection tile where the HBM weight base comes from V_TOPK expert id."""
        vram_matrix = self._require_var(vram_matrix, VRAMMatrixVar, "vram_matrix")
        weight_template = self._require_var(weight_template, InputVar, "weight_template")
        target = self._require_var(target, VRAMMatrixVar, "target")
        self._ensure_vram_sub_matrix_registered(vram_matrix)
        self._ensure_hbm_sub_matrix_registered(weight_template)
        if auto_reset_mram:
            super().reset_mram()
        self._gpt_oss_dynamic_load_sub_matrix_col_v0(
            weight_template=weight_template,
            col_idx=weight_col_idx,
            expert_indices_int_base=expert_indices_int_base,
            pair_idx=pair_idx,
            table_base=table_base,
            per_expert_stride=per_expert_stride,
            expert_base_table_int_base=expert_base_table_int_base,
            k_block_start=k_block_start,
            k_block_count=k_block_count,
            name=name,
        )
        super().vram_sub_projection_to(
            vram_mat_name=vram_matrix.name,
            vram_row_idx=vram_row_idx,
            mram_mat_name=weight_template.name,
            mram_col_idx=weight_col_idx,
            target_matrix=target.name,
            target_row_idx=target_row_idx,
            target_col_idx=target_col_idx,
            k_block_start=k_block_start,
            k_block_count=k_block_count,
        )

    def gpt_oss_dynamic_linear_projection_v0(
        self,
        input_var: VRAMMatrixVar,
        weight_template: InputVar,
        *,
        expert_indices_int_base: int,
        pair_idx: int,
        table_base: int,
        per_expert_stride: int,
        expert_base_table_int_base: int | None = None,
        name: str,
        physical_shape: tuple[int, int] | None = None,
    ) -> VRAMMatrixVar:
        """Tiled linear projection with runtime expert-id weight selection."""
        mlen = self.mlen
        rows, _k_total = input_var.shape
        _weight_rows, out_features = weight_template.shape
        if physical_shape is None:
            # K-split accumulation uses 64x64 block adds.  Routed dynamic
            # projections often have only 4/8 logical rows, so keep outputs
            # tile-backed to prevent block-add accumulation from walking into
            # the next column block.
            physical_rows = max(self.mlen, input_var.physical_shape[0], math.ceil(rows / self.blen) * self.blen)
            physical_out_features = weight_template.physical_shape[1]
        else:
            physical_rows, physical_out_features = physical_shape
            if physical_rows < rows or physical_out_features < out_features:
                raise ValueError(
                    f"physical_shape {physical_shape} cannot be smaller than logical output {(rows, out_features)}"
                )

        physical_k = max(input_var.physical_shape[1], weight_template.physical_shape[0])
        num_row_blocks = math.ceil(physical_rows / mlen)
        num_col_blocks = math.ceil(physical_out_features / mlen)
        num_k_tiles = math.ceil(physical_k / mlen)
        max_k_tiles = self.mram_tile_capacity

        output = self.alloc(
            name,
            rows,
            out_features,
            strict=False,
            physical_shape=(physical_rows, physical_out_features),
        )

        def emit_projection(row_idx, col_idx, target, target_row_idx, target_col_idx, **k_split) -> None:
            self.gpt_oss_dynamic_vram_sub_projection_to_v0(
                input_var,
                row_idx,
                weight_template,
                col_idx,
                target,
                target_row_idx,
                target_col_idx,
                expert_indices_int_base=expert_indices_int_base,
                pair_idx=pair_idx,
                table_base=table_base,
                per_expert_stride=per_expert_stride,
                expert_base_table_int_base=expert_base_table_int_base,
                name=f"{name}_pair{pair_idx}",
                **k_split,
            )

        if num_k_tiles <= max_k_tiles:
            for col_idx in range(num_col_blocks):
                for row_idx in range(num_row_blocks):
                    emit_projection(row_idx, col_idx, output, row_idx, col_idx)
            return output

        temp = self.alloc(f"{name}_temp", mlen, mlen)
        for k_chunk_idx, k_block_start in enumerate(range(0, num_k_tiles, max_k_tiles)):
            k_block_count = min(max_k_tiles, num_k_tiles - k_block_start)
            k_split = {"k_block_start": k_block_start, "k_block_count": k_block_count}
            for col_idx in range(num_col_blocks):
                for row_idx in range(num_row_blocks):
                    if k_chunk_idx == 0:
                        emit_projection(row_idx, col_idx, output, row_idx, col_idx, **k_split)
                    else:
                        emit_projection(row_idx, col_idx, temp, 0, 0, **k_split)
                        self.vram_block_add_to(output, row_idx, col_idx, temp, 0, 0, output, row_idx, col_idx)
        self.free_tensor(temp)
        return output

    def gpt_oss_add_dynamic_expert_bias_v0(
        self,
        dst: VRAMMatrixVar,
        bias_table: VRAMMatrixVar,
        *,
        expert_indices_int_base: int,
        pair_idx: int,
        rows: int,
        width: int,
        name: str = "gpt_oss_dynamic_bias",
    ) -> None:
        """Add BF16 bias selected by true expert id from a VRAM bias table."""
        if width % self.mlen != 0:
            raise ValueError(f"{name}: width={width} must be divisible by MLEN={self.mlen}")
        if rows > self.blen:
            raise ValueError(f"{name}: v0 expects one routed pair slot (rows<=BLEN), got rows={rows}")
        self._ensure_vram_sub_matrix_registered(dst)
        self._ensure_vram_sub_matrix_registered(bias_table)
        num_col_blocks = width // self.mlen
        bias_rows = bias_table.physical_shape[0]
        expert_row_stride = self.blen * self.mlen

        gp_table, gp_expert, gp_stride, gp_expert_offset, gp_src_base, gp_src, gp_dst = self._reg.allocate_gp(7)
        try:
            asm = IsaBuilder().comment(f"GPT-OSS dynamic expert bias add {name}: pair={pair_idx}, rows={rows}")
            asm.instr("S_ADDI_INT", gp(gp_table), gp(0), expert_indices_int_base)
            asm.instr("S_LD_INT", gp(gp_expert), gp(gp_table), pair_idx)
            asm.instr("S_ADDI_INT", gp(gp_stride), gp(0), expert_row_stride)
            asm.instr("S_MUL_INT", gp(gp_expert_offset), gp(gp_expert), gp(gp_stride))
            for col_block in range(num_col_blocks):
                src_col_base = self._vram_matrix_row_addr(bias_table, 0, col_block)
                for row_idx in range(rows):
                    dst_addr = self._vram_matrix_row_addr(dst, row_idx, col_block)
                    asm.instr("S_ADDI_INT", gp(gp_src_base), gp(0), src_col_base + row_idx * self.mlen)
                    asm.instr("S_ADD_INT", gp(gp_src), gp(gp_src_base), gp(gp_expert_offset))
                    asm.instr("S_ADDI_INT", gp(gp_dst), gp(0), dst_addr)
                    asm.instr("V_ADD_VV", gp(gp_dst), gp(gp_dst), gp(gp_src), 0)
            self._emit(asm)
        finally:
            self._reg.free_gp([gp_table, gp_expert, gp_stride, gp_expert_offset, gp_src_base, gp_src, gp_dst])

    def gpt_oss_materialize_topk_route_weight_v0(
        self,
        *,
        weights_fp_base: int,
        pair_idx: int,
        rows: int,
        hidden: int,
        zero_row: FPVar | None = None,
        fp_scratch: FPVar | None = None,
        name: str = "gpt_oss_device_route_weight",
    ) -> VRAMMatrixVar:
        """Expand device V_TOPK scalar weight into a VRAM route matrix."""
        if rows > self.blen:
            raise ValueError(f"{name}: v0 expects one routed pair slot (rows<=BLEN), got rows={rows}")
        if hidden % self.mlen != 0:
            raise ValueError(f"{name}: hidden={hidden} must be divisible by MLEN={self.mlen}")
        route = self.alloc(name, rows=rows, cols=hidden, strict=False, physical_shape=(self.blen, hidden))
        self.gpt_oss_true_zero_vram_rows_v0(
            route,
            rows=list(range(self.blen)),
            hidden=hidden,
            zero_row=zero_row,
            name=f"{name}_zero",
        )
        fp_scratch = fp_scratch or self.fp_var(f"{name}_fp_row", size=self.mlen)
        gp_dst, gp_fp = self._reg.allocate_gp(2)
        try:
            # The scalar route weight depends only on pair_idx, so broadcast it into
            # fp_scratch once; S_MAP_V_FP only reads fp_scratch and never mutates it.
            self.fpvar_fill_from_fpram_asm(fp_scratch.address, weights_fp_base + pair_idx, self.mlen)
            for col_block in range(hidden // self.mlen):
                asm = IsaBuilder().comment(
                    f"GPT-OSS materialize route weight pair={pair_idx}, col_block={col_block}"
                )
                asm.instr("S_ADDI_INT", gp(gp_dst), gp(0), self._vram_matrix_row_addr(route, 0, col_block))
                asm.instr("S_ADDI_INT", gp(gp_fp), gp(0), fp_scratch.address)
                asm.instr("S_MAP_V_FP", gp(gp_dst), gp(gp_fp), 0)
                self._emit(asm)
        finally:
            self._reg.free_gp([gp_dst, gp_fp])
        return route

    def gpt_oss_materialize_route_weights_for_active_rows_v0(
        self,
        *,
        weights_fp_base: int,
        pair_indices: Sequence[int],
        active_rows: Sequence[int],
        rows: int,
        hidden: int,
        zero_row: FPVar | None = None,
        fp_scratch: FPVar | None = None,
        name: str = "gpt_oss_device_route_weights_grouped",
    ) -> VRAMMatrixVar:
        """Expand selected scalar route weights into specific active VRAM rows."""
        if len(pair_indices) != len(active_rows):
            raise ValueError(
                f"{name}: pair_indices={len(pair_indices)} active_rows={len(active_rows)} length mismatch"
            )
        if rows <= 0:
            raise ValueError(f"{name}: rows must be positive")
        if hidden % self.mlen != 0:
            raise ValueError(f"{name}: hidden={hidden} must be divisible by MLEN={self.mlen}")
        active_list = [int(row) for row in active_rows]
        pair_list = [int(pair) for pair in pair_indices]
        physical_rows = max(self.blen, math.ceil(rows / self.blen) * self.blen)
        if active_list and (min(active_list) < 0 or max(active_list) >= physical_rows):
            raise ValueError(f"{name}: active rows {active_list} exceed physical rows={physical_rows}")

        route = self.alloc(name, rows=rows, cols=hidden, strict=False, physical_shape=(physical_rows, hidden))
        self.gpt_oss_true_zero_vram_rows_v0(
            route,
            rows=list(range(physical_rows)),
            hidden=hidden,
            zero_row=zero_row,
            name=f"{name}_zero",
        )
        fp_scratch = fp_scratch or self.fp_var(f"{name}_fp_row", size=self.mlen)
        gp_dst, gp_fp = self._reg.allocate_gp(2)
        try:
            for pair_idx, active_row in zip(pair_list, active_list, strict=True):
                # Fill depends only on pair_idx (not col_block); broadcast once per pair.
                self.fpvar_fill_from_fpram_asm(fp_scratch.address, weights_fp_base + pair_idx, self.mlen)
                for col_block in range(hidden // self.mlen):
                    asm = IsaBuilder().comment(
                        f"GPT-OSS materialize route weight pair={pair_idx}, active_row={active_row}, col_block={col_block}"
                    )
                    asm.instr("S_ADDI_INT", gp(gp_dst), gp(0), self._vram_matrix_row_addr(route, active_row, col_block))
                    asm.instr("S_ADDI_INT", gp(gp_fp), gp(0), fp_scratch.address)
                    asm.instr("S_MAP_V_FP", gp(gp_dst), gp(gp_fp), 0)
                    self._emit(asm)
        finally:
            self._reg.free_gp([gp_dst, gp_fp])
        return route

    def gpt_oss_dynamic_expert_pair_v0(
        self,
        x: VRAMMatrixVar,
        weights: ExpertWeights,
        *,
        weight_table_bases: tuple[int, int, int],
        weight_table_strides: tuple[int, int, int],
        expert_indices_int_base: int,
        weights_fp_base: int,
        pair_idx: int,
        bias_tables: ExpertBiases | None,
        rows: int,
        intermediate: int,
        constants: GptOssFPConstants,
        zero_row: FPVar | None = None,
        route_fp_scratch: FPVar | None = None,
        activation_policy: str = "gpt_oss_clamp_gated",
        name: str = "gpt_oss_dynamic_expert_pair",
    ) -> VRAMMatrixVar:
        """Run one routed pair using true expert id from device V_TOPK output."""
        w_gate, w_up, w_down = weights
        gate_bias_table, up_bias_table, down_bias_table = bias_tables or (None, None, None)
        gate_base, up_base, down_base = weight_table_bases
        gate_stride, up_stride, down_stride = weight_table_strides
        projection_rows = max(self.mlen, x.physical_shape[0], math.ceil(rows / self.blen) * self.blen)

        gate = self.gpt_oss_dynamic_linear_projection_v0(
            x,
            w_gate,
            expert_indices_int_base=expert_indices_int_base,
            pair_idx=pair_idx,
            table_base=gate_base,
            per_expert_stride=gate_stride,
            name=f"{name}_gate",
            physical_shape=(projection_rows, w_gate.physical_shape[1]),
        )
        up = self.gpt_oss_dynamic_linear_projection_v0(
            x,
            w_up,
            expert_indices_int_base=expert_indices_int_base,
            pair_idx=pair_idx,
            table_base=up_base,
            per_expert_stride=up_stride,
            name=f"{name}_up",
            physical_shape=(projection_rows, w_up.physical_shape[1]),
        )
        if gate_bias_table is not None:
            self.gpt_oss_add_dynamic_expert_bias_v0(
                gate,
                gate_bias_table,
                expert_indices_int_base=expert_indices_int_base,
                pair_idx=pair_idx,
                rows=rows,
                width=intermediate,
                name=f"{name}_gate_bias",
            )
        if up_bias_table is not None:
            self.gpt_oss_add_dynamic_expert_bias_v0(
                up,
                up_bias_table,
                expert_indices_int_base=expert_indices_int_base,
                pair_idx=pair_idx,
                rows=rows,
                width=intermediate,
                name=f"{name}_up_bias",
            )
        hidden = self.moe_expert_activation_v0(
            gate,
            up,
            rows=rows,
            intermediate=intermediate,
            constants=constants,
            activation_policy=activation_policy,
            name=name,
        )
        out = self.gpt_oss_dynamic_linear_projection_v0(
            hidden,
            w_down,
            expert_indices_int_base=expert_indices_int_base,
            pair_idx=pair_idx,
            table_base=down_base,
            per_expert_stride=down_stride,
            name=f"{name}_out",
            physical_shape=(projection_rows, w_down.physical_shape[1]),
        )
        if down_bias_table is not None:
            self.gpt_oss_add_dynamic_expert_bias_v0(
                out,
                down_bias_table,
                expert_indices_int_base=expert_indices_int_base,
                pair_idx=pair_idx,
                rows=rows,
                width=w_down.physical_shape[1],
                name=f"{name}_down_bias",
            )
        route = self.gpt_oss_materialize_topk_route_weight_v0(
            weights_fp_base=weights_fp_base,
            pair_idx=pair_idx,
            rows=rows,
            hidden=w_down.physical_shape[1],
            zero_row=zero_row,
            fp_scratch=route_fp_scratch,
            name=f"{name}_route",
        )
        self.vram_mul(out, route, num_rows=rows)
        return out

    def gpt_oss_gather_token_rows_from_hbm_v0(
        self,
        x_input: InputVar,
        *,
        token_offsets_int_base: int,
        pair_count: int,
        hidden: int,
        zero_row: FPVar | None = None,
        name: str = "gpt_oss_gathered_x",
    ) -> VRAMMatrixVar:
        """Gather routed token rows from HBM into compact BF16 VRAM rows.

        ``token_offsets_int_base`` points into int SRAM.  Entry ``i`` contains
        the element offset of the source token row inside ``x_input``'s HBM
        element stream, i.e. ``token_index * hidden``.  The loop count remains
        compile-time fixed; only the HBM row offset is loaded at runtime.

        H_PREFETCH_V transfers four VLEN rows per call.  Each routed pair
        therefore owns a four-row slot.  The active row is the first row of the
        slot, while the remaining rows are cleared after prefetch.  This is
        intentionally wasteful but keeps the first L2 correctness path exact
        under the current ISA and avoids a copy-through-vector-ALU rounding
        step.
        """
        if pair_count <= 0:
            raise ValueError("pair_count must be positive")
        if hidden % self.mlen != 0:
            raise ValueError(f"gather hidden={hidden} must be divisible by MLEN={self.mlen}")
        if hidden > x_input.shape[1]:
            raise ValueError(f"gather hidden={hidden} exceeds x_input width={x_input.shape[1]}")

        logical_rows = pair_count * self.blen
        physical_rows = max(self.blen, math.ceil(logical_rows / self.blen) * self.blen)
        gathered = self.alloc(
            name,
            rows=logical_rows,
            cols=hidden,
            strict=False,
            physical_shape=(physical_rows, hidden),
        )

        x_rows, x_cols = x_input.physical_shape
        if x_cols != hidden:
            raise ValueError(f"gather currently expects x_input physical width {hidden}, got {x_cols}")
        num_col_blocks = hidden // self.mlen

        gp_table, gp_token_offset, gp_col, gp_offset, gp_dst, gp_scale, gp_stride = self._reg.allocate_gp(7)
        addr_reg = self._reg.allocate_addr(1)[0]
        try:
            asm = IsaBuilder().comment(
                f"GPT-OSS gather token rows from HBM: pairs={pair_count}, hidden={hidden}, slot_rows={self.blen}"
            )
            asm.instr("S_ADDI_INT", gp(gp_table), gp(0), token_offsets_int_base)
            asm.instr("S_ADDI_INT", gp(gp_scale), gp(0), x_rows * x_cols)
            asm.instr("C_SET_SCALE_REG", gp(gp_scale))
            asm.instr("S_ADDI_INT", gp(gp_stride), gp(0), x_cols)
            asm.instr("C_SET_STRIDE_REG", gp(gp_stride))
            asm.instr("S_ADDI_INT", gp(gp_offset), gp(0), x_input.hbm_addr)
            asm.instr("C_SET_ADDR_REG", areg(addr_reg), gp(0), gp(gp_offset))

            for pair_idx in range(pair_count):
                active_row = pair_idx * self.blen
                asm.comment(f"Gather pair slot {pair_idx}: dynamic token row offset from int SRAM")
                asm.instr("S_LD_INT", gp(gp_token_offset), gp(gp_table), pair_idx)
                for col_block in range(num_col_blocks):
                    col_offset = col_block * self.mlen
                    dst_addr = self._vram_matrix_row_addr(gathered, active_row, col_block)
                    asm.instr("S_ADDI_INT", gp(gp_col), gp(0), col_offset)
                    asm.instr("S_ADD_INT", gp(gp_offset), gp(gp_token_offset), gp(gp_col))
                    asm.instr("S_ADDI_INT", gp(gp_dst), gp(0), dst_addr)
                    asm.instr("H_PREFETCH_V", gp(gp_dst), gp(gp_offset), areg(addr_reg), 1, 0)
            self._emit(asm)
        finally:
            self._reg.free_gp([gp_table, gp_token_offset, gp_col, gp_offset, gp_dst, gp_scale, gp_stride])
            self._reg.free_addr([addr_reg])

        padding_rows = [
            pair_idx * self.blen + pad_idx
            for pair_idx in range(pair_count)
            for pad_idx in range(1, self.blen)
        ]
        if padding_rows:
            # Same true-FP-zero clear used by the VRAM gather path; keep it in one place.
            self.gpt_oss_true_zero_vram_rows_v0(
                gathered,
                rows=padding_rows,
                hidden=hidden,
                zero_row=zero_row,
                name=f"{name}_pad_zero",
            )

        return gathered

    def gpt_oss_gather_token_rows_from_vram_v0(
        self,
        x: VRAMMatrixVar,
        *,
        token_indices: Sequence[int],
        hidden: int,
        zero_row: FPVar | None = None,
        name: str = "gpt_oss_gathered_x_vram",
    ) -> VRAMMatrixVar:
        """Copy routed token rows from BF16 VRAM into BLEN-row pair slots.

        This is the decoder-block counterpart to
        :meth:`gpt_oss_gather_token_rows_from_hbm_v0`.  A real block feeds MoE
        from the VRAM-resident post-attention RMSNorm output, so this helper
        must not emit HBM prefetches, ``C_SET_SCALE_REG``, or activation
        quantization.  Each routed pair still owns one BLEN-row slot to match
        the existing dynamic expert-pair path; row 0 of each slot is active and
        padding rows are written with true zeros.
        """
        self._ensure_vram_sub_matrix_registered(x)
        if hidden % self.mlen != 0:
            raise ValueError(f"VRAM gather hidden={hidden} must be divisible by MLEN={self.mlen}")
        if hidden > x.shape[1]:
            raise ValueError(f"VRAM gather hidden={hidden} exceeds x width={x.shape[1]}")

        token_list = [int(token) for token in token_indices]
        if not token_list:
            raise ValueError("VRAM gather token_indices must be non-empty")
        if min(token_list) < 0 or max(token_list) >= x.physical_shape[0]:
            raise ValueError(f"VRAM gather token_indices {token_list} exceed x physical rows={x.physical_shape[0]}")

        pair_count = len(token_list)
        logical_rows = pair_count * self.blen
        physical_rows = max(self.blen, math.ceil(logical_rows / self.blen) * self.blen)
        gathered = self.alloc(
            name,
            rows=logical_rows,
            cols=hidden,
            strict=False,
            physical_shape=(physical_rows, hidden),
        )

        self.gpt_oss_true_zero_vram_rows_v0(
            gathered,
            rows=list(range(physical_rows)),
            hidden=hidden,
            zero_row=zero_row,
            name=f"{name}_zero",
        )

        num_col_blocks = hidden // self.mlen
        gp_dst, gp_src = self._reg.allocate_gp(2)
        try:
            asm = IsaBuilder().comment(
                f"GPT-OSS gather token rows from VRAM: pairs={pair_count}, hidden={hidden}, slot_rows={self.blen}"
            )
            for pair_idx, token_idx in enumerate(token_list):
                active_row = pair_idx * self.blen
                asm.comment(f"VRAM gather pair slot {pair_idx}: token row {token_idx}")
                for col_block in range(num_col_blocks):
                    dst_addr = self._vram_matrix_row_addr(gathered, active_row, col_block)
                    src_addr = self._vram_matrix_row_addr(x, token_idx, col_block)
                    asm.instr("S_ADDI_INT", gp(gp_dst), gp(0), dst_addr)
                    asm.instr("S_ADDI_INT", gp(gp_src), gp(0), src_addr)
                    asm.instr("V_ADD_VV", gp(gp_dst), gp(gp_dst), gp(gp_src), 0)
            self._emit(asm)
        finally:
            self._reg.free_gp([gp_dst, gp_src])

        return gathered

    def gpt_oss_true_zero_vram_rows_v0(
        self,
        matrix: VRAMMatrixVar,
        *,
        rows: Sequence[int],
        hidden: int,
        zero_row: FPVar | None = None,
        name: str = "gpt_oss_zero_rows",
    ) -> None:
        """Clear selected VRAM rows by mapping a true FP zero row.

        Do not use vector multiply-by-zero here: padding/gather slots can
        contain NaNs, and ``NaN * 0`` remains NaN.  This helper writes real
        zeros through ``S_MAP_V_FP`` and is therefore safe for gather padding
        and scatter accumulators.
        """
        if hidden % self.mlen != 0:
            raise ValueError(f"zero hidden={hidden} must be divisible by MLEN={self.mlen}")
        row_list = [int(row) for row in rows]
        if not row_list:
            return
        if min(row_list) < 0 or max(row_list) >= matrix.physical_shape[0]:
            raise ValueError(f"zero rows {row_list} exceed physical rows={matrix.physical_shape[0]}")

        num_col_blocks = hidden // self.mlen
        fp_zero_row = zero_row or self.fp_var(f"{name}_zero_row", size=self.mlen)
        gp_fp, gp_dst, gp_loop = self._reg.allocate_gp(3)
        try:
            asm = IsaBuilder().comment(f"GPT-OSS true-zero VRAM rows {row_list} in {matrix.name}")
            asm.instr("S_ADDI_INT", gp(gp_fp), gp(0), fp_zero_row.address)
            asm.instr("C_LOOP_START", gp(gp_loop), self.mlen)
            asm.instr("S_ST_FP", fp(0), gp(gp_fp), 0)
            asm.instr("S_ADDI_INT", gp(gp_fp), gp(gp_fp), 1)
            asm.instr("C_LOOP_END", gp(gp_loop))
            # The clear loop leaves gp_fp at address+mlen; reset it once here. The
            # map loop below never mutates gp_fp, so no per-iteration reset is needed.
            asm.instr("S_ADDI_INT", gp(gp_fp), gp(0), fp_zero_row.address)
            for row_idx in row_list:
                for col_block in range(num_col_blocks):
                    dst_addr = self._vram_matrix_row_addr(matrix, row_idx, col_block)
                    asm.instr("S_ADDI_INT", gp(gp_dst), gp(0), dst_addr)
                    asm.instr("S_MAP_V_FP", gp(gp_dst), gp(gp_fp), 0)
            self._emit(asm)
        finally:
            self._reg.free_gp([gp_fp, gp_dst, gp_loop])

    def gpt_oss_scatter_add_active_rows_v0(
        self,
        dst: VRAMMatrixVar,
        src: VRAMMatrixVar,
        *,
        token_indices: Sequence[int],
        active_rows: Sequence[int],
        hidden: int,
        name: str = "gpt_oss_scatter_add",
    ) -> None:
        """Add routed active slot rows into final token rows in VRAM.

        ``src`` is a 4-row-slot tensor produced by gather/expert execution;
        only one active row per slot is added into ``dst[token]``.  This is the
        VRAM-only half of L2 scatter-combine and intentionally does not emit
        HBM stores.
        """
        if hidden % self.mlen != 0:
            raise ValueError(f"scatter hidden={hidden} must be divisible by MLEN={self.mlen}")
        if len(token_indices) != len(active_rows):
            raise ValueError(f"token_indices={len(token_indices)} active_rows={len(active_rows)} length mismatch")

        token_list = [int(token) for token in token_indices]
        active_list = [int(row) for row in active_rows]
        if token_list and (min(token_list) < 0 or max(token_list) >= dst.physical_shape[0]):
            raise ValueError(f"scatter tokens {token_list} exceed dst physical rows={dst.physical_shape[0]}")
        if active_list and (min(active_list) < 0 or max(active_list) >= src.physical_shape[0]):
            raise ValueError(f"scatter active rows {active_list} exceed src physical rows={src.physical_shape[0]}")

        num_col_blocks = hidden // self.mlen
        gp_dst, gp_src = self._reg.allocate_gp(2)
        try:
            asm = IsaBuilder().comment(f"GPT-OSS VRAM scatter-add {name}: {len(token_list)} active rows")
            for token_idx, active_row in zip(token_list, active_list, strict=True):
                for col_block in range(num_col_blocks):
                    dst_addr = self._vram_matrix_row_addr(dst, token_idx, col_block)
                    src_addr = self._vram_matrix_row_addr(src, active_row, col_block)
                    asm.instr("S_ADDI_INT", gp(gp_dst), gp(0), dst_addr)
                    asm.instr("S_ADDI_INT", gp(gp_src), gp(0), src_addr)
                    asm.instr("V_ADD_VV", gp(gp_dst), gp(gp_dst), gp(gp_src), 0)
            self._emit(asm)
        finally:
            self._reg.free_gp([gp_dst, gp_src])

    def gpt_oss_clamp_gated_activation_v0(
        self,
        gate: VRAMMatrixVar,
        up: VRAMMatrixVar,
        *,
        rows: int,
        intermediate: int,
        constants: GptOssFPConstants,
        name: str,
    ) -> VRAMMatrixVar:
        """Emit GPT-OSS clamp-gated activation and return hidden in ``up``.

        Computes ``(clamp(up)+1) * clamp(gate) * sigmoid(1.702 * clamp(gate))``.
        The implementation uses ``exp(-1.702 * gate)`` and reciprocal to form
        sigmoid. Inputs and outputs are BF16 VRAM tensors.
        """
        self._validate_gpt_oss_constants(constants, rows)
        _, limit_pos, limit_neg, one, neg_alpha = constants
        active_rows = list(range(rows))
        physical_rows = max(self.mlen, math.ceil(rows / self.mlen) * self.mlen)
        num_col_blocks = math.ceil(intermediate / self.mlen)
        sigmoid = self.alloc(
            f"{name}_sigmoid",
            rows=rows,
            cols=intermediate,
            physical_shape=(physical_rows, intermediate),
            strict=False,
        )

        for col_block in range(num_col_blocks):
            self.tile_row_min_fp(gate, limit_pos, rows=active_rows, tile_col_idx=col_block)
            self.tile_row_min_fp(up, limit_pos, rows=active_rows, tile_col_idx=col_block)
            self.tile_row_max_fp(up, limit_neg, rows=active_rows, tile_col_idx=col_block)

        self.vram_fill_zero(sigmoid, rows=active_rows)
        self.vram_add(sigmoid, gate, num_rows=rows)

        for col_block in range(num_col_blocks):
            self.tile_row_mul_fp(sigmoid, neg_alpha, rows=active_rows, tile_col_idx=col_block)
            self.tile_row_exp(sigmoid, rows=active_rows, tile_col_idx=col_block)
            self.tile_row_add_fp(sigmoid, one, rows=active_rows, tile_col_idx=col_block)
            self.tile_row_reci(sigmoid, rows=active_rows, tile_col_idx=col_block)
        self.vram_mul(gate, sigmoid, num_rows=rows)

        for col_block in range(num_col_blocks):
            self.tile_row_add_fp(up, one, rows=active_rows, tile_col_idx=col_block)
        self.vram_mul(up, gate, num_rows=rows)
        return up

    def standard_swiglu_activation_v0(
        self,
        gate: VRAMMatrixVar,
        up: VRAMMatrixVar,
        *,
        rows: int,
        intermediate: int,
        constants: GptOssFPConstants,
        name: str,
    ) -> VRAMMatrixVar:
        """Emit standard SwiGLU activation and return hidden in ``up``.

        Computes ``silu(gate) * up = gate * sigmoid(gate) * up``.  Qwen-style
        experts use this non-clamped path, unlike GPT-OSS' clamp-gated variant.
        Inputs and outputs are BF16 VRAM tensors.
        """
        self._validate_standard_swiglu_constants(constants, rows)
        _, _unused_pos, _unused_neg, one, neg_one = constants
        active_rows = list(range(rows))
        physical_rows = max(self.mlen, math.ceil(rows / self.mlen) * self.mlen)
        num_col_blocks = math.ceil(intermediate / self.mlen)
        sigmoid = self.alloc(
            f"{name}_sigmoid",
            rows=rows,
            cols=intermediate,
            physical_shape=(physical_rows, intermediate),
            strict=False,
        )

        self.vram_fill_zero(sigmoid, rows=active_rows)
        self.vram_add(sigmoid, gate, num_rows=rows)

        for col_block in range(num_col_blocks):
            self.tile_row_mul_fp(sigmoid, neg_one, rows=active_rows, tile_col_idx=col_block)
            self.tile_row_exp(sigmoid, rows=active_rows, tile_col_idx=col_block)
            self.tile_row_add_fp(sigmoid, one, rows=active_rows, tile_col_idx=col_block)
            self.tile_row_reci(sigmoid, rows=active_rows, tile_col_idx=col_block)
        self.vram_mul(gate, sigmoid, num_rows=rows)
        self.vram_mul(up, gate, num_rows=rows)
        return up

    def gpt_oss_expert_v0(
        self,
        x: VRAMMatrixVar,
        weights: ExpertWeights,
        *,
        biases: ExpertBiases | None = None,
        rows: int,
        intermediate: int,
        constants: GptOssFPConstants,
        name: str,
    ) -> VRAMMatrixVar:
        """Emit one GPT-OSS expert and return its output."""
        w_gate, w_up, w_down = weights
        gate_bias, up_bias, down_bias = biases or (None, None, None)
        # The K-split projection path accumulates partial sums with a 64x64
        # block add.  Routed slots are often only 4/8/12 physical rows, so keep
        # expert projection outputs tile-backed to prevent the block add from
        # walking into the next column block.
        projection_rows = max(self.mlen, x.physical_shape[0], math.ceil(rows / self.blen) * self.blen)
        gate = self.linear_projection(
            x,
            w_gate,
            name=f"{name}_gate",
            physical_shape=(projection_rows, w_gate.physical_shape[1]),
        )
        up = self.linear_projection(
            x,
            w_up,
            name=f"{name}_up",
            physical_shape=(projection_rows, w_up.physical_shape[1]),
        )
        if gate_bias is not None:
            self.vram_add(gate, gate_bias, num_rows=rows)
        if up_bias is not None:
            self.vram_add(up, up_bias, num_rows=rows)
        hidden = self.gpt_oss_clamp_gated_activation_v0(
            gate,
            up,
            rows=rows,
            intermediate=intermediate,
            constants=constants,
            name=name,
        )
        out = self.linear_projection(
            hidden,
            w_down,
            name=f"{name}_out",
            physical_shape=(projection_rows, w_down.physical_shape[1]),
        )
        if down_bias is not None:
            self.vram_add(out, down_bias, num_rows=rows)
        return out

    def gpt_oss_moe_fixed_routing_v0(
        self,
        x: VRAMMatrixVar,
        experts: Sequence[ExpertWeights],
        route_weights: Sequence[VRAMMatrixVar],
        *,
        expert_biases: Sequence[ExpertBiases | None] | None = None,
        rows: int,
        intermediate: int,
        constants: GptOssFPConstants,
        name: str = "gpt_oss_moe",
    ) -> VRAMMatrixVar:
        """Emit fixed-routing MoE v0 and return the combined output.

        ``route_weights`` must already be loaded in VRAM and expanded to the
        expert output shape. The first expert output is used as the accumulator.
        """
        if not experts:
            raise ValueError("At least one expert is required")
        if len(experts) != len(route_weights):
            raise ValueError(f"experts={len(experts)} does not match route_weights={len(route_weights)}")
        if expert_biases is not None and len(expert_biases) != len(experts):
            raise ValueError(f"expert_biases={len(expert_biases)} does not match experts={len(experts)}")

        acc: VRAMMatrixVar | None = None
        for idx, (weights, route) in enumerate(zip(experts, route_weights, strict=True)):
            biases = None if expert_biases is None else expert_biases[idx]
            expert_out = self.gpt_oss_expert_v0(
                x,
                weights,
                biases=biases,
                rows=rows,
                intermediate=intermediate,
                constants=constants,
                name=f"{name}_expert{idx}",
            )
            self.vram_mul(expert_out, route, num_rows=rows)
            if acc is None:
                acc = expert_out
            else:
                self.vram_add(acc, expert_out, num_rows=rows)

        assert acc is not None
        return acc

    def moe_expert_activation_v0(
        self,
        gate: VRAMMatrixVar,
        up: VRAMMatrixVar,
        *,
        rows: int,
        intermediate: int,
        constants: GptOssFPConstants,
        activation_policy: str = "gpt_oss_clamp_gated",
        name: str,
    ) -> VRAMMatrixVar:
        """Generic substrate wrapper for expert activation backends."""
        if activation_policy == "gpt_oss_clamp_gated":
            return self.gpt_oss_clamp_gated_activation_v0(
                gate,
                up,
                rows=rows,
                intermediate=intermediate,
                constants=constants,
                name=name,
            )
        if activation_policy == "standard_swiglu":
            return self.standard_swiglu_activation_v0(
                gate,
                up,
                rows=rows,
                intermediate=intermediate,
                constants=constants,
                name=name,
            )
        raise NotImplementedError(
            "moe_expert_activation_v0 supports activation_policy in "
            "{'gpt_oss_clamp_gated', 'standard_swiglu'}, got "
            f"{activation_policy!r}"
        )


__all__ = ["ProgramRoutedMoeMixin"]
