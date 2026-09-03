"""Routed-MoE v0 program-builder helpers."""

from __future__ import annotations

import math
from collections.abc import Sequence

from compiler.aten.isa_builder import IsaBuilder, addr as areg, fp, gp
from compiler.aten.plena.vars import FPVar, InputVar, VRAMMatrixVar

GptOssFPConstants = tuple[FPVar, FPVar, FPVar, FPVar, FPVar]
ExpertWeights = tuple[InputVar, InputVar, InputVar]
ExpertBiases = tuple[VRAMMatrixVar | None, VRAMMatrixVar | None, VRAMMatrixVar | None]

# ============================================================================
# Stage markers
# ============================================================================

#: Prefix of the explicit stage marker comment. The emulator's stage profiler
#: (``transactional_emulator/src/stage_profile.rs``) keys on this exact string.
MOE_STAGE_MARKER_PREFIX = "@stage="

#: Every stage name the emulator's ``StageKind`` understands.
#:
#: Stage name that closes the MoE region. The emulator resolves it to a distinct
#: ``StageKind`` of its own -- not the unclassified fallback -- so an epilogue is
#: told apart from a region the classifier had no opinion about. Declared as a
#: constant because both repositories key on the exact string.
MOE_END_STAGE = "non_moe"

#: Marker emission is validated against this set, so a typo fails at ASM-gen time
#: instead of quietly collapsing a region into ``other``.
MOE_STAGES: frozenset[str] = frozenset(
    {
        "router_topk",
        # The MoE sublayer's input preparation, before any routing happens:
        # zeroing the residual buffer, copying the post-attention hidden state
        # into it, and the input RMSNorm with its norm-weight multiply. Distinct
        # from `accumulator_init`, which is the combine accumulator only -- these
        # rows are a residual copy that is added back *after* the experts run.
        "residual_setup",
        "accumulator_init",
        "gather",
        "expert_weight_address",
        "expert_weight_prefetch",
        "expert_projection",
        "expert_activation",
        "expert_bias",
        "expert_route_weight",
        "scatter_combine",
        "shared_expert_projection",
        "shared_expert_activation",
        "shared_expert_gate",
        # Terminator. Markers are sticky and there was no way to say "the MoE
        # region is over", so every instruction after the last MoE marker --
        # the lm_head, the next sublayer, anything at all -- kept that marker
        # to the end of the program, and that cost lands in the shared-vs-routed
        # ratio. A marker like any other, so it is emitted where the region ends
        # rather than inferred.
        #
        # Spelled literally, not as `MOE_END_STAGE`: both repositories recover
        # this set with a parser that refuses anything it cannot evaluate, so a
        # name reference here fails to parse rather than resolving. The constant
        # and this entry are held equal by a test instead.
        "non_moe",
    }
)


#: Comment token the emulator's ``extract_pair_id`` keys on to bucket work by
#: routed ``(token, expert)`` pair (``PAIR_ID_VOCABULARY`` in
#: ``transactional_emulator/src/stage_profile.rs``). Emitting it means "this
#: instruction belongs to routed pair N", so an emitter that is really indexing
#: plain token rows must not use it.
_PAIR_INDEX_LABEL = "pair="

#: What a stage-polymorphic emitter writes instead when its index is a token row
#: rather than a routed pair. Deliberately not a ``PAIR_ID_VOCABULARY`` member --
#: the point is that nothing picks it up.
_ROW_INDEX_LABEL = "row="

#: Stages whose index really is a routed ``(token, expert)`` pair. Everything
#: else reusing a routing emitter is indexing token rows.
_PAIR_INDEXED_STAGES: frozenset[str] = frozenset({"expert_route_weight"})


# ============================================================================
# V_TOPK routing policy
# ============================================================================

#: ``rmask`` value that makes V_TOPK read its shape from ``C_SET_TOPK_REG``
#: instead of the two-entry hardwired table.
_TOPK_POLICY_REGISTER_RMASK = 15

#: Bit position of ``num_experts`` inside the packed ``C_SET_TOPK_REG`` value.
#: Must match ``AcceleratorRegFile::topk_policy`` in the emulator.
_TOPK_POLICY_EXPERT_SHIFT = 8

#: Widest packed policy a *single* ``S_ADDI_INT`` can materialise.
#:
#: That instruction's immediate is the 18-bit ``IMM_2_WIDTH`` field at bits
#: 14..32 (``assembler/assembly_to_binary.py`` shifts it by ``opcode_width +
#: 2 * operand_width``; ``doc/configuration.svh`` sets the widths), *not* the
#: 22-bit ``IMM_WIDTH`` field that ``S_LUI_INT`` and ``C_LOOP_START`` carry. So
#: one instruction reaches 1023 experts, not 16383.
_TOPK_POLICY_SINGLE_ADDI_MAX_PACKED = (1 << 18) - 1

#: Widest packed policy this emitter will produce at all.
#:
#: Values above ``_TOPK_POLICY_SINGLE_ADDI_MAX_PACKED`` are still emitted
#: correctly: ``IsaBuilder.render`` runs ``legalize_large_immediates``, which
#: rewrites an over-wide ``S_ADDI_INT`` into ``S_LUI_INT`` + ``S_ADDI_INT``.
#: They only lose the single-instruction property the 8-bit shift buys.
_TOPK_POLICY_MAX_PACKED = (1 << 22) - 1


def _pack_topk_policy(num_experts: int, top_k: int) -> int:
    """Pack ``(num_experts, top_k)`` for ``C_SET_TOPK_REG``.

    Layout is ``(num_experts << 8) | top_k``. The emulator unpacks it in
    ``AcceleratorRegFile::topk_policy``; nothing but the unit tests on either side
    checks that the two agree, so the bounds are asserted here rather than left to
    produce a silently truncated policy.
    """
    if not 0 < top_k <= num_experts:
        raise ValueError(f"top_k={top_k} must be in [1, num_experts={num_experts}]")
    if top_k >= (1 << _TOPK_POLICY_EXPERT_SHIFT):
        raise ValueError(
            f"top_k={top_k} does not fit {_TOPK_POLICY_EXPERT_SHIFT} bits; the C_SET_TOPK_REG packing bounds it at 255"
        )
    packed = (num_experts << _TOPK_POLICY_EXPERT_SHIFT) | top_k
    if packed > _TOPK_POLICY_MAX_PACKED:
        raise ValueError(
            f"num_experts={num_experts} packs to {packed}, past the C_SET_TOPK_REG packing "
            f"ceiling of {_TOPK_POLICY_MAX_PACKED}; it tops out at "
            f"{_TOPK_POLICY_MAX_PACKED >> _TOPK_POLICY_EXPERT_SHIFT} experts"
        )
    return packed


def moe_stage_marker(stage: str, detail: str = "") -> str:
    """Format the explicit stage marker comment for ``stage``.

    The marker is *authoritative and sticky*: once a program contains any marker,
    the emulator stops applying its legacy substring rules entirely and every
    instruction is attributed to the most recent marker. So a marker must be
    emitted whenever the stage changes, and must not be emitted mid-stage.

    The corollary is that work done inside a general-purpose helper called from a
    marked region -- ``linear_projection``'s HBM weight prefetch, for instance --
    is attributed to the enclosing marker rather than to
    ``expert_weight_prefetch``. That is why the routed path marks its own dynamic
    prefetch explicitly; the shared path deliberately does not, folding weight
    traffic into ``shared_expert_projection`` where it belongs.

    Because markers are sticky and a program does not end where its MoE region
    does, the last MoE marker otherwise runs to the end of the file. Emit
    :func:`moe_end_marker` at the point the region closes; see
    :data:`MOE_END_STAGE`.
    """
    if stage not in MOE_STAGES:
        raise ValueError(f"unknown MoE stage {stage!r}; expected one of {sorted(MOE_STAGES)}")
    return f"{MOE_STAGE_MARKER_PREFIX}{stage}" + (f" {detail}" if detail else "")


def moe_end_marker(detail: str = "") -> str:
    """Close the MoE region, so what follows is billed to no MoE stage.

    A separate entry point rather than ``moe_stage_marker(MOE_END_STAGE)``
    because closing a region and setting one are different acts, and the caller
    that has to remember this is the one assembling a decoder program, not the
    one writing an emitter. Emit it once, after the last MoE work in the
    program -- the combine, or the shared/routed add.

    Omitting it is not silent: the profile reports how many instructions carry
    the final marker, so an epilogue billed to a MoE stage is visible in the
    JSON without reading the source that produced it.
    """
    return moe_stage_marker(MOE_END_STAGE, detail)


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

    def moe_router_logits_bf16_v0(
        self,
        x: VRAMMatrixVar,
        router_weight_rows: VRAMMatrixVar,
        *,
        rows: int,
        hidden: int,
        num_experts: int,
        policy_name: str = "gpt_oss",
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
                f"router_weight_rows must have shape at least ({num_experts}, {hidden}), got {router_weight_rows.shape}"
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
                moe_stage_marker(
                    "router_topk",
                    f"[{policy_name}] BF16 vector-dot logits: rows={rows}, hidden={hidden}, experts={num_experts}",
                )
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
        policy_name: str = "gpt_oss",
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
                moe_stage_marker(
                    "router_topk",
                    f"[{policy_name}] {label} logits pack: rows={rows}, experts={num_experts}, blocks={expert_blocks}",
                )
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

        # `linear_projection_bf16*` is a general helper with no marker of its own,
        # so without this the router GEMM inherits whatever marker preceded the call.
        self._emit(
            IsaBuilder().comment(
                moe_stage_marker(
                    "router_topk",
                    f"[qwen3] BF16 matrix logits: rows={rows}, hidden={hidden}, experts={num_experts}",
                )
            )
        )

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


        self._emit(
            IsaBuilder().comment(
                moe_stage_marker(
                    "router_topk",
                    f"[qwen3] packed-skinny BF16 logits: rows={rows}, hidden={hidden}, experts={num_experts}",
                )
            )
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

    def moe_router_select_v0(
        self,
        logits: VRAMMatrixVar,
        *,
        token_idx: int,
        weights_fp_base: int,
        indices_int_base: int,
        num_experts: int = 32,
        top_k: int = 4,
        policy_name: str = "gpt_oss",
        name: str = "moe_router_select",
    ) -> None:
        """Emit V_TOPK for one router-logit row.

        V_TOPK v0 reads one BF16 router-logit row from VRAM, performs a
        linear-scan top-k with low-index tie break, stores the selected expert
        ids to INT SRAM, and stores the softmax-over-selected weights to FP
        SRAM.  The instruction intentionally keeps router/top-k on the BF16
        path and does not touch MX scale state.

        ``(num_experts, top_k)`` is arbitrary. The two hardwired ``rmask``
        policies are used when they match exactly; every other shape goes through
        ``C_SET_TOPK_REG`` and ``rmask=15``.
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
        if top_k < 1 or top_k > num_experts:
            raise ValueError(f"top_k={top_k} must be in [1, num_experts={num_experts}]")

        # The fixed rmask table is preferred where it applies so existing GPT-OSS
        # and Qwen3 programs keep emitting byte-identical ASM.
        policy_rmask = {(32, 4): 0, (128, 8): 1}.get((num_experts, top_k))
        packed_policy = None
        if policy_rmask is None:
            policy_rmask = _TOPK_POLICY_REGISTER_RMASK
            packed_policy = _pack_topk_policy(num_experts, top_k)

        gp_weights, gp_logits, gp_indices = self._reg.allocate_gp(3)
        try:
            asm = IsaBuilder().comment(
                moe_stage_marker(
                    "router_topk",
                    f"[{policy_name}] V_TOPK {name}: token={token_idx}, experts={num_experts}, "
                    f"top_k={top_k}, weights_fp={weights_fp_base}, indices_int={indices_int_base}",
                )
            )
            if packed_policy is not None:
                # C_SET_TOPK_REG is sticky, but hoisting it out of the per-token loop
                # would mean tracking the live register value across every other
                # emitter that can run in between; two scalar instructions are cheap.
                asm.instr("S_ADDI_INT", gp(gp_weights), gp(0), packed_policy)
                asm.instr("C_SET_TOPK_REG", gp(gp_weights))
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
            moe_stage_marker(
                "expert_weight_address",
                f"{name}: pair={pair_idx}, table_base={table_base}, stride={per_expert_stride}",
            )
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
            moe_stage_marker(
                "expert_weight_address",
                f"{name}: table lookup pair={pair_idx}, base_table_int={expert_base_table_int_base}",
            )
        )
        asm.instr("S_ADDI_INT", gp(gp_table), gp(0), expert_indices_int_base)
        asm.instr("S_LD_INT", gp(gp_expert), gp(gp_table), pair_idx)
        asm.instr("S_LD_INT", gp(gp_base), gp(gp_expert), expert_base_table_int_base)
        asm.instr("C_SET_ADDR_REG", areg(addr_reg), gp(0), gp(gp_base))

    def moe_expert_id_to_weight_base_v0(
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

    def _moe_dynamic_load_sub_matrix_col_v0(
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
                moe_stage_marker(
                    "expert_weight_prefetch",
                    f"dynamic HBM weight prefetch: template={weight_template.name}, pair={pair_idx}, col={col_idx}",
                )
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
            # Address selection has its own stage marker. Hand attribution back
            # before setup and H_PREFETCH_M so transfer cycles and bytes are not
            # silently charged to ``expert_weight_address``.
            asm.comment(
                moe_stage_marker(
                    "expert_weight_prefetch",
                    f"issue dynamic HBM weight tile: template={weight_template.name}, "
                    f"pair={pair_idx}, col={col_idx}",
                )
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

    def moe_dynamic_vram_sub_projection_to_v0(
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
        self._moe_dynamic_load_sub_matrix_col_v0(
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
        # The helper above marked `expert_weight_prefetch`; hand the stage back
        # before the GEMM, or every matmul in this tile is billed to prefetch. It
        # cannot move into `vram_sub_projection_to`, which non-MoE programs share
        # and which must stay marker-free.
        self._emit(
            IsaBuilder().comment(
                moe_stage_marker("expert_projection", f"{name}: pair={pair_idx}, col={weight_col_idx}")
            )
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

    def moe_dynamic_linear_projection_v0(
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
        reuse_weight_across_row_blocks: bool = False,
        weight_panel_mode: str = "blocking",
        panel_k_tiles: int | None = None,
    ) -> VRAMMatrixVar:
        """Tiled linear projection with runtime expert-id weight selection.

        ``reuse_weight_across_row_blocks`` is the fixed-route grouped-execution
        contract.  A weight column tile is loaded into Matrix SRAM once and is
        consumed by every token-row block belonging to the same expert before
        the next tile replaces it.  The default remains the historical
        load-per-row-block behavior so existing programs are unchanged.

        ``weight_panel_mode="pingpong"`` reserves two disjoint Matrix-SRAM
        panels. It emits the next fill before consuming the current panel and
        reuses a slot only after its previous panel has been consumed.
        Functional execution remains dependency-safe; a dependency-aware
        simulator may overlap a pending fill with work on the ready panel.
        """
        if weight_panel_mode not in ("blocking", "pingpong"):
            raise ValueError(f"unsupported weight_panel_mode={weight_panel_mode!r}")
        if weight_panel_mode != "blocking" and not reuse_weight_across_row_blocks:
            raise ValueError("buffered weight panels require grouped row-block reuse")
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
        max_k_tiles = self.mram_tile_capacity if panel_k_tiles is None else int(panel_k_tiles)
        if max_k_tiles <= 0:
            raise ValueError(f"panel_k_tiles must be positive, got {max_k_tiles}")
        if max_k_tiles > self.mram_tile_capacity:
            raise ValueError(
                f"panel_k_tiles={max_k_tiles} exceeds mram_tile_capacity={self.mram_tile_capacity}"
            )

        output = self.alloc(
            name,
            rows,
            out_features,
            strict=False,
            physical_shape=(physical_rows, physical_out_features),
        )

        if weight_panel_mode != "blocking":
            panel_tiles = 4 if panel_k_tiles is None else int(panel_k_tiles)
            if panel_tiles <= 0:
                raise ValueError(f"panel_k_tiles must be positive, got {panel_tiles}")
            panel_depth = 2
            if panel_depth * panel_tiles > self.mram_tile_capacity:
                raise ValueError(
                    "buffered panels exceed Matrix-SRAM capacity: "
                    f"{panel_depth}*{panel_tiles} > mram_tile_capacity={self.mram_tile_capacity}"
                )

            super().reset_mram()
            panel_elems = panel_tiles * mlen * mlen
            panel_bases = tuple(
                self.mram_allocator.allocate(f"{name}_panel{slot}", panel_elems)
                for slot in range(panel_depth)
            )
            panels = []
            for k_chunk_idx, k_block_start in enumerate(range(0, num_k_tiles, panel_tiles)):
                k_block_count = min(panel_tiles, num_k_tiles - k_block_start)
                for col_idx in range(num_col_blocks):
                    panels.append((k_chunk_idx, k_block_start, k_block_count, col_idx))

            def prefetch_panel(panel, mram_start_addr: int) -> None:
                _chunk_idx, k_block_start, k_block_count, col_idx = panel
                self._moe_dynamic_load_sub_matrix_col_v0(
                    weight_template=weight_template,
                    col_idx=col_idx,
                    expert_indices_int_base=expert_indices_int_base,
                    pair_idx=pair_idx,
                    table_base=table_base,
                    per_expert_stride=per_expert_stride,
                    expert_base_table_int_base=expert_base_table_int_base,
                    mram_start_addr=mram_start_addr,
                    k_block_start=k_block_start,
                    k_block_count=k_block_count,
                    name=f"{name}_pair{pair_idx}_pingpong",
                )

            def consume_panel(
                panel,
                mram_start_addr: int,
                target: VRAMMatrixVar,
                row_idx: int,
                target_row_idx: int,
                target_col_idx: int,
            ) -> None:
                _chunk_idx, k_block_start, k_block_count, col_idx = panel
                self._emit(
                    IsaBuilder().comment(
                        moe_stage_marker(
                            "expert_projection",
                            f"{name}: pingpong resident weight pair={pair_idx}, col={col_idx}, "
                            f"row_block={row_idx}, mram={mram_start_addr}",
                        )
                    )
                )
                super(ProgramRoutedMoeMixin, self).vram_sub_projection_to(
                    vram_mat_name=input_var.name,
                    vram_row_idx=row_idx,
                    mram_mat_name=weight_template.name,
                    mram_col_idx=col_idx,
                    target_matrix=target.name,
                    target_row_idx=target_row_idx,
                    target_col_idx=target_col_idx,
                    k_block_start=k_block_start,
                    k_block_count=k_block_count,
                    resident_mram_start_addr=mram_start_addr,
                )

            temp = self.alloc(f"{name}_temp", mlen, mlen) if num_k_tiles > panel_tiles else None
            initial_fill = min(panel_depth, len(panels))
            for panel_idx in range(initial_fill):
                prefetch_panel(panels[panel_idx], panel_bases[panel_idx])
            for panel_idx, panel in enumerate(panels):
                current_base = panel_bases[panel_idx % panel_depth]
                k_chunk_idx, _k_start, _k_count, col_idx = panel
                for row_idx in range(num_row_blocks):
                    if k_chunk_idx == 0:
                        consume_panel(panel, current_base, output, row_idx, row_idx, col_idx)
                    else:
                        assert temp is not None
                        consume_panel(panel, current_base, temp, row_idx, 0, 0)
                        self.vram_block_add_to(output, row_idx, col_idx, temp, 0, 0, output, row_idx, col_idx)
                next_panel_idx = panel_idx + panel_depth
                if next_panel_idx < len(panels):
                    prefetch_panel(panels[next_panel_idx], current_base)
            if temp is not None:
                self.free_tensor(temp)
            return output

        def emit_projection(
            row_idx,
            col_idx,
            target,
            target_row_idx,
            target_col_idx,
            *,
            load_weight: bool,
            **k_split,
        ) -> None:
            if load_weight:
                self.moe_dynamic_vram_sub_projection_to_v0(
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
                return

            self._emit(
                IsaBuilder().comment(
                    moe_stage_marker(
                        "expert_projection",
                        f"{name}: resident weight pair={pair_idx}, col={col_idx}, row_block={row_idx}",
                    )
                )
            )
            super(ProgramRoutedMoeMixin, self).vram_sub_projection_to(
                vram_mat_name=input_var.name,
                vram_row_idx=row_idx,
                mram_mat_name=weight_template.name,
                mram_col_idx=col_idx,
                target_matrix=target.name,
                target_row_idx=target_row_idx,
                target_col_idx=target_col_idx,
                **k_split,
            )

        if num_k_tiles <= max_k_tiles:
            for col_idx in range(num_col_blocks):
                for row_idx in range(num_row_blocks):
                    emit_projection(
                        row_idx,
                        col_idx,
                        output,
                        row_idx,
                        col_idx,
                        load_weight=not reuse_weight_across_row_blocks or row_idx == 0,
                    )
            return output

        temp = self.alloc(f"{name}_temp", mlen, mlen)
        for k_chunk_idx, k_block_start in enumerate(range(0, num_k_tiles, max_k_tiles)):
            k_block_count = min(max_k_tiles, num_k_tiles - k_block_start)
            k_split = {"k_block_start": k_block_start, "k_block_count": k_block_count}
            for col_idx in range(num_col_blocks):
                for row_idx in range(num_row_blocks):
                    if k_chunk_idx == 0:
                        emit_projection(
                            row_idx,
                            col_idx,
                            output,
                            row_idx,
                            col_idx,
                            load_weight=not reuse_weight_across_row_blocks or row_idx == 0,
                            **k_split,
                        )
                    else:
                        emit_projection(
                            row_idx,
                            col_idx,
                            temp,
                            0,
                            0,
                            load_weight=not reuse_weight_across_row_blocks or row_idx == 0,
                            **k_split,
                        )
                        self.vram_block_add_to(output, row_idx, col_idx, temp, 0, 0, output, row_idx, col_idx)
        self.free_tensor(temp)
        return output

    def moe_add_dynamic_expert_bias_v0(
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
            asm = IsaBuilder().comment(
                moe_stage_marker("expert_bias", f"dynamic expert bias add {name}: pair={pair_idx}, rows={rows}")
            )
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

    def moe_materialize_topk_route_weight_v0(
        self,
        *,
        weights_fp_base: int,
        pair_idx: int,
        rows: int,
        hidden: int,
        zero_row: FPVar | None = None,
        fp_scratch: FPVar | None = None,
        policy_name: str = "gpt_oss",
        name: str = "gpt_oss_device_route_weight",
    ) -> VRAMMatrixVar:
        """Expand device V_TOPK scalar weight into a VRAM route matrix."""
        if rows > self.blen:
            raise ValueError(f"{name}: v0 expects one routed pair slot (rows<=BLEN), got rows={rows}")
        if hidden % self.mlen != 0:
            raise ValueError(f"{name}: hidden={hidden} must be divisible by MLEN={self.mlen}")
        route = self.alloc(name, rows=rows, cols=hidden, strict=False, physical_shape=(self.blen, hidden))
        self.moe_true_zero_vram_rows_v0(
            route,
            rows=list(range(self.blen)),
            hidden=hidden,
            zero_row=zero_row,
            policy_name=policy_name,
            stage="expert_route_weight",
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
                    moe_stage_marker(
                        "expert_route_weight",
                        f"[{policy_name}] materialize route weight pair={pair_idx}, col_block={col_block}",
                    )
                )
                asm.instr("S_ADDI_INT", gp(gp_dst), gp(0), self._vram_matrix_row_addr(route, 0, col_block))
                asm.instr("S_ADDI_INT", gp(gp_fp), gp(0), fp_scratch.address)
                asm.instr("S_MAP_V_FP", gp(gp_dst), gp(gp_fp), 0)
                self._emit(asm)
        finally:
            self._reg.free_gp([gp_dst, gp_fp])
        return route

    def moe_materialize_route_weights_for_active_rows_v0(
        self,
        *,
        weights_fp_base: int,
        pair_indices: Sequence[int],
        active_rows: Sequence[int],
        rows: int,
        hidden: int,
        zero_row: FPVar | None = None,
        fp_scratch: FPVar | None = None,
        policy_name: str = "gpt_oss",
        stage: str,
        name: str = "moe_device_row_weights_grouped",
    ) -> VRAMMatrixVar:
        """Expand selected scalar per-row FP weights into specific active VRAM rows.

        ``stage`` exists because this broadcast is not exclusive to routing. The
        shared-expert sigmoid gate produces the same thing -- one FP scalar per
        token, broadcast across hidden -- and reuses this emitter. Without the
        parameter every gate instruction bills to ``expert_route_weight``, which is
        both wrong and specifically misleading: a program with no routing at all
        would appear to spend time computing route weights.

        It is deliberately **required**: under sticky marker attribution a wrong
        stage is silent -- the totals still add up -- so the omission has to be
        caught where it is written, not where it is read.

        ``stage`` also picks the *index label*: stages in ``_PAIR_INDEXED_STAGES``
        emit ``pair=``, which the emulator's ``extract_pair_id`` buckets by routed
        ``(token, expert)`` pair; everything else emits ``row=``. A shared-gate row
        index emitted as ``pair=`` invents pairs that do not exist.
        """
        if len(pair_indices) != len(active_rows):
            raise ValueError(f"{name}: pair_indices={len(pair_indices)} active_rows={len(active_rows)} length mismatch")
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
        self.moe_true_zero_vram_rows_v0(
            route,
            rows=list(range(physical_rows)),
            hidden=hidden,
            zero_row=zero_row,
            policy_name=policy_name,
            stage=stage,
            name=f"{name}_zero",
        )
        index_label = _PAIR_INDEX_LABEL if stage in _PAIR_INDEXED_STAGES else _ROW_INDEX_LABEL
        fp_scratch = fp_scratch or self.fp_var(f"{name}_fp_row", size=self.mlen)
        gp_dst, gp_fp = self._reg.allocate_gp(2)
        try:
            for pair_idx, active_row in zip(pair_list, active_list, strict=True):
                # Fill depends only on pair_idx (not col_block); broadcast once per pair.
                self.fpvar_fill_from_fpram_asm(fp_scratch.address, weights_fp_base + pair_idx, self.mlen)
                for col_block in range(hidden // self.mlen):
                    asm = IsaBuilder().comment(
                        moe_stage_marker(
                            stage,
                            f"[{policy_name}] materialize row weight {index_label}{pair_idx}, "
                            f"active_row={active_row}, col_block={col_block}",
                        )
                    )
                    asm.instr("S_ADDI_INT", gp(gp_dst), gp(0), self._vram_matrix_row_addr(route, active_row, col_block))
                    asm.instr("S_ADDI_INT", gp(gp_fp), gp(0), fp_scratch.address)
                    asm.instr("S_MAP_V_FP", gp(gp_dst), gp(gp_fp), 0)
                    self._emit(asm)
        finally:
            self._reg.free_gp([gp_dst, gp_fp])
        return route

    def moe_dynamic_expert_pair_v0(
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
        policy_name: str = "gpt_oss",
        activation_policy: str = "gpt_oss_clamp_gated",
        name: str = "moe_dynamic_expert_pair",
    ) -> VRAMMatrixVar:
        """Run one routed pair using true expert id from device V_TOPK output."""
        w_gate, w_up, w_down = weights
        gate_bias_table, up_bias_table, down_bias_table = bias_tables or (None, None, None)
        gate_base, up_base, down_base = weight_table_bases
        gate_stride, up_stride, down_stride = weight_table_strides
        projection_rows = max(self.mlen, x.physical_shape[0], math.ceil(rows / self.blen) * self.blen)

        gate = self.moe_dynamic_linear_projection_v0(
            x,
            w_gate,
            expert_indices_int_base=expert_indices_int_base,
            pair_idx=pair_idx,
            table_base=gate_base,
            per_expert_stride=gate_stride,
            name=f"{name}_gate",
            physical_shape=(projection_rows, w_gate.physical_shape[1]),
        )
        up = self.moe_dynamic_linear_projection_v0(
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
            self.moe_add_dynamic_expert_bias_v0(
                gate,
                gate_bias_table,
                expert_indices_int_base=expert_indices_int_base,
                pair_idx=pair_idx,
                rows=rows,
                width=intermediate,
                name=f"{name}_gate_bias",
            )
        if up_bias_table is not None:
            self.moe_add_dynamic_expert_bias_v0(
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
            stage="expert_activation",
            name=name,
        )
        out = self.moe_dynamic_linear_projection_v0(
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
            self.moe_add_dynamic_expert_bias_v0(
                out,
                down_bias_table,
                expert_indices_int_base=expert_indices_int_base,
                pair_idx=pair_idx,
                rows=rows,
                width=w_down.physical_shape[1],
                name=f"{name}_down_bias",
            )
        route = self.moe_materialize_topk_route_weight_v0(
            weights_fp_base=weights_fp_base,
            pair_idx=pair_idx,
            rows=rows,
            hidden=w_down.physical_shape[1],
            zero_row=zero_row,
            fp_scratch=route_fp_scratch,
            policy_name=policy_name,
            name=f"{name}_route",
        )
        # Re-mark: `vram_mul` is a general-purpose helper with no marker of its own.
        self._emit(IsaBuilder().comment(moe_stage_marker("expert_route_weight", f"[{policy_name}] apply {name}")))
        self.vram_mul(out, route, num_rows=rows)
        return out

    def moe_dynamic_expert_group_v0(
        self,
        x: VRAMMatrixVar,
        weights: ExpertWeights,
        *,
        weight_table_bases: tuple[int, int, int],
        weight_table_strides: tuple[int, int, int],
        expert_indices_int_base: int,
        weights_fp_base: int,
        pair_indices: Sequence[int],
        bias_tables: ExpertBiases | None,
        rows: int,
        intermediate: int,
        constants: GptOssFPConstants,
        zero_row: FPVar | None = None,
        route_fp_scratch: FPVar | None = None,
        policy_name: str = "gpt_oss",
        activation_policy: str = "gpt_oss_clamp_gated",
        weight_panel_mode: str = "blocking",
        panel_k_tiles: int | None = None,
        name: str = "moe_dynamic_expert_group",
    ) -> VRAMMatrixVar:
        """Run one fixed-route expert for a compact group of token rows.

        The caller guarantees that every entry in ``pair_indices`` names the
        same expert in integer SRAM.  The first pair resolves the expert's HBM
        base once; its gate/up/down tiles remain in Matrix SRAM while all
        compact token rows consume them.  Distinct per-pair route weights are
        then materialized row by row before scatter-add.

        This helper intentionally does not sort device-selected routing.  It is
        legal for fixed-route/trace replay where grouping is known at compile
        time; a runtime router needs a separate sort/segment mechanism before
        it may call the same execution primitive.
        """
        pair_list = [int(pair) for pair in pair_indices]
        if rows <= 0:
            raise ValueError(f"{name}: rows must be positive")
        if len(pair_list) != rows:
            raise ValueError(f"{name}: pair_indices={len(pair_list)} must equal rows={rows}")
        if len(set(pair_list)) != len(pair_list) or min(pair_list) < 0:
            raise ValueError(f"{name}: pair_indices must be distinct non-negative values")
        if x.shape[0] != rows:
            raise ValueError(f"{name}: compact x rows={x.shape[0]} must equal rows={rows}")
        if bias_tables is not None and rows > self.blen:
            raise ValueError(
                f"{name}: grouped execution with expert bias supports at most "
                f"BLEN={self.blen} rows, got rows={rows}"
            )

        representative_pair = pair_list[0]
        w_gate, w_up, w_down = weights
        gate_bias_table, up_bias_table, down_bias_table = bias_tables or (None, None, None)
        gate_base, up_base, down_base = weight_table_bases
        gate_stride, up_stride, down_stride = weight_table_strides
        projection_rows = max(self.mlen, x.physical_shape[0], math.ceil(rows / self.blen) * self.blen)

        def project(input_var, weight, table_base, stride, suffix, output_shape):
            return self.moe_dynamic_linear_projection_v0(
                input_var,
                weight,
                expert_indices_int_base=expert_indices_int_base,
                pair_idx=representative_pair,
                table_base=table_base,
                per_expert_stride=stride,
                name=f"{name}_{suffix}",
                physical_shape=output_shape,
                reuse_weight_across_row_blocks=True,
                weight_panel_mode=weight_panel_mode,
                panel_k_tiles=panel_k_tiles,
            )

        gate = project(
            x,
            w_gate,
            gate_base,
            gate_stride,
            "gate",
            (projection_rows, w_gate.physical_shape[1]),
        )
        up = project(
            x,
            w_up,
            up_base,
            up_stride,
            "up",
            (projection_rows, w_up.physical_shape[1]),
        )
        if gate_bias_table is not None:
            self.moe_add_dynamic_expert_bias_v0(
                gate,
                gate_bias_table,
                expert_indices_int_base=expert_indices_int_base,
                pair_idx=representative_pair,
                rows=rows,
                width=intermediate,
                name=f"{name}_gate_bias",
            )
        if up_bias_table is not None:
            self.moe_add_dynamic_expert_bias_v0(
                up,
                up_bias_table,
                expert_indices_int_base=expert_indices_int_base,
                pair_idx=representative_pair,
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
            policy_name=policy_name,
            stage="expert_activation",
            name=name,
        )
        out = project(
            hidden,
            w_down,
            down_base,
            down_stride,
            "out",
            (projection_rows, w_down.physical_shape[1]),
        )
        if down_bias_table is not None:
            self.moe_add_dynamic_expert_bias_v0(
                out,
                down_bias_table,
                expert_indices_int_base=expert_indices_int_base,
                pair_idx=representative_pair,
                rows=rows,
                width=w_down.physical_shape[1],
                name=f"{name}_down_bias",
            )

        route = self.moe_materialize_route_weights_for_active_rows_v0(
            weights_fp_base=weights_fp_base,
            pair_indices=pair_list,
            active_rows=list(range(rows)),
            rows=rows,
            hidden=w_down.physical_shape[1],
            zero_row=zero_row,
            fp_scratch=route_fp_scratch,
            policy_name=policy_name,
            stage="expert_route_weight",
            name=f"{name}_route",
        )
        self._emit(
            IsaBuilder().comment(
                moe_stage_marker(
                    "expert_route_weight",
                    f"[{policy_name}] apply grouped route weights: rows={rows}, representative_pair={representative_pair}",
                )
            )
        )
        self.vram_mul(out, route, num_rows=rows)
        return out

    def moe_gather_token_rows_from_hbm_v0(
        self,
        x_input: InputVar,
        *,
        token_offsets_int_base: int,
        pair_count: int,
        hidden: int,
        zero_row: FPVar | None = None,
        policy_name: str = "gpt_oss",
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
                moe_stage_marker(
                    "gather",
                    f"[{policy_name}] gather token rows from HBM: pairs={pair_count}, "
                    f"hidden={hidden}, slot_rows={self.blen}",
                )
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
            pair_idx * self.blen + pad_idx for pair_idx in range(pair_count) for pad_idx in range(1, self.blen)
        ]
        if padding_rows:
            # Same true-FP-zero clear used by the VRAM gather path; keep it in one place.
            self.moe_true_zero_vram_rows_v0(
                gathered,
                rows=padding_rows,
                hidden=hidden,
                zero_row=zero_row,
                policy_name=policy_name,
                stage="gather",
                name=f"{name}_pad_zero",
            )

        return gathered

    def moe_gather_token_rows_from_vram_v0(
        self,
        x: VRAMMatrixVar,
        *,
        token_indices: Sequence[int],
        hidden: int,
        zero_row: FPVar | None = None,
        policy_name: str = "gpt_oss",
        name: str = "gpt_oss_gathered_x_vram",
    ) -> VRAMMatrixVar:
        """Copy routed token rows from BF16 VRAM into BLEN-row pair slots.

        This is the decoder-block counterpart to
        :meth:`moe_gather_token_rows_from_hbm_v0`.  A real block feeds MoE
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

        self.moe_true_zero_vram_rows_v0(
            gathered,
            rows=list(range(physical_rows)),
            hidden=hidden,
            zero_row=zero_row,
            policy_name=policy_name,
            stage="gather",
            name=f"{name}_zero",
        )

        num_col_blocks = hidden // self.mlen
        gp_dst, gp_src = self._reg.allocate_gp(2)
        try:
            asm = IsaBuilder().comment(
                moe_stage_marker(
                    "gather",
                    f"[{policy_name}] gather token rows from VRAM: pairs={pair_count}, "
                    f"hidden={hidden}, slot_rows={self.blen}",
                )
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

    def moe_gather_token_rows_compact_from_vram_v0(
        self,
        x: VRAMMatrixVar,
        *,
        token_indices: Sequence[int],
        hidden: int,
        zero_row: FPVar | None = None,
        policy_name: str = "gpt_oss",
        name: str = "moe_grouped_x_vram",
    ) -> VRAMMatrixVar:
        """Gather one expert's token rows densely for weight-reuse execution.

        Unlike :meth:`moe_gather_token_rows_from_vram_v0`, which reserves one
        BLEN-row slot per routed pair, this mapping places token ``j`` at row
        ``j``.  Padding is restricted to the final BLEN tile, so one resident
        expert weight tile serves all valid rows in the group.
        """
        self._ensure_vram_sub_matrix_registered(x)
        if hidden % self.mlen != 0:
            raise ValueError(f"compact gather hidden={hidden} must be divisible by MLEN={self.mlen}")
        if hidden > x.shape[1]:
            raise ValueError(f"compact gather hidden={hidden} exceeds x width={x.shape[1]}")

        token_list = [int(token) for token in token_indices]
        if not token_list:
            raise ValueError("compact gather token_indices must be non-empty")
        if len(set(token_list)) != len(token_list):
            raise ValueError("compact gather requires unique token rows for one expert")
        if min(token_list) < 0 or max(token_list) >= x.physical_shape[0]:
            raise ValueError(f"compact gather token_indices {token_list} exceed x physical rows={x.physical_shape[0]}")

        rows = len(token_list)
        physical_rows = max(self.blen, math.ceil(rows / self.blen) * self.blen)
        gathered = self.alloc(
            name,
            rows=rows,
            cols=hidden,
            strict=False,
            physical_shape=(physical_rows, hidden),
        )
        self.moe_true_zero_vram_rows_v0(
            gathered,
            rows=list(range(physical_rows)),
            hidden=hidden,
            zero_row=zero_row,
            policy_name=policy_name,
            stage="gather",
            name=f"{name}_zero",
        )

        num_col_blocks = hidden // self.mlen
        gp_dst, gp_src = self._reg.allocate_gp(2)
        try:
            asm = IsaBuilder().comment(
                moe_stage_marker(
                    "gather",
                    f"[{policy_name}] compact expert-major VRAM gather: rows={rows}, hidden={hidden}",
                )
            )
            for active_row, token_idx in enumerate(token_list):
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

    def moe_true_zero_vram_rows_v0(
        self,
        matrix: VRAMMatrixVar,
        *,
        rows: Sequence[int],
        hidden: int,
        zero_row: FPVar | None = None,
        policy_name: str = "gpt_oss",
        stage: str,
        name: str = "moe_zero_rows",
    ) -> None:
        """Clear selected VRAM rows by mapping a true FP zero row.

        Do not use vector multiply-by-zero here: padding/gather slots can
        contain NaNs, and ``NaN * 0`` remains NaN.  This helper writes real
        zeros through ``S_MAP_V_FP`` and is therefore safe for gather padding
        and scatter accumulators.

        ``stage`` is required, not defaulted: the same clear serves three different
        phases -- zeroing the combine accumulator, clearing gather padding rows, and
        zeroing a route-weight tile -- and only the caller knows which. A default
        would make a wrong stage silent again.
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
            asm = IsaBuilder().comment(
                moe_stage_marker(stage, f"[{policy_name}] true-zero VRAM rows {row_list} in {matrix.name}")
            )
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

    def moe_scatter_add_active_rows_v0(
        self,
        dst: VRAMMatrixVar,
        src: VRAMMatrixVar,
        *,
        token_indices: Sequence[int],
        active_rows: Sequence[int],
        hidden: int,
        policy_name: str = "gpt_oss",
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
            asm = IsaBuilder().comment(
                moe_stage_marker(
                    "scatter_combine",
                    f"[{policy_name}] VRAM scatter-add {name}: {len(token_list)} active rows",
                )
            )
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

    def moe_expert_v0(
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

    def moe_fixed_routing_v0(
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
            expert_out = self.moe_expert_v0(
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
        policy_name: str = "gpt_oss",
        stage: str,
        name: str,
    ) -> VRAMMatrixVar:
        """Generic substrate wrapper for expert activation backends.

        ``stage`` is required, not defaulted: the shared and routed branches run the
        identical backend, so the live marker is the only thing separating them in
        the profile and a default would silently merge them.
        """
        # Both backends are built entirely from general-purpose tile helpers
        # (`tile_row_exp`, `vram_mul`, ...) that emit their own unmarked comments.
        # One marker here covers the whole region.
        self._emit(
            IsaBuilder().comment(moe_stage_marker(stage, f"[{policy_name}] {activation_policy} {name}: rows={rows}"))
        )
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


# ============================================================================
# Deprecated aliases
# ============================================================================

#: Pre-generalization method names, kept so in-flight callers (and the GPT-OSS
#: bring-up tests that predate the rename) keep working. The ``moe_*`` defaults
#: preserve GPT-OSS behaviour.
_DEPRECATED_METHOD_ALIASES = {
    "gpt_oss_router_logits_bf16_v0": "moe_router_logits_bf16_v0",
    "gpt_oss_router_topk_softmax_v0": "moe_router_select_v0",
    "gpt_oss_expert_v0": "moe_expert_v0",
    "gpt_oss_dynamic_expert_pair_v0": "moe_dynamic_expert_pair_v0",
    "gpt_oss_dynamic_linear_projection_v0": "moe_dynamic_linear_projection_v0",
    "gpt_oss_dynamic_vram_sub_projection_to_v0": "moe_dynamic_vram_sub_projection_to_v0",
    "gpt_oss_expert_id_to_weight_base_v0": "moe_expert_id_to_weight_base_v0",
    "gpt_oss_add_dynamic_expert_bias_v0": "moe_add_dynamic_expert_bias_v0",
    "gpt_oss_materialize_topk_route_weight_v0": "moe_materialize_topk_route_weight_v0",
    "gpt_oss_materialize_route_weights_for_active_rows_v0": ("moe_materialize_route_weights_for_active_rows_v0"),
    "gpt_oss_gather_token_rows_from_hbm_v0": "moe_gather_token_rows_from_hbm_v0",
    "gpt_oss_gather_token_rows_from_vram_v0": "moe_gather_token_rows_from_vram_v0",
    "gpt_oss_scatter_add_active_rows_v0": "moe_scatter_add_active_rows_v0",
    "gpt_oss_true_zero_vram_rows_v0": "moe_true_zero_vram_rows_v0",
    "gpt_oss_moe_fixed_routing_v0": "moe_fixed_routing_v0",
}

for _old, _new in _DEPRECATED_METHOD_ALIASES.items():
    setattr(ProgramRoutedMoeMixin, _old, getattr(ProgramRoutedMoeMixin, _new))
del _old, _new


__all__ = [
    "MOE_END_STAGE",
    "MOE_STAGES",
    "MOE_STAGE_MARKER_PREFIX",
    "ProgramRoutedMoeMixin",
    "moe_end_marker",
    "moe_stage_marker",
]
