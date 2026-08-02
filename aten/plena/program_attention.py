"""Flash-attention operations for the PLENA program builder."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

from compiler.asm_templates._imm import add_large_int
from compiler.asm_templates._imm import load_large_int
from compiler.aten.isa_builder import RepeatAxis
from compiler.aten.plena.kv_residency import KVResidencyPlan, plan_kv_residency
from compiler.aten.plena.vars import InputVar, VRAMMatrixVar


@dataclass(frozen=True)
class PackedGQASchedule:
    """Compile-time schedule for one logical packed-GQA attention block."""

    batch_size: int
    seq_len: int
    kv_seq_len: int
    rows_per_batch: int
    num_kv_heads: int
    gqa_ratio: int
    physical_broadcast: int
    chunks_per_kv: int
    full_chunks: int
    tail_heads: int
    q_blocks: int
    k_blocks: int
    resident_kv: bool
    resident_kv_tiles: int
    kv_residency_plan: KVResidencyPlan

    @classmethod
    def build(
        cls,
        *,
        batch_size: int,
        seq_len: int,
        kv_seq_len: int,
        rows_per_batch: int,
        num_kv_heads: int,
        gqa_ratio: int,
        physical_broadcast: int,
        mlen: int,
        mram_tile_capacity: int,
        kv_residency_policy: str = "raw-tiles",
    ) -> "PackedGQASchedule":
        values = {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "kv_seq_len": kv_seq_len,
            "rows_per_batch": rows_per_batch,
            "num_kv_heads": num_kv_heads,
            "gqa_ratio": gqa_ratio,
            "physical_broadcast": physical_broadcast,
            "mlen": mlen,
            "mram_tile_capacity": mram_tile_capacity,
        }
        for name, value in values.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if rows_per_batch < max(seq_len, kv_seq_len):
            raise ValueError(
                f"rows_per_batch={rows_per_batch} cannot cover "
                f"seq_len={seq_len}, kv_seq_len={kv_seq_len}"
            )

        chunks_per_kv = math.ceil(gqa_ratio / physical_broadcast)
        full_chunks, tail_heads = divmod(gqa_ratio, physical_broadcast)
        q_blocks = math.ceil(seq_len / mlen)
        k_blocks = math.ceil(kv_seq_len / mlen)
        resident_kv_tiles = 2 * k_blocks
        residency = plan_kv_residency(
            k_blocks=k_blocks,
            mlen=mlen,
            matrix_sram_tiles=mram_tile_capacity,
            requested_residency_fraction=(
                0.0 if kv_residency_policy == "streaming" else None
            ),
            policy=kv_residency_policy,
            force_streaming=kv_residency_policy == "streaming",
        )
        return cls(
            batch_size=batch_size,
            seq_len=seq_len,
            kv_seq_len=kv_seq_len,
            rows_per_batch=rows_per_batch,
            num_kv_heads=num_kv_heads,
            gqa_ratio=gqa_ratio,
            physical_broadcast=physical_broadcast,
            chunks_per_kv=chunks_per_kv,
            full_chunks=full_chunks,
            tail_heads=tail_heads,
            q_blocks=q_blocks,
            k_blocks=k_blocks,
            resident_kv=residency.full_resident,
            resident_kv_tiles=resident_kv_tiles,
            kv_residency_plan=residency,
        )


class ProgramAttentionMixin:
    # ========================================================================
    # Flash Attention Operations
    # ========================================================================

    def _needs_explicit_valid_col_mask(self, valid_cols: int | None) -> bool:
        """Return true when softmax must ignore padded K columns.

        The vector mask path is not sufficient for reductions on current
        behavioral hardware: padded score lanes can still affect the softmax
        denominator.  Materialize a VRAM score mask for every partial K block
        so padded columns are explicitly set to -inf before online softmax.
        """
        if valid_cols is None or valid_cols >= self.mlen:
            return False
        return True

    def _build_valid_col_mask(self, name: str, valid_cols: int) -> VRAMMatrixVar:
        """Materialize an MLEN x MLEN score mask with -inf in padded columns."""
        if valid_cols < 0 or valid_cols > self.mlen:
            raise ValueError(f"valid_cols must be in [0, {self.mlen}], got {valid_cols}")

        mask = self.alloc(name, self.mlen, self.mlen)
        mask_addr = self.get_vram_addr(mask.name)
        fp_scratch_base = self._ONLINE_SOFTMAX_FPSRAM_BASE
        gp_mask, gp_fp, gp_loop = self.register_allocator.allocate_gp(3)

        lines = [
            f"; === Build valid-column score mask: valid_cols={valid_cols}, MLEN={self.mlen} ===",
            f"S_ADDI_INT gp{gp_fp}, gp0, {fp_scratch_base}",
            "S_LD_FP f7, gp0, 2",
        ]
        for col in range(valid_cols):
            lines.append(f"S_ST_FP f0, gp{gp_fp}, {col}")
        for col in range(valid_cols, self.mlen):
            lines.append(f"S_ST_FP f7, gp{gp_fp}, {col}")
        lines.extend(load_large_int(gp_mask, mask_addr))
        lines.extend(
            [
                f"C_LOOP_START gp{gp_loop}, {self.mlen}",
                f"S_MAP_V_FP gp{gp_mask}, gp{gp_fp}, 0",
                f"S_ADDI_INT gp{gp_mask}, gp{gp_mask}, {self.mlen}",
                f"C_LOOP_END gp{gp_loop}",
            ]
        )

        self.register_allocator.free_gp([gp_mask, gp_fp, gp_loop])
        previous_stage = getattr(self, "_active_cost_stage", None)
        if (
            getattr(self, "_cost_sink", None) is not None
            and getattr(self, "vector_scalar_schedule", "legacy")
            in {"compiler-v1", "rtl-v2", "rtl-v3", "rtl-v4", "rtl-v5"}
        ):
            self._active_cost_stage = "global/valid_col_mask"
        try:
            self.emit("\n".join(lines) + "\n")
        finally:
            self._active_cost_stage = previous_stage
        if hasattr(self, "record_vector_scalar_stats"):
            self.record_vector_scalar_stats({"valid_mask_build_count": 1})
        return mask

    def _get_valid_col_mask(self, valid_cols: int) -> VRAMMatrixVar:
        """Return one program-lifetime mask for a partial K tile."""
        if not self._needs_explicit_valid_col_mask(valid_cols):
            raise ValueError(f"valid_cols={valid_cols} does not require an explicit mask")

        cache = getattr(self, "_valid_col_mask_cache", None)
        if cache is None:
            cache = {}
            self._valid_col_mask_cache = cache
        key = (self.mlen, valid_cols)
        cached = cache.get(key)
        if cached is not None and cached.name in getattr(self, "_tensors", {}):
            return cached

        mask = self._build_valid_col_mask(f"_valid_col_mask_m{self.mlen}_c{valid_cols}", valid_cols)
        cache[key] = mask
        return mask

    def _valid_col_mask_for_tile(
        self,
        valid_cols: int,
        *,
        causal_mask: bool | VRAMMatrixVar | None,
        apply_causal_mask: bool,
        causal_mask_covers_padding: bool,
    ) -> VRAMMatrixVar | None:
        """Lazily return a padding mask only when the causal mask is insufficient."""

        if not self._needs_explicit_valid_col_mask(valid_cols):
            return None
        if (
            getattr(self, "vector_scalar_schedule", "legacy")
            in {"compiler-v1", "rtl-v2", "rtl-v3", "rtl-v4", "rtl-v5"}
            and isinstance(causal_mask, VRAMMatrixVar)
            and apply_causal_mask
            and causal_mask_covers_padding
        ):
            if hasattr(self, "record_vector_scalar_stats"):
                self.record_vector_scalar_stats(
                    {"redundant_valid_masks_elided": 1}
                )
            return None
        return self._get_valid_col_mask(valid_cols)

    def rope_packed_q(
        self,
        Q_full: VRAMMatrixVar,
        rope_matrix: InputVar,
        cos_var: VRAMMatrixVar,
        sin_var: VRAMMatrixVar,
        *,
        slab_count: int,
        rows_per_slab: int,
        active_rows: int,
    ) -> None:
        """Apply packed Q RoPE with one resident rotation tile and a slab loop."""
        if slab_count <= 0:
            raise ValueError(f"slab_count must be positive, got {slab_count}")
        if active_rows <= 0 or active_rows > rows_per_slab:
            raise ValueError(
                f"active_rows must be in [1, rows_per_slab], got {active_rows}, {rows_per_slab}"
            )
        if rows_per_slab % self.mlen != 0:
            raise ValueError(
                f"rows_per_slab={rows_per_slab} must be a multiple of MLEN={self.mlen}"
            )
        if self.mlen % self.blen != 0:
            raise ValueError(f"Packed Q RoPE requires MLEN % BLEN == 0, got {self.mlen} % {self.blen}")
        expected_elements = slab_count * rows_per_slab * self.mlen
        q_elements = Q_full.physical_shape[0] * Q_full.physical_shape[1]
        if q_elements < expected_elements:
            raise ValueError(
                f"Q_full storage has {q_elements} elements but packed slabs require {expected_elements}"
            )

        self._ensure_hbm_sub_matrix_registered(rope_matrix)
        rope_layout = self.get_hbm_layout(rope_matrix.name)
        if rope_layout.num_row_blocks != 1 or rope_layout.num_col_blocks != 1:
            raise ValueError(
                "Packed Q RoPE requires one MLEN x MLEN rotation tile, got "
                f"{rope_layout.num_row_blocks}x{rope_layout.num_col_blocks} tiles"
            )
        self.reset_mram()
        self._emit_hbm_matrix_load(
            rope_layout,
            3,
            lambda addr_reg, gp_regs: self.load_sub_matrix_asm(
                name=rope_matrix.name,
                row_idx=0,
                col_idx=0,
                mram_dest_addr=0,
                hbm_addr_reg=addr_reg,
                gp_regs=gp_regs,
            ),
        )
        if getattr(self, "_cost_sink", None) is not None:
            rope_rows, rope_cols = rope_layout.physical_shape or rope_layout.full_shape
            self.record_dma_stream(
                self.make_exact_mx_dma_transfer(
                    opcode="H_PREFETCH_M",
                    precision="weight",
                    hbm_base=rope_layout.hbm_base_addr,
                    total_elements=rope_rows * rope_cols,
                    element_offset=0,
                    dim=self.mlen,
                    amount=self.hbm_m_prefetch_amount,
                    stride=rope_cols,
                    rstride=1,
                    source=f"packed_q_rope:{rope_matrix.name}",
                )
            )

        x_rot = self.alloc(
            "_packed_q_rot_slab",
            active_rows,
            self.mlen,
            strict=False,
            physical_shape=(rows_per_slab, self.mlen),
        )
        vec_scratch = self.alloc(
            "_packed_q_rope_vec_scratch",
            1,
            self.mlen,
            strict=False,
            physical_shape=(1, self.mlen),
        )
        q_base = self.get_vram_addr(Q_full.name)
        x_rot_base = self.get_vram_addr(x_rot.name)
        cos_base = self.get_vram_addr(cos_var.name)
        sin_base = self.get_vram_addr(sin_var.name)
        vec_scratch_base = self.get_vram_addr(vec_scratch.name)
        slab_stride = rows_per_slab * self.mlen
        tile_elems = self.mlen * self.mlen
        col_groups = self.mlen // self.blen
        q_blocks = math.ceil(active_rows / self.mlen)

        (
            gp_q_slab,
            gp_act_row,
            gp_act,
            gp_mat,
            gp_result,
            gp_result_col,
            gp_x,
            gp_x_rot,
            gp_cos,
            gp_sin,
            gp_vec_scratch,
            gp_slab_loop,
            gp_col_loop,
            gp_row_loop,
            gp_rope_loop,
        ) = self.register_allocator.allocate_gp(15)
        try:
            lines = [
                f"; === Packed Q RoPE: slabs={slab_count}, active_rows={active_rows} ===",
                *load_large_int(gp_q_slab, q_base),
            ]
            if slab_count > 1:
                lines.append(f"C_LOOP_START gp{gp_slab_loop}, {slab_count}")

            for q_block in range(q_blocks):
                rows = min(self.mlen, active_rows - q_block * self.mlen)
                row_groups = math.ceil(rows / self.blen)
                lines.extend(
                    [
                        f"; Packed Q rotate-half projection q_block={q_block}, rows={rows}",
                        f"S_ADDI_INT gp{gp_act_row}, gp{gp_q_slab}, 0",
                        *add_large_int(
                            gp_act_row,
                            gp_act_row,
                            q_block * tile_elems,
                            temp_reg=gp_rope_loop,
                        ),
                        f"S_ADDI_INT gp{gp_mat}, gp0, 0",
                        *load_large_int(gp_result_col, x_rot_base + q_block * tile_elems),
                        f"C_LOOP_START gp{gp_col_loop}, {col_groups}",
                        f"S_ADDI_INT gp{gp_act}, gp{gp_act_row}, 0",
                        f"S_ADDI_INT gp{gp_result}, gp{gp_result_col}, 0",
                        f"C_LOOP_START gp{gp_row_loop}, {row_groups}",
                        f"M_MM 0, gp{gp_mat}, gp{gp_act}",
                        f"M_MM_WO gp{gp_result}, gp0, 0",
                        f"S_ADDI_INT gp{gp_act}, gp{gp_act}, {self.blen * self.mlen}",
                        f"S_ADDI_INT gp{gp_result}, gp{gp_result}, {self.blen * self.mlen}",
                        f"C_LOOP_END gp{gp_row_loop}",
                        f"S_ADDI_INT gp{gp_mat}, gp{gp_mat}, {self.blen}",
                        f"S_ADDI_INT gp{gp_result_col}, gp{gp_result_col}, {self.blen}",
                        f"C_LOOP_END gp{gp_col_loop}",
                    ]
                )

            lines.extend(
                [
                    "; Packed Q RoPE vector phase",
                    f"S_ADDI_INT gp{gp_x}, gp{gp_q_slab}, 0",
                    *load_large_int(gp_x_rot, x_rot_base),
                    *load_large_int(gp_cos, cos_base),
                    *load_large_int(gp_sin, sin_base),
                    *load_large_int(gp_vec_scratch, vec_scratch_base),
                    f"C_LOOP_START gp{gp_rope_loop}, {active_rows}",
                    f"V_MUL_VV gp{gp_vec_scratch}, gp{gp_x_rot}, gp{gp_sin}, 0",
                    f"V_MUL_VV gp{gp_x}, gp{gp_x}, gp{gp_cos}, 0",
                    f"V_ADD_VV gp{gp_x}, gp{gp_x}, gp{gp_vec_scratch}, 0",
                    f"S_ADDI_INT gp{gp_x}, gp{gp_x}, {self.mlen}",
                    f"S_ADDI_INT gp{gp_x_rot}, gp{gp_x_rot}, {self.mlen}",
                    f"S_ADDI_INT gp{gp_cos}, gp{gp_cos}, {self.mlen}",
                    f"S_ADDI_INT gp{gp_sin}, gp{gp_sin}, {self.mlen}",
                    f"C_LOOP_END gp{gp_rope_loop}",
                ]
            )
            if slab_count > 1:
                lines.extend(
                    [
                        *add_large_int(
                            gp_q_slab,
                            gp_q_slab,
                            slab_stride,
                            temp_reg=gp_rope_loop,
                        ),
                        f"C_LOOP_END gp{gp_slab_loop}",
                    ]
                )
            self.emit("\n".join(lines) + "\n")
        finally:
            self.register_allocator.free_gp(
                [
                    gp_q_slab,
                    gp_act_row,
                    gp_act,
                    gp_mat,
                    gp_result,
                    gp_result_col,
                    gp_x,
                    gp_x_rot,
                    gp_cos,
                    gp_sin,
                    gp_vec_scratch,
                    gp_slab_loop,
                    gp_col_loop,
                    gp_row_loop,
                    gp_rope_loop,
                ]
            )
            self.free_tensor(x_rot)
            self.free_tensor(vec_scratch)

    def flash_attention(
        self,
        Q,
        K,
        V,
        scale=None,
        hq=1,
        hkv=1,
        h_qkv=None,
        causal_mask=None,
        batch_size: int = 1,
        seq_len: int | None = None,
        kv_seq_len: int | None = None,
    ):
        """Emit flash attention, dispatching to MHA or fused GQA codegen by shape."""
        if hq == 1 and hkv == 1:
            return self._flash_attention_mha(
                Q,
                K,
                V,
                scale,
                causal_mask=causal_mask,
                batch_size=batch_size,
                seq_len=seq_len,
                kv_seq_len=kv_seq_len,
            )

        if h_qkv is None:
            raise ValueError("GQA mode requires h_qkv to be specified")
        if causal_mask is not None:
            raise NotImplementedError("causal_mask is not yet supported for GQA flash attention")
        return self._flash_attention_gqa_fused(
            Q,
            K,
            V,
            scale,
            hq,
            hkv,
            h_qkv,
            batch_size=batch_size,
            seq_len=seq_len,
            kv_seq_len=kv_seq_len,
        )

    def _flash_attention_mha(
        self,
        Q,
        K,
        V,
        scale,
        causal_mask=None,
        *,
        batch_size: int = 1,
        seq_len: int | None = None,
        kv_seq_len: int | None = None,
    ):
        """Single-head online-softmax flash attention using compiler primitives."""
        total_q_rows, head_dim = Q.shape
        mlen = self.mlen

        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if seq_len is None:
            if total_q_rows % batch_size != 0:
                raise ValueError(f"Q rows {total_q_rows} are not divisible by batch_size={batch_size}")
            seq_len = total_q_rows // batch_size
        elif total_q_rows < batch_size * seq_len:
            raise ValueError(f"Q rows {total_q_rows} cannot cover batch_size*seq_len={batch_size * seq_len}")

        total_k_rows, _ = K.shape
        if kv_seq_len is None:
            if total_k_rows % batch_size != 0:
                raise ValueError(f"K rows {total_k_rows} are not divisible by batch_size={batch_size}")
            kv_seq_len = total_k_rows // batch_size
        elif total_k_rows < batch_size * kv_seq_len:
            raise ValueError(f"K rows {total_k_rows} cannot cover batch_size*kv_seq_len={batch_size * kv_seq_len}")
        if V.shape[0] < batch_size * kv_seq_len:
            raise ValueError(f"V rows {V.shape[0]} cannot cover batch_size*kv_seq_len={batch_size * kv_seq_len}")

        if Q.physical_shape[0] % batch_size != 0:
            raise ValueError(f"Q physical rows {Q.physical_shape[0]} are not divisible by batch_size={batch_size}")
        if K.physical_shape[0] % batch_size != 0:
            raise ValueError(f"K physical rows {K.physical_shape[0]} are not divisible by batch_size={batch_size}")
        if V.physical_shape[0] % batch_size != 0:
            raise ValueError(f"V physical rows {V.physical_shape[0]} are not divisible by batch_size={batch_size}")

        q_rows_per_batch = Q.physical_shape[0] // batch_size
        k_rows_per_batch = K.physical_shape[0] // batch_size
        v_rows_per_batch = V.physical_shape[0] // batch_size
        if q_rows_per_batch < max(mlen, seq_len):
            raise ValueError(f"Q physical rows per batch {q_rows_per_batch} are too small for seq_len={seq_len}")
        if k_rows_per_batch < max(mlen, kv_seq_len):
            raise ValueError(f"K physical rows per batch {k_rows_per_batch} are too small for kv_seq_len={kv_seq_len}")
        if v_rows_per_batch < max(mlen, kv_seq_len):
            raise ValueError(f"V physical rows per batch {v_rows_per_batch} are too small for kv_seq_len={kv_seq_len}")
        if batch_size > 1 and q_rows_per_batch % mlen != 0:
            raise ValueError(f"Q physical rows per batch {q_rows_per_batch} must be multiple of MLEN={mlen}")
        if batch_size > 1 and k_rows_per_batch % mlen != 0:
            raise ValueError(f"K physical rows per batch {k_rows_per_batch} must be multiple of MLEN={mlen}")
        if batch_size > 1 and v_rows_per_batch % mlen != 0:
            raise ValueError(f"V physical rows per batch {v_rows_per_batch} must be multiple of MLEN={mlen}")

        if scale is None:
            scale = 1.0 / math.sqrt(head_dim)

        num_q_blocks = math.ceil(seq_len / mlen)
        num_k_blocks = math.ceil(kv_seq_len / mlen)
        k_row_blocks_per_batch = max(1, math.ceil(k_rows_per_batch / mlen))
        valid_col_masks: dict[int, VRAMMatrixVar] = {}
        for k_idx in range(num_k_blocks):
            block_cols = min(mlen, kv_seq_len - k_idx * mlen)
            if self._needs_explicit_valid_col_mask(block_cols):
                valid_col_masks[block_cols] = self._get_valid_col_mask(block_cols)

        S_block = self.alloc("S", mlen, mlen)
        pv_rows = min(mlen, seq_len)
        PV = self.alloc("PV", pv_rows, head_dim, strict=False)
        O = self.alloc(
            "O",
            batch_size * seq_len,
            head_dim,
            strict=False,
            physical_shape=(max(mlen, batch_size * seq_len), max(mlen, Q.physical_shape[1])),
        )

        q_base = self.get_vram_addr(Q.name)
        o_base = self.get_vram_addr(O.name)
        # Tensors are column-block-major: col-block cb of a tensor with physical height R
        # starts at cb*R*mlen, and row r within a col-block is at r*mlen. So advancing one
        # batch (q_rows_per_batch rows) within col-block 0 skips q_rows_per_batch*mlen flat
        # elements — NOT q_rows_per_batch*physical_shape[1], which over-skips by head_dim/mlen
        # col-blocks when head_dim > mlen. The two expressions coincide at head_dim == mlen
        # (physical_shape[1] == mlen), so the head_dim <= mlen path is unchanged.
        q_batch_stride = q_rows_per_batch * mlen
        # O packs batches contiguously by seq_len rows (the decoder reads O_h batch b at
        # b*seq_len*mlen), so its per-batch base offset is seq_len*mlen.
        o_batch_stride = seq_len * mlen

        # Position of the first query relative to the keys: queries are the LAST
        # seq_len of the kv_seq_len key positions, so query row r of block q_idx is
        # global position q_offset + q_idx*mlen + r. For prefill (kv_seq_len ==
        # seq_len) this is 0.
        q_offset = kv_seq_len - seq_len

        for batch_idx in range(batch_size):
            if batch_size == 1 and total_q_rows == seq_len:
                Q_batch = Q
                O_batch = O
            else:
                Q_batch = self.alloc_at(
                    f"_mha_Q_b{batch_idx}",
                    seq_len,
                    head_dim,
                    q_base + batch_idx * q_batch_stride,
                    # Preserve the full height so the col-block stride stays R*mlen; the
                    # per-batch base offset then lands batch b correctly inside every
                    # col-block. Truncating to q_rows_per_batch only works for a single
                    # col-block (head_dim <= mlen) and mis-strides higher col-blocks.
                    physical_shape=Q.physical_shape,
                )
                O_batch = self.alloc_at(
                    f"_mha_O_b{batch_idx}",
                    seq_len,
                    head_dim,
                    o_base + batch_idx * o_batch_stride,
                    # Same reasoning as Q_batch: keep O's full height so writes to higher
                    # col-blocks (head_dim > mlen) land at R*mlen strides.
                    physical_shape=O.physical_shape,
                )

            batch_k_block_base = batch_idx * k_row_blocks_per_batch
            for q_idx in range(num_q_blocks):
                block_rows = min(mlen, seq_len - q_idx * mlen)
                self.init_online_softmax(q_idx, O_batch, rows=block_rows)

                for k_idx in range(num_k_blocks):
                    block_cols = min(mlen, kv_seq_len - k_idx * mlen)
                    needs_triangular_mask = False
                    if causal_mask is not None:
                        # Causal geometry across tiles. Query rows of block q_idx are
                        # global positions q_offset + q_idx*mlen + [0, block_rows);
                        # key cols of block k_idx are k_idx*mlen + [0, block_cols). A
                        # key block entirely in the strict future of every query row
                        # contributes nothing (exp(-inf)=0) and is skipped; one
                        # entirely in the past is fully visible (no mask); only a
                        # straddling block needs the triangular mask. The static
                        # (mlen, mlen) mask encodes a zero-diagonal triangle, which is
                        # exactly right when the straddle sits on the q_idx == k_idx
                        # diagonal (q_offset == 0, i.e. prefill). seq_len <= mlen is
                        # the single-block special case of this and is unchanged.
                        key_first = k_idx * mlen
                        query_first = q_offset + q_idx * mlen
                        query_last = query_first + block_rows - 1
                        if key_first > query_last:
                            continue  # whole key block is in the strict future
                        needs_triangular_mask = key_first + block_cols - 1 > query_first
                        if needs_triangular_mask and query_first != key_first:
                            raise NotImplementedError(
                                "Causal mask across tiles with kv_seq_len != seq_len "
                                "(non-zero query offset) is unsupported: the static "
                                "(mlen, mlen) mask only encodes the zero diagonal "
                                f"(q_offset={q_offset}, q_idx={q_idx}, k_idx={k_idx})."
                            )
                    physical_k_idx = batch_k_block_base + k_idx
                    self.vram_sub_projection_T_to(
                        Q_batch,
                        q_idx,
                        K,
                        physical_k_idx,
                        S_block,
                        target_row_idx=0,
                        target_col_idx=0,
                    )
                    valid_col_mask = valid_col_masks.get(block_cols)
                    if valid_col_mask is not None:
                        self.vram_add(S_block, valid_col_mask, num_rows=block_rows)
                    if needs_triangular_mask:
                        self.vram_add(S_block, causal_mask)
                    softmax_valid_cols = None if valid_col_mask is not None else block_cols
                    self.online_softmax_block(S_block, scale, rows=block_rows, valid_cols=softmax_valid_cols)
                    self.compute_pv(S_block, V, physical_k_idx, PV, head_dim, rows=block_rows)
                    self.scale_o_row(O_batch, q_idx, rows=block_rows)
                    self.vram_add(O_batch, PV, dst_row_offset=q_idx * mlen, num_rows=block_rows)

                self.final_scale_o(q_idx, O_batch, rows=block_rows)

        return O

    def _flash_attention_gqa_fused(
        self,
        Q,
        K,
        V,
        scale,
        hq,
        hkv,
        h_qkv,
        *,
        batch_size: int = 1,
        seq_len: int | None = None,
        kv_seq_len: int | None = None,
    ):
        """GQA flash attention using compiler-owned packed-head primitives."""
        if hq % hkv != 0:
            raise ValueError(f"hq={hq} must be divisible by hkv={hkv}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        ratio = hq // hkv
        mlen = self.mlen
        broadcast_amount = mlen // h_qkv

        if broadcast_amount * h_qkv != mlen:
            raise ValueError(
                f"GQA constraint: BROADCAST_AMOUNT*h_qkv must equal mlen "
                f"({broadcast_amount}*{h_qkv} != {mlen})."
            )
        if ratio > broadcast_amount:
            raise ValueError(
                f"GQA ratio hq/hkv={ratio} exceeds packed broadcast lanes "
                f"{broadcast_amount}."
            )
        if hkv != 1:
            raise NotImplementedError(
                "ATen fused GQA currently supports one packed KV group in this "
                "entry point. Multi-KV packed decoder lowering should use "
                "flash_attention_packed_group per KV group."
            )

        total_q_rows, q_total_dim = Q.shape
        total_k_rows, _k_total_dim = K.shape

        if seq_len is None:
            if total_q_rows % batch_size != 0:
                raise ValueError(f"Q rows {total_q_rows} are not divisible by batch_size={batch_size}")
            s_q = total_q_rows // batch_size
        else:
            s_q = seq_len
            if total_q_rows < batch_size * s_q:
                raise ValueError(
                    f"Q rows {total_q_rows} cannot cover batch_size*seq_len={batch_size * s_q}"
                )

        if kv_seq_len is None:
            if total_k_rows % batch_size != 0:
                raise ValueError(f"K rows {total_k_rows} are not divisible by batch_size={batch_size}")
            s_kv = total_k_rows // batch_size
        else:
            s_kv = kv_seq_len
            if total_k_rows < batch_size * s_kv:
                raise ValueError(
                    f"K rows {total_k_rows} cannot cover batch_size*kv_seq_len={batch_size * s_kv}"
                )

        if scale is None:
            scale = 1.0 / math.sqrt(h_qkv)

        if s_q > mlen or s_kv > mlen:
            raise NotImplementedError("Packed GQA lowering currently supports one sequence tile.")
        if V.shape[0] < batch_size * s_kv:
            raise ValueError(f"V rows {V.shape[0]} cannot cover batch_size*kv_seq_len={batch_size * s_kv}")

        q_physical_rows_total, q_physical_cols = Q.physical_shape
        k_physical_rows_total, _k_physical_cols = K.physical_shape
        v_physical_rows_total, _v_physical_cols = V.physical_shape
        if q_physical_rows_total % batch_size != 0:
            raise ValueError(f"Q physical rows {q_physical_rows_total} are not divisible by batch_size={batch_size}")
        if k_physical_rows_total % batch_size != 0:
            raise ValueError(f"K physical rows {k_physical_rows_total} are not divisible by batch_size={batch_size}")
        if v_physical_rows_total % batch_size != 0:
            raise ValueError(f"V physical rows {v_physical_rows_total} are not divisible by batch_size={batch_size}")

        q_rows_per_batch = q_physical_rows_total // batch_size
        k_rows_per_batch = k_physical_rows_total // batch_size
        v_rows_per_batch = v_physical_rows_total // batch_size
        if q_rows_per_batch < max(mlen, s_q):
            raise ValueError(f"Q physical rows per batch {q_rows_per_batch} are too small for seq_len={s_q}")
        if k_rows_per_batch < max(mlen, s_kv):
            raise ValueError(f"K physical rows per batch {k_rows_per_batch} are too small for kv_seq_len={s_kv}")
        if v_rows_per_batch < max(mlen, s_kv):
            raise ValueError(f"V physical rows per batch {v_rows_per_batch} are too small for kv_seq_len={s_kv}")
        if k_rows_per_batch % mlen != 0:
            raise ValueError(f"K physical rows per batch {k_rows_per_batch} must be multiple of MLEN={mlen}")
        if v_rows_per_batch % mlen != 0:
            raise ValueError(f"V physical rows per batch {v_rows_per_batch} must be multiple of MLEN={mlen}")

        o_name = self._scoped_name("O")
        logical_rows = batch_size * s_q
        physical_rows = max(mlen, logical_rows)
        physical_cols = math.ceil((hq * h_qkv) / mlen) * mlen
        self.allocate_vram_matrix(
            name=o_name,
            rows=logical_rows,
            cols=hq * h_qkv,
            strict=False,
            physical_shape=(physical_rows, physical_cols),
        )
        o_addr = self.get_vram_addr(o_name)
        scratch = self.alloc(
            "_gqa_internal_scratch",
            mlen * broadcast_amount,
            mlen,
            strict=True,
        )

        q_base = self.get_vram_addr(Q.name)
        scratch_addr = self.get_vram_addr(scratch.name)
        q_batch_stride = q_rows_per_batch * q_physical_cols
        o_batch_stride = s_q * mlen
        k_row_blocks_per_batch = k_rows_per_batch // mlen

        for batch_idx in range(batch_size):
            if batch_size == 1 and total_q_rows == s_q:
                Q_batch = Q
            else:
                Q_batch = self.alloc_at(
                    f"_gqa_Q_b{batch_idx}",
                    s_q,
                    q_total_dim,
                    q_base + batch_idx * q_batch_stride,
                    physical_shape=(q_rows_per_batch, q_physical_cols),
                )
            self._emit_packed_attention_group_internal(
                Q_group=Q_batch,
                K=K,
                V=V,
                group_heads=ratio,
                head_slot_dim=h_qkv,
                output_base_address=o_addr + batch_idx * o_batch_stride,
                output_physical_rows=physical_rows,
                scratch_base_address=scratch_addr,
                broadcast_amount=broadcast_amount,
                scale=scale,
                causal_mask=None,
                output_head_base=0,
                k_idx=batch_idx * k_row_blocks_per_batch,
                valid_cols=s_kv,
            )

        self.free_tensor(scratch)
        O = VRAMMatrixVar(
            self,
            o_name,
            (logical_rows, hq * h_qkv),
            display_name="O",
            physical_shape=(physical_rows, physical_cols),
        )
        self._tensors[o_name] = O
        return O

    def _emit_packed_qkt_to_s(
        self,
        *,
        Q_group: VRAMMatrixVar,
        K: InputVar,
        q_idx: int,
        k_idx: int,
        s_base_address: int,
        k_head_offset: int = 0,
    ) -> None:
        """Emit packed QK^T with M_BTMM/M_BMM_WO into head-major S tiles."""
        self._emit_packed_kv_prefetch(K, ((k_idx, 0),))

        q_base = self.get_vram_addr(Q_group.name) + q_idx * self.mlen * self.mlen
        gp_q, gp_s = self.register_allocator.allocate_gp(2)
        lines = [
            "; === Packed GQA QK^T using compiler M_BTMM ===",
            *load_large_int(gp_q, q_base),
            f"M_BTMM {k_head_offset}, gp0, gp{gp_q}",
            *load_large_int(gp_s, s_base_address),
            f"M_BMM_WO gp{gp_s}, 0",
        ]
        self.register_allocator.free_gp([gp_q, gp_s])
        self.emit("\n".join(lines) + "\n")

    def _emit_packed_kv_prefetch(
        self,
        input_var: InputVar,
        tiles: tuple[tuple[int, int], ...],
        *,
        record_dma: bool = True,
    ) -> None:
        """Prefetch packed K/V sequence tiles using KeyValue precision."""
        self._ensure_hbm_sub_matrix_registered(input_var)
        layout = self.get_hbm_layout(input_var.name)

        def build_body(addr_reg, gp_regs):
            gp_scale, gp_stride, gp_mram = gp_regs
            rows, cols = layout.physical_shape or layout.full_shape
            lines = [
                *load_large_int(gp_scale, rows * cols),
                f"C_SET_SCALE_REG gp{gp_scale}",
                *load_large_int(gp_stride, cols),
                f"C_SET_STRIDE_REG gp{gp_stride}",
            ]
            for row_idx, mram_dest in tiles:
                hbm_offset = layout.get_sub_block(row_idx, 0).hbm_offset
                lines.append(
                    f"; Load SubMatrix {input_var.name}[{row_idx}][0] -> "
                    f"MRAM[{mram_dest}] (KeyValue precision)"
                )
                lines.extend(load_large_int(gp_mram, mram_dest))
                lines.extend(load_large_int(gp_scale, hbm_offset))
                lines.append(
                    f"H_PREFETCH_M gp{gp_mram}, gp{gp_scale}, a{addr_reg}, 1, 1"
                )
            return "\n".join(lines) + "\n"

        self._emit_hbm_matrix_load(layout, 3, build_body)
        if record_dma and getattr(self, "_cost_sink", None) is not None:
            rows, cols = layout.physical_shape or layout.full_shape
            total_elements = rows * cols
            for row_idx, _ in tiles:
                hbm_offset = layout.get_sub_block(row_idx, 0).hbm_offset
                self.record_dma_stream(
                    self.make_exact_mx_dma_transfer(
                        opcode="H_PREFETCH_M",
                        precision="matrix_kv",
                        hbm_base=layout.hbm_base_addr,
                        total_elements=total_elements,
                        element_offset=hbm_offset,
                        dim=self.mlen,
                        amount=self.hbm_m_prefetch_amount,
                        stride=cols,
                        rstride=1,
                        source=f"packed_kv:{input_var.name}:tile{row_idx}",
                    )
                )

    def _emit_packed_kv_prefetch_affine_summary(
        self,
        input_var: InputVar,
        tiles: tuple[tuple[int, int], ...],
    ) -> bool:
        """Compress one Q-block's contiguous streaming K/V DMA occurrences."""

        sink = getattr(self, "_cost_sink", None)
        if (
            sink is None
            or getattr(sink, "granularity", "detailed")
            != "affine-block-summary-v1"
            or not tiles
        ):
            return False
        row_indices = tuple(row for row, _destination in tiles)
        if row_indices != tuple(range(row_indices[0], row_indices[0] + len(tiles))):
            return False

        self._ensure_hbm_sub_matrix_registered(input_var)
        layout = self.get_hbm_layout(input_var.name)
        rows, cols = layout.physical_shape or layout.full_shape
        total_elements = rows * cols

        def immediate_shape(value: int) -> tuple[str, ...]:
            return tuple(
                line.split(maxsplit=1)[0]
                for line in load_large_int(1, value)
            )

        instruction_census: dict[
            tuple[tuple[str, ...], tuple[str, ...]],
            list[object],
        ] = {}
        for row_idx, destination in tiles:
            hbm_offset = layout.get_sub_block(row_idx, 0).hbm_offset
            shape_key = (
                immediate_shape(destination),
                immediate_shape(hbm_offset),
            )
            entry = instruction_census.get(shape_key)
            if entry is None:
                instruction_census[shape_key] = [
                    row_idx,
                    destination,
                    1,
                ]
            else:
                entry[2] = int(entry[2]) + 1

        for shape_key, (
            representative_row,
            representative_destination,
            repetitions,
        ) in instruction_census.items():
            row_idx = int(representative_row)
            destination = int(representative_destination)
            key = (
                "packed_kv_prefetch_instruction_v2",
                self.mlen,
                self.hbm_m_prefetch_amount,
                rows * cols,
                cols,
                *shape_key,
            )
            repetitions = int(repetitions)
            if not self.replay_cost_summary_template(
                key,
                count=repetitions,
            ):
                with self.cost_summary_template(key):
                    self._emit_packed_kv_prefetch(
                        input_var,
                        ((row_idx, destination),),
                        record_dma=False,
                    )
                if repetitions > 1:
                    assert self.replay_cost_summary_template(
                        key,
                        count=repetitions - 1,
                    )

        first_row = row_indices[0]
        first_offset = layout.get_sub_block(first_row, 0).hbm_offset
        count = len(tiles)
        axes = (
            (
                RepeatAxis(
                    "streaming_kv_block",
                    count,
                    element_base_delta=self.mlen * cols,
                    scale_base_delta=(self.mlen * cols) // 8,
                    logical_element_delta=self.mlen * cols,
                    logical_scale_delta=(self.mlen * cols) // 8,
                ),
            )
            if count > 1
            else ()
        )
        self.record_dma_stream(
            self.make_exact_mx_dma_transfer(
                opcode="H_PREFETCH_M",
                precision="matrix_kv",
                hbm_base=layout.hbm_base_addr,
                total_elements=total_elements,
                element_offset=first_offset,
                dim=self.mlen,
                amount=self.hbm_m_prefetch_amount,
                stride=cols,
                rstride=1,
                source=(
                    f"packed_kv_affine:{input_var.name}:"
                    f"tile{first_row}:count{count}"
                ),
            ),
            multiplicity=count,
            axes=axes,
        )
        return True

    def _emit_packed_qkt_from_mram(
        self,
        *,
        Q_group: VRAMMatrixVar,
        q_idx: int,
        k_mram_address: int,
        s_base_address: int,
        summary_repetitions: int = 1,
    ) -> None:
        """Emit packed QK^T against a K tile already resident in MRAM."""
        q_base = self.get_vram_addr(Q_group.name) + q_idx * self.mlen * self.mlen
        template_key = self._packed_qkt_from_mram_template_key(
            q_base=q_base,
            k_mram_address=k_mram_address,
            s_base_address=s_base_address,
        )
        if self.replay_cost_summary_template(
            template_key,
            count=summary_repetitions,
        ):
            return
        with self.cost_summary_template(template_key):
            self._emit_packed_qkt_from_mram_body(
                q_base=q_base,
                k_mram_address=k_mram_address,
                s_base_address=s_base_address,
            )
        if summary_repetitions > 1:
            assert self.replay_cost_summary_template(
                template_key,
                count=summary_repetitions - 1,
            )

    def _packed_qkt_from_mram_template_key(
        self,
        *,
        q_base: int,
        k_mram_address: int,
        s_base_address: int,
    ) -> tuple[object, ...]:
        """Return the exact cost shape of one memory-free packed QK kernel."""

        def immediate_shape(value: int) -> tuple[str, ...]:
            return tuple(
                line.split(maxsplit=1)[0]
                for line in load_large_int(1, value)
            )

        return (
            "packed_qkt_from_mram_v2",
            self.mlen,
            self.blen,
            immediate_shape(q_base),
            immediate_shape(k_mram_address),
            immediate_shape(s_base_address),
        )

    def _emit_packed_qkt_from_mram_body(
        self,
        *,
        q_base: int,
        k_mram_address: int,
        s_base_address: int,
    ) -> None:
        gp_q, gp_k, gp_s = self.register_allocator.allocate_gp(3)
        lines = [
            "; === Packed GQA QK^T using resident K tile ===",
            *load_large_int(gp_q, q_base),
            *load_large_int(gp_k, k_mram_address),
            f"M_BTMM 0, gp{gp_k}, gp{gp_q}",
            *load_large_int(gp_s, s_base_address),
            f"M_BMM_WO gp{gp_s}, 0",
        ]
        self.register_allocator.free_gp([gp_q, gp_k, gp_s])
        self.emit("\n".join(lines) + "\n")

    def _compute_pv_from_mram(
        self,
        *,
        s_block: VRAMMatrixVar,
        pv_matrix: VRAMMatrixVar,
        v_mram_address: int,
        head_slot_dim: int,
        rows: int,
        preserve_full_tile_work: bool = False,
    ) -> None:
        """Emit PV using a V tile that is already resident in MRAM."""
        gp_p, gp_v, gp_pv, gp_pv_col, gp_v_loop, gp_p_loop = (
            self.register_allocator.allocate_gp(6)
        )
        p_address = self.get_vram_addr(s_block.name)
        pv_address = self.get_vram_addr(pv_matrix.name)
        v_col_groups = (
            self.mlen // self.blen
            if preserve_full_tile_work
            else max(1, math.ceil(head_slot_dim / self.blen))
        )
        p_row_groups = max(1, math.ceil(rows / self.blen))
        lines = [
            "; === PV Multiply using resident V tile ===",
            *load_large_int(gp_v, v_mram_address),
            *load_large_int(gp_pv_col, pv_address),
            f"C_LOOP_START gp{gp_v_loop}, {v_col_groups}",
            *load_large_int(gp_p, p_address),
            f"S_ADDI_INT gp{gp_pv}, gp{gp_pv_col}, 0",
            f"C_LOOP_START gp{gp_p_loop}, {p_row_groups}",
            f"M_MM 0, gp{gp_v}, gp{gp_p}",
            f"M_MM_WO gp{gp_pv}, gp0, 0",
            f"S_ADDI_INT gp{gp_p}, gp{gp_p}, {self.blen * self.mlen}",
            f"S_ADDI_INT gp{gp_pv}, gp{gp_pv}, {self.blen * self.mlen}",
            f"C_LOOP_END gp{gp_p_loop}",
            f"S_ADDI_INT gp{gp_v}, gp{gp_v}, {self.blen}",
            f"S_ADDI_INT gp{gp_pv_col}, gp{gp_pv_col}, {self.blen}",
            f"C_LOOP_END gp{gp_v_loop}",
        ]
        self.register_allocator.free_gp(
            [gp_p, gp_v, gp_pv, gp_pv_col, gp_v_loop, gp_p_loop]
        )
        self.emit("\n".join(lines) + "\n")

    def _prefetch_packed_kv_prefix(
        self,
        K: InputVar,
        V: InputVar,
        *,
        first_k_idx: int,
        residency: KVResidencyPlan,
    ) -> None:
        """Load the deterministic resident K/V prefix once."""
        self._ensure_hbm_sub_matrix_registered(K)
        self._ensure_hbm_sub_matrix_registered(V)
        count = residency.resident_prefix_blocks
        if count == 0:
            return

        self.emit(
            f"; === Packed GQA resident K/V prefix: first_k_idx={first_k_idx}, "
            f"prefix_blocks={count}, tiles={2 * count} ===\n"
        )
        self._emit_packed_kv_prefetch(
            K,
            tuple(
                (first_k_idx + block, residency.k_address(block))
                for block in range(count)
            ),
        )
        self._emit_packed_kv_prefetch(
            V,
            tuple(
                (first_k_idx + block, residency.v_address(block))
                for block in range(count)
            ),
        )

    def _emit_packed_qkt_to_s_dynamic(
        self,
        *,
        q_base_gp: int,
        k_hbm_addr_reg: int,
        s_base_address: int,
        k_layout,
        k_head_offset: int = 0,
        k_hbm_offset: int = 0,
    ) -> None:
        """Emit packed QK^T using loop-carried Q and K base registers."""
        gp_mram, gp_hbm, gp_s = self.register_allocator.allocate_gp(3)
        rows, cols = k_layout.physical_shape or k_layout.full_shape
        lines = [
            "; === Packed GQA QK^T using compiler M_BTMM (KV-looped) ===",
            *load_large_int(gp_hbm, rows * cols),
            f"C_SET_SCALE_REG gp{gp_hbm}",
            *load_large_int(gp_hbm, cols),
            f"C_SET_STRIDE_REG gp{gp_hbm}",
            f"S_ADDI_INT gp{gp_mram}, gp0, 0",
            *load_large_int(gp_hbm, k_hbm_offset),
            f"H_PREFETCH_M gp{gp_mram}, gp{gp_hbm}, a{k_hbm_addr_reg}, 1, 1",
            f"M_BTMM {k_head_offset}, gp0, gp{q_base_gp}",
            *load_large_int(gp_s, s_base_address),
            f"M_BMM_WO gp{gp_s}, 0",
        ]
        self.register_allocator.free_gp([gp_mram, gp_hbm, gp_s])
        self.emit("\n".join(lines) + "\n")

    def _reset_vram_from_gp(
        self,
        *,
        base_gp: int,
        rows: int,
    ) -> None:
        """Reset an MLEN-wide VRAM row range whose base is held in a GP register."""
        gp_addr, gp_loop = self.register_allocator.allocate_gp(2)
        lines = [
            "; Reset loop-carried packed attention output group",
            f"S_ADDI_INT gp{gp_addr}, gp{base_gp}, 0",
            f"C_LOOP_START gp{gp_loop}, {rows}",
            f"V_MUL_VF gp{gp_addr}, gp{gp_addr}, f0, 0",
            f"S_ADDI_INT gp{gp_addr}, gp{gp_addr}, {self.mlen}",
            f"C_LOOP_END gp{gp_loop}",
        ]
        self.register_allocator.free_gp([gp_addr, gp_loop])
        self.emit("\n".join(lines) + "\n")

    def _scale_direct_o_from_gp(self, *, output_base_gp: int, rows: int) -> None:
        """Scale a loop-carried single-slot O block by online-softmax m_res."""
        gp_m_res, gp_o, gp_loop = self.register_allocator.allocate_gp(3)
        m_res_address = self._ONLINE_SOFTMAX_FPSRAM_BASE + self.mlen
        lines = [
            "; === Scale direct packed O by m_res ===",
            *load_large_int(gp_m_res, m_res_address),
            f"S_ADDI_INT gp{gp_o}, gp{output_base_gp}, 0",
            f"C_LOOP_START gp{gp_loop}, {rows}",
            f"S_LD_FP f1, gp{gp_m_res}, 0",
            f"V_MUL_VF gp{gp_o}, gp{gp_o}, f1, 0",
            f"S_ADDI_INT gp{gp_m_res}, gp{gp_m_res}, 1",
            f"S_ADDI_INT gp{gp_o}, gp{gp_o}, {self.mlen}",
            f"C_LOOP_END gp{gp_loop}",
        ]
        self.register_allocator.free_gp([gp_m_res, gp_o, gp_loop])
        self.emit("\n".join(lines) + "\n")

    def _add_pv_to_direct_o_from_gp(
        self,
        *,
        output_base_gp: int,
        pv: VRAMMatrixVar,
        rows: int,
    ) -> None:
        """Accumulate a static PV tile into a loop-carried single-slot O block."""
        gp_o, gp_pv, gp_loop = self.register_allocator.allocate_gp(3)
        lines = [
            "; === Add PV to direct packed O ===",
            f"S_ADDI_INT gp{gp_o}, gp{output_base_gp}, 0",
            *load_large_int(gp_pv, self.get_vram_addr(pv.name)),
            f"C_LOOP_START gp{gp_loop}, {rows}",
            f"V_ADD_VV gp{gp_o}, gp{gp_o}, gp{gp_pv}, 0",
            f"S_ADDI_INT gp{gp_o}, gp{gp_o}, {self.mlen}",
            f"S_ADDI_INT gp{gp_pv}, gp{gp_pv}, {self.mlen}",
            f"C_LOOP_END gp{gp_loop}",
        ]
        self.register_allocator.free_gp([gp_o, gp_pv, gp_loop])
        self.emit("\n".join(lines) + "\n")

    def _final_scale_direct_o_from_gp(self, *, output_base_gp: int, rows: int) -> None:
        """Apply the final online-softmax normalization to loop-carried O."""
        gp_l, gp_o, gp_loop = self.register_allocator.allocate_gp(3)
        l_address = self._ONLINE_SOFTMAX_FPSRAM_BASE + 2 * self.mlen
        lines = [
            "; === Final scale direct packed O ===",
            *load_large_int(gp_l, l_address),
            f"S_ADDI_INT gp{gp_o}, gp{output_base_gp}, 0",
            f"C_LOOP_START gp{gp_loop}, {rows}",
            f"S_LD_FP f1, gp{gp_l}, 0",
            f"S_RECI_FP f1, f1, 0",
            f"V_MUL_VF gp{gp_o}, gp{gp_o}, f1, 0",
            f"S_ADDI_INT gp{gp_l}, gp{gp_l}, 1",
            f"S_ADDI_INT gp{gp_o}, gp{gp_o}, {self.mlen}",
            f"C_LOOP_END gp{gp_loop}",
        ]
        self.register_allocator.free_gp([gp_l, gp_o, gp_loop])
        self.emit("\n".join(lines) + "\n")

    def _emit_resident_single_head_chunks_looped(
        self,
        *,
        q_base_address: int | None,
        o_base_address: int | None,
        group_stride: int,
        chunk_count: int,
        seq_len: int,
        kv_seq_len: int,
        head_slot_dim: int,
        scratch_base_address: int,
        scale: float,
        causal_mask: bool | VRAMMatrixVar | None,
        resident_k_mram: tuple[int, ...],
        resident_v_mram: tuple[int, ...],
        q_base_gp: int | None = None,
        o_base_gp: int | None = None,
        advance_base_pointers: bool = False,
    ) -> None:
        """Loop all single-head chunks while reusing resident K/V tiles."""
        if chunk_count <= 0:
            raise ValueError(f"chunk_count must be positive, got {chunk_count}")
        num_q_blocks = math.ceil(seq_len / self.mlen)
        num_k_blocks = math.ceil(kv_seq_len / self.mlen)
        if len(resident_k_mram) != num_k_blocks or len(resident_v_mram) != num_k_blocks:
            raise ValueError("resident K/V address lists do not match sequence K blocks")
        softmax_scale = scale / 0.25
        optimized_schedule = (
            getattr(self, "packed_attention_schedule", "legacy")
            == "direct-first-block-v1"
        )
        s_head = self.alloc_at(
            "_resident_chunk_loop_S",
            self.mlen,
            self.mlen,
            scratch_base_address,
            physical_shape=(self.mlen, self.mlen),
        )
        pv = self.alloc(
            "_resident_chunk_loop_PV",
            self.mlen,
            head_slot_dim,
            strict=False,
            physical_shape=(self.mlen, self.mlen),
        )
        if (q_base_gp is None) != (o_base_gp is None):
            raise ValueError("q_base_gp and o_base_gp must be provided together")
        if q_base_gp is None:
            if q_base_address is None or o_base_address is None:
                raise ValueError("static q/o base addresses are required when base GPs are absent")
            local_regs = self.register_allocator.allocate_gp(6)
            gp_q, gp_o, gp_q_block, gp_o_block, gp_chunk_loop, gp_tmp = local_regs
        else:
            local_regs = self.register_allocator.allocate_gp(4)
            gp_q = q_base_gp
            gp_o = o_base_gp
            gp_q_block, gp_o_block, gp_chunk_loop, gp_tmp = local_regs
        try:
            setup = [
                f"; === Resident packed GQA full-chunk loop: chunks={chunk_count} ===",
            ]
            if q_base_gp is None:
                setup.extend(load_large_int(gp_q, q_base_address))
                setup.extend(load_large_int(gp_o, o_base_address))
            if chunk_count > 1:
                setup.append(f"C_LOOP_START gp{gp_chunk_loop}, {chunk_count}")
            self.emit("\n".join(setup) + "\n")

            for q_block in range(num_q_blocks):
                rows = min(self.mlen, seq_len - q_block * self.mlen)
                q_block_offset = q_block * self.mlen * self.mlen
                block_setup = [
                    f"; Resident packed GQA q_block={q_block}, rows={rows}",
                    f"S_ADDI_INT gp{gp_q_block}, gp{gp_q}, 0",
                    *add_large_int(
                        gp_q_block,
                        gp_q_block,
                        q_block_offset,
                        temp_reg=gp_tmp,
                    ),
                    f"S_ADDI_INT gp{gp_o_block}, gp{gp_o}, 0",
                    *add_large_int(
                        gp_o_block,
                        gp_o_block,
                        q_block_offset,
                        temp_reg=gp_tmp,
                    ),
                ]
                self.emit("\n".join(block_setup) + "\n")
                if optimized_schedule:
                    self._packed_attention_stats[
                        "softmax_state_initializations_elided"
                    ] += chunk_count
                    self._packed_attention_stats[
                        "softmax_state_initialization_rows_elided"
                    ] += rows * chunk_count
                    self._packed_attention_stats[
                        "temporary_o_matrices_elided"
                    ] += chunk_count
                else:
                    self.emit(
                        self._reset_fpsram_asm(
                            self._ONLINE_SOFTMAX_FPSRAM_BASE,
                            self.mlen,
                            2,
                        )
                    )
                    self.emit(
                        self._reset_fpsram_asm(
                            self._ONLINE_SOFTMAX_FPSRAM_BASE + 2 * self.mlen,
                            self.mlen,
                            0,
                        )
                    )

                first_processed_k_block = True
                for k_block in range(num_k_blocks):
                    if causal_mask is not None and k_block > q_block:
                        self.emit(
                            f"; Skip future packed-GQA K block q_block={q_block}, "
                            f"k_block={k_block}\n"
                        )
                        continue
                    self.emit(
                        "\n".join(
                            [
                                "; === Packed GQA QK^T using resident K tile and loop-carried Q ===",
                                *load_large_int(gp_tmp, resident_k_mram[k_block]),
                                f"M_BTMM 0, gp{gp_tmp}, gp{gp_q_block}",
                                *load_large_int(gp_tmp, scratch_base_address),
                                f"M_BMM_WO gp{gp_tmp}, 0",
                            ]
                        )
                        + "\n"
                    )
                    self._packed_attention_stats["qk_compute_count"] += chunk_count
                    self._packed_attention_stats[
                        "ideal_qk_compute_count"
                    ] += chunk_count
                    block_cols = min(self.mlen, kv_seq_len - k_block * self.mlen)
                    apply_causal_mask = causal_mask is not None and k_block == q_block
                    valid_col_mask = self._valid_col_mask_for_tile(
                        block_cols,
                        causal_mask=causal_mask,
                        apply_causal_mask=apply_causal_mask,
                        causal_mask_covers_padding=(
                            seq_len <= self.mlen and kv_seq_len <= self.mlen
                        ),
                    )
                    if valid_col_mask is not None:
                        self.vram_add(s_head, valid_col_mask, num_rows=rows)
                    if isinstance(causal_mask, VRAMMatrixVar) and apply_causal_mask:
                        self.vram_add(s_head, causal_mask, num_rows=rows)
                    elif causal_mask is True and apply_causal_mask:
                        self.emit(
                            "; NOTE: packed attention received causal_mask=True without a VRAM mask; "
                            "no mask applied.\n"
                        )
                    softmax_valid_cols = (
                        None
                        if valid_col_mask is not None
                        or (isinstance(causal_mask, VRAMMatrixVar) and apply_causal_mask)
                        else block_cols
                    )
                    if optimized_schedule and first_processed_k_block:
                        self.online_softmax_first_block(
                            s_head,
                            softmax_scale,
                            rows=rows,
                            valid_cols=softmax_valid_cols,
                        )
                        self._packed_attention_stats[
                            "softmax_first_block_specialized_count"
                        ] += chunk_count
                        self._packed_attention_stats[
                            "softmax_first_block_specialized_rows"
                        ] += rows * chunk_count
                    else:
                        self.online_softmax_block(
                            s_head,
                            softmax_scale,
                            rows=rows,
                            valid_cols=softmax_valid_cols,
                        )
                    self._compute_pv_from_mram(
                        s_block=s_head,
                        pv_matrix=pv,
                        v_mram_address=resident_v_mram[k_block],
                        head_slot_dim=head_slot_dim,
                        rows=rows,
                    )
                    self._packed_attention_stats["pv_compute_count"] += chunk_count
                    if not optimized_schedule or not first_processed_k_block:
                        self._scale_direct_o_from_gp(
                            output_base_gp=gp_o_block, rows=rows
                        )
                    self._add_pv_to_direct_o_from_gp(
                        output_base_gp=gp_o_block,
                        pv=pv,
                        rows=rows,
                    )
                    if optimized_schedule:
                        self._packed_attention_stats[
                            "direct_o_lane_updates"
                        ] += rows * chunk_count
                    first_processed_k_block = False
                self._final_scale_direct_o_from_gp(output_base_gp=gp_o_block, rows=rows)

            if chunk_count > 1 or advance_base_pointers:
                update = [
                    *add_large_int(gp_q, gp_q, group_stride, temp_reg=gp_tmp),
                    *add_large_int(gp_o, gp_o, group_stride, temp_reg=gp_tmp),
                ]
                if chunk_count > 1:
                    update.append(f"C_LOOP_END gp{gp_chunk_loop}")
                self.emit("\n".join(update) + "\n")
        finally:
            self.register_allocator.free_gp(local_regs)
            self.free_tensor(s_head)
            self.free_tensor(pv)

    def _pack_o_head_to_output(
        self,
        *,
        o_head: VRAMMatrixVar,
        output_base_address: int,
        output_physical_rows: int,
        head_slot: int,
        head_slot_dim: int,
        rows: int,
        scratch_address: int,
    ) -> None:
        """Pack the first head_slot_dim columns of o_head into one packed output lane."""
        del output_physical_rows
        shift = head_slot * head_slot_dim
        gp_src, gp_dst, gp_scratch, gp_shift, gp_loop = self.register_allocator.allocate_gp(5)
        src_addr = self.get_vram_addr(o_head.name)
        lines = [
            f"; === Pack O head lane {head_slot} into packed output ===",
            *load_large_int(gp_src, src_addr),
            *load_large_int(gp_dst, output_base_address),
            *load_large_int(gp_scratch, scratch_address),
            *load_large_int(gp_shift, shift),
            f"C_LOOP_START gp{gp_loop}, {rows}",
            f"V_SHIFT_V gp{gp_scratch}, gp{gp_src}, gp{gp_shift}",
            f"V_ADD_VV gp{gp_dst}, gp{gp_dst}, gp{gp_scratch}, 0",
            f"S_ADDI_INT gp{gp_src}, gp{gp_src}, {self.mlen}",
            f"S_ADDI_INT gp{gp_dst}, gp{gp_dst}, {self.mlen}",
            f"C_LOOP_END gp{gp_loop}",
        ]
        self.register_allocator.free_gp([gp_src, gp_dst, gp_scratch, gp_shift, gp_loop])
        self.emit("\n".join(lines) + "\n")

    def _pack_o_head_to_output_dynamic(
        self,
        *,
        o_head: VRAMMatrixVar,
        output_base_gp: int,
        head_slot: int,
        head_slot_dim: int,
        rows: int,
        scratch_address: int,
    ) -> None:
        """Pack one O scratch head into a loop-carried packed-output group."""
        shift = head_slot * head_slot_dim
        gp_src, gp_dst, gp_scratch, gp_shift, gp_loop = self.register_allocator.allocate_gp(5)
        src_addr = self.get_vram_addr(o_head.name)
        lines = [
            f"; === Pack O head lane {head_slot} into KV-looped packed output ===",
            *load_large_int(gp_src, src_addr),
            f"S_ADDI_INT gp{gp_dst}, gp{output_base_gp}, 0",
            *load_large_int(gp_scratch, scratch_address),
            *load_large_int(gp_shift, shift),
            f"C_LOOP_START gp{gp_loop}, {rows}",
            f"V_SHIFT_V gp{gp_scratch}, gp{gp_src}, gp{gp_shift}",
            f"V_ADD_VV gp{gp_dst}, gp{gp_dst}, gp{gp_scratch}, 0",
            f"S_ADDI_INT gp{gp_src}, gp{gp_src}, {self.mlen}",
            f"S_ADDI_INT gp{gp_dst}, gp{gp_dst}, {self.mlen}",
            f"C_LOOP_END gp{gp_loop}",
        ]
        self.register_allocator.free_gp([gp_src, gp_dst, gp_scratch, gp_shift, gp_loop])
        self.emit("\n".join(lines) + "\n")

    def _scale_packed_o_lane(
        self,
        *,
        output_base_address: int,
        head_slot: int,
        rows: int,
        row_ranges: tuple[tuple[int, int], ...] | None = None,
    ) -> None:
        """Scale one packed output lane by the current online-softmax m_res."""

        if getattr(self, "gqa_pipeline_schedule", "row-serial") == "row-interleaved-v1":
            self._emit_pipelined_packed_o_scale(
                value_address=self._ONLINE_SOFTMAX_FPSRAM_BASE + self.mlen,
                output_base_address=output_base_address,
                head_slot=head_slot,
                rows=rows,
                row_ranges=row_ranges,
                reciprocal=False,
            )
            return

        gp_m_res, gp_o, gp_loop = self.register_allocator.allocate_gp(3)
        mask = 1 << head_slot
        lines = [
            f"; === Scale packed O lane {head_slot} by m_res ===",
            f"S_ADDI_INT gp{gp_loop}, gp0, {mask}",
            f"C_SET_V_MASK_REG gp{gp_loop}",
        ]
        for start, end in row_ranges or ((0, rows),):
            lines.extend(
                [
                    *load_large_int(
                        gp_m_res,
                        self._ONLINE_SOFTMAX_FPSRAM_BASE + self.mlen + start,
                    ),
                    *load_large_int(
                        gp_o, output_base_address + start * self.mlen
                    ),
                    f"C_LOOP_START gp{gp_loop}, {end - start}",
                    f"S_LD_FP f1, gp{gp_m_res}, 0",
                    f"V_MUL_VF gp{gp_o}, gp{gp_o}, f1, 1",
                    f"S_ADDI_INT gp{gp_m_res}, gp{gp_m_res}, 1",
                    f"S_ADDI_INT gp{gp_o}, gp{gp_o}, {self.mlen}",
                    f"C_LOOP_END gp{gp_loop}",
                ]
            )
        self.register_allocator.free_gp([gp_m_res, gp_o, gp_loop])
        self.emit("\n".join(lines) + "\n")

    def _emit_pipelined_packed_o_scale(
        self,
        *,
        value_address: int,
        output_base_address: int,
        head_slot: int,
        rows: int,
        row_ranges: tuple[tuple[int, int], ...] | None,
        reciprocal: bool,
    ) -> None:
        """Pipeline independent scalar loads/reciprocals and masked O updates."""

        width = 8
        timing = self.gqa_timing_profile
        if timing is None:
            raise RuntimeError("row-interleaved packed-O scaling requires RTL-v3 timing")
        if width > timing.rob_depth or width > timing.fp_register_count - 1:
            raise RuntimeError(
                f"packed-O width {width} exceeds timing profile ROB/register capacity"
            )
        fp_regs = self.register_allocator.allocate_fp(width)
        gp_value, gp_o, gp_loop, gp_mask = self.register_allocator.allocate_gp(4)
        mask = 1 << head_slot
        lines = [
            "; === RTL-v3 row-interleaved final O scale ==="
            if reciprocal
            else "; === RTL-v3 row-interleaved O scale by m_res ===",
            f"S_ADDI_INT gp{gp_mask}, gp0, {mask}",
            f"C_SET_V_MASK_REG gp{gp_mask}",
        ]

        def body(body_width: int) -> list[str]:
            result = [
                f"S_LD_FP f{fp_regs[row]}, gp{gp_value}, {row}"
                for row in range(body_width)
            ]
            if reciprocal:
                result.extend(
                    f"S_RECI_FP f{fp_regs[row]}, f{fp_regs[row]}, 0"
                    for row in range(body_width)
                )
            for row in range(body_width):
                result.extend(
                    (
                        f"V_MUL_VF gp{gp_o}, gp{gp_o}, f{fp_regs[row]}, 1",
                        f"S_ADDI_INT gp{gp_o}, gp{gp_o}, {self.mlen}",
                    )
                )
            result.append(
                f"S_ADDI_INT gp{gp_value}, gp{gp_value}, {body_width}"
            )
            return result

        active_rows = 0
        try:
            for start, end in row_ranges or ((0, rows),):
                count = end - start
                active_rows += count
                lines.extend(load_large_int(gp_value, value_address + start))
                lines.extend(
                    load_large_int(gp_o, output_base_address + start * self.mlen)
                )
                full_groups, tail = divmod(count, width)
                if full_groups:
                    lines.append(f"C_LOOP_START gp{gp_loop}, {full_groups}")
                    lines.extend(body(width))
                    lines.append(f"C_LOOP_END gp{gp_loop}")
                if tail:
                    lines.append(f"; packed-O pipeline tail rows={tail}")
                    lines.extend(body(tail))
            self.emit("\n".join(lines) + "\n")
            self.record_gqa_pipeline_stats(
                {
                    "o_scale_pipeline_width": width,
                    "interleaved_o_rows": active_rows,
                }
            )
        finally:
            self.register_allocator.free_fp(fp_regs)
            self.register_allocator.free_gp([gp_value, gp_o, gp_loop, gp_mask])

    def _add_pv_to_packed_o_lane(
        self,
        *,
        output_base_address: int,
        pv: VRAMMatrixVar,
        head_slot: int,
        head_slot_dim: int,
        rows: int,
        scratch_address: int,
        scratch_rows: int = 1,
        row_ranges: tuple[tuple[int, int], ...] | None = None,
    ) -> None:
        """Shift one PV head into its packed lane and accumulate it in place."""

        pipelined = getattr(self, "gqa_pipeline_schedule", "row-serial") == "row-interleaved-v1"
        ring_width = max(
            (candidate for candidate in (16, 8, 4, 1) if candidate <= scratch_rows),
            default=1,
        ) if pipelined else 1
        gp_src, gp_dst, gp_scratch, gp_scratch_read, gp_shift, gp_loop = (
            self.register_allocator.allocate_gp(6)
        )
        shift = head_slot * head_slot_dim
        mask = 1 << head_slot
        lines = [
            f"; === Add PV directly to packed O lane {head_slot} ===",
            *load_large_int(gp_scratch, scratch_address),
            *load_large_int(gp_shift, shift),
            f"S_ADDI_INT gp{gp_loop}, gp0, {mask}",
            f"C_SET_V_MASK_REG gp{gp_loop}",
        ]
        pv_address = self.get_vram_addr(pv.name)

        def ring_body(body_width: int) -> list[str]:
            result = [
                *load_large_int(gp_scratch, scratch_address),
                *load_large_int(gp_scratch_read, scratch_address),
            ]
            for _ in range(body_width):
                result.extend(
                    (
                        f"V_SHIFT_V gp{gp_scratch}, gp{gp_src}, gp{gp_shift}",
                        f"S_ADDI_INT gp{gp_scratch}, gp{gp_scratch}, {self.mlen}",
                        f"S_ADDI_INT gp{gp_src}, gp{gp_src}, {self.mlen}",
                    )
                )
            for _ in range(body_width):
                result.extend(
                    (
                        f"V_ADD_VV gp{gp_dst}, gp{gp_dst}, gp{gp_scratch_read}, 1",
                        f"S_ADDI_INT gp{gp_scratch_read}, gp{gp_scratch_read}, {self.mlen}",
                        f"S_ADDI_INT gp{gp_dst}, gp{gp_dst}, {self.mlen}",
                    )
                )
            return result

        active_rows = 0
        for start, end in row_ranges or ((0, rows),):
            count = end - start
            active_rows += count
            lines.extend(load_large_int(gp_src, pv_address + start * self.mlen))
            lines.extend(load_large_int(gp_dst, output_base_address + start * self.mlen))
            if ring_width == 1:
                lines.extend(
                    (
                        f"C_LOOP_START gp{gp_loop}, {count}",
                        f"V_SHIFT_V gp{gp_scratch}, gp{gp_src}, gp{gp_shift}",
                        f"V_ADD_VV gp{gp_dst}, gp{gp_dst}, gp{gp_scratch}, 1",
                        f"S_ADDI_INT gp{gp_src}, gp{gp_src}, {self.mlen}",
                        f"S_ADDI_INT gp{gp_dst}, gp{gp_dst}, {self.mlen}",
                        f"C_LOOP_END gp{gp_loop}",
                    )
                )
                continue
            full_groups, tail = divmod(count, ring_width)
            if full_groups:
                lines.append(f"C_LOOP_START gp{gp_loop}, {full_groups}")
                lines.extend(ring_body(ring_width))
                lines.append(f"C_LOOP_END gp{gp_loop}")
            if tail:
                lines.append(f"; packed-O shift ring tail rows={tail}")
                lines.extend(ring_body(tail))
        self.register_allocator.free_gp(
            [gp_src, gp_dst, gp_scratch, gp_scratch_read, gp_shift, gp_loop]
        )
        self.emit("\n".join(lines) + "\n")
        if pipelined:
            self.record_gqa_pipeline_stats(
                {
                    "o_shift_ring_width": ring_width,
                    "interleaved_o_rows": active_rows,
                }
            )

    def _final_scale_packed_o_lane(
        self,
        *,
        output_base_address: int,
        head_slot: int,
        state_head: int = 0,
        rows: int,
        row_ranges: tuple[tuple[int, int], ...] | None = None,
    ) -> None:
        """Apply the final online-softmax normalization to one packed lane."""

        state_layout = getattr(self, "_softmax_state_layout", None)
        l_address = (
            state_layout.l_base(state_head)
            if state_layout is not None
            else self._ONLINE_SOFTMAX_FPSRAM_BASE + 2 * self.mlen
        )
        immediate_shape = lambda value: tuple(
            line.split(maxsplit=1)[0] for line in load_large_int(1, value)
        )
        template_key = (
            "packed_final_scale_v1",
            self.mlen,
            rows,
            tuple(row_ranges or ((0, rows),)),
            head_slot,
            state_head,
            getattr(self, "gqa_pipeline_schedule", "row-serial"),
            immediate_shape(l_address),
            immediate_shape(output_base_address),
        )
        stats_cache = getattr(self, "_packed_final_scale_summary_stats", None)
        if stats_cache is None:
            stats_cache = {}
            self._packed_final_scale_summary_stats = stats_cache
        if self.replay_cost_summary_template(template_key):
            for name, delta in stats_cache[template_key].items():
                self._packed_attention_stats[name] += delta
            return

        before_stats = dict(self._packed_attention_stats)
        with self.cost_summary_template(template_key):
            self._final_scale_packed_o_lane_body(
                output_base_address=output_base_address,
                head_slot=head_slot,
                rows=rows,
                row_ranges=row_ranges,
                l_address=l_address,
            )
        stats_cache[template_key] = {
            name: int(value) - int(before_stats.get(name, 0))
            for name, value in self._packed_attention_stats.items()
            if isinstance(value, (int, bool))
            and int(value) != int(before_stats.get(name, 0))
        }

    def _final_scale_packed_o_lane_body(
        self,
        *,
        output_base_address: int,
        head_slot: int,
        rows: int,
        row_ranges: tuple[tuple[int, int], ...] | None,
        l_address: int,
    ) -> None:
        if getattr(self, "gqa_pipeline_schedule", "row-serial") == "row-interleaved-v1":
            self._emit_pipelined_packed_o_scale(
                value_address=l_address,
                output_base_address=output_base_address,
                head_slot=head_slot,
                rows=rows,
                row_ranges=row_ranges,
                reciprocal=True,
            )
            return

        gp_l, gp_o, gp_loop = self.register_allocator.allocate_gp(3)
        mask = 1 << head_slot
        lines = [
            f"; === Final scale packed O lane {head_slot} ===",
            f"S_ADDI_INT gp{gp_loop}, gp0, {mask}",
            f"C_SET_V_MASK_REG gp{gp_loop}",
        ]
        for start, end in row_ranges or ((0, rows),):
            lines.extend(
                [
                    *load_large_int(
                        gp_l,
                        l_address + start,
                    ),
                    *load_large_int(
                        gp_o, output_base_address + start * self.mlen
                    ),
                    f"C_LOOP_START gp{gp_loop}, {end - start}",
                    f"S_LD_FP f1, gp{gp_l}, 0",
                    f"S_RECI_FP f1, f1, 0",
                    f"V_MUL_VF gp{gp_o}, gp{gp_o}, f1, 1",
                    f"S_ADDI_INT gp{gp_l}, gp{gp_l}, 1",
                    f"S_ADDI_INT gp{gp_o}, gp{gp_o}, {self.mlen}",
                    f"C_LOOP_END gp{gp_loop}",
                ]
            )
        self.register_allocator.free_gp([gp_l, gp_o, gp_loop])
        self.emit("\n".join(lines) + "\n")

    def _emit_packed_attention_group_k_major(
        self,
        *,
        Q_group: VRAMMatrixVar,
        K: InputVar,
        V: InputVar,
        group_heads: int,
        head_slot_dim: int,
        output_base_address: int,
        output_physical_rows: int,
        scratch_base_address: int,
        q_lane_base: int,
        softmax_scale: float,
        causal_mask: bool | VRAMMatrixVar | None,
        k_idx: int,
        active_k_cols: int,
        resident_k_mram: tuple[int, ...] | None,
        resident_v_mram: tuple[int, ...] | None,
        kv_residency_plan: KVResidencyPlan | None,
    ) -> None:
        """Emit one packed GQA group in Q-block/K-block/head order.

        ``M_BMM_WO`` writes every broadcast score tile.  Keeping the head loop
        inside the K loop consumes each tile once and avoids recomputing QK for
        every head.  K and V are likewise loaded once per K block.
        """

        mlen = self.mlen
        seq_len = Q_group.shape[0]
        num_q_blocks = math.ceil(seq_len / mlen)
        num_k_blocks = math.ceil(active_k_cols / mlen)
        s_views = [
            self.alloc_at(
                f"_packed_kmajor_S_h{head}",
                mlen,
                mlen,
                scratch_base_address + (q_lane_base + head) * mlen * mlen,
                physical_shape=(mlen, mlen),
            )
            for head in range(group_heads)
        ]
        pv = self.alloc(
            "_packed_kmajor_PV",
            mlen,
            head_slot_dim,
            strict=False,
            physical_shape=(mlen, mlen),
        )
        pack_scratch = None
        for candidate in (16, 8, 4, 1):
            try:
                pack_scratch = self.alloc(
                    "_packed_kmajor_pack_scratch",
                    candidate,
                    mlen,
                    strict=False,
                    physical_shape=(candidate, mlen),
                )
                break
            except MemoryError:
                continue
        if pack_scratch is None:
            raise MemoryError("unable to allocate packed K-major O scratch")
        pack_scratch_addr = self.get_vram_addr(pack_scratch.name)
        pack_scratch_rows = pack_scratch.physical_shape[0]
        sequence_plan = getattr(self, "_native_sequence_packing", None)
        kv_slot_elems = mlen * mlen
        summary_head_census: dict[
            tuple[object, ...], list[object]
        ] = {}
        summary_qk_census: dict[
            tuple[object, ...], list[object]
        ] = {}

        try:
            for q_block in range(num_q_blocks):
                rows = min(mlen, seq_len - q_block * mlen)
                if (
                    sequence_plan is not None
                    and num_q_blocks == 1
                    and int(sequence_plan.attention_group_seq_len) == seq_len
                ):
                    active_ranges = tuple(
                        (
                            slot * int(sequence_plan.batch_slot_rows),
                            slot * int(sequence_plan.batch_slot_rows)
                            + int(sequence_plan.seq_len),
                        )
                        for slot in range(int(sequence_plan.batch_pack_factor))
                    )
                else:
                    active_ranges = ((0, rows),)
                active_rows = sum(end - start for start, end in active_ranges)
                head_outputs: list[tuple[int, int]] = []
                for head in range(group_heads):
                    output_head = q_lane_base + head
                    output_col_block = (output_head * head_slot_dim) // mlen
                    output_lane = (
                        (output_head * head_slot_dim) % mlen // head_slot_dim
                    )
                    output_block_base = (
                        output_base_address
                        + output_col_block * output_physical_rows * mlen
                        + q_block * mlen * mlen
                    )
                    head_outputs.append((output_block_base, output_lane))

                processed_k_blocks = tuple(
                    block
                    for block in range(num_k_blocks)
                    if causal_mask is None or block <= q_block
                )
                has_streaming_kv = (
                    kv_residency_plan is None
                    or kv_residency_plan.streaming_blocks > 0
                )
                double_buffered_kv = (
                    getattr(self, "gqa_pipeline_schedule", "row-serial")
                    == "row-interleaved-v1"
                    and has_streaming_kv
                    and self.mram_tile_capacity >= 2
                )
                if double_buffered_kv:
                    self.record_gqa_pipeline_stats(
                        {"gqa_kv_double_buffered": True}
                    )
                elif (
                    getattr(self, "gqa_pipeline_schedule", "row-serial")
                    == "row-interleaved-v1"
                    and has_streaming_kv
                ):
                    self.record_gqa_pipeline_stats(
                        {
                            "gqa_pipeline_fallback_reason":
                            "mram_tile_capacity_lt_2"
                        }
                    )
                if (
                    self.cost_affine_summary_enabled()
                    and kv_residency_plan is not None
                ):
                    # The unordered CostEmitter summary needs only a finite
                    # census of exact kernel shapes.  Enumerating every
                    # causal Q/K/head pair first made long-context compile
                    # time quadratic even though those pairs were later
                    # merged into the same templates.
                    processed_count = len(processed_k_blocks)
                    resident_count = min(
                        kv_residency_plan.resident_prefix_blocks,
                        processed_count,
                    )
                    streaming_start = resident_count
                    streaming_count = processed_count - resident_count
                    if streaming_count:
                        streaming_blocks = tuple(
                            range(streaming_start, processed_count)
                        )
                        k_tiles = tuple(
                            (
                                k_idx + block,
                                kv_residency_plan.k_address(block),
                            )
                            for block in streaming_blocks
                        )
                        v_tiles = tuple(
                            (
                                k_idx + block,
                                kv_residency_plan.v_address(block),
                            )
                            for block in streaming_blocks
                        )
                        affine_kv_summary = (
                            self._emit_packed_kv_prefetch_affine_summary(
                                K,
                                k_tiles,
                            )
                            and self._emit_packed_kv_prefetch_affine_summary(
                                V,
                                v_tiles,
                            )
                        )
                        if not affine_kv_summary:
                            raise ValueError(
                                "affine CostEmitter summary requires "
                                "contiguous streaming K/V blocks"
                            )
                        self._packed_attention_stats[
                            "kv_tile_load_count"
                        ] += 2 * streaming_count
                        self._packed_attention_stats[
                            "kv_cache_misses"
                        ] += 2 * streaming_count
                        if double_buffered_kv:
                            self.record_gqa_pipeline_stats(
                                {
                                    "gqa_dma_overlap_eligible_occurrences":
                                        streaming_count
                                }
                            )
                    if resident_count:
                        self._packed_attention_stats[
                            "kv_cache_hits"
                        ] += 2 * resident_count

                    stats = self._packed_attention_stats
                    stats["qk_compute_count"] += processed_count
                    stats["ideal_qk_compute_count"] += processed_count

                    q_base = (
                        self.get_vram_addr(Q_group.name)
                        + q_block * mlen * mlen
                    )

                    def add_qk_class(
                        block: int,
                        repetitions: int,
                    ) -> None:
                        if repetitions <= 0:
                            return
                        k_address = kv_residency_plan.k_address(block)
                        qkt_key = self._packed_qkt_from_mram_template_key(
                            q_base=q_base,
                            k_mram_address=k_address,
                            s_base_address=scratch_base_address,
                        )
                        qkt_entry = summary_qk_census.get(qkt_key)
                        if qkt_entry is None:
                            summary_qk_census[qkt_key] = [
                                q_block,
                                k_address,
                                repetitions,
                            ]
                        else:
                            qkt_entry[2] = (
                                int(qkt_entry[2]) + repetitions
                            )

                    # Resident addresses can cross immediate-encoding
                    # boundaries, so retain one lightweight classification
                    # per resident block. All streamed blocks share one MRAM
                    # slot and therefore one exact QK instruction shape.
                    for block in range(resident_count):
                        add_qk_class(block, 1)
                    if streaming_count:
                        add_qk_class(streaming_start, streaming_count)

                    q_head_kernel_census: dict[
                        tuple[object, ...], list[object]
                    ] = {}
                    valid_mask_cache: dict[
                        tuple[int, bool], VRAMMatrixVar | None
                    ] = {}

                    def add_head_class(
                        block: int,
                        repetitions: int,
                    ) -> None:
                        if repetitions <= 0:
                            return
                        block_cols = min(
                            mlen,
                            active_k_cols - block * mlen,
                        )
                        apply_causal_mask = (
                            causal_mask is not None
                            and block == q_block
                        )
                        valid_mask_key = (
                            block_cols,
                            apply_causal_mask,
                        )
                        if valid_mask_key not in valid_mask_cache:
                            valid_mask_cache[valid_mask_key] = (
                                self._valid_col_mask_for_tile(
                                    block_cols,
                                    causal_mask=causal_mask,
                                    apply_causal_mask=(
                                        apply_causal_mask
                                    ),
                                    causal_mask_covers_padding=(
                                        seq_len <= mlen
                                        and active_k_cols <= mlen
                                    ),
                                )
                            )
                        valid_col_mask = valid_mask_cache[
                            valid_mask_key
                        ]
                        softmax_valid_cols = (
                            None
                            if valid_col_mask is not None
                            or (
                                isinstance(
                                    causal_mask,
                                    VRAMMatrixVar,
                                )
                                and apply_causal_mask
                            )
                            else block_cols
                        )
                        first_block = block == 0
                        last_block = block == processed_count - 1
                        v_mram_address = (
                            kv_residency_plan.v_address(block)
                        )
                        kernel_class = (
                            first_block,
                            last_block,
                            softmax_valid_cols,
                            valid_col_mask is not None,
                            isinstance(causal_mask, VRAMMatrixVar)
                            and apply_causal_mask,
                            causal_mask is True
                            and apply_causal_mask,
                            tuple(
                                line.split(maxsplit=1)[0]
                                for line in load_large_int(
                                    1,
                                    v_mram_address,
                                )
                            ),
                        )
                        class_entry = q_head_kernel_census.get(
                            kernel_class
                        )
                        if class_entry is None:
                            q_head_kernel_census[kernel_class] = [
                                {
                                    "causal_mask": causal_mask,
                                    "valid_col_mask": valid_col_mask,
                                    "apply_causal_mask":
                                        apply_causal_mask,
                                    "softmax_valid_cols":
                                        softmax_valid_cols,
                                    "first_block": first_block,
                                    "last_block": last_block,
                                    "physical_k_idx": k_idx + block,
                                    "v_mram_address":
                                        v_mram_address,
                                },
                                repetitions,
                            ]
                        else:
                            class_entry[1] = (
                                int(class_entry[1]) + repetitions
                            )

                    for block in range(resident_count):
                        add_head_class(block, 1)

                    if streaming_count:
                        special_streaming = {
                            block
                            for block in (
                                streaming_start,
                                processed_count - 1,
                                num_k_blocks - 1,
                            )
                            if streaming_start
                            <= block
                            < processed_count
                        }
                        for block in sorted(special_streaming):
                            add_head_class(block, 1)
                        ordinary_streaming_count = (
                            streaming_count - len(special_streaming)
                        )
                        if ordinary_streaming_count:
                            representative = next(
                                block
                                for block in range(
                                    streaming_start,
                                    processed_count,
                                )
                                if block not in special_streaming
                            )
                            add_head_class(
                                representative,
                                ordinary_streaming_count,
                            )

                    for class_kwargs, repetitions in (
                        q_head_kernel_census.values()
                    ):
                        for head, s_head in enumerate(s_views):
                            output_block_base, output_lane = (
                                head_outputs[head]
                            )
                            head_kwargs = dict(
                                s_block=s_head,
                                pv_matrix=pv,
                                softmax_scale=softmax_scale,
                                state_head=head,
                                output_block_base=output_block_base,
                                output_lane=output_lane,
                                head_slot_dim=head_slot_dim,
                                rows=rows,
                                scratch_address=pack_scratch_addr,
                                scratch_rows=pack_scratch_rows,
                                row_ranges=active_ranges,
                                active_rows=active_rows,
                                **class_kwargs,
                            )
                            template_key = (
                                self._packed_kmajor_head_template_key(
                                    **head_kwargs
                                )
                            )
                            entry = summary_head_census.get(template_key)
                            if entry is None:
                                summary_head_census[template_key] = [
                                    head_kwargs,
                                    int(repetitions),
                                ]
                            else:
                                entry[1] = (
                                    int(entry[1])
                                    + int(repetitions)
                                )

                    for head, (
                        output_block_base,
                        output_lane,
                    ) in enumerate(head_outputs):
                        self._final_scale_packed_o_lane(
                            output_base_address=output_block_base,
                            head_slot=output_lane,
                            state_head=head,
                            rows=rows,
                            row_ranges=active_ranges,
                        )
                    continue

                summary_streaming_blocks = tuple(
                    block
                    for block in processed_k_blocks
                    if kv_residency_plan is not None
                    and not kv_residency_plan.is_resident(block)
                )
                affine_kv_summary = False
                if summary_streaming_blocks:
                    k_tiles = tuple(
                        (
                            k_idx + block,
                            kv_residency_plan.k_address(block),
                        )
                        for block in summary_streaming_blocks
                    )
                    v_tiles = tuple(
                        (
                            k_idx + block,
                            kv_residency_plan.v_address(block),
                        )
                        for block in summary_streaming_blocks
                    )
                    affine_kv_summary = (
                        self._emit_packed_kv_prefetch_affine_summary(K, k_tiles)
                        and self._emit_packed_kv_prefetch_affine_summary(V, v_tiles)
                    )
                    if affine_kv_summary:
                        self._packed_attention_stats["kv_tile_load_count"] += (
                            2 * len(summary_streaming_blocks)
                        )
                        self._packed_attention_stats["kv_cache_misses"] += (
                            2 * len(summary_streaming_blocks)
                        )
                q_head_kernel_census: dict[
                    tuple[object, ...], list[object]
                ] = {}
                valid_mask_cache: dict[
                    tuple[int, bool], VRAMMatrixVar | None
                ] = {}
                for processed_index, k_block in enumerate(processed_k_blocks):
                    block_cols = min(mlen, active_k_cols - k_block * mlen)
                    physical_k_idx = k_idx + k_block
                    block_resident = (
                        kv_residency_plan is not None
                        and kv_residency_plan.is_resident(k_block)
                    )
                    if block_resident:
                        stats = self._packed_attention_stats
                        stats["kv_cache_hits"] += 2
                        qkt_mram_address = kv_residency_plan.k_address(k_block)
                    elif kv_residency_plan is not None:
                        if not affine_kv_summary:
                            self._emit_packed_kv_prefetch(
                                K,
                                (
                                    (
                                        physical_k_idx,
                                        kv_residency_plan.k_address(k_block),
                                    ),
                                ),
                            )
                        qkt_mram_address = kv_residency_plan.k_address(k_block)
                    elif resident_k_mram is None:
                        self._emit_packed_qkt_to_s(
                            Q_group=Q_group,
                            K=K,
                            q_idx=q_block,
                            k_idx=physical_k_idx,
                            s_base_address=scratch_base_address,
                        )
                        qkt_mram_address = None
                    else:
                        qkt_mram_address = resident_k_mram[k_block]
                    if qkt_mram_address is not None:
                        if self.cost_affine_summary_enabled():
                            q_base = (
                                self.get_vram_addr(Q_group.name)
                                + q_block * mlen * mlen
                            )
                            qkt_key = self._packed_qkt_from_mram_template_key(
                                q_base=q_base,
                                k_mram_address=qkt_mram_address,
                                s_base_address=scratch_base_address,
                            )
                            entry = summary_qk_census.get(qkt_key)
                            if entry is None:
                                summary_qk_census[qkt_key] = [
                                    q_block,
                                    qkt_mram_address,
                                    1,
                                ]
                            else:
                                entry[2] = int(entry[2]) + 1
                        else:
                            self._emit_packed_qkt_from_mram(
                                Q_group=Q_group,
                                q_idx=q_block,
                                k_mram_address=qkt_mram_address,
                                s_base_address=scratch_base_address,
                            )
                    stats = self._packed_attention_stats
                    stats["qk_compute_count"] += 1
                    stats["ideal_qk_compute_count"] += 1

                    if block_resident:
                        v_mram_address = kv_residency_plan.v_address(k_block)
                    elif kv_residency_plan is not None:
                        if not affine_kv_summary:
                            self._emit_packed_kv_prefetch(
                                V,
                                (
                                    (
                                        physical_k_idx,
                                        kv_residency_plan.v_address(k_block),
                                    ),
                                ),
                            )
                            stats["kv_tile_load_count"] += 2
                            stats["kv_cache_misses"] += 2
                        v_mram_address = kv_residency_plan.v_address(k_block)
                        if double_buffered_kv:
                            self.record_gqa_pipeline_stats(
                                {"gqa_dma_overlap_eligible_occurrences": 1}
                            )
                    elif resident_v_mram is None:
                        self._emit_packed_kv_prefetch(
                            V,
                            (
                                (
                                    physical_k_idx,
                                    kv_slot_elems if double_buffered_kv else 0,
                                ),
                            ),
                        )
                        stats["kv_tile_load_count"] += 2
                        v_mram_address = kv_slot_elems if double_buffered_kv else 0
                    else:
                        v_mram_address = resident_v_mram[k_block]

                    apply_causal_mask = (
                        causal_mask is not None and k_block == q_block
                    )
                    valid_mask_key = (block_cols, apply_causal_mask)
                    if valid_mask_key not in valid_mask_cache:
                        valid_mask_cache[valid_mask_key] = (
                            self._valid_col_mask_for_tile(
                                block_cols,
                                causal_mask=causal_mask,
                                apply_causal_mask=apply_causal_mask,
                                causal_mask_covers_padding=(
                                    seq_len <= mlen
                                    and active_k_cols <= mlen
                                ),
                            )
                        )
                    valid_col_mask = valid_mask_cache[valid_mask_key]
                    softmax_valid_cols = (
                        None
                        if valid_col_mask is not None
                        or (
                            isinstance(causal_mask, VRAMMatrixVar)
                            and apply_causal_mask
                        )
                        else block_cols
                    )
                    first_block = processed_index == 0
                    last_block = processed_index + 1 == len(processed_k_blocks)
                    if self.cost_affine_summary_enabled():
                        kernel_class = (
                            first_block,
                            last_block,
                            softmax_valid_cols,
                            valid_col_mask is not None,
                            isinstance(causal_mask, VRAMMatrixVar)
                            and apply_causal_mask,
                            causal_mask is True and apply_causal_mask,
                            tuple(
                                line.split(maxsplit=1)[0]
                                for line in load_large_int(
                                    1, v_mram_address
                                )
                            ),
                        )
                        class_entry = q_head_kernel_census.get(kernel_class)
                        if class_entry is None:
                            q_head_kernel_census[kernel_class] = [
                                {
                                    "causal_mask": causal_mask,
                                    "valid_col_mask": valid_col_mask,
                                    "apply_causal_mask": apply_causal_mask,
                                    "softmax_valid_cols": softmax_valid_cols,
                                    "first_block": first_block,
                                    "last_block": last_block,
                                    "physical_k_idx": physical_k_idx,
                                    "v_mram_address": v_mram_address,
                                },
                                1,
                            ]
                        else:
                            class_entry[1] = int(class_entry[1]) + 1
                        continue
                    for head, s_head in enumerate(s_views):
                        output_block_base, output_lane = head_outputs[head]
                        head_kwargs = dict(
                            s_block=s_head,
                            pv_matrix=pv,
                            causal_mask=causal_mask,
                            valid_col_mask=valid_col_mask,
                            apply_causal_mask=apply_causal_mask,
                            softmax_scale=softmax_scale,
                            softmax_valid_cols=softmax_valid_cols,
                            state_head=head,
                            first_block=first_block,
                            last_block=last_block,
                            physical_k_idx=physical_k_idx,
                            v_mram_address=v_mram_address,
                            output_block_base=output_block_base,
                            output_lane=output_lane,
                            head_slot_dim=head_slot_dim,
                            rows=rows,
                            scratch_address=pack_scratch_addr,
                            scratch_rows=pack_scratch_rows,
                            row_ranges=active_ranges,
                            active_rows=active_rows,
                        )
                        self._emit_packed_kmajor_head_kernel(
                            **head_kwargs
                        )

                    if (
                        double_buffered_kv
                        and kv_residency_plan is None
                        and processed_index + 1 < len(processed_k_blocks)
                    ):
                        self._emit_packed_kv_prefetch(
                            K,
                            ((k_idx + processed_k_blocks[processed_index + 1], 0),),
                        )
                        self.record_gqa_pipeline_stats(
                            {"gqa_dma_overlap_eligible_occurrences": 1}
                        )

                if self.cost_affine_summary_enabled():
                    for class_kwargs, repetitions in (
                        q_head_kernel_census.values()
                    ):
                        for head, s_head in enumerate(s_views):
                            output_block_base, output_lane = (
                                head_outputs[head]
                            )
                            head_kwargs = dict(
                                s_block=s_head,
                                pv_matrix=pv,
                                softmax_scale=softmax_scale,
                                state_head=head,
                                output_block_base=output_block_base,
                                output_lane=output_lane,
                                head_slot_dim=head_slot_dim,
                                rows=rows,
                                scratch_address=pack_scratch_addr,
                                scratch_rows=pack_scratch_rows,
                                row_ranges=active_ranges,
                                active_rows=active_rows,
                                **class_kwargs,
                            )
                            template_key = (
                                self._packed_kmajor_head_template_key(
                                    **head_kwargs
                                )
                            )
                            entry = summary_head_census.get(template_key)
                            if entry is None:
                                summary_head_census[template_key] = [
                                    head_kwargs,
                                    int(repetitions),
                                ]
                            else:
                                entry[1] = (
                                    int(entry[1]) + int(repetitions)
                                )

                for head, (output_block_base, output_lane) in enumerate(
                    head_outputs
                ):
                    self._final_scale_packed_o_lane(
                        output_base_address=output_block_base,
                        head_slot=output_lane,
                        state_head=head,
                        rows=rows,
                        row_ranges=active_ranges,
                    )
            for q_block, qkt_mram_address, repetitions in (
                summary_qk_census.values()
            ):
                self._emit_packed_qkt_from_mram(
                    Q_group=Q_group,
                    q_idx=int(q_block),
                    k_mram_address=int(qkt_mram_address),
                    s_base_address=scratch_base_address,
                    summary_repetitions=int(repetitions),
                )
            for head_kwargs, repetitions in summary_head_census.values():
                self._emit_packed_kmajor_head_kernel(
                    **head_kwargs,
                    summary_repetitions=int(repetitions),
                )
        finally:
            self.free_tensor(pv)
            self.free_tensor(pack_scratch)

    def _packed_kmajor_head_template_key(
        self,
        *,
        s_block: VRAMMatrixVar,
        pv_matrix: VRAMMatrixVar,
        causal_mask: bool | VRAMMatrixVar | None,
        valid_col_mask: VRAMMatrixVar | None,
        apply_causal_mask: bool,
        softmax_scale: float,
        softmax_valid_cols: int | None,
        state_head: int,
        first_block: bool,
        last_block: bool,
        physical_k_idx: int,
        v_mram_address: int,
        output_block_base: int,
        output_lane: int,
        head_slot_dim: int,
        rows: int,
        scratch_address: int,
        scratch_rows: int,
        row_ranges: tuple[tuple[int, int], ...],
        active_rows: int,
    ) -> tuple[object, ...]:
        del softmax_scale, physical_k_idx, active_rows

        def immediate_shape(value: int) -> tuple[str, ...]:
            return tuple(
                line.split(maxsplit=1)[0]
                for line in load_large_int(1, value)
            )

        return (
            "packed_kmajor_head_v1",
            self.mlen,
            self.blen,
            rows,
            tuple(row_ranges),
            head_slot_dim,
            state_head,
            first_block,
            last_block,
            softmax_valid_cols,
            valid_col_mask is not None,
            isinstance(causal_mask, VRAMMatrixVar) and apply_causal_mask,
            causal_mask is True and apply_causal_mask,
            output_lane,
            scratch_rows,
            immediate_shape(self.get_vram_addr(s_block.name)),
            immediate_shape(self.get_vram_addr(pv_matrix.name)),
            immediate_shape(v_mram_address),
            immediate_shape(output_block_base),
            immediate_shape(scratch_address),
        )

    def _emit_packed_kmajor_head_kernel(
        self,
        *,
        s_block: VRAMMatrixVar,
        pv_matrix: VRAMMatrixVar,
        causal_mask: bool | VRAMMatrixVar | None,
        valid_col_mask: VRAMMatrixVar | None,
        apply_causal_mask: bool,
        softmax_scale: float,
        softmax_valid_cols: int | None,
        state_head: int,
        first_block: bool,
        last_block: bool,
        physical_k_idx: int,
        v_mram_address: int,
        output_block_base: int,
        output_lane: int,
        head_slot_dim: int,
        rows: int,
        scratch_address: int,
        scratch_rows: int,
        row_ranges: tuple[tuple[int, int], ...],
        active_rows: int,
        summary_repetitions: int = 1,
    ) -> None:
        """Lower or algebraically replay one memory-free K-major head kernel."""

        template_key = self._packed_kmajor_head_template_key(
            s_block=s_block,
            pv_matrix=pv_matrix,
            causal_mask=causal_mask,
            valid_col_mask=valid_col_mask,
            apply_causal_mask=apply_causal_mask,
            softmax_scale=softmax_scale,
            softmax_valid_cols=softmax_valid_cols,
            state_head=state_head,
            first_block=first_block,
            last_block=last_block,
            physical_k_idx=physical_k_idx,
            v_mram_address=v_mram_address,
            output_block_base=output_block_base,
            output_lane=output_lane,
            head_slot_dim=head_slot_dim,
            rows=rows,
            scratch_address=scratch_address,
            scratch_rows=scratch_rows,
            row_ranges=row_ranges,
            active_rows=active_rows,
        )
        stats_cache = getattr(self, "_packed_kmajor_summary_stats", None)
        if stats_cache is None:
            stats_cache = {}
            self._packed_kmajor_summary_stats = stats_cache
        if self.replay_cost_summary_template(
            template_key, count=summary_repetitions
        ):
            for name, delta in stats_cache[template_key].items():
                self._packed_attention_stats[name] += (
                    delta * summary_repetitions
                )
            return

        before_stats = dict(self._packed_attention_stats)
        with self.cost_summary_template(template_key):
            if valid_col_mask is not None:
                self.vram_add(s_block, valid_col_mask, num_rows=rows)
            if isinstance(causal_mask, VRAMMatrixVar) and apply_causal_mask:
                self.vram_add(s_block, causal_mask, num_rows=rows)
            elif causal_mask is True and apply_causal_mask:
                self.emit(
                    "; NOTE: packed attention received causal_mask=True "
                    "without a VRAM mask; no mask applied.\n"
                )
            if first_block:
                self.online_softmax_first_block(
                    s_block,
                    softmax_scale,
                    rows=rows,
                    valid_cols=softmax_valid_cols,
                    state_head=state_head,
                    last_block=last_block,
                    stream_state=True,
                )
                stats = self._packed_attention_stats
                stats["softmax_first_block_specialized_count"] += 1
                stats["softmax_first_block_specialized_rows"] += rows
                stats["softmax_m_moves_elided"] += rows
                stats["softmax_l_moves_elided"] += rows
                if last_block:
                    stats["softmax_m_stores_elided"] += rows
            else:
                self.online_softmax_block(
                    s_block,
                    softmax_scale,
                    rows=rows,
                    valid_cols=softmax_valid_cols,
                    state_head=state_head,
                    output_address=output_block_base,
                    output_head_slot=output_lane,
                    last_block=last_block,
                )
                stats = self._packed_attention_stats
                stats["m_res_stores_elided"] += rows
                stats["m_res_loads_elided"] += rows
                stats["m_res_streamed_rows"] += rows
                if last_block:
                    stats["softmax_m_stores_elided"] += rows

            self.emit(f"; Compute PV = P @ V[k_idx={physical_k_idx}]\n")
            self._compute_pv_from_mram(
                s_block=s_block,
                pv_matrix=pv_matrix,
                v_mram_address=v_mram_address,
                head_slot_dim=head_slot_dim,
                rows=rows,
            )
            self._packed_attention_stats["pv_compute_count"] += 1
            self._add_pv_to_packed_o_lane(
                output_base_address=output_block_base,
                pv=pv_matrix,
                head_slot=output_lane,
                head_slot_dim=head_slot_dim,
                rows=rows,
                scratch_address=scratch_address,
                scratch_rows=scratch_rows,
                row_ranges=row_ranges,
            )
            self._packed_attention_stats["direct_o_lane_updates"] += active_rows

        stats_cache[template_key] = {
            name: int(value) - int(before_stats.get(name, 0))
            for name, value in self._packed_attention_stats.items()
            if isinstance(value, (int, bool))
            and int(value) != int(before_stats.get(name, 0))
        }
        if summary_repetitions > 1:
            self.replay_cost_summary_template(
                template_key, count=summary_repetitions - 1
            )
            for name, delta in stats_cache[template_key].items():
                self._packed_attention_stats[name] += (
                    delta * (summary_repetitions - 1)
                )

    def _emit_packed_attention_group_internal(
        self,
        *,
        Q_group: VRAMMatrixVar,
        K: InputVar,
        V: InputVar,
        group_heads: int,
        head_slot_dim: int,
        output_base_address: int,
        output_physical_rows: int,
        scratch_base_address: int,
        broadcast_amount: int,
        scale: float,
        causal_mask: bool | VRAMMatrixVar | None,
        output_head_base: int = 0,
        k_idx: int = 0,
        valid_cols: int | None = None,
        resident_k_mram: tuple[int, ...] | None = None,
        resident_v_mram: tuple[int, ...] | None = None,
        kv_residency_plan: KVResidencyPlan | None = None,
        output_precleared: bool = False,
        direct_output: bool = False,
        q_lane_base: int = 0,
    ) -> None:
        """Compiler-owned packed-head flash attention for one KV group."""
        seq_len, q_width = Q_group.shape
        mlen = self.mlen
        q_physical_width = Q_group.physical_shape[1]
        if q_width > mlen or q_physical_width != mlen:
            raise ValueError(
                f"packed Q group must fit in one physical MLEN row, "
                f"got logical_width={q_width}, physical_width={q_physical_width}, MLEN={mlen}"
            )
        if group_heads > broadcast_amount:
            raise ValueError(f"group_heads={group_heads} exceeds broadcast_amount={broadcast_amount}")
        if q_lane_base < 0 or q_lane_base + group_heads > broadcast_amount:
            raise ValueError(
                f"Q lane range [{q_lane_base}, {q_lane_base + group_heads}) "
                f"exceeds hardware broadcast lanes={broadcast_amount}"
            )
        if broadcast_amount * head_slot_dim > mlen:
            raise ValueError(
                f"broadcast_amount*head_slot_dim must fit MLEN "
                f"({broadcast_amount}*{head_slot_dim} > {mlen})"
            )
        if getattr(self, "hlen", head_slot_dim) != head_slot_dim:
            raise ValueError(f"Packed attention requires HLEN={head_slot_dim}, got {self.hlen}")

        self._ensure_hbm_sub_matrix_registered(K)
        self._ensure_hbm_sub_matrix_registered(V)
        self._ensure_vram_matrix_layout(Q_group.name)

        active_k_cols = valid_cols or seq_len
        if active_k_cols <= 0:
            raise ValueError(f"Packed attention requires positive active K columns, got {active_k_cols}")
        if active_k_cols > K.shape[0]:
            raise ValueError(f"valid_cols={active_k_cols} exceeds K rows {K.shape[0]}")
        if active_k_cols > V.shape[0]:
            raise ValueError(f"valid_cols={active_k_cols} exceeds V rows {V.shape[0]}")

        num_q_blocks = math.ceil(seq_len / mlen)
        num_k_blocks = math.ceil(active_k_cols / mlen)
        if (resident_k_mram is None) != (resident_v_mram is None):
            raise ValueError("resident K and V MRAM address lists must be provided together")
        if kv_residency_plan is not None and resident_k_mram is not None:
            raise ValueError(
                "provide either kv_residency_plan or legacy resident K/V lists, not both"
            )
        if (
            kv_residency_plan is not None
            and kv_residency_plan.k_blocks != num_k_blocks
        ):
            raise ValueError(
                "K/V residency plan block count does not match attention shape: "
                f"{kv_residency_plan.k_blocks} != {num_k_blocks}"
            )
        if resident_k_mram is not None:
            if len(resident_k_mram) != num_k_blocks or len(resident_v_mram) != num_k_blocks:
                raise ValueError(
                    f"resident K/V tile count must equal k_blocks={num_k_blocks}, got "
                    f"{len(resident_k_mram)} and {len(resident_v_mram)}"
                )
        if direct_output and (group_heads != 1 or broadcast_amount != 1 or output_head_base != 0):
            raise ValueError(
                "direct packed-GQA output requires one active head, physical broadcast 1, "
                "and output_head_base 0"
            )
        # M_BTMM applies the emulator's fixed bmm_scale (0.25). Compensate in
        # online softmax so the effective QK scale remains caller-provided.
        softmax_scale = scale / 0.25
        if (
            getattr(self, "packed_qk_schedule", "head-major-v1")
            == "broadcast-k-major-v1"
        ):
            if (
                getattr(self, "softmax_state_schedule", "sram-v1")
                != "streamed-v2"
            ):
                raise ValueError(
                    "broadcast-k-major-v1 requires "
                    "softmax_state_schedule='streamed-v2'"
                )
            state_layout = getattr(self, "_softmax_state_layout", None)
            if (
                state_layout is None
                or state_layout.active_broadcast_heads < group_heads
            ):
                # Native frontends install the exact shared layout after head
                # packing. Low-level builder APIs do not have that planner, so
                # lazily provision the minimum layout their group requires.
                self.configure_softmax_state_layout(group_heads)
            if not output_precleared:
                # Public packed-attention entry points historically owned
                # output initialization. Preserve that contract while the
                # native decoder can still clear a shared packed-O allocation
                # once and pass ``output_precleared=True``.
                self.emit(
                    self._reset_vram_asm(
                        start_address=output_base_address,
                        rows=seq_len,
                        cols=mlen,
                        total_rows=output_physical_rows,
                        mlen=mlen,
                    )
                )
                output_precleared = True
            self._emit_packed_attention_group_k_major(
                Q_group=Q_group,
                K=K,
                V=V,
                group_heads=group_heads,
                head_slot_dim=head_slot_dim,
                output_base_address=output_base_address,
                output_physical_rows=output_physical_rows,
                scratch_base_address=scratch_base_address,
                q_lane_base=q_lane_base,
                softmax_scale=softmax_scale,
                causal_mask=causal_mask,
                k_idx=k_idx,
                active_k_cols=active_k_cols,
                resident_k_mram=resident_k_mram,
                resident_v_mram=resident_v_mram,
                kv_residency_plan=kv_residency_plan,
            )
            return

        s_views = [
            self.alloc_at(
                f"_packed_S_h{head}",
                mlen,
                mlen,
                scratch_base_address + (q_lane_base + head) * mlen * mlen,
                physical_shape=(mlen, mlen),
            )
            for head in range(group_heads)
        ]
        pv = self.alloc("_packed_PV", mlen, head_slot_dim, strict=False, physical_shape=(mlen, mlen))
        optimized_schedule = (
            getattr(self, "packed_attention_schedule", "legacy")
            == "direct-first-block-v1"
        )
        optimized_direct_output = optimized_schedule and output_precleared
        sequence_plan = getattr(self, "_native_sequence_packing", None)
        if (
            sequence_plan is not None
            and num_q_blocks == 1
            and int(sequence_plan.attention_group_seq_len) == seq_len
            and int(sequence_plan.rows_per_attention_group)
            == Q_group.physical_shape[0]
        ):
            compact_active_output_ranges = tuple(
                (
                    slot * int(sequence_plan.batch_slot_rows),
                    slot * int(sequence_plan.batch_slot_rows)
                    + int(sequence_plan.seq_len),
                )
                for slot in range(int(sequence_plan.batch_pack_factor))
            )
        else:
            compact_active_output_ranges = None
        pack_scratch = None
        pack_scratch_addr = None
        pack_scratch_rows = 1
        if not direct_output or optimized_direct_output:
            requested_rows = (
                16
                if getattr(self, "gqa_pipeline_schedule", "row-serial")
                == "row-interleaved-v1"
                and optimized_direct_output
                else 1
            )
            for candidate in (16, 8, 4, 1):
                if candidate > requested_rows:
                    continue
                try:
                    pack_scratch = self.alloc(
                        "_packed_pack_scratch",
                        candidate,
                        mlen,
                        strict=False,
                        physical_shape=(candidate, mlen),
                    )
                    pack_scratch_rows = candidate
                    break
                except MemoryError:
                    continue
            if pack_scratch is None:
                raise MemoryError("unable to allocate even one packed-O scratch row")
            pack_scratch_addr = self.get_vram_addr(pack_scratch.name)

        for q_block in range(num_q_blocks):
            rows = min(mlen, seq_len - q_block * mlen)
            active_output_ranges = compact_active_output_ranges or ((0, rows),)
            active_output_rows = sum(
                end - start for start, end in active_output_ranges
            )
            if not output_precleared:
                self.emit(
                    self._reset_vram_asm(
                        start_address=output_base_address,
                        rows=rows,
                        cols=mlen,
                        total_rows=output_physical_rows,
                        mlen=mlen,
                        row_offset=q_block * mlen,
                    )
                )
            for head, s_head in enumerate(s_views):
                output_head = output_head_base + head
                output_col_block = (output_head * head_slot_dim) // mlen
                output_lane = (output_head * head_slot_dim) % mlen // head_slot_dim
                output_block_base = (
                    output_base_address
                    + output_col_block * output_physical_rows * mlen
                    + q_block * mlen * mlen
                )
                use_direct_output = direct_output or optimized_direct_output
                if use_direct_output:
                    o_head = self.alloc_at(
                        f"_packed_direct_O_q{q_block}_head{head}",
                        rows,
                        head_slot_dim,
                        output_block_base,
                        physical_shape=(output_physical_rows, mlen),
                    )
                else:
                    o_head = self.alloc(
                        f"_packed_O_q{q_block}_head{head}",
                        rows,
                        head_slot_dim,
                        strict=False,
                        physical_shape=(mlen, mlen),
                    )
                self.init_online_softmax(
                    0,
                    o_head,
                    rows=rows,
                    reset_state=not optimized_schedule,
                    # The first-block recurrence removes the zero-output
                    # rescale, but a fallback scratch O still needs a known
                    # zero value.  Only the direct path may rely on the
                    # caller's one-time O_full clear.
                    reset_output=not optimized_direct_output,
                )
                if optimized_schedule:
                    stats = self._packed_attention_stats
                    stats["softmax_state_initializations_elided"] += 1
                    stats["softmax_state_initialization_rows_elided"] += rows
                    if optimized_direct_output:
                        stats["temporary_o_matrices_elided"] += 1

                first_processed_k_block = True
                processed_k_blocks = tuple(
                    block
                    for block in range(num_k_blocks)
                    if causal_mask is None or block <= q_block
                )
                streamed_state = (
                    optimized_direct_output
                    and getattr(self, "softmax_state_schedule", "sram-v1")
                    == "streamed-v2"
                )
                double_buffered_kv = (
                    getattr(self, "gqa_pipeline_schedule", "row-serial")
                    == "row-interleaved-v1"
                    and resident_k_mram is None
                    and self.mram_tile_capacity >= 2
                )
                kv_slot_elems = mlen * mlen
                if double_buffered_kv and processed_k_blocks:
                    first_k_block = processed_k_blocks[0]
                    self._emit_packed_kv_prefetch(
                        K, ((k_idx + first_k_block, 0),)
                    )
                    self.record_gqa_pipeline_stats(
                        {
                            "gqa_kv_double_buffered": True,
                        }
                    )
                elif (
                    getattr(self, "gqa_pipeline_schedule", "row-serial")
                    == "row-interleaved-v1"
                    and resident_k_mram is None
                ):
                    self.record_gqa_pipeline_stats(
                        {
                            "gqa_pipeline_fallback_reason": (
                                "mram_tile_capacity_lt_2"
                            )
                        }
                    )
                for k_block in range(num_k_blocks):
                    if causal_mask is not None and k_block > q_block:
                        self.emit(
                            f"; Skip future packed-GQA K block q_block={q_block}, k_block={k_block}\n"
                        )
                        continue
                    block_cols = min(mlen, active_k_cols - k_block * mlen)
                    physical_k_idx = k_idx + k_block
                    if double_buffered_kv:
                        self._emit_packed_qkt_from_mram(
                            Q_group=Q_group,
                            q_idx=q_block,
                            k_mram_address=0,
                            s_base_address=scratch_base_address,
                        )
                    elif resident_k_mram is None:
                        self._emit_packed_qkt_to_s(
                            Q_group=Q_group,
                            K=K,
                            q_idx=q_block,
                            k_idx=physical_k_idx,
                            s_base_address=scratch_base_address,
                        )
                    else:
                        self._emit_packed_qkt_from_mram(
                            Q_group=Q_group,
                            q_idx=q_block,
                            k_mram_address=resident_k_mram[k_block],
                            s_base_address=scratch_base_address,
                        )
                    stats = self._packed_attention_stats
                    stats["qk_compute_count"] += 1
                    if head == 0:
                        stats["ideal_qk_compute_count"] += 1
                    if resident_k_mram is None:
                        stats["kv_tile_load_count"] += 2
                        if head == 0:
                            stats["ideal_kv_tile_load_count"] += 2

                    apply_causal_mask = causal_mask is not None and k_block == q_block
                    valid_col_mask = self._valid_col_mask_for_tile(
                        block_cols,
                        causal_mask=causal_mask,
                        apply_causal_mask=apply_causal_mask,
                        # A one-tile packed mask starts as all -inf and opens
                        # only real batch/token blocks.  The local triangular
                        # mask used by long-context diagonal tiles does not
                        # cover the final tile's padded columns.
                        causal_mask_covers_padding=(
                            seq_len <= mlen and active_k_cols <= mlen
                        ),
                    )
                    if valid_col_mask is not None:
                        self.vram_add(s_head, valid_col_mask, num_rows=rows)
                    if isinstance(causal_mask, VRAMMatrixVar) and apply_causal_mask:
                        self.vram_add(s_head, causal_mask, num_rows=rows)
                    elif causal_mask is True and apply_causal_mask:
                        self.emit("; NOTE: packed attention received causal_mask=True without a VRAM mask; no mask applied.\n")
                    softmax_valid_cols = (
                        None
                        if valid_col_mask is not None or (isinstance(causal_mask, VRAMMatrixVar) and apply_causal_mask)
                        else block_cols
                    )
                    if double_buffered_kv:
                        # Matrix SRAM slot 1 is independent of Vector SRAM, so
                        # the DMA can progress while the following softmax uses
                        # only the score tile in Vector SRAM.
                        self._emit_packed_kv_prefetch(
                            V, ((physical_k_idx, kv_slot_elems),)
                        )
                        self.record_gqa_pipeline_stats(
                            {"gqa_dma_overlap_eligible_occurrences": 1}
                        )
                    processed_position = processed_k_blocks.index(k_block)
                    last_processed_k_block = (
                        processed_position + 1 == len(processed_k_blocks)
                    )
                    if optimized_schedule and first_processed_k_block:
                        self.online_softmax_first_block(
                            s_head,
                            softmax_scale,
                            rows=rows,
                            valid_cols=softmax_valid_cols,
                            state_head=0,
                            last_block=last_processed_k_block,
                            stream_state=streamed_state,
                        )
                        stats["softmax_first_block_specialized_count"] += 1
                        stats["softmax_first_block_specialized_rows"] += rows
                        if streamed_state:
                            stats["softmax_m_moves_elided"] += rows
                            stats["softmax_l_moves_elided"] += rows
                            if last_processed_k_block:
                                stats["softmax_m_stores_elided"] += rows
                    else:
                        self.online_softmax_block(
                            s_head,
                            softmax_scale,
                            rows=rows,
                            valid_cols=softmax_valid_cols,
                            state_head=0,
                            output_address=(
                                output_block_base if streamed_state else None
                            ),
                            output_head_slot=(
                                output_lane if streamed_state else None
                            ),
                            last_block=last_processed_k_block,
                        )
                        if streamed_state:
                            stats["m_res_stores_elided"] += rows
                            stats["m_res_loads_elided"] += rows
                            stats["m_res_streamed_rows"] += rows
                            if last_processed_k_block:
                                stats["softmax_m_stores_elided"] += rows
                    if double_buffered_kv:
                        self._compute_pv_from_mram(
                            s_block=s_head,
                            pv_matrix=pv,
                            v_mram_address=kv_slot_elems,
                            head_slot_dim=head_slot_dim,
                            rows=rows,
                            preserve_full_tile_work=True,
                        )
                    elif resident_v_mram is None:
                        self.compute_pv(s_head, V, physical_k_idx, pv, head_slot_dim, rows=rows)
                    else:
                        self._compute_pv_from_mram(
                            s_block=s_head,
                            pv_matrix=pv,
                            v_mram_address=resident_v_mram[k_block],
                            head_slot_dim=head_slot_dim,
                            rows=rows,
                        )
                    stats["pv_compute_count"] += 1
                    if double_buffered_kv:
                        processed_position = processed_k_blocks.index(k_block)
                        if processed_position + 1 < len(processed_k_blocks):
                            next_k_block = processed_k_blocks[processed_position + 1]
                            self._emit_packed_kv_prefetch(
                                K, ((k_idx + next_k_block, 0),)
                            )
                            self.record_gqa_pipeline_stats(
                                {"gqa_dma_overlap_eligible_occurrences": 1}
                            )
                    if optimized_direct_output:
                        if not first_processed_k_block and not streamed_state:
                            self._scale_packed_o_lane(
                                output_base_address=output_block_base,
                                head_slot=output_lane,
                                rows=rows,
                                row_ranges=active_output_ranges,
                            )
                        self._add_pv_to_packed_o_lane(
                            output_base_address=output_block_base,
                            pv=pv,
                            head_slot=output_lane,
                            head_slot_dim=head_slot_dim,
                            rows=rows,
                            scratch_address=pack_scratch_addr,
                            scratch_rows=pack_scratch_rows,
                            row_ranges=active_output_ranges,
                        )
                        stats["direct_o_lane_updates"] += active_output_rows
                    else:
                        if not optimized_schedule or not first_processed_k_block:
                            self.scale_o_row(o_head, 0, rows=rows)
                        self.vram_add(o_head, pv, num_rows=rows)
                    first_processed_k_block = False

                if optimized_direct_output:
                    self._final_scale_packed_o_lane(
                        output_base_address=output_block_base,
                        head_slot=output_lane,
                        rows=rows,
                        row_ranges=active_output_ranges,
                    )
                else:
                    self.final_scale_o(0, o_head, rows=rows)
                if not use_direct_output:
                    self._pack_o_head_to_output(
                        o_head=o_head,
                        output_base_address=output_block_base,
                        output_physical_rows=output_physical_rows,
                        head_slot=output_lane,
                        head_slot_dim=head_slot_dim,
                        rows=rows,
                        scratch_address=pack_scratch_addr,
                    )
                self.free_tensor(o_head)

        self.free_tensor(pv)
        if pack_scratch is not None:
            self.free_tensor(pack_scratch)

    def flash_attention_packed_gqa(
        self,
        Q_full: VRAMMatrixVar,
        O_full: VRAMMatrixVar,
        kv_pairs: list[tuple[InputVar, InputVar]],
        *,
        batch_size: int,
        seq_len: int,
        kv_seq_len: int | None = None,
        rows_per_batch: int,
        gqa_ratio: int,
        physical_broadcast: int,
        head_slot_dim: int,
        scratch_base_address: int,
        groups_per_storage_block: int = 1,
        attention_group_width: int | None = None,
        storage_block_count: int | None = None,
        scale: float | None = None,
        causal_mask: bool | VRAMMatrixVar | None = True,
    ) -> PackedGQASchedule:
        """Canonical packed-GQA lowering scheduled by logical KV group.

        Q/O storage blocks may contain several groups. Since Vector SRAM reads
        are MLEN aligned, the current KV head is replicated across all lanes in
        that block; ``q_lane_base`` selects the group whose scores and output
        are retained. No operation broadcasts different KV heads together.
        """
        if kv_seq_len is None:
            kv_seq_len = seq_len
        schedule = PackedGQASchedule.build(
            batch_size=batch_size,
            seq_len=seq_len,
            kv_seq_len=kv_seq_len,
            rows_per_batch=rows_per_batch,
            num_kv_heads=len(kv_pairs),
            gqa_ratio=gqa_ratio,
            physical_broadcast=physical_broadcast,
            mlen=self.mlen,
            mram_tile_capacity=self.mram_tile_capacity,
            kv_residency_policy=self.kv_residency_policy,
        )
        full_q_tiles, q_tail_rows = divmod(seq_len, self.mlen)
        self._packed_attention_stats["full_q_tiles"] += full_q_tiles
        if q_tail_rows:
            self._packed_attention_stats["q_tail_rows"] = max(
                self._packed_attention_stats["q_tail_rows"], q_tail_rows
            )
            q_groups = len(kv_pairs) * schedule.chunks_per_kv
            tail_k_blocks = schedule.q_blocks if causal_mask is not None else schedule.k_blocks
            head_factor = (
                1
                if getattr(self, "packed_qk_schedule", "head-major-v1")
                == "broadcast-k-major-v1"
                else physical_broadcast
            )
            self._packed_attention_stats["tail_bmm_occurrences"] += (
                batch_size * q_groups * tail_k_blocks * head_factor
            )
        if not hasattr(self, "_packed_gqa_looped_batch"):
            self._packed_gqa_looped_batch = False
            self._packed_gqa_looped_kv_heads = False
            self._packed_gqa_looped_full_chunks = False
        if not kv_pairs:
            raise ValueError("kv_pairs must not be empty")
        if self.mlen % self.blen != 0:
            raise ValueError(f"Packed GQA requires MLEN % BLEN == 0, got {self.mlen} % {self.blen}")
        if head_slot_dim > self.mlen:
            raise ValueError(
                f"Packed GQA head_slot_dim={head_slot_dim} exceeds MLEN={self.mlen}"
            )
        if physical_broadcast * head_slot_dim > self.mlen:
            raise ValueError(
                "physical_broadcast*head_slot_dim must fit MLEN "
                f"({physical_broadcast}*{head_slot_dim} > {self.mlen})"
            )
        self.configure_softmax_state_layout(physical_broadcast)
        if attention_group_width is None:
            attention_group_width = physical_broadcast * head_slot_dim
        if attention_group_width < physical_broadcast * head_slot_dim:
            raise ValueError(
                "attention_group_width cannot be smaller than the active packed heads "
                f"({attention_group_width} < {physical_broadcast * head_slot_dim})"
            )
        if groups_per_storage_block <= 0:
            raise ValueError(
                f"groups_per_storage_block must be positive, got {groups_per_storage_block}"
            )
        if groups_per_storage_block * attention_group_width > self.mlen:
            raise ValueError(
                "packed attention groups do not fit one MLEN storage block: "
                f"{groups_per_storage_block}*{attention_group_width} > {self.mlen}"
            )
        if rows_per_batch % self.mlen != 0:
            raise ValueError(
                f"rows_per_batch={rows_per_batch} must be a multiple of MLEN={self.mlen}"
            )
        if Q_full.physical_shape[0] < batch_size * rows_per_batch:
            raise ValueError(
                f"Q_full physical rows {Q_full.physical_shape[0]} cannot cover "
                f"batch_size*rows_per_batch={batch_size * rows_per_batch}"
            )
        if O_full.physical_shape[0] < batch_size * rows_per_batch:
            raise ValueError(
                f"O_full physical rows {O_full.physical_shape[0]} cannot cover "
                f"batch_size*rows_per_batch={batch_size * rows_per_batch}"
            )

        group_count = len(kv_pairs) * schedule.chunks_per_kv
        expected_storage_blocks = math.ceil(group_count / groups_per_storage_block)
        if storage_block_count is None:
            storage_block_count = expected_storage_blocks
        if storage_block_count != expected_storage_blocks:
            raise ValueError(
                f"storage_block_count={storage_block_count} does not match packed group "
                f"geometry ({expected_storage_blocks})"
            )
        required_width = storage_block_count * self.mlen
        hardware_broadcast = groups_per_storage_block * physical_broadcast
        if hardware_broadcast * head_slot_dim > self.mlen:
            raise ValueError(
                "storage-block broadcast does not fit MLEN: "
                f"{hardware_broadcast}*{head_slot_dim} > {self.mlen}"
            )
        if getattr(self, "broadcast_amount", hardware_broadcast) != hardware_broadcast:
            raise ValueError(
                "compiler BROADCAST_AMOUNT must match all packed head lanes in one "
                f"storage block ({self.broadcast_amount} != {hardware_broadcast})"
            )
        if Q_full.physical_shape[1] < required_width:
            raise ValueError(
                f"Q_full physical width {Q_full.physical_shape[1]} cannot hold "
                f"{group_count} packed groups ({required_width} columns)"
            )
        if O_full.physical_shape[1] < required_width:
            raise ValueError(
                f"O_full physical width {O_full.physical_shape[1]} cannot hold "
                f"{group_count} packed groups ({required_width} columns)"
            )
        if scale is None:
            scale = 1.0 / math.sqrt(head_slot_dim)

        q_base = self.get_vram_addr(Q_full.name)
        o_base = self.get_vram_addr(O_full.name)
        total_physical_rows = Q_full.physical_shape[0]
        q_group_stride = total_physical_rows * self.mlen
        o_group_stride = O_full.physical_shape[0] * self.mlen
        batch_row_stride = rows_per_batch * self.mlen
        row_blocks_per_batch = rows_per_batch // self.mlen

        def group_offset(group_idx: int, physical_rows: int) -> int:
            block_idx, slot_idx = divmod(group_idx, groups_per_storage_block)
            return (
                block_idx * physical_rows * self.mlen
                + slot_idx * attention_group_width
            )

        self.emit(
            "; === Canonical packed GQA schedule: logical KV groups, "
            f"resident_prefix={schedule.kv_residency_plan.resident_prefix_blocks}/"
            f"{schedule.k_blocks} ===\n"
        )
        for batch_idx in range(batch_size):
            first_k_idx = batch_idx * row_blocks_per_batch
            batch_offset = batch_idx * batch_row_stride
            for kv_head, (K, V) in enumerate(kv_pairs):
                resident_k = resident_v = None
                requested_residency = schedule.kv_residency_plan
                use_prefix_plan = (
                    getattr(self, "packed_qk_schedule", "head-major-v1")
                    == "broadcast-k-major-v1"
                    or requested_residency.full_resident
                )
                residency = (
                    requested_residency
                    if use_prefix_plan
                    else plan_kv_residency(
                        k_blocks=schedule.k_blocks,
                        mlen=self.mlen,
                        matrix_sram_tiles=self.mram_tile_capacity,
                        policy="head-major-streaming-compat",
                        force_streaming=True,
                    )
                )
                if use_prefix_plan and residency.resident_prefix_blocks:
                    self._prefetch_packed_kv_prefix(
                        K,
                        V,
                        first_k_idx=first_k_idx,
                        residency=residency,
                    )
                    resident_tile_loads = 2 * residency.resident_prefix_blocks
                    self._packed_attention_stats["kv_tile_load_count"] += (
                        resident_tile_loads
                    )
                    self._packed_attention_stats["kv_cache_misses"] += (
                        resident_tile_loads
                    )
                if use_prefix_plan:
                    self._packed_attention_stats[
                        "ideal_kv_tile_load_count"
                    ] += 2 * schedule.k_blocks
                self._kv_residency_plan_metadata = residency.metadata(
                    q_blocks=schedule.q_blocks,
                    causal=causal_mask is not None,
                )
                if residency.full_resident:
                    resident_k = residency.resident_k_addresses
                    resident_v = residency.resident_v_addresses
                if (
                    residency.full_resident
                    and physical_broadcast == 1
                    and groups_per_storage_block == 1
                    and q_group_stride == o_group_stride
                ):
                    first_group = kv_head * schedule.chunks_per_kv
                    self._emit_resident_single_head_chunks_looped(
                        q_base_address=(
                            q_base + first_group * q_group_stride + batch_offset
                        ),
                        o_base_address=(
                            o_base + first_group * o_group_stride + batch_offset
                        ),
                        group_stride=q_group_stride,
                        chunk_count=schedule.chunks_per_kv,
                        seq_len=seq_len,
                        kv_seq_len=kv_seq_len,
                        head_slot_dim=head_slot_dim,
                        scratch_base_address=scratch_base_address,
                        scale=scale,
                        causal_mask=causal_mask,
                        resident_k_mram=resident_k,
                        resident_v_mram=resident_v,
                    )
                    self._packed_gqa_looped_full_chunks = (
                        self._packed_gqa_looped_full_chunks
                        or schedule.chunks_per_kv > 1
                    )
                    continue
                for chunk in range(schedule.chunks_per_kv):
                    chunk_start = chunk * physical_broadcast
                    group_heads = min(physical_broadcast, gqa_ratio - chunk_start)
                    if group_heads <= 0:
                        continue
                    group_idx = kv_head * schedule.chunks_per_kv + chunk
                    block_idx, slot_idx = divmod(
                        group_idx, groups_per_storage_block
                    )
                    q_lane_base = slot_idx * physical_broadcast
                    q_group = self.alloc_at(
                        f"_canonical_Q_b{batch_idx}_kv{kv_head}_c{chunk}",
                        seq_len,
                        self.mlen,
                        q_base
                        + block_idx * total_physical_rows * self.mlen
                        + batch_offset,
                        physical_shape=(rows_per_batch, self.mlen),
                    )
                    self._emit_packed_attention_group_internal(
                        Q_group=q_group,
                        K=K,
                        V=V,
                        group_heads=group_heads,
                        head_slot_dim=head_slot_dim,
                        output_base_address=(
                            o_base
                            + block_idx * O_full.physical_shape[0] * self.mlen
                            + batch_offset
                        ),
                        output_physical_rows=rows_per_batch,
                        scratch_base_address=scratch_base_address,
                        broadcast_amount=hardware_broadcast,
                        scale=scale,
                        causal_mask=causal_mask,
                        k_idx=first_k_idx,
                        valid_cols=kv_seq_len,
                        resident_k_mram=(
                            None
                            if getattr(
                                self, "packed_qk_schedule", "head-major-v1"
                            )
                            == "broadcast-k-major-v1"
                            else resident_k
                        ),
                        resident_v_mram=(
                            None
                            if getattr(
                                self, "packed_qk_schedule", "head-major-v1"
                            )
                            == "broadcast-k-major-v1"
                            else resident_v
                        ),
                        kv_residency_plan=(
                            residency
                            if getattr(
                                self, "packed_qk_schedule", "head-major-v1"
                            )
                            == "broadcast-k-major-v1"
                            else None
                        ),
                        output_precleared=True,
                        direct_output=hardware_broadcast == 1,
                        output_head_base=q_lane_base,
                        q_lane_base=q_lane_base,
                    )
                    self.free_tensor(q_group)

        return schedule

    def _emit_packed_attention_groups_tiled_looped(
        self,
        *,
        Q_full: VRAMMatrixVar,
        kv_pairs: list[tuple[InputVar, InputVar]],
        group_heads: int,
        head_slot_dim: int,
        output_base_address: int,
        scratch_base_address: int,
        broadcast_amount: int,
        scale: float,
        causal_mask: bool | VRAMMatrixVar | None,
        q_base_address: int | None = None,
        group_stride_blocks: int = 1,
    ) -> None:
        """Emit sequence-tiled packed GQA with a hardware loop over KV heads."""
        seq_len, _q_width = Q_full.shape
        mlen = self.mlen
        num_kv_heads = len(kv_pairs)
        q_physical_rows, _q_physical_cols = Q_full.physical_shape
        num_q_blocks = math.ceil(seq_len / mlen)
        num_k_blocks = math.ceil(seq_len / mlen)
        if group_stride_blocks <= 0:
            raise ValueError(f"group_stride_blocks must be positive, got {group_stride_blocks}")
        q_base_address = self.get_vram_addr(Q_full.name) if q_base_address is None else q_base_address
        q_group_stride = q_physical_rows * mlen * group_stride_blocks
        o_group_stride = q_physical_rows * mlen * group_stride_blocks
        block_stride = mlen * mlen
        softmax_scale = scale / 0.25

        for K, V in kv_pairs:
            self._ensure_hbm_sub_matrix_registered(K)
            self._ensure_hbm_sub_matrix_registered(V)
            if K.physical_shape != kv_pairs[0][0].physical_shape:
                raise ValueError("all looped K heads must share physical_shape")
            if V.physical_shape != kv_pairs[0][1].physical_shape:
                raise ValueError("all looped V heads must share physical_shape")

        if num_kv_heads > 1:
            k_stride = kv_pairs[1][0].hbm_addr - kv_pairs[0][0].hbm_addr
            v_stride = kv_pairs[1][1].hbm_addr - kv_pairs[0][1].hbm_addr
            for idx in range(1, num_kv_heads):
                if kv_pairs[idx][0].hbm_addr != kv_pairs[0][0].hbm_addr + idx * k_stride:
                    raise ValueError("K head HBM bases are not affine; cannot roll KV groups")
                if kv_pairs[idx][1].hbm_addr != kv_pairs[0][1].hbm_addr + idx * v_stride:
                    raise ValueError("V head HBM bases are not affine; cannot roll KV groups")
        else:
            k_stride = 0
            v_stride = 0

        k_layout = self.get_hbm_layout(kv_pairs[0][0].name)
        s_views = [
            self.alloc_at(
                f"_packed_tiled_loop_S_h{head}",
                mlen,
                mlen,
                scratch_base_address + head * mlen * mlen,
                physical_shape=(mlen, mlen),
            )
            for head in range(group_heads)
        ]
        pv = self.alloc("_packed_tiled_loop_PV", mlen, head_slot_dim, strict=False, physical_shape=(mlen, mlen))
        pack_scratch = self.alloc("_packed_tiled_loop_pack_scratch", 1, mlen, strict=False, physical_shape=(1, mlen))
        pack_scratch_addr = self.get_vram_addr(pack_scratch.name)

        gp_q, gp_o, gp_k, gp_v, gp_tmp, gp_kv_loop = self.register_allocator.allocate_gp(6)
        k_addr_reg, v_addr_reg = self.register_allocator.allocate_addr(2)
        try:
            for q_block in range(num_q_blocks):
                rows = min(mlen, seq_len - q_block * mlen)
                q_block_offset = q_block * block_stride

                for head, s_head in enumerate(s_views):
                    o_head = self.alloc(
                        f"_packed_tiled_loop_O_q{q_block}_head{head}",
                        rows,
                        head_slot_dim,
                        strict=False,
                        physical_shape=(mlen, mlen),
                    )
                    setup_lines = [
                        f"; === Sequence-tiled packed GQA loop over KV groups (q_block={q_block}, head={head}) ===",
                        *load_large_int(gp_q, q_base_address),
                        *add_large_int(gp_q, gp_q, q_block_offset, temp_reg=gp_tmp),
                        *load_large_int(gp_o, output_base_address),
                        *add_large_int(gp_o, gp_o, q_block_offset, temp_reg=gp_tmp),
                        *load_large_int(gp_k, kv_pairs[0][0].hbm_addr),
                        *load_large_int(gp_v, kv_pairs[0][1].hbm_addr),
                        f"C_LOOP_START gp{gp_kv_loop}, {num_kv_heads}",
                        f"C_SET_ADDR_REG a{k_addr_reg}, gp0, gp{gp_k}",
                        f"C_SET_ADDR_REG a{v_addr_reg}, gp0, gp{gp_v}",
                    ]
                    self.emit("\n".join(setup_lines) + "\n")
                    if head == 0:
                        self._reset_vram_from_gp(base_gp=gp_o, rows=rows)
                    self.init_online_softmax(0, o_head, rows=rows)

                    for k_block in range(num_k_blocks):
                        if causal_mask is not None and k_block > q_block:
                            self.emit(
                                f"; Skip future packed-GQA K block q_block={q_block}, k_block={k_block}\n"
                            )
                            continue
                        block_cols = min(mlen, seq_len - k_block * mlen)
                        k_block_offset = k_block * block_stride
                        self._emit_packed_qkt_to_s_dynamic(
                            q_base_gp=gp_q,
                            k_hbm_addr_reg=k_addr_reg,
                            s_base_address=scratch_base_address,
                            k_layout=k_layout,
                            k_hbm_offset=k_block_offset,
                        )

                        apply_causal_mask = causal_mask is not None and k_block == q_block
                        valid_col_mask = self._valid_col_mask_for_tile(
                            block_cols,
                            causal_mask=causal_mask,
                            apply_causal_mask=apply_causal_mask,
                            causal_mask_covers_padding=False,
                        )
                        if valid_col_mask is not None:
                            self.vram_add(s_head, valid_col_mask, num_rows=rows)
                        if isinstance(causal_mask, VRAMMatrixVar) and apply_causal_mask:
                            self.vram_add(s_head, causal_mask, num_rows=rows)
                        elif causal_mask is True and apply_causal_mask:
                            self.emit("; NOTE: packed attention received causal_mask=True without a VRAM mask; no mask applied.\n")
                        softmax_valid_cols = (
                            None
                            if valid_col_mask is not None or (isinstance(causal_mask, VRAMMatrixVar) and apply_causal_mask)
                            else block_cols
                        )
                        self.online_softmax_block(s_head, softmax_scale, rows=rows, valid_cols=softmax_valid_cols)
                        self.emit(
                            self._pv_multiply_asm(
                                mlen=mlen,
                                blen=self.blen,
                                head_dim=head_slot_dim,
                                p_address=self.get_vram_addr(s_head.name),
                                v_hbm_offset_reg=v_addr_reg,
                                v_hbm_offset=k_block_offset,
                                pv_address=self.get_vram_addr(pv.name),
                                rows=rows,
                            )
                        )
                        self.scale_o_row(o_head, 0, rows=rows)
                        self.vram_add(o_head, pv, num_rows=rows)

                    self.final_scale_o(0, o_head, rows=rows)
                    self._pack_o_head_to_output_dynamic(
                        o_head=o_head,
                        output_base_gp=gp_o,
                        head_slot=head,
                        head_slot_dim=head_slot_dim,
                        rows=rows,
                        scratch_address=pack_scratch_addr,
                    )
                    self.free_tensor(o_head)

                    update_lines = []
                    update_lines.extend(add_large_int(gp_q, gp_q, q_group_stride, temp_reg=gp_tmp))
                    update_lines.extend(add_large_int(gp_o, gp_o, o_group_stride, temp_reg=gp_tmp))
                    update_lines.extend(add_large_int(gp_k, gp_k, k_stride, temp_reg=gp_tmp))
                    update_lines.extend(add_large_int(gp_v, gp_v, v_stride, temp_reg=gp_tmp))
                    update_lines.append(f"C_LOOP_END gp{gp_kv_loop}")
                    self.emit("\n".join(update_lines) + "\n")
        finally:
            self.register_allocator.free_addr([k_addr_reg, v_addr_reg])
            self.register_allocator.free_gp([gp_q, gp_o, gp_k, gp_v, gp_tmp, gp_kv_loop])
            self.free_tensor(pv)
            self.free_tensor(pack_scratch)

    def flash_attention_packed_groups_looped(
        self,
        Q_full: VRAMMatrixVar,
        kv_pairs: list[tuple[InputVar, InputVar]],
        *,
        group_heads: int,
        head_slot_dim: int,
        output_base_address: int,
        scratch_base_address: int,
        broadcast_amount: int,
        scale=None,
        causal_mask: bool | VRAMMatrixVar | None = True,
        q_base_address: int | None = None,
        group_stride_blocks: int = 1,
    ) -> None:
        """Emit packed GQA attention with one hardware loop over KV groups.

        This rolls only the attention core. Q/K/V projection and RoPE lowering
        still happen before this call, but the repeated QK/softmax/PV/O body is
        emitted once and parameterized by loop-carried Q/O VRAM and K/V HBM
        pointers.
        """
        if not kv_pairs:
            raise ValueError("kv_pairs must not be empty")
        seq_len, _q_width = Q_full.shape
        mlen = self.mlen
        num_kv_heads = len(kv_pairs)
        q_physical_rows, q_physical_cols = Q_full.physical_shape
        if group_stride_blocks <= 0:
            raise ValueError(f"group_stride_blocks must be positive, got {group_stride_blocks}")
        if q_physical_cols < num_kv_heads * mlen:
            raise ValueError(
                f"Q_full physical cols {q_physical_cols} cannot hold {num_kv_heads} MLEN-wide groups"
            )
        q_base_addr = self.get_vram_addr(Q_full.name) if q_base_address is None else q_base_address
        if seq_len > mlen:
            if os.environ.get("PLENA_PACKED_GQA_PYTHON_EXPAND") != "1":
                try:
                    self._emit_packed_attention_groups_tiled_looped(
                        Q_full=Q_full,
                        kv_pairs=kv_pairs,
                        group_heads=group_heads,
                        head_slot_dim=head_slot_dim,
                        output_base_address=output_base_address,
                        scratch_base_address=scratch_base_address,
                        broadcast_amount=broadcast_amount,
                        scale=scale or (1.0 / math.sqrt(head_slot_dim)),
                        causal_mask=causal_mask,
                        q_base_address=q_base_addr,
                        group_stride_blocks=group_stride_blocks,
                    )
                    return
                except ValueError as exc:
                    self.emit(
                        "; NOTE: sequence-tiled packed GQA hardware loop unavailable "
                        f"({exc}); falling back to Python-expanded KV groups.\n"
                    )
            self.emit(
                "; NOTE: seq_len exceeds MLEN; using packed GQA sequence-tiled codegen "
                "with Python-expanded KV groups.\n"
            )
            q_group_stride = q_physical_rows * mlen * group_stride_blocks
            o_group_stride = q_physical_rows * mlen * group_stride_blocks
            for kv_head, (K, V) in enumerate(kv_pairs):
                Q_group = self.alloc_at(
                    f"_packed_loop_tiled_Q_group{kv_head}",
                    seq_len,
                    mlen,
                    q_base_addr + kv_head * q_group_stride,
                    physical_shape=(q_physical_rows, mlen),
                )
                self._emit_packed_attention_group_internal(
                    Q_group=Q_group,
                    K=K,
                    V=V,
                    group_heads=group_heads,
                    head_slot_dim=head_slot_dim,
                    output_base_address=output_base_address + kv_head * o_group_stride,
                    output_physical_rows=q_physical_rows,
                    scratch_base_address=scratch_base_address,
                    broadcast_amount=broadcast_amount,
                    scale=scale or (1.0 / math.sqrt(head_slot_dim)),
                    causal_mask=causal_mask,
                )
            return
        if group_heads > broadcast_amount:
            raise ValueError(f"group_heads={group_heads} exceeds broadcast_amount={broadcast_amount}")
        if broadcast_amount * head_slot_dim > mlen:
            raise ValueError(
                f"broadcast_amount*head_slot_dim must fit MLEN "
                f"({broadcast_amount}*{head_slot_dim} > {mlen})"
            )
        if getattr(self, "hlen", head_slot_dim) != head_slot_dim:
            raise ValueError(f"Packed attention requires HLEN={head_slot_dim}, got {self.hlen}")
        if scale is None:
            scale = 1.0 / math.sqrt(head_slot_dim)

        for K, V in kv_pairs:
            self._ensure_hbm_sub_matrix_registered(K)
            self._ensure_hbm_sub_matrix_registered(V)
            if K.physical_shape != kv_pairs[0][0].physical_shape:
                raise ValueError("all looped K heads must share physical_shape")
            if V.physical_shape != kv_pairs[0][1].physical_shape:
                raise ValueError("all looped V heads must share physical_shape")

        if num_kv_heads > 1:
            k_stride = kv_pairs[1][0].hbm_addr - kv_pairs[0][0].hbm_addr
            v_stride = kv_pairs[1][1].hbm_addr - kv_pairs[0][1].hbm_addr
            for idx in range(1, num_kv_heads):
                if kv_pairs[idx][0].hbm_addr != kv_pairs[0][0].hbm_addr + idx * k_stride:
                    raise ValueError("K head HBM bases are not affine; cannot roll KV groups")
                if kv_pairs[idx][1].hbm_addr != kv_pairs[0][1].hbm_addr + idx * v_stride:
                    raise ValueError("V head HBM bases are not affine; cannot roll KV groups")
        else:
            k_stride = 0
            v_stride = 0

        q_group_stride = q_physical_rows * mlen * group_stride_blocks
        o_group_stride = q_physical_rows * mlen * group_stride_blocks
        rows = min(mlen, seq_len)
        softmax_scale = scale / 0.25
        k_layout = self.get_hbm_layout(kv_pairs[0][0].name)

        s_views = [
            self.alloc_at(
                f"_packed_loop_S_h{head}",
                mlen,
                mlen,
                scratch_base_address + head * mlen * mlen,
                physical_shape=(mlen, mlen),
            )
            for head in range(group_heads)
        ]
        pv = self.alloc("_packed_loop_PV", rows, head_slot_dim, strict=False, physical_shape=(mlen, mlen))
        pack_scratch = self.alloc("_packed_loop_pack_scratch", 1, mlen, strict=False, physical_shape=(1, mlen))
        pack_scratch_addr = self.get_vram_addr(pack_scratch.name)
        valid_col_mask = self._valid_col_mask_for_tile(
            seq_len,
            causal_mask=causal_mask,
            apply_causal_mask=causal_mask is not None,
            causal_mask_covers_padding=seq_len <= mlen,
        )

        gp_q, gp_o, gp_k, gp_v, gp_tmp, gp_kv_loop = self.register_allocator.allocate_gp(6)
        k_addr_reg, v_addr_reg = self.register_allocator.allocate_addr(2)
        try:
            setup_lines = [
                "; === Packed GQA attention core loop over KV groups ===",
                *load_large_int(gp_q, q_base_addr),
                *load_large_int(gp_o, output_base_address),
                *load_large_int(gp_k, kv_pairs[0][0].hbm_addr),
                *load_large_int(gp_v, kv_pairs[0][1].hbm_addr),
                f"C_LOOP_START gp{gp_kv_loop}, {num_kv_heads}",
                f"C_SET_ADDR_REG a{k_addr_reg}, gp0, gp{gp_k}",
                f"C_SET_ADDR_REG a{v_addr_reg}, gp0, gp{gp_v}",
            ]
            self.emit("\n".join(setup_lines) + "\n")

            self._reset_vram_from_gp(base_gp=gp_o, rows=rows)
            self._emit_packed_qkt_to_s_dynamic(
                q_base_gp=gp_q,
                k_hbm_addr_reg=k_addr_reg,
                s_base_address=scratch_base_address,
                k_layout=k_layout,
            )

            for head, s_head in enumerate(s_views):
                o_head = self.alloc(
                    f"_packed_loop_O_head{head}",
                    rows,
                    head_slot_dim,
                    strict=False,
                    physical_shape=(mlen, mlen),
                )
                self.init_online_softmax(0, o_head, rows=rows)
                if valid_col_mask is not None:
                    self.vram_add(s_head, valid_col_mask, num_rows=rows)
                if isinstance(causal_mask, VRAMMatrixVar):
                    self.vram_add(s_head, causal_mask, num_rows=rows)
                elif causal_mask is True:
                    self.emit("; NOTE: packed attention received causal_mask=True without a VRAM mask; no mask applied.\n")
                softmax_valid_cols = (
                    None
                    if valid_col_mask is not None or isinstance(causal_mask, VRAMMatrixVar)
                    else seq_len
                )
                self.online_softmax_block(s_head, softmax_scale, rows=rows, valid_cols=softmax_valid_cols)
                self.emit(
                    self._pv_multiply_asm(
                        mlen=mlen,
                        blen=self.blen,
                        head_dim=head_slot_dim,
                        p_address=self.get_vram_addr(s_head.name),
                        v_hbm_offset_reg=v_addr_reg,
                        v_hbm_offset=0,
                        pv_address=self.get_vram_addr(pv.name),
                        rows=rows,
                    )
                )
                self.scale_o_row(o_head, 0, rows=rows)
                self.vram_add(o_head, pv, num_rows=rows)
                self.final_scale_o(0, o_head, rows=rows)
                self._pack_o_head_to_output_dynamic(
                    o_head=o_head,
                    output_base_gp=gp_o,
                    head_slot=head,
                    head_slot_dim=head_slot_dim,
                    rows=rows,
                    scratch_address=pack_scratch_addr,
                )
                self.free_tensor(o_head)

            update_lines = []
            update_lines.extend(add_large_int(gp_q, gp_q, q_group_stride, temp_reg=gp_tmp))
            update_lines.extend(add_large_int(gp_o, gp_o, o_group_stride, temp_reg=gp_tmp))
            update_lines.extend(add_large_int(gp_k, gp_k, k_stride, temp_reg=gp_tmp))
            update_lines.extend(add_large_int(gp_v, gp_v, v_stride, temp_reg=gp_tmp))
            update_lines.append(f"C_LOOP_END gp{gp_kv_loop}")
            self.emit("\n".join(update_lines) + "\n")
        finally:
            self.register_allocator.free_addr([k_addr_reg, v_addr_reg])
            self.register_allocator.free_gp([gp_q, gp_o, gp_k, gp_v, gp_tmp, gp_kv_loop])
            self.free_tensor(pv)
            self.free_tensor(pack_scratch)

    def flash_attention_packed_group(
        self,
        Q_group: VRAMMatrixVar,
        K: InputVar,
        V: InputVar,
        *,
        group_heads: int,
        head_slot_dim: int,
        output_base_address: int,
        scratch_base_address: int,
        broadcast_amount: int,
        scale=None,
        causal_mask: bool | VRAMMatrixVar | None = True,
        k_idx: int = 0,
        valid_cols: int | None = None,
    ) -> None:
        """Emit one KV group's packed-head flash-attention body.

        Q_group and the output use an MLEN-wide row where active Q heads occupy
        HLEN-sized lanes. K/V are one KV head stored as MLEN-padded HBM rows.
        """
        seq_len, q_width = Q_group.shape
        mlen = self.mlen
        if q_width != mlen:
            raise ValueError(f"packed Q group must be one MLEN row wide, got {q_width}")
        if group_heads > broadcast_amount:
            raise ValueError(
                f"group_heads={group_heads} exceeds broadcast_amount={broadcast_amount}"
            )
        if broadcast_amount * head_slot_dim > mlen:
            raise ValueError(
                f"broadcast_amount*head_slot_dim must fit MLEN "
                f"({broadcast_amount}*{head_slot_dim} > {mlen})"
            )
        if scale is None:
            scale = 1.0 / math.sqrt(head_slot_dim)

        self._emit_packed_attention_group_internal(
            Q_group=Q_group,
            K=K,
            V=V,
            group_heads=group_heads,
            head_slot_dim=head_slot_dim,
            output_base_address=output_base_address,
            output_physical_rows=Q_group.physical_shape[0],
            scratch_base_address=scratch_base_address,
            broadcast_amount=broadcast_amount,
            scale=scale,
            causal_mask=causal_mask,
            k_idx=k_idx,
            valid_cols=valid_cols,
        )

    def init_online_softmax(
        self,
        q_idx: int,
        o_matrix: VRAMMatrixVar,
        rows: int | None = None,
        *,
        reset_state: bool = True,
        reset_output: bool = True,
    ):
        """Initialize Online Softmax state: m=-inf, l=0, O_row=0"""
        o_info = super().get_tensor_info(o_matrix.name)
        seq_len, head_dim = o_info.shape

        super().init_online_softmax(
            q_idx=q_idx,
            o_matrix=o_matrix.name,
            seq_len=seq_len,
            head_dim=head_dim,
            rows=rows,
            reset_state=reset_state,
            reset_output=reset_output,
        )

    def online_softmax_first_block(
        self,
        s_block: VRAMMatrixVar,
        scale: float,
        rows: int | None = None,
        valid_cols: int | None = None,
        state_head: int = 0,
        last_block: bool = False,
        stream_state: bool = False,
    ):
        """Initialize online-softmax state directly from the first K block."""
        super().online_softmax_first_block(
            s_block_matrix=s_block.name,
            scale=scale,
            rows=rows,
            valid_cols=valid_cols,
            state_head=state_head,
            last_block=last_block,
            stream_state=stream_state,
        )

    def online_softmax_block(
        self,
        s_block: VRAMMatrixVar,
        scale: float,
        rows: int | None = None,
        valid_cols: int | None = None,
        state_head: int = 0,
        output_address: int | None = None,
        output_head_slot: int | None = None,
        last_block: bool = False,
    ):
        """Perform Online Softmax on S block"""
        super().online_softmax_block(
            s_block_matrix=s_block.name,
            scale=scale,
            rows=rows,
            valid_cols=valid_cols,
            state_head=state_head,
            output_address=output_address,
            output_head_slot=output_head_slot,
            last_block=last_block,
        )

    def compute_pv(
        self,
        s_block: VRAMMatrixVar,
        v_input: InputVar,
        k_idx: int,
        pv_matrix: VRAMMatrixVar,
        head_dim: int,
        rows: int | None = None,
    ):
        """Compute PV = P @ V[k_idx] where P is stored in s_block."""
        if not isinstance(s_block, VRAMMatrixVar):
            raise TypeError(f"s_block must be VRAMMatrixVar, got {type(s_block)}")
        if not isinstance(v_input, InputVar):
            raise TypeError(f"v_input must be InputVar, got {type(v_input)}")
        if not isinstance(pv_matrix, VRAMMatrixVar):
            raise TypeError(f"pv_matrix must be VRAMMatrixVar, got {type(pv_matrix)}")

        self._ensure_hbm_sub_matrix_registered(v_input)
        super().compute_pv(
            s_block_matrix=s_block.name,
            v_sub_matrix=v_input.name,
            k_idx=k_idx,
            pv_matrix=pv_matrix.name,
            head_dim=head_dim,
            rows=rows,
        )
        if getattr(self, "_cost_sink", None) is not None:
            layout = self.get_hbm_layout(v_input.name)
            physical_rows, physical_head_dim = layout.physical_shape or layout.full_shape
            col_blocks = max(1, math.ceil(head_dim / self.mlen))
            element_offset = k_idx * self.mlen * physical_head_dim
            axes = (
                RepeatAxis(
                    "v_column_tile",
                    col_blocks,
                    element_base_delta=self.mlen,
                    scale_base_delta=self.mlen // 8,
                ),
            ) if col_blocks > 1 else ()
            self.record_dma_stream(
                self.make_exact_mx_dma_transfer(
                    opcode="H_PREFETCH_M",
                    precision="matrix_kv",
                    hbm_base=layout.hbm_base_addr,
                    total_elements=physical_rows * physical_head_dim,
                    element_offset=element_offset,
                    dim=self.mlen,
                    amount=self.hbm_m_prefetch_amount,
                    stride=physical_head_dim,
                    rstride=1,
                    source=f"packed_pv:{v_input.name}:tile{k_idx}",
                ),
                multiplicity=col_blocks,
                axes=axes,
            )

    def scale_o_row(self, o_matrix: VRAMMatrixVar, q_idx: int, rows: int | None = None):
        """Scale current row block of O by m_res"""
        o_info = super().get_tensor_info(o_matrix.name)
        seq_len, head_dim = o_info.shape

        super().scale_o_row(
            o_matrix=o_matrix.name,
            q_idx=q_idx,
            seq_len=seq_len,
            head_dim=head_dim,
            rows=rows,
        )

    def final_scale_o(self, q_idx: int, o_matrix: VRAMMatrixVar, rows: int | None = None):
        """Final scaling: O[q_idx] = O[q_idx] / l"""
        o_info = super().get_tensor_info(o_matrix.name)
        seq_len, head_dim = o_info.shape

        super().final_scale_o(
            q_idx=q_idx,
            o_matrix=o_matrix.name,
            seq_len=seq_len,
            head_dim=head_dim,
            rows=rows,
        )


__all__ = ["ProgramAttentionMixin"]
