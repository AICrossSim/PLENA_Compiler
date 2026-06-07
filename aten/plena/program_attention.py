"""Flash-attention operations for the PLENA program builder."""

from __future__ import annotations

import math
import os
from compiler.asm_templates._imm import add_large_int
from compiler.asm_templates._imm import load_large_int
from compiler.aten.plena.vars import InputVar, VRAMMatrixVar


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
        self.emit("\n".join(lines) + "\n")
        return mask

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
        kv_residency: bool = False,
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
                kv_residency=kv_residency,
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
        kv_residency: bool = False,
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

        # --- KV-residency planning (opt-in, vision MHA prefill) ---------------
        # Load each head's full K and V into MRAM ONCE per (batch) before the
        # q_idx loop, then read by k_idx slot in the inner loop instead of
        # re-prefetching K/V for every (q_idx, k_idx). Same bytes -> identical
        # numerics; only the redundant reload/reset is removed. Disabled unless
        # the configured MRAM tile capacity can hold the resident K+V regions,
        # so the default (small-MRAM) path stays byte-for-byte unchanged.
        tile_elems = mlen * mlen
        num_k_col_blocks = max(1, math.ceil(head_dim / mlen))
        k_tile_span = num_k_col_blocks * tile_elems
        v_tile_span = num_k_col_blocks * tile_elems
        resident_tiles = num_k_blocks * (num_k_col_blocks + num_k_col_blocks)
        # Kill-switch for A/B byte-identity validation against the per-block path.
        residency_disabled = os.environ.get("PLENA_DISABLE_KV_RESIDENCY") == "1"
        use_residency = (
            bool(kv_residency)
            and not residency_disabled
            and (resident_tiles <= self.mram_tile_capacity)
        )
        if use_residency:
            k_resident_base = 0
            v_resident_base = num_k_blocks * k_tile_span

        valid_col_masks: dict[int, VRAMMatrixVar] = {}
        for k_idx in range(num_k_blocks):
            block_cols = min(mlen, kv_seq_len - k_idx * mlen)
            if self._needs_explicit_valid_col_mask(block_cols):
                valid_col_masks[block_cols] = self._build_valid_col_mask(
                    f"_mha_valid_col_mask_{block_cols}", block_cols
                )

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

            if use_residency:
                # Bulk-load this head/batch's full K and V into MRAM ONCE.
                # K[physical_k_idx] -> K_base + k_idx*k_tile_span (4 col-blocks,
                # tile stride mlen*mlen, exactly the load_sub_matrix_row layout
                # the QK^T M_TMM expects). V via residency_load_v into V_base.
                self._ensure_hbm_sub_matrix_registered(K)
                self._ensure_hbm_sub_matrix_registered(V)
                self.reset_mram()
                for k_idx in range(num_k_blocks):
                    physical_k_idx = batch_k_block_base + k_idx
                    k_slot = k_resident_base + k_idx * k_tile_span
                    self.load_sub_matrix_row(
                        name=K.name,
                        row_idx=physical_k_idx,
                        mram_start_addr=k_slot,
                    )
                self.residency_load_v(
                    v_sub_matrix=V.name,
                    k_indices=[batch_k_block_base + k for k in range(num_k_blocks)],
                    head_dim=head_dim,
                    v_resident_mram_base=v_resident_base,
                )

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
                    k_override = (
                        k_resident_base + k_idx * k_tile_span if use_residency else None
                    )
                    v_resident_base_k = (
                        v_resident_base + k_idx * v_tile_span if use_residency else None
                    )
                    self.vram_sub_projection_T_to(
                        Q_batch,
                        q_idx,
                        K,
                        physical_k_idx,
                        S_block,
                        target_row_idx=0,
                        target_col_idx=0,
                        mram_start_override=k_override,
                    )
                    valid_col_mask = valid_col_masks.get(block_cols)
                    if valid_col_mask is not None:
                        self.vram_add(S_block, valid_col_mask, num_rows=block_rows)
                    if needs_triangular_mask:
                        self.vram_add(S_block, causal_mask)
                    softmax_valid_cols = None if valid_col_mask is not None else block_cols
                    self.online_softmax_block(S_block, scale, rows=block_rows, valid_cols=softmax_valid_cols)
                    self.compute_pv(
                        S_block,
                        V,
                        physical_k_idx,
                        PV,
                        head_dim,
                        rows=block_rows,
                        v_resident_mram_base=v_resident_base_k,
                    )
                    self.scale_o_row(O_batch, q_idx, rows=block_rows)
                    self.vram_add(O_batch, PV, dst_row_offset=q_idx * mlen, num_rows=block_rows)

                self.final_scale_o(q_idx, O_batch, rows=block_rows)

        for mask in valid_col_masks.values():
            self.free_tensor(mask)

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
        self._ensure_hbm_sub_matrix_registered(K)
        k_layout = self.get_hbm_layout(K.name)
        self._emit_hbm_matrix_load(
            k_layout,
            3,
            lambda addr_reg, gp_regs: self.load_sub_matrix_asm(
                name=K.name,
                row_idx=k_idx,
                col_idx=0,
                mram_dest_addr=0,
                hbm_addr_reg=addr_reg,
                gp_regs=gp_regs,
            ),
        )

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

    def _emit_packed_qkt_to_s_dynamic(
        self,
        *,
        q_base_gp: int,
        k_hbm_addr_reg: int,
        s_base_address: int,
        k_layout,
        k_head_offset: int = 0,
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
            f"S_ADDI_INT gp{gp_hbm}, gp0, 0",
            f"H_PREFETCH_M gp{gp_mram}, gp{gp_hbm}, a{k_hbm_addr_reg}, 1, 0",
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
        if seq_len > mlen:
            raise NotImplementedError("Packed attention currently supports one sequence tile.")
        if group_heads > broadcast_amount:
            raise ValueError(f"group_heads={group_heads} exceeds broadcast_amount={broadcast_amount}")
        if broadcast_amount * head_slot_dim != mlen:
            raise ValueError(
                f"broadcast_amount*head_slot_dim must equal MLEN "
                f"({broadcast_amount}*{head_slot_dim} != {mlen})"
            )
        if getattr(self, "hlen", head_slot_dim) != head_slot_dim:
            raise ValueError(f"Packed attention requires HLEN={head_slot_dim}, got {self.hlen}")

        self._ensure_hbm_sub_matrix_registered(K)
        self._ensure_hbm_sub_matrix_registered(V)
        self._ensure_vram_matrix_layout(Q_group.name)

        rows = min(mlen, seq_len)
        # M_BTMM applies the emulator's fixed bmm_scale (0.25). Compensate in
        # online softmax so the effective QK scale remains caller-provided.
        softmax_scale = scale / 0.25

        s_views = [
            self.alloc_at(
                f"_packed_S_h{head}",
                mlen,
                mlen,
                scratch_base_address + head * mlen * mlen,
                physical_shape=(mlen, mlen),
            )
            for head in range(group_heads)
        ]
        pv = self.alloc("_packed_PV", rows, head_slot_dim, strict=False, physical_shape=(mlen, mlen))
        pack_scratch = self.alloc("_packed_pack_scratch", 1, mlen, strict=False, physical_shape=(1, mlen))
        pack_scratch_addr = self.get_vram_addr(pack_scratch.name)

        self.emit(
            self._reset_vram_asm(
                start_address=output_base_address,
                rows=rows,
                cols=mlen,
                total_rows=output_physical_rows,
                mlen=mlen,
            )
        )
        self._emit_packed_qkt_to_s(
            Q_group=Q_group,
            K=K,
            q_idx=0,
            k_idx=k_idx,
            s_base_address=scratch_base_address,
        )

        active_cols = valid_cols or seq_len
        valid_col_mask = (
            self._build_valid_col_mask(f"_packed_valid_col_mask_{active_cols}", active_cols)
            if self._needs_explicit_valid_col_mask(active_cols)
            else None
        )
        for head, s_head in enumerate(s_views):
            o_head = self.alloc(
                f"_packed_O_head{head}",
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
                else active_cols
            )
            self.online_softmax_block(s_head, softmax_scale, rows=rows, valid_cols=softmax_valid_cols)
            self.compute_pv(s_head, V, k_idx, pv, head_slot_dim, rows=rows)
            self.scale_o_row(o_head, 0, rows=rows)
            self.vram_add(o_head, pv, num_rows=rows)
            self.final_scale_o(0, o_head, rows=rows)
            output_head = output_head_base + head
            output_col_block = (output_head * head_slot_dim) // mlen
            output_lane = (output_head * head_slot_dim) % mlen // head_slot_dim
            output_block_base = output_base_address + output_col_block * output_physical_rows * mlen
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

        if valid_col_mask is not None:
            self.free_tensor(valid_col_mask)
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
        if seq_len > mlen:
            raise NotImplementedError("KV-group looped packed attention currently supports one sequence tile.")
        if q_physical_cols < num_kv_heads * mlen:
            raise ValueError(
                f"Q_full physical cols {q_physical_cols} cannot hold {num_kv_heads} MLEN-wide groups"
            )
        if group_heads > broadcast_amount:
            raise ValueError(f"group_heads={group_heads} exceeds broadcast_amount={broadcast_amount}")
        if broadcast_amount * head_slot_dim != mlen:
            raise ValueError(
                f"broadcast_amount*head_slot_dim must equal MLEN "
                f"({broadcast_amount}*{head_slot_dim} != {mlen})"
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

        q_group_stride = q_physical_rows * mlen
        o_group_stride = q_physical_rows * mlen
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
        valid_col_mask = (
            self._build_valid_col_mask(f"_packed_loop_valid_col_mask_{seq_len}", seq_len)
            if self._needs_explicit_valid_col_mask(seq_len)
            else None
        )

        gp_q, gp_o, gp_k, gp_v, gp_tmp, gp_kv_loop = self.register_allocator.allocate_gp(6)
        k_addr_reg, v_addr_reg = self.register_allocator.allocate_addr(2)
        try:
            setup_lines = [
                "; === Packed GQA attention core loop over KV groups ===",
                *load_large_int(gp_q, self.get_vram_addr(Q_full.name)),
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
            if valid_col_mask is not None:
                self.free_tensor(valid_col_mask)
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
        if broadcast_amount * head_slot_dim != mlen:
            raise ValueError(
                f"broadcast_amount*head_slot_dim must equal MLEN "
                f"({broadcast_amount}*{head_slot_dim} != {mlen})"
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

    def init_online_softmax(self, q_idx: int, o_matrix: VRAMMatrixVar, rows: int | None = None):
        """Initialize Online Softmax state: m=-inf, l=0, O_row=0"""
        o_info = super().get_tensor_info(o_matrix.name)
        seq_len, head_dim = o_info.shape

        super().init_online_softmax(
            q_idx=q_idx,
            o_matrix=o_matrix.name,
            seq_len=seq_len,
            head_dim=head_dim,
            rows=rows,
        )

    def online_softmax_block(
        self,
        s_block: VRAMMatrixVar,
        scale: float,
        rows: int | None = None,
        valid_cols: int | None = None,
    ):
        """Perform Online Softmax on S block"""
        super().online_softmax_block(
            s_block_matrix=s_block.name,
            scale=scale,
            rows=rows,
            valid_cols=valid_cols,
        )

    def compute_pv(
        self,
        s_block: VRAMMatrixVar,
        v_input: InputVar,
        k_idx: int,
        pv_matrix: VRAMMatrixVar,
        head_dim: int,
        rows: int | None = None,
        v_resident_mram_base: int | None = None,
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
            v_resident_mram_base=v_resident_mram_base,
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
