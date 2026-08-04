"""Flash-attention operations for the PLENA program builder."""

from __future__ import annotations

import math
from compiler.asm_templates import preload_addr_reg_asm
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

    def _scale_scores_before_mask(
        self,
        scores: VRAMMatrixVar,
        scale: float,
        rows: int,
    ) -> float:
        """Scale score rows before adding a finite representation of -infinity."""
        if scale != 1.0:
            self.tile_row_mul_fp_broadcast_asm(
                self.get_vram_addr(scores.name),
                1,
                list(range(rows)),
            )
        return 1.0

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
                    block_scale = scale
                    if valid_col_mask is not None or needs_triangular_mask:
                        block_scale = self._scale_scores_before_mask(
                            S_block,
                            scale,
                            block_rows,
                        )
                    if valid_col_mask is not None:
                        self.vram_add(S_block, valid_col_mask, num_rows=block_rows)
                    if needs_triangular_mask:
                        self.vram_add(S_block, causal_mask)
                    softmax_valid_cols = None if valid_col_mask is not None else block_cols
                    self.online_softmax_block(
                        S_block,
                        block_scale,
                        rows=block_rows,
                        valid_cols=softmax_valid_cols,
                    )
                    self.compute_pv(S_block, V, physical_k_idx, PV, head_dim, rows=block_rows)
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
                "ATen fused GQA supports one packed KV group in this "
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
            raise NotImplementedError("Packed GQA lowering supports one sequence tile.")
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
        k_head_selector: int = 0,
        q_rows: int | None = None,
        kv_cols: int | None = None,
        prefetch_k: bool = True,
    ) -> None:
        """Emit packed QK^T with M_BTMM/M_BMM_WO into head-major S tiles.

        The per-head systolic (M_BTMM) produces one BLEN(query) x BLEN(kv) tile per
        head per op. The full per-head S is tiled BLEN x BLEN: outer loop over
        query-row tiles (qt), inner over kv-column tiles (kt). Each M_BTMM reads
          rs1 = K MRAM base + kt*BLEN*MLEN  (the kv-tile's block rows, transposed)
          rs2 = Q VRAM base + qt*BLEN*MLEN  (the query-row tile)
        and M_BMM_WO writes to s_base + qt*BLEN*MLEN + kt*BLEN; the DFC per-head drain
        then scatters the ROW_BLOCK_NUM heads head-major (base + head*MLEN*MLEN).
        """
        self._ensure_hbm_sub_matrix_registered(K)
        k_layout = self.get_hbm_layout(K.name)
        if prefetch_k:
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
                    matrix_precision="key_value",
                ),
            )

        mlen, blen = self.mlen, self.blen
        selector_count = mlen // self.hlen
        if not 0 <= k_head_selector < selector_count:
            raise ValueError(
                f"K head selector {k_head_selector} outside [0,{selector_count})"
            )
        q_base = self.get_vram_addr(Q_group.name) + q_idx * mlen * mlen
        n_q_tiles = max(1, (min(mlen, q_rows or mlen) + blen - 1) // blen)
        n_kv_tiles = max(1, (min(mlen, kv_cols or mlen) + blen - 1) // blen)

        gp_q, gp_s, gp_k = self.register_allocator.allocate_gp(3)
        lines = ["; === Packed GQA QK^T using compiler M_BTMM "
                 f"({n_q_tiles} query-tile x {n_kv_tiles} kv-tile) ==="]
        for qt in range(n_q_tiles):
            for kt in range(n_kv_tiles):
                k_off = kt * blen * mlen
                # kt==0 reads K from MRAM base 0 (gp0); kt>0 offsets by kt*BLEN*MLEN
                # (the kv-tile's block rows, read transposed).
                k_reg = "gp0" if k_off == 0 else f"gp{gp_k}"
                lines += ([] if k_off == 0 else load_large_int(gp_k, k_off))
                lines += [
                    *load_large_int(gp_q, q_base + qt * blen * mlen),
                    f"M_BTMM {k_head_selector}, {k_reg}, gp{gp_q}",
                    *load_large_int(gp_s, s_base_address + qt * blen * mlen + kt * blen),
                    f"M_BMM_WO gp{gp_s}, 0",
                ]
        self.register_allocator.free_gp([gp_q, gp_s, gp_k])
        self.emit("\n".join(lines) + "\n")

    def _emit_packed_qkt_to_s_dynamic(
        self,
        *,
        q_base_gp: int,
        k_hbm_addr_reg: int,
        k_hbm_offset_gp: int,
        s_base_address: int,
        k_layout,
        k_head_selector: int = 0,
    ) -> None:
        """Emit packed QK^T using loop-carried Q and K byte offsets."""
        selector_count = self.mlen // self.hlen
        if not 0 <= k_head_selector < selector_count:
            raise ValueError(
                f"K head selector {k_head_selector} outside [0,{selector_count})"
            )
        gp_mram, gp_hbm, gp_s = self.register_allocator.allocate_gp(3)
        _, cols = k_layout.physical_shape or k_layout.full_shape
        lines = [
            "; === Packed GQA QK^T using compiler M_BTMM (KV-looped) ===",
            *load_large_int(gp_hbm, k_layout.element_plane_bytes),
            f"C_SET_SCALE_REG gp{gp_hbm}",
            *load_large_int(
                gp_hbm,
                k_layout.element_stride_bytes(cols),
            ),
            f"C_SET_STRIDE_REG gp{gp_hbm}",
            f"S_ADDI_INT gp{gp_mram}, gp0, 0",
            f"H_PREFETCH_M gp{gp_mram}, gp{k_hbm_offset_gp}, "
            f"a{k_hbm_addr_reg}, 1, 1",
            f"M_BTMM {k_head_selector}, gp0, gp{q_base_gp}",
            *load_large_int(gp_s, s_base_address),
            f"M_BMM_WO gp{gp_s}, 0",
        ]
        self.register_allocator.free_gp([gp_mram, gp_hbm, gp_s])
        self.emit("\n".join(lines) + "\n")

    def _emit_packed_matrix_prefetch_dynamic(
        self,
        *,
        layout,
        hbm_addr_reg: int,
        hbm_offset_gp: int,
        mram_dest_addr: int,
    ) -> None:
        """Prefetch one packed matrix tile at a loop-carried byte offset."""

        gp_mram, gp_setup = self.register_allocator.allocate_gp(2)
        _, cols = layout.physical_shape or layout.full_shape
        lines = [
            f"; Load SubMatrix {layout.name} at loop-carried row block",
            *load_large_int(gp_setup, layout.element_plane_bytes),
            f"C_SET_SCALE_REG gp{gp_setup}",
            *load_large_int(gp_setup, layout.element_stride_bytes(cols)),
            f"C_SET_STRIDE_REG gp{gp_setup}",
            *load_large_int(gp_mram, mram_dest_addr),
            f"H_PREFETCH_M gp{gp_mram}, gp{hbm_offset_gp}, "
            f"a{hbm_addr_reg}, 1, 1",
        ]
        self.register_allocator.free_gp([gp_mram, gp_setup])
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
        ]
        if getattr(self, "unroll_attention", False):
            # Straight-line pack (no C_LOOP). The last head's pack C_LOOP_END sits
            # right before the program's C_BREAK, and the decoder's early_loop_end_stall
            # lookahead pins pc_reg there (C_BREAK never decodes -> hang). Unrolling
            # removes every pack loop, so no C_LOOP_END precedes the trailing C_BREAK.
            for i in range(rows):
                lines.append(f"V_SHIFT_V gp{gp_scratch}, gp{gp_src}, gp{gp_shift}")
                lines.append(f"V_ADD_VV gp{gp_dst}, gp{gp_dst}, gp{gp_scratch}, 0")
                if i < rows - 1:
                    lines.append(f"S_ADDI_INT gp{gp_src}, gp{gp_src}, {self.mlen}")
                    lines.append(f"S_ADDI_INT gp{gp_dst}, gp{gp_dst}, {self.mlen}")
        else:
            lines += [
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
        kv_head_selector: int = 0,
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
            raise NotImplementedError("Packed attention supports one sequence tile.")
        if group_heads > broadcast_amount:
            raise ValueError(f"group_heads={group_heads} exceeds broadcast_amount={broadcast_amount}")
        if broadcast_amount * head_slot_dim != mlen:
            raise ValueError(
                f"broadcast_amount*head_slot_dim must equal MLEN "
                f"({broadcast_amount}*{head_slot_dim} != {mlen})"
            )
        if getattr(self, "hlen", head_slot_dim) != head_slot_dim:
            raise ValueError(f"Packed attention requires HLEN={head_slot_dim}, got {self.hlen}")
        if not 0 <= kv_head_selector < mlen // head_slot_dim:
            raise ValueError(
                f"KV head selector {kv_head_selector} outside "
                f"[0,{mlen // head_slot_dim})"
            )

        self._ensure_hbm_sub_matrix_registered(K)
        self._ensure_hbm_sub_matrix_registered(V)
        self._ensure_vram_matrix_layout(Q_group.name)

        rows = min(mlen, seq_len)
        active_cols = valid_cols if valid_cols is not None else seq_len
        if active_cols <= 0:
            raise ValueError("packed attention requires at least one KV column")
        kv_blocks = math.ceil(active_cols / mlen)
        if kv_blocks > 1 and rows != 1:
            raise NotImplementedError(
                "multi-tile PackedKV lowering is restricted to q_len=1 decode"
            )
        # M_BTMM leaves QK scores unscaled; softmax applies the declared scale.
        softmax_scale = scale

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
        if kv_blocks > 1:
            full_block_count, tail_columns = divmod(active_cols, mlen)
            o_heads = [
                self.alloc(
                    f"_packed_O_head{head}",
                    rows,
                    head_slot_dim,
                    strict=False,
                    physical_shape=(mlen, mlen),
                )
                for head in range(group_heads)
            ]
            state_stride = rows
            state_bases = [
                self._ONLINE_SOFTMAX_FPSRAM_BASE
                + head * 3 * state_stride
                for head in range(group_heads)
            ]
            for head, o_head in enumerate(o_heads):
                self.emit(
                    f"; PackedKV softmax state head {head}, "
                    f"base {state_bases[head]}, stride {state_stride}\n"
                )
                self.init_online_softmax(
                    0,
                    o_head,
                    rows=rows,
                    state_base_address=state_bases[head],
                    state_stride=state_stride,
                )

            tail_mask = (
                self._build_valid_col_mask(
                    f"_packed_valid_col_mask_{tail_columns}",
                    tail_columns,
                )
                if tail_columns
                else None
            )
            k_layout = self.get_hbm_layout(K.name)
            v_layout = self.get_hbm_layout(V.name)
            k_block_step = k_layout.element_offset_bytes(mlen * mlen)
            v_block_step = v_layout.element_offset_bytes(mlen * mlen)
            v_selector_offset = v_layout.element_offset_bytes(
                kv_head_selector * head_slot_dim
            )
            gp_k_offset, gp_v_offset, gp_step, gp_loop = (
                self.register_allocator.allocate_gp(4)
            )
            k_addr_reg, v_addr_reg = self.register_allocator.allocate_addr(2)
            try:
                setup = preload_addr_reg_asm(
                    addr_reg_to_set=[k_addr_reg, v_addr_reg],
                    available_registers=[gp_step, gp_loop],
                    addr_reg_val=[k_layout.hbm_base_addr, v_layout.hbm_base_addr],
                )
                setup += "\n".join(
                    [
                        *load_large_int(gp_k_offset, k_idx * k_block_step),
                        *load_large_int(
                            gp_v_offset,
                            k_idx * v_block_step + v_selector_offset,
                        ),
                    ]
                ) + "\n"
                self.emit(setup)

                def emit_sequence_block(
                    block_cols: int,
                    valid_col_mask: VRAMMatrixVar | None,
                ) -> None:
                    self._emit_packed_matrix_prefetch_dynamic(
                        layout=k_layout,
                        hbm_addr_reg=k_addr_reg,
                        hbm_offset_gp=gp_k_offset,
                        mram_dest_addr=0,
                    )
                    self._emit_packed_qkt_to_s(
                        Q_group=Q_group,
                        K=K,
                        q_idx=0,
                        k_idx=0,
                        s_base_address=scratch_base_address,
                        k_head_selector=kv_head_selector,
                        q_rows=rows,
                        kv_cols=block_cols,
                        prefetch_k=False,
                    )
                    self._emit_packed_matrix_prefetch_dynamic(
                        layout=v_layout,
                        hbm_addr_reg=v_addr_reg,
                        hbm_offset_gp=gp_v_offset,
                        mram_dest_addr=0,
                    )
                    for head, s_head in enumerate(s_views):
                        if valid_col_mask is not None:
                            block_scale = self._scale_scores_before_mask(
                                s_head,
                                softmax_scale,
                                rows,
                            )
                            self.vram_add(
                                s_head,
                                valid_col_mask,
                                num_rows=rows,
                            )
                        else:
                            block_scale = softmax_scale
                        self.online_softmax_block(
                            s_head,
                            block_scale,
                            rows=rows,
                            valid_cols=(
                                None
                                if valid_col_mask is not None
                                else block_cols
                            ),
                            state_base_address=state_bases[head],
                            state_stride=state_stride,
                        )
                        self.compute_pv(
                            s_head,
                            V,
                            0,
                            pv,
                            head_slot_dim,
                            rows=rows,
                            prefetch_v=False,
                        )
                        self.scale_o_row(
                            o_heads[head],
                            0,
                            rows=rows,
                            state_base_address=state_bases[head],
                            state_stride=state_stride,
                        )
                        self.vram_add(o_heads[head], pv, num_rows=rows)

                if full_block_count:
                    self.emit(
                        "; PackedKV compact full-block loop\n"
                        f"C_LOOP_START gp{gp_loop}, {full_block_count}\n"
                    )
                    emit_sequence_block(mlen, None)
                    advances = [
                        *add_large_int(
                            gp_k_offset,
                            gp_k_offset,
                            k_block_step,
                            temp_reg=gp_step,
                        ),
                        *add_large_int(
                            gp_v_offset,
                            gp_v_offset,
                            v_block_step,
                            temp_reg=gp_step,
                        ),
                        f"C_LOOP_END gp{gp_loop}",
                    ]
                    self.emit("\n".join(advances) + "\n")
                if tail_columns:
                    self.emit(
                        "; PackedKV compact masked-tail block, "
                        f"valid columns {tail_columns}\n"
                    )
                    emit_sequence_block(tail_columns, tail_mask)
                self.emit("; PackedKV compact sequence complete\n")
            finally:
                self.register_allocator.free_addr([k_addr_reg, v_addr_reg])
                self.register_allocator.free_gp(
                    [gp_k_offset, gp_v_offset, gp_step, gp_loop]
                )

            for head, o_head in enumerate(o_heads):
                self.final_scale_o(
                    0,
                    o_head,
                    rows=rows,
                    state_base_address=state_bases[head],
                    state_stride=state_stride,
                )
                output_head = output_head_base + head
                output_col_block = (
                    output_head * head_slot_dim
                ) // mlen
                output_lane = (
                    (output_head * head_slot_dim) % mlen
                ) // head_slot_dim
                output_block_base = (
                    output_base_address
                    + output_col_block * output_physical_rows * mlen
                )
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

            if tail_mask is not None:
                self.free_tensor(tail_mask)
            for s_view in s_views:
                self.free_tensor(s_view)
            self.free_tensor(pv)
            self.free_tensor(pack_scratch)
            return

        self._emit_packed_qkt_to_s(
            Q_group=Q_group,
            K=K,
            q_idx=0,
            k_idx=k_idx,
            s_base_address=scratch_base_address,
            k_head_selector=kv_head_selector,
            q_rows=rows,
            kv_cols=(valid_cols if valid_cols is not None else seq_len),
        )

        valid_col_mask = (
            self._build_valid_col_mask(f"_packed_valid_col_mask_{active_cols}", active_cols)
            if self._needs_explicit_valid_col_mask(active_cols)
            else None
        )
        # Warm V into MSRAM once so head 0's cold-start-reprimed first tile reads V
        # (not the QK^T's stale K) on row 0 too — it lands during head 0's softmax.
        v_element_offset = kv_head_selector * head_slot_dim
        self.warm_v_prefetch(V.name, k_idx, v_element_offset=v_element_offset)
        for head, s_head in enumerate(s_views):
            o_head = self.alloc(
                f"_packed_O_head{head}",
                rows,
                head_slot_dim,
                strict=False,
                physical_shape=(mlen, mlen),
            )
            self.init_online_softmax(0, o_head, rows=rows)
            applied_scale = softmax_scale
            if valid_col_mask is not None or isinstance(causal_mask, VRAMMatrixVar):
                applied_scale = self._scale_scores_before_mask(
                    s_head,
                    softmax_scale,
                    rows,
                )
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
            # Single KV block (fused packed path): normalise P register-direct in the
            # softmax block, so pv = softmax(S)@V = O directly. This avoids the FP-SRAM
            # l/m_res round-trip used by scale_o_row/final_scale_o, whose per-row store
            # is racy on this RTL (drops -> O saturates to FP12 max; see online_softmax.py).
            # o_head was zeroed by init_online_softmax, so vram_add copies pv into O.
            self.online_softmax_block(s_head, applied_scale, rows=rows,
                                      valid_cols=softmax_valid_cols, inline_normalize=True)
            self.compute_pv(
                s_head,
                V,
                k_idx,
                pv,
                head_slot_dim,
                rows=rows,
                v_element_offset=v_element_offset,
                prefetch_v=False,
            )
            self.vram_add(o_head, pv, num_rows=rows)
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
        for s_view in s_views:
            self.free_tensor(s_view)
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
            raise NotImplementedError("KV-group looped packed attention supports one sequence tile.")
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
        softmax_scale = scale
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
                k_hbm_offset_gp=0,
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
                applied_scale = softmax_scale
                if valid_col_mask is not None or isinstance(causal_mask, VRAMMatrixVar):
                    applied_scale = self._scale_scores_before_mask(
                        s_head,
                        softmax_scale,
                        rows,
                    )
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
                self.online_softmax_block(s_head, applied_scale, rows=rows, valid_cols=softmax_valid_cols)
                self.emit(
                    self._pv_multiply_asm(
                        mlen=mlen,
                        blen=self.blen,
                        head_dim=head_slot_dim,
                        p_address=self.get_vram_addr(s_head.name),
                        v_hbm_offset_reg=v_addr_reg,
                        v_hbm_offset=0,
                        v_element_bits=k_layout.hbm_element_width,
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
        kv_head_selector: int = 0,
    ) -> None:
        """Emit one KV group's packed-head flash-attention body.

        Q_group and the output use an MLEN-wide row where active Q heads occupy
        HLEN-sized lanes. ``kv_head_selector`` chooses an HLEN lane when K/V
        share one PackedKV row and remains zero for padded per-head inputs.
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
            kv_head_selector=kv_head_selector,
        )

    def _flash_attention_packed_cache_reused(
        self,
        Q_full: VRAMMatrixVar,
        K_packed: InputVar,
        V_packed: InputVar,
        *,
        num_kv_heads: int,
        group_heads: int,
        head_slot_dim: int,
        output_base_address: int,
        scratch_base_address: int,
        broadcast_amount: int,
        scale: float | None,
        causal_mask: bool | VRAMMatrixVar | None,
        k_idx: int,
        active_cols: int,
        batch_size: int,
        rows_per_batch: int,
        cache_rows_per_batch: int,
    ) -> None:
        """Reuse each resident packed K/V tile across every local selector."""

        if group_heads <= 0 or group_heads > broadcast_amount:
            raise ValueError(
                f"group_heads={group_heads} must be in "
                f"[1,broadcast_amount={broadcast_amount}]"
            )
        if broadcast_amount * head_slot_dim != self.mlen:
            raise ValueError(
                "broadcast_amount*head_slot_dim must equal MLEN "
                f"({broadcast_amount}*{head_slot_dim} != {self.mlen})"
            )
        if self.mram_tile_capacity < 2:
            raise ValueError("KV-head reuse requires two resident MRAM tiles")
        total_query_heads = num_kv_heads * group_heads
        state_end = self._ONLINE_SOFTMAX_FPSRAM_BASE + 3 * total_query_heads
        if state_end > self.fpram_allocator.total_size:
            raise ValueError(
                "KV-head reuse softmax state exceeds scalar FP SRAM capacity"
            )

        mlen = self.mlen
        q_physical_rows = Q_full.physical_shape[0]
        batch_stride = rows_per_batch * mlen
        group_stride = q_physical_rows * mlen
        cache_blocks_per_batch = cache_rows_per_batch // mlen
        kv_blocks = math.ceil(active_cols / mlen)
        resident_v_tile = mlen * mlen
        softmax_scale = (
            1.0 / math.sqrt(head_slot_dim) if scale is None else scale
        )
        workspace_rows = math.ceil(total_query_heads / mlen) * mlen

        s_views = [
            self.alloc_at(
                f"_packed_reuse_S_h{head}",
                mlen,
                mlen,
                scratch_base_address + head * mlen * mlen,
                physical_shape=(mlen, mlen),
            )
            for head in range(group_heads)
        ]
        pv = self.alloc(
            "_packed_reuse_PV",
            1,
            head_slot_dim,
            strict=False,
            physical_shape=(mlen, mlen),
        )
        pack_scratch = self.alloc(
            "_packed_reuse_pack_scratch",
            1,
            mlen,
            strict=False,
            physical_shape=(1, mlen),
        )
        pack_scratch_addr = self.get_vram_addr(pack_scratch.name)
        q_base = self.get_vram_addr(Q_full.name)

        try:
            for batch_idx in range(batch_size):
                o_workspace = self.alloc(
                    f"_packed_reuse_O_b{batch_idx}",
                    total_query_heads,
                    head_slot_dim,
                    strict=False,
                    physical_shape=(workspace_rows, mlen),
                )
                o_workspace_base = self.get_vram_addr(o_workspace.name)
                o_heads: list[VRAMMatrixVar] = []
                q_groups: list[VRAMMatrixVar] = []
                try:
                    for selector in range(num_kv_heads):
                        output_group_base = (
                            output_base_address
                            + selector * group_stride
                            + batch_idx * batch_stride
                        )
                        self.emit(
                            self._reset_vram_asm(
                                start_address=output_group_base,
                                rows=1,
                                cols=mlen,
                                total_rows=rows_per_batch,
                                mlen=mlen,
                            )
                        )
                        q_groups.append(
                            self.alloc_at(
                                f"_packed_reuse_Q_b{batch_idx}_g{selector}",
                                1,
                                mlen,
                                (
                                    q_base
                                    + selector * group_stride
                                    + batch_idx * batch_stride
                                ),
                                physical_shape=(rows_per_batch, mlen),
                            )
                        )
                        for head in range(group_heads):
                            global_head = selector * group_heads + head
                            o_head = self.alloc_at(
                                f"_packed_reuse_O_b{batch_idx}_h{global_head}",
                                1,
                                head_slot_dim,
                                o_workspace_base + global_head * mlen,
                                physical_shape=(workspace_rows, mlen),
                            )
                            state_base = (
                                self._ONLINE_SOFTMAX_FPSRAM_BASE
                                + 3 * global_head
                            )
                            self.init_online_softmax(
                                0,
                                o_head,
                                rows=1,
                                state_base_address=state_base,
                                state_stride=1,
                            )
                            o_heads.append(o_head)

                    batch_k_idx = k_idx + batch_idx * cache_blocks_per_batch
                    full_block_count, tail_columns = divmod(active_cols, mlen)
                    tail_mask = (
                        self._build_valid_col_mask(
                            f"_packed_reuse_valid_cols_{tail_columns}",
                            tail_columns,
                        )
                        if tail_columns
                        else None
                    )
                    k_layout = self.get_hbm_layout(K_packed.name)
                    v_layout = self.get_hbm_layout(V_packed.name)
                    k_block_step = k_layout.element_offset_bytes(mlen * mlen)
                    v_block_step = v_layout.element_offset_bytes(mlen * mlen)
                    gp_k_offset, gp_v_offset, gp_step, gp_loop = (
                        self.register_allocator.allocate_gp(4)
                    )
                    k_addr_reg, v_addr_reg = (
                        self.register_allocator.allocate_addr(2)
                    )
                    try:
                        setup = preload_addr_reg_asm(
                            addr_reg_to_set=[k_addr_reg, v_addr_reg],
                            available_registers=[gp_step, gp_loop],
                            addr_reg_val=[
                                k_layout.hbm_base_addr,
                                v_layout.hbm_base_addr,
                            ],
                        )
                        setup += "\n".join(
                            [
                                *load_large_int(
                                    gp_k_offset,
                                    batch_k_idx * k_block_step,
                                ),
                                *load_large_int(
                                    gp_v_offset,
                                    batch_k_idx * v_block_step,
                                ),
                            ]
                        ) + "\n"
                        self.emit(setup)

                        def emit_reused_block(
                            block_cols: int,
                            valid_col_mask: VRAMMatrixVar | None,
                        ) -> None:
                            self._emit_packed_matrix_prefetch_dynamic(
                                layout=k_layout,
                                hbm_addr_reg=k_addr_reg,
                                hbm_offset_gp=gp_k_offset,
                                mram_dest_addr=0,
                            )
                            self._emit_packed_matrix_prefetch_dynamic(
                                layout=v_layout,
                                hbm_addr_reg=v_addr_reg,
                                hbm_offset_gp=gp_v_offset,
                                mram_dest_addr=resident_v_tile,
                            )
                            for selector, q_group in enumerate(q_groups):
                                self.emit(
                                    f"; PackedKV reused batch {batch_idx}, "
                                    f"selector {selector}\n"
                                )
                                self._emit_packed_qkt_to_s(
                                    Q_group=q_group,
                                    K=K_packed,
                                    q_idx=0,
                                    k_idx=0,
                                    s_base_address=scratch_base_address,
                                    k_head_selector=selector,
                                    q_rows=1,
                                    kv_cols=block_cols,
                                    prefetch_k=False,
                                )
                                for head, s_head in enumerate(s_views):
                                    global_head = selector * group_heads + head
                                    o_head = o_heads[global_head]
                                    state_base = (
                                        self._ONLINE_SOFTMAX_FPSRAM_BASE
                                        + 3 * global_head
                                    )
                                    applied_scale = softmax_scale
                                    if valid_col_mask is not None or isinstance(
                                        causal_mask,
                                        VRAMMatrixVar,
                                    ):
                                        applied_scale = self._scale_scores_before_mask(
                                            s_head,
                                            softmax_scale,
                                            1,
                                        )
                                    if valid_col_mask is not None:
                                        self.vram_add(
                                            s_head,
                                            valid_col_mask,
                                            num_rows=1,
                                        )
                                    if isinstance(causal_mask, VRAMMatrixVar):
                                        self.vram_add(
                                            s_head,
                                            causal_mask,
                                            num_rows=1,
                                        )
                                    elif causal_mask is True:
                                        self.emit(
                                            "; NOTE: reused cached attention received "
                                            "causal_mask=True without a VRAM mask; "
                                            "no mask applied.\n"
                                        )

                                    single_block = kv_blocks == 1
                                    self.online_softmax_block(
                                        s_head,
                                        applied_scale,
                                        rows=1,
                                        valid_cols=(
                                            None
                                            if valid_col_mask is not None
                                            or isinstance(
                                                causal_mask,
                                                VRAMMatrixVar,
                                            )
                                            else block_cols
                                        ),
                                        inline_normalize=single_block,
                                        state_base_address=state_base,
                                        state_stride=1,
                                    )
                                    if not single_block:
                                        self.scale_o_row(
                                            o_head,
                                            0,
                                            rows=1,
                                            state_base_address=state_base,
                                            state_stride=1,
                                        )
                                    self.compute_pv(
                                        s_head,
                                        V_packed,
                                        0,
                                        pv,
                                        head_slot_dim,
                                        rows=1,
                                        prefetch_v=False,
                                        v_mram_base=(
                                            resident_v_tile
                                            + selector * head_slot_dim * mlen
                                        ),
                                    )
                                    self.vram_add(o_head, pv, num_rows=1)

                        if full_block_count:
                            self.emit(
                                "; PackedKV reused compact full-block loop\n"
                                f"C_LOOP_START gp{gp_loop}, "
                                f"{full_block_count}\n"
                            )
                            emit_reused_block(mlen, None)
                            advances = [
                                *add_large_int(
                                    gp_k_offset,
                                    gp_k_offset,
                                    k_block_step,
                                    temp_reg=gp_step,
                                ),
                                *add_large_int(
                                    gp_v_offset,
                                    gp_v_offset,
                                    v_block_step,
                                    temp_reg=gp_step,
                                ),
                                f"C_LOOP_END gp{gp_loop}",
                            ]
                            self.emit("\n".join(advances) + "\n")
                        if tail_columns:
                            self.emit(
                                "; PackedKV reused compact masked-tail block, "
                                f"valid columns {tail_columns}\n"
                            )
                            emit_reused_block(tail_columns, tail_mask)
                        self.emit(
                            "; PackedKV reused compact sequence complete\n"
                        )
                    finally:
                        self.register_allocator.free_addr(
                            [k_addr_reg, v_addr_reg]
                        )
                        self.register_allocator.free_gp(
                            [gp_k_offset, gp_v_offset, gp_step, gp_loop]
                        )
                        if tail_mask is not None:
                            self.free_tensor(tail_mask)

                    for selector in range(num_kv_heads):
                        output_group_base = (
                            output_base_address
                            + selector * group_stride
                            + batch_idx * batch_stride
                        )
                        for head in range(group_heads):
                            global_head = selector * group_heads + head
                            o_head = o_heads[global_head]
                            if kv_blocks > 1:
                                state_base = (
                                    self._ONLINE_SOFTMAX_FPSRAM_BASE
                                    + 3 * global_head
                                )
                                self.final_scale_o(
                                    0,
                                    o_head,
                                    rows=1,
                                    state_base_address=state_base,
                                    state_stride=1,
                                )
                            self._pack_o_head_to_output(
                                o_head=o_head,
                                output_base_address=output_group_base,
                                output_physical_rows=rows_per_batch,
                                head_slot=head,
                                head_slot_dim=head_slot_dim,
                                rows=1,
                                scratch_address=pack_scratch_addr,
                            )
                finally:
                    for q_group in q_groups:
                        self.free_tensor(q_group)
                    for o_head in o_heads:
                        self.free_tensor(o_head)
                    self.free_tensor(o_workspace)
        finally:
            for s_view in s_views:
                self.free_tensor(s_view)
            self.free_tensor(pv)
            self.free_tensor(pack_scratch)

    def flash_attention_packed_cache(
        self,
        Q_full: VRAMMatrixVar,
        K_packed: InputVar,
        V_packed: InputVar,
        *,
        num_kv_heads: int,
        group_heads: int,
        head_slot_dim: int,
        output_base_address: int,
        scratch_base_address: int,
        broadcast_amount: int,
        scale=None,
        causal_mask: bool | VRAMMatrixVar | None = True,
        k_idx: int = 0,
        valid_cols: int | None = None,
        cache_position: int | None = None,
        batch_size: int = 1,
        rows_per_batch: int | None = None,
        query_rows_per_batch: int | None = None,
        cache_rows_per_batch: int | None = None,
        kv_head_reuse: bool = False,
    ) -> None:
        """Unroll independent batch slabs and KV selectors over one packed cache."""
        if not isinstance(kv_head_reuse, bool):
            raise TypeError("kv_head_reuse must be boolean")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if num_kv_heads <= 0 or num_kv_heads > self.mlen // head_slot_dim:
            raise ValueError(
                f"num_kv_heads={num_kv_heads} exceeds "
                f"{self.mlen // head_slot_dim} selector slots"
            )
        q_rows, _ = Q_full.shape
        q_physical_rows, q_physical_cols = Q_full.physical_shape
        if rows_per_batch is None:
            if q_physical_rows % batch_size:
                raise ValueError(
                    f"Q physical rows {q_physical_rows} are not divisible by "
                    f"batch_size={batch_size}"
                )
            rows_per_batch = q_physical_rows // batch_size
        if rows_per_batch <= 0 or batch_size * rows_per_batch != q_physical_rows:
            raise ValueError(
                f"batch_size*rows_per_batch must equal Q physical rows "
                f"({batch_size}*{rows_per_batch} != {q_physical_rows})"
            )
        if rows_per_batch % self.mlen:
            raise ValueError(
                f"rows_per_batch={rows_per_batch} must be a multiple of MLEN={self.mlen}"
            )
        if query_rows_per_batch is None:
            if q_rows % batch_size:
                raise ValueError(
                    f"Q logical rows {q_rows} are not divisible by batch_size={batch_size}"
                )
            query_rows_per_batch = q_rows // batch_size
        if not 0 < query_rows_per_batch <= min(rows_per_batch, self.mlen):
            raise ValueError(
                f"query_rows_per_batch={query_rows_per_batch} must be in "
                f"[1,{min(rows_per_batch, self.mlen)}]"
            )
        if valid_cols is not None and valid_cols <= 0:
            raise ValueError("valid_cols must be positive")
        if query_rows_per_batch == 1:
            if valid_cols is None:
                raise ValueError(
                    "cached q_len=1 attention requires an explicit valid_cols"
                )
            if cache_position is None:
                raise ValueError(
                    "cached q_len=1 attention requires an explicit cache_position"
                )
            if (
                isinstance(cache_position, bool)
                or not isinstance(cache_position, int)
                or cache_position < 0
            ):
                raise ValueError("cache_position must be a non-negative integer")
            if cache_position != valid_cols - 1:
                raise ValueError(
                    "cached q_len=1 attention requires cache_position == "
                    "valid_cols - 1"
                )
        elif cache_position is not None:
            raise ValueError(
                "cache_position is only valid for cached q_len=1 attention"
            )
        if k_idx < 0:
            raise ValueError("k_idx must be non-negative")
        if q_physical_cols < num_kv_heads * self.mlen:
            raise ValueError(
                f"Q_full physical width {q_physical_cols} cannot hold "
                f"{num_kv_heads} MLEN groups"
            )
        k_physical_rows = K_packed.physical_shape[0]
        v_physical_rows = V_packed.physical_shape[0]
        if k_physical_rows != v_physical_rows:
            raise ValueError("PackedKV K and V physical row counts must match")
        if cache_rows_per_batch is None:
            if k_physical_rows % batch_size:
                raise ValueError(
                    f"PackedKV rows {k_physical_rows} are not divisible by "
                    f"batch_size={batch_size}"
                )
            cache_rows_per_batch = k_physical_rows // batch_size
        if (
            cache_rows_per_batch <= 0
            or cache_rows_per_batch % self.mlen
        ):
            raise ValueError(
                "cache_rows_per_batch must be a positive multiple of MLEN"
            )
        if batch_size * cache_rows_per_batch > k_physical_rows:
            raise ValueError("PackedKV physical rows cannot cover all batch slabs")
        active_cols = (
            query_rows_per_batch
            if valid_cols is None
            else valid_cols
        )
        if active_cols > cache_rows_per_batch:
            raise ValueError(
                f"valid_cols={active_cols} exceeds the "
                f"{cache_rows_per_batch}-row cache slab"
            )
        cache_blocks_per_batch = cache_rows_per_batch // self.mlen
        required_k_rows = (
            k_idx + batch_size * cache_blocks_per_batch
        ) * self.mlen
        for tensor, role in ((K_packed, "K"), (V_packed, "V")):
            physical = tensor.physical_shape
            if physical[1] != self.mlen:
                raise ValueError(
                    f"PackedKV {role} rows must be MLEN-wide, got {physical[1]}"
                )
            if physical[0] < required_k_rows:
                raise ValueError(
                    f"PackedKV {role} has {physical[0]} rows, "
                    f"requires at least {required_k_rows}"
                )
        q_base = self.get_vram_addr(Q_full.name)
        batch_stride = rows_per_batch * self.mlen
        group_stride = q_physical_rows * self.mlen
        if kv_head_reuse and num_kv_heads > 1:
            if query_rows_per_batch != 1:
                raise NotImplementedError(
                    "KV-head reuse requires cached q_len=1 attention"
                )
            self._flash_attention_packed_cache_reused(
                Q_full,
                K_packed,
                V_packed,
                num_kv_heads=num_kv_heads,
                group_heads=group_heads,
                head_slot_dim=head_slot_dim,
                output_base_address=output_base_address,
                scratch_base_address=scratch_base_address,
                broadcast_amount=broadcast_amount,
                scale=scale,
                causal_mask=causal_mask,
                k_idx=k_idx,
                active_cols=active_cols,
                batch_size=batch_size,
                rows_per_batch=rows_per_batch,
                cache_rows_per_batch=cache_rows_per_batch,
            )
            return
        for batch_idx in range(batch_size):
            batch_k_idx = (
                k_idx + batch_idx * cache_blocks_per_batch
            )
            for kv_head in range(num_kv_heads):
                self.emit(
                    f"; PackedKV batch {batch_idx}, selector {kv_head}, "
                    f"K block {batch_k_idx}\n"
                )
                q_group = self.alloc_at(
                    f"_packed_cache_Q_b{batch_idx}_group{kv_head}",
                    query_rows_per_batch,
                    self.mlen,
                    (
                        q_base
                        + kv_head * group_stride
                        + batch_idx * batch_stride
                    ),
                    physical_shape=(rows_per_batch, self.mlen),
                )
                try:
                    self.flash_attention_packed_group(
                        q_group,
                        K_packed,
                        V_packed,
                        group_heads=group_heads,
                        head_slot_dim=head_slot_dim,
                        output_base_address=(
                            output_base_address
                            + kv_head * group_stride
                            + batch_idx * batch_stride
                        ),
                        scratch_base_address=scratch_base_address,
                        broadcast_amount=broadcast_amount,
                        scale=scale,
                        causal_mask=causal_mask,
                        k_idx=batch_k_idx,
                        valid_cols=valid_cols,
                        kv_head_selector=kv_head,
                    )
                finally:
                    self.free_tensor(q_group)

    def init_online_softmax(
        self,
        q_idx: int,
        o_matrix: VRAMMatrixVar,
        rows: int | None = None,
        state_base_address: int | None = None,
        state_stride: int | None = None,
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
            state_base_address=state_base_address,
            state_stride=state_stride,
        )

    def online_softmax_block(
        self,
        s_block: VRAMMatrixVar,
        scale: float,
        rows: int | None = None,
        valid_cols: int | None = None,
        inline_normalize: bool = False,
        state_base_address: int | None = None,
        state_stride: int | None = None,
    ):
        """Perform Online Softmax on S block"""
        super().online_softmax_block(
            s_block_matrix=s_block.name,
            scale=scale,
            rows=rows,
            valid_cols=valid_cols,
            inline_normalize=inline_normalize,
            state_base_address=state_base_address,
            state_stride=state_stride,
        )

    def compute_pv(
        self,
        s_block: VRAMMatrixVar,
        v_input: InputVar,
        k_idx: int,
        pv_matrix: VRAMMatrixVar,
        head_dim: int,
        rows: int | None = None,
        v_element_offset: int = 0,
        prefetch_v: bool = True,
        v_mram_base: int = 0,
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
            v_element_offset=v_element_offset,
            prefetch_v=prefetch_v,
            v_mram_base=v_mram_base,
        )

    def scale_o_row(
        self,
        o_matrix: VRAMMatrixVar,
        q_idx: int,
        rows: int | None = None,
        state_base_address: int | None = None,
        state_stride: int | None = None,
    ):
        """Scale current row block of O by m_res"""
        o_info = super().get_tensor_info(o_matrix.name)
        seq_len, head_dim = o_info.shape

        super().scale_o_row(
            o_matrix=o_matrix.name,
            q_idx=q_idx,
            seq_len=seq_len,
            head_dim=head_dim,
            rows=rows,
            state_base_address=state_base_address,
            state_stride=state_stride,
        )

    def final_scale_o(
        self,
        q_idx: int,
        o_matrix: VRAMMatrixVar,
        rows: int | None = None,
        state_base_address: int | None = None,
        state_stride: int | None = None,
    ):
        """Final scaling: O[q_idx] = O[q_idx] / l"""
        o_info = super().get_tensor_info(o_matrix.name)
        seq_len, head_dim = o_info.shape

        super().final_scale_o(
            q_idx=q_idx,
            o_matrix=o_matrix.name,
            seq_len=seq_len,
            head_dim=head_dim,
            rows=rows,
            state_base_address=state_base_address,
            state_stride=state_stride,
        )


__all__ = ["ProgramAttentionMixin"]
