"""Flash-attention ISA helpers for IsaCompiler."""

from __future__ import annotations

import math

from compiler.asm_templates._imm import load_large_int


class IsaAttentionMixin:
    # =========================================================================
    # Flash Attention Implementation
    # =========================================================================

    def _online_softmax_asm(
        self,
        mlen: int,
        s_address: int,
        m_start_address: int,
        scale: float = 1.0,
        rows: int | None = None,
        valid_cols: int | None = None,
        inline_normalize: bool = False,
    ) -> str:
        """
        Online Softmax Computation.

        Per row of S:
          1. m_curr = max(S[row], m_old)
          2. m_res = exp(m_old - m_curr)              # used to update O downstream
          3. S'[row] = S[row] - m_curr
          4. P[row] = exp(S'[row])
          5. l_new = l_old * m_res + sum(P[row])

        FP SRAM layout (from m_start_address):
          [0, mlen):        m_old / m_curr
          [mlen, 2*mlen):   m_res = exp(m_old - m_curr)
          [2*mlen, 3*mlen): l_old / l_new
        """
        if getattr(self, "unroll_attention", False):
            return self._online_softmax_asm_unrolled(
                mlen=mlen,
                s_address=s_address,
                m_start_address=m_start_address,
                scale=scale,
                valid_cols=valid_cols,
                inline_normalize=inline_normalize,
            )

        gp_regs = self.register_allocator.allocate_gp(5)
        gp_s = gp_regs[0]
        gp_m_addr = gp_regs[1]
        gp_m_res_addr = gp_regs[2]
        gp_l_addr = gp_regs[3]
        gp_loop = gp_regs[4]

        # Fixed FP register allocation for online softmax pipeline.
        # These registers are shared across _online_softmax_asm, _scale_o_asm,
        # and _final_scaling_asm — they MUST remain consistent across all three.
        # WARNING: Do not use f1-f6 in any code that calls these methods.
        fp_m_old = 1  # f1: m_old value
        fp_m_res = 2  # f2: exp(m_old - m_curr)
        fp_l_old = 3  # f3: l_old value
        fp_sum_p = 4  # f4: sum(P)
        fp_scale = 5  # f5: scale factor
        fp_row_max = 6  # f6: current row max (temporary)

        lines = []
        lines.append("; === Online Softmax ===")

        # Set address registers
        lines.extend(load_large_int(gp_s, s_address))
        lines.extend(load_large_int(gp_m_addr, m_start_address))
        lines.append(f"S_ADDI_INT gp{gp_m_res_addr}, gp{gp_m_addr}, {mlen}")
        lines.append(f"S_ADDI_INT gp{gp_l_addr}, gp{gp_m_res_addr}, {mlen}")

        mask_en = 0
        if valid_cols is not None and valid_cols < mlen:
            mask_unit = getattr(self, "hlen", mlen)
            valid_lanes = max(1, math.ceil(valid_cols / mask_unit))
            mask_bits = (1 << valid_lanes) - 1
            lines.append(f"S_ADDI_INT gp{gp_loop}, gp0, {mask_bits}")
            lines.append(f"C_SET_V_MASK_REG gp{gp_loop}")
            mask_en = 1

        # scale factor is pre-loaded at FP SRAM addr 1 by the flash-attention driver.
        if scale != 1.0:
            lines.append(f"S_LD_FP f{fp_scale}, gp0, 1")

        loop_rows = mlen if rows is None else rows
        lines.append(f"C_LOOP_START gp{gp_loop}, {loop_rows}")
        lines.append(f"S_LD_FP f{fp_m_old}, gp{gp_m_addr}, 0")
        lines.append(f"S_ADD_FP f{fp_m_res}, f{fp_m_old}, f0")

        if scale != 1.0:
            lines.append(f"V_MUL_VF gp{gp_s}, gp{gp_s}, f{fp_scale}, {mask_en}")

        # V_RED_MAX accumulates into its destination FP register in the
        # emulator, so clear the per-row max accumulator before each row.
        lines.append(f"S_LD_FP f{fp_row_max}, gp0, 2")
        lines.append(f"V_RED_MAX f{fp_row_max}, gp{gp_s}, {mask_en}")

        # m_curr = max(row_max, m_old) — online softmax must retain the running max.
        lines.append(f"S_MAX_FP f{fp_m_old}, f{fp_row_max}, f{fp_m_old}")

        lines.append(f"S_SUB_FP f{fp_m_res}, f{fp_m_res}, f{fp_m_old}")
        lines.append(f"S_EXP_FP f{fp_m_res}, f{fp_m_res}, 0")

        lines.append(f"S_ST_FP f{fp_m_res}, gp{gp_m_res_addr}, 0")
        lines.append(f"S_ST_FP f{fp_m_old}, gp{gp_m_addr}, 0")

        lines.append(f"V_SUB_VF gp{gp_s}, gp{gp_s}, f{fp_m_old}, {mask_en}, 0")
        lines.append(f"V_EXP_V gp{gp_s}, gp{gp_s}, {mask_en}, 0")

        if inline_normalize:
            # Single-block fast path (parity with _online_softmax_asm_unrolled).
            lines.append(f"S_ADD_FP f{fp_sum_p}, f0, f0")
            lines.append(f"V_RED_SUM f{fp_sum_p}, gp{gp_s}, {mask_en}, 0")
            lines.append(f"S_RECI_FP f{fp_sum_p}, f{fp_sum_p}, 0")
            for _ in range(4):  # retire multi-cycle S_RECI_FP before V_MUL_VF reads it
                lines.append("S_ADDI_INT gp0, gp0, 0")
            lines.append(f"V_MUL_VF gp{gp_s}, gp{gp_s}, f{fp_sum_p}, {mask_en}")
        else:
            lines.append(f"S_LD_FP f{fp_l_old}, gp{gp_l_addr}, 0")

            lines.append(f"S_ADD_FP f{fp_sum_p}, f0, f0")
            lines.append(f"V_RED_SUM f{fp_sum_p}, gp{gp_s}, {mask_en}, 0")

            lines.append(f"S_MUL_FP f{fp_l_old}, f{fp_l_old}, f{fp_m_res}")
            lines.append(f"S_ADD_FP f{fp_l_old}, f{fp_l_old}, f{fp_sum_p}")

            lines.append(f"S_ST_FP f{fp_l_old}, gp{gp_l_addr}, 0")

        lines.append(f"S_ADDI_INT gp{gp_s}, gp{gp_s}, {mlen}")
        lines.append(f"S_ADDI_INT gp{gp_m_addr}, gp{gp_m_addr}, 1")
        lines.append(f"S_ADDI_INT gp{gp_m_res_addr}, gp{gp_m_res_addr}, 1")
        lines.append(f"S_ADDI_INT gp{gp_l_addr}, gp{gp_l_addr}, 1")
        lines.append(f"C_LOOP_END gp{gp_loop}")

        self.register_allocator.free_gp(gp_regs)
        return "\n".join(lines) + "\n"

    def _online_softmax_asm_unrolled(
        self,
        mlen: int,
        s_address: int,
        m_start_address: int,
        scale: float = 1.0,
        valid_cols: int | None = None,
        inline_normalize: bool = False,
    ) -> str:
        """Legacy Python-unrolled online softmax emission, kept for A/B comparisons."""
        gp_regs = self.register_allocator.allocate_gp(5)
        gp_s = gp_regs[0]
        gp_m_addr = gp_regs[1]
        gp_m_res_addr = gp_regs[2]
        gp_l_addr = gp_regs[3]
        gp_mask = gp_regs[4]

        fp_m_old = 1
        fp_m_res = 2
        fp_l_old = 3
        fp_sum_p = 4
        fp_scale = 5
        fp_row_max = 6

        lines = []
        lines.append("; === Online Softmax ===")
        lines.extend(load_large_int(gp_s, s_address))
        lines.extend(load_large_int(gp_m_addr, m_start_address))
        lines.append(f"S_ADDI_INT gp{gp_m_res_addr}, gp{gp_m_addr}, {mlen}")
        lines.append(f"S_ADDI_INT gp{gp_l_addr}, gp{gp_m_res_addr}, {mlen}")

        if scale != 1.0:
            lines.append(f"S_LD_FP f{fp_scale}, gp0, 1")

        mask_en = 0
        if valid_cols is not None and valid_cols < mlen:
            mask_unit = getattr(self, "hlen", mlen)
            valid_lanes = max(1, math.ceil(valid_cols / mask_unit))
            mask_bits = (1 << valid_lanes) - 1
            lines.append(f"S_ADDI_INT gp{gp_mask}, gp0, {mask_bits}")
            lines.append(f"C_SET_V_MASK_REG gp{gp_mask}")
            mask_en = 1

        for row in range(mlen):
            lines.append(f"; Row {row}")
            lines.append(f"S_LD_FP f{fp_m_old}, gp{gp_m_addr}, {row}")
            lines.append(f"S_ADD_FP f{fp_m_res}, f{fp_m_old}, f0")

            if scale != 1.0:
                lines.append(f"V_MUL_VF gp{gp_s}, gp{gp_s}, f{fp_scale}, {mask_en}")

            lines.append(f"S_LD_FP f{fp_row_max}, gp0, 2")
            lines.append(f"V_RED_MAX f{fp_row_max}, gp{gp_s}, {mask_en}")

            # m_curr = max(row_max, m_old) — online softmax must retain the running max.
            lines.append(f"S_MAX_FP f{fp_m_old}, f{fp_row_max}, f{fp_m_old}")

            lines.append(f"S_SUB_FP f{fp_m_res}, f{fp_m_res}, f{fp_m_old}")
            lines.append(f"S_EXP_FP f{fp_m_res}, f{fp_m_res}, 0")

            lines.append(f"S_ST_FP f{fp_m_res}, gp{gp_m_res_addr}, {row}")
            lines.append(f"S_ST_FP f{fp_m_old}, gp{gp_m_addr}, {row}")

            lines.append(f"V_SUB_VF gp{gp_s}, gp{gp_s}, f{fp_m_old}, {mask_en}, 0")
            lines.append(f"V_EXP_V gp{gp_s}, gp{gp_s}, {mask_en}, 0")

            if inline_normalize:
                # Single-block fast path: normalise P register-direct (no FP-SRAM l
                # round-trip, whose per-row store is racy in the unrolled kernel and
                # PC-pins when looped). softmax(S)=exp(S-m)/sum(exp(S-m)); for one KV
                # block this also gives correct attention O since (P/l)@V == (P@V)/l.
                lines.append(f"S_ADD_FP f{fp_sum_p}, f0, f0")
                lines.append(f"V_RED_SUM f{fp_sum_p}, gp{gp_s}, {mask_en}, 0")
                lines.append(f"S_RECI_FP f{fp_sum_p}, f{fp_sum_p}, 0")
                for _ in range(4):  # retire multi-cycle S_RECI_FP before V_MUL_VF reads it
                    lines.append("S_ADDI_INT gp0, gp0, 0")
                lines.append(f"V_MUL_VF gp{gp_s}, gp{gp_s}, f{fp_sum_p}, {mask_en}")
            else:
                lines.append(f"S_LD_FP f{fp_l_old}, gp{gp_l_addr}, {row}")

                lines.append(f"S_ADD_FP f{fp_sum_p}, f0, f0")
                lines.append(f"V_RED_SUM f{fp_sum_p}, gp{gp_s}, {mask_en}, 0")

                lines.append(f"S_MUL_FP f{fp_l_old}, f{fp_l_old}, f{fp_m_res}")
                lines.append(f"S_ADD_FP f{fp_l_old}, f{fp_l_old}, f{fp_sum_p}")

                lines.append(f"S_ST_FP f{fp_l_old}, gp{gp_l_addr}, {row}")

            lines.append(f"S_ADDI_INT gp{gp_s}, gp{gp_s}, {mlen}")

        self.register_allocator.free_gp(gp_regs)
        return "\n".join(lines) + "\n"

    def _pv_multiply_asm(
        self,
        mlen: int,
        blen: int,
        head_dim: int,
        p_address: int,
        v_hbm_offset_reg: int,
        v_hbm_offset: int,
        pv_address: int,
        rows: int | None = None,
        pv_physical_rows: int | None = None,
    ) -> str:
        """
        Compute PV = P @ V via M_MM.

        P:  (mlen, mlen)     in VRAM   (softmax output)
        V:  (mlen, head_dim) in HBM    (prefetched into MSRAM in mlen-wide column blocks)
        PV: (mlen, head_dim) in VRAM

        M_MM computes one (blen, mlen) @ (mlen, blen) -> (blen, blen) in a single op
        (K=mlen done in one shot). For head_dim > mlen, V is split into head_dim/mlen
        column blocks; the outer loop iterates blocks, middle loop iterates blen-wide
        V columns within a block, inner loop iterates blen-wide P rows.
        """
        # PV is stored column-block-major with `pv_physical_rows` rows (= min(mlen,
        # seq_len) for the decoder), so each head_dim col-block spans
        # pv_physical_rows*mlen, not mlen*mlen. They coincide at seq>=mlen, but for
        # seq<mlen using mlen*mlen writes col-blocks 1.. out of bounds, leaving
        # head_dim cols mlen.. zero (PV shrinks by mlen/head_dim).
        if pv_physical_rows is None:
            pv_physical_rows = mlen

        if getattr(self, "unroll_attention", False):
            return self._pv_multiply_asm_unrolled(
                mlen=mlen,
                blen=blen,
                head_dim=head_dim,
                p_address=p_address,
                v_hbm_offset_reg=v_hbm_offset_reg,
                v_hbm_offset=v_hbm_offset,
                pv_address=pv_address,
                pv_physical_rows=pv_physical_rows,
            )

        gp_regs = self.register_allocator.allocate_gp(8)
        gp_p = gp_regs[0]
        gp_v = gp_regs[1]
        gp_pv = gp_regs[2]
        gp_hbm = gp_regs[3]
        gp_stride = gp_regs[4]
        gp_pv_col_base = gp_regs[5]
        gp_v_loop = gp_regs[6]
        gp_p_loop = gp_regs[7]

        num_v_col_blocks = max(1, math.ceil(head_dim / mlen))
        tiles_per_mlen = mlen // blen
        p_row_groups = tiles_per_mlen if rows is None else max(1, min(tiles_per_mlen, (rows + blen - 1) // blen))

        lines = []
        lines.append("; === PV Multiply (P @ V) using M_MM ===")
        lines.append(f"; P: ({mlen}, {mlen}) @ V: ({mlen}, {head_dim}) -> PV: ({mlen}, {head_dim})")
        lines.append("; M_MM: (blen, mlen) @ (mlen, blen) -> (blen, blen), K=mlen in one shot")
        lines.append(f"; V split into {num_v_col_blocks} column blocks of width {mlen}")
        lines.append("; Storage layout: (batch, mlen, hidden/mlen), column-block major")

        # STRIDE was set to mlen by the flash-attention driver — do not overwrite it here.
        # M_MM_WO requires a nonzero stride reg (gp0=0 would be interpreted as stride=1).
        # With column-block-major storage, consecutive rows within a column block are
        # adjacent, so the writeback stride = 1.
        lines.append(f"S_ADDI_INT gp{gp_stride}, gp0, 1")

        for v_col_block in range(num_v_col_blocks):
            lines.append(
                f"; --- V column block {v_col_block} (columns {v_col_block * mlen} to {(v_col_block + 1) * mlen - 1}) ---"
            )

            # Prefetch V[:, v_col_block*mlen:(v_col_block+1)*mlen] (mlen × mlen) to MSRAM.
            # V is row-major in HBM: V[row, col] at offset row*head_dim + col, so the
            # column-block base offset = v_hbm_offset + v_col_block * mlen (elements).
            v_block_hbm_offset = v_hbm_offset + v_col_block * mlen
            lines.append(f"S_ADDI_INT gp{gp_v}, gp0, 0")
            lines.extend(load_large_int(gp_hbm, v_block_hbm_offset))
            # funct1=0 => PREFETCH_M_H (High Precision, m_controller_precision_select=0),
            # matching how V is staged in HBM (same High-Precision MXINT format as the K/W
            # weights, which load via _H). funct1=1 (_L / Low Precision) mis-decodes the
            # High-staged V into garbage (~constant mantissas) -> rank-1 P@V collapse.
            lines.append(f"H_PREFETCH_M gp{gp_v}, gp{gp_hbm}, a{v_hbm_offset_reg}, 1, 0")

            pv_col_block_base = pv_address + v_col_block * pv_physical_rows * mlen
            lines.extend(load_large_int(gp_pv_col_base, pv_col_block_base))
            # Only cover the output columns this block actually has. head_dim < mlen
            # (e.g. GQA head_slot_dim=4) would otherwise over-iterate V's zero-pad
            # columns, writing garbage into PV cols head_dim..mlen (spilled by the pack).
            block_cols = min(head_dim - v_col_block * mlen, mlen)
            v_col_tiles = max(1, math.ceil(block_cols / blen))
            lines.append(f"C_LOOP_START gp{gp_v_loop}, {v_col_tiles}")
            lines.extend(load_large_int(gp_p, p_address))
            lines.append(f"S_ADDI_INT gp{gp_pv}, gp{gp_pv_col_base}, 0")
            lines.append(f"C_LOOP_START gp{gp_p_loop}, {p_row_groups}")
            lines.append(f"M_MM 0, gp{gp_v}, gp{gp_p}")
            lines.append(f"M_MM_WO gp{gp_pv}, gp{gp_stride}, 0")
            lines.append(f"S_ADDI_INT gp{gp_p}, gp{gp_p}, {blen * mlen}")
            lines.append(f"S_ADDI_INT gp{gp_pv}, gp{gp_pv}, {blen * mlen}")
            lines.append(f"C_LOOP_END gp{gp_p_loop}")
            lines.append(f"S_ADDI_INT gp{gp_v}, gp{gp_v}, {blen}")
            lines.append(f"S_ADDI_INT gp{gp_pv_col_base}, gp{gp_pv_col_base}, {blen}")
            lines.append(f"C_LOOP_END gp{gp_v_loop}")

        self.register_allocator.free_gp(gp_regs)
        return "\n".join(lines) + "\n"

    def _pv_multiply_asm_unrolled(
        self,
        mlen: int,
        blen: int,
        head_dim: int,
        p_address: int,
        v_hbm_offset_reg: int,
        v_hbm_offset: int,
        pv_address: int,
        pv_physical_rows: int | None = None,
    ) -> str:
        """Legacy Python-unrolled P @ V emission, kept for A/B comparisons."""
        if pv_physical_rows is None:
            pv_physical_rows = mlen
        gp_regs = self.register_allocator.allocate_gp(5)
        gp_p = gp_regs[0]
        gp_v = gp_regs[1]
        gp_pv = gp_regs[2]
        gp_hbm = gp_regs[3]
        gp_stride = gp_regs[4]

        num_v_col_blocks = max(1, math.ceil(head_dim / mlen))

        lines = []
        lines.append("; === PV Multiply (P @ V) using M_MM ===")
        lines.append(f"; P: ({mlen}, {mlen}) @ V: ({mlen}, {head_dim}) -> PV: ({mlen}, {head_dim})")
        lines.append("; M_MM: (blen, mlen) @ (mlen, blen) -> (blen, blen), K=mlen in one shot")
        lines.append(f"; V split into {num_v_col_blocks} column blocks of width {mlen}")
        lines.append("; Storage layout: (batch, mlen, hidden/mlen), column-block major")
        lines.append(f"S_ADDI_INT gp{gp_stride}, gp0, 1")

        for v_col_block in range(num_v_col_blocks):
            lines.append(
                f"; --- V column block {v_col_block} (columns {v_col_block * mlen} to {(v_col_block + 1) * mlen - 1}) ---"
            )
            v_block_hbm_offset = v_hbm_offset + v_col_block * mlen
            lines.append(f"S_ADDI_INT gp{gp_v}, gp0, 0")
            lines.extend(load_large_int(gp_hbm, v_block_hbm_offset))
            # funct1=0 => PREFETCH_M_H (High Precision), matching V's HBM staging; _L
            # (Low Precision) mis-decodes it to garbage. See _pv_multiply_asm above.
            lines.append(f"H_PREFETCH_M gp{gp_v}, gp{gp_hbm}, a{v_hbm_offset_reg}, 1, 0")

            # Only emit output-column tiles that this block actually covers. When
            # head_dim < mlen (e.g. GQA head_slot_dim=4), mlen//blen would over-iterate
            # over V's zero-padded columns, writing GARBAGE into PV cols head_dim..mlen
            # that the pack (V_SHFT_V + V_ADD_VV over all lanes) then spills across heads.
            block_cols = min(head_dim - v_col_block * mlen, mlen)
            v_col_tiles = max(1, math.ceil(block_cols / blen))
            for v_col in range(v_col_tiles):
                lines.append(f"; V column {v_col_block * mlen + v_col * blen}")
                v_msram_offset = v_col * blen
                lines.append(f"S_ADDI_INT gp{gp_v}, gp0, {v_msram_offset}")

                for p_row in range(mlen // blen):
                    p_row_addr = p_address + p_row * blen * mlen
                    lines.extend(load_large_int(gp_p, p_row_addr))
                    lines.append(f"M_MM 0, gp{gp_v}, gp{gp_p}")

                    pv_offset = v_col_block * pv_physical_rows * mlen + p_row * blen * mlen + v_col * blen
                    lines.extend(load_large_int(gp_pv, pv_address + pv_offset))
                    lines.append(f"M_MM_WO gp{gp_pv}, gp{gp_stride}, 0")

        self.register_allocator.free_gp(gp_regs)
        return "\n".join(lines) + "\n"

    def _scale_o_asm(
        self,
        mlen: int,
        head_dim: int,
        seq_len: int,
        m_res_address: int,
        o_address: int,
        row_offset: int = 0,
        rows: int | None = None,
    ) -> str:
        """Scale each row of O by m_res: O[row] *= m_res[row]."""
        if getattr(self, "unroll_attention", False):
            return self._scale_o_asm_unrolled(
                mlen=mlen,
                head_dim=head_dim,
                seq_len=seq_len,
                m_res_address=m_res_address,
                o_address=o_address,
                row_offset=row_offset,
            )

        gp_regs = self.register_allocator.allocate_gp(4)
        gp_m_res = gp_regs[0]
        gp_o_row_base = gp_regs[1]
        gp_o = gp_regs[2]
        gp_row_loop = gp_regs[3]
        fp_m_res = 1

        num_col_blocks = max(1, math.ceil(head_dim / mlen))
        loop_rows = mlen if rows is None else rows

        lines = []
        lines.append("; === Scale O by m_res ===")
        lines.append(f"; head_dim = {head_dim}, {num_col_blocks} mlen-blocks per row")
        lines.append(f"; seq_len = {seq_len}, row_offset = {row_offset}")

        if num_col_blocks == 1:
            o_addr = o_address + row_offset * mlen
            lines.extend(load_large_int(gp_m_res, m_res_address))
            lines.extend(load_large_int(gp_o, o_addr))
            lines.append(f"C_LOOP_START gp{gp_row_loop}, {loop_rows}")
            lines.append(f"S_LD_FP f{fp_m_res}, gp{gp_m_res}, 0")
            lines.append(f"V_MUL_VF gp{gp_o}, gp{gp_o}, f{fp_m_res}, 0")
            lines.append(f"S_ADDI_INT gp{gp_m_res}, gp{gp_m_res}, 1")
            lines.append(f"S_ADDI_INT gp{gp_o}, gp{gp_o}, {mlen}")
            lines.append(f"C_LOOP_END gp{gp_row_loop}")
        else:
            gp_col_loop = self.register_allocator.allocate_gp(1)[0]
            o_addr = o_address + row_offset * mlen
            lines.extend(load_large_int(gp_m_res, m_res_address))
            lines.extend(load_large_int(gp_o_row_base, o_addr))
            lines.append(f"C_LOOP_START gp{gp_row_loop}, {loop_rows}")
            lines.append(f"S_LD_FP f{fp_m_res}, gp{gp_m_res}, 0")
            lines.append(f"S_ADDI_INT gp{gp_o}, gp{gp_o_row_base}, 0")
            lines.append(f"C_LOOP_START gp{gp_col_loop}, {num_col_blocks}")
            lines.append(f"V_MUL_VF gp{gp_o}, gp{gp_o}, f{fp_m_res}, 0")
            lines.append(f"S_ADDI_INT gp{gp_o}, gp{gp_o}, {seq_len * mlen}")
            lines.append(f"C_LOOP_END gp{gp_col_loop}")
            lines.append(f"S_ADDI_INT gp{gp_m_res}, gp{gp_m_res}, 1")
            lines.append(f"S_ADDI_INT gp{gp_o_row_base}, gp{gp_o_row_base}, {mlen}")
            lines.append(f"C_LOOP_END gp{gp_row_loop}")
            self.register_allocator.free_gp([gp_col_loop])

        self.register_allocator.free_gp(gp_regs)
        return "\n".join(lines) + "\n"

    def _scale_o_asm_unrolled(
        self,
        mlen: int,
        head_dim: int,
        seq_len: int,
        m_res_address: int,
        o_address: int,
        row_offset: int = 0,
    ) -> str:
        """Legacy Python-unrolled O *= m_res emission, kept for A/B comparisons."""
        gp_regs = self.register_allocator.allocate_gp(2)
        gp_m_res = gp_regs[0]
        gp_o = gp_regs[1]
        fp_m_res = 1

        num_col_blocks = max(1, math.ceil(head_dim / mlen))

        lines = []
        lines.append("; === Scale O by m_res ===")
        lines.append(f"; head_dim = {head_dim}, {num_col_blocks} mlen-blocks per row")
        lines.append(f"; seq_len = {seq_len}, row_offset = {row_offset}")
        lines.extend(load_large_int(gp_m_res, m_res_address))

        for row in range(mlen):
            lines.append(f"S_LD_FP f{fp_m_res}, gp{gp_m_res}, {row}")
            actual_row = row_offset + row

            for col_block in range(num_col_blocks):
                o_addr = o_address + col_block * seq_len * mlen + actual_row * mlen
                lines.extend(load_large_int(gp_o, o_addr))
                lines.append(f"V_MUL_VF gp{gp_o}, gp{gp_o}, f{fp_m_res}, 0")

        self.register_allocator.free_gp(gp_regs)
        return "\n".join(lines) + "\n"

    def _add_pv_to_o_asm(
        self,
        mlen: int,
        head_dim: int,
        seq_len: int,
        pv_address: int,
        o_address: int,
        row_offset: int = 0,
    ) -> str:
        """Accumulate PV into O: O[row] += PV[row]."""
        gp_regs = self.register_allocator.allocate_gp(2)
        gp_o = gp_regs[0]
        gp_pv = gp_regs[1]

        num_col_blocks = max(1, math.ceil(head_dim / mlen))

        lines = []
        lines.append("; === Add PV to O ===")
        lines.append(f"; head_dim = {head_dim}, {num_col_blocks} mlen-blocks per row")
        lines.append(f"; seq_len = {seq_len}, row_offset = {row_offset}")

        for row in range(mlen):
            actual_row = row_offset + row

            for col_block in range(num_col_blocks):
                o_addr = o_address + col_block * seq_len * mlen + actual_row * mlen
                pv_addr = pv_address + col_block * mlen * mlen + row * mlen

                lines.extend(load_large_int(gp_o, o_addr))
                lines.extend(load_large_int(gp_pv, pv_addr))
                lines.append(f"V_ADD_VV gp{gp_o}, gp{gp_o}, gp{gp_pv}, 0")

        self.register_allocator.free_gp(gp_regs)
        return "\n".join(lines) + "\n"

    def _final_scaling_asm(
        self,
        mlen: int,
        head_dim: int,
        seq_len: int,
        l_address: int,
        o_address: int,
        row_offset: int = 0,
        rows: int | None = None,
    ) -> str:
        """
        Final scaling: O[row] /= l[row].

        V_MUL_VF processes mlen elements at a time; when head_dim > mlen,
        each row is split into head_dim // mlen mlen-wide blocks.
        """
        if getattr(self, "unroll_attention", False):
            return self._final_scaling_asm_unrolled(
                mlen=mlen,
                head_dim=head_dim,
                seq_len=seq_len,
                l_address=l_address,
                o_address=o_address,
                row_offset=row_offset,
            )

        gp_regs = self.register_allocator.allocate_gp(4)
        gp_l = gp_regs[0]
        gp_o_row_base = gp_regs[1]
        gp_o = gp_regs[2]
        gp_row_loop = gp_regs[3]
        fp_l = 1

        num_col_blocks = max(1, math.ceil(head_dim / mlen))
        loop_rows = mlen if rows is None else rows

        lines = []
        lines.append("; === Final Scaling O = O / l ===")
        lines.append(f"; head_dim = {head_dim}, {num_col_blocks} mlen-blocks per row")
        lines.append("; Storage layout: (seq_len, mlen, head_dim/mlen), column-block major")
        lines.append(f"; seq_len = {seq_len}, row_offset = {row_offset}")

        if num_col_blocks == 1:
            o_addr = o_address + row_offset * mlen
            lines.extend(load_large_int(gp_l, l_address))
            lines.extend(load_large_int(gp_o, o_addr))
            lines.append(f"C_LOOP_START gp{gp_row_loop}, {loop_rows}")
            lines.append(f"S_LD_FP f{fp_l}, gp{gp_l}, 0")
            lines.append(f"S_RECI_FP f{fp_l}, f{fp_l}, 0")
            lines.append(f"V_MUL_VF gp{gp_o}, gp{gp_o}, f{fp_l}, 0")
            lines.append(f"S_ADDI_INT gp{gp_l}, gp{gp_l}, 1")
            lines.append(f"S_ADDI_INT gp{gp_o}, gp{gp_o}, {mlen}")
            lines.append(f"C_LOOP_END gp{gp_row_loop}")
        else:
            gp_col_loop = self.register_allocator.allocate_gp(1)[0]
            o_addr = o_address + row_offset * mlen
            lines.extend(load_large_int(gp_l, l_address))
            lines.extend(load_large_int(gp_o_row_base, o_addr))
            lines.append(f"C_LOOP_START gp{gp_row_loop}, {loop_rows}")
            lines.append(f"S_LD_FP f{fp_l}, gp{gp_l}, 0")
            lines.append(f"S_RECI_FP f{fp_l}, f{fp_l}, 0")
            lines.append(f"S_ADDI_INT gp{gp_o}, gp{gp_o_row_base}, 0")
            lines.append(f"C_LOOP_START gp{gp_col_loop}, {num_col_blocks}")
            lines.append(f"V_MUL_VF gp{gp_o}, gp{gp_o}, f{fp_l}, 0")
            lines.append(f"S_ADDI_INT gp{gp_o}, gp{gp_o}, {seq_len * mlen}")
            lines.append(f"C_LOOP_END gp{gp_col_loop}")
            lines.append(f"S_ADDI_INT gp{gp_l}, gp{gp_l}, 1")
            lines.append(f"S_ADDI_INT gp{gp_o_row_base}, gp{gp_o_row_base}, {mlen}")
            lines.append(f"C_LOOP_END gp{gp_row_loop}")
            self.register_allocator.free_gp([gp_col_loop])

        self.register_allocator.free_gp(gp_regs)
        return "\n".join(lines) + "\n"

    def _final_scaling_asm_unrolled(
        self,
        mlen: int,
        head_dim: int,
        seq_len: int,
        l_address: int,
        o_address: int,
        row_offset: int = 0,
    ) -> str:
        """Legacy Python-unrolled final O /= l emission, kept for A/B comparisons."""
        gp_regs = self.register_allocator.allocate_gp(2)
        gp_l = gp_regs[0]
        gp_o = gp_regs[1]
        fp_l = 1

        num_col_blocks = max(1, math.ceil(head_dim / mlen))

        lines = []
        lines.append("; === Final Scaling O = O / l ===")
        lines.append(f"; head_dim = {head_dim}, {num_col_blocks} mlen-blocks per row")
        lines.append("; Storage layout: (seq_len, mlen, head_dim/mlen), column-block major")
        lines.append(f"; seq_len = {seq_len}, row_offset = {row_offset}")
        lines.extend(load_large_int(gp_l, l_address))

        for row in range(mlen):
            lines.append(f"S_LD_FP f{fp_l}, gp{gp_l}, {row}")
            lines.append(f"S_RECI_FP f{fp_l}, f{fp_l}, 0")
            actual_row = row_offset + row

            for col_block in range(num_col_blocks):
                o_addr = o_address + col_block * seq_len * mlen + actual_row * mlen
                lines.extend(load_large_int(gp_o, o_addr))
                lines.append(f"V_MUL_VF gp{gp_o}, gp{gp_o}, f{fp_l}, 0")

        self.register_allocator.free_gp(gp_regs)
        return "\n".join(lines) + "\n"

    def _reset_fpsram_asm(
        self,
        start_address: int,
        count: int,
        value_address: int,  # FP SRAM slot: 0 = zero, 2 = -inf
    ) -> str:
        """Reset a region of FP SRAM to the value at value_address."""
        gp_regs = self.register_allocator.allocate_gp(2)
        gp_addr = gp_regs[0]
        gp_loop = gp_regs[1]

        lines = []
        lines.append(f"; Reset FP SRAM [{start_address}, {start_address + count})")

        lines.append(f"S_ADDI_INT gp{gp_addr}, gp0, {start_address}")
        # Use f1 for FP scalar - FP registers don't go through GP allocator
        lines.append(f"S_LD_FP f1, gp0, {value_address}")

        if getattr(self, "unroll_attention", False):
            for i in range(count):
                lines.append(f"S_ST_FP f1, gp{gp_addr}, {i}")
        else:
            lines.append(f"C_LOOP_START gp{gp_loop}, {count}")
            lines.append(f"S_ST_FP f1, gp{gp_addr}, 0")
            lines.append(f"S_ADDI_INT gp{gp_addr}, gp{gp_addr}, 1")
            lines.append(f"C_LOOP_END gp{gp_loop}")

        self.register_allocator.free_gp(gp_regs)
        return "\n".join(lines) + "\n"

    def _reset_vram_asm(
        self,
        start_address: int,
        rows: int,
        cols: int,
        total_rows: int,
        mlen: int = 64,
        row_offset: int = 0,
    ) -> str:
        """
        Reset a region of VRAM to zero.

        V_MUL_VF processes mlen elements at a time; when cols > mlen, each
        row is split into cols // mlen mlen-wide blocks.
        """
        gp_regs = self.register_allocator.allocate_gp(2)
        gp_addr = gp_regs[0]
        gp_loop = gp_regs[1]

        num_col_blocks = (cols + mlen - 1) // mlen

        lines = []
        lines.append(f"; Reset VRAM rows [{row_offset}, {row_offset + rows}) of matrix at {start_address}")
        lines.append(f"; {rows} rows x {cols} cols, {num_col_blocks} blocks per row")
        lines.append("; Storage layout: (total_rows, mlen, cols/mlen), column-block major")
        lines.append(f"; total_rows = {total_rows}, row_offset = {row_offset}")

        if getattr(self, "unroll_attention", False):
            for row in range(rows):
                actual_row = row_offset + row
                for col_block in range(num_col_blocks):
                    addr = start_address + col_block * total_rows * mlen + actual_row * mlen
                    lines.extend(load_large_int(gp_addr, addr))
                    lines.append(f"V_MUL_VF gp{gp_addr}, gp{gp_addr}, f0, 0")
        else:
            for col_block in range(num_col_blocks):
                addr = start_address + col_block * total_rows * mlen + row_offset * mlen
                lines.append(f"; Column block {col_block}")
                lines.extend(load_large_int(gp_addr, addr))
                lines.append(f"C_LOOP_START gp{gp_loop}, {rows}")
                lines.append(f"V_MUL_VF gp{gp_addr}, gp{gp_addr}, f0, 0")
                lines.append(f"S_ADDI_INT gp{gp_addr}, gp{gp_addr}, {mlen}")
                lines.append(f"C_LOOP_END gp{gp_loop}")

        self.register_allocator.free_gp(gp_regs)
        return "\n".join(lines) + "\n"

    # =========================================================================
    # Expanded Flash Attention Operations
    # =========================================================================

    def init_online_softmax(
        self,
        q_idx: int,
        o_matrix: str,
        seq_len: int,
        head_dim: int,
        rows: int | None = None,
    ) -> str:
        """
        Initialize Online Softmax state for Q block q_idx:
          m_old = -inf (FP SRAM), l = 0 (FP SRAM), O_row = 0 (VRAM).
        """
        fp_sram_start = self._ONLINE_SOFTMAX_FPSRAM_BASE
        m_old_addr = fp_sram_start
        l_addr = fp_sram_start + 2 * self.mlen  # skip m_res region

        o_info = self[o_matrix]
        o_vram_addr = o_info.vram_addr
        physical_seq_len = o_info.physical_shape[0]
        row_offset = q_idx * self.mlen

        isa_code = f"; === Init Online Softmax for Q block {q_idx} ===\n"

        isa_code += self._reset_fpsram_asm(m_old_addr, self.mlen, 2)  # slot 2 = -inf
        isa_code += self._reset_fpsram_asm(l_addr, self.mlen, 0)  # slot 0 = 0.0
        isa_code += self._reset_vram_asm(
            start_address=o_vram_addr,
            rows=self.mlen if rows is None else rows,
            cols=head_dim,
            total_rows=physical_seq_len,
            mlen=self.mlen,
            row_offset=row_offset,
        )

        return self._emit(isa_code)

    def online_softmax_block(
        self,
        s_block_matrix: str,
        scale: float,
        rows: int | None = None,
        valid_cols: int | None = None,
        inline_normalize: bool = False,
    ) -> str:
        """
        Run Online Softmax on one S block.
          Input:   S_block (mlen × mlen) in VRAM
          Output:  P (mlen × mlen) in-place in VRAM
          Updates: m_old, m_res, l in FP SRAM
          ``scale`` is the QK^T scaling factor (typically 1/sqrt(d)).
          ``inline_normalize`` (single-block only): normalise P by its row-sum
          register-direct here and skip the FP-SRAM l/final-scale path.
        """
        s_info = self[s_block_matrix]
        s_address = s_info.vram_addr

        fp_sram_start = self._ONLINE_SOFTMAX_FPSRAM_BASE
        m_start_address = fp_sram_start

        isa_code = f"; === Online Softmax Block {s_block_matrix} ===\n"
        isa_code += self._online_softmax_asm(
            mlen=self.mlen,
            s_address=s_address,
            m_start_address=m_start_address,
            scale=scale,
            rows=rows,
            valid_cols=valid_cols,
            inline_normalize=inline_normalize,
        )

        return self._emit(isa_code)

    def compute_pv(
        self,
        s_block_matrix: str,
        v_sub_matrix: str,
        k_idx: int,
        pv_matrix: str,
        head_dim: int,
        rows: int | None = None,
    ) -> str:
        """
        Compute PV = P @ V[k_idx].

        P lives in s_block_matrix (softmax result); V is prefetched from
        HBM; PV is written to VRAM via pv_matrix.
        """
        s_info = self[s_block_matrix]
        p_address = s_info.vram_addr

        pv_info = self[pv_matrix]
        pv_address = pv_info.vram_addr

        v_layout = self.get_hbm_layout(v_sub_matrix)
        physical_head_dim = (v_layout.physical_shape or v_layout.full_shape)[1]
        v_hbm_offset = k_idx * self.mlen * physical_head_dim

        isa_code = f"; === Compute PV = P @ V[k_idx={k_idx}] ===\n"

        addr_regs = self.register_allocator.allocate_addr(1)
        v_hbm_reg = addr_regs[0]
        gp_regs = self.register_allocator.allocate_gp(2)

        from compiler.asm_templates import preload_addr_reg_asm

        isa_code += preload_addr_reg_asm(
            addr_reg_to_set=[v_hbm_reg], available_registers=gp_regs, addr_reg_val=[v_layout.hbm_base_addr]
        )

        isa_code += self._pv_multiply_asm(
            mlen=self.mlen,
            blen=self.blen,
            head_dim=head_dim,
            p_address=p_address,
            v_hbm_offset_reg=v_hbm_reg,
            v_hbm_offset=v_hbm_offset,
            pv_address=pv_address,
            rows=rows,
            pv_physical_rows=pv_info.physical_shape[0],
        )

        self.register_allocator.free_gp(gp_regs)
        self.register_allocator.free_addr(addr_regs)

        return self._emit(isa_code)

    def warm_v_prefetch(self, v_sub_matrix: str, k_idx: int) -> str:
        """Prefetch V[k_idx] into MSRAM ONCE (High precision) before the per-head P@V
        loop so it has fully landed by head 0's first tile. Paired with the RTL
        cold-start re-prime at the M_BTMM->M_MM boundary: the re-prime re-warms the
        weight-read pipe, but its ~20-cycle spin can outrun head 0's own (late) per-head
        prefetch, leaving row 0 reading stale K. Landing V here (during head 0's softmax)
        guarantees the spin reads V for row 0 too."""
        v_layout = self.get_hbm_layout(v_sub_matrix)
        physical_head_dim = (v_layout.physical_shape or v_layout.full_shape)[1]
        v_hbm_offset = k_idx * self.mlen * physical_head_dim
        addr_regs = self.register_allocator.allocate_addr(1)
        v_hbm_reg = addr_regs[0]
        gp_regs = self.register_allocator.allocate_gp(2)

        from compiler.asm_templates import preload_addr_reg_asm

        isa_code = "; === Warm V into MSRAM once (head-0 row-0 stale-K guard) ===\n"
        isa_code += preload_addr_reg_asm(
            addr_reg_to_set=[v_hbm_reg], available_registers=gp_regs, addr_reg_val=[v_layout.hbm_base_addr]
        )
        isa_code += f"S_ADDI_INT gp{gp_regs[0]}, gp0, 0\n"
        isa_code += "\n".join(load_large_int(gp_regs[1], v_hbm_offset)) + "\n"
        isa_code += f"H_PREFETCH_M gp{gp_regs[0]}, gp{gp_regs[1]}, a{v_hbm_reg}, 1, 0\n"

        self.register_allocator.free_gp(gp_regs)
        self.register_allocator.free_addr(addr_regs)
        return self._emit(isa_code)

    def scale_o_row(
        self,
        o_matrix: str,
        q_idx: int,
        seq_len: int,
        head_dim: int,
        rows: int | None = None,
    ) -> str:
        """Scale the current row block of O by m_res: O[q_idx] *= m_res."""
        o_info = self[o_matrix]
        o_address = o_info.vram_addr
        physical_seq_len = o_info.physical_shape[0]

        fp_sram_start = self._ONLINE_SOFTMAX_FPSRAM_BASE
        m_res_addr = fp_sram_start + self.mlen

        row_offset = q_idx * self.mlen

        isa_code = f"; === Scale O[q_idx={q_idx}] by m_res ===\n"
        isa_code += self._scale_o_asm(
            mlen=self.mlen,
            head_dim=head_dim,
            seq_len=physical_seq_len,
            m_res_address=m_res_addr,
            o_address=o_address,
            row_offset=row_offset,
            rows=rows,
        )

        return self._emit(isa_code)

    def final_scale_o(
        self,
        q_idx: int,
        o_matrix: str,
        seq_len: int,
        head_dim: int,
        rows: int | None = None,
    ) -> str:
        """Final scaling: O[q_idx] /= l."""
        o_info = self[o_matrix]
        o_address = o_info.vram_addr
        physical_seq_len = o_info.physical_shape[0]

        fp_sram_start = self._ONLINE_SOFTMAX_FPSRAM_BASE
        l_addr = fp_sram_start + 2 * self.mlen

        row_offset = q_idx * self.mlen

        isa_code = f"; === Final Scale O for Q block {q_idx} ===\n"
        isa_code += self._final_scaling_asm(
            mlen=self.mlen,
            head_dim=head_dim,
            seq_len=physical_seq_len,
            l_address=l_addr,
            o_address=o_address,
            row_offset=row_offset,
            rows=rows,
        )

        return self._emit(isa_code)


__all__ = ["IsaAttentionMixin"]
