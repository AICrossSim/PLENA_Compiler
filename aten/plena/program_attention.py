"""Flash-attention operations for the PLENA program builder."""

from __future__ import annotations

import math

from compiler.asm_templates import preload_addr_reg_asm
from compiler.asm_templates.flashattn import flash_attn_asm
from compiler.aten.plena.vars import InputVar, VRAMMatrixVar


class ProgramAttentionMixin:
    # ========================================================================
    # Flash Attention Operations
    # ========================================================================

    def flash_attention(self, Q, K, V, scale=None, hq=1, hkv=1, h_qkv=None, causal_mask=None):
        """Emit flash attention, dispatching to MHA or fused GQA codegen by shape."""
        if hq == 1 and hkv == 1:
            return self._flash_attention_mha(Q, K, V, scale, causal_mask=causal_mask)

        if h_qkv is None:
            raise ValueError("GQA mode requires h_qkv to be specified")
        if causal_mask is not None:
            raise NotImplementedError("causal_mask is not yet supported for GQA flash attention")
        return self._flash_attention_gqa_fused(Q, K, V, scale, hq, hkv, h_qkv)

    def _flash_attention_mha(self, Q, K, V, scale, causal_mask=None):
        """Single-head online-softmax flash attention using compiler primitives."""
        seq_len, head_dim = Q.shape
        mlen = self.mlen

        if scale is None:
            scale = 1.0 / math.sqrt(head_dim)

        num_q_blocks = seq_len // mlen
        num_k_blocks = seq_len // mlen

        S_block = self.alloc("S", mlen, mlen)
        PV = self.alloc("PV", mlen, head_dim)
        O = self.alloc("O", seq_len, head_dim)

        for q_idx in range(num_q_blocks):
            self.init_online_softmax(q_idx, O)

            for k_idx in range(num_k_blocks):
                self.vram_sub_projection_T_to(
                    Q,
                    q_idx,
                    K,
                    k_idx,
                    S_block,
                    target_row_idx=0,
                    target_col_idx=0,
                )
                if causal_mask is not None:
                    self.vram_add(S_block, causal_mask)
                self.online_softmax_block(S_block, scale)
                self.compute_pv(S_block, V, k_idx, PV, head_dim)
                self.scale_o_row(O, q_idx)
                self.vram_add(O, PV, dst_row_offset=q_idx * mlen)

            self.final_scale_o(q_idx, O)

        return O

    def _flash_attention_gqa_fused(self, Q, K, V, scale, hq, hkv, h_qkv):
        """GQA flash attention using the fused M_BTMM template."""
        ratio = hq // hkv
        mlen = self.mlen
        blen = self.blen
        vlen = mlen

        if ratio != blen:
            raise ValueError(
                f"GQA ratio hq/hkv={ratio} must equal blen={blen} "
                "(hardware packs heads into blen)."
            )
        if ratio * h_qkv != mlen:
            raise ValueError(
                f"GQA constraint: (hq/hkv)*h_qkv = {ratio * h_qkv} must equal mlen={mlen}."
            )

        s_q, _q_total_dim = Q.shape
        s_kv, _k_total_dim = K.shape

        if scale is None:
            scale = 1.0 / math.sqrt(h_qkv)

        self._ensure_hbm_sub_matrix_registered(K)
        self._ensure_hbm_sub_matrix_registered(V)
        alloc = self.register_allocator
        k_addr, v_addr = alloc.allocate_addr(2)
        gp_for_preload = alloc.allocate_gp(2)
        setup = preload_addr_reg_asm(
            addr_reg_to_set=[k_addr, v_addr],
            available_registers=gp_for_preload,
            addr_reg_val=[K.hbm_addr, V.hbm_addr],
        )
        alloc.free_gp(gp_for_preload)
        self.emit(setup)

        q_vram_base = self.get_vram_addr(Q.name)
        s_name = self._scoped_name("_gqa_S")
        pv_name = self._scoped_name("_gqa_PV")
        o_name = self._scoped_name("O")

        self.allocate_vram_matrix(name=s_name, rows=mlen * ratio, cols=mlen, strict=False)
        self.allocate_vram_matrix(name=pv_name, rows=mlen * ratio, cols=mlen, strict=False)
        self.allocate_vram_matrix(name=o_name, rows=s_q, cols=hq * h_qkv, strict=False)

        br = min(mlen, s_q)
        fp_allocs = self.fpram_allocator
        if "_gqa_fp_const_zero" not in fp_allocs.allocations:
            fp_allocs.allocate(name="_gqa_fp_const_zero", size=1)
            fp_allocs.allocate(name="_gqa_fp_const_scale", size=1)
            fp_allocs.allocate(name="_gqa_fp_const_neg_inf", size=1)
        fp_info = self.add_fpram_object(name="_gqa_softmax_state", size=3 * br * ratio)
        if fp_info.fpram_addr is None:
            raise RuntimeError("Failed to allocate FPRAM for GQA softmax state")

        self.emit(
            flash_attn_asm(
                mlen=mlen,
                vlen=vlen,
                blen=blen,
                batch=1,
                hq=hq,
                hkv=hkv,
                d=h_qkv,
                q_len=s_q,
                kv_len=s_kv,
                alive_registers_int=list(range(1, 16)),
                alive_registers_fp=list(range(1, 8)),
                vector_sram_base_address=q_vram_base,
                fp_sram_start_address=fp_info.fpram_addr,
                k_base_hbm_offset_reg=k_addr,
                v_base_hbm_offset_reg=v_addr,
            )
        )

        alloc.free_addr([k_addr, v_addr])
        O = VRAMMatrixVar(self, o_name, (s_q, hq * h_qkv), display_name="O")
        self._tensors[o_name] = O
        return O

    def init_online_softmax(self, q_idx: int, o_matrix: VRAMMatrixVar):
        """Initialize Online Softmax state: m=-inf, l=0, O_row=0"""
        o_info = super().get_tensor_info(o_matrix.name)
        seq_len, head_dim = o_info.shape

        super().init_online_softmax(
            q_idx=q_idx,
            o_matrix=o_matrix.name,
            seq_len=seq_len,
            head_dim=head_dim,
        )

    def online_softmax_block(self, s_block: VRAMMatrixVar, scale: float):
        """Perform Online Softmax on S block"""
        super().online_softmax_block(
            s_block_matrix=s_block.name,
            scale=scale,
        )

    def compute_pv(
        self,
        s_block: VRAMMatrixVar,
        v_input: InputVar,
        k_idx: int,
        pv_matrix: VRAMMatrixVar,
        head_dim: int,
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
        )

    def scale_o_row(self, o_matrix: VRAMMatrixVar, q_idx: int):
        """Scale current row block of O by m_res"""
        o_info = super().get_tensor_info(o_matrix.name)
        seq_len, head_dim = o_info.shape

        super().scale_o_row(
            o_matrix=o_matrix.name,
            q_idx=q_idx,
            seq_len=seq_len,
            head_dim=head_dim,
        )

    def final_scale_o(self, q_idx: int, o_matrix: VRAMMatrixVar):
        """Final scaling: O[q_idx] = O[q_idx] / l"""
        o_info = super().get_tensor_info(o_matrix.name)
        seq_len, head_dim = o_info.shape

        super().final_scale_o(
            q_idx=q_idx,
            o_matrix=o_matrix.name,
            seq_len=seq_len,
            head_dim=head_dim,
        )


__all__ = ["ProgramAttentionMixin"]
