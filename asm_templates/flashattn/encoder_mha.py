"""Encoder-focused Flash Attention wrapper.

This module provides a strict MHA contract for encoder paths:
- Q/K/V are consumed in head-major form.
- Output is written directly in chunk-major form [num_chunks, seq, vlen].
"""

from .overall import flash_attn_asm


def flash_attn_encoder_mha_asm(
    *,
    mlen: int,
    vlen: int,
    blen: int,
    batch: int,
    hq: int,
    hkv: int,
    d: int,
    q_len: int,
    kv_len: int,
    alive_registers_int: list[int],
    alive_registers_fp: list[int],
    vector_sram_base_address: int,
    fp_sram_start_address: int,
    k_base_hbm_offset_reg: int,
    v_base_hbm_offset_reg: int,
    attn_scale_fp_address: int = 5,
    inf_fp_address: int = 0,
    causal_mask: bool = False,
    kv_valid_len: int | None = None,
) -> str:
    """Generate encoder-oriented Flash Attention ASM.

    This variant forces per-head execution and chunk-major output writeback,
    which matches encoder projection input layout and avoids repacking.
    """
    if hq != hkv:
        raise ValueError(
            f"flash_attn_encoder_mha_asm requires MHA (hq == hkv), got hq={hq}, hkv={hkv}"
        )

    return flash_attn_asm(
        mlen=mlen,
        vlen=vlen,
        blen=blen,
        batch=batch,
        hq=hq,
        hkv=hkv,
        d=d,
        q_len=q_len,
        kv_len=kv_len,
        kv_valid_len=kv_valid_len,
        alive_registers_int=alive_registers_int,
        alive_registers_fp=alive_registers_fp,
        vector_sram_base_address=vector_sram_base_address,
        fp_sram_start_address=fp_sram_start_address,
        k_base_hbm_offset_reg=k_base_hbm_offset_reg,
        v_base_hbm_offset_reg=v_base_hbm_offset_reg,
        attn_scale_fp_address=attn_scale_fp_address,
        inf_fp_address=inf_fp_address,
        causal_mask=causal_mask,
        force_per_head=True,
        output_chunk_major=True,
    )
