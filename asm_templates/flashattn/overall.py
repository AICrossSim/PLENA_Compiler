from __future__ import annotations

"""Main Flash Attention assembly code generation - orchestrates all components."""

from .._imm import load_large_int_str as _load_large_int
from ..reset_reg_asm import reset_fpreg_asm, reset_reg_asm, reset_vmask_asm
from .online_softmax import online_softmax_code
from .output import computing_o_code, computing_row_wise_scaling_code
from .pv import computing_pv_code
from .qkt import qkt_multiply
from .reset import reset_fpsram_code, reset_kv_prefetch, reset_vssram_code

IMM2_BOUND = 2**18 - 1

# Constants the caller seeds below the softmax state (zero, attention scale,
# -inf, RMS eps, 1/hidden, SiLU one).
FP_SRAM_CONSTANT_SLOTS = 6

# Running max, its exponentiated residual and the running sum.
SOFTMAX_SCALARS_PER_ROW = 3


def softmax_state_slots(
    query_rows: int,
    broadcast_heads: int,
    constant_slots: int = FP_SRAM_CONSTANT_SLOTS,
) -> int:
    """Scalar FP SRAM slots the online softmax holds live across one key sweep.

    Flash attention folds every key tile into the same running state, so the
    three scalars each query row of each broadcast head carries stay live for
    the whole sweep. The count is therefore set by the rows swept at once, not
    by the cache length.
    """
    return constant_slots + SOFTMAX_SCALARS_PER_ROW * query_rows * broadcast_heads


def softmax_row_tile(
    query_rows: int,
    broadcast_heads: int,
    fp_sram_depth: int,
    constant_slots: int = FP_SRAM_CONSTANT_SLOTS,
    blen: int = 1,
) -> int:
    """Largest query-row tile whose softmax state fits the scalar FP SRAM.

    Sweeping every query row at once makes the live state grow with the row
    count and with the broadcast head count, so it is unbounded in the array
    geometry. Tiling the rows caps it: one sweep carries
    ``3 * tile * broadcast_heads`` scalars whatever the cache length or MLEN is.
    The whole broadcast group stays live within a tile because those heads share
    one K/V tile, so splitting them would re-read the cache instead.
    """
    per_row = SOFTMAX_SCALARS_PER_ROW * broadcast_heads
    budget = fp_sram_depth - constant_slots
    if budget < per_row:
        raise ValueError(
            f"scalar FP SRAM holds {fp_sram_depth} slots, of which "
            f"{constant_slots} are constants, but one query row of "
            f"{broadcast_heads} broadcast heads needs {per_row}"
        )
    # The emitter sizes the last tile with `min(row_tile, br - row_base)`, so a
    # tile that does not divide the row count is handled directly. Rounding down
    # to a divisor instead would waste scalar SRAM and, for a row count with no
    # convenient factor, collapse the tile to a single row.
    tile = min(query_rows, budget // per_row)
    # QKt issues `ceil(rows / BLEN)` query blocks, so rows past the last whole
    # block are computed and discarded. Snapping the tile to a block boundary
    # spends the leftover slots on nothing and saves those issues.
    if blen > 1 and tile > blen:
        tile -= tile % blen
    return tile


def _kv_head_reuse_body(
    *,
    mlen: int,
    vlen: int,
    blen: int,
    hkv: int,
    hq: int,
    d: int,
    ratio: int,
    broadcast_amount: int,
    stage: str,
    br: int,
    k_seq_iteration_number: int,
    q_seq_iteration_number: int,
    row_tile_starts: list[int],
    row_tile: int,
    o_row_stride: int,
    q_base_address: int,
    s_base_address: int,
    pv_base_address: int,
    o_old_base_address: int,
    q_group_stride: int | None,
    o_group_stride: int | None,
    packed_group_layout: bool,
    fp_sram_start_address: int,
    attn_scale_fp_address: int,
    inf_fp_address: int,
    causal_mask: bool,
    k_base_hbm_offset_reg: int,
    v_base_hbm_offset_reg: int,
    alive_registers_int: list[int],
    alive_registers_fp: list[int],
) -> str:
    """Sweep the key cache once, selecting each KV head out of the resident tile.

    The default schedule gives every KV head its own pass over the cache, so a
    packed KV row — one row holding every head's HLEN window — is fetched once
    per head. Here the KV-head loop sits inside the key-tile loop instead: one
    prefetch brings the packed row in, and `M_BTMM`'s head-selector field picks
    each head's window out of it, so the row crosses HBM once however many heads
    read it.

    What that costs is scalar FP SRAM. Every head's online-softmax state is now
    live across the same sweep rather than one head's at a time, so the state is
    `3 * rows * ratio * hkv` slots and the query-row tile shrinks to match; the
    caller sizes `row_tile` for that before calling.
    """
    code = ""
    ig = alive_registers_int
    fg = alive_registers_fp
    state_stride = 3 * row_tile
    heads_live = ratio * hkv

    def _group_q_base(kv_head: int) -> int:
        stride = q_group_stride if q_group_stride is not None else ratio * d
        return q_base_address + kv_head * stride

    def _group_o_base(kv_head: int) -> int:
        stride = o_group_stride if o_group_stride is not None else 0
        return o_old_base_address + kv_head * stride

    def _emit_packed_k_prefetch(k_tile_index: int) -> str:
        tile_offset = k_tile_index * mlen * mlen
        emitted = f"; Packed K prefetch for key tile {k_tile_index} \n"
        emitted += _load_large_int(ig[1], tile_offset)
        emitted += (
            f"H_PREFETCH_M gp0, gp{ig[1]}, a{k_base_hbm_offset_reg}, 0, 1 \n"
        )
        emitted += reset_reg_asm(ig[1:2])
        return emitted

    def _emit_packed_v_prefetch(k_tile_index: int) -> str:
        tile_offset = k_tile_index * mlen * mlen
        emitted = f"; Packed V prefetch for key tile {k_tile_index} \n"
        emitted += _load_large_int(ig[1], tile_offset)
        emitted += _load_large_int(ig[2], mlen * mlen)
        emitted += (
            f"H_PREFETCH_M gp{ig[2]}, gp{ig[1]}, a{v_base_hbm_offset_reg}, 0, 1 \n"
        )
        emitted += reset_reg_asm(ig[1:3])
        return emitted

    # Zero every group's accumulator once; row tiles write disjoint rows.
    for kv_head_index in range(hkv):
        if packed_group_layout:
            code += reset_vssram_code(
                reset_start_address=_group_o_base(kv_head_index),
                vect_dim=vlen,
                per_stride_dim=br,
                reset_stride=br,
                reset_amount=1,
                alive_registers_int=ig[0:3],
            )
        else:
            code += reset_vssram_code(
                reset_start_address=o_old_base_address,
                vect_dim=vlen,
                per_stride_dim=d,
                reset_stride=ratio * br,
                reset_amount=ratio,
                alive_registers_int=ig[0:3],
            )

    # Decode has one complete query tile per iteration. Prefetch K before the
    # first QK product, issue V once that product releases the DMA engine, and
    # overlap V with online softmax. The next K then overlaps row finalization.
    pipelined_decode = q_seq_iteration_number == 1
    iteration_count = len(row_tile_starts) * k_seq_iteration_number
    iteration_index = 0
    if pipelined_decode and iteration_count:
        code += _emit_packed_k_prefetch(0)

    for row_base in row_tile_starts:
        tile_rows = min(row_tile, br - row_base)
        tile_state_stride = 3 * tile_rows

        # One running-max/-sum triple per query row of every live head.
        code += reset_fpsram_code(
            reset_start_address=fp_sram_start_address,
            per_stride_dim=tile_rows,
            stride_dist=tile_state_stride,
            reset_amount=heads_live,
            reset_val_address=inf_fp_address,
            alive_registers_fp=fg[0:1],
            alive_registers_int=ig[0:4],
        )
        code += reset_fpsram_code(
            reset_start_address=fp_sram_start_address + 2 * tile_rows,
            per_stride_dim=tile_rows,
            stride_dist=tile_state_stride,
            reset_amount=heads_live,
            reset_val_address=0,
            alive_registers_fp=fg[0:1],
            alive_registers_int=ig[0:4],
            use_zero_reg=True,
        )

        for k_tile_index in range(k_seq_iteration_number):
            tile_offset = k_tile_index * mlen * mlen
            if not pipelined_decode:
                code += _emit_packed_k_prefetch(k_tile_index)
                code += _emit_packed_v_prefetch(k_tile_index)

            for kv_head_index in range(hkv):
                group_q_base = _group_q_base(kv_head_index)
                tile_o_base = (
                    _group_o_base(kv_head_index) if packed_group_layout
                    else o_old_base_address
                ) + row_base * o_row_stride
                tile_pv_base = pv_base_address + row_base * vlen
                head_state_base = (
                    fp_sram_start_address
                    + kv_head_index * ratio * tile_state_stride
                )

                for _ in range(q_seq_iteration_number):
                    code += qkt_multiply(
                        d=d,
                        mlen=mlen,
                        stage=stage,
                        alive_registers=ig[0:6],
                        q_base_address=group_q_base,
                        k_base_hbm_offset_reg=k_base_hbm_offset_reg,
                        q_head_index=0 if packed_group_layout else kv_head_index * ratio,
                        k_head_index=kv_head_index,
                        k_tile_offset=tile_offset,
                        s_base_address=s_base_address,
                        s_head_offset=0,
                        use_batched=True,
                        blen=blen,
                        prefetch_k=False,
                        q_row_offset=row_base,
                        q_rows=tile_rows,
                        k_head_selector=kv_head_index,
                    )
                    code += reset_reg_asm(ig[0:6])
                    if pipelined_decode and kv_head_index == 0:
                        code += _emit_packed_v_prefetch(k_tile_index)

                    m_state = head_state_base
                    for inner_q_head_index in range(ratio):
                        stored_m_res = m_state + tile_rows
                        code += online_softmax_code(
                            mlen=mlen,
                            stage=stage,
                            alive_registers_int=ig[0:5],
                            alive_registers_fp=fg[0:5],
                            s_address=(
                                s_base_address
                                + inner_q_head_index * mlen * mlen
                                + row_base * mlen
                            ),
                            m_start_address=m_state,
                            qk_scale_address=attn_scale_fp_address,
                            causal_mask=causal_mask,
                            rows=tile_rows,
                        )
                        code += reset_fpreg_asm(fg[0:6])
                        code += reset_reg_asm(ig[0:6])
                        code += computing_pv_code(
                            head_dim=d,
                            blen=blen,
                            mlen=mlen,
                            vlen=vlen,
                            stage=stage,
                            alive_registers=ig[0:6],
                            p_base_address=s_base_address + row_base * mlen,
                            v_base_hbm_offset_reg=v_base_hbm_offset_reg,
                            q_head_index=inner_q_head_index,
                            v_head_index=kv_head_index,
                            v_tile_offset=tile_offset,
                            output_base_address=tile_pv_base,
                            head_offset=inner_q_head_index,
                            v_msram_base=mlen * mlen,
                            rows=tile_rows,
                            prefetch_v=False,
                            v_head_selector=kv_head_index,
                        )
                        code += reset_reg_asm(ig[0:6])
                        code += reset_vmask_asm(ig[0], 1 << inner_q_head_index)
                        code += computing_o_code(
                            mlen=mlen,
                            stage=stage,
                            alive_registers_int=ig[0:4],
                            alive_registers_fp=fg[0:1],
                            m_res_base_address=stored_m_res,
                            pv_base_address=tile_pv_base,
                            o_old_base_address=tile_o_base,
                            head_dim=d,
                            q_head_num=(broadcast_amount if packed_group_layout else hq),
                            rows=tile_rows,
                        )
                        m_state += tile_state_stride

            iteration_index += 1
            if pipelined_decode and iteration_index < iteration_count:
                next_k_tile = iteration_index % k_seq_iteration_number
                code += _emit_packed_k_prefetch(next_k_tile)

        # The sweep is done, so every head's running sum is final.
        for kv_head_index in range(hkv):
            tile_o_base = (
                _group_o_base(kv_head_index) if packed_group_layout
                else o_old_base_address
            ) + row_base * o_row_stride
            head_state_base = (
                fp_sram_start_address + kv_head_index * ratio * tile_state_stride
            )
            for scale_head_index in range(ratio):
                code += reset_reg_asm(ig[0:3])
                code += reset_fpreg_asm(fg[0:1])
                code += reset_vmask_asm(ig[0], 1 << scale_head_index)
                code += computing_row_wise_scaling_code(
                    mlen=mlen,
                    stage=stage,
                    alive_registers_int=ig[0:3],
                    alive_registers_fp=fg[0:1],
                    o_old_base_address=tile_o_base,
                    l_old_base_address=(
                        head_state_base
                        + scale_head_index * tile_state_stride
                        + 2 * tile_rows
                    ),
                    o_row_stride=o_row_stride,
                    use_mask=True,
                    rows=tile_rows,
                )

    return code


def flash_attn_asm(
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
    causal_mask: bool = True,
    broadcast_amount: int | None = None,
    q_group_stride: int | None = None,
    o_group_stride: int | None = None,
    scratch_base_address: int | None = None,
    output_base_address: int | None = None,
    packed_group_layout: bool = False,
    fp_sram_depth: int | None = None,
    kv_head_reuse: bool = False,
) -> str:
    """
    Args:
    vector_sram_base_address: the base address of the vector SRAM
    fp_sram_start_address: the start address of the fp SRAM
    k_base_hbm_offset_reg: the offset register of the k base address in HBM
    v_base_hbm_offset_reg: the offset register of the v base address in HBM
    attn_scale_fp_address: FP SRAM slot holding 1/sqrt(head_dim) for QK
        scaling. Defaults to 5 to match
        ``mem_layout_lib.json::fp_sram::attn_scale``. The scheduler-provided
        slot is forwarded by the caller; previously this was hardcoded to 1
        (the eps slot) and produced catastrophically mis-scaled attention
        logits once fp_sram.bin was seeded per the JSON convention.
    inf_fp_address: FP SRAM slot holding the -inf sentinel used to seed the
        running-max state at the start of each kv tile. Defaults to 0 to match
        ``mem_layout_lib.json::fp_sram::infinity``. Previously hardcoded to 2
        (the hid_reciprocal slot ~= 0.016), which made the running-max start
        at a positive value and corrupted the entire flash softmax accumulation.
    Description:
        This part of asm takes the multi-loops, looping over kv head, then two loops for the flash atten, with small loops over q head per kv head within the inner loop.
    """
    # Iteration Settings
    q_seq_iteration_number = (q_len + mlen - 1) // mlen
    k_seq_iteration_number = (kv_len + mlen - 1) // mlen
    q_index_2_kv_index_ratio = hq // hkv
    broadcast_amount = blen if broadcast_amount is None else broadcast_amount

    stage = "decode" if q_len == 1 else "prefill"
    br = min(mlen, q_len)
    bc = min(mlen, kv_len)

    # Batched path (M_BTMM) when the Q heads for one KV group occupy hardware
    # lanes inside one MLEN row. Legacy callers used BLEN as the broadcast
    # amount; packed ATen callers pass the simulator BROADCAST_AMOUNT/HW HLEN
    # explicitly and may have ratio < broadcast_amount.
    if packed_group_layout:
        use_batched = q_index_2_kv_index_ratio <= broadcast_amount
    else:
        use_batched = q_index_2_kv_index_ratio == blen

    # Memory Layout:
    # -- FP SRAM --
    # ``inf_fp_address`` (default 0) - infinity sentinel for running-max init.
    # ``attn_scale_fp_address`` (default 5) - 1/sqrt(head_dim) (QK scale).
    # Both slot indices are forwarded by the caller from
    # ``scheduler["memory_layout"]["fp_sram"]`` so mem_layout_lib.json is the
    # single source of truth for FPRAM constant placement; this template no
    # longer hardcodes addresses 1 and 2.
    # ``fp_sram_start_address`` onwards holds, for each q head in the kv-group:
    # - m old (br)
    # - m res (br)
    # - l old (br)

    print("=" * 5, "VSRAM Memory Layout", "=" * 5)
    # -- Vector SRAM --
    # Q  (q_len, hq, d) - Q is stored with shape [seq_len, num_q_heads, head_dim]
    q_base_address = vector_sram_base_address
    print(f"Q Base Address: {q_base_address}")
    # tmp S (MLEN, MLEN, s_tile_count) and also tmp P.
    # Batched path: M_BMM_WO writes blen tiles; allocate blen tiles even though
    # only ratio are consumed by softmax/PV (harmless dead writes).
    # Per-head path: only 1 S tile needed at a time (reused per head).
    s_tile_count = broadcast_amount if use_batched else 1
    # Q size = seq_len * num_q_heads * head_dim for legacy row-packed layout.
    # Packed group layout stores each KV group as a separate MLEN-wide column
    # block, so the caller passes an explicit scratch base.
    s_base_address = (
        scratch_base_address
        if scratch_base_address is not None
        else q_base_address + q_len * hq * d
    )
    print(f"S Base Address: {s_base_address}")
    # PV (q_index_2_kv_index_ratio, mlen, mlen)
    pv_base_address = s_base_address + mlen * mlen * s_tile_count
    print(f"PV Base Address: {pv_base_address}")
    # O_Old (q_len, HEAD_DIM * Hq * batch)
    o_old_base_address = (
        output_base_address
        if output_base_address is not None
        else pv_base_address + mlen * mlen * q_index_2_kv_index_ratio
    )
    print(f"O_Old Base Address: {o_old_base_address}")

    generated_code = "; Flash Attention Generation \n"
    generated_code += reset_kv_prefetch(
        hkv=hkv,
        d=d,
        mlen=mlen,
        kv_len=kv_len,
        batch=batch,
        alive_registers_int=alive_registers_int[0:1],
    )

    # Decode software pipelining. Matrix ops stall behind any in-flight matrix
    # load (single-slot load engine), so a K/V prefetch only overlaps with
    # vector/scalar work, and the engine serializes back-to-back prefetches at
    # issue. Decode (q_seq == 1) therefore splits the two loads around their
    # hiding windows: K for the NEXT (kv head, kv tile) iteration is issued at
    # the tail of the current one (its DMA hides behind the softmax-state
    # resets), and V for the CURRENT iteration is issued right after QKT (its
    # DMA hides behind the online softmax). K occupies MSRAM tile 0, V tile 1,
    # so a resident tile is never clobbered before its last matrix use.
    def _emit_k_prefetch(head_index: int, tile_index: int) -> str:
        code = f"; Pipelined K prefetch for KV head {head_index} tile {tile_index} \n"
        code += _load_large_int(
            alive_registers_int[1],
            tile_index * mlen * mlen
            + (0 if packed_group_layout else head_index * d),
        )
        code += f"H_PREFETCH_M gp0, gp{alive_registers_int[1]}, a{k_base_hbm_offset_reg}, 0, 1 \n"
        code += reset_reg_asm(alive_registers_int[1:2])
        return code

    def _emit_v_prefetch(head_index: int, tile_index: int) -> str:
        code = f"; Pipelined V prefetch for KV head {head_index} tile {tile_index} \n"
        code += _load_large_int(
            alive_registers_int[1],
            tile_index * mlen * mlen
            + (0 if packed_group_layout else head_index * d),
        )
        code += _load_large_int(alive_registers_int[2], mlen * mlen)
        code += f"H_PREFETCH_M gp{alive_registers_int[2]}, gp{alive_registers_int[1]}, a{v_base_hbm_offset_reg}, 0, 1 \n"
        code += reset_reg_asm(alive_registers_int[1:3])
        return code

    # The pipelined schedule is valid whenever one (kv head, kv tile) iteration
    # contains the entire q loop (q_seq == 1). Long-q prefill (q_seq > 1) keeps
    # the original at-use prefetch placement.
    pipelined_decode = q_seq_iteration_number == 1

    # Query-row tiling bounds the softmax state. The emitter walks
    # (kv head, row tile, key tile); when every query row is intentionally bound
    # to one shared cache, as in the synthetic decoder testbench, that re-reads
    # its K/V range per row tile. A serving batch instead binds rows to disjoint
    # per-sequence caches, so this loop shape is not a general cache-reread term.
    # Reusing one resident KV tile across the heads keeps every head's softmax
    # state live at once, so the tile shrinks by the KV head count.
    live_head_lanes = q_index_2_kv_index_ratio * (hkv if kv_head_reuse else 1)
    row_tile = (
        br
        if fp_sram_depth is None
        else softmax_row_tile(br, live_head_lanes, fp_sram_depth, blen=blen)
    )
    row_tile_starts = list(range(0, br, row_tile))
    o_row_stride = mlen if packed_group_layout else hq * d

    if kv_head_reuse:
        return generated_code + _kv_head_reuse_body(
            mlen=mlen,
            vlen=vlen,
            blen=blen,
            hkv=hkv,
            hq=hq,
            d=d,
            ratio=q_index_2_kv_index_ratio,
            broadcast_amount=broadcast_amount,
            stage=stage,
            br=br,
            k_seq_iteration_number=k_seq_iteration_number,
            q_seq_iteration_number=q_seq_iteration_number,
            row_tile_starts=row_tile_starts,
            row_tile=row_tile,
            o_row_stride=o_row_stride,
            q_base_address=q_base_address,
            s_base_address=s_base_address,
            pv_base_address=pv_base_address,
            o_old_base_address=o_old_base_address,
            q_group_stride=q_group_stride,
            o_group_stride=o_group_stride,
            packed_group_layout=packed_group_layout,
            fp_sram_start_address=fp_sram_start_address,
            attn_scale_fp_address=attn_scale_fp_address,
            inf_fp_address=inf_fp_address,
            causal_mask=causal_mask,
            k_base_hbm_offset_reg=k_base_hbm_offset_reg,
            v_base_hbm_offset_reg=v_base_hbm_offset_reg,
            alive_registers_int=alive_registers_int,
            alive_registers_fp=alive_registers_fp,
        )

    kv_iters = [
        (h, t)
        for h in range(hkv)
        for _row in row_tile_starts
        for t in range(k_seq_iteration_number)
    ]
    kv_iter_idx = 0
    if pipelined_decode:
        generated_code += _emit_k_prefetch(*kv_iters[0])

    # loop over kv heads
    for kv_head_index in range(hkv):
        group_q_base_address = q_base_address + kv_head_index * (
            q_group_stride if q_group_stride is not None else q_index_2_kv_index_ratio * d
        )
        group_o_base_address = o_old_base_address + kv_head_index * (
            o_group_stride if o_group_stride is not None else 0
        )

        # Reset O_old with zeros. Packed group layout stores one KV group's
        # active Q heads in HLEN-sized lanes of an MLEN-wide row, so zero
        # the valid rows once instead of using the legacy head-stride reset.
        # Row tiles write disjoint rows, so the accumulator is zeroed once for
        # the whole KV head rather than per tile.
        if packed_group_layout:
            generated_code += reset_vssram_code(
                reset_start_address=group_o_base_address,
                vect_dim=vlen,
                per_stride_dim=br,
                reset_stride=br,
                reset_amount=1,
                alive_registers_int=alive_registers_int[0:3],
            )
        else:
            generated_code += reset_vssram_code(
                reset_start_address=o_old_base_address,
                vect_dim=vlen,
                per_stride_dim=d,
                reset_stride=q_index_2_kv_index_ratio * br,
                reset_amount=q_index_2_kv_index_ratio,
                alive_registers_int=alive_registers_int[0:3],
            )

        o_base = group_o_base_address if packed_group_layout else o_old_base_address

        # A row tile carries its own running state through the whole key sweep,
        # so the state for the tile's rows is initialised here and consumed by
        # the row-wise scaling at the end of the tile.
        for row_base in row_tile_starts:
            tile_rows = min(row_tile, br - row_base)
            tile_o_base = o_base + row_base * o_row_stride
            tile_pv_base = pv_base_address + row_base * vlen
            state_stride = 3 * tile_rows

            # Reset m old for every q_index_2_kv_index_ratio q heads with -inf.
            # ``inf_fp_address`` is forwarded by the caller from
            # ``scheduler["memory_layout"]["fp_sram"]["infinity"]`` (default 0
            # per mem_layout_lib.json).
            generated_code += reset_fpsram_code(
                reset_start_address=fp_sram_start_address,
                per_stride_dim=tile_rows,
                stride_dist=state_stride,
                reset_amount=q_index_2_kv_index_ratio,
                reset_val_address=inf_fp_address,
                alive_registers_fp=alive_registers_fp[0:1],
                alive_registers_int=alive_registers_int[0:4],
            )

            # Reset l_old to zero (use hardware f0, not FP SRAM slot 0 which is -inf)
            generated_code += reset_fpsram_code(
                reset_start_address=fp_sram_start_address + 2 * tile_rows,
                per_stride_dim=tile_rows,
                stride_dist=state_stride,
                reset_amount=q_index_2_kv_index_ratio,
                reset_val_address=0,
                alive_registers_fp=alive_registers_fp[0:1],
                alive_registers_int=alive_registers_int[0:4],
                use_zero_reg=True,
            )

            for k_tile_index in range(k_seq_iteration_number):
                print(f" Computing {q_index_2_kv_index_ratio} Q heads for KV head {kv_head_index} in GQA mode")

                # Per-head cursor over the softmax state; the state itself
                # persists across key tiles, so only the walk is restarted here.
                m_fp_sram_start_address = fp_sram_start_address

                # # loop over per q_index_2_kv_index_ratio q heads (q_len // MLEN), compute q_index_2_kv_index_ratio heads in parallel.
                for _ in range(q_seq_iteration_number):
                    stored_m_fp_res_address = m_fp_sram_start_address + tile_rows

                    if use_batched:
                        # --- Batched path: M_BTMM computes all ratio heads at once ---
                        # Q layout: (batch, s_q, num_q_heads, h_qkv)
                        generated_code += qkt_multiply(
                            d=d,
                            mlen=mlen,
                            stage=stage,
                            alive_registers=alive_registers_int[0:6],
                            q_base_address=group_q_base_address,
                            k_base_hbm_offset_reg=k_base_hbm_offset_reg,
                            q_head_index=0 if packed_group_layout else kv_head_index * q_index_2_kv_index_ratio,
                            k_head_index=kv_head_index,
                            k_tile_offset=k_tile_index * mlen * mlen,
                            s_base_address=s_base_address,
                            s_head_offset=0,
                            use_batched=True,
                            blen=blen,
                            # Pipelined decode: K was prefetched at the previous
                            # iteration's tail (or the pre-loop prologue).
                            prefetch_k=not pipelined_decode,
                            # Each row tile sweeps the whole cache on its own, so
                            # it scores only its own query rows.
                            q_row_offset=row_base,
                            q_rows=tile_rows,
                            k_head_selector=(
                                kv_head_index if packed_group_layout else None
                            ),
                        )
                        generated_code += reset_reg_asm(alive_registers_int[0:6])
                        # V for this iteration: issued now so its DMA hides behind
                        # the online softmax; PV is its first consumer.
                        if pipelined_decode:
                            generated_code += _emit_v_prefetch(kv_head_index, k_tile_index)

                    for inner_q_head_index in range(q_index_2_kv_index_ratio):
                        if not use_batched:
                            # --- Per-head path: M_TMM computes one head's QKT ---
                            # K lives in MSRAM tile 0 and V in its own tile (see
                            # the pv call below), so K/V are prefetched only on the
                            # first Q head of the KV group and stay resident for
                            # the remaining heads — no per-head HBM re-reads.
                            abs_q_head = kv_head_index * q_index_2_kv_index_ratio + inner_q_head_index
                            generated_code += qkt_multiply(
                                d=d,
                                mlen=mlen,
                                stage=stage,
                                alive_registers=alive_registers_int[0:9],
                                q_base_address=group_q_base_address,
                                k_base_hbm_offset_reg=k_base_hbm_offset_reg,
                                q_head_index=inner_q_head_index if packed_group_layout else abs_q_head,
                                k_head_index=kv_head_index,
                                k_tile_offset=k_tile_index * mlen * mlen,
                                s_base_address=s_base_address,
                                s_head_offset=0,  # single S tile, always at offset 0
                                use_batched=False,
                                blen=blen,
                                # K shared across the group's Q heads: fetch on the
                                # first head only; pipelined decode fetched it at
                                # the previous iteration's tail already.
                                prefetch_k=(not pipelined_decode) and inner_q_head_index == 0,
                            )
                            generated_code += reset_reg_asm(alive_registers_int[0:9])
                            # V for this iteration: issued after the first head's
                            # QKT so its DMA hides behind the online softmax.
                            if pipelined_decode and inner_q_head_index == 0:
                                generated_code += _emit_v_prefetch(kv_head_index, k_tile_index)

                        # Per Q head level online softmax.  ``attn_scale_fp_address``
                        # is forwarded by the caller from
                        # ``scheduler["memory_layout"]["fp_sram"]["attn_scale"]``
                        # so this template no longer hardcodes the QK-scale slot.
                        #
                        # For batched path M_BMM_WO writes each broadcast result
                        # as a full physical MLEN x MLEN tile, even when the
                        # logical br/bc for a short sequence is smaller.
                        # For per-head path: S tile is always at s_base (offset 0).
                        # S rows are MLEN apart, so the tile's rows start
                        # ``row_base`` rows into the head's tile.
                        s_softmax_addr = (
                            s_base_address
                            + (inner_q_head_index * mlen * mlen if use_batched else 0)
                            + row_base * mlen
                        )
                        generated_code += online_softmax_code(
                            mlen=mlen,
                            stage=stage,
                            alive_registers_int=alive_registers_int[0:5],
                            alive_registers_fp=alive_registers_fp[0:5],
                            s_address=s_softmax_addr,
                            m_start_address=m_fp_sram_start_address,
                            qk_scale_address=attn_scale_fp_address,
                            causal_mask=causal_mask,
                            rows=tile_rows,
                        )
                        # P is stored in s_base_address (per-head) or
                        # s_base_address + inner_q_head_index * mlen * mlen (batched)
                        m_fp_sram_start_address += state_stride
                        generated_code += reset_fpreg_asm(alive_registers_fp[0:6])
                        generated_code += reset_reg_asm(alive_registers_int[0:6])

                        # Compute PV = P @ V and write directly to packed output
                        # Output layout: each row is VLEN elements with heads packed
                        # [head0: d][head1: d][...][headN: d]
                        generated_code += computing_pv_code(
                            head_dim=d,
                            blen=blen,
                            mlen=mlen,
                            vlen=vlen,  # VLEN = total width of all heads = num_q_heads * head_dim
                            stage=stage,
                            alive_registers=alive_registers_int[0:6],
                            p_base_address=s_base_address + row_base * mlen,
                            v_base_hbm_offset_reg=v_base_hbm_offset_reg,
                            # The batched path leaves one S tile per broadcast
                            # head, so PV selects its head's tile. The per-head
                            # path recomputes a single S tile for each head, so
                            # its scores are always at the base; indexing by
                            # head there would read past the S allocation.
                            q_head_index=inner_q_head_index if use_batched else 0,
                            v_head_index=kv_head_index,
                            v_tile_offset=k_tile_index * mlen * mlen,
                            output_base_address=tile_pv_base,
                            head_offset=inner_q_head_index,  # This head's position within the row
                            # V gets its own MSRAM tile (after K's tile 0) and is
                            # fetched once per KV group; later heads reuse it.
                            # Pipelined decode fetched it at the previous
                            # iteration's tail already.
                            v_msram_base=mlen * mlen,
                            rows=tile_rows,
                            prefetch_v=(not pipelined_decode) and inner_q_head_index == 0,
                            v_head_selector=(
                                kv_head_index if packed_group_layout else None
                            ),
                        )

                        generated_code += reset_reg_asm(alive_registers_int[0:6])
                        generated_code += reset_vmask_asm(alive_registers_int[0], 1 << inner_q_head_index)
                        # Use VLEN-aligned address - V_MASK selects the correct head slot
                        generated_code += computing_o_code(
                            mlen=mlen,
                            stage=stage,
                            alive_registers_int=alive_registers_int[0:4],
                            alive_registers_fp=alive_registers_fp[0:1],
                            m_res_base_address=stored_m_fp_res_address,
                            pv_base_address=tile_pv_base,
                            o_old_base_address=tile_o_base,
                            head_dim=d,
                            q_head_num=(broadcast_amount if packed_group_layout else hq),
                            rows=tile_rows,
                        )
                        stored_m_fp_res_address += state_stride

                # Pipelined decode: issue the NEXT iteration's K prefetch here. The
                # last matrix op of this iteration has retired, so overwriting
                # MSRAM tile 0 is safe, and the DMA overlaps the vector-only
                # softmax-state resets that separate it from its first matrix use.
                if pipelined_decode:
                    kv_iter_idx += 1
                    if kv_iter_idx < len(kv_iters):
                        generated_code += _emit_k_prefetch(*kv_iters[kv_iter_idx])

            # Every key tile has been folded into this row tile, so its running
            # sum is final: scale each head's rows by 1/l once.
            # With packed output format, each row has all heads: [h0:d][h1:d][h2:d][h3:d]
            # We use V_MASK to select only this head's elements when scaling
            for scale_head_index in range(q_index_2_kv_index_ratio):
                # Reset registers and set V_MASK for this head
                generated_code += reset_reg_asm(alive_registers_int[0:3])
                generated_code += reset_fpreg_asm(alive_registers_fp[0:1])
                generated_code += reset_vmask_asm(alive_registers_int[0], 1 << scale_head_index)

                # l_old for this head sits after its running max and residual.
                l_old_base_address = (
                    fp_sram_start_address + scale_head_index * state_stride + 2 * tile_rows
                )

                # Output is at the row tile's base with packed format
                # V_MASK selects the correct head's elements within each row
                generated_code += computing_row_wise_scaling_code(
                    mlen=mlen,
                    stage=stage,
                    alive_registers_int=alive_registers_int[0:3],
                    alive_registers_fp=alive_registers_fp[0:1],
                    o_old_base_address=tile_o_base,
                    l_old_base_address=l_old_base_address,
                    o_row_stride=o_row_stride,
                    use_mask=True,
                    rows=tile_rows,
                )

    return generated_code
