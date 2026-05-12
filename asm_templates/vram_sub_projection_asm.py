"""Pure emitter for VRAM sub-projection ISA.

Shared implementation kernel used by ``IsaCompiler.vram_sub_projection_asm``
and ``IsaCompiler.vram_sub_projection_T_asm``. The caller resolves all
instance-dependent state (register allocator, tile layouts, MRAM addresses,
``unroll_loops`` default) and passes it in as plain parameters, so this
emitter can be unit-tested in isolation.
"""

from __future__ import annotations

import math

from ._imm import load_large_int as _load_large_int_list


def vram_sub_projection_asm_impl(
    mlen: int,
    blen: int,
    unroll_loops: bool,
    header_lines: list[str],
    vram_row_start_addr: int,
    mram_start_addr: int,
    result_vram_addr: int,
    full_batch: int,
    num_hidden_blocks: int,
    mat_col_stride: int,
    transposed: bool,
    gp_regs: list[int],
    caller_name: str,
) -> str:
    """
    Shared implementation kernel for vram_sub_projection_asm and
    vram_sub_projection_T_asm.

    Parameters resolved by the caller before this point:
        mlen              -- matrix tile size (rows)
        blen              -- vector tile size (batch lanes)
        unroll_loops      -- if True, bake all addresses at ASM-gen time;
                             if False, emit C_LOOP_START/END loops
        header_lines      -- comment lines already assembled by the caller
        vram_row_start_addr -- VRAM address of the first activation block
        mram_start_addr   -- MRAM address of the first weight block
        result_vram_addr  -- VRAM destination address for the (mlen, mlen) result
        full_batch        -- full batch dimension of the activation VRAM matrix
        num_hidden_blocks -- number of K-blocks to accumulate over
        mat_col_stride    -- MRAM outer-column stride (blen for M_MM, blen*mlen for M_TMM)
        transposed        -- True → emit M_TMM with (act, mat) operand order;
                             False → emit M_MM with (mat, act) operand order
        gp_regs           -- list of at least 9 GP register indices
        caller_name       -- used in error messages only

    Returns:
        ISA code string.
    """
    if len(gp_regs) < 9:
        raise ValueError(f"{caller_name} requires at least 9 gp registers, got {len(gp_regs)}")

    gp_act = gp_regs[0]
    gp_mat = gp_regs[1]
    gp_result = gp_regs[2]
    gp_loop_outer = gp_regs[3]
    gp_loop_middle = gp_regs[4]
    gp_loop_inner = gp_regs[5]
    gp_act_row_base = gp_regs[6]
    gp_mat_col_base = gp_regs[7]
    gp_result_col_base = gp_regs[8]

    tiles_per_mlen = mlen // blen
    vram_hidden_block_stride = full_batch * mlen
    mram_hidden_block_stride = mlen * mlen
    output_row_stride = blen * mlen
    # Middle loop: for sub-mlen batch only iterate the real row groups.
    row_loop_count = min(tiles_per_mlen, math.ceil(full_batch / blen))

    do_unroll = unroll_loops

    lines = list(header_lines)

    if do_unroll:
        # Fully unrolled: bake all addresses at ASM-gen time.
        # No C_LOOP_START/END or S_ADDI_INT increments.
        for oc in range(tiles_per_mlen):
            mat_col_addr = mram_start_addr + oc * mat_col_stride
            result_col_addr = result_vram_addr + oc * blen
            for or_ in range(row_loop_count):
                act_row_addr = vram_row_start_addr + or_ * output_row_stride
                result_addr = result_col_addr + or_ * output_row_stride
                for ih in range(num_hidden_blocks):
                    act_addr = act_row_addr + ih * vram_hidden_block_stride
                    mat_addr = mat_col_addr + ih * mram_hidden_block_stride
                    lines.extend(_load_large_int_list(gp_act, act_addr))
                    lines.extend(_load_large_int_list(gp_mat, mat_addr))
                    if transposed:
                        lines.append(f"M_TMM 0, gp{gp_act}, gp{gp_mat}")
                    else:
                        lines.append(f"M_MM 0, gp{gp_mat}, gp{gp_act}")
                lines.extend(_load_large_int_list(gp_result, result_addr))
                lines.append(f"M_MM_WO gp{gp_result}, gp0, 0")
    else:
        lines.extend(_load_large_int_list(gp_mat_col_base, mram_start_addr))
        lines.extend(_load_large_int_list(gp_result_col_base, result_vram_addr))
        lines.append(f"C_LOOP_START gp{gp_loop_outer}, {tiles_per_mlen}")
        lines.extend(_load_large_int_list(gp_act_row_base, vram_row_start_addr))
        lines.append(f"S_ADDI_INT gp{gp_result}, gp{gp_result_col_base}, 0")
        lines.append(f"C_LOOP_START gp{gp_loop_middle}, {row_loop_count}")
        lines.append(f"S_ADDI_INT gp{gp_act}, gp{gp_act_row_base}, 0")
        lines.append(f"S_ADDI_INT gp{gp_mat}, gp{gp_mat_col_base}, 0")
        lines.append(f"C_LOOP_START gp{gp_loop_inner}, {num_hidden_blocks}")
        if transposed:
            lines.append(f"M_TMM 0, gp{gp_act}, gp{gp_mat}")
        else:
            lines.append(f"M_MM 0, gp{gp_mat}, gp{gp_act}")
        lines.append(f"S_ADDI_INT gp{gp_act}, gp{gp_act}, {vram_hidden_block_stride}")
        lines.append(f"S_ADDI_INT gp{gp_mat}, gp{gp_mat}, {mram_hidden_block_stride}")
        lines.append(f"C_LOOP_END gp{gp_loop_inner}")
        lines.append(f"M_MM_WO gp{gp_result}, gp0, 0")
        lines.append(f"S_ADDI_INT gp{gp_act_row_base}, gp{gp_act_row_base}, {output_row_stride}")
        lines.append(f"S_ADDI_INT gp{gp_result}, gp{gp_result}, {output_row_stride}")
        lines.append(f"C_LOOP_END gp{gp_loop_middle}")
        lines.append(f"S_ADDI_INT gp{gp_mat_col_base}, gp{gp_mat_col_base}, {mat_col_stride}")
        lines.append(f"S_ADDI_INT gp{gp_result_col_base}, gp{gp_result_col_base}, {blen}")
        lines.append(f"C_LOOP_END gp{gp_loop_outer}")

    return "\n".join(lines) + "\n"
