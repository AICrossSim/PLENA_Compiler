import math


def batched_matmul_asm(
    mlen: int,
    blen: int,
    b: int,
    m: int,
    k: int,
    n: int,
    alive_registers: list[int],
    w_base_hbm_offset_reg: int,
    w_prefetch_amount: int,
    # w_precision: int, # in bytes
    a_base_hbm_offset_reg: int,
    a_prefetch_amount: int,
    # a_precision: int, # in bytes
    # on_chip_mem_space: int,
    result_base_address: int,
) -> str:
    """
    Generates assembly code for a general batched matrix multiplication operation.
    activation(Batch, M, K) @ weight(Batched, K, N) -> result(Batch, M, N)

    Args:
        (blen (int), mlen (int)) formating the shape of the matrix.
        b, m, k, n are the dimensions of the matrix.
        alive_registers are the registers that are alive.
        w_base_hbm_offset_reg is the base hbm offset of the weight matrix.
        a_base_hbm_offset_reg is the base hbm offset of the activation matrix.
        result_base_address is the base address of the result matrix.
    Assumption:
        Assuming the two tenssors are stored in the continous memory in mx data format, with the scales stored at the end of each tensor.
    Returns:
        str: Generated assembly code for projection, including dot product and RoPE(cond)
    """
    generated_code = "; Batched Matrix Multiplication Generation \n"
    assert k % mlen == 0, "k must be divisible by mlen"
    assert m % blen == 0, "m must be divisible by blen"
    assert n % blen == 0, "n must be divisible by blen"
    print(f"b = {b}, m = {m}, k = {k}, n = {n}")
    print(f"mlen = {mlen}, blen = {blen}")

    a_actual_register = alive_registers[0]
    w_actual_register = alive_registers[1]
    result_actual_register = alive_registers[2]

    k_tiles = k // mlen            # number of k-dimension tiles
    n_tiles = n // blen             # total n-dimension tiles
    tiles_per_mlen = mlen // blen   # n-tiles that fit in one matrix SRAM bank width
    n_groups = n // mlen            # number of mlen-wide column groups
    stride_len = n // mlen          # VRAM row stride for mm_wo (N/mlen vectors per result row)

    generated_code += f"S_ADDI_INT gp{result_actual_register}, gp0, {result_base_address} \n"
    for batch in range(1, b + 1):
        batch_offset = (batch - 1) * m * n
        for i in range(n_tiles):
            # --- Weight prefetch: reload matrix SRAM when starting a new mlen-wide group ---
            if i % tiles_per_mlen == 0:
                assert w_prefetch_amount >= k, "w_prefetch_amount must be greater than or equal to k"
                # Set scale and stride for weight matrix prefetch
                # Weight layout in HBM: (B, K, N), scale = K*N per batch, stride = N
                generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, {b * k * n} \n"
                generated_code += f"C_SET_SCALE_REG gp{w_actual_register} \n"
                generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, {n} \n"
                generated_code += f"C_SET_STRIDE_REG gp{w_actual_register} \n"
                # Prefetch all k-tiles of weight into separate matrix SRAM banks
                # HBM column offset for this n-group
                n_group_col_offset = (i // tiles_per_mlen) * mlen
                generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, 0 \n"
                generated_code += f"S_ADDI_INT gp{a_actual_register}, gp0, {(batch - 1) * k * n + n_group_col_offset} \n"
                for g in range(k_tiles):
                    generated_code += (
                        f"H_PREFETCH_M gp{w_actual_register}, gp{a_actual_register}, a{w_base_hbm_offset_reg}, 1, 0 \n"
                    )
                    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp{w_actual_register}, {mlen * mlen} \n"
                    generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{a_actual_register}, {mlen * n} \n"

                # Set scale and stride for activation prefetch
                # Activation layout in HBM: (B, M, K), scale = M*K per batch, stride = K
                generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, {b * m * k} \n"
                generated_code += f"C_SET_SCALE_REG gp{w_actual_register} \n"
                generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, {k} \n"
                generated_code += f"C_SET_STRIDE_REG gp{w_actual_register} \n"

            # Column offset within the current matrix SRAM bank
            col_offset = (i % tiles_per_mlen) * blen

            for j in range(m // blen):
                # Prefetch all k-tiles of this j-tile's activation rows into VRAM.
                # For batch b (1-indexed), j-tile j, k-tile g:
                #   HBM element byte offset = (batch-1)*M*K + j*blen*K + g*mlen
                #   VRAM dest (element addr) = g*blen*mlen
                hbm_j_base = (batch - 1) * m * k + j * blen * k
                for g in range(k_tiles):
                    vram_dest = g * blen * mlen
                    hbm_g_off = hbm_j_base + g * mlen
                    generated_code += f"S_ADDI_INT gp{result_actual_register}, gp0, {vram_dest} \n"
                    generated_code += f"S_ADDI_INT gp{a_actual_register}, gp0, {hbm_g_off} \n"
                    generated_code += f"H_PREFETCH_V gp{result_actual_register}, gp{a_actual_register}, a{a_base_hbm_offset_reg}, 1, 0 \n"
                # Set weight register to column offset within the first bank
                generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, {col_offset} \n"
                # M_MM for each k-tile; a_reg explicitly set to k-tile's VRAM address
                for g in range(k_tiles):
                    a_vram = g * blen * mlen
                    generated_code += f"S_ADDI_INT gp{a_actual_register}, gp0, {a_vram} \n"
                    generated_code += f"M_MM 0, gp{w_actual_register}, gp{a_actual_register} \n"
                    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp{w_actual_register}, {mlen * mlen} \n"
                # Set result address BEFORE M_MM_WO:
                # v_addr = result_base + batch_offset + j*blen*n + (i//tiles_per_mlen)*mlen + (i%tiles_per_mlen)*blen
                result_addr = result_base_address + batch_offset + j * blen * n + (i // tiles_per_mlen) * mlen + col_offset
                generated_code += f"S_ADDI_INT gp{result_actual_register}, gp0, {result_addr} \n"
                # Set stride register for mm_wo (N/mlen vectors per result row)
                generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, {stride_len} \n"
                generated_code += f"M_MM_WO gp{result_actual_register}, gp{w_actual_register}, 0 \n"
            generated_code += f"S_ADDI_INT gp{a_actual_register}, gp0, 0 \n"
            generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, 0 \n"

    return generated_code
