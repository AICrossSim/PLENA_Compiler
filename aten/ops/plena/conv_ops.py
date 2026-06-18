"""PLENA backend for conv2d: on-chip im2col + systolic matmul.

Two im2col paths:
- Default (use_shift=False): uses only formally documented instructions
  (H_PREFETCH_V + V_MUL_VV + V_RED_SUM + S_ST_FP + S_MAP_V_FP).  Slower
  because each patch element is extracted scalar-by-scalar through FP_SRAM.
- Opt-in (use_shift=True, or CONV_USE_SHIFT=1 env var): uses the
  V_SHIFT_V opcode for vector-level patch placement.  Faster for long
  patches.  Requires the emulator to support V_SHIFT_V (opcode 0x31,
  LSB-first right-shift per main.rs fix by George Wu, commit 24eb011).

After im2col the systolic matmul uses the compiler's standard linear path.

HBM layout convention (caller must arrange data accordingly):
  input_raw shape  = (C_in * H, W_padded)   — each row is one spatial row of one channel
  weight_2d shape  = (K_col, C_out)          — standard im2col weight layout

Alignment requirement:
  H_PREFETCH_V requires the HBM element address to be a multiple of 64.
  With W_padded=64 and ow=0 (OW=1): offset = (c*H + oh+kr) * 64 → always aligned.
"""

_PREFETCH_V_AMOUNT = 4  # H_PREFETCH_V always loads this many VRAM rows


def conv2d_plena(
    prog,
    input_raw_var,
    weight_2d_var,
    C_in: int,
    H: int,
    W: int,
    K: int,
    OH: int,
    OW: int,
    M: int,
    W_padded: int | None = None,
    fp_one_reg: int = 1,
    use_shift: bool = False,
    stride: int = 1,
    return_im2col: bool = False,
):
    """
    PLENA backend: hardware im2col + systolic matmul.
    Default path avoids V_SHIFT_V; use_shift=True opts into the vector-shift path.

    Args:
        prog:           PlenaCompiler instance.
        input_raw_var:  InputVar, raw NCHW input in HBM,
                        shape = (C_in*H, W_padded).
                        Element [c, h, w] is at HBM offset (c*H+h)*W_padded+w.
        weight_2d_var:  InputVar, reshaped weight in HBM,
                        shape = (K_col, C_out)  where K_col = C_in*K*K.
        C_in:   Number of input channels.
        H:      Input spatial height.
        W:      Input spatial width (real, not padded).
        K:      Kernel size (K×K square kernel).
        OH:     Output height  = (H - K) // stride + 1.
        OW:     Output width   = (W - K) // stride + 1.
        M:      Total output positions = OH * OW  (for batch=1).
        W_padded:
            Padded HBM row width for 64-element alignment.
            Must satisfy W_padded % 64 == 0 and W_padded >= W.
            Defaults to next multiple of 64 >= W.
        stride:
            Spatial convolution stride.  Native ViT/SigLIP patch embedding
            uses stride == K; legacy conv tests use the default stride 1.
        return_im2col:
            Debug/testing hook.  If true, return the generated im2col VRAM
            matrix before the systolic projection.

    Returns:
        VRAMMatrixVar for the output, shape (M, C_out).
    """
    # Allow env-var override for ASM profiling without touching test files
    import os

    if os.environ.get("CONV_USE_SHIFT") == "1":
        use_shift = True
    elif os.environ.get("CONV_USE_SHIFT") == "0":
        use_shift = False

    # FYP: native CONV_2D engine instruction (DEFAULT-ON). When the shape matches
    # the patch-embed signature (stride == K, C_in <= 4), replace the whole im2col
    # + systolic-GEMM lowering with a CONV_2D latency stub (modeled cycles of a
    # dedicated conv MAC engine). Latency-only -- no real conv is performed (the
    # output tensor is left as-allocated; numerical correctness is out of scope).
    # Toggle off with CONV_USE_CONV2D_INSTR=0 (-> im2col / im2col-shift below).
    if os.environ.get("CONV_USE_CONV2D_INSTR", "1") != "0" and stride == K and C_in <= 4:
        from compiler.asm_templates.conv2d_asm import conv2d_instr_asm

        C_out = weight_2d_var.shape[1]
        c_out_par = int(os.environ.get("CONV2D_C_OUT_PAR", "8"))
        conv_out = prog.alloc("conv2d_out", M, C_out, strict=False)
        out_addr = prog.get_vram_addr(conv_out.name)
        asm_code, _modeled_cycles = conv2d_instr_asm(
            M=M,
            C_out=C_out,
            c_out_par=c_out_par,
            out_vram_addr=out_addr,
            out_reg=1,
        )
        prog.emit(asm_code)
        return conv_out

    # Lazy imports to avoid circular dependencies at module load time
    if use_shift:
        from compiler.asm_templates.im2col_asm import im2col_asm, PREFETCH_V_AMOUNT
    else:
        from compiler.asm_templates.im2col_asm_no_shift import (
            im2col_asm_no_shift,
            PREFETCH_V_AMOUNT,
        )

    vlen = prog.mlen
    K_col = C_in * K * K
    # Pad K_col to next multiple of vlen so column-block-major tiles don't overflow VRAM
    K_col_padded = ((K_col + vlen - 1) // vlen) * vlen

    if W_padded is None:
        W_padded = ((W + 63) // 64) * 64  # next multiple of 64

    assert W_padded % 64 == 0, f"W_padded={W_padded} must be a multiple of 64"

    # ------------------------------------------------------------------
    # Allocate VRAM regions
    # ------------------------------------------------------------------
    if use_shift:
        # Single mask vector: [1*K, 0*(VLEN-K)] — host must preload before exec
        mask_mat = prog.alloc("im2col_mask", 1, vlen, strict=False)
    else:
        # Basis vector for every intra-load position used by stride-aware
        # extraction.  Stride-1 tests only need K rows; patch embedding with
        # stride=K can need positions across the whole 64-element HBM load.
        basis_positions = {
            (ow * stride) % 64 + kc
            for ow in range(OW)
            for kc in range(K)
        }
        basis_mat = prog.alloc("im2col_basis", len(basis_positions), vlen, strict=False)
    # Scratch area for H_PREFETCH_V landing (needs PREFETCH_V_AMOUNT rows)
    scratch_mat = prog.alloc("im2col_scratch", PREFETCH_V_AMOUNT, vlen, strict=False)
    # Temp row for V_MUL_VV result (used by no_shift variant)
    temp_mat = prog.alloc("im2col_temp", 1, vlen, strict=False)
    # Output im2col matrix: M rows × K_col_padded cols (padded for tile alignment)
    output_mat = prog.alloc("im2col_out", M, K_col_padded, strict=False)
    output_physical_rows = output_mat.physical_shape[0]

    # ------------------------------------------------------------------
    # Look up VRAM base addresses from the symbol table
    # ------------------------------------------------------------------
    if use_shift:
        mask_vec_vram_addr = prog.get_vram_addr(mask_mat.name)
    else:
        basis_vram_base = prog.get_vram_addr(basis_mat.name)
    scratch_vram_addr = prog.get_vram_addr(scratch_mat.name)
    temp_vram_addr = prog.get_vram_addr(temp_mat.name)
    output_vram_base = prog.get_vram_addr(output_mat.name)

    # ------------------------------------------------------------------
    # GP register allocation
    # alive_registers: [scratch_reg, temp_reg, off_reg, out_reg, basis_reg]
    #   im2col_asm (with_shift) needs a 6th register for shift_amount
    # setup_gp: used once to load HBM base into the 'a' address register
    # ------------------------------------------------------------------
    alive_registers = [1, 2, 3, 4, 5]
    if use_shift:
        alive_registers = [1, 2, 3, 4, 5, 6]
        setup_gp = 7
    else:
        setup_gp = 6
    addr_reg_idx = 0  # use a0 for input HBM base

    # ------------------------------------------------------------------
    # Emit: set address register a0 = input_raw HBM base address
    # ------------------------------------------------------------------
    hbm_base = input_raw_var.hbm_addr
    setup_lines = []
    if hbm_base <= 262143:
        setup_lines.append(f"S_ADDI_INT gp{setup_gp}, gp0, {hbm_base}")
    else:
        setup_lines.append(f"S_LUI_INT gp{setup_gp}, {hbm_base >> 12}")
        setup_lines.append(f"S_ADDI_INT gp{setup_gp}, gp{setup_gp}, {hbm_base & 0xFFF}")
    setup_lines.append(f"C_SET_ADDR_REG a{addr_reg_idx}, gp0, gp{setup_gp}")
    prog.emit("\n".join(setup_lines) + "\n")

    # ------------------------------------------------------------------
    # Emit: im2col assembly
    # ------------------------------------------------------------------
    if use_shift:
        asm_code = im2col_asm(
            mlen=vlen,
            vlen=vlen,
            C_in=C_in,
            H=H,
            W=W,
            K=K,
            OH=OH,
            OW=OW,
            M=M,
            alive_registers=alive_registers,
            input_hbm_base_addr_reg=addr_reg_idx,
            mask_vec_vram_addr=mask_vec_vram_addr,
            scratch_vram_addr=scratch_vram_addr,
            output_vram_base=output_vram_base,
            output_physical_rows=output_physical_rows,
            W_padded=W_padded,
            fp_one_reg=fp_one_reg,
            stride=stride,
        )
    else:
        asm_code = im2col_asm_no_shift(
            mlen=vlen,
            vlen=vlen,
            C_in=C_in,
            H=H,
            W=W,
            K=K,
            OH=OH,
            OW=OW,
            M=M,
            alive_registers=alive_registers,
            input_hbm_base_addr_reg=addr_reg_idx,
            basis_vram_base=basis_vram_base,
            scratch_vram_addr=scratch_vram_addr,
            temp_vram_addr=temp_vram_addr,
            output_vram_base=output_vram_base,
            output_physical_rows=output_physical_rows,
            W_padded=W_padded,
            fp_one_reg=fp_one_reg,  # f1 = 1.0 by default (must be in fp_preload[fp_one_reg])
            fp_ex_reg=2,  # f2 = V_RED_SUM accumulator
            stride=stride,
    )
    prog.emit(asm_code)

    # ------------------------------------------------------------------
    # Systolic matmul: im2col_out @ weight_2d  -> (M, C_out)
    # ------------------------------------------------------------------
    if return_im2col:
        return output_mat
    return prog.linear(output_mat, weight_2d_var)
