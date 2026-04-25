from ._imm import load_large_int as _load_large_int_list


def rope_asm(
    alive_registers: list,
    x_base_address: int,
    x_rot_base_address: int,
    cos_base_address: int,
    sin_base_address: int,
    scratchpad_base_address: int,
    vlen: int,
    seq_len: int,
    head_dim: int,
) -> str:
    """
    Generate assembly for Rotary Position Embedding (RoPE) applied in-place:
        x = x * cos + rotate_half(x) * sin

    Memory layout (same for x, x_rot, cos, sin):
        VRAM stores (head_dim // vlen, seq_len, vlen) — same interleaved layout
        as load_batch output.
        Chunk j, position i is at: base + j * seq_len * vlen + i * vlen

    Args:
        alive_registers: 5 GP registers [x_addr, xrot_addr, cos_addr, sin_addr, scratch_addr]
        x_base_address:        VRAM base of x (shape: seq_len × head_dim)
        x_rot_base_address:    VRAM base of rotate_half(x), preloaded from HBM
        cos_base_address:      VRAM base of cos values (seq_len × head_dim)
        sin_base_address:      VRAM base of sin values (seq_len × head_dim)
        scratchpad_base_address: 1-row VRAM scratch
        vlen:     vector length (MLEN)
        seq_len:  number of sequence positions
        head_dim: head dimension (must be multiple of vlen)
    """
    assert head_dim % vlen == 0, f"head_dim ({head_dim}) must be divisible by vlen ({vlen})"

    x_addr = alive_registers[0]
    xrot_addr = alive_registers[1]
    cos_addr = alive_registers[2]
    sin_addr = alive_registers[3]
    scratch_addr = alive_registers[4]

    num_chunks = head_dim // vlen

    lines = ["; RoPE: x = x * cos + rotate_half(x) * sin  (in-place)"]
    lines.extend(_load_large_int_list(scratch_addr, scratchpad_base_address))

    for j in range(num_chunks):
        chunk_base = j * seq_len * vlen
        for i in range(seq_len):
            addr = chunk_base + i * vlen
            lines.extend(_load_large_int_list(x_addr, x_base_address + addr))
            lines.extend(_load_large_int_list(xrot_addr, x_rot_base_address + addr))
            lines.extend(_load_large_int_list(cos_addr, cos_base_address + addr))
            lines.extend(_load_large_int_list(sin_addr, sin_base_address + addr))
            lines.append(f"V_MUL_VV gp{scratch_addr}, gp{xrot_addr}, gp{sin_addr}, 0 ")
            lines.append(f"V_MUL_VV gp{x_addr}, gp{x_addr}, gp{cos_addr}, 0 ")
            lines.append(f"V_ADD_VV gp{x_addr}, gp{x_addr}, gp{scratch_addr}, 0 ")

    return "\n".join(lines) + "\n"
