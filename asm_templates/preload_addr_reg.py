from ._imm import load_large_int_str as _load_large_int


def preload_addr_reg_asm(addr_reg_to_set: list[int], available_registers: list[int], addr_reg_val: list[int]) -> str:
    """
    Generate assembly for preloading 64-bit HBM address registers.

    GP registers are 32-bit.  Low addresses retain the historical one-scratch
    sequence; addresses above 4 GiB materialise their high and low halves in
    two distinct scratch GPs before ``C_SET_ADDR_REG`` combines them.
    """
    if len(addr_reg_to_set) != len(addr_reg_val):
        raise ValueError("address-register and address-value counts must match")
    if len(available_registers) < len(addr_reg_val):
        raise ValueError("one low-half scratch GP is required per address")
    generated_code = "; Preload Addr Reg Generation \n"
    for i, value in enumerate(addr_reg_val):
        if not 0 <= value < 1 << 64:
            raise ValueError(f"HBM address must fit u64, got {value}")
        low_reg = available_registers[i]
        low = value & 0xFFFF_FFFF
        high = value >> 32
        generated_code += _load_large_int(low_reg, low)
        if high:
            high_reg = next(
                (reg for reg in available_registers if reg != low_reg), None
            )
            if high_reg is None:
                raise ValueError(
                    f"HBM address {value} requires two distinct scratch GP registers"
                )
            generated_code += _load_large_int(high_reg, high)
            generated_code += (
                f"C_SET_ADDR_REG a{addr_reg_to_set[i]}, gp{high_reg}, gp{low_reg} \n"
            )
        else:
            generated_code += (
                f"C_SET_ADDR_REG a{addr_reg_to_set[i]}, gp0, gp{low_reg} \n"
            )

    return generated_code
