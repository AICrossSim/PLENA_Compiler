from ._imm import load_large_int_str as _load_large_int


def preload_addr_reg_asm(addr_reg_to_set: list[int], available_registers: list[int], addr_reg_val: list[int]) -> str:
    """
    Generates assembly code for preloading address registers.
    """
    generated_code = "; Preload Addr Reg Generation \n"
    for i in range(len(addr_reg_val)):
        generated_code += _load_large_int(available_registers[i], addr_reg_val[i])
        generated_code += f"C_SET_ADDR_REG a{addr_reg_to_set[i]}, gp0, gp{available_registers[i]} \n"

    return generated_code
