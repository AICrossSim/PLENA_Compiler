from ._sigmoid_activation import emit_sigmoid_activation


def gelu_asm(
    const_one_fp_address: int,
    const_1702_fp_address: int,
    alive_registers: list[int],
    activation_base_address: int,
    scratchpad_base_address: int,
    vlen: int,
    batch_size: int,
    hidden_dim: int,
) -> str:
    """
    Generate assembly code for GELU activation using sigmoid approximation.

    Sigmoid approximation: GELU(x) = x * sigmoid(1.702 * x)

    The approximation simplifies to:
    GELU(x) ≈ x * (1 / (1 + exp(-1.702 * x)))

    Args:
        const_one_fp_address: FP SRAM address containing constant 1.0
        const_1702_fp_address: FP SRAM address containing constant 1.702
        alive_registers: List of available integer registers
        activation_base_address: VRAM base address for input activations
        scratchpad_base_address: VRAM base address for intermediate results
        vlen: Vector length (number of elements per vector)
        batch_size: Batch size dimension
        hidden_dim: Hidden dimension size

    Returns:
        Generated assembly code string
    """
    return emit_sigmoid_activation(
        banner="; GELU Activation Generation\n",
        const_one_fp_address=const_one_fp_address,
        pre_scale_fp_address=const_1702_fp_address,
        alive_registers=alive_registers,
        activation_base_address=activation_base_address,
        scratchpad_base_address=scratchpad_base_address,
        vlen=vlen,
        batch_size=batch_size,
        hidden_dim=hidden_dim,
    )
