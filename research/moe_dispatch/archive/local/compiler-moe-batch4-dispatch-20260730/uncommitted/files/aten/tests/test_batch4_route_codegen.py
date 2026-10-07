"""Focused codegen checks for the four-token expert-major route path."""

import pytest

from compiler.aten.plena import PlenaCompiler
from compiler.aten.plena.program_routed_moe import _pack_rtl_topk_policy


def test_batch4_expert_major_codegen_emits_one_dynamic_expert_body():
    prog = PlenaCompiler(mlen=8, blen=4, mram_tile_capacity=4)
    constants = (
        prog.fp_var("zero_row", size=8),
        prog.fp_var("unused_pos", size=4),
        prog.fp_var("unused_neg", size=4),
        prog.fp_var("one", size=4),
        prog.fp_var("neg_one", size=4),
    )
    x_input = prog.input("X", shape=(4, 8), physical_shape=(4, 8))
    x = prog.load_batch(x_input, name="X")
    logits = prog.alloc(
        "router_logits",
        rows=16,
        cols=8,
        strict=False,
        physical_shape=(16, 8),
    )
    table_bases = (0x5000, 0x8000, 0xB000)
    weights = tuple(
        prog.input(name, shape=(8, 8), physical_shape=(8, 8), hbm_addr=base)
        for name, base in zip(("W_gate", "W_up", "W_down"), table_bases, strict=True)
    )
    stride = prog.hbm_tensor_size(8 * 8)

    output = prog.gpt_oss_dynamic_moe_batch4_expert_major_v0(
        x,
        logits,
        weights,
        weight_table_bases=table_bases,
        weight_table_strides=(stride, stride, stride),
        expert_indices_int_base=0,
        weights_fp_base=64,
        num_experts=32,
        top_k=4,
        bias_tables=None,
        rows=4,
        intermediate=8,
        constants=constants,
        activation_policy="standard_swiglu",
    )
    code = prog.get_code()
    instructions = [line for line in code.splitlines() if line and not line.startswith(";")]

    def count_opcode(opcode):
        return sum(line == opcode or line.startswith(f"{opcode} ") for line in instructions)

    assert output.shape == (4, 8)
    assert count_opcode("C_ROUTE_BEGIN") == 1
    assert count_opcode("V_TOPK") == 4
    assert count_opcode("C_ROUTE_LOOP_START") == 1
    assert count_opcode("C_ROUTE_LOOP_END") == 1
    assert count_opcode("V_ROUTE_MUL") == 4
    assert count_opcode("H_PREFETCH_M") == 3
    route_begin = code.index("C_ROUTE_BEGIN")
    first_topk = code.index("V_TOPK")
    loop_start = code.index("C_ROUTE_LOOP_START")
    assert route_begin < first_topk < loop_start

    loop_body = code[loop_start:]
    assert "S_LD_INT" not in loop_body
    assert loop_body.count("S_MUL_INT") == 3


def test_batch4_expert_major_rejects_int_sram_overflow():
    prog = PlenaCompiler(mlen=8, blen=4)
    constants = tuple(prog.fp_var(name, size=8) for name in ("zero", "p", "n", "one", "neg"))
    x = prog.alloc("X", rows=4, cols=8, strict=False, physical_shape=(4, 8))
    logits = prog.alloc("logits", rows=64, cols=8, strict=False, physical_shape=(64, 8))
    weights = tuple(
        prog.input(name, shape=(8, 8), physical_shape=(8, 8), hbm_addr=base)
        for name, base in zip(("Wg", "Wu", "Wd"), (0x1000, 0x2000, 0x3000), strict=True)
    )

    try:
        prog.gpt_oss_dynamic_moe_batch4_expert_major_v0(
            x,
            logits,
            weights,
            weight_table_bases=(0x1000, 0x2000, 0x3000),
            weight_table_strides=(0x1000, 0x1000, 0x1000),
            expert_indices_int_base=1,
            weights_fp_base=64,
            num_experts=128,
            top_k=8,
            bias_tables=None,
            rows=4,
            intermediate=8,
            constants=constants,
        )
    except ValueError as error:
        assert "32-entry INT SRAM" in str(error)
    else:
        raise AssertionError("Qwen route overflow was accepted")


def test_batch4_generic_policy_is_configured_once_before_route_collection():
    prog = PlenaCompiler(mlen=8, blen=4, mram_tile_capacity=4)
    constants = tuple(
        prog.fp_var(name, size=8) for name in ("zero", "p", "n", "one", "neg")
    )
    x = prog.alloc("X", rows=4, cols=8, strict=False, physical_shape=(4, 8))
    logits = prog.alloc("logits", rows=32, cols=8, strict=False, physical_shape=(32, 8))
    bases = (0x1000, 0x2000, 0x3000)
    weights = tuple(
        prog.input(name, shape=(8, 8), physical_shape=(8, 8), hbm_addr=base)
        for name, base in zip(("Wg", "Wu", "Wd"), bases, strict=True)
    )

    prog.gpt_oss_dynamic_moe_batch4_expert_major_v0(
        x,
        logits,
        weights,
        weight_table_bases=bases,
        weight_table_strides=(0x1000, 0x1000, 0x1000),
        expert_indices_int_base=0,
        weights_fp_base=64,
        num_experts=60,
        top_k=4,
        bias_tables=None,
        rows=4,
        intermediate=8,
        constants=constants,
        activation_policy="standard_swiglu",
    )
    instructions = [
        line
        for line in prog.get_code().splitlines()
        if line and not line.startswith(";")
    ]
    opcodes = [line.split()[0] for line in instructions]

    assert opcodes.count("C_SET_TOPK_REG") == 1
    assert opcodes.count("C_ROUTE_BEGIN") == 1
    assert opcodes.count("V_TOPK") == 4
    csr_index = opcodes.index("C_SET_TOPK_REG")
    route_index = opcodes.index("C_ROUTE_BEGIN")
    first_topk_index = opcodes.index("V_TOPK")
    assert csr_index < route_index < first_topk_index
    assert instructions[route_index].endswith(", 15")
    assert all(line.endswith(", 15") for line in instructions if line.startswith("V_TOPK "))


def test_rtl_topk_policy_limits_are_explicit():
    assert _pack_rtl_topk_policy(16, 1) == 0x1001
    assert _pack_rtl_topk_policy(60, 4) == 0x3C04
    assert _pack_rtl_topk_policy(64, 6) == 0x4006
    assert _pack_rtl_topk_policy(256, 8) == 0x10008

    for experts, top_k in ((0, 1), (4, 0), (4, 5)):
        with pytest.raises(ValueError):
            _pack_rtl_topk_policy(experts, top_k)
    for experts, top_k in ((257, 8), (256, 9)):
        with pytest.raises(NotImplementedError):
            _pack_rtl_topk_policy(experts, top_k)
