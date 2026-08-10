"""Focused code-generation checks for four-token expert-major routing."""

from pathlib import Path

import pytest

from assembler import AssemblyToBinary
from compiler.aten.plena import PlenaCompiler
from compiler.aten.plena.program_routed_moe import _route_dispatch_policy


COMPILER_ROOT = Path(__file__).resolve().parents[2]


def _constants(prog: PlenaCompiler):
    return tuple(prog.fp_var(name, size=8) for name in ("zero", "positive", "negative", "one", "neg_one"))


def _weights(prog: PlenaCompiler, bases: tuple[int, int, int]):
    return tuple(
        prog.input(name, shape=(8, 8), physical_shape=(8, 8), hbm_addr=base)
        for name, base in zip(("W_gate", "W_up", "W_down"), bases, strict=True)
    )


def test_batch4_expert_major_codegen_emits_one_dynamic_expert_body(tmp_path):
    prog = PlenaCompiler(mlen=8, blen=4, mram_tile_capacity=4)
    x_input = prog.input("X", shape=(4, 8), physical_shape=(4, 8))
    x = prog.load_batch(x_input, name="X")
    logits = prog.alloc("router_logits", rows=16, cols=8, strict=False, physical_shape=(16, 8))
    bases = (0x5000, 0x8000, 0xB000)
    weights = _weights(prog, bases)
    stride = prog.hbm_tensor_size(8 * 8)

    output = prog.moe_dynamic_batch4_expert_major_v0(
        x,
        logits,
        weights,
        weight_table_bases=bases,
        weight_table_strides=(stride, stride, stride),
        expert_indices_int_base=0,
        weights_fp_base=64,
        num_experts=32,
        top_k=4,
        bias_tables=None,
        rows=4,
        intermediate=8,
        constants=_constants(prog),
        activation_policy="standard_swiglu",
    )
    code = prog.get_code()
    instructions = [line for line in code.splitlines() if line and not line.startswith(";")]

    def count_opcode(opcode: str) -> int:
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
    assert "@stage=expert_projection" in loop_body
    assert "@stage=expert_route_weight" in loop_body
    assert "@stage=scatter_combine" in loop_body

    asm_path = tmp_path / "batch4.asm"
    mem_path = tmp_path / "batch4.mem"
    asm_path.write_text(code)
    words = AssemblyToBinary(
        COMPILER_ROOT / "doc" / "operation.svh",
        COMPILER_ROOT / "doc" / "configuration.svh",
    ).generate_binary(str(asm_path), str(mem_path))
    assert len(words) == len(instructions)
    assert all(0 <= word <= 0xFFFF_FFFF for word in words)


def test_batch4_expert_major_rejects_int_sram_overflow():
    prog = PlenaCompiler(mlen=8, blen=4)
    x = prog.alloc("X", rows=4, cols=8, strict=False, physical_shape=(4, 8))
    logits = prog.alloc("logits", rows=64, cols=8, strict=False, physical_shape=(64, 8))
    bases = (0x1000, 0x2000, 0x3000)

    with pytest.raises(ValueError, match="32-entry INT SRAM"):
        prog.moe_dynamic_batch4_expert_major_v0(
            x,
            logits,
            _weights(prog, bases),
            weight_table_bases=bases,
            weight_table_strides=(0x1000, 0x1000, 0x1000),
            expert_indices_int_base=1,
            weights_fp_base=64,
            num_experts=128,
            top_k=8,
            bias_tables=None,
            rows=4,
            intermediate=8,
            constants=_constants(prog),
        )


def test_batch4_generic_policy_is_configured_once_before_route_collection():
    prog = PlenaCompiler(mlen=8, blen=4, mram_tile_capacity=4)
    x = prog.alloc("X", rows=4, cols=8, strict=False, physical_shape=(4, 8))
    logits = prog.alloc("logits", rows=32, cols=8, strict=False, physical_shape=(32, 8))
    bases = (0x1000, 0x2000, 0x3000)

    prog.moe_dynamic_batch4_expert_major_v0(
        x,
        logits,
        _weights(prog, bases),
        weight_table_bases=bases,
        weight_table_strides=(0x1000, 0x1000, 0x1000),
        expert_indices_int_base=0,
        weights_fp_base=64,
        num_experts=60,
        top_k=4,
        bias_tables=None,
        rows=4,
        intermediate=8,
        constants=_constants(prog),
        activation_policy="standard_swiglu",
    )
    instructions = [
        line for line in prog.get_code().splitlines() if line and not line.startswith(";")
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


def test_route_dispatch_policy_limits_are_explicit():
    assert _route_dispatch_policy(32, 4) == (0, None)
    assert _route_dispatch_policy(128, 8) == (1, None)
    assert _route_dispatch_policy(60, 4) == (15, 0x3C04)

    for experts, top_k in ((0, 1), (4, 0), (4, 5)):
        with pytest.raises(ValueError):
            _route_dispatch_policy(experts, top_k)
    for experts, top_k in ((257, 8), (256, 9)):
        with pytest.raises(NotImplementedError):
            _route_dispatch_policy(experts, top_k)


def test_full_hidden_shared_expert_fails_closed_at_rtl_vram_capacity():
    capacity = 1024 * 8
    prog = PlenaCompiler(mlen=8, blen=4, vram_total_size=capacity)
    x = prog.alloc(
        "X",
        rows=4,
        cols=2048,
        strict=False,
        physical_shape=(4, 2048),
    )
    weights = (
        prog.input("W_shared_gate", shape=(2048, 5632)),
        prog.input("W_shared_up", shape=(2048, 5632)),
        prog.input("W_shared_down", shape=(5632, 2048)),
    )

    with pytest.raises(MemoryError, match="VRAM overflow"):
        prog.moe_shared_expert_v0(
            x,
            weights,
            rows=4,
            intermediate=5632,
            constants=_constants(prog),
            policy_name="qwen2_moe",
            name="full_hidden_shared",
        )
