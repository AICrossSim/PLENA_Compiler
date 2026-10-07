"""Compiler tests for the shared Mamba descriptor and persistent-state ABI."""

from __future__ import annotations

import pytest
import torch

from assembler.assembly_to_binary import AssemblyToBinary
from assembler.generated_contract import MAMBA_DESCRIPTOR_FLAG_BITS
from assembler.mamba_abi import decode_mamba_instruction, unpack_mamba_descriptor
from aten.models.nemotron3_mamba2.lowering import (
    MambaCommandCompiler,
    MambaCommandSpec,
    MambaSubop,
    allocate_mamba_tensor_bindings,
    materialize_mamba_io_images,
)
from aten.models.nemotron3_mamba2.memory import (
    ByteAddressArena,
    MambaPersistentStateAllocator,
)
from aten.models.nemotron3_mamba2.reference import Mamba2Config, PrecisionPolicy


def _config() -> Mamba2Config:
    return Mamba2Config(
        d_model=16,
        d_inner=16,
        num_heads=2,
        head_dim=8,
        state_dim=16,
        groups=2,
        chunk_size=8,
        conv_kernel=4,
    )


def _compiler_fixture(policy: PrecisionPolicy = PrecisionPolicy.FP32_REFERENCE):
    arena = ByteAddressArena(0x1_0000_0000, 64 * 1024 * 1024)
    config = _config()
    bindings = allocate_mamba_tensor_bindings(
        arena,
        config,
        batch_capacity=2,
        sequence_capacity=9,
        precision_policy=policy,
        prefix="layer7",
        include_in_proj_bias=True,
        include_conv_bias=True,
        include_out_proj_bias=True,
    )
    states = MambaPersistentStateAllocator(arena)
    return arena, config, bindings, states


def test_prefill_and_separate_step_programs_reuse_persistent_state():
    arena, config, bindings, states = _compiler_fixture()
    prefill = MambaCommandCompiler(arena, states).compile(
        config,
        bindings,
        MambaCommandSpec(
            subop=MambaSubop.PREFILL,
            context_id=11,
            layer_id=7,
            batch_size=2,
            sequence_length=9,
            queue_id=3,
            completion_event=40,
        ),
        state_batch_capacity=2,
    )
    step = MambaCommandCompiler(arena, states).compile(
        config,
        bindings,
        MambaCommandSpec(
            subop=MambaSubop.STEP,
            context_id=11,
            layer_id=7,
            batch_size=2,
            sequence_length=1,
            queue_id=4,
            continue_state=True,
            dependency_event=40,
            completion_event=41,
        ),
        state_batch_capacity=2,
    )

    assert prefill.state == step.state
    assert prefill.descriptor.region.address != step.descriptor.region.address
    prefill_fields = unpack_mamba_descriptor(prefill.descriptor.data)
    step_fields = unpack_mamba_descriptor(step.descriptor.data)
    assert prefill_fields["ssm_state_addr"] == step_fields["ssm_state_addr"]
    assert prefill_fields["conv_state_addr"] == step_fields["conv_state_addr"]
    assert prefill_fields["scratch_addr"] == prefill_fields["scratch_bytes"] == 0
    continue_mask = 1 << MAMBA_DESCRIPTOR_FLAG_BITS["CONTINUE_STATE"]
    assert not prefill_fields["flags"] & continue_mask
    assert step_fields["flags"] & continue_mask
    assert step_fields["dependency_event"] == 40
    assert decode_mamba_instruction(step.command_word) == {
        "context_gp": 1,
        "descriptor_offset_gp": 2,
        "descriptor_hbm_reg": 0,
        "queue_id": 4,
        "subop": 1,
    }
    assert "C_SET_ADDR_REG a0, gp3, gp4" in step.assembly
    assert "X_MAMBA gp1, gp2, a0, 4, 1" in step.assembly


def test_bf16_descriptor_uses_byte_strides_but_fp32_state():
    arena, config, bindings, states = _compiler_fixture(
        PrecisionPolicy.BF16_ACTIVATION_FP32_STATE
    )
    program = MambaCommandCompiler(arena, states).compile(
        config,
        bindings,
        MambaCommandSpec(
            subop=MambaSubop.PREFILL,
            context_id=2,
            layer_id=0,
            batch_size=1,
            sequence_length=5,
            precision_policy=PrecisionPolicy.BF16_ACTIVATION_FP32_STATE,
        ),
    )
    fields = unpack_mamba_descriptor(program.descriptor.data)
    assert fields["input_token_stride"] == config.d_model * 2
    assert fields["input_batch_stride"] == 9 * config.d_model * 2
    assert fields["state_head_stride"] == config.head_dim * config.state_dim * 4
    assert fields["state_request_stride"] == config.d_inner * config.state_dim * 4
    assert (
        fields["conv_request_stride"] == config.conv_channels * config.conv_kernel * 4
    )

    value = torch.arange(2 * config.d_model, dtype=torch.float32).reshape(
        2, 1, config.d_model
    )
    input_image, output_image = materialize_mamba_io_images(bindings, value)
    second_batch_offset = bindings.input_batch_stride
    expected_second_row = (
        value[1, 0].to(torch.bfloat16).view(torch.uint16).numpy().tobytes()
    )
    assert (
        input_image.data[
            second_batch_offset : second_batch_offset + len(expected_second_row)
        ]
        == expected_second_row
    )
    assert not any(output_image.data)


@pytest.mark.skip(reason='Retired independent X_MAMBA ABI; current primary ISA uses Matrix/L_TILE. Mamba reference, descriptor layout and static recurrent tests remain active.')
def test_complete_lowered_program_assembles_to_the_canonical_command(tmp_path):
    arena, config, bindings, states = _compiler_fixture()
    program = MambaCommandCompiler(arena, states).compile(
        config,
        bindings,
        MambaCommandSpec(
            subop=MambaSubop.PREFILL,
            context_id=5,
            layer_id=1,
            batch_size=1,
            sequence_length=3,
            queue_id=6,
        ),
    )
    asm_path = tmp_path / "program.asm"
    binary_path = tmp_path / "program.hex"
    asm_path.write_text(program.assembly, encoding="ascii")
    words = AssemblyToBinary(
        "doc/operation.svh", "doc/configuration.svh"
    ).generate_binary(str(asm_path), str(binary_path))
    assert words[-1] == program.command_word
    assert program.descriptor.region in program.memory_map
    assert program.state.ssm in program.memory_map


def test_state_keys_do_not_alias_and_cannot_be_silently_relocated():
    _arena, config, _bindings, states = _compiler_fixture()
    first = states.allocate(1, 0, config, 1)
    second = states.allocate(2, 0, config, 1)
    assert first.ssm.end_address <= first.conv.address
    assert first.conv.end_address <= second.ssm.address
    assert states.allocate(1, 0, config, 1) is first
    with pytest.raises(ValueError, match="growing it would invalidate"):
        states.allocate(1, 0, config, 2)
    incompatible = Mamba2Config(
        d_model=16,
        d_inner=16,
        num_heads=2,
        head_dim=8,
        state_dim=8,
        groups=2,
        chunk_size=8,
    )
    with pytest.raises(ValueError, match="different Mamba shape"):
        states.allocate(1, 0, incompatible, 1)


def test_lowering_rejects_invalid_step_and_undersized_tensor_capacity():
    arena, config, bindings, states = _compiler_fixture()
    compiler = MambaCommandCompiler(arena, states)
    with pytest.raises(ValueError, match="sequence_length=1"):
        compiler.compile(
            config,
            bindings,
            MambaCommandSpec(
                subop=MambaSubop.STEP,
                context_id=0,
                layer_id=0,
                batch_size=1,
                sequence_length=2,
                continue_state=True,
            ),
        )
    with pytest.raises(ValueError, match="requires continue_state=True"):
        compiler.compile(
            config,
            bindings,
            MambaCommandSpec(
                subop=MambaSubop.STEP,
                context_id=0,
                layer_id=0,
                batch_size=1,
                sequence_length=1,
            ),
        )
    with pytest.raises(ValueError, match="command sequence exceeds"):
        compiler.compile(
            config,
            bindings,
            MambaCommandSpec(
                subop=MambaSubop.PREFILL,
                context_id=0,
                layer_id=0,
                batch_size=2,
                sequence_length=10,
            ),
        )


def test_byte_arena_rejects_overlap_and_preserves_alignment():
    arena = ByteAddressArena(0x1000, 0x1000)
    first = arena.reserve("first", 0x1000, 65, kind="tensor")
    second = arena.allocate("second", 1)
    assert first.address == 0x1000
    assert second.address == 0x1080
    with pytest.raises(ValueError, match="overlaps"):
        arena.reserve("bad", 0x1040, 64)
