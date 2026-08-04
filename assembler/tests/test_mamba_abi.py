import struct

import pytest

from assembler.assembly_to_binary import AssemblyToBinary
from assembler.generated_contract import (
    MAMBA_COMPLETION_SIZE,
    MAMBA_DESCRIPTOR_MAGIC,
    MAMBA_DESCRIPTOR_SIZE,
    MAMBA_SUBOPS,
)
from assembler.mamba_abi import (
    decode_mamba_instruction,
    encode_mamba_instruction,
    pack_mamba_completion,
    pack_mamba_descriptor,
    unpack_mamba_completion,
    unpack_mamba_descriptor,
)


def test_mamba_instruction_exact_encoding_and_round_trip():
    word = encode_mamba_instruction(1, 2, 3, 4, MAMBA_SUBOPS["STEP"])
    assert word == 0x0050C879
    assert decode_mamba_instruction(word) == {
        "context_gp": 1,
        "descriptor_offset_gp": 2,
        "descriptor_hbm_reg": 3,
        "queue_id": 4,
        "subop": MAMBA_SUBOPS["STEP"],
    }


def test_assembler_encodes_x_mamba_with_queue_and_subop(tmp_path):
    asm_path = tmp_path / "mamba.asm"
    output_path = tmp_path / "mamba.hex"
    asm_path.write_text("X_MAMBA gp1, gp2, a3, 4, 1\n", encoding="ascii")
    assembler = AssemblyToBinary("doc/operation.svh", "doc/configuration.svh")
    words = assembler.generate_binary(str(asm_path), str(output_path))
    assert words == [encode_mamba_instruction(1, 2, 3, 4, MAMBA_SUBOPS["STEP"])]


@pytest.mark.parametrize(
    "args,match",
    [
        ((16, 0, 0, 0, 0), "context_gp"),
        ((0, 0, 8, 0, 0), "HBM address registers"),
        ((0, 0, 0, 16, 0), "queue_id"),
        ((0, 0, 0, 0, 15), "unknown Mamba sub-operation"),
    ],
)
def test_mamba_instruction_rejects_nonportable_fields(args, match):
    with pytest.raises(ValueError, match=match):
        encode_mamba_instruction(*args)


def test_mamba_descriptor_exact_size_offsets_and_round_trip():
    fields = {
        "flags": 0x5,
        "context_id": 0x10203040,
        "sequence_length": 129,
        "batch_size": 2,
        "d_model": 2688,
        "d_inner": 4096,
        "num_heads": 64,
        "head_dim": 64,
        "state_dim": 128,
        "groups": 8,
        "chunk_size": 128,
        "conv_kernel": 4,
        "dt_bias_addr": 0x123456789ABCDEF0,
        "dependency_event": 0xFFFFFFFF,
        "completion_event": 7,
        "dt_max_f32_bits": 0x7F800000,
    }
    data = pack_mamba_descriptor(fields)
    assert len(data) == MAMBA_DESCRIPTOR_SIZE
    assert struct.unpack_from("<I", data, 0)[0] == MAMBA_DESCRIPTOR_MAGIC
    assert struct.unpack_from("<Q", data, 184)[0] == fields["dt_bias_addr"]
    decoded = unpack_mamba_descriptor(data)
    for name, value in fields.items():
        assert decoded[name] == value


def test_mamba_descriptor_rejects_bad_header_and_unknown_field():
    data = bytearray(pack_mamba_descriptor({}))
    struct.pack_into("<I", data, 0, 0)
    with pytest.raises(ValueError, match="descriptor magic"):
        unpack_mamba_descriptor(bytes(data))
    with pytest.raises(ValueError, match="unknown ABI fields"):
        pack_mamba_descriptor({"not_a_field": 1})


def test_mamba_completion_exact_size_and_round_trip():
    values = {"status": 1, "completion_event": 9, "elapsed_cycles": 0x123456789}
    data = pack_mamba_completion(values)
    assert len(data) == MAMBA_COMPLETION_SIZE
    assert unpack_mamba_completion(data) == values
