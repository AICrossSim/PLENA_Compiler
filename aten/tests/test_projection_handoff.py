"""Static producer/consumer retention must preserve data and scalar state."""

import numpy as np
import pytest

from assembler.assembly_to_binary import AssemblyToBinary
from assembler.parser import parse_asm_file
from compiler.aten.plena.mview import MatrixViewShape
from compiler.aten.plena.projection_handoff import retain_vector_handoffs
from compiler.aten.plena.recurrent_coefficients import lower_bf16_gather


def execute(text, path):
    path.write_text(text)
    instructions = parse_asm_file(str(path))
    assembler = AssemblyToBinary("doc/operation.svh", "doc/configuration.svh")
    for instruction in instructions:
        assembler._convert_to_binary(instruction)
    gp = [0] * 16
    vector = np.zeros(64 * 2048, dtype=np.float32)
    # Different source and destination values expose stale-row reuse.
    hbm = (np.arange(65536, dtype=np.float32) % 67) - 33
    for base in (8192, 16384, 24576):
        hbm[base // 2 : base // 2 + 2048] += base // 4096
    loops, pc = [], 0
    while pc < len(instructions):
        inst = instructions[pc]
        op = inst.opcode
        if op == "S_ADDI_INT":
            gp[inst.rd] = gp[inst.rs1] + inst.imm
        elif op == "S_LUI_INT":
            gp[inst.rd] = inst.imm << 12
        elif op in ("H_PREFETCH_V", "H_STORE_V"):
            v, h = gp[inst.rd], gp[inst.rs1] // 2
            if op == "H_PREFETCH_V":
                vector[v : v + 2048] = hbm[h : h + 2048]
            else:
                hbm[h : h + 2048] = vector[v : v + 2048]
        elif op == "V_SHFT_V":
            dst, src, shift = gp[inst.rd], gp[inst.rs1], gp[inst.rs2]
            assert shift == 0
            vector[dst : dst + 2048] = vector[src : src + 2048].copy()
        elif op == "V_ADD_VV":
            dst, a, b = gp[inst.rd], gp[inst.rs1], gp[inst.rs2]
            vector[dst : dst + 2048] = vector[a : a + 2048] + vector[b : b + 2048]
        elif op == "C_LOOP_START":
            loops.append([pc, inst.imm])
            gp[inst.rd] = inst.imm
        elif op == "C_LOOP_END":
            loops[-1][1] -= 1
            gp[inst.rd] = loops[-1][1]
            if loops[-1][1]:
                pc = loops[-1][0]
            else:
                loops.pop()
        else:
            raise AssertionError(op)
        pc += 1
    return hbm, gp


def dma(row, address, store=False):
    return (
        f"S_ADDI_INT gp1, gp0, {row * 2048}\n"
        f"S_ADDI_INT gp2, gp0, {address}\n"
        f"H_{'STORE' if store else 'PREFETCH'}_V gp1, gp2, a0, 0, 2\n"
    )


def test_retention_survives_source_clobber_and_preserves_private_requests(tmp_path):
    source = (
        dma(0, 8192)
        + dma(0, 40960, True)
        + dma(0, 16384)
        + dma(0, 49152, True)
        + dma(0, 24576)
        + dma(1, 40960)
        + dma(1, 57344, True)
        + dma(2, 49152)
        + dma(2, 65536, True)
    )
    revised, report = retain_vector_handoffs(source, slots=2)
    expected, expected_gp = execute(source, tmp_path / "old.asm")
    actual, actual_gp = execute(revised, tmp_path / "new.asm")
    np.testing.assert_array_equal(actual, expected)
    assert actual_gp == expected_gp
    assert report["retains"] == 2
    assert report["bypassed_reads"] == 2
    assert report["peak_live_rows"] == 2
    assert report["hbm_read_bytes_saved"] == 8192
    assert revised.count("H_STORE_V ") == source.count("H_STORE_V ")


def test_partial_overwrite_invalidates_stale_retention(tmp_path):
    source = (
        dma(0, 8192)
        + dma(0, 40960, True)
        + dma(1, 16384)
        + dma(1, 41024, True)
        + dma(2, 40960)
        + dma(2, 57344, True)
    )
    revised, report = retain_vector_handoffs(source, slots=1)
    np.testing.assert_array_equal(
        execute(revised, tmp_path / "new.asm")[0],
        execute(source, tmp_path / "old.asm")[0],
    )
    assert report["bypassed_reads"] == 0


def test_one_slot_evicts_by_future_use_without_hidden_storage(tmp_path):
    source = (
        dma(0, 8192)
        + dma(0, 40960, True)
        + dma(1, 16384)
        + dma(1, 49152, True)
        + dma(2, 49152)
        + dma(2, 57344, True)
        + dma(3, 40960)
        + dma(3, 65536, True)
    )
    revised, report = retain_vector_handoffs(source, slots=1)
    np.testing.assert_array_equal(
        execute(revised, tmp_path / "new.asm")[0],
        execute(source, tmp_path / "old.asm")[0],
    )
    assert report["peak_live_rows"] == 1
    assert report["bypassed_reads"] == 1


def test_static_loop_is_preserved_and_address_checked(tmp_path):
    source = (
        dma(0, 8192)
        + dma(0, 40960, True)
        + "C_LOOP_START gp15, 2\nV_ADD_VV gp1, gp1, gp1, 0\nC_LOOP_END gp15\n"
        + dma(1, 40960)
        + dma(1, 57344, True)
    )
    revised, report = retain_vector_handoffs(source)
    np.testing.assert_array_equal(
        execute(revised, tmp_path / "new.asm")[0],
        execute(source, tmp_path / "old.asm")[0],
    )
    assert revised.count("C_LOOP_START") == 1
    assert report["loop_barriers"] == 2
    assert report["bypassed_reads"] == 0
    crossing = (
        "S_ADDI_INT gp1, gp0, 96256\nS_ADDI_INT gp2, gp0, 8192\n"
        "C_LOOP_START gp15, 2\nH_PREFETCH_V gp1, gp2, a0, 0, 2\n"
        "S_ADDI_INT gp1, gp1, 2048\nC_LOOP_END gp15\n"
    )
    with pytest.raises(ValueError, match="reserved"):
        retain_vector_handoffs(crossing)


def test_candidate_batched_projection_checks_every_request():
    shape = MatrixViewShape(256, 32).pack()
    source = (
        f"S_ADDI_INT gp12, gp0, {shape}\nL_TILE_CFG 0, gp12, gp0\n"
        "S_ADDI_INT gp1, gp0, 0\nS_ADDI_INT gp3, gp0, 96256\n"
        "S_ADDI_INT gp4, gp0, 65571\nM_MM.P gp1, gp2, gp3, gp4, 0\n"
    )
    with pytest.raises(ValueError, match="reserved"):
        retain_vector_handoffs(source)


@pytest.mark.parametrize(
    "source",
    [
        "S_LD_INT gp1, gp2, 0\n",
        "C_SET_ADDR_REG a0, gp1, gp2\n",
        "H_PREFETCH_V gp1, gp2, a0, 0, 0\n",
        "C_BREAK\n",
    ],
)
def test_unknown_effects_fail_closed(source):
    with pytest.raises(ValueError):
        retain_vector_handoffs(source)


@pytest.mark.parametrize("strategy", ["reference", "grouped", "pattern", "cached"])
def test_gather_respects_reserved_workspace(strategy):
    # Repeat one source value to create a reusable mask and contribution.
    source = [(8192, i % 3) for i in range(32)] * 128
    assembly = lower_bf16_gather(
        source, 32768, 0, 4096, strategy=strategy, vector_rows=7
    )
    # No legal gather operation may touch rows 8..63; the pass checks this.
    retain_vector_handoffs(assembly, first_row=8, slots=56)
    with pytest.raises(ValueError, match="workspace"):
        lower_bf16_gather(source, 32768, 0, 4096, strategy=strategy, vector_rows=6)


@pytest.mark.parametrize("prefix", ["; @operator=", "; operator="])
def test_projection_and_handoff_pool_share_storage_only_at_disjoint_phases(
    tmp_path, prefix
):
    before = prefix + "prep\n" + dma(0, 8192) + dma(0, 40960, True)
    projection = prefix + "projection\n" + dma(48, 16384) + dma(48, 40960, True)
    after = (
        prefix
        + "conv\n"
        + dma(1, 40960)
        + dma(1, 49152, True)
        + dma(2, 49152)
        + dma(2, 57344, True)
    )
    source = before + projection + after
    revised, report = retain_vector_handoffs(source, projection_stages=("projection",))
    np.testing.assert_array_equal(
        execute(revised, tmp_path / "new.asm")[0],
        execute(source, tmp_path / "old.asm")[0],
    )
    assert projection in revised  # no copies or retention inside projection
    assert before in revised  # do not retain across the projection boundary
    assert report["retains"] == report["bypassed_reads"] == 1
    assert report["projection_phase_barriers"] == 2
    assert report["simultaneous_projection_reserved_bytes"] == 0
    with pytest.raises(ValueError, match="reserved"):
        retain_vector_handoffs(source)
    with pytest.raises(ValueError, match="missing"):
        retain_vector_handoffs(source, projection_stages=("missing_projection",))


def test_projection_names_cannot_hide_a_nonprojection_overwrite():
    source = (
        "; @operator=projection\n"
        + dma(48, 8192)
        + "; @operator=conv\n"
        + dma(48, 16384)
    )
    with pytest.raises(ValueError, match="reserved"):
        retain_vector_handoffs(source, projection_stages=("projection",))
