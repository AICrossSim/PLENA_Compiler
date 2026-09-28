"""Check emitted address lifetimes independently of the Rust arithmetic model."""

from dataclasses import replace

import numpy as np
import pytest

from assembler.assembly_to_binary import AssemblyToBinary
from assembler.parser import parse_asm_file
from compiler.aten.plena.isa_matrix_projection import (
    Projection,
    lower_compact_projection,
    projection_config,
)
from compiler.aten.plena.mview import MatrixViewShape, validate_matrix_view_dominance
from compiler.aten.plena.matrix_access_packets import (
    PacketGeometry,
    extract_matrix_access_packets,
    matrix_access_instruction_count,
)


def fixture(batch, k=2305, n=65):
    inputs = [request * 0x20000 for request in range(batch)]
    outputs = [0x1000000 + request * 0x20000 for request in range(batch)]
    p = Projection(inputs[0], 0x2000000, outputs[0], 0x4000000, k, n, 256)
    return p, inputs, outputs


def execute_addresses(p, inputs, outputs, assembly, path, vector_rows):
    """Integer-exact sparse dot products expose aliases, padding and wrong slices.

    This interpreter deliberately has no scheduling policy: it executes the
    emitted GP addresses/DMA order. Arithmetic uses independent NumPy dots;
    small integers are exact under every relevant BF16 tree boundary.
    """
    path.write_text(assembly)
    validate_matrix_view_dominance(assembly)
    assembler = AssemblyToBinary("doc/operation.svh", "doc/configuration.svh")
    instructions = parse_asm_file(str(path))
    for instruction in instructions:
        assembler._convert_to_binary(instruction)

    hbm = {p.zero: np.zeros(2048, dtype=np.float32)}
    x = []
    for request, address in enumerate(inputs):
        values = np.full(p.input_values, 77, dtype=np.float32)
        values[: p.k] = (np.arange(p.k) + 3 * request) % 7 - 3
        hbm[address] = values
        x.append(values[: p.k])
    for address in outputs:
        hbm[address] = np.full(p.output_values, 99, dtype=np.float32)
    weights = np.zeros((p.n, p.k), dtype=np.float32)
    for column in range(p.n):
        weights[column, (column * 37 + 3) % p.k] = 1 + column % 2
    for column, k0, rows, address in p.packets():
        block = np.zeros((rows, 32), dtype=np.float32)
        krange, nrange = min(rows, p.k - k0), min(32, p.n - column)
        block[:krange, :nrange] = weights[column : column + nrange, k0 : k0 + krange].T
        hbm[address] = block.ravel()

    def owned(address, count):
        matches = [
            (base, values)
            for base, values in hbm.items()
            if base <= address and address + count * 2 <= base + values.size * 2
        ]
        assert len(matches) == 1, (address, count, matches)
        base, values = matches[0]
        start = (address - base) // 2
        return values[start : start + count]

    gp, views, matrix = [0] * 16, {}, {}
    vector = np.full(64 * 2048, np.nan, dtype=np.float32)
    trace = {"input_reads": [], "weight_reads": [], "writes": [], "batch_sizes": []}
    for instruction in instructions:
        op = instruction.opcode
        if op == "S_ADDI_INT":
            gp[instruction.rd] = gp[instruction.rs1] + instruction.imm
        elif op == "S_LUI_INT":
            gp[instruction.rd] = instruction.imm << 12
        elif op == "L_TILE_CFG":
            views[instruction.rd] = MatrixViewShape.unpack(gp[instruction.rs1])
        elif op == "H_PREFETCH_V.MV":
            shape = views[instruction.funct2]
            matrix[gp[instruction.rd]] = (
                owned(gp[instruction.rs1], shape.rows * shape.cols)
                .reshape(shape.rows, shape.cols)
                .copy()
            )
            trace["weight_reads"].append(gp[instruction.rs1])
        elif op in ("H_PREFETCH_V", "H_STORE_V"):
            base, address = gp[instruction.rd], gp[instruction.rs1]
            assert base % 2048 == 0 and 0 <= base <= (vector_rows - 1) * 2048
            if op == "H_PREFETCH_V":
                vector[base : base + 2048] = owned(address, 2048)
                if address != p.zero:
                    trace["input_reads"].append(address)
            else:
                owned(address, 2048)[:] = vector[base : base + 2048]
                trace["writes"].append(address)
        elif op == "M_MM.P":
            config = gp[instruction.rstride]
            count, xstride, ystride = (
                (config & 3) + 1,
                ((config >> 2) & 255) * 256,
                ((config >> 10) & 255) * 32,
            )
            assert config >> 18 == 0
            trace["batch_sizes"].append(count)
            weight = matrix[gp[instruction.rs1]]
            assert weight.shape == (views[instruction.funct1].rows, 32)
            for request in range(count):
                source = gp[instruction.rs2] + request * xstride
                destination = gp[instruction.rd] + request * ystride
                assert (
                    source % 256 == 0
                    and source // 2048 == (source + weight.shape[0] - 1) // 2048
                )
                assert (
                    destination % 32 == 0
                    and destination // 2048 == (destination + 31) // 2048
                )
                assert (
                    max(source + weight.shape[0], destination + 32)
                    <= vector_rows * 2048
                )
                vector[destination : destination + 32] += (
                    vector[source : source + weight.shape[0]] @ weight
                )
        else:
            raise AssertionError(op)
    for request, address in enumerate(outputs):
        np.testing.assert_array_equal(hbm[address][: p.n], x[request] @ weights.T)
        np.testing.assert_array_equal(hbm[address][p.n :], 0)
    assert sorted(trace["weight_reads"]) == sorted(
        address for _, _, _, address in p.packets()
    )
    return trace


@pytest.mark.parametrize("batch", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("batch_tile", [1, 4])
def test_compact_projection_private_requests_tails_and_finite_cache(
    tmp_path, batch, batch_tile
):
    p, inputs, outputs = fixture(batch)
    # Force a mixture of cached and streaming chunks, including a partial
    # batch group for B1/B2. Distinct requests catch mistaken stride reuse.
    rows = batch + batch_tile + max(1, batch_tile)
    code = lower_compact_projection(
        p, inputs, outputs, batch_tile=batch_tile, vector_rows=rows
    )
    trace = execute_addresses(p, inputs, outputs, code, tmp_path / "compact.asm", rows)
    assert set(trace["batch_sizes"]) == {min(batch, batch_tile)}
    for address in trace["input_reads"]:
        matches = [
            base for base in inputs if base <= address < base + p.input_values * 2
        ]
        assert len(matches) == 1 and (address - matches[0]) % 4096 == 0


def test_compact_projection_preserves_output_rows_and_uses_no_diagnostic_stores(
    tmp_path,
):
    p, inputs, outputs = fixture(4, k=257, n=2051)
    code = lower_compact_projection(p, inputs, outputs, batch_tile=4)
    trace = execute_addresses(
        p, inputs, outputs, code, tmp_path / "output_tail.asm", 58
    )
    assert len(trace["writes"]) == 8
    assert len(trace["input_reads"]) == 4


def test_compact_projection_capacity_and_private_ownership_errors():
    p, inputs, outputs = fixture(4)
    for kwargs in ({"batch_tile": 2}, {"vector_rows": 4}, {"vector_rows": 65}):
        with pytest.raises(ValueError):
            lower_compact_projection(p, inputs, outputs, **kwargs)
    with pytest.raises(ValueError, match="overlap"):
        lower_compact_projection(p, [inputs[0]] * 4, outputs)
    with pytest.raises(ValueError, match="K256"):
        lower_compact_projection(replace(p, k_tile=128), inputs, outputs)
    with pytest.raises(ValueError, match="capacity"):
        lower_compact_projection(replace(p, k=16385), inputs, outputs)


def test_projection_config_is_bounded_and_nonaliasing():
    assert projection_config(4) == 3 | (8 << 2) | (64 << 10)
    for args in [(0,), (5,), (1, 0), (1, 257), (1, 65536), (1, 256, 0), (1, 256, 8192)]:
        with pytest.raises(ValueError):
            projection_config(*args)


def test_projection_weight_packet_accounting_includes_candidate_mode():
    p, inputs, outputs = fixture(4, k=257, n=33)
    assembly = lower_compact_projection(p, inputs, outputs, batch_tile=4)
    packets = extract_matrix_access_packets(assembly, PacketGeometry(2048, 32, 16))
    assert matrix_access_instruction_count(assembly) == 8
    reads = [packet for packet in packets if packet.opcode == "M_MM.P"]
    assert len(reads) == 4
    assert (
        sum(packet.values_per_packet * packet.repeats for packet in reads)
        == p.weight_bytes // 2
    )
