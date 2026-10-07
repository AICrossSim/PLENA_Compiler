"""Projection loop selection with no additional Matrix datapath resources.

The resident M_MV schedule retains K256 rounding. The static-transpose M_TMV
schedule also supports K512/K1024 with explicit new reduction boundaries.
Both use real aligned input windows. Neither enables replay, M_MM.P,
segmented reductions, a new slice selector, or overlapping DMA.
Matrix views remain part of the common research platform: this is not a claim
that the program runs unchanged on the original paper's complete ISA/RTL.
"""

from dataclasses import replace

from compiler.aten.plena.isa_matrix_projection import Projection, lower_resident_projection


def lower_software_projection(
    p: Projection, inputs=None, outputs=None, *, request_tile=16, vector_rows=58,
) -> str:
    """Trade weight reloads for input residency using existing SRAM only.

    Each request group finishes all output columns before the next group.
    Within a group, weights are shared across requests and output accumulators
    retain their historical K order. Reloading weights between groups is
    explicit DMA, never discounted. Cached rows are software-owned scratchpad
    allocations, not a hardware cache. A 64-row budget is legal only when the
    caller releases the six codec rows (for example, a BF16-only phase).
    """
    inputs = (p.inputs,) if inputs is None else tuple(inputs)
    outputs = (p.outputs,) if outputs is None else tuple(outputs)
    if not 1 <= len(inputs) <= 16 or len(inputs) != len(outputs):
        raise ValueError("one private input/output per request, batch 1..16")
    if type(request_tile) is not int or request_tile not in (1, 2, 4, 8, 16):
        raise ValueError("request tile must be 1/2/4/8/16")
    if type(vector_rows) is not int or not min(request_tile, len(inputs)) + 1 <= vector_rows <= 64:
        raise ValueError("software workspace exceeds existing Vector SRAM")
    for source, destination in zip(inputs, outputs):
        replace(p, inputs=source, outputs=destination).validate()
    # Check all groups together: validating each group alone misses aliases
    # between an early output and a later request's not-yet-consumed input.
    regions = [(x, x + p.input_values * 2) for x in inputs]
    regions += [(x, x + p.output_values * 2) for x in outputs]
    for i, (start, end) in enumerate(regions):
        if any(start < other_end and other_start < end for other_start, other_end in regions[:i]):
            raise ValueError("projection request allocations overlap")
    lines = [f"; @software_projection request_tile={request_tile} vector_rows={vector_rows}"]
    for first in range(0, len(inputs), request_tile):
        xs, ys = inputs[first:first + request_tile], outputs[first:first + request_tile]
        lines.append(lower_resident_projection(
            replace(p, inputs=xs[0], outputs=ys[0]), xs, ys, vector_rows=vector_rows,
        ))
    return "\n".join(lines) + "\n"


def lower_transposed_projection(p: Projection, inputs=None, outputs=None, *, vector_rows=58):
    """Static N-by-K weight packets and existing M_TMV/M_MV_WO.

    Each output row contains contiguous K weights. A K1024 packet fits the
    existing four-by-1024 operand latch; four output rows are read serially.
    The Compiler changes layout and K grouping, not arithmetic resources.
    K1024 changes BF16 reduction order relative to the K256 reference and
    therefore needs its own numerical comparison. No replay/slice ISA.
    """
    from compiler.aten.plena.ltile_v2 import Emitter, View
    from compiler.aten.plena.mview import (
        MatrixViewAllocation, MatrixViewDescriptor, MatrixViewShape,
        MatrixViewMap, validate_disjoint_matrix_views,
    )
    if any(type(v) is not int or v < 0 for v in vars(p).values()):
        raise ValueError("integer nonnegative projection arguments required")
    if p.mlen != 2048 or p.blen != 32 or p.k < 1 or p.n < 1 or p.k_tile not in (256, 512, 1024):
        raise ValueError("transposed projection requires N32 and K256/512/1024")
    inputs = (p.inputs,) if inputs is None else tuple(inputs)
    outputs = (p.outputs,) if outputs is None else tuple(outputs)
    if not 1 <= len(inputs) <= 16 or len(inputs) != len(outputs):
        raise ValueError("one private input/output per request, batch 1..16")
    if type(vector_rows) is not int or not len(inputs) + 1 <= vector_rows <= 64:
        raise ValueError("Vector workspace exceeds existing SRAM")
    spans = [(x, p.input_values * 2) for x in inputs] + [(x, p.output_values * 2) for x in outputs]
    spans += [(p.weights, p.weight_bytes), (p.zero, p.mlen * 2)]
    for i, (base, size) in enumerate(spans):
        if type(base) is not int or base < 0 or base % 64 or base + size > 2**32 or any(
            base < b + n and b < base + size for b, n in spans[:i]
        ):
            raise ValueError("unaligned, overlapping or overflowing projection buffers")
    starts = tuple(range(0, p.k, p.k_tile))
    columns = p.mlen // p.k_tile
    if len(starts) > columns * 8:
        raise ValueError("complete K-by-N32 panel exceeds existing Matrix SRAM")
    views = [View(
        (i // columns) * 32 * p.mlen + (i % columns) * p.k_tile,
        MatrixViewDescriptor(
            MatrixViewShape(32, (min(p.k_tile, p.k - k0) + 31) // 32 * 32), MatrixViewMap(32),
        ),
    ) for i, k0 in enumerate(starts)]
    validate_disjoint_matrix_views(
        [MatrixViewAllocation(str(i), v.base, v.descriptor) for i, v in enumerate(views)],
        mlen=2048, banks=64, bank_width=32, depth_rows=256,
    )
    e = Emitter()
    e.lines.append(f"; @software_projection transposed K={p.k_tile} vector_rows={vector_rows}")

    def transfer(row, address, store=False):
        e.address(1, row * p.mlen)
        e.address(2, address)
        e.lines.append(f"H_{'STORE' if store else 'PREFETCH'}_V gp1, gp2, a0, 0, 2")

    cache = {}
    for k0 in starts:
        for request, source in enumerate(inputs):
            row = len(inputs) + 1 + len(cache)
            if row < vector_rows:
                cache[request, k0] = row
                transfer(row, source + k0 * 2)
    packets = iter(p.packets())
    for row0 in range(0, p.n, p.mlen):
        for request in range(len(inputs)):
            transfer(request + 1, p.zero)
        for col in range(row0, min(row0 + p.mlen, p.n), 32):
            for view, k0 in zip(views, starts):
                column, start, _, address = next(packets)
                assert (column, start) == (col, k0)
                e.dma(view, address)
            for request, source in enumerate(inputs):
                for view, k0 in zip(views, starts):
                    row = cache.get((request, k0), 0)
                    if row == 0:
                        transfer(0, source + k0 * 2)
                    e.view(0, view.descriptor)
                    e.address(1, view.base)
                    e.address(2, row * p.mlen)
                    e.lines.append("M_TMV 0, gp1, gp2, 0")
                e.address(1, (request + 1) * p.mlen + col - row0)
                e.lines.append("M_MV_WO gp1, 0")
        for request, destination in enumerate(outputs):
            transfer(request + 1, destination + row0 * 2, True)
    return "\n".join(e.lines) + "\n"
