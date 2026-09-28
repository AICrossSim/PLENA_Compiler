"""Rectangular-view Matrix projections with explicit HBM lifetime bounds.

Legacy lowering uses M_MV / M_MV_WO. The separately selected compact lowering
uses the bounded M_MM.P extension (an explicit M_MM function code). Weights are static
BF16 output-block-major / K-block-major packets. Quantized weight conversion
is outside this ABI and must not be claimed as NVFP4 execution.
"""

from dataclasses import dataclass
from compiler.aten.plena.ltile_v2 import Emitter, View
from compiler.aten.plena.mview import (
    MatrixViewDescriptor,
    MatrixViewShape,
    MatrixViewMap,
)


@dataclass(frozen=True)
class Projection:
    inputs: int
    weights: int
    outputs: int
    zero: int
    k: int
    n: int
    k_tile: int = 128
    mlen: int = 2048
    blen: int = 32

    @property
    def output_values(self):
        return (self.n + self.mlen - 1) // self.mlen * self.mlen

    @property
    def input_values(self):
        # Every K tile uses an ordinary full-row DMA. The final read must stay
        # within an owned padded allocation even though Matrix uses fewer lanes.
        return (self.k - 1) // self.k_tile * self.k_tile + self.mlen

    def packets(self):
        offset = 0
        for column in range(0, self.n, self.blen):
            for k0 in range(0, self.k, self.k_tile):
                rows = (min(self.k_tile, self.k - k0) + 31) // 32 * 32
                yield column, k0, rows, self.weights + offset
                offset += rows * self.blen * 2

    @property
    def weight_bytes(self):
        return sum(rows * self.blen * 2 for _, _, rows, _ in self.packets())

    def validate(self):
        for value in vars(self).values():
            if type(value) is not int or value < 0:
                raise ValueError("integer nonnegative projection arguments required")
        if not (self.k > 0 and self.n > 0 and self.mlen == 2048 and self.blen == 32):
            raise ValueError(
                "B1 projection ABI requires VLEN=2048, BLEN=32 and positive K/N"
            )
        if not 32 <= self.k_tile <= 256 or self.k_tile % 32:
            raise ValueError("K tile must be 32..256 in complete Matrix bank words")
        regions = [
            (self.inputs, self.input_values * 2),
            (self.weights, self.weight_bytes),
            (self.outputs, self.output_values * 2),
            (self.zero, self.mlen * 2),
        ]
        for i, (start, size) in enumerate(regions):
            if start % 64 or start + size > 2**32:
                raise ValueError("unaligned or overflowing HBM allocation")
            if any(start < b + n and b < start + size for b, n in regions[:i]):
                raise ValueError("projection allocations overlap")


def lower_b1_projection(p: Projection) -> str:
    """One live Matrix packet, Vector rows 0 (input) and 1 (output).

    The caller owns input_values, output_values, weight_bytes and a zero row.
    M_MV retains its BLEN partial sums through K blocks. M_MV_WO commits each
    block; the full output row is stored only after all its blocks are valid.
    This deliberately serial schedule makes no buffering/overlap claims.
    """
    p.validate()
    e = Emitter()
    e.lines.append("; @stage=matrix_projection_bf16_rectangular")

    def transfer(row, hbm, store=False):
        e.address(1, row * p.mlen)
        e.address(2, hbm)
        e.lines.append(f"H_{'STORE' if store else 'PREFETCH'}_V gp1, gp2, a0, 0, 2")

    packets = iter(p.packets())
    for row0 in range(0, p.n, p.mlen):
        transfer(1, p.zero)
        for col in range(row0, min(row0 + p.mlen, p.n), p.blen):
            for k0 in range(0, p.k, p.k_tile):
                packet_col, packet_k, rows, address = next(packets)
                assert (packet_col, packet_k) == (col, k0)
                view = View(
                    0,
                    MatrixViewDescriptor(
                        MatrixViewShape(rows, p.blen), MatrixViewMap(rows)
                    ),
                )
                e.dma(view, address)
                transfer(0, p.inputs + k0 * 2)
                e.view(0, view.descriptor)
                e.address(1, 0)
                e.address(2, 0)
                e.lines.append("M_MV 0, gp1, gp2, 0")
            e.address(1, p.mlen + col - row0)
            e.lines.append("M_MV_WO gp1, 0")
        transfer(1, p.outputs + row0 * 2, store=True)
    return "\n".join(e.lines) + "\n"


def lower_batch_projection(p: Projection, inputs, outputs):
    """Reuse each complete K-by-32 weight panel across private requests.

    K packets occupy disjoint columns of existing Matrix SRAM. Partial sums
    for only one request live in Matrix accumulators; request outputs spill
    through the existing Vector row. This is deliberately serial and needs no
    additional accumulator, SRAM capacity, ports, or simultaneous requests.
    """
    from dataclasses import replace
    from compiler.aten.plena.mview import (
        MatrixViewAllocation,
        validate_disjoint_matrix_views,
    )

    inputs, outputs = tuple(inputs), tuple(outputs)
    if not inputs or len(inputs) != len(outputs):
        raise ValueError("one private output per batch input")
    for source, destination in zip(inputs, outputs):
        replace(p, inputs=source, outputs=destination).validate()
    # Check request allocations jointly, not only one request at a time.
    ranges = [(x, x + p.input_values * 2) for x in inputs] + [
        (x, x + p.output_values * 2) for x in outputs
    ]
    for i, (a, b) in enumerate(ranges):
        if any(a < d and c < b for c, d in ranges[:i]):
            raise ValueError("batch input/output allocations overlap")
    views = []
    for index, k0 in enumerate(range(0, p.k, p.k_tile)):
        rows = (min(p.k_tile, p.k - k0) + 31) // 32 * 32
        views.append(
            View(
                index * 32,
                MatrixViewDescriptor(MatrixViewShape(rows, 32), MatrixViewMap(rows)),
            )
        )
    validate_disjoint_matrix_views(
        [
            MatrixViewAllocation(str(i), v.base, v.descriptor)
            for i, v in enumerate(views)
        ],
        mlen=2048,
        banks=64,
        bank_width=32,
        depth_rows=256,
    )
    e = Emitter()
    e.lines.append("; @stage=matrix_projection_shared_weight_panel")

    def transfer(row, hbm, store=False):
        e.address(1, row * p.mlen)
        e.address(2, hbm)
        e.lines.append(f"H_{'STORE' if store else 'PREFETCH'}_V gp1, gp2, a0, 0, 2")

    packets = iter(p.packets())
    for col in range(0, p.n, 32):
        for view in views:
            _, _, _, address = next(packets)
            e.dma(view, address)
        for source, destination in zip(inputs, outputs):
            row0 = col // 2048 * 2048
            transfer(1, p.zero if col == row0 else destination + row0 * 2)
            for k0, view in zip(range(0, p.k, p.k_tile), views):
                transfer(0, source + k0 * 2)
                e.view(0, view.descriptor)
                e.address(1, view.base)
                e.address(2, 0)
                e.lines.append("M_MV 0, gp1, gp2, 0")
            e.address(1, p.mlen + col - row0)
            e.lines.append("M_MV_WO gp1, 0")
            transfer(1, destination + row0 * 2, True)
    return "\n".join(e.lines) + "\n"


def lower_resident_projection(
    p: Projection, inputs=None, outputs=None, *, vector_rows=58
):
    """Keep private output rows and a bounded set of input windows in SRAM.

    The arithmetic order and weight packet ABI are unchanged. Rows 58..63
    remain available to the finite weight decoder in the common platform.
    Each cached input is an actual, owned 2048-element DMA window: this does
    not assume unaligned Vector reads or a left-shift/rotate instruction.
    Uncached windows stream through row zero. No compute/DMA overlap is used.

    Complete K-by-32 weight panels are shared when they fit Matrix SRAM.
    Larger K falls back to one request at a time, explicitly rereading weights.
    """
    from dataclasses import replace
    from compiler.aten.plena.mview import (
        MatrixViewAllocation,
        validate_disjoint_matrix_views,
    )

    inputs = (p.inputs,) if inputs is None else tuple(inputs)
    outputs = (p.outputs,) if outputs is None else tuple(outputs)
    if not inputs or len(inputs) != len(outputs) or len(inputs) > 16:
        raise ValueError("one private output per request, batch 1..16")
    if type(vector_rows) is not int or not len(inputs) + 1 <= vector_rows <= 64:
        raise ValueError(
            "Vector workspace must fit streaming input and private outputs"
        )
    for source, destination in zip(inputs, outputs):
        replace(p, inputs=source, outputs=destination).validate()
    ranges = [(x, x + p.input_values * 2) for x in inputs] + [
        (x, x + p.output_values * 2) for x in outputs
    ]
    for i, (a, b) in enumerate(ranges):
        if any(a < d and c < b for c, d in ranges[:i]):
            raise ValueError("batch input/output allocations overlap")
    if (p.k + p.k_tile - 1) // p.k_tile > 64 and len(inputs) > 1:
        return "".join(
            lower_resident_projection(
                replace(p, inputs=x, outputs=y), vector_rows=vector_rows
            )
            for x, y in zip(inputs, outputs)
        )

    starts = tuple(range(0, p.k, p.k_tile))
    shared = len(inputs) > 1
    views = [
        View(
            index * 32 if shared else 0,
            MatrixViewDescriptor(
                MatrixViewShape((min(p.k_tile, p.k - k0) + 31) // 32 * 32, 32),
                MatrixViewMap((min(p.k_tile, p.k - k0) + 31) // 32 * 32),
            ),
        )
        for index, k0 in enumerate(starts)
    ]
    if shared:
        validate_disjoint_matrix_views(
            [
                MatrixViewAllocation(str(i), v.base, v.descriptor)
                for i, v in enumerate(views)
            ],
            mlen=2048,
            banks=64,
            bank_width=32,
            depth_rows=256,
        )
    e = Emitter()
    e.lines.append(f"; @stage=matrix_projection_resident_rows workspace={vector_rows}")

    def transfer(row, hbm, store=False):
        e.address(1, row * p.mlen)
        e.address(2, hbm)
        e.lines.append(f"H_{'STORE' if store else 'PREFETCH'}_V gp1, gp2, a0, 0, 2")

    # K-major choice distributes a limited cache across requests. All windows
    # have equal reuse count (one per output32 block), so no profile is fitted.
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
            panel = [next(packets) for _ in starts]
            if shared:
                for view, (_, _, _, address) in zip(views, panel):
                    e.dma(view, address)
            for request, source in enumerate(inputs):
                for view, (packet_col, k0, _, address) in zip(views, panel):
                    assert packet_col == col
                    if not shared:
                        e.dma(view, address)
                    row = cache.get((request, k0), 0)
                    if row == 0:
                        transfer(0, source + k0 * 2)
                    e.view(0, view.descriptor)
                    e.address(1, view.base)
                    e.address(2, row * p.mlen)
                    e.lines.append("M_MV 0, gp1, gp2, 0")
                e.address(1, (request + 1) * p.mlen + col - row0)
                e.lines.append("M_MV_WO gp1, 0")
        for request, destination in enumerate(outputs):
            transfer(request + 1, destination + row0 * 2, True)
    return "\n".join(e.lines) + "\n"


def projection_config(rows: int, input_stride: int = 2048, output_stride: int = 2048) -> int:
    """Pack M_MM.P's GP configuration, with strides measured in BF16 elements.

    Bits 0..1 hold rows-1, bits 2..9 the input stride/256 and bits 10..17
    the output stride/32. Bits 18..31 are reserved zero. Zero strides are
    rejected even for one request, making accidental cross-request aliasing
    impossible when the same configuration is reused for a larger batch.
    """
    if type(rows) is not int or not 1 <= rows <= 4:
        raise ValueError("M_MM.P accepts one to four private request rows")
    if type(input_stride) is not int or input_stride % 256 or not 1 <= input_stride // 256 <= 255:
        raise ValueError("M_MM.P input stride must be 256..65280 in steps of 256")
    if type(output_stride) is not int or output_stride % 32 or not 1 <= output_stride // 32 <= 255:
        raise ValueError("M_MM.P output stride must be 32..8160 in steps of 32")
    return (rows - 1) | ((input_stride // 256) << 2) | ((output_stride // 32) << 10)


def lower_compact_projection(
    p: Projection, inputs=None, outputs=None, *, batch_tile=1, vector_rows=58
) -> str:
    """Use nonoverlapping input rows and bounded 1/4-request Matrix execution.

    The BF16 K256-by-N32 weight packet ABI is unchanged. A complete K-by-32
    weight panel lives in Matrix SRAM and is shared by every request before
    the next panel is loaded. Inputs use actual aligned 2048-element DMA rows,
    then M_MM.P selects one of their eight 256-element slices; this is explicit
    new input selection hardware, not a capability claimed for legacy M_MV.

    Vector rows 0..B-1 own one private output row each. The next batch_tile
    rows are streaming inputs. Remaining rows cache complete input chunks in
    K-major/request-group order. A cache entry always contains a contiguous
    group so both the input and output request strides are exactly 2048.
    Input DMA padding is owned by Projection.input_values. Rows at or above
    vector_rows remain available to the common decoder/caller. No overlap of
    DMA and compute, nor projection and recurrence, is assumed here.

    M_MM.P performs the historical per-column BF16 reduction, adds the tile
    contribution to the existing BF16 output slice, and writes only 32 output
    elements. The other output slices remain live until the full row store.
    """
    from dataclasses import replace
    from compiler.aten.plena.mview import (
        MatrixViewAllocation,
        validate_disjoint_matrix_views,
    )

    inputs = (p.inputs,) if inputs is None else tuple(inputs)
    outputs = (p.outputs,) if outputs is None else tuple(outputs)
    if not inputs or len(inputs) != len(outputs) or len(inputs) > 16:
        raise ValueError("one private output per request, batch 1..16")
    if type(batch_tile) is not int or batch_tile not in (1, 4):
        raise ValueError("compact projection batch_tile must be 1 or 4")
    if type(vector_rows) is not int or not len(inputs) + batch_tile <= vector_rows <= 64:
        raise ValueError("Vector workspace must fit private outputs and bounded streaming rows")
    if p.k_tile != 256:
        raise ValueError("compact projection requires the K256 weight packet ABI")
    for source, destination in zip(inputs, outputs):
        replace(p, inputs=source, outputs=destination).validate()
    ranges = [(x, x + p.input_values * 2) for x in inputs] + [
        (x, x + p.output_values * 2) for x in outputs
    ]
    for index, (a, b) in enumerate(ranges):
        if any(a < d and c < b for c, d in ranges[:index]):
            raise ValueError("batch input/output allocations overlap")

    starts = tuple(range(0, p.k, p.k_tile))
    if len(starts) > 64:
        raise ValueError("complete K256-by-32 weight panel exceeds Matrix SRAM capacity")
    views = [
        View(
            index * p.blen,
            MatrixViewDescriptor(
                MatrixViewShape((min(p.k_tile, p.k - k0) + 31) // 32 * 32, p.blen),
                MatrixViewMap((min(p.k_tile, p.k - k0) + 31) // 32 * 32),
            ),
        )
        for index, k0 in enumerate(starts)
    ]
    validate_disjoint_matrix_views(
        [MatrixViewAllocation(str(i), view.base, view.descriptor) for i, view in enumerate(views)],
        mlen=p.mlen, banks=64, bank_width=p.blen, depth_rows=256,
    )

    e = Emitter()
    e.lines.append(
        f"; @stage=matrix_projection_compact_rows workspace={vector_rows} batch_tile={batch_tile}"
    )

    def transfer(row, address, store=False):
        e.address(1, row * p.mlen)
        e.address(2, address)
        e.lines.append(f"H_{'STORE' if store else 'PREFETCH'}_V gp1, gp2, a0, 0, 2")

    batch = len(inputs)
    groups = [(first, min(batch_tile, batch - first)) for first in range(0, batch, batch_tile)]
    chunks = tuple(range(0, p.k, p.mlen))
    cache = {}
    next_row = batch + batch_tile
    for chunk in chunks:
        for first, count in groups:
            if next_row + count <= vector_rows:
                cache[chunk, first] = next_row
                for offset in range(count):
                    transfer(next_row + offset, inputs[first + offset] + chunk * 2)
                next_row += count

    packets = iter(p.packets())
    for row0 in range(0, p.n, p.mlen):
        for request in range(batch):
            transfer(request, p.zero)
        for column in range(row0, min(row0 + p.mlen, p.n), p.blen):
            for k0, view in zip(starts, views):
                packet_column, packet_k, _, address = next(packets)
                assert (packet_column, packet_k) == (column, k0)
                e.dma(view, address)
            for chunk in chunks:
                for first, count in groups:
                    input_row = cache.get((chunk, first), batch)
                    if (chunk, first) not in cache:
                        for offset in range(count):
                            transfer(input_row + offset, inputs[first + offset] + chunk * 2)
                    for k0 in range(chunk, min(chunk + p.mlen, p.k), p.k_tile):
                        view = views[k0 // p.k_tile]
                        e.view(0, view.descriptor)
                        e.address(1, first * p.mlen + column - row0)
                        e.address(2, view.base)
                        e.address(3, input_row * p.mlen + k0 - chunk)
                        e.address(4, projection_config(count))
                        e.lines.append("M_MM.P gp1, gp2, gp3, gp4, 0")
        for request, destination in enumerate(outputs):
            transfer(request, destination + row0 * 2, store=True)
    return "\n".join(e.lines) + "\n"
