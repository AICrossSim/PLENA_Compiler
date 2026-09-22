"""B1 rectangular-view Matrix projection with explicit HBM lifetime bounds.

Uses existing M_MV / M_MV_WO, not a new projection opcode. Weights are static
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
