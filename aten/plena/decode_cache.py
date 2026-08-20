"""Persistent decode-cache allocation and append helpers.

The cache backing is row-major HBM while producer tensors use PLENA's
column-block-major VRAM layout.  ``DecodeCacheTensor.append_row`` copies one
logical producer row into every row of a writeback-sized VRAM packet before
issuing ``H_STORE_V`` at the selected HBM row.  Only packet row zero is
logically valid; finite replicas in the guard rows prevent masked attention
lanes from observing uninitialized-VRAM NaNs.  The backing includes guard rows
because one store writes ``HBM_V_Writeback_Amount`` rows at a time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from compiler.aten.plena.vars import InputVar, VRAMMatrixVar


@dataclass(frozen=True)
class DecodeCacheTensor:
    """One BF16 row-major HBM tensor used by incremental decode."""

    backing: InputVar
    max_tokens: int
    storage_rows: int
    width: int
    element_bytes: int = 2
    hbm_precision: int = 1
    persistent: bool = True

    @property
    def row_bytes(self) -> int:
        return self.width * self.element_bytes

    @property
    def byte_capacity(self) -> int:
        return self.backing.hbm_size

    def prefix(self, token_count: int) -> InputVar:
        """Return a logical prefix view without registering overlapping HBM."""
        if not 1 <= token_count <= self.max_tokens:
            raise ValueError(
                f"{self.backing.name}: token_count={token_count} outside "
                f"[1, {self.max_tokens}]"
            )
        return InputVar(
            self.backing._program,
            self.backing.name,
            (token_count, self.width),
            self.backing.hbm_addr,
            self.backing.hbm_size,
            display_name=self.backing.display_name,
            physical_shape=(self.storage_rows, self.width),
        )

    def append_row(
        self,
        prog,
        source: VRAMMatrixVar,
        *,
        token_index: int,
        source_row: int = 0,
        name: str | None = None,
    ) -> None:
        """Append one producer row while respecting H_STORE_V granularity."""
        if not 0 <= token_index < self.max_tokens:
            raise ValueError(
                f"{self.backing.name}: token_index={token_index} outside "
                f"[0, {self.max_tokens})"
            )
        if not 0 <= source_row < source.shape[0]:
            raise ValueError(
                f"{self.backing.name}: source_row={source_row} outside "
                f"source rows={source.shape[0]}"
            )
        if source.shape[1] != self.width:
            raise ValueError(
                f"{self.backing.name}: source width={source.shape[1]} does not "
                f"match cache width={self.width}"
            )

        packet_rows = prog.hbm_v_writeback_amount
        if token_index + packet_rows > self.storage_rows:
            raise ValueError(
                f"{self.backing.name}: append at row {token_index} needs "
                f"{packet_rows} physical rows but storage has {self.storage_rows}"
            )
        packet_name = name or f"{self.backing.display_name}_append_t{token_index}"
        packet = prog.alloc(
            packet_name,
            rows=packet_rows,
            cols=self.width,
            strict=False,
            physical_shape=(packet_rows, self.width),
        )
        for packet_row in range(packet_rows):
            prog.vram_copy_region(
                packet,
                source,
                num_rows=1,
                num_cols=self.width,
                dst_row_offset=packet_row,
                src_row_offset=source_row,
            )
        prog.emit(
            f"; DECODE_CACHE_APPEND {self.backing.name} token={token_index} "
            f"row_bytes={self.row_bytes}\n"
        )
        prog.store_to_hbm(
            tensor_name=packet.name,
            hbm_addr=self.backing.hbm_addr + token_index * self.row_bytes,
            vlen=prog.mlen,
            precision=self.hbm_precision,
            hbm_element_bytes=self.element_bytes,
            hbm_real_data_ratio=float(self.element_bytes),
        )
        prog.free_tensor(packet)

    def overwrite_from(self, prog, source: VRAMMatrixVar) -> None:
        """Overwrite a reusable cache/scratch tile from a full VRAM tensor."""
        if source.shape[1] != self.width:
            raise ValueError(
                f"{self.backing.name}: source width={source.shape[1]} does not "
                f"match cache width={self.width}"
            )
        if source.physical_shape[0] > self.storage_rows:
            raise ValueError(
                f"{self.backing.name}: source physical rows={source.physical_shape[0]} "
                f"exceed storage rows={self.storage_rows}"
            )
        prog.emit(
            f"; DECODE_CACHE_OVERWRITE {self.backing.name} "
            f"rows={source.physical_shape[0]} width={self.width}\n"
        )
        prog.store_to_hbm(
            tensor_name=source.name,
            hbm_addr=self.backing.hbm_addr,
            vlen=prog.mlen,
            precision=self.hbm_precision,
            hbm_element_bytes=self.element_bytes,
            hbm_real_data_ratio=float(self.element_bytes),
        )


def allocate_decode_cache_tensor(
    prog,
    *,
    name: str,
    max_tokens: int,
    width: int,
    element_bytes: int = 2,
    hbm_precision: int = 1,
    persistent: bool = True,
) -> DecodeCacheTensor:
    """Reserve one cache tensor, including writeback guard rows."""
    if max_tokens <= 0:
        raise ValueError(f"max_tokens must be positive, got {max_tokens}")
    if width <= 0 or width % prog.mlen:
        raise ValueError(
            f"cache width must be a positive MLEN multiple, got width={width}, "
            f"MLEN={prog.mlen}"
        )
    if element_bytes <= 0:
        raise ValueError(f"element_bytes must be positive, got {element_bytes}")
    if hbm_precision not in (0, 1):
        raise ValueError(f"hbm_precision must be 0 or 1, got {hbm_precision}")

    guarded_rows = max_tokens + prog.hbm_v_writeback_amount - 1
    storage_rows = math.ceil(guarded_rows / prog.mlen) * prog.mlen
    backing = prog.input(
        name,
        shape=(max_tokens, width),
        physical_shape=(storage_rows, width),
        real_data_ratio=float(element_bytes),
    )
    return DecodeCacheTensor(
        backing=backing,
        max_tokens=max_tokens,
        storage_rows=storage_rows,
        width=width,
        element_bytes=element_bytes,
        hbm_precision=hbm_precision,
        persistent=persistent,
    )


__all__ = ["DecodeCacheTensor", "allocate_decode_cache_tensor"]
