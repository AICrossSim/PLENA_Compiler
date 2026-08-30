"""FPRAM, FPVar, and tile-row operations for the PLENA program builder."""

from __future__ import annotations

from collections.abc import Iterable

from compiler.aten.plena.vars import FPVar, VRAMMatrixVar


class ProgramFPTileOpsMixin:
    # ========================================================================
    # FP Variable (FPRAM)
    # ========================================================================

    def allocate_fpram(
        self,
        internal_name: str,
        size: int = 1,
        display_name: str | None = None,
    ) -> FPVar:
        """Allocate FPRAM with an explicit internal name and return an FPVar."""
        if size <= 0:
            raise ValueError(f"FPRAM allocation size must be positive, got {size}")

        address = super().allocate_fpram(internal_name, size)
        var = FPVar(
            self,
            internal_name,
            address,
            size,
            display_name=display_name if display_name is not None else internal_name,
        )
        self._fp_vars[internal_name] = var
        return var

    def free_fpram(self, internal_name: str, strict: bool = True):
        super().free_fpram(internal_name, strict=strict)
        self._fp_vars.pop(internal_name, None)

    def fp_var(self, name: str, size: int = 1) -> FPVar:
        return self.allocate_fpram(
            internal_name=self._scoped_name(name),
            size=size,
            display_name=name,
        )

    # ========================================================================
    # Shared argument normalization
    # ========================================================================

    def _resolve_fpram_addr(self, addr_or_var: int | FPVar, offset: int = 0) -> int:
        if isinstance(addr_or_var, FPVar):
            if offset < 0 or offset >= addr_or_var.size:
                raise ValueError(
                    f"FPVar offset out of range: offset={offset}, size={addr_or_var.size}, var={addr_or_var.name}"
                )
            return addr_or_var.address + offset
        if not isinstance(addr_or_var, int):
            raise TypeError(f"Expected int or FPVar, got {type(addr_or_var)}")
        return addr_or_var + offset

    def _resolve_rows(
        self,
        row_idx: int | None = None,
        rows: Iterable[int] | None = None,
    ) -> list[int]:
        if row_idx is not None and rows is not None:
            raise ValueError("Provide either row_idx or rows, not both")
        if rows is not None:
            return list(rows)
        if row_idx is not None:
            return [row_idx]
        return list(range(self.mlen))

    def _default_rows(self, rows: Iterable[int] | None, *, total_rows: int | None = None) -> list[int]:
        return list(range(self.mlen if total_rows is None else total_rows)) if rows is None else list(rows)

    def _fpram_row_map(
        self,
        fpram_addr: int | FPVar,
        *,
        row_idx: int | None = None,
        rows: Iterable[int] | None = None,
        single_offset: int = 0,
        base_offset: int = 0,
    ) -> list[tuple[int, int]]:
        resolved_rows = self._resolve_rows(row_idx=row_idx, rows=rows)
        offsets = [single_offset] if len(resolved_rows) == 1 else [base_offset + i for i in range(len(resolved_rows))]
        return [(row, self._resolve_fpram_addr(fpram_addr, offset)) for row, offset in zip(resolved_rows, offsets)]

    def _fp_count(self, vars_: Iterable[FPVar], count: int | None, *, default: int | None = None) -> int:
        fp_vars = list(vars_)
        resolved_count = default if count is None and default is not None else count
        if resolved_count is None:
            resolved_count = min(var.size for var in fp_vars)
        if any(resolved_count > var.size for var in fp_vars):
            sizes = ", ".join(f"{var.name}.size={var.size}" for var in fp_vars)
            raise ValueError(f"count={resolved_count} exceeds FPVar size: {sizes}")
        return resolved_count

    def _fpvar_unary(self, isa_method: str, src: FPVar, dst: FPVar, count: int | None = None):
        count = self._fp_count([src, dst], count)
        return getattr(super(), isa_method)(src.name, dst.name, count)

    def _fpvar_binary(self, isa_method: str, src1: FPVar, src2: FPVar, dst: FPVar, count: int | None = None):
        count = self._fp_count([src1, src2, dst], count)
        return getattr(super(), isa_method)(src1.name, src2.name, dst.name, count)

    # ========================================================================
    # FPRAM tile-row operations
    # ========================================================================

    def _tile_row_reduce_to_fpram(
        self,
        isa_method: str,
        target_fpram_addr: int | FPVar,
        source: VRAMMatrixVar,
        row_idx: int | None,
        rows: Iterable[int] | None,
        target_offset: int,
        target_base_offset: int,
    ):
        return getattr(super(), isa_method)(
            source.name,
            self._fpram_row_map(
                target_fpram_addr,
                row_idx=row_idx,
                rows=rows,
                single_offset=target_offset,
                base_offset=target_base_offset,
            ),
        )

    def tile_row_max(
        self,
        target_fpram_addr: int | FPVar,
        source: VRAMMatrixVar,
        row_idx: int | None = None,
        rows: Iterable[int] | None = None,
        target_offset: int = 0,
        target_base_offset: int = 0,
    ):
        return self._tile_row_reduce_to_fpram(
            "tile_row_max", target_fpram_addr, source, row_idx, rows, target_offset, target_base_offset
        )

    def tile_row_sum(
        self,
        target_fpram_addr: int | FPVar,
        source: VRAMMatrixVar,
        row_idx: int | None = None,
        rows: Iterable[int] | None = None,
        target_offset: int = 0,
        target_base_offset: int = 0,
    ):
        return self._tile_row_reduce_to_fpram(
            "tile_row_sum", target_fpram_addr, source, row_idx, rows, target_offset, target_base_offset
        )

    def tile_row_exp(
        self,
        source: VRAMMatrixVar,
        row_idx: int | None = None,
        rows: Iterable[int] | None = None,
        tile_col_idx: int = 0,
    ):
        super().tile_row_exp(source.name, self._resolve_rows(row_idx=row_idx, rows=rows), tile_col_idx=tile_col_idx)

    def tile_row_reci(
        self,
        source: VRAMMatrixVar,
        rows: Iterable[int] | None = None,
        tile_col_idx: int = 0,
    ):
        super().tile_row_reci(source.name, self._default_rows(rows), tile_col_idx=tile_col_idx)

    def tile_row_softplus(
        self,
        source: VRAMMatrixVar,
        rows: Iterable[int] | None = None,
        tile_col_idx: int = 0,
    ):
        """In-place `x = log(1 + exp(x))` over whole VRAM rows (V_SOFTPLUS_V)."""
        super().tile_row_softplus(source.name, self._default_rows(rows), tile_col_idx=tile_col_idx)

    def tile_row_to_fpram(
        self,
        source: VRAMMatrixVar,
        target_fpram_addr: int | FPVar,
        rows: Iterable[int] | None = None,
        target_base_offset: int = 0,
        tile_col_idx: int = 0,
    ):
        """Copy whole VRAM rows into FPRAM (S_MAP_FP_V), MLEN scalars per row.

        Unlike `tile_row_sum`/`tile_row_max`, which reduce a row to ONE FPRAM slot,
        this moves the row across intact. Row `rows[i]` lands at
        `target_fpram_addr + target_base_offset + i * MLEN`.
        """
        resolved_rows = self._default_rows(rows)
        row_map = [
            (row, self._resolve_fpram_addr(target_fpram_addr, target_base_offset + i * self.mlen))
            for i, row in enumerate(resolved_rows)
        ]
        return super().tile_row_to_fpram(source.name, row_map, tile_col_idx=tile_col_idx)

    def tile_row_sub_fp(
        self,
        source: VRAMMatrixVar,
        fpram_addr: int | FPVar,
        row_idx: int | None = None,
        rows: Iterable[int] | None = None,
        fpram_offset: int = 0,
        fpram_base_offset: int = 0,
        tile_col_idx: int = 0,
    ):
        return self._tile_row_fp_scalar(
            "tile_row_sub_fp", source, fpram_addr, row_idx, rows, fpram_offset, fpram_base_offset, tile_col_idx
        )

    def tile_row_mul_fp(
        self,
        source: VRAMMatrixVar,
        fpram_addr: int | FPVar,
        row_idx: int | None = None,
        rows: Iterable[int] | None = None,
        fpram_offset: int = 0,
        fpram_base_offset: int = 0,
        tile_col_idx: int = 0,
    ):
        return self._tile_row_fp_scalar(
            "tile_row_mul_fp", source, fpram_addr, row_idx, rows, fpram_offset, fpram_base_offset, tile_col_idx
        )

    def tile_row_max_fp(
        self,
        source: VRAMMatrixVar,
        fpram_addr: int | FPVar,
        row_idx: int | None = None,
        rows: Iterable[int] | None = None,
        fpram_offset: int = 0,
        fpram_base_offset: int = 0,
        tile_col_idx: int = 0,
    ):
        return self._tile_row_fp_scalar(
            "tile_row_max_fp", source, fpram_addr, row_idx, rows, fpram_offset, fpram_base_offset, tile_col_idx
        )

    def tile_row_min_fp(
        self,
        source: VRAMMatrixVar,
        fpram_addr: int | FPVar,
        row_idx: int | None = None,
        rows: Iterable[int] | None = None,
        fpram_offset: int = 0,
        fpram_base_offset: int = 0,
        tile_col_idx: int = 0,
    ):
        return self._tile_row_fp_scalar(
            "tile_row_min_fp", source, fpram_addr, row_idx, rows, fpram_offset, fpram_base_offset, tile_col_idx
        )

    def tile_row_add_fp(
        self,
        source: VRAMMatrixVar,
        fp_var: FPVar,
        rows: Iterable[int] | None = None,
        tile_col_idx: int = 0,
    ):
        resolved_rows = self._default_rows(rows)
        super().tile_row_add_fp(source.name, [(row, fp_var[row]) for row in resolved_rows], tile_col_idx=tile_col_idx)

    def _tile_row_binary(
        self,
        isa_method: str,
        dst: VRAMMatrixVar,
        src: VRAMMatrixVar,
        rows: Iterable[int] | None,
        tile_col_idx: int = 0,
    ):
        # One index for both operands: a binary row op is elementwise, so the
        # two tiles are the same width by construction and a column block of one
        # only ever pairs with the same block of the other. `vram_fill_zero`
        # walks the blocks itself because it takes a single matrix; here the
        # caller drives the loop, because which blocks are live differs per
        # kernel (prefill's `contrib` is wider than the range any one of its
        # three uses touches).
        return getattr(super(), isa_method)(
            dst.name,
            src.name,
            self._default_rows(rows),
            dst_tile_col_idx=tile_col_idx,
            src_tile_col_idx=tile_col_idx,
        )

    def _tile_row_fp_scalar(
        self,
        isa_method: str,
        source: VRAMMatrixVar,
        fpram_addr: int | FPVar,
        row_idx: int | None,
        rows: Iterable[int] | None,
        fpram_offset: int,
        fpram_base_offset: int,
        tile_col_idx: int = 0,
    ):
        return getattr(super(), isa_method)(
            source.name,
            self._fpram_row_map(
                fpram_addr,
                row_idx=row_idx,
                rows=rows,
                single_offset=fpram_offset,
                base_offset=fpram_base_offset,
            ),
            tile_col_idx=tile_col_idx,
        )

    def tile_row_add(
        self,
        dst: VRAMMatrixVar,
        src: VRAMMatrixVar,
        rows: Iterable[int] | None = None,
        tile_col_idx: int = 0,
    ):
        return self._tile_row_binary("tile_row_add", dst, src, rows, tile_col_idx)

    def tile_row_sub(
        self,
        dst: VRAMMatrixVar,
        src: VRAMMatrixVar,
        rows: Iterable[int] | None = None,
        tile_col_idx: int = 0,
    ):
        return self._tile_row_binary("tile_row_sub", dst, src, rows, tile_col_idx)

    def tile_row_mul(
        self,
        dst: VRAMMatrixVar,
        src: VRAMMatrixVar,
        rows: Iterable[int] | None = None,
        tile_col_idx: int = 0,
    ):
        return self._tile_row_binary("tile_row_mul", dst, src, rows, tile_col_idx)

    def tile_row_mul_fp_broadcast(
        self,
        source: VRAMMatrixVar,
        fpram_scalar_addr: int | FPVar,
        row_idx: int | None = None,
        rows: Iterable[int] | None = None,
        fpram_offset: int = 0,
        tile_col_idx: int = 0,
    ):
        scalar_addr = self._resolve_fpram_addr(fpram_scalar_addr, fpram_offset)
        super().tile_row_mul_fp_broadcast(
            source.name,
            scalar_addr,
            self._resolve_rows(row_idx=row_idx, rows=rows),
            tile_col_idx=tile_col_idx,
        )

    def tile_row_fma_fp_sweep(
        self,
        dst: VRAMMatrixVar,
        src: VRAMMatrixVar,
        fpram_base: int | FPVar,
        dst_rows: Iterable[int],
        src_rows: Iterable[int],
        fpram_offset: int = 0,
    ):
        """``dst[d] += src[s] * FPRAM[base + i]``, one slot per row pair.

        The FMA reads ``dst`` as well as writing it, so unlike every other tile
        helper here the destination must already hold what is being accumulated
        into -- typically a zeroed accumulator row or the state itself.
        """
        dst_rows, src_rows = list(dst_rows), list(src_rows)
        base = self._resolve_fpram_addr(fpram_base, fpram_offset)
        # The predecessor resolved -- and bounds-checked -- one slot per
        # iteration. This resolves the base once and lets the hardware walk the
        # rest, so an over-long sweep would read into whatever FPVar was
        # allocated next with nothing to say so. Check the far end too.
        if isinstance(fpram_base, FPVar) and dst_rows:
            end = fpram_offset + len(dst_rows) - 1
            if end >= fpram_base.size:
                raise ValueError(
                    f"FPRAM sweep of {len(dst_rows)} rows from offset "
                    f"{fpram_offset} reads slot {end}, past {fpram_base.name}'s "
                    f"{fpram_base.size}"
                )
        super().tile_row_fma_fp_sweep(dst.name, src.name, base, dst_rows, src_rows)

    def tile_row_fma_fp_broadcast(
        self,
        dst: VRAMMatrixVar,
        src: VRAMMatrixVar,
        fpram_scalar_addr: int | FPVar,
        dst_rows: Iterable[int],
        src_rows: Iterable[int],
        fpram_offset: int = 0,
    ):
        """``dst[d] += src[s] * FPRAM[addr]``, one slot for every row pair."""
        addr = self._resolve_fpram_addr(fpram_scalar_addr, fpram_offset)
        super().tile_row_fma_fp_broadcast(
            dst.name, src.name, addr, list(dst_rows), list(src_rows)
        )

    def tile_multirow_mul_fp(
        self,
        source: VRAMMatrixVar,
        fpram_base: int | FPVar,
        rows: Iterable[int],
        *,
        fpram_offset: int = 0,
        fp_step: int,
    ):
        resolved_rows = list(rows)
        base = self._resolve_fpram_addr(fpram_base, fpram_offset)
        if isinstance(fpram_base, FPVar) and resolved_rows and fp_step:
            end = fpram_offset + len(resolved_rows) - 1
            if end >= fpram_base.size:
                raise ValueError(
                    f"FPRAM packet sweep reads slot {end}, past {fpram_base.name}'s "
                    f"{fpram_base.size}"
                )
        super().tile_multirow_mul_fp(
            source.name,
            base,
            resolved_rows,
            fp_step=fp_step,
        )

    def tile_multirow_fma_fp_sweep(
        self,
        dst: VRAMMatrixVar,
        src: VRAMMatrixVar,
        fpram_base: int | FPVar,
        dst_rows: Iterable[int],
        src_rows: Iterable[int],
        *,
        fpram_offset: int = 0,
    ):
        dst_rows, src_rows = list(dst_rows), list(src_rows)
        base = self._resolve_fpram_addr(fpram_base, fpram_offset)
        if isinstance(fpram_base, FPVar) and dst_rows:
            end = fpram_offset + len(dst_rows) - 1
            if end >= fpram_base.size:
                raise ValueError(
                    f"FPRAM packet sweep reads slot {end}, past {fpram_base.name}'s "
                    f"{fpram_base.size}"
                )
        super().tile_multirow_fma_fp_sweep(
            dst.name,
            src.name,
            base,
            dst_rows,
            src_rows,
        )

    # One FPRAM slot applied to EVERY listed row, as opposed to the
    # `tile_row_*_fp` family which walks a different slot per row.
    def tile_row_add_fp_broadcast(
        self,
        source: VRAMMatrixVar,
        fpram_scalar_addr: int | FPVar,
        rows: Iterable[int] | None = None,
        fpram_offset: int = 0,
        tile_col_idx: int = 0,
    ):
        scalar_addr = self._resolve_fpram_addr(fpram_scalar_addr, fpram_offset)
        super().tile_row_add_fp_broadcast(
            source.name, scalar_addr, self._default_rows(rows), tile_col_idx=tile_col_idx
        )

    def tile_row_max_fp_broadcast(
        self,
        source: VRAMMatrixVar,
        fpram_scalar_addr: int | FPVar,
        rows: Iterable[int] | None = None,
        fpram_offset: int = 0,
        tile_col_idx: int = 0,
    ):
        scalar_addr = self._resolve_fpram_addr(fpram_scalar_addr, fpram_offset)
        super().tile_row_max_fp_broadcast(
            source.name, scalar_addr, self._default_rows(rows), tile_col_idx=tile_col_idx
        )

    def tile_row_min_fp_broadcast(
        self,
        source: VRAMMatrixVar,
        fpram_scalar_addr: int | FPVar,
        rows: Iterable[int] | None = None,
        fpram_offset: int = 0,
        tile_col_idx: int = 0,
    ):
        scalar_addr = self._resolve_fpram_addr(fpram_scalar_addr, fpram_offset)
        super().tile_row_min_fp_broadcast(
            source.name, scalar_addr, self._default_rows(rows), tile_col_idx=tile_col_idx
        )

    def tile_row_sub_fp_broadcast(
        self,
        source: VRAMMatrixVar,
        fpram_scalar_addr: int | FPVar,
        rows: Iterable[int] | None = None,
        fpram_offset: int = 0,
        reverse: bool = False,
        tile_col_idx: int = 0,
    ):
        """``row -= c``, or ``row = c - row`` when ``reverse``.

        The reverse form is `V_SUB_VF`'s `rorder=1`. Mamba's decay matrix builds
        ``cs_i - cs_j`` from a row of ``cs_j`` and a scalar ``cs_i`` with it.
        """
        scalar_addr = self._resolve_fpram_addr(fpram_scalar_addr, fpram_offset)
        super().tile_row_sub_fp_broadcast(
            source.name,
            scalar_addr,
            self._default_rows(rows),
            reverse=reverse,
            tile_col_idx=tile_col_idx,
        )

    # ========================================================================
    # FPVar operations
    # ========================================================================

    def fpvar_reci(self, src: FPVar, dst: FPVar, count: int | None = None):
        return self._fpvar_unary("fpram_reci", src, dst, count)

    def fpvar_exp(self, src: FPVar, dst: FPVar, count: int | None = None):
        return self._fpvar_unary("fpram_exp", src, dst, count)

    def fpvar_copy(self, src: FPVar, dst: FPVar, count: int | None = None):
        return self._fpvar_unary("fpram_copy", src, dst, count)

    def fpvar_max(self, src1: FPVar, src2: FPVar, dst: FPVar, count: int | None = None):
        return self._fpvar_binary("fpram_max", src1, src2, dst, count)

    def fpvar_sub(self, src1: FPVar, src2: FPVar, dst: FPVar, count: int | None = None):
        return self._fpvar_binary("fpram_sub", src1, src2, dst, count)

    def fpvar_mul(self, src1: FPVar, src2: FPVar, dst: FPVar, count: int | None = None):
        return self._fpvar_binary("fpram_mul", src1, src2, dst, count)

    def fpvar_add(self, src1: FPVar, src2: FPVar, dst: FPVar, count: int | None = None):
        return self._fpvar_binary("fpram_add", src1, src2, dst, count)

    def fpvar_sum(self, src: FPVar, dst: FPVar, count: int | None = None):
        count = self._fp_count([src], count, default=src.size)
        return super().fpram_sum(src.name, dst.name, count)

    def fpvar_shift(
        self,
        src: FPVar,
        dst: FPVar,
        shift: int,
        count: int | None = None,
        fill: FPVar | None = None,
    ):
        count = self._fp_count([src, dst], count)
        return super().fpram_shift(
            src_name=src.name,
            dst_name=dst.name,
            shift=shift,
            count=count,
            fill_fpram_name=None if fill is None else fill.name,
        )

    def fpvar_fill_from_fpram(
        self,
        dst: FPVar,
        src_fpram_addr: int,
        count: int | None = None,
    ):
        count = self._fp_count([dst], count, default=dst.size)
        return super().fpram_fill_from_fpram(dst.name, src_fpram_addr, count)

    def vram_fill_zero(
        self,
        matrix: VRAMMatrixVar,
        rows: Iterable[int] | None = None,
    ):
        resolved_rows = self._default_rows(rows, total_rows=matrix.shape[0])
        total_rows, cols = matrix.shape
        if any(row < 0 or row >= total_rows for row in resolved_rows):
            raise ValueError(
                f"vram_fill_zero rows out of bounds for {matrix.name}: shape={matrix.shape}, rows={resolved_rows}"
            )

        # VRAM matrices are column-block-major. The low-level helper zeros one
        # tile column, so walk every column block for wide matrices.
        num_col_blocks = (cols + self.mlen - 1) // self.mlen
        for col_block in range(num_col_blocks):
            super().vram_fill_zero(matrix.name, resolved_rows, tile_col_idx=col_block)


__all__ = ["ProgramFPTileOpsMixin"]
