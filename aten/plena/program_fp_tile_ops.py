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
        offsets = (
            [single_offset]
            if len(resolved_rows) == 1
            else [base_offset + i for i in range(len(resolved_rows))]
        )
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
    ):
        super().tile_row_exp(source.name, self._resolve_rows(row_idx=row_idx, rows=rows))

    def tile_row_reci(
        self,
        source: VRAMMatrixVar,
        rows: Iterable[int] | None = None,
    ):
        super().tile_row_reci(source.name, self._default_rows(rows))

    def tile_row_sub_fp(
        self,
        source: VRAMMatrixVar,
        fpram_addr: int | FPVar,
        row_idx: int | None = None,
        rows: Iterable[int] | None = None,
        fpram_offset: int = 0,
        fpram_base_offset: int = 0,
    ):
        return self._tile_row_fp_scalar(
            "tile_row_sub_fp", source, fpram_addr, row_idx, rows, fpram_offset, fpram_base_offset
        )

    def tile_row_mul_fp(
        self,
        source: VRAMMatrixVar,
        fpram_addr: int | FPVar,
        row_idx: int | None = None,
        rows: Iterable[int] | None = None,
        fpram_offset: int = 0,
        fpram_base_offset: int = 0,
    ):
        return self._tile_row_fp_scalar(
            "tile_row_mul_fp", source, fpram_addr, row_idx, rows, fpram_offset, fpram_base_offset
        )

    def tile_row_add_fp(
        self,
        source: VRAMMatrixVar,
        fp_var: FPVar,
        rows: Iterable[int] | None = None,
    ):
        resolved_rows = self._default_rows(rows)
        super().tile_row_add_fp(source.name, [(row, fp_var[row]) for row in resolved_rows])

    def _tile_row_binary(self, isa_method: str, dst: VRAMMatrixVar, src: VRAMMatrixVar, rows: Iterable[int] | None):
        return getattr(super(), isa_method)(dst.name, src.name, self._default_rows(rows))

    def _tile_row_fp_scalar(
        self,
        isa_method: str,
        source: VRAMMatrixVar,
        fpram_addr: int | FPVar,
        row_idx: int | None,
        rows: Iterable[int] | None,
        fpram_offset: int,
        fpram_base_offset: int,
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
        )

    def tile_row_add(
        self,
        dst: VRAMMatrixVar,
        src: VRAMMatrixVar,
        rows: Iterable[int] | None = None,
    ):
        return self._tile_row_binary("tile_row_add", dst, src, rows)

    def tile_row_sub(
        self,
        dst: VRAMMatrixVar,
        src: VRAMMatrixVar,
        rows: Iterable[int] | None = None,
    ):
        return self._tile_row_binary("tile_row_sub", dst, src, rows)

    def tile_row_mul(
        self,
        dst: VRAMMatrixVar,
        src: VRAMMatrixVar,
        rows: Iterable[int] | None = None,
    ):
        return self._tile_row_binary("tile_row_mul", dst, src, rows)

    def tile_row_mul_fp_broadcast(
        self,
        source: VRAMMatrixVar,
        fpram_scalar_addr: int | FPVar,
        row_idx: int | None = None,
        rows: Iterable[int] | None = None,
        fpram_offset: int = 0,
    ):
        scalar_addr = self._resolve_fpram_addr(fpram_scalar_addr, fpram_offset)
        super().tile_row_mul_fp_broadcast(source.name, scalar_addr, self._resolve_rows(row_idx=row_idx, rows=rows))

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
