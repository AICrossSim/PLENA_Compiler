"""FPRAM, FPVar, and tile-row operations for the PLENA DSL."""

from __future__ import annotations

from compiler.aten.plena.vars import FPVar, VRAMMatrixVar


class DslFPTileOpsMixin:
    # ========================================================================
    # FP Variable (FPRAM)
    # ========================================================================

    def allocate_fpram(
        self,
        internal_name: str,
        size: int = 1,
        display_name: str | None = None,
    ) -> FPVar:
        """
        Allocate FPRAM with explicit internal name and return FPVar proxy.
        """
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
        """
        Free FPRAM allocation by internal name.
        """
        super().free_fpram(internal_name, strict=strict)
        self._fp_vars.pop(internal_name, None)

    def fp_var(self, name: str, size: int = 1) -> FPVar:
        """
        Declare an FP variable in FPRAM.

        Allocates a contiguous region in FPRAM and returns an FPVar proxy.
        Within function scope, names are automatically prefixed.

        Args:
            name: variable name
            size: number of f16 elements to allocate (default 1)

        Returns:
            FPVar proxy object (use .address for ISA generation)

        Example:
            scale = prog.fp_var("scale")          # 1 element
            m_old = prog.fp_var("m_old", size=64) # 64 elements
            prog.compiler  # access compiler for ISA if needed
        """
        display_name = name
        internal_name = self._scoped_name(name)

        return self.allocate_fpram(
            internal_name=internal_name,
            size=size,
            display_name=display_name,
        )

    def save_fpram_state(self) -> int:
        """Save FPRAM allocator snapshot"""
        return super().save_fpram_state()

    def restore_fpram_state(self, snapshot: int):
        """Restore FPRAM allocator snapshot"""
        super().restore_fpram_state(snapshot)
        # Remove FPVar proxies that are no longer allocated in allocator.
        allocations = set(super().list_fpram_allocations())
        to_remove = [n for n in self._fp_vars if n not in allocations]
        for n in to_remove:
            del self._fp_vars[n]

    # ========================================================================
    # FPRAM Tile Operations
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
        rows: list[int] | None = None,
    ) -> list[int]:
        if row_idx is not None and rows is not None:
            raise ValueError("Provide either row_idx or rows, not both")
        if rows is not None:
            return rows
        if row_idx is not None:
            return [row_idx]
        return list(range(self.mlen))

    def tile_row_max(
        self,
        target_fpram_addr: int | FPVar,
        source: VRAMMatrixVar,
        row_idx: int | None = None,
        rows: list[int] | None = None,
        target_offset: int = 0,
        target_base_offset: int = 0,
    ):
        """
        Tile Row Max: reduce a single row to max, store to FPRAM address.

        Args:
            target_fpram_addr: FPRAM address or FPVar to write result
            source: VRAM tile (mlen x mlen)
            row_idx: single row index (legacy path)
            rows: multiple row indices
            target_offset: element offset when target_fpram_addr is FPVar
            target_base_offset: base offset for multi-row writes (contiguous)

        Example:
            m = prog.fp_var("m", size=1)
            S = prog.alloc("S", 64, 64)
            for row in range(64):
                prog.tile_row_max(m, S, rows=list(range(64)))
        """
        resolved_rows = self._resolve_rows(row_idx=row_idx, rows=rows)
        if len(resolved_rows) == 1:
            offsets = [target_offset]
        else:
            offsets = [target_base_offset + i for i in range(len(resolved_rows))]
        row_map = [(r, self._resolve_fpram_addr(target_fpram_addr, off)) for r, off in zip(resolved_rows, offsets)]
        super().tile_row_max(
            source_matrix=source.name,
            row_map=row_map,
        )

    def tile_row_sum(
        self,
        target_fpram_addr: int | FPVar,
        source: VRAMMatrixVar,
        row_idx: int | None = None,
        rows: list[int] | None = None,
        target_offset: int = 0,
        target_base_offset: int = 0,
    ):
        """
        Tile Row Sum: reduce a single row to sum, store to FPRAM address.

        Args:
            target_fpram_addr: FPRAM address or FPVar to write result
            source: VRAM tile (mlen x mlen)
            row_idx: single row index (legacy path)
            rows: multiple row indices
            target_offset: element offset when target_fpram_addr is FPVar
            target_base_offset: base offset for multi-row writes (contiguous)
        """
        resolved_rows = self._resolve_rows(row_idx=row_idx, rows=rows)
        if len(resolved_rows) == 1:
            offsets = [target_offset]
        else:
            offsets = [target_base_offset + i for i in range(len(resolved_rows))]
        row_map = [(r, self._resolve_fpram_addr(target_fpram_addr, off)) for r, off in zip(resolved_rows, offsets)]
        super().tile_row_sum(source.name, row_map)

    def tile_row_exp(
        self,
        source: VRAMMatrixVar,
        row_idx: int | None = None,
        rows: list[int] | None = None,
    ):
        """
        Tile Row Exp: in-place exp on specified rows.

        For each row i: source[i, :] = exp(source[i, :])
        """
        resolved_rows = self._resolve_rows(row_idx=row_idx, rows=rows)
        super().tile_row_exp(source.name, resolved_rows)

    def tile_row_reci(
        self,
        source: VRAMMatrixVar,
        rows: list[int] | None = None,
    ):
        """
        Tile Row Reciprocal: in-place 1/x on specified rows.

        For each row i: source[i, :] = 1.0 / source[i, :]
        """
        if rows is None:
            rows = list(range(self.mlen))
        super().tile_row_reci(source.name, rows)

    def tile_row_sub_fp(
        self,
        source: VRAMMatrixVar,
        fpram_addr: int | FPVar,
        row_idx: int | None = None,
        rows: list[int] | None = None,
        fpram_offset: int = 0,
        fpram_base_offset: int = 0,
    ):
        """
        Tile Row Sub FP: subtract FPRAM scalar from a single row.

        For row i: source[i, :] = source[i, :] - FPRAM[fpram_addr]
        """
        resolved_rows = self._resolve_rows(row_idx=row_idx, rows=rows)
        if len(resolved_rows) == 1:
            offsets = [fpram_offset]
        else:
            offsets = [fpram_base_offset + i for i in range(len(resolved_rows))]
        row_map = [(r, self._resolve_fpram_addr(fpram_addr, off)) for r, off in zip(resolved_rows, offsets)]
        super().tile_row_sub_fp(source.name, row_map)

    def tile_row_mul_fp(
        self,
        source: VRAMMatrixVar,
        fpram_addr: int | FPVar,
        row_idx: int | None = None,
        rows: list[int] | None = None,
        fpram_offset: int = 0,
        fpram_base_offset: int = 0,
    ):
        """
        Tile Row Mul FP: multiply a single row by FPRAM scalar.

        For row i: source[i, :] = source[i, :] * FPRAM[fpram_addr]
        """
        resolved_rows = self._resolve_rows(row_idx=row_idx, rows=rows)
        if len(resolved_rows) == 1:
            offsets = [fpram_offset]
        else:
            offsets = [fpram_base_offset + i for i in range(len(resolved_rows))]
        row_map = [(r, self._resolve_fpram_addr(fpram_addr, off)) for r, off in zip(resolved_rows, offsets)]
        super().tile_row_mul_fp(source.name, row_map)

    def tile_row_add_fp(
        self,
        source: VRAMMatrixVar,
        fp_var: FPVar,
        rows: list[int] | None = None,
    ):
        """
        Tile Row Add FP: add FPRAM scalar to each specified row.

        For each row i: source[i, :] = source[i, :] + fp_var[i]
        """
        if rows is None:
            rows = list(range(self.mlen))
        row_map = [(r, fp_var[r]) for r in rows]
        super().tile_row_add_fp(source.name, row_map)

    def tile_row_add(
        self,
        dst: VRAMMatrixVar,
        src: VRAMMatrixVar,
        rows: list[int] | None = None,
    ):
        """
        Tile Row Add: dst[i, :] += src[i, :] for specified rows.
        """
        if rows is None:
            rows = list(range(self.mlen))
        super().tile_row_add(dst.name, src.name, rows)

    def tile_row_sub(
        self,
        dst: VRAMMatrixVar,
        src: VRAMMatrixVar,
        rows: list[int] | None = None,
    ):
        """
        Tile Row Sub: dst[i, :] -= src[i, :] for specified rows.
        """
        if rows is None:
            rows = list(range(self.mlen))
        super().tile_row_sub(dst.name, src.name, rows)

    def tile_row_mul(
        self,
        dst: VRAMMatrixVar,
        src: VRAMMatrixVar,
        rows: list[int] | None = None,
    ):
        """
        Tile Row Mul: dst[i, :] *= src[i, :] for specified rows.
        """
        if rows is None:
            rows = list(range(self.mlen))
        super().tile_row_mul(dst.name, src.name, rows)

    def fpvar_reci(
        self,
        src: FPVar,
        dst: FPVar,
        count: int | None = None,
    ):
        """
        FPVar Reciprocal: compute 1/x for FPRAM scalar array.

        For each element i: dst[i] = 1.0 / src[i]

        Args:
            src: source FPVar
            dst: destination FPVar
            count: number of elements (default: min(src.size, dst.size))

        Example:
            l = prog.fp_var("l", size=64)
            inv_l = prog.fp_var("inv_l", size=64)
            prog.fpvar_reci(l, inv_l)  # inv_l = 1/l
        """
        if count is None:
            count = min(src.size, dst.size)
        if count > src.size or count > dst.size:
            raise ValueError(f"count={count} exceeds FPVar size: src.size={src.size}, dst.size={dst.size}")
        super().fpram_reci(src.name, dst.name, count)

    def fpvar_max(
        self,
        src1: FPVar,
        src2: FPVar,
        dst: FPVar,
        count: int | None = None,
    ):
        """
        FPVar Max: element-wise max for FPRAM scalar arrays.

        For each element i: dst[i] = max(src1[i], src2[i])

        Example:
            m_new = prog.fp_var("m_new", size=64)
            prog.fpvar_max(m_old, row_max, m_new)  # m_new = max(m_old, row_max)
        """
        if count is None:
            count = min(src1.size, src2.size, dst.size)
        super().fpram_max(src1.name, src2.name, dst.name, count)

    def fpvar_sub(
        self,
        src1: FPVar,
        src2: FPVar,
        dst: FPVar,
        count: int | None = None,
    ):
        """
        FPVar Subtract: element-wise subtraction for FPRAM scalar arrays.

        For each element i: dst[i] = src1[i] - src2[i]

        Example:
            diff = prog.fp_var("diff", size=64)
            prog.fpvar_sub(m_old, m_new, diff)  # diff = m_old - m_new
        """
        if count is None:
            count = min(src1.size, src2.size, dst.size)
        super().fpram_sub(src1.name, src2.name, dst.name, count)

    def fpvar_exp(
        self,
        src: FPVar,
        dst: FPVar,
        count: int | None = None,
    ):
        """
        FPVar Exp: element-wise exp for FPRAM scalar array.

        For each element i: dst[i] = exp(src[i])

        Example:
            m_res = prog.fp_var("m_res", size=64)
            prog.fpvar_exp(diff, m_res)  # m_res = exp(diff)
        """
        if count is None:
            count = min(src.size, dst.size)
        super().fpram_exp(src.name, dst.name, count)

    def fpvar_mul(
        self,
        src1: FPVar,
        src2: FPVar,
        dst: FPVar,
        count: int | None = None,
    ):
        """
        FPVar Multiply: element-wise multiplication for FPRAM scalar arrays.

        For each element i: dst[i] = src1[i] * src2[i]

        Example:
            result = prog.fp_var("result", size=64)
            prog.fpvar_mul(l_old, m_res, result)  # result = l_old * m_res
        """
        if count is None:
            count = min(src1.size, src2.size, dst.size)
        super().fpram_mul(src1.name, src2.name, dst.name, count)

    def fpvar_add(
        self,
        src1: FPVar,
        src2: FPVar,
        dst: FPVar,
        count: int | None = None,
    ):
        """
        FPVar Add: element-wise addition for FPRAM scalar arrays.

        For each element i: dst[i] = src1[i] + src2[i]

        Example:
            l_new = prog.fp_var("l_new", size=64)
            prog.fpvar_add(l_old, sum_p, l_new)  # l_new = l_old + sum_p
        """
        if count is None:
            count = min(src1.size, src2.size, dst.size)
        super().fpram_add(src1.name, src2.name, dst.name, count)

    def fpvar_copy(
        self,
        src: FPVar,
        dst: FPVar,
        count: int | None = None,
    ):
        """
        FPVar Copy: copy FPRAM scalar array.

        For each element i: dst[i] = src[i]

        Example:
            m_old_saved = prog.fp_var("m_old_saved", size=64)
            prog.fpvar_copy(m_old, m_old_saved)  # backup m_old
        """
        if count is None:
            count = min(src.size, dst.size)
        super().fpram_copy(src.name, dst.name, count)

    def fpvar_sum(
        self,
        src: FPVar,
        dst: FPVar,
        count: int | None = None,
    ):
        """
        FPVar Sum: reduction sum of src into dst[0] (via compiler FPRAM op).
        """
        if count is None:
            count = src.size
        super().fpram_sum(src.name, dst.name, count)

    def fpvar_shift(
        self,
        src: FPVar,
        dst: FPVar,
        shift: int,
        count: int | None = None,
        fill: FPVar | None = None,
    ):
        """
        FPVar Shift: shift src into dst, filling out-of-range slots with fill (default FPRAM zero).
        """
        if count is None:
            count = min(src.size, dst.size)
        fill_name = None if fill is None else fill.name
        super().fpram_shift(
            src_name=src.name,
            dst_name=dst.name,
            shift=shift,
            count=count,
            fill_fpram_name=fill_name,
        )

    def tile_row_mul_fp_broadcast(
        self,
        source: VRAMMatrixVar,
        fpram_scalar_addr: int | FPVar,
        row_idx: int | None = None,
        rows: list[int] | None = None,
        fpram_offset: int = 0,
    ):
        """
        Tile Row Mul FP Broadcast: multiply a single row by a FPRAM scalar.

        For row i: source[i, :] = source[i, :] * FPRAM[fpram_scalar_addr]

        Args:
            source: VRAM tile (mlen x mlen)
            fpram_scalar_addr: FPRAM address or FPVar of the scalar
            row_idx: single row index (legacy path)
            rows: multiple row indices

        Example:
            scale_fp = prog.fp_var("scale", size=1)
            for row in range(64):
                prog.tile_row_mul_fp_broadcast(S, scale_fp, rows=list(range(64)))
        """
        resolved_rows = self._resolve_rows(row_idx=row_idx, rows=rows)
        scalar_addr = self._resolve_fpram_addr(fpram_scalar_addr, fpram_offset)
        super().tile_row_mul_fp_broadcast(source.name, scalar_addr, resolved_rows)

    def fpvar_fill_from_fpram(
        self,
        dst: FPVar,
        src_fpram_addr: int,
        count: int | None = None,
    ):
        """
        FPVar Fill from FPRAM: fill all elements with a value from FPRAM.

        For each element i: dst[i] = FPRAM[src_fpram_addr]

        Args:
            dst: destination FPVar
            src_fpram_addr: source FPRAM address (e.g., 0 for 0.0, 2 for -inf)
            count: number of elements (default: dst.size)

        Example:
            m_old = prog.fp_var("m_old", size=64)
            prog.fpvar_fill_from_fpram(m_old, 2)  # fill with -inf from address 2
        """
        if count is None:
            count = dst.size
        super().fpram_fill_from_fpram(dst.name, src_fpram_addr, count)

    def vram_fill_zero(
        self,
        matrix: VRAMMatrixVar,
        rows: list[int] | None = None,
    ):
        """
        VRAM Fill Zero: fill specified rows with 0.

        Args:
            matrix: VRAM matrix
            rows: which rows to fill (default: all rows)

        Example:
            O = prog.alloc("O", 128, 128)
            prog.vram_fill_zero(O, rows=range(64, 128))  # zero out second half
        """
        if rows is None:
            rows = list(range(matrix.shape[0]))
        else:
            rows = list(rows)

        total_rows, cols = matrix.shape
        if any(row < 0 or row >= total_rows for row in rows):
            raise ValueError(f"vram_fill_zero rows out of bounds for {matrix.name}: shape={matrix.shape}, rows={rows}")

        # VRAM matrices are column-block-major. The low-level tile helper zeros
        # one 64-column tile, so walk every column block for wide matrices.
        num_col_blocks = (cols + self.mlen - 1) // self.mlen
        for col_block in range(num_col_blocks):
            super().vram_fill_zero(matrix.name, rows, tile_col_idx=col_block)


__all__ = ["DslFPTileOpsMixin"]
