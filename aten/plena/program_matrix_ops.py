"""Matrix projection, RoPE, and VRAM operations for the PLENA program builder."""

from __future__ import annotations

import math

from compiler.aten.plena.vars import InputVar, TensorVar, VRAMMatrixVar

MAX_K_TILES = 4  # MRAM capacity: 4 x mlen^2 elements


def _iter_k_chunks(num_k_tiles: int):
    k_start = 0
    while k_start < num_k_tiles:
        k_end = min(k_start + MAX_K_TILES, num_k_tiles)
        yield k_start, k_end - k_start
        k_start = k_end


class ProgramMatrixOpsMixin:
    # ========================================================================
    # Matrix Projection and VRAM Operations
    # ========================================================================

    def _require_var(self, value, expected_type, label: str):
        if not isinstance(value, expected_type):
            raise TypeError(f"{label} must be {expected_type.__name__}, got {type(value)}")
        return value

    def _ensure_hbm_sub_matrix_registered(self, input_var: InputVar):
        """Ensure an HBM input is registered in compiler sub-matrix manager."""
        if self._registered_hbm_sub_matrices.get(input_var.name):
            return
        h, w = input_var.shape
        super().ensure_hbm_sub_matrix(
            name=input_var.name,
            hbm_addr=input_var.hbm_addr,
            shape=(h, w),
            real_data_ratio=self.real_data_ratio,
        )
        self._registered_hbm_sub_matrices[input_var.name] = True

    def _ensure_vram_sub_matrix_registered(self, matrix_var: VRAMMatrixVar):
        """Ensure a VRAM matrix is registered in compiler sub-matrix manager."""
        if self._registered_vram_sub_matrices.get(matrix_var.name):
            return
        super().ensure_vram_matrix_layout(
            name=matrix_var.name,
            shape=matrix_var.shape,
        )
        self._registered_vram_sub_matrices[matrix_var.name] = True

    def _prepare_projection(self, vram_matrix, mram_input, target, auto_reset_mram: bool):
        vram_matrix = self._require_var(vram_matrix, VRAMMatrixVar, "vram_matrix")
        mram_input = self._require_var(mram_input, InputVar, "mram_input")
        target = self._require_var(target, VRAMMatrixVar, "target")
        self._ensure_vram_sub_matrix_registered(vram_matrix)
        self._ensure_hbm_sub_matrix_registered(mram_input)
        if auto_reset_mram:
            super().reset_mram()
        return vram_matrix, mram_input, target

    def vram_sub_projection_to(
        self,
        vram_matrix: VRAMMatrixVar,
        vram_row_idx: int,
        mram_input: InputVar,
        mram_col_idx: int,
        target: VRAMMatrixVar,
        target_row_idx: int,
        target_col_idx: int,
        auto_reset_mram: bool = True,
        k_block_start: int = 0,
        k_block_count: int | None = None,
    ):
        """
        target[target_row_idx][target_col_idx] = vram_matrix[vram_row_idx][:] @ mram_input[:][mram_col_idx]
        Supports K-split: k_block_start/k_block_count select a subset of K tiles.
        """
        vram_matrix, mram_input, target = self._prepare_projection(
            vram_matrix, mram_input, target, auto_reset_mram
        )
        super().load_sub_matrix_col(
            name=mram_input.name,
            col_idx=mram_col_idx,
            k_block_start=k_block_start,
            k_block_count=k_block_count,
        )
        super().vram_sub_projection_to(
            vram_mat_name=vram_matrix.name,
            vram_row_idx=vram_row_idx,
            mram_mat_name=mram_input.name,
            mram_col_idx=mram_col_idx,
            target_matrix=target.name,
            target_row_idx=target_row_idx,
            target_col_idx=target_col_idx,
            k_block_start=k_block_start,
            k_block_count=k_block_count,
        )

    def vram_sub_projection_T_to(
        self,
        vram_matrix: VRAMMatrixVar,
        vram_row_idx: int,
        mram_input: InputVar,
        mram_row_idx: int,
        target: VRAMMatrixVar,
        target_row_idx: int,
        target_col_idx: int,
        auto_reset_mram: bool = True,
    ):
        """
        target[target_row_idx][target_col_idx] = vram_matrix[vram_row_idx][:] @ mram_input[mram_row_idx][:]^T
        """
        vram_matrix, mram_input, target = self._prepare_projection(
            vram_matrix, mram_input, target, auto_reset_mram
        )
        super().load_sub_matrix_row(name=mram_input.name, row_idx=mram_row_idx)
        super().vram_sub_projection_T_to(
            vram_mat_name=vram_matrix.name,
            vram_row_idx=vram_row_idx,
            mram_mat_name=mram_input.name,
            mram_row_idx=mram_row_idx,
            target_matrix=target.name,
            target_row_idx=target_row_idx,
            target_col_idx=target_col_idx,
        )

    def linear_projection(self, input_var: VRAMMatrixVar, weight_var: InputVar, name: str = "linear_out"):
        """Emit tiled PLENA linear projection, including K-split accumulation."""
        mlen = self.mlen

        rows, k_total = input_var.shape
        _, out_features = weight_var.shape
        num_row_blocks = math.ceil(rows / mlen)
        if out_features % mlen != 0:
            raise ValueError(f"out_features ({out_features}) must be a multiple of mlen ({mlen})")
        num_col_blocks = out_features // mlen
        num_k_tiles = math.ceil(k_total / mlen)

        # When rows is not a multiple of mlen the hardware still operates on
        # full tiles; only the first `rows` rows contain valid output.
        output = self.alloc(name, rows, out_features, strict=rows % mlen == 0)

        def emit_projection(row_idx, col_idx, target, target_row_idx, target_col_idx, **k_split):
            self.vram_sub_projection_to(
                input_var,
                row_idx,
                weight_var,
                col_idx,
                target,
                target_row_idx,
                target_col_idx,
                **k_split,
            )

        if num_k_tiles <= MAX_K_TILES:
            for col_idx in range(num_col_blocks):
                for row_idx in range(num_row_blocks):
                    emit_projection(row_idx, col_idx, output, row_idx, col_idx)
            return output

        # Temp buffer for one partial-sum tile. Allocating the full output shape
        # here can overlap with the real output for wide projections.
        temp = self.alloc(f"{name}_temp", mlen, mlen)
        for k_chunk_idx, (k_block_start, k_block_count) in enumerate(_iter_k_chunks(num_k_tiles)):
            k_split = {
                "k_block_start": k_block_start,
                "k_block_count": k_block_count,
            }
            for col_idx in range(num_col_blocks):
                for row_idx in range(num_row_blocks):
                    if k_chunk_idx == 0:
                        emit_projection(row_idx, col_idx, output, row_idx, col_idx, **k_split)
                    else:
                        emit_projection(row_idx, col_idx, temp, 0, 0, **k_split)
                        self.vram_block_add_to(
                            output,
                            row_idx,
                            col_idx,
                            temp,
                            0,
                            0,
                            output,
                            row_idx,
                            col_idx,
                        )
        self.free_tensor(temp)
        return output

    def linear(self, input_var: VRAMMatrixVar, weight_var: InputVar):
        """Default linear op compatibility surface."""
        return self.linear_projection(input_var, weight_var)

    # ========================================================================
    # RoPE (1D Positional Encoding)
    # ========================================================================

    def rope(
        self,
        x_var: VRAMMatrixVar,
        x_rot_var: VRAMMatrixVar,
        cos_var: VRAMMatrixVar,
        sin_var: VRAMMatrixVar,
    ) -> VRAMMatrixVar:
        """Apply Rotary Position Embedding in-place: x = x * cos + rotate_half(x) * sin

        x_rot_var must already be in VRAM as rotate_half(x), preloaded by caller.
        Returns x_var (modified in-place).
        """
        super().rope(
            x_name=x_var.name,
            x_rot_name=x_rot_var.name,
            cos_name=cos_var.name,
            sin_name=sin_var.name,
        )
        return x_var

    # ========================================================================
    # VRAM Matrix Addition
    # ========================================================================

    def vram_add(
        self,
        dst: VRAMMatrixVar,
        src: VRAMMatrixVar,
        dst_row_offset: int = 0,
        src_row_offset: int = 0,
        num_rows: int | None = None,
    ):
        """VRAM matrix add: dst[row_offset:] += src"""
        super().vram_matrix_add(
            dst_matrix=dst.name,
            src_matrix=src.name,
            dst_row_offset=dst_row_offset,
            src_row_offset=src_row_offset,
            num_rows=num_rows,
        )

    def embedding_add(self, input_var: VRAMMatrixVar, pos_weight_var: VRAMMatrixVar):
        """Add learned/positional embedding weights to input in-place."""
        self.vram_add(input_var, pos_weight_var)
        return input_var

    def vram_block_add_to(
        self,
        src1: TensorVar,
        src1_row_idx: int,
        src1_col_idx: int,
        src2: TensorVar,
        src2_row_idx: int,
        src2_col_idx: int,
        target: TensorVar,
        target_row_idx: int,
        target_col_idx: int,
    ):
        """
        mlen x mlen block add:
            target[target_row_idx][target_col_idx] =
                src1[src1_row_idx][src1_col_idx] + src2[src2_row_idx][src2_col_idx]

        Supports writing back to the same matrix/block (in-place overwrite).
        """
        src1 = self._require_var(src1, VRAMMatrixVar, "src1")
        src2 = self._require_var(src2, VRAMMatrixVar, "src2")
        target = self._require_var(target, VRAMMatrixVar, "target")

        super().vram_block_add_to(
            src1_matrix=src1.name,
            src1_row_idx=src1_row_idx,
            src1_col_idx=src1_col_idx,
            src2_matrix=src2.name,
            src2_row_idx=src2_row_idx,
            src2_col_idx=src2_col_idx,
            target_matrix=target.name,
            target_row_idx=target_row_idx,
            target_col_idx=target_col_idx,
        )


__all__ = ["ProgramMatrixOpsMixin"]
