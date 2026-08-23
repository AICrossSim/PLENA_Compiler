"""Matrix projection, RoPE, and VRAM operations for the PLENA program builder."""

from __future__ import annotations

import math

from compiler.asm_templates._imm import load_large_int
from compiler.aten.isa_builder import IsaBuilder, addr as areg, gp
from compiler.aten.plena.vars import InputVar, TensorVar, VRAMMatrixVar


def _iter_k_chunks(num_k_tiles: int, max_k_tiles: int):
    if max_k_tiles <= 0:
        raise ValueError(f"max_k_tiles must be > 0, got {max_k_tiles}")
    k_start = 0
    while k_start < num_k_tiles:
        k_end = min(k_start + max_k_tiles, num_k_tiles)
        yield k_start, k_end - k_start
        k_start = k_end


def _matrix_precision_code(matrix_precision: str | int) -> int:
    if isinstance(matrix_precision, int):
        if matrix_precision not in (0, 1):
            raise ValueError(
                f"matrix_precision int must be 0 or 1, got {matrix_precision}"
            )
        return matrix_precision
    normalized = matrix_precision.lower().replace("-", "_")
    if normalized in {"weight", "weights", "hbm_m_weight", "hbm_m_weight_type"}:
        return 0
    if normalized in {"kv", "keyvalue", "key_value", "hbm_m_kv", "hbm_m_kv_type"}:
        return 1
    raise ValueError(f"unknown matrix_precision={matrix_precision!r}")


class ProgramMatrixOpsMixin:
    # ========================================================================
    # Matrix Projection and VRAM Operations
    # ========================================================================

    def _require_var(self, value, expected_type, label: str):
        if not isinstance(value, expected_type):
            raise TypeError(
                f"{label} must be {expected_type.__name__}, got {type(value)}"
            )
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
            physical_shape=input_var.physical_shape,
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
            physical_shape=matrix_var.physical_shape,
        )
        self._registered_vram_sub_matrices[matrix_var.name] = True

    def _prepare_projection(
        self, vram_matrix, mram_input, target, auto_reset_mram: bool
    ):
        vram_matrix = self._require_var(vram_matrix, VRAMMatrixVar, "vram_matrix")
        mram_input = self._require_var(mram_input, InputVar, "mram_input")
        target = self._require_var(target, VRAMMatrixVar, "target")
        self._ensure_vram_sub_matrix_registered(vram_matrix)
        self._ensure_hbm_sub_matrix_registered(mram_input)
        if auto_reset_mram:
            super().reset_mram()
        return vram_matrix, mram_input, target

    def _compact_projection_vram_addr(
        self,
        matrix: VRAMMatrixVar,
        *,
        row_idx: int = 0,
        tile_col_idx: int = 0,
    ) -> int:
        """Return one logical-row address in the column-major VRAM tile layout."""
        row_block = row_idx // self.mlen
        row_in_block = row_idx % self.mlen
        return (
            self.get_vram_tile_addr(matrix.name, row_block, tile_col_idx)
            + row_in_block * self.mlen
        )

    def _compact_row_major_linear_projection(
        self,
        input_var: VRAMMatrixVar,
        weight_var: InputVar,
        *,
        output_col_offset: int,
        output_features: int,
        name: str,
        physical_shape: tuple[int, int],
        matrix_precision: str | int,
        set_scale: bool,
        hbm_element_bytes: int,
    ) -> VRAMMatrixVar:
        """Emit a size-bounded decode GEMM with runtime output-column loops.

        K chunks stay static because each chunk has at most the MRAM tile
        capacity. The potentially very wide N traversal is represented by a
        ``C_LOOP``. HBM keeps the existing row-major tensor format, so this is
        binary compaction rather than a new weight layout.
        """
        input_var = self._require_var(input_var, VRAMMatrixVar, "input_var")
        weight_var = self._require_var(weight_var, InputVar, "weight_var")
        rows, _ = input_var.shape
        physical_rows, physical_out_features = physical_shape
        if rows < 1 or rows > self.blen:
            raise NotImplementedError(
                f"{name}: compact Matrix loops support decode rows in [1, BLEN={self.blen}], got {rows}"
            )
        if physical_rows < rows or physical_rows > self.mlen:
            raise ValueError(
                f"{name}: physical rows must cover the logical rows and fit one MLEN tile, "
                f"got logical={rows}, physical={physical_rows}, MLEN={self.mlen}"
            )
        if hbm_element_bytes not in (1, 2):
            raise ValueError(
                f"{name}: compact Matrix loops support 1- or 2-byte HBM elements, "
                f"got {hbm_element_bytes}"
            )
        if output_col_offset < 0 or output_col_offset % self.mlen:
            raise ValueError(
                f"{name}: output_col_offset={output_col_offset} must be a non-negative MLEN multiple"
            )
        if output_features <= 0 or output_features % self.mlen:
            raise ValueError(
                f"{name}: output_features={output_features} must be a positive MLEN multiple"
            )
        if physical_out_features < output_features or physical_out_features % self.mlen:
            raise ValueError(
                f"{name}: physical output width={physical_out_features} must cover "
                f"{output_features} and be MLEN-aligned"
            )
        if output_col_offset + output_features > weight_var.physical_shape[1]:
            raise ValueError(
                f"{name}: requested columns [{output_col_offset}, "
                f"{output_col_offset + output_features}) exceed weight physical width "
                f"{weight_var.physical_shape[1]}"
            )

        physical_k = max(input_var.physical_shape[1], weight_var.physical_shape[0])
        weight_rows, weight_cols = weight_var.physical_shape
        if physical_k != weight_rows or physical_k % self.mlen:
            raise ValueError(
                f"{name}: compact Matrix K must be identical and MLEN-aligned across "
                f"activation/weight storage, got activation={input_var.physical_shape[1]}, "
                f"weight={weight_rows}"
            )
        if weight_cols % self.mlen:
            raise ValueError(
                f"{name}: weight physical width={weight_cols} must be MLEN-aligned"
            )
        data_plane_bytes = weight_rows * weight_cols * hbm_element_bytes
        if data_plane_bytes > 1 << 32:
            raise NotImplementedError(
                f"{name}: one row-major weight data plane must fit the 32-bit H_PREFETCH_M "
                f"offset, got {data_plane_bytes} bytes"
            )
        row_stride_bytes = weight_cols * hbm_element_bytes
        if row_stride_bytes >= 1 << 32:
            raise ValueError(f"{name}: HBM row stride does not fit 32 bits")
        if set_scale and weight_rows * weight_cols >= 1 << 32:
            raise ValueError(f"{name}: MX scale-plane offset does not fit 32 bits")

        self._ensure_vram_sub_matrix_registered(input_var)
        self._ensure_hbm_sub_matrix_registered(weight_var)
        output = self.alloc(
            name,
            rows,
            output_features,
            strict=False,
            physical_shape=physical_shape,
        )
        self._ensure_vram_sub_matrix_registered(output)

        num_k_tiles = physical_k // self.mlen
        num_col_tiles = output_features // self.mlen
        chunks = tuple(_iter_k_chunks(num_k_tiles, self.mram_tile_capacity))
        scratch = None
        if len(chunks) > 1:
            scratch = self.alloc(
                f"{name}_compact_k_scratch",
                rows=rows,
                cols=self.mlen,
                strict=False,
                physical_shape=(physical_rows, self.mlen),
            )
            self._ensure_vram_sub_matrix_registered(scratch)

        # VRAM/MRAM addresses are element addresses; HBM offsets are bytes.
        input_stride = input_var.physical_shape[0] * self.mlen
        output_stride = output.physical_shape[0] * self.mlen
        block_size = self.mlen * self.mlen
        precision = _matrix_precision_code(matrix_precision)

        regs = self._reg.allocate_gp(12)
        (
            gp_col_loop,
            gp_micro_loop,
            gp_k_loop,
            gp_col_offset,
            gp_prefetch_offset,
            gp_mram,
            gp_act,
            gp_mat,
            gp_target,
            gp_target_base,
            gp_accum,
            gp_work,
        ) = regs
        addr_reg = self._reg.allocate_addr(1)[0]
        try:
            super().reset_mram()
            asm = IsaBuilder().comment(
                f"compact row-major Matrix projection {name}: "
                f"Ktiles={num_k_tiles}, Ntiles={num_col_tiles}"
            )
            asm.extend(load_large_int(gp_work, weight_var.hbm_addr >> 32))
            asm.extend(
                load_large_int(gp_prefetch_offset, weight_var.hbm_addr & 0xFFFF_FFFF)
            )
            asm.instr(
                "C_SET_ADDR_REG",
                areg(addr_reg),
                gp(gp_work),
                gp(gp_prefetch_offset),
            )
            if set_scale:
                asm.extend(load_large_int(gp_work, weight_rows * weight_cols))
                asm.instr("C_SET_SCALE_REG", gp(gp_work))
            asm.extend(load_large_int(gp_work, row_stride_bytes))
            asm.instr("C_SET_STRIDE_REG", gp(gp_work))

            first_col_offset = output_col_offset * hbm_element_bytes
            for chunk_index, (k_start, k_count) in enumerate(chunks):
                target = output if chunk_index == 0 else scratch
                assert target is not None
                asm.comment(
                    f"compact K chunk {chunk_index}: tiles [{k_start}, {k_start + k_count})"
                )
                asm.extend(load_large_int(gp_col_offset, first_col_offset))
                if chunk_index == 0:
                    asm.extend(
                        load_large_int(
                            gp_target_base,
                            self._compact_projection_vram_addr(output),
                        )
                    )
                else:
                    asm.extend(
                        load_large_int(
                            gp_target_base,
                            self._compact_projection_vram_addr(target),
                        )
                    )
                    asm.extend(
                        load_large_int(
                            gp_accum,
                            self._compact_projection_vram_addr(output),
                        )
                    )

                asm.instr("C_LOOP_START", gp(gp_col_loop), num_col_tiles)
                for local_k, row_idx in enumerate(range(k_start, k_start + k_count)):
                    row_base = row_idx * self.mlen * row_stride_bytes
                    asm.extend(load_large_int(gp_prefetch_offset, row_base))
                    asm.instr(
                        "S_ADD_INT",
                        gp(gp_prefetch_offset),
                        gp(gp_prefetch_offset),
                        gp(gp_col_offset),
                    )
                    asm.extend(load_large_int(gp_mram, local_k * block_size))
                    asm.instr(
                        "H_PREFETCH_M",
                        gp(gp_mram),
                        gp(gp_prefetch_offset),
                        areg(addr_reg),
                        1,
                        precision,
                    )

                asm.instr("S_ADDI_INT", gp(gp_mram), gp(0), 0)
                asm.instr("S_ADDI_INT", gp(gp_target), gp(gp_target_base), 0)
                asm.instr(
                    "C_LOOP_START",
                    gp(gp_micro_loop),
                    self.mlen // self.blen,
                )
                asm.extend(
                    load_large_int(
                        gp_act,
                        self._compact_projection_vram_addr(
                            input_var,
                            tile_col_idx=k_start,
                        ),
                    )
                )
                asm.instr("S_ADDI_INT", gp(gp_mat), gp(gp_mram), 0)
                asm.instr("C_LOOP_START", gp(gp_k_loop), k_count)
                asm.instr("M_MM", 0, gp(gp_mat), gp(gp_act))
                asm.instr("S_ADDI_INT", gp(gp_act), gp(gp_act), input_stride)
                asm.instr("S_ADDI_INT", gp(gp_mat), gp(gp_mat), block_size)
                asm.instr("C_LOOP_END", gp(gp_k_loop))
                asm.instr("M_MM_WO", gp(gp_target), gp(0), 0)
                asm.instr(
                    "S_ADDI_INT",
                    gp(gp_mram),
                    gp(gp_mram),
                    self.blen * self.mlen,
                )
                asm.instr("S_ADDI_INT", gp(gp_target), gp(gp_target), self.blen)
                asm.instr("C_LOOP_END", gp(gp_micro_loop))

                if chunk_index:
                    asm.instr("S_ADDI_INT", gp(gp_target), gp(gp_target_base), 0)
                    asm.instr("C_LOOP_START", gp(gp_k_loop), rows)
                    asm.instr(
                        "V_ADD_VV",
                        gp(gp_accum),
                        gp(gp_accum),
                        gp(gp_target),
                        0,
                    )
                    asm.instr("S_ADDI_INT", gp(gp_accum), gp(gp_accum), self.mlen)
                    asm.instr("S_ADDI_INT", gp(gp_target), gp(gp_target), self.mlen)
                    asm.instr("C_LOOP_END", gp(gp_k_loop))
                    asm.instr(
                        "S_ADDI_INT",
                        gp(gp_accum),
                        gp(gp_accum),
                        output_stride - rows * self.mlen,
                    )
                else:
                    asm.instr(
                        "S_ADDI_INT",
                        gp(gp_target_base),
                        gp(gp_target_base),
                        output_stride,
                    )

                asm.instr(
                    "S_ADDI_INT",
                    gp(gp_col_offset),
                    gp(gp_col_offset),
                    self.mlen * hbm_element_bytes,
                )
                asm.instr("C_LOOP_END", gp(gp_col_loop))

            self._emit(asm)
        finally:
            self._reg.free_gp(regs)
            self._reg.free_addr([addr_reg])
        if scratch is not None:
            self.free_tensor(scratch)
        return output

    def _compact_row_major_stream_k_accum_projection(
        self,
        input_var: VRAMMatrixVar,
        weight_var: InputVar,
        *,
        name: str,
        physical_shape: tuple[int, int],
        max_k_tiles: int,
    ) -> VRAMMatrixVar:
        """Compact BF16 decode GEMM that writes once after all K chunks.

        Router logits are sensitive to a BF16 round-trip between K chunks.
        This lowering reloads MRAM for each output micro-column but preserves
        the Matrix FP32 accumulator until every K tile has contributed. Both N
        and the output micro-column traversal are hardware loops.
        """
        input_var = self._require_var(input_var, VRAMMatrixVar, "input_var")
        weight_var = self._require_var(weight_var, InputVar, "weight_var")
        rows, _ = input_var.shape
        out_features = weight_var.shape[1]
        physical_rows, physical_out_features = physical_shape
        if rows < 1 or rows > self.blen:
            raise NotImplementedError(
                f"{name}: compact stream-K projection supports rows in "
                f"[1, BLEN={self.blen}], got {rows}"
            )
        if physical_rows < rows or physical_rows > self.mlen:
            raise ValueError(
                f"{name}: physical rows={physical_rows} must cover {rows} rows "
                f"and fit one MLEN tile"
            )
        if physical_out_features < out_features or physical_out_features % self.mlen:
            raise ValueError(
                f"{name}: physical output width={physical_out_features} must cover "
                f"{out_features} and be MLEN-aligned"
            )
        if max_k_tiles <= 0 or max_k_tiles > self.mram_tile_capacity:
            raise ValueError(
                f"{name}: max_k_tiles={max_k_tiles} must be in "
                f"[1, MRAM capacity={self.mram_tile_capacity}]"
            )

        weight_rows, weight_cols = weight_var.physical_shape
        if input_var.physical_shape[1] != weight_rows or weight_rows % self.mlen:
            raise ValueError(
                f"{name}: activation K={input_var.physical_shape[1]} and weight "
                f"K={weight_rows} must match and be MLEN-aligned"
            )
        if weight_cols != physical_out_features:
            raise ValueError(
                f"{name}: compact stream-K requires output storage to match the "
                f"weight physical width, got output={physical_out_features}, weight={weight_cols}"
            )
        data_plane_bytes = weight_rows * weight_cols * 2
        if data_plane_bytes > 1 << 32:
            raise NotImplementedError(
                f"{name}: BF16 weight data plane exceeds the 32-bit prefetch offset"
            )
        row_stride_bytes = weight_cols * 2

        self._ensure_vram_sub_matrix_registered(input_var)
        self._ensure_hbm_sub_matrix_registered(weight_var)
        output = self.alloc(
            name,
            rows,
            out_features,
            strict=False,
            physical_shape=physical_shape,
        )
        self._ensure_vram_sub_matrix_registered(output)

        num_k_tiles = weight_rows // self.mlen
        num_col_tiles = weight_cols // self.mlen
        chunks = tuple(_iter_k_chunks(num_k_tiles, max_k_tiles))
        input_stride = input_var.physical_shape[0] * self.mlen
        output_stride = output.physical_shape[0] * self.mlen
        block_size = self.mlen * self.mlen

        regs = self._reg.allocate_gp(12)
        (
            gp_col_loop,
            gp_micro_loop,
            gp_k_loop,
            gp_col_offset,
            gp_prefetch_offset,
            gp_mram,
            gp_act,
            gp_mat,
            gp_target,
            gp_target_base,
            gp_micro_offset,
            gp_work,
        ) = regs
        addr_reg = self._reg.allocate_addr(1)[0]
        try:
            super().reset_mram()
            asm = IsaBuilder().comment(
                f"compact BF16 stream-K Matrix projection {name}: "
                f"Ktiles={num_k_tiles}, Ntiles={num_col_tiles}"
            )
            asm.extend(load_large_int(gp_work, weight_var.hbm_addr >> 32))
            asm.extend(
                load_large_int(gp_prefetch_offset, weight_var.hbm_addr & 0xFFFF_FFFF)
            )
            asm.instr(
                "C_SET_ADDR_REG",
                areg(addr_reg),
                gp(gp_work),
                gp(gp_prefetch_offset),
            )
            asm.extend(load_large_int(gp_work, row_stride_bytes))
            asm.instr("C_SET_STRIDE_REG", gp(gp_work))
            asm.instr("S_ADDI_INT", gp(gp_col_offset), gp(0), 0)
            asm.extend(
                load_large_int(
                    gp_target_base,
                    self._compact_projection_vram_addr(output),
                )
            )
            asm.instr("C_LOOP_START", gp(gp_col_loop), num_col_tiles)
            asm.instr("S_ADDI_INT", gp(gp_micro_offset), gp(0), 0)
            asm.instr("S_ADDI_INT", gp(gp_target), gp(gp_target_base), 0)
            asm.instr(
                "C_LOOP_START",
                gp(gp_micro_loop),
                self.mlen // self.blen,
            )

            for chunk_index, (k_start, k_count) in enumerate(chunks):
                asm.comment(
                    f"stream-K chunk {chunk_index}: tiles [{k_start}, {k_start + k_count})"
                )
                for local_k, row_idx in enumerate(range(k_start, k_start + k_count)):
                    row_base = row_idx * self.mlen * row_stride_bytes
                    asm.extend(load_large_int(gp_prefetch_offset, row_base))
                    asm.instr(
                        "S_ADD_INT",
                        gp(gp_prefetch_offset),
                        gp(gp_prefetch_offset),
                        gp(gp_col_offset),
                    )
                    asm.extend(load_large_int(gp_mram, local_k * block_size))
                    asm.instr(
                        "H_PREFETCH_M",
                        gp(gp_mram),
                        gp(gp_prefetch_offset),
                        areg(addr_reg),
                        1,
                        1,
                    )

                asm.extend(
                    load_large_int(
                        gp_act,
                        self._compact_projection_vram_addr(
                            input_var,
                            tile_col_idx=k_start,
                        ),
                    )
                )
                asm.instr("S_ADDI_INT", gp(gp_mat), gp(gp_micro_offset), 0)
                asm.instr("C_LOOP_START", gp(gp_k_loop), k_count)
                asm.instr("M_MM", 0, gp(gp_mat), gp(gp_act))
                asm.instr("S_ADDI_INT", gp(gp_act), gp(gp_act), input_stride)
                asm.instr("S_ADDI_INT", gp(gp_mat), gp(gp_mat), block_size)
                asm.instr("C_LOOP_END", gp(gp_k_loop))

            asm.instr("M_MM_WO", gp(gp_target), gp(0), 0)
            asm.instr(
                "S_ADDI_INT",
                gp(gp_micro_offset),
                gp(gp_micro_offset),
                self.blen * self.mlen,
            )
            asm.instr("S_ADDI_INT", gp(gp_target), gp(gp_target), self.blen)
            asm.instr("C_LOOP_END", gp(gp_micro_loop))
            asm.instr(
                "S_ADDI_INT",
                gp(gp_col_offset),
                gp(gp_col_offset),
                self.mlen * 2,
            )
            asm.instr(
                "S_ADDI_INT",
                gp(gp_target_base),
                gp(gp_target_base),
                output_stride,
            )
            asm.instr("C_LOOP_END", gp(gp_col_loop))
            self._emit(asm)
        finally:
            self._reg.free_gp(regs)
            self._reg.free_addr([addr_reg])
        return output

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
        matrix_precision: str | int = "weights",
        set_scale: bool = True,
        hbm_element_bytes: int = 1,
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
            precision=_matrix_precision_code(matrix_precision),
            set_scale=set_scale,
            hbm_element_bytes=hbm_element_bytes,
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
        matrix_precision: str | int = "weights",
        set_scale: bool = True,
        hbm_element_bytes: int = 1,
    ):
        """
        target[target_row_idx][target_col_idx] = vram_matrix[vram_row_idx][:] @ mram_input[mram_row_idx][:]^T
        """
        vram_matrix, mram_input, target = self._prepare_projection(
            vram_matrix, mram_input, target, auto_reset_mram
        )
        super().load_sub_matrix_row(
            name=mram_input.name,
            row_idx=mram_row_idx,
            precision=_matrix_precision_code(matrix_precision),
            set_scale=set_scale,
            hbm_element_bytes=hbm_element_bytes,
        )
        super().vram_sub_projection_T_to(
            vram_mat_name=vram_matrix.name,
            vram_row_idx=vram_row_idx,
            mram_mat_name=mram_input.name,
            mram_row_idx=mram_row_idx,
            target_matrix=target.name,
            target_row_idx=target_row_idx,
            target_col_idx=target_col_idx,
        )

    def vram_sub_projection_stream_k_accum_to(
        self,
        vram_matrix: VRAMMatrixVar,
        vram_row_idx: int,
        mram_input: InputVar,
        mram_col_idx: int,
        target: VRAMMatrixVar,
        target_row_idx: int,
        target_col_idx: int,
        *,
        max_k_tiles: int,
        matrix_precision: str | int = "keyvalue",
        set_scale: bool = False,
        hbm_element_bytes: int = 2,
    ):
        """Project one output tile while keeping K chunks in the FP32 accumulator.

        The ordinary wide-K path materializes each chunk to BF16 VRAM and then
        adds chunks with vector ops.  That is fine for most projections but can
        flip Qwen router top-k near rank boundaries.  This helper is deliberately
        narrow: it reloads MRAM per 4x4 output microtile and only writes once,
        preserving the matrix-machine accumulator across K chunks.
        """
        vram_matrix, mram_input, target = self._prepare_projection(
            vram_matrix, mram_input, target, auto_reset_mram=True
        )
        if max_k_tiles <= 0:
            raise ValueError(f"max_k_tiles must be > 0, got {max_k_tiles}")

        vram_layout = self.vram_matrices[vram_matrix.name]
        vram_row_blocks = vram_layout.get_row_blocks(vram_row_idx)
        physical_k = max(vram_matrix.physical_shape[1], mram_input.physical_shape[0])
        num_k_tiles = math.ceil(physical_k / self.mlen)
        tiles_per_mlen = self.mlen // self.blen
        valid_rows = (
            vram_row_blocks[0].valid_shape[0]
            if vram_row_blocks[0].valid_shape
            else self.mlen
        )
        row_loop_count = min(tiles_per_mlen, max(1, math.ceil(valid_rows / self.blen)))
        chunks = list(_iter_k_chunks(num_k_tiles, max_k_tiles))

        for micro_col_idx in range(tiles_per_mlen):
            for micro_row_idx in range(row_loop_count):
                for chunk_idx, (k_block_start, k_block_count) in enumerate(chunks):
                    super().reset_mram()
                    super().load_sub_matrix_col(
                        name=mram_input.name,
                        col_idx=mram_col_idx,
                        k_block_start=k_block_start,
                        k_block_count=k_block_count,
                        precision=_matrix_precision_code(matrix_precision),
                        set_scale=set_scale,
                        hbm_element_bytes=hbm_element_bytes,
                    )
                    super().vram_sub_projection_microtile_accumulate_to(
                        vram_mat_name=vram_matrix.name,
                        vram_row_idx=vram_row_idx,
                        mram_mat_name=mram_input.name,
                        mram_col_idx=mram_col_idx,
                        target_matrix=target.name,
                        target_row_idx=target_row_idx,
                        target_col_idx=target_col_idx,
                        micro_row_idx=micro_row_idx,
                        micro_col_idx=micro_col_idx,
                        k_block_start=k_block_start,
                        k_block_count=k_block_count,
                        write_out=(chunk_idx == len(chunks) - 1),
                    )

    def vram_sub_projection_packed_skinny_stream_k_accum_to(
        self,
        vram_matrix: VRAMMatrixVar,
        vram_row_idx: int,
        packed_mram_input: InputVar,
        packed_col_base_idx: int,
        target: VRAMMatrixVar,
        target_row_idx: int,
        target_col_idx: int,
        *,
        max_k_tiles_per_packed_tile: int,
        matrix_precision: str | int = "keyvalue",
        set_scale: bool = False,
        hbm_element_bytes: int = 2,
    ):
        """Compile-only packed-skinny projection probe.

        ``packed_mram_input`` is not a normal weight matrix.  It is expected to
        contain one full HBM tile per output micro-column and K group.  Within a
        tile, consecutive skinny K slices occupy columns
        ``0:blen, blen:2*blen, ...``.  The helper proves cap8-equivalent router
        scheduling can fit in one MRAM tile per K group while preserving M_MM's
        existing full-tile contract.
        """
        vram_matrix, packed_mram_input, target = self._prepare_projection(
            vram_matrix,
            packed_mram_input,
            target,
            auto_reset_mram=True,
        )
        if max_k_tiles_per_packed_tile <= 0:
            raise ValueError(
                f"max_k_tiles_per_packed_tile must be > 0, got {max_k_tiles_per_packed_tile}"
            )

        vram_layout = self.vram_matrices[vram_matrix.name]
        vram_row_blocks = vram_layout.get_row_blocks(vram_row_idx)
        num_k_tiles = len(vram_row_blocks)
        tiles_per_mlen = self.mlen // self.blen
        if max_k_tiles_per_packed_tile > tiles_per_mlen:
            raise ValueError(
                f"packed tile can hold at most {tiles_per_mlen} skinny slices, "
                f"got {max_k_tiles_per_packed_tile}"
            )

        packed_layout = self.hbm_matrices[packed_mram_input.name]
        chunks = list(_iter_k_chunks(num_k_tiles, max_k_tiles_per_packed_tile))
        if packed_layout.num_row_blocks < len(chunks):
            raise ValueError(
                f"packed_mram_input has {packed_layout.num_row_blocks} row groups, "
                f"but {len(chunks)} are needed"
            )
        if packed_layout.num_col_blocks < packed_col_base_idx + tiles_per_mlen:
            raise ValueError(
                f"packed_mram_input has {packed_layout.num_col_blocks} col blocks, "
                f"but base {packed_col_base_idx} plus {tiles_per_mlen} micro-columns are needed"
            )

        valid_rows = (
            vram_row_blocks[0].valid_shape[0]
            if vram_row_blocks[0].valid_shape
            else self.mlen
        )
        row_loop_count = min(tiles_per_mlen, max(1, math.ceil(valid_rows / self.blen)))

        for micro_col_idx in range(tiles_per_mlen):
            packed_col_idx = packed_col_base_idx + micro_col_idx
            for micro_row_idx in range(row_loop_count):
                for group_idx, (k_block_start, k_block_count) in enumerate(chunks):
                    super().reset_mram()
                    super().load_sub_matrix(
                        name=packed_mram_input.name,
                        row_idx=group_idx,
                        col_idx=packed_col_idx,
                        mram_dest_addr=0,
                        precision=_matrix_precision_code(matrix_precision),
                        set_scale=set_scale,
                        hbm_element_bytes=hbm_element_bytes,
                    )
                    super().vram_sub_projection_packed_skinny_microtile_accumulate_to(
                        vram_mat_name=vram_matrix.name,
                        vram_row_idx=vram_row_idx,
                        packed_mram_mat_name=packed_mram_input.name,
                        packed_group_idx=group_idx,
                        packed_col_idx=packed_col_idx,
                        target_matrix=target.name,
                        target_row_idx=target_row_idx,
                        target_col_idx=target_col_idx,
                        micro_row_idx=micro_row_idx,
                        micro_col_idx=micro_col_idx,
                        k_block_start=k_block_start,
                        k_block_count=k_block_count,
                        write_out=(group_idx == len(chunks) - 1),
                    )

    def linear_projection(
        self,
        input_var: VRAMMatrixVar,
        weight_var: InputVar,
        name: str = "linear_out",
        physical_shape: tuple[int, int] | None = None,
        matrix_precision: str | int = "weights",
        set_scale: bool = True,
        hbm_element_bytes: int = 1,
    ):
        """Emit tiled PLENA linear projection, including K-split accumulation."""
        mlen = self.mlen

        rows, k_total = input_var.shape
        _, out_features = weight_var.shape
        if physical_shape is None:
            physical_rows = max(
                input_var.physical_shape[0], math.ceil(rows / self.blen) * self.blen
            )
            physical_out_features = weight_var.physical_shape[1]
        else:
            physical_rows, physical_out_features = physical_shape
            if physical_rows < rows or physical_out_features < out_features:
                raise ValueError(
                    f"physical_shape {physical_shape} cannot be smaller than "
                    f"logical output {(rows, out_features)}"
                )
        physical_k = max(input_var.physical_shape[1], weight_var.physical_shape[0])
        num_row_blocks = math.ceil(physical_rows / mlen)
        num_col_blocks = math.ceil(physical_out_features / mlen)
        num_k_tiles = math.ceil(physical_k / mlen)
        max_k_tiles = self.mram_tile_capacity

        if self.compact_matrix_loops and rows <= self.blen:
            return self._compact_row_major_linear_projection(
                input_var,
                weight_var,
                output_col_offset=0,
                output_features=out_features,
                name=name,
                physical_shape=(physical_rows, physical_out_features),
                matrix_precision=matrix_precision,
                set_scale=set_scale,
                hbm_element_bytes=hbm_element_bytes,
            )

        # When rows is not a multiple of mlen the hardware still operates on
        # full tiles; only the first `rows` rows contain valid output.
        output = self.alloc(
            name,
            rows,
            out_features,
            strict=False,
            physical_shape=(physical_rows, physical_out_features),
        )

        def emit_projection(
            row_idx, col_idx, target, target_row_idx, target_col_idx, **k_split
        ):
            self.vram_sub_projection_to(
                input_var,
                row_idx,
                weight_var,
                col_idx,
                target,
                target_row_idx,
                target_col_idx,
                matrix_precision=matrix_precision,
                set_scale=set_scale,
                hbm_element_bytes=hbm_element_bytes,
                **k_split,
            )

        if num_k_tiles <= max_k_tiles:
            for col_idx in range(num_col_blocks):
                for row_idx in range(num_row_blocks):
                    emit_projection(row_idx, col_idx, output, row_idx, col_idx)
            return output

        # Temp buffer for one partial-sum tile. Allocating the full output shape
        # here can overlap with the real output for wide projections.
        temp = self.alloc(f"{name}_temp", mlen, mlen)
        for k_chunk_idx, (k_block_start, k_block_count) in enumerate(
            _iter_k_chunks(num_k_tiles, max_k_tiles)
        ):
            k_split = {
                "k_block_start": k_block_start,
                "k_block_count": k_block_count,
            }
            for col_idx in range(num_col_blocks):
                for row_idx in range(num_row_blocks):
                    if k_chunk_idx == 0:
                        emit_projection(
                            row_idx, col_idx, output, row_idx, col_idx, **k_split
                        )
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

    def linear_projection_slice(
        self,
        input_var: VRAMMatrixVar,
        weight_var: InputVar,
        *,
        output_col_offset: int,
        output_features: int,
        name: str,
        physical_shape: tuple[int, int] | None = None,
        matrix_precision: str | int = "weights",
        set_scale: bool = True,
        hbm_element_bytes: int = 1,
    ) -> VRAMMatrixVar:
        """Project one MLEN-aligned output-column slice of a wide weight.

        Kimi MLA consumes Q/K/V one head at a time. Materializing all 96 heads
        simultaneously exceeds Vector SRAM, so this helper streams only the
        requested columns from the existing row-major HBM weight and writes a
        compact output whose first column is the slice's first column.
        """
        if output_col_offset < 0 or output_col_offset % self.mlen:
            raise ValueError(
                f"output_col_offset={output_col_offset} must be a non-negative MLEN multiple"
            )
        if output_features <= 0 or output_features % self.mlen:
            raise ValueError(
                f"output_features={output_features} must be a positive MLEN multiple"
            )
        if output_col_offset + output_features > weight_var.shape[1]:
            raise ValueError(
                f"slice [{output_col_offset}, {output_col_offset + output_features}) "
                f"exceeds weight width={weight_var.shape[1]}"
            )

        rows, _ = input_var.shape
        if physical_shape is None:
            physical_rows = max(
                input_var.physical_shape[0],
                math.ceil(rows / self.blen) * self.blen,
            )
            physical_out_features = output_features
        else:
            physical_rows, physical_out_features = physical_shape
        if physical_rows < rows or physical_out_features < output_features:
            raise ValueError(
                f"physical_shape {(physical_rows, physical_out_features)} cannot cover "
                f"logical output {(rows, output_features)}"
            )

        if self.compact_matrix_loops and rows <= self.blen:
            return self._compact_row_major_linear_projection(
                input_var,
                weight_var,
                output_col_offset=output_col_offset,
                output_features=output_features,
                name=name,
                physical_shape=(physical_rows, physical_out_features),
                matrix_precision=matrix_precision,
                set_scale=set_scale,
                hbm_element_bytes=hbm_element_bytes,
            )

        output = self.alloc(
            name,
            rows,
            output_features,
            strict=False,
            physical_shape=(physical_rows, physical_out_features),
        )
        physical_k = max(input_var.physical_shape[1], weight_var.physical_shape[0])
        num_k_tiles = math.ceil(physical_k / self.mlen)
        num_row_blocks = math.ceil(physical_rows / self.mlen)
        num_output_blocks = output_features // self.mlen
        weight_col_base = output_col_offset // self.mlen
        chunks = tuple(_iter_k_chunks(num_k_tiles, self.mram_tile_capacity))
        temp = None
        if len(chunks) > 1:
            temp = self.alloc(f"{name}_temp", self.mlen, self.mlen)

        for chunk_index, (k_block_start, k_block_count) in enumerate(chunks):
            k_split = {
                "k_block_start": k_block_start,
                "k_block_count": k_block_count,
            }
            for local_col in range(num_output_blocks):
                weight_col = weight_col_base + local_col
                for row_idx in range(num_row_blocks):
                    if chunk_index == 0:
                        self.vram_sub_projection_to(
                            input_var,
                            row_idx,
                            weight_var,
                            weight_col,
                            output,
                            row_idx,
                            local_col,
                            matrix_precision=matrix_precision,
                            set_scale=set_scale,
                            hbm_element_bytes=hbm_element_bytes,
                            **k_split,
                        )
                    else:
                        assert temp is not None
                        self.vram_sub_projection_to(
                            input_var,
                            row_idx,
                            weight_var,
                            weight_col,
                            temp,
                            0,
                            0,
                            matrix_precision=matrix_precision,
                            set_scale=set_scale,
                            hbm_element_bytes=hbm_element_bytes,
                            **k_split,
                        )
                        self.vram_block_add_to(
                            output,
                            row_idx,
                            local_col,
                            temp,
                            0,
                            0,
                            output,
                            row_idx,
                            local_col,
                        )
        if temp is not None:
            self.free_tensor(temp)
        return output

    def linear_projection_bf16_stream_k_accum(
        self,
        input_var: VRAMMatrixVar,
        weight_var: InputVar,
        name: str = "linear_out_bf16_stream_k_accum",
        physical_shape: tuple[int, int] | None = None,
        max_k_tiles: int | None = None,
    ):
        """BF16 projection with cross-K-chunk matrix accumulator retention."""
        mlen = self.mlen
        rows, _k_total = input_var.shape
        _weight_rows, out_features = weight_var.shape
        if physical_shape is None:
            physical_rows = max(
                input_var.physical_shape[0], math.ceil(rows / self.blen) * self.blen
            )
            physical_out_features = weight_var.physical_shape[1]
        else:
            physical_rows, physical_out_features = physical_shape
            if physical_rows < rows or physical_out_features < out_features:
                raise ValueError(
                    f"physical_shape {physical_shape} cannot be smaller than "
                    f"logical output {(rows, out_features)}"
                )

        physical_k = max(input_var.physical_shape[1], weight_var.physical_shape[0])
        num_row_blocks = math.ceil(physical_rows / mlen)
        num_col_blocks = math.ceil(physical_out_features / mlen)
        num_k_tiles = math.ceil(physical_k / mlen)
        max_tiles = self.mram_tile_capacity if max_k_tiles is None else max_k_tiles

        if self.compact_matrix_loops and rows <= self.blen:
            return self._compact_row_major_stream_k_accum_projection(
                input_var,
                weight_var,
                name=name,
                physical_shape=(physical_rows, physical_out_features),
                max_k_tiles=max_tiles,
            )

        output = self.alloc(
            name,
            rows,
            out_features,
            strict=False,
            physical_shape=(physical_rows, physical_out_features),
        )

        if num_k_tiles <= max_tiles:
            for col_idx in range(num_col_blocks):
                for row_idx in range(num_row_blocks):
                    self.vram_sub_projection_to(
                        input_var,
                        row_idx,
                        weight_var,
                        col_idx,
                        output,
                        row_idx,
                        col_idx,
                        matrix_precision="keyvalue",
                        set_scale=False,
                        hbm_element_bytes=2,
                    )
            return output

        for col_idx in range(num_col_blocks):
            for row_idx in range(num_row_blocks):
                self.vram_sub_projection_stream_k_accum_to(
                    input_var,
                    row_idx,
                    weight_var,
                    col_idx,
                    output,
                    row_idx,
                    col_idx,
                    max_k_tiles=max_tiles,
                    matrix_precision="keyvalue",
                    set_scale=False,
                    hbm_element_bytes=2,
                )
        return output

    def linear_projection_bf16(
        self,
        input_var: VRAMMatrixVar,
        weight_var: InputVar,
        name: str = "linear_out_bf16",
        physical_shape: tuple[int, int] | None = None,
    ):
        """Emit a high-precision BF16 matrix projection through HBM_M_KV_TYPE.

        The build must configure HBM_M_KV_TYPE as a Plain BF16 type for tensors
        that use this path.  No C_SET_SCALE_REG is emitted because plain BF16
        has no MX scale stream.
        """
        return self.linear_projection(
            input_var,
            weight_var,
            name=name,
            physical_shape=physical_shape,
            matrix_precision="keyvalue",
            set_scale=False,
            hbm_element_bytes=2,
        )

    def linear_projection_bias_bf16(
        self,
        input_var: VRAMMatrixVar,
        weight_var: InputVar,
        bias_var: VRAMMatrixVar | None = None,
        name: str = "linear_out_bf16",
        physical_shape: tuple[int, int] | None = None,
        bias_rows: int | None = None,
    ):
        """Emit a BF16 projection and optional BF16 VRAM bias add.

        This is the shared high-precision projection substrate for router and
        attention projections.  It deliberately keeps projection/bias separate
        from RoPE so callers can do one wide Q/K projection and then apply
        model-specific RoPE on per-head VRAM views without duplicating the
        projection path.
        """
        out = self.linear_projection_bf16(
            input_var,
            weight_var,
            name=name,
            physical_shape=physical_shape,
        )
        if bias_var is not None:
            self.vram_add(out, bias_var, num_rows=bias_rows)
        return out

    def runtime_rope_projection_bf16(
        self,
        x_var: VRAMMatrixVar,
        rotate_weight_var: InputVar,
        cos_var: VRAMMatrixVar,
        sin_var: VRAMMatrixVar,
        name: str = "runtime_rope_rot",
    ) -> VRAMMatrixVar:
        """Apply RoPE to a runtime projection output.

        Existing RoPE expects ``rotate_half(x)`` to already exist in VRAM.  This
        helper computes that rotate-half tensor through the BF16 HBM_M_KV
        projection path, applies RoPE in-place to ``x_var``, then releases the
        temporary.  The rotate matrix can be model-specific, which keeps the
        substrate generic for GPT-OSS/Qwen/DeepSeek adapters.
        """
        x_rot = self.linear_projection_bf16(
            x_var,
            rotate_weight_var,
            name=name,
            physical_shape=x_var.physical_shape,
        )
        self.rope(x_var, x_rot, cos_var, sin_var)
        self.free_tensor(x_rot)
        return x_var

    def head_runtime_rope_bf16(
        self,
        head_var: VRAMMatrixVar,
        rotate_weight_var: InputVar,
        cos_var: VRAMMatrixVar,
        sin_var: VRAMMatrixVar,
        *,
        norm_weight_var: VRAMMatrixVar | None = None,
        eps_offset: int | None = None,
        reci_hid_offset: int | None = None,
        num_rows: int | None = None,
        name: str = "head_runtime_rope",
    ) -> VRAMMatrixVar:
        """Apply optional per-head RMSNorm and runtime RoPE to a BF16 head.

        GPT-OSS uses projection+bias+RoPE, while Qwen-style adapters also need
        per-head Q/K RMSNorm before RoPE.  Keeping this sequence in one helper
        prevents each model harness from hand-rolling a slightly different
        projection-to-attention path.  The projection itself remains BF16 and
        emits no MX scale setup.
        """
        if norm_weight_var is not None:
            if eps_offset is None or reci_hid_offset is None:
                raise ValueError(
                    "head_runtime_rope_bf16 requires eps/reci offsets when norm_weight_var is set"
                )
            self.rms_norm(
                head_var, eps_offset=eps_offset, reci_hid_offset=reci_hid_offset
            )
            self.vram_mul(head_var, norm_weight_var, num_rows=num_rows)
        return self.runtime_rope_projection_bf16(
            head_var,
            rotate_weight_var,
            cos_var,
            sin_var,
            name=name,
        )

    def linear(
        self,
        input_var: VRAMMatrixVar,
        weight_var: InputVar,
        physical_shape: tuple[int, int] | None = None,
    ):
        """Default linear op compatibility surface."""
        return self.linear_projection(
            input_var, weight_var, physical_shape=physical_shape
        )

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

    def vram_copy_region(
        self,
        dst: VRAMMatrixVar,
        src: VRAMMatrixVar,
        *,
        num_rows: int | None = None,
        num_cols: int | None = None,
        dst_row_offset: int = 0,
        src_row_offset: int = 0,
        dst_col_offset: int = 0,
        src_col_offset: int = 0,
    ) -> VRAMMatrixVar:
        """Copy one tile-aligned VRAM region without reading or clearing HBM.

        ``V_ADD_VF(..., f0)`` is a real move because architectural ``f0`` is
        zero.  This avoids the unsafe ``dst *= 0; dst += src`` pattern, which
        does not clear NaNs and used to make residual/intermediate wiring rely
        on the simulator's initial VRAM contents.
        """
        dst = self._require_var(dst, VRAMMatrixVar, "dst")
        src = self._require_var(src, VRAMMatrixVar, "src")
        if num_rows is None:
            num_rows = src.shape[0] - src_row_offset
        if num_cols is None:
            num_cols = src.shape[1] - src_col_offset
        if num_rows <= 0 or num_cols <= 0:
            raise ValueError(
                f"copy dimensions must be positive, got rows={num_rows}, cols={num_cols}"
            )
        if any(
            offset < 0
            for offset in (
                dst_row_offset,
                src_row_offset,
                dst_col_offset,
                src_col_offset,
            )
        ):
            raise ValueError("VRAM copy offsets must be non-negative")
        if dst_row_offset + num_rows > dst.shape[0]:
            raise ValueError("VRAM copy exceeds destination rows")
        if src_row_offset + num_rows > src.shape[0]:
            raise ValueError("VRAM copy exceeds source rows")
        if dst_col_offset + num_cols > dst.shape[1]:
            raise ValueError("VRAM copy exceeds destination columns")
        if src_col_offset + num_cols > src.shape[1]:
            raise ValueError("VRAM copy exceeds source columns")
        if (
            num_cols % self.mlen
            or dst_col_offset % self.mlen
            or src_col_offset % self.mlen
        ):
            raise ValueError(
                "VRAM copy columns and column offsets must be MLEN-aligned: "
                f"cols={num_cols}, dst_offset={dst_col_offset}, src_offset={src_col_offset}, "
                f"MLEN={self.mlen}"
            )

        dst_base = self.get_vram_addr(dst.name)
        src_base = self.get_vram_addr(src.name)
        dst_physical_rows = dst.physical_shape[0]
        src_physical_rows = src.physical_shape[0]
        dst_col_block = dst_col_offset // self.mlen
        src_col_block = src_col_offset // self.mlen
        col_blocks = num_cols // self.mlen
        gp_dst, gp_src = self.register_allocator.allocate_gp(2)
        try:
            lines = [
                "; === VRAM copy region ===",
                f"; {src.name}[{src_row_offset}:{src_row_offset + num_rows}, "
                f"{src_col_offset}:{src_col_offset + num_cols}] -> "
                f"{dst.name}[{dst_row_offset}:{dst_row_offset + num_rows}, "
                f"{dst_col_offset}:{dst_col_offset + num_cols}]",
            ]
            for col_idx in range(col_blocks):
                for row_idx in range(num_rows):
                    dst_addr = (
                        dst_base
                        + (dst_col_block + col_idx) * dst_physical_rows * self.mlen
                        + (dst_row_offset + row_idx) * self.mlen
                    )
                    src_addr = (
                        src_base
                        + (src_col_block + col_idx) * src_physical_rows * self.mlen
                        + (src_row_offset + row_idx) * self.mlen
                    )
                    lines.extend(load_large_int(gp_dst, dst_addr))
                    lines.extend(load_large_int(gp_src, src_addr))
                    lines.append(f"V_ADD_VF gp{gp_dst}, gp{gp_src}, f0, 0")
            self.emit("\n".join(lines) + "\n")
        finally:
            self.register_allocator.free_gp([gp_dst, gp_src])
        return dst

    def vram_copy(
        self,
        src: VRAMMatrixVar,
        *,
        name: str,
        num_rows: int | None = None,
    ) -> VRAMMatrixVar:
        """Allocate a same-shaped tensor and copy valid rows from ``src``."""
        rows = src.shape[0] if num_rows is None else num_rows
        out = self.alloc(
            name,
            rows=src.shape[0],
            cols=src.shape[1],
            strict=False,
            physical_shape=src.physical_shape,
        )
        return self.vram_copy_region(
            out,
            src,
            num_rows=rows,
            num_cols=src.shape[1],
        )

    def embedding_add(self, input_var: VRAMMatrixVar, pos_weight_var: VRAMMatrixVar):
        """Add learned/positional embedding weights to input in-place."""
        self.vram_add(input_var, pos_weight_var)
        return input_var

    def vram_mul(
        self,
        dst: VRAMMatrixVar,
        src: VRAMMatrixVar,
        dst_row_offset: int = 0,
        src_row_offset: int = 0,
        num_rows: int | None = None,
    ):
        """VRAM matrix multiply: dst[row_offset:] *= src."""
        super().vram_matrix_mul(
            dst_matrix=dst.name,
            src_matrix=src.name,
            dst_row_offset=dst_row_offset,
            src_row_offset=src_row_offset,
            num_rows=num_rows,
        )
        return dst

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
