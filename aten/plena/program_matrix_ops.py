"""Matrix projection, RoPE, and VRAM operations for the PLENA program builder."""

from __future__ import annotations

import math

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
            raise ValueError(f"matrix_precision int must be 0 or 1, got {matrix_precision}")
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
        matrix_precision: str | int = "weights",
        set_scale: bool = True,
        hbm_element_bytes: int = 1,
    ):
        """
        target[target_row_idx][target_col_idx] = vram_matrix[vram_row_idx][:] @ mram_input[:][mram_col_idx]
        Supports K-split: k_block_start/k_block_count select a subset of K tiles.
        """
        vram_matrix, mram_input, target = self._prepare_projection(vram_matrix, mram_input, target, auto_reset_mram)
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
        vram_matrix, mram_input, target = self._prepare_projection(vram_matrix, mram_input, target, auto_reset_mram)
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
        valid_rows = vram_row_blocks[0].valid_shape[0] if vram_row_blocks[0].valid_shape else self.mlen
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

        valid_rows = vram_row_blocks[0].valid_shape[0] if vram_row_blocks[0].valid_shape else self.mlen
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
            physical_rows = max(input_var.physical_shape[0], math.ceil(rows / self.blen) * self.blen)
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

        # When rows is not a multiple of mlen the hardware still operates on
        # full tiles; only the first `rows` rows contain valid output.
        output = self.alloc(
            name,
            rows,
            out_features,
            strict=False,
            physical_shape=(physical_rows, physical_out_features),
        )

        def emit_projection(row_idx, col_idx, target, target_row_idx, target_col_idx, **k_split):
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
        for k_chunk_idx, (k_block_start, k_block_count) in enumerate(_iter_k_chunks(num_k_tiles, max_k_tiles)):
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
            physical_rows = max(input_var.physical_shape[0], math.ceil(rows / self.blen) * self.blen)
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
                raise ValueError("head_runtime_rope_bf16 requires eps/reci offsets when norm_weight_var is set")
            self.rms_norm(head_var, eps_offset=eps_offset, reci_hid_offset=reci_hid_offset)
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
        return self.linear_projection(input_var, weight_var, physical_shape=physical_shape)

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
