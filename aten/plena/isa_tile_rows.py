"""Tile-row ISA helpers for IsaCompiler."""

from __future__ import annotations

from compiler.aten.isa_builder import IsaBuilder, fp, gp
from compiler.aten.plena.affine_layout import AffineLayout, LayoutKind
from compiler.aten.plena.lstream import (
    StreamBinding,
    StreamConfigField,
    emit_stream_configuration,
)


class IsaTileRowMixin:
    # =========================================================================
    # Tile-row helpers (name-based)
    # =========================================================================

    def _tile_addr(
        self, matrix_name: str, tile_row_idx: int = 0, tile_col_idx: int = 0
    ) -> int:
        return self.get_vram_tile_addr(matrix_name, tile_row_idx, tile_col_idx)

    def _tile_row_single_matrix_op(
        self,
        asm_method: str,
        matrix_name: str,
        arg,
        tile_row_idx: int = 0,
        tile_col_idx: int = 0,
    ) -> str:
        return getattr(self, asm_method)(
            self._tile_addr(matrix_name, tile_row_idx, tile_col_idx), arg
        )

    def _tile_row_binary_matrix_op(
        self,
        asm_method: str,
        dst_matrix: str,
        src_matrix: str,
        rows: list[int],
        dst_tile_row_idx: int = 0,
        dst_tile_col_idx: int = 0,
        src_tile_row_idx: int = 0,
        src_tile_col_idx: int = 0,
    ) -> str:
        return getattr(self, asm_method)(
            self._tile_addr(dst_matrix, dst_tile_row_idx, dst_tile_col_idx),
            self._tile_addr(src_matrix, src_tile_row_idx, src_tile_col_idx),
            rows,
        )

    def tile_row_max(
        self,
        source_matrix: str,
        row_map: list[tuple[int, int]],
        tile_row_idx: int = 0,
        tile_col_idx: int = 0,
    ) -> str:
        return self._tile_row_single_matrix_op(
            "tile_row_max_asm", source_matrix, row_map, tile_row_idx, tile_col_idx
        )

    def tile_row_sum(
        self,
        source_matrix: str,
        row_map: list[tuple[int, int]],
        tile_row_idx: int = 0,
        tile_col_idx: int = 0,
    ) -> str:
        return self._tile_row_single_matrix_op(
            "tile_row_sum_asm", source_matrix, row_map, tile_row_idx, tile_col_idx
        )

    def tile_row_exp(
        self,
        matrix_name: str,
        rows: list[int],
        tile_row_idx: int = 0,
        tile_col_idx: int = 0,
    ) -> str:
        return self._tile_row_single_matrix_op(
            "tile_row_exp_asm", matrix_name, rows, tile_row_idx, tile_col_idx
        )

    def tile_row_reci(
        self,
        matrix_name: str,
        rows: list[int],
        tile_row_idx: int = 0,
        tile_col_idx: int = 0,
    ) -> str:
        return self._tile_row_single_matrix_op(
            "tile_row_reci_asm", matrix_name, rows, tile_row_idx, tile_col_idx
        )

    def tile_row_softplus(
        self,
        matrix_name: str,
        rows: list[int],
        tile_row_idx: int = 0,
        tile_col_idx: int = 0,
    ) -> str:
        return self._tile_row_single_matrix_op(
            "tile_row_softplus_asm", matrix_name, rows, tile_row_idx, tile_col_idx
        )

    def tile_row_to_fpram(
        self,
        matrix_name: str,
        row_map: list[tuple[int, int]],
        tile_row_idx: int = 0,
        tile_col_idx: int = 0,
    ) -> str:
        return self._tile_row_single_matrix_op(
            "tile_row_to_fpram_asm", matrix_name, row_map, tile_row_idx, tile_col_idx
        )

    def tile_row_sub_fp(
        self,
        matrix_name: str,
        row_map: list[tuple[int, int]],
        tile_row_idx: int = 0,
        tile_col_idx: int = 0,
    ) -> str:
        return self._tile_row_single_matrix_op(
            "tile_row_sub_fp_asm", matrix_name, row_map, tile_row_idx, tile_col_idx
        )

    def tile_row_mul_fp(
        self,
        matrix_name: str,
        row_map: list[tuple[int, int]],
        tile_row_idx: int = 0,
        tile_col_idx: int = 0,
    ) -> str:
        return self._tile_row_single_matrix_op(
            "tile_row_mul_fp_asm", matrix_name, row_map, tile_row_idx, tile_col_idx
        )

    def tile_row_max_fp(
        self,
        matrix_name: str,
        row_map: list[tuple[int, int]],
        tile_row_idx: int = 0,
        tile_col_idx: int = 0,
    ) -> str:
        return self._tile_row_single_matrix_op(
            "tile_row_max_fp_asm", matrix_name, row_map, tile_row_idx, tile_col_idx
        )

    def tile_row_min_fp(
        self,
        matrix_name: str,
        row_map: list[tuple[int, int]],
        tile_row_idx: int = 0,
        tile_col_idx: int = 0,
    ) -> str:
        return self._tile_row_single_matrix_op(
            "tile_row_min_fp_asm", matrix_name, row_map, tile_row_idx, tile_col_idx
        )

    def tile_row_add_fp(
        self,
        matrix_name: str,
        row_map: list[tuple[int, int]],
        tile_row_idx: int = 0,
        tile_col_idx: int = 0,
    ) -> str:
        return self._tile_row_single_matrix_op(
            "tile_row_add_fp_asm", matrix_name, row_map, tile_row_idx, tile_col_idx
        )

    def tile_row_add(
        self,
        dst_matrix: str,
        src_matrix: str,
        rows: list[int],
        dst_tile_row_idx: int = 0,
        dst_tile_col_idx: int = 0,
        src_tile_row_idx: int = 0,
        src_tile_col_idx: int = 0,
    ) -> str:
        return self._tile_row_binary_matrix_op(
            "tile_row_add_asm",
            dst_matrix,
            src_matrix,
            rows,
            dst_tile_row_idx,
            dst_tile_col_idx,
            src_tile_row_idx,
            src_tile_col_idx,
        )

    def tile_row_sub(
        self,
        dst_matrix: str,
        src_matrix: str,
        rows: list[int],
        dst_tile_row_idx: int = 0,
        dst_tile_col_idx: int = 0,
        src_tile_row_idx: int = 0,
        src_tile_col_idx: int = 0,
    ) -> str:
        return self._tile_row_binary_matrix_op(
            "tile_row_sub_asm",
            dst_matrix,
            src_matrix,
            rows,
            dst_tile_row_idx,
            dst_tile_col_idx,
            src_tile_row_idx,
            src_tile_col_idx,
        )

    def tile_row_mul(
        self,
        dst_matrix: str,
        src_matrix: str,
        rows: list[int],
        dst_tile_row_idx: int = 0,
        dst_tile_col_idx: int = 0,
        src_tile_row_idx: int = 0,
        src_tile_col_idx: int = 0,
    ) -> str:
        return self._tile_row_binary_matrix_op(
            "tile_row_mul_asm",
            dst_matrix,
            src_matrix,
            rows,
            dst_tile_row_idx,
            dst_tile_col_idx,
            src_tile_row_idx,
            src_tile_col_idx,
        )

    def tile_row_add_fp_broadcast(
        self,
        matrix_name: str,
        fpram_scalar_addr: int,
        rows: list[int],
        tile_row_idx: int = 0,
        tile_col_idx: int = 0,
    ) -> str:
        return self.tile_row_add_fp_broadcast_asm(
            self._tile_addr(matrix_name, tile_row_idx, tile_col_idx),
            fpram_scalar_addr,
            rows,
        )

    def tile_row_max_fp_broadcast(
        self,
        matrix_name: str,
        fpram_scalar_addr: int,
        rows: list[int],
        tile_row_idx: int = 0,
        tile_col_idx: int = 0,
    ) -> str:
        return self.tile_row_max_fp_broadcast_asm(
            self._tile_addr(matrix_name, tile_row_idx, tile_col_idx),
            fpram_scalar_addr,
            rows,
        )

    def tile_row_min_fp_broadcast(
        self,
        matrix_name: str,
        fpram_scalar_addr: int,
        rows: list[int],
        tile_row_idx: int = 0,
        tile_col_idx: int = 0,
    ) -> str:
        return self.tile_row_min_fp_broadcast_asm(
            self._tile_addr(matrix_name, tile_row_idx, tile_col_idx),
            fpram_scalar_addr,
            rows,
        )

    def tile_row_sub_fp_broadcast(
        self,
        matrix_name: str,
        fpram_scalar_addr: int,
        rows: list[int],
        reverse: bool = False,
        tile_row_idx: int = 0,
        tile_col_idx: int = 0,
    ) -> str:
        return self.tile_row_sub_fp_broadcast_asm(
            self._tile_addr(matrix_name, tile_row_idx, tile_col_idx),
            fpram_scalar_addr,
            rows,
            reverse,
        )

    def tile_row_mul_fp_broadcast(
        self,
        matrix_name: str,
        fpram_scalar_addr: int,
        rows: list[int],
        tile_row_idx: int = 0,
        tile_col_idx: int = 0,
    ) -> str:
        return self.tile_row_mul_fp_broadcast_asm(
            self._tile_addr(matrix_name, tile_row_idx, tile_col_idx),
            fpram_scalar_addr,
            rows,
        )

    def tile_row_fma_fp_sweep(
        self,
        dst_matrix: str,
        src_matrix: str,
        fpram_base: int,
        dst_rows: list[int],
        src_rows: list[int],
        dst_tile_row_idx: int = 0,
        dst_tile_col_idx: int = 0,
        src_tile_row_idx: int = 0,
        src_tile_col_idx: int = 0,
    ) -> str:
        return self.tile_row_fma_fp_sweep_asm(
            self._tile_addr(dst_matrix, dst_tile_row_idx, dst_tile_col_idx),
            self._tile_addr(src_matrix, src_tile_row_idx, src_tile_col_idx),
            fpram_base,
            dst_rows,
            src_rows,
        )

    def tile_row_fma_fp_broadcast(
        self,
        dst_matrix: str,
        src_matrix: str,
        fpram_scalar_addr: int,
        dst_rows: list[int],
        src_rows: list[int],
        dst_tile_row_idx: int = 0,
        dst_tile_col_idx: int = 0,
        src_tile_row_idx: int = 0,
        src_tile_col_idx: int = 0,
    ) -> str:
        return self.tile_row_fma_fp_broadcast_asm(
            self._tile_addr(dst_matrix, dst_tile_row_idx, dst_tile_col_idx),
            self._tile_addr(src_matrix, src_tile_row_idx, src_tile_col_idx),
            fpram_scalar_addr,
            dst_rows,
            src_rows,
        )

    def tile_multirow_mul_fp(
        self,
        matrix_name: str,
        fpram_base: int,
        rows: list[int],
        *,
        fp_step: int,
        tile_row_idx: int = 0,
        tile_col_idx: int = 0,
    ) -> str:
        return self.tile_multirow_mul_fp_asm(
            self._tile_addr(matrix_name, tile_row_idx, tile_col_idx),
            fpram_base,
            rows,
            fp_step=fp_step,
        )

    def tile_multirow_fma_fp_sweep(
        self,
        dst_matrix: str,
        src_matrix: str,
        fpram_base: int,
        dst_rows: list[int],
        src_rows: list[int],
        dst_tile_row_idx: int = 0,
        dst_tile_col_idx: int = 0,
        src_tile_row_idx: int = 0,
        src_tile_col_idx: int = 0,
    ) -> str:
        return self.tile_multirow_fma_fp_sweep_asm(
            self._tile_addr(dst_matrix, dst_tile_row_idx, dst_tile_col_idx),
            self._tile_addr(src_matrix, src_tile_row_idx, src_tile_col_idx),
            fpram_base,
            dst_rows,
            src_rows,
        )

    def vram_fill_zero(
        self,
        matrix_name: str,
        rows: list[int],
        tile_row_idx: int = 0,
        tile_col_idx: int = 0,
    ) -> str:
        return self._tile_row_single_matrix_op(
            "vram_fill_zero_asm", matrix_name, rows, tile_row_idx, tile_col_idx
        )

    # =========================================================================
    # Tile-row ISA helpers (address-based)
    # =========================================================================

    def _arith_progression(self, values: list[int]) -> tuple[int, int, int] | None:
        """Return (start, count, step) if values form an arithmetic progression."""
        if not values:
            return None
        if len(values) == 1:
            return (values[0], 1, 0)
        step = values[1] - values[0]
        for i in range(2, len(values)):
            if values[i] - values[i - 1] != step:
                return None
        # A step of 0 is a valid progression. The old code refused it, saying it
        # "would cause infinite HW loop" -- that is not what happens: C_LOOP_START
        # takes its trip count as an immediate and C_LOOP_END decrements a
        # dedicated loop register, neither of which the address step touches. The
        # single-element case above has always returned step 0 and looped fine.
        #
        # It matters because a pinned side is exactly a step of 0, and every
        # recurrent contraction has one: the prediction and the read-out walk the
        # state rows against one accumulator row. Refusing it forced those onto
        # the unrolled path.
        return (values[0], len(values), step)

    def _row_progression(self, rows: list[int]) -> tuple[int, int, int] | None:
        return None if self._unroll else self._arith_progression(rows)

    def _append_stream_binding(
        self,
        asm: IsaBuilder,
        *,
        value_gp: int,
        slot: int,
        target_register: int,
        target_is_fp: bool,
        base: int,
        advance: int,
        count: int,
        packet_elements: int,
        storage_atom: int,
        packet_stride: int | None = None,
        packetized: bool = False,
        extent_minor: int | None = None,
        extent_major: int | None = None,
        auto_advance: bool = True,
        write: bool = False,
        affine: bool = False,
    ) -> None:
        if count <= 0:
            raise ValueError(f"stream trip count must be positive, got {count}")
        if advance < 0:
            raise ValueError("L-stream v1 does not encode negative address advances")
        if not packetized and advance and advance % packet_elements:
            raise ValueError(
                f"stream advance {advance} is not a multiple of packet width "
                f"{packet_elements}"
            )
        if packetized:
            if packet_elements % storage_atom:
                raise ValueError("packetized stream must contain whole storage atoms")
            segments = packet_elements // storage_atom
            if count % segments:
                raise ValueError(
                    f"packetized trip count {count} must be divisible by {segments} segments"
                )
            minors = packet_elements if extent_minor is None else extent_minor
            majors = count if extent_major is None else extent_major
        else:
            step_units = advance // packet_elements if advance else 0
            span = (count - 1) * step_units + 1 if step_units else 1
            majors = span if extent_major is None else extent_major
            minors = packet_elements if extent_minor is None else extent_minor
        use_affine = affine and not target_is_fp
        if packetized and use_affine and base % packet_elements:
            raise ValueError(
                "compact affine packet base must be aligned to packet_elements"
            )
        layout = AffineLayout(
            kind=(
                LayoutKind.AFFINE_SKEW
                if use_affine
                and (
                    self.stream_affine_alpha
                    or self.stream_affine_beta
                    or self.stream_affine_gamma
                )
                else LayoutKind.ROW_MAJOR
            ),
            groups=1,
            fields=1,
            majors=majors,
            minors=minors,
            alpha=self.stream_affine_alpha if use_affine else 0,
            beta=self.stream_affine_beta if use_affine else 0,
            gamma=self.stream_affine_gamma if use_affine else 0,
            bank_row_base=(
                0
                if target_is_fp
                else base
                // (packet_elements if packetized and use_affine else self.mlen)
            ),
        )
        binding = StreamBinding(
            slot=slot,
            target_register=target_register,
            target_is_fp=target_is_fp,
            base=base,
            advance=advance,
            packet_elements=packet_elements,
            storage_atom=storage_atom,
            packet_stride=packet_stride,
            auto_advance=auto_advance,
            write=write,
            packetized=packetized,
        )
        asm.extend(
            emit_stream_configuration(
                value_gp=value_gp, binding=binding, layout=layout
            ).items
        )

    def _packet_rows_supported(
        self, rows: list[int]
    ) -> tuple[int, int, int, int] | None:
        if not (self.stream_addressing and self.stream_packetized):
            return None
        if self.mlen % self.stream_storage_atom:
            return None
        progression = self._arith_progression(rows)
        if progression is None:
            return None
        start, count, step = progression
        packet_elements = self.stream_packet_elements
        segments = packet_elements // self.stream_storage_atom
        if step != 1 or count < segments or count % segments:
            return None
        minor_steps = self.mlen // self.stream_storage_atom
        packet_steps = minor_steps * (count // segments)
        return start, count, packet_elements, packet_steps

    def tile_multirow_mul_fp_asm(
        self,
        vram_addr: int,
        fpram_base: int,
        rows: list[int],
        *,
        fp_step: int,
    ) -> str:
        """Multiply rows by a segmented scalar packet when the walk is regular.

        One packetized ``V_MUL_VF`` computes ``stream_packet_elements`` values.
        Those lanes are assembled from short logical rows, and one FPRAM scalar
        is broadcast over each storage atom. The default packet width remains
        ``MLEN``; a paper-width configuration may coalesce several MLEN rows.
        """

        if fp_step not in (0, 1):
            raise ValueError(f"packet scalar step must be 0 or 1, got {fp_step}")
        supported = self._packet_rows_supported(rows)
        if supported is None:
            row_map = [
                (row, fpram_base + index * fp_step) for index, row in enumerate(rows)
            ]
            return self._emit_tile_row_fp_scalar(
                "Mul", "V_MUL_VF", vram_addr, row_map, (0,)
            )

        row_start, count, packet_elements, packet_steps = supported
        gp_data, gp_loop, gp_value = self._reg.allocate_gp(3)
        try:
            asm = IsaBuilder().comment(
                f"Packetized multi-row Mul: VRAM[{vram_addr}], rows={count}, "
                f"packet_elements={packet_elements}"
            )
            self._append_stream_binding(
                asm,
                value_gp=gp_value,
                slot=0,
                target_register=gp_data,
                target_is_fp=False,
                base=vram_addr + row_start * self.mlen,
                advance=self.stream_storage_atom,
                count=count,
                packet_elements=packet_elements,
                storage_atom=self.stream_storage_atom,
                packet_stride=self.mlen,
                packetized=True,
                extent_minor=self.mlen,
                extent_major=count,
                write=True,
                affine=True,
            )
            self._append_stream_binding(
                asm,
                value_gp=gp_value,
                slot=1,
                target_register=1,
                target_is_fp=True,
                base=fpram_base,
                advance=0,
                count=count,
                packet_elements=packet_elements,
                storage_atom=self.stream_storage_atom,
                packet_stride=fp_step,
                packetized=True,
                extent_minor=self.mlen,
                extent_major=count,
            )
            asm.instr("C_LOOP_START", gp(gp_loop), packet_steps)
            asm.instr("V_MUL_VF", gp(gp_data), gp(gp_data), fp(1), 0)
            asm.instr("C_LOOP_END", gp(gp_loop))
            self._append_stream_reset(
                asm, slot=0, target_register=gp_data, target_is_fp=False
            )
            self._append_stream_reset(asm, slot=1, target_register=1, target_is_fp=True)
            return self._emit(asm)
        finally:
            self._reg.free_gp([gp_data, gp_loop, gp_value])

    def tile_multirow_fma_fp_sweep_asm(
        self,
        dst_addr: int,
        src_addr: int,
        fpram_base: int,
        dst_rows: list[int],
        src_rows: list[int],
    ) -> str:
        """Packetized ``dst += src * scalar`` for moving destinations.

        A pinned destination is deliberately rejected: it is a cross-row
        reduction, not independent lane arithmetic, and remains on the ordinary
        row path.
        """

        if len(dst_rows) != len(src_rows):
            raise ValueError("packet FMA source and destination row counts differ")
        supported = self._packet_rows_supported(dst_rows)
        src_progression = self._arith_progression(src_rows)
        if (
            supported is None
            or src_progression is None
            or src_progression[2] not in (0, 1)
        ):
            return self.tile_row_fma_fp_sweep_asm(
                dst_addr, src_addr, fpram_base, dst_rows, src_rows
            )

        dst_start, count, packet_elements, packet_steps = supported
        src_start, src_count, src_step = src_progression
        if src_count != count:
            raise ValueError("packet FMA source and destination row counts differ")
        gp_dst, gp_src, gp_loop, gp_value = self._reg.allocate_gp(4)
        try:
            asm = IsaBuilder().comment(
                f"Packetized multi-row FMA: VRAM[{dst_addr}] += VRAM[{src_addr}] * FPRAM, "
                f"packet_elements={packet_elements}"
            )
            for slot, target, base, stride, write, affine in (
                (0, gp_dst, dst_addr + dst_start * self.mlen, self.mlen, True, True),
                (
                    1,
                    gp_src,
                    src_addr + src_start * self.mlen,
                    src_step * self.mlen,
                    False,
                    bool(src_step),
                ),
            ):
                self._append_stream_binding(
                    asm,
                    value_gp=gp_value,
                    slot=slot,
                    target_register=target,
                    target_is_fp=False,
                    base=base,
                    advance=self.stream_storage_atom,
                    count=count,
                    packet_elements=packet_elements,
                    storage_atom=self.stream_storage_atom,
                    packet_stride=stride,
                    packetized=True,
                    extent_minor=self.mlen,
                    extent_major=count,
                    write=write,
                    affine=affine,
                )
            self._append_stream_binding(
                asm,
                value_gp=gp_value,
                slot=2,
                target_register=1,
                target_is_fp=True,
                base=fpram_base,
                advance=0,
                count=count,
                packet_elements=packet_elements,
                storage_atom=self.stream_storage_atom,
                packet_stride=1,
                packetized=True,
                extent_minor=self.mlen,
                extent_major=count,
            )
            asm.instr("C_LOOP_START", gp(gp_loop), packet_steps)
            asm.instr("V_FMA_VF", gp(gp_dst), gp(gp_src), fp(1), 0)
            asm.instr("C_LOOP_END", gp(gp_loop))
            self._append_stream_reset(
                asm, slot=0, target_register=gp_dst, target_is_fp=False
            )
            self._append_stream_reset(
                asm, slot=1, target_register=gp_src, target_is_fp=False
            )
            self._append_stream_reset(asm, slot=2, target_register=1, target_is_fp=True)
            return self._emit(asm)
        finally:
            self._reg.free_gp([gp_dst, gp_src, gp_loop, gp_value])

    @staticmethod
    def _stream_is_profitable(
        *,
        trips: int,
        slots: int,
        saved_per_trip: int,
        baseline_setup_instructions: int,
    ) -> bool:
        """Whether explicit stream setup beats the ordinary loop issue count.

        A linear slot needs at most fifteen setup instructions after default
        elision (large immediates may add one more). The conservative threshold
        keeps short and irregular loops on the ordinary Arlo path.
        """

        stream_setup = slots * 15
        return trips * saved_per_trip + baseline_setup_instructions > stream_setup

    @staticmethod
    def _append_stream_reset(
        asm: IsaBuilder, *, slot: int, target_register: int, target_is_fp: bool
    ) -> None:
        target = fp(target_register) if target_is_fp else gp(target_register)
        asm.instr("L_STREAM_CFG", gp(0), target, slot, int(StreamConfigField.RESET))

    def _emit_tile_row_reduce(
        self,
        label: str,
        source_vram_addr: int,
        row_map: list[tuple[int, int]],
        opcode: str,
        opcode_extra_args: tuple[int, ...] = (),
        clear_accumulator: bool = False,
    ) -> str:
        gp_regs = self._reg.allocate_gp(3)
        gp_src, gp_dst, gp_loop = gp_regs
        try:
            asm = IsaBuilder().comment(
                f"Tile Row {label} from VRAM[{source_vram_addr}]"
            )
            rows = [row for row, _ in row_map]
            fp_addrs = [addr_ for _, addr_ in row_map]
            row_prog = self._row_progression(rows)
            fp_prog = self._row_progression(fp_addrs)

            if row_prog is not None and fp_prog is not None:
                row_start, row_count, row_step = row_prog
                fp_start, _, fp_step = fp_prog
                use_stream = (
                    self.stream_addressing
                    and row_step >= 0
                    and self._stream_is_profitable(
                        trips=row_count,
                        slots=1,
                        saved_per_trip=int(bool(row_step)),
                        baseline_setup_instructions=1,
                    )
                )
                if use_stream:
                    gp_value = self._reg.allocate_gp(1)[0]
                    try:
                        self._append_stream_binding(
                            asm,
                            value_gp=gp_value,
                            slot=0,
                            target_register=gp_src,
                            target_is_fp=False,
                            base=source_vram_addr + row_start * self.mlen,
                            advance=row_step * self.mlen,
                            count=row_count,
                            packet_elements=self.mlen,
                            storage_atom=self.stream_storage_atom,
                        )
                        asm.instr("S_ADDI_INT", gp(gp_dst), gp(0), fp_start)
                        asm.instr("C_LOOP_START", gp(gp_loop), row_count)
                        if clear_accumulator:
                            asm.instr("S_ADD_FP", fp(1), fp(0), fp(0))
                        asm.instr(opcode, fp(1), gp(gp_src), 0, *opcode_extra_args)
                        asm.instr("S_ST_FP", fp(1), gp(gp_dst), 0)
                        if fp_step:
                            asm.instr("S_ADDI_INT", gp(gp_dst), gp(gp_dst), fp_step)
                        asm.instr("C_LOOP_END", gp(gp_loop))
                        self._append_stream_reset(
                            asm,
                            slot=0,
                            target_register=gp_src,
                            target_is_fp=False,
                        )
                    finally:
                        self._reg.free_gp([gp_value])
                else:
                    asm.instr(
                        "S_ADDI_INT",
                        gp(gp_src),
                        gp(0),
                        source_vram_addr + row_start * self.mlen,
                    )
                    asm.instr("S_ADDI_INT", gp(gp_dst), gp(0), fp_start)
                    asm.instr("C_LOOP_START", gp(gp_loop), row_count)
                    if clear_accumulator:
                        asm.instr("S_ADD_FP", fp(1), fp(0), fp(0))
                    asm.instr(opcode, fp(1), gp(gp_src), 0, *opcode_extra_args)
                    asm.instr("S_ST_FP", fp(1), gp(gp_dst), 0)
                    if row_step:
                        asm.instr(
                            "S_ADDI_INT",
                            gp(gp_src),
                            gp(gp_src),
                            row_step * self.mlen,
                        )
                    if fp_step:
                        asm.instr("S_ADDI_INT", gp(gp_dst), gp(gp_dst), fp_step)
                    asm.instr("C_LOOP_END", gp(gp_loop))
            else:
                for row_idx, fpram_addr in row_map:
                    row_addr = source_vram_addr + row_idx * self.mlen
                    asm.instr("S_ADDI_INT", gp(gp_src), gp(0), row_addr)
                    if clear_accumulator:
                        asm.instr("S_ADD_FP", fp(1), fp(0), fp(0))
                    asm.instr(opcode, fp(1), gp(gp_src), 0, *opcode_extra_args)
                    asm.instr("S_ADDI_INT", gp(gp_dst), gp(0), fpram_addr)
                    asm.instr("S_ST_FP", fp(1), gp(gp_dst), 0)

            return self._emit(asm)
        finally:
            self._reg.free_gp(gp_regs)

    def _emit_tile_row_unary(
        self, label: str, opcode: str, vram_addr: int, rows: list[int]
    ) -> str:
        gp_regs = self._reg.allocate_gp(2)
        gp_src, gp_loop = gp_regs
        try:
            asm = IsaBuilder().comment(f"Tile Row {label} on VRAM[{vram_addr}]")
            prog = self._row_progression(rows)

            if prog is not None:
                row_start, row_count, row_step = prog
                use_stream = (
                    self.stream_addressing
                    and row_step >= 0
                    and self._stream_is_profitable(
                        trips=row_count,
                        slots=1,
                        saved_per_trip=int(bool(row_step)),
                        baseline_setup_instructions=1,
                    )
                )
                if use_stream:
                    gp_value = self._reg.allocate_gp(1)[0]
                    try:
                        self._append_stream_binding(
                            asm,
                            value_gp=gp_value,
                            slot=0,
                            target_register=gp_src,
                            target_is_fp=False,
                            base=vram_addr + row_start * self.mlen,
                            advance=row_step * self.mlen,
                            count=row_count,
                            packet_elements=self.mlen,
                            storage_atom=self.stream_storage_atom,
                            write=True,
                        )
                        asm.instr("C_LOOP_START", gp(gp_loop), row_count)
                        asm.instr(opcode, gp(gp_src), gp(gp_src), 0)
                        asm.instr("C_LOOP_END", gp(gp_loop))
                        self._append_stream_reset(
                            asm,
                            slot=0,
                            target_register=gp_src,
                            target_is_fp=False,
                        )
                    finally:
                        self._reg.free_gp([gp_value])
                else:
                    asm.instr(
                        "S_ADDI_INT",
                        gp(gp_src),
                        gp(0),
                        vram_addr + row_start * self.mlen,
                    )
                    asm.instr("C_LOOP_START", gp(gp_loop), row_count)
                    asm.instr(opcode, gp(gp_src), gp(gp_src), 0)
                    if row_step:
                        asm.instr(
                            "S_ADDI_INT",
                            gp(gp_src),
                            gp(gp_src),
                            row_step * self.mlen,
                        )
                    asm.instr("C_LOOP_END", gp(gp_loop))
            else:
                for row_idx in rows:
                    row_addr = vram_addr + row_idx * self.mlen
                    asm.instr("S_ADDI_INT", gp(gp_src), gp(0), row_addr)
                    asm.instr(opcode, gp(gp_src), gp(gp_src), 0)

            return self._emit(asm)
        finally:
            self._reg.free_gp(gp_regs)

    def _emit_tile_row_fp_scalar(
        self,
        label: str,
        opcode: str,
        vram_addr: int,
        row_map: list[tuple[int, int]],
        opcode_extra_args: tuple[int, ...] = (),
    ) -> str:
        gp_regs = self._reg.allocate_gp(3)
        gp_src, gp_fp, gp_loop = gp_regs
        try:
            asm = IsaBuilder().comment(f"Tile Row {label} FP on VRAM[{vram_addr}]")
            rows = [row for row, _ in row_map]
            fp_addrs = [addr_ for _, addr_ in row_map]
            row_prog = self._row_progression(rows)
            fp_prog = self._row_progression(fp_addrs)

            if row_prog is not None and fp_prog is not None:
                row_start, row_count, row_step = row_prog
                fp_start, _, fp_step = fp_prog
                use_stream = (
                    self.stream_addressing
                    and min(row_step, fp_step) >= 0
                    and self._stream_is_profitable(
                        trips=row_count,
                        slots=2,
                        saved_per_trip=1 + int(bool(row_step)) + int(bool(fp_step)),
                        baseline_setup_instructions=2,
                    )
                )
                if use_stream:
                    gp_value = self._reg.allocate_gp(1)[0]
                    try:
                        self._append_stream_binding(
                            asm,
                            value_gp=gp_value,
                            slot=0,
                            target_register=gp_src,
                            target_is_fp=False,
                            base=vram_addr + row_start * self.mlen,
                            advance=row_step * self.mlen,
                            count=row_count,
                            packet_elements=self.mlen,
                            storage_atom=self.stream_storage_atom,
                        )
                        self._append_stream_binding(
                            asm,
                            value_gp=gp_value,
                            slot=1,
                            target_register=1,
                            target_is_fp=True,
                            base=fp_start,
                            advance=fp_step,
                            count=row_count,
                            packet_elements=1,
                            storage_atom=1,
                        )
                        asm.instr("C_LOOP_START", gp(gp_loop), row_count)
                        asm.instr(
                            opcode, gp(gp_src), gp(gp_src), fp(1), *opcode_extra_args
                        )
                        asm.instr("C_LOOP_END", gp(gp_loop))
                        self._append_stream_reset(
                            asm, slot=0, target_register=gp_src, target_is_fp=False
                        )
                        self._append_stream_reset(
                            asm, slot=1, target_register=1, target_is_fp=True
                        )
                    finally:
                        self._reg.free_gp([gp_value])
                else:
                    asm.instr(
                        "S_ADDI_INT",
                        gp(gp_src),
                        gp(0),
                        vram_addr + row_start * self.mlen,
                    )
                    asm.instr("S_ADDI_INT", gp(gp_fp), gp(0), fp_start)
                    asm.instr("C_LOOP_START", gp(gp_loop), row_count)
                    asm.instr("S_LD_FP", fp(1), gp(gp_fp), 0)
                    asm.instr(opcode, gp(gp_src), gp(gp_src), fp(1), *opcode_extra_args)
                    if row_step:
                        asm.instr(
                            "S_ADDI_INT", gp(gp_src), gp(gp_src), row_step * self.mlen
                        )
                    if fp_step:
                        asm.instr("S_ADDI_INT", gp(gp_fp), gp(gp_fp), fp_step)
                    asm.instr("C_LOOP_END", gp(gp_loop))
            else:
                for row_idx, fpram_addr in row_map:
                    row_addr = vram_addr + row_idx * self.mlen
                    asm.instr("S_ADDI_INT", gp(gp_src), gp(0), row_addr)
                    asm.instr("S_ADDI_INT", gp(gp_fp), gp(0), fpram_addr)
                    asm.instr("S_LD_FP", fp(1), gp(gp_fp), 0)
                    asm.instr(opcode, gp(gp_src), gp(gp_src), fp(1), *opcode_extra_args)

            return self._emit(asm)
        finally:
            self._reg.free_gp(gp_regs)

    def _emit_tile_row_fma(
        self,
        dst_addr: int,
        src_addr: int,
        row_map: list[tuple[int, int, int]],
    ) -> str:
        """``dst[d] += src[s] * FPRAM[f]`` for each ``(d, s, f)`` in ``row_map``.

        Four pointers, so four GP registers. When all three walks are arithmetic
        progressions this collapses to one hardware loop -- the case that
        matters, since every recurrent sweep either steps a state tile against a
        pinned accumulator or the reverse, and a pinned side is a step of 0.

        The predecessor -- ``mamba_row_copy`` then ``tile_row_mul_fp`` then
        ``mamba_row_add`` -- could not loop at all: the scratch row it needed was
        the *same* row every iteration, so the copy's destination was constant
        and broke the progression.
        """
        gp_regs = self._reg.allocate_gp(4)
        gp_dst, gp_src, gp_fp, gp_loop = gp_regs
        try:
            asm = IsaBuilder().comment(
                f"Tile Row FMA: VRAM[{dst_addr}] += VRAM[{src_addr}] * FPRAM"
            )
            dst_rows = [d for d, _, _ in row_map]
            src_rows = [s for _, s, _ in row_map]
            fp_addrs = [f for _, _, f in row_map]

            dst_prog = self._row_progression(dst_rows)
            src_prog = self._row_progression(src_rows)
            fp_prog = self._row_progression(fp_addrs)

            if dst_prog is not None and src_prog is not None and fp_prog is not None:
                dst_start, count, dst_step = dst_prog
                src_start, _, src_step = src_prog
                fp_start, _, fp_step = fp_prog
                use_stream = (
                    self.stream_addressing
                    and min(dst_step, src_step, fp_step) >= 0
                    and self._stream_is_profitable(
                        trips=count,
                        slots=3,
                        saved_per_trip=(
                            1
                            + int(bool(dst_step))
                            + int(bool(src_step))
                            + int(bool(fp_step))
                        ),
                        baseline_setup_instructions=3,
                    )
                )
                if use_stream:
                    gp_value = self._reg.allocate_gp(1)[0]
                    try:
                        for slot, target, base, step, write in (
                            (
                                0,
                                gp_dst,
                                dst_addr + dst_start * self.mlen,
                                dst_step,
                                True,
                            ),
                            (
                                1,
                                gp_src,
                                src_addr + src_start * self.mlen,
                                src_step,
                                False,
                            ),
                        ):
                            self._append_stream_binding(
                                asm,
                                value_gp=gp_value,
                                slot=slot,
                                target_register=target,
                                target_is_fp=False,
                                base=base,
                                advance=step * self.mlen,
                                count=count,
                                packet_elements=self.mlen,
                                storage_atom=self.stream_storage_atom,
                                write=write,
                            )
                        self._append_stream_binding(
                            asm,
                            value_gp=gp_value,
                            slot=2,
                            target_register=1,
                            target_is_fp=True,
                            base=fp_start,
                            advance=fp_step,
                            count=count,
                            packet_elements=1,
                            storage_atom=1,
                        )
                        asm.instr("C_LOOP_START", gp(gp_loop), count)
                        asm.instr("V_FMA_VF", gp(gp_dst), gp(gp_src), fp(1), 0)
                        asm.instr("C_LOOP_END", gp(gp_loop))
                        self._append_stream_reset(
                            asm, slot=0, target_register=gp_dst, target_is_fp=False
                        )
                        self._append_stream_reset(
                            asm, slot=1, target_register=gp_src, target_is_fp=False
                        )
                        self._append_stream_reset(
                            asm, slot=2, target_register=1, target_is_fp=True
                        )
                    finally:
                        self._reg.free_gp([gp_value])
                else:
                    asm.instr(
                        "S_ADDI_INT",
                        gp(gp_dst),
                        gp(0),
                        dst_addr + dst_start * self.mlen,
                    )
                    asm.instr(
                        "S_ADDI_INT",
                        gp(gp_src),
                        gp(0),
                        src_addr + src_start * self.mlen,
                    )
                    asm.instr("S_ADDI_INT", gp(gp_fp), gp(0), fp_start)
                    asm.instr("C_LOOP_START", gp(gp_loop), count)
                    asm.instr("S_LD_FP", fp(1), gp(gp_fp), 0)
                    asm.instr("V_FMA_VF", gp(gp_dst), gp(gp_src), fp(1), 0)
                    # A zero step is a no-op add; skipping it keeps the pinned side
                    # from costing an instruction per iteration.
                    if dst_step:
                        asm.instr(
                            "S_ADDI_INT", gp(gp_dst), gp(gp_dst), dst_step * self.mlen
                        )
                    if src_step:
                        asm.instr(
                            "S_ADDI_INT", gp(gp_src), gp(gp_src), src_step * self.mlen
                        )
                    if fp_step:
                        asm.instr("S_ADDI_INT", gp(gp_fp), gp(gp_fp), fp_step)
                    asm.instr("C_LOOP_END", gp(gp_loop))
            else:
                for dst_row, src_row, fpram_addr in row_map:
                    asm.instr(
                        "S_ADDI_INT", gp(gp_dst), gp(0), dst_addr + dst_row * self.mlen
                    )
                    asm.instr(
                        "S_ADDI_INT", gp(gp_src), gp(0), src_addr + src_row * self.mlen
                    )
                    asm.instr("S_ADDI_INT", gp(gp_fp), gp(0), fpram_addr)
                    asm.instr("S_LD_FP", fp(1), gp(gp_fp), 0)
                    asm.instr("V_FMA_VF", gp(gp_dst), gp(gp_src), fp(1), 0)

            return self._emit(asm)
        finally:
            self._reg.free_gp(gp_regs)

    def _emit_tile_row_vector_op(
        self,
        label: str,
        opcode: str,
        dst_addr: int,
        src_addr: int,
        rows: list[int],
    ) -> str:
        gp_regs = self._reg.allocate_gp(3)
        gp_dst, gp_src, gp_loop = gp_regs
        try:
            assignment_op = {"Add": "+", "Sub": "-", "Mul": "*"}.get(label, label)
            asm = IsaBuilder().comment(
                f"Tile Row {label}: VRAM[{dst_addr}] {assignment_op}= VRAM[{src_addr}]"
            )
            prog = self._row_progression(rows)

            if prog is not None:
                row_start, row_count, row_step = prog
                use_stream = (
                    self.stream_addressing
                    and row_step >= 0
                    and self._stream_is_profitable(
                        trips=row_count,
                        slots=2,
                        saved_per_trip=2 * int(bool(row_step)),
                        baseline_setup_instructions=2,
                    )
                )
                if use_stream:
                    gp_value = self._reg.allocate_gp(1)[0]
                    try:
                        for slot, target, base, write in (
                            (0, gp_dst, dst_addr + row_start * self.mlen, True),
                            (1, gp_src, src_addr + row_start * self.mlen, False),
                        ):
                            self._append_stream_binding(
                                asm,
                                value_gp=gp_value,
                                slot=slot,
                                target_register=target,
                                target_is_fp=False,
                                base=base,
                                advance=row_step * self.mlen,
                                count=row_count,
                                packet_elements=self.mlen,
                                storage_atom=self.stream_storage_atom,
                                write=write,
                            )
                        asm.instr("C_LOOP_START", gp(gp_loop), row_count)
                        asm.instr(opcode, gp(gp_dst), gp(gp_dst), gp(gp_src), 0)
                        asm.instr("C_LOOP_END", gp(gp_loop))
                        self._append_stream_reset(
                            asm,
                            slot=0,
                            target_register=gp_dst,
                            target_is_fp=False,
                        )
                        self._append_stream_reset(
                            asm,
                            slot=1,
                            target_register=gp_src,
                            target_is_fp=False,
                        )
                    finally:
                        self._reg.free_gp([gp_value])
                else:
                    asm.instr(
                        "S_ADDI_INT",
                        gp(gp_dst),
                        gp(0),
                        dst_addr + row_start * self.mlen,
                    )
                    asm.instr(
                        "S_ADDI_INT",
                        gp(gp_src),
                        gp(0),
                        src_addr + row_start * self.mlen,
                    )
                    asm.instr("C_LOOP_START", gp(gp_loop), row_count)
                    asm.instr(opcode, gp(gp_dst), gp(gp_dst), gp(gp_src), 0)
                    if row_step:
                        asm.instr(
                            "S_ADDI_INT",
                            gp(gp_dst),
                            gp(gp_dst),
                            row_step * self.mlen,
                        )
                        asm.instr(
                            "S_ADDI_INT",
                            gp(gp_src),
                            gp(gp_src),
                            row_step * self.mlen,
                        )
                    asm.instr("C_LOOP_END", gp(gp_loop))
            else:
                for row_idx in rows:
                    dst_row_addr = dst_addr + row_idx * self.mlen
                    src_row_addr = src_addr + row_idx * self.mlen
                    asm.instr("S_ADDI_INT", gp(gp_dst), gp(0), dst_row_addr)
                    asm.instr("S_ADDI_INT", gp(gp_src), gp(0), src_row_addr)
                    asm.instr(opcode, gp(gp_dst), gp(gp_dst), gp(gp_src), 0)

            return self._emit(asm)
        finally:
            self._reg.free_gp(gp_regs)

    def tile_row_max_asm(
        self, source_vram_addr: int, row_map: list[tuple[int, int]]
    ) -> str:
        return self._emit_tile_row_reduce("Max", source_vram_addr, row_map, "V_RED_MAX")

    def tile_row_sum_asm(
        self, source_vram_addr: int, row_map: list[tuple[int, int]]
    ) -> str:
        return self._emit_tile_row_reduce(
            "Sum",
            source_vram_addr,
            row_map,
            "V_RED_SUM",
            opcode_extra_args=(0,),
            clear_accumulator=True,
        )

    def tile_row_exp_asm(self, vram_addr: int, rows: list[int]) -> str:
        return self._emit_tile_row_unary("Exp", "V_EXP_V", vram_addr, rows)

    def tile_row_reci_asm(self, vram_addr: int, rows: list[int]) -> str:
        return self._emit_tile_row_unary("Reciprocal", "V_RECI_V", vram_addr, rows)

    def tile_row_softplus_asm(self, vram_addr: int, rows: list[int]) -> str:
        return self._emit_tile_row_unary("Softplus", "V_SOFTPLUS_V", vram_addr, rows)

    def tile_row_to_fpram_asm(
        self, vram_addr: int, row_map: list[tuple[int, int]]
    ) -> str:
        """Copy whole VRAM rows into FPRAM via S_MAP_FP_V, one instruction per row.

        `row_map` is [(vram_row_idx, fpram_base_addr)], and each entry moves the full
        MLEN-wide row to FPRAM[fpram_base_addr : +MLEN]. This is the bulk inverse of
        `S_MAP_V_FP`; the alternative (one-hot `V_MUL_VV` + `V_RED_SUM` + `S_ST_FP`
        per element) costs 3 instructions per *scalar* rather than 1 per *row*.

        Note the operand order: `S_MAP_FP_V rd, rs1, imm` takes rd = FPRAM base
        register and rs1 = VRAM row register, mirroring `S_MAP_V_FP` so that in both
        cases `rd` names the destination memory.
        """
        gp_regs = self._reg.allocate_gp(3)
        gp_src, gp_dst, gp_loop = gp_regs
        try:
            asm = IsaBuilder().comment(f"Tile Row -> FPRAM from VRAM[{vram_addr}]")
            rows = [row for row, _ in row_map]
            fp_addrs = [addr_ for _, addr_ in row_map]
            row_prog = self._row_progression(rows)
            fp_prog = self._row_progression(fp_addrs)

            if row_prog is not None and fp_prog is not None:
                row_start, row_count, row_step = row_prog
                fp_start, _, fp_step = fp_prog
                use_stream = (
                    self.stream_addressing
                    and min(row_step, fp_step) >= 0
                    and self._stream_is_profitable(
                        trips=row_count,
                        slots=2,
                        saved_per_trip=int(bool(row_step)) + int(bool(fp_step)),
                        baseline_setup_instructions=2,
                    )
                )
                if use_stream:
                    gp_value = self._reg.allocate_gp(1)[0]
                    try:
                        self._append_stream_binding(
                            asm,
                            value_gp=gp_value,
                            slot=0,
                            target_register=gp_src,
                            target_is_fp=False,
                            base=vram_addr + row_start * self.mlen,
                            advance=row_step * self.mlen,
                            count=row_count,
                            packet_elements=self.mlen,
                            storage_atom=self.stream_storage_atom,
                        )
                        self._append_stream_binding(
                            asm,
                            value_gp=gp_value,
                            slot=1,
                            target_register=gp_dst,
                            target_is_fp=False,
                            base=fp_start,
                            advance=fp_step,
                            count=row_count,
                            packet_elements=1,
                            storage_atom=1,
                            write=True,
                        )
                        asm.instr("C_LOOP_START", gp(gp_loop), row_count)
                        asm.instr("S_MAP_FP_V", gp(gp_dst), gp(gp_src), 0)
                        asm.instr("C_LOOP_END", gp(gp_loop))
                        self._append_stream_reset(
                            asm,
                            slot=0,
                            target_register=gp_src,
                            target_is_fp=False,
                        )
                        self._append_stream_reset(
                            asm,
                            slot=1,
                            target_register=gp_dst,
                            target_is_fp=False,
                        )
                    finally:
                        self._reg.free_gp([gp_value])
                else:
                    asm.instr(
                        "S_ADDI_INT",
                        gp(gp_src),
                        gp(0),
                        vram_addr + row_start * self.mlen,
                    )
                    asm.instr("S_ADDI_INT", gp(gp_dst), gp(0), fp_start)
                    asm.instr("C_LOOP_START", gp(gp_loop), row_count)
                    asm.instr("S_MAP_FP_V", gp(gp_dst), gp(gp_src), 0)
                    if row_step:
                        asm.instr(
                            "S_ADDI_INT",
                            gp(gp_src),
                            gp(gp_src),
                            row_step * self.mlen,
                        )
                    if fp_step:
                        asm.instr("S_ADDI_INT", gp(gp_dst), gp(gp_dst), fp_step)
                    asm.instr("C_LOOP_END", gp(gp_loop))
            else:
                for row_idx, fpram_addr in row_map:
                    asm.instr(
                        "S_ADDI_INT", gp(gp_src), gp(0), vram_addr + row_idx * self.mlen
                    )
                    asm.instr("S_ADDI_INT", gp(gp_dst), gp(0), fpram_addr)
                    asm.instr("S_MAP_FP_V", gp(gp_dst), gp(gp_src), 0)

            return self._emit(asm)
        finally:
            self._reg.free_gp(gp_regs)

    def tile_row_sub_fp_asm(
        self, vram_addr: int, row_map: list[tuple[int, int]]
    ) -> str:
        return self._emit_tile_row_fp_scalar(
            "Sub", "V_SUB_VF", vram_addr, row_map, opcode_extra_args=(0, 0)
        )

    def tile_row_mul_fp_asm(
        self, vram_addr: int, row_map: list[tuple[int, int]]
    ) -> str:
        return self._emit_tile_row_fp_scalar(
            "Mul", "V_MUL_VF", vram_addr, row_map, opcode_extra_args=(0,)
        )

    def tile_row_max_fp_asm(
        self, vram_addr: int, row_map: list[tuple[int, int]]
    ) -> str:
        return self._emit_tile_row_fp_scalar(
            "Max", "V_MAX_VF", vram_addr, row_map, opcode_extra_args=(0,)
        )

    def tile_row_min_fp_asm(
        self, vram_addr: int, row_map: list[tuple[int, int]]
    ) -> str:
        return self._emit_tile_row_fp_scalar(
            "Min", "V_MIN_VF", vram_addr, row_map, opcode_extra_args=(0,)
        )

    def tile_row_add_fp_asm(
        self, vram_addr: int, row_map: list[tuple[int, int]]
    ) -> str:
        return self._emit_tile_row_fp_scalar(
            "Add", "V_ADD_VF", vram_addr, row_map, opcode_extra_args=(0,)
        )

    def tile_row_add_asm(self, dst_addr: int, src_addr: int, rows: list[int]) -> str:
        return self._emit_tile_row_vector_op(
            "Add", "V_ADD_VV", dst_addr, src_addr, rows
        )

    def tile_row_sub_asm(self, dst_addr: int, src_addr: int, rows: list[int]) -> str:
        return self._emit_tile_row_vector_op(
            "Sub", "V_SUB_VV", dst_addr, src_addr, rows
        )

    def tile_row_mul_asm(self, dst_addr: int, src_addr: int, rows: list[int]) -> str:
        return self._emit_tile_row_vector_op(
            "Mul", "V_MUL_VV", dst_addr, src_addr, rows
        )

    def tile_row_fma_fp_asm(
        self,
        dst_addr: int,
        src_addr: int,
        row_map: list[tuple[int, int, int]],
    ) -> str:
        """``dst[d] += src[s] * FPRAM[f]``; entries applied in order."""
        return self._emit_tile_row_fma(dst_addr, src_addr, row_map)

    def tile_row_fma_fp_sweep_asm(
        self,
        dst_addr: int,
        src_addr: int,
        fpram_base: int,
        dst_rows: list[int],
        src_rows: list[int],
    ) -> str:
        """Walk one FPRAM slot per row pair, starting at ``fpram_base``.

        Named ``_sweep`` and not ``_broadcast``: in this file ``_broadcast``
        means one slot applied to *every* row, which is the opposite of walking
        a slot per row. See :meth:`tile_row_fma_fp_broadcast_asm` for that.
        """
        if len(dst_rows) != len(src_rows):
            raise ValueError(
                f"tile_row_fma row counts differ: "
                f"{len(dst_rows)} destinations, {len(src_rows)} sources"
            )
        row_map = [
            (d, s, fpram_base + i) for i, (d, s) in enumerate(zip(dst_rows, src_rows))
        ]
        return self._emit_tile_row_fma(dst_addr, src_addr, row_map)

    def tile_row_fma_fp_broadcast_asm(
        self,
        dst_addr: int,
        src_addr: int,
        fpram_scalar_addr: int,
        dst_rows: list[int],
        src_rows: list[int],
    ) -> str:
        """One FPRAM slot applied to every row pair."""
        if len(dst_rows) != len(src_rows):
            raise ValueError(
                f"tile_row_fma row counts differ: "
                f"{len(dst_rows)} destinations, {len(src_rows)} sources"
            )
        row_map = [(d, s, fpram_scalar_addr) for d, s in zip(dst_rows, src_rows)]
        return self._emit_tile_row_fma(dst_addr, src_addr, row_map)

    def tile_row_mul_fp_broadcast_asm(
        self, vram_addr: int, fpram_scalar_addr: int, rows: list[int]
    ) -> str:
        row_map = [(r, fpram_scalar_addr) for r in rows]
        return self.tile_row_mul_fp_asm(vram_addr, row_map)

    # The broadcast variants below apply ONE FPRAM slot to every listed row, as
    # opposed to the `tile_row_*_fp` family which walks a different slot per row.
    # Both shapes are needed by Mamba: the decay scalars are per-row, while the
    # dt clamp bounds and the +1.0 of the sigmoid are single constants.
    def tile_row_add_fp_broadcast_asm(
        self, vram_addr: int, fpram_scalar_addr: int, rows: list[int]
    ) -> str:
        return self.tile_row_add_fp_asm(
            vram_addr, [(r, fpram_scalar_addr) for r in rows]
        )

    def tile_row_max_fp_broadcast_asm(
        self, vram_addr: int, fpram_scalar_addr: int, rows: list[int]
    ) -> str:
        return self.tile_row_max_fp_asm(
            vram_addr, [(r, fpram_scalar_addr) for r in rows]
        )

    def tile_row_min_fp_broadcast_asm(
        self, vram_addr: int, fpram_scalar_addr: int, rows: list[int]
    ) -> str:
        return self.tile_row_min_fp_asm(
            vram_addr, [(r, fpram_scalar_addr) for r in rows]
        )

    def tile_row_sub_fp_broadcast_asm(
        self,
        vram_addr: int,
        fpram_scalar_addr: int,
        rows: list[int],
        reverse: bool = False,
    ) -> str:
        """``row = row - c`` (reverse=False) or ``row = c - row`` (reverse=True).

        The reverse form is `V_SUB_VF`'s `rorder=1`, and it is the only way to
        negate a vector on this ISA (``0.0 - x`` with the hardwired ``f0``). The
        Mamba decay matrix needs it to form ``cs_i - cs_j`` from a row holding
        ``cs_j`` and a scalar holding ``cs_i``.
        """
        row_map = [(r, fpram_scalar_addr) for r in rows]
        return self._emit_tile_row_fp_scalar(
            "RSub" if reverse else "Sub",
            "V_SUB_VF",
            vram_addr,
            row_map,
            opcode_extra_args=(0, 1 if reverse else 0),
        )

    def vram_fill_zero_asm(
        self,
        vram_addr: int,
        rows: list[int],
    ) -> str:
        """
        VRAM Fill Zero: fill specified rows with 0.

        For each row_idx in rows:
            VRAM[row] = 0
        """
        if not rows:
            return self._emit(
                IsaBuilder().comment(
                    f"=== VRAM Fill Zero: VRAM[{vram_addr}] rows [] = 0 ==="
                )
            )

        gp_regs = self._reg.allocate_gp(2)
        gp_dst, gp_loop = gp_regs
        try:
            asm = IsaBuilder().comment(
                f"=== VRAM Fill Zero: VRAM[{vram_addr}] rows {rows} = 0 ==="
            )
            prog = self._row_progression(rows)

            if prog is not None:
                row_start, row_count, row_step = prog
                asm.instr(
                    "S_ADDI_INT", gp(gp_dst), gp(0), vram_addr + row_start * self.mlen
                )
                asm.instr("C_LOOP_START", gp(gp_loop), row_count)
                asm.instr("V_MUL_VF", gp(gp_dst), gp(gp_dst), fp(0), 0)
                asm.instr("S_ADDI_INT", gp(gp_dst), gp(gp_dst), row_step * self.mlen)
                asm.instr("C_LOOP_END", gp(gp_loop))
            else:
                for row_idx in rows:
                    row_addr = vram_addr + row_idx * self.mlen
                    asm.instr("S_ADDI_INT", gp(gp_dst), gp(0), row_addr)
                    asm.instr("V_MUL_VF", gp(gp_dst), gp(gp_dst), fp(0), 0)

            return self._emit(asm)
        finally:
            self._reg.free_gp(gp_regs)


__all__ = ["IsaTileRowMixin"]
