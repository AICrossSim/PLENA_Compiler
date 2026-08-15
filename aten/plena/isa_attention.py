"""Flash-attention ISA helpers for IsaCompiler."""

from __future__ import annotations

import math

from compiler.asm_templates._imm import load_large_int
from compiler.aten.plena.attention_pipeline_plan import (
    GQATimingProfile,
    RowPipelineOp,
    interleave_row_chains,
)
from compiler.aten.plena.native_layout import SoftmaxRowGroupPlan
from compiler.aten.plena.native_layout import (
    SOFTMAX_ROW_ISSUE_SCHEDULE_GROUP_SERIAL_V1,
    SOFTMAX_ROW_ISSUE_SCHEDULE_WAVEFRONT_V1,
)


class IsaAttentionMixin:
    # =========================================================================
    # Flash Attention Implementation
    # =========================================================================

    def _online_softmax_row_bank_asm(
        self,
        *,
        mlen: int,
        s_address: int,
        state_address: int,
        output_address: int | None,
        scale: float,
        rows: int,
        valid_cols: int | None,
        first_block: bool,
    ) -> str:
        """Lower one online-softmax block through the rtl-v6 row engine."""

        plan = SoftmaxRowGroupPlan.build(
            sequence=getattr(self, "_native_sequence_packing", None),
            rows=rows,
            mlen=mlen,
            row_lanes=int(self.softmax_row_lanes),
            valid_cols=valid_cols,
        )
        gp_s, gp_state, gp_o, gp_loop = self.register_allocator.allocate_gp(4)
        fp_scale = 1
        lines = [
            "; === Multi-row online softmax / row-bank state ===",
        ]
        if scale != 1.0:
            lines.append(f"S_LD_FP f{fp_scale}, gp0, 1")
        runs = plan.execution_runs()

        def emit_phase(
            run,
            command: str,
            *,
            use_score: bool = False,
            use_state: bool = False,
            use_output: bool = False,
        ) -> None:
            group = run.first
            if use_score:
                lines.extend(
                    load_large_int(gp_s, s_address + group.base_row * mlen)
                )
            if use_state:
                lines.extend(
                    load_large_int(gp_state, state_address + group.base_row)
                )
            if use_output:
                if output_address is None:
                    raise ValueError(
                        "row-bank softmax phase requires an output address"
                    )
                lines.extend(
                    load_large_int(gp_o, output_address + group.base_row * mlen)
                )
            if run.count > 1:
                lines.append(f"C_LOOP_START gp{gp_loop}, {run.count}")
            lines.append(
                command.format(
                    active_rows=group.active_rows,
                    row_log2=int(math.log2(group.row_lanes)),
                )
            )
            if run.count > 1:
                if use_score:
                    lines.append(
                        f"S_ADDI_INT gp{gp_s}, gp{gp_s}, "
                        f"{group.row_lanes * mlen}"
                    )
                if use_state:
                    lines.append(
                        f"S_ADDI_INT gp{gp_state}, gp{gp_state}, "
                        f"{group.row_lanes}"
                    )
                if use_output:
                    lines.append(
                        f"S_ADDI_INT gp{gp_o}, gp{gp_o}, "
                        f"{group.row_lanes * mlen}"
                    )
                lines.append(f"C_LOOP_END gp{gp_loop}")

        phase_specs: list[tuple[str, bool, bool, bool]] = []
        if scale != 1.0:
            phase_specs.append(
                (
                    f"V_MUL_ROWS_F gp{gp_s}, gp{gp_s}, f{fp_scale}, "
                    "{active_rows}, {row_log2}",
                    True,
                    False,
                    False,
                )
            )
        phase_specs.extend(
            (
                (
                    f"V_RED_MAX_ROWS gp{gp_state}, gp{gp_s}, "
                    "{active_rows}, {row_log2}",
                    True,
                    True,
                    False,
                ),
                (
                    f"V_SFM_MAX_ROWS gp{gp_state}, gp{gp_state}, "
                    "{active_rows}, {row_log2}",
                    False,
                    True,
                    False,
                ),
            )
        )
        if not first_block:
            if output_address is None:
                raise ValueError("recurrent row-bank softmax requires output address")
            phase_specs.append(
                (
                    f"V_MUL_ROWS_STATS gp{gp_o}, gp{gp_o}, gp{gp_state}, "
                    "{active_rows}, {row_log2}",
                    False,
                    True,
                    True,
                )
            )
        phase_specs.extend(
            (
                (
                    f"V_SUB_ROWS gp{gp_s}, gp{gp_s}, gp{gp_state}, "
                    "{active_rows}, {row_log2}",
                    True,
                    True,
                    False,
                ),
                (
                    f"V_EXP_ROWS gp{gp_s}, gp{gp_s}, "
                    "{active_rows}, {row_log2}",
                    True,
                    False,
                    False,
                ),
                (
                    f"V_RED_SUM_ROWS gp{gp_state}, gp{gp_s}, "
                    "{active_rows}, {row_log2}",
                    True,
                    True,
                    False,
                ),
                (
                    f"V_SFM_SUM_ROWS gp{gp_state}, gp{gp_state}, "
                    "{active_rows}, {row_log2}",
                    False,
                    True,
                    False,
                ),
            )
        )

        if self.softmax_row_issue_schedule == SOFTMAX_ROW_ISSUE_SCHEDULE_WAVEFRONT_V1:
            lines.append("; wavefront-v1: stream independent groups one phase at a time")
            for command, use_score, use_state, use_output in phase_specs:
                for run in runs:
                    emit_phase(
                        run,
                        command,
                        use_score=use_score,
                        use_state=use_state,
                        use_output=use_output,
                    )
        elif self.softmax_row_issue_schedule == SOFTMAX_ROW_ISSUE_SCHEDULE_GROUP_SERIAL_V1:
            lines.append("; group-serial-v1 compatibility schedule")
            for run in runs:
                # A multi-group affine run must be split here; otherwise the
                # hardware loop would still execute phase-major.
                for group in run.groups:
                    singleton = type(run)((group,))
                    for command, use_score, use_state, use_output in phase_specs:
                        emit_phase(
                            singleton,
                            command,
                            use_score=use_score,
                            use_state=use_state,
                            use_output=use_output,
                        )
        else:  # Constructor validation should make this unreachable.
            raise ValueError(
                f"unsupported softmax row issue schedule "
                f"{self.softmax_row_issue_schedule!r}"
            )
        self.register_allocator.free_gp([gp_s, gp_state, gp_o, gp_loop])
        if hasattr(self, "record_softmax_row_stats"):
            metadata = plan.metadata()
            vector_fp_width = int(getattr(self, "vector_fp_width", 0))
            metadata.update(
                {
                    "softmax_read_width_bits_per_issue": (
                        int(metadata["softmax_read_elements_per_issue"])
                        * vector_fp_width
                    ),
                    "softmax_row_group_ii": 1,
                    "softmax_independent_ii": 1,
                    "softmax_dependent_ii": "rtl-shadow-pending",
                    "softmax_row_issue_schedule": self.softmax_row_issue_schedule,
                    "softmax_row_group_ii_fidelity": (
                        "architectural_ideal_ii1_unvalidated"
                    ),
                    "rtl_vector_machine_integration": (
                        "production-vector-machine-banked-sram-module-v1"
                    ),
                    "rtl_vector_machine_validation_status": (
                        "module-cocotb-r1-r2-r4-r8-passed"
                    ),
                    "rtl_top_level_validation_status": "not-run-deferred",
                    "rtl_full_machine_integration": False,
                    "rtl_timing_validation_status": (
                        "module-functional-passed-physical-timing-pending"
                    ),
                }
            )
            active = plan.active_rows
            metadata.update(
                {
                    "softmax_state_reads": (0 if first_block else 2 * active),
                    "softmax_state_writes": 2 * active,
                    "scalar_state_loads_elided": (0 if first_block else 2 * active),
                    "scalar_state_stores_elided": (1 if first_block else 2) * active,
                }
            )
            self.record_softmax_row_stats(metadata)
        return "\n".join(lines) + "\n"

    def _online_softmax_asm(
        self,
        mlen: int,
        s_address: int,
        m_start_address: int,
        scale: float = 1.0,
        rows: int | None = None,
        valid_cols: int | None = None,
    ) -> str:
        """
        Online Softmax Computation.

        Per row of S:
          1. m_curr = max(S[row], m_old)
          2. m_res = exp(m_old - m_curr)              # used to update O downstream
          3. S'[row] = S[row] - m_curr
          4. P[row] = exp(S'[row])
          5. l_new = l_old * m_res + sum(P[row])

        FP SRAM layout (from m_start_address):
          [0, mlen):        m_old / m_curr
          [mlen, 2*mlen):   m_res = exp(m_old - m_curr)
          [2*mlen, 3*mlen): l_old / l_new
        """
        pipelined = self._online_softmax_pipeline_asm(
            mlen=mlen,
            s_address=s_address,
            m_start_address=m_start_address,
            scale=scale,
            rows=rows,
            valid_cols=valid_cols,
            first_block=False,
        )
        if pipelined is not None:
            return pipelined
        segmented = self._online_softmax_segmented_asm(
            mlen=mlen,
            s_address=s_address,
            m_start_address=m_start_address,
            scale=scale,
            rows=rows,
            valid_cols=valid_cols,
            first_block=False,
        )
        if segmented is not None:
            return segmented
        if getattr(self, "unroll_attention", False):
            return self._online_softmax_asm_unrolled(
                mlen=mlen,
                s_address=s_address,
                m_start_address=m_start_address,
                scale=scale,
                valid_cols=valid_cols,
            )

        gp_regs = self.register_allocator.allocate_gp(5)
        gp_s = gp_regs[0]
        gp_m_addr = gp_regs[1]
        gp_m_res_addr = gp_regs[2]
        gp_l_addr = gp_regs[3]
        gp_loop = gp_regs[4]

        # Fixed FP register allocation for online softmax pipeline.
        # These registers are shared across _online_softmax_asm, _scale_o_asm,
        # and _final_scaling_asm — they MUST remain consistent across all three.
        # WARNING: Do not use f1-f6 in any code that calls these methods.
        fp_m_old = 1  # f1: m_old value
        fp_m_res = 2  # f2: exp(m_old - m_curr)
        fp_l_old = 3  # f3: l_old value
        fp_sum_p = 4  # f4: sum(P)
        fp_scale = 5  # f5: scale factor
        fp_row_max = 6  # f6: current row max (temporary)
        rtl_v2 = getattr(self, "vector_scalar_schedule", "legacy") in {
            "rtl-v2",
            "rtl-v3",
            "rtl-v4",
            "rtl-v5",
            "rtl-v6",
        }

        lines = []
        lines.append("; === Online Softmax ===")

        # Set address registers
        lines.extend(load_large_int(gp_s, s_address))
        lines.extend(load_large_int(gp_m_addr, m_start_address))
        lines.append(f"S_ADDI_INT gp{gp_m_res_addr}, gp{gp_m_addr}, {mlen}")
        lines.append(f"S_ADDI_INT gp{gp_l_addr}, gp{gp_m_res_addr}, {mlen}")

        mask_en = 0
        if valid_cols is not None and valid_cols < mlen:
            mask_unit = getattr(self, "hlen", mlen)
            valid_lanes = max(1, math.ceil(valid_cols / mask_unit))
            mask_bits = (1 << valid_lanes) - 1
            lines.append(f"S_ADDI_INT gp{gp_loop}, gp0, {mask_bits}")
            lines.append(f"C_SET_V_MASK_REG gp{gp_loop}")
            mask_en = 1

        # scale factor is pre-loaded at FP SRAM addr 1 by the flash-attention driver.
        if scale != 1.0:
            lines.append(f"S_LD_FP f{fp_scale}, gp0, 1")

        loop_rows = mlen if rows is None else rows
        lines.append(f"C_LOOP_START gp{gp_loop}, {loop_rows}")
        lines.append(f"S_LD_FP f{fp_m_old}, gp{gp_m_addr}, 0")
        lines.append(
            f"S_MV_FP f{fp_m_res}, f{fp_m_old}"
            if rtl_v2
            else f"S_ADD_FP f{fp_m_res}, f{fp_m_old}, f0"
        )

        if scale != 1.0:
            lines.append(f"V_MUL_VF gp{gp_s}, gp{gp_s}, f{fp_scale}, {mask_en}")

        # V_RED_MAX accumulates into its destination FP register in the
        # emulator, so clear the per-row max accumulator before each row.
        lines.append(f"S_LD_FP f{fp_row_max}, gp0, 2")
        lines.append(f"V_RED_MAX f{fp_row_max}, gp{gp_s}, {mask_en}")

        # m_curr = max(row_max, m_old) — online softmax must retain the running max.
        lines.append(f"S_MAX_FP f{fp_m_old}, f{fp_row_max}, f{fp_m_old}")

        lines.append(f"S_SUB_FP f{fp_m_res}, f{fp_m_res}, f{fp_m_old}")
        lines.append(f"S_EXP_FP f{fp_m_res}, f{fp_m_res}, 0")

        lines.append(f"S_ST_FP f{fp_m_res}, gp{gp_m_res_addr}, 0")
        lines.append(f"S_ST_FP f{fp_m_old}, gp{gp_m_addr}, 0")

        lines.append(f"V_SUB_VF gp{gp_s}, gp{gp_s}, f{fp_m_old}, {mask_en}, 0")
        lines.append(f"V_EXP_V gp{gp_s}, gp{gp_s}, {mask_en}, 0")

        lines.append(f"S_LD_FP f{fp_l_old}, gp{gp_l_addr}, 0")

        lines.append(
            f"S_MV_FP f{fp_sum_p}, f0"
            if rtl_v2
            else f"S_ADD_FP f{fp_sum_p}, f0, f0"
        )
        lines.append(f"V_RED_SUM f{fp_sum_p}, gp{gp_s}, {mask_en}, 0")

        lines.append(f"S_MUL_FP f{fp_l_old}, f{fp_l_old}, f{fp_m_res}")
        lines.append(f"S_ADD_FP f{fp_l_old}, f{fp_l_old}, f{fp_sum_p}")

        lines.append(f"S_ST_FP f{fp_l_old}, gp{gp_l_addr}, 0")

        lines.append(f"S_ADDI_INT gp{gp_s}, gp{gp_s}, {mlen}")
        lines.append(f"S_ADDI_INT gp{gp_m_addr}, gp{gp_m_addr}, 1")
        lines.append(f"S_ADDI_INT gp{gp_m_res_addr}, gp{gp_m_res_addr}, 1")
        lines.append(f"S_ADDI_INT gp{gp_l_addr}, gp{gp_l_addr}, 1")
        lines.append(f"C_LOOP_END gp{gp_loop}")

        self.register_allocator.free_gp(gp_regs)
        return "\n".join(lines) + "\n"

    def _online_softmax_streamed_asm(
        self,
        *,
        mlen: int,
        s_address: int,
        m_address: int,
        l_address: int,
        output_address: int,
        output_head_slot: int,
        scale: float = 1.0,
        rows: int | None = None,
        valid_cols: int | None = None,
        last_block: bool = False,
    ) -> str:
        """Update online-softmax state and consume ``m_res`` in-register.

        This is the streamed-v2 recurrence.  The output lane is scaled in the
        same row chain that creates ``m_res``; no m_res SRAM allocation exists.
        """

        gp_s, gp_m, gp_l, gp_o, gp_loop, gp_aux, gp_selector = (
            self.register_allocator.allocate_gp(7)
        )
        fp_m = 1
        fp_m_res = 2
        fp_l = 3
        fp_sum = 4
        fp_scale = 5
        fp_row_max = 6
        lines = [
            "; === Online Softmax Streamed v2 ===",
        ]
        plan = getattr(self, "_native_sequence_packing", None)
        compact_segments = (
            plan is not None
            and plan.mode == "compact"
            and plan.seq_len <= mlen
            and (valid_cols is None or valid_cols >= mlen)
            and (rows is None or rows == plan.attention_group_seq_len)
        )
        if compact_segments:
            slot_rows = int(plan.batch_slot_rows)
            if slot_rows <= 0 or slot_rows & (slot_rows - 1):
                raise ValueError(
                    f"streamed softmax segment width must be a power of two, "
                    f"got {slot_rows}"
                )
            row_groups = tuple(
                (
                    slot * slot_rows,
                    int(plan.seq_len),
                    slot,
                    slot_rows,
                )
                for slot in range(int(plan.batch_pack_factor))
            )
        else:
            row_groups = ((0, mlen if rows is None else rows, None, mlen),)
        valid_mask = None
        mask_en = 0
        if valid_cols is not None and valid_cols < mlen:
            mask_unit = getattr(self, "hlen", mlen)
            valid_lanes = max(1, math.ceil(valid_cols / mask_unit))
            valid_mask = (1 << valid_lanes) - 1
            lines.extend(
                (
                    f"S_ADDI_INT gp{gp_aux}, gp0, {valid_mask}",
                    f"C_SET_V_MASK_REG gp{gp_aux}",
                )
            )
            mask_en = 1
        if scale != 1.0:
            lines.append(f"S_LD_FP f{fp_scale}, gp0, 1")
        overwrite = (
            getattr(self, "reduction_output_mode", "accumulate-v1")
            == "overwrite-v1"
        )
        hoist_selector = (
            getattr(self, "selector_schedule", "legacy") == "hoisted-v1"
        )
        for row_start, row_count, segment, segment_width in row_groups:
            lines.extend(load_large_int(gp_s, s_address + row_start * mlen))
            lines.extend(load_large_int(gp_m, m_address + row_start))
            lines.extend(load_large_int(gp_l, l_address + row_start))
            lines.extend(
                load_large_int(
                    gp_o, output_address + row_start * mlen
                )
            )
            if segment is not None and hoist_selector:
                lines.append(f"S_ADDI_INT gp{gp_selector}, gp0, {segment}")
            lines.append(f"C_LOOP_START gp{gp_loop}, {row_count}")
            lines.extend(
                (
                    f"S_LD_FP f{fp_m}, gp{gp_m}, 0",
                    f"S_MV_FP f{fp_m_res}, f{fp_m}",
                )
            )
            if scale != 1.0:
                lines.append(
                    f"V_MUL_VF gp{gp_s}, gp{gp_s}, f{fp_scale}, {mask_en}"
                )
            if not overwrite:
                lines.append(f"S_LD_FP f{fp_row_max}, gp0, 2")
            if segment is None:
                lines.append(
                    f"V_RED_MAX{'_OVR' if overwrite else ''} "
                    f"f{fp_row_max}, gp{gp_s}, {mask_en}"
                )
            else:
                if not hoist_selector:
                    lines.append(f"S_ADDI_INT gp{gp_selector}, gp0, {segment}")
                lines.append(
                    f"V_RED_MAX_SEG{'_OVR' if overwrite else ''} "
                    f"f{fp_row_max}, gp{gp_s}, gp{gp_selector}, "
                    f"{int(math.log2(segment_width))}"
                )
            lines.extend(
                (
                    f"S_MAX_FP f{fp_m}, f{fp_row_max}, f{fp_m}",
                    f"S_SUB_FP f{fp_m_res}, f{fp_m_res}, f{fp_m}",
                    f"S_EXP_FP f{fp_m_res}, f{fp_m_res}, 0",
                )
            )
            if not last_block:
                lines.append(f"S_ST_FP f{fp_m}, gp{gp_m}, 0")
            output_mask = 1 << output_head_slot
            lines.extend(
                (
                    f"S_ADDI_INT gp{gp_aux}, gp0, {output_mask}",
                    f"C_SET_V_MASK_REG gp{gp_aux}",
                    f"V_MUL_VF gp{gp_o}, gp{gp_o}, f{fp_m_res}, 1",
                )
            )
            if valid_mask is not None:
                lines.extend(
                    (
                        f"S_ADDI_INT gp{gp_aux}, gp0, {valid_mask}",
                        f"C_SET_V_MASK_REG gp{gp_aux}",
                    )
                )
            lines.extend(
                (
                    f"V_SUB_VF gp{gp_s}, gp{gp_s}, f{fp_m}, {mask_en}, 0",
                    f"V_EXP_V gp{gp_s}, gp{gp_s}, {mask_en}, 0",
                    f"S_LD_FP f{fp_l}, gp{gp_l}, 0",
                )
            )
            if not overwrite:
                lines.append(f"S_MV_FP f{fp_sum}, f0")
            if segment is None:
                lines.append(
                    f"V_RED_SUM{'_OVR' if overwrite else ''} "
                    f"f{fp_sum}, gp{gp_s}, {mask_en}, 0"
                )
            else:
                if not hoist_selector:
                    lines.append(f"S_ADDI_INT gp{gp_selector}, gp0, {segment}")
                lines.append(
                    f"V_RED_SUM_SEG{'_OVR' if overwrite else ''} "
                    f"f{fp_sum}, gp{gp_s}, gp{gp_selector}, "
                    f"{int(math.log2(segment_width))}"
                )
            lines.extend(
                (
                    f"S_MUL_FP f{fp_l}, f{fp_l}, f{fp_m_res}",
                    f"S_ADD_FP f{fp_l}, f{fp_l}, f{fp_sum}",
                    f"S_ST_FP f{fp_l}, gp{gp_l}, 0",
                    f"S_ADDI_INT gp{gp_s}, gp{gp_s}, {mlen}",
                    f"S_ADDI_INT gp{gp_m}, gp{gp_m}, 1",
                    f"S_ADDI_INT gp{gp_l}, gp{gp_l}, 1",
                    f"S_ADDI_INT gp{gp_o}, gp{gp_o}, {mlen}",
                    f"C_LOOP_END gp{gp_loop}",
                )
            )
        if hasattr(self, "record_vector_scalar_stats"):
            active_rows = sum(row_count for _, row_count, _, _ in row_groups)
            selector_rows = sum(
                row_count
                for _, row_count, segment, _ in row_groups
                if segment is not None
            )
            setup_count = sum(
                1 for _, _, segment, _ in row_groups if segment is not None
            )
            self.record_vector_scalar_stats(
                {
                    "selector_loads_before": 2 * selector_rows,
                    "selector_loads_hoisted": (
                        2 * selector_rows - setup_count if hoist_selector else 0
                    ),
                    "selector_setup_instructions": (
                        setup_count if hoist_selector else 2 * selector_rows
                    ),
                    "neutral_accumulator_setups_before": 2 * active_rows,
                    "neutral_accumulator_setups_elided": (
                        2 * active_rows if overwrite else 0
                    ),
                }
            )
        self.register_allocator.free_gp(
            [gp_s, gp_m, gp_l, gp_o, gp_loop, gp_aux, gp_selector]
        )
        return "\n".join(lines) + "\n"

    def _online_softmax_first_block_streamed_asm(
        self,
        *,
        mlen: int,
        s_address: int,
        m_address: int,
        l_address: int,
        scale: float = 1.0,
        rows: int | None = None,
        valid_cols: int | None = None,
        last_block: bool = False,
    ) -> str:
        """Create first-block state without scalar copy operations."""

        gp_s, gp_m, gp_l, gp_loop, gp_aux, gp_selector = (
            self.register_allocator.allocate_gp(6)
        )
        fp_m = 1
        fp_l = 3
        fp_scale = 5
        lines = [
            "; === Online Softmax First Block Streamed v2 ===",
        ]
        plan = getattr(self, "_native_sequence_packing", None)
        compact_segments = (
            plan is not None
            and plan.mode == "compact"
            and plan.seq_len <= mlen
            and (valid_cols is None or valid_cols >= mlen)
            and (rows is None or rows == plan.attention_group_seq_len)
        )
        if compact_segments:
            slot_rows = int(plan.batch_slot_rows)
            if slot_rows <= 0 or slot_rows & (slot_rows - 1):
                raise ValueError(
                    f"streamed softmax segment width must be a power of two, "
                    f"got {slot_rows}"
                )
            row_groups = tuple(
                (
                    slot * slot_rows,
                    int(plan.seq_len),
                    slot,
                    slot_rows,
                )
                for slot in range(int(plan.batch_pack_factor))
            )
        else:
            row_groups = ((0, mlen if rows is None else rows, None, mlen),)
        mask_en = 0
        if valid_cols is not None and valid_cols < mlen:
            mask_unit = getattr(self, "hlen", mlen)
            valid_lanes = max(1, math.ceil(valid_cols / mask_unit))
            lines.extend(
                (
                    f"S_ADDI_INT gp{gp_aux}, gp0, {(1 << valid_lanes) - 1}",
                    f"C_SET_V_MASK_REG gp{gp_aux}",
                )
            )
            mask_en = 1
        if scale != 1.0:
            lines.append(f"S_LD_FP f{fp_scale}, gp0, 1")
        overwrite = (
            getattr(self, "reduction_output_mode", "accumulate-v1")
            == "overwrite-v1"
        )
        hoist_selector = (
            getattr(self, "selector_schedule", "legacy") == "hoisted-v1"
        )
        for row_start, row_count, segment, segment_width in row_groups:
            lines.extend(load_large_int(gp_s, s_address + row_start * mlen))
            lines.extend(load_large_int(gp_m, m_address + row_start))
            lines.extend(load_large_int(gp_l, l_address + row_start))
            if segment is not None and hoist_selector:
                lines.append(f"S_ADDI_INT gp{gp_selector}, gp0, {segment}")
            lines.append(f"C_LOOP_START gp{gp_loop}, {row_count}")
            if scale != 1.0:
                lines.append(
                    f"V_MUL_VF gp{gp_s}, gp{gp_s}, f{fp_scale}, {mask_en}"
                )
            if not overwrite:
                lines.append(f"S_LD_FP f{fp_m}, gp0, 2")
            if segment is None:
                lines.append(
                    f"V_RED_MAX{'_OVR' if overwrite else ''} "
                    f"f{fp_m}, gp{gp_s}, {mask_en}"
                )
            else:
                if not hoist_selector:
                    lines.append(f"S_ADDI_INT gp{gp_selector}, gp0, {segment}")
                lines.append(
                    f"V_RED_MAX_SEG{'_OVR' if overwrite else ''} "
                    f"f{fp_m}, gp{gp_s}, gp{gp_selector}, "
                    f"{int(math.log2(segment_width))}"
                )
            if not last_block:
                lines.append(f"S_ST_FP f{fp_m}, gp{gp_m}, 0")
            lines.extend(
                (
                    f"V_SUB_VF gp{gp_s}, gp{gp_s}, f{fp_m}, {mask_en}, 0",
                    f"V_EXP_V gp{gp_s}, gp{gp_s}, {mask_en}, 0",
                )
            )
            if not overwrite:
                lines.append(f"S_LD_FP f{fp_l}, gp0, 0")
            if segment is None:
                lines.append(
                    f"V_RED_SUM{'_OVR' if overwrite else ''} "
                    f"f{fp_l}, gp{gp_s}, {mask_en}, 0"
                )
            else:
                if not hoist_selector:
                    lines.append(f"S_ADDI_INT gp{gp_selector}, gp0, {segment}")
                lines.append(
                    f"V_RED_SUM_SEG{'_OVR' if overwrite else ''} "
                    f"f{fp_l}, gp{gp_s}, gp{gp_selector}, "
                    f"{int(math.log2(segment_width))}"
                )
            lines.extend(
                (
                    f"S_ST_FP f{fp_l}, gp{gp_l}, 0",
                    f"S_ADDI_INT gp{gp_s}, gp{gp_s}, {mlen}",
                    f"S_ADDI_INT gp{gp_m}, gp{gp_m}, 1",
                    f"S_ADDI_INT gp{gp_l}, gp{gp_l}, 1",
                    f"C_LOOP_END gp{gp_loop}",
                )
            )
        if hasattr(self, "record_vector_scalar_stats"):
            active_rows = sum(row_count for _, row_count, _, _ in row_groups)
            selector_rows = sum(
                row_count
                for _, row_count, segment, _ in row_groups
                if segment is not None
            )
            setup_count = sum(
                1 for _, _, segment, _ in row_groups if segment is not None
            )
            self.record_vector_scalar_stats(
                {
                    "selector_loads_before": 2 * selector_rows,
                    "selector_loads_hoisted": (
                        2 * selector_rows - setup_count if hoist_selector else 0
                    ),
                    "selector_setup_instructions": (
                        setup_count if hoist_selector else 2 * selector_rows
                    ),
                    "neutral_accumulator_setups_before": 2 * active_rows,
                    "neutral_accumulator_setups_elided": (
                        2 * active_rows if overwrite else 0
                    ),
                }
            )
        self.register_allocator.free_gp(
            [gp_s, gp_m, gp_l, gp_loop, gp_aux, gp_selector]
        )
        return "\n".join(lines) + "\n"

    def _online_softmax_pipeline_asm(
        self,
        *,
        mlen: int,
        s_address: int,
        m_start_address: int,
        scale: float,
        rows: int | None,
        valid_cols: int | None,
        first_block: bool,
    ) -> str | None:
        """Interleave independent softmax rows using measured RTL-v3 timing.

        Every row retains the exact arithmetic instruction order emitted by the
        row-serial lowering. Only instructions from different rows are mixed.
        The generated hardware-loop body is parsed by both the assembler and
        CostEmitter, so the two paths cannot drift in opcode order.
        """

        if getattr(self, "gqa_pipeline_schedule", "row-serial") != "row-interleaved-v1":
            return None
        if getattr(self, "vector_scalar_schedule", "legacy") not in {
            "rtl-v3",
            "rtl-v4",
            "rtl-v5",
            "rtl-v6",
        }:
            raise ValueError(
                "row-interleaved-v1 requires vector_scalar_schedule='rtl-v3' "
                "'rtl-v4', 'rtl-v5', or 'rtl-v6'"
            )

        timing: GQATimingProfile = getattr(self, "gqa_timing_profile", None)
        if timing is None:
            timing = GQATimingProfile.load(
                getattr(self, "gqa_timing_calibration", None)
            )
            self.gqa_timing_profile = timing

        width = 3 if first_block else 2
        regs_per_row = 4 if first_block else 5
        required_fp = width * regs_per_row + int(scale != 1.0)
        if required_fp > timing.fp_register_count - 1:
            raise RuntimeError(
                f"softmax width {width} needs {required_fp} allocatable FP registers, "
                f"artifact provides {timing.fp_register_count - 1}"
            )
        if width > timing.rob_depth:
            raise RuntimeError(
                f"softmax width {width} exceeds scalar ROB depth {timing.rob_depth}"
            )

        sequence_plan = getattr(self, "_native_sequence_packing", None)
        compact = (
            sequence_plan is not None
            and sequence_plan.mode == "compact"
            and sequence_plan.seq_len <= mlen
            and valid_cols is None
            and (rows is None or rows == sequence_plan.attention_group_seq_len)
        )
        if compact:
            slot_rows = int(sequence_plan.batch_slot_rows)
            if slot_rows <= 0 or slot_rows & (slot_rows - 1):
                raise ValueError(
                    f"segment softmax slot width must be a power of two, got {slot_rows}"
                )
            row_ranges = tuple(
                (slot * slot_rows, int(sequence_plan.seq_len), slot, slot_rows)
                for slot in range(int(sequence_plan.batch_pack_factor))
            )
        else:
            loop_rows = mlen if rows is None else int(rows)
            row_ranges = ((0, loop_rows, None, mlen),)

        fp_regs = self.register_allocator.allocate_fp(required_fp)
        score_gps = self.register_allocator.allocate_gp(width)
        gp_m, gp_m_res, gp_l, gp_loop, gp_aux = self.register_allocator.allocate_gp(5)
        fp_scale = fp_regs[-1] if scale != 1.0 else None
        row_fp = [
            fp_regs[row * regs_per_row : (row + 1) * regs_per_row]
            for row in range(width)
        ]
        lines = [
            "; === RTL-v3 row-interleaved online softmax ===",
            f"; rows_in_flight={width}, first_block={int(first_block)}, timing_sha256={timing.sha256}",
        ]
        if fp_scale is not None:
            lines.append(f"S_LD_FP f{fp_scale}, gp0, 1")

        scalar = timing.scalar_ready
        scalar_ii = timing.scalar_ii

        def scalar_op(text: str, kind: str) -> RowPipelineOp:
            return RowPipelineOp(
                text,
                f"scalar_{kind}",
                scalar[kind],
                scalar_ii.get(kind, 1),
            )

        def vector_op(
            text: str,
            kind: str,
            *,
            latency: int | None = None,
            blocking: bool = False,
        ) -> RowPipelineOp:
            return RowPipelineOp(
                text,
                "vector",
                timing.vector_ready[kind] if latency is None else latency,
                timing.vector_ii,
                is_blocking_reduction=blocking,
            )

        def build_chain(
            row: int,
            *,
            segment: int | None,
            segment_width: int,
        ) -> tuple[RowPipelineOp, ...]:
            regs = row_fp[row]
            if first_block:
                fp_m, fp_l, fp_sum, fp_row_max = regs
                fp_m_res = None
            else:
                fp_m, fp_m_res, fp_l, fp_sum, fp_row_max = regs
            gp_s = score_gps[row]
            mask_en = int(
                segment is None
                and valid_cols is not None
                and valid_cols < mlen
            )
            chain: list[RowPipelineOp] = []
            if fp_scale is not None:
                chain.append(
                    vector_op(
                        f"V_MUL_VF gp{gp_s}, gp{gp_s}, f{fp_scale}, {mask_en}",
                        "mul_vf",
                    )
                )
            chain.append(scalar_op(f"S_LD_FP f{fp_row_max}, gp0, 2", "load"))
            if segment is None:
                reduction = f"V_RED_MAX f{fp_row_max}, gp{gp_s}, {mask_en}"
            else:
                reduction = (
                    f"V_RED_MAX_SEG f{fp_row_max}, gp{gp_s}, gp{gp_aux}, "
                    f"{int(math.log2(segment_width))}"
                )
            chain.append(
                vector_op(
                    reduction,
                    "reduction",
                    latency=timing.reduction_latency(
                        kind="max", segment_width=segment_width
                    ),
                    blocking=True,
                )
            )
            if first_block:
                chain.extend(
                    (
                        scalar_op(f"S_MV_FP f{fp_m}, f{fp_row_max}", "move"),
                        scalar_op(f"S_ST_FP f{fp_m}, gp{gp_m}, {row}", "store"),
                    )
                )
            else:
                assert fp_m_res is not None
                chain.extend(
                    (
                        scalar_op(f"S_LD_FP f{fp_m}, gp{gp_m}, {row}", "load"),
                        scalar_op(f"S_MV_FP f{fp_m_res}, f{fp_m}", "move"),
                        scalar_op(
                            f"S_MAX_FP f{fp_m}, f{fp_row_max}, f{fp_m}", "max"
                        ),
                        scalar_op(
                            f"S_SUB_FP f{fp_m_res}, f{fp_m_res}, f{fp_m}", "sub"
                        ),
                        scalar_op(
                            f"S_EXP_FP f{fp_m_res}, f{fp_m_res}, 0", "exp"
                        ),
                        scalar_op(
                            f"S_ST_FP f{fp_m_res}, gp{gp_m_res}, {row}", "store"
                        ),
                        scalar_op(f"S_ST_FP f{fp_m}, gp{gp_m}, {row}", "store"),
                    )
                )
            chain.extend(
                (
                    vector_op(
                        f"V_SUB_VF gp{gp_s}, gp{gp_s}, f{fp_m}, {mask_en}, 0",
                        "sub_vf",
                    ),
                    vector_op(f"V_EXP_V gp{gp_s}, gp{gp_s}, {mask_en}, 0", "exp"),
                    scalar_op(f"S_MV_FP f{fp_sum}, f0", "move"),
                )
            )
            if segment is None:
                reduction = f"V_RED_SUM f{fp_sum}, gp{gp_s}, {mask_en}, 0"
            else:
                reduction = (
                    f"V_RED_SUM_SEG f{fp_sum}, gp{gp_s}, gp{gp_aux}, "
                    f"{int(math.log2(segment_width))}"
                )
            chain.append(
                vector_op(
                    reduction,
                    "reduction",
                    latency=timing.reduction_latency(
                        kind="sum", segment_width=segment_width
                    ),
                    blocking=True,
                )
            )
            if first_block:
                chain.extend(
                    (
                        scalar_op(f"S_MV_FP f{fp_l}, f{fp_sum}", "move"),
                        scalar_op(f"S_ST_FP f{fp_l}, gp{gp_l}, {row}", "store"),
                    )
                )
            else:
                assert fp_m_res is not None
                chain.extend(
                    (
                        scalar_op(f"S_LD_FP f{fp_l}, gp{gp_l}, {row}", "load"),
                        scalar_op(
                            f"S_MUL_FP f{fp_l}, f{fp_l}, f{fp_m_res}", "mul"
                        ),
                        scalar_op(
                            f"S_ADD_FP f{fp_l}, f{fp_l}, f{fp_sum}", "add"
                        ),
                        scalar_op(f"S_ST_FP f{fp_l}, gp{gp_l}, {row}", "store"),
                    )
                )
            return tuple(chain)

        try:
            for row_start, row_count, segment, segment_width in row_ranges:
                for lane, gp_s in enumerate(score_gps):
                    lines.extend(
                        load_large_int(
                            gp_s, s_address + (row_start + lane) * mlen
                        )
                    )
                lines.extend(load_large_int(gp_m, m_start_address + row_start))
                lines.extend(
                    load_large_int(gp_m_res, m_start_address + mlen + row_start)
                )
                lines.extend(
                    load_large_int(gp_l, m_start_address + 2 * mlen + row_start)
                )
                if segment is not None:
                    lines.append(f"S_ADDI_INT gp{gp_aux}, gp0, {segment}")
                elif valid_cols is not None and valid_cols < mlen:
                    mask_unit = getattr(self, "hlen", mlen)
                    valid_lanes = max(1, math.ceil(valid_cols / mask_unit))
                    lines.extend(
                        (
                            f"S_ADDI_INT gp{gp_aux}, gp0, {(1 << valid_lanes) - 1}",
                            f"C_SET_V_MASK_REG gp{gp_aux}",
                        )
                    )

                full_groups, tail = divmod(row_count, width)
                if full_groups:
                    lines.append(f"C_LOOP_START gp{gp_loop}, {full_groups}")
                    chains = tuple(
                        build_chain(
                            row,
                            segment=segment,
                            segment_width=segment_width,
                        )
                        for row in range(width)
                    )
                    lines.extend(interleave_row_chains(chains))
                    for gp_s in score_gps:
                        lines.append(
                            f"S_ADDI_INT gp{gp_s}, gp{gp_s}, {width * mlen}"
                        )
                    lines.extend(
                        (
                            f"S_ADDI_INT gp{gp_m}, gp{gp_m}, {width}",
                            f"S_ADDI_INT gp{gp_m_res}, gp{gp_m_res}, {width}",
                            f"S_ADDI_INT gp{gp_l}, gp{gp_l}, {width}",
                            f"C_LOOP_END gp{gp_loop}",
                        )
                    )
                if tail:
                    lines.append(f"; row-interleaved tail rows={tail}")
                    chains = tuple(
                        build_chain(
                            row,
                            segment=segment,
                            segment_width=segment_width,
                        )
                        for row in range(tail)
                    )
                    lines.extend(interleave_row_chains(chains))

            if hasattr(self, "record_gqa_pipeline_stats"):
                active_rows = sum(row_count for _, row_count, _, _ in row_ranges)
                self.record_gqa_pipeline_stats(
                    {
                        "softmax_first_block_pipeline_width"
                        if first_block
                        else "softmax_recurrent_pipeline_width": width,
                        "interleaved_softmax_rows": active_rows,
                    }
                )
            return "\n".join(lines) + "\n"
        finally:
            self.register_allocator.free_fp(fp_regs)
            self.register_allocator.free_gp(
                [*score_gps, gp_m, gp_m_res, gp_l, gp_loop, gp_aux]
            )

    def _online_softmax_segmented_asm(
        self,
        *,
        mlen: int,
        s_address: int,
        m_start_address: int,
        scale: float,
        rows: int | None,
        valid_cols: int | None,
        first_block: bool,
    ) -> str | None:
        """Use aligned batch slots and the RTL segment-reduction opcodes.

        QK still computes a full MLEN-wide score row. The block-diagonal causal
        mask makes every non-local slot ``-inf``; full-vector subtract/exp turns
        those entries into zero before PV. Only the max/sum trees are shortened
        to the selected power-of-two batch slot.
        """

        if getattr(self, "vector_scalar_schedule", "legacy") not in {
            "rtl-v2",
            "rtl-v3",
            "rtl-v4",
            "rtl-v5",
            "rtl-v6",
        }:
            return None
        plan = getattr(self, "_native_sequence_packing", None)
        if plan is None or plan.mode != "compact" or plan.seq_len > mlen:
            return None
        if valid_cols is not None:
            return None
        if rows is not None and rows != plan.attention_group_seq_len:
            return None
        slot_rows = int(plan.batch_slot_rows)
        if slot_rows <= 0 or slot_rows & (slot_rows - 1):
            raise ValueError(f"segment softmax slot width must be a power of two, got {slot_rows}")
        segment_log2 = int(math.log2(slot_rows))
        gp_s, gp_m, gp_m_res, gp_l, gp_loop, gp_segment = self.register_allocator.allocate_gp(6)
        fp_m = 1
        fp_m_res = 2
        fp_l = 3
        fp_sum = 4
        fp_scale = 5
        fp_row_max = 6
        lines = [
            "; === Aligned-slot segment online softmax ===",
        ]
        if scale != 1.0:
            lines.append(f"S_LD_FP f{fp_scale}, gp0, 1")

        for slot in range(plan.batch_pack_factor):
            row_start = slot * slot_rows
            lines.extend(load_large_int(gp_s, s_address + row_start * mlen))
            lines.extend(load_large_int(gp_m, m_start_address + row_start))
            lines.extend(load_large_int(gp_m_res, m_start_address + mlen + row_start))
            lines.extend(load_large_int(gp_l, m_start_address + 2 * mlen + row_start))
            lines.append(f"S_ADDI_INT gp{gp_segment}, gp0, {slot}")
            lines.append(f"C_LOOP_START gp{gp_loop}, {plan.seq_len}")
            if scale != 1.0:
                lines.append(f"V_MUL_VF gp{gp_s}, gp{gp_s}, f{fp_scale}, 0")
            lines.extend(
                [
                    f"S_LD_FP f{fp_row_max}, gp0, 2",
                    f"V_RED_MAX_SEG f{fp_row_max}, gp{gp_s}, gp{gp_segment}, {segment_log2}",
                ]
            )
            if first_block:
                lines.extend(
                    [
                        f"S_MV_FP f{fp_m}, f{fp_row_max}",
                        f"S_ST_FP f{fp_m}, gp{gp_m}, 0",
                    ]
                )
            else:
                lines.extend(
                    [
                        f"S_LD_FP f{fp_m}, gp{gp_m}, 0",
                        f"S_MV_FP f{fp_m_res}, f{fp_m}",
                        f"S_MAX_FP f{fp_m}, f{fp_row_max}, f{fp_m}",
                        f"S_SUB_FP f{fp_m_res}, f{fp_m_res}, f{fp_m}",
                        f"S_EXP_FP f{fp_m_res}, f{fp_m_res}, 0",
                        f"S_ST_FP f{fp_m_res}, gp{gp_m_res}, 0",
                        f"S_ST_FP f{fp_m}, gp{gp_m}, 0",
                    ]
                )
            lines.extend(
                [
                    f"V_SUB_VF gp{gp_s}, gp{gp_s}, f{fp_m}, 0, 0",
                    f"V_EXP_V gp{gp_s}, gp{gp_s}, 0, 0",
                    f"S_MV_FP f{fp_sum}, f0",
                    f"V_RED_SUM_SEG f{fp_sum}, gp{gp_s}, gp{gp_segment}, {segment_log2}",
                ]
            )
            if first_block:
                lines.extend(
                    [
                        f"S_MV_FP f{fp_l}, f{fp_sum}",
                        f"S_ST_FP f{fp_l}, gp{gp_l}, 0",
                    ]
                )
            else:
                lines.extend(
                    [
                        f"S_LD_FP f{fp_l}, gp{gp_l}, 0",
                        f"S_MUL_FP f{fp_l}, f{fp_l}, f{fp_m_res}",
                        f"S_ADD_FP f{fp_l}, f{fp_l}, f{fp_sum}",
                        f"S_ST_FP f{fp_l}, gp{gp_l}, 0",
                    ]
                )
            lines.extend(
                [
                    f"S_ADDI_INT gp{gp_s}, gp{gp_s}, {mlen}",
                    f"S_ADDI_INT gp{gp_m}, gp{gp_m}, 1",
                    f"S_ADDI_INT gp{gp_m_res}, gp{gp_m_res}, 1",
                    f"S_ADDI_INT gp{gp_l}, gp{gp_l}, 1",
                    f"C_LOOP_END gp{gp_loop}",
                ]
            )

        if hasattr(self, "record_vector_scalar_stats"):
            reductions = 2 * plan.batch_pack_factor * plan.seq_len
            self.record_vector_scalar_stats(
                {
                    "softmax_segment_reductions_emitted": reductions,
                    "softmax_inactive_rows_elided": plan.attention_group_seq_len
                    - plan.batch_pack_factor * plan.seq_len,
                }
            )
        self.register_allocator.free_gp(
            [gp_s, gp_m, gp_m_res, gp_l, gp_loop, gp_segment]
        )
        return "\n".join(lines) + "\n"

    def _online_softmax_asm_unrolled(
        self,
        mlen: int,
        s_address: int,
        m_start_address: int,
        scale: float = 1.0,
        valid_cols: int | None = None,
    ) -> str:
        """Legacy Python-unrolled online softmax emission, kept for A/B comparisons."""
        gp_regs = self.register_allocator.allocate_gp(5)
        gp_s = gp_regs[0]
        gp_m_addr = gp_regs[1]
        gp_m_res_addr = gp_regs[2]
        gp_l_addr = gp_regs[3]
        gp_mask = gp_regs[4]

        fp_m_old = 1
        fp_m_res = 2
        fp_l_old = 3
        fp_sum_p = 4
        fp_scale = 5
        fp_row_max = 6
        rtl_v2 = getattr(self, "vector_scalar_schedule", "legacy") in {
            "rtl-v2",
            "rtl-v3",
            "rtl-v4",
            "rtl-v5",
            "rtl-v6",
        }

        lines = []
        lines.append("; === Online Softmax ===")
        lines.extend(load_large_int(gp_s, s_address))
        lines.extend(load_large_int(gp_m_addr, m_start_address))
        lines.append(f"S_ADDI_INT gp{gp_m_res_addr}, gp{gp_m_addr}, {mlen}")
        lines.append(f"S_ADDI_INT gp{gp_l_addr}, gp{gp_m_res_addr}, {mlen}")

        if scale != 1.0:
            lines.append(f"S_LD_FP f{fp_scale}, gp0, 1")

        mask_en = 0
        if valid_cols is not None and valid_cols < mlen:
            mask_unit = getattr(self, "hlen", mlen)
            valid_lanes = max(1, math.ceil(valid_cols / mask_unit))
            mask_bits = (1 << valid_lanes) - 1
            lines.append(f"S_ADDI_INT gp{gp_mask}, gp0, {mask_bits}")
            lines.append(f"C_SET_V_MASK_REG gp{gp_mask}")
            mask_en = 1

        for row in range(mlen):
            lines.append(f"; Row {row}")
            lines.append(f"S_LD_FP f{fp_m_old}, gp{gp_m_addr}, {row}")
            lines.append(
                f"S_MV_FP f{fp_m_res}, f{fp_m_old}"
                if rtl_v2
                else f"S_ADD_FP f{fp_m_res}, f{fp_m_old}, f0"
            )

            if scale != 1.0:
                lines.append(f"V_MUL_VF gp{gp_s}, gp{gp_s}, f{fp_scale}, {mask_en}")

            lines.append(f"S_LD_FP f{fp_row_max}, gp0, 2")
            lines.append(f"V_RED_MAX f{fp_row_max}, gp{gp_s}, {mask_en}")

            # m_curr = max(row_max, m_old) — online softmax must retain the running max.
            lines.append(f"S_MAX_FP f{fp_m_old}, f{fp_row_max}, f{fp_m_old}")

            lines.append(f"S_SUB_FP f{fp_m_res}, f{fp_m_res}, f{fp_m_old}")
            lines.append(f"S_EXP_FP f{fp_m_res}, f{fp_m_res}, 0")

            lines.append(f"S_ST_FP f{fp_m_res}, gp{gp_m_res_addr}, {row}")
            lines.append(f"S_ST_FP f{fp_m_old}, gp{gp_m_addr}, {row}")

            lines.append(f"V_SUB_VF gp{gp_s}, gp{gp_s}, f{fp_m_old}, {mask_en}, 0")
            lines.append(f"V_EXP_V gp{gp_s}, gp{gp_s}, {mask_en}, 0")

            lines.append(f"S_LD_FP f{fp_l_old}, gp{gp_l_addr}, {row}")

            lines.append(
                f"S_MV_FP f{fp_sum_p}, f0"
                if rtl_v2
                else f"S_ADD_FP f{fp_sum_p}, f0, f0"
            )
            lines.append(f"V_RED_SUM f{fp_sum_p}, gp{gp_s}, {mask_en}, 0")

            lines.append(f"S_MUL_FP f{fp_l_old}, f{fp_l_old}, f{fp_m_res}")
            lines.append(f"S_ADD_FP f{fp_l_old}, f{fp_l_old}, f{fp_sum_p}")

            lines.append(f"S_ST_FP f{fp_l_old}, gp{gp_l_addr}, {row}")

            lines.append(f"S_ADDI_INT gp{gp_s}, gp{gp_s}, {mlen}")

        self.register_allocator.free_gp(gp_regs)
        return "\n".join(lines) + "\n"

    def _online_softmax_first_block_asm(
        self,
        mlen: int,
        s_address: int,
        m_start_address: int,
        scale: float = 1.0,
        rows: int | None = None,
        valid_cols: int | None = None,
    ) -> str:
        """Initialize online-softmax state from the first valid K block.

        The generic recurrence starts from ``m=-inf`` and ``l=0``.  For the
        first block those values make ``m_res`` exactly zero, so loading and
        updating the old state cannot affect the result.  This form writes the
        same running ``m`` and ``l`` state needed by later K blocks while
        avoiding the redundant scalar recurrence and FP-SRAM initialization.
        """

        pipelined = self._online_softmax_pipeline_asm(
            mlen=mlen,
            s_address=s_address,
            m_start_address=m_start_address,
            scale=scale,
            rows=rows,
            valid_cols=valid_cols,
            first_block=True,
        )
        if pipelined is not None:
            return pipelined
        segmented = self._online_softmax_segmented_asm(
            mlen=mlen,
            s_address=s_address,
            m_start_address=m_start_address,
            scale=scale,
            rows=rows,
            valid_cols=valid_cols,
            first_block=True,
        )
        if segmented is not None:
            return segmented

        gp_s, gp_m_addr, gp_l_addr, gp_loop = self.register_allocator.allocate_gp(4)
        fp_m = 1
        fp_l = 3
        fp_sum_p = 4
        fp_scale = 5
        fp_row_max = 6
        rtl_v2 = getattr(self, "vector_scalar_schedule", "legacy") in {
            "rtl-v2",
            "rtl-v3",
            "rtl-v4",
            "rtl-v5",
            "rtl-v6",
        }

        lines = [
            "; === Online Softmax First Block ===",
            *load_large_int(gp_s, s_address),
            *load_large_int(gp_m_addr, m_start_address),
            f"S_ADDI_INT gp{gp_l_addr}, gp{gp_m_addr}, {2 * mlen}",
        ]

        mask_en = 0
        if valid_cols is not None and valid_cols < mlen:
            mask_unit = getattr(self, "hlen", mlen)
            valid_lanes = max(1, math.ceil(valid_cols / mask_unit))
            mask_bits = (1 << valid_lanes) - 1
            lines.extend(
                [
                    f"S_ADDI_INT gp{gp_loop}, gp0, {mask_bits}",
                    f"C_SET_V_MASK_REG gp{gp_loop}",
                ]
            )
            mask_en = 1

        if scale != 1.0:
            lines.append(f"S_LD_FP f{fp_scale}, gp0, 1")

        loop_rows = mlen if rows is None else rows
        lines.append(f"C_LOOP_START gp{gp_loop}, {loop_rows}")
        if scale != 1.0:
            lines.append(f"V_MUL_VF gp{gp_s}, gp{gp_s}, f{fp_scale}, {mask_en}")

        lines.extend(
            [
                f"S_LD_FP f{fp_row_max}, gp0, 2",
                f"V_RED_MAX f{fp_row_max}, gp{gp_s}, {mask_en}",
                # Preserve the scalar-copy rounding of the generic path while
                # replacing max(row_max, -inf) with its exact result.
                (
                    f"S_MV_FP f{fp_m}, f{fp_row_max}"
                    if rtl_v2
                    else f"S_ADD_FP f{fp_m}, f{fp_row_max}, f0"
                ),
                f"S_ST_FP f{fp_m}, gp{gp_m_addr}, 0",
                f"V_SUB_VF gp{gp_s}, gp{gp_s}, f{fp_m}, {mask_en}, 0",
                f"V_EXP_V gp{gp_s}, gp{gp_s}, {mask_en}, 0",
                (
                    f"S_MV_FP f{fp_sum_p}, f0"
                    if rtl_v2
                    else f"S_ADD_FP f{fp_sum_p}, f0, f0"
                ),
                f"V_RED_SUM f{fp_sum_p}, gp{gp_s}, {mask_en}, 0",
                # The generic first update computes 0 * 0 + sum(P).  Keep a
                # scalar ALU copy so the destination format/rounding is equal.
                (
                    f"S_MV_FP f{fp_l}, f{fp_sum_p}"
                    if rtl_v2
                    else f"S_ADD_FP f{fp_l}, f{fp_sum_p}, f0"
                ),
                f"S_ST_FP f{fp_l}, gp{gp_l_addr}, 0",
                f"S_ADDI_INT gp{gp_s}, gp{gp_s}, {mlen}",
                f"S_ADDI_INT gp{gp_m_addr}, gp{gp_m_addr}, 1",
                f"S_ADDI_INT gp{gp_l_addr}, gp{gp_l_addr}, 1",
                f"C_LOOP_END gp{gp_loop}",
            ]
        )
        self.register_allocator.free_gp([gp_s, gp_m_addr, gp_l_addr, gp_loop])
        return "\n".join(lines) + "\n"

    def _pv_multiply_asm(
        self,
        mlen: int,
        blen: int,
        head_dim: int,
        p_address: int,
        v_hbm_offset_reg: int,
        v_hbm_offset: int,
        pv_address: int,
        rows: int | None = None,
        pv_physical_rows: int | None = None,
    ) -> str:
        """
        Compute PV = P @ V via M_MM.

        P:  (mlen, mlen)     in VRAM   (softmax output)
        V:  (mlen, head_dim) in HBM    (prefetched into MSRAM in mlen-wide column blocks)
        PV: (mlen, head_dim) in VRAM

        M_MM computes one (blen, mlen) @ (mlen, blen) -> (blen, blen) in a single op
        (K=mlen done in one shot). For head_dim > mlen, V is split into head_dim/mlen
        column blocks; the outer loop iterates blocks, middle loop iterates blen-wide
        V columns within a block, inner loop iterates blen-wide P rows.
        """
        # PV is stored column-block-major with `pv_physical_rows` rows (= min(mlen,
        # seq_len) for the decoder), so each head_dim col-block spans
        # pv_physical_rows*mlen, not mlen*mlen. They coincide at seq>=mlen, but for
        # seq<mlen using mlen*mlen writes col-blocks 1.. out of bounds, leaving
        # head_dim cols mlen.. zero (PV shrinks by mlen/head_dim).
        if pv_physical_rows is None:
            pv_physical_rows = mlen

        if getattr(self, "unroll_attention", False):
            return self._pv_multiply_asm_unrolled(
                mlen=mlen,
                blen=blen,
                head_dim=head_dim,
                p_address=p_address,
                v_hbm_offset_reg=v_hbm_offset_reg,
                v_hbm_offset=v_hbm_offset,
                pv_address=pv_address,
                pv_physical_rows=pv_physical_rows,
            )

        gp_regs = self.register_allocator.allocate_gp(8)
        gp_p = gp_regs[0]
        gp_v = gp_regs[1]
        gp_pv = gp_regs[2]
        gp_hbm = gp_regs[3]
        gp_stride = gp_regs[4]
        gp_pv_col_base = gp_regs[5]
        gp_v_loop = gp_regs[6]
        gp_p_loop = gp_regs[7]

        num_v_col_blocks = max(1, math.ceil(head_dim / mlen))
        tiles_per_mlen = mlen // blen
        p_row_groups = tiles_per_mlen if rows is None else max(1, min(tiles_per_mlen, (rows + blen - 1) // blen))

        lines = []
        lines.append("; === PV Multiply (P @ V) using M_MM ===")
        lines.append(f"; P: ({mlen}, {mlen}) @ V: ({mlen}, {head_dim}) -> PV: ({mlen}, {head_dim})")
        lines.append("; M_MM: (blen, mlen) @ (mlen, blen) -> (blen, blen), K=mlen in one shot")
        lines.append(f"; V split into {num_v_col_blocks} column blocks of width {mlen}")
        lines.append("; Storage layout: (batch, mlen, hidden/mlen), column-block major")

        # STRIDE was set to mlen by the flash-attention driver — do not overwrite it here.
        # M_MM_WO requires a nonzero stride reg (gp0=0 would be interpreted as stride=1).
        # With column-block-major storage, consecutive rows within a column block are
        # adjacent, so the writeback stride = 1.
        lines.append(f"S_ADDI_INT gp{gp_stride}, gp0, 1")

        for v_col_block in range(num_v_col_blocks):
            lines.append(
                f"; --- V column block {v_col_block} (columns {v_col_block * mlen} to {(v_col_block + 1) * mlen - 1}) ---"
            )

            # Prefetch V[:, v_col_block*mlen:(v_col_block+1)*mlen] (mlen × mlen) to MSRAM.
            # V is row-major in HBM: V[row, col] at offset row*head_dim + col, so the
            # column-block base offset = v_hbm_offset + v_col_block * mlen (elements).
            v_block_hbm_offset = v_hbm_offset + v_col_block * mlen
            lines.append(f"S_ADDI_INT gp{gp_v}, gp0, 0")
            lines.extend(load_large_int(gp_hbm, v_block_hbm_offset))
            lines.append(f"H_PREFETCH_M gp{gp_v}, gp{gp_hbm}, a{v_hbm_offset_reg}, 1, 1")

            pv_col_block_base = pv_address + v_col_block * pv_physical_rows * mlen
            lines.extend(load_large_int(gp_pv_col_base, pv_col_block_base))
            lines.append(f"C_LOOP_START gp{gp_v_loop}, {tiles_per_mlen}")
            lines.extend(load_large_int(gp_p, p_address))
            lines.append(f"S_ADDI_INT gp{gp_pv}, gp{gp_pv_col_base}, 0")
            lines.append(f"C_LOOP_START gp{gp_p_loop}, {p_row_groups}")
            lines.append(f"M_MM 0, gp{gp_v}, gp{gp_p}")
            lines.append(f"M_MM_WO gp{gp_pv}, gp{gp_stride}, 0")
            lines.append(f"S_ADDI_INT gp{gp_p}, gp{gp_p}, {blen * mlen}")
            lines.append(f"S_ADDI_INT gp{gp_pv}, gp{gp_pv}, {blen * mlen}")
            lines.append(f"C_LOOP_END gp{gp_p_loop}")
            lines.append(f"S_ADDI_INT gp{gp_v}, gp{gp_v}, {blen}")
            lines.append(f"S_ADDI_INT gp{gp_pv_col_base}, gp{gp_pv_col_base}, {blen}")
            lines.append(f"C_LOOP_END gp{gp_v_loop}")

        self.register_allocator.free_gp(gp_regs)
        return "\n".join(lines) + "\n"

    def _pv_multiply_asm_unrolled(
        self,
        mlen: int,
        blen: int,
        head_dim: int,
        p_address: int,
        v_hbm_offset_reg: int,
        v_hbm_offset: int,
        pv_address: int,
        pv_physical_rows: int | None = None,
    ) -> str:
        """Legacy Python-unrolled P @ V emission, kept for A/B comparisons."""
        if pv_physical_rows is None:
            pv_physical_rows = mlen
        gp_regs = self.register_allocator.allocate_gp(5)
        gp_p = gp_regs[0]
        gp_v = gp_regs[1]
        gp_pv = gp_regs[2]
        gp_hbm = gp_regs[3]
        gp_stride = gp_regs[4]

        num_v_col_blocks = max(1, math.ceil(head_dim / mlen))

        lines = []
        lines.append("; === PV Multiply (P @ V) using M_MM ===")
        lines.append(f"; P: ({mlen}, {mlen}) @ V: ({mlen}, {head_dim}) -> PV: ({mlen}, {head_dim})")
        lines.append("; M_MM: (blen, mlen) @ (mlen, blen) -> (blen, blen), K=mlen in one shot")
        lines.append(f"; V split into {num_v_col_blocks} column blocks of width {mlen}")
        lines.append("; Storage layout: (batch, mlen, hidden/mlen), column-block major")
        lines.append(f"S_ADDI_INT gp{gp_stride}, gp0, 1")

        for v_col_block in range(num_v_col_blocks):
            lines.append(
                f"; --- V column block {v_col_block} (columns {v_col_block * mlen} to {(v_col_block + 1) * mlen - 1}) ---"
            )
            v_block_hbm_offset = v_hbm_offset + v_col_block * mlen
            lines.append(f"S_ADDI_INT gp{gp_v}, gp0, 0")
            lines.extend(load_large_int(gp_hbm, v_block_hbm_offset))
            lines.append(f"H_PREFETCH_M gp{gp_v}, gp{gp_hbm}, a{v_hbm_offset_reg}, 1, 1")

            for v_col in range(mlen // blen):
                lines.append(f"; V column {v_col_block * mlen + v_col * blen}")
                v_msram_offset = v_col * blen
                lines.append(f"S_ADDI_INT gp{gp_v}, gp0, {v_msram_offset}")

                for p_row in range(mlen // blen):
                    p_row_addr = p_address + p_row * blen * mlen
                    lines.extend(load_large_int(gp_p, p_row_addr))
                    lines.append(f"M_MM 0, gp{gp_v}, gp{gp_p}")

                    pv_offset = v_col_block * pv_physical_rows * mlen + p_row * blen * mlen + v_col * blen
                    lines.extend(load_large_int(gp_pv, pv_address + pv_offset))
                    lines.append(f"M_MM_WO gp{gp_pv}, gp{gp_stride}, 0")

        self.register_allocator.free_gp(gp_regs)
        return "\n".join(lines) + "\n"

    def _scale_o_asm(
        self,
        mlen: int,
        head_dim: int,
        seq_len: int,
        m_res_address: int,
        o_address: int,
        row_offset: int = 0,
        rows: int | None = None,
    ) -> str:
        """Scale each row of O by m_res: O[row] *= m_res[row]."""
        if getattr(self, "unroll_attention", False):
            return self._scale_o_asm_unrolled(
                mlen=mlen,
                head_dim=head_dim,
                seq_len=seq_len,
                m_res_address=m_res_address,
                o_address=o_address,
                row_offset=row_offset,
            )

        gp_regs = self.register_allocator.allocate_gp(4)
        gp_m_res = gp_regs[0]
        gp_o_row_base = gp_regs[1]
        gp_o = gp_regs[2]
        gp_row_loop = gp_regs[3]
        fp_m_res = 1

        num_col_blocks = max(1, math.ceil(head_dim / mlen))
        loop_rows = mlen if rows is None else rows

        lines = []
        lines.append("; === Scale O by m_res ===")
        lines.append(f"; head_dim = {head_dim}, {num_col_blocks} mlen-blocks per row")
        lines.append(f"; seq_len = {seq_len}, row_offset = {row_offset}")

        if num_col_blocks == 1:
            o_addr = o_address + row_offset * mlen
            lines.extend(load_large_int(gp_m_res, m_res_address))
            lines.extend(load_large_int(gp_o, o_addr))
            lines.append(f"C_LOOP_START gp{gp_row_loop}, {loop_rows}")
            lines.append(f"S_LD_FP f{fp_m_res}, gp{gp_m_res}, 0")
            lines.append(f"V_MUL_VF gp{gp_o}, gp{gp_o}, f{fp_m_res}, 0")
            lines.append(f"S_ADDI_INT gp{gp_m_res}, gp{gp_m_res}, 1")
            lines.append(f"S_ADDI_INT gp{gp_o}, gp{gp_o}, {mlen}")
            lines.append(f"C_LOOP_END gp{gp_row_loop}")
        else:
            gp_col_loop = self.register_allocator.allocate_gp(1)[0]
            o_addr = o_address + row_offset * mlen
            lines.extend(load_large_int(gp_m_res, m_res_address))
            lines.extend(load_large_int(gp_o_row_base, o_addr))
            lines.append(f"C_LOOP_START gp{gp_row_loop}, {loop_rows}")
            lines.append(f"S_LD_FP f{fp_m_res}, gp{gp_m_res}, 0")
            lines.append(f"S_ADDI_INT gp{gp_o}, gp{gp_o_row_base}, 0")
            lines.append(f"C_LOOP_START gp{gp_col_loop}, {num_col_blocks}")
            lines.append(f"V_MUL_VF gp{gp_o}, gp{gp_o}, f{fp_m_res}, 0")
            lines.append(f"S_ADDI_INT gp{gp_o}, gp{gp_o}, {seq_len * mlen}")
            lines.append(f"C_LOOP_END gp{gp_col_loop}")
            lines.append(f"S_ADDI_INT gp{gp_m_res}, gp{gp_m_res}, 1")
            lines.append(f"S_ADDI_INT gp{gp_o_row_base}, gp{gp_o_row_base}, {mlen}")
            lines.append(f"C_LOOP_END gp{gp_row_loop}")
            self.register_allocator.free_gp([gp_col_loop])

        self.register_allocator.free_gp(gp_regs)
        return "\n".join(lines) + "\n"

    def _scale_o_asm_unrolled(
        self,
        mlen: int,
        head_dim: int,
        seq_len: int,
        m_res_address: int,
        o_address: int,
        row_offset: int = 0,
    ) -> str:
        """Legacy Python-unrolled O *= m_res emission, kept for A/B comparisons."""
        gp_regs = self.register_allocator.allocate_gp(2)
        gp_m_res = gp_regs[0]
        gp_o = gp_regs[1]
        fp_m_res = 1

        num_col_blocks = max(1, math.ceil(head_dim / mlen))

        lines = []
        lines.append("; === Scale O by m_res ===")
        lines.append(f"; head_dim = {head_dim}, {num_col_blocks} mlen-blocks per row")
        lines.append(f"; seq_len = {seq_len}, row_offset = {row_offset}")
        lines.extend(load_large_int(gp_m_res, m_res_address))

        for row in range(mlen):
            lines.append(f"S_LD_FP f{fp_m_res}, gp{gp_m_res}, {row}")
            actual_row = row_offset + row

            for col_block in range(num_col_blocks):
                o_addr = o_address + col_block * seq_len * mlen + actual_row * mlen
                lines.extend(load_large_int(gp_o, o_addr))
                lines.append(f"V_MUL_VF gp{gp_o}, gp{gp_o}, f{fp_m_res}, 0")

        self.register_allocator.free_gp(gp_regs)
        return "\n".join(lines) + "\n"

    def _add_pv_to_o_asm(
        self,
        mlen: int,
        head_dim: int,
        seq_len: int,
        pv_address: int,
        o_address: int,
        row_offset: int = 0,
    ) -> str:
        """Accumulate PV into O: O[row] += PV[row]."""
        gp_regs = self.register_allocator.allocate_gp(2)
        gp_o = gp_regs[0]
        gp_pv = gp_regs[1]

        num_col_blocks = max(1, math.ceil(head_dim / mlen))

        lines = []
        lines.append("; === Add PV to O ===")
        lines.append(f"; head_dim = {head_dim}, {num_col_blocks} mlen-blocks per row")
        lines.append(f"; seq_len = {seq_len}, row_offset = {row_offset}")

        for row in range(mlen):
            actual_row = row_offset + row

            for col_block in range(num_col_blocks):
                o_addr = o_address + col_block * seq_len * mlen + actual_row * mlen
                pv_addr = pv_address + col_block * mlen * mlen + row * mlen

                lines.extend(load_large_int(gp_o, o_addr))
                lines.extend(load_large_int(gp_pv, pv_addr))
                lines.append(f"V_ADD_VV gp{gp_o}, gp{gp_o}, gp{gp_pv}, 0")

        self.register_allocator.free_gp(gp_regs)
        return "\n".join(lines) + "\n"

    def _final_scaling_asm(
        self,
        mlen: int,
        head_dim: int,
        seq_len: int,
        l_address: int,
        o_address: int,
        row_offset: int = 0,
        rows: int | None = None,
    ) -> str:
        """
        Final scaling: O[row] /= l[row].

        V_MUL_VF processes mlen elements at a time; when head_dim > mlen,
        each row is split into head_dim // mlen mlen-wide blocks.
        """
        if getattr(self, "unroll_attention", False):
            return self._final_scaling_asm_unrolled(
                mlen=mlen,
                head_dim=head_dim,
                seq_len=seq_len,
                l_address=l_address,
                o_address=o_address,
                row_offset=row_offset,
            )

        gp_regs = self.register_allocator.allocate_gp(4)
        gp_l = gp_regs[0]
        gp_o_row_base = gp_regs[1]
        gp_o = gp_regs[2]
        gp_row_loop = gp_regs[3]
        fp_l = 1

        num_col_blocks = max(1, math.ceil(head_dim / mlen))
        loop_rows = mlen if rows is None else rows

        lines = []
        lines.append("; === Final Scaling O = O / l ===")
        lines.append(f"; head_dim = {head_dim}, {num_col_blocks} mlen-blocks per row")
        lines.append("; Storage layout: (seq_len, mlen, head_dim/mlen), column-block major")
        lines.append(f"; seq_len = {seq_len}, row_offset = {row_offset}")

        if num_col_blocks == 1:
            o_addr = o_address + row_offset * mlen
            lines.extend(load_large_int(gp_l, l_address))
            lines.extend(load_large_int(gp_o, o_addr))
            lines.append(f"C_LOOP_START gp{gp_row_loop}, {loop_rows}")
            lines.append(f"S_LD_FP f{fp_l}, gp{gp_l}, 0")
            lines.append(f"S_RECI_FP f{fp_l}, f{fp_l}, 0")
            lines.append(f"V_MUL_VF gp{gp_o}, gp{gp_o}, f{fp_l}, 0")
            lines.append(f"S_ADDI_INT gp{gp_l}, gp{gp_l}, 1")
            lines.append(f"S_ADDI_INT gp{gp_o}, gp{gp_o}, {mlen}")
            lines.append(f"C_LOOP_END gp{gp_row_loop}")
        else:
            gp_col_loop = self.register_allocator.allocate_gp(1)[0]
            o_addr = o_address + row_offset * mlen
            lines.extend(load_large_int(gp_l, l_address))
            lines.extend(load_large_int(gp_o_row_base, o_addr))
            lines.append(f"C_LOOP_START gp{gp_row_loop}, {loop_rows}")
            lines.append(f"S_LD_FP f{fp_l}, gp{gp_l}, 0")
            lines.append(f"S_RECI_FP f{fp_l}, f{fp_l}, 0")
            lines.append(f"S_ADDI_INT gp{gp_o}, gp{gp_o_row_base}, 0")
            lines.append(f"C_LOOP_START gp{gp_col_loop}, {num_col_blocks}")
            lines.append(f"V_MUL_VF gp{gp_o}, gp{gp_o}, f{fp_l}, 0")
            lines.append(f"S_ADDI_INT gp{gp_o}, gp{gp_o}, {seq_len * mlen}")
            lines.append(f"C_LOOP_END gp{gp_col_loop}")
            lines.append(f"S_ADDI_INT gp{gp_l}, gp{gp_l}, 1")
            lines.append(f"S_ADDI_INT gp{gp_o_row_base}, gp{gp_o_row_base}, {mlen}")
            lines.append(f"C_LOOP_END gp{gp_row_loop}")
            self.register_allocator.free_gp([gp_col_loop])

        self.register_allocator.free_gp(gp_regs)
        return "\n".join(lines) + "\n"

    def _final_scaling_asm_unrolled(
        self,
        mlen: int,
        head_dim: int,
        seq_len: int,
        l_address: int,
        o_address: int,
        row_offset: int = 0,
    ) -> str:
        """Legacy Python-unrolled final O /= l emission, kept for A/B comparisons."""
        gp_regs = self.register_allocator.allocate_gp(2)
        gp_l = gp_regs[0]
        gp_o = gp_regs[1]
        fp_l = 1

        num_col_blocks = max(1, math.ceil(head_dim / mlen))

        lines = []
        lines.append("; === Final Scaling O = O / l ===")
        lines.append(f"; head_dim = {head_dim}, {num_col_blocks} mlen-blocks per row")
        lines.append("; Storage layout: (seq_len, mlen, head_dim/mlen), column-block major")
        lines.append(f"; seq_len = {seq_len}, row_offset = {row_offset}")
        lines.extend(load_large_int(gp_l, l_address))

        for row in range(mlen):
            lines.append(f"S_LD_FP f{fp_l}, gp{gp_l}, {row}")
            lines.append(f"S_RECI_FP f{fp_l}, f{fp_l}, 0")
            actual_row = row_offset + row

            for col_block in range(num_col_blocks):
                o_addr = o_address + col_block * seq_len * mlen + actual_row * mlen
                lines.extend(load_large_int(gp_o, o_addr))
                lines.append(f"V_MUL_VF gp{gp_o}, gp{gp_o}, f{fp_l}, 0")

        self.register_allocator.free_gp(gp_regs)
        return "\n".join(lines) + "\n"

    def _reset_fpsram_asm(
        self,
        start_address: int,
        count: int,
        value_address: int,  # FP SRAM slot: 0 = zero, 2 = -inf
    ) -> str:
        """Reset a region of FP SRAM to the value at value_address."""
        gp_regs = self.register_allocator.allocate_gp(2)
        gp_addr = gp_regs[0]
        gp_loop = gp_regs[1]

        lines = []
        lines.append(f"; Reset FP SRAM [{start_address}, {start_address + count})")

        lines.append(f"S_ADDI_INT gp{gp_addr}, gp0, {start_address}")
        # Use f1 for FP scalar - FP registers don't go through GP allocator
        lines.append(f"S_LD_FP f1, gp0, {value_address}")

        if getattr(self, "unroll_attention", False):
            for i in range(count):
                lines.append(f"S_ST_FP f1, gp{gp_addr}, {i}")
        else:
            lines.append(f"C_LOOP_START gp{gp_loop}, {count}")
            lines.append(f"S_ST_FP f1, gp{gp_addr}, 0")
            lines.append(f"S_ADDI_INT gp{gp_addr}, gp{gp_addr}, 1")
            lines.append(f"C_LOOP_END gp{gp_loop}")

        self.register_allocator.free_gp(gp_regs)
        return "\n".join(lines) + "\n"

    def _reset_vram_asm(
        self,
        start_address: int,
        rows: int,
        cols: int,
        total_rows: int,
        mlen: int = 64,
        row_offset: int = 0,
    ) -> str:
        """
        Reset a region of VRAM to zero.

        V_MUL_VF processes mlen elements at a time; when cols > mlen, each
        row is split into cols // mlen mlen-wide blocks.
        """
        gp_regs = self.register_allocator.allocate_gp(2)
        gp_addr = gp_regs[0]
        gp_loop = gp_regs[1]

        num_col_blocks = (cols + mlen - 1) // mlen

        lines = []
        lines.append(f"; Reset VRAM rows [{row_offset}, {row_offset + rows}) of matrix at {start_address}")
        lines.append(f"; {rows} rows x {cols} cols, {num_col_blocks} blocks per row")
        lines.append("; Storage layout: (total_rows, mlen, cols/mlen), column-block major")
        lines.append(f"; total_rows = {total_rows}, row_offset = {row_offset}")

        if getattr(self, "unroll_attention", False):
            for row in range(rows):
                actual_row = row_offset + row
                for col_block in range(num_col_blocks):
                    addr = start_address + col_block * total_rows * mlen + actual_row * mlen
                    lines.extend(load_large_int(gp_addr, addr))
                    lines.append(f"V_MUL_VF gp{gp_addr}, gp{gp_addr}, f0, 0")
        else:
            for col_block in range(num_col_blocks):
                addr = start_address + col_block * total_rows * mlen + row_offset * mlen
                lines.append(f"; Column block {col_block}")
                lines.extend(load_large_int(gp_addr, addr))
                lines.append(f"C_LOOP_START gp{gp_loop}, {rows}")
                lines.append(f"V_MUL_VF gp{gp_addr}, gp{gp_addr}, f0, 0")
                lines.append(f"S_ADDI_INT gp{gp_addr}, gp{gp_addr}, {mlen}")
                lines.append(f"C_LOOP_END gp{gp_loop}")

        self.register_allocator.free_gp(gp_regs)
        return "\n".join(lines) + "\n"

    # =========================================================================
    # Expanded Flash Attention Operations
    # =========================================================================

    def init_online_softmax(
        self,
        q_idx: int,
        o_matrix: str,
        seq_len: int,
        head_dim: int,
        rows: int | None = None,
        reset_state: bool = True,
        reset_output: bool = True,
    ) -> str:
        """
        Initialize Online Softmax state for Q block q_idx:
          m_old = -inf (FP SRAM), l = 0 (FP SRAM), O_row = 0 (VRAM).
        """
        fp_sram_start = self._ONLINE_SOFTMAX_FPSRAM_BASE
        m_old_addr = fp_sram_start
        l_addr = fp_sram_start + 2 * self.mlen  # skip m_res region

        o_info = self[o_matrix]
        o_vram_addr = o_info.vram_addr
        physical_seq_len = o_info.physical_shape[0]
        row_offset = q_idx * self.mlen

        isa_code = f"; === Init Online Softmax for Q block {q_idx} ===\n"

        if reset_state:
            isa_code += self._reset_fpsram_asm(m_old_addr, self.mlen, 2)  # slot 2 = -inf
            isa_code += self._reset_fpsram_asm(l_addr, self.mlen, 0)  # slot 0 = 0.0
        if reset_output:
            isa_code += self._reset_vram_asm(
                start_address=o_vram_addr,
                rows=self.mlen if rows is None else rows,
                cols=head_dim,
                total_rows=physical_seq_len,
                mlen=self.mlen,
                row_offset=row_offset,
            )

        return self._emit(isa_code)

    def online_softmax_first_block(
        self,
        s_block_matrix: str,
        scale: float,
        rows: int | None = None,
        valid_cols: int | None = None,
        state_head: int = 0,
        last_block: bool = False,
        stream_state: bool = False,
    ) -> str:
        """Create running online-softmax state from the first K block."""

        s_info = self[s_block_matrix]
        isa_code = f"; === Online Softmax First Block {s_block_matrix} ===\n"
        if (
            getattr(self, "softmax_state_schedule", "sram-v1")
            == "row-bank-simd-v3"
        ):
            layout = self._softmax_state_layout
            isa_code += self._online_softmax_row_bank_asm(
                mlen=self.mlen,
                s_address=s_info.vram_addr,
                state_address=layout.m_base(state_head),
                output_address=None,
                scale=scale,
                rows=self.mlen if rows is None else rows,
                valid_cols=valid_cols,
                first_block=True,
            )
            return self._emit(isa_code)
        if (
            getattr(self, "softmax_state_schedule", "sram-v1") == "streamed-v2"
            and stream_state
        ):
            layout = self._softmax_state_layout
            isa_code += self._online_softmax_first_block_streamed_asm(
                mlen=self.mlen,
                s_address=s_info.vram_addr,
                m_address=layout.m_base(state_head),
                l_address=layout.l_base(state_head),
                scale=scale,
                rows=rows,
                valid_cols=valid_cols,
                last_block=last_block,
            )
            return self._emit(isa_code)
        isa_code += self._online_softmax_first_block_asm(
            mlen=self.mlen,
            s_address=s_info.vram_addr,
            m_start_address=self._ONLINE_SOFTMAX_FPSRAM_BASE,
            scale=scale,
            rows=rows,
            valid_cols=valid_cols,
        )
        return self._emit(isa_code)

    def online_softmax_block(
        self,
        s_block_matrix: str,
        scale: float,
        rows: int | None = None,
        valid_cols: int | None = None,
        state_head: int = 0,
        output_address: int | None = None,
        output_head_slot: int | None = None,
        last_block: bool = False,
    ) -> str:
        """
        Run Online Softmax on one S block.
          Input:   S_block (mlen × mlen) in VRAM
          Output:  P (mlen × mlen) in-place in VRAM
          Updates: m_old, m_res, l in FP SRAM
          ``scale`` is the QK^T scaling factor (typically 1/sqrt(d)).
        """
        s_info = self[s_block_matrix]
        s_address = s_info.vram_addr
        if (
            getattr(self, "softmax_state_schedule", "sram-v1")
            == "row-bank-simd-v3"
        ):
            if output_address is None:
                raise ValueError("row-bank recurrent softmax requires output_address")
            layout = self._softmax_state_layout
            lane_address = output_address + int(output_head_slot or 0) * int(
                getattr(self, "hlen", self.mlen)
            )
            isa_code = f"; === Online Softmax Block {s_block_matrix} ===\n"
            isa_code += self._online_softmax_row_bank_asm(
                mlen=self.mlen,
                s_address=s_address,
                state_address=layout.m_base(state_head),
                output_address=lane_address,
                scale=scale,
                rows=self.mlen if rows is None else rows,
                valid_cols=valid_cols,
                first_block=False,
            )
            return self._emit(isa_code)
        if (
            getattr(self, "softmax_state_schedule", "sram-v1") == "streamed-v2"
            and output_address is not None
            and output_head_slot is not None
        ):
            layout = self._softmax_state_layout
            isa_code = f"; === Online Softmax Block {s_block_matrix} ===\n"
            isa_code += self._online_softmax_streamed_asm(
                mlen=self.mlen,
                s_address=s_address,
                m_address=layout.m_base(state_head),
                l_address=layout.l_base(state_head),
                output_address=output_address,
                output_head_slot=output_head_slot,
                scale=scale,
                rows=rows,
                valid_cols=valid_cols,
                last_block=last_block,
            )
            return self._emit(isa_code)

        fp_sram_start = self._ONLINE_SOFTMAX_FPSRAM_BASE
        m_start_address = fp_sram_start

        isa_code = f"; === Online Softmax Block {s_block_matrix} ===\n"
        isa_code += self._online_softmax_asm(
            mlen=self.mlen,
            s_address=s_address,
            m_start_address=m_start_address,
            scale=scale,
            rows=rows,
            valid_cols=valid_cols,
        )

        return self._emit(isa_code)

    def compute_pv(
        self,
        s_block_matrix: str,
        v_sub_matrix: str,
        k_idx: int,
        pv_matrix: str,
        head_dim: int,
        rows: int | None = None,
    ) -> str:
        """
        Compute PV = P @ V[k_idx].

        P lives in s_block_matrix (softmax result); V is prefetched from
        HBM; PV is written to VRAM via pv_matrix.
        """
        s_info = self[s_block_matrix]
        p_address = s_info.vram_addr

        pv_info = self[pv_matrix]
        pv_address = pv_info.vram_addr

        v_layout = self.get_hbm_layout(v_sub_matrix)
        physical_head_dim = (v_layout.physical_shape or v_layout.full_shape)[1]
        v_hbm_offset = k_idx * self.mlen * physical_head_dim

        isa_code = f"; === Compute PV = P @ V[k_idx={k_idx}] ===\n"

        addr_regs = self.register_allocator.allocate_addr(1)
        v_hbm_reg = addr_regs[0]
        gp_regs = self.register_allocator.allocate_gp(2)

        from compiler.asm_templates import preload_addr_reg_asm

        isa_code += preload_addr_reg_asm(
            addr_reg_to_set=[v_hbm_reg], available_registers=gp_regs, addr_reg_val=[v_layout.hbm_base_addr]
        )

        isa_code += self._pv_multiply_asm(
            mlen=self.mlen,
            blen=self.blen,
            head_dim=head_dim,
            p_address=p_address,
            v_hbm_offset_reg=v_hbm_reg,
            v_hbm_offset=v_hbm_offset,
            pv_address=pv_address,
            rows=rows,
            pv_physical_rows=pv_info.physical_shape[0],
        )

        self.register_allocator.free_gp(gp_regs)
        self.register_allocator.free_addr(addr_regs)

        return self._emit(isa_code)

    def scale_o_row(
        self,
        o_matrix: str,
        q_idx: int,
        seq_len: int,
        head_dim: int,
        rows: int | None = None,
    ) -> str:
        """Scale the current row block of O by m_res: O[q_idx] *= m_res."""
        o_info = self[o_matrix]
        o_address = o_info.vram_addr
        physical_seq_len = o_info.physical_shape[0]

        fp_sram_start = self._ONLINE_SOFTMAX_FPSRAM_BASE
        m_res_addr = fp_sram_start + self.mlen

        row_offset = q_idx * self.mlen

        isa_code = f"; === Scale O[q_idx={q_idx}] by m_res ===\n"
        isa_code += self._scale_o_asm(
            mlen=self.mlen,
            head_dim=head_dim,
            seq_len=physical_seq_len,
            m_res_address=m_res_addr,
            o_address=o_address,
            row_offset=row_offset,
            rows=rows,
        )

        return self._emit(isa_code)

    def final_scale_o(
        self,
        q_idx: int,
        o_matrix: str,
        seq_len: int,
        head_dim: int,
        rows: int | None = None,
    ) -> str:
        """Final scaling: O[q_idx] /= l."""
        o_info = self[o_matrix]
        o_address = o_info.vram_addr
        physical_seq_len = o_info.physical_shape[0]

        fp_sram_start = self._ONLINE_SOFTMAX_FPSRAM_BASE
        l_addr = fp_sram_start + 2 * self.mlen

        row_offset = q_idx * self.mlen

        isa_code = f"; === Final Scale O for Q block {q_idx} ===\n"
        isa_code += self._final_scaling_asm(
            mlen=self.mlen,
            head_dim=head_dim,
            seq_len=physical_seq_len,
            l_address=l_addr,
            o_address=o_address,
            row_offset=row_offset,
            rows=rows,
        )

        return self._emit(isa_code)


__all__ = ["IsaAttentionMixin"]
