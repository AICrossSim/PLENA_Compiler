"""Mamba-2 recurrent decode step (`seq_len == 1`) and cross-token state residency.

At `seq_len == 1` the chunked scan collapses to the plain recurrence

    h[h, n, :] = dA_h * h[h, n, :] + B[g, n] * (dt_h * x[h, :])
    y[h, :]    = sum_n C[g, n] * h[h, n, :] + D_h * x[h, :]

which is pure elementwise work on the state plus two contractions. No matmul, no
chunking, no decay matrix.

State layout -- the decision everything else follows from
--------------------------------------------------------
The state is laid out ``[head][state][head_dim]``: one VRAM row per
``(head, state index)`` pair, with **head_dim on the lane axis**. Under that
layout all three terms become existing opcodes:

* the decay ``dA_h`` is constant across a head's ``state_size`` consecutive rows,
  so it is a ``V_MUL_VF`` broadcast whose FPRAM row map is an arithmetic
  progression with step 0 -- one ``S_LD_FP`` per head, not per row;
* the input term is rank-1, but folding ``dt_h`` into ``x`` *first* reduces the
  distinct per-row scalars from ``num_heads*head_dim*state_size`` to just
  ``n_groups*state_size``, which is the difference between fitting in FPRAM and
  not. It then becomes ``V_MUL_VF`` + ``V_ADD_VV`` per row;
* the ``sum_n`` contraction becomes a **cross-row** accumulation (``V_ADD_VV``
  into one output row per head) instead of an intra-row reduction. That matters:
  the transposed layout would need ``V_RED_SUM`` per ``(head, head_dim)`` pair,
  producing ``num_heads*head_dim`` scalars that overflow FPRAM.

This layout requires ``VLEN <= head_dim``, which :meth:`Mamba2Shape.validate`
already enforces (as ``head_dim % mlen == 0``).

What this path does *not* do
----------------------------
It does not batch the per-head work through ``M_BMV``. That instruction's
broadcast runs the wrong way: it reads one VRAM row as ``broadcast_amount`` lane
groups and multiplies **every** group by one **shared** MRAM slice. Mamba-2
decode needs the opposite -- a per-head state *matrix* contracted against a
group-shared ``C`` vector. Two further disqualifiers: the state is an activation
and MRAM is writable only by ``H_PREFETCH_M``, so any ``M_*`` form on ``h`` costs
a full HBM round trip per token; and every ``M_B*`` form silently multiplies its
result by a hardwired ``bmm_scale = 0.25`` that no opcode exposes.
"""

from __future__ import annotations

from collections.abc import Sequence

from compiler.aten.plena.program_mamba_common import (
    Mamba2Shape,
    MambaFPConstants,
    mamba_stage_marker,
)
from compiler.aten.plena.vars import FPVar, InputVar, VRAMMatrixVar


class ProgramSSMRecurrentMixin:
    """Mamba-2 single-token decode emitters, plus cross-token state residency."""

    # ========================================================================
    # Cross-token persistent state
    # ========================================================================

    def pin_hbm_region(self, name: str, size: int, hbm_element_bytes: int = 2) -> int:
        """Reserve an HBM range that ``_allocate_hbm`` will never hand out.

        This is the first persistent-state mechanism in the compiler: attention
        takes K/V as ordinary per-block ``InputVar``s and nothing in the repo
        survives across program regions (a repo-wide grep for ``kv_cache`` /
        ``past_key_value`` / ``persistent`` returns no functional hits). Mamba
        decode needs it because the SSM state and the conv state must be read at
        the top of a step and written back at the bottom, at the *same* address,
        for the next token to find.

        Returns the pinned base address. Implemented by bumping the auto-allocator
        past the region rather than by adding a second allocator, so a later
        ``input()``/``store()`` with no explicit address cannot collide with it.

        `hbm_element_bytes` defaults to 2 because that is what
        :meth:`ssm_store_state_v0` writes; sizing the region with the MX layout
        instead would under-reserve it by ~78% and the write-back would land on the
        next tensor.
        """
        if size <= 0:
            raise ValueError(f"pinned HBM region {name!r} must have positive size, got {size}")
        addr = self._allocate_hbm(self.hbm_tensor_size(size, hbm_element_bytes=hbm_element_bytes))
        if not hasattr(self, "_pinned_hbm_regions"):
            self._pinned_hbm_regions: dict[str, tuple[int, int]] = {}
        self._pinned_hbm_regions[name] = (addr, size)
        self.emit_comment(f"pinned HBM region {name}: [{addr}, {addr + size}) elements")
        return addr

    def pinned_hbm_region(self, name: str) -> tuple[int, int]:
        regions = getattr(self, "_pinned_hbm_regions", {})
        if name not in regions:
            raise KeyError(f"no pinned HBM region named {name!r}; call pin_hbm_region first")
        return regions[name]

    def ssm_load_state_v0(
        self,
        name: str,
        rows: int,
        cols: int,
        hbm_addr: int,
    ) -> VRAMMatrixVar:
        """Prefetch a pinned state region from HBM into VRAM.

        Declared with ``real_data_ratio=1.0`` and loaded through the BF16 path:
        the SSM state is a multiplicative accumulator carried across the whole
        sequence, and MX-FP8's 3 mantissa bits would compound. Quantisation error
        in a decaying accumulator is amplified by ``1 / sqrt(1 - lambda^2)`` where
        ``lambda = exp(A*dt)``, so at ``lambda = 0.99`` a 3.6% element error
        becomes roughly 25% state error -- and ``lambda -> 1`` is exactly the
        long-memory regime that justifies using an SSM at all. A KV cache has no
        such amplification because it is written once and never read-modify-written.
        """
        self.emit_comment(mamba_stage_marker("mamba_state_load", f"{name} [{rows},{cols}]"))
        var = self.input(name, (rows, cols), hbm_addr=hbm_addr, real_data_ratio=1.0)
        # Must mirror ssm_store_state_v0 exactly: same precision class (KeyValue)
        # and same bytes per element (2). Until load_batch took these, it emitted
        # Activation/1-byte with half the row stride and a different scale-section
        # base, so the state read back was not the state written -- and the failure
        # mode is a wrong answer, not an error.
        return self.load_batch(
            var, name=f"{name}_vram", storage_precision=2, precision=1
        )

    def ssm_store_state_v0(self, state: VRAMMatrixVar, name: str, hbm_addr: int) -> InputVar:
        """Write the updated state back over the same pinned HBM range."""
        self.emit_comment(mamba_stage_marker("mamba_state_store", name))
        return self.store(
            state,
            name=name,
            hbm_addr=hbm_addr,
            precision=1,
            hbm_element_bytes=2,
            real_data_ratio=1.0,
        )

    # ========================================================================
    # Decode step
    # ========================================================================

    def ssm_decode_step_v0(
        self,
        *,
        state: VRAMMatrixVar,
        x: VRAMMatrixVar,
        b_fp: FPVar,
        c_fp: FPVar,
        da_fp: FPVar,
        dt_fp: FPVar,
        d_fp: FPVar,
        y: VRAMMatrixVar,
        scratch: VRAMMatrixVar,
        shape: Mamba2Shape,
        consts: MambaFPConstants,
        head_rows: Sequence[int] | None = None,
    ):
        """One recurrent step, all heads.

        Arguments carry the per-head / per-(group, state) scalars in FPRAM:
        `da_fp[h]` is ``exp(A_h * dt_h)``, `dt_fp[h]` is ``dt_h``, `d_fp[h]` is the
        skip coefficient ``D_h``, and `b_fp` / `c_fp` hold ``B[g, n]`` / ``C[g, n]``
        laid out ``n``-major within each group. They get there via ``S_MAP_FP_V``
        (see :meth:`ssm_decode_scalars_to_fpram_v0`).

        `state` is ``[num_heads * state_size, head_dim]`` in the layout described
        in the module docstring; `x` is ``[num_heads, head_dim]``; `y` is
        ``[num_heads, head_dim]``.
        """
        heads = list(range(shape.num_heads)) if head_rows is None else list(head_rows)
        n_state = shape.state_size
        self.emit_comment(
            mamba_stage_marker("mamba_state_update", f"decode heads={len(heads)} state={n_state}")
        )

        for h in heads:
            group = h // shape.heads_per_group
            base = h * n_state

            # xs = dt_h * x[h] -- fold dt into x once per head so the per-row
            # scalars below are only B[g, n], not dt_h * B[g, n] * ...
            self.mamba_row_copy(scratch, 0, x, h)
            self.tile_row_mul_fp_broadcast(scratch, dt_fp, rows=[0], fpram_offset=h)

            # h[h, n, :] = dA_h * h[h, n, :] + B[g, n] * xs
            #
            # The rank-1 update walks the state rows against a pinned scratch
            # row 0, so both walks are progressions and the whole thing is one
            # hardware loop. The copy/multiply/add triple it replaces could not
            # loop: it needed a scratch row per step, and that row was the same
            # row every iteration, which broke the progression.
            state_rows = list(range(base, base + n_state))
            self.tile_row_mul_fp_broadcast(state, da_fp, rows=state_rows, fpram_offset=h)
            self.tile_row_fma_fp_sweep(
                state, scratch, b_fp,
                dst_rows=state_rows, src_rows=[0] * n_state,
                fpram_offset=group * n_state,
            )

            # y[h] = sum_n C[g, n] * h[h, n, :] -- the mirror: source walks,
            # destination pinned.
            self.vram_fill_zero(y, rows=[h])
            self.tile_row_fma_fp_sweep(
                y, state, c_fp,
                dst_rows=[h] * n_state, src_rows=state_rows,
                fpram_offset=group * n_state,
            )

            # y[h] += D_h * x[h] -- one row, one slot, so a broadcast FMA. This
            # is where the accumulate saves a whole scratch copy on its own.
            self.emit_comment(mamba_stage_marker("mamba_skip", f"head={h}"))
            self.tile_row_fma_fp_broadcast(
                y, x, d_fp, dst_rows=[h], src_rows=[h], fpram_offset=h
            )

        return y

    def ssm_decode_scalars_to_fpram_v0(
        self,
        *,
        source: VRAMMatrixVar,
        target: FPVar,
        rows: Sequence[int],
        stage: str,
    ):
        """Move computed per-head / per-state scalars from VRAM lanes into FPRAM.

        One ``S_MAP_FP_V`` per row, MLEN scalars each. Every value the decode step
        broadcasts (``dA_h``, ``dt_h``, ``B[g,n]``, ``C[g,n]``) is produced on chip
        by the in_proj and lands in a VRAM row, but ``V_MUL_VF`` can only broadcast
        from an FP register -- and before ``S_MAP_FP_V`` the only VRAM-to-scalar
        path was ``V_RED_SUM``/``V_RED_MAX``, which collapse the whole row. The
        one-hot workaround costs 3 instructions per scalar; this costs 1 per row.
        """
        self.emit_comment(mamba_stage_marker(stage, "vram lanes -> FPRAM"))
        self.tile_row_to_fpram(source, target, rows=list(rows))
        return target

    def ssm_conv_state_roll_v0(
        self,
        conv_state: VRAMMatrixVar,
        new_row_src: VRAMMatrixVar,
        new_row_idx: int,
        shape: Mamba2Shape,
    ):
        """Shift the rolling conv1d history by one timestep and append the new one.

        `conv_state` holds ``conv_kernel - 1`` rows of history, oldest first. The
        shift is a physical copy: address immediates are baked at ASM-gen time and
        there is no data-dependent addressing (no compare, no branch), so a
        runtime ring pointer is not expressible. It is cheap -- ``conv_kernel - 1``
        row copies per token.

        ``V_SHFT_V`` is the wrong tool twice over: it shifts *lanes* within one
        row (the channel axis here), not rows, and its direction is contradictory
        between ``doc/plena_isa_spec.md`` (left) and the emulator's
        ``vector_machine.rs`` (right).
        """
        history = shape.conv_kernel - 1
        if history <= 0:
            return conv_state
        if conv_state.shape[0] < history:
            raise ValueError(f"conv_state needs {history} rows, has {conv_state.shape[0]}")
        self.emit_comment(mamba_stage_marker("mamba_conv1d", f"roll history={history}"))
        for i in range(history - 1):
            self.mamba_row_copy(conv_state, i, conv_state, i + 1)
        self.mamba_row_copy(conv_state, history - 1, new_row_src, new_row_idx)
        return conv_state


__all__ = ["ProgramSSMRecurrentMixin"]
