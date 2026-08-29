"""KDA's three short causal convolutions, one decode step at a time.

The reference (``aten/models/kda/reference.py::_causal_conv_step``) is::

    updated      = roll(state, -1, dim=-1)      # drop the oldest tap
    updated[..., -1] = value                    # append this token
    output       = sum_t updated[..., t] * weight[..., t]
    output       = silu(output + bias)

It is **depthwise**: every channel carries its own ``kernel``-long tap vector.
So the tap weight varies per *lane*, which is why ``V_MUL_VF`` is the wrong tool
(it broadcasts one FP register across every lane) and the weight lives in VRAM
as a matrix for ``V_MUL_VV``. ``mamba_conv1d_v0`` makes the same argument at
``program_mamba_common.py:481-484``; this module is the same kernel with two
differences that make it its own emitter rather than a call into that one:

* Mamba convolves a *sequence* held in ``x`` with the taps at row offsets.
  KDA decode has one new token and a **carried** conv state, so the shift is a
  real row copy every step rather than a free change of offset.
* KDA's channel counts are ``num_heads * key_dim`` and ``num_heads * value_dim``
  -- 12,288 apiece for Kimi K3 -- so everything is blocked by ``mlen``, and the
  block index has to reach the inner loop. ``mamba_conv1d_v0`` addresses rows
  from 0 and has nowhere to put it.

Adding a base-row argument to ``mamba_conv1d_v0`` would have avoided the
duplication, but that emitter is on Mamba's decode and prefill paths and this
work has no business changing their signatures. The inner loop below is
deliberately the same shape so the two stay legible together.

Layout, all tiles exactly ``mlen`` wide:

===============  =====================================  ==========================
tile             shape                                  row index
===============  =====================================  ==========================
``conv_state``   ``[channel_blocks * kernel, mlen]``    ``kda_conv_state_row``
``weight``       ``[channel_blocks * kernel, mlen]``    same
``bias``         ``[channel_blocks, mlen]``             channel block
``x_new``        ``[channel_blocks, mlen]``             channel block
``out``          ``[channel_blocks, mlen]``             channel block
===============  =====================================  ==========================

Taps run oldest (0) to newest (``kernel - 1``), matching the reference's
``roll(-1)`` then write-at-``[-1]``.
"""

from __future__ import annotations

from compiler.aten.plena.program_kda_common import (
    kda_conv_blocks,
    kda_conv_state_row,
    kda_stage_marker,
)
from compiler.aten.plena.vars import VRAMMatrixVar

__all__ = ["ProgramKdaConvMixin"]


class ProgramKdaConvMixin:
    """One causal-conv decode step, blocked by channel.

    Requires ``ProgramMambaCommonMixin`` (``mamba_row_copy``, ``mamba_row_mul``,
    ``mamba_row_add``, ``mamba_silu_v0``) and ``ProgramFPTileOpsMixin``
    (``vram_fill_zero``).
    """

    def kda_conv_step_v0(
        self,
        *,
        x_new: VRAMMatrixVar,
        conv_state: VRAMMatrixVar,
        weight: VRAMMatrixVar,
        bias: VRAMMatrixVar | None,
        out: VRAMMatrixVar,
        scratch: VRAMMatrixVar,
        consts,
        channels: int,
        kernel: int,
        apply_silu: bool = True,
        x_new_row_base: int = 0,
        x_new_row_stride: int = 1,
    ) -> VRAMMatrixVar:
        """Advance the history by one token and convolve.

        ``conv_state`` is updated in place: this is the carried state, and the
        next token reads what this call leaves behind.

        ``x_new_row_base`` and ``x_new_row_stride`` say where block ``cb`` of
        this token lives in ``x_new``: at row ``base + cb * stride``. The
        default 0/1 is a dense tile, one feature block per row.

        A stride exists because the projection cannot produce a dense tile.
        ``M_MM_WO`` writes a ``blen x blen`` sub-tile and the writeback loops
        cover ``mlen / blen`` column groups, so the smallest thing a projection
        can lay down is ``blen`` token-rows by ``mlen`` lanes -- column block
        ``c`` lands at row ``c * blen`` whatever the weights look like. The
        convolution wants one token's blocks as consecutive rows. The mismatch
        is exactly ``blen`` and it is a property of the matrix writeback, not
        of how the weight matrices are packed, so no rearrangement of the
        projection removes it.

        Reading across it costs nothing. ``mamba_row_copy`` takes a row index,
        and the sweeps underneath collapse any arithmetic progression into one
        hardware loop where the step is an ``S_ADDI_INT`` immediate -- a step
        of ``blen`` is the same instruction as a step of 1. So passing the
        projection tile with ``x_new_row_stride=blen`` does what a gather into
        a dense tile does, for nothing.
        """
        blocks = kda_conv_blocks(channels, self.mlen)
        if kernel < 1:
            raise ValueError(f"kernel must be at least 1, got {kernel}")
        for name, tile in (
            ("x_new", x_new), ("conv_state", conv_state), ("weight", weight),
            ("out", out), ("scratch", scratch),
        ):
            if tile.shape[1] != self.mlen:
                raise ValueError(
                    f"{name} is {tile.shape[1]} columns wide; every KDA tile must be "
                    f"exactly mlen ({self.mlen})"
                )
        needed = blocks * kernel
        if conv_state.shape[0] < needed:
            raise ValueError(
                f"conv_state needs {needed} rows for {blocks} blocks x kernel {kernel}, "
                f"has {conv_state.shape[0]}"
            )
        if weight.shape[0] < needed:
            raise ValueError(f"weight needs {needed} rows, has {weight.shape[0]}")
        if x_new_row_stride < 1:
            raise ValueError(f"x_new_row_stride must be at least 1, got {x_new_row_stride}")
        if x_new_row_base < 0:
            raise ValueError(f"x_new_row_base must not be negative, got {x_new_row_base}")
        x_new_last = x_new_row_base + (blocks - 1) * x_new_row_stride
        if out.shape[0] < blocks:
            raise ValueError(f"out needs {blocks} rows, has {out.shape[0]}")
        if x_new.shape[0] <= x_new_last:
            raise ValueError(
                f"x_new needs row {x_new_last} for block {blocks - 1} "
                f"(base {x_new_row_base}, stride {x_new_row_stride}), "
                f"has {x_new.shape[0]} rows"
            )
        aliased = {conv_state.name, out.name, x_new.name, weight.name}
        if bias is not None:
            aliased.add(bias.name)
        if scratch.name in aliased:
            # mamba_row_copy is zero-then-add, so a scratch aliased onto any
            # operand wipes it mid-loop.
            raise ValueError(f"scratch must be distinct from {sorted(aliased)}")

        self.emit_comment(
            kda_stage_marker("kda_conv1d", f"channels={channels} k={kernel} blocks={blocks}")
        )
        for cb in range(blocks):
            # Shift the history one tap older, then append this token. A real
            # copy, not a ring pointer: address immediates are baked at ASM-gen
            # time and there is no data-dependent addressing.
            for tap in range(kernel - 1):
                self.mamba_row_copy(
                    conv_state,
                    kda_conv_state_row(channels, self.mlen, kernel, cb, tap),
                    conv_state,
                    kda_conv_state_row(channels, self.mlen, kernel, cb, tap + 1),
                )
            self.mamba_row_copy(
                conv_state,
                kda_conv_state_row(channels, self.mlen, kernel, cb, kernel - 1),
                x_new,
                x_new_row_base + cb * x_new_row_stride,
            )

            # out[cb] = sum_t state[cb, t] * weight[cb, t]
            self.vram_fill_zero(out, rows=[cb])
            for tap in range(kernel):
                row = kda_conv_state_row(channels, self.mlen, kernel, cb, tap)
                self.mamba_row_copy(scratch, 0, conv_state, row)
                self.mamba_row_mul(scratch, 0, weight, row)
                self.mamba_row_add(out, cb, scratch, 0)
            if bias is not None:
                self.mamba_row_add(out, cb, bias, cb)

        if apply_silu:
            # `marker` is required, and this is why: mamba_silu_v0 serves both
            # stage vocabularies, and a Mamba default would bill KDA's SiLU --
            # and, because markers are sticky, everything emitted after it -- to
            # a Mamba stage.
            self.mamba_silu_v0(
                out,
                scratch,
                consts,
                rows=list(range(blocks)),
                marker=kda_stage_marker("kda_conv1d", "silu"),
            )
        return out
