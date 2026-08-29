"""The KDA recurrent decode step: two sweeps over the transposed state tile.

The maths, per head, with state held transposed as ``T[key, value]`` (see
``program_kda_common``)::

    T[k, :]  = decay[k] * T[k, :]                    for each k
    pred[:]  = sum_k k_hat[k] * T[k, :]
    err[:]   = beta * (v[:] - pred[:])
    T[k, :] += err[:] * k_hat[k]                     for each k
    out[:]   = output_scale * sum_k q_hat[k] * T[k, :]

Sweep 1 applies the per-key decay and accumulates the prediction. The error is
then a value-length vector. Sweep 2 applies the rank-1 update and reads the
output out of the *updated* state -- the reference does
``updated = state + err*k; state = updated; reduced += updated*q`` inside one
loop, so update and read-out cannot be reordered into separate passes over the
state. Updating every key row and then reducing over all of them *is* legal,
because each key's update touches only that key's row; that equivalence is what
lets both be single row sweeps.

State is read twice and written once. That is inherent to the delta rule, not an
artefact of this lowering.

Fused form
----------
Each sweep is one ``V_FMA_VF`` inside one hardware loop. This module was first
written with only instructions that already existed -- the ``copy + multiply +
add`` triple, via ``mamba_row_copy`` / ``tile_row_mul_fp`` / ``mamba_row_add``
-- to establish that the static ISA expresses KDA with no new opcode. It does,
and that is now demonstrated rather than claimed, so the triple has been
retired: it staged every row through a scratch row, and because that was the
*same* row each iteration the destination never formed a progression, which
forced the whole sweep to unroll. Static cost went from 7,345 instructions at
``key_dim`` 128 to 76, and stopped depending on ``key_dim`` at all.

Column blocks
-------------
``value_dim`` may exceed ``mlen``; the block index is folded into the row
(``program_kda_common``'s module docstring explains why). Every tile here is
therefore exactly one column block wide, and the recurrence runs once per
block. The two sweeps are per ``(head, block)``, and within one the keys sit at
consecutive rows -- so each stays a single hardware loop.

Blocks are independent for the decay, update and read-out, because those are
elementwise along ``value``. The prediction is not: it contracts over ``key``
for a fixed ``value`` lane, so block ``c`` of ``pred`` only ever sees block
``c`` of the state. That is why the whole recurrence nests block outside key
rather than the other way round.
"""

from __future__ import annotations

from typing import Sequence

from compiler.aten.models.kda.shape import KdaShape
from compiler.aten.plena.program_kda_common import (
    kda_blocks,
    kda_stage_marker,
    kda_state_row,
    kda_vector_row,
)
from compiler.aten.plena.vars import FPVar, VRAMMatrixVar

__all__ = ["ProgramKdaRecurrentMixin"]


class ProgramKdaRecurrentMixin:
    """The KDA decode recurrence.

    Requires ``ProgramKdaCommonMixin`` (stage markers, layout helpers),
    ``ProgramMambaCommonMixin`` (``mamba_row_copy``, ``mamba_row_add``) and
    ``ProgramFPTileOpsMixin`` (``tile_row_mul_fp``, ``vram_fill_zero``).
    """

    def kda_decode_step_v0(
        self,
        *,
        state: VRAMMatrixVar,
        q_fp: FPVar,
        k_fp: FPVar,
        decay_fp: FPVar,
        beta_fp: FPVar,
        v: VRAMMatrixVar,
        o: VRAMMatrixVar,
        pred: VRAMMatrixVar,
        err: VRAMMatrixVar,
        shape: KdaShape,
        output_scale_fp: FPVar,
        head_rows: Sequence[int] | None = None,
        fp_head_stride: int | None = None,
    ) -> VRAMMatrixVar:
        """One recurrent KDA token, all heads.

        ``q_fp`` / ``k_fp`` / ``decay_fp`` are FPRAM arrays holding the
        already-normalised ``q_hat[h, k]``, ``k_hat[h, k]`` and ``decay[h, k]``,
        laid out ``k``-major within each head -- so head ``h``'s scalars start at
        offset ``h * key_dim``. ``beta_fp[h]`` and ``output_scale_fp[0]`` are
        single scalars. They get to FPRAM through ``S_MAP_FP_V``; see
        ``ssm_decode_scalars_to_fpram_v0``.

        ``state`` is ``[num_heads * blocks * key_dim, mlen]``, indexed by
        :func:`kda_state_row`. ``v``, ``o``, ``pred`` and ``err`` are
        ``[num_heads * blocks, mlen]``, indexed by :func:`kda_vector_row`.
        Every tile is exactly one column block wide -- the block index lives in
        the row. There is no scratch tile: the FMA accumulates in place, which
        is most of what it bought. Before Task 8 each sweep staged a row through
        scratch, and because that was the *same* row every iteration the
        destination never formed a progression, so the sweep could not become a
        hardware loop.

        A convenience wrapper over :meth:`kda_decode_predict_v0` and
        :meth:`kda_decode_update_v0`. Call those two directly when ``q_fp`` and
        ``decay_fp`` must share one FPRAM window; see the predict half.
        """
        heads, stride = self._kda_decode_setup(
            shape=shape, tiles=(("state", state), ("v", v), ("o", o),
                                ("pred", pred), ("err", err)),
            state=state, pred=pred, err=err,
            head_rows=head_rows, fp_head_stride=fp_head_stride,
        )
        self.kda_decode_predict_v0(
            state=state, k_fp=k_fp, decay_fp=decay_fp,
            beta_fp=beta_fp, v=v, pred=pred, err=err, shape=shape,
            head_rows=heads, fp_head_stride=stride,
        )
        self.kda_decode_update_v0(
            state=state, k_fp=k_fp, q_fp=q_fp, o=o, err=err,
            shape=shape, output_scale_fp=output_scale_fp,
            head_rows=heads, fp_head_stride=stride,
        )
        return o

    def _kda_decode_setup(
        self, *, shape, tiles, state, pred, err, head_rows, fp_head_stride
    ):
        """Shared validation for both halves; returns (heads, stride)."""
        for name, tile in tiles:
            if tile.shape[1] != self.mlen:
                raise ValueError(
                    f"{name} is {tile.shape[1]} columns wide; every KDA tile must be "
                    f"exactly mlen ({self.mlen}), with the column block folded into "
                    f"the row index -- see program_kda_common's module docstring"
                )
        for name, tile in (("pred", pred), ("err", err)):
            if tile is not None and tile.name == state.name:
                raise ValueError(f"{name} must not alias state")

        heads = list(range(shape.num_heads)) if head_rows is None else list(head_rows)
        # How far apart consecutive heads' scalars sit in FPRAM. The default,
        # key_dim, is the layout where every head's decay / q_hat / k_hat are
        # materialised at once. Pass 0 when the caller streams one head at a
        # time into a reused window -- FPRAM is 512 slots and the all-at-once
        # layout wants 3 * num_heads * key_dim, which is 36,960 for Kimi K3.
        stride = shape.key_dim if fp_head_stride is None else fp_head_stride
        if stride < 0:
            raise ValueError(f"fp_head_stride must not be negative, got {stride}")
        if stride == 0 and len(heads) > 1:
            raise ValueError(
                "fp_head_stride=0 means every head reads the same FPRAM window, "
                f"so only one head may be lowered per call; got {len(heads)}"
            )
        return heads, stride

    def kda_decode_predict_v0(
        self,
        *,
        state: VRAMMatrixVar,
        k_fp: FPVar,
        decay_fp: FPVar,
        beta_fp: FPVar,
        v: VRAMMatrixVar,
        pred: VRAMMatrixVar,
        err: VRAMMatrixVar,
        shape: KdaShape,
        head_rows: Sequence[int] | None = None,
        fp_head_stride: int | None = None,
    ) -> VRAMMatrixVar:
        """First half: decay the state, predict, and form the error.

        Reads ``decay_fp`` and ``k_fp``; never touches ``q_fp``. Leaves ``err``
        holding ``beta * (v - pred)`` and ``state`` holding the decayed state.

        Split from the second half so a caller that is short of FPRAM can let
        ``q_fp`` and ``decay_fp`` be the *same* window: decay is dead once every
        block of this head has been predicted, and ``q_hat`` is not read until
        the read-out. That halves the per-head window from ``3 * key_dim`` to
        ``2 * key_dim``, which is what makes Kimi K3 fit in 512 slots at all.
        """
        heads, stride = self._kda_decode_setup(
            shape=shape, tiles=(("state", state), ("v", v), ("pred", pred),
                                ("err", err)),
            state=state, pred=pred, err=err,
            head_rows=head_rows, fp_head_stride=fp_head_stride,
        )
        blocks = kda_blocks(shape, self.mlen)
        k_dim = shape.key_dim

        for h in heads:
            fp_base = h * stride
            for c in range(blocks):
                first = kda_state_row(shape, self.mlen, h, c, 0)
                key_rows = list(range(first, first + k_dim))
                acc = kda_vector_row(shape, self.mlen, h, c)

                # decay is per (head, key), so this walks a different FPRAM
                # slot per row -- the tile_row_mul_fp family, not _broadcast.
                self.emit_comment(kda_stage_marker("kda_decay", f"head={h} block={c}"))
                self._kda_scale_rows(state, decay_fp, key_rows, fp_base)

                self.emit_comment(
                    kda_stage_marker("kda_state_update", f"predict head={h} block={c}")
                )
                # pred[acc] += state[row] * k_hat[i], contracting over key.
                # The destination is pinned and the source walks, so both are
                # progressions and the whole contraction is one hardware loop.
                self.vram_fill_zero(pred, rows=[acc])
                self.tile_row_fma_fp_sweep(
                    pred, state, k_fp,
                    dst_rows=[acc] * len(key_rows), src_rows=key_rows,
                    fpram_offset=fp_base,
                )

                # err = beta_h * (v - pred), per block. beta_fp is one slot per
                # head and produced for every head at once, so it is indexed by
                # head number and not through fp_base.
                self.emit_comment(
                    kda_stage_marker("kda_state_update", f"error head={h} block={c}")
                )
                self.mamba_row_copy(err, acc, v, acc)
                self.tile_row_sub(err, pred, rows=[acc])
                self._kda_scale_rows(err, beta_fp, [acc], h)
        return err

    def kda_decode_update_v0(
        self,
        *,
        state: VRAMMatrixVar,
        k_fp: FPVar,
        q_fp: FPVar,
        o: VRAMMatrixVar,
        err: VRAMMatrixVar,
        shape: KdaShape,
        output_scale_fp: FPVar,
        head_rows: Sequence[int] | None = None,
        fp_head_stride: int | None = None,
    ) -> VRAMMatrixVar:
        """Second half: the rank-1 update, then the read-out.

        Reads ``k_fp`` and ``q_fp``; never touches ``decay_fp``. Must run after
        :meth:`kda_decode_predict_v0` on the same head, and must see the ``err``
        it produced.
        """
        heads, stride = self._kda_decode_setup(
            shape=shape, tiles=(("state", state), ("o", o), ("err", err)),
            state=state, pred=None, err=err,
            head_rows=head_rows, fp_head_stride=fp_head_stride,
        )
        blocks = kda_blocks(shape, self.mlen)
        k_dim = shape.key_dim

        for h in heads:
            fp_base = h * stride
            for c in range(blocks):
                first = kda_state_row(shape, self.mlen, h, c, 0)
                key_rows = list(range(first, first + k_dim))
                acc = kda_vector_row(shape, self.mlen, h, c)

                # Every key's update touches only that key's row, so updating
                # all of them and then reducing is identical to the reference's
                # interleaved `updated = ...; reduced += updated*q`.
                self.emit_comment(
                    kda_stage_marker("kda_state_update", f"rank-1 head={h} block={c}")
                )
                # state[row] += err[acc] * k_hat[i] -- the mirror of the
                # prediction: the source is pinned and the destination walks.
                self.tile_row_fma_fp_sweep(
                    state, err, k_fp,
                    dst_rows=key_rows, src_rows=[acc] * len(key_rows),
                    fpram_offset=fp_base,
                )

                self.emit_comment(kda_stage_marker("kda_readout", f"head={h} block={c}"))
                self.vram_fill_zero(o, rows=[acc])
                self.tile_row_fma_fp_sweep(
                    o, state, q_fp,
                    dst_rows=[acc] * len(key_rows), src_rows=key_rows,
                    fpram_offset=fp_base,
                )
                self._kda_scale_rows(o, output_scale_fp, [acc], 0)
        return o
