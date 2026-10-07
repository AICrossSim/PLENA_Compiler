"""KDA's decay and beta scalars, and their route into FPRAM.

The recurrence consumes ``decay[h, k]``, ``beta[h]``, ``q_hat[h, k]`` and
``k_hat[h, k]`` as **FPRAM scalars**, because ``V_MUL_VF`` broadcasts one FP
register across every lane and each of those varies per row of the state tile.
They are produced on chip as VRAM rows, so each has to cross with
``S_MAP_FP_V`` -- one instruction per row, versus the one-hot
``V_MUL_VV`` + ``V_RED_SUM`` + ``S_ST_FP`` triple per *scalar* that was the only
route before it existed.

From ``aten/models/kda/reference.py::activate_log_decay`` and ``kda_step``::

    rate      = exp(a_log[h])                                   # host constant
    log_decay = gate_lower_bound * sigmoid(rate * (gate + dt_bias))
    decay     = exp(log_decay)
    beta      = sigmoid(beta_logit)

``rate`` is a function of a weight alone, so the host precomputes it and it
arrives as an FPRAM constant rather than costing a vector ``exp`` per token.
``gate_lower_bound`` likewise -- it lives on ``KdaShape`` and is a build
constant.

Sigmoid is done in place with four vector ops and no scratch::

    x <- -x            V_MUL_VF by consts.neg_one
    x <- exp(x)        V_EXP_V
    x <- x + 1         V_ADD_VF by consts.one
    x <- 1/x           V_RECI_V

``mamba_silu_v0`` runs the same sequence but needs a scratch tile because it
keeps ``x`` alive for a final multiply. Nothing here does, so the scratch and
its copy are dropped.

Layout. ``gate``, ``dt_bias`` and the decay output are **key**-width, blocked
one row per ``(head, key block)``; ``beta_logit`` is one value per head, blocked
by head. Both use their own row helpers rather than
:func:`kda_vector_row`, which blocks *value*-width tiles -- for Kimi K3 the two
happen to have the same block count, which is exactly the kind of coincidence
that hides a mix-up.

The FPRAM landing addresses fall out of the row order. ``tile_row_to_fpram``
puts row ``i`` at ``base + i * mlen``, so iterating ``(head, key block)`` in
order lands head ``h``'s block ``kb`` at ``h * key_dim + kb * mlen`` -- which is
exactly the ``key``-major-within-head layout the recurrence indexes.
"""

from __future__ import annotations

from typing import Sequence

from compiler.aten.models.kda.shape import KdaShape
from compiler.aten.plena.program_kda_common import kda_stage_marker
from compiler.aten.plena.vars import FPVar, VRAMMatrixVar

__all__ = [
    "ProgramKdaGatesMixin",
    "kda_head_blocks",
    "kda_key_blocks",
    "kda_key_row",
]


def kda_key_blocks(shape: KdaShape, mlen: int) -> int:
    """Column blocks a key-width vector spans."""
    if mlen <= 0:
        raise ValueError(f"mlen must be positive, got {mlen}")
    if shape.key_dim % mlen:
        raise ValueError(
            f"key_dim ({shape.key_dim}) must be a multiple of mlen ({mlen}); a "
            f"partial trailing block would carry lanes that hold nothing into "
            f"the FPRAM scalars"
        )
    return shape.key_dim // mlen


def kda_key_row(shape: KdaShape, mlen: int, head: int, block: int) -> int:
    """Row holding block ``block`` of head ``head``'s key-width vector.

    ``head * key_blocks + block``: head outermost, so a single
    ``tile_row_to_fpram`` over rows ``0..num_heads*key_blocks-1`` lands the
    scalars key-major within each head, matching what the recurrence indexes.
    """
    blocks = kda_key_blocks(shape, mlen)
    if not 0 <= head < shape.num_heads:
        raise ValueError(f"head {head} out of range for {shape.num_heads} heads")
    if not 0 <= block < blocks:
        raise ValueError(f"block {block} out of range for {blocks} key blocks")
    return head * blocks + block


def kda_head_blocks(shape: KdaShape, mlen: int) -> int:
    """Rows needed to hold one value per head.

    Unlike the key and value widths this is allowed a partial trailing block --
    Kimi K3 has 96 heads against ``mlen`` 64. The lanes past ``num_heads`` are
    never indexed, because the recurrence reads ``beta_fp[h]`` for
    ``h < num_heads``.
    """
    if mlen <= 0:
        raise ValueError(f"mlen must be positive, got {mlen}")
    return (shape.num_heads + mlen - 1) // mlen


class ProgramKdaGatesMixin:
    """Decay and beta, from projected activations to FPRAM scalars."""

    def _kda_sigmoid_inplace(self, tile: VRAMMatrixVar, rows: list[int], consts) -> None:
        """``tile[rows] <- 1 / (1 + exp(-tile[rows]))``, no scratch.

        ``V_EXP_V`` saturates its input to ``[-88, 88]`` rather than
        overflowing, so the tails need no explicit clamp: a large positive ``x``
        gives ``exp(-x) -> 0`` and sigmoid ``-> 1``; a large negative gives
        ``exp(-x) -> e^88`` and sigmoid ``-> 0``. Neither produces inf or NaN.
        """
        self.tile_row_mul_fp_broadcast(tile, consts.neg_one, rows=rows)
        self.tile_row_exp(tile, rows=rows)
        self.tile_row_add_fp_broadcast(tile, consts.one, rows=rows)
        self.tile_row_reci(tile, rows=rows)

    def kda_decay_scalars_v0(
        self,
        *,
        gate: VRAMMatrixVar,
        dt_bias: VRAMMatrixVar,
        rate_fp: FPVar,
        lower_bound_fp: FPVar,
        decay_fp: FPVar,
        consts,
        shape: KdaShape,
        heads: Sequence[int] | None = None,
    ) -> FPVar:
        """``decay[h,k] = exp(lower_bound * sigmoid(rate[h] * (gate + dt_bias)))``.

        ``gate`` is consumed in place and left holding the decay. ``rate_fp``
        carries ``exp(a_log[h])``, precomputed by the host because it depends
        only on a weight. ``lower_bound_fp`` carries ``shape.gate_lower_bound``.

        ``heads`` restricts the work to a subset, which is how the mixer keeps
        FPRAM bounded. Producing every head's scalars at once needs
        ``num_heads * key_dim`` slots -- 12,288 for Kimi K3 against an FPRAM of
        512 -- so the layer streams one head at a time and reuses the window.
        With a single head the scalars land at ``decay_fp[0 .. key_dim)``,
        because ``tile_row_to_fpram`` addresses by position in ``rows`` rather
        than by row index.
        """
        blocks = kda_key_blocks(shape, self.mlen)
        selected = list(range(shape.num_heads)) if heads is None else list(heads)
        rows = [kda_key_row(shape, self.mlen, h, b) for h in selected for b in range(blocks)]
        for name, tile in (("gate", gate), ("dt_bias", dt_bias)):
            if tile.shape[1] != self.mlen:
                raise ValueError(
                    f"{name} is {tile.shape[1]} columns wide; must be exactly "
                    f"mlen ({self.mlen})"
                )
            # max(rows) + 1, not len(rows): with a head subset the rows are a
            # slice out of the middle, so the count is smaller than the highest
            # row number. Checking the count accepts a tile that the emitter
            # then reads and writes past the end of -- silently, into whatever
            # VRAM object follows.
            needed = max(rows) + 1
            if tile.shape[0] < needed:
                raise ValueError(
                    f"{name} needs {needed} rows to reach head {max(selected)}, "
                    f"has {tile.shape[0]}"
                )
        if rate_fp.size < max(selected) + 1:
            raise ValueError(
                f"rate_fp holds {rate_fp.size} slots but head {max(selected)} is "
                f"selected; it is indexed by head number, not by position"
            )
        for h in selected:
            if not 0 <= h < shape.num_heads:
                raise ValueError(f"head {h} out of range for {shape.num_heads} heads")
        # Sized from what S_MAP_FP_V writes, which here happens to equal
        # num_heads * key_dim exactly because key_dim is a whole number of
        # blocks. Stated in terms of the write so it stays correct if that
        # changes.
        written = len(rows) * self.mlen
        if decay_fp.size < written:
            raise ValueError(
                f"decay_fp holds {decay_fp.size} slots but S_MAP_FP_V writes "
                f"{written} ({len(rows)} rows x mlen {self.mlen})"
            )

        self.emit_comment(kda_stage_marker("kda_decay", f"heads={len(selected)}"))
        self.tile_row_add(gate, dt_bias, rows=rows)

        # rate is per head, and rate_fp is indexed by head number -- so the
        # offset is the head, not its position in `selected`. For a fixed key
        # block a contiguous head range gives a row progression and a slot
        # progression; a scattered one falls back to unrolled, still correct.
        for kb in range(blocks):
            for h in selected:
                self._kda_scale_rows(
                    gate, rate_fp, [kda_key_row(shape, self.mlen, h, kb)], h
                )

        self._kda_sigmoid_inplace(gate, rows, consts)
        self.tile_row_mul_fp_broadcast(gate, lower_bound_fp, rows=rows)
        self.tile_row_exp(gate, rows=rows)

        # Row i -> decay_fp[i * mlen], and i = h*key_blocks + kb, so head h's
        # block kb lands at h*key_dim + kb*mlen: key-major within head.
        self.tile_row_to_fpram(gate, decay_fp, rows=rows)
        return decay_fp

    def kda_beta_scalars_v0(
        self,
        *,
        beta_logit: VRAMMatrixVar,
        beta_fp: FPVar,
        consts,
        shape: KdaShape,
    ) -> FPVar:
        """``beta[h] = sigmoid(beta_logit[h])``, consumed in place.

        ``beta_logit`` holds one value per head, so its last row may be partly
        empty -- Kimi K3 is 96 heads against ``mlen`` 64. Sigmoid of whatever
        those lanes hold is harmless to *read*: they are never indexed. It is
        not harmless to *write*, which is why ``beta_fp`` must be sized to the
        padded row count; see the check below.
        """
        if beta_logit.shape[1] != self.mlen:
            raise ValueError(
                f"beta_logit is {beta_logit.shape[1]} columns wide; must be exactly "
                f"mlen ({self.mlen})"
            )
        rows = list(range(kda_head_blocks(shape, self.mlen)))
        if beta_logit.shape[0] < len(rows):
            raise ValueError(f"beta_logit needs {len(rows)} rows")
        # S_MAP_FP_V moves a whole mlen-wide row, so the write is rows*mlen
        # slots, not num_heads. Checking num_heads lets the tail of the last row
        # land in the next FPVar: at 96 heads and mlen 64 that is 32 slots of
        # sigmoid(padding) written outside the allocation. `_resolve_fpram_addr`
        # only bounds-checks the base offset, and the emulator's scalar SRAM only
        # knows the whole file, so nothing downstream catches it either.
        written = len(rows) * self.mlen
        if beta_fp.size < written:
            raise ValueError(
                f"beta_fp holds {beta_fp.size} slots but S_MAP_FP_V writes "
                f"{written} ({len(rows)} rows x mlen {self.mlen}); size it to the "
                f"padded row count, not to num_heads"
            )

        self.emit_comment(kda_stage_marker("kda_decay", f"beta heads={shape.num_heads}"))
        self._kda_sigmoid_inplace(beta_logit, rows, consts)
        self.tile_row_to_fpram(beta_logit, beta_fp, rows=rows)
        return beta_fp
