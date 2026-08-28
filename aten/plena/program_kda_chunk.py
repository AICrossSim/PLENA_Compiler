"""KDA chunk primitives: the cumulative decay, and the UT transform.

Chunked prefill collapses a chunk of ``C`` tokens into four products against the
incoming state, instead of ``C`` sequential rank-1 updates. For the gated delta
rule the collapse is not free: within a chunk the update is a product of rank-1
projectors, and turning that into a single matrix needs the WY / UT transform,
whose core is the inverse of a ``[C, C]`` lower-triangular matrix.

The algebra, verified against ``kda_step`` run sequentially
-----------------------------------------------------------
With state ``S[value, key]``, per-step decay ``a_t[key]`` and cumulative decay
``A_t[key] = prod_{s<=t} a_s[key]``::

    k_hat_t = k_t * A_t                     bounded by |k|, since A <= 1
    q_hat_t = q_t * A_t
    M[t, s] = sum_key k_t k_s A_t/A_s       for s < t
    N[t, s] = sum_key q_t k_s A_t/A_s       for s <= t
    T       = (I + tril(diag(beta) M, -1))^-1 diag(beta)
    E       = T (V - K_hat S_0^T)                       [C, value]
    out_t   = scale * (S_0 q_hat_t + sum_{s<=t} N[t,s] E_s)
    S_C     = A_C * S_0 + sum_s E_s outer (k_s * A_C/A_s)

**KDA's decay is channel-wise on the key axis**, not one scalar per timestep --
``reference.py`` applies ``exp(log_decay)[:, :, None, :]``. The textbook chunking
assumes a scalar decay and gives ``M = tril(diag(beta) K K^T, -1)`` with no decay
weighting at all; against the sequential reference that form is wrong by 1.8e-1,
which is an error and not a rounding. The ``A_t/A_s`` weighting is not optional.

Two numerical constraints, both measured
----------------------------------------
**The cumulative decay is a running product, not ``exp`` of a running sum.**
``A_t = exp(c_t)`` where ``c_t`` is the cumulative log-decay, and it is tempting
to follow ``program_ssd.py``'s cumsum -- a matmul against lower-triangular ones,
accumulated in f32. But ``c`` reaches ``chunk * gate_lower_bound = -80``, and at
that magnitude bf16's ulp is 0.31, so storing ``c`` and *then* exponentiating
costs a 17% relative error on ``A``. Multiplying the per-step decays instead
keeps every intermediate in ``[e^-5, 1]``, where bf16's error is relative rather
than magnified. End-to-end against the sequential reference, chunk 16, key 128:

    route            output error   state error
    cumsum in f32      8.1e-05        9.5e-04     (the algebra's own floor)
    cumsum in bf16     3.7e-04        1.1e-02
    cumprod in bf16    5.9e-05        2.4e-03

**``chunk * |gate_lower_bound|`` must stay under 88.** ``M`` and ``N`` are formed
as a matmul of ``k * A`` against ``k / A``, and ``1/A`` reaches
``exp(chunk * |lower_bound|)``. bf16's maximum is 3.39e38, whose log is 88.72, so
at Kimi K3's lower bound of -5 the last chunk size that works is **17**. Chunk 18
overflows to inf. That is why this file asserts the product rather than the chunk
size: raising either without the other is what breaks it.

The reciprocal is only ever used inside a *difference* -- every place ``1/A_s``
appears it is multiplied by some ``A_t`` with ``t > s``, so the product is
bounded by 1. Forming the two halves separately is what costs the range, and it
is done that way because it is the only form the matrix engine can contract. The
cancellation this causes is severe in ``M``'s smallest entries (up to 315%
relative) but does not reach the output: those entries contribute nothing, and
end-to-end the matmul form and an exactly-computed one agree to 6e-5.

Where the key axis lives: on lanes
----------------------------------
A ``[chunk, key_dim]`` tile here is ``chunk`` rows of ``key_dim`` lanes, spread
across ``kda_key_blocks`` **column blocks**. It is not the row-folded
``[chunk * key_blocks, mlen]`` layout the decode path uses for its state.

The choice is forced by the projections, and it is what lifts ``key_dim > mlen``.
Five of prefill's seven products contract over the key axis, and the systolic
array contracts a VRAM operand's *lanes*: ``vram_sub_projection_asm_impl`` walks
the operand's column blocks at stride ``physical_rows * mlen`` and accumulates
inside the array, so a key axis on lanes contracts across as many blocks as it
needs with no extra instruction and no explicit sum. With the key block folded
into rows instead, one projection can only ever see ``mlen`` of the key axis, and
a gram of the folded tile computes the ``kb != kb'`` cross terms as well -- half
the MACs thrown away, plus a strided gather to recover the diagonal.

The row ops pay for it, and the price is small: what used to be one sweep over
``chunk * key_blocks`` rows is ``key_blocks`` sweeps over ``chunk`` rows, passing
``tile_col_idx``. Same instruction count to within the loop headers, and rows
inside a column block are still consecutive, so each sweep is still one hardware
loop.
"""

from __future__ import annotations

# NOTE: these emitters bill to the existing ``kda_decay`` and
# ``kda_state_update`` stages. Prefill deserves its own stage names so Phase 4
# can separate its cost from decode's, but KDA_STAGES is guarded bidirectionally
# against the emulator's stage_profile.rs, so adding one means editing both
# repos together. Done with the prefill layer, not here.

import math

from compiler.aten.models.kda.shape import KdaShape
from compiler.aten.plena.program_kda_common import kda_stage_marker
from compiler.aten.plena.program_kda_gates import kda_key_blocks
from compiler.aten.plena.vars import FPVar, VRAMMatrixVar

__all__ = [
    "ProgramKdaChunkMixin",
    "kda_chunk_cols",
    "kda_chunk_rows",
    "kda_max_chunk_for",
    "kda_round_up",
]

#: ``ln`` of bf16's largest finite value. ``1/A`` over a chunk reaches
#: ``exp(chunk * |gate_lower_bound|)``, and this is where that stops being
#: representable.
_BF16_MAX_LOG = math.log(3.3895313892515355e38)


def kda_max_chunk_for(gate_lower_bound: float) -> int:
    """Largest chunk whose ``1/A`` still fits bf16. 17 at Kimi K3's -5."""
    if gate_lower_bound >= 0:
        raise ValueError(f"gate_lower_bound must be negative, got {gate_lower_bound}")
    return int(_BF16_MAX_LOG // abs(gate_lower_bound))


def kda_round_up(n: int, mlen: int) -> int:
    """``n`` rounded up to a whole number of ``mlen``-sized blocks, minimum one.

    Every prefill tile is allocated at block granularity in both axes. Rows,
    because a projection writes a whole ``mlen x mlen`` block and a spill
    prefetches one; columns, because ``_target_tile_addr`` places column block
    ``c`` at ``c * physical_rows * mlen``, so a tile whose declared width is not
    a multiple of ``mlen`` reserves less than the blocks it will be addressed by
    and the last one runs off the end.
    """
    if mlen <= 0:
        raise ValueError(f"mlen must be positive, got {mlen}")
    return max(1, -(-n // mlen)) * mlen


def kda_chunk_rows(shape: KdaShape, mlen: int, chunk: int) -> int:
    """Rows of a ``[chunk, key_dim]`` prefill tile: one per timestep.

    The key axis is on **lanes**, not folded into rows -- see the module
    docstring. ``shape`` is taken for the key-block validation it carries and
    for symmetry with the rest of the family.
    """
    kda_key_blocks(shape, mlen)
    return kda_round_up(chunk, mlen)


def kda_chunk_cols(shape: KdaShape, mlen: int) -> int:
    """Columns of a ``[chunk, key_dim]`` prefill tile: the whole key axis."""
    return kda_key_blocks(shape, mlen) * mlen


class ProgramKdaChunkMixin:
    """Chunk-level primitives for KDA prefill."""

    def kda_chunk_check_range(self, chunk: int, shape: KdaShape) -> None:
        """Refuse a chunk whose reciprocal decay would overflow bf16.

        Asserted rather than commented because the failure is silent: ``1/A``
        becomes ``inf``, ``M`` becomes ``nan``, and the triangular solve returns
        ``nan`` for the whole chunk.
        """
        if chunk < 1:
            raise ValueError(f"chunk must be positive, got {chunk}")
        span = chunk * abs(shape.gate_lower_bound)
        if span >= _BF16_MAX_LOG:
            raise ValueError(
                f"chunk {chunk} at gate_lower_bound {shape.gate_lower_bound} gives a "
                f"reciprocal decay of exp({span:.1f}), past bf16's exp({_BF16_MAX_LOG:.1f}). "
                f"The largest chunk that fits is "
                f"{kda_max_chunk_for(shape.gate_lower_bound)}"
            )

    def kda_chunk_decay_cumprod_v0(
        self,
        *,
        decay: VRAMMatrixVar,
        prev: VRAMMatrixVar,
        chunk: int,
        shape: KdaShape,
    ) -> VRAMMatrixVar:
        """``decay[t] <- prod_{s<=t} decay[s]``, in place, per key channel.

        ``decay`` arrives holding the **per-step** decay ``exp(log_decay[t, key])``
        -- which is exactly what :meth:`kda_decay_scalars_v0` leaves in ``gate``.
        It leaves holding the cumulative decay ``A_t``.

        A running product and not ``exp`` of a running sum; the module docstring
        has the measurement. ``prev`` is scratch of the same shape: the row ops
        take one row list for both operands, so the previous timestep has to be
        staged at the *current* timestep's row before the multiply.
        """
        self.kda_chunk_check_range(chunk, shape)
        blocks = kda_key_blocks(shape, self.mlen)
        rows = kda_chunk_rows(shape, self.mlen, chunk)
        cols = kda_chunk_cols(shape, self.mlen)
        for name, tile in (("decay", decay), ("prev", prev)):
            if tile.shape[1] != cols:
                raise ValueError(
                    f"{name} is {tile.shape[1]} columns wide; a [chunk, key_dim] "
                    f"tile carries the key axis on lanes and must be exactly "
                    f"key_dim ({cols}) wide"
                )
            if tile.shape[0] < rows:
                raise ValueError(f"{name} needs {rows} rows, has {tile.shape[0]}")
        if prev.name == decay.name:
            raise ValueError("prev must be distinct from decay")

        self.emit_comment(kda_stage_marker("kda_decay", f"chunk cumprod chunk={chunk}"))
        for t in range(1, chunk):
            # One copy covers every key block: `mamba_block_copy` goes through
            # `vram_fill_zero` and `vram_add`, both of which walk the column
            # blocks themselves. Only the multiply has to be driven per block,
            # because a binary row op pairs one block with one block.
            self.mamba_block_copy(
                prev, decay,
                dst_row_offset=t, src_row_offset=t - 1, num_rows=1,
            )
            for kb in range(blocks):
                self.tile_row_mul(decay, prev, rows=[t], tile_col_idx=kb)
        return decay

    def kda_ut_transform_v0(
        self,
        *,
        m: VRAMMatrixVar,
        identity: VRAMMatrixVar,
        beta_fp: FPVar,
        t_out: VRAMMatrixVar,
        m_fp: FPVar,
        consts,
        chunk: int,
        shape: KdaShape,
        beta_offset: int = 0,
    ) -> VRAMMatrixVar:
        """``T = (I + tril(diag(beta) M, -1))^-1 diag(beta)``, by forward substitution.

        ``m`` is the ``[chunk, chunk]`` decay-weighted key gram, consumed in place
        and left negated. ``identity`` is a host-staged ``[chunk, chunk]`` identity
        -- staged rather than built, for the reason ``ssd_lower_triangular_ones``
        gives: materialising a constant tile on chip costs one ``S_ST_FP`` per
        element. ``m_fp`` needs one padded row (``mlen`` slots).

        Solving ``L T = diag(beta)`` row by row, with ``L = I + tril(diag(beta)M, -1)``
        lower unitriangular::

            T[i] = beta_i * (e_i - sum_{j<i} M[i,j] T[j])

        which is one FMA sweep per row: the destination is pinned to row ``i``,
        the source walks rows ``0..i-1``, and the FPRAM slot walks ``M[i, 0..i-1]``.
        Both are progressions, so each row is a single hardware loop -- 16 of them
        for a chunk of 16, whatever ``key_dim`` is.

        ``M`` is negated up front so the sweep can accumulate: ``V_FMA_VF`` adds,
        and the substitution subtracts.
        """
        self.kda_chunk_check_range(chunk, shape)
        for name, tile in (("m", m), ("identity", identity), ("t_out", t_out)):
            if tile.shape[0] < chunk:
                raise ValueError(f"{name} needs {chunk} rows, has {tile.shape[0]}")
            if tile.shape[1] != self.mlen:
                raise ValueError(
                    f"{name} is {tile.shape[1]} columns wide; must be exactly "
                    f"mlen ({self.mlen})"
                )
        if chunk > self.mlen:
            raise ValueError(
                f"chunk {chunk} exceeds mlen {self.mlen}; the [chunk, chunk] tiles "
                f"here are one row per timestep and must fit a single block"
            )
        if m_fp.size < self.mlen:
            raise ValueError(
                f"m_fp holds {m_fp.size} slots but S_MAP_FP_V writes a whole row "
                f"({self.mlen}); size it to mlen, not to chunk"
            )
        if beta_fp.size < beta_offset + chunk:
            raise ValueError(
                f"beta_fp needs {beta_offset + chunk} slots, has {beta_fp.size}"
            )
        for name, tile in (("t_out", t_out), ("identity", identity)):
            if tile.name == m.name:
                raise ValueError(f"{name} must not alias m")

        self.emit_comment(kda_stage_marker("kda_state_update", f"UT transform chunk={chunk}"))
        rows = list(range(chunk))
        # The sweep accumulates, so negate once rather than subtracting per row.
        self.tile_row_mul_fp_broadcast(m, consts.neg_one, rows=rows)

        for i in rows:
            self.mamba_row_copy(t_out, i, identity, i)
            if i:
                # Row i of the negated M into FPRAM, then one loop over j < i.
                self.tile_row_to_fpram(m, m_fp, rows=[i])
                self.tile_row_fma_fp_sweep(
                    t_out, t_out, m_fp,
                    dst_rows=[i] * i, src_rows=list(range(i)),
                )
            self.tile_row_mul_fp_broadcast(
                t_out, beta_fp, rows=[i], fpram_offset=beta_offset + i
            )
        return t_out
