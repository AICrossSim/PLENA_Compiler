"""KDA chunked prefill: one chunk of ``C`` tokens against the incoming state.

The algebra is derived and measured in ``program_kda_chunk``'s module docstring;
this file composes it. Per head, with ``S_0[value, key]`` the incoming state and
``A_t[key]`` the cumulative decay from :meth:`kda_chunk_decay_cumprod_v0`::

    k_hat = k * A          q_hat = q * A          k_tilde = k / A
    M     = tril(k_hat @ k_tilde^T, -1)
    N     = tril(q_hat @ k_tilde^T)                 diagonal included
    T     = (I + tril(diag(beta) M, -1))^-1 diag(beta)
    W     = V - k_hat @ S_0^T
    E     = T @ W
    out   = scale * (q_hat @ S_0^T + N @ E)
    S_C   = A_C * S_0 + E^T @ (k * A_C / A)

Seven matrix products. Every one has a **dynamic** second operand -- there is no
weight here, only activations and state -- so each is spilled to HBM and
re-prefetched into MRAM, which is the arrangement ``ssd_chunk_head_v0`` already
uses for Mamba's ``B``: MRAM is writable only by ``H_PREFETCH_M``.

``q`` and ``k`` must arrive L2-normalised
----------------------------------------
This is a hard precondition, not a nicety. ``reference.py::kda_step`` normalises
both with ``rsqrt(sum + 1e-6)`` before the recurrence sees them, and
``kda_mixer_step_v0`` does the same. Feeding raw projections instead makes the
substitution's matrix ``L = I + tril(diag(beta) M, -1)`` catastrophically
ill-conditioned, and the whole chunk is then worthless:

    inputs                       cond(L)        f32 error    bf16 error
    normalised, decay e^-5..1    1.0 - 1.1       2e-08        4e-04
    raw projections, |k| ~ 5     1e6 - 6e10      3e+21        2e+25

Chunk size
----------
:meth:`kda_chunk_check_range` bounds it at 17, where ``1/A`` stops fitting bf16.
Within that, **the error does not depend on the chunk size**, measured against
the sequential recurrence at three seeds each, worst case:

    chunk    cond(L)     f32        bf16 out    bf16 state
      4      1.0         9e-09      1.5e-04     2.5e-03
      8      1.0         1.1e-08    5.0e-04     2.0e-03
     16      1.0 - 1.1   2.2e-08    3.8e-04     3.8e-03

Nor does it **compound across chunks**. Eight chunks chained against an
equal-length sequential reference, worst state error per chunk:

    chunk  8:  1.7e-3  3.9e-3  1.4e-3  5.2e-3  1.7e-3  2.2e-3  5.2e-3  3.1e-3
    chunk 16:  2.3e-3  2.9e-3  2.4e-3  3.8e-3  3.2e-3  2.9e-3  3.9e-3  2.0e-3

Flat, because the decay makes the recurrence strongly contracting -- old error is
damped, not accumulated. An earlier version of this file claimed the opposite,
recommending chunk 8 on the strength of a conditioning table that ran from 3 to
759. That table was measured with **unnormalised** ``q``/``k`` and unnaturally
mild decays, so it described the top row of the table above and not this kernel.
Chunk 16 stands, and it is the size that makes the instruction count worth it.

Only the read-out is masked
---------------------------
``N`` is masked inclusively -- token ``t`` reads the state after its own update,
and letting it see ``s > t`` would be a causality violation worth 36% error.

``M`` is **not masked at all**. The substitution reads only ``M[i, j]`` for
``j < i``, so a mask on it is a second defence against the same mistake -- and a
second defence that nothing distinguishes from the first is worse than none:
with both present, swapping ``M``'s mask for the inclusive one, dropping the
``-1`` from it, or extending the substitution's sweep to ``j <= i`` each left the
emulator's answer **bit-identical**. Only applying two of them together produced
an error, of seven orders of magnitude. The one defence that remains is pinned by
``test_ut_transform_ignores_m_on_and_above_the_diagonal``, which feeds a polluted
diagonal and demands an identical result.

``M``'s upper triangle is left holding large values -- ``A_t/A_s`` for ``t < s``
is a product of reciprocal decays -- but they are finite by construction, which
is what :meth:`kda_chunk_check_range` really guarantees, and they are never read.

"""

from __future__ import annotations

from compiler.aten.models.kda.shape import KdaShape
from compiler.aten.plena.affine_layout import AffineLayout
from compiler.aten.plena.program_kda_chunk import kda_round_up
from compiler.aten.plena.program_kda_common import kda_stage_marker
from compiler.aten.plena.vars import FPVar, InputVar, VRAMMatrixVar

__all__ = [
    "KdaPrefillBuffers",
    "ProgramKdaPrefillMixin",
    "kda_prefill_state_transpose_shapes",
    "kda_prefill_tile_shapes",
]


def kda_prefill_tile_shapes(
    shape: KdaShape, mlen: int, chunk: int
) -> dict[str, tuple[int, int]]:
    """Required ``(rows, cols)`` of every tile :class:`KdaPrefillBuffers` holds.

    One source of truth, consulted by the emitter's validation *and* by whoever
    allocates. It exists because the two used to be written separately: the
    emitter carried a hand-kept list of row counts and every caller allocated
    ``mlen x mlen`` regardless, and the gap between them is exactly how a spill
    came to prefetch seven kilobytes past its allocation.

    Both axes are block-rounded; :func:`kda_round_up` says why. The width is what
    encodes each tile's role, so it is worth reading off:

    * ``key_dim`` wide -- the key axis is on lanes and gets contracted by a
      projection: ``q``, ``k``, ``decay``, ``k_tilde``, ``k_end``, ``prev``,
      and the state and its two scratch tiles.
    * ``value_dim`` wide -- value is the output column axis: ``out`` and the
      read-out's accumulator.
    * ``chunk`` wide -- time is on lanes, which is what makes ``E^T @ k_end``
      a contraction the array can do without an explicit transpose: ``v_t``,
      ``err_t``, and the four ``[chunk, chunk]`` tiles.
    """
    r = lambda n: kda_round_up(n, mlen)  # noqa: E731
    chunk_rows, chunk_cols = r(chunk), r(chunk)
    key_cols = r(shape.key_dim)
    value_rows, value_cols = r(shape.value_dim), r(shape.value_dim)
    return {
        "q": (chunk_rows, key_cols),
        "k": (chunk_rows, key_cols),
        "decay": (chunk_rows, key_cols),
        "k_tilde": (chunk_rows, key_cols),
        "k_end": (chunk_rows, key_cols),
        "prev": (chunk_rows, key_cols),
        "state": (value_rows, key_cols),
        "scale_scratch": (value_rows, key_cols),
        "state_contrib": (value_rows, key_cols),
        "v_t": (value_rows, chunk_cols),
        "err_t": (value_rows, chunk_cols),
        "contrib": (value_rows, chunk_cols),
        "out": (chunk_rows, value_cols),
        "readout_contrib": (chunk_rows, value_cols),
        "gram": (chunk_rows, chunk_cols),
        "readout": (chunk_rows, chunk_cols),
        "t_mat": (chunk_rows, chunk_cols),
        "identity": (chunk_rows, chunk_cols),
        "causal_inclusive": (chunk_rows, chunk_cols),
    }


def kda_prefill_state_transpose_shapes(
    shape: KdaShape, mlen: int
) -> dict[str, tuple[int, int]]:
    """Required ``(rows, cols)`` for :meth:`kda_prefill_state_to_decode_layout_v0`.

    Separate from :func:`kda_prefill_tile_shapes` because the transpose runs once
    per layer rather than once per chunk, and because its ``identity`` is a
    different tile from the chunk loop's: that one is ``[chunk, chunk]``, this
    one has to span the whole key axis.
    """
    r = lambda n: kda_round_up(n, mlen)  # noqa: E731
    return {
        "state": (r(shape.value_dim), r(shape.key_dim)),
        "identity": (r(shape.key_dim), r(shape.key_dim)),
        "out": (r(shape.key_dim), r(shape.value_dim)),
    }


class KdaPrefillBuffers:
    """Tiles and FPRAM windows one chunk of prefill needs, for one head.

    Grouped for the same reason ``KdaMixerBuffers`` is: a positional mistake
    between two same-shaped ``[chunk, chunk]`` tiles is a silent wrong answer.
    :func:`kda_prefill_tile_shapes` gives every tile's required shape, and the
    emitter checks each one against it.

    Three accumulators, not one
    ---------------------------
    ``contrib``, ``readout_contrib`` and ``state_contrib`` are all
    projection targets that get combined into something else, and a single
    scratch tile used to serve all three. It cannot any more: they are
    ``[value_dim, chunk]``, ``[chunk, value_dim]`` and ``[value_dim, key_dim]``,
    and once those widths differ ``vram_add`` refuses the mismatch -- which is
    the good outcome. Sizing one tile to the maximum of the three would have
    passed the shape check by coincidence whenever ``key_dim >= chunk``, and
    silently added the wrong lanes when it did not.

    Layout, and why two of these are transposed
    ------------------------------------------
    ``q``, ``k`` and ``decay`` are ``[chunk, key_dim]``: one row per timestep,
    the key axis on **lanes** across ``kda_key_blocks`` column blocks. See
    ``program_kda_chunk``'s module docstring for why the key axis is there and
    not folded into rows -- it is what lets ``key_dim`` exceed ``mlen``.
    ``v_t`` and ``err_t`` are ``[value_dim, chunk]``, **time on lanes**.

    That is not a preference. The final state needs ``E^T @ k_end``, a
    contraction over time on *both* operands, and the systolic array contracts a
    VRAM operand's lanes against an MRAM operand's rows. Carrying the error
    transposed makes every one of the seven products fall onto
    ``vram_sub_projection_to`` or its ``_T_to`` sibling with no explicit
    transpose anywhere -- the same move ``ssd_state_update_v0`` makes when it
    demands ``b_t_chunk`` with time on lanes.
    """

    def __init__(
        self,
        *,
        q: VRAMMatrixVar,
        k: VRAMMatrixVar,
        v_t: VRAMMatrixVar,
        decay: VRAMMatrixVar,
        k_tilde: VRAMMatrixVar,
        k_end: VRAMMatrixVar,
        gram: VRAMMatrixVar,
        readout: VRAMMatrixVar,
        t_mat: VRAMMatrixVar,
        identity: VRAMMatrixVar,
        causal_inclusive: VRAMMatrixVar,
        err_t: VRAMMatrixVar,
        state: VRAMMatrixVar,
        contrib: VRAMMatrixVar,
        readout_contrib: VRAMMatrixVar,
        state_contrib: VRAMMatrixVar,
        out: VRAMMatrixVar,
        prev: VRAMMatrixVar,
        scale_scratch: VRAMMatrixVar,
        beta_fp: FPVar,
        m_fp: FPVar,
        output_scale_fp: FPVar,
        consts,
    ):
        self.q, self.k, self.v_t = q, k, v_t
        self.decay, self.k_tilde, self.k_end = decay, k_tilde, k_end
        self.gram, self.readout = gram, readout
        self.t_mat, self.identity = t_mat, identity
        self.causal_inclusive = causal_inclusive
        self.err_t = err_t
        self.state, self.contrib, self.out = state, contrib, out
        self.readout_contrib, self.state_contrib = readout_contrib, state_contrib
        self.prev, self.scale_scratch = prev, scale_scratch
        self.beta_fp, self.m_fp = beta_fp, m_fp
        self.output_scale_fp = output_scale_fp
        self.consts = consts

    @staticmethod
    def causal_mask_values(chunk: int, mlen: int) -> list[list[float]]:
        """Host contents of ``causal_inclusive``: 1 iff ``s <= t``.

        Inclusive because token ``t`` reads the state *after* its own update.
        There is deliberately no strict variant: the only other place one was
        used was on the gram feeding the substitution, which reads only
        ``j < i`` on its own. Staged rather than built on chip for the reason
        ``ssd_lower_triangular_ones`` gives: materialising a constant tile costs
        one ``S_ST_FP`` per element.
        """
        return [
            [1.0 if (s < chunk and s <= t) else 0.0 for s in range(mlen)]
            for t in range(chunk)
        ]

    @staticmethod
    def identity_values(chunk: int, mlen: int) -> list[list[float]]:
        return [[1.0 if s == t else 0.0 for s in range(mlen)] for t in range(chunk)]

    @staticmethod
    def state_transpose_identity_values(key_dim: int) -> list[list[float]]:
        """Host contents of the identity :meth:`kda_prefill_state_to_decode_layout_v0` projects against.

        A different tile from :meth:`identity_values`, and bigger: the transpose
        contracts over the whole key axis, so this one is ``key_dim x key_dim``
        while the UT transform's is ``chunk x chunk``. Handing over the smaller
        one gives a finite, wrong answer whenever ``key_dim > chunk``, which is
        every real shape -- hence the separate name.
        """
        return [[1.0 if s == t else 0.0 for s in range(key_dim)] for t in range(key_dim)]


class ProgramKdaPrefillMixin:
    """One chunk of KDA prefill, for one head.

    Requires ``ProgramKdaChunkMixin`` (the cumulative decay and the UT
    transform), ``ProgramMatrixOpsMixin`` (the projections) and the tile-row
    family.
    """

    def _kda_blocks(self, n: int) -> int:
        """MLEN-wide tile blocks spanning ``n`` elements.

        Same arithmetic as ``ProgramSsdMixin._blocks``; duplicated rather than
        inherited so the KDA path does not drag in the whole SSD mixin.
        """
        return max(1, -(-n // self.mlen))

    def kda_prefill_state_to_decode_layout_v0(
        self,
        *,
        state: VRAMMatrixVar,
        identity: VRAMMatrixVar,
        out: VRAMMatrixVar,
        shape: KdaShape,
        precision: dict,
        output_layout: AffineLayout | None = None,
    ) -> VRAMMatrixVar:
        """Transpose the carried state from prefill's layout into decode's.

        **The two paths hold the state transposed relative to each other**, and
        at Kimi K3 ``key_dim == value_dim == 128``, so the shapes match and
        handing one to the other produces a finite, plausible, wrong answer
        rather than an error. Both layouts are deliberate:

        * decode is ``[key, value]`` (:func:`kda_state_row`) so that each sweep
          is one arithmetic row progression and becomes a hardware loop -- see
          ``program_kda_common``'s module docstring. Mamba-2 stores state the
          same way, which is why they share emitters.
        * prefill is ``[value, key]`` because that is what makes all seven of its
          products land on the projection primitives without an explicit
          transpose.

        So the conversion belongs at the boundary, once per layer, not inside
        either path. It is a projection against a staged identity:
        ``out[i][j] = sum_k I[i][k] * state[j][k] = state[j][i]``.

        With more than one block the block indices swap as well as the elements,
        and both loops here do exactly that. Writing the result block index pair
        as ``(ib, jb)`` with ``ib`` over **key** blocks and ``jb`` over **value**
        blocks::

            out[ib][jb][r][c] = sum_k I[ib*mlen+r][k] * state[jb*mlen+c][k]
                              = state[jb*mlen+c][ib*mlen+r]

        so the element transpose and the block transpose fall out of the same
        pair of loops, provided ``identity`` spans the whole key axis. It is a
        separate, larger tile from the chunk loop's ``[chunk, chunk]`` identity:
        :meth:`KdaPrefillBuffers.state_transpose_identity_values` builds it and
        :func:`kda_prefill_state_transpose_shapes` gives all three shapes.

        The result lands directly in decode's layout with no further move.
        Decode addresses its state at row ``block * key_dim + key`` (see
        :func:`kda_state_row`), and a VRAM matrix is column-block-major with
        column block ``jb`` at ``jb * physical_rows * mlen`` -- so with ``out``
        exactly ``key_dim`` rows tall the two coincide. That is a real
        constraint, not an observation, and it is asserted below: pad ``out`` to
        ``mlen`` rows when ``key_dim < mlen`` and the value blocks would land at
        the wrong stride.
        """
        want = kda_prefill_state_transpose_shapes(shape, self.mlen)
        for name, tile in (("state", state), ("identity", identity), ("out", out)):
            rows, cols = want[name]
            if tile.shape[0] < rows or tile.shape[1] != cols:
                raise ValueError(
                    f"{name} must be at least {rows} rows and exactly {cols} "
                    f"columns for key_dim {shape.key_dim} x value_dim "
                    f"{shape.value_dim} at mlen {self.mlen}; got {tile.shape}"
                )
        if out.name == state.name:
            raise ValueError("out must not alias state -- a projection overwrites it")

        key_blocks = self._kda_blocks(shape.key_dim)
        val_blocks = self._kda_blocks(shape.value_dim)
        if val_blocks > 1 and out.physical_shape[0] != shape.key_dim:
            raise ValueError(
                f"out is {out.physical_shape[0]} rows tall but decode indexes its "
                f"state at block * key_dim + key, so with {val_blocks} value "
                f"blocks it must be exactly key_dim ({shape.key_dim}) rows -- "
                f"otherwise the value blocks land at the wrong stride"
            )

        self.emit_comment(kda_stage_marker("kda_state_store", "prefill -> decode layout"))
        state_hbm = self.kda_prefill_spill_v0(
            tile=state, name="kda_state_transpose_spill",
            live_rows=shape.value_dim, precision=precision,
        )
        for ib in range(key_blocks):
            for jb in range(val_blocks):
                self.vram_sub_projection_T_to(
                    identity,
                    ib,
                    state_hbm,
                    jb,
                    out,
                    ib,
                    jb,
                    output_layout=output_layout,
                    **precision,
                )
        return out

    def kda_prefill_broadcast_row_v0(
        self,
        *,
        dst: VRAMMatrixVar,
        src: VRAMMatrixVar,
        src_row: int,
        count: int,
    ) -> VRAMMatrixVar:
        """Copy one row of ``src`` into the first ``count`` rows of ``dst``, by doubling.

        The row ops take one row list for both operands, so broadcasting a
        single row across a tile means a copy per destination row -- 64 of them
        at ``mlen`` 64, and 128 at Kimi K3's ``value_dim``. Copying the
        already-copied region instead doubles the filled span each time, so it
        is ``ceil(log2(count)) + 1`` copies: 7 instead of 64.

        Every copy covers all of the tile's column blocks, so the whole key axis
        is broadcast at once: ``mamba_block_copy`` goes through
        ``vram_fill_zero`` and ``vram_add``, and both walk the blocks
        themselves. ``dst`` and ``src`` must therefore be the same width --
        ``vram_add`` asserts it -- which for both callers here means both are
        ``key_dim`` wide.
        """
        if count < 1:
            raise ValueError(f"count must be positive, got {count}")
        if dst.shape[1] != src.shape[1]:
            raise ValueError(
                f"broadcast needs matching widths: dst {dst.shape[1]}, src "
                f"{src.shape[1]}. Every copy spans all column blocks"
            )
        self.mamba_block_copy(
            dst, src, dst_row_offset=0, src_row_offset=src_row, num_rows=1
        )
        filled = 1
        while filled < count:
            take = min(filled, count - filled)
            self.mamba_block_copy(
                dst, dst, dst_row_offset=filled, src_row_offset=0, num_rows=take
            )
            filled += take
        return dst

    def kda_prefill_spill_v0(
        self,
        *,
        tile: VRAMMatrixVar,
        name: str,
        live_rows: int,
        precision: dict,
    ) -> InputVar:
        """Spill ``tile`` to HBM for use as an MRAM operand, safely and reusably.

        Two things this does that a bare :meth:`store` does not.

        **It zero-fills the rows past ``live_rows``.** ``store`` sizes the HBM
        region from the tile's real height, but ``load_sub_matrix_col`` /
        ``load_sub_matrix_row`` prefetch a whole ``mlen x mlen`` block
        unconditionally -- ``k_block_count`` selects whole blocks and cannot trim
        a partial one. At ``chunk`` 8 and ``mlen`` 64 that is 1,024 bytes written
        and 8,192 read: the prefetch runs 7,168 bytes past the allocation, which
        at 1,024-byte spacing is straight through the next seven spills.

        For the four ``_T_to`` products the over-read rows land in output columns
        past ``chunk`` and are never used. But
        :meth:`kda_prefill_state_tail_v0` uses the plain ``_to`` form, which
        contracts over the MRAM operand's **rows** -- so garbage there is summed
        directly into the carried state. It has been correct only because
        ``k_end`` happens to be allocated last, with nothing after it. Adding a
        spill, or reordering them, would have broken it silently. Requiring the
        tile to be ``mlen`` rows tall and zeroing the tail makes the prefetch
        read exactly what was written.

        **It reuses one HBM region per name.** Each call used to allocate fresh:
        six regions per chunk per head, never reclaimed, about 20 KB per chunk --
        10 MB per head over a 4k-token prefill. Reuse would have been unsafe
        before, because the over-read would then have found the *previous*
        chunk's real data instead of unallocated zeros. The two defects had to be
        fixed together, and this is where.
        """
        if tile.shape[0] < self.mlen:
            raise ValueError(
                f"{name} spills into an mlen x mlen MRAM block, so its tile must "
                f"be {self.mlen} rows tall; got {tile.shape[0]}. The rows past "
                f"live_rows are zeroed here so the prefetch reads what was written"
            )
        if live_rows < tile.shape[0]:
            # To the tile's own height, not to `mlen`. A tile taller than one
            # block -- the state at `value_dim` 128 against `mlen` 64 -- has dead
            # rows past `mlen` too, and bounding the fill at `mlen` made the
            # range empty and zeroed nothing at all.
            self.vram_fill_zero(tile, rows=list(range(live_rows, tile.shape[0])))
        regions = getattr(self, "_kda_spill_regions", None)
        if regions is None:
            regions = self._kda_spill_regions = {}
        addr = regions.get(name)
        var = self.store(
            tile, name=name, hbm_addr=addr, precision=1,
            hbm_element_bytes=precision["hbm_element_bytes"],
        )
        regions[name] = self.get_hbm_layout(var.name).hbm_base_addr
        return var

    def kda_prefill_reciprocal_decay_v0(
        self,
        *,
        decay: VRAMMatrixVar,
        k: VRAMMatrixVar,
        k_tilde: VRAMMatrixVar,
        chunk: int,
        shape: KdaShape,
    ) -> VRAMMatrixVar:
        """``k_tilde = k / A``, the one unbounded quantity in the chunk.

        ``1/A`` reaches ``exp(chunk * |gate_lower_bound|)``, which is why
        :meth:`kda_chunk_check_range` refuses a chunk past 17 at Kimi K3's -5.
        It is formed here and consumed immediately by the two grams; nothing
        downstream keeps it.

        Note this is a *reciprocal of the cumulative decay*, not a separate
        reverse scan: ``V_RECI_V`` on a value in ``[e^-80, 1]`` is exact to
        bf16's relative precision, whereas a reverse product would accumulate a
        second set of roundings.
        """
        self.kda_chunk_check_range(chunk, shape)
        rows = list(range(chunk))
        self.emit_comment(kda_stage_marker("kda_decay", f"reciprocal chunk={chunk}"))
        # One copy for every column block; the two row ops once per block.
        self.mamba_block_copy(k_tilde, decay, num_rows=chunk)
        for kb in range(self._kda_blocks(shape.key_dim)):
            self.tile_row_reci(k_tilde, rows=rows, tile_col_idx=kb)
            self.tile_row_mul(k_tilde, k, rows=rows, tile_col_idx=kb)
        return k_tilde

    def kda_prefill_gram_v0(
        self,
        *,
        left: VRAMMatrixVar,
        right_hbm: InputVar,
        out: VRAMMatrixVar,
        chunk: int,
        shape: KdaShape,
        precision: dict,
        mask: VRAMMatrixVar | None = None,
    ) -> VRAMMatrixVar:
        """``out = mask * (left @ right_hbm^T)``, a ``[chunk, chunk]`` gram.

        ``left`` is ``k_hat`` or ``q_hat`` in VRAM; ``right_hbm`` is ``k_tilde``
        spilled. When a mask is given it is applied after the matmul rather than
        folded into the operands, because the systolic array has no triangular
        mode -- the same arrangement the attention path uses for its causal mask.

        ``mask=None`` is for the gram feeding the substitution, which reads only
        ``M[i, j]`` for ``j < i``. Masking it too was a second defence that made
        the first one untestable; see the module docstring.
        """
        i_blocks = self._kda_blocks(chunk)
        j_blocks = self._kda_blocks(chunk)
        self.emit_comment(kda_stage_marker("kda_state_update", f"gram chunk={chunk}"))
        for ib in range(i_blocks):
            for jb in range(j_blocks):
                self.vram_sub_projection_T_to(
                    left, ib, right_hbm, jb, out, ib, jb, **precision
                )
        if mask is not None:
            self.tile_row_mul(out, mask, rows=list(range(chunk)))
        return out

    def kda_prefill_chunk_v0(
        self,
        *,
        buffers: KdaPrefillBuffers,
        chunk: int,
        shape: KdaShape,
        precision: dict | None = None,
    ) -> tuple[VRAMMatrixVar, VRAMMatrixVar]:
        """One chunk of prefill for one head: ``(out, state)``.

        ``buffers.state`` is read as ``S_0`` and left holding ``S_C``, so chunks
        chain by calling this again. ``q``, ``k`` and ``decay`` are consumed in
        place: ``decay`` becomes the cumulative decay and then ``A_C/A``, and
        ``q``/``k`` become ``q_hat``/``k_hat``. Reseed all three per chunk.

        ``precision`` defaults to the spilled-activation class, which must be
        **Plain BF16**, and this method checks that itself against the active
        TOML. Under the shipped MX-FP8 KV types the spills decode as e4m3 and the
        read walks into the scale stream, whose ``0x7f`` bytes are e4m3 NaN --
        every output ``nan``. Even where it does not NaN, the state is a
        multiplicative accumulator carried across every chunk, so three mantissa
        bits at each boundary compounds without bound.
        """
        from compiler.aten.plena.program_ssd import SPILLED_ACTIVATION

        precision = SPILLED_ACTIVATION if precision is None else precision
        if precision is SPILLED_ACTIVATION:
            # The emitter picks the keyvalue class, so the emitter checks it.
            # Leaving this to the caller is what left it unchecked everywhere.
            self.require_bf16_kv_precision_from_active_build()
        self.kda_chunk_check_range(chunk, shape)
        b = buffers
        key_blocks = self._kda_blocks(shape.key_dim)
        val_blocks = self._kda_blocks(shape.value_dim)
        t_blocks = self._kda_blocks(chunk)
        chunk_rows = list(range(chunk))

        if chunk > self.mlen:
            raise ValueError(
                f"chunk {chunk} exceeds mlen {self.mlen}; the [chunk, chunk] tiles "
                f"are one row per timestep and must fit a single block"
            )
        # Every tile against the one table. Rows must be at least what the table
        # asks (a caller is free to over-allocate); the width must match exactly,
        # because a projection derives the length of its contraction from the
        # operand's column-block count -- too narrow contracts over part of the
        # axis, too wide contracts against lanes that hold nothing, and both are
        # finite wrong answers rather than errors.
        want = kda_prefill_tile_shapes(shape, self.mlen, chunk)
        for name in want:
            tile = getattr(b, name)
            rows, cols = want[name]
            if tile.shape[0] < rows:
                raise ValueError(
                    f"{name} needs {rows} rows, has {tile.shape[0]}"
                )
            if tile.shape[1] != cols:
                raise ValueError(
                    f"{name} is {tile.shape[1]} columns wide; it must be exactly "
                    f"{cols} at key_dim {shape.key_dim}, value_dim "
                    f"{shape.value_dim}, chunk {chunk}, mlen {self.mlen}"
                )
        for left, right in (
            ("prev", "decay"), ("out", "readout_contrib"),
            ("state", "state_contrib"), ("state", "scale_scratch"),
            ("v_t", "contrib"), ("k_tilde", "k_end"),
        ):
            if getattr(b, left).name == getattr(b, right).name:
                raise ValueError(
                    f"{left} and {right} must be distinct tiles; one is read "
                    f"while the other is written"
                )

        self.emit_comment(
            kda_stage_marker("kda_state_update", f"prefill chunk={chunk}")
        )

        # -- 1. A_t = prod_{s<=t} a_s, per key channel ------------------------
        self.kda_chunk_decay_cumprod_v0(
            decay=b.decay, prev=b.prev, chunk=chunk, shape=shape
        )

        # -- 2. k_tilde = k / A, the one unbounded quantity -------------------
        self.kda_prefill_reciprocal_decay_v0(
            decay=b.decay, k=b.k, k_tilde=b.k_tilde, chunk=chunk, shape=shape
        )
        k_tilde_hbm = self.kda_prefill_spill_v0(
            tile=b.k_tilde, name="kda_k_tilde_spill",
            live_rows=chunk, precision=precision,
        )

        # -- 3. k_end = k_tilde * A_C, while k is still the raw key ----------
        # k_end[t] = k[t] * A_C / A_t, and k_tilde already carries the 1/A_t.
        # A_C <= A_t for every t in the chunk (decays are <= 1), so the product
        # is bounded by |k| even though k_tilde on its own is not.
        self.kda_prefill_final_decay_ratio_v0(
            k_tilde=b.k_tilde, decay=b.decay, k_end=b.k_end,
            chunk=chunk, shape=shape,
        )

        # -- 4. k_hat and q_hat, in place -------------------------------------
        # Once per key block: a binary row op pairs one column block of the
        # destination with the same block of the source.
        for kb in range(key_blocks):
            self.tile_row_mul(b.k, b.decay, rows=chunk_rows, tile_col_idx=kb)
            self.tile_row_mul(b.q, b.decay, rows=chunk_rows, tile_col_idx=kb)

        # -- 5. the two grams, masked ---------------------------------------
        # No mask on this one. See the module docstring: the substitution reads
        # only j < i, so masking M was a redundant second defence -- and it made
        # the first one impossible to test, because breaking either alone left
        # the answer bit-identical.
        self.kda_prefill_gram_v0(
            left=b.k, right_hbm=k_tilde_hbm,
            out=b.gram, chunk=chunk, shape=shape, precision=precision,
        )
        # This one *is* load-bearing: N[t, s] for s > t would let token t read a
        # future token's error. Swapping it for the strict mask is a 36% error.
        self.kda_prefill_gram_v0(
            left=b.q, right_hbm=k_tilde_hbm, mask=b.causal_inclusive,
            out=b.readout, chunk=chunk, shape=shape, precision=precision,
        )

        # -- 6. T = (I + tril(diag(beta) M, -1))^-1 diag(beta) ----------------
        self.kda_ut_transform_v0(
            m=b.gram, identity=b.identity, beta_fp=b.beta_fp, t_out=b.t_mat,
            m_fp=b.m_fp, consts=b.consts, chunk=chunk, shape=shape,
        )

        # -- 7. W^T = v^T - S_0 @ k_hat^T -------------------------------------
        # Carried transposed: (S_0 @ k_hat^T)[v, t] = sum_key S_0[v,key] k_hat[t,key],
        # which is `_T_to` with the state natural in VRAM and k_hat in MRAM.
        k_hat_hbm = self.kda_prefill_spill_v0(
            tile=b.k, name="kda_k_hat_spill",
            live_rows=chunk, precision=precision,
        )
        for vb in range(val_blocks):
            for tb in range(t_blocks):
                self.vram_sub_projection_T_to(
                    b.state, vb, k_hat_hbm, tb, b.contrib, vb, tb, **precision
                )
        self.tile_row_sub(b.v_t, b.contrib, rows=list(range(shape.value_dim)))

        # -- 8. E^T = W^T @ T^T -----------------------------------------------
        # (W^T T^T)[v, t] = sum_s W^T[v,s] T[t,s] -- again `_T_to`, with T in MRAM.
        t_hbm = self.kda_prefill_spill_v0(
            tile=b.t_mat, name="kda_t_spill",
            live_rows=chunk, precision=precision,
        )
        for vb in range(val_blocks):
            for tb in range(t_blocks):
                self.vram_sub_projection_T_to(
                    b.v_t, vb, t_hbm, tb, b.err_t, vb, tb, **precision
                )

        # -- 9. out = scale * (q_hat @ S_0^T + N @ E) -------------------------
        state_hbm = self.kda_prefill_spill_v0(
            tile=b.state, name="kda_state_spill",
            live_rows=shape.value_dim, precision=precision,
        )
        for tb in range(t_blocks):
            for vb in range(val_blocks):
                self.vram_sub_projection_T_to(
                    b.q, tb, state_hbm, vb, b.out, tb, vb, **precision
                )
        err_hbm = self.kda_prefill_spill_v0(
            tile=b.err_t, name="kda_err_spill",
            live_rows=shape.value_dim, precision=precision,
        )
        # (N @ E)[t, v] = sum_s N[t,s] E^T[v,s] -- it lands in a scratch tile
        # because a projection overwrites its target (M_MM accumulates in the
        # array, M_MM_WO writes and clears), so it cannot add into `out`.
        # `readout_contrib` and not `contrib`: this one is [chunk, value_dim],
        # and step 7's is [value_dim, chunk].
        for tb in range(t_blocks):
            for vb in range(val_blocks):
                self.vram_sub_projection_T_to(
                    b.readout, tb, err_hbm, vb, b.readout_contrib, tb, vb, **precision
                )
        self.vram_add(b.out, b.readout_contrib, num_rows=chunk)
        self.emit_comment(kda_stage_marker("kda_readout", f"prefill chunk={chunk}"))
        # Once per value block. `out` is `[chunk, value_dim]`, and a row op
        # reaches one column block per call -- so at value_dim 128 against mlen
        # 64 a single call left the upper half of every token unscaled, i.e.
        # `sqrt(key_dim)` = 11.3x too large. It read as a rounding problem in the
        # aggregate (94.7% of values still inside a 5e-2 tolerance on data of
        # order 1e-3) and only separating the blocks showed it.
        for vb in range(val_blocks):
            self.tile_row_mul_fp_broadcast(
                b.out, b.output_scale_fp, rows=chunk_rows, tile_col_idx=vb
            )

        # -- 10. S_C = A_C * S_0 + E^T @ (k * A_C / A) ------------------------
        # k_end is bounded by |k|: A_C/A_s is a product of decays over (s, C],
        # all <= 1. Forming k * A_C / A_s rather than (k/A_s) * A_C keeps it that
        # way -- the second grouping goes through 1/A_s, which is what bounds the
        # chunk size in the first place.
        self.kda_prefill_state_tail_v0(
            k_end=b.k_end, decay=b.decay, err_t=b.err_t, state=b.state,
            state_contrib=b.state_contrib, scale_scratch=b.scale_scratch,
            chunk=chunk, shape=shape, precision=precision,
        )
        return b.out, b.state

    def kda_prefill_final_decay_ratio_v0(
        self,
        *,
        k_tilde: VRAMMatrixVar,
        decay: VRAMMatrixVar,
        k_end: VRAMMatrixVar,
        chunk: int,
        shape: KdaShape,
    ) -> VRAMMatrixVar:
        """``k_end[t] = k[t] * A_C / A_t``, from ``k_tilde = k / A``.

        The final state contracts ``E^T`` against this. Written as
        ``k_tilde * A_C`` rather than ``(k * A_C) / A`` because the first form
        reuses a tile that already exists, and both are the same number: the
        ratio ``A_C / A_t`` is a product of decays over ``(t, C]``, every one at
        most 1, so ``k_end`` is bounded by ``|k|`` however small ``A_t`` gets.
        """
        blocks = self._kda_blocks(shape.key_dim)
        rows = list(range(chunk))
        self.emit_comment(
            kda_stage_marker("kda_state_store", f"final decay ratio chunk={chunk}")
        )
        # A_C is the last timestep's row, broadcast down every timestep. One row
        # per timestep now, spanning every key block, so no destination pitch.
        self.kda_prefill_broadcast_row_v0(
            dst=k_end, src=decay, src_row=chunk - 1, count=chunk
        )
        for kb in range(blocks):
            self.tile_row_mul(k_end, k_tilde, rows=rows, tile_col_idx=kb)
        return k_end

    def kda_prefill_state_tail_v0(
        self,
        *,
        k_end: VRAMMatrixVar,
        decay: VRAMMatrixVar,
        err_t: VRAMMatrixVar,
        state: VRAMMatrixVar,
        state_contrib: VRAMMatrixVar,
        scale_scratch: VRAMMatrixVar,
        chunk: int,
        shape: KdaShape,
        precision: dict,
    ) -> VRAMMatrixVar:
        """``state = A_C * state + E^T @ k_end``, closing the chunk.

        ``A_C`` is per key channel, and the state's rows are *values*, so the
        decay is applied along the lane axis of every row -- one
        :meth:`tile_row_mul` against the broadcast row, not a per-row scalar.
        That only works because the state is held ``[value, key]``; under the
        reference's ``[value, key]`` it is the same thing, and under a
        ``[key, value]`` layout it would be a per-row scalar instead.
        """
        key_blocks = self._kda_blocks(shape.key_dim)
        val_blocks = self._kda_blocks(shape.value_dim)
        self.emit_comment(kda_stage_marker("kda_state_store", f"prefill chunk={chunk}"))

        k_end_hbm = self.kda_prefill_spill_v0(
            tile=k_end, name="kda_k_end_spill",
            live_rows=chunk, precision=precision,
        )
        for vb in range(val_blocks):
            for kb in range(key_blocks):
                self.vram_sub_projection_to(
                    err_t, vb, k_end_hbm, kb, state_contrib, vb, kb, **precision
                )

        # state *= A_C. The row ops take one row list for both operands, so
        # A_C -- a single row -- is broadcast into a tile shaped like the state
        # first. Same staging the cumulative-decay scan needs, for the same
        # reason.
        self.kda_prefill_broadcast_row_v0(
            dst=scale_scratch, src=decay, src_row=chunk - 1, count=shape.value_dim
        )
        value_rows = list(range(shape.value_dim))
        for kb in range(key_blocks):
            self.tile_row_mul(state, scale_scratch, rows=value_rows, tile_col_idx=kb)
        self.vram_add(state, state_contrib, num_rows=shape.value_dim)
        return state
