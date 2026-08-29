"""KDA (Kimi Delta Attention) state layout and data movement.

State layout
------------
Per head the recurrent state is stored ``[key_dim, value_dim]`` -- the
*transpose* of the reference's ``[value_dim, key_dim]`` in
``aten/models/kda/reference.py``. This is deliberate, and it is the decision the
whole KDA lowering rests on.

In the reference orientation the per-key decay is strided across rows and both
contractions (``prediction = S @ k`` and ``output = S @ q``) run *within* a row,
so each costs a ``V_MUL_VV`` plus a ``V_RED_SUM`` per row and lands its result in
an FP register that then has to be gathered back into a vector.

Transposed, ``key`` becomes the row axis and all four steps of the recurrence
reduce to "one row times one scalar"::

    decay     T[k, :] *= decay[k]
    predict   pred[:] += k_hat[k] * T[k, :]
    update    T[k, :] += err[:]   * k_hat[k]
    read out  out[:]  += q_hat[k] * T[k, :]

Each is then a single arithmetic row progression over the state tile, which
``_row_progression`` recognises and turns into a hardware loop rather than an
unrolled block. ``aten/tests/test_kda_reference.py``'s
``test_decode_step_matches_the_transposed_formulation`` pins the two orientations
together so the lowering cannot be validated against a plausible-but-different
reimplementation of the maths.

Mamba-2 already stores state the same way -- ``state_size`` as the row axis, see
``program_ssm_recurrent.py`` -- so the two algorithms share the emitters rather
than each getting their own.

**Prefill does not use this layout.** ``program_kda_prefill`` holds the state
``[value, key]``, because that is what makes its seven chunk-level products land
on the projection primitives without an explicit transpose. At Kimi K3
``key_dim == value_dim == 128``, so the two shapes match and passing one path's
state to the other is a finite, plausible, wrong answer rather than an error.
The conversion is ``kda_prefill_state_to_decode_layout_v0``, and it belongs at
the boundary -- once per layer, not inside either path.

Column blocks are folded into the row index
-------------------------------------------
A VRAM row is ``mlen`` elements wide, and Kimi K3's ``value_dim`` is 128 against
a default ``mlen`` of 64. Storing state as ``[key, value]`` would put one key's
row across two column blocks, and the helper family that would have to walk them
is inconsistent about it: ``tile_row_mul_fp`` takes a ``tile_col_idx``, but
``tile_row_sum``, ``tile_row_mul``, ``tile_row_sub``, ``vram_fill_zero``,
``mamba_row_copy`` and ``mamba_row_add`` do not. Mixing "walks every block" with
"silently block 0 only" is how a state ends up correct in its first 64 lanes and
stale in the rest -- no error, plausible numbers.

So the block index is folded into the row instead:
``row = (head * blocks + block) * key_dim + key``. Every row is then exactly one
block wide, no helper needs a column argument, and fixing ``(head, block)``
leaves the keys at consecutive rows -- a unit-stride progression, so each sweep
is still one hardware loop.

When ``value_dim == mlen`` this is byte-identical to the un-blocked layout, so it
is a generalisation rather than a second scheme to maintain. ``Mamba2Shape``
takes the other route for the same problem -- it *rejects* shapes wider than
``mlen`` (``program_mamba_common.py:175-188``) and its error text names the
helper fix as "deliberately not done yet". Folding the block into the row gets
the same safety without touching emitters that Mamba and attention are using.

Precision -- and what this module can and cannot promise
--------------------------------------------------------
State travels through the ``keyvalue`` precision class (``precision=1``), the
same one ``ssm_load_state_v0`` uses. **That class selects a name, not a width.**
The width comes from the active ``[<MODE>.PRECISION]`` table, and the shipped
one declares ``HBM_V_KV_TYPE`` as ``format = "Mx"`` with e4m3 elements --
1 byte plus a scale stream, not 2 and not 4. ``storage_precision`` only feeds
the compiler's own address arithmetic; it cannot change what the DMA decodes.

So this module **requires** Plain BF16 KV types rather than asserting them.
Call :meth:`kda_require_state_precision_v0` with the parsed precision table
before lowering a layer. Under an MX KV type the state decodes as e4m3 while
the address stride assumes 2 bytes: garbage, silently, with no runtime error.
``f5eb36a`` found and fixed exactly this for the Mamba SSD path
(``require_bf16_kv_precision``); KDA reuses that guard rather than repeating
the mistake.

Why BF16 and not something narrower: the state is a multiplicative accumulator
carried across the whole sequence, so quantisation error is amplified by
``1 / sqrt(1 - lambda^2)``. KDA's decay is per-key and driven toward 1 by
``gate_lower_bound`` -- exactly the long-memory regime where the amplification
is worst. e4m3's 3 mantissa bits compound badly there. FP32 would be better
still, but there is no FP32 path through the KV precision class, so BF16 is
the widest this ISA offers and the numerical claim is scoped to it.

Load and store must agree on both ``storage_precision`` and
``hbm_element_bytes``: a mismatch changes the row stride and the scale-section
base, so the state read back is not the state written -- again a wrong answer
rather than an error. Both are 2 here, and
:meth:`kda_pin_state_v0` reserves at the same width so the write-back cannot
overrun onto the next tensor.
"""

from __future__ import annotations

from compiler.aten.models.kda.shape import KdaShape
from compiler.aten.plena.vars import FPVar, InputVar, VRAMMatrixVar

__all__ = [
    "KDA_STAGES",
    "KdaShape",
    "ProgramKdaCommonMixin",
    "kda_blocks",
    "kda_conv_blocks",
    "kda_conv_channels",
    "kda_conv_state_row",
    "kda_head_base",
    "kda_stage_marker",
    "kda_state_row",
    "kda_state_rows",
    "kda_vector_row",
    "kda_vector_rows",
]

KDA_STAGE_MARKER_PREFIX = "@stage="

#: Every KDA stage name the emulator's ``StageKind`` understands.
#:
#: Kept in lockstep with ``transactional_emulator/src/stage_profile.rs``; the
#: cross-repo test in ``aten/tests/test_kda_stage_contract.py`` pins it. Marker
#: emission is validated against this set so a typo fails at ASM-gen time rather
#: than silently collapsing a region into the preceding stage.
KDA_STAGES: frozenset[str] = frozenset(
    {
        "kda_qkv_proj",
        "kda_conv1d",
        "kda_normalize",
        "kda_decay",
        "kda_state_update",
        "kda_readout",
        "kda_gated_norm",
        "kda_out_proj",
        "kda_state_load",
        "kda_state_store",
    }
)


def kda_stage_marker(stage: str, detail: str = "") -> str:
    """Format the explicit stage marker comment for ``stage``.

    The ``@`` is load-bearing. ``stage_profile.rs``'s ``extract_stage_tag``
    matches on ``"@stage="``; a bare ``stage=`` comment is not a marker at all,
    so every instruction under it bills to the *previous* marker -- silently,
    and the resulting profile looks perfectly plausible.

    Markers are authoritative and sticky, exactly as in ``mamba_stage_marker``:
    once a program contains any marker the emulator stops applying its legacy
    substring rules. Emit one whenever the stage changes and never mid-stage.
    """
    if stage not in KDA_STAGES:
        raise ValueError(f"unknown KDA stage {stage!r}; expected one of {sorted(KDA_STAGES)}")
    return f"{KDA_STAGE_MARKER_PREFIX}{stage}" + (f" {detail}" if detail else "")


def kda_blocks(shape: KdaShape, mlen: int) -> int:
    """Column blocks one value vector spans.

    A VRAM row is ``mlen`` elements wide, so a ``value_dim``-wide vector needs
    ``value_dim / mlen`` of them. Kimi K3 is ``value_dim = 128`` against a
    default ``mlen = 64``, so 2.
    """
    if mlen <= 0:
        raise ValueError(f"mlen must be positive, got {mlen}")
    if shape.value_dim % mlen:
        raise ValueError(
            f"value_dim ({shape.value_dim}) must be a multiple of mlen ({mlen}); a "
            f"partial trailing block would leave lanes past value_dim holding "
            f"whatever was there before, and they would still be summed"
        )
    return shape.value_dim // mlen


def kda_state_rows(shape: KdaShape, mlen: int) -> int:
    """Rows of the flattened state tile: one per ``(head, block, key)``."""
    return shape.num_heads * kda_blocks(shape, mlen) * shape.key_dim


def kda_state_row(shape: KdaShape, mlen: int, head: int, block: int, key: int) -> int:
    """Row holding ``block`` of head ``head``'s state at key ``key``.

    ``(head * blocks + block) * key_dim + key`` -- head outermost, then column
    block, then key. Fixing ``(head, block)`` therefore gives keys
    ``0..key_dim-1`` at consecutive rows, a unit-stride arithmetic progression,
    which is what ``_row_progression`` needs to emit a hardware loop instead of
    an unrolled block.
    """
    blocks = kda_blocks(shape, mlen)
    if not 0 <= head < shape.num_heads:
        raise ValueError(f"head {head} out of range for {shape.num_heads} heads")
    if not 0 <= block < blocks:
        raise ValueError(f"block {block} out of range for {blocks} blocks")
    if not 0 <= key < shape.key_dim:
        raise ValueError(f"key {key} out of range for key_dim {shape.key_dim}")
    return (head * blocks + block) * shape.key_dim + key


def kda_head_base(shape: KdaShape, mlen: int, head: int, block: int = 0) -> int:
    """First state row of ``head``'s ``block``."""
    return kda_state_row(shape, mlen, head, block, 0)


def kda_vector_rows(shape: KdaShape, mlen: int) -> int:
    """Rows of a flattened ``[num_heads, value_dim]`` tile.

    ``q``, ``v``, the output, and the two accumulators are all value-width, so
    they are blocked the same way the state is -- one row per
    ``(head, block)``. Keeping every value-width tile on the same convention is
    the point: a mixture is how a sweep ends up reading block 0 of one tile
    against block 1 of another.
    """
    return shape.num_heads * kda_blocks(shape, mlen)


def kda_vector_row(shape: KdaShape, mlen: int, head: int, block: int) -> int:
    """Row holding ``block`` of head ``head``'s value-width vector."""
    blocks = kda_blocks(shape, mlen)
    if not 0 <= head < shape.num_heads:
        raise ValueError(f"head {head} out of range for {shape.num_heads} heads")
    if not 0 <= block < blocks:
        raise ValueError(f"block {block} out of range for {blocks} blocks")
    return head * blocks + block


def kda_conv_blocks(channels: int, mlen: int) -> int:
    """Column blocks a ``channels``-wide vector spans.

    KDA's conv channel counts are ``num_heads * key_dim`` for q and k and
    ``num_heads * value_dim`` for v -- 12,288 and 12,288 for Kimi K3. Everything
    the conv does is elementwise across channels, so the blocks are independent
    and this is a plain loop bound, not a correctness question the way the
    key-axis reduction is.
    """
    if mlen <= 0:
        raise ValueError(f"mlen must be positive, got {mlen}")
    if channels % mlen:
        raise ValueError(
            f"channels ({channels}) must be a multiple of mlen ({mlen}); a partial "
            f"trailing block would convolve lanes that hold nothing"
        )
    return channels // mlen


def kda_conv_state_row(channels: int, mlen: int, kernel: int, block: int, tap: int) -> int:
    """Row holding tap ``tap`` of channel block ``block``.

    ``block * kernel + tap``, so one block's taps are consecutive and the
    history shift is a run of adjacent row copies -- the same reason the state
    layout puts keys consecutive within a ``(head, block)``.

    ``tap`` runs oldest (0) to newest (``kernel - 1``), matching the reference's
    ``torch.roll(state, shifts=-1, dims=-1)`` followed by writing the new value
    at ``[..., -1]``.
    """
    blocks = kda_conv_blocks(channels, mlen)
    if not 0 <= block < blocks:
        raise ValueError(f"block {block} out of range for {blocks} blocks")
    if not 0 <= tap < kernel:
        raise ValueError(f"tap {tap} out of range for kernel {kernel}")
    return block * kernel + tap


def kda_conv_channels(shape: KdaShape) -> int:
    """Channels in the q/k/v short-convolution history.

    ``2 * key_dim + value_dim`` per head: q and k are key-width, v is
    value-width. Matches ``KdaShape.conv_state_elements`` in the reference.
    """
    return shape.num_heads * (2 * shape.key_dim + shape.value_dim)


class ProgramKdaCommonMixin:
    """State residency and data movement for KDA.

    Requires ``ProgramSSMRecurrentMixin`` (``pin_hbm_region``),
    ``ProgramMambaCommonMixin`` (``mamba_row_copy``, ``mamba_rsqrt_fpram``) and
    ``ProgramTensorMixin`` (``input``, ``load_batch``, ``store``).
    """

    # -- residency ---------------------------------------------------------

    def kda_require_state_precision_v0(self, settings: dict | None = None) -> None:
        """Fail unless this build configures the KV precision classes as Plain BF16.

        Delegates to ``require_bf16_kv_precision``; see the module docstring for
        why a check and not an assertion. Passing ``settings=None`` raises rather
        than passing, because a check that could not run must not look like a
        check that passed.
        """
        self.require_bf16_kv_precision(settings)

    def kda_pin_state_v0(self, name: str, shape: KdaShape) -> int:
        """Reserve the pinned HBM range for one head-group's recurrent state.

        Sized from the flattened tile, ``state_rows x mlen``. That is the same
        element count as ``num_heads * key_dim * value_dim`` -- folding the block
        into the row redistributes elements, it does not add any.
        """
        return self.pin_hbm_region(
            name,
            kda_state_rows(shape, self.mlen) * self.mlen,
            hbm_element_bytes=2,
        )

    def kda_load_state_v0(self, name: str, shape: KdaShape, hbm_addr: int) -> VRAMMatrixVar:
        """Prefetch the pinned FP32 state tile into VRAM, transposed.

        Returns a ``[num_heads * blocks * key_dim, mlen]`` tile. Use
        :func:`kda_state_row` to index it; head ``h``'s block ``c`` occupies
        ``key_dim`` consecutive rows starting at
        ``(h * blocks + c) * key_dim``.
        """
        rows = kda_state_rows(shape, self.mlen)
        self.emit_comment(kda_stage_marker("kda_state_load", f"{name} [{rows},{self.mlen}]"))
        var = self.input(
            name,
            (rows, self.mlen),
            hbm_addr=hbm_addr,
            real_data_ratio=1.0,
        )
        return self.load_batch(
            var, name=f"{name}_vram", storage_precision=2, precision=1
        )

    def kda_store_state_v0(
        self, state: VRAMMatrixVar, name: str, hbm_addr: int
    ) -> InputVar:
        """Write the updated state back over the same pinned HBM range.

        Mirrors :meth:`kda_load_state_v0` exactly -- same precision class,
        same bytes per element. See the module docstring for why that matters.
        """
        self.emit_comment(kda_stage_marker("kda_state_store", name))
        return self.store(
            state,
            name=name,
            hbm_addr=hbm_addr,
            precision=1,
            hbm_element_bytes=2,
            real_data_ratio=1.0,
        )

    # -- short convolution history ----------------------------------------

    def kda_conv_state_roll_v0(
        self,
        conv_state: VRAMMatrixVar,
        new_row_src: VRAMMatrixVar,
        new_row_idx: int,
        shape: KdaShape,
    ) -> VRAMMatrixVar:
        """Shift the q/k/v conv1d history by one timestep and append the new one.

        Structurally identical to ``ssm_conv_state_roll_v0``: a physical copy,
        not a ring pointer. Address immediates are baked at ASM-gen time and
        there is no data-dependent addressing, so a runtime read pointer is not
        expressible. It costs ``conv_kernel - 1`` row copies per token.
        """
        history = shape.conv_kernel - 1
        if history <= 0:
            return conv_state
        if conv_state.shape[0] < history:
            raise ValueError(
                f"conv_state needs {history} rows, has {conv_state.shape[0]}"
            )
        self.emit_comment(kda_stage_marker("kda_conv1d", f"roll history={history}"))
        for i in range(history - 1):
            self.mamba_row_copy(conv_state, i, conv_state, i + 1)
        self.mamba_row_copy(conv_state, history - 1, new_row_src, new_row_idx)
        return conv_state

    # -- normalisation -----------------------------------------------------

    def kda_l2_normalize_v0(
        self,
        vec: VRAMMatrixVar,
        rows: list[int],
        sq_scratch: VRAMMatrixVar,
        acc_fp: FPVar,
        consts,
    ) -> VRAMMatrixVar:
        """In-place L2 normalisation of the listed VRAM rows.

        ``x <- x * rsqrt(sum(x^2) + eps)``.

        The epsilon goes **inside** the rsqrt, matching FlashKDA's recurrent
        kernel and the CPU reference (``reference.py``: ``rsqrt(sum + 1e-6)``).
        ``torch.nn.functional.normalize`` clamps the norm instead and does not
        agree; see the D3 note in the execution log for how easily that
        difference hides under a loose tolerance.

        ``consts.reci_group`` must be ``1.0``: ``mamba_rsqrt_fpram`` computes
        ``1 / sqrt(acc * reci_group + eps)``, and L2 is ``sqrt(sum)``, not
        ``sqrt(mean)``. Passing a gated-RMSNorm constants block here, whose
        ``reci_group`` is ``1 / group_width``, silently normalises by the wrong
        factor -- and produces plausible finite numbers while doing it. Callers
        get a constants block from :meth:`kda_fp_constants`.

        Follows ``mamba_gated_rmsnorm_v0`` step for step: block-copy, square by
        multiplying the copy against the source, ``V_RED_SUM`` into FPRAM,
        scalar rsqrt, then a per-row broadcast multiply. There is no vector
        square root -- ``S_FP_OP`` carries ``SQRT_FP``, ``V_ELEMENT_OP`` does
        not -- so the rsqrt is unavoidably scalar, one FP op per row.

        **Each row is normalised by its own norm.** That is right for a vector
        that fits one row, and wrong for one that does not.

        This matters because KDA normalises ``q`` and ``k``, which are
        *key*-width, and Kimi K3 is ``key_dim = 128`` against ``mlen = 64``. The
        row folding that solved ``value_dim`` does **not** apply here: the L2
        norm contracts over the key axis, so the two halves of one ``q`` share a
        single norm. Splitting them into two rows and calling this once would
        normalise each half by its own partial norm -- finite, plausible, wrong.

        Nor can the sums simply be accumulated: ``tile_row_sum`` emits
        ``S_ADD_FP f1, f0, f0`` before each ``V_RED_SUM``, so a second call
        overwrites the FPRAM slot rather than adding to it, even though the
        ``V_RED_SUM`` instruction itself accumulates into its FP register.

        The layer lowering therefore has to reduce each block into its own slot
        and combine them with scalar ``S_ADD_FP`` before the rsqrt -- ``blocks-1``
        extra scalar adds per vector. Until that exists, callers must pass
        vectors that fit one row.
        """
        if not rows:
            return vec
        if sq_scratch.name == vec.name:
            # mamba_block_copy is zero-then-add, so dst is src zeroes the vector
            # and every later step operates on zeros: sum 0, rsqrt(1e-6) = 1000,
            # vec *= 1000 on zeros. Silently returns an all-zero vector.
            raise ValueError("kda_l2_normalize_v0 needs a scratch tile distinct from vec")
        if acc_fp.size < len(rows):
            raise ValueError(
                f"acc_fp holds {acc_fp.size} slots but {len(rows)} rows are being "
                f"normalised; tile_row_sum writes one slot per row"
            )
        if vec.shape[1] != self.mlen:
            # Every tile in the KDA path is exactly one column block wide (see
            # the module docstring). A wider one would be normalised by a
            # partial norm, because tile_row_mul / tile_row_sum / tile_row_mul_fp
            # all default to tile_col_idx=0 while vram_fill_zero walks them all.
            raise ValueError(
                f"kda_l2_normalize_v0 needs a tile exactly mlen ({self.mlen}) wide, "
                f"got {vec.shape[1]}"
            )
        self.emit_comment(kda_stage_marker("kda_normalize", f"rows={len(rows)}"))
        self.mamba_block_copy(sq_scratch, vec, num_rows=vec.shape[0])
        self.tile_row_mul(sq_scratch, vec, rows=rows)
        self.tile_row_sum(acc_fp, sq_scratch, rows=rows, target_base_offset=0)
        self.mamba_rsqrt_fpram(acc_fp, consts, count=len(rows))
        self.tile_row_mul_fp(vec, acc_fp, rows=rows, fpram_base_offset=0)
        return vec

    def _kda_scale_rows(
        self,
        tile: VRAMMatrixVar,
        fp: FPVar,
        rows: list[int],
        fpram_offset: int,
    ) -> None:
        """``tile[rows[i]] *= FPRAM[fp + fpram_offset + i]``.

        Wraps a trap in ``_fpram_row_map``: with more than one row it walks
        ``base_offset + i``, but with exactly one row it ignores ``base_offset``
        and uses ``single_offset`` instead. Passing ``fpram_base_offset`` for a
        single row therefore silently reads slot 0 -- which for the per-key
        scalars here means every row scaled by head 0's first key.
        """
        if len(rows) == 1:
            self.tile_row_mul_fp(tile, fp, rows=rows, fpram_offset=fpram_offset)
        else:
            self.tile_row_mul_fp(tile, fp, rows=rows, fpram_base_offset=fpram_offset)

    def _kda_row_sum(self, target_fp, source, rows: list[int], offset: int) -> None:
        """``target_fp[offset + i] = sum(source[rows[i]])``.

        Wraps the same ``_fpram_row_map`` trap :meth:`_kda_scale_rows` above documents:
        with more than one row it walks ``base_offset + i``, but with exactly one
        it ignores ``base_offset`` and uses ``single_offset``. Passing the wrong
        one sends every call to slot 0, which for a per-block reduction means
        only the last block survives -- finite, plausible, wrong, and invisible
        at ``vectors > 1``.
        """
        if len(rows) == 1:
            self.tile_row_sum(target_fp, source, rows=rows, target_offset=offset)
        else:
            self.tile_row_sum(target_fp, source, rows=rows, target_base_offset=offset)

    def kda_l2_normalize_blocked_v0(
        self,
        vec: VRAMMatrixVar,
        *,
        vectors: int,
        blocks: int,
        sq_scratch: VRAMMatrixVar,
        part_fp: FPVar,
        acc_fp: FPVar,
        consts,
        first_row: int = 0,
    ) -> VRAMMatrixVar:
        """L2-normalise ``vectors`` vectors that each span ``blocks`` rows.

        ``vec`` is ``[vectors * blocks, mlen]`` with row ``v * blocks + c``.

        This is the cross-block case :meth:`kda_l2_normalize_v0` cannot do. KDA
        normalises ``q`` and ``k``, which are *key*-width, and Kimi K3 is
        ``key_dim = 128`` against ``mlen = 64`` -- but the norm **contracts over
        the key axis**, so the two halves of one ``q`` share a single norm.
        Normalising each row on its own would divide each half by its own partial
        norm: finite, plausible, wrong.

        The sums cannot simply be accumulated either. ``tile_row_sum`` emits
        ``S_ADD_FP f1, f0, f0`` before each ``V_RED_SUM``, so a second call into
        the same slot overwrites it, even though ``V_RED_SUM`` itself accumulates
        into its FP register.

        So each block is reduced into its own slot and the slots are folded with
        scalar ``S_ADD_FP``. The partial sums are laid out **block-major**,
        ``part_fp[c * vectors + v]``, precisely so the fold is ``blocks - 1``
        contiguous elementwise FPRAM adds of length ``vectors`` rather than
        ``vectors`` separate reductions.

        ``part_fp`` needs ``vectors * blocks`` slots, ``acc_fp`` needs
        ``vectors``. ``consts.reci_group`` must be 1.0 -- see
        :meth:`kda_fp_constant_values`.
        """
        if vectors < 1 or blocks < 1:
            raise ValueError(f"vectors and blocks must be positive, got {vectors}, {blocks}")
        if vec.shape[1] != self.mlen:
            raise ValueError(
                f"vec is {vec.shape[1]} columns wide; must be exactly mlen ({self.mlen})"
            )
        if vec.shape[0] < first_row + vectors * blocks:
            raise ValueError(
                f"vec needs {first_row + vectors * blocks} rows for {vectors} vectors "
                f"x {blocks} blocks starting at {first_row}, has {vec.shape[0]}"
            )
        if sq_scratch.name == vec.name:
            raise ValueError("sq_scratch must be distinct from vec")
        if sq_scratch.shape[0] < first_row + vectors * blocks or sq_scratch.shape[1] != self.mlen:
            raise ValueError("sq_scratch must match vec's shape")
        if part_fp.size < vectors * blocks:
            raise ValueError(
                f"part_fp holds {part_fp.size} slots, needs {vectors * blocks}"
            )
        if acc_fp.size < vectors:
            raise ValueError(f"acc_fp holds {acc_fp.size} slots, needs {vectors}")

        self.emit_comment(
            kda_stage_marker("kda_normalize", f"vectors={vectors} blocks={blocks}")
        )
        # `first_row` lets the mixer normalise one head's slice of a tile that
        # holds every head, rather than copying it out first.
        rows = list(range(first_row, first_row + vectors * blocks))
        # Copy only the live rows. VRAM row counts must be a multiple of mlen
        # (memory_state.py:309), so vec.shape[0] is the padded height -- at
        # mlen=64 with two live rows that is 64 rows copied for 2 rows of work,
        # about 380 wasted dynamic instructions. Only `rows` is reduced below,
        # so the padding never contributes.
        # Offset both sides so the copy is `vectors * blocks` rows wherever the
        # slice sits. Copying from row 0 up to the last live row instead made
        # the cost O(first_row), and the mixer walks first_row across every
        # head -- at Kimi K3 that is 18,624 rows moved for 384 rows of live
        # data. The rows stay at their original indices in sq_scratch so the
        # multiply and the reductions below can share one row list.
        self.mamba_block_copy(
            sq_scratch, vec,
            dst_row_offset=first_row, src_row_offset=first_row,
            num_rows=vectors * blocks,
        )
        self.tile_row_mul(sq_scratch, vec, rows=rows)

        # Block c of every vector reduces into part_fp[c * vectors ..]. Rows for
        # fixed c step by `blocks` and slots step by 1 -- both progressions, so
        # each block is one hardware loop.
        for c in range(blocks):
            self._kda_row_sum(
                part_fp,
                sq_scratch,
                [first_row + v * blocks + c for v in range(vectors)],
                c * vectors,
            )

        # Fold the per-block sums: acc[0..vectors) = sum_c part[c*vectors ..].
        # Explicit addresses: the FPVar-level helpers take whole vars, and these
        # operate on a slice of part_fp.
        self.fpvar_copy_asm(part_fp.address, acc_fp.address, vectors)
        for c in range(1, blocks):
            self.fpvar_add_asm(
                acc_fp.address, part_fp.address + c * vectors, acc_fp.address, vectors
            )

        self.mamba_rsqrt_fpram(acc_fp, consts, count=vectors)

        # Every row of vector v scales by the single acc[v].
        for c in range(blocks):
            self._kda_scale_rows(
                vec, acc_fp, [first_row + v * blocks + c for v in range(vectors)], 0
            )
        return vec

    def kda_fp_constants(self, name_prefix: str = "kda"):
        """Allocate the KDA layer's FPRAM scalar block.

        Reuses ``MambaFPConstants`` -- same seven slots, and a second allocator
        would bump-walk FP_MEM twice. ``mamba_fp_constants`` takes no shape, so
        nothing about Mamba's geometry comes along.

        ``dt_min`` / ``dt_max`` are unused by KDA (they clamp Mamba's ``dt``);
        they stay allocated so slot order matches what ``mamba_rsqrt_fpram`` and
        the other shared emitters index into.
        """
        return self.mamba_fp_constants(name_prefix=name_prefix)

    @staticmethod
    def kda_fp_constant_values(eps: float = 1.0e-6) -> list[float]:
        """Host-side values matching :meth:`kda_fp_constants`, in slot order.

        Two deliberate differences from ``mamba_fp_constant_values``:

        ``reci_group = 1.0``. ``mamba_rsqrt_fpram`` computes
        ``1 / sqrt(acc * reci_group + eps)``. Mamba wants ``sqrt(mean)`` so it
        passes ``1 / group_width``; KDA's q/k normalisation is the L2 norm
        ``sqrt(sum)``, so the factor must be 1.

        ``eps = 1e-6``, not ``1e-5``, matching FlashKDA's recurrent kernel and
        the CPU reference (``reference.py``: ``rsqrt(sum + 1.0e-6)``). The
        difference is small but it is a real convention, and
        ``test_kda_reference.py`` pins it.

        ``dt_min`` / ``dt_max`` are zero and bf16-max: unused by KDA, but a
        NaN or +inf parked in a live FPRAM slot would poison any emitter that
        later reached for it.
        """
        return [0.0, 1.0, -1.0, 0.0, 3.3895313892515355e38, 1.0, eps]
