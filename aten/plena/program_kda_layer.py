"""A whole KDA layer: the input projection, the convolutions, the mixer, the
output projection.

`program_kda_mixer` stops at the state-engine boundary, which is the right place
for it -- the projections are ordinary Matrix work. This file is what puts them
back, because a whole-model instruction count needs the whole layer.

The reshape that is not a reshape
---------------------------------
The input projection produces one tile::

    projected = hidden @ W_in        [tokens, 3*key_width + value_width + heads]

and the mixer wants ``q``, ``k``, ``gate`` as ``[.., mlen]`` tiles per
``(head, key block)``. Those look like different layouts, and a naive conversion
is a scatter of one row per column block -- 700-odd of them per layer at Kimi
K3's shape, each a full block operation.

They are the same bytes. A VRAM tile's column block ``c`` sits at
``base + c * physical_rows * mlen`` (``memory.py``'s ``col_block_base``), and
rows *within* a block are linear. So the projection tile already **is** a dense
``[blocks * physical_rows, mlen]`` tile: feature block ``c``'s token ``t`` is at
row ``c * stride + t``, with ``stride = physical_rows``.

:meth:`vram_tall_view` names that whole tile and
:meth:`vram_column_block_view` names a single block; both register a VRAM object
at the computed address rather than allocating, so **neither emits an
instruction**. The stride is ``physical_rows * mlen`` and not ``mlen * mlen`` --
those coincide only when a tile happens to be exactly ``mlen`` rows tall, and a
projection allocated ``strict=False`` is padded to ``blen`` rows instead.

One gather, and the emitters are as they were
---------------------------------------------
Everything on the decode path past the projection wants a **dense** tile whose
row is a feature block. :meth:`kda_gather_projection_v0` moves one section into
place with a single ``V_FMA_VF`` sweep over the tall view -- both the source rows
(step ``stride``) and the destination rows (step 1) are arithmetic progressions,
so it is one hardware loop.

**Fourteen static instructions per section, independent of how many blocks it
spans.** All five sections of a Kimi K3 layer come to **70**, against 53,757 for
the three convolutions and 39,526 for the mixer.

Those are *image* sizes, and they are the wrong number to price the gather with.
The loop body runs once per feature block -- 192 of them for q, k, v and gate --
so the gather **issues 4,650 instructions against 492,681 for the layer, 0.94%**.
The static count is what the ``V_FMA_VF`` conversion bought: a program image that
does not grow with ``key_dim``. The dynamic count is what the gather costs, and
it is the only one of the two that may be compared against another way of
getting the sections into place. ``test_instruction_budget.py`` pins both.

Why the gather exists, and how to not need it
---------------------------------------------
Not because the sections share a tile. ``M_MM_WO`` writes a ``blen x blen``
sub-tile and the writeback loops cover ``mlen / blen`` column groups, so the
smallest thing a projection can lay down is ``blen`` token-rows by ``mlen``
lanes: column block ``c`` lands at row ``c * blen``, whatever the weights look
like. The consumers want one token's blocks as consecutive rows, and the
mismatch is exactly ``blen`` -- a property of the matrix writeback, not of the
packing.

So splitting the packed projection into five, each writing straight to its
consumer, does not help, and measurement agrees: identical ``M_MM`` count,
0.12% more instructions for the extra setup, the same 4,650-instruction gather.
``test_separate_projections_do_not_help`` records the numbers.

What does help is not moving the data at all. Every consumer takes explicit row
indices and ``_row_progression`` accepts any constant step, so reading the
projection at stride ``blen`` collapses into the same single hardware loop --
the step is an ``S_ADDI_INT`` immediate, and ``blen`` costs what ``1`` costs.
``kda_conv_step_v0`` takes ``x_new_row_base`` and ``x_new_row_stride`` for
this; feeding it the tall view directly removes **4,041 of the 4,650** issued
instructions, the q/k/v share, for nothing. The gate and beta sections reach a
different consumer and still gather.

An earlier version of this file claimed the layer could not be assembled without
first giving ``kda_conv_step_v0`` and ``kda_l2_normalize_blocked_v0`` a
column-block index. Both halves of that were wrong. The conv reads ``x_new`` at a
single row index rather than walking blocks as rows; and the normalisation never
sees the projection at all, because the mixer's ``q`` and ``k`` are the
convolutions' output, which the compiler allocates dense --
``test_kda_mixer.py``'s conv/mixer seam test already pins that. Threading a
column-block index through the Var layer was measured at +25% on the
normalisation, and it would have lost a cross-block hardware loop, to solve a
problem that is not on the path.
"""

from __future__ import annotations

from compiler.aten.models.kda.shape import KdaShape
from compiler.aten.plena.program_kda_common import kda_stage_marker
from compiler.aten.plena.program_kda_gates import kda_key_blocks
from compiler.aten.plena.vars import InputVar, VRAMMatrixVar

__all__ = [
    "ProgramKdaLayerMixin",
    "kda_projection_features",
    "kda_projection_sections",
    "kda_projection_width",
]


def kda_projection_features(shape: KdaShape) -> int:
    """Logical width of the input projection, matching ``kda_state_engine_step``.

    ``3 * key_width + value_width + num_heads``: q, k and gate are key-width,
    v is value-width, beta is one per head.
    """
    key_width = shape.projection_size
    return 3 * key_width + shape.num_heads * shape.value_dim + shape.num_heads


def kda_projection_sections(shape: KdaShape, mlen: int) -> list[tuple[str, int, int]]:
    """``(name, first column block, block count)`` for each projected section.

    Sections are **block-aligned**, in the order ``kda_state_engine_step``
    splits them. Only ``beta`` is not naturally a whole block -- it is one value
    per head, 96 at Kimi K3 against an ``mlen`` of 64 -- so it is padded to two,
    the same padding ``kda_head_blocks`` already applies. The lanes past
    ``num_heads`` are never indexed.

    Alignment is what makes the split free: a section that began mid-block could
    not be named as a view, and would need a scatter instead.
    """
    key_width = shape.projection_size
    value_width = shape.num_heads * shape.value_dim
    out, block = [], 0
    for name, width in (
        ("q", key_width), ("k", key_width), ("v", value_width),
        ("gate", key_width), ("beta", shape.num_heads),
    ):
        count = max(1, -(-width // mlen))
        out.append((name, block, count))
        block += count
    return out


def kda_projection_width(shape: KdaShape, mlen: int) -> int:
    """Padded width of the input projection.

    Larger than :func:`kda_projection_features` by the padding on ``beta``. The
    **host must materialise ``W_in`` at this width**, with the padding columns
    zero -- the lowering reads whole blocks, so a narrower weight would leave
    the tail of beta's block holding whatever preceded it in HBM.
    """
    name, block, count = kda_projection_sections(shape, mlen)[-1]
    return (block + count) * mlen


class ProgramKdaLayerMixin:
    """Assembles a KDA layer around the mixer."""

    def vram_column_block_view(
        self,
        tile: VRAMMatrixVar,
        block: int,
        *,
        name: str,
        rows: int | None = None,
    ) -> VRAMMatrixVar:
        """Name column block ``block`` of ``tile`` as its own ``[rows, mlen]`` tile.

        A view over the parent's bytes -- writing through one is visible through
        the other -- and it **emits nothing**.

        The address comes from :meth:`_tile_addr`, the same function every other
        consumer uses. Column blocks are strided by ``physical_rows * mlen``
        (``memory.py``'s ``col_block_base``), not by ``mlen * mlen``; those
        coincide only when a tile happens to be exactly ``mlen`` rows tall,
        which was the one shape the first version of this was tested at. A
        projection allocated ``strict=False`` is padded to ``blen`` rows, not
        ``mlen``, so a real decode projection was off by ``mlen / blen``.
        """
        mlen = self.mlen
        # The layout is authoritative, not the var: for a view they disagree,
        # the var carrying the rows asked for and the layout the parent's.
        layout = self.get_vram_layout(tile.name)
        phys_rows, phys_cols = layout.physical_shape
        # ceil, though `alloc` pads physical_cols to a multiple of mlen either
        # way -- strict requires it, non-strict rounds up. Kept as ceil so a
        # partial trailing block would stay addressable if that changes.
        blocks = -(-phys_cols // mlen)
        if not 0 <= block < blocks:
            raise ValueError(
                f"column block {block} out of range for {tile.name}, which has "
                f"{blocks} ({phys_cols} columns at mlen {mlen})"
            )
        rows = phys_rows if rows is None else rows
        if rows > phys_rows:
            raise ValueError(f"view of {rows} rows exceeds {tile.name}'s {phys_rows}")
        return self._named_view(
            name, rows, mlen, self._tile_addr(tile.name, 0, block), (phys_rows, mlen)
        )

    def vram_tall_view(
        self, tile: VRAMMatrixVar, *, name: str
    ) -> tuple[VRAMMatrixVar, int]:
        """The whole of ``tile`` as one dense ``[blocks * physical_rows, mlen]`` tile.

        Returns ``(view, stride)`` where ``stride`` is ``physical_rows`` -- the
        row distance between consecutive column blocks.

        Column block ``c`` sits at ``base + c * physical_rows * mlen`` and rows
        *within* a block are linear, so the tile's bytes already are a dense
        single-column-block tile. Feature block ``c``'s token ``t`` is at row
        ``c * stride + t``. That is what lets a gather be one hardware loop
        rather than a copy per block: both the source rows (step ``stride``) and
        the destination rows (step 1) are arithmetic progressions.

        Emits nothing.
        """
        mlen = self.mlen
        layout = self.get_vram_layout(tile.name)
        phys_rows, phys_cols = layout.physical_shape
        blocks = -(-phys_cols // mlen)
        return (
            self._named_view(
                name, blocks * phys_rows, mlen, layout.vram_base_addr,
                (blocks * phys_rows, mlen),
            ),
            phys_rows,
        )

    def _named_view(
        self, name: str, rows: int, cols: int, addr: int,
        physical_shape: tuple[int, int],
    ) -> VRAMMatrixVar:
        """`alloc_at` with a duplicate-name guard.

        ``register_vram_matrix`` overwrites without complaint and a
        ``VRAMMatrixVar`` resolves its address by name at emit time, so a
        duplicate silently repoints every view already handed out under that
        name. With a default prefix and a layer that runs 93 times that is a
        certainty rather than a risk.
        """
        internal = self._scoped_name(name)
        if internal in self.vram_matrices:
            raise ValueError(
                f"a VRAM object named {internal!r} already exists; give this view "
                f"a distinct name (views resolve by name at emit time, so a "
                f"duplicate repoints the earlier one)"
            )
        return self.alloc_at(name, rows, cols, addr, physical_shape=physical_shape)

    def kda_gather_projection_v0(
        self,
        *,
        projected: VRAMMatrixVar,
        dst: VRAMMatrixVar,
        section: str,
        shape: KdaShape,
        consts,
        token: int = 0,
        name: str = "kda_tall",
    ) -> VRAMMatrixVar:
        """Gather one section of ``projected`` into ``dst``'s first rows.

        The section's feature blocks live at rows ``(first + c) * stride + token``
        of the tall view; ``dst`` wants them at rows ``0 .. count-1``. Both are
        arithmetic progressions, so ``_emit_tile_row_fma`` collapses the whole
        gather into **one hardware loop** -- 14 static instructions per section,
        independent of how many blocks it spans. All five sections of a Kimi K3
        layer come to about 70 in the image; expanded, the loop bodies issue
        4,650 instructions, 0.94% of the layer's 492,681. See the module
        docstring on why only the second number prices the gather.

        The multiply is by ``1.0``: the FMA accumulates, so ``dst`` is zeroed
        first and the sweep adds the source in. There is no VRAM-to-VRAM move
        opcode, and this is cheaper than the copy idiom because the copy's
        destination would be a different row each iteration through
        ``mamba_row_copy``, which does not form a progression.
        """
        sections = {n: (first, count) for n, first, count in
                    kda_projection_sections(shape, self.mlen)}
        if section not in sections:
            raise ValueError(f"unknown section {section!r}; have {sorted(sections)}")
        first, count = sections[section]
        if dst.shape[0] < count:
            raise ValueError(f"dst needs {count} rows for {section}, has {dst.shape[0]}")

        tall, stride = self.vram_tall_view(projected, name=f"{name}_{section}")
        if token >= stride:
            raise ValueError(
                f"token {token} is past the projection's {stride} physical rows"
            )
        self.emit_comment(
            kda_stage_marker("kda_qkv_proj", f"gather {section} blocks={count}")
        )
        rows = list(range(count))
        self.vram_fill_zero(dst, rows=rows)
        self.tile_row_fma_fp_broadcast(
            dst, tall, consts.one,
            dst_rows=rows,
            src_rows=[(first + c) * stride + token for c in range(count)],
        )
        return dst

    def kda_split_projection_v0(
        self,
        *,
        projected: VRAMMatrixVar,
        shape: KdaShape,
        rows: int,
        prefix: str = "kda",
    ) -> dict[str, list[VRAMMatrixVar]]:
        """Views onto ``projected`` for each of q, k, v, gate and beta.

        Returns lists of per-column-block views, in the order
        ``kda_state_engine_step`` splits them: q, k, v, gate, beta. Every entry
        is an alias into ``projected`` -- nothing is copied.
        """
        mlen = self.mlen
        # The invariant the split rests on is per *head*, not per section:
        # feature `h*key_dim + b*mlen` must start on a block boundary. A section
        # width can be a clean multiple while key_dim is not -- 4 heads of
        # key_dim 6 gives key_width 24, divisible by mlen 8, with heads 1, 2 and
        # 3 all starting mid-block. kda_key_blocks already raises exactly this.
        kda_key_blocks(shape, mlen)
        value_width = shape.num_heads * shape.value_dim
        if value_width % mlen:
            raise ValueError(
                f"value_width ({value_width}) must be a multiple of mlen "
                f"({mlen}); a partial block cannot be named as a view"
            )
        out: dict[str, list[VRAMMatrixVar]] = {}
        for name, first, count in kda_projection_sections(shape, mlen):
            out[name] = [
                self.vram_column_block_view(
                    projected, first + i, name=f"{prefix}_{name}{i}", rows=rows
                )
                for i in range(count)
            ]
        return out

    def kda_layer_from_projected_v0(
        self,
        *,
        projected: VRAMMatrixVar,
        gathered: dict[str, VRAMMatrixVar],
        conv_state: dict[str, VRAMMatrixVar],
        conv_weight: dict[str, VRAMMatrixVar],
        conv_bias: dict[str, VRAMMatrixVar | None],
        conv_scratch: VRAMMatrixVar,
        mixer_buffers,
        shape: KdaShape,
        token: int = 0,
    ) -> VRAMMatrixVar:
        """One KDA layer for one decode token, from the projection to ``out``.

        Safe to call repeatedly in one program: the views it takes are named
        after ``projected``, so stacking layers needs nothing from the caller
        beyond giving each its own projection tile.

        Exactly ``kda_state_engine_step``'s boundary: takes ``projected``,
        returns the mixer's output. The input and output projections are
        ordinary Matrix work and stay with the caller -- the same line
        ``program_kda_mixer`` draws.

        ``gathered`` supplies a dense destination per section; ``conv_state`` is
        carried across tokens and updated in place.

        The order follows the reference: gather, convolve q, k and v, then the
        gates, then the recurrence. All three convolutions must finish before
        the mixer starts, because the mixer normalises ``q`` and ``k`` in place
        and they are the convolutions' output.
        """
        for section in ("q", "k", "v", "gate", "beta"):
            if section not in gathered:
                raise ValueError(f"gathered is missing {section!r}")
            self.kda_gather_projection_v0(
                projected=projected, dst=gathered[section], section=section,
                shape=shape, consts=mixer_buffers.consts, token=token,
                # From the projection's own name, which is unique per layer.
                # A fixed prefix collides on the second layer -- the guard in
                # `_named_view` catches it, but only after the caller has
                # already written the layer.
                name=f"{projected.display_name}_tall_t{token}",
            )

        widths = {
            "q": shape.projection_size,
            "k": shape.projection_size,
            "v": shape.num_heads * shape.value_dim,
        }
        outs = {"q": mixer_buffers.q, "k": mixer_buffers.k, "v": mixer_buffers.v}
        for section, channels in widths.items():
            self.kda_conv_step_v0(
                x_new=gathered[section], conv_state=conv_state[section],
                weight=conv_weight[section], bias=conv_bias.get(section),
                out=outs[section], scratch=conv_scratch,
                consts=mixer_buffers.consts, channels=channels,
                kernel=shape.conv_kernel,
            )

        mixer_buffers.gate = gathered["gate"]
        mixer_buffers.beta_logit = gathered["beta"]
        self.kda_beta_scalars_v0(
            beta_logit=mixer_buffers.beta_logit, beta_fp=mixer_buffers.beta_fp,
            consts=mixer_buffers.consts, shape=shape,
        )
        self.kda_mixer_step_v0(buffers=mixer_buffers, shape=shape)
        return mixer_buffers.out
