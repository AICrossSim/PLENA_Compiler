"""Mamba-2 SSD (state-space duality) chunked-scan lowering -- the prefill path.

The chunked form of the Mamba-2 recurrence splits the sequence into chunks of
``chunk_size`` and, per chunk, computes

    G       = C @ B^T                       [chunk, chunk]   (per group)
    L[i,j]  = exp(cs_i - cs_j) for i >= j    [chunk, chunk]   (per head, causal)
    Y_intra = (L o G) @ X                   [chunk, head_dim] (per head)
    h_new   = B^T @ X_d + h_prev * decay    [state, head_dim] (per head)
    Y_inter = C @ h_prev, row i scaled by exp(cs_i)
    Y       = Y_intra + Y_inter + D_h * X

where ``a_t = A_h * dt_t`` and ``cs = cumsum_t(a)`` within the chunk.

Why this maps well onto PLENA
-----------------------------
Three of the four matmuls drop onto the existing flash-attention templates with
no structural change: ``G = C @ B^T`` is ``asm_templates/flashattn/qkt.py``'s
``M_TMM`` loop with ``state_size`` substituted for ``head_dim``; ``Y_intra`` is
``pv.py``'s ``M_MM`` loop with ``X`` in place of ``V``; and the inter-chunk
recurrence is ``output.py``'s ``computing_o_code`` with the scalar source changed
from the softmax max-delta to the chunk segment sum -- and *simpler*, because
attention needs a different scalar per row while Mamba needs one scalar for the
whole ``[state_size, head_dim]`` block, so the ``S_LD_FP`` hoists out of the loop.
Dropping the online softmax removes about ten of the fourteen instructions in
attention's per-row inner loop.

The one structural obstacle
---------------------------
``M_MM`` and ``M_TMM`` both contract over the **VRAM operand's lane axis**
(``matrix_machine.rs``). Three of the four SSD products satisfy that natively, but
the state accumulation ``h[n,p] = sum_t B[t,n] * X[t,p]`` contracts over the
**row** axis of both operands, which neither expresses. There is no VRAM
transpose, no transpose instruction, and ``H_PREFETCH_M`` does not transpose
either -- ``dma.rs`` gathers contiguous runs at a stride, a strided row gather;
the sentence in ``doc/plena_isa_spec.md`` about stride mode describes a host-side
storage convention, not DMA behaviour.

The fix is to produce ``B^T`` at its source: :meth:`ssd_transposed_projection_v0`
stages a host-transposed weight into VRAM and the layer input into MRAM, then
emits ``M_TMM``. Nothing in the ISA forbids weight-in-VRAM / activation-in-MRAM;
it is purely a convention that no existing emitter happens to break. The cost is
bounded because only the *narrow* slices need it -- ``B^T`` is
``n_groups*state_size`` wide and ``dt^T`` is ``num_heads`` wide, while the wide
``x``/``z`` slices stay in their natural orientation.

What this path does NOT support
-------------------------------
Enforced by :meth:`Mamba2Shape.validate`, and worth stating up front because the
module reads as if it were general:

* ``chunk_size == head_dim == state_size == MLEN`` and ``d_inner/n_groups == MLEN``.
  The ``tile_row_*`` family works within ONE MLEN column block, and one
  ``S_MAP_FP_V`` moves exactly MLEN scalars, so anything wider leaves the later
  column blocks untouched and reads uninitialised FPRAM.
* Consequently **this path is not usable at the ANALYTIC target MLEN=2048.**
  ``ssd_decay_mask_v0`` holds one head's whole ``cs`` row in FPRAM, and FPRAM is a
  hardware-fixed 1024 slots, so ``chunk_size`` -- and therefore MLEN -- is capped
  near 1017. Lifting it means looping ``tile_col_idx`` in every emitter and
  blocking the cs row; neither is done. Note the analytic cost model in
  ``analytic_models/performance/mamba2_model.py`` models the *unrestricted* shape
  family, so its numbers describe a lowering this path currently rejects.
* ``hidden_size <= mram_tile_capacity * MLEN``.
  :meth:`ssd_transposed_projection_v0` contracts over ``hidden_size`` and
  ``vram_sub_projection_T_to`` has no K-split, so a wider hidden dimension is an
  MRAM allocator overflow rather than a shape error. The module docstring's cost
  argument bounds the OUTPUT dimension; the binding constraint is the contraction.
* ``batch_size == 1``. No emitter scales a row range or a state block by it.

Numerical choices
-----------------
* **The cumsum goes through the systolic array**, not a chained ``V_ADD_VV``.
  ``cs = a @ U`` against a constant lower-triangular-ones MRAM tile sums in the
  f32 accumulator (all four accumulators in ``matrix_machine.rs`` are f32) and
  rounds to bf16 exactly once at the ``M_MM_WO`` flush. A chained vector scan
  would re-quantise to bf16 at every step, and ``cs`` feeds an *exponent*: at
  ``|cs| ~ 30`` a 0.9% relative error is 0.27 absolute, i.e. a 31% error on the
  decay. See :meth:`ssd_chunk_cumsum_v0`.
* **The decay matrix is never factored** as ``exp(cs_i) * exp(-cs_j)``. That form
  saves about half the instructions and destroys the answer: the systolic array
  computes the full LxL product including the upper triangle, where the two
  factors multiply to ``e^{+60}`` and overflow bf16 *before* the causal mask is
  applied. Instead ``D = cs_i - cs_j`` is formed directly, clamped to ``<= 0``
  with ``V_MIN_VF`` (free, and exact on the causal half because ``cs`` is
  non-increasing), and only then exponentiated -- so every value fed to
  ``V_EXP_V`` is non-positive and every result lands in ``(0, 1]``.
* **Activation round trips use the BF16 keyvalue path**, never MX-FP8. See the
  module docstring of ``program_mamba_common.py``.
"""

from __future__ import annotations

from collections.abc import Sequence

from compiler.aten.plena.program_mamba_common import (
    Mamba2Shape,
    MambaFPConstants,
    mamba_stage_marker,
)
from compiler.aten.plena.vars import FPVar, InputVar, VRAMMatrixVar
from compiler.utils.load_config import load_toml_config

#: MRAM operand precision for a tensor the *host* staged into HBM. These arrive
#: through the ordinary MX-FP8 weight path that create_sim_env/create_mem_for_sim
#: writes, so reading them as BF16 would decode the wrong bytes -- garbage, which
#: shows up as NaN rather than as a small error.
HOST_STAGED = {"matrix_precision": "weights", "hbm_element_bytes": 1, "set_scale": True}

#: MRAM operand precision for an *activation* the program spilled to HBM and is
#: reloading.
#:
#: This selects the ``keyvalue`` precision CLASS, not a width. It is BF16 only if
#: the build configures ``HBM_M_KV_TYPE`` / ``HBM_V_KV_TYPE`` as a Plain BF16 type
#: -- exactly the requirement ``linear_projection_bf16`` states at
#: ``program_matrix_ops.py``. Under the MX-FP8 e4m3 KV types that
#: ``plena_settings.toml`` ships, ``hbm_element_bytes=2`` only doubles the address
#: stride while the data is still decoded as 1-byte e4m3 plus a scale stream, and
#: ``set_scale=False`` leaves the scale register holding whatever the previous
#: load left. Call :meth:`ProgramSSDMixin.require_bf16_kv_precision` to make that
#: requirement a hard failure rather than silent garbage.
#:
#: Why BF16 matters here and not for a KV cache: the SSM state is a multiplicative
#: accumulator carried across every chunk, so requantising it to e4m3's 3 mantissa
#: bits at each chunk boundary compounds without bound. A KV cache is written once
#: and never read-modify-written.
SPILLED_ACTIVATION = {"matrix_precision": "keyvalue", "hbm_element_bytes": 2, "set_scale": False}


class ProgramSSDMixin:
    """Mamba-2 chunked-scan (prefill) emitters."""

    # ========================================================================
    # Build-level precision requirement
    # ========================================================================

    def require_bf16_kv_precision_from_active_build(self) -> None:
        """Read the active TOML and check it, once per program.

        ``require_bf16_kv_precision`` puts the burden on a caller, and the
        result was that **nothing ever called it** -- not the SSD emitters, not
        the KDA ones, not a single test. That is the wrong division: the
        ``keyvalue`` precision class is chosen by the *emitter*, in
        ``SPILLED_ACTIVATION``, so the emitter is what knows the requirement
        exists. A caller cannot be expected to know that passing the default
        precision commits them to a build-level constraint.

        Cached on the instance: one spill-using emitter per layer would
        otherwise re-read and re-parse the TOML per chunk per head.
        """
        if getattr(self, "_bf16_kv_checked", False):
            return
        from compiler.aten.plena.compiler import _find_plena_settings_toml

        path = _find_plena_settings_toml()
        if path is None or not path.exists():
            raise ValueError(
                "The spilled-activation path needs the active PRECISION table to "
                "verify the KV types are Plain BF16, and no plena_settings.toml "
                "was found (set PLENA_SETTINGS_TOML). Refusing to proceed without "
                "having checked -- under an MX KV type the spills decode as e4m3 "
                "and produce a wrong answer rather than an error."
            )
        self.require_bf16_kv_precision(
            load_toml_config(path, "PRECISION", mode="TRANSACTIONAL")
        )
        self._bf16_kv_checked = True

    def require_bf16_kv_precision(self, settings: dict | None = None) -> None:
        """Fail unless the build configures the KV precision classes as Plain BF16.

        ``SPILLED_ACTIVATION`` selects the ``keyvalue`` class; the *width* comes
        from the active TOML. Under the shipped MX-FP8 KV types the spilled
        activations decode as e4m3 and the numerical argument in this module's
        docstring does not hold. Nothing detects that at runtime -- it surfaces as
        a wrong answer, not as an error -- so the check has to be explicit.

        `settings` is the parsed ``[<MODE>.PRECISION]`` table; when None the check
        reports that it could not verify rather than silently passing, because a
        check that cannot run must not look like a check that passed.
        """
        if settings is None:
            raise ValueError(
                "require_bf16_kv_precision needs the active PRECISION table; pass the "
                "parsed [<MODE>.PRECISION] section. Refusing to report success without "
                "having checked -- SPILLED_ACTIVATION is silently wrong under MX-FP8 KV "
                "types and produces a wrong answer rather than an error."
            )
        bad = []
        for key in ("HBM_M_KV_TYPE", "HBM_V_KV_TYPE"):
            node = settings.get(key, {})
            # The TOML spells this `format`, not `kind` -- see
            # plena_settings.toml's [*.PRECISION.*_TYPE] tables, where a Plain
            # entry is `format = "Plain"` with a `.DATA_TYPE` subtable and an Mx
            # entry is `format = "Mx"` with `.ELEM` and `.SCALE`. Reading `kind`
            # returned None for *every* configuration, so this rejected the
            # Plain BF16 build it exists to require as well as the MX one it
            # exists to refuse -- a check that could not pass, which is why
            # nothing called it.
            fmt = node.get("format") if isinstance(node, dict) else None
            if fmt != "Plain":
                bad.append(f"{key}={node}")
        if bad:
            raise ValueError(
                "The Mamba SSD path spills activations through the keyvalue precision class "
                "and requires it to be Plain BF16, but this build declares: "
                + "; ".join(bad)
                + ". Under an MX type the spilled operands decode as 1-byte e4m3 plus a "
                "scale stream while the address stride assumes 2 bytes -- garbage, silently. "
                "Either configure the KV types as Plain Fp(8,7) for this build, or pass "
                "precision=HOST_STAGED and accept e4m3 for the state (see the module "
                "docstring for why that compounds)."
            )

    # ========================================================================
    # Constant tiles
    # ========================================================================

    def ssd_lower_triangular_ones(self, chunk: int, *, name: str = "ssd_tri_ones") -> InputVar:
        """Declare the ``[chunk, chunk]`` lower-triangular-ones constant in HBM.

        Staged as an HBM constant and prefetched, *not* materialised on chip: the
        attention path's ``_build_causal_score_mask`` emits ``MLEN*MLEN`` fully
        unrolled ``S_ST_FP`` instructions, which is 4096 at MLEN=64 and 4.2
        million at the target MLEN=2048. The host writes the values; see
        :meth:`ssd_lower_triangular_ones_values`.
        """
        return self.input(name, (chunk, chunk))

    @staticmethod
    def ssd_lower_triangular_ones_values(chunk: int) -> list[list[float]]:
        """Host-side contents of :meth:`ssd_lower_triangular_ones`.

        ``U[s, t] = 1`` iff ``s <= t``. Used two ways, which is why it is one
        tile and not two: as the cumsum operator (``cs[t] = sum_{s<=t} a[s]``),
        and transposed-by-indexing as the causal mask on the score matrix.
        """
        return [[1.0 if s <= t else 0.0 for t in range(chunk)] for s in range(chunk)]

    # ========================================================================
    # Chunk cumsum
    # ========================================================================

    def _blocks(self, n: int) -> int:
        """Number of MLEN-wide tile blocks spanning `n` elements.

        The projection helpers index tensors in MLEN x MLEN *blocks*, not logical
        rows -- ``vram_sub_projection_to(v, i, m, j, t, r, c)`` computes one output
        tile from block row `i` of `v` and block column `j` of `m`. Getting this
        wrong is a `KeyError` deep inside the layout, not a wrong answer, but the
        helper keeps the arithmetic in one place.
        """
        return max(1, -(-n // self.mlen))

    def ssd_chunk_cumsum_v0(
        self,
        a_t: VRAMMatrixVar,
        tri_ones: InputVar,
        cs_t: VRAMMatrixVar,
        shape: Mamba2Shape,
        *,
        precision: dict | None = None,
    ):
        """``cs_t[h, t] = sum_{s <= t} a_t[h, s]``, all heads in one pass.

        `a_t` is ``[num_heads, chunk]`` -- heads on rows, time on lanes -- which is
        what the transposed ``dt`` projection produces. The sum along `t` is then a
        contraction over the VRAM operand's lane axis, exactly what ``M_MM`` does,
        against the lower-triangular-ones tile in MRAM. One matmul covers MLEN
        heads at a time.

        This replaces the missing prefix scan, and does it *better than* the
        instruction PLENA declares but does not implement. ``V_PS_V`` (0x31) would
        not help even if wired up: its micro-op is ``PREFIX_SCAN_V_ELEMENT``, an
        intra-row scan over a VLEN tile, so under any layout that keeps heads on
        lanes it would cumsum across heads, which is meaningless. Today it also
        assembles silently and then decodes to ``Invalid``, so emitting it is worse
        than not emitting it.

        Routing the sum through the systolic array is also the numerically right
        choice: all four accumulators in the matrix machine are f32, so the whole
        chunk sums in f32 and rounds to bf16 exactly once at the ``M_MM_WO``
        flush. A chained ``V_ADD_VV`` scan would re-quantise every step, and `cs`
        feeds an exponent, so its error is magnified rather than averaged.
        """
        precision = HOST_STAGED if precision is None else precision
        self.emit_comment(
            mamba_stage_marker(
                "mamba_chunk_cumsum", f"heads={shape.num_heads} chunk={shape.chunk_size}"
            )
        )
        head_blocks = self._blocks(shape.num_heads)
        time_blocks = self._blocks(shape.chunk_size)
        for hb in range(head_blocks):
            for tb in range(time_blocks):
                self.vram_sub_projection_to(
                    a_t,
                    hb,
                    tri_ones,
                    tb,
                    cs_t,
                    hb,
                    tb,
                    **precision,
                )
        return cs_t

    # ========================================================================
    # Decay mask
    # ========================================================================

    def ssd_decay_mask_v0(
        self,
        cs_t: VRAMMatrixVar,
        cs_fp: FPVar,
        decay: VRAMMatrixVar,
        causal: VRAMMatrixVar,
        consts: MambaFPConstants,
        shape: Mamba2Shape,
        *,
        head_row: int,
    ):
        """Build ``decay[i, j] = exp(min(cs_i - cs_j, 0)) * causal[i, j]`` for one head.

        `cs_t` is ``[num_heads, chunk]``; row `head_row` holds this head's
        cumulative sums with time on lanes. `cs_fp` receives that row via a single
        ``S_MAP_FP_V`` -- the whole reason that opcode exists. Without it each
        ``cs_i`` would cost a one-hot ``V_MUL_VV`` + ``V_RED_SUM`` + ``S_ST_FP``
        triple, roughly 15,000 instructions per chunk at 80 heads just to move
        5,120 numbers from the lane domain to the scalar domain -- more than the
        matmuls those numbers feed.

        Then per output row `i`: copy ``cs_j`` across, reverse-subtract the scalar
        ``cs_i`` to get ``cs_i - cs_j``, clamp to ``<= 0``, exponentiate, and
        multiply by the causal 0/1 row.

        The clamp is what makes this safe. ``cs`` is a cumsum of ``A_h * dt``
        with ``A_h < 0``, so it is non-increasing and ``cs_i - cs_j <= 0``
        wherever ``i >= j`` -- the clamp is a no-op on the causal half and merely
        stops the (about-to-be-masked) upper triangle from overflowing ``V_EXP_V``.
        """
        chunk = shape.chunk_size
        if cs_fp.size < chunk:
            raise ValueError(f"cs_fp needs {chunk} slots for one chunk row, has {cs_fp.size}")
        self.emit_comment(mamba_stage_marker("mamba_decay_mask", f"head_row={head_row}"))

        # One instruction: whole cs row -> FPRAM.
        self.tile_row_to_fpram(cs_t, cs_fp, rows=[head_row])

        for i in range(chunk):
            self.mamba_row_copy(decay, i, cs_t, head_row)
            # decay[i] = cs_i - cs_j  (rorder=1 reverse subtract)
            self.tile_row_sub_fp_broadcast(decay, cs_fp, rows=[i], fpram_offset=i, reverse=True)
            self.tile_row_min_fp_broadcast(decay, consts.zero, rows=[i])
            self.tile_row_exp(decay, rows=[i])
            self.mamba_row_mul(decay, i, causal, i)
        return decay

    # ========================================================================
    # Transposed projection
    # ========================================================================

    def ssd_transposed_projection_v0(
        self,
        weight_t_vram: VRAMMatrixVar,
        x_mram: InputVar,
        target: VRAMMatrixVar,
        shape: Mamba2Shape,
        *,
        out_rows: int,
        stage: str,
        precision: dict | None = None,
    ):
        """``target[n, t] = sum_k weight_t_vram[n, k] * x_mram[t, k]`` via ``M_TMM``.

        The operand roles are inverted relative to every other emitter in the
        repo: the *weight* (host-transposed, so ``[out, hidden]``) sits in VRAM
        and the layer *input* sits in MRAM. That is what makes ``B^T`` and
        ``dt^T`` -- with the sequence axis on lanes -- expressible at all; see the
        module docstring. Nothing in the ISA forbids it: ``H_PREFETCH_V`` loads
        any HBM region into VRAM and ``H_PREFETCH_M`` loads any HBM region into
        MRAM.

        Use this only for the narrow slices (``B``, ``dt``). The wide ``x``/``z``
        slices should go through the ordinary ``linear_projection``.
        """
        precision = SPILLED_ACTIVATION if precision is None else precision
        if precision is SPILLED_ACTIVATION:
            self.require_bf16_kv_precision_from_active_build()
        self.emit_comment(mamba_stage_marker(stage, f"transposed, out_rows={out_rows}"))
        out_blocks = self._blocks(out_rows)
        time_blocks = self._blocks(shape.chunk_size)
        for nb in range(out_blocks):
            for tb in range(time_blocks):
                self.vram_sub_projection_T_to(
                    weight_t_vram,
                    nb,
                    x_mram,
                    tb,
                    target,
                    nb,
                    tb,
                    **precision,
                )
        return target

    # ========================================================================
    # One chunk, one head
    # ========================================================================

    def ssd_chunk_head_v0(
        self,
        *,
        b_chunk: InputVar,
        c_chunk: VRAMMatrixVar,
        x_chunk: InputVar,
        decay: VRAMMatrixVar,
        scores: VRAMMatrixVar,
        y_out: VRAMMatrixVar,
        shape: Mamba2Shape,
        head_block_base: int = 0,
        precision: dict | None = None,
    ):
        """``Y_intra = (decay o (C @ B^T)) @ X`` for one head of one chunk.

        Two matmuls with an elementwise multiply between them, all reusing the
        flash-attention shapes:

        * ``scores = C @ B^T`` -- ``M_TMM`` with `C` natural in VRAM and `B`
          natural in MRAM (``qkt.py``'s form with ``state_size`` for ``head_dim``).
          `B` has to be spilled to HBM and re-prefetched because MRAM is writable
          only by ``H_PREFETCH_M``; ``qkt.py`` already does exactly this for `K`.
        * ``scores *= decay`` -- the per-head fan-out point. ``scores`` is shared
          across a group (``B`` and ``C`` are group-shared) but ``decay`` is
          per-head, so this multiply is what makes the result head-specific.
        * ``y_out = scores @ X`` -- ``M_MM``, ``pv.py``'s form with `X` for `V`.
        """
        chunk = shape.chunk_size
        precision = SPILLED_ACTIVATION if precision is None else precision
        if precision is SPILLED_ACTIVATION:
            self.require_bf16_kv_precision_from_active_build()
        self.emit_comment(mamba_stage_marker("mamba_intra_chunk", f"chunk={chunk}"))
        i_blocks = self._blocks(chunk)
        j_blocks = self._blocks(chunk)
        p_blocks = self._blocks(shape.head_dim)

        # scores = C @ B^T
        for ib in range(i_blocks):
            for jb in range(j_blocks):
                self.vram_sub_projection_T_to(
                    c_chunk,
                    ib,
                    b_chunk,
                    jb,
                    scores,
                    ib,
                    jb,
                    **precision,
                )

        # scores *= decay -- the per-head fan-out point.
        self.tile_row_mul(scores, decay, rows=list(range(chunk)))

        # Y_intra = scores @ X
        for ib in range(i_blocks):
            for pb in range(p_blocks):
                self.vram_sub_projection_to(
                    scores,
                    ib,
                    x_chunk,
                    pb,
                    y_out,
                    head_block_base + ib,
                    pb,
                    **precision,
                )
        return y_out

    # ========================================================================
    # Inter-chunk state
    # ========================================================================

    def ssd_state_update_v0(
        self,
        *,
        state: VRAMMatrixVar,
        b_t_chunk: VRAMMatrixVar,
        x_d_chunk: InputVar,
        decay_fp: FPVar,
        shape: Mamba2Shape,
        decay_offset: int = 0,
        precision: dict | None = None,
    ):
        """``state = state * exp(sum_t a_t) + B^T @ X_d`` for one head.

        The decay is a single per-head scalar applied to the whole
        ``[state_size, head_dim]`` block, so the ``S_LD_FP`` hoists out of the row
        loop entirely -- strictly cheaper than the attention analogue in
        ``output.py``, which needs a different scalar per query row.

        ``B^T @ X_d`` is the contraction that motivated the transposed projection:
        `b_t_chunk` must be ``[state_size, chunk]`` with time on lanes, produced by
        :meth:`ssd_transposed_projection_v0`, and `x_d_chunk` the decay-scaled
        ``[chunk, head_dim]`` activation spilled to HBM.
        """
        precision = SPILLED_ACTIVATION if precision is None else precision
        if precision is SPILLED_ACTIVATION:
            self.require_bf16_kv_precision_from_active_build()
        self.emit_comment(mamba_stage_marker("mamba_state_update", f"state_size={shape.state_size}"))
        n_blocks = self._blocks(shape.state_size)
        p_blocks = self._blocks(shape.head_dim)

        # The projection *overwrites* its target tile (M_MM accumulates in the
        # systolic array, then M_MM_WO writes and clears), so the new contribution
        # has to land in scratch and be added -- writing straight into `state`
        # would drop the decayed history.
        contrib = self.alloc("ssd_state_contrib", shape.state_size, shape.head_dim, strict=False)
        try:
            for nb in range(n_blocks):
                for pb in range(p_blocks):
                    self.vram_sub_projection_to(
                        b_t_chunk,
                        nb,
                        x_d_chunk,
                        pb,
                        contrib,
                        nb,
                        pb,
                        **precision,
                    )
            state_rows = list(range(state.shape[0]))
            self.tile_row_mul_fp_broadcast(state, decay_fp, rows=state_rows, fpram_offset=decay_offset)
            self.vram_add(state, contrib, num_rows=min(state.shape[0], contrib.shape[0]))
        finally:
            self.free_tensor(contrib)
        return state

    def ssd_inter_chunk_output_v0(
        self,
        *,
        c_chunk: VRAMMatrixVar,
        state_prev: InputVar,
        y_out: VRAMMatrixVar,
        cs_fp: FPVar,
        shape: Mamba2Shape,
        head_row_base: int = 0,
        precision: dict | None = None,
    ):
        """``y_out[i] += (C @ h_prev)[i] * exp(cs_i)`` for one head.

        ``M_MM`` with `C` natural in VRAM (state index on lanes) and `h_prev`
        prefetched into MRAM as ``[state_size, head_dim]``. That orientation is
        exactly what :meth:`ssd_state_update_v0` produces, which is why the two
        compose without a transpose.

        The per-row ``exp(cs_i)`` scale comes from the FPRAM copy of the cs row,
        so it costs one ``S_LD_FP`` + one ``V_MUL_VF`` per row and no extraction.
        """
        chunk = shape.chunk_size
        precision = SPILLED_ACTIVATION if precision is None else precision
        if precision is SPILLED_ACTIVATION:
            self.require_bf16_kv_precision_from_active_build()
        self.emit_comment(mamba_stage_marker("mamba_inter_chunk", f"chunk={chunk}"))
        i_blocks = self._blocks(chunk)
        p_blocks = self._blocks(shape.head_dim)
        scratch = self.alloc("ssd_inter_scratch", chunk, shape.head_dim, strict=False)
        try:
            for ib in range(i_blocks):
                for pb in range(p_blocks):
                    self.vram_sub_projection_to(
                        c_chunk,
                        ib,
                        state_prev,
                        pb,
                        scratch,
                        ib,
                        pb,
                        **precision,
                    )
            # scale row i by exp(cs_i): the cs row is already in FPRAM, so this is
            # a plain per-row scalar walk with no extraction.
            self.tile_row_exp_fpram_scale(scratch, cs_fp, rows=list(range(chunk)))
            self.vram_add(y_out, scratch, dst_row_offset=head_row_base, src_row_offset=0, num_rows=chunk)
        finally:
            self.free_tensor(scratch)
        return y_out

    def tile_row_exp_fpram_scale(self, target: VRAMMatrixVar, cs_fp: FPVar, rows: Sequence[int]):
        """``target[i] *= exp(cs_fp[i])``.

        The exponential is taken in the scalar unit (``S_EXP_FP``) because the
        value is already a scalar; doing it in the vector unit would mean
        splatting it to a row first.
        """
        from compiler.aten.isa_builder import IsaBuilder, fp, gp

        resolved = list(rows)
        gp_regs = self._reg.allocate_gp(2)
        gp_addr, gp_row = gp_regs
        try:
            base = self._tile_addr(target.name)
            asm = IsaBuilder().comment(f"Row scale by exp(FPRAM[{cs_fp.address}:+{len(resolved)}])")
            for i, row in enumerate(resolved):
                asm.instr("S_ADDI_INT", gp(gp_addr), gp(0), cs_fp.address + i)
                asm.instr("S_LD_FP", fp(1), gp(gp_addr), 0)
                asm.instr("S_EXP_FP", fp(1), fp(1))
                asm.instr("S_ADDI_INT", gp(gp_row), gp(0), base + row * self.mlen)
                asm.instr("V_MUL_VF", gp(gp_row), gp(gp_row), fp(1), 0)
            return self._emit(asm)
        finally:
            self._reg.free_gp(gp_regs)


__all__ = ["ProgramSSDMixin"]
