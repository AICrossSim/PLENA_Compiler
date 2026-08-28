"""Shared Mamba / Mamba-2 (selective state-space model) program-builder helpers.

This module holds the pieces both the prefill (`program_ssd.py`) and the decode
(`program_ssm_recurrent.py`) lowerings need: the shape contract, the FPRAM
constant block, the causal depthwise conv1d, the `dt` activation, SiLU, and the
gated RMSNorm. The two paths are deliberately separate modules because on PLENA
they are *not* the same kernel at a different sequence length -- they use
different VRAM layouts and different opcode mixes, and because `C_LOOP_START`
takes a compile-time immediate trip count and the ISA has no branch, they are two
separately compiled programs regardless.

Scope: **Mamba-2 only**. Mamba-1's `A` is a `[d_inner, state_size]` diagonal,
which would need a `state_size`-wide vector replicated across `head_dim` rows and
applied with `V_MUL_VV`, because PLENA's only broadcast is scalar-to-vector
(`v_broadcast_en` in `doc/operation.svh` is wired solely to the `*_VF` path).
Mamba-2's `A` is one scalar per head, which is exactly what `V_MUL_VF` plus the
FPRAM row map already express.

Hardware constraints that shaped this lowering
----------------------------------------------
* **MRAM is write-only from HBM** (`doc/memory_layout.md`), so every
  activation-times-activation product costs an `H_STORE_V` + `H_PREFETCH_M` round
  trip. All such round trips here take the BF16 `keyvalue` path
  (`hbm_element_bytes=2`), never the default MX-FP8 `weights` path: the SSM state
  is a multiplicative accumulator carried across every chunk, and requantising it
  to 3 mantissa bits at each chunk boundary compounds without bound. That is the
  sharp difference from a KV cache, which is written once and never
  read-modify-written, and is why e4m3 is fine there and not here.
* **`V_MUL_VF` broadcasts one FP register across all VLEN lanes.** Per-head
  scalar broadcasts (the decay `dA_h`, the skip `D_h`, the per-row `exp(cs_i)`)
  are only correct when one VRAM row does not straddle two heads -- i.e. when
  `head_dim` is a multiple of MLEN. :meth:`Mamba2Shape.validate` enforces that
  rather than letting one head's scalar silently apply to its neighbours.
* **There is no logarithm and no float-to-int conversion**, so `softplus` has no
  exact software lowering and no lookup-table lowering either. It uses the
  `V_SOFTPLUS_V` opcode added alongside this substrate.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from compiler.aten.plena.vars import FPVar, VRAMMatrixVar

# ============================================================================
# Stage markers
# ============================================================================

#: Prefix of the explicit stage marker comment, shared with the routed-MoE
#: substrate. The emulator's stage profiler keys on this exact string.
MAMBA_STAGE_MARKER_PREFIX = "@stage="

#: Every Mamba stage name the emulator's ``StageKind`` understands.
#:
#: Kept in lockstep with ``transactional_emulator/src/stage_profile.rs``; the
#: cross-repo test in ``aten/tests/test_mamba_stage_contract.py`` pins it. Marker
#: emission is validated against this set so a typo fails at ASM-gen time instead
#: of silently collapsing a region into ``other``.
MAMBA_STAGES: frozenset[str] = frozenset(
    {
        "mamba_in_proj",
        "mamba_conv1d",
        "mamba_dt",
        "mamba_chunk_cumsum",
        "mamba_decay_mask",
        "mamba_intra_chunk",
        "mamba_state_update",
        "mamba_inter_chunk",
        "mamba_skip",
        "mamba_gated_norm",
        "mamba_out_proj",
        "mamba_state_load",
        "mamba_state_store",
    }
)


def mamba_stage_marker(stage: str, detail: str = "") -> str:
    """Format the explicit stage marker comment for ``stage``.

    Markers are *authoritative and sticky*: once a program contains any marker,
    the emulator stops applying its legacy substring rules and attributes every
    instruction to the most recent marker. A marker must therefore be emitted
    whenever the stage changes, and must not be emitted mid-stage.

    The corollary is that work done inside a general-purpose helper called from a
    marked region -- ``linear_projection``'s HBM weight prefetch, for instance --
    bills to the enclosing marker rather than to a stage of its own.
    """
    if stage not in MAMBA_STAGES:
        raise ValueError(f"unknown Mamba stage {stage!r}; expected one of {sorted(MAMBA_STAGES)}")
    return f"{MAMBA_STAGE_MARKER_PREFIX}{stage}" + (f" {detail}" if detail else "")


# ============================================================================
# Shape contract
# ============================================================================


@dataclass(frozen=True)
class Mamba2Shape:
    """Resolved Mamba-2 mixer dimensions for one layer.

    Field names mirror HuggingFace's ``Mamba2Config`` so a config JSON maps across
    without a translation table.
    """

    hidden_size: int
    num_heads: int
    head_dim: int
    state_size: int
    n_groups: int
    conv_kernel: int
    chunk_size: int
    seq_len: int
    batch_size: int = 1
    #: softplus(dt) is clamped to this range, matching HF's ``time_step_limit``.
    time_step_min: float = 0.0
    time_step_max: float = float("inf")

    @property
    def d_inner(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def group_state(self) -> int:
        """Width of B or C: ``n_groups * state_size``."""
        return self.n_groups * self.state_size

    @property
    def conv_dim(self) -> int:
        """Width of the tensor the depthwise conv1d runs over: ``[x, B, C]``."""
        return self.d_inner + 2 * self.group_state

    @property
    def in_proj_out(self) -> int:
        """Width of the fused in_proj output: ``[z, x, B, C, dt]``."""
        return 2 * self.d_inner + 2 * self.group_state + self.num_heads

    @property
    def heads_per_group(self) -> int:
        return self.num_heads // self.n_groups

    @property
    def num_chunks(self) -> int:
        return math.ceil(self.seq_len / self.chunk_size)

    @property
    def slice_offsets(self) -> dict[str, tuple[int, int]]:
        """Column offset and width of each slice within the fused in_proj output."""
        z0 = 0
        x0 = z0 + self.d_inner
        b0 = x0 + self.d_inner
        c0 = b0 + self.group_state
        dt0 = c0 + self.group_state
        return {
            "z": (z0, self.d_inner),
            "x": (x0, self.d_inner),
            "B": (b0, self.group_state),
            "C": (c0, self.group_state),
            "dt": (dt0, self.num_heads),
        }

    def validate(self, mlen: int) -> None:
        """Reject shapes this lowering would silently get wrong.

        Every check corresponds to a real failure mode, not defensive pedantry --
        see the module docstring.
        """
        # head_dim must not be SMALLER than MLEN either: V_MUL_VF broadcasts one FP
        # register across all VLEN lanes, so a row spanning two heads would apply one
        # head's decay/skip scalar to its neighbour, silently and with no diagnostic.
        if self.chunk_size != mlen:
            raise ValueError(
                f"chunk_size ({self.chunk_size}) must equal MLEN ({mlen}) in this lowering. "
                "The tile_row_* family is per-MLEN-column-block (it defaults tile_col_idx=0 "
                "and addresses rows as base + row*MLEN inside block 0), and one S_MAP_FP_V "
                "moves exactly MLEN scalars, so a wider chunk would leave the columns past "
                "the first block untouched and read uninitialised FPRAM for cs_i -- "
                "half-processed rather than uniformly wrong, which is far harder to notice. "
                "Mamba-2's default chunk_size is 256; supporting it needs the emitters to "
                "loop tile_col_idx over ceil(cols/MLEN), which is deliberately not done yet."
            )
        if self.head_dim != mlen:
            raise ValueError(
                f"head_dim ({self.head_dim}) must equal MLEN ({mlen}) in this lowering, for "
                "the same per-column-block reason as chunk_size above; head_dim > MLEN would "
                "silently process only the first MLEN lanes of each head."
            )
        if self.state_size != mlen:
            raise ValueError(
                f"state_size ({self.state_size}) must equal MLEN ({mlen}) in this lowering, "
                "for the same per-column-block reason as chunk_size above."
            )
        if self.d_inner // self.n_groups != mlen:
            raise ValueError(
                f"gated RMSNorm reduces one MLEN-wide VRAM row but divides by "
                f"d_inner/n_groups ({self.d_inner // self.n_groups}); those must match "
                f"(MLEN={mlen}) or the normalisation constant is wrong. Accumulating "
                "V_RED_SUM across group_width/MLEN rows would lift this."
            )
        if self.n_groups <= 0 or self.num_heads % self.n_groups != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be divisible by n_groups ({self.n_groups}); "
                "B and C are shared across the heads of a group."
            )
        if self.conv_kernel < 1:
            raise ValueError(f"conv_kernel must be >= 1, got {self.conv_kernel}")
        if self.seq_len < 1:
            raise ValueError(f"seq_len must be >= 1, got {self.seq_len}")
        if self.time_step_min > self.time_step_max:
            raise ValueError(
                f"time_step_limit is inverted: min={self.time_step_min} > max={self.time_step_max}"
            )
        if self.batch_size != 1:
            raise ValueError(
                f"batch_size ({self.batch_size}) must be 1: this lowering emits one sequence "
                "per program. No emitter scales a row range or a state block by it, so "
                "accepting it would silently produce batch-1 code. Batch by invoking the "
                "program once per sequence."
            )
        # FPRAM is a hardware-fixed 1024 slots (FPRAMAllocator's default, never
        # overridden), and ssd_decay_mask_v0 needs chunk_size of them for one head's
        # cs row -- there is no column-block loop, so the whole row must fit at once.
        # With the constant block on top, that caps MLEN well below the ANALYTIC
        # target of 2048. Reject it as a shape with a reason, rather than letting it
        # surface as an allocator MemoryError thousands of ISA lines later.
        fpram_slots = 1024
        constants = 7
        if self.chunk_size + constants > fpram_slots:
            raise ValueError(
                f"chunk_size ({self.chunk_size}) plus the {constants}-slot Mamba constant "
                f"block exceeds the {fpram_slots}-slot FPRAM. ssd_decay_mask_v0 holds one "
                "head's whole cs row in FPRAM (one S_MAP_FP_V, no column-block loop), so "
                "chunk_size is bounded by FPRAM. Since validate() also requires "
                "chunk_size == MLEN, this is equivalently an MLEN ceiling: the SSD prefill "
                "path is not usable at the ANALYTIC target MLEN=2048."
            )


@dataclass(frozen=True)
class MambaFPConstants:
    """The FPRAM scalars every Mamba emitter needs.

    Allocated once per layer by :meth:`ProgramMambaCommonMixin.mamba_fp_constants`
    rather than per emitter, because ``FPRAMAllocator`` bump-allocates and every
    re-allocation would walk further into FP_MEM.
    """

    zero: FPVar
    one: FPVar
    neg_one: FPVar
    dt_min: FPVar
    dt_max: FPVar
    #: 1 / normalized_group_width, for the gated RMSNorm.
    reci_group: FPVar
    eps: FPVar

    def as_list(self) -> list[FPVar]:
        return [self.zero, self.one, self.neg_one, self.dt_min, self.dt_max, self.reci_group, self.eps]


class ProgramMambaCommonMixin:
    """Emitters shared by the Mamba-2 prefill and decode lowerings."""

    # ========================================================================
    # Constants
    # ========================================================================

    def mamba_fp_constants(
        self,
        shape: Mamba2Shape | None = None,
        *,
        name_prefix: str = "mamba",
    ) -> MambaFPConstants:
        """Allocate (but do not initialise) the layer's FPRAM scalar block.

        ``shape`` is accepted for symmetry with :meth:`mamba_fp_constant_values`
        and is unused here -- the block is seven slots regardless of geometry.
        It is optional so callers with no Mamba shape (the KDA path) do not have
        to fabricate one; before this it was ``Mamba2Shape.__new__(Mamba2Shape)``,
        an uninitialised frozen dataclass that would ``AttributeError`` the day
        this method started reading a field.

        The values come from the host through the FP_MEM preload image, the same
        way the attention path seeds ``attn_scale`` / ``-inf`` / ``eps``. The
        returned ``FPVar.address`` values tell the host where to write them; see
        :meth:`mamba_fp_constant_values` for the matching value list.
        """
        return MambaFPConstants(
            zero=self.fp_var(f"{name_prefix}_zero"),
            one=self.fp_var(f"{name_prefix}_one"),
            neg_one=self.fp_var(f"{name_prefix}_neg_one"),
            dt_min=self.fp_var(f"{name_prefix}_dt_min"),
            dt_max=self.fp_var(f"{name_prefix}_dt_max"),
            reci_group=self.fp_var(f"{name_prefix}_reci_group"),
            eps=self.fp_var(f"{name_prefix}_eps"),
        )

    @staticmethod
    def mamba_fp_constant_values(
        shape: Mamba2Shape,
        *,
        norm_group_width: int | None = None,
        eps: float = 1e-5,
    ) -> list[float]:
        """Host-side values matching :meth:`mamba_fp_constants`, in slot order.

        Kept next to the allocator so the two cannot drift: a mismatch here is a
        silent wrong answer, because FPRAM contents are never checked at runtime.
        """
        group_width = norm_group_width if norm_group_width is not None else shape.d_inner // shape.n_groups
        if group_width <= 0:
            raise ValueError(f"norm_group_width must be positive, got {group_width}")
        dt_max = shape.time_step_max
        # bf16's finite maximum; +inf would poison V_MIN_VF's clamp.
        if not math.isfinite(dt_max):
            dt_max = 3.3895313892515355e38
        return [0.0, 1.0, -1.0, shape.time_step_min, dt_max, 1.0 / group_width, eps]

    # ========================================================================
    # Single-row primitives
    #
    # The `tile_row_*` family operates on matching row indices across two
    # matrices. Mamba's conv and state sweeps need *cross-row* moves (out[s] from
    # x[s-3]), which `vram_add`/`vram_mul` already express through their row
    # offsets. These three wrappers name the intent.
    # ========================================================================

    def mamba_row_copy(self, dst: VRAMMatrixVar, dst_row: int, src: VRAMMatrixVar, src_row: int):
        """``dst[dst_row] = src[src_row]``.

        There is no VRAM->VRAM move opcode, so this is zero-then-add -- the
        established idiom (``program_routed_moe.py`` uses the same one).
        """
        self.vram_fill_zero(dst, rows=[dst_row])
        self.vram_add(dst, src, dst_row_offset=dst_row, src_row_offset=src_row, num_rows=1)

    def mamba_row_add(self, dst: VRAMMatrixVar, dst_row: int, src: VRAMMatrixVar, src_row: int):
        """``dst[dst_row] += src[src_row]``."""
        self.vram_add(dst, src, dst_row_offset=dst_row, src_row_offset=src_row, num_rows=1)

    def mamba_row_mul(self, dst: VRAMMatrixVar, dst_row: int, src: VRAMMatrixVar, src_row: int):
        """``dst[dst_row] *= src[src_row]``."""
        self.vram_mul(dst, src, dst_row_offset=dst_row, src_row_offset=src_row, num_rows=1)

    def mamba_block_copy(
        self,
        dst: VRAMMatrixVar,
        src: VRAMMatrixVar,
        *,
        dst_row_offset: int = 0,
        src_row_offset: int = 0,
        num_rows: int | None = None,
    ):
        """Zero-then-add copy of a contiguous row range between two matrices."""
        rows = num_rows if num_rows is not None else src.shape[0]
        self.vram_fill_zero(dst, rows=list(range(dst_row_offset, dst_row_offset + rows)))
        self.vram_add(
            dst, src, dst_row_offset=dst_row_offset, src_row_offset=src_row_offset, num_rows=rows
        )

    # ========================================================================
    # SiLU
    # ========================================================================

    def mamba_silu_v0(
        self,
        target: VRAMMatrixVar,
        scratch: VRAMMatrixVar,
        consts: MambaFPConstants,
        *,
        marker: str,
        rows: Sequence[int] | None = None,
    ):
        """In-place ``target = target * sigmoid(target)``.

        `marker` is required and is emitted verbatim. This emitter serves two
        stage vocabularies -- Mamba's and KDA's -- so there is no default that
        is right for both, and the one it used to carry (``mamba_gated_norm``)
        was wrong for every KDA caller. Markers are sticky, so a caller that
        accepted the default billed its SiLU *and everything it emitted
        afterwards* to a Mamba stage.

        Build it with ``mamba_stage_marker(stage, "silu")`` or
        ``kda_stage_marker(stage, "silu")`` at the call site, which is the only
        place that knows which vocabulary applies.

        Five vector ops per row after the copy, the same shape as
        ``asm_templates/silu_asm.py``: negate, exp, +1, reciprocal, multiply.
        `scratch` must be a distinct VRAM matrix of the same width; the sigmoid is
        built there so `target` still holds `x` for the final multiply.

        The tails are safe without an explicit clamp: ``V_EXP_V`` saturates its
        input at [-88, 88], so for very negative `x` the reciprocal underflows
        toward zero and ``silu(x) -> 0``, which is the correct limit.
        """
        resolved = list(range(target.shape[0])) if rows is None else list(rows)
        self.emit_comment(marker)
        self.mamba_block_copy(scratch, target, num_rows=target.shape[0])
        self.tile_row_mul_fp_broadcast(scratch, consts.neg_one, rows=resolved)
        self.tile_row_exp(scratch, rows=resolved)
        self.tile_row_add_fp_broadcast(scratch, consts.one, rows=resolved)
        self.tile_row_reci(scratch, rows=resolved)
        self.tile_row_mul(target, scratch, rows=resolved)
        return target

    # ========================================================================
    # dt activation
    # ========================================================================

    def mamba_dt_activation_v0(
        self,
        dt: VRAMMatrixVar,
        dt_bias: VRAMMatrixVar | None,
        consts: MambaFPConstants,
        shape: Mamba2Shape,
        *,
        rows: Sequence[int] | None = None,
    ):
        """In-place ``dt = clamp(softplus(dt + dt_bias), time_step_limit)``.

        `dt_bias` is a per-head static parameter and must already be resident in
        VRAM with the same layout as `dt`. Pass None when the host has folded the
        bias into the in_proj weight.

        The clamp reuses ``V_MAX_VF`` / ``V_MIN_VF``, the opcodes added for the
        GPT-OSS expert clamp -- byte-for-byte the same shape.
        """
        resolved = list(range(dt.shape[0])) if rows is None else list(rows)
        self.emit_comment(mamba_stage_marker("mamba_dt", "softplus + clamp"))
        if dt_bias is not None:
            # dt_bias is ONE row broadcast across every row of dt. `tile_row_add`
            # would be wrong here: it pairs matching row indices, so with a
            # single-row bias every row past the first would read an unallocated
            # source row and produce NaN -- silently, and only in the rows past
            # the first, which is easy to miss in an aggregate error metric.
            if dt_bias.shape[0] == 1:
                for row in resolved:
                    self.vram_add(dt, dt_bias, dst_row_offset=row, src_row_offset=0, num_rows=1)
            elif dt_bias.shape[0] > max(resolved):
                self.tile_row_add(dt, dt_bias, rows=resolved)
            else:
                raise ValueError(
                    f"dt_bias has {dt_bias.shape[0]} rows; needs either 1 (broadcast) or more "
                    f"than {max(resolved)} -- tile_row_add pairs dst row r with SRC row r, not "
                    "with the i-th source row, so a non-zero-based `rows` needs the bias to "
                    "cover those same indices"
                )
        self.tile_row_softplus(dt, rows=resolved)
        # Skip degenerate clamps. softplus output is already strictly positive, so
        # a max against 0.0 is a no-op that would only cost instructions.
        if shape.time_step_min > 0.0:
            self.tile_row_max_fp_broadcast(dt, consts.dt_min, rows=resolved)
        if math.isfinite(shape.time_step_max):
            self.tile_row_min_fp_broadcast(dt, consts.dt_max, rows=resolved)
        return dt

    # ========================================================================
    # Causal depthwise conv1d
    # ========================================================================

    def mamba_conv1d_v0(
        self,
        x: VRAMMatrixVar,
        conv_weight: VRAMMatrixVar,
        conv_bias: VRAMMatrixVar | None,
        out: VRAMMatrixVar,
        scratch: VRAMMatrixVar,
        shape: Mamba2Shape,
        *,
        num_rows: int | None = None,
        history_rows: int = 0,
    ):
        """Causal depthwise conv1d of width ``conv_kernel`` along the sequence.

        Layout contract: `x` holds one VRAM row per timestep with **sequence on
        the row axis** -- which is what ``doc/memory_layout.md``'s
        ``[h//VLEN, b, s, VLEN]`` gives. A shift along `t` is then a pure row
        offset (free), and the per-channel tap weight `w[k, :]` is exactly a
        VLEN-wide ``V_MUL_VV`` operand. Depthwise conv is the one kernel this
        layout is perfect for.

        Deliberately *not* routed through ``aten/ops/plena/conv_ops.py``: that
        path is a dense im2col which hardcodes a square KxK spatial patch, has no
        padding support (so it cannot express the causal left pad), and would
        materialise a dense ``(M, C_in*K*K)`` GEMM against a block-diagonal
        weight, wasting `conv_dim`-fold MACs.

        ``V_MUL_VF`` is the wrong tool for the tap weight: it broadcasts one FP
        register across every lane, but the conv weight varies *per channel*, i.e.
        per lane. Hence `conv_weight` as a VRAM matrix and ``V_MUL_VV``.

        `history_rows` is how many leading rows of `x` are carried-over history
        from a previous call rather than new timesteps; outputs are produced for
        rows ``[history_rows, history_rows + num_rows)``. With `history_rows == 0`
        the first `conv_kernel - 1` outputs see the causal zero pad.
        """
        k = shape.conv_kernel
        rows = num_rows if num_rows is not None else x.shape[0] - history_rows
        if conv_weight.shape[0] < k:
            raise ValueError(f"conv_weight needs at least conv_kernel={k} rows, got {conv_weight.shape[0]}")
        if history_rows + rows > x.shape[0]:
            raise ValueError(
                f"conv1d would read past x: history_rows={history_rows} + num_rows={rows} "
                f"> x.shape[0]={x.shape[0]}"
            )
        if rows > out.shape[0]:
            raise ValueError(f"conv1d output needs {rows} rows, out has {out.shape[0]}")

        self.emit_comment(mamba_stage_marker("mamba_conv1d", f"k={k} rows={rows} history={history_rows}"))
        for s in range(rows):
            self.vram_fill_zero(out, rows=[s])
            for j in range(k):
                # Tap j reads x[t - (k-1) + j]; taps before the sequence start are
                # the causal zero pad and are skipped rather than masked, because
                # V_MASK is HLEN-granular and cannot gate a single row anyway.
                src = history_rows + s - (k - 1) + j
                if src < 0:
                    continue
                self.mamba_row_copy(scratch, 0, x, src)
                self.mamba_row_mul(scratch, 0, conv_weight, j)
                self.mamba_row_add(out, s, scratch, 0)
            if conv_bias is not None:
                self.mamba_row_add(out, s, conv_bias, 0)
        return out

    # ========================================================================
    # Gated RMSNorm
    # ========================================================================

    def mamba_gated_rmsnorm_v0(
        self,
        y: VRAMMatrixVar,
        z: VRAMMatrixVar,
        norm_weight: VRAMMatrixVar | None,
        gate_scratch: VRAMMatrixVar,
        sq_scratch: VRAMMatrixVar,
        rms_fp: FPVar,
        consts: MambaFPConstants,
        shape: Mamba2Shape,
        *,
        rows: Sequence[int] | None = None,
    ):
        """``y = RMSNorm(y) * silu(z)``, in place on `y`.

        Mamba-2 normalises over ``d_inner / n_groups``, not over ``hidden_size``,
        so this cannot reuse ``asm_templates/normalization_asm.py`` unchanged --
        that template hardcodes its reduction bound as ``hidden_dim // vlen`` and
        takes a single ``1/hidden_dim`` FPRAM slot. ``consts.reci_group`` carries
        the group reciprocal instead.

        `rms_fp` must have room for one slot per row in `rows`; the caller owns it
        so the allocation does not churn FPRAM once per call.
        """
        resolved = list(range(y.shape[0])) if rows is None else list(rows)
        if rms_fp.size < len(resolved):
            raise ValueError(f"rms_fp needs {len(resolved)} slots, has {rms_fp.size}")
        self.emit_comment(mamba_stage_marker("mamba_gated_norm", "rmsnorm(y * silu(z))"))

        # The gate is applied BEFORE the variance. Both upstream implementations do
        # this -- HuggingFace's MambaRMSNormGated multiplies by silu(gate) and only
        # then reduces, and mamba_ssm's Mamba2 uses norm_before_gate=False. The two
        # orders are different functions, not different roundings.
        self.mamba_block_copy(gate_scratch, z, num_rows=z.shape[0])
        self.mamba_silu_v0(
            gate_scratch, sq_scratch, consts, rows=resolved,
            marker=mamba_stage_marker("mamba_gated_norm", "silu"),
        )
        self.tile_row_mul(y, gate_scratch, rows=resolved)

        # sum(y^2) per row -> rms_fp
        self.mamba_block_copy(sq_scratch, y, num_rows=y.shape[0])
        self.tile_row_mul(sq_scratch, y, rows=resolved)
        self.tile_row_sum(rms_fp, sq_scratch, rows=resolved, target_base_offset=0)

        # rms_fp = 1 / sqrt(sum/n + eps), computed in the scalar unit. There is no
        # vector sqrt (S_FP_OP has SQRT, V_ELEMENT_OP does not), so per-row scalar
        # work is the only option here.
        self.mamba_rsqrt_fpram(rms_fp, consts, count=len(resolved))

        self.tile_row_mul_fp(y, rms_fp, rows=resolved, fpram_base_offset=0)
        if norm_weight is not None:
            self.tile_row_mul(y, norm_weight, rows=resolved)
        return y

    def mamba_rsqrt_fpram(self, acc: FPVar, consts: MambaFPConstants, *, count: int):
        """In place ``acc[i] = 1 / sqrt(acc[i] * reci_group + eps)`` for i < count.

        Emitted as scalar FP ops because the ISA has no vector square root:
        ``S_FP_OP`` carries ``SQRT_FP`` but ``V_ELEMENT_OP`` does not (see
        ``doc/operation.svh``).
        """
        from compiler.aten.isa_builder import IsaBuilder, fp, gp

        gp_regs = self._reg.allocate_gp(1)
        # Allocate the FP registers rather than hardcoding f1-f3: RegisterAllocator
        # hands them out descending (f7, f6, f5, ...), so a caller already holding
        # several live FP registers would have had one silently clobbered here.
        fp_regs = self.allocate_fp_reg(3)
        (gp_addr,) = gp_regs
        f_val, f_scale, f_eps = fp_regs
        try:
            asm = IsaBuilder().comment(f"Mamba rsqrt over FPRAM[{acc.address}:+{count}]")
            asm.instr("S_ADDI_INT", gp(gp_addr), gp(0), consts.reci_group.address)
            asm.instr("S_LD_FP", fp(f_scale), gp(gp_addr), 0)
            asm.instr("S_ADDI_INT", gp(gp_addr), gp(0), consts.eps.address)
            asm.instr("S_LD_FP", fp(f_eps), gp(gp_addr), 0)
            for i in range(count):
                asm.instr("S_ADDI_INT", gp(gp_addr), gp(0), acc.address + i)
                asm.instr("S_LD_FP", fp(f_val), gp(gp_addr), 0)
                asm.instr("S_MUL_FP", fp(f_val), fp(f_val), fp(f_scale))  # mean = sum / n
                asm.instr("S_ADD_FP", fp(f_val), fp(f_val), fp(f_eps))    # + eps
                asm.instr("S_SQRT_FP", fp(f_val), fp(f_val))
                asm.instr("S_RECI_FP", fp(f_val), fp(f_val))
                # normalization_asm.py documents a 4-instruction spacer after
                # S_RECI_FP before the value is consumed. Use the same instruction it
                # uses -- S_ADDI_INT gp0, gp0, 0, a write to the hardwired-zero GP
                # register, i.e. an integer-pipe NOP. An S_ADD_FP would issue on the
                # very scalar-FP unit whose latency the spacer exists to cover, and
                # would need a register to write.
                for _ in range(4):
                    asm.instr("S_ADDI_INT", gp(0), gp(0), 0)
                asm.instr("S_ST_FP", fp(f_val), gp(gp_addr), 0)
            return self._emit(asm)
        finally:
            self._reg.free_gp(gp_regs)
            self.free_fp_reg(fp_regs)


__all__ = [
    "MAMBA_STAGES",
    "MAMBA_STAGE_MARKER_PREFIX",
    "Mamba2Shape",
    "MambaFPConstants",
    "ProgramMambaCommonMixin",
    "mamba_stage_marker",
]
