"""Mamba-2 mixer ASM templates: causal depthwise conv1d and the chunked SSD scan.

Both emitters live here because they are the two Mamba-2 kernels with no existing
template to reuse; everything else in the mixer (in_proj / out_proj / RMSNorm /
SiLU) is lowered by ``projection_asm``, ``normalization_asm`` and ``silu_asm``.

They are consumed by the *generator* pipeline (``generator/passes/code_gen.py``),
which imports them directly rather than through ``asm_templates/__init__.py``.

VRAM layout contract
--------------------
Every tensor here is addressed as a flat run of VLEN-wide VRAM rows with the
**sequence on the row axis**, i.e. the row holding channel-tile ``c`` at timestep
``s`` sits at ``base + (c * seq_len + s) * VLEN``.  That is the layout
``doc/memory_layout.md`` calls ``[h//VLEN, b, s, VLEN]``.  It is what makes the
depthwise conv1d cheap: a causal shift along ``t`` becomes a pure row-address
offset (free), and the per-channel tap weight ``w[j, :]`` is exactly a VLEN-wide
``V_MUL_VV`` operand.

Why the conv is *not* lowered through ``im2col_asm``
---------------------------------------------------
``im2col_asm`` hardcodes a square KxK spatial patch, has no padding support (so
it cannot express the causal left pad), and would materialise a dense
``(M, C_in*K*K)`` GEMM against a block-diagonal weight -- wasting ``conv_dim``-fold
MACs.  ``V_MUL_VF`` is equally wrong: it broadcasts one FP register across every
lane, but a depthwise tap varies *per channel*, i.e. per lane.  Hence
``V_MUL_VV`` against a resident tap row.

ISA facts these emitters respect
--------------------------------
* ``C_LOOP_START`` takes a compile-time immediate trip count and there is no
  branch, so every trip count below is a Python-side constant.
* There is no logarithm, so ``softplus`` uses the ``V_SOFTPLUS_V`` opcode; the
  ``time_step_limit`` clamp reuses ``V_MAX_VF`` / ``V_MIN_VF`` (there is no
  ``V_MAX_VV`` / ``V_MIN_VV``).
* There is no VRAM->VRAM move and no gather.  A row is zeroed by multiplying it
  by ``f0`` (hardwired 0.0) -- see ``_zero_row``.
* MRAM is write-only from HBM, so an activation-times-activation product costs a
  ``H_STORE_V`` + ``H_PREFETCH_M`` round trip.  The SSD scan has four such
  products per chunk and pays for all four explicitly (``_spill_asm`` +
  ``batched_matmul_asm``); that round trip is the dominant real cost of this
  kernel on PLENA and hiding it would make the utilization report a lie.

Fidelity note
-------------
This is the *generator* pipeline, whose purpose is structural and utilization
analysis rather than numerics (see ``docs/COMPILATION_PIPELINES.md``).  The
instruction mix, the loop nest and the memory traffic are modelled faithfully;
the individual addresses are not a numerically executable Mamba-2.  Places where
that gap is more than addressing are called out inline with ``; APPROX:``.
"""

from __future__ import annotations

import math

from ._imm import add_large_int as _add_large_int
from ._imm import load_large_int as _load_large_int
from .batched_matmul_asm import batched_matmul_asm

__all__ = ["mamba_conv1d_asm", "mamba_ssd_scan_asm"]


def _zero_row(reg: int) -> str:
    """Zero the VRAM row addressed by ``gp{reg}``.

    ``f0`` is hardwired to 0.0, so ``x * 0`` is the cheapest zero the ISA offers
    (there is no VRAM immediate store and no VRAM->VRAM move).  The one input it
    is not a true zero for is a NaN/Inf left over in the row; the generator path
    never depends on the pre-state of a scratch row.
    """
    return f"V_MUL_VF gp{reg}, gp{reg}, f0, 0"


def _vec_row_loop(
    *,
    body: list[str],
    ptr_regs: list[int],
    trip: int,
    loop_reg: int,
    vlen: int,
    comment: str = "",
) -> list[str]:
    """Wrap ``body`` in a hardware loop of ``trip`` iterations, bumping each
    pointer in ``ptr_regs`` by one VRAM row per iteration.

    ``trip`` is a compile-time immediate: ``C_LOOP_START`` cannot take a
    register-held trip count.
    """
    if trip <= 0:
        return []
    lines: list[str] = []
    if comment:
        lines.append(f"; {comment} ({trip} rows)")
    lines.append(f"C_LOOP_START gp{loop_reg}, {trip}")
    lines.extend(body)
    for r in ptr_regs:
        lines.append(f"S_ADDI_INT gp{r}, gp{r}, {vlen}")
    lines.append(f"C_LOOP_END gp{loop_reg}")
    return lines


def _spill_asm(
    *,
    vram_base: int,
    elements: int,
    hbm_addr_reg: int,
    vram_reg: int,
    hbm_reg: int,
    loop_reg: int,
    vlen: int,
    writeback_amount: int,
    label: str,
) -> list[str]:
    """VRAM -> HBM spill of ``elements`` elements, so they can come back as a
    matrix operand through ``H_PREFETCH_M``.

    MRAM cannot be written from VRAM; the only path is HBM.  Each ``H_STORE_V``
    moves ``HBM_V_Writeback_Amount * VLEN`` elements.
    """
    per_issue = writeback_amount * vlen
    issues = math.ceil(elements / per_issue) if elements > 0 else 0
    if issues == 0:
        return []
    lines = [f"; spill {label}: {elements} elements -> HBM[a{hbm_addr_reg}] ({issues} x H_STORE_V)"]
    lines.extend(_load_large_int(vram_reg, vram_base))
    lines.append(f"S_ADDI_INT gp{hbm_reg}, gp0, 0")
    lines.append(f"C_LOOP_START gp{loop_reg}, {issues}")
    lines.append(f"H_STORE_V gp{vram_reg}, gp{hbm_reg}, a{hbm_addr_reg}, 1, 0")
    lines.append(f"S_ADDI_INT gp{vram_reg}, gp{vram_reg}, {per_issue}")
    lines.append(f"S_ADDI_INT gp{hbm_reg}, gp{hbm_reg}, {per_issue}")
    lines.append(f"C_LOOP_END gp{loop_reg}")
    return lines


def mamba_conv1d_asm(
    *,
    vlen: int,
    seq_len: int,
    conv_dim: int,
    conv_kernel: int,
    alive_registers: list[int],
    input_base_address: int,
    output_base_address: int,
    weight_base_address: int,
    scratch_base_address: int,
    bias_base_address: int | None = None,
) -> str:
    """Causal depthwise conv1d of width ``conv_kernel`` along the sequence axis.

    ``out[c, s] = bias[c] + sum_j w[j, c] * x[c, s - (conv_kernel - 1) + j]``
    with taps that fall before ``s = 0`` dropped (the causal left pad).

    Rows are addressed per the module-level layout contract.  Tap ``j`` of
    channel-tile ``c`` lives at ``weight_base_address + (j * n_tiles + c) * VLEN``,
    i.e. ``conv_dim`` elements apart per tap.

    The first ``conv_kernel - 1`` timesteps are unrolled because each sees a
    different number of live taps; the remaining ``seq_len - conv_kernel + 1``
    share one body and run in a hardware loop.

    Args:
        alive_registers: at least 8 GP registers.  In order: x pointer, out
            pointer, tap-0 pointer, two address temporaries, the product scratch
            row, the loop counter, and the bias pointer.
    """
    if conv_kernel < 1:
        raise ValueError(f"conv_kernel must be >= 1, got {conv_kernel}")
    if seq_len < 1:
        raise ValueError(f"seq_len must be >= 1, got {seq_len}")
    if len(alive_registers) < 8:
        raise ValueError(f"mamba_conv1d_asm needs 8 GP registers, got {len(alive_registers)}")

    x_ptr, o_ptr, w_ptr, tmp_a, tmp_b, scr, loop_reg, bias_reg = alive_registers[:8]

    n_tiles = math.ceil(conv_dim / vlen)
    k = conv_kernel
    # Tap stride in elements: consecutive taps are a whole channel row apart.
    tap_stride = n_tiles * vlen

    lines = [
        "; === Mamba-2 causal depthwise conv1d ===",
        f"; conv_dim={conv_dim} kernel={k} seq_len={seq_len} channel_tiles={n_tiles} VLEN={vlen}",
        "; Sequence is on the VRAM row axis, so the causal shift is a row-address offset",
        "; and each tap is a VLEN-wide V_MUL_VV operand (V_MUL_VF would broadcast one",
        "; scalar over all lanes, but a depthwise tap varies per channel = per lane).",
    ]
    if conv_dim % vlen:
        lines.append(
            f"; APPROX: conv_dim={conv_dim} is not a multiple of VLEN={vlen}; the final tile is"
        )
        lines.append("; emitted full-width (lane masking via C_SET_V_MASK_REG is not modelled here).")

    lines.extend(_load_large_int(scr, scratch_base_address))

    for c in range(n_tiles):
        tile_off = c * seq_len * vlen
        lines.append(f"; -- channel tile {c}/{n_tiles} --")
        lines.extend(_load_large_int(w_ptr, weight_base_address + c * vlen))
        if bias_base_address is not None:
            lines.extend(_load_large_int(bias_reg, bias_base_address + c * vlen))

        # Prologue: timesteps that see the causal zero pad.
        for s in range(min(k - 1, seq_len)):
            first_live = (k - 1) - s
            lines.append(f"; t={s}: taps 0..{first_live - 1} fall before the sequence (zero pad)")
            lines.extend(_load_large_int(o_ptr, output_base_address + tile_off + s * vlen))
            for n, j in enumerate(range(first_live, k)):
                src = s - (k - 1) + j
                lines.extend(_load_large_int(tmp_a, input_base_address + tile_off + src * vlen))
                lines.extend(_add_large_int(tmp_b, w_ptr, j * tap_stride))
                if n == 0:
                    # First live tap initialises the accumulator, so no zeroing pass.
                    lines.append(f"V_MUL_VV gp{o_ptr}, gp{tmp_a}, gp{tmp_b}, 0")
                else:
                    lines.append(f"V_MUL_VV gp{scr}, gp{tmp_a}, gp{tmp_b}, 0")
                    lines.append(f"V_ADD_VV gp{o_ptr}, gp{o_ptr}, gp{scr}, 0")
            if bias_base_address is not None:
                lines.append(f"V_ADD_VV gp{o_ptr}, gp{o_ptr}, gp{bias_reg}, 0")

        # Steady state: every timestep from k-1 on sees all k taps.
        steady = seq_len - (k - 1)
        if steady > 0:
            lines.append(f"; t={k - 1}..{seq_len - 1}: all {k} taps live")
            # x pointer trails the output pointer by (k-1) rows.
            lines.extend(_load_large_int(x_ptr, input_base_address + tile_off))
            lines.extend(_load_large_int(o_ptr, output_base_address + tile_off + (k - 1) * vlen))
            body: list[str] = []
            for j in range(k):
                body.append(f"S_ADDI_INT gp{tmp_a}, gp{x_ptr}, {j * vlen}")
                body.extend(_add_large_int(tmp_b, w_ptr, j * tap_stride))
                if j == 0:
                    body.append(f"V_MUL_VV gp{o_ptr}, gp{tmp_a}, gp{tmp_b}, 0")
                else:
                    body.append(f"V_MUL_VV gp{scr}, gp{tmp_a}, gp{tmp_b}, 0")
                    body.append(f"V_ADD_VV gp{o_ptr}, gp{o_ptr}, gp{scr}, 0")
            if bias_base_address is not None:
                body.append(f"V_ADD_VV gp{o_ptr}, gp{o_ptr}, gp{bias_reg}, 0")
            lines.extend(
                _vec_row_loop(
                    body=body,
                    ptr_regs=[x_ptr, o_ptr],
                    trip=steady,
                    loop_reg=loop_reg,
                    vlen=vlen,
                )
            )

    return "\n".join(lines) + "\n"


def mamba_ssd_scan_asm(
    *,
    mlen: int,
    vlen: int,
    blen: int,
    seq_len: int,
    chunk_size: int,
    num_heads: int,
    head_dim: int,
    state_size: int,
    n_groups: int,
    alive_registers: list[int],
    vram: dict[str, int],
    act_spill_addr_reg: int,
    wt_spill_addr_reg: int,
    writeback_amount: int = 4,
    dt_min_fp_address: int = 0,
    dt_max_fp_address: int = 0,
    a_decay_fp_address: int = 0,
    d_skip_fp_address: int = 0,
) -> str:
    """Chunked state-space-duality scan for one Mamba-2 layer.

    Emits, per chunk of ``chunk_size`` timesteps, the six stages of the SSD
    recurrence:

    1. ``dt = clamp(softplus(dt), time_step_limit)``           (``V_SOFTPLUS_V``)
    2. ``decay = exp(cumsum(dt * A))`` within the chunk        (serial ``V_ADD_VV`` chain)
    3. ``G = C @ B^T`` then ``M = G * L``                      (GEMM + ``V_MUL_VV``)
    4. ``Y_diag = M @ X``                                      (GEMM)
    5. ``S = decay * S + B^T @ (X * dt)``                      (GEMM + ``V_MUL_VF``)
    6. ``Y = Y_diag + C @ S + D * X``                          (GEMM + ``V_ADD_VV``)

    The four GEMMs go through :func:`batched_matmul_asm` with the head (or group)
    axis as its batch axis, which is exactly the shape it was written for.  Both
    of their operands are activations, so each is spilled to HBM first -- see the
    module docstring on why that round trip is unavoidable and is emitted rather
    than elided.

    ``vram`` maps region names to VLEN-aligned element addresses and must supply:
    ``x``, ``B``, ``C``, ``dt``, ``decay``, ``score``, ``state``, ``y``,
    ``scratch``.

    Args:
        act_spill_addr_reg / wt_spill_addr_reg: HBM address registers for the two
            operand spill areas.  They must differ: ``batched_matmul_asm``
            addresses both operands from offset 0 of their own register.
        alive_registers: at least 8 GP registers.
    """
    if len(alive_registers) < 8:
        raise ValueError(f"mamba_ssd_scan_asm needs 8 GP registers, got {len(alive_registers)}")
    if act_spill_addr_reg == wt_spill_addr_reg:
        raise ValueError(
            "act_spill_addr_reg and wt_spill_addr_reg must differ: batched_matmul_asm "
            "addresses each operand from offset 0 of its own HBM address register, so "
            "sharing one register would overlay the two operands."
        )
    for key in ("x", "B", "C", "dt", "decay", "score", "state", "y", "scratch", "row0", "row1"):
        if key not in vram:
            raise KeyError(f"vram map is missing region {key!r}")
    if state_size % mlen or chunk_size % mlen:
        raise ValueError(
            f"state_size ({state_size}) and chunk_size ({chunk_size}) must be multiples of "
            f"MLEN ({mlen}): both are K dimensions of the SSD GEMMs."
        )
    if num_heads % n_groups:
        raise ValueError(f"num_heads ({num_heads}) must be divisible by n_groups ({n_groups})")

    p0, p1, p2, p3, tmp, scr, loop_reg, hbm_reg = alive_registers[:8]

    num_chunks = math.ceil(seq_len / chunk_size)
    heads_per_group = num_heads // n_groups
    group_state = n_groups * state_size

    lines = [
        "; === Mamba-2 chunked SSD scan ===",
        f"; seq_len={seq_len} chunk={chunk_size} chunks={num_chunks}",
        f"; heads={num_heads} head_dim={head_dim} state={state_size} groups={n_groups}"
        f" (heads/group={heads_per_group})",
        "; Four activation x activation GEMMs per chunk; each operand is spilled",
        f"; VRAM -> HBM (a{act_spill_addr_reg} / a{wt_spill_addr_reg}) because MRAM is",
        "; write-only from HBM.",
    ]
    # gp{scr} holds a one-row temporary for the whole kernel.  batched_matmul_asm
    # only clobbers p1/p2/p3, so this address survives every GEMM below.
    lines.extend(_load_large_int(scr, vram["row1"]))

    def spill(vram_base: int, elements: int, addr_reg: int, label: str) -> list[str]:
        return _spill_asm(
            vram_base=vram_base,
            elements=elements,
            hbm_addr_reg=addr_reg,
            vram_reg=p0,
            hbm_reg=hbm_reg,
            loop_reg=loop_reg,
            vlen=vlen,
            writeback_amount=writeback_amount,
            label=label,
        )

    def gemm(b: int, m: int, k: int, n: int, result: int, label: str) -> list[str]:
        out = [f"; -- GEMM {label}: ({b}, {m}, {k}) @ ({b}, {k}, {n}) --"]
        out.append(
            batched_matmul_asm(
                mlen=mlen,
                blen=blen,
                b=b,
                m=m,
                k=k,
                n=n,
                alive_registers=[p1, p2, p3],
                w_base_hbm_offset_reg=wt_spill_addr_reg,
                # w_prefetch_amount is only used by an internal sanity assert; the
                # spilled operand is contiguous so K tiles are always reachable.
                w_prefetch_amount=k,
                a_base_hbm_offset_reg=act_spill_addr_reg,
                a_prefetch_amount=k,
                result_base_address=result,
            ).rstrip("\n")
        )
        return out

    for ch in range(num_chunks):
        rows = min(chunk_size, seq_len - ch * chunk_size)
        lines.append(f"; ================ chunk {ch}/{num_chunks} (rows {ch * chunk_size}..{ch * chunk_size + rows - 1}) ================")
        if rows != chunk_size:
            lines.append(
                f"; APPROX: tail chunk holds {rows} of {chunk_size} timesteps; the GEMMs below are"
            )
            lines.append("; emitted at full chunk width (the pad rows are computed and discarded).")

        x_chunk = vram["x"] + ch * chunk_size * num_heads * head_dim
        b_chunk = vram["B"] + ch * chunk_size * group_state
        c_chunk = vram["C"] + ch * chunk_size * group_state
        dt_chunk = vram["dt"] + ch * chunk_size * num_heads
        y_chunk = vram["y"] + ch * chunk_size * num_heads * head_dim

        # ---------------- 1. dt activation ----------------
        lines.append("; @stage=mamba_dt  softplus + time_step_limit clamp")
        lines.append(f"S_LD_FP f1, gp0, {dt_min_fp_address}")
        lines.append(f"S_LD_FP f2, gp0, {dt_max_fp_address}")
        lines.extend(_load_large_int(p0, dt_chunk))
        dt_rows = max(1, math.ceil(chunk_size * num_heads / vlen))
        lines.extend(
            _vec_row_loop(
                body=[
                    # No logarithm in the ISA: softplus is a dedicated opcode.
                    f"V_SOFTPLUS_V gp{p0}, gp{p0}, 0",
                    f"V_MAX_VF gp{p0}, gp{p0}, f1, 0",
                    f"V_MIN_VF gp{p0}, gp{p0}, f2, 0",
                ],
                ptr_regs=[p0],
                trip=dt_rows,
                loop_reg=loop_reg,
                vlen=vlen,
                comment="dt rows",
            )
        )

        # ---------------- 2. intra-chunk decay cumsum ----------------
        lines.append("; @stage=mamba_chunk_cumsum  decay = exp(cumsum(dt * A)) along the chunk")
        lines.append("; Serial dependent scan: V_PS_V is a within-row prefix scan, but the")
        lines.append("; cumsum here runs across rows (sequence is on the row axis), so the")
        lines.append("; recurrence is an explicit V_ADD_VV chain of chunk_size steps.")
        lines.append(f"S_LD_FP f3, gp0, {a_decay_fp_address}")
        lines.extend(_load_large_int(p1, dt_chunk))
        lines.extend(_load_large_int(p2, vram["decay"]))
        lines.extend(_load_large_int(p3, vram["row0"]))
        lines.append(_zero_row(p3))  # running sum accumulator
        lines.extend(
            _vec_row_loop(
                body=[
                    f"V_MUL_VF gp{scr}, gp{p1}, f3, 0",
                    f"V_ADD_VV gp{p3}, gp{p3}, gp{scr}, 0",
                    f"V_EXP_V gp{p2}, gp{p3}, 0",
                ],
                ptr_regs=[p1, p2],
                trip=dt_rows,
                loop_reg=loop_reg,
                vlen=vlen,
                comment="cumsum steps",
            )
        )
        # Publish the per-head chunk decay scalars into FPRAM so the state update
        # below can broadcast them with V_MUL_VF -- this is exactly what the
        # S_MAP_FP_V (VRAM row -> VLEN FPRAM slots) opcode exists for.
        lines.extend(_load_large_int(p2, vram["decay"]))
        lines.extend(_load_large_int(tmp, a_decay_fp_address))
        lines.append(f"S_MAP_FP_V gp{tmp}, gp{p2}, 0")

        # ---------------- 3. intra-chunk scores ----------------
        lines.append("; @stage=mamba_intra_chunk  G = C @ B^T")
        lines.extend(spill(c_chunk, n_groups * chunk_size * state_size, act_spill_addr_reg, "C_chunk"))
        lines.append("; B is spilled untransposed; the transpose is absorbed by H_PREFETCH_M's")
        lines.append("; stride mode (element(row, col) at col*stride + row).")
        lines.extend(spill(b_chunk, n_groups * chunk_size * state_size, wt_spill_addr_reg, "B_chunk"))
        lines.extend(
            gemm(n_groups, chunk_size, state_size, chunk_size, vram["score"], "C @ B^T")
        )

        lines.append("; @stage=mamba_decay_mask  M = G * L (segment-sum decay mask)")
        lines.append("; APPROX: the (chunk, chunk) causal segment-sum mask L is an outer")
        lines.append("; difference of the decay vector; the ISA has no outer-product primitive,")
        lines.append("; so it is modelled as a per-row V_MUL_VV against the materialised decay")
        lines.append("; rows.  The instruction count and traffic match; the values do not.")
        lines.extend(_load_large_int(p0, vram["score"]))
        lines.extend(_load_large_int(p1, vram["decay"]))
        lines.extend(
            _vec_row_loop(
                body=[f"V_MUL_VV gp{p0}, gp{p0}, gp{p1}, 0"],
                ptr_regs=[p0],
                trip=max(1, num_heads * chunk_size * chunk_size // vlen),
                loop_reg=loop_reg,
                vlen=vlen,
                comment="score rows",
            )
        )

        # ---------------- 4. intra-chunk output ----------------
        lines.append("; @stage=mamba_intra_chunk  Y_diag = M @ X")
        lines.extend(spill(vram["score"], num_heads * chunk_size * chunk_size, act_spill_addr_reg, "M"))
        lines.extend(spill(x_chunk, num_heads * chunk_size * head_dim, wt_spill_addr_reg, "X_chunk"))
        lines.extend(gemm(num_heads, chunk_size, chunk_size, head_dim, y_chunk, "M @ X"))

        # ---------------- 5. state update ----------------
        lines.append("; @stage=mamba_state_update  S = decay * S + B^T @ (X * dt)")
        lines.append("; dt is one scalar per (token, head) but X is head_dim wide, so the dt")
        lines.append("; pointer is deliberately not advanced with X: one dt row feeds")
        lines.append("; head_dim/VLEN consecutive X rows.")
        lines.extend(_load_large_int(p0, vram["scratch"]))
        lines.extend(_load_large_int(p1, x_chunk))
        lines.extend(_load_large_int(p2, dt_chunk))
        lines.extend(
            _vec_row_loop(
                body=[f"V_MUL_VV gp{p0}, gp{p1}, gp{p2}, 0"],
                ptr_regs=[p0, p1],
                trip=max(1, num_heads * chunk_size * head_dim // vlen),
                loop_reg=loop_reg,
                vlen=vlen,
                comment="X * dt",
            )
        )
        lines.extend(spill(b_chunk, num_heads * chunk_size * state_size, act_spill_addr_reg, "B^T"))
        lines.extend(spill(vram["scratch"], num_heads * chunk_size * head_dim, wt_spill_addr_reg, "X*dt"))
        lines.extend(
            gemm(num_heads, state_size, chunk_size, head_dim, vram["scratch"], "B^T @ (X*dt)")
        )
        lines.append("; decay the carried state, then fold in this chunk's contribution.")
        lines.append(f"S_LD_FP f4, gp0, {a_decay_fp_address}")
        lines.extend(_load_large_int(p0, vram["state"]))
        lines.extend(_load_large_int(p1, vram["scratch"]))
        lines.extend(
            _vec_row_loop(
                body=[
                    f"V_MUL_VF gp{p0}, gp{p0}, f4, 0",
                    f"V_ADD_VV gp{p0}, gp{p0}, gp{p1}, 0",
                ],
                ptr_regs=[p0, p1],
                trip=max(1, num_heads * state_size * head_dim // vlen),
                loop_reg=loop_reg,
                vlen=vlen,
                comment="state rows",
            )
        )

        # ---------------- 6. inter-chunk contribution + skip ----------------
        lines.append("; @stage=mamba_inter_chunk  Y += C @ S")
        lines.extend(spill(c_chunk, num_heads * chunk_size * state_size, act_spill_addr_reg, "C_chunk"))
        lines.extend(spill(vram["state"], num_heads * state_size * head_dim, wt_spill_addr_reg, "S"))
        lines.extend(gemm(num_heads, chunk_size, state_size, head_dim, vram["scratch"], "C @ S"))
        lines.extend(_load_large_int(p0, y_chunk))
        lines.extend(_load_large_int(p1, vram["scratch"]))
        lines.extend(
            _vec_row_loop(
                body=[f"V_ADD_VV gp{p0}, gp{p0}, gp{p1}, 0"],
                ptr_regs=[p0, p1],
                trip=max(1, num_heads * chunk_size * head_dim // vlen),
                loop_reg=loop_reg,
                vlen=vlen,
                comment="Y += Y_off",
            )
        )

        lines.append("; @stage=mamba_skip  Y += D * X  (per-head scalar skip)")
        lines.append(f"S_LD_FP f5, gp0, {d_skip_fp_address}")
        lines.extend(_load_large_int(p0, y_chunk))
        lines.extend(_load_large_int(p1, x_chunk))
        lines.extend(
            _vec_row_loop(
                body=[
                    f"V_MUL_VF gp{scr}, gp{p1}, f5, 0",
                    f"V_ADD_VV gp{p0}, gp{p0}, gp{scr}, 0",
                ],
                ptr_regs=[p0, p1],
                trip=max(1, num_heads * chunk_size * head_dim // vlen),
                loop_reg=loop_reg,
                vlen=vlen,
                comment="skip rows",
            )
        )

    return "\n".join(lines) + "\n"
