"""PLENA intrinsic descriptors.

Each intrinsic in PLENA's ISA gets a Python descriptor here:
    - canonical name used in T.call_extern("handle", "plena.<name>", ...)
    - operand scope constraints (which RAM each operand must live in)
    - simple printer used by the codegen pass

This is the place to add new ops as the compiler grows. The codegen
walks the TIR, finds plena.* extern calls, looks them up here, verifies
scopes, and emits ISA text.

FPRAM operand convention:
    FPRAM is treated as a flat scalar register file, not a buffered
    region. Every FP operand position is a SCALAR address (PrimExpr or
    int), counted in element units from address 0. The kernel is
    responsible for adding any per-slot base offset before passing the
    value in. There are no FPRAM buffer handles in TIR anymore.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Sequence

from . import scope as _scope


@dataclass(frozen=True)
class IntrinsicSpec:
    name: str
    # Required scope per buffer-typed operand position.
    # `None` means "scalar / immediate / FP address, no scope check".
    operand_scopes: Sequence[str | None]
    # Friendly printer: takes a list of resolved operand strings and
    # any trailing scalar args, returns one ISA line.
    emit: Callable[[list[str]], str]


_REGISTRY: Dict[str, IntrinsicSpec] = {}


def register(spec: IntrinsicSpec) -> None:
    if spec.name in _REGISTRY:
        raise ValueError(f"duplicate intrinsic: {spec.name}")
    _REGISTRY[spec.name] = spec


def lookup(name: str) -> IntrinsicSpec:
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown PLENA intrinsic: {name!r}. "
            f"Known: {sorted(_REGISTRY.keys())}"
        )
    return _REGISTRY[name]


def all_names() -> list[str]:
    return sorted(_REGISTRY.keys())


# ---------------------------------------------------------------------------
# DMA / matmul / vector ops
# ---------------------------------------------------------------------------

register(IntrinsicSpec(
    name="plena.dma_h2v",
    operand_scopes=(_scope.HBM, _scope.VRAM, None),  # src, dst, size
    emit=lambda a: f"DMA_H2V  src={a[0]}  dst={a[1]}  size={a[2]}",
))

register(IntrinsicSpec(
    name="plena.dma_h2m",
    operand_scopes=(_scope.HBM, _scope.MRAM, None),
    emit=lambda a: f"DMA_H2M  src={a[0]}  dst={a[1]}  size={a[2]}",
))

register(IntrinsicSpec(
    name="plena.dma_v2h",
    operand_scopes=(_scope.VRAM, _scope.HBM, None),
    emit=lambda a: f"DMA_V2H  src={a[0]}  dst={a[1]}  size={a[2]}",
))

register(IntrinsicSpec(
    name="plena.btmm",
    operand_scopes=(_scope.VRAM, _scope.MRAM, _scope.VRAM, None),
    emit=lambda a: f"BTMM     A={a[0]}  B={a[1]}  C={a[2]}  group_heads={a[3]}",
))

register(IntrinsicSpec(
    # Lane-fused matrix-vector. LHS is a 1-D vector (lane-packed across
    # heads, MLEN-wide); RHS is a (mlen, lane_count, hlen) MRAM matrix
    # (same layout as BTMM's RHS). DST is a row-stacked 1-D vector that
    # M_BMV_WO writes out as `lane_count` MLEN-wide rows.
    # Maps to one M_BTMV + M_BMV_WO pair, parallel to plena.btmm's
    # M_BTMM + M_BMM_WO.
    name="plena.btmv",
    operand_scopes=(_scope.VRAM, _scope.MRAM, _scope.VRAM, None),
    emit=lambda a: f"BTMV     A={a[0]}  B={a[1]}  C={a[2]}  group_heads={a[3]}",
))

register(IntrinsicSpec(
    name="plena.v_add",
    operand_scopes=(_scope.VRAM, _scope.VRAM, _scope.VRAM),
    emit=lambda a: f"V_ADD    lhs={a[0]}  rhs={a[1]}  dst={a[2]}",
))

register(IntrinsicSpec(
    name="plena.v_sub",
    operand_scopes=(_scope.VRAM, _scope.VRAM, _scope.VRAM),
    emit=lambda a: f"V_SUB    lhs={a[0]}  rhs={a[1]}  dst={a[2]}",
))

register(IntrinsicSpec(
    name="plena.v_mul",
    operand_scopes=(_scope.VRAM, _scope.VRAM, _scope.VRAM),
    emit=lambda a: f"V_MUL    lhs={a[0]}  rhs={a[1]}  dst={a[2]}",
))

register(IntrinsicSpec(
    name="plena.mm",
    operand_scopes=(_scope.VRAM, _scope.MRAM, _scope.VRAM),
    emit=lambda a: f"MM       A={a[0]}  B={a[1]}  C={a[2]}",
))

register(IntrinsicSpec(
    # Unified `(M, K) @ (K, N) -> (M, N)`. Replaces plena.mm + plena.mm_slot.
    # K reduction is folded into the matmul op itself (M_MM accumulate +
    # M_MM_WO drain), so no caller-side scratch + v_add is needed for K.
    # N may exceed mlen and walks across mlen-wide B-tile blocks internally.
    #
    # Trailing scalar args (offsets) let the same op address sub-regions
    # of larger buffers without buffer slicing in HLIR:
    #   lhs_offset      : element offset added to A_v's base (int or PrimExpr)
    #   rhs_offset      : element offset added to B_m's base (int)
    #   dst_offset      : element offset added to C_v's base (int)
    #   dst_row_stride  : C row stride in elements (int) -- defaults to N
    #                     when callers pass 0 here.
    name="plena.matmul",
    operand_scopes=(
        _scope.VRAM, _scope.MRAM, _scope.VRAM,
        None, None, None,         # M_tiles, K_tiles, N
        None, None, None, None,   # lhs_offset, rhs_offset, dst_offset, dst_row_stride
    ),
    emit=lambda a: (
        f"MATMUL   A={a[0]}  B={a[1]}  C={a[2]}  "
        f"M_tiles={a[3]}  K_tiles={a[4]}  N={a[5]}  "
        f"lhs_off={a[6]}  rhs_off={a[7]}  dst_off={a[8]}  dst_row_stride={a[9]}"
    ),
))

register(IntrinsicSpec(
    # Per-head matrix-vector, single-lane M_MV + M_MV_WO.
    # Used for the P @ V step of decode where the LHS is one row of a
    # row-stacked S_loc fragment (one head's score vector). Each call
    # handles ONE head; the kernel author wraps it in a per-lane loop
    # (T.serial(lane_count) or T.unroll), exactly mirroring how
    # flash_attention_min uses plena.matmul (M_MM) per head.
    #
    # Trailing offsets are element offsets added to each buffer's base.
    # All three may be int OR PrimExpr (materialized to gp registers).
    name="plena.mv",
    operand_scopes=(
        _scope.VRAM, _scope.MRAM, _scope.VRAM,
        None, None, None,         # lhs_offset, rhs_offset, dst_offset
    ),
    emit=lambda a: (
        f"MV       A={a[0]}  B={a[1]}  C={a[2]}  "
        f"lhs_off={a[3]}  rhs_off={a[4]}  dst_off={a[5]}"
    ),
))

register(IntrinsicSpec(
    name="plena.mm_slot",
    operand_scopes=(_scope.VRAM, _scope.MRAM, _scope.VRAM, None, None, None, None),
    emit=lambda a: (
        f"MM_SLOT  A={a[0]}  B={a[1]}  C={a[2]}  "
        f"lhs_row_offset={a[3]}  rhs_col_offset={a[4]}  "
        f"dst_col_offset={a[5]}  col_count={a[6]}"
    ),
))

register(IntrinsicSpec(
    name="plena.zero_v",
    operand_scopes=(_scope.VRAM,),
    emit=lambda a: f"ZERO_V   dst={a[0]}",
))


# ---------------------------------------------------------------------------
# FP scalar ops (`_at` only). FP operands are SCALAR addresses, counted
# in element units. Kernel computes `slot_base + h*rows + row` and passes
# the result; codegen does no per-slot offset math of its own.
# ---------------------------------------------------------------------------

register(IntrinsicSpec(
    name="plena.fp_copy_at",
    operand_scopes=(None, None),  # src_addr, dst_addr
    emit=lambda a: f"FP_COPY_AT  src={a[0]} dst={a[1]}",
))

register(IntrinsicSpec(
    name="plena.fp_add_at",
    operand_scopes=(None, None, None),  # lhs_addr, rhs_addr, dst_addr
    emit=lambda a: f"FP_ADD_AT   lhs={a[0]} rhs={a[1]} dst={a[2]}",
))

register(IntrinsicSpec(
    name="plena.fp_sub_at",
    operand_scopes=(None, None, None),
    emit=lambda a: f"FP_SUB_AT   lhs={a[0]} rhs={a[1]} dst={a[2]}",
))

register(IntrinsicSpec(
    name="plena.fp_mul_at",
    operand_scopes=(None, None, None),
    emit=lambda a: f"FP_MUL_AT   lhs={a[0]} rhs={a[1]} dst={a[2]}",
))

register(IntrinsicSpec(
    name="plena.fp_max_at",
    operand_scopes=(None, None, None),
    emit=lambda a: f"FP_MAX_AT   lhs={a[0]} rhs={a[1]} dst={a[2]}",
))

register(IntrinsicSpec(
    name="plena.fp_exp_at",
    operand_scopes=(None, None),
    emit=lambda a: f"FP_EXP_AT   src={a[0]} dst={a[1]}",
))

register(IntrinsicSpec(
    name="plena.fp_reci_at",
    operand_scopes=(None, None),
    emit=lambda a: f"FP_RECI_AT  src={a[0]} dst={a[1]}",
))

register(IntrinsicSpec(
    name="plena.fp_sqrt_at",
    operand_scopes=(None, None),
    emit=lambda a: f"FP_SQRT_AT  src={a[0]} dst={a[1]}",
))


# ---------------------------------------------------------------------------
# Row ops (`_at` only). VRAM-side dim2/dim3 select the row to operate on
# and synthesize any packed-head V_MASK; FP-side operand is a SCALAR
# address, identical to the FP `_at` family above.
# ---------------------------------------------------------------------------

register(IntrinsicSpec(
    name="plena.row_reduce_max_at",
    # vram_src, fp_dst_addr, dim2, dim3
    operand_scopes=(_scope.VRAM, None, None, None),
    emit=lambda a: f"ROW_REDUCE_MAX_AT src={a[0]} dst={a[1]} dim2={a[2]} dim3={a[3]}",
))

register(IntrinsicSpec(
    name="plena.row_reduce_sum_at",
    operand_scopes=(_scope.VRAM, None, None, None),
    emit=lambda a: f"ROW_REDUCE_SUM_AT src={a[0]} dst={a[1]} dim2={a[2]} dim3={a[3]}",
))

register(IntrinsicSpec(
    name="plena.row_exp_at",
    # vram_src, vram_dst, dim2, dim3 (no FP operand)
    operand_scopes=(_scope.VRAM, _scope.VRAM, None, None),
    emit=lambda a: f"ROW_EXP_AT src={a[0]} dst={a[1]} dim2={a[2]} dim3={a[3]}",
))

register(IntrinsicSpec(
    name="plena.row_sub_fp_at",
    # vram_src, fp_addr, vram_dst, dim2, dim3
    operand_scopes=(_scope.VRAM, None, _scope.VRAM, None, None),
    emit=lambda a: f"ROW_SUB_FP_AT src={a[0]} rhs={a[1]} dst={a[2]} dim2={a[3]} dim3={a[4]}",
))

register(IntrinsicSpec(
    name="plena.row_mul_fp_at",
    operand_scopes=(_scope.VRAM, None, _scope.VRAM, None, None),
    emit=lambda a: f"ROW_MUL_FP_AT src={a[0]} rhs={a[1]} dst={a[2]} dim2={a[3]} dim3={a[4]}",
))

register(IntrinsicSpec(
    name="plena.row_add_fp_at",
    operand_scopes=(_scope.VRAM, None, _scope.VRAM, None, None),
    emit=lambda a: f"ROW_ADD_FP_AT src={a[0]} rhs={a[1]} dst={a[2]} dim2={a[3]} dim3={a[4]}",
))


# ---------------------------------------------------------------------------
# Row-wide VRAM <-> FPRAM transfers. Each call moves exactly mlen elements
# (one full row); call inside a TIR loop for multi-row tiles. VRAM side is
# (buffer + element offset); FP side is a flat scalar address.
# ---------------------------------------------------------------------------

register(IntrinsicSpec(
    name="plena.row_load_v_to_fp",
    # vram_src_buf, vram_offset, fp_dst_addr
    operand_scopes=(_scope.VRAM, None, None),
    emit=lambda a: f"ROW_LOAD_V_TO_FP src={a[0]}+{a[1]} dst={a[2]}",
))

register(IntrinsicSpec(
    # Single MLEN-wide row copy in VRAM, lane-fused. Lowers to
    # ``V_ADD_VF dst, src, f0, 0`` which (with f0 reserved == 0) computes
    # ``dst[i] = src[i] + 0 = src[i]`` for the full HW vector. Used by
    # the "tensor cache" path: a small VRAM region pre-populated by a
    # testbench-side FPRAM->VRAM stub feeds the kernel via this op,
    # avoiding HBM DMA for vector-shape (seq=1) tensors.
    name="plena.copy_v_to_v",
    # src_buf, src_offset, dst_buf, dst_offset
    operand_scopes=(_scope.VRAM, None, _scope.VRAM, None),
    emit=lambda a: f"COPY_V_TO_V  src={a[0]}+{a[1]}  dst={a[2]}+{a[3]}",
))

register(IntrinsicSpec(
    name="plena.row_store_fp_to_v",
    # fp_src_addr, vram_dst_buf, vram_offset
    operand_scopes=(None, _scope.VRAM, None),
    emit=lambda a: f"ROW_STORE_FP_TO_V src={a[0]} dst={a[1]}+{a[2]}",
))


# ---------------------------------------------------------------------------
# Slice DMA variants (variadic args; only the first two operands are
# scope-checked).
# ---------------------------------------------------------------------------
register(IntrinsicSpec(
    name="plena.dma_h2v_slice",
    operand_scopes=(_scope.HBM, _scope.VRAM, None),
    emit=lambda a: f"DMA_H2V_SLICE  src={a[0]}  dst={a[1]}  ndim={a[2]}",
))

register(IntrinsicSpec(
    name="plena.dma_h2m_slice",
    operand_scopes=(_scope.HBM, _scope.MRAM, None),
    emit=lambda a: f"DMA_H2M_SLICE  src={a[0]}  dst={a[1]}  ndim={a[2]}",
))

register(IntrinsicSpec(
    name="plena.dma_v2h_slice",
    operand_scopes=(_scope.VRAM, _scope.HBM, None),
    emit=lambda a: f"DMA_V2H_SLICE  src={a[0]}  dst={a[1]}  ndim={a[2]}",
))
