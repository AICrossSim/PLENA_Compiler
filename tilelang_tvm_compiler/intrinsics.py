"""PLENA intrinsic descriptors.

Each intrinsic in PLENA's ISA gets a Python descriptor here:
    - canonical name used in T.call_extern("handle", "plena.<name>", ...)
    - operand scope constraints (which RAM each operand must live in)
    - simple printer used by the codegen pass

This is the place to add new ops as the compiler grows. The codegen
walks the TIR, finds plena.* extern calls, looks them up here, verifies
scopes, and emits ISA text.

Why call_extern (not registered TVM intrinsics):
    - we never lower these to LLVM/CUDA, only to our own ISA text
    - call_extern preserves the symbolic name through TIR transforms
    - keeps the registration story trivial (no C++ / FFI involved)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Sequence

from . import scope as _scope


@dataclass(frozen=True)
class IntrinsicSpec:
    name: str
    # Required scope per buffer-typed operand position.
    # `None` means "scalar / immediate, no scope check".
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
# Initial intrinsic set (intentionally tiny — enough for one end-to-end test).
# Add new ops here as you bring up more kernels.
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
    # BTMM: C (vram) = A (vram) @ B (mram), with group_heads as scalar attr
    operand_scopes=(_scope.VRAM, _scope.MRAM, _scope.VRAM, None),
    emit=lambda a: f"BTMM     A={a[0]}  B={a[1]}  C={a[2]}  group_heads={a[3]}",
))

register(IntrinsicSpec(
    name="plena.v_add",
    operand_scopes=(_scope.VRAM, _scope.VRAM, _scope.VRAM),  # lhs, rhs, dst
    emit=lambda a: f"V_ADD    lhs={a[0]}  rhs={a[1]}  dst={a[2]}",
))

# Single-head matrix multiply: lhs (vram, one mlen*mlen tile)
# @ rhs (mram, one mlen*mlen tile) -> dst (vram, one mlen*mlen tile).
# Lowered to the M_MM / M_MM_WO instruction pair via emit_matmul.
# This is the "regular" MM hardware path; multi-head iteration must be
# expressed in TIR (head loop) since M_MM has no lane structure.
register(IntrinsicSpec(
    name="plena.mm",
    operand_scopes=(_scope.VRAM, _scope.MRAM, _scope.VRAM),  # lhs, rhs, dst
    emit=lambda a: f"MM       A={a[0]}  B={a[1]}  C={a[2]}",
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

# Zero an mlen*mlen VRAM tile in-place. Used to clear an accumulator
# before a streaming MM contraction loop (V_ADD-based reduce).
register(IntrinsicSpec(
    name="plena.zero_v",
    operand_scopes=(_scope.VRAM,),
    emit=lambda a: f"ZERO_V   dst={a[0]}",
))

register(IntrinsicSpec(
    name="plena.map_fp_to_v",
    operand_scopes=(_scope.FPRAM, _scope.VRAM),
    emit=lambda a: f"MAP_FP_V src={a[0]} dst={a[1]}",
))

register(IntrinsicSpec(
    name="plena.map_v_to_fp",
    operand_scopes=(_scope.VRAM, _scope.FPRAM),
    emit=lambda a: f"MAP_V_FP src={a[0]} dst={a[1]}",
))

register(IntrinsicSpec(
    name="plena.fp_copy",
    operand_scopes=(_scope.FPRAM, _scope.FPRAM),
    emit=lambda a: f"FP_COPY  src={a[0]} dst={a[1]}",
))
register(IntrinsicSpec(
    name="plena.fp_copy_at",
    operand_scopes=(_scope.FPRAM, _scope.FPRAM, None),
    emit=lambda a: f"FP_COPY_AT  src={a[0]} dst={a[1]} row={a[2]}",
))

register(IntrinsicSpec(
    name="plena.fp_add",
    operand_scopes=(_scope.FPRAM, _scope.FPRAM, _scope.FPRAM),
    emit=lambda a: f"FP_ADD   lhs={a[0]} rhs={a[1]} dst={a[2]}",
))
register(IntrinsicSpec(
    name="plena.fp_add_at",
    operand_scopes=(_scope.FPRAM, _scope.FPRAM, _scope.FPRAM, None),
    emit=lambda a: f"FP_ADD_AT   lhs={a[0]} rhs={a[1]} dst={a[2]} row={a[3]}",
))

register(IntrinsicSpec(
    name="plena.fp_sub",
    operand_scopes=(_scope.FPRAM, _scope.FPRAM, _scope.FPRAM),
    emit=lambda a: f"FP_SUB   lhs={a[0]} rhs={a[1]} dst={a[2]}",
))
register(IntrinsicSpec(
    name="plena.fp_sub_at",
    operand_scopes=(_scope.FPRAM, _scope.FPRAM, _scope.FPRAM, None),
    emit=lambda a: f"FP_SUB_AT   lhs={a[0]} rhs={a[1]} dst={a[2]} row={a[3]}",
))

register(IntrinsicSpec(
    name="plena.fp_mul",
    operand_scopes=(_scope.FPRAM, _scope.FPRAM, _scope.FPRAM),
    emit=lambda a: f"FP_MUL   lhs={a[0]} rhs={a[1]} dst={a[2]}",
))
register(IntrinsicSpec(
    name="plena.fp_mul_at",
    operand_scopes=(_scope.FPRAM, _scope.FPRAM, _scope.FPRAM, None),
    emit=lambda a: f"FP_MUL_AT   lhs={a[0]} rhs={a[1]} dst={a[2]} row={a[3]}",
))

register(IntrinsicSpec(
    name="plena.fp_max",
    operand_scopes=(_scope.FPRAM, _scope.FPRAM, _scope.FPRAM),
    emit=lambda a: f"FP_MAX   lhs={a[0]} rhs={a[1]} dst={a[2]}",
))
register(IntrinsicSpec(
    name="plena.fp_max_at",
    operand_scopes=(_scope.FPRAM, _scope.FPRAM, _scope.FPRAM, None),
    emit=lambda a: f"FP_MAX_AT   lhs={a[0]} rhs={a[1]} dst={a[2]} row={a[3]}",
))

register(IntrinsicSpec(
    name="plena.fp_exp",
    operand_scopes=(_scope.FPRAM, _scope.FPRAM),
    emit=lambda a: f"FP_EXP   src={a[0]} dst={a[1]}",
))
register(IntrinsicSpec(
    name="plena.fp_exp_at",
    operand_scopes=(_scope.FPRAM, _scope.FPRAM, None),
    emit=lambda a: f"FP_EXP_AT   src={a[0]} dst={a[1]} row={a[2]}",
))

register(IntrinsicSpec(
    name="plena.fp_reci",
    operand_scopes=(_scope.FPRAM, _scope.FPRAM),
    emit=lambda a: f"FP_RECI  src={a[0]} dst={a[1]}",
))
register(IntrinsicSpec(
    name="plena.fp_reci_at",
    operand_scopes=(_scope.FPRAM, _scope.FPRAM, None),
    emit=lambda a: f"FP_RECI_AT  src={a[0]} dst={a[1]} row={a[2]}",
))

register(IntrinsicSpec(
    name="plena.fp_sqrt",
    operand_scopes=(_scope.FPRAM, _scope.FPRAM),
    emit=lambda a: f"FP_SQRT  src={a[0]} dst={a[1]}",
))
register(IntrinsicSpec(
    name="plena.fp_sqrt_at",
    operand_scopes=(_scope.FPRAM, _scope.FPRAM, None),
    emit=lambda a: f"FP_SQRT_AT  src={a[0]} dst={a[1]} row={a[2]}",
))

register(IntrinsicSpec(
    name="plena.row_reduce_max",
    operand_scopes=(_scope.VRAM, _scope.FPRAM),
    emit=lambda a: f"ROW_REDUCE_MAX src={a[0]} dst={a[1]}",
))

register(IntrinsicSpec(
    name="plena.row_reduce_sum",
    operand_scopes=(_scope.VRAM, _scope.FPRAM),
    emit=lambda a: f"ROW_REDUCE_SUM src={a[0]} dst={a[1]}",
))

register(IntrinsicSpec(
    name="plena.row_exp",
    operand_scopes=(_scope.VRAM, _scope.VRAM),
    emit=lambda a: f"ROW_EXP  src={a[0]} dst={a[1]}",
))

register(IntrinsicSpec(
    name="plena.row_reci",
    operand_scopes=(_scope.VRAM, _scope.VRAM),
    emit=lambda a: f"ROW_RECI src={a[0]} dst={a[1]}",
))

register(IntrinsicSpec(
    name="plena.row_add_fp",
    operand_scopes=(_scope.VRAM, _scope.FPRAM, _scope.VRAM),
    emit=lambda a: f"ROW_ADD_FP src={a[0]} rhs={a[1]} dst={a[2]}",
))

register(IntrinsicSpec(
    name="plena.row_sub_fp",
    operand_scopes=(_scope.VRAM, _scope.FPRAM, _scope.VRAM),
    emit=lambda a: f"ROW_SUB_FP src={a[0]} rhs={a[1]} dst={a[2]}",
))

register(IntrinsicSpec(
    name="plena.row_mul_fp",
    operand_scopes=(_scope.VRAM, _scope.FPRAM, _scope.VRAM),
    emit=lambda a: f"ROW_MUL_FP src={a[0]} rhs={a[1]} dst={a[2]}",
))

register(IntrinsicSpec(
    name="plena.row_reduce_max_mask",
    operand_scopes=(_scope.VRAM, _scope.FPRAM, None),
    emit=lambda a: f"ROW_REDUCE_MAX_MASK src={a[0]} dst={a[1]} mask={a[2]}",
))
# `_at`: logical per-vector variant. Scalars are the source buffer's logical
# (dim2, dim3) indices; the emitter resolves them to the physical VRAM row,
# FP-state offset, and any packed-head V_MASK needed for narrow D tiles.
register(IntrinsicSpec(
    name="plena.row_reduce_max_at",
    operand_scopes=(_scope.VRAM, _scope.FPRAM, None, None),
    emit=lambda a: f"ROW_REDUCE_MAX_AT src={a[0]} dst={a[1]} dim2={a[2]} dim3={a[3]}",
))

register(IntrinsicSpec(
    name="plena.row_reduce_sum_mask",
    operand_scopes=(_scope.VRAM, _scope.FPRAM, None),
    emit=lambda a: f"ROW_REDUCE_SUM_MASK src={a[0]} dst={a[1]} mask={a[2]}",
))
register(IntrinsicSpec(
    name="plena.row_reduce_sum_at",
    operand_scopes=(_scope.VRAM, _scope.FPRAM, None, None),
    emit=lambda a: f"ROW_REDUCE_SUM_AT src={a[0]} dst={a[1]} dim2={a[2]} dim3={a[3]}",
))

register(IntrinsicSpec(
    name="plena.row_exp_mask",
    operand_scopes=(_scope.VRAM, _scope.VRAM, None),
    emit=lambda a: f"ROW_EXP_MASK src={a[0]} dst={a[1]} mask={a[2]}",
))
# row_exp_at: VRAM-only, scalars are the source buffer's logical (dim2, dim3).
register(IntrinsicSpec(
    name="plena.row_exp_at",
    operand_scopes=(_scope.VRAM, _scope.VRAM, None, None),
    emit=lambda a: f"ROW_EXP_AT src={a[0]} dst={a[1]} dim2={a[2]} dim3={a[3]}",
))

register(IntrinsicSpec(
    name="plena.row_reci_mask",
    operand_scopes=(_scope.VRAM, _scope.VRAM, None),
    emit=lambda a: f"ROW_RECI_MASK src={a[0]} dst={a[1]} mask={a[2]}",
))

register(IntrinsicSpec(
    name="plena.row_add_fp_mask",
    operand_scopes=(_scope.VRAM, _scope.FPRAM, _scope.VRAM, None),
    emit=lambda a: f"ROW_ADD_FP_MASK src={a[0]} rhs={a[1]} dst={a[2]} mask={a[3]}",
))

register(IntrinsicSpec(
    name="plena.row_sub_fp_mask",
    operand_scopes=(_scope.VRAM, _scope.FPRAM, _scope.VRAM, None),
    emit=lambda a: f"ROW_SUB_FP_MASK src={a[0]} rhs={a[1]} dst={a[2]} mask={a[3]}",
))
register(IntrinsicSpec(
    name="plena.row_sub_fp_at",
    operand_scopes=(_scope.VRAM, _scope.FPRAM, _scope.VRAM, None, None),
    emit=lambda a: f"ROW_SUB_FP_AT src={a[0]} rhs={a[1]} dst={a[2]} dim2={a[3]} dim3={a[4]}",
))

register(IntrinsicSpec(
    name="plena.row_mul_fp_mask",
    operand_scopes=(_scope.VRAM, _scope.FPRAM, _scope.VRAM, None),
    emit=lambda a: f"ROW_MUL_FP_MASK src={a[0]} rhs={a[1]} dst={a[2]} mask={a[3]}",
))
register(IntrinsicSpec(
    name="plena.row_mul_fp_at",
    operand_scopes=(_scope.VRAM, _scope.FPRAM, _scope.VRAM, None, None),
    emit=lambda a: f"ROW_MUL_FP_AT src={a[0]} rhs={a[1]} dst={a[2]} dim2={a[3]} dim3={a[4]}",
))
register(IntrinsicSpec(
    name="plena.row_add_fp_at",
    operand_scopes=(_scope.VRAM, _scope.FPRAM, _scope.VRAM, None, None),
    emit=lambda a: f"ROW_ADD_FP_AT src={a[0]} rhs={a[1]} dst={a[2]} dim2={a[3]} dim3={a[4]}",
))


# ---------------------------------------------------------------------------
# Slice variants (for kernels that need to copy a sub-region of an HBM
# tensor instead of the whole thing). The call signature is structured:
#
#     plena.dma_h2v_slice(src_buf, dst_buf, ndim,
#                         start_0, start_1, ..., start_{ndim-1},
#                         ext_0,   ext_1,   ..., ext_{ndim-1})
#
# Pass 1 in codegen.py packs (src_buf, starts, extents) into a BufferSlice
# and produces an HLIR Op of the same kind (no separate slice op kind --
# the HLIR Op's first buffer_arg is just BufferSlice instead of str).
#
# operand_scopes here is the MINIMUM signature -- variadic args (the
# starts and extents) are not scope-checked. The first two scopes are
# the fixed src/dst slots; everything after `None`s is filtered out.
# ---------------------------------------------------------------------------
register(IntrinsicSpec(
    name="plena.dma_h2v_slice",
    operand_scopes=(_scope.HBM, _scope.VRAM, None),  # src_parent, dst, ndim
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
