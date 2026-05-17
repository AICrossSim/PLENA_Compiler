"""Pass 3: turn HLIR (with addresses) into real PLENA ISA text.

Each HLIR op kind has a small dispatcher that pulls the right addresses
off the buffers and forwards them to ISAEmitter. This is intentionally
mechanical -- if you add a new op kind to `intrinsics.py`, you add one
case here too.

For BTMM specifically, the runtime convention is:
    - the actual `M_BTMM` instruction takes packed lhs (vram) + rhs (mram)
    - it does NOT itself write the result; the result is committed to
      VRAM by a paired `M_BMM_WO` instruction
The HLIR view collapses this to one `btmm` op that names lhs/rhs/dst;
this pass expands it into the `emit_btmm` + `emit_btmm_wo` pair so the
emitter contract is honoured.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

import tvm
from tvm import tir

from . import hlir as _hlir
from . import scope as _scope
from .expr_materializer import ExprMaterializer, MaterializedExpr
from .frontend.mid_ir.cluster_guard import MLEN as _HW_MLEN
from .isa_emitter import ISAEmitter
from .program_shim import ProgramShim


class IsaEmissionError(RuntimeError):
    pass


# Maximum unsigned literal that fits in the S_ADDI_INT three-operand
# immediate slot. opcode(6) + 2*operand(4) = 14 bits taken by other
# fields, leaving 32 - 14 = 18 bits for imm. Mirrors _S_ADDI_MAX in
# expr_materializer.py and _normalize_large_addi_immediates in
# tilelang_runtime_compier/tile_tensor_program/_program.py.
_S_ADDI_IMM_MAX = (1 << 18) - 1  # 262143


def _normalize_large_addi_immediates(asm_code: str) -> str:
    """Rewrite `S_ADDI_INT rd, rs1, imm` when imm overflows the 18-bit slot.

    Strategy:
      - rs1 == gp0:           LUI rd, hi    ; ADDI rd, rd, lo
      - rs1 != gp0, rd != rs1: LUI rd, hi   ; ADDI rd, rd, lo ; S_ADD_INT rd, rd, rs1
      - rs1 != gp0, rd == rs1: cannot expand text-only without a scratch reg.
        Emit a warning and leave the line untouched (the binary-stage
        instruction mask will then truncate it, producing wrong code that
        the user can spot via the warning).
    """
    lines: List[str] = []
    for raw_line in asm_code.splitlines():
        line = raw_line.rstrip("\n")
        stripped = line.strip()
        if not stripped or stripped.startswith(";"):
            lines.append(line)
            continue

        parts = stripped.split(None, 1)
        if len(parts) != 2 or parts[0] != "S_ADDI_INT":
            lines.append(line)
            continue

        operands = [item.strip() for item in parts[1].split(",")]
        if len(operands) != 3:
            lines.append(line)
            continue

        rd, rs1, imm_text = operands
        try:
            imm_value = int(imm_text)
        except ValueError:
            lines.append(line)
            continue

        if 0 <= imm_value <= _S_ADDI_IMM_MAX:
            lines.append(line)
            continue

        if imm_value < 0:
            print(
                f"[isa_pass] WARN: negative imm in {stripped!r}; "
                f"normalize pass only handles unsigned overflow."
            )
            lines.append(line)
            continue

        upper = imm_value >> 12
        lower = imm_value & 0xFFF

        if rs1 == "gp0":
            lines.append(f"S_LUI_INT {rd}, {upper}")
            lines.append(f"S_ADDI_INT {rd}, {rd}, {lower}")
            continue

        if rd != rs1:
            lines.append(f"S_LUI_INT {rd}, {upper}")
            lines.append(f"S_ADDI_INT {rd}, {rd}, {lower}")
            lines.append(f"S_ADD_INT {rd}, {rd}, {rs1}")
            continue

        print(
            f"[isa_pass] WARN: cannot expand large-imm S_ADDI_INT in-place "
            f"(rd==rs1=={rd}, imm={imm_value}): {stripped!r}. "
            f"Need a scratch register — fix the emitter to use a separate rd."
        )
        lines.append(line)

    normalized = "\n".join(lines)
    if asm_code.endswith("\n"):
        normalized += "\n"
    return normalized


class IsaEmitterPass:
    def __init__(self, shim: ProgramShim) -> None:
        self.shim = shim
        self.emitter = ISAEmitter(shim)
        # Symbol table: tir.Var -> currently-bound GP register id. Loop
        # bodies push entries on enter and pop on exit. ExprMaterializer
        # consults this table to resolve Var references in scalar args.
        self.symbol_table: Dict[tir.Var, int] = {}
        self.materializer = ExprMaterializer(shim, self.symbol_table)
        self._dispatch: Dict[str, Callable[[_hlir.HLIRModule, _hlir.Op], None]] = {
            "dma_h2v": self._emit_dma_h2v,
            "dma_h2m": self._emit_dma_h2m,
            "dma_v2h": self._emit_dma_v2h,
            "dma_h2v_slice": self._emit_dma_h2v_slice,
            "dma_h2m_slice": self._emit_dma_h2m_slice,
            "dma_v2h_slice": self._emit_dma_v2h_slice,
            "btmm": self._emit_btmm,
            "btmv": self._emit_btmv,
            "mm": self._emit_mm,
            "mm_slot": self._emit_mm_slot,
            "matmul": self._emit_matmul,
            "mv": self._emit_mv,
            # 1-D vector VRAM ops — each op handles ONE logical 1D
            # vector of length ``n_elem``. The emitter unrolls it into
            # ``ceil(n_elem / mlen)`` HW issues; multi-row tiles are
            # expressed by the lowering wrapping ``v_*`` inside a
            # ``for row`` so a 2-D tile becomes M explicit per-row
            # vector ops in the HLIR.
            "v_zero": self._emit_v_zero,
            "v_add": self._emit_v_add,
            "v_sub": self._emit_v_sub,
            "v_mul": self._emit_v_mul,
            "v_exp": self._emit_v_exp,
            "v_reci": self._emit_v_reci,
            "v_sqrt": self._emit_v_sqrt,
            "fp_copy_at": self._emit_fp_copy_at,
            "fp_zero_at": self._emit_fp_zero_at,
            "fp_add_at": self._emit_fp_add_at,
            "fp_sub_at": self._emit_fp_sub_at,
            "fp_mul_at": self._emit_fp_mul_at,
            "fp_max_at": self._emit_fp_max_at,
            "fp_exp_at": self._emit_fp_exp_at,
            "fp_reci_at": self._emit_fp_reci_at,
            "fp_sqrt_at": self._emit_fp_sqrt_at,
            "v_fp_transfer_slice_v_to_fp": self._emit_v_fp_transfer_slice_v_to_fp,
            "v_fp_transfer_slice_fp_to_v": self._emit_v_fp_transfer_slice_fp_to_v,
            "copy_v_to_v": self._emit_copy_v_to_v,
            # Row-level VRAM/FP ops. Contract: one HLIR op = one HW
            # instruction over a SINGLE row. Multi-row callers must wrap
            # in an outer HLIR ``for row``.
            "row_reduce_max_at": self._emit_row_reduce_max_at,
            "row_reduce_sum_at": self._emit_row_reduce_sum_at,
            "row_exp": self._emit_row_exp,
            "row_sub_fp": self._emit_row_sub_fp,
            "row_mul_fp": self._emit_row_mul_fp,
            "row_add_fp": self._emit_row_add_fp,
            "for": self._emit_for,
        }

    def run(self, mod: _hlir.HLIRModule) -> str:
        _hlir.assert_addresses_resolved(mod)
        # Emit a header so generated dumps are easy to identify in build/.
        self.shim.compiler.generated_code = (
            f"; PLENA ISA  --  kernel: {mod.name}\n"
            f"; generated by tilelang_tvm_compiler (real-ISA path)\n"
            f"; ============================================================\n"
            f"; buffer layout:\n"
        )
        for buf in mod.buffers.values():
            self.shim.compiler.generated_code += (
                f";   {buf.name:<10s} scope={buf.scope:<5s} addr={buf.address}  "
                f"shape={'x'.join(str(s) for s in buf.shape)}\n"
            )
        self.shim.compiler.generated_code += (
            "; ============================================================\n\n"
        )

        ra = self.shim.compiler.register_allocator
        for i, op in enumerate(mod.ops):
            handler = self._dispatch.get(op.kind)
            if handler is None:
                raise IsaEmissionError(
                    f"no ISA dispatcher for HLIR op kind {op.kind!r}. "
                    f"Either add it to isa_pass dispatch table, or guard "
                    f"the op out of HLIR earlier."
                )
            ra.push_site(f"op[{i}] {op.kind}")
            try:
                handler(mod, op)
            finally:
                ra.pop_site()
        self.shim.compiler.generated_code = _normalize_large_addi_immediates(
            self.shim.compiler.generated_code
        )
        return self.shim.compiler.generated_code

    @staticmethod
    def _logical_2d(
        shape: Tuple[int, ...], layout: str = "BSHD",
    ) -> Tuple[int, int]:
        return _hlir.logical_2d_extents(shape, layout)

    def _vram_row_shape(self, buf: _hlir.Buffer, op_kind: str, role: str) -> Tuple[int, int]:
        _check_scope(buf, _scope.VRAM, op_kind, role)
        rows, cols = self._logical_2d(buf.shape, buf.layout)
        if cols != self.shim.mlen:
            raise IsaEmissionError(
                f"{op_kind} {role} buffer {buf.name!r} must have logical row width "
                f"mlen={self.shim.mlen}; got logical 2D ({rows}, {cols})"
            )
        return rows, cols

    def _resolve_row_at_coords(
        self,
        buf: _hlir.Buffer,
        op_kind: str,
        role: str,
        row_expr,
        head_expr,
        op_axes: Optional[Tuple[Tuple[str, int], ...]] = None,
    ) -> Tuple[tir.PrimExpr, tir.PrimExpr | None]:
        """Translate logical ``(head=H-idx, row=S-idx)`` coords on a
        VRAM buffer into a physical vram-row index + optional V_MASK.

        Driven by the per-op ``op_axes`` table that mid_ir → HLIR
        lowering stamps on ``hlir.Op.buffer_axes``. Each entry pairs
        a role string with the dim's extent:

            ``"simd"``     — innermost D / vector axis
            ``"cluster"``  — lane / head axis
            ``"batch"``    — row-fanout axis (rows OR degenerate
                             leading B placeholder; pick the rows
                             one by largest extent)

        Computation:

            flat_row = row * row_stride + head * head_stride

        where ``row_stride`` / ``head_stride`` is the product of every
        physical dim's extent strictly inside that role's position
        (i.e. between the role and the innermost dim).

        Then:

          * ``D >= MLEN`` and ``D % MLEN == 0``: each flat_row is one
            full mlen vector. ``vram_row = flat_row``; no mask.
          * ``D < MLEN`` (``lane_count = MLEN/D``): ``lane_count``
            consecutive flat_rows pack into one mlen-row.
            ``vram_row = flat_row // lane_count``,
            ``mask = 1 << (flat_row % lane_count)``.
        """
        _check_scope(buf, _scope.VRAM, op_kind, role)
        if not buf.shape:
            raise IsaEmissionError(
                f"{op_kind} {role} buffer {buf.name!r}: empty shape"
            )
        mlen = int(self.shim.mlen)
        rank = len(buf.shape)

        if op_axes is None:
            raise IsaEmissionError(
                f"{op_kind} {role} buffer {buf.name!r}: op_axes required "
                f"(callers must thread per-op axes from hlir.Op.buffer_axes "
                f"so this resolver can locate the rows / cluster / D dim "
                f"by role tag instead of guessing from shape)."
            )
        if len(op_axes) != rank:
            raise IsaEmissionError(
                f"{op_kind} {role} buffer {buf.name!r}: op_axes rank "
                f"{len(op_axes)} doesn't match buffer rank {rank} "
                f"(shape={list(buf.shape)} op_axes={list(op_axes)})"
            )

        # Locate dims by role. ``simd`` must exist and is the innermost
        # vector axis. ``cluster`` is optional (rank-2 fragments don't
        # have one). ``batch`` may appear multiple times — pick the one
        # with the largest extent as the rows axis; extent-1 batch
        # entries are pad-to-4D / cluster-expand placeholders.
        d_axis: Optional[int] = None
        cluster_dim: Optional[int] = None
        rows_axis: Optional[int] = None
        rows_extent = -1
        for i, (role_name, _extent) in enumerate(op_axes):
            if role_name == "simd":
                d_axis = i
            elif role_name == "cluster":
                cluster_dim = i
            elif role_name == "batch":
                if int(_extent) > rows_extent:
                    rows_extent = int(_extent)
                    rows_axis = i
        if d_axis is None:
            raise IsaEmissionError(
                f"{op_kind} {role} buffer {buf.name!r}: no ``simd`` axis "
                f"in op_axes {list(op_axes)}"
            )
        d_dim = int(buf.shape[d_axis])

        if rank >= 3 and rows_axis is not None:
            head_stride = 1
            if cluster_dim is not None:
                lo = min(cluster_dim, d_axis) + 1
                hi = max(cluster_dim, d_axis)
                for axis in range(lo, hi):
                    head_stride *= int(buf.shape[axis])
            row_stride = 1
            lo = min(rows_axis, d_axis) + 1
            hi = max(rows_axis, d_axis)
            for axis in range(lo, hi):
                row_stride *= int(buf.shape[axis])
            terms = []
            if head_stride == 1:
                terms.append(head_expr)
            elif head_stride > 1:
                terms.append(
                    tir.Mul(head_expr, tir.IntImm("int32", head_stride))
                )
            if row_stride == 1:
                terms.append(row_expr)
            else:
                terms.append(
                    tir.Mul(row_expr, tir.IntImm("int32", row_stride))
                )
            flat_row = terms[0]
            for t in terms[1:]:
                flat_row = tir.Add(flat_row, t)
        else:
            flat_row = row_expr

        # Case 1: D-wide row is at least a full mlen vector. Each
        # flat_row IS one full mlen-row; no mask.
        if d_dim >= mlen and d_dim % mlen == 0:
            return flat_row, None

        # Case 2: D < MLEN — ``lane_count`` flat_rows pack into one
        # mlen-row. vram_row = flat_row // lane_count;
        # mask = 1 << (flat_row % lane_count).
        if mlen % d_dim == 0:
            lane_count = mlen // d_dim
            log2_lc = (lane_count - 1).bit_length()
            if (1 << log2_lc) != lane_count:
                raise IsaEmissionError(
                    f"{op_kind} {role} buffer {buf.name!r}: lane_count "
                    f"{lane_count} (=MLEN/{d_dim}) is not a power of two; "
                    f"shift / mask shortcut for the narrow-D path requires it."
                )
            vram_row_expr = tir.shift_right(
                flat_row, tir.IntImm("int32", log2_lc),
            )
            # PLENA has no bitwise-AND; ``flat_row % lane_count`` is
            # ``flat_row - ((flat_row >> k) << k)``.
            quotient_shifted_back = tir.shift_left(
                vram_row_expr, tir.IntImm("int32", log2_lc),
            )
            col_in_row = tir.Sub(flat_row, quotient_shifted_back)
            mask_expr = tir.shift_left(tir.IntImm("int32", 1), col_in_row)
            return vram_row_expr, mask_expr

        raise IsaEmissionError(
            f"{op_kind} {role} buffer {buf.name!r}: innermost dim {d_dim} "
            f"is neither a multiple of MLEN ({mlen}) nor a divisor — no "
            f"unified row_*_at addressing path for it."
        )

    def _resolve_fp_scalar_addr_arg(
        self,
        mod: _hlir.HLIRModule,
        arg,
        op_kind: str,
        role: str,
    ):
        if isinstance(arg, _hlir.BufferElement):
            buf = mod.get_buffer(arg.buffer)
            _check_scope(buf, _scope.FPRAM, op_kind, role)
            if len(arg.indices) != len(buf.shape):
                raise IsaEmissionError(
                    f"{op_kind} {role} buffer element {buf.name!r} has index rank {len(arg.indices)} "
                    f"but buffer shape rank {len(buf.shape)}"
                )
            offset = tir.IntImm("int32", 0)
            stride = 1
            for dim, idx in zip(reversed(buf.shape), reversed(arg.indices)):
                idx_expr = tir.IntImm("int32", int(idx)) if isinstance(idx, int) else idx
                term = idx_expr if stride == 1 else tir.Mul(
                    idx_expr, tir.IntImm("int32", int(stride)),
                )
                offset = term if stride == 1 and isinstance(offset, tir.IntImm) and int(offset.value) == 0 else tir.Add(term, offset)
                stride *= int(dim)
            return tir.Add(tir.IntImm("int32", int(buf.address)), offset)
        if isinstance(arg, (int, tir.PrimExpr)):
            return arg
        raise IsaEmissionError(
            f"{op_kind} {role} expects an FPRAM address or buffer element ref; "
            f"got {type(arg).__name__}: {arg!r}"
        )

    def _emit_fp_scalar_op_at(
        self,
        mod: _hlir.HLIRModule,
        op: _hlir.Op,
        *,
        kernel_op: str,
    ) -> None:
        # FP `_at` operands are scalar fpram addresses (PrimExpr or int),
        # already including any per-slot base offset. We materialize each
        # address into its own GP register and emit S_LD_FP / S_ST_FP.
        if kernel_op in {"copy", "exp", "reci", "sqrt"}:
            expected = 2
        else:
            expected = 3
        if len(op.scalar_args) != expected:
            raise IsaEmissionError(
                f"{op.kind} expects {expected} scalar address args, got {len(op.scalar_args)}"
            )

        addr_exprs = [
            self._resolve_fp_scalar_addr_arg(mod, a, op.kind, f"arg{i}")
            for i, a in enumerate(op.scalar_args)
        ]
        # Materialize one address expression at a time, commit its ISA to
        # ``generated_code`` immediately, and ``pin_gp`` the result reg so
        # the next materialize() call cannot auto-spill it.
        #
        # Why pinning is required:
        # ExprMaterializer eagerly frees operand registers after writing a
        # binop's ISA text. ``allocate_gp`` then auto-spills the
        # most-recently-allocated in-use reg when pressure is high — and
        # that "most-recently-allocated" reg can be the previous mats[i]'s
        # final reg itself. The spill stores the value to IntRAM but the
        # MaterializedExpr's ``register`` field still names the same reg,
        # which the next materialize() then overwrites. Net effect: by
        # the time we emit S_LD_FP/S_MUL_FP, mats[0]/mats[1] both point
        # at a reg holding mats[2]'s value. Pinning blocks that path.
        ra = self.shim.compiler.register_allocator
        mats: List[MaterializedExpr] = []
        for a in addr_exprs:
            m = self.materializer.materialize(a)
            self.shim.compiler.generated_code += m.isa
            ra.pin_gp(m.register)
            mats.append(m)

        try:
            lines = [f"; fp scalar task {op.annotations.get('intrinsic', op.kind)} op={kernel_op}"]
            if kernel_op in {"copy", "exp", "reci", "sqrt"}:
                gp_src, gp_dst = mats[0].register, mats[1].register
                lines.append(f"S_LD_FP f1, gp{gp_src}, 0")
                if kernel_op == "exp":
                    lines.append("S_EXP_FP f1, f1, 0")
                elif kernel_op == "reci":
                    lines.append("S_RECI_FP f1, f1")
                elif kernel_op == "sqrt":
                    lines.append("S_SQRT_FP f1, f1")
                lines.append(f"S_ST_FP f1, gp{gp_dst}, 0")
            else:
                gp_lhs, gp_rhs, gp_dst = mats[0].register, mats[1].register, mats[2].register
                opcode = {
                    "add": "S_ADD_FP",
                    "sub": "S_SUB_FP",
                    "mul": "S_MUL_FP",
                    "max": "S_MAX_FP",
                }[kernel_op]
                lines.append(f"S_LD_FP f1, gp{gp_lhs}, 0")
                lines.append(f"S_LD_FP f2, gp{gp_rhs}, 0")
                lines.append(f"{opcode} f1, f1, f2")
                lines.append(f"S_ST_FP f1, gp{gp_dst}, 0")
            self.shim.compiler.generated_code += "\n".join(lines) + "\n"
        finally:
            for m in reversed(mats):
                ra.unpin_gp(m.register)
                m.release()

    def _tile_layout_strides(self, buf: _hlir.Buffer):
        """Element-strides for the 7D physical layout of a VRAM/MRAM buffer.

        Mirrors the stride math ``_slice_tile_grid`` uses for the
        ``tile_layout is not None`` branch but factored out so non-DMA
        emitters can read it. Returns a dict::

            {
              "d_tiles":       outer d-tile count,
              "s_tiles":       outer s-tile count,
              "h_groups":      outer h-group count,
              "logical_b":     batch count (almost always 1),
              "mlen":          inner s-tile height,
              "lane_count":    inner lane count (1 when D >= mlen),
              "d_inner":       inner d width (mlen when D >= mlen),
              "s_inner_stride": elements between consecutive s_inner rows
                                in the same tile (= lane_count * d_inner),
              "h_grp_stride":  elements between consecutive h_groups,
              "s_tile_stride": elements between consecutive s_tiles,
              "d_tile_stride": elements between consecutive d_tiles,
            }

        Buffers without a ``tile_layout`` (rank < 4, or 4D with
        ``d_tiles == s_tiles == h_groups == logical_b == 1`` flattened
        away upstream) get a ``None`` return — callers fall back to
        their pre-7D row-major-flat path.
        """
        tl = getattr(buf, "tile_layout", None)
        if tl is None:
            return None
        inner_d = int(tl.d_inner)
        inner_lane = int(tl.lane_count) * inner_d
        inner_s = int(tl.mlen) * inner_lane
        b_stride = inner_s
        inner_b = int(tl.logical_b) * inner_s
        h_grp_stride = inner_b
        s_tile_stride = int(tl.h_groups) * inner_b
        d_tile_stride = int(tl.s_tiles) * s_tile_stride
        return {
            "d_tiles":        int(tl.d_tiles),
            "s_tiles":        int(tl.s_tiles),
            "h_groups":       int(tl.h_groups),
            "logical_b":      int(tl.logical_b),
            "mlen":           int(tl.mlen),
            "lane_count":     int(tl.lane_count),
            "d_inner":        int(tl.d_inner),
            "s_inner_stride": inner_lane,
            "h_grp_stride":   h_grp_stride,
            "s_tile_stride":  s_tile_stride,
            "d_tile_stride":  d_tile_stride,
            "b_stride":       b_stride,
        }

    def _buffer_tile_grid_iter(self, buf: _hlir.Buffer):
        """Yield every mlen-row of ``buf`` in physical address order.

        For each ``(d_tile, s_tile, h_grp, b, s_inner)`` cell of the
        buffer's 7D physical layout, yields
        ``(d_tile, s_tile, h_grp, b, s_inner, phys_offset)`` where
        ``phys_offset`` is in *elements* relative to ``buf.address``.
        One yielded entry == one HW mlen-wide vector op
        (``V_*_VV`` / ``V_*_VF`` / ``V_RED_*`` / ``V_EXP_V`` / etc.).

        Walks the outer tile grid in physical-address order
        (d_tile slowest, b fastest at the tile level), then steps
        through s_inner inside each tile. ``s_inner`` covers
        ``tl.mlen`` rows because each s_inner row is one HW vector
        (its width is ``lane_count * d_inner == mlen`` by tile-layout
        invariant). Use this in any emitter that wants to issue one
        HW vector op per mlen-row covering the whole buffer (v_zero,
        tile_add, tile_mul, …) — it hides the 7D layout behind one
        loop and stays in lock-step with what ``_slice_tile_grid``
        walks for DMAs.

        Buffers that aren't 4D (no ``tile_layout``) raise — callers
        with 1D / 2D scratch must use the legacy flat-offset
        emitters; everything pad-to-4D'd by to_plena hits this path.
        """
        info = self._tile_layout_strides(buf)
        if info is None:
            raise IsaEmissionError(
                f"_buffer_tile_grid_iter: buffer {buf.name!r} has no "
                f"tile_layout (rank={len(buf.shape)}, shape={tuple(buf.shape)}) "
                f"— this helper only handles 4D buffers; 1D/2D callers must "
                f"stay on the flat-offset path."
            )
        s_inner_stride = info["s_inner_stride"]
        for d_tile in range(info["d_tiles"]):
            for s_tile in range(info["s_tiles"]):
                for h_grp in range(info["h_groups"]):
                    for b in range(info["logical_b"]):
                        tile_base = (
                            d_tile * info["d_tile_stride"]
                            + s_tile * info["s_tile_stride"]
                            + h_grp * info["h_grp_stride"]
                            + b * info["b_stride"]
                        )
                        for s_inner in range(info["mlen"]):
                            phys = tile_base + s_inner * s_inner_stride
                            yield (d_tile, s_tile, h_grp, b,
                                   s_inner, phys)

    def _logical_to_phys_row_offset(
        self,
        buf: _hlir.Buffer,
        region: _hlir.VramRegion,
    ):
        """Translate a single-row ``VramRegion`` into the physical
        7D *mlen-row base* offset (in elements, relative to
        ``buf.address``) plus an optional packed-head mask expression.

        Returns ``(phys_offset_expr, mask_expr_or_None, info)``.

        ``region`` must describe exactly one logical row:
        ``extents = (..., 1, ..., 1, D_full)`` with non-D extents
        equal to 1. ``starts`` is 4 entries in physical-axis order
        (matching ``buf.shape``); the helper routes each entry to
        the right stride **by the buffer's role tags**, so it works
        for any lane-fusion mode:

            * col_pack:  shape=(B=1, S, H=lane, D_narrow), cluster_dim=2
              → starts[2] is the lane index, gets split into
              (h_grp, lane); mask = 1 << lane.
            * row_stack: shape=(B=lane, S, H=1, MLEN), cluster_dim=0
              → starts[0] is the lane index, same split logic.
            * bshd_lift / no cluster: cluster_dim is None, every
              non-D, non-rows axis is either a B placeholder or H
              placeholder; their starts contribute via the matching
              stride (b_stride / h_grp_stride). lane_count == 1 here,
              so mask_expr is None.
        """
        info = self._tile_layout_strides(buf)
        if info is None:
            raise IsaEmissionError(
                f"_logical_to_phys_row_offset: buffer {buf.name!r} has no "
                f"tile_layout (rank={len(buf.shape)}, shape={tuple(buf.shape)})"
            )
        mlen = info["mlen"]
        lane_count = info["lane_count"]
        if len(region.starts) != 4 or len(region.extents) != 4:
            raise IsaEmissionError(
                f"_logical_to_phys_row_offset: region {region.parent!r} "
                f"must be 4D; got starts={tuple(region.starts)} "
                f"extents={tuple(region.extents)}"
            )

        cluster_dim = getattr(buf, "cluster_dim", None)
        rank = len(buf.shape)
        d_axis = rank - 1

        # Per-axis role assignment (matches the layout
        # ``_hlir_axes_for_buffer`` produces in mid_ir): the innermost
        # axis is "d", the cluster_dim (if set) is "lane", everything
        # else is "batch". Among batch axes the one with the largest
        # extent acts as "rows"; the rest are pad placeholders that
        # carry a B=1 or H=1 stride term.
        roles: List[str] = []
        for i in range(rank):
            if i == d_axis:
                roles.append("d")
            elif cluster_dim is not None and i == cluster_dim:
                roles.append("lane")
            else:
                roles.append("batch")

        def _to_expr(x):
            if isinstance(x, int):
                return tir.IntImm("int32", int(x))
            return x

        # Per-physical-axis "step one unit along this axis" stride
        # table (in elements). Indexed by axis position the same way
        # ``roles`` is, so any axis-handling branch (rows / lane /
        # batch placeholder) can look up its stride without caring
        # about which role label happens to live there. The S axis
        # gets its s_inner_stride here; the s_tile/s_inner split
        # for the multi-s_tile case is handled in the i==1 branch.
        axis_stride = [
            info["b_stride"],
            info["s_inner_stride"],
            info["h_grp_stride"],
            1,
        ]

        # Per-axis stride contribution. We iterate physical axes
        # in order so any 7D nuance (s_tile / s_inner split when
        # s_tiles > 1) lives next to its axis's stride choice.
        terms: List = []
        mask_expr = None

        for i, role in enumerate(roles):
            start = _to_expr(region.starts[i])
            if role == "d":
                # D start is folded into d_tile bump by the caller's
                # outer loop; the per-issue base is always at the
                # logical row's d-tile=0 chunk. Non-zero d-starts are
                # not supported here.
                if not (isinstance(region.starts[i], (int, tir.IntImm))
                        and (int(region.starts[i])
                             if isinstance(region.starts[i], int)
                             else int(region.starts[i].value)) == 0):
                    raise IsaEmissionError(
                        f"row_*_at on {buf.name!r}: d-axis start must be "
                        f"0; got {region.starts[i]!r}"
                    )
                continue
            if role == "lane":
                # col_pack puts cluster_dim at axis 2 (H) with
                # lane_count>1 (within-mlen packing); axis_stride[2] =
                # h_grp_stride is correct there.
                # row_stack puts cluster_dim at axis 0 (B) with
                # lane_count==1 (no packed-head — d already fills mlen).
                # Per the M_BMM_WO / M_BMV_WO hardware writeback in
                # ``transactional_emulator/src/main.rs``, lane j's data
                # lands at ``vec_base + j * (per_lane_elems)`` where
                # per_lane_elems = product(shape[1:]) — i.e. the flat
                # row-major stride for axis 0. For S=mlen this equals
                # ``mlen*inner_lane = b_stride`` (default works); for
                # S<mlen (flash_decode rows=1 S_loc) b_stride overshoots.
                lane_stride = axis_stride[i]
                if lane_count == 1 and i == 0:
                    lane_stride = 1
                    for dim in buf.shape[i + 1:]:
                        lane_stride *= int(dim)
                if lane_count > 1:
                    h_grp = tir.floordiv(
                        start, tir.IntImm("int32", lane_count),
                    )
                    lane = tir.floormod(
                        start, tir.IntImm("int32", lane_count),
                    )
                    if lane_stride:
                        terms.append(
                            tir.Mul(h_grp,
                                    tir.IntImm("int32", lane_stride))
                        )
                    mask_expr = tir.shift_left(
                        tir.IntImm("int32", 1), lane,
                    )
                else:
                    if lane_stride:
                        terms.append(
                            tir.Mul(start,
                                    tir.IntImm("int32", lane_stride))
                        )
                continue
            # role == "batch": could be the rows axis or a degenerate
            # placeholder. The stride is determined by where this axis
            # sits in the physical layout.
            #
            # Two physical positions matter:
            #   * S row (s_inner inside an s_tile, multiplied by
            #     ``s_inner_stride``). When s_tiles > 1 the value is
            #     also split into (s_tile, s_inner).
            #   * B placeholder (a leading batch dim). It rides on
            #     ``b_stride``.
            #   * H placeholder (when cluster_dim is at a different
            #     position than this batch axis, e.g. row_stack puts
            #     H=1 at axis 2 with role "batch"). It rides on
            #     ``h_grp_stride``.
            #
            # We disambiguate by axis index: the axis index immediately
            # before d_axis (or cluster_dim, whichever is later) is
            # treated as H if it differs from the rows position. The
            # leading axis is B. The S axis is the only batch axis
            # whose extent in ``buf.shape`` matches a "real" rows
            # dimension (not 1).
            #
            # In practice we route by axis index:
            #   i == 0 and cluster_dim != 0 → B placeholder (b_stride)
            #   i == 1 → S (s_inner_stride / s_tile_stride split)
            #   i == 2 and cluster_dim != 2 → H placeholder (h_grp_stride)
            if i == 0:
                if info["b_stride"]:
                    terms.append(
                        tir.Mul(start, tir.IntImm("int32", info["b_stride"]))
                    )
            elif i == 1:
                if info["s_tiles"] > 1:
                    s_tile = tir.floordiv(start, tir.IntImm("int32", mlen))
                    s_inner = tir.floormod(start, tir.IntImm("int32", mlen))
                    terms.append(
                        tir.Mul(s_tile,
                                tir.IntImm("int32", info["s_tile_stride"]))
                    )
                    terms.append(
                        tir.Mul(s_inner,
                                tir.IntImm("int32", info["s_inner_stride"]))
                    )
                else:
                    if info["s_inner_stride"] == 1:
                        terms.append(start)
                    else:
                        terms.append(
                            tir.Mul(start,
                                    tir.IntImm("int32", info["s_inner_stride"]))
                        )
            elif i == 2:
                if info["h_grp_stride"]:
                    terms.append(
                        tir.Mul(start,
                                tir.IntImm("int32", info["h_grp_stride"]))
                    )
            else:
                raise IsaEmissionError(
                    f"row_*_at on {buf.name!r}: unexpected batch axis at "
                    f"physical index {i}"
                )

        if not terms:
            return tir.IntImm("int32", 0), mask_expr, info
        expr = terms[0]
        for t in terms[1:]:
            expr = tir.Add(expr, t)
        return expr, mask_expr, info

    def _region_origin_offset(self, buf: _hlir.Buffer,
                              region) -> "tir.PrimExpr | int":
        """Translate a Region's ``starts`` into a physical element
        offset against ``buf.address``.

        Handles the cluster axis specially: when the cluster axis is
        packed-head (``lane_count > 1``), an index value < lane_count
        is a within-mlen lane segment whose stride is ``d_inner``
        (not ``h_grp_stride``). Larger indices split into
        ``h_grp = idx // lane_count`` (walks ``h_grp_stride``) and
        ``lane = idx % lane_count`` (walks ``d_inner``). For non-
        packed buffers (``lane_count == 1``) the cluster axis stride
        is ``h_grp_stride`` directly.
        """
        tl_info = self._tile_layout_strides(buf)
        if tl_info is None:
            for i, s in enumerate(region.starts):
                if isinstance(s, int) and s == 0:
                    continue
                if isinstance(s, tir.IntImm) and int(s.value) == 0:
                    continue
                raise IsaEmissionError(
                    f"_region_origin_offset: {region.parent!r} non-zero "
                    f"start at axis {i} but parent has no tile_layout"
                )
            return 0
        cluster_dim = getattr(buf, "cluster_dim", None)
        lane_count = int(tl_info["lane_count"])
        d_inner = int(tl_info["d_inner"])
        h_grp_stride = int(tl_info["h_grp_stride"])
        s_inner_stride = int(tl_info["s_inner_stride"])
        b_stride = int(tl_info["b_stride"])

        def _is_zero(x) -> bool:
            if isinstance(x, int) and x == 0:
                return True
            if isinstance(x, tir.IntImm) and int(x.value) == 0:
                return True
            return False

        def _mul(expr, k: int):
            if k == 0:
                return tir.IntImm("int32", 0)
            if isinstance(expr, int):
                return tir.IntImm("int32", expr * k)
            if isinstance(expr, tir.IntImm):
                return tir.IntImm("int32", int(expr.value) * k)
            if k == 1:
                return expr
            return tir.Mul(expr, tir.IntImm("int32", k))

        terms = []
        for i, s in enumerate(region.starts):
            if _is_zero(s):
                continue
            if (cluster_dim is not None and i == cluster_dim
                    and lane_count == 1 and i == 0):
                # row_stack lane axis: B carries lane stacking, no
                # packed-head (lane_count == 1). Each lane occupies
                # ``shape[1] * shape[2] * shape[3]`` elements
                # (logical_s * h_groups_etc * inner_lane in the 7D
                # picture), NOT the per-axis-table ``b_stride =
                # mlen * inner_lane`` (which assumes B is outer batch
                # over a full-mlen s_tile). For S_loc with S=mlen the
                # two values coincide; for S<mlen (flash_decode's
                # rows=1 S_loc) b_stride overshoots and lane 1's data
                # ends up past the buffer.
                lane_stride = 1
                for dim in buf.shape[i + 1:]:
                    lane_stride *= int(dim)
                terms.append(_mul(s, lane_stride))
                continue
            if cluster_dim is not None and i == cluster_dim and lane_count > 1:
                # Packed-head lane axis. For a static int we split
                # into (h_grp, lane) the usual way. For a PrimExpr
                # (typically the cluster phase var, which mid_ir
                # guarantees is < lane_count) we skip the floordiv /
                # floormod split and emit ``s * d_inner`` directly —
                # the floordiv/floormod expansion materialises into
                # an SRLI+SLLI+SUB chain that easily exhausts the
                # 16-GP budget when nested inside outer loops, and
                # the result simplifies to the same expression.
                if isinstance(s, int):
                    h_grp = s // lane_count
                    lane = s % lane_count
                    if h_grp:
                        terms.append(
                            tir.IntImm("int32", h_grp * h_grp_stride)
                        )
                    if lane:
                        terms.append(
                            tir.IntImm("int32", lane * d_inner)
                        )
                elif isinstance(s, tir.IntImm):
                    val = int(s.value)
                    h_grp = val // lane_count
                    lane = val % lane_count
                    if h_grp:
                        terms.append(
                            tir.IntImm("int32", h_grp * h_grp_stride)
                        )
                    if lane:
                        terms.append(
                            tir.IntImm("int32", lane * d_inner)
                        )
                else:
                    terms.append(_mul(s, d_inner))
                continue
            # Regular axis: pick stride from per-axis table.
            if i == 0:
                stride = b_stride
            elif i == 1:
                stride = s_inner_stride
            elif i == 2:
                stride = h_grp_stride
            else:
                stride = 1
            terms.append(_mul(s, stride))

        terms = [t for t in terms
                 if not (isinstance(t, tir.IntImm) and int(t.value) == 0)]
        if not terms:
            return 0
        acc = terms[0]
        for t in terms[1:]:
            acc = tir.Add(acc, t)
        return acc

    def _d_tile_info(self, buf: _hlir.Buffer) -> Tuple[int, int]:
        """Return ``(n_d_tiles, d_tile_stride_elems)`` for a VRAM buffer.

        Thin convenience wrapper over ``_tile_layout_strides`` —
        ``row_*_at`` emitters only ever walk the d-tile axis (the row /
        head coords pick a specific s_inner + h_grp + b within one
        d-tile plane), so they just need the outer d-tile count and
        the bump amount.
        """
        info = self._tile_layout_strides(buf)
        if info is None or info["d_tiles"] <= 1:
            return 1, 0
        return info["d_tiles"], info["d_tile_stride"]

    def _emit_row_scalar_op_at(
        self,
        mod: _hlir.HLIRModule,
        op: _hlir.Op,
        *,
        row_op: str,
        reduce: bool = False,
        masked: bool = False,
        has_fp: bool = False,
    ) -> None:
        """Row-scalar HW vector op on a single logical row of VRAM.

        Region schema (every variant):
            * reduce_max / reduce_sum:
                buffer_args = [src_region]
                scalar_args = [fp_addr (BufferElement)]
            * exp / reci (no FP operand):
                buffer_args = [src_region, dst_region]
                scalar_args = []
            * add / sub / mul fp:
                buffer_args = [src_region, dst_region]
                scalar_args = [fp_addr (BufferElement)]

        ``src_region`` / ``dst_region`` are ``VramRegion`` with 4D
        BSHD starts/extents picking exactly one (b, s, h) logical
        row. ``extents`` must be ``(1, 1, 1, D_full)`` — the emitter
        walks d_tiles itself. ``starts[2]`` (the H index) is *not*
        clipped to a single lane in packed-head buffers: it carries
        the actual head idx (0..head_count-1), and the emitter splits
        it into (h_grp, lane) — h_grp picks the mlen-row, lane drives
        the ``V_MASK`` bit so the V_*_VF only updates the target
        head's data.
        """
        has_fp = has_fp or reduce
        if reduce:
            if len(op.buffer_args) != 1:
                raise IsaEmissionError(
                    f"{op.kind} expects 1 buffer_arg (src region); "
                    f"got {len(op.buffer_args)}"
                )
            expected_scalar = 1
        elif has_fp:
            if len(op.buffer_args) != 2:
                raise IsaEmissionError(
                    f"{op.kind} expects 2 buffer_args (src/dst regions); "
                    f"got {len(op.buffer_args)}"
                )
            expected_scalar = 1
        else:
            if len(op.buffer_args) != 2:
                raise IsaEmissionError(
                    f"{op.kind} expects 2 buffer_args (src/dst regions); "
                    f"got {len(op.buffer_args)}"
                )
            expected_scalar = 0
        if len(op.scalar_args) != expected_scalar:
            raise IsaEmissionError(
                f"{op.kind} expects {expected_scalar} scalar_args; "
                f"got {len(op.scalar_args)}"
            )
        for slot, name in enumerate(
            ("src",) if reduce else ("src", "dst")
        ):
            if not isinstance(op.buffer_args[slot], _hlir.VramRegion):
                raise IsaEmissionError(
                    f"{op.kind} {name}: expected VramRegion, got "
                    f"{type(op.buffer_args[slot]).__name__}"
                )

        src_region: _hlir.VramRegion = op.buffer_args[0]
        src = mod.get_buffer(src_region.parent)
        _check_scope(src, _scope.VRAM, op.kind, "src")
        if len(src_region.extents) != 4:
            raise IsaEmissionError(
                f"{op.kind} src: region must be 4D; got "
                f"extents={tuple(src_region.extents)}"
            )
        # All non-D extents must be 1 (one logical row per op).
        if any(int(e) != 1 for e in src_region.extents[:3]):
            raise IsaEmissionError(
                f"{op.kind} src: row_*_at processes one logical row, "
                f"non-D extents must be 1; got "
                f"{tuple(src_region.extents[:3])}"
            )

        fp_addr_expr = None
        if has_fp:
            fp_addr_expr = self._resolve_fp_scalar_addr_arg(
                mod, op.scalar_args[0], op.kind, "fp",
            )

        src_base_off, src_mask_expr, src_info = self._logical_to_phys_row_offset(
            src, src_region,
        )
        emit_v_mask = masked and src_mask_expr is not None
        use_mask_flag = 1 if emit_v_mask else 0

        mats = []
        m_src = self.materializer.materialize(
            tir.Add(tir.IntImm("int32", int(src.address)), src_base_off)
        )
        self.shim.compiler.generated_code += m_src.isa
        mats.append(m_src)
        gp_src = m_src.register

        gp_mask = None
        try:
            lines = [
                f"; row scalar task {op.annotations.get('intrinsic', op.kind)} "
                f"op={row_op} "
                f"src.parent={src_region.parent} "
                f"starts={list(src_region.starts)!r}"
            ]
            if emit_v_mask:
                m_mask = self.materializer.materialize(src_mask_expr)
                self.shim.compiler.generated_code += m_mask.isa
                mats.append(m_mask)
                gp_mask = m_mask.register
                lines.append(f"C_SET_V_MASK_REG gp{gp_mask}")

            n_d_tiles = src_info["d_tiles"]
            d_tile_stride_s = src_info["d_tile_stride"]

            if reduce:
                # buffer_args=[src_region]; FP destination is scalar_args[0].
                m_dst = self.materializer.materialize(fp_addr_expr)
                self.shim.compiler.generated_code += m_dst.isa
                mats.append(m_dst)
                opcode = {"reduce_max": "V_RED_MAX",
                          "reduce_sum": "V_RED_SUM"}[row_op]
                # V_RED_* accumulate into f1; load the FPRAM slot
                # first so kernels that pre-seeded it see the seed.
                # Across d_tiles, accumulate into the same f1.
                lines.append(f"S_LD_FP f1, gp{m_dst.register}, 0")
                for t in range(n_d_tiles):
                    lines.append(f"{opcode} f1, gp{gp_src}, {use_mask_flag}")
                    if t < n_d_tiles - 1:
                        lines.append(
                            f"S_ADDI_INT gp{gp_src}, gp{gp_src}, "
                            f"{d_tile_stride_s}"
                        )
                lines.append(f"S_ST_FP f1, gp{m_dst.register}, 0")
            else:
                dst_region: _hlir.VramRegion = op.buffer_args[1]
                dst = mod.get_buffer(dst_region.parent)
                _check_scope(dst, _scope.VRAM, op.kind, "dst")
                if len(dst_region.extents) != 4:
                    raise IsaEmissionError(
                        f"{op.kind} dst: region must be 4D; got "
                        f"extents={tuple(dst_region.extents)}"
                    )
                if any(int(e) != 1 for e in dst_region.extents[:3]):
                    raise IsaEmissionError(
                        f"{op.kind} dst: non-D extents must be 1; "
                        f"got {tuple(dst_region.extents[:3])}"
                    )
                dst_base_off, dst_mask_expr, dst_info = (
                    self._logical_to_phys_row_offset(dst, dst_region)
                )
                if emit_v_mask and dst_mask_expr is None:
                    raise IsaEmissionError(
                        f"{op.kind} src requires packed-head mask but dst "
                        f"{dst.name!r} does not"
                    )
                if emit_v_mask and dst_region.parent != src_region.parent:
                    warnings.warn(
                        f"{op.kind}: masked V_*_V with dst "
                        f"{dst_region.parent!r} != src "
                        f"{src_region.parent!r} — unmasked heads will "
                        f"overwrite dst with src; previous cross-lane "
                        f"writes to dst will be lost. Use in-place "
                        f"(dst == src) or insert an explicit copy_v_to_v.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                if dst_info["d_tiles"] != n_d_tiles:
                    raise IsaEmissionError(
                        f"{op.kind}: src/dst d_tiles mismatch "
                        f"({n_d_tiles} vs {dst_info['d_tiles']})"
                    )
                d_tile_stride_d = dst_info["d_tile_stride"]
                m_dst = self.materializer.materialize(
                    tir.Add(tir.IntImm("int32", int(dst.address)), dst_base_off)
                )
                self.shim.compiler.generated_code += m_dst.isa
                mats.append(m_dst)

                if fp_addr_expr is None:
                    # exp / reci
                    opcode = {"exp": "V_EXP_V", "reci": "V_RECI_V"}[row_op]
                    for t in range(n_d_tiles):
                        lines.append(
                            f"{opcode} gp{m_dst.register}, gp{gp_src}, "
                            f"{use_mask_flag}"
                        )
                        if t < n_d_tiles - 1:
                            lines.append(
                                f"S_ADDI_INT gp{gp_src}, gp{gp_src}, "
                                f"{d_tile_stride_s}"
                            )
                            lines.append(
                                f"S_ADDI_INT gp{m_dst.register}, "
                                f"gp{m_dst.register}, {d_tile_stride_d}"
                            )
                else:
                    # add / sub / mul with FP scalar
                    m_rhs = self.materializer.materialize(fp_addr_expr)
                    self.shim.compiler.generated_code += m_rhs.isa
                    mats.append(m_rhs)
                    lines.append(f"S_LD_FP f1, gp{m_rhs.register}, 0")
                    for t in range(n_d_tiles):
                        if row_op == "sub":
                            lines.append(
                                f"V_SUB_VF gp{m_dst.register}, gp{gp_src}, "
                                f"f1, {use_mask_flag}, 0"
                            )
                        else:
                            opcode = {"add": "V_ADD_VF",
                                      "mul": "V_MUL_VF"}[row_op]
                            lines.append(
                                f"{opcode} gp{m_dst.register}, gp{gp_src}, "
                                f"f1, {use_mask_flag}"
                            )
                        if t < n_d_tiles - 1:
                            lines.append(
                                f"S_ADDI_INT gp{gp_src}, gp{gp_src}, "
                                f"{d_tile_stride_s}"
                            )
                            lines.append(
                                f"S_ADDI_INT gp{m_dst.register}, "
                                f"gp{m_dst.register}, {d_tile_stride_d}"
                            )

            if emit_v_mask:
                lines.append(f"S_ADDI_INT gp{gp_mask}, gp0, 0")
                lines.append(f"C_SET_V_MASK_REG gp{gp_mask}")
            self.shim.compiler.generated_code += "\n".join(lines) + "\n"
        finally:
            for m in reversed(mats):
                m.release()

    # ------------------------------------------------------------------
    # Per-op dispatchers. Each one is a thin glue between HLIR buffer
    # references and ISAEmitter's positional/keyword API.
    # ------------------------------------------------------------------
    # ---- DMA decomposition --------------------------------------------
    # Each emit_load/store_tile_from_hbm transfers exactly ONE mlen x mlen
    # tile. For HBM buffers whose logical 2D shape spans multiple tiles
    # we issue one ISA emit call per tile, walking the buffer in
    # col-block-major order (matches --stage-output / runtime helper).
    def _iter_tile_offsets(self, hbm_buf: _hlir.Buffer):
        """Yield (vram_offset_elems, hbm_offset_elems) for each tile of buf."""
        mlen = self.shim.mlen
        ann = hbm_buf.annotations
        rows = ann.get("logical_rows", mlen)
        cols = ann.get("logical_cols", mlen)
        row_blocks = ann.get("row_blocks", 1)
        col_blocks = ann.get("col_blocks", 1)
        tile_elems = mlen * mlen
        idx = 0
        for j in range(col_blocks):
            for i in range(row_blocks):
                # HBM offset (in elements) of the (i,j) tile within this
                # logical 2D buffer. Logical layout is row-major so column
                # j contributes j*mlen, row i contributes i*mlen*cols.
                hbm_off = i * mlen * cols + j * mlen
                vram_off = idx * tile_elems
                yield vram_off, hbm_off
                idx += 1

    def _emit_dma_h2v(self, mod: _hlir.HLIRModule, op: _hlir.Op) -> None:
        src = mod.get_buffer(op.buffer_args[0])
        dst = mod.get_buffer(op.buffer_args[1])
        _check_scope(src, _scope.HBM, op.kind, "src")
        _check_scope(dst, _scope.VRAM, op.kind, "dst")
        for vram_off, hbm_off in self._iter_tile_offsets(src):
            self.shim.compiler.generated_code += (
                f"; dma_h2v tile  {src.name}[hbm+{hbm_off}] -> "
                f"{dst.name}[vram+{vram_off}]\n"
            )
            self.emitter.emit_load_tile_from_hbm(
                hbm_addr=src.address,
                vram_addr=dst.address + vram_off,
                hbm_stride=src.hbm_stride,
                hbm_scale_size=src.hbm_scale_size,
                hbm_start_offset=src.hbm_offset + hbm_off,
            )

    def _emit_dma_h2m(self, mod: _hlir.HLIRModule, op: _hlir.Op) -> None:
        src = mod.get_buffer(op.buffer_args[0])
        dst = mod.get_buffer(op.buffer_args[1])
        _check_scope(src, _scope.HBM, op.kind, "src")
        _check_scope(dst, _scope.MRAM, op.kind, "dst")
        for vram_off, hbm_off in self._iter_tile_offsets(src):
            self.shim.compiler.generated_code += (
                f"; dma_h2m tile  {src.name}[hbm+{hbm_off}] -> "
                f"{dst.name}[mram+{vram_off}]\n"
            )
            self.emitter.emit_hbm_tile_to_mram(
                hbm_addr=src.address,
                mram_addr=dst.address + vram_off,
                hbm_offset=src.hbm_offset + hbm_off,
                hbm_scale=src.hbm_scale_size,
                hbm_stride=src.hbm_stride,
            )

    def _emit_dma_v2h(self, mod: _hlir.HLIRModule, op: _hlir.Op) -> None:
        # Convention: HBM buffers are BSHD; VRAM buffers may be BHSD if
        # they came out of BTMM/BMM_WO (head-major physical layout). This
        # iteration walks the HBM (dst) in col-block-major order, which
        # happens to land vram_off = idx * tile_elems on each head's tile
        # boundary -- exactly matching BMM_WO's BHSD VRAM packing. The
        # store thus reorders BHSD -> BSHD as a side-effect of the walk.
        src = mod.get_buffer(op.buffer_args[0])
        dst = mod.get_buffer(op.buffer_args[1])
        _check_scope(src, _scope.VRAM, op.kind, "src")
        _check_scope(dst, _scope.HBM, op.kind, "dst")
        if src.num_elements != dst.num_elements:
            raise IsaEmissionError(
                f"dma_v2h: src ({src.name}, {src.num_elements} elems) and dst "
                f"({dst.name}, {dst.num_elements} elems) must have the same total "
                f"size, but their layouts may differ (VRAM=BHSD, HBM=BSHD)."
            )
        for vram_off, hbm_off in self._iter_tile_offsets(dst):
            self.shim.compiler.generated_code += (
                f"; dma_v2h tile  {src.name}[vram+{vram_off}] -> "
                f"{dst.name}[hbm+{hbm_off}]\n"
            )
            self.emitter.emit_store_tile_to_hbm(
                vram_addr=src.address + vram_off,
                hbm_addr=dst.address,
                hbm_stride=dst.hbm_stride,
                hbm_scale_size=dst.hbm_scale_size,
                hbm_start_offset=dst.hbm_offset + hbm_off,
            )

    # ------------------------------------------------------------------
    # Sliced DMA dispatchers. The slice is one of buffer_args (src or
    # dst depending on direction). For now we restrict to STATIC starts
    # (all Python ints / IntImm); dynamic starts (PrimExpr) raise a
    # clear error pointing at the next phase.
    # ------------------------------------------------------------------
    def _slice_offset_static(
        self, parent: _hlir.Buffer, sl: _hlir.BufferSlice,
    ) -> int:
        """Same math as `_build_slice_offset_expr`, restricted to all-int
        starts. Used in the static fast-path (avoids the extra
        S_ADDI_INT...mov that the dynamic path inserts)."""
        offset = 0
        shape = parent.shape
        for i, s in enumerate(sl.starts):
            stride_below = 1
            for d in shape[i + 1:]:
                stride_below *= int(d)
            offset += int(s) * stride_below
        return offset

    @staticmethod
    def _slice_has_dynamic_start(sl: _hlir.BufferSlice) -> bool:
        return any(not isinstance(s, int) for s in sl.starts)

    def _build_slice_offset_expr(
        self, parent: _hlir.Buffer, sl: _hlir.BufferSlice,
    ):
        """Build a PrimExpr for the slice's element offset in `parent`'s
        HBM region. Mixes static (Python int / IntImm) and dynamic
        (PrimExpr) starts uniformly. ExprMaterializer's constant
        folding will collapse static sub-trees automatically.
        """
        offset = tir.IntImm("int32", 0)
        shape = parent.shape
        for i, s in enumerate(sl.starts):
            stride_below = 1
            for d in shape[i + 1:]:
                stride_below *= int(d)
            if isinstance(s, int):
                term = tir.IntImm("int32", s * stride_below)
            else:
                # `s` is a PrimExpr (loop var or compound); multiply by
                # stride at the IR level so the materialiser can apply
                # strength reduction (e.g. SLLI when stride is 2^k).
                term = s * tir.IntImm("int32", stride_below)
            offset = offset + term
        return offset

    def _check_slice_single_tile(
        self, parent: _hlir.Buffer, sl: _hlir.BufferSlice,
    ) -> None:
        """For input slices (h2v / h2m): must fit in exactly one mlen*mlen
        tile after H*D logical-2D collapse, since the destination
        VRAM/MRAM buffer is a single-tile staging area.
        """
        mlen = self.shim.mlen
        ext = sl.extents
        if len(ext) != len(parent.shape):
            raise IsaEmissionError(
                f"slice on {parent.name!r}: extents length {len(ext)} != "
                f"parent ndim {len(parent.shape)}"
            )
        rows, cols = _hlir.logical_2d_extents(ext, parent.layout)
        if rows != mlen or cols != mlen:
            raise IsaEmissionError(
                f"slice on {parent.name!r} extents={ext} (layout={parent.layout!r}) "
                f"maps to logical 2D ({rows}, {cols}); h2v/h2m input slices "
                f"must fit a single mlen*mlen tile."
            )

    def _iter_slice_tiles_per_head(self, parent: _hlir.Buffer, sl: _hlir.BufferSlice):
        """Per-head multi-tile iterator for v2h-slice writeback.

        Used when the slice covers `eh` heads but a single mlen-aligned
        block in seq and dim dims. Each tile is one head's contribution.
        Yields tuples `(h_idx, vram_off_in_src, tile_const_in_parent_elems)`:
            * `h_idx`            -- which head within the slice (0..eh-1)
            * `vram_off_in_src`  -- offset in elements from the VRAM source
                                    buffer's base. Assumes BHSD physical
                                    layout (BMM_WO output convention),
                                    where head h's tile sits at
                                    `h * tile_elems`.
            * `tile_const_in_parent_elems` -- additive offset to combine
                                    with the slice's BASE offset to land
                                    at this tile's element 0 in the parent.

        Constraints (matches the v2h_slice dispatcher's expectations):
            * parent is 4D BSHD
            * eb == 1 (single batch)
            * es == mlen (single seq tile per head)
            * ed == mlen (single dim tile per head)
            * eh >= 1 (any number of heads; eh > 1 is the multi-tile case)
        """
        mlen = self.shim.mlen
        if len(parent.shape) != 4:
            raise IsaEmissionError(
                f"per-head slice tiling requires 4D parent; got "
                f"shape {parent.shape}"
            )
        # Permute parent shape and slice extents into canonical (B, S, H, D)
        # order per parent.layout. Downstream math works in BSHD.
        B, S, H, D = _hlir._select_axes(parent.shape, parent.layout)
        eb, es, eh, ed = _hlir._select_axes(sl.extents, parent.layout)
        if eb != 1:
            raise IsaEmissionError(
                f"per-head slice tiling does not support batch slicing "
                f"(eb={eb})"
            )
        if es != mlen:
            raise IsaEmissionError(
                f"per-head slice tiling requires es == mlen ({mlen}); "
                f"got es={es}"
            )
        if ed != mlen:
            raise IsaEmissionError(
                f"per-head slice tiling requires ed == mlen ({mlen}); "
                f"got ed={ed}"
            )
        if eh < 1:
            raise IsaEmissionError(f"slice has no heads to iterate (eh={eh})")
        # Per-head HBM offset is h_idx * h_stride, where h_stride is
        # the canonical channel-axis stride in HBM-row-major order.
        # For BSHD (head outer of D): h_stride = D — what the legacy
        # ``h_idx * D`` formula assumed. For NCHW (channel outer of
        # H*W): h_stride = H*W — different by a row-count factor.
        # ``hbm_strides_for_layout`` returns the right number for any
        # layout we register.
        _hb, _hs, h_stride, _hd = _hlir.hbm_strides_for_layout(
            parent.shape, parent.layout,
        )
        tile_elems = mlen * mlen
        for h_idx in range(eh):
            yield h_idx, h_idx * tile_elems, h_idx * int(h_stride)

    def _slice_is_single_logical_tile(
        self, parent: _hlir.Buffer, sl: _hlir.BufferSlice,
    ) -> bool:
        ext = sl.extents
        if len(ext) != len(parent.shape):
            return False
        rows, cols = _hlir.logical_2d_extents(ext, parent.layout)
        return rows == self.shim.mlen and cols == self.shim.mlen

    def _materialise_slice_offset(
        self, parent: _hlir.Buffer, sl: _hlir.BufferSlice,
    ):
        """Returns either:
            (None, int_offset)  -- all starts static; caller uses
                                    existing int-offset emit path.
            (MaterializedExpr, None) -- dynamic; caller uses *_reg emit
                                    path and must release the result.
        """
        if not self._slice_has_dynamic_start(sl):
            return None, parent.hbm_offset + self._slice_offset_static(parent, sl)
        # Dynamic path: build expr (includes parent.hbm_offset for safety)
        # and lower via ExprMaterializer.
        expr = self._build_slice_offset_expr(parent, sl)
        if parent.hbm_offset:
            expr = expr + tir.IntImm("int32", parent.hbm_offset)
        m = self.materializer.materialize(expr)
        self.shim.compiler.generated_code += m.isa
        return m, None

    @staticmethod
    def _format_starts(sl: _hlir.BufferSlice) -> str:
        return ",".join(
            str(s) if isinstance(s, int) else f"<{type(s).__name__}>"
            for s in sl.starts
        )

    def _slice_tile_grid(
        self,
        parent: _hlir.Buffer,
        sl: _hlir.BufferSlice,
        on_chip: _hlir.Buffer,
    ):
        """Compute the inner-tile grid for one HBM↔on-chip slice copy.

        Symmetric between h2v load and v2h store: the iteration grid
        only depends on the on-chip buffer's physical layout and the
        slice's footprint, not on direction. The returned strides feed
        either ``emit_load_tile_from_hbm`` or ``emit_store_tile_to_hbm``.

        Returns a tuple ``(d_tiles, s_tiles, h_groups, logical_b,
        inner_mlen, lane_count, hbm_strides, vram_strides)`` where
        ``hbm_strides`` is ``(b, s, h)`` and ``vram_strides`` is
        ``(d_tile, s_tile, h_grp, b)`` — all in elements.

        Single tile (one mlen×mlen tile, or smaller) is the degenerate
        case where every tile count is 1 — the caller runs one loop
        iteration and emits a single H_LOAD_V / H_STORE_V.
        """
        dst = on_chip
        mlen = self.shim.mlen

        if dst.tile_layout is not None:
            # 4D BSHD path. Parent must be 4D — we read its layout-
            # aware per-axis HBM strides (b/s/h/d).
            if len(parent.shape) != 4:
                raise IsaEmissionError(
                    f"dma_h2v_slice: HBM parent {parent.name!r} must "
                    f"be 4D for tile_layout-driven dst; got shape "
                    f"{tuple(parent.shape)}"
                )
            hbm_stride_b, hbm_stride_s, hbm_stride_h, _hbm_stride_d = (
                _hlir.hbm_strides_for_layout(parent.shape, parent.layout)
            )
            tl = dst.tile_layout
            inner_d = tl.d_inner
            inner_lane = tl.lane_count * inner_d
            inner_s = tl.mlen * inner_lane
            b_stride = inner_s
            inner_b = tl.logical_b * inner_s
            h_grp_stride = inner_b
            s_tile_stride = tl.h_groups * inner_b
            d_tile_stride = tl.s_tiles * s_tile_stride
            return (
                tl.d_tiles, tl.s_tiles, tl.h_groups, tl.logical_b,
                tl.mlen, tl.lane_count,
                (hbm_stride_b, hbm_stride_s, hbm_stride_h),
                (d_tile_stride, s_tile_stride, h_grp_stride, b_stride),
            )

        # No tile_layout: row-major flat view. dst dictates the tile
        # grid; HBM strides come from parent's row-major flat layout
        # treating axes[-2] as row and axes[-1] as col regardless of
        # layout-name semantics. This is the right model for any
        # plain ``T.alloc_shared((rows, cols))`` staging buffer.
        if len(dst.shape) == 2:
            dst_rows = int(dst.shape[0])
            dst_cols = int(dst.shape[1])
        elif len(dst.shape) == 1:
            dst_rows = 1
            dst_cols = int(dst.shape[0])
        else:
            raise IsaEmissionError(
                f"dma_h2v_slice on {parent.name!r}: dst {dst.name!r} "
                f"has rank {len(dst.shape)} with no tile_layout; "
                f"only 1D / 2D dst supported on the row-major-flat path"
            )
        if dst_rows % mlen != 0 or dst_cols % mlen != 0:
            raise IsaEmissionError(
                f"dma_h2v_slice on {parent.name!r}: dst {dst.name!r} "
                f"shape ({dst_rows}, {dst_cols}) is not (MLEN={mlen})-"
                f"aligned on both axes; partial-tile loads not supported"
            )
        s_tiles = dst_rows // mlen
        d_tiles = dst_cols // mlen
        d_tile_stride = mlen
        s_tile_stride = mlen * dst_cols

        # Parent HBM row stride = product of axes after the row axis.
        # For 4D ``(N, C, H, W)`` and a per-channel slice, row =
        # axes[-2] (H), col = axes[-1] (W); per-row HBM stride = W.
        # We map this to the existing (b/s/h) stride triple by routing
        # the row stride through ``hbm_stride_s`` (s_tile multiplier in
        # the emitter); h-stride / b-stride aren't iterated in this
        # path (h_groups == logical_b == 1).
        pshape = [int(x) for x in parent.shape]
        if len(pshape) < 2:
            # 1D HBM parent — degenerate row stride = 1.
            hbm_row_stride = 1
        else:
            hbm_row_stride = 1
            for d in pshape[-1:]:
                hbm_row_stride = int(d)
        return (
            d_tiles, s_tiles, 1, 1,
            mlen, 1,
            (0, hbm_row_stride, 0),
            (d_tile_stride, s_tile_stride, 0, 0),
        )

    def _emit_dma_h2v_slice(self, mod: _hlir.HLIRModule, op: _hlir.Op) -> None:
        """Emit H_LOAD_V instructions for one HBM→VRAM slice copy.

        Single path for all dst shapes: compute an inner-tile grid via
        :meth:`_h2v_tile_grid` and walk it. The grid is ``1×1×1×1`` for
        a slice that fits one mlen×mlen tile; larger dst (2D row-major
        or 4D BSHD with ``tile_layout``) expand into multiple issues.
        """
        sl = op.buffer_args[0]
        _arg1 = op.buffer_args[1]
        if isinstance(_arg1, _hlir.BufferSlice):
            raise IsaEmissionError(
                f"dma_h2v_slice: dst (buffer_args[1]) must be a whole-buffer "
                f"name; got BufferSlice(parent={_arg1.parent!r}, "
                f"starts={list(_arg1.starts)}, extents={list(_arg1.extents)})"
            )
        dst = mod.get_buffer(_arg1)
        if not isinstance(sl, _hlir.BufferSlice):
            raise IsaEmissionError(
                f"dma_h2v_slice: buffer_args[0] must be BufferSlice, got "
                f"{type(sl).__name__}"
            )
        parent = mod.get_buffer(sl.parent)
        _check_scope(parent, _scope.HBM, op.kind, "src.parent")
        _check_scope(dst, _scope.VRAM, op.kind, "dst")

        (d_tiles, s_tiles, h_groups, logical_b,
         inner_mlen, lane_count,
         (hbm_stride_b, hbm_stride_s, hbm_stride_h),
         (d_tile_stride, s_tile_stride, h_grp_stride, b_stride)) = (
            self._slice_tile_grid(parent, sl, dst)
        )

        m_off, slice_static = self._materialise_slice_offset(parent, sl)
        base_static = parent.hbm_offset + (
            slice_static if slice_static is not None else 0
        )

        starts_s = self._format_starts(sl)
        self.shim.compiler.generated_code += (
            f"; dma_h2v_slice  {parent.name}[{starts_s}]+{list(sl.extents)} "
            f"-> {dst.name}  "
            f"(grid d_tiles={d_tiles}, s_tiles={s_tiles}, "
            f"h_groups={h_groups}, b={logical_b}"
            f"{', dyn base gp' + str(m_off.register) if m_off is not None else ''})\n"
        )
        for d_tile in range(d_tiles):
            for s_tile in range(s_tiles):
                for h_grp in range(h_groups):
                    for b in range(logical_b):
                        hbm_off = (
                            base_static
                            + b * hbm_stride_b
                            + s_tile * inner_mlen * hbm_stride_s
                            + h_grp * lane_count * hbm_stride_h
                            + d_tile * inner_mlen
                        )
                        vram_off = (
                            d_tile * d_tile_stride
                            + s_tile * s_tile_stride
                            + h_grp * h_grp_stride
                            + b * b_stride
                        )
                        self.shim.compiler.generated_code += (
                            f";   tile (d={d_tile}, s={s_tile}, h={h_grp}, "
                            f"b={b}): hbm_off={hbm_off}  vram_off={vram_off}\n"
                        )
                        if m_off is not None:
                            self.emitter.emit_load_tile_from_hbm(
                                hbm_addr=parent.address,
                                vram_addr=dst.address + vram_off,
                                hbm_stride=parent.hbm_stride,
                                hbm_scale_size=parent.hbm_scale_size,
                                hbm_start_offset=hbm_off,
                                hbm_start_offset_reg=m_off.register,
                            )
                        else:
                            self.emitter.emit_load_tile_from_hbm(
                                hbm_addr=parent.address,
                                vram_addr=dst.address + vram_off,
                                hbm_stride=parent.hbm_stride,
                                hbm_scale_size=parent.hbm_scale_size,
                                hbm_start_offset=hbm_off,
                            )
        if m_off is not None:
            m_off.release()

    def _emit_dma_h2m_slice(self, mod: _hlir.HLIRModule, op: _hlir.Op) -> None:
        sl = op.buffer_args[0]
        dst = mod.get_buffer(op.buffer_args[1])
        if not isinstance(sl, _hlir.BufferSlice):
            raise IsaEmissionError(
                f"dma_h2m_slice: buffer_args[0] must be BufferSlice"
            )
        parent = mod.get_buffer(sl.parent)
        _check_scope(parent, _scope.HBM, op.kind, "src.parent")
        _check_scope(dst, _scope.MRAM, op.kind, "dst")
        self._check_slice_single_tile(parent, sl)

        m_off, static_off = self._materialise_slice_offset(parent, sl)
        starts_s = self._format_starts(sl)
        if m_off is None:
            self.shim.compiler.generated_code += (
                f"; dma_h2m_slice  {parent.name}[{starts_s}]+{list(sl.extents)} "
                f"-> {dst.name}  (parent_off={static_off} elems)\n"
            )
            self.emitter.emit_hbm_tile_to_mram(
                hbm_addr=parent.address, mram_addr=dst.address,
                hbm_offset=static_off,
                hbm_scale=parent.hbm_scale_size, hbm_stride=parent.hbm_stride,
            )
        else:
            self.shim.compiler.generated_code += (
                f"; dma_h2m_slice  {parent.name}[{starts_s}]+{list(sl.extents)} "
                f"-> {dst.name}  (parent_off=gp{m_off.register} dyn)\n"
            )
            self.emitter.emit_hbm_tile_to_mram(
                hbm_addr=parent.address, mram_addr=dst.address,
                hbm_offset_reg=m_off.register,
                hbm_scale=parent.hbm_scale_size, hbm_stride=parent.hbm_stride,
            )
            m_off.release()

    def _emit_dma_v2h_slice(self, mod: _hlir.HLIRModule, op: _hlir.Op) -> None:
        """Emit H_STORE_V instructions for one VRAM→HBM slice copy.

        Mirror of :meth:`_emit_dma_h2v_slice`: same tile-grid model
        from :meth:`_slice_tile_grid`, same per-tile offset math, just
        store-direction emit. Single tile = ``1×1×1×1`` grid → one
        H_STORE_V. Multi-tile (4D with tile_layout, or 2D row-major
        larger than one mlen tile) expands naturally.
        """
        src = mod.get_buffer(op.buffer_args[0])
        sl = op.buffer_args[1]
        if not isinstance(sl, _hlir.BufferSlice):
            raise IsaEmissionError(
                f"dma_v2h_slice: buffer_args[1] must be BufferSlice"
            )
        parent = mod.get_buffer(sl.parent)
        _check_scope(src, _scope.VRAM, op.kind, "src")
        _check_scope(parent, _scope.HBM, op.kind, "dst.parent")

        ra = self.shim.compiler.register_allocator

        (d_tiles, s_tiles, h_groups, logical_b,
         inner_mlen, lane_count,
         (hbm_stride_b, hbm_stride_s, hbm_stride_h),
         (d_tile_stride, s_tile_stride, h_grp_stride, b_stride)) = (
            self._slice_tile_grid(parent, sl, src)
        )

        m_base, static_base = self._materialise_slice_offset(parent, sl)
        is_dyn = m_base is not None
        base_static = static_base if static_base is not None else 0

        starts_s = self._format_starts(sl)
        self.shim.compiler.generated_code += (
            f"; dma_v2h_slice  {src.name} -> "
            f"{parent.name}[{starts_s}]+{list(sl.extents)}  "
            f"(grid d_tiles={d_tiles}, s_tiles={s_tiles}, "
            f"h_groups={h_groups}, b={logical_b}"
            f"{', dyn base gp' + str(m_base.register) if is_dyn else ''})\n"
        )
        for d_tile in range(d_tiles):
            for s_tile in range(s_tiles):
                for h_grp in range(h_groups):
                    for b in range(logical_b):
                        tile_const = (
                            b * hbm_stride_b
                            + s_tile * inner_mlen * hbm_stride_s
                            + h_grp * lane_count * hbm_stride_h
                            + d_tile * inner_mlen
                        )
                        vram_off = (
                            d_tile * d_tile_stride
                            + s_tile * s_tile_stride
                            + h_grp * h_grp_stride
                            + b * b_stride
                        )
                        tile_vram = src.address + vram_off
                        self.shim.compiler.generated_code += (
                            f";   tile (d={d_tile}, s={s_tile}, h={h_grp}, "
                            f"b={b}): vram[+{vram_off}] -> "
                            f"hbm[base+{tile_const}]\n"
                        )
                        if is_dyn:
                            if tile_const == 0:
                                tile_off_reg = m_base.register
                                tile_off_owned = False
                            else:
                                tile_off_reg = ra.allocate_gp(1)[0]
                                tile_off_owned = True
                                self.shim.compiler.generated_code += (
                                    f"S_ADDI_INT gp{tile_off_reg}, "
                                    f"gp{m_base.register}, {tile_const}\n"
                                )
                            self.emitter.emit_store_tile_to_hbm(
                                vram_addr=tile_vram, hbm_addr=parent.address,
                                hbm_stride=parent.hbm_stride,
                                hbm_scale_size=parent.hbm_scale_size,
                                hbm_start_offset_reg=tile_off_reg,
                            )
                            if tile_off_owned:
                                ra.free_gp([tile_off_reg])
                        else:
                            self.emitter.emit_store_tile_to_hbm(
                                vram_addr=tile_vram, hbm_addr=parent.address,
                                hbm_stride=parent.hbm_stride,
                                hbm_scale_size=parent.hbm_scale_size,
                                hbm_start_offset=base_static + tile_const,
                            )
        if is_dyn:
            m_base.release()

        if is_dyn:
            m_base.release()

    def _emit_btmm(self, mod: _hlir.HLIRModule, op: _hlir.Op) -> None:
        """Lane-fused (packed-head) Q @ K^T.

        Region schema:
            buffer_args = [a_region (VramRegion), b_region (MramRegion),
                           c_region (VramRegion)]
            scalar_args = [a_dim_roles, b_dim_roles, c_dim_roles]

        BTMM HW takes the whole packed-head tile in one issue; the
        regions describe the full operands and the emitter doesn't
        need to walk M/K/N internally — just hand off the three
        physical base addresses. Per-lane offsets are zero here
        (multi-lane HW instruction spans every lane natively).
        """
        if len(op.buffer_args) != 3:
            raise IsaEmissionError(
                f"plena.btmm expects 3 buffer_args (regions); "
                f"got {len(op.buffer_args)}"
            )
        a_reg, b_reg, c_reg = op.buffer_args
        if not isinstance(a_reg, _hlir.VramRegion):
            raise IsaEmissionError(
                f"plena.btmm a: expected VramRegion, got "
                f"{type(a_reg).__name__}"
            )
        if not isinstance(b_reg, _hlir.MramRegion):
            raise IsaEmissionError(
                f"plena.btmm b: expected MramRegion, got "
                f"{type(b_reg).__name__}"
            )
        if not isinstance(c_reg, _hlir.VramRegion):
            raise IsaEmissionError(
                f"plena.btmm c: expected VramRegion, got "
                f"{type(c_reg).__name__}"
            )
        lhs = mod.get_buffer(a_reg.parent)
        rhs = mod.get_buffer(b_reg.parent)
        dst = mod.get_buffer(c_reg.parent)
        _check_scope(lhs, _scope.VRAM, op.kind, "lhs")
        _check_scope(rhs, _scope.MRAM, op.kind, "rhs")
        _check_scope(dst, _scope.VRAM, op.kind, "dst")

        # Result-tile-count for the writeback. For our minimal_btmm:
        # lhs (mlen, gh, hlen) x rhs (mlen, gh, hlen) -> dst (mlen, gh, mlen)
        # so dst has gh tiles per group, total tile_count =
        # (mlen*gh*mlen)/tile_elems.
        tile_count = max(1, dst.num_elements // self.shim.tile_elems)

        self.emitter.emit_btmm(
            lhs_packed_vram_addr=lhs.address,
            rhs_mram_addr=rhs.address,
            task_id=op.annotations.get("intrinsic", "btmm"),
        )
        self.emitter.emit_btmm_wo(
            base_addr=dst.address,
            tile_count=tile_count,
            task_id=op.annotations.get("intrinsic", "btmm") + ".wo",
        )

    def _emit_mv(self, mod: _hlir.HLIRModule, op: _hlir.Op) -> None:
        """Per-head matrix-vector: M_MV + M_MV_WO.

        Region schema (same shape as matmul):
            buffer_args = [a_region, b_region, c_region]
                a_region: VramRegion (rank-1 LHS row, M extent == 1)
                b_region: MramRegion
                c_region: VramRegion
            scalar_args = [a_dim_roles, b_dim_roles, c_dim_roles]
                4-tuples of "M"/"K"/"N"/"_".

        Origin offsets come from the regions' starts (lane_var on the
        cluster axis when wrapped in CLUSTER), translated to physical
        offsets via ``_tile_layout_strides``.
        """
        if len(op.buffer_args) != 3:
            raise IsaEmissionError(
                f"plena.mv expects 3 buffer_args (a/b/c regions); "
                f"got {len(op.buffer_args)}"
            )
        a_reg, b_reg, c_reg = op.buffer_args
        if not isinstance(a_reg, _hlir.VramRegion):
            raise IsaEmissionError(
                f"plena.mv a: expected VramRegion, got "
                f"{type(a_reg).__name__}"
            )
        if not isinstance(b_reg, _hlir.MramRegion):
            raise IsaEmissionError(
                f"plena.mv b: expected MramRegion, got "
                f"{type(b_reg).__name__}"
            )
        if not isinstance(c_reg, _hlir.VramRegion):
            raise IsaEmissionError(
                f"plena.mv c: expected VramRegion, got "
                f"{type(c_reg).__name__}"
            )
        lhs = mod.get_buffer(a_reg.parent)
        rhs = mod.get_buffer(b_reg.parent)
        dst = mod.get_buffer(c_reg.parent)
        _check_scope(lhs, _scope.VRAM, op.kind, "lhs")
        _check_scope(rhs, _scope.MRAM, op.kind, "rhs")
        _check_scope(dst, _scope.VRAM, op.kind, "dst")

        lhs_raw_off = self._region_origin_offset(lhs, a_reg)
        rhs_raw_off = self._region_origin_offset(rhs, b_reg)
        dst_raw_off = self._region_origin_offset(dst, c_reg)

        def _resolve(expr, name):
            """Returns (static_int_or_None, gp_register_or_None, handle_or_None)."""
            if isinstance(expr, tir.IntImm):
                return int(expr.value), None, None
            if isinstance(expr, int):
                return int(expr), None, None
            m = self.materializer.materialize(expr)
            self.shim.compiler.generated_code += m.isa
            return None, m.register, m

        lhs_static, lhs_reg, lhs_h = _resolve(lhs_raw_off, "lhs_offset")
        rhs_static, rhs_reg, rhs_h = _resolve(rhs_raw_off, "rhs_offset")
        dst_static, dst_reg, dst_h = _resolve(dst_raw_off, "dst_offset")

        try:
            self.emitter.emit_mv(
                lhs_vram_addr=lhs.address + (lhs_static or 0),
                rhs_mram_addr=rhs.address + (rhs_static or 0),
                dst_vram_addr=dst.address + (dst_static or 0),
                lhs_offset_reg=lhs_reg,
                rhs_offset_reg=rhs_reg,
                dst_offset_reg=dst_reg,
                task_id=op.annotations.get("intrinsic", "mv"),
            )
        finally:
            for h in (lhs_h, rhs_h, dst_h):
                if h is not None:
                    h.release()

    def _emit_btmv(self, mod: _hlir.HLIRModule, op: _hlir.Op) -> None:
        """Lane-fused matrix-vector (decode-style btmm with M=1).

        Region schema same as _emit_btmm; differs only in op kind.
        """
        if len(op.buffer_args) != 3:
            raise IsaEmissionError(
                f"plena.btmv expects 3 buffer_args (regions); "
                f"got {len(op.buffer_args)}"
            )
        a_reg, b_reg, c_reg = op.buffer_args
        if not isinstance(a_reg, _hlir.VramRegion):
            raise IsaEmissionError(
                f"plena.btmv a: expected VramRegion, got "
                f"{type(a_reg).__name__}"
            )
        if not isinstance(b_reg, _hlir.MramRegion):
            raise IsaEmissionError(
                f"plena.btmv b: expected MramRegion, got "
                f"{type(b_reg).__name__}"
            )
        if not isinstance(c_reg, _hlir.VramRegion):
            raise IsaEmissionError(
                f"plena.btmv c: expected VramRegion, got "
                f"{type(c_reg).__name__}"
            )
        lhs = mod.get_buffer(a_reg.parent)
        rhs = mod.get_buffer(b_reg.parent)
        dst = mod.get_buffer(c_reg.parent)
        _check_scope(lhs, _scope.VRAM, op.kind, "lhs")
        _check_scope(rhs, _scope.MRAM, op.kind, "rhs")
        _check_scope(dst, _scope.VRAM, op.kind, "dst")

        self.emitter.emit_btmv(
            lhs_packed_vram_addr=lhs.address,
            rhs_mram_addr=rhs.address,
            task_id=op.annotations.get("intrinsic", "btmv"),
        )
        self.emitter.emit_bmv_wo(
            base_addr=dst.address,
            task_id=op.annotations.get("intrinsic", "btmv") + ".wo",
        )

    def _emit_mm(self, mod: _hlir.HLIRModule, op: _hlir.Op) -> None:
        """Single-tile, single-head matrix multiply.

        Maps `plena.mm(lhs_vram, rhs_mram, dst_vram)` to one M_MM /
        M_MM_WO sequence (via ISAEmitter.emit_matmul with a single
        lhs/rhs pair). The dst tile is fully overwritten — no implicit
        accumulation across calls. Streaming-style accumulation is the
        kernel author's job (tile_zero + mm + tile_add into a separate
        accumulator tile, see kernels/tiled_mm.py).
        """
        lhs = mod.get_buffer(op.buffer_args[0])
        rhs = mod.get_buffer(op.buffer_args[1])
        dst = mod.get_buffer(op.buffer_args[2])
        _check_scope(lhs, _scope.VRAM, op.kind, "lhs")
        _check_scope(rhs, _scope.MRAM, op.kind, "rhs")
        _check_scope(dst, _scope.VRAM, op.kind, "dst")
        lhs_rows, lhs_cols = self._logical_2d(lhs.shape)
        rhs_rows, rhs_cols = self._logical_2d(rhs.shape)
        dst_rows, dst_cols = self._logical_2d(dst.shape)
        if lhs_rows != self.shim.mlen or lhs_cols != self.shim.mlen:
            raise IsaEmissionError(
                f"plena.mm lhs must be one full mlen*mlen tile; got logical 2D "
                f"({lhs_rows}, {lhs_cols}) for buffer {lhs.name}"
            )
        if rhs_rows != self.shim.mlen:
            raise IsaEmissionError(
                f"plena.mm rhs must have mlen rows; got logical 2D "
                f"({rhs_rows}, {rhs_cols}) for buffer {rhs.name}"
            )
        if dst_rows != self.shim.mlen:
            raise IsaEmissionError(
                f"plena.mm dst must have mlen rows; got logical 2D "
                f"({dst_rows}, {dst_cols}) for buffer {dst.name}"
            )
        if rhs_cols != dst_cols:
            raise IsaEmissionError(
                f"plena.mm rhs/dst logical widths must match; got rhs={rhs_cols} dst={dst_cols}"
            )
        # Use the hw-loop emitter (tens of static lines) instead of the
        # Python-unrolled emit_matmul (~2k lines per call). Dynamic
        # instruction count is identical; hw loops just shrink the ISA
        # text. Important for kernels that invoke plena.mm under several
        # unrolled outer levels (q*h*d*kv) where ASM size scales with
        # the product.
        if rhs_cols == self.shim.mlen and dst_cols == self.shim.mlen:
            self.emitter.emit_matmul_single_tile_hwloop(
                lhs_vram_addr=lhs.address,
                rhs_mram_addr=rhs.address,
                dst_vram_addr=dst.address,
                task_id=op.annotations.get("intrinsic", "mm"),
            )
            return
        self.emitter.emit_matmul_narrow_tile_hwloop(
            lhs_vram_addr=lhs.address,
            rhs_mram_addr=rhs.address,
            dst_vram_addr=dst.address,
            hlen=rhs_cols,
            dst_row_stride=dst_cols,
            task_id=op.annotations.get("intrinsic", "mm"),
        )

    def _emit_matmul(self, mod: _hlir.HLIRModule, op: _hlir.Op) -> None:
        """Unified `(M, K) @ (K, N) -> (M, N)` matmul.

        Region schema (Einstein-style):
            buffer_args = [a_region, b_region, c_region]
                Vram/MramRegion (4D start+extent on the parent buffer's
                physical shape). a_region is VRAM, b_region is MRAM,
                c_region is VRAM. starts encode per-lane / per-batch
                origin; extents are the per-axis logical span.
            scalar_args = [a_dim_roles, b_dim_roles, c_dim_roles]
                each a 4-tuple of "M"/"K"/"N"/"_" labels aligned with
                the matching region. K appears in a and b but not in
                c (contracted); M in a and c; N in b and c. The
                ordering of K vs N inside b's roles tells the emitter
                whether B is K-inner (standard, M_MM) or K-outer
                (transpose_B, M_TMM).
        """
        if len(op.buffer_args) != 3:
            raise IsaEmissionError(
                f"plena.matmul expects 3 buffer_args (a/b/c regions); "
                f"got {len(op.buffer_args)}"
            )
        a_reg, b_reg, c_reg = op.buffer_args
        if not isinstance(a_reg, _hlir.VramRegion):
            raise IsaEmissionError(
                f"plena.matmul a: expected VramRegion, got "
                f"{type(a_reg).__name__}"
            )
        if not isinstance(b_reg, _hlir.MramRegion):
            raise IsaEmissionError(
                f"plena.matmul b: expected MramRegion, got "
                f"{type(b_reg).__name__}"
            )
        if not isinstance(c_reg, _hlir.VramRegion):
            raise IsaEmissionError(
                f"plena.matmul c: expected VramRegion, got "
                f"{type(c_reg).__name__}"
            )
        if len(op.scalar_args) != 3:
            raise IsaEmissionError(
                f"plena.matmul expects 3 scalar_args (a/b/c dim_roles); "
                f"got {len(op.scalar_args)}"
            )
        a_roles, b_roles, c_roles = op.scalar_args
        if len(a_roles) != 4 or len(b_roles) != 4 or len(c_roles) != 4:
            raise IsaEmissionError(
                f"plena.matmul dim_roles must each be 4-tuples; got "
                f"a={a_roles!r} b={b_roles!r} c={c_roles!r}"
            )

        lhs = mod.get_buffer(a_reg.parent)
        rhs = mod.get_buffer(b_reg.parent)
        dst = mod.get_buffer(c_reg.parent)
        _check_scope(lhs, _scope.VRAM, op.kind, "lhs")
        _check_scope(rhs, _scope.MRAM, op.kind, "rhs")
        _check_scope(dst, _scope.VRAM, op.kind, "dst")

        mlen = int(self.shim.mlen)

        def _find_role_axis(roles: Tuple[str, ...], role: str,
                            operand: str) -> Optional[int]:
            hits = [i for i, r in enumerate(roles) if r == role]
            if not hits:
                return None
            if len(hits) > 1:
                raise IsaEmissionError(
                    f"plena.matmul {operand}: role {role!r} appears at "
                    f"multiple axes {hits} in roles {roles!r}"
                )
            return hits[0]

        c_M_axis = _find_role_axis(c_roles, "M", "c")
        c_N_axis = _find_role_axis(c_roles, "N", "c")
        a_M_axis = _find_role_axis(a_roles, "M", "a")
        a_K_axis = _find_role_axis(a_roles, "K", "a")
        b_K_axis = _find_role_axis(b_roles, "K", "b")
        b_N_axis = _find_role_axis(b_roles, "N", "b")
        for axis, name in (
            (c_M_axis, "c.M"), (c_N_axis, "c.N"),
            (a_M_axis, "a.M"), (a_K_axis, "a.K"),
            (b_K_axis, "b.K"), (b_N_axis, "b.N"),
        ):
            if axis is None:
                raise IsaEmissionError(
                    f"plena.matmul: missing {name} axis in dim_roles; "
                    f"a={a_roles!r} b={b_roles!r} c={c_roles!r}"
                )

        M = int(a_reg.extents[a_M_axis])
        K = int(a_reg.extents[a_K_axis])
        N = int(b_reg.extents[b_N_axis])
        if int(b_reg.extents[b_K_axis]) != K:
            raise IsaEmissionError(
                f"plena.matmul: a.K extent {K} != b.K extent "
                f"{int(b_reg.extents[b_K_axis])}"
            )
        if int(c_reg.extents[c_M_axis]) != M:
            raise IsaEmissionError(
                f"plena.matmul: c.M extent {int(c_reg.extents[c_M_axis])} "
                f"!= a.M extent {M}"
            )
        if int(c_reg.extents[c_N_axis]) != N:
            raise IsaEmissionError(
                f"plena.matmul: c.N extent {int(c_reg.extents[c_N_axis])} "
                f"!= b.N extent {N}"
            )

        if M % mlen != 0 or K % mlen != 0:
            raise IsaEmissionError(
                f"plena.matmul: M ({M}) and K ({K}) must be multiples of "
                f"MLEN ({mlen})"
            )
        M_tiles = M // mlen
        K_tiles = K // mlen
        # transpose_b: standard layout has B = (K, N) row-major, i.e.
        # K is the outer (slower-varying) dim and N is inner — in
        # physical axis indices, ``K_axis < N_axis``. When the kernel
        # author intends ``B = (N, K)`` (nn.Linear weight convention)
        # the order flips: ``N_axis < K_axis``, and the emitter must
        # swap M_MM for M_TMM so the systolic array sees the right
        # operand orientation.
        transpose_b = b_N_axis < b_K_axis

        # The legacy dst_row_stride was the product of every physical
        # dim of dst strictly after the M axis (= "elements between
        # consecutive rows of C"). With a 4D BSHD c_region we can
        # derive it from the region's extents directly.
        dst_row_stride = 1
        for ax in range(c_M_axis + 1, len(c_reg.extents)):
            dst_row_stride *= int(c_reg.extents[ax])
        if dst_row_stride <= 0:
            dst_row_stride = None

        lhs_raw_off = self._region_origin_offset(lhs, a_reg)
        rhs_raw_off = self._region_origin_offset(rhs, b_reg)
        dst_raw_off = self._region_origin_offset(dst, c_reg)

        # Each of lhs / rhs / dst offsets supports either a compile-time
        # int (folded into the emitter's static residual) or an arbitrary
        # PrimExpr (materialised to a gp register here, passed in via the
        # matching `*_offset_reg`). Two offsets that are structurally the
        # same PrimExpr (e.g. ``rhs = by*hlen``, ``dst = by*hlen``) share
        # one materialised register so we don't run into the 16-GP cap.
        # Materialised registers are released after the emit returns.
        materialised_handles: List = []
        cached: List = []  # list of (raw_expr, register) for CSE lookup

        def _resolve_offset(raw, name: str):
            if isinstance(raw, tir.IntImm):
                return int(raw.value), None
            if isinstance(raw, int):
                return int(raw), None
            if isinstance(raw, tir.PrimExpr):
                for prev_raw, prev_reg in cached:
                    if tvm.ir.structural_equal(prev_raw, raw):
                        return 0, prev_reg
                m = self.materializer.materialize(raw)
                self.shim.compiler.generated_code += m.isa
                materialised_handles.append(m)
                cached.append((raw, m.register))
                # Pin so the emit_matmul_general body below can't pick
                # this register as a spill candidate while the inner
                # ``allocate_gp(7)`` runs. Unpinned, auto-spill would
                # save the offset value to IntRAM and then hand the
                # physical register out to ``gp_act_orow`` / etc,
                # silently corrupting the offset.
                self.shim.compiler.register_allocator.pin_gp(m.register)
                return 0, m.register
            raise IsaEmissionError(
                f"plena.matmul {name} must be int or PrimExpr; got {raw!r}"
            )

        # Pre-allocate the 7 scratch GPs the emitter needs and pin them
        # BEFORE materialising the dynamic offsets. Order matters: if we
        # materialised first, _auto_spill triggered by allocate_gp(7) could
        # spill the offset regs (despite pin_gp) and then hand the same
        # physical registers back as scratch — silently aliasing
        # `lhs_offset_reg`/`rhs_offset_reg`/`dst_offset_reg` with
        # `gp_act_orow`/`gp_out_orow`/`gp_mat`. By taking scratch first we
        # guarantee offset regs are disjoint from scratch regs.
        ra = self.shim.compiler.register_allocator
        scratch_regs = ra.allocate_gp(7)
        for r in scratch_regs:
            ra.pin_gp(r)

        try:
            lhs_off_static, lhs_off_reg = _resolve_offset(lhs_raw_off, "lhs_offset")
            rhs_off_static, rhs_off_reg = _resolve_offset(rhs_raw_off, "rhs_offset")
            dst_off_static, dst_off_reg = _resolve_offset(dst_raw_off, "dst_offset")

            self.emitter.emit_matmul_general(
                M_tiles=M_tiles,
                K_tiles=K_tiles,
                N=N,
                lhs_vram_base=int(lhs.address),
                lhs_offset=lhs_off_static,
                lhs_offset_reg=lhs_off_reg,
                rhs_mram_base=int(rhs.address),
                rhs_offset=rhs_off_static,
                rhs_offset_reg=rhs_off_reg,
                dst_vram_base=int(dst.address),
                dst_offset=dst_off_static,
                dst_offset_reg=dst_off_reg,
                dst_row_stride=dst_row_stride,
                task_id=op.annotations.get("intrinsic", "matmul"),
                scratch_regs=scratch_regs,
                transpose_b=transpose_b,
                unroll_loops=False,
            )
        finally:
            for m in materialised_handles:
                ra.unpin_gp(m.register)
                m.release()
            for r in scratch_regs:
                ra.unpin_gp(r)
            ra.free_gp(scratch_regs)

    def _emit_mm_slot(self, mod: _hlir.HLIRModule, op: _hlir.Op) -> None:
        lhs = mod.get_buffer(op.buffer_args[0])
        rhs = mod.get_buffer(op.buffer_args[1])
        dst = mod.get_buffer(op.buffer_args[2])
        _check_scope(lhs, _scope.VRAM, op.kind, "lhs")
        _check_scope(rhs, _scope.MRAM, op.kind, "rhs")
        _check_scope(dst, _scope.VRAM, op.kind, "dst")
        if len(op.scalar_args) != 4:
            raise IsaEmissionError(
                f"plena.mm_slot expects exactly 4 scalar args "
                f"(lhs_row_offset, rhs_col_offset, dst_col_offset, col_count); "
                f"got {len(op.scalar_args)}"
            )
        lhs_row_offset_raw = op.scalar_args[0]
        rhs_col_offset_raw = op.scalar_args[1]
        dst_col_offset_raw = op.scalar_args[2]
        col_count_raw = op.scalar_args[3]
        # lhs_row_offset can be either a compile-time int (literal / IntImm)
        # or a dynamic PrimExpr (e.g. `h * mlen * mlen` from a TIR loop).
        # Static case: fold into the lhs_vram_addr literal.
        # Dynamic case: materialize `lhs.address + offset` to a register and
        # pass that as lhs_vram_addr_reg.
        lhs_addr_m = None
        if isinstance(lhs_row_offset_raw, tir.IntImm):
            lhs_row_offset = int(lhs_row_offset_raw.value)
        elif isinstance(lhs_row_offset_raw, int):
            lhs_row_offset = int(lhs_row_offset_raw)
        elif isinstance(lhs_row_offset_raw, tir.PrimExpr):
            lhs_row_offset = None
            full_addr_expr = tir.Add(
                tir.IntImm("int32", int(lhs.address)),
                lhs_row_offset_raw,
            )
            lhs_addr_m = self.materializer.materialize(full_addr_expr)
            self.shim.compiler.generated_code += lhs_addr_m.isa
        else:
            raise IsaEmissionError(
                f"plena.mm_slot lhs_row_offset must be int or PrimExpr; "
                f"got {type(lhs_row_offset_raw).__name__}: {lhs_row_offset_raw!r}"
            )
        if lhs_row_offset is not None and lhs_row_offset < 0:
            raise IsaEmissionError(
                f"plena.mm_slot lhs_row_offset must be >= 0; got {lhs_row_offset}"
            )
        if isinstance(rhs_col_offset_raw, tir.PrimExpr) and not isinstance(rhs_col_offset_raw, tir.IntImm):
            rhs_col_offset = None
            rhs_off_m = self.materializer.materialize(rhs_col_offset_raw)
            self.shim.compiler.generated_code += rhs_off_m.isa
        else:
            rhs_col_offset = int(rhs_col_offset_raw)
            rhs_off_m = None
        if isinstance(dst_col_offset_raw, tir.PrimExpr) and not isinstance(dst_col_offset_raw, tir.IntImm):
            dst_col_offset = None
            dst_off_m = self.materializer.materialize(dst_col_offset_raw)
            self.shim.compiler.generated_code += dst_off_m.isa
        else:
            dst_col_offset = int(dst_col_offset_raw)
            dst_off_m = None
        try:
            col_count = int(col_count_raw)
        except TypeError as exc:
            raise IsaEmissionError(
                f"plena.mm_slot col_count must be a compile-time integer; got "
                f"{type(col_count_raw).__name__}: {col_count_raw!r}"
            ) from exc
        lhs_rows, lhs_cols = self._logical_2d(lhs.shape)
        rhs_rows, rhs_cols = self._logical_2d(rhs.shape)
        dst_rows, dst_cols = self._logical_2d(dst.shape)
        # LHS must contain at least one mlen*mlen tile. For static offsets
        # we can range-check at compile time; for dynamic offsets the kernel
        # author is responsible for keeping the offset in range.
        tile_elems = self.shim.mlen * self.shim.mlen
        if lhs_row_offset is not None and lhs_row_offset + tile_elems > lhs.num_elements:
            raise IsaEmissionError(
                f"plena.mm_slot lhs tile out of range; "
                f"lhs_row_offset={lhs_row_offset} + mlen*mlen={tile_elems} "
                f"exceeds buffer {lhs.name} num_elements={lhs.num_elements}"
            )
        if rhs_rows != self.shim.mlen or dst_rows != self.shim.mlen:
            raise IsaEmissionError(
                f"plena.mm_slot rhs/dst must have mlen rows; got rhs=({rhs_rows}, {rhs_cols}) "
                f"dst=({dst_rows}, {dst_cols})"
            )
        rhs_col_offset_check = 0 if rhs_col_offset is None else rhs_col_offset
        dst_col_offset_check = 0 if dst_col_offset is None else dst_col_offset
        if rhs_col_offset_check < 0 or dst_col_offset_check < 0 or col_count <= 0:
            raise IsaEmissionError(
                f"plena.mm_slot requires non-negative offsets and positive col_count; got "
                f"rhs_col_offset={rhs_col_offset_raw} dst_col_offset={dst_col_offset_raw} col_count={col_count}"
            )
        if rhs_col_offset is not None and rhs_col_offset + col_count > rhs_cols:
            raise IsaEmissionError(
                f"plena.mm_slot rhs slot exceeds rhs width; rhs_width={rhs_cols} "
                f"rhs_col_offset={rhs_col_offset} col_count={col_count}"
            )
        if dst_col_offset is not None and dst_col_offset + col_count > dst_cols:
            raise IsaEmissionError(
                f"plena.mm_slot dst slot exceeds dst width; dst_width={dst_cols} "
                f"dst_col_offset={dst_col_offset} col_count={col_count}"
            )
        try:
            if lhs_addr_m is not None:
                lhs_vram_addr_arg = 0  # ignored when reg form is used
                lhs_vram_addr_reg = lhs_addr_m.register
            else:
                lhs_vram_addr_arg = lhs.address + lhs_row_offset
                lhs_vram_addr_reg = None
            self.emitter.emit_slot_matmul(
                lhs_vram_addr=lhs_vram_addr_arg,
                lhs_vram_addr_reg=lhs_vram_addr_reg,
                rhs_mram_addr=rhs.address,
                rhs_col_offset=0 if rhs_col_offset is None else rhs_col_offset,
                rhs_col_offset_reg=None if rhs_off_m is None else rhs_off_m.register,
                dst_vram_addr=dst.address,
                dst_col_offset=0 if dst_col_offset is None else dst_col_offset,
                dst_col_offset_reg=None if dst_off_m is None else dst_off_m.register,
                col_count=col_count,
                task_id=op.annotations.get("intrinsic", "mm_slot"),
            )
        finally:
            if rhs_off_m is not None:
                rhs_off_m.release()
            if dst_off_m is not None:
                dst_off_m.release()
            if lhs_addr_m is not None:
                lhs_addr_m.release()

    def _emit_v_zero(self, mod: _hlir.HLIRModule, op: _hlir.Op) -> None:
        """Region-based zero-fill on VRAM: ``dst[region] = 0``.

        Schema (region layer):
            buffer_args = [dst_region]   (VramRegion with 4D BSHD)
            scalar_args = []

        Lowers to ``V_MUL_VF dst, dst, f0, 0`` per mlen-wide chunk
        (f0 == 0 by convention).
        """
        if len(op.buffer_args) != 1:
            raise IsaEmissionError(
                f"v_zero expects 1 buffer_arg (dst region); "
                f"got {len(op.buffer_args)}"
            )
        if not isinstance(op.buffer_args[0], _hlir.VramRegion):
            raise IsaEmissionError(
                f"v_zero dst: expected VramRegion, got "
                f"{type(op.buffer_args[0]).__name__}"
            )
        if op.scalar_args:
            raise IsaEmissionError(
                f"v_zero expects 0 scalar_args; got {len(op.scalar_args)}"
            )
        dst_region: _hlir.VramRegion = op.buffer_args[0]
        dst = mod.get_buffer(dst_region.parent)
        _check_scope(dst, _scope.VRAM, op.kind, "dst")
        self.shim.compiler.generated_code += (
            f"; v_zero dst.parent={dst_region.parent} "
            f"starts={list(dst_region.starts)!r} "
            f"extents={list(dst_region.extents)!r}\n"
        )
        for d_off, _ in self._vram_region_iter_chunks(dst, dst_region):
            dst_addr = tir.Add(
                tir.IntImm("int32", int(dst.address)), d_off,
            )
            m_dst = self.materializer.materialize(dst_addr)
            self.shim.compiler.generated_code += m_dst.isa
            try:
                self.shim.compiler.generated_code += (
                    f"V_MUL_VF gp{m_dst.register}, gp{m_dst.register}, "
                    f"f0, 0\n"
                )
            finally:
                m_dst.release()

    def _emit_v_binary(self, mod: _hlir.HLIRModule, op: _hlir.Op,
                       *, binary_op: str) -> None:
        """Region-based vector binary op:
        ``dst[region] = lhs[region] <binop> rhs[region]`` elementwise.

        Schema (region layer):
            buffer_args = [lhs_region, rhs_region, dst_region]
                each is a ``VramRegion(parent, starts, extents)`` with
                ``starts`` / ``extents`` length 4 in canonical BSHD
                order. Every (b, s, h, d) cell within ``extents`` of
                the three regions is paired up for the elementwise
                op. The three regions must agree on ``extents``.
            scalar_args = []  (region carries everything)

        Emission walks each region with ``_vram_region_iter_chunks``,
        which folds the cluster (packed-head) axis and unrolls the
        d_tile axis automatically. One HLIR op may therefore emit
        N V_*_VV instructions (N = product of non-cluster outer
        extents × d_chunks).
        """
        op_to_insn = {
            "add": "V_ADD_VV",
            "sub": "V_SUB_VV",
            "mul": "V_MUL_VV",
        }
        opcode = op_to_insn[binary_op]
        if len(op.buffer_args) != 3:
            raise IsaEmissionError(
                f"{op.kind} expects 3 buffer_args (lhs/rhs/dst regions); "
                f"got {len(op.buffer_args)}"
            )
        for slot, name in enumerate(("lhs", "rhs", "dst")):
            if not isinstance(op.buffer_args[slot], _hlir.VramRegion):
                raise IsaEmissionError(
                    f"{op.kind} {name}: expected VramRegion, got "
                    f"{type(op.buffer_args[slot]).__name__}"
                )
        if op.scalar_args:
            raise IsaEmissionError(
                f"{op.kind} expects 0 scalar_args (region carries shape); "
                f"got {len(op.scalar_args)}"
            )
        lhs_region: _hlir.VramRegion = op.buffer_args[0]
        rhs_region: _hlir.VramRegion = op.buffer_args[1]
        dst_region: _hlir.VramRegion = op.buffer_args[2]
        lhs = mod.get_buffer(lhs_region.parent)
        rhs = mod.get_buffer(rhs_region.parent)
        dst = mod.get_buffer(dst_region.parent)
        _check_scope(lhs, _scope.VRAM, op.kind, "lhs")
        _check_scope(rhs, _scope.VRAM, op.kind, "rhs")
        _check_scope(dst, _scope.VRAM, op.kind, "dst")
        if (tuple(lhs_region.extents) != tuple(dst_region.extents)
                or tuple(rhs_region.extents) != tuple(dst_region.extents)):
            raise IsaEmissionError(
                f"{op.kind}: lhs/rhs/dst region extents must match; "
                f"lhs={tuple(lhs_region.extents)} "
                f"rhs={tuple(rhs_region.extents)} "
                f"dst={tuple(dst_region.extents)}"
            )

        self.shim.compiler.generated_code += (
            f"; v binary {op.kind} {opcode} "
            f"dst.parent={dst_region.parent} "
            f"starts={list(dst_region.starts)!r} "
            f"extents={list(dst_region.extents)!r}\n"
        )

        # Walk all three regions in lock-step. Each yield gives the
        # ``(vram_offset_expr, fp_step_elems)`` for one mlen-wide
        # chunk; we discard fp_step (no FPRAM here) and materialise
        # the three per-operand absolute addresses.
        lhs_iter = self._vram_region_iter_chunks(lhs, lhs_region)
        rhs_iter = self._vram_region_iter_chunks(rhs, rhs_region)
        dst_iter = self._vram_region_iter_chunks(dst, dst_region)
        for (l_off, _), (r_off, _), (d_off, _) in zip(
            lhs_iter, rhs_iter, dst_iter
        ):
            lhs_addr = tir.Add(
                tir.IntImm("int32", int(lhs.address)), l_off,
            )
            rhs_addr = tir.Add(
                tir.IntImm("int32", int(rhs.address)), r_off,
            )
            dst_addr = tir.Add(
                tir.IntImm("int32", int(dst.address)), d_off,
            )
            m_lhs = self.materializer.materialize(lhs_addr)
            self.shim.compiler.generated_code += m_lhs.isa
            m_rhs = self.materializer.materialize(rhs_addr)
            self.shim.compiler.generated_code += m_rhs.isa
            m_dst = self.materializer.materialize(dst_addr)
            self.shim.compiler.generated_code += m_dst.isa
            try:
                self.shim.compiler.generated_code += (
                    f"{opcode} gp{m_dst.register}, gp{m_lhs.register}, "
                    f"gp{m_rhs.register}, 0\n"
                )
            finally:
                m_dst.release()
                m_rhs.release()
                m_lhs.release()
        return

    def _emit_v_add(self, mod: _hlir.HLIRModule, op: _hlir.Op) -> None:
        self._emit_v_binary(mod, op, binary_op="add")

    def _emit_v_sub(self, mod: _hlir.HLIRModule, op: _hlir.Op) -> None:
        self._emit_v_binary(mod, op, binary_op="sub")

    def _emit_v_mul(self, mod: _hlir.HLIRModule, op: _hlir.Op) -> None:
        self._emit_v_binary(mod, op, binary_op="mul")

    def _emit_v_unary(self, mod: _hlir.HLIRModule, op: _hlir.Op,
                      *, opcode: str) -> None:
        """Region-based vector unary op: ``dst[region] = op(src[region])``.

        Schema (region layer):
            buffer_args = [src_region, dst_region]
                each is a ``VramRegion`` with 4D BSHD (starts, extents).
                The two regions must agree on ``extents``.
            scalar_args = []
        """
        if len(op.buffer_args) != 2:
            raise IsaEmissionError(
                f"{op.kind} expects 2 buffer_args (src/dst regions); "
                f"got {len(op.buffer_args)}"
            )
        for slot, name in enumerate(("src", "dst")):
            if not isinstance(op.buffer_args[slot], _hlir.VramRegion):
                raise IsaEmissionError(
                    f"{op.kind} {name}: expected VramRegion, got "
                    f"{type(op.buffer_args[slot]).__name__}"
                )
        if op.scalar_args:
            raise IsaEmissionError(
                f"{op.kind} expects 0 scalar_args (region carries shape); "
                f"got {len(op.scalar_args)}"
            )
        src_region: _hlir.VramRegion = op.buffer_args[0]
        dst_region: _hlir.VramRegion = op.buffer_args[1]
        src = mod.get_buffer(src_region.parent)
        dst = mod.get_buffer(dst_region.parent)
        _check_scope(src, _scope.VRAM, op.kind, "src")
        _check_scope(dst, _scope.VRAM, op.kind, "dst")
        if tuple(src_region.extents) != tuple(dst_region.extents):
            raise IsaEmissionError(
                f"{op.kind}: src/dst region extents must match; "
                f"src={tuple(src_region.extents)} dst={tuple(dst_region.extents)}"
            )

        self.shim.compiler.generated_code += (
            f"; v unary {op.kind} {opcode} "
            f"dst.parent={dst_region.parent} "
            f"starts={list(dst_region.starts)!r} "
            f"extents={list(dst_region.extents)!r}\n"
        )
        src_iter = self._vram_region_iter_chunks(src, src_region)
        dst_iter = self._vram_region_iter_chunks(dst, dst_region)
        for (s_off, _), (d_off, _) in zip(src_iter, dst_iter):
            src_addr = tir.Add(
                tir.IntImm("int32", int(src.address)), s_off,
            )
            dst_addr = tir.Add(
                tir.IntImm("int32", int(dst.address)), d_off,
            )
            m_src = self.materializer.materialize(src_addr)
            self.shim.compiler.generated_code += m_src.isa
            m_dst = self.materializer.materialize(dst_addr)
            self.shim.compiler.generated_code += m_dst.isa
            try:
                self.shim.compiler.generated_code += (
                    f"{opcode} gp{m_dst.register}, gp{m_src.register}, 0\n"
                )
            finally:
                m_dst.release()
                m_src.release()

    def _emit_v_exp(self, mod: _hlir.HLIRModule, op: _hlir.Op) -> None:
        self._emit_v_unary(mod, op, opcode="V_EXP_V")

    def _emit_v_reci(self, mod: _hlir.HLIRModule, op: _hlir.Op) -> None:
        self._emit_v_unary(mod, op, opcode="V_RECI_V")

    def _emit_v_sqrt(self, mod: _hlir.HLIRModule, op: _hlir.Op) -> None:
        self._emit_v_unary(mod, op, opcode="V_SQRT_V")

    def _emit_fp_copy_at(self, mod: _hlir.HLIRModule, op: _hlir.Op) -> None:
        self._emit_fp_scalar_op_at(mod, op, kernel_op="copy")

    def _emit_fp_zero_at(self, mod: _hlir.HLIRModule, op: _hlir.Op) -> None:
        """Store FP zero to one FPRAM slot via ``S_ST_FP f0, gp{dst}, 0``.

        Relies on the same ``f0 == 0`` convention plena.tile_zero and
        plena.copy_v_to_v already depend on. Single scalar arg = the
        FPRAM destination address (allowed to be a PrimExpr — the
        materialiser folds in the fragment's allocated FPRAM base)."""
        if len(op.scalar_args) != 1:
            raise IsaEmissionError(
                f"{op.kind} expects 1 scalar address arg, got {len(op.scalar_args)}"
            )
        dst_addr_expr = self._resolve_fp_scalar_addr_arg(
            mod, op.scalar_args[0], op.kind, "dst",
        )
        m_dst = self.materializer.materialize(dst_addr_expr)
        self.shim.compiler.generated_code += m_dst.isa
        try:
            lines = [
                f"; fp scalar task {op.annotations.get('intrinsic', op.kind)} op=zero",
                f"S_ST_FP f0, gp{m_dst.register}, 0",
            ]
            self.shim.compiler.generated_code += "\n".join(lines) + "\n"
        finally:
            m_dst.release()

    def _emit_fp_add_at(self, mod: _hlir.HLIRModule, op: _hlir.Op) -> None:
        self._emit_fp_scalar_op_at(mod, op, kernel_op="add")

    def _emit_fp_sub_at(self, mod: _hlir.HLIRModule, op: _hlir.Op) -> None:
        self._emit_fp_scalar_op_at(mod, op, kernel_op="sub")

    def _emit_fp_mul_at(self, mod: _hlir.HLIRModule, op: _hlir.Op) -> None:
        self._emit_fp_scalar_op_at(mod, op, kernel_op="mul")

    def _emit_fp_max_at(self, mod: _hlir.HLIRModule, op: _hlir.Op) -> None:
        self._emit_fp_scalar_op_at(mod, op, kernel_op="max")

    def _emit_fp_exp_at(self, mod: _hlir.HLIRModule, op: _hlir.Op) -> None:
        self._emit_fp_scalar_op_at(mod, op, kernel_op="exp")

    def _emit_fp_reci_at(self, mod: _hlir.HLIRModule, op: _hlir.Op) -> None:
        self._emit_fp_scalar_op_at(mod, op, kernel_op="reci")

    def _emit_fp_sqrt_at(self, mod: _hlir.HLIRModule, op: _hlir.Op) -> None:
        self._emit_fp_scalar_op_at(mod, op, kernel_op="sqrt")

    # `_at` row ops: scalars are (FP scalar address, dim2, dim3) for the
    # variants that touch fpram, or just (dim2, dim3) for exp. The emitter
    # maps (dim2, dim3) to a physical VRAM row and synthesizes a V_MASK
    # for narrow packed D tiles.
    def _emit_row_reduce_max_at(self, mod: _hlir.HLIRModule, op: _hlir.Op) -> None:
        self._emit_row_scalar_op_at(
            mod, op, row_op="reduce_max", reduce=True, masked=True,
        )

    def _emit_row_reduce_sum_at(self, mod: _hlir.HLIRModule, op: _hlir.Op) -> None:
        self._emit_row_scalar_op_at(
            mod, op, row_op="reduce_sum", reduce=True, masked=True,
        )

    # Single-row VRAM × FPRAM-scalar ops. One HLIR op = one HW
    # instruction. Multi-row callers wrap in outer ``for row``.
    def _emit_row_exp(self, mod: _hlir.HLIRModule, op: _hlir.Op) -> None:
        self._emit_row_scalar_op_at(mod, op, row_op="exp", masked=True)

    def _emit_row_sub_fp(self, mod: _hlir.HLIRModule, op: _hlir.Op) -> None:
        self._emit_row_scalar_op_at(mod, op, row_op="sub", masked=True, has_fp=True)

    def _emit_row_mul_fp(self, mod: _hlir.HLIRModule, op: _hlir.Op) -> None:
        self._emit_row_scalar_op_at(mod, op, row_op="mul", masked=True, has_fp=True)

    def _emit_row_add_fp(self, mod: _hlir.HLIRModule, op: _hlir.Op) -> None:
        self._emit_row_scalar_op_at(mod, op, row_op="add", masked=True, has_fp=True)

    # ------------------------------------------------------------------
    # Slice-level VRAM <-> FPRAM transfer. HLIR carries the whole logical
    # region (VramRegion: starts + extents on the parent buffer); this
    # emitter splits it into HW-MLEN-wide chunks, computes each chunk's
    # physical VRAM offset via the parent's 7D tile layout, and emits
    # one S_MAP_*_FP/V per chunk.
    #
    # The parent is always a 4D ``(B, S, H, D)`` BSHD buffer (the
    # pad-to-4D step in to_plena guarantees rank==4 on every VRAM/MRAM
    # buffer). Its physical placement in VRAM is the 7D tile layout
    # described in ``hlir.TileLayout``:
    #
    #     (D_TILES, S_TILES, H_GROUPS, B, MLEN, LANE_COUNT, D_INNER)
    #
    # A logical position ``(b, s, h, d)`` decomposes as:
    #     d_tile  = d  // MLEN          d_inner_off = d  %  MLEN
    #     s_tile  = s  // MLEN          s_inner_off = s  %  MLEN
    #     h_grp   = h  // LANE_COUNT    lane        = h  %  LANE_COUNT
    # ------------------------------------------------------------------
    def _vram_region_iter_chunks(
        self,
        parent: _hlir.Buffer,
        region: _hlir.VramRegion,
    ):
        """Yield ``(vram_offset_expr, fp_step_elems)`` for each HW-MLEN
        chunk inside ``region``. ``fp_step_elems`` is the cumulative
        element count consumed by all chunks so far — callers add it
        to the base fp address.

        Region semantics (post pad-to-4D): every parent is rank 4
        BSHD; ``starts`` / ``extents`` are 4-tuples. The region's
        last-axis extent (``ed``) drives the chunking — one S_MAP per
        ``D_INNER`` slots along D. The (b, s, h, d_tile) outer
        coordinates are walked once each; the chunk's physical VRAM
        offset is the 7D inner-tile address.
        """
        starts = region.starts
        extents = region.extents
        if len(parent.shape) != 4:
            raise IsaEmissionError(
                f"VramRegion(parent={region.parent!r}) expects 4D BSHD "
                f"parent; got shape {tuple(parent.shape)}. pad-to-4D "
                f"in to_plena should have normalised this."
            )
        if len(starts) != 4 or len(extents) != 4:
            raise IsaEmissionError(
                f"VramRegion(parent={region.parent!r}) rank mismatch: "
                f"starts={tuple(starts)} extents={tuple(extents)}; "
                f"both must be 4-tuples"
            )
        # Row-major-flat path: parent has no tile_layout AND no
        # cluster_dim. This covers (a) author-pinned global.vram /
        # global.mram tensor caches (testbench-loaded contiguous,
        # not 7D-tile-padded) and (b) any small buffer that fits a
        # single tile (logical extent ≤ mlen on every dim). Each
        # mlen-wide region chunk maps directly to a flat row-major
        # slice. Buffers with a non-None tile_layout keep the 7D path
        # below — their physical layout walks mlen-row tiles.
        cluster_dim_pre = getattr(parent, "cluster_dim", None)
        if cluster_dim_pre is None and parent.tile_layout is None:
            mlen = self.shim.mlen
            shape = [int(d) for d in parent.shape]
            # Row-major strides on the BSHD shape (rank-4).
            row_strides = [1] * 4
            for i in range(2, -1, -1):
                row_strides[i] = row_strides[i + 1] * shape[i + 1]
            eb, es, eh, ed = (int(x) for x in extents)
            total_elems = eb * es * eh * ed
            if total_elems % mlen != 0:
                raise IsaEmissionError(
                    f"VramRegion(parent={region.parent!r}, cluster-less): "
                    f"total region elems={total_elems} not a multiple of "
                    f"MLEN={mlen}"
                )
            chunks = total_elems // mlen

            def _start_plus_simple(axis: int):
                s = starts[axis]
                if isinstance(s, int):
                    return tir.IntImm("int32", int(s))
                return s

            def _mul_s(expr, k: int):
                if k == 0:
                    return tir.IntImm("int32", 0)
                if k == 1:
                    return expr
                if isinstance(expr, tir.IntImm):
                    return tir.IntImm("int32", int(expr.value) * k)
                return tir.Mul(expr, tir.IntImm("int32", int(k)))

            def _sum_s(terms):
                nz = [t for t in terms if not (isinstance(t, tir.IntImm) and int(t.value) == 0)]
                if not nz:
                    return tir.IntImm("int32", 0)
                acc = nz[0]
                for t in nz[1:]:
                    acc = tir.Add(acc, t)
                return acc

            base_off = _sum_s([
                _mul_s(_start_plus_simple(0), row_strides[0]),
                _mul_s(_start_plus_simple(1), row_strides[1]),
                _mul_s(_start_plus_simple(2), row_strides[2]),
                _start_plus_simple(3),
            ])
            fp_elems = 0
            for c in range(chunks):
                if c == 0:
                    yield base_off, fp_elems
                else:
                    yield tir.Add(base_off, tir.IntImm("int32", c * mlen)), fp_elems
                fp_elems += mlen
            return

        tl = parent.tile_layout
        if tl is None:
            # ``make_tile_layout`` returns None for buffers that fit a
            # single inner tile (s ≤ mlen ∧ d ≤ mlen on BSHD). Synthesise
            # the trivial 1×1×1×1 layout so the offset math below works
            # uniformly without a separate code path.
            b_sz, s_sz, h_sz, d_sz = (int(x) for x in parent.shape)
            tl = _hlir.TileLayout(
                logical_b=b_sz, logical_s=s_sz, logical_h=h_sz, logical_d=d_sz,
                d_tiles=1, s_tiles=1, h_groups=1,
                mlen=self.shim.mlen, lane_count=1,
                d_inner=d_sz if d_sz > 0 else self.shim.mlen,
            )

        eb, es, eh, ed = (int(x) for x in extents)
        if ed % tl.d_inner != 0:
            raise IsaEmissionError(
                f"VramRegion(parent={region.parent!r}): innermost extent "
                f"ed={ed} not a multiple of D_INNER={tl.d_inner}"
            )
        d_chunks = ed // tl.d_inner

        # Lane axis is sync-wrap-folded by mid_ir: a single
        # ``S_MAP_*_FP/V`` instruction covers every lane in one issue,
        # so the emitter must NOT iterate the lane axis (doing so would
        # re-issue the same multi-lane instruction lane_count times at
        # offsets that no longer align to mlen). Assert the region
        # covers the full lane span on whatever axis the parent's
        # ``cluster_dim`` marks, then fold that axis out of the walk.
        cluster_dim = getattr(parent, "cluster_dim", None)
        outer_iter = {"b": eb, "s": es, "h": eh}
        if cluster_dim is not None:
            # BSHD positions: 0=B, 1=S, 2=H, 3=D. Lane never lands at D.
            lane_key = {0: "b", 1: "s", 2: "h"}.get(cluster_dim)
            if lane_key is None:
                raise IsaEmissionError(
                    f"VramRegion(parent={region.parent!r}): cluster_dim "
                    f"={cluster_dim} is not a recognised BSHD lane "
                    f"position (0=B / 1=S / 2=H)"
                )
            lane_span = int(parent.shape[cluster_dim])
            lane_ext = outer_iter[lane_key]
            if lane_ext != lane_span:
                raise IsaEmissionError(
                    f"VramRegion(parent={region.parent!r}): lane axis "
                    f"({lane_key.upper()}, cluster_dim={cluster_dim}) "
                    f"must cover the full lane span "
                    f"({lane_span}) under sync wrap; got extent "
                    f"{lane_ext}"
                )
            # Fold lane axis out — one S_MAP per (b, s, h_grp, d_chunk)
            # except along the lane direction itself.
            outer_iter[lane_key] = 1

        # 7D physical strides (in elements). One inner tile holds
        # MLEN * LANE_COUNT * D_INNER contiguous values; outer tiles
        # walk (B, H_GROUPS, S_TILES, D_TILES) with the standard 7D
        # stride pattern.
        tile_elems = tl.mlen * tl.lane_count * tl.d_inner
        b_stride = tile_elems
        h_grp_stride = tl.logical_b * tile_elems
        s_tile_stride = tl.h_groups * h_grp_stride
        d_tile_stride = tl.s_tiles * s_tile_stride

        # Helper: render ``starts[axis]`` + extra as a PrimExpr.
        def _start_plus(axis: int, extra: int):
            s = starts[axis]
            if isinstance(s, int):
                v = s + extra
                return tir.IntImm("int32", int(v))
            if extra == 0:
                return s
            return tir.Add(s, tir.IntImm("int32", int(extra)))

        def _floordiv(expr, divisor: int):
            if divisor == 1:
                return expr
            if isinstance(expr, tir.IntImm):
                return tir.IntImm("int32", int(expr.value) // divisor)
            return tir.FloorDiv(expr, tir.IntImm("int32", int(divisor)))

        def _floormod(expr, divisor: int):
            if divisor == 1:
                return tir.IntImm("int32", 0)
            if isinstance(expr, tir.IntImm):
                return tir.IntImm("int32", int(expr.value) % divisor)
            return tir.FloorMod(expr, tir.IntImm("int32", int(divisor)))

        def _mul(expr, k: int):
            if k == 0:
                return tir.IntImm("int32", 0)
            if k == 1:
                return expr
            if isinstance(expr, tir.IntImm):
                return tir.IntImm("int32", int(expr.value) * k)
            return tir.Mul(expr, tir.IntImm("int32", int(k)))

        def _sum(terms):
            non_zero = [
                t for t in terms
                if not (isinstance(t, tir.IntImm) and int(t.value) == 0)
            ]
            if not non_zero:
                return tir.IntImm("int32", 0)
            acc = non_zero[0]
            for t in non_zero[1:]:
                acc = tir.Add(acc, t)
            return acc

        fp_elems_so_far = 0
        # Cartesian walk over (b_off, h_off, s_off, d_chunk). The lane
        # axis among (B, S, H) was folded to extent 1 above so a
        # single S_MAP per chunk covers every lane.
        for d_chunk in range(d_chunks):
            for s_off in range(outer_iter["s"]):
                for h_off in range(outer_iter["h"]):
                    for b_off in range(outer_iter["b"]):
                        b_expr = _start_plus(0, b_off)
                        s_expr = _start_plus(1, s_off)
                        h_expr = _start_plus(2, h_off)
                        # d_start covers the current D_INNER-wide slot
                        # within the region; it indexes into parent's D.
                        d_expr = _start_plus(3, d_chunk * tl.d_inner)

                        d_tile = _floordiv(d_expr, tl.mlen)
                        d_inner = _floormod(d_expr, tl.mlen)
                        s_tile = _floordiv(s_expr, tl.mlen)
                        s_inner = _floormod(s_expr, tl.mlen)
                        h_grp = _floordiv(h_expr, tl.lane_count)
                        lane = _floormod(h_expr, tl.lane_count)

                        # 7D physical flat offset.
                        terms = [
                            _mul(d_tile, d_tile_stride),
                            _mul(s_tile, s_tile_stride),
                            _mul(h_grp, h_grp_stride),
                            _mul(b_expr, b_stride),
                            _mul(s_inner, tl.lane_count * tl.d_inner),
                            _mul(lane, tl.d_inner),
                            d_inner,
                        ]
                        vram_off = _sum(terms)
                        yield vram_off, fp_elems_so_far
                        # One S_MAP transfers ``lane_count * d_inner``
                        # (= MLEN) contiguous FPRAM slots — the whole
                        # lane group's slice in one issue.
                        fp_elems_so_far += tl.lane_count * tl.d_inner

    def _emit_v_fp_transfer_slice(
        self,
        mod: _hlir.HLIRModule,
        op: _hlir.Op,
        *,
        direction: str,                          # "v_to_fp" or "fp_to_v"
    ) -> None:
        if len(op.buffer_args) != 1 or not isinstance(op.buffer_args[0], _hlir.VramRegion):
            raise IsaEmissionError(
                f"{op.kind}: buffer_args[0] must be VramRegion; "
                f"got {op.buffer_args!r}"
            )
        if len(op.scalar_args) != 1:
            raise IsaEmissionError(
                f"{op.kind}: expected 1 scalar arg (fp_addr); "
                f"got {len(op.scalar_args)}"
            )
        region: _hlir.VramRegion = op.buffer_args[0]
        vram = mod.get_buffer(region.parent)
        _check_scope(vram, _scope.VRAM, op.kind, "vram")

        fp_addr_base = self._resolve_fp_scalar_addr_arg(
            mod, op.scalar_args[0], op.kind, "fp",
        )
        opcode = "S_MAP_FP_V" if direction == "v_to_fp" else "S_MAP_V_FP"

        self.shim.compiler.generated_code += (
            f"; v↔fp transfer slice {op.kind} parent={region.parent} "
            f"starts={list(region.starts)!r} extents={list(region.extents)!r}\n"
        )

        for vram_off_expr, fp_step in self._vram_region_iter_chunks(vram, region):
            vram_addr_expr = tir.Add(
                tir.IntImm("int32", int(vram.address)),
                vram_off_expr,
            )
            fp_chunk_addr = (
                fp_addr_base if fp_step == 0
                else tir.Add(fp_addr_base, tir.IntImm("int32", int(fp_step)))
            )
            m_vram = self.materializer.materialize(vram_addr_expr)
            self.shim.compiler.generated_code += m_vram.isa
            m_fp = self.materializer.materialize(fp_chunk_addr)
            self.shim.compiler.generated_code += m_fp.isa
            try:
                if direction == "v_to_fp":
                    self.shim.compiler.generated_code += (
                        f"{opcode} gp{m_fp.register}, gp{m_vram.register}, 0\n"
                    )
                else:
                    self.shim.compiler.generated_code += (
                        f"{opcode} gp{m_vram.register}, gp{m_fp.register}, 0\n"
                    )
            finally:
                m_fp.release()
                m_vram.release()

    def _emit_v_fp_transfer_slice_v_to_fp(
        self, mod: _hlir.HLIRModule, op: _hlir.Op,
    ) -> None:
        self._emit_v_fp_transfer_slice(mod, op, direction="v_to_fp")

    def _emit_v_fp_transfer_slice_fp_to_v(
        self, mod: _hlir.HLIRModule, op: _hlir.Op,
    ) -> None:
        self._emit_v_fp_transfer_slice(mod, op, direction="fp_to_v")

    def _emit_copy_v_to_v(self, mod: _hlir.HLIRModule, op: _hlir.Op) -> None:
        """Region-based VRAM→VRAM copy: ``dst[region] = src[region]``.

        Schema (region layer):
            buffer_args = [src_region, dst_region]   (VramRegion 4D BSHD)
            scalar_args = []

        Each mlen-wide chunk emits one ``V_ADD_VF dst, src, f0, 0`` —
        f0 == 0 by convention so ``src + 0`` is just src.
        """
        if len(op.buffer_args) != 2:
            raise IsaEmissionError(
                f"copy_v_to_v expects 2 buffer_args (src/dst regions); "
                f"got {len(op.buffer_args)}"
            )
        for slot, name in enumerate(("src", "dst")):
            if not isinstance(op.buffer_args[slot], _hlir.VramRegion):
                raise IsaEmissionError(
                    f"copy_v_to_v {name}: expected VramRegion, got "
                    f"{type(op.buffer_args[slot]).__name__}"
                )
        if op.scalar_args:
            raise IsaEmissionError(
                f"copy_v_to_v expects 0 scalar_args; "
                f"got {len(op.scalar_args)}"
            )
        src_region: _hlir.VramRegion = op.buffer_args[0]
        dst_region: _hlir.VramRegion = op.buffer_args[1]
        src = mod.get_buffer(src_region.parent)
        dst = mod.get_buffer(dst_region.parent)
        _check_scope(src, _scope.VRAM, op.kind, "src")
        _check_scope(dst, _scope.VRAM, op.kind, "dst")
        if tuple(src_region.extents) != tuple(dst_region.extents):
            raise IsaEmissionError(
                f"copy_v_to_v: src/dst region extents must match; "
                f"src={tuple(src_region.extents)} "
                f"dst={tuple(dst_region.extents)}"
            )
        self.shim.compiler.generated_code += (
            f"; copy_v_to_v src.parent={src_region.parent} -> "
            f"dst.parent={dst_region.parent} "
            f"extents={list(dst_region.extents)!r}\n"
        )
        src_iter = self._vram_region_iter_chunks(src, src_region)
        dst_iter = self._vram_region_iter_chunks(dst, dst_region)
        for (s_off, _), (d_off, _) in zip(src_iter, dst_iter):
            src_addr = tir.Add(
                tir.IntImm("int32", int(src.address)), s_off,
            )
            dst_addr = tir.Add(
                tir.IntImm("int32", int(dst.address)), d_off,
            )
            m_src = self.materializer.materialize(src_addr)
            self.shim.compiler.generated_code += m_src.isa
            m_dst = self.materializer.materialize(dst_addr)
            self.shim.compiler.generated_code += m_dst.isa
            try:
                self.shim.compiler.generated_code += (
                    f"V_ADD_VF gp{m_dst.register}, gp{m_src.register}, "
                    f"f0, 0\n"
                )
            finally:
                m_dst.release()
                m_src.release()

    # ------------------------------------------------------------------
    # Structured ops: For
    # ------------------------------------------------------------------
    def _emit_for(self, mod: _hlir.HLIRModule, op: _hlir.Op) -> None:
        """Emit `C_LOOP_START / body / inc / C_LOOP_END` for a structured
        For op.

        PLENA's hardware loop is:
            C_LOOP_START gp_loop, IMM_count
              ...
            C_LOOP_END gp_loop
        where IMM_count is a literal iteration count and gp_loop is an
        internal counter the hardware decrements -- it is NOT the
        iteration index. So we need TWO registers:
            * `gp_loop`  -- hardware counter, opaque to body
            * `gp_idx`   -- body-visible iteration variable, bound to
                            the TIR loop_var via symbol_table; manually
                            initialised before C_LOOP_START and
                            incremented by 1 at the end of every
                            iteration.

        Constraints:
            * extent must be a Python int / IntImm (immediate field of
              C_LOOP_START is a literal). PrimExpr extents are not
              supported -- they would need a compile-time evaluation
              pass or a different lowering (no native loop-with-runtime-
              count instruction in PLENA's ISA).
            * init must be int (typically 0). PrimExpr inits are
              unsupported for the same reason: would force runtime
              loop-bound recomputation.
        """
        loop_var = op.annotations.get("loop_var")
        extent = op.annotations.get("extent")
        init = op.annotations.get("init", 0)
        if loop_var is None or extent is None:
            raise IsaEmissionError(
                f"for-op missing loop_var or extent annotation: {op!r}"
            )
        if not isinstance(extent, (int, tir.IntImm)):
            raise IsaEmissionError(
                f"for-op extent must be a compile-time integer (PLENA's "
                f"C_LOOP_START takes an immediate). Got {type(extent).__name__}: "
                f"{extent!r}. Restructure the kernel so the loop bound is known "
                f"at TIR-construction time."
            )
        if not isinstance(init, (int, tir.IntImm)):
            raise IsaEmissionError(
                f"for-op init must be a compile-time integer. Got "
                f"{type(init).__name__}: {init!r}."
            )
        extent_imm = int(extent.value) if isinstance(extent, tir.IntImm) else int(extent)
        init_imm = int(init.value) if isinstance(init, tir.IntImm) else int(init)
        if loop_var in self.symbol_table:
            raise IsaEmissionError(
                f"loop_var {loop_var.name!r} (id={id(loop_var)}) already "
                f"bound; nested loops reusing the same Var aren't supported. "
                f"Active bindings: "
                f"{[(v.name, id(v)) for v in self.symbol_table]!r}"
            )

        ra = self.shim.compiler.register_allocator
        loop_kind = op.annotations.get("loop_kind", "serial")

        # Compile-time unroll: emit the body N times back-to-back with
        # loop_var rebound to a literal each iteration. Use this to break
        # out of MAX_LOOP_INSTRUCTIONS-per-iter when one outer iteration's
        # body would otherwise dispatch too many dynamic instructions
        # (e.g. an inner kv_block accumulation containing a 16x16 unrolled
        # emit_matmul). Costs one S_ADDI_INT per iter to re-init gp_idx;
        # the hardware loop overhead disappears entirely.
        if loop_kind in ("unroll", "unrolled"):
            gp_idx = ra.allocate_gp(1)[0]
            self.shim.compiler.generated_code += (
                f"; unroll for {loop_var.name} in "
                f"[{init_imm}, {init_imm + extent_imm}) -- idx gp{gp_idx}\n"
            )
            self.symbol_table[loop_var] = gp_idx
            ra.pin_gp(gp_idx)
            try:
                for i in range(extent_imm):
                    iter_val = init_imm + i
                    self.shim.compiler.generated_code += (
                        f"; ... unroll iter {i} -> {loop_var.name}={iter_val}\n"
                        f"S_ADDI_INT gp{gp_idx}, gp0, {iter_val}\n"
                    )
                    for j, sub_op in enumerate(op.body or []):
                        handler = self._dispatch.get(sub_op.kind)
                        if handler is None:
                            raise IsaEmissionError(
                                f"no ISA dispatcher for nested op kind "
                                f"{sub_op.kind!r} inside unrolled for-loop"
                            )
                        ra.push_site(f"unroll[{i}].body[{j}] {sub_op.kind}")
                        try:
                            handler(mod, sub_op)
                        finally:
                            ra.pop_site()
            finally:
                ra.unpin_gp(gp_idx)
                del self.symbol_table[loop_var]
            ra.free_gp([gp_idx])
            return

        # gp_loop is the PLENA hw counter — C_LOOP_END decrements it, so
        # it MUST stay in a GP and MUST be pinned for the whole body.
        gp_loop = ra.allocate_gp(1)[0]
        ra.pin_gp(gp_loop)

        # idx lives in IntRAM, not a GP. Deep nests (flash_attention with
        # an inner matmul, conv2d's 6-level grid) used to exhaust the GP
        # file when every loop pinned two GPs. Storing the idx in IntRAM
        # turns it into 1 GP per loop -- the materializer re-loads the
        # idx on every use via S_LD_INT.
        idx_addr = ra.claim_idx_slot()
        # Init: 0 -> intram[idx_addr]. gp0 is constant zero, so we can
        # store it directly without using a scratch GP.
        if init_imm == 0:
            self.shim.compiler.generated_code += (
                f"; for {loop_var.name} in [{init_imm}, {init_imm + extent_imm}) "
                f"-- hw counter gp{gp_loop}, idx ram[{idx_addr}]\n"
                f"S_ST_INT gp0, gp0, {idx_addr}\n"
                f"C_LOOP_START gp{gp_loop}, {extent_imm}\n"
            )
        else:
            # Non-zero init: borrow one GP to compute the value, store,
            # free immediately. Allocator is free to spill if needed.
            init_gp = ra.allocate_gp(1)[0]
            self.shim.compiler.generated_code += (
                f"; for {loop_var.name} in [{init_imm}, {init_imm + extent_imm}) "
                f"-- hw counter gp{gp_loop}, idx ram[{idx_addr}]\n"
                f"S_ADDI_INT gp{init_gp}, gp0, {init_imm}\n"
                f"S_ST_INT gp{init_gp}, gp0, {idx_addr}\n"
                f"C_LOOP_START gp{gp_loop}, {extent_imm}\n"
            )
            ra.free_gp([init_gp])

        self.symbol_table[loop_var] = ("ram", idx_addr)
        try:
            for j, sub_op in enumerate(op.body or []):
                handler = self._dispatch.get(sub_op.kind)
                if handler is None:
                    raise IsaEmissionError(
                        f"no ISA dispatcher for nested op kind {sub_op.kind!r} "
                        f"inside for-loop"
                    )
                ra.push_site(f"for[{loop_var.name}].body[{j}] {sub_op.kind}")
                try:
                    handler(mod, sub_op)
                finally:
                    ra.pop_site()
        finally:
            del self.symbol_table[loop_var]

        # idx += 1: load -> addi -> store. Borrow one GP for the round-
        # trip (auto-spill may briefly displace some other live GP, but
        # gp_loop is pinned so it cannot be the victim).
        inc_gp = ra.allocate_gp(1)[0]
        self.shim.compiler.generated_code += (
            f"; idx {loop_var.name} += 1 (ram[{idx_addr}])\n"
            f"S_LD_INT gp{inc_gp}, gp0, {idx_addr}\n"
            f"S_ADDI_INT gp{inc_gp}, gp{inc_gp}, 1\n"
            f"S_ST_INT gp{inc_gp}, gp0, {idx_addr}\n"
            f"C_LOOP_END gp{gp_loop}\n"
        )
        ra.free_gp([inc_gp])

        ra.unpin_gp(gp_loop)
        ra.free_gp([gp_loop])
        ra.release_idx_slot(idx_addr)


def _check_scope(buf: _hlir.Buffer, expected: str, op_kind: str, role: str) -> None:
    # `global.<phys>` is treated as `<phys>` for ISA-level scope checks —
    # the user-declared global flag changes lane-fusion behaviour but the
    # buffer's physical residency (and therefore the legal operand-scope
    # rules for each instruction) is identical. Keep `buf.scope` as the
    # original string so JSON dumps / debug output retain the global flag.
    if _scope.physical_scope(buf.scope) != expected:
        raise IsaEmissionError(
            f"{op_kind} {role} buffer {buf.name!r} must be in scope {expected!r}, "
            f"got {buf.scope!r}"
        )


__all__ = ["IsaEmitterPass", "IsaEmissionError"]
