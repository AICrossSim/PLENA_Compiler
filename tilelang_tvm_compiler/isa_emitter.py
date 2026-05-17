"""ISAEmitter: turns prepared tile/FP operations into ISA strings.

Owns all `emit_*` methods (HBM/VRAM transfer, BTMM, matmul, FP kernels,
row operations, etc.). Managers and TileTensorProgram hold a reference
to an ISAEmitter and call its methods directly rather than going through
the program object.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# This file is at .../compiler/tilelang_tvm_compiler/isa_emitter.py.
# Walking three parents lands at the project root so that
# `compiler.asm_templates` resolves regardless of CWD/PYTHONPATH.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from compiler.asm_templates import preload_addr_reg_asm, reset_reg_asm

# NOTE on stripped imports:
#   The original runtime version did `from ._types import *` and
#   `from ._helpers import *` to pull in tile/tensor types and small
#   helpers used by the higher-order methods (emit_matmul / emit_fp_kernel
#   / emit_row_operation, etc.).
#
#   For the TVM port we ONLY use the simple methods that take physical
#   addresses and produce ISA: emit_load_tile_from_hbm,
#   emit_store_tile_to_hbm, emit_hbm_tile_to_mram, emit_btmm,
#   emit_btmm_wo, emit_zero_vram_tile, emit_map_v_fp_tile,
#   emit_map_fp_v_tile. None of those reference _types/_helpers symbols,
#   so we drop the deep imports and only pull `Sequence` from typing.
#
#   Calling the heavier methods (emit_matmul / emit_fp_kernel / row ops)
#   will raise NameError until those types are ported. That's intentional:
#   we want the failure to be loud when we try to use a method whose
#   contract we haven't validated yet.


class ISAEmitter:
    """Emit ISA strings for already-prepared tensor/FP operations."""

    def __init__(self, program: "TileTensorProgram") -> None:
        self.program = program

    def _emit_preload_tile_isa(
        self,
        *,
        vlen: int,
        preload_len: int,
        batch: int,
        hidden_size: int,
        act_vram_offset: int,
        alive_registers: List[int],
        activation_offset_reg: int,
        stride_size: Optional[int] = None,
        scale_size: Optional[int] = None,
        hbm_start_offset: int = 0,
        # PLENA TVM extension: when supplied, the offset is COPIED from
        # `hbm_start_offset_reg` instead of loaded as a literal. Used by
        # the slice-aware DMA dispatcher when the slice has a runtime-
        # computed start (e.g. derived from a loop var).
        hbm_start_offset_reg: Optional[int] = None,
    ) -> str:
        generated_code = "; Preload Activation Generation \n"
        a_actual_register = alive_registers[0]
        set_stride_register = alive_registers[1]
        result_register = alive_registers[2]
        outer_loop_register = alive_registers[3]
        inner_loop_register = alive_registers[4]

        stride_len = vlen if stride_size is None else int(stride_size)
        scale_len = hidden_size * batch if scale_size is None else int(scale_size)
        load_amount_per_hidden = math.ceil(hidden_size / vlen)

        generated_code += f"S_ADDI_INT gp{a_actual_register}, gp0, {scale_len} \n"
        generated_code += f"C_SET_SCALE_REG gp{a_actual_register} \n"
        if hbm_start_offset_reg is not None:
            # Dynamic base + static residual: ``a = dyn_reg + static``.
            # The static piece carries per-tile constant offsets (one
            # invocation per inner tile in the multi-tile DMA grid).
            generated_code += (
                f"S_ADDI_INT gp{a_actual_register}, gp{hbm_start_offset_reg}, "
                f"{int(hbm_start_offset)} \n"
            )
        else:
            generated_code += f"S_ADDI_INT gp{a_actual_register}, gp0, {int(hbm_start_offset)} \n"
        generated_code += f"S_ADDI_INT gp{result_register}, gp0, {act_vram_offset} \n"

        if batch == 1:
            elements_per_prefetch = vlen * preload_len
            for _ in range(math.ceil(hidden_size / elements_per_prefetch)):
                generated_code += (
                    f"H_PREFETCH_V gp{result_register}, gp{a_actual_register}, "
                    f"a{activation_offset_reg}, 0, 0, 0 \n"
                )
                generated_code += (
                    f"S_ADDI_INT gp{result_register}, gp{result_register}, {elements_per_prefetch} \n"
                )
                generated_code += (
                    f"S_ADDI_INT gp{a_actual_register}, gp{a_actual_register}, {elements_per_prefetch} \n"
                )
            return generated_code

        generated_code += f"S_ADDI_INT gp{set_stride_register}, gp0, {stride_len} \n"
        generated_code += f"C_SET_STRIDE_REG gp{set_stride_register} \n"
        a_offset_register = set_stride_register
        generated_code += f"C_LOOP_START gp{outer_loop_register}, {load_amount_per_hidden} \n"
        generated_code += f"S_ADDI_INT gp{a_offset_register}, gp{a_actual_register}, 0 \n"
        if batch > preload_len:
            generated_code += f"C_LOOP_START gp{inner_loop_register}, {math.ceil(batch / preload_len)} \n"
        generated_code += f"H_PREFETCH_V gp{result_register}, gp{a_offset_register}, a{activation_offset_reg}, 1, 0 \n"
        generated_code += f"S_ADDI_INT gp{result_register}, gp{result_register}, {vlen * preload_len} \n"
        if batch > preload_len:
            generated_code += (
                f"S_ADDI_INT gp{a_offset_register}, gp{a_offset_register}, {stride_len * preload_len} \n"
            )
            generated_code += f"C_LOOP_END gp{inner_loop_register} \n"
        generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{a_actual_register}, {vlen} \n"
        generated_code += f"C_LOOP_END gp{outer_loop_register} \n"
        return generated_code

    def _emit_store_tile_isa(
        self,
        *,
        vlen: int,
        batch: int,
        hidden_size: int,
        alive_registers: List[int],
        act_vram_offset: int,
        hbm_addr_reg: int,
        stride_size: Optional[int] = None,
        scale_size: Optional[int] = None,
        hbm_start_offset: int = 0,
        store_amount: int = 4,
        # PLENA TVM extension (see emit_preload_tile_isa for rationale).
        hbm_start_offset_reg: Optional[int] = None,
    ) -> str:
        generated_code = "; Store Activation Generation\n"

        hbm_offset_reg = alive_registers[0]
        set_stride_register = alive_registers[1]
        vram_reg = alive_registers[2]
        outer_loop_register = alive_registers[3]
        inner_loop_register = alive_registers[4]

        stride_len = hidden_size if stride_size is None else int(stride_size)
        scale_len = hidden_size * batch if scale_size is None else int(scale_size)
        store_amount_per_hidden = math.ceil(hidden_size / vlen)

        generated_code += f"S_ADDI_INT gp{vram_reg}, gp0, {act_vram_offset}\n"
        generated_code += f"S_ADDI_INT gp{hbm_offset_reg}, gp0, {scale_len}\n"
        generated_code += f"C_SET_SCALE_REG gp{hbm_offset_reg}\n"
        if hbm_start_offset_reg is not None:
            # Dynamic base + static residual (see _emit_preload_tile_isa).
            generated_code += (
                f"S_ADDI_INT gp{hbm_offset_reg}, gp{hbm_start_offset_reg}, "
                f"{int(hbm_start_offset)}\n"
            )
        else:
            generated_code += f"S_ADDI_INT gp{hbm_offset_reg}, gp0, {int(hbm_start_offset)}\n"

        if batch == 1:
            elements_per_store = vlen * store_amount
            for _ in range(math.ceil(hidden_size / elements_per_store)):
                generated_code += f"H_STORE_V gp{vram_reg}, gp{hbm_offset_reg}, a{hbm_addr_reg}, 0, 0\n"
                generated_code += f"S_ADDI_INT gp{vram_reg}, gp{vram_reg}, {elements_per_store}\n"
                generated_code += f"S_ADDI_INT gp{hbm_offset_reg}, gp{hbm_offset_reg}, {elements_per_store}\n"
            return generated_code

        generated_code += f"S_ADDI_INT gp{set_stride_register}, gp0, {stride_len}\n"
        generated_code += f"C_SET_STRIDE_REG gp{set_stride_register}\n"
        hbm_base_reg = set_stride_register
        generated_code += f"C_LOOP_START gp{outer_loop_register}, {store_amount_per_hidden}\n"
        generated_code += f"S_ADDI_INT gp{hbm_base_reg}, gp{hbm_offset_reg}, 0\n"
        if batch > store_amount:
            generated_code += f"C_LOOP_START gp{inner_loop_register}, {math.ceil(batch / store_amount)}\n"
        generated_code += f"H_STORE_V gp{vram_reg}, gp{hbm_base_reg}, a{hbm_addr_reg}, 1, 0\n"
        generated_code += f"S_ADDI_INT gp{vram_reg}, gp{vram_reg}, {vlen * store_amount}\n"
        if batch > store_amount:
            generated_code += f"S_ADDI_INT gp{hbm_base_reg}, gp{hbm_base_reg}, {stride_len * store_amount}\n"
            generated_code += f"C_LOOP_END gp{inner_loop_register}\n"
        generated_code += f"S_ADDI_INT gp{hbm_offset_reg}, gp{hbm_offset_reg}, {vlen}\n"
        generated_code += f"C_LOOP_END gp{outer_loop_register}\n"
        return generated_code

    def emit_hbm_tile_to_mram(
        self,
        *,
        hbm_addr: int,
        mram_addr: int,
        hbm_offset: int = 0,
        hbm_scale: Optional[int] = None,
        hbm_stride: Optional[int] = None,
        # PLENA TVM extension: when set, the offset is sourced from
        # this GP register (caller owns it). `hbm_offset` is ignored.
        hbm_offset_reg: Optional[int] = None,
    ) -> None:
        addr_reg = self.program.compiler.register_allocator.allocate_addr(1)[0]
        gp_addr = self.program.compiler.register_allocator.allocate_gp(2)
        gp_exec = self.program.compiler.register_allocator.allocate_gp(3)
        gp_scale, gp_stride, gp_mram = gp_exec
        scale_val = self.program.tile_elems if hbm_scale is None else int(hbm_scale)
        stride_val = self.program.mlen if hbm_stride is None else int(hbm_stride)

        isa = ""
        isa += preload_addr_reg_asm(
            addr_reg_to_set=[addr_reg],
            available_registers=gp_addr,
            addr_reg_val=[hbm_addr],
        )
        isa += f"S_ADDI_INT gp{gp_scale}, gp0, {scale_val}\n"
        isa += f"C_SET_SCALE_REG gp{gp_scale}\n"
        isa += f"S_ADDI_INT gp{gp_stride}, gp0, {stride_val}\n"
        isa += f"C_SET_STRIDE_REG gp{gp_stride}\n"
        isa += f"S_ADDI_INT gp{gp_mram}, gp0, {mram_addr}\n"
        if hbm_offset_reg is not None:
            # Dynamic base + static residual.
            isa += f"S_ADDI_INT gp{gp_scale}, gp{hbm_offset_reg}, {hbm_offset}\n"
        else:
            isa += f"S_ADDI_INT gp{gp_scale}, gp0, {hbm_offset}\n"
        isa += f"H_PREFETCH_M gp{gp_mram}, gp{gp_scale}, a{addr_reg}, 1, 0\n"
        isa += f"S_ADDI_INT gp{gp_scale}, gp0, {self.program.tile_elems}\n"
        isa += f"C_SET_SCALE_REG gp{gp_scale}\n"
        isa += f"S_ADDI_INT gp{gp_stride}, gp0, {self.program.mlen}\n"
        isa += f"C_SET_STRIDE_REG gp{gp_stride}\n"
        self.program.compiler.generated_code += isa

        self.program.compiler.register_allocator.free_gp(gp_addr)
        self.program.compiler.register_allocator.free_gp(gp_exec)
        self.program.compiler.register_allocator.free_addr([addr_reg])

    def emit_load_tile_from_hbm(
        self,
        *,
        hbm_addr: int,
        vram_addr: int,
        hbm_stride: Optional[int] = None,
        hbm_scale_size: Optional[int] = None,
        hbm_start_offset: int = 0,
        # PLENA TVM extension: when set, the runtime-computed offset
        # comes from this GP register (caller owns it; emitter just
        # reads). `hbm_start_offset` is ignored in that case.
        hbm_start_offset_reg: Optional[int] = None,
    ) -> None:
        ra = self.program.compiler.register_allocator
        addr_reg = ra.allocate_addr(1)[0]
        # Need 1 (addr-init scratch) + 5 (preload scratch). Use
        # spill_borrow so the allocator can move long-lived outer GPs
        # (loop counters / indices) to IntRAM temporarily. The
        # caller-supplied ``hbm_start_offset_reg`` is read inside the
        # emit body, so protect it from being spilled.
        protect = [hbm_start_offset_reg] if hbm_start_offset_reg is not None else []
        borrowed, token = ra.spill_borrow(
            6, compiler=self.program.compiler, protect=protect,
        )
        gp_addr = [borrowed[0]]
        gp_preload = borrowed[1:6]

        isa = ""
        isa += preload_addr_reg_asm(
            addr_reg_to_set=[addr_reg],
            available_registers=gp_addr,
            addr_reg_val=[int(hbm_addr)],
        )
        isa += reset_reg_asm(alive_registers=gp_preload)
        isa += self._emit_preload_tile_isa(
            vlen=self.program.mlen,
            preload_len=self.program.blen,
            batch=self.program.mlen,
            hidden_size=self.program.mlen,
            act_vram_offset=vram_addr,
            alive_registers=gp_preload,
            activation_offset_reg=addr_reg,
            stride_size=self.program.mlen if hbm_stride is None else int(hbm_stride),
            scale_size=self.program.tile_elems if hbm_scale_size is None else int(hbm_scale_size),
            hbm_start_offset=int(hbm_start_offset),
            hbm_start_offset_reg=hbm_start_offset_reg,
        )
        self.program.compiler.generated_code += isa

        ra.spill_return(token, compiler=self.program.compiler)
        ra.free_addr([addr_reg])

    def emit_store_tile_to_hbm(
        self,
        *,
        vram_addr: int,
        hbm_addr: int,
        hbm_stride: Optional[int] = None,
        hbm_scale_size: Optional[int] = None,
        hbm_start_offset: int = 0,
        # PLENA TVM extension; see emit_load_tile_from_hbm.
        hbm_start_offset_reg: Optional[int] = None,
    ) -> None:
        ra = self.program.compiler.register_allocator
        addr_reg = ra.allocate_addr(1)[0]
        protect = [hbm_start_offset_reg] if hbm_start_offset_reg is not None else []
        borrowed, token = ra.spill_borrow(
            6, compiler=self.program.compiler, protect=protect,
        )
        gp_addr = [borrowed[0]]
        gp_store = borrowed[1:6]

        isa = ""
        isa += preload_addr_reg_asm(
            addr_reg_to_set=[addr_reg],
            available_registers=gp_addr,
            addr_reg_val=[int(hbm_addr)],
        )
        isa += self._emit_store_tile_isa(
            vlen=self.program.mlen,
            batch=self.program.mlen,
            hidden_size=self.program.mlen,
            alive_registers=gp_store,
            act_vram_offset=vram_addr,
            hbm_addr_reg=addr_reg,
            stride_size=self.program.mlen if hbm_stride is None else int(hbm_stride),
            scale_size=self.program.tile_elems if hbm_scale_size is None else int(hbm_scale_size),
            hbm_start_offset=int(hbm_start_offset),
            store_amount=self.program.blen,
            hbm_start_offset_reg=hbm_start_offset_reg,
        )
        self.program.compiler.generated_code += isa

        ra.spill_return(token, compiler=self.program.compiler)
        ra.free_addr([addr_reg])

    def emit_zero_vram_tile(self, vram_addr: int, num_rows: Optional[int] = None) -> None:
        # `num_rows` is how many MLEN-wide rows to zero. Defaults to MLEN
        # for legacy callers that always zero a full MLEN*MLEN tile.
        # Buffers smaller than that (e.g. a (1, MLEN) accumulator) MUST
        # pass the actual row count or the loop will write past the
        # buffer's end into adjacent VRAM (silent corruption of whatever
        # follows in the address map).
        loop_count = self.program.mlen if num_rows is None else int(num_rows)
        if loop_count < 1:
            raise ValueError(f"num_rows must be >= 1, got {loop_count}")
        gp_regs = self.program.compiler.register_allocator.allocate_gp(2)
        gp, gp_loop = gp_regs
        lines = [f"; zero tile vram[{vram_addr}] rows={loop_count}"]
        lines.append(f"S_ADDI_INT gp{gp}, gp0, {vram_addr}")
        if loop_count == 1:
            lines.append(f"V_MUL_VF gp{gp}, gp{gp}, f0, 0")
        else:
            lines.append(f"C_LOOP_START gp{gp_loop}, {loop_count}")
            lines.append(f"V_MUL_VF gp{gp}, gp{gp}, f0, 0")
            lines.append(f"S_ADDI_INT gp{gp}, gp{gp}, {self.program.mlen}")
            lines.append(f"C_LOOP_END gp{gp_loop}")
        self.program.compiler.register_allocator.free_gp(gp_regs)
        self.program.compiler.generated_code += "\n".join(lines) + "\n"

    def emit_map_v_fp_tile(
        self,
        *,
        vram_addr: int,
        fpram_addr: int,
        row_count: int,
        row_width: int,
        task_id: str = "map_v_fp_tile",
    ) -> None:
        if row_count <= 0 or row_width <= 0:
            raise ValueError(f"emit_map_v_fp_tile expects positive row_count/row_width, got {row_count}/{row_width}")
        if row_width != self.program.mlen:
            raise ValueError(
                f"emit_map_v_fp_tile currently requires row_width == mlen == {self.program.mlen}, got {row_width}"
            )
        gp_regs = self.program.compiler.register_allocator.allocate_gp(3)
        gp_dst, gp_src, gp_loop = gp_regs
        lines = [f"; map fp tile task {task_id} fpram[{fpram_addr}] -> vram[{vram_addr}]"]
        lines.append(f"S_ADDI_INT gp{gp_dst}, gp0, {vram_addr}")
        lines.append(f"S_ADDI_INT gp{gp_src}, gp0, {fpram_addr}")
        lines.append(f"C_LOOP_START gp{gp_loop}, {row_count}")
        lines.append(f"S_MAP_V_FP gp{gp_dst}, gp{gp_src}, 0")
        lines.append(f"S_ADDI_INT gp{gp_dst}, gp{gp_dst}, {row_width}")
        lines.append(f"S_ADDI_INT gp{gp_src}, gp{gp_src}, {row_width}")
        lines.append(f"C_LOOP_END gp{gp_loop}")
        self.program.compiler.register_allocator.free_gp(gp_regs)
        self.program.compiler.generated_code += "\n".join(lines) + "\n"

    def emit_map_fp_v_tile(
        self,
        *,
        fpram_addr: int,
        vram_addr: int,
        row_count: int,
        row_width: int,
        task_id: str = "map_fp_v_tile",
    ) -> None:
        if row_count <= 0 or row_width <= 0:
            raise ValueError(f"emit_map_fp_v_tile expects positive row_count/row_width, got {row_count}/{row_width}")
        if row_width != self.program.mlen:
            raise ValueError(
                f"emit_map_fp_v_tile currently requires row_width == mlen == {self.program.mlen}, got {row_width}"
            )
        gp_regs = self.program.compiler.register_allocator.allocate_gp(3)
        gp_dst, gp_src, gp_loop = gp_regs
        lines = [f"; map fp tile task {task_id} vram[{vram_addr}] -> fpram[{fpram_addr}]"]
        lines.append(f"S_ADDI_INT gp{gp_dst}, gp0, {fpram_addr}")
        lines.append(f"S_ADDI_INT gp{gp_src}, gp0, {vram_addr}")
        lines.append(f"C_LOOP_START gp{gp_loop}, {row_count}")
        lines.append(f"S_MAP_FP_V gp{gp_dst}, gp{gp_src}, 0")
        lines.append(f"S_ADDI_INT gp{gp_dst}, gp{gp_dst}, {row_width}")
        lines.append(f"S_ADDI_INT gp{gp_src}, gp{gp_src}, {row_width}")
        lines.append(f"C_LOOP_END gp{gp_loop}")
        self.program.compiler.register_allocator.free_gp(gp_regs)
        self.program.compiler.generated_code += "\n".join(lines) + "\n"

    def emit_btmm(
        self,
        *,
        lhs_packed_vram_addr: int,
        rhs_mram_addr: int,
        task_id: str = "btmm",
    ) -> None:
        gp_regs = self.program.compiler.register_allocator.allocate_gp(2)
        gp_mram_base, gp_lhs_base = gp_regs
        lines = [
            (
                f"; btmm task {task_id} lhs_packed=vram[{lhs_packed_vram_addr}] "
                f"rhs_mram={rhs_mram_addr} lanes={self.program.btmm_lane_count} head_width={self.program.btmm_hlen}"
            ),
            f"S_ADDI_INT gp{gp_mram_base}, gp0, {rhs_mram_addr}",
            f"S_ADDI_INT gp{gp_lhs_base}, gp0, {lhs_packed_vram_addr}",
            f"M_BTMM gp0, gp{gp_mram_base}, gp{gp_lhs_base}",
        ]
        self.program.compiler.generated_code += "\n".join(lines) + "\n"
        self.program.compiler.register_allocator.free_gp(gp_regs)

    def emit_btmm_wo(
        self,
        *,
        base_addr: int,
        tile_count: int,
        task_id: str = "btmm_wo",
    ) -> None:
        gp_out = self.program.compiler.register_allocator.allocate_gp(1)[0]
        lines = [
            (
                f"; btmm write-only task {task_id} out=vram[{base_addr}] "
                f"tiles={tile_count} lanes={self.program.btmm_lane_count} head_width={self.program.btmm_hlen}"
            ),
            f"S_ADDI_INT gp{gp_out}, gp0, {base_addr}",
            f"M_BMM_WO gp{gp_out}, 0",
        ]
        self.program.compiler.generated_code += "\n".join(lines) + "\n"
        self.program.compiler.register_allocator.free_gp([gp_out])

    def emit_mv(
        self,
        *,
        lhs_vram_addr,
        rhs_mram_addr,
        dst_vram_addr,
        task_id: str = "mv",
        lhs_offset_reg=None,
        rhs_offset_reg=None,
        dst_offset_reg=None,
        n: int | None = None,
    ) -> None:
        """Per-head M_MV + M_MV_WO (single-lane matrix-vector).

        Each (M_MV, M_MV_WO) pair processes BLEN-wide column blocks: M_MV
        accumulates ``vec[mlen] @ mat[mlen, blen]`` into the systolic array
        first row, M_MV_WO drains those blen elements to VRAM. To cover
        ``n`` columns total (defaults to ``btmm_hlen`` -- one full head),
        we loop ``n / blen`` times, advancing both the matrix column
        offset and the destination offset by blen each iteration.

        Mirrors emit_matmul's blen-loop (used by plena.matmul / M_MM) but
        with the LHS being a single row and the writeback being M_MV_WO.
        """
        if n is None:
            n = int(self.program.btmm_hlen)
        blen = int(self.program.blen)
        if n % blen != 0:
            raise ValueError(
                f"emit_mv: column extent n={n} must be a multiple of blen={blen}"
            )
        tiles = n // blen

        gp_regs = self.program.compiler.register_allocator.allocate_gp(3)
        gp_v, gp_m, gp_o = gp_regs
        lines = [
            (
                f"; mv task {task_id} v=vram[{lhs_vram_addr}]"
                f" m=mram[{rhs_mram_addr}] dst=vram[{dst_vram_addr}]"
                f" tiles={tiles} blen={blen}"
            ),
            # Set up vector base (lhs).
            f"S_ADDI_INT gp{gp_v}, gp0, {lhs_vram_addr}",
        ]
        if lhs_offset_reg is not None:
            lines.append(f"S_ADD_INT gp{gp_v}, gp{gp_v}, gp{lhs_offset_reg}")

        # Each iteration walks the matrix and dst by blen elements. Set
        # up the per-iteration starting m_addr / dst_addr (head base) once,
        # then bump them by blen inside the loop body.
        lines.append(f"S_ADDI_INT gp{gp_m}, gp0, {rhs_mram_addr}")
        if rhs_offset_reg is not None:
            lines.append(f"S_ADD_INT gp{gp_m}, gp{gp_m}, gp{rhs_offset_reg}")
        lines.append(f"S_ADDI_INT gp{gp_o}, gp0, {dst_vram_addr}")
        if dst_offset_reg is not None:
            lines.append(f"S_ADD_INT gp{gp_o}, gp{gp_o}, gp{dst_offset_reg}")

        for t in range(tiles):
            lines.append(f"M_MV gp0, gp{gp_m}, gp{gp_v}")
            lines.append(f"M_MV_WO gp{gp_o}, 0")
            if t < tiles - 1:
                lines.append(f"S_ADDI_INT gp{gp_m}, gp{gp_m}, {blen}")
                lines.append(f"S_ADDI_INT gp{gp_o}, gp{gp_o}, {blen}")

        self.program.compiler.generated_code += "\n".join(lines) + "\n"
        self.program.compiler.register_allocator.free_gp(gp_regs)

    def emit_btmv(
        self,
        *,
        lhs_packed_vram_addr: int,
        rhs_mram_addr: int,
        task_id: str = "btmv",
    ) -> None:
        """Lane-fused vector × matrix^T (M_BTMV).

        Mirrors emit_btmm — same MRAM/VRAM register setup, same operand
        order, just the M_BTMM opcode swapped for M_BTMV. The hardware
        consumes a 1-row vector LHS instead of an mlen-row matrix LHS.
        """
        gp_regs = self.program.compiler.register_allocator.allocate_gp(2)
        gp_mram_base, gp_lhs_base = gp_regs
        lines = [
            (
                f"; btmv task {task_id} lhs_packed=vram[{lhs_packed_vram_addr}] "
                f"rhs_mram={rhs_mram_addr} lanes={self.program.btmm_lane_count} head_width={self.program.btmm_hlen}"
            ),
            f"S_ADDI_INT gp{gp_mram_base}, gp0, {rhs_mram_addr}",
            f"S_ADDI_INT gp{gp_lhs_base}, gp0, {lhs_packed_vram_addr}",
            f"M_BTMV gp0, gp{gp_mram_base}, gp{gp_lhs_base}",
        ]
        self.program.compiler.generated_code += "\n".join(lines) + "\n"
        self.program.compiler.register_allocator.free_gp(gp_regs)

    def emit_bmv_wo(
        self,
        *,
        base_addr: int,
        task_id: str = "bmv_wo",
    ) -> None:
        """Drain accumulator from systolic-array first row to VRAM
        (M_BMV_WO). Writes lane_count MLEN-wide rows starting at base_addr.
        """
        gp_out = self.program.compiler.register_allocator.allocate_gp(1)[0]
        lines = [
            (
                f"; bmv write-only task {task_id} out=vram[{base_addr}] "
                f"lanes={self.program.btmm_lane_count} head_width={self.program.btmm_hlen}"
            ),
            f"S_ADDI_INT gp{gp_out}, gp0, {base_addr}",
            f"M_BMV_WO gp{gp_out}, 0",
        ]
        self.program.compiler.generated_code += "\n".join(lines) + "\n"
        self.program.compiler.register_allocator.free_gp([gp_out])

    def emit_matmul(
        self,
        *,
        lhs_vram_addrs: Sequence[int],
        rhs_mram_addrs: Sequence[int],
        dst_vram_addr: int,
        task_id: str = "matmul",
        zero_dst: bool = False,
    ) -> None:
        if len(lhs_vram_addrs) != len(rhs_mram_addrs):
            raise ValueError("lhs_vram_addrs and rhs_mram_addrs must have equal lengths")
        if zero_dst:
            self.emit_zero_vram_tile(dst_vram_addr)

        gp_regs = self.program.compiler.register_allocator.allocate_gp(5)
        gp_act, gp_mat, gp_out, gp_stride, gp_loop = gp_regs
        tiles_per_mlen = self.program.mlen // self.program.blen
        lines = [f"; matmul task {task_id}"]
        lines.append(f"S_ADDI_INT gp{gp_stride}, gp0, 1")
        lhs_prog = self.program._arith_progression([int(addr) for addr in lhs_vram_addrs])
        rhs_prog = self.program._arith_progression([int(addr) for addr in rhs_mram_addrs])

        for oc in range(tiles_per_mlen):
            for orow in range(tiles_per_mlen):
                if lhs_prog is not None and rhs_prog is not None:
                    lhs_start, pair_count, lhs_step = lhs_prog
                    rhs_start, _, rhs_step = rhs_prog
                    act_addr = lhs_start + orow * self.program.blen * self.program.mlen
                    mat_addr = rhs_start + oc * self.program.blen
                    lines.append(f"S_ADDI_INT gp{gp_act}, gp0, {act_addr}")
                    lines.append(f"S_ADDI_INT gp{gp_mat}, gp0, {mat_addr}")
                    lines.append(f"C_LOOP_START gp{gp_loop}, {pair_count}")
                    lines.append(f"M_MM 0, gp{gp_mat}, gp{gp_act}")
                    lines.append(f"S_ADDI_INT gp{gp_act}, gp{gp_act}, {lhs_step}")
                    lines.append(f"S_ADDI_INT gp{gp_mat}, gp{gp_mat}, {rhs_step}")
                    lines.append(f"C_LOOP_END gp{gp_loop}")
                else:
                    for lhs_addr, rhs_addr in zip(lhs_vram_addrs, rhs_mram_addrs):
                        act_addr = lhs_addr + orow * self.program.blen * self.program.mlen
                        mat_addr = rhs_addr + oc * self.program.blen
                        lines.append(f"S_ADDI_INT gp{gp_act}, gp0, {act_addr}")
                        lines.append(f"S_ADDI_INT gp{gp_mat}, gp0, {mat_addr}")
                        lines.append(f"M_MM 0, gp{gp_mat}, gp{gp_act}")
                out_addr = dst_vram_addr + orow * self.program.blen * self.program.mlen + oc * self.program.blen
                lines.append(f"S_ADDI_INT gp{gp_out}, gp0, {out_addr}")
                lines.append(f"M_MM_WO gp{gp_out}, gp0, 0")

        self.program.compiler.register_allocator.free_gp(gp_regs)
        self.program.compiler.generated_code += "\n".join(lines) + "\n"

    def emit_matmul_single_tile_hwloop(
        self,
        *,
        lhs_vram_addr: int,
        rhs_mram_addr: int,
        dst_vram_addr: int,
        task_id: str = "matmul_single_hwloop",
    ) -> None:
        """Single-tile (mlen*mlen) MM emitted with hardware loops over the
        blen-tiled output (oc, orow), instead of the Python-unrolled form
        used by `emit_matmul`.

        Generates O((mlen/blen)^2) M_MM/M_MM_WO pairs *dynamically* but
        only ~15 lines of *static* ISA — vs. ~7*256 ≈ 1800 lines for the
        unrolled `emit_matmul` on a 64/4 (mlen/blen) configuration.
        Identical dynamic instruction count, so loop-instruction caps
        in the emulator behave the same.

        Loop structure (mirrors sub_matrix_manager.vram_sub_projection_asm
        with num_hidden_blocks == 1, so the innermost accumulation loop
        collapses to a single M_MM):

            for oc in tiles_per_mlen:                # output blen-cols
              for orow in tiles_per_mlen:            # output blen-rows
                M_MM 0, gp_mat, gp_act
                M_MM_WO gp_result, gp0, 0
        """
        ra = self.program.compiler.register_allocator
        # Single allocate_gp call -> single free_gp at the end. The loop
        # counters (gp_loop_outer, gp_loop_middle) stay marked in-use for
        # the entire emit, so nested ISAEmitter calls (none today, but
        # future-proof against a body sub-emit) cannot collide with them.
        gp_regs = ra.allocate_gp(6)
        (gp_act_row_base, gp_mat_col_base, gp_result_col_base, gp_result,
         gp_loop_outer, gp_loop_middle) = gp_regs

        tiles_per_mlen = self.program.mlen // self.program.blen
        output_row_stride = self.program.blen * self.program.mlen
        blen = self.program.blen

        # Single accumulation pair -> drop the inner accum C_LOOP and the
        # gp_act/gp_mat copies that the multi-pair runtime version needs.
        # M_MM reads its operand regs and does not mutate them, so we
        # pass gp_act_row_base / gp_mat_col_base directly. gp_act_row_base
        # is advanced inside the orow loop (output_row_stride per iter)
        # and re-loaded with lhs_vram_addr at the top of each oc iter.
        lines = [
            f"; matmul (single-tile, hw-loop) task {task_id} "
            f"lhs=vram[{lhs_vram_addr}] rhs=mram[{rhs_mram_addr}] "
            f"dst=vram[{dst_vram_addr}]  "
            f"regs: act_row_base=gp{gp_act_row_base} "
            f"mat_col_base=gp{gp_mat_col_base} "
            f"result_col_base=gp{gp_result_col_base} "
            f"result=gp{gp_result} "
            f"hw_loops=gp{gp_loop_outer}/gp{gp_loop_middle}",
            f"S_ADDI_INT gp{gp_mat_col_base}, gp0, {rhs_mram_addr}",
            f"S_ADDI_INT gp{gp_result_col_base}, gp0, {dst_vram_addr}",
            f"C_LOOP_START gp{gp_loop_outer}, {tiles_per_mlen}",
            f"S_ADDI_INT gp{gp_act_row_base}, gp0, {lhs_vram_addr}",
            f"S_ADDI_INT gp{gp_result}, gp{gp_result_col_base}, 0",
            f"C_LOOP_START gp{gp_loop_middle}, {tiles_per_mlen}",
            f"M_MM 0, gp{gp_mat_col_base}, gp{gp_act_row_base}",
            f"M_MM_WO gp{gp_result}, gp0, 0",
            f"S_ADDI_INT gp{gp_act_row_base}, gp{gp_act_row_base}, {output_row_stride}",
            f"S_ADDI_INT gp{gp_result}, gp{gp_result}, {output_row_stride}",
            f"C_LOOP_END gp{gp_loop_middle}",
            f"S_ADDI_INT gp{gp_mat_col_base}, gp{gp_mat_col_base}, {blen}",
            f"S_ADDI_INT gp{gp_result_col_base}, gp{gp_result_col_base}, {blen}",
            f"C_LOOP_END gp{gp_loop_outer}",
        ]
        self.program.compiler.generated_code += "\n".join(lines) + "\n"
        ra.free_gp(gp_regs)

    def emit_slot_matmul(
        self,
        *,
        lhs_vram_addr: int,
        lhs_vram_addr_reg: Optional[int] = None,
        rhs_mram_addr: int,
        rhs_col_offset: int = 0,
        rhs_col_offset_reg: Optional[int] = None,
        dst_vram_addr: int,
        dst_col_offset: int = 0,
        dst_col_offset_reg: Optional[int] = None,
        col_count: int,
        task_id: str = "slot_matmul",
        zero_dst: bool = False,
    ) -> None:
        if col_count <= 0:
            raise ValueError("emit_slot_matmul requires one positive col_count")
        if col_count % self.program.blen != 0:
            raise ValueError(
                f"emit_slot_matmul requires col_count divisible by blen={self.program.blen}, got {col_count}"
            )
        if zero_dst:
            self.emit_zero_vram_tile(dst_vram_addr)

        gp_regs = self.program.compiler.register_allocator.allocate_gp(5)
        gp_act, gp_mat, gp_out, gp_stride, gp_loop = gp_regs
        tiles_per_mlen = self.program.mlen // self.program.blen
        tiles_per_slot = col_count // self.program.blen
        lines = [
            f"; slot matmul task {task_id}"
            f" rhs_col_offset="
            f"{'gp' + str(rhs_col_offset_reg) if rhs_col_offset_reg is not None else rhs_col_offset}"
            f" dst_col_offset="
            f"{'gp' + str(dst_col_offset_reg) if dst_col_offset_reg is not None else dst_col_offset}"
        ]
        lines.append(f"S_ADDI_INT gp{gp_stride}, gp0, 1")

        for oc in range(tiles_per_slot):
            if lhs_vram_addr_reg is not None:
                # base = register-held lhs address (already includes any
                # dynamic lhs_row_offset). Reset gp_act to base each oc tile.
                lines.append(f"S_ADDI_INT gp{gp_act}, gp{lhs_vram_addr_reg}, 0")
            else:
                act_addr = lhs_vram_addr
                lines.append(f"S_ADDI_INT gp{gp_act}, gp0, {act_addr}")
            if rhs_col_offset_reg is not None:
                lines.append(f"S_ADDI_INT gp{gp_mat}, gp{rhs_col_offset_reg}, {rhs_mram_addr + oc * self.program.blen}")
            else:
                mat_addr = rhs_mram_addr + rhs_col_offset + oc * self.program.blen
                lines.append(f"S_ADDI_INT gp{gp_mat}, gp0, {mat_addr}")
            if dst_col_offset_reg is not None:
                lines.append(f"S_ADDI_INT gp{gp_out}, gp{dst_col_offset_reg}, {dst_vram_addr + oc * self.program.blen}")
            else:
                out_addr = dst_vram_addr + dst_col_offset + oc * self.program.blen
                lines.append(f"S_ADDI_INT gp{gp_out}, gp0, {out_addr}")
            lines.append(f"C_LOOP_START gp{gp_loop}, {tiles_per_mlen}")
            lines.append(f"M_MM 0, gp{gp_mat}, gp{gp_act}")
            lines.append(f"M_MM_WO gp{gp_out}, gp0, 0")
            lines.append(f"S_ADDI_INT gp{gp_act}, gp{gp_act}, {self.program.blen * self.program.mlen}")
            lines.append(f"S_ADDI_INT gp{gp_out}, gp{gp_out}, {self.program.blen * self.program.mlen}")
            lines.append(f"C_LOOP_END gp{gp_loop}")

        self.program.compiler.register_allocator.free_gp(gp_regs)
        self.program.compiler.generated_code += "\n".join(lines) + "\n"

    def emit_matmul_narrow_tile_hwloop(
        self,
        *,
        lhs_vram_addr: int,
        rhs_mram_addr: int,
        dst_vram_addr: int,
        hlen: int,
        rhs_col_offset: int = 0,
        dst_col_offset: int = 0,
        dst_row_stride: Optional[int] = None,
        task_id: str = "matmul_narrow_hwloop",
        zero_dst: bool = False,
    ) -> None:
        """Emit `mlen x mlen @ mlen x hlen` via the regular M_MM path."""
        if hlen <= 0:
            raise ValueError("emit_matmul_narrow_tile_hwloop requires positive hlen")
        if hlen > self.program.mlen:
            raise ValueError(
                f"emit_matmul_narrow_tile_hwloop requires hlen <= mlen={self.program.mlen}, got {hlen}"
            )
        if hlen % self.program.blen != 0:
            raise ValueError(
                f"emit_matmul_narrow_tile_hwloop requires hlen divisible by blen={self.program.blen}, got {hlen}"
            )
        if dst_row_stride is None:
            dst_row_stride = int(hlen)
        if dst_row_stride < hlen:
            raise ValueError(
                f"emit_matmul_narrow_tile_hwloop requires dst_row_stride >= hlen ({hlen}), got {dst_row_stride}"
            )
        if zero_dst:
            self.emit_zero_vram_tile(dst_vram_addr)

        ra = self.program.compiler.register_allocator
        gp_regs = ra.allocate_gp(5)
        gp_act, gp_mat, gp_out, gp_stride, gp_loop = gp_regs
        tiles_per_mlen = self.program.mlen // self.program.blen
        tiles_per_slot = hlen // self.program.blen
        output_row_stride = self.program.blen * int(dst_row_stride)
        lines = [
            f"; narrow matmul task {task_id} lhs=vram[{lhs_vram_addr}] "
            f"rhs=mram[{rhs_mram_addr}] rhs_col_offset={rhs_col_offset} "
            f"dst=vram[{dst_vram_addr}] dst_col_offset={dst_col_offset} "
            f"hlen={hlen} dst_row_stride={dst_row_stride}"
        ]
        lines.append(f"S_ADDI_INT gp{gp_stride}, gp0, 1")

        for oc in range(tiles_per_slot):
            act_addr = lhs_vram_addr
            mat_addr = rhs_mram_addr + rhs_col_offset + oc * self.program.blen
            out_addr = dst_vram_addr + dst_col_offset + oc * self.program.blen
            lines.append(f"S_ADDI_INT gp{gp_act}, gp0, {act_addr}")
            lines.append(f"S_ADDI_INT gp{gp_mat}, gp0, {mat_addr}")
            lines.append(f"S_ADDI_INT gp{gp_out}, gp0, {out_addr}")
            lines.append(f"C_LOOP_START gp{gp_loop}, {tiles_per_mlen}")
            lines.append(f"M_MM 0, gp{gp_mat}, gp{gp_act}")
            lines.append(f"M_MM_WO gp{gp_out}, gp0, 0")
            lines.append(f"S_ADDI_INT gp{gp_act}, gp{gp_act}, {self.program.blen * self.program.mlen}")
            lines.append(f"S_ADDI_INT gp{gp_out}, gp{gp_out}, {output_row_stride}")
            lines.append(f"C_LOOP_END gp{gp_loop}")

        self.program.compiler.generated_code += "\n".join(lines) + "\n"
        ra.free_gp(gp_regs)

    def emit_matmul_general(
        self,
        *,
        M_tiles: int,
        K_tiles: int,
        N: int,
        lhs_vram_base: int,
        lhs_offset: int = 0,
        lhs_offset_reg: Optional[int] = None,
        lhs_m_tile_stride: Optional[int] = None,
        lhs_k_tile_stride: Optional[int] = None,
        rhs_mram_base: int,
        rhs_offset: int = 0,
        rhs_offset_reg: Optional[int] = None,
        rhs_k_tile_stride: Optional[int] = None,
        rhs_n_mlen_tile_stride: Optional[int] = None,
        dst_vram_base: int,
        dst_offset: int = 0,
        dst_offset_reg: Optional[int] = None,
        dst_m_tile_stride: Optional[int] = None,
        dst_row_stride: Optional[int] = None,
        task_id: str = "matmul",
        scratch_regs: Optional[List[int]] = None,
        transpose_b: bool = False,
        unroll_loops: bool = False,
    ) -> None:
        """Unified `(M, K) @ (K, N) -> (M, N)` matmul.

        K is folded into the systolic-array accumulator: each output
        BLEN×BLEN sub-tile is produced by K_tiles `M_MM` issuances followed
        by one `M_MM_WO`. No software scratch / v_add is needed for K
        accumulation.

        When ``transpose_b=True``, B is expected in MRAM as ``(N, K)``
        row-major (matches the nn.Linear weight convention). The inner
        op switches from ``M_MM`` to ``M_TMM`` which transposes the
        (mlen, mlen) MRAM tile on the fly, and the per-N column
        sub-tile step changes from ``blen`` (columns of (K, N)) to
        ``blen * mlen`` (rows of (N, K)) — sim enforces the latter via
        ``mat_offset.assert_multiple_of(mlen)`` inside ``M_TMM``.

        Shape constraints:
          M : multiple of mlen,  M_tiles = M / mlen
          K : multiple of mlen,  K_tiles = K / mlen
          N : multiple of hlen   (hlen = btmm_hlen on the shim)

        N may exceed mlen — the emitter walks B in (K_tiles × N_mlen_tiles)
        mlen-wide blocks, where N_mlen_tiles = ceil(N / mlen). The trailing
        N-mlen block is allowed to carry < mlen valid columns (down to the
        nearest hlen boundary).

        Layout assumptions (defaults match a packed tile-grid layout):
          A in VRAM    : (M_tiles × K_tiles) grid of (mlen, mlen) tiles,
                         packed:    A_tile(m, k) = base + m*K_tiles*mlen² + k*mlen².
          B in MRAM    : (K_tiles × N_mlen_tiles) grid of (mlen, mlen) tiles,
                         packed K-major:
                         B_tile(k, nm) = base + k*N_mlen_tiles*mlen² + nm*mlen².
          C in VRAM    : row-major (M, N) with `dst_row_stride` elements
                         between consecutive output rows (defaults to N).
                         M-tile spacing defaults to `mlen * dst_row_stride`.

        Offsets:
          `lhs_offset` / `rhs_offset` / `dst_offset` are static element
          offsets added to the corresponding base addresses. Useful when
          A/B/C are sub-regions of larger packed buffers (mm_slot pattern).
          For each side, pass the ``*_offset_reg`` form instead to use a
          dynamic (PrimExpr-derived) offset already materialised to a gp
          register; when ``*_offset_reg`` is set, the matching static
          ``*_offset`` is ignored and the per-iteration pointer is formed
          via ``S_ADDI_INT gp_dst, gp{*_offset_reg}, <static-residual>``.
        """
        mlen = self.program.mlen
        blen = self.program.blen
        hlen = int(self.program.btmm_hlen)
        if M_tiles <= 0 or K_tiles <= 0 or N <= 0:
            raise ValueError(f"M_tiles, K_tiles, N must be positive; got {M_tiles}, {K_tiles}, {N}")
        if N % hlen != 0:
            raise ValueError(f"N must be divisible by hlen={hlen}; got N={N}")

        N_mlen_tiles = (N + mlen - 1) // mlen

        if lhs_k_tile_stride is None:
            lhs_k_tile_stride = mlen * mlen
        if lhs_m_tile_stride is None:
            lhs_m_tile_stride = K_tiles * mlen * mlen
        # When ``transpose_b`` is set, B is laid out as ``(N, K)`` —
        # tiles are packed N-major (one full K-row of mlen-tiles per
        # N-mlen step). When unset, B is ``(K, N)`` and tiles are
        # packed K-major. The inner-tile layout stays row-major in both
        # cases; M_TMM transposes the (mlen, mlen) tile on the fly.
        if rhs_n_mlen_tile_stride is None:
            if transpose_b:
                rhs_n_mlen_tile_stride = K_tiles * mlen * mlen
            else:
                rhs_n_mlen_tile_stride = mlen * mlen
        if rhs_k_tile_stride is None:
            if transpose_b:
                rhs_k_tile_stride = mlen * mlen
            else:
                rhs_k_tile_stride = N_mlen_tiles * mlen * mlen
        if dst_row_stride is None:
            dst_row_stride = N
        if dst_m_tile_stride is None:
            dst_m_tile_stride = mlen * int(dst_row_stride)

        tiles_per_mlen = mlen // blen
        a_orow_step = blen * mlen
        # M_MM_WO writes 4 rows (i=0..blen-1) at vram[vec_base + i*mlen]
        # — physical row stride is always mlen, regardless of how dense
        # the dst's logical N maps inside each mlen-row (e.g. N=16 with
        # 4 lanes packed: 4 cols per lane stored together inside one
        # mlen-row). The outer orow advance must therefore step by
        # ``blen * mlen`` to jump 4 physical mlen-rows; the previous
        # ``blen * dst_row_stride`` formula collapsed to a 1-mlen-row
        # step for narrow N (≤ mlen) and made the kernel re-write the
        # same mlen-rows repeatedly.
        c_orow_step = blen * mlen

        ra = self.program.compiler.register_allocator
        # Caller can pre-allocate the 7 scratch GPs (and pin them) so they're
        # disjoint from any offset registers it materialised. When caller
        # passes them in we don't free here either — caller owns the lifetime.
        if scratch_regs is not None:
            if len(scratch_regs) != 7:
                raise ValueError(
                    f"emit_matmul_general expects 7 scratch_regs, got {len(scratch_regs)}"
                )
            gp_regs = list(scratch_regs)
            caller_owns_scratch = True
        else:
            gp_regs = ra.allocate_gp(7)
            caller_owns_scratch = False
        (gp_act_orow, gp_out_orow, gp_act, gp_mat, gp_out,
         gp_loop_orow, gp_loop_k) = gp_regs

        lines = [
            f"; matmul (general) task {task_id} "
            f"M={M_tiles * mlen} K={K_tiles * mlen} N={N}  "
            f"(M_tiles={M_tiles} K_tiles={K_tiles} N_mlen_tiles={N_mlen_tiles})"
        ]

        for m in range(M_tiles):
            # Static-residual addresses (everything that doesn't depend on
            # a dynamic offset register). When the matching `*_offset_reg`
            # is set we issue `S_ADDI_INT gp_X, gp{reg}, <static>` to fold
            # the runtime offset in; otherwise we just load the absolute
            # static value.
            lhs_static_full = int(lhs_vram_base) + int(lhs_offset) + m * int(lhs_m_tile_stride)
            lhs_static_dyn  = int(lhs_vram_base) + m * int(lhs_m_tile_stride)
            dst_m_base_static_full = int(dst_vram_base) + int(dst_offset) + m * int(dst_m_tile_stride)
            dst_m_base_static_dyn  = int(dst_vram_base) + m * int(dst_m_tile_stride)
            for n_mlen in range(N_mlen_tiles):
                rhs_n_mlen_static_full = (
                    int(rhs_mram_base) + int(rhs_offset)
                    + n_mlen * int(rhs_n_mlen_tile_stride)
                )
                rhs_n_mlen_static_dyn = (
                    int(rhs_mram_base) + n_mlen * int(rhs_n_mlen_tile_stride)
                )
                cols_here = min(mlen, N - n_mlen * mlen)
                tiles_per_n_mlen = cols_here // blen
                # Per-oc B offset within the current (mlen, mlen) tile:
                #   M_MM picks (mlen, blen) columns at byte stride blen.
                #   M_TMM picks blen ROWS (transposed -> the same blen
                #   columns of B^T) at byte stride mlen*blen — sim asserts
                #   ``mat_offset / mlen ∈ [0, mlen)`` after the inner
                #   ``assert_multiple_of(mlen)``, so the per-row scale is
                #   exactly mlen.
                oc_b_step = blen * mlen if transpose_b else blen
                # Matmul opcode: M_TMM transposes the (mlen, mlen) MRAM
                # tile on the fly; its (rs1, rs2) order is also swapped
                # vs M_MM (rs1 = vram_lhs, rs2 = mram_rhs).
                mm_opcode = "M_TMM" if transpose_b else "M_MM"
                for oc in range(tiles_per_n_mlen):
                    dst_col = n_mlen * mlen + oc * blen
                    if lhs_offset_reg is not None:
                        lines.append(
                            f"S_ADDI_INT gp{gp_act_orow}, gp{lhs_offset_reg}, "
                            f"{lhs_static_dyn}"
                        )
                    else:
                        lines.append(f"S_ADDI_INT gp{gp_act_orow}, gp0, {lhs_static_full}")
                    if dst_offset_reg is not None:
                        lines.append(
                            f"S_ADDI_INT gp{gp_out_orow}, gp{dst_offset_reg}, "
                            f"{dst_m_base_static_dyn + dst_col}"
                        )
                    else:
                        lines.append(
                            f"S_ADDI_INT gp{gp_out_orow}, gp0, "
                            f"{dst_m_base_static_full + dst_col}"
                        )

                    if unroll_loops:
                        # Fully unrolled body: emit ``tiles_per_mlen``
                        # copies of the (K_tiles M_MM + M_MM_WO) cell,
                        # each with its own static lhs_act / mat base.
                        # No C_LOOP nesting — diagnostic mode for the
                        # debugger to read straight through.
                        for orow in range(tiles_per_mlen):
                            act_static = lhs_static_full + orow * a_orow_step
                            dst_static = dst_m_base_static_full + dst_col + orow * c_orow_step
                            if lhs_offset_reg is not None:
                                act_dyn = lhs_static_dyn + orow * a_orow_step
                                lines.append(
                                    f"S_ADDI_INT gp{gp_act}, gp{lhs_offset_reg}, "
                                    f"{act_dyn}"
                                )
                            else:
                                lines.append(
                                    f"S_ADDI_INT gp{gp_act}, gp0, {act_static}"
                                )
                            for k in range(K_tiles):
                                act_k = act_static + k * int(lhs_k_tile_stride) - (orow * a_orow_step + lhs_static_full - lhs_static_full)
                                # Recompute act/mat per k explicitly so
                                # there is no incremental S_ADDI between
                                # M_MMs (matches unroll-only style).
                                if k > 0:
                                    if lhs_offset_reg is not None:
                                        lines.append(
                                            f"S_ADDI_INT gp{gp_act}, "
                                            f"gp{lhs_offset_reg}, "
                                            f"{lhs_static_dyn + orow * a_orow_step + k * int(lhs_k_tile_stride)}"
                                        )
                                    else:
                                        lines.append(
                                            f"S_ADDI_INT gp{gp_act}, gp0, "
                                            f"{act_static + k * int(lhs_k_tile_stride)}"
                                        )
                                mat_static = (
                                    rhs_n_mlen_static_full
                                    + oc * oc_b_step
                                    + k * int(rhs_k_tile_stride)
                                )
                                if rhs_offset_reg is not None:
                                    mat_dyn = (
                                        rhs_n_mlen_static_dyn
                                        + oc * oc_b_step
                                        + k * int(rhs_k_tile_stride)
                                    )
                                    lines.append(
                                        f"S_ADDI_INT gp{gp_mat}, "
                                        f"gp{rhs_offset_reg}, {mat_dyn}"
                                    )
                                else:
                                    lines.append(
                                        f"S_ADDI_INT gp{gp_mat}, gp0, {mat_static}"
                                    )
                                if transpose_b:
                                    lines.append(
                                        f"M_TMM 0, gp{gp_act}, gp{gp_mat}"
                                    )
                                else:
                                    lines.append(
                                        f"M_MM 0, gp{gp_mat}, gp{gp_act}"
                                    )
                            if dst_offset_reg is not None:
                                lines.append(
                                    f"S_ADDI_INT gp{gp_out_orow}, "
                                    f"gp{dst_offset_reg}, "
                                    f"{dst_m_base_static_dyn + dst_col + orow * c_orow_step}"
                                )
                            else:
                                lines.append(
                                    f"S_ADDI_INT gp{gp_out_orow}, gp0, "
                                    f"{dst_static}"
                                )
                            lines.append(
                                f"M_MM_WO gp{gp_out_orow}, gp0, 0"
                            )
                        continue

                    lines.append(f"C_LOOP_START gp{gp_loop_orow}, {tiles_per_mlen}")
                    lines.append(f"S_ADDI_INT gp{gp_act}, gp{gp_act_orow}, 0")
                    if rhs_offset_reg is not None:
                        lines.append(
                            f"S_ADDI_INT gp{gp_mat}, gp{rhs_offset_reg}, "
                            f"{rhs_n_mlen_static_dyn + oc * oc_b_step}"
                        )
                    else:
                        lines.append(
                            f"S_ADDI_INT gp{gp_mat}, gp0, "
                            f"{rhs_n_mlen_static_full + oc * oc_b_step}"
                        )
                    lines.append(f"C_LOOP_START gp{gp_loop_k}, {K_tiles}")
                    if transpose_b:
                        lines.append(f"M_TMM 0, gp{gp_act}, gp{gp_mat}")
                    else:
                        lines.append(f"M_MM 0, gp{gp_mat}, gp{gp_act}")
                    lines.append(f"S_ADDI_INT gp{gp_act}, gp{gp_act}, {int(lhs_k_tile_stride)}")
                    lines.append(f"S_ADDI_INT gp{gp_mat}, gp{gp_mat}, {int(rhs_k_tile_stride)}")
                    lines.append(f"C_LOOP_END gp{gp_loop_k}")
                    lines.append(f"M_MM_WO gp{gp_out_orow}, gp0, 0")
                    lines.append(f"S_ADDI_INT gp{gp_act_orow}, gp{gp_act_orow}, {a_orow_step}")
                    lines.append(f"S_ADDI_INT gp{gp_out_orow}, gp{gp_out_orow}, {c_orow_step}")
                    lines.append(f"C_LOOP_END gp{gp_loop_orow}")

        if not caller_owns_scratch:
            ra.free_gp(gp_regs)
        self.program.compiler.generated_code += "\n".join(lines) + "\n"

    def emit_tile_binary(
        self,
        *,
        lhs_vram_addr: int,
        rhs_vram_addr: int,
        dst_vram_addr: int,
        op: str = "add",
        task_id: str = "tile_binary",
        num_rows: Optional[int] = None,
    ) -> None:
        """One ``V_*_VV`` per MLEN-wide row, looped ``num_rows`` times.

        ``num_rows`` defaults to MLEN (legacy behavior — assumes a full
        MLEN×MLEN tile per operand, which is what flash_attention /
        BTMM-style kernels with lane-fused (rows, hlen) post-expansion
        buffers want). Callers with smaller operands (e.g. one
        MLEN-wide row, or any (1, …, MLEN) BSHD buffer where the
        flattened element count is below MLEN²) must pass the actual
        row count or the loop will over-iterate past the operand's end
        and corrupt whatever VRAM follows it.
        """
        op_to_insn = {
            "add": "V_ADD_VV",
            "sub": "V_SUB_VV",
            "mul": "V_MUL_VV",
        }
        if op not in op_to_insn:
            raise ValueError(f"Unsupported tile binary op={op!r}")
        loop_count = self.program.mlen if num_rows is None else int(num_rows)
        if loop_count < 1:
            raise ValueError(f"num_rows must be >= 1, got {loop_count}")
        gp_regs = self.program.compiler.register_allocator.allocate_gp(4)
        gp_dst, gp_lhs, gp_rhs, gp_loop = gp_regs
        lines = [
            f"; tile binary task {task_id} op={op} rows={loop_count}",
        ]
        lines.append(f"S_ADDI_INT gp{gp_dst}, gp0, {dst_vram_addr}")
        lines.append(f"S_ADDI_INT gp{gp_lhs}, gp0, {lhs_vram_addr}")
        lines.append(f"S_ADDI_INT gp{gp_rhs}, gp0, {rhs_vram_addr}")
        lines.append(f"C_LOOP_START gp{gp_loop}, {loop_count}")
        lines.append(f"{op_to_insn[op]} gp{gp_dst}, gp{gp_lhs}, gp{gp_rhs}, 0")
        lines.append(f"S_ADDI_INT gp{gp_dst}, gp{gp_dst}, {self.program.mlen}")
        lines.append(f"S_ADDI_INT gp{gp_lhs}, gp{gp_lhs}, {self.program.mlen}")
        lines.append(f"S_ADDI_INT gp{gp_rhs}, gp{gp_rhs}, {self.program.mlen}")
        lines.append(f"C_LOOP_END gp{gp_loop}")
        self.program.compiler.register_allocator.free_gp(gp_regs)
        self.program.compiler.generated_code += "\n".join(lines) + "\n"

    def emit_tile_add(
        self,
        *,
        lhs_vram_addr: int,
        rhs_vram_addr: int,
        dst_vram_addr: int,
        task_id: str = "tile_add",
    ) -> None:
        self.emit_tile_binary(
            lhs_vram_addr=lhs_vram_addr,
            rhs_vram_addr=rhs_vram_addr,
            dst_vram_addr=dst_vram_addr,
            op="add",
            task_id=task_id,
        )

    def emit_fp_kernel(
        self,
        *,
        src1_addrs: Sequence[int],
        dst_addrs: Sequence[int],
        src2_addrs: Optional[Sequence[int]] = None,
        op: str,
        task_id: str = "fp_kernel",
    ) -> None:
        unary_copy = {"copy", "fill"}
        unary_math = {"exp": "S_EXP_FP", "reci": "S_RECI_FP", "sqrt": "S_SQRT_FP"}
        binary_math = {"add": "S_ADD_FP", "sub": "S_SUB_FP", "mul": "S_MUL_FP", "max": "S_MAX_FP"}
        if len(src1_addrs) != len(dst_addrs):
            raise ValueError("emit_fp_kernel expects matched src1/dst lengths")
        if src2_addrs is not None and len(src2_addrs) != len(dst_addrs):
            raise ValueError("emit_fp_kernel expects matched src2/dst lengths")
        if op in unary_copy:
            gp_regs = self.program.compiler.register_allocator.allocate_gp(3)
            gp_src, gp_dst, gp_loop = gp_regs
            lines = [f"; fp kernel task {task_id} op={op}"]
            src_prog = self.program._arith_progression([int(addr) for addr in src1_addrs])
            dst_prog = self.program._arith_progression([int(addr) for addr in dst_addrs])
            if src_prog is not None and dst_prog is not None:
                src_start, count, src_step = src_prog
                dst_start, _, dst_step = dst_prog
                lines.append(f"S_ADDI_INT gp{gp_src}, gp0, {src_start}")
                lines.append(f"S_ADDI_INT gp{gp_dst}, gp0, {dst_start}")
                lines.append(f"C_LOOP_START gp{gp_loop}, {count}")
                lines.append(f"S_LD_FP f1, gp{gp_src}, 0")
                lines.append(f"S_ST_FP f1, gp{gp_dst}, 0")
                lines.append(f"S_ADDI_INT gp{gp_src}, gp{gp_src}, {src_step}")
                lines.append(f"S_ADDI_INT gp{gp_dst}, gp{gp_dst}, {dst_step}")
                lines.append(f"C_LOOP_END gp{gp_loop}")
            else:
                for src_addr, dst_addr in zip(src1_addrs, dst_addrs):
                    lines.append(f"S_ADDI_INT gp{gp_src}, gp0, {int(src_addr)}")
                    lines.append(f"S_ADDI_INT gp{gp_dst}, gp0, {int(dst_addr)}")
                    lines.append(f"S_LD_FP f1, gp{gp_src}, 0")
                    lines.append(f"S_ST_FP f1, gp{gp_dst}, 0")
            self.program.compiler.register_allocator.free_gp(gp_regs)
            self.program.compiler.generated_code += "\n".join(lines) + "\n"
            return
        if op in unary_math:
            gp_regs = self.program.compiler.register_allocator.allocate_gp(3)
            gp_src, gp_dst, gp_loop = gp_regs
            lines = [f"; fp kernel task {task_id} op={op}"]
            src_prog = self.program._arith_progression([int(addr) for addr in src1_addrs])
            dst_prog = self.program._arith_progression([int(addr) for addr in dst_addrs])
            if src_prog is not None and dst_prog is not None:
                src_start, count, src_step = src_prog
                dst_start, _, dst_step = dst_prog
                lines.append(f"S_ADDI_INT gp{gp_src}, gp0, {src_start}")
                lines.append(f"S_ADDI_INT gp{gp_dst}, gp0, {dst_start}")
                lines.append(f"C_LOOP_START gp{gp_loop}, {count}")
                lines.append(f"S_LD_FP f1, gp{gp_src}, 0")
                if op in {"exp", "reci"}:
                    lines.append(f"{unary_math[op]} f1, f1, 0")
                else:
                    lines.append(f"{unary_math[op]} f1, f1")
                lines.append(f"S_ST_FP f1, gp{gp_dst}, 0")
                lines.append(f"S_ADDI_INT gp{gp_src}, gp{gp_src}, {src_step}")
                lines.append(f"S_ADDI_INT gp{gp_dst}, gp{gp_dst}, {dst_step}")
                lines.append(f"C_LOOP_END gp{gp_loop}")
            else:
                for src_addr, dst_addr in zip(src1_addrs, dst_addrs):
                    lines.append(f"S_ADDI_INT gp{gp_src}, gp0, {int(src_addr)}")
                    lines.append(f"S_ADDI_INT gp{gp_dst}, gp0, {int(dst_addr)}")
                    lines.append(f"S_LD_FP f1, gp{gp_src}, 0")
                    if op in {"exp", "reci"}:
                        lines.append(f"{unary_math[op]} f1, f1, 0")
                    else:
                        lines.append(f"{unary_math[op]} f1, f1")
                    lines.append(f"S_ST_FP f1, gp{gp_dst}, 0")
            self.program.compiler.register_allocator.free_gp(gp_regs)
            self.program.compiler.generated_code += "\n".join(lines) + "\n"
            return
        if op in binary_math:
            if src2_addrs is None:
                raise ValueError(f"emit_fp_kernel op={op!r} requires src2_addrs")
            gp_regs = self.program.compiler.register_allocator.allocate_gp(4)
            gp_a, gp_b, gp_dst, gp_loop = gp_regs
            lines = [f"; fp kernel task {task_id} op={op}"]
            src1_prog = self.program._arith_progression([int(addr) for addr in src1_addrs])
            src2_prog = self.program._arith_progression([int(addr) for addr in src2_addrs])
            dst_prog = self.program._arith_progression([int(addr) for addr in dst_addrs])
            if src1_prog is not None and src2_prog is not None and dst_prog is not None:
                src1_start, count, src1_step = src1_prog
                src2_start, _, src2_step = src2_prog
                dst_start, _, dst_step = dst_prog
                lines.append(f"S_ADDI_INT gp{gp_a}, gp0, {src1_start}")
                lines.append(f"S_ADDI_INT gp{gp_b}, gp0, {src2_start}")
                lines.append(f"S_ADDI_INT gp{gp_dst}, gp0, {dst_start}")
                lines.append(f"C_LOOP_START gp{gp_loop}, {count}")
                lines.append(f"S_LD_FP f1, gp{gp_a}, 0")
                lines.append(f"S_LD_FP f2, gp{gp_b}, 0")
                lines.append(f"{binary_math[op]} f1, f1, f2")
                lines.append(f"S_ST_FP f1, gp{gp_dst}, 0")
                lines.append(f"S_ADDI_INT gp{gp_a}, gp{gp_a}, {src1_step}")
                lines.append(f"S_ADDI_INT gp{gp_b}, gp{gp_b}, {src2_step}")
                lines.append(f"S_ADDI_INT gp{gp_dst}, gp{gp_dst}, {dst_step}")
                lines.append(f"C_LOOP_END gp{gp_loop}")
            else:
                for src1_addr, src2_addr, dst_addr in zip(src1_addrs, src2_addrs, dst_addrs):
                    lines.append(f"S_ADDI_INT gp{gp_a}, gp0, {int(src1_addr)}")
                    lines.append(f"S_ADDI_INT gp{gp_b}, gp0, {int(src2_addr)}")
                    lines.append(f"S_ADDI_INT gp{gp_dst}, gp0, {int(dst_addr)}")
                    lines.append(f"S_LD_FP f1, gp{gp_a}, 0")
                    lines.append(f"S_LD_FP f2, gp{gp_b}, 0")
                    lines.append(f"{binary_math[op]} f1, f1, f2")
                    lines.append(f"S_ST_FP f1, gp{gp_dst}, 0")
            self.program.compiler.register_allocator.free_gp(gp_regs)
            self.program.compiler.generated_code += "\n".join(lines) + "\n"
            return
        raise ValueError(f"Unsupported emit_fp_kernel op={op!r}")

    def emit_row_operation(
        self,
        *,
        src_vram_addr: int,
        dst_vram_addr: Optional[int] = None,
        op: str,
        row_count: int,
        dst_addrs: Optional[Sequence[int]] = None,
        rhs_addrs: Optional[Sequence[int]] = None,
        mask_val: Optional[int] = None,
        task_id: str = "row_operations",
    ) -> None:
        if row_count <= 0:
            return
        unary_ops = {"exp", "reci"}
        reduce_ops = {"reduce_max": "V_RED_MAX", "reduce_sum": "V_RED_SUM"}
        binary_ops = {"mul": "V_MUL_VF", "add": "V_ADD_VF", "sub": "V_SUB_VF"}
        if op not in unary_ops | set(reduce_ops) | set(binary_ops):
            raise ValueError(f"Unsupported emit_row_operation op={op!r}")

        gp_regs = self.program.compiler.register_allocator.allocate_gp(5)
        gp_src, gp_fp, gp_dst, gp_loop, gp_mask = gp_regs
        lines = [f"; row operation task {task_id} op={op} rows={row_count}"]
        lines.append(f"S_ADDI_INT gp{gp_src}, gp0, {int(src_vram_addr)}")
        dst_vram_addr = int(src_vram_addr if dst_vram_addr is None else dst_vram_addr)
        lines.append(f"S_ADDI_INT gp{gp_dst}, gp0, {dst_vram_addr}")
        use_mask = mask_val is not None
        if use_mask:
            lines.append(f"; row operation mask {int(mask_val)}")
            lines.append(f"S_ADDI_INT gp{gp_mask}, gp0, {int(mask_val)}")
            lines.append(f"C_SET_V_MASK_REG gp{gp_mask}")

        if op in unary_ops:
            for row_index in range(int(row_count)):
                row_addr = int(src_vram_addr) + row_index * self.program.mlen
                dst_row_addr = dst_vram_addr + row_index * self.program.mlen
                lines.append(f"S_ADDI_INT gp{gp_src}, gp0, {row_addr}")
                lines.append(f"S_ADDI_INT gp{gp_dst}, gp0, {dst_row_addr}")
                if op == "exp":
                    lines.append(f"V_EXP_V gp{gp_dst}, gp{gp_src}, {1 if use_mask else 0}")
                else:
                    lines.append(f"V_RECI_V gp{gp_dst}, gp{gp_src}, {1 if use_mask else 0}")
        elif op in reduce_ops:
            if dst_addrs is None or len(dst_addrs) != row_count:
                raise ValueError(f"emit_row_operation op={op!r} expects one dst fp addr per row")
            for row_index, dst_addr in enumerate(dst_addrs):
                row_addr = int(src_vram_addr) + row_index * self.program.mlen
                lines.append(f"S_ADDI_INT gp{gp_src}, gp0, {row_addr}")
                lines.append(f"S_ADDI_INT gp{gp_dst}, gp0, {int(dst_addr)}")
                lines.append(f"S_LD_FP f1, gp{gp_dst}, 0")
                lines.append(f"{reduce_ops[op]} f1, gp{gp_src}, {1 if use_mask else 0}")
                lines.append(f"S_ST_FP f1, gp{gp_dst}, 0")
        else:
            if rhs_addrs is None or len(rhs_addrs) not in (1, row_count):
                raise ValueError(f"emit_row_operation op={op!r} expects one rhs fp addr or one per row")
            if len(rhs_addrs) == 1:
                rhs_addr = int(rhs_addrs[0])
                for row_index in range(int(row_count)):
                    row_addr = int(src_vram_addr) + row_index * self.program.mlen
                    dst_row_addr = dst_vram_addr + row_index * self.program.mlen
                    lines.append(f"S_ADDI_INT gp{gp_src}, gp0, {row_addr}")
                    lines.append(f"S_ADDI_INT gp{gp_dst}, gp0, {dst_row_addr}")
                    lines.append(f"S_ADDI_INT gp{gp_fp}, gp0, {rhs_addr}")
                    lines.append(f"S_LD_FP f1, gp{gp_fp}, 0")
                    if op == "sub":
                        lines.append(f"V_SUB_VF gp{gp_dst}, gp{gp_src}, f1, {1 if use_mask else 0}, 0")
                    else:
                        lines.append(f"{binary_ops[op]} gp{gp_dst}, gp{gp_src}, f1, {1 if use_mask else 0}")
            else:
                for row_index, rhs_addr in enumerate(rhs_addrs):
                    row_addr = int(src_vram_addr) + row_index * self.program.mlen
                    dst_row_addr = dst_vram_addr + row_index * self.program.mlen
                    lines.append(f"S_ADDI_INT gp{gp_src}, gp0, {row_addr}")
                    lines.append(f"S_ADDI_INT gp{gp_dst}, gp0, {dst_row_addr}")
                    lines.append(f"S_ADDI_INT gp{gp_fp}, gp0, {int(rhs_addr)}")
                    lines.append(f"S_LD_FP f1, gp{gp_fp}, 0")
                    if op == "sub":
                        lines.append(f"V_SUB_VF gp{gp_dst}, gp{gp_src}, f1, {1 if use_mask else 0}, 0")
                    else:
                        lines.append(f"{binary_ops[op]} gp{gp_dst}, gp{gp_src}, f1, {1 if use_mask else 0}")

        if use_mask:
            lines.append("S_ADDI_INT gp{0}, gp0, 0".format(gp_mask))
            lines.append(f"C_SET_V_MASK_REG gp{gp_mask}")

        self.program.compiler.register_allocator.free_gp(gp_regs)
        self.program.compiler.generated_code += "\n".join(lines) + "\n"
