"""Pass 2: assign physical addresses to every HLIR buffer.

Three independent bump allocators (one per memory space):
    - HBM   : starts at HBM_BASE, advances by buffer.byte_size
    - VRAM  : starts at 0, advances by buffer.num_elements
    - MRAM  : starts at 0, advances by buffer.num_elements
    - FPRAM : starts at FPRAM_USER_BASE, advances by buffer.num_elements

Bump-only is sufficient for the kernels we care about right now (no
buffer is reused after its last op). When we start emitting kernels with
long-lived staging buffers we'll swap this for a liveness-aware allocator;
the pass interface won't change.

We also fill in stride/scale defaults for every HBM buffer:
    hbm_stride       <- mlen
    hbm_scale_size   <- mlen * mlen   (i.e. tile_elems)
The runtime emitter applies the same defaults internally; setting them
here makes the values explicit in HLIR so a debug dump shows what the
emitter will actually use.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing import Tuple

from . import hlir as _hlir
from . import scope as _scope


def _logical_2d(shape: Tuple[int, ...]) -> Tuple[int, int]:
    """Collapse N-D shape -> (rows, cols) using the BSHD convention.

    For 3D+ shapes we treat the LAST TWO dims as (heads, head_dim) and
    merge them into the col dimension; everything before them folds into
    rows. This is the "head merging" the runtime compiler does for BTMM
    inputs:
        (B, S, H, D)  -> (B*S, H*D)
        (S, H, D)     -> (S, H*D)
        (rows, cols)  -> (rows, cols)
        (n,)          -> (1, n)

    The whole point: for BTMM, GROUP_HEADS narrow heads of width HLEN
    pack into a single mlen-wide tile (GROUP_HEADS*HLEN == mlen). The
    HBM layout already has them physically contiguous (innermost dims),
    so this merge is a free reinterpretation -- no data movement.
    """
    if not shape:
        return (1, 1)
    if len(shape) == 1:
        return (1, int(shape[0]))
    if len(shape) == 2:
        return (int(shape[0]), int(shape[1]))
    # 3D and 4D: merge last two dims into cols.
    rows = 1
    for s in shape[:-2]:
        rows *= int(s)
    cols = int(shape[-2]) * int(shape[-1])
    return (rows, cols)


# Conservative defaults. Pick non-zero HBM base so address-zero stays
# reserved (handy when debugging null-pointer-style bugs in emitted ISA).
_HBM_BASE = 0x0000
_VRAM_BASE = 0
_MRAM_BASE = 0
# Runtime compiler reserves the first 32 FP slots for system/hardware
# constants and expects them to stay zero-initialized. TVM-generated
# kernels must honor the same contract or FPSRAM preloads/results end up
# shifted relative to the emulator/runtime view.
FPRAM_USER_BASE = 32
_FPRAM_BASE = FPRAM_USER_BASE


@dataclass
class AddressAllocConfig:
    mlen: int
    blen: int
    hbm_base: int = _HBM_BASE
    vram_base: int = _VRAM_BASE
    mram_base: int = _MRAM_BASE
    fpram_base: int = _FPRAM_BASE

    # HBM packing parameters for the BEHAVIOR sim. These mirror the
    # plena_settings.toml [BEHAVIOR.PRECISION.HBM_*_TYPE] entries that
    # `create_mem_for_sim` -> `map_mx_data_to_hbm_for_behave_sim` use to
    # lay out tensors in `hbm_for_behave_sim.bin`. We must match them
    # exactly here, otherwise our ISA's HBM addresses point into the wrong
    # tensor (or padding) and emulator reads garbage.
    hbm_row_width: int = 512   # bytes -- BEHAVIOR.CONFIG.HBM_WIDTH
    hbm_elem_bits: int = 8     # FP4: sign(1)+exp(4)+mant(3) = 8 bits per element
    hbm_scale_bits: int = 8    # 1 byte per scale (Fp(s=0,e=8,m=0))
    hbm_block_size: int = 8    # 1 scale per 8 elements

    @property
    def tile_elems(self) -> int:
        return self.mlen * self.mlen


def _align_up(n: int, mul: int) -> int:
    return ((n + mul - 1) // mul) * mul


def _hbm_packed_byte_size(num_elements: int, cfg: "AddressAllocConfig") -> int:
    """Bytes a single tensor occupies in hbm_for_behave_sim.bin after
    `map_mx_data_to_hbm_for_behave_sim` packs it.

    Layout: [element bytes, padded to hbm_row_width][scale bytes, padded to
    hbm_row_width][total padded to 64 bytes].
    """
    elem_bytes = num_elements * cfg.hbm_elem_bits // 8
    elem_bytes = _align_up(elem_bytes, cfg.hbm_row_width)

    num_scales = num_elements // cfg.hbm_block_size
    scale_bytes = num_scales * cfg.hbm_scale_bits // 8
    if scale_bytes:
        scale_bytes = _align_up(scale_bytes, cfg.hbm_row_width)

    return _align_up(elem_bytes + scale_bytes, 64)


class AddressAllocationPass:
    def __init__(self, cfg: AddressAllocConfig) -> None:
        self.cfg = cfg

    def run(self, mod: _hlir.HLIRModule) -> _hlir.HLIRModule:
        hbm_cur = self.cfg.hbm_base
        vram_cur = self.cfg.vram_base
        mram_cur = self.cfg.mram_base
        fpram_cur = self.cfg.fpram_base

        for buf in mod.buffers.values():
            if buf.scope == _scope.HBM:
                buf.address = hbm_cur
                # IMPORTANT: increment by the MXFP-packed byte size, not by
                # the raw fp16 buf.byte_size. `create_mem_for_sim` packs
                # tensors into hbm_for_behave_sim.bin using FP4 elements
                # (1 byte each) plus 1/8 byte scales, padded to row width.
                # If we use buf.byte_size here our HBM addresses won't match
                # what's actually on disk and H_PREFETCH_M reads garbage.
                hbm_cur += _hbm_packed_byte_size(buf.num_elements, self.cfg)
                rows, cols = _logical_2d(buf.shape)
                # stride = logical 2D cols (full row width). Per-tile DMAs
                # in the ISA pass walk the buffer with this stride so each
                # loaded mlen-wide tile contains adjacent rows.
                if buf.hbm_stride is None:
                    buf.hbm_stride = cols
                # scale_size = total elements of the HBM region (rows*cols),
                # NOT one tile. The runtime ValueManager always uses the HBM
                # backing object's full shape product; defaulting to
                # mlen*mlen only happens to be correct when the buffer is
                # exactly one mlen-square tile. For our multi-tile buffers
                # (e.g. C_hbm = 64x256) this MUST be 16384, not 4096, or the
                # emulator's HBM addressing wraps wrong.
                if buf.hbm_scale_size is None:
                    buf.hbm_scale_size = rows * cols
                # Stash the logical 2D dims as annotations the ISA pass
                # can read to decide per-tile decomposition.
                buf.annotations["logical_rows"] = rows
                buf.annotations["logical_cols"] = cols
                buf.annotations["row_blocks"] = max(1, rows // self.cfg.mlen)
                buf.annotations["col_blocks"] = max(1, cols // self.cfg.mlen)
            elif buf.scope == _scope.VRAM:
                buf.address = vram_cur
                vram_cur += buf.num_elements
            elif buf.scope == _scope.MRAM:
                buf.address = mram_cur
                mram_cur += buf.num_elements
            elif buf.scope == _scope.FPRAM:
                # FPRAM stores scalar FP values; address them in element units
                # to match S_LD_FP / S_ST_FP and the emulator's fpsram indexing.
                buf.address = fpram_cur
                fpram_cur += buf.num_elements
            else:
                raise ValueError(f"buffer {buf.name!r}: unknown scope {buf.scope!r}")

        _hlir.assert_addresses_resolved(mod)
        return mod


__all__ = ["AddressAllocConfig", "AddressAllocationPass", "FPRAM_USER_BASE"]
