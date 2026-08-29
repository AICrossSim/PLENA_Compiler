"""PLENA backend for the Mamba-2 ops registered in native_ops.yaml.

Each function is a thin registry adapter over the substrate mixins in
``aten/plena/`` -- the emitters live there because they are also called directly
by ``plena_frontend`` and by the emulator testbenches, which do not go through the
registry. Keeping the adapters this thin means there is exactly one lowering per
operation rather than two that can drift.

Signature convention (shared with every other op in ``aten/ops/plena/``): the
first argument is the ``PlenaCompiler``, the rest are ``VRAMMatrixVar`` /
``InputVar`` / ``FPVar`` proxies, and the return value is a proxy for the result.
ISA is emitted eagerly as a side effect.
"""

from __future__ import annotations

from compiler.aten.plena.program_mamba_common import Mamba2Shape, MambaFPConstants
from compiler.aten.plena.vars import FPVar, InputVar, VRAMMatrixVar


def causal_conv1d_plena(
    prog,
    x: VRAMMatrixVar,
    weight: VRAMMatrixVar,
    out: VRAMMatrixVar,
    scratch: VRAMMatrixVar,
    shape: Mamba2Shape,
    bias: VRAMMatrixVar | None = None,
    num_rows: int | None = None,
    history_rows: int = 0,
):
    return prog.mamba_conv1d_v0(
        x, weight, bias, out, scratch, shape, num_rows=num_rows, history_rows=history_rows
    )


def dt_activation_plena(
    prog,
    dt: VRAMMatrixVar,
    consts: MambaFPConstants,
    shape: Mamba2Shape,
    dt_bias: VRAMMatrixVar | None = None,
    rows: list[int] | None = None,
):
    return prog.mamba_dt_activation_v0(dt, dt_bias, consts, shape, rows=rows)


def gated_rmsnorm_plena(
    prog,
    y: VRAMMatrixVar,
    z: VRAMMatrixVar,
    gate_scratch: VRAMMatrixVar,
    sq_scratch: VRAMMatrixVar,
    rms_fp: FPVar,
    consts: MambaFPConstants,
    shape: Mamba2Shape,
    norm_weight: VRAMMatrixVar | None = None,
    rows: list[int] | None = None,
):
    return prog.mamba_gated_rmsnorm_v0(
        y, z, norm_weight, gate_scratch, sq_scratch, rms_fp, consts, shape, rows=rows
    )


def ssd_scan_plena(
    prog,
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
    """One chunk, one head, of the intra-chunk term.

    Deliberately *not* a whole-layer entry point: the chunk loop, the state
    carry and the transposed in_proj are the caller's, because a single op that
    hid them would also hide the HBM round trips they force (MRAM is writable
    only by ``H_PREFETCH_M``) and make the resulting cost model unreadable.
    """
    return prog.ssd_chunk_head_v0(
        b_chunk=b_chunk,
        c_chunk=c_chunk,
        x_chunk=x_chunk,
        decay=decay,
        scores=scores,
        y_out=y_out,
        shape=shape,
        head_block_base=head_block_base,
        precision=precision,
    )


def ssm_recurrent_step_plena(
    prog,
    *,
    state: VRAMMatrixVar,
    x: VRAMMatrixVar,
    b_fp: FPVar,
    c_fp: FPVar,
    da_fp: FPVar,
    dt_fp: FPVar,
    d_fp: FPVar,
    y: VRAMMatrixVar,
    scratch: VRAMMatrixVar,
    shape: Mamba2Shape,
    consts: MambaFPConstants,
):
    return prog.ssm_decode_step_v0(
        state=state,
        x=x,
        b_fp=b_fp,
        c_fp=c_fp,
        da_fp=da_fp,
        dt_fp=dt_fp,
        d_fp=d_fp,
        y=y,
        scratch=scratch,
        shape=shape,
        consts=consts,
    )


__all__ = [
    "causal_conv1d_plena",
    "dt_activation_plena",
    "gated_rmsnorm_plena",
    "ssd_scan_plena",
    "ssm_recurrent_step_plena",
]
