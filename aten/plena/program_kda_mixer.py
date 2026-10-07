"""The KDA mixer for one decode token: conv, gates, recurrence.

This is the boundary `aten/models/kda/reference.py::kda_state_engine_step`
defines -- everything between the input projection and the output projection.
The projections themselves are ordinary Matrix work and are not part of it.

    q_raw, k_raw, v_raw, gate, beta = split(projected)
    q, k, v = causal_conv(q_raw), causal_conv(k_raw), causal_conv(v_raw)
    output  = kda_step(q, k, v, gate, beta, state, a_log, dt_bias)

Why it streams per head
-----------------------
FPRAM is 512 slots (``FP_SRAM_DEPTH``, ``PLENA_RTL/src/definitions/configuration.svh``).
The recurrence reads ``decay[h,k]``, ``q_hat[h,k]`` and ``k_hat[h,k]`` as FPRAM
scalars, because ``V_MUL_VF`` broadcasts one FP register across every lane while
each of those varies per row of the state tile. Materialising all of them costs
``3 * num_heads * key_dim`` slots -- 36,960 for Kimi K3, seventy-two times the
file.

So the scalars are produced one head at a time into a window that is reused
before the next head. The window is ``2 * key_dim``, not ``3``: ``decay`` and
``q_hat`` are never live at the same time, because decay is dead once the
predict half has run and ``q_hat`` is not read until the read-out. That is why
the recurrence is called as its two halves here rather than through
``kda_decode_step_v0``, and why a caller may pass the same ``FPVar`` as both
``decay_fp`` and ``q_hat_fp``. At Kimi K3 the difference is 620 slots against
492 -- over the 512-slot file, versus inside it. The convolutions do **not** stream: they touch no FPRAM, they are
elementwise across channels, and running them once for all heads keeps their
per-channel-block hardware loops intact.

Ordering constraint
-------------------
``q`` and ``k`` are normalised **in place**, and the normalisation of head ``h``
must happen after the conv has written every head. That is why the three convs
run to completion before the per-head loop starts, rather than being folded into
it: a per-head conv would have to re-roll the shared conv history, and the
history is one tile covering all channels.
"""

from __future__ import annotations

from typing import Sequence

from compiler.aten.models.kda.shape import KdaShape
from compiler.aten.plena.program_kda_common import kda_blocks, kda_stage_marker
from compiler.aten.plena.program_kda_gates import (
    kda_head_blocks,
    kda_key_blocks,
    kda_key_row,
)
from compiler.aten.plena.vars import FPVar, VRAMMatrixVar

__all__ = ["KdaMixerBuffers", "ProgramKdaMixerMixin"]


class KdaMixerBuffers:
    """The tiles and FPRAM windows one mixer step needs.

    Grouped into an object because there are fourteen of them and a positional
    mistake between two same-shaped tiles is a silent wrong answer.
    """

    def __init__(
        self,
        *,
        q: VRAMMatrixVar,
        k: VRAMMatrixVar,
        v: VRAMMatrixVar,
        gate: VRAMMatrixVar,
        dt_bias: VRAMMatrixVar,
        beta_logit: VRAMMatrixVar,
        state: VRAMMatrixVar,
        out: VRAMMatrixVar,
        pred: VRAMMatrixVar,
        err: VRAMMatrixVar,
        sq_scratch: VRAMMatrixVar,
        decay_fp: FPVar,
        q_hat_fp: FPVar,
        k_hat_fp: FPVar,
        beta_fp: FPVar,
        part_fp: FPVar,
        acc_fp: FPVar,
        output_scale_fp: FPVar,
        rate_fp: FPVar,
        lower_bound_fp: FPVar,
        consts,
    ):
        self.q, self.k, self.v = q, k, v
        self.gate, self.dt_bias = gate, dt_bias
        self.beta_logit = beta_logit
        self.state, self.out = state, out
        self.pred, self.err = pred, err
        self.sq_scratch = sq_scratch
        self.decay_fp, self.q_hat_fp, self.k_hat_fp = decay_fp, q_hat_fp, k_hat_fp
        self.beta_fp, self.part_fp, self.acc_fp = beta_fp, part_fp, acc_fp
        self.output_scale_fp = output_scale_fp
        self.rate_fp, self.lower_bound_fp = rate_fp, lower_bound_fp
        self.consts = consts


class ProgramKdaMixerMixin:
    """Assembles the conv, gate and recurrence emitters into one decode step."""

    def kda_mixer_fpram_slots(self, shape: KdaShape) -> dict[str, int]:
        """Every FPRAM slot one mixer step needs, itemised, plus ``"total"``.

        Reported so a caller can check it against ``FP_SRAM_DEPTH`` (512) before
        lowering rather than after allocation fails -- note that the compiler's
        own :class:`FPRAMAllocator` defaults to 1024 and so will *not* catch an
        overflow. The two numbers disagree; the SystemVerilog is authoritative.

        Counting only the per-head window used to hide the overflow. ``beta_fp``
        is not one slot: ``S_MAP_FP_V`` moves whole rows, so it costs
        ``ceil(num_heads / mlen) * mlen`` -- 128 for Kimi K3, not 1. ``rate_fp``
        is indexed by head number and so costs ``num_heads``. With those counted
        honestly the old ``3 * key_dim`` window came to 620 slots at Kimi K3,
        over the file by 108. Sharing one window between ``decay`` and ``q_hat``
        (see :meth:`kda_decode_predict_v0`) brings it to 492.
        """
        key_blocks = kda_key_blocks(shape, self.mlen)
        items = {
            # decay and q_hat share one window; k_hat needs its own, because it
            # is read by both halves of the recurrence.
            "k_hat": shape.key_dim,
            "decay_or_q_hat": shape.key_dim,
            "beta": kda_head_blocks(shape, self.mlen) * self.mlen,
            "rate": shape.num_heads,
            "part": key_blocks,
            "acc": 1,
            "output_scale": 1,
            "lower_bound": 1,
            "consts": len(self.kda_fp_constant_values()),
        }
        items["total"] = sum(items.values())
        return items

    def kda_mixer_step_v0(
        self,
        *,
        buffers: KdaMixerBuffers,
        shape: KdaShape,
        head_rows: Sequence[int] | None = None,
    ) -> VRAMMatrixVar:
        """One KDA decode token, from post-conv activations to ``out``.

        ``buffers.q`` / ``.k`` hold the *convolved* q and k, key-width and
        blocked by :func:`kda_key_row`; ``.v`` is value-width and blocked by
        :func:`kda_vector_row`. ``.gate`` is key-width and is consumed in place.
        Call :meth:`kda_conv_step_v0` three times first to produce them --
        see the module docstring for why that is not folded in here.

        **Call :meth:`kda_beta_scalars_v0` once before this**, not per call.
        It is not folded in because it consumes ``beta_logit`` in place and
        sigmoid is not idempotent: a second application moves the value by ~0.43.
        Folding it in made ``head_rows`` silently wrong -- lowering three heads
        in three calls gave head 0 the right answer and the rest
        ``sigmoid(sigmoid(...))``.

        In-place consumption, and what it means for a second token:

        * ``gate`` becomes the decay. **Not idempotent** (~0.90 off), so reseed it.
        * ``beta_logit`` becomes beta. **Not idempotent** (~0.43 off), reseed it.
        * ``q`` and ``k`` are normalised. Idempotent -- a unit vector normalises
          to itself -- but they must be reseeded anyway because the next token's
          projections differ.
        * ``state`` is updated in place, and that is the point: it carries.
        """
        heads = list(range(shape.num_heads)) if head_rows is None else list(head_rows)
        key_blocks = kda_key_blocks(shape, self.mlen)
        b = buffers

        for name, want in (
            ("q_hat_fp", shape.key_dim),
            ("k_hat_fp", shape.key_dim),
            ("decay_fp", shape.key_dim),
        ):
            fp = getattr(b, name)
            if fp.size < want:
                raise ValueError(
                    f"{name} holds {fp.size} slots but the per-head window needs "
                    f"{want}; it is reused every head, so size it to key_dim and "
                    f"not to num_heads * key_dim"
                )
        if b.part_fp.size < key_blocks:
            raise ValueError(f"part_fp needs {key_blocks} slots, has {b.part_fp.size}")
        if b.acc_fp.size < 1:
            raise ValueError("acc_fp needs at least one slot")

        for h in heads:
            first = kda_key_row(shape, self.mlen, h, 0)
            rows = [kda_key_row(shape, self.mlen, h, kb) for kb in range(key_blocks)]

            # k first: the predict half reads k_hat and decay, and q_hat may be
            # the same storage as decay (see the module docstring), so q cannot
            # be materialised until the predict half is done with decay.
            self.emit_comment(kda_stage_marker("kda_normalize", f"k head={h}"))
            self.kda_l2_normalize_blocked_v0(
                b.k, vectors=1, blocks=key_blocks, sq_scratch=b.sq_scratch,
                part_fp=b.part_fp, acc_fp=b.acc_fp, consts=b.consts,
                first_row=first,
            )
            # Row `first + i` lands at offset `i * mlen`, so the window is
            # key-major and starts at 0 -- which is what both halves index with
            # fp_base = 0 for a single head.
            self.tile_row_to_fpram(b.k, b.k_hat_fp, rows=rows)

            self.kda_decay_scalars_v0(
                gate=b.gate, dt_bias=b.dt_bias,
                rate_fp=b.rate_fp, lower_bound_fp=b.lower_bound_fp,
                decay_fp=b.decay_fp, consts=b.consts, shape=shape, heads=[h],
            )
            self.kda_decode_predict_v0(
                state=b.state,
                k_fp=b.k_hat_fp, decay_fp=b.decay_fp, beta_fp=b.beta_fp,
                v=b.v, pred=b.pred, err=b.err, shape=shape, head_rows=[h],
                # The scalar window is reused every head, so this head's decay
                # and k_hat sit at offset 0 -- not at h * key_dim. beta_fp is
                # the exception: one slot per head, produced once before this
                # call, and indexed by head number directly.
                fp_head_stride=0,
            )

            # decay is dead now, so q_hat may overwrite it.
            self.emit_comment(kda_stage_marker("kda_normalize", f"q head={h}"))
            self.kda_l2_normalize_blocked_v0(
                b.q, vectors=1, blocks=key_blocks, sq_scratch=b.sq_scratch,
                part_fp=b.part_fp, acc_fp=b.acc_fp, consts=b.consts,
                first_row=first,
            )
            self.tile_row_to_fpram(b.q, b.q_hat_fp, rows=rows)

            self.kda_decode_update_v0(
                state=b.state,
                k_fp=b.k_hat_fp, q_fp=b.q_hat_fp, o=b.out, err=b.err,
                shape=shape, output_scale_fp=b.output_scale_fp,
                head_rows=[h], fp_head_stride=0,
            )
        return b.out
