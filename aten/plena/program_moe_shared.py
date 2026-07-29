"""Shared-expert MoE program-builder helpers.

A *shared* expert is an FFN every token passes through, running alongside the
routed experts and summed into the same output:

    y = shared(x) + sum_k route_weight_k * routed_expert_k(x)

Architectures that use one:

===================================  =========  ======================  =========
model                                n_shared   shared intermediate     gate
===================================  =========  ======================  =========
DeepSeek-V2 / V3, Kimi K2            1-2        moe_inter * n_shared    none
Qwen2-MoE (A2.7B / A14B)             1          independent             sigmoid
Llama-4 Scout / Maverick             1          == routed               none
GLM-4.5, Qwen3-Next                  1          independent             none
GPT-OSS, Qwen3-MoE, Mixtral          0          --                      --
===================================  =========  ======================  =========

``n_shared`` is deliberately not a parameter here. DeepSeek stores its shared
experts pre-concatenated and builds a *single* MLP of width
``moe_intermediate_size * n_shared_experts`` -- so the count is a checkpoint
loading concern, not an emission one. :func:`fused_shared_intermediate` makes that
conversion explicit rather than leaving callers to rediscover it.

Cost shape, and why it is worth measuring separately
----------------------------------------------------
The shared expert is dense over every token, so its cost is
``rows * (2*hidden*inter_s + inter_s*hidden)`` regardless of routing, while the
routed branch scales with ``rows * top_k``. Its weights are also *hot* -- every
token needs them -- so unlike routed weights they do not have to be re-fetched
per (token, expert) pair.

That last difference is why these emitters bill to their own stages
(``shared_expert_*``) rather than reusing the routed ones: without the split, the
profile cannot answer what fraction of MoE time is architecturally dense.

**Read the resulting split with care.** The routed path here processes one
(token, expert) pair at a time in a BLEN-row slot, re-fetching expert weights per
pair -- ``moe_gather_token_rows_from_hbm_v0`` calls this "intentionally wasteful
but keeps the first L2 correctness path exact". The shared path has no such
constraint and runs as a plain batched projection over all rows. A shared-vs-routed
cycle ratio measured today therefore reflects *the routed lowering's per-pair
overhead* at least as much as it reflects the architecture. It is a valid
measurement of this compiler's output; it is not a statement about MoE hardware in
general.
"""

from __future__ import annotations

import math

from compiler.aten.isa_builder import IsaBuilder, fp, gp
from compiler.aten.plena.program_routed_moe import (
    ExpertBiases,
    ExpertWeights,
    GptOssFPConstants,
    moe_stage_marker,
)
from compiler.aten.plena.vars import FPVar, VRAMMatrixVar


def fused_shared_intermediate(moe_intermediate_size: int, n_shared_experts: int) -> int:
    """Width of the single MLP that stands in for ``n_shared_experts`` shared experts.

    DeepSeek-V2/V3 do not run their shared experts separately; ``DeepseekV2MoE``
    instantiates one ``DeepseekV2MLP`` of width
    ``moe_intermediate_size * n_shared_experts`` whose weights are the shared
    experts concatenated along the intermediate axis. Because SwiGLU is applied
    elementwise along that axis and the down projection sums over it, the fused
    MLP is exactly equal to the sum of the individual shared experts -- no
    approximation.
    """
    if n_shared_experts < 1:
        raise ValueError(f"n_shared_experts must be >= 1, got {n_shared_experts}")
    if moe_intermediate_size < 1:
        raise ValueError(f"moe_intermediate_size must be >= 1, got {moe_intermediate_size}")
    return moe_intermediate_size * n_shared_experts


class ProgramMoeSharedMixin:
    """Shared-expert emit helpers layered on the routed-MoE substrate."""

    def moe_shared_gate_v0(
        self,
        x: VRAMMatrixVar,
        gate_weight_row: VRAMMatrixVar,
        *,
        rows: int,
        hidden: int,
        constants: GptOssFPConstants,
        gate_fp_scratch: FPVar | None = None,
        zero_row: FPVar | None = None,
        policy_name: str = "qwen2_moe",
        name: str = "moe_shared_gate",
    ) -> VRAMMatrixVar:
        """Emit Qwen2-MoE's ``sigmoid(x @ w_gate)`` shared-expert gate.

        ``gate_weight_row`` holds the ``[hidden, 1]`` gate projection stored as a
        single BF16 VRAM row of length ``hidden``, so the per-token projection is a
        dot product rather than a matmul. Going through the matrix path would mean
        padding a one-column weight out to MLEN and discarding 63/64 of the result.

        Returns a ``[rows, hidden]`` VRAM matrix holding each token's scalar gate
        broadcast across the hidden axis, ready to multiply the shared expert
        output. That is the same shape and mechanism the routed branch uses for
        route weights, so the two are combined identically downstream.

        The sigmoid is evaluated in the scalar FP unit
        (``S_SUB_FP`` / ``S_EXP_FP`` / ``S_ADD_FP`` / ``S_RECI_FP``) on one value
        per token, not as a vector pass over a broadcast row -- there is exactly one
        gate scalar per token, and the vector form would compute MLEN copies of it.
        """
        if hidden % self.mlen != 0:
            raise ValueError(f"{name}: hidden={hidden} must be divisible by MLEN={self.mlen}")
        if rows > x.physical_shape[0]:
            raise ValueError(f"{name}: rows={rows} exceeds x physical rows={x.physical_shape[0]}")
        if hidden > x.shape[1]:
            raise ValueError(f"{name}: hidden={hidden} exceeds x width={x.shape[1]}")
        if gate_weight_row.shape[1] < hidden:
            raise ValueError(
                f"{name}: gate_weight_row width={gate_weight_row.shape[1]} is narrower than hidden={hidden}"
            )
        _zero, _limit_pos, _limit_neg, one, _neg = constants

        scratch = self.alloc(
            f"{name}_dot_scratch",
            rows=1,
            cols=self.mlen,
            strict=False,
            physical_shape=(1, self.mlen),
        )
        gate_scalars = gate_fp_scratch or self.fp_var(f"{name}_scalars", size=rows)
        if gate_scalars.size < rows:
            raise ValueError(f"{name}: gate scratch size={gate_scalars.size} is smaller than rows={rows}")

        k_blocks = hidden // self.mlen
        x_step = x.physical_shape[0] * self.mlen
        w_step = gate_weight_row.physical_shape[0] * self.mlen

        gp_x, gp_w, gp_scratch, gp_out, gp_loop = self._reg.allocate_gp(5)
        fp_acc, fp_one, fp_tmp = self.allocate_fp_reg(3)
        try:
            asm = IsaBuilder().comment(
                moe_stage_marker(
                    "shared_expert_gate",
                    f"[{policy_name}] {name}: rows={rows}, hidden={hidden}",
                )
            )
            asm.instr("S_ADDI_INT", gp(gp_scratch), gp(0), self.get_vram_addr(scratch.name))
            # 1.0 is loop-invariant; hoist the load out of the token loop.
            asm.instr("S_ADDI_INT", gp(gp_out), gp(0), one.address)
            asm.instr("S_LD_FP", fp(fp_one), gp(gp_out), 0)

            for token_idx in range(rows):
                asm.comment(f"shared gate token {token_idx}")
                asm.instr("S_ADD_FP", fp(fp_acc), fp(0), fp(0))
                asm.instr("S_ADDI_INT", gp(gp_x), gp(0), self._vram_matrix_row_addr(x, token_idx, 0))
                asm.instr("S_ADDI_INT", gp(gp_w), gp(0), self._vram_matrix_row_addr(gate_weight_row, 0, 0))
                asm.instr("C_LOOP_START", gp(gp_loop), k_blocks)
                asm.instr("V_MUL_VV", gp(gp_scratch), gp(gp_x), gp(gp_w), 0)
                asm.instr("V_RED_SUM", fp(fp_acc), gp(gp_scratch), 0, 0)
                asm.instr("S_ADDI_INT", gp(gp_x), gp(gp_x), x_step)
                asm.instr("S_ADDI_INT", gp(gp_w), gp(gp_w), w_step)
                asm.instr("C_LOOP_END", gp(gp_loop))

                # sigmoid(acc) = 1 / (1 + exp(-acc)). fp register 0 reads as zero, so
                # the negation is a subtract rather than a multiply by a -1 constant
                # the caller would otherwise have to preload.
                asm.instr("S_SUB_FP", fp(fp_tmp), fp(0), fp(fp_acc))
                asm.instr("S_EXP_FP", fp(fp_tmp), fp(fp_tmp))
                asm.instr("S_ADD_FP", fp(fp_tmp), fp(fp_tmp), fp(fp_one))
                asm.instr("S_RECI_FP", fp(fp_tmp), fp(fp_tmp))
                asm.instr("S_ADDI_INT", gp(gp_out), gp(0), gate_scalars.address + token_idx)
                asm.instr("S_ST_FP", fp(fp_tmp), gp(gp_out), 0)
            self._emit(asm)
        finally:
            self.free_fp_reg([fp_acc, fp_one, fp_tmp])
            self._reg.free_gp([gp_x, gp_w, gp_scratch, gp_out, gp_loop])

        # Broadcasting one FP scalar per row across `hidden` is exactly what the
        # routed branch does with V_TOPK route weights, so reuse that emitter
        # instead of growing a second copy of the S_MAP_V_FP loop. It bills to the
        # stage it is told to: re-marking *after* the call would be too late, since
        # markers are sticky forwards and the whole broadcast would already have
        # been attributed to `expert_route_weight`.
        return self.moe_materialize_route_weights_for_active_rows_v0(
            weights_fp_base=gate_scalars.address,
            pair_indices=list(range(rows)),
            active_rows=list(range(rows)),
            rows=rows,
            hidden=hidden,
            zero_row=zero_row,
            policy_name=policy_name,
            stage="shared_expert_gate",
            name=f"{name}_broadcast",
        )

    def moe_shared_expert_v0(
        self,
        x: VRAMMatrixVar,
        weights: ExpertWeights,
        *,
        biases: ExpertBiases | None = None,
        rows: int,
        intermediate: int,
        constants: GptOssFPConstants,
        gate_weight_row: VRAMMatrixVar | None = None,
        gate_fp_scratch: FPVar | None = None,
        zero_row: FPVar | None = None,
        activation_policy: str = "standard_swiglu",
        policy_name: str = "deepseek_moe",
        name: str = "moe_shared_expert",
    ) -> VRAMMatrixVar:
        """Emit the shared expert over **all** ``rows`` tokens and return its output.

        Unlike :meth:`moe_dynamic_expert_pair_v0` this takes a static
        :class:`InputVar` weight triple: the shared expert is the same for every
        token, so there is no expert id to resolve, no ``C_SET_ADDR_REG`` dance, and
        no per-pair HBM base computation. It is an ordinary batched projection.

        ``gate_weight_row`` turns on the Qwen2-MoE sigmoid gate. Leave it ``None``
        for DeepSeek, Llama-4 and GLM, which add the shared output unweighted.
        """
        if rows < 1:
            raise ValueError(f"{name}: rows must be positive, got {rows}")
        if intermediate % self.mlen != 0:
            raise ValueError(f"{name}: intermediate={intermediate} must be divisible by MLEN={self.mlen}")
        w_gate, w_up, w_down = weights
        gate_bias, up_bias, down_bias = biases or (None, None, None)

        # Matches the routed path: the K-split projection accumulates with a
        # 64x64 block add, so keep outputs tile-backed rather than letting a
        # short row count leave the block add walking into the next column block.
        projection_rows = max(self.mlen, x.physical_shape[0], math.ceil(rows / self.blen) * self.blen)

        self._emit(
            IsaBuilder().comment(
                moe_stage_marker(
                    "shared_expert_projection",
                    f"[{policy_name}] {name} gate/up: rows={rows}, intermediate={intermediate}",
                )
            )
        )
        gate = self.linear_projection(
            x,
            w_gate,
            name=f"{name}_gate",
            physical_shape=(projection_rows, w_gate.physical_shape[1]),
        )
        up = self.linear_projection(
            x,
            w_up,
            name=f"{name}_up",
            physical_shape=(projection_rows, w_up.physical_shape[1]),
        )
        if gate_bias is not None:
            self.vram_add(gate, gate_bias, num_rows=rows)
        if up_bias is not None:
            self.vram_add(up, up_bias, num_rows=rows)

        hidden = self.moe_expert_activation_v0(
            gate,
            up,
            rows=rows,
            intermediate=intermediate,
            constants=constants,
            activation_policy=activation_policy,
            policy_name=policy_name,
            stage="shared_expert_activation",
            name=name,
        )

        self._emit(
            IsaBuilder().comment(
                moe_stage_marker("shared_expert_projection", f"[{policy_name}] {name} down: rows={rows}")
            )
        )
        out = self.linear_projection(
            hidden,
            w_down,
            name=f"{name}_out",
            physical_shape=(projection_rows, w_down.physical_shape[1]),
        )
        if down_bias is not None:
            self.vram_add(out, down_bias, num_rows=rows)

        if gate_weight_row is not None:
            gate_matrix = self.moe_shared_gate_v0(
                x,
                gate_weight_row,
                rows=rows,
                hidden=w_down.physical_shape[1],
                constants=constants,
                gate_fp_scratch=gate_fp_scratch,
                zero_row=zero_row,
                policy_name=policy_name,
                name=f"{name}_gate_sigmoid",
            )
            self.vram_mul(out, gate_matrix, num_rows=rows)
        return out

    def moe_combine_shared_and_routed_v0(
        self,
        accumulator: VRAMMatrixVar,
        shared_out: VRAMMatrixVar,
        *,
        rows: int,
        policy_name: str = "deepseek_moe",
        name: str = "moe_shared_combine",
    ) -> VRAMMatrixVar:
        """Add the shared-expert output into the routed accumulator in place.

        Thin over :meth:`vram_add`, but it exists rather than being inlined at the
        call site so the combine is attributed to ``scatter_combine`` instead of
        inheriting whichever stage the caller happened to leave live. Under
        marker-authoritative classification an unmarked helper is billed to the
        previous marker, which here would be ``shared_expert_projection`` -- making
        the shared branch look more expensive than it is.

        The shared output is added *unweighted*: no architecture scales it by a
        route weight. Qwen2-MoE's sigmoid gate is already folded into
        ``shared_out`` by :meth:`moe_shared_expert_v0`.
        """
        if accumulator.physical_shape[1] != shared_out.physical_shape[1]:
            raise ValueError(
                f"{name}: accumulator width={accumulator.physical_shape[1]} does not match "
                f"shared output width={shared_out.physical_shape[1]}"
            )
        if rows > accumulator.physical_shape[0]:
            raise ValueError(f"{name}: rows={rows} exceeds accumulator rows={accumulator.physical_shape[0]}")
        self._emit(IsaBuilder().comment(moe_stage_marker("scatter_combine", f"[{policy_name}] {name}: rows={rows}")))
        self.vram_add(accumulator, shared_out, num_rows=rows)
        return accumulator


__all__ = [
    "ProgramMoeSharedMixin",
    "fused_shared_intermediate",
]
