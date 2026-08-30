"""Reproducible real-shape Compiler report for hybrid L-Compute.

The report deliberately separates three claims:

* official model manifests establish layer schedules and dimensions;
* emitted assembly establishes what the Compiler actually lowers;
* affine-layout plans price only the producer/consumer buffer service.

It does not turn issued instructions into hardware cycles and it does not claim
that symbolic projection weights are a real checkpoint execution.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from compiler.aten.models.kda.shape import KdaShape
from compiler.aten.plena import PlenaCompiler
from compiler.aten.plena.affine_layout import BankGeometry
from compiler.aten.plena.hybrid_workloads import (
    kimi_k3_manifest,
    kimi_k3_projection_layout_request,
    nemotron3_manifest,
    nemotron_projection_layout_request,
    state_multirow_layout_request,
)
from compiler.aten.plena.instruction_stream import (
    arithmetic_share,
    dynamic_count,
    opcode_census,
    self_advance_counts,
    static_count,
)
from compiler.aten.plena.layout_planner import AffineLayoutPlanner
from compiler.aten.plena.program_kda_common import kda_state_rows, kda_vector_rows
from compiler.aten.plena.program_kda_gates import kda_head_blocks, kda_key_blocks
from compiler.aten.plena.program_kda_mixer import KdaMixerBuffers
from compiler.aten.plena.program_mamba_common import Mamba2Shape


@dataclass(frozen=True)
class AssemblyMetrics:
    static_instructions: int
    dynamic_issued_instructions: int
    arithmetic_share: float
    foldable_self_advances: int
    unfoldable_self_advances: int
    scalar_loads: int
    opcode_census: dict[str, int]
    packetized_opcode_census: dict[str, int]
    contains_l_stream_cfg: bool
    contains_model_specific_state_opcode: bool

    @classmethod
    def from_assembly(cls, assembly: str) -> "AssemblyMetrics":
        census = opcode_census(assembly)
        foldable, unfoldable = self_advance_counts(assembly)
        forbidden = {"X_STATE", "MAMBA_STEP", "KDA_STEP"}
        return cls(
            static_instructions=static_count(assembly),
            dynamic_issued_instructions=dynamic_count(assembly),
            arithmetic_share=arithmetic_share(assembly),
            foldable_self_advances=foldable,
            unfoldable_self_advances=unfoldable,
            scalar_loads=census.get("S_LD_FP", 0),
            opcode_census=dict(sorted(census.items())),
            packetized_opcode_census=_packetized_opcode_census(assembly),
            contains_l_stream_cfg="L_STREAM_CFG" in census,
            contains_model_specific_state_opcode=bool(forbidden & set(census)),
        )


def _packetized_opcode_census(assembly: str) -> dict[str, int]:
    """Count arithmetic issued by explicitly marked multi-row packets.

    Packet mode deliberately reuses existing Vector opcodes, so the ordinary
    opcode census cannot distinguish a full-row ``V_FMA_VF`` from one fed by
    sixteen four-element logical rows.  The emitters own a stable marker and a
    single hardware loop for each packet sweep; count that loop here so the
    Simulator can price physical bank service without guessing from shapes.
    """

    lines = [line.strip() for line in assembly.splitlines() if line.strip()]
    counts: dict[str, int] = {}
    for index, line in enumerate(lines):
        if not line.startswith("; Packetized multi-row "):
            continue
        loop_index = next(
            (
                cursor
                for cursor in range(index + 1, len(lines))
                if lines[cursor].startswith("C_LOOP_START")
            ),
            None,
        )
        if loop_index is None:
            raise ValueError("packetized sweep is missing its hardware loop")
        trips = int(re.findall(r"(-?\d+)", lines[loop_index])[-1])
        body = []
        for cursor in range(loop_index + 1, len(lines)):
            if lines[cursor].startswith("C_LOOP_END"):
                break
            if not lines[cursor].startswith(";"):
                body.append(lines[cursor].split()[0].rstrip(","))
        arithmetic = [opcode for opcode in body if opcode.startswith("V_")]
        if len(arithmetic) != 1:
            raise ValueError(
                "packetized sweep must contain exactly one Vector arithmetic opcode"
            )
        opcode = arithmetic[0]
        counts[opcode] = counts.get(opcode, 0) + trips
    return dict(sorted(counts.items()))


def _up(value: int, multiple: int) -> int:
    return math.ceil(value / multiple) * multiple


def _compiler(
    *,
    stream: bool,
    affine: bool = False,
    packetized: bool = False,
    mlen: int = 64,
) -> PlenaCompiler:
    return PlenaCompiler(
        mlen=mlen,
        blen=4,
        mram_tile_capacity=64,
        stream_addressing=stream,
        stream_affine_alpha=int(affine),
        stream_storage_atom=4,
        stream_packetized=packetized,
    )


def kimi_k3_mixer_assembly(
    *, stream: bool, affine: bool = False, packetized: bool = False
) -> str:
    """Official Kimi K3 recurrent mixer geometry, excluding Matrix projections."""

    mlen = 64
    shape = KdaShape.kimi_k3()
    program = _compiler(
        stream=stream, affine=affine, packetized=packetized, mlen=mlen
    )
    key_blocks = kda_key_blocks(shape, mlen)

    def alloc(name: str, rows: int):
        return program.alloc(name, _up(rows, mlen), mlen, strict=False)

    decay_or_q = program.fp_var("decay_or_q", size=shape.key_dim)
    buffers = KdaMixerBuffers(
        q=alloc("q", shape.num_heads * key_blocks),
        k=alloc("k", shape.num_heads * key_blocks),
        v=alloc("v", kda_vector_rows(shape, mlen)),
        gate=alloc("gate", shape.num_heads * key_blocks),
        dt_bias=alloc("dt_bias", shape.num_heads * key_blocks),
        beta_logit=alloc("beta_logit", kda_head_blocks(shape, mlen)),
        state=alloc("state", kda_state_rows(shape, mlen)),
        out=alloc("out", kda_vector_rows(shape, mlen)),
        pred=alloc("pred", kda_vector_rows(shape, mlen)),
        err=alloc("err", kda_vector_rows(shape, mlen)),
        sq_scratch=alloc("sq_scratch", shape.num_heads * key_blocks),
        decay_fp=decay_or_q,
        q_hat_fp=decay_or_q,
        k_hat_fp=program.fp_var("k_hat", size=shape.key_dim),
        beta_fp=program.fp_var(
            "beta", size=kda_head_blocks(shape, mlen) * mlen
        ),
        part_fp=program.fp_var("part", size=key_blocks),
        acc_fp=program.fp_var("acc", size=1),
        output_scale_fp=program.fp_var("output_scale", size=1),
        rate_fp=program.fp_var("rate", size=shape.num_heads),
        lower_bound_fp=program.fp_var("lower_bound", size=1),
        consts=program.kda_fp_constants(),
    )
    start = len(program.get_code())
    program.kda_beta_scalars_v0(
        beta_logit=buffers.beta_logit,
        beta_fp=buffers.beta_fp,
        consts=buffers.consts,
        shape=shape,
    )
    program.kda_mixer_step_v0(buffers=buffers, shape=shape)
    return program.get_code()[start:]


def nemotron3_mamba_decode_assembly(
    *, stream: bool, affine: bool = False, packetized: bool = False
) -> str:
    """Real Nemotron 3 recurrent geometry, after projection/conv scalar setup.

    B/C are shared by eight heads.  The fixed 1024-entry FPRAM cannot hold all
    eight groups' B and C simultaneously (that alone would need 2048 entries),
    so the static schedule explicitly reuses one 128-entry B/C window per
    group.  This is compiler-managed tiling, not a hidden state cache.
    """

    mlen = 64
    group_shape = Mamba2Shape(
        hidden_size=8 * 64,
        num_heads=8,
        head_dim=64,
        state_size=128,
        n_groups=1,
        conv_kernel=4,
        chunk_size=128,
        seq_len=1,
    )
    program = _compiler(
        stream=stream, affine=affine, packetized=packetized, mlen=mlen
    )
    b_fp = program.fp_var("b_group_window", size=group_shape.state_size)
    c_fp = program.fp_var("c_group_window", size=group_shape.state_size)
    da_fp = program.fp_var("da_group_window", size=group_shape.num_heads)
    dt_fp = program.fp_var("dt_group_window", size=group_shape.num_heads)
    d_fp = program.fp_var("d_group_window", size=group_shape.num_heads)
    consts = program.mamba_fp_constants(group_shape)
    start = len(program.get_code())
    for group in range(8):
        state = program.alloc(
            f"state_group{group}", group_shape.num_heads * group_shape.state_size, mlen
        )
        x = program.alloc(
            f"x_group{group}", _up(group_shape.num_heads, mlen), mlen, strict=False
        )
        y = program.alloc(
            f"y_group{group}", _up(group_shape.num_heads, mlen), mlen, strict=False
        )
        scratch = program.alloc(f"scratch_group{group}", mlen, mlen)
        program.ssm_decode_step_v0(
            state=state,
            x=x,
            b_fp=b_fp,
            c_fp=c_fp,
            da_fp=da_fp,
            dt_fp=dt_fp,
            d_fp=d_fp,
            y=y,
            scratch=scratch,
            shape=group_shape,
            consts=consts,
        )
    return program.get_code()[start:]


def generic_affine_saxpy_assembly(
    *, stream: bool, affine: bool = False, packetized: bool = False
) -> str:
    """A non-model-specific affine row sweep used as the ISA generality gate."""

    mlen = 64
    program = _compiler(
        stream=stream, affine=affine, packetized=packetized, mlen=mlen
    )
    dst = program.alloc("generic_dst", 256, mlen)
    src = program.alloc("generic_src", 256, mlen)
    scalars = program.fp_var("generic_scalars", size=256)
    start = len(program.get_code())
    program.tile_row_fma_fp_sweep(
        dst,
        src,
        scalars,
        dst_rows=list(range(256)),
        src_rows=list(range(256)),
    )
    return program.get_code()[start:]


def _pair(builder) -> dict[str, object]:
    baseline = AssemblyMetrics.from_assembly(builder(stream=False))
    # Keep address-stream extraction and physical co-layout independent.  The
    # instruction reduction below is therefore measured with an identity
    # layout; affine banking is priced separately by the layout planner.
    stream = AssemblyMetrics.from_assembly(builder(stream=True, affine=False))
    packet_row = AssemblyMetrics.from_assembly(
        builder(stream=True, affine=False, packetized=True)
    )
    packet_affine = AssemblyMetrics.from_assembly(
        builder(stream=True, affine=True, packetized=True)
    )
    return {
        "baseline": asdict(baseline),
        "stream": asdict(stream),
        "packet_row_major": asdict(packet_row),
        "packet_affine": asdict(packet_affine),
        "dynamic_issue_reduction": (
            baseline.dynamic_issued_instructions
            / max(1, stream.dynamic_issued_instructions)
        ),
        "postincrement_only": {
            "dynamic_issued_instructions": (
                baseline.dynamic_issued_instructions
                - baseline.foldable_self_advances
            ),
            "removed_foldable_self_advances": baseline.foldable_self_advances,
            "preserved_unfoldable_self_advances": baseline.unfoldable_self_advances,
            "scope": "compiler_issue_stream_not_hardware_cycles",
        },
        "scope": "compiler_issue_stream_not_hardware_cycles",
        "physical_layout": "identity; affine co-layout is reported separately",
    }


def _layout_summary(plan) -> dict[str, object]:
    keep = {"row_major", "consumer_major", "transpose", plan.selected.name}
    candidates = [candidate.to_dict() for candidate in plan.candidates if candidate.name in keep]
    return {
        "request": plan.request,
        "selected": plan.selected.name,
        "baseline_cycles": plan.baseline.total_cycles,
        "selected_cycles": plan.selected.total_cycles,
        "layout_service_speedup": plan.speedup,
        "scope": "layout_buffer_service_only",
        "reported_candidates": candidates,
        "enumerated_candidate_count": len(plan.candidates),
    }


def build_report(model_lib: Path) -> dict[str, object]:
    nemotron = nemotron3_manifest(model_lib / "nemotron-3-nano-30b-a3b.json")
    kimi = kimi_k3_manifest(model_lib / "kimi-k3-text.json")
    geometry = BankGeometry(banks=16, bank_width=4, read_ports=1, write_ports=1)
    planner = AffineLayoutPlanner(geometry)
    layout_plans = {
        "nemotron_mamba_projection": _layout_summary(
            planner.plan(nemotron_projection_layout_request(nemotron, geometry))
        ),
        "kimi_k3_kda_projection": _layout_summary(
            planner.plan(kimi_k3_projection_layout_request(kimi, geometry))
        ),
        "nemotron_mamba_state": _layout_summary(
            planner.plan(
                state_multirow_layout_request(
                    name="nemotron_mamba_state_multirow",
                    groups=nemotron.dimensions["mamba_heads"],
                    rows_per_group=nemotron.dimensions["mamba_state_dim"],
                    row_elements=nemotron.dimensions["mamba_head_dim"],
                    geometry=geometry,
                    parallel_rows=8,
                    repeats=(
                        nemotron.dimensions["mamba_heads"]
                        * math.ceil(nemotron.dimensions["mamba_state_dim"] / 8)
                        * math.ceil(
                            nemotron.dimensions["mamba_head_dim"]
                            / geometry.bank_width
                        )
                    ),
                )
            )
        ),
        "kimi_k3_kda_state": _layout_summary(
            planner.plan(
                state_multirow_layout_request(
                    name="kimi_k3_kda_state_multirow",
                    groups=kimi.dimensions["kda_heads"],
                    rows_per_group=kimi.dimensions["kda_key_dim"],
                    row_elements=kimi.dimensions["kda_value_dim"],
                    geometry=geometry,
                    parallel_rows=8,
                    repeats=(
                        kimi.dimensions["kda_heads"]
                        * math.ceil(kimi.dimensions["kda_key_dim"] / 8)
                        * math.ceil(
                            kimi.dimensions["kda_value_dim"]
                            / geometry.bank_width
                        )
                    ),
                )
            )
        ),
    }
    return {
        "schema_version": 2,
        "claim_boundaries": {
            "weights": "symbolic addresses; no full checkpoint numeric execution",
            "dimensions": "official pinned model dimensions",
            "layout_cycles": "layout-buffer service only",
            "instruction_counts": "compiler issue stream; not PLENA hardware cycles",
        },
        "workloads": {
            "nemotron3": nemotron.to_dict(),
            "kimi_k3": kimi.to_dict(),
        },
        "assembly": {
            "nemotron_mamba_decode_recurrence": _pair(
                nemotron3_mamba_decode_assembly
            ),
            "kimi_k3_decode_recurrent_mixer": _pair(kimi_k3_mixer_assembly),
            "generic_affine_saxpy": _pair(generic_affine_saxpy_assembly),
        },
        "layout_plans": layout_plans,
        "isa": {
            "new_opcode": "L_STREAM_CFG",
            "math_opcodes": "existing Matrix/Vector ISA",
            "loop_opcode": "existing C_LOOP_START/C_LOOP_END",
            "model_specific_opcode": False,
            "cache": False,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-lib", type=Path, default=Path("doc/Model_Lib"))
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)
    report = build_report(args.model_lib)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
