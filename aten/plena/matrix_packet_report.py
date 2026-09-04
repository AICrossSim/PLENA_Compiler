"""Build reproducible Matrix-SRAM packet evidence from real-shape lowerings."""

from __future__ import annotations

import argparse
import functools
import json
from collections import Counter
from pathlib import Path

from compiler.asm_templates.flashattn.qkt import qkt_multiply
from compiler.aten.plena import PlenaCompiler
from compiler.aten.plena.matrix_access_packets import (
    PacketGeometry,
    coissued_packet_groups,
    coissued_packet_histogram,
    extract_matrix_access_packets,
    matrix_access_instruction_count,
    packet_histogram,
)
from compiler.aten.plena.matrix_recurrence_lowering import (
    KIMI_KDA,
    NEMOTRON_MAMBA,
    MatrixRecurrenceSpec,
    RecurrenceLayout,
    build_recurrence_working_set,
    lower_matrix_recurrence,
    lowering_metrics,
)
from compiler.aten.plena.mview import (
    MatrixViewDescriptor,
    MatrixViewMap,
    MatrixViewShape,
)


def _consumer_descriptor(
    *, mlen: int, blen: int, consumer_row_elements: int
) -> MatrixViewDescriptor:
    if consumer_row_elements % blen or mlen % consumer_row_elements:
        raise ValueError(
            "consumer row width must contain whole BLEN words and divide MLEN"
        )
    return MatrixViewDescriptor(
        shape=MatrixViewShape(
            rows=1,
            cols=consumer_row_elements,
            tile_count=mlen // consumer_row_elements,
        ),
        mapping=MatrixViewMap(
            tile_pitch_rows=consumer_row_elements // blen,
        ),
    )


def _projection_assembly(
    *,
    stage: str,
    input_features: int,
    output_features: int,
    matrix_view: bool = False,
    consumer_row_elements: int | None = None,
    mlen: int = 64,
    blen: int = 4,
) -> str:
    # The legacy projection emitter allocates MLEN-square Matrix tiles even
    # though the published flattened array consumes BLEN-by-MLEN panels.  Two
    # legacy slots are the minimum structural fixture: one view scratch plus
    # one streamed weight chunk.  This override is not a residency claim; the
    # report exposes that boundary explicitly below.
    program = PlenaCompiler(mlen=mlen, blen=blen, mram_tile_capacity=2)
    hidden = program.alloc(
        f"{stage}_hidden",
        1,
        input_features,
        strict=False,
        physical_shape=(blen, input_features),
    )
    weight = program.input(
        f"{stage}_weight",
        (input_features, output_features),
        physical_shape=(input_features, output_features),
        hbm_element_bytes=2,
    )
    program.emit_comment(f"@stage={stage}")
    descriptor = None
    if matrix_view:
        if consumer_row_elements is None:
            raise ValueError("matrix_view requires the real consumer row width")
        descriptor = _consumer_descriptor(
            mlen=mlen,
            blen=blen,
            consumer_row_elements=consumer_row_elements,
        )
    program.linear_projection_bf16(
        hidden,
        weight,
        name=f"{stage}_output",
        matrix_view_descriptor=descriptor,
    )
    return program.compile()


def _case(
    *,
    model: str,
    stage: str,
    input_features: int,
    output_features: int,
    repeats_in_model: int,
    matrix_view: bool = False,
    consumer_row_elements: int | None = None,
) -> dict[str, object]:
    # Use the published PLENA point rather than the transactional smoke-test
    # geometry.  The model dimensions and the machine packet width are thus
    # both the dimensions used by the D'/D physical campaign.
    geometry = PacketGeometry(mlen=2048, blen=32, hlen=128)
    assembly = _projection_assembly(
        stage=stage,
        input_features=input_features,
        output_features=output_features,
        mlen=geometry.mlen,
        blen=geometry.blen,
        matrix_view=matrix_view,
        consumer_row_elements=consumer_row_elements,
    )
    packets = extract_matrix_access_packets(assembly, geometry)
    emitted_accesses = matrix_access_instruction_count(assembly)
    extracted_accesses = len({packet.instruction_index for packet in packets})
    opcodes = Counter(packet.opcode for packet in packets)
    descriptor = (
        _consumer_descriptor(
            mlen=geometry.mlen,
            blen=geometry.blen,
            consumer_row_elements=consumer_row_elements,
        )
        if matrix_view and consumer_row_elements is not None
        else None
    )
    return {
        "model": model,
        "stage": stage,
        "real_shape": [1, input_features, output_features],
        "repeats_in_model": repeats_in_model,
        "lowering": "matrix_view" if matrix_view else "baseline",
        "consumer_descriptor": (
            {
                "rows": descriptor.shape.rows,
                "cols": descriptor.shape.cols,
                "tile_count": descriptor.shape.tile_count,
                "tile_pitch_rows": descriptor.mapping.tile_pitch_rows,
                "fixed_wiring_alpha": 1,
                "packet_values": (
                    descriptor.shape.rows
                    * descriptor.shape.cols
                    * descriptor.shape.tile_count
                ),
            }
            if descriptor is not None
            else None
        ),
        "geometry": {
            "mlen": geometry.mlen,
            "blen": geometry.blen,
            "hlen": geometry.hlen,
        },
        "emitted_matrix_access_instructions": emitted_accesses,
        "extracted_matrix_access_instructions": extracted_accesses,
        "extraction_coverage_complete": extracted_accesses == emitted_accesses,
        "static_matrix_operand_packets": len(packets),
        "dynamic_packet_repeats": sum(packet.repeats for packet in packets),
        "opcode_census": dict(sorted(opcodes.items())),
        "histogram": packet_histogram(packets),
        "coissued_histogram": coissued_packet_histogram(packets),
        "service_groups": coissued_packet_groups(packets),
        "packets": [packet.to_dict() for packet in packets[:32]],
        "packets_truncated": max(0, len(packets) - 32),
        "source": "PlenaCompiler.linear_projection_bf16 official-shape emitted assembly",
        "evidence_level": (
            "official tensor dimensions and executable instruction topology; "
            "symbolic weights and a legacy square-tile scheduling fixture"
        ),
    }


def _recurrence_case(
    spec: MatrixRecurrenceSpec,
    *,
    model: str,
    repeats_in_model: int,
    co_layout: bool,
) -> dict[str, object]:
    geometry = PacketGeometry(mlen=2048, blen=32, hlen=128)
    layout = RecurrenceLayout.AFFINE if co_layout else RecurrenceLayout.FIXED
    working_set = build_recurrence_working_set(spec, layout=layout)
    assembly = lower_matrix_recurrence(
        spec,
        layout=layout,
        mlen=geometry.mlen,
        blen=geometry.blen,
    )
    packets = extract_matrix_access_packets(assembly, geometry)
    emitted_accesses = matrix_access_instruction_count(assembly)
    extracted_accesses = len({packet.instruction_index for packet in packets})
    opcodes = Counter(packet.opcode for packet in packets)
    return {
        "model": model,
        "stage": f"{spec.name}_matrix_recurrence",
        "real_shape": [spec.heads, spec.recurrence_rows, spec.row_elements],
        "repeats_in_model": repeats_in_model,
        "lowering": f"matrix_recurrence_{layout}",
        "geometry": {
            "mlen": geometry.mlen,
            "blen": geometry.blen,
            "hlen": geometry.hlen,
        },
        "emitted_matrix_access_instructions": emitted_accesses,
        "extracted_matrix_access_instructions": extracted_accesses,
        "extraction_coverage_complete": extracted_accesses == emitted_accesses,
        "static_matrix_operand_packets": len(packets),
        "dynamic_packet_repeats": sum(packet.repeats for packet in packets),
        "opcode_census": dict(sorted(opcodes.items())),
        "working_set": working_set.to_dict(),
        "lowering_metrics": lowering_metrics(assembly),
        "histogram": packet_histogram(packets),
        "coissued_histogram": coissued_packet_histogram(packets),
        "service_groups": coissued_packet_groups(packets),
        "packets": [packet.to_dict() for packet in packets[:32]],
        "packets_truncated": max(0, len(packets) - 32),
        "source": "official-shape Matrix recurrence packet-contract lowering",
        "evidence_level": (
            "executable one-read-port Matrix-SRAM phase and operation-count contract; "
            "complete recurrence arithmetic is validated separately by Rust numerical tests"
        ),
    }


def _attention_qkt_case(
    *,
    model: str,
    stage: str,
    heads: int,
    kv_heads: int,
    head_dim: int,
    repeats_in_model: int,
) -> dict[str, object]:
    """Extract the real per-head transposed QK lowering at paper geometry."""

    geometry = PacketGeometry(mlen=2048, blen=32, hlen=128)
    assembly = f"; @stage={stage}\n" + qkt_multiply(
        d=head_dim,
        mlen=geometry.mlen,
        stage="decode",
        alive_registers=list(range(1, 10)),
        q_base_address=0,
        k_base_hbm_offset_reg=0,
        q_head_index=0,
        k_head_index=0,
        use_batched=False,
        blen=geometry.blen,
    )
    packets = extract_matrix_access_packets(assembly, geometry)
    emitted_accesses = matrix_access_instruction_count(assembly)
    extracted_accesses = len({packet.instruction_index for packet in packets})
    opcodes = Counter(packet.opcode for packet in packets)
    return {
        "model": model,
        "stage": stage,
        "real_shape": {
            "query_heads": heads,
            "kv_heads": kv_heads,
            "head_dim": head_dim,
        },
        "repeats_in_model": repeats_in_model,
        "lowering": "real_attention_qkt_template",
        "geometry": {
            "mlen": geometry.mlen,
            "blen": geometry.blen,
            "hlen": geometry.hlen,
        },
        "emitted_matrix_access_instructions": emitted_accesses,
        "extracted_matrix_access_instructions": extracted_accesses,
        "extraction_coverage_complete": extracted_accesses == emitted_accesses,
        "static_matrix_operand_packets": len(packets),
        "dynamic_packet_repeats": sum(packet.repeats for packet in packets),
        "opcode_census": dict(sorted(opcodes.items())),
        "histogram": packet_histogram(packets),
        "coissued_histogram": coissued_packet_histogram(packets),
        "service_groups": coissued_packet_groups(packets),
        "packets": [packet.to_dict() for packet in packets[:32]],
        "packets_truncated": max(0, len(packets) - 32),
        "source": "asm_templates.flashattn.qkt.qkt_multiply per-head decode lowering",
        "evidence_level": (
            "official head topology and executable M_TMV column-read instruction; "
            "symbolic Q/K values"
        ),
    }


@functools.lru_cache(maxsize=1)
def build_report() -> dict[str, object]:
    """Cover Matrix traffic from both hybrid models and all four layer families.

    Projection dimensions are the official model dimensions.  Attention also
    contributes the executable per-head QK-transpose column read; MoE's
    nonlinear/vector stages do not access Matrix SRAM.
    """

    cases = [
        _case(
            model="Nemotron-3 Nano 30B-A3B",
            stage="mamba_in_projection",
            input_features=2688,
            output_features=10304,
            repeats_in_model=23,
        ),
        _case(
            model="Nemotron-3 Nano 30B-A3B",
            stage="mamba_in_projection_lcompute",
            input_features=2688,
            output_features=10304,
            repeats_in_model=23,
            matrix_view=True,
            consumer_row_elements=NEMOTRON_MAMBA.row_elements,
        ),
        _case(
            model="Kimi K3",
            stage="kda_q_projection",
            input_features=7168,
            output_features=12288,
            repeats_in_model=69,
        ),
        _case(
            model="Kimi K3",
            stage="kda_q_projection_lcompute",
            input_features=7168,
            output_features=12288,
            repeats_in_model=69,
            matrix_view=True,
            consumer_row_elements=KIMI_KDA.row_elements,
        ),
        _case(
            model="Nemotron-3 Nano 30B-A3B",
            stage="gqa_q_projection",
            input_features=2688,
            output_features=4096,
            repeats_in_model=6,
        ),
        _case(
            model="Kimi K3",
            stage="mla_q_a_projection",
            input_features=7168,
            output_features=1536,
            repeats_in_model=24,
        ),
        _case(
            model="Nemotron-3 Nano 30B-A3B",
            stage="moe_gate_projection",
            input_features=2688,
            output_features=1856,
            repeats_in_model=23,
        ),
        _case(
            model="Kimi K3",
            stage="latent_moe_gate_projection",
            input_features=7168,
            output_features=3072,
            repeats_in_model=92,
        ),
        _attention_qkt_case(
            model="Nemotron-3 Nano 30B-A3B",
            stage="gqa_attention_qkt",
            heads=32,
            kv_heads=2,
            head_dim=128,
            repeats_in_model=6,
        ),
        _attention_qkt_case(
            model="Kimi K3",
            stage="mla_attention_qkt",
            heads=96,
            kv_heads=96,
            head_dim=192,
            repeats_in_model=24,
        ),
        _recurrence_case(
            NEMOTRON_MAMBA,
            model="Nemotron-3 Nano 30B-A3B",
            repeats_in_model=23,
            co_layout=False,
        ),
        _recurrence_case(
            NEMOTRON_MAMBA,
            model="Nemotron-3 Nano 30B-A3B",
            repeats_in_model=23,
            co_layout=True,
        ),
        _recurrence_case(
            KIMI_KDA,
            model="Kimi K3",
            repeats_in_model=69,
            co_layout=False,
        ),
        _recurrence_case(
            KIMI_KDA,
            model="Kimi K3",
            repeats_in_model=69,
            co_layout=True,
        ),
    ]
    coissued_histograms = [
        entry for case in cases for entry in case["coissued_histogram"]
    ]
    return {
        "schema_version": 2,
        "evidence": "official-shape compiler-emitted Matrix instruction topology",
        "coverage": {
            "all_cases_complete": all(
                case["extraction_coverage_complete"] for case in cases
            ),
            "definition": (
                "Every recognized Matrix access instruction in each supplied lowering "
                "has at least one extracted operand packet. Multi-operand instructions "
                "may produce more than one packet."
            ),
        },
        "capacity_contract": {
            "published_point": {
                "mlen": 2048,
                "blen": 32,
                "matrix_sram_depth_rows": 256,
                "matrix_sram_bf16_bytes": 2048 * 256 * 2,
            },
            "compact_view_footprints": {
                "nemotron_two_operands_bytes": 2 * 32 * 2048 * 2,
                "kimi_two_operands_bytes": 2 * 16 * 2048 * 2,
            },
            "legacy_projection_limit": (
                "The current projection allocator schedules MLEN-square tiles. The "
                "official-shape projection cases therefore use the minimum two-slot "
                "structural fixture and do not claim that an MLEN-square tile resides "
                "in the published 256-row SRAM. Compact L_TILE operands do fit."
            ),
        },
        "scope_boundary": (
            "Cases cover representative official-shape Matrix lowerings for Mamba, "
            "KDA, GQA, MLA and MoE. They are not a real-weight first-to-last-layer "
            "transactional execution."
        ),
        "cases": cases,
        "current_isa_finding": {
            "baseline_packets_name_one_tile": all(
                entry["tiles"] == 1
                for case in cases
                if case["lowering"] == "baseline"
                for entry in case["histogram"]
            ),
            "per_tile_phase_has_current_consumer": any(
                entry["per_tile_phase_can_help"] for entry in coissued_histograms
            ),
            "interpretation": (
                "Arlo's M_* lowering remains the one-tile baseline. The executable "
                "L_TILE path adds a compiler-emitted multi-tile packet after direct "
                "Matrix-accumulator phased writeback."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    report = build_report()
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["current_isa_finding"], indent=2))


if __name__ == "__main__":
    main()
