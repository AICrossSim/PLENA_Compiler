from compiler.aten.plena.matrix_packet_report import build_report


def test_real_shape_report_covers_both_models_and_four_layer_families() -> None:
    report = build_report()
    stages = {case["stage"] for case in report["cases"]}
    assert stages == {
        "mamba_in_projection",
        "mamba_in_projection_lcompute",
        "kda_q_projection",
        "kda_q_projection_lcompute",
        "gqa_q_projection",
        "gqa_attention_qkt",
        "mla_q_a_projection",
        "mla_attention_qkt",
        "moe_gate_projection",
        "latent_moe_gate_projection",
        "nemotron3_mamba2_matrix_recurrence",
        "kimi_k3_kda_matrix_recurrence",
    }
    assert {case["model"] for case in report["cases"]} == {
        "Nemotron-3 Nano 30B-A3B",
        "Kimi K3",
    }


def test_attention_cases_extract_the_real_column_read() -> None:
    by_stage = {case["stage"]: case for case in build_report()["cases"]}
    for stage in ("gqa_attention_qkt", "mla_attention_qkt"):
        case = by_stage[stage]
        assert case["lowering"] == "real_attention_qkt_template"
        assert case["opcode_census"]["M_TMV"] == 1
        column_packets = [
            entry for entry in case["histogram"] if entry["axis"] == "column"
        ]
        assert column_packets == [
            {
                "stage": stage,
                "axis": "column",
                "tiles": 1,
                "elements_per_tile": 32,
                "static_packets": 1,
                "dynamic_packets": 131_072,
                "values_per_packet": 32,
                "per_tile_skew_can_help": False,
            }
        ]


def test_report_separates_the_one_tile_baseline_from_executable_lcompute() -> None:
    finding = build_report()["current_isa_finding"]
    assert finding["baseline_packets_name_one_tile"]
    assert finding["per_tile_skew_has_current_consumer"]


def test_every_report_case_proves_complete_matrix_access_coverage() -> None:
    report = build_report()
    assert report["schema_version"] == 2
    assert report["coverage"]["all_cases_complete"] is True
    for case in report["cases"]:
        assert case["extraction_coverage_complete"] is True
        assert case["emitted_matrix_access_instructions"] > 0
        assert (
            case["extracted_matrix_access_instructions"]
            == case["emitted_matrix_access_instructions"]
        )


def test_report_does_not_hide_the_legacy_square_tile_capacity_mismatch() -> None:
    contract = build_report()["capacity_contract"]
    assert contract["published_point"]["matrix_sram_bf16_bytes"] == 1024 * 1024
    assert contract["compact_view_footprints"] == {
        "nemotron_two_operands_bytes": 256 * 1024,
        "kimi_two_operands_bytes": 128 * 1024,
    }
    assert "do not claim" in contract["legacy_projection_limit"]


def test_recurrence_report_records_real_same_cycle_multi_operand_packets() -> None:
    recurrence = [
        case
        for case in build_report()["cases"]
        if case["lowering"]
        in {"matrix_recurrence_pitch1", "matrix_recurrence_colayout"}
    ]
    assert len(recurrence) == 4
    for case in recurrence:
        reads = [
            entry
            for entry in case["coissued_histogram"]
            if entry["direction"] == "read"
        ]
        assert reads
        assert any(entry["same_cycle_operands"] == 2 for entry in reads)
        # Fixed diagonal wiring fills the compiler-selected pitch gaps while
        # preserving the same 131072-element recurrence block stride.
        expected_stride = 131_072
        for group in case["service_groups"]:
            for operand in group["operands"]:
                assert operand["address_stride_elements"] == expected_stride

    by_model_and_lowering = {
        (case["model"], case["lowering"]): case for case in recurrence
    }
    for model, pitch in (
        ("Nemotron-3 Nano 30B-A3B", 2),
        ("Kimi K3", 4),
    ):
        pitch1 = by_model_and_lowering[(model, "matrix_recurrence_pitch1")]
        colayout = by_model_and_lowering[(model, "matrix_recurrence_colayout")]
        assert pitch1["opcode_census"] == colayout["opcode_census"]
        assert pitch1["dynamic_packet_repeats"] == colayout["dynamic_packet_repeats"]
        assert {
            operand["tile_pitch_rows"]
            for group in pitch1["service_groups"]
            for operand in group["operands"]
        } == {1}
        assert {
            operand["tile_pitch_rows"]
            for group in colayout["service_groups"]
            for operand in group["operands"]
        } == {pitch}


def test_projection_writeback_uses_the_real_consumer_head_shape() -> None:
    by_stage = {case["stage"]: case for case in build_report()["cases"]}
    assert by_stage["mamba_in_projection_lcompute"]["consumer_descriptor"] == {
        "rows": 1,
        "cols": 64,
        "tile_count": 32,
        "tile_pitch_rows": 2,
        "fixed_wiring_alpha": 1,
        "packet_values": 2048,
    }
    assert by_stage["kda_q_projection_lcompute"]["consumer_descriptor"] == {
        "rows": 1,
        "cols": 128,
        "tile_count": 16,
        "tile_pitch_rows": 4,
        "fixed_wiring_alpha": 1,
        "packet_values": 2048,
    }
    for stage in ("mamba_in_projection_lcompute", "kda_q_projection_lcompute"):
        writes = [
            entry
            for entry in by_stage[stage]["histogram"]
            if entry["axis"] == "producer_writeback"
        ]
        assert writes == [
            {
                "stage": stage,
                "axis": "producer_writeback",
                "tiles": 1,
                "elements_per_tile": 32,
                # With the minimum legal structural fixture (one scratch plus
                # one streamed weight chunk), every final BLEN fragment is
                # explicit instead of being hidden behind an unrealistically
                # large square-tile capacity.
                "static_packets": 384,
                "dynamic_packets": 384,
                "values_per_packet": 32,
                "per_tile_skew_can_help": False,
            }
        ]
