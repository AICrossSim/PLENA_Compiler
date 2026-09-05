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
                "per_tile_phase_can_help": False,
            }
        ]


def test_report_separates_the_one_tile_baseline_from_executable_lcompute() -> None:
    finding = build_report()["current_isa_finding"]
    assert finding["baseline_packets_name_one_tile"]
    assert finding["per_tile_phase_has_current_consumer"]


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


def test_recurrence_report_records_every_physical_l_tile_service_phase() -> None:
    recurrence = [
        case
        for case in build_report()["cases"]
        if case["lowering"]
        in {"matrix_recurrence_fixed", "matrix_recurrence_affine"}
    ]
    assert len(recurrence) == 4
    for case in recurrence:
        l_tile_groups = [
            entry
            for entry in case["coissued_histogram"]
            if entry["opcode"] == "L_TILE_EXEC"
        ]
        assert {
            (entry["direction"], entry["axis"])
            for entry in l_tile_groups
        } == {
            ("read", "l_tile_dst_read"),
            ("read", "l_tile_source_read"),
            ("read", "l_tile_scale_read"),
            ("write", "l_tile_dst_write"),
        }
        # The paper point has one Matrix-SRAM read port per bank.  Destination,
        # source, and per-segment scales are therefore consecutive physical
        # service phases inside L_TILE, not fictitious same-cycle operands.
        assert all(entry["same_cycle_operands"] == 1 for entry in l_tile_groups)
        assert all(entry["dynamic_service_groups"] > 0 for entry in l_tile_groups)

        dma_groups = [
            entry
            for entry in case["coissued_histogram"]
            if entry["axis"] == "view_dma"
        ]
        assert {entry["direction"] for entry in dma_groups} == {"read", "write"}

        # L_TILE performs the internal row walk itself.  No surrounding hardware
        # loop mutates its Matrix bases, so the executable packet extractor must
        # report invariant base addresses rather than inventing an outer stride.
        for group in case["service_groups"]:
            for operand in group["operands"]:
                assert operand["address_stride_elements"] == 0

    by_model_and_lowering = {
        (case["model"], case["lowering"]): case for case in recurrence
    }
    for model in ("Nemotron-3 Nano 30B-A3B", "Kimi K3"):
        fixed = by_model_and_lowering[(model, "matrix_recurrence_fixed")]
        affine = by_model_and_lowering[(model, "matrix_recurrence_affine")]
        assert fixed["working_set"]["layout"] == "fixed"
        assert affine["working_set"]["layout"] == "affine"
        assert fixed["working_set"]["capacity_bytes"] == 1024 * 1024
        assert affine["working_set"]["capacity_bytes"] == 1024 * 1024
        assert fixed["lowering_metrics"]["contains_l_tile"] is True
        assert affine["lowering_metrics"]["contains_l_tile"] is True
        # Both variants execute the same official recurrence, but affine packing
        # may retain a larger head/state chunk and therefore remove fixed-layout
        # reloads.  That difference is a measured mechanism benefit, not a hidden
        # capacity increase.
        assert (
            affine["working_set"]["group_heads"]
            >= fixed["working_set"]["group_heads"]
        )
        assert (
            affine["lowering_metrics"]["state_transfer_values"]
            <= fixed["lowering_metrics"]["state_transfer_values"]
        )


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
                "static_packets": 64,
                "dynamic_packets": 64,
                "values_per_packet": 32,
                "per_tile_phase_can_help": False,
            }
        ]


def test_direct_view_packet_evidence_does_not_claim_full_multi_output_execution() -> None:
    for case in build_report()["cases"]:
        if case["lowering"] != "matrix_view":
            continue
        # The old report generated six output blocks into the same view. The
        # fixture now preserves the real weight shape but emits just one.
        assert case["real_shape"][2] > 2048
        assert case["emitted_output_columns"] == [0, 2048]
        assert case["full_output_tile_count"] == 6
        assert case["full_projection_emitted"] is False
        assert "one output packet only" in case["evidence_level"]
        assert "first output tile" in case["source"]
