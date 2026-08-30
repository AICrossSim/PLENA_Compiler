from pathlib import Path

from compiler.aten.plena.hybrid_compile_report import build_report


def test_real_shape_report_uses_one_general_stream_isa_and_reduces_issue():
    model_lib = Path(__file__).resolve().parents[2] / "doc/Model_Lib"
    report = build_report(model_lib)

    assert report["workloads"]["nemotron3"]["layer_counts"] == {
        "mamba": 23,
        "moe": 23,
        "gqa": 6,
    }
    assert report["workloads"]["kimi_k3"]["layer_counts"] == {
        "kda": 69,
        "mla": 24,
        "dense_ffn": 1,
        "latent_moe": 92,
    }
    for name, pair in report["assembly"].items():
        assert pair["stream"]["contains_l_stream_cfg"], name
        assert not pair["stream"]["contains_model_specific_state_opcode"], name
        assert pair["dynamic_issue_reduction"] > 1.0, name
        assert (
            pair["postincrement_only"]["dynamic_issued_instructions"]
            == pair["baseline"]["dynamic_issued_instructions"]
            - pair["baseline"]["foldable_self_advances"]
        ), name
    assert report["isa"]["cache"] is False


def test_layout_speedups_are_explicitly_local_not_layer_claims():
    model_lib = Path(__file__).resolve().parents[2] / "doc/Model_Lib"
    report = build_report(model_lib)
    for plan in report["layout_plans"].values():
        assert plan["scope"] == "layout_buffer_service_only"
        assert plan["selected_cycles"] <= plan["baseline_cycles"]
    assert set(report["layout_plans"]) == {
        "nemotron_mamba_projection",
        "nemotron_mamba_state",
        "kimi_k3_kda_projection",
        "kimi_k3_kda_state",
    }
