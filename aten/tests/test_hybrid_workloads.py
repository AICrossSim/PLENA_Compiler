from pathlib import Path

from compiler.aten.plena.affine_layout import BankGeometry
from compiler.aten.plena.hybrid_workloads import (
    kimi_k3_manifest,
    kimi_k3_projection_layout_request,
    nemotron3_manifest,
    nemotron_projection_layout_request,
    state_multirow_layout_request,
)
from compiler.aten.plena.layout_planner import AffineLayoutPlanner


ROOT = Path(__file__).resolve().parents[2]


def test_official_nemotron_and_kimi_manifests_are_exact():
    nemotron = nemotron3_manifest(ROOT / "doc/Model_Lib/nemotron-3-nano-30b-a3b.json")
    kimi = kimi_k3_manifest(ROOT / "doc/Model_Lib/kimi-k3-text.json")

    assert nemotron.layer_counts() == {"mamba": 23, "moe": 23, "gqa": 6}
    assert nemotron.dimensions["mamba_projection_width"] == 10_304
    assert nemotron.dimensions["mamba_state_dim"] == 128
    assert nemotron.precisions["recurrent_state"] == "float32"

    assert kimi.layer_counts() == {"kda": 69, "mla": 24, "dense_ffn": 1, "latent_moe": 92}
    assert kimi.layers[-1].mixer == "mla"
    assert kimi.dimensions["kda_heads"] == 96
    assert kimi.dimensions["mla_cache_elements_per_token"] == 576
    assert kimi.precisions["recurrent_state"] == "fp32"


def test_real_shape_projection_requests_are_model_independent_planner_inputs():
    geometry = BankGeometry(banks=16, bank_width=4)
    nemotron = nemotron3_manifest(ROOT / "doc/Model_Lib/nemotron-3-nano-30b-a3b.json")
    kimi = kimi_k3_manifest(ROOT / "doc/Model_Lib/kimi-k3-text.json")

    nem_request = nemotron_projection_layout_request(nemotron, geometry)
    kimi_request = kimi_k3_projection_layout_request(kimi, geometry)
    assert len(nem_request.consumer_packets[0].coords) == 56
    assert len(kimi_request.consumer_packets[0].coords) == 52

    planner = AffineLayoutPlanner(geometry)
    nem_plan = planner.plan(nem_request)
    kimi_plan = planner.plan(kimi_request)
    assert nem_plan.selected.total_cycles <= nem_plan.baseline.total_cycles
    assert kimi_plan.selected.total_cycles <= kimi_plan.baseline.total_cycles
    assert "mamba" not in nem_plan.selected.name
    assert "kda" not in kimi_plan.selected.name


def test_kda_request_keeps_eight_independent_projection_fields():
    geometry = BankGeometry(banks=16, bank_width=4)
    kimi = kimi_k3_manifest(ROOT / "doc/Model_Lib/kimi-k3-text.json")
    request = kimi_k3_projection_layout_request(kimi, geometry)
    assert request.fields == 8
    assert {coord.field for coord in request.consumer_packets[0].coords} == {0, 1, 4, 5}


def test_state_producer_repeat_is_exactly_equivalent_to_explicit_rows():
    geometry = BankGeometry(banks=4, bank_width=2)
    request = state_multirow_layout_request(
        name="generic_state",
        groups=2,
        rows_per_group=3,
        row_elements=8,
        geometry=geometry,
        parallel_rows=2,
        repeats=12,
    )
    assert len(request.producer_packets) == 1
    assert request.producer_packets[0].repeats == 6
    assert len(request.producer_packets[0].coords) == 8
