"""Byte-contract, numerical and routing checks for the normal-buffer exporter."""
import hashlib
import json
import math

import pytest

from compiler.aten.plena.moe_normal_export import (
    bf16_bits, bf16_value, decode_element, decode_matrix, demo_arrays,
    encode_matrix, export_workload, group_routes, numerical_reference,
)


def _unit_arrays():
    return {"inputs": [[1.0]],
            "experts": [{"id": 3, "gate": [[1.0]], "up": [[1.0]], "down": [[1.0]]}],
            "routes": [{"token": 0, "slot": 0, "expert": 3, "weight": 1.0}]}


def test_actual_plena_codec_known_bytes_rounding_subnormals_and_tail():
    elements, scales, shape = encode_matrix([
        [1, 1.064, 1.09375, -1, 0, 2**-10, 2**-7, 2**-6, 2],
    ])
    # 1.064 rounds down with PLENA's quarter-truncated mantissa rounding.
    # 2^-7 saturates the local subnormal mantissa, unlike standard E4M3.
    assert elements == bytes([0x38, 0x38, 0x39, 0xB8, 0, 1, 7, 8, 0x38] + [0] * 7)
    assert scales == bytes([127, 128])
    assert shape == {"rows": 1, "cols": 9, "element_row_stride": 16, "scale_row_stride": 2}
    assert decode_element(0x07, 127) == 7 / 1024
    assert decode_element(0x87, 127) == -7 / 1024
    assert decode_element(0x38, 128) == 2
    assert decode_element(0x38, 0) == 0
    with pytest.raises(ValueError, match="non-finite"):
        decode_element(0x78, 127)


def test_row_boundaries_never_share_scale_blocks():
    elements, scales, shape = encode_matrix([[1] * 9, [8] * 9])
    assert list(scales) == [127, 127, 130, 130]
    assert elements[9:16] == bytes(7)
    assert elements[25:32] == bytes(7)
    region = {**shape, "element_base": 0, "scale_base": len(elements)}
    assert decode_matrix(elements + scales, region) == [[1] * 9, [8] * 9]


def test_bf16_round_nearest_even_is_explicit():
    assert bf16_bits(1 + 1 / 256) == 0x3F80
    assert bf16_bits(1 + 3 / 256) == 0x3F82
    assert bf16_value(0xBF80) == -1.0
    for value in (math.nan, math.inf, 1e100):
        with pytest.raises(ValueError):
            bf16_bits(value)


def test_hand_computed_nonzero_swiglu_fixture(tmp_path):
    result = export_workload(tmp_path, **_unit_arrays())
    # SiLU(1) is 0.731058..., rounded BF16 is 187/256; unit Up and Down.
    assert result["golden"]["output_bf16"] == [[0x3F3B]]
    assert result["golden"]["output_f32"] == [[187 / 256]]
    assert result["golden"]["pre_round_output_f32"] == [[187 / 256]]


def test_grouping_retains_weights_slots_duplicates_and_zero_weights():
    routes = [
        {"token": 2, "slot": 7, "expert": 5, "weight": 0.0},
        {"token": 0, "slot": 1, "expert": 3, "weight": 0.75},
        {"token": 0, "slot": 0, "expert": 3, "weight": 0.25},
    ]
    grouped = group_routes(routes, 3, {3, 5, 8})
    assert [g["expert"] for g in grouped] == [3, 5]
    assert grouped[0]["routes"] == [routes[2], routes[1]]
    assert grouped[1]["routes"] == [routes[0]]
    assert sum(len(g["routes"]) for g in grouped) == len(routes)


@pytest.mark.parametrize("routes, message", [
    ([{"token": 0, "slot": 0, "expert": 3, "weight": 1}] * 2, "duplicate"),
    ([{"token": 1, "slot": 0, "expert": 3, "weight": 1}], "out-of-range"),
    ([{"token": 0, "slot": -1, "expert": 3, "weight": 1}], "out-of-range"),
    ([{"token": 0, "slot": 0, "expert": 4, "weight": 1}], "unknown"),
    ([{"token": 0, "slot": 0, "expert": 3, "weight": math.nan}], "finite"),
    ([{"token": 0.0, "slot": 0, "expert": 3, "weight": 1}], "integer"),
])
def test_invalid_routes_rejected(routes, message):
    with pytest.raises(ValueError, match=message):
        group_routes(routes, 1, {3})


def test_stable_slot_combine_and_same_expert_multiple_routes(tmp_path):
    arrays = _unit_arrays()
    arrays["routes"] = [{"token": 0, "slot": slot, "expert": 3, "weight": weight}
                        for slot, weight in enumerate((0.25, 0.75))]
    first = export_workload(tmp_path / "a", **arrays)
    arrays["routes"].reverse()
    second = export_workload(tmp_path / "b", **arrays)
    assert first["golden"]["output_bf16"] == [[0x3F3B]]
    assert first["golden"] == second["golden"]
    assert len(first["golden"]["route_contributions"]) == 2


def test_shared_only_and_unrouted_tokens(tmp_path):
    arrays = _unit_arrays()
    arrays["routes"] = []
    arrays["inputs"] = [[1.0], [0.0]]
    no_routes = export_workload(tmp_path / "empty", **arrays)
    assert no_routes["golden"]["output_bf16"] == [[0], [0]]
    shared = export_workload(tmp_path / "shared", **arrays,
                             shared_expert={"expert": 3, "weight": 1.0})
    assert shared["golden"]["output_bf16"] == [[0x3F3B], [0]]


def test_demo_manifest_roundtrip_alignment_tails_scales_and_provenance(tmp_path):
    arrays = demo_arrays()
    result = export_workload(tmp_path, **arrays,
                             provenance={"scope": "synthetic_test", "source_sha256": "test-source"})
    manifest = json.loads((tmp_path / "workload.json").read_text())
    golden = json.loads((tmp_path / "golden.json").read_text())
    hbm = (tmp_path / "weights.bin").read_bytes()
    assert manifest == result["workload"]
    assert [len(g["routes"]) for g in manifest["grouped_routes"]] == [6, 2, 1]
    assert len(hbm) % 64 == 0
    assert len(manifest["experts"]) == 4  # one deliberately unrouted expert
    scale_values = set()
    for expert in manifest["experts"]:
        for stage in ("gate", "up", "down"):
            region = expert[stage]
            assert region["element_base"] % 64 == region["scale_base"] % 64 == 0
            assert region["cols"] % 8 != 0
            assert region["rows"] % 4 != 0
            start = region["scale_base"]
            scale_values.update(hbm[start:start + region["rows"] * region["scale_row_stride"]])
    assert len(scale_values) > 1
    assert any(x != 0 for row in golden["output_bf16"] for x in row)
    assert manifest["metadata"]["hbm_sha256"] == hashlib.sha256(hbm).hexdigest()
    assert manifest["metadata"]["source_arrays_sha256"] == hashlib.sha256((tmp_path / "source_arrays.json").read_bytes()).hexdigest()
    assert golden["workload_sha256"] == hashlib.sha256((tmp_path / "workload.json").read_bytes()).hexdigest()
    assert manifest["metadata"]["provenance"]["scope"] == "synthetic_test"
    assert numerical_reference(manifest, hbm)["output_bf16"] == golden["output_bf16"]


def test_shape_and_weight_validation_before_export(tmp_path):
    arrays = _unit_arrays()
    arrays["experts"][0]["up"] = [[1.0, 2.0]]
    with pytest.raises(ValueError, match="gate/up"):
        export_workload(tmp_path, **arrays)
    assert not (tmp_path / "workload.json").exists()
    with pytest.raises(ValueError, match="non-finite"):
        encode_matrix([[math.inf]])


def test_combine_uses_fp32_and_canonical_slot_order(tmp_path):
    arrays = _unit_arrays()
    # The middle contribution is lost at this FP32 magnitude. Slot order is
    # observable; a double-precision sum or completion-order sum is incorrect.
    arrays["routes"] = [{"token": 0, "slot": slot, "expert": 3, "weight": weight}
                        for slot, weight in enumerate((2**28, 1, -(2**28)))]
    result = export_workload(tmp_path / "canonical", **arrays)
    assert result["golden"]["output_bf16"] == [[0]]
    arrays["routes"].reverse()
    reversed_input = export_workload(tmp_path / "shuffled", **arrays)
    assert reversed_input["golden"]["output_bf16"] == [[0]]
    # Actually change the slot order: large terms cancel before the small one.
    arrays["routes"][0]["slot"], arrays["routes"][1]["slot"] = 1, 2
    changed_slots = export_workload(tmp_path / "changed_slots", **arrays)
    assert changed_slots["golden"]["output_bf16"] == [[0x3F3B]]
