from __future__ import annotations

import pytest

from compiler.aten.plena.kv_residency import (
    MATRIX_SRAM_POLICIES,
    derive_matrix_sram_policy,
    plan_kv_residency,
)


@pytest.mark.parametrize(
    ("tiles", "resident"),
    [(2, 0), (4, 1), (66, 32), (256, 128)],
)
def test_raw_capacity_uses_safe_prefix_and_stream_slots(
    tiles: int, resident: int
) -> None:
    plan = plan_kv_residency(
        k_blocks=128,
        mlen=512,
        matrix_sram_tiles=tiles,
    )
    assert plan.resident_prefix_blocks == resident
    assert plan.peak_live_tiles <= tiles
    addresses = [
        *plan.resident_k_addresses,
        *plan.resident_v_addresses,
        *(
            []
            if plan.stream_k_address is None
            else [plan.stream_k_address, plan.stream_v_address]
        ),
    ]
    assert len(addresses) == len(set(addresses))
    assert all(address % (512 * 512) == 0 for address in addresses)


def test_causal_load_formula_and_monotonicity() -> None:
    loads = []
    for policy in MATRIX_SRAM_POLICIES:
        plan = derive_matrix_sram_policy(
            policy=policy,
            k_blocks=128,
            mlen=512,
            projection_tiles=16,
        )
        loads.append(
            plan.expected_tile_loads(q_blocks=128, causal=True)
        )
        expected = 2 * plan.resident_prefix_blocks + 2 * sum(
            max(0, min(q + 1, 128) - plan.resident_prefix_blocks)
            for q in range(128)
        )
        assert loads[-1] == expected
    assert loads[0] >= loads[2] >= loads[3] >= loads[4] >= loads[5]
    assert loads[-1] == 256


def test_cache_hits_count_resident_prefix_demands() -> None:
    plan = plan_kv_residency(
        k_blocks=3,
        mlen=16,
        matrix_sram_tiles=4,
    )
    metadata = plan.metadata(q_blocks=3, causal=True)
    assert plan.resident_prefix_blocks == 1
    assert metadata["kv_cache_hits"] == 6
    assert metadata["kv_cache_misses"] == 8
    assert metadata["average_live_tiles"] == pytest.approx(3.0)


def test_full_and_streaming_are_binary_endpoints() -> None:
    streaming = plan_kv_residency(
        k_blocks=8,
        mlen=64,
        matrix_sram_tiles=2,
    )
    resident = plan_kv_residency(
        k_blocks=8,
        mlen=64,
        matrix_sram_tiles=16,
    )
    assert streaming.resident_prefix_blocks == 0
    assert streaming.stream_k_address == 0
    assert streaming.stream_v_address == 64 * 64
    assert resident.full_resident
    assert resident.stream_k_address is None
    assert resident.expected_tile_loads(q_blocks=8, causal=True) == 16


def test_projection_policy_uses_spare_capacity_opportunistically() -> None:
    plan = derive_matrix_sram_policy(
        policy="projection-full",
        k_blocks=8,
        mlen=64,
        projection_tiles=10,
    )
    assert plan.matrix_sram_tiles == 10
    assert plan.resident_prefix_blocks == 4
    assert plan.realized_residency_fraction == 0.5
