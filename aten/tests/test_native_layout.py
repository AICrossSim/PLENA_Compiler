import math

import pytest

from compiler.aten.plena.native_layout import (
    SequencePackingPlan,
    build_attention_head_packing,
)


@pytest.mark.parametrize("mlen,pack_factor,groups", [(512, 1, 16), (1024, 2, 8), (2048, 4, 4)])
def test_qwen32b_sequence_packing_keeps_physical_rows_constant(mlen, pack_factor, groups):
    plan = SequencePackingPlan.build(batch_size=16, seq_len=482, mlen=mlen)
    assert plan.batch_pack_factor == pack_factor
    assert plan.attention_group_count == groups
    assert plan.compile_seq_rows == 8192
    assert len(set(plan.active_physical_rows())) == 16 * 482


@pytest.mark.parametrize(
    "mlen,groups_per_block,hardware_broadcast",
    [(512, 1, 4), (1024, 1, 8), (2048, 2, 16)],
)
def test_qwen32b_compact_head_storage_stays_at_logical_q_width(
    mlen, groups_per_block, hardware_broadcast
):
    packing = build_attention_head_packing(
        mlen=mlen,
        hlen=128,
        head_dim=128,
        logical_broadcast_amount=8,
        gqa_ratio=8,
        num_kv_heads=8,
    )
    # Qwen3-32B has 64 Q heads x 128 columns = 8192 logical Q/O columns.
    assert packing.total_q_dim == 8192
    assert packing.logical_group_count == 8 * math.ceil(8 / packing.broadcast_amount)
    assert packing.groups_per_storage_block == groups_per_block
    # Vector SRAM addresses are MLEN aligned. A shared block replicates the
    # current KV head across all lanes, then selects the target group's lanes.
    assert packing.hardware_broadcast_amount == hardware_broadcast
    assert packing.heads_per_storage_block == hardware_broadcast
    assert packing.execution_head_lane_utilization == {
        512: 1.0,
        1024: 1.0,
        2048: 0.5,
    }[mlen]


def test_dummy_batch_tail_has_stable_mapping():
    plan = SequencePackingPlan.build(batch_size=3, seq_len=7, mlen=16)
    assert plan.batch_pack_factor == 2
    assert plan.padded_batch_size == 4
    assert plan.dummy_batch_count == 1
    assert plan.compile_seq_rows == 32
    assert plan.physical_row(2, 0) == 16
    assert plan.physical_row(2, 6) == 22
    assert plan.active_row_ranges() == ((0, 14), (16, 23))


def test_multi_tile_sequence_falls_back_to_one_batch_per_group():
    plan = SequencePackingPlan.build(batch_size=3, seq_len=20, mlen=16)
    assert plan.batch_pack_factor == 1
    assert plan.attention_group_count == 3
    assert plan.rows_per_attention_group == 32
    assert plan.compile_seq_rows == 96
    assert plan.active_row_ranges() == ((0, 20), (32, 52), (64, 84))


@pytest.mark.parametrize("mlen", [512, 1024, 2048])
def test_qwen32b_active_ranges_cover_exactly_logical_rows(mlen):
    plan = SequencePackingPlan.build(batch_size=16, seq_len=482, mlen=mlen)
    ranges = plan.active_row_ranges()
    assert sum(end - start for start, end in ranges) == 16 * 482
    assert all(0 <= start < end <= plan.compile_seq_rows for start, end in ranges)


def test_legacy_layout_preserves_one_group_per_batch_and_head_group():
    sequence = SequencePackingPlan.build(
        batch_size=16, seq_len=482, mlen=2048, mode="legacy"
    )
    heads = build_attention_head_packing(
        mlen=2048,
        hlen=128,
        head_dim=128,
        logical_broadcast_amount=8,
        gqa_ratio=8,
        num_kv_heads=8,
        mode="legacy",
    )
    assert sequence.compile_seq_rows == 32768
    assert heads.groups_per_storage_block == 1
    assert heads.total_q_dim == 16384


def test_compact_head_group_locations_share_storage_block_without_overlap():
    packing = build_attention_head_packing(
        mlen=16,
        hlen=4,
        head_dim=4,
        logical_broadcast_amount=2,
        gqa_ratio=2,
        num_kv_heads=2,
    )
    assert packing.attention_group_width == 8
    assert packing.groups_per_storage_block == 2
    assert packing.storage_block_count == 1
    assert packing.group_location(0) == (0, 0)
    assert packing.group_location(1) == (0, 8)
    assert packing.head_start_col(kv_head=1, local_head=1) == 12
