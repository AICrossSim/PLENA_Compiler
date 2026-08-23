"""Guards for the connected Kimi K3 lowering boundary, MLA included.

MLA was the last mixer with no emitter. It needs no new attention kernel -- a
192-wide key and a 128-wide value are ordinary -- but it does need the low-rank
chain in front of one, and it has to share a tile geometry with the KDA mixers it
is interleaved with.
"""

from __future__ import annotations

import pytest

from aten.kda.scheduler import KIMI_K3_KDA_LAYERS, KdaScheduleConfig
from aten.kimi3.program import MlaWidths, build_kimi_k3_program, mla_layer_ids
from aten.kimi3.scheduler import KimiK3Architecture
from aten.mamba.scheduler import SchedulePhase
from aten.plena import validate_symbolic_hbm_bindings


def test_mla_layers_are_the_complement_of_the_kda_schedule() -> None:
    # KIMI_K3_KDA_LAYERS enumerates layers 1..92, so layer 93 is not a multiple
    # of four. Deriving MLA from its own "every fourth layer" rule drops it and
    # yields 23 instead of 24, and then asks for KDA assembly the schedule never
    # emitted.
    arch = KimiK3Architecture()
    mla = mla_layer_ids(arch.num_layers)
    assert len(mla) == 24
    assert len(KIMI_K3_KDA_LAYERS) == 69
    assert len(mla) + len(KIMI_K3_KDA_LAYERS) == arch.num_layers == 93
    assert not set(mla) & set(KIMI_K3_KDA_LAYERS)
    assert arch.num_layers - 1 in mla


def test_every_mla_width_shares_the_kda_tile() -> None:
    # Unlike Nemotron's fused GQA path, whose broadcast factor forces mlen to be
    # a multiple of the 128-wide head and so collides with the 10,304-wide Mamba
    # projection, MLA has no width that fights the 64-wide tile.
    widths = MlaWidths.from_architecture(KimiK3Architecture())
    assert widths.heads == 96
    assert widths.qk_head == 192
    assert widths.kv_a_out == 576
    assert widths.unaligned(64) == []


def test_a_tile_that_splits_a_head_is_rejected() -> None:
    config = KdaScheduleConfig(phase=SchedulePhase.DECODE, decode_tokens=1)
    with pytest.raises(ValueError, match="does not tile"):
        build_kimi_k3_program(config, mlen=128, heads=2)


def test_prefill_is_not_claimed() -> None:
    config = KdaScheduleConfig(
        phase=SchedulePhase.PREFILL, sequence_length=16, chunk_size=16
    )
    with pytest.raises(ValueError, match="decode program"):
        build_kimi_k3_program(config)


def test_context_must_fit_the_staged_kv_tile() -> None:
    config = KdaScheduleConfig(phase=SchedulePhase.DECODE, decode_tokens=1)
    with pytest.raises(ValueError, match="exceeds"):
        build_kimi_k3_program(config, context_length=4096, heads=2)


@pytest.mark.slow
def test_full_scale_connected_program_fits_the_machine_code_budget() -> None:
    config = KdaScheduleConfig(phase=SchedulePhase.DECODE, decode_tokens=1)
    program = build_kimi_k3_program(config)

    validate_symbolic_hbm_bindings(program)
    assert program.layer_counts == {
        "kda": 69,
        "mla": 24,
        "latent_moe": 92,
        "dense_ffn": 1,
    }
    assert program.instruction_count * 4 <= 64 * 1024 * 1024
    assert len(program.symbolic_hbm_bindings) == 2713
    assert all(count > 0 for count in program.stage_instruction_counts.values())
