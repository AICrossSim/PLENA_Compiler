"""Guards for the whole-model Nemotron 3 physical program.

The Mamba mixers and the attention/MoE emitters are built by different
frameworks, and the seam between them is only safe if no layer leaves a register
live and both halves agree on the tile geometry. These assertions pin both, plus
the shape constraints that made naive wiring impossible.
"""

from __future__ import annotations

from functools import lru_cache

import pytest

from aten.mamba.scheduler import MambaScheduleConfig, SchedulePhase
from aten.nemotron3.program import build_nemotron3_program
from aten.nemotron3.scheduler import Nemotron3Architecture
from aten.state.hbm_image import assemble_words, render_mem_image


@lru_cache(maxsize=1)
def _program():
    return build_nemotron3_program(
        MambaScheduleConfig(phase=SchedulePhase.DECODE, decode_tokens=1)
    )


@pytest.mark.slow
def test_every_layer_is_lowered_to_real_instructions() -> None:
    program = _program()
    assert program.layer_counts == {"mamba": 23, "moe": 23, "attention": 6}
    summary = program.to_dict()
    assert summary["machine_code_bytes"] == 4 * program.instruction_count
    assert summary["machine_code_mib"] == summary["machine_code_bytes"] / (1024 * 1024)
    assert summary["assembly_text_bytes"] > summary["machine_code_bytes"]
    # Not a single stage may be missing: a semantic-only layer would show up as a
    # zero here, which is what "MLA/MoE are still events" looked like before.
    for stage, count in program.stage_instruction_counts.items():
        assert count > 0, f"stage {stage} emitted no instructions"
    assert {"block_rms_norm", "mamba_mixer", "attention", "moe", "final_rms_norm"} <= set(
        program.stage_instruction_counts
    )
    assert program.output_vram_addr is not None
    assert program.fpram_preload
    assert program.assembly.startswith(
        "; @stage=non_moe connected Nemotron pre-MoE region\n"
    )


@pytest.mark.slow
def test_spliced_mamba_matches_the_standalone_program() -> None:
    from aten.mamba.scheduler import Nemotron3MambaScheduler
    from aten.state.isa_lowering import lower_mamba_trace_to_existing_isa

    config = MambaScheduleConfig(phase=SchedulePhase.DECODE, decode_tokens=1)
    standalone = lower_mamba_trace_to_existing_isa(
        Nemotron3MambaScheduler(config).build(), mlen=64, blen=4, vlen=64
    )
    program = _program()
    # The splice must carry the Mamba blocks through unchanged; a mismatch means
    # the geometry passed to the two paths diverged.
    assert (
        program.stage_instruction_counts["mamba_mixer"] == standalone.instruction_count
    )


@pytest.mark.slow
def test_whole_program_assembles_to_legal_machine_words() -> None:
    program = _program()
    words = assemble_words(program.assembly)
    assert len(words) == program.instruction_count
    assert all(0 <= word <= 0xFFFFFFFF for word in words)
    assert all(word != 0 for word in words)
    render_mem_image(words)
    # 23 STEP + 23 consumer FENCE + one drain fence.
    assert sum(1 for word in words if word & 0x3F == 0x3D) == 47
    assert sum(1 for word in words if word & 0x3F == 0x3F) == 23


def test_tile_geometry_that_neither_half_accepts_is_rejected() -> None:
    config = MambaScheduleConfig(phase=SchedulePhase.DECODE, decode_tokens=1)
    # 128 would satisfy the fused GQA broadcast rule but not the 10,304-wide
    # Mamba projection, so one geometry cannot serve both and the build must say
    # so rather than emitting a program with mismatched addresses.
    with pytest.raises(ValueError, match="does not tile"):
        build_nemotron3_program(config, mlen=2688 // 7)


def test_prefill_is_not_claimed() -> None:
    config = MambaScheduleConfig(
        phase=SchedulePhase.PREFILL, sequence_length=128, chunk_size=128
    )
    with pytest.raises(ValueError, match="decode program"):
        build_nemotron3_program(config)


def test_context_must_fit_the_staged_kv_tile() -> None:
    config = MambaScheduleConfig(phase=SchedulePhase.DECODE, decode_tokens=1)
    with pytest.raises(ValueError, match="exceeds"):
        build_nemotron3_program(config, context_length=4096)


def test_architecture_layer_pattern_is_the_real_one() -> None:
    arch = Nemotron3Architecture()
    assert len(arch.pattern) == arch.num_layers == 52
