"""The Mamba stage-marker vocabulary is a cross-repo contract; pin both ends.

The compiler emits ``; @stage=<name>`` comments and the emulator's ``StageKind``
(``transactional_emulator/src/stage_profile.rs``) must know every one of them.
Markers are *authoritative and sticky*: once a program contains any marker the
emulator abandons its legacy substring rules entirely and bills every instruction
to the most recent marker. So a name the compiler emits but the emulator lacks
does not error -- it lands in ``unresolved_stage_markers`` and that region's
cycles are attributed to whatever stage preceded it. Silent, and it looks exactly
like a correct profile.

This test owns the compiler end. The Rust test
``stage_marker_names_match_the_compiler_vocabulary`` owns the other end and
carries the same list, so a stage added on one side without the other fails a
build rather than skewing a profile.
"""

from __future__ import annotations

import unittest

from compiler.aten.plena.program_mamba_common import (
    MAMBA_STAGE_MARKER_PREFIX,
    MAMBA_STAGES,
    mamba_stage_marker,
)

#: Mirrors the Mamba entries of ``COMPILER_MOE_STAGES`` in
#: ``transactional_emulator/src/stage_profile.rs``.
EMULATOR_MAMBA_STAGES = frozenset(
    {
        "mamba_in_proj",
        "mamba_conv1d",
        "mamba_dt",
        "mamba_chunk_cumsum",
        "mamba_decay_mask",
        "mamba_intra_chunk",
        "mamba_state_update",
        "mamba_inter_chunk",
        "mamba_skip",
        "mamba_gated_norm",
        "mamba_out_proj",
        "mamba_state_load",
        "mamba_state_store",
    }
)


class TestMambaStageContract(unittest.TestCase):
    def test_vocabulary_matches_the_emulator(self):
        missing_in_emulator = MAMBA_STAGES - EMULATOR_MAMBA_STAGES
        missing_in_compiler = EMULATOR_MAMBA_STAGES - MAMBA_STAGES
        self.assertEqual(
            missing_in_emulator,
            set(),
            "compiler emits stages the emulator's StageKind does not know; they would "
            "land in unresolved_stage_markers and bill to the preceding stage",
        )
        self.assertEqual(
            missing_in_compiler,
            set(),
            "emulator declares Mamba stages no compiler marker produces; those rows "
            "can never be reached and will always read zero",
        )

    def test_marker_prefix_matches_the_routed_moe_substrate(self):
        # Both substrates share one prefix because the emulator has one parser.
        from compiler.aten.plena.program_routed_moe import MOE_STAGE_MARKER_PREFIX

        self.assertEqual(MAMBA_STAGE_MARKER_PREFIX, MOE_STAGE_MARKER_PREFIX)

    def test_marker_names_do_not_collide_with_the_moe_vocabulary(self):
        from compiler.aten.plena.program_routed_moe import MOE_STAGES

        self.assertEqual(
            MAMBA_STAGES & MOE_STAGES,
            set(),
            "a name in both vocabularies would merge Mamba and MoE cost in the profile",
        )

    def test_unknown_stage_is_rejected_at_asm_generation_time(self):
        # Failing here rather than at profile-read time is the whole point: a typo
        # otherwise produces a plausible-looking profile with a missing region.
        with self.assertRaises(ValueError):
            mamba_stage_marker("mamba_not_a_stage")

    def test_marker_format_carries_optional_detail(self):
        self.assertEqual(mamba_stage_marker("mamba_dt"), "@stage=mamba_dt")
        self.assertEqual(
            mamba_stage_marker("mamba_dt", "softplus + clamp"), "@stage=mamba_dt softplus + clamp"
        )


if __name__ == "__main__":
    unittest.main()
