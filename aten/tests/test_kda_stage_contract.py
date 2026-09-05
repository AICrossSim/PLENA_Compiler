"""The KDA stage-marker vocabulary is a cross-repo contract; pin both ends.

Same contract, same failure mode, as ``test_mamba_stage_contract.py``: the
compiler emits ``; @stage=<name>`` comments and the emulator's ``StageKind``
(``transactional_emulator/src/stage_profile.rs``) must know every one of them.
Markers are authoritative and sticky -- once a program contains any marker the
emulator abandons its legacy substring rules and bills every instruction to the
most recent marker. A name the compiler emits but the emulator lacks does not
error; it lands in ``unresolved_stage_markers`` and that region's cycles go to
whatever stage preceded it.

There is a sharper version of the same bug, and this file exists because the
KDA emitters shipped with it: a comment reading ``stage=kda_normalize``, with no
``@``, is not a marker at all. ``extract_stage_tag`` matches on ``"@stage="``,
so such a comment is invisible and the whole region bills to the previous
marker -- with nothing in ``unresolved_stage_markers`` to show for it. That is
why every KDA emitter routes through ``kda_stage_marker`` rather than
formatting its own string.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

from compiler.aten.plena.program_kda_common import (
    KDA_STAGE_MARKER_PREFIX,
    KDA_STAGES,
    kda_stage_marker,
)

#: Mirrors the KDA entries of ``StageKind`` in
#: ``transactional_emulator/src/stage_profile.rs``.
EMULATOR_KDA_STAGES = frozenset(
    {
        "kda_qkv_proj",
        "kda_conv1d",
        "kda_normalize",
        "kda_decay",
        "kda_state_update",
        "kda_readout",
        "kda_gated_norm",
        "kda_out_proj",
        "kda_state_load",
        "kda_state_store",
    }
)

_PLENA = Path(__file__).resolve().parents[1] / "plena"
#: Every module that emits KDA stage markers. The grep guards below are only
#: worth anything if they cover all of them -- program_kda_conv.py was added
#: after this file and was not scanned until it was listed here.
_KDA_EMITTERS = sorted(_PLENA.glob("program_kda_*.py"))


class KdaStageContractTest(unittest.TestCase):
    def test_compiler_and_emulator_vocabularies_agree(self):
        self.assertEqual(KDA_STAGES, EMULATOR_KDA_STAGES)

    def test_marker_prefix_is_the_one_the_emulator_matches(self):
        """A bare ``stage=`` comment is silently not a marker."""
        self.assertEqual(KDA_STAGE_MARKER_PREFIX, "@stage=")
        self.assertTrue(kda_stage_marker("kda_normalize").startswith("@stage="))

    def test_marker_rejects_an_unknown_stage(self):
        with self.assertRaises(ValueError):
            kda_stage_marker("kda_not_a_stage")

    def test_detail_does_not_leak_into_the_tag(self):
        """``extract_stage_tag`` takes the first whitespace-delimited token, so a
        detail string must never run into the name."""
        marker = kda_stage_marker("kda_state_load", "weights [16,8]")
        tag = marker[len(KDA_STAGE_MARKER_PREFIX) :].split()[0]
        self.assertEqual(tag, "kda_state_load")

    def test_the_guard_scans_every_kda_emitter(self):
        """A module added later is not covered until it is listed. The grep
        guards below pass vacuously on files they never read."""
        names = {p.name for p in _KDA_EMITTERS}
        self.assertIn("program_kda_common.py", names)
        self.assertIn("program_kda_conv.py", names)
        self.assertIn("program_kda_recurrent.py", names)

    def test_no_emitter_formats_a_marker_by_hand(self):
        """Every ``stage=`` in the KDA emitters must come from kda_stage_marker.

        The shipped bug was four hand-written f-strings missing the ``@``. A
        grep guard is crude, but it is the only thing that catches the fifth one.
        """
        source = "\n".join(p.read_text() for p in _KDA_EMITTERS)
        offenders = [
            line.strip()
            for line in source.splitlines()
            if "stage=" in line
            and "kda_stage_marker" not in line
            and "KDA_STAGE_MARKER_PREFIX" not in line
            and not line.lstrip().startswith(("#", "*", '"', "'"))
            and "``" not in line
        ]
        self.assertEqual(offenders, [], f"hand-formatted stage markers: {offenders}")

    def test_every_emitted_marker_name_is_in_the_vocabulary(self):
        """Catches a marker whose name was typo'd inside a kda_stage_marker call."""
        source = "\n".join(p.read_text() for p in _KDA_EMITTERS)
        names = set(re.findall(r'kda_stage_marker\(\s*"([^"]+)"', source))
        self.assertTrue(names, "no markers found; this guard would pass vacuously")
        self.assertLessEqual(names, KDA_STAGES)


if __name__ == "__main__":
    unittest.main()
