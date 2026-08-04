from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from compiler.assembler.assembly_to_binary import AssemblyToBinary
from compiler.assembler.parser import parse_asm_file
from compiler.aten.plena import PlenaCompiler


COMPILER_ROOT = Path(__file__).resolve().parents[2]


def _assemble(code: str) -> list[int]:
    assembler = AssemblyToBinary(
        str(COMPILER_ROOT / "doc" / "operation.svh"),
        str(COMPILER_ROOT / "doc" / "configuration.svh"),
    )
    with tempfile.NamedTemporaryFile("w", suffix=".asm") as handle:
        handle.write(code)
        handle.flush()
        return [
            assembler._convert_to_binary(instruction)
            for instruction in parse_asm_file(handle.name)
        ]


class SegmentedNormalizationTests(unittest.TestCase):
    def test_packed_head_equivalence_differs_from_whole_vector_norm(self) -> None:
        packed = [3.0, 4.0, 0.0, 10.0]
        gamma = [2.0, 0.5]
        epsilon = 1.0e-6
        segmented = []
        for start in range(0, len(packed), 2):
            head = packed[start:start + 2]
            scale = 1.0 / math.sqrt(
                sum(value * value for value in head) / len(head) + epsilon
            )
            segmented.extend(
                value * scale * affine
                for value, affine in zip(head, gamma)
            )

        whole_scale = 1.0 / math.sqrt(
            sum(value * value for value in packed) / len(packed) + epsilon
        )
        whole = [
            value * whole_scale * gamma[index % 2]
            for index, value in enumerate(packed)
        ]

        self.assertAlmostEqual(segmented[0], 3.0 * 2.0 / math.sqrt(12.5 + epsilon))
        self.assertAlmostEqual(segmented[3], 10.0 * 0.5 / math.sqrt(50.0 + epsilon))
        self.assertNotAlmostEqual(segmented[0], whole[0])
        self.assertNotAlmostEqual(segmented[3], whole[3])

    def test_qwen_head_norm_emits_executable_masked_sequence(self) -> None:
        program = PlenaCompiler(mlen=64, blen=8, unroll_loops=True)
        q = program.alloc(
            "q",
            1,
            128,
            strict=False,
            physical_shape=(8, 128),
        )
        q_weight = program.alloc(
            "q_norm_weight_expanded",
            1,
            128,
            strict=False,
            physical_shape=(8, 128),
        )

        program.segmented_affine_rms_norm(
            q,
            q_weight,
            segment_width=8,
            eps_offset=3,
            reci_segment_offset=6,
        )
        code = program.get_code()

        self.assertEqual(code.count("C_SET_V_MASK_REG"), 16)
        self.assertEqual(code.count("V_RED_SUM"), 16)
        self.assertEqual(
            len(
                [
                    line
                    for line in code.splitlines()
                    if line.startswith("V_MUL_VV") and line.endswith(", 1")
                ]
            ),
            16,
        )
        self.assertEqual(
            len(
                [
                    line
                    for line in code.splitlines()
                    if line.startswith("V_MUL_VF") and line.endswith(", 1")
                ]
            ),
            16,
        )
        self.assertEqual(
            len(
                [
                    line
                    for line in code.splitlines()
                    if line.startswith("V_MUL_VV") and line.endswith(", 0")
                ]
            ),
            2,
        )
        self.assertTrue(_assemble(code))

    def test_compact_affine_pattern_is_reused_per_row_and_packed_group(self) -> None:
        program = PlenaCompiler(mlen=64, blen=8, unroll_loops=True)
        q = program.alloc(
            "q",
            2,
            128,
            strict=False,
            physical_shape=(8, 128),
        )
        q_weight_pattern = program.alloc(
            "q_norm_weight_pattern",
            4,
            64,
            strict=False,
            physical_shape=(4, 64),
        )

        program.segmented_affine_rms_norm(
            q,
            q_weight_pattern,
            segment_width=8,
            eps_offset=3,
            reci_segment_offset=6,
        )
        code = program.get_code()

        # Sixteen independent heads per row produce 32 reductions. A whole-row
        # RMSNorm would emit only four reductions for this two-vector layout.
        self.assertEqual(code.count("V_RED_SUM"), 32)
        self.assertEqual(code.count("; === VRAM Broadcast Row Mul"), 1)
        self.assertEqual(
            len(
                [
                    line
                    for line in code.splitlines()
                    if line.startswith("V_MUL_VV") and line.endswith(", 0")
                ]
            ),
            4,
        )
        self.assertTrue(_assemble(code))

    def test_segmented_norm_rejects_partial_heads(self) -> None:
        program = PlenaCompiler(mlen=64, blen=8, unroll_loops=True)
        q = program.alloc(
            "q",
            1,
            120,
            strict=False,
            physical_shape=(8, 128),
        )
        q_weight = program.alloc(
            "q_weight",
            1,
            120,
            strict=False,
            physical_shape=(8, 128),
        )

        with self.assertRaisesRegex(ValueError, "complete segments"):
            program.segmented_affine_rms_norm(
                q,
                q_weight,
                segment_width=16,
                reci_segment_offset=6,
            )

    def test_affine_norm_requires_identical_backing(self) -> None:
        program = PlenaCompiler(mlen=64, blen=8, unroll_loops=True)
        activation = program.alloc(
            "activation",
            1,
            128,
            strict=False,
            physical_shape=(8, 128),
        )
        weight = program.alloc(
            "weight",
            1,
            128,
            strict=False,
            physical_shape=(16, 128),
        )

        with self.assertRaisesRegex(ValueError, "physical storage"):
            program.affine_rms_norm(activation, weight)

    def test_silu_is_an_instruction_sequence(self) -> None:
        program = PlenaCompiler(mlen=64, blen=8, unroll_loops=False)
        gate = program.alloc(
            "gate",
            1,
            256,
            strict=False,
            physical_shape=(8, 256),
        )

        program.silu(gate)
        code = program.get_code()

        self.assertIn("V_SUB_VF", code)
        self.assertIn("V_EXP_V", code)
        self.assertIn("V_RECI_V", code)
        self.assertIn("V_MUL_VV", code)
        self.assertTrue(_assemble(code))


if __name__ == "__main__":
    unittest.main()
