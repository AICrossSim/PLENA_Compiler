from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from compiler.assembler.assembly_to_binary import AssemblyToBinary
from compiler.assembler.parser import parse_asm_file
from compiler.aten.plena import PlenaCompiler
from compiler.aten.qwen_rope_validation import (
    QwenRopeValidationCost,
    permute_norm_weight,
    rotate_half,
    rotate_projection_columns,
)


COMPILER_ROOT = Path(__file__).resolve().parents[2]


def _matmul_row(
    activation: list[float],
    weight: list[list[float]],
) -> list[float]:
    return [
        sum(value * row[column] for value, row in zip(activation, weight))
        for column in range(len(weight[0]))
    ]


def _rms_norm(
    values: list[float],
    weight: list[float],
    epsilon: float,
) -> list[float]:
    scale = 1.0 / math.sqrt(
        sum(value * value for value in values) / len(values) + epsilon
    )
    return [
        value * scale * gamma
        for value, gamma in zip(values, weight)
    ]


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


class QwenRopeValidationTests(unittest.TestCase):
    def test_rotated_projection_and_permuted_norm_are_exact(self) -> None:
        activation = [0.25, -0.5, 0.75]
        weight = [
            [0.5, -0.25, 0.75, 1.0, -0.5, 0.25, -1.0, 0.125],
            [-0.25, 0.5, 0.125, -0.75, 1.0, -1.0, 0.25, 0.5],
            [1.0, 0.25, -0.5, 0.375, -0.125, 0.75, 0.5, -0.25],
        ]
        gamma = [0.75, 1.0, 1.25, 0.5, 1.5, 0.875, 1.125, 0.625]

        projected = _matmul_row(activation, weight)
        rotated_projected = _matmul_row(
            activation,
            rotate_projection_columns(weight, head_dim=8),
        )
        self.assertEqual(rotated_projected, rotate_half(projected))

        normalized = _rms_norm(projected, gamma, 1e-6)
        rotated_normalized = _rms_norm(
            rotated_projected,
            permute_norm_weight(gamma),
            1e-6,
        )
        for actual, expected in zip(
            rotated_normalized,
            rotate_half(normalized),
        ):
            self.assertAlmostEqual(actual, expected, places=14)

    def test_scaled_geometry_accounts_for_doubled_qk_path(self) -> None:
        cost = QwenRopeValidationCost(
            hidden_size=128,
            query_heads=16,
            kv_heads=2,
            head_dim=8,
        )
        metadata = cost.to_dict()

        self.assertEqual(cost.extra_weight_elements, 18_432)
        self.assertEqual(cost.extra_projection_macs_per_token, 18_432)
        self.assertEqual(cost.extra_norm_elements_per_token, 144)
        self.assertEqual(metadata["projection_multiplier_q_plus_k"], 2)
        self.assertFalse(metadata["runtime_rotated_tensor_input"])
        self.assertFalse(metadata["headline_datapath"])

    def test_scaled_q_path_assembles_without_rotated_tensor_input(self) -> None:
        program = PlenaCompiler(mlen=64, blen=8, unroll_loops=True)
        source = program.alloc(
            "source",
            1,
            128,
            strict=False,
            physical_shape=(8, 128),
        )
        weight = program.input("q_weight", shape=(128, 128))
        rotated_weight = program.input(
            "q_weight_rotate_half",
            shape=(128, 128),
        )
        q = program.linear_projection(
            source,
            weight,
            name="q",
            physical_shape=(8, 128),
        )
        q_rot = program.linear_projection(
            source,
            rotated_weight,
            name="q_rotate_half",
            physical_shape=(8, 128),
        )
        gamma = program.alloc(
            "q_norm_weight",
            1,
            128,
            strict=False,
            physical_shape=(8, 128),
        )
        gamma_rot = program.alloc(
            "q_norm_weight_rotate_half",
            1,
            128,
            strict=False,
            physical_shape=(8, 128),
        )
        for tensor, affine in ((q, gamma), (q_rot, gamma_rot)):
            program.segmented_affine_rms_norm(
                tensor,
                affine,
                segment_width=8,
                eps_offset=3,
                reci_segment_offset=6,
            )

        cos = program.alloc(
            "rope_cos",
            1,
            64,
            strict=False,
            physical_shape=(8, 64),
        )
        sin = program.alloc(
            "rope_sin",
            1,
            64,
            strict=False,
            physical_shape=(8, 64),
        )
        q_base = program.get_vram_addr(q.name)
        q_rot_base = program.get_vram_addr(q_rot.name)
        for group in range(2):
            q_group = program.alloc_at(
                f"q_group_{group}",
                1,
                64,
                q_base + group * 8 * 64,
                physical_shape=(8, 64),
            )
            q_rot_group = program.alloc_at(
                f"q_rot_group_{group}",
                1,
                64,
                q_rot_base + group * 8 * 64,
                physical_shape=(8, 64),
            )
            program.rope(q_group, q_rot_group, cos, sin)

        code = program.get_code()
        self.assertEqual(
            sum(line.startswith("M_MM ") for line in code.splitlines()),
            64,
        )
        self.assertEqual(code.count("C_SET_V_MASK_REG"), 32)
        self.assertEqual(code.count("V_RED_SUM"), 32)
        self.assertTrue(_assemble(code))


if __name__ == "__main__":
    unittest.main()
