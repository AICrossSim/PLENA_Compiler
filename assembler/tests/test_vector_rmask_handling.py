import unittest
from pathlib import Path

from assembler.assembly_to_binary import AssemblyToBinary
from assembler.parser import Instruction, parse_asm_file


class TestVectorRmaskHandling(unittest.TestCase):
    def setUp(self):
        compiler_root = Path(__file__).resolve().parents[2]
        self.asm = AssemblyToBinary(
            str(compiler_root / "doc/operation.svh"),
            str(compiler_root / "doc/configuration.svh"),
        )

    def test_parser_sets_default_rmask_for_three_operand_vector_binary(self):
        asm_path = "/tmp/plena_test_vector_binary_missing_rmask.asm"
        with open(asm_path, "w") as f:
            f.write("V_ADD_VV gp1, gp2, gp3\n")

        parsed = parse_asm_file(asm_path)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0].rmask, 0)

    def test_encoder_defaults_missing_rmask_to_zero(self):
        explicit_mask = Instruction("V_ADD_VV", 1, 2, 3, 0, None, None, None)
        missing_mask = Instruction("V_ADD_VV", 1, 2, 3, None, None, None, None)

        self.assertEqual(self.asm._convert_to_binary(missing_mask), self.asm._convert_to_binary(explicit_mask))

    def test_segment_reduction_preserves_segment_fields(self):
        instruction = Instruction("V_RED_SUM_SEG", 1, 2, 3, 7, None, None, None)
        encoded = self.asm._convert_to_binary(instruction)

        self.assertEqual(encoded & 0x3F, 0x35)
        self.assertEqual((encoded >> 6) & 0xF, 1)
        self.assertEqual((encoded >> 10) & 0xF, 2)
        self.assertEqual((encoded >> 14) & 0xF, 3)
        self.assertEqual((encoded >> 18) & 0xF, 7)

    def test_scalar_extension_uses_unary_encoding(self):
        instruction = Instruction("S_RSQRT_FP", 5, 6, None, None, None, None, None)
        encoded = self.asm._convert_to_binary(instruction)

        self.assertEqual(encoded & 0x3F, 0x38)
        self.assertEqual((encoded >> 6) & 0xF, 5)
        self.assertEqual((encoded >> 10) & 0xF, 6)

    def test_compact_stat_alias_uses_vseg_compact_encoding(self):
        asm_path = "/tmp/plena_test_compact_stats.asm"
        with open(asm_path, "w") as f:
            f.write("V_STAT_RSQRT gp1, gp2, f0, 16\n")

        instruction = parse_asm_file(asm_path)[0]
        encoded = self.asm._convert_to_binary(instruction)
        self.assertEqual(encoded & 0x3F, 0x3B)
        self.assertEqual((encoded >> 18) & 0xF, 15)
        self.assertEqual((encoded >> 22) & 0xF, 0xA)

    def test_extended_compact_stat_alias_uses_log2_tier_encoding(self):
        for lanes, encoded_log2 in ((32, 5), (64, 6)):
            with self.subTest(lanes=lanes):
                asm_path = f"/tmp/plena_test_compact_stats_{lanes}.asm"
                with open(asm_path, "w") as f:
                    f.write(f"V_STAT_RSQRT gp1, gp2, f0, {lanes}\n")

                instruction = parse_asm_file(asm_path)[0]
                encoded = self.asm._convert_to_binary(instruction)
                self.assertEqual((encoded >> 18) & 0xF, encoded_log2)
                self.assertEqual((encoded >> 22) & 0xF, 0xE)

    def test_compact_stat_alias_rejects_unsupported_extended_counts(self):
        for lanes in (17, 24, 48, 65):
            with self.subTest(lanes=lanes):
                asm_path = f"/tmp/plena_test_compact_stats_invalid_{lanes}.asm"
                with open(asm_path, "w") as f:
                    f.write(f"V_STAT_RSQRT gp1, gp2, f0, {lanes}\n")
                with self.assertRaisesRegex(ValueError, "compact-stat"):
                    parse_asm_file(asm_path)

    def test_reduction_overwrite_alias_sets_funct_bit(self):
        asm_path = "/tmp/plena_test_reduction_overwrite.asm"
        with open(asm_path, "w") as f:
            f.write("V_RED_SUM_SEG_OVR f1, gp2, gp3, 7\n")

        instruction = parse_asm_file(asm_path)[0]
        encoded = self.asm._convert_to_binary(instruction)
        self.assertEqual(encoded & 0x3F, 0x35)
        self.assertEqual((encoded >> 22) & 0x1, 1)

    def test_full_reduction_overwrite_alias_sets_funct_bit(self):
        for mnemonic in ("V_RED_SUM_OVR", "V_RED_MAX_OVR"):
            with self.subTest(mnemonic=mnemonic):
                asm_path = f"/tmp/plena_test_{mnemonic.lower()}.asm"
                with open(asm_path, "w") as f:
                    f.write(f"{mnemonic} f1, gp2, 0\n")

                instruction = parse_asm_file(asm_path)[0]
                encoded = self.asm._convert_to_binary(instruction)
                self.assertEqual((encoded >> 22) & 0x1, 1)
                self.assertEqual((encoded >> 18) & 0xF, 0)


if __name__ == "__main__":
    unittest.main()
