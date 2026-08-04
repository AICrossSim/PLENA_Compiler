import unittest
import tempfile
from pathlib import Path

from assembler.assembly_to_binary import AssemblyToBinary
from assembler.parser import Instruction, parse_asm_file


class TestVectorRmaskHandling(unittest.TestCase):
    def setUp(self):
        compiler_root = Path(__file__).resolve().parents[2]
        self.asm = AssemblyToBinary(
            str(compiler_root / "doc" / "operation.svh"),
            str(compiler_root / "doc" / "configuration.svh"),
        )

    def test_parser_sets_default_rmask_for_three_operand_vector_binary(self):
        with tempfile.NamedTemporaryFile("w", suffix=".asm") as handle:
            handle.write("V_ADD_VV gp1, gp2, gp3\n")
            handle.flush()
            parsed = parse_asm_file(handle.name)

        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0].rmask, 0)

    def test_encoder_defaults_missing_rmask_to_zero(self):
        explicit_mask = Instruction("V_ADD_VV", 1, 2, 3, 0, None, None, None)
        missing_mask = Instruction("V_ADD_VV", 1, 2, 3, None, None, None, None)

        self.assertEqual(self.asm._convert_to_binary(missing_mask), self.asm._convert_to_binary(explicit_mask))

    def test_mask_register_and_reduction_rmask_encode(self):
        with tempfile.NamedTemporaryFile("w", suffix=".asm") as handle:
            handle.write("C_SET_V_MASK_REG gp4\nV_RED_SUM f2, gp3, 1\n")
            handle.flush()
            parsed = parse_asm_file(handle.name)

        self.assertEqual(parsed[0].opcode, "C_SET_V_MASK_REG")
        self.assertEqual(parsed[0].rd, 4)
        self.assertEqual(parsed[1].rmask, 1)
        self.assertEqual(self.asm._convert_to_binary(parsed[0]), (4 << 6) | 0x2E)
        self.assertNotEqual(
            self.asm._convert_to_binary(parsed[1]),
            self.asm._convert_to_binary(
                Instruction("V_RED_SUM", 2, 3, None, 0, None, None, None)
            ),
        )


if __name__ == "__main__":
    unittest.main()
