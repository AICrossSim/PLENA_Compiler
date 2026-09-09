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

    def test_qwen_topk8_preserves_all_operands_and_rmask(self):
        instruction = Instruction("V_TOPK", 1, 2, 3, 1, None, None, None)
        encoded = self.asm._convert_to_binary(instruction)

        self.assertEqual(encoded & 0x3F, 0x37)
        self.assertEqual((encoded >> 6) & 0xF, 1)
        self.assertEqual((encoded >> 10) & 0xF, 2)
        self.assertEqual((encoded >> 14) & 0xF, 3)
        self.assertEqual((encoded >> 18) & 0xF, 1)

    def test_qwen_router_linear_preserves_all_operands_and_policy(self):
        instruction = Instruction(
            "V_ROUTER_LINEAR_BF16", 3, 7, 12, 1, None, None, None
        )
        encoded = self.asm._convert_to_binary(instruction)

        self.assertEqual(encoded & 0x3F, 0x36)
        self.assertEqual((encoded >> 6) & 0xF, 3)
        self.assertEqual((encoded >> 10) & 0xF, 7)
        self.assertEqual((encoded >> 14) & 0xF, 12)
        self.assertEqual((encoded >> 18) & 0xF, 1)

    def test_qwen_fp32_route_multiply_preserves_all_operands(self):
        instruction = Instruction(
            "V_MUL_ROUTE_F32", 4, 5, 6, 0, None, None, None
        )
        encoded = self.asm._convert_to_binary(instruction)

        self.assertEqual(encoded & 0x3F, 0x38)
        self.assertEqual((encoded >> 6) & 0xF, 4)
        self.assertEqual((encoded >> 10) & 0xF, 5)
        self.assertEqual((encoded >> 14) & 0xF, 6)
        self.assertEqual((encoded >> 18) & 0xF, 0)

    def test_qwen_exact_expert_combine_preserves_descriptor_addr_register(self):
        instruction = Instruction(
            "V_QWEN3_EXPERT_COMBINE_BF16", 4, 5, 6, 13, None, None, None
        )
        encoded = self.asm._convert_to_binary(instruction)

        self.assertEqual(encoded & 0x3F, 0x39)
        self.assertEqual((encoded >> 6) & 0xF, 4)
        self.assertEqual((encoded >> 10) & 0xF, 5)
        self.assertEqual((encoded >> 14) & 0xF, 6)
        self.assertEqual((encoded >> 18) & 0xF, 13)

    def test_qwen_exact_rmsnorm_preserves_policy(self):
        instruction = Instruction(
            "V_QWEN3_RMSNORM_BF16", 7, 8, 9, 1, None, None, None
        )
        encoded = self.asm._convert_to_binary(instruction)

        self.assertEqual(encoded & 0x3F, 0x3A)
        self.assertEqual((encoded >> 6) & 0xF, 7)
        self.assertEqual((encoded >> 10) & 0xF, 8)
        self.assertEqual((encoded >> 14) & 0xF, 9)
        self.assertEqual((encoded >> 18) & 0xF, 1)


if __name__ == "__main__":
    unittest.main()
