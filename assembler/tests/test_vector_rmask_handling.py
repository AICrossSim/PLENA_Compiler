import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from assembler.assembly_to_binary import AssemblyToBinary
from assembler.parser import Instruction, parse_asm_file


REPO_ROOT = Path(__file__).resolve().parents[2]


class TestVectorRmaskHandling(unittest.TestCase):
    def setUp(self):
        self.asm = AssemblyToBinary(
            str(REPO_ROOT / "doc" / "operation.svh"),
            str(REPO_ROOT / "doc" / "configuration.svh"),
        )

    def test_parser_sets_default_rmask_for_three_operand_vector_binary(self):
        with TemporaryDirectory() as tmpdir:
            asm_path = Path(tmpdir) / "vector_binary_missing_rmask.asm"
            asm_path.write_text("V_ADD_VV gp1, gp2, gp3\n")

            parsed = parse_asm_file(str(asm_path))
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0].rmask, 0)

    def test_encoder_defaults_missing_rmask_to_zero(self):
        explicit_mask = Instruction("V_ADD_VV", 1, 2, 3, 0, None, None, None)
        missing_mask = Instruction("V_ADD_VV", 1, 2, 3, None, None, None, None)

        self.assertEqual(
            self.asm._convert_to_binary(missing_mask),
            self.asm._convert_to_binary(explicit_mask),
        )

    def test_vector_scalar_minmax_encode_like_masked_vector_ops(self):
        # Distinct rd/rs1/rs2 and a non-zero rmask so an operand-swap or a
        # dropped rmask lane would change the encoding and fail the test.
        max_instr = Instruction("V_MAX_VF", 1, 2, 3, 1, None, None, None)
        min_instr = Instruction("V_MIN_VF", 1, 2, 3, 1, None, None, None)

        max_binary = self.asm._convert_to_binary(max_instr)
        min_binary = self.asm._convert_to_binary(min_instr)

        for binary, name in ((max_binary, "V_MAX_VF"), (min_binary, "V_MIN_VF")):
            self.assertEqual(binary & 0x3F, self.asm.isa_definitions[name])
            self.assertEqual((binary >> 6) & 0xF, 1)  # rd
            self.assertEqual((binary >> 10) & 0xF, 2)  # rs1
            self.assertEqual((binary >> 14) & 0xF, 3)  # rs2
            self.assertEqual((binary >> 18) & 0xF, 1)  # rmask

    def test_v_topk_encodes_like_masked_vector_op(self):
        # These exact words are shared ABI fixtures with PLENA_Simulator and RTL.
        # rd=gp1, rs1=gp2, rs2=gp3; rmask selects the routing policy.
        fixtures = ((0, 0x0000C877), (1, 0x0004C877))

        for rmask, expected in fixtures:
            instr = Instruction("V_TOPK", 1, 2, 3, rmask, None, None, None)
            binary = self.asm._convert_to_binary(instr)

            self.assertEqual(binary, expected)
            self.assertEqual(binary & 0x3F, self.asm.isa_definitions["V_TOPK"])
            self.assertEqual((binary >> 6) & 0xF, 1)  # rd
            self.assertEqual((binary >> 10) & 0xF, 2)  # rs1
            self.assertEqual((binary >> 14) & 0xF, 3)  # rs2
            self.assertEqual((binary >> 18) & 0xF, rmask)

    def test_v_topk_text_parser_preserves_policy_rmask(self):
        with TemporaryDirectory() as tmpdir:
            asm_path = Path(tmpdir) / "topk.asm"
            output_path = Path(tmpdir) / "topk.mem"
            asm_path.write_text("V_TOPK gp1, gp2, gp3, 0\nV_TOPK gp1, gp2, gp3, 1\n")

            parsed = parse_asm_file(str(asm_path))
            self.assertEqual([instruction.rmask for instruction in parsed], [0, 1])
            words = self.asm.generate_binary(str(asm_path), str(output_path))
            self.assertEqual(words, [0x0000C877, 0x0004C877])


if __name__ == "__main__":
    unittest.main()
