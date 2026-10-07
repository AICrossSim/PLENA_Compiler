import unittest

from assembler.assembly_to_binary import AssemblyToBinary
from assembler.parser import Instruction, parse_asm_file


class TestVectorRmaskHandling(unittest.TestCase):
    def setUp(self):
        self.asm = AssemblyToBinary("doc/operation.svh", "doc/configuration.svh")

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

    def test_vector_scalar_minmax_encode_like_masked_vector_ops(self):
        max_instr = Instruction("V_MAX_VF", 1, 2, 3, 0, None, None, None)
        min_instr = Instruction("V_MIN_VF", 1, 2, 3, 0, None, None, None)

        max_binary = self.asm._convert_to_binary(max_instr)
        min_binary = self.asm._convert_to_binary(min_instr)

        self.assertEqual(max_binary & 0x3F, self.asm.isa_definitions["V_MAX_VF"])
        self.assertEqual(min_binary & 0x3F, self.asm.isa_definitions["V_MIN_VF"])

    def test_v_topk_encodes_like_masked_vector_op(self):
        instr = Instruction("V_TOPK", 1, 2, 3, 0, None, None, None)
        binary = self.asm._convert_to_binary(instr)

        self.assertEqual(binary & 0x3F, self.asm.isa_definitions["V_TOPK"])
        self.assertEqual((binary >> 6) & 0xF, 1)
        self.assertEqual((binary >> 10) & 0xF, 2)
        self.assertEqual((binary >> 14) & 0xF, 3)
        self.assertEqual((binary >> 18) & 0xF, 0)


if __name__ == "__main__":
    unittest.main()
