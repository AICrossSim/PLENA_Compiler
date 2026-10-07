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


if __name__ == "__main__":
    unittest.main()
