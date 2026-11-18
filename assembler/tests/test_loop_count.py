import unittest

from assembler.assembly_to_binary import AssemblyToBinary
from assembler.parser import Instruction


class TestLoopCount(unittest.TestCase):
    def setUp(self):
        self.asm = AssemblyToBinary("doc/operation.svh", "doc/configuration.svh")

    def test_zero_loop_count_is_rejected(self):
        # The hardware tests the count after the body, so a count of 0 would
        # wrap and run the body 2^IMM_WIDTH times.
        with self.assertRaisesRegex(ValueError, "at least 1"):
            self.asm._convert_to_binary(Instruction("C_LOOP_START", 4, None, None, None, None, None, 0))

    def test_loop_count_of_one_encodes(self):
        binary = self.asm._convert_to_binary(Instruction("C_LOOP_START", 4, None, None, None, None, None, 1))
        self.assertEqual(binary & 0x3F, self.asm.isa_definitions["C_LOOP_START"])
        self.assertEqual((binary >> 6) & 0xF, 4)  # rd
        self.assertEqual(binary >> 10, 1)         # imm


if __name__ == "__main__":
    unittest.main()
