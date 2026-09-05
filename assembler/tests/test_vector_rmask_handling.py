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
        # Distinct rd/rs1/rs2 and a non-zero rmask so an operand-swap or a
        # dropped rmask lane would change the encoding and fail the test.
        max_instr = Instruction("V_MAX_VF", 1, 2, 3, 1, None, None, None)
        min_instr = Instruction("V_MIN_VF", 1, 2, 3, 1, None, None, None)

        max_binary = self.asm._convert_to_binary(max_instr)
        min_binary = self.asm._convert_to_binary(min_instr)

        for binary, name in ((max_binary, "V_MAX_VF"), (min_binary, "V_MIN_VF")):
            self.assertEqual(binary & 0x3F, self.asm.isa_definitions[name])
            self.assertEqual((binary >> 6) & 0xF, 1)   # rd
            self.assertEqual((binary >> 10) & 0xF, 2)  # rs1
            self.assertEqual((binary >> 14) & 0xF, 3)  # rs2
            self.assertEqual((binary >> 18) & 0xF, 1)  # rmask

    def test_masked_unary_and_reduction_ops_place_rmask_in_rs3(self):
        """V_EXP_V / V_RECI_V / V_RED_SUM / V_RED_MAX must carry rmask in the rs3 field.

        These four opcodes used to sit in `_IMM_RS1_RD_OPS` / `_RS1_RD_OPS` as well
        as `_RMASK_VECTOR_OPS`, and the encoder tested the former first. The mask
        therefore landed at bit 14 (the rs2 field) while the emulator reads it from
        bit 18 (rs3, see `Opcode::decode` in transactional_emulator/src/op.rs), so a
        masked V_EXP_V silently ran on the whole tile. Regression guard for that.
        """
        for name in ("V_EXP_V", "V_RECI_V", "V_RED_SUM", "V_RED_MAX"):
            with self.subTest(opcode=name):
                instr = Instruction(name, 1, 2, None, 5, None, None, None)
                binary = self.asm._convert_to_binary(instr)

                self.assertEqual(binary & 0x3F, self.asm.isa_definitions[name])
                self.assertEqual((binary >> 6) & 0xF, 1)   # rd
                self.assertEqual((binary >> 10) & 0xF, 2)  # rs1
                self.assertEqual((binary >> 18) & 0xF, 5)  # rmask lands in rs3

    def test_unmasked_encoding_is_unchanged_by_the_rmask_ordering_fix(self):
        """rmask == 0 must encode identically under the old and new branch order.

        This is what makes the ordering fix safe for every existing unmasked call
        site: the only bits that move are the ones that were previously wrong.
        """
        for name in ("V_EXP_V", "V_RECI_V", "V_RED_SUM", "V_RED_MAX"):
            with self.subTest(opcode=name):
                opcode = self.asm.isa_definitions[name]
                instr = Instruction(name, 3, 7, None, 0, None, None, None)
                expected = (7 << 10) + (3 << 6) + opcode
                self.assertEqual(self.asm._convert_to_binary(instr), expected)

    def test_masked_three_operand_unary_round_trips_through_the_parser(self):
        """`V_EXP_V gp1, gp2, 1` must reach the encoder as rmask=1, not imm=1."""
        asm_path = "/tmp/plena_test_masked_unary.asm"
        with open(asm_path, "w") as f:
            f.write("V_EXP_V gp1, gp2, 1\n")

        parsed = parse_asm_file(asm_path)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0].rmask, 1)

        binary = self.asm._convert_to_binary(parsed[0])
        self.assertEqual((binary >> 18) & 0xF, 1)

    def test_v_softplus_encodes_like_masked_unary_op(self):
        instr = Instruction("V_SOFTPLUS_V", 1, 2, None, 1, None, None, None)
        binary = self.asm._convert_to_binary(instr)

        self.assertEqual(binary & 0x3F, self.asm.isa_definitions["V_SOFTPLUS_V"])
        self.assertEqual((binary >> 6) & 0xF, 1)   # rd
        self.assertEqual((binary >> 10) & 0xF, 2)  # rs1
        self.assertEqual((binary >> 18) & 0xF, 1)  # rmask

    def test_s_map_fp_v_encodes_like_its_mirror_s_map_v_fp(self):
        """S_MAP_FP_V is the exact inverse of S_MAP_V_FP and shares its operand shape."""
        forward = Instruction("S_MAP_V_FP", 1, 2, None, None, None, None, 4)
        inverse = Instruction("S_MAP_FP_V", 1, 2, None, None, None, None, 4)

        fwd_binary = self.asm._convert_to_binary(forward)
        inv_binary = self.asm._convert_to_binary(inverse)

        # Same field placement, different opcode.
        self.assertEqual(fwd_binary >> 6, inv_binary >> 6)
        self.assertEqual(inv_binary & 0x3F, self.asm.isa_definitions["S_MAP_FP_V"])
        self.assertEqual((inv_binary >> 6) & 0xF, 1)    # rd  (FPRAM base gp reg)
        self.assertEqual((inv_binary >> 10) & 0xF, 2)   # rs1 (VRAM row gp reg)
        self.assertEqual((inv_binary >> 14) & 0x3FFFF, 4)  # imm

    def test_v_topk_encodes_like_masked_vector_op(self):
        # Non-zero rmask so the rmask-lane assertion actually exercises the field
        # (with rmask=0 it would pass even if the encoder dropped the lane).
        instr = Instruction("V_TOPK", 1, 2, 3, 1, None, None, None)
        binary = self.asm._convert_to_binary(instr)

        self.assertEqual(binary & 0x3F, self.asm.isa_definitions["V_TOPK"])
        self.assertEqual((binary >> 6) & 0xF, 1)   # rd
        self.assertEqual((binary >> 10) & 0xF, 2)  # rs1
        self.assertEqual((binary >> 14) & 0xF, 3)  # rs2
        self.assertEqual((binary >> 18) & 0xF, 1)  # rmask


if __name__ == "__main__":
    unittest.main()
