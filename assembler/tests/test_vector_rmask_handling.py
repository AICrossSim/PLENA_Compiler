import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

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

    def test_batch4_route_control_and_scale_encodings(self):
        with TemporaryDirectory() as tmpdir:
            asm_path = Path(tmpdir) / "batch4_route.asm"
            output_path = Path(tmpdir) / "batch4_route.mem"
            asm_path.write_text(
                "C_SET_TOPK_REG gp6\n"
                "C_ROUTE_BEGIN gp1, gp2, gp3, 1\n"
                "C_ROUTE_LOOP_START\n"
                "V_ROUTE_MUL gp4, gp5, gp0, 3\n"
                "C_ROUTE_LOOP_END\n"
            )

            parsed = parse_asm_file(str(asm_path))
            self.assertEqual(
                [item.opcode for item in parsed],
                [
                    "C_SET_TOPK_REG",
                    "C_ROUTE_BEGIN",
                    "C_ROUTE_LOOP_START",
                    "V_ROUTE_MUL",
                    "C_ROUTE_LOOP_END",
                ],
            )
            words = self.asm.generate_binary(str(asm_path), str(output_path))

        self.assertEqual(
            words,
            [0x000001B8, 0x0004C879, 0x0000003A, 0x000C153C, 0x0000003B],
        )

    def test_batch4_route_encoder_rejects_unsupported_fields(self):
        invalid = (
            Instruction("C_ROUTE_BEGIN", 1, 2, 3, 2, None, None, None),
            Instruction("V_ROUTE_MUL", 1, 2, 1, 0, None, None, None),
            Instruction("V_ROUTE_MUL", 1, 2, 0, 4, None, None, None),
        )

        for instruction in invalid:
            with self.subTest(instruction=instruction), self.assertRaises(ValueError):
                self.asm._convert_to_binary(instruction)


if __name__ == "__main__":
    unittest.main()
