from utils.load_config import load_svh_settings

from compiler.aten.plena.isa_matrix_view import (
    LTilePrimitive,
    MatrixViewAxis,
    encode_matrix_view_dma_word,
    encode_l_tile_exec,
    encode_l_tile_cfg,
    validate_matrix_view_dominance,
)

from .parser import load_isa_definitions, parse_asm_file


# Opcode groups for binary encoding. Module-level frozensets so they are not rebuilt
# (and scanned linearly) on every _convert_to_binary call — that runs once per emitted
# instruction (millions of times for large programs). Membership is O(1) and identical
# to the previous `opcode in [ ... ]` list literals.
_RMASK_VECTOR_OPS = frozenset(
    {
        "V_ADD_VV",
        "V_ADD_VF",
        "V_MUL_VV",
        "V_SUB_VV",
        "V_MUL_VF",
        "V_EXP_V",
        "V_RECI_V",
        "V_RED_SUM",
        "V_RED_MAX",
        "V_MAX_VF",
        "V_MIN_VF",
        "V_TOPK",
    }
)
_PSEUDO_OPCODE_ALIASES = {
    "L_TILE_CFG": "L_TILE",
    "L_TILE_EXEC": "L_TILE",
    "V_ADD_VV.MV": "V_ADD_VV",
    "V_SUB_VV.MV": "V_SUB_VV",
    "V_MUL_VV.MV": "V_MUL_VV",
    "H_PREFETCH_V.MV": "H_PREFETCH_V",
    "H_STORE_V.MV": "H_STORE_V",
}

_IMM_RS1_RD_OPS = frozenset(
    {
        "S_ADDI_INT",
        "M_MM_WO",
        "S_LD_FP",
        "S_ST_FP",
        "S_LD_INT",
        "S_ST_INT",
        "S_MAP_V_FP",
        "V_RED_MAX",
        "V_RECI_V",
        "V_EXP_V",
    }
)
_IMM_RD_OPS = frozenset({"S_LUI_INT", "M_MV_WO", "M_BMM_WO", "M_BMV_WO"})
_RS1_RD_OPS = frozenset({"S_MV_FP", "S_RECI_FP", "S_EXP_FP", "S_SQRT_FP", "V_EXP_V", "V_RED_SUM"})
_RD_ONLY_OPS = frozenset({"C_SET_SCALE_REG", "C_SET_STRIDE_REG", "C_SET_V_MASK_REG", "C_SET_TOPK_REG", "C_LOOP_END"})
_FUNCT_RSTRIDE_OPS = frozenset({"H_PREFETCH_M", "H_PREFETCH_V", "H_STORE_V", "V_SUB_VF"})
_RS2_RS1_RD_OPS = frozenset(
    {
        "S_ADD_INT",
        "S_ADD_FP",
        "S_SUB_INT",
        "S_SUB_FP",
        "S_MUL_INT",
        "S_MUL_FP",
        "S_MAX_FP",
        "M_MM",
        "M_MV",
        "M_BMM",
        "M_BMV",
        "M_TMM",
        "M_TMV",
        "M_BTMM",
        "M_BTMV",
        "C_SET_ADDR_REG",
    }
)


class AssemblyToBinary:
    def __init__(self, isa_definition_file: str, config_file: str):
        """
        Initialize the Assembler with the ISA file.

        :param isa_definition_file: Path to the ISA file
        """
        self.isa_definitions = load_isa_definitions(isa_definition_file)
        self.isa_definition_file = isa_definition_file
        config_settings = load_svh_settings(config_file)
        self.opcode_width = config_settings.get("OPCODE_WIDTH", 0)
        self.operands_width = config_settings.get("OPERAND_WIDTH", 0)
        self.imm_width = config_settings.get("IMM_WIDTH", 0)
        self.imm2_width = config_settings.get("IMM_2_WIDTH", 0)
        self.instruction_length = config_settings.get("INSTRUCTION_LENGTH", 0)
        self.funct_width = config_settings.get("FUNCT_WIDTH", 0)
        self.funct_dist = self.instruction_length - 2 * self.funct_width

    def _convert_to_binary(self, instruction):
        """
        Convert an instruction to its binary representation.

        :param instruction: Instruction object
        :return: Binary representation of the instruction
        """
        # Example conversion logic (to be replaced with actual logic)
        mnemonic = instruction.opcode
        opcode = self.isa_definitions[_PSEUDO_OPCODE_ALIASES.get(mnemonic, mnemonic)]
        rd = instruction.rd
        rs1 = instruction.rs1
        rs2 = instruction.rs2
        rstride = instruction.rstride
        funct1 = instruction.funct1
        imm = instruction.imm
        rmask = instruction.rmask
        binary_instruction = 0
        ow = self.operands_width
        opw = self.opcode_width

        if instruction.opcode in _RMASK_VECTOR_OPS and rmask is None:
            # Treat omitted rmask deterministically as "mask disabled" instead of crashing on None << ...
            rmask = 0

        if mnemonic == "L_TILE_CFG":
            # Text: L_TILE_CFG slot, gp_shape, gp_map.
            slot = rd
            shape_register = rs1
            map_register = rs2
            if slot is None or shape_register is None or map_register is None:
                raise ValueError("L_TILE_CFG requires slot, shape register, and map register")
            if not 0 <= slot < 4:
                raise ValueError(f"L_TILE_CFG slot must be in [0, 4), got {slot}")
            binary_instruction = encode_l_tile_cfg(
                slot=slot,
                shape_register=shape_register,
                map_register=map_register,
            )
        elif mnemonic == "L_TILE_EXEC":
            # Text: L_TILE_EXEC gp_dst, gp_src1, gp_scale, primitive[, axis_mask].
            # axis_mask[0] selects source columns and axis_mask[1] selects scale
            # columns.  Omission is the byte-compatible row/row form.
            if rd is None or rs1 is None or rs2 is None or rstride is None:
                raise ValueError(
                    "L_TILE_EXEC requires destination/source base registers and primitive"
                )
            try:
                primitive = LTilePrimitive(rstride)
            except ValueError as error:
                raise ValueError(f"reserved L_TILE primitive {rstride}") from error
            axis_mask = 0 if funct1 is None else funct1
            if not isinstance(axis_mask, int) or not 0 <= axis_mask <= 0b11:
                raise ValueError("L_TILE_EXEC axis mask must be in [0, 3]")
            binary_instruction = encode_l_tile_exec(
                dst_register=rd,
                src1_register=rs1,
                src2_register=rs2,
                primitive=primitive,
                source_axis=MatrixViewAxis((axis_mask >> 0) & 1),
                scale_axis=MatrixViewAxis((axis_mask >> 1) & 1),
            )
        elif mnemonic in {"H_PREFETCH_V.MV", "H_STORE_V.MV"}:
            # Existing vector DMA with an explicit Matrix-view destination or
            # source: rd, rs1, rs2, rstride, precision, view_slot.
            if None in (rd, rs1, rs2, rstride, funct1, instruction.funct2):
                raise ValueError(f"{mnemonic} requires all six operands")
            if not isinstance(funct1, int) or not 0 <= funct1 <= 2:
                raise ValueError(f"{mnemonic}: precision must be in [0, 2]")
            slot = instruction.funct2
            if not isinstance(slot, int) or not 0 <= slot < 4:
                raise ValueError(f"{mnemonic}: Matrix view slot must be in [0, 4)")
            legacy_word = (
                (funct1 << (opw + 4 * ow))
                + (rstride << (opw + 3 * ow))
                + (rs2 << (opw + 2 * ow))
                + (rs1 << (opw + ow))
                + (rd << opw)
                + opcode
            )
            binary_instruction = encode_matrix_view_dma_word(legacy_word, slot=slot)
        elif mnemonic == "M_MM_WO" and rstride is not None:
            # View-qualified existing writeback. The 18-bit immediate uses its
            # top bit as an explicit view marker and the next two bits as the
            # slot. Legacy three-operand words retain marker=0 byte-for-byte.
            if rd is None or rs1 is None or imm is None:
                raise ValueError("M_MM_WO requires base, stride register, and offset")
            if not 0 <= rstride < 4:
                raise ValueError(f"M_MM_WO Matrix view slot must be in [0, 4), got {rstride}")
            if not 0 <= imm < (1 << 15):
                raise ValueError(
                    f"view-qualified M_MM_WO offset must fit 15 bits, got {imm}"
                )
            encoded_imm = (1 << 17) | (rstride << 15) | imm
            binary_instruction = (
                (encoded_imm << (opw + 2 * ow))
                + (rs1 << (opw + ow))
                + (rd << opw)
                + opcode
            )
        elif mnemonic in _RS2_RS1_RD_OPS and mnemonic.startswith("M_") and rstride is not None:
            # A fourth Matrix operand is an explicit view slot. funct1=0 keeps
            # the legacy no-view word; codes 1..4 select slots 0..3.
            if not 0 <= rstride < 4:
                raise ValueError(f"{mnemonic}: Matrix view slot must be in [0, 4), got {rstride}")
            binary_instruction = (
                ((rstride + 1) << (opw + 4 * ow))
                + (rs2 << (opw + 2 * ow))
                + (rs1 << (opw + ow))
                + (rd << opw)
                + opcode
            )
        elif mnemonic in {"V_ADD_VV.MV", "V_SUB_VV.MV", "V_MUL_VV.MV"}:
            if not isinstance(funct1, int) or isinstance(funct1, bool):
                raise TypeError(f"{mnemonic}: Matrix-view operand mask must be an int")
            if funct1 == 0:
                raise ValueError(f"{mnemonic}: Matrix-view operand mask cannot be zero")
            if not 1 <= funct1 <= 7:
                raise ValueError(f"{mnemonic}: Matrix-view operand mask must be in [1, 7]")
            binary_instruction = (
                ((funct1 | 0x8) << (opw + 4 * ow))
                + ((rmask or 0) << (opw + 3 * ow))
                + ((rs2 or 0) << (opw + 2 * ow))
                + ((rs1 or 0) << (opw + ow))
                + ((rd or 0) << opw)
                + opcode
            )
        elif instruction.opcode in _IMM_RS1_RD_OPS:
            if mnemonic == "M_MM_WO" and not 0 <= imm < (1 << 17):
                raise ValueError(
                    "legacy M_MM_WO offset must fit 17 bits; bit 17 is the "
                    f"Matrix-view marker, got {imm}"
                )
            binary_instruction = (imm << (opw + 2 * ow)) + (rs1 << (opw + ow)) + (rd << opw) + opcode
        elif instruction.opcode in _IMM_RD_OPS:
            binary_instruction = (imm << (opw + ow)) + (rd << opw) + opcode
        elif instruction.opcode in _RS1_RD_OPS:
            binary_instruction = (rs1 << (opw + ow)) + (rd << opw) + opcode
        elif instruction.opcode == "C_BREAK":
            binary_instruction = opcode
        elif instruction.opcode in _RD_ONLY_OPS:
            binary_instruction = (rd << opw) + opcode
        elif instruction.opcode == "C_LOOP_START":
            # C_LOOP_START rd, imm - uses 22-bit immediate like S_LUI_INT
            binary_instruction = (imm << (opw + ow)) + (rd << opw) + opcode
        elif instruction.opcode in _FUNCT_RSTRIDE_OPS:
            binary_instruction = (
                (funct1 << (opw + 4 * ow))
                + (rstride << (opw + 3 * ow))
                + (rs2 << (opw + 2 * ow))
                + (rs1 << (opw + ow))
                + (rd << opw)
                + opcode
            )
        elif instruction.opcode in _RMASK_VECTOR_OPS:
            binary_instruction = (
                (rmask << (opw + 3 * ow)) + (rs2 << (opw + 2 * ow)) + (rs1 << (opw + ow)) + (rd << opw) + opcode
            )
        elif instruction.opcode in _RS2_RS1_RD_OPS:
            binary_instruction = (rs2 << (opw + 2 * ow)) + (rs1 << (opw + ow)) + (rd << opw) + opcode
        else:
            binary_instruction = (rs2 << (opw + 2 * ow)) + (rs1 << (opw + ow)) + (rd << opw) + opcode

        if binary_instruction > 0xFFFFFFFF:
            raise ValueError(
                f"Instruction encoding overflow (0x{binary_instruction:X} > 32 bits): "
                f"mnemonic={instruction.opcode}, rd={rd}, rs1={rs1}, rs2={rs2}, imm={imm}. "
                f"Use load_large_int from asm_templates._imm for immediates >= {1 << 18}."
            )
        return binary_instruction

    def write_binary_to_file(self, binary_instructions, output_file: str):
        with open(output_file, "w") as file:
            for instruction in binary_instructions:
                file.write(f"0x{instruction:08X}\n")

    def generate_binary(self, asm_file: str, output_file: str):
        """
        Generate binary instructions from the assembled instructions.
        """
        with open(asm_file) as source:
            validate_matrix_view_dominance(source.read())
        instructions = parse_asm_file(asm_file)
        binary_instructions = []
        for instruction in instructions:
            # Convert each instruction to binary format
            binary_instruction = self._convert_to_binary(instruction)
            binary_instructions.append(binary_instruction)
        # Write the binary instructions to a file
        self.write_binary_to_file(binary_instructions, output_file)
        return binary_instructions
