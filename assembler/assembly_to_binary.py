from utils.load_config import load_svh_settings
from compiler.aten.plena.mview import validate_matrix_view_dominance

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
        "V_FMA_VF",
        "V_EXP_V",
        "V_RECI_V",
        "V_RED_SUM",
        "V_RED_MAX",
        "V_MAX_VF",
        "V_MIN_VF",
        "V_TOPK",
        "V_SOFTPLUS_V",
        "V_ADD_VV.MV",
        "V_SUB_VV.MV",
        "V_MUL_VV.MV",
    }
)
# These existing arithmetic mnemonics interpret funct1[2:0] as a three-slot
# L-Compute consumer-view mask. funct1[3] is the arithmetic-variant bit used by
# the V_FMA_VF pseudo-op. Other vector opcodes retain their existing funct1
# meaning (for example V_SUB_VF uses it for operand order).
_LSTREAM_VIEW_OPS = frozenset(
    {
        "V_ADD_VV",
        "V_ADD_VF",
        "V_MUL_VV",
        "V_SUB_VV",
        "V_MUL_VF",
        "V_FMA_VF",
        "V_EXP_V",
        "V_RECI_V",
        "V_RED_SUM",
        "V_RED_MAX",
        "V_MAX_VF",
        "V_MIN_VF",
        "V_SOFTPLUS_V",
        "V_ADD_VV.MV",
        "V_SUB_VV.MV",
        "V_MUL_VV.MV",
    }
)
_PSEUDO_OPCODE_ALIASES = {
    "V_FMA_VF": "V_MUL_VF",
    "L_CFG": "L_MVIEW",
    "L_MVIEW_FULL": "L_MVIEW",
    "L_MVIEW_FIELD": "L_MVIEW",
    "V_ADD_VV.MV": "V_ADD_VV",
    "V_SUB_VV.MV": "V_SUB_VV",
    "V_MUL_VV.MV": "V_MUL_VV",
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
        "S_MAP_FP_V",
    }
)
_IMM_RD_OPS = frozenset({"S_LUI_INT", "M_MV_WO", "M_BMM_WO", "M_BMV_WO"})
_RS1_RD_OPS = frozenset({"S_MV_FP", "S_RECI_FP", "S_EXP_FP", "S_SQRT_FP"})
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
        physical_mnemonic = _PSEUDO_OPCODE_ALIASES.get(mnemonic, mnemonic)
        opcode = self.isa_definitions[physical_mnemonic]
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

        if mnemonic in _RMASK_VECTOR_OPS and rmask is None:
            # Treat omitted rmask deterministically as "mask disabled" instead of crashing on None << ...
            rmask = 0

        if mnemonic in _RMASK_VECTOR_OPS:
            # funct1 is part of the existing vector encoding. Omitted funct1
            # remains canonical zero and therefore byte-identical to the legacy
            # form. L-Compute-capable opcodes interpret it as an explicit slot
            # mask; other opcodes retain their pre-existing meaning.
            if funct1 is None:
                funct1 = 0
            elif not isinstance(funct1, int) or isinstance(funct1, bool):
                raise TypeError(
                    f"{mnemonic}: funct1 must be an int, "
                    f"got {funct1!r}"
                )
            elif not 0 <= funct1 <= 0xF:
                raise ValueError(
                    f"{mnemonic}: funct1 {funct1} outside 0..15"
                )
            elif funct1 != 0 and mnemonic not in _LSTREAM_VIEW_OPS:
                raise ValueError(
                    f"{mnemonic}: nonzero funct1 is not a supported "
                    "L-Compute view mask"
                )
            elif mnemonic in _LSTREAM_VIEW_OPS and funct1 > 0x7:
                raise ValueError(
                    f"{mnemonic}: L-Compute consumer mask {funct1} uses reserved "
                    "funct1[3]; only slots 0..2 are selectable"
                )

        # V_FMA_VF is an assembly-level alias for the model-independent multiply-
        # accumulate variant of V_MUL_VF. Keeping the alias makes generated code
        # readable without spending a physical opcode. The high funct1 bit is
        # injected only here, so spelling V_MUL_VF with a non-canonical mode bit
        # cannot silently change its arithmetic semantics.
        encoded_funct1 = funct1
        if mnemonic.endswith(".MV"):
            if funct1 == 0:
                raise ValueError(f"{mnemonic}: Matrix-view operand mask cannot be zero")
            # funct1[3] is an explicit Matrix-view addressing marker for the
            # VV family. funct1[2:0] select dst/src1/src2 view slots. The
            # physical arithmetic opcode is unchanged and legacy words retain
            # funct1=0 byte-for-byte.
            encoded_funct1 = funct1 | 0x8
        elif mnemonic == "V_FMA_VF":
            encoded_funct1 = funct1 | 0x8

        # _RMASK_VECTOR_OPS MUST be tested before _IMM_RS1_RD_OPS / _RS1_RD_OPS.
        # V_EXP_V, V_RECI_V, V_RED_MAX and V_RED_SUM appear in more than one set,
        # and the earlier branch encodes the third operand as `imm` at bit
        # OPCODE_WIDTH + 2*OPERAND_WIDTH (the rs2 field) while the emulator reads
        # rmask from rs3 at OPCODE_WIDTH + 3*OPERAND_WIDTH (op.rs `Opcode::decode`).
        # With the old ordering a masked V_EXP_V silently executed on the whole
        # tile: no diagnostic, wrong answer. rmask == 0 encodes identically under
        # either ordering, so this is a no-op for every unmasked call site.
        if mnemonic == "L_MVIEW_FULL":
            # Text: L_MVIEW_FULL slot, gp_shape, gp_map.
            slot = rd
            shape_register = rs1
            map_register = rs2
            if slot is None or shape_register is None or map_register is None:
                raise ValueError("L_MVIEW_FULL requires slot, shape register, and map register")
            if not 0 <= slot < 4:
                raise ValueError(f"L_MVIEW_FULL slot must be in [0, 4), got {slot}")
            binary_instruction = (
                opcode
                + (shape_register << opw)
                + (map_register << (opw + ow))
                + (slot << (opw + 2 * ow))
                + (1 << (opw + 4 * ow))
            )
        elif mnemonic == "L_MVIEW_FIELD":
            # Text: L_MVIEW_FIELD slot, field, gp_value.
            slot = rd
            field = imm
            value_register = rs2
            if slot is None or field is None or value_register is None:
                raise ValueError("L_MVIEW_FIELD requires slot, field, and value register")
            if not 0 <= slot < 4:
                raise ValueError(f"L_MVIEW_FIELD slot must be in [0, 4), got {slot}")
            if not 0 <= field < 3:
                raise ValueError(f"L_MVIEW_FIELD field must be in [0, 3), got {field}")
            binary_instruction = (
                opcode
                + (value_register << opw)
                + (field << (opw + ow))
                + (slot << (opw + 2 * ow))
                + (2 << (opw + 4 * ow))
            )
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
        elif mnemonic in _RMASK_VECTOR_OPS or mnemonic.endswith(".MV"):
            # 2- and 3-operand forms leave rs1/rs2 unset; the hardware reads those
            # fields regardless, so encode them as 0 rather than crashing on None.
            binary_instruction = (
                (encoded_funct1 << (opw + 4 * ow))
                + (rmask << (opw + 3 * ow))
                + ((rs2 or 0) << (opw + 2 * ow))
                + ((rs1 or 0) << (opw + ow))
                + ((rd or 0) << opw)
                + opcode
            )
        elif mnemonic in _IMM_RS1_RD_OPS:
            if mnemonic == "M_MM_WO" and not 0 <= imm < (1 << 17):
                raise ValueError(
                    "legacy M_MM_WO offset must fit 17 bits; bit 17 is the "
                    f"Matrix-view marker, got {imm}"
                )
            binary_instruction = (imm << (opw + 2 * ow)) + (rs1 << (opw + ow)) + (rd << opw) + opcode
        elif mnemonic in _IMM_RD_OPS:
            binary_instruction = (imm << (opw + ow)) + (rd << opw) + opcode
        elif mnemonic in _RS1_RD_OPS:
            binary_instruction = (rs1 << (opw + ow)) + (rd << opw) + opcode
        elif mnemonic == "C_BREAK":
            binary_instruction = opcode
        elif mnemonic in _RD_ONLY_OPS:
            binary_instruction = (rd << opw) + opcode
        elif mnemonic == "C_LOOP_START":
            # C_LOOP_START rd, imm - uses 22-bit immediate like S_LUI_INT
            binary_instruction = (imm << (opw + ow)) + (rd << opw) + opcode
        elif mnemonic == "L_CFG":
            # L_CFG value_gp, target, slot, field
            # Parser stores the numeric slot in imm and the fourth operand in
            # rstride. Bits [31:22] are canonical zero.
            slot = imm
            field = rstride
            if rd is None or rs1 is None or slot is None or field is None:
                raise ValueError(
                    "L_CFG requires value register, target register, slot, and field"
                )
            if not 0 <= slot < 4:
                raise ValueError(f"L_CFG slot must be in [0, 4), got {slot}")
            if not 0 <= field < 16:
                raise ValueError(f"L_CFG field must be in [0, 16), got {field}")
            binary_instruction = (
                (field << (opw + 3 * ow))
                + (slot << (opw + 2 * ow))
                + (rs1 << (opw + ow))
                + (rd << opw)
                + opcode
            )
        elif mnemonic in _FUNCT_RSTRIDE_OPS:
            binary_instruction = (
                (funct1 << (opw + 4 * ow))
                + (rstride << (opw + 3 * ow))
                + (rs2 << (opw + 2 * ow))
                + (rs1 << (opw + ow))
                + (rd << opw)
                + opcode
            )
        elif mnemonic in _RS2_RS1_RD_OPS:
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
