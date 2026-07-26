from __future__ import annotations

import re


def load_isa_definitions(file_path: str) -> dict:
    """
    Parse a SystemVerilog enum from a .svh file and return it as a dictionary.
    """
    enum_dict = {}
    inside_enum = False
    pattern = re.compile(r"(\w+)\s*=\s*(\d+)\'h([0-9A-Fa-f]+)")

    with open(file_path) as f:
        for line in f:
            line = line.strip()

            # Detect the start of the enum
            if line.startswith("typedef enum") and "OPCODE_WIDTH" in line:
                inside_enum = True
                continue

            if inside_enum:
                # End of enum
                if line.endswith("} CUSTOM_ISA_OPCODE;"):
                    break

                # Match line like: S_ADD_FP = 6'h0E,
                match = pattern.search(line)
                if match:
                    name = match.group(1)
                    value = int(match.group(3), 16)
                    enum_dict[name] = value

    # Mainline operation.svh currently names opcode 0x31 V_PS_V, while the
    # compiler templates and transactional emulator still use V_SHIFT_V for the
    # same vector-shift instruction.
    if "V_SHIFT_V" not in enum_dict and "V_PS_V" in enum_dict:
        enum_dict["V_SHIFT_V"] = enum_dict["V_PS_V"]
    if "C_AGU_CONFIG" in enum_dict:
        enum_dict["C_AGU_BIND"] = enum_dict["C_AGU_CONFIG"]
        enum_dict["C_AGU_LOOP_LEN"] = enum_dict["C_AGU_CONFIG"]

    return enum_dict


def load_isa_settings(file_path: str) -> dict:
    param_pattern = re.compile(r"parameter\s+(\w+)\s*=\s*([^;]+);")
    param_dict = {}
    isa_settings_param = ["OPERAND_WIDTH", "OPCODE_WIDTH", "IMM_WIDTH", "IMM_2_WIDTH"]
    # First pass: collect simple constant values
    with open(file_path) as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if line.startswith("//") or not line or "parameter" not in line:
            continue

        match = param_pattern.match(line)
        if match:
            key = match.group(1)
            value = match.group(2).strip()

            if key not in isa_settings_param:
                continue

            # Try to resolve constant integer values
            try:
                param_dict[key] = int(value)
            except ValueError:
                param_dict[key] = value  # Expression, to evaluate later
    return param_dict


class Instruction:
    def __init__(
        self,
        opcode: str,
        rd: str,
        rs1: str | None,
        rs2: str | None,
        rstride: str | None,
        funct1: int | None,
        funct2: int | None,
        imm: int | None = None,
        rflag: int | None = None,
    ):

        self.opcode = opcode
        self.rd = rd
        self.rs1 = rs1
        self.rs2 = rs2
        self.rstride = rstride
        self.funct1 = funct1
        self.funct2 = funct2
        self.imm = imm
        self.rmask = rstride

    def __repr__(self):
        return f"Instruction(opcode='{self.opcode}', rd='{self.rd}', rs1='{self.rs1}', rs2='{self.rs2}', rstride = '{self.rstride}', funct1={self.funct1}, funct2={self.funct2}, imm={self.imm}, rflag={self.rflag})"


_REG_PREFIXES = ("gp", "f", "a")
# Hoisted to module scope: these were previously re-created for every line of the
# .asm (millions of times for large programs), which dominated sim_env re-parse time.
vector_masked_unary_or_reduction_ops = frozenset(
    {
        "V_EXP_V",
        "V_RECI_V",
        "V_RED_SUM",
        "V_RED_MAX",
        "V_RED_SUM_OVR",
        "V_RED_MAX_OVR",
    }
)
vector_masked_binary_ops = frozenset({"V_ADD_VV", "V_ADD_VF", "V_MUL_VV", "V_SUB_VV", "V_MUL_VF"})
_VSEG_ALIASES = {
    "V_ADD_VSEG": 0,
    "V_SUB_VSEG": 1,
    "V_MUL_VSEG": 2,
}
_COMPACT_STAT_ALIASES = {
    "V_STAT_MUL_F": 0,
    "V_STAT_ADD_F": 1,
    "V_STAT_RSQRT": 2,
}
_REDUCTION_OVERWRITE_ALIASES = {
    "V_RED_SUM_OVR": "V_RED_SUM",
    "V_RED_MAX_OVR": "V_RED_MAX",
    "V_RED_SUM_SEG_OVR": "V_RED_SUM_SEG",
    "V_RED_MAX_SEG_OVR": "V_RED_MAX_SEG",
}
_MULTI_SEGMENT_REDUCTIONS = frozenset({"V_RED_SUM_SEGS", "V_RED_MAX_SEGS"})


def _parse_operand(operand):
    """Parse a register (gp/f/a prefix, decimal index) or integer operand; None if neither.

    Operands are already whitespace-stripped by the caller (the line-132 split), so this
    does no stripping. Hoisted out of the per-line loop, where it was redefined every line.
    """
    if operand.endswith(";"):
        operand = operand[:-1]
    if operand.startswith("gp"):
        return int(operand[2:])  # decimal, not hex
    elif operand.startswith(("f", "a")):
        return int(operand[1:])  # decimal, not hex
    else:
        try:
            return int(operand)
        except ValueError:
            return None


def parse_asm_file(file_path: str) -> list[Instruction]:
    """
    Parse an ASM file into a list of Instruction objects.

    Supported formats:
    - opcode rd, rs1, rs2, rs3, funct1, funct2;
    - opcode rd, rs1, rs2, funct1, funct2;
    - opcode rd, rs1, rs2;
    - opcode rd, rs1, imm;
    - opcode rd, rs1;
    - opcode rd;

    :param file_path: Path to the .asm file
    :return: List of Instruction objects
    """
    instructions = []

    with open(file_path) as file:
        for line in file:
            # Strip once, then skip blanks and whole-line comments with cheap char checks
            # (most lines are neither). Comments are // or ; style.
            line = line.strip()
            if not line or line[0] == ";" or line.startswith("//"):
                continue
            # Remove inline // and ; comments only when present.
            c = line.find("//")
            if c != -1:
                line = line[:c]
            c = line.find(";")
            if c != -1:
                line = line[:c]

            # Split the opcode and operands
            parts = line.split()
            if len(parts) < 1 or ";" in parts[0]:
                continue  # Invalid line
            opcode = parts[0]

            # Handle instructions with no operands (e.g., C_BREAK)
            if len(parts) == 1:
                instructions.append(Instruction(opcode, None, None, None, None, None, None, None))
                continue

            operands = [part.strip() for part in " ".join(parts[1:]).split(",")]
            # print(f"Parsing instruction: {line}", "operand length:", len(operands), "operands:", operands)

            # Decode based on number of operands, case-structure by length
            rd = None
            rs1 = None
            rs2 = None
            rstride = None
            funct1 = None
            funct2 = None
            imm = None

            if opcode == "C_AGU_LOOP_LEN":
                if len(operands) != 1:
                    raise ValueError(
                        f"C_AGU_LOOP_LEN expects one body-length operand: {line}"
                    )
                rd = 0
                imm = _parse_operand(operands[0])
                instructions.append(
                    Instruction(opcode, rd, None, None, None, None, None, imm)
                )
                continue
            if opcode in {"C_AGU_BIND", "C_LOOP_START_AGU"}:
                if len(operands) != 2:
                    raise ValueError(f"{opcode} expects register, immediate: {line}")
                rd = _parse_operand(operands[0])
                imm = _parse_operand(operands[1])
                instructions.append(
                    Instruction(opcode, rd, None, None, None, None, None, imm)
                )
                continue

            if len(operands) == 1:
                operand_0 = operands[0]
                rd = _parse_operand(operand_0)
            elif len(operands) == 2:
                operand_0 = operands[0]
                operand_1 = operands[1]
                rd = _parse_operand(operand_0)
                # rs1 is a register, imm is a number
                # Heuristics: if it looks like a reg, it's rs1; else, it's imm
                if operand_1.startswith(("gp", "f", "a")):
                    rs1 = _parse_operand(operand_1)
                else:
                    try:
                        imm = int(operand_1)
                    except ValueError:
                        imm = None
            elif len(operands) == 3:
                operand_0, operand_1, operand_2 = operands
                rd = _parse_operand(operand_0)
                # If looks like register, rs1; else, imm
                if operand_1.startswith(("gp", "f", "a")):
                    rs1 = _parse_operand(operand_1)
                else:
                    try:
                        imm = int(operand_1)
                    except ValueError:
                        imm = None
                # If it looks like register, rs2; else, imm (overwrites imm if rs1 not present)
                if operand_2.startswith(("gp", "f", "a")):
                    rs2 = _parse_operand(operand_2)
                else:
                    try:
                        imm = int(operand_2)
                    except ValueError:
                        pass
                if opcode in vector_masked_unary_or_reduction_ops:
                    # Keep rmask/rstride aligned for 3-operand masked unary/reduction forms.
                    rstride = imm
                elif opcode in vector_masked_binary_ops:
                    # Allow 3-operand vector ALU forms by defaulting omitted rmask to 0.
                    rstride = 0
            elif len(operands) == 4:
                operand_0, operand_1, operand_2, operand_3 = operands
                rd = _parse_operand(operand_0)
                if operand_1.startswith(("gp", "f", "a")):
                    rs1 = _parse_operand(operand_1)
                else:
                    try:
                        imm = int(operand_1)
                    except ValueError:
                        imm = None
                if operand_2.startswith(("gp", "f", "a")):
                    rs2 = _parse_operand(operand_2)
                else:
                    try:
                        imm = int(operand_2)
                    except ValueError:
                        pass
                # Interpret 4th operand as rstride if int
                try:
                    rstride = int(operand_3)
                except ValueError:
                    rstride = None
            elif len(operands) == 5:
                operand_0, operand_1, operand_2, operand_3, operand_4 = operands
                rd = _parse_operand(operand_0)
                if operand_1.startswith(("gp", "f", "a")):
                    rs1 = _parse_operand(operand_1)
                else:
                    try:
                        imm = int(operand_1)
                    except ValueError:
                        imm = None
                if operand_2.startswith(("gp", "f", "a")):
                    rs2 = _parse_operand(operand_2)
                else:
                    try:
                        imm = int(operand_2)
                    except ValueError:
                        pass
                try:
                    rstride = int(operand_3)
                except ValueError:
                    rstride = None
                funct1_raw = operand_4.strip()
                if funct1_raw.endswith(";"):
                    funct1_raw = funct1_raw[:-1]
                try:
                    funct1 = int(funct1_raw)
                except ValueError:
                    funct1 = funct1_raw  # fallback, if not int, keep as string
            elif len(operands) == 6:
                operand_0, operand_1, operand_2, operand_3, operand_4, operand_5 = operands
                rd = _parse_operand(operand_0)
                if operand_1.startswith(("gp", "f", "a")):
                    rs1 = _parse_operand(operand_1)
                else:
                    try:
                        imm = int(operand_1)
                    except ValueError:
                        imm = None
                if operand_2.startswith(("gp", "f", "a")):
                    rs2 = _parse_operand(operand_2)
                else:
                    try:
                        imm = int(operand_2)
                    except ValueError:
                        pass
                try:
                    rstride = int(operand_3)
                except ValueError:
                    rstride = None
                funct1_raw = operand_4.strip()
                if funct1_raw.endswith(";"):
                    funct1_raw = funct1_raw[:-1]
                try:
                    funct1 = int(funct1_raw)
                except ValueError:
                    funct1 = funct1_raw  # fallback, if not int, keep as string
                funct2_raw = operand_5.strip()
                if funct2_raw.endswith(";"):
                    funct2_raw = funct2_raw[:-1]
                try:
                    funct2 = int(funct2_raw)
                except ValueError:
                    funct2 = funct2_raw  # fallback, if not int, keep as string

            if opcode in _REDUCTION_OVERWRITE_ALIASES:
                opcode = _REDUCTION_OVERWRITE_ALIASES[opcode]
                funct1 = 1

            if opcode in _MULTI_SEGMENT_REDUCTIONS:
                # Syntax: V_RED_*_SEGS gpDst, gpSrc, segment_log2.
                # The third parsed operand is an immediate; place it in the
                # rstride field consumed by the RTL decoder.
                rstride = imm
                rs2 = 0
            elif opcode in _VSEG_ALIASES:
                # Syntax: V_*_VSEG gpDst, gpVector, gpStats, segment_log2[, mask].
                # funct1[1:0] selects ADD/SUB/MUL and funct1[2] enables the
                # existing HLEN-granular Vector SRAM write mask.
                mask_enable = 0 if funct1 is None else int(funct1)
                if mask_enable not in (0, 1):
                    raise ValueError(f"VSEG mask flag must be 0 or 1, got {mask_enable}")
                funct1 = _VSEG_ALIASES[opcode] | (mask_enable << 2)
                opcode = "V_ALU_VSEG"
            elif opcode in _COMPACT_STAT_ALIASES:
                # Syntax: V_STAT_* gpDst, gpSrc, fScalar, segment_count.
                # Compact operations reuse V_ALU_VSEG with funct1[3] set.
                # Unlike ordinary VSEG, rstride stores count-1 rather than
                # log2(segment width).
                segment_count = rstride
                if segment_count is None or not 1 <= segment_count <= 16:
                    raise ValueError(
                        "compact-stat segment_count must be in [1, 16], got "
                        f"{segment_count}"
                    )
                rstride = segment_count - 1
                funct1 = 0x8 | _COMPACT_STAT_ALIASES[opcode]
                opcode = "V_ALU_VSEG"

            instructions.append(Instruction(opcode, rd, rs1, rs2, rstride, funct1, funct2, imm))

    return instructions


if __name__ == "__main__":
    # Example usage
    # file_path = '/home/george/Coprocessor_for_Llama/src/definitions/operation.svh'
    # enum_dict = load_isa_definitions(file_path)
    # print(enum_dict)

    asm_file_path = "/home/george/Coprocessor_for_Llama/src/system/test/benchmarks/fixed.asm"
    loaded_instr = parse_asm_file(asm_file_path)
    for instr in loaded_instr:
        print(instr)
