from __future__ import annotations

import re
from collections.abc import Iterable, Iterator


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
    {"V_EXP_V", "V_RECI_V", "V_RED_SUM", "V_RED_MAX"}
)
vector_masked_binary_ops = frozenset(
    {
        "V_ADD_VV",
        "V_ADD_VF",
        "V_MUL_VV",
        "V_SUB_VV",
        "V_MUL_VF",
        "V_MAX_VF",
        "V_MIN_VF",
        "V_TOPK",
    }
)


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


def _optional_int(value: str) -> int | None:
    try:
        return int(value.rstrip(";"))
    except ValueError:
        return None


def _int_or_text(value: str) -> int | str:
    value = value.rstrip(";")
    try:
        return int(value)
    except ValueError:
        return value


def parse_asm_lines(lines: Iterable[str]) -> Iterator[Instruction]:
    """Yield instructions without retaining the complete assembly in memory."""
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line[0] == ";" or line.startswith("//"):
            continue
        comment = line.find("//")
        if comment != -1:
            line = line[:comment]
        comment = line.find(";")
        if comment != -1:
            line = line[:comment]
        parts = line.split()
        if not parts or ";" in parts[0]:
            continue

        opcode = parts[0]
        if len(parts) == 1:
            yield Instruction(opcode, None, None, None, None, None, None, None)
            continue

        operands = [part.strip() for part in " ".join(parts[1:]).split(",")]
        if not 1 <= len(operands) <= 6:
            raise ValueError(
                f"{opcode} has {len(operands)} operands; parser supports 1..6"
            )

        rd = _parse_operand(operands[0])
        rs1 = None
        rs2 = None
        rstride = None
        funct1 = None
        funct2 = None
        imm = None

        if len(operands) >= 2:
            if operands[1].startswith(_REG_PREFIXES):
                rs1 = _parse_operand(operands[1])
            else:
                imm = _optional_int(operands[1])
        if len(operands) >= 3:
            if operands[2].startswith(_REG_PREFIXES):
                rs2 = _parse_operand(operands[2])
            else:
                parsed = _optional_int(operands[2])
                if parsed is not None:
                    imm = parsed
        if len(operands) == 3:
            if opcode in vector_masked_unary_or_reduction_ops:
                rstride = imm
            elif opcode in vector_masked_binary_ops:
                rstride = 0
        if len(operands) >= 4:
            rstride = _optional_int(operands[3])
        if len(operands) >= 5:
            funct1 = _int_or_text(operands[4])
        if len(operands) == 6:
            funct2 = _int_or_text(operands[5])

        yield Instruction(opcode, rd, rs1, rs2, rstride, funct1, funct2, imm)


def iter_asm_file(file_path: str) -> Iterator[Instruction]:
    """Stream instructions from one assembly file."""
    with open(file_path) as file:
        yield from parse_asm_lines(file)


def parse_asm_file(file_path: str) -> list[Instruction]:
    """Parse an assembly file into a compatibility list."""
    return list(iter_asm_file(file_path))


if __name__ == "__main__":
    # Example usage
    # file_path = '/home/george/Coprocessor_for_Llama/src/definitions/operation.svh'
    # enum_dict = load_isa_definitions(file_path)
    # print(enum_dict)

    asm_file_path = (
        "/home/george/Coprocessor_for_Llama/src/system/test/benchmarks/fixed.asm"
    )
    loaded_instr = parse_asm_file(asm_file_path)
    for instr in loaded_instr:
        print(instr)
