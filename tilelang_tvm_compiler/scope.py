"""PLENA storage scopes.

In TVM TIR, a buffer's "storage scope" is just a string attached to the
buffer. We pick a fixed vocabulary here so different parts of the compiler
agree on which physical memory each buffer lives in.

Physical scope vocabulary (mirrors PLENA hardware):
    HBM   -- main DRAM, source/sink for DMA
    VRAM  -- vector SRAM, LHS operand of BTMM/MM and target of vector ops
    MRAM  -- matrix SRAM, RHS operand of BTMM/MM
    FPRAM -- on-chip FP buffer (small, used for staging)

Block-private declared scopes (tilelang DSL surface — these participate in
the compiler's lane-fusion expansion via `allocate_group_memory`):
    shared.dyn       -- T.alloc_shared default; resolved to vram / mram by
                        scope_inference based on usage
    local.fragment   -- T.alloc_fragment default; resolved to vram / fpram

Global declared scopes (user-declared, authoritative — these do NOT
participate in lane-fusion expansion; their shape IS their physical layout):
    global.vram, global.fpram, global.mram

`global.*` is for SRAM tensors that outlive the kernel — typically a buffer
that the testbench preloads before the kernel runs (e.g. weights into FPRAM)
or reads directly after the kernel finishes (e.g. an output cache in VRAM).
The user writes the physical shape they want; the compiler keeps it as-is.

PrimFunc parameters (function arguments) are treated as HBM by default.
"""

HBM = "hbm"
VRAM = "vram"
MRAM = "mram"
FPRAM = "fpram"

PHYSICAL_SCOPES = (HBM, VRAM, MRAM, FPRAM)

GLOBAL_PREFIX = "global."
GLOBAL_VRAM = GLOBAL_PREFIX + VRAM
GLOBAL_FPRAM = GLOBAL_PREFIX + FPRAM
GLOBAL_MRAM = GLOBAL_PREFIX + MRAM

GLOBAL_SCOPES = (GLOBAL_VRAM, GLOBAL_FPRAM, GLOBAL_MRAM)

# All scope strings the compiler treats as final answers (no inference).
ALL_SCOPES = PHYSICAL_SCOPES + GLOBAL_SCOPES


def is_known(scope: str) -> bool:
    return scope in ALL_SCOPES


def is_global_scope(scope: str) -> bool:
    """True for user-declared `global.<phys>` scopes that bypass lane-fusion
    expansion. Buffers with these scopes carry their physical layout in their
    declared shape and must not be rewritten by `allocate_group_memory`."""
    return scope.startswith(GLOBAL_PREFIX) and scope in GLOBAL_SCOPES


def physical_scope(scope: str) -> str:
    """Strip the `global.` prefix if present. Downstream passes that only
    care about *where* a buffer lives in hardware (address allocation,
    codegen, ISA emit) can call this to collapse `global.vram` and `vram`
    to the same answer (`vram`)."""
    if scope.startswith(GLOBAL_PREFIX):
        return scope[len(GLOBAL_PREFIX):]
    return scope
