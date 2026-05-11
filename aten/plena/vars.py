"""Tensor proxy classes for the ATen PLENA compiler path."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from compiler.aten.plena_compiler import PlenaCompiler


class TensorVar:
    """
    Tensor proxy object base class

    All tensor variables inherit from this class.
    Supports __matmul__ (`@`) operator, which automatically dispatches to appropriate PlenaCompiler methods.

    Dual naming:
    - display_name: User-visible name (e.g., "temp", "Q", "S")
    - internal_name: System internal name (e.g., "my_func_0/temp"), used for symbol table and ISA generation
    """

    def __init__(
        self,
        program: PlenaCompiler,
        internal_name: str,
        kind: str,
        shape: tuple[int, int],
        display_name: str | None = None,
    ):
        self._program = program
        self.internal_name = internal_name  # System internal name (with scope prefix), used by symbol table
        self.display_name = display_name if display_name is not None else internal_name  # User-visible name
        self.kind = kind  # "input", "batch", "matrix", "vram_matrix"
        self.shape = shape

    @property
    def name(self) -> str:
        """Compatibility property: returns internal_name for internal system use"""
        return self.internal_name

    def __matmul__(self, other):
        """A @ B: Dispatch to appropriate computation based on operand types"""
        return self._program._dispatch_matmul(self, other)

    def __repr__(self):
        if self.display_name != self.internal_name:
            return (
                f"{self.__class__.__name__}(display={self.display_name!r}, "
                f"internal={self.internal_name!r}, shape={self.shape})"
            )
        return f"{self.__class__.__name__}({self.display_name!r}, shape={self.shape})"


class InputVar(TensorVar):
    """
    Input variable: tensor declared in HBM

    Not yet loaded to VRAM; needs to be loaded via load_batch / load_matrix.

    If ``prestaged_vram_addr`` is not None the tensor is assumed to be already
    present in VRAM at that byte address.  ``load_batch`` will register it at
    that address without emitting any HBM→VRAM prefetch instructions.
    """

    def __init__(
        self,
        program: PlenaCompiler,
        name: str,
        shape: tuple[int, int],
        hbm_addr: int,
        hbm_size: int,
        display_name: str | None = None,
        prestaged_vram_addr: int | None = None,
    ):
        super().__init__(program, name, "input", shape, display_name=display_name)
        self.hbm_addr = hbm_addr
        self.hbm_size = hbm_size
        self.prestaged_vram_addr = prestaged_vram_addr


class FPVar:
    """
    FP variable: maps to a contiguous region in FPRAM

    Declared via prog.fp_var("scale", size=1), automatically allocates FPRAM space.
    Provides .address for ISA generation (S_LD_FP / S_ST_FP).

    Usage:
        scale = prog.fp_var("scale", size=1)
        m_old = prog.fp_var("m_old", size=64)

        scale.address   # -> FPRAM address (int)
        scale.size      # -> number of elements
        scale[3]        # -> address + 3 (element offset)
    """

    def __init__(
        self, program: PlenaCompiler, internal_name: str, address: int, size: int, display_name: str | None = None
    ):
        self._program = program
        self.internal_name = internal_name
        self.display_name = display_name if display_name is not None else internal_name
        self.address = address
        self.size = size

    @property
    def name(self) -> str:
        return self.internal_name

    def __getitem__(self, idx: int) -> int:
        """Element offset: fp_var[i] -> address + i"""
        if idx < 0 or idx >= self.size:
            raise IndexError(f"FPVar '{self.display_name}' index {idx} out of range [0, {self.size})")
        return self.address + idx

    def __repr__(self):
        return f"FPVar({self.display_name!r}, addr={self.address}, size={self.size})"


class VRAMMatrixVar(TensorVar):
    """
    VRAM matrix variable: large matrix allocated via alloc

    Used to store intermediate results (e.g., S block, PV, O).
    Supports sub-block indexed writes: `O[r][c] = ...`
    """

    def __init__(self, program: PlenaCompiler, name: str, shape: tuple[int, int], display_name: str | None = None):
        super().__init__(program, name, "vram_matrix", shape, display_name=display_name)


class TensorKind(Enum):
    """Identifies which memory the tensor lives in / which proxy backs it."""

    HBM = "hbm"  # legacy: InputVar
    VRAM = "vram"  # legacy: VRAMMatrixVar
    FPRAM = "fpram"  # legacy: FPVar


# Type alias: a "Tensor" is any of the legacy proxy objects.  Callers can use
# this annotation without worrying about which specific backing allocator
# a given tensor lives in.
Tensor = TensorVar | InputVar | VRAMMatrixVar | FPVar


def tensor_kind(tensor: Tensor) -> TensorKind:
    """Return the backing storage of a tensor proxy."""
    if isinstance(tensor, FPVar):
        return TensorKind.FPRAM
    if isinstance(tensor, VRAMMatrixVar):
        return TensorKind.VRAM
    if isinstance(tensor, InputVar):
        return TensorKind.HBM
    if isinstance(tensor, TensorVar):
        # Generic TensorVar without a specific backing — classify by ``kind``.
        kind = getattr(tensor, "kind", "")
        if kind in ("vram_matrix", "batch", "matrix"):
            return TensorKind.VRAM
        if kind == "input":
            return TensorKind.HBM
    raise TypeError(f"Unknown tensor type: {type(tensor).__name__}")
