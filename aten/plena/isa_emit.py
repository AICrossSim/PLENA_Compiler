"""Emission and FPRAM allocation helpers for IsaCompiler."""

from __future__ import annotations

from compiler.aten.isa_builder import AsmInput, IsaBuilder, render_asm
from compiler.aten.plena.registers import RegisterAllocator


class IsaEmitMixin:
    # =========================================================================
    # FP Register & FPRAM Management (inlined from former FPRAMCompiler).
    # All state lives on self (register_allocator, fpram_allocator, etc.).
    # =========================================================================

    @property
    def _reg(self) -> RegisterAllocator:
        """Shorthand for self.register_allocator (used by FPVar ISA helpers)."""
        return self.register_allocator

    @property
    def _unroll(self) -> bool:
        """Shorthand for self.unroll_loops."""
        return self.unroll_loops

    def _emit(self, isa_code: AsmInput) -> str:
        """Append ISA text to the output buffer and return it."""
        rendered = render_asm(isa_code)
        self.generated_code += rendered
        return rendered

    def emit(self, isa_code: AsmInput) -> str:
        """Public emission hook for code outside IsaCompiler internals."""
        return self._emit(isa_code)

    def emit_comment(self, text: str) -> str:
        """Append one assembly comment line."""
        return self._emit(IsaBuilder().comment(text))

    # ------------------------------------------------------------------
    # FP Register management
    # ------------------------------------------------------------------

    def allocate_fp_reg(self, count: int = 1) -> list[int]:
        """Allocate FP registers (f0-f7)."""
        return self._reg.allocate_fp(count)

    def free_fp_reg(self, registers: list[int]):
        """Free FP registers."""
        self._reg.free_fp(registers)

    # ------------------------------------------------------------------
    # FPRAM address-space management
    # ------------------------------------------------------------------

    def allocate_fpram(self, name: str, size: int) -> int:
        """Allocate FPRAM space, returns base address."""
        info = self.add_fpram_object(name=name, size=size)
        if info.fpram_addr is None:
            raise RuntimeError(f"Failed to allocate FPRAM for '{name}'")
        return info.fpram_addr

    def free_fpram(self, name: str, strict: bool = True):
        """Free FPRAM object by name."""
        return self.free_fpram_object(name, strict=strict)

    def get_fpram_addr(self, name: str) -> int:
        """Get FPRAM base address from object name."""
        return self.get_fpram_layout(name).fpram_addr

    def get_fpram_size(self, name: str) -> int:
        """Get FPRAM allocation size from object name."""
        return self.get_fpram_layout(name).size


__all__ = ["IsaEmitMixin"]
