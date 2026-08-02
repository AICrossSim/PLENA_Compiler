"""Emission and FPRAM allocation helpers for IsaCompiler."""

from __future__ import annotations

from contextlib import contextmanager

from compiler.aten.isa_builder import (
    AsmInput,
    IsaBuilder,
    RepeatAxis,
    Sequence,
    Stage,
    final_sequence,
    render_asm,
)
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

    # ------------------------------------------------------------------
    # Generated ISA buffer
    #
    # Backed by a list of rendered chunks rather than one growing string:
    # ``self.generated_code += rendered`` per instruction is O(n) per call
    # (it copies the whole buffer), i.e. O(n^2) overall, which runs away when
    # the instruction count is large (e.g. mlen=16 vision attention tiles into
    # 4 col-blocks). Appending to a list is amortised O(1); the getter joins
    # on read, producing a byte-identical string. Callers still see a ``str``.
    # ------------------------------------------------------------------
    @property
    def generated_code(self) -> str:
        return "".join(getattr(self, "_code_chunks", ()))

    @generated_code.setter
    def generated_code(self, value: str) -> None:
        self._code_chunks = [value] if value else []

    @property
    def _unroll(self) -> bool:
        """Shorthand for self.unroll_loops."""
        return self.unroll_loops

    def _emit(self, isa_code: AsmInput) -> str:
        """Append ISA text to the output buffer and return it."""
        rendered = render_asm(isa_code) if getattr(self, "_emit_assembly", True) else ""
        if rendered:
            self._code_chunks.append(rendered)
        sink = getattr(self, "_symbolic_cost_sink", None)
        if sink is not None:
            schedule = final_sequence(isa_code)
            stage_stack = getattr(self, "_cost_stage_stack", ())
            if stage_stack:
                schedule = Sequence((Stage("/".join(stage_stack), schedule),))
            sink.consume(schedule)
        return rendered

    def emit(self, isa_code: AsmInput) -> str:
        """Public emission hook for code outside IsaCompiler internals."""
        return self._emit(isa_code)

    def emit_comment(self, text: str) -> str:
        """Append one assembly comment line."""
        return self._emit(IsaBuilder().comment(text))

    @contextmanager
    def cost_stage(self, stage_path: str):
        """Assign opaque hierarchical ownership to subsequently emitted ISA."""
        if not stage_path or stage_path.startswith("/") or stage_path.endswith("/"):
            raise ValueError(f"invalid stage path {stage_path!r}")
        self._cost_stage_stack.append(stage_path)
        try:
            yield self
        finally:
            popped = self._cost_stage_stack.pop()
            if popped != stage_path:
                raise RuntimeError("cost stage stack corrupted")

    def get_cost_trace(self, **metadata):
        """Finalize the compiler-assisted trace without affecting ASM output."""
        sink = getattr(self, "_symbolic_cost_sink", None)
        if sink is None:
            raise RuntimeError("cost tracing was not enabled for this compiler")
        return sink.finish(**metadata)

    def record_dma(self, transfer, *, multiplicity: int = 1, axes=()) -> None:
        """Attach exact compiler-owned DMA geometry to the active stage."""
        sink = getattr(self, "_symbolic_cost_sink", None)
        if sink is not None:
            stage_stack = getattr(self, "_cost_stage_stack", ())
            if not stage_stack:
                sink.record_dma(transfer, multiplicity=multiplicity, axes=tuple(axes))
                return
            stage = "/".join(stage_stack)
            sink.begin_stage(stage)
            try:
                sink.record_dma(transfer, multiplicity=multiplicity, axes=tuple(axes))
            finally:
                sink.end_stage(stage)

    @property
    def cost_summary_enabled(self) -> bool:
        sink = getattr(self, "_symbolic_cost_sink", None)
        return bool(sink is not None and sink.summary_enabled)

    @contextmanager
    def cost_repeat_region(
        self,
        count: int,
        *,
        axis: RepeatAxis | None = None,
        kind: str = "compiler-summary",
    ):
        sink = getattr(self, "_symbolic_cost_sink", None)
        if sink is None or not sink.summary_enabled:
            raise RuntimeError("cost_repeat_region requires summary cost tracing")
        sink.begin_repeat(count, axis, kind)
        try:
            yield self
        finally:
            sink.end_repeat(count, axis, kind)

    @contextmanager
    def suppress_cost_dma(self):
        sink = getattr(self, "_symbolic_cost_sink", None)
        if sink is None:
            yield self
            return
        sink.begin_dma_suppression()
        try:
            yield self
        finally:
            sink.end_dma_suppression()

    @contextmanager
    def cost_summary_template(self, key: tuple[object, ...]):
        sink = getattr(self, "_symbolic_cost_sink", None)
        if sink is None or not sink.summary_enabled:
            yield self
            return
        sink.begin_template(key)
        try:
            yield self
        finally:
            sink.end_template(key)

    def replay_cost_summary_template(
        self,
        key: tuple[object, ...],
        *,
        count: int = 1,
        axes: tuple[RepeatAxis, ...] = (),
        dma_address_delta_bytes: int = 0,
    ) -> bool:
        sink = getattr(self, "_symbolic_cost_sink", None)
        return bool(
            sink is not None
            and sink.summary_enabled
            and sink.replay_template(
                key,
                count=count,
                axes=axes,
                dma_address_delta_bytes=dma_address_delta_bytes,
            )
        )

    def emit_cost_opcode_counts(self, counts, *, provenance: str) -> None:
        sink = getattr(self, "_symbolic_cost_sink", None)
        if sink is None:
            raise RuntimeError("cost tracing is not enabled")
        stage_stack = getattr(self, "_cost_stage_stack", ())
        if not stage_stack:
            sink.add_opcode_counts(counts, provenance=provenance)
            return
        stage = "/".join(stage_stack)
        sink.begin_stage(stage)
        try:
            sink.add_opcode_counts(counts, provenance=provenance)
        finally:
            sink.end_stage(stage)

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
