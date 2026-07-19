"""Emission and FPRAM allocation helpers for IsaCompiler."""

from __future__ import annotations

from contextlib import contextmanager

from compiler.aten.cost_emitter import CostSink
from compiler.aten.isa_builder import (
    AsmInput,
    DmaTransfer,
    Instr,
    IsaBuilder,
    Sequence,
    Stage,
    parse_legacy_asm,
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
        cost_sink: CostSink | None = getattr(self, "_cost_sink", None)
        if cost_sink is not None:
            if isinstance(isa_code, str):
                parsed = parse_legacy_asm(isa_code)
                symbolic = Sequence(tuple(self._annotate_dma(item) for item in parsed.items))
            elif isinstance(isa_code, IsaBuilder):
                symbolic = Sequence(tuple(isa_code.items))
            else:
                symbolic = isa_code
            stage = getattr(self, "_active_cost_stage", None)
            cost_sink.emit(Sequence((Stage(stage, symbolic),)) if stage else symbolic)
            if getattr(self, "_emission_mode", "asm") == "cost":
                return ""
        rendered = render_asm(isa_code)
        self._code_chunks.append(rendered)
        return rendered

    def _annotate_dma(self, item):
        """Legacy instruction parser: retain opcode counts, not guessed geometry."""
        if not isinstance(item, Instr) or item.opcode not in {
            "H_PREFETCH_M",
            "H_PREFETCH_V",
            "H_STORE_V",
        }:
            return item
        return item

    def make_exact_mx_dma_transfer(
        self,
        *,
        opcode: str,
        precision: str,
        hbm_base: int,
        total_elements: int,
        element_offset: int,
        dim: int,
        amount: int,
        stride: int,
        rstride: int,
        source: str,
    ) -> DmaTransfer:
        """Resolve reference MXFP8 streams and preserve precision-neutral offsets."""
        if element_offset < 0 or element_offset % 8:
            raise ValueError(
                f"exact MX DMA offset must be nonnegative and block-8 aligned, got {element_offset}"
            )
        return DmaTransfer(
            opcode=opcode,
            direction="write" if opcode == "H_STORE_V" else "read",
            precision=precision,
            element_base=hbm_base + element_offset,
            scale_base=hbm_base + total_elements + element_offset // 8,
            dim=dim,
            amount=amount,
            stride=stride,
            rstride=rstride,
            write_amount=dim if opcode == "H_PREFETCH_M" else 1,
            geometry_fidelity="exact",
            source=source,
            memory_object=f"hbm:{hbm_base}:{total_elements}",
            logical_object_elements=total_elements,
            precision_role=precision,
            logical_element_offset=element_offset,
            logical_scale_offset=element_offset // 8,
            logical_stride=stride,
        )

    def emit(self, isa_code: AsmInput) -> str:
        """Public emission hook for code outside IsaCompiler internals."""
        return self._emit(isa_code)

    def emit_comment(self, text: str) -> str:
        """Append one assembly comment line."""
        return self._emit(IsaBuilder().comment(text))

    def emit_cost_counts(
        self,
        *,
        static_opcodes,
        dynamic_opcodes,
        memory_streams=(),
        schedule_reason="counts_only_kernel_summary",
    ) -> None:
        cost_sink: CostSink | None = getattr(self, "_cost_sink", None)
        if cost_sink is None:
            raise RuntimeError("emit_cost_counts requires cost or both emission mode")
        stage = getattr(self, "_active_cost_stage", None) or "global"
        cost_sink.add_counts(
            static_opcodes=static_opcodes,
            dynamic_opcodes=dynamic_opcodes,
            stage=stage,
            schedule_reason=schedule_reason,
        )
        for stream in memory_streams:
            cost_sink.add_memory_event(
                transfer=stream.transfer,
                multiplicity=stream.multiplicity,
                stage=stage,
                axes=stream.axes,
            )

    def emit_cost_schedule(
        self,
        *,
        static_opcodes,
        dynamic_opcodes,
        schedule,
        memory_streams=(),
        rendered_asm: AsmInput | None = None,
    ) -> None:
        """Emit a compact ordered kernel schedule plus algebraic counts.

        ``rendered_asm`` is only needed by a lowering shared between cost and
        assembly modes.  In ``both`` mode it preserves the original program
        without sending the expanded instructions through CostSink again.
        """
        cost_sink: CostSink | None = getattr(self, "_cost_sink", None)
        if cost_sink is None:
            raise RuntimeError("emit_cost_schedule requires cost or both emission mode")
        stage = getattr(self, "_active_cost_stage", None) or "global"
        stream_indices = tuple(
            cost_sink.add_memory_event(
                transfer=stream.transfer,
                multiplicity=stream.multiplicity,
                stage=stage,
                axes=stream.axes,
            )
            for stream in memory_streams
        )
        cost_sink.add_ordered_schedule(
            static_opcodes=static_opcodes,
            dynamic_opcodes=dynamic_opcodes,
            schedule=schedule,
            stage=stage,
            memory_stream_indices=stream_indices,
        )
        if (
            rendered_asm is not None
            and getattr(self, "_emission_mode", "asm") == "both"
        ):
            self._code_chunks.append(render_asm(rendered_asm))

    def record_dma_stream(self, transfer, *, multiplicity=1, axes=()) -> None:
        """Attach exact DMA metadata without changing rendered assembly."""
        cost_sink: CostSink | None = getattr(self, "_cost_sink", None)
        if cost_sink is None:
            return
        stage = getattr(self, "_active_cost_stage", None) or "global"
        cost_sink.add_memory_event(
            transfer=transfer,
            multiplicity=multiplicity,
            stage=stage,
            axes=axes,
        )

    @contextmanager
    def cost_stage(self, name: str):
        """Attach a semantic stage to subsequently emitted cost events."""
        previous = getattr(self, "_active_cost_stage", None)
        self._active_cost_stage = name if previous is None else f"{previous}/{name}"
        try:
            yield
        finally:
            self._active_cost_stage = previous

    @contextmanager
    def cost_repeat_region(
        self, count: int, *, name: str, repeat_kind: str = "compile_time"
    ):
        """Compress repeated cost-only operations without rendering ASM."""
        cost_sink: CostSink | None = getattr(self, "_cost_sink", None)
        if cost_sink is None or getattr(self, "_emission_mode", "asm") != "cost":
            raise RuntimeError("cost_repeat_region is available only in cost mode")
        with cost_sink.repeated_region(
            count, name=name, repeat_kind=repeat_kind
        ):
            yield

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
