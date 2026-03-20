"""
Shared compilation context for VLM code generation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from math import prod
from typing import Any

from asm_lib.plena_program import FPVar, InputVar, PLENAProgram, TensorVar, VRAMMatrixVar


def safe_codegen_name(raw: str, prefix: str = "") -> str:
    """Convert an arbitrary graph/module name into a stable symbol name."""
    normalized = re.sub(r"[^0-9A-Za-z_]+", "_", raw).strip("_")
    if not normalized:
        normalized = "anon"
    if prefix:
        return f"{prefix}_{normalized}"
    return normalized


def matrix_shape_from_meta(meta: dict[str, Any] | None) -> tuple[int, int] | None:
    """Collapse runtime tensor metadata into a 2D matrix shape for PLENA lowering."""
    if meta is None:
        return None
    shape = meta.get("shape")
    if not isinstance(shape, list) or not shape:
        return None
    if len(shape) == 1:
        return (1, int(shape[0]))
    rows = int(prod(shape[:-1]))
    cols = int(shape[-1])
    return (rows, cols)


@dataclass
class SymbolRecord:
    symbol: str
    tensor: TensorVar
    producer: str
    semantic_shape: tuple[int, ...] | None
    padded_shape: tuple[int, int] | None
    birth_order: int
    last_use_order: int
    is_model_input: bool = False
    is_model_output: bool = False
    load_source: InputVar | None = None


@dataclass
class ParameterRecord:
    canonical_name: str
    handle: InputVar
    logical_shape: tuple[int, int]
    source_shape: tuple[int, ...] | None
    layout: str
    hbm_addr: int
    hbm_size: int


class VLMSharedCodegenContext:
    """Owns the shared PLENA program plus graph-level symbol bookkeeping."""

    def __init__(
        self,
        *,
        mlen: int,
        blen: int,
        real_data_ratio: float = 1.125,
    ) -> None:
        self.mlen = mlen
        self.blen = blen
        self.real_data_ratio = real_data_ratio
        self.program = PLENAProgram(
            mlen=mlen,
            blen=blen,
            real_data_ratio=real_data_ratio,
        )
        self.symbols: dict[str, SymbolRecord] = {}
        self.parameters: dict[str, ParameterRecord] = {}
        self.constants: dict[str, int] = {}
        self._constant_vars: dict[str, FPVar] = {}
        self._name_counts: dict[str, int] = {}
        self._symbol_liveness: dict[str, dict[str, Any]] = {}
        self._has_shared_lowering = False

    @property
    def has_shared_lowering(self) -> bool:
        return self._has_shared_lowering

    def mark_shared_lowering_used(self) -> None:
        self._has_shared_lowering = True

    def reserve_name(self, raw: str, prefix: str = "") -> str:
        base = safe_codegen_name(raw, prefix=prefix)
        count = self._name_counts.get(base, 0)
        self._name_counts[base] = count + 1
        if count == 0:
            return base
        return f"{base}_{count}"

    def prepare_graph(self, nodes: list[dict[str, Any]]) -> None:
        """Precompute birth/last-use metadata for graph symbols."""
        symbol_uses: dict[str, list[int]] = {}
        births: dict[str, int] = {}
        ordered_nodes = sorted(nodes, key=lambda node: int(node.get("order", 0)))
        max_order = max((int(node.get("order", 0)) for node in ordered_nodes), default=0)

        for node in ordered_nodes:
            order = int(node.get("order", 0))
            for sym in node.get("in_syms") or []:
                symbol_uses.setdefault(sym, []).append(order)
                births.setdefault(sym, -1)
            for sym in node.get("out_syms") or []:
                births.setdefault(sym, order)

        produced_symbols = set(births)
        output_symbols = {
            sym
            for sym in produced_symbols
            if sym not in symbol_uses or max(symbol_uses.get(sym, []), default=-1) < births[sym]
        }

        self._symbol_liveness = {}
        for sym, birth_order in births.items():
            uses = symbol_uses.get(sym, [])
            if uses:
                last_use = max(uses)
            elif sym in output_symbols:
                last_use = max_order
            else:
                last_use = birth_order
            self._symbol_liveness[sym] = {
                "birth_order": birth_order,
                "last_use_order": last_use,
                "is_model_input": birth_order < 0,
                "is_model_output": sym in output_symbols,
            }

    def liveness_for(self, symbol: str) -> dict[str, Any]:
        return self._symbol_liveness.get(
            symbol,
            {
                "birth_order": -1,
                "last_use_order": -1,
                "is_model_input": True,
                "is_model_output": False,
            },
        )

    def is_symbol_dead_after(self, symbol: str, order: int) -> bool:
        return int(self.liveness_for(symbol)["last_use_order"]) <= int(order)

    def bind_symbol(
        self,
        symbol: str,
        tensor: TensorVar,
        *,
        producer: str,
        semantic_shape: tuple[int, ...] | None,
        padded_shape: tuple[int, int] | None = None,
        load_source: InputVar | None = None,
    ) -> SymbolRecord:
        info = self.liveness_for(symbol)
        record = SymbolRecord(
            symbol=symbol,
            tensor=tensor,
            producer=producer,
            semantic_shape=semantic_shape,
            padded_shape=padded_shape,
            birth_order=int(info["birth_order"]),
            last_use_order=int(info["last_use_order"]),
            is_model_input=bool(info["is_model_input"]),
            is_model_output=bool(info["is_model_output"]),
            load_source=load_source,
        )
        self.symbols[symbol] = record
        return record

    def resolve_symbol(self, symbol: str) -> SymbolRecord | None:
        return self.symbols.get(symbol)

    def materialize_symbol_input(
        self,
        *,
        symbol: str,
        matrix_shape: tuple[int, int],
        producer: str,
        hbm_addr: int | None = None,
    ) -> SymbolRecord:
        input_name = self.reserve_name(symbol, prefix="sym")
        input_handle = self.program.input(input_name, shape=matrix_shape, hbm_addr=hbm_addr)
        tensor = self.program.load_batch(input_handle, name=input_name)
        return self.bind_symbol(
            symbol,
            tensor,
            producer=producer,
            semantic_shape=matrix_shape,
            padded_shape=matrix_shape,
            load_source=input_handle,
        )

    def ensure_symbol_tensor(
        self,
        *,
        symbol: str,
        matrix_shape: tuple[int, int],
        producer: str,
        hbm_addr: int | None = None,
    ) -> VRAMMatrixVar:
        record = self.resolve_symbol(symbol)
        if record is None:
            record = self.materialize_symbol_input(
                symbol=symbol,
                matrix_shape=matrix_shape,
                producer=producer,
                hbm_addr=hbm_addr,
            )
        tensor = record.tensor
        if not isinstance(tensor, VRAMMatrixVar):
            if not isinstance(tensor, InputVar):
                raise TypeError(f"Unsupported symbol binding for {symbol}: {type(tensor)}")
            loaded = self.program.load_batch(
                tensor,
                name=self.reserve_name(f"{symbol}_loaded", prefix="sym"),
            )
            record = self.bind_symbol(
                symbol,
                loaded,
                producer=record.producer,
                semantic_shape=record.semantic_shape,
                padded_shape=record.padded_shape,
                load_source=tensor,
            )
            tensor = record.tensor
        return tensor

    def bind_external_symbol(
        self,
        *,
        symbol: str,
        matrix_shape: tuple[int, int],
        producer: str,
        vram_addr: int,
        semantic_shape: tuple[int, ...] | None = None,
        name_hint: str | None = None,
    ) -> SymbolRecord:
        bound_name = self.reserve_name(name_hint or symbol, prefix="ext")
        tensor = self.program.bind_vram(bound_name, shape=matrix_shape, vram_addr=vram_addr)
        return self.bind_symbol(
            symbol,
            tensor,
            producer=producer,
            semantic_shape=semantic_shape or matrix_shape,
            padded_shape=matrix_shape,
        )

    def release_symbol(self, symbol: str) -> None:
        record = self.symbols.pop(symbol, None)
        if record is None:
            return
        if record.is_model_output:
            self.symbols[symbol] = record
            return
        if isinstance(record.tensor, VRAMMatrixVar):
            self.program.free_tensor(record.tensor)

    def discard_symbol(self, symbol: str) -> None:
        self.symbols.pop(symbol, None)

    def register_parameter(
        self,
        *,
        canonical_name: str,
        logical_shape: tuple[int, int],
        source_shape: tuple[int, ...] | None = None,
        layout: str = "logical",
        hbm_addr: int | None = None,
    ) -> InputVar:
        existing = self.parameters.get(canonical_name)
        if existing is not None:
            if existing.logical_shape != logical_shape:
                raise ValueError(
                    f"Parameter shape mismatch for {canonical_name}: "
                    f"{existing.logical_shape} vs {logical_shape}"
                )
            if hbm_addr is not None and existing.hbm_addr != hbm_addr:
                raise ValueError(
                    f"Parameter HBM address mismatch for {canonical_name}: "
                    f"{existing.hbm_addr} vs {hbm_addr}"
                )
            return existing.handle

        handle = self.program.input(
            self.reserve_name(canonical_name, prefix="param"),
            shape=logical_shape,
            hbm_addr=hbm_addr,
        )
        self.parameters[canonical_name] = ParameterRecord(
            canonical_name=canonical_name,
            handle=handle,
            logical_shape=logical_shape,
            source_shape=source_shape,
            layout=layout,
            hbm_addr=handle.hbm_addr,
            hbm_size=handle.hbm_size,
        )
        return handle

    def register_constant(self, name: str, address: int | None = None) -> int:
        if name in self.constants:
            return self.constants[name]
        if address is None:
            fp_var = self.program.fp_var(self.reserve_name(name, prefix="const"), size=1)
            self._constant_vars[name] = fp_var
            address = fp_var.address
        self.constants[name] = address
        return address

    def mark_result_symbol(self, symbol: str) -> None:
        record = self.symbols.get(symbol)
        if record is None:
            return
        record.is_model_output = True
        if isinstance(record.tensor, VRAMMatrixVar):
            self.program.result(record.tensor)

    def debug_dump_lines(self) -> list[str]:
        lines = ["; === Shared Context Debug Dump ==="]
        lines.append(f"; symbols={len(self.symbols)}  parameters={len(self.parameters)}  constants={len(self.constants)}")
        if self.symbols:
            lines.append("; symbols:")
            for symbol, record in sorted(self.symbols.items()):
                addr_info = ""
                if isinstance(record.tensor, VRAMMatrixVar):
                    addr_info = f", vram={self.program.compiler.get_vram_addr(record.tensor.name)}"
                elif isinstance(record.tensor, InputVar):
                    addr_info = f", hbm={record.tensor.hbm_addr}"
                lines.append(
                    f";   {symbol} -> {record.tensor.name} "
                    f"(producer={record.producer}, last_use={record.last_use_order}{addr_info})"
                )
        if self.parameters:
            lines.append("; parameters:")
            for name, record in sorted(self.parameters.items()):
                lines.append(
                    f";   {name} -> {record.handle.name} "
                    f"logical_shape={record.logical_shape} layout={record.layout} "
                    f"hbm={record.hbm_addr} size={record.hbm_size}"
                )
        if self.constants:
            lines.append("; constants:")
            for name, address in sorted(self.constants.items()):
                lines.append(f";   {name} -> fpram[{address}]")
        allocator = self.program.compiler.register_allocator
        lines.append(
            "; registers: "
            f"free_gp={allocator.gp_registers} used_gp={allocator.used_gp} "
            f"free_addr={allocator.addr_registers} used_addr={allocator.used_addr} "
            f"free_fp={allocator.fp_registers} used_fp={allocator.used_fp}"
        )
        return lines
