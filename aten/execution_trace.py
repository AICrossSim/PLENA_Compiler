"""Structured dynamic-instruction traces for emitted PLENA assembly.

The compiler emits compact hardware loops.  This module accounts for those
loops with integer multiplicities, so a trace records dynamic work without
materialising repeated instructions.  The resulting entries retain the
execution stage, hardware precision role, tile geometry, and HBM transfer
metadata needed by the decode cost model.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Iterable, Iterator, Mapping, Sequence


EXECUTION_TRACE_SCHEMA = "plena-structured-execution-trace-v1"
REQUEST_MEMORY_TRACE_SCHEMA = "plena-compiler-request-memory-trace-v2"
COMPILATION_ARTIFACT_SCHEMA = "plena-compilation-artifact-v1"
NO_DMA = "none"
HBM_READ = "hbm_to_onchip"
HBM_WRITE = "onchip_to_hbm"

# This is the canonical decoder-stage vocabulary used by
# analytic_models.performance.decode_stage_validation.  Keeping it beside the
# assembly parser prevents the compiler trace and emulator PC attribution from
# silently assigning the same instruction to different stages.
SECTION_TO_STAGE: tuple[tuple[str, str | None], ...] = (
    ("; Load_Batch", "Activation load"),
    ("; Preload Activation", "Activation load"),
    ("; RMS Norm", "RMSNorm"),
    ("; Normalize", "RMSNorm"),
    ("; VRAM Sub Projection", "Q/K/V + W_O projection"),
    ("; Load SubMatrix Col", "Q/K/V + W_O projection"),
    ("; RoPE", "RoPE"),
    ("; Store ", "KV store"),
    ("; Flash Attention", "Flash attention"),
    ("; QKT", "Flash attention"),
    ("; Online Softmax", "Flash attention"),
    ("; PV Per KV Head", "Flash attention"),
    ("; Computing O", "Flash attention"),
    ("; Row-wise Scaling", "Flash attention"),
    ("; Pipelined K prefetch", "Flash attention"),
    ("; Pipelined V prefetch", "Flash attention"),
    ("; PackedKV", "Flash attention"),
    ("; Reset KV Prefetch", "Flash attention"),
    ("; VRAM Matrix Add", "Residual add"),
    ("; VRAM Block Add", "Residual add"),
    ("; FFN", "FFN (gate/up/down)"),
    ("; SILU", "FFN (gate/up/down)"),
    ("; === LM head", "LM head"),
    ("; Projection_T", "LM head"),
    ("; Linear T:", "LM head"),
    ("; Middle loop", None),
    ("; Inner loop", None),
    ("; Outer loop", None),
)

_REGISTER_RE = re.compile(r"^(?P<prefix>gp|a)(?P<index>\d+)$")
_LOAD_BATCH_RE = re.compile(r"^;\s*Load_Batch\s+(?P<name>\S+)\s+->")
_LOAD_MATRIX_RE = re.compile(
    r"^;\s*Load SubMatrix(?:\s+(?:Row|Col))?\s+(?P<name>[^\[\s]+)"
)
_STORE_RE = re.compile(r"^;\s*Store\s+(?P<name>\S+)\s+from VRAM to HBM")


def classify_stage_comment(comment: str, current_stage: str) -> str:
    """Return the decoder stage selected by an assembly comment."""

    stripped = comment.strip()
    if not stripped.startswith(";"):
        stripped = "; " + stripped
    for prefix, stage in SECTION_TO_STAGE:
        if stripped.startswith(prefix):
            return current_stage if stage is None else stage
    return current_stage


@dataclass(frozen=True)
class TensorTraceMetadata:
    """HBM layout information used to identify and size DMA instructions."""

    name: str
    hbm_address: int
    precision_mode: str
    element_bits: int
    block_size: int
    scale_bits: int
    physical_shape: tuple[int, int] = ()
    element_plane_bytes: int = 0
    hbm_size: int = 0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("trace tensor name must be non-empty")
        if self.hbm_address < 0:
            raise ValueError("trace tensor HBM address must be non-negative")
        if not self.precision_mode:
            raise ValueError("trace tensor precision mode must be non-empty")
        if min(self.element_bits, self.block_size, self.scale_bits) <= 0:
            raise ValueError("trace tensor storage widths must be positive")
        layout_values = (
            bool(self.physical_shape),
            bool(self.element_plane_bytes),
            bool(self.hbm_size),
        )
        if any(layout_values) and not all(layout_values):
            raise ValueError(
                "request-memory tracing requires tensor shape and plane sizes"
            )
        if self.physical_shape:
            if len(self.physical_shape) != 2 or any(
                value <= 0 for value in self.physical_shape
            ):
                raise ValueError("trace tensor physical shape must be two-dimensional")
            if self.element_plane_bytes >= self.hbm_size:
                raise ValueError("trace tensor element plane must precede its scale plane")


@dataclass(frozen=True)
class ExecutionTraceEntry:
    """One algebraically aggregated dynamic instruction class."""

    opcode: str
    precision_mode: str
    dynamic_count: int
    stage: str
    tile_shape: tuple[int, ...] = ()
    dma_shape: tuple[int, ...] = ()
    dma_bytes: int = 0
    dma_direction: str = NO_DMA
    tensor: str = ""

    def __post_init__(self) -> None:
        if not self.opcode or not self.precision_mode or not self.stage:
            raise ValueError("trace opcode, precision mode, and stage are required")
        if self.dynamic_count <= 0:
            raise ValueError("trace dynamic count must be positive")
        if any(value <= 0 for value in self.tile_shape + self.dma_shape):
            raise ValueError("trace shapes must contain positive dimensions")
        if self.dma_bytes < 0:
            raise ValueError("trace DMA size must be non-negative")
        if self.dma_direction not in {NO_DMA, HBM_READ, HBM_WRITE}:
            raise ValueError(f"unsupported trace DMA direction {self.dma_direction!r}")
        is_dma = self.dma_direction != NO_DMA
        if is_dma != bool(self.dma_bytes and self.dma_shape and self.tensor):
            raise ValueError("DMA entries require size, shape, direction, and tensor")

    @property
    def key(self) -> tuple[object, ...]:
        """Stable key required by the compiler-derived cost contract."""

        return (
            self.opcode,
            self.precision_mode,
            self.dynamic_count,
            self.stage,
            self.tile_shape,
            self.dma_shape,
            self.dma_bytes,
            self.dma_direction,
            self.tensor,
        )

    @property
    def total_dma_bytes(self) -> int:
        return self.dynamic_count * self.dma_bytes

    def to_dict(self) -> dict[str, object]:
        return {
            "opcode": self.opcode,
            "precision_mode": self.precision_mode,
            "dynamic_count": self.dynamic_count,
            "stage": self.stage,
            "tile_shape": list(self.tile_shape),
            "dma_shape": list(self.dma_shape),
            "dma_bytes": self.dma_bytes,
            "dma_direction": self.dma_direction,
            "tensor": self.tensor,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "ExecutionTraceEntry":
        required = {
            "opcode",
            "precision_mode",
            "dynamic_count",
            "stage",
            "tile_shape",
            "dma_shape",
            "dma_bytes",
            "dma_direction",
            "tensor",
        }
        if set(value) != required:
            raise ValueError("execution-trace entry fields differ from the schema")
        return cls(
            opcode=str(value["opcode"]),
            precision_mode=str(value["precision_mode"]),
            dynamic_count=int(value["dynamic_count"]),
            stage=str(value["stage"]),
            tile_shape=tuple(int(item) for item in value["tile_shape"]),  # type: ignore[arg-type]
            dma_shape=tuple(int(item) for item in value["dma_shape"]),  # type: ignore[arg-type]
            dma_bytes=int(value["dma_bytes"]),
            dma_direction=str(value["dma_direction"]),
            tensor=str(value["tensor"]),
        )


@dataclass(frozen=True)
class ExecutionTrace:
    """Structured trace bound to the assembly and array geometry that made it."""

    entries: tuple[ExecutionTraceEntry, ...]
    assembly_sha256: str
    mlen: int
    blen: int
    vlen: int
    hlen: int
    static_instruction_count: int
    schema_version: str = EXECUTION_TRACE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != EXECUTION_TRACE_SCHEMA:
            raise ValueError("unsupported execution-trace schema")
        if len(self.assembly_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in self.assembly_sha256
        ):
            raise ValueError("execution trace requires a lowercase SHA-256 digest")
        if min(self.mlen, self.blen, self.vlen, self.hlen) <= 0:
            raise ValueError("execution-trace geometry must be positive")
        if self.static_instruction_count < 0:
            raise ValueError("static instruction count must be non-negative")
        if self.static_instruction_count and not self.entries:
            raise ValueError("a non-empty assembly trace requires entries")

    @property
    def dynamic_instruction_count(self) -> int:
        return sum(entry.dynamic_count for entry in self.entries)

    @property
    def opcode_histogram(self) -> dict[str, int]:
        histogram: dict[str, int] = {}
        for entry in self.entries:
            histogram[entry.opcode] = histogram.get(entry.opcode, 0) + entry.dynamic_count
        return dict(sorted(histogram.items()))

    @property
    def stage_order(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(entry.stage for entry in self.entries))

    def entries_for_stage(self, stage: str) -> tuple[ExecutionTraceEntry, ...]:
        return tuple(entry for entry in self.entries if entry.stage == stage)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "assembly_sha256": self.assembly_sha256,
            "geometry": {
                "mlen": self.mlen,
                "blen": self.blen,
                "vlen": self.vlen,
                "hlen": self.hlen,
            },
            "static_instruction_count": self.static_instruction_count,
            "dynamic_instruction_count": self.dynamic_instruction_count,
            "opcode_histogram": self.opcode_histogram,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "ExecutionTrace":
        required = {
            "schema_version",
            "assembly_sha256",
            "geometry",
            "static_instruction_count",
            "dynamic_instruction_count",
            "opcode_histogram",
            "entries",
        }
        if set(value) != required:
            raise ValueError("execution-trace fields differ from the schema")
        geometry = value["geometry"]
        if not isinstance(geometry, Mapping) or set(geometry) != {
            "mlen",
            "blen",
            "vlen",
            "hlen",
        }:
            raise ValueError("execution-trace geometry differs from the schema")
        raw_entries = value["entries"]
        if not isinstance(raw_entries, list):
            raise ValueError("execution-trace entries must be a list")
        trace = cls(
            entries=tuple(ExecutionTraceEntry.from_dict(item) for item in raw_entries),
            assembly_sha256=str(value["assembly_sha256"]),
            mlen=int(geometry["mlen"]),
            blen=int(geometry["blen"]),
            vlen=int(geometry["vlen"]),
            hlen=int(geometry["hlen"]),
            static_instruction_count=int(value["static_instruction_count"]),
            schema_version=str(value["schema_version"]),
        )
        if trace.dynamic_instruction_count != int(value["dynamic_instruction_count"]):
            raise ValueError("execution-trace dynamic count is inconsistent")
        if trace.opcode_histogram != value["opcode_histogram"]:
            raise ValueError("execution-trace opcode histogram is inconsistent")
        return trace


def execution_trace_entry_sha256(entry: ExecutionTraceEntry) -> str:
    """Return a content identity for the settled trace-entry schema."""

    payload = json.dumps(
        entry.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class CompilerDMARequest:
    """One exact DMA command derived by executing compiler address arithmetic."""

    opcode: str
    stage: str
    precision_mode: str
    tensor: str
    address: int
    rows: int
    elements_per_row: int
    stride_bytes: int
    element_bits: int
    direction: str
    scale_bits: int
    block_size: int
    scale_address: int
    scale_stride_bytes: int
    partial_write_rmw: bool

    def __post_init__(self) -> None:
        if self.opcode not in {"H_PREFETCH_M", "H_PREFETCH_V", "H_STORE_V"}:
            raise ValueError(f"unsupported compiler DMA opcode {self.opcode!r}")
        if not self.stage or not self.precision_mode or not self.tensor:
            raise ValueError("compiler DMA stage, precision, and tensor are required")
        positive_values = (
            self.rows,
            self.elements_per_row,
            self.stride_bytes,
            self.element_bits,
            self.scale_bits,
            self.block_size,
            self.scale_stride_bytes,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in positive_values
        ):
            raise ValueError("compiler DMA coordinates and widths must be positive integers")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (self.address, self.scale_address)
        ):
            raise ValueError("compiler DMA addresses must be non-negative integers")
        if self.direction not in {"read", "write"}:
            raise ValueError("compiler DMA direction must be read or write")
        expected = "write" if self.opcode == "H_STORE_V" else "read"
        if self.direction != expected:
            raise ValueError("compiler DMA direction disagrees with its opcode")
        if self.partial_write_rmw != (self.direction == "write"):
            raise ValueError("compiler stores require physical read-modify-write")

    def to_dict(self) -> dict[str, object]:
        return {
            "opcode": self.opcode,
            "stage": self.stage,
            "precision_mode": self.precision_mode,
            "tensor": self.tensor,
            "address": self.address,
            "rows": self.rows,
            "elements_per_row": self.elements_per_row,
            "stride_bytes": self.stride_bytes,
            "element_bits": self.element_bits,
            "direction": self.direction,
            "scale_bits": self.scale_bits,
            "block_size": self.block_size,
            "scale_address": self.scale_address,
            "scale_stride_bytes": self.scale_stride_bytes,
            "partial_write_rmw": self.partial_write_rmw,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "CompilerDMARequest":
        required = {
            "opcode",
            "stage",
            "precision_mode",
            "tensor",
            "address",
            "rows",
            "elements_per_row",
            "stride_bytes",
            "element_bits",
            "direction",
            "scale_bits",
            "block_size",
            "scale_address",
            "scale_stride_bytes",
            "partial_write_rmw",
        }
        if set(value) != required:
            raise ValueError("compiler DMA request fields differ from the schema")
        integer_fields = (
            "address",
            "rows",
            "elements_per_row",
            "stride_bytes",
            "element_bits",
            "scale_bits",
            "block_size",
            "scale_address",
            "scale_stride_bytes",
        )
        if any(
            isinstance(value[name], bool) or not isinstance(value[name], int)
            for name in integer_fields
        ):
            raise TypeError("compiler DMA request coordinates must be integers")
        if not isinstance(value["partial_write_rmw"], bool):
            raise TypeError("compiler DMA read-modify-write flag must be boolean")
        return cls(
            opcode=str(value["opcode"]),
            stage=str(value["stage"]),
            precision_mode=str(value["precision_mode"]),
            tensor=str(value["tensor"]),
            address=value["address"],
            rows=value["rows"],
            elements_per_row=value["elements_per_row"],
            stride_bytes=value["stride_bytes"],
            element_bits=value["element_bits"],
            direction=str(value["direction"]),
            scale_bits=value["scale_bits"],
            block_size=value["block_size"],
            scale_address=value["scale_address"],
            scale_stride_bytes=value["scale_stride_bytes"],
            partial_write_rmw=value["partial_write_rmw"],
        )


@dataclass(frozen=True)
class CompilerDMARequestRun:
    """A compact affine sequence of otherwise identical DMA commands."""

    request: CompilerDMARequest
    repetitions: int
    address_step_bytes: int = 0
    scale_address_step_bytes: int = 0

    def __post_init__(self) -> None:
        if (
            isinstance(self.repetitions, bool)
            or not isinstance(self.repetitions, int)
            or self.repetitions <= 0
        ):
            raise ValueError("compiler DMA repetitions must be a positive integer")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (
                self.address_step_bytes,
                self.scale_address_step_bytes,
            )
        ):
            raise ValueError("compiler DMA affine steps must be non-negative integers")
        if self.repetitions == 1 and (
            self.address_step_bytes or self.scale_address_step_bytes
        ):
            raise ValueError("a single compiler DMA request cannot carry an affine step")
        if self.repetitions > 1:
            ratio_numerator = (
                self.request.element_bits * self.request.block_size
            )
            if ratio_numerator % self.request.scale_bits:
                raise ValueError("compiler DMA affine run has a fractional scale ratio")
            ratio = ratio_numerator // self.request.scale_bits
            if self.address_step_bytes != self.scale_address_step_bytes * ratio:
                raise ValueError(
                    "compiler DMA affine element and scale steps disagree"
                )

    def request_at(self, index: int) -> CompilerDMARequest:
        """Resolve one request without materializing the complete affine run."""

        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("compiler DMA run index must be an integer")
        if not 0 <= index < self.repetitions:
            raise IndexError("compiler DMA run index is outside its repetitions")
        if index == 0:
            return self.request
        value = self.request.to_dict()
        value["address"] = self.request.address + index * self.address_step_bytes
        value["scale_address"] = (
            self.request.scale_address + index * self.scale_address_step_bytes
        )
        return CompilerDMARequest.from_dict(value)

    def to_dict(self) -> dict[str, object]:
        return {
            "request": self.request.to_dict(),
            "repetitions": self.repetitions,
            "address_step_bytes": self.address_step_bytes,
            "scale_address_step_bytes": self.scale_address_step_bytes,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "CompilerDMARequestRun":
        required = {
            "request",
            "repetitions",
            "address_step_bytes",
            "scale_address_step_bytes",
        }
        legacy = {"request", "repetitions"}
        if frozenset(value) not in {frozenset(required), frozenset(legacy)}:
            raise ValueError("compiler DMA request-run fields differ from the schema")
        request = value["request"]
        repetitions = value["repetitions"]
        if not isinstance(request, Mapping):
            raise TypeError("compiler DMA request run must contain a request object")
        if isinstance(repetitions, bool) or not isinstance(repetitions, int):
            raise TypeError("compiler DMA request repetitions must be an integer")
        return cls(
            request=CompilerDMARequest.from_dict(request),
            repetitions=repetitions,
            address_step_bytes=int(value.get("address_step_bytes", 0)),
            scale_address_step_bytes=int(
                value.get("scale_address_step_bytes", 0)
            ),
        )


@dataclass(frozen=True)
class CompilerTraceRequestBinding:
    """Address-resolved requests corresponding to one trace DMA entry."""

    trace_entry_index: int
    trace_entry_sha256: str
    runs: tuple[CompilerDMARequestRun, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.trace_entry_index, bool)
            or not isinstance(self.trace_entry_index, int)
            or self.trace_entry_index < 0
        ):
            raise ValueError("compiler DMA trace index must be non-negative")
        if len(self.trace_entry_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.trace_entry_sha256
        ):
            raise ValueError("compiler DMA trace entry requires a SHA-256 identity")
        if not self.runs:
            raise ValueError("compiler DMA trace binding requires request runs")

    @property
    def request_count(self) -> int:
        """Number of DMA commands the affine runs stand for."""

        return sum(run.repetitions for run in self.runs)

    def request_at(self, ordinal: int) -> CompilerDMARequest:
        """Resolve the ordinal-th DMA command across this binding's runs."""

        if isinstance(ordinal, bool) or not isinstance(ordinal, int):
            raise TypeError("compiler DMA binding ordinal must be an integer")
        if ordinal < 0:
            raise IndexError("compiler DMA binding ordinal must be non-negative")
        remaining = ordinal
        for run in self.runs:
            if remaining < run.repetitions:
                return run.request_at(remaining)
            remaining -= run.repetitions
        raise IndexError("compiler DMA binding ordinal is outside its requests")

    def iter_requests(self) -> Iterator[CompilerDMARequest]:
        """Expand the affine runs back into every DMA command they encode."""

        for run in self.runs:
            for index in range(run.repetitions):
                yield run.request_at(index)

    def to_dict(self) -> dict[str, object]:
        return {
            "trace_entry_index": self.trace_entry_index,
            "trace_entry_sha256": self.trace_entry_sha256,
            "runs": [run.to_dict() for run in self.runs],
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, object],
    ) -> "CompilerTraceRequestBinding":
        required = {"trace_entry_index", "trace_entry_sha256", "runs"}
        if set(value) != required:
            raise ValueError("compiler DMA binding fields differ from the schema")
        index = value["trace_entry_index"]
        runs = value["runs"]
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("compiler DMA trace index must be an integer")
        if not isinstance(runs, list):
            raise TypeError("compiler DMA binding runs must be a list")
        if any(not isinstance(run, Mapping) for run in runs):
            raise TypeError("compiler DMA binding runs must contain objects")
        return cls(
            trace_entry_index=index,
            trace_entry_sha256=str(value["trace_entry_sha256"]),
            runs=tuple(CompilerDMARequestRun.from_dict(run) for run in runs),
        )


@dataclass(frozen=True)
class CompilerRequestMemoryTrace:
    """Physical-memory sidecar emitted independently of trace entry fields."""

    trace_assembly_sha256: str
    mlen: int
    blen: int
    vlen: int
    hlen: int
    bindings: tuple[CompilerTraceRequestBinding, ...]
    schema_version: str = REQUEST_MEMORY_TRACE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != REQUEST_MEMORY_TRACE_SCHEMA:
            raise ValueError("unsupported compiler request-memory trace schema")
        if len(self.trace_assembly_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.trace_assembly_sha256
        ):
            raise ValueError("compiler request-memory trace requires an assembly SHA-256")
        if min(self.mlen, self.blen, self.vlen, self.hlen) <= 0:
            raise ValueError("compiler request-memory geometry must be positive")
        indexes = [binding.trace_entry_index for binding in self.bindings]
        if len(indexes) != len(set(indexes)):
            raise ValueError("compiler request-memory trace has duplicate bindings")

    @property
    def sidecar_sha256(self) -> str:
        payload = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "trace_assembly_sha256": self.trace_assembly_sha256,
            "geometry": {
                "mlen": self.mlen,
                "blen": self.blen,
                "vlen": self.vlen,
                "hlen": self.hlen,
            },
            "bindings": [binding.to_dict() for binding in self.bindings],
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, object],
    ) -> "CompilerRequestMemoryTrace":
        required = {
            "schema_version",
            "trace_assembly_sha256",
            "geometry",
            "bindings",
        }
        if set(value) != required:
            raise ValueError(
                "compiler request-memory trace fields differ from the schema"
            )
        geometry = value["geometry"]
        if not isinstance(geometry, Mapping) or set(geometry) != {
            "mlen",
            "blen",
            "vlen",
            "hlen",
        }:
            raise ValueError(
                "compiler request-memory geometry differs from the schema"
            )
        if any(
            isinstance(geometry[name], bool)
            or not isinstance(geometry[name], int)
            for name in ("mlen", "blen", "vlen", "hlen")
        ):
            raise TypeError("compiler request-memory geometry must use integers")
        bindings = value["bindings"]
        if not isinstance(bindings, list):
            raise TypeError("compiler request-memory bindings must be a list")
        if any(not isinstance(binding, Mapping) for binding in bindings):
            raise TypeError("compiler request-memory bindings must contain objects")
        return cls(
            trace_assembly_sha256=str(value["trace_assembly_sha256"]),
            mlen=geometry["mlen"],
            blen=geometry["blen"],
            vlen=geometry["vlen"],
            hlen=geometry["hlen"],
            bindings=tuple(
                CompilerTraceRequestBinding.from_dict(binding)
                for binding in bindings
            ),
            schema_version=str(value["schema_version"]),
        )

    def validate_trace(self, trace: ExecutionTrace) -> None:
        if self.trace_assembly_sha256 != trace.assembly_sha256:
            raise ValueError("compiler request-memory assembly differs from the trace")
        if (self.mlen, self.blen, self.vlen, self.hlen) != (
            trace.mlen,
            trace.blen,
            trace.vlen,
            trace.hlen,
        ):
            raise ValueError("compiler request-memory geometry differs from the trace")
        expected = {
            index
            for index, entry in enumerate(trace.entries)
            if entry.dma_direction != NO_DMA
        }
        observed = {binding.trace_entry_index for binding in self.bindings}
        if observed != expected:
            raise ValueError("compiler request-memory DMA coverage differs from the trace")
        for binding in self.bindings:
            entry = trace.entries[binding.trace_entry_index]
            if binding.trace_entry_sha256 != execution_trace_entry_sha256(entry):
                raise ValueError("compiler request-memory trace entry identity is stale")
            if sum(run.repetitions for run in binding.runs) != entry.dynamic_count:
                raise ValueError("compiler DMA multiplicity differs from the trace")
            expected_direction = (
                "write" if entry.dma_direction == HBM_WRITE else "read"
            )
            for run in binding.runs:
                request = run.request
                if (
                    request.opcode != entry.opcode
                    or request.stage != entry.stage
                    or request.precision_mode != entry.precision_mode
                    or request.tensor != entry.tensor
                ):
                    raise ValueError(
                        "compiler DMA identity differs from its trace entry"
                    )
                if request.direction != expected_direction:
                    raise ValueError(
                        "compiler DMA direction differs from its trace entry"
                    )
                if (request.rows, request.elements_per_row) != entry.dma_shape:
                    raise ValueError(
                        "compiler DMA shape differs from its trace entry"
                    )
                request_bytes = _transfer_bytes(
                    (request.rows, request.elements_per_row),
                    element_bits=request.element_bits,
                    block_size=request.block_size,
                    scale_bits=request.scale_bits,
                )
                if request_bytes != entry.dma_bytes:
                    raise ValueError(
                        "compiler DMA byte count differs from its trace entry"
                    )


@dataclass(frozen=True)
class CompilationArtifact:
    """Assembly and its structured execution trace from one compiler lowering."""

    assembly: str
    execution_trace: ExecutionTrace
    request_memory: CompilerRequestMemoryTrace | None = None

    def __post_init__(self) -> None:
        digest = hashlib.sha256(self.assembly.encode("utf-8")).hexdigest()
        if digest != self.execution_trace.assembly_sha256:
            raise ValueError("compilation artifact assembly and trace differ")
        if self.request_memory is not None:
            self.request_memory.validate_trace(self.execution_trace)

    @property
    def trace(self) -> ExecutionTrace:
        """Concise compatibility alias for consumers that call the field trace."""

        return self.execution_trace

    def to_dict(self) -> dict[str, object]:
        """Serialize the compiler output without weakening its trace bindings."""

        return {
            "schema_version": COMPILATION_ARTIFACT_SCHEMA,
            "assembly": self.assembly,
            "execution_trace": self.execution_trace.to_dict(),
            "request_memory": (
                self.request_memory.to_dict()
                if self.request_memory is not None
                else None
            ),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "CompilationArtifact":
        required = {
            "schema_version",
            "assembly",
            "execution_trace",
            "request_memory",
        }
        if set(value) != required:
            raise ValueError("compilation artifact fields differ from the schema")
        if value["schema_version"] != COMPILATION_ARTIFACT_SCHEMA:
            raise ValueError("unsupported compilation artifact schema")
        assembly = value["assembly"]
        trace = value["execution_trace"]
        request_memory = value["request_memory"]
        if not isinstance(assembly, str):
            raise TypeError("compilation artifact assembly must be text")
        if not isinstance(trace, Mapping):
            raise TypeError("compilation artifact execution trace must be an object")
        if request_memory is not None and not isinstance(request_memory, Mapping):
            raise TypeError(
                "compilation artifact request-memory trace must be an object"
            )
        return cls(
            assembly=assembly,
            execution_trace=ExecutionTrace.from_dict(trace),
            request_memory=(
                CompilerRequestMemoryTrace.from_dict(request_memory)
                if request_memory is not None
                else None
            ),
        )


def _parse_register(value: str, prefix: str) -> int | None:
    match = _REGISTER_RE.fullmatch(value.strip())
    if match is None or match.group("prefix") != prefix:
        return None
    return int(match.group("index"))


def _instruction_parts(line: str) -> tuple[str, tuple[str, ...]]:
    code = line.partition(";")[0].strip()
    opcode, separator, remainder = code.partition(" ")
    args = tuple(item.strip() for item in remainder.split(",")) if separator else ()
    return opcode, args


def _loop_count(args: tuple[str, ...]) -> int:
    if len(args) < 2:
        raise ValueError("C_LOOP_START is missing its constant trip count")
    try:
        count = int(args[-1], 0)
    except ValueError as error:
        raise ValueError("C_LOOP_START trip count must be an integer") from error
    if count <= 0:
        raise ValueError("C_LOOP_START trip count must be positive")
    return count


@dataclass(frozen=True)
class LoopScopedLine:
    """One assembly line together with the trip counts enclosing it.

    ``multiplicity`` is the product of the constant trip counts of the hardware
    loops the line sits inside, so a static line stands for ``multiplicity``
    dynamic executions.  ``C_LOOP_START`` carries the multiplicity of its
    enclosing scope and ``C_LOOP_END`` that of the loop it closes.
    """

    line_number: int
    text: str
    opcode: str
    args: tuple[str, ...]
    multiplicity: int
    loop_depth: int

    @property
    def is_comment(self) -> bool:
        return not self.opcode


def iter_loop_scoped_lines(assembly: str) -> Iterator[LoopScopedLine]:
    """Walk assembly once, attaching each line's hardware-loop multiplicity."""

    loop_counts: list[int] = []
    for line_number, raw_line in enumerate(assembly.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        multiplicity = math.prod(loop_counts)
        if line.startswith(";"):
            yield LoopScopedLine(
                line_number=line_number,
                text=line,
                opcode="",
                args=(),
                multiplicity=multiplicity,
                loop_depth=len(loop_counts),
            )
            continue
        opcode, args = _instruction_parts(line)
        if not opcode:
            continue
        yield LoopScopedLine(
            line_number=line_number,
            text=line,
            opcode=opcode,
            args=args,
            multiplicity=multiplicity,
            loop_depth=len(loop_counts),
        )
        if opcode == "C_LOOP_START":
            loop_counts.append(_loop_count(args))
        elif opcode == "C_LOOP_END":
            if not loop_counts:
                raise ValueError("assembly contains an unmatched C_LOOP_END")
            loop_counts.pop()
    if loop_counts:
        raise ValueError("assembly contains an unterminated C_LOOP_START")


def _tile_shape(
    opcode: str,
    *,
    mlen: int,
    blen: int,
    vlen: int,
    hlen: int,
) -> tuple[int, ...]:
    if opcode in {"M_MM", "M_TMM"}:
        return (blen, blen, mlen)
    if opcode in {"M_BMM", "M_BTMM"}:
        return (blen, blen, hlen)
    if opcode in {"M_MV", "M_TMV"}:
        return (1, mlen, mlen)
    if opcode in {"M_BMV", "M_BTMV"}:
        return (1, mlen, hlen)
    if opcode == "M_MM_WO":
        return (blen, blen)
    if opcode == "M_BMM_WO":
        return (blen, mlen)
    if opcode in {"M_MV_WO", "M_BMV_WO"}:
        return (1, mlen)
    if opcode.startswith("V_"):
        return (vlen,)
    return ()


def _compute_precision(opcode: str, stage: str) -> str:
    if opcode.startswith("M_"):
        if stage == "Flash attention":
            if opcode.startswith(("M_B", "M_TB")):
                return "query_key"
            return "probability_value"
        return "activation_weight"
    if opcode.startswith("V_"):
        return "vector"
    if opcode.startswith("S_"):
        return "scalar_fp" if "_FP" in opcode else "scalar_integer"
    if opcode.startswith("C_"):
        return "control"
    return "native"


def _precision_from_dma(opcode: str, args: tuple[str, ...]) -> str:
    try:
        selector = int(args[-1], 0)
    except (IndexError, ValueError) as error:
        raise ValueError(f"{opcode} lacks a valid precision selector") from error
    if selector not in (0, 1):
        raise ValueError(f"{opcode} precision selector must be 0 or 1")
    if opcode == "H_PREFETCH_M":
        return "weight" if selector == 0 else "key_value"
    return "activation" if selector == 0 else "key_value"


def _transfer_bytes(
    shape: tuple[int, ...],
    *,
    element_bits: int,
    block_size: int,
    scale_bits: int,
) -> int:
    elements = math.prod(shape)
    element_bytes = (elements * element_bits + 7) // 8
    scale_count = math.ceil(elements / block_size)
    scale_bytes = (scale_count * scale_bits + 7) // 8
    return element_bytes + scale_bytes


def _update_tensor_hint(comment: str, current: str) -> str:
    for pattern in (_LOAD_BATCH_RE, _LOAD_MATRIX_RE, _STORE_RE):
        match = pattern.match(comment)
        if match is not None:
            return match.group("name")
    if comment.startswith("; Pipelined K prefetch"):
        return "key"
    if comment.startswith("; Pipelined V prefetch"):
        return "value"
    return current


def _update_register_values(
    opcode: str,
    args: tuple[str, ...],
    gp_values: dict[int, int],
    address_values: dict[int, int],
) -> None:
    if opcode == "S_ADDI_INT" and len(args) == 3:
        destination = _parse_register(args[0], "gp")
        source = _parse_register(args[1], "gp")
        if destination is None:
            return
        try:
            immediate = int(args[2], 0)
        except ValueError:
            gp_values.pop(destination, None)
            return
        if source in gp_values:
            gp_values[destination] = gp_values[source] + immediate
        else:
            gp_values.pop(destination, None)
        return
    if opcode == "S_LUI_INT" and len(args) == 2:
        destination = _parse_register(args[0], "gp")
        if destination is not None:
            try:
                gp_values[destination] = int(args[1], 0) << 12
            except ValueError:
                gp_values.pop(destination, None)
        return
    if opcode in {"S_ADD_INT", "S_SUB_INT"} and len(args) == 3:
        destination = _parse_register(args[0], "gp")
        left = _parse_register(args[1], "gp")
        right = _parse_register(args[2], "gp")
        if destination is None:
            return
        if left in gp_values and right in gp_values:
            value = gp_values[left] + gp_values[right]
            if opcode == "S_SUB_INT":
                value = gp_values[left] - gp_values[right]
            gp_values[destination] = value
        else:
            gp_values.pop(destination, None)
        return
    if opcode == "C_SET_ADDR_REG" and len(args) >= 3:
        destination = _parse_register(args[0], "a")
        left = _parse_register(args[1], "gp")
        right = _parse_register(args[2], "gp")
        if destination is None:
            return
        if left in gp_values and right in gp_values:
            address_values[destination] = (
                (gp_values[left] & 0xFFFFFFFF) << 32
            ) | (gp_values[right] & 0xFFFFFFFF)
        else:
            address_values.pop(destination, None)


def _tensor_candidates_by_address(
    metadata: Sequence[TensorTraceMetadata],
) -> dict[int, tuple[TensorTraceMetadata, ...]]:
    grouped: dict[int, list[TensorTraceMetadata]] = {}
    for tensor in metadata:
        grouped.setdefault(tensor.hbm_address, []).append(tensor)
    return {address: tuple(tensors) for address, tensors in grouped.items()}


def _select_tensor(
    candidates: Sequence[TensorTraceMetadata],
    dma_precision: str,
    tensor_hint: str,
) -> TensorTraceMetadata | None:
    if not candidates:
        return None
    compatible = [
        tensor
        for tensor in candidates
        if dma_precision == tensor.precision_mode
        or (dma_precision == "key_value" and tensor.precision_mode in {"key", "value", "key_value"})
        or (dma_precision == "activation" and tensor.precision_mode == "activation")
    ]
    for tensor in compatible:
        if tensor.name == tensor_hint:
            return tensor
    return (compatible or list(candidates))[-1]


def build_execution_trace(
    assembly: str,
    *,
    mlen: int,
    blen: int,
    vlen: int | None = None,
    hlen: int | None = None,
    vector_prefetch_amount: int = 4,
    vector_store_amount: int = 4,
    default_element_bits: int = 8,
    default_block_size: int = 8,
    default_scale_bits: int = 8,
    tensors: Iterable[TensorTraceMetadata] = (),
) -> ExecutionTrace:
    """Build a compact dynamic trace from the exact assembly sent downstream.

    Constant-count hardware loops are represented by products of trip counts.
    The function therefore runs in time proportional to emitted assembly size,
    independent of the dynamic context length represented by those loops.
    """

    if vlen is None:
        vlen = mlen
    if hlen is None:
        hlen = mlen
    dimensions = (
        mlen,
        blen,
        vlen,
        hlen,
        vector_prefetch_amount,
        vector_store_amount,
        default_element_bits,
        default_block_size,
        default_scale_bits,
    )
    if any(value <= 0 for value in dimensions):
        raise ValueError("execution-trace geometry and storage widths must be positive")

    tensor_metadata = tuple(tensors)
    by_address = _tensor_candidates_by_address(tensor_metadata)
    gp_values: dict[int, int] = {0: 0}
    address_values: dict[int, int] = {}
    current_stage = "Setup"
    current_tensor = ""
    static_count = 0

    # Dynamic count is deliberately excluded from this aggregation key and
    # summed as the value.  The final immutable entry exposes the full key,
    # including that algebraically derived count.
    aggregated: dict[tuple[object, ...], int] = {}

    for scoped in iter_loop_scoped_lines(assembly):
        if scoped.is_comment:
            current_stage = classify_stage_comment(scoped.text, current_stage)
            current_tensor = _update_tensor_hint(scoped.text, current_tensor)
            continue

        opcode, args = scoped.opcode, scoped.args
        multiplier = scoped.multiplicity
        static_count += 1

        tile_shape = _tile_shape(
            opcode,
            mlen=mlen,
            blen=blen,
            vlen=vlen,
            hlen=hlen,
        )
        dma_shape: tuple[int, ...] = ()
        dma_bytes = 0
        dma_direction = NO_DMA
        tensor_name = ""
        precision_mode = _compute_precision(opcode, current_stage)

        if opcode in {"H_PREFETCH_M", "H_PREFETCH_V", "H_STORE_V"}:
            precision_mode = _precision_from_dma(opcode, args)
            address_register = _parse_register(args[2], "a") if len(args) > 2 else None
            address = address_values.get(address_register) if address_register is not None else None
            tensor = _select_tensor(
                by_address.get(address, ()),
                precision_mode,
                current_tensor,
            )
            if tensor is not None:
                tensor_name = tensor.name
                precision_mode = tensor.precision_mode
                element_bits = tensor.element_bits
                block_size = tensor.block_size
                scale_bits = tensor.scale_bits
            else:
                tensor_name = current_tensor or precision_mode
                element_bits = default_element_bits
                block_size = default_block_size
                scale_bits = default_scale_bits
            if opcode == "H_PREFETCH_M":
                dma_shape = (mlen, mlen)
                dma_direction = HBM_READ
            elif opcode == "H_PREFETCH_V":
                dma_shape = (vector_prefetch_amount, vlen)
                dma_direction = HBM_READ
            else:
                dma_shape = (vector_store_amount, vlen)
                dma_direction = HBM_WRITE
            tile_shape = dma_shape
            dma_bytes = _transfer_bytes(
                dma_shape,
                element_bits=element_bits,
                block_size=block_size,
                scale_bits=scale_bits,
            )

        key = (
            opcode,
            precision_mode,
            current_stage,
            tile_shape,
            dma_shape,
            dma_bytes,
            dma_direction,
            tensor_name,
        )
        aggregated[key] = aggregated.get(key, 0) + multiplier

        _update_register_values(opcode, args, gp_values, address_values)

    entries = tuple(
        ExecutionTraceEntry(
            opcode=key[0],
            precision_mode=key[1],
            dynamic_count=count,
            stage=key[2],
            tile_shape=key[3],
            dma_shape=key[4],
            dma_bytes=key[5],
            dma_direction=key[6],
            tensor=key[7],
        )
        for key, count in aggregated.items()
    )
    return ExecutionTrace(
        entries=entries,
        assembly_sha256=hashlib.sha256(assembly.encode("utf-8")).hexdigest(),
        mlen=mlen,
        blen=blen,
        vlen=vlen,
        hlen=hlen,
        static_instruction_count=static_count,
    )


@dataclass(frozen=True)
class _StaticInstruction:
    opcode: str
    args: tuple[str, ...]
    stage: str
    tensor_hint: str


def _static_instructions(assembly: str) -> tuple[_StaticInstruction, ...]:
    instructions = []
    stage = "Setup"
    tensor_hint = ""
    for raw_line in assembly.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(";"):
            stage = classify_stage_comment(line, stage)
            tensor_hint = _update_tensor_hint(line, tensor_hint)
            continue
        opcode, args = _instruction_parts(line)
        if opcode:
            instructions.append(_StaticInstruction(opcode, args, stage, tensor_hint))
    return tuple(instructions)


def _required_register(value: str, prefix: str, opcode: str) -> int:
    register = _parse_register(value, prefix)
    if register is None:
        raise ValueError(f"{opcode} has an invalid {prefix} register {value!r}")
    return register


def _required_immediate(value: str, opcode: str) -> int:
    try:
        return int(value, 0)
    except ValueError as error:
        raise ValueError(f"{opcode} has an invalid integer immediate {value!r}") from error


def _request_tensor(
    tensors: Sequence[TensorTraceMetadata],
    *,
    element_address: int,
    scale_address: int,
    dma_precision: str,
    tensor_hint: str,
) -> TensorTraceMetadata:
    candidates = [
        tensor
        for tensor in tensors
        if tensor.physical_shape
        and tensor.hbm_address
        <= element_address
        < tensor.hbm_address + tensor.element_plane_bytes
        and tensor.hbm_address + tensor.element_plane_bytes
        <= scale_address
        < tensor.hbm_address + tensor.hbm_size
    ]
    tensor = _select_tensor(candidates, dma_precision, tensor_hint)
    if tensor is None:
        raise ValueError(
            "cannot bind compiler DMA addresses to a tensor layout: "
            f"element={element_address}, scale={scale_address}, hint={tensor_hint!r}"
        )
    return tensor


def _request_shape_key(request: CompilerDMARequest) -> tuple[object, ...]:
    """Return every request field except the two affine stream addresses."""

    value = request.to_dict()
    value.pop("address")
    value.pop("scale_address")
    return tuple(sorted(value.items()))


def _compress_dma_requests(
    requests: Sequence[CompilerDMARequest],
) -> tuple[CompilerDMARequestRun, ...]:
    """Encode consecutive request addresses as affine runs, preserving order.

    Every request is recoverable by ordinal through
    ``CompilerTraceRequestBinding.request_at``, so the encoding keeps the exact
    issue order of the DMA stream rather than only its multiset of addresses.
    """

    runs: list[CompilerDMARequestRun] = []
    index = 0
    while index < len(requests):
        first = requests[index]
        if index + 1 == len(requests):
            runs.append(CompilerDMARequestRun(first, 1))
            break
        second = requests[index + 1]
        same_shape = _request_shape_key(first) == _request_shape_key(second)
        address_step = second.address - first.address
        scale_step = second.scale_address - first.scale_address
        if not same_shape or address_step < 0 or scale_step < 0:
            runs.append(CompilerDMARequestRun(first, 1))
            index += 1
            continue
        stop = index + 2
        while stop < len(requests):
            candidate = requests[stop]
            ordinal = stop - index
            if (
                _request_shape_key(candidate) != _request_shape_key(first)
                or candidate.address
                != first.address + ordinal * address_step
                or candidate.scale_address
                != first.scale_address + ordinal * scale_step
            ):
                break
            stop += 1
        repetitions = stop - index
        runs.append(
            CompilerDMARequestRun(
                first,
                repetitions,
                address_step_bytes=(address_step if repetitions > 1 else 0),
                scale_address_step_bytes=(
                    scale_step if repetitions > 1 else 0
                ),
            )
        )
        index = stop
    return tuple(runs)


def build_request_memory_trace(
    assembly: str,
    trace: ExecutionTrace,
    *,
    vector_prefetch_amount: int,
    vector_store_amount: int,
    tensors: Iterable[TensorTraceMetadata],
    max_dynamic_instructions: int = 10_000_000,
) -> CompilerRequestMemoryTrace:
    """Execute integer/control ISA to emit exact address-varying DMA runs.

    The interpreter mirrors the transactional emulator's register, loop, HBM
    address, scale, and stride semantics.  It deliberately rejects integer
    SRAM loads and loop breaks inside active loops because their values/control
    flow cannot be proven from assembly alone.  Callers therefore fail closed
    rather than emitting a guessed physical-memory sidecar.
    """

    if vector_prefetch_amount <= 0 or vector_store_amount <= 0:
        raise ValueError("vector DMA amounts must be positive")
    if max_dynamic_instructions <= 0:
        raise ValueError("dynamic instruction limit must be positive")
    tensor_metadata = tuple(tensors)
    if not tensor_metadata or any(not tensor.physical_shape for tensor in tensor_metadata):
        raise ValueError(
            "request-memory tracing requires complete physical tensor metadata"
        )
    instructions = _static_instructions(assembly)
    gp = [0] * 16
    hbm_address = [0] * 16
    scalar_int: dict[int, int] = {}
    scale_offset = 0
    stride = 1
    loop_stack: list[tuple[int, int]] = []
    pc = 0
    executed = 0
    by_entry: dict[int, list[CompilerDMARequest]] = {}

    dma_entries: dict[
        tuple[str, str, str, str, tuple[int, ...], str], int
    ] = {}
    for index, entry in enumerate(trace.entries):
        if entry.dma_direction == NO_DMA:
            continue
        lookup_key = (
            entry.opcode,
            entry.stage,
            entry.precision_mode,
            entry.tensor,
            entry.dma_shape,
            entry.dma_direction,
        )
        if lookup_key in dma_entries:
            raise ValueError("compiler trace contains ambiguous DMA aggregation")
        dma_entries[lookup_key] = index

    while pc < len(instructions):
        executed += 1
        if executed > max_dynamic_instructions:
            raise ValueError(
                "request-memory trace exceeds the exact dynamic instruction limit"
            )
        instruction = instructions[pc]
        opcode, args = instruction.opcode, instruction.args
        next_pc = pc + 1

        if opcode in {"S_ADD_INT", "S_SUB_INT", "S_MUL_INT"}:
            if len(args) != 3:
                raise ValueError(f"{opcode} requires three operands")
            destination = _required_register(args[0], "gp", opcode)
            left = gp[_required_register(args[1], "gp", opcode)]
            right = gp[_required_register(args[2], "gp", opcode)]
            if opcode == "S_ADD_INT":
                value = left + right
            elif opcode == "S_SUB_INT":
                value = left - right
            else:
                value = left * right
            gp[destination] = value & 0xFFFFFFFF
        elif opcode == "S_ADDI_INT":
            if len(args) != 3:
                raise ValueError("S_ADDI_INT requires three operands")
            destination = _required_register(args[0], "gp", opcode)
            source = _required_register(args[1], "gp", opcode)
            gp[destination] = (
                gp[source] + _required_immediate(args[2], opcode)
            ) & 0xFFFFFFFF
        elif opcode == "S_LUI_INT":
            if len(args) != 2:
                raise ValueError("S_LUI_INT requires two operands")
            destination = _required_register(args[0], "gp", opcode)
            gp[destination] = (
                _required_immediate(args[1], opcode) << 12
            ) & 0xFFFFFFFF
        elif opcode == "S_ST_INT":
            if len(args) != 3:
                raise ValueError("S_ST_INT requires three operands")
            source = _required_register(args[0], "gp", opcode)
            base = _required_register(args[1], "gp", opcode)
            address = (gp[base] + _required_immediate(args[2], opcode)) & 0xFFFFFFFF
            scalar_int[address] = gp[source]
        elif opcode == "S_LD_INT":
            if len(args) != 3:
                raise ValueError("S_LD_INT requires three operands")
            destination = _required_register(args[0], "gp", opcode)
            base = _required_register(args[1], "gp", opcode)
            address = (gp[base] + _required_immediate(args[2], opcode)) & 0xFFFFFFFF
            if address not in scalar_int:
                raise ValueError(
                    "request-memory tracing cannot prove an integer SRAM load"
                )
            gp[destination] = scalar_int[address]
        elif opcode == "C_SET_ADDR_REG":
            if len(args) != 3:
                raise ValueError("C_SET_ADDR_REG requires three operands")
            destination = _required_register(args[0], "a", opcode)
            high = gp[_required_register(args[1], "gp", opcode)]
            low = gp[_required_register(args[2], "gp", opcode)]
            hbm_address[destination] = (high << 32) | low
        elif opcode == "C_SET_SCALE_REG":
            if len(args) != 1:
                raise ValueError("C_SET_SCALE_REG requires one operand")
            scale_offset = gp[_required_register(args[0], "gp", opcode)]
        elif opcode == "C_SET_STRIDE_REG":
            if len(args) != 1:
                raise ValueError("C_SET_STRIDE_REG requires one operand")
            stride = gp[_required_register(args[0], "gp", opcode)]
            if stride <= 0:
                raise ValueError("compiler DMA stride must be positive")
        elif opcode == "C_LOOP_START":
            if len(args) < 2:
                raise ValueError("C_LOOP_START requires a register and trip count")
            loop_register = _required_register(args[0], "gp", opcode)
            trip_count = _required_immediate(args[-1], opcode)
            if trip_count <= 0:
                raise ValueError("compiler loop trip count must be positive")
            gp[loop_register] = trip_count
            loop_stack.append((pc, loop_register))
        elif opcode == "C_LOOP_END":
            if len(args) != 1:
                raise ValueError("C_LOOP_END requires one operand")
            loop_register = _required_register(args[0], "gp", opcode)
            if not loop_stack or loop_stack[-1][1] != loop_register:
                raise ValueError("request-memory tracing requires properly nested loops")
            if gp[loop_register] > 1:
                gp[loop_register] -= 1
                next_pc = loop_stack[-1][0] + 1
            else:
                gp[loop_register] = 0
                loop_stack.pop()
        elif opcode == "C_BREAK":
            if loop_stack:
                raise ValueError(
                    "request-memory tracing cannot prove C_BREAK inside a loop"
                )
            break
        elif opcode in {"H_PREFETCH_M", "H_PREFETCH_V", "H_STORE_V"}:
            if len(args) != 5:
                raise ValueError(f"{opcode} requires five operands")
            offset_register = _required_register(args[1], "gp", opcode)
            address_register = _required_register(args[2], "a", opcode)
            rstride = _required_immediate(args[3], opcode)
            if rstride not in (0, 1):
                raise ValueError("compiler DMA stride selector must be zero or one")
            dma_precision = _precision_from_dma(opcode, args)
            element_address = hbm_address[address_register] + gp[offset_register]

            # Tensor metadata supplies the exact element/scale widths needed to
            # reproduce the emulator's element_scale_ratio calculation.
            provisional = [
                tensor
                for tensor in tensor_metadata
                if tensor.physical_shape
                and tensor.hbm_address
                <= element_address
                < tensor.hbm_address + tensor.element_plane_bytes
            ]
            tensor = _select_tensor(
                provisional,
                dma_precision,
                instruction.tensor_hint,
            )
            if tensor is None:
                raise ValueError(
                    "cannot bind compiler DMA element address to a tensor layout"
                )
            if tensor.element_bits <= 0 or tensor.scale_bits <= 0:
                raise ValueError("compiler DMA tensor has invalid storage widths")
            element_scale_ratio_numerator = (
                tensor.element_bits * tensor.block_size
            )
            if element_scale_ratio_numerator % tensor.scale_bits:
                raise ValueError("compiler DMA element/scale ratio is not integral")
            element_scale_ratio = (
                element_scale_ratio_numerator // tensor.scale_bits
            )
            offset = gp[offset_register]
            if offset % element_scale_ratio:
                raise ValueError("compiler DMA offset is not scale-stream aligned")
            scale_address = (
                hbm_address[address_register]
                + scale_offset
                + offset // element_scale_ratio
            )
            tensor = _request_tensor(
                tensor_metadata,
                element_address=element_address,
                scale_address=scale_address,
                dma_precision=dma_precision,
                tensor_hint=instruction.tensor_hint,
            )
            rows = (
                trace.mlen
                if opcode == "H_PREFETCH_M"
                else vector_store_amount
                if opcode == "H_STORE_V"
                else vector_prefetch_amount
            )
            elements_per_row = trace.mlen if opcode == "H_PREFETCH_M" else trace.vlen
            element_row_bytes = math.ceil(
                elements_per_row * tensor.element_bits / 8
            )
            stride_bytes = stride if rstride else element_row_bytes
            scale_stride_bytes = stride_bytes // element_scale_ratio
            if stride_bytes % element_scale_ratio:
                raise ValueError("compiler DMA stride is not scale-stream aligned")
            direction = "write" if opcode == "H_STORE_V" else "read"
            dma_direction = HBM_WRITE if direction == "write" else HBM_READ
            dma_shape = (rows, elements_per_row)
            lookup_key = (
                opcode,
                instruction.stage,
                tensor.precision_mode,
                tensor.name,
                dma_shape,
                dma_direction,
            )
            try:
                entry_index = dma_entries[lookup_key]
            except KeyError as error:
                raise ValueError(
                    "address-resolved DMA does not match a compiler trace entry"
                ) from error
            request = CompilerDMARequest(
                opcode=opcode,
                stage=instruction.stage,
                precision_mode=tensor.precision_mode,
                tensor=tensor.name,
                address=element_address,
                rows=rows,
                elements_per_row=elements_per_row,
                stride_bytes=stride_bytes,
                element_bits=tensor.element_bits,
                direction=direction,
                scale_bits=tensor.scale_bits,
                block_size=tensor.block_size,
                scale_address=scale_address,
                scale_stride_bytes=scale_stride_bytes,
                partial_write_rmw=direction == "write",
            )
            by_entry.setdefault(entry_index, []).append(request)

        pc = next_pc

    if loop_stack:
        raise ValueError("request-memory trace ended inside a hardware loop")
    bindings = tuple(
        CompilerTraceRequestBinding(
            trace_entry_index=index,
            trace_entry_sha256=execution_trace_entry_sha256(trace.entries[index]),
            runs=_compress_dma_requests(requests),
        )
        for index, requests in sorted(by_entry.items())
    )
    sidecar = CompilerRequestMemoryTrace(
        trace_assembly_sha256=trace.assembly_sha256,
        mlen=trace.mlen,
        blen=trace.blen,
        vlen=trace.vlen,
        hlen=trace.hlen,
        bindings=bindings,
    )
    sidecar.validate_trace(trace)
    return sidecar


__all__ = [
    "COMPILATION_ARTIFACT_SCHEMA",
    "REQUEST_MEMORY_TRACE_SCHEMA",
    "EXECUTION_TRACE_SCHEMA",
    "HBM_READ",
    "HBM_WRITE",
    "NO_DMA",
    "SECTION_TO_STAGE",
    "CompilationArtifact",
    "CompilerDMARequest",
    "CompilerDMARequestRun",
    "CompilerRequestMemoryTrace",
    "CompilerTraceRequestBinding",
    "ExecutionTrace",
    "ExecutionTraceEntry",
    "TensorTraceMetadata",
    "build_execution_trace",
    "build_request_memory_trace",
    "classify_stage_comment",
    "execution_trace_entry_sha256",
]
