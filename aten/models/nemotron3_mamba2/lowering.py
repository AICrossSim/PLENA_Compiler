"""Typed lowering of Nemotron Mamba-2 prefill and decode commands."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import IntEnum

import torch

from asm_templates._imm import load_large_int
from assembler.generated_contract import (
    MAMBA_COMPLETION_ALIGNMENT,
    MAMBA_COMPLETION_SIZE,
    MAMBA_DESCRIPTOR_ALIGNMENT,
    MAMBA_DESCRIPTOR_FLAG_BITS,
    MAMBA_DESCRIPTOR_SIZE,
    MAMBA_NO_EVENT,
    MAMBA_SUBOPS,
)
from assembler.mamba_abi import (
    encode_mamba_instruction,
    pack_mamba_completion,
    pack_mamba_descriptor,
)
from aten.isa_builder import IsaBuilder, addr, gp

from .memory import (
    U32_LIMIT,
    ByteAddressArena,
    MambaPersistentStateAllocator,
    MambaStateAllocation,
    MemoryRegion,
)
from .reference import Mamba2Config, Mamba2Weights, PrecisionPolicy


class MambaSubop(IntEnum):
    PREFILL = MAMBA_SUBOPS["PREFILL"]
    STEP = MAMBA_SUBOPS["STEP"]


@dataclass(frozen=True)
class MambaTensorBindings:
    """HBM regions for one mixer layer, all allocated in the command arena."""

    config: Mamba2Config
    precision_policy: PrecisionPolicy
    batch_capacity: int
    sequence_capacity: int
    input_token_stride: int
    input_batch_stride: int
    output_token_stride: int
    output_batch_stride: int
    input: MemoryRegion
    output: MemoryRegion
    in_proj_weight: MemoryRegion
    conv_weight: MemoryRegion
    a_log: MemoryRegion
    dt_bias: MemoryRegion
    d_skip: MemoryRegion
    norm_weight: MemoryRegion
    out_proj_weight: MemoryRegion
    in_proj_bias: MemoryRegion | None = None
    conv_bias: MemoryRegion | None = None
    out_proj_bias: MemoryRegion | None = None


@dataclass(frozen=True)
class MambaCommandSpec:
    subop: MambaSubop
    context_id: int
    layer_id: int
    batch_size: int
    sequence_length: int
    precision_policy: PrecisionPolicy = PrecisionPolicy.FP32_REFERENCE
    queue_id: int = 0
    continue_state: bool = False
    write_completion: bool = True
    profile: bool = False
    dependency_event: int = MAMBA_NO_EVENT
    completion_event: int = MAMBA_NO_EVENT


@dataclass(frozen=True)
class MambaRegisterAssignment:
    context_gp: int = 1
    descriptor_offset_gp: int = 2
    address_high_gp: int = 3
    address_low_gp: int = 4
    descriptor_hbm_reg: int = 0

    def validate(self) -> None:
        gp_registers = (
            self.context_gp,
            self.descriptor_offset_gp,
            self.address_high_gp,
            self.address_low_gp,
        )
        if any(
            not isinstance(index, int) or not 1 <= index < 16 for index in gp_registers
        ):
            raise ValueError("Mamba lowering GP registers must be in gp1..gp15")
        if len(set(gp_registers)) != len(gp_registers):
            raise ValueError("Mamba lowering GP registers must be distinct")
        if (
            not isinstance(self.descriptor_hbm_reg, int)
            or not 0 <= self.descriptor_hbm_reg < 8
        ):
            raise ValueError("descriptor HBM address register must be in a0..a7")


@dataclass(frozen=True)
class MemoryImage:
    region: MemoryRegion
    data: bytes


@dataclass(frozen=True)
class MambaProgram:
    assembly: str
    command_word: int
    descriptor: MemoryImage
    completion: MemoryImage | None
    state: MambaStateAllocation
    memory_map: tuple[MemoryRegion, ...]


def _element_bytes(policy: PrecisionPolicy) -> int:
    if policy == PrecisionPolicy.FP32_REFERENCE:
        return 4
    if policy == PrecisionPolicy.BF16_ACTIVATION_FP32_STATE:
        return 2
    raise ValueError(f"unsupported precision policy {policy!r}")


def _f32_bits(value: float) -> int:
    try:
        return struct.unpack("<I", struct.pack("<f", value))[0]
    except (OverflowError, struct.error) as error:
        raise ValueError(f"value {value!r} is not representable as binary32") from error


def _minimum_tensor_sizes(
    config: Mamba2Config,
    batch_size: int,
    sequence_length: int,
    element_bytes: int,
) -> dict[str, int]:
    tokens = batch_size * sequence_length
    return {
        "input": tokens * config.d_model * element_bytes,
        "output": tokens * config.d_model * element_bytes,
        "in_proj_weight": config.projection_size * config.d_model * element_bytes,
        "in_proj_bias": config.projection_size * element_bytes,
        "conv_weight": config.conv_channels * config.conv_kernel * element_bytes,
        "conv_bias": config.conv_channels * element_bytes,
        "a_log": config.num_heads * element_bytes,
        "dt_bias": config.num_heads * element_bytes,
        "d_skip": config.num_heads * element_bytes,
        "norm_weight": config.d_inner * element_bytes,
        "out_proj_weight": config.d_model
        * config.output_projection_size
        * element_bytes,
        "out_proj_bias": config.d_model * element_bytes,
    }


def _strided_tensor_extent(
    batch_capacity: int,
    sequence_capacity: int,
    batch_stride: int,
    token_stride: int,
    row_bytes: int,
) -> int:
    return (
        (batch_capacity - 1) * batch_stride
        + (sequence_capacity - 1) * token_stride
        + row_bytes
    )


def allocate_mamba_tensor_bindings(
    arena: ByteAddressArena,
    config: Mamba2Config,
    *,
    batch_capacity: int,
    sequence_capacity: int,
    precision_policy: PrecisionPolicy,
    prefix: str,
    include_in_proj_bias: bool,
    include_conv_bias: bool,
    include_out_proj_bias: bool,
) -> MambaTensorBindings:
    """Allocate a complete, non-aliasing row-major tensor map for one layer."""
    if batch_capacity <= 0 or sequence_capacity <= 0:
        raise ValueError("tensor batch and sequence capacities must be positive")
    if not prefix:
        raise ValueError("tensor binding prefix must not be empty")
    sizes = _minimum_tensor_sizes(
        config,
        batch_capacity,
        sequence_capacity,
        _element_bytes(precision_policy),
    )
    element_bytes = _element_bytes(precision_policy)
    token_stride = config.d_model * element_bytes
    batch_stride = sequence_capacity * token_stride

    def allocate(name: str) -> MemoryRegion:
        return arena.allocate(f"{prefix}.{name}", sizes[name], kind="tensor")

    return MambaTensorBindings(
        config=config,
        precision_policy=precision_policy,
        batch_capacity=batch_capacity,
        sequence_capacity=sequence_capacity,
        input_token_stride=token_stride,
        input_batch_stride=batch_stride,
        output_token_stride=token_stride,
        output_batch_stride=batch_stride,
        input=allocate("input"),
        output=allocate("output"),
        in_proj_weight=allocate("in_proj_weight"),
        in_proj_bias=allocate("in_proj_bias") if include_in_proj_bias else None,
        conv_weight=allocate("conv_weight"),
        conv_bias=allocate("conv_bias") if include_conv_bias else None,
        a_log=allocate("a_log"),
        dt_bias=allocate("dt_bias"),
        d_skip=allocate("d_skip"),
        norm_weight=allocate("norm_weight"),
        out_proj_weight=allocate("out_proj_weight"),
        out_proj_bias=allocate("out_proj_bias") if include_out_proj_bias else None,
    )


def _tensor_storage_bytes(tensor: torch.Tensor, policy: PrecisionPolicy) -> bytes:
    value = tensor.detach().cpu().contiguous()
    if policy == PrecisionPolicy.FP32_REFERENCE:
        return value.float().numpy().astype("<f4", copy=False).tobytes(order="C")
    if policy == PrecisionPolicy.BF16_ACTIVATION_FP32_STATE:
        return (
            value.to(torch.bfloat16)
            .view(torch.uint16)
            .numpy()
            .astype("<u2", copy=False)
            .tobytes(order="C")
        )
    raise ValueError(f"unsupported precision policy {policy!r}")


def materialize_mamba_weight_images(
    bindings: MambaTensorBindings,
    weights: Mamba2Weights,
) -> tuple[MemoryImage, ...]:
    """Serialize extracted weights into the exact row-major descriptor regions."""
    weights.validate(bindings.config)
    images: list[MemoryImage] = []
    fields = (
        ("in_proj_weight", weights.in_proj_weight),
        ("in_proj_bias", weights.in_proj_bias),
        ("conv_weight", weights.conv_weight),
        ("conv_bias", weights.conv_bias),
        ("a_log", weights.a_log),
        ("dt_bias", weights.dt_bias),
        ("d_skip", weights.d_skip),
        ("norm_weight", weights.norm_weight),
        ("out_proj_weight", weights.out_proj_weight),
        ("out_proj_bias", weights.out_proj_bias),
    )
    for name, tensor in fields:
        region = getattr(bindings, name)
        if (region is None) != (tensor is None):
            raise ValueError(
                f"tensor binding and extracted weight disagree on presence of {name}"
            )
        if region is None or tensor is None:
            continue
        data = _tensor_storage_bytes(tensor, bindings.precision_policy)
        if len(data) != region.size_bytes:
            raise ValueError(
                f"serialized {name} has {len(data)} bytes, region has {region.size_bytes}"
            )
        images.append(MemoryImage(region, data))
    return tuple(images)


def materialize_mamba_io_images(
    bindings: MambaTensorBindings,
    input_value: torch.Tensor,
) -> tuple[MemoryImage, MemoryImage]:
    """Pack ``[batch, sequence, d_model]`` input using the descriptor strides."""
    if input_value.ndim != 3 or input_value.shape[2] != bindings.config.d_model:
        raise ValueError(
            "Mamba input must have shape [batch, sequence, d_model], got "
            f"{tuple(input_value.shape)}"
        )
    batch_size, sequence_length, _ = input_value.shape
    if batch_size <= 0 or sequence_length <= 0:
        raise ValueError("Mamba input batch and sequence dimensions must be positive")
    if batch_size > bindings.batch_capacity:
        raise ValueError("Mamba input batch exceeds the tensor binding capacity")
    if sequence_length > bindings.sequence_capacity:
        raise ValueError("Mamba input sequence exceeds the tensor binding capacity")

    row_bytes = bindings.config.d_model * _element_bytes(bindings.precision_policy)
    if row_bytes > bindings.input_token_stride:
        raise ValueError("Mamba input row does not fit the declared token stride")
    final_end = (
        (batch_size - 1) * bindings.input_batch_stride
        + (sequence_length - 1) * bindings.input_token_stride
        + row_bytes
    )
    if final_end > bindings.input.size_bytes:
        raise ValueError("Mamba input does not fit the allocated HBM region")

    input_data = bytearray(bindings.input.size_bytes)
    for batch in range(batch_size):
        for token in range(sequence_length):
            row = _tensor_storage_bytes(
                input_value[batch, token], bindings.precision_policy
            )
            offset = (
                batch * bindings.input_batch_stride
                + token * bindings.input_token_stride
            )
            input_data[offset : offset + len(row)] = row
    return (
        MemoryImage(bindings.input, bytes(input_data)),
        MemoryImage(bindings.output, bytes(bindings.output.size_bytes)),
    )


class MambaCommandCompiler:
    """Build one descriptor and one ``X_MAMBA`` command per compile call."""

    def __init__(
        self,
        arena: ByteAddressArena,
        state_allocator: MambaPersistentStateAllocator,
        *,
        registers: MambaRegisterAssignment | None = None,
    ) -> None:
        if state_allocator.arena is not arena:
            raise ValueError(
                "command and persistent-state allocation must share one HBM arena"
            )
        self.arena = arena
        self.state_allocator = state_allocator
        self.registers = registers or MambaRegisterAssignment()
        self.registers.validate()

    def _validate_spec(self, spec: MambaCommandSpec) -> None:
        if not isinstance(spec.subop, MambaSubop):
            raise ValueError(
                "only MAMBA_PREFILL and MAMBA_STEP are supported by this lowering"
            )
        for name, value in (
            ("context_id", spec.context_id),
            ("layer_id", spec.layer_id),
        ):
            if not isinstance(value, int) or not 0 <= value < U32_LIMIT:
                raise ValueError(f"{name} must be an unsigned 32-bit integer")
        if not isinstance(spec.batch_size, int) or spec.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not isinstance(spec.sequence_length, int) or spec.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if spec.subop == MambaSubop.STEP:
            if spec.sequence_length != 1:
                raise ValueError("MAMBA_STEP requires sequence_length=1")
            if not spec.continue_state:
                raise ValueError("MAMBA_STEP requires continue_state=True")
        if not isinstance(spec.queue_id, int) or not 0 <= spec.queue_id < 16:
            raise ValueError("queue_id must fit the 4-bit command queue field")
        for name, value in (
            ("dependency_event", spec.dependency_event),
            ("completion_event", spec.completion_event),
        ):
            if not isinstance(value, int) or not 0 <= value < U32_LIMIT:
                raise ValueError(f"{name} must be an unsigned 32-bit integer")
        _element_bytes(spec.precision_policy)

    def _validate_bindings(
        self,
        bindings: MambaTensorBindings,
        config: Mamba2Config,
        spec: MambaCommandSpec,
    ) -> None:
        if bindings.config != config:
            raise ValueError(
                "tensor bindings were allocated for a different Mamba configuration"
            )
        if bindings.precision_policy != spec.precision_policy:
            raise ValueError(
                "tensor binding precision policy does not match the command"
            )
        if spec.batch_size > bindings.batch_capacity:
            raise ValueError("command batch exceeds the tensor binding capacity")
        if spec.sequence_length > bindings.sequence_capacity:
            raise ValueError("command sequence exceeds the tensor binding capacity")

        element_bytes = _element_bytes(spec.precision_policy)
        row_bytes = config.d_model * element_bytes
        for name, token_stride, batch_stride in (
            ("input", bindings.input_token_stride, bindings.input_batch_stride),
            ("output", bindings.output_token_stride, bindings.output_batch_stride),
        ):
            if token_stride < row_bytes or token_stride % element_bytes:
                raise ValueError(
                    f"{name} token stride is incompatible with its element layout"
                )
            if (
                batch_stride < bindings.sequence_capacity * token_stride
                or batch_stride % element_bytes
            ):
                raise ValueError(
                    f"{name} batch stride is incompatible with its sequence layout"
                )
            if token_stride >= U32_LIMIT or batch_stride >= U32_LIMIT:
                raise ValueError(f"{name} stride does not fit the descriptor u32 field")

        sizes = _minimum_tensor_sizes(
            config,
            bindings.batch_capacity,
            bindings.sequence_capacity,
            element_bytes,
        )
        sizes["input"] = _strided_tensor_extent(
            bindings.batch_capacity,
            bindings.sequence_capacity,
            bindings.input_batch_stride,
            bindings.input_token_stride,
            row_bytes,
        )
        sizes["output"] = _strided_tensor_extent(
            bindings.batch_capacity,
            bindings.sequence_capacity,
            bindings.output_batch_stride,
            bindings.output_token_stride,
            row_bytes,
        )
        required_names = (
            "input",
            "output",
            "in_proj_weight",
            "conv_weight",
            "a_log",
            "dt_bias",
            "d_skip",
            "norm_weight",
            "out_proj_weight",
        )
        optional_names = ("in_proj_bias", "conv_bias", "out_proj_bias")
        for name in (*required_names, *optional_names):
            region = getattr(bindings, name)
            if region is None:
                if name in required_names:
                    raise ValueError(f"required tensor binding {name} is missing")
                continue
            if not self.arena.owns(region):
                raise ValueError(
                    f"tensor binding {name} is not owned by the command HBM arena"
                )
            if region.address % 64:
                raise ValueError(f"tensor binding {name} is not 64-byte aligned")
            if region.size_bytes < sizes[name]:
                raise ValueError(
                    f"tensor binding {name} has {region.size_bytes} bytes; "
                    f"command requires {sizes[name]}"
                )

    def compile(
        self,
        config: Mamba2Config,
        bindings: MambaTensorBindings,
        spec: MambaCommandSpec,
        *,
        state_batch_capacity: int | None = None,
    ) -> MambaProgram:
        self._validate_spec(spec)
        self._validate_bindings(bindings, config, spec)
        capacity = (
            spec.batch_size if state_batch_capacity is None else state_batch_capacity
        )
        if capacity < spec.batch_size:
            raise ValueError("state batch capacity is smaller than this command batch")
        state = self.state_allocator.allocate(
            spec.context_id,
            spec.layer_id,
            config,
            capacity,
        )

        prefix = f"mamba.command.ctx{spec.context_id}.layer{spec.layer_id}"
        completion_region = None
        completion_image = None
        if spec.write_completion:
            completion_region = self.arena.allocate_unique(
                f"{prefix}.completion",
                MAMBA_COMPLETION_SIZE,
                alignment=MAMBA_COMPLETION_ALIGNMENT,
                kind="completion",
            )
            completion_image = MemoryImage(
                completion_region,
                pack_mamba_completion(
                    {"status": 0, "completion_event": 0, "elapsed_cycles": 0}
                ),
            )
        descriptor_region = self.arena.allocate_unique(
            f"{prefix}.descriptor",
            MAMBA_DESCRIPTOR_SIZE,
            alignment=MAMBA_DESCRIPTOR_ALIGNMENT,
            kind="descriptor",
        )

        strides = (
            bindings.input_token_stride,
            bindings.output_token_stride,
            bindings.input_batch_stride,
            bindings.output_batch_stride,
            state.state_head_stride,
            state.state_request_stride,
            state.conv_request_stride,
        )
        if any(value >= U32_LIMIT for value in strides):
            raise ValueError("a Mamba descriptor byte stride does not fit u32")

        flags = 0
        for enabled, name in (
            (spec.continue_state, "CONTINUE_STATE"),
            (spec.write_completion, "WRITE_COMPLETION"),
            (spec.profile, "PROFILE"),
        ):
            if enabled:
                flags |= 1 << MAMBA_DESCRIPTOR_FLAG_BITS[name]

        def optional_address(region: MemoryRegion | None) -> int:
            return 0 if region is None else region.address

        descriptor_data = pack_mamba_descriptor(
            {
                "flags": flags,
                "context_id": spec.context_id,
                "sequence_length": spec.sequence_length,
                "batch_size": spec.batch_size,
                "d_model": config.d_model,
                "d_inner": config.d_inner,
                "num_heads": config.num_heads,
                "head_dim": config.head_dim,
                "state_dim": config.state_dim,
                "groups": config.groups,
                "chunk_size": config.chunk_size,
                "conv_kernel": config.conv_kernel,
                "precision_policy": int(spec.precision_policy),
                "rms_norm_eps_f32_bits": _f32_bits(config.rms_norm_eps),
                "input_addr": bindings.input.address,
                "output_addr": bindings.output.address,
                "in_proj_weight_addr": bindings.in_proj_weight.address,
                "in_proj_bias_addr": optional_address(bindings.in_proj_bias),
                "conv_weight_addr": bindings.conv_weight.address,
                "conv_bias_addr": optional_address(bindings.conv_bias),
                "a_log_addr": bindings.a_log.address,
                "d_skip_addr": bindings.d_skip.address,
                "norm_weight_addr": bindings.norm_weight.address,
                "out_proj_weight_addr": bindings.out_proj_weight.address,
                "out_proj_bias_addr": optional_address(bindings.out_proj_bias),
                "ssm_state_addr": state.ssm.address,
                "conv_state_addr": state.conv.address,
                "scratch_addr": 0,
                "completion_addr": 0
                if completion_region is None
                else completion_region.address,
                "dt_bias_addr": bindings.dt_bias.address,
                "input_batch_stride": bindings.input_batch_stride,
                "input_token_stride": bindings.input_token_stride,
                "output_batch_stride": bindings.output_batch_stride,
                "output_token_stride": bindings.output_token_stride,
                "state_request_stride": state.state_request_stride,
                "state_head_stride": state.state_head_stride,
                "conv_request_stride": state.conv_request_stride,
                "scratch_bytes": 0,
                "dependency_event": spec.dependency_event,
                "completion_event": spec.completion_event,
                "layer_id": spec.layer_id,
                "dt_min_f32_bits": _f32_bits(config.dt_min),
                "dt_max_f32_bits": _f32_bits(config.dt_max),
                "d_mlp": config.d_mlp,
            }
        )

        registers = self.registers
        descriptor_high = descriptor_region.address >> 32
        descriptor_low = descriptor_region.address & 0xFFFFFFFF
        builder = IsaBuilder().comment(
            f"Mamba {spec.subop.name.lower()} context={spec.context_id} layer={spec.layer_id}"
        )
        for register, value in (
            (registers.address_high_gp, descriptor_high),
            (registers.address_low_gp, descriptor_low),
            (registers.context_gp, spec.context_id),
            (registers.descriptor_offset_gp, 0),
        ):
            for line in load_large_int(register, value):
                builder.raw(line)
        builder.instr(
            "C_SET_ADDR_REG",
            addr(registers.descriptor_hbm_reg),
            gp(registers.address_high_gp),
            gp(registers.address_low_gp),
        )
        builder.instr(
            "X_MAMBA",
            gp(registers.context_gp),
            gp(registers.descriptor_offset_gp),
            addr(registers.descriptor_hbm_reg),
            spec.queue_id,
            int(spec.subop),
        )
        command_word = encode_mamba_instruction(
            registers.context_gp,
            registers.descriptor_offset_gp,
            registers.descriptor_hbm_reg,
            spec.queue_id,
            int(spec.subop),
        )
        return MambaProgram(
            assembly=builder.render(),
            command_word=command_word,
            descriptor=MemoryImage(descriptor_region, descriptor_data),
            completion=completion_image,
            state=state,
            memory_map=self.arena.regions,
        )
