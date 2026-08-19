"""Command-line trace generator for the proposed Nemotron 3 Mamba scheduler."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aten.state import (
    PrecisionCode,
    ResidencyTarget,
    apply_residency_plan,
    load_dse_residency_plan,
    lower_state_trace,
)
from aten.state.projection import ProjectionLayout
from aten.state.isa_lowering import lower_mamba_trace_to_existing_isa

from .scheduler import (
    CachePolicy,
    MambaScheduleConfig,
    NEMOTRON3_MAMBA_LAYERS,
    Nemotron3MambaScheduler,
    SchedulePhase,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=[phase.value for phase in SchedulePhase], required=True
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--decode-tokens", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--state-cache-entries", type=int, default=0)
    parser.add_argument("--dse-report", type=Path)
    parser.add_argument(
        "--dse-target",
        type=ResidencyTarget,
        choices=ResidencyTarget,
        default=ResidencyTarget.CAPACITY_KNEE,
    )
    parser.add_argument("--async-pipeline", action="store_true")
    parser.add_argument(
        "--cache-policy",
        choices=[policy.value for policy in CachePolicy],
        default="none",
    )
    parser.add_argument(
        "--state-precision",
        choices=[item.name.lower() for item in PrecisionCode],
        default="fp32",
    )
    parser.add_argument(
        "--conv-state-precision",
        choices=[item.name.lower() for item in PrecisionCode],
        help="defaults to --state-precision when omitted",
    )
    parser.add_argument(
        "--activation-precision",
        choices=[item.name.lower() for item in PrecisionCode],
        default="bf16",
    )
    parser.add_argument(
        "--parameter-precision",
        choices=[item.name.lower() for item in PrecisionCode],
        default="bf16",
    )
    parser.add_argument("--descriptor-base", type=lambda value: int(value, 0), default=0x7000_0000)
    parser.add_argument("--descriptor-image", type=Path)
    parser.add_argument("--layout-descriptor-image", type=Path)
    parser.add_argument("--lowered-output", type=Path)
    parser.add_argument("--physical-asm-output", type=Path)
    parser.add_argument("--physical-output", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--projection-layout",
        type=ProjectionLayout,
        choices=ProjectionLayout,
        default=ProjectionLayout.GROUP_MAJOR_SKEWED,
    )
    parser.add_argument("--projection-buffer-banks", type=int, default=16)
    parser.add_argument("--projection-buffer-ports-per-bank", type=int, default=1)
    parser.add_argument("--projection-fifo-values", type=int, default=64)
    parser.add_argument("--matrix-result-burst-values", type=int, default=64)
    parser.add_argument("--projection-spill-write-values-per-cycle", type=int, default=16)
    parser.add_argument("--disable-projection-bypass", action="store_true")
    parser.add_argument("--state-head-lanes", type=int, default=8)
    parser.add_argument("--state-head-dim-lanes", type=int, default=4)
    parser.add_argument("--state-dim-lanes", type=int, default=8)
    parser.add_argument("--matrix-input-features", type=int, default=2688)
    return parser


def main() -> None:
    args = _parser().parse_args()
    phase = SchedulePhase(args.phase)
    config = MambaScheduleConfig(
        phase=phase,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length if phase == SchedulePhase.PREFILL else 1,
        decode_tokens=args.decode_tokens,
        chunk_size=args.chunk_size,
        state_cache_entries=args.state_cache_entries,
        cache_policy=CachePolicy(args.cache_policy),
        state_precision=PrecisionCode[args.state_precision.upper()],
        conv_state_precision=(
            PrecisionCode[args.conv_state_precision.upper()]
            if args.conv_state_precision
            else None
        ),
        activation_precision=PrecisionCode[args.activation_precision.upper()],
        parameter_precision=PrecisionCode[args.parameter_precision.upper()],
        async_pipeline=args.async_pipeline,
        projection_layout=args.projection_layout,
        projection_buffer_banks=args.projection_buffer_banks,
        projection_buffer_ports_per_bank=args.projection_buffer_ports_per_bank,
        projection_fifo_values=args.projection_fifo_values,
        matrix_result_burst_values=args.matrix_result_burst_values,
        projection_spill_write_values_per_cycle=args.projection_spill_write_values_per_cycle,
        projection_direct_bypass=not args.disable_projection_bypass,
        state_head_lanes=args.state_head_lanes,
        state_head_dim_lanes=args.state_head_dim_lanes,
        state_dim_lanes=args.state_dim_lanes,
        matrix_input_features=args.matrix_input_features,
    )
    if args.dse_report is not None:
        if args.state_cache_entries or CachePolicy(args.cache_policy) != CachePolicy.NONE:
            raise ValueError("--dse-report cannot be combined with manual cache settings")
        plan = load_dse_residency_plan(
            args.dse_report,
            state_precision=config.state_precision,
            layer_ids=NEMOTRON3_MAMBA_LAYERS,
            batch_size=config.batch_size,
            target=args.dse_target,
        )
        config = apply_residency_plan(config, plan)
    trace = Nemotron3MambaScheduler(config).build()
    rendered = json.dumps(trace.to_dict(), indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    if args.descriptor_image or args.layout_descriptor_image or args.lowered_output:
        lowered = lower_state_trace(trace, descriptor_base=args.descriptor_base)
        if args.descriptor_image:
            args.descriptor_image.parent.mkdir(parents=True, exist_ok=True)
            args.descriptor_image.write_bytes(lowered.descriptor_image)
        if args.layout_descriptor_image:
            args.layout_descriptor_image.parent.mkdir(parents=True, exist_ok=True)
            args.layout_descriptor_image.write_bytes(lowered.layout_descriptor_image)
        if args.lowered_output:
            args.lowered_output.parent.mkdir(parents=True, exist_ok=True)
            args.lowered_output.write_text(
                json.dumps(lowered.to_dict(), indent=2) + "\n"
            )
    if args.physical_asm_output or args.physical_output:
        physical = lower_mamba_trace_to_existing_isa(
            trace,
            descriptor_base=args.descriptor_base,
        )
        if args.physical_asm_output:
            args.physical_asm_output.parent.mkdir(parents=True, exist_ok=True)
            args.physical_asm_output.write_text(physical.assembly)
        if args.physical_output:
            args.physical_output.parent.mkdir(parents=True, exist_ok=True)
            args.physical_output.write_text(
                json.dumps(physical.to_dict(), indent=2) + "\n"
            )
    print(rendered)


if __name__ == "__main__":
    main()
