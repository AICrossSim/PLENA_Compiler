"""Generate a full 52-layer Nemotron 3 body trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aten.mamba.scheduler import CachePolicy, MambaScheduleConfig, SchedulePhase
from aten.state import (
    PrecisionCode,
    ResidencyTarget,
    apply_residency_plan,
    load_dse_residency_plan,
)
from aten.state.isa_lowering import lower_mamba_trace_to_existing_isa
from aten.state.projection import ProjectionLayout

from .scheduler import NEMOTRON3_PATTERN, Nemotron3HybridScheduler


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", type=SchedulePhase, choices=SchedulePhase, required=True)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--decode-tokens", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument(
        "--state-precision",
        type=lambda value: PrecisionCode[value.upper()],
        default=PrecisionCode.BF16,
    )
    parser.add_argument(
        "--conv-state-precision",
        type=lambda value: PrecisionCode[value.upper()],
        help="defaults to --state-precision when omitted",
    )
    parser.add_argument("--dse-report", type=Path)
    parser.add_argument(
        "--dse-target",
        type=ResidencyTarget,
        choices=ResidencyTarget,
        default=ResidencyTarget.CAPACITY_KNEE,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mamba-physical-asm-output", type=Path)
    parser.add_argument("--mamba-physical-output", type=Path)
    parser.add_argument("--mamba-descriptor-image", type=Path)
    parser.add_argument("--mamba-layout-descriptor-image", type=Path)
    parser.add_argument(
        "--projection-layout",
        type=ProjectionLayout,
        choices=ProjectionLayout,
        default=ProjectionLayout.GROUP_MAJOR_SKEWED,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = MambaScheduleConfig(
        phase=args.phase,
        sequence_length=args.sequence_length if args.phase == SchedulePhase.PREFILL else 1,
        decode_tokens=args.decode_tokens,
        chunk_size=args.chunk_size,
        state_precision=args.state_precision,
        conv_state_precision=args.conv_state_precision,
        cache_policy=CachePolicy.NONE,
        projection_layout=args.projection_layout,
    )
    if args.dse_report is not None:
        plan = load_dse_residency_plan(
            args.dse_report,
            state_precision=config.state_precision,
            layer_ids=tuple(
                index for index, symbol in enumerate(NEMOTRON3_PATTERN) if symbol == "M"
            ),
            batch_size=config.batch_size,
            target=args.dse_target,
        )
        config = apply_residency_plan(config, plan)
    trace = Nemotron3HybridScheduler(config).build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(trace.to_dict(), indent=2) + "\n")
    if (
        args.mamba_physical_asm_output
        or args.mamba_physical_output
        or args.mamba_descriptor_image
        or args.mamba_layout_descriptor_image
    ):
        physical = lower_mamba_trace_to_existing_isa(trace.mamba_trace)
        if args.mamba_physical_asm_output is not None:
            args.mamba_physical_asm_output.parent.mkdir(parents=True, exist_ok=True)
            args.mamba_physical_asm_output.write_text(physical.assembly)
        if args.mamba_physical_output is not None:
            args.mamba_physical_output.parent.mkdir(parents=True, exist_ok=True)
            args.mamba_physical_output.write_text(
                json.dumps(physical.to_dict(), indent=2) + "\n"
            )
        if args.mamba_descriptor_image is not None:
            args.mamba_descriptor_image.parent.mkdir(parents=True, exist_ok=True)
            args.mamba_descriptor_image.write_bytes(physical.descriptor_image)
        if args.mamba_layout_descriptor_image is not None:
            args.mamba_layout_descriptor_image.parent.mkdir(parents=True, exist_ok=True)
            args.mamba_layout_descriptor_image.write_bytes(
                physical.layout_descriptor_image
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
