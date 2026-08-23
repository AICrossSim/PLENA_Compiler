"""Command-line trace generator for Kimi K3 KDA X_STATE scheduling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aten.mamba.scheduler import CachePolicy, SchedulePhase
from aten.state import (
    PrecisionCode,
    apply_residency_plan,
    build_capacity_residency_plan,
    kda_resident_bytes,
    lower_state_trace,
)
from aten.state.isa_lowering import lower_kda_trace_to_existing_isa
from aten.state.projection import ProjectionLayout

from .scheduler import KIMI_K3_KDA_LAYERS, KdaScheduleConfig, KimiK3KdaScheduler


MIB = 1024 * 1024


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=[item.value for item in SchedulePhase], required=True
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--decode-tokens", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--state-cache-entries", type=int, default=0)
    parser.add_argument(
        "--state-cache-mib",
        type=int,
        help="Generate an explicit pinned KDA residency map from this byte capacity",
    )
    parser.add_argument("--async-pipeline", action="store_true")
    parser.add_argument(
        "--cache-policy",
        choices=[item.value for item in CachePolicy],
        default=CachePolicy.NONE.value,
    )
    parser.add_argument(
        "--state-precision",
        choices=[item.name.lower() for item in PrecisionCode],
        default=PrecisionCode.FP32.name.lower(),
    )
    parser.add_argument(
        "--activation-precision",
        choices=[item.name.lower() for item in PrecisionCode],
        default=PrecisionCode.BF16.name.lower(),
    )
    parser.add_argument(
        "--conv-state-precision",
        choices=[item.name.lower() for item in PrecisionCode],
        default=PrecisionCode.BF16.name.lower(),
    )
    parser.add_argument(
        "--parameter-precision",
        choices=[item.name.lower() for item in PrecisionCode],
        default=PrecisionCode.BF16.name.lower(),
    )
    parser.add_argument(
        "--descriptor-base", type=lambda value: int(value, 0), default=0x7000_0000
    )
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
    parser.add_argument(
        "--projection-spill-write-values-per-cycle", type=int, default=16
    )
    bypass = parser.add_mutually_exclusive_group()
    bypass.add_argument(
        "--enable-projection-bypass",
        dest="projection_direct_bypass",
        action="store_true",
        help="Experimental: requires a field-aware reorder buffer for independent KDA projections",
    )
    bypass.add_argument(
        "--disable-projection-bypass",
        dest="projection_direct_bypass",
        action="store_false",
    )
    parser.set_defaults(projection_direct_bypass=False)
    parser.add_argument("--state-head-lanes", type=int, default=8)
    parser.add_argument("--state-head-dim-lanes", type=int, default=4)
    parser.add_argument("--state-dim-lanes", type=int, default=8)
    parser.add_argument("--kda-q-bank-rotation", type=int, default=0)
    parser.add_argument("--kda-k-bank-rotation", type=int, default=8)
    parser.add_argument("--kda-v-bank-rotation", type=int, default=0)
    parser.add_argument("--kda-decay-bank-rotation", type=int, default=0)
    parser.add_argument("--kda-beta-bank-rotation", type=int, default=0)
    parser.add_argument("--kda-beta-group-stride", type=int, default=1)
    parser.add_argument("--matrix-input-features", type=int, default=7168)
    args = parser.parse_args()
    phase = SchedulePhase(args.phase)
    config = KdaScheduleConfig(
        phase=phase,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length if phase == SchedulePhase.PREFILL else 1,
        decode_tokens=args.decode_tokens,
        chunk_size=args.chunk_size,
        state_cache_entries=args.state_cache_entries,
        cache_policy=CachePolicy(args.cache_policy),
        state_precision=PrecisionCode[args.state_precision.upper()],
        conv_state_precision=PrecisionCode[args.conv_state_precision.upper()],
        activation_precision=PrecisionCode[args.activation_precision.upper()],
        parameter_precision=PrecisionCode[args.parameter_precision.upper()],
        async_pipeline=args.async_pipeline,
        projection_layout=args.projection_layout,
        projection_buffer_banks=args.projection_buffer_banks,
        projection_buffer_ports_per_bank=args.projection_buffer_ports_per_bank,
        projection_fifo_values=args.projection_fifo_values,
        matrix_result_burst_values=args.matrix_result_burst_values,
        projection_spill_write_values_per_cycle=args.projection_spill_write_values_per_cycle,
        projection_direct_bypass=args.projection_direct_bypass,
        state_head_lanes=args.state_head_lanes,
        state_head_dim_lanes=args.state_head_dim_lanes,
        state_dim_lanes=args.state_dim_lanes,
        kda_q_bank_rotation=args.kda_q_bank_rotation,
        kda_k_bank_rotation=args.kda_k_bank_rotation,
        kda_v_bank_rotation=args.kda_v_bank_rotation,
        kda_decay_bank_rotation=args.kda_decay_bank_rotation,
        kda_beta_bank_rotation=args.kda_beta_bank_rotation,
        kda_beta_group_stride=args.kda_beta_group_stride,
        matrix_input_features=args.matrix_input_features,
    )
    if args.state_cache_mib is not None:
        if args.state_cache_entries or CachePolicy(args.cache_policy) != CachePolicy.NONE:
            raise ValueError("--state-cache-mib cannot be combined with manual cache settings")
        capacity_bytes = args.state_cache_mib * MIB
        plan = build_capacity_residency_plan(
            model_key="kimi_k3_kda",
            capacity_bytes=capacity_bytes,
            entry_bytes=kda_resident_bytes(
                config.state_precision,
                config.conv_state_precision or config.state_precision,
            ),
            state_precision=config.state_precision,
            layer_ids=KIMI_K3_KDA_LAYERS,
            batch_size=config.batch_size,
            source=f"cli:{args.state_cache_mib}MiB",
        )
        config = apply_residency_plan(config, plan)
    trace = KimiK3KdaScheduler(config).build()
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
        physical = lower_kda_trace_to_existing_isa(
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
