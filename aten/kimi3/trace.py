"""Generate the full Kimi K3 KDA/MLA/LatentMoE structural trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aten.kda.scheduler import (
    KIMI_K3_KDA_LAYERS,
    KdaScheduleConfig,
    KimiK3KdaScheduler,
)
from aten.mamba.scheduler import CachePolicy, SchedulePhase
from aten.state import (
    PrecisionCode,
    apply_residency_plan,
    build_capacity_residency_plan,
    kda_resident_bytes,
)
from aten.state.hbm_image import assemble_words, build_memory_images, render_mem_image
from aten.state.isa_lowering import lower_kda_trace_to_existing_isa
from aten.state.projection import ProjectionLayout

from .scheduler import KimiK3HybridScheduler


MIB = 1024 * 1024


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", type=SchedulePhase, choices=SchedulePhase, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--decode-tokens", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--state-cache-entries", type=int, default=0)
    parser.add_argument(
        "--state-cache-mib",
        type=int,
        help="Build the explicit pinned KDA request/layer map from byte capacity",
    )
    parser.add_argument("--cache-policy", type=CachePolicy, choices=CachePolicy, default=CachePolicy.NONE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kda-physical-output", type=Path)
    parser.add_argument("--kda-physical-asm-output", type=Path)
    parser.add_argument("--kda-descriptor-image", type=Path)
    parser.add_argument("--kda-layout-descriptor-image", type=Path)
    parser.add_argument(
        "--kda-physical-mem-output",
        type=Path,
        help="Assembled hex words, the format the emulator's --opcode expects",
    )
    parser.add_argument(
        "--descriptor-base",
        type=lambda value: int(value, 0),
        default=0x7000_0000,
        help="Where the descriptor image lives. Must sit below --hbm-arena-base.",
    )
    parser.add_argument(
        "--hbm-arena-base",
        type=lambda value: int(value, 0),
        help=(
            "Pack every HBM region into one arena starting here. Required to run "
            "the program on the emulator: the sparse default bases span 24.5 GiB, "
            "which the emulator's flat HBM allocation cannot cover. Omit to keep "
            "the previously emitted sparse descriptors byte-identical."
        ),
    )
    parser.add_argument(
        "--memory-image-dir",
        type=Path,
        help="Emit the emulator's --hbm/--fpsram/--intsram preloads plus a manifest",
    )
    parser.add_argument("--memory-image-stem", default="kimi_k3_kda")
    parser.add_argument(
        "--projection-layout",
        type=ProjectionLayout,
        choices=ProjectionLayout,
        default=ProjectionLayout.GROUP_MAJOR_SKEWED,
    )
    args = parser.parse_args()

    config = KdaScheduleConfig(
        phase=args.phase,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length if args.phase == SchedulePhase.PREFILL else 1,
        decode_tokens=args.decode_tokens,
        chunk_size=args.chunk_size,
        state_cache_entries=args.state_cache_entries,
        cache_policy=args.cache_policy,
        state_precision=PrecisionCode.FP32,
        conv_state_precision=PrecisionCode.BF16,
        hbm_arena_base=args.hbm_arena_base,
        projection_layout=args.projection_layout,
    )
    if args.state_cache_mib is not None:
        if args.state_cache_entries or args.cache_policy != CachePolicy.NONE:
            raise ValueError("--state-cache-mib cannot be combined with manual cache settings")
        plan = build_capacity_residency_plan(
            model_key="kimi_k3_full_text",
            capacity_bytes=args.state_cache_mib * MIB,
            entry_bytes=kda_resident_bytes(PrecisionCode.FP32, PrecisionCode.BF16),
            state_precision=PrecisionCode.FP32,
            layer_ids=KIMI_K3_KDA_LAYERS,
            batch_size=args.batch_size,
            source=f"cli:{args.state_cache_mib}MiB",
        )
        config = apply_residency_plan(config, plan)
    trace = KimiK3HybridScheduler(config).build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(trace.to_dict(), indent=2) + "\n")

    wants_physical = (
        args.kda_physical_output
        or args.kda_physical_asm_output
        or args.kda_physical_mem_output
        or args.kda_descriptor_image
        or args.kda_layout_descriptor_image
        or args.memory_image_dir
    )
    if wants_physical:
        physical = lower_kda_trace_to_existing_isa(
            trace.kda_trace, descriptor_base=args.descriptor_base
        )
        if args.kda_physical_output:
            args.kda_physical_output.parent.mkdir(parents=True, exist_ok=True)
            args.kda_physical_output.write_text(json.dumps(physical.to_dict(), indent=2) + "\n")
        if args.kda_physical_asm_output:
            args.kda_physical_asm_output.parent.mkdir(parents=True, exist_ok=True)
            args.kda_physical_asm_output.write_text(physical.assembly)
        if args.kda_physical_mem_output:
            args.kda_physical_mem_output.parent.mkdir(parents=True, exist_ok=True)
            args.kda_physical_mem_output.write_text(
                render_mem_image(assemble_words(physical.assembly))
            )
        if args.kda_descriptor_image:
            args.kda_descriptor_image.parent.mkdir(parents=True, exist_ok=True)
            args.kda_descriptor_image.write_bytes(physical.descriptor_image)
        if args.kda_layout_descriptor_image:
            args.kda_layout_descriptor_image.parent.mkdir(parents=True, exist_ok=True)
            args.kda_layout_descriptor_image.write_bytes(
                physical.layout_descriptor_image
            )
        if args.memory_image_dir:
            layout = KimiK3KdaScheduler(config).hbm_layout()
            images = build_memory_images(
                descriptor_image=physical.descriptor_image,
                descriptor_base=physical.descriptor_base,
                layout_descriptor_image=physical.layout_descriptor_image,
                layout_descriptor_base=physical.layout_descriptor_base,
                arena_base=layout.arena_base,
                arena_bytes=layout.realized_arena_bytes(len(trace.kda_trace.events)),
            )
            written = images.write(args.memory_image_dir, stem=args.memory_image_stem)
            print(json.dumps({name: str(path) for name, path in written.items()}, indent=2))


if __name__ == "__main__":
    main()
