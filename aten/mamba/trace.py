"""Command-line trace generator for the proposed Nemotron 3 Mamba scheduler."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contract import PrecisionCode, ProjectionLayout
from .scheduler import (
    CachePolicy,
    MambaScheduleConfig,
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
        "--layout",
        choices=[item.name.lower() for item in ProjectionLayout],
        default="group_major_skewed",
    )
    parser.add_argument("--output", type=Path)
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
        projection_layout=ProjectionLayout[args.layout.upper()],
    )
    rendered = json.dumps(Nemotron3MambaScheduler(config).build().to_dict(), indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
