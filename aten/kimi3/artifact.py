"""Build the bounded Kimi K3 decode machine-code artifact and address contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aten.kda.scheduler import KdaScheduleConfig
from aten.kimi3.program import build_kimi_k3_program
from aten.mamba.scheduler import SchedulePhase
from aten.plena.artifact import write_symbolic_decode_artifact


def build_artifact(
    output_dir: Path,
    *,
    max_machine_code_mib: float = 64.0,
    write_assembly: bool = False,
) -> dict[str, object]:
    program = build_kimi_k3_program(
        KdaScheduleConfig(phase=SchedulePhase.DECODE, decode_tokens=1)
    )
    return write_symbolic_decode_artifact(
        program,
        output_dir,
        stem="kimi_k3_decode_b1_symbolic",
        artifact_contract="plena.kimi_k3_decode_artifact/v1",
        scope="full_93_layer_single_token_decode_with_symbolic_weights",
        max_machine_code_mib=max_machine_code_mib,
        claims={
            "machine_code_is_legal": True,
            "all_93_layers_are_lowered": True,
            "weights_are_bound": False,
            "real_checkpoint_was_executed": False,
            "multi_token_mla_cache_is_supported": False,
            "prefill_is_supported": False,
        },
        extra_summary={"heads": 96},
        write_assembly=write_assembly,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--max-machine-code-mib", type=float, default=64.0)
    parser.add_argument("--write-assembly", action="store_true")
    args = parser.parse_args()
    summary = build_artifact(
        args.output_dir,
        max_machine_code_mib=args.max_machine_code_mib,
        write_assembly=args.write_assembly,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
