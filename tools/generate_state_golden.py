#!/usr/bin/env python3
"""Generate byte-exact X_STATE descriptors shared with the Simulator tests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from assembler.assembly_to_binary import AssemblyToBinary
from assembler.parser import Instruction
from aten.state import (
    KdaPayload,
    Mamba2Payload,
    PrecisionCode,
    StateDescriptor,
    StateSubop,
    encode_instruction,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "spec" / "x_state_v2_golden.json"


def _descriptor_entry(descriptor: StateDescriptor) -> dict[str, str]:
    packed = descriptor.pack()
    return {
        "sha256": hashlib.sha256(packed).hexdigest(),
        "hex": packed.hex(),
    }


def build_golden() -> dict[str, object]:
    assembler = AssemblyToBinary(
        str(ROOT / "doc" / "operation.svh"),
        str(ROOT / "doc" / "configuration.svh"),
    )
    policy_word = assembler._convert_to_binary(
        Instruction("C_SET_TOPK_REG", 7, None, None, None, None, None, None)
    )
    bias_word = assembler._convert_to_binary(
        Instruction("C_SET_TOPK_REG", 13, None, None, None, None, None, 1)
    )
    mamba_real = StateDescriptor(
        payload=Mamba2Payload(
            conv_weight_addr=0x1_0000,
            a_log_addr=0x2_0000,
            dt_bias_addr=0x3_0000,
            d_skip_addr=0x4_0000,
        ),
        sequence_length=257,
        token_offset=256,
        valid_tokens=1,
        state_hbm_addr=0x10_0000,
        conv_state_hbm_addr=0x40_0000,
    )
    kda_real = StateDescriptor(
        payload=KdaPayload(
            q_conv_weight_addr=0x1_0000,
            k_conv_weight_addr=0x2_0000,
            v_conv_weight_addr=0x3_0000,
            a_log_addr=0x4_0000,
            dt_bias_addr=0x5_0000,
        ),
        num_heads=96,
        chunk_size=16,
        state_precision=PrecisionCode.FP32,
        conv_state_precision=PrecisionCode.BF16,
        state_hbm_addr=0x80_0000,
        conv_state_hbm_addr=0xC0_0000,
    )
    common = {
        "batch_size": 1,
        "num_heads": 1,
        "sequence_length": 1,
        "valid_tokens": 1,
        "chunk_size": 1,
        "state_precision": PrecisionCode.FP32,
        "activation_precision": PrecisionCode.BF16,
        "parameter_precision": PrecisionCode.BF16,
        "context_id": 7,
        "request_id": 11,
        "layer_id": 13,
        "input_vram_addr": 0,
        "output_vram_addr": 64,
        "input_token_stride": 64,
        "output_token_stride": 64,
        "state_hbm_addr": 1024,
        "conv_state_hbm_addr": 1088,
    }
    mamba_tiny = StateDescriptor(
        payload=Mamba2Payload(
            head_dim=2,
            state_dim=2,
            groups=1,
            conv_kernel=2,
            xbc_offset=2,
            dt_offset=8,
            conv_weight_addr=2048,
            a_log_addr=2112,
            dt_bias_addr=2176,
            d_skip_addr=2240,
        ),
        **common,
    )
    kda_tiny = StateDescriptor(
        payload=KdaPayload(
            key_dim=2,
            value_dim=2,
            conv_kernel=2,
            q_offset=0,
            k_offset=2,
            v_offset=4,
            decay_offset=6,
            beta_offset=8,
            q_conv_weight_addr=2048,
            k_conv_weight_addr=2112,
            v_conv_weight_addr=2176,
            a_log_addr=2240,
            dt_bias_addr=2304,
            output_scale=0.5,
        ),
        **common,
    )
    return {
        "contract": "plena-x-state-v2",
        "instruction_words": {
            "step": f"{encode_instruction(1, 2, 3, 4, StateSubop.STEP):08x}",
            "fence": f"{encode_instruction(0, 0, 0, 3, StateSubop.FENCE):08x}",
        },
        "control_instruction_words": {
            "c_set_topk_policy_gp7": f"{policy_word:08x}",
            "c_set_topk_bias_gp13": f"{bias_word:08x}",
        },
        "descriptors": {
            "mamba2_real": _descriptor_entry(mamba_real),
            "kda_real": _descriptor_entry(kda_real),
            "mamba2_tiny": _descriptor_entry(mamba_tiny),
            "kda_tiny": _descriptor_entry(kda_tiny),
        },
    }


def render_golden() -> str:
    return json.dumps(build_golden(), indent=2) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rendered = render_golden()
    if args.check:
        if not args.output.exists() or args.output.read_text(encoding="utf-8") != rendered:
            raise SystemExit(f"stale X_STATE golden: run {Path(__file__).name}")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
