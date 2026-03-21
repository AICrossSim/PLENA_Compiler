"""
Compatibility wrapper for VLM assembly code generation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    import sys

    _PROJECT_ROOT = Path(__file__).resolve().parent.parent
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

    from multi_model import INPUTS_DIR, OUTPUTS_DIR
    from multi_model.vlm_codegen_env import DEFAULT_HW, DEFAULT_SCHED, VLMCodegenEnvironment
    from multi_model.vlm_codegen_generator import VLMAssemblyGenerator
    from multi_model.vlm_parser import VLMModelParser, template_qwen3_vl_inputs
else:
    from . import INPUTS_DIR, OUTPUTS_DIR
    from .vlm_codegen_env import DEFAULT_HW, DEFAULT_SCHED, VLMCodegenEnvironment
    from .vlm_codegen_generator import VLMAssemblyGenerator
    from .vlm_parser import VLMModelParser, template_qwen3_vl_inputs


def vlm_codegen(
    nodes: list[dict],
    model_info: dict[str, Any],
    mode: str = "default",
    hw: dict[str, Any] | None = None,
    sched: dict[str, Any] | None = None,
) -> str:
    """Generate PLENA assembly from traced VLM nodes."""
    env = VLMCodegenEnvironment(hw=hw, sched=sched)
    generator = VLMAssemblyGenerator(env)
    if mode == "debug":
        generator.debug_mode(True)
    return generator.generate(nodes, model_info)


__all__ = [
    "DEFAULT_HW",
    "DEFAULT_SCHED",
    "Path",
    "VLMAssemblyGenerator",
    "VLMCodegenEnvironment",
    "VLMModelParser",
    "template_qwen3_vl_inputs",
    "vlm_codegen",
]


if __name__ == "__main__":
    import torch

    MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"
    IMAGE_PATH = INPUTS_DIR / "img" / "image.png"

    parser = VLMModelParser(MODEL_NAME)
    parser.load_model()
    inputs = template_qwen3_vl_inputs(parser.processor, IMAGE_PATH)

    trace_tree = parser.trace_leaf_modules(
        parser.model.model,
        {**inputs, "use_cache": False, "return_dict": False},
    )
    nodes = parser.flattened_traced_tree
    if nodes is None:
        raise RuntimeError("flattened_traced_tree was not populated after tracing")

    cfg = parser.config
    txt_cfg = cfg.text_config
    vis_cfg = cfg.vision_config

    model_info: dict[str, Any] = {
        "model_name": MODEL_NAME,
        "batch_size": 1,
        "hidden_size": txt_cfg.hidden_size,
        "intermediate_size": txt_cfg.intermediate_size,
        "num_attention_heads": txt_cfg.num_attention_heads,
        "num_key_value_heads": txt_cfg.num_key_value_heads,
        "num_layers": txt_cfg.num_hidden_layers,
        "head_dim": getattr(txt_cfg, "head_dim", txt_cfg.hidden_size // txt_cfg.num_attention_heads),
        "vocab_size": txt_cfg.vocab_size,
        "vision_hidden_size": getattr(vis_cfg, "hidden_size", txt_cfg.hidden_size),
        "vision_head_dim": getattr(
            vis_cfg,
            "head_dim",
            getattr(vis_cfg, "hidden_size", txt_cfg.hidden_size)
            // getattr(vis_cfg, "num_heads", txt_cfg.num_attention_heads),
        ),
    }

    print(
        f"Model info: hidden={model_info['hidden_size']}, "
        f"layers={model_info['num_layers']}, "
        f"head_dim={model_info['head_dim']}"
    )
    print(f"Trace nodes: {len(nodes)}")

    asm = vlm_codegen(nodes, model_info)

    out_path = OUTPUTS_DIR / "vlm_output.asm"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(asm)

    print(f"Generated {len(asm.splitlines())} ASM lines -> {out_path}")
