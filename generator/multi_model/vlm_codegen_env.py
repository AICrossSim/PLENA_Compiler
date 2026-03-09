"""
Environment and registration layer for VLM assembly code generation.
"""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

_THIS_DIR = Path(__file__).parent
_PROJECT_ROOT = _THIS_DIR.parent.parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from vlm_codegen_context import VLMSharedCodegenContext


DEFAULT_HW: dict[str, Any] = {
    "mlen": 512,
    "blen": 4,
    "vlen": 16,
    "real_data_ratio": 1.125,
    "alive_registers": [i for i in range(15)],
}

DEFAULT_SCHED: dict[str, Any] = {
    "activation_base_address": 0x0000,
    "memory_layout": {
        "vector_sram_addr": {
            "block1": 0x0000,
            "block2": 0x1000,
            "block3": 0x2000,
            "block5": 0x3000,
        },
        "fp_sram": {
            "one": 0x0000,
            "half": 0x0001,
            "two": 0x0002,
            "neg_one": 0x0003,
            "gelu_scale": 0x0004,
            "gelu_cubic": 0x0005,
            "silu_e": 0x0100,
            "eps": 0x0200,
            "hid_reciprocal": 0x0300,
        },
    },
    "register_assignment": {
        "hbm_addr_reg": {
            "token_table_offset": 1,
            "q_weight_offset": 2,
            "k_weight_offset": 3,
            "v_weight_offset": 4,
            "ffn_weight_offset": 5,
            "rope_params_offset": 6,
            "previous_activation_offset": 7,
        }
    },
}


TemplateFn = Callable[..., str]


def _stub_asm(template_name: str) -> TemplateFn:
    def _fn(*args, **kwargs) -> str:
        return f"; [stub] {template_name} — asm_templates not on PYTHONPATH\n"

    _fn.__name__ = template_name
    return _fn


class VLMCodegenEnvironment:
    """Owns template imports, module registration, hardware, and scheduler state."""

    def __init__(
        self,
        hw: dict[str, Any] | None = None,
        sched: dict[str, Any] | None = None,
    ) -> None:
        self.hw = deepcopy(DEFAULT_HW if hw is None else hw)
        self.sched = deepcopy(DEFAULT_SCHED if sched is None else sched)
        self.templates_ok = False
        self.templates: dict[str, TemplateFn] = {}      # Note: this will eventually be removed as handler will generate using function in asm_lib
        self.type_registry: dict[str, str] = {}
        self.shared_context = self._new_shared_context()
        self._load_templates()
        self._register_default_types()
        self._bref_hw()

    def _new_shared_context(self) -> VLMSharedCodegenContext:
        return VLMSharedCodegenContext(
            mlen=int(self.hw_value("mlen", 64)),
            blen=int(self.hw_value("blen", 4)),
            real_data_ratio=float(self.hw_value("real_data_ratio", 1.125)),
        )

    # Note: this will eventually be removed as handler will generate using function in asm_lib
    def _load_templates(self) -> None:
        try:
            from asm_templates import (  # type: ignore[import]
                batched_matmul_asm,
                elementwise_add_asm,
                embedding_asm,
                ffn_asm,
                flash_attn_asm,
                layer_norm_asm,
                preload_addr_reg_asm,
                projection_asm,
                rms_norm_asm,
            )

            self.templates_ok = True
            self.templates = {
                "batched_matmul": batched_matmul_asm,
                "elementwise_add": elementwise_add_asm,
                "embedding": embedding_asm,
                "ffn": ffn_asm,
                "flash_attn": flash_attn_asm,
                "layer_norm": layer_norm_asm,
                "preload_addr_reg": preload_addr_reg_asm,
                "projection": projection_asm,
                "rms_norm": rms_norm_asm,
            }
        except ImportError:
            self.templates_ok = False
            self.templates = {
                "batched_matmul": _stub_asm("batched_matmul_asm"),
                "elementwise_add": _stub_asm("elementwise_add_asm"),
                "embedding": _stub_asm("embedding_asm"),
                "ffn": _stub_asm("ffn_asm"),
                "flash_attn": _stub_asm("flash_attn_asm"),
                "layer_norm": _stub_asm("layer_norm_asm"),
                "preload_addr_reg": _stub_asm("preload_addr_reg_asm"),
                "projection": _stub_asm("projection_asm"),
                "rms_norm": _stub_asm("rms_norm_asm"),
            }

    def _register_default_types(self) -> None:
        # type : operation
        self.register_types(
            {
                "Embedding": "embedding",
                "Linear": "linear",
                "LayerNorm": "layer_norm",
                "Conv3d": "conv3d",
                "Qwen3VLTextRMSNorm": "rms_norm",
                "Qwen3VLTextAttention": "text_attention",
                "Qwen3VLTextMLP": "ffn",
                "Qwen3VLVisionAttention": "vision_attention",
                "Qwen3VLVisionMLP": "vision_mlp_plena",
                "Qwen3VLVisionPatchEmbed": "conv3d",
                "Qwen3VLVisionPatchMerger": "linear",
                "Qwen2VLTextRMSNorm": "rms_norm",
                "Qwen2VLTextAttention": "text_attention",
                "Qwen2VLTextMLP": "ffn",
                "Qwen2VLVisionAttention": "vision_attention",
                "Qwen2VLVisionMLP": "vision_mlp_plena",
                "elementwise_add": "elementwise_add",
                "mlp": "mlp",
            }
        )

    def _bref_hw(self):
        print("===== Hardware configuration =====")
        for key, value in self.hw.items():
            print(f"{key}: {value}")
        print("=================================")
    
    def register_type(self, module_type: str, operation_key: str) -> None:
        self.type_registry[module_type] = operation_key

    def register_types(self, mapping: dict[str, str]) -> None:
        for module_type, operation_key in mapping.items():
            self.register_type(module_type, operation_key)

    def operation_for(self, node_type: str) -> str | None:
        return self.type_registry.get(node_type)

    # Note: this will eventually be removed as handler will generate using function in asm_lib
    def template(self, template_key: str) -> TemplateFn:
        return self.templates[template_key]

    def reset_shared_context(self, nodes: list[dict[str, Any]]) -> VLMSharedCodegenContext:
        self.shared_context = self._new_shared_context()
        self.shared_context.prepare_graph(nodes)
        self._register_default_constants()
        return self.shared_context

    def _register_default_constants(self) -> None:
        for name in ("one", "half", "two", "neg_one", "gelu_scale", "gelu_cubic"):
            address = self.fp_mem(name, -1)
            self.shared_context.register_constant(name, None if address < 0 else address)

    def hw_value(self, key: str, default: Any) -> Any:
        return self.hw.get(key, default)

    def reg(self, name: str, default: int = 0) -> int:
        return (
            self.sched.get("register_assignment", {})
            .get("hbm_addr_reg", {})
            .get(name, default)
        )

    def mem(self, block: str, default: int = 0) -> int:
        return (
            self.sched.get("memory_layout", {})
            .get("vector_sram_addr", {})
            .get(block, default)
        )

    def fp_mem(self, key: str, default: int = 0) -> int:
        return (
            self.sched.get("memory_layout", {})
            .get("fp_sram", {})
            .get(key, default)
        )
