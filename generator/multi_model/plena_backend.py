from __future__ import annotations
from typing import Dict, Callable, Any
import importlib
import pkgutil
import torch
import torch.nn as nn
from torch.fx.passes.graph_drawer import FxGraphDrawer
from pathlib import Path
from collections import Counter

# ---------------------------------------------------------------------------
# Module class name → asm template module name.
# This is the single source of truth — no TARGETS needed inside asm_templates.
# Extend this table when adding support for new module types.
# ---------------------------------------------------------------------------
MODULE_TYPE_TO_TEMPLATE: Dict[str, str] = {
    # PyTorch built-ins
    "Linear":           "projection_asm",
    "LayerNorm":        "normalization_asm",
    "Embedding":        "embedding_asm",
    "GELU":             "gelu_asm",
    "SiLU":             "silu_asm",
    "Conv3d":           "batched_matmul_asm",
    # Qwen3-VL specific
    "Qwen3VLTextRMSNorm":       "normalization_asm",
    "Qwen3VLVisionMLP":         "ffn_asm",
    "Qwen3VLTextMLP":           "ffn_asm",
    "Qwen3VLVisionAttention":   "flash_attn_asm",
    "Qwen3VLTextAttention":     "flash_attn_asm",
    "Qwen3VLVisionPatchMerger": "projection_asm",
    "Qwen3VLVisionPatchEmbed":  "batched_matmul_asm",
    # Qwen2-VL (same templates, different class names)
    "Qwen2VLTextRMSNorm":       "normalization_asm",
    "Qwen2VLVisionMLP":         "ffn_asm",
    "Qwen2VLTextMLP":           "ffn_asm",
    "Qwen2VLVisionAttention":   "flash_attn_asm",
    "Qwen2VLTextAttention":     "flash_attn_asm",
}


class PLENA_BACKEND:
    def __init__(self):
        self.asm_file = Path("plena.asm")
        self.template_path = "compiler.asm_templates"
        self.ops_registry = {}          # call_function / call_method ops (legacy TARGETS path)
        self.module_type_registry: Dict[str, str] = {}  # call_module: class name → template
        self._clear = False
    
    def registry_canonical_key(self, target) -> str:
        if hasattr(target, "name"):
            s = str(target.name).lower()
            return s

        builtin = getattr(target, "__name__", None)
        if builtin is not None:
            builtin = builtin.lower()
        if builtin in ("add",):
            return "add"
        if builtin in ("mul",):
            return "mul"
        if builtin in ("sub",):
            return "sub"
        if builtin in ("ffn",):
            return "ffn"
        if builtin in ("gelu",):
            return "gelu"
        if builtin in ("linear",):
            return "linear"
        return str(target).lower()

    def build_plena_registry(self, package: str) -> Dict[str, str]:
        reg: Dict[str, str] = {}
        pkg = importlib.import_module(package)

        for modinfo in pkgutil.iter_modules(pkg.__path__, pkg.__name__ + "."):
            mod = importlib.import_module(modinfo.name)

            targets = getattr(mod, "TARGETS", None)
            if not targets:
                continue

            for t in targets:
                k = t if isinstance(t, str) else self.registry_canonical_key(t)
                reg[k.lower()] = modinfo.name.split(".")[-1]

        return reg
    
    def load_registry(self) -> None:
        """Legacy: populate ops_registry from TARGETS in asm_template modules."""
        self.ops_registry = self.build_plena_registry(self.template_path)

    def build_registry_from_model(self, model: nn.Module) -> None:
        """Scan model.named_modules() and register each discovered module type.

        No TARGETS needed in asm_templates — the MODULE_TYPE_TO_TEMPLATE table
        in this file is the only thing that needs updating when adding new ops.
        """
        for _, m in model.named_modules():
            cls_name = type(m).__name__
            if cls_name in MODULE_TYPE_TO_TEMPLATE:
                self.module_type_registry[cls_name] = MODULE_TYPE_TO_TEMPLATE[cls_name]

    def print_registry(self):
        print("ops_registry (call_function/method):", self.ops_registry)
        print("module_type_registry (call_module):", self.module_type_registry)
    
    def _clear_asm_file(self):
        if not self._clear:
            self.asm_file.write_text("")
            self._clear = True
        
    def plena_backend_ops_match(self, gm: torch.fx.GraphModule, example_inputs):
        self._clear_asm_file()

        def _tensor_meta_of(node: torch.fx.Node):
            tm = node.meta.get("tensor_meta", None)
            if tm is not None:
                return ("tensor_meta", tm)
            v = node.meta.get("val", None) or node.meta.get("example_value", None)
            if isinstance(v, torch.Tensor):
                return ("tensor", v)
            return (None, None)

        def _fmt_tensor_meta(kind, obj):
            if kind == "tensor_meta":
                dev = getattr(obj, "device", None)
                return f"shape={tuple(obj.shape)} dtype={obj.dtype}" + (f" device={dev}" if dev is not None else "")
            if kind == "tensor":
                return f"shape={tuple(obj.shape)} dtype={obj.dtype} device={obj.device}"
            return "?"

        def _fmt_arg(x):
            if isinstance(x, torch.fx.Node):
                kind, obj = _tensor_meta_of(x)
                return f"{x.name}:{_fmt_tensor_meta(kind, obj)}"
            if isinstance(x, torch.Tensor):
                return f"tensor:shape={tuple(x.shape)} dtype={x.dtype} device={x.device}"
            if isinstance(x, (tuple, list)):
                return "[" + ", ".join(_fmt_arg(i) for i in x) + "]"
            if isinstance(x, dict):
                return "{" + ", ".join(f"{k}={_fmt_arg(v)}" for k, v in x.items()) + "}"
            return repr(x)

        def _fmt_node_out(n: torch.fx.Node):
            kind, obj = _tensor_meta_of(n)
            return _fmt_tensor_meta(kind, obj)

        try:
            ShapeProp(gm).propagate(*example_inputs)
        except Exception as e:
            pass

        asm_lines = []
        asm_lines.append("\n ===== PLENA ASM DUMP =====")
        asm_lines.append(f"graph id: {id(gm)}")

        for n in gm.graph.nodes:
            out_meta = _fmt_node_out(n)

            if n.op in ("call_function", "call_method"):
                key = self.registry_canonical_key(n.target)
                in_args = _fmt_arg(n.args)
                in_kwargs = _fmt_arg(n.kwargs) if n.kwargs else "{}"

                if key in self.ops_registry:
                    mod_name = self.ops_registry[key]
                    asm_lines.append(
                        f"SUPPORTED {mod_name:<12s} ; "
                        f"op={n.op} key={key} target={n.target} "
                        f"in_args={in_args} in_kwargs={in_kwargs} "
                        f"out={out_meta}"
                    )
                else:
                    tname = n.target.__name__ if hasattr(n.target, "__name__") else str(n.target)
                    asm_lines.append(
                        f"UNSUPPORTED          ; "
                        f"op={n.op} target={tname} "
                        f"in_args={in_args} in_kwargs={in_kwargs} "
                        f"out={out_meta}"
                    )

            elif n.op == "call_module":
                submod = gm.get_submodule(n.target)
                cls_name = type(submod).__name__
                in_args = _fmt_arg(n.args)
                in_kwargs = _fmt_arg(n.kwargs) if n.kwargs else "{}"

                if cls_name in self.module_type_registry:
                    tmpl = self.module_type_registry[cls_name]
                    asm_lines.append(
                        f"SUPPORTED {tmpl:<12s} ; "
                        f"op=call_module type={cls_name} name={n.target} "
                        f"in_args={in_args} in_kwargs={in_kwargs} "
                        f"out={out_meta}"
                    )
                else:
                    asm_lines.append(
                        f"UNSUPPORTED          ; "
                        f"op=call_module type={cls_name} name={n.target} "
                        f"in_args={in_args} in_kwargs={in_kwargs} "
                        f"out={out_meta}"
                    )
            elif n.op == "output":
                out_val = n.args[0]
                asm_lines.append(f"output       ; output={_fmt_arg(out_val)}")

            else:
                asm_lines.append(
                    f"{n.op:12s} ; {n.target} ; out={out_meta}"
                )

        asm_text = "\n".join(asm_lines)

        print("=== PLENA ASM ===")
        print(asm_text)

        out_path = Path("/home/yw7422/FYP/Coprocessor_for_Llama/compiler/parser/outputs/plena.asm")
        out_path.open("a").write(asm_text + "\n")

        def run(*args, **kwargs):
            return gm(*args, **kwargs)

        return run
        
    def plena_backend_print_gm(self, gm: torch.fx.GraphModule, example_inputs):
        print("==== FX GraphModule ====")
        print(gm)
        print("\n==== gm.graph ====")
        print(gm.graph)
        print("\n==== tabular ====")
        gm.graph.print_tabular()

        gm.recompile()
        return gm.forward
    
    def plena_backend_dump_gm(self, gm: torch.fx.GraphModule, example_inputs):
        print("==== gm.code ====")
        print(gm.code)

        Path("/home/yw7422/FYP/Coprocessor_for_Llama/compiler/parser/outputs/captured_gm.py").write_text(gm.code)
        gm.recompile()
        return gm.forward

    def plena_backend_svg_gm(self, gm: torch.fx.GraphModule, example_inputs):
        drawer = FxGraphDrawer(gm, "compiled_graph")
        dot = drawer.get_dot_graph()
        dot.write_svg("/home/yw7422/FYP/Coprocessor_for_Llama/compiler/parser/outputs/compiled_graph.svg")

        print("Wrote compiled_graph.svg")
        gm.recompile()
        return gm.forward

class TinyBlock(nn.Module):
    def __init__(self, d_model: int = 64, hidden: int = 256):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, d_model)
        self.ln2 = nn.LayerNorm(d_model)

    @torch.compile(fullgraph=True, dynamic=True)
    def forward(self, x: torch.Tensor, use_skip: bool = True) -> torch.Tensor:
        y = self.ln1(x)
        y = self.fc2(self.act(self.fc1(y)))

        if use_skip:
            x = x + y
        else:
            x = y

        z = self.ln2(x)
        return x + z


if __name__ == "__main__":
    backend = PLENA_BACKEND()
    # backend.load_registry()
    backend.print_registry()
    model = TinyBlock()
    x = torch.randn(2, 128, 64)
    compiled = torch.compile(model, backend=backend.plena_backend_ops_match)
    compiled(x)
    
    