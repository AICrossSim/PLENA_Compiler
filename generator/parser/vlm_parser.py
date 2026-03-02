import types
from typing import Dict, Any, Optional, List
from pathlib import Path
from PIL import Image
import torch
import torch.fx as fx
from transformers import AutoModel, AutoConfig, AutoProcessor, PreTrainedModel, Qwen3VLForConditionalGeneration, Qwen3VLConfig
from torch.export import draft_export
from plena_backend import PLENA_BACKEND
import torch.nn as nn
import inspect

class VLMModelParser:
    def __init__(self, model_name_or_path: str):
        self.model_name_or_path = model_name_or_path
        self.config = None
        self.model = None
        self.processor = None
        self.symbolic_graph = None
        self.backend = None
    
    def load_model(self):
        try:
            name = self.model_name_or_path.lower()
            # ---------- Qwen3-VL ----------
            # This is required due to the method of loading QWen3 model are unique
            if "qwen3" in name:
                print(f"===== Loading {name} from pretrained =====")
                self.config = Qwen3VLConfig.from_pretrained(
                    self.model_name_or_path,
                    trust_remote_code=True,
                )
                self.processor = AutoProcessor.from_pretrained(self.model_name_or_path)
                self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                    self.model_name_or_path,
                    config=self.config,
                    dtype=torch.float32,
                    trust_remote_code=True,
                )
                visual = self.model.model.visual

            # ---------- fallback ----------
            else:
                self.config = AutoConfig.from_pretrained(self.model_name_or_path)
                self.model = AutoModel.from_pretrained(self.model_name_or_path, torch_dtype=torch.float32)
                self.model.eval()

        except Exception as e:
            raise RuntimeError(
                f"Failed to load model {self.model_name_or_path}: {e}"
            )
            
    def load_inputs(self, inputs):
        print("===== Loading inputs =====")
        self.inputs = inputs
        
    def _use_torch_fx(self) -> bool:
        if not isinstance(self.model, PreTrainedModel):
            return False

        return self.model.config.model_type in {
            "qwen3_vl",
            "qwen2_vl",
            "qwen_vl",
        }
    
    def _torch_fx_symbolic_graph(self) -> fx.GraphModule:
        if self.inputs is None:
            raise RuntimeError("Failed to load inputs")

        exported = torch.export.export(
            self.model,
            args=(),
            kwargs={
                **self.inputs,
                "use_cache": False,
                "return_dict": False,
            }
        )
        gm: fx.GraphModule = exported.graph_module
        return gm
    
    def load_backend(self,backend_fn=None,mode="default"):
        
        if backend_fn is None:
            BACKEND = PLENA_BACKEND()
            if mode == "default":
                print("===== Load backend with plena_backend_ops_match =====")
                self.backend = BACKEND.plena_backend_ops_match
            elif mode == "print":
                print("===== Load backend with plena_backend_print_gm =====")
                self.backend = BACKEND.plena_backend_print_gm
            elif mode == "dump":
                print("===== Load backend with plena_backend_dump_gm =====")
                self.backend = BACKEND.plena_backend_dump_gm
            elif mode == "viz":
                print("===== Load backend with plena_backend_svg_gm =====")
                self.backend = BACKEND.plena_backend_svg_gm
    
    def torch_compile_with_plena_backend(self, model = None, inputs = {}):
        if model is None:
            self.load_model()
        
        compiled = torch.compile(model, backend=self.backend)
        return compiled(**inputs, use_cache=False)
        
    
    def create_symbolic_graph(self):
        """Create a symbolic graph with execution orders"""
        if self.model is None:
            self.load_model()

        if self._use_torch_fx():
            print("Using torch fx")
            return self._torch_fx_symbolic_graph()
    
    def print_summary(self):
        ...
    def print_symbolic_graph_details(self):
        ...

    def trace_leaf_modules(self, model: nn.Module, forward_kwargs: dict):
        logs = []
        handles = []

        def is_leaf(m: nn.Module) -> bool:
            return len(list(m.children())) == 0

        def shape_of(x):
            if isinstance(x, torch.Tensor):
                return list(x.shape), str(x.dtype), str(x.device)
            return type(x).__name__

        def hook(name: str):
            def _hook(mod, args, out):
                in_shapes = [shape_of(a) for a in args]
                if isinstance(out, (tuple, list)):
                    out_shapes = [shape_of(o) for o in out]
                else:
                    out_shapes = shape_of(out)
                logs.append({
                    "name": name,
                    "type": type(mod).__name__,
                    "in": in_shapes,
                    "out": out_shapes,
                })
            return _hook

        for name, m in model.named_modules():
            if is_leaf(m):
                print(f"hooked: {name}")
                handles.append(m.register_forward_hook(hook(name)))

        with torch.no_grad():
            _ = model(**forward_kwargs)

        for h in handles:
            h.remove()

        return logs     

def template_qwen3_vl_inputs(
    processor: AutoProcessor,
    image_path: str,
    text: str = "Describe this image.",
    device: Optional[torch.device] = None,
    add_generation_prompt: bool = True,
) -> Dict[str, torch.Tensor]:
    
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": text},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        return_dict=True,
        return_tensors="pt",
    )

    return dict(inputs)


if __name__ == "__main__":
    TEST_MODE = "compile" # ["trace", "compile", "export"]
    # ===== config parser =====
    parser = VLMModelParser("Qwen/Qwen3-VL-2B-Instruct")
    parser.load_backend(mode = "default") # ["default", "print", "dump", "viz"]
    parser.load_model()
    inputs = template_qwen3_vl_inputs(parser.processor, "./inputs/img/image.png")
    parser.load_inputs(inputs)
    # ===== run tests =====
    if TEST_MODE == "trace":
        logs = parser.trace_leaf_modules(parser.model.model, {**inputs, "use_cache": False, "return_dict": False})
        def log_print(f, s):
            print(s)
            f.write(s + "\n")
        with open("/home/yw7422/FYP/Coprocessor_for_Llama/compiler/parser/outputs/trace.txt", "w") as f:
            for i, r in enumerate(logs):
                log_print(
                    f,
                    f"{i:04d} {r['name']} {r['type']} in={r['in']} out={r['out']}"
                )

        vision = [r for r in logs if r["name"].startswith("visual.")]
        lang  = [r for r in logs if r["name"].startswith("language_model.")]
        print("vision", len(vision), "lang", len(lang))
    elif TEST_MODE == "compile":
        parser.torch_compile_with_plena_backend(parser.model ,inputs)
    elif TEST_MODE == "export":
        parser.create_symbolic_graph()