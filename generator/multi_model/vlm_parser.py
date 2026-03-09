import json
import operator
import types
from typing import Any, Dict, List, Optional
from pathlib import Path
from PIL import Image
import torch
import torch.fx as fx
from transformers import AutoModel, AutoConfig, AutoProcessor, PreTrainedModel, Qwen3VLForConditionalGeneration, Qwen3VLConfig
from plena_backend import PLENA_BACKEND
import torch.nn as nn
import inspect

def _static_attrs(m: nn.Module) -> dict:
    """Extract weight shapes and scalar hyperparams from a module at registration time."""
    if isinstance(m, nn.Linear):
        return {"in_features": m.in_features, "out_features": m.out_features, "bias": m.bias is not None}
    if isinstance(m, nn.LayerNorm):
        return {"normalized_shape": list(m.normalized_shape), "eps": m.eps}
    if isinstance(m, nn.Embedding):
        return {"num_embeddings": m.num_embeddings, "embedding_dim": m.embedding_dim}
    if isinstance(m, nn.Conv2d):
        return {
            "in_channels": m.in_channels,
            "out_channels": m.out_channels,
            "kernel_size": list(m.kernel_size),
            "stride": list(m.stride),
        }
    if isinstance(m, nn.Conv3d):
        return {
            "in_channels": m.in_channels,
            "out_channels": m.out_channels,
            "kernel_size": list(m.kernel_size),
            "stride": list(m.stride),
        }
    # Qwen3VLTextRMSNorm: identified by variance_epsilon + weight attrs (avoids importing the class)
    if hasattr(m, "variance_epsilon") and hasattr(m, "weight"):
        return {"hidden_size": m.weight.shape[0], "eps": m.variance_epsilon}
    # Qwen3VLVisionRotaryEmbedding: dim + theta + inv_freq buffer
    if hasattr(m, "dim") and hasattr(m, "theta") and hasattr(m, "inv_freq"):
        return {"dim": m.dim, "theta": m.theta, "freq_table_shape": list(m.inv_freq.shape)}
    # Qwen3VLTextRotaryEmbedding: mrope_section + rope_type + inv_freq buffer
    if hasattr(m, "mrope_section") and hasattr(m, "inv_freq"):
        return {
            "rope_type": m.rope_type,
            "mrope_section": list(m.mrope_section),
            "freq_table_shape": list(m.inv_freq.shape),
        }
    return {}


def _extract_norm_eps(module_name: str, module: nn.Module, attrs: Optional[dict[str, Any]] = None) -> Optional[float]:
    """Best-effort eps extraction for norm-like modules."""
    norm_hint = f"{module_name} {type(module).__name__}".lower()
    if not isinstance(module, nn.LayerNorm) and "norm" not in norm_hint:
        return None

    if isinstance(module, nn.LayerNorm):
        return float(module.eps)

    if attrs is not None:
        value = attrs.get("eps")
        if isinstance(value, (int, float)):
            return float(value)

    for attr_name in ("eps", "variance_epsilon", "layer_norm_eps", "rms_norm_eps", "norm_eps"):
        value = getattr(module, attr_name, None)
        if isinstance(value, (int, float)):
            return float(value)

    return None


# Maps torch.fx call_function targets → ISA-relevant operation type names.
# Extend this table when adding support for new operation types.
_FX_OPS: dict[Any, str] = {
    operator.add:                      "elementwise_add",
    operator.mul:                      "elementwise_mul",
    torch.nn.functional.silu:          "silu",
    torch.nn.functional.gelu:          "gelu",
    torch.nn.functional.relu:          "relu",
    torch.nn.functional.softmax:       "softmax",
}

_VISION_HINTS = ("visual", "vision", "image", "patch", "pixel")
_TEXT_HINTS = ("language", "text", "token", "embed_tokens", "lm_head")
_TEXT_INPUT_HINTS = ("input_ids", "inputs_embeds", "token", "text", "ids", "attention_mask")
_VISION_INPUT_HINTS = ("pixel_values", "image", "images", "vision", "pixel", "patch")


class VLMModelParser:
    def __init__(self, model_name_or_path: str = ""):
        self.model_name_or_path = model_name_or_path
        self.config = None
        self.model = None
        self.processor = None
        self.symbolic_graph = None
        self.backend = None
        self.plena_backend = None  # PLENA_BACKEND instance (set by load_backend)
    
    def load_model(self, model = None):
        if model is not None:
            if isinstance(model, nn.Module):
                self.model = model
            else:
                raise RuntimeError(
                    f"fail to load model from provided object: expected nn.Module, got {type(model)}"
                )
        else:
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

        # Auto-populate the module_type_registry from the loaded model hierarchy.
        # This replaces the old TARGETS-based discovery — no asm_templates changes needed.
        if self.plena_backend is not None:
            self.plena_backend.build_registry_from_model(self.model)
            print(f"===== module_type_registry populated: {list(self.plena_backend.module_type_registry.keys())} =====")

    def load_inputs(self, inputs):
        print("===== Loading inputs =====")
        self.inputs = inputs

    def _first_tensor(self, obj: Any) -> Optional[torch.Tensor]:
        """Return the first tensor found in a nested inputs structure."""
        if isinstance(obj, torch.Tensor):
            return obj
        if isinstance(obj, dict):
            for value in obj.values():
                found = self._first_tensor(value)
                if found is not None:
                    return found
        if isinstance(obj, (list, tuple)):
            for value in obj:
                found = self._first_tensor(value)
                if found is not None:
                    return found
        return None

    def _tensor_input_shapes(self, inputs: Any) -> dict[str, list[int]]:
        """Collect top-level tensor input shapes for debugging / reporting."""
        if not isinstance(inputs, dict):
            return {}
        return {
            name: list(value.shape)
            for name, value in inputs.items()
            if isinstance(value, torch.Tensor)
        }

    def _infer_num_layers(self, model: nn.Module) -> int | None:
        """Best-effort layer count for custom models."""
        for attr in ("layers", "blocks", "h"):
            value = getattr(model, attr, None)
            if isinstance(value, (nn.ModuleList, list, tuple)):
                return len(value)

        for parent_attr in ("encoder", "decoder", "model"):
            parent = getattr(model, parent_attr, None)
            if parent is None:
                continue
            for attr in ("layers", "blocks", "h", "layer"):
                value = getattr(parent, attr, None)
                if isinstance(value, (nn.ModuleList, list, tuple)):
                    return len(value)

        return 1 if any(True for _ in model.children()) else 0

    def _infer_num_attention_heads(self, model: nn.Module) -> int | None:
        """Best-effort attention-head count for custom models."""
        attention_like = False
        for name, module in model.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                return module.num_heads
            for attr in ("num_attention_heads", "num_heads", "nhead"):
                value = getattr(module, attr, None)
                if isinstance(value, int) and value > 0:
                    return value
            lname = name.lower()
            if lname.endswith(("q", "k", "v", "q_proj", "k_proj", "v_proj", "attn", "self_attn")):
                attention_like = True
            if isinstance(module, nn.Softmax):
                attention_like = True

        if attention_like:
            return 1
        return None

    def _infer_custom_model_info(self, model: nn.Module, sample_input: Optional[torch.Tensor]) -> dict[str, Any]:
        """Infer model-level metadata for custom nn.Module graphs."""
        info: dict[str, Any] = {}

        if sample_input is not None:
            if sample_input.dim() >= 1:
                info["batch_size"] = int(sample_input.shape[0])
            if sample_input.dim() in (2, 3):
                info["seq_len"] = int(sample_input.shape[1])
            if sample_input.dtype.is_floating_point and sample_input.dim() in (2, 3):
                info["hidden_size"] = int(sample_input.shape[-1])

        first_embedding: Optional[nn.Embedding] = None
        first_linear: Optional[nn.Linear] = None
        first_conv: Optional[nn.Module] = None
        first_norm_dim: Optional[int] = None
        first_eps: Optional[float] = None
        intermediate_candidates: list[int] = []

        for module in model.modules():
            if first_embedding is None and isinstance(module, nn.Embedding):
                first_embedding = module

            if first_linear is None and isinstance(module, nn.Linear):
                first_linear = module

            if first_conv is None and isinstance(module, (nn.Conv2d, nn.Conv3d)):
                first_conv = module

            if isinstance(module, nn.LayerNorm):
                if first_norm_dim is None:
                    normalized_shape = module.normalized_shape
                    first_norm_dim = int(normalized_shape[0] if isinstance(normalized_shape, (list, tuple)) else normalized_shape)
                if first_eps is None:
                    first_eps = float(module.eps)

            if hasattr(module, "variance_epsilon") and hasattr(module, "weight"):
                if first_norm_dim is None and getattr(module, "weight", None) is not None:
                    first_norm_dim = int(module.weight.shape[0])
                if first_eps is None:
                    first_eps = float(module.variance_epsilon)

            if isinstance(module, nn.Linear):
                intermediate_candidates.append(int(module.out_features))

        if "hidden_size" not in info:
            if first_embedding is not None:
                info["hidden_size"] = int(first_embedding.embedding_dim)
            elif first_linear is not None:
                info["hidden_size"] = int(first_linear.in_features)
            elif first_conv is not None:
                info["hidden_size"] = int(first_conv.out_channels)
            elif first_norm_dim is not None:
                info["hidden_size"] = first_norm_dim

        hidden_size = info.get("hidden_size")
        if hidden_size is not None:
            larger_dims = [dim for dim in intermediate_candidates if dim > hidden_size]
            if larger_dims:
                info["intermediate_size"] = max(larger_dims)

        if first_embedding is not None:
            info["vocab_size"] = int(first_embedding.num_embeddings)

        if first_eps is not None:
            info["eps"] = first_eps

        num_heads = self._infer_num_attention_heads(model)
        if num_heads is not None:
            info["num_attention_heads"] = num_heads
            info["num_key_value_heads"] = num_heads

        num_layers = self._infer_num_layers(model)
        if num_layers is not None:
            info["num_layers"] = num_layers

        if hidden_size is not None and num_heads is not None and num_heads > 0:
            info["head_dim"] = hidden_size // num_heads

        return info

    def _guess_branch_from_name(self, name: str) -> Optional[str]:
        lname = name.lower()
        if any(token in lname for token in _VISION_HINTS):
            return "vision"
        if any(token in lname for token in _TEXT_HINTS):
            return "text"
        return None

    def _branch_sample_input(self, inputs: Any, branch: str) -> Optional[torch.Tensor]:
        if not isinstance(inputs, dict):
            return self._first_tensor(inputs)

        hints = _VISION_INPUT_HINTS if branch == "vision" else _TEXT_INPUT_HINTS
        for hint in hints:
            for name, value in inputs.items():
                if isinstance(value, torch.Tensor) and hint in name.lower():
                    return value
        return None

    def _normalize_dim_value(self, value: Any) -> Optional[int | list[int]]:
        if value is None:
            return None
        if isinstance(value, torch.Size):
            value = list(value)
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, (list, tuple)) and value:
            normalized: list[int] = []
            for item in value:
                if not isinstance(item, (int, float)):
                    return None
                normalized.append(int(item))
            return normalized if len(normalized) > 1 else normalized[0]
        return None

    def _module_descendant_count(self, module: nn.Module) -> int:
        return sum(1 for _ in module.modules())

    def _find_branch_root(self, model: nn.Module, branch: str) -> Optional[nn.Module]:
        best_module: Optional[nn.Module] = None
        best_score: Optional[tuple[int, int, int, int]] = None

        for name, module in model.named_modules():
            if not name:
                continue
            branch_guess = self._guess_branch_from_name(f"{name} {type(module).__name__}")
            if branch_guess != branch:
                continue

            has_children = 1 if any(True for _ in module.children()) else 0
            descendants = self._module_descendant_count(module)
            depth = -name.count(".")
            name_len = -len(name)
            score = (has_children, descendants, depth, name_len)
            if best_score is None or score > best_score:
                best_module = module
                best_score = score

        return best_module

    def _first_named_attr(
        self,
        model: nn.Module,
        attr_names: tuple[str, ...],
    ) -> Optional[int | list[int]]:
        for attr in attr_names:
            normalized = self._normalize_dim_value(getattr(model, attr, None))
            if normalized is not None:
                return normalized

        for _, module in model.named_modules():
            for attr in attr_names:
                normalized = self._normalize_dim_value(getattr(module, attr, None))
                if normalized is not None:
                    return normalized
        return None

    def _infer_image_size(
        self,
        model: nn.Module,
        sample_input: Optional[torch.Tensor],
    ) -> Optional[int | list[int]]:
        image_size = self._first_named_attr(model, ("image_size", "img_size", "image_resolution", "input_size"))
        if image_size is not None:
            return image_size

        if isinstance(sample_input, torch.Tensor) and sample_input.dim() >= 4:
            return [int(sample_input.shape[-2]), int(sample_input.shape[-1])]
        return None

    def _infer_patch_size(self, model: nn.Module) -> Optional[int | list[int]]:
        patch_size = self._first_named_attr(model, ("patch_size", "patch_shape", "patch_resolution"))
        if patch_size is not None:
            return patch_size

        for name, module in model.named_modules():
            lname = name.lower()
            if isinstance(module, (nn.Conv2d, nn.Conv3d)) and any(token in lname for token in ("patch", "embed", "proj")):
                patch_size = self._normalize_dim_value(module.kernel_size)
                if patch_size is not None:
                    return patch_size
        return None

    def _derive_patch_num(
        self,
        image_size: Optional[int | list[int]],
        patch_size: Optional[int | list[int]],
    ) -> Optional[int]:
        if image_size is None or patch_size is None:
            return None

        if isinstance(image_size, int) and isinstance(patch_size, int):
            if patch_size <= 0 or image_size <= 0 or image_size % patch_size != 0:
                return None
            patches_per_side = image_size // patch_size
            return patches_per_side * patches_per_side

        image_dims = [image_size] if isinstance(image_size, int) else list(image_size)
        patch_dims = [patch_size] if isinstance(patch_size, int) else list(patch_size)
        if not image_dims or not patch_dims:
            return None

        if len(patch_dims) == 1 and len(image_dims) > 1:
            patch_dims = patch_dims * len(image_dims)
        elif len(image_dims) != len(patch_dims):
            dims = min(len(image_dims), len(patch_dims))
            image_dims = image_dims[-dims:]
            patch_dims = patch_dims[-dims:]

        patch_num = 1
        for img_dim, patch_dim in zip(image_dims, patch_dims):
            if patch_dim <= 0 or img_dim <= 0 or img_dim % patch_dim != 0:
                return None
            patch_num *= img_dim // patch_dim
        return patch_num

    def _infer_patch_num(
        self,
        model: nn.Module,
        image_size: Optional[int | list[int]],
        patch_size: Optional[int | list[int]],
    ) -> Optional[int]:
        patch_num = self._first_named_attr(model, ("num_patches", "patch_num", "patch_count"))
        if isinstance(patch_num, int):
            return patch_num
        return self._derive_patch_num(image_size, patch_size)

    def _infer_vlm_branch_info(self, model: nn.Module, inputs: Any) -> dict[str, dict[str, Any]]:
        branch_info: dict[str, dict[str, Any]] = {}

        for branch in ("text", "vision"):
            branch_root = self._find_branch_root(model, branch)
            if branch_root is None:
                continue

            sample_input = self._branch_sample_input(inputs, branch)
            info = self._infer_custom_model_info(branch_root, sample_input)

            if branch == "vision":
                image_size = self._infer_image_size(branch_root, sample_input)
                if image_size is not None:
                    info["image_size"] = image_size

                patch_size = self._infer_patch_size(branch_root)
                if patch_size is not None:
                    info["patch_size"] = patch_size

                patch_num = self._infer_patch_num(branch_root, image_size, patch_size)
                if patch_num is not None:
                    info["patch_num"] = patch_num

            if info:
                branch_info[branch] = info

        return branch_info

    def extract_model_info(
        self,
        model: Optional[nn.Module] = None,
        inputs: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Extract model-level metadata for code generation.

        For HuggingFace / Qwen models, prefer config-derived fields.
        For custom nn.Module graphs, fall back to lightweight structural
        inference plus any loaded runtime inputs.
        """
        if model is None:
            if self.model is None:
                self.load_model()
            model = self.model

        if model is None:
            raise RuntimeError("Failed to extract model info: no model is loaded")

        active_inputs = inputs if inputs is not None else getattr(self, "inputs", None)
        sample_input = self._first_tensor(active_inputs)

        if self.config is not None:
            cfg = self.config
            text_cfg = getattr(cfg, "text_config", cfg)
            vision_cfg = getattr(cfg, "vision_config", None)

            model_info: dict[str, Any] = {
                "model_name": self.model_name_or_path or type(model).__name__,
                "architecture": getattr(cfg, "architectures", [type(model).__name__])[0],
            }

            if sample_input is not None:
                if sample_input.dim() >= 1:
                    model_info["batch_size"] = int(sample_input.shape[0])
                if sample_input.dim() >= 2:
                    model_info["seq_len"] = int(sample_input.shape[1])

            for key, value in (
                ("hidden_size", getattr(text_cfg, "hidden_size", None)),
                ("intermediate_size", getattr(text_cfg, "intermediate_size", None)),
                ("num_attention_heads", getattr(text_cfg, "num_attention_heads", None)),
                ("num_key_value_heads", getattr(text_cfg, "num_key_value_heads", getattr(text_cfg, "num_attention_heads", None))),
                ("num_layers", getattr(text_cfg, "num_hidden_layers", None)),
                ("vocab_size", getattr(text_cfg, "vocab_size", getattr(cfg, "vocab_size", None))),
                ("context_length", getattr(text_cfg, "max_position_embeddings", getattr(cfg, "max_position_embeddings", None))),
                ("eps", getattr(text_cfg, "rms_norm_eps", getattr(text_cfg, "layer_norm_eps", None))),
            ):
                if value is not None:
                    model_info[key] = value

            hidden_size = model_info.get("hidden_size")
            num_heads = model_info.get("num_attention_heads")
            head_dim = getattr(text_cfg, "head_dim", None)
            if head_dim is not None:
                model_info["head_dim"] = head_dim
            elif hidden_size is not None and num_heads:
                model_info["head_dim"] = hidden_size // num_heads

            if vision_cfg is not None:
                vision_hidden = getattr(vision_cfg, "hidden_size", None)
                if vision_hidden is not None:
                    model_info["vision_hidden_size"] = vision_hidden
                vision_heads = getattr(vision_cfg, "num_heads", getattr(vision_cfg, "num_attention_heads", None))
                if vision_heads is not None:
                    model_info["vision_num_attention_heads"] = vision_heads
                vision_layers = getattr(vision_cfg, "num_hidden_layers", getattr(vision_cfg, "depth", None))
                if vision_layers is not None:
                    model_info["vision_num_layers"] = vision_layers
                if vision_hidden is not None and vision_heads:
                    model_info["vision_head_dim"] = getattr(vision_cfg, "head_dim", vision_hidden // vision_heads)
                image_size = self._normalize_dim_value(
                    getattr(vision_cfg, "image_size", getattr(vision_cfg, "img_size", None))
                )
                if image_size is not None:
                    model_info["image_size"] = image_size
                patch_size = self._normalize_dim_value(
                    getattr(vision_cfg, "patch_size", getattr(vision_cfg, "spatial_patch_size", None))
                )
                if patch_size is not None:
                    model_info["patch_size"] = patch_size
                patch_num = getattr(vision_cfg, "num_patches", None)
                if patch_num is None:
                    patch_num = self._derive_patch_num(image_size, patch_size)
                if patch_num is not None:
                    model_info["patch_num"] = int(patch_num)
        else:
            model_info = {
                "model_name": type(model).__name__,
                "architecture": type(model).__name__,
            }
            model_info.update(self._infer_custom_model_info(model, sample_input))

            branch_info = self._infer_vlm_branch_info(model, active_inputs)
            text_info = branch_info.get("text")
            if text_info is not None:
                for key in (
                    "batch_size",
                    "seq_len",
                    "hidden_size",
                    "intermediate_size",
                    "num_attention_heads",
                    "num_key_value_heads",
                    "num_layers",
                    "vocab_size",
                    "eps",
                    "head_dim",
                ):
                    if key in text_info:
                        model_info[key] = text_info[key]

            vision_info = branch_info.get("vision")
            if vision_info is not None:
                for src_key, dst_key in (
                    ("hidden_size", "vision_hidden_size"),
                    ("intermediate_size", "vision_intermediate_size"),
                    ("num_attention_heads", "vision_num_attention_heads"),
                    ("num_layers", "vision_num_layers"),
                    ("head_dim", "vision_head_dim"),
                    ("image_size", "image_size"),
                    ("patch_size", "patch_size"),
                    ("patch_num", "patch_num"),
                ):
                    if src_key in vision_info:
                        model_info[dst_key] = vision_info[src_key]

        input_shapes = self._tensor_input_shapes(active_inputs)
        if input_shapes:
            model_info["input_shapes"] = input_shapes

        return model_info
        
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
            self.plena_backend = BACKEND  # keep reference so we can call build_registry_from_model later
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

    def trace(self):
        m = self.model.model if hasattr(self.model, "model") else self.model
        return self.trace_leaf_modules(
            m, {**self.inputs}
        )
    
    def trace_leaf_modules(self, model: nn.Module, forward_kwargs: dict, verbose: bool = False) -> dict:
        """
        Run a forward pass and return a call tree capturing the full module hierarchy,
        execution order, I/O shapes, direct weight parameters, and intermediate tensor symbols.

        Each tree node is a dict:
            {
                "name":           str         dotted module path ("" for the root)
                "type":           str         module class name
                "order":          int         call-entry index (0 = first module called)
                "attrs":          dict        static weight / config attrs (from _static_attrs)
                "in":             list[dict]  tensor input shapes at call time
                "out":            ...         tensor output shape(s) at return time
                "children":       list[dict]  sub-calls made inside this module's forward
                "weights":        list[dict]  direct parameters: [{"name": str, "shape": list}]
                "in_syms":        list[str]   runtime symbol for each input tensor (parallel to "in")
                "out_syms":       list[str]   runtime symbol for each output tensor
                "in_sym_sources": dict        {sym: producer_module_name} ("" = model input)
                "meta":           dict        extra per-node metadata, e.g. {"schema": {"eps": float}}
            }

        Use flatten_call_tree(root) to get a flat, order-sorted list.
        """
        counter = [0]
        stack: list[dict] = []
        root: list[dict] = [{}]
        pre_handles = []
        post_handles = []

        # Symbol tracking: tensor storage ptr → symbol name, symbol → producer module
        sym_ctr: list[int] = [0]
        sym_tbl: dict[int, str] = {}
        sym_prod: dict[str, str] = {}

        def _sym(t: torch.Tensor) -> str:
            key = t.data_ptr() or id(t)
            if key not in sym_tbl:
                sym_tbl[key] = f"%{sym_ctr[0]}"
                sym_ctr[0] += 1
            return sym_tbl[key]

        def shape_of(x) -> dict | None:
            if isinstance(x, torch.Tensor):
                return {"shape": list(x.shape), "dtype": str(x.dtype), "device": str(x.device)}
            return None

        def pre_hook(name: str, mod_type: str, attrs: dict):
            def _pre_hook(m, args, kwargs):
                in_ts = [a for a in list(args) + list(kwargs.values()) if isinstance(a, torch.Tensor)]
                in_syms = [_sym(t) for t in in_ts]
                schema: dict[str, Any] = {}
                norm_eps = _extract_norm_eps(name, m, attrs)
                if norm_eps is not None:
                    schema["eps"] = norm_eps
                node = {
                    "name": name,
                    "type": mod_type,
                    "order": counter[0],
                    "attrs": attrs,
                    "meta": {"schema": schema},
                    "in": [shape_of(t) for t in in_ts],
                    "out": None,
                    "children": [],
                    "weights": [
                        {"name": pn, "shape": list(p.shape)}
                        for pn, p in m.named_parameters(recurse=False)
                        if p is not None
                    ],
                    "in_syms": in_syms,
                    "in_sym_sources": {s: sym_prod.get(s, "") for s in in_syms},
                    "out_syms": [],
                }
                counter[0] += 1
                if stack:
                    stack[-1]["children"].append(node)
                else:
                    root[0] = node  # first module entered = tree root
                stack.append(node)
            return _pre_hook

        def post_hook(name: str):
            def _post_hook(mod, args, kwargs, out):
                if not stack:
                    return
                node = stack[-1]
                if isinstance(out, torch.Tensor):
                    out_ts = [out]
                elif isinstance(out, (tuple, list)):
                    out_ts = [o for o in out if isinstance(o, torch.Tensor)]
                else:
                    out_ts = []
                out_syms = []
                for t in out_ts:
                    s = _sym(t)
                    out_syms.append(s)
                    sym_prod[s] = name
                node["out"] = [shape_of(o) for o in out] if isinstance(out, (tuple, list)) else shape_of(out)
                node["out_syms"] = out_syms
                stack.pop()
            return _post_hook

        for name, m in model.named_modules():
            attrs = _static_attrs(m)
            if verbose:
                print(f"hooking: {name or type(m).__name__} ({type(m).__name__})")
            pre_handles.append(m.register_forward_pre_hook(pre_hook(name, type(m).__name__, attrs), with_kwargs=True))
            post_handles.append(m.register_forward_hook(post_hook(name), with_kwargs=True))

        try:
            with torch.no_grad():
                _ = model(**forward_kwargs)
        finally:
            for h in pre_handles + post_handles:
                h.remove()

        return root[0]

    def trace_fx_module_level(
        self,
        model: nn.Module,
        leaf_type_names: set[str] | None = None,
    ) -> list[dict]:
        """
        Trace *model* at the nn.Module level using torch.fx symbolic tracing.
        Returns a flat list of operation nodes in graph execution order.

        Unlike trace_leaf_modules (which requires a real forward pass and real
        input tensors), this method does NOT need runtime inputs — it works
        purely symbolically.  This makes it suitable for static sub-networks
        such as the Qwen3-VL text decoder where the shapes are fixed.

        What it captures
        ----------------
        call_module nodes  → every nn.Module whose type IS in leaf_type_names
        call_function nodes → elementwise ops from _FX_OPS (operator.add, F.silu …)

        What it CANNOT capture
        ----------------------
        Models with data-dependent control flow (use trace_leaf_modules instead).
        Actual tensor shapes (all "in"/"out" fields are None).

        Node schema (same keys as flatten_call_tree output)
        ---------------------------------------------------
        {name, type, order, attrs, in=None, out=None, source="fx", children=[]}

        Parameters
        ----------
        model           : the nn.Module to trace
        leaf_type_names : class names that should NOT be traced into.
                          Defaults to common PyTorch and Qwen3-VL types.
        """
        if leaf_type_names is None:
            leaf_type_names = {
                # PyTorch built-ins
                "Linear", "LayerNorm", "Embedding", "Conv3d",
                "SiLU", "GELU", "ReLU",
                # Qwen3-VL text decoder
                "Qwen3VLTextRMSNorm",
                "Qwen3VLTextAttention",
                "Qwen3VLTextMLP",
                # Qwen3-VL vision encoder
                "Qwen3VLVisionAttention",
                "Qwen3VLVisionMLP",
                "Qwen3VLVisionRotaryEmbedding",
                "Qwen3VLTextRotaryEmbedding",
            }

        class _LeafTracer(fx.Tracer):
            def is_leaf_module(self, m: nn.Module, _: str) -> bool:
                return type(m).__name__ in leaf_type_names  # type: ignore[name-defined]

        tracer = _LeafTracer()
        try:
            graph = tracer.trace(model)
        except Exception as e:
            raise RuntimeError(
                f"torch.fx tracing failed for {type(model).__name__}: {e}\n"
                "If the model has data-dependent control flow, use "
                "trace_leaf_modules (hook-based) instead."
            ) from e

        gm = fx.GraphModule(model, graph)

        nodes: list[dict] = []
        for node in gm.graph.nodes:
            if node.op == "call_module":
                submod = gm.get_submodule(node.target)
                nodes.append({
                    "name": node.target,
                    "type": type(submod).__name__,
                    "order": len(nodes),
                    "attrs": _static_attrs(submod),
                    "in": None,
                    "out": None,
                    "source": "fx",
                    "children": [],
                })
            elif node.op == "call_function" and node.target in _FX_OPS:
                nodes.append({
                    "name": node.name,
                    "type": _FX_OPS[node.target],
                    "order": len(nodes),
                    "attrs": {},
                    "in": None,
                    "out": None,
                    "source": "fx",
                    "children": [],
                })
        print("===== FX trace complete: captured {} nodes =====".format(len(nodes)))
        return nodes


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


def flatten_call_tree(root: dict) -> list[dict]:
    """
    Walk a call tree returned by trace_leaf_modules and return a flat list
    of all nodes sorted by execution order (pre-order DFS matches call order).
    """
    result = []
    def _walk(node: dict):
        result.append(node)
        for child in node["children"]:
            _walk(child)
    _walk(root)
    print("===== Call tree flattened: {} nodes =====".format(len(result)))
    return sorted(result, key=lambda n: n["order"])


def export_flatten_call_tree(flattened: list[dict], output_path: str | Path) -> Path:
    """
    Write a flattened call tree to a standalone JSON file.
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    output_file.write_text(json.dumps(flattened, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"===== Flattened call tree exported to {output_file} =====")
    return output_file


def export_model_info(model_info: dict[str, Any], output_path: str | Path) -> Path:
    """
    Write extract_model_info output to a standalone JSON file.
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    output_file.write_text(json.dumps(model_info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"===== Model info exported to {output_file} =====")
    return output_file


def combine_traces(hook_trace: dict, fx_trace: list[dict]) -> list[dict]:
    """
    Merge a hook-based call tree (e.g. dynamic visual encoder) with an
    FX-based flat trace (e.g. static text decoder) into a single flat list
    sorted by global execution order.

    Hook nodes come first (source="hook"), FX nodes follow (source="fx").
    Orders in fx_trace are renumbered to start immediately after the last
    hook node so the final list has consecutive 0-based orders.

    Parameters
    ----------
    hook_trace : tree dict returned by VLMModelParser.trace_leaf_modules
    fx_trace   : flat list returned by VLMModelParser.trace_fx_module_level
    """
    hook_flat = [{**n, "source": "hook"} for n in flatten_call_tree(hook_trace)]
    offset = len(hook_flat)
    fx_renumbered = [{**n, "order": offset + i} for i, n in enumerate(fx_trace)]
    return hook_flat + fx_renumbered


if __name__ == "__main__":
    # Modes:
    #   "trace"    – hook trace of full model.model (real shapes, no residuals)
    #   "fx"       – FX trace of language_model only (residuals visible, no real shapes)
    #   "combined" – hook trace visual encoder + FX trace language model, merged
    #   "compile"  – torch.compile with plena backend
    #   "export"   – torch.export symbolic graph
    TEST_MODE = "export_info"

    # ===== config parser =====
    parser = VLMModelParser("Qwen/Qwen3-VL-2B-Instruct")
    parser.load_backend(mode="default")  # ["default", "print", "dump", "viz"]
    parser.load_model()
    inputs = template_qwen3_vl_inputs(parser.processor, "./inputs/img/image.png")
    parser.load_inputs(inputs)

    def _log_print(f, s: str) -> None:
        f.write(s + "\n")

    def _qwen3_text_decoder_fx(p: "VLMModelParser") -> list[dict]:
        """
        Build a flat module-level trace of the Qwen3-VL text decoder.

        Neither Qwen3VLTextModel nor Qwen3VLTextDecoderLayer can be
        FX-traced:
          - Qwen3VLTextModel._deepstack_process() uses Proxy.__setitem__
          - Qwen3VLTextDecoderLayer.forward() does
              hidden_states, _ = self.self_attn(...)
            which calls Proxy.__iter__ on the leaf-module return value.

        Since the per-layer structure is always the same static pattern
        (input_layernorm → self_attn → residual_add →
         post_attention_layernorm → mlp → residual_add),
        we hardcode it directly from each layer's submodules.
        """
        lang = p.model.model.language_model
        n_layers = len(lang.layers)

        nodes: list[dict] = [{
            "name": "embed_tokens", "type": "Embedding", "order": 0,
            "attrs": _static_attrs(lang.embed_tokens),
            "in": None, "out": None, "source": "fx", "children": [],
        }]
        for li in range(n_layers):
            layer = lang.layers[li]
            # Fixed 6-node pattern matching Qwen3VLTextDecoderLayer.forward()
            layer_pattern = [
                ("input_layernorm",          type(layer.input_layernorm).__name__,          layer.input_layernorm),
                ("self_attn",                type(layer.self_attn).__name__,                layer.self_attn),
                ("add_attn",                 "elementwise_add",                             None),
                ("post_attention_layernorm", type(layer.post_attention_layernorm).__name__, layer.post_attention_layernorm),
                ("mlp",                      type(layer.mlp).__name__,                      layer.mlp),
                ("add_mlp",                  "elementwise_add",                             None),
            ]
            for local_name, type_name, mod in layer_pattern:
                nodes.append({
                    "name": f"layers.{li}.{local_name}",
                    "type": type_name,
                    "order": len(nodes),
                    "attrs": _static_attrs(mod) if mod is not None else {},
                    "in": None, "out": None, "source": "fx", "children": [],
                })
        nodes.append({
            "name": "norm", "type": type(lang.norm).__name__, "order": len(nodes),
            "attrs": _static_attrs(lang.norm),
            "in": None, "out": None, "source": "fx", "children": [],
        })
        return nodes

    # ===== run tests =====
    def _fmt_shape(sh: dict | None) -> str:
        if sh is None:
            return "?"
        return f"{sh['shape']} {sh['dtype']} {sh['device']}"

    if TEST_MODE == "trace":
        # Hook trace: real I/O shapes, weights, and tensor symbols at every module call.
        tree = parser.trace_leaf_modules(
            parser.model.model, {**inputs, "use_cache": False, "return_dict": False}
        )
        logs = flatten_call_tree(tree)
        with open("./outputs/trace.txt", "w") as f:
            for r in logs:
                f.write(f"Order: {r['order']:04d}\n")
                f.write(f"Module Type: {r['type']}\n")
                f.write(f"Module Name: {r['name']}\n")

                # In shapes
                f.write("In Shape:\n")
                for sh in (r.get("in") or []):
                    f.write(f"  {_fmt_shape(sh)}\n")

                # Out shapes
                f.write("Out Shape:\n")
                out = r.get("out")
                if isinstance(out, list):
                    for sh in out:
                        if sh is not None:
                            f.write(f"  {_fmt_shape(sh)}\n")
                elif out is not None:
                    f.write(f"  {_fmt_shape(out)}\n")

                # Weights
                weights = r.get("weights") or []
                if weights:
                    f.write("Weights:\n")
                    for w in weights:
                        f.write(f"  {w['name']}  {w['shape']}\n")

                # Symbols — show each input/output tensor symbol and its producer
                in_syms   = r.get("in_syms") or []
                out_syms  = r.get("out_syms") or []
                sources   = r.get("in_sym_sources") or {}
                in_shapes = r.get("in") or []
                if in_syms or out_syms:
                    f.write("Symbols:\n")
                    for sym, sh in zip(in_syms, in_shapes):
                        src = sources.get(sym, "")
                        src_str = src if src else "(model input)"
                        f.write(f"  {sym}  {_fmt_shape(sh)}  <- {src_str}\n")
                    # out_syms are parallel to only-tensor outputs; skip Nones in out
                    sym_iter = iter(out_syms)
                    out_list = out if isinstance(out, list) else ([out] if out else [])
                    for sh in out_list:
                        if sh is not None:
                            sym = next(sym_iter, "?")
                            f.write(f"  {sym}  {_fmt_shape(sh)}  -> {r['name']}\n")

                f.write("\n")

        vision = [r for r in logs if r["name"].startswith("visual.")]
        lang   = [r for r in logs if r["name"].startswith("language_model.")]
        print(f"Hook trace: {len(logs)} total  (vision={len(vision)}, lang={len(lang)})")
        print("Saved to ./outputs/trace.txt")

    elif TEST_MODE == "fx":
        # FX trace of the text decoder only — no real inputs needed.
        # Residual elementwise_add ops ARE visible; actual tensor shapes are None.
        # Uses per-layer replication to avoid Proxy.__setitem__ in Qwen3VLTextModel.
        lang_fx = _qwen3_text_decoder_fx(parser)
        with open("./outputs/fx_trace.txt", "w") as f:
            for i, r in enumerate(lang_fx):
                _log_print(f, f"{i:04d} {r['name']} {r['type']}")
        adds = sum(1 for r in lang_fx if r["type"] == "elementwise_add")
        print(f"FX trace: {len(lang_fx)} nodes  ({adds} elementwise_add ops)")
        print("Saved to ./outputs/fx_trace.txt")

    elif TEST_MODE == "combined":
        # Step 1 – hook trace the full model to get real I/O shapes (especially
        #           for the visual encoder where shapes are data-dependent).
        full_tree = parser.trace_leaf_modules(
            parser.model.model, {**inputs, "use_cache": False, "return_dict": False}
        )
        # Step 2 – extract only the visual encoder subtree.
        visual_tree = next(
            (c for c in full_tree["children"] if c["name"] == "visual"), None
        )
        if visual_tree is None:
            raise RuntimeError(
                "Could not find 'visual' child in hook trace. "
                "Check that parser.model.model has a .visual attribute."
            )
        # Step 3 – FX trace the text decoder (captures residual adds).
        # Uses per-layer replication to avoid Proxy.__setitem__ in Qwen3VLTextModel.
        lang_fx = _qwen3_text_decoder_fx(parser)
        # Step 4 – merge: visual (real shapes) first, then language (residuals visible).
        combined = combine_traces(visual_tree, lang_fx)
        with open("./outputs/combined_trace.txt", "w") as f:
            for i, r in enumerate(combined):
                _log_print(
                    f,
                    f"{i:04d} [{r['source']:4s}] {r['name']} {r['type']} "
                    f"in={r['in']} out={r['out']}"
                )
        hook_nodes = [r for r in combined if r["source"] == "hook"]
        fx_nodes_combined = [r for r in combined if r["source"] == "fx"]
        adds = sum(1 for r in fx_nodes_combined if r["type"] == "elementwise_add")
        print(f"Combined trace: {len(combined)} total")
        print(f"  hook (visual encoder):  {len(hook_nodes)} nodes  (real I/O shapes)")
        print(f"  fx   (text decoder):    {len(fx_nodes_combined)} nodes  ({adds} elementwise_add ops)")
        print("Saved to ./outputs/combined_trace.txt")

    elif TEST_MODE == "compile":
        parser.torch_compile_with_plena_backend(parser.model, inputs)

    elif TEST_MODE == "export":
        parser.create_symbolic_graph()
        
    elif TEST_MODE == "export_info":
        model_info = parser.extract_model_info(parser.model, inputs)
        export_model_info(model_info, "./outputs/model_info.json")
