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
                node = {
                    "name": name,
                    "type": mod_type,
                    "order": counter[0],
                    "attrs": attrs,
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
    return sorted(result, key=lambda n: n["order"])


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
    TEST_MODE = "trace"

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