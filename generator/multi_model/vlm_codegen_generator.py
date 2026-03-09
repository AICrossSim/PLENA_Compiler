"""
Assembly generation engine for traced VLM nodes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import prod
from typing import Any, Callable

from asm_lib.plena_program import VRAMMatrixVar
from asm_lib.qwen3_vl_vision_mlp import (
    GELUTanhFPRAMConstants,
    Qwen3VLVisionMLPCompiler,
    Qwen3VLVisionMLPSpec,
)
from vlm_codegen_context import matrix_shape_from_meta, safe_codegen_name
from vlm_codegen_env import VLMCodegenEnvironment


@dataclass
class LoweringResult:
    mode: str
    asm: str = ""
    setup_asm: str = ""
    reuse_label: str | None = None
    comments: list[str] = field(default_factory=list)


@dataclass
class RuntimeParameterBinding:
    canonical_name: str
    logical_shape: tuple[int, int]
    source_shape: tuple[int, ...] | None
    layout: str
    hbm_addr: int
    hbm_size: int


@dataclass
class RuntimeActivationBinding:
    symbol: str
    address: int
    producer: str
    source_kind: str


class VLMAssemblyGenerator:
    """Turns traced nodes into assembly using a VLMCodegenEnvironment."""

    def __init__(self, env: VLMCodegenEnvironment) -> None:
        self.env = env
        self.debug = False
        self.operation_handlers: dict[str, Callable[[dict[str, Any], dict[str, Any]], LoweringResult]] = {
            "embedding": self._codegen_embedding,
            "rms_norm": self._codegen_rms_norm,
            "layer_norm": self._codegen_layer_norm,
            "text_attention": self._codegen_text_attention,
            "vision_attention": self._codegen_vision_attention,
            "ffn": self._codegen_ffn,
            "mlp": self._codegen_mlp,
            "linear": self._codegen_linear,
            "conv3d": self._codegen_conv3d,
            "elementwise_add": self._codegen_elementwise_add,
            
            # qwen handlers
            "vision_mlp_plena": self._codegen_qwen_vision_mlp,
        }
    
    def debug_mode(self, enable: bool = True) -> None:
        self.debug = enable
        
    def _in_shape(self, node: dict[str, Any], idx: int = 0) -> list[int] | None:
        in_list = node.get("in") or []
        if idx < len(in_list) and in_list[idx]:
            return in_list[idx]["shape"]
        return None

    def _out_shape(self, node: dict[str, Any]) -> list[int] | None:
        out = node.get("out")
        if isinstance(out, dict):
            return out.get("shape")
        if isinstance(out, list):
            for sh in out:
                if sh is not None:
                    return sh["shape"]
        return None

    def _batch_from_node(self, node: dict[str, Any], model_info: dict[str, Any]) -> int:
        sh = self._in_shape(node)
        if sh and len(sh) >= 1:
            return sh[0]
        return model_info.get("batch_size", 1)

    def _seq_len_from_node(self, node: dict[str, Any], model_info: dict[str, Any]) -> int:
        sh = self._in_shape(node)
        if sh and len(sh) >= 2:
            return sh[1]
        return model_info.get("seq_len", 1)

    def _hidden_from_node(self, node: dict[str, Any], model_info: dict[str, Any]) -> int:
        attrs = node.get("attrs") or {}
        if "hidden_size" in attrs:
            return attrs["hidden_size"]
        sh = self._in_shape(node)
        if sh:
            return sh[-1]
        return model_info.get("hidden_size", 2048)

    def _shape_tuple(self, meta: dict[str, Any] | None) -> tuple[int, ...] | None:
        if meta is None:
            return None
        shape = meta.get("shape")
        if not isinstance(shape, list):
            return None
        return tuple(int(v) for v in shape)

    def _matrix_shape(self, meta: dict[str, Any] | None) -> tuple[int, int] | None:
        return matrix_shape_from_meta(meta)

    def _reset_runtime_state(self) -> None:
        self._runtime_parameters: dict[str, RuntimeParameterBinding] = {}
        self._runtime_symbol_hbm_inputs: dict[str, int] = {}
        self._runtime_hbm_next: int = 0
        self._template_symbol_bindings: dict[str, RuntimeActivationBinding] = {}
        self._covered_by: dict[str, str] = {}

    def _allocate_hbm_range(self, shape: tuple[int, ...]) -> tuple[int, int]:
        size = int(prod(shape) * float(self.env.hw_value("real_data_ratio", 1.125)))
        base = self._runtime_hbm_next
        self._runtime_hbm_next += size
        return base, size

    def _ensure_runtime_symbol_input(self, symbol: str, matrix_shape: tuple[int, int]) -> int:
        existing = self._runtime_symbol_hbm_inputs.get(symbol)
        if existing is not None:
            return existing
        hbm_addr, _ = self._allocate_hbm_range(matrix_shape)
        self._runtime_symbol_hbm_inputs[symbol] = hbm_addr
        return hbm_addr

    def _ensure_runtime_parameter(
        self,
        *,
        canonical_name: str,
        logical_shape: tuple[int, int],
        source_shape: tuple[int, ...] | None = None,
        layout: str = "logical",
    ) -> RuntimeParameterBinding:
        existing = self._runtime_parameters.get(canonical_name)
        if existing is not None:
            if existing.logical_shape != logical_shape:
                raise ValueError(
                    f"Runtime parameter shape mismatch for {canonical_name}: "
                    f"{existing.logical_shape} vs {logical_shape}"
                )
            return existing

        hbm_addr, hbm_size = self._allocate_hbm_range(logical_shape)
        binding = RuntimeParameterBinding(
            canonical_name=canonical_name,
            logical_shape=logical_shape,
            source_shape=source_shape,
            layout=layout,
            hbm_addr=hbm_addr,
            hbm_size=hbm_size,
        )
        self._runtime_parameters[canonical_name] = binding
        return binding

    def _shared_parameter_handle(
        self,
        *,
        canonical_name: str,
        logical_shape: tuple[int, int],
        source_shape: tuple[int, ...] | None = None,
        layout: str = "logical",
    ):
        binding = self._ensure_runtime_parameter(
            canonical_name=canonical_name,
            logical_shape=logical_shape,
            source_shape=source_shape,
            layout=layout,
        )
        handle = self.env.shared_context.register_parameter(
            canonical_name=canonical_name,
            logical_shape=logical_shape,
            source_shape=source_shape,
            layout=layout,
            hbm_addr=binding.hbm_addr,
        )
        return handle, binding

    def _candidate_activation_blocks(self) -> list[int]:
        candidates = [
            self.env.mem("block1", 0),
            self.env.mem("block2", 0),
            self.env.mem("block5", self.env.mem("block2", 0)),
        ]
        ordered: list[int] = []
        seen: set[int] = set()
        for addr in candidates:
            if addr in seen:
                continue
            ordered.append(addr)
            seen.add(addr)
        return ordered

    def _choose_activation_address(self, preferred: int, avoid: set[int] | None = None) -> int:
        blocked = avoid or set()
        if preferred not in blocked:
            return preferred
        for addr in self._candidate_activation_blocks():
            if addr not in blocked:
                return addr
        return preferred

    def _bind_template_outputs(self, node: dict[str, Any], address: int) -> None:
        for out_sym in node.get("out_syms") or []:
            self._template_symbol_bindings[out_sym] = RuntimeActivationBinding(
                symbol=out_sym,
                address=address,
                producer=node.get("name", ""),
                source_kind="template",
            )

    def _shared_symbol_binding(self, symbol: str) -> RuntimeActivationBinding | None:
        record = self.env.shared_context.resolve_symbol(symbol)
        if record is None or not isinstance(record.tensor, VRAMMatrixVar):
            return None
        return RuntimeActivationBinding(
            symbol=symbol,
            address=self.env.shared_context.program.compiler.get_vram_addr(record.tensor.name),
            producer=record.producer,
            source_kind="shared",
        )

    def _resolve_symbol_binding(self, symbol: str | None) -> RuntimeActivationBinding | None:
        if not symbol:
            return None
        binding = self._template_symbol_bindings.get(symbol)
        if binding is not None:
            return binding
        return self._shared_symbol_binding(symbol)

    def _resolve_input_activation(
        self,
        node: dict[str, Any],
        *,
        idx: int = 0,
        default_block: str = "block1",
    ) -> tuple[int, RuntimeActivationBinding | None]:
        in_syms = node.get("in_syms") or []
        symbol = in_syms[idx] if idx < len(in_syms) else None
        binding = self._resolve_symbol_binding(symbol)
        if binding is not None:
            return binding.address, binding
        return self.env.mem(default_block, 0), None

    def _emit_addr_reg_preload(
        self,
        bindings: list[tuple[int, RuntimeParameterBinding]],
        *,
        alive_registers: list[int],
    ) -> str:
        if not bindings:
            return ""
        staging_regs = [reg for reg in alive_registers if reg != 0][: len(bindings)]
        if len(staging_regs) < len(bindings):
            return ""
        return self.env.template("preload_addr_reg")(
            addr_reg_to_set=[addr_reg for addr_reg, _ in bindings],
            available_registers=staging_regs,
            addr_reg_val=[binding.hbm_addr for _, binding in bindings],
        )

    def _program_header(self, model_info: dict[str, Any]) -> str:
        return (
            "; ============================================================\n"
            f"; PLENA VLM Assembly  -  {model_info.get('model_name', 'unknown')}\n"
            f"; hidden_size={model_info.get('hidden_size', '?')}  "
            f"layers={model_info.get('num_layers', '?')}  "
            f"heads={model_info.get('num_attention_heads', '?')}\n"
            "; ============================================================\n"
        )

    def _program_footer(self) -> str:
        return "; ============================================================\n; END\n"

    def _node_header(self, node: dict[str, Any]) -> str:
        name = node["name"] or "(root)"
        ntype = node["type"]
        in_syms = node.get("in_syms") or []
        out_syms = node.get("out_syms") or []
        sh_in = self._in_shape(node)
        sh_out = self._out_shape(node)

        lines = [f"; --- [{ntype}]  {name}"]
        if sh_in:
            lines.append(f";     in  {sh_in}  sym={in_syms}")
        if sh_out:
            lines.append(f";     out {sh_out}  sym={out_syms}")

        weights = node.get("weights") or []
        if weights:
            w_str = "  ".join(f"{w['name']}:{w['shape']}" for w in weights)
            lines.append(f";     weights  {w_str}")

        return "\n".join(lines)

    def _render_node_section(self, node: dict[str, Any], result: LoweringResult) -> str:
        lines = [self._node_header(node), f";     lowering={result.mode}"]
        for comment in result.comments:
            lines.append(f";     {comment}")
        if result.asm and not self.debug:
            lines.append(result.asm.rstrip())
        elif result.asm and self.debug:
            lines.append(";     asm_omitted_in_debug_mode")
        return "\n".join(lines) + "\n"

    def _effective_nodes(self, nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Nodes visible to graph-level liveness and symbol bookkeeping."""
        effective: list[dict[str, Any]] = []
        for node in nodes:
            node_type = node.get("type", "")
            op_key = self.env.operation_for(node_type)
            handler = self.operation_handlers.get(op_key) if op_key is not None else None
            if handler is None:
                continue
            effective.append(node)
        return effective

    def _wrap_template(
        self,
        asm: str,
        *,
        mode: str = "template",
        setup_asm: str = "",
        reuse_label: str | None = None,
        comments: list[str] | None = None,
    ) -> LoweringResult:
        return LoweringResult(
            mode=mode,
            asm=asm,
            setup_asm=setup_asm,
            reuse_label=reuse_label,
            comments=comments or [],
        )

    def _child_linear(self, node: dict[str, Any], suffixes: tuple[str, ...], default_index: int | None = None) -> dict[str, Any] | None:
        children = [child for child in (node.get("children") or []) if child.get("type") == "Linear"]
        if not children:
            return None
        for suffix in suffixes:
            for child in children:
                if child.get("name", "").endswith(suffix):
                    return child
        if default_index is not None and default_index < len(children):
            return children[default_index]
        return None

    def _linear_weight_logical_shape(self, node: dict[str, Any]) -> tuple[int, int] | None:
        attrs = node.get("attrs") or {}
        in_features = attrs.get("in_features")
        out_features = attrs.get("out_features")
        if in_features is not None and out_features is not None:
            return (int(in_features), int(out_features))
        source_shape = self._weight_shape(node, "weight")
        if source_shape is not None and len(source_shape) == 2:
            return (int(source_shape[1]), int(source_shape[0]))
        return None

    def _linear_weight_binding(self, node: dict[str, Any]) -> RuntimeParameterBinding | None:
        logical_shape = self._linear_weight_logical_shape(node)
        if logical_shape is None:
            return None
        return self._ensure_runtime_parameter(
            canonical_name=self._canonical_param_name(node.get("name", "linear"), "weight"),
            logical_shape=logical_shape,
            source_shape=self._weight_shape(node, "weight"),
            layout="linear_weight_in_out",
        )

    def _linear_weight_reg_binding(self, node: dict[str, Any]) -> tuple[str, int]:
        name = str(node.get("name", "")).lower()
        input_sources = " ".join(str(src).lower() for src in (node.get("in_sym_sources") or {}).values() if src)
        context = f"{name} {input_sources}"

        for token, reg_name, default_reg in (
            ("q_proj", "q_weight_offset", 2),
            (".q", "q_weight_offset", 2),
            ("k_proj", "k_weight_offset", 3),
            (".k", "k_weight_offset", 3),
            ("v_proj", "v_weight_offset", 4),
            (".v", "v_weight_offset", 4),
        ):
            if token in context:
                return reg_name, self.env.reg(reg_name, default_reg)

        if any(
            token in context
            for token in ("mlp", "ffn", "feed_forward", "gate_proj", "up_proj", "down_proj", "fc1", "fc2")
        ):
            return "ffn_weight_offset", self.env.reg("ffn_weight_offset", 5)

        return "q_weight_offset", self.env.reg("q_weight_offset", 2)

    def _row_block_bias_binding(self, node: dict[str, Any], out_features: int) -> RuntimeParameterBinding | None:
        attrs = node.get("attrs") or {}
        if not bool(attrs.get("bias", False)):
            return None
        mlen = int(self.env.hw_value("mlen", 64))
        return self._ensure_runtime_parameter(
            canonical_name=self._canonical_param_name(node.get("name", "linear"), "bias"),
            logical_shape=(mlen, out_features),
            source_shape=self._weight_shape(node, "bias"),
            layout="row_block_bias",
        )

    def _covered_node_result(self, node: dict[str, Any], covering_name: str) -> LoweringResult:
        comments = [f"covered_by={covering_name}"]
        input_addr, input_binding = self._resolve_input_activation(node)
        comments.append(
            "input_activation="
            f"{input_addr}"
            + (f" ({input_binding.source_kind}:{input_binding.producer})" if input_binding is not None else " (default)")
        )
        out_syms = node.get("out_syms") or []
        if out_syms:
            out_binding = self._resolve_symbol_binding(out_syms[0])
            if out_binding is not None:
                comments.append(f"output_activation={out_binding.address} ({out_binding.source_kind}:{out_binding.producer})")
        for weight in node.get("weights") or []:
            binding = self._runtime_parameters.get(self._canonical_param_name(node.get("name", ""), weight.get("name", "")))
            if binding is not None:
                comments.append(f"{weight.get('name')} hbm={binding.hbm_addr} layout={binding.layout}")
        return LoweringResult(mode="covered", comments=comments)

    def _codegen_embedding(self, node: dict[str, Any], model_info: dict[str, Any]) -> LoweringResult:
        attrs = node.get("attrs") or {}
        n_emb = attrs.get("num_embeddings", model_info.get("vocab_size", 32000))
        emb_dim = attrs.get("embedding_dim", model_info.get("hidden_size", 2048))
        batch = self._batch_from_node(node, model_info)
        regs = self.env.hw_value("alive_registers", [1, 2, 3, 4])
        activation_addr = self.env.sched.get("activation_base_address", 0)

        comments: list[str] = []
        prelude = ""
        source_shape = self._weight_shape(node, "weight")
        if source_shape is not None:
            binding = self._ensure_runtime_parameter(
                canonical_name=self._canonical_param_name(node.get("name", "embedding"), "weight"),
                logical_shape=(int(n_emb), int(emb_dim)),
                source_shape=source_shape,
                layout="embedding_table",
            )
            prelude = self._emit_addr_reg_preload(
                [(self.env.reg("token_table_offset", 1), binding)],
                alive_registers=regs,
            )
            comments.append(f"token_table hbm={binding.hbm_addr}")

        asm = self.env.template("embedding")(
            mlen=self.env.hw_value("mlen", 64),
            blen=self.env.hw_value("blen", 4),
            batch=batch,
            hidden_size=emb_dim,
            alive_registers=regs,
            voc_table_row_size=n_emb,
            activation_base_address=activation_addr,
            voc_table_base_addr_reg_index=self.env.reg("token_table_offset", 1),
            input_ids=[1 for _ in range(batch)],
        )
        self._bind_template_outputs(node, activation_addr)
        return self._wrap_template(
            asm,
            setup_asm=prelude,
            reuse_label=node.get("type"),
            comments=comments,
        )

    def _codegen_rms_norm(self, node: dict[str, Any], model_info: dict[str, Any]) -> LoweringResult:
        hidden = self._hidden_from_node(node, model_info)
        batch = self._batch_from_node(node, model_info)
        activation_addr, input_binding = self._resolve_input_activation(node, default_block="block1")
        scratchpad_addr = self._choose_activation_address(self.env.mem("block2", 0), avoid={activation_addr})

        asm = self.env.template("rms_norm")(
            _eps_offset=self.env.fp_mem("eps", 0),
            reci_hid_offset=self.env.fp_mem("hid_reciprocal", 0),
            alive_registers=self.env.hw_value("alive_registers", [1, 2, 3, 4])[:3],
            activation_base_address=activation_addr,
            scratchpad_base_address=scratchpad_addr,
            vlen=self.env.hw_value("vlen", 16),
            batch_size=batch,
            hidden_dim=hidden,
        )
        self._bind_template_outputs(node, activation_addr)
        comments = [f"activation={activation_addr}"]
        if input_binding is not None:
            comments.append(f"input_source={input_binding.source_kind}:{input_binding.producer}")
        return self._wrap_template(asm, reuse_label=node.get("type"), comments=comments)

    def _codegen_layer_norm(self, node: dict[str, Any], model_info: dict[str, Any]) -> LoweringResult:
        attrs = node.get("attrs") or {}
        ns = attrs.get("normalized_shape", [model_info.get("hidden_size", 2048)])
        hidden = ns[0] if isinstance(ns, list) else ns
        batch = self._batch_from_node(node, model_info)
        activation_addr, input_binding = self._resolve_input_activation(node, default_block="block1")
        scratchpad_addr = self._choose_activation_address(self.env.mem("block2", 0), avoid={activation_addr})

        asm = self.env.template("layer_norm")(
            _eps_offset=self.env.fp_mem("eps", 0),
            reci_hid_offset=self.env.fp_mem("hid_reciprocal", 0),
            alive_registers=self.env.hw_value("alive_registers", [1, 2, 3, 4])[:3],
            activation_base_address=activation_addr,
            scratchpad_base_address=scratchpad_addr,
            vlen=self.env.hw_value("vlen", 16),
            batch_size=batch,
            hidden_dim=hidden,
        )
        self._bind_template_outputs(node, activation_addr)
        comments = [f"activation={activation_addr}"]
        if input_binding is not None:
            comments.append(f"input_source={input_binding.source_kind}:{input_binding.producer}")
        return self._wrap_template(asm, reuse_label=node.get("type"), comments=comments)

    def _codegen_text_attention(self, node: dict[str, Any], model_info: dict[str, Any]) -> LoweringResult:
        sh = self._in_shape(node)
        hidden_size = sh[-1] if sh else model_info.get("hidden_size", 2048)
        head_dim = model_info.get("head_dim", hidden_size // model_info.get("num_attention_heads", 16))
        batch = sh[0] if sh else model_info.get("batch_size", 1)
        regs = self.env.hw_value("alive_registers", [1, 2, 3, 4, 5, 6, 7, 8])
        activation_addr, input_binding = self._resolve_input_activation(node, default_block="block1")
        result_addr = self._choose_activation_address(self.env.mem("block2", 0), avoid={activation_addr})
        q_child = self._child_linear(node, ("q_proj", ".q"), default_index=0)
        k_child = self._child_linear(node, ("k_proj", ".k"), default_index=1)
        v_child = self._child_linear(node, ("v_proj", ".v"), default_index=2)
        q_binding = self._linear_weight_binding(q_child) if q_child is not None else None
        k_binding = self._linear_weight_binding(k_child) if k_child is not None else None
        v_binding = self._linear_weight_binding(v_child) if v_child is not None else None
        prelude = self._emit_addr_reg_preload(
            [
                pair
                for pair in (
                    (self.env.reg("q_weight_offset", 2), q_binding),
                    (self.env.reg("k_weight_offset", 3), k_binding),
                    (self.env.reg("v_weight_offset", 4), v_binding),
                )
                if pair[1] is not None
            ],
            alive_registers=regs,
        )

        proj_common = dict(
            mlen=self.env.hw_value("mlen", 64),
            blen=self.env.hw_value("blen", 4),
            batch=batch,
            hidden_size=hidden_size,
            alive_registers=regs,
            out_features=head_dim,
            rope_hbm_offset_reg=self.env.reg("rope_params_offset", 6),
            rope_on_chip_address=self.env.mem("block3", 0),
            activation_base_address=activation_addr,
            result_base_address=result_addr,
        )

        code = "; --- Q projection\n"
        code += self.env.template("projection")(
            **proj_common,
            w_base_hbm_offset_reg=self.env.reg("q_weight_offset", 2),
            rope_enabled=True,
        )
        code += "; --- K projection\n"
        code += self.env.template("projection")(
            **proj_common,
            w_base_hbm_offset_reg=self.env.reg("k_weight_offset", 3),
            rope_enabled=True,
        )
        code += "; --- V projection\n"
        code += self.env.template("projection")(
            **proj_common,
            w_base_hbm_offset_reg=self.env.reg("v_weight_offset", 4),
            rope_enabled=False,
        )
        code += "; --- Flash Attention (TODO: wire flash_attn_asm)\n"
        self._bind_template_outputs(node, result_addr)
        comments = [f"activation={activation_addr}", f"result={result_addr}"]
        if input_binding is not None:
            comments.append(f"input_source={input_binding.source_kind}:{input_binding.producer}")
        for label, binding in (("q", q_binding), ("k", k_binding), ("v", v_binding)):
            if binding is not None:
                comments.append(f"{label}_weight hbm={binding.hbm_addr}")
        return self._wrap_template(
            code,
            setup_asm=prelude,
            reuse_label=node.get("type"),
            comments=comments,
        )

    def _codegen_vision_attention(self, node: dict[str, Any], model_info: dict[str, Any]) -> LoweringResult:
        sh = self._in_shape(node)
        hidden_size = sh[-1] if sh else model_info.get("vision_hidden_size", model_info.get("hidden_size", 2048))
        head_dim = model_info.get("vision_head_dim", hidden_size // model_info.get("num_attention_heads", 16))
        regs = self.env.hw_value("alive_registers", [1, 2, 3, 4, 5, 6, 7, 8])
        activation_addr, input_binding = self._resolve_input_activation(node, default_block="block1")
        result_addr = self._choose_activation_address(self.env.mem("block2", 0), avoid={activation_addr})
        q_child = self._child_linear(node, ("q_proj", ".q"), default_index=0)
        k_child = self._child_linear(node, ("k_proj", ".k"), default_index=1)
        v_child = self._child_linear(node, ("v_proj", ".v"), default_index=2)
        q_binding = self._linear_weight_binding(q_child) if q_child is not None else None
        k_binding = self._linear_weight_binding(k_child) if k_child is not None else None
        v_binding = self._linear_weight_binding(v_child) if v_child is not None else None
        prelude = self._emit_addr_reg_preload(
            [
                pair
                for pair in (
                    (self.env.reg("q_weight_offset", 2), q_binding),
                    (self.env.reg("k_weight_offset", 3), k_binding),
                    (self.env.reg("v_weight_offset", 4), v_binding),
                )
                if pair[1] is not None
            ],
            alive_registers=regs,
        )

        proj_common = dict(
            mlen=self.env.hw_value("mlen", 64),
            blen=self.env.hw_value("blen", 4),
            batch=1,
            hidden_size=hidden_size,
            alive_registers=regs,
            out_features=head_dim,
            rope_hbm_offset_reg=self.env.reg("rope_params_offset", 6),
            rope_on_chip_address=self.env.mem("block3", 0),
            activation_base_address=activation_addr,
            result_base_address=result_addr,
            rope_enabled=False,
        )

        code = "; --- Vision Q projection\n"
        code += self.env.template("projection")(
            **proj_common,
            w_base_hbm_offset_reg=self.env.reg("q_weight_offset", 2),
        )
        code += "; --- Vision K projection\n"
        code += self.env.template("projection")(
            **proj_common,
            w_base_hbm_offset_reg=self.env.reg("k_weight_offset", 3),
        )
        code += "; --- Vision V projection\n"
        code += self.env.template("projection")(
            **proj_common,
            w_base_hbm_offset_reg=self.env.reg("v_weight_offset", 4),
        )
        code += "; --- Vision Flash Attention (TODO: wire flash_attn_asm)\n"
        self._bind_template_outputs(node, result_addr)
        comments = [f"activation={activation_addr}", f"result={result_addr}"]
        if input_binding is not None:
            comments.append(f"input_source={input_binding.source_kind}:{input_binding.producer}")
        for label, binding in (("q", q_binding), ("k", k_binding), ("v", v_binding)):
            if binding is not None:
                comments.append(f"{label}_weight hbm={binding.hbm_addr}")
        return self._wrap_template(
            code,
            setup_asm=prelude,
            reuse_label=node.get("type"),
            comments=comments,
        )

    def _codegen_mlp(self, node: dict[str, Any], model_info: dict[str, Any]) -> LoweringResult:
        children = node.get("children") or []
        if not children:
            return LoweringResult(mode="unsupported", comments=["composite MLP has no child ops"])

        parts: list[str] = []
        non_template = False
        for child in children:
            op_key = self.env.operation_for(child.get("type", ""))
            handler = self.operation_handlers.get(op_key) if op_key is not None else None
            if handler is None:
                parts.append(f"; [unsupported child in MLP] {child.get('name', '')} ({child.get('type', '')})")
                non_template = True
                continue
            try:
                result = handler(child, model_info)
            except Exception as exc:
                result = LoweringResult(mode="partial", comments=[f"child lowering failed: {exc}"])
            if result.mode != "template":
                non_template = True
            parts.append(self._render_node_section(child, result).rstrip())

        return LoweringResult(
            mode="partial" if non_template else "template",
            asm="\n".join(parts),
            comments=["composite MLP expanded through child handlers"],
        )

    def _codegen_ffn(self, node: dict[str, Any], model_info: dict[str, Any]) -> LoweringResult:
        sh = self._in_shape(node)
        hidden_size = sh[-1] if sh else model_info.get("hidden_size", 2048)
        batch = sh[0] if sh else model_info.get("batch_size", 1)
        seq_len = sh[1] if sh and len(sh) >= 2 else self._seq_len_from_node(node, model_info)

        attrs = node.get("attrs") or {}
        inter_size = attrs.get("intermediate_size") or model_info.get("intermediate_size") or hidden_size * 4
        activation_addr, input_binding = self._resolve_input_activation(node, default_block="block1")
        regs = self.env.hw_value("alive_registers", [1, 2, 3, 4, 5, 6, 7, 8])
        gate_child = self._child_linear(node, ("gate_proj",), default_index=0)
        up_child = self._child_linear(node, ("up_proj",), default_index=1)
        down_child = self._child_linear(node, ("down_proj",), default_index=2)

        gate_binding = self._linear_weight_binding(gate_child) if gate_child is not None else None
        up_binding = self._linear_weight_binding(up_child) if up_child is not None else None
        down_binding = self._linear_weight_binding(down_child) if down_child is not None else None
        preload_pairs = []
        gate_reg = self.env.reg("gate_weight_offset", 5)
        up_reg = self.env.reg("up_weight_offset", 6)
        down_reg = self.env.reg("down_weight_offset", 7)
        for reg, binding in ((gate_reg, gate_binding), (up_reg, up_binding), (down_reg, down_binding)):
            if binding is not None:
                preload_pairs.append((reg, binding))
        prelude = self._emit_addr_reg_preload(preload_pairs, alive_registers=regs)

        asm = self.env.template("ffn")(
            mlen=self.env.hw_value("mlen", 64),
            vlen=self.env.hw_value("vlen", 16),
            blen=self.env.hw_value("blen", 4),
            batch=batch,
            seq_len=seq_len,
            hidden_size=hidden_size,
            intermediate_size=inter_size,
            alive_registers=regs,
            gate_weight_hbm_offset_reg=gate_reg,
            up_weight_hbm_offset_reg=up_reg,
            down_weight_hbm_offset_reg=down_reg,
            const_one_fp_address=self.env.fp_mem("one", 0),
            activation_base_address=activation_addr,
            use_loop_instructions=True,
        )
        self._bind_template_outputs(node, activation_addr)
        comments = [f"activation={activation_addr}"]
        if input_binding is not None:
            comments.append(f"input_source={input_binding.source_kind}:{input_binding.producer}")
        for label, binding in (("gate", gate_binding), ("up", up_binding), ("down", down_binding)):
            if binding is not None:
                comments.append(f"{label}_weight hbm={binding.hbm_addr}")
        return self._wrap_template(
            asm,
            setup_asm=prelude,
            reuse_label=node.get("type"),
            comments=comments,
        )

    def _codegen_linear(self, node: dict[str, Any], model_info: dict[str, Any]) -> LoweringResult:
        attrs = node.get("attrs") or {}
        # Standalone Linear nodes can belong to attention Q/K/V projections or to MLP/FFN.
        # Pick the preload register accordingly so the generated template uses the right HBM base.
        reg_name, weight_reg = self._linear_weight_reg_binding(node)
        in_features = int(attrs.get("in_features", model_info.get("hidden_size", 2048)))
        out_features = int(attrs.get("out_features", model_info.get("hidden_size", 2048)))
        batch = self._batch_from_node(node, model_info)
        regs = self.env.hw_value("alive_registers", [1, 2, 3, 4, 5, 6, 7, 8])
        activation_addr, input_binding = self._resolve_input_activation(node, default_block="block1")
        result_addr = self._choose_activation_address(self.env.mem("block2", 0), avoid={activation_addr})
        weight_binding = self._linear_weight_binding(node)
        prelude = ""
        if weight_binding is not None:
            prelude = self._emit_addr_reg_preload(
                [(weight_reg, weight_binding)],
                alive_registers=regs,
            )

        asm = self.env.template("projection")(
            mlen=self.env.hw_value("mlen", 64),
            blen=self.env.hw_value("blen", 4),
            batch=batch,
            hidden_size=in_features,
            alive_registers=regs,
            out_features=out_features,
            w_base_hbm_offset_reg=weight_reg,
            rope_hbm_offset_reg=self.env.reg("rope_params_offset", 6),
            rope_on_chip_address=self.env.mem("block3", 0),
            activation_base_address=activation_addr,
            result_base_address=result_addr,
            rope_enabled=False,
        )
        self._bind_template_outputs(node, result_addr)
        comments = [f"activation={activation_addr}", f"result={result_addr}"]
        if input_binding is not None:
            comments.append(f"input_source={input_binding.source_kind}:{input_binding.producer}")
        if weight_binding is not None:
            comments.append(f"weight hbm={weight_binding.hbm_addr}")
        comments.append(f"weight_reg={reg_name}:a{weight_reg}")
        return self._wrap_template(
            asm,
            setup_asm=prelude,
            reuse_label=node.get("type"),
            comments=comments,
        )

    def _codegen_conv3d(self, node: dict[str, Any], model_info: dict[str, Any]) -> LoweringResult:
        attrs = node.get("attrs") or {}
        out_ch = attrs.get("out_channels", model_info.get("hidden_size", 2048))

        asm = self.env.template("batched_matmul")(
            mlen=self.env.hw_value("mlen", 64),
            blen=self.env.hw_value("blen", 4),
            hidden_size=out_ch,
            alive_registers=self.env.hw_value("alive_registers", [1, 2, 3, 4, 5, 6, 7, 8]),
            activation_base_address=self.env.mem("block1", 0),
            result_base_address=self.env.mem("block2", 0),
            weight_hbm_offset_reg=self.env.reg("q_weight_offset", 2),
        )
        return self._wrap_template(asm, reuse_label=node.get("type"))

    def _codegen_elementwise_add(self, node: dict[str, Any], model_info: dict[str, Any]) -> LoweringResult:
        sh = self._in_shape(node)
        hidden_size = (sh[-1] if sh else None) or model_info.get("hidden_size", 2048)
        batch = (sh[0] if sh else None) or model_info.get("batch_size", 1)
        stored_addr, stored_binding = self._resolve_input_activation(node, idx=0, default_block="block1")
        previous_addr, previous_binding = self._resolve_input_activation(node, idx=1, default_block="block2")

        asm = self.env.template("elementwise_add")(
            vlen=self.env.hw_value("vlen", 16),
            hidden_size=hidden_size,
            batch=batch,
            alive_registers=self.env.hw_value("alive_registers", [1, 2, 3, 4])[:3],
            stored_activation_base_address=stored_addr,
            previous_activation_base_address=previous_addr,
            previous_act_on_chip_addr_reg_index=self.env.reg("previous_activation_offset", 7),
        )
        self._bind_template_outputs(node, stored_addr)
        comments = [f"stored_activation={stored_addr}", f"previous_activation={previous_addr}"]
        if stored_binding is not None:
            comments.append(f"stored_source={stored_binding.source_kind}:{stored_binding.producer}")
        if previous_binding is not None:
            comments.append(f"previous_source={previous_binding.source_kind}:{previous_binding.producer}")
        return self._wrap_template(asm, reuse_label=node.get("type"), comments=comments)

    def _vision_linear_child(self, node: dict[str, Any], key: str) -> dict[str, Any] | None:
        children = [child for child in (node.get("children") or []) if child.get("type") == "Linear"]
        if not children:
            return None
        for child in children:
            if child.get("name", "").endswith(key):
                return child
        if key == "linear_fc1":
            return children[0]
        if key == "linear_fc2" and len(children) >= 2:
            return children[1]
        return None

    def _weight_shape(self, node: dict[str, Any], weight_name: str) -> tuple[int, ...] | None:
        for weight in node.get("weights") or []:
            if weight.get("name") == weight_name:
                shape = weight.get("shape")
                if isinstance(shape, list):
                    return tuple(int(v) for v in shape)
        return None

    def _canonical_param_name(self, node_name: str, leaf: str) -> str:
        return f"{node_name}.{leaf}"

    def _codegen_qwen_vision_mlp(self, node: dict[str, Any], model_info: dict[str, Any]) -> LoweringResult:
        fc1 = self._vision_linear_child(node, "linear_fc1")
        fc2 = self._vision_linear_child(node, "linear_fc2")
        if fc1 is None or fc2 is None:
            return LoweringResult(
                mode="unsupported",
                comments=["vision MLP lowering needs child Linear nodes for fc1/fc2"],
            )

        fc1_attrs = fc1.get("attrs") or {}
        fc2_attrs = fc2.get("attrs") or {}
        hidden_size = int(fc1_attrs.get("in_features", self._hidden_from_node(node, model_info)))
        intermediate_size = int(fc1_attrs.get("out_features", fc2_attrs.get("in_features", hidden_size * 4)))
        output_hidden = int(fc2_attrs.get("out_features", hidden_size))
        if output_hidden != hidden_size:
            return LoweringResult(
                mode="unsupported",
                comments=[f"fc2 out_features={output_hidden} does not match hidden_size={hidden_size}"],
            )

        input_meta = (node.get("in") or [None])[0]
        input_matrix_shape = self._matrix_shape(input_meta)
        if input_matrix_shape is None:
            return LoweringResult(mode="unsupported", comments=["vision MLP lowering needs concrete input shape"])
        rows, cols = input_matrix_shape
        if cols != hidden_size:
            return LoweringResult(
                mode="unsupported",
                comments=[f"input hidden width {cols} does not match fc1 in_features={hidden_size}"],
            )

        mlen = int(self.env.hw_value("mlen", 64))
        for dim_name, dim_value in (
            ("rows", rows),
            ("hidden_size", hidden_size),
            ("intermediate_size", intermediate_size),
        ):
            if dim_value % mlen != 0:
                print(f"===== Dimension {dim_name}={dim_value} is not aligned to mlen={mlen}, cannot apply plena_shared lowering =====")
                return LoweringResult(
                    mode="unsupported",
                    comments=[f"{dim_name}={dim_value} is not aligned to mlen={mlen}"],
                )

        context = self.env.shared_context
        primary_in_sym = ((node.get("in_syms") or [None])[:1] or [None])[0]
        if primary_in_sym is None:
            primary_in_sym = self._canonical_param_name(node.get("name", "vision_mlp"), "input")
        producer = (node.get("in_sym_sources") or {}).get(primary_in_sym, "model_input")
        input_binding = self._resolve_symbol_binding(primary_in_sym)
        if input_binding is not None:
            input_record = context.bind_external_symbol(
                symbol=primary_in_sym,
                matrix_shape=input_matrix_shape,
                producer=input_binding.producer,
                vram_addr=input_binding.address,
                semantic_shape=self._shape_tuple(input_meta),
                name_hint=primary_in_sym,
            )
            input_tensor = input_record.tensor
            if not isinstance(input_tensor, VRAMMatrixVar):
                raise TypeError(f"Expected shared external binding for {primary_in_sym} to produce VRAMMatrixVar")
        else:
            input_tensor = context.ensure_symbol_tensor(
                symbol=primary_in_sym,
                matrix_shape=input_matrix_shape,
                producer=producer,
                hbm_addr=self._ensure_runtime_symbol_input(primary_in_sym, input_matrix_shape),
            )

        fc1_weight, fc1_weight_binding = self._shared_parameter_handle(
            canonical_name=self._canonical_param_name(fc1.get("name", node.get("name", "vision_mlp")), "weight"),
            logical_shape=(hidden_size, intermediate_size),
            source_shape=self._weight_shape(fc1, "weight"),
            layout="linear_weight_in_out",
        )
        fc2_weight, fc2_weight_binding = self._shared_parameter_handle(
            canonical_name=self._canonical_param_name(fc2.get("name", node.get("name", "vision_mlp")), "weight"),
            logical_shape=(intermediate_size, hidden_size),
            source_shape=self._weight_shape(fc2, "weight"),
            layout="linear_weight_in_out",
        )

        fc1_bias = None
        fc1_bias_binding = None
        if bool(fc1_attrs.get("bias", False)):
            fc1_bias, fc1_bias_binding = self._shared_parameter_handle(
                canonical_name=self._canonical_param_name(fc1.get("name", node.get("name", "vision_mlp")), "bias"),
                logical_shape=(mlen, intermediate_size),
                source_shape=self._weight_shape(fc1, "bias"),
                layout="row_block_bias",
            )

        fc2_bias = None
        fc2_bias_binding = None
        if bool(fc2_attrs.get("bias", False)):
            fc2_bias, fc2_bias_binding = self._shared_parameter_handle(
                canonical_name=self._canonical_param_name(fc2.get("name", node.get("name", "vision_mlp")), "bias"),
                logical_shape=(mlen, hidden_size),
                source_shape=self._weight_shape(fc2, "bias"),
                layout="row_block_bias",
            )

        gelu_constants = GELUTanhFPRAMConstants(
            one=context.constants["one"],
            half=context.constants["half"],
            two=context.constants["two"],
            neg_one=context.constants["neg_one"],
            gelu_scale=context.constants["gelu_scale"],
            gelu_cubic=context.constants["gelu_cubic"],
        )
        compiler = Qwen3VLVisionMLPCompiler(
            spec=Qwen3VLVisionMLPSpec(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=(node.get("attrs") or {}).get("hidden_act", "gelu_pytorch_tanh"),
            ),
            mlen=mlen,
            blen=int(self.env.hw_value("blen", 4)),
            real_data_ratio=float(self.env.hw_value("real_data_ratio", 1.125)),
            gelu_constants=gelu_constants,
        )

        namespace = safe_codegen_name(node.get("name", "vision_mlp"), prefix="node")
        free_input = context.is_symbol_dead_after(primary_in_sym, int(node.get("order", 0)))
        emission = compiler.emit(
            prog=context.program,
            hidden_state=input_tensor,
            linear_fc1_weight=fc1_weight,
            linear_fc2_weight=fc2_weight,
            namespace=namespace,
            linear_fc1_bias=fc1_bias,
            linear_fc2_bias=fc2_bias,
            free_hidden_state=free_input,
        )
        if free_input:
            context.discard_symbol(primary_in_sym)

        output_meta = node.get("out")
        if isinstance(output_meta, list):
            output_meta = next((meta for meta in output_meta if meta is not None), None)
        semantic_shape = self._shape_tuple(output_meta if isinstance(output_meta, dict) else input_meta)
        out_symbols = node.get("out_syms") or []
        if not out_symbols:
            out_symbols = [self._canonical_param_name(node.get("name", "vision_mlp"), "out")]
        for out_sym in out_symbols:
            context.bind_symbol(
                out_sym,
                emission.output,
                producer=node.get("name", ""),
                semantic_shape=semantic_shape,
                padded_shape=emission.output.shape,
            )
            if context.liveness_for(out_sym).get("is_model_output"):
                context.mark_result_symbol(out_sym)

        context.mark_shared_lowering_used()
        return LoweringResult(
            mode="plena_shared",
            comments=[
                f"shared symbol input={primary_in_sym}",
                f"shared output={out_symbols[0]}",
                f"namespace={namespace}",
                (
                    f"input_activation={input_binding.address} "
                    f"({input_binding.source_kind}:{input_binding.producer})"
                    if input_binding is not None
                    else f"input_hbm={self._runtime_symbol_hbm_inputs.get(primary_in_sym, 'auto')}"
                ),
                f"fc1_weight hbm={fc1_weight_binding.hbm_addr}",
                f"fc2_weight hbm={fc2_weight_binding.hbm_addr}",
                *([f"fc1_bias hbm={fc1_bias_binding.hbm_addr}"] if fc1_bias_binding is not None else []),
                *([f"fc2_bias hbm={fc2_bias_binding.hbm_addr}"] if fc2_bias_binding is not None else []),
            ],
        )

    def generate(self, nodes: list[dict[str, Any]], model_info: dict[str, Any]) -> str:
        self._reset_runtime_state()
        self.env.reset_shared_context(self._effective_nodes(nodes))

        sections: list[str] = [self._program_header(model_info).rstrip()]
        covered_prefixes: dict[str, str] = {}
        type_bodies: dict[str, str] = {}
        stats = {
            "template": 0,
            "plena_shared": 0,
            "partial": 0,
            "unsupported": 0,
            "covered": 0,
            "skipped_unknown": 0,
        }
        unknown: list[tuple[str, str]] = []
        template_sections: list[str] = []
        shared_sections: list[str] = []
        covered_sections: list[str] = []

        for node in nodes:
            name = node.get("name", "")
            node_type = node.get("type", "")

            covering_name = next((owner for prefix, owner in covered_prefixes.items() if name and name.startswith(prefix)), None)
            if covering_name is not None:
                stats["covered"] += 1
                self._covered_by[name] = covering_name
                covered_sections.append(self._render_node_section(node, self._covered_node_result(node, covering_name)))
                continue

            op_key = self.env.operation_for(node_type)
            handler = self.operation_handlers.get(op_key) if op_key is not None else None
            if handler is None:
                stats["skipped_unknown"] += 1
                unknown.append((name, node_type))
                continue

            try:
                result = handler(node, model_info)
            except Exception as exc:
                result = LoweringResult(mode="partial", comments=[f"handler error: {exc}"])

            stats[result.mode] = stats.get(result.mode, 0) + 1
            render_result = result
            if result.reuse_label and not self.debug:
                body = result.asm.rstrip()
                setup = result.setup_asm
                existing_body = type_bodies.get(result.reuse_label)
                if existing_body is None:
                    type_bodies[result.reuse_label] = body
                    render_result = LoweringResult(
                        mode=result.mode,
                        asm=(setup + f"call {result.reuse_label}:\n").rstrip() + "\n",
                        comments=[*result.comments, f"type_body={result.reuse_label} (generated once)"],
                    )
                elif existing_body == body:
                    render_result = LoweringResult(
                        mode=result.mode,
                        asm=(setup + f"call {result.reuse_label}:\n").rstrip() + "\n",
                        comments=[*result.comments, f"type_body={result.reuse_label} (reused)"],
                    )
                else:
                    render_result = LoweringResult(
                        mode=result.mode,
                        asm=result.setup_asm + result.asm,
                        comments=[*result.comments, f"type_body={result.reuse_label} (signature mismatch, inlined)"],
                    )
            elif result.setup_asm:
                render_result = LoweringResult(
                    mode=result.mode,
                    asm=result.setup_asm + result.asm,
                    comments=result.comments,
                )

            rendered = self._render_node_section(node, render_result)
            if result.mode == "plena_shared":
                shared_sections.append(rendered)
            else:
                template_sections.append(rendered)

            if name and node.get("children") and result.mode != "unsupported":
                covered_prefixes[name + "."] = name

        if template_sections:
            sections.append("; === Template / Partial Lowering ===")
            sections.extend(s.rstrip() for s in template_sections)

        if covered_sections:
            sections.append("; === Covered Node Bindings ===")
            sections.extend(s.rstrip() for s in covered_sections)

        if shared_sections:
            sections.append("; === Shared-Context Lowering Summary ===")
            sections.extend(s.rstrip() for s in shared_sections)

        if type_bodies:
            sections.append("; === Reusable Type Bodies ===")
            for label, body in type_bodies.items():
                sections.append(f"{label}:")
                if body:
                    sections.append(body)

        if self.env.shared_context.has_shared_lowering:
            sections.append("; === Shared PLENA Program ===")
            shared_program = self.env.shared_context.program.compile().rstrip()
            if shared_program and not self.debug:
                sections.append(shared_program)
            sections.extend(self.env.shared_context.debug_dump_lines())

        sections.append(self._program_footer().rstrip())
        sections.append(
            "; stats: "
            f"template={stats['template']}  "
            f"plena_shared={stats['plena_shared']}  "
            f"partial={stats['partial']}  "
            f"unsupported={stats['unsupported']}  "
            f"covered={stats['covered']}  "
            f"skipped_unknown={stats['skipped_unknown']}"
        )
        if unknown:
            sections.append("; unknown nodes:")
            for name, node_type in unknown:
                sections.append(f";   {name} ({node_type})")

        return "\n".join(section for section in sections if section) + "\n"
