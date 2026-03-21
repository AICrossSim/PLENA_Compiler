"""
Assembly generation engine for traced VLM nodes.
"""

from __future__ import annotations

from math import prod
from typing import Any, Mapping

from .kernel_compilers.plena.compiler.plena_program import VRAMMatrixVar
from .vlm_codegen_context import matrix_shape_from_meta
from .vlm_codegen_env import VLMCodegenEnvironment
from .vlm_codegen_handlers import (
    LoweringResult,
    RuntimeActivationBinding,
    RuntimeParameterBinding,
    VLMCodegenHandler,
    register_default_handlers,
)


class VLMAssemblyGenerator:
    """Turns traced nodes into assembly using a VLMCodegenEnvironment."""

    def __init__(self, env: VLMCodegenEnvironment, *, auto_register_default_handlers: bool = True) -> None:
        self.env = env
        self.debug = False
        self.operation_handlers: dict[str, VLMCodegenHandler] = {}
        if auto_register_default_handlers:
            register_default_handlers(self)

    def debug_mode(self, enable: bool = True) -> None:
        self.debug = enable

    def register_handler(self, operation_key: str, handler: VLMCodegenHandler) -> None:
        self.operation_handlers[operation_key] = handler

    def register_handlers(self, mapping: Mapping[str, VLMCodegenHandler]) -> None:
        for operation_key, handler in mapping.items():
            self.register_handler(operation_key, handler)

    def clear_handlers(self) -> None:
        self.operation_handlers.clear()

    def operation_key_for(self, node_type: str) -> str | None:
        return self.env.operation_for(node_type)

    def handler_for_operation(self, operation_key: str | None) -> VLMCodegenHandler | None:
        if operation_key is None:
            return None
        return self.operation_handlers.get(operation_key)

    def handler_for_node(self, node: dict[str, Any]) -> VLMCodegenHandler | None:
        return self.handler_for_operation(self.operation_key_for(node.get("type", "")))

    def in_shape(self, node: dict[str, Any], idx: int = 0) -> list[int] | None:
        return self._in_shape(node, idx)

    def out_shape(self, node: dict[str, Any]) -> list[int] | None:
        return self._out_shape(node)

    def batch_from_node(self, node: dict[str, Any], model_info: dict[str, Any]) -> int:
        return self._batch_from_node(node, model_info)

    def seq_len_from_node(self, node: dict[str, Any], model_info: dict[str, Any]) -> int:
        return self._seq_len_from_node(node, model_info)

    def hidden_from_node(self, node: dict[str, Any], model_info: dict[str, Any]) -> int:
        return self._hidden_from_node(node, model_info)

    def shape_tuple(self, meta: dict[str, Any] | None) -> tuple[int, ...] | None:
        return self._shape_tuple(meta)

    def matrix_shape(self, meta: dict[str, Any] | None) -> tuple[int, int] | None:
        return self._matrix_shape(meta)

    def runtime_symbol_input_address(self, symbol: str) -> int | None:
        return getattr(self, "_runtime_symbol_hbm_inputs", {}).get(symbol)

    def ensure_runtime_symbol_input(self, symbol: str, matrix_shape: tuple[int, int]) -> int:
        return self._ensure_runtime_symbol_input(symbol, matrix_shape)

    def ensure_runtime_parameter(
        self,
        *,
        canonical_name: str,
        logical_shape: tuple[int, int],
        source_shape: tuple[int, ...] | None = None,
        layout: str = "logical",
    ) -> RuntimeParameterBinding:
        return self._ensure_runtime_parameter(
            canonical_name=canonical_name,
            logical_shape=logical_shape,
            source_shape=source_shape,
            layout=layout,
        )

    def shared_parameter_handle(
        self,
        *,
        canonical_name: str,
        logical_shape: tuple[int, int],
        source_shape: tuple[int, ...] | None = None,
        layout: str = "logical",
    ):
        return self._shared_parameter_handle(
            canonical_name=canonical_name,
            logical_shape=logical_shape,
            source_shape=source_shape,
            layout=layout,
        )

    def choose_activation_address(self, preferred: int, avoid: set[int] | None = None) -> int:
        return self._choose_activation_address(preferred, avoid)

    def bind_template_outputs(self, node: dict[str, Any], address: int) -> None:
        self._bind_template_outputs(node, address)

    def resolve_symbol_binding(self, symbol: str | None) -> RuntimeActivationBinding | None:
        return self._resolve_symbol_binding(symbol)

    def resolve_input_activation(
        self,
        node: dict[str, Any],
        *,
        idx: int = 0,
        default_block: str = "block1",
    ) -> tuple[int, RuntimeActivationBinding | None]:
        return self._resolve_input_activation(node, idx=idx, default_block=default_block)

    def emit_addr_reg_preload(
        self,
        bindings: list[tuple[int, RuntimeParameterBinding]],
        *,
        alive_registers: list[int],
    ) -> str:
        return self._emit_addr_reg_preload(bindings, alive_registers=alive_registers)

    def wrap_template(
        self,
        asm: str,
        *,
        mode: str = "template",
        setup_asm: str = "",
        reuse_label: str | None = None,
        comments: list[str] | None = None,
    ) -> LoweringResult:
        return self._wrap_template(
            asm,
            mode=mode,
            setup_asm=setup_asm,
            reuse_label=reuse_label,
            comments=comments,
        )

    def render_node_section(self, node: dict[str, Any], result: LoweringResult) -> str:
        return self._render_node_section(node, result)

    def child_linear(
        self,
        node: dict[str, Any],
        suffixes: tuple[str, ...],
        default_index: int | None = None,
    ) -> dict[str, Any] | None:
        return self._child_linear(node, suffixes, default_index)

    def linear_weight_binding(self, node: dict[str, Any]) -> RuntimeParameterBinding | None:
        return self._linear_weight_binding(node)

    def linear_weight_reg_binding(self, node: dict[str, Any]) -> tuple[str, int]:
        return self._linear_weight_reg_binding(node)

    def row_block_bias_binding(self, node: dict[str, Any], out_features: int) -> RuntimeParameterBinding | None:
        return self._row_block_bias_binding(node, out_features)

    def vision_linear_child(self, node: dict[str, Any], key: str) -> dict[str, Any] | None:
        return self._vision_linear_child(node, key)

    def weight_shape(self, node: dict[str, Any], weight_name: str) -> tuple[int, ...] | None:
        return self._weight_shape(node, weight_name)

    def canonical_param_name(self, node_name: str, leaf: str) -> str:
        return self._canonical_param_name(node_name, leaf)

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
            handler = self.handler_for_node(node)
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

            handler = self.handler_for_node(node)
            if handler is None:
                stats["skipped_unknown"] += 1
                unknown.append((name, node_type))
                continue

            try:
                result = handler(self, node, model_info)
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
