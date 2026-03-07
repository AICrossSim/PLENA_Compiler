"""
Assembly generation engine for traced VLM nodes.
"""

from __future__ import annotations

from typing import Any

from vlm_codegen_env import VLMCodegenEnvironment


class VLMAssemblyGenerator:
    """Turns traced nodes into assembly using a VLMCodegenEnvironment."""

    def __init__(self, env: VLMCodegenEnvironment) -> None:
        self.env = env
        self.operation_handlers = {
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
        }

    def _in_shape(self, node: dict, idx: int = 0) -> list[int] | None:
        in_list = node.get("in") or []
        if idx < len(in_list) and in_list[idx]:
            return in_list[idx]["shape"]
        return None

    def _out_shape(self, node: dict) -> list[int] | None:
        out = node.get("out")
        if isinstance(out, dict):
            return out.get("shape")
        if isinstance(out, list):
            for sh in out:
                if sh is not None:
                    return sh["shape"]
        return None

    def _batch_from_node(self, node: dict, model_info: dict) -> int:
        sh = self._in_shape(node)
        if sh and len(sh) >= 1:
            return sh[0]
        return model_info.get("batch_size", 1)

    def _seq_len_from_node(self, node: dict, model_info: dict) -> int:
        sh = self._in_shape(node)
        if sh and len(sh) >= 2:
            return sh[1]
        return model_info.get("seq_len", 1)

    def _hidden_from_node(self, node: dict, model_info: dict) -> int:
        attrs = node.get("attrs") or {}
        if "hidden_size" in attrs:
            return attrs["hidden_size"]
        sh = self._in_shape(node)
        if sh:
            return sh[-1]
        return model_info.get("hidden_size", 2048)

    def _program_header(self, model_info: dict) -> str:
        return (
            "; ============================================================\n"
            f"; PLENA VLM Assembly  —  {model_info.get('model_name', 'unknown')}\n"
            f"; hidden_size={model_info.get('hidden_size', '?')}  "
            f"layers={model_info.get('num_layers', '?')}  "
            f"heads={model_info.get('num_attention_heads', '?')}\n"
            "; ============================================================\n"
        )

    def _program_footer(self) -> str:
        return "; ============================================================\n; END\n"

    def _node_header(self, node: dict) -> str:
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

        return "\n".join(lines) + "\n"

    def _codegen_embedding(self, node: dict, model_info: dict) -> str:
        attrs = node.get("attrs") or {}
        n_emb = attrs.get("num_embeddings", model_info.get("vocab_size", 32000))
        emb_dim = attrs.get("embedding_dim", model_info.get("hidden_size", 2048))
        batch = self._batch_from_node(node, model_info)

        return self.env.template("embedding")(
            mlen=self.env.hw_value("mlen", 16),
            blen=self.env.hw_value("blen", 16),
            batch=batch,
            hidden_size=emb_dim,
            alive_registers=self.env.hw_value("alive_registers", [1, 2, 3, 4]),
            voc_table_row_size=n_emb,
            activation_base_address=self.env.sched.get("activation_base_address", 0),
            voc_table_base_addr_reg_index=self.env.reg("token_table_offset", 1),
            input_ids=[1 for _ in range(batch)],
        )

    def _codegen_rms_norm(self, node: dict, model_info: dict) -> str:
        hidden = self._hidden_from_node(node, model_info)
        batch = self._batch_from_node(node, model_info)

        return self.env.template("rms_norm")(
            _eps_offset=self.env.fp_mem("eps", 0),
            reci_hid_offset=self.env.fp_mem("hid_reciprocal", 0),
            alive_registers=self.env.hw_value("alive_registers", [1, 2, 3, 4])[:3],
            activation_base_address=self.env.mem("block1", 0),
            scratchpad_base_address=self.env.mem("block2", 0),
            vlen=self.env.hw_value("vlen", 16),
            batch_size=batch,
            hidden_dim=hidden,
        )

    def _codegen_layer_norm(self, node: dict, model_info: dict) -> str:
        attrs = node.get("attrs") or {}
        ns = attrs.get("normalized_shape", [model_info.get("hidden_size", 2048)])
        hidden = ns[0] if isinstance(ns, list) else ns
        batch = self._batch_from_node(node, model_info)

        return self.env.template("rms_norm")(
            _eps_offset=self.env.fp_mem("eps", 0),
            reci_hid_offset=self.env.fp_mem("hid_reciprocal", 0),
            alive_registers=self.env.hw_value("alive_registers", [1, 2, 3, 4])[:3],
            activation_base_address=self.env.mem("block1", 0),
            scratchpad_base_address=self.env.mem("block2", 0),
            vlen=self.env.hw_value("vlen", 16),
            batch_size=batch,
            hidden_dim=hidden,
        )

    def _codegen_text_attention(self, node: dict, model_info: dict) -> str:
        sh = self._in_shape(node)
        hidden_size = sh[-1] if sh else model_info.get("hidden_size", 2048)
        head_dim = model_info.get("head_dim", hidden_size // model_info.get("num_attention_heads", 16))
        batch = sh[0] if sh else model_info.get("batch_size", 1)
        regs = self.env.hw_value("alive_registers", [1, 2, 3, 4, 5, 6, 7, 8])

        proj_common = dict(
            mlen=self.env.hw_value("mlen", 16),
            blen=self.env.hw_value("blen", 16),
            batch=batch,
            hidden_size=hidden_size,
            alive_registers=regs,
            out_features=head_dim,
            rope_hbm_offset_reg=self.env.reg("rope_params_offset", 6),
            rope_on_chip_address=self.env.mem("block3", 0),
            activation_base_address=self.env.mem("block1", 0),
            result_base_address=self.env.mem("block2", 0),
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
        return code

    def _codegen_vision_attention(self, node: dict, model_info: dict) -> str:
        sh = self._in_shape(node)
        hidden_size = sh[-1] if sh else model_info.get("vision_hidden_size", model_info.get("hidden_size", 2048))
        head_dim = model_info.get("vision_head_dim", hidden_size // model_info.get("num_attention_heads", 16))
        regs = self.env.hw_value("alive_registers", [1, 2, 3, 4, 5, 6, 7, 8])

        proj_common = dict(
            mlen=self.env.hw_value("mlen", 16),
            blen=self.env.hw_value("blen", 16),
            batch=1,
            hidden_size=hidden_size,
            alive_registers=regs,
            out_features=head_dim,
            rope_hbm_offset_reg=self.env.reg("rope_params_offset", 6),
            rope_on_chip_address=self.env.mem("block3", 0),
            activation_base_address=self.env.mem("block1", 0),
            result_base_address=self.env.mem("block2", 0),
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
        return code

    def _codegen_mlp(self, node: dict, model_info: dict) -> str:
        children = node.get("children") or []
        if not children:
            return "; [unsupported composite MLP] no child ops available\n"

        parts: list[str] = []
        for child in children:
            child_type = child.get("type", "")
            op_key = self.env.operation_for(child_type)
            handler = self.operation_handlers.get(op_key) if op_key is not None else None
            if handler is None:
                parts.append(f"; [unsupported child in MLP] {child.get('name', '')} ({child_type})\n")
                continue
            try:
                parts.append(self._node_header(child) + handler(child, model_info))
            except Exception as exc:
                parts.append(f"; [ERROR in MLP child {child_type} handler: {exc}]\n")

        return "\n".join(parts)

    def _codegen_ffn(self, node: dict, model_info: dict) -> str:
        sh = self._in_shape(node)
        hidden_size = sh[-1] if sh else model_info.get("hidden_size", 2048)
        batch = sh[0] if sh else model_info.get("batch_size", 1)
        seq_len = sh[1] if sh and len(sh) >= 2 else self._seq_len_from_node(node, model_info)

        attrs = node.get("attrs") or {}
        inter_size = attrs.get("intermediate_size") or model_info.get("intermediate_size") or hidden_size * 4

        return self.env.template("ffn")(
            mlen=self.env.hw_value("mlen", 16),
            vlen=self.env.hw_value("vlen", 16),
            blen=self.env.hw_value("blen", 16),
            batch=batch,
            seq_len=seq_len,
            hidden_size=hidden_size,
            intermediate_size=inter_size,
            alive_registers=self.env.hw_value("alive_registers", [1, 2, 3, 4, 5, 6, 7, 8]),
            gate_weight_hbm_offset_reg=self.env.reg("gate_weight_offset", self.env.reg("ffn_weight_offset", 5)),
            up_weight_hbm_offset_reg=self.env.reg("up_weight_offset", self.env.reg("ffn_weight_offset", 5)),
            down_weight_hbm_offset_reg=self.env.reg("down_weight_offset", self.env.reg("ffn_weight_offset", 5)),
            const_one_fp_address=self.env.fp_mem("one", 0),
            activation_base_address=self.env.mem("block1", 0),
            use_loop_instructions=True,
        )

    def _codegen_linear(self, node: dict, model_info: dict) -> str:
        attrs = node.get("attrs") or {}
        in_features = attrs.get("in_features", model_info.get("hidden_size", 2048))
        out_features = attrs.get("out_features", model_info.get("hidden_size", 2048))
        batch = self._batch_from_node(node, model_info)

        return self.env.template("projection")(
            mlen=self.env.hw_value("mlen", 16),
            blen=self.env.hw_value("blen", 16),
            batch=batch,
            hidden_size=in_features,
            alive_registers=self.env.hw_value("alive_registers", [1, 2, 3, 4, 5, 6, 7, 8]),
            out_features=out_features,
            w_base_hbm_offset_reg=self.env.reg("q_weight_offset", 2),
            rope_hbm_offset_reg=self.env.reg("rope_params_offset", 6),
            rope_on_chip_address=self.env.mem("block3", 0),
            activation_base_address=self.env.mem("block1", 0),
            result_base_address=self.env.mem("block2", 0),
            rope_enabled=False,
        )

    def _codegen_conv3d(self, node: dict, model_info: dict) -> str:
        attrs = node.get("attrs") or {}
        out_ch = attrs.get("out_channels", model_info.get("hidden_size", 2048))

        return self.env.template("batched_matmul")(
            mlen=self.env.hw_value("mlen", 16),
            blen=self.env.hw_value("blen", 16),
            hidden_size=out_ch,
            alive_registers=self.env.hw_value("alive_registers", [1, 2, 3, 4, 5, 6, 7, 8]),
            activation_base_address=self.env.mem("block1", 0),
            result_base_address=self.env.mem("block2", 0),
            weight_hbm_offset_reg=self.env.reg("q_weight_offset", 2),
        )

    def _codegen_elementwise_add(self, node: dict, model_info: dict) -> str:
        sh = self._in_shape(node)
        hidden_size = (sh[-1] if sh else None) or model_info.get("hidden_size", 2048)
        batch = (sh[0] if sh else None) or model_info.get("batch_size", 1)

        return self.env.template("elementwise_add")(
            vlen=self.env.hw_value("vlen", 16),
            hidden_size=hidden_size,
            batch=batch,
            alive_registers=self.env.hw_value("alive_registers", [1, 2, 3, 4])[:3],
            stored_activation_base_address=self.env.mem("block1", 0),
            previous_activation_base_address=self.env.mem("block2", 0),
            previous_act_on_chip_addr_reg_index=self.env.reg("previous_activation_offset", 7),
        )

    def _codegen_unsupported(self, node: dict) -> str:
        return f"; [unsupported] {node['type']} — no codegen handler registered\n"

    def generate(self, nodes: list[dict], model_info: dict[str, Any]) -> str:
        asm_parts = [self._program_header(model_info)]
        covered: list[str] = []
        stats = {"generated": 0, "skipped_covered": 0, "skipped_unknown": 0}
        unknown: list[tuple[str, str]] = []

        for node in nodes:
            name = node.get("name", "")
            node_type = node.get("type", "")

            if name and any(name.startswith(prefix) for prefix in covered):
                stats["skipped_covered"] += 1
                continue

            op_key = self.env.operation_for(node_type)
            handler = self.operation_handlers.get(op_key) if op_key is not None else None
            if handler is None:
                stats["skipped_unknown"] += 1
                unknown.append((name, node_type))
                continue

            try:
                asm_block = handler(node, model_info)
            except Exception as exc:
                asm_block = f"; [ERROR in {node_type} handler: {exc}]\n"

            asm_parts.append(self._node_header(node) + asm_block)
            stats["generated"] += 1

            if name:
                covered.append(name + ".")

        asm_parts.append(self._program_footer())
        asm_parts.append(
            f"; stats: generated={stats['generated']}  "
            f"skipped_covered={stats['skipped_covered']}  "
            f"skipped_unknown={stats['skipped_unknown']}\n"
        )
        if unknown:
            asm_parts.append("; unknown nodes:\n")
            for name, node_type in unknown:
                asm_parts.append(f";   {name} ({node_type})\n")

        return "\n".join(asm_parts)
