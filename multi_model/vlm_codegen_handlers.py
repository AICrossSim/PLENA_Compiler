"""
Default lowering handlers and registration helpers for VLM assembly generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from asm_lib.plena_program import VRAMMatrixVar
from asm_lib.qwen3_vl_vision_mlp import (
    GELUTanhFPRAMConstants,
    Qwen3VLVisionMLPCompiler,
    Qwen3VLVisionMLPSpec,
)
from vlm_codegen_context import safe_codegen_name

if TYPE_CHECKING:
    from vlm_codegen_generator import VLMAssemblyGenerator


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


VLMCodegenHandler = Callable[["VLMAssemblyGenerator", dict[str, Any], dict[str, Any]], LoweringResult]


def embedding_handler(
    generator: "VLMAssemblyGenerator",
    node: dict[str, Any],
    model_info: dict[str, Any],
) -> LoweringResult:
    attrs = node.get("attrs") or {}
    n_emb = attrs.get("num_embeddings", model_info.get("vocab_size", 32000))
    emb_dim = attrs.get("embedding_dim", model_info.get("hidden_size", 2048))
    batch = generator.batch_from_node(node, model_info)
    regs = generator.env.hw_value("alive_registers", [1, 2, 3, 4])
    activation_addr = generator.env.sched.get("activation_base_address", 0)

    comments: list[str] = []
    prelude = ""
    source_shape = generator.weight_shape(node, "weight")
    if source_shape is not None:
        binding = generator.ensure_runtime_parameter(
            canonical_name=generator.canonical_param_name(node.get("name", "embedding"), "weight"),
            logical_shape=(int(n_emb), int(emb_dim)),
            source_shape=source_shape,
            layout="embedding_table",
        )
        prelude = generator.emit_addr_reg_preload(
            [(generator.env.reg("token_table_offset", 1), binding)],
            alive_registers=regs,
        )
        comments.append(f"token_table hbm={binding.hbm_addr}")

    asm = generator.env.template("embedding")(
        mlen=generator.env.hw_value("mlen", 64),
        blen=generator.env.hw_value("blen", 4),
        batch=batch,
        hidden_size=emb_dim,
        alive_registers=regs,
        voc_table_row_size=n_emb,
        activation_base_address=activation_addr,
        voc_table_base_addr_reg_index=generator.env.reg("token_table_offset", 1),
        input_ids=[1 for _ in range(batch)],
    )
    generator.bind_template_outputs(node, activation_addr)
    return generator.wrap_template(
        asm,
        setup_asm=prelude,
        reuse_label=node.get("type"),
        comments=comments,
    )


def rms_norm_handler(
    generator: "VLMAssemblyGenerator",
    node: dict[str, Any],
    model_info: dict[str, Any],
) -> LoweringResult:
    hidden = generator.hidden_from_node(node, model_info)
    batch = generator.batch_from_node(node, model_info)
    activation_addr, input_binding = generator.resolve_input_activation(node, default_block="block1")
    scratchpad_addr = generator.choose_activation_address(
        generator.env.mem("block2", 0),
        avoid={activation_addr},
    )

    asm = generator.env.template("rms_norm")(
        _eps_offset=generator.env.fp_mem("eps", 0),
        reci_hid_offset=generator.env.fp_mem("hid_reciprocal", 0),
        alive_registers=generator.env.hw_value("alive_registers", [1, 2, 3, 4])[:3],
        activation_base_address=activation_addr,
        scratchpad_base_address=scratchpad_addr,
        vlen=generator.env.hw_value("vlen", 16),
        batch_size=batch,
        hidden_dim=hidden,
    )
    generator.bind_template_outputs(node, activation_addr)
    comments = [f"activation={activation_addr}"]
    if input_binding is not None:
        comments.append(f"input_source={input_binding.source_kind}:{input_binding.producer}")
    return generator.wrap_template(asm, reuse_label=node.get("type"), comments=comments)


def layer_norm_handler(
    generator: "VLMAssemblyGenerator",
    node: dict[str, Any],
    model_info: dict[str, Any],
) -> LoweringResult:
    attrs = node.get("attrs") or {}
    normalized_shape = attrs.get("normalized_shape", [model_info.get("hidden_size", 2048)])
    hidden = normalized_shape[0] if isinstance(normalized_shape, list) else normalized_shape
    batch = generator.batch_from_node(node, model_info)
    activation_addr, input_binding = generator.resolve_input_activation(node, default_block="block1")
    scratchpad_addr = generator.choose_activation_address(
        generator.env.mem("block2", 0),
        avoid={activation_addr},
    )

    asm = generator.env.template("layer_norm")(
        _eps_offset=generator.env.fp_mem("eps", 0),
        reci_hid_offset=generator.env.fp_mem("hid_reciprocal", 0),
        alive_registers=generator.env.hw_value("alive_registers", [1, 2, 3, 4])[:3],
        activation_base_address=activation_addr,
        scratchpad_base_address=scratchpad_addr,
        vlen=generator.env.hw_value("vlen", 16),
        batch_size=batch,
        hidden_dim=hidden,
    )
    generator.bind_template_outputs(node, activation_addr)
    comments = [f"activation={activation_addr}"]
    if input_binding is not None:
        comments.append(f"input_source={input_binding.source_kind}:{input_binding.producer}")
    return generator.wrap_template(asm, reuse_label=node.get("type"), comments=comments)


def text_attention_handler(
    generator: "VLMAssemblyGenerator",
    node: dict[str, Any],
    model_info: dict[str, Any],
) -> LoweringResult:
    input_shape = generator.in_shape(node)
    hidden_size = input_shape[-1] if input_shape else model_info.get("hidden_size", 2048)
    head_dim = model_info.get("head_dim", hidden_size // model_info.get("num_attention_heads", 16))
    batch = input_shape[0] if input_shape else model_info.get("batch_size", 1)
    regs = generator.env.hw_value("alive_registers", [1, 2, 3, 4, 5, 6, 7, 8])
    activation_addr, input_binding = generator.resolve_input_activation(node, default_block="block1")
    result_addr = generator.choose_activation_address(generator.env.mem("block2", 0), avoid={activation_addr})
    q_child = generator.child_linear(node, ("q_proj", ".q"), default_index=0)
    k_child = generator.child_linear(node, ("k_proj", ".k"), default_index=1)
    v_child = generator.child_linear(node, ("v_proj", ".v"), default_index=2)
    q_binding = generator.linear_weight_binding(q_child) if q_child is not None else None
    k_binding = generator.linear_weight_binding(k_child) if k_child is not None else None
    v_binding = generator.linear_weight_binding(v_child) if v_child is not None else None
    prelude = generator.emit_addr_reg_preload(
        [
            pair
            for pair in (
                (generator.env.reg("q_weight_offset", 2), q_binding),
                (generator.env.reg("k_weight_offset", 3), k_binding),
                (generator.env.reg("v_weight_offset", 4), v_binding),
            )
            if pair[1] is not None
        ],
        alive_registers=regs,
    )

    proj_common = dict(
        mlen=generator.env.hw_value("mlen", 64),
        blen=generator.env.hw_value("blen", 4),
        batch=batch,
        hidden_size=hidden_size,
        alive_registers=regs,
        out_features=head_dim,
        rope_hbm_offset_reg=generator.env.reg("rope_params_offset", 6),
        rope_on_chip_address=generator.env.mem("block3", 0),
        activation_base_address=activation_addr,
        result_base_address=result_addr,
    )

    code = "; --- Q projection\n"
    code += generator.env.template("projection")(
        **proj_common,
        w_base_hbm_offset_reg=generator.env.reg("q_weight_offset", 2),
        rope_enabled=True,
    )
    code += "; --- K projection\n"
    code += generator.env.template("projection")(
        **proj_common,
        w_base_hbm_offset_reg=generator.env.reg("k_weight_offset", 3),
        rope_enabled=True,
    )
    code += "; --- V projection\n"
    code += generator.env.template("projection")(
        **proj_common,
        w_base_hbm_offset_reg=generator.env.reg("v_weight_offset", 4),
        rope_enabled=False,
    )
    code += "; --- Flash Attention (TODO: wire flash_attn_asm)\n"
    generator.bind_template_outputs(node, result_addr)
    comments = [f"activation={activation_addr}", f"result={result_addr}"]
    if input_binding is not None:
        comments.append(f"input_source={input_binding.source_kind}:{input_binding.producer}")
    for label, binding in (("q", q_binding), ("k", k_binding), ("v", v_binding)):
        if binding is not None:
            comments.append(f"{label}_weight hbm={binding.hbm_addr}")
    return generator.wrap_template(
        code,
        setup_asm=prelude,
        reuse_label=node.get("type"),
        comments=comments,
    )


def vision_attention_handler(
    generator: "VLMAssemblyGenerator",
    node: dict[str, Any],
    model_info: dict[str, Any],
) -> LoweringResult:
    input_shape = generator.in_shape(node)
    hidden_size = (
        input_shape[-1]
        if input_shape
        else model_info.get("vision_hidden_size", model_info.get("hidden_size", 2048))
    )
    head_dim = model_info.get("vision_head_dim", hidden_size // model_info.get("num_attention_heads", 16))
    regs = generator.env.hw_value("alive_registers", [1, 2, 3, 4, 5, 6, 7, 8])
    activation_addr, input_binding = generator.resolve_input_activation(node, default_block="block1")
    result_addr = generator.choose_activation_address(generator.env.mem("block2", 0), avoid={activation_addr})
    q_child = generator.child_linear(node, ("q_proj", ".q"), default_index=0)
    k_child = generator.child_linear(node, ("k_proj", ".k"), default_index=1)
    v_child = generator.child_linear(node, ("v_proj", ".v"), default_index=2)
    q_binding = generator.linear_weight_binding(q_child) if q_child is not None else None
    k_binding = generator.linear_weight_binding(k_child) if k_child is not None else None
    v_binding = generator.linear_weight_binding(v_child) if v_child is not None else None
    prelude = generator.emit_addr_reg_preload(
        [
            pair
            for pair in (
                (generator.env.reg("q_weight_offset", 2), q_binding),
                (generator.env.reg("k_weight_offset", 3), k_binding),
                (generator.env.reg("v_weight_offset", 4), v_binding),
            )
            if pair[1] is not None
        ],
        alive_registers=regs,
    )

    proj_common = dict(
        mlen=generator.env.hw_value("mlen", 64),
        blen=generator.env.hw_value("blen", 4),
        batch=1,
        hidden_size=hidden_size,
        alive_registers=regs,
        out_features=head_dim,
        rope_hbm_offset_reg=generator.env.reg("rope_params_offset", 6),
        rope_on_chip_address=generator.env.mem("block3", 0),
        activation_base_address=activation_addr,
        result_base_address=result_addr,
        rope_enabled=False,
    )

    code = "; --- Vision Q projection\n"
    code += generator.env.template("projection")(
        **proj_common,
        w_base_hbm_offset_reg=generator.env.reg("q_weight_offset", 2),
    )
    code += "; --- Vision K projection\n"
    code += generator.env.template("projection")(
        **proj_common,
        w_base_hbm_offset_reg=generator.env.reg("k_weight_offset", 3),
    )
    code += "; --- Vision V projection\n"
    code += generator.env.template("projection")(
        **proj_common,
        w_base_hbm_offset_reg=generator.env.reg("v_weight_offset", 4),
    )
    code += "; --- Vision Flash Attention (TODO: wire flash_attn_asm)\n"
    generator.bind_template_outputs(node, result_addr)
    comments = [f"activation={activation_addr}", f"result={result_addr}"]
    if input_binding is not None:
        comments.append(f"input_source={input_binding.source_kind}:{input_binding.producer}")
    for label, binding in (("q", q_binding), ("k", k_binding), ("v", v_binding)):
        if binding is not None:
            comments.append(f"{label}_weight hbm={binding.hbm_addr}")
    return generator.wrap_template(
        code,
        setup_asm=prelude,
        reuse_label=node.get("type"),
        comments=comments,
    )


def mlp_handler(
    generator: "VLMAssemblyGenerator",
    node: dict[str, Any],
    model_info: dict[str, Any],
) -> LoweringResult:
    children = node.get("children") or []
    if not children:
        return LoweringResult(mode="unsupported", comments=["composite MLP has no child ops"])

    parts: list[str] = []
    non_template = False
    for child in children:
        handler = generator.handler_for_node(child)
        if handler is None:
            parts.append(f"; [unsupported child in MLP] {child.get('name', '')} ({child.get('type', '')})")
            non_template = True
            continue
        try:
            result = handler(generator, child, model_info)
        except Exception as exc:
            result = LoweringResult(mode="partial", comments=[f"child lowering failed: {exc}"])
        if result.mode != "template":
            non_template = True
        parts.append(generator.render_node_section(child, result).rstrip())

    return LoweringResult(
        mode="partial" if non_template else "template",
        asm="\n".join(parts),
        comments=["composite MLP expanded through child handlers"],
    )


def ffn_handler(
    generator: "VLMAssemblyGenerator",
    node: dict[str, Any],
    model_info: dict[str, Any],
) -> LoweringResult:
    input_shape = generator.in_shape(node)
    hidden_size = input_shape[-1] if input_shape else model_info.get("hidden_size", 2048)
    batch = input_shape[0] if input_shape else model_info.get("batch_size", 1)
    seq_len = input_shape[1] if input_shape and len(input_shape) >= 2 else generator.seq_len_from_node(node, model_info)

    attrs = node.get("attrs") or {}
    inter_size = attrs.get("intermediate_size") or model_info.get("intermediate_size") or hidden_size * 4
    activation_addr, input_binding = generator.resolve_input_activation(node, default_block="block1")
    regs = generator.env.hw_value("alive_registers", [1, 2, 3, 4, 5, 6, 7, 8])
    gate_child = generator.child_linear(node, ("gate_proj",), default_index=0)
    up_child = generator.child_linear(node, ("up_proj",), default_index=1)
    down_child = generator.child_linear(node, ("down_proj",), default_index=2)

    gate_binding = generator.linear_weight_binding(gate_child) if gate_child is not None else None
    up_binding = generator.linear_weight_binding(up_child) if up_child is not None else None
    down_binding = generator.linear_weight_binding(down_child) if down_child is not None else None
    preload_pairs = []
    gate_reg = generator.env.reg("gate_weight_offset", 5)
    up_reg = generator.env.reg("up_weight_offset", 6)
    down_reg = generator.env.reg("down_weight_offset", 7)
    for reg, binding in ((gate_reg, gate_binding), (up_reg, up_binding), (down_reg, down_binding)):
        if binding is not None:
            preload_pairs.append((reg, binding))
    prelude = generator.emit_addr_reg_preload(preload_pairs, alive_registers=regs)

    asm = generator.env.template("ffn")(
        mlen=generator.env.hw_value("mlen", 64),
        vlen=generator.env.hw_value("vlen", 16),
        blen=generator.env.hw_value("blen", 4),
        batch=batch,
        seq_len=seq_len,
        hidden_size=hidden_size,
        intermediate_size=inter_size,
        alive_registers=regs,
        gate_weight_hbm_offset_reg=gate_reg,
        up_weight_hbm_offset_reg=up_reg,
        down_weight_hbm_offset_reg=down_reg,
        const_one_fp_address=generator.env.fp_mem("one", 0),
        activation_base_address=activation_addr,
        use_loop_instructions=True,
    )
    generator.bind_template_outputs(node, activation_addr)
    comments = [f"activation={activation_addr}"]
    if input_binding is not None:
        comments.append(f"input_source={input_binding.source_kind}:{input_binding.producer}")
    for label, binding in (("gate", gate_binding), ("up", up_binding), ("down", down_binding)):
        if binding is not None:
            comments.append(f"{label}_weight hbm={binding.hbm_addr}")
    return generator.wrap_template(
        asm,
        setup_asm=prelude,
        reuse_label=node.get("type"),
        comments=comments,
    )


def linear_handler(
    generator: "VLMAssemblyGenerator",
    node: dict[str, Any],
    model_info: dict[str, Any],
) -> LoweringResult:
    attrs = node.get("attrs") or {}
    reg_name, weight_reg = generator.linear_weight_reg_binding(node)
    in_features = int(attrs.get("in_features", model_info.get("hidden_size", 2048)))
    out_features = int(attrs.get("out_features", model_info.get("hidden_size", 2048)))
    batch = generator.batch_from_node(node, model_info)
    regs = generator.env.hw_value("alive_registers", [1, 2, 3, 4, 5, 6, 7, 8])
    activation_addr, input_binding = generator.resolve_input_activation(node, default_block="block1")
    result_addr = generator.choose_activation_address(generator.env.mem("block2", 0), avoid={activation_addr})
    weight_binding = generator.linear_weight_binding(node)
    prelude = ""
    if weight_binding is not None:
        prelude = generator.emit_addr_reg_preload(
            [(weight_reg, weight_binding)],
            alive_registers=regs,
        )

    asm = generator.env.template("projection")(
        mlen=generator.env.hw_value("mlen", 64),
        blen=generator.env.hw_value("blen", 4),
        batch=batch,
        hidden_size=in_features,
        alive_registers=regs,
        out_features=out_features,
        w_base_hbm_offset_reg=weight_reg,
        rope_hbm_offset_reg=generator.env.reg("rope_params_offset", 6),
        rope_on_chip_address=generator.env.mem("block3", 0),
        activation_base_address=activation_addr,
        result_base_address=result_addr,
        rope_enabled=False,
    )
    generator.bind_template_outputs(node, result_addr)
    comments = [f"activation={activation_addr}", f"result={result_addr}"]
    if input_binding is not None:
        comments.append(f"input_source={input_binding.source_kind}:{input_binding.producer}")
    if weight_binding is not None:
        comments.append(f"weight hbm={weight_binding.hbm_addr}")
    comments.append(f"weight_reg={reg_name}:a{weight_reg}")
    return generator.wrap_template(
        asm,
        setup_asm=prelude,
        reuse_label=node.get("type"),
        comments=comments,
    )


def conv3d_handler(
    generator: "VLMAssemblyGenerator",
    node: dict[str, Any],
    model_info: dict[str, Any],
) -> LoweringResult:
    attrs = node.get("attrs") or {}
    out_ch = attrs.get("out_channels", model_info.get("hidden_size", 2048))

    asm = generator.env.template("batched_matmul")(
        mlen=generator.env.hw_value("mlen", 64),
        blen=generator.env.hw_value("blen", 4),
        hidden_size=out_ch,
        alive_registers=generator.env.hw_value("alive_registers", [1, 2, 3, 4, 5, 6, 7, 8]),
        activation_base_address=generator.env.mem("block1", 0),
        result_base_address=generator.env.mem("block2", 0),
        weight_hbm_offset_reg=generator.env.reg("q_weight_offset", 2),
    )
    return generator.wrap_template(asm, reuse_label=node.get("type"))


def elementwise_add_handler(
    generator: "VLMAssemblyGenerator",
    node: dict[str, Any],
    model_info: dict[str, Any],
) -> LoweringResult:
    input_shape = generator.in_shape(node)
    hidden_size = (input_shape[-1] if input_shape else None) or model_info.get("hidden_size", 2048)
    batch = (input_shape[0] if input_shape else None) or model_info.get("batch_size", 1)
    stored_addr, stored_binding = generator.resolve_input_activation(node, idx=0, default_block="block1")
    previous_addr, previous_binding = generator.resolve_input_activation(node, idx=1, default_block="block2")

    asm = generator.env.template("elementwise_add")(
        vlen=generator.env.hw_value("vlen", 16),
        hidden_size=hidden_size,
        batch=batch,
        alive_registers=generator.env.hw_value("alive_registers", [1, 2, 3, 4])[:3],
        stored_activation_base_address=stored_addr,
        previous_activation_base_address=previous_addr,
        previous_act_on_chip_addr_reg_index=generator.env.reg("previous_activation_offset", 7),
    )
    generator.bind_template_outputs(node, stored_addr)
    comments = [f"stored_activation={stored_addr}", f"previous_activation={previous_addr}"]
    if stored_binding is not None:
        comments.append(f"stored_source={stored_binding.source_kind}:{stored_binding.producer}")
    if previous_binding is not None:
        comments.append(f"previous_source={previous_binding.source_kind}:{previous_binding.producer}")
    return generator.wrap_template(asm, reuse_label=node.get("type"), comments=comments)


def qwen_vision_mlp_handler(
    generator: "VLMAssemblyGenerator",
    node: dict[str, Any],
    model_info: dict[str, Any],
) -> LoweringResult:
    fc1 = generator.vision_linear_child(node, "linear_fc1")
    fc2 = generator.vision_linear_child(node, "linear_fc2")
    if fc1 is None or fc2 is None:
        return LoweringResult(
            mode="unsupported",
            comments=["vision MLP lowering needs child Linear nodes for fc1/fc2"],
        )

    fc1_attrs = fc1.get("attrs") or {}
    fc2_attrs = fc2.get("attrs") or {}
    hidden_size = int(fc1_attrs.get("in_features", generator.hidden_from_node(node, model_info)))
    intermediate_size = int(fc1_attrs.get("out_features", fc2_attrs.get("in_features", hidden_size * 4)))
    output_hidden = int(fc2_attrs.get("out_features", hidden_size))
    if output_hidden != hidden_size:
        return LoweringResult(
            mode="unsupported",
            comments=[f"fc2 out_features={output_hidden} does not match hidden_size={hidden_size}"],
        )

    input_meta = (node.get("in") or [None])[0]
    input_matrix_shape = generator.matrix_shape(input_meta)
    if input_matrix_shape is None:
        return LoweringResult(mode="unsupported", comments=["vision MLP lowering needs concrete input shape"])
    rows, cols = input_matrix_shape
    if cols != hidden_size:
        return LoweringResult(
            mode="unsupported",
            comments=[f"input hidden width {cols} does not match fc1 in_features={hidden_size}"],
        )

    mlen = int(generator.env.hw_value("mlen", 64))
    for dim_name, dim_value in (
        ("rows", rows),
        ("hidden_size", hidden_size),
        ("intermediate_size", intermediate_size),
    ):
        if dim_value % mlen != 0:
            print(
                f"===== Dimension {dim_name}={dim_value} is not aligned to mlen={mlen}, "
                "cannot apply plena_shared lowering ====="
            )
            return LoweringResult(
                mode="unsupported",
                comments=[f"{dim_name}={dim_value} is not aligned to mlen={mlen}"],
            )

    context = generator.env.shared_context
    primary_in_sym = ((node.get("in_syms") or [None])[:1] or [None])[0]
    if primary_in_sym is None:
        primary_in_sym = generator.canonical_param_name(node.get("name", "vision_mlp"), "input")
    producer = (node.get("in_sym_sources") or {}).get(primary_in_sym, "model_input")
    input_binding = generator.resolve_symbol_binding(primary_in_sym)
    if input_binding is not None:
        input_record = context.bind_external_symbol(
            symbol=primary_in_sym,
            matrix_shape=input_matrix_shape,
            producer=input_binding.producer,
            vram_addr=input_binding.address,
            semantic_shape=generator.shape_tuple(input_meta),
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
            hbm_addr=generator.ensure_runtime_symbol_input(primary_in_sym, input_matrix_shape),
        )

    fc1_weight, fc1_weight_binding = generator.shared_parameter_handle(
        canonical_name=generator.canonical_param_name(fc1.get("name", node.get("name", "vision_mlp")), "weight"),
        logical_shape=(hidden_size, intermediate_size),
        source_shape=generator.weight_shape(fc1, "weight"),
        layout="linear_weight_in_out",
    )
    fc2_weight, fc2_weight_binding = generator.shared_parameter_handle(
        canonical_name=generator.canonical_param_name(fc2.get("name", node.get("name", "vision_mlp")), "weight"),
        logical_shape=(intermediate_size, hidden_size),
        source_shape=generator.weight_shape(fc2, "weight"),
        layout="linear_weight_in_out",
    )

    fc1_bias = None
    fc1_bias_binding = None
    if bool(fc1_attrs.get("bias", False)):
        fc1_bias, fc1_bias_binding = generator.shared_parameter_handle(
            canonical_name=generator.canonical_param_name(fc1.get("name", node.get("name", "vision_mlp")), "bias"),
            logical_shape=(mlen, intermediate_size),
            source_shape=generator.weight_shape(fc1, "bias"),
            layout="row_block_bias",
        )

    fc2_bias = None
    fc2_bias_binding = None
    if bool(fc2_attrs.get("bias", False)):
        fc2_bias, fc2_bias_binding = generator.shared_parameter_handle(
            canonical_name=generator.canonical_param_name(fc2.get("name", node.get("name", "vision_mlp")), "bias"),
            logical_shape=(mlen, hidden_size),
            source_shape=generator.weight_shape(fc2, "bias"),
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
        blen=int(generator.env.hw_value("blen", 4)),
        real_data_ratio=float(generator.env.hw_value("real_data_ratio", 1.125)),
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
    semantic_shape = generator.shape_tuple(output_meta if isinstance(output_meta, dict) else input_meta)
    out_symbols = node.get("out_syms") or []
    if not out_symbols:
        out_symbols = [generator.canonical_param_name(node.get("name", "vision_mlp"), "out")]
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
    runtime_input_addr = generator.runtime_symbol_input_address(primary_in_sym)
    return LoweringResult(
        mode="plena_shared",
        comments=[
            f"shared symbol input={primary_in_sym}",
            f"shared output={out_symbols[0]}",
            f"namespace={namespace}",
            (
                f"input_activation={input_binding.address} ({input_binding.source_kind}:{input_binding.producer})"
                if input_binding is not None
                else f"input_hbm={runtime_input_addr if runtime_input_addr is not None else 'auto'}"
            ),
            f"fc1_weight hbm={fc1_weight_binding.hbm_addr}",
            f"fc2_weight hbm={fc2_weight_binding.hbm_addr}",
            *([f"fc1_bias hbm={fc1_bias_binding.hbm_addr}"] if fc1_bias_binding is not None else []),
            *([f"fc2_bias hbm={fc2_bias_binding.hbm_addr}"] if fc2_bias_binding is not None else []),
        ],
    )


DEFAULT_OPERATION_HANDLERS: dict[str, VLMCodegenHandler] = {
    "embedding": embedding_handler,
    "rms_norm": rms_norm_handler,
    "layer_norm": layer_norm_handler,
    "text_attention": text_attention_handler,
    "vision_attention": vision_attention_handler,
    "ffn": ffn_handler,
    "mlp": mlp_handler,
    "linear": linear_handler,
    "conv3d": conv3d_handler,
    "elementwise_add": elementwise_add_handler,
    "vision_mlp_plena": qwen_vision_mlp_handler,
}


def default_operation_handlers() -> dict[str, VLMCodegenHandler]:
    return dict(DEFAULT_OPERATION_HANDLERS)


def register_default_handlers(generator: "VLMAssemblyGenerator") -> None:
    generator.register_handlers(default_operation_handlers())


__all__ = [
    "DEFAULT_OPERATION_HANDLERS",
    "LoweringResult",
    "RuntimeActivationBinding",
    "RuntimeParameterBinding",
    "VLMCodegenHandler",
    "default_operation_handlers",
    "register_default_handlers",
]
