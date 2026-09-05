from typing import Any


def _report_flash_attn_utilization(
    node: dict[str, Any], model_info: dict[str, Any], context_len: int, m: int, n: int, k: int
) -> None:
    """
    Report the utilization of flash attention for a given node.
    """
    dims = node["dimensions"]
    batch_size = model_info["batch_size"]
    hidden_size = dims["hidden_size"]
    num_attn_heads = dims["num_attention_heads"]
    num_kv_heads = dims["num_key_value_heads"]

    head_dim = dims["head_dim"]
    input_token_size = context_len
    theoretical_operation = 0
    attainable_operation = 0
    overall_operation_amount = 0

    # Decoding
    # Projection
    operation_amount = ((head_dim * num_attn_heads) // m) * (hidden_size // k) + ((head_dim * num_kv_heads) // m) * (
        hidden_size // k
    ) * 2
    overall_operation_amount += operation_amount
    attainable_operation += operation_amount * (m * k * batch_size)
    theoretical_operation += operation_amount * (m * k * n)
    # QKT
    operation_amount = batch_size * num_attn_heads * (head_dim // k) * (input_token_size // n)
    overall_operation_amount += operation_amount
    attainable_operation += operation_amount * (m * k)
    theoretical_operation += operation_amount * (m * k * n)

    # PV
    operation_amount = batch_size * num_attn_heads * (input_token_size // k) * (head_dim // n)
    overall_operation_amount += operation_amount
    attainable_operation += operation_amount * (m * k)
    theoretical_operation += operation_amount * (m * k * n)

    return [operation_amount, attainable_operation, theoretical_operation]


def _report_embedding_utilization(node: dict[str, Any], model_info: dict[str, Any], m: int, n: int, k: int) -> None:
    """
    Report the utilization of flash attention for a given node.
    """

    batch_size = model_info["batch_size"]
    hidden_size = model_info["hidden_size"]

    theoretical_operation = 0
    attainable_operation = 0

    # Assuming Decoding only
    operation_amount = (hidden_size // m) * (hidden_size // k)
    attainable_operation += operation_amount * (m * k * batch_size)
    theoretical_operation += operation_amount * (m * k * n)

    return [operation_amount, attainable_operation, theoretical_operation]


def _report_ffn_utilization(node: dict[str, Any], model_info: dict[str, Any], m: int, n: int, k: int) -> None:
    """
    Report the utilization of flash attention for a given node.
    """

    dims = node["dimensions"]
    batch_size = model_info["batch_size"]
    hidden_size = dims["hidden_size"]
    intermediate_size = dims["intermediate_size"]
    overall_operation_amount = 0
    theoretical_operation = 0
    attainable_operation = 0

    # Up Projection
    operation_amount = (intermediate_size // m) * (hidden_size // k)
    overall_operation_amount += operation_amount
    attainable_operation += operation_amount * (m * k * batch_size)
    theoretical_operation += operation_amount * (m * k * n)

    # Gate Projection
    operation_amount = (intermediate_size // m) * (hidden_size // k)
    overall_operation_amount += operation_amount
    attainable_operation += operation_amount * (m * k * batch_size)
    theoretical_operation += operation_amount * (m * k * n)

    # Down Projection
    operation_amount = (hidden_size // m) * (intermediate_size // k)
    overall_operation_amount += operation_amount
    attainable_operation += operation_amount * (m * k * batch_size)
    theoretical_operation += operation_amount * (m * k * n)

    return [overall_operation_amount, attainable_operation, theoretical_operation]


def _report_projection_utilization(node: dict[str, Any], model_info: dict[str, Any], m: int, k: int, n: int) -> list:
    """Utilization of a dense linear layer (Mamba-2 in_proj / out_proj).

    Tile count follows the same shape as one term of ``_report_ffn_utilization``:
    ``(out_features // m) * (in_features // k)``.

    ``attainable`` differs from the attention path's convention on purpose.  That
    one multiplies by ``model_info["batch_size"]`` because it models *decoding*,
    where a single token leaves ``n - 1`` systolic rows idle.  A Mamba-2 prefill
    projects every token in the sequence, so the row count is
    ``batch_size * seq_len`` and the array is fully fed whenever that exceeds
    ``n`` -- capping at ``n`` is what makes attainable meet theoretical here.
    """
    dims = node["dimensions"]
    in_features = dims["in_features"]
    out_features = dims["out_features"]
    rows = model_info.get("batch_size", 1) * model_info.get("seq_len", 1)

    operation_amount = max(1, out_features // m) * max(1, in_features // k)
    attainable = operation_amount * (m * k * min(rows, n))
    theoretical = operation_amount * (m * k * n)
    return [operation_amount, attainable, theoretical]


def _report_ssd_utilization(node: dict[str, Any], model_info: dict[str, Any], m: int, k: int, n: int) -> list:
    """Utilization of the chunked SSD scan's four per-chunk GEMMs.

    Per chunk, with ``Q = chunk_size``, ``P = state_size``, ``D = head_dim``:

    * ``C @ B^T``      -- (Q, P) @ (P, Q), batched over ``n_groups``
    * ``M @ X``        -- (Q, Q) @ (Q, D), batched over ``num_heads``
    * ``B^T @ (X*dt)`` -- (P, Q) @ (Q, D), batched over ``num_heads``
    * ``C @ S``        -- (Q, P) @ (P, D), batched over ``num_heads``

    The elementwise stages (dt softplus, the decay cumsum, the decay mask, the D
    skip) run on the vector unit and contribute nothing to systolic utilization,
    the same way ``normalization`` and ``elementwise_add`` nodes already report
    zero here.  They are not free -- the cumsum in particular is a serial chain
    of ``chunk_size`` dependent vector ops -- so a systolic-only report
    understates the real cost of this kernel.
    """
    dims = node["dimensions"]
    num_heads = dims["num_heads"]
    head_dim = dims["head_dim"]
    state_size = dims["state_size"]
    n_groups = dims["n_groups"]
    chunk = dims["chunk_size"]
    num_chunks = dims["num_chunks"]

    per_chunk = (
        n_groups * max(1, chunk // m) * max(1, state_size // k)  # C @ B^T
        + num_heads * max(1, head_dim // m) * max(1, chunk // k)  # M @ X
        + num_heads * max(1, head_dim // m) * max(1, chunk // k)  # B^T @ (X*dt)
        + num_heads * max(1, head_dim // m) * max(1, state_size // k)  # C @ S
    )
    operation_amount = num_chunks * per_chunk
    # Rows fed per GEMM are chunk-many (or state-many for the state update);
    # chunk_size >= MLEN by construction, so these tiles are always full.
    attainable = operation_amount * (m * k * min(chunk, n))
    theoretical = operation_amount * (m * k * n)
    return [operation_amount, attainable, theoretical]


def _report_utilization(node: dict[str, Any], model_info: dict[str, Any], m: int, k: int, n: int) -> str:
    """Generate assembly code for a single symbolic graph node."""
    operation_type = node["operation_type"]

    if operation_type == "embedding":
        return _report_embedding_utilization(node, model_info, m, k, n)
    elif operation_type == "attention":
        return _report_flash_attn_utilization(node, model_info, 1024, m, k, n)
    elif operation_type == "ffn":
        return _report_ffn_utilization(node, model_info, m, k, n)
    elif operation_type == "projection":
        return _report_projection_utilization(node, model_info, m, k, n)
    elif operation_type == "ssd_scan":
        return _report_ssd_utilization(node, model_info, m, k, n)
    else:
        return [0, 0, 0]


def _report_lm_head_utilization(model_info: dict[str, Any], m: int, k: int, n: int) -> str:
    """
    Report the utilization of LM head for a given node.
    """
    batch_size = model_info["batch_size"]
    vocab_size = model_info.get("vocab_size", 128256)
    hidden_size = model_info["hidden_size"]

    theoretical_operation = 0
    attainable_operation = 0

    # Assuming Decoding only
    operation_amount = (vocab_size // m) * (hidden_size // k)
    attainable_operation += operation_amount * (m * k * batch_size)
    theoretical_operation += operation_amount * (m * k * n)

    return [operation_amount, attainable_operation, theoretical_operation]


def analyse_overall_utilization(
    symbolic_graph: dict[str, Any], model_info: dict[str, Any], m: int, k: int, n: int
) -> str:
    """
    Transform the complete symbolic graph into assembly code.

    Args:
        symbolic_graph: The symbolic graph from LLMModelParser
        model_info: Model metadata for header generation

    Returns:
        Complete assembly program as string
    """
    # Process each node in execution order
    nodes = symbolic_graph["nodes"]
    execution_order = symbolic_graph["execution_order"]

    # Create a mapping from node names to nodes for efficient lookup
    node_map = {node["name"]: node for node in nodes}

    # "projection" and "ssd_scan" are the Mamba-2 systolic kernels; they stay 0
    # for attention models, which never emit those node types.
    accounted = ["embedding", "attention", "ffn", "projection", "ssd_scan"]
    overall_operations = dict.fromkeys([*accounted, "lm_head"], 0)
    overall_attainable_flops = dict.fromkeys([*accounted, "lm_head"], 0)
    overall_theoretical_flops = dict.fromkeys([*accounted, "lm_head"], 0)

    # Generate code for each node in execution order
    for node_name in execution_order:
        if node_name in node_map:
            node = node_map[node_name]
            single_op_operation = _report_utilization(node, model_info, m, k, n)
            operation_type = node["operation_type"]
            if operation_type in accounted:
                overall_operations[operation_type] += single_op_operation[0]
                overall_attainable_flops[operation_type] += single_op_operation[1]
                overall_theoretical_flops[operation_type] += single_op_operation[2]

    single_op_operation = _report_lm_head_utilization(model_info, m, k, n)
    overall_operations["lm_head"] += single_op_operation[0]
    overall_attainable_flops["lm_head"] += single_op_operation[1]
    overall_theoretical_flops["lm_head"] += single_op_operation[2]

    return {
        "operations": overall_operations,
        "attainable_FLOPS": overall_attainable_flops,
        "theoretical_FLOPS": overall_theoretical_flops,
    }
