import math
from typing import Any

import torch
from transformers import AutoConfig, AutoModel

# Architecture strings that select the Mamba-2 (selective state-space) lowering.
# ``model_type == "mamba2"`` is the primary signal; the architecture string is
# checked too so a config that omits model_type still routes correctly.
MAMBA2_ARCHITECTURES = frozenset({"Mamba2ForCausalLM", "Mamba2Model"})


class LLMModelParser:
    def __init__(self, model_name_or_path: str):
        self.model_name_or_path = model_name_or_path
        self.config = None
        self.model = None
        self.symbolic_graph = None

    def load_model(self):
        """Load the model and configuration from HuggingFace"""
        try:
            self.config = AutoConfig.from_pretrained(self.model_name_or_path)
            self.model = AutoModel.from_pretrained(self.model_name_or_path, torch_dtype=torch.float32)
            self.model.eval()
        except Exception as e:
            raise ValueError(f"Failed to load model {self.model_name_or_path}: {e}")

    def _resolve_text_config(self):
        """For multimodal models, return text decoder sub-config."""
        if hasattr(self.config, "text_config") and self.config.text_config is not None:
            return self.config.text_config
        return self.config

    def is_mamba2(self) -> bool:
        """True when this config selects the Mamba-2 selective-SSM lowering.

        Checks ``model_type`` first (the field HuggingFace's ``Mamba2Config``
        pins to ``"mamba2"``) and falls back to the ``architectures`` list so a
        hand-written config without a model_type still routes correctly.
        """
        if self.config is None:
            self.load_model()
        text_cfg = self._resolve_text_config()
        if str(getattr(text_cfg, "model_type", "") or "").lower() == "mamba2":
            return True
        architectures = getattr(self.config, "architectures", None) or []
        return any(arch in MAMBA2_ARCHITECTURES for arch in architectures)

    def _extract_mamba2_dimensions(self) -> dict[str, Any]:
        """Resolve the Mamba-2 mixer shape from a ``Mamba2Config``.

        Field names mirror HuggingFace's ``Mamba2Config`` so a config JSON maps
        across without a translation table.  Derived quantities:

        * ``d_inner   = expand * hidden_size``   -- the mixer's inner width
        * ``conv_dim  = d_inner + 2 * n_groups * state_size``
                      -- the depthwise conv runs over ``[x, B, C]``
        * ``in_proj_out = 2 * d_inner + 2 * n_groups * state_size + num_heads``
                      -- the fused in_proj emits ``[z, x, B, C, dt]``
        """
        text_cfg = self._resolve_text_config()

        hidden_size = getattr(text_cfg, "hidden_size")
        expand = getattr(text_cfg, "expand", 2)
        head_dim = getattr(text_cfg, "head_dim", 64)
        state_size = getattr(text_cfg, "state_size", 128)
        n_groups = getattr(text_cfg, "n_groups", 1)
        conv_kernel = getattr(text_cfg, "conv_kernel", 4)
        chunk_size = getattr(text_cfg, "chunk_size", 256)

        d_inner = expand * hidden_size
        num_heads = getattr(text_cfg, "num_heads", None)
        if num_heads is None:
            num_heads = d_inner // head_dim
        # A mismatch here silently corrupts every downstream dimension (the dt
        # slice width, the SSD batch axis, the gated-norm group count), so fail
        # loudly instead of trusting whichever field happens to be wrong.
        if num_heads * head_dim != d_inner:
            raise ValueError(
                f"Mamba-2 config is inconsistent: num_heads ({num_heads}) * head_dim "
                f"({head_dim}) = {num_heads * head_dim}, but expand ({expand}) * hidden_size "
                f"({hidden_size}) = {d_inner}.  num_heads must equal "
                "expand * hidden_size / head_dim."
            )
        if num_heads % n_groups:
            raise ValueError(
                f"Mamba-2 config is inconsistent: num_heads ({num_heads}) is not divisible by "
                f"n_groups ({n_groups}); B and C are shared across the heads of a group."
            )

        time_step_limit = getattr(text_cfg, "time_step_limit", (0.0, float("inf")))
        try:
            time_step_min, time_step_max = float(time_step_limit[0]), float(time_step_limit[1])
        except (TypeError, IndexError, ValueError):
            time_step_min, time_step_max = 0.0, float("inf")

        return {
            "hidden_size": hidden_size,
            "expand": expand,
            "d_inner": d_inner,
            "state_size": state_size,
            "n_groups": n_groups,
            "conv_kernel": conv_kernel,
            "head_dim": head_dim,
            "num_heads": num_heads,
            "heads_per_group": num_heads // n_groups,
            "chunk_size": chunk_size,
            "conv_dim": d_inner + 2 * n_groups * state_size,
            "in_proj_out": 2 * d_inner + 2 * n_groups * state_size + num_heads,
            "use_conv_bias": getattr(text_cfg, "use_conv_bias", True),
            "use_bias": getattr(text_cfg, "use_bias", False),
            "rms_norm": getattr(text_cfg, "rms_norm", True),
            "activation": getattr(text_cfg, "hidden_act", "silu"),
            "eps": getattr(text_cfg, "layer_norm_epsilon", getattr(text_cfg, "rms_norm_eps", 1e-5)),
            "time_step_min": time_step_min,
            "time_step_max": time_step_max,
        }

    def _mamba2_shape(self, batch_size: int, seq_len: int) -> dict[str, Any]:
        """Per-layer Mamba-2 shape stamped onto every mixer node.

        Carried on each node (rather than looked up from a global) so code_gen
        can derive a single consistent VRAM map from any one of them without
        needing the whole graph.
        """
        mamba = self._extract_mamba2_dimensions()
        shape = {
            k: mamba[k]
            for k in (
                "hidden_size",
                "d_inner",
                "conv_dim",
                "in_proj_out",
                "state_size",
                "n_groups",
                "conv_kernel",
                "head_dim",
                "num_heads",
                "heads_per_group",
                "chunk_size",
            )
        }
        shape["seq_len"] = seq_len
        shape["batch_size"] = batch_size
        shape["num_chunks"] = math.ceil(seq_len / mamba["chunk_size"]) if mamba["chunk_size"] else 1
        return shape

    def extract_critical_dimensions(self) -> dict[str, Any]:
        """Extract dimensions for attention, RMSNorm, FFN operations"""
        if self.config is None:
            self.load_model()

        text_cfg = self._resolve_text_config()
        dimensions = {}

        # Common dimensions
        dimensions["vocab_size"] = getattr(text_cfg, "vocab_size", None)
        dimensions["hidden_size"] = getattr(text_cfg, "hidden_size", None)
        dimensions["num_hidden_layers"] = getattr(text_cfg, "num_hidden_layers", None)
        dimensions["max_position_embeddings"] = getattr(text_cfg, "max_position_embeddings", None)

        # Attention dimensions
        dimensions["attention"] = self._extract_attention_dimensions()

        # FFN dimensions
        dimensions["ffn"] = self._extract_ffn_dimensions()

        # RMSNorm dimensions
        dimensions["rms_norm"] = self._extract_rms_norm_dimensions()

        # Mamba-2 mixer dimensions.  A Mamba-2 config has no attention and no
        # gated FFN, so the ``attention`` / ``ffn`` sections above are vacuous
        # for it; ``mamba`` is the authoritative section.
        if self.is_mamba2():
            dimensions["mamba"] = self._extract_mamba2_dimensions()

        # Include vision encoder dimensions if present
        if hasattr(self.config, "vision_config") and self.config.vision_config is not None:
            vcfg = self.config.vision_config
            vhidden = getattr(vcfg, "hidden_size", None)
            vheads = getattr(vcfg, "num_attention_heads", 1)
            dimensions["vision"] = {
                "hidden_size": vhidden,
                "num_hidden_layers": getattr(vcfg, "num_hidden_layers", None),
                "num_attention_heads": vheads,
                "intermediate_size": getattr(vcfg, "intermediate_size", None),
                "head_dim": getattr(vcfg, "head_dim", (vhidden // vheads) if vhidden and vheads else None),
                "image_size": getattr(vcfg, "image_size", None),
                "patch_size": getattr(vcfg, "patch_size", None),
            }

        return dimensions

    def _extract_attention_dimensions(self) -> dict[str, Any]:
        """Extract attention-specific dimensions"""
        text_cfg = self._resolve_text_config()
        attention_dims = {}

        # Multi-head attention parameters
        attention_dims["num_attention_heads"] = getattr(text_cfg, "num_attention_heads", None)
        attention_dims["num_key_value_heads"] = getattr(
            text_cfg, "num_key_value_heads", getattr(text_cfg, "num_attention_heads", None)
        )

        hidden_size = getattr(text_cfg, "hidden_size", 0)
        num_heads = getattr(text_cfg, "num_attention_heads", 1)
        num_kv_heads = getattr(text_cfg, "num_key_value_heads", num_heads)

        if hidden_size and num_heads:
            # Use explicit head_dim if available (e.g. SmolVLM2 has head_dim=64)
            attention_dims["head_dim"] = getattr(text_cfg, "head_dim", hidden_size // num_heads)
            attention_dims["key_value_head_dim"] = num_kv_heads * attention_dims["head_dim"]

        return attention_dims

    def _extract_ffn_dimensions(self) -> dict[str, Any]:
        """Extract FFN (Feed-Forward Network) dimensions"""
        text_cfg = self._resolve_text_config()
        ffn_dims = {}

        hidden_size = getattr(text_cfg, "hidden_size", 0)
        intermediate_size = getattr(text_cfg, "intermediate_size", hidden_size * 4)

        ffn_dims["hidden_size"] = hidden_size
        ffn_dims["intermediate_size"] = intermediate_size
        ffn_dims["activation"] = getattr(text_cfg, "hidden_act", "silu")

        return ffn_dims

    def _extract_rms_norm_dimensions(self) -> dict[str, Any]:
        """Extract RMSNorm dimensions"""
        text_cfg = self._resolve_text_config()
        rms_dims = {}

        hidden_size = getattr(text_cfg, "hidden_size", 0)

        rms_dims["normalized_shape"] = hidden_size
        rms_dims["eps"] = getattr(text_cfg, "rms_norm_eps", 1e-6)

        return rms_dims

    def create_symbolic_graph(self, batch_size: int = 1, seq_len: int = 512) -> dict[str, Any]:
        """Create a symbolic graph with execution orders"""
        # TODO: this is in fixed ordering and thus would only support only LlamaForCausalLM architecture such as AICrossSim/clm-60m that we know the detail
        # TODO: Additional work is needed to make it more flexible (maybe use MASEGraph or torch.fx)
        if self.config is None:
            self.load_model()

        # Mamba-2 replaces the whole attention + FFN sublayer pair with a single
        # SSM mixer, so it gets its own builder rather than a set of branches
        # threaded through the Llama-shaped one below.
        if self.is_mamba2():
            return self._create_mamba2_symbolic_graph(batch_size=batch_size, seq_len=seq_len)

        text_cfg = self._resolve_text_config()

        # Compute GQA-aware projection dimensions
        hidden_size = text_cfg.hidden_size
        num_attention_heads = getattr(text_cfg, "num_attention_heads", 1)
        num_key_value_heads = getattr(text_cfg, "num_key_value_heads", num_attention_heads)
        head_dim = getattr(text_cfg, "head_dim", hidden_size // num_attention_heads)
        kv_dim = num_key_value_heads * head_dim

        symbolic_nodes = []
        execution_order = []
        order_counter = 0

        # Start with input embedding
        # Include embed_tokens when config has a vocabulary; also check model for nested VLM architectures
        if getattr(text_cfg, "vocab_size", None) is not None:
            embed_info = {
                "name": "embed_tokens",
                "operation_type": "embedding",
                "operation_category": "embedding",
                "execution_order": order_counter,
                "input_shape": [batch_size, seq_len],  # input_ids shape
                "output_shape": [batch_size, seq_len, hidden_size],  # embedded tokens
                "dimensions": {"num_embeddings": getattr(text_cfg, "vocab_size", None), "hidden_size": hidden_size},
                "is_data_placeholder": True,
            }
            symbolic_nodes.append(embed_info)
            execution_order.append("embed_tokens")
            order_counter += 1

        # Process transformer layers
        num_layers = getattr(text_cfg, "num_hidden_layers", 0)

        for layer_idx in range(num_layers):
            current_shape = [batch_size, seq_len, hidden_size]

            # Input layer norm
            norm_info = {
                "name": f"layer_{layer_idx}_input_layernorm",
                "operation_type": "normalization",
                "operation_category": "normalization",
                "execution_order": order_counter,
                "input_shape": current_shape,
                "output_shape": current_shape,  # normalization preserves shape
                "dimensions": {
                    "normalized_shape": hidden_size,
                    "eps": getattr(text_cfg, "rms_norm_eps", 1e-6),
                },
                "is_data_placeholder": False,
            }
            symbolic_nodes.append(norm_info)
            execution_order.append(f"layer_{layer_idx}_input_layernorm")
            order_counter += 1

            # Self-attention block (fused)
            attn_info = {
                "name": f"layer_{layer_idx}_self_attn",
                "operation_type": "attention",
                "operation_category": "attention",
                "execution_order": order_counter,
                "input_shape": current_shape,
                "output_shape": current_shape,  # attention preserves shape
                "dimensions": {
                    "hidden_size": hidden_size,
                    "num_attention_heads": num_attention_heads,
                    "num_key_value_heads": num_key_value_heads,
                    "head_dim": head_dim,
                    "q_proj": {"in_features": hidden_size, "out_features": num_attention_heads * head_dim},
                    "k_proj": {"in_features": hidden_size, "out_features": kv_dim},
                    "v_proj": {"in_features": hidden_size, "out_features": kv_dim},
                    "o_proj": {"in_features": num_attention_heads * head_dim, "out_features": hidden_size},
                },
                "is_data_placeholder": False,
            }
            symbolic_nodes.append(attn_info)
            execution_order.append(f"layer_{layer_idx}_self_attn")
            order_counter += 1

            # Residual connection (attention)
            residual_info = {
                "name": f"layer_{layer_idx}_attn_residual",
                "operation_type": "elementwise_add",
                "operation_category": "elementwise_add",
                "execution_order": order_counter,
                "input_shape": [current_shape, current_shape],  # two inputs of same shape
                "output_shape": current_shape,  # elementwise add preserves shape
                "dimensions": {"shape": [hidden_size]},
                "is_data_placeholder": False,
            }
            symbolic_nodes.append(residual_info)
            execution_order.append(f"layer_{layer_idx}_attn_residual")
            order_counter += 1

            # Post-attention layer norm
            post_norm_info = {
                "name": f"layer_{layer_idx}_post_attention_layernorm",
                "operation_type": "normalization",
                "operation_category": "normalization",
                "execution_order": order_counter,
                "input_shape": current_shape,
                "output_shape": current_shape,  # normalization preserves shape
                "dimensions": {
                    "normalized_shape": hidden_size,
                    "eps": getattr(text_cfg, "rms_norm_eps", 1e-6),
                },
                "is_data_placeholder": False,
            }
            symbolic_nodes.append(post_norm_info)
            execution_order.append(f"layer_{layer_idx}_post_attention_layernorm")
            order_counter += 1

            # MLP/FFN block (fused)
            intermediate_size = getattr(text_cfg, "intermediate_size", hidden_size * 4)
            mlp_info = {
                "name": f"layer_{layer_idx}_mlp",
                "operation_type": "ffn",
                "operation_category": "ffn",
                "execution_order": order_counter,
                "input_shape": current_shape,
                "output_shape": current_shape,  # FFN preserves shape
                "dimensions": {
                    "hidden_size": hidden_size,
                    "intermediate_size": intermediate_size,
                    "activation": getattr(text_cfg, "hidden_act", "silu"),
                    "gate_proj": {
                        "in_features": hidden_size,
                        "out_features": intermediate_size,
                    },
                    "up_proj": {
                        "in_features": hidden_size,
                        "out_features": intermediate_size,
                    },
                    "down_proj": {
                        "in_features": intermediate_size,
                        "out_features": hidden_size,
                    },
                },
                "is_data_placeholder": False,
            }
            symbolic_nodes.append(mlp_info)
            execution_order.append(f"layer_{layer_idx}_mlp")
            order_counter += 1

            # Residual connection (FFN)
            ffn_residual_info = {
                "name": f"layer_{layer_idx}_ffn_residual",
                "operation_type": "elementwise_add",
                "operation_category": "elementwise_add",
                "execution_order": order_counter,
                "input_shape": [current_shape, current_shape],  # two inputs of same shape
                "output_shape": current_shape,  # elementwise add preserves shape
                "dimensions": {"shape": [hidden_size]},
                "is_data_placeholder": False,
            }
            symbolic_nodes.append(ffn_residual_info)
            execution_order.append(f"layer_{layer_idx}_ffn_residual")
            order_counter += 1

        # Final layer norm
        final_shape = [batch_size, seq_len, hidden_size]
        final_norm_info = {
            "name": "final_layernorm",
            "operation_type": "normalization",
            "operation_category": "normalization",
            "execution_order": order_counter,
            "input_shape": final_shape,
            "output_shape": final_shape,  # normalization preserves shape
            "dimensions": {
                "normalized_shape": hidden_size,
                "eps": getattr(text_cfg, "rms_norm_eps", 1e-6),
            },
            "is_data_placeholder": False,
        }
        symbolic_nodes.append(final_norm_info)
        execution_order.append("final_layernorm")
        order_counter += 1

        # LM head: final hidden→vocab_size projection
        vocab_size = getattr(text_cfg, "vocab_size", None)
        if vocab_size is not None:
            lm_head_info = {
                "name": "lm_head",
                "operation_type": "lm_head",
                "operation_category": "lm_head",
                "execution_order": order_counter,
                "input_shape": [batch_size, seq_len, hidden_size],
                "output_shape": [batch_size, seq_len, vocab_size],
                "dimensions": {
                    "hidden_size": hidden_size,
                    "vocab_size": vocab_size,
                },
                "is_data_placeholder": False,
            }
            symbolic_nodes.append(lm_head_info)
            execution_order.append("lm_head")
            order_counter += 1

        self.symbolic_graph = {
            "nodes": symbolic_nodes,
            "execution_order": execution_order,
            "total_nodes": len(symbolic_nodes),
        }

        return self.symbolic_graph

    def _create_mamba2_symbolic_graph(self, batch_size: int = 1, seq_len: int = 512) -> dict[str, Any]:
        """Build the symbolic graph for a Mamba-2 (``Mamba2ForCausalLM``) model.

        Per layer the mixer is::

            norm -> in_proj -> conv1d -> ssd_scan -> gated_rmsnorm -> out_proj -> residual

        which is the HuggingFace ``Mamba2Mixer`` forward, one node per stage that
        maps to a distinct PLENA kernel:

        * ``in_proj``  emits the fused ``[z, x, B, C, dt]`` bundle in one GEMM,
          exactly as the reference implementation does -- splitting it would cost
          five weight prefetches instead of one.
        * ``conv1d``   is depthwise and causal over ``[x, B, C]`` only (``z`` and
          ``dt`` bypass it), hence ``conv_dim`` channels rather than
          ``in_proj_out``.  Its SiLU is carried on the node as ``activation``
          instead of a separate node because the two always fuse.
        * ``ssd_scan`` is the chunked state-space-duality recurrence.  It is one
          node, not a chunk-loop of nodes, because ``chunk_size`` is a lowering
          parameter of a single kernel, not a graph-level structure.
        * ``gated_rmsnorm`` normalises over ``d_inner / n_groups`` and multiplies
          by ``silu(z)``; it cannot be folded into the plain ``normalization``
          node because that one reduces over ``hidden_size``.

        There is a single residual per layer (Mamba-2 has one sublayer, not the
        attention + FFN pair), so a layer is 7 nodes rather than Llama's 6.
        """
        text_cfg = self._resolve_text_config()
        mamba = self._extract_mamba2_dimensions()
        shape = self._mamba2_shape(batch_size=batch_size, seq_len=seq_len)

        hidden_size = mamba["hidden_size"]
        d_inner = mamba["d_inner"]
        conv_dim = mamba["conv_dim"]
        in_proj_out = mamba["in_proj_out"]
        n_groups = mamba["n_groups"]
        state_size = mamba["state_size"]
        num_heads = mamba["num_heads"]
        eps = mamba["eps"]

        symbolic_nodes: list[dict[str, Any]] = []
        execution_order: list[str] = []
        order_counter = 0

        def add(node: dict[str, Any]) -> None:
            nonlocal order_counter
            node["execution_order"] = order_counter
            symbolic_nodes.append(node)
            execution_order.append(node["name"])
            order_counter += 1

        current_shape = [batch_size, seq_len, hidden_size]

        vocab_size = getattr(text_cfg, "vocab_size", None)
        if vocab_size is not None:
            add(
                {
                    "name": "embed_tokens",
                    "operation_type": "embedding",
                    "operation_category": "embedding",
                    "input_shape": [batch_size, seq_len],
                    "output_shape": current_shape,
                    "dimensions": {"num_embeddings": vocab_size, "hidden_size": hidden_size},
                    "is_data_placeholder": True,
                }
            )

        # Column offsets of each slice inside the fused in_proj output, in the
        # order Mamba2Mixer concatenates them.
        z_off = 0
        x_off = z_off + d_inner
        b_off = x_off + d_inner
        c_off = b_off + n_groups * state_size
        dt_off = c_off + n_groups * state_size
        in_proj_slices = {
            "z": [z_off, d_inner],
            "x": [x_off, d_inner],
            "B": [b_off, n_groups * state_size],
            "C": [c_off, n_groups * state_size],
            "dt": [dt_off, num_heads],
        }

        num_layers = getattr(text_cfg, "num_hidden_layers", 0)
        for layer_idx in range(num_layers):
            add(
                {
                    "name": f"layer_{layer_idx}_input_layernorm",
                    "operation_type": "normalization",
                    "operation_category": "normalization",
                    "input_shape": current_shape,
                    "output_shape": current_shape,
                    "dimensions": {"normalized_shape": hidden_size, "eps": eps},
                    "is_data_placeholder": False,
                }
            )

            add(
                {
                    "name": f"layer_{layer_idx}_in_proj",
                    "operation_type": "projection",
                    "operation_category": "projection",
                    "input_shape": current_shape,
                    "output_shape": [batch_size, seq_len, in_proj_out],
                    "dimensions": {
                        "role": "mamba_in_proj",
                        "in_features": hidden_size,
                        "out_features": in_proj_out,
                        "use_bias": mamba["use_bias"],
                        "slices": in_proj_slices,
                        "mamba_shape": shape,
                    },
                    "is_data_placeholder": False,
                }
            )

            add(
                {
                    "name": f"layer_{layer_idx}_conv1d",
                    "operation_type": "conv1d",
                    "operation_category": "conv1d",
                    "input_shape": [batch_size, seq_len, conv_dim],
                    "output_shape": [batch_size, seq_len, conv_dim],
                    "dimensions": {
                        "in_channels": conv_dim,
                        "out_channels": conv_dim,
                        "groups": conv_dim,  # depthwise: one group per channel
                        "conv_dim": conv_dim,
                        "kernel_size": mamba["conv_kernel"],
                        "stride": 1,
                        "padding": mamba["conv_kernel"] - 1,
                        "causal": True,
                        "depthwise": True,
                        "use_conv_bias": mamba["use_conv_bias"],
                        "activation": mamba["activation"],
                        "seq_len": seq_len,
                        "mamba_shape": shape,
                    },
                    "is_data_placeholder": False,
                }
            )

            add(
                {
                    "name": f"layer_{layer_idx}_ssd_scan",
                    "operation_type": "ssd_scan",
                    "operation_category": "ssd_scan",
                    "input_shape": [batch_size, seq_len, conv_dim],
                    "output_shape": [batch_size, seq_len, d_inner],
                    "dimensions": {
                        "d_inner": d_inner,
                        "num_heads": num_heads,
                        "head_dim": mamba["head_dim"],
                        "state_size": state_size,
                        "n_groups": n_groups,
                        "heads_per_group": mamba["heads_per_group"],
                        "chunk_size": mamba["chunk_size"],
                        "num_chunks": shape["num_chunks"],
                        "seq_len": seq_len,
                        "time_step_min": mamba["time_step_min"],
                        "time_step_max": mamba["time_step_max"],
                        "mamba_shape": shape,
                    },
                    "is_data_placeholder": False,
                }
            )

            add(
                {
                    "name": f"layer_{layer_idx}_gated_rmsnorm",
                    "operation_type": "gated_rmsnorm",
                    "operation_category": "normalization",
                    "input_shape": [[batch_size, seq_len, d_inner], [batch_size, seq_len, d_inner]],
                    "output_shape": [batch_size, seq_len, d_inner],
                    "dimensions": {
                        "normalized_shape": d_inner,
                        # Mamba-2 normalises per group, not over the full width.
                        "group_size": d_inner // n_groups,
                        "n_groups": n_groups,
                        "eps": eps,
                        "norm_type": "gated_rms_norm",
                        "gate_activation": mamba["activation"],
                        "mamba_shape": shape,
                    },
                    "is_data_placeholder": False,
                }
            )

            add(
                {
                    "name": f"layer_{layer_idx}_out_proj",
                    "operation_type": "projection",
                    "operation_category": "projection",
                    "input_shape": [batch_size, seq_len, d_inner],
                    "output_shape": current_shape,
                    "dimensions": {
                        "role": "mamba_out_proj",
                        "in_features": d_inner,
                        "out_features": hidden_size,
                        "use_bias": mamba["use_bias"],
                        "mamba_shape": shape,
                    },
                    "is_data_placeholder": False,
                }
            )

            add(
                {
                    "name": f"layer_{layer_idx}_residual",
                    "operation_type": "elementwise_add",
                    "operation_category": "elementwise_add",
                    "input_shape": [current_shape, current_shape],
                    "output_shape": current_shape,
                    "dimensions": {"shape": [hidden_size]},
                    "is_data_placeholder": False,
                }
            )

        add(
            {
                "name": "final_layernorm",
                "operation_type": "normalization",
                "operation_category": "normalization",
                "input_shape": current_shape,
                "output_shape": current_shape,
                "dimensions": {"normalized_shape": hidden_size, "eps": eps},
                "is_data_placeholder": False,
            }
        )

        if vocab_size is not None:
            add(
                {
                    "name": "lm_head",
                    "operation_type": "lm_head",
                    "operation_category": "lm_head",
                    "input_shape": current_shape,
                    "output_shape": [batch_size, seq_len, vocab_size],
                    "dimensions": {"hidden_size": hidden_size, "vocab_size": vocab_size},
                    "is_data_placeholder": False,
                }
            )

        self.symbolic_graph = {
            "nodes": symbolic_nodes,
            "execution_order": execution_order,
            "total_nodes": len(symbolic_nodes),
            "architecture_family": "mamba2",
            "mamba_shape": shape,
        }
        return self.symbolic_graph

    def create_vision_symbolic_graph(self, batch_size: int = 1) -> dict | None:
        """Create symbolic graph for vision encoder (SigLIP/ViT style).
        Returns None if no vision_config present.
        """
        if self.config is None:
            self.load_model()
        if not hasattr(self.config, "vision_config") or self.config.vision_config is None:
            return None

        vcfg = self.config.vision_config
        image_size = getattr(vcfg, "image_size", 224)
        patch_size = getattr(vcfg, "patch_size", 16)
        num_patches = (image_size // patch_size) ** 2
        num_channels = getattr(vcfg, "num_channels", 3)
        hidden_size = getattr(vcfg, "hidden_size", 768)
        num_layers = getattr(vcfg, "num_hidden_layers", 12)
        num_heads = getattr(vcfg, "num_attention_heads", 12)
        intermediate_size = getattr(vcfg, "intermediate_size", hidden_size * 4)
        head_dim = getattr(vcfg, "head_dim", hidden_size // num_heads)
        norm_eps = getattr(vcfg, "layer_norm_eps", getattr(vcfg, "norm_eps", 1e-6))
        hidden_act = getattr(vcfg, "hidden_act", "gelu")

        symbolic_nodes = []
        execution_order = []
        order_counter = 0
        current_shape = [batch_size, num_patches, hidden_size]

        # Patch embedding: Conv2d(num_channels, hidden_size, kernel=patch_size, stride=patch_size)
        # Implemented as im2col -> matmul on PLENA.  Emits a dedicated conv2d node so
        # code_gen can wrap the im2col + projection template pair.
        patch_embed = {
            "name": "vision_patch_embed",
            "operation_type": "conv2d",
            "operation_category": "conv2d",
            "execution_order": order_counter,
            "input_shape": [batch_size, num_channels, image_size, image_size],
            "output_shape": current_shape,
            "dimensions": {
                "in_channels": num_channels,
                "out_channels": hidden_size,
                "image_size": image_size,
                "patch_size": patch_size,
                "kernel_size": patch_size,
                "stride": patch_size,
                "num_patches": num_patches,
                "hidden_size": hidden_size,
            },
            "is_data_placeholder": True,
        }
        symbolic_nodes.append(patch_embed)
        execution_order.append("vision_patch_embed")
        order_counter += 1

        # ViT transformer layers (pre-norm architecture)
        for layer_idx in range(num_layers):
            # Pre-attention layernorm
            pre_attn_norm = {
                "name": f"vision_layer_{layer_idx}_pre_attn_norm",
                "operation_type": "normalization",
                "operation_category": "normalization",
                "execution_order": order_counter,
                "input_shape": current_shape,
                "output_shape": current_shape,
                "dimensions": {
                    "normalized_shape": hidden_size,
                    "eps": norm_eps,
                    "norm_type": "layer_norm",
                },
                "is_data_placeholder": False,
            }
            symbolic_nodes.append(pre_attn_norm)
            execution_order.append(f"vision_layer_{layer_idx}_pre_attn_norm")
            order_counter += 1

            # Self-attention (ViT has no GQA: num_kv_heads == num_heads, and SigLIP is bidirectional)
            attn_info = {
                "name": f"vision_layer_{layer_idx}_self_attn",
                "operation_type": "attention",
                "operation_category": "attention",
                "execution_order": order_counter,
                "input_shape": current_shape,
                "output_shape": current_shape,
                "dimensions": {
                    "hidden_size": hidden_size,
                    "num_attention_heads": num_heads,
                    "num_key_value_heads": num_heads,
                    "head_dim": head_dim,
                    "causal_mask": False,  # SigLIP / ViT uses bidirectional attention
                    "q_proj": {"in_features": hidden_size, "out_features": num_heads * head_dim},
                    "k_proj": {"in_features": hidden_size, "out_features": num_heads * head_dim},
                    "v_proj": {"in_features": hidden_size, "out_features": num_heads * head_dim},
                    "o_proj": {"in_features": num_heads * head_dim, "out_features": hidden_size},
                },
                "is_data_placeholder": False,
            }
            symbolic_nodes.append(attn_info)
            execution_order.append(f"vision_layer_{layer_idx}_self_attn")
            order_counter += 1

            # Attention residual
            attn_residual = {
                "name": f"vision_layer_{layer_idx}_attn_residual",
                "operation_type": "elementwise_add",
                "operation_category": "elementwise_add",
                "execution_order": order_counter,
                "input_shape": [current_shape, current_shape],
                "output_shape": current_shape,
                "dimensions": {"shape": [hidden_size]},
                "is_data_placeholder": False,
            }
            symbolic_nodes.append(attn_residual)
            execution_order.append(f"vision_layer_{layer_idx}_attn_residual")
            order_counter += 1

            # Pre-FFN layernorm
            pre_ffn_norm = {
                "name": f"vision_layer_{layer_idx}_pre_ffn_norm",
                "operation_type": "normalization",
                "operation_category": "normalization",
                "execution_order": order_counter,
                "input_shape": current_shape,
                "output_shape": current_shape,
                "dimensions": {
                    "normalized_shape": hidden_size,
                    "eps": norm_eps,
                    "norm_type": "layer_norm",
                },
                "is_data_placeholder": False,
            }
            symbolic_nodes.append(pre_ffn_norm)
            execution_order.append(f"vision_layer_{layer_idx}_pre_ffn_norm")
            order_counter += 1

            # FFN (ViT-style: fc1 -> activation -> fc2, no gate projection)
            mlp_info = {
                "name": f"vision_layer_{layer_idx}_mlp",
                "operation_type": "ffn",
                "operation_category": "ffn",
                "execution_order": order_counter,
                "input_shape": current_shape,
                "output_shape": current_shape,
                "dimensions": {
                    "hidden_size": hidden_size,
                    "intermediate_size": intermediate_size,
                    "activation": hidden_act,
                    "arch": "vit",  # no gate projection — two-linear FFN
                    "fc1": {"in_features": hidden_size, "out_features": intermediate_size},
                    "fc2": {"in_features": intermediate_size, "out_features": hidden_size},
                },
                "is_data_placeholder": False,
            }
            symbolic_nodes.append(mlp_info)
            execution_order.append(f"vision_layer_{layer_idx}_mlp")
            order_counter += 1

            # FFN residual
            ffn_residual = {
                "name": f"vision_layer_{layer_idx}_ffn_residual",
                "operation_type": "elementwise_add",
                "operation_category": "elementwise_add",
                "execution_order": order_counter,
                "input_shape": [current_shape, current_shape],
                "output_shape": current_shape,
                "dimensions": {"shape": [hidden_size]},
                "is_data_placeholder": False,
            }
            symbolic_nodes.append(ffn_residual)
            execution_order.append(f"vision_layer_{layer_idx}_ffn_residual")
            order_counter += 1

        # Final layernorm
        final_norm = {
            "name": "vision_final_norm",
            "operation_type": "normalization",
            "operation_category": "normalization",
            "execution_order": order_counter,
            "input_shape": current_shape,
            "output_shape": current_shape,
            "dimensions": {
                "normalized_shape": hidden_size,
                "eps": norm_eps,
                "norm_type": "layer_norm",
            },
            "is_data_placeholder": False,
        }
        symbolic_nodes.append(final_norm)
        execution_order.append("vision_final_norm")
        order_counter += 1

        # Vision -> text connector (pixel-shuffle + linear projection).
        # SmolVLM uses a "scale_factor" pixel-shuffle that reshapes
        # (num_patches, hidden_vision) -> (num_patches // scale_factor**2,
        #                                  hidden_vision * scale_factor**2)
        # then a Linear(hidden_vision * scale_factor**2 -> hidden_text).
        text_cfg = self._resolve_text_config()
        hidden_text = getattr(text_cfg, "hidden_size", hidden_size)
        scale_factor = getattr(self.config, "scale_factor", 1) or 1
        pixel_shuffle_dim = hidden_size * (scale_factor * scale_factor)
        shuffled_patches = max(num_patches // (scale_factor * scale_factor), 1)

        connector_info = {
            "name": "vision_connector",
            "operation_type": "vision_projection",
            "operation_category": "vision_projection",
            "execution_order": order_counter,
            "input_shape": current_shape,
            "output_shape": [batch_size, shuffled_patches, hidden_text],
            "dimensions": {
                "in_features": pixel_shuffle_dim,
                "out_features": hidden_text,
                "hidden_size": hidden_text,
                "vision_hidden_size": hidden_size,
                "scale_factor": scale_factor,
                "num_patches_in": num_patches,
                "num_patches_out": shuffled_patches,
            },
            "is_data_placeholder": False,
        }
        symbolic_nodes.append(connector_info)
        execution_order.append("vision_connector")
        order_counter += 1

        return {
            "nodes": symbolic_nodes,
            "execution_order": execution_order,
            "total_nodes": len(symbolic_nodes),
            "component": "vision_encoder",
        }

    def print_summary(self):
        """Print a summary of the model dimensions and structure"""
        dims = self.extract_critical_dimensions()
        self._resolve_text_config()

        print(f"Model: {self.model_name_or_path}")
        print(f"Architecture: {getattr(self.config, 'architectures', ['Unknown'])[0]}")
        print("\n=== Critical Dimensions ===")

        print(f"Vocabulary Size: {dims['vocab_size']}")
        print(f"Hidden Size: {dims['hidden_size']}")
        print(f"Number of Layers: {dims['num_hidden_layers']}")
        print(f"Max Position Embeddings: {dims['max_position_embeddings']}")

        if "mamba" in dims:
            # Mamba-2 has neither attention nor a gated FFN; printing those
            # sections would just report the vacuous defaults.
            m = dims["mamba"]
            print("\n=== Mamba-2 Mixer Dimensions ===")
            print(f"Expand: {m['expand']}  ->  d_inner: {m['d_inner']}")
            print(f"Heads: {m['num_heads']} x head_dim {m['head_dim']}")
            print(f"State Size: {m['state_size']}, Groups: {m['n_groups']}")
            print(f"Conv Kernel: {m['conv_kernel']} over conv_dim {m['conv_dim']} (depthwise, causal)")
            print(f"in_proj: {m['hidden_size']} -> {m['in_proj_out']}  [z, x, B, C, dt]")
            print(f"out_proj: {m['d_inner']} -> {m['hidden_size']}")
            print(f"Chunk Size: {m['chunk_size']}")
            print(f"time_step_limit: ({m['time_step_min']}, {m['time_step_max']})")

            print("\n=== Gated RMSNorm Dimensions ===")
            print(f"Normalized Shape: {m['d_inner'] // m['n_groups']} (per group)")
            print(f"Epsilon: {m['eps']}")
        else:
            print("\n=== Attention Dimensions ===")
            att_dims = dims["attention"]
            print(f"Number of Attention Heads: {att_dims['num_attention_heads']}")
            print(f"Number of Key-Value Heads: {att_dims['num_key_value_heads']}")
            print(f"Head Dimension: {att_dims['head_dim']}")
            print(f"Key-Value Head Dimension: {att_dims['key_value_head_dim']}")

            print("\n=== FFN Dimensions ===")
            ffn_dims = dims["ffn"]
            print(f"Hidden Size: {ffn_dims['hidden_size']}")
            print(f"Intermediate Size: {ffn_dims['intermediate_size']}")
            print(f"Activation: {ffn_dims['activation']}")

            print("\n=== RMSNorm Dimensions ===")
            rms_dims = dims["rms_norm"]
            print(f"Normalized Shape: {rms_dims['normalized_shape']}")
            print(f"Epsilon: {rms_dims['eps']}")

        # Print vision dimensions if present
        if "vision" in dims:
            v = dims["vision"]
            print("\n=== Vision Encoder Dimensions ===")
            print(f"Hidden Size: {v['hidden_size']}")
            print(f"Number of Layers: {v['num_hidden_layers']}")
            print(f"Number of Attention Heads: {v['num_attention_heads']}")
            print(f"Intermediate Size: {v['intermediate_size']}")
            print(f"Head Dimension: {v['head_dim']}")
            print(f"Image Size: {v['image_size']}")
            print(f"Patch Size: {v['patch_size']}")

        # Print symbolic graph summary
        if self.symbolic_graph:
            print("\n=== Symbolic Graph ===")
            print(f"Total Operations: {self.symbolic_graph['total_nodes']}")

            # Group operations by category
            categories = {}
            for node in self.symbolic_graph["nodes"]:
                cat = node.get("operation_category", "unknown")
                categories[cat] = categories.get(cat, 0) + 1

            for cat, count in sorted(categories.items()):
                print(f"{cat}: {count}")

    def print_symbolic_graph_details(self):
        """Print detailed symbolic graph with execution order"""
        if not self.symbolic_graph:
            self.create_symbolic_graph()

        print("\n=== Symbolic Graph Execution Order ===")
        for node in self.symbolic_graph["nodes"]:
            name = node["name"]
            op_type = node["operation_type"]
            category = node.get("operation_category", "unknown")
            is_placeholder = node.get("is_data_placeholder", False)

            placeholder_marker = " [DATA PLACEHOLDER]" if is_placeholder else ""
            print(f"{node['execution_order']:3d}. {name} [{op_type}] -> {category}{placeholder_marker}")

            # Print input/output shapes
            if node.get("input_shape"):
                input_shape = node["input_shape"]
                if isinstance(input_shape[0], list):  # multiple inputs
                    print(f"     Input shapes: {input_shape}")
                else:  # single input
                    print(f"     Input shape: {input_shape}")

            if node.get("output_shape"):
                print(f"     Output shape: {node['output_shape']}")

            # Print operation-specific details
            if node.get("dimensions"):
                dims = node["dimensions"]
                if category == "attention":
                    print(
                        f"     Attention: heads={dims.get('num_attention_heads')}, kv_heads={dims.get('num_key_value_heads')}, head_dim={dims.get('head_dim')}"
                    )
                elif category == "normalization":
                    print(f"     Norm: shape={dims.get('normalized_shape')}, eps={dims.get('eps')}")
                elif category == "embedding":
                    print(f"     Embedding: {dims.get('num_embeddings')} x {dims.get('hidden_size')}")
                elif category == "ffn":
                    print(
                        f"     FFN: {dims.get('hidden_size')} -> {dims.get('intermediate_size')} -> {dims.get('hidden_size')}, activation={dims.get('activation')}"
                    )
                elif category == "elementwise_add":
                    print(f"     Add: shape={dims.get('shape')}")
                elif category == "projection":
                    print(
                        f"     Linear ({dims.get('role', 'projection')}): "
                        f"{dims.get('in_features')} -> {dims.get('out_features')}"
                    )
                elif category == "conv1d":
                    print(
                        f"     Conv1d: channels={dims.get('conv_dim')}, kernel={dims.get('kernel_size')}, "
                        f"depthwise={dims.get('depthwise')}, causal={dims.get('causal')}, "
                        f"activation={dims.get('activation')}"
                    )
                elif category == "ssd_scan":
                    print(
                        f"     SSD: heads={dims.get('num_heads')}x{dims.get('head_dim')}, "
                        f"state={dims.get('state_size')}, groups={dims.get('n_groups')}, "
                        f"chunk={dims.get('chunk_size')} ({dims.get('num_chunks')} chunks)"
                    )

            print()
