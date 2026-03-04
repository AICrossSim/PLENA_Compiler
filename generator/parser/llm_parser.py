from typing import Any

import torch
from transformers import AutoConfig, AutoModel


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

    def _has_embed_tokens(self):
        """Find embed_tokens in model, supporting nested VLM architectures."""
        model = self.model
        candidate_paths = [
            ["embed_tokens"],
            ["model", "embed_tokens"],
            ["model", "language_model", "model", "embed_tokens"],
            ["language_model", "model", "embed_tokens"],
        ]
        for path in candidate_paths:
            obj = model
            for attr in path:
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            if obj is not None:
                return True
        return False

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
                "dimensions": {"num_embeddings": getattr(text_cfg, "vocab_size", None), "embedding_dim": hidden_size},
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

        self.symbolic_graph = {
            "nodes": symbolic_nodes,
            "execution_order": execution_order,
            "total_nodes": len(symbolic_nodes),
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

        # Patch embedding node
        patch_embed = {
            "name": "vision_patch_embed",
            "operation_type": "embedding",
            "operation_category": "embedding",
            "execution_order": order_counter,
            "input_shape": [batch_size, 3, image_size, image_size],
            "output_shape": current_shape,
            "dimensions": {
                "num_embeddings": num_patches,
                "embedding_dim": hidden_size,
                "patch_size": patch_size,
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

            # Self-attention (ViT has no GQA: num_kv_heads == num_heads)
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

            # FFN
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

        return {
            "nodes": symbolic_nodes,
            "execution_order": execution_order,
            "total_nodes": len(symbolic_nodes),
            "component": "vision_encoder",
        }

    def print_summary(self):
        """Print a summary of the model dimensions and structure"""
        dims = self.extract_critical_dimensions()
        text_cfg = self._resolve_text_config()

        print(f"Model: {self.model_name_or_path}")
        print(f"Architecture: {getattr(self.config, 'architectures', ['Unknown'])[0]}")
        print("\n=== Critical Dimensions ===")

        print(f"Vocabulary Size: {dims['vocab_size']}")
        print(f"Hidden Size: {dims['hidden_size']}")
        print(f"Number of Layers: {dims['num_hidden_layers']}")
        print(f"Max Position Embeddings: {dims['max_position_embeddings']}")

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
                    print(f"     Embedding: {dims.get('num_embeddings')} x {dims.get('embedding_dim')}")
                elif category == "ffn":
                    print(
                        f"     FFN: {dims.get('hidden_size')} -> {dims.get('intermediate_size')} -> {dims.get('hidden_size')}, activation={dims.get('activation')}"
                    )
                elif category == "elementwise_add":
                    print(f"     Add: shape={dims.get('shape')}")

            print()
