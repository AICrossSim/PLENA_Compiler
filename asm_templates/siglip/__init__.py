"""SigLIP-specific assembly emitters.

These functions are stateless ISA emitters intended for reuse by harnesses
and compiler pipelines.
"""

from .embedding import build_embedding_stage_asm
from .encoder_layer import build_encoder_layer_asm, build_mlp_block
from .full_model_pipeline import build_full_model_asm, compute_hbm_data_order

__all__ = [
    "build_embedding_stage_asm",
    "build_encoder_layer_asm",
    "build_full_model_asm",
    "build_mlp_block",
    "compute_hbm_data_order",
]
