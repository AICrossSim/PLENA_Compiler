"""
Shared ISA instruction emitters -- used by BOTH compilation pipelines.

These templates emit raw PLENA ISA instruction strings. They are
called by:
  - Pipeline 1 (ATen): via PlenaCompiler's op implementations in aten/ops/
  - Pipeline 2 (Generator): via code_gen_pass's node dispatch in generator/passes/

The templates are stateless -- they take dimensions, register indices,
and addresses as parameters and return ISA strings. All hardware-specific
state (VRAM layout, HBM offsets, FPRAM slots) is managed by the calling
pipeline.

See docs/COMPILATION_PIPELINES.md for the full architecture overview.
"""

from .batched_matmul_asm import batched_matmul_asm
from .elementwise_add_asm import elementwise_add_asm
from .embedding_asm import embedding_asm
from .ffn_asm import ffn_asm, ffn_intermediate_asm, ffn_up_silu_asm
from .flash_attn_asm import flash_attn_asm
from .gelu_asm import gelu_asm
from .im2col_asm import im2col_asm
from .im2col_asm_no_shift import im2col_asm_no_shift
from .lm_head import lm_head_asm
from .normalization_asm import layer_norm_asm, rms_norm_asm
from .preload_act import preload_act_asm
from .preload_addr_reg import preload_addr_reg_asm
from .projection_asm import projection_asm, projection_T_asm
from .reset_reg_asm import reset_fpreg_asm, reset_reg_asm
from .rope_asm import rope_asm
from .silu_asm import silu_asm
from .store_act_asm import store_act_asm
from .gemv_asm import gemv_asm

__all__ = [
    "batched_matmul_asm",
    "elementwise_add_asm",
    "embedding_asm",
    "ffn_asm",
    "ffn_intermediate_asm",
    "ffn_up_silu_asm",
    "flash_attn_asm",
    "gelu_asm",
    "gemv_asm",
    "im2col_asm",
    "im2col_asm_no_shift",
    "layer_norm_asm",
    "lm_head_asm",
    "preload_act_asm",
    "preload_addr_reg_asm",
    "projection_T_asm",
    "projection_asm",
    "reset_fpreg_asm",
    "reset_reg_asm",
    "rms_norm_asm",
    "rope_asm",
    "silu_asm",
    "store_act_asm",
]
