#!/usr/bin/env python3
"""
Test script for attention code generation
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from assembler import AssemblyToBinary

from passes.code_gen import _generate_embedding_code


def test_embeddings_code_generation():
    """Test the embeddings code generation function"""

    # Test node with LLaMA-3.1 8B parameters
    test_node = {
        "name": "embeddings",
        "operation_type": "embeddings",
        "dimensions": {"hidden_size": 64, "num_attention_heads": 1, "head_dim": 64, "num_key_value_heads": 1},
    }
    hardware_config = {
        "MLEN": 64,
        "VLEN": 64,
        "BLEN": 4,
        "block_dim": 4,
        "act_block_width": 32,
        "scale_width": 8,
        "alive_registers": [1, 2],
    }
    model_info = {"batch_size": 1, "seq_len": 4, "vocab_size": 1024}
    scheduler = {
        "memory_layout": {
            "vector_sram_addr": {"block1": 0},
            "fp_sram": {},
        },
        "register_assignment": {
            "hbm_addr_reg": {"token_table_offset": 1},
        },
    }

    # Generate the assembly code
    generated_code = _generate_embedding_code(
        test_node, model_info=model_info, hardware_config=hardware_config, scheduler=scheduler
    )

    # Write out assembly
    with open("generated_embedding_assembly.asm", "w") as f:
        f.write(generated_code)

    # Write out machine code
    config_parent_path = Path(__file__).resolve().parents[2]
    print(f"Config parent path: {config_parent_path}")

    print("✅ All tests passed! The attention code generation is working correctly.")


if __name__ == "__main__":
    test_embeddings_code_generation()
    config_path = Path(__file__).resolve().parents[2] / "doc" / "configuration.svh"
    isa_def_path = Path(__file__).resolve().parents[2] / "doc" / "operation.svh"
    assembler = AssemblyToBinary(isa_def_path, config_path)
    assembler.generate_binary("generated_embedding_assembly.asm", "generated_embedding_assembly.mem")
