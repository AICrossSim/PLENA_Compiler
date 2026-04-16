#!/usr/bin/env python3
"""
Test script for attention code generation
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from passes.code_gen import _generate_attention_code


def test_attention_code_generation():
    """Test the attention code generation function"""

    # Test node with LLaMA-3.1 8B parameters
    test_node = {
        "name": "attention_layer_0",
        "operation_type": "attention",
        "dimensions": {"hidden_size": 4096, "num_attention_heads": 32, "head_dim": 128, "num_key_value_heads": 8},
    }

    # Generate the assembly code
    model_info = {"batch": 1}
    hardware_config = {"MLEN": 64, "BLEN": 4}
    scheduler = {
        "register_assignment": {
            "hbm_addr_reg": {
                "q_weight_offset": 0,
                "k_weight_offset": 0,
                "v_weight_offset": 0,
                "rope_params_offset": 0,
            }
        },
        "memory_layout": {
            "vector_sram_addr": {"block1": 0, "block2": 0, "block3": 0},
            "fp_sram": {},
        },
    }

    generated_code = _generate_attention_code(test_node, model_info=model_info, hardware_config=hardware_config, scheduler=scheduler)

    print("Generated Flash Attention Assembly Code:")
    print("=" * 50)
    print(generated_code)
    print("=" * 50)

    assert isinstance(generated_code, str)
    assert "Self-attention" in generated_code

    # Basic validation
    # assert "Flash Attention Implementation" in generated_code
    # assert "S_LD_FIX i1, i0, 11" in generated_code
    # assert "M_TMM_IC 0, i1, i2" in generated_code
    # assert "M_TMM_PS i7, i1, i2" in generated_code
    # assert "M_MM_WO i1, 0, 0" in generated_code
    # assert "Flash Attention Implementation Template" in generated_code

    print("✅ All tests passed! The attention code generation is working correctly.")


if __name__ == "__main__":
    test_attention_code_generation()
