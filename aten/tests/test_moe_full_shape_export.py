import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest
import torch

SOURCE = Path(__file__).resolve().parents[1] / 'plena'
sys.path.insert(0, str(SOURCE))
import moe_normal_export as scalar
import moe_full_shape_export as full


def test_packing_matches_reference_for_every_finite_e4m3_encoding():
    from plena_quant.mxfp.utils import pack_fp_to_bin
    # Signed zero follows the existing packer's canonical positive-zero rule.
    fields = [(e-7, (-1 if s else 1) * ((m/8) if e == 0 else 1+m/8))
              for s in range(2) for e in range(15) for m in range(8)]
    exponent = torch.tensor([e for e,m in fields], dtype=torch.float32)
    mantissa = torch.tensor([m for e,m in fields], dtype=torch.float32)
    assert torch.equal(full.pack_e4m3_vectorized(exponent,mantissa),
                       pack_fp_to_bin(exponent,mantissa,exp_width=4,man_width=3))
    with pytest.raises(ValueError):
        full.pack_e4m3_vectorized(torch.tensor([9.]), torch.tensor([1.]))


def test_vectorized_codec_matches_existing_exporter_and_scalar_decode():
    values = full.generated_matrix(13, 19, 27)
    element, scale, decoded, shape = full.encode_array(values)
    old_element, old_scale, old_shape = scalar.encode_matrix(values)
    assert (element, scale, shape) == (old_element, old_scale, old_shape)
    for row in range(13):
        for k in range(19):
            assert decoded[row,k] == scalar.decode_element(element[row*24+k],scale[row*3+k//8])


def test_streamed_full_shape_oracle_matches_independent_scalar(tmp_path):
    routes = [dict(token=t, slot=s, expert=(t+s)%3, weight=0.25*(s+1)) for t in range(5) for s in range(2)]
    workload, golden = full.export_full_shape(tmp_path,input_dim=17,expert_hidden_dim=19,
        tokens=5,routes=routes,name='oracle_check')
    reference = scalar.numerical_reference(workload,(tmp_path/'weights.bin').read_bytes())
    assert golden['output_bf16'] == reference['output_bf16']
    assert golden['pre_round_output_f32'] == reference['pre_round_output_f32']
    assert any(value != 0 for row in golden['output_f32'] for value in row)


def test_gemm_preserves_ascending_k_cancellation_order():
    x = np.array([[1,1,1]],dtype=np.float32)
    w = np.array([[2**24,1,-2**24]],dtype=np.float32)
    assert full.gemm_reference(x,w).item() == 0


def test_nonfinite_values_are_rejected():
    with pytest.raises(ValueError): full.round_bf16(np.array([np.inf],dtype=np.float32))
    with pytest.raises(ValueError): full.encode_array(np.array([[np.nan]],dtype=np.float32))
