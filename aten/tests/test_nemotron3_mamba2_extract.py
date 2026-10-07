"""Nemotron mixer extraction tests without Transformers or model downloads."""

from __future__ import annotations

import struct
from types import SimpleNamespace

import pytest
import torch

from aten.models.nemotron3_mamba2.extract import (
    extract_nemotron3_mamba2_config,
    extract_nemotron3_mamba2_layer,
)
from aten.models.nemotron3_mamba2.lowering import (
    allocate_mamba_tensor_bindings,
    materialize_mamba_weight_images,
)
from aten.models.nemotron3_mamba2.memory import ByteAddressArena
from aten.models.nemotron3_mamba2.reference import PrecisionPolicy


class _TinyMixer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_heads = 2
        self.head_dim = 8
        self.hidden_size = 16
        self.intermediate_size = 16
        self.ssm_state_size = 5
        self.n_groups = 1
        self.chunk_size = 4
        self.conv_kernel_size = 3
        self.conv_dim = 26
        self.layer_idx = 9
        self.activation = "silu"
        self.time_step_min = 0.001
        self.time_step_max = 0.1
        self.time_step_limit = (0.0, float("inf"))
        self.in_proj = torch.nn.Linear(16, 44, bias=False)
        self.conv1d = torch.nn.Conv1d(26, 26, 3, groups=26, bias=True)
        self.A_log = torch.nn.Parameter(torch.zeros(2))
        self.dt_bias = torch.nn.Parameter(torch.zeros(2))
        self.D = torch.nn.Parameter(torch.ones(2))
        self.norm = SimpleNamespace(
            weight=torch.nn.Parameter(torch.ones(16)),
            variance_epsilon=1e-5,
            group_size=16,
        )
        self.out_proj = torch.nn.Linear(16, 16, bias=False)


def test_official_nemotron3_config_metadata_is_extracted_without_model_weights():
    config = extract_nemotron3_mamba2_config(
        SimpleNamespace(
            hidden_size=2688,
            mamba_num_heads=64,
            mamba_head_dim=64,
            ssm_state_size=128,
            n_groups=8,
            chunk_size=128,
            conv_kernel=4,
            layer_norm_epsilon=1e-5,
            mamba_hidden_act="silu",
        )
    )
    assert config == type(config).nemotron3_nano()
    assert config.projection_size == 10304
    assert config.conv_channels == 6144
    assert config.dt_min == 0.0
    assert config.dt_max == float("inf")


def test_decoder_layer_adapter_extracts_exact_weight_orientation_and_biases():
    mixer = _TinyMixer()
    extracted = extract_nemotron3_mamba2_layer(
        SimpleNamespace(block_type="mamba", mixer=mixer, layer_idx=9)
    )
    assert extracted.layer_id == 9
    assert extracted.config.projection_size == 44
    assert tuple(extracted.weights.in_proj_weight.shape) == (44, 16)
    assert tuple(extracted.weights.conv_weight.shape) == (26, 3)
    assert extracted.weights.in_proj_bias is None
    assert extracted.weights.conv_bias is not None
    assert extracted.weights.out_proj_bias is None
    assert extracted.weights.in_proj_weight.dtype == torch.float32


def test_runtime_dt_limit_is_not_confused_with_initialization_range():
    mixer = _TinyMixer()
    mixer.time_step_limit = (0.02, 0.5)
    config = extract_nemotron3_mamba2_config(mixer)
    assert config.dt_min == 0.02
    assert config.dt_max == 0.5


def test_official_bias_presence_and_weight_storage_match_the_bindings():
    extracted = extract_nemotron3_mamba2_layer(_TinyMixer())
    arena = ByteAddressArena(0x100000, 1024 * 1024)
    bindings = allocate_mamba_tensor_bindings(
        arena,
        extracted.config,
        batch_capacity=1,
        sequence_capacity=1,
        precision_policy=PrecisionPolicy.FP32_REFERENCE,
        prefix="layer9",
        include_in_proj_bias=False,
        include_conv_bias=True,
        include_out_proj_bias=False,
    )
    images = materialize_mamba_weight_images(bindings, extracted.weights)
    image_names = {image.region.name for image in images}
    assert "layer9.in_proj_bias" not in image_names
    assert "layer9.conv_bias" in image_names
    assert "layer9.out_proj_bias" not in image_names
    assert all(len(image.data) == image.region.size_bytes for image in images)

    wrong_bindings = allocate_mamba_tensor_bindings(
        arena,
        extracted.config,
        batch_capacity=1,
        sequence_capacity=1,
        precision_policy=PrecisionPolicy.FP32_REFERENCE,
        prefix="wrong",
        include_in_proj_bias=True,
        include_conv_bias=True,
        include_out_proj_bias=False,
    )
    with pytest.raises(ValueError, match="disagree on presence of in_proj_bias"):
        materialize_mamba_weight_images(wrong_bindings, extracted.weights)


def test_bf16_weight_images_store_raw_little_endian_bfloat16_bits():
    extracted = extract_nemotron3_mamba2_layer(_TinyMixer())
    arena = ByteAddressArena(0x200000, 1024 * 1024)
    bindings = allocate_mamba_tensor_bindings(
        arena,
        extracted.config,
        batch_capacity=1,
        sequence_capacity=1,
        precision_policy=PrecisionPolicy.BF16_ACTIVATION_FP32_STATE,
        prefix="bf16",
        include_in_proj_bias=False,
        include_conv_bias=True,
        include_out_proj_bias=False,
    )
    images = materialize_mamba_weight_images(bindings, extracted.weights)
    in_proj = next(
        image for image in images if image.region.name == "bf16.in_proj_weight"
    )
    expected = (
        extracted.weights.in_proj_weight.flatten()[0]
        .to(torch.bfloat16)
        .view(torch.uint16)
        .item()
    )
    assert len(in_proj.data) == extracted.weights.in_proj_weight.numel() * 2
    assert struct.unpack_from("<H", in_proj.data)[0] == expected


def test_extractor_rejects_non_mamba_and_non_depthwise_modules():
    with pytest.raises(ValueError, match="block_type"):
        extract_nemotron3_mamba2_layer(
            SimpleNamespace(block_type="attention", mixer=_TinyMixer())
        )
    mixer = _TinyMixer()
    mixer.conv1d.groups = 1
    with pytest.raises(ValueError, match="groups must equal"):
        extract_nemotron3_mamba2_layer(mixer)


def test_extractor_rejects_unsupported_projection_split_and_activation():
    mixer = _TinyMixer()
    mixer.in_proj = torch.nn.Linear(16, 43, bias=False)
    with pytest.raises(ValueError, match="cannot be split"):
        extract_nemotron3_mamba2_config(mixer)
    mixer = _TinyMixer()
    mixer.activation = "gelu"
    with pytest.raises(ValueError, match="must be SiLU"):
        extract_nemotron3_mamba2_config(mixer)
