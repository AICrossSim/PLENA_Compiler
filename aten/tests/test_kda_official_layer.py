"""Official Kimi K3 KDA layer contract, not only its recurrent boundary."""

from __future__ import annotations

import math

import torch

from compiler.aten.models.kda.reference import (
    KdaConvWeights,
    KdaOfficialLayerWeights,
    KdaRecurrentState,
    kda_official_layer_step,
)
from compiler.aten.models.kda.shape import KdaShape
from compiler.aten.plena import PlenaCompiler
from compiler.aten.plena.program_kda_common import kda_state_rows, kda_vector_rows
from compiler.aten.plena.program_kda_conv import kda_conv_blocks
from compiler.aten.plena.program_kda_gates import kda_head_blocks, kda_key_blocks
from compiler.aten.plena.program_kda_layer import (
    KdaOfficialLayerBuffers,
    KdaOfficialProjectionWeights,
)
from compiler.aten.plena.program_kda_mixer import KdaMixerBuffers


MLEN = 8


def _shape() -> KdaShape:
    return KdaShape(
        hidden_size=16,
        num_heads=2,
        key_dim=MLEN,
        value_dim=MLEN,
        conv_kernel=4,
    )


def _reference_case(seed: int = 19):
    shape = _shape()
    g = torch.Generator().manual_seed(seed)
    rand = lambda *size: torch.randn(*size, generator=g) * 0.1  # noqa: E731
    key_width = shape.projection_size
    value_width = shape.num_heads * shape.value_dim
    decay_rank = MLEN
    conv = KdaConvWeights(
        rand(key_width, shape.conv_kernel),
        rand(key_width, shape.conv_kernel),
        rand(value_width, shape.conv_kernel),
    )
    weights = KdaOfficialLayerWeights(
        q=rand(shape.hidden_size, key_width),
        k=rand(shape.hidden_size, key_width),
        v=rand(shape.hidden_size, value_width),
        decay_a=rand(shape.hidden_size, decay_rank),
        decay_b=rand(decay_rank, key_width),
        beta=rand(shape.hidden_size, shape.num_heads),
        output_gate=rand(shape.hidden_size, value_width),
        output=rand(value_width, shape.hidden_size),
        conv=conv,
        norm_weight=1.0 + rand(shape.value_dim),
        a_log=rand(shape.num_heads),
        dt_bias=rand(shape.num_heads, shape.key_dim),
    )
    hidden = rand(1, shape.hidden_size)
    state = KdaRecurrentState(
        recurrent=rand(1, shape.num_heads, shape.value_dim, shape.key_dim),
        conv=rand(
            1,
            shape.num_heads * (2 * shape.key_dim + shape.value_dim),
            shape.conv_kernel,
        ),
    )
    return shape, hidden, state, weights


def test_official_reference_covers_all_eight_projections_and_output_gate():
    shape, hidden, state, weights = _reference_case()
    got, next_state = kda_official_layer_step(
        hidden,
        state,
        weights,
        shape,
        state_storage="fp32",
        conv_state_storage="fp32",
    )

    assert got.shape == hidden.shape
    assert next_state.recurrent.shape == state.recurrent.shape
    assert next_state.conv.shape == state.conv.shape

    no_gate = KdaOfficialLayerWeights(
        **{**weights.__dict__, "output_gate": torch.zeros_like(weights.output_gate)}
    )
    ungated, _ = kda_official_layer_step(
        hidden,
        state,
        no_gate,
        shape,
        state_storage="fp32",
        conv_state_storage="fp32",
    )
    assert not torch.equal(got, ungated), "the output-gate projection must affect the layer"


def _compiler_layer():
    shape = _shape()
    p = PlenaCompiler(mlen=MLEN, blen=2, mram_tile_capacity=4)
    consts = p.kda_fp_constants()
    norm_consts = p.mamba_fp_constants(name_prefix="kda_output_norm")
    key_width = shape.projection_size
    value_width = shape.num_heads * shape.value_dim
    kb = kda_key_blocks(shape, MLEN)
    vb_rows = kda_vector_rows(shape, MLEN)
    up = lambda n: math.ceil(n / MLEN) * MLEN  # noqa: E731
    a = lambda n, r: p.alloc(n, up(r), MLEN, strict=False)  # noqa: E731

    hidden = p.alloc(
        "hidden", 1, shape.hidden_size, strict=False,
        physical_shape=(p.blen, shape.hidden_size),
    )

    def weight(name: str, rows: int, cols: int):
        return p.input(
            name,
            (rows, cols),
            physical_shape=(up(rows), up(cols)),
            hbm_element_bytes=2,
        )

    weights = KdaOfficialProjectionWeights(
        q=weight("W_q", shape.hidden_size, key_width),
        k=weight("W_k", shape.hidden_size, key_width),
        v=weight("W_v", shape.hidden_size, value_width),
        decay_a=weight("W_decay_a", shape.hidden_size, MLEN),
        decay_b=weight("W_decay_b", MLEN, key_width),
        beta=weight("W_beta", shape.hidden_size, shape.num_heads),
        output_gate=weight("W_output_gate", shape.hidden_size, value_width),
        output=weight("W_output", value_width, shape.hidden_size),
    )
    widths = {"q": key_width, "k": key_width, "v": value_width}
    conv_state = {
        n: a(f"conv_state_{n}", kda_conv_blocks(width, MLEN) * shape.conv_kernel)
        for n, width in widths.items()
    }
    conv_weight = {
        n: a(f"conv_weight_{n}", kda_conv_blocks(width, MLEN) * shape.conv_kernel)
        for n, width in widths.items()
    }
    decay = a("decay", shape.num_heads * kb)
    beta = a("beta", kda_head_blocks(shape, MLEN))
    mixer = KdaMixerBuffers(
        q=a("q", shape.num_heads * kb),
        k=a("k", shape.num_heads * kb),
        v=a("v", vb_rows),
        gate=decay,
        dt_bias=a("dt_bias", shape.num_heads * kb),
        beta_logit=beta,
        state=a("state", kda_state_rows(shape, MLEN)),
        out=a("mixed", vb_rows),
        pred=a("pred", vb_rows),
        err=a("err", vb_rows),
        sq_scratch=a("mix_sq", shape.num_heads * kb),
        decay_fp=p.fp_var("decay_or_q", size=shape.key_dim),
        q_hat_fp=None,
        k_hat_fp=p.fp_var("k_hat", size=shape.key_dim),
        beta_fp=p.fp_var("beta_fp", size=kda_head_blocks(shape, MLEN) * MLEN),
        part_fp=p.fp_var("part", size=kb),
        acc_fp=p.fp_var("acc", size=1),
        output_scale_fp=p.fp_var("output_scale", size=1),
        rate_fp=p.fp_var("rate", size=shape.num_heads),
        lower_bound_fp=p.fp_var("lower_bound", size=1),
        consts=consts,
    )
    mixer.q_hat_fp = mixer.decay_fp
    buffers = KdaOfficialLayerBuffers(
        mixer=mixer,
        conv_state=conv_state,
        conv_weight=conv_weight,
        conv_bias={},
        conv_scratch=a("conv_scratch", max(kda_conv_blocks(w, MLEN) for w in widths.values())),
        decay=decay,
        beta=beta,
        output_gate=a("output_gate", vb_rows),
        norm_weight=a("norm_weight", shape.value_dim // MLEN),
        norm_sq_scratch=a("norm_sq", vb_rows),
        norm_part_fp=p.fp_var("norm_part", size=shape.value_dim // MLEN),
        norm_acc_fp=p.fp_var("norm_acc", size=1),
        norm_consts=norm_consts,
        packed_output=p.alloc(
            "packed_output", 1, value_width, strict=False,
            physical_shape=(p.blen, value_width),
        ),
    )
    out = p.kda_official_layer_decode_v0(
        hidden=hidden,
        weights=weights,
        buffers=buffers,
        shape=shape,
    )
    return p, out


def test_official_lowering_emits_eight_matrix_projections_and_all_stages():
    p, out = _compiler_layer()
    code = p.compile()
    assert out.shape == (1, _shape().hidden_size)
    for stage in (
        "kda_qkv_proj",
        "kda_conv1d",
        "kda_decay",
        "kda_state_update",
        "kda_readout",
        "kda_gated_norm",
        "kda_out_proj",
    ):
        assert f"@stage={stage}" in code
    for weight in (
        "W_q",
        "W_k",
        "W_v",
        "W_decay_a",
        "W_decay_b",
        "W_beta",
        "W_output_gate",
        "W_output",
    ):
        assert weight in code


def test_consecutive_bf16_weights_reserve_the_bytes_the_matrix_reader_consumes():
    p = PlenaCompiler(mlen=MLEN, blen=2)
    first = p.input(
        "first_bf16",
        (16, 16),
        physical_shape=(16, 16),
        hbm_element_bytes=2,
    )
    second = p.input(
        "second_bf16",
        (16, 16),
        physical_shape=(16, 16),
        hbm_element_bytes=2,
    )
    assert second.hbm_addr >= first.hbm_addr + 16 * 16 * 2


def test_plain_bf16_batch_load_programs_a_byte_stride():
    """A multi-row BF16 load advances by bytes, not element count."""
    p = PlenaCompiler(mlen=MLEN, blen=2, real_data_ratio=2.0)
    source = p.input(
        "plain_bf16",
        (8, 16),
        physical_shape=(8, 16),
        real_data_ratio=2.0,
        hbm_element_bytes=2,
    )

    assert source.hbm_element_bytes == 2
    p.load_batch(source, precision=1)
    code = p.compile()

    assert "S_ADDI_INT gp3, gp0, 32" in code
    assert "C_SET_STRIDE_REG gp3" in code
    assert "S_ADDI_INT gp2, gp2, 16" in code


def test_plain_bf16_single_batch_load_advances_hbm_offsets_in_bytes():
    p = PlenaCompiler(mlen=MLEN, blen=2, real_data_ratio=2.0)
    source = p.input(
        "plain_bf16_single",
        (1, 64),
        physical_shape=(1, 64),
        real_data_ratio=2.0,
        hbm_element_bytes=2,
    )

    assert source.hbm_element_bytes == 2
    p.load_batch(source, precision=1)
    code = p.compile()

    # One H_PREFETCH_V moves 4 * MLEN = 32 BF16 values, hence 64 HBM bytes.
    assert "S_ADDI_INT gp2, gp2, 64" in code


def test_bf16_store_preserves_the_element_width_for_a_later_reload():
    p = PlenaCompiler(mlen=MLEN, blen=2, real_data_ratio=2.0)
    source = p.input(
        "plain_bf16_source",
        (1, 64),
        physical_shape=(1, 64),
        real_data_ratio=2.0,
        hbm_element_bytes=2,
    )
    value = p.load_batch(source, precision=1)
    stored = p.store(
        value,
        name="plain_bf16_stored",
        precision=1,
        hbm_element_bytes=2,
        real_data_ratio=1.0,
    )

    assert stored.hbm_element_bytes == 2
    assert stored.hbm_size >= 1 * 64 * 2
    prefix = p.compile()
    p.load_batch(stored, name="plain_bf16_reloaded", precision=1)
    reload_code = p.compile()[len(prefix) :]

    assert "S_ADDI_INT gp2, gp2, 64" in reload_code
