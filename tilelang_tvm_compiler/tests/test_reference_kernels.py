"""Smoke-test the reference kernels under ``kernels/`` against the new
frontend pipeline. Each kernel must compile through to ISA without
errors, and the resulting ISA must contain the expected hardware opcodes.
"""

from __future__ import annotations

import re

import tilelang_tvm_compiler  # bootstrap TVM 0.23

from tilelang_tvm_compiler.frontend import compile_func, compile_to_tir_text
from tilelang_tvm_compiler.kernels.mm64 import make_mm64
from tilelang_tvm_compiler.kernels.qk_btmm import make_qk_btmm
from tilelang_tvm_compiler.pipeline import compile_kernel, PlenaTarget


def test_mm64_reference_full_pipeline():
    func = compile_func(make_mm64())
    ck = compile_kernel(func, target=PlenaTarget(), name="mm64")
    isa = ck.isa_text
    assert "M_MM" in isa
    assert "M_MM_WO" in isa
    # No btmm opcodes should sneak in.
    assert "M_BTMM" not in isa
    assert "M_BMM_WO" not in isa


def test_mm64_reference_tir_text_shape():
    text = compile_to_tir_text(make_mm64(), name="mm64")
    # One matmul call, three DMAs (2 in, 1 out).
    assert text.count("plena.matmul") == 1
    assert text.count("plena.dma_h2v_slice") == 1
    assert text.count("plena.dma_h2m_slice") == 1
    assert text.count("plena.dma_v2h_slice") == 1
    # No surviving thread or lane loops.
    assert "blockIdx" not in text
    assert "threadIdx" not in text
    assert "for by" not in text


def test_qk_btmm_reference_full_pipeline():
    func = compile_func(make_qk_btmm())
    ck = compile_kernel(func, target=PlenaTarget(), name="qk_btmm")
    isa = ck.isa_text
    assert "M_BTMM" in isa
    assert "M_BMM_WO" in isa


def test_qk_btmm_reference_lane_fusion():
    text = compile_to_tir_text(make_qk_btmm(), name="qk_btmm")
    # Per-head for-loop is dropped — everything fused into one multi-lane
    # HW op per role.
    assert "for by" not in text
    # plena.btmm carries lane_count=4 as the trailing arg.
    assert re.search(r"plena\.btmm.*?, 4\)", text), text
    # Lane-fused DMAs: H position (3rd extent) == lane_count = 4.
    assert re.search(r"plena\.dma_h2v_slice.*?, 1, 64, 4, 16", text), text
    assert re.search(r"plena\.dma_h2m_slice.*?, 1, 64, 4, 16", text), text


def test_qk_btmm_reference_buffer_scopes():
    text = compile_to_tir_text(make_qk_btmm(), name="qk_btmm")
    # BTMM input that comes from H_PREFETCH_M lands in mram; the other
    # in vram. S_loc is the BTMM output (vram).
    assert 'scope="mram"' in text
    assert 'scope="vram"' in text


def test_qk_btmm_reference_buffer_expansion():
    text = compile_to_tir_text(make_qk_btmm(), name="qk_btmm")
    # Per-lane (64, 16) → 4D (1, 64, 4, 16) BSHD-packed.
    assert re.search(r"Q_sh = T\.alloc_buffer\(\(1, 64, 4, 16\)", text), text
    assert re.search(r"K_sh = T\.alloc_buffer\(\(1, 64, 4, 16\)", text), text
    # BTMM output (64, 64) → 4D (1, 4, 64, 64) BHSD-stacked.
    assert re.search(r"S_loc = T\.alloc_buffer\(\(1, 4, 64, 64\)", text), text


if __name__ == "__main__":
    test_mm64_reference_full_pipeline()
    test_mm64_reference_tir_text_shape()
    test_qk_btmm_reference_full_pipeline()
    test_qk_btmm_reference_lane_fusion()
    test_qk_btmm_reference_buffer_scopes()
    test_qk_btmm_reference_buffer_expansion()
    print("reference kernel tests passed")
