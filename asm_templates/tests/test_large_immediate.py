"""
Unit tests for LUI+ADDI large immediate fix in ASM templates.
Verifies that S_ADDI_INT is no longer used for values > 2^18,
and that S_LUI_INT + S_ADDI_INT pair is emitted for large matrix sizes.

The list-returning helper lives in `asm_templates._imm`; the str-returning
helper used inside ffn_asm is exercised indirectly via ffn_asm().
"""

import unittest

from asm_templates._imm import load_large_int as _load_large_int
from asm_templates.projection_asm import projection_asm, projection_T_asm


class TestLoadLargeInt(unittest.TestCase):
    """Test the _load_large_int helper from projection_asm directly (returns list[str])."""

    def test_small_value_single_addi(self):
        """Values < 4096 (upper=0) use a single S_ADDI_INT."""
        result = _load_large_int(1, 256)
        asm = "\n".join(result)
        self.assertIn("S_ADDI_INT gp1, gp0, 256", asm)
        self.assertNotIn("S_LUI_INT", asm)

    def test_small_value_below_2_18(self):
        """128*256 = 32768 < 2^18: fits in 18-bit immediate -> single S_ADDI_INT from gp0."""
        val = 128 * 256  # 32768
        result = _load_large_int(2, val)
        asm = "\n".join(result)
        # 32768 < 2^18, so single S_ADDI_INT from gp0 (no LUI needed)
        self.assertIn("S_ADDI_INT gp2, gp0, 32768", asm)
        self.assertNotIn("S_LUI_INT", asm)

    def test_boundary_value_2_18(self):
        """Value = 2^18 = 262144 needs LUI (upper=64, lower=0)."""
        result = _load_large_int(2, 262144)
        asm = "\n".join(result)
        self.assertIn("S_LUI_INT gp2, 64", asm)  # 262144 >> 12 = 64
        self.assertNotIn("S_ADDI_INT", asm)  # lower = 262144 & 0xFFF = 0

    def test_2048x2048(self):
        """2048*2048 = 4194304: upper=1024, lower=0 -> LUI only."""
        val = 2048 * 2048  # 4194304
        result = _load_large_int(3, val)
        asm = "\n".join(result)
        self.assertIn("S_LUI_INT gp3, 1024", asm)  # 4194304 >> 12 = 1024
        self.assertNotIn("S_ADDI_INT", asm)  # lower = 0

    def test_2048x8192(self):
        """2048*8192 = 16777216: upper=4096, lower=0 -> LUI only."""
        val = 2048 * 8192  # 16777216
        result = _load_large_int(4, val)
        asm = "\n".join(result)
        self.assertIn("S_LUI_INT gp4, 4096", asm)  # 16777216 >> 12 = 4096
        self.assertNotIn("S_ADDI_INT", asm)  # lower = 0

    def test_value_with_remainder(self):
        """Value with non-zero lower 12 bits emits both LUI and ADDI."""
        val = (100 << 12) + 500  # upper=100, lower=500
        result = _load_large_int(5, val)
        asm = "\n".join(result)
        self.assertIn("S_LUI_INT gp5, 100", asm)
        self.assertIn("S_ADDI_INT gp5, gp5, 500", asm)

    def test_zero_value(self):
        """Zero should emit S_ADDI_INT gp{reg}, gp0, 0."""
        result = _load_large_int(0, 0)
        asm = "\n".join(result)
        self.assertIn("S_ADDI_INT gp0, gp0, 0", asm)
        self.assertNotIn("S_LUI_INT", asm)

    def test_64x64_small(self):
        """64*64 = 4096 < 2^18: fits in 18-bit immediate -> single S_ADDI_INT from gp0."""
        val = 64 * 64  # 4096
        result = _load_large_int(1, val)
        asm = "\n".join(result)
        # 4096 < 2^18, so single S_ADDI_INT from gp0 (no LUI needed)
        self.assertIn("S_ADDI_INT gp1, gp0, 4096", asm)
        self.assertNotIn("S_LUI_INT", asm)


class TestProjectionAsmLargeMatrix(unittest.TestCase):
    """Test that projection_asm generates valid code for large matrices."""

    BASE_ARGS = dict(
        mlen=64,
        blen=4,
        batch=4,
        alive_registers=[1, 2, 3, 4, 5, 6],
        w_base_hbm_offset_reg=0,
        activation_base_address=0,
        result_base_address=4096,
    )

    def test_small_matrix_no_change(self):
        """64x64 should still work and contain expected instructions."""
        asm = projection_asm(hidden_size=64, **self.BASE_ARGS)
        self.assertIn("C_SET_SCALE_REG", asm)
        self.assertIn("C_SET_STRIDE_REG", asm)

    def test_small_matrix_no_lui(self):
        """64x64 = 4096 elements < 2^18: single S_ADDI_INT from gp0 (no LUI needed)."""
        asm = projection_asm(hidden_size=64, **self.BASE_ARGS)
        # 4096 < 2^18, so scale loads via S_ADDI_INT gp3, gp0, 4096 (act_reg = alive_registers[2] = 3)
        self.assertIn("S_ADDI_INT gp3, gp0, 4096", asm)
        # Must NOT use LUI for this small value
        self.assertNotIn("S_LUI_INT gp3, 1", asm)

    def test_128x256_no_assertion_error(self):
        """128*256 = 32768 < 2^18: should not raise AssertionError."""
        try:
            asm = projection_asm(hidden_size=128, out_features=256, **self.BASE_ARGS)
        except AssertionError as e:
            self.fail(f"projection_asm raised AssertionError for 128x256: {e}")
        self.assertIn("C_SET_SCALE_REG", asm)

    def test_512x512_boundary(self):
        """512*512 = 262144 = 2^18: needs LUI (upper=64, lower=0)."""
        try:
            asm = projection_asm(hidden_size=512, out_features=512, **self.BASE_ARGS)
        except AssertionError as e:
            self.fail(f"projection_asm raised AssertionError for 512x512: {e}")
        self.assertIn("S_LUI_INT", asm)
        self.assertNotIn("S_ADDI_INT gp3, gp0, 262144", asm)

    def test_2048x2048_no_assertion_error(self):
        """2048x2048 should not raise AssertionError and must use LUI."""
        try:
            asm = projection_asm(hidden_size=2048, out_features=2048, **self.BASE_ARGS)
        except AssertionError as e:
            self.fail(f"projection_asm raised AssertionError for 2048x2048: {e}")
        self.assertIn("S_LUI_INT", asm)
        # Must not have the raw large value in a single ADDI from gp0
        self.assertNotIn("S_ADDI_INT gp3, gp0, 4194304", asm)

    def test_2048x8192_no_assertion_error(self):
        """2048x8192 = 16777216 should not raise AssertionError and must use LUI."""
        try:
            asm = projection_asm(hidden_size=2048, out_features=8192, **self.BASE_ARGS)
        except AssertionError as e:
            self.fail(f"projection_asm raised AssertionError for 2048x8192: {e}")
        self.assertIn("S_LUI_INT", asm)
        self.assertNotIn("S_ADDI_INT gp3, gp0, 16777216", asm)
        _check_all_addi_immediates(self, asm, "projection_asm(2048,8192)")


class TestProjectionTAsmLargeMatrix(unittest.TestCase):
    """Test that projection_T_asm also handles large matrices correctly."""

    BASE_ARGS = dict(
        mlen=64,
        blen=4,
        batch=4,
        alive_registers=[1, 2, 3, 4, 5, 6],
        w_base_hbm_offset_reg=0,
        activation_base_address=0,
        result_base_address=4096,
    )

    def test_small_matrix(self):
        """64x64 works without error."""
        asm = projection_T_asm(hidden_size=64, **self.BASE_ARGS)
        self.assertIn("C_SET_SCALE_REG", asm)

    def test_2048x2048_no_assertion_error(self):
        """2048x2048 projection_T_asm should not raise AssertionError."""
        try:
            asm = projection_T_asm(hidden_size=2048, out_features=2048, **self.BASE_ARGS)
        except AssertionError as e:
            self.fail(f"projection_T_asm raised AssertionError for 2048x2048: {e}")
        self.assertIn("S_LUI_INT", asm)
        _check_all_addi_immediates(self, asm, "projection_T_asm(2048,2048)")


class TestFfnAsmLargeMatrix(unittest.TestCase):
    """Test that ffn_asm handles large matrices without AssertionError.

    ffn_asm has its own _load_large_int (returns str), tested indirectly.
    """

    BASE_ARGS = dict(
        mlen=64,
        vlen=64,
        blen=4,
        batch=4,
        seq_len=64,
        alive_registers=list(range(12)),
        gate_weight_hbm_offset_reg=0,
        up_weight_hbm_offset_reg=1,
        down_weight_hbm_offset_reg=2,
        const_one_fp_address=5,
        activation_base_address=0,
    )

    def _import_ffn_asm(self):
        from asm_templates.ffn_asm import ffn_asm

        return ffn_asm

    def test_small_ffn_no_error(self):
        """hidden=64, intermediate=128 (current test scale) works fine."""
        ffn_asm = self._import_ffn_asm()
        try:
            asm = ffn_asm(hidden_size=64, intermediate_size=128, **self.BASE_ARGS)
        except AssertionError as e:
            self.fail(f"ffn_asm raised AssertionError for 64x128: {e}")
        self.assertIsInstance(asm, str)
        self.assertGreater(len(asm), 0)

    def test_large_ffn_hidden128_inter256_no_assertion_error(self):
        """hidden=128, intermediate=256 (smollm2 test scale) should not raise.
        128*256=32768 < 2^18: fits in 18-bit immediate, no LUI needed."""
        ffn_asm = self._import_ffn_asm()
        try:
            asm = ffn_asm(hidden_size=128, intermediate_size=256, **self.BASE_ARGS)
        except AssertionError as e:
            self.fail(f"ffn_asm raised AssertionError for 128x256: {e}")
        # 128*256=32768 < 2^18 -> single S_ADDI_INT from gp0, no LUI needed
        self.assertIn("S_ADDI_INT", asm)
        self.assertNotIn("S_LUI_INT", asm)

    def test_large_ffn_2048_8192_no_assertion_error(self):
        """hidden=2048, intermediate=8192 (SmolVLM2 FFN scale) should not raise."""
        ffn_asm = self._import_ffn_asm()
        try:
            asm = ffn_asm(hidden_size=2048, intermediate_size=8192, **self.BASE_ARGS)
        except AssertionError as e:
            self.fail(f"ffn_asm raised AssertionError for 2048x8192: {e}")
        self.assertIn("S_LUI_INT", asm)
        # S_ADD_INT must appear (proving LUI+ADD pattern for large loop increments)
        self.assertIn("S_ADD_INT", asm)
        # All S_ADDI_INT immediates must be < 2^18 regardless of rs1
        _check_all_addi_immediates(self, asm, "ffn_asm(2048, 8192)")

    def test_large_ffn_2048_8192_gate_result_absolute(self):
        """gate_result address must be computed as absolute, not relative with large offset."""
        ffn_asm = self._import_ffn_asm()
        asm = ffn_asm(hidden_size=2048, intermediate_size=8192, **self.BASE_ARGS)
        # gate_result = batch*seq_len*(hidden+inter) = 4*64*(2048+8192) = 2621440
        # Must use LUI, not a single ADDI
        self.assertNotIn("S_ADDI_INT gp5, gp3, 2097152", asm)  # old bad pattern

    def test_small_ffn_all_addi_bounded(self):
        """Even for small FFN, all S_ADDI_INT immediates must fit in 18 bits."""
        ffn_asm = self._import_ffn_asm()
        asm = ffn_asm(hidden_size=64, intermediate_size=128, **self.BASE_ARGS)
        _check_all_addi_immediates(self, asm, "ffn_asm(64, 128)")

    def test_loop_path_large_2048_8192_no_assertion_error(self):
        """use_loop_instructions=True (production path) must handle large matrices."""
        ffn_asm = self._import_ffn_asm()
        args = dict(self.BASE_ARGS)
        args["alive_registers"] = list(range(10))  # loop version needs 10
        try:
            asm = ffn_asm(hidden_size=2048, intermediate_size=8192, use_loop_instructions=True, **args)
        except AssertionError as e:
            self.fail(f"ffn_asm loop path raised AssertionError for 2048x8192: {e}")
        self.assertIn("S_LUI_INT", asm)
        _check_all_addi_immediates(self, asm, "ffn_asm(loop,2048,8192)")

    def test_loop_path_smollm2_scale(self):
        """Production path with smollm2 scale (hidden=128, inter=256) all immediates bounded."""
        ffn_asm = self._import_ffn_asm()
        args = dict(self.BASE_ARGS)
        args["alive_registers"] = list(range(10))
        asm = ffn_asm(hidden_size=128, intermediate_size=256, use_loop_instructions=True, **args)
        _check_all_addi_immediates(self, asm, "ffn_asm(loop,128,256)")


def _check_all_addi_immediates(test_case, asm: str, label: str) -> None:
    """Assert every S_ADDI_INT immediate in asm is < 2^18."""
    IMM2_BOUND = 1 << 18
    for line in asm.splitlines():
        line = line.strip()
        if not line.startswith("S_ADDI_INT"):
            continue
        parts = line.split(",")
        if len(parts) >= 3:
            try:
                imm = int(parts[-1].strip())
                test_case.assertLess(
                    imm, IMM2_BOUND, f"[{label}] S_ADDI_INT with large immediate {imm} (>= 2^18): {line}"
                )
            except ValueError:
                pass  # not a plain integer literal, skip


if __name__ == "__main__":
    unittest.main(verbosity=2)
