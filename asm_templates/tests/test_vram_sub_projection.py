"""Unit tests for the pure emitter ``vram_sub_projection_asm_impl``.

Verifies the extracted free function produces the expected ISA output for
looped/unrolled and transposed/non-transposed variants, and asserts
byte-identical parity with the delegating
``IsaCompiler._vram_sub_projection_asm_impl`` method.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))  # project root
sys.path.insert(0, str(_PROJECT_ROOT / "tools"))  # `quant` package lives here

import unittest  # noqa: E402

from compiler.asm_templates.vram_sub_projection_asm import vram_sub_projection_asm_impl  # noqa: E402


def _base_kwargs(**overrides):
    kwargs = dict(
        mlen=64,
        blen=4,
        unroll_loops=False,
        header_lines=["; header"],
        vram_row_start_addr=0,
        mram_start_addr=100,
        result_vram_addr=200,
        full_batch=64,
        num_hidden_blocks=2,
        mat_col_stride=4,
        transposed=False,
        gp_regs=list(range(1, 10)),
        caller_name="test",
    )
    kwargs.update(overrides)
    return kwargs


class TestVramSubProjectionAsmImpl(unittest.TestCase):
    def test_requires_9_gp_regs(self):
        """Passing fewer than 9 gp regs raises ValueError with caller_name."""
        with self.assertRaises(ValueError) as cm:
            vram_sub_projection_asm_impl(**_base_kwargs(gp_regs=[1, 2, 3, 4, 5, 6, 7, 8]))
        self.assertIn("requires at least 9 gp registers", str(cm.exception))

    def test_looped_non_transposed_basic(self):
        """Looped (non-unrolled), non-transposed emits C_LOOP with M_MM order (mat, act)."""
        asm = vram_sub_projection_asm_impl(**_base_kwargs())

        # Starts with the header line.
        self.assertTrue(asm.startswith("; header\n"))

        # tiles_per_mlen = mlen // blen = 64 // 4 = 16
        self.assertIn("C_LOOP_START gp4, 16", asm)
        # num_hidden_blocks = 2 (inner loop)
        self.assertIn("C_LOOP_START gp6, 2", asm)

        # Non-transposed uses M_MM with (mat, act) operand order.
        self.assertIn("M_MM 0, gp2, gp1", asm)
        self.assertNotIn("M_TMM", asm)

        # Three C_LOOP_END lines (outer, middle, inner).
        self.assertEqual(asm.count("C_LOOP_END"), 3)

        # Result write-out instruction.
        self.assertIn("M_MM_WO gp3, gp0, 0", asm)

    def test_looped_transposed_uses_m_tmm(self):
        """Looped, transposed uses M_TMM with (act, mat) operand order."""
        asm = vram_sub_projection_asm_impl(**_base_kwargs(transposed=True))

        # Transposed => M_TMM with act-first, mat-second.
        self.assertIn("M_TMM 0, gp1, gp2", asm)

        # Must NOT contain an M_MM instruction (M_MM_WO is fine — stricter check below).
        for line in asm.splitlines():
            stripped = line.strip()
            if stripped.startswith("M_MM "):  # trailing space distinguishes from M_MM_WO
                self.fail(f"Unexpected M_MM (non-transposed) instruction in transposed output: {stripped}")

    def test_unrolled_no_loops(self):
        """Fully unrolled emits no C_LOOP and bakes every M_MM by hand."""
        asm = vram_sub_projection_asm_impl(**_base_kwargs(unroll_loops=True))

        self.assertNotIn("C_LOOP_START", asm)
        self.assertNotIn("C_LOOP_END", asm)

        # Multiple inlined M_MM 0, gp2, gp1 — one per (oc, or_, ih) iteration.
        mm_count = sum(1 for line in asm.splitlines() if line.strip().startswith("M_MM 0,"))
        self.assertGreater(mm_count, 1)

        # At least one write-out per output-row tile.
        self.assertIn("M_MM_WO", asm)

    def test_output_byte_identical_to_method(self):
        """The free function must produce byte-identical output to IsaCompiler's method."""
        from compiler.aten.plena import IsaCompiler

        compiler = IsaCompiler(mlen=64, blen=4, unroll_loops=False)

        method_kwargs = dict(
            header_lines=["; header"],
            vram_row_start_addr=0,
            mram_start_addr=100,
            result_vram_addr=200,
            full_batch=64,
            num_hidden_blocks=2,
            mat_col_stride=4,
            transposed=False,
            gp_regs=list(range(1, 10)),
            caller_name="test",
        )

        method_out = compiler._vram_sub_projection_asm_impl(**method_kwargs)
        free_out = vram_sub_projection_asm_impl(
            mlen=64,
            blen=4,
            unroll_loops=False,
            **method_kwargs,
        )

        self.assertEqual(method_out, free_out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
