from __future__ import annotations

import ast
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from compiler.aten.execution_trace import HBM_READ, iter_loop_scoped_lines
from compiler.aten.model_extract import extract_model_config
from compiler.aten.plena.packed_kv import (
    PackedKVAblation,
    PackedKVLayout,
    validate_selector_lowering,
)


class PackedKVLayoutTests(unittest.TestCase):
    @staticmethod
    def _full_model_decode_trace(
        weight_bits: int,
        value_bits: int | None = None,
        *,
        batch_size: int = 2,
    ):
        from compiler.aten.plena_frontend import compile_native_hf_decoder

        config = SimpleNamespace(
            hidden_size=16,
            num_attention_heads=2,
            num_key_value_heads=2,
            intermediate_size=32,
            rms_norm_eps=1.0e-5,
            rope_theta=10000.0,
            vocab_size=32,
            model_type="llama",
        )
        model = SimpleNamespace(config=config, layers=[None])
        return compile_native_hf_decoder(
            model,
            seq_len=1,
            batch_size=batch_size,
            num_layers=1,
            mlen=16,
            blen=2,
            hlen=8,
            broadcast_amount=2,
            attention_head_packing=True,
            packed_kv_layout=PackedKVLayout(
                kv_heads=2,
                head_dim=8,
                mlen=16,
                element_bits=4,
            ),
            packed_value_layout=(
                PackedKVLayout(
                    kv_heads=2,
                    head_dim=8,
                    mlen=16,
                    element_bits=value_bits,
                )
                if value_bits is not None
                else None
            ),
            decode_context_tokens=17,
            external_packed_kv_cache=True,
            trace_only=True,
            output_head_location="prefill_chip",
            weight_element_bits=weight_bits,
        )

    @staticmethod
    def _external_decode_trace(
        *,
        hidden_size: int = 32,
        num_heads: int = 4,
        num_kv_heads: int = 2,
        head_dim: int = 8,
        intermediate_size: int = 64,
        mlen: int = 16,
        context_tokens: int = 17,
        tensor_parallel_degree: int = 1,
        tensor_parallel_rank: int = 0,
        kv_parallel_degree: int = 1,
        kv_parallel_rank: int = 0,
        layout_kv_heads: int | None = None,
        kv_head_reuse: bool = True,
        bind_parallel_arguments: bool = True,
    ):
        from compiler.aten.plena_frontend import compile_native_hf_decoder

        config = SimpleNamespace(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            head_dim=head_dim,
            intermediate_size=intermediate_size,
            rms_norm_eps=1.0e-5,
            rope_theta=10000.0,
            vocab_size=32,
            model_type="llama",
        )
        compile_arguments = {}
        if bind_parallel_arguments:
            compile_arguments.update(
                tensor_parallel_degree=tensor_parallel_degree,
                tensor_parallel_rank=tensor_parallel_rank,
                kv_parallel_degree=kv_parallel_degree,
                kv_parallel_rank=kv_parallel_rank,
                kv_token_sharding="round_robin",
                kv_head_reuse=kv_head_reuse,
            )
        return compile_native_hf_decoder(
            SimpleNamespace(config=config, layers=[None]),
            seq_len=1,
            batch_size=1,
            num_layers=1,
            mlen=mlen,
            blen=mlen // head_dim,
            hlen=head_dim,
            broadcast_amount=mlen // head_dim,
            attention_head_packing=True,
            packed_kv_layout=PackedKVLayout(
                kv_heads=(
                    num_kv_heads // tensor_parallel_degree
                    if layout_kv_heads is None
                    else layout_kv_heads
                ),
                head_dim=head_dim,
                mlen=mlen,
                element_bits=4,
            ),
            decode_context_tokens=context_tokens,
            external_packed_kv_cache=True,
            trace_only=True,
            output_head_location="prefill_chip",
            weight_element_bits=4,
            **compile_arguments,
        )

    @staticmethod
    def _physical_cache_read_bytes(result, tensor: str) -> int:
        return sum(
            entry.total_dma_bytes
            for entry in result["compilation_artifact"].execution_trace.entries
            if entry.tensor == tensor and entry.dma_direction == HBM_READ
        )

    @staticmethod
    def _cache_read_runs(result, tensor: str):
        artifact = result["compilation_artifact"]
        bindings = {
            binding.trace_entry_index: binding
            for binding in artifact.request_memory.bindings
        }
        return [
            run
            for index, entry in enumerate(artifact.execution_trace.entries)
            if entry.tensor == tensor and entry.dma_direction == HBM_READ
            for run in bindings[index].runs
        ]

    def test_full_model_external_cache_appends_each_batch_tail(self) -> None:
        for batch_size in (1, 2):
            with self.subTest(batch_size=batch_size):
                result = self._full_model_decode_trace(
                    4,
                    batch_size=batch_size,
                )
                artifact = result["compilation_artifact"]
                artifact.request_memory.validate_trace(
                    artifact.execution_trace
                )

                self.assertEqual(result["info"]["batch_size"], batch_size)
                self.assertEqual(result["info"]["active_rows"], batch_size)
                self.assertEqual(
                    result["isa"].count("H_STORE_V"),
                    2 * batch_size,
                )
                tail_index = result["info"]["decode_context_tokens"] - 1
                bindings = {
                    binding.trace_entry_index: binding
                    for binding in artifact.request_memory.bindings
                }
                for cache_name in ("K_cache_0", "V_cache_0"):
                    reads = []
                    appends = []
                    for index, entry in enumerate(artifact.execution_trace.entries):
                        if entry.tensor != cache_name:
                            continue
                        binding = bindings.get(index)
                        if binding is None:
                            continue
                        if entry.dma_direction == HBM_READ:
                            reads.extend(binding.iter_requests())
                        elif entry.opcode == "H_STORE_V":
                            appends.extend(binding.iter_requests())

                    self.assertEqual(len(appends), batch_size)
                    self.assertGreater(len(reads), 0)
                    self.assertEqual(len(reads) % batch_size, 0)
                    reads_per_batch = len(reads) // batch_size
                    for batch_index, request in enumerate(appends):
                        base = reads[batch_index * reads_per_batch]
                        token_bytes = (
                            request.elements_per_row * request.element_bits // 8
                        )
                        expected = base.address + tail_index * token_bytes
                        self.assertEqual(request.address, expected)

    def test_full_model_external_cache_trace_has_exact_subbyte_requests(self) -> None:
        result = self._full_model_decode_trace(4)
        artifact = result["compilation_artifact"]

        self.assertEqual(
            result["info"]["artifact_scope"],
            "full_model_decode_step_independent_request_batch",
        )
        self.assertFalse(result["info"]["output_head_included"])
        self.assertEqual(result["info"]["output_head_location"], "prefill_chip")
        self.assertIsNotNone(artifact.request_memory)
        artifact.request_memory.validate_trace(artifact.execution_trace)
        matrix_tensors = {
            run.request.tensor
            for binding in artifact.request_memory.bindings
            for run in binding.runs
            if run.request.opcode == "H_PREFETCH_M"
        }
        self.assertTrue(
            {"W_o_0", "W_gate_0", "W_up_0", "W_down_0"}
            <= matrix_tensors
        )
        self.assertEqual(result["isa"].count("H_STORE_V"), 4)
        self.assertFalse(result["info"]["compiled_qk_norm"])

    def test_qwen_config_preserves_explicit_head_dim_and_qk_norm_default(self) -> None:
        model = SimpleNamespace(
            config=SimpleNamespace(
                hidden_size=5120,
                num_attention_heads=64,
                num_key_value_heads=8,
                head_dim=128,
                intermediate_size=25_600,
                rms_norm_eps=1.0e-6,
                rope_theta=1_000_000.0,
                vocab_size=151_936,
                model_type="qwen3",
            )
        )

        config = extract_model_config(model)

        self.assertEqual(config.head_dim, 128)
        self.assertTrue(config.qk_norm)

    def test_qwen_trace_normalizes_each_packed_q_and_k_head_before_rope(self) -> None:
        from compiler.aten.plena_frontend import compile_native_hf_decoder

        config = SimpleNamespace(
            hidden_size=16,
            num_attention_heads=4,
            num_key_value_heads=1,
            head_dim=8,
            intermediate_size=32,
            rms_norm_eps=1.0e-6,
            rope_theta=10000.0,
            vocab_size=32,
            model_type="qwen3",
        )
        result = compile_native_hf_decoder(
            SimpleNamespace(config=config, layers=[None]),
            seq_len=1,
            batch_size=2,
            num_layers=1,
            mlen=32,
            blen=2,
            hlen=8,
            broadcast_amount=4,
            attention_head_packing=True,
            packed_kv_layout=PackedKVLayout(
                kv_heads=1,
                head_dim=8,
                mlen=32,
                element_bits=4,
            ),
            decode_context_tokens=17,
            external_packed_kv_cache=True,
            trace_only=True,
            output_head_location="prefill_chip",
            weight_element_bits=4,
            weight_storage_format="mxfp",
            kv_storage_format="mxint",
        )

        info = result["info"]
        self.assertTrue(info["compiled_qk_norm"])
        self.assertEqual(info["head_dim"], 8)
        self.assertEqual(info["qk_norm_segment_width"], 8)
        self.assertEqual(info["qk_norm_reciprocal_fp_offset"], 6)
        self.assertEqual(info["qk_norm_affine_storage_shape"], [4, 32])
        self.assertEqual(info["weight_storage_format"], "mxfp")
        self.assertEqual(info["kv_storage_format"], "mxint")
        self.assertEqual(result["fp_preload"][6], 0.125)
        self.assertEqual(
            result["data_order"].count("W_q_norm_0")
            + result["data_order"].count("W_k_norm_0"),
            2,
        )

        assembly = result["isa"]
        self.assertEqual(assembly.count("; Segmented RMSNorm"), 2)
        self.assertEqual(assembly.count("; === VRAM Broadcast Row Mul"), 2)
        q_norm = assembly.index("; Segmented RMSNorm")
        q_rope = assembly.index("; VRAM View Q_group0_b0_0")
        k_norm = assembly.index("; Segmented RMSNorm", q_norm + 1)
        k_rope = assembly.index("; Allocate VRAM Matrix K_rot_0_h0")
        self.assertLess(q_norm, q_rope)
        self.assertLess(k_norm, k_rope)

    def test_full_model_external_cache_trace_preserves_eight_bit_path(self) -> None:
        result = self._full_model_decode_trace(8)
        artifact = result["compilation_artifact"]

        self.assertIsNotNone(artifact.request_memory)
        artifact.request_memory.validate_trace(artifact.execution_trace)
        self.assertEqual(result["info"]["batch_size"], 2)
        self.assertEqual(result["info"]["decode_context_tokens"], 17)

    def test_full_model_external_cache_trace_seals_split_key_value_widths(self) -> None:
        result = self._full_model_decode_trace(4, value_bits=2)
        info = result["info"]

        self.assertEqual(info["packed_key_element_bits"], 4)
        self.assertEqual(info["packed_value_element_bits"], 2)
        self.assertNotEqual(
            info["packed_key_layout_id"],
            info["packed_value_layout_id"],
        )
        self.assertGreater(
            result["hbm_sizes"]["K_cache_0"],
            result["hbm_sizes"]["V_cache_0"],
        )
        self.assertEqual(
            result["tensor_layouts"]["K_cache_0"]["hbm_element_width"],
            4,
        )
        self.assertEqual(
            result["tensor_layouts"]["V_cache_0"]["hbm_element_width"],
            2,
        )

        requests = [
            run.request
            for binding in result["compilation_artifact"].request_memory.bindings
            for run in binding.runs
        ]
        self.assertEqual(
            {request.element_bits for request in requests if request.tensor == "K_cache_0"},
            {4},
        )
        self.assertEqual(
            {request.element_bits for request in requests if request.tensor == "V_cache_0"},
            {2},
        )

    def test_kv_head_reuse_preserves_compute_and_reduces_physical_reads(self) -> None:
        compute_opcodes = ("M_BTMM", "M_BMM_WO", "M_MM", "M_MM_WO")
        for selector_count in (2, 4):
            with self.subTest(selector_count=selector_count):
                head_dim = 32 // selector_count
                baseline = self._external_decode_trace(
                    hidden_size=32,
                    num_heads=selector_count,
                    num_kv_heads=selector_count,
                    head_dim=head_dim,
                    intermediate_size=64,
                    mlen=32,
                    context_tokens=33,
                    kv_head_reuse=False,
                )
                reused = self._external_decode_trace(
                    hidden_size=32,
                    num_heads=selector_count,
                    num_kv_heads=selector_count,
                    head_dim=head_dim,
                    intermediate_size=64,
                    mlen=32,
                    context_tokens=33,
                    kv_head_reuse=True,
                )

                for result in (baseline, reused):
                    artifact = result["compilation_artifact"]
                    artifact.request_memory.validate_trace(
                        artifact.execution_trace
                    )
                self.assertFalse(baseline["info"]["compiled_kv_head_reuse"])
                self.assertTrue(reused["info"]["compiled_kv_head_reuse"])
                self.assertEqual(
                    reused["info"]["local_kv_head_selector_count"],
                    selector_count,
                )
                self.assertEqual(baseline["hbm_addrs"], reused["hbm_addrs"])
                self.assertEqual(baseline["hbm_sizes"], reused["hbm_sizes"])
                self.assertEqual(
                    baseline["comparison_params"],
                    reused["comparison_params"],
                )

                baseline_histogram = (
                    baseline["compilation_artifact"]
                    .execution_trace.opcode_histogram
                )
                reused_histogram = (
                    reused["compilation_artifact"]
                    .execution_trace.opcode_histogram
                )
                self.assertEqual(
                    {opcode: baseline_histogram[opcode] for opcode in compute_opcodes},
                    {opcode: reused_histogram[opcode] for opcode in compute_opcodes},
                )
                for selector in range(selector_count):
                    self.assertIn(f"M_BTMM {selector},", reused["isa"])

                for cache_name in ("K_cache_0", "V_cache_0"):
                    baseline_bytes = self._physical_cache_read_bytes(
                        baseline,
                        cache_name,
                    )
                    reused_bytes = self._physical_cache_read_bytes(
                        reused,
                        cache_name,
                    )
                    self.assertEqual(
                        baseline_bytes,
                        selector_count * reused_bytes,
                    )

                    baseline_runs = self._cache_read_runs(
                        baseline,
                        cache_name,
                    )
                    reused_runs = self._cache_read_runs(reused, cache_name)
                    baseline_coordinates = {
                        (
                            run.request.address,
                            run.request.scale_address,
                            run.request.rows,
                            run.request.elements_per_row,
                        )
                        for run in baseline_runs
                    }
                    reused_coordinates = {
                        (
                            run.request.address,
                            run.request.scale_address,
                            run.request.rows,
                            run.request.elements_per_row,
                        )
                        for run in reused_runs
                    }
                    self.assertTrue(
                        reused_coordinates <= baseline_coordinates
                    )
                    self.assertEqual(
                        sum(run.repetitions for run in baseline_runs),
                        selector_count
                        * sum(run.repetitions for run in reused_runs),
                    )

    def test_tensor_parallel_ranks_lower_identical_local_programs(self) -> None:
        global_rank = self._external_decode_trace()
        rank_zero = self._external_decode_trace(
            tensor_parallel_degree=2,
            tensor_parallel_rank=0,
        )
        rank_one = self._external_decode_trace(
            tensor_parallel_degree=2,
            tensor_parallel_rank=1,
        )

        self.assertEqual(
            rank_zero["compilation_artifact"].to_dict(),
            rank_one["compilation_artifact"].to_dict(),
        )
        self.assertEqual(rank_zero["hbm_sizes"], rank_one["hbm_sizes"])
        self.assertEqual(
            rank_zero["tensor_layouts"],
            rank_one["tensor_layouts"],
        )
        for rank, result in enumerate((rank_zero, rank_one)):
            info = result["info"]
            self.assertEqual(info["tensor_parallel_degree"], 2)
            self.assertEqual(info["tensor_parallel_rank"], rank)
            self.assertEqual(info["local_num_heads"], 2)
            self.assertEqual(info["local_num_kv_heads"], 1)
            self.assertEqual(info["local_inter_dim"], 32)
            self.assertEqual(
                info["tensor_parallel_query_head_range"],
                [rank * 2, rank * 2 + 2],
            )
            self.assertEqual(
                info["tensor_parallel_kv_head_range"],
                [rank, rank + 1],
            )
            self.assertEqual(
                info["external_collectives"],
                [
                    "attention_output_all_reduce",
                    "ffn_down_output_all_reduce",
                ],
            )

        for tensor in (
            "W_q_0",
            "W_o_0",
            "W_gate_0",
            "W_up_0",
            "W_down_0",
        ):
            self.assertEqual(
                global_rank["hbm_sizes"][tensor],
                2 * rank_zero["hbm_sizes"][tensor],
            )
        self.assertEqual(
            sum(
                size
                for name, size in global_rank["hbm_sizes"].items()
                if name.startswith("W_k_0_h")
            ),
            2 * rank_zero["hbm_sizes"]["W_k_0_h0"],
        )
        self.assertEqual(
            sum(
                size
                for name, size in global_rank["hbm_sizes"].items()
                if name.startswith("W_v_0_h")
            ),
            2 * rank_zero["hbm_sizes"]["W_v_0_h0"],
        )

    def test_parallel_degree_one_is_an_exact_compiler_identity(self) -> None:
        defaults = self._external_decode_trace(
            bind_parallel_arguments=False,
        )
        explicit = self._external_decode_trace(
            tensor_parallel_degree=1,
            tensor_parallel_rank=0,
            kv_parallel_degree=1,
            kv_parallel_rank=0,
            kv_head_reuse=True,
        )

        self.assertEqual(defaults["isa"], explicit["isa"])
        self.assertEqual(
            defaults["compilation_artifact"].to_dict(),
            explicit["compilation_artifact"].to_dict(),
        )
        self.assertEqual(defaults["hbm_addrs"], explicit["hbm_addrs"])
        self.assertEqual(defaults["hbm_sizes"], explicit["hbm_sizes"])
        self.assertEqual(
            defaults["tensor_layouts"],
            explicit["tensor_layouts"],
        )
        self.assertEqual(
            defaults["comparison_params"],
            explicit["comparison_params"],
        )

    def test_round_robin_kv_rank_uses_local_tail_and_append_addresses(self) -> None:
        cases = (
            (16, 3, 4, 3, 16),
            (17, 0, 5, 4, 16),
            (64, 3, 16, 15, 32),
            (65, 0, 17, 16, 32),
        )
        for context, rank, local_context, local_tail, cache_rows in cases:
            with self.subTest(context=context, rank=rank):
                result = self._external_decode_trace(
                    context_tokens=context,
                    kv_parallel_degree=4,
                    kv_parallel_rank=rank,
                    kv_head_reuse=False,
                )
                artifact = result["compilation_artifact"]
                artifact.request_memory.validate_trace(
                    artifact.execution_trace
                )
                info = result["info"]
                self.assertEqual(info["decode_context_tokens"], context)
                self.assertEqual(
                    info["local_decode_context_tokens"],
                    local_context,
                )
                self.assertEqual(info["local_cache_position"], local_tail)
                self.assertTrue(info["owns_current_kv_token"])
                self.assertTrue(info["kv_append_enabled"])
                self.assertEqual(info["cache_rows_per_batch"], cache_rows)
                self.assertEqual(
                    info["external_collectives"],
                    ["attention_logsumexp_reduce"],
                )

                bindings = {
                    binding.trace_entry_index: binding
                    for binding in artifact.request_memory.bindings
                }
                for cache_name in ("K_cache_0", "V_cache_0"):
                    self.assertEqual(
                        result["tensor_layouts"][cache_name]["storage_shape"],
                        [cache_rows, 16],
                    )
                    reads = []
                    appends = []
                    for index, entry in enumerate(
                        artifact.execution_trace.entries
                    ):
                        if entry.tensor != cache_name or index not in bindings:
                            continue
                        if entry.dma_direction == HBM_READ:
                            reads.extend(bindings[index].runs)
                        elif entry.opcode == "H_STORE_V":
                            appends.extend(bindings[index].runs)
                    self.assertTrue(reads)
                    self.assertEqual(len(appends), 1)
                    base_request = reads[0].request
                    append_request = appends[0].request
                    token_bytes = (
                        append_request.elements_per_row
                        * append_request.element_bits
                        // 8
                    )
                    self.assertEqual(
                        append_request.address,
                        base_request.address + local_tail * token_bytes,
                    )

    def test_round_robin_lowering_rejects_a_non_owner_critical_rank(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "KVP rank that owns the current token",
        ):
            self._external_decode_trace(
                context_tokens=17,
                kv_parallel_degree=4,
                kv_parallel_rank=1,
            )

    def test_vector_dma_uses_physical_subbyte_addresses(self) -> None:
        from compiler.asm_templates.preload_act import preload_act_asm
        from compiler.asm_templates.store_act_asm import store_act_asm

        preload = preload_act_asm(
            vlen=16,
            preload_len=4,
            batch=8,
            hidden_size=32,
            act_vram_offset=0,
            alive_registers=[1, 2, 3, 4, 5],
            activation_offset_reg=1,
            stride_size=32,
            element_bits=2,
            element_plane_bytes=64,
        )
        self.assertIn("C_SET_SCALE_REG gp1", preload)
        self.assertIn("S_ADDI_INT gp2, gp0, 8", preload)
        self.assertIn("S_ADDI_INT gp2, gp2, 32", preload)
        self.assertIn("S_ADDI_INT gp1, gp1, 4", preload)
        self.assertIn("H_PREFETCH_V gp3, gp2, a1, 1, 0", preload)

        store = store_act_asm(
            vlen=16,
            batch=8,
            hidden_size=32,
            alive_registers=[1, 2, 3, 4, 5],
            act_vram_offset=0,
            hbm_addr_reg=1,
            stride_size=32,
            element_bits=2,
            element_plane_bytes=64,
        )
        self.assertIn("C_SET_SCALE_REG gp2", store)
        self.assertIn("S_ADDI_INT gp2, gp0, 8", store)
        self.assertIn("S_ADDI_INT gp2, gp2, 32", store)
        self.assertIn("S_ADDI_INT gp1, gp1, 4", store)

    def test_load_batch_binds_aligned_element_plane(self) -> None:
        from compiler.aten.plena import PlenaCompiler

        compiler = PlenaCompiler(
            mlen=16,
            blen=4,
            hbm_element_width=2,
            hbm_block_size=8,
            hbm_scale_width=8,
        )
        source = compiler.input(
            "A",
            shape=(1, 16),
            physical_shape=(1, 16),
            hbm_element_width=2,
        )
        compiler.load_batch(source)
        assembly = compiler.compile()
        self.assertRegex(
            assembly,
            r"S_ADDI_INT gp\d+, gp0, 32\s*\nC_SET_SCALE_REG",
        )
        self.assertRegex(
            assembly,
            r"H_PREFETCH_V gp\d+, gp\d+, a\d+, 0, 0",
        )

        kv_compiler = PlenaCompiler(
            mlen=16,
            blen=4,
            hbm_element_width=2,
        )
        kv = kv_compiler.input(
            "V",
            shape=(1, 16),
            physical_shape=(1, 16),
            hbm_element_width=2,
            precision_role="value",
        )
        kv_compiler.load_batch(kv)
        self.assertRegex(
            kv_compiler.compile(),
            r"H_PREFETCH_V gp\d+, gp\d+, a\d+, 0, 1",
        )

    def test_qwen_rows_are_aligned_and_reduce_bytes(self) -> None:
        for bits in (2, 4, 8):
            with self.subTest(bits=bits):
                layout = PackedKVLayout(
                    kv_heads=8,
                    head_dim=128,
                    mlen=1024,
                    element_bits=bits,
                )
                self.assertEqual(layout.element_plane_bytes % 64, 0)
                self.assertEqual(layout.scale_plane_bytes % 64, 0)
                self.assertEqual(layout.byte_reduction, 8.0)
                self.assertGreaterEqual(layout.byte_reduction, 7.5)

    def test_pack_unpack_and_selector_offsets(self) -> None:
        layout = PackedKVLayout(kv_heads=8, head_dim=128, mlen=1024)
        heads = tuple(
            tuple(head * 1000 + lane for lane in range(128))
            for head in range(8)
        )
        row = layout.pack_token(heads)
        self.assertEqual(len(row), 1024)
        self.assertEqual(layout.unpack_token(row), heads)
        self.assertEqual(
            [layout.selector_offset_elements(head) for head in range(8)],
            [0, 128, 256, 384, 512, 640, 768, 896],
        )
        self.assertEqual(
            layout.value_row_offset_elements(2, 3),
            2 * 1024 + 3 * 128,
        )

    def test_ablation_keeps_dense_storage_constant(self) -> None:
        layout = PackedKVLayout(kv_heads=8, head_dim=128, mlen=1024)
        metrics = layout.ablation_metrics(tokens=512)
        dense = metrics[PackedKVAblation.DENSE_COMPILER.value]["physical_bytes"]
        selected = metrics[PackedKVAblation.DENSE_SELECTOR.value]["physical_bytes"]
        ideal = metrics[PackedKVAblation.IDEAL_TRAFFIC.value]["physical_bytes"]
        padded = metrics[PackedKVAblation.PADDED_PER_HEAD.value]["physical_bytes"]
        self.assertEqual(dense, selected)
        self.assertEqual(selected, ideal)
        self.assertEqual(padded, 8 * selected)

    def test_inactive_padding_is_zero(self) -> None:
        layout = PackedKVLayout(kv_heads=4, head_dim=128, mlen=1024)
        row = layout.pack_token(tuple((1.0,) * 128 for _ in range(4)))
        self.assertEqual(row[:512], (1.0,) * 512)
        self.assertEqual(row[512:], (0.0,) * 512)

    def test_small_planes_are_physically_aligned(self) -> None:
        layout = PackedKVLayout(kv_heads=2, head_dim=8, mlen=16)
        self.assertEqual(layout.logical_element_plane_bytes, 8)
        self.assertEqual(layout.logical_scale_plane_bytes, 2)
        self.assertEqual(layout.element_plane_bytes, 64)
        self.assertEqual(layout.scale_plane_bytes, 64)

    def test_qwen_hbm_allocator_uses_bound_cache_precision(self) -> None:
        from compiler.aten.plena import PlenaCompiler

        layout = PackedKVLayout(
            kv_heads=8,
            head_dim=128,
            mlen=1024,
            element_bits=4,
        )
        compiler = PlenaCompiler(
            mlen=1024,
            blen=8,
            hbm_element_width=layout.element_bits,
            hbm_block_size=layout.block_size,
            hbm_scale_width=layout.scale_bits,
        )
        self.assertEqual(compiler.hbm_tensor_size(1024), layout.packed_row_bytes)
        self.assertEqual(
            compiler.hbm_tensor_size(2 * 1024),
            2 * layout.packed_row_bytes,
        )

    def test_hbm_matrix_offsets_and_strides_are_physical_bytes(self) -> None:
        from compiler.aten.plena.memory import MatrixBlockLayout

        for bits in (2, 4, 8):
            with self.subTest(bits=bits):
                layout = MatrixBlockLayout(
                    name="packed",
                    full_shape=(32, 32),
                    physical_shape=(32, 32),
                    block_size=16,
                    hbm_element_width=bits,
                    hbm_block_size=8,
                    hbm_scale_width=8,
                )
                self.assertEqual(
                    layout.element_stride_bytes(32),
                    32 * bits // 8,
                )
                self.assertEqual(
                    layout.get_sub_block(0, 1).hbm_offset,
                    16 * bits // 8,
                )
                self.assertEqual(
                    layout.get_sub_block(1, 0).hbm_offset,
                    16 * 32 * bits // 8,
                )
                self.assertEqual(
                    layout.element_plane_bytes,
                    32 * 32 * bits // 8,
                )
                self.assertEqual(layout.scale_plane_bytes, 128)
        eight_bit = MatrixBlockLayout(
            name="legacy",
            full_shape=(32, 32),
            block_size=16,
            hbm_element_width=8,
        )
        self.assertEqual(eight_bit.element_offset_bytes(513), 513)
        self.assertEqual(eight_bit.element_stride_bytes(32), 32)

    def test_hbm_subbyte_offsets_reject_partial_bytes(self) -> None:
        from compiler.aten.plena.memory import MatrixBlockLayout

        for bits in (2, 4):
            with self.subTest(bits=bits):
                layout = MatrixBlockLayout(
                    name="tail",
                    full_shape=(16, 24),
                    physical_shape=(16, 24),
                    block_size=16,
                    hbm_element_width=bits,
                    hbm_block_size=8,
                    hbm_scale_width=8,
                )
                self.assertEqual(
                    layout.element_plane_bytes,
                    ((16 * 24 * bits // 8 + 31) // 32) * 32,
                )
                self.assertEqual(layout.scale_plane_bytes, 64)
                with self.assertRaisesRegex(ValueError, "byte boundaries"):
                    layout.element_offset_bytes(1)

    def test_selector_lowering_contract_fails_closed(self) -> None:
        layout = PackedKVLayout(kv_heads=2, head_dim=8, mlen=16)
        validate_selector_lowering(
            layout,
            mlen=16,
            kv_heads=2,
            head_dim=8,
            batch_size=1,
        )
        for batch_size in (2, 4):
            validate_selector_lowering(
                layout,
                mlen=16,
                kv_heads=2,
                head_dim=8,
                batch_size=batch_size,
            )
        with self.assertRaisesRegex(ValueError, "positive"):
            validate_selector_lowering(
                layout,
                mlen=16,
                kv_heads=2,
                head_dim=8,
                batch_size=0,
            )
        with self.assertRaisesRegex(ValueError, "block_size=8"):
            validate_selector_lowering(
                PackedKVLayout(
                    kv_heads=1,
                    head_dim=16,
                    mlen=16,
                    block_size=16,
                ),
                mlen=16,
                kv_heads=1,
                head_dim=16,
                batch_size=1,
            )
        with self.assertRaisesRegex(ValueError, "2-, 4-, or 8-bit"):
            validate_selector_lowering(
                PackedKVLayout(
                    kv_heads=2,
                    head_dim=8,
                    mlen=16,
                    element_bits=6,
                ),
                mlen=16,
                kv_heads=2,
                head_dim=8,
                batch_size=1,
            )
        with self.assertRaisesRegex(
            ValueError,
            r"local_kv_heads=17, selector_limit=16",
        ):
            validate_selector_lowering(
                PackedKVLayout(
                    kv_heads=17,
                    head_dim=8,
                    mlen=136,
                ),
                mlen=136,
                kv_heads=17,
                head_dim=8,
                batch_size=1,
            )

    def test_tensor_parallelism_can_make_packed_width_locally_legal(self) -> None:
        result = self._external_decode_trace(
            hidden_size=32,
            num_heads=4,
            num_kv_heads=4,
            head_dim=8,
            intermediate_size=64,
            mlen=16,
            tensor_parallel_degree=2,
            tensor_parallel_rank=0,
            layout_kv_heads=2,
        )
        self.assertEqual(result["info"]["num_kv_heads"], 4)
        self.assertEqual(result["info"]["local_num_kv_heads"], 2)
        self.assertEqual(
            result["info"]["local_packed_kv_active_elements"],
            16,
        )
        with self.assertRaisesRegex(
            ValueError,
            r"PackedKV has 4 heads, expected 2",
        ):
            self._external_decode_trace(
                hidden_size=32,
                num_heads=4,
                num_kv_heads=4,
                head_dim=8,
                intermediate_size=64,
                mlen=32,
                tensor_parallel_degree=2,
                tensor_parallel_rank=0,
                layout_kv_heads=4,
            )

    def test_frontend_explicitly_dispatches_to_packed_cache(self) -> None:
        source = (
            Path(__file__).resolve().parents[1] / "plena_frontend.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)
        functions = {
            node.name: node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        compile_fn = functions["compile_native_hf_decoder"]
        self.assertIn("packed_kv_layout", [arg.arg for arg in compile_fn.args.args])
        attention_fn = functions["_emit_packed_attention_block"]
        calls = [
            node
            for node in ast.walk(attention_fn)
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "flash_attention_packed_cache"
            )
        ]
        self.assertEqual(len(calls), 1)
        self.assertTrue(
            {"batch_size", "rows_per_batch", "query_rows_per_batch"}
            <= {keyword.arg for keyword in calls[0].keywords}
        )

    def test_compiler_emits_literal_head_selectors(self) -> None:
        from compiler.aten.plena import PlenaCompiler

        compiler = PlenaCompiler(mlen=16, blen=4)
        compiler.hlen = 8
        compiler.broadcast_amount = 2
        query = compiler.alloc(
            "Q",
            4,
            32,
            strict=False,
            physical_shape=(16, 32),
        )
        output = compiler.alloc(
            "O",
            4,
            32,
            strict=False,
            physical_shape=(16, 32),
        )
        scratch = compiler.alloc("S", 16 * 5, 16, strict=True)
        key = compiler.input(
            "K_packed",
            shape=(4, 16),
            physical_shape=(16, 16),
        )
        value = compiler.input(
            "V_packed",
            shape=(4, 16),
            physical_shape=(16, 16),
        )

        compiler.flash_attention_packed_cache(
            query,
            key,
            value,
            num_kv_heads=2,
            group_heads=2,
            head_slot_dim=8,
            output_base_address=compiler.get_vram_addr(output.name),
            scratch_base_address=compiler.get_vram_addr(scratch.name),
            broadcast_amount=2,
            causal_mask=False,
        )
        assembly = compiler.compile()
        self.assertIn("M_BTMM 0,", assembly)
        self.assertIn("M_BTMM 1,", assembly)
        selector_one = assembly.index("M_BTMM 1,")
        warm_start = assembly.index("; === Warm V", selector_one)
        warm_end = assembly.index("H_PREFETCH_M", warm_start)
        self.assertRegex(
            assembly[warm_start:warm_end],
            r"S_ADDI_INT gp\d+, gp0, 8",
        )

    def test_compiler_keeps_packed_batch_slabs_independent(self) -> None:
        from compiler.aten.plena import PlenaCompiler

        for batch_size in (2, 4):
            with self.subTest(batch_size=batch_size):
                compiler = PlenaCompiler(mlen=16, blen=4)
                compiler.hlen = 8
                compiler.broadcast_amount = 2
                physical_rows = batch_size * 16
                query = compiler.alloc(
                    "Q",
                    batch_size,
                    32,
                    strict=False,
                    physical_shape=(physical_rows, 32),
                )
                output = compiler.alloc(
                    "O",
                    batch_size,
                    32,
                    strict=False,
                    physical_shape=(physical_rows, 32),
                )
                scratch = compiler.alloc("S", 16 * 5, 16, strict=True)
                key = compiler.input(
                    "K_packed",
                    shape=(batch_size * 8, 16),
                    physical_shape=(physical_rows, 16),
                )
                value = compiler.input(
                    "V_packed",
                    shape=(batch_size * 8, 16),
                    physical_shape=(physical_rows, 16),
                )
                q_base = compiler.get_vram_addr(query.name)
                o_base = compiler.get_vram_addr(output.name)

                compiler.flash_attention_packed_cache(
                    query,
                    key,
                    value,
                    num_kv_heads=2,
                    group_heads=2,
                    head_slot_dim=8,
                    output_base_address=o_base,
                    scratch_base_address=compiler.get_vram_addr(scratch.name),
                    broadcast_amount=2,
                    causal_mask=False,
                    valid_cols=8,
                    cache_position=7,
                    batch_size=batch_size,
                    rows_per_batch=16,
                    query_rows_per_batch=1,
                )
                assembly = compiler.compile()
                self.assertEqual(assembly.count("M_BTMM 0,"), 2 * batch_size)
                self.assertEqual(assembly.count("M_BTMM 1,"), 2 * batch_size)
                for head in range(2):
                    marker = (
                        f"; === VRAM Matrix Add: _packed_S_h{head}"
                    )
                    search_from = 0
                    seen = 0
                    while True:
                        mask_pos = assembly.find(marker, search_from)
                        if mask_pos < 0:
                            break
                        scale_pos = assembly.rfind(
                            "; Tile Row Mul FP on VRAM",
                            0,
                            mask_pos,
                        )
                        softmax_pos = assembly.find(
                            f"; === Online Softmax Block _packed_S_h{head}",
                            mask_pos,
                        )
                        self.assertGreaterEqual(scale_pos, 0)
                        self.assertGreater(softmax_pos, mask_pos)
                        self.assertLess(scale_pos, mask_pos)
                        search_from = softmax_pos + 1
                        seen += 1
                    self.assertEqual(seen, batch_size * 2)

                batch_stride = 16 * 16
                group_stride = physical_rows * 16
                for batch_idx in range(batch_size):
                    for selector in range(2):
                        marker = (
                            f"; PackedKV batch {batch_idx}, selector {selector}, "
                            f"K block {batch_idx}"
                        )
                        self.assertEqual(assembly.count(marker), 1)
                        section_start = assembly.index(marker)
                        next_start = assembly.find("; PackedKV batch ", section_start + 1)
                        section = assembly[
                            section_start : next_start if next_start >= 0 else None
                        ]
                        q_addr = (
                            q_base
                            + selector * group_stride
                            + batch_idx * batch_stride
                        )
                        self.assertIn(
                            f"(1, 16) at VRAM[{q_addr}]",
                            section,
                        )
                        o_addr = (
                            o_base
                            + selector * group_stride
                            + batch_idx * batch_stride
                        )
                        self.assertIn(
                            f"matrix at {o_addr}",
                            section,
                        )
                        warm_start = section.index("; === Warm V")
                        warm_end = section.index("H_PREFETCH_M", warm_start)
                        v_offset = batch_idx * 16 * 16 + selector * 8
                        self.assertRegex(
                            section[warm_start:warm_end],
                            rf"S_ADDI_INT gp\d+, gp0, {v_offset}",
                        )

    def test_q1_decode_covers_every_cache_tile(self) -> None:
        from compiler.aten.plena import PlenaCompiler

        batch_size = 2
        compiler = PlenaCompiler(mlen=16, blen=4)
        compiler.hlen = 8
        compiler.broadcast_amount = 2
        query_rows = batch_size * 16
        cache_rows = batch_size * 48
        query = compiler.alloc(
            "Q",
            batch_size,
            32,
            strict=False,
            physical_shape=(query_rows, 32),
        )
        output = compiler.alloc(
            "O",
            batch_size,
            32,
            strict=False,
            physical_shape=(query_rows, 32),
        )
        scratch = compiler.alloc("S", 16 * 5, 16, strict=True)
        key = compiler.input(
            "K_packed",
            shape=(batch_size * 33, 16),
            physical_shape=(cache_rows, 16),
        )
        value = compiler.input(
            "V_packed",
            shape=(batch_size * 33, 16),
            physical_shape=(cache_rows, 16),
        )

        compiler.flash_attention_packed_cache(
            query,
            key,
            value,
            num_kv_heads=2,
            group_heads=2,
            head_slot_dim=8,
            output_base_address=compiler.get_vram_addr(output.name),
            scratch_base_address=compiler.get_vram_addr(scratch.name),
            broadcast_amount=2,
            causal_mask=True,
            valid_cols=33,
            cache_position=32,
            batch_size=batch_size,
            rows_per_batch=16,
            query_rows_per_batch=1,
            cache_rows_per_batch=48,
        )
        artifact = compiler.compile_with_trace()
        assembly = artifact.assembly
        # 33 cached tokens at MLEN=16 are two full sequence blocks plus a
        # one-column tail; at BLEN=4 that is 4 + 4 + 1 KV tiles per selector.
        tiles_per_selector = 4 + 4 + 1
        dynamic_selectors = [
            int(scoped.args[0])
            for scoped in iter_loop_scoped_lines(assembly)
            if scoped.opcode == "M_BTMM"
            for _ in range(scoped.multiplicity)
        ]
        self.assertEqual(
            dynamic_selectors.count(0),
            batch_size * tiles_per_selector,
        )
        self.assertEqual(
            dynamic_selectors.count(1),
            batch_size * tiles_per_selector,
        )
        self.assertEqual(
            dynamic_selectors,
            [
                selector
                for _batch in range(batch_size)
                for selector in range(2)
                for _tile in range(tiles_per_selector)
            ],
        )
        # The compaction encodes the two full blocks as one hardware loop and
        # emits the masked tail once, for every batch and selector.
        self.assertEqual(
            assembly.count("; PackedKV compact full-block loop"),
            batch_size * 2,
        )
        self.assertEqual(
            assembly.count(
                "; PackedKV compact masked-tail block, valid columns 1"
            ),
            batch_size * 2,
        )
        self.assertEqual(
            assembly.count(
                "; PackedKV softmax state head 0, base 10, stride 1"
            ),
            batch_size * 2,
        )
        self.assertEqual(
            assembly.count(
                "; PackedKV softmax state head 1, base 13, stride 1"
            ),
            batch_size * 2,
        )
        self.assertIn(
            "; PackedKV batch 1, selector 0, K block 3",
            assembly,
        )
        key_layout = compiler.get_hbm_layout(key.name)
        value_layout = compiler.get_hbm_layout(value.name)
        bindings = {
            binding.trace_entry_index: binding
            for binding in artifact.request_memory.bindings
        }
        executed = {key.name: [], value.name: []}
        for index, entry in enumerate(artifact.execution_trace.entries):
            if entry.tensor not in executed or entry.dma_direction != HBM_READ:
                continue
            executed[entry.tensor].extend(
                request.address
                for request in bindings[index].iter_requests()
            )
        self.assertEqual(
            executed[key.name],
            [
                key_layout.hbm_base_addr
                + key_layout.element_offset_bytes(
                    (batch * 3 + block) * 16 * 16
                )
                for batch in range(batch_size)
                for _selector in range(2)
                for block in range(3)
            ],
        )
        self.assertEqual(
            executed[value.name],
            [
                value_layout.hbm_base_addr
                + value_layout.element_offset_bytes(
                    (batch * 3 + block) * 16 * 16 + selector * 8
                )
                for batch in range(batch_size)
                for selector in range(2)
                for block in range(3)
            ],
        )
        self.assertNotIn("KV-looped", assembly)

    def test_cached_q1_requires_the_exact_tail_position(self) -> None:
        from compiler.aten.plena import PlenaCompiler

        compiler = PlenaCompiler(mlen=16, blen=4)
        compiler.hlen = 8
        compiler.broadcast_amount = 2
        query = compiler.alloc(
            "Q",
            1,
            32,
            strict=False,
            physical_shape=(16, 32),
        )
        output = compiler.alloc(
            "O",
            1,
            32,
            strict=False,
            physical_shape=(16, 32),
        )
        scratch = compiler.alloc("S", 16 * 5, 16, strict=True)
        key = compiler.input(
            "K_packed",
            shape=(8, 16),
            physical_shape=(16, 16),
        )
        value = compiler.input(
            "V_packed",
            shape=(8, 16),
            physical_shape=(16, 16),
        )
        common = {
            "num_kv_heads": 2,
            "group_heads": 2,
            "head_slot_dim": 8,
            "output_base_address": compiler.get_vram_addr(output.name),
            "scratch_base_address": compiler.get_vram_addr(scratch.name),
            "broadcast_amount": 2,
            "causal_mask": False,
            "valid_cols": 8,
            "batch_size": 1,
            "rows_per_batch": 16,
            "query_rows_per_batch": 1,
        }

        with self.assertRaisesRegex(ValueError, "explicit cache_position"):
            compiler.flash_attention_packed_cache(
                query,
                key,
                value,
                **common,
            )
        with self.assertRaisesRegex(ValueError, "valid_cols - 1"):
            compiler.flash_attention_packed_cache(
                query,
                key,
                value,
                cache_position=6,
                **common,
            )

    def test_compiler_rejects_out_of_range_selector(self) -> None:
        from compiler.aten.plena import PlenaCompiler

        compiler = PlenaCompiler(mlen=16, blen=4)
        compiler.hlen = 8
        compiler.broadcast_amount = 2
        query = compiler.alloc(
            "Q",
            4,
            16,
            strict=False,
            physical_shape=(16, 16),
        )
        output = compiler.alloc(
            "O",
            4,
            16,
            strict=False,
            physical_shape=(16, 16),
        )
        scratch = compiler.alloc("S", 16 * 5, 16, strict=True)
        key = compiler.input(
            "K_packed",
            shape=(4, 16),
            physical_shape=(16, 16),
        )
        value = compiler.input(
            "V_packed",
            shape=(4, 16),
            physical_shape=(16, 16),
        )

        with self.assertRaisesRegex(ValueError, "selector"):
            compiler.flash_attention_packed_group(
                query,
                key,
                value,
                group_heads=2,
                head_slot_dim=8,
                output_base_address=compiler.get_vram_addr(output.name),
                scratch_base_address=compiler.get_vram_addr(scratch.name),
                broadcast_amount=2,
                causal_mask=False,
                kv_head_selector=2,
            )

    def test_kv_store_encodes_key_value_precision(self) -> None:
        from compiler.assembler.parser import parse_asm_file
        from compiler.aten.plena import PlenaCompiler

        compiler = PlenaCompiler(mlen=16, blen=4)
        cache = compiler.alloc(
            "KV",
            4,
            16,
            strict=False,
            physical_shape=(16, 16),
        )
        compiler.store(cache, name="KV_stored", precision=1)
        with tempfile.TemporaryDirectory() as temporary:
            assembly_path = Path(temporary) / "kv-store.asm"
            assembly_path.write_text(compiler.compile(), encoding="utf-8")
            stores = [
                instruction
                for instruction in parse_asm_file(str(assembly_path))
                if instruction.opcode == "H_STORE_V"
            ]
        self.assertTrue(stores)
        self.assertTrue(all(instruction.funct1 == 1 for instruction in stores))


if __name__ == "__main__":
    unittest.main()
