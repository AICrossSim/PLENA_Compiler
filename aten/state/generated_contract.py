"""Generated from spec/x_state_v2.json; do not edit by hand."""

CONTRACT_SHA256 = "2edcd9b6ceb22428680f27c9262b4c16015b07f30a26b6b4f23b83625b81b3fc"
X_STATE_OPCODE = 61
STATE_ALGORITHMS = {'MAMBA2': 0, 'KDA': 1}
STATE_SUBOPS = {'PRELOAD': 0, 'RESET': 1, 'PREFILL': 2, 'STEP': 3, 'COMMIT': 4, 'EVICT': 5, 'FENCE': 6}
STATE_PRECISIONS = {'FP32': 0, 'BF16': 1, 'FP16': 2, 'MX8_B128': 3}
STATE_PRECISION_BYTES = {0: 4, 1: 2, 2: 2, 3: 1}
STATE_FLAG_BITS = {'LAST_CHUNK': 0, 'WRITE_COMPLETION': 1, 'PROFILE': 2}
STATE_DESCRIPTOR_MAGIC = 844387160
STATE_DESCRIPTOR_VERSION = 2
STATE_DESCRIPTOR_SIZE = 256
STATE_DESCRIPTOR_ALIGNMENT = 64
STATE_STREAMING_SRAM_OFFSET = 4294967295
STATE_COMMON_FIELDS = {'magic': (0, 'u32'), 'version': (4, 'u16'), 'size_bytes': (6, 'u16'), 'algorithm': (8, 'u8'), 'state_precision': (9, 'u8'), 'activation_precision': (10, 'u8'), 'accumulator_precision': (11, 'u8'), 'flags': (12, 'u32'), 'context_id': (16, 'u32'), 'request_id': (20, 'u32'), 'layer_id': (24, 'u32'), 'state_id': (28, 'u32'), 'batch_size': (32, 'u16'), 'num_heads': (34, 'u16'), 'sequence_length': (36, 'u32'), 'token_offset': (40, 'u32'), 'valid_tokens': (44, 'u16'), 'chunk_size': (46, 'u16'), 'state_sram_offset': (48, 'u32'), 'state_bytes': (52, 'u32'), 'conv_state_bytes': (56, 'u32'), 'parameter_precision': (60, 'u8'), 'conv_state_precision': (61, 'u8'), 'reserved0': (62, 'bytes2'), 'input_vram_addr': (64, 'u32'), 'output_vram_addr': (68, 'u32'), 'input_token_stride': (72, 'u32'), 'output_token_stride': (76, 'u32'), 'state_hbm_addr': (80, 'u64'), 'conv_state_hbm_addr': (88, 'u64'), 'state_scale_addr': (96, 'u64'), 'completion_addr': (104, 'u64'), 'dependency_event': (112, 'u32'), 'completion_event': (116, 'u32'), 'state_scale_bytes': (120, 'u32'), 'conv_state_scale_bytes': (124, 'u32')}
STATE_PAYLOAD_FIELDS = {'MAMBA2': {'head_dim': (128, 'u16'), 'state_dim': (130, 'u16'), 'groups': (132, 'u16'), 'conv_kernel': (134, 'u16'), 'xbc_offset': (136, 'u32'), 'dt_offset': (140, 'u32'), 'conv_weight_addr': (144, 'u64'), 'conv_bias_addr': (152, 'u64'), 'a_log_addr': (160, 'u64'), 'dt_bias_addr': (168, 'u64'), 'd_skip_addr': (176, 'u64'), 'parameter_scale_addr': (184, 'u64'), 'dt_min_f32_bits': (192, 'u32'), 'dt_max_f32_bits': (196, 'u32'), 'reserved_payload': (200, 'bytes56')}, 'KDA': {'key_dim': (128, 'u16'), 'value_dim': (130, 'u16'), 'conv_kernel': (132, 'u16'), 'reserved_kda0': (134, 'u16'), 'q_offset': (136, 'u32'), 'k_offset': (140, 'u32'), 'v_offset': (144, 'u32'), 'decay_offset': (148, 'u32'), 'beta_offset': (152, 'u32'), 'reserved_kda1': (156, 'u32'), 'q_conv_weight_addr': (160, 'u64'), 'k_conv_weight_addr': (168, 'u64'), 'v_conv_weight_addr': (176, 'u64'), 'q_conv_bias_addr': (184, 'u64'), 'k_conv_bias_addr': (192, 'u64'), 'v_conv_bias_addr': (200, 'u64'), 'a_log_addr': (208, 'u64'), 'dt_bias_addr': (216, 'u64'), 'parameter_scale_addr': (224, 'u64'), 'output_scale_f32_bits': (232, 'u32'), 'gate_lower_bound_f32_bits': (236, 'u32'), 'reserved_payload': (240, 'bytes16')}}
STATE_COMPLETION_SIZE = 16
STATE_COMPLETION_ALIGNMENT = 64
STATE_COMPLETION_FIELDS = {'status': (0, 'u32'), 'completion_event': (4, 'u32'), 'elapsed_cycles': (8, 'u64')}
STATE_STATUS = {'EMPTY': 0, 'SUCCESS': 1, 'INVALID_DESCRIPTOR': 2, 'UNSUPPORTED_ALGORITHM': 3, 'ADDRESS_ERROR': 4, 'STATE_HAZARD': 5, 'UNSUPPORTED_PRECISION': 6, 'INTERNAL_ERROR': 255}
STATE_NO_EVENT = 4294967295
