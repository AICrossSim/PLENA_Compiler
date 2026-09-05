"""Stream full-dimension, synthetic-value MoE fixtures through the real codec.

The NumPy oracle vectorizes independent outputs only: every dot product still
uses ascending K, separate FP32 multiply/add, and the V0 BF16 boundaries.
No BLAS summation order or dense model inference is implied.
"""
from __future__ import annotations
import hashlib
import inspect
import math
from pathlib import Path
import numpy as np
import torch
import moe_normal_export as base


def round_bf16(values):
    values = np.asarray(values, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("nonfinite FP32 intermediate")
    bits = values.view(np.uint32)
    rounded = ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
    result = (rounded.astype(np.uint32) << 16).view(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("BF16 overflow")
    return result


def generated_matrix(rows, cols, seed, activation=False):
    r = np.arange(rows, dtype=np.uint32)[:, None]
    c = np.arange(cols, dtype=np.uint32)[None, :]
    mixed = ((r + 1) * np.uint32(2654435761)) ^ ((c + seed + 1) * np.uint32(2246822519))
    mixed ^= mixed >> 13
    values = (mixed % 31).astype(np.float32) - 15
    scale = 1 / 64 if activation else 2.0 ** (-math.ceil(math.log2(math.sqrt(cols))) - 5)
    return values * np.float32(scale)


def pack_e4m3_vectorized(signed_exponent, signed_mantissa):
    """Same arithmetic as plena_quant.pack_fp_to_bin, with a batched check.

    The reference packer's per-element Python assertions dominate full weight
    export. This replaces only that loop, retaining its FP32 packing arithmetic.
    Compatibility is tested over every finite E4M3 encoding and quantized tails.
    """
    exponent = signed_exponent + 7
    if not torch.isfinite(exponent).all() or not torch.isfinite(signed_mantissa).all():
        raise ValueError('nonfinite codec fields')
    if torch.any((exponent < 0) | (exponent > 15)):
        raise ValueError('exponent outside E4M3')
    sign = (signed_mantissa < 0).to(torch.int64)
    mantissa = torch.abs(signed_mantissa)
    fraction = torch.where(exponent == 0, mantissa, mantissa - 1) * 8
    return (sign * 128 + exponent * 8 + fraction).int()


def encode_array(values):
    from plena_quant.mxfp.quantizer import _mx_fp_quantize_hardware
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or min(values.shape) == 0 or not np.isfinite(values).all():
        raise ValueError("weights require a finite nonempty matrix")
    n, k = values.shape
    stride = ((k + 7) // 8) * 8
    padded = np.pad(values, ((0, 0), (0, stride-k)))
    with torch.no_grad():
        _, exp, man, scales = _mx_fp_quantize_hardware(
            torch.from_numpy(padded), width=8, exponent_width=4,
            exponent_bias_width=8, block_size=[8])
        elements = pack_e4m3_vectorized(exp, man)
    elements = elements.cpu().numpy().astype(np.uint8).reshape(n, stride)
    raw_scales = scales.cpu().numpy().reshape(n, stride // 8)
    if np.any((raw_scales < 0) | (raw_scales > 254)) or np.any((elements & 0x78) == 0x78):
        raise ValueError("nonfinite encoded weight")
    scales = raw_scales.astype(np.uint8)
    # Independently decode using the scalar reference's format definitions.
    lookup = np.array([base.decode_element(i, 127) if (i & 0x78) != 0x78 else 0
                       for i in range(256)], dtype=np.float32)
    scale_values = np.ldexp(np.ones(scales.shape, dtype=np.float32), scales.astype(np.int32)-127)
    scale_values[scales == 0] = 0
    decoded = round_bf16(lookup[elements] * np.repeat(scale_values, 8, axis=1))[:, :k].copy()
    return elements.tobytes(), scales.tobytes(), decoded, dict(
        rows=n, cols=k, element_row_stride=stride, scale_row_stride=stride // 8)


def gemm_reference(x, weights):
    x, weights = np.asarray(x, dtype=np.float32), np.asarray(weights, dtype=np.float32)
    if x.shape[1] != weights.shape[1]:
        raise ValueError("GEMM reduction dimensions differ")
    accum = np.zeros((x.shape[0], weights.shape[0]), dtype=np.float32)
    product = np.empty_like(accum)
    for k in range(x.shape[1]):
        np.multiply(x[:, k, None], weights[None, :, k], out=product)
        np.add(accum, product, out=accum)
    return round_bf16(accum)


def swiglu_reference(gate, up):
    # Match the scalar Python oracle's double exp -> FP32 boundary. Vector
    # library approximations to exp need not have identical rounding.
    exponential = np.fromiter((base.f32(math.exp(-float(g))) for g in gate.flat),
                              dtype=np.float32, count=gate.size).reshape(gate.shape)
    return round_bf16(np.multiply(np.divide(gate, np.add(np.float32(1), exponential)), up))


def export_full_shape(output_dir, *, input_dim, expert_hidden_dim, tokens,
                      routes, name, provenance=None):
    """Generate synthetic values at actual dimensions; routes are caller data.

    The regenerated FP32 arrays are bound by a streaming SHA-256 and the exact
    generator source. Only active routed experts are exported. No shared model
    gate/expert, router, residual or full model execution is represented.
    """
    if min(input_dim, expert_hidden_dim, tokens) <= 0:
        raise ValueError("positive dimensions and tokens are required")
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    ids = {r['expert'] for r in routes}
    if not ids:
        raise ValueError("full-shape campaign requires active experts")
    groups = base.group_routes(routes, tokens, ids)
    normalized = sorted([r for g in groups for r in g['routes']], key=lambda r:(r['token'],r['slot']))
    x_raw = generated_matrix(tokens, input_dim, 91, activation=True)
    x = round_bf16(x_raw)
    source_hash = hashlib.sha256(x_raw.tobytes())
    route_index = {(r['token'],r['slot']):i for i,r in enumerate(normalized)}
    contributions = np.zeros((len(normalized), input_dim), dtype=np.float32)
    experts = []
    image_path = output / 'weights.bin'
    image_hash = hashlib.sha256()
    with image_path.open('wb') as image:
        def append(payload):
            padding = bytes((-image.tell()) % 64)
            image.write(padding); image_hash.update(padding)
            address = image.tell(); image.write(payload); image_hash.update(payload)
            return address
        for group in groups:
            expert = group['expert']; views = {'id':expert}; decoded = {}
            for stage, n, k, offset in [('gate',expert_hidden_dim,input_dim,1),
                                        ('up',expert_hidden_dim,input_dim,2),
                                        ('down',input_dim,expert_hidden_dim,3)]:
                values = generated_matrix(n, k, expert*3+offset)
                source_hash.update(values.tobytes())
                elements, scales, decoded[stage], shape = encode_array(values)
                views[stage] = dict(shape, element_base=append(elements), scale_base=append(scales))
            inputs = x[[r['token'] for r in group['routes']]]
            gate = gemm_reference(inputs, decoded['gate'])
            up = gemm_reference(inputs, decoded['up'])
            y = gemm_reference(swiglu_reference(gate, up), decoded['down'])
            for r, values in zip(group['routes'], y):
                contributions[route_index[(r['token'],r['slot'])]] = values
            experts.append(views)
        append(b'')
    sums = np.zeros_like(x)
    for r, values in zip(normalized, contributions):
        sums[r['token']] += np.float32(r['weight']) * values
    final = round_bf16(sums)
    import plena_quant.mxfp.quantizer as quantizer
    import plena_quant.mxfp.utils as packer
    import plena_quant.common.minifloat as minifloat
    import plena_quant.common.hardware_utils as hardware_utils
    import plena_quant.common.utils as common_utils
    sources = [Path(__file__), Path(base.__file__), *(Path(inspect.getfile(m)) for m in
               [quantizer, packer, minifloat, hardware_utils, common_utils])]
    metadata = dict(weight_format=base.FORMAT, weight_layout='output_major_N_K',
        block_size=8, scale_axis='per_row_reduction_K', hbm_sha256=image_hash.hexdigest(),
        synthetic_source_arrays_sha256=source_hash.hexdigest(), quantizer_torch_version=torch.__version__,
        sources=[dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sources],
        evidence_scope='full matrix dimensions; synthetic inputs/weights; caller fixed routed experts; not full-model inference',
        provenance=provenance or {})
    workload = dict(schema_version=1, name=name, input_dim=input_dim, expert_hidden_dim=expert_hidden_dim,
        hbm_file='weights.bin', inputs_bf16=(x.view(np.uint32)>>16).tolist(), routes=normalized,
        experts=experts, shared_expert=None, grouped_routes=groups, metadata=metadata)
    workload_bytes = base._json_bytes(workload)
    golden = dict(schema_version=1, name=name, semantics='ascending K separate FP32 multiply/add; BF16 stages; scalar exp; stable token/slot combine',
        output_bf16=(final.view(np.uint32)>>16).tolist(), output_f32=final.tolist(),
        pre_round_output_f32=sums.tolist(), workload_sha256=hashlib.sha256(workload_bytes).hexdigest(),
        hbm_sha256=image_hash.hexdigest())
    (output/'workload.json').write_bytes(workload_bytes)
    (output/'golden.json').write_bytes(base._json_bytes(golden))
    return workload, golden
