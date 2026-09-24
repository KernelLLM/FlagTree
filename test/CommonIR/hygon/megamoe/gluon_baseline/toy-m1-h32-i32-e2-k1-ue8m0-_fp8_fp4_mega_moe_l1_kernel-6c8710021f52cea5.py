# Gluon implementation exported from the pinned source.
# Original: /public/home/scnethpc2653/lhd/env/projects/hard81_group2_20260911/src/FlagGems/src/flag_gems/fused/fp8_fp4_mega_moe.py
# Provenance: results/gluon/gluon_baseline/toy-m1-h32-i32-e2-k1-ue8m0-_fp8_fp4_mega_moe_l1_kernel-6c8710021f52cea5.json
# Original library copyright and license: see ../../licenses/ and the project README.
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
import hcu_ops as __h81

from importlib import import_module as _import_module

@g.jit
def _fp8_fp4_mega_moe_l1_kernel_120b33aff9(x_ptr, x_scale_ptr, topk_idx_ptr, l1_w_ptr, l1_s_ptr, l1_out_ptr, M: gl.constexpr, H: gl.constexpr, I: gl.constexpr, TOP_K: gl.constexpr, stride_xm, stride_xh, stride_xsm, stride_xsg, stride_topkm, stride_topkk, stride_w_e, stride_w_n, stride_w_kp, stride_s_e, stride_s_n, stride_s_g, stride_out_m, stride_out_k, stride_out_n, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr, SCALE_IS_UE8M0: gl.constexpr):
    """HCU helper SHA256: 64e253bc2ea2d0226c3aa239e6a8d880a9a88457caf7a1c38287b1279c838b63"""
    pid_m = __h81.freeze(gl.program_id(0))
    pid_topk = __h81.freeze(gl.program_id(1))
    pid_n = __h81.freeze(gl.program_id(2))
    n_offsets = __h81.freeze(__h81.to_auto(pid_n) * __h81.to_auto(BLOCK_N) + __h81.arange(0, __h81.to_auto(BLOCK_N), layout=gl.SliceLayout(1, gl.BlockedLayout([1, 1], [16, 4], [1, 4], [0, 1]))), _layout=gl.SliceLayout(1, gl.BlockedLayout([1, 1], [16, 4], [1, 4], [0, 1])))
    k_offsets = __h81.freeze(__h81.arange(0, __h81.to_auto(BLOCK_K), layout=gl.SliceLayout(0, gl.BlockedLayout([1, 1], [16, 4], [1, 4], [0, 1]))), _layout=gl.SliceLayout(0, gl.BlockedLayout([1, 1], [16, 4], [1, 4], [0, 1])))
    expert = __h81.freeze(__h81.load(__h81.to_auto(topk_idx_ptr) + __h81.to_auto(pid_m) * __h81.to_auto(stride_topkm) + __h81.to_auto(pid_topk) * __h81.to_auto(stride_topkk)))
    valid_expert = __h81.freeze(__h81.compare(__h81.to_auto(expert), 0, 'ge'))
    acc = __h81.freeze(__h81.zeros([__h81.to_auto(BLOCK_N)], dtype=gl.float32), _layout=gl.SliceLayout(1, gl.BlockedLayout([1, 1], [16, 4], [1, 4], [0, 1])))
    for k0 in range(0, __h81.to_auto(H), __h81.to_auto(BLOCK_K)):
        ks = __h81.freeze(__h81.to_auto(k0) + __h81.to_auto(k_offsets))
        x = __h81.freeze(__h81.load(__h81.to_auto(x_ptr) + __h81.to_auto(pid_m) * __h81.to_auto(stride_xm) + __h81.to_auto(ks) * __h81.to_auto(stride_xh), mask=__h81.compare(__h81.to_auto(ks), __h81.to_auto(H), 'lt'), other=0.0, _layout=gl.SliceLayout(0, gl.BlockedLayout([1, 1], [16, 4], [1, 4], [0, 1]))).to(gl.float32), _layout=gl.SliceLayout(0, gl.BlockedLayout([1, 1], [16, 4], [1, 4], [0, 1])))
        x_scale = __h81.freeze(__h81.load(__h81.to_auto(x_scale_ptr) + __h81.to_auto(pid_m) * __h81.to_auto(stride_xsm) + __h81.to_auto(ks) // 32 * __h81.to_auto(stride_xsg), mask=__h81.compare(__h81.to_auto(ks), __h81.to_auto(H), 'lt'), other=0.0, _layout=gl.SliceLayout(0, gl.BlockedLayout([1, 1], [16, 4], [1, 4], [0, 1]))).to(gl.float32), _layout=gl.SliceLayout(0, gl.BlockedLayout([1, 1], [16, 4], [1, 4], [0, 1])))
        x = __h81.freeze(__h81.to_auto(x) * __h81.to_auto(x_scale), _layout=gl.SliceLayout(0, gl.BlockedLayout([1, 1], [16, 4], [1, 4], [0, 1])))
        w = __h81.freeze(_load_fp4_weight_4cfbeeb897(l1_w_ptr, l1_s_ptr, expert, n_offsets, ks, stride_w_e, stride_w_n, stride_w_kp, stride_s_e, stride_s_n, stride_s_g, __h81.pin(2 * __h81.to_auto(I)), H, SCALE_IS_UE8M0), _layout=gl.BlockedLayout([1, 1], [16, 4], [1, 4], [0, 1]))
        acc = __h81.freeze(__h81.to_auto(acc) + __h81.to_auto(gl.sum(__h81.concrete(__h81.to_auto(w) * __h81.expand_dims(__h81.to_auto(x), 0), _layout=gl.BlockedLayout([1, 1], [16, 4], [1, 4], [0, 1])), axis=1)), _layout=gl.SliceLayout(1, gl.BlockedLayout([1, 1], [16, 4], [1, 4], [0, 1])))
    out_ptrs = __h81.freeze(__h81.to_auto(l1_out_ptr) + __h81.to_auto(pid_m) * __h81.to_auto(stride_out_m) + __h81.to_auto(pid_topk) * __h81.to_auto(stride_out_k) + __h81.to_auto(n_offsets) * __h81.to_auto(stride_out_n), _layout=gl.BlockedLayout([1], [64], [4], [0]))
    __h81.store(__h81.to_auto(out_ptrs), __h81.to_auto(acc), mask=__h81.to_auto(valid_expert) & __h81.compare(__h81.to_auto(n_offsets), 2 * __h81.to_auto(I), 'lt'), _layout=gl.BlockedLayout([1], [64], [4], [0]))

@g.jit
def zeros_1ed385672d(shape, dtype):
    """HCU helper SHA256: 64e253bc2ea2d0226c3aa239e6a8d880a9a88457caf7a1c38287b1279c838b63"""
    '\n    Returns a tensor filled with the scalar value 0 for the given :code:`shape` and :code:`dtype`.\n\n    :param shape: Shape of the new array, e.g., (8, 16) or (8, )\n    :type shape: tuple of ints\n    :param dtype: Data-type of the new array, e.g., :code:`tl.float16`\n    :type dtype: DType\n    '
    return __h81.full(__h81.to_auto(shape), 0, __h81.to_auto(dtype))

@g.jit
def _load_fp4_weight_4cfbeeb897(packed_ptr, scale_ptr, expert, n_offsets, k_offsets, stride_e, stride_n, stride_kp, scale_stride_e, scale_stride_n, scale_stride_g, N: gl.constexpr, K: gl.constexpr, SCALE_IS_UE8M0: gl.constexpr):
    """HCU helper SHA256: 64e253bc2ea2d0226c3aa239e6a8d880a9a88457caf7a1c38287b1279c838b63"""
    packed_offsets = __h81.freeze(__h81.to_auto(expert) * __h81.to_auto(stride_e) + __h81.expand_dims(__h81.to_auto(n_offsets), 1) * __h81.to_auto(stride_n) + __h81.expand_dims(__h81.to_auto(k_offsets), 0) // 2 * __h81.to_auto(stride_kp), _layout=gl.BlockedLayout([1, 1], [16, 4], [1, 4], [0, 1]))
    packed = __h81.freeze(__h81.load(__h81.to_auto(packed_ptr) + __h81.to_auto(packed_offsets), mask=__h81.compare(__h81.expand_dims(__h81.to_auto(n_offsets), 1), __h81.to_auto(N), 'lt') & __h81.compare(__h81.expand_dims(__h81.to_auto(k_offsets), 0), __h81.to_auto(K), 'lt'), other=0, _layout=gl.BlockedLayout([1, 1], [16, 4], [1, 4], [0, 1])).to(gl.uint8), _layout=gl.BlockedLayout([1, 1], [16, 4], [1, 4], [0, 1]))
    low = __h81.freeze(__h81.to_auto(packed) & 15, _layout=gl.BlockedLayout([1, 1], [16, 4], [1, 4], [0, 1]))
    high = __h81.freeze(__h81.to_auto(packed) >> 4 & 15, _layout=gl.BlockedLayout([1, 1], [16, 4], [1, 4], [0, 1]))
    code = __h81.freeze(gl.where(__h81.compare(__h81.expand_dims(__h81.to_auto(k_offsets), 0) & 1, 0, 'eq'), __h81.to_auto(low), __h81.to_auto(high)), _layout=gl.BlockedLayout([1, 1], [16, 4], [1, 4], [0, 1]))
    values = __h81.freeze(_decode_e2m1_24ca82eab8(code), _layout=gl.BlockedLayout([1, 1], [16, 4], [1, 4], [0, 1]))
    scale_offsets = __h81.freeze(__h81.to_auto(expert) * __h81.to_auto(scale_stride_e) + __h81.expand_dims(__h81.to_auto(n_offsets), 1) * __h81.to_auto(scale_stride_n) + __h81.expand_dims(__h81.to_auto(k_offsets), 0) // 32 * __h81.to_auto(scale_stride_g), _layout=gl.BlockedLayout([1, 1], [16, 4], [1, 4], [0, 1]))
    raw_scale = __h81.freeze(__h81.load(__h81.to_auto(scale_ptr) + __h81.to_auto(scale_offsets), mask=__h81.compare(__h81.expand_dims(__h81.to_auto(n_offsets), 1), __h81.to_auto(N), 'lt') & __h81.compare(__h81.expand_dims(__h81.to_auto(k_offsets), 0), __h81.to_auto(K), 'lt'), other=0, _layout=gl.BlockedLayout([1, 1], [16, 4], [1, 4], [0, 1])), _layout=gl.BlockedLayout([1, 1], [16, 4], [1, 4], [0, 1]))
    scale = __h81.freeze(_ue8m0_to_f32_16e5c50f8c(raw_scale) if __h81.to_auto(SCALE_IS_UE8M0) else __h81.cast(raw_scale, gl.float32), _layout=gl.BlockedLayout([1, 1], [16, 4], [1, 4], [0, 1]))
    return __h81.to_auto(values) * __h81.to_auto(scale)

@g.jit
def sum_f3120590e3(input, axis=None, keep_dims=False, dtype: gl.constexpr=None):
    """HCU helper SHA256: 64e253bc2ea2d0226c3aa239e6a8d880a9a88457caf7a1c38287b1279c838b63"""
    out_dtype: gl.constexpr = _pick_sum_dtype_768abbebc3(__h81.pin(__h81.to_auto(input).dtype), dtype)
    if __h81.to_auto(out_dtype) is not None:
        input = __h81.freeze(__h81.cast(input, __h81.to_auto(out_dtype)), _layout=__h81.layout_of(input))
    return __h81.to_auto(gl.reduce(__h81.concrete(__h81.to_auto(input), _layout=gl.BlockedLayout([1, 1], [16, 4], [1, 4], [0, 1])), __h81.to_auto(axis), _sum_combine_9669e8607d, keep_dims=__h81.to_auto(keep_dims)))

@g.jit
def _decode_e2m1_24ca82eab8(code):
    """HCU helper SHA256: 64e253bc2ea2d0226c3aa239e6a8d880a9a88457caf7a1c38287b1279c838b63"""
    idx = __h81.freeze(__h81.to_auto(code) & 7, _layout=gl.BlockedLayout([1, 1], [16, 4], [1, 4], [0, 1]))
    mag = __h81.freeze(gl.where(__h81.compare(__h81.to_auto(idx), 0, 'eq'), 0.0, gl.where(__h81.compare(__h81.to_auto(idx), 1, 'eq'), 0.5, gl.where(__h81.compare(__h81.to_auto(idx), 2, 'eq'), 1.0, gl.where(__h81.compare(__h81.to_auto(idx), 3, 'eq'), 1.5, gl.where(__h81.compare(__h81.to_auto(idx), 4, 'eq'), 2.0, gl.where(__h81.compare(__h81.to_auto(idx), 5, 'eq'), 3.0, gl.where(__h81.compare(__h81.to_auto(idx), 6, 'eq'), 4.0, 6.0))))))), _layout=gl.BlockedLayout([1, 1], [16, 4], [1, 4], [0, 1]))
    return gl.where(__h81.compare(__h81.to_auto(code) & 8, 0, 'ne'), -__h81.to_auto(mag), __h81.to_auto(mag))

@g.jit
def _ue8m0_to_f32_16e5c50f8c(x):
    """HCU helper SHA256: 64e253bc2ea2d0226c3aa239e6a8d880a9a88457caf7a1c38287b1279c838b63"""
    return gl.exp2(__h81.cast(x, gl.float32) - 127.0)

@g.constexpr_function
def _pick_sum_dtype_768abbebc3(in_dtype, dtype):
    """HCU helper SHA256: 64e253bc2ea2d0226c3aa239e6a8d880a9a88457caf7a1c38287b1279c838b63"""
    if dtype is not None:
        return dtype
    out_dtype = None
    if in_dtype.is_int_signed():
        out_dtype = gl.int32 if in_dtype.int_bitwidth < 32 else None
    elif in_dtype.is_int_unsigned():
        out_dtype = gl.uint32 if in_dtype.int_bitwidth < 32 else None
    return out_dtype

@g.jit
def _sum_combine_9669e8607d(a, b):
    """HCU helper SHA256: 64e253bc2ea2d0226c3aa239e6a8d880a9a88457caf7a1c38287b1279c838b63"""
    return __h81.to_auto(a) + __h81.to_auto(b)

KERNEL = _fp8_fp4_mega_moe_l1_kernel_120b33aff9
