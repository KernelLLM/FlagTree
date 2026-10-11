"""Frozen TLE device functions extracted without changing bodies.
Unchanged original source and licensing are retained under tle/.
"""
import triton
import triton.language as tl
from triton.experimental import tle
from triton.language.extra import libdevice
from router_gemm import tle_ext as ext
CACHE_USAGE_THRESHOLD = 0.8
TLE_CLUSTER_SIZE = 2
TLE_REMOTE_BM = 64
TLE_REMOTE_BN = 256
TLE_REMOTE_BK = 64
TLE_REMOTE_NUM_WARPS = 8
TLE_REMOTE_NUM_STAGES = 2
TLE_REMOTE_A_SLOTS = 2

@triton.jit
def prev_multiple_of(a, b):
    return tl.cdiv(a, b) * b - b

@triton.jit
def mm_kernel_general(A, B, C, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr, IS_FP64: tl.constexpr=False):
    pid = ext.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + pid % group_size
    pid_n = pid % width // group_size
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M).to(tl.int64)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N).to(tl.int64)
    rm = rm.to(tl.int64)
    rn = rn.to(tl.int64)
    prev_multiple = prev_multiple_of(K, BLOCK_K)
    if IS_FP64:
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float64)
    else:
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for start_k in range(0, prev_multiple, BLOCK_K):
        rk = (start_k + tl.arange(0, BLOCK_K)).to(tl.int64)
        a = tl.load(A + (ram[:, None] * stride_am + rk[None, :] * stride_ak))
        b = tl.load(B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn))
        if a.dtype != b.dtype:
            a = a.to(C.dtype.element_ty)
            b = b.to(C.dtype.element_ty)
        if IS_FP64:
            acc += tl.dot(a, b, allow_tf32=False)
        else:
            acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)
    rk = (prev_multiple + tl.arange(0, BLOCK_K)).to(tl.int64)
    mask_k = rk < K
    a = tl.load(A + (ram[:, None] * stride_am + rk[None, :] * stride_ak), mask=mask_k[None, :], other=0.0)
    b = tl.load(B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn), mask=mask_k[:, None], other=0.0)
    if a.dtype != b.dtype:
        a = a.to(C.dtype.element_ty)
        b = b.to(C.dtype.element_ty)
    if IS_FP64:
        acc += tl.dot(a, b, allow_tf32=False)
    else:
        acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)
    acc = acc.to(C.dtype.element_ty)
    rm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    rn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    tl.store(C, acc, mask=mask)

@triton.jit
def mm_kernel_syrk(A, C, M, K, stride_am, stride_ak, stride_cm, stride_cn, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    pid_f = pid.to(tl.float32)
    pid_m = tl.floor((tl.sqrt(8.0 * pid_f + 1.0) - 1.0) / 2.0).to(tl.int32)
    tri_start = pid_m * (pid_m + 1) // 2
    pid_m = tl.where(tri_start > pid, pid_m - 1, pid_m)
    next_tri_start = (pid_m + 1) * (pid_m + 2) // 2
    pid_m = tl.where(next_tri_start <= pid, pid_m + 1, pid_m)
    tri_start = pid_m * (pid_m + 1) // 2
    pid_n = pid - tri_start
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_M + tl.arange(0, BLOCK_M)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M).to(tl.int64)
    ran = tl.max_contiguous(tl.multiple_of(rn % M, BLOCK_M), BLOCK_M).to(tl.int64)
    rm = rm.to(tl.int64)
    rn = rn.to(tl.int64)
    acc = tl.zeros((BLOCK_M, BLOCK_M), dtype=tl.float32)
    for start_k in range(0, K, BLOCK_K):
        rk = (start_k + tl.arange(0, BLOCK_K)).to(tl.int64)
        mask_k = rk < K
        a = tl.load(A + (ram[:, None] * stride_am + rk[None, :] * stride_ak), mask=mask_k[None, :], other=0.0)
        b = tl.load(A + (rk[:, None] * stride_ak + ran[None, :] * stride_am), mask=mask_k[:, None], other=0.0)
        acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)
    out = acc.to(C.dtype.element_ty)
    c_ptr = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < M)[None, :]
    tl.store(c_ptr, out, mask=mask)
    if pid_m > pid_n:
        c_t_ptr = C + (rn[:, None] * stride_cm + rm[None, :] * stride_cn)
        mask_t = (rn < M)[:, None] & (rm < M)[None, :]
        tl.store(c_t_ptr, tl.trans(out), mask=mask_t)
