from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import cdna3

@g.jit
def _router_tiles(A, B, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, pid_m, pid_n, start_k, BOUND, BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr, A_LAYOUT: gl.constexpr, B_LAYOUT: gl.constexpr):
    rows = pid_m * BM + gl.arange(0, BM, layout=gl.SliceLayout(1, A_LAYOUT))
    columns = pid_n * BN + gl.arange(0, BN, layout=gl.SliceLayout(0, B_LAYOUT))
    rows = gl.max_contiguous(gl.multiple_of(rows % M, BM), BM).to(gl.int64)
    columns = gl.max_contiguous(gl.multiple_of(columns % N, BN), BN).to(gl.int64)
    ak = (start_k + gl.arange(0, BK, layout=gl.SliceLayout(0, A_LAYOUT))).to(gl.int64)
    bk = (start_k + gl.arange(0, BK, layout=gl.SliceLayout(1, B_LAYOUT))).to(gl.int64)
    a = cdna3.buffer_load(A, (rows[:, None] * stride_am + ak[None, :] * stride_ak).to(gl.int32), ak[None, :] < BOUND, 0.0)
    b = cdna3.buffer_load(B, (bk[:, None] * stride_bk + columns[None, :] * stride_bn).to(gl.int32), bk[:, None] < BOUND, 0.0)
    return (a, b)

@g.jit
def mm_kernel_general(A, B, C, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr, GROUP_M: gl.constexpr, IS_FP64: gl.constexpr, A_LAYOUT: gl.constexpr, B_LAYOUT: gl.constexpr, OUT_LAYOUT: gl.constexpr, MMA_LAYOUT: gl.constexpr, A_SHARED: gl.constexpr, B_SHARED: gl.constexpr, PIPE_DEPTH: gl.constexpr, K_WIDTH: gl.constexpr):
    gl.static_assert(not IS_FP64, 'Only the historical BF16-to-FP32 router scope is registered')
    pid = gl.program_id(0).to(gl.int64)
    grid_m = gl.cdiv(M, BLOCK_M)
    grid_n = gl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = gl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + pid % group_size
    pid_n = pid % width // group_size
    previous = gl.cdiv(K, BLOCK_K) * BLOCK_K - BLOCK_K
    acc = gl.full((BLOCK_M, BLOCK_N), 0, gl.float32, MMA_LAYOUT)
    a_ring = gl.allocate_shared_memory(A.dtype.element_ty, (PIPE_DEPTH, BLOCK_M, BLOCK_K), A_SHARED)
    b_ring = gl.allocate_shared_memory(B.dtype.element_ty, (PIPE_DEPTH, BLOCK_K, BLOCK_N), B_SHARED)
    for stage in gl.static_range(PIPE_DEPTH):
        prefill_a, prefill_b = _router_tiles(A, B, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, pid_m, pid_n, stage * BLOCK_K, previous, BLOCK_M, BLOCK_N, BLOCK_K, A_LAYOUT, B_LAYOUT)
        a_ring.index(stage).store(prefill_a)
        b_ring.index(stage).store(prefill_b)
    gl.thread_barrier()
    for start_k in range(0, previous, BLOCK_K):
        slot = start_k // BLOCK_K % PIPE_DEPTH
        a = a_ring.index(slot).load(gl.DotOperandLayout(0, MMA_LAYOUT, K_WIDTH))
        b = b_ring.index(slot).load(gl.DotOperandLayout(1, MMA_LAYOUT, K_WIDTH))
        acc = cdna3.mfma(a, b, acc)
        gl.thread_barrier()
        future = start_k + PIPE_DEPTH * BLOCK_K
        an, bn = _router_tiles(A, B, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, pid_m, pid_n, future, previous, BLOCK_M, BLOCK_N, BLOCK_K, A_LAYOUT, B_LAYOUT)
        a_ring.index(slot).store(an)
        b_ring.index(slot).store(bn)
        gl.thread_barrier()
    a, b = _router_tiles(A, B, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, pid_m, pid_n, previous, K, BLOCK_M, BLOCK_N, BLOCK_K, A_LAYOUT, B_LAYOUT)
    a_ring.index(0).store(a)
    b_ring.index(0).store(b)
    gl.thread_barrier()
    a = a_ring.index(0).load(gl.DotOperandLayout(0, MMA_LAYOUT, K_WIDTH))
    b = b_ring.index(0).load(gl.DotOperandLayout(1, MMA_LAYOUT, K_WIDTH))
    acc = cdna3.mfma(a, b, acc)
    values = gl.convert_layout(acc, OUT_LAYOUT).to(C.dtype.element_ty)
    rows = (pid_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, OUT_LAYOUT))).to(gl.int64)
    columns = (pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, OUT_LAYOUT))).to(gl.int64)
    cdna3.buffer_store(values.to(C.dtype.element_ty), C, (rows[:, None] * stride_cm + columns[None, :] * stride_cn).to(gl.int32), (rows < M)[:, None] & (columns < N)[None, :])
