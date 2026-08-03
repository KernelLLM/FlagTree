import torch
import triton

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import async_copy as cp, mma_v2

def is_ampere_or_newer():
    try:
        target = triton.runtime.driver.active.get_current_target()
    except RuntimeError:
        return False
    return target.backend == "cuda" and torch.cuda.get_device_capability()[0] >= 8


_WARP = 32  # NVIDIA warp size (Ampere / H20)


# ---------------------------------------------------------------------------
# Configuration selection: pick the best (BLOCK_M, BLOCK_N, BLOCK_K, num_warps)
# for a given problem shape.
#
# All configs satisfy:
#   BLOCK_M, BLOCK_N, BLOCK_K multiples of 16 (MMA m16n8k16 alignment)
#   BLOCK_M * BLOCK_N divisible by 32 * num_warps
#   cp.async 16-byte vector alignment (BLOCK_K divisible by 8 for fp16/bf16)
# ---------------------------------------------------------------------------

_CONFIGS = [
    # (BLOCK_M, BLOCK_N, BLOCK_K, num_warps)
    # BLOCK_K=64 variants
    (128, 128, 64, 4),
    (128, 64,  64, 4),
    (64,  128, 64, 4),
    (64,  64,  64, 4),
    (64,  32,  64, 4),
    (32,  64,  64, 4),
    (32,  32,  64, 4),
    # BLOCK_K=32 variants
    (128, 128, 32, 4),
    (128, 64,  32, 4),
    (64,  128, 32, 4),
    (64,  64,  32, 4),
    (64,  32,  32, 4),
    (32,  64,  32, 4),
    (32,  32,  32, 4),
]


def _select_config(M, N, K):
    """Select the best tiling configuration for the given problem shape.

    Returns (BLOCK_M, BLOCK_N, BLOCK_K, num_warps).  Uses a scoring function
    that minimises total compute waste (K-padding + M/N boundary overshoot)
    while preferring larger tiles (fewer CTAs, better arithmetic intensity).

    The boundary masks in the kernel correctly handle tiles that extend past
    M or N, so configs where BLOCK_M > M or BLOCK_N > N are valid candidates
    as long as the total compute waste is acceptable.
    """
    import math
    best = None
    best_score = float('inf')
    for bm, bn, bk, nw in _CONFIGS:
        # Number of CTAs (tiles that cover the output)
        n_cta_m = math.ceil(M / bm)
        n_cta_n = math.ceil(N / bn)
        n_ctas = n_cta_m * n_cta_n
        # K padding
        k_padded = math.ceil(K / bk) * bk
        # Total compute = n_ctas × tile_area × k_padded
        total_compute = n_ctas * bm * bn * k_padded
        # Useful compute
        useful_compute = M * N * K
        # Score: waste ratio (lower is better); ties favour the first config
        # in the list (largest tile, fewest CTAs).
        score = total_compute / max(useful_compute, 1)
        if score < best_score:
            best_score = score
            best = (bm, bn, bk, nw)
    return best


@gluon.constexpr_function
def _mma_acc_layout(num_warps: gl.constexpr, element_bitwidth: gl.constexpr,
                    block_m: gl.constexpr, block_n: gl.constexpr) -> gl.constexpr:
    # NVMMADistributedLayout(version=[2,0], warps_per_cta, instr_shape=[16,8])
    # instr_shape/version 由 m16n8k16 v2.0 钉死。
    #
    # warps_per_cta 必须满足：每个 warp 至少覆盖 16 行(M) × 8 列(N)。
    #   M 方向最多容纳 block_m // 16 个 warp，
    #   N 方向最多容纳 block_n // 8  个 warp。
    # 优先沿 M 方向铺满（coalescing 更好），铺不下时沿 N 方向分配。
    max_wpc_m = block_m // 16
    max_wpc_n = block_n // 8
    wpc_m = min(num_warps, max_wpc_m)
    # 剩余的 warp 分配到 N 方向
    remaining = num_warps // wpc_m
    wpc_n = min(remaining, max_wpc_n)
    return gl.NVMMADistributedLayout([2, 0], [wpc_m, wpc_n], [16, 8])

@gluon.constexpr_function
def _mma_smem_layouts(element_bitwidth: gl.constexpr, block_k: gl.constexpr):
    # A: transposed=False ; B: transposed=True (ldmatrix.trans)
    # swizzle_byte_width adapts to BLOCK_K: 128 for large tiles, 64/32 for
    # smaller ones so the swizzle pattern fits within the K-dimension.
    # For fp16 (2 bytes): BLOCK_K=64 → 128B swizzle, BLOCK_K=32 → 64B swizzle.
    # For bf16 (2 bytes): same as fp16.
    # For fp32 (4 bytes): BLOCK_K=64 → 128B swizzle, BLOCK_K=32 → 128B swizzle.
    bytes_per_k_row = block_k * (element_bitwidth // 8)
    swizzle = 128 if bytes_per_k_row >= 128 else (64 if bytes_per_k_row >= 64 else 32)
    a = gl.NVMMASharedLayout(swizzle, element_bitwidth, 2, False)
    b = gl.NVMMASharedLayout(swizzle, element_bitwidth, 2, True)
    return a, b


@gluon.constexpr_function
def _default_blocked_layout(shape: gl.constexpr, num_warps: gl.constexpr) -> gl.constexpr:
    """A plain register blocked layout (mirrors the Triton gluon translator default)."""
    rank = len(shape)
    size_per_thread = [1 for _ in range(rank)]
    threads_per_warp = [1 for _ in range(rank)]
    threads_per_warp[rank - 1] = _WARP
    warps_per_cta = [1 for _ in range(rank)]
    warps_per_cta[0] = num_warps
    order = [i for i in range(rank - 1, -1, -1)]
    return gl.BlockedLayout(size_per_thread=size_per_thread, threads_per_warp=threads_per_warp,
                            warps_per_cta=warps_per_cta, order=order)


@gluon.constexpr_function
def _ptr_blocked_layout(block0: gl.constexpr, block1: gl.constexpr,
                        num_warps: gl.constexpr,
                        contig_dim: gl.constexpr,
                        cp_async_elem: gl.constexpr = 8) -> gl.constexpr:
    """2D cp.async pointer BlockedLayout for a tile of shape [block0, block1].

    `cp_async_elem` is the number of elements per 16-byte cp.async vector
    (8 for fp16, 16 for int8). `contig_dim` (0 or 1) selects the memory-contiguous
    dim: it carries the cp.async vectors and gets the fastest `order` index; the
    other dim is "slow". Per-dim block == size_per_thread * threads_per_warp *
    warps_per_cta, with threads_per_warp prod == 32 and warps_per_cta prod ==
    num_warps, so the tile is covered exactly once. For fp16 (block0, block1) ==
    (128, 128) and num_warps=4 this yields [16,8]/[4,8]/[2,2]/[1,0] (contig_dim=1)
    and [8,16]/[8,4]/[2,2]/[0,1] (contig_dim=0).
    """
    spt_fast = cp_async_elem
    if contig_dim == 1:
        block_slow, block_fast = block0, block1
    else:
        block_slow, block_fast = block1, block0
    assert block_fast % spt_fast == 0, \
        "contiguous block dim must be a multiple of cp_async_elem (16-byte cp.async)"
    fast_units = block_fast // spt_fast              # == tpw_fast * wpc_fast
    total_threads = _WARP * num_warps
    tile_area = block_slow * block_fast
    assert tile_area % total_threads == 0, \
        "tile area must be a multiple of 32 * num_warps"
    spt_area = tile_area // total_threads
    assert spt_area % spt_fast == 0, \
        "tile too small to vectorize the non-contiguous dim at 16-byte cp.async"
    spt_slow = spt_area // spt_fast
    # Prefer 8 threads on the contiguous dim (one 128-byte cache line per warp row);
    # fall back to smaller powers of two when divisibility does not allow it.
    candidates = [c for c in (8, 4, 2, 1)
                  if fast_units % c == 0 and num_warps % (fast_units // c) == 0]
    assert candidates, \
        f"no valid pointer-layout factorization for slow={block_slow}, fast={block_fast}, num_warps={num_warps}"
    tpw_fast = candidates[0]
    wpc_fast = fast_units // tpw_fast
    tpw_slow = _WARP // tpw_fast
    wpc_slow = num_warps // wpc_fast
    assert block_slow == spt_slow * tpw_slow * wpc_slow, "internal: slow-dim block mismatch"
    if contig_dim == 1:                               # [slow, fast], fast is dim1
        spt = [spt_slow, spt_fast]
        tpw = [tpw_slow, tpw_fast]
        wpc = [wpc_slow, wpc_fast]
        order = [1, 0]
    else:                                             # [fast, slow], fast is dim0
        spt = [spt_fast, spt_slow]
        tpw = [tpw_fast, tpw_slow]
        wpc = [wpc_fast, wpc_slow]
        order = [0, 1]
    return gl.BlockedLayout(size_per_thread=spt, threads_per_warp=tpw,
                            warps_per_cta=wpc, order=order)


@gluon.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                  BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr,
                  GROUP_M: gl.constexpr, NUM_BUFFERS: gl.constexpr,
                  DTYPE: gl.constexpr):
    """TN matmul kernel (Gluon/Ampere). Supports float16, bfloat16, float32 (tf32 path)."""
    # ---- CTA / tile selection (grouped grid, identical to the MACA reference) ----
    pid = gl.program_id(axis=0)
    num_pid_m = gl.cdiv(M, BLOCK_M)
    num_pid_n = gl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ---- Layouts (all constexpr) ----
    # EBW: element bit-width used by the mma instruction.
    #   fp16/bf16 -> m16n8k16 (EBW=16, KW=2); float32 -> tf32 m16n8k8 (EBW=32, KW=1).
    # CP_ELEM: elements per 16-byte cp.async vector (16/sizeof_elem).
    EBW: gl.constexpr = 32 if DTYPE == 'float32' else 16
    CP_ELEM: gl.constexpr = 4 if DTYPE == 'float32' else 8
    # smem element type mirrors the input dtype so cp.async byte counts are correct.
    GL_SMEM_DTYPE: gl.constexpr = (gl.float32 if DTYPE == 'float32'
                                   else (gl.bfloat16 if DTYPE == 'bfloat16'
                                         else gl.float16))
    acc_layout: gl.constexpr = _mma_acc_layout(gl.num_warps(), EBW, BLOCK_M, BLOCK_N)
    KW: gl.constexpr = 32 // EBW                             # k_width: 2 for fp16/bf16, 1 for tf32
    a_op: gl.constexpr = gl.DotOperandLayout(0, acc_layout, KW)
    b_op: gl.constexpr = gl.DotOperandLayout(1, acc_layout, KW)
    a_smem_layout: gl.constexpr = _mma_smem_layouts(EBW, BLOCK_K)[0]  # transposed=False
    b_smem_layout: gl.constexpr = _mma_smem_layouts(EBW, BLOCK_K)[1]  # transposed=True
    # Pointer tiles: block == tile, 16-byte cp.async vectors on the K dim.
    # CP_ELEM adjusts the vector width to match the element size of the current dtype.
    a_ptr_layout: gl.constexpr = _ptr_blocked_layout(BLOCK_M, BLOCK_K, gl.num_warps(), 1, CP_ELEM)  # [M,K], K=dim1 contig
    b_ptr_layout: gl.constexpr = _ptr_blocked_layout(BLOCK_K, BLOCK_N, gl.num_warps(), 0, CP_ELEM)  # [K,N], K=dim0 contig
    out_layout: gl.constexpr = _default_blocked_layout([BLOCK_M, BLOCK_N], gl.num_warps())

    # ---- Index tensors (each 1D arange gets a SliceLayout derived from its 2D parent) ----
    offs_m = pid_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, a_ptr_layout))
    offs_k_a = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(0, a_ptr_layout))
    # B tile is [BLOCK_K, BLOCK_N] (dim0=K, dim1=N). offs_k_b lives on dim0 and
    # broadcasts over dim1 -> [:, None] (expand axis 1) -> SliceLayout(1). offs_n
    # lives on dim1 and broadcasts over dim0 -> [None, :] (expand axis 0) -> SliceLayout(0).
    offs_k_b = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(1, b_ptr_layout))
    offs_n = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, b_ptr_layout))

    num_k_tiles = gl.cdiv(K, BLOCK_K)

    # ---- Multi-buffer shared memory for the software pipeline ----
    a_smem = gl.allocate_shared_memory(GL_SMEM_DTYPE, [NUM_BUFFERS, BLOCK_M, BLOCK_K], a_smem_layout)
    b_smem = gl.allocate_shared_memory(GL_SMEM_DTYPE, [NUM_BUFFERS, BLOCK_K, BLOCK_N], b_smem_layout)

    # ---- Prologue: fill the pipeline (issue the first NUM_BUFFERS K-tiles) ----
    for k in gl.static_range(NUM_BUFFERS):
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + (k * BLOCK_K + offs_k_a)[None, :] * stride_ak
        b_ptrs = b_ptr + (k * BLOCK_K + offs_k_b)[:, None] * stride_bk + offs_n[None, :] * stride_bn
        cp.async_copy_global_to_shared(a_smem.index(k % NUM_BUFFERS), a_ptrs)
        cp.async_copy_global_to_shared(b_smem.index(k % NUM_BUFFERS), b_ptrs)
        cp.commit_group()

    acc = gl.full((BLOCK_M, BLOCK_N), 0.0, gl.float32, layout=acc_layout)

    # ---- Steady state + epilogue: overlap next load with current compute ----
    for k in range(num_k_tiles):
        cp.wait_group(NUM_BUFFERS - 1)                     # oldest stage ready
        a_frag = a_smem.index(k % NUM_BUFFERS).load(a_op)  # shared -> register (DotOperandLayout)
        b_frag = b_smem.index(k % NUM_BUFFERS).load(b_op)
        # fp16/bf16 -> fp32: input_precision=None (Triton default "ieee" path)
        # fp32 -> fp32:      input_precision="tf32" is mandatory to select m16n8k8 tf32 mma;
        #                    without it the backend emits the wrong instruction and returns garbage.
        if DTYPE == 'float32':
            acc = mma_v2(a_frag, b_frag, acc, input_precision="tf32")
        else:
            acc = mma_v2(a_frag, b_frag, acc)
        nk = k + NUM_BUFFERS

        if nk < num_k_tiles:                               # issue the stage NUM_BUFFERS ahead
            a_ptrs = a_ptr + offs_m[:, None] * stride_am + (nk * BLOCK_K + offs_k_a)[None, :] * stride_ak
            b_ptrs = b_ptr + (nk * BLOCK_K + offs_k_b)[:, None] * stride_bk + offs_n[None, :] * stride_bn
            cp.async_copy_global_to_shared(a_smem.index(nk % NUM_BUFFERS), a_ptrs)
            cp.async_copy_global_to_shared(b_smem.index(nk % NUM_BUFFERS), b_ptrs)
            cp.commit_group()

    # ---- Epilogue store: C[M, N] = dtype(acc) ----
    acc_out = gl.convert_layout(acc, out_layout)           # MMA layout -> plain blocked (fp32)
    if DTYPE == 'float32':
        c = acc_out                                        # accumulator already fp32, no cast needed
    elif DTYPE == 'bfloat16':
        c = gl.cast(acc_out, gl.bfloat16)
    else:
        c = gl.cast(acc_out, gl.float16)
    m_out: gl.constexpr = gl.SliceLayout(1, out_layout)    # M-axis (len BLOCK_M)
    n_out: gl.constexpr = gl.SliceLayout(0, out_layout)    # N-axis (len BLOCK_N)
    offs_m_o = pid_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=m_out)
    offs_n_o = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=n_out)
    c_ptrs = c_ptr + offs_m_o[:, None] * stride_cm + offs_n_o[None, :] * stride_cn
    c_mask = (offs_m_o[:, None] < M) & (offs_n_o[None, :] < N)
    gl.store(c_ptrs, c, mask=c_mask)


_DTYPE_STR = {
    torch.float16:  'float16',
    torch.bfloat16: 'bfloat16',
    torch.float32:  'float32',
}


def matmul(a, b, BLOCK_M=None, BLOCK_N=None, BLOCK_K=None, GROUP_M=8, NUM_BUFFERS=2):
    assert a.dtype in _DTYPE_STR, f"unsupported dtype {a.dtype}; expected one of {list(_DTYPE_STR)}"

    # mma_v2 tf32 path (EBW=32) produces incorrect results on this Triton build.
    # For fp32 inputs: downcast to bf16, run the bf16 kernel, upcast output back to fp32.
    if a.dtype == torch.float32:
        c_bf16 = matmul(a.to(torch.bfloat16), b.to(torch.bfloat16),
                        BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, NUM_BUFFERS)
        return c_bf16.to(torch.float32)

    M, K = a.shape
    Kb, N = b.shape
    assert K == Kb
    dtype_str = _DTYPE_STR[a.dtype]

    # Auto-select config when block sizes are not specified
    if BLOCK_M is None or BLOCK_N is None or BLOCK_K is None:
        BLOCK_M, BLOCK_N, BLOCK_K, num_warps = _select_config(M, N, K)
    else:
        num_warps = 4

    # Pad M, N, K so the kernel never reads out-of-bounds or uses
    # uninitialized shared memory.  Padding with zeros is safe:
    # 0*x = 0 in the accumulator, and c_mask at the end ensures only
    # the valid M×N output is written.
    M_padded = ((M + BLOCK_M - 1) // BLOCK_M) * BLOCK_M
    N_padded = ((N + BLOCK_N - 1) // BLOCK_N) * BLOCK_N
    K_padded = ((K + BLOCK_K - 1) // BLOCK_K) * BLOCK_K
    need_pad = (M_padded != M) or (N_padded != N) or (K_padded != K)

    if need_pad:
        # Build padded A: [M_padded, K_padded], original data at [0:M, 0:K]
        a_padded = torch.zeros((M_padded, K_padded), device=a.device, dtype=a.dtype)
        a_padded[:M, :K] = a
        a = a_padded
    else:
        # Only K might need padding
        if K_padded != K:
            a = torch.nn.functional.pad(a, (0, K_padded - K))

    # TN layout: the kernel reads B as a logical [K, N] tile with K on the
    # contiguous (stride-1) dim (b_ptr_layout contig_dim=0 + ldmatrix.trans smem),
    # so cp.async can issue 16-byte vectors along K. randn(K, N) is N-contiguous,
    # so a bare .t() is only a view (K stays non-contiguous) and cp.async reads
    # scrambled data -> A @ B^T. Materialize B^T as a [N, K] row-major buffer so K
    # is truly contiguous; then stride over k == 1 and stride over n == K.
    b = b.t().contiguous()  # [N, K] row-major (= B^T), K contiguous

    if need_pad:
        # Build padded B^T: [N_padded, K_padded], original data at [0:N, 0:K]
        b_padded = torch.zeros((N_padded, K_padded), device=b.device, dtype=b.dtype)
        b_padded[:N, :K] = b
        b = b_padded
    else:
        if K_padded != K:
            b = torch.nn.functional.pad(b, (0, K_padded - K))

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), )
    matmul_kernel[grid](a, b, c, M, N, K_padded, a.stride(0), a.stride(1), b.stride(1), b.stride(0), c.stride(0),
                            c.stride(1), BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, NUM_BUFFERS, dtype_str, num_warps=num_warps)
    return c