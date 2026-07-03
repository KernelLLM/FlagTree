import torch
import triton
import triton.language as tl

# Adapted from:
# https://github.com/DeepLink-org/DLBlas/blob/main/dlblas/utils/op_helper.py
# Original project licensed under the MIT License.

@triton.jit
def grouped_launch_diagonal(pid, num_pid_m, num_pid_n, BLOCK_TRESHHOLD: tl.constexpr):
    if (num_pid_m >= BLOCK_TRESHHOLD) and (num_pid_n >= BLOCK_TRESHHOLD):
        # 对角线分核代码实现
        curThresholdM = (
            BLOCK_TRESHHOLD
            if pid < (num_pid_m // BLOCK_TRESHHOLD * BLOCK_TRESHHOLD) * num_pid_n
            else num_pid_m % BLOCK_TRESHHOLD
        )
        curThresholdM_thresholdN = curThresholdM * BLOCK_TRESHHOLD
        curThresholdN = (
            BLOCK_TRESHHOLD
            if pid % (num_pid_n * BLOCK_TRESHHOLD)
            < (curThresholdM * num_pid_n)
            // curThresholdM_thresholdN
            * curThresholdM_thresholdN
            else num_pid_n % BLOCK_TRESHHOLD
        )
        localRelativeBlock = (
            pid % (BLOCK_TRESHHOLD * num_pid_n) % (BLOCK_TRESHHOLD * curThresholdM)
        )
        task_m_idx = (
            localRelativeBlock % curThresholdM
            + pid // (BLOCK_TRESHHOLD * num_pid_n) * BLOCK_TRESHHOLD
        )
        # 求最小公倍数，方便求基本块的坐标
        x, y = (
            curThresholdM,
            curThresholdN if curThresholdM > curThresholdN else curThresholdN,
            curThresholdM,
        )
        while y != 0:
            x, y = y, x % y
        lcm = curThresholdM * curThresholdN // x
        task_n_idx = (
            localRelativeBlock + (localRelativeBlock // lcm)
        ) % curThresholdN + pid % (
            BLOCK_TRESHHOLD * num_pid_n
        ) // curThresholdM_thresholdN * BLOCK_TRESHHOLD
    else:
        task_m_idx = pid // num_pid_n
        task_n_idx = pid % num_pid_n
    return task_m_idx, task_n_idx

@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 4}
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 5}
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 6}
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 7}
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 8}
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 4}
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 5}
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 6}
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 7}
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 8}
        ),
    ],
    key=["N", "K"],
)
@triton.jit
def matmul_kernel(
    mat_a,
    mat_b,
    mat_c,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_CORES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_TRESHHOLD: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    NUM_BLOCKS_M = tl.cdiv(M, BLOCK_M)
    NUM_BLOCKS_N = tl.cdiv(N, BLOCK_N)
    NUM_BLOCKS = NUM_BLOCKS_M * NUM_BLOCKS_N
    for block_idx in range(pid, NUM_BLOCKS, NUM_CORES):
        task_m_idx, task_n_idx = grouped_launch_diagonal(
            block_idx, NUM_BLOCKS_M, NUM_BLOCKS_N, BLOCK_TRESHHOLD
        )
        m_start = task_m_idx * BLOCK_M
        n_start = task_n_idx * BLOCK_N
        mat_c_block = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, K, BLOCK_K):
            mat_a_offset = ((m_start + tl.arange(0, BLOCK_M)) * K)[:, None] + (
                k_start + tl.arange(0, BLOCK_K)
            )[None, :]
            mat_a_mask = ((m_start + tl.arange(0, BLOCK_M)) < M)[:, None] & (
                (k_start + tl.arange(0, BLOCK_K)) < K
            )[None, :]
            mat_a_block = tl.load(mat_a + mat_a_offset, mask=mat_a_mask, other=0.0)
            mat_b_offset = ((k_start + tl.arange(0, BLOCK_K)) * N)[:, None] + (
                n_start + tl.arange(0, BLOCK_N)
            )[None, :]
            mat_b_mask = ((k_start + tl.arange(0, BLOCK_K)) < K)[:, None] & (
                (n_start + tl.arange(0, BLOCK_N)) < N
            )[None, :]
            mat_b_block = tl.load(mat_b + mat_b_offset, mask=mat_b_mask, other=0.0)
            mat_c_block = tl.dot(mat_a_block, mat_b_block, mat_c_block)
        mat_c_offset = ((m_start + tl.arange(0, BLOCK_M)) * N)[:, None] + (
            n_start + tl.arange(0, BLOCK_N)
        )[None, :]
        mat_c_mask = ((m_start + tl.arange(0, BLOCK_M)) < M)[:, None] & (
            (n_start + tl.arange(0, BLOCK_N)) < N
        )[None, :]
        tl.store(
            mat_c + mat_c_offset,
            mat_c_block.to(mat_c.dtype.element_ty),
            mask=mat_c_mask,
        )


def call(mat_a, mat_b):
    m = mat_a.shape[0]
    k = mat_a.shape[1]
    n = mat_b.shape[1]
    mat_c = torch.empty(m, n, dtype=mat_a.dtype, device=mat_a.device)
    """
    NPU芯片更加亲和512B对齐场景,如下分块通用性能较好,可以使用autotune选取最优
    BLOCK_M = 128
    BLOCK_N = 256
    BLOCK_K = 256    
    """
    num_cores = get_number_cores()
    matmul_kernel[(num_cores,)](mat_a, mat_b, mat_c, m, n, k, num_cores)
    # print(f"matmul_kernel best config {matmul_kernel.best_config}", flush = True)
    return mat_c

