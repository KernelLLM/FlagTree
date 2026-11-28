@host
def _launch_qwen_sparse_attn_block_128_fp8_bf16(
    out: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn_mask: Tensor,
    metadata: Tensor,
    tasks: Tensor,
    pages: Tensor,
    scale: float,
    group: Constexpr[int],
    queries_per_tile: Constexpr[int],
    used_core_num: Constexpr[int],
):
    QwenSparseAttnBlock128(
        group,
        dtypes.float8_e4m3fn,
        dtypes.bfloat16,
        dtypes.bfloat16,
        True,
        queries_per_tile,
    )[used_core_num](out, q, k, v, attn_mask, metadata, tasks, pages, scale)
