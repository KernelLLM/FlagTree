@host
def launch_qwen_sparse_attn_block_128_fp16(
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
        dtypes.float16,
        dtypes.float16,
        dtypes.float16,
        False,
        queries_per_tile,
    )[used_core_num](out, q, k, v, attn_mask, metadata, tasks, pages, scale)
