@host
def launch(
    self,
    q: Tensor,
    k: Tensor,
    bias: Tensor,
    metadata: Tensor,
    q_seq_lens: Tensor,
    kv_seq_lens: Tensor,
    num_prompt_tokens: Tensor,
    indices: Tensor,
    lens: Tensor,
    kv_heads: int,
    group_size: int,
    alpha: float,
    block_dim: int,
):
    StemIndexerKernel(
        self.attrs,
        self.topk_score_precision,
    )[block_dim](
        q,
        k,
        bias,
        metadata,
        q_seq_lens,
        kv_seq_lens,
        num_prompt_tokens,
        indices,
        lens,
        kv_heads,
        group_size,
        alpha,
    )
