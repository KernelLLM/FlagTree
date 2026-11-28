@host
def launch(
    self,
    out: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    q_descale: Tensor,
    k_descale: Tensor,
    v_descale: Tensor,
    p_scale: Tensor | None,
    sparse_indices: Tensor,
    sparse_seq_len: Tensor,
    metadata: Tensor,
    block_table: Tensor,
    seqused_kv: Tensor,
    cu_seqlens_q: Tensor,
    softmax_scale: float,
    atten_mask: Tensor | None = None,
    softmax_lse: Tensor | None = None,
):
    # Q/scale 统一按 head 取 GM 视图；只调整视图轴，不重排或重新量化。
    # TND 的 token stride 为 N1*128 / N1，NTD 为 128 / 1。
    if const_expr(self.layout_q == "TND"):
        query = permute(query, (1, 0, 2))
        q_descale = permute(q_descale, (1, 0))
    # 输出始终保持公开 TND 布局，不随 Query layout 改变。
    out = permute(out, (1, 0, 2))
    # k_descale 保留公开四维 stride，仅在设备选页后将连续子块转成一维视图。
    op = QuantBlockSparseAttnKernel(
        query.shape[0],
        sparse_seq_len.shape[2],
        query.shape[0] // key.shape[1],
        self.mask_mode,
        self.return_softmax_lse,
        self.shared_l0a,
    )
    op[self.block_dim](
        out,
        query,
        key,
        value,
        q_descale,
        k_descale,
        v_descale,
        p_scale,
        sparse_indices,
        sparse_seq_len,
        metadata,
        block_table,
        seqused_kv,
        cu_seqlens_q,
        softmax_scale,
        atten_mask,
        softmax_lse,
    )
