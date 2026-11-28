@host
def launch(
    self,
    q: Tensor,
    k: Tensor,
    page_table: Tensor,
    actual_seq: Tensor,
    query_positions: Tensor,
    metadata: Tensor,
    gm_out: Tensor,
    gm_score_workspace: Tensor,
):
    QSAIndexerKernel(self.spec)[self.block_dim](
        q,
        k,
        page_table,
        actual_seq,
        query_positions,
        metadata,
        gm_out,
        gm_score_workspace,
    )
