@host
def run(
    self,
    kg: Tensor,
    decay: Tensor,
    qg: Tensor,
    w: Tensor,
    u: Tensor,
    mqk: Tensor,
    output: Tensor,
    state_in: Tensor,
    state_out: Tensor,
    metadata: Tensor,
    batch_head_count: int,
    seq_len: int,
    core_num: int,
):
    op = FlashKdaCpKernel(self.head_dim)
    op[core_num](kg, decay, qg, w, u, mqk, output, state_in, state_out, metadata, batch_head_count, seq_len)
