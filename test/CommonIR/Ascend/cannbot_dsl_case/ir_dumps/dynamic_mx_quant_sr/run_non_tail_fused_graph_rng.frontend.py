@host
def run_non_tail_fused_graph_rng(
    self,
    x0,
    x1,
    y,
    scales,
    quant_length: int,
    blocks: int,
    seed_tensor,
    offset_tensor,
    intragraph,
):
    op = _new_kernel(self)
    op[blocks].run_non_tail_fused(
        x0, x1, y, scales, quant_length, (seed_tensor, offset_tensor, intragraph)
    )
