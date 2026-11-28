@host
def run_graph_rng(
    self,
    x,
    y,
    scales,
    quant_length: int,
    blocks: int,
    seed_tensor,
    offset_tensor,
    intragraph,
):
    op = _new_kernel(self, tail_axis=True)
    op[blocks](x, y, scales, quant_length, (seed_tensor, offset_tensor, intragraph))
