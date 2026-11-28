@host
def run_wide8_non_tail_graph_rng(
    self,
    x,
    y,
    scales,
    quant_length: int,
    width: int,
    blocks: int,
    seed_tensor,
    offset_tensor,
    intragraph,
):
    op = _new_kernel(self)
    op[blocks].run_wide8_non_tail(
        x, y, scales, quant_length, width, (seed_tensor, offset_tensor, intragraph)
    )
