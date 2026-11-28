@host
def run_non_tail_fused(
    self, x0, x1, y, scales, quant_length: int, blocks: int, rng_state
):
    op = _new_kernel(self)
    op[blocks].run_non_tail_fused(x0, x1, y, scales, quant_length, rng_state)
