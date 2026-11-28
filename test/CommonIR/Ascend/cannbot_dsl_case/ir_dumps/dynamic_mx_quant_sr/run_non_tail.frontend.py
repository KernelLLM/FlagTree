@host
def run_non_tail(
    self, x0, x1, y, scales, quant_length: int, blocks: int, rng_state
):
    op = _new_kernel(self)
    op[blocks].run_non_tail(x0, x1, y, scales, quant_length, rng_state)
