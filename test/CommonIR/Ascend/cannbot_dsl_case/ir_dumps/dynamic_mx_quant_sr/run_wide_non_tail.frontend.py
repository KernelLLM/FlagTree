@host
def run_wide_non_tail(
    self, x, y, scales, quant_length: int, width: int, blocks: int, rng_state
):
    op = _new_kernel(self)
    op[blocks].run_wide_non_tail(x, y, scales, quant_length, width, rng_state)
