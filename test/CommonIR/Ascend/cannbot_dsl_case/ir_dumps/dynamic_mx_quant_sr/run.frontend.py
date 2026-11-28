@host
def run(self, x, y, scales, quant_length: int, blocks: int, rng_state):
    op = _new_kernel(self, tail_axis=True)
    op[blocks](x, y, scales, quant_length, rng_state)
