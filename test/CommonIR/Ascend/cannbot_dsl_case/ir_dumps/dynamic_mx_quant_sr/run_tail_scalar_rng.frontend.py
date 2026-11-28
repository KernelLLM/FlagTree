@host
def run_tail_scalar_rng(
    self, x, y, scales, quant_length: int, blocks: int, seed_bits, block_offset_bits
):
    op = _new_kernel(self, tail_axis=True)
    op[blocks].run_tail_scalar_rng(
        x, y, scales, quant_length, seed_bits, block_offset_bits
    )
