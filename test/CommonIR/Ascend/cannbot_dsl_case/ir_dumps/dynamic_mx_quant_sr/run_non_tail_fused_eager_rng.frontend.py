@host
def run_non_tail_fused_eager_rng(
    self, x0, x1, y, scales, quant_length: int, blocks: int, seed_bits, offset_bits
):
    op = _new_kernel(self)
    op[blocks].run_non_tail_fused(
        x0, x1, y, scales, quant_length, (seed_bits, offset_bits)
    )
