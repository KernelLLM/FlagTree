@host
def run_wide_non_tail_eager_rng(
    self,
    x,
    y,
    scales,
    quant_length: int,
    width: int,
    blocks: int,
    seed_bits,
    offset_bits,
):
    op = _new_kernel(self)
    op[blocks].run_wide_non_tail(
        x, y, scales, quant_length, width, (seed_bits, offset_bits)
    )
