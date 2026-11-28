@host
def run(
    self,
    gm_a: Tensor,
    gm_b: Tensor,
    gm_sa: Tensor,
    gm_sb: Tensor,
    gm_bias: Tensor,
    gm_c: Tensor,
):
    op = MixQuantKernel(self.t, self.has_bias)
    op[self.t.launch_block_count](gm_a, gm_b, gm_sa, gm_sb, gm_bias, gm_c)
