@host
def run(
    self,
    gm_partial: Tensor,
    gm_delta: Tensor,
    gm_query: Tensor,
    gm_numerator: Tensor,
    gm_max: Tensor,
    gm_sum: Tensor,
    gm_h: Tensor,
    gm_gamma: Tensor,
    rows_per_core: int,
    block_num: int,
):
    op = UpdateKernel(
        self.d,
        self.epsilon,
        self.delta_dtype,
        self.tile_t,
        self.buffer_depth,
        self.norm_eps,
    )
    op[block_num](
        gm_partial,
        gm_delta,
        gm_query,
        gm_numerator,
        gm_max,
        gm_sum,
        gm_h,
        gm_gamma,
        rows_per_core,
    )
