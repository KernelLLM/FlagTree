@host
def run_fused(
    self,
    latent,
    weight,
    gamma,
    rope_sin,
    rope_cos,
    cache_index,
    cache,
    scale_cache,
    projected,
    epsilon: dtypes.float32,
    cube_block_num,
    vec_block_num,
):
    self.projection.kernel[cube_block_num](latent, weight, projected)
    self.postprocess.kernel[vec_block_num](
        projected,
        rope_sin,
        rope_cos,
        gamma,
        cache_index,
        cache,
        scale_cache,
        epsilon,
    )
