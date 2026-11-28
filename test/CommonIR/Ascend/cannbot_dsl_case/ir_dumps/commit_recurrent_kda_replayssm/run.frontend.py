@host
def run(
    self,
    state_gm: Tensor,
    replay_u_gm: Tensor,
    replay_k_gm: Tensor,
    replay_decay_gm: Tensor,
    accepted_gm: Tensor,
    launch_blocks: int,
):
    self.kernel[launch_blocks](
        state_gm,
        replay_u_gm,
        replay_k_gm,
        replay_decay_gm,
        accepted_gm,
    )
