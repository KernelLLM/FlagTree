@host
def run(
    self,
    K: Tensor,
    V: Tensor,
    Q: Tensor,
    beta: Tensor,
    g: Tensor,
    qk_exchange: Tensor,
    A_log: Tensor,
    dt_bias: Tensor,
    kr: Tensor,
    gamma: Tensor,
    qd: Tensor,
    w: Tensor,
    mqk: Tensor,
    u: Tensor,
    out: Tensor,
    state_in: Tensor,
    state_out: Tensor,
    identity: Tensor,
    metadata: Tensor,
    scale: float,
    lower_bound: float,
    core_num: int,
):
    op = flash_kda_kernel()
    op[core_num](
        K, V, Q, beta, g, qk_exchange, A_log, dt_bias, kr, gamma, qd, w, mqk, u, out, state_in, state_out, identity,
        metadata, scale, lower_bound,
    )
