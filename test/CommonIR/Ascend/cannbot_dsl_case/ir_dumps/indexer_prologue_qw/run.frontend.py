@host
def run(
    self,
    x_gm: Tensor,
    qr_gm: Tensor,
    wqb_gm: Tensor,
    ww_gm: Tensor,
    descale_qr_gm: Tensor,
    descale_wqb_gm: Tensor,
    rope_sin_gm: Tensor,
    rope_cos_gm: Tensor,
    lut_gm: Tensor,
    q_gm: Tensor,
    descale_q_gm: Tensor,
    w_gm: Tensor,
    workspace_gm: Tensor,
    t_live: I64,
    softmax_scale: F32,
):
    p = self.p
    # The grid, read back out of the marker rows the host put on the end of
    # ``workspace``.  Symbolic, so it is not in the binary key -- a host integer here
    # is what used to make the core count a specialisation, since the block
    # dim rides along in the verified IR that key is taken over.
    #
    # It has to be the core count and not the work count: one block per work
    # item deadlocks as soon as the work outruns the chip, because the surplus
    # blocks are scheduled in a second wave while the W reduction rendezvous
    # waits on blocks that have not launched.  T=1024 is 8 tiles x 16 head
    # blocks = 128 blocks on a 64-AIV part, and it hangs to an aicore timeout.
    if const_expr(p.w_workspace):
        tiles = (x_gm.shape[0] + (p.base_m - 1)) // p.base_m
        grid = workspace_gm.shape[0] - tiles * p.n_head_blocks * p.base_m
    else:
        grid = workspace_gm.shape[0]
    IndexerPrologueQwKernel(
        p.t, p.dim, p.q_lora, p.n_heads, p.d, p.dr, p.cube_cores, p.t_dyn
    )[grid](
        x_gm,
        qr_gm,
        wqb_gm,
        ww_gm,
        descale_qr_gm,
        descale_wqb_gm,
        rope_sin_gm,
        rope_cos_gm,
        lut_gm,
        q_gm,
        descale_q_gm,
        w_gm,
        workspace_gm,
        t_live,
        softmax_scale,
    )
