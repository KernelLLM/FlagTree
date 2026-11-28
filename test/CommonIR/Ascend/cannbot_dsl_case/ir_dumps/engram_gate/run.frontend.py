@host
def run(
    self,
    gm_x,
    gm_key,
    gm_value,
    gm_weight,
    gm_out,
    mask: Tensor | None = None,
):
    total_tokens = gm_x.shape[0]
    if const_expr(
        len(gm_x.shape) != 3
        or gm_x.shape[1] != self.hc_mult
        or gm_x.shape[2] != self.dim
    ):
        raise ValueError(
            f"x must have shape [T, {self.hc_mult}, {self.dim}] to match "
            "this EngramGate instance"
        )
    if const_expr(
        tuple(gm_key.shape) != tuple(gm_x.shape)
        or tuple(gm_out.shape) != tuple(gm_x.shape)
        or tuple(gm_value.shape) != (total_tokens, self.dim)
        or tuple(gm_weight.shape) != (self.hc_mult, self.dim)
    ):
        raise ValueError(
            "key and out must match x; value must have shape "
            f"[{total_tokens}, {self.dim}]; weight must have shape "
            f"[{self.hc_mult}, {self.dim}]"
        )
    if const_expr(mask is not None and mask.shape[0] != total_tokens):
        raise ValueError(f"mask must have shape [{total_tokens}] to match x")
    rows = total_tokens * self.hc_mult
    block_num = min(self.max_blocks, rows)
    op = EngramGateKernel(total_tokens, self.hc_mult, self.dim)
    if mask is not None:
        op[block_num](
            gm_x,
            gm_key,
            gm_value,
            gm_weight,
            gm_out,
            self.eps,
            self.clamp_value,
            mask,
        )
    else:
        op[block_num](
            gm_x,
            gm_key,
            gm_value,
            gm_weight,
            gm_out,
            self.eps,
            self.clamp_value,
        )
