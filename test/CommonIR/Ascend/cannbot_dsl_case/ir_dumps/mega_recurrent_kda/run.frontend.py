@host
def run(self, out, q, k, v, decay, gate, beta, fa, norm, x, wqkv6, wqkv12, wqkv24, wa, wb6, wb12, wb24, wbeta6,
    wbeta12, wbeta24, wz6, wz12, wz24, gamma, wo6, wo12, wo24, alog, bias, initial, final, indices, accepted, conv,
    conv_indices, conv_accepted, wc, scale, lower, eps, plan_rows, plan_width, plan_policy, launch_cores):
    self.kernel[launch_cores](out, q, k, v, decay, gate, beta, fa, norm, x, wqkv6, wqkv12, wqkv24, wa, wb6,
        wb12, wb24, wbeta6, wbeta12, wbeta24, wz6, wz12, wz24, gamma, wo6, wo12, wo24, alog, bias, initial,
        final, indices, accepted, conv, conv_indices, conv_accepted, wc, scale, lower, eps, plan_rows, plan_width,
        plan_policy)
