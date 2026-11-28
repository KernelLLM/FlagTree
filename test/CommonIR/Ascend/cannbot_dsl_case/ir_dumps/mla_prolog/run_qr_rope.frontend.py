@host
@_surface_signature(ROPE_QR_HOST_ARGUMENTS, ROPE_QR_HOST_ANNOTATIONS)
def run_qr_rope(self, *args):
    core_num = args[-1]
    bm = self.dims["bm"]
    MlaPrologKernel(bm, self.nz_active, self.nz_wfront, self.nz_wqb)[
        core_num
    ].run_qr_rope(*args[:-1])
