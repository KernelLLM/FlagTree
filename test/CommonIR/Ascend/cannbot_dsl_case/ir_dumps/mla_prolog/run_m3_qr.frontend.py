@host
@_surface_signature(M3_QR_HOST_ARGUMENTS, M3_QR_HOST_ANNOTATIONS)
def run_m3_qr(self, *args):
    core_num = args[-1]
    bm = self.dims["bm"]
    MlaPrologKernel(bm, self.nz_active, self.nz_wfront, self.nz_wqb)[
        core_num
    ].run_m3_qr(*args[:-1])
