@host
@_surface_signature(M2_ROPE_HOST_ARGUMENTS, M2_ROPE_HOST_ANNOTATIONS)
def run_m2_rope(self, *args):
    core_num = args[-1]
    bm = self.dims["bm"]
    MlaPrologKernel(bm, self.nz_active, self.nz_wfront, self.nz_wqb)[
        core_num
    ].run_m2_rope(*args[:-1])
