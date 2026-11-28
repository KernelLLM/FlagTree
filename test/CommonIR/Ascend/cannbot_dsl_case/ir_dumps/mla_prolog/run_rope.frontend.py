@host
@_surface_signature(ROPE_HOST_ARGUMENTS, ROPE_HOST_ANNOTATIONS)
def run_rope(self, *args):
    core_num = args[-1]
    bm = self.dims["bm"]
    MlaPrologKernel(bm, self.nz_active, self.nz_wfront, self.nz_wqb)[
        core_num
    ].run_rope(*args[:-1])
