@host
def run(self, gm_x: Tensor, gm_indices: Tensor, gm_y: Tensor):
    """Launch the selected device implementation with its compile-time geometry."""
    op = GatherV2Kernel(self.tiling)
    op[dim3(self.tiling.core_num), dim3(self.tiling.thread_num)](
        gm_x, gm_indices, gm_y
    )
