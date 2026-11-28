@host
def run(self, input1_gm: Tensor, input2_gm: Tensor, output_gm: Tensor):
    kernel_instance = AddDmaKernel(self.tiling)
    kernel_instance[self.tiling.core_count](input1_gm, input2_gm, output_gm)
