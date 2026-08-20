#ifndef TRITON_NVIDIA_COMMON_IR_TO_TTGIR_PASSES
#define TRITON_NVIDIA_COMMON_IR_TO_TTGIR_PASSES

include "mlir/Pass/PassBase.td"

def CommonIRToTTGIR
    : Pass<"convert-common-ir-to-ttgir", "mlir::ModuleOp"> {
  let summary = "convert GPU CommonIR buffers to TritonGPU descriptors";
  let description = [{
    Converts the target-independent TileIR produced by tle.gpu primitives into
    the NVIDIA TritonGPU descriptor form. In particular, tile.alloc becomes
    ttg.local_alloc with a ttg.memdesc result and tile.local_ptr becomes
    tle.local_pointers.
  }];
  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::triton::TritonDialect",
                           "mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::tle::TleDialect",
                           "mlir::triton::tile::TileIRDialect"];
}

#endif
