add_triton_library(TritonToTensorView
  TritonToTensorView.cpp

  DEPENDS
  TritonToTensorViewConversionPassIncGen

  LINK_LIBS
  TensorViewIR
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRSupport
  MLIRArithDialect
  MLIRFuncDialect
  TritonIR
)
