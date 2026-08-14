add_triton_library(TensorViewLowering
  TensorViewLowering.cpp

  DEPENDS
  TensorViewLoweringConversionPassIncGen

  LINK_LIBS
  TensorViewIR
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRSupport
  MLIRMemRefDialect
  MLIRBufferizationDialect
  MLIRArithDialect
  MLIRLinalgDialect
  MLIRFuncDialect
  MLIRSCFDialect
  MLIRTensorDialect
  TritonIR
)
