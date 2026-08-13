add_triton_library(TensorViewToHIVM
  TensorViewToHIVM.cpp

  DEPENDS
  TensorViewToHIVMConversionPassIncGen

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
