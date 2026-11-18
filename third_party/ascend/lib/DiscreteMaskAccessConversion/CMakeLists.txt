add_triton_library(DiscreteMaskAccessConversion
  DiscreteMaskAccessConversionPass.cpp

  DEPENDS
  DiscreteMaskAccessConversionPassIncGen

  LINK_LIBS
  BiShengIRHIVMDialect
  MLIRIR
  MLIRPass
  MLIRSCFDialect
  MLIRTransforms
  MLIRSupport
  TritonIR
  TritonToLinalgIncubated
  FlirTritonMemoryAccessAnalysis
)
