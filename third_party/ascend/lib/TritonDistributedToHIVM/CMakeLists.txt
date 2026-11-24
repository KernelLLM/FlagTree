
add_triton_library(TritonDistributedToHIVM
    DistributedOpToHIVM.cpp
    ConvertTritonDistributedToHIVM.cpp

    DEPENDS
    TritonDistributedToHIVMConversionPassIncGen

    LINK_LIBS PUBLIC
    DistributedIR
    BiShengIRHIVMDialect
)

if(TARGET MLIRBiShengIREnumsIncGen)
  add_dependencies(TritonDistributedToHIVM
    MLIRBiShengIREnumsIncGen
    MLIRBiShengIRAttributesIncGen)
endif()
