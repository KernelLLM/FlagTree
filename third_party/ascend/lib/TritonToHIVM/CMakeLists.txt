# TLE DSA support is optional. TritonToHIVM.cpp guards every use of it with
# #ifdef __TLE_DSA__, so only depend on the TLE targets when that subtree is
# part of the build.
set(_TritonToHIVM_TLE_DEPENDS "")
set(_TritonToHIVM_TLE_LINK_LIBS "")
if(FLAGTREE_TLE_DSA)
  set(_TritonToHIVM_TLE_DEPENDS TleDSATableGen)
  set(_TritonToHIVM_TLE_LINK_LIBS TleDSAToHIVM)
endif()

add_triton_library(TritonToHIVM
  TritonToHIVM.cpp

  DEPENDS
  TritonToHIVMConversionPassIncGen
  ${_TritonToHIVM_TLE_DEPENDS}

  LINK_LIBS
  BiShengIRHIVMDialect
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRSupport
  TritonIR
  ${_TritonToHIVM_TLE_LINK_LIBS}
)
