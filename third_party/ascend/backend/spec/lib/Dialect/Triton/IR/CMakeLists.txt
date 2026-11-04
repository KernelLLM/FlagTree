# Only Traits.cpp is specialized: it relaxes the power-of-two tensor size check
# (same change triton-ascend applies to the core file). The Ops.cpp and
# TritonOps.td overrides were 3.5-era snapshots whose deltas are all superseded
# by the 3.6 core, so they were dropped and the core versions are used.
add_triton_library(FlagTree_ascend_TritonIR
  Traits.cpp

  DEPENDS
  TritonTableGen
  TritonCanonicalizeIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRArithDialect
  MLIRMathDialect
  MLIRSCFDialect
)

target_include_directories(FlagTree_ascend_TritonIR PRIVATE
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_BINARY_DIR}/lib/Dialect/Triton/IR
)
