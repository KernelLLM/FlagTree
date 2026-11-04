# The Dialect/ subtree held a v3.5-style spec library (FlagTree_ascend_TritonIR)
# that replaced core Triton sources. On this branch the core files carry no
# FLAGTREE_SPEC_* guards, so such a library can never suppress them; source
# specialization goes through third_party/ascend/spec_cpp instead, which
# FlagTreeBackendSpec.cmake substitutes into the owning core target.
add_subdirectory(runtime)
