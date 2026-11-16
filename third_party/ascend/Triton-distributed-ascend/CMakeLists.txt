# Vendored subset of Triton-distributed-ascend for the Ascend backend.
#
# Only include/, lib/ and python/src/ are kept here. The upstream top-level
# build (3rdparty/triton-ascend, the NVIDIA/AMD backends and the AOT csrc
# directory) is intentionally not vendored.
#
# This file marks the tree as initialized for the existence check in
# third_party/ascend/CMakeLists.txt. The actual targets are defined by
# include/TritonDistributed and lib/.
