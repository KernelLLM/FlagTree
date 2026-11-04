add_library(entryC SHARED
  libentry.cpp
)
target_compile_options(entryC PRIVATE -fexceptions -frtti)
# libentry.cpp includes "runtime/libentry/libentry.h", which lives in this
# backend's spec include tree. Scope the path to this target: the global
# BACKEND_SPEC_INCLUDE_DIR points at spec_cpp/include, which is the source
# override tree consumed by FlagTreeBackendSpec.cmake.
target_include_directories(entryC PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../include
)
