add_subdirectory(TritonStructured)
add_subdirectory(TritonAscend)

if(TRITON_BUILD_TD)
  add_subdirectory(Distributed)
endif()
