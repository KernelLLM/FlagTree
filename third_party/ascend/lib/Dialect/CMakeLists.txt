add_subdirectory(TritonAscend)
add_subdirectory(TritonStructured)

if(TRITON_BUILD_TD)
  add_subdirectory(Distributed)
endif()
