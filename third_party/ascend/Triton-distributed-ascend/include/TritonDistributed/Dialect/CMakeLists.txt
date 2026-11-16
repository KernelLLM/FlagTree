add_subdirectory(Distributed)
if (NOT TRITON_USE_ASCEND)
    add_subdirectory(SIMT)
endif()
