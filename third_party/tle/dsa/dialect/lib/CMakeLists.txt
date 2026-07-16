# Copyright 2026- Xcoresigma Technology Co., Ltd

add_subdirectory(Analysis)
if(FLAGTREE_BACKEND STREQUAL "ascend")
  add_subdirectory(Conversion)
endif()
add_subdirectory(IR)
