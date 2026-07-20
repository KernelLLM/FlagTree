
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
# flagtree tle
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/dialect/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/dialect/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/utils/include)

add_subdirectory(dialect)
if(TRITON_BUILD_PYTHON_MODULE)
  set(TLE_TILEIR_LINK_LIBS "")
  if(TARGET TileIRIR)
    list(APPEND TLE_TILEIR_LINK_LIBS TileIRIR)
  endif()
  if(TARGET TileIRTransforms)
    list(APPEND TLE_TILEIR_LINK_LIBS TileIRTransforms)
  endif()

  add_triton_plugin(TritonTLE
    ${CMAKE_CURRENT_SOURCE_DIR}/triton_tle.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/triton_tle_raw.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/lib/Protocol.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/lib/AnalyzeReturnType.cpp
    LINK_LIBS
    MLIRTargetLLVMIRImport
    TritonTLETransforms
    TritonIR
    ${TLE_TILEIR_LINK_LIBS}
    )

  target_link_libraries(TritonTLE PRIVATE Python3::Module pybind11::headers MLIRTargetLLVMIRImport)
  add_dependencies(TritonTLE TleTableGen TritonTLETransformsIncGen)
endif()

add_subdirectory(test)
