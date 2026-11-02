include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/third_party/flir/include)
include_directories(${CMAKE_BINARY_DIR}/third_party/flir/include)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Triton Ascend is dependent on AscendNPU IR
# NOTE: AscendNPU-IR add_subdirectory is handled by the root CMakeLists.txt (line 344).
# Variables ASCENDNPU_IR_SRC_DIR/BINARY_DIR and include_directories are set in root CMakeLists.txt.
# Only add local includes here that the root doesn't cover.
include_directories(${ASCENDNPU_IR_SRC_DIR}/bishengir/include)
include_directories(${ASCENDNPU_IR_BINARY_DIR}/bishengir/include) # Tablegen'd files

add_subdirectory(backend/spec/lib)

add_subdirectory(${PROJECT_SOURCE_DIR}/include
                 ${PROJECT_BINARY_DIR}/include)
add_subdirectory(${PROJECT_SOURCE_DIR}/lib
                 ${PROJECT_BINARY_DIR}/lib)
add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(bin)

add_triton_library(Registrar Registrar.cc)

if(LLVM_MAJOR_VERSION_21_COMPATIBLE OR LLVM_MAJOR_VERSION_22_OR_HIGHER)
  # Only link MLIRMeshDialect if the library exists in the LLVM installation
  find_library(_MLIRMeshDialect_FOUND MLIRMeshDialect PATHS ${LLVM_LIBRARY_DIR} NO_DEFAULT_PATH)
  if(_MLIRMeshDialect_FOUND)
    set(_MLIRMeshDialect_LIB MLIRMeshDialect)
  else()
    set(_MLIRMeshDialect_LIB "")
  endif()
else()
  set(_MLIRMeshDialect_LIB "")
endif()

add_triton_plugin(TritonAscend
  ${CMAKE_CURRENT_SOURCE_DIR}/triton_ascend.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/ascend_ir.cc
  LINK_LIBS
  TritonToLinalgIncubated
  TensorViewIR
  TensorViewLowering
  BiShengIRScopeDialect
  BiShengIRHIVMDialect
  ${_MLIRMeshDialect_LIB}
)
add_dependencies(TritonAscend AscendCustomOpsBitcode)

target_include_directories(TritonAscend BEFORE PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/python/src)

option(TRITON_ENABLE_COVERAGE_LLVM_COV "Enable code llvm-cov coverage tool for Ascend plugin " OFF)
if(TRITON_ENABLE_COVERAGE_LLVM_COV)
  message(STATUS "Enabling llvm-cov coverage tool flags for TritonAscend")
  target_compile_options(TritonAscend PRIVATE
    -fprofile-arcs
    -ftest-coverage
    -O0
    -fprofile-update=atomic
    --coverage
  )
  target_link_options(TritonAscend PRIVATE
    --coverage
    -lgcov
  )
  # branch coverage
  target_compile_definitions(TritonAscend PRIVATE
    COVERAGE_ENABLED=1
  )
endif()


# To enable hitest coverage tool
if(TRITON_ENABLE_COVERAGE_HITEST)
  set_target_properties(TritonAscend PROPERTIES
    RULE_LAUNCH_COMPILE "hitestwrapper"
    RULE_LAUNCH_LINK "hitestwrapper"
  )
endif()

if(TRITON_BUILD_UT)
  add_subdirectory(unittest)
endif()
