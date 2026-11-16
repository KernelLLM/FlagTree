add_triton_library(TritonGPUIR
  Dialect.cpp
  LinearLayoutConversions.cpp
  Ops.cpp
  Types.cpp

  DEPENDS
  TritonGPUCTAAttrIncGen
  TritonGPUTableGen
  TritonGPUAttrDefsIncGen
  TritonGPUTypeInterfacesIncGen
  TritonGPUOpInterfacesIncGen
  # TritonGPU headers include the Nvidia GPU dialect, so make sure its
  # generated headers exist before this library is compiled (parallel builds).
  TritonNvidiaGPUTableGen
  TritonNvidiaGPUAttrDefsIncGen
  TritonNvidiaGPUOpInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRGPUDialect
  TritonIR
  TritonTools
)

add_dependencies(TritonGPUIR TritonGPUCTAAttrIncGen)
