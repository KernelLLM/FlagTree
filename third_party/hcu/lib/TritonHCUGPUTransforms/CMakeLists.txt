add_triton_library(TritonHCUGPUTransforms
  # AccelerateHCUMatmul.cpp
  BlockPingpong.cpp
  CanonicalizePointers.cpp
  CoalesceAsyncCopy.cpp
  ConvertToBufferOps.cpp
  OptimizeEpilogue.cpp
  OptimizeDotOperands.cpp
  HoistLayoutConversions.cpp
  ReorderInstructions.cpp
  Pipeline.cpp
  ScheduleLoops.cpp
  LowerLoops.cpp
  # MfmaGroup.cpp
  WmmaGroup.cpp
  InThreadTranspose.cpp
  FoldTrueCmpIOp.cpp
  UpdateAsyncWaitCount.cpp
  Utility.cpp
  # ---- HCU code ---- #
  AccelerateHCUMatmul.cpp
  MmacGroup.cpp
  MlsStreamPipeline.cpp
  MlsGroup.cpp
  MlsEncodingInsertion.cpp
  MlsLowering.cpp
  LayoutTuningDomain.cpp
  LayoutTuningFacts.cpp
  LayoutTuningProjection.cpp
  LayoutTuningTransport.cpp
  LayoutTuningCollective.cpp
  LayoutExecutionContext.cpp
  LayoutTuningLearned.cpp
  LayoutTuningUnified.cpp
  LayoutTuning.cpp
  LayoutTuningResources.cpp
  LayoutTuningMemory.cpp
  LayoutTuningIndex.cpp
  LayoutTuningPass.cpp

  DEPENDS
  TritonHCUGPUIR
  TritonHCUGPUTransformsIncGen
  TritonGPUIR
  TritonHCUUtils
  TritonHCUAnalysis
  DistributedIR
  TritonTLETransformsIncGen

  LINK_LIBS PUBLIC
  TritonTLETransforms
)

target_include_directories(TritonHCUGPUTransforms PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
target_include_directories(TritonHCUGPUTransforms PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/../../include)
