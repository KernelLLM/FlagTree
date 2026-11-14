#ifndef TRITON_HCU_LAYOUT_TUNING_MEMORY_H
#define TRITON_HCU_LAYOUT_TUNING_MEMORY_H
#include <cstdint>
#include <optional>
#include <vector>
namespace layoutdp {
struct MemoryPattern {
  std::vector<int64_t> shape, sizePerThread, threadsPerWarp, warpsPerCTA, order;
  std::vector<int64_t> contiguity, alignment;
  unsigned elementBytes = 2;
  // Proven runs of identical mask bits, in elements; empty means unmasked.
  std::vector<int64_t> maskAlignment;
};

struct MemoryAccessWork {
  double work = 0, instructions = 0, segmentUtilization = 0;
  unsigned vectorElements = 0, uniqueLanes = 0, segments = 0;
  uint64_t laneStrideBytes = 0;
};
std::optional<MemoryAccessWork> candidateMemoryWork(const MemoryPattern &pattern);

}
#endif
