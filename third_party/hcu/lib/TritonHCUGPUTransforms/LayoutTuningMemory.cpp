#include "TritonHCUGPUTransforms/LayoutTuningMemory.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
namespace layoutdp {
std::optional<MemoryAccessWork> candidateMemoryWork(const MemoryPattern &p) {
  // Validate geometry directly. The learned-facts path must not evaluate the
  // historical memory score even to decide whether an access is supported.
  size_t rank = p.shape.size();
  if (!rank || !p.elementBytes || p.sizePerThread.size() != rank ||
      p.threadsPerWarp.size() != rank || p.warpsPerCTA.size() != rank ||
      p.order.size() != rank || p.elementBytes > 16 ||
      (p.elementBytes & (p.elementBytes - 1))) return std::nullopt;
  std::set<int64_t> dimensions;
  double extentProduct = 1;
  for (size_t d = 0; d < rank; ++d) {
    if (p.shape[d] <= 0 || p.sizePerThread[d] <= 0 ||
        p.threadsPerWarp[d] <= 0 || p.warpsPerCTA[d] <= 0 ||
        p.order[d] < 0 || size_t(p.order[d]) >= rank) return std::nullopt;
    dimensions.insert(p.order[d]);
    extentProduct *= double(p.shape[d]) * p.sizePerThread[d] *
                     p.threadsPerWarp[d] * p.warpsPerCTA[d];
  }
  if (dimensions.size() != rank || !std::isfinite(extentProduct)) return std::nullopt;
  // Singleton/redundant dimensions have no lane/register basis. Moving them
  // in the textual order must not change the score of the same mapping.
  auto registerDim = p.shape.size(), laneDim = p.shape.size();
  for (auto i : p.order) {
    if (registerDim == p.shape.size() && std::min(p.sizePerThread[i], p.shape[i]) > 1)
      registerDim = i;
    if (laneDim == p.shape.size() &&
        std::min(p.threadsPerWarp[i], p.shape[i] / p.sizePerThread[i]) > 1)
      laneDim = i;
  }
  const size_t d = laneDim < p.shape.size() ? laneDim :
      (registerDim < p.shape.size() ? registerDim : 0);
  auto axis = [&](const std::vector<int64_t> &v) { return d < v.size() ? v[d] : int64_t(1); };
  const int64_t contig = axis(p.contiguity), align = axis(p.alignment);
  const int64_t mask = p.maskAlignment.empty() ? contig : axis(p.maskAlignment);
  if (contig < 1 || align < 1 || mask < 1 || p.threadsPerWarp[d] > 64)
    return std::nullopt;
  // Only power-of-two blocked dimensions have the modulo/folding semantics
  // used here. Unknown geometry is explicitly unavailable, never zero work.
  auto power2 = [](int64_t x) { return x > 0 && (x & (x - 1)) == 0; };
  double elements = 1;
  for (size_t i = 0; i < p.shape.size(); ++i) {
    if (!power2(p.shape[i]) || !power2(p.sizePerThread[i]) ||
        !power2(p.threadsPerWarp[i]) || !power2(p.warpsPerCTA[i])) return std::nullopt;
    const double tile = double(p.sizePerThread[i]) * p.threadsPerWarp[i] * p.warpsPerCTA[i];
    // A register basis beyond the tensor extent repeats an existing element.
    elements *= std::ceil(p.shape[i] / tile) * std::min(p.sizePerThread[i], p.shape[i]);
  }
  // When registers and lanes advance on different axes, do not assume their
  // byte runs are adjacent. This restricted proxy conservatively scalarizes.
  const int64_t cap = std::min({registerDim == d ? p.sizePerThread[d] : int64_t(1), p.shape[d], contig, align, mask,
                                int64_t(16 / p.elementBytes)});
  unsigned vector = 1;
  while (2 * vector <= cap) vector *= 2;
  const int64_t lanes = std::min({p.threadsPerWarp[d],
      std::max<int64_t>(1, p.shape[d] / p.sizePerThread[d]),
      std::max<int64_t>(1, contig / p.sizePerThread[d])});
  // Model alignment uncertainty conservatively. The base may have any offset
  // consistent with its proven power-of-two alignment. Contiguity says nothing
  // about the distance to the next run; those runs must not be coalesced here.
  unsigned alignmentBytes = p.elementBytes;
  while (alignmentBytes < 32 && align % (2 * alignmentBytes / p.elementBytes) == 0)
    alignmentBytes *= 2;
  unsigned worst = 0;
  for (unsigned offset = 0; offset < 32; offset += alignmentBytes) {
    std::set<uint64_t> segments;
    for (int64_t lane = 0; lane < lanes; ++lane)
      for (unsigned byte = 0; byte < vector * p.elementBytes; ++byte)
        segments.insert((offset + uint64_t(lane) * p.sizePerThread[d] * p.elementBytes + byte) / 32);
    worst = std::max(worst, unsigned(segments.size()));
  }
  const double utilization = double(lanes * vector * p.elementBytes) / (32 * worst);
  const double instructions = std::ceil(elements / vector);
  const double work = instructions / utilization;
  if (!std::isfinite(work) || work <= 0) return std::nullopt;
  return MemoryAccessWork{work, instructions, utilization, vector, unsigned(lanes), worst,
                          uint64_t(p.sizePerThread[d]) * p.elementBytes};
}
}
