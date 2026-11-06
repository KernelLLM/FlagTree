#ifndef TRITON_HCU_LAYOUT_TUNING_DOMAIN_H
#define TRITON_HCU_LAYOUT_TUNING_DOMAIN_H
#include <array>
#include <cstdint>
#include <string>
#include <vector>
namespace layoutdp {
struct BlockedLayoutChoice {
  std::vector<unsigned> sizePerThread, threadsPerWarp, warpsPerCTA, order;
  bool operator==(const BlockedLayoutChoice &b) const {
    return sizePerThread == b.sizePerThread && threadsPerWarp == b.threadsPerWarp &&
           warpsPerCTA == b.warpsPerCTA && order == b.order;
  }
};
struct BlockedLayoutDomain {
  uint64_t count = 0;
  bool complete = false;
  std::string reason;
  std::vector<BlockedLayoutChoice> choices;
};
// All power-of-two SPT tuples <= cap, ALL physical thread/warp factorizations
// and orders. Union the original seed. Fixed CTA partition, rank, warp count.
// No price/model dependence, no exact-cover filter, no silent truncation.
BlockedLayoutDomain enumerateBlockedDomain(const BlockedLayoutChoice &seed,
                                           unsigned sptCap, uint64_t maxChoices);
// Bounded structural proposal pool, NOT a complete Cartesian domain.
// Exact attributes stay distinct, including declared order.
std::vector<BlockedLayoutChoice> generateBlockedProposalPool(
    const BlockedLayoutChoice &current, unsigned limit = 128,
    unsigned sptProductCap = 16);
struct LayoutProposalMetrics {
  // memory segment work, memory instructions, shuffle words, LDS words,
  // block sync sites, register words. Different units are never added.
  std::array<double, 6> costs{};
  bool available = false;
};
// Index zero is the current layout. Pareto layers are priority heuristics,
// not a proof of hardware dominance; retain diversity and exploration slots.
std::vector<unsigned> selectBlockedProposals(
    const std::vector<BlockedLayoutChoice> &pool,
    const std::vector<LayoutProposalMetrics> &metrics, unsigned limit,
    bool useCosts, uint64_t rotation = 0);
} // namespace layoutdp
#endif
