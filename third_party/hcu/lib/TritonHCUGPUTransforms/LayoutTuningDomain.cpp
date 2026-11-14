#include "TritonHCUGPUTransforms/LayoutTuningDomain.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <functional>
#include <numeric>
namespace layoutdp {
namespace {
bool powerOfTwo(unsigned v) { return v && !(v & (v - 1)); }
std::vector<std::vector<unsigned>> factors(unsigned total, unsigned rank) {
  if (rank == 1) return {{total}};
  std::vector<std::vector<unsigned>> result;
  for (unsigned n = 1; n <= total; ++n) if (total % n == 0)
    for (auto tail : factors(total / n, rank - 1)) {
      tail.insert(tail.begin(), n); result.push_back(std::move(tail));
    }
  return result;
}
}
BlockedLayoutDomain enumerateBlockedDomain(const BlockedLayoutChoice &seed,
                                           unsigned sptCap, uint64_t maxChoices) {
  BlockedLayoutDomain result;
  unsigned rank = seed.order.size();
  if (!rank || rank > 4 || !powerOfTwo(sptCap) || sptCap > 16 ||
      seed.sizePerThread.size() != rank || seed.threadsPerWarp.size() != rank || seed.warpsPerCTA.size() != rank) {
    result.reason = "invalid bounded blocked domain parameters"; return result;
  }
  unsigned threads = 1, warps = 1;
  auto order = seed.order; std::sort(order.begin(), order.end());
  for (unsigned j = 0; j < rank; ++j) {
    if (order[j] != j || !seed.sizePerThread[j] || !powerOfTwo(seed.threadsPerWarp[j]) || !powerOfTwo(seed.warpsPerCTA[j]) ||
        seed.threadsPerWarp[j] > 128 || seed.warpsPerCTA[j] > 32) {
      result.reason = "invalid physical partition"; return result;
    }
    // Check before multiply, never overflow on an externally supplied seed.
    if (threads > 128 / seed.threadsPerWarp[j] || warps > 32 / seed.warpsPerCTA[j]) {
      result.reason = "physical partition budget"; return result;
    }
    threads *= seed.threadsPerWarp[j]; warps *= seed.warpsPerCTA[j];
  }
  auto threadChoices = factors(threads, rank), warpChoices = factors(warps, rank);
  std::vector<unsigned> widths;
  for (unsigned n = 1; n <= sptCap; n *= 2) widths.push_back(n);
  uint64_t sptCount = 1, orders = 1;
  for (unsigned j = 0; j < rank; ++j) { sptCount *= widths.size(); orders *= j + 1; }
  bool seedInBound = std::all_of(seed.sizePerThread.begin(), seed.sizePerThread.end(), [&](unsigned v) { return powerOfTwo(v) && v <= sptCap; });
  result.count = sptCount * orders * threadChoices.size() * warpChoices.size() + !seedInBound;
  if (result.count > maxChoices) { result.reason = "domain-exceeds-budget-not-truncated"; return result; }
  result.choices.push_back(seed);
  std::vector<unsigned> spt(rank, 1);
  std::function<void(unsigned)> generate = [&](unsigned dim) {
    if (dim != rank) {
      for (unsigned width : widths) { spt[dim] = width; generate(dim + 1); }
      return;
    }
    for (const auto &t : threadChoices) for (const auto &w : warpChoices) {
      std::iota(order.begin(), order.end(), 0u);
      do {
        BlockedLayoutChoice point{spt, t, w, order};
        if (!(point == seed)) result.choices.push_back(std::move(point));
      } while (std::next_permutation(order.begin(), order.end()));
    }
  };
  generate(0);
  result.complete = result.count == result.choices.size();
  result.reason = result.complete ? "complete-bounded-blocked-domain" : "internal-domain-count-error";
  return result;
}
namespace {
double layoutDistance(const BlockedLayoutChoice &a, const BlockedLayoutChoice &b) {
  double distance = 0;
  for (auto fields : {std::pair{&a.sizePerThread, &b.sizePerThread},
                      std::pair{&a.threadsPerWarp, &b.threadsPerWarp},
                      std::pair{&a.warpsPerCTA, &b.warpsPerCTA}})
    for (unsigned i = 0; i < fields.first->size(); ++i)
      distance += std::abs(std::log2(double((*fields.first)[i])) -
                           std::log2(double((*fields.second)[i])));
  for (unsigned i = 0; i < a.order.size(); ++i) distance += a.order[i] != b.order[i];
  return distance;
}
bool dominates(const LayoutProposalMetrics &a, const LayoutProposalMetrics &b) {
  if (!a.available || !b.available) return false;
  bool strict = false;
  for (unsigned k = 0; k < a.costs.size(); ++k) {
    if (!std::isfinite(a.costs[k]) || !std::isfinite(b.costs[k]) ||
        a.costs[k] > b.costs[k]) return false;
    strict |= a.costs[k] < b.costs[k];
  }
  return strict;
}
}
std::vector<BlockedLayoutChoice> generateBlockedProposalPool(
    const BlockedLayoutChoice &current, unsigned limit, unsigned sptProductCap) {
  if (!limit) return {};
  std::vector<BlockedLayoutChoice> result{current};
  unsigned rank = current.order.size(), threads = 1, warps = 1;
  if (!rank || rank > 4 || current.sizePerThread.size() != rank ||
      current.threadsPerWarp.size() != rank || current.warpsPerCTA.size() != rank)
    return result;
  for (unsigned i = 0; i < rank; ++i) {
    if (!powerOfTwo(current.sizePerThread[i]) || !powerOfTwo(current.threadsPerWarp[i]) ||
        !powerOfTwo(current.warpsPerCTA[i]) ||
        threads > 128 / current.threadsPerWarp[i] || warps > 32 / current.warpsPerCTA[i])
      return result;
    threads *= current.threadsPerWarp[i]; warps *= current.warpsPerCTA[i];
  }
  auto threadChoices = factors(threads, rank), warpChoices = factors(warps, rank);
  std::vector<std::vector<unsigned>> sptChoices, orders;
  std::vector<unsigned> spt(rank, 1), order(rank);
  std::function<void(unsigned, unsigned)> makeSpt = [&](unsigned dim, unsigned product) {
    if (dim == rank) { sptChoices.push_back(spt); return; }
    for (unsigned n : {1u, 2u, 4u, 8u}) if (product <= sptProductCap / n) {
      spt[dim] = n; makeSpt(dim + 1, product * n);
    }
  };
  makeSpt(0, 1);
  std::iota(order.begin(), order.end(), 0u);
  do { orders.push_back(order); } while (std::next_permutation(order.begin(), order.end()));
  std::vector<std::vector<BlockedLayoutChoice>> families(5);
  for (auto &v : sptChoices) { auto c = current; c.sizePerThread = v; families[0].push_back(c); }
  for (auto &v : threadChoices) { auto c = current; c.threadsPerWarp = v; families[1].push_back(c); }
  for (auto &v : warpChoices) { auto c = current; c.warpsPerCTA = v; families[2].push_back(c); }
  for (auto &v : orders) { auto c = current; c.order = v; families[3].push_back(c); }
  // Cover joint mutations without lexicographic Cartesian-prefix bias.
  for (unsigned i = 0; i < limit && !sptChoices.empty(); ++i)
    families[4].push_back({sptChoices[i % sptChoices.size()],
        threadChoices[(i * 5 + i / threadChoices.size()) % threadChoices.size()],
        warpChoices[(i * 3 + i / warpChoices.size()) % warpChoices.size()],
        orders[i % orders.size()]});
  for (unsigned round = 0; result.size() < limit; ++round) {
    bool available = false;
    for (const auto &family : families) if (round < family.size()) {
      available = true;
      if (std::find(result.begin(), result.end(), family[round]) == result.end())
        result.push_back(family[round]);
      if (result.size() == limit) break;
    }
    if (!available) break;
  }
  return result;
}
std::vector<unsigned> selectBlockedProposals(
    const std::vector<BlockedLayoutChoice> &pool,
    const std::vector<LayoutProposalMetrics> &metrics, unsigned limit,
    bool useCosts, uint64_t rotation) {
  if (pool.empty() || !limit) return {};
  limit = std::min<unsigned>(limit, pool.size());
  std::vector<unsigned> selected{0}; std::vector<bool> used(pool.size(), false); used[0] = true;
  auto add = [&](unsigned i) { if (!used[i]) { used[i] = true; selected.push_back(i); } };
  unsigned costSlots = useCosts ? (limit - 1) / 2 : 0;
  std::vector<unsigned> layer(pool.size(), 0);
  if (useCosts && metrics.size() == pool.size()) {
    std::vector<bool> assigned(pool.size(), false);
    for (unsigned level = 0, left = pool.size(); left; ++level) {
      std::vector<unsigned> front;
      for (unsigned i = 0; i < pool.size(); ++i) if (!assigned[i]) {
        bool worse = false;
        for (unsigned j = 0; j < pool.size() && !worse; ++j)
          if (!assigned[j] && j != i && dominates(metrics[j], metrics[i])) worse = true;
        if (!worse) front.push_back(i);
      }
      if (front.empty()) break;
      for (unsigned i : front) { layer[i] = level; assigned[i] = true; --left; }
    }
    for (unsigned slot = 0; slot < costSlots; ++slot) {
      unsigned best = pool.size(), objective = (slot + rotation) % 6;
      for (unsigned i = 1; i < pool.size(); ++i) if (!used[i] && metrics[i].available) {
        if (best == pool.size() || layer[i] < layer[best] ||
            (layer[i] == layer[best] && metrics[i].costs[objective] < metrics[best].costs[objective])) best = i;
      }
      if (best != pool.size()) add(best);
    }
  }
  // Reserve two slots for deterministic rotating exploration in cost mode.
  unsigned diversityEnd = useCosts && limit > 3 ? limit - 2 : limit;
  while (selected.size() < diversityEnd) {
    double bestDistance = -1; unsigned best = pool.size();
    for (unsigned i = 1; i < pool.size(); ++i) if (!used[i]) {
      double distance = std::numeric_limits<double>::infinity();
      for (unsigned j : selected) distance = std::min(distance, layoutDistance(pool[i], pool[j]));
      if (distance > bestDistance) { bestDistance = distance; best = i; }
    }
    if (best == pool.size()) break;
    add(best);
  }
  for (unsigned k = 0; selected.size() < limit && k < pool.size(); ++k)
    add((rotation + k * 37) % pool.size());
  for (unsigned i = 0; selected.size() < limit && i < pool.size(); ++i) add(i);
  return selected;
}
} // namespace layoutdp
