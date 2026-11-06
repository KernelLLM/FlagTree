// Physical/structural analysis shared by the two independent scoring passes.
// No price table, fixed latency constants, seed, or learned prediction here.
#include "TritonHCUGPUTransforms/LayoutTuningFacts.h"
#include "TritonHCUGPUTransforms/LayoutTuningResources.h"
#include "TritonHCUGPUTransforms/LayoutTuningIndex.h"
#include "TritonHCUGPUTransforms/LayoutExecutionContext.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "Dialect/TritonHCUGPU/IR/Dialect.h"
#include "Analysis/HCUGPUAllocation.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Dominance.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Tools/GenericSwizzling.h"
#include "triton/Tools/LayoutUtils.h"
#include "llvm/Support/MathExtras.h"
#include <cmath>
#include <map>
#include <set>
using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace hcu = mlir::triton::hcugpu;
namespace layoutdp {
namespace {
std::string printed(Type type) {
  std::string text; llvm::raw_string_ostream out(text); type.print(out); return text;
}
llvm::json::Value number(double x) {
  return std::isfinite(x) ? llvm::json::Value(x) : llvm::json::Value(nullptr);
}
double executionWeight(Operation *op) {
  double weight = 1;
  auto integer = [](Value value) -> std::optional<int64_t> {
    if (auto c = value.getDefiningOp<arith::ConstantOp>())
      if (auto i = dyn_cast<IntegerAttr>(c.getValue())) return i.getInt();
    return {};
  };
  for (Operation *parent = op->getParentOp(); parent; parent = parent->getParentOp())
    if (auto loop = dyn_cast<scf::ForOp>(parent)) {
      auto a=integer(loop.getLowerBound()), b=integer(loop.getUpperBound()), step=integer(loop.getStep());
      if (a && b && step && *step > 0)
        weight *= std::max(0.0, std::ceil((double(*b)-double(*a))/double(*step)));
    }
  return weight;
}
Attribute rootEncoding(Attribute value) {
  while (auto slice = dyn_cast_or_null<ttg::SliceEncodingAttr>(value)) value=slice.getParent();
  return value;
}

SmallVector<unsigned> declaredBlockedOrder(Attribute encoding) {
  if (auto blocked = dyn_cast_or_null<ttg::BlockedEncodingAttr>(encoding))
    return SmallVector<unsigned>(blocked.getOrder());
  if (auto slice = dyn_cast_or_null<ttg::SliceEncodingAttr>(encoding)) {
    auto parent = declaredBlockedOrder(slice.getParent());
    SmallVector<unsigned> order;
    for (unsigned dim : parent)
      if (dim != slice.getDim()) order.push_back(dim > slice.getDim() ? dim - 1 : dim);
    return order;
  }
  return {};
}

// The same geometry routine handles every distributed encoding. Encoding
// spelling, operator family and the compiler's initial layout are not inputs.
// Unknown outer strides stay separate; 32B is a proxy granularity, not a
// hardware transaction claim. Scalar accesses have separate logical units.
llvm::json::Object accessGeometry(Value value, Value address, Value mask,
                                  Value base, tt::ModuleAxisInfoAnalysis &axes,
                                  unsigned vectorLimit = 0, RankedTensorType typeOverride = {}) {
  Type valueType = typeOverride ? Type(typeOverride) : value.getType();
  llvm::json::Object row{{"available", false}, {"data_type", printed(valueType)},
      {"address_type", printed(address.getType())}, {"masked", bool(mask)},
      {"mask_contents_known", false}, {"hardware_transactions_known", false}};
  auto fail = [&](const char *reason) {
    row["reason"] = reason;
    return std::move(row);
  };
  Type element = getElementTypeOrSelf(valueType);
  if (!element.isIntOrFloat()) return fail("non-numeric memory payload");
  unsigned bits = element.getIntOrFloatBitWidth();
  unsigned bytes = std::max(1u, (bits + 7) / 8);
  row["element_bits"] = int64_t(bits); row["element_bytes"] = int64_t(bytes);
  if (bytes > 16 || (bytes & (bytes - 1))) return fail("unsupported memory element width");
  auto type = dyn_cast<RankedTensorType>(valueType);
  if (!type) {
    if (isa<RankedTensorType>(address.getType())) return fail("scalar payload with tensor address");
    row["available"] = true; row["kind"] = "scalar";
    row["units"] = "logical-scalar-access-sites-and-bytes-not-tensor-segment-work";
    row["logical_sites"] = 1.0; row["logical_bytes"] = double(bytes);
    row["active_lanes_known"] = false;
    return row;
  }
  row["kind"] = "tensor";
  auto addressType = dyn_cast<RankedTensorType>(address.getType());
  if (!addressType || addressType.getShape() != type.getShape())
    return fail("tensor payload/address shape mismatch");
  if (!type.getEncoding() || !isa<ttg::DistributedEncodingTrait>(type.getEncoding()))
    return fail("non-distributed memory layout");
  auto layout = ttg::toLinearLayout(type);
  auto *ctx = value.getContext();
  auto reg = StringAttr::get(ctx, "register"), lane = StringAttr::get(ctx, "lane"),
       warp = StringAttr::get(ctx, "warp"), block = StringAttr::get(ctx, "block");
  if (layout.getInDimSize(block) != 1) return fail("multi-CTA memory geometry");
  layout = tt::actionRemoveBroadcastedRegs(layout).apply(layout);
  unsigned registers = layout.getInDimSize(reg), lanes = layout.getInDimSize(lane),
           warps = layout.getInDimSize(warp);
  // A structural budget, independent of layout labels and cost predictions.
  constexpr uint64_t enumerationBudget = 1048576;
  uint64_t enumerationPoints = uint64_t(registers) * lanes * warps;
  row["enumeration_points"] = int64_t(enumerationPoints);
  row["enumeration_budget"] = int64_t(enumerationBudget);
  if (enumerationPoints > enumerationBudget || lanes > 64 || warps > 32)
    return fail("LinearLayout memory enumeration budget");
  auto *info = axes.getAxisInfo(address);
  if (!info || info->getContiguity().size() != unsigned(type.getRank()) ||
      info->getDivisibility().size() != unsigned(type.getRank()))
    return fail("missing address AxisInfo");
  auto linear = ttg::LinearEncodingAttr::get(ctx, layout);
  auto order = linear.getOrder();
  if (order.empty() || order[0] >= unsigned(type.getRank())) return fail("invalid memory dimension order");
  unsigned dim = order[0];
  int64_t contig = info->getContiguity()[dim];
  int64_t alignment = info->getDivisibility()[dim];
  if (isa<tt::PointerType>(addressType.getElementType())) alignment = std::max<int64_t>(1, alignment / bytes);
  if (base) {
    auto *baseInfo = axes.getAxisInfo(base);
    alignment = std::min<int64_t>(alignment, baseInfo ? std::max<int64_t>(1, baseInfo->getDivisibility(0) / bytes) : 1);
  }
  if (contig < 1 || alignment < 1) return fail("non-positive memory contiguity/alignment");
  unsigned cap = std::min({registers, linear.getContigPerThread()[dim],
      unsigned(std::min<int64_t>(16 / bytes, std::min(contig, alignment)))});
  if (vectorLimit) cap = std::min(cap, vectorLimit);
  if (mask) cap = std::min(cap, axes.getMaskAlignment(mask));
  unsigned vector = 1;
  while (2 * vector <= cap && registers % (2 * vector) == 0) vector *= 2;
  unsigned alignBytes = bytes;
  while (alignBytes < 32 && std::min(contig, alignment) % (2 * alignBytes / bytes) == 0) alignBytes *= 2;
  double work = 0; unsigned maxSegments = 0, maxUnique = 0;
  for (unsigned r = 0; r < registers; r += vector) {
    double worstWarp = 0;
    for (unsigned w = 0; w < warps; ++w) {
      std::map<std::vector<int64_t>, std::set<int64_t>> groups;
      std::set<std::vector<int64_t>> coordinates;
      for (unsigned l = 0; l < lanes; ++l) for (unsigned v = 0; v < vector; ++v) {
        auto mapped = layout.apply({{reg, int32_t(r + v)}, {lane, int32_t(l)},
                                    {warp, int32_t(w)}, {block, 0}});
        std::vector<int64_t> coordinate;
        for (auto item : mapped) coordinate.push_back(item.second);
        coordinates.insert(coordinate);
        int64_t position = coordinate[dim] % contig;
        coordinate[dim] /= contig;
        groups[coordinate].insert(position);
      }
      unsigned segments = 0;
      for (const auto &group : groups) {
        unsigned worst = 0;
        for (unsigned offset = 0; offset < 32; offset += alignBytes) {
          std::set<uint64_t> touched;
          for (int64_t coordinate : group.second) for (unsigned byte = 0; byte < bytes; ++byte)
            touched.insert((offset + uint64_t(coordinate) * bytes + byte) / 32);
          worst = std::max(worst, unsigned(touched.size()));
        }
        segments += worst;
      }
      worstWarp = std::max(worstWarp, 32.0 * segments / (coordinates.size() * bytes));
      maxSegments = std::max(maxSegments, segments);
      maxUnique = std::max(maxUnique, unsigned(coordinates.size()));
    }
    work += worstWarp;
  }
  if (!std::isfinite(work) || work <= 0) return fail("non-finite memory geometry");
  row["available"] = true; row["reason"] = "";
  row["method"] = "uniform-linearlayout-segments-v114";
  row["work"] = work; row["instructions"] = double(registers / vector);
  row["vector_elements"] = int64_t(vector); row["unique_registers"] = int64_t(registers);
  row["max_unique_elements_per_warp_vector"] = int64_t(maxUnique);
  row["max_segments_per_warp_vector"] = int64_t(maxSegments);
  row["contiguity_elements"] = contig; row["alignment_elements"] = alignment;
  row["segment_granularity_bytes"] = 32; row["unknown_stride_runs_coalesced"] = false;
  row["instruction_count_is_lowering_proxy"] = true;
  return row;
}

// These are static operation counts of the generic native scan lowering,
// before predicate simplification, CSE or target specialization. Scratch
// allocation is already included in resources and must not be added again.
llvm::json::Object scanGeometry(tt::ScanOp op) {
  llvm::json::Object row{{"available", false}, {"axis", int64_t(op.getAxis())},
      {"reverse", op.getReverse()}, {"emitted_instruction_count_known", false},
      {"units", "per-thread-static-generic-lowering-work-before-predicate-CSE"}};
  auto fail = [&](const char *reason) { row["reason"] = reason; return std::move(row); };
  if (!op->getNumOperands()) return fail("empty scan");
  auto type = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  if (!type || !isa<ttg::BlockedEncodingAttr>(type.getEncoding()))
    return fail("scan layout unsupported by native ScanLoweringHelper");
  auto layout = ttg::toLinearLayout(type);
  auto reg = StringAttr::get(op.getContext(), "register");
  if (layout.getInDimSize(StringAttr::get(op.getContext(), "block")) != 1)
    return fail("multi-CTA scan");
  if (layout.getInDimSizeLog2(reg) > 12) return fail("scan register analysis budget");
  ScanLoweringHelper helper(op);
  if (!helper.isSupported()) return fail("unsupported native scan lowering");
  unsigned tupleWords = 0, combineWordOps = 0;
  bool supported = true;
  llvm::json::Array types, combiner;
  for (Type t : op.getElementTypes()) {
    if (!t.isIntOrFloat() || t.getIntOrFloatBitWidth() > 64) supported = false;
    else tupleWords += (t.getIntOrFloatBitWidth() + 31) / 32;
    types.push_back(printed(t));
  }
  op.getCombineOp().walk([&](Operation *node) {
    if (isa<tt::ScanReturnOp, arith::ConstantOp>(node)) return;
    auto name = node->getName().getStringRef();
    if (node->getNumRegions() || (!name.starts_with("arith.") && !name.starts_with("math.")))
      supported = false;
    for (Type t : node->getResultTypes()) {
      if (!t.isIntOrFloat() || t.getIntOrFloatBitWidth() > 64) supported = false;
      else combineWordOps += (t.getIntOrFloatBitWidth() + 31) / 32;
    }
    combiner.push_back(name.str());
  });
  if (!supported || !tupleWords || !combineWordOps) return fail("unknown scan combiner semantics");
  unsigned registers = ttg::getTotalElemsPerThread(type);
  unsigned contiguous = helper.getAxisNumElementsPerThread();
  unsigned lanes = helper.getAxisNumThreadsPerWarpWithUniqueData();
  unsigned warps = helper.getAxisNumWarpsWithUniqueData();
  unsigned axisBlocks = helper.getAxisNumBlocks();
  unsigned parallel = helper.getNonAxisNumElementsPerThread() * helper.getNonAxisNumBlocks();
  if (!contiguous || !lanes || !warps || !parallel || !axisBlocks ||
      uint64_t(parallel) * axisBlocks * contiguous != registers)
    return fail("inconsistent native scan geometry");
  double chunks = double(registers) / contiguous;
  double warpCalls = chunks * std::log2(lanes);
  double carryCalls = 0, carryShuffles = 0;
  if (warps > 1) {
    carryCalls = chunks * warps - parallel + registers;
    carryShuffles = chunks;
  } else if (registers > 1) {
    carryCalls = registers - parallel;
    if (lanes > 1) carryShuffles = chunks * (axisBlocks > 1 ? 2 : 1);
  }
  auto module = op->getParentOfType<ModuleOp>();
  auto warpWidth = module->getAttrOfType<IntegerAttr>("ttg.threads-per-warp");
  if (!warpWidth || warpWidth.getInt() < 1) return fail("missing scan hardware warp width");
  llvm::json::Object work{
      {"scan_local_combine_work", (registers - chunks) * combineWordOps},
      {"scan_lane_combine_work", warpCalls * combineWordOps},
      {"scan_carry_combine_work", carryCalls * combineWordOps},
      {"scan_shuffle_word_work", (warpCalls + carryShuffles) * tupleWords},
      {"scan_reverse_shuffle_word_work", op.getReverse() ? 2.0 * registers * std::log2(warpWidth.getInt()) * tupleWords : 0.0},
      {"scan_lds_store_words", warps > 1 ? chunks * tupleWords : 0.0},
      {"scan_lds_load_words", warps > 1 ? chunks * warps * tupleWords : 0.0},
      {"scan_block_sync_sites", warps > 1 ? 1.0 : 0.0}};
  row["available"] = true; row["element_types"] = std::move(types);
  row["combiner_operations"] = std::move(combiner);
  row["combiner_word_operations"] = int64_t(combineWordOps);
  row["tuple_words"] = int64_t(tupleWords); row["lowering_registers"] = int64_t(registers);
  row["unique_registers"] = int64_t(tt::actionRemoveBroadcastedRegs(layout).apply(layout).getInDimSize(reg));
  row["contiguous_axis_elements"] = int64_t(contiguous);
  row["axis_lanes"] = int64_t(lanes); row["axis_warps"] = int64_t(warps);
  row["axis_blocks"] = int64_t(axisBlocks); row["parallel_groups"] = int64_t(parallel);
  row["scratch_bytes"] = int64_t(helper.getScratchSizeInBytes());
  row["scratch_already_owned_by_resources"] = true;
  row["work"] = std::move(work);
  return row;
}

// A layout conversion preserves every logical element. Inserting one on an
// extern operand must not turn a compile-time constant into a dynamic value.
// Follow only verified value-preserving layout/shape views of a uniform
// integer; do not interpret arbitrary arithmetic, loaded contents or calls.
std::optional<int64_t> uniformIntegerConstant(Value value) {
  for (unsigned depth = 0; depth < 64; ++depth) {
    auto *op = value.getDefiningOp();
    if (!op) return {};
    if (auto c = dyn_cast<arith::ConstantOp>(op)) {
      if (auto i = dyn_cast<IntegerAttr>(c.getValue())) return i.getInt();
      if (auto dense = dyn_cast<DenseIntElementsAttr>(c.getValue()); dense && dense.isSplat())
        return dense.getSplatValue<APInt>().getSExtValue();
      return {};
    }
    if (isa<ttg::ConvertLayoutOp, tt::SplatOp, tt::BroadcastOp,
            tt::ExpandDimsOp, tt::TransOp, tt::ReshapeOp, tt::SplitOp>(op) &&
        op->getNumOperands() == 1) {
      value = op->getOperand(0);
      continue;
    }
    return {};
  }
  return {};
}

// Only the builtin OCML functions whose numeric signatures are declared in
// hcu/language/hip/libdevice.py are interpreted. Caller-supplied libraries,
// impure calls and unknown symbols remain explicit gaps. Symbol strings are
// diagnostic metadata, not model features or an operator-family dispatch.
llvm::json::Object externGeometry(tt::ExternElementwiseOp op) {
  llvm::json::Object row{{"available", false}, {"symbol", op.getSymbol().str()},
      {"pure", op.getPure()}, {"emitted_instruction_count_known", false}};
  auto fail = [&](const char *reason) { row["reason"] = reason; return std::move(row); };
  if (!op.getPure() || !op.getLibname().empty() || !op.getLibpath().empty())
    return fail("extern requires pure builtin OCML linkage");
  StringRef symbol = op.getSymbol();
  if (!symbol.consume_front("__ocml_")) return fail("unknown extern semantics");
  unsigned bits = 0;
  if (symbol.consume_back("_f32")) bits = 32;
  else if (symbol.consume_back("_f64")) bits = 64;
  else return fail("unsupported OCML numeric signature");
  bool power = symbol == "pown";
  bool rounding = symbol == "nearbyint";
  if (symbol != "tanh" && symbol != "erf" && !power && !rounding) return fail("unknown OCML semantics");
  if (op->getNumOperands() != (power ? 2u : 1u)) return fail("OCML arity mismatch");
  Type element = getElementTypeOrSelf(op.getType());
  if (!isa<FloatType>(element) || element.getIntOrFloatBitWidth() != bits ||
      getElementTypeOrSelf(op->getOperand(0).getType()) != element ||
      (power && !getElementTypeOrSelf(op->getOperand(1).getType()).isInteger(32)))
    return fail("OCML dtype mismatch");
  double elements = 1;
  if (auto tensor = dyn_cast<RankedTensorType>(op.getType())) {
    if (!tensor.getEncoding() || !isa<ttg::DistributedEncodingTrait>(tensor.getEncoding()))
      return fail("non-distributed extern layout");
    auto layout = ttg::toLinearLayout(tensor);
    if (layout.getInDimSize(StringAttr::get(op.getContext(), "block")) != 1)
      return fail("multi-CTA extern");
    elements = tt::actionRemoveBroadcastedRegs(layout).apply(layout).getInDimSize(StringAttr::get(op.getContext(), "register"));
  }
  std::optional<int64_t> exponent;
  if (rounding) row["semantic_family"] = "round-to-integral; declared pure unary OCML, not zero cost or an emitted ISA count";
  if (power) exponent = uniformIntegerConstant(op->getOperand(1));
  llvm::json::Object work{
      {"extern_math_element_work", power ? 0.0 : elements},
      {"extern_power_element_work", power ? elements : 0.0},
      {"extern_input_word_work", elements * (bits / 32.0 + (power ? 1 : 0))},
      {"extern_output_word_work", elements * bits / 32.0},
      {"extern_constant_exponent_work", exponent ? elements : 0.0},
      {"extern_dynamic_exponent_work", power && !exponent ? elements : 0.0},
      {"extern_exponent_magnitude_work", exponent ? elements * std::abs(double(*exponent)) : 0.0}};
  row["available"] = true; row["semantic_kind"] = power ? "integer-power" : "transcendental";
  row["element_bits"] = int64_t(bits); row["elements_per_thread"] = elements;
  row["integer_exponent_known"] = bool(exponent);
  row["unknown_exponent_policy"] = "not-proven-constant; no claim of runtime variability";
  if (exponent) row["integer_exponent"] = *exponent;
  row["units"] = "per-thread-logical-element-word-work-not-OCML-ISA-instructions";
  row["work"] = std::move(work);
  return row;
}

llvm::json::Object sharedWordGeometry(Value value) {
  llvm::json::Object row{{"available", false}, {"address_space", 3},
      {"bank_conflicts_known", false}, {"mask_contents_known", false},
      {"units", "static-per-thread-payload-word-sites-before-predicate-CSE"}};
  auto element = getElementTypeOrSelf(value.getType());
  if (!element.isIntOrFloat() || element.getIntOrFloatBitWidth() > 64) {
    row["reason"] = "unsupported LDS payload type"; return row;
  }
  double elements = 1;
  if (auto type = dyn_cast<RankedTensorType>(value.getType())) {
    if (!type.getEncoding() || !isa<ttg::DistributedEncodingTrait>(type.getEncoding())) {
      row["reason"] = "unsupported LDS distributed layout"; return row;
    }
    auto layout = ttg::toLinearLayout(type);
    if (layout.getInDimSize(StringAttr::get(value.getContext(), "block")) != 1) {
      row["reason"] = "cross-CTA LDS access"; return row;
    }
    elements = tt::actionRemoveBroadcastedRegs(layout).apply(layout)
        .getInDimSize(StringAttr::get(value.getContext(), "register"));
  }
  row["available"] = true; row["reason"] = "";
  row["elements"] = elements;
  row["word_sites"] = elements * ((element.getIntOrFloatBitWidth() + 31) / 32);
  return row;
}

llvm::json::Object collectiveGeometry(Operation *op) {
  llvm::json::Object row{{"available", false}, {"emitted_instruction_count_known", false},
      {"units", "static-native-lowering-sites-with-predicated-serial-work-separated"},
      {"scratch_ownership", "already included once by HCUAllocationAnalysis"}};
  auto fail = [&](const char *reason) { row["reason"] = reason; return std::move(row); };
  llvm::json::Object work;
  if (auto cumsum = dyn_cast<tt::tle::ExclusiveCumsumOp>(op)) {
    auto type = dyn_cast<RankedTensorType>(cumsum.getSrc().getType());
    if (!type || type.getRank() != 1 || !isa<ttg::DistributedEncodingTrait>(type.getEncoding()) ||
        !type.getElementType().isIntOrFloat() || type.getElementTypeBitWidth() > 64)
      return fail("unsupported native exclusive-cumsum geometry");
    auto layout = ttg::toLinearLayout(type);
    if (layout.getInDimSize(StringAttr::get(op->getContext(), "block")) != 1)
      return fail("multi-CTA exclusive-cumsum");
    double extent = type.getShape()[0], regs = ttg::getTotalElemsPerThread(type);
    unsigned lanes = ttg::TritonGPUDialect::getThreadsPerWarp(op->getParentOfType<ModuleOp>()),
             warps = ttg::lookupNumWarps(op);
    double words = (type.getElementTypeBitWidth() + 31) / 32;
    bool parallel = regs == 1 && extent <= lanes * warps && lanes % 32 == 0 && warps > 0 && warps <= 32;
    bool ordered = parallel && !cumsum.getReverse() && extent == lanes * warps;
    row["route"] = ordered ? "ordered-CTA-warp-prefix" : parallel ? "staged-warp-prefix" : "serial-thread0-fallback";
    row["register_values"] = regs; row["extent"] = extent;
    if (extent < 1 || (lanes != 32 && lanes != 64)) return fail("invalid cumsum extent/warp width");
    if (parallel) {
      work["scan_shuffle_word_work"] = (std::log2(lanes) + 1) * words;
      work["collective_parallel_word_work"] = (std::log2(lanes) + 2) * words;
      work["collective_serial_word_work"] = warps * words;
      work["lds_load_word_sites"] = (warps + (ordered ? 2 : 4)) * words;
      work["lds_store_word_sites"] = (warps + (ordered ? 2 : 4)) * words;
      work["scan_block_sync_sites"] = ordered ? 2. : 4.;
    } else {
      work["collective_serial_word_work"] = extent * words;
      work["lds_load_word_sites"] = (extent + regs + 1) * words;
      work["lds_store_word_sites"] = (extent + regs + 1) * words;
      work["scan_block_sync_sites"] = 2.;
    }
  } else if (auto histogram = dyn_cast<tt::HistogramOp>(op)) {
    auto src = histogram.getSrc().getType(), dst = histogram.getType();
    if (!isa<ttg::DistributedEncodingTrait>(src.getEncoding()) ||
        !isa<ttg::DistributedEncodingTrait>(dst.getEncoding()) || src.getElementTypeBitWidth() != 32 ||
        dst.getElementTypeBitWidth() != 32 || dst.getRank() != 1)
      return fail("unsupported native histogram geometry");
    auto layout = ttg::toLinearLayout(src);
    if (layout.getInDimSize(StringAttr::get(op->getContext(), "block")) != 1)
      return fail("multi-CTA histogram");
    unsigned lanes = ttg::TritonGPUDialect::getThreadsPerWarp(op->getParentOfType<ModuleOp>()),
             warps = ttg::lookupNumWarps(op);
    unsigned bins = std::max<unsigned>(dst.getNumElements(), lanes);
    if ((lanes != 32 && lanes != 64) || !llvm::isPowerOf2_32(bins) || !warps)
      return fail("unsupported histogram bin/warp geometry");
    double regs = ttg::getTotalElemsPerThread(src), perLane = bins / lanes,
           bits = std::log2(bins), laneBits = std::log2(lanes), maskWords = lanes / 32.;
    work["collective_ballot_sites"] = regs * (bits + double(bool(histogram.getMask())));
    work["collective_popcount_sites"] = regs * perLane;
    work["collective_bitwise_word_work"] = regs * (bits + laneBits * (1 + 2 * maskWords) + maskWords +
        2 * maskWords * perLane * (bits - laneBits));
    work["collective_parallel_word_work"] = regs * perLane;
    work["lds_store_word_sites"] = std::ceil(double(bins) / (lanes * warps));
    work["lds_atomic_word_sites"] = perLane;
    work["lds_load_word_sites"] = double(ttg::getTotalElemsPerThread(dst));
    work["lds_block_sync_sites"] = 2.;
    row["route"] = "warp-bit-ballots+shared-atomic-histogram";
    row["input_register_values"] = regs; row["padded_bins"] = int64_t(bins);
    row["input_distribution_known"] = false;
  } else return fail("unknown collective");
  row["available"] = true; row["reason"] = ""; row["work"] = std::move(work);
  return row;
}

llvm::json::Object atomicGeometry(tt::AtomicRMWOp op, tt::ModuleAxisInfoAnalysis &axes) {
  llvm::json::Object row{{"available", false}, {"result_used", !op.getResult().use_empty()},
      {"rmw", tt::stringifyRMWOp(op.getAtomicRmwOp()).str()},
      {"memory_semantic", tt::stringifyMemSemantic(op.getSem()).str()},
      {"scope", tt::stringifyMemSyncScope(op.getScope()).str()},
      {"contention_known", false}, {"retry_count_known", false},
      {"emitted_instruction_count_known", false}, {"mask_contents_known", false}};
  auto fail = [&](const char *reason) { row["reason"] = reason; return std::move(row); };
  auto ptr = dyn_cast<tt::PointerType>(getElementTypeOrSelf(op.getPtr().getType()));
  if (!ptr || (ptr.getAddressSpace() != 1 && ptr.getAddressSpace() != 3)) return fail("unsupported atomic address space");
  bool shared = ptr.getAddressSpace() == 3;
  Type element = getElementTypeOrSelf(op.getVal().getType());
  if (!element.isIntOrFloat()) return fail("non-numeric atomic payload");
  unsigned bits = element.getIntOrFloatBitWidth();
  if (bits != 16 && bits != 32 && bits != 64) return fail("unsupported atomic width");
  // Scalarize ADDRESS geometry only. This is neither the number of packed
  // atomic instructions nor an estimate of actual contended requests.
  auto geometry = shared ? sharedWordGeometry(op.getVal()) : accessGeometry(op.getVal(), op.getPtr(), op.getMask(), {}, axes, 1);
  if (!geometry.getBoolean("available").value_or(false)) return fail("unknown atomic address geometry");
  bool scalar = !isa<RankedTensorType>(op.getVal().getType());
  double elements = scalar ? 0.0 : geometry.getNumber(shared ? "elements" : "unique_registers").value_or(0);
  double words = elements * ((bits + 31) / 32);
  bool used = !op.getResult().use_empty();
  bool resultLDS = scalar && used;
  if (!scalar && used) {
    auto layout = ttg::toLinearLayout(cast<RankedTensorType>(op.getType()));
    for (auto item : layout.getFreeVariableMasks()) resultLDS |= item.second != 0;
  }
  double resultWords = resultLDS ? (scalar ? double((bits + 31) / 32) : words) : 0.0;
  bool acquire = op.getSem() == tt::MemSemantic::ACQUIRE || op.getSem() == tt::MemSemantic::ACQUIRE_RELEASE;
  bool release = op.getSem() == tt::MemSemantic::RELEASE || op.getSem() == tt::MemSemantic::ACQUIRE_RELEASE;
  auto rmw = op.getAtomicRmwOp();
  bool bitwise = rmw == tt::RMWOp::AND || rmw == tt::RMWOp::OR || rmw == tt::RMWOp::XOR;
  bool exchange = rmw == tt::RMWOp::XCHG;
  bool floating = rmw == tt::RMWOp::FADD;
  llvm::json::Object work{
      {"atomic_tensor_element_work", elements}, {"atomic_tensor_word_work", words},
      {"atomic_tensor_bytes", elements * bits / 8.0},
      {"atomic_address_segment_work", scalar ? 0.0 : geometry.getNumber("work").value_or(0)},
      {"atomic_scalar_sites", scalar ? 1.0 : 0.0},
      {"atomic_scalar_bytes", scalar ? bits / 8.0 : 0.0},
      {"atomic_return_word_work", used ? words : 0.0},
      {"atomic_scalar_return_sites", scalar && used ? 1.0 : 0.0},
      {"atomic_result_lds_store_words", resultWords}, {"atomic_result_lds_load_words", resultWords},
      {"atomic_result_block_sync_sites", resultLDS ? 1.0 : 0.0},
      {"atomic_float_add_work", floating ? words : 0.0},
      {"atomic_integer_arithmetic_work", !floating && !bitwise && !exchange ? words : 0.0},
      {"atomic_bitwise_work", bitwise ? words : 0.0},
      {"atomic_exchange_work", exchange ? words : 0.0},
      {"atomic_acquire_sites", acquire ? 1.0 : 0.0}, {"atomic_release_sites", release ? 1.0 : 0.0},
      {"atomic_system_scope_sites", op.getScope() == tt::MemSyncScope::SYSTEM ? 1.0 : 0.0},
      {"atomic_gpu_scope_sites", op.getScope() == tt::MemSyncScope::GPU ? 1.0 : 0.0},
      {"atomic_cta_scope_sites", op.getScope() == tt::MemSyncScope::CTA ? 1.0 : 0.0}};
  row["address_space"] = int64_t(ptr.getAddressSpace());
  if (shared) {
    for (const char *key : {"atomic_tensor_element_work", "atomic_tensor_word_work", "atomic_tensor_bytes",
         "atomic_address_segment_work", "atomic_scalar_sites", "atomic_scalar_bytes"}) work[key] = 0.;
    work["lds_atomic_word_sites"] = (scalar ? 1. : elements) * ((bits + 31) / 32);
  }
  row["available"] = true; row["address_space"] = 1; row["element_bits"] = int64_t(bits);
  row["kind"] = scalar ? "scalar" : "tensor";
  row["units"] = "logical-scalar-sites-or-per-thread-unique-tensor-work-not-dynamic-atomic-requests";
  row["packing_and_wave_reduction_known"] = false;
  row["result_lds_broadcast"] = resultLDS;
  row["scratch_already_owned_by_resources"] = true;
  row["address_geometry"] = std::move(geometry); row["work"] = std::move(work);
  return row;
}
// Mirror the two native GatherOpToLLVM routes. Counts describe primitive
// lowering work, not latency or runtime-dependent indices/bank conflicts.
llvm::json::Object gatherGeometry(tt::GatherOp op) {
  llvm::json::Object row{{"available", false}, {"axis", int64_t(op.getAxis())},
      {"data_dependent_indices_known", false}, {"emitted_instruction_count_known", false}};
  auto fail = [&](const char *reason) { row["reason"] = reason; return std::move(row); };
  auto src = op.getSrc().getType(), idx = op.getIndices().getType();
  if (!src.getEncoding() || !idx.getEncoding() ||
      !isa<ttg::DistributedEncodingTrait>(src.getEncoding()) ||
      !isa<ttg::DistributedEncodingTrait>(idx.getEncoding()) ||
      !src.getElementType().isIntOrFloat())
    return fail("unsupported distributed gather types");
  auto *ctx = op.getContext();
  auto reg = StringAttr::get(ctx, "register"), block = StringAttr::get(ctx, "block");
  auto srcLayout = ttg::toLinearLayout(src), idxLayout = ttg::toLinearLayout(idx);
  if (srcLayout.getInDimSize(block) != 1 || idxLayout.getInDimSize(block) != 1)
    return fail("cross-CTA gather");
  unsigned srcRegs = srcLayout.getInDimSize(reg), idxRegs = idxLayout.getInDimSize(reg);
  unsigned words = (src.getElementTypeBitWidth() + 31) / 32;
  GatherLoweringHelper helper(op);
  llvm::json::Object work;
  bool warpLocal = helper.isWarpLocal();
  if (warpLocal) {
    SmallVector<StringAttr> dims;
    for (unsigned i = 0; i < unsigned(src.getRank()); ++i)
      dims.push_back(StringAttr::get(ctx, "dim" + std::to_string(i)));
    auto gatherDim = dims[op.getAxis()];
    auto inverse = srcLayout.pseudoinvert().sublayout(dims, {reg});
    inverse = inverse.removeZeroBasesAlongDim(gatherDim);
    unsigned perColumn = inverse.getInDimSize(gatherDim);
    // The generic lowering emits a shuffle, compare and select for each
    // source register in the gathered column, for every destination register.
    work["shuffle_word_ops"] = double(idxRegs) * perColumn * words;
    work["payload_regular_work"] = double(idxRegs) * perColumn * 2;
    row["source_registers_per_column"] = int64_t(perColumn);
    row["route"] = "warp-shuffle-and-select";
  } else {
    work["lds_store_word_sites"] = double(srcRegs) * words;
    work["lds_load_word_sites"] = double(idxRegs) * words;
    work["explicit_block_sync_sites"] = 1.0;
    row["route"] = "shared-memory";
    row["scratch_bytes_already_owned_by_resources"] = int64_t(helper.getScratchSizeInBytes());
  }
  work["index_regular_work"] = double(idxRegs) * src.getRank() +
      (warpLocal ? 0.0 : double(srcRegs) * src.getRank());
  row["available"] = true;
  row["source_registers_per_thread"] = int64_t(srcRegs);
  row["index_registers_per_thread"] = int64_t(idxRegs);
  row["element_words"] = int64_t(words);
  row["work"] = std::move(work);
  return row;
}
} // namespace

llvm::json::Object analyzeLayoutAccessHeuristic(
    Value value, Value address, Value mask, Value base,
    tt::ModuleAxisInfoAnalysis &axes, RankedTensorType candidateType) {
  return accessGeometry(value, address, mask, base, axes, 0, candidateType);
}

llvm::json::Object analyzeLayoutConversionFacts(ttg::ConvertLayoutOp op) {
  auto src = cast<RankedTensorType>(op.getSrc().getType());
  auto dst = cast<RankedTensorType>(op.getType());
  auto *ctx = op.getContext();
  auto reg = StringAttr::get(ctx, "register"), lane = StringAttr::get(ctx, "lane"),
       warp = StringAttr::get(ctx, "warp"), block = StringAttr::get(ctx, "block"),
       reps = StringAttr::get(ctx, "reps");
  unsigned bits = tt::getBitwidth(src);
  auto srcLL = ttg::toLinearLayout(src), dstLL = ttg::toLinearLayout(dst);
  auto srcUnique = tt::actionRemoveBroadcastedRegs(srcLL).apply(srcLL);
  auto dstUnique = tt::actionRemoveBroadcastedRegs(dstLL).apply(dstLL);
  unsigned srcRegs = srcUnique.getInDimSize(reg), dstRegs = dstUnique.getInDimSize(reg);
  auto conversion = mlir::minimalCvtLayout(src, dst);
  llvm::json::Object facts;
  facts = llvm::json::Object{{"src_type", printed(Type(src))}, {"dst_type", printed(Type(dst))},
      {"bitwidth", int64_t(bits)}, {"src_unique_elems_per_thread", int64_t(srcRegs)},
      {"dst_unique_elems_per_thread", int64_t(dstRegs)}, {"calibrated", false}};
  // Blocked.order can influence later backend heuristics even when the small
  // tensor's LinearLayout bases coincide. Describe its register grouping in
  // physical units, rather than using an encoding name or axis ID as a feature.
  double orderWork = 0, orderMismatches = 0;
  for (auto type : {src, dst}) {
    if (!type.getRank()) { orderWork += std::max(1u, (bits + 31) / 32); continue; }
    auto linear = ttg::toLinearEncoding(type);
    auto order = declaredBlockedOrder(type.getEncoding());
    if (order.empty()) order = linear.getOrder();
    auto contig = linear.getContigPerThread();
    auto layout = ttg::toLinearLayout(type);
    unsigned uniqueRegs = tt::actionRemoveBroadcastedRegs(layout).apply(layout).getInDimSize(reg);
    orderWork += double(uniqueRegs) * std::max(1u, (bits + 31) / 32) /
                 std::max(1u, contig[order.front()]);
    orderMismatches += order.front() != linear.getOrder().front();
  }
  facts["declared_order_register_word_groups"] = orderWork;
  facts["declared_order_minor_disagreements"] = orderMismatches;
  facts["declared_order_units"] = "register-word-groups-in-declared-minor-axis; not emitted instructions";
  auto dims = conversion.getOutDimNames();
  if (llvm::is_contained(dims, block)) {
    facts["route"] = "cross-cta-unsupported";
  } else if (mlir::cvtReordersRegisters(src, dst)) {
    facts["route"] = "register-only";
    facts["reason"] = "native LLVM SSA reordering; alias-aware liveness tracked separately";
  } else if (mlir::cvtNeedsWarpShuffle(src, dst)) {
    facts["route"] = "warp-shuffle";
    auto decomposition = mlir::getWarpLayoutConvertDecomposition(src, dst, bits);
    bool lanePermutation = !tt::squareSublayoutIsIdentity(decomposition.pLane, lane);
    unsigned packed = decomposition.pReg.getInDimSize(reg) >> decomposition.nPack;
    packed *= std::max(1u, (bits + 31) / 32);
    unsigned mixed = decomposition.mixedTranspositions.size();
    facts["mixed_transpositions"] = int64_t(mixed);
    facts["lane_permutation"] = lanePermutation;
    facts["packing_log2"] = int64_t(decomposition.nPack);
    facts["packed_register_words"] = int64_t(packed);
    facts["target_specialization"] = "generic decomposition; HCU permlane specialization may reduce instructions";
  } else {
    facts["route"] = "lds";
    auto withinBlock = [&](const tt::LinearLayout &layout) {
      auto within = layout.sublayout({reg, lane, warp}, llvm::to_vector(layout.getOutDimNames()));
      return tt::actionRemoveBroadcastedRegs(within).apply(within);
    };
    auto shared = ttg::optimalSwizzlingLdSt(withinBlock(srcLL), withinBlock(dstLL), std::max(8u, bits));
    unsigned iterations = shared.getInDimSize(reps);
    unsigned swizzledBytes = tt::getNumScratchElemsSwizzledCvt(src, dst) * std::max(8u, bits) / 8;
    auto padded = tt::HCU::getScratchConfigForCvt(src, dst);
    unsigned paddedBytes = tt::HCU::getConvertLayoutScratchInBytes(src, dst, true);
    bool warpSync = mlir::isCvtWarpSync(srcLL, dstLL);
    facts["warp_sync"] = warpSync;
    facts["swizzled_repetitions"] = int64_t(iterations);
    facts["swizzled_sync_points"] = int64_t(2 * iterations - 1);
    facts["scratch_swizzled_bytes"] = int64_t(swizzledBytes);
    facts["scratch_padded_bytes"] = int64_t(paddedBytes);
    facts["padded_store_vector"] = int64_t(padded.inVec);
    facts["padded_load_vector"] = int64_t(padded.outVec);
    facts["strategy"] = op->hasAttr(tt::HCU::AttrSharedMemPadded) ? "padded" : "swizzled-current";
    facts["strategy_uncertainty"] = "later OptimizeLDSUsage may change strategy or split the conversion; scratch sizes are alternatives, not additive";
    // Vector op/address counts are proxies, not emitted instruction counts.
    double vectorOps = std::ceil(double(srcRegs) / padded.inVec) + std::ceil(double(dstRegs) / padded.outVec);
    facts["padded_vector_ops_proxy"] = vectorOps;
    facts["estimate_basis"] = "swizzled repetitions with padded-vector proxy; heuristic, not emitted instruction counts";
  }
  return facts;
}

llvm::json::Object analyzeLayoutReductionFacts(ModuleOp module,
                                   const DenseMap<Operation *, unsigned> &ids,
                                   const LayoutExecutionContext *context,
                                   bool modelExtensions) {
  auto executionWeight = [&](Operation *op) { return layoutExecutionWeight(op, context); };
  llvm::json::Array operations, issues;
  std::map<std::string, double> totals;
  for (auto key : {"reduce_local_combine_work", "reduce_lane_combine_work", "reduce_lds_store_words",
                   "reduce_lds_load_words", "reduce_block_sync_sites"}) totals[key] = 0;
  if (modelExtensions) {
    totals["reduce_combine_word_work"] = 0;
    totals["reduce_shuffle_word_work"] = 0;
  }
  module.walk([&](tt::ReduceOp op) {
    llvm::json::Object row{{"operation", int64_t(ids.lookup(op))}, {"id_scope", "materialized-candidate"},
        {"axis", int64_t(op.getAxis())}, {"frequency", executionWeight(op)}};
    auto unknown = [&](const char *why) {
      row["available"] = false; row["reason"] = why; operations.push_back(std::move(row)); issues.push_back(why);
    };
    if (!op->getNumOperands()) { unknown("empty reduction"); return; }
    auto type = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    // The evaluation prefix has already accelerated dot accumulators. Their
    // reductions use MMA/linear encodings; the native reduction lowering and
    // ReduceOpHelper operate on the same LinearLayout for those encodings.
    if (!type || !type.getEncoding() ||
        !(modelExtensions ? isa<ttg::DistributedEncodingTrait>(type.getEncoding())
                          : isa<ttg::BlockedEncodingAttr>(rootEncoding(type.getEncoding())))) {
      unknown("unsupported reduction encoding"); return;
    }
    ReduceOpHelper helper(op);
    if (!helper.isReduceWithinCTA()) { unknown("cross-CTA reduction"); return; }
    auto linear = ttg::toLinearLayout(type);
    auto reg = StringAttr::get(module.getContext(), "register");
    if (linear.getInDimSizeLog2(reg) > 12) { unknown("reduction register analysis budget"); return; }
    auto block = StringAttr::get(module.getContext(), "block");
    if (linear.getInDimSize(block) != 1) { unknown("multi-CTA reduction proxy"); return; }
    llvm::json::Object basis;
    for (const auto &[name, vectors] : linear.getBases()) {
      llvm::json::Array list;
      for (const auto &v : vectors) {
        llvm::json::Array coords; for (int x : v) coords.push_back(x);
        list.push_back(std::move(coords));
      }
      basis[name.getValue()] = std::move(list);
    }
    std::set<std::vector<int32_t>> unique, groups;
    for (int i = 0; i < linear.getInDimSize(reg); ++i) {
      SmallVector<std::pair<StringAttr, int32_t>> input;
      for (auto name : linear.getInDimNames()) input.push_back({name, name == reg ? i : 0});
      std::vector<int32_t> offset;
      for (auto [name, value] : linear.apply(input)) { (void)name; offset.push_back(value); }
      unique.insert(offset); offset[op.getAxis()] = 0; groups.insert(offset);
    }
    unsigned combineOps = 0, combineWordOps = 0; bool supported = true;
    llvm::json::Array combiner, combinerIds;
    op.getCombineOp().walk([&](Operation *node) {
      if (isa<tt::ReduceReturnOp, arith::ConstantOp>(node)) return;
      auto name = node->getName().getStringRef();
      if (node->getNumRegions() || (!name.starts_with("arith.") && !name.starts_with("math."))) supported = false;
      for (Type result : node->getResultTypes()) {
        if (!result.isIntOrFloat()) supported = false;
        else combineWordOps += (result.getIntOrFloatBitWidth() + 31) / 32;
      }
      ++combineOps; combiner.push_back(name.str()); combinerIds.push_back(int64_t(ids.lookup(node)));
    });
    unsigned wordsPerTuple = 0;
    for (Type t : op.getElementTypes()) {
      if (!t.isIntOrFloat() || (!modelExtensions && t.getIntOrFloatBitWidth() == 1)) supported = false;
      else wordsPerTuple += (t.getIntOrFloatBitWidth() + 31) / 32;
    }
    if (!supported || !combineOps) { unknown("unknown combiner or target i1 fastpath"); return; }
    unsigned lanes = helper.getIntraWarpSizeWithUniqueData(), warps = helper.getInterWarpSizeWithUniqueData();
    bool crossWarp = !helper.isWarpSynchronous();
    auto lane = StringAttr::get(module.getContext(), "lane"), warp = StringAttr::get(module.getContext(), "warp");
    uint64_t threads = uint64_t(linear.getInDimSize(lane)) * linear.getInDimSize(warp);
    auto scratchShape = helper.getScratchRepShape();
    uint64_t scratchElements = 1;
    for (unsigned size : scratchShape) scratchElements *= size;
    uint64_t rounds = crossWarp ? std::max<uint64_t>(1, scratchElements / threads) : 0;
    unsigned resultWords = 0;
    for (Value value : op->getResults()) {
      Type t = value.getType(); unsigned n = isa<RankedTensorType>(t) ? ttg::getTotalElemsPerThread(t) : 1;
      Type elem = isa<RankedTensorType>(t) ? cast<RankedTensorType>(t).getElementType() : t;
      resultWords += n * ((elem.getIntOrFloatBitWidth() + 31) / 32);
    }
    double local = double(unique.size() - groups.size()) * combineOps;
    double laneCalls = groups.size() * std::log2(lanes) + rounds * std::log2(warps);
    double laneWork = laneCalls * combineOps;
    double store = crossWarp ? (groups.size() + rounds) * wordsPerTuple : 0;
    double load = crossWarp ? rounds * wordsPerTuple + resultWords : 0;
    std::map<std::string, double> work{{"reduce_local_combine_work", local}, {"reduce_lane_combine_work", laneWork},
        {"reduce_lds_store_words", store}, {"reduce_lds_load_words", load}, {"reduce_block_sync_sites", crossWarp ? 2.0 : 0.0}};
    if (modelExtensions) {
      work["reduce_combine_word_work"] = (double(unique.size() - groups.size()) + laneCalls) * combineWordOps;
      work["reduce_shuffle_word_work"] = laneCalls * wordsPerTuple;
    }
    llvm::json::Object numeric;
    for (const auto &[key, value] : work) { numeric[key] = number(value); totals[key] += executionWeight(op) * value; }
    llvm::json::Array shape; for (int64_t size : type.getShape()) shape.push_back(size);
    llvm::json::Array resultTypes; for (Value v : op->getResults()) resultTypes.push_back(printed(v.getType()));
    row["available"] = true; row["shape"] = std::move(shape); row["linear_bases"] = std::move(basis);
    row["unique_input_registers"] = int64_t(unique.size()); row["accumulator_groups"] = int64_t(groups.size());
    row["combiner_operations"] = std::move(combiner); row["combiner_operation_count"] = int64_t(combineOps);
    row["combiner_word_operation_count"] = int64_t(combineWordOps);
    row["shuffle_work_policy"] = "generic tree transport words; target fastpath instruction selection not inferred";
    row["combiner_operation_ids"] = std::move(combinerIds);
    row["unique_lanes"] = int64_t(lanes); row["unique_warps"] = int64_t(warps); row["threads_per_cta"] = int64_t(threads);
    row["cross_warp"] = crossWarp; row["scratch_elements"] = int64_t(scratchElements);
    row["scratch_bytes"] = int64_t(helper.getScratchSizeInBytes()); row["partial_rounds"] = int64_t(rounds);
    row["tuple_words"] = int64_t(wordsPerTuple); row["result_load_words"] = int64_t(resultWords);
    row["result_types"] = std::move(resultTypes); row["work"] = std::move(numeric);
    operations.push_back(std::move(row));
  });
  llvm::json::Object sums; for (const auto &[key, value] : totals) sums[key] = number(value);
  bool available = issues.empty();
  return llvm::json::Object{{"schema", 1}, {"version", "native-reduction-work-v1"}, {"available", available},
      {"operations", std::move(operations)}, {"issues", std::move(issues)}, {"totals", std::move(sums)},
      {"units", "generic-lowering-static-work-proxies-before-predicate-CSE-target-specialization"},
      {"ownership", "one actual reduce definition; scalar combiner excluded from tensor payload; scratch bytes already in LDS resources, not a new charge"}};
}

llvm::json::Object analyzeLayoutFacts(ModuleOp module, unsigned numStages,
                                    bool diagnostics, llvm::StringRef executionPayload) {
  LayoutExecutionContext context(module, executionPayload);
  auto executionWeight = [&](Operation *op) { return layoutExecutionWeight(op, &context); };
  std::map<std::string, double> totals;
  for (const char *key : {
      "payload_regular_work", "payload_divrem_work", "payload_special_math_work",
      "scalar_arithmetic_sites", "dot_sites", "dot_multiply_accumulate_work",
      "load_sites", "store_sites", "masked_load_sites", "masked_store_sites",
      "load_work", "store_work", "load_instructions", "store_instructions",
      "load_vector_elements_sum", "store_vector_elements_sum",
      "load_element_bytes_sum", "store_element_bytes_sum",
      "shuffle_word_ops", "lds_vector_ops", "lds_block_sync_sites", "lds_warp_sync_sites",
      "conversion_address_work", "packed_live_words", "lds_allocated_bytes",
      "conversion_temporary_words", "reduce_local_combine_work", "reduce_lane_combine_work",
      "reduce_lds_store_words", "reduce_lds_load_words", "reduce_block_sync_sites",
      "index_regular_work", "index_dynamic_divrem_work", "index_coordinate_work",
      "dynamic_loop_sites", "conditional_sites", "atomic_sites", "opaque_operation_sites",
      "unknown_memory_sites", "num_warps", "threads_per_warp", "num_ctas", "num_stages",
      "scalar_load_sites", "scalar_store_sites", "scalar_load_bytes", "scalar_store_bytes",
      "cfg_conditional_sites", "cfg_unconditional_sites", "cfg_backedge_sites", "explicit_block_sync_sites",
      "payload_bitcast_sites", "payload_float_convert_elements", "payload_float_convert_input_words",
      "payload_float_convert_output_words", "payload_float_convert_fp8_elements", "payload_float_convert_bf16_elements",
      "payload_float_rounding_sites", "payload_integer_convert_elements", "payload_clamp_elements",
      "scan_local_combine_work", "scan_lane_combine_work", "scan_carry_combine_work",
      "scan_shuffle_word_work", "scan_reverse_shuffle_word_work", "scan_lds_store_words",
      "scan_lds_load_words", "scan_block_sync_sites", "extern_math_element_work",
      "extern_power_element_work", "extern_input_word_work", "extern_output_word_work",
      "extern_constant_exponent_work", "extern_dynamic_exponent_work", "extern_exponent_magnitude_work",
      "atomic_tensor_element_work", "atomic_tensor_word_work", "atomic_tensor_bytes",
      "atomic_address_segment_work", "atomic_scalar_sites", "atomic_scalar_bytes",
      "atomic_return_word_work", "atomic_scalar_return_sites", "atomic_float_add_work",
      "atomic_result_lds_store_words", "atomic_result_lds_load_words", "atomic_result_block_sync_sites",
      "atomic_integer_arithmetic_work", "atomic_bitwise_work", "atomic_exchange_work",
      "atomic_acquire_sites", "atomic_release_sites", "atomic_system_scope_sites",
      "atomic_gpu_scope_sites", "atomic_cta_scope_sites"})
    totals[key] = 0;
  for (const char *key : {"lds_load_word_sites", "lds_store_word_sites", "lds_atomic_word_sites",
       "lds_pointer_coordinate_sites", "collective_parallel_word_work", "collective_serial_word_work",
       "collective_ballot_sites", "collective_popcount_sites", "collective_bitwise_word_work"}) totals[key] = 0;
  totals["conversion_declared_order_work"] = 0;
  totals["conversion_declared_order_mismatch_sites"] = 0;
  llvm::json::Array issues, conversions, accesses, controlFlow, typeOperations, scans, externs, atomics;
  auto issue = [&](const std::string &message) { issues.push_back(message); };
  if (!context.valid()) issue(context.failure());
  for (const auto &item : context.scaleFacts(module))
    totals[item.first.str()] = item.second.getAsNumber().value_or(0);
  DenseMap<Operation *, unsigned> ids;
  std::vector<double> executionWeights;
  std::vector<Operation *> orderedOperations;
  module.walk<WalkOrder::PreOrder>([&](Operation *op) {
    ids[op] = ids.size();
    executionWeights.push_back(executionWeight(op));
    orderedOperations.push_back(op);
  });
  module.walk([&](ttg::ConvertLayoutOp op) {
    auto facts=analyzeLayoutConversionFacts(op);
    facts["operation"]=int64_t(ids.lookup(op));
    facts["execution_weight"]=executionWeight(op);
    totals["conversion_declared_order_work"] += executionWeight(op) *
        facts.getNumber("declared_order_register_word_groups").value_or(0);
    totals["conversion_declared_order_mismatch_sites"] += executionWeight(op) *
        facts.getNumber("declared_order_minor_disagreements").value_or(0);
    if (facts.getString("route")=="cross-cta-unsupported") issue("cross-CTA conversion");
    if (facts.getString("route")=="warp-shuffle")
      totals["shuffle_word_ops"]+=executionWeight(op)*facts.getNumber("packed_register_words").value_or(0)*
          std::max(1.0,facts.getNumber("mixed_transpositions").value_or(0)+double(facts.getBoolean("lane_permutation").value_or(false)));
    conversions.push_back(std::move(facts));
  });
  auto resources=analyzeLayoutResources(module,conversions);
  auto index=analyzeLayoutIndex(module,conversions,true);
  auto reductions=analyzeLayoutReductionFacts(module,ids,&context,true);
  if (resources.getInteger("unmodeled_values").value_or(0)) issue("unmodeled resource value types");
  if (auto *p=resources.getObject("packed_peak")) totals["packed_live_words"]=p->getNumber("packed_words").value_or(0);
  if (auto *p=resources.getObject("with_conversion_temporaries_peak"))
    totals["conversion_temporary_words"]=p->getNumber("extra_temporary_words_proxy").value_or(0);
  if (auto *p=resources.getObject("lds_current");p && p->getBoolean("available").value_or(false))
    totals["lds_allocated_bytes"]=p->getNumber("allocated_bytes").value_or(0);
  else issue("unknown LDS allocation");
  if (!reductions.getBoolean("available").value_or(false)) {
    issue("unknown reduction work");
    if (auto *reasons = reductions.getArray("issues"))
      for (const auto &reason : *reasons)
        if (auto text = reason.getAsString()) issue("reduction: " + text->str());
  }
  if (auto *values=reductions.getObject("totals"))
    for (const auto &entry:*values) totals[entry.first.str()]=entry.second.getAsNumber().value_or(0);
  std::set<unsigned> indexOps;
  if (auto *ops=index.getArray("operations")) for (const auto &item:*ops) {
    const auto &row=*item.getAsObject();
    indexOps.insert(row.getInteger("operation").value_or(-1));
    auto category=row.getString("category").value_or("");
    if (row.getString("owner").value_or("")!="index-expression" ||
        category=="constant" || category=="program-coordinate" || category=="layout-coordinate" ||
        category=="shape-view" || category=="pointer-integer-cast") continue;
    auto id = row.getInteger("operation");
    double ratio = 1;
    if (id && *id >= 0 && uint64_t(*id) < executionWeights.size()) {
      double oldWeight = layoutExecutionWeight(orderedOperations[*id]);
      ratio = oldWeight ? executionWeights[*id] / oldWeight : 0;
    }
    totals[category=="div-rem-dynamic"?"index_dynamic_divrem_work":"index_regular_work"]+=
        ratio * row.getNumber("weighted_work_proxy").value_or(0);
  }
  if (auto *implicit=index.getArray("implicit_lowering")) for (const auto &item:*implicit) {
    const auto &row=*item.getAsObject();
    if(row.getString("kind")=="make-range-coordinate") {
      auto operation = row.getInteger("operation");
      if (!operation || *operation < 0 || uint64_t(*operation) >= executionWeights.size()) {
        issue("missing coordinate operation identity");
      } else if(auto *mapping=row.getObject("mapping"))
        totals["index_coordinate_work"]+=mapping->getNumber("lane_warp_output_terms").value_or(0)*
            executionWeights[*operation];
    }
    if(row.getString("owner")!="conversion")continue;
    auto oi=row.getInteger("operation");
    const llvm::json::Object *conversion=nullptr;
    for(const auto &c:conversions)if(c.getAsObject()->getInteger("operation")==oi)conversion=c.getAsObject();
    if(!conversion || conversion->getString("route")!="lds")continue;
    auto *scenarios=row.getArray("scratch_address_scenarios");
    if(!scenarios){issue("missing LDS address scenarios");continue;}
    auto wanted=conversion->getString("strategy")=="padded"?"padded":"swizzled";
    bool found=false;double repeats=0,weight=conversion->getNumber("execution_weight").value_or(1);
    for(const auto &item:*scenarios){const auto &scenario=*item.getAsObject();
      if(scenario.getString("strategy")!=wanted)continue;
      found=true;
      for(const char *side:{"store","load"}){
        const auto *facts=scenario.getObject(side);
        if(!facts || !facts->getBoolean("available").value_or(false)){issue("unknown LDS address side");continue;}
        totals["lds_vector_ops"]+=weight*facts->getNumber("vector_addresses_before_cse").value_or(0);
        repeats=std::max(repeats,facts->getNumber("repetitions").value_or(0));
        if(auto *dag=facts->getArray("address_template_dag"))for(const auto &node:*dag){const auto &n=*node.getAsObject();
          if(!n.getString("op").value_or("").starts_with("linear-layout-"))
            totals["conversion_address_work"]+=weight*n.getNumber("instances_before_cse").value_or(0);
        }
      }
    }
    if(!found || repeats<1)issue("missing active LDS strategy");
    totals[conversion->getBoolean("warp_sync").value_or(false)?"lds_warp_sync_sites":"lds_block_sync_sites"]+=
        weight*std::max(0.0,2*repeats-1);
  }
  tt::ModuleAxisInfoAnalysis axes(module);
  DominanceInfo dominance(module);
  llvm::json::Object census;
  module.walk([&](Operation *op) {
    auto name=op->getName().getStringRef();double weight=executionWeight(op);
    census[name]=census.getInteger(name).value_or(0)+1;
    // Coverage is a semantic contract, not just finite arithmetic. New
    // collectives, descriptor/async memory and custom operations need facts
    // before a learned model may claim a complete candidate.
    static const std::set<std::string> structuralOrHandled = {
        "builtin.module", "tt.func", "tt.return", "tt.reduce", "tt.reduce.return", "tt.scan", "tt.scan.return",
        "scf.for", "scf.yield", "scf.if", "scf.while", "scf.condition", "scf.execute_region",
        "tt.addptr", "tt.get_program_id", "tt.get_num_programs", "tt.make_range",
        "tt.splat", "tt.broadcast", "tt.expand_dims", "tt.trans", "tt.reshape",
        "tt.join", "tt.split", "tt.cat", "tt.gather", "tt.int_to_ptr", "tt.ptr_to_int",
        "tt.dot", "tt.dot_scaled", "tt.load", "tt.store", "ttg.convert_layout",
        "hcug.buffer_load", "hcug.buffer_store", "cf.br", "cf.cond_br",
        "ttg.local_alloc", "ttg.local_load", "ttg.local_store", "ttg.local_dealloc",
        "ttg.memdesc_index", "ttg.memdesc_subslice", "ttg.memdesc_trans", "ttg.memdesc_reshape",
        "tle.local_pointers", "tle.exclusive_cumsum", "tt.histogram",
        "tt.bitcast", "tt.fp_to_fp", "tt.clampf", "gpu.barrier"};
    if (!name.starts_with("arith.") && !name.starts_with("math.") &&
        !structuralOrHandled.count(name.str()) && !name.contains("atomic") &&
        !name.contains("inline_asm") && !name.contains("extern"))
      issue("operation requires a physical fact adapter: " + name.str());
    if(auto loop=dyn_cast<scf::ForOp>(op)) {
      auto constant=[](Value v){return bool(v.getDefiningOp<arith::ConstantOp>());};
      if(!constant(loop.getLowerBound()) || !constant(loop.getUpperBound()) || !constant(loop.getStep()))
        totals["dynamic_loop_sites"]+=1;
    }
    if(isa<scf::WhileOp>(op))totals["dynamic_loop_sites"]+=1;
    if(isa<scf::IfOp>(op))totals["conditional_sites"]+=1;
    if (name == "gpu.barrier") totals["explicit_block_sync_sites"] += weight;
    if (name == "cf.br" || name == "cf.cond_br") {
      bool conditional = name == "cf.cond_br";
      totals[conditional ? "cfg_conditional_sites" : "cfg_unconditional_sites"] += weight;
      if (conditional) totals["conditional_sites"] += 1;
      unsigned backedges = 0;
      for (Block *successor : op->getSuccessors())
        backedges += dominance.dominates(successor, op->getBlock());
      totals["cfg_backedge_sites"] += backedges;
      controlFlow.push_back(llvm::json::Object{{"operation", int64_t(ids.lookup(op))},
          {"conditional", conditional}, {"successors", int64_t(op->getNumSuccessors())},
          {"natural_backedges", int64_t(backedges)}, {"static_enclosing_loop_weight", weight},
          {"execution_frequency_known", false}, {"branch_probabilities_known", false},
          {"units", "syntactic-CFG-sites-not-dynamic-branch-counts"}});
    }
    bool bitcast = name == "tt.bitcast" || name == "arith.bitcast";
    bool floatCast = name == "tt.fp_to_fp" || name == "arith.extf" || name == "arith.truncf";
    bool intCast = name == "arith.extsi" || name == "arith.extui" || name == "arith.trunci" ||
                   name == "arith.index_cast" || name == "arith.index_castui";
    bool clamp = name == "tt.clampf";
    bool typeOperation = bitcast || floatCast || intCast || clamp;
    if (typeOperation && op->getNumOperands() && op->getNumResults()) {
      Type src = getElementTypeOrSelf(op->getOperand(0).getType());
      Type dst = getElementTypeOrSelf(op->getResult(0).getType());
      double elements = 1;
      if (auto tensor = dyn_cast<RankedTensorType>(op->getResult(0).getType())) {
        if (!tensor.getEncoding() || !isa<ttg::DistributedEncodingTrait>(tensor.getEncoding())) {
          issue("unmodeled type operation layout");
          elements = 0;
        } else {
          auto layout = ttg::toLinearLayout(tensor);
          elements = tt::actionRemoveBroadcastedRegs(layout).apply(layout).getInDimSize(StringAttr::get(module.getContext(), "register"));
        }
      }
      bool payload = !indexOps.count(ids.lookup(op)) && !op->getParentOfType<tt::ReduceOp>() && !op->getParentOfType<tt::ScanOp>();
      if (payload) {
        if (bitcast) totals["payload_bitcast_sites"] += weight;
        if (intCast) totals["payload_integer_convert_elements"] += weight * elements;
        if (clamp) totals["payload_clamp_elements"] += weight * elements;
        if (floatCast && isa<FloatType>(src) && isa<FloatType>(dst)) {
          unsigned srcBits = src.getIntOrFloatBitWidth(), dstBits = dst.getIntOrFloatBitWidth();
          totals["payload_float_convert_elements"] += weight * elements;
          totals["payload_float_convert_input_words"] += weight * elements * srcBits / 32.0;
          totals["payload_float_convert_output_words"] += weight * elements * dstBits / 32.0;
          if (srcBits == 8 || dstBits == 8) totals["payload_float_convert_fp8_elements"] += weight * elements;
          if (src.isBF16() || dst.isBF16()) totals["payload_float_convert_bf16_elements"] += weight * elements;
          if (op->hasAttr("rounding")) totals["payload_float_rounding_sites"] += weight;
        }
      }
      typeOperations.push_back(llvm::json::Object{{"operation", int64_t(ids.lookup(op))},
          {"source_type", printed(src)}, {"result_type", printed(dst)}, {"elements_per_thread", elements},
          {"execution_weight", weight}, {"owner", payload ? "payload" : "index-or-collective"},
          {"kind", bitcast ? "bit-reinterpretation" : floatCast ? "float-conversion" : intCast ? "integer-conversion" : "clamp"},
          {"emitted_instruction_count_known", false}});
    }
    auto addOperationFacts = [&](llvm::json::Object row, llvm::json::Array &out) {
      row["operation"] = int64_t(ids.lookup(op)); row["execution_weight"] = weight;
      if (!row.getBoolean("available").value_or(false))
        issue(name.str() + ": " + row.getString("reason").value_or("unknown physical work").str());
      else if (auto *work = row.getObject("work"))
        for (const auto &item : *work) totals[item.first.str()] += weight * item.second.getAsNumber().value_or(0);
      out.push_back(std::move(row));
    };
    if (isa<tt::tle::ExclusiveCumsumOp, tt::HistogramOp>(op))
      addOperationFacts(collectiveGeometry(op), scans);
    auto addSharedValue = [&](Value v, const char *key) {
      auto row = sharedWordGeometry(v);
      if (row.getBoolean("available").value_or(false))
        row["work"] = llvm::json::Object{{key, row.getNumber("word_sites").value_or(0)}};
      addOperationFacts(std::move(row), accesses);
    };
    if (auto gather = dyn_cast<tt::GatherOp>(op))
      addOperationFacts(gatherGeometry(gather), accesses);
    if (auto cat = dyn_cast<tt::CatOp>(op)) {
      // CatOpToLLVM concatenates LLVM SSA register lists without a device
      // instruction. Its layouts/conversions and live values are accounted
      // for elsewhere; check the same register-count contract as lowering.
      unsigned lhs = ttg::getTotalElemsPerThread(cat.getLhs().getType());
      unsigned rhs = ttg::getTotalElemsPerThread(cat.getRhs().getType());
      unsigned dst = ttg::getTotalElemsPerThread(cat.getType());
      addOperationFacts(llvm::json::Object{{"available", lhs + rhs == dst},
          {"reason", "cat register concatenation must match native lowering"},
          {"kind", "register-list-concatenation"}, {"runtime_work", 0},
          {"source_registers", int64_t(lhs + rhs)}, {"result_registers", int64_t(dst)},
          {"work", llvm::json::Object{}}}, accesses);
    }
    if (auto alloc = dyn_cast<ttg::LocalAllocOp>(op)) {
      auto type = alloc.getType();
      if (!isa<ttg::SharedMemorySpaceAttr>(type.getMemorySpace())) issue("unsupported local allocation space");
      else if (alloc.getSrc()) addSharedValue(alloc.getSrc(), "lds_store_word_sites");
    }
    if (auto local = dyn_cast<ttg::LocalLoadOp>(op)) addSharedValue(local.getResult(), "lds_load_word_sites");
    if (auto local = dyn_cast<ttg::LocalStoreOp>(op)) addSharedValue(local.getSrc(), "lds_store_word_sites");
    if (auto pointers = dyn_cast<tt::tle::LocalPointersOp>(op)) {
      auto desc = dyn_cast<ttg::MemDescType>(pointers.getSrc().getType());
      auto ptr = dyn_cast<tt::PointerType>(getElementTypeOrSelf(pointers.getResult().getType()));
      if (!desc || !isa<ttg::SharedMemorySpaceAttr>(desc.getMemorySpace()) || !ptr || ptr.getAddressSpace() != 3)
        issue("unsupported TLE local-pointer geometry");
      else {
        double coordinates = 1;
        if (auto t = dyn_cast<RankedTensorType>(pointers.getResult().getType()))
          coordinates = ttg::getTotalElemsPerThread(t);
        totals["lds_pointer_coordinate_sites"] += weight * coordinates * pointers.getIndices().size();
      }
    }
    if (auto scan = dyn_cast<tt::ScanOp>(op)) addOperationFacts(scanGeometry(scan), scans);
    if(name.contains("atomic")){
      totals["atomic_sites"]+=weight;
      if (auto atomic = dyn_cast<tt::AtomicRMWOp>(op)) addOperationFacts(atomicGeometry(atomic, axes), atomics);
      else issue("atomic work requires a dedicated fact adapter: "+name.str());
    }
    if(name.contains("inline_asm") || name.contains("extern")){
      if (auto external = dyn_cast<tt::ExternElementwiseOp>(op)) {
        auto row = externGeometry(external);
        if (op->getParentOfType<tt::ReduceOp>() || op->getParentOfType<tt::ScanOp>() || indexOps.count(ids.lookup(op))) {
          row["available"] = false; row["reason"] = "extern index/collective ownership requires adapter";
        }
        if (!row.getBoolean("available").value_or(false)) totals["opaque_operation_sites"] += weight;
        addOperationFacts(std::move(row), externs);
      } else {
        totals["opaque_operation_sites"]+=weight;
        issue("opaque operation requires a dedicated fact adapter: "+name.str());
      }
    }
    if(name=="tt.dot" || name=="tt.dot_scaled" || name.contains("mfma") || name.contains("mmac")) {
      totals["dot_sites"]+=weight;
      if(op->getNumOperands()>=2) {
        auto a=dyn_cast<RankedTensorType>(op->getOperand(0).getType());
        auto b=dyn_cast<RankedTensorType>(op->getOperand(1).getType());
        if(a && b && a.getRank()>=2 && b.getRank()>=2)
          totals["dot_multiply_accumulate_work"]+=weight*a.getNumElements()*b.getShape().back();
        else issue("unknown dot geometry");
      }
    }
    if((name.starts_with("arith.") || name.starts_with("math.")) && name!="arith.constant" && !typeOperation &&
        !indexOps.count(ids.lookup(op)) && !op->getParentOfType<tt::ReduceOp>() && !op->getParentOfType<tt::ScanOp>()) {
      bool tensor=false;
      for(Value result:op->getResults())if(auto type=dyn_cast<RankedTensorType>(result.getType())) {
        if(!type.getEncoding() || !isa<ttg::DistributedEncodingTrait>(type.getEncoding())){
          issue("unmodeled arithmetic layout");continue;
        }
        auto layout=ttg::toLinearLayout(type);
        auto unique=tt::actionRemoveBroadcastedRegs(layout).apply(layout);
        auto elements=unique.getInDimSize(StringAttr::get(module.getContext(),"register"));
        const char *key=name.contains("div") || name.contains("rem")?"payload_divrem_work":
            name.starts_with("math.")?"payload_special_math_work":"payload_regular_work";
        totals[key]+=weight*elements;tensor=true;
      }
      if(!tensor)totals["scalar_arithmetic_sites"]+=weight;
    }
    Value value,address,mask,base;bool load=false,access=false;
    if(auto a=dyn_cast<tt::LoadOp>(op)){value=a.getResult();address=a.getPtr();mask=a.getMask();load=access=true;}
    else if(auto a=dyn_cast<tt::StoreOp>(op)){value=a.getValue();address=a.getPtr();mask=a.getMask();access=true;}
    else if(auto a=dyn_cast<hcu::BufferLoadOp>(op)){value=a.getResult();address=a.getOffsets();mask=a.getMask();base=a.getPtr();load=access=true;}
    else if(auto a=dyn_cast<hcu::BufferStoreOp>(op)){value=a.getValue();address=a.getOffsets();mask=a.getMask();base=a.getPtr();access=true;}
    if(!access)return;
    if (auto ptr = dyn_cast<tt::PointerType>(getElementTypeOrSelf(address.getType()));
        ptr && ptr.getAddressSpace() == 3) {
      addSharedValue(value, load ? "lds_load_word_sites" : "lds_store_word_sites");
      return;
    }
    totals[load?"load_sites":"store_sites"]+=weight;
    if(mask)totals[load?"masked_load_sites":"masked_store_sites"]+=weight;
    auto row = accessGeometry(value, address, mask, base, axes);
    row["operation"] = int64_t(ids.lookup(op)); row["operation_name"] = name.str();
    row["load"] = load; row["execution_weight"] = weight;
    if (!row.getBoolean("available").value_or(false)) {
      totals["unknown_memory_sites"] += 1;
      issue("unknown memory geometry: " + name.str() + ": " + row.getString("reason").value_or("unknown").str());
    } else if (row.getString("kind") == "scalar") {
      totals[load ? "scalar_load_sites" : "scalar_store_sites"] += weight;
      totals[load ? "scalar_load_bytes" : "scalar_store_bytes"] += weight * row.getNumber("logical_bytes").value_or(0);
    } else {
      totals[load ? "load_work" : "store_work"] += weight * row.getNumber("work").value_or(0);
      totals[load ? "load_instructions" : "store_instructions"] += weight * row.getNumber("instructions").value_or(0);
      totals[load ? "load_vector_elements_sum" : "store_vector_elements_sum"] += weight * row.getNumber("vector_elements").value_or(0);
      totals[load ? "load_element_bytes_sum" : "store_element_bytes_sum"] += weight * row.getNumber("element_bytes").value_or(0);
    }
    accesses.push_back(std::move(row));
  });
  for(const char *key:{"ttg.num-warps","ttg.threads-per-warp","ttg.num-ctas"}) {
    auto attr=module->getAttrOfType<IntegerAttr>(key);
    const char *dest=StringRef(key)=="ttg.num-warps"?"num_warps":
        StringRef(key)=="ttg.threads-per-warp"?"threads_per_warp":"num_ctas";
    if(!attr || attr.getInt()<1)issue(std::string("missing launch attribute: ")+key);
    else totals[dest]=attr.getInt();
  }
  totals["num_stages"]=numStages;
  // Share physical costs before the model's log transform. In particular,
  // log(local)+log(lane) must not reward eliminating only the local term.
  totals["collective_combine_word_work"] = totals["reduce_combine_word_work"] +
      totals["scan_local_combine_work"] + totals["scan_lane_combine_work"] + totals["scan_carry_combine_work"];
  totals["collective_shuffle_word_work"] = totals["shuffle_word_ops"] +
      totals["reduce_shuffle_word_work"] + totals["scan_shuffle_word_work"] + totals["scan_reverse_shuffle_word_work"];
  totals["collective_lds_word_work"] = totals["reduce_lds_load_words"] + totals["reduce_lds_store_words"] +
      totals["scan_lds_load_words"] + totals["scan_lds_store_words"] +
      totals["lds_load_word_sites"] + totals["lds_store_word_sites"] +
      totals["atomic_result_lds_load_words"] + totals["atomic_result_lds_store_words"];
  totals["collective_block_sync_sites"] = totals["reduce_block_sync_sites"] +
      totals["scan_block_sync_sites"] + totals["explicit_block_sync_sites"] +
      totals["lds_block_sync_sites"] + totals["atomic_result_block_sync_sites"];
  llvm::json::Object values;
  for(const auto &[key,value]:totals){
    if(!std::isfinite(value) || value<0)issue("non-finite physical fact: "+key);
    values[key]=number(value);
  }
  auto target=module->getAttrOfType<StringAttr>("ttg.target");
  if(!target)issue("missing target");
  bool complete=issues.empty();
  if (!diagnostics) {
    llvm::json::Array names;
    for (const auto &entry : totals) names.push_back(entry.first);
    return llvm::json::Object{{"feature_version",layoutFactsVersion},
        {"pipeline_stage",layoutTuningStage}, {"target",target?target.getValue().str():""},
        {"complete",complete}, {"issues",std::move(issues)},
        {"feature_names",std::move(names)}, {"values",std::move(values)},
        {"seed_reference_input",false}, {"prices_used",false}, {"timing_inputs_used",false}};
  }
  return llvm::json::Object{{"feature_version",layoutFactsVersion},{"pipeline_stage",layoutTuningStage},
      {"target",target?target.getValue().str():""},{"complete",complete},{"issues",std::move(issues)},
      {"values",std::move(values)},{"operation_census",std::move(census)},
      {"memory_access_facts",std::move(accesses)}, {"control_flow_facts",std::move(controlFlow)},
      {"type_operation_facts",std::move(typeOperations)},
      {"scan_facts",std::move(scans)}, {"extern_facts",std::move(externs)}, {"atomic_facts",std::move(atomics)},
      {"conversion_facts",std::move(conversions)},{"reduction_facts",std::move(reductions)},
      {"resources",std::move(resources)},{"index_dag",std::move(index)},
      {"frequency_policy","static multiplicities only; unresolved dynamic loops/branches remain explicitly flagged; no invented trip count"},
      {"launch_context_policy","IR launch attributes; runtime grid and bound scalar extents not yet supplied in v97 coverage audit"},
      {"seed_reference_input",false},{"prices_used",false},{"timing_inputs_used",false}};
}
} // namespace layoutdp
