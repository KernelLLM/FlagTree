#include "TritonHCUGPUTransforms/LayoutExecutionContext.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include <cmath>

using namespace mlir;
namespace tt = mlir::triton;
namespace layoutdp {
namespace {
constexpr double maxExact = 9007199254740991.;
unsigned width(Type t) {
  if (auto i = dyn_cast<IntegerType>(t)) return i.getWidth();
  return t.isIndex() ? 64 : 0;
}
}

LayoutExecutionContext::LayoutExecutionContext(ModuleOp module, llvm::StringRef payload) {
  if (payload.empty()) return;
  present = true;
  auto parsed = llvm::json::parse(payload);
  if (!parsed) { llvm::consumeError(parsed.takeError()); error = "invalid execution context JSON"; return; }
  auto *doc = parsed->getAsObject();
  if (!doc || doc->size() != 5 ||
      doc->getString("schema") != "hcu-layout-execution-context-v1" ||
      doc->getString("origin") != "jit.bound_args+grid") {
    error = "invalid generic execution context schema/provenance"; return;
  }
  auto *arguments = doc->getArray("arguments");
  auto arity = doc->getInteger("launch_arity");
  if (!arguments || !arity || *arity < 0 || *arity > 1024 || arguments->size() > 1024) {
    error = "invalid execution context argument count"; return;
  }
  tt::FuncOp entry;
  for (auto f : module.getOps<tt::FuncOp>()) if (!f.isPrivate() && !f.isExternal()) {
    if (entry) { error = "execution context requires one entry function"; return; }
    entry = f;
  }
  if (!entry || entry.getNumArguments() != arguments->size()) {
    error = "execution context ABI arity mismatch"; return;
  }
  if (auto *dims = doc->getArray("grid")) {
    double product = 1;
    if (dims->size() != 3) { error = "invalid execution grid"; return; }
    for (unsigned i = 0; i < 3; ++i) {
      auto n = (*dims)[i].getAsInteger();
      if (!n || *n < 0 || *n > INT32_MAX) { error = "invalid execution grid extent"; return; }
      grid[i] = *n; product *= *n;
    }
    if (product > maxExact) { error = "execution grid exceeds exact range"; return; }
    gridKnown = true;
  } else if (!doc->get("grid") || !doc->get("grid")->getAsNull()) {
    error = "execution grid must be known or explicit null"; return;
  }
  for (unsigned i = 0; i < arguments->size(); ++i) {
    auto *row = (*arguments)[i].getAsObject();
    if (!row || row->getInteger("index") != i || !row->getArray("path")) {
      error = "invalid execution context ABI index/path"; return;
    }
    auto *path = row->getArray("path");
    if (path->empty() || path->size() > 32) { error = "invalid execution context argument path"; return; }
    for (unsigned j = 0; j < path->size(); ++j) {
      auto x = (*path)[j].getAsInteger();
      if (!x || *x < 0 || *x >= 1024 || (!j && *x >= *arity)) {
        error = "invalid execution context argument path index"; return;
      }
    }
    auto arg = entry.getArgument(i);
    auto kind = row->getString("kind");
    if (kind == "integer") {
      auto bits = row->getInteger("bits"), value = row->getInteger("value");
      auto unsign = row->getBoolean("unsigned");
      if (!bits || !value || !unsign || *bits != width(arg.getType()) || *bits < 1 || *bits > 64 ||
          std::abs(double(*value)) > maxExact) {
        error = "execution context integer ABI/type mismatch"; return;
      }
      if ((*unsign && *value < 0) ||
          (*bits < 64 && (*unsign ? uint64_t(*value) >= (1ULL << *bits) :
              *value < -(1LL << (*bits - 1)) || *value >= (1LL << (*bits - 1))))) {
        error = "execution context integer does not fit ABI"; return;
      }
      // i1 true and unsigned values with the high bit set do not fit the
      // signed range of their ABI width. Preserve the declared signedness.
      scalars.insert({arg, llvm::APInt(*bits, uint64_t(*value), !*unsign)});
    } else if (kind == "pointer") {
      auto pointer = dyn_cast<tt::PointerType>(arg.getType());
      if (!pointer) { error = "execution context pointer ABI mismatch"; return; }
      auto *t = row->getObject("tensor");
      if (!t) { ++unknownTensors; continue; }
      auto *shape = t->getArray("shape"), *strides = t->getArray("strides");
      auto bytes = t->getInteger("element_bytes");
      auto element = pointer.getPointeeType();
      if (!shape || !strides || shape->size() != strides->size() || shape->size() > 32 ||
          !bytes || !element.isIntOrFloat() || *bytes != std::max(1u, (element.getIntOrFloatBitWidth() + 7) / 8)) {
        error = "execution context tensor metadata/type mismatch"; return;
      }
      double elements = 1, span = 1;
      for (unsigned j = 0; j < shape->size(); ++j) {
        auto n = (*shape)[j].getAsInteger(), stride = (*strides)[j].getAsInteger();
        if (!n || !stride || *n < 0 || *stride < 0 || double(*n) > maxExact || double(*stride) > maxExact) {
          error = "invalid execution context tensor extent/stride"; return;
        }
        elements *= *n; span += std::max<int64_t>(0, *n - 1) * double(*stride);
        if (*stride == 1) ++contiguousAxes;
      }
      if (!elements) span = 0;
      if (std::max(elements, span) * *bytes > maxExact) { error = "execution tensor extent exceeds exact range"; return; }
      ++tensorCount; tensorElements += elements; tensorBytes += elements * *bytes;
      tensorSpanBytes += span * *bytes;
    } else if (kind != "other" || isa<IntegerType, tt::PointerType>(arg.getType())) {
      error = "execution context argument kind mismatch"; return;
    }
  }
}

std::optional<llvm::APInt> LayoutExecutionContext::integer(Value value, unsigned depth) const {
  unsigned bits = width(value.getType());
  if (!bits || bits > 64 || depth > 64) return {};
  if (auto it = scalars.find(value); it != scalars.end()) return it->second;
  if (auto c = value.getDefiningOp<arith::ConstantOp>())
    if (auto a = dyn_cast<IntegerAttr>(c.getValue())) return a.getValue();
  if (auto it = memo.find(value); it != memo.end()) return it->second;
  memo[value] = {};
  auto *op = value.getDefiningOp();
  if (!op) return {};
  auto done = [&](llvm::APInt x) -> std::optional<llvm::APInt> { memo[value] = x; return x; };
  if (auto p = dyn_cast<tt::GetNumProgramsOp>(op))
    if (gridKnown) return done(llvm::APInt(bits, grid[unsigned(p.getAxis())]));
  auto name = op->getName().getStringRef();
  if (op->getNumOperands() == 1) {
    auto a = integer(op->getOperand(0), depth + 1); if (!a) return {};
    if (name == "arith.extsi" || name == "arith.index_cast") return done(a->sextOrTrunc(bits));
    if (name == "arith.extui" || name == "arith.index_castui" || name == "arith.trunci") return done(a->zextOrTrunc(bits));
  }
  if (op->getNumOperands() != 2) return {};
  auto a = integer(op->getOperand(0), depth + 1), b = integer(op->getOperand(1), depth + 1);
  if (!a || !b || a->getBitWidth() != b->getBitWidth()) return {};
  if (name == "arith.addi") return done(*a + *b);
  if (name == "arith.subi") return done(*a - *b);
  if (name == "arith.muli") return done(*a * *b);
  if (name == "arith.andi") return done(*a & *b);
  if (name == "arith.ori") return done(*a | *b);
  if (name == "arith.xori") return done(*a ^ *b);
  if (name == "arith.divui" && !b->isZero()) return done(a->udiv(*b));
  if (name == "arith.remui" && !b->isZero()) return done(a->urem(*b));
  if ((name == "arith.divsi" || name == "arith.remsi") && !b->isZero() && !(a->isMinSignedValue() && b->isAllOnes()))
    return done(name == "arith.divsi" ? a->sdiv(*b) : a->srem(*b));
  if (name == "arith.minsi") return done(a->slt(*b) ? *a : *b);
  if (name == "arith.maxsi") return done(a->sgt(*b) ? *a : *b);
  if (name == "arith.minui") return done(a->ult(*b) ? *a : *b);
  if (name == "arith.maxui") return done(a->ugt(*b) ? *a : *b);
  if (b->ult(a->getBitWidth())) {
    if (name == "arith.shli") return done(a->shl(b->getZExtValue()));
    if (name == "arith.shrsi") return done(a->ashr(b->getZExtValue()));
    if (name == "arith.shrui") return done(a->lshr(b->getZExtValue()));
  }
  return {};
}

std::optional<double> LayoutExecutionContext::trips(scf::ForOp loop) const {
  auto a = integer(loop.getLowerBound()), b = integer(loop.getUpperBound()), s = integer(loop.getStep());
  if (!a || !b || !s || s->getSExtValue() <= 0 || loop->hasAttr("unsignedCmp")) return {};
  __int128 distance = __int128(b->getSExtValue()) - a->getSExtValue();
  if (distance <= 0) return 0.;
  __int128 step = s->getSExtValue(), n = (distance + step - 1) / step;
  if (n > int64_t(maxExact)) return {};
  return double(n);
}

double layoutExecutionWeight(Operation *op, const LayoutExecutionContext *context) {
  double weight = 1;
  for (auto *p = op->getParentOp(); p; p = p->getParentOp()) if (auto loop = dyn_cast<scf::ForOp>(p)) {
    if (context) { if (auto n = context->trips(loop)) weight *= *n; }
    else {
      auto integer = [](Value v) -> std::optional<int64_t> {
        if (auto c = v.getDefiningOp<arith::ConstantOp>())
          if (auto i = dyn_cast<IntegerAttr>(c.getValue())) return i.getInt();
        return {};
      };
      auto a = integer(loop.getLowerBound()), b = integer(loop.getUpperBound()), s = integer(loop.getStep());
      if (a && b && s && *s > 0) weight *= std::max(0., std::ceil((double(*b) - double(*a)) / double(*s)));
    }
  }
  return weight;
}

llvm::json::Object LayoutExecutionContext::scaleFacts(ModuleOp module) const {
  double resolved = 0, unknown = 0, sum = 0, max = 0;
  module.walk([&](scf::ForOp loop) {
    bool dynamic = !loop.getLowerBound().getDefiningOp<arith::ConstantOp>() ||
        !loop.getUpperBound().getDefiningOp<arith::ConstantOp>() || !loop.getStep().getDefiningOp<arith::ConstantOp>();
    if (!dynamic) return;
    if (auto n = trips(loop)) { ++resolved; sum += *n; max = std::max(max, *n); }
    else ++unknown;
  });
  module.walk([&](scf::WhileOp) { ++unknown; });
  return llvm::json::Object{{"execution_context_present", double(present)},
      {"execution_grid_known", double(gridKnown)},
      {"execution_grid_x", double(grid[0])}, {"execution_grid_y", double(grid[1])}, {"execution_grid_z", double(grid[2])},
      {"execution_cta_count", gridKnown ? double(grid[0]) * grid[1] * grid[2] : 0.},
      {"execution_tensor_count", tensorCount}, {"execution_tensor_metadata_unknown", unknownTensors},
      {"execution_tensor_elements", tensorElements}, {"execution_tensor_logical_bytes", tensorBytes},
      {"execution_tensor_span_bytes", tensorSpanBytes}, {"execution_tensor_unit_stride_axes", contiguousAxes},
      {"execution_resolved_loop_sites", resolved}, {"execution_unresolved_loop_sites", unknown},
      {"execution_resolved_trip_sum", sum}, {"execution_resolved_trip_max", max},
      {"execution_mask_population_known", 0.}, {"execution_branch_probabilities_known", 0.}};
}
} // namespace layoutdp
