//===- TensorViewAttrs.cpp - TensorView attributes ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ascend/include/Dialect/TensorView/IR/TensorViewAttrs.h"

#include "ascend/include/Dialect/TensorView/IR/TensorViewDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h" // required by `Attrs.cpp.inc`
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h" // required by `Attrs.cpp.inc`

using namespace mlir;
using namespace mlir::triton::tv;

#include "ascend/include/Dialect/TensorView/IR/TensorViewEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "ascend/include/Dialect/TensorView/IR/TensorViewAttrs.cpp.inc"

//===----------------------------------------------------------------------===//
// Custom assembly helpers for the encoding attributes.
//
//   #tv.partition_view<tile = [128], dim_map = [0], padding = zero>
//===----------------------------------------------------------------------===//

// Parse `<keyword> = [i64, i64, ...]`.
static ParseResult parseKeywordIntArray(AsmParser &parser, StringRef keyword,
                                        SmallVectorImpl<int64_t> &out) {
  if (parser.parseKeyword(keyword) || parser.parseEqual())
    return failure();
  return parser.parseCommaSeparatedList(
      AsmParser::Delimiter::Square, [&]() -> ParseResult {
        int64_t v;
        if (parser.parseInteger(v))
          return failure();
        out.push_back(v);
        return success();
      });
}

// Parse `padding = <symbol>`.
static ParseResult parsePaddingKeyword(AsmParser &parser, PadKind &padding) {
  if (parser.parseKeyword("padding") || parser.parseEqual())
    return failure();
  StringRef sym;
  llvm::SMLoc loc = parser.getCurrentLocation();
  if (parser.parseKeyword(&sym))
    return failure();
  std::optional<PadKind> kind = symbolizePadKind(sym);
  if (!kind)
    return parser.emitError(loc, "invalid padding kind: ") << sym;
  padding = *kind;
  return success();
}

static void printIntArray(AsmPrinter &printer, StringRef keyword,
                          ArrayRef<int64_t> values) {
  printer << keyword << " = [";
  llvm::interleaveComma(values, printer, [&](int64_t v) { printer << v; });
  printer << "]";
}

//===----------------------------------------------------------------------===//
// PartitionViewAttr assembly
//===----------------------------------------------------------------------===//
Attribute PartitionViewAttr::parse(AsmParser &parser, Type) {
  llvm::SMLoc loc = parser.getCurrentLocation();
  SmallVector<int64_t> tile, dimMap;
  PadKind padding;
  if (parser.parseLess() || parseKeywordIntArray(parser, "tile", tile) ||
      parser.parseComma() || parseKeywordIntArray(parser, "dim_map", dimMap) ||
      parser.parseComma() || parsePaddingKeyword(parser, padding) ||
      parser.parseGreater())
    return {};
  return getChecked([&] { return parser.emitError(loc); }, parser.getContext(),
                    tile, dimMap, padding);
}

void PartitionViewAttr::print(AsmPrinter &printer) const {
  printer << "<";
  printIntArray(printer, "tile", getTile());
  printer << ", ";
  printIntArray(printer, "dim_map", getDimMap());
  printer << ", padding = " << stringifyPadKind(getPadding()) << ">";
}

//===----------------------------------------------------------------------===//
// StridedViewAttr assembly
//===----------------------------------------------------------------------===//
Attribute StridedViewAttr::parse(AsmParser &parser, Type) {
  llvm::SMLoc loc = parser.getCurrentLocation();
  SmallVector<int64_t> tile, dimMap, traversalStrides;
  PadKind padding;
  if (parser.parseLess() || parseKeywordIntArray(parser, "tile", tile) ||
      parser.parseComma() || parseKeywordIntArray(parser, "dim_map", dimMap) ||
      parser.parseComma() ||
      parseKeywordIntArray(parser, "traversal_strides", traversalStrides) ||
      parser.parseComma() || parsePaddingKeyword(parser, padding) ||
      parser.parseGreater())
    return {};
  return getChecked([&] { return parser.emitError(loc); }, parser.getContext(),
                    tile, dimMap, traversalStrides, padding);
}

void StridedViewAttr::print(AsmPrinter &printer) const {
  printer << "<";
  printIntArray(printer, "tile", getTile());
  printer << ", ";
  printIntArray(printer, "dim_map", getDimMap());
  printer << ", ";
  printIntArray(printer, "traversal_strides", getTraversalStrides());
  printer << ", padding = " << stringifyPadKind(getPadding()) << ">";
}

//===----------------------------------------------------------------------===//
// GatherScatterViewAttr assembly
//===----------------------------------------------------------------------===//
Attribute GatherScatterViewAttr::parse(AsmParser &parser, Type) {
  llvm::SMLoc loc = parser.getCurrentLocation();
  SmallVector<int64_t> tile, sparseDim;
  PadKind padding;
  if (parser.parseLess() || parseKeywordIntArray(parser, "tile", tile) ||
      parser.parseComma() ||
      parseKeywordIntArray(parser, "sparse_dim", sparseDim) ||
      parser.parseComma() || parsePaddingKeyword(parser, padding) ||
      parser.parseGreater())
    return {};
  return getChecked([&] { return parser.emitError(loc); }, parser.getContext(),
                    tile, sparseDim, padding);
}

void GatherScatterViewAttr::print(AsmPrinter &printer) const {
  printer << "<";
  printIntArray(printer, "tile", getTile());
  printer << ", ";
  printIntArray(printer, "sparse_dim", getSparseDim());
  printer << ", padding = " << stringifyPadKind(getPadding()) << ">";
}

//===----------------------------------------------------------------------===//
// Encoding verifiers (structural checks that do not depend on the base rank).
//===----------------------------------------------------------------------===//

LogicalResult
PartitionViewAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                          ArrayRef<int64_t> tile, ArrayRef<int64_t> dimMap,
                          PadKind padding) {
  if (tile.empty())
    return emitError() << "partition_view tile must be non-empty";
  if (tile.size() != dimMap.size())
    return emitError() << "partition_view tile and dim_map must have equal length";
  return success();
}

LogicalResult
StridedViewAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                        ArrayRef<int64_t> tile, ArrayRef<int64_t> dimMap,
                        ArrayRef<int64_t> traversalStrides, PadKind padding) {
  if (tile.empty())
    return emitError() << "strided_view tile must be non-empty";
  if (tile.size() != dimMap.size())
    return emitError() << "strided_view tile and dim_map must have equal length";
  if (tile.size() != traversalStrides.size())
    return emitError()
           << "strided_view tile and traversal_strides must have equal length";
  return success();
}

LogicalResult
GatherScatterViewAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                              ArrayRef<int64_t> tile, ArrayRef<int64_t> sparseDim,
                              PadKind padding) {
  if (tile.empty())
    return emitError() << "gather_scatter_view tile must be non-empty";
  if (sparseDim.empty())
    return emitError() << "gather_scatter_view sparse_dim must be non-empty";
  return success();
}

//===----------------------------------------------------------------------===//
// View-encoding helpers.
//===----------------------------------------------------------------------===//

bool mlir::triton::tv::isViewEncoding(Attribute enc) {
  return llvm::isa_and_nonnull<PartitionViewAttr, StridedViewAttr,
                               GatherScatterViewAttr>(enc);
}

llvm::SmallVector<int64_t>
mlir::triton::tv::getEncodingTileShape(Attribute enc) {
  return llvm::TypeSwitch<Attribute, llvm::SmallVector<int64_t>>(enc)
      .Case<PartitionViewAttr>(
          [](auto a) { return llvm::to_vector(a.getTile()); })
      .Case<StridedViewAttr>(
          [](auto a) { return llvm::to_vector(a.getTile()); })
      .Case<GatherScatterViewAttr>(
          [](auto a) { return llvm::to_vector(a.getTile()); })
      .Default([](Attribute) { return llvm::SmallVector<int64_t>{}; });
}

int64_t mlir::triton::tv::getEncodingIndexSpaceRank(Attribute enc) {
  return static_cast<int64_t>(getEncodingTileShape(enc).size());
}

PadKind mlir::triton::tv::getEncodingPadding(Attribute enc) {
  return llvm::TypeSwitch<Attribute, PadKind>(enc)
      .Case<PartitionViewAttr>([](auto a) { return a.getPadding(); })
      .Case<StridedViewAttr>([](auto a) { return a.getPadding(); })
      .Case<GatherScatterViewAttr>([](auto a) { return a.getPadding(); })
      .Default([](Attribute) { return PadKind::Zero; });
}
