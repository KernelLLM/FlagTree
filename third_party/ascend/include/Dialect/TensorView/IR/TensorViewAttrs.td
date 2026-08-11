//===-- TensorViewAttrs.td - TensorView attributes ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TENSORVIEW_ATTRS
#define TENSORVIEW_ATTRS

include "ascend/include/Dialect/TensorView/IR/TensorViewDialect.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// Padding kind
//===----------------------------------------------------------------------===//

def TV_PadZero  : I32EnumAttrCase<"Zero",  0, "zero">;
def TV_PadNan   : I32EnumAttrCase<"Nan",   1, "nan">;
def TV_PadInf   : I32EnumAttrCase<"Inf",   2, "inf">;
def TV_PadNInf  : I32EnumAttrCase<"NInf",  3, "ninf">;
def TV_PadUndef : I32EnumAttrCase<"Undef", 4, "undef">;

def TV_PadKindEnum : I32EnumAttr<"PadKind", "TensorView out-of-bounds padding kind",
    [TV_PadZero, TV_PadNan, TV_PadInf, TV_PadNInf, TV_PadUndef]> {
  let cppNamespace = "::mlir::triton::tv";
  let genSpecializedAttr = 0;
}

// Standalone `#tv.pad<...>` attribute, used as an op attribute (e.g. ptr_load).
def TV_PadKindAttr : EnumAttr<TensorView_Dialect, TV_PadKindEnum, "pad"> {
  let assemblyFormat = "`<` $value `>`";
}

//===----------------------------------------------------------------------===//
// Encoding attributes
//===----------------------------------------------------------------------===//

class TensorView_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<TensorView_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
  let cppNamespace = "::mlir::triton::tv";
}

// #tv.partition_view<tile=[...], dim_map=[...], padding=zero>
def TV_PartitionViewAttr : TensorView_Attr<"PartitionView", "partition_view"> {
  let summary = "Partition (tiling) view encoding.";
  let parameters = (ins
    ArrayRefParameter<"int64_t">:$tile,
    ArrayRefParameter<"int64_t">:$dimMap,
    EnumParameter<TV_PadKindEnum>:$padding);
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
  let builders = [
    AttrBuilder<(ins "::llvm::ArrayRef<int64_t>":$tile), [{
      ::llvm::SmallVector<int64_t> dimMap;
      for (int64_t i = 0, e = tile.size(); i < e; ++i) dimMap.push_back(i);
      return $_get($_ctxt, tile, dimMap, PadKind::Zero);
    }]>,
    AttrBuilder<(ins "::llvm::ArrayRef<int64_t>":$tile,
                     "::llvm::ArrayRef<int64_t>":$dimMap), [{
      return $_get($_ctxt, tile, dimMap, PadKind::Zero);
    }]>,
  ];
}

// #tv.strided_view<tile=[...], dim_map=[...], traversal_strides=[...], padding=zero>
def TV_StridedViewAttr : TensorView_Attr<"StridedView", "strided_view"> {
  let summary = "Strided (partition + inter-tile stride) view encoding.";
  let parameters = (ins
    ArrayRefParameter<"int64_t">:$tile,
    ArrayRefParameter<"int64_t">:$dimMap,
    ArrayRefParameter<"int64_t">:$traversalStrides,
    EnumParameter<TV_PadKindEnum>:$padding);
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
  let builders = [
    AttrBuilder<(ins "::llvm::ArrayRef<int64_t>":$tile,
                     "::llvm::ArrayRef<int64_t>":$traversalStrides), [{
      ::llvm::SmallVector<int64_t> dimMap;
      for (int64_t i = 0, e = tile.size(); i < e; ++i) dimMap.push_back(i);
      return $_get($_ctxt, tile, dimMap, traversalStrides, PadKind::Zero);
    }]>,
  ];
}

// #tv.gather_scatter_view<tile=[...], sparse_dim=[...], padding=zero>
def TV_GatherScatterViewAttr : TensorView_Attr<"GatherScatterView", "gather_scatter_view"> {
  let summary = "Gather/scatter view encoding.";
  let parameters = (ins
    ArrayRefParameter<"int64_t">:$tile,
    ArrayRefParameter<"int64_t">:$sparseDim,
    EnumParameter<TV_PadKindEnum>:$padding);
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
  let builders = [
    AttrBuilder<(ins "::llvm::ArrayRef<int64_t>":$tile,
                     "::llvm::ArrayRef<int64_t>":$sparseDim), [{
      return $_get($_ctxt, tile, sparseDim, PadKind::Zero);
    }]>,
  ];
}

#endif // TENSORVIEW_ATTRS
