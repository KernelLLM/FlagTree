//===-- TensorViewOps.td - TensorView operations --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TENSORVIEW_OPS
#define TENSORVIEW_OPS

include "ascend/include/Dialect/TensorView/IR/TensorViewDialect.td"
include "ascend/include/Dialect/TensorView/IR/TensorViewTypes.td"
include "ascend/include/Dialect/TensorView/IR/TensorViewAttrs.td"
include "ascend/include/Dialect/TensorView/IR/TensorViewInterfaces.td"

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class TensorView_Op<string mnemonic, list<Trait> traits = []>
    : Op<TensorView_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// make_tensor_view : materialize a base tensor_view from a pointer.
//===----------------------------------------------------------------------===//
def TV_MakeTensorViewOp : TensorView_Op<"make_tensor_view",
    [Pure, AttrSizedOperandSegments]> {
  let summary = "Materialize a base tensor_view (no encoding) from a tv.ptr.";
  let arguments = (ins
    TV_PtrType:$source,
    Variadic<Index>:$sizes,
    Variadic<Index>:$strides);
  let results = (outs TV_TensorViewType:$result);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $source `,` `sizes` `=` `[` $sizes `]` `,` `strides` `=` `[` $strides `]`
    attr-dict `:` type($source) `->` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// make_partition_view / make_strided_view / make_gather_scatter_view :
// attach a view encoding to a base tensor_view.
//===----------------------------------------------------------------------===//
class TV_MakeViewOp<string mnemonic>
    : TensorView_Op<mnemonic, [Pure, DeclareOpInterfaceMethods<ViewOpInterface>]> {
  let arguments = (ins TV_TensorViewType:$source);
  let results = (outs TV_TensorViewType:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source attr-dict `:` type($source) `->` type($result)";
}

def TV_MakePartitionViewOp     : TV_MakeViewOp<"make_partition_view"> {
  let summary = "Attach a #tv.partition_view encoding to a base view.";
}
def TV_MakeStridedViewOp       : TV_MakeViewOp<"make_strided_view"> {
  let summary = "Attach a #tv.strided_view encoding to a base view.";
}
def TV_MakeGatherScatterViewOp : TV_MakeViewOp<"make_gather_scatter_view"> {
  let summary = "Attach a #tv.gather_scatter_view encoding to a base view.";
}

//===----------------------------------------------------------------------===//
// view_load / view_store : load/store a tile of an encoded view as a tensor.
//===----------------------------------------------------------------------===//
def TV_ViewLoadOp : TensorView_Op<"view_load",
    [DeclareOpInterfaceMethods<ViewOpInterface>, AttrSizedOperandSegments,
     MemoryEffects<[MemRead]>]> {
  let summary = "Load the tile at the given index-space position into a tensor.";
  let arguments = (ins
    TV_TensorViewType:$view,
    Variadic<TV_ViewIndex>:$indices,
    Optional<TV_BoolTensor>:$mask);
  let results = (outs AnyRankedTensor:$result);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $view `[` $indices `]` (`,` `mask` $mask^ `:` type($mask))?
    attr-dict `:` type($view) `,` type($indices) `->` type($result)
  }];
}

def TV_ViewStoreOp : TensorView_Op<"view_store",
    [DeclareOpInterfaceMethods<ViewOpInterface>, AttrSizedOperandSegments,
     MemoryEffects<[MemWrite]>]> {
  let summary = "Store a tensor into the tile at the given index-space position.";
  let arguments = (ins
    TV_TensorViewType:$view,
    AnyRankedTensor:$value,
    Variadic<TV_ViewIndex>:$indices,
    Optional<TV_BoolTensor>:$mask);
  let results = (outs);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $view `[` $indices `]` `,` $value (`,` `mask` $mask^ `:` type($mask))?
    attr-dict `:` type($view) `,` type($value) `,` type($indices)
  }];
}

//===----------------------------------------------------------------------===//
// ptr_load / ptr_store : discrete gather/scatter over a tensor of pointers.
//===----------------------------------------------------------------------===//
def TV_PtrLoadOp : TensorView_Op<"ptr_load",
    [AttrSizedOperandSegments, MemoryEffects<[MemRead]>]> {
  let summary = "Element-wise (diagonal) gather over a tensor of pointers.";
  let arguments = (ins
    TV_PtrTensor:$ptrs,
    Variadic<TV_IndexTensor>:$indices,
    Optional<TV_BoolTensor>:$mask,
    TV_PadKindAttr:$padding);
  let results = (outs AnyRankedTensor:$result);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $ptrs `,` `indices` `=` `[` $indices `]` (`,` `mask` $mask^ `:` type($mask))?
    attr-dict `:` type($ptrs) `,` type($indices) `->` type($result)
  }];
}

def TV_PtrStoreOp : TensorView_Op<"ptr_store",
    [AttrSizedOperandSegments, MemoryEffects<[MemWrite]>]> {
  let summary = "Element-wise (diagonal) scatter over a tensor of pointers.";
  let arguments = (ins
    TV_PtrTensor:$ptrs,
    AnyRankedTensor:$value,
    Variadic<TV_IndexTensor>:$indices,
    Optional<TV_BoolTensor>:$mask,
    TV_PadKindAttr:$padding);
  let results = (outs);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $ptrs `,` $value `,` `indices` `=` `[` $indices `]`
    (`,` `mask` $mask^ `:` type($mask))?
    attr-dict `:` type($ptrs) `,` type($value) `,` type($indices)
  }];
}

#endif // TENSORVIEW_OPS
