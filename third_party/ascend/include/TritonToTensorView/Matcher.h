//===- Matcher.h - tt access pattern -> tv, reusable entry points --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The structured/gather/discrete access matcher that TritonToTensorView.cpp
// (Pass A) uses to convert `tt.load` / `tt.store` into `tv` ops.  Exposed here
// so it can also be called *inline*, at the moment a `tl.load`/`tl.store` is
// being lowered (see triton_ascend.cc's `try_tv_load` / `try_tv_store` and
// python/triton/language/semantic.py) -- i.e. `tv` is emitted directly by the
// tl->tt lowering instead of being recovered by a later pass over completed
// `tt.load`/`tt.store` ops.  Both call sites share this one matcher
// implementation (in TritonToTensorView.cpp); Pass A remains a safety net for
// whatever the inline attempt cannot match (e.g. patterns only visible after
// canonicalization).
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_TRITON_TO_TENSOR_VIEW_MATCHER_H
#define ASCEND_TRITON_TO_TENSOR_VIEW_MATCHER_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"

namespace mlir {
namespace triton {
namespace tv {

// Retype `v` from `!tt.ptr<T>` to `!tv.ptr<T>` in place (no-op if `v` is
// already `!tv.ptr` or isn't a pointer at all).  When `v` is a `tt.func`
// block argument, also syncs the function's signature.  Lazy / per-value, so
// only pointers an access actually matches get retyped -- pointers a plain
// tt.load/tt.store still reads (e.g. an index array) are left as `!tt.ptr`.
Value ensureTvPtr(Value v);

// Try to recognize `ptr` (+ optional `mask`) as a structured tv access
// (partition / strided / gather) or a discrete ptr_load, and emit the
// corresponding `tv` ops.  `resultTy` is the load's result type.  Returns a
// null Value if the pattern isn't recognized -- the caller should fall back
// to building a plain `tt.load`.
Value tryEmitTvLoad(OpBuilder &b, Location loc, Value ptr, Value mask,
                    Type resultTy);

// Symmetric for stores.  Returns false if the pattern isn't recognized (the
// caller should fall back to building a plain `tt.store`).
bool tryEmitTvStore(OpBuilder &b, Location loc, Value ptr, Value value,
                    Value mask);

} // namespace tv
} // namespace triton
} // namespace mlir

#endif // ASCEND_TRITON_TO_TENSOR_VIEW_MATCHER_H
