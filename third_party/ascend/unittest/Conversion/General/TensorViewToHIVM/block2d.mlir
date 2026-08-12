// RUN: triton-opt %s --tensor-view-to-hivm | FileCheck %s

// A rank-2 partition view_load (dynamic row stride) lowers to a 2-D
// reinterpret_cast (strided) + 2-D UB alloc + hivm.hir.load (full tile).

// CHECK-LABEL: tt.func public @block2d
// CHECK-SAME: memref<?xf32, #hivm.address_space<gm>>
tt.func public @block2d(%in: !tv.ptr<f32>, %N: index, %im: index, %jn: index) {
  %c1 = arith.constant 1 : index
  %v = tv.make_tensor_view %in, sizes = [%N, %N], strides = [%N, %c1]
       : !tv.ptr<f32> -> !tv.tensor_view<?x?xf32, strides=[?, 1]>
  %pv = tv.make_partition_view %v
        : !tv.tensor_view<?x?xf32, strides=[?, 1]>
       -> !tv.tensor_view<?x?xf32, strides=[?, 1], #tv.partition_view<tile = [64, 64], dim_map = [0, 1], padding = zero>>
  // CHECK: memref.reinterpret_cast
  // CHECK: memref.alloc() : memref<64x64xf32, #hivm.address_space<ub>>
  // CHECK: hivm.hir.load
  // CHECK: bufferization.to_tensor
  %t = tv.view_load %pv[%im, %jn]
       : !tv.tensor_view<?x?xf32, strides=[?, 1], #tv.partition_view<tile = [64, 64], dim_map = [0, 1], padding = zero>>
      -> tensor<64x64xf32>
  tt.return
}

// CHECK-NOT: tv.view_load
// CHECK-NOT: tv.make_tensor_view
