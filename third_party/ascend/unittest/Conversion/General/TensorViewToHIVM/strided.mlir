// RUN: triton-opt %s --tensor-view-to-hivm | FileCheck %s

// A strided (overlapping-tile) view_load lowers like partition but the GM tile
// origin steps by the traversal stride (512), not the tile size (1024).

// CHECK-LABEL: tt.func public @sliding
// CHECK-SAME: memref<?xf32, #hivm.address_space<gm>>
tt.func public @sliding(%in: !tv.ptr<f32>, %n: index, %i: index) {
  %c1 = arith.constant 1 : index
  %v = tv.make_tensor_view %in, sizes = [%n], strides = [%c1]
       : !tv.ptr<f32> -> !tv.tensor_view<?xf32, strides=[1]>
  %sv = tv.make_strided_view %v
        : !tv.tensor_view<?xf32, strides=[1]>
       -> !tv.tensor_view<?xf32, strides=[1], #tv.strided_view<tile = [1024], dim_map = [0], traversal_strides = [512], padding = zero>>
  // origin = i * 512 (traversal), tile size 1024
  // CHECK: memref.alloc() : memref<1024xf32>
  // CHECK: arith.constant 512 : index
  // CHECK: memref.reinterpret_cast
  // CHECK: hivm.hir.load
  %t = tv.view_load %sv[%i]
       : !tv.tensor_view<?xf32, strides=[1], #tv.strided_view<tile = [1024], dim_map = [0], traversal_strides = [512], padding = zero>>
      -> tensor<1024xf32>
  tt.return
}

// CHECK-NOT: tv.view_load
// CHECK-NOT: tv.make_strided_view
