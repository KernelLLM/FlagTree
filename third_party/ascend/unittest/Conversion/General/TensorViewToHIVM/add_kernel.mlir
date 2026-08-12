// RUN: triton-opt %s --tensor-view-to-hivm | FileCheck %s

// The tv access ops (Pass A output) lower to memref + HIVM DMA:
//   !tv.ptr args -> GM memref; view_load -> reinterpret_cast+alloc(UB)+hir.load+to_tensor;
//   view_store -> to_buffer+reinterpret_cast+hir.store.  arith.addf is untouched.

// CHECK-LABEL: tt.func public @add_kernel
// CHECK-SAME: memref<?xf32, #hivm.address_space<gm>>
tt.func public @add_kernel(%a: !tv.ptr<f32>, %b: !tv.ptr<f32>, %c: !tv.ptr<f32>, %n: i32) {
  %n_idx = arith.index_cast %n : i32 to index
  %c1 = arith.constant 1 : index
  %pid = tt.get_program_id x : i32
  %i = arith.index_cast %pid : i32 to index

  %va = tv.make_tensor_view %a, sizes = [%n_idx], strides = [%c1]
        : !tv.ptr<f32> -> !tv.tensor_view<?xf32, strides=[1]>
  %pa = tv.make_partition_view %va
        : !tv.tensor_view<?xf32, strides=[1]>
       -> !tv.tensor_view<?xf32, strides=[1], #tv.partition_view<tile = [1024], dim_map = [0], padding = zero>>
  // len = min(n - off, tile); pad the tile; DMA the valid prefix only.
  // CHECK: memref.alloc() : memref<1024xf32>
  // CHECK: arith.minsi
  // CHECK: memref.reinterpret_cast
  // CHECK-SAME: #hivm.address_space<gm>
  // CHECK: linalg.fill
  // CHECK: memref.subview
  // CHECK: hivm.hir.load
  // CHECK: bufferization.to_tensor
  %ta = tv.view_load %pa[%i]
        : !tv.tensor_view<?xf32, strides=[1], #tv.partition_view<tile = [1024], dim_map = [0], padding = zero>>
       -> tensor<1024xf32>

  %vb = tv.make_tensor_view %b, sizes = [%n_idx], strides = [%c1]
        : !tv.ptr<f32> -> !tv.tensor_view<?xf32, strides=[1]>
  %pb = tv.make_partition_view %vb
        : !tv.tensor_view<?xf32, strides=[1]>
       -> !tv.tensor_view<?xf32, strides=[1], #tv.partition_view<tile = [1024], dim_map = [0], padding = zero>>
  %tb = tv.view_load %pb[%i]
        : !tv.tensor_view<?xf32, strides=[1], #tv.partition_view<tile = [1024], dim_map = [0], padding = zero>>
       -> tensor<1024xf32>

  // CHECK: arith.addf %{{.*}}, %{{.*}} : tensor<1024xf32>
  %sum = arith.addf %ta, %tb : tensor<1024xf32>

  %vc = tv.make_tensor_view %c, sizes = [%n_idx], strides = [%c1]
        : !tv.ptr<f32> -> !tv.tensor_view<?xf32, strides=[1]>
  %pc = tv.make_partition_view %vc
        : !tv.tensor_view<?xf32, strides=[1]>
       -> !tv.tensor_view<?xf32, strides=[1], #tv.partition_view<tile = [1024], dim_map = [0], padding = zero>>
  // CHECK: bufferization.to_{{(memref|buffer)}}
  // CHECK: memref.subview
  // CHECK: hivm.hir.store
  tv.view_store %pc[%i], %sum
        : !tv.tensor_view<?xf32, strides=[1], #tv.partition_view<tile = [1024], dim_map = [0], padding = zero>>, tensor<1024xf32>
  tt.return
}

// CHECK-NOT: tv.view_load
// CHECK-NOT: tv.view_store
// CHECK-NOT: tv.make_tensor_view
// CHECK-NOT: !tv.ptr
