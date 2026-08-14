// RUN: triton-opt %s --tensor-view-lowering | FileCheck %s

// Gather (generic path, native "extracted load" form): loop over the sparse
// dim only; each iteration memref.copy's a contiguous block (tile with sparse
// dim = 1) from the data-dependent offset.  {ExtractedLoadOrStore} marks it.

// CHECK-LABEL: tt.func public @gather_last_dim
// CHECK-SAME: memref<?xf32>
tt.func public @gather_last_dim(%src: !tv.ptr<f32>,
                                %cols: tensor<4xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c12 = arith.constant 12 : index
  %base = tv.make_tensor_view %src, sizes = [%c4, %c12], strides = [%c12, %c1]
      : !tv.ptr<f32> -> !tv.tensor_view<4x12xf32, strides=[12, 1]>
  %view = tv.make_gather_scatter_view %base
      : !tv.tensor_view<4x12xf32, strides=[12, 1]>
     -> !tv.tensor_view<4x12xf32, strides=[12, 1], #tv.gather_scatter_view<tile = [4, 4], sparse_dim = [1], padding = zero>>
  // CHECK: memref.alloc
  // CHECK: scf.for
  // CHECK: tensor.extract
  // CHECK: memref.reinterpret_cast
  // CHECK: memref.subview
  // CHECK: memref.copy
  // CHECK: bufferization.to_tensor
  // CHECK-NOT: hivm
  %result = tv.view_load %view[%c0, %cols]
      : !tv.tensor_view<4x12xf32, strides=[12, 1], #tv.gather_scatter_view<tile = [4, 4], sparse_dim = [1], padding = zero>>, index, tensor<4xi32>
     -> tensor<4x4xf32>
  tt.return
}

// Non-last-dimension gather: same scalar loop decomposition.

// CHECK-LABEL: tt.func public @gather_non_last_dim
tt.func public @gather_non_last_dim(%src: !tv.ptr<f32>,
                                    %rows: tensor<4xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c12 = arith.constant 12 : index
  %base = tv.make_tensor_view %src, sizes = [%c12, %c4], strides = [%c4, %c1]
      : !tv.ptr<f32> -> !tv.tensor_view<12x4xf32, strides=[4, 1]>
  %view = tv.make_gather_scatter_view %base
      : !tv.tensor_view<12x4xf32, strides=[4, 1]>
     -> !tv.tensor_view<12x4xf32, strides=[4, 1], #tv.gather_scatter_view<tile = [4, 4], sparse_dim = [0], padding = zero>>
  // CHECK: memref.alloc
  // CHECK: scf.for
  // CHECK: tensor.extract %{{.*}}[%{{.*}}] : tensor<4xi32>
  // CHECK: memref.reinterpret_cast
  // CHECK: memref.subview
  // CHECK: memref.copy
  // CHECK: bufferization.to_tensor
  // CHECK-NOT: hivm
  %result = tv.view_load %view[%rows, %c0]
      : !tv.tensor_view<12x4xf32, strides=[4, 1], #tv.gather_scatter_view<tile = [4, 4], sparse_dim = [0], padding = zero>>, tensor<4xi32>, index
     -> tensor<4x4xf32>
  tt.return
}

// Scatter (generic path): scalar scf.for over the tile, per-element
// materialize_in_destination into the 1-element GM slot (no scatter_store).

// CHECK-LABEL: tt.func public @scatter_non_last_dim
tt.func public @scatter_non_last_dim(%dst: !tv.ptr<f32>,
                                     %rows: tensor<4xi32>,
                                     %value: tensor<4x4xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c12 = arith.constant 12 : index
  %base = tv.make_tensor_view %dst, sizes = [%c12, %c4], strides = [%c4, %c1]
      : !tv.ptr<f32> -> !tv.tensor_view<12x4xf32, strides=[4, 1]>
  %view = tv.make_gather_scatter_view %base
      : !tv.tensor_view<12x4xf32, strides=[4, 1]>
     -> !tv.tensor_view<12x4xf32, strides=[4, 1], #tv.gather_scatter_view<tile = [4, 4], sparse_dim = [0], padding = zero>>
  // CHECK: scf.for
  // CHECK: tensor.extract %{{.*}}[%{{.*}}] : tensor<4xi32>
  // CHECK: memref.reinterpret_cast
  // CHECK: tensor.extract_slice
  // CHECK: bufferization.materialize_in_destination
  // CHECK-NOT: hivm
  tv.view_store %view[%rows, %c0], %value
      : !tv.tensor_view<12x4xf32, strides=[4, 1], #tv.gather_scatter_view<tile = [4, 4], sparse_dim = [0], padding = zero>>, tensor<4x4xf32>, tensor<4xi32>, index
  tt.return
}

// A last-dimension scatter uses the same scalar decomposition.

// CHECK-LABEL: tt.func public @scatter_last_dim
tt.func public @scatter_last_dim(%dst: !tv.ptr<f32>,
                                 %cols: tensor<4xi32>,
                                 %value: tensor<4x4xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c12 = arith.constant 12 : index
  %base = tv.make_tensor_view %dst, sizes = [%c4, %c12], strides = [%c12, %c1]
      : !tv.ptr<f32> -> !tv.tensor_view<4x12xf32, strides=[12, 1]>
  %view = tv.make_gather_scatter_view %base
      : !tv.tensor_view<4x12xf32, strides=[12, 1]>
     -> !tv.tensor_view<4x12xf32, strides=[12, 1], #tv.gather_scatter_view<tile = [4, 4], sparse_dim = [1], padding = zero>>
  // CHECK: scf.for
  // CHECK: bufferization.materialize_in_destination
  // CHECK-NOT: hivm
  tv.view_store %view[%c0, %cols], %value
      : !tv.tensor_view<4x12xf32, strides=[12, 1], #tv.gather_scatter_view<tile = [4, 4], sparse_dim = [1], padding = zero>>, tensor<4x4xf32>, index, tensor<4xi32>
  tt.return
}

// CHECK-NOT: tv.view_load
// CHECK-NOT: tv.view_store
// CHECK-NOT: tv.make_gather_scatter_view
// CHECK-NOT: tv.make_tensor_view
// CHECK-NOT: !tv.ptr
