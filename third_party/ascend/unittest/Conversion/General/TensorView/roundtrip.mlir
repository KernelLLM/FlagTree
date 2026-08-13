// RUN: triton-opt %s | triton-opt | FileCheck %s

// Note: in operand-type positions MLIR prints the statically-constrained tv
// types in stripped form (e.g. `<f32>` for `!tv.ptr<f32>`); the CHECK lines
// below therefore match op mnemonics, encodings and result tensor types, which
// print stably.

// CHECK-LABEL: func.func @base_and_partition
func.func @base_and_partition(%p: !tv.ptr<f32>, %n: index, %c1: index, %i: index) {
  // CHECK: tv.make_tensor_view %{{.*}}, sizes = [%{{.*}}], strides = [%{{.*}}]
  %v = tv.make_tensor_view %p, sizes = [%n], strides = [%c1]
       : !tv.ptr<f32> -> !tv.tensor_view<?xf32, strides=[1]>
  // CHECK: tv.make_partition_view
  // CHECK-SAME: #tv.partition_view<tile = [128], dim_map = [0], padding = zero>
  %pv = tv.make_partition_view %v
        : !tv.tensor_view<?xf32, strides=[1]>
       -> !tv.tensor_view<?xf32, strides=[1], #tv.partition_view<tile = [128], dim_map = [0], padding = zero>>
  // CHECK: tv.view_load %{{.*}}[%{{.*}}]
  // CHECK-SAME: tensor<128xf32>
  %t = tv.view_load %pv[%i]
       : !tv.tensor_view<?xf32, strides=[1], #tv.partition_view<tile = [128], dim_map = [0], padding = zero>>, index
      -> tensor<128xf32>
  // CHECK: tv.view_store
  tv.view_store %pv[%i], %t
       : !tv.tensor_view<?xf32, strides=[1], #tv.partition_view<tile = [128], dim_map = [0], padding = zero>>, tensor<128xf32>, index
  return
}

// CHECK-LABEL: func.func @partition_2d
func.func @partition_2d(%p: !tv.ptr<f32>, %m: index, %n: index, %sm: index, %s1: index,
                        %i: index, %j: index) {
  %v = tv.make_tensor_view %p, sizes = [%m, %n], strides = [%sm, %s1]
       : !tv.ptr<f32> -> !tv.tensor_view<?x?xf32, strides=[?,1]>
  // CHECK: #tv.partition_view<tile = [128, 256], dim_map = [0, 1], padding = zero>
  %pv = tv.make_partition_view %v
        : !tv.tensor_view<?x?xf32, strides=[?,1]>
       -> !tv.tensor_view<?x?xf32, strides=[?,1], #tv.partition_view<tile = [128, 256], dim_map = [0, 1], padding = zero>>
  // CHECK: tv.view_load
  // CHECK-SAME: tensor<128x256xf32>
  %t = tv.view_load %pv[%i, %j]
       : !tv.tensor_view<?x?xf32, strides=[?,1], #tv.partition_view<tile = [128, 256], dim_map = [0, 1], padding = zero>>, index, index
      -> tensor<128x256xf32>
  return
}

// CHECK-LABEL: func.func @strided_and_gather
func.func @strided_and_gather(%p: !tv.ptr<f32>, %s8: index, %s1: index,
                              %sparse: tensor<4xi32>, %j: index) {
  %v = tv.make_tensor_view %p, sizes = [%s8, %s8], strides = [%s8, %s1]
       : !tv.ptr<f32> -> !tv.tensor_view<8x8xf32, strides=[8,1]>
  // CHECK: tv.make_strided_view
  // CHECK-SAME: #tv.strided_view<tile = [4, 4], dim_map = [0, 1], traversal_strides = [3, 3], padding = zero>
  %sv = tv.make_strided_view %v
        : !tv.tensor_view<8x8xf32, strides=[8,1]>
       -> !tv.tensor_view<8x8xf32, strides=[8,1], #tv.strided_view<tile = [4, 4], dim_map = [0, 1], traversal_strides = [3, 3], padding = zero>>
  // CHECK: tv.make_gather_scatter_view
  // CHECK-SAME: #tv.gather_scatter_view<tile = [4, 4], sparse_dim = [0], padding = zero>
  %gv = tv.make_gather_scatter_view %v
        : !tv.tensor_view<8x8xf32, strides=[8,1]>
       -> !tv.tensor_view<8x8xf32, strides=[8,1], #tv.gather_scatter_view<tile = [4, 4], sparse_dim = [0], padding = zero>>
  // CHECK: tv.view_load
  // CHECK-SAME: tensor<4xi32>, index -> tensor<4x4xf32>
  %gt = tv.view_load %gv[%sparse, %j]
        : !tv.tensor_view<8x8xf32, strides=[8,1], #tv.gather_scatter_view<tile = [4, 4], sparse_dim = [0], padding = zero>>, tensor<4xi32>, index
       -> tensor<4x4xf32>
  // CHECK: tv.view_store
  tv.view_store %gv[%sparse, %j], %gt
        : !tv.tensor_view<8x8xf32, strides=[8,1], #tv.gather_scatter_view<tile = [4, 4], sparse_dim = [0], padding = zero>>, tensor<4x4xf32>, tensor<4xi32>, index
  return
}

// CHECK-LABEL: func.func @ptr_gather_scatter
func.func @ptr_gather_scatter(%base: !tv.ptr<f32>, %i0: tensor<4xindex>,
                              %vals: tensor<4xf32>) {
  // CHECK: tv.ptr_load %{{.*}}, indices = [%{{.*}}]
  // CHECK-SAME: padding = #tv.pad<zero>
  // CHECK-SAME: tensor<4xf32>
  %g = tv.ptr_load %base, indices = [%i0] {padding = #tv.pad<zero>}
       : !tv.ptr<f32>, tensor<4xindex> -> tensor<4xf32>
  // CHECK: tv.ptr_store
  // CHECK-SAME: padding = #tv.pad<zero>
  tv.ptr_store %base, %vals, indices = [%i0] {padding = #tv.pad<zero>}
       : !tv.ptr<f32>, tensor<4xf32>, tensor<4xindex>
  return
}
