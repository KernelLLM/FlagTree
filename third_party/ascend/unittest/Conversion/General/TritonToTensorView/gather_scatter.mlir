// RUN: triton-opt %s --triton-to-tensor-view --canonicalize | FileCheck %s

// A data-dependent row index plus a regular contiguous column range is a
// gather/scatter view.  The index load itself remains a regular partition
// view; the data load and store carry that loaded tensor as sparse_dim 0.

// CHECK-LABEL: tt.func public @gather_scatter_2d
tt.func public @gather_scatter_2d(%indices: !tt.ptr<i32>, %src: !tt.ptr<f32>,
                                  %dst: !tt.ptr<f32>) {
  %c4 = arith.constant 4 : i32
  %pid = tt.get_program_id x : i32
  %origin = arith.muli %pid, %c4 : i32
  %range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %origin_splat = tt.splat %origin : i32 -> tensor<4xi32>
  %index_offsets = arith.addi %origin_splat, %range : tensor<4xi32>
  %index_base = tt.splat %indices : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
  %index_ptrs = tt.addptr %index_base, %index_offsets
      : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  // CHECK: tv.make_partition_view
  // CHECK: %[[SPARSE:.*]] = tv.view_load
  %rows = tt.load %index_ptrs : tensor<4x!tt.ptr<i32>>

  %rows_2d = tt.expand_dims %rows {axis = 1 : i32}
      : tensor<4xi32> -> tensor<4x1xi32>
  %row_stride = arith.constant dense<12> : tensor<4x1xi32>
  %row_offsets = arith.muli %rows_2d, %row_stride : tensor<4x1xi32>
  %cols_2d = tt.expand_dims %range {axis = 0 : i32}
      : tensor<4xi32> -> tensor<1x4xi32>
  %cols = tt.broadcast %cols_2d : tensor<1x4xi32> -> tensor<4x4xi32>

  %src_base = tt.splat %src : !tt.ptr<f32> -> tensor<4x1x!tt.ptr<f32>>
  %src_rows = tt.addptr %src_base, %row_offsets
      : tensor<4x1x!tt.ptr<f32>>, tensor<4x1xi32>
  %src_rows_2d = tt.broadcast %src_rows
      : tensor<4x1x!tt.ptr<f32>> -> tensor<4x4x!tt.ptr<f32>>
  %src_ptrs = tt.addptr %src_rows_2d, %cols
      : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>

  // CHECK: tv.make_tensor_view
  // CHECK-SAME: strides=[12, 1]
  // CHECK: tv.make_gather_scatter_view
  // CHECK-SAME: #tv.gather_scatter_view<tile = [4, 4], sparse_dim = [0], padding = zero>
  // CHECK: tv.view_load %{{.*}}[%[[SPARSE]], %{{.*}}]
  // CHECK-SAME: tensor<4xi32>, index -> tensor<4x4xf32>
  %value = tt.load %src_ptrs : tensor<4x4x!tt.ptr<f32>>

  %dst_base = tt.splat %dst : !tt.ptr<f32> -> tensor<4x1x!tt.ptr<f32>>
  %dst_rows = tt.addptr %dst_base, %row_offsets
      : tensor<4x1x!tt.ptr<f32>>, tensor<4x1xi32>
  %dst_rows_2d = tt.broadcast %dst_rows
      : tensor<4x1x!tt.ptr<f32>> -> tensor<4x4x!tt.ptr<f32>>
  %dst_ptrs = tt.addptr %dst_rows_2d, %cols
      : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>

  // CHECK: tv.make_gather_scatter_view
  // CHECK-SAME: #tv.gather_scatter_view<tile = [4, 4], sparse_dim = [0], padding = zero>
  // CHECK: tv.view_store
  // CHECK-SAME: [%[[SPARSE]], %{{[^]]+}}]
  // CHECK-SAME: #tv.gather_scatter_view<tile = [4, 4], sparse_dim = [0], padding = zero>
  // CHECK-SAME: tensor<4x4xf32>, tensor<4xi32>, index
  tt.store %dst_ptrs, %value : tensor<4x4x!tt.ptr<f32>>
  tt.return
}

// The same matcher also preserves frontend generality when the sparse
// dimension is the last dimension.

// CHECK-LABEL: tt.func public @gather_last_dim
tt.func public @gather_last_dim(%src: !tt.ptr<f32>,
                                %dst: !tt.ptr<f32>,
                                %cols: tensor<4xi32>) {
  %rows = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %rows_2d = tt.expand_dims %rows {axis = 1 : i32}
      : tensor<4xi32> -> tensor<4x1xi32>
  %row_stride = arith.constant dense<12> : tensor<4x1xi32>
  %row_offsets = arith.muli %rows_2d, %row_stride : tensor<4x1xi32>
  %cols_2d = tt.expand_dims %cols {axis = 0 : i32}
      : tensor<4xi32> -> tensor<1x4xi32>
  %col_offsets = tt.broadcast %cols_2d
      : tensor<1x4xi32> -> tensor<4x4xi32>

  %src_base = tt.splat %src : !tt.ptr<f32> -> tensor<4x1x!tt.ptr<f32>>
  %src_rows = tt.addptr %src_base, %row_offsets
      : tensor<4x1x!tt.ptr<f32>>, tensor<4x1xi32>
  %src_rows_2d = tt.broadcast %src_rows
      : tensor<4x1x!tt.ptr<f32>> -> tensor<4x4x!tt.ptr<f32>>
  %src_ptrs = tt.addptr %src_rows_2d, %col_offsets
      : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>

  // CHECK: tv.make_gather_scatter_view
  // CHECK-SAME: #tv.gather_scatter_view<tile = [4, 4], sparse_dim = [1], padding = zero>
  // CHECK: tv.view_load %{{.*}}[%{{.*}}, %{{.*}}]
  // CHECK-SAME: index, tensor<4xi32> -> tensor<4x4xf32>
  %value = tt.load %src_ptrs : tensor<4x4x!tt.ptr<f32>>

  // Keep the gather result live across canonicalization with a regular 2-D
  // store.  This store is intentionally a partition access, independent of
  // the sparse source-column indices above.
  %regular_cols_2d = tt.expand_dims %rows {axis = 0 : i32}
      : tensor<4xi32> -> tensor<1x4xi32>
  %regular_cols = tt.broadcast %regular_cols_2d
      : tensor<1x4xi32> -> tensor<4x4xi32>
  %dst_base = tt.splat %dst : !tt.ptr<f32> -> tensor<4x1x!tt.ptr<f32>>
  %dst_rows = tt.addptr %dst_base, %row_offsets
      : tensor<4x1x!tt.ptr<f32>>, tensor<4x1xi32>
  %dst_rows_2d = tt.broadcast %dst_rows
      : tensor<4x1x!tt.ptr<f32>> -> tensor<4x4x!tt.ptr<f32>>
  %dst_ptrs = tt.addptr %dst_rows_2d, %regular_cols
      : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>
  // CHECK: tv.make_partition_view
  // CHECK: tv.view_store
  tt.store %dst_ptrs, %value : tensor<4x4x!tt.ptr<f32>>
  tt.return
}

// CHECK-NOT: tt.load
// CHECK-NOT: tt.store
// CHECK-NOT: tt.addptr
// CHECK-NOT: !tt.ptr
