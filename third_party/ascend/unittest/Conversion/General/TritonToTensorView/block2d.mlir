// RUN: triton-opt %s --triton-to-tensor-view --canonicalize | FileCheck %s

// 2-D block copy (GEMM-tile-like): [BM,BN] tile from a row-major matrix with
// dynamic row stride N.  Lowers to a rank-2 make_tensor_view (strides=[?,1]) +
// make_partition_view (tile=[64,64]) + view_load/store.

// CHECK-LABEL: tt.func public @block_copy_2d
tt.func public @block_copy_2d(%in_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>, %N: i32) {
  %c64 = arith.constant 64 : i32
  %pid_m = tt.get_program_id x : i32
  %pid_n = tt.get_program_id y : i32
  %offs_m = arith.muli %pid_m, %c64 : i32
  %r = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
  %offs_m_1 = tt.splat %offs_m : i32 -> tensor<64xi32>
  %offs_m_2 = arith.addi %offs_m_1, %r : tensor<64xi32>
  %offs_n = arith.muli %pid_n, %c64 : i32
  %offs_n_3 = tt.splat %offs_n : i32 -> tensor<64xi32>
  %offs_n_4 = arith.addi %offs_n_3, %r : tensor<64xi32>
  %em = tt.expand_dims %offs_m_2 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
  %Ns = tt.splat %N : i32 -> tensor<64x1xi32>
  %rowc = arith.muli %em, %Ns : tensor<64x1xi32>
  %inp = tt.splat %in_ptr : !tt.ptr<f32> -> tensor<64x1x!tt.ptr<f32>>
  %inrp = tt.addptr %inp, %rowc : tensor<64x1x!tt.ptr<f32>>, tensor<64x1xi32>
  %en = tt.expand_dims %offs_n_4 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
  %inbp = tt.broadcast %inrp : tensor<64x1x!tt.ptr<f32>> -> tensor<64x64x!tt.ptr<f32>>
  %enb = tt.broadcast %en : tensor<1x64xi32> -> tensor<64x64xi32>
  %inap = tt.addptr %inbp, %enb : tensor<64x64x!tt.ptr<f32>>, tensor<64x64xi32>
  // CHECK: tv.make_tensor_view
  // CHECK-SAME: strides=[?, 1]
  // CHECK: tv.make_partition_view
  // CHECK-SAME: #tv.partition_view<tile = [64, 64], dim_map = [0, 1], padding = zero>
  // CHECK: tv.view_load
  // CHECK-SAME: tensor<64x64xf32>
  %x = tt.load %inap : tensor<64x64x!tt.ptr<f32>>

  %outp = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<64x1x!tt.ptr<f32>>
  %outrp = tt.addptr %outp, %rowc : tensor<64x1x!tt.ptr<f32>>, tensor<64x1xi32>
  %outbp = tt.broadcast %outrp : tensor<64x1x!tt.ptr<f32>> -> tensor<64x64x!tt.ptr<f32>>
  %outap = tt.addptr %outbp, %enb : tensor<64x64x!tt.ptr<f32>>, tensor<64x64xi32>
  // CHECK: tv.view_store
  tt.store %outap, %x : tensor<64x64x!tt.ptr<f32>>
  tt.return
}

// CHECK-NOT: tt.addptr
// CHECK-NOT: tt.broadcast
// CHECK-NOT: tt.expand_dims
// CHECK-NOT: !tt.ptr
