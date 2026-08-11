// RUN: triton-opt %s --triton-to-tensor-view --canonicalize | FileCheck %s

// Overlapping-window read (STEP=512 != WINDOW=1024) lowers to make_strided_view;
// the contiguous store (STEP=1024 == WINDOW) stays make_partition_view.

// CHECK-LABEL: tt.func public @sliding
tt.func public @sliding(%in: !tt.ptr<f32>, %out: !tt.ptr<f32>) {
  %c512 = arith.constant 512 : i32
  %c1024 = arith.constant 1024 : i32
  %pid = tt.get_program_id x : i32
  %r = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>

  // strided read: origin steps by 512, tile size 1024 -> overlap
  %in_start = arith.muli %pid, %c512 : i32
  %in_ss = tt.splat %in_start : i32 -> tensor<1024xi32>
  %in_off = arith.addi %in_ss, %r : tensor<1024xi32>
  %in_p = tt.splat %in : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
  %in_ap = tt.addptr %in_p, %in_off : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
  // CHECK: tv.make_strided_view
  // CHECK-SAME: #tv.strided_view<tile = [1024], dim_map = [0], traversal_strides = [512], padding = zero>
  // CHECK: tv.view_load
  %x = tt.load %in_ap : tensor<1024x!tt.ptr<f32>>

  // contiguous store: origin steps by 1024 == tile
  %out_start = arith.muli %pid, %c1024 : i32
  %out_ss = tt.splat %out_start : i32 -> tensor<1024xi32>
  %out_off = arith.addi %out_ss, %r : tensor<1024xi32>
  %out_p = tt.splat %out : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
  %out_ap = tt.addptr %out_p, %out_off : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
  // CHECK: tv.make_partition_view
  // CHECK-SAME: #tv.partition_view<tile = [1024], dim_map = [0], padding = zero>
  // CHECK: tv.view_store
  tt.store %out_ap, %x : tensor<1024x!tt.ptr<f32>>
  tt.return
}

// CHECK-NOT: tt.addptr
// CHECK-NOT: tt.load
// CHECK-NOT: !tt.ptr
