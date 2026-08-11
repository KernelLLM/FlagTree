// RUN: triton-opt %s --triton-to-tensor-view --canonicalize | FileCheck %s

// Overlapping-window read (STEP=512 != WINDOW=1024) -> make_strided_view; the
// contiguous store (STEP=1024 == WINDOW) -> make_partition_view.  Uses the
// "hoisted" frontend form: the tile origin (pid*STEP) is a scalar tt.addptr on
// the base, then splatted, then + arange.

// CHECK-LABEL: tt.func public @sliding
tt.func public @sliding(%in: !tt.ptr<f32>, %out: !tt.ptr<f32>) {
  %c512 = arith.constant 512 : i32
  %c1024 = arith.constant 1024 : i32
  %pid = tt.get_program_id x : i32
  %r = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>

  // strided read (hoisted): in + pid*512, splat, + arange
  %in_start = arith.muli %pid, %c512 : i32
  %in2 = tt.addptr %in, %in_start : !tt.ptr<f32>, i32
  %in_sp = tt.splat %in2 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
  %in_ap = tt.addptr %in_sp, %r : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
  // CHECK: tv.make_strided_view
  // CHECK-SAME: #tv.strided_view<tile = [1024], dim_map = [0], traversal_strides = [512], padding = zero>
  // CHECK: tv.view_load
  %x = tt.load %in_ap : tensor<1024x!tt.ptr<f32>>

  // contiguous store (hoisted): out + pid*1024, splat, + arange
  %out_start = arith.muli %pid, %c1024 : i32
  %out2 = tt.addptr %out, %out_start : !tt.ptr<f32>, i32
  %out_sp = tt.splat %out2 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
  %out_ap = tt.addptr %out_sp, %r : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
  // CHECK: tv.make_partition_view
  // CHECK-SAME: #tv.partition_view<tile = [1024], dim_map = [0], padding = zero>
  // CHECK: tv.view_store
  tt.store %out_ap, %x : tensor<1024x!tt.ptr<f32>>
  tt.return
}

// CHECK-NOT: tt.addptr
// CHECK-NOT: tt.load
// CHECK-NOT: !tt.ptr
