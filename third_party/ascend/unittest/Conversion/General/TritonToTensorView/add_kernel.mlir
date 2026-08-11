// RUN: triton-opt %s --triton-to-tensor-view --canonicalize | FileCheck %s

// The add_kernel access pattern (tensor-of-ptr form, as produced by the Triton
// frontend) should be lowered to tv view ops, with the pointer arguments
// rewritten to !tv.ptr and the compute (arith.addf) left untouched.

// CHECK-LABEL: tt.func public @add_kernel
// CHECK-SAME: !tv.ptr<f32>
// CHECK-SAME: !tv.ptr<f32>
// CHECK-SAME: !tv.ptr<f32>
tt.func public @add_kernel(%x_ptr: !tt.ptr<f32>, %y_ptr: !tt.ptr<f32>,
                           %out_ptr: !tt.ptr<f32>, %n: i32) {
  %cst = arith.constant dense<0.000000e+00> : tensor<1024xf32>
  %c1024 = arith.constant 1024 : i32
  %pid = tt.get_program_id x : i32
  %bs = arith.muli %pid, %c1024 : i32
  %r = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
  %bss = tt.splat %bs : i32 -> tensor<1024xi32>
  %off = arith.addi %bss, %r : tensor<1024xi32>
  %ns = tt.splat %n : i32 -> tensor<1024xi32>
  %mask = arith.cmpi slt, %off, %ns : tensor<1024xi32>

  // CHECK: tv.make_tensor_view
  // CHECK: tv.make_partition_view
  // CHECK-SAME: #tv.partition_view<tile = [1024], dim_map = [0], padding = zero>
  // CHECK: tv.view_load
  // CHECK-SAME: tensor<1024xf32>
  %xs = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
  %xp = tt.addptr %xs, %off : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
  %x = tt.load %xp, %mask, %cst : tensor<1024x!tt.ptr<f32>>

  // CHECK: tv.view_load
  %ys = tt.splat %y_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
  %yp = tt.addptr %ys, %off : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
  %y = tt.load %yp, %mask, %cst : tensor<1024x!tt.ptr<f32>>

  // CHECK: arith.addf %{{.*}}, %{{.*}} : tensor<1024xf32>
  %sum = arith.addf %x, %y : tensor<1024xf32>

  // CHECK: tv.view_store
  %os = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
  %op = tt.addptr %os, %off : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
  tt.store %op, %sum, %mask : tensor<1024x!tt.ptr<f32>>
  tt.return
}

// CHECK-NOT: tt.addptr
// CHECK-NOT: tt.load
// CHECK-NOT: tt.store
// CHECK-NOT: !tt.ptr
