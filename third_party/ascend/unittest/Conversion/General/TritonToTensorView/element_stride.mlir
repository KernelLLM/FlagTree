// RUN: triton-opt %s --triton-to-tensor-view --canonicalize | FileCheck %s

// strided_copy: read in[i*2] (element stride 2, contiguous partition tiling),
// contiguous write.  The load's base view carries strides=[2].

// CHECK-LABEL: tt.func public @strided_copy
tt.func public @strided_copy(%in: !tt.ptr<f32>, %out: !tt.ptr<f32>, %n: i32) {
  %cst = arith.constant dense<0.000000e+00> : tensor<1024xf32>
  %c1024 = arith.constant 1024 : i32
  %c2 = arith.constant dense<2> : tensor<1024xi32>
  %pid = tt.get_program_id x : i32
  %bs = arith.muli %pid, %c1024 : i32
  %r = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
  %bss = tt.splat %bs : i32 -> tensor<1024xi32>
  %off = arith.addi %bss, %r : tensor<1024xi32>
  %ns = tt.splat %n : i32 -> tensor<1024xi32>
  %mask = arith.cmpi slt, %off, %ns : tensor<1024xi32>
  %soff = arith.muli %off, %c2 : tensor<1024xi32>
  %inp = tt.splat %in : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
  %inap = tt.addptr %inp, %soff : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
  // CHECK: tv.make_tensor_view
  // CHECK-SAME: strides=[2]
  // CHECK: tv.make_partition_view
  // CHECK: tv.view_load
  %x = tt.load %inap, %mask, %cst : tensor<1024x!tt.ptr<f32>>

  %outp = tt.splat %out : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
  %outap = tt.addptr %outp, %off : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
  // CHECK: tv.view_store
  tt.store %outap, %x, %mask : tensor<1024x!tt.ptr<f32>>
  tt.return
}

// CHECK-NOT: tt.addptr
// CHECK-NOT: tt.load
// CHECK-NOT: !tt.ptr
