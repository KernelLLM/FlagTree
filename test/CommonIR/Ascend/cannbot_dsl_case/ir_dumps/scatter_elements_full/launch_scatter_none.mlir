module {
  func.func @_launch_scatter_none_3d8417e36b65a10515095373859a6078(%arg0: !cannir.tensor<<gm, i32>, <<(4)>, <(1)>>, <<(4)>, <(1)>>, identity>, %arg1: !cannir.tensor<<gm, i64>, <<(2)>, <(1)>>, <<(2)>, <(1)>>, identity>, %arg2: !cannir.tensor<<gm, i32>, <<(2)>, <(1)>>, <<(2)>, <(1)>>, identity>, %arg3: !cannir.tensor<<gm, i32>, <<(4)>, <(1)>>, <<(4)>, <(1)>>, identity>, %arg4: !cannir.tensor<<gm, i64>, <<(37)>, <(1)>>, <<(37)>, <(1)>>, identity>) attributes {cannbotdsl.contract_digest = "3d8417e36b65a10515095373859a607879ba8a19a14dd4a906d351e6e16664db", cannbotdsl.public_slot_count = 5 : i64, cannbotdsl.slot_dtype_bits = [32, 64, 32, 32, 64], cannbotdsl.slot_dtype_codes = [0, 0, 0, 0, 0], cannbotdsl.slot_dtype_lanes = [1, 1, 1, 1, 1], cannbotdsl.slot_flags = [1, 1, 1, 1, 1], cannbotdsl.slot_kinds = [1, 1, 1, 1, 1], cannbotdsl.slot_type_digests = ["ec5bf40ff868ab297a8582aafc213f00dc9214173b0931392ed5ab2c1a5e843f", "4126671a83041647cd98afed881383228865da287ea44573939f32b1d956765b", "d40864c5f2017cb31c810e2dda653607219d1b20485f1df30c3d133fa11754d9", "ec5bf40ff868ab297a8582aafc213f00dc9214173b0931392ed5ab2c1a5e843f", "b9c66a60553cb72cfe65b2a7b1c1abdf583a983445c890111b3816d107a5d9fd"], cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.host} {
    %c1_i64 = arith.constant 1 : i64
    cannir.kernel_launch @kernel__scatter_none_simt_0(%c1_i64) [%arg0, %arg1, %arg2, %arg3, %arg4] {block_shape = array<i64: 1, 1, 1>, cannbotdsl.profile = {indices = array<i64: 0, 1, 2, 3, 4>, name = "_scatter_none_simt", op_type = "_scatter_none_simt", roles = array<i64: 0, 0, 0, 1, 0>}, simt_kernel = true, thread_dim = 256 : i64, thread_shape = array<i64: 256, 1, 1>} : !cannir.tensor<<gm, i32>, <<(4)>, <(1)>>, <<(4)>, <(1)>>, identity>, !cannir.tensor<<gm, i64>, <<(2)>, <(1)>>, <<(2)>, <(1)>>, identity>, !cannir.tensor<<gm, i32>, <<(2)>, <(1)>>, <<(2)>, <(1)>>, identity>, !cannir.tensor<<gm, i32>, <<(4)>, <(1)>>, <<(4)>, <(1)>>, identity>, !cannir.tensor<<gm, i64>, <<(37)>, <(1)>>, <<(37)>, <(1)>>, identity>
    return
  }
  func.func @kernel__scatter_none_simt_0(%arg0: !cannir.tensor<<gm, i32>, <<(4)>, <(1)>>, <<(4)>, <(1)>>, identity>, %arg1: !cannir.tensor<<gm, i64>, <<(2)>, <(1)>>, <<(2)>, <(1)>>, identity>, %arg2: !cannir.tensor<<gm, i32>, <<(2)>, <(1)>>, <<(2)>, <(1)>>, identity>, %arg3: !cannir.tensor<<gm, i32>, <<(4)>, <(1)>>, <<(4)>, <(1)>>, identity>, %arg4: !cannir.tensor<<gm, i64>, <<(37)>, <(1)>>, <<(37)>, <(1)>>, identity>) attributes {cannir.buffer.auto_sync, cannir.buffer.sync_analyzed, cannir.buffer.sync_verified, cannir.buffer.tensor_state, cannir.core = "aiv", cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.device, cannir.kernel_group = "kernel__scatter_none_simt_0", cannir.pure_simt} {
    %alloc = memref.alloc() {cannir.arg_index = 0 : i64} : memref<4xi32>
    %alloc_0 = memref.alloc() {cannir.arg_index = 1 : i64} : memref<2xi64>
    %alloc_1 = memref.alloc() {cannir.arg_index = 2 : i64} : memref<2xi32>
    %alloc_2 = memref.alloc() {cannir.arg_index = 3 : i64} : memref<4xi32>
    %alloc_3 = memref.alloc() {cannir.arg_index = 4 : i64} : memref<37xi64>
    ascsimt.vf(%alloc_3, %alloc_0, %alloc_1, %alloc_2, %alloc : memref<37xi64>, memref<2xi64>, memref<2xi32>, memref<4xi32>, memref<4xi32>) {
    ^bb0(%arg5: memref<37xi64>, %arg6: memref<2xi64>, %arg7: memref<2xi32>, %arg8: memref<4xi32>, %arg9: memref<4xi32>):
      %c4_i64 = arith.constant 4 : i64
      %true = arith.constant true
      %c16_i64 = arith.constant 16 : i64
      %c8_i64 = arith.constant 8 : i64
      %false = arith.constant false
      %c24_i64 = arith.constant 24 : i64
      %c1_i64 = arith.constant 1 : i64
      %c0_i64 = arith.constant 0 : i64
      %c34 = arith.constant 34 : index
      %c33 = arith.constant 33 : index
      %c32 = arith.constant 32 : index
      %0 = ascsimt.thread_idx {ascsimt.scope = 1 : i64, axis = "x", cannir.vf_threads = 1024 : i64} : i32
      %1 = arith.extsi %0 {cannir.vf_threads = 1024 : i64} : i32 to i64
      %2 = ascsimt.block_idx {ascsimt.scope = 1 : i64, axis = "x", cannir.vf_threads = 1024 : i64} : i32
      %3 = arith.extsi %2 {cannir.vf_threads = 1024 : i64} : i32 to i64
      %4 = ascsimt.thread_dim {ascsimt.scope = 1 : i64, axis = "x", cannir.vf_threads = 1024 : i64} : i32
      %5 = arith.extsi %4 {cannir.vf_threads = 1024 : i64} : i32 to i64
      %6 = ascsimt.block_dim {ascsimt.scope = 1 : i64, axis = "x", cannir.vf_threads = 1024 : i64} : i32
      %7 = arith.extsi %6 {cannir.vf_threads = 1024 : i64} : i32 to i64
      %8 = ascsimt.load %arg5[%c32] : memref<37xi64> -> i64
      %9 = ascsimt.load %arg5[%c33] : memref<37xi64> -> i64
      %10 = ascsimt.load %arg5[%c34] : memref<37xi64> -> i64
      %11 = arith.muli %3, %5 {cannir.vf_threads = 1024 : i64} : i64
      %12 = arith.addi %11, %1 {cannir.vf_threads = 1024 : i64} : i64
      %13 = arith.muli %7, %5 {cannir.vf_threads = 1024 : i64} : i64
      scf.for %arg10 = %12 to %c4_i64 step %13  : i64 {
        %14:5 = scf.for %arg11 = %c0_i64 to %8 step %c1_i64 iter_args(%arg12 = %arg10, %arg13 = %c0_i64, %arg14 = %true, %arg15 = %c0_i64, %arg16 = %c0_i64) -> (i64, i64, i1, i64, i64)  : i64 {
          %16 = arith.index_cast %arg11 : i64 to index
          %17 = ascsimt.load %arg5[%16] : memref<37xi64> -> i64
          %18 = arith.remsi %arg12, %17 : i64
          %19 = arith.cmpi ne, %18, %c0_i64 : i64
          %20 = arith.cmpi slt, %18, %c0_i64 : i64
          %21 = arith.cmpi slt, %17, %c0_i64 : i64
          %22 = arith.xori %20, %21 : i1
          %23 = arith.andi %19, %22 : i1
          %24 = arith.divsi %arg12, %17 : i64
          %25 = arith.extui %23 : i1 to i64
          %26 = arith.subi %24, %25 : i64
          %27 = arith.muli %26, %17 : i64
          %28 = arith.subi %arg12, %27 : i64
          %29 = arith.cmpi eq, %arg11, %9 : i64
          %30 = arith.select %29, %26, %arg13 : i64
          %31:3 = scf.if %29 -> (i1, i64, i64) {
            scf.yield %arg14, %arg15, %arg16 : i1, i64, i64
          } else {
            %32 = arith.addi %arg11, %c24_i64 : i64
            %33 = arith.index_cast %32 : i64 to index
            %34 = ascsimt.load %arg5[%33] : memref<37xi64> -> i64
            %35 = arith.cmpi sge, %26, %34 : i64
            %36 = arith.select %35, %false, %arg14 : i1
            %37 = arith.addi %arg11, %c8_i64 : i64
            %38 = arith.index_cast %37 : i64 to index
            %39 = ascsimt.load %arg5[%38] : memref<37xi64> -> i64
            %40 = arith.muli %26, %39 : i64
            %41 = arith.addi %arg15, %40 : i64
            %42 = arith.addi %arg11, %c16_i64 : i64
            %43 = arith.index_cast %42 : i64 to index
            %44 = ascsimt.load %arg5[%43] : memref<37xi64> -> i64
            %45 = arith.muli %26, %44 : i64
            %46 = arith.addi %arg16, %45 : i64
            scf.yield %36, %41, %46 : i1, i64, i64
          }
          scf.yield %28, %30, %31#0, %31#1, %31#2 : i64, i64, i1, i64, i64
        }
        %15:2 = scf.if %14#2 -> (i1, i64) {
          %16 = arith.addi %9, %c24_i64 : i64
          %17 = arith.index_cast %16 : i64 to index
          %18 = ascsimt.load %arg5[%17] : memref<37xi64> -> i64
          %19 = arith.addi %9, %c8_i64 : i64
          %20 = arith.index_cast %19 : i64 to index
          %21 = ascsimt.load %arg5[%20] : memref<37xi64> -> i64
          %22 = arith.addi %9, %c16_i64 : i64
          %23 = arith.index_cast %22 : i64 to index
          %24 = ascsimt.load %arg5[%23] : memref<37xi64> -> i64
          %25:2 = scf.for %arg11 = %c0_i64 to %18 step %c1_i64 iter_args(%arg12 = %false, %arg13 = %c0_i64) -> (i1, i64)  : i64 {
            %26 = arith.muli %arg11, %21 : i64
            %27 = arith.addi %14#3, %26 : i64
            %28 = arith.index_cast %27 : i64 to index
            %29 = ascsimt.load %arg6[%28] : memref<2xi64> -> i64
            %30 = arith.cmpi eq, %29, %14#1 : i64
            %31 = arith.cmpi sge, %29, %c0_i64 : i64
            %32 = arith.cmpi slt, %29, %10 : i64
            %33 = arith.select %31, %32, %false : i1
            %34 = arith.select %30, %33, %false : i1
            %35 = arith.select %34, %true, %arg12 : i1
            %36 = arith.muli %arg11, %24 : i64
            %37 = arith.addi %14#4, %36 : i64
            %38 = arith.select %34, %37, %arg13 : i64
            scf.yield %35, %38 : i1, i64
          }
          scf.yield %25#0, %25#1 : i1, i64
        } else {
          scf.yield %false, %c0_i64 : i1, i64
        }
        scf.if %15#0 {
          %16 = arith.index_cast %15#1 : i64 to index
          %17 = ascsimt.load %arg7[%16] : memref<2xi32> -> i32
          %18 = arith.index_cast %arg10 : i64 to index
          ascsimt.store %arg8[%18], %17 : memref<4xi32>, i32
        } else {
          %16 = arith.index_cast %arg10 : i64 to index
          %17 = ascsimt.load %arg9[%16] : memref<4xi32> -> i32
          ascsimt.store %arg8[%16], %17 : memref<4xi32>, i32
        }
      } {cannir.vf_threads = 1024 : i64}
    } {threads = 1024 : i64}
    return
  }
}
