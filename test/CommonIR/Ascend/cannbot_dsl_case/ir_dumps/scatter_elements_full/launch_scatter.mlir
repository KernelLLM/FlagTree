module {
  func.func @_launch_scatter_c2bdca9cbead816f7364c4a8e630d7f0(%arg0: !cannir.tensor<<gm, f16>, <<(30)>, <(1)>>, <<(30)>, <(1)>>, identity>, %arg1: !cannir.tensor<<gm, i32>, <<(16)>, <(1)>>, <<(16)>, <(1)>>, identity>, %arg2: !cannir.tensor<<gm, f16>, <<(30)>, <(1)>>, <<(30)>, <(1)>>, identity>, %arg3: !cannir.tensor<<gm, f16>, <<(30)>, <(1)>>, <<(30)>, <(1)>>, identity>, %arg4: !cannir.tensor<<gm, i64>, <<(37)>, <(1)>>, <<(37)>, <(1)>>, identity>) attributes {cannbotdsl.contract_digest = "c2bdca9cbead816f7364c4a8e630d7f0a42d1d6cc3351ee0402fe87eb4a9348b", cannbotdsl.public_slot_count = 5 : i64, cannbotdsl.slot_dtype_bits = [16, 32, 16, 16, 64], cannbotdsl.slot_dtype_codes = [2, 0, 2, 2, 0], cannbotdsl.slot_dtype_lanes = [1, 1, 1, 1, 1], cannbotdsl.slot_flags = [1, 1, 1, 1, 1], cannbotdsl.slot_kinds = [1, 1, 1, 1, 1], cannbotdsl.slot_type_digests = ["eba22d29d94a21f8156e3f0fe84dba4b5ce65a343a0ee7df1bcb8a02ea3f231d", "e5e05137b267875876fb923e7cf5b41e85610e843900b5a9d5988dfdb9fd5ee4", "eba22d29d94a21f8156e3f0fe84dba4b5ce65a343a0ee7df1bcb8a02ea3f231d", "eba22d29d94a21f8156e3f0fe84dba4b5ce65a343a0ee7df1bcb8a02ea3f231d", "b9c66a60553cb72cfe65b2a7b1c1abdf583a983445c890111b3816d107a5d9fd"], cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.host} {
    %c1_i64 = arith.constant 1 : i64
    cannir.kernel_launch @kernel__scatter_simt_0(%c1_i64) [%arg0, %arg1, %arg2, %arg3, %arg4] {block_shape = array<i64: 1, 1, 1>, cannbotdsl.profile = {indices = array<i64: 0, 1, 2, 3, 3, 4>, name = "_scatter_simt", op_type = "_scatter_simt", roles = array<i64: 0, 0, 0, 0, 1, 0>}, simt_kernel = true, thread_dim = 256 : i64, thread_shape = array<i64: 256, 1, 1>} : !cannir.tensor<<gm, f16>, <<(30)>, <(1)>>, <<(30)>, <(1)>>, identity>, !cannir.tensor<<gm, i32>, <<(16)>, <(1)>>, <<(16)>, <(1)>>, identity>, !cannir.tensor<<gm, f16>, <<(30)>, <(1)>>, <<(30)>, <(1)>>, identity>, !cannir.tensor<<gm, f16>, <<(30)>, <(1)>>, <<(30)>, <(1)>>, identity>, !cannir.tensor<<gm, i64>, <<(37)>, <(1)>>, <<(37)>, <(1)>>, identity>
    return
  }
  func.func @kernel__scatter_simt_0(%arg0: !cannir.tensor<<gm, f16>, <<(30)>, <(1)>>, <<(30)>, <(1)>>, identity>, %arg1: !cannir.tensor<<gm, i32>, <<(16)>, <(1)>>, <<(16)>, <(1)>>, identity>, %arg2: !cannir.tensor<<gm, f16>, <<(30)>, <(1)>>, <<(30)>, <(1)>>, identity>, %arg3: !cannir.tensor<<gm, f16>, <<(30)>, <(1)>>, <<(30)>, <(1)>>, identity>, %arg4: !cannir.tensor<<gm, i64>, <<(37)>, <(1)>>, <<(37)>, <(1)>>, identity>) attributes {cannir.buffer.auto_sync, cannir.buffer.sync_analyzed, cannir.buffer.sync_verified, cannir.buffer.tensor_state, cannir.core = "aiv", cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.device, cannir.kernel_group = "kernel__scatter_simt_0", cannir.pure_simt} {
    %alloc = memref.alloc() {cannir.arg_index = 0 : i64} : memref<30xf16>
    %alloc_0 = memref.alloc() {cannir.arg_index = 1 : i64} : memref<16xi32>
    %alloc_1 = memref.alloc() {cannir.arg_index = 2 : i64} : memref<30xf16>
    %alloc_2 = memref.alloc() {cannir.arg_index = 3 : i64} : memref<30xf16>
    %alloc_3 = memref.alloc() {cannir.arg_index = 4 : i64} : memref<37xi64>
    ascsimt.vf(%alloc_3, %alloc, %alloc_2, %alloc_0, %alloc_1 : memref<37xi64>, memref<30xf16>, memref<30xf16>, memref<16xi32>, memref<30xf16>) {
    ^bb0(%arg5: memref<37xi64>, %arg6: memref<30xf16>, %arg7: memref<30xf16>, %arg8: memref<16xi32>, %arg9: memref<30xf16>):
      %c30_i64 = arith.constant 30 : i64
      %cst = arith.constant 0.000000e+00 : f16
      %c0_i32 = arith.constant 0 : i32
      %true = arith.constant true
      %c16_i64 = arith.constant 16 : i64
      %c8_i64 = arith.constant 8 : i64
      %false = arith.constant false
      %c24_i64 = arith.constant 24 : i64
      %c1_i64 = arith.constant 1 : i64
      %c0_i64 = arith.constant 0 : i64
      %c36 = arith.constant 36 : index
      %c35 = arith.constant 35 : index
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
      %8 = arith.muli %3, %5 {cannir.vf_threads = 1024 : i64} : i64
      %9 = arith.addi %8, %1 {cannir.vf_threads = 1024 : i64} : i64
      %10 = arith.muli %7, %5 {cannir.vf_threads = 1024 : i64} : i64
      %11 = ascsimt.load %arg5[%c32] : memref<37xi64> -> i64
      %12 = ascsimt.load %arg5[%c33] : memref<37xi64> -> i64
      %13 = ascsimt.load %arg5[%c34] : memref<37xi64> -> i64
      %14 = ascsimt.load %arg5[%c35] : memref<37xi64> -> i64
      %15 = ascsimt.load %arg5[%c36] : memref<37xi64> -> i64
      scf.for %arg10 = %9 to %c30_i64 step %10  : i64 {
        %16 = arith.index_cast %arg10 : i64 to index
        %17 = ascsimt.load %arg6[%16] : memref<30xf16> -> f16
        ascsimt.store %arg7[%16], %17 : memref<30xf16>, f16
        %18:5 = scf.for %arg11 = %c0_i64 to %11 step %c1_i64 iter_args(%arg12 = %arg10, %arg13 = %c0_i64, %arg14 = %true, %arg15 = %c0_i64, %arg16 = %c0_i64) -> (i64, i64, i1, i64, i64)  : i64 {
          %19 = arith.index_cast %arg11 : i64 to index
          %20 = ascsimt.load %arg5[%19] : memref<37xi64> -> i64
          %21 = arith.remsi %arg12, %20 : i64
          %22 = arith.cmpi ne, %21, %c0_i64 : i64
          %23 = arith.cmpi slt, %21, %c0_i64 : i64
          %24 = arith.cmpi slt, %20, %c0_i64 : i64
          %25 = arith.xori %23, %24 : i1
          %26 = arith.andi %22, %25 : i1
          %27 = arith.divsi %arg12, %20 : i64
          %28 = arith.extui %26 : i1 to i64
          %29 = arith.subi %27, %28 : i64
          %30 = arith.muli %29, %20 : i64
          %31 = arith.subi %arg12, %30 : i64
          %32 = arith.cmpi eq, %arg11, %12 : i64
          %33 = arith.select %32, %29, %arg13 : i64
          %34:3 = scf.if %32 -> (i1, i64, i64) {
            scf.yield %arg14, %arg15, %arg16 : i1, i64, i64
          } else {
            %35 = arith.addi %arg11, %c24_i64 : i64
            %36 = arith.index_cast %35 : i64 to index
            %37 = ascsimt.load %arg5[%36] : memref<37xi64> -> i64
            %38 = arith.cmpi sge, %29, %37 : i64
            %39 = arith.select %38, %false, %arg14 : i1
            %40 = arith.addi %arg11, %c8_i64 : i64
            %41 = arith.index_cast %40 : i64 to index
            %42 = ascsimt.load %arg5[%41] : memref<37xi64> -> i64
            %43 = arith.muli %29, %42 : i64
            %44 = arith.addi %arg15, %43 : i64
            %45 = arith.addi %arg11, %c16_i64 : i64
            %46 = arith.index_cast %45 : i64 to index
            %47 = ascsimt.load %arg5[%46] : memref<37xi64> -> i64
            %48 = arith.muli %29, %47 : i64
            %49 = arith.addi %arg16, %48 : i64
            scf.yield %39, %44, %49 : i1, i64, i64
          }
          scf.yield %31, %33, %34#0, %34#1, %34#2 : i64, i64, i1, i64, i64
        }
        scf.if %18#2 {
          %19 = arith.addi %12, %c24_i64 : i64
          %20 = arith.index_cast %19 : i64 to index
          %21 = ascsimt.load %arg5[%20] : memref<37xi64> -> i64
          %22 = arith.addi %12, %c8_i64 : i64
          %23 = arith.index_cast %22 : i64 to index
          %24 = ascsimt.load %arg5[%23] : memref<37xi64> -> i64
          %25 = arith.addi %12, %c16_i64 : i64
          %26 = arith.index_cast %25 : i64 to index
          %27 = ascsimt.load %arg5[%26] : memref<37xi64> -> i64
          scf.for %arg11 = %c0_i64 to %21 step %c1_i64  : i64 {
            %28 = arith.muli %arg11, %24 : i64
            %29 = arith.addi %18#3, %28 : i64
            %30 = arith.index_cast %29 : i64 to index
            %31 = ascsimt.load %arg8[%30] : memref<16xi32> -> i32
            %32 = arith.extsi %31 : i32 to i64
            %33 = arith.cmpi eq, %32, %18#1 : i64
            %34 = arith.cmpi sge, %31, %c0_i32 : i32
            %35 = arith.cmpi slt, %32, %13 : i64
            %36 = arith.select %34, %35, %false : i1
            %37 = arith.select %33, %36, %false : i1
            scf.if %37 {
              %38 = arith.muli %arg11, %27 : i64
              %39 = arith.addi %18#4, %38 : i64
              %40 = arith.index_cast %39 : i64 to index
              %41 = ascsimt.load %arg9[%40] : memref<30xf16> -> f16
              %42 = arith.cmpi eq, %14, %c0_i64 : i64
              scf.if %42 {
                ascsimt.store %arg7[%16], %41 : memref<30xf16>, f16
              } else {
                %43 = arith.cmpi eq, %14, %c1_i64 : i64
                scf.if %43 {
                  %44 = arith.cmpi ne, %15, %c0_i64 : i64
                  scf.if %44 {
                    %45 = ascsimt.load %arg7[%16] : memref<30xf16> -> f16
                    %46 = arith.addf %45, %41 : f16
                    %47 = arith.cmpf une, %46, %cst : f16
                    %48 = arith.uitofp %47 : i1 to f16
                    ascsimt.store %arg7[%16], %48 : memref<30xf16>, f16
                  } else {
                    %45 = ascsimt.load %arg7[%16] : memref<30xf16> -> f16
                    %46 = arith.addf %45, %41 : f16
                    ascsimt.store %arg7[%16], %46 : memref<30xf16>, f16
                  }
                } else {
                  %44 = ascsimt.load %arg7[%16] : memref<30xf16> -> f16
                  %45 = arith.mulf %44, %41 : f16
                  ascsimt.store %arg7[%16], %45 : memref<30xf16>, f16
                }
              }
            }
          }
        }
      } {cannir.vf_threads = 1024 : i64}
    } {threads = 1024 : i64}
    return
  }
}
