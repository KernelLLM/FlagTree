module {
  func.func @_launch_scatter_int64_reduce_36569e1db22680edc3133590a76cf386(%arg0: !cannir.tensor<<gm, i32>, <<(10)>, <(1)>>, <<(10)>, <(1)>>, identity>, %arg1: !cannir.tensor<<gm, i64>, <<(10)>, <(1)>>, <<(10)>, <(1)>>, identity>, %arg2: !cannir.tensor<<gm, i32>, <<(20)>, <(1)>>, <<(20)>, <(1)>>, identity>, %arg3: !cannir.tensor<<gm, i32>, <<(10)>, <(1)>>, <<(10)>, <(1)>>, identity>, %arg4: !cannir.tensor<<gm, i64>, <<(37)>, <(1)>>, <<(37)>, <(1)>>, identity>) attributes {cannbotdsl.contract_digest = "36569e1db22680edc3133590a76cf38617894662ee17a88587466be38d3a2982", cannbotdsl.public_slot_count = 5 : i64, cannbotdsl.slot_dtype_bits = [32, 64, 32, 32, 64], cannbotdsl.slot_dtype_codes = [0, 0, 0, 0, 0], cannbotdsl.slot_dtype_lanes = [1, 1, 1, 1, 1], cannbotdsl.slot_flags = [1, 1, 1, 1, 1], cannbotdsl.slot_kinds = [1, 1, 1, 1, 1], cannbotdsl.slot_type_digests = ["5cdef4fa2a34913a38da554d8834ebd0b71da96405860aca0ac9f37ab9340855", "4daf26c567b37651f171602163d947b73110cf25edbe22648bfebaebad80e46b", "550bfc46ff605d503f1d8a8e13b8b1445267545b6971496523369b38652c7c37", "5cdef4fa2a34913a38da554d8834ebd0b71da96405860aca0ac9f37ab9340855", "b9c66a60553cb72cfe65b2a7b1c1abdf583a983445c890111b3816d107a5d9fd"], cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.host} {
    %c1_i64 = arith.constant 1 : i64
    cannir.kernel_launch @kernel__scatter_int64_reduce_simt_0(%c1_i64) [%arg0, %arg1, %arg2, %arg3, %arg4] {block_shape = array<i64: 1, 1, 1>, cannbotdsl.profile = {indices = array<i64: 0, 1, 2, 3, 4>, name = "_scatter_int64_reduce_simt", op_type = "_scatter_int64_reduce_simt", roles = array<i64: 0, 0, 0, 1, 0>}, simt_kernel = true, thread_dim = 256 : i64, thread_shape = array<i64: 256, 1, 1>} : !cannir.tensor<<gm, i32>, <<(10)>, <(1)>>, <<(10)>, <(1)>>, identity>, !cannir.tensor<<gm, i64>, <<(10)>, <(1)>>, <<(10)>, <(1)>>, identity>, !cannir.tensor<<gm, i32>, <<(20)>, <(1)>>, <<(20)>, <(1)>>, identity>, !cannir.tensor<<gm, i32>, <<(10)>, <(1)>>, <<(10)>, <(1)>>, identity>, !cannir.tensor<<gm, i64>, <<(37)>, <(1)>>, <<(37)>, <(1)>>, identity>
    return
  }
  func.func @kernel__scatter_int64_reduce_simt_0(%arg0: !cannir.tensor<<gm, i32>, <<(10)>, <(1)>>, <<(10)>, <(1)>>, identity>, %arg1: !cannir.tensor<<gm, i64>, <<(10)>, <(1)>>, <<(10)>, <(1)>>, identity>, %arg2: !cannir.tensor<<gm, i32>, <<(20)>, <(1)>>, <<(20)>, <(1)>>, identity>, %arg3: !cannir.tensor<<gm, i32>, <<(10)>, <(1)>>, <<(10)>, <(1)>>, identity>, %arg4: !cannir.tensor<<gm, i64>, <<(37)>, <(1)>>, <<(37)>, <(1)>>, identity>) attributes {cannir.buffer.auto_sync, cannir.buffer.sync_analyzed, cannir.buffer.sync_verified, cannir.buffer.tensor_state, cannir.core = "aiv", cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.device, cannir.kernel_group = "kernel__scatter_int64_reduce_simt_0", cannir.pure_simt} {
    %alloc = memref.alloc() {cannir.arg_index = 0 : i64} : memref<10xi32>
    %alloc_0 = memref.alloc() {cannir.arg_index = 1 : i64} : memref<10xi64>
    %alloc_1 = memref.alloc() {cannir.arg_index = 2 : i64} : memref<20xi32>
    %alloc_2 = memref.alloc() {cannir.arg_index = 3 : i64} : memref<10xi32>
    %alloc_3 = memref.alloc() {cannir.arg_index = 4 : i64} : memref<37xi64>
    ascsimt.vf(%alloc_3, %alloc, %alloc_0, %alloc_1, %alloc_2 : memref<37xi64>, memref<10xi32>, memref<10xi64>, memref<20xi32>, memref<10xi32>) {
    ^bb0(%arg5: memref<37xi64>, %arg6: memref<10xi32>, %arg7: memref<10xi64>, %arg8: memref<20xi32>, %arg9: memref<10xi32>):
      %c5_i64 = arith.constant 5 : i64
      %c2147483648_i64 = arith.constant 2147483648 : i64
      %c4294967296_i64 = arith.constant 4294967296 : i64
      %true = arith.constant true
      %c16_i64 = arith.constant 16 : i64
      %c8_i64 = arith.constant 8 : i64
      %false = arith.constant false
      %c24_i64 = arith.constant 24 : i64
      %c0_i64 = arith.constant 0 : i64
      %c1_i64 = arith.constant 1 : i64
      %c2_i64 = arith.constant 2 : i64
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
      %8 = ascsimt.load %arg5[%c32] : memref<37xi64> -> i64
      %9 = ascsimt.load %arg5[%c33] : memref<37xi64> -> i64
      %10 = ascsimt.load %arg5[%c34] : memref<37xi64> -> i64
      %11 = ascsimt.load %arg5[%c35] : memref<37xi64> -> i64
      %12 = arith.muli %3, %5 {cannir.vf_threads = 1024 : i64} : i64
      %13 = arith.addi %12, %1 {cannir.vf_threads = 1024 : i64} : i64
      %14 = arith.muli %7, %5 {cannir.vf_threads = 1024 : i64} : i64
      scf.for %arg10 = %13 to %c5_i64 step %14  : i64 {
        %15 = arith.muli %arg10, %c2_i64 : i64
        %16 = arith.index_cast %15 : i64 to index
        %17 = ascsimt.load %arg6[%16] : memref<10xi32> -> i32
        %18 = arith.addi %15, %c1_i64 : i64
        %19 = arith.index_cast %18 : i64 to index
        %20 = ascsimt.load %arg6[%19] : memref<10xi32> -> i32
        %21:5 = scf.for %arg11 = %c0_i64 to %8 step %c1_i64 iter_args(%arg12 = %arg10, %arg13 = %c0_i64, %arg14 = %true, %arg15 = %c0_i64, %arg16 = %c0_i64) -> (i64, i64, i1, i64, i64)  : i64 {
          %23 = arith.index_cast %arg11 : i64 to index
          %24 = ascsimt.load %arg5[%23] : memref<37xi64> -> i64
          %25 = arith.remsi %arg12, %24 : i64
          %26 = arith.cmpi ne, %25, %c0_i64 : i64
          %27 = arith.cmpi slt, %25, %c0_i64 : i64
          %28 = arith.cmpi slt, %24, %c0_i64 : i64
          %29 = arith.xori %27, %28 : i1
          %30 = arith.andi %26, %29 : i1
          %31 = arith.divsi %arg12, %24 : i64
          %32 = arith.extui %30 : i1 to i64
          %33 = arith.subi %31, %32 : i64
          %34 = arith.muli %33, %24 : i64
          %35 = arith.subi %arg12, %34 : i64
          %36 = arith.cmpi eq, %arg11, %9 : i64
          %37 = arith.select %36, %33, %arg13 : i64
          %38:3 = scf.if %36 -> (i1, i64, i64) {
            scf.yield %arg14, %arg15, %arg16 : i1, i64, i64
          } else {
            %39 = arith.addi %arg11, %c24_i64 : i64
            %40 = arith.index_cast %39 : i64 to index
            %41 = ascsimt.load %arg5[%40] : memref<37xi64> -> i64
            %42 = arith.cmpi sge, %33, %41 : i64
            %43 = arith.select %42, %false, %arg14 : i1
            %44 = arith.addi %arg11, %c8_i64 : i64
            %45 = arith.index_cast %44 : i64 to index
            %46 = ascsimt.load %arg5[%45] : memref<37xi64> -> i64
            %47 = arith.muli %33, %46 : i64
            %48 = arith.addi %arg15, %47 : i64
            %49 = arith.addi %arg11, %c16_i64 : i64
            %50 = arith.index_cast %49 : i64 to index
            %51 = ascsimt.load %arg5[%50] : memref<37xi64> -> i64
            %52 = arith.muli %33, %51 : i64
            %53 = arith.addi %arg16, %52 : i64
            scf.yield %43, %48, %53 : i1, i64, i64
          }
          scf.yield %35, %37, %38#0, %38#1, %38#2 : i64, i64, i1, i64, i64
        }
        %22:2 = scf.if %21#2 -> (i32, i32) {
          %23 = arith.addi %9, %c24_i64 : i64
          %24 = arith.index_cast %23 : i64 to index
          %25 = ascsimt.load %arg5[%24] : memref<37xi64> -> i64
          %26 = arith.addi %9, %c8_i64 : i64
          %27 = arith.index_cast %26 : i64 to index
          %28 = ascsimt.load %arg5[%27] : memref<37xi64> -> i64
          %29 = arith.addi %9, %c16_i64 : i64
          %30 = arith.index_cast %29 : i64 to index
          %31 = ascsimt.load %arg5[%30] : memref<37xi64> -> i64
          %32:2 = scf.for %arg11 = %c0_i64 to %25 step %c1_i64 iter_args(%arg12 = %20, %arg13 = %17) -> (i32, i32)  : i64 {
            %33 = arith.muli %arg11, %28 : i64
            %34 = arith.addi %21#3, %33 : i64
            %35 = arith.index_cast %34 : i64 to index
            %36 = ascsimt.load %arg7[%35] : memref<10xi64> -> i64
            %37 = arith.cmpi eq, %36, %21#1 : i64
            %38 = arith.cmpi sge, %36, %c0_i64 : i64
            %39 = arith.cmpi slt, %36, %10 : i64
            %40 = arith.select %38, %39, %false : i1
            %41 = arith.select %37, %40, %false : i1
            %42:2 = scf.if %41 -> (i32, i32) {
              %43 = arith.muli %arg11, %31 : i64
              %44 = arith.addi %21#4, %43 : i64
              %45 = arith.muli %44, %c2_i64 : i64
              %46 = arith.index_cast %45 : i64 to index
              %47 = ascsimt.load %arg8[%46] : memref<20xi32> -> i32
              %48 = arith.addi %45, %c1_i64 : i64
              %49 = arith.index_cast %48 : i64 to index
              %50 = ascsimt.load %arg8[%49] : memref<20xi32> -> i32
              %51 = arith.extui %arg13 : i32 to i64
              %52 = arith.extui %47 : i32 to i64
              %53 = arith.cmpi eq, %11, %c1_i64 : i64
              %54:2 = scf.if %53 -> (i32, i32) {
                %55 = arith.addi %arg13, %47 : i32
                %56 = arith.addi %51, %52 : i64
                %57 = arith.cmpi sge, %56, %c4294967296_i64 : i64
                %58 = arith.extui %57 : i1 to i32
                %59 = arith.addi %arg12, %50 : i32
                %60 = arith.addi %59, %58 : i32
                scf.yield %60, %55 : i32, i32
              } else {
                %55 = arith.muli %arg13, %47 : i32
                %56 = ascsimt.math_binary %arg13, %47, "mulhi" : i32, i32 -> i32
                %57 = arith.cmpi sge, %51, %c2147483648_i64 : i64
                %58 = arith.addi %56, %47 : i32
                %59 = arith.select %57, %58, %56 : i32
                %60 = arith.cmpi sge, %52, %c2147483648_i64 : i64
                %61 = arith.addi %59, %arg13 : i32
                %62 = arith.select %60, %61, %59 : i32
                %63 = arith.muli %arg13, %50 : i32
                %64 = arith.addi %62, %63 : i32
                %65 = arith.muli %arg12, %47 : i32
                %66 = arith.addi %64, %65 : i32
                scf.yield %66, %55 : i32, i32
              }
              scf.yield %54#0, %54#1 : i32, i32
            } else {
              scf.yield %arg12, %arg13 : i32, i32
            }
            scf.yield %42#0, %42#1 : i32, i32
          }
          scf.yield %32#0, %32#1 : i32, i32
        } else {
          scf.yield %20, %17 : i32, i32
        }
        ascsimt.store %arg9[%16], %22#1 : memref<10xi32>, i32
        ascsimt.store %arg9[%19], %22#0 : memref<10xi32>, i32
      } {cannir.vf_threads = 1024 : i64}
    } {threads = 1024 : i64}
    return
  }
}
