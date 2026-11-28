module {
  func.func @_launch_axis0_owner_add_f280142536f4050e4d023c9343167e14(%arg0: !cannir.tensor<<gm, f32>, <<(4930)>, <(1)>>, <<(4930)>, <(1)>>, identity>, %arg1: !cannir.tensor<<gm, i64>, <<(4930)>, <(1)>>, <<(4930)>, <(1)>>, identity>, %arg2: !cannir.tensor<<gm, f32>, <<(4930)>, <(1)>>, <<(4930)>, <(1)>>, identity>, %arg3: !cannir.tensor<<gm, f32>, <<(4930)>, <(1)>>, <<(4930)>, <(1)>>, identity>) attributes {cannbotdsl.contract_digest = "f280142536f4050e4d023c9343167e145632576523e48ed5b63e15557b089e9c", cannbotdsl.public_slot_count = 4 : i64, cannbotdsl.slot_dtype_bits = [32, 64, 32, 32], cannbotdsl.slot_dtype_codes = [2, 0, 2, 2], cannbotdsl.slot_dtype_lanes = [1, 1, 1, 1], cannbotdsl.slot_flags = [1, 1, 1, 1], cannbotdsl.slot_kinds = [1, 1, 1, 1], cannbotdsl.slot_type_digests = ["f32c1771152b19c77ea627a4a56939bd1306d89ca25a34c95b57725269b55a1d", "f9a5bf36342e468b9cfbd78a19fa35d09eb3d13e767b660ad0dc53223776e369", "f32c1771152b19c77ea627a4a56939bd1306d89ca25a34c95b57725269b55a1d", "f32c1771152b19c77ea627a4a56939bd1306d89ca25a34c95b57725269b55a1d"], cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.host} {
    %c20_i64 = arith.constant 20 : i64
    cannir.kernel_launch @kernel__scatter_axis0_owner_add_simt_0(%c20_i64) [%arg0, %arg1, %arg2, %arg3] {block_shape = array<i64: 20, 1, 1>, cannbotdsl.profile = {indices = array<i64: 0, 1, 2, 3>, name = "_scatter_axis0_owner_add_simt", op_type = "_scatter_axis0_owner_add_simt", roles = array<i64: 0, 0, 0, 1>}, simt_kernel = true, thread_dim = 512 : i64, thread_shape = array<i64: 512, 1, 1>} : !cannir.tensor<<gm, f32>, <<(4930)>, <(1)>>, <<(4930)>, <(1)>>, identity>, !cannir.tensor<<gm, i64>, <<(4930)>, <(1)>>, <<(4930)>, <(1)>>, identity>, !cannir.tensor<<gm, f32>, <<(4930)>, <(1)>>, <<(4930)>, <(1)>>, identity>, !cannir.tensor<<gm, f32>, <<(4930)>, <(1)>>, <<(4930)>, <(1)>>, identity>
    return
  }
  func.func @kernel__scatter_axis0_owner_add_simt_0(%arg0: !cannir.tensor<<gm, f32>, <<(4930)>, <(1)>>, <<(4930)>, <(1)>>, identity>, %arg1: !cannir.tensor<<gm, i64>, <<(4930)>, <(1)>>, <<(4930)>, <(1)>>, identity>, %arg2: !cannir.tensor<<gm, f32>, <<(4930)>, <(1)>>, <<(4930)>, <(1)>>, identity>, %arg3: !cannir.tensor<<gm, f32>, <<(4930)>, <(1)>>, <<(4930)>, <(1)>>, identity>) attributes {cannir.buffer.auto_sync, cannir.buffer.sync_analyzed, cannir.buffer.sync_verified, cannir.buffer.tensor_state, cannir.core = "aiv", cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.device, cannir.kernel_group = "kernel__scatter_axis0_owner_add_simt_0", cannir.pure_simt} {
    %alloc = memref.alloc() {cannir.arg_index = 0 : i64} : memref<4930xf32>
    %alloc_0 = memref.alloc() {cannir.arg_index = 1 : i64} : memref<4930xi64>
    %alloc_1 = memref.alloc() {cannir.arg_index = 2 : i64} : memref<4930xf32>
    %alloc_2 = memref.alloc() {cannir.arg_index = 3 : i64} : memref<4930xf32>
    ascsimt.vf(%alloc, %alloc_0, %alloc_1, %alloc_2 : memref<4930xf32>, memref<4930xi64>, memref<4930xf32>, memref<4930xf32>) {
    ^bb0(%arg4: memref<4930xf32>, %arg5: memref<4930xi64>, %arg6: memref<4930xf32>, %arg7: memref<4930xf32>):
      %c4930_i64 = arith.constant 4930 : i64
      %c4437_i32 = arith.constant 4437 : i32
      %c3944_i32 = arith.constant 3944 : i32
      %c3451_i32 = arith.constant 3451 : i32
      %c2958_i32 = arith.constant 2958 : i32
      %c2465_i32 = arith.constant 2465 : i32
      %c1972_i32 = arith.constant 1972 : i32
      %c1479_i32 = arith.constant 1479 : i32
      %c986_i32 = arith.constant 986 : i32
      %c0_i32 = arith.constant 0 : i32
      %c493_i32 = arith.constant 493 : i32
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
      scf.for %arg8 = %9 to %c4930_i64 step %10  : i64 {
        %11 = arith.trunci %arg8 : i64 to i32
        %12 = arith.remsi %11, %c493_i32 : i32
        %13 = arith.cmpi ne, %12, %c0_i32 : i32
        %14 = arith.cmpi slt, %12, %c0_i32 : i32
        %15 = arith.andi %13, %14 : i1
        %16 = arith.divsi %11, %c493_i32 : i32
        %17 = arith.extui %15 : i1 to i32
        %18 = arith.subi %16, %17 : i32
        %19 = arith.select %15, %c493_i32, %c0_i32 : i32
        %20 = arith.addi %12, %19 overflow<nsw> : i32
        %21 = arith.index_cast %arg8 : i64 to index
        %22 = ascsimt.load %arg4[%21] : memref<4930xf32> -> f32
        %23 = arith.index_cast %20 : i32 to index
        %24 = ascsimt.load %arg5[%23] : memref<4930xi64> -> i64
        %25 = arith.extsi %18 : i32 to i64
        %26 = arith.cmpi eq, %24, %25 : i64
        %27 = scf.if %26 -> (f32) {
          %73 = ascsimt.load %arg6[%23] : memref<4930xf32> -> f32
          %74 = arith.addf %22, %73 : f32
          scf.yield %74 : f32
        } else {
          scf.yield %22 : f32
        }
        %28 = arith.addi %20, %c493_i32 overflow<nsw> : i32
        %29 = arith.index_cast %28 : i32 to index
        %30 = ascsimt.load %arg5[%29] : memref<4930xi64> -> i64
        %31 = arith.cmpi eq, %30, %25 : i64
        %32 = scf.if %31 -> (f32) {
          %73 = ascsimt.load %arg6[%29] : memref<4930xf32> -> f32
          %74 = arith.addf %27, %73 : f32
          scf.yield %74 : f32
        } else {
          scf.yield %27 : f32
        }
        %33 = arith.addi %20, %c986_i32 overflow<nsw> : i32
        %34 = arith.index_cast %33 : i32 to index
        %35 = ascsimt.load %arg5[%34] : memref<4930xi64> -> i64
        %36 = arith.cmpi eq, %35, %25 : i64
        %37 = scf.if %36 -> (f32) {
          %73 = ascsimt.load %arg6[%34] : memref<4930xf32> -> f32
          %74 = arith.addf %32, %73 : f32
          scf.yield %74 : f32
        } else {
          scf.yield %32 : f32
        }
        %38 = arith.addi %20, %c1479_i32 overflow<nsw> : i32
        %39 = arith.index_cast %38 : i32 to index
        %40 = ascsimt.load %arg5[%39] : memref<4930xi64> -> i64
        %41 = arith.cmpi eq, %40, %25 : i64
        %42 = scf.if %41 -> (f32) {
          %73 = ascsimt.load %arg6[%39] : memref<4930xf32> -> f32
          %74 = arith.addf %37, %73 : f32
          scf.yield %74 : f32
        } else {
          scf.yield %37 : f32
        }
        %43 = arith.addi %20, %c1972_i32 overflow<nsw> : i32
        %44 = arith.index_cast %43 : i32 to index
        %45 = ascsimt.load %arg5[%44] : memref<4930xi64> -> i64
        %46 = arith.cmpi eq, %45, %25 : i64
        %47 = scf.if %46 -> (f32) {
          %73 = ascsimt.load %arg6[%44] : memref<4930xf32> -> f32
          %74 = arith.addf %42, %73 : f32
          scf.yield %74 : f32
        } else {
          scf.yield %42 : f32
        }
        %48 = arith.addi %20, %c2465_i32 overflow<nsw> : i32
        %49 = arith.index_cast %48 : i32 to index
        %50 = ascsimt.load %arg5[%49] : memref<4930xi64> -> i64
        %51 = arith.cmpi eq, %50, %25 : i64
        %52 = scf.if %51 -> (f32) {
          %73 = ascsimt.load %arg6[%49] : memref<4930xf32> -> f32
          %74 = arith.addf %47, %73 : f32
          scf.yield %74 : f32
        } else {
          scf.yield %47 : f32
        }
        %53 = arith.addi %20, %c2958_i32 overflow<nsw> : i32
        %54 = arith.index_cast %53 : i32 to index
        %55 = ascsimt.load %arg5[%54] : memref<4930xi64> -> i64
        %56 = arith.cmpi eq, %55, %25 : i64
        %57 = scf.if %56 -> (f32) {
          %73 = ascsimt.load %arg6[%54] : memref<4930xf32> -> f32
          %74 = arith.addf %52, %73 : f32
          scf.yield %74 : f32
        } else {
          scf.yield %52 : f32
        }
        %58 = arith.addi %20, %c3451_i32 overflow<nsw> : i32
        %59 = arith.index_cast %58 : i32 to index
        %60 = ascsimt.load %arg5[%59] : memref<4930xi64> -> i64
        %61 = arith.cmpi eq, %60, %25 : i64
        %62 = scf.if %61 -> (f32) {
          %73 = ascsimt.load %arg6[%59] : memref<4930xf32> -> f32
          %74 = arith.addf %57, %73 : f32
          scf.yield %74 : f32
        } else {
          scf.yield %57 : f32
        }
        %63 = arith.addi %20, %c3944_i32 overflow<nsw> : i32
        %64 = arith.index_cast %63 : i32 to index
        %65 = ascsimt.load %arg5[%64] : memref<4930xi64> -> i64
        %66 = arith.cmpi eq, %65, %25 : i64
        %67 = scf.if %66 -> (f32) {
          %73 = ascsimt.load %arg6[%64] : memref<4930xf32> -> f32
          %74 = arith.addf %62, %73 : f32
          scf.yield %74 : f32
        } else {
          scf.yield %62 : f32
        }
        %68 = arith.addi %20, %c4437_i32 overflow<nsw> : i32
        %69 = arith.index_cast %68 : i32 to index
        %70 = ascsimt.load %arg5[%69] : memref<4930xi64> -> i64
        %71 = arith.cmpi eq, %70, %25 : i64
        %72 = scf.if %71 -> (f32) {
          %73 = ascsimt.load %arg6[%69] : memref<4930xf32> -> f32
          %74 = arith.addf %67, %73 : f32
          scf.yield %74 : f32
        } else {
          scf.yield %67 : f32
        }
        ascsimt.store %arg7[%21], %72 : memref<4930xf32>, f32
      } {cannir.vf_threads = 1024 : i64}
    } {threads = 1024 : i64}
    return
  }
}
