module {
  func.func @run_3de7f2d85ed97514485173cd1d6e60a4(%arg0: !cannir.tensor<<gm, i8>, <<(1683)>, <(1)>>, <<(1683)>, <(1)>>, identity>, %arg1: !cannir.tensor<<gm, i64>, <<(50)>, <(1)>>, <<(50)>, <(1)>>, identity>, %arg2: !cannir.tensor<<gm, i8>, <<(4950)>, <(1)>>, <<(4950)>, <(1)>>, identity>) attributes {cannbotdsl.contract_digest = "3de7f2d85ed97514485173cd1d6e60a4e4254f83c1cfac3420830158d7ae85da", cannbotdsl.public_slot_count = 3 : i64, cannbotdsl.slot_dtype_bits = [8, 64, 8], cannbotdsl.slot_dtype_codes = [0, 0, 0], cannbotdsl.slot_dtype_lanes = [1, 1, 1], cannbotdsl.slot_flags = [1, 1, 1], cannbotdsl.slot_kinds = [1, 1, 1], cannbotdsl.slot_type_digests = ["34ee4e660eccbbb596e84de433eff3901d3fe698505a5d5eefe39210fa1f91a3", "07039dc4bf407039b13cb2a8bb50761434ee2dc6b6a1970baa80efe1297e7157", "f6850fb65444794b3cc1d1b0d3f64382098185748777380a15ce1980c3ed2390"], cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.host} {
    %c52_i64 = arith.constant 52 : i64
    cannir.kernel_launch @kernel_GatherV2Kernel___call___0(%c52_i64) [%arg0, %arg1, %arg2] {block_shape = array<i64: 52, 1, 1>, cannbotdsl.profile = {indices = array<i64: 0, 1, 2>, name = "GatherV2Kernel___call__", op_type = "GatherV2Kernel___call__", roles = array<i64: 0, 0, 1>}, simt_kernel = true, thread_dim = 256 : i64, thread_shape = array<i64: 256, 1, 1>} : !cannir.tensor<<gm, i8>, <<(1683)>, <(1)>>, <<(1683)>, <(1)>>, identity>, !cannir.tensor<<gm, i64>, <<(50)>, <(1)>>, <<(50)>, <(1)>>, identity>, !cannir.tensor<<gm, i8>, <<(4950)>, <(1)>>, <<(4950)>, <(1)>>, identity>
    return
  }
  func.func @kernel_GatherV2Kernel___call___0(%arg0: !cannir.tensor<<gm, i8>, <<(1683)>, <(1)>>, <<(1683)>, <(1)>>, identity>, %arg1: !cannir.tensor<<gm, i64>, <<(50)>, <(1)>>, <<(50)>, <(1)>>, identity>, %arg2: !cannir.tensor<<gm, i8>, <<(4950)>, <(1)>>, <<(4950)>, <(1)>>, identity>) attributes {cannir.buffer.auto_sync, cannir.buffer.sync_analyzed, cannir.buffer.sync_verified, cannir.buffer.tensor_state, cannir.core = "aiv", cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.device, cannir.kernel_group = "kernel_GatherV2Kernel___call___0", cannir.pure_simt} {
    %alloc = memref.alloc() {cannir.arg_index = 0 : i64} : memref<1683xi8>
    %alloc_0 = memref.alloc() {cannir.arg_index = 1 : i64} : memref<50xi64>
    %alloc_1 = memref.alloc() {cannir.arg_index = 2 : i64} : memref<4950xi8>
    ascsimt.vf(%alloc_0, %alloc, %alloc_1 : memref<50xi64>, memref<1683xi8>, memref<4950xi8>) {
    ^bb0(%arg3: memref<50xi64>, %arg4: memref<1683xi8>, %arg5: memref<4950xi8>):
      %c0_i8 = arith.constant 0 : i8
      %c17_i32 = arith.constant 17 : i32
      %c17_i64 = arith.constant 17 : i64
      %c0_i64 = arith.constant 0 : i64
      %c50_i32 = arith.constant 50 : i32
      %c0_i32 = arith.constant 0 : i32
      %c99_i32 = arith.constant 99 : i32
      %c4950_i32 = arith.constant 4950 : i32
      %c96_i32 = arith.constant 96 : i32
      %0 = ascsimt.block_idx {ascsimt.scope = 1 : i64, axis = "x", cannir.vf_threads = 1024 : i64} : i32
      %1 = ascsimt.thread_idx {ascsimt.scope = 1 : i64, axis = "x", cannir.vf_threads = 1024 : i64} : i32
      %2 = ascsimt.thread_dim {ascsimt.scope = 1 : i64, axis = "x", cannir.vf_threads = 1024 : i64} : i32
      %3 = arith.muli %0, %c96_i32 {cannir.vf_threads = 1024 : i64} : i32
      scf.for %arg6 = %1 to %c96_i32 step %2  : i32 {
        %4 = arith.addi %3, %arg6 : i32
        %5 = arith.cmpi slt, %4, %c4950_i32 : i32
        scf.if %5 {
          %6 = arith.remsi %4, %c99_i32 : i32
          %7 = arith.cmpi ne, %6, %c0_i32 : i32
          %8 = arith.cmpi slt, %6, %c0_i32 : i32
          %9 = arith.andi %7, %8 : i1
          %10 = arith.divsi %4, %c99_i32 : i32
          %11 = arith.extui %9 : i1 to i32
          %12 = arith.subi %10, %11 : i32
          %13 = arith.muli %12, %c99_i32 : i32
          %14 = arith.subi %4, %13 : i32
          %15 = arith.remsi %12, %c50_i32 : i32
          %16 = arith.cmpi ne, %15, %c0_i32 : i32
          %17 = arith.cmpi slt, %15, %c0_i32 : i32
          %18 = arith.andi %16, %17 : i1
          %19 = arith.divsi %12, %c50_i32 : i32
          %20 = arith.extui %18 : i1 to i32
          %21 = arith.subi %19, %20 : i32
          %22 = arith.index_cast %12 : i32 to index
          %23 = ascsimt.load %arg3[%22] : memref<50xi64> -> i64
          %24 = arith.cmpi slt, %23, %c0_i64 : i64
          %25 = arith.addi %23, %c17_i64 : i64
          %26 = arith.select %24, %25, %23 : i64
          %27 = arith.cmpi sge, %26, %c0_i64 : i64
          scf.if %27 {
            %28 = arith.cmpi slt, %26, %c17_i64 : i64
            scf.if %28 {
              %29 = arith.trunci %26 : i64 to i32
              %30 = arith.muli %21, %c17_i32 : i32
              %31 = arith.addi %30, %29 : i32
              %32 = arith.muli %31, %c99_i32 : i32
              %33 = arith.addi %32, %14 : i32
              %34 = arith.index_cast %33 : i32 to index
              %35 = ascsimt.load %arg4[%34] : memref<1683xi8> -> i8
              %36 = arith.index_cast %4 : i32 to index
              ascsimt.store %arg5[%36], %35 : memref<4950xi8>, i8
            } else {
              %29 = arith.index_cast %4 : i32 to index
              ascsimt.store %arg5[%29], %c0_i8 : memref<4950xi8>, i8
            }
          } else {
            %28 = arith.index_cast %4 : i32 to index
            ascsimt.store %arg5[%28], %c0_i8 : memref<4950xi8>, i8
          }
        }
      } {cannir.vf_threads = 1024 : i64}
    } {threads = 1024 : i64}
    return
  }
}
