module {
  func.func @run_cc31a78c01678f91dce0a2f7135b693a(%arg0: !cannir.tensor<<gm, i32>, <<(86016)>, <(1)>>, <<(86016)>, <(1)>>, identity>, %arg1: !cannir.tensor<<gm, i32>, <<(86016)>, <(1)>>, <<(86016)>, <(1)>>, identity>, %arg2: !cannir.tensor<<gm, i64>, <<(65184)>, <(1)>>, <<(65184)>, <(1)>>, identity>, %arg3: !cannir.tensor<<gm, i32>, <<(65184)>, <(1)>>, <<(65184)>, <(1)>>, identity>, %arg4: !cannir.tensor<<gm, i32>, <<(86016)>, <(1)>>, <<(86016)>, <(1)>>, identity>) attributes {cannbotdsl.contract_digest = "cc31a78c01678f91dce0a2f7135b693acef4ce4757c321303e076a7b22a79ff9", cannbotdsl.public_slot_count = 5 : i64, cannbotdsl.slot_dtype_bits = [32, 32, 64, 32, 32], cannbotdsl.slot_dtype_codes = [0, 0, 0, 0, 0], cannbotdsl.slot_dtype_lanes = [1, 1, 1, 1, 1], cannbotdsl.slot_flags = [1, 1, 1, 1, 1], cannbotdsl.slot_kinds = [1, 1, 1, 1, 1], cannbotdsl.slot_type_digests = ["cb2b9ed0cb2aa87cc39e29d193a0b64d1c3bb6aa4fca075e4424b454d2d844b0", "cb2b9ed0cb2aa87cc39e29d193a0b64d1c3bb6aa4fca075e4424b454d2d844b0", "96cb92bcd5917e70da3b25f218339b4a4be829eda7b93509b35c26711fb8851d", "61ae37a45cbf031052315bef5db61cf4391b8e999069d09be9a18e8e27cc2544", "cb2b9ed0cb2aa87cc39e29d193a0b64d1c3bb6aa4fca075e4424b454d2d844b0"], cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.host} {
    %c64_i64 = arith.constant 64 : i64
    cannir.kernel_launch @kernel_ScatterNativeKernel___call___0(%c64_i64) [%arg0, %arg1, %arg2, %arg3, %arg4] {cannbotdsl.profile = {indices = array<i64: 0, 1, 2, 3, 4>, name = "ScatterNativeKernel___call__", op_type = "ScatterNativeKernel___call__", roles = array<i64: 0, 1, 0, 0, 1>}} : !cannir.tensor<<gm, i32>, <<(86016)>, <(1)>>, <<(86016)>, <(1)>>, identity>, !cannir.tensor<<gm, i32>, <<(86016)>, <(1)>>, <<(86016)>, <(1)>>, identity>, !cannir.tensor<<gm, i64>, <<(65184)>, <(1)>>, <<(65184)>, <(1)>>, identity>, !cannir.tensor<<gm, i32>, <<(65184)>, <(1)>>, <<(65184)>, <(1)>>, identity>, !cannir.tensor<<gm, i32>, <<(86016)>, <(1)>>, <<(86016)>, <(1)>>, identity>
    return
  }
  func.func @kernel_ScatterNativeKernel___call___0(%arg0: !cannir.tensor<<gm, i32>, <<(86016)>, <(1)>>, <<(86016)>, <(1)>>, identity>, %arg1: !cannir.tensor<<gm, i32>, <<(86016)>, <(1)>>, <<(86016)>, <(1)>>, identity>, %arg2: !cannir.tensor<<gm, i64>, <<(65184)>, <(1)>>, <<(65184)>, <(1)>>, identity>, %arg3: !cannir.tensor<<gm, i32>, <<(65184)>, <(1)>>, <<(65184)>, <(1)>>, identity>, %arg4: !cannir.tensor<<gm, i32>, <<(86016)>, <(1)>>, <<(86016)>, <(1)>>, identity>) attributes {cannir.buffer.auto_sync, cannir.buffer.sync_analyzed, cannir.buffer.sync_verified, cannir.buffer.tensor_state, cannir.core = "aiv", cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.device, cannir.explicit_ring_buffer_cursors, cannir.kernel_group = "kernel_ScatterNativeKernel___call___0", cannir.resource.ids_prepared} {
    %c4096_i32 = arith.constant 4096 : i32
    %c0 = arith.constant 0 : index
    %c84_i64 = arith.constant 84 : i64
    %c64_i64 = arith.constant 64 : i64
    %c0_i32 = arith.constant 0 : i32
    %c4096_i64 = arith.constant 4096 : i64
    %c1_i64 = arith.constant 1 : i64
    %c1_i32 = arith.constant 1 : i32
    %c32768_i64 = arith.constant 32768 : i64
    %c0_i64 = arith.constant 0 : i64
    %c8_i64 = arith.constant 8 : i64
    %alloc = memref.alloc() {cannir.arg_index = 0 : i64} : memref<86016xi32>
    %alloc_0 = memref.alloc() {cannir.arg_index = 1 : i64} : memref<86016xi32>
    %alloc_1 = memref.alloc() {cannir.arg_index = 2 : i64} : memref<65184xi64>
    %alloc_2 = memref.alloc() {cannir.arg_index = 3 : i64} : memref<65184xi32>
    %alloc_3 = memref.alloc() {cannir.arg_index = 4 : i64} : memref<86016xi32>
    %alloc_4 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<1024xi32, 1>
    %0 = ascscalar.get_block_idx
    scf.for %arg5 = %0 to %c84_i64 step %c64_i64  : i64 {
      %1 = ascscalar.bitcast_memref %alloc_4 byte_offset = %c0_i64 : memref<1024xi32, 1> -> memref<?xi32, 1>
      %2 = arith.muli %arg5, %c4096_i64 : i64
      ascvec.get_buf mte2, %c0_i64, 0 {cannir.buffer.generated_sync}
      "ascvec.copy_gm2ub_plan"(%1, %alloc, %c0_i32, %c0_i32, %c0_i64, %2, %c1_i64, %c4096_i64, %c0_i64, %c4096_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 344064 : i64, pad_storage_bits = false, storage_dtype = i32, ub_base = 0 : i64, ub_capacity = 4096 : i64}> : (memref<?xi32, 1>, memref<86016xi32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
      ascvec.release_buf mte2, %c0_i64, 0 {cannir.buffer.generated_sync}
      %3 = arith.muli %arg5, %c32768_i64 : i64
      %4 = ascscalar.memref_address %alloc_0 : memref<86016xi32>
      %5 = arith.divui %3, %c8_i64 : i64
      %6 = arith.addi %4, %5 : i64
      %7 = ascscalar.bind_gm_ptr %6 : memref<1xi32>
      ascvec.get_buf mte3, %c0_i64, 0 {cannir.buffer.generated_sync}
      ascvec.copy_ub2gm %7[%c0], %1[%c0], %c1_i32, %c4096_i32, %c0_i64, %c4096_i32 l2_cache_ctl %c0_i32 : memref<1xi32>, memref<?xi32, 1>
      ascvec.release_buf mte3, %c0_i64, 0 {cannir.buffer.generated_sync}
    }
    ascvec.sync_pipe mte3
    ascvec.sync_block_arrive mte3, %c1_i64 {mode = 0 : i32}
    ascvec.sync_block_wait s, %c1_i64 {mode = 0 : i32}
    ascsimt.vf(%alloc_1, %alloc_2, %alloc_3 : memref<65184xi64>, memref<65184xi32>, memref<86016xi32>) {
    ^bb0(%arg5: memref<65184xi64>, %arg6: memref<65184xi32>, %arg7: memref<86016xi32>):
      %c32768_i32 = arith.constant 32768 : i32
      %c65184_i32 = arith.constant 65184 : i32
      %c128_i32 = arith.constant 128 : i32
      %c0_i32_5 = arith.constant 0 : i32
      %c97_i32 = arith.constant 97 : i32
      %false = arith.constant false
      %c128_i64 = arith.constant 128 : i64
      %c0_i64_6 = arith.constant 0 : i64
      %c512_i32 = arith.constant 512 : i32
      %1 = ascsimt.thread_idx {ascsimt.scope = 1 : i64, axis = "x", cannir.vf_threads = 512 : i64} : i32
      %2 = ascsimt.block_idx {ascsimt.scope = 1 : i64, axis = "x", cannir.vf_threads = 512 : i64} : i32
      %3 = arith.muli %2, %c512_i32 {cannir.vf_threads = 512 : i64} : i32
      %4 = arith.addi %3, %1 {cannir.vf_threads = 512 : i64} : i32
      scf.for %arg8 = %4 to %c65184_i32 step %c32768_i32  : i32 {
        %5 = arith.index_cast %arg8 : i32 to index
        %6 = ascsimt.load %arg5[%5] : memref<65184xi64> -> i64
        %7 = arith.cmpi sge, %6, %c0_i64_6 : i64
        %8 = arith.cmpi slt, %6, %c128_i64 : i64
        %9 = arith.select %7, %8, %false : i1
        scf.if %9 {
          %10 = arith.trunci %6 : i64 to i32
          %11 = arith.remsi %arg8, %c97_i32 : i32
          %12 = arith.cmpi ne, %11, %c0_i32_5 : i32
          %13 = arith.cmpi slt, %11, %c0_i32_5 : i32
          %14 = arith.andi %12, %13 : i1
          %15 = arith.divsi %arg8, %c97_i32 : i32
          %16 = arith.extui %14 : i1 to i32
          %17 = arith.subi %15, %16 : i32
          %18 = arith.muli %17, %c128_i32 : i32
          %19 = arith.addi %18, %10 : i32
          %20 = ascsimt.load %arg6[%5] : memref<65184xi32> -> i32
          %21 = arith.index_cast %19 : i32 to index
          ascsimt.store %arg7[%21], %20 : memref<86016xi32>, i32
        }
      } {cannir.vf_threads = 512 : i64}
    } {threads = 512 : i64}
    return
  }
}
