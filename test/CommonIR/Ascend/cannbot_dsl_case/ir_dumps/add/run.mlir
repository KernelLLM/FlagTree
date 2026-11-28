module {
  func.func @run_2d982d51f40eb62103cdc4c38e99eb36(%arg0: !cannir.tensor<<gm, f32>, <<(1)>, <(1)>>, <<(1)>, <(1)>>, identity>, %arg1: !cannir.tensor<<gm, f32>, <<(1)>, <(1)>>, <<(1)>, <(1)>>, identity>, %arg2: !cannir.tensor<<gm, f32>, <<(1)>, <(1)>>, <<(1)>, <(1)>>, identity>) attributes {cannbotdsl.contract_digest = "2d982d51f40eb62103cdc4c38e99eb36ae048d745d6819e5a98dedf64e9c2235", cannbotdsl.public_slot_count = 3 : i64, cannbotdsl.slot_dtype_bits = [32, 32, 32], cannbotdsl.slot_dtype_codes = [2, 2, 2], cannbotdsl.slot_dtype_lanes = [1, 1, 1], cannbotdsl.slot_flags = [1, 1, 1], cannbotdsl.slot_kinds = [1, 1, 1], cannbotdsl.slot_type_digests = ["b134b54071cdca5322ca9f5d39ee9881e5a8f90b566373e7bf45d235577c8f89", "b134b54071cdca5322ca9f5d39ee9881e5a8f90b566373e7bf45d235577c8f89", "b134b54071cdca5322ca9f5d39ee9881e5a8f90b566373e7bf45d235577c8f89"], cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.host} {
    %c1_i64 = arith.constant 1 : i64
    cannir.kernel_launch @kernel_AddDmaKernel___call___0(%c1_i64) [%arg0, %arg1, %arg2] {cannbotdsl.profile = {indices = array<i64: 0, 1, 2>, name = "AddDmaKernel___call__", op_type = "AddDmaKernel___call__", roles = array<i64: 0, 0, 1>}} : !cannir.tensor<<gm, f32>, <<(1)>, <(1)>>, <<(1)>, <(1)>>, identity>, !cannir.tensor<<gm, f32>, <<(1)>, <(1)>>, <<(1)>, <(1)>>, identity>, !cannir.tensor<<gm, f32>, <<(1)>, <(1)>>, <<(1)>, <(1)>>, identity>
    return
  }
  func.func @kernel_AddDmaKernel___call___0(%arg0: !cannir.tensor<<gm, f32>, <<(1)>, <(1)>>, <<(1)>, <(1)>>, identity>, %arg1: !cannir.tensor<<gm, f32>, <<(1)>, <(1)>>, <<(1)>, <(1)>>, identity>, %arg2: !cannir.tensor<<gm, f32>, <<(1)>, <(1)>>, <<(1)>, <(1)>>, identity>) attributes {cannir.buffer.auto_sync, cannir.buffer.sync_analyzed, cannir.buffer.sync_verified, cannir.buffer.tensor_state, cannir.core = "aiv", cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.device, cannir.explicit_ring_buffer_cursors, cannir.kernel_group = "kernel_AddDmaKernel___call___0", cannir.resource.ids_prepared} {
    %c4_i32 = arith.constant 4 : i32
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c32_i64 = arith.constant 32 : i64
    %false = arith.constant false
    %c1_i32 = arith.constant 1 : i32
    %c5_i64 = arith.constant 5 : i64
    %c4_i64 = arith.constant 4 : i64
    %c3_i64 = arith.constant 3 : i64
    %c2_i64 = arith.constant 2 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %alloc = memref.alloc() {cannir.arg_index = 0 : i64} : memref<1xf32>
    %alloc_0 = memref.alloc() {cannir.arg_index = 1 : i64} : memref<1xf32>
    %alloc_1 = memref.alloc() {cannir.arg_index = 2 : i64} : memref<1xf32>
    %alloc_2 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<8xf32, 1>
    %alloc_3 = memref.alloc() {cannir.byte_offset = 32 : i64} : memref<8xf32, 1>
    %alloc_4 = memref.alloc() {cannir.byte_offset = 64 : i64} : memref<8xf32, 1>
    %alloc_5 = memref.alloc() {cannir.byte_offset = 96 : i64} : memref<8xf32, 1>
    %alloc_6 = memref.alloc() {cannir.byte_offset = 128 : i64} : memref<8xf32, 1>
    %alloc_7 = memref.alloc() {cannir.byte_offset = 160 : i64} : memref<8xf32, 1>
    %0:3 = scf.for %arg3 = %c0_i64 to %c3_i64 step %c1_i64 iter_args(%arg4 = %c0_i64, %arg5 = %c0_i64, %arg6 = %c0_i64) -> (i64, i64, i64)  : i64 {
      %1 = arith.subi %arg3, %c1_i64 overflow<nsw> : i64
      %2 = arith.subi %arg3, %c2_i64 overflow<nsw> : i64
      %3 = arith.cmpi slt, %arg3, %c1_i64 : i64
      %4 = scf.if %3 -> (i64) {
        %11 = arith.cmpi eq, %arg4, %c0_i64 : i64
        %12 = ascscalar.memref_address %alloc_2 : memref<8xf32, 1>
        %13 = ascscalar.memref_address %alloc_3 : memref<8xf32, 1>
        %14 = arith.select %11, %12, %13 : i64
        %15 = arith.cmpi ne, %arg4, %c0_i64 : i64
        %16 = arith.extui %15 : i1 to i64
        %17 = arith.addi %arg4, %c1_i64 overflow<nsw> : i64
        %18 = arith.remui %17, %c2_i64 : i64
        %19 = ascscalar.memref_address %alloc_4 : memref<8xf32, 1>
        %20 = ascscalar.memref_address %alloc_5 : memref<8xf32, 1>
        %21 = arith.select %11, %19, %20 : i64
        %22 = arith.select %11, %c2_i64, %c3_i64 : i64
        %23 = ascscalar.bind_local_ptr %14 : memref<1xf32, 1>
        ascvec.get_buf mte2, %16, 0 {cannir.buffer.generated_sync}
        "ascvec.copy_gm2ub_plan"(%23, %alloc, %c0_i32, %c0_i32, %c0_i64, %c0_i64, %c1_i64, %c4_i64, %c0_i64, %c4_i64, %c0_i64, %c0_i64, %c1_i64, %c4_i64, %c32_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 4 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = -1 : i64, ub_capacity = 4 : i64}> : (memref<1xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
        ascvec.release_buf mte2, %16, 0 {cannir.buffer.generated_sync}
        %24 = ascscalar.bind_local_ptr %21 : memref<1xf32, 1>
        ascvec.get_buf mte2, %22, 0 {cannir.buffer.generated_sync}
        "ascvec.copy_gm2ub_plan"(%24, %alloc_0, %c0_i32, %c0_i32, %c0_i64, %c0_i64, %c1_i64, %c4_i64, %c0_i64, %c4_i64, %c0_i64, %c0_i64, %c1_i64, %c4_i64, %c32_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 4 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = -1 : i64, ub_capacity = 4 : i64}> : (memref<1xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
        ascvec.release_buf mte2, %22, 0 {cannir.buffer.generated_sync}
        scf.yield %18 : i64
      } else {
        scf.yield %arg4 : i64
      }
      %5 = arith.cmpi sge, %1, %c0_i64 : i64
      %6 = arith.cmpi slt, %1, %c1_i64 : i64
      %7 = arith.select %5, %6, %false : i1
      %8 = scf.if %7 -> (i64) {
        %11 = arith.cmpi eq, %arg5, %c0_i64 : i64
        %12 = ascscalar.memref_address %alloc_2 : memref<8xf32, 1>
        %13 = ascscalar.memref_address %alloc_3 : memref<8xf32, 1>
        %14 = arith.select %11, %12, %13 : i64
        %15 = ascscalar.bind_local_ptr %14 : memref<8xf32, 1>
        %16 = arith.cmpi ne, %arg5, %c0_i64 : i64
        %17 = arith.extui %16 : i1 to i64
        %18 = arith.addi %arg5, %c1_i64 overflow<nsw> : i64
        %19 = arith.remui %18, %c2_i64 : i64
        %20 = ascscalar.memref_address %alloc_4 : memref<8xf32, 1>
        %21 = ascscalar.memref_address %alloc_5 : memref<8xf32, 1>
        %22 = arith.select %11, %20, %21 : i64
        %23 = ascscalar.bind_local_ptr %22 : memref<8xf32, 1>
        %24 = arith.select %11, %c2_i64, %c3_i64 : i64
        %25 = ascscalar.memref_address %alloc_6 : memref<8xf32, 1>
        %26 = ascscalar.memref_address %alloc_7 : memref<8xf32, 1>
        %27 = arith.select %11, %25, %26 : i64
        %28 = ascscalar.bind_local_ptr %27 : memref<8xf32, 1>
        %29 = arith.select %11, %c4_i64, %c5_i64 : i64
        ascvec.get_buf v, %17, 0 {cannir.buffer.generated_sync}
        ascvec.get_buf v, %24, 0 {cannir.buffer.generated_sync}
        ascvec.get_buf v, %29, 0 {cannir.buffer.generated_sync}
        ascsimt.vf(%15, %23, %28 : memref<8xf32, 1>, memref<8xf32, 1>, memref<8xf32, 1>) {
        ^bb0(%arg7: memref<8xf32, 1>, %arg8: memref<8xf32, 1>, %arg9: memref<8xf32, 1>):
          %c1024_i32 = arith.constant 1024 : i32
          %c1_i32_8 = arith.constant 1 : i32
          %30 = ascsimt.thread_idx {ascsimt.scope = 1 : i64, axis = "x", cannir.vf_threads = 1024 : i64} : i32
          scf.for %arg10 = %30 to %c1_i32_8 step %c1024_i32  : i32 {
            %31 = arith.index_cast %arg10 : i32 to index
            %32 = ascsimt.load %arg7[%31] : memref<8xf32, 1> -> f32
            %33 = ascsimt.load %arg8[%31] : memref<8xf32, 1> -> f32
            %34 = arith.addf %32, %33 : f32
            ascsimt.store %arg9[%31], %34 : memref<8xf32, 1>, f32
          } {cannir.vf_threads = 1024 : i64}
        } {threads = 1024 : i64}
        ascvec.release_buf v, %29, 0 {cannir.buffer.generated_sync}
        ascvec.release_buf v, %24, 0 {cannir.buffer.generated_sync}
        ascvec.release_buf v, %17, 0 {cannir.buffer.generated_sync}
        scf.yield %19 : i64
      } else {
        scf.yield %arg5 : i64
      }
      %9 = arith.cmpi sge, %2, %c0_i64 : i64
      %10 = scf.if %9 -> (i64) {
        %11 = arith.cmpi eq, %arg6, %c0_i64 : i64
        %12 = ascscalar.memref_address %alloc_6 : memref<8xf32, 1>
        %13 = ascscalar.memref_address %alloc_7 : memref<8xf32, 1>
        %14 = arith.select %11, %12, %13 : i64
        %15 = arith.select %11, %c4_i64, %c5_i64 : i64
        %16 = arith.addi %arg6, %c1_i64 overflow<nsw> : i64
        %17 = arith.remui %16, %c2_i64 : i64
        %18 = ascscalar.bind_local_ptr %14 : memref<1xf32, 1>
        ascvec.get_buf mte3, %15, 0 {cannir.buffer.generated_sync}
        ascvec.copy_ub2gm %alloc_1[%c0], %18[%c0], %c1_i32, %c4_i32, %c0_i64, %c4_i32 l2_cache_ctl %c0_i32 : memref<1xf32>, memref<1xf32, 1>
        ascvec.release_buf mte3, %15, 0 {cannir.buffer.generated_sync}
        scf.yield %17 : i64
      } else {
        scf.yield %arg6 : i64
      }
      scf.yield %4, %8, %10 : i64, i64, i64
    }
    return
  }
}
