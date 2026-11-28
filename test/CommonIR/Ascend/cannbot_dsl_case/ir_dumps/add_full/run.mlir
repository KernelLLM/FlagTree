module {
  func.func @run_892069a3456b1ff021f797191413011e(%arg0: !cannir.tensor<<gm, i32>, <<(131079)>, <(1)>>, <<(131079)>, <(1)>>, identity>, %arg1: !cannir.tensor<<gm, i32>, <<(131079)>, <(1)>>, <<(131079)>, <(1)>>, identity>, %arg2: !cannir.tensor<<gm, i32>, <<(131079)>, <(1)>>, <<(131079)>, <(1)>>, identity>) attributes {cannbotdsl.contract_digest = "892069a3456b1ff021f797191413011ee4b47caec5c98f04f65a9be385b9c3b7", cannbotdsl.public_slot_count = 3 : i64, cannbotdsl.slot_dtype_bits = [32, 32, 32], cannbotdsl.slot_dtype_codes = [0, 0, 0], cannbotdsl.slot_dtype_lanes = [1, 1, 1], cannbotdsl.slot_flags = [1, 1, 1], cannbotdsl.slot_kinds = [1, 1, 1], cannbotdsl.slot_type_digests = ["4edd7c6079918de914f15daa2184e7eca23ba28dde094ac9df83611120b28fe6", "4edd7c6079918de914f15daa2184e7eca23ba28dde094ac9df83611120b28fe6", "4edd7c6079918de914f15daa2184e7eca23ba28dde094ac9df83611120b28fe6"], cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.host} {
    %c2_i64 = arith.constant 2 : i64
    cannir.kernel_launch @kernel_AddDmaKernel___call___0(%c2_i64) [%arg0, %arg1, %arg2] {cannbotdsl.profile = {indices = array<i64: 0, 1, 2>, name = "AddDmaKernel___call__", op_type = "AddDmaKernel___call__", roles = array<i64: 0, 0, 1>}} : !cannir.tensor<<gm, i32>, <<(131079)>, <(1)>>, <<(131079)>, <(1)>>, identity>, !cannir.tensor<<gm, i32>, <<(131079)>, <(1)>>, <<(131079)>, <(1)>>, identity>, !cannir.tensor<<gm, i32>, <<(131079)>, <(1)>>, <<(131079)>, <(1)>>, identity>
    return
  }
  func.func @kernel_AddDmaKernel___call___0(%arg0: !cannir.tensor<<gm, i32>, <<(131079)>, <(1)>>, <<(131079)>, <(1)>>, identity>, %arg1: !cannir.tensor<<gm, i32>, <<(131079)>, <(1)>>, <<(131079)>, <(1)>>, identity>, %arg2: !cannir.tensor<<gm, i32>, <<(131079)>, <(1)>>, <<(131079)>, <(1)>>, identity>) attributes {cannir.buffer.auto_sync, cannir.buffer.sync_analyzed, cannir.buffer.sync_verified, cannir.buffer.tensor_state, cannir.core = "aiv", cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.device, cannir.explicit_ring_buffer_cursors, cannir.kernel_group = "kernel_AddDmaKernel___call___0", cannir.resource.ids_prepared} {
    %c36864_i32 = arith.constant 36864 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0 = arith.constant 0 : index
    %c8192_i64 = arith.constant 8192 : i64
    %c8_i64 = arith.constant 8 : i64
    %c8193_i64 = arith.constant 8193 : i64
    %c131079_i64 = arith.constant 131079 : i64
    %c10_i64 = arith.constant 10 : i64
    %c9216_i64 = arith.constant 9216 : i64
    %c0_i32 = arith.constant 0 : i32
    %false = arith.constant false
    %c32_i64 = arith.constant 32 : i64
    %c4 = arith.constant 4 : index
    %c524316_i64 = arith.constant 524316 : i64
    %c5_i64 = arith.constant 5 : i64
    %c4_i64 = arith.constant 4 : i64
    %c3_i64 = arith.constant 3 : i64
    %c2_i64 = arith.constant 2 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %alloc = memref.alloc() {cannir.arg_index = 0 : i64} : memref<131079xi32>
    %alloc_0 = memref.alloc() {cannir.arg_index = 1 : i64} : memref<131079xi32>
    %alloc_1 = memref.alloc() {cannir.arg_index = 2 : i64} : memref<131079xi32>
    %alloc_2 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<9216xi32, 1>
    %alloc_3 = memref.alloc() {cannir.byte_offset = 36864 : i64} : memref<9216xi32, 1>
    %alloc_4 = memref.alloc() {cannir.byte_offset = 73728 : i64} : memref<9216xi32, 1>
    %alloc_5 = memref.alloc() {cannir.byte_offset = 110592 : i64} : memref<9216xi32, 1>
    %alloc_6 = memref.alloc() {cannir.byte_offset = 147456 : i64} : memref<9216xi32, 1>
    %alloc_7 = memref.alloc() {cannir.byte_offset = 184320 : i64} : memref<9216xi32, 1>
    %0 = ascscalar.get_block_idx
    %1 = arith.muli %0, %c8192_i64 : i64
    %2 = arith.minsi %0, %c1_i64 : i64
    %3 = arith.addi %1, %2 : i64
    %4 = arith.muli %3, %c8_i64 : i64
    %5 = arith.cmpi slt, %0, %c1_i64 : i64
    %6 = arith.select %5, %c8193_i64, %c8192_i64 : i64
    %7 = arith.muli %6, %c8_i64 overflow<nsw> : i64
    %8 = arith.subi %c131079_i64, %4 : i64
    %9 = arith.minsi %7, %8 : i64
    %10:3 = scf.for %arg3 = %c0_i64 to %c10_i64 step %c1_i64 iter_args(%arg4 = %c0_i64, %arg5 = %c0_i64, %arg6 = %c0_i64) -> (i64, i64, i64)  : i64 {
      %11 = arith.subi %arg3, %c1_i64 overflow<nsw> : i64
      %12 = arith.subi %arg3, %c2_i64 overflow<nsw> : i64
      %13 = arith.cmpi slt, %arg3, %c8_i64 : i64
      %14 = scf.if %13 -> (i64) {
        %21 = arith.muli %arg3, %c9216_i64 overflow<nsw> : i64
        %22 = arith.addi %4, %21 : i64
        %23 = arith.subi %9, %21 : i64
        %24 = arith.minsi %23, %c9216_i64 : i64
        %25 = arith.cmpi eq, %arg4, %c0_i64 : i64
        %26 = ascscalar.memref_address %alloc_2 : memref<9216xi32, 1>
        %27 = ascscalar.memref_address %alloc_3 : memref<9216xi32, 1>
        %28 = arith.select %25, %26, %27 : i64
        %29 = arith.cmpi ne, %arg4, %c0_i64 : i64
        %30 = arith.extui %29 : i1 to i64
        %31 = arith.addi %arg4, %c1_i64 overflow<nsw> : i64
        %32 = arith.remui %31, %c2_i64 : i64
        %33 = ascscalar.memref_address %alloc_4 : memref<9216xi32, 1>
        %34 = ascscalar.memref_address %alloc_5 : memref<9216xi32, 1>
        %35 = arith.select %25, %33, %34 : i64
        %36 = arith.select %25, %c2_i64, %c3_i64 : i64
        %37 = arith.muli %22, %c4_i64 : i64
        %38 = arith.muli %24, %c4_i64 : i64
        %39 = ascscalar.bind_local_ptr %28 : memref<1xi32, 1>
        ascvec.get_buf mte2, %30, 0 {cannir.buffer.generated_sync}
        "ascvec.copy_gm2ub_plan"(%39, %alloc, %c0_i32, %c0_i32, %c0_i64, %37, %c1_i64, %38, %c0_i64, %38, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 524316 : i64, pad_storage_bits = false, storage_dtype = i32, ub_base = -1 : i64, ub_capacity = 4 : i64}> : (memref<1xi32, 1>, memref<131079xi32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
        ascvec.release_buf mte2, %30, 0 {cannir.buffer.generated_sync}
        %40 = ascscalar.bind_local_ptr %35 : memref<1xi32, 1>
        ascvec.get_buf mte2, %36, 0 {cannir.buffer.generated_sync}
        "ascvec.copy_gm2ub_plan"(%40, %alloc_0, %c0_i32, %c0_i32, %c0_i64, %37, %c1_i64, %38, %c0_i64, %38, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 524316 : i64, pad_storage_bits = false, storage_dtype = i32, ub_base = -1 : i64, ub_capacity = 4 : i64}> : (memref<1xi32, 1>, memref<131079xi32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
        ascvec.release_buf mte2, %36, 0 {cannir.buffer.generated_sync}
        scf.yield %32 : i64
      } else {
        scf.yield %arg4 : i64
      }
      %15 = arith.cmpi sge, %11, %c0_i64 : i64
      %16 = arith.cmpi slt, %11, %c8_i64 : i64
      %17 = arith.select %15, %16, %false : i1
      %18 = scf.if %17 -> (i64) {
        %21 = arith.muli %11, %c9216_i64 overflow<nsw> : i64
        %22 = arith.subi %9, %21 : i64
        %23 = arith.minsi %22, %c9216_i64 : i64
        %24 = arith.cmpi eq, %arg5, %c0_i64 : i64
        %25 = ascscalar.memref_address %alloc_2 : memref<9216xi32, 1>
        %26 = ascscalar.memref_address %alloc_3 : memref<9216xi32, 1>
        %27 = arith.select %24, %25, %26 : i64
        %28 = ascscalar.bind_local_ptr %27 : memref<9216xi32, 1>
        %29 = arith.cmpi ne, %arg5, %c0_i64 : i64
        %30 = arith.extui %29 : i1 to i64
        %31 = arith.addi %arg5, %c1_i64 overflow<nsw> : i64
        %32 = arith.remui %31, %c2_i64 : i64
        %33 = ascscalar.memref_address %alloc_4 : memref<9216xi32, 1>
        %34 = ascscalar.memref_address %alloc_5 : memref<9216xi32, 1>
        %35 = arith.select %24, %33, %34 : i64
        %36 = ascscalar.bind_local_ptr %35 : memref<9216xi32, 1>
        %37 = arith.select %24, %c2_i64, %c3_i64 : i64
        %38 = ascscalar.memref_address %alloc_6 : memref<9216xi32, 1>
        %39 = ascscalar.memref_address %alloc_7 : memref<9216xi32, 1>
        %40 = arith.select %24, %38, %39 : i64
        %41 = ascscalar.bind_local_ptr %40 : memref<9216xi32, 1>
        %42 = arith.select %24, %c4_i64, %c5_i64 : i64
        ascvec.get_buf v, %30, 0 {cannir.buffer.generated_sync}
        ascvec.get_buf v, %37, 0 {cannir.buffer.generated_sync}
        ascvec.get_buf v, %42, 0 {cannir.buffer.generated_sync}
        ascsimt.vf(%23, %28, %36, %41 : i64, memref<9216xi32, 1>, memref<9216xi32, 1>, memref<9216xi32, 1>) {
        ^bb0(%arg7: i64, %arg8: memref<9216xi32, 1>, %arg9: memref<9216xi32, 1>, %arg10: memref<9216xi32, 1>):
          %c1024_i32 = arith.constant 1024 : i32
          %43 = ascsimt.thread_idx {ascsimt.scope = 1 : i64, axis = "x", cannir.vf_threads = 1024 : i64} : i32
          %44 = arith.trunci %arg7 {cannir.vf_threads = 1024 : i64} : i64 to i32
          scf.for %arg11 = %43 to %44 step %c1024_i32  : i32 {
            %45 = arith.index_cast %arg11 : i32 to index
            %46 = ascsimt.load %arg8[%45] : memref<9216xi32, 1> -> i32
            %47 = ascsimt.load %arg9[%45] : memref<9216xi32, 1> -> i32
            %48 = arith.addi %46, %47 : i32
            ascsimt.store %arg10[%45], %48 : memref<9216xi32, 1>, i32
          } {cannir.vf_threads = 1024 : i64}
        } {threads = 1024 : i64}
        ascvec.release_buf v, %42, 0 {cannir.buffer.generated_sync}
        ascvec.release_buf v, %37, 0 {cannir.buffer.generated_sync}
        ascvec.release_buf v, %30, 0 {cannir.buffer.generated_sync}
        scf.yield %32 : i64
      } else {
        scf.yield %arg5 : i64
      }
      %19 = arith.cmpi sge, %12, %c0_i64 : i64
      %20 = scf.if %19 -> (i64) {
        %21 = arith.muli %12, %c9216_i64 overflow<nsw> : i64
        %22 = arith.addi %4, %21 : i64
        %23 = arith.subi %9, %21 : i64
        %24 = arith.minsi %23, %c9216_i64 : i64
        %25 = arith.cmpi eq, %arg6, %c0_i64 : i64
        %26 = ascscalar.memref_address %alloc_6 : memref<9216xi32, 1>
        %27 = ascscalar.memref_address %alloc_7 : memref<9216xi32, 1>
        %28 = arith.select %25, %26, %27 : i64
        %29 = arith.select %25, %c4_i64, %c5_i64 : i64
        %30 = arith.addi %arg6, %c1_i64 overflow<nsw> : i64
        %31 = arith.remui %30, %c2_i64 : i64
        %32 = arith.muli %22, %c32_i64 : i64
        %33 = ascscalar.memref_address %alloc_1 : memref<131079xi32>
        %34 = arith.divui %32, %c8_i64 : i64
        %35 = arith.addi %33, %34 : i64
        %36 = ascscalar.bind_gm_ptr %35 : memref<1xi32>
        %37 = ascscalar.bind_local_ptr %28 : memref<1xi32, 1>
        %38 = arith.index_cast %24 : i64 to index
        %39 = arith.muli %38, %c4 : index
        %40 = arith.cmpi ugt, %24, %c0_i64 : i64
        scf.if %40 {
          %41 = arith.index_cast %39 : index to i64
          ascvec.get_buf mte3, %29, 0 {cannir.buffer.generated_sync}
          %42 = arith.trunci %41 : i64 to i32
          ascvec.copy_ub2gm %36[%c0], %37[%c0], %c1_i32, %42, %c524316_i64, %c36864_i32 l2_cache_ctl %c0_i32 : memref<1xi32>, memref<1xi32, 1>
          ascvec.release_buf mte3, %29, 0 {cannir.buffer.generated_sync}
        }
        scf.yield %31 : i64
      } else {
        scf.yield %arg6 : i64
      }
      scf.yield %14, %18, %20 : i64, i64, i64
    }
    return
  }
}
