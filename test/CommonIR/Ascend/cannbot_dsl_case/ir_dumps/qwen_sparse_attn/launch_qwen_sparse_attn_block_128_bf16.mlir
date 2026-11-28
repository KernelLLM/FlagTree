module {
  func.func @launch_qwen_sparse_attn_block_128_bf16_79d6b96160512e43c27b695d9797cf43(%arg0: !cannir.tensor<<gm, bf16>, <<(-1, -1, 128)>, <(-1, 128, 1)>>, <<(-1, -1, 128)>, <(-1, 128, 1)>>, identity>, %arg1: !cannir.tensor<<gm, bf16>, <<(-1, -1, 128)>, <(-1, 128, 1)>>, <<(-1, -1, 128)>, <(-1, 128, 1)>>, identity>, %arg2: !cannir.tensor<<gm, bf16>, <<(-1, -1, 128, 128)>, <(-1, 128, -1, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 128, -1, 1)>>, identity>, %arg3: !cannir.tensor<<gm, bf16>, <<(-1, -1, 128, 128)>, <(-1, 128, -1, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 128, -1, 1)>>, identity>, %arg4: !cannir.tensor<<gm, i8>, <<(2048, 2048)>, <(2048, 1)>>, <<(2048, 2048)>, <(2048, 1)>>, identity>, %arg5: !cannir.tensor<<gm, i64>, <<(2, -1)>, <(-1, 1)>>, <<(2, -1)>, <(-1, 1)>>, identity>, %arg6: !cannir.tensor<<gm, i64>, <<(5, -1)>, <(-1, 1)>>, <<(5, -1)>, <(-1, 1)>>, identity>, %arg7: !cannir.tensor<<gm, i32>, <<(2, -1)>, <(-1, 1)>>, <<(2, -1)>, <(-1, 1)>>, identity>, %arg8: f32) attributes {cannbotdsl.contract_digest = "79d6b96160512e43c27b695d9797cf43e51158ef204ef9be4da09a6502448f52", cannbotdsl.public_slot_count = 9 : i64, cannbotdsl.slot_dtype_bits = [16, 16, 16, 16, 8, 64, 64, 32, 32], cannbotdsl.slot_dtype_codes = [4, 4, 4, 4, 0, 0, 0, 0, 2], cannbotdsl.slot_dtype_lanes = [1, 1, 1, 1, 1, 1, 1, 1, 1], cannbotdsl.slot_flags = [1, 1, 1, 1, 1, 1, 1, 1, 1], cannbotdsl.slot_kinds = [1, 1, 1, 1, 1, 1, 1, 1, 2], cannbotdsl.slot_type_digests = ["c3bfbd45519ac644d82ae5b3fd34f43b68f19643b42acebfa1d6ba8001f5fc6c", "c3bfbd45519ac644d82ae5b3fd34f43b68f19643b42acebfa1d6ba8001f5fc6c", "1616b50f8fdb2ad37c4587abe2ccbac07149f91bca84ed869f4951317924b749", "1616b50f8fdb2ad37c4587abe2ccbac07149f91bca84ed869f4951317924b749", "63649f3e6e296c53ebd705f85f4ed997c2453444b6d83c8a06a9dd87262b2411", "38118a8d25541e41378ae6cc983260d72eed7f74b94e13118e5609d3d922e429", "9691e4f3a979b35bb52ad1f2d384dce457e84946552fd3579d2ead48c2a1cd6a", "b279423ca759fd57dd673491bf7a7e05a53fbe97c11a11c3d872daa4d92a56e3", "4a5363e2ec1eb8cefee037ab0b231653c07f9b44a406ef92ed9ba3e6f5541dbf"], cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.host} {
    %c32_i64 = arith.constant 32 : i64
    cannir.kernel_launch @kernel_QwenSparseAttnBlock128___call___0(%c32_i64) [%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8] {cannbotdsl.profile = {indices = array<i64>, name = "Qwen_Sparse_Attn", op_type = "Qwen_Sparse_Attn", roles = array<i64>}} : !cannir.tensor<<gm, bf16>, <<(-1, -1, 128)>, <(-1, 128, 1)>>, <<(-1, -1, 128)>, <(-1, 128, 1)>>, identity>, !cannir.tensor<<gm, bf16>, <<(-1, -1, 128)>, <(-1, 128, 1)>>, <<(-1, -1, 128)>, <(-1, 128, 1)>>, identity>, !cannir.tensor<<gm, bf16>, <<(-1, -1, 128, 128)>, <(-1, 128, -1, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 128, -1, 1)>>, identity>, !cannir.tensor<<gm, bf16>, <<(-1, -1, 128, 128)>, <(-1, 128, -1, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 128, -1, 1)>>, identity>, !cannir.tensor<<gm, i8>, <<(2048, 2048)>, <(2048, 1)>>, <<(2048, 2048)>, <(2048, 1)>>, identity>, !cannir.tensor<<gm, i64>, <<(2, -1)>, <(-1, 1)>>, <<(2, -1)>, <(-1, 1)>>, identity>, !cannir.tensor<<gm, i64>, <<(5, -1)>, <(-1, 1)>>, <<(5, -1)>, <(-1, 1)>>, identity>, !cannir.tensor<<gm, i32>, <<(2, -1)>, <(-1, 1)>>, <<(2, -1)>, <(-1, 1)>>, identity>, f32
    return
  }
  func.func @kernel_QwenSparseAttnBlock128___call___0_aiv(%arg0: !cannir.tensor<<gm, bf16>, <<(-1, -1, 128)>, <(-1, 128, 1)>>, <<(-1, -1, 128)>, <(-1, 128, 1)>>, identity>, %arg1: !cannir.tensor<<gm, bf16>, <<(-1, -1, 128)>, <(-1, 128, 1)>>, <<(-1, -1, 128)>, <(-1, 128, 1)>>, identity>, %arg2: !cannir.tensor<<gm, bf16>, <<(-1, -1, 128, 128)>, <(-1, 128, -1, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 128, -1, 1)>>, identity>, %arg3: !cannir.tensor<<gm, bf16>, <<(-1, -1, 128, 128)>, <(-1, 128, -1, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 128, -1, 1)>>, identity>, %arg4: !cannir.tensor<<gm, i8>, <<(2048, 2048)>, <(2048, 1)>>, <<(2048, 2048)>, <(2048, 1)>>, identity>, %arg5: !cannir.tensor<<gm, i64>, <<(2, -1)>, <(-1, 1)>>, <<(2, -1)>, <(-1, 1)>>, identity>, %arg6: !cannir.tensor<<gm, i64>, <<(5, -1)>, <(-1, 1)>>, <<(5, -1)>, <(-1, 1)>>, identity>, %arg7: !cannir.tensor<<gm, i32>, <<(2, -1)>, <(-1, 1)>>, <<(2, -1)>, <(-1, 1)>>, identity>, %arg8: f32) attributes {cannir.buffer.auto_sync, cannir.buffer.sync_analyzed, cannir.buffer.sync_verified, cannir.buffer.tensor_state, cannir.core = "aiv", cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.device, cannir.explicit_ring_buffer_cursors, cannir.kernel_group = "kernel_QwenSparseAttnBlock128___call___0", cannir.resource.ids_prepared} {
    %c256_i32 = arith.constant 256 : i32
    %c16_i32 = arith.constant 16 : i32
    %c8_i32 = arith.constant 8 : i32
    %c4 = arith.constant 4 : index
    %c3 = arith.constant 3 : index
    %c16_i64 = arith.constant 16 : i64
    %c6_i64 = arith.constant 6 : i64
    %c5_i64 = arith.constant 5 : i64
    %c4_i64 = arith.constant 4 : i64
    %c3_i64 = arith.constant 3 : i64
    %c2_i64 = arith.constant 2 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %c8_i64 = arith.constant 8 : i64
    %c21_i64 = arith.constant 21 : i64
    %c22_i64 = arith.constant 22 : i64
    %c23_i64 = arith.constant 23 : i64
    %c24_i64 = arith.constant 24 : i64
    %c25_i64 = arith.constant 25 : i64
    %c26_i64 = arith.constant 26 : i64
    %c27_i64 = arith.constant 27 : i64
    %c2 = arith.constant 2 : index
    %c1_i32 = arith.constant 1 : i32
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %c2048_i64 = arith.constant 2048 : i64
    %c127_i64 = arith.constant 127 : i64
    %c-1_i64 = arith.constant -1 : i64
    %false = arith.constant false
    %c0_i32 = arith.constant 0 : i32
    %c128_i64 = arith.constant 128 : i64
    %c32_i64 = arith.constant 32 : i64
    %alloc = memref.alloc() {cannir.arg_index = 0 : i64} : memref<1xbf16>
    %alloc_0 = memref.alloc() {cannir.arg_index = 4 : i64} : memref<4194304xi8>
    %alloc_1 = memref.alloc() {cannir.arg_index = 5 : i64} : memref<1xi64>
    %alloc_2 = memref.alloc() {cannir.arg_index = 6 : i64} : memref<1xi64>
    %alloc_3 = memref.alloc() {cannir.arg_index = 7 : i64} : memref<1xi32>
    %alloc_4 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<128xf32, 1>
    %alloc_5 = memref.alloc() {cannir.byte_offset = 512 : i64} : memref<128xf32, 1>
    %alloc_6 = memref.alloc() {cannir.byte_offset = 1024 : i64} : memref<128xf32, 1>
    %alloc_7 = memref.alloc() {cannir.byte_offset = 1536 : i64} : memref<128xf32, 1>
    %alloc_8 = memref.alloc() {cannir.byte_offset = 2048 : i64} : memref<128xf32, 1>
    %alloc_9 = memref.alloc() {cannir.byte_offset = 2560 : i64} : memref<128xf32, 1>
    %alloc_10 = memref.alloc() {cannir.byte_offset = 3072 : i64} : memref<128xf32, 1>
    %alloc_11 = memref.alloc() {cannir.byte_offset = 3584 : i64} : memref<2048xf32, 1>
    %alloc_12 = memref.alloc() {cannir.byte_offset = 11776 : i64} : memref<128xi8, 1>
    %alloc_13 = memref.alloc() {cannir.byte_offset = 11904 : i64} : memref<128xi8, 1>
    %alloc_14 = memref.alloc() {cannir.byte_offset = 12032 : i64} : memref<2048xi8, 1>
    %alloc_15 = memref.alloc() {cannir.byte_offset = 14080 : i64} : memref<2048xbf16, 1>
    %alloc_16 = memref.alloc() {cannir.byte_offset = 18176 : i64} : memref<2048xbf16, 1>
    %alloc_17 = memref.alloc() {cannir.byte_offset = 22272 : i64} : memref<2048xbf16, 1>
    %alloc_18 = memref.alloc() {cannir.byte_offset = 30976 : i64} : memref<2048xf32, 1>
    %alloc_19 = memref.alloc() {cannir.byte_offset = 39168 : i64} : memref<2048xf32, 1>
    %alloc_20 = memref.alloc() {cannir.byte_offset = 47360 : i64} : memref<2048xf32, 1>
    %alloc_21 = memref.alloc() {cannir.byte_offset = 55552 : i64} : memref<2048xf32, 1>
    %alloc_22 = memref.alloc() {cannir.byte_offset = 63744 : i64} : memref<2048xf32, 1>
    %alloc_23 = memref.alloc() {cannir.byte_offset = 212992 : i64} : memref<4096xbf16, 2>
    %alloc_24 = memref.alloc() {cannir.byte_offset = 204800 : i64} : memref<4096xbf16, 2>
    ascvec.sync_intra_arrive v, %c4_i64
    ascvec.sync_intra_arrive v, %c3_i64
    ascvec.sync_intra_arrive v, %c2_i64
    ascvec.sync_intra_arrive v, %c1_i64
    ascvec.sync_intra_arrive v, %c0_i64
    %0 = ascscalar.get_block_idx
    %1 = ascscalar.get_subblock_id
    %2 = arith.index_cast %0 : i64 to index
    %3 = memref.load %alloc_1[%2] : memref<1xi64>
    %4 = cannir.extract_layout %arg5 : !cannir.tensor<<gm, i64>, <<(2, -1)>, <(-1, 1)>>, <<(2, -1)>, <(-1, 1)>>, identity>
    %5 = cannir.extract_stride %4 : <<(2, -1)>, <(-1, 1)>> -> <(-1, 1)>
    %6 = cannir.extract_dim %5[0] : !cannir.stride<(-1, 1)>
    %7 = arith.index_cast %6 : i64 to index
    %8 = arith.addi %7, %2 : index
    %9 = memref.load %alloc_1[%8] : memref<1xi64>
    %10:11 = scf.for %arg9 = %3 to %9 step %c1_i64 iter_args(%arg10 = %c0_i64, %arg11 = %c0_i64, %arg12 = %c0_i64, %arg13 = %c0_i64, %arg14 = %c0_i64, %arg15 = %c0_i64, %arg16 = %c0_i64, %arg17 = %c0_i64, %arg18 = %c0_i64, %arg19 = %c0_i64, %arg20 = %c0_i64) -> (i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64)  : i64 {
      %11 = arith.index_cast %arg9 : i64 to index
      %12 = memref.load %alloc_2[%11] : memref<1xi64>
      %13 = cannir.extract_layout %arg6 : !cannir.tensor<<gm, i64>, <<(5, -1)>, <(-1, 1)>>, <<(5, -1)>, <(-1, 1)>>, identity>
      %14 = cannir.extract_stride %13 : <<(5, -1)>, <(-1, 1)>> -> <(-1, 1)>
      %15 = cannir.extract_dim %14[0] : !cannir.stride<(-1, 1)>
      %16 = arith.index_cast %15 : i64 to index
      %17 = arith.addi %16, %11 : index
      %18 = memref.load %alloc_2[%17] : memref<1xi64>
      %19 = arith.muli %16, %c2 : index
      %20 = arith.addi %19, %11 : index
      %21 = memref.load %alloc_2[%20] : memref<1xi64>
      %22 = arith.muli %16, %c3 : index
      %23 = arith.addi %22, %11 : index
      %24 = memref.load %alloc_2[%23] : memref<1xi64>
      %25 = arith.muli %16, %c4 : index
      %26 = arith.addi %25, %11 : index
      %27 = memref.load %alloc_2[%26] : memref<1xi64>
      %28 = arith.muli %21, %c32_i64 : i64
      %29 = arith.subi %c128_i64, %28 : i64
      %30 = arith.cmpi sgt, %29, %c32_i64 : i64
      %31 = arith.cmpi sgt, %29, %c0_i64 : i64
      %32 = arith.select %31, %29, %c0_i64 : i64
      %33 = arith.select %30, %c32_i64, %32 : i64
      %34 = arith.addi %33, %c1_i64 : i64
      %35 = arith.divsi %34, %c2_i64 : i64
      %36 = arith.divsi %33, %c2_i64 : i64
      %37 = arith.cmpi eq, %1, %c0_i64 : i64
      %38 = arith.select %37, %35, %36 : i64
      %39 = arith.select %37, %c0_i64, %35 : i64
      %40 = arith.cmpi sgt, %27, %c0_i64 : i64
      %41:11 = scf.if %40 -> (i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) {
        %42 = arith.addi %27, %c3_i64 : i64
        %43:11 = scf.for %arg21 = %c0_i64 to %42 step %c1_i64 iter_args(%arg22 = %arg10, %arg23 = %arg11, %arg24 = %arg12, %arg25 = %arg13, %arg26 = %arg14, %arg27 = %arg15, %arg28 = %arg16, %arg29 = %arg17, %arg30 = %arg18, %arg31 = %arg19, %arg32 = %arg20) -> (i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64)  : i64 {
          %44 = arith.cmpi slt, %arg21, %27 : i64
          %45:3 = scf.if %44 -> (i64, i64, i64) {
            %58 = arith.addi %arg22, %c1_i64 : i64
            %59 = arith.remui %58, %c3_i64 : i64
            %60 = arith.addi %arg26, %c1_i64 : i64
            %61 = arith.remui %60, %c2_i64 : i64
            %62 = arith.addi %arg25, %c1_i64 : i64
            %63 = arith.remui %62, %c2_i64 : i64
            scf.yield %59, %63, %61 : i64, i64, i64
          } else {
            scf.yield %arg22, %arg25, %arg26 : i64, i64, i64
          }
          %46 = arith.cmpi sge, %arg21, %c1_i64 : i64
          %47 = arith.addi %27, %c1_i64 : i64
          %48 = arith.cmpi slt, %arg21, %47 : i64
          %49 = arith.select %46, %48, %false : i1
          %50:4 = scf.if %49 -> (i64, i64, i64, i64) {
            %58 = arith.addi %24, %arg21 : i64
            %59 = arith.subi %58, %c1_i64 : i64
            %60 = arith.index_cast %59 : i64 to index
            %61 = arith.cmpi eq, %arg21, %c1_i64 : i64
            %62 = arith.cmpi eq, %arg23, %c1_i64 : i64
            %63 = arith.cmpi eq, %arg23, %c0_i64 : i64
            %64 = arith.addi %arg23, %c1_i64 : i64
            %65 = arith.remui %64, %c3_i64 : i64
            %66 = arith.addi %39, %38 : i64
            %67 = arith.maxsi %39, %c0_i64 : i64
            %68 = arith.minsi %66, %c128_i64 : i64
            %69 = arith.subi %68, %67 : i64
            %70 = arith.maxsi %69, %c0_i64 : i64
            %71 = arith.cmpi sgt, %70, %c0_i64 : i64
            %72 = scf.if %71 -> (i64) {
              %101 = cannir.extract_layout %arg7 : !cannir.tensor<<gm, i32>, <<(2, -1)>, <(-1, 1)>>, <<(2, -1)>, <(-1, 1)>>, identity>
              %102 = cannir.extract_stride %101 : <<(2, -1)>, <(-1, 1)>> -> <(-1, 1)>
              %103 = cannir.extract_dim %102[0] : !cannir.stride<(-1, 1)>
              %104 = arith.index_cast %103 : i64 to index
              %105 = arith.addi %104, %60 : index
              %106 = memref.load %alloc_3[%105] : memref<1xi32>
              %107 = arith.extsi %106 : i32 to i64
              %108 = arith.subi %107, %c1_i64 : i64
              %109 = arith.maxsi %108, %c-1_i64 : i64
              %110 = arith.minsi %109, %c127_i64 : i64
              %111 = arith.cmpi sge, %110, %c0_i64 : i64
              %112 = arith.select %111, %110, %c0_i64 : i64
              %113 = arith.subi %c0_i64, %110 : i64
              %114 = arith.select %111, %c0_i64, %113 : i64
              %115 = arith.cmpi eq, %arg29, %c0_i64 : i64
              %116 = ascscalar.memref_address %alloc_12 : memref<128xi8, 1>
              %117 = ascscalar.memref_address %alloc_13 : memref<128xi8, 1>
              %118 = arith.select %115, %116, %117 : i64
              %119 = arith.cmpi ne, %arg29, %c0_i64 : i64
              %120 = arith.extui %119 : i1 to i64
              %121 = arith.addi %arg29, %c1_i64 : i64
              %122 = arith.remui %121, %c2_i64 : i64
              %123 = arith.muli %112, %c2048_i64 : i64
              %124 = arith.addi %123, %114 : i64
              %125 = ascscalar.bind_local_ptr %118 : memref<1xi8, 1>
              ascvec.get_buf mte2, %120, 0 {cannir.buffer.generated_sync}
              "ascvec.copy_gm2ub_plan"(%125, %alloc_0, %c0_i32, %c0_i32, %c0_i64, %124, %c1_i64, %c128_i64, %c0_i64, %c128_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 4194304 : i64, pad_storage_bits = false, storage_dtype = i8, ub_base = -1 : i64, ub_capacity = 1 : i64}> : (memref<1xi8, 1>, memref<4194304xi8>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
              ascvec.release_buf mte2, %120, 0 {cannir.buffer.generated_sync}
              %126 = arith.subi %67, %39 : i64
              %127 = ascscalar.bind_local_ptr %118 : memref<128xi8, 1>
              ascvec.get_buf v, %120, 0 {cannir.buffer.generated_sync}
              ascvec.vf(%127, %126, %alloc_14, %70 : memref<128xi8, 1>, i64, memref<2048xi8, 1>, i64) {
              ^bb0(%arg33: memref<128xi8, 1>, %arg34: i64, %arg35: memref<2048xi8, 1>, %arg36: i64):
                %c1_i64_25 = arith.constant 1 : i64
                %c0_i64_26 = arith.constant 0 : i64
                %c128_i64_27 = arith.constant 128 : i64
                %c0_28 = arith.constant 0 : index
                %c128_i32 = arith.constant 128 : i32
                %mask, %remaining_out = ascvec.update_mask %c128_i32 elem_bits 8 : !ascvec.mask
                %128 = ascvec.loadalign %arg33[%c0_28] : memref<128xi8, 1>, index -> <i8>
                scf.for %arg37 = %c0_i64_26 to %arg36 step %c1_i64_25  : i64 {
                  %129 = arith.addi %arg34, %arg37 : i64
                  %130 = arith.muli %129, %c128_i64_27 : i64
                  %131 = arith.index_cast %130 : i64 to index
                  ascvec.storealign %arg35[%131], %128, %mask : memref<2048xi8, 1>, index, <i8>
                }
              } {cannir.vf_mode = "simd"}
              ascvec.release_buf v, %120, 0 {cannir.buffer.generated_sync}
              scf.yield %122 : i64
            } else {
              scf.yield %arg29 : i64
            }
            %73 = ascscalar.memref_address %alloc_19 : memref<2048xf32, 1>
            %74 = ascscalar.memref_address %alloc_20 : memref<2048xf32, 1>
            %75 = arith.select %62, %73, %74 : i64
            %76 = ascscalar.memref_address %alloc_18 : memref<2048xf32, 1>
            %77 = arith.select %63, %76, %75 : i64
            %78 = ascscalar.bind_local_ptr %77 : memref<2048xf32, 1>
            %79 = arith.select %62, %c22_i64, %c23_i64 : i64
            %80 = arith.select %63, %c21_i64, %79 : i64
            %81 = arith.cmpi eq, %arg30, %c0_i64 : i64
            %82 = ascscalar.memref_address %alloc_15 : memref<2048xbf16, 1>
            %83 = ascscalar.memref_address %alloc_16 : memref<2048xbf16, 1>
            %84 = arith.select %81, %82, %83 : i64
            %85 = ascscalar.bind_local_ptr %84 : memref<2048xbf16, 1>
            %86 = arith.select %81, %c2_i64, %c3_i64 : i64
            %87 = arith.addi %arg30, %c1_i64 : i64
            %88 = arith.remui %87, %c2_i64 : i64
            %89 = scf.if %61 -> (i64) {
              %101 = arith.cmpi eq, %80, %c22_i64 : i64
              %102 = arith.extui %101 : i1 to i64
              %103 = arith.cmpi eq, %80, %c23_i64 : i64
              %104 = arith.select %103, %c2_i64, %102 : i64
              ascvec.sync_intra_wait v, %104 {cannir.sync.banks = 3 : i32}
              ascvec.vf(%78, %arg8, %alloc_14, %alloc_4, %38 : memref<2048xf32, 1>, f32, memref<2048xi8, 1>, memref<128xf32, 1>, i64) {
              ^bb0(%arg33: memref<2048xf32, 1>, %arg34: f32, %arg35: memref<2048xi8, 1>, %arg36: memref<128xf32, 1>, %arg37: i64):
                %c1_i64_25 = arith.constant 1 : i64
                %c0_i64_26 = arith.constant 0 : i64
                %c8_i64_27 = arith.constant 8 : i64
                %cst = arith.constant -1.000000e+30 : f32
                %cst_28 = arith.constant -3.000000e+38 : f32
                %cst_29 = arith.constant 0.000000e+00 : f32
                %c64_i64 = arith.constant 64 : i64
                %c128_i32 = arith.constant 128 : i32
                %c128_i64_30 = arith.constant 128 : i64
                %c64_i32 = arith.constant 64 : i32
                %mask, %remaining_out = ascvec.update_mask %c64_i32 : !ascvec.mask
                scf.for %arg38 = %c0_i64_26 to %arg37 step %c1_i64_25  : i64 {
                  %105 = arith.muli %arg38, %c128_i64_30 : i64
                  %mask_31, %remaining_out_32 = ascvec.update_mask %c128_i32 elem_bits 16 : !ascvec.mask
                  %106 = arith.index_cast %105 : i64 to index
                  %107 = ascvec.loadalign %arg33[%106] : memref<2048xf32, 1>, index -> <f32>
                  %108 = ascvec.cast %107, %mask {reg_layout = 0 : i32} : <f32> -> <bf16>
                  %109 = arith.addi %105, %c64_i64 : i64
                  %110 = arith.index_cast %109 : i64 to index
                  %111 = ascvec.loadalign %arg33[%110] : memref<2048xf32, 1>, index -> <f32>
                  %112 = ascvec.cast %111, %mask {reg_layout = 0 : i32} : <f32> -> <bf16>
                  %113 = ascvec.cast %108, %mask_31 {reg_layout = 0 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                  %114 = ascvec.cast %112, %mask_31 {reg_layout = 0 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                  %115 = ascvec.mul_scalar %113, %arg34, %mask : <f32>, f32
                  %116 = ascvec.cast %115, %mask {reg_layout = 0 : i32} : <f32> -> <bf16>
                  %117 = ascvec.mul_scalar %114, %arg34, %mask : <f32>, f32
                  %118 = ascvec.cast %117, %mask {reg_layout = 0 : i32} : <f32> -> <bf16>
                  %119 = ascvec.cast %116, %mask_31 {reg_layout = 0 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                  %120 = ascvec.cast %118, %mask_31 {reg_layout = 0 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                  %121 = ascvec.loadalign_unpack %arg35[%106] {mode = "b8_to_b32"} : memref<2048xi8, 1>, index -> <ui32>
                  %122 = ascvec.reinterpret %121 : <ui32> -> <i32>
                  %123 = ascvec.cast %122, %mask : <i32> -> <f32>
                  %124 = ascvec.cmp_eq_scalar %123, %cst_29, %mask : <f32>, f32 -> !ascvec.mask
                  %125 = ascvec.duplicate_scalar %cst_28, %mask : f32 -> <f32>
                  %126 = ascvec.select %119, %125, %124 : <f32>
                  %127 = ascvec.loadalign_unpack %arg35[%110] {mode = "b8_to_b32"} : memref<2048xi8, 1>, index -> <ui32>
                  %128 = ascvec.reinterpret %127 : <ui32> -> <i32>
                  %129 = ascvec.cast %128, %mask : <i32> -> <f32>
                  %130 = ascvec.cmp_eq_scalar %129, %cst_29, %mask : <f32>, f32 -> !ascvec.mask
                  %131 = ascvec.select %120, %125, %130 : <f32>
                  ascvec.storealign %arg33[%106], %126, %mask : memref<2048xf32, 1>, index, <f32>
                  ascvec.storealign %arg33[%110], %131, %mask : memref<2048xf32, 1>, index, <f32>
                  %132 = ascvec.max %126, %131, %mask : <f32>
                  %133 = ascvec.reduce_max %132, %mask : <f32>
                  %134 = ascvec.cmp_le_scalar %133, %cst, %mask : <f32>, f32 -> !ascvec.mask
                  %135 = ascvec.duplicate_scalar %cst_29, %mask : f32 -> <f32>
                  %136 = ascvec.select %135, %133, %134 : <f32>
                  %137 = arith.muli %arg38, %c8_i64_27 : i64
                  %138 = arith.index_cast %137 : i64 to index
                  ascvec.storealign_1st %arg36[%138], %136 : memref<128xf32, 1>, index, <f32>
                }
              } {cannir.vf_mode = "simd"}
              ascvec.get_buf v, %86, 0 {cannir.buffer.generated_sync}
              ascvec.vf(%alloc_4, %78, %85, %alloc_5, %38 : memref<128xf32, 1>, memref<2048xf32, 1>, memref<2048xbf16, 1>, memref<128xf32, 1>, i64) {
              ^bb0(%arg33: memref<128xf32, 1>, %arg34: memref<2048xf32, 1>, %arg35: memref<2048xbf16, 1>, %arg36: memref<128xf32, 1>, %arg37: i64):
                %c1_i64_25 = arith.constant 1 : i64
                %c0_i64_26 = arith.constant 0 : i64
                %c16_27 = arith.constant 16 : index
                %c16_i64_28 = arith.constant 16 : i64
                %c64_i64 = arith.constant 64 : i64
                %c0_29 = arith.constant 0 : index
                %c8_i64_30 = arith.constant 8 : i64
                %c128_i64_31 = arith.constant 128 : i64
                %c128_i32 = arith.constant 128 : i32
                %c64_i32 = arith.constant 64 : i32
                %mask, %remaining_out = ascvec.update_mask %c64_i32 : !ascvec.mask
                %mask_32, %remaining_out_33 = ascvec.update_mask %c128_i32 elem_bits 16 : !ascvec.mask
                scf.for %arg38 = %c0_i64_26 to %arg37 step %c1_i64_25  : i64 {
                  %105 = arith.muli %arg38, %c128_i64_31 : i64
                  %106 = arith.muli %arg38, %c8_i64_30 : i64
                  %107 = arith.index_cast %106 : i64 to index
                  %108 = ascvec.loadalign_brc %arg33[%107] : memref<128xf32, 1>, index -> <f32>
                  %109 = arith.index_cast %105 : i64 to index
                  %110 = ascvec.loadalign %arg34[%109] : memref<2048xf32, 1>, index -> <f32>
                  %111 = ascvec.exp_sub %110, %108, %mask : <f32>, <f32> -> <f32>
                  %112 = arith.addi %105, %c64_i64 : i64
                  %113 = arith.index_cast %112 : i64 to index
                  %114 = ascvec.loadalign %arg34[%113] : memref<2048xf32, 1>, index -> <f32>
                  %115 = ascvec.exp_sub %114, %108, %mask : <f32>, <f32> -> <f32>
                  %116 = ascvec.cast %111, %mask {reg_layout = 0 : i32} : <f32> -> <bf16>
                  %117 = ascvec.cast %115, %mask {reg_layout = 0 : i32} : <f32> -> <bf16>
                  %result0, %result1 = ascvec.deintlv %116, %117 : <bf16>, <bf16> -> <bf16>, <bf16>
                  %118 = arith.remsi %arg38, %c16_i64_28 : i64
                  %119 = arith.muli %118, %c16_i64_28 overflow<nsw> : i64
                  %120 = arith.index_cast %119 : i64 to index
                  ascvec.storealign_strided %arg35[%120], %result0, %mask_32, %c16_27, %c0_29 : memref<2048xbf16, 1>, <bf16>
                  %121 = ascvec.add %111, %115, %mask : <f32>
                  %122 = ascvec.reduce_sum %121, %mask : <f32>
                  ascvec.storealign_1st %arg36[%107], %122 : memref<128xf32, 1>, index, <f32>
                }
              } {cannir.vf_mode = "simd"}
              ascvec.release_buf v, %86, 0 {cannir.buffer.generated_sync}
              ascvec.sync_intra_arrive v, %104 {cannir.sync.banks = 3 : i32}
              scf.yield %arg27 : i64
            } else {
              %101 = arith.cmpi eq, %arg27, %c1_i64 : i64
              %102 = ascscalar.memref_address %alloc_9 : memref<128xf32, 1>
              %103 = ascscalar.memref_address %alloc_10 : memref<128xf32, 1>
              %104 = arith.select %101, %102, %103 : i64
              %105 = arith.cmpi eq, %arg27, %c0_i64 : i64
              %106 = ascscalar.memref_address %alloc_8 : memref<128xf32, 1>
              %107 = arith.select %105, %106, %104 : i64
              %108 = ascscalar.bind_local_ptr %107 : memref<128xf32, 1>
              %109 = arith.addi %arg27, %c1_i64 : i64
              %110 = arith.remui %109, %c3_i64 : i64
              %111 = arith.cmpi eq, %80, %c22_i64 : i64
              %112 = arith.extui %111 : i1 to i64
              %113 = arith.cmpi eq, %80, %c23_i64 : i64
              %114 = arith.select %113, %c2_i64, %112 : i64
              ascvec.sync_intra_wait v, %114 {cannir.sync.banks = 3 : i32}
              ascvec.vf(%78, %arg8, %alloc_14, %alloc_6, %38, %alloc_4 : memref<2048xf32, 1>, f32, memref<2048xi8, 1>, memref<128xf32, 1>, i64, memref<128xf32, 1>) {
              ^bb0(%arg33: memref<2048xf32, 1>, %arg34: f32, %arg35: memref<2048xi8, 1>, %arg36: memref<128xf32, 1>, %arg37: i64, %arg38: memref<128xf32, 1>):
                %c1_i64_25 = arith.constant 1 : i64
                %c0_i64_26 = arith.constant 0 : i64
                %c8_i64_27 = arith.constant 8 : i64
                %cst = arith.constant -1.000000e+30 : f32
                %cst_28 = arith.constant -3.000000e+38 : f32
                %cst_29 = arith.constant 0.000000e+00 : f32
                %c64_i64 = arith.constant 64 : i64
                %c128_i32 = arith.constant 128 : i32
                %c128_i64_30 = arith.constant 128 : i64
                %c64_i32 = arith.constant 64 : i32
                %mask, %remaining_out = ascvec.update_mask %c64_i32 : !ascvec.mask
                scf.for %arg39 = %c0_i64_26 to %arg37 step %c1_i64_25  : i64 {
                  %115 = arith.muli %arg39, %c128_i64_30 : i64
                  %mask_31, %remaining_out_32 = ascvec.update_mask %c128_i32 elem_bits 16 : !ascvec.mask
                  %116 = arith.index_cast %115 : i64 to index
                  %117 = ascvec.loadalign %arg33[%116] : memref<2048xf32, 1>, index -> <f32>
                  %118 = ascvec.cast %117, %mask {reg_layout = 0 : i32} : <f32> -> <bf16>
                  %119 = arith.addi %115, %c64_i64 : i64
                  %120 = arith.index_cast %119 : i64 to index
                  %121 = ascvec.loadalign %arg33[%120] : memref<2048xf32, 1>, index -> <f32>
                  %122 = ascvec.cast %121, %mask {reg_layout = 0 : i32} : <f32> -> <bf16>
                  %123 = ascvec.cast %118, %mask_31 {reg_layout = 0 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                  %124 = ascvec.cast %122, %mask_31 {reg_layout = 0 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                  %125 = ascvec.mul_scalar %123, %arg34, %mask : <f32>, f32
                  %126 = ascvec.cast %125, %mask {reg_layout = 0 : i32} : <f32> -> <bf16>
                  %127 = ascvec.mul_scalar %124, %arg34, %mask : <f32>, f32
                  %128 = ascvec.cast %127, %mask {reg_layout = 0 : i32} : <f32> -> <bf16>
                  %129 = ascvec.cast %126, %mask_31 {reg_layout = 0 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                  %130 = ascvec.cast %128, %mask_31 {reg_layout = 0 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                  %131 = ascvec.loadalign_unpack %arg35[%116] {mode = "b8_to_b32"} : memref<2048xi8, 1>, index -> <ui32>
                  %132 = ascvec.reinterpret %131 : <ui32> -> <i32>
                  %133 = ascvec.cast %132, %mask : <i32> -> <f32>
                  %134 = ascvec.cmp_eq_scalar %133, %cst_29, %mask : <f32>, f32 -> !ascvec.mask
                  %135 = ascvec.duplicate_scalar %cst_28, %mask : f32 -> <f32>
                  %136 = ascvec.select %129, %135, %134 : <f32>
                  %137 = ascvec.loadalign_unpack %arg35[%120] {mode = "b8_to_b32"} : memref<2048xi8, 1>, index -> <ui32>
                  %138 = ascvec.reinterpret %137 : <ui32> -> <i32>
                  %139 = ascvec.cast %138, %mask : <i32> -> <f32>
                  %140 = ascvec.cmp_eq_scalar %139, %cst_29, %mask : <f32>, f32 -> !ascvec.mask
                  %141 = ascvec.select %130, %135, %140 : <f32>
                  ascvec.storealign %arg33[%116], %136, %mask : memref<2048xf32, 1>, index, <f32>
                  ascvec.storealign %arg33[%120], %141, %mask : memref<2048xf32, 1>, index, <f32>
                  %142 = ascvec.max %136, %141, %mask : <f32>
                  %143 = ascvec.reduce_max %142, %mask : <f32>
                  %144 = ascvec.cmp_le_scalar %143, %cst, %mask : <f32>, f32 -> !ascvec.mask
                  %145 = ascvec.duplicate_scalar %cst_29, %mask : f32 -> <f32>
                  %146 = ascvec.select %145, %143, %144 : <f32>
                  %147 = arith.muli %arg39, %c8_i64_27 : i64
                  %148 = arith.index_cast %147 : i64 to index
                  ascvec.storealign_1st %arg36[%148], %146 : memref<128xf32, 1>, index, <f32>
                }
                ascvec.mem_bar vst_vld
                scf.for %arg39 = %c0_i64_26 to %arg37 step %c1_i64_25  : i64 {
                  %115 = arith.muli %arg39, %c8_i64_27 : i64
                  %116 = arith.index_cast %115 : i64 to index
                  %117 = ascvec.loadalign_brc %arg38[%116] : memref<128xf32, 1>, index -> <f32>
                  %118 = ascvec.loadalign_brc %arg36[%116] : memref<128xf32, 1>, index -> <f32>
                  %119 = ascvec.max %117, %118, %mask : <f32>
                  ascvec.storealign_1st %arg36[%116], %119 : memref<128xf32, 1>, index, <f32>
                }
              } {cannir.vf_mode = "simd"}
              ascvec.get_buf v, %86, 0 {cannir.buffer.generated_sync}
              ascvec.vf(%alloc_6, %78, %85, %alloc_7, %38 : memref<128xf32, 1>, memref<2048xf32, 1>, memref<2048xbf16, 1>, memref<128xf32, 1>, i64) {
              ^bb0(%arg33: memref<128xf32, 1>, %arg34: memref<2048xf32, 1>, %arg35: memref<2048xbf16, 1>, %arg36: memref<128xf32, 1>, %arg37: i64):
                %c1_i64_25 = arith.constant 1 : i64
                %c0_i64_26 = arith.constant 0 : i64
                %c16_27 = arith.constant 16 : index
                %c16_i64_28 = arith.constant 16 : i64
                %c64_i64 = arith.constant 64 : i64
                %c0_29 = arith.constant 0 : index
                %c8_i64_30 = arith.constant 8 : i64
                %c128_i64_31 = arith.constant 128 : i64
                %c128_i32 = arith.constant 128 : i32
                %c64_i32 = arith.constant 64 : i32
                %mask, %remaining_out = ascvec.update_mask %c64_i32 : !ascvec.mask
                %mask_32, %remaining_out_33 = ascvec.update_mask %c128_i32 elem_bits 16 : !ascvec.mask
                scf.for %arg38 = %c0_i64_26 to %arg37 step %c1_i64_25  : i64 {
                  %115 = arith.muli %arg38, %c128_i64_31 : i64
                  %116 = arith.muli %arg38, %c8_i64_30 : i64
                  %117 = arith.index_cast %116 : i64 to index
                  %118 = ascvec.loadalign_brc %arg33[%117] : memref<128xf32, 1>, index -> <f32>
                  %119 = arith.index_cast %115 : i64 to index
                  %120 = ascvec.loadalign %arg34[%119] : memref<2048xf32, 1>, index -> <f32>
                  %121 = ascvec.exp_sub %120, %118, %mask : <f32>, <f32> -> <f32>
                  %122 = arith.addi %115, %c64_i64 : i64
                  %123 = arith.index_cast %122 : i64 to index
                  %124 = ascvec.loadalign %arg34[%123] : memref<2048xf32, 1>, index -> <f32>
                  %125 = ascvec.exp_sub %124, %118, %mask : <f32>, <f32> -> <f32>
                  %126 = ascvec.cast %121, %mask {reg_layout = 0 : i32} : <f32> -> <bf16>
                  %127 = ascvec.cast %125, %mask {reg_layout = 0 : i32} : <f32> -> <bf16>
                  %result0, %result1 = ascvec.deintlv %126, %127 : <bf16>, <bf16> -> <bf16>, <bf16>
                  %128 = arith.remsi %arg38, %c16_i64_28 : i64
                  %129 = arith.muli %128, %c16_i64_28 overflow<nsw> : i64
                  %130 = arith.index_cast %129 : i64 to index
                  ascvec.storealign_strided %arg35[%130], %result0, %mask_32, %c16_27, %c0_29 : memref<2048xbf16, 1>, <bf16>
                  %131 = ascvec.add %121, %125, %mask : <f32>
                  %132 = ascvec.reduce_sum %131, %mask : <f32>
                  ascvec.storealign_1st %arg36[%117], %132 : memref<128xf32, 1>, index, <f32>
                }
              } {cannir.vf_mode = "simd"}
              ascvec.release_buf v, %86, 0 {cannir.buffer.generated_sync}
              ascvec.sync_intra_arrive v, %114 {cannir.sync.banks = 3 : i32}
              ascvec.vf(%alloc_4, %alloc_6, %alloc_5, %alloc_7, %108, %38 : memref<128xf32, 1>, memref<128xf32, 1>, memref<128xf32, 1>, memref<128xf32, 1>, memref<128xf32, 1>, i64) {
              ^bb0(%arg33: memref<128xf32, 1>, %arg34: memref<128xf32, 1>, %arg35: memref<128xf32, 1>, %arg36: memref<128xf32, 1>, %arg37: memref<128xf32, 1>, %arg38: i64):
                %c1_i64_25 = arith.constant 1 : i64
                %c0_i64_26 = arith.constant 0 : i64
                %c8_i64_27 = arith.constant 8 : i64
                %c64_i32 = arith.constant 64 : i32
                %mask, %remaining_out = ascvec.update_mask %c64_i32 : !ascvec.mask
                scf.for %arg39 = %c0_i64_26 to %arg38 step %c1_i64_25  : i64 {
                  %115 = arith.muli %arg39, %c8_i64_27 : i64
                  %116 = arith.index_cast %115 : i64 to index
                  %117 = ascvec.loadalign_brc %arg33[%116] : memref<128xf32, 1>, index -> <f32>
                  %118 = ascvec.loadalign_brc %arg34[%116] : memref<128xf32, 1>, index -> <f32>
                  %119 = ascvec.exp_sub %117, %118, %mask : <f32>, <f32> -> <f32>
                  %120 = ascvec.loadalign_brc %arg35[%116] : memref<128xf32, 1>, index -> <f32>
                  %121 = ascvec.loadalign_brc %arg36[%116] : memref<128xf32, 1>, index -> <f32>
                  %122 = ascvec.madd_two_address %120, %119, %121, %mask : <f32>
                  ascvec.storealign_1st %arg37[%116], %119 : memref<128xf32, 1>, index, <f32>
                  ascvec.storealign_1st %arg33[%116], %118 : memref<128xf32, 1>, index, <f32>
                  ascvec.storealign_1st %arg35[%116], %122 : memref<128xf32, 1>, index, <f32>
                }
              } {cannir.vf_mode = "simd"}
              scf.yield %110 : i64
            }
            %90 = ascscalar.memref_address %alloc_24 : memref<4096xbf16, 2>
            %91 = ascscalar.memref_address %alloc_23 : memref<4096xbf16, 2>
            %92 = arith.select %81, %90, %91 : i64
            %93 = arith.select %81, %c26_i64, %c27_i64 : i64
            %94 = ascscalar.bind_local_ptr %84 : memref<1xbf16, 1>
            %95 = ascscalar.bind_local_ptr %92 : memref<1xbf16, 2>
            %96 = arith.muli %1, %c16_i64 : i64
            %97 = arith.index_cast %96 : i64 to index
            %98 = arith.muli %97, %c16 : index
            %99 = arith.cmpi eq, %93, %c27_i64 : i64
            %100 = arith.select %99, %c6_i64, %c5_i64 : i64
            ascvec.sync_intra_wait mte3, %100 {cannir.sync.banks = 3 : i32}
            ascvec.get_buf mte3, %86, 0 {cannir.buffer.generated_sync}
            ascvec.copy_ub2l1_strided %95[%98], %94[%c0], %c8_i32, %c16_i32, %c0_i32, %c16_i32 : memref<1xbf16, 2>, memref<1xbf16, 1>
            ascvec.release_buf mte3, %86, 0 {cannir.buffer.generated_sync}
            ascvec.sync_intra_arrive mte3, %100 {cannir.sync.banks = 3 : i32}
            scf.yield %65, %89, %72, %88 : i64, i64, i64, i64
          } else {
            scf.yield %arg23, %arg27, %arg29, %arg30 : i64, i64, i64, i64
          }
          %51 = arith.cmpi sge, %arg21, %c2_i64 : i64
          %52 = arith.addi %27, %c2_i64 : i64
          %53 = arith.cmpi slt, %arg21, %52 : i64
          %54 = arith.select %51, %53, %false : i1
          %55:4 = scf.if %54 -> (i64, i64, i64, i64) {
            %58 = arith.addi %arg31, %c1_i64 : i64
            %59 = arith.remui %58, %c2_i64 : i64
            %60 = arith.addi %arg24, %c1_i64 : i64
            %61 = arith.remui %60, %c3_i64 : i64
            %62 = arith.addi %45#1, %c1_i64 : i64
            %63 = arith.remui %62, %c2_i64 : i64
            %64 = arith.addi %45#2, %c1_i64 : i64
            %65 = arith.remui %64, %c2_i64 : i64
            scf.yield %61, %63, %65, %59 : i64, i64, i64, i64
          } else {
            scf.yield %arg24, %45#1, %45#2, %arg31 : i64, i64, i64, i64
          }
          %56 = arith.cmpi sge, %arg21, %c3_i64 : i64
          %57:2 = scf.if %56 -> (i64, i64) {
            %58 = arith.cmpi eq, %arg21, %c3_i64 : i64
            %59 = arith.cmpi eq, %arg32, %c0_i64 : i64
            %60 = ascscalar.memref_address %alloc_21 : memref<2048xf32, 1>
            %61 = ascscalar.memref_address %alloc_22 : memref<2048xf32, 1>
            %62 = arith.select %59, %60, %61 : i64
            %63 = ascscalar.bind_local_ptr %62 : memref<2048xf32, 1>
            %64 = arith.select %59, %c24_i64, %c25_i64 : i64
            %65 = arith.addi %arg32, %c1_i64 : i64
            %66 = arith.remui %65, %c2_i64 : i64
            %67 = scf.if %58 -> (i64) {
              %68 = arith.cmpi eq, %64, %c25_i64 : i64
              %69 = arith.select %68, %c4_i64, %c3_i64 : i64
              ascvec.sync_intra_wait v, %69 {cannir.sync.banks = 3 : i32}
              ascvec.vf(%63, %alloc_11, %38 : memref<2048xf32, 1>, memref<2048xf32, 1>, i64) {
              ^bb0(%arg33: memref<2048xf32, 1>, %arg34: memref<2048xf32, 1>, %arg35: i64):
                %c1_i64_25 = arith.constant 1 : i64
                %c0_i64_26 = arith.constant 0 : i64
                %c1_i32_27 = arith.constant 1 : i32
                %c2_i32 = arith.constant 2 : i32
                %c0_i32_28 = arith.constant 0 : i32
                %c128_i64_29 = arith.constant 128 : i64
                %c64_i32 = arith.constant 64 : i32
                %mask, %remaining_out = ascvec.update_mask %c64_i32 : !ascvec.mask
                scf.for %arg36 = %c0_i64_26 to %arg35 step %c1_i64_25  : i64 {
                  scf.for %arg37 = %c0_i32_28 to %c2_i32 step %c1_i32_27  : i32 {
                    %70 = arith.muli %arg36, %c128_i64_29 : i64
                    %71 = arith.muli %arg37, %c64_i32 overflow<nsw> : i32
                    %72 = arith.extsi %71 : i32 to i64
                    %73 = arith.addi %70, %72 : i64
                    %74 = arith.index_cast %73 : i64 to index
                    %75 = ascvec.loadalign %arg33[%74] : memref<2048xf32, 1>, index -> <f32>
                    ascvec.storealign %arg34[%74], %75, %mask : memref<2048xf32, 1>, index, <f32>
                  }
                }
              } {cannir.vf_mode = "simd"}
              ascvec.sync_intra_arrive v, %69 {cannir.sync.banks = 3 : i32}
              scf.yield %arg28 : i64
            } else {
              %68 = arith.cmpi eq, %arg28, %c1_i64 : i64
              %69 = ascscalar.memref_address %alloc_9 : memref<128xf32, 1>
              %70 = ascscalar.memref_address %alloc_10 : memref<128xf32, 1>
              %71 = arith.select %68, %69, %70 : i64
              %72 = arith.cmpi eq, %arg28, %c0_i64 : i64
              %73 = ascscalar.memref_address %alloc_8 : memref<128xf32, 1>
              %74 = arith.select %72, %73, %71 : i64
              %75 = ascscalar.bind_local_ptr %74 : memref<128xf32, 1>
              %76 = arith.addi %arg28, %c1_i64 : i64
              %77 = arith.remui %76, %c3_i64 : i64
              %78 = arith.cmpi eq, %64, %c25_i64 : i64
              %79 = arith.select %78, %c4_i64, %c3_i64 : i64
              ascvec.sync_intra_wait v, %79 {cannir.sync.banks = 3 : i32}
              ascvec.vf(%75, %63, %alloc_11, %38 : memref<128xf32, 1>, memref<2048xf32, 1>, memref<2048xf32, 1>, i64) {
              ^bb0(%arg33: memref<128xf32, 1>, %arg34: memref<2048xf32, 1>, %arg35: memref<2048xf32, 1>, %arg36: i64):
                %c1_i64_25 = arith.constant 1 : i64
                %c0_i64_26 = arith.constant 0 : i64
                %c1_i32_27 = arith.constant 1 : i32
                %c2_i32 = arith.constant 2 : i32
                %c0_i32_28 = arith.constant 0 : i32
                %c128_i64_29 = arith.constant 128 : i64
                %c8_i64_30 = arith.constant 8 : i64
                %c64_i32 = arith.constant 64 : i32
                %mask, %remaining_out = ascvec.update_mask %c64_i32 : !ascvec.mask
                scf.for %arg37 = %c0_i64_26 to %arg36 step %c1_i64_25  : i64 {
                  %80 = arith.muli %arg37, %c8_i64_30 : i64
                  %81 = arith.index_cast %80 : i64 to index
                  %82 = ascvec.loadalign_brc %arg33[%81] : memref<128xf32, 1>, index -> <f32>
                  scf.for %arg38 = %c0_i32_28 to %c2_i32 step %c1_i32_27  : i32 {
                    %83 = arith.muli %arg37, %c128_i64_29 : i64
                    %84 = arith.muli %arg38, %c64_i32 overflow<nsw> : i32
                    %85 = arith.extsi %84 : i32 to i64
                    %86 = arith.addi %83, %85 : i64
                    %87 = arith.index_cast %86 : i64 to index
                    %88 = ascvec.loadalign %arg34[%87] : memref<2048xf32, 1>, index -> <f32>
                    %89 = ascvec.loadalign %arg35[%87] : memref<2048xf32, 1>, index -> <f32>
                    %90 = ascvec.madd_two_address %89, %82, %88, %mask : <f32>
                    ascvec.storealign %arg35[%87], %90, %mask : memref<2048xf32, 1>, index, <f32>
                  }
                }
              } {cannir.vf_mode = "simd"}
              ascvec.sync_intra_arrive v, %79 {cannir.sync.banks = 3 : i32}
              scf.yield %77 : i64
            }
            scf.yield %67, %66 : i64, i64
          } else {
            scf.yield %arg28, %arg32 : i64, i64
          }
          scf.yield %45#0, %50#0, %55#0, %55#1, %55#2, %50#1, %57#0, %50#2, %50#3, %55#3, %57#1 : i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64
        }
        ascvec.get_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
        ascvec.vf(%alloc_5, %alloc_11, %alloc_17, %38 : memref<128xf32, 1>, memref<2048xf32, 1>, memref<2048xbf16, 1>, i64) {
        ^bb0(%arg21: memref<128xf32, 1>, %arg22: memref<2048xf32, 1>, %arg23: memref<2048xbf16, 1>, %arg24: i64):
          %c1_i64_25 = arith.constant 1 : i64
          %c0_i64_26 = arith.constant 0 : i64
          %c64_i64 = arith.constant 64 : i64
          %c128_i64_27 = arith.constant 128 : i64
          %c8_i64_28 = arith.constant 8 : i64
          %c128_i32 = arith.constant 128 : i32
          %c64_i32 = arith.constant 64 : i32
          %mask, %remaining_out = ascvec.update_mask %c64_i32 : !ascvec.mask
          %mask_29, %remaining_out_30 = ascvec.update_mask %c128_i32 elem_bits 16 : !ascvec.mask
          scf.for %arg25 = %c0_i64_26 to %arg24 step %c1_i64_25  : i64 {
            %44 = arith.muli %arg25, %c8_i64_28 : i64
            %45 = arith.index_cast %44 : i64 to index
            %46 = ascvec.loadalign_brc %arg21[%45] : memref<128xf32, 1>, index -> <f32>
            %47 = arith.muli %arg25, %c128_i64_27 : i64
            %48 = arith.index_cast %47 : i64 to index
            %49 = ascvec.loadalign %arg22[%48] : memref<2048xf32, 1>, index -> <f32>
            %50 = ascvec.div %49, %46, %mask : <f32>
            %51 = arith.addi %47, %c64_i64 : i64
            %52 = arith.index_cast %51 : i64 to index
            %53 = ascvec.loadalign %arg22[%52] : memref<2048xf32, 1>, index -> <f32>
            %54 = ascvec.div %53, %46, %mask : <f32>
            %55 = ascvec.cast %50, %mask {reg_layout = 0 : i32} : <f32> -> <bf16>
            %56 = ascvec.cast %54, %mask {reg_layout = 0 : i32} : <f32> -> <bf16>
            %result0, %result1 = ascvec.deintlv %55, %56 : <bf16>, <bf16> -> <bf16>, <bf16>
            ascvec.storealign %arg23[%48], %result0, %mask_29 : memref<2048xbf16, 1>, index, <bf16>
          }
        } {cannir.vf_mode = "simd"}
        ascvec.release_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
        scf.yield %43#0, %43#1, %43#2, %43#3, %43#4, %43#5, %43#6, %43#7, %43#8, %43#9, %43#10 : i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64
      } else {
        ascvec.get_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
        ascvec.vf(%alloc_17, %38 : memref<2048xbf16, 1>, i64) {
        ^bb0(%arg21: memref<2048xbf16, 1>, %arg22: i64):
          %c1_i64_25 = arith.constant 1 : i64
          %c0_i64_26 = arith.constant 0 : i64
          %c128_i64_27 = arith.constant 128 : i64
          %cst = arith.constant 0.000000e+00 : bf16
          %c128_i32 = arith.constant 128 : i32
          %mask, %remaining_out = ascvec.update_mask %c128_i32 elem_bits 16 : !ascvec.mask
          %42 = ascvec.duplicate_scalar %cst, %mask : bf16 -> <bf16>
          scf.for %arg23 = %c0_i64_26 to %arg22 step %c1_i64_25  : i64 {
            %43 = arith.muli %arg23, %c128_i64_27 : i64
            %44 = arith.index_cast %43 : i64 to index
            ascvec.storealign %arg21[%44], %42, %mask : memref<2048xbf16, 1>, index, <bf16>
          }
        } {cannir.vf_mode = "simd"}
        ascvec.release_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
        scf.yield %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18, %arg19, %arg20 : i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64
      }
      ascvec.sync_all
      scf.for %arg21 = %c0_i64 to %38 step %c1_i64  : i64 {
        %42 = arith.addi %28, %39 : i64
        %43 = arith.addi %42, %arg21 : i64
        %44 = arith.cmpi slt, %43, %c128_i64 : i64
        scf.if %44 {
          %45 = arith.muli %18, %c128_i64 : i64
          %46 = arith.addi %45, %43 : i64
          %47 = cannir.extract_physical_layout %arg0 : !cannir.tensor<<gm, bf16>, <<(-1, -1, 128)>, <(-1, 128, 1)>>, <<(-1, -1, 128)>, <(-1, 128, 1)>>, identity>
          %48 = cannir.extract_stride %47 : <<(-1, -1, 128)>, <(-1, 128, 1)>> -> <(-1, 128, 1)>
          %49 = cannir.extract_dim %48[0] : !cannir.stride<(-1, 128, 1)>
          %50 = arith.muli %12, %49 : i64
          %51 = arith.muli %46, %c128_i64 : i64
          %52 = arith.addi %50, %51 : i64
          %53 = arith.muli %52, %c16_i64 : i64
          %54 = ascscalar.memref_address %alloc : memref<1xbf16>
          %55 = arith.divui %53, %c8_i64 : i64
          %56 = arith.addi %54, %55 : i64
          %57 = ascscalar.bind_gm_ptr %56 : memref<1xbf16>
          %58 = arith.muli %arg21, %c2048_i64 : i64
          %59 = ascscalar.memref_address %alloc_17 : memref<2048xbf16, 1>
          %60 = arith.divui %58, %c8_i64 : i64
          %61 = arith.addi %59, %60 : i64
          %62 = ascscalar.bind_local_ptr %61 : memref<1xbf16, 1>
          ascvec.get_buf mte3, %c4_i64, 0 {cannir.buffer.generated_sync}
          ascvec.copy_ub2gm %57[%c0], %62[%c0], %c1_i32, %c256_i32, %c0_i64, %c256_i32 l2_cache_ctl %c0_i32 : memref<1xbf16>, memref<1xbf16, 1>
          ascvec.release_buf mte3, %c4_i64, 0 {cannir.buffer.generated_sync}
        }
      }
      scf.yield %41#0, %41#1, %41#2, %41#3, %41#4, %41#5, %41#6, %41#7, %41#8, %41#9, %41#10 : i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64
    }
    ascvec.sync_intra_wait mte3, %c5_i64
    ascvec.sync_intra_wait mte3, %c6_i64
    return
  }
  func.func @kernel_QwenSparseAttnBlock128___call___0_aic(%arg0: !cannir.tensor<<gm, bf16>, <<(-1, -1, 128)>, <(-1, 128, 1)>>, <<(-1, -1, 128)>, <(-1, 128, 1)>>, identity>, %arg1: !cannir.tensor<<gm, bf16>, <<(-1, -1, 128)>, <(-1, 128, 1)>>, <<(-1, -1, 128)>, <(-1, 128, 1)>>, identity>, %arg2: !cannir.tensor<<gm, bf16>, <<(-1, -1, 128, 128)>, <(-1, 128, -1, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 128, -1, 1)>>, identity>, %arg3: !cannir.tensor<<gm, bf16>, <<(-1, -1, 128, 128)>, <(-1, 128, -1, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 128, -1, 1)>>, identity>, %arg4: !cannir.tensor<<gm, i8>, <<(2048, 2048)>, <(2048, 1)>>, <<(2048, 2048)>, <(2048, 1)>>, identity>, %arg5: !cannir.tensor<<gm, i64>, <<(2, -1)>, <(-1, 1)>>, <<(2, -1)>, <(-1, 1)>>, identity>, %arg6: !cannir.tensor<<gm, i64>, <<(5, -1)>, <(-1, 1)>>, <<(5, -1)>, <(-1, 1)>>, identity>, %arg7: !cannir.tensor<<gm, i32>, <<(2, -1)>, <(-1, 1)>>, <<(2, -1)>, <(-1, 1)>>, identity>, %arg8: f32) attributes {cannir.buffer.auto_sync, cannir.buffer.sync_analyzed, cannir.buffer.sync_verified, cannir.buffer.tensor_state, cannir.core = "aic", cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.device, cannir.explicit_ring_buffer_cursors, cannir.kernel_group = "kernel_QwenSparseAttnBlock128___call___0", cannir.resource.ids_prepared} {
    %c8_i32 = arith.constant 8 : i32
    %c32_i32 = arith.constant 32 : i32
    %c4 = arith.constant 4 : index
    %c3 = arith.constant 3 : index
    %c16_i64 = arith.constant 16 : i64
    %c6_i64 = arith.constant 6 : i64
    %c5_i64 = arith.constant 5 : i64
    %c4_i64 = arith.constant 4 : i64
    %c3_i64 = arith.constant 3 : i64
    %c2_i64 = arith.constant 2 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %c7_i64 = arith.constant 7 : i64
    %c8_i64 = arith.constant 8 : i64
    %c9_i64 = arith.constant 9 : i64
    %c10_i64 = arith.constant 10 : i64
    %c11_i64 = arith.constant 11 : i64
    %c12_i64 = arith.constant 12 : i64
    %c17_i64 = arith.constant 17 : i64
    %c18_i64 = arith.constant 18 : i64
    %c19_i64 = arith.constant 19 : i64
    %c20_i64 = arith.constant 20 : i64
    %c21_i64 = arith.constant 21 : i64
    %c22_i64 = arith.constant 22 : i64
    %c23_i64 = arith.constant 23 : i64
    %c24_i64 = arith.constant 24 : i64
    %c25_i64 = arith.constant 25 : i64
    %c26_i64 = arith.constant 26 : i64
    %c27_i64 = arith.constant 27 : i64
    %c256_i64 = arith.constant 256 : i64
    %c2 = arith.constant 2 : index
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0 = arith.constant 0 : index
    %c128_i32 = arith.constant 128 : i32
    %false = arith.constant false
    %c0_i32 = arith.constant 0 : i32
    %c128_i64 = arith.constant 128 : i64
    %c32_i64 = arith.constant 32 : i64
    %alloc = memref.alloc() {cannir.arg_index = 1 : i64} : memref<1xbf16>
    %alloc_0 = memref.alloc() {cannir.arg_index = 2 : i64} : memref<1xbf16>
    %alloc_1 = memref.alloc() {cannir.arg_index = 3 : i64} : memref<1xbf16>
    %alloc_2 = memref.alloc() {cannir.arg_index = 5 : i64} : memref<1xi64>
    %alloc_3 = memref.alloc() {cannir.arg_index = 6 : i64} : memref<1xi64>
    %alloc_4 = memref.alloc() {cannir.arg_index = 7 : i64} : memref<1xi32>
    %alloc_5 = memref.alloc() {cannir.byte_offset = 30976 : i64} : memref<2048xf32, 1>
    %alloc_6 = memref.alloc() {cannir.byte_offset = 39168 : i64} : memref<2048xf32, 1>
    %alloc_7 = memref.alloc() {cannir.byte_offset = 47360 : i64} : memref<2048xf32, 1>
    %alloc_8 = memref.alloc() {cannir.byte_offset = 55552 : i64} : memref<2048xf32, 1>
    %alloc_9 = memref.alloc() {cannir.byte_offset = 63744 : i64} : memref<2048xf32, 1>
    %alloc_10 = memref.alloc() {cannir.byte_offset = 16384 : i64} : memref<4096xf32, 5>
    %alloc_11 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<4096xf32, 5>
    %alloc_12 = memref.alloc() {cannir.byte_offset = 32768 : i64} : memref<16384xbf16, 4>
    %alloc_13 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<16384xbf16, 4>
    %alloc_14 = memref.alloc() {cannir.byte_offset = 8192 : i64} : memref<4096xbf16, 3>
    %alloc_15 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<4096xbf16, 3>
    %alloc_16 = memref.alloc() {cannir.byte_offset = 106496 : i64} : memref<16384xbf16, 2>
    %alloc_17 = memref.alloc() {cannir.byte_offset = 172032 : i64} : memref<16384xbf16, 2>
    %alloc_18 = memref.alloc() {cannir.byte_offset = 139264 : i64} : memref<16384xbf16, 2>
    %alloc_19 = memref.alloc() {cannir.byte_offset = 212992 : i64} : memref<4096xbf16, 2>
    %alloc_20 = memref.alloc() {cannir.byte_offset = 204800 : i64} : memref<4096xbf16, 2>
    %alloc_21 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<4096xbf16, 2>
    %alloc_22 = memref.alloc() {cannir.byte_offset = 8192 : i64} : memref<16384xbf16, 2>
    %alloc_23 = memref.alloc() {cannir.byte_offset = 73728 : i64} : memref<16384xbf16, 2>
    %alloc_24 = memref.alloc() {cannir.byte_offset = 40960 : i64} : memref<16384xbf16, 2>
    asccube.sync_intra_arrive mte1, %c6_i64
    asccube.sync_intra_arrive mte1, %c22_i64
    asccube.sync_intra_arrive mte1, %c5_i64
    asccube.sync_intra_arrive mte1, %c21_i64
    %0 = ascscalar.get_block_idx
    %1 = ascscalar.get_subblock_id
    %2 = arith.index_cast %0 : i64 to index
    %3 = memref.load %alloc_2[%2] : memref<1xi64>
    %4 = cannir.extract_layout %arg5 : !cannir.tensor<<gm, i64>, <<(2, -1)>, <(-1, 1)>>, <<(2, -1)>, <(-1, 1)>>, identity>
    %5 = cannir.extract_stride %4 : <<(2, -1)>, <(-1, 1)>> -> <(-1, 1)>
    %6 = cannir.extract_dim %5[0] : !cannir.stride<(-1, 1)>
    %7 = arith.index_cast %6 : i64 to index
    %8 = arith.addi %7, %2 : index
    %9 = memref.load %alloc_2[%8] : memref<1xi64>
    %10:11 = scf.for %arg9 = %3 to %9 step %c1_i64 iter_args(%arg10 = %c0_i64, %arg11 = %c0_i64, %arg12 = %c0_i64, %arg13 = %c0_i64, %arg14 = %c0_i64, %arg15 = %c0_i64, %arg16 = %c0_i64, %arg17 = %c0_i64, %arg18 = %c0_i64, %arg19 = %c0_i64, %arg20 = %c0_i64) -> (i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64)  : i64 {
      %11 = arith.index_cast %arg9 : i64 to index
      %12 = memref.load %alloc_3[%11] : memref<1xi64>
      %13 = cannir.extract_layout %arg6 : !cannir.tensor<<gm, i64>, <<(5, -1)>, <(-1, 1)>>, <<(5, -1)>, <(-1, 1)>>, identity>
      %14 = cannir.extract_stride %13 : <<(5, -1)>, <(-1, 1)>> -> <(-1, 1)>
      %15 = cannir.extract_dim %14[0] : !cannir.stride<(-1, 1)>
      %16 = arith.index_cast %15 : i64 to index
      %17 = arith.addi %16, %11 : index
      %18 = memref.load %alloc_3[%17] : memref<1xi64>
      %19 = arith.muli %16, %c2 : index
      %20 = arith.addi %19, %11 : index
      %21 = memref.load %alloc_3[%20] : memref<1xi64>
      %22 = arith.muli %16, %c3 : index
      %23 = arith.addi %22, %11 : index
      %24 = memref.load %alloc_3[%23] : memref<1xi64>
      %25 = arith.muli %16, %c4 : index
      %26 = arith.addi %25, %11 : index
      %27 = memref.load %alloc_3[%26] : memref<1xi64>
      %28 = arith.muli %21, %c32_i64 : i64
      %29 = arith.subi %c128_i64, %28 : i64
      %30 = arith.cmpi sgt, %29, %c32_i64 : i64
      %31 = arith.cmpi sgt, %29, %c0_i64 : i64
      %32 = arith.select %31, %29, %c0_i64 : i64
      %33 = arith.select %30, %c32_i64, %32 : i64
      %34 = arith.addi %33, %c1_i64 : i64
      %35 = arith.divsi %34, %c2_i64 : i64
      %36 = arith.divsi %33, %c2_i64 : i64
      %37 = arith.cmpi eq, %1, %c0_i64 : i64
      %38 = arith.select %37, %35, %36 : i64
      %39 = arith.select %37, %c0_i64, %35 : i64
      %40 = arith.cmpi sgt, %27, %c0_i64 : i64
      %41:11 = scf.if %40 -> (i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) {
        %42 = arith.muli %18, %c128_i64 : i64
        %43 = arith.addi %42, %28 : i64
        %44 = cannir.extract_layout %arg1 : !cannir.tensor<<gm, bf16>, <<(-1, -1, 128)>, <(-1, 128, 1)>>, <<(-1, -1, 128)>, <(-1, 128, 1)>>, identity>
        %45 = cannir.extract_shape %44 : <<(-1, -1, 128)>, <(-1, 128, 1)>> -> <(-1, -1, 128)>
        %46 = cannir.extract_dim %45[1] : !cannir.shape<(-1, -1, 128)>
        %47 = arith.subi %46, %43 : i64
        %48 = ascscalar.bitcast_memref %alloc_21 byte_offset = %c0_i64 : memref<4096xbf16, 2> -> memref<?xbf16, 2>
        %49 = cannir.extract_physical_layout %arg1 : !cannir.tensor<<gm, bf16>, <<(-1, -1, 128)>, <(-1, 128, 1)>>, <<(-1, -1, 128)>, <(-1, 128, 1)>>, identity>
        %50 = cannir.extract_stride %49 : <<(-1, -1, 128)>, <(-1, 128, 1)>> -> <(-1, 128, 1)>
        %51 = cannir.extract_dim %50[0] : !cannir.stride<(-1, 128, 1)>
        %52 = arith.muli %12, %51 : i64
        %53 = arith.index_cast %52 : i64 to index
        %54 = arith.muli %43, %c128_i64 : i64
        %55 = arith.index_cast %54 : i64 to index
        %56 = arith.addi %53, %55 : index
        %57 = arith.divsi %47, %c32_i64 : i64
        %58 = arith.minsi %57, %c0_i64 : i64
        %59 = arith.muli %58, %c32_i64 : i64
        %60 = arith.subi %47, %59 : i64
        %61 = arith.minsi %60, %c32_i64 : i64
        %62 = arith.cmpi sge, %57, %c0_i64 : i64
        %63 = arith.select %62, %61, %c0_i64 : i64
        asccube.get_buf mte2, %c0_i64, 0 {cannir.buffer.generated_sync}
        %64 = arith.cmpi ugt, %63, %c0_i64 : i64
        scf.if %64 {
          %67 = arith.trunci %63 : i64 to i32
          asccube.copy_gm2l1 %48[%c0], %alloc[%56], %67, %c128_i32, %c256_i64, %c1_i32, %c0_i64, %c0_i64, %c32_i32 l2_cache_ctl %c0_i32 {nd2nz = true} : memref<?xbf16, 2>, memref<1xbf16>
        }
        asccube.release_buf mte2, %c0_i64, 0 {cannir.buffer.generated_sync}
        %65 = arith.addi %27, %c3_i64 : i64
        %66:11 = scf.for %arg21 = %c0_i64 to %65 step %c1_i64 iter_args(%arg22 = %arg10, %arg23 = %arg11, %arg24 = %arg12, %arg25 = %arg13, %arg26 = %arg14, %arg27 = %arg15, %arg28 = %arg16, %arg29 = %arg17, %arg30 = %arg18, %arg31 = %arg19, %arg32 = %arg20) -> (i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64)  : i64 {
          %67 = arith.cmpi slt, %arg21, %27 : i64
          %68:7 = scf.if %67 -> (i64, i64, i64, i64, i64, i64, i64) {
            %81 = arith.addi %24, %arg21 : i64
            %82 = arith.index_cast %81 : i64 to index
            %83 = memref.load %alloc_4[%82] : memref<1xi32>
            %84 = arith.extsi %83 : i32 to i64
            %85 = arith.cmpi eq, %arg22, %c1_i64 : i64
            %86 = ascscalar.memref_address %alloc_24 : memref<16384xbf16, 2>
            %87 = ascscalar.memref_address %alloc_23 : memref<16384xbf16, 2>
            %88 = arith.select %85, %86, %87 : i64
            %89 = arith.cmpi eq, %arg22, %c0_i64 : i64
            %90 = ascscalar.memref_address %alloc_22 : memref<16384xbf16, 2>
            %91 = arith.select %89, %90, %88 : i64
            %92 = arith.select %85, %c2_i64, %c3_i64 : i64
            %93 = arith.select %89, %c1_i64, %92 : i64
            %94 = arith.addi %arg22, %c1_i64 : i64
            %95 = arith.remui %94, %c3_i64 : i64
            %96 = ascscalar.bind_local_ptr %91 : memref<1xbf16, 2>
            %97 = cannir.extract_physical_layout %arg2 : !cannir.tensor<<gm, bf16>, <<(-1, -1, 128, 128)>, <(-1, 128, -1, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 128, -1, 1)>>, identity>
            %98 = cannir.extract_stride %97 : <<(-1, -1, 128, 128)>, <(-1, 128, -1, 1)>> -> <(-1, 128, -1, 1)>
            %99 = cannir.extract_dim %98[0] : !cannir.stride<(-1, 128, -1, 1)>
            %100 = arith.muli %84, %99 : i64
            %101 = arith.index_cast %100 : i64 to index
            %102 = arith.index_cast %42 : i64 to index
            %103 = arith.addi %101, %102 : index
            %104 = cannir.extract_layout %arg2 : !cannir.tensor<<gm, bf16>, <<(-1, -1, 128, 128)>, <(-1, 128, -1, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 128, -1, 1)>>, identity>
            %105 = cannir.extract_stride %104 : <<(-1, -1, 128, 128)>, <(-1, 128, -1, 1)>> -> <(-1, 128, -1, 1)>
            %106 = cannir.extract_dim %105[2] : !cannir.stride<(-1, 128, -1, 1)>
            %107 = arith.index_cast %106 : i64 to index
            %108 = arith.muli %107, %c2 : index
            %109 = arith.index_cast %108 : index to i64
            asccube.get_buf mte2, %93, 0 {cannir.buffer.generated_sync}
            asccube.copy_gm2l1 %96[%c0], %alloc_0[%103], %c128_i32, %c128_i32, %109, %c1_i32, %c0_i64, %c0_i64, %c128_i32 l2_cache_ctl %c0_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
            asccube.release_buf mte2, %93, 0 {cannir.buffer.generated_sync}
            %110 = arith.cmpi eq, %arg26, %c0_i64 : i64
            %111 = ascscalar.memref_address %alloc_11 : memref<4096xf32, 5>
            %112 = ascscalar.memref_address %alloc_10 : memref<4096xf32, 5>
            %113 = arith.select %110, %111, %112 : i64
            %114 = arith.select %110, %c11_i64, %c12_i64 : i64
            %115 = arith.addi %arg26, %c1_i64 : i64
            %116 = arith.remui %115, %c2_i64 : i64
            %117 = arith.cmpi eq, %arg25, %c0_i64 : i64
            %118 = ascscalar.memref_address %alloc_15 : memref<4096xbf16, 3>
            %119 = ascscalar.memref_address %alloc_14 : memref<4096xbf16, 3>
            %120 = arith.select %117, %118, %119 : i64
            %121 = arith.select %117, %c7_i64, %c8_i64 : i64
            %122 = arith.addi %arg25, %c1_i64 : i64
            %123 = arith.remui %122, %c2_i64 : i64
            %124 = ascscalar.bind_local_ptr %120 : memref<1xbf16, 3>
            asccube.get_buf mte1, %121, 0 {cannir.buffer.generated_sync}
            asccube.get_buf mte1, %c0_i64, 0 {cannir.buffer.generated_sync}
            asccube.copy_l12l0a %124[%c0], %48[%c0], %c0_i32, %c0_i32, %c2_i32, %c8_i32, %c2_i32, %c2_i32 : memref<1xbf16, 3>, memref<?xbf16, 2>
            asccube.release_buf mte1, %c0_i64, 0 {cannir.buffer.generated_sync}
            asccube.release_buf mte1, %121, 0 {cannir.buffer.generated_sync}
            %125 = ascscalar.memref_address %alloc_13 : memref<16384xbf16, 4>
            %126 = ascscalar.memref_address %alloc_12 : memref<16384xbf16, 4>
            %127 = arith.select %117, %125, %126 : i64
            %128 = arith.select %117, %c9_i64, %c10_i64 : i64
            %129 = ascscalar.bind_local_ptr %127 : memref<1xbf16, 4>
            asccube.get_buf mte1, %128, 0 {cannir.buffer.generated_sync}
            asccube.get_buf mte1, %93, 0 {cannir.buffer.generated_sync}
            asccube.copy_l12l0b %129[%c0], %96[%c0], %c0_i32, %c0_i32, %c8_i32, %c8_i32, %c8_i32, %c8_i32 : memref<1xbf16, 4>, memref<1xbf16, 2>
            asccube.release_buf mte1, %93, 0 {cannir.buffer.generated_sync}
            asccube.release_buf mte1, %128, 0 {cannir.buffer.generated_sync}
            %130 = ascscalar.bind_local_ptr %113 : memref<1xf32, 5>
            asccube.get_buf m, %114, 0 {cannir.buffer.generated_sync}
            asccube.get_buf m, %121, 0 {cannir.buffer.generated_sync}
            asccube.get_buf m, %128, 0 {cannir.buffer.generated_sync}
            asccube.mmad %130[%c0], %124, %129, %c32_i32, %c128_i32, %c128_i32, %c0_i32 {c_matrix_init_val = true} : memref<1xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
            asccube.release_buf m, %128, 0 {cannir.buffer.generated_sync}
            asccube.release_buf m, %121, 0 {cannir.buffer.generated_sync}
            asccube.release_buf m, %114, 0 {cannir.buffer.generated_sync}
            %131 = ascscalar.memref_address %alloc_6 : memref<2048xf32, 1>
            %132 = ascscalar.memref_address %alloc_7 : memref<2048xf32, 1>
            %133 = arith.select %85, %131, %132 : i64
            %134 = ascscalar.memref_address %alloc_5 : memref<2048xf32, 1>
            %135 = arith.select %89, %134, %133 : i64
            %136 = arith.select %85, %c22_i64, %c23_i64 : i64
            %137 = arith.select %89, %c21_i64, %136 : i64
            %138 = ascscalar.bind_local_ptr %135 : memref<1xf32, 1>
            %139 = arith.cmpi eq, %137, %c22_i64 : i64
            %140 = arith.extui %139 : i1 to i64
            %141 = arith.cmpi eq, %137, %c23_i64 : i64
            %142 = arith.select %141, %c2_i64, %140 : i64
            asccube.sync_intra_wait fixpipe, %142 {cannir.sync.banks = 3 : i32}
            %143 = arith.addi %142, %c16_i64 overflow<nsw> : i64
            asccube.sync_intra_wait fixpipe, %143 {cannir.sync.banks = 3 : i32}
            asccube.get_buf fixpipe, %114, 0 {cannir.buffer.generated_sync}
            asccube.copy_l0c2ub %138[%c0], %130[%c0], %c128_i32, %c32_i32, %c128_i32, %c32_i32, %c0_i32 sub_block_id %c0_i32 {dual_dst_ctl = 1 : i32, nz2nd_en = true, relu_en = false} : memref<1xf32, 1>, memref<1xf32, 5>
            asccube.release_buf fixpipe, %114, 0 {cannir.buffer.generated_sync}
            asccube.sync_intra_arrive fixpipe, %142 {cannir.sync.banks = 3 : i32}
            asccube.sync_intra_arrive fixpipe, %143 {cannir.sync.banks = 3 : i32}
            scf.yield %95, %123, %123, %123, %123, %116, %116 : i64, i64, i64, i64, i64, i64, i64
          } else {
            scf.yield %arg22, %arg25, %arg25, %arg25, %arg25, %arg26, %arg26 : i64, i64, i64, i64, i64, i64, i64
          }
          %69 = arith.cmpi sge, %arg21, %c1_i64 : i64
          %70 = arith.addi %27, %c1_i64 : i64
          %71 = arith.cmpi slt, %arg21, %70 : i64
          %72 = arith.select %69, %71, %false : i1
          %73:4 = scf.if %72 -> (i64, i64, i64, i64) {
            %81 = arith.addi %24, %arg21 : i64
            %82 = arith.subi %81, %c1_i64 : i64
            %83 = arith.index_cast %82 : i64 to index
            %84 = memref.load %alloc_4[%83] : memref<1xi32>
            %85 = arith.extsi %84 : i32 to i64
            %86 = arith.cmpi eq, %arg21, %c1_i64 : i64
            %87 = arith.cmpi eq, %arg23, %c1_i64 : i64
            %88 = ascscalar.memref_address %alloc_18 : memref<16384xbf16, 2>
            %89 = ascscalar.memref_address %alloc_17 : memref<16384xbf16, 2>
            %90 = arith.select %87, %88, %89 : i64
            %91 = arith.cmpi eq, %arg23, %c0_i64 : i64
            %92 = ascscalar.memref_address %alloc_16 : memref<16384xbf16, 2>
            %93 = arith.select %91, %92, %90 : i64
            %94 = arith.select %87, %c5_i64, %c6_i64 : i64
            %95 = arith.select %91, %c4_i64, %94 : i64
            %96 = arith.addi %arg23, %c1_i64 : i64
            %97 = arith.remui %96, %c3_i64 : i64
            %98 = ascscalar.bind_local_ptr %93 : memref<1xbf16, 2>
            %99 = cannir.extract_physical_layout %arg3 : !cannir.tensor<<gm, bf16>, <<(-1, -1, 128, 128)>, <(-1, 128, -1, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 128, -1, 1)>>, identity>
            %100 = cannir.extract_stride %99 : <<(-1, -1, 128, 128)>, <(-1, 128, -1, 1)>> -> <(-1, 128, -1, 1)>
            %101 = cannir.extract_dim %100[0] : !cannir.stride<(-1, 128, -1, 1)>
            %102 = arith.muli %85, %101 : i64
            %103 = arith.index_cast %102 : i64 to index
            %104 = arith.index_cast %42 : i64 to index
            %105 = arith.addi %103, %104 : index
            %106 = cannir.extract_layout %arg3 : !cannir.tensor<<gm, bf16>, <<(-1, -1, 128, 128)>, <(-1, 128, -1, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 128, -1, 1)>>, identity>
            %107 = cannir.extract_stride %106 : <<(-1, -1, 128, 128)>, <(-1, 128, -1, 1)>> -> <(-1, 128, -1, 1)>
            %108 = cannir.extract_dim %107[2] : !cannir.stride<(-1, 128, -1, 1)>
            %109 = arith.index_cast %108 : i64 to index
            %110 = arith.muli %109, %c2 : index
            %111 = arith.index_cast %110 : index to i64
            asccube.get_buf mte2, %95, 0 {cannir.buffer.generated_sync}
            asccube.copy_gm2l1 %98[%c0], %alloc_1[%105], %c128_i32, %c128_i32, %111, %c1_i32, %c0_i64, %c0_i64, %c128_i32 l2_cache_ctl %c0_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
            asccube.release_buf mte2, %95, 0 {cannir.buffer.generated_sync}
            %112 = arith.addi %39, %38 : i64
            %113 = arith.maxsi %39, %c0_i64 : i64
            %114 = arith.minsi %112, %c128_i64 : i64
            %115 = arith.subi %114, %113 : i64
            %116 = arith.maxsi %115, %c0_i64 : i64
            %117 = arith.cmpi sgt, %116, %c0_i64 : i64
            %118 = scf.if %117 -> (i64) {
              %122 = arith.addi %arg29, %c1_i64 : i64
              %123 = arith.remui %122, %c2_i64 : i64
              scf.yield %123 : i64
            } else {
              scf.yield %arg29 : i64
            }
            %119 = arith.addi %arg30, %c1_i64 : i64
            %120 = arith.remui %119, %c2_i64 : i64
            %121 = scf.if %86 -> (i64) {
              scf.yield %arg27 : i64
            } else {
              %122 = arith.addi %arg27, %c1_i64 : i64
              %123 = arith.remui %122, %c3_i64 : i64
              scf.yield %123 : i64
            }
            scf.yield %97, %121, %118, %120 : i64, i64, i64, i64
          } else {
            scf.yield %arg23, %arg27, %arg29, %arg30 : i64, i64, i64, i64
          }
          %74 = arith.cmpi sge, %arg21, %c2_i64 : i64
          %75 = arith.addi %27, %c2_i64 : i64
          %76 = arith.cmpi slt, %arg21, %75 : i64
          %77 = arith.select %74, %76, %false : i1
          %78:4 = scf.if %77 -> (i64, i64, i64, i64) {
            %81 = arith.cmpi eq, %arg31, %c0_i64 : i64
            %82 = ascscalar.memref_address %alloc_20 : memref<4096xbf16, 2>
            %83 = ascscalar.memref_address %alloc_19 : memref<4096xbf16, 2>
            %84 = arith.select %81, %82, %83 : i64
            %85 = arith.select %81, %c26_i64, %c27_i64 : i64
            %86 = arith.addi %arg31, %c1_i64 : i64
            %87 = arith.remui %86, %c2_i64 : i64
            %88 = arith.cmpi eq, %arg24, %c1_i64 : i64
            %89 = ascscalar.memref_address %alloc_18 : memref<16384xbf16, 2>
            %90 = ascscalar.memref_address %alloc_17 : memref<16384xbf16, 2>
            %91 = arith.select %88, %89, %90 : i64
            %92 = arith.cmpi eq, %arg24, %c0_i64 : i64
            %93 = ascscalar.memref_address %alloc_16 : memref<16384xbf16, 2>
            %94 = arith.select %92, %93, %91 : i64
            %95 = arith.select %88, %c5_i64, %c6_i64 : i64
            %96 = arith.select %92, %c4_i64, %95 : i64
            %97 = arith.addi %arg24, %c1_i64 : i64
            %98 = arith.remui %97, %c3_i64 : i64
            %99 = arith.cmpi eq, %68#1, %c0_i64 : i64
            %100 = ascscalar.memref_address %alloc_15 : memref<4096xbf16, 3>
            %101 = ascscalar.memref_address %alloc_14 : memref<4096xbf16, 3>
            %102 = arith.select %99, %100, %101 : i64
            %103 = arith.select %99, %c7_i64, %c8_i64 : i64
            %104 = arith.addi %68#1, %c1_i64 : i64
            %105 = arith.remui %104, %c2_i64 : i64
            %106 = ascscalar.bind_local_ptr %102 : memref<1xbf16, 3>
            %107 = ascscalar.bind_local_ptr %84 : memref<1xbf16, 2>
            %108 = arith.cmpi eq, %85, %c27_i64 : i64
            %109 = arith.select %108, %c6_i64, %c5_i64 : i64
            asccube.sync_intra_wait mte1, %109 {cannir.sync.banks = 3 : i32}
            %110 = arith.addi %109, %c16_i64 overflow<nsw> : i64
            asccube.sync_intra_wait mte1, %110 {cannir.sync.banks = 3 : i32}
            asccube.get_buf mte1, %103, 0 {cannir.buffer.generated_sync}
            asccube.copy_l12l0a %106[%c0], %107[%c0], %c0_i32, %c0_i32, %c2_i32, %c8_i32, %c2_i32, %c2_i32 : memref<1xbf16, 3>, memref<1xbf16, 2>
            asccube.release_buf mte1, %103, 0 {cannir.buffer.generated_sync}
            asccube.sync_intra_arrive mte1, %109 {cannir.sync.banks = 3 : i32}
            asccube.sync_intra_arrive mte1, %110 {cannir.sync.banks = 3 : i32}
            %111 = arith.cmpi eq, %68#3, %c0_i64 : i64
            %112 = ascscalar.memref_address %alloc_13 : memref<16384xbf16, 4>
            %113 = ascscalar.memref_address %alloc_12 : memref<16384xbf16, 4>
            %114 = arith.select %111, %112, %113 : i64
            %115 = arith.select %111, %c9_i64, %c10_i64 : i64
            %116 = ascscalar.bind_local_ptr %114 : memref<1xbf16, 4>
            %117 = ascscalar.bind_local_ptr %94 : memref<1xbf16, 2>
            asccube.get_buf mte1, %115, 0 {cannir.buffer.generated_sync}
            asccube.get_buf mte1, %96, 0 {cannir.buffer.generated_sync}
            asccube.copy_l12l0b %116[%c0], %117[%c0], %c0_i32, %c0_i32, %c8_i32, %c8_i32, %c8_i32, %c8_i32 {transpose = true} : memref<1xbf16, 4>, memref<1xbf16, 2>
            asccube.release_buf mte1, %96, 0 {cannir.buffer.generated_sync}
            asccube.release_buf mte1, %115, 0 {cannir.buffer.generated_sync}
            %118 = arith.cmpi eq, %68#5, %c0_i64 : i64
            %119 = ascscalar.memref_address %alloc_11 : memref<4096xf32, 5>
            %120 = ascscalar.memref_address %alloc_10 : memref<4096xf32, 5>
            %121 = arith.select %118, %119, %120 : i64
            %122 = arith.select %118, %c11_i64, %c12_i64 : i64
            %123 = arith.addi %68#5, %c1_i64 : i64
            %124 = arith.remui %123, %c2_i64 : i64
            %125 = arith.cmpi eq, %68#2, %c0_i64 : i64
            %126 = arith.select %125, %100, %101 : i64
            %127 = arith.select %125, %c7_i64, %c8_i64 : i64
            %128 = arith.cmpi eq, %68#4, %c0_i64 : i64
            %129 = arith.select %128, %112, %113 : i64
            %130 = arith.select %128, %c9_i64, %c10_i64 : i64
            %131 = ascscalar.bind_local_ptr %121 : memref<1xf32, 5>
            %132 = ascscalar.bind_local_ptr %126 : memref<1xbf16, 3>
            %133 = ascscalar.bind_local_ptr %129 : memref<1xbf16, 4>
            asccube.get_buf m, %122, 0 {cannir.buffer.generated_sync}
            asccube.get_buf m, %127, 0 {cannir.buffer.generated_sync}
            asccube.get_buf m, %130, 0 {cannir.buffer.generated_sync}
            asccube.mmad %131[%c0], %132, %133, %c32_i32, %c128_i32, %c128_i32, %c0_i32 {c_matrix_init_val = true} : memref<1xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
            asccube.release_buf m, %130, 0 {cannir.buffer.generated_sync}
            asccube.release_buf m, %127, 0 {cannir.buffer.generated_sync}
            asccube.release_buf m, %122, 0 {cannir.buffer.generated_sync}
            %134 = ascscalar.memref_address %alloc_8 : memref<2048xf32, 1>
            %135 = ascscalar.memref_address %alloc_9 : memref<2048xf32, 1>
            %136 = arith.select %81, %134, %135 : i64
            %137 = arith.select %81, %c24_i64, %c25_i64 : i64
            %138 = arith.cmpi eq, %68#6, %c0_i64 : i64
            %139 = arith.select %138, %119, %120 : i64
            %140 = arith.select %138, %c11_i64, %c12_i64 : i64
            %141 = ascscalar.bind_local_ptr %136 : memref<1xf32, 1>
            %142 = ascscalar.bind_local_ptr %139 : memref<1xf32, 5>
            %143 = arith.cmpi eq, %137, %c25_i64 : i64
            %144 = arith.select %143, %c4_i64, %c3_i64 : i64
            asccube.sync_intra_wait fixpipe, %144 {cannir.sync.banks = 3 : i32}
            %145 = arith.addi %144, %c16_i64 overflow<nsw> : i64
            asccube.sync_intra_wait fixpipe, %145 {cannir.sync.banks = 3 : i32}
            asccube.get_buf fixpipe, %140, 0 {cannir.buffer.generated_sync}
            asccube.copy_l0c2ub %141[%c0], %142[%c0], %c128_i32, %c32_i32, %c128_i32, %c32_i32, %c0_i32 sub_block_id %c0_i32 {dual_dst_ctl = 1 : i32, nz2nd_en = true, relu_en = false} : memref<1xf32, 1>, memref<1xf32, 5>
            asccube.release_buf fixpipe, %140, 0 {cannir.buffer.generated_sync}
            asccube.sync_intra_arrive fixpipe, %144 {cannir.sync.banks = 3 : i32}
            asccube.sync_intra_arrive fixpipe, %145 {cannir.sync.banks = 3 : i32}
            scf.yield %98, %105, %124, %87 : i64, i64, i64, i64
          } else {
            scf.yield %arg24, %68#1, %68#5, %arg31 : i64, i64, i64, i64
          }
          %79 = arith.cmpi sge, %arg21, %c3_i64 : i64
          %80:2 = scf.if %79 -> (i64, i64) {
            %81 = arith.cmpi eq, %arg21, %c3_i64 : i64
            %82 = arith.addi %arg32, %c1_i64 : i64
            %83 = arith.remui %82, %c2_i64 : i64
            %84 = scf.if %81 -> (i64) {
              scf.yield %arg28 : i64
            } else {
              %85 = arith.addi %arg28, %c1_i64 : i64
              %86 = arith.remui %85, %c3_i64 : i64
              scf.yield %86 : i64
            }
            scf.yield %84, %83 : i64, i64
          } else {
            scf.yield %arg28, %arg32 : i64, i64
          }
          scf.yield %68#0, %73#0, %78#0, %78#1, %78#2, %73#1, %80#0, %73#2, %73#3, %78#3, %80#1 : i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64
        }
        scf.yield %66#0, %66#1, %66#2, %66#3, %66#4, %66#5, %66#6, %66#7, %66#8, %66#9, %66#10 : i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64
      } else {
        scf.yield %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18, %arg19, %arg20 : i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64
      }
      scf.yield %41#0, %41#1, %41#2, %41#3, %41#4, %41#5, %41#6, %41#7, %41#8, %41#9, %41#10 : i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64
    }
    asccube.sync_intra_wait fixpipe, %c0_i64
    asccube.sync_intra_wait fixpipe, %c16_i64
    asccube.sync_intra_wait fixpipe, %c1_i64
    asccube.sync_intra_wait fixpipe, %c17_i64
    asccube.sync_intra_wait fixpipe, %c2_i64
    asccube.sync_intra_wait fixpipe, %c18_i64
    asccube.sync_intra_wait fixpipe, %c3_i64
    asccube.sync_intra_wait fixpipe, %c19_i64
    asccube.sync_intra_wait fixpipe, %c4_i64
    asccube.sync_intra_wait fixpipe, %c20_i64
    return
  }
}
