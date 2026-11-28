module {
  func.func @_launch_qwen_sparse_attn_block_128_fp8_fp16_92531743f59f4bb7fed9b72594b110fd(%arg0: !cannir.tensor<<gm, f16>, <<(-1, -1, 128)>, <(-1, 128, 1)>>, <<(-1, -1, 128)>, <(-1, 128, 1)>>, identity>, %arg1: !cannir.tensor<<gm, f8E4M3FN>, <<(-1, -1, 128)>, <(-1, 128, 1)>>, <<(-1, -1, 128)>, <(-1, 128, 1)>>, identity>, %arg2: !cannir.tensor<<gm, f8E4M3FN>, <<(-1, -1, 128, 128)>, <(-1, 128, -1, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 128, -1, 1)>>, identity>, %arg3: !cannir.tensor<<gm, f8E4M3FN>, <<(-1, -1, 128, 128)>, <(-1, 128, -1, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 128, -1, 1)>>, identity>, %arg4: !cannir.tensor<<gm, i8>, <<(2048, 2048)>, <(2048, 1)>>, <<(2048, 2048)>, <(2048, 1)>>, identity>, %arg5: !cannir.tensor<<gm, i64>, <<(2, -1)>, <(-1, 1)>>, <<(2, -1)>, <(-1, 1)>>, identity>, %arg6: !cannir.tensor<<gm, i64>, <<(5, -1)>, <(-1, 1)>>, <<(5, -1)>, <(-1, 1)>>, identity>, %arg7: !cannir.tensor<<gm, i32>, <<(2, -1)>, <(-1, 1)>>, <<(2, -1)>, <(-1, 1)>>, identity>, %arg8: f32) attributes {cannbotdsl.contract_digest = "92531743f59f4bb7fed9b72594b110fd11f2c6f120892be137a38737216eb1c1", cannbotdsl.public_slot_count = 9 : i64, cannbotdsl.slot_dtype_bits = [16, 8, 8, 8, 8, 64, 64, 32, 32], cannbotdsl.slot_dtype_codes = [2, 10, 10, 10, 0, 0, 0, 0, 2], cannbotdsl.slot_dtype_lanes = [1, 1, 1, 1, 1, 1, 1, 1, 1], cannbotdsl.slot_flags = [1, 1, 1, 1, 1, 1, 1, 1, 1], cannbotdsl.slot_kinds = [1, 1, 1, 1, 1, 1, 1, 1, 2], cannbotdsl.slot_type_digests = ["2f4e0e651731e482b93e21ebb68954c21d3aa199e544e76056fe2be56598ab1e", "c5106bd4fe31287de951b04c622a1a034c6dfce99a5416fab11cee61a2a2e3d8", "0dd0aa7d47177b98ff0a7ee41832284611118162fd5106129ed48bb568d2f860", "0dd0aa7d47177b98ff0a7ee41832284611118162fd5106129ed48bb568d2f860", "63649f3e6e296c53ebd705f85f4ed997c2453444b6d83c8a06a9dd87262b2411", "38118a8d25541e41378ae6cc983260d72eed7f74b94e13118e5609d3d922e429", "9691e4f3a979b35bb52ad1f2d384dce457e84946552fd3579d2ead48c2a1cd6a", "b279423ca759fd57dd673491bf7a7e05a53fbe97c11a11c3d872daa4d92a56e3", "4a5363e2ec1eb8cefee037ab0b231653c07f9b44a406ef92ed9ba3e6f5541dbf"], cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.host} {
    %c32_i64 = arith.constant 32 : i64
    cannir.kernel_launch @kernel_QwenSparseAttnBlock128___call___0(%c32_i64) [%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8] {cannbotdsl.profile = {indices = array<i64>, name = "Qwen_Sparse_Attn", op_type = "Qwen_Sparse_Attn", roles = array<i64>}} : !cannir.tensor<<gm, f16>, <<(-1, -1, 128)>, <(-1, 128, 1)>>, <<(-1, -1, 128)>, <(-1, 128, 1)>>, identity>, !cannir.tensor<<gm, f8E4M3FN>, <<(-1, -1, 128)>, <(-1, 128, 1)>>, <<(-1, -1, 128)>, <(-1, 128, 1)>>, identity>, !cannir.tensor<<gm, f8E4M3FN>, <<(-1, -1, 128, 128)>, <(-1, 128, -1, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 128, -1, 1)>>, identity>, !cannir.tensor<<gm, f8E4M3FN>, <<(-1, -1, 128, 128)>, <(-1, 128, -1, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 128, -1, 1)>>, identity>, !cannir.tensor<<gm, i8>, <<(2048, 2048)>, <(2048, 1)>>, <<(2048, 2048)>, <(2048, 1)>>, identity>, !cannir.tensor<<gm, i64>, <<(2, -1)>, <(-1, 1)>>, <<(2, -1)>, <(-1, 1)>>, identity>, !cannir.tensor<<gm, i64>, <<(5, -1)>, <(-1, 1)>>, <<(5, -1)>, <(-1, 1)>>, identity>, !cannir.tensor<<gm, i32>, <<(2, -1)>, <(-1, 1)>>, <<(2, -1)>, <(-1, 1)>>, identity>, f32
    return
  }
  func.func @kernel_QwenSparseAttnBlock128___call___0_aiv(%arg0: !cannir.tensor<<gm, f16>, <<(-1, -1, 128)>, <(-1, 128, 1)>>, <<(-1, -1, 128)>, <(-1, 128, 1)>>, identity>, %arg1: !cannir.tensor<<gm, f8E4M3FN>, <<(-1, -1, 128)>, <(-1, 128, 1)>>, <<(-1, -1, 128)>, <(-1, 128, 1)>>, identity>, %arg2: !cannir.tensor<<gm, f8E4M3FN>, <<(-1, -1, 128, 128)>, <(-1, 128, -1, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 128, -1, 1)>>, identity>, %arg3: !cannir.tensor<<gm, f8E4M3FN>, <<(-1, -1, 128, 128)>, <(-1, 128, -1, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 128, -1, 1)>>, identity>, %arg4: !cannir.tensor<<gm, i8>, <<(2048, 2048)>, <(2048, 1)>>, <<(2048, 2048)>, <(2048, 1)>>, identity>, %arg5: !cannir.tensor<<gm, i64>, <<(2, -1)>, <(-1, 1)>>, <<(2, -1)>, <(-1, 1)>>, identity>, %arg6: !cannir.tensor<<gm, i64>, <<(5, -1)>, <(-1, 1)>>, <<(5, -1)>, <(-1, 1)>>, identity>, %arg7: !cannir.tensor<<gm, i32>, <<(2, -1)>, <(-1, 1)>>, <<(2, -1)>, <(-1, 1)>>, identity>, %arg8: f32) attributes {cannir.buffer.auto_sync, cannir.buffer.sync_analyzed, cannir.buffer.sync_verified, cannir.buffer.tensor_state, cannir.core = "aiv", cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.device, cannir.explicit_ring_buffer_cursors, cannir.kernel_group = "kernel_QwenSparseAttnBlock128___call___0", cannir.resource.ids_prepared} {
    %c16_i32 = arith.constant 16 : i32
    %c4_i32 = arith.constant 4 : i32
    %c4 = arith.constant 4 : index
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c16_i64 = arith.constant 16 : i64
    %c8_i64 = arith.constant 8 : i64
    %c7_i64 = arith.constant 7 : i64
    %c6_i64 = arith.constant 6 : i64
    %c5_i64 = arith.constant 5 : i64
    %c4_i64 = arith.constant 4 : i64
    %c3_i64 = arith.constant 3 : i64
    %c2_i64 = arith.constant 2 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %c25_i64 = arith.constant 25 : i64
    %c26_i64 = arith.constant 26 : i64
    %c27_i64 = arith.constant 27 : i64
    %c28_i64 = arith.constant 28 : i64
    %c29_i64 = arith.constant 29 : i64
    %c30_i64 = arith.constant 30 : i64
    %c31_i64 = arith.constant 31 : i64
    %c32 = arith.constant 32 : index
    %c1_i32 = arith.constant 1 : i32
    %c256_i32 = arith.constant 256 : i32
    %c2048_i64 = arith.constant 2048 : i64
    %c127_i64 = arith.constant 127 : i64
    %c-1_i64 = arith.constant -1 : i64
    %false = arith.constant false
    %c0 = arith.constant 0 : index
    %c128_i64 = arith.constant 128 : i64
    %c0_i32 = arith.constant 0 : i32
    %c32_i64 = arith.constant 32 : i64
    %alloc = memref.alloc() {cannir.arg_index = 0 : i64} : memref<1xf16>
    %alloc_0 = memref.alloc() {cannir.arg_index = 1 : i64} : memref<1xf8E4M3FN>
    %alloc_1 = memref.alloc() {cannir.arg_index = 4 : i64} : memref<4194304xi8>
    %alloc_2 = memref.alloc() {cannir.arg_index = 5 : i64} : memref<1xi64>
    %alloc_3 = memref.alloc() {cannir.arg_index = 6 : i64} : memref<1xi64>
    %alloc_4 = memref.alloc() {cannir.arg_index = 7 : i64} : memref<1xi32>
    %alloc_5 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<128xf32, 1>
    %alloc_6 = memref.alloc() {cannir.byte_offset = 512 : i64} : memref<128xf32, 1>
    %alloc_7 = memref.alloc() {cannir.byte_offset = 1024 : i64} : memref<128xf32, 1>
    %alloc_8 = memref.alloc() {cannir.byte_offset = 1536 : i64} : memref<128xf32, 1>
    %alloc_9 = memref.alloc() {cannir.byte_offset = 2048 : i64} : memref<128xf32, 1>
    %alloc_10 = memref.alloc() {cannir.byte_offset = 2560 : i64} : memref<128xf32, 1>
    %alloc_11 = memref.alloc() {cannir.byte_offset = 3072 : i64} : memref<128xf32, 1>
    %alloc_12 = memref.alloc() {cannir.byte_offset = 3584 : i64} : memref<2048xf32, 1>
    %alloc_13 = memref.alloc() {cannir.byte_offset = 11776 : i64} : memref<128xi8, 1>
    %alloc_14 = memref.alloc() {cannir.byte_offset = 11904 : i64} : memref<128xi8, 1>
    %alloc_15 = memref.alloc() {cannir.byte_offset = 12032 : i64} : memref<2048xi8, 1>
    %alloc_16 = memref.alloc() {cannir.byte_offset = 14080 : i64} : memref<2048xi8, 1>
    %alloc_17 = memref.alloc() {cannir.byte_offset = 16128 : i64} : memref<2048xi8, 1>
    %alloc_18 = memref.alloc() {cannir.byte_offset = 18176 : i64} : memref<2048xf16, 1>
    %alloc_19 = memref.alloc() {cannir.byte_offset = 22272 : i64} : memref<128xi8, 1>
    %alloc_20 = memref.alloc() {cannir.byte_offset = 22400 : i64} : memref<128xi8, 1>
    %alloc_21 = memref.alloc() {cannir.byte_offset = 22528 : i64} : memref<2048xi8, 1>
    %alloc_22 = memref.alloc() {cannir.byte_offset = 24576 : i64} : memref<2048xf32, 1>
    %alloc_23 = memref.alloc() {cannir.byte_offset = 32768 : i64} : memref<2048xf32, 1>
    %alloc_24 = memref.alloc() {cannir.byte_offset = 40960 : i64} : memref<2048xf32, 1>
    %alloc_25 = memref.alloc() {cannir.byte_offset = 49152 : i64} : memref<2048xf32, 1>
    %alloc_26 = memref.alloc() {cannir.byte_offset = 57344 : i64} : memref<2048xf32, 1>
    %alloc_27 = memref.alloc() {cannir.byte_offset = 4096 : i64} : memref<4096xf8E4M3FN, 2>
    %alloc_28 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<4096xf8E4M3FN, 2>
    %alloc_29 = memref.alloc() {cannir.byte_offset = 110592 : i64} : memref<4096xf8E4M3FN, 2>
    %alloc_30 = memref.alloc() {cannir.byte_offset = 106496 : i64} : memref<4096xf8E4M3FN, 2>
    ascvec.sync_intra_arrive v, %c6_i64
    ascvec.sync_intra_arrive v, %c5_i64
    ascvec.sync_intra_arrive v, %c4_i64
    ascvec.sync_intra_arrive v, %c3_i64
    ascvec.sync_intra_arrive v, %c2_i64
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
    %10:14 = scf.for %arg9 = %3 to %9 step %c1_i64 iter_args(%arg10 = %c0_i64, %arg11 = %c0_i64, %arg12 = %c0_i64, %arg13 = %c0_i64, %arg14 = %c0_i64, %arg15 = %c0_i64, %arg16 = %c0_i64, %arg17 = %c0_i64, %arg18 = %c0_i64, %arg19 = %c0_i64, %arg20 = %c0_i64, %arg21 = %c0_i64, %arg22 = %c0_i64, %arg23 = %c0_i64) -> (i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64)  : i64 {
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
      %29 = arith.subi %c3_i64, %28 : i64
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
      %41:14 = scf.if %40 -> (i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) {
        %42 = arith.cmpi eq, %arg9, %3 : i64
        %43:2 = scf.if %42 -> (i64, i64) {
          ascvec.sync_all
          %53 = scf.for %arg24 = %c0_i64 to %38 step %c1_i64 iter_args(%arg25 = %arg21) -> (i64)  : i64 {
            %69 = arith.addi %28, %39 : i64
            %70 = arith.addi %69, %arg24 : i64
            %71 = arith.muli %18, %c3_i64 : i64
            %72 = arith.addi %71, %70 : i64
            %73 = arith.cmpi eq, %arg25, %c0_i64 : i64
            %74 = ascscalar.memref_address %alloc_19 : memref<128xi8, 1>
            %75 = ascscalar.memref_address %alloc_20 : memref<128xi8, 1>
            %76 = arith.select %73, %74, %75 : i64
            %77 = arith.select %73, %c5_i64, %c6_i64 : i64
            %78 = arith.addi %arg25, %c1_i64 overflow<nsw> : i64
            %79 = arith.remui %78, %c2_i64 : i64
            %80 = cannir.extract_physical_layout %arg1 : !cannir.tensor<<gm, f8E4M3FN>, <<(-1, -1, 128)>, <(-1, 128, 1)>>, <<(-1, -1, 128)>, <(-1, 128, 1)>>, identity>
            %81 = cannir.extract_stride %80 : <<(-1, -1, 128)>, <(-1, 128, 1)>> -> <(-1, 128, 1)>
            %82 = cannir.extract_dim %81[0] : !cannir.stride<(-1, 128, 1)>
            %83 = arith.muli %12, %82 : i64
            %84 = arith.muli %72, %c128_i64 : i64
            %85 = arith.addi %83, %84 : i64
            %86 = ascscalar.bind_local_ptr %76 : memref<1xi8, 1>
            ascvec.get_buf mte2, %77, 0 {cannir.buffer.generated_sync}
            "ascvec.copy_gm2ub_plan"(%86, %alloc_0, %c0_i32, %c0_i64, %c0_i64, %85, %c1_i64, %c128_i64, %c0_i64, %c128_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = true, storage_dtype = f8E4M3FN, ub_base = -1 : i64, ub_capacity = 1 : i64}> : (memref<1xi8, 1>, memref<1xf8E4M3FN>, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
            ascvec.release_buf mte2, %77, 0 {cannir.buffer.generated_sync}
            %87 = ascscalar.bind_local_ptr %76 : memref<128xi8, 1>
            %88 = arith.remsi %arg24, %c16_i64 : i64
            %89 = arith.muli %88, %c32_i64 overflow<nsw> : i64
            %90 = arith.index_cast %89 : i64 to index
            ascvec.get_buf v, %77, 0 {cannir.buffer.generated_sync}
            %91 = ascscalar.bitcast_memref %87 byte_offset = %c0_i64 : memref<128xi8, 1> -> memref<?xf8E4M3FN, 1>
            ascvec.vf(%91, %alloc_21, %90 : memref<?xf8E4M3FN, 1>, memref<2048xi8, 1>, index) {
            ^bb0(%arg26: memref<?xf8E4M3FN, 1>, %arg27: memref<2048xi8, 1>, %arg28: index):
              %c16 = arith.constant 16 : index
              %c0_31 = arith.constant 0 : index
              %c128_i32 = arith.constant 128 : i32
              %c0_32 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              scf.for %arg29 = %c0_32 to %c1 step %c1 {
                %mask, %remaining_out = ascvec.update_mask %c128_i32 elem_bits 8 : !ascvec.mask
                %92 = ascvec.loadalign %arg26[%c0_31] : memref<?xf8E4M3FN, 1>, index -> <f8E4M3FN>
                ascvec.storealign_strided %arg27[%arg28], %92, %mask, %c16, %c0_31 : memref<2048xi8, 1>, <f8E4M3FN>
              }
            } {cannir.vf_mode = "simd"}
            ascvec.release_buf v, %77, 0 {cannir.buffer.generated_sync}
            scf.yield %79 : i64
          }
          ascvec.sync_all
          %54 = arith.cmpi eq, %arg10, %c0_i64 : i64
          %55 = ascscalar.memref_address %alloc_28 : memref<4096xf8E4M3FN, 2>
          %56 = ascscalar.memref_address %alloc_27 : memref<4096xf8E4M3FN, 2>
          %57 = arith.select %54, %55, %56 : i64
          %58 = arith.cmpi ne, %arg10, %c0_i64 : i64
          %59 = arith.extui %58 : i1 to i64
          %60 = arith.addi %arg10, %c1_i64 overflow<nsw> : i64
          %61 = arith.remui %60, %c2_i64 : i64
          %62 = ascscalar.bitcast_memref %alloc_21 byte_offset = %c0_i64 : memref<2048xi8, 1> -> memref<?xf8E4M3FN, 1>
          %63 = ascscalar.bind_local_ptr %57 : memref<1xf8E4M3FN, 2>
          %64 = arith.muli %1, %c16_i64 : i64
          %65 = arith.index_cast %64 : i64 to index
          %66 = arith.muli %65, %c32 : index
          %67 = arith.cmpi eq, %59, %c1_i64 : i64
          %68 = arith.extui %67 : i1 to i64
          ascvec.sync_intra_wait mte3, %68 {cannir.sync.banks = 3 : i32}
          ascvec.copy_ub2l1_strided %63[%66], %62[%c0], %c4_i32, %c16_i32, %c0_i32, %c16_i32 : memref<1xf8E4M3FN, 2>, memref<?xf8E4M3FN, 1>
          ascvec.sync_intra_arrive mte3, %68 {cannir.sync.banks = 3 : i32}
          scf.yield %61, %53 : i64, i64
        } else {
          scf.yield %arg10, %arg21 : i64, i64
        }
        %44 = ascscalar.memref_address %alloc_28 : memref<4096xf8E4M3FN, 2>
        %45 = ascscalar.memref_address %alloc_27 : memref<4096xf8E4M3FN, 2>
        %46 = arith.addi %arg11, %c1_i64 overflow<nsw> : i64
        %47 = arith.remui %46, %c2_i64 : i64
        %48 = arith.addi %arg9, %c1_i64 : i64
        %49 = arith.cmpi slt, %48, %9 : i64
        %50:2 = scf.if %49 -> (i64, i64) {
          %53 = arith.index_cast %48 : i64 to index
          %54 = memref.load %alloc_3[%53] : memref<1xi64>
          %55 = arith.addi %16, %53 : index
          %56 = memref.load %alloc_3[%55] : memref<1xi64>
          %57 = arith.addi %19, %53 : index
          %58 = memref.load %alloc_3[%57] : memref<1xi64>
          %59 = arith.addi %25, %53 : index
          %60 = memref.load %alloc_3[%59] : memref<1xi64>
          %61 = arith.muli %58, %c32_i64 : i64
          %62 = arith.subi %c3_i64, %61 : i64
          %63 = arith.cmpi sgt, %62, %c32_i64 : i64
          %64 = arith.cmpi sgt, %62, %c0_i64 : i64
          %65 = arith.select %64, %62, %c0_i64 : i64
          %66 = arith.select %63, %c32_i64, %65 : i64
          %67 = arith.addi %66, %c1_i64 : i64
          %68 = arith.divsi %67, %c2_i64 : i64
          %69 = arith.divsi %66, %c2_i64 : i64
          %70 = arith.select %37, %68, %69 : i64
          %71 = arith.select %37, %c0_i64, %68 : i64
          %72 = arith.cmpi sgt, %60, %c0_i64 : i64
          %73:2 = scf.if %72 -> (i64, i64) {
            ascvec.sync_all
            %74 = scf.for %arg24 = %c0_i64 to %70 step %c1_i64 iter_args(%arg25 = %43#1) -> (i64)  : i64 {
              %88 = arith.addi %61, %71 : i64
              %89 = arith.addi %88, %arg24 : i64
              %90 = arith.muli %56, %c3_i64 : i64
              %91 = arith.addi %90, %89 : i64
              %92 = arith.cmpi eq, %arg25, %c0_i64 : i64
              %93 = ascscalar.memref_address %alloc_19 : memref<128xi8, 1>
              %94 = ascscalar.memref_address %alloc_20 : memref<128xi8, 1>
              %95 = arith.select %92, %93, %94 : i64
              %96 = arith.select %92, %c5_i64, %c6_i64 : i64
              %97 = arith.addi %arg25, %c1_i64 overflow<nsw> : i64
              %98 = arith.remui %97, %c2_i64 : i64
              %99 = cannir.extract_physical_layout %arg1 : !cannir.tensor<<gm, f8E4M3FN>, <<(-1, -1, 128)>, <(-1, 128, 1)>>, <<(-1, -1, 128)>, <(-1, 128, 1)>>, identity>
              %100 = cannir.extract_stride %99 : <<(-1, -1, 128)>, <(-1, 128, 1)>> -> <(-1, 128, 1)>
              %101 = cannir.extract_dim %100[0] : !cannir.stride<(-1, 128, 1)>
              %102 = arith.muli %54, %101 : i64
              %103 = arith.muli %91, %c128_i64 : i64
              %104 = arith.addi %102, %103 : i64
              %105 = ascscalar.bind_local_ptr %95 : memref<1xi8, 1>
              ascvec.get_buf mte2, %96, 0 {cannir.buffer.generated_sync}
              "ascvec.copy_gm2ub_plan"(%105, %alloc_0, %c0_i32, %c0_i64, %c0_i64, %104, %c1_i64, %c128_i64, %c0_i64, %c128_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = true, storage_dtype = f8E4M3FN, ub_base = -1 : i64, ub_capacity = 1 : i64}> : (memref<1xi8, 1>, memref<1xf8E4M3FN>, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
              ascvec.release_buf mte2, %96, 0 {cannir.buffer.generated_sync}
              %106 = ascscalar.bind_local_ptr %95 : memref<128xi8, 1>
              %107 = arith.remsi %arg24, %c16_i64 : i64
              %108 = arith.muli %107, %c32_i64 overflow<nsw> : i64
              %109 = arith.index_cast %108 : i64 to index
              ascvec.get_buf v, %96, 0 {cannir.buffer.generated_sync}
              %110 = ascscalar.bitcast_memref %106 byte_offset = %c0_i64 : memref<128xi8, 1> -> memref<?xf8E4M3FN, 1>
              ascvec.vf(%110, %alloc_21, %109 : memref<?xf8E4M3FN, 1>, memref<2048xi8, 1>, index) {
              ^bb0(%arg26: memref<?xf8E4M3FN, 1>, %arg27: memref<2048xi8, 1>, %arg28: index):
                %c16 = arith.constant 16 : index
                %c0_31 = arith.constant 0 : index
                %c128_i32 = arith.constant 128 : i32
                %c0_32 = arith.constant 0 : index
                %c1 = arith.constant 1 : index
                scf.for %arg29 = %c0_32 to %c1 step %c1 {
                  %mask, %remaining_out = ascvec.update_mask %c128_i32 elem_bits 8 : !ascvec.mask
                  %111 = ascvec.loadalign %arg26[%c0_31] : memref<?xf8E4M3FN, 1>, index -> <f8E4M3FN>
                  ascvec.storealign_strided %arg27[%arg28], %111, %mask, %c16, %c0_31 : memref<2048xi8, 1>, <f8E4M3FN>
                }
              } {cannir.vf_mode = "simd"}
              ascvec.release_buf v, %96, 0 {cannir.buffer.generated_sync}
              scf.yield %98 : i64
            }
            ascvec.sync_all
            %75 = arith.cmpi eq, %43#0, %c0_i64 : i64
            %76 = arith.select %75, %44, %45 : i64
            %77 = arith.cmpi ne, %43#0, %c0_i64 : i64
            %78 = arith.extui %77 : i1 to i64
            %79 = arith.addi %43#0, %c1_i64 overflow<nsw> : i64
            %80 = arith.remui %79, %c2_i64 : i64
            %81 = ascscalar.bitcast_memref %alloc_21 byte_offset = %c0_i64 : memref<2048xi8, 1> -> memref<?xf8E4M3FN, 1>
            %82 = ascscalar.bind_local_ptr %76 : memref<1xf8E4M3FN, 2>
            %83 = arith.muli %1, %c16_i64 : i64
            %84 = arith.index_cast %83 : i64 to index
            %85 = arith.muli %84, %c32 : index
            %86 = arith.cmpi eq, %78, %c1_i64 : i64
            %87 = arith.extui %86 : i1 to i64
            ascvec.sync_intra_wait mte3, %87 {cannir.sync.banks = 3 : i32}
            ascvec.copy_ub2l1_strided %82[%85], %81[%c0], %c4_i32, %c16_i32, %c0_i32, %c16_i32 : memref<1xf8E4M3FN, 2>, memref<?xf8E4M3FN, 1>
            ascvec.sync_intra_arrive mte3, %87 {cannir.sync.banks = 3 : i32}
            scf.yield %80, %74 : i64, i64
          } else {
            scf.yield %43#0, %43#1 : i64, i64
          }
          scf.yield %73#0, %73#1 : i64, i64
        } else {
          scf.yield %43#0, %43#1 : i64, i64
        }
        %51 = arith.addi %27, %c3_i64 : i64
        %52:11 = scf.for %arg24 = %c0_i64 to %51 step %c1_i64 iter_args(%arg25 = %arg12, %arg26 = %arg13, %arg27 = %arg14, %arg28 = %arg15, %arg29 = %arg16, %arg30 = %arg17, %arg31 = %arg18, %arg32 = %arg19, %arg33 = %arg20, %arg34 = %arg22, %arg35 = %arg23) -> (i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64)  : i64 {
          %53 = arith.cmpi slt, %arg24, %27 : i64
          %54:3 = scf.if %53 -> (i64, i64, i64) {
            %67 = arith.addi %arg25, %c1_i64 : i64
            %68 = arith.remui %67, %c3_i64 : i64
            %69 = arith.addi %arg29, %c1_i64 : i64
            %70 = arith.remui %69, %c2_i64 : i64
            %71 = arith.addi %arg28, %c1_i64 : i64
            %72 = arith.remui %71, %c2_i64 : i64
            scf.yield %68, %72, %70 : i64, i64, i64
          } else {
            scf.yield %arg25, %arg28, %arg29 : i64, i64, i64
          }
          %55 = arith.cmpi sge, %arg24, %c1_i64 : i64
          %56 = arith.addi %27, %c1_i64 : i64
          %57 = arith.cmpi slt, %arg24, %56 : i64
          %58 = arith.select %55, %57, %false : i1
          %59:4 = scf.if %58 -> (i64, i64, i64, i64) {
            %67 = arith.addi %24, %arg24 : i64
            %68 = arith.subi %67, %c1_i64 : i64
            %69 = arith.index_cast %68 : i64 to index
            %70 = arith.cmpi eq, %arg24, %c1_i64 : i64
            %71 = arith.cmpi eq, %arg26, %c1_i64 : i64
            %72 = arith.cmpi eq, %arg26, %c0_i64 : i64
            %73 = arith.addi %arg26, %c1_i64 : i64
            %74 = arith.remui %73, %c3_i64 : i64
            %75 = arith.addi %39, %38 : i64
            %76 = arith.maxsi %39, %c0_i64 : i64
            %77 = arith.minsi %75, %c3_i64 : i64
            %78 = arith.subi %77, %76 : i64
            %79 = arith.maxsi %78, %c0_i64 : i64
            %80 = arith.cmpi sgt, %79, %c0_i64 : i64
            %81 = scf.if %80 -> (i64) {
              %110 = cannir.extract_layout %arg7 : !cannir.tensor<<gm, i32>, <<(2, -1)>, <(-1, 1)>>, <<(2, -1)>, <(-1, 1)>>, identity>
              %111 = cannir.extract_stride %110 : <<(2, -1)>, <(-1, 1)>> -> <(-1, 1)>
              %112 = cannir.extract_dim %111[0] : !cannir.stride<(-1, 1)>
              %113 = arith.index_cast %112 : i64 to index
              %114 = arith.addi %113, %69 : index
              %115 = memref.load %alloc_4[%114] : memref<1xi32>
              %116 = arith.extsi %115 : i32 to i64
              %117 = arith.subi %116, %c1_i64 : i64
              %118 = arith.maxsi %117, %c-1_i64 : i64
              %119 = arith.minsi %118, %c127_i64 : i64
              %120 = arith.cmpi sge, %119, %c0_i64 : i64
              %121 = arith.select %120, %119, %c0_i64 : i64
              %122 = arith.subi %c0_i64, %119 : i64
              %123 = arith.select %120, %c0_i64, %122 : i64
              %124 = arith.cmpi eq, %arg32, %c0_i64 : i64
              %125 = ascscalar.memref_address %alloc_13 : memref<128xi8, 1>
              %126 = ascscalar.memref_address %alloc_14 : memref<128xi8, 1>
              %127 = arith.select %124, %125, %126 : i64
              %128 = arith.cmpi ne, %arg32, %c0_i64 : i64
              %129 = arith.extui %128 : i1 to i64
              %130 = arith.addi %arg32, %c1_i64 : i64
              %131 = arith.remui %130, %c2_i64 : i64
              %132 = arith.muli %121, %c2048_i64 : i64
              %133 = arith.addi %132, %123 : i64
              %134 = ascscalar.bind_local_ptr %127 : memref<1xi8, 1>
              ascvec.get_buf mte2, %129, 0 {cannir.buffer.generated_sync}
              "ascvec.copy_gm2ub_plan"(%134, %alloc_1, %c0_i32, %c0_i32, %c0_i64, %133, %c1_i64, %c128_i64, %c0_i64, %c128_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 4194304 : i64, pad_storage_bits = false, storage_dtype = i8, ub_base = -1 : i64, ub_capacity = 1 : i64}> : (memref<1xi8, 1>, memref<4194304xi8>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
              ascvec.release_buf mte2, %129, 0 {cannir.buffer.generated_sync}
              %135 = arith.subi %76, %39 : i64
              %136 = ascscalar.bind_local_ptr %127 : memref<128xi8, 1>
              ascvec.get_buf v, %129, 0 {cannir.buffer.generated_sync}
              ascvec.vf(%136, %135, %alloc_15, %79 : memref<128xi8, 1>, i64, memref<2048xi8, 1>, i64) {
              ^bb0(%arg36: memref<128xi8, 1>, %arg37: i64, %arg38: memref<2048xi8, 1>, %arg39: i64):
                %c1_i64_31 = arith.constant 1 : i64
                %c0_i64_32 = arith.constant 0 : i64
                %c128_i64_33 = arith.constant 128 : i64
                %c0_34 = arith.constant 0 : index
                %c128_i32 = arith.constant 128 : i32
                %mask, %remaining_out = ascvec.update_mask %c128_i32 elem_bits 8 : !ascvec.mask
                %137 = ascvec.loadalign %arg36[%c0_34] : memref<128xi8, 1>, index -> <i8>
                scf.for %arg40 = %c0_i64_32 to %arg39 step %c1_i64_31  : i64 {
                  %138 = arith.addi %arg37, %arg40 : i64
                  %139 = arith.muli %138, %c128_i64_33 : i64
                  %140 = arith.index_cast %139 : i64 to index
                  ascvec.storealign %arg38[%140], %137, %mask : memref<2048xi8, 1>, index, <i8>
                }
              } {cannir.vf_mode = "simd"}
              ascvec.release_buf v, %129, 0 {cannir.buffer.generated_sync}
              scf.yield %131 : i64
            } else {
              scf.yield %arg32 : i64
            }
            %82 = ascscalar.memref_address %alloc_23 : memref<2048xf32, 1>
            %83 = ascscalar.memref_address %alloc_24 : memref<2048xf32, 1>
            %84 = arith.select %71, %82, %83 : i64
            %85 = ascscalar.memref_address %alloc_22 : memref<2048xf32, 1>
            %86 = arith.select %72, %85, %84 : i64
            %87 = ascscalar.bind_local_ptr %86 : memref<2048xf32, 1>
            %88 = arith.select %71, %c26_i64, %c27_i64 : i64
            %89 = arith.select %72, %c25_i64, %88 : i64
            %90 = arith.cmpi eq, %arg33, %c0_i64 : i64
            %91 = ascscalar.memref_address %alloc_16 : memref<2048xi8, 1>
            %92 = ascscalar.memref_address %alloc_17 : memref<2048xi8, 1>
            %93 = arith.select %90, %91, %92 : i64
            %94 = ascscalar.bind_local_ptr %93 : memref<2048xi8, 1>
            %95 = arith.select %90, %c2_i64, %c3_i64 : i64
            %96 = arith.addi %arg33, %c1_i64 : i64
            %97 = arith.remui %96, %c2_i64 : i64
            %98 = scf.if %70 -> (i64) {
              %110 = arith.cmpi eq, %89, %c26_i64 : i64
              %111 = arith.select %110, %c3_i64, %c2_i64 : i64
              %112 = arith.cmpi eq, %89, %c27_i64 : i64
              %113 = arith.select %112, %c4_i64, %111 : i64
              ascvec.sync_intra_wait v, %113 {cannir.sync.banks = 3 : i32}
              ascvec.vf(%87, %arg8, %alloc_15, %alloc_5, %38 : memref<2048xf32, 1>, f32, memref<2048xi8, 1>, memref<128xf32, 1>, i64) {
              ^bb0(%arg36: memref<2048xf32, 1>, %arg37: f32, %arg38: memref<2048xi8, 1>, %arg39: memref<128xf32, 1>, %arg40: i64):
                %c1_i64_31 = arith.constant 1 : i64
                %c0_i64_32 = arith.constant 0 : i64
                %c8_i64_33 = arith.constant 8 : i64
                %cst = arith.constant -1.000000e+30 : f32
                %cst_34 = arith.constant -3.000000e+38 : f32
                %cst_35 = arith.constant 0.000000e+00 : f32
                %c64_i64 = arith.constant 64 : i64
                %c128_i64_36 = arith.constant 128 : i64
                %c64_i32 = arith.constant 64 : i32
                %mask, %remaining_out = ascvec.update_mask %c64_i32 : !ascvec.mask
                scf.for %arg41 = %c0_i64_32 to %arg40 step %c1_i64_31  : i64 {
                  %115 = arith.muli %arg41, %c128_i64_36 : i64
                  %116 = arith.index_cast %115 : i64 to index
                  %117 = ascvec.loadalign %arg36[%116] : memref<2048xf32, 1>, index -> <f32>
                  %118 = ascvec.mul_scalar %117, %arg37, %mask : <f32>, f32
                  %119 = arith.addi %115, %c64_i64 : i64
                  %120 = arith.index_cast %119 : i64 to index
                  %121 = ascvec.loadalign %arg36[%120] : memref<2048xf32, 1>, index -> <f32>
                  %122 = ascvec.mul_scalar %121, %arg37, %mask : <f32>, f32
                  %123 = ascvec.loadalign_unpack %arg38[%116] {mode = "b8_to_b32"} : memref<2048xi8, 1>, index -> <ui32>
                  %124 = ascvec.reinterpret %123 : <ui32> -> <i32>
                  %125 = ascvec.cast %124, %mask : <i32> -> <f32>
                  %126 = ascvec.cmp_eq_scalar %125, %cst_35, %mask : <f32>, f32 -> !ascvec.mask
                  %127 = ascvec.duplicate_scalar %cst_34, %mask : f32 -> <f32>
                  %128 = ascvec.select %118, %127, %126 : <f32>
                  %129 = ascvec.loadalign_unpack %arg38[%120] {mode = "b8_to_b32"} : memref<2048xi8, 1>, index -> <ui32>
                  %130 = ascvec.reinterpret %129 : <ui32> -> <i32>
                  %131 = ascvec.cast %130, %mask : <i32> -> <f32>
                  %132 = ascvec.cmp_eq_scalar %131, %cst_35, %mask : <f32>, f32 -> !ascvec.mask
                  %133 = ascvec.select %122, %127, %132 : <f32>
                  ascvec.storealign %arg36[%116], %128, %mask : memref<2048xf32, 1>, index, <f32>
                  ascvec.storealign %arg36[%120], %133, %mask : memref<2048xf32, 1>, index, <f32>
                  %134 = ascvec.max %128, %133, %mask : <f32>
                  %135 = ascvec.reduce_max %134, %mask : <f32>
                  %136 = ascvec.cmp_le_scalar %135, %cst, %mask : <f32>, f32 -> !ascvec.mask
                  %137 = ascvec.duplicate_scalar %cst_35, %mask : f32 -> <f32>
                  %138 = ascvec.select %137, %135, %136 : <f32>
                  %139 = arith.muli %arg41, %c8_i64_33 : i64
                  %140 = arith.index_cast %139 : i64 to index
                  ascvec.storealign_1st %arg39[%140], %138 : memref<128xf32, 1>, index, <f32>
                }
              } {cannir.vf_mode = "simd"}
              ascvec.get_buf v, %95, 0 {cannir.buffer.generated_sync}
              %114 = ascscalar.bitcast_memref %94 byte_offset = %c0_i64 : memref<2048xi8, 1> -> memref<?xf8E4M3FN, 1>
              ascvec.vf(%alloc_5, %87, %alloc_6, %114, %38 : memref<128xf32, 1>, memref<2048xf32, 1>, memref<128xf32, 1>, memref<?xf8E4M3FN, 1>, i64) {
              ^bb0(%arg36: memref<128xf32, 1>, %arg37: memref<2048xf32, 1>, %arg38: memref<128xf32, 1>, %arg39: memref<?xf8E4M3FN, 1>, %arg40: i64):
                %c1_i64_31 = arith.constant 1 : i64
                %c0_i64_32 = arith.constant 0 : i64
                %c16 = arith.constant 16 : index
                %c32_i64_33 = arith.constant 32 : i64
                %c16_i64_34 = arith.constant 16 : i64
                %cst = arith.constant 4.480000e+02 : f32
                %c64_i64 = arith.constant 64 : i64
                %c0_35 = arith.constant 0 : index
                %c8_i64_36 = arith.constant 8 : i64
                %c128_i64_37 = arith.constant 128 : i64
                %c128_i32 = arith.constant 128 : i32
                %c256_i32_38 = arith.constant 256 : i32
                %c64_i32 = arith.constant 64 : i32
                %mask, %remaining_out = ascvec.update_mask %c64_i32 : !ascvec.mask
                %mask_39, %remaining_out_40 = ascvec.update_mask %c256_i32_38 elem_bits 8 : !ascvec.mask
                %mask_41, %remaining_out_42 = ascvec.update_mask %c128_i32 elem_bits 8 : !ascvec.mask
                scf.for %arg41 = %c0_i64_32 to %arg40 step %c1_i64_31  : i64 {
                  %115 = arith.muli %arg41, %c128_i64_37 : i64
                  %116 = arith.muli %arg41, %c8_i64_36 : i64
                  %117 = arith.index_cast %116 : i64 to index
                  %118 = ascvec.loadalign_brc %arg36[%117] : memref<128xf32, 1>, index -> <f32>
                  %119 = arith.index_cast %115 : i64 to index
                  %120 = ascvec.loadalign %arg37[%119] : memref<2048xf32, 1>, index -> <f32>
                  %121 = ascvec.exp_sub %120, %118, %mask : <f32>, <f32> -> <f32>
                  %122 = arith.addi %115, %c64_i64 : i64
                  %123 = arith.index_cast %122 : i64 to index
                  %124 = ascvec.loadalign %arg37[%123] : memref<2048xf32, 1>, index -> <f32>
                  %125 = ascvec.exp_sub %124, %118, %mask : <f32>, <f32> -> <f32>
                  %126 = ascvec.add %121, %125, %mask : <f32>
                  %127 = ascvec.reduce_sum %126, %mask : <f32>
                  ascvec.storealign_1st %arg38[%117], %127 : memref<128xf32, 1>, index, <f32>
                  %128 = ascvec.mul_scalar %121, %cst, %mask : <f32>, f32
                  %129 = ascvec.mul_scalar %125, %cst, %mask : <f32>, f32
                  %result0, %result1 = ascvec.deintlv %128, %129 : <f32>, <f32> -> <f32>, <f32>
                  %130 = ascvec.cast %result0, %mask {reg_layout = 0 : i32, saturate = true} : <f32> -> <f8E4M3FN>
                  %131 = ascvec.cast %result1, %mask {reg_layout = 2 : i32, saturate = true} : <f32> -> <f8E4M3FN>
                  %132 = ascvec.or %130, %131, %mask_39 : <f8E4M3FN>, <f8E4M3FN> -> <f8E4M3FN>
                  %result0_43, %result1_44 = ascvec.deintlv %132, %132 : <f8E4M3FN>, <f8E4M3FN> -> <f8E4M3FN>, <f8E4M3FN>
                  %133 = arith.remsi %arg41, %c16_i64_34 : i64
                  %134 = arith.muli %133, %c32_i64_33 overflow<nsw> : i64
                  %135 = arith.index_cast %134 : i64 to index
                  ascvec.storealign_strided %arg39[%135], %result0_43, %mask_41, %c16, %c0_35 : memref<?xf8E4M3FN, 1>, <f8E4M3FN>
                }
              } {cannir.vf_mode = "simd"}
              ascvec.release_buf v, %95, 0 {cannir.buffer.generated_sync}
              ascvec.sync_intra_arrive v, %113 {cannir.sync.banks = 3 : i32}
              scf.yield %arg30 : i64
            } else {
              %110 = arith.cmpi eq, %arg30, %c1_i64 : i64
              %111 = ascscalar.memref_address %alloc_10 : memref<128xf32, 1>
              %112 = ascscalar.memref_address %alloc_11 : memref<128xf32, 1>
              %113 = arith.select %110, %111, %112 : i64
              %114 = arith.cmpi eq, %arg30, %c0_i64 : i64
              %115 = ascscalar.memref_address %alloc_9 : memref<128xf32, 1>
              %116 = arith.select %114, %115, %113 : i64
              %117 = ascscalar.bind_local_ptr %116 : memref<128xf32, 1>
              %118 = arith.addi %arg30, %c1_i64 : i64
              %119 = arith.remui %118, %c3_i64 : i64
              %120 = arith.cmpi eq, %89, %c26_i64 : i64
              %121 = arith.select %120, %c3_i64, %c2_i64 : i64
              %122 = arith.cmpi eq, %89, %c27_i64 : i64
              %123 = arith.select %122, %c4_i64, %121 : i64
              ascvec.sync_intra_wait v, %123 {cannir.sync.banks = 3 : i32}
              ascvec.vf(%87, %arg8, %alloc_15, %alloc_7, %38, %alloc_5 : memref<2048xf32, 1>, f32, memref<2048xi8, 1>, memref<128xf32, 1>, i64, memref<128xf32, 1>) {
              ^bb0(%arg36: memref<2048xf32, 1>, %arg37: f32, %arg38: memref<2048xi8, 1>, %arg39: memref<128xf32, 1>, %arg40: i64, %arg41: memref<128xf32, 1>):
                %c1_i64_31 = arith.constant 1 : i64
                %c0_i64_32 = arith.constant 0 : i64
                %c8_i64_33 = arith.constant 8 : i64
                %cst = arith.constant -1.000000e+30 : f32
                %cst_34 = arith.constant -3.000000e+38 : f32
                %cst_35 = arith.constant 0.000000e+00 : f32
                %c64_i64 = arith.constant 64 : i64
                %c128_i64_36 = arith.constant 128 : i64
                %c64_i32 = arith.constant 64 : i32
                %mask, %remaining_out = ascvec.update_mask %c64_i32 : !ascvec.mask
                scf.for %arg42 = %c0_i64_32 to %arg40 step %c1_i64_31  : i64 {
                  %125 = arith.muli %arg42, %c128_i64_36 : i64
                  %126 = arith.index_cast %125 : i64 to index
                  %127 = ascvec.loadalign %arg36[%126] : memref<2048xf32, 1>, index -> <f32>
                  %128 = ascvec.mul_scalar %127, %arg37, %mask : <f32>, f32
                  %129 = arith.addi %125, %c64_i64 : i64
                  %130 = arith.index_cast %129 : i64 to index
                  %131 = ascvec.loadalign %arg36[%130] : memref<2048xf32, 1>, index -> <f32>
                  %132 = ascvec.mul_scalar %131, %arg37, %mask : <f32>, f32
                  %133 = ascvec.loadalign_unpack %arg38[%126] {mode = "b8_to_b32"} : memref<2048xi8, 1>, index -> <ui32>
                  %134 = ascvec.reinterpret %133 : <ui32> -> <i32>
                  %135 = ascvec.cast %134, %mask : <i32> -> <f32>
                  %136 = ascvec.cmp_eq_scalar %135, %cst_35, %mask : <f32>, f32 -> !ascvec.mask
                  %137 = ascvec.duplicate_scalar %cst_34, %mask : f32 -> <f32>
                  %138 = ascvec.select %128, %137, %136 : <f32>
                  %139 = ascvec.loadalign_unpack %arg38[%130] {mode = "b8_to_b32"} : memref<2048xi8, 1>, index -> <ui32>
                  %140 = ascvec.reinterpret %139 : <ui32> -> <i32>
                  %141 = ascvec.cast %140, %mask : <i32> -> <f32>
                  %142 = ascvec.cmp_eq_scalar %141, %cst_35, %mask : <f32>, f32 -> !ascvec.mask
                  %143 = ascvec.select %132, %137, %142 : <f32>
                  ascvec.storealign %arg36[%126], %138, %mask : memref<2048xf32, 1>, index, <f32>
                  ascvec.storealign %arg36[%130], %143, %mask : memref<2048xf32, 1>, index, <f32>
                  %144 = ascvec.max %138, %143, %mask : <f32>
                  %145 = ascvec.reduce_max %144, %mask : <f32>
                  %146 = ascvec.cmp_le_scalar %145, %cst, %mask : <f32>, f32 -> !ascvec.mask
                  %147 = ascvec.duplicate_scalar %cst_35, %mask : f32 -> <f32>
                  %148 = ascvec.select %147, %145, %146 : <f32>
                  %149 = arith.muli %arg42, %c8_i64_33 : i64
                  %150 = arith.index_cast %149 : i64 to index
                  ascvec.storealign_1st %arg39[%150], %148 : memref<128xf32, 1>, index, <f32>
                }
                ascvec.mem_bar vst_vld
                scf.for %arg42 = %c0_i64_32 to %arg40 step %c1_i64_31  : i64 {
                  %125 = arith.muli %arg42, %c8_i64_33 : i64
                  %126 = arith.index_cast %125 : i64 to index
                  %127 = ascvec.loadalign_brc %arg41[%126] : memref<128xf32, 1>, index -> <f32>
                  %128 = ascvec.loadalign_brc %arg39[%126] : memref<128xf32, 1>, index -> <f32>
                  %129 = ascvec.max %127, %128, %mask : <f32>
                  ascvec.storealign_1st %arg39[%126], %129 : memref<128xf32, 1>, index, <f32>
                }
              } {cannir.vf_mode = "simd"}
              ascvec.get_buf v, %95, 0 {cannir.buffer.generated_sync}
              %124 = ascscalar.bitcast_memref %94 byte_offset = %c0_i64 : memref<2048xi8, 1> -> memref<?xf8E4M3FN, 1>
              ascvec.vf(%alloc_7, %87, %alloc_8, %124, %38 : memref<128xf32, 1>, memref<2048xf32, 1>, memref<128xf32, 1>, memref<?xf8E4M3FN, 1>, i64) {
              ^bb0(%arg36: memref<128xf32, 1>, %arg37: memref<2048xf32, 1>, %arg38: memref<128xf32, 1>, %arg39: memref<?xf8E4M3FN, 1>, %arg40: i64):
                %c1_i64_31 = arith.constant 1 : i64
                %c0_i64_32 = arith.constant 0 : i64
                %c16 = arith.constant 16 : index
                %c32_i64_33 = arith.constant 32 : i64
                %c16_i64_34 = arith.constant 16 : i64
                %cst = arith.constant 4.480000e+02 : f32
                %c64_i64 = arith.constant 64 : i64
                %c0_35 = arith.constant 0 : index
                %c8_i64_36 = arith.constant 8 : i64
                %c128_i64_37 = arith.constant 128 : i64
                %c128_i32 = arith.constant 128 : i32
                %c256_i32_38 = arith.constant 256 : i32
                %c64_i32 = arith.constant 64 : i32
                %mask, %remaining_out = ascvec.update_mask %c64_i32 : !ascvec.mask
                %mask_39, %remaining_out_40 = ascvec.update_mask %c256_i32_38 elem_bits 8 : !ascvec.mask
                %mask_41, %remaining_out_42 = ascvec.update_mask %c128_i32 elem_bits 8 : !ascvec.mask
                scf.for %arg41 = %c0_i64_32 to %arg40 step %c1_i64_31  : i64 {
                  %125 = arith.muli %arg41, %c128_i64_37 : i64
                  %126 = arith.muli %arg41, %c8_i64_36 : i64
                  %127 = arith.index_cast %126 : i64 to index
                  %128 = ascvec.loadalign_brc %arg36[%127] : memref<128xf32, 1>, index -> <f32>
                  %129 = arith.index_cast %125 : i64 to index
                  %130 = ascvec.loadalign %arg37[%129] : memref<2048xf32, 1>, index -> <f32>
                  %131 = ascvec.exp_sub %130, %128, %mask : <f32>, <f32> -> <f32>
                  %132 = arith.addi %125, %c64_i64 : i64
                  %133 = arith.index_cast %132 : i64 to index
                  %134 = ascvec.loadalign %arg37[%133] : memref<2048xf32, 1>, index -> <f32>
                  %135 = ascvec.exp_sub %134, %128, %mask : <f32>, <f32> -> <f32>
                  %136 = ascvec.add %131, %135, %mask : <f32>
                  %137 = ascvec.reduce_sum %136, %mask : <f32>
                  ascvec.storealign_1st %arg38[%127], %137 : memref<128xf32, 1>, index, <f32>
                  %138 = ascvec.mul_scalar %131, %cst, %mask : <f32>, f32
                  %139 = ascvec.mul_scalar %135, %cst, %mask : <f32>, f32
                  %result0, %result1 = ascvec.deintlv %138, %139 : <f32>, <f32> -> <f32>, <f32>
                  %140 = ascvec.cast %result0, %mask {reg_layout = 0 : i32, saturate = true} : <f32> -> <f8E4M3FN>
                  %141 = ascvec.cast %result1, %mask {reg_layout = 2 : i32, saturate = true} : <f32> -> <f8E4M3FN>
                  %142 = ascvec.or %140, %141, %mask_39 : <f8E4M3FN>, <f8E4M3FN> -> <f8E4M3FN>
                  %result0_43, %result1_44 = ascvec.deintlv %142, %142 : <f8E4M3FN>, <f8E4M3FN> -> <f8E4M3FN>, <f8E4M3FN>
                  %143 = arith.remsi %arg41, %c16_i64_34 : i64
                  %144 = arith.muli %143, %c32_i64_33 overflow<nsw> : i64
                  %145 = arith.index_cast %144 : i64 to index
                  ascvec.storealign_strided %arg39[%145], %result0_43, %mask_41, %c16, %c0_35 : memref<?xf8E4M3FN, 1>, <f8E4M3FN>
                }
              } {cannir.vf_mode = "simd"}
              ascvec.release_buf v, %95, 0 {cannir.buffer.generated_sync}
              ascvec.sync_intra_arrive v, %123 {cannir.sync.banks = 3 : i32}
              ascvec.vf(%alloc_5, %alloc_7, %alloc_6, %alloc_8, %117, %38 : memref<128xf32, 1>, memref<128xf32, 1>, memref<128xf32, 1>, memref<128xf32, 1>, memref<128xf32, 1>, i64) {
              ^bb0(%arg36: memref<128xf32, 1>, %arg37: memref<128xf32, 1>, %arg38: memref<128xf32, 1>, %arg39: memref<128xf32, 1>, %arg40: memref<128xf32, 1>, %arg41: i64):
                %c1_i64_31 = arith.constant 1 : i64
                %c0_i64_32 = arith.constant 0 : i64
                %c8_i64_33 = arith.constant 8 : i64
                %c64_i32 = arith.constant 64 : i32
                %mask, %remaining_out = ascvec.update_mask %c64_i32 : !ascvec.mask
                scf.for %arg42 = %c0_i64_32 to %arg41 step %c1_i64_31  : i64 {
                  %125 = arith.muli %arg42, %c8_i64_33 : i64
                  %126 = arith.index_cast %125 : i64 to index
                  %127 = ascvec.loadalign_brc %arg36[%126] : memref<128xf32, 1>, index -> <f32>
                  %128 = ascvec.loadalign_brc %arg37[%126] : memref<128xf32, 1>, index -> <f32>
                  %129 = ascvec.exp_sub %127, %128, %mask : <f32>, <f32> -> <f32>
                  %130 = ascvec.loadalign_brc %arg38[%126] : memref<128xf32, 1>, index -> <f32>
                  %131 = ascvec.loadalign_brc %arg39[%126] : memref<128xf32, 1>, index -> <f32>
                  %132 = ascvec.madd_two_address %130, %129, %131, %mask : <f32>
                  ascvec.storealign_1st %arg40[%126], %129 : memref<128xf32, 1>, index, <f32>
                  ascvec.storealign_1st %arg36[%126], %128 : memref<128xf32, 1>, index, <f32>
                  ascvec.storealign_1st %arg38[%126], %132 : memref<128xf32, 1>, index, <f32>
                }
              } {cannir.vf_mode = "simd"}
              scf.yield %119 : i64
            }
            %99 = ascscalar.memref_address %alloc_30 : memref<4096xf8E4M3FN, 2>
            %100 = ascscalar.memref_address %alloc_29 : memref<4096xf8E4M3FN, 2>
            %101 = arith.select %90, %99, %100 : i64
            %102 = arith.select %90, %c30_i64, %c31_i64 : i64
            %103 = ascscalar.bind_local_ptr %93 : memref<1xf8E4M3FN, 1>
            %104 = ascscalar.bind_local_ptr %101 : memref<1xf8E4M3FN, 2>
            %105 = arith.muli %1, %c16_i64 : i64
            %106 = arith.index_cast %105 : i64 to index
            %107 = arith.muli %106, %c32 : index
            %108 = arith.cmpi eq, %102, %c31_i64 : i64
            %109 = arith.select %108, %c8_i64, %c7_i64 : i64
            ascvec.sync_intra_wait mte3, %109 {cannir.sync.banks = 3 : i32}
            ascvec.get_buf mte3, %95, 0 {cannir.buffer.generated_sync}
            ascvec.copy_ub2l1_strided %104[%107], %103[%c0], %c4_i32, %c16_i32, %c0_i32, %c16_i32 : memref<1xf8E4M3FN, 2>, memref<1xf8E4M3FN, 1>
            ascvec.release_buf mte3, %95, 0 {cannir.buffer.generated_sync}
            ascvec.sync_intra_arrive mte3, %109 {cannir.sync.banks = 3 : i32}
            scf.yield %74, %98, %81, %97 : i64, i64, i64, i64
          } else {
            scf.yield %arg26, %arg30, %arg32, %arg33 : i64, i64, i64, i64
          }
          %60 = arith.cmpi sge, %arg24, %c2_i64 : i64
          %61 = arith.addi %27, %c2_i64 : i64
          %62 = arith.cmpi slt, %arg24, %61 : i64
          %63 = arith.select %60, %62, %false : i1
          %64:4 = scf.if %63 -> (i64, i64, i64, i64) {
            %67 = arith.addi %arg34, %c1_i64 : i64
            %68 = arith.remui %67, %c2_i64 : i64
            %69 = arith.addi %arg27, %c1_i64 : i64
            %70 = arith.remui %69, %c3_i64 : i64
            %71 = arith.addi %54#1, %c1_i64 : i64
            %72 = arith.remui %71, %c2_i64 : i64
            %73 = arith.addi %54#2, %c1_i64 : i64
            %74 = arith.remui %73, %c2_i64 : i64
            scf.yield %70, %72, %74, %68 : i64, i64, i64, i64
          } else {
            scf.yield %arg27, %54#1, %54#2, %arg34 : i64, i64, i64, i64
          }
          %65 = arith.cmpi sge, %arg24, %c3_i64 : i64
          %66:2 = scf.if %65 -> (i64, i64) {
            %67 = arith.cmpi eq, %arg24, %c3_i64 : i64
            %68 = arith.cmpi eq, %arg35, %c0_i64 : i64
            %69 = ascscalar.memref_address %alloc_25 : memref<2048xf32, 1>
            %70 = ascscalar.memref_address %alloc_26 : memref<2048xf32, 1>
            %71 = arith.select %68, %69, %70 : i64
            %72 = ascscalar.bind_local_ptr %71 : memref<2048xf32, 1>
            %73 = arith.select %68, %c28_i64, %c29_i64 : i64
            %74 = arith.addi %arg35, %c1_i64 : i64
            %75 = arith.remui %74, %c2_i64 : i64
            %76 = scf.if %67 -> (i64) {
              %77 = arith.cmpi eq, %73, %c29_i64 : i64
              %78 = arith.select %77, %c6_i64, %c5_i64 : i64
              ascvec.sync_intra_wait v, %78 {cannir.sync.banks = 3 : i32}
              ascvec.vf(%72, %alloc_12, %38 : memref<2048xf32, 1>, memref<2048xf32, 1>, i64) {
              ^bb0(%arg36: memref<2048xf32, 1>, %arg37: memref<2048xf32, 1>, %arg38: i64):
                %c1_i64_31 = arith.constant 1 : i64
                %c0_i64_32 = arith.constant 0 : i64
                %c1_i32_33 = arith.constant 1 : i32
                %c2_i32 = arith.constant 2 : i32
                %c0_i32_34 = arith.constant 0 : i32
                %c128_i64_35 = arith.constant 128 : i64
                %c64_i32 = arith.constant 64 : i32
                %mask, %remaining_out = ascvec.update_mask %c64_i32 : !ascvec.mask
                scf.for %arg39 = %c0_i64_32 to %arg38 step %c1_i64_31  : i64 {
                  scf.for %arg40 = %c0_i32_34 to %c2_i32 step %c1_i32_33  : i32 {
                    %79 = arith.muli %arg39, %c128_i64_35 : i64
                    %80 = arith.muli %arg40, %c64_i32 overflow<nsw> : i32
                    %81 = arith.extsi %80 : i32 to i64
                    %82 = arith.addi %79, %81 : i64
                    %83 = arith.index_cast %82 : i64 to index
                    %84 = ascvec.loadalign %arg36[%83] : memref<2048xf32, 1>, index -> <f32>
                    ascvec.storealign %arg37[%83], %84, %mask : memref<2048xf32, 1>, index, <f32>
                  }
                }
              } {cannir.vf_mode = "simd"}
              ascvec.sync_intra_arrive v, %78 {cannir.sync.banks = 3 : i32}
              scf.yield %arg31 : i64
            } else {
              %77 = arith.cmpi eq, %arg31, %c1_i64 : i64
              %78 = ascscalar.memref_address %alloc_10 : memref<128xf32, 1>
              %79 = ascscalar.memref_address %alloc_11 : memref<128xf32, 1>
              %80 = arith.select %77, %78, %79 : i64
              %81 = arith.cmpi eq, %arg31, %c0_i64 : i64
              %82 = ascscalar.memref_address %alloc_9 : memref<128xf32, 1>
              %83 = arith.select %81, %82, %80 : i64
              %84 = ascscalar.bind_local_ptr %83 : memref<128xf32, 1>
              %85 = arith.addi %arg31, %c1_i64 : i64
              %86 = arith.remui %85, %c3_i64 : i64
              %87 = arith.cmpi eq, %73, %c29_i64 : i64
              %88 = arith.select %87, %c6_i64, %c5_i64 : i64
              ascvec.sync_intra_wait v, %88 {cannir.sync.banks = 3 : i32}
              ascvec.vf(%84, %72, %alloc_12, %38 : memref<128xf32, 1>, memref<2048xf32, 1>, memref<2048xf32, 1>, i64) {
              ^bb0(%arg36: memref<128xf32, 1>, %arg37: memref<2048xf32, 1>, %arg38: memref<2048xf32, 1>, %arg39: i64):
                %c1_i64_31 = arith.constant 1 : i64
                %c0_i64_32 = arith.constant 0 : i64
                %c1_i32_33 = arith.constant 1 : i32
                %c2_i32 = arith.constant 2 : i32
                %c0_i32_34 = arith.constant 0 : i32
                %c128_i64_35 = arith.constant 128 : i64
                %c8_i64_36 = arith.constant 8 : i64
                %c64_i32 = arith.constant 64 : i32
                %mask, %remaining_out = ascvec.update_mask %c64_i32 : !ascvec.mask
                scf.for %arg40 = %c0_i64_32 to %arg39 step %c1_i64_31  : i64 {
                  %89 = arith.muli %arg40, %c8_i64_36 : i64
                  %90 = arith.index_cast %89 : i64 to index
                  %91 = ascvec.loadalign_brc %arg36[%90] : memref<128xf32, 1>, index -> <f32>
                  scf.for %arg41 = %c0_i32_34 to %c2_i32 step %c1_i32_33  : i32 {
                    %92 = arith.muli %arg40, %c128_i64_35 : i64
                    %93 = arith.muli %arg41, %c64_i32 overflow<nsw> : i32
                    %94 = arith.extsi %93 : i32 to i64
                    %95 = arith.addi %92, %94 : i64
                    %96 = arith.index_cast %95 : i64 to index
                    %97 = ascvec.loadalign %arg37[%96] : memref<2048xf32, 1>, index -> <f32>
                    %98 = ascvec.loadalign %arg38[%96] : memref<2048xf32, 1>, index -> <f32>
                    %99 = ascvec.madd_two_address %98, %91, %97, %mask : <f32>
                    ascvec.storealign %arg38[%96], %99, %mask : memref<2048xf32, 1>, index, <f32>
                  }
                }
              } {cannir.vf_mode = "simd"}
              ascvec.sync_intra_arrive v, %88 {cannir.sync.banks = 3 : i32}
              scf.yield %86 : i64
            }
            scf.yield %76, %75 : i64, i64
          } else {
            scf.yield %arg31, %arg35 : i64, i64
          }
          scf.yield %54#0, %59#0, %64#0, %64#1, %64#2, %59#1, %66#0, %59#2, %59#3, %64#3, %66#1 : i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64
        }
        ascvec.get_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
        ascvec.vf(%alloc_6, %alloc_12, %alloc_18, %38 : memref<128xf32, 1>, memref<2048xf32, 1>, memref<2048xf16, 1>, i64) {
        ^bb0(%arg24: memref<128xf32, 1>, %arg25: memref<2048xf32, 1>, %arg26: memref<2048xf16, 1>, %arg27: i64):
          %c1_i64_31 = arith.constant 1 : i64
          %c0_i64_32 = arith.constant 0 : i64
          %c64_i64 = arith.constant 64 : i64
          %cst = arith.constant 0.00223214296 : f32
          %c128_i64_33 = arith.constant 128 : i64
          %c8_i64_34 = arith.constant 8 : i64
          %c128_i32 = arith.constant 128 : i32
          %c64_i32 = arith.constant 64 : i32
          %mask, %remaining_out = ascvec.update_mask %c64_i32 : !ascvec.mask
          %mask_35, %remaining_out_36 = ascvec.update_mask %c128_i32 elem_bits 16 : !ascvec.mask
          scf.for %arg28 = %c0_i64_32 to %arg27 step %c1_i64_31  : i64 {
            %53 = arith.muli %arg28, %c8_i64_34 : i64
            %54 = arith.index_cast %53 : i64 to index
            %55 = ascvec.loadalign_brc %arg24[%54] : memref<128xf32, 1>, index -> <f32>
            %56 = arith.muli %arg28, %c128_i64_33 : i64
            %57 = arith.index_cast %56 : i64 to index
            %58 = ascvec.loadalign %arg25[%57] : memref<2048xf32, 1>, index -> <f32>
            %59 = ascvec.div %58, %55, %mask : <f32>
            %60 = ascvec.mul_scalar %59, %cst, %mask : <f32>, f32
            %61 = arith.addi %56, %c64_i64 : i64
            %62 = arith.index_cast %61 : i64 to index
            %63 = ascvec.loadalign %arg25[%62] : memref<2048xf32, 1>, index -> <f32>
            %64 = ascvec.div %63, %55, %mask : <f32>
            %65 = ascvec.mul_scalar %64, %cst, %mask : <f32>, f32
            %66 = ascvec.cast %60, %mask {reg_layout = 0 : i32} : <f32> -> <f16>
            %67 = ascvec.cast %65, %mask {reg_layout = 0 : i32} : <f32> -> <f16>
            %result0, %result1 = ascvec.deintlv %66, %67 : <f16>, <f16> -> <f16>, <f16>
            ascvec.storealign %arg26[%57], %result0, %mask_35 : memref<2048xf16, 1>, index, <f16>
          }
        } {cannir.vf_mode = "simd"}
        ascvec.release_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
        scf.yield %50#0, %47, %52#0, %52#1, %52#2, %52#3, %52#4, %52#5, %52#6, %52#7, %52#8, %50#1, %52#9, %52#10 : i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64
      } else {
        %42 = arith.addi %arg9, %c1_i64 : i64
        %43 = arith.cmpi slt, %42, %9 : i64
        %44:2 = scf.if %43 -> (i64, i64) {
          %45 = arith.index_cast %42 : i64 to index
          %46 = memref.load %alloc_3[%45] : memref<1xi64>
          %47 = arith.addi %16, %45 : index
          %48 = memref.load %alloc_3[%47] : memref<1xi64>
          %49 = arith.addi %19, %45 : index
          %50 = memref.load %alloc_3[%49] : memref<1xi64>
          %51 = arith.addi %25, %45 : index
          %52 = memref.load %alloc_3[%51] : memref<1xi64>
          %53 = arith.muli %50, %c32_i64 : i64
          %54 = arith.subi %c3_i64, %53 : i64
          %55 = arith.cmpi sgt, %54, %c32_i64 : i64
          %56 = arith.cmpi sgt, %54, %c0_i64 : i64
          %57 = arith.select %56, %54, %c0_i64 : i64
          %58 = arith.select %55, %c32_i64, %57 : i64
          %59 = arith.addi %58, %c1_i64 : i64
          %60 = arith.divsi %59, %c2_i64 : i64
          %61 = arith.divsi %58, %c2_i64 : i64
          %62 = arith.select %37, %60, %61 : i64
          %63 = arith.select %37, %c0_i64, %60 : i64
          %64 = arith.cmpi sgt, %52, %c0_i64 : i64
          %65:2 = scf.if %64 -> (i64, i64) {
            ascvec.sync_all
            %66 = scf.for %arg24 = %c0_i64 to %62 step %c1_i64 iter_args(%arg25 = %arg21) -> (i64)  : i64 {
              %82 = arith.addi %53, %63 : i64
              %83 = arith.addi %82, %arg24 : i64
              %84 = arith.muli %48, %c3_i64 : i64
              %85 = arith.addi %84, %83 : i64
              %86 = arith.cmpi eq, %arg25, %c0_i64 : i64
              %87 = ascscalar.memref_address %alloc_19 : memref<128xi8, 1>
              %88 = ascscalar.memref_address %alloc_20 : memref<128xi8, 1>
              %89 = arith.select %86, %87, %88 : i64
              %90 = arith.select %86, %c5_i64, %c6_i64 : i64
              %91 = arith.addi %arg25, %c1_i64 overflow<nsw> : i64
              %92 = arith.remui %91, %c2_i64 : i64
              %93 = cannir.extract_physical_layout %arg1 : !cannir.tensor<<gm, f8E4M3FN>, <<(-1, -1, 128)>, <(-1, 128, 1)>>, <<(-1, -1, 128)>, <(-1, 128, 1)>>, identity>
              %94 = cannir.extract_stride %93 : <<(-1, -1, 128)>, <(-1, 128, 1)>> -> <(-1, 128, 1)>
              %95 = cannir.extract_dim %94[0] : !cannir.stride<(-1, 128, 1)>
              %96 = arith.muli %46, %95 : i64
              %97 = arith.muli %85, %c128_i64 : i64
              %98 = arith.addi %96, %97 : i64
              %99 = ascscalar.bind_local_ptr %89 : memref<1xi8, 1>
              ascvec.get_buf mte2, %90, 0 {cannir.buffer.generated_sync}
              "ascvec.copy_gm2ub_plan"(%99, %alloc_0, %c0_i32, %c0_i64, %c0_i64, %98, %c1_i64, %c128_i64, %c0_i64, %c128_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = true, storage_dtype = f8E4M3FN, ub_base = -1 : i64, ub_capacity = 1 : i64}> : (memref<1xi8, 1>, memref<1xf8E4M3FN>, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
              ascvec.release_buf mte2, %90, 0 {cannir.buffer.generated_sync}
              %100 = ascscalar.bind_local_ptr %89 : memref<128xi8, 1>
              %101 = arith.remsi %arg24, %c16_i64 : i64
              %102 = arith.muli %101, %c32_i64 overflow<nsw> : i64
              %103 = arith.index_cast %102 : i64 to index
              ascvec.get_buf v, %90, 0 {cannir.buffer.generated_sync}
              %104 = ascscalar.bitcast_memref %100 byte_offset = %c0_i64 : memref<128xi8, 1> -> memref<?xf8E4M3FN, 1>
              ascvec.vf(%104, %alloc_21, %103 : memref<?xf8E4M3FN, 1>, memref<2048xi8, 1>, index) {
              ^bb0(%arg26: memref<?xf8E4M3FN, 1>, %arg27: memref<2048xi8, 1>, %arg28: index):
                %c16 = arith.constant 16 : index
                %c0_31 = arith.constant 0 : index
                %c128_i32 = arith.constant 128 : i32
                %c0_32 = arith.constant 0 : index
                %c1 = arith.constant 1 : index
                scf.for %arg29 = %c0_32 to %c1 step %c1 {
                  %mask, %remaining_out = ascvec.update_mask %c128_i32 elem_bits 8 : !ascvec.mask
                  %105 = ascvec.loadalign %arg26[%c0_31] : memref<?xf8E4M3FN, 1>, index -> <f8E4M3FN>
                  ascvec.storealign_strided %arg27[%arg28], %105, %mask, %c16, %c0_31 : memref<2048xi8, 1>, <f8E4M3FN>
                }
              } {cannir.vf_mode = "simd"}
              ascvec.release_buf v, %90, 0 {cannir.buffer.generated_sync}
              scf.yield %92 : i64
            }
            ascvec.sync_all
            %67 = arith.cmpi eq, %arg10, %c0_i64 : i64
            %68 = ascscalar.memref_address %alloc_28 : memref<4096xf8E4M3FN, 2>
            %69 = ascscalar.memref_address %alloc_27 : memref<4096xf8E4M3FN, 2>
            %70 = arith.select %67, %68, %69 : i64
            %71 = arith.cmpi ne, %arg10, %c0_i64 : i64
            %72 = arith.extui %71 : i1 to i64
            %73 = arith.addi %arg10, %c1_i64 overflow<nsw> : i64
            %74 = arith.remui %73, %c2_i64 : i64
            %75 = ascscalar.bitcast_memref %alloc_21 byte_offset = %c0_i64 : memref<2048xi8, 1> -> memref<?xf8E4M3FN, 1>
            %76 = ascscalar.bind_local_ptr %70 : memref<1xf8E4M3FN, 2>
            %77 = arith.muli %1, %c16_i64 : i64
            %78 = arith.index_cast %77 : i64 to index
            %79 = arith.muli %78, %c32 : index
            %80 = arith.cmpi eq, %72, %c1_i64 : i64
            %81 = arith.extui %80 : i1 to i64
            ascvec.sync_intra_wait mte3, %81 {cannir.sync.banks = 3 : i32}
            ascvec.copy_ub2l1_strided %76[%79], %75[%c0], %c4_i32, %c16_i32, %c0_i32, %c16_i32 : memref<1xf8E4M3FN, 2>, memref<?xf8E4M3FN, 1>
            ascvec.sync_intra_arrive mte3, %81 {cannir.sync.banks = 3 : i32}
            scf.yield %74, %66 : i64, i64
          } else {
            scf.yield %arg10, %arg21 : i64, i64
          }
          scf.yield %65#0, %65#1 : i64, i64
        } else {
          scf.yield %arg10, %arg21 : i64, i64
        }
        ascvec.get_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
        ascvec.vf(%alloc_18, %38 : memref<2048xf16, 1>, i64) {
        ^bb0(%arg24: memref<2048xf16, 1>, %arg25: i64):
          %c1_i64_31 = arith.constant 1 : i64
          %c0_i64_32 = arith.constant 0 : i64
          %c128_i64_33 = arith.constant 128 : i64
          %cst = arith.constant 0.000000e+00 : f16
          %c128_i32 = arith.constant 128 : i32
          %mask, %remaining_out = ascvec.update_mask %c128_i32 elem_bits 16 : !ascvec.mask
          %45 = ascvec.duplicate_scalar %cst, %mask : f16 -> <f16>
          scf.for %arg26 = %c0_i64_32 to %arg25 step %c1_i64_31  : i64 {
            %46 = arith.muli %arg26, %c128_i64_33 : i64
            %47 = arith.index_cast %46 : i64 to index
            ascvec.storealign %arg24[%47], %45, %mask : memref<2048xf16, 1>, index, <f16>
          }
        } {cannir.vf_mode = "simd"}
        ascvec.release_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
        scf.yield %44#0, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18, %arg19, %arg20, %44#1, %arg22, %arg23 : i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64
      }
      ascvec.sync_all
      scf.for %arg24 = %c0_i64 to %38 step %c1_i64  : i64 {
        %42 = arith.addi %28, %39 : i64
        %43 = arith.addi %42, %arg24 : i64
        %44 = arith.cmpi slt, %43, %c3_i64 : i64
        scf.if %44 {
          %45 = arith.muli %18, %c3_i64 : i64
          %46 = arith.addi %45, %43 : i64
          %47 = cannir.extract_physical_layout %arg0 : !cannir.tensor<<gm, f16>, <<(-1, -1, 128)>, <(-1, 128, 1)>>, <<(-1, -1, 128)>, <(-1, 128, 1)>>, identity>
          %48 = cannir.extract_stride %47 : <<(-1, -1, 128)>, <(-1, 128, 1)>> -> <(-1, 128, 1)>
          %49 = cannir.extract_dim %48[0] : !cannir.stride<(-1, 128, 1)>
          %50 = arith.muli %12, %49 : i64
          %51 = arith.muli %46, %c128_i64 : i64
          %52 = arith.addi %50, %51 : i64
          %53 = arith.muli %52, %c16_i64 : i64
          %54 = ascscalar.memref_address %alloc : memref<1xf16>
          %55 = arith.divui %53, %c8_i64 : i64
          %56 = arith.addi %54, %55 : i64
          %57 = ascscalar.bind_gm_ptr %56 : memref<1xf16>
          %58 = arith.muli %arg24, %c2048_i64 : i64
          %59 = ascscalar.memref_address %alloc_18 : memref<2048xf16, 1>
          %60 = arith.divui %58, %c8_i64 : i64
          %61 = arith.addi %59, %60 : i64
          %62 = ascscalar.bind_local_ptr %61 : memref<1xf16, 1>
          ascvec.get_buf mte3, %c4_i64, 0 {cannir.buffer.generated_sync}
          ascvec.copy_ub2gm %57[%c0], %62[%c0], %c1_i32, %c256_i32, %c0_i64, %c256_i32 l2_cache_ctl %c0_i32 : memref<1xf16>, memref<1xf16, 1>
          ascvec.release_buf mte3, %c4_i64, 0 {cannir.buffer.generated_sync}
        }
      }
      scf.yield %41#0, %41#1, %41#2, %41#3, %41#4, %41#5, %41#6, %41#7, %41#8, %41#9, %41#10, %41#11, %41#12, %41#13 : i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64
    }
    ascvec.sync_intra_wait mte3, %c0_i64
    ascvec.sync_intra_wait mte3, %c1_i64
    ascvec.sync_intra_wait mte3, %c7_i64
    ascvec.sync_intra_wait mte3, %c8_i64
    return
  }
  func.func @kernel_QwenSparseAttnBlock128___call___0_aic(%arg0: !cannir.tensor<<gm, f16>, <<(-1, -1, 128)>, <(-1, 128, 1)>>, <<(-1, -1, 128)>, <(-1, 128, 1)>>, identity>, %arg1: !cannir.tensor<<gm, f8E4M3FN>, <<(-1, -1, 128)>, <(-1, 128, 1)>>, <<(-1, -1, 128)>, <(-1, 128, 1)>>, identity>, %arg2: !cannir.tensor<<gm, f8E4M3FN>, <<(-1, -1, 128, 128)>, <(-1, 128, -1, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 128, -1, 1)>>, identity>, %arg3: !cannir.tensor<<gm, f8E4M3FN>, <<(-1, -1, 128, 128)>, <(-1, 128, -1, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 128, -1, 1)>>, identity>, %arg4: !cannir.tensor<<gm, i8>, <<(2048, 2048)>, <(2048, 1)>>, <<(2048, 2048)>, <(2048, 1)>>, identity>, %arg5: !cannir.tensor<<gm, i64>, <<(2, -1)>, <(-1, 1)>>, <<(2, -1)>, <(-1, 1)>>, identity>, %arg6: !cannir.tensor<<gm, i64>, <<(5, -1)>, <(-1, 1)>>, <<(5, -1)>, <(-1, 1)>>, identity>, %arg7: !cannir.tensor<<gm, i32>, <<(2, -1)>, <(-1, 1)>>, <<(2, -1)>, <(-1, 1)>>, identity>, %arg8: f32) attributes {cannir.buffer.auto_sync, cannir.buffer.sync_analyzed, cannir.buffer.sync_verified, cannir.buffer.tensor_state, cannir.core = "aic", cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.device, cannir.explicit_ring_buffer_cursors, cannir.kernel_group = "kernel_QwenSparseAttnBlock128___call___0", cannir.resource.ids_prepared} {
    %c32_i32 = arith.constant 32 : i32
    %c8_i32 = arith.constant 8 : i32
    %c4_i32 = arith.constant 4 : i32
    %c4 = arith.constant 4 : index
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c16_i64 = arith.constant 16 : i64
    %c8_i64 = arith.constant 8 : i64
    %c7_i64 = arith.constant 7 : i64
    %c6_i64 = arith.constant 6 : i64
    %c5_i64 = arith.constant 5 : i64
    %c4_i64 = arith.constant 4 : i64
    %c3_i64 = arith.constant 3 : i64
    %c2_i64 = arith.constant 2 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %c9_i64 = arith.constant 9 : i64
    %c10_i64 = arith.constant 10 : i64
    %c11_i64 = arith.constant 11 : i64
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
    %c28_i64 = arith.constant 28 : i64
    %c29_i64 = arith.constant 29 : i64
    %c30_i64 = arith.constant 30 : i64
    %c31_i64 = arith.constant 31 : i64
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %false = arith.constant false
    %c0 = arith.constant 0 : index
    %c128_i32 = arith.constant 128 : i32
    %c128_i64 = arith.constant 128 : i64
    %c0_i32 = arith.constant 0 : i32
    %c32_i64 = arith.constant 32 : i64
    %alloc = memref.alloc() {cannir.arg_index = 2 : i64} : memref<1xf8E4M3FN>
    %alloc_0 = memref.alloc() {cannir.arg_index = 3 : i64} : memref<1xf8E4M3FN>
    %alloc_1 = memref.alloc() {cannir.arg_index = 5 : i64} : memref<1xi64>
    %alloc_2 = memref.alloc() {cannir.arg_index = 6 : i64} : memref<1xi64>
    %alloc_3 = memref.alloc() {cannir.arg_index = 7 : i64} : memref<1xi32>
    %alloc_4 = memref.alloc() {cannir.byte_offset = 24576 : i64} : memref<2048xf32, 1>
    %alloc_5 = memref.alloc() {cannir.byte_offset = 32768 : i64} : memref<2048xf32, 1>
    %alloc_6 = memref.alloc() {cannir.byte_offset = 40960 : i64} : memref<2048xf32, 1>
    %alloc_7 = memref.alloc() {cannir.byte_offset = 49152 : i64} : memref<2048xf32, 1>
    %alloc_8 = memref.alloc() {cannir.byte_offset = 57344 : i64} : memref<2048xf32, 1>
    %alloc_9 = memref.alloc() {cannir.byte_offset = 4096 : i64} : memref<4096xf8E4M3FN, 2>
    %alloc_10 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<4096xf8E4M3FN, 2>
    %alloc_11 = memref.alloc() {cannir.byte_offset = 16384 : i64} : memref<4096xf32, 5>
    %alloc_12 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<4096xf32, 5>
    %alloc_13 = memref.alloc() {cannir.byte_offset = 16384 : i64} : memref<16384xf8E4M3FN, 4>
    %alloc_14 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<16384xf8E4M3FN, 4>
    %alloc_15 = memref.alloc() {cannir.byte_offset = 4096 : i64} : memref<4096xf8E4M3FN, 3>
    %alloc_16 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<4096xf8E4M3FN, 3>
    %alloc_17 = memref.alloc() {cannir.byte_offset = 57344 : i64} : memref<16384xf8E4M3FN, 2>
    %alloc_18 = memref.alloc() {cannir.byte_offset = 90112 : i64} : memref<16384xf8E4M3FN, 2>
    %alloc_19 = memref.alloc() {cannir.byte_offset = 73728 : i64} : memref<16384xf8E4M3FN, 2>
    %alloc_20 = memref.alloc() {cannir.byte_offset = 110592 : i64} : memref<4096xf8E4M3FN, 2>
    %alloc_21 = memref.alloc() {cannir.byte_offset = 106496 : i64} : memref<4096xf8E4M3FN, 2>
    %alloc_22 = memref.alloc() {cannir.byte_offset = 8192 : i64} : memref<16384xf8E4M3FN, 2>
    %alloc_23 = memref.alloc() {cannir.byte_offset = 40960 : i64} : memref<16384xf8E4M3FN, 2>
    %alloc_24 = memref.alloc() {cannir.byte_offset = 24576 : i64} : memref<16384xf8E4M3FN, 2>
    asccube.sync_intra_arrive mte1, %c8_i64
    asccube.sync_intra_arrive mte1, %c24_i64
    asccube.sync_intra_arrive mte1, %c7_i64
    asccube.sync_intra_arrive mte1, %c23_i64
    asccube.sync_intra_arrive mte1, %c1_i64
    asccube.sync_intra_arrive mte1, %c17_i64
    asccube.sync_intra_arrive mte1, %c0_i64
    asccube.sync_intra_arrive mte1, %c16_i64
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
    %10:14 = scf.for %arg9 = %3 to %9 step %c1_i64 iter_args(%arg10 = %c0_i64, %arg11 = %c0_i64, %arg12 = %c0_i64, %arg13 = %c0_i64, %arg14 = %c0_i64, %arg15 = %c0_i64, %arg16 = %c0_i64, %arg17 = %c0_i64, %arg18 = %c0_i64, %arg19 = %c0_i64, %arg20 = %c0_i64, %arg21 = %c0_i64, %arg22 = %c0_i64, %arg23 = %c0_i64) -> (i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64)  : i64 {
      %11 = arith.index_cast %arg9 : i64 to index
      %12 = cannir.extract_layout %arg6 : !cannir.tensor<<gm, i64>, <<(5, -1)>, <(-1, 1)>>, <<(5, -1)>, <(-1, 1)>>, identity>
      %13 = cannir.extract_stride %12 : <<(5, -1)>, <(-1, 1)>> -> <(-1, 1)>
      %14 = cannir.extract_dim %13[0] : !cannir.stride<(-1, 1)>
      %15 = arith.index_cast %14 : i64 to index
      %16 = arith.addi %15, %11 : index
      %17 = memref.load %alloc_2[%16] : memref<1xi64>
      %18 = arith.muli %15, %c2 : index
      %19 = arith.addi %18, %11 : index
      %20 = memref.load %alloc_2[%19] : memref<1xi64>
      %21 = arith.muli %15, %c3 : index
      %22 = arith.addi %21, %11 : index
      %23 = memref.load %alloc_2[%22] : memref<1xi64>
      %24 = arith.muli %15, %c4 : index
      %25 = arith.addi %24, %11 : index
      %26 = memref.load %alloc_2[%25] : memref<1xi64>
      %27 = arith.muli %20, %c32_i64 : i64
      %28 = arith.subi %c3_i64, %27 : i64
      %29 = arith.cmpi sgt, %28, %c32_i64 : i64
      %30 = arith.cmpi sgt, %28, %c0_i64 : i64
      %31 = arith.select %30, %28, %c0_i64 : i64
      %32 = arith.select %29, %c32_i64, %31 : i64
      %33 = arith.addi %32, %c1_i64 : i64
      %34 = arith.divsi %33, %c2_i64 : i64
      %35 = arith.divsi %32, %c2_i64 : i64
      %36 = arith.cmpi eq, %1, %c0_i64 : i64
      %37 = arith.select %36, %34, %35 : i64
      %38 = arith.select %36, %c0_i64, %34 : i64
      %39 = arith.cmpi sgt, %26, %c0_i64 : i64
      %40:14 = scf.if %39 -> (i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) {
        %41 = arith.cmpi eq, %arg9, %3 : i64
        %42:2 = scf.if %41 -> (i64, i64) {
          %57 = scf.for %arg24 = %c0_i64 to %37 step %c1_i64 iter_args(%arg25 = %arg21) -> (i64)  : i64 {
            %60 = arith.addi %arg25, %c1_i64 overflow<nsw> : i64
            %61 = arith.remui %60, %c2_i64 : i64
            scf.yield %61 : i64
          }
          %58 = arith.addi %arg10, %c1_i64 overflow<nsw> : i64
          %59 = arith.remui %58, %c2_i64 : i64
          scf.yield %59, %57 : i64, i64
        } else {
          scf.yield %arg10, %arg21 : i64, i64
        }
        %43 = arith.cmpi eq, %arg11, %c0_i64 : i64
        %44 = ascscalar.memref_address %alloc_10 : memref<4096xf8E4M3FN, 2>
        %45 = ascscalar.memref_address %alloc_9 : memref<4096xf8E4M3FN, 2>
        %46 = arith.select %43, %44, %45 : i64
        %47 = arith.cmpi ne, %arg11, %c0_i64 : i64
        %48 = arith.extui %47 : i1 to i64
        %49 = arith.addi %arg11, %c1_i64 overflow<nsw> : i64
        %50 = arith.remui %49, %c2_i64 : i64
        %51 = arith.addi %arg9, %c1_i64 : i64
        %52 = arith.cmpi slt, %51, %9 : i64
        %53:2 = scf.if %52 -> (i64, i64) {
          %57 = arith.index_cast %51 : i64 to index
          %58 = arith.addi %18, %57 : index
          %59 = memref.load %alloc_2[%58] : memref<1xi64>
          %60 = arith.addi %24, %57 : index
          %61 = memref.load %alloc_2[%60] : memref<1xi64>
          %62 = arith.muli %59, %c32_i64 : i64
          %63 = arith.subi %c3_i64, %62 : i64
          %64 = arith.cmpi sgt, %63, %c32_i64 : i64
          %65 = arith.cmpi sgt, %63, %c0_i64 : i64
          %66 = arith.select %65, %63, %c0_i64 : i64
          %67 = arith.select %64, %c32_i64, %66 : i64
          %68 = arith.addi %67, %c1_i64 : i64
          %69 = arith.divsi %68, %c2_i64 : i64
          %70 = arith.divsi %67, %c2_i64 : i64
          %71 = arith.select %36, %69, %70 : i64
          %72 = arith.cmpi sgt, %61, %c0_i64 : i64
          %73:2 = scf.if %72 -> (i64, i64) {
            %74 = scf.for %arg24 = %c0_i64 to %71 step %c1_i64 iter_args(%arg25 = %42#1) -> (i64)  : i64 {
              %77 = arith.addi %arg25, %c1_i64 overflow<nsw> : i64
              %78 = arith.remui %77, %c2_i64 : i64
              scf.yield %78 : i64
            }
            %75 = arith.addi %42#0, %c1_i64 overflow<nsw> : i64
            %76 = arith.remui %75, %c2_i64 : i64
            scf.yield %76, %74 : i64, i64
          } else {
            scf.yield %42#0, %42#1 : i64, i64
          }
          scf.yield %73#0, %73#1 : i64, i64
        } else {
          scf.yield %42#0, %42#1 : i64, i64
        }
        %54 = arith.addi %26, %c3_i64 : i64
        %55 = arith.cmpi sgt, %54, %c0_i64 : i64
        scf.if %55 {
          %57 = arith.cmpi eq, %48, %c1_i64 : i64
          %58 = arith.extui %57 : i1 to i64
          asccube.sync_intra_wait mte1, %58 {cannir.sync.banks = 3 : i32}
          %59 = arith.addi %58, %c16_i64 overflow<nsw> : i64
          asccube.sync_intra_wait mte1, %59 {cannir.sync.banks = 3 : i32}
        }
        %56:11 = scf.for %arg24 = %c0_i64 to %54 step %c1_i64 iter_args(%arg25 = %arg12, %arg26 = %arg13, %arg27 = %arg14, %arg28 = %arg15, %arg29 = %arg16, %arg30 = %arg17, %arg31 = %arg18, %arg32 = %arg19, %arg33 = %arg20, %arg34 = %arg22, %arg35 = %arg23) -> (i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64)  : i64 {
          %57 = arith.cmpi slt, %arg24, %26 : i64
          %58:7 = scf.if %57 -> (i64, i64, i64, i64, i64, i64, i64) {
            %71 = arith.addi %23, %arg24 : i64
            %72 = arith.index_cast %71 : i64 to index
            %73 = memref.load %alloc_3[%72] : memref<1xi32>
            %74 = arith.extsi %73 : i32 to i64
            %75 = arith.cmpi eq, %arg25, %c1_i64 : i64
            %76 = ascscalar.memref_address %alloc_24 : memref<16384xf8E4M3FN, 2>
            %77 = ascscalar.memref_address %alloc_23 : memref<16384xf8E4M3FN, 2>
            %78 = arith.select %75, %76, %77 : i64
            %79 = arith.cmpi eq, %arg25, %c0_i64 : i64
            %80 = ascscalar.memref_address %alloc_22 : memref<16384xf8E4M3FN, 2>
            %81 = arith.select %79, %80, %78 : i64
            %82 = arith.select %75, %c1_i64, %c2_i64 : i64
            %83 = arith.select %79, %c0_i64, %82 : i64
            %84 = arith.addi %arg25, %c1_i64 : i64
            %85 = arith.remui %84, %c3_i64 : i64
            %86 = ascscalar.bind_local_ptr %81 : memref<1xf8E4M3FN, 2>
            %87 = cannir.extract_physical_layout %arg2 : !cannir.tensor<<gm, f8E4M3FN>, <<(-1, -1, 128, 128)>, <(-1, 128, -1, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 128, -1, 1)>>, identity>
            %88 = cannir.extract_stride %87 : <<(-1, -1, 128, 128)>, <(-1, 128, -1, 1)>> -> <(-1, 128, -1, 1)>
            %89 = cannir.extract_dim %88[0] : !cannir.stride<(-1, 128, -1, 1)>
            %90 = arith.muli %74, %89 : i64
            %91 = arith.index_cast %90 : i64 to index
            %92 = arith.muli %17, %c128_i64 : i64
            %93 = arith.index_cast %92 : i64 to index
            %94 = arith.addi %91, %93 : index
            %95 = cannir.extract_layout %arg2 : !cannir.tensor<<gm, f8E4M3FN>, <<(-1, -1, 128, 128)>, <(-1, 128, -1, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 128, -1, 1)>>, identity>
            %96 = cannir.extract_stride %95 : <<(-1, -1, 128, 128)>, <(-1, 128, -1, 1)>> -> <(-1, 128, -1, 1)>
            %97 = cannir.extract_dim %96[2] : !cannir.stride<(-1, 128, -1, 1)>
            asccube.get_buf mte2, %83, 0 {cannir.buffer.generated_sync}
            asccube.copy_gm2l1 %86[%c0], %alloc[%94], %c128_i32, %c128_i32, %97, %c1_i32, %c0_i64, %c0_i64, %c128_i32 l2_cache_ctl %c0_i32 {nd2nz = true} : memref<1xf8E4M3FN, 2>, memref<1xf8E4M3FN>
            asccube.release_buf mte2, %83, 0 {cannir.buffer.generated_sync}
            %98 = arith.cmpi eq, %arg29, %c0_i64 : i64
            %99 = ascscalar.memref_address %alloc_12 : memref<4096xf32, 5>
            %100 = ascscalar.memref_address %alloc_11 : memref<4096xf32, 5>
            %101 = arith.select %98, %99, %100 : i64
            %102 = arith.select %98, %c10_i64, %c11_i64 : i64
            %103 = arith.addi %arg29, %c1_i64 : i64
            %104 = arith.remui %103, %c2_i64 : i64
            %105 = arith.cmpi eq, %arg28, %c0_i64 : i64
            %106 = ascscalar.memref_address %alloc_16 : memref<4096xf8E4M3FN, 3>
            %107 = ascscalar.memref_address %alloc_15 : memref<4096xf8E4M3FN, 3>
            %108 = arith.select %105, %106, %107 : i64
            %109 = arith.select %105, %c6_i64, %c7_i64 : i64
            %110 = arith.addi %arg28, %c1_i64 : i64
            %111 = arith.remui %110, %c2_i64 : i64
            %112 = ascscalar.bind_local_ptr %108 : memref<1xf8E4M3FN, 3>
            %113 = ascscalar.bind_local_ptr %46 : memref<1xf8E4M3FN, 2>
            asccube.get_buf mte1, %109, 0 {cannir.buffer.generated_sync}
            asccube.copy_l12l0a %112[%c0], %113[%c0], %c0_i32, %c0_i32, %c2_i32, %c4_i32, %c2_i32, %c2_i32 : memref<1xf8E4M3FN, 3>, memref<1xf8E4M3FN, 2>
            asccube.release_buf mte1, %109, 0 {cannir.buffer.generated_sync}
            %114 = ascscalar.memref_address %alloc_14 : memref<16384xf8E4M3FN, 4>
            %115 = ascscalar.memref_address %alloc_13 : memref<16384xf8E4M3FN, 4>
            %116 = arith.select %105, %114, %115 : i64
            %117 = arith.select %105, %c8_i64, %c9_i64 : i64
            %118 = ascscalar.bind_local_ptr %116 : memref<1xf8E4M3FN, 4>
            asccube.get_buf mte1, %117, 0 {cannir.buffer.generated_sync}
            asccube.get_buf mte1, %83, 0 {cannir.buffer.generated_sync}
            asccube.copy_l12l0b %118[%c0], %86[%c0], %c0_i32, %c0_i32, %c8_i32, %c4_i32, %c8_i32, %c8_i32 : memref<1xf8E4M3FN, 4>, memref<1xf8E4M3FN, 2>
            asccube.release_buf mte1, %83, 0 {cannir.buffer.generated_sync}
            asccube.release_buf mte1, %117, 0 {cannir.buffer.generated_sync}
            %119 = ascscalar.bind_local_ptr %101 : memref<1xf32, 5>
            asccube.get_buf m, %102, 0 {cannir.buffer.generated_sync}
            asccube.get_buf m, %109, 0 {cannir.buffer.generated_sync}
            asccube.get_buf m, %117, 0 {cannir.buffer.generated_sync}
            asccube.mmad %119[%c0], %112, %118, %c32_i32, %c128_i32, %c128_i32, %c0_i32 {c_matrix_init_val = true} : memref<1xf32, 5>, memref<1xf8E4M3FN, 3>, memref<1xf8E4M3FN, 4>
            asccube.release_buf m, %117, 0 {cannir.buffer.generated_sync}
            asccube.release_buf m, %109, 0 {cannir.buffer.generated_sync}
            asccube.release_buf m, %102, 0 {cannir.buffer.generated_sync}
            %120 = ascscalar.memref_address %alloc_5 : memref<2048xf32, 1>
            %121 = ascscalar.memref_address %alloc_6 : memref<2048xf32, 1>
            %122 = arith.select %75, %120, %121 : i64
            %123 = ascscalar.memref_address %alloc_4 : memref<2048xf32, 1>
            %124 = arith.select %79, %123, %122 : i64
            %125 = arith.select %75, %c26_i64, %c27_i64 : i64
            %126 = arith.select %79, %c25_i64, %125 : i64
            %127 = ascscalar.bind_local_ptr %124 : memref<1xf32, 1>
            %128 = arith.cmpi eq, %126, %c26_i64 : i64
            %129 = arith.select %128, %c3_i64, %c2_i64 : i64
            %130 = arith.cmpi eq, %126, %c27_i64 : i64
            %131 = arith.select %130, %c4_i64, %129 : i64
            asccube.sync_intra_wait fixpipe, %131 {cannir.sync.banks = 3 : i32}
            %132 = arith.addi %131, %c16_i64 overflow<nsw> : i64
            asccube.sync_intra_wait fixpipe, %132 {cannir.sync.banks = 3 : i32}
            asccube.get_buf fixpipe, %102, 0 {cannir.buffer.generated_sync}
            asccube.copy_l0c2ub %127[%c0], %119[%c0], %c128_i32, %c32_i32, %c128_i32, %c32_i32, %c0_i32 sub_block_id %c0_i32 {dual_dst_ctl = 1 : i32, nz2nd_en = true, relu_en = false} : memref<1xf32, 1>, memref<1xf32, 5>
            asccube.release_buf fixpipe, %102, 0 {cannir.buffer.generated_sync}
            asccube.sync_intra_arrive fixpipe, %131 {cannir.sync.banks = 3 : i32}
            asccube.sync_intra_arrive fixpipe, %132 {cannir.sync.banks = 3 : i32}
            scf.yield %85, %111, %111, %111, %111, %104, %104 : i64, i64, i64, i64, i64, i64, i64
          } else {
            scf.yield %arg25, %arg28, %arg28, %arg28, %arg28, %arg29, %arg29 : i64, i64, i64, i64, i64, i64, i64
          }
          %59 = arith.cmpi sge, %arg24, %c1_i64 : i64
          %60 = arith.addi %26, %c1_i64 : i64
          %61 = arith.cmpi slt, %arg24, %60 : i64
          %62 = arith.select %59, %61, %false : i1
          %63:4 = scf.if %62 -> (i64, i64, i64, i64) {
            %71 = arith.addi %23, %arg24 : i64
            %72 = arith.subi %71, %c1_i64 : i64
            %73 = arith.index_cast %72 : i64 to index
            %74 = memref.load %alloc_3[%73] : memref<1xi32>
            %75 = arith.extsi %74 : i32 to i64
            %76 = arith.cmpi eq, %arg24, %c1_i64 : i64
            %77 = arith.cmpi eq, %arg26, %c1_i64 : i64
            %78 = ascscalar.memref_address %alloc_19 : memref<16384xf8E4M3FN, 2>
            %79 = ascscalar.memref_address %alloc_18 : memref<16384xf8E4M3FN, 2>
            %80 = arith.select %77, %78, %79 : i64
            %81 = arith.cmpi eq, %arg26, %c0_i64 : i64
            %82 = ascscalar.memref_address %alloc_17 : memref<16384xf8E4M3FN, 2>
            %83 = arith.select %81, %82, %80 : i64
            %84 = arith.select %77, %c4_i64, %c5_i64 : i64
            %85 = arith.select %81, %c3_i64, %84 : i64
            %86 = arith.addi %arg26, %c1_i64 : i64
            %87 = arith.remui %86, %c3_i64 : i64
            %88 = ascscalar.bind_local_ptr %83 : memref<1xf8E4M3FN, 2>
            %89 = cannir.extract_physical_layout %arg3 : !cannir.tensor<<gm, f8E4M3FN>, <<(-1, -1, 128, 128)>, <(-1, 128, -1, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 128, -1, 1)>>, identity>
            %90 = cannir.extract_stride %89 : <<(-1, -1, 128, 128)>, <(-1, 128, -1, 1)>> -> <(-1, 128, -1, 1)>
            %91 = cannir.extract_dim %90[0] : !cannir.stride<(-1, 128, -1, 1)>
            %92 = arith.muli %75, %91 : i64
            %93 = arith.index_cast %92 : i64 to index
            %94 = arith.muli %17, %c128_i64 : i64
            %95 = arith.index_cast %94 : i64 to index
            %96 = arith.addi %93, %95 : index
            %97 = cannir.extract_layout %arg3 : !cannir.tensor<<gm, f8E4M3FN>, <<(-1, -1, 128, 128)>, <(-1, 128, -1, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 128, -1, 1)>>, identity>
            %98 = cannir.extract_stride %97 : <<(-1, -1, 128, 128)>, <(-1, 128, -1, 1)>> -> <(-1, 128, -1, 1)>
            %99 = cannir.extract_dim %98[2] : !cannir.stride<(-1, 128, -1, 1)>
            asccube.get_buf mte2, %85, 0 {cannir.buffer.generated_sync}
            asccube.copy_gm2l1 %88[%c0], %alloc_0[%96], %c128_i32, %c128_i32, %99, %c1_i32, %c0_i64, %c0_i64, %c128_i32 l2_cache_ctl %c0_i32 {nd2nz = true} : memref<1xf8E4M3FN, 2>, memref<1xf8E4M3FN>
            asccube.release_buf mte2, %85, 0 {cannir.buffer.generated_sync}
            %100 = arith.addi %38, %37 : i64
            %101 = arith.maxsi %38, %c0_i64 : i64
            %102 = arith.minsi %100, %c3_i64 : i64
            %103 = arith.subi %102, %101 : i64
            %104 = arith.maxsi %103, %c0_i64 : i64
            %105 = arith.cmpi sgt, %104, %c0_i64 : i64
            %106 = scf.if %105 -> (i64) {
              %110 = arith.addi %arg32, %c1_i64 : i64
              %111 = arith.remui %110, %c2_i64 : i64
              scf.yield %111 : i64
            } else {
              scf.yield %arg32 : i64
            }
            %107 = arith.addi %arg33, %c1_i64 : i64
            %108 = arith.remui %107, %c2_i64 : i64
            %109 = scf.if %76 -> (i64) {
              scf.yield %arg30 : i64
            } else {
              %110 = arith.addi %arg30, %c1_i64 : i64
              %111 = arith.remui %110, %c3_i64 : i64
              scf.yield %111 : i64
            }
            scf.yield %87, %109, %106, %108 : i64, i64, i64, i64
          } else {
            scf.yield %arg26, %arg30, %arg32, %arg33 : i64, i64, i64, i64
          }
          %64 = arith.cmpi sge, %arg24, %c2_i64 : i64
          %65 = arith.addi %26, %c2_i64 : i64
          %66 = arith.cmpi slt, %arg24, %65 : i64
          %67 = arith.select %64, %66, %false : i1
          %68:4 = scf.if %67 -> (i64, i64, i64, i64) {
            %71 = arith.cmpi eq, %arg34, %c0_i64 : i64
            %72 = ascscalar.memref_address %alloc_21 : memref<4096xf8E4M3FN, 2>
            %73 = ascscalar.memref_address %alloc_20 : memref<4096xf8E4M3FN, 2>
            %74 = arith.select %71, %72, %73 : i64
            %75 = arith.select %71, %c30_i64, %c31_i64 : i64
            %76 = arith.addi %arg34, %c1_i64 : i64
            %77 = arith.remui %76, %c2_i64 : i64
            %78 = arith.cmpi eq, %arg27, %c1_i64 : i64
            %79 = ascscalar.memref_address %alloc_19 : memref<16384xf8E4M3FN, 2>
            %80 = ascscalar.memref_address %alloc_18 : memref<16384xf8E4M3FN, 2>
            %81 = arith.select %78, %79, %80 : i64
            %82 = arith.cmpi eq, %arg27, %c0_i64 : i64
            %83 = ascscalar.memref_address %alloc_17 : memref<16384xf8E4M3FN, 2>
            %84 = arith.select %82, %83, %81 : i64
            %85 = arith.select %78, %c4_i64, %c5_i64 : i64
            %86 = arith.select %82, %c3_i64, %85 : i64
            %87 = arith.addi %arg27, %c1_i64 : i64
            %88 = arith.remui %87, %c3_i64 : i64
            %89 = arith.cmpi eq, %58#1, %c0_i64 : i64
            %90 = ascscalar.memref_address %alloc_16 : memref<4096xf8E4M3FN, 3>
            %91 = ascscalar.memref_address %alloc_15 : memref<4096xf8E4M3FN, 3>
            %92 = arith.select %89, %90, %91 : i64
            %93 = arith.select %89, %c6_i64, %c7_i64 : i64
            %94 = arith.addi %58#1, %c1_i64 : i64
            %95 = arith.remui %94, %c2_i64 : i64
            %96 = ascscalar.bind_local_ptr %92 : memref<1xf8E4M3FN, 3>
            %97 = ascscalar.bind_local_ptr %74 : memref<1xf8E4M3FN, 2>
            %98 = arith.cmpi eq, %75, %c31_i64 : i64
            %99 = arith.select %98, %c8_i64, %c7_i64 : i64
            asccube.sync_intra_wait mte1, %99 {cannir.sync.banks = 3 : i32}
            %100 = arith.addi %99, %c16_i64 overflow<nsw> : i64
            asccube.sync_intra_wait mte1, %100 {cannir.sync.banks = 3 : i32}
            asccube.get_buf mte1, %93, 0 {cannir.buffer.generated_sync}
            asccube.copy_l12l0a %96[%c0], %97[%c0], %c0_i32, %c0_i32, %c2_i32, %c4_i32, %c2_i32, %c2_i32 : memref<1xf8E4M3FN, 3>, memref<1xf8E4M3FN, 2>
            asccube.release_buf mte1, %93, 0 {cannir.buffer.generated_sync}
            asccube.sync_intra_arrive mte1, %99 {cannir.sync.banks = 3 : i32}
            asccube.sync_intra_arrive mte1, %100 {cannir.sync.banks = 3 : i32}
            %101 = arith.cmpi eq, %58#3, %c0_i64 : i64
            %102 = ascscalar.memref_address %alloc_14 : memref<16384xf8E4M3FN, 4>
            %103 = ascscalar.memref_address %alloc_13 : memref<16384xf8E4M3FN, 4>
            %104 = arith.select %101, %102, %103 : i64
            %105 = arith.select %101, %c8_i64, %c9_i64 : i64
            %106 = ascscalar.bind_local_ptr %104 : memref<1xf8E4M3FN, 4>
            %107 = ascscalar.bind_local_ptr %84 : memref<1xf8E4M3FN, 2>
            asccube.get_buf mte1, %105, 0 {cannir.buffer.generated_sync}
            asccube.get_buf mte1, %86, 0 {cannir.buffer.generated_sync}
            asccube.copy_l12l0b %106[%c0], %107[%c0], %c0_i32, %c0_i32, %c8_i32, %c4_i32, %c8_i32, %c8_i32 {transpose = true} : memref<1xf8E4M3FN, 4>, memref<1xf8E4M3FN, 2>
            asccube.release_buf mte1, %86, 0 {cannir.buffer.generated_sync}
            asccube.release_buf mte1, %105, 0 {cannir.buffer.generated_sync}
            %108 = arith.cmpi eq, %58#5, %c0_i64 : i64
            %109 = ascscalar.memref_address %alloc_12 : memref<4096xf32, 5>
            %110 = ascscalar.memref_address %alloc_11 : memref<4096xf32, 5>
            %111 = arith.select %108, %109, %110 : i64
            %112 = arith.select %108, %c10_i64, %c11_i64 : i64
            %113 = arith.addi %58#5, %c1_i64 : i64
            %114 = arith.remui %113, %c2_i64 : i64
            %115 = arith.cmpi eq, %58#2, %c0_i64 : i64
            %116 = arith.select %115, %90, %91 : i64
            %117 = arith.select %115, %c6_i64, %c7_i64 : i64
            %118 = arith.cmpi eq, %58#4, %c0_i64 : i64
            %119 = arith.select %118, %102, %103 : i64
            %120 = arith.select %118, %c8_i64, %c9_i64 : i64
            %121 = ascscalar.bind_local_ptr %111 : memref<1xf32, 5>
            %122 = ascscalar.bind_local_ptr %116 : memref<1xf8E4M3FN, 3>
            %123 = ascscalar.bind_local_ptr %119 : memref<1xf8E4M3FN, 4>
            asccube.get_buf m, %112, 0 {cannir.buffer.generated_sync}
            asccube.get_buf m, %117, 0 {cannir.buffer.generated_sync}
            asccube.get_buf m, %120, 0 {cannir.buffer.generated_sync}
            asccube.mmad %121[%c0], %122, %123, %c32_i32, %c128_i32, %c128_i32, %c0_i32 {c_matrix_init_val = true} : memref<1xf32, 5>, memref<1xf8E4M3FN, 3>, memref<1xf8E4M3FN, 4>
            asccube.release_buf m, %120, 0 {cannir.buffer.generated_sync}
            asccube.release_buf m, %117, 0 {cannir.buffer.generated_sync}
            asccube.release_buf m, %112, 0 {cannir.buffer.generated_sync}
            %124 = ascscalar.memref_address %alloc_7 : memref<2048xf32, 1>
            %125 = ascscalar.memref_address %alloc_8 : memref<2048xf32, 1>
            %126 = arith.select %71, %124, %125 : i64
            %127 = arith.select %71, %c28_i64, %c29_i64 : i64
            %128 = arith.cmpi eq, %58#6, %c0_i64 : i64
            %129 = arith.select %128, %109, %110 : i64
            %130 = arith.select %128, %c10_i64, %c11_i64 : i64
            %131 = ascscalar.bind_local_ptr %126 : memref<1xf32, 1>
            %132 = ascscalar.bind_local_ptr %129 : memref<1xf32, 5>
            %133 = arith.cmpi eq, %127, %c29_i64 : i64
            %134 = arith.select %133, %c6_i64, %c5_i64 : i64
            asccube.sync_intra_wait fixpipe, %134 {cannir.sync.banks = 3 : i32}
            %135 = arith.addi %134, %c16_i64 overflow<nsw> : i64
            asccube.sync_intra_wait fixpipe, %135 {cannir.sync.banks = 3 : i32}
            asccube.get_buf fixpipe, %130, 0 {cannir.buffer.generated_sync}
            asccube.copy_l0c2ub %131[%c0], %132[%c0], %c128_i32, %c32_i32, %c128_i32, %c32_i32, %c0_i32 sub_block_id %c0_i32 {dual_dst_ctl = 1 : i32, nz2nd_en = true, relu_en = false} : memref<1xf32, 1>, memref<1xf32, 5>
            asccube.release_buf fixpipe, %130, 0 {cannir.buffer.generated_sync}
            asccube.sync_intra_arrive fixpipe, %134 {cannir.sync.banks = 3 : i32}
            asccube.sync_intra_arrive fixpipe, %135 {cannir.sync.banks = 3 : i32}
            scf.yield %88, %95, %114, %77 : i64, i64, i64, i64
          } else {
            scf.yield %arg27, %58#1, %58#5, %arg34 : i64, i64, i64, i64
          }
          %69 = arith.cmpi sge, %arg24, %c3_i64 : i64
          %70:2 = scf.if %69 -> (i64, i64) {
            %71 = arith.cmpi eq, %arg24, %c3_i64 : i64
            %72 = arith.addi %arg35, %c1_i64 : i64
            %73 = arith.remui %72, %c2_i64 : i64
            %74 = scf.if %71 -> (i64) {
              scf.yield %arg31 : i64
            } else {
              %75 = arith.addi %arg31, %c1_i64 : i64
              %76 = arith.remui %75, %c3_i64 : i64
              scf.yield %76 : i64
            }
            scf.yield %74, %73 : i64, i64
          } else {
            scf.yield %arg31, %arg35 : i64, i64
          }
          scf.yield %58#0, %63#0, %68#0, %68#1, %68#2, %63#1, %70#0, %63#2, %63#3, %68#3, %70#1 : i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64
        }
        scf.if %55 {
          %57 = arith.cmpi eq, %48, %c1_i64 : i64
          %58 = arith.extui %57 : i1 to i64
          asccube.sync_intra_arrive mte1, %58 {cannir.sync.banks = 3 : i32}
          %59 = arith.addi %58, %c16_i64 overflow<nsw> : i64
          asccube.sync_intra_arrive mte1, %59 {cannir.sync.banks = 3 : i32}
        }
        scf.yield %53#0, %50, %56#0, %56#1, %56#2, %56#3, %56#4, %56#5, %56#6, %56#7, %56#8, %53#1, %56#9, %56#10 : i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64
      } else {
        %41 = arith.addi %arg9, %c1_i64 : i64
        %42 = arith.cmpi slt, %41, %9 : i64
        %43:2 = scf.if %42 -> (i64, i64) {
          %44 = arith.index_cast %41 : i64 to index
          %45 = arith.addi %18, %44 : index
          %46 = memref.load %alloc_2[%45] : memref<1xi64>
          %47 = arith.addi %24, %44 : index
          %48 = memref.load %alloc_2[%47] : memref<1xi64>
          %49 = arith.muli %46, %c32_i64 : i64
          %50 = arith.subi %c3_i64, %49 : i64
          %51 = arith.cmpi sgt, %50, %c32_i64 : i64
          %52 = arith.cmpi sgt, %50, %c0_i64 : i64
          %53 = arith.select %52, %50, %c0_i64 : i64
          %54 = arith.select %51, %c32_i64, %53 : i64
          %55 = arith.addi %54, %c1_i64 : i64
          %56 = arith.divsi %55, %c2_i64 : i64
          %57 = arith.divsi %54, %c2_i64 : i64
          %58 = arith.select %36, %56, %57 : i64
          %59 = arith.cmpi sgt, %48, %c0_i64 : i64
          %60:2 = scf.if %59 -> (i64, i64) {
            %61 = scf.for %arg24 = %c0_i64 to %58 step %c1_i64 iter_args(%arg25 = %arg21) -> (i64)  : i64 {
              %64 = arith.addi %arg25, %c1_i64 overflow<nsw> : i64
              %65 = arith.remui %64, %c2_i64 : i64
              scf.yield %65 : i64
            }
            %62 = arith.addi %arg10, %c1_i64 overflow<nsw> : i64
            %63 = arith.remui %62, %c2_i64 : i64
            scf.yield %63, %61 : i64, i64
          } else {
            scf.yield %arg10, %arg21 : i64, i64
          }
          scf.yield %60#0, %60#1 : i64, i64
        } else {
          scf.yield %arg10, %arg21 : i64, i64
        }
        scf.yield %43#0, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18, %arg19, %arg20, %43#1, %arg22, %arg23 : i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64
      }
      scf.yield %40#0, %40#1, %40#2, %40#3, %40#4, %40#5, %40#6, %40#7, %40#8, %40#9, %40#10, %40#11, %40#12, %40#13 : i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64
    }
    asccube.sync_intra_wait fixpipe, %c2_i64
    asccube.sync_intra_wait fixpipe, %c18_i64
    asccube.sync_intra_wait fixpipe, %c3_i64
    asccube.sync_intra_wait fixpipe, %c19_i64
    asccube.sync_intra_wait fixpipe, %c4_i64
    asccube.sync_intra_wait fixpipe, %c20_i64
    asccube.sync_intra_wait fixpipe, %c5_i64
    asccube.sync_intra_wait fixpipe, %c21_i64
    asccube.sync_intra_wait fixpipe, %c6_i64
    asccube.sync_intra_wait fixpipe, %c22_i64
    return
  }
}
