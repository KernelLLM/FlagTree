module {
  func.func @run_5708e2c192915514c312e2376c4cb822(%arg0: !cannir.tensor<<gm, bf16>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg1: !cannir.tensor<<gm, bf16>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg2: !cannir.tensor<<gm, bf16>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg3: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg4: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg5: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg6: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg7: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg8: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg9: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg10: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg11: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg12: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg13: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg14: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg15: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg16: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg17: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg18: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg19: !cannir.tensor<<gm, f32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg20: !cannir.tensor<<gm, bf16>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg21: !cannir.tensor<<gm, f32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg22: !cannir.datastruct<"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>, %arg23: i32, %arg24: !cannir.tensor<<gm, f32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>) attributes {cannbotdsl.contract_digest = "5708e2c192915514c312e2376c4cb8227050524ec4f75fe4c039b2f67491f4c1", cannbotdsl.public_slot_count = 25 : i64, cannbotdsl.slot_dtype_bits = [16, 16, 16, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 16, 32, 0, 32, 32], cannbotdsl.slot_dtype_codes = [4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 4, 2, 0, 0, 2], cannbotdsl.slot_dtype_lanes = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], cannbotdsl.slot_flags = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], cannbotdsl.slot_kinds = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 2, 1], cannbotdsl.slot_type_digests = ["ff53d9dd034e044fe0291f46b6a8b6d20d70832f969b84adfb49acaa957e065d", "8e0cb6816ea1089ba7f91f4881bee97725b5b6911109a261befefc13c2261d7d", "0e70b00477d640ee610135ae00e19f1a68b52b4e410bfd8b2dbdcb2946bce792", "f6fed84a6d6be7c3f01d01ace0d1ba22c52123312df26f99ba9bdfb687301b96", "3148b880629f3865a4bcecd66f9242371011d4dae86cd2c11d4f6bea24c97534", "f98293015458c2e557927571e78d3f0de9d71e7ccfec112cdd51bec283ca63b7", "522f5b74e301c4e75b9e3159f2802f4b5037f95bf4a73b879e1073fcd80b62dc", "cc4bf3d448da51e633f59b3cba9e51127ab1a0aa0492135e79d12d567aea07f7", "9f587a2bbf1679f7bf78bb34f926eaf75c527cb1d3ac04e1ef77133ab136d94c", "3ec0d6b7dd681df73d3c42f4e705b587500b2865f580d8a683d566f19caa9fd4", "c990af3d155d2a0163c01af90460a177a38c1ea4f762e44c6509eda414ee788e", "d0240e8d660251020ca256b55cd6cb794d1d2147e9688f474548d1d934f98a9e", "fd1a78ffd411f715f127ce67b8f067418d8f04644ec94605f484a37e0b5982f1", "cf31fed0621bd270faeb22e8a9fcec373c62f166bcecd5c7cc814bf42ef4c8b8", "1e0067edbb92fa241d0ab2ac19d9e5780395ec60531fd0d678d2a72f23b8308d", "fd938d07d7b4d335369b0bf338f4890cea59a2d2e2342566f983595ce64dff62", "b871e64a6760b9de68904b0b4d26c1d0efda72963666b209f3106699c1e7d36c", "cc162d5fb514a6f0218f2b54154afceed8ba133fea50af77a20916337e63a2ea", "3350aa221d173da3c382052bcf9faa6a59f088a86b7d4043d33f4b9bcc141874", "47f7338481ace04f1e9c8d41bd5bf48c1dbf687292921fb0cca5b44d34d1fe5a", "d4e6aa0d25329637f0388c182f9a1231642c19329d4c70fce57b529afd06fbb3", "3342f6171a84c25ec72a6acb5e00a6caffe45384df8a2018e56cb9f1e67c50e5", "588f8adbf42ada4b98f3b40707158d15417c1c6bf9729fcd8edebfeaa618fc27", "7dacf501f968d2328c075490e60b2746783f658fa5c50bfc54eaeb02cb98a3a5", "f0190022ca7ae23fdb3d1444fc3cc1608fcd26c24fe0db6326cab4f0bffba32d"], cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.host} {
    %0 = arith.extsi %arg23 : i32 to i64
    cannir.kernel_launch @kernel_SparseN64Kernel___call___0(%0) [%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18, %arg19, %arg20, %arg21, %arg22, %arg24] {cannbotdsl.profile = {indices = array<i64: 0, 1, 2, 3, 4, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 23, 23>, name = "SparseN64Kernel___call__", op_type = "SparseN64Kernel___call__", roles = array<i64: 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1>}} : !cannir.tensor<<gm, bf16>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, !cannir.tensor<<gm, bf16>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, !cannir.tensor<<gm, bf16>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, !cannir.tensor<<gm, f32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, !cannir.tensor<<gm, bf16>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, !cannir.tensor<<gm, f32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, !cannir.datastruct<"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>, !cannir.tensor<<gm, f32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>
    return
  }
  func.func @kernel_SparseN64Kernel___call___0_aiv(%arg0: !cannir.tensor<<gm, bf16>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg1: !cannir.tensor<<gm, bf16>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg2: !cannir.tensor<<gm, bf16>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg3: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg4: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg5: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg6: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg7: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg8: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg9: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg10: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg11: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg12: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg13: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg14: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg15: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg16: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg17: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg18: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg19: !cannir.tensor<<gm, f32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg20: !cannir.tensor<<gm, bf16>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg21: !cannir.tensor<<gm, f32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg22: !cannir.datastruct<"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>, %arg23: !cannir.tensor<<gm, f32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>) attributes {cannir.buffer.auto_sync, cannir.buffer.sync_analyzed, cannir.buffer.sync_verified, cannir.buffer.tensor_state, cannir.core = "aiv", cannir.cube_state_mode = 1 : i32, cannir.cube_state_planned, cannir.device, cannir.explicit_ring_buffer_cursors, cannir.kernel_group = "kernel_SparseN64Kernel___call___0", cannir.resource.ids_prepared} {
    %c900 = arith.constant 900 : index
    %c1024_i32 = arith.constant 1024 : i32
    %c48_i32 = arith.constant 48 : i32
    %c8_i32 = arith.constant 8 : i32
    %c16_i64 = arith.constant 16 : i64
    %c9_i64 = arith.constant 9 : i64
    %c8_i64 = arith.constant 8 : i64
    %c7_i64 = arith.constant 7 : i64
    %c6_i64 = arith.constant 6 : i64
    %c5_i64 = arith.constant 5 : i64
    %c4_i64 = arith.constant 4 : i64
    %c3_i64 = arith.constant 3 : i64
    %c2_i64 = arith.constant 2 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %c10_i64 = arith.constant 10 : i64
    %c11_i64 = arith.constant 11 : i64
    %c12_i64 = arith.constant 12 : i64
    %c13_i64 = arith.constant 13 : i64
    %c17_i64 = arith.constant 17 : i64
    %c18_i64 = arith.constant 18 : i64
    %c19_i64 = arith.constant 19 : i64
    %c21_i64 = arith.constant 21 : i64
    %c22_i64 = arith.constant 22 : i64
    %c23_i64 = arith.constant 23 : i64
    %c32_i64 = arith.constant 32 : i64
    %c33_i64 = arith.constant 33 : i64
    %c330_i64 = arith.constant 330 : i64
    %c329_i64 = arith.constant 329 : i64
    %c328_i64 = arith.constant 328 : i64
    %c327_i64 = arith.constant 327 : i64
    %c326_i64 = arith.constant 326 : i64
    %c325_i64 = arith.constant 325 : i64
    %c16384_i64 = arith.constant 16384 : i64
    %c49_i64 = arith.constant 49 : i64
    %c127_i64 = arith.constant 127 : i64
    %c262144_i64 = arith.constant 262144 : i64
    %c768 = arith.constant 768 : index
    %c8192_i64 = arith.constant 8192 : i64
    %c4 = arith.constant 4 : index
    %c256 = arith.constant 256 : index
    %c2 = arith.constant 2 : index
    %c512 = arith.constant 512 : index
    %c16 = arith.constant 16 : index
    %c324_i64 = arith.constant 324 : i64
    %c48_i64 = arith.constant 48 : i64
    %c512_i32 = arith.constant 512 : i32
    %c16_i32 = arith.constant 16 : i32
    %c-1_i32 = arith.constant -1 : i32
    %c2048_i64 = arith.constant 2048 : i64
    %c-1_i64 = arith.constant -1 : i64
    %c0 = arith.constant 0 : index
    %c32_i32 = arith.constant 32 : i32
    %c256_i64 = arith.constant 256 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %c64_i64 = arith.constant 64 : i64
    %c32768_i64 = arith.constant 32768 : i64
    %c512_i64 = arith.constant 512 : i64
    %c128_i64 = arith.constant 128 : i64
    %c4_i32 = arith.constant 4 : i32
    %c3_i32 = arith.constant 3 : i32
    %c1_i32 = arith.constant 1 : i32
    %false = arith.constant false
    %true = arith.constant true
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %alloc = memref.alloc() {cannir.arg_index = 1 : i64} : memref<1xbf16>
    %alloc_0 = memref.alloc() {cannir.arg_index = 2 : i64} : memref<1xbf16>
    %alloc_1 = memref.alloc() {cannir.arg_index = 3 : i64} : memref<1xi32>
    %alloc_2 = memref.alloc() {cannir.arg_index = 4 : i64} : memref<1xi32>
    %alloc_3 = memref.alloc() {cannir.arg_index = 9 : i64} : memref<1xi32>
    %alloc_4 = memref.alloc() {cannir.arg_index = 10 : i64} : memref<1xi32>
    %alloc_5 = memref.alloc() {cannir.arg_index = 11 : i64} : memref<1xi32>
    %alloc_6 = memref.alloc() {cannir.arg_index = 12 : i64} : memref<1xi32>
    %alloc_7 = memref.alloc() {cannir.arg_index = 13 : i64} : memref<1xi32>
    %alloc_8 = memref.alloc() {cannir.arg_index = 14 : i64} : memref<1xi32>
    %alloc_9 = memref.alloc() {cannir.arg_index = 15 : i64} : memref<1xi32>
    %alloc_10 = memref.alloc() {cannir.arg_index = 16 : i64} : memref<1xi32>
    %alloc_11 = memref.alloc() {cannir.arg_index = 17 : i64} : memref<1xi32>
    %alloc_12 = memref.alloc() {cannir.arg_index = 18 : i64} : memref<1xi32>
    %alloc_13 = memref.alloc() {cannir.arg_index = 20 : i64} : memref<1xbf16>
    %alloc_14 = memref.alloc() {cannir.arg_index = 21 : i64} : memref<1xf32>
    %alloc_15 = memref.alloc() {cannir.arg_index = 23 : i64} : memref<1xf32>
    %alloc_16 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<4096xf32, 1>
    %alloc_17 = memref.alloc() {cannir.byte_offset = 16384 : i64} : memref<4224xbf16, 1>
    %alloc_18 = memref.alloc() {cannir.byte_offset = 24832 : i64} : memref<32xf32, 1>
    %alloc_19 = memref.alloc() {cannir.byte_offset = 24960 : i64} : memref<128xf32, 1>
    %alloc_20 = memref.alloc() {cannir.byte_offset = 25472 : i64} : memref<128xf32, 1>
    %alloc_21 = memref.alloc() {cannir.byte_offset = 25984 : i64} : memref<256xf32, 1>
    %alloc_22 = memref.alloc() {cannir.byte_offset = 27008 : i64} : memref<16384xbf16, 1>
    %alloc_23 = memref.alloc() {cannir.byte_offset = 59776 : i64} : memref<32xf32, 1>
    %alloc_24 = memref.alloc() {cannir.byte_offset = 59904 : i64} : memref<16384xf32, 1>
    %alloc_25 = memref.alloc() {cannir.byte_offset = 125440 : i64} : memref<16384xf32, 1>
    %alloc_26 = memref.alloc() {cannir.byte_offset = 190976 : i64} : memref<8192xbf16, 1>
    %alloc_27 = memref.alloc() {cannir.byte_offset = 207360 : i64} : memref<8192xbf16, 1>
    %alloc_28 = memref.alloc() {cannir.byte_offset = 223744 : i64} : memref<8704xbf16, 1>
    %alloc_29 = memref.alloc() {cannir.byte_offset = 241152 : i64} : memref<8704xbf16, 1>
    %alloc_30 = memref.alloc() {cannir.byte_offset = 258560 : i64} : memref<64xi32, 1>
    %alloc_31 = memref.alloc() {cannir.byte_offset = 258816 : i64} : memref<64xi32, 1>
    %alloc_32 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<9216xf32, 1>
    %alloc_33 = memref.alloc() {cannir.byte_offset = 36864 : i64} : memref<8192xf32, 1>
    %alloc_34 = memref.alloc() {cannir.byte_offset = 69632 : i64} : memref<8192xf32, 1>
    %alloc_35 = memref.alloc() {cannir.byte_offset = 102400 : i64} : memref<8192xf32, 1>
    %alloc_36 = memref.alloc() {cannir.byte_offset = 135168 : i64} : memref<8192xbf16, 1>
    %alloc_37 = memref.alloc() {cannir.byte_offset = 151552 : i64} : memref<64xf32, 1>
    %alloc_38 = memref.alloc() {cannir.byte_offset = 151808 : i64} : memref<64xf32, 1>
    %alloc_39 = memref.alloc() {cannir.byte_offset = 152064 : i64} : memref<4608xf32, 1>
    %alloc_40 = memref.alloc() {cannir.byte_offset = 170496 : i64} : memref<64xf32, 1>
    %alloc_41 = memref.alloc() {cannir.byte_offset = 327680 : i64} : memref<32768xbf16, 2>
    %alloc_42 = memref.alloc() {cannir.byte_offset = 458752 : i64} : memref<32768xbf16, 2>
    %alloc_43 = memref.alloc() {cannir.byte_offset = 393216 : i64} : memref<32768xbf16, 2>
    %alloc_44 = memref.alloc() {cannir.byte_offset = 114688 : i64} : memref<32768xbf16, 2>
    %alloc_45 = memref.alloc() {cannir.byte_offset = 245760 : i64} : memref<32768xbf16, 2>
    %alloc_46 = memref.alloc() {cannir.byte_offset = 180224 : i64} : memref<32768xbf16, 2>
    %alloc_47 = memref.alloc() {cannir.byte_offset = 311296 : i64} : memref<8192xbf16, 2>
    ascvec.sync_intra_arrive v, %c2_i64
    ascvec.sync_intra_arrive v, %c1_i64
    ascvec.sync_intra_arrive v, %c0_i64
    %0 = ascscalar.get_block_idx
    %1 = arith.muli %0, %c9_i64 : i64
    %2 = arith.addi %1, %c3_i64 : i64
    %3 = arith.index_cast %2 : i64 to index
    %4 = memref.load %alloc_8[%3] : memref<1xi32>
    %5 = arith.extsi %4 : i32 to i64
    %6 = arith.addi %1, %c6_i64 : i64
    %7 = arith.index_cast %6 : i64 to index
    %8 = memref.load %alloc_8[%7] : memref<1xi32>
    %9 = arith.extsi %8 : i32 to i64
    %10 = arith.addi %1, %c7_i64 : i64
    %11 = arith.index_cast %10 : i64 to index
    %12 = memref.load %alloc_8[%11] : memref<1xi32>
    %13 = arith.extsi %12 : i32 to i64
    %14 = arith.addi %1, %c1_i64 : i64
    %15 = arith.index_cast %14 : i64 to index
    %16 = memref.load %alloc_8[%15] : memref<1xi32>
    %17 = arith.extsi %16 : i32 to i64
    %18 = arith.addi %1, %c4_i64 : i64
    %19 = arith.index_cast %18 : i64 to index
    %20 = memref.load %alloc_8[%19] : memref<1xi32>
    %21 = arith.extsi %20 : i32 to i64
    %22 = cannir.datastruct_get %arg22["query_rows_per_batch"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
    %23 = arith.extsi %22 : i32 to i64
    %24 = arith.muli %17, %23 : i64
    %25 = arith.addi %1, %c2_i64 : i64
    %26 = arith.index_cast %25 : i64 to index
    %27 = memref.load %alloc_8[%26] : memref<1xi32>
    %28 = arith.extsi %27 : i32 to i64
    %29 = arith.addi %24, %28 : i64
    %30 = arith.muli %21, %23 : i64
    %31 = arith.addi %1, %c5_i64 : i64
    %32 = arith.index_cast %31 : i64 to index
    %33 = memref.load %alloc_8[%32] : memref<1xi32>
    %34 = arith.extsi %33 : i32 to i64
    %35 = arith.addi %30, %34 : i64
    %36 = cannir.alloc_array {init_values = dense<0> : tensor<1xi64>} : <1xi64>
    %37 = cannir.alloc_array : <5xi64>
    %38 = cannir.alloc_array : <5xi64>
    %39 = cannir.alloc_array : <5xi64>
    %40 = cannir.alloc_array : <5xi64>
    %41 = cannir.alloc_array : <5xi64>
    %42 = cannir.alloc_array : <5xi64>
    %43 = cannir.alloc_array : <5xi64>
    %44 = cannir.alloc_array : <5xi64>
    %45 = cannir.alloc_array : <5xi64>
    %46 = cannir.alloc_array : <5xi64>
    %47 = cannir.alloc_array : <5xi64>
    %48 = cannir.alloc_array : <5xi64>
    %49 = cannir.alloc_array : <5xi64>
    %50 = cannir.alloc_array : <5xi64>
    %51 = ascscalar.get_subblock_id
    %52 = arith.muli %51, %c32_i64 : i64
    %53 = cannir.datastruct_get %arg22["num_heads_q"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
    %54 = arith.extsi %53 : i32 to i64
    %55 = arith.cmpi sgt, %54, %52 : i64
    %56 = arith.subi %54, %52 : i64
    %57 = arith.cmpi slt, %56, %c32_i64 : i64
    %58 = arith.select %57, %56, %c32_i64 : i64
    %59 = arith.select %55, %58, %c0_i64 : i64
    ascvec.vf(%alloc_18 : memref<32xf32, 1>) {
    ^bb0(%arg24: memref<32xf32, 1>):
      %c0_48 = arith.constant 0 : index
      %c-8388608_i32 = arith.constant -8388608 : i32
      %c32_i32_49 = arith.constant 32 : i32
      %c0_50 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg25 = %c0_50 to %c1 step %c1 {
        %mask, %remaining_out = ascvec.update_mask %c32_i32_49 : !ascvec.mask
        %96 = ascvec.duplicate_scalar %c-8388608_i32, %mask : i32 -> <ui32>
        %97 = ascvec.reinterpret %96 : <ui32> -> <f32>
        ascvec.storealign %arg24[%c0_48], %97, %mask : memref<32xf32, 1>, index, <f32>
      }
    } {cannir.vf_mode = "simd"}
    ascvec.get_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
    ascvec.get_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
    ascvec.vf(%alloc_22, %alloc_23 : memref<16384xbf16, 1>, memref<32xf32, 1>) {
    ^bb0(%arg24: memref<16384xbf16, 1>, %arg25: memref<32xf32, 1>):
      %cst = arith.constant 0.000000e+00 : f32
      %c32_i32_48 = arith.constant 32 : i32
      %c1_i32_49 = arith.constant 1 : i32
      %c0_i32_50 = arith.constant 0 : i32
      %c0_51 = arith.constant 0 : index
      %c128_i32 = arith.constant 128 : i32
      %cst_52 = arith.constant 0.000000e+00 : bf16
      %96 = ascvec.create_mask elem_bits 16 : !ascvec.mask
      %97 = ascvec.duplicate_scalar %cst_52, %96 : bf16 -> <bf16>
      scf.for %arg26 = %c0_i32_50 to %c128_i32 step %c1_i32_49  : i32 {
        %98 = arith.muli %arg26, %c128_i32 overflow<nsw> : i32
        %99 = arith.index_cast %98 : i32 to index
        ascvec.storealign %arg24[%99], %97, %96 : memref<16384xbf16, 1>, index, <bf16>
      } {cannir.loop_unroll = 1 : i64}
      %c0_53 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg26 = %c0_53 to %c1 step %c1 {
        %mask, %remaining_out = ascvec.update_mask %c32_i32_48 : !ascvec.mask
        %98 = ascvec.duplicate_scalar %cst, %mask : f32 -> <f32>
        ascvec.storealign %arg25[%c0_51], %98, %mask : memref<32xf32, 1>, index, <f32>
      }
    } {cannir.vf_mode = "simd"}
    ascvec.release_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
    ascvec.release_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
    ascvec.get_buf v, %c7_i64, 0 {cannir.buffer.generated_sync}
    ascvec.vf(%alloc_26 : memref<8192xbf16, 1>) {
    ^bb0(%arg24: memref<8192xbf16, 1>):
      %c1_i32_48 = arith.constant 1 : i32
      %c64_i32 = arith.constant 64 : i32
      %c0_i32_49 = arith.constant 0 : i32
      %c128_i32 = arith.constant 128 : i32
      %cst = arith.constant 0.000000e+00 : bf16
      %96 = ascvec.create_mask elem_bits 16 : !ascvec.mask
      %97 = ascvec.duplicate_scalar %cst, %96 : bf16 -> <bf16>
      scf.for %arg25 = %c0_i32_49 to %c64_i32 step %c1_i32_48  : i32 {
        %98 = arith.muli %arg25, %c128_i32 overflow<nsw> : i32
        %99 = arith.index_cast %98 : i32 to index
        ascvec.storealign %arg24[%99], %97, %96 : memref<8192xbf16, 1>, index, <bf16>
      }
    } {cannir.vf_mode = "simd"}
    ascvec.release_buf v, %c7_i64, 0 {cannir.buffer.generated_sync}
    ascvec.get_buf v, %c8_i64, 0 {cannir.buffer.generated_sync}
    ascvec.vf(%alloc_27 : memref<8192xbf16, 1>) {
    ^bb0(%arg24: memref<8192xbf16, 1>):
      %c1_i32_48 = arith.constant 1 : i32
      %c64_i32 = arith.constant 64 : i32
      %c0_i32_49 = arith.constant 0 : i32
      %c128_i32 = arith.constant 128 : i32
      %cst = arith.constant 0.000000e+00 : bf16
      %96 = ascvec.create_mask elem_bits 16 : !ascvec.mask
      %97 = ascvec.duplicate_scalar %cst, %96 : bf16 -> <bf16>
      scf.for %arg25 = %c0_i32_49 to %c64_i32 step %c1_i32_48  : i32 {
        %98 = arith.muli %arg25, %c128_i32 overflow<nsw> : i32
        %99 = arith.index_cast %98 : i32 to index
        ascvec.storealign %arg24[%99], %97, %96 : memref<8192xbf16, 1>, index, <bf16>
      }
    } {cannir.vf_mode = "simd"}
    ascvec.release_buf v, %c8_i64, 0 {cannir.buffer.generated_sync}
    %60 = cannir.array_load %36[%c0] : <1xi64>
    %61 = arith.remui %60, %c5_i64 : i64
    %62 = arith.index_castui %61 : i64 to index
    cannir.array_store %c0_i64, %37[%62] : <5xi64>
    cannir.array_store %c0_i64, %38[%62] : <5xi64>
    cannir.array_store %c0_i64, %39[%62] : <5xi64>
    cannir.array_store %c0_i64, %40[%62] : <5xi64>
    cannir.array_store %c0_i64, %41[%62] : <5xi64>
    cannir.array_store %c0_i64, %42[%62] : <5xi64>
    cannir.array_store %c0_i64, %43[%62] : <5xi64>
    cannir.array_store %c0_i64, %44[%62] : <5xi64>
    cannir.array_store %c0_i64, %45[%62] : <5xi64>
    cannir.array_store %c0_i64, %46[%62] : <5xi64>
    cannir.array_store %c0_i64, %47[%62] : <5xi64>
    cannir.array_store %c0_i64, %48[%62] : <5xi64>
    cannir.array_store %c0_i64, %49[%62] : <5xi64>
    cannir.array_store %c0_i64, %50[%62] : <5xi64>
    %63 = cannir.array_load %36[%c0] : <1xi64>
    %64 = arith.addi %63, %c1_i64 : i64
    cannir.array_store %64, %36[%c0] : <1xi64>
    %65 = cannir.array_load %36[%c0] : <1xi64>
    %66 = arith.remui %65, %c5_i64 : i64
    %67 = arith.index_castui %66 : i64 to index
    cannir.array_store %c0_i64, %37[%67] : <5xi64>
    cannir.array_store %c0_i64, %38[%67] : <5xi64>
    cannir.array_store %c0_i64, %39[%67] : <5xi64>
    cannir.array_store %c0_i64, %40[%67] : <5xi64>
    cannir.array_store %c0_i64, %41[%67] : <5xi64>
    cannir.array_store %c0_i64, %42[%67] : <5xi64>
    cannir.array_store %c0_i64, %43[%67] : <5xi64>
    cannir.array_store %c0_i64, %44[%67] : <5xi64>
    cannir.array_store %c0_i64, %45[%67] : <5xi64>
    cannir.array_store %c0_i64, %46[%67] : <5xi64>
    cannir.array_store %c0_i64, %47[%67] : <5xi64>
    cannir.array_store %c0_i64, %48[%67] : <5xi64>
    cannir.array_store %c0_i64, %49[%67] : <5xi64>
    cannir.array_store %c0_i64, %50[%67] : <5xi64>
    %68 = cannir.array_load %36[%c0] : <1xi64>
    %69 = arith.addi %68, %c1_i64 : i64
    cannir.array_store %69, %36[%c0] : <1xi64>
    %70 = cannir.array_load %36[%c0] : <1xi64>
    %71 = arith.remui %70, %c5_i64 : i64
    %72 = arith.index_castui %71 : i64 to index
    cannir.array_store %c0_i64, %37[%72] : <5xi64>
    cannir.array_store %c0_i64, %38[%72] : <5xi64>
    cannir.array_store %c0_i64, %39[%72] : <5xi64>
    cannir.array_store %c0_i64, %40[%72] : <5xi64>
    cannir.array_store %c0_i64, %41[%72] : <5xi64>
    cannir.array_store %c0_i64, %42[%72] : <5xi64>
    cannir.array_store %c0_i64, %43[%72] : <5xi64>
    cannir.array_store %c0_i64, %44[%72] : <5xi64>
    cannir.array_store %c0_i64, %45[%72] : <5xi64>
    cannir.array_store %c0_i64, %46[%72] : <5xi64>
    cannir.array_store %c0_i64, %47[%72] : <5xi64>
    cannir.array_store %c0_i64, %48[%72] : <5xi64>
    cannir.array_store %c0_i64, %49[%72] : <5xi64>
    cannir.array_store %c0_i64, %50[%72] : <5xi64>
    %73 = cannir.array_load %36[%c0] : <1xi64>
    %74 = arith.addi %73, %c1_i64 : i64
    cannir.array_store %74, %36[%c0] : <1xi64>
    %75 = cannir.array_load %36[%c0] : <1xi64>
    %76 = arith.remui %75, %c5_i64 : i64
    %77 = arith.index_castui %76 : i64 to index
    cannir.array_store %c0_i64, %37[%77] : <5xi64>
    cannir.array_store %c0_i64, %38[%77] : <5xi64>
    cannir.array_store %c0_i64, %39[%77] : <5xi64>
    cannir.array_store %c0_i64, %40[%77] : <5xi64>
    cannir.array_store %c0_i64, %41[%77] : <5xi64>
    cannir.array_store %c0_i64, %42[%77] : <5xi64>
    cannir.array_store %c0_i64, %43[%77] : <5xi64>
    cannir.array_store %c0_i64, %44[%77] : <5xi64>
    cannir.array_store %c0_i64, %45[%77] : <5xi64>
    cannir.array_store %c0_i64, %46[%77] : <5xi64>
    cannir.array_store %c0_i64, %47[%77] : <5xi64>
    cannir.array_store %c0_i64, %48[%77] : <5xi64>
    cannir.array_store %c0_i64, %49[%77] : <5xi64>
    cannir.array_store %c0_i64, %50[%77] : <5xi64>
    %78 = cannir.array_load %36[%c0] : <1xi64>
    %79 = arith.addi %78, %c1_i64 : i64
    cannir.array_store %79, %36[%c0] : <1xi64>
    %80 = cannir.array_load %36[%c0] : <1xi64>
    %81 = arith.remui %80, %c5_i64 : i64
    %82 = arith.index_castui %81 : i64 to index
    cannir.array_store %c0_i64, %37[%82] : <5xi64>
    cannir.array_store %c0_i64, %38[%82] : <5xi64>
    cannir.array_store %c0_i64, %39[%82] : <5xi64>
    cannir.array_store %c0_i64, %40[%82] : <5xi64>
    cannir.array_store %c0_i64, %41[%82] : <5xi64>
    cannir.array_store %c0_i64, %42[%82] : <5xi64>
    cannir.array_store %c0_i64, %43[%82] : <5xi64>
    cannir.array_store %c0_i64, %44[%82] : <5xi64>
    cannir.array_store %c0_i64, %45[%82] : <5xi64>
    cannir.array_store %c0_i64, %46[%82] : <5xi64>
    cannir.array_store %c0_i64, %47[%82] : <5xi64>
    cannir.array_store %c0_i64, %48[%82] : <5xi64>
    cannir.array_store %c0_i64, %49[%82] : <5xi64>
    cannir.array_store %c0_i64, %50[%82] : <5xi64>
    %83 = cannir.array_load %36[%c0] : <1xi64>
    %84 = arith.addi %83, %c1_i64 : i64
    cannir.array_store %84, %36[%c0] : <1xi64>
    %85 = arith.index_cast %1 : i64 to index
    %86 = memref.load %alloc_8[%85] : memref<1xi32>
    %87 = arith.cmpi ne, %86, %c0_i32 : i32
    %88 = scf.if %87 -> (i64) {
      %96 = memref.load %alloc_8[%15] : memref<1xi32>
      %97 = arith.extsi %96 : i32 to i64
      %98 = memref.load %alloc_8[%26] : memref<1xi32>
      %99 = arith.extsi %98 : i32 to i64
      %100 = memref.load %alloc_8[%19] : memref<1xi32>
      %101 = arith.extsi %100 : i32 to i64
      %102 = memref.load %alloc_8[%32] : memref<1xi32>
      %103 = arith.extsi %102 : i32 to i64
      %104 = arith.cmpi sgt, %9, %c0_i64 : i64
      %105 = arith.extui %104 : i1 to i64
      %106 = arith.addi %103, %105 : i64
      %107 = cannir.datastruct_get %arg22["batch_count"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
      %108 = arith.extsi %107 : i32 to i64
      %109 = arith.cmpi slt, %101, %108 : i64
      %110 = arith.addi %101, %c1_i64 : i64
      %111 = arith.select %109, %110, %108 : i64
      %112 = scf.for %arg24 = %97 to %111 step %c1_i64 iter_args(%arg25 = %c0_i64) -> (i64)  : i64 {
        %113 = cannir.datastruct_get %arg22["query_rows_per_batch"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
        %114 = arith.extsi %113 : i32 to i64
        %115 = cannir.datastruct_get %arg22["batch_mode"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
        %116 = arith.cmpi eq, %115, %c2_i32 : i32
        %117 = scf.if %116 -> (i64) {
          %126 = arith.addi %arg24, %c1_i64 : i64
          %127 = arith.index_cast %126 : i64 to index
          %128 = memref.load %alloc_10[%127] : memref<1xi32>
          %129 = arith.index_cast %arg24 : i64 to index
          %130 = memref.load %alloc_10[%129] : memref<1xi32>
          %131 = arith.subi %128, %130 : i32
          %132 = arith.extsi %131 : i32 to i64
          scf.yield %132 : i64
        } else {
          scf.yield %114 : i64
        }
        %118 = arith.cmpi eq, %arg24, %97 : i64
        %119 = arith.select %118, %99, %c0_i64 : i64
        %120 = arith.cmpi eq, %arg24, %101 : i64
        %121 = arith.select %120, %106, %117 : i64
        %122 = arith.cmpi sgt, %121, %119 : i64
        %123 = arith.subi %121, %119 : i64
        %124 = arith.select %122, %123, %c0_i64 : i64
        %125 = arith.addi %arg25, %124 : i64
        scf.yield %125 : i64
      }
      scf.yield %112 : i64
    } else {
      scf.yield %c0_i64 : i64
    }
    %89 = arith.addi %88, %c1_i64 : i64
    %90 = ascscalar.memref_address %alloc_44 : memref<32768xbf16, 2>
    %91 = ascscalar.memref_address %alloc_41 : memref<32768xbf16, 2>
    %92 = ascscalar.memref_address %alloc_30 : memref<64xi32, 1>
    %93:41 = scf.for %arg24 = %c0_i64 to %89 step %c1_i64 iter_args(%arg25 = %c0_i64, %arg26 = %90, %arg27 = %c0_i64, %arg28 = %c32768_i64, %arg29 = %c1_i64, %arg30 = %c32768_i64, %arg31 = %c1_i64, %arg32 = %90, %arg33 = %c0_i64, %arg34 = %90, %arg35 = %c0_i64, %arg36 = %91, %arg37 = %c0_i64, %arg38 = %c32768_i64, %arg39 = %c1_i64, %arg40 = %c32768_i64, %arg41 = %c1_i64, %arg42 = %91, %arg43 = %c0_i64, %arg44 = %91, %arg45 = %c0_i64, %arg46 = %92, %arg47 = %c0_i64, %arg48 = %c64_i64, %arg49 = %c1_i64, %arg50 = %c64_i64, %arg51 = %c1_i64, %arg52 = %92, %arg53 = %c0_i64, %arg54 = %92, %arg55 = %c0_i64, %arg56 = %c0_i64, %arg57 = %c0_i64, %arg58 = %c0_i64, %arg59 = %c0_i64, %arg60 = %c0_i64, %arg61 = %c1_i64, %arg62 = %c1_i64, %arg63 = %c17_i64, %arg64 = %c21_i64, %arg65 = %c12_i64) -> (i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64)  : i64 {
      %96 = arith.cmpi sge, %arg59, %88 : i64
      %97 = arith.cmpi sgt, %arg58, %c0_i64 : i64
      %98 = arith.select %97, %c5_i64, %c0_i64 : i64
      %99:13 = scf.if %96 -> (i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) {
        scf.yield %arg25, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %98 : i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64
      } else {
        %108 = memref.load %alloc_8[%85] : memref<1xi32>
        %109 = arith.cmpi ne, %108, %c0_i32 : i32
        %110:5 = scf.if %109 -> (i64, i64, i64, i64, i64) {
          %130 = memref.load %alloc_8[%15] : memref<1xi32>
          %131 = arith.extsi %130 : i32 to i64
          %132 = memref.load %alloc_8[%26] : memref<1xi32>
          %133 = arith.extsi %132 : i32 to i64
          %134 = memref.load %alloc_8[%19] : memref<1xi32>
          %135 = arith.extsi %134 : i32 to i64
          %136 = memref.load %alloc_8[%32] : memref<1xi32>
          %137 = arith.extsi %136 : i32 to i64
          %138 = arith.cmpi sgt, %9, %c0_i64 : i64
          %139 = arith.extui %138 : i1 to i64
          %140 = arith.addi %137, %139 : i64
          %141 = cannir.datastruct_get %arg22["batch_mode"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
          %142 = arith.cmpi eq, %141, %c2_i32 : i32
          %143:5 = scf.if %142 -> (i64, i64, i64, i64, i64) {
            %144 = arith.index_cast %130 : i32 to index
            %145 = memref.load %alloc_10[%144] : memref<1xi32>
            %146 = arith.extsi %145 : i32 to i64
            %147 = arith.addi %146, %133 : i64
            %148 = cannir.datastruct_get %arg22["batch_count"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
            %149 = arith.extsi %148 : i32 to i64
            %150 = arith.index_cast %148 : i32 to index
            %151 = memref.load %alloc_10[%150] : memref<1xi32>
            %152 = arith.extsi %151 : i32 to i64
            %153 = arith.cmpi slt, %135, %149 : i64
            %154 = scf.if %153 -> (i64) {
              %158 = arith.index_cast %134 : i32 to index
              %159 = memref.load %alloc_10[%158] : memref<1xi32>
              %160 = arith.extsi %159 : i32 to i64
              %161 = arith.addi %135, %c1_i64 : i64
              %162 = arith.index_cast %161 : i64 to index
              %163 = memref.load %alloc_10[%162] : memref<1xi32>
              %164 = arith.extsi %163 : i32 to i64
              %165 = arith.subi %164, %160 : i64
              %166 = arith.cmpi slt, %140, %165 : i64
              %167 = arith.select %166, %140, %165 : i64
              %168 = arith.addi %160, %167 : i64
              scf.yield %168 : i64
            } else {
              scf.yield %152 : i64
            }
            %155 = arith.addi %147, %arg59 : i64
            %156 = arith.cmpi sge, %arg59, %c0_i64 : i64
            %157:5 = scf.if %156 -> (i64, i64, i64, i64, i64) {
              %158 = arith.cmpi slt, %155, %154 : i64
              %159 = arith.select %158, %155, %c0_i64 : i64
              %160 = arith.extui %158 : i1 to i64
              %161:3 = scf.if %158 -> (i64, i64, i64) {
                %162 = cannir.datastruct_get %arg22["batch_count"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
                %163 = arith.extsi %162 : i32 to i64
                %164 = arith.cmpi slt, %135, %163 : i64
                %165 = arith.subi %163, %c1_i64 : i64
                %166 = arith.select %164, %135, %165 : i64
                %167 = memref.load %alloc_10[%144] : memref<1xi32>
                %168 = arith.extsi %167 : i32 to i64
                %169 = arith.addi %131, %c1_i64 : i64
                %170 = arith.index_cast %169 : i64 to index
                %171 = memref.load %alloc_10[%170] : memref<1xi32>
                %172 = arith.extsi %171 : i32 to i64
                %173 = arith.cmpi slt, %131, %166 : i64
                %174:3 = scf.if %173 -> (i64, i64, i64) {
                  %180 = memref.load %alloc_10[%c0] : memref<1xi32>
                  %181 = arith.extsi %180 : i32 to i64
                  %182 = cannir.datastruct_get %arg22["batch_count"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
                  %183 = arith.extsi %182 : i32 to i64
                  %184 = arith.index_cast %182 : i32 to index
                  %185 = memref.load %alloc_10[%184] : memref<1xi32>
                  %186 = arith.extsi %185 : i32 to i64
                  %187 = arith.subi %186, %181 : i64
                  %188 = arith.remsi %187, %183 : i64
                  %189 = arith.cmpi ne, %188, %c0_i64 : i64
                  %190 = arith.cmpi slt, %188, %c0_i64 : i64
                  %191 = arith.cmpi slt, %183, %c0_i64 : i64
                  %192 = arith.xori %190, %191 : i1
                  %193 = arith.andi %189, %192 : i1
                  %194 = arith.divsi %187, %183 : i64
                  %195 = arith.extui %193 : i1 to i64
                  %196 = arith.subi %194, %195 : i64
                  %197 = arith.cmpi sgt, %196, %c0_i64 : i64
                  %198 = arith.select %197, %196, %c1_i64 : i64
                  %199 = arith.subi %155, %181 : i64
                  %200 = arith.remsi %199, %198 : i64
                  %201 = arith.cmpi ne, %200, %c0_i64 : i64
                  %202 = arith.cmpi slt, %200, %c0_i64 : i64
                  %203 = arith.andi %201, %202 : i1
                  %204 = arith.divsi %199, %198 : i64
                  %205 = arith.extui %203 : i1 to i64
                  %206 = arith.subi %204, %205 : i64
                  %207 = arith.cmpi slt, %206, %131 : i64
                  %208 = arith.select %207, %131, %206 : i64
                  %209 = arith.cmpi sgt, %208, %166 : i64
                  %210 = arith.select %209, %166, %208 : i64
                  %211 = arith.index_cast %210 : i64 to index
                  %212 = memref.load %alloc_10[%211] : memref<1xi32>
                  %213 = arith.extsi %212 : i32 to i64
                  %214 = arith.addi %210, %c1_i64 : i64
                  %215 = arith.index_cast %214 : i64 to index
                  %216 = memref.load %alloc_10[%215] : memref<1xi32>
                  %217 = arith.extsi %216 : i32 to i64
                  %218 = arith.cmpi slt, %155, %213 : i64
                  %219 = arith.cmpi sge, %155, %217 : i64
                  %220 = arith.select %218, %true, %219 : i1
                  %221:3 = scf.if %220 -> (i64, i64, i64) {
                    %222:2 = scf.while (%arg66 = %131, %arg67 = %166) : (i64, i64) -> (i64, i64) {
                      %230 = arith.cmpi slt, %arg66, %arg67 : i64
                      scf.condition(%230) %arg66, %arg67 : i64, i64
                    } do {
                    ^bb0(%arg66: i64, %arg67: i64):
                      %230 = arith.addi %arg66, %arg67 : i64
                      %231 = arith.remsi %230, %c2_i64 : i64
                      %232 = arith.cmpi ne, %231, %c0_i64 : i64
                      %233 = arith.cmpi slt, %231, %c0_i64 : i64
                      %234 = arith.andi %232, %233 : i1
                      %235 = arith.divsi %230, %c2_i64 : i64
                      %236 = arith.extui %234 : i1 to i64
                      %237 = arith.subi %235, %236 : i64
                      %238 = arith.addi %237, %c1_i64 : i64
                      %239 = arith.index_cast %238 : i64 to index
                      %240 = memref.load %alloc_10[%239] : memref<1xi32>
                      %241 = arith.extsi %240 : i32 to i64
                      %242 = arith.cmpi sle, %241, %155 : i64
                      %243 = arith.select %242, %238, %arg66 : i64
                      %244 = arith.select %242, %arg67, %237 : i64
                      scf.yield %243, %244 : i64, i64
                    }
                    %223 = arith.index_cast %222#0 : i64 to index
                    %224 = memref.load %alloc_10[%223] : memref<1xi32>
                    %225 = arith.extsi %224 : i32 to i64
                    %226 = arith.addi %222#0, %c1_i64 : i64
                    %227 = arith.index_cast %226 : i64 to index
                    %228 = memref.load %alloc_10[%227] : memref<1xi32>
                    %229 = arith.extsi %228 : i32 to i64
                    scf.yield %222#0, %225, %229 : i64, i64, i64
                  } else {
                    scf.yield %210, %213, %217 : i64, i64, i64
                  }
                  scf.yield %221#0, %221#1, %221#2 : i64, i64, i64
                } else {
                  scf.yield %131, %168, %172 : i64, i64, i64
                }
                %175 = arith.subi %155, %174#1 : i64
                %176 = arith.subi %174#2, %174#1 : i64
                %177 = cannir.datastruct_get %arg22["has_seqused_q"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
                %178 = arith.cmpi ne, %177, %c0_i32 : i32
                %179 = scf.if %178 -> (i64) {
                  %180 = arith.index_cast %174#0 : i64 to index
                  %181 = memref.load %alloc_9[%180] : memref<1xi32>
                  %182 = arith.extsi %181 : i32 to i64
                  scf.yield %182 : i64
                } else {
                  scf.yield %176 : i64
                }
                scf.yield %174#0, %175, %179 : i64, i64, i64
              } else {
                scf.yield %c0_i64, %c0_i64, %c0_i64 : i64, i64, i64
              }
              scf.yield %159, %161#0, %161#1, %161#2, %160 : i64, i64, i64, i64, i64
            } else {
              scf.yield %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64 : i64, i64, i64, i64, i64
            }
            scf.yield %157#0, %157#1, %157#2, %157#3, %157#4 : i64, i64, i64, i64, i64
          } else {
            %144 = cannir.datastruct_get %arg22["query_rows_per_batch"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
            %145 = arith.extsi %144 : i32 to i64
            %146 = arith.muli %131, %145 : i64
            %147 = arith.addi %146, %133 : i64
            %148 = arith.muli %135, %145 : i64
            %149 = arith.cmpi slt, %140, %145 : i64
            %150 = arith.select %149, %140, %145 : i64
            %151 = arith.addi %148, %150 : i64
            %152 = arith.addi %147, %arg59 : i64
            %153 = arith.cmpi sge, %arg59, %c0_i64 : i64
            %154:5 = scf.if %153 -> (i64, i64, i64, i64, i64) {
              %155 = arith.cmpi slt, %152, %151 : i64
              %156 = arith.select %155, %152, %c0_i64 : i64
              %157 = arith.extui %155 : i1 to i64
              %158:3 = scf.if %155 -> (i64, i64, i64) {
                %159 = arith.remsi %152, %145 : i64
                %160 = arith.cmpi ne, %159, %c0_i64 : i64
                %161 = arith.cmpi slt, %159, %c0_i64 : i64
                %162 = arith.cmpi slt, %145, %c0_i64 : i64
                %163 = arith.xori %161, %162 : i1
                %164 = arith.andi %160, %163 : i1
                %165 = arith.divsi %152, %145 : i64
                %166 = arith.extui %164 : i1 to i64
                %167 = arith.subi %165, %166 : i64
                %168 = arith.muli %167, %145 : i64
                %169 = arith.subi %152, %168 : i64
                %170 = cannir.datastruct_get %arg22["has_seqused_q"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
                %171 = arith.cmpi ne, %170, %c0_i32 : i32
                %172 = scf.if %171 -> (i64) {
                  %173 = arith.index_cast %167 : i64 to index
                  %174 = memref.load %alloc_9[%173] : memref<1xi32>
                  %175 = arith.extsi %174 : i32 to i64
                  scf.yield %175 : i64
                } else {
                  scf.yield %145 : i64
                }
                scf.yield %167, %169, %172 : i64, i64, i64
              } else {
                scf.yield %c0_i64, %c0_i64, %c0_i64 : i64, i64, i64
              }
              scf.yield %156, %158#0, %158#1, %158#2, %157 : i64, i64, i64, i64, i64
            } else {
              scf.yield %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64 : i64, i64, i64, i64, i64
            }
            scf.yield %154#0, %154#1, %154#2, %154#3, %154#4 : i64, i64, i64, i64, i64
          }
          scf.yield %143#0, %143#1, %143#2, %143#3, %143#4 : i64, i64, i64, i64, i64
        } else {
          scf.yield %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64 : i64, i64, i64, i64, i64
        }
        %111 = arith.cmpi ne, %110#4, %c0_i64 : i64
        %112 = arith.cmpi slt, %110#2, %110#3 : i64
        %113 = arith.select %111, %112, %false : i1
        %114:9 = scf.if %113 -> (i64, i64, i64, i64, i64, i64, i64, i64, i64) {
          %130 = cannir.datastruct_get %arg22["ori_kv_capacity"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
          %131 = arith.extsi %130 : i32 to i64
          %132 = arith.muli %110#1, %131 : i64
          %133 = cannir.datastruct_get %arg22["cmp_kv_capacity"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
          %134 = arith.extsi %133 : i32 to i64
          %135 = arith.muli %110#1, %134 : i64
          %136 = cannir.datastruct_get %arg22["kv_mode"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
          %137 = arith.cmpi eq, %136, %c1_i32 : i32
          %138:4 = scf.if %137 -> (i64, i64, i64, i64) {
            scf.yield %c0_i64, %c0_i64, %131, %134 : i64, i64, i64, i64
          } else {
            %185 = cannir.datastruct_get %arg22["batch_mode"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
            %186 = arith.cmpi eq, %185, %c2_i32 : i32
            %187:4 = scf.if %186 -> (i64, i64, i64, i64) {
              %188 = arith.index_cast %110#1 : i64 to index
              %189 = memref.load %alloc_11[%188] : memref<1xi32>
              %190 = arith.extsi %189 : i32 to i64
              %191 = arith.addi %110#1, %c1_i64 : i64
              %192 = arith.index_cast %191 : i64 to index
              %193 = memref.load %alloc_11[%192] : memref<1xi32>
              %194 = arith.extsi %193 : i32 to i64
              %195 = arith.subi %194, %190 : i64
              %196 = cannir.datastruct_get %arg22["has_cmp"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
              %197 = arith.cmpi ne, %196, %c0_i32 : i32
              %198:2 = scf.if %197 -> (i64, i64) {
                %199 = memref.load %alloc_12[%188] : memref<1xi32>
                %200 = arith.extsi %199 : i32 to i64
                %201 = memref.load %alloc_12[%192] : memref<1xi32>
                %202 = arith.extsi %201 : i32 to i64
                %203 = arith.subi %202, %200 : i64
                scf.yield %200, %203 : i64, i64
              } else {
                scf.yield %135, %134 : i64, i64
              }
              scf.yield %190, %195, %198#0, %198#1 : i64, i64, i64, i64
            } else {
              scf.yield %132, %131, %135, %134 : i64, i64, i64, i64
            }
            scf.yield %187#0, %187#2, %187#1, %187#3 : i64, i64, i64, i64
          }
          %139 = cannir.datastruct_get %arg22["has_cmp"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
          %140 = arith.cmpi eq, %139, %c0_i32 : i32
          %141 = arith.select %140, %c0_i64, %138#1 : i64
          %142 = arith.select %140, %c0_i64, %138#3 : i64
          %143 = cannir.datastruct_get %arg22["has_seqused_ori_kv"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
          %144 = arith.cmpi ne, %143, %c0_i32 : i32
          %145 = scf.if %144 -> (i64) {
            %185 = arith.index_cast %110#1 : i64 to index
            %186 = memref.load %alloc_5[%185] : memref<1xi32>
            %187 = arith.extsi %186 : i32 to i64
            %188 = arith.cmpi sgt, %187, %c0_i64 : i64
            %189 = arith.cmpi slt, %187, %138#2 : i64
            %190 = arith.select %189, %187, %138#2 : i64
            %191 = arith.select %188, %190, %c0_i64 : i64
            scf.yield %191 : i64
          } else {
            scf.yield %138#2 : i64
          }
          %146 = arith.cmpi ne, %139, %c0_i32 : i32
          %147 = scf.if %146 -> (i1) {
            %185 = cannir.datastruct_get %arg22["has_seqused_cmp_kv"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
            %186 = arith.cmpi ne, %185, %c0_i32 : i32
            scf.yield %186 : i1
          } else {
            scf.yield %false : i1
          }
          %148 = scf.if %147 -> (i64) {
            %185 = arith.index_cast %110#1 : i64 to index
            %186 = memref.load %alloc_6[%185] : memref<1xi32>
            %187 = arith.extsi %186 : i32 to i64
            %188 = arith.cmpi sgt, %187, %c0_i64 : i64
            %189 = arith.cmpi slt, %187, %142 : i64
            %190 = arith.select %189, %187, %142 : i64
            %191 = arith.select %188, %190, %c0_i64 : i64
            scf.yield %191 : i64
          } else {
            scf.yield %142 : i64
          }
          %149 = cannir.datastruct_get %arg22["ori_mask_mode"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
          %150 = arith.cmpi eq, %149, %c3_i32 : i32
          %151 = arith.subi %145, %110#3 : i64
          %152 = arith.addi %151, %110#2 : i64
          %153 = arith.addi %152, %c1_i64 : i64
          %154 = arith.cmpi sgt, %153, %c0_i64 : i64
          %155 = arith.cmpi slt, %153, %145 : i64
          %156 = arith.select %155, %153, %145 : i64
          %157 = arith.select %154, %156, %c0_i64 : i64
          %158 = arith.select %150, %157, %145 : i64
          %159 = arith.cmpi eq, %149, %c4_i32 : i32
          %160:2 = scf.if %159 -> (i64, i64) {
            %185 = cannir.datastruct_get %arg22["ori_win_left"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
            %186 = arith.cmpi sge, %185, %c0_i32 : i32
            %187 = scf.if %186 -> (i64) {
              %191 = cannir.datastruct_get %arg22["ori_win_left"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
              %192 = arith.extsi %191 : i32 to i64
              %193 = arith.subi %152, %192 : i64
              %194 = arith.cmpi sgt, %193, %c0_i64 : i64
              %195 = arith.cmpi slt, %193, %145 : i64
              %196 = arith.select %195, %193, %145 : i64
              %197 = arith.select %194, %196, %c0_i64 : i64
              scf.yield %197 : i64
            } else {
              scf.yield %c0_i64 : i64
            }
            %188 = cannir.datastruct_get %arg22["ori_win_right"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
            %189 = arith.cmpi sge, %188, %c0_i32 : i32
            %190 = scf.if %189 -> (i64) {
              %191 = cannir.datastruct_get %arg22["ori_win_right"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
              %192 = arith.extsi %191 : i32 to i64
              %193 = arith.addi %152, %192 : i64
              %194 = arith.addi %193, %c1_i64 : i64
              %195 = arith.cmpi sgt, %194, %c0_i64 : i64
              %196 = arith.cmpi slt, %194, %145 : i64
              %197 = arith.select %196, %194, %145 : i64
              %198 = arith.select %195, %197, %c0_i64 : i64
              scf.yield %198 : i64
            } else {
              scf.yield %158 : i64
            }
            scf.yield %187, %190 : i64, i64
          } else {
            scf.yield %c0_i64, %158 : i64, i64
          }
          %161 = arith.cmpi sgt, %160#1, %160#0 : i64
          %162 = arith.subi %160#1, %160#0 : i64
          %163 = arith.select %161, %162, %c0_i64 : i64
          %164 = cannir.datastruct_get %arg22["is_ori_dense"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
          %165 = arith.cmpi eq, %164, %c0_i32 : i32
          %166 = scf.if %165 -> (i64) {
            %185 = cannir.datastruct_get %arg22["ori_sparse_index_width"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
            %186 = arith.extsi %185 : i32 to i64
            %187 = arith.cmpi slt, %163, %186 : i64
            %188 = arith.select %187, %163, %186 : i64
            %189 = cannir.datastruct_get %arg22["has_ori_topk_length"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
            %190 = arith.cmpi ne, %189, %c0_i32 : i32
            %191 = scf.if %190 -> (i64) {
              %192 = arith.index_cast %110#0 : i64 to index
              %193 = memref.load %alloc_3[%192] : memref<1xi32>
              %194 = arith.extsi %193 : i32 to i64
              %195 = arith.cmpi sgt, %194, %c0_i64 : i64
              %196 = arith.cmpi slt, %188, %194 : i64
              %197 = arith.select %196, %188, %194 : i64
              %198 = arith.select %195, %197, %c0_i64 : i64
              scf.yield %198 : i64
            } else {
              scf.yield %188 : i64
            }
            scf.yield %191 : i64
          } else {
            scf.yield %163 : i64
          }
          %167 = scf.if %146 -> (i64) {
            %185 = cannir.datastruct_get %arg22["cmp_mask_mode"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
            %186 = arith.cmpi eq, %185, %c3_i32 : i32
            %187 = scf.if %186 -> (i64) {
              %195 = cannir.datastruct_get %arg22["cmp_ratio"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
              %196 = arith.extsi %195 : i32 to i64
              %197 = arith.muli %148, %196 : i64
              %198 = arith.cmpi ne, %195, %c1_i32 : i32
              %199 = scf.if %198 -> (i64) {
                %216 = arith.index_cast %110#1 : i64 to index
                %217 = memref.load %alloc_7[%216] : memref<1xi32>
                %218 = arith.extsi %217 : i32 to i64
                %219 = arith.addi %197, %218 : i64
                scf.yield %219 : i64
              } else {
                scf.yield %197 : i64
              }
              %200 = arith.subi %199, %110#3 : i64
              %201 = arith.addi %200, %110#2 : i64
              %202 = arith.addi %201, %c1_i64 : i64
              %203 = arith.remsi %202, %196 : i64
              %204 = arith.cmpi ne, %203, %c0_i64 : i64
              %205 = arith.cmpi slt, %203, %c0_i64 : i64
              %206 = arith.cmpi slt, %196, %c0_i64 : i64
              %207 = arith.xori %205, %206 : i1
              %208 = arith.andi %204, %207 : i1
              %209 = arith.divsi %202, %196 : i64
              %210 = arith.extui %208 : i1 to i64
              %211 = arith.subi %209, %210 : i64
              %212 = arith.cmpi sgt, %211, %c0_i64 : i64
              %213 = arith.cmpi slt, %211, %148 : i64
              %214 = arith.select %213, %211, %148 : i64
              %215 = arith.select %212, %214, %c0_i64 : i64
              scf.yield %215 : i64
            } else {
              scf.yield %148 : i64
            }
            %188 = cannir.datastruct_get %arg22["cmp_sparse_index_width"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
            %189 = arith.extsi %188 : i32 to i64
            %190 = arith.cmpi slt, %187, %189 : i64
            %191 = arith.select %190, %187, %189 : i64
            %192 = cannir.datastruct_get %arg22["has_cmp_topk_length"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
            %193 = arith.cmpi ne, %192, %c0_i32 : i32
            %194 = scf.if %193 -> (i64) {
              %195 = arith.index_cast %110#0 : i64 to index
              %196 = memref.load %alloc_4[%195] : memref<1xi32>
              %197 = arith.extsi %196 : i32 to i64
              %198 = arith.cmpi sgt, %197, %c0_i64 : i64
              %199 = arith.cmpi slt, %191, %197 : i64
              %200 = arith.select %199, %191, %197 : i64
              %201 = arith.select %198, %200, %c0_i64 : i64
              scf.yield %201 : i64
            } else {
              scf.yield %191 : i64
            }
            scf.yield %194 : i64
          } else {
            scf.yield %c0_i64 : i64
          }
          %168 = arith.addi %166, %c127_i64 : i64
          %169 = arith.remsi %168, %c128_i64 : i64
          %170 = arith.cmpi ne, %169, %c0_i64 : i64
          %171 = arith.cmpi slt, %169, %c0_i64 : i64
          %172 = arith.andi %170, %171 : i1
          %173 = arith.divsi %168, %c128_i64 : i64
          %174 = arith.extui %172 : i1 to i64
          %175 = arith.subi %173, %174 : i64
          %176 = arith.addi %167, %c127_i64 : i64
          %177 = arith.remsi %176, %c128_i64 : i64
          %178 = arith.cmpi ne, %177, %c0_i64 : i64
          %179 = arith.cmpi slt, %177, %c0_i64 : i64
          %180 = arith.andi %178, %179 : i1
          %181 = arith.divsi %176, %c128_i64 : i64
          %182 = arith.extui %180 : i1 to i64
          %183 = arith.subi %181, %182 : i64
          %184 = arith.addi %175, %183 : i64
          scf.yield %138#0, %141, %138#2, %142, %160#0, %166, %167, %175, %184 : i64, i64, i64, i64, i64, i64, i64, i64, i64
        } else {
          scf.yield %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64 : i64, i64, i64, i64, i64, i64, i64, i64, i64
        }
        %115 = arith.cmpi eq, %114#5, %c0_i64 : i64
        %116 = arith.cmpi eq, %114#6, %c0_i64 : i64
        %117 = arith.extui %116 : i1 to i64
        %118 = arith.select %115, %117, %c0_i64 : i64
        %119 = arith.select %115, %116, %false : i1
        scf.if %119 {
          %130 = arith.cmpi sgt, %59, %c0_i64 : i64
          scf.if %130 {
            %131 = cannir.datastruct_get %arg22["num_heads_q"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
            %132 = arith.extsi %131 : i32 to i64
            %133 = arith.muli %110#0, %132 : i64
            %134 = arith.addi %133, %52 : i64
            %135 = arith.muli %59, %c512_i64 : i64
            %136 = arith.muli %134, %c8192_i64 : i64
            %137 = ascscalar.memref_address %alloc_13 : memref<1xbf16>
            %138 = arith.divui %136, %c8_i64 : i64
            %139 = arith.addi %137, %138 : i64
            %140 = ascscalar.bind_gm_ptr %139 : memref<1xbf16>
            %141 = ascscalar.bitcast_memref %alloc_22 byte_offset = %c0_i64 : memref<16384xbf16, 1> -> memref<?xbf16, 1>
            %142 = arith.index_cast %135 : i64 to index
            %143 = arith.muli %142, %c2 : index
            %144 = arith.cmpi ugt, %135, %c0_i64 : i64
            scf.if %144 {
              %158 = arith.index_cast %143 : index to i64
              ascvec.get_buf mte3, %c3_i64, 0 {cannir.buffer.generated_sync}
              %159 = arith.trunci %158 : i64 to i32
              ascvec.copy_ub2gm %140[%c0], %141[%c0], %c1_i32, %159, %c0_i64, %159 l2_cache_ctl %c4_i32 : memref<1xbf16>, memref<?xbf16, 1>
              ascvec.release_buf mte3, %c3_i64, 0 {cannir.buffer.generated_sync}
            }
            %145 = cannir.datastruct_get %arg22["num_heads_q"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
            %146 = arith.extsi %145 : i32 to i64
            %147 = arith.muli %110#0, %146 : i64
            %148 = arith.addi %147, %52 : i64
            %149 = arith.muli %148, %c32_i64 : i64
            %150 = ascscalar.memref_address %alloc_14 : memref<1xf32>
            %151 = arith.divui %149, %c8_i64 : i64
            %152 = arith.addi %150, %151 : i64
            %153 = ascscalar.bind_gm_ptr %152 : memref<1xf32>
            %154 = ascscalar.bitcast_memref %alloc_23 byte_offset = %c0_i64 : memref<32xf32, 1> -> memref<?xf32, 1>
            %155 = arith.index_cast %59 : i64 to index
            %156 = arith.muli %155, %c4 : index
            %157 = arith.cmpi ugt, %59, %c0_i64 : i64
            scf.if %157 {
              %158 = arith.index_cast %156 : index to i64
              ascvec.get_buf mte3, %c4_i64, 0 {cannir.buffer.generated_sync}
              %159 = arith.trunci %158 : i64 to i32
              ascvec.copy_ub2gm %153[%c0], %154[%c0], %c1_i32, %159, %c0_i64, %159 l2_cache_ctl %c0_i32 : memref<1xf32>, memref<?xf32, 1>
              ascvec.release_buf mte3, %c4_i64, 0 {cannir.buffer.generated_sync}
            }
          }
        }
        %120 = arith.cmpi eq, %arg59, %c0_i64 : i64
        %121 = arith.select %120, %5, %c0_i64 : i64
        %122 = arith.addi %arg59, %c1_i64 : i64
        %123 = arith.cmpi eq, %122, %88 : i64
        %124 = arith.cmpi sgt, %9, %c0_i64 : i64
        %125 = arith.select %123, %124, %false : i1
        %126 = arith.minsi %114#8, %9 : i64
        %127 = arith.select %125, %126, %114#8 : i64
        %128 = arith.subi %127, %121 : i64
        %129 = arith.maxsi %128, %c0_i64 : i64
        scf.yield %110#0, %110#1, %114#0, %114#1, %114#2, %114#3, %114#4, %114#5, %114#6, %114#7, %129, %118, %129 : i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64
      }
      %100:37 = scf.for %arg66 = %c0_i64 to %99#12 step %c1_i64 iter_args(%arg67 = %arg26, %arg68 = %arg27, %arg69 = %arg28, %arg70 = %arg29, %arg71 = %arg30, %arg72 = %arg31, %arg73 = %arg32, %arg74 = %arg33, %arg75 = %arg34, %arg76 = %arg35, %arg77 = %arg36, %arg78 = %arg37, %arg79 = %arg38, %arg80 = %arg39, %arg81 = %arg40, %arg82 = %arg41, %arg83 = %arg42, %arg84 = %arg43, %arg85 = %arg44, %arg86 = %arg45, %arg87 = %arg46, %arg88 = %arg47, %arg89 = %arg48, %arg90 = %arg49, %arg91 = %arg50, %arg92 = %arg51, %arg93 = %arg52, %arg94 = %arg53, %arg95 = %arg54, %arg96 = %arg55, %arg97 = %arg56, %arg98 = %arg60, %arg99 = %arg61, %arg100 = %arg62, %arg101 = %arg63, %arg102 = %arg64, %arg103 = %arg65) -> (i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64)  : i64 {
        %108 = arith.cmpi eq, %arg103, %c13_i64 : i64
        %109 = arith.select %108, %c12_i64, %c11_i64 : i64
        %110 = arith.divui %arg88, %c8_i64 : i64
        %111 = arith.addi %arg87, %110 : i64
        %112 = ascscalar.bind_local_ptr %111 : memref<1xi32, 1>
        %113 = arith.select %96, %c-1_i64, %arg66 : i64
        %114 = arith.cmpi eq, %99#0, %29 : i64
        %115 = arith.select %114, %5, %c0_i64 : i64
        %116 = arith.addi %113, %115 : i64
        %117 = arith.cmpi sge, %116, %99#9 : i64
        %118 = arith.subi %116, %99#9 : i64
        %119 = arith.select %117, %118, %116 : i64
        %120 = arith.select %117, %99#8, %99#7 : i64
        %121 = cannir.array_load %36[%c0] : <1xi64>
        %122 = arith.remui %121, %c5_i64 : i64
        %123 = arith.index_castui %122 : i64 to index
        cannir.array_store %arg58, %37[%123] : <5xi64>
        cannir.array_store %99#0, %38[%123] : <5xi64>
        cannir.array_store %113, %39[%123] : <5xi64>
        cannir.array_store %120, %40[%123] : <5xi64>
        cannir.array_store %99#10, %41[%123] : <5xi64>
        cannir.array_store %arg57, %42[%123] : <5xi64>
        cannir.array_store %119, %43[%123] : <5xi64>
        cannir.array_store %99#1, %44[%123] : <5xi64>
        cannir.array_store %99#2, %45[%123] : <5xi64>
        cannir.array_store %99#3, %46[%123] : <5xi64>
        cannir.array_store %99#4, %47[%123] : <5xi64>
        cannir.array_store %99#5, %48[%123] : <5xi64>
        cannir.array_store %99#6, %49[%123] : <5xi64>
        %124 = arith.extui %117 : i1 to i64
        cannir.array_store %124, %50[%123] : <5xi64>
        %125 = arith.cmpi sge, %113, %c0_i64 : i64
        %126 = scf.if %125 -> (i64) {
          %192 = arith.cmpi eq, %arg98, %c0_i64 : i64
          %193 = ascscalar.memref_address %alloc_31 : memref<64xi32, 1>
          %194 = arith.select %192, %92, %193 : i64
          %195 = arith.select %192, %c11_i64, %c12_i64 : i64
          %196 = arith.addi %arg98, %c1_i64 : i64
          %197 = arith.remui %196, %c2_i64 : i64
          scf.if %117 {
            %198 = cannir.datastruct_get %arg22["cmp_sparse_index_width"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
            %199 = arith.muli %119, %c128_i64 : i64
            %200 = arith.muli %51, %c64_i64 : i64
            %201 = arith.addi %199, %200 : i64
            %202 = arith.subi %120, %201 : i64
            %203 = arith.cmpi sgt, %202, %c0_i64 : i64
            %204 = arith.cmpi slt, %202, %c64_i64 : i64
            %205 = arith.select %204, %202, %c64_i64 : i64
            %206 = arith.select %203, %205, %c0_i64 : i64
            %207 = arith.cmpi sgt, %206, %c0_i64 : i64
            scf.if %207 {
              %208 = arith.extsi %198 : i32 to i64
              %209 = arith.muli %99#0, %208 : i64
              %210 = arith.muli %209, %c4_i64 : i64
              %211 = arith.muli %201, %c4_i64 : i64
              %212 = arith.addi %210, %211 : i64
              %213 = arith.muli %206, %c4_i64 : i64
              %214 = ascscalar.bind_local_ptr %194 : memref<1xi32, 1>
              ascvec.get_buf mte2, %195, 0 {cannir.buffer.generated_sync}
              "ascvec.copy_gm2ub_plan"(%214, %alloc_2, %c0_i32, %c0_i32, %c0_i64, %212, %c1_i64, %213, %c0_i64, %213, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = i32, ub_base = -1 : i64, ub_capacity = 4 : i64}> : (memref<1xi32, 1>, memref<1xi32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
              ascvec.release_buf mte2, %195, 0 {cannir.buffer.generated_sync}
            }
          } else {
            %198 = cannir.datastruct_get %arg22["ori_sparse_index_width"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
            %199 = arith.muli %119, %c128_i64 : i64
            %200 = arith.muli %51, %c64_i64 : i64
            %201 = arith.addi %199, %200 : i64
            %202 = arith.subi %120, %201 : i64
            %203 = arith.cmpi sgt, %202, %c0_i64 : i64
            %204 = arith.cmpi slt, %202, %c64_i64 : i64
            %205 = arith.select %204, %202, %c64_i64 : i64
            %206 = arith.select %203, %205, %c0_i64 : i64
            %207 = arith.cmpi sgt, %206, %c0_i64 : i64
            scf.if %207 {
              %208 = arith.extsi %198 : i32 to i64
              %209 = arith.muli %99#0, %208 : i64
              %210 = arith.muli %209, %c4_i64 : i64
              %211 = arith.muli %201, %c4_i64 : i64
              %212 = arith.addi %210, %211 : i64
              %213 = arith.muli %206, %c4_i64 : i64
              %214 = ascscalar.bind_local_ptr %194 : memref<1xi32, 1>
              ascvec.get_buf mte2, %195, 0 {cannir.buffer.generated_sync}
              "ascvec.copy_gm2ub_plan"(%214, %alloc_1, %c0_i32, %c0_i32, %c0_i64, %212, %c1_i64, %213, %c0_i64, %213, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = i32, ub_base = -1 : i64, ub_capacity = 4 : i64}> : (memref<1xi32, 1>, memref<1xi32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
              ascvec.release_buf mte2, %195, 0 {cannir.buffer.generated_sync}
            }
          }
          scf.yield %197 : i64
        } else {
          scf.yield %arg98 : i64
        }
        %127 = cannir.array_load %36[%c0] : <1xi64>
        %128 = arith.addi %127, %c4_i64 : i64
        %129 = arith.remui %128, %c5_i64 : i64
        %130 = arith.index_castui %129 : i64 to index
        %131 = cannir.array_load %39[%130] : <5xi64>
        %132 = cannir.array_load %40[%130] : <5xi64>
        %133 = cannir.array_load %43[%130] : <5xi64>
        %134 = cannir.array_load %47[%130] : <5xi64>
        %135 = cannir.array_load %48[%130] : <5xi64>
        %136 = cannir.array_load %50[%130] : <5xi64>
        %137 = arith.addi %127, %c2_i64 : i64
        %138 = arith.remui %137, %c5_i64 : i64
        %139 = arith.index_castui %138 : i64 to index
        %140 = cannir.array_load %37[%139] : <5xi64>
        %141 = cannir.array_load %38[%139] : <5xi64>
        %142 = cannir.array_load %39[%139] : <5xi64>
        %143 = cannir.array_load %40[%139] : <5xi64>
        %144 = cannir.array_load %41[%139] : <5xi64>
        %145 = cannir.array_load %43[%139] : <5xi64>
        %146 = cannir.array_load %47[%139] : <5xi64>
        %147 = cannir.array_load %48[%139] : <5xi64>
        %148 = cannir.array_load %50[%139] : <5xi64>
        %149 = arith.addi %127, %c1_i64 : i64
        %150 = arith.remui %149, %c5_i64 : i64
        %151 = arith.index_castui %150 : i64 to index
        %152 = cannir.array_load %37[%151] : <5xi64>
        %153 = cannir.array_load %38[%151] : <5xi64>
        %154 = cannir.array_load %39[%151] : <5xi64>
        %155 = cannir.array_load %41[%151] : <5xi64>
        %156 = arith.cmpi ne, %148, %c0_i64 : i64
        %157 = arith.select %156, %147, %146 : i64
        %158 = arith.cmpi sge, %arg97, %c1_i64 : i64
        %159 = arith.cmpi sge, %131, %c0_i64 : i64
        %160 = arith.select %158, %159, %false : i1
        scf.if %160 {
          %192 = arith.cmpi ne, %136, %c0_i64 : i64
          scf.if %192 {
            %193 = arith.cmpi slt, %132, %135 : i64
            %194 = arith.select %193, %132, %135 : i64
            %195 = arith.muli %133, %c128_i64 : i64
            %196 = arith.subi %194, %195 : i64
            %197 = arith.muli %51, %c64_i64 : i64
            %198 = arith.subi %196, %197 : i64
            %199 = arith.cmpi sgt, %198, %c0_i64 : i64
            %200 = arith.cmpi slt, %198, %c64_i64 : i64
            %201 = arith.select %200, %198, %c64_i64 : i64
            %202 = arith.select %199, %201, %c0_i64 : i64
            %203 = arith.cmpi sge, %202, %c16_i64 : i64
            scf.if %203 {
              scf.for %arg104 = %c0_i64 to %c8_i64 step %c1_i64  : i64 {
                %206 = arith.muli %arg104, %c2_i64 overflow<nsw> : i64
                ascvec.get_buf s, %109, 0 {cannir.buffer.generated_sync}
                %207 = arith.index_cast %arg90 : i64 to index
                %208 = arith.index_cast %206 : i64 to index
                %209 = arith.muli %208, %207 : index
                %210 = memref.load %112[%209] : memref<1xi32, 1>
                %211 = arith.extui %210 : i32 to i64
                %212 = arith.addi %206, %c1_i64 overflow<nsw> : i64
                %213 = arith.index_cast %212 : i64 to index
                %214 = arith.muli %213, %207 : index
                %215 = memref.load %112[%214] : memref<1xi32, 1>
                ascvec.release_buf s, %109, 0 {cannir.buffer.generated_sync}
                %216 = arith.extui %215 : i32 to i64
                %217 = arith.cmpi slt, %211, %216 : i64
                %218 = arith.select %217, %211, %216 : i64
                %219 = arith.select %217, %216, %211 : i64
                %220 = arith.muli %218, %c1024_i64 : i64
                %221 = arith.muli %219, %c1024_i64 : i64
                %222 = arith.subi %221, %220 : i64
                %223 = arith.muli %arg104, %c2048_i64 overflow<nsw> : i64
                ascvec.get_buf mte2, %c7_i64, 0 {cannir.buffer.generated_sync}
                "ascvec.copy_gm2ub_plan"(%alloc_26, %alloc_0, %c1_i32, %c0_i32, %223, %220, %c2_i64, %c1024_i64, %222, %c1024_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = 190976 : i64, ub_capacity = 16384 : i64}> : (memref<8192xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                ascvec.release_buf mte2, %c7_i64, 0 {cannir.buffer.generated_sync}
              }
            } else {
              scf.for %arg104 = %c0_i64 to %c8_i64 step %c1_i64  : i64 {
                %206 = arith.muli %arg104, %c2_i64 overflow<nsw> : i64
                %207 = arith.cmpi slt, %206, %202 : i64
                %208 = scf.if %207 -> (i32) {
                  ascvec.get_buf s, %109, 0 {cannir.buffer.generated_sync}
                  %215 = arith.index_cast %arg90 : i64 to index
                  %216 = arith.index_cast %206 : i64 to index
                  %217 = arith.muli %216, %215 : index
                  %218 = memref.load %112[%217] : memref<1xi32, 1>
                  ascvec.release_buf s, %109, 0 {cannir.buffer.generated_sync}
                  scf.yield %218 : i32
                } else {
                  scf.yield %c-1_i32 : i32
                }
                %209 = arith.addi %206, %c1_i64 overflow<nsw> : i64
                %210 = arith.cmpi slt, %209, %202 : i64
                %211 = scf.if %210 -> (i32) {
                  ascvec.get_buf s, %109, 0 {cannir.buffer.generated_sync}
                  %215 = arith.index_cast %arg90 : i64 to index
                  %216 = arith.index_cast %209 : i64 to index
                  %217 = arith.muli %216, %215 : index
                  %218 = memref.load %112[%217] : memref<1xi32, 1>
                  ascvec.release_buf s, %109, 0 {cannir.buffer.generated_sync}
                  scf.yield %218 : i32
                } else {
                  scf.yield %c-1_i32 : i32
                }
                %212 = arith.cmpi sge, %208, %c0_i32 : i32
                %213 = arith.cmpi sge, %211, %c0_i32 : i32
                %214 = arith.select %212, %213, %false : i1
                scf.if %214 {
                  %215 = arith.extui %208 : i32 to i64
                  %216 = arith.extui %211 : i32 to i64
                  %217 = arith.cmpi slt, %215, %216 : i64
                  %218 = arith.select %217, %215, %216 : i64
                  %219 = arith.select %217, %216, %215 : i64
                  %220 = arith.muli %218, %c1024_i64 : i64
                  %221 = arith.muli %219, %c1024_i64 : i64
                  %222 = arith.subi %221, %220 : i64
                  %223 = arith.muli %arg104, %c2048_i64 overflow<nsw> : i64
                  ascvec.get_buf mte2, %c7_i64, 0 {cannir.buffer.generated_sync}
                  "ascvec.copy_gm2ub_plan"(%alloc_26, %alloc_0, %c1_i32, %c0_i32, %223, %220, %c2_i64, %c1024_i64, %222, %c1024_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = 190976 : i64, ub_capacity = 16384 : i64}> : (memref<8192xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                  ascvec.release_buf mte2, %c7_i64, 0 {cannir.buffer.generated_sync}
                } else {
                  %215 = arith.select %212, %true, %213 : i1
                  scf.if %215 {
                    %216 = arith.extsi %208 : i32 to i64
                    %217 = arith.extsi %211 : i32 to i64
                    %218 = arith.select %212, %216, %217 : i64
                    %219 = cannir.extract_layout %arg2 : !cannir.tensor<<gm, bf16>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>
                    %220 = cannir.extract_shape %219 : <<(-1)>, <(1)>> -> <(-1)>
                    %221 = cannir.extract_dim %220[0] : !cannir.shape<(-1)>
                    %222 = arith.divui %221, %c512_i64 : i64
                    %223 = arith.minsi %218, %222 : i64
                    %224 = arith.subi %222, %223 : i64
                    %225 = arith.minsi %224, %c1_i64 : i64
                    %226 = arith.cmpi sle, %218, %222 : i64
                    %227 = arith.select %226, %225, %c0_i64 : i64
                    %228 = arith.muli %218, %c1024_i64 : i64
                    %229 = arith.muli %arg104, %c2048_i64 overflow<nsw> : i64
                    %230 = arith.muli %227, %c1024_i64 : i64
                    %231 = arith.cmpi eq, %230, %c0_i64 : i64
                    %232 = arith.cmpi ne, %230, %c0_i64 : i64
                    %233 = arith.extui %232 : i1 to i64
                    %234 = arith.select %231, %c2_i64, %230 : i64
                    ascvec.get_buf mte2, %c7_i64, 0 {cannir.buffer.generated_sync}
                    "ascvec.copy_gm2ub_plan"(%alloc_26, %alloc_0, %c1_i32, %c0_i32, %229, %228, %233, %234, %c0_i64, %234, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = 190976 : i64, ub_capacity = 16384 : i64}> : (memref<8192xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                    ascvec.release_buf mte2, %c7_i64, 0 {cannir.buffer.generated_sync}
                  }
                }
              }
            }
            %204 = arith.cmpi sge, %202, %c32_i64 : i64
            scf.if %204 {
              scf.for %arg104 = %c0_i64 to %c8_i64 step %c1_i64  : i64 {
                %206 = arith.muli %arg104, %c2_i64 overflow<nsw> : i64
                %207 = arith.addi %206, %c16_i64 overflow<nsw> : i64
                ascvec.get_buf s, %109, 0 {cannir.buffer.generated_sync}
                %208 = arith.index_cast %arg90 : i64 to index
                %209 = arith.index_cast %207 : i64 to index
                %210 = arith.muli %209, %208 : index
                %211 = memref.load %112[%210] : memref<1xi32, 1>
                %212 = arith.extui %211 : i32 to i64
                %213 = arith.addi %206, %c17_i64 overflow<nsw> : i64
                %214 = arith.index_cast %213 : i64 to index
                %215 = arith.muli %214, %208 : index
                %216 = memref.load %112[%215] : memref<1xi32, 1>
                ascvec.release_buf s, %109, 0 {cannir.buffer.generated_sync}
                %217 = arith.extui %216 : i32 to i64
                %218 = arith.cmpi slt, %212, %217 : i64
                %219 = arith.select %218, %212, %217 : i64
                %220 = arith.select %218, %217, %212 : i64
                %221 = arith.muli %219, %c1024_i64 : i64
                %222 = arith.muli %220, %c1024_i64 : i64
                %223 = arith.subi %222, %221 : i64
                %224 = arith.muli %arg104, %c2048_i64 overflow<nsw> : i64
                ascvec.get_buf mte2, %c8_i64, 0 {cannir.buffer.generated_sync}
                "ascvec.copy_gm2ub_plan"(%alloc_27, %alloc_0, %c1_i32, %c0_i32, %224, %221, %c2_i64, %c1024_i64, %223, %c1024_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = 207360 : i64, ub_capacity = 16384 : i64}> : (memref<8192xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                ascvec.release_buf mte2, %c8_i64, 0 {cannir.buffer.generated_sync}
              }
            } else {
              scf.for %arg104 = %c0_i64 to %c8_i64 step %c1_i64  : i64 {
                %206 = arith.muli %arg104, %c2_i64 overflow<nsw> : i64
                %207 = arith.addi %206, %c16_i64 overflow<nsw> : i64
                %208 = arith.cmpi slt, %207, %202 : i64
                %209 = scf.if %208 -> (i32) {
                  ascvec.get_buf s, %109, 0 {cannir.buffer.generated_sync}
                  %216 = arith.index_cast %arg90 : i64 to index
                  %217 = arith.index_cast %207 : i64 to index
                  %218 = arith.muli %217, %216 : index
                  %219 = memref.load %112[%218] : memref<1xi32, 1>
                  ascvec.release_buf s, %109, 0 {cannir.buffer.generated_sync}
                  scf.yield %219 : i32
                } else {
                  scf.yield %c-1_i32 : i32
                }
                %210 = arith.addi %206, %c17_i64 overflow<nsw> : i64
                %211 = arith.cmpi slt, %210, %202 : i64
                %212 = scf.if %211 -> (i32) {
                  ascvec.get_buf s, %109, 0 {cannir.buffer.generated_sync}
                  %216 = arith.index_cast %arg90 : i64 to index
                  %217 = arith.index_cast %210 : i64 to index
                  %218 = arith.muli %217, %216 : index
                  %219 = memref.load %112[%218] : memref<1xi32, 1>
                  ascvec.release_buf s, %109, 0 {cannir.buffer.generated_sync}
                  scf.yield %219 : i32
                } else {
                  scf.yield %c-1_i32 : i32
                }
                %213 = arith.cmpi sge, %209, %c0_i32 : i32
                %214 = arith.cmpi sge, %212, %c0_i32 : i32
                %215 = arith.select %213, %214, %false : i1
                scf.if %215 {
                  %216 = arith.extui %209 : i32 to i64
                  %217 = arith.extui %212 : i32 to i64
                  %218 = arith.cmpi slt, %216, %217 : i64
                  %219 = arith.select %218, %216, %217 : i64
                  %220 = arith.select %218, %217, %216 : i64
                  %221 = arith.muli %219, %c1024_i64 : i64
                  %222 = arith.muli %220, %c1024_i64 : i64
                  %223 = arith.subi %222, %221 : i64
                  %224 = arith.muli %arg104, %c2048_i64 overflow<nsw> : i64
                  ascvec.get_buf mte2, %c8_i64, 0 {cannir.buffer.generated_sync}
                  "ascvec.copy_gm2ub_plan"(%alloc_27, %alloc_0, %c1_i32, %c0_i32, %224, %221, %c2_i64, %c1024_i64, %223, %c1024_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = 207360 : i64, ub_capacity = 16384 : i64}> : (memref<8192xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                  ascvec.release_buf mte2, %c8_i64, 0 {cannir.buffer.generated_sync}
                } else {
                  %216 = arith.select %213, %true, %214 : i1
                  scf.if %216 {
                    %217 = arith.extsi %209 : i32 to i64
                    %218 = arith.extsi %212 : i32 to i64
                    %219 = arith.select %213, %217, %218 : i64
                    %220 = cannir.extract_layout %arg2 : !cannir.tensor<<gm, bf16>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>
                    %221 = cannir.extract_shape %220 : <<(-1)>, <(1)>> -> <(-1)>
                    %222 = cannir.extract_dim %221[0] : !cannir.shape<(-1)>
                    %223 = arith.divui %222, %c512_i64 : i64
                    %224 = arith.minsi %219, %223 : i64
                    %225 = arith.subi %223, %224 : i64
                    %226 = arith.minsi %225, %c1_i64 : i64
                    %227 = arith.cmpi sle, %219, %223 : i64
                    %228 = arith.select %227, %226, %c0_i64 : i64
                    %229 = arith.muli %219, %c1024_i64 : i64
                    %230 = arith.muli %arg104, %c2048_i64 overflow<nsw> : i64
                    %231 = arith.muli %228, %c1024_i64 : i64
                    %232 = arith.cmpi eq, %231, %c0_i64 : i64
                    %233 = arith.cmpi ne, %231, %c0_i64 : i64
                    %234 = arith.extui %233 : i1 to i64
                    %235 = arith.select %232, %c2_i64, %231 : i64
                    ascvec.get_buf mte2, %c8_i64, 0 {cannir.buffer.generated_sync}
                    "ascvec.copy_gm2ub_plan"(%alloc_27, %alloc_0, %c1_i32, %c0_i32, %230, %229, %234, %235, %c0_i64, %235, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = 207360 : i64, ub_capacity = 16384 : i64}> : (memref<8192xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                    ascvec.release_buf mte2, %c8_i64, 0 {cannir.buffer.generated_sync}
                  }
                }
              }
            }
            ascvec.get_buf v, %c7_i64, 0 {cannir.buffer.generated_sync}
            ascvec.get_buf v, %c9_i64, 0 {cannir.buffer.generated_sync}
            ascvec.vf(%alloc_26, %alloc_28 : memref<8192xbf16, 1>, memref<8704xbf16, 1>) {
            ^bb0(%arg104: memref<8192xbf16, 1>, %arg105: memref<8704xbf16, 1>):
              %c1_i32_48 = arith.constant 1 : i32
              %c0_i32_49 = arith.constant 0 : i32
              %c6528_i32 = arith.constant 6528 : i32
              %c384_i32 = arith.constant 384 : i32
              %c4352_i32 = arith.constant 4352 : i32
              %c256_i32 = arith.constant 256 : i32
              %c2176_i32 = arith.constant 2176 : i32
              %c128_i32 = arith.constant 128 : i32
              %c17 = arith.constant 17 : index
              %c16_i32_50 = arith.constant 16 : i32
              %c0_51 = arith.constant 0 : index
              %c512_i32_52 = arith.constant 512 : i32
              %206 = ascvec.create_mask elem_bits 16 : !ascvec.mask
              scf.for %arg106 = %c0_i32_49 to %c16_i32_50 step %c1_i32_48  : i32 {
                %207 = arith.muli %arg106, %c512_i32_52 overflow<nsw> : i32
                %208 = arith.index_cast %207 : i32 to index
                %209 = ascvec.loadalign %arg104[%208] : memref<8192xbf16, 1>, index -> <bf16>
                %210 = arith.muli %arg106, %c16_i32_50 overflow<nsw> : i32
                %211 = arith.index_cast %210 : i32 to index
                ascvec.storealign_strided %arg105[%211], %209, %206, %c17, %c0_51 : memref<8704xbf16, 1>, <bf16>
                %212 = arith.addi %207, %c128_i32 overflow<nsw> : i32
                %213 = arith.index_cast %212 : i32 to index
                %214 = ascvec.loadalign %arg104[%213] : memref<8192xbf16, 1>, index -> <bf16>
                %215 = arith.addi %210, %c2176_i32 overflow<nsw> : i32
                %216 = arith.index_cast %215 : i32 to index
                ascvec.storealign_strided %arg105[%216], %214, %206, %c17, %c0_51 : memref<8704xbf16, 1>, <bf16>
                %217 = arith.addi %207, %c256_i32 overflow<nsw> : i32
                %218 = arith.index_cast %217 : i32 to index
                %219 = ascvec.loadalign %arg104[%218] : memref<8192xbf16, 1>, index -> <bf16>
                %220 = arith.addi %210, %c4352_i32 overflow<nsw> : i32
                %221 = arith.index_cast %220 : i32 to index
                ascvec.storealign_strided %arg105[%221], %219, %206, %c17, %c0_51 : memref<8704xbf16, 1>, <bf16>
                %222 = arith.addi %207, %c384_i32 overflow<nsw> : i32
                %223 = arith.index_cast %222 : i32 to index
                %224 = ascvec.loadalign %arg104[%223] : memref<8192xbf16, 1>, index -> <bf16>
                %225 = arith.addi %210, %c6528_i32 overflow<nsw> : i32
                %226 = arith.index_cast %225 : i32 to index
                ascvec.storealign_strided %arg105[%226], %224, %206, %c17, %c0_51 : memref<8704xbf16, 1>, <bf16>
              }
            } {cannir.vf_mode = "simd"}
            ascvec.release_buf v, %c9_i64, 0 {cannir.buffer.generated_sync}
            ascvec.release_buf v, %c7_i64, 0 {cannir.buffer.generated_sync}
            %205 = arith.cmpi eq, %51, %c0_i64 : i64
            scf.if %205 {
              %206 = ascscalar.bitcast_memref %alloc_28 byte_offset = %c0_i64 : memref<8704xbf16, 1> -> memref<?xbf16, 1>
              %207 = ascscalar.bind_local_ptr %arg67 : memref<1xbf16, 2>
              %208 = arith.index_cast %arg68 : i64 to index
              %209 = arith.divui %208, %c16 : index
              %210 = arith.cmpi eq, %arg101, %c18_i64 : i64
              %211 = arith.select %210, %c4_i64, %c3_i64 : i64
              %212 = arith.cmpi eq, %arg101, %c19_i64 : i64
              %213 = arith.select %212, %c5_i64, %211 : i64
              ascvec.sync_intra_wait mte3, %213 {cannir.sync.banks = 1 : i32}
              ascvec.get_buf mte3, %c9_i64, 0 {cannir.buffer.generated_sync}
              ascvec.copy_ub2l1_strided %207[%209], %206[%c0], %c32_i32, %c16_i32, %c1_i32, %c48_i32 : memref<1xbf16, 2>, memref<?xbf16, 1>
              ascvec.release_buf mte3, %c9_i64, 0 {cannir.buffer.generated_sync}
            } else {
              %206 = ascscalar.bitcast_memref %alloc_28 byte_offset = %c0_i64 : memref<8704xbf16, 1> -> memref<?xbf16, 1>
              %207 = ascscalar.bind_local_ptr %arg77 : memref<1xbf16, 2>
              %208 = arith.index_cast %arg78 : i64 to index
              %209 = arith.divui %208, %c16 : index
              %210 = arith.cmpi eq, %arg102, %c22_i64 : i64
              %211 = arith.select %210, %c8_i64, %c7_i64 : i64
              %212 = arith.cmpi eq, %arg102, %c23_i64 : i64
              %213 = arith.select %212, %c9_i64, %211 : i64
              ascvec.sync_intra_wait mte3, %213 {cannir.sync.banks = 2 : i32}
              ascvec.get_buf mte3, %c9_i64, 0 {cannir.buffer.generated_sync}
              ascvec.copy_ub2l1_strided %207[%209], %206[%c0], %c32_i32, %c16_i32, %c1_i32, %c48_i32 : memref<1xbf16, 2>, memref<?xbf16, 1>
              ascvec.release_buf mte3, %c9_i64, 0 {cannir.buffer.generated_sync}
            }
            ascvec.get_buf v, %c8_i64, 0 {cannir.buffer.generated_sync}
            ascvec.get_buf v, %c10_i64, 0 {cannir.buffer.generated_sync}
            ascvec.vf(%alloc_27, %alloc_29 : memref<8192xbf16, 1>, memref<8704xbf16, 1>) {
            ^bb0(%arg104: memref<8192xbf16, 1>, %arg105: memref<8704xbf16, 1>):
              %c1_i32_48 = arith.constant 1 : i32
              %c0_i32_49 = arith.constant 0 : i32
              %c6528_i32 = arith.constant 6528 : i32
              %c384_i32 = arith.constant 384 : i32
              %c4352_i32 = arith.constant 4352 : i32
              %c256_i32 = arith.constant 256 : i32
              %c2176_i32 = arith.constant 2176 : i32
              %c128_i32 = arith.constant 128 : i32
              %c17 = arith.constant 17 : index
              %c16_i32_50 = arith.constant 16 : i32
              %c0_51 = arith.constant 0 : index
              %c512_i32_52 = arith.constant 512 : i32
              %206 = ascvec.create_mask elem_bits 16 : !ascvec.mask
              scf.for %arg106 = %c0_i32_49 to %c16_i32_50 step %c1_i32_48  : i32 {
                %207 = arith.muli %arg106, %c512_i32_52 overflow<nsw> : i32
                %208 = arith.index_cast %207 : i32 to index
                %209 = ascvec.loadalign %arg104[%208] : memref<8192xbf16, 1>, index -> <bf16>
                %210 = arith.muli %arg106, %c16_i32_50 overflow<nsw> : i32
                %211 = arith.index_cast %210 : i32 to index
                ascvec.storealign_strided %arg105[%211], %209, %206, %c17, %c0_51 : memref<8704xbf16, 1>, <bf16>
                %212 = arith.addi %207, %c128_i32 overflow<nsw> : i32
                %213 = arith.index_cast %212 : i32 to index
                %214 = ascvec.loadalign %arg104[%213] : memref<8192xbf16, 1>, index -> <bf16>
                %215 = arith.addi %210, %c2176_i32 overflow<nsw> : i32
                %216 = arith.index_cast %215 : i32 to index
                ascvec.storealign_strided %arg105[%216], %214, %206, %c17, %c0_51 : memref<8704xbf16, 1>, <bf16>
                %217 = arith.addi %207, %c256_i32 overflow<nsw> : i32
                %218 = arith.index_cast %217 : i32 to index
                %219 = ascvec.loadalign %arg104[%218] : memref<8192xbf16, 1>, index -> <bf16>
                %220 = arith.addi %210, %c4352_i32 overflow<nsw> : i32
                %221 = arith.index_cast %220 : i32 to index
                ascvec.storealign_strided %arg105[%221], %219, %206, %c17, %c0_51 : memref<8704xbf16, 1>, <bf16>
                %222 = arith.addi %207, %c384_i32 overflow<nsw> : i32
                %223 = arith.index_cast %222 : i32 to index
                %224 = ascvec.loadalign %arg104[%223] : memref<8192xbf16, 1>, index -> <bf16>
                %225 = arith.addi %210, %c6528_i32 overflow<nsw> : i32
                %226 = arith.index_cast %225 : i32 to index
                ascvec.storealign_strided %arg105[%226], %224, %206, %c17, %c0_51 : memref<8704xbf16, 1>, <bf16>
              }
            } {cannir.vf_mode = "simd"}
            ascvec.release_buf v, %c10_i64, 0 {cannir.buffer.generated_sync}
            ascvec.release_buf v, %c8_i64, 0 {cannir.buffer.generated_sync}
            ascvec.get_buf mte3, %c10_i64, 0 {cannir.buffer.generated_sync}
            scf.if %205 {
              %206 = ascscalar.bitcast_memref %alloc_29 byte_offset = %c0_i64 : memref<8704xbf16, 1> -> memref<?xbf16, 1>
              %207 = ascscalar.bind_local_ptr %arg67 : memref<1xbf16, 2>
              %208 = arith.index_cast %arg68 : i64 to index
              %209 = arith.divui %208, %c16 : index
              %210 = arith.addi %209, %c256 : index
              ascvec.copy_ub2l1_strided %207[%210], %206[%c0], %c32_i32, %c16_i32, %c1_i32, %c48_i32 : memref<1xbf16, 2>, memref<?xbf16, 1>
            } else {
              %206 = ascscalar.bitcast_memref %alloc_29 byte_offset = %c0_i64 : memref<8704xbf16, 1> -> memref<?xbf16, 1>
              %207 = ascscalar.bind_local_ptr %arg77 : memref<1xbf16, 2>
              %208 = arith.index_cast %arg78 : i64 to index
              %209 = arith.divui %208, %c16 : index
              %210 = arith.addi %209, %c256 : index
              ascvec.copy_ub2l1_strided %207[%210], %206[%c0], %c32_i32, %c16_i32, %c1_i32, %c48_i32 : memref<1xbf16, 2>, memref<?xbf16, 1>
            }
            ascvec.release_buf mte3, %c10_i64, 0 {cannir.buffer.generated_sync}
          } else {
            %193 = arith.cmpi slt, %132, %134 : i64
            %194 = arith.select %193, %132, %134 : i64
            %195 = arith.muli %133, %c128_i64 : i64
            %196 = arith.subi %194, %195 : i64
            %197 = arith.muli %51, %c64_i64 : i64
            %198 = arith.subi %196, %197 : i64
            %199 = arith.cmpi sgt, %198, %c0_i64 : i64
            %200 = arith.cmpi slt, %198, %c64_i64 : i64
            %201 = arith.select %200, %198, %c64_i64 : i64
            %202 = arith.select %199, %201, %c0_i64 : i64
            %203 = arith.cmpi sge, %202, %c16_i64 : i64
            scf.if %203 {
              scf.for %arg104 = %c0_i64 to %c8_i64 step %c1_i64  : i64 {
                %206 = arith.muli %arg104, %c2_i64 overflow<nsw> : i64
                ascvec.get_buf s, %109, 0 {cannir.buffer.generated_sync}
                %207 = arith.index_cast %arg90 : i64 to index
                %208 = arith.index_cast %206 : i64 to index
                %209 = arith.muli %208, %207 : index
                %210 = memref.load %112[%209] : memref<1xi32, 1>
                %211 = arith.extui %210 : i32 to i64
                %212 = arith.addi %206, %c1_i64 overflow<nsw> : i64
                %213 = arith.index_cast %212 : i64 to index
                %214 = arith.muli %213, %207 : index
                %215 = memref.load %112[%214] : memref<1xi32, 1>
                ascvec.release_buf s, %109, 0 {cannir.buffer.generated_sync}
                %216 = arith.extui %215 : i32 to i64
                %217 = arith.cmpi slt, %211, %216 : i64
                %218 = arith.select %217, %211, %216 : i64
                %219 = arith.select %217, %216, %211 : i64
                %220 = arith.muli %218, %c1024_i64 : i64
                %221 = arith.muli %219, %c1024_i64 : i64
                %222 = arith.subi %221, %220 : i64
                %223 = arith.muli %arg104, %c2048_i64 overflow<nsw> : i64
                ascvec.get_buf mte2, %c7_i64, 0 {cannir.buffer.generated_sync}
                "ascvec.copy_gm2ub_plan"(%alloc_26, %alloc, %c1_i32, %c0_i32, %223, %220, %c2_i64, %c1024_i64, %222, %c1024_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = 190976 : i64, ub_capacity = 16384 : i64}> : (memref<8192xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                ascvec.release_buf mte2, %c7_i64, 0 {cannir.buffer.generated_sync}
              }
            } else {
              scf.for %arg104 = %c0_i64 to %c8_i64 step %c1_i64  : i64 {
                %206 = arith.muli %arg104, %c2_i64 overflow<nsw> : i64
                %207 = arith.cmpi slt, %206, %202 : i64
                %208 = scf.if %207 -> (i32) {
                  ascvec.get_buf s, %109, 0 {cannir.buffer.generated_sync}
                  %215 = arith.index_cast %arg90 : i64 to index
                  %216 = arith.index_cast %206 : i64 to index
                  %217 = arith.muli %216, %215 : index
                  %218 = memref.load %112[%217] : memref<1xi32, 1>
                  ascvec.release_buf s, %109, 0 {cannir.buffer.generated_sync}
                  scf.yield %218 : i32
                } else {
                  scf.yield %c-1_i32 : i32
                }
                %209 = arith.addi %206, %c1_i64 overflow<nsw> : i64
                %210 = arith.cmpi slt, %209, %202 : i64
                %211 = scf.if %210 -> (i32) {
                  ascvec.get_buf s, %109, 0 {cannir.buffer.generated_sync}
                  %215 = arith.index_cast %arg90 : i64 to index
                  %216 = arith.index_cast %209 : i64 to index
                  %217 = arith.muli %216, %215 : index
                  %218 = memref.load %112[%217] : memref<1xi32, 1>
                  ascvec.release_buf s, %109, 0 {cannir.buffer.generated_sync}
                  scf.yield %218 : i32
                } else {
                  scf.yield %c-1_i32 : i32
                }
                %212 = arith.cmpi sge, %208, %c0_i32 : i32
                %213 = arith.cmpi sge, %211, %c0_i32 : i32
                %214 = arith.select %212, %213, %false : i1
                scf.if %214 {
                  %215 = arith.extui %208 : i32 to i64
                  %216 = arith.extui %211 : i32 to i64
                  %217 = arith.cmpi slt, %215, %216 : i64
                  %218 = arith.select %217, %215, %216 : i64
                  %219 = arith.select %217, %216, %215 : i64
                  %220 = arith.muli %218, %c1024_i64 : i64
                  %221 = arith.muli %219, %c1024_i64 : i64
                  %222 = arith.subi %221, %220 : i64
                  %223 = arith.muli %arg104, %c2048_i64 overflow<nsw> : i64
                  ascvec.get_buf mte2, %c7_i64, 0 {cannir.buffer.generated_sync}
                  "ascvec.copy_gm2ub_plan"(%alloc_26, %alloc, %c1_i32, %c0_i32, %223, %220, %c2_i64, %c1024_i64, %222, %c1024_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = 190976 : i64, ub_capacity = 16384 : i64}> : (memref<8192xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                  ascvec.release_buf mte2, %c7_i64, 0 {cannir.buffer.generated_sync}
                } else {
                  %215 = arith.select %212, %true, %213 : i1
                  scf.if %215 {
                    %216 = arith.extsi %208 : i32 to i64
                    %217 = arith.extsi %211 : i32 to i64
                    %218 = arith.select %212, %216, %217 : i64
                    %219 = cannir.extract_layout %arg1 : !cannir.tensor<<gm, bf16>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>
                    %220 = cannir.extract_shape %219 : <<(-1)>, <(1)>> -> <(-1)>
                    %221 = cannir.extract_dim %220[0] : !cannir.shape<(-1)>
                    %222 = arith.divui %221, %c512_i64 : i64
                    %223 = arith.minsi %218, %222 : i64
                    %224 = arith.subi %222, %223 : i64
                    %225 = arith.minsi %224, %c1_i64 : i64
                    %226 = arith.cmpi sle, %218, %222 : i64
                    %227 = arith.select %226, %225, %c0_i64 : i64
                    %228 = arith.muli %218, %c1024_i64 : i64
                    %229 = arith.muli %arg104, %c2048_i64 overflow<nsw> : i64
                    %230 = arith.muli %227, %c1024_i64 : i64
                    %231 = arith.cmpi eq, %230, %c0_i64 : i64
                    %232 = arith.cmpi ne, %230, %c0_i64 : i64
                    %233 = arith.extui %232 : i1 to i64
                    %234 = arith.select %231, %c2_i64, %230 : i64
                    ascvec.get_buf mte2, %c7_i64, 0 {cannir.buffer.generated_sync}
                    "ascvec.copy_gm2ub_plan"(%alloc_26, %alloc, %c1_i32, %c0_i32, %229, %228, %233, %234, %c0_i64, %234, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = 190976 : i64, ub_capacity = 16384 : i64}> : (memref<8192xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                    ascvec.release_buf mte2, %c7_i64, 0 {cannir.buffer.generated_sync}
                  }
                }
              }
            }
            %204 = arith.cmpi sge, %202, %c32_i64 : i64
            scf.if %204 {
              scf.for %arg104 = %c0_i64 to %c8_i64 step %c1_i64  : i64 {
                %206 = arith.muli %arg104, %c2_i64 overflow<nsw> : i64
                %207 = arith.addi %206, %c16_i64 overflow<nsw> : i64
                ascvec.get_buf s, %109, 0 {cannir.buffer.generated_sync}
                %208 = arith.index_cast %arg90 : i64 to index
                %209 = arith.index_cast %207 : i64 to index
                %210 = arith.muli %209, %208 : index
                %211 = memref.load %112[%210] : memref<1xi32, 1>
                %212 = arith.extui %211 : i32 to i64
                %213 = arith.addi %206, %c17_i64 overflow<nsw> : i64
                %214 = arith.index_cast %213 : i64 to index
                %215 = arith.muli %214, %208 : index
                %216 = memref.load %112[%215] : memref<1xi32, 1>
                ascvec.release_buf s, %109, 0 {cannir.buffer.generated_sync}
                %217 = arith.extui %216 : i32 to i64
                %218 = arith.cmpi slt, %212, %217 : i64
                %219 = arith.select %218, %212, %217 : i64
                %220 = arith.select %218, %217, %212 : i64
                %221 = arith.muli %219, %c1024_i64 : i64
                %222 = arith.muli %220, %c1024_i64 : i64
                %223 = arith.subi %222, %221 : i64
                %224 = arith.muli %arg104, %c2048_i64 overflow<nsw> : i64
                ascvec.get_buf mte2, %c8_i64, 0 {cannir.buffer.generated_sync}
                "ascvec.copy_gm2ub_plan"(%alloc_27, %alloc, %c1_i32, %c0_i32, %224, %221, %c2_i64, %c1024_i64, %223, %c1024_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = 207360 : i64, ub_capacity = 16384 : i64}> : (memref<8192xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                ascvec.release_buf mte2, %c8_i64, 0 {cannir.buffer.generated_sync}
              }
            } else {
              scf.for %arg104 = %c0_i64 to %c8_i64 step %c1_i64  : i64 {
                %206 = arith.muli %arg104, %c2_i64 overflow<nsw> : i64
                %207 = arith.addi %206, %c16_i64 overflow<nsw> : i64
                %208 = arith.cmpi slt, %207, %202 : i64
                %209 = scf.if %208 -> (i32) {
                  ascvec.get_buf s, %109, 0 {cannir.buffer.generated_sync}
                  %216 = arith.index_cast %arg90 : i64 to index
                  %217 = arith.index_cast %207 : i64 to index
                  %218 = arith.muli %217, %216 : index
                  %219 = memref.load %112[%218] : memref<1xi32, 1>
                  ascvec.release_buf s, %109, 0 {cannir.buffer.generated_sync}
                  scf.yield %219 : i32
                } else {
                  scf.yield %c-1_i32 : i32
                }
                %210 = arith.addi %206, %c17_i64 overflow<nsw> : i64
                %211 = arith.cmpi slt, %210, %202 : i64
                %212 = scf.if %211 -> (i32) {
                  ascvec.get_buf s, %109, 0 {cannir.buffer.generated_sync}
                  %216 = arith.index_cast %arg90 : i64 to index
                  %217 = arith.index_cast %210 : i64 to index
                  %218 = arith.muli %217, %216 : index
                  %219 = memref.load %112[%218] : memref<1xi32, 1>
                  ascvec.release_buf s, %109, 0 {cannir.buffer.generated_sync}
                  scf.yield %219 : i32
                } else {
                  scf.yield %c-1_i32 : i32
                }
                %213 = arith.cmpi sge, %209, %c0_i32 : i32
                %214 = arith.cmpi sge, %212, %c0_i32 : i32
                %215 = arith.select %213, %214, %false : i1
                scf.if %215 {
                  %216 = arith.extui %209 : i32 to i64
                  %217 = arith.extui %212 : i32 to i64
                  %218 = arith.cmpi slt, %216, %217 : i64
                  %219 = arith.select %218, %216, %217 : i64
                  %220 = arith.select %218, %217, %216 : i64
                  %221 = arith.muli %219, %c1024_i64 : i64
                  %222 = arith.muli %220, %c1024_i64 : i64
                  %223 = arith.subi %222, %221 : i64
                  %224 = arith.muli %arg104, %c2048_i64 overflow<nsw> : i64
                  ascvec.get_buf mte2, %c8_i64, 0 {cannir.buffer.generated_sync}
                  "ascvec.copy_gm2ub_plan"(%alloc_27, %alloc, %c1_i32, %c0_i32, %224, %221, %c2_i64, %c1024_i64, %223, %c1024_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = 207360 : i64, ub_capacity = 16384 : i64}> : (memref<8192xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                  ascvec.release_buf mte2, %c8_i64, 0 {cannir.buffer.generated_sync}
                } else {
                  %216 = arith.select %213, %true, %214 : i1
                  scf.if %216 {
                    %217 = arith.extsi %209 : i32 to i64
                    %218 = arith.extsi %212 : i32 to i64
                    %219 = arith.select %213, %217, %218 : i64
                    %220 = cannir.extract_layout %arg1 : !cannir.tensor<<gm, bf16>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>
                    %221 = cannir.extract_shape %220 : <<(-1)>, <(1)>> -> <(-1)>
                    %222 = cannir.extract_dim %221[0] : !cannir.shape<(-1)>
                    %223 = arith.divui %222, %c512_i64 : i64
                    %224 = arith.minsi %219, %223 : i64
                    %225 = arith.subi %223, %224 : i64
                    %226 = arith.minsi %225, %c1_i64 : i64
                    %227 = arith.cmpi sle, %219, %223 : i64
                    %228 = arith.select %227, %226, %c0_i64 : i64
                    %229 = arith.muli %219, %c1024_i64 : i64
                    %230 = arith.muli %arg104, %c2048_i64 overflow<nsw> : i64
                    %231 = arith.muli %228, %c1024_i64 : i64
                    %232 = arith.cmpi eq, %231, %c0_i64 : i64
                    %233 = arith.cmpi ne, %231, %c0_i64 : i64
                    %234 = arith.extui %233 : i1 to i64
                    %235 = arith.select %232, %c2_i64, %231 : i64
                    ascvec.get_buf mte2, %c8_i64, 0 {cannir.buffer.generated_sync}
                    "ascvec.copy_gm2ub_plan"(%alloc_27, %alloc, %c1_i32, %c0_i32, %230, %229, %234, %235, %c0_i64, %235, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = 207360 : i64, ub_capacity = 16384 : i64}> : (memref<8192xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                    ascvec.release_buf mte2, %c8_i64, 0 {cannir.buffer.generated_sync}
                  }
                }
              }
            }
            ascvec.get_buf v, %c7_i64, 0 {cannir.buffer.generated_sync}
            ascvec.get_buf v, %c9_i64, 0 {cannir.buffer.generated_sync}
            ascvec.vf(%alloc_26, %alloc_28 : memref<8192xbf16, 1>, memref<8704xbf16, 1>) {
            ^bb0(%arg104: memref<8192xbf16, 1>, %arg105: memref<8704xbf16, 1>):
              %c1_i32_48 = arith.constant 1 : i32
              %c0_i32_49 = arith.constant 0 : i32
              %c6528_i32 = arith.constant 6528 : i32
              %c384_i32 = arith.constant 384 : i32
              %c4352_i32 = arith.constant 4352 : i32
              %c256_i32 = arith.constant 256 : i32
              %c2176_i32 = arith.constant 2176 : i32
              %c128_i32 = arith.constant 128 : i32
              %c17 = arith.constant 17 : index
              %c16_i32_50 = arith.constant 16 : i32
              %c0_51 = arith.constant 0 : index
              %c512_i32_52 = arith.constant 512 : i32
              %206 = ascvec.create_mask elem_bits 16 : !ascvec.mask
              scf.for %arg106 = %c0_i32_49 to %c16_i32_50 step %c1_i32_48  : i32 {
                %207 = arith.muli %arg106, %c512_i32_52 overflow<nsw> : i32
                %208 = arith.index_cast %207 : i32 to index
                %209 = ascvec.loadalign %arg104[%208] : memref<8192xbf16, 1>, index -> <bf16>
                %210 = arith.muli %arg106, %c16_i32_50 overflow<nsw> : i32
                %211 = arith.index_cast %210 : i32 to index
                ascvec.storealign_strided %arg105[%211], %209, %206, %c17, %c0_51 : memref<8704xbf16, 1>, <bf16>
                %212 = arith.addi %207, %c128_i32 overflow<nsw> : i32
                %213 = arith.index_cast %212 : i32 to index
                %214 = ascvec.loadalign %arg104[%213] : memref<8192xbf16, 1>, index -> <bf16>
                %215 = arith.addi %210, %c2176_i32 overflow<nsw> : i32
                %216 = arith.index_cast %215 : i32 to index
                ascvec.storealign_strided %arg105[%216], %214, %206, %c17, %c0_51 : memref<8704xbf16, 1>, <bf16>
                %217 = arith.addi %207, %c256_i32 overflow<nsw> : i32
                %218 = arith.index_cast %217 : i32 to index
                %219 = ascvec.loadalign %arg104[%218] : memref<8192xbf16, 1>, index -> <bf16>
                %220 = arith.addi %210, %c4352_i32 overflow<nsw> : i32
                %221 = arith.index_cast %220 : i32 to index
                ascvec.storealign_strided %arg105[%221], %219, %206, %c17, %c0_51 : memref<8704xbf16, 1>, <bf16>
                %222 = arith.addi %207, %c384_i32 overflow<nsw> : i32
                %223 = arith.index_cast %222 : i32 to index
                %224 = ascvec.loadalign %arg104[%223] : memref<8192xbf16, 1>, index -> <bf16>
                %225 = arith.addi %210, %c6528_i32 overflow<nsw> : i32
                %226 = arith.index_cast %225 : i32 to index
                ascvec.storealign_strided %arg105[%226], %224, %206, %c17, %c0_51 : memref<8704xbf16, 1>, <bf16>
              }
            } {cannir.vf_mode = "simd"}
            ascvec.release_buf v, %c9_i64, 0 {cannir.buffer.generated_sync}
            ascvec.release_buf v, %c7_i64, 0 {cannir.buffer.generated_sync}
            %205 = arith.cmpi eq, %51, %c0_i64 : i64
            scf.if %205 {
              %206 = ascscalar.bitcast_memref %alloc_28 byte_offset = %c0_i64 : memref<8704xbf16, 1> -> memref<?xbf16, 1>
              %207 = ascscalar.bind_local_ptr %arg67 : memref<1xbf16, 2>
              %208 = arith.index_cast %arg68 : i64 to index
              %209 = arith.divui %208, %c16 : index
              %210 = arith.cmpi eq, %arg101, %c18_i64 : i64
              %211 = arith.select %210, %c4_i64, %c3_i64 : i64
              %212 = arith.cmpi eq, %arg101, %c19_i64 : i64
              %213 = arith.select %212, %c5_i64, %211 : i64
              ascvec.sync_intra_wait mte3, %213 {cannir.sync.banks = 1 : i32}
              ascvec.get_buf mte3, %c9_i64, 0 {cannir.buffer.generated_sync}
              ascvec.copy_ub2l1_strided %207[%209], %206[%c0], %c32_i32, %c16_i32, %c1_i32, %c48_i32 : memref<1xbf16, 2>, memref<?xbf16, 1>
              ascvec.release_buf mte3, %c9_i64, 0 {cannir.buffer.generated_sync}
            } else {
              %206 = ascscalar.bitcast_memref %alloc_28 byte_offset = %c0_i64 : memref<8704xbf16, 1> -> memref<?xbf16, 1>
              %207 = ascscalar.bind_local_ptr %arg77 : memref<1xbf16, 2>
              %208 = arith.index_cast %arg78 : i64 to index
              %209 = arith.divui %208, %c16 : index
              %210 = arith.cmpi eq, %arg102, %c22_i64 : i64
              %211 = arith.select %210, %c8_i64, %c7_i64 : i64
              %212 = arith.cmpi eq, %arg102, %c23_i64 : i64
              %213 = arith.select %212, %c9_i64, %211 : i64
              ascvec.sync_intra_wait mte3, %213 {cannir.sync.banks = 2 : i32}
              ascvec.get_buf mte3, %c9_i64, 0 {cannir.buffer.generated_sync}
              ascvec.copy_ub2l1_strided %207[%209], %206[%c0], %c32_i32, %c16_i32, %c1_i32, %c48_i32 : memref<1xbf16, 2>, memref<?xbf16, 1>
              ascvec.release_buf mte3, %c9_i64, 0 {cannir.buffer.generated_sync}
            }
            ascvec.get_buf v, %c8_i64, 0 {cannir.buffer.generated_sync}
            ascvec.get_buf v, %c10_i64, 0 {cannir.buffer.generated_sync}
            ascvec.vf(%alloc_27, %alloc_29 : memref<8192xbf16, 1>, memref<8704xbf16, 1>) {
            ^bb0(%arg104: memref<8192xbf16, 1>, %arg105: memref<8704xbf16, 1>):
              %c1_i32_48 = arith.constant 1 : i32
              %c0_i32_49 = arith.constant 0 : i32
              %c6528_i32 = arith.constant 6528 : i32
              %c384_i32 = arith.constant 384 : i32
              %c4352_i32 = arith.constant 4352 : i32
              %c256_i32 = arith.constant 256 : i32
              %c2176_i32 = arith.constant 2176 : i32
              %c128_i32 = arith.constant 128 : i32
              %c17 = arith.constant 17 : index
              %c16_i32_50 = arith.constant 16 : i32
              %c0_51 = arith.constant 0 : index
              %c512_i32_52 = arith.constant 512 : i32
              %206 = ascvec.create_mask elem_bits 16 : !ascvec.mask
              scf.for %arg106 = %c0_i32_49 to %c16_i32_50 step %c1_i32_48  : i32 {
                %207 = arith.muli %arg106, %c512_i32_52 overflow<nsw> : i32
                %208 = arith.index_cast %207 : i32 to index
                %209 = ascvec.loadalign %arg104[%208] : memref<8192xbf16, 1>, index -> <bf16>
                %210 = arith.muli %arg106, %c16_i32_50 overflow<nsw> : i32
                %211 = arith.index_cast %210 : i32 to index
                ascvec.storealign_strided %arg105[%211], %209, %206, %c17, %c0_51 : memref<8704xbf16, 1>, <bf16>
                %212 = arith.addi %207, %c128_i32 overflow<nsw> : i32
                %213 = arith.index_cast %212 : i32 to index
                %214 = ascvec.loadalign %arg104[%213] : memref<8192xbf16, 1>, index -> <bf16>
                %215 = arith.addi %210, %c2176_i32 overflow<nsw> : i32
                %216 = arith.index_cast %215 : i32 to index
                ascvec.storealign_strided %arg105[%216], %214, %206, %c17, %c0_51 : memref<8704xbf16, 1>, <bf16>
                %217 = arith.addi %207, %c256_i32 overflow<nsw> : i32
                %218 = arith.index_cast %217 : i32 to index
                %219 = ascvec.loadalign %arg104[%218] : memref<8192xbf16, 1>, index -> <bf16>
                %220 = arith.addi %210, %c4352_i32 overflow<nsw> : i32
                %221 = arith.index_cast %220 : i32 to index
                ascvec.storealign_strided %arg105[%221], %219, %206, %c17, %c0_51 : memref<8704xbf16, 1>, <bf16>
                %222 = arith.addi %207, %c384_i32 overflow<nsw> : i32
                %223 = arith.index_cast %222 : i32 to index
                %224 = ascvec.loadalign %arg104[%223] : memref<8192xbf16, 1>, index -> <bf16>
                %225 = arith.addi %210, %c6528_i32 overflow<nsw> : i32
                %226 = arith.index_cast %225 : i32 to index
                ascvec.storealign_strided %arg105[%226], %224, %206, %c17, %c0_51 : memref<8704xbf16, 1>, <bf16>
              }
            } {cannir.vf_mode = "simd"}
            ascvec.release_buf v, %c10_i64, 0 {cannir.buffer.generated_sync}
            ascvec.release_buf v, %c8_i64, 0 {cannir.buffer.generated_sync}
            ascvec.get_buf mte3, %c10_i64, 0 {cannir.buffer.generated_sync}
            scf.if %205 {
              %206 = ascscalar.bitcast_memref %alloc_29 byte_offset = %c0_i64 : memref<8704xbf16, 1> -> memref<?xbf16, 1>
              %207 = ascscalar.bind_local_ptr %arg67 : memref<1xbf16, 2>
              %208 = arith.index_cast %arg68 : i64 to index
              %209 = arith.divui %208, %c16 : index
              %210 = arith.addi %209, %c256 : index
              ascvec.copy_ub2l1_strided %207[%210], %206[%c0], %c32_i32, %c16_i32, %c1_i32, %c48_i32 : memref<1xbf16, 2>, memref<?xbf16, 1>
            } else {
              %206 = ascscalar.bitcast_memref %alloc_29 byte_offset = %c0_i64 : memref<8704xbf16, 1> -> memref<?xbf16, 1>
              %207 = ascscalar.bind_local_ptr %arg77 : memref<1xbf16, 2>
              %208 = arith.index_cast %arg78 : i64 to index
              %209 = arith.divui %208, %c16 : index
              %210 = arith.addi %209, %c256 : index
              ascvec.copy_ub2l1_strided %207[%210], %206[%c0], %c32_i32, %c16_i32, %c1_i32, %c48_i32 : memref<1xbf16, 2>, memref<?xbf16, 1>
            }
            ascvec.release_buf mte3, %c10_i64, 0 {cannir.buffer.generated_sync}
          }
        }
        %161 = arith.cmpi sge, %arg97, %c3_i64 : i64
        %162 = arith.cmpi sge, %142, %c0_i64 : i64
        %163 = arith.select %161, %162, %false : i1
        scf.if %163 {
          %192 = arith.cmpi eq, %141, %29 : i64
          %193 = arith.cmpi sgt, %5, %c0_i64 : i64
          %194 = arith.select %192, %193, %false : i1
          %195 = arith.cmpi eq, %141, %35 : i64
          %196 = arith.cmpi sgt, %9, %c0_i64 : i64
          %197 = arith.select %195, %196, %false : i1
          %198 = arith.extui %193 : i1 to i64
          %199 = arith.addi %13, %198 : i64
          %200 = arith.select %197, %199, %c-1_i64 : i64
          %201 = arith.select %194, %13, %200 : i64
          %202 = arith.cmpi slt, %143, %157 : i64
          %203 = arith.select %202, %143, %157 : i64
          %204 = arith.muli %145, %c128_i64 : i64
          %205 = arith.subi %203, %204 : i64
          %206 = arith.cmpi sgt, %205, %c0_i64 : i64
          %207 = arith.cmpi slt, %205, %c128_i64 : i64
          %208 = arith.select %207, %205, %c128_i64 : i64
          %209 = arith.select %206, %208, %c0_i64 : i64
          %210 = arith.cmpi eq, %209, %c128_i64 : i64
          scf.if %210 {
            %214 = arith.cmpi eq, %142, %c0_i64 : i64
            scf.if %214 {
              %215 = cannir.datastruct_get %arg22["softmax_scale"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
              %216 = arith.remsi %140, %c2_i64 : i64
              %217 = arith.cmpi ne, %216, %c0_i64 : i64
              %218 = arith.cmpi slt, %216, %c0_i64 : i64
              %219 = arith.andi %217, %218 : i1
              %220 = arith.select %219, %c2_i64, %c0_i64 : i64
              %221 = arith.addi %216, %220 overflow<nsw> : i64
              %222 = arith.muli %221, %c64_i64 overflow<nsw> : i64
              %223 = arith.extui %194 : i1 to i32
              %224 = arith.index_cast %222 : i64 to index
              %225 = arith.muli %221, %c128_i64 overflow<nsw> : i64
              %226 = arith.index_cast %225 : i64 to index
              ascvec.sync_intra_wait v, %c0_i64 {cannir.sync.banks = 3 : i32}
              ascvec.get_buf v, %c1_i64, 0 {cannir.buffer.generated_sync}
              ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
              ascvec.get_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
              ascvec.vf(%alloc_18, %223, %alloc_19, %224, %alloc_20, %alloc_16, %215, %222, %alloc_17, %alloc_21, %226 : memref<32xf32, 1>, i32, memref<128xf32, 1>, index, memref<128xf32, 1>, memref<4096xf32, 1>, f32, i64, memref<4224xbf16, 1>, memref<256xf32, 1>, index) {
              ^bb0(%arg104: memref<32xf32, 1>, %arg105: i32, %arg106: memref<128xf32, 1>, %arg107: index, %arg108: memref<128xf32, 1>, %arg109: memref<4096xf32, 1>, %arg110: f32, %arg111: i64, %arg112: memref<4224xbf16, 1>, %arg113: memref<256xf32, 1>, %arg114: index):
                %c33 = arith.constant 33 : index
                %c16_i32_48 = arith.constant 16 : i32
                %c32_i32_49 = arith.constant 32 : i32
                %c0_i32_50 = arith.constant 0 : i32
                %c128_i32 = arith.constant 128 : i32
                %cst = arith.constant 0.000000e+00 : f32
                %cst_51 = arith.constant 1.000000e+00 : f32
                %cst_52 = arith.constant -3.40282347E+38 : f32
                %c1_i32_53 = arith.constant 1 : i32
                %c-8388608_i32 = arith.constant -8388608 : i32
                %c0_54 = arith.constant 0 : index
                %c2_i32_55 = arith.constant 2 : i32
                %227 = ascvec.create_mask : !ascvec.mask
                %228 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                %mask, %remaining_out = ascvec.update_mask %c2_i32_55 : !ascvec.mask
                %229 = ascvec.loadalign %arg104[%c0_54] : memref<32xf32, 1>, index -> <f32>
                %230 = ascvec.duplicate_scalar_nomask %c-8388608_i32 : i32 -> <ui32>
                %231 = ascvec.reinterpret %230 : <ui32> -> <f32>
                %232 = ascvec.duplicate_scalar_nomask %arg105 : i32 -> <i32>
                %233 = ascvec.cmp_ge_scalar %232, %c1_i32_53, %227 : <i32>, i32 -> !ascvec.mask
                %234 = ascvec.select %231, %229, %233 : <f32>
                %235 = ascvec.reinterpret %234 : <f32> -> <ui32>
                %236 = ascvec.cmp_eq_scalar %235, %c-8388608_i32, %227 : <ui32>, i32 -> !ascvec.mask
                %237 = ascvec.duplicate_scalar %cst_52, %236 merge %234 : !ascvec.vreg<f32> {mode = "merging"} : f32 -> <f32>
                %low, %high = ascvec.intlv %237, %237 : <f32>, <f32> -> <f32>, <f32>
                ascvec.storealign %arg106[%arg107], %low, %227 : memref<128xf32, 1>, index, <f32>
                %238 = ascvec.duplicate_scalar_nomask %cst_51 : f32 -> <f32>
                %239 = ascvec.duplicate_scalar %cst, %236 merge %238 : !ascvec.vreg<f32> {mode = "merging"} : f32 -> <f32>
                %low_56, %high_57 = ascvec.intlv %239, %239 : <f32>, <f32> -> <f32>, <f32>
                ascvec.storealign %arg108[%arg107], %low_56, %227 : memref<128xf32, 1>, index, <f32>
                ascvec.mem_bar vst_vld
                %240 = ascvec.loadalign %arg106[%arg107] : memref<128xf32, 1>, index -> <f32>
                ascvec.mem_bar vld_vst
                %241 = ascvec.loadalign %arg108[%arg107] : memref<128xf32, 1>, index -> <f32>
                ascvec.store_unalign_begin %arg106 : memref<128xf32, 1>
                scf.for %arg115 = %c0_i32_50 to %c32_i32_49 step %c1_i32_53  : i32 {
                  %242 = arith.muli %arg115, %c128_i32 overflow<nsw> : i32
                  %243 = arith.index_cast %242 : i32 to index
                  %result0, %result1 = ascvec.loadalign_deintlv %arg109[%243] {dintlv_width = "b32"} : memref<4096xf32, 1>, index -> <f32>, <f32>
                  %244 = ascvec.mul_scalar %result0, %arg110, %227 : <f32>, f32
                  %245 = ascvec.mul_scalar %result1, %arg110, %227 : <f32>, f32
                  %246 = ascvec.reduce_max %244, %227 : <f32>
                  %247 = ascvec.reduce_max %245, %227 : <f32>
                  %248 = ascvec.max %246, %247, %227 : <f32>
                  %249 = ascvec.duplicate %248, %227 : <f32> -> <f32>
                  %250 = arith.muli %arg115, %c2_i32_55 overflow<nsw> : i32
                  %251 = arith.extsi %250 : i32 to i64
                  %252 = arith.addi %arg111, %251 : i64
                  %253 = arith.index_cast %252 : i64 to index
                  %254 = ascvec.loadalign_brc %arg106[%253] : memref<128xf32, 1>, index -> <f32>
                  %255 = ascvec.max %249, %254, %227 : <f32>
                  %256 = ascvec.squeeze_store_reg %255, %mask : <f32>, !ascvec.mask -> <f32>
                  ascvec.store_unalign %arg106[%arg107], %256 : memref<128xf32, 1>, <f32>
                } {cannir.loop_unroll = 1 : i64}
                %c0_58 = arith.constant 0 : index
                %c1 = arith.constant 1 : index
                scf.for %arg115 = %c0_58 to %c1 step %c1 {
                  ascvec.store_unalign_post %arg106[%arg107] : memref<128xf32, 1>
                }
                ascvec.mem_bar vst_vld
                %c0_59 = arith.constant 0 : index
                %c1_60 = arith.constant 1 : index
                scf.for %arg115 = %c0_59 to %c1_60 step %c1_60 {
                  ascvec.store_unalign_begin %arg108 : memref<128xf32, 1>
                }
                scf.for %arg115 = %c0_i32_50 to %c32_i32_49 step %c1_i32_53  : i32 {
                  %242 = arith.muli %arg115, %c128_i32 overflow<nsw> : i32
                  %243 = arith.index_cast %242 : i32 to index
                  %result0, %result1 = ascvec.loadalign_deintlv %arg109[%243] {dintlv_width = "b32"} : memref<4096xf32, 1>, index -> <f32>, <f32>
                  %244 = ascvec.mul_scalar %result0, %arg110, %227 : <f32>, f32
                  %245 = ascvec.mul_scalar %result1, %arg110, %227 : <f32>, f32
                  %246 = arith.muli %arg115, %c2_i32_55 overflow<nsw> : i32
                  %247 = arith.extsi %246 : i32 to i64
                  %248 = arith.addi %arg111, %247 : i64
                  %249 = arith.index_cast %248 : i64 to index
                  %250 = ascvec.loadalign_brc %arg106[%249] : memref<128xf32, 1>, index -> <f32>
                  %251 = ascvec.exp_sub %244, %250, %227 : <f32>, <f32> -> <f32>
                  %252 = ascvec.exp_sub %245, %250, %227 : <f32>, <f32> -> <f32>
                  %253 = ascvec.reduce_sum %251, %227 : <f32>
                  %254 = ascvec.reduce_sum %252, %227 : <f32>
                  %255 = ascvec.add %253, %254, %227 : <f32>
                  %256 = ascvec.duplicate %255, %227 : <f32> -> <f32>
                  %257 = ascvec.squeeze_store_reg %256, %mask : <f32>, !ascvec.mask -> <f32>
                  ascvec.store_unalign %arg108[%arg107], %257 : memref<128xf32, 1>, <f32>
                  %258 = ascvec.cast %251, %227 {reg_layout = 0 : i32} : <f32> -> <bf16>
                  %259 = ascvec.cast %252, %227 {reg_layout = 1 : i32} : <f32> -> <bf16>
                  %260 = ascvec.add %258, %259, %228 : <bf16>
                  %261 = arith.muli %arg115, %c16_i32_48 overflow<nsw> : i32
                  %262 = arith.index_cast %261 : i32 to index
                  ascvec.storealign_strided %arg112[%262], %260, %228, %c33, %c0_54 : memref<4224xbf16, 1>, <bf16>
                } {cannir.loop_unroll = 1 : i64}
                %c0_61 = arith.constant 0 : index
                %c1_62 = arith.constant 1 : index
                scf.for %arg115 = %c0_61 to %c1_62 step %c1_62 {
                  ascvec.store_unalign_post %arg108[%arg107] : memref<128xf32, 1>
                }
                ascvec.mem_bar vst_vld
                %c0_63 = arith.constant 0 : index
                %c1_64 = arith.constant 1 : index
                scf.for %arg115 = %c0_63 to %c1_64 step %c1_64 {
                  %242 = ascvec.loadalign %arg106[%arg107] : memref<128xf32, 1>, index -> <f32>
                  %243 = ascvec.exp_sub %240, %242, %227 : <f32>, <f32> -> <f32>
                  ascvec.storealign %arg113[%arg114], %243, %227 : memref<256xf32, 1>, index, <f32>
                  %244 = ascvec.loadalign %arg108[%arg107] : memref<128xf32, 1>, index -> <f32>
                  %245 = ascvec.madd_two_address %241, %243, %244, %227 : <f32>
                  ascvec.storealign %arg108[%arg107], %245, %227 : memref<128xf32, 1>, index, <f32>
                }
              } {cannir.vf_mode = "simd"}
              ascvec.release_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
              ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
              ascvec.release_buf v, %c1_i64, 0 {cannir.buffer.generated_sync}
              ascvec.sync_intra_arrive v, %c0_i64 {cannir.sync.banks = 3 : i32}
            } else {
              %215 = cannir.datastruct_get %arg22["softmax_scale"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
              %216 = arith.remsi %140, %c2_i64 : i64
              %217 = arith.cmpi ne, %216, %c0_i64 : i64
              %218 = arith.cmpi slt, %216, %c0_i64 : i64
              %219 = arith.andi %217, %218 : i1
              %220 = arith.select %219, %c2_i64, %c0_i64 : i64
              %221 = arith.addi %216, %220 overflow<nsw> : i64
              %222 = arith.muli %221, %c64_i64 overflow<nsw> : i64
              %223 = arith.index_cast %222 : i64 to index
              %224 = arith.muli %221, %c128_i64 overflow<nsw> : i64
              %225 = arith.remsi %142, %c2_i64 : i64
              %226 = arith.muli %225, %c64_i64 overflow<nsw> : i64
              %227 = arith.addi %224, %226 overflow<nsw> : i64
              %228 = arith.index_cast %227 : i64 to index
              ascvec.sync_intra_wait v, %c0_i64 {cannir.sync.banks = 3 : i32}
              ascvec.get_buf v, %c1_i64, 0 {cannir.buffer.generated_sync}
              ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
              ascvec.get_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
              ascvec.vf(%alloc_19, %223, %alloc_20, %alloc_16, %215, %222, %alloc_17, %alloc_21, %228 : memref<128xf32, 1>, index, memref<128xf32, 1>, memref<4096xf32, 1>, f32, i64, memref<4224xbf16, 1>, memref<256xf32, 1>, index) {
              ^bb0(%arg104: memref<128xf32, 1>, %arg105: index, %arg106: memref<128xf32, 1>, %arg107: memref<4096xf32, 1>, %arg108: f32, %arg109: i64, %arg110: memref<4224xbf16, 1>, %arg111: memref<256xf32, 1>, %arg112: index):
                %c33 = arith.constant 33 : index
                %c16_i32_48 = arith.constant 16 : i32
                %c1_i32_49 = arith.constant 1 : i32
                %c32_i32_50 = arith.constant 32 : i32
                %c0_i32_51 = arith.constant 0 : i32
                %c128_i32 = arith.constant 128 : i32
                %c0_52 = arith.constant 0 : index
                %c2_i32_53 = arith.constant 2 : i32
                %229 = ascvec.create_mask : !ascvec.mask
                %230 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                %mask, %remaining_out = ascvec.update_mask %c2_i32_53 : !ascvec.mask
                %231 = ascvec.loadalign %arg104[%arg105] : memref<128xf32, 1>, index -> <f32>
                ascvec.mem_bar vld_vst
                %232 = ascvec.loadalign %arg106[%arg105] : memref<128xf32, 1>, index -> <f32>
                ascvec.store_unalign_begin %arg104 : memref<128xf32, 1>
                scf.for %arg113 = %c0_i32_51 to %c32_i32_50 step %c1_i32_49  : i32 {
                  %233 = arith.muli %arg113, %c128_i32 overflow<nsw> : i32
                  %234 = arith.index_cast %233 : i32 to index
                  %result0, %result1 = ascvec.loadalign_deintlv %arg107[%234] {dintlv_width = "b32"} : memref<4096xf32, 1>, index -> <f32>, <f32>
                  %235 = ascvec.mul_scalar %result0, %arg108, %229 : <f32>, f32
                  %236 = ascvec.mul_scalar %result1, %arg108, %229 : <f32>, f32
                  %237 = ascvec.reduce_max %235, %229 : <f32>
                  %238 = ascvec.reduce_max %236, %229 : <f32>
                  %239 = ascvec.max %237, %238, %229 : <f32>
                  %240 = ascvec.duplicate %239, %229 : <f32> -> <f32>
                  %241 = arith.muli %arg113, %c2_i32_53 overflow<nsw> : i32
                  %242 = arith.extsi %241 : i32 to i64
                  %243 = arith.addi %arg109, %242 : i64
                  %244 = arith.index_cast %243 : i64 to index
                  %245 = ascvec.loadalign_brc %arg104[%244] : memref<128xf32, 1>, index -> <f32>
                  %246 = ascvec.max %240, %245, %229 : <f32>
                  %247 = ascvec.squeeze_store_reg %246, %mask : <f32>, !ascvec.mask -> <f32>
                  ascvec.store_unalign %arg104[%arg105], %247 : memref<128xf32, 1>, <f32>
                } {cannir.loop_unroll = 1 : i64}
                %c0_54 = arith.constant 0 : index
                %c1 = arith.constant 1 : index
                scf.for %arg113 = %c0_54 to %c1 step %c1 {
                  ascvec.store_unalign_post %arg104[%arg105] : memref<128xf32, 1>
                }
                ascvec.mem_bar vst_vld
                %c0_55 = arith.constant 0 : index
                %c1_56 = arith.constant 1 : index
                scf.for %arg113 = %c0_55 to %c1_56 step %c1_56 {
                  ascvec.store_unalign_begin %arg106 : memref<128xf32, 1>
                }
                scf.for %arg113 = %c0_i32_51 to %c32_i32_50 step %c1_i32_49  : i32 {
                  %233 = arith.muli %arg113, %c128_i32 overflow<nsw> : i32
                  %234 = arith.index_cast %233 : i32 to index
                  %result0, %result1 = ascvec.loadalign_deintlv %arg107[%234] {dintlv_width = "b32"} : memref<4096xf32, 1>, index -> <f32>, <f32>
                  %235 = ascvec.mul_scalar %result0, %arg108, %229 : <f32>, f32
                  %236 = ascvec.mul_scalar %result1, %arg108, %229 : <f32>, f32
                  %237 = arith.muli %arg113, %c2_i32_53 overflow<nsw> : i32
                  %238 = arith.extsi %237 : i32 to i64
                  %239 = arith.addi %arg109, %238 : i64
                  %240 = arith.index_cast %239 : i64 to index
                  %241 = ascvec.loadalign_brc %arg104[%240] : memref<128xf32, 1>, index -> <f32>
                  %242 = ascvec.exp_sub %235, %241, %229 : <f32>, <f32> -> <f32>
                  %243 = ascvec.exp_sub %236, %241, %229 : <f32>, <f32> -> <f32>
                  %244 = ascvec.reduce_sum %242, %229 : <f32>
                  %245 = ascvec.reduce_sum %243, %229 : <f32>
                  %246 = ascvec.add %244, %245, %229 : <f32>
                  %247 = ascvec.duplicate %246, %229 : <f32> -> <f32>
                  %248 = ascvec.squeeze_store_reg %247, %mask : <f32>, !ascvec.mask -> <f32>
                  ascvec.store_unalign %arg106[%arg105], %248 : memref<128xf32, 1>, <f32>
                  %249 = ascvec.cast %242, %229 {reg_layout = 0 : i32} : <f32> -> <bf16>
                  %250 = ascvec.cast %243, %229 {reg_layout = 1 : i32} : <f32> -> <bf16>
                  %251 = ascvec.add %249, %250, %230 : <bf16>
                  %252 = arith.muli %arg113, %c16_i32_48 overflow<nsw> : i32
                  %253 = arith.index_cast %252 : i32 to index
                  ascvec.storealign_strided %arg110[%253], %251, %230, %c33, %c0_52 : memref<4224xbf16, 1>, <bf16>
                } {cannir.loop_unroll = 1 : i64}
                %c0_57 = arith.constant 0 : index
                %c1_58 = arith.constant 1 : index
                scf.for %arg113 = %c0_57 to %c1_58 step %c1_58 {
                  ascvec.store_unalign_post %arg106[%arg105] : memref<128xf32, 1>
                }
                ascvec.mem_bar vst_vld
                %c0_59 = arith.constant 0 : index
                %c1_60 = arith.constant 1 : index
                scf.for %arg113 = %c0_59 to %c1_60 step %c1_60 {
                  %233 = ascvec.loadalign %arg104[%arg105] : memref<128xf32, 1>, index -> <f32>
                  %234 = ascvec.exp_sub %231, %233, %229 : <f32>, <f32> -> <f32>
                  ascvec.storealign %arg111[%arg112], %234, %229 : memref<256xf32, 1>, index, <f32>
                  %235 = ascvec.loadalign %arg106[%arg105] : memref<128xf32, 1>, index -> <f32>
                  %236 = ascvec.madd_two_address %232, %234, %235, %229 : <f32>
                  ascvec.storealign %arg106[%arg105], %236, %229 : memref<128xf32, 1>, index, <f32>
                }
              } {cannir.vf_mode = "simd"}
              ascvec.release_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
              ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
              ascvec.release_buf v, %c1_i64, 0 {cannir.buffer.generated_sync}
              ascvec.sync_intra_arrive v, %c0_i64 {cannir.sync.banks = 3 : i32}
            }
          } else {
            %214 = arith.cmpi eq, %142, %c0_i64 : i64
            scf.if %214 {
              %215 = cannir.datastruct_get %arg22["softmax_scale"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
              %216 = arith.remsi %140, %c2_i64 : i64
              %217 = arith.cmpi ne, %216, %c0_i64 : i64
              %218 = arith.cmpi slt, %216, %c0_i64 : i64
              %219 = arith.andi %217, %218 : i1
              %220 = arith.select %219, %c2_i64, %c0_i64 : i64
              %221 = arith.addi %216, %220 overflow<nsw> : i64
              %222 = arith.muli %221, %c64_i64 overflow<nsw> : i64
              %223 = arith.extui %194 : i1 to i32
              %224 = arith.index_cast %222 : i64 to index
              %225 = arith.trunci %209 : i64 to i32
              %226 = arith.muli %221, %c128_i64 overflow<nsw> : i64
              %227 = arith.index_cast %226 : i64 to index
              ascvec.sync_intra_wait v, %c0_i64 {cannir.sync.banks = 3 : i32}
              ascvec.get_buf v, %c1_i64, 0 {cannir.buffer.generated_sync}
              ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
              ascvec.get_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
              ascvec.vf(%alloc_18, %223, %alloc_19, %224, %alloc_20, %225, %alloc_16, %215, %222, %alloc_17, %alloc_21, %227 : memref<32xf32, 1>, i32, memref<128xf32, 1>, index, memref<128xf32, 1>, i32, memref<4096xf32, 1>, f32, i64, memref<4224xbf16, 1>, memref<256xf32, 1>, index) {
              ^bb0(%arg104: memref<32xf32, 1>, %arg105: i32, %arg106: memref<128xf32, 1>, %arg107: index, %arg108: memref<128xf32, 1>, %arg109: i32, %arg110: memref<4096xf32, 1>, %arg111: f32, %arg112: i64, %arg113: memref<4224xbf16, 1>, %arg114: memref<256xf32, 1>, %arg115: index):
                %c33 = arith.constant 33 : index
                %c16_i32_48 = arith.constant 16 : i32
                %c32_i32_49 = arith.constant 32 : i32
                %c128_i32 = arith.constant 128 : i32
                %c0_i32_50 = arith.constant 0 : i32
                %cst = arith.constant 0.000000e+00 : f32
                %cst_51 = arith.constant 1.000000e+00 : f32
                %cst_52 = arith.constant -3.40282347E+38 : f32
                %c1_i32_53 = arith.constant 1 : i32
                %c-8388608_i32 = arith.constant -8388608 : i32
                %c0_54 = arith.constant 0 : index
                %c2_i32_55 = arith.constant 2 : i32
                %228 = ascvec.create_mask : !ascvec.mask
                %229 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                %mask, %remaining_out = ascvec.update_mask %c2_i32_55 : !ascvec.mask
                %230 = ascvec.loadalign %arg104[%c0_54] : memref<32xf32, 1>, index -> <f32>
                %231 = ascvec.duplicate_scalar_nomask %c-8388608_i32 : i32 -> <ui32>
                %232 = ascvec.reinterpret %231 : <ui32> -> <f32>
                %233 = ascvec.duplicate_scalar_nomask %arg105 : i32 -> <i32>
                %234 = ascvec.cmp_ge_scalar %233, %c1_i32_53, %228 : <i32>, i32 -> !ascvec.mask
                %235 = ascvec.select %232, %230, %234 : <f32>
                %236 = ascvec.reinterpret %235 : <f32> -> <ui32>
                %237 = ascvec.cmp_eq_scalar %236, %c-8388608_i32, %228 : <ui32>, i32 -> !ascvec.mask
                %238 = ascvec.duplicate_scalar %cst_52, %237 merge %235 : !ascvec.vreg<f32> {mode = "merging"} : f32 -> <f32>
                %low, %high = ascvec.intlv %238, %238 : <f32>, <f32> -> <f32>, <f32>
                ascvec.storealign %arg106[%arg107], %low, %228 : memref<128xf32, 1>, index, <f32>
                %239 = ascvec.duplicate_scalar_nomask %cst_51 : f32 -> <f32>
                %240 = ascvec.duplicate_scalar %cst, %237 merge %239 : !ascvec.vreg<f32> {mode = "merging"} : f32 -> <f32>
                %low_56, %high_57 = ascvec.intlv %240, %240 : <f32>, <f32> -> <f32>, <f32>
                ascvec.storealign %arg108[%arg107], %low_56, %228 : memref<128xf32, 1>, index, <f32>
                ascvec.mem_bar vst_vld
                %241 = ascvec.loadalign %arg106[%arg107] : memref<128xf32, 1>, index -> <f32>
                ascvec.mem_bar vld_vst
                %242 = ascvec.loadalign %arg108[%arg107] : memref<128xf32, 1>, index -> <f32>
                %243 = ascvec.arange %c0_i32_50 : i32 -> <i32>
                %244 = ascvec.mul_scalar %243, %c2_i32_55, %228 : <i32>, i32
                %245 = ascvec.add_scalar %244, %c1_i32_53, %228 : <i32>, i32
                %246 = ascvec.cmp_ge_scalar %244, %arg109, %228 : <i32>, i32 -> !ascvec.mask
                %247 = ascvec.cmp_ge_scalar %245, %arg109, %228 : <i32>, i32 -> !ascvec.mask
                ascvec.store_unalign_begin %arg106 : memref<128xf32, 1>
                scf.for %arg116 = %c0_i32_50 to %c32_i32_49 step %c1_i32_53  : i32 {
                  %248 = arith.muli %arg116, %c128_i32 overflow<nsw> : i32
                  %249 = arith.index_cast %248 : i32 to index
                  %result0, %result1 = ascvec.loadalign_deintlv %arg110[%249] {dintlv_width = "b32"} : memref<4096xf32, 1>, index -> <f32>, <f32>
                  %250 = ascvec.mul_scalar %result0, %arg111, %228 : <f32>, f32
                  %251 = ascvec.mul_scalar %result1, %arg111, %228 : <f32>, f32
                  %252 = ascvec.select %232, %250, %246 : <f32>
                  %253 = ascvec.select %232, %251, %247 : <f32>
                  %254 = ascvec.reduce_max %252, %228 : <f32>
                  %255 = ascvec.reduce_max %253, %228 : <f32>
                  %256 = ascvec.max %254, %255, %228 : <f32>
                  %257 = ascvec.duplicate %256, %228 : <f32> -> <f32>
                  %258 = arith.muli %arg116, %c2_i32_55 overflow<nsw> : i32
                  %259 = arith.extsi %258 : i32 to i64
                  %260 = arith.addi %arg112, %259 : i64
                  %261 = arith.index_cast %260 : i64 to index
                  %262 = ascvec.loadalign_brc %arg106[%261] : memref<128xf32, 1>, index -> <f32>
                  %263 = ascvec.max %257, %262, %228 : <f32>
                  %264 = ascvec.squeeze_store_reg %263, %mask : <f32>, !ascvec.mask -> <f32>
                  ascvec.store_unalign %arg106[%arg107], %264 : memref<128xf32, 1>, <f32>
                } {cannir.loop_unroll = 1 : i64}
                %c0_58 = arith.constant 0 : index
                %c1 = arith.constant 1 : index
                scf.for %arg116 = %c0_58 to %c1 step %c1 {
                  ascvec.store_unalign_post %arg106[%arg107] : memref<128xf32, 1>
                }
                ascvec.mem_bar vst_vld
                %c0_59 = arith.constant 0 : index
                %c1_60 = arith.constant 1 : index
                scf.for %arg116 = %c0_59 to %c1_60 step %c1_60 {
                  ascvec.store_unalign_begin %arg108 : memref<128xf32, 1>
                }
                scf.for %arg116 = %c0_i32_50 to %c32_i32_49 step %c1_i32_53  : i32 {
                  %248 = arith.muli %arg116, %c128_i32 overflow<nsw> : i32
                  %249 = arith.index_cast %248 : i32 to index
                  %result0, %result1 = ascvec.loadalign_deintlv %arg110[%249] {dintlv_width = "b32"} : memref<4096xf32, 1>, index -> <f32>, <f32>
                  %250 = ascvec.mul_scalar %result0, %arg111, %228 : <f32>, f32
                  %251 = ascvec.mul_scalar %result1, %arg111, %228 : <f32>, f32
                  %252 = ascvec.select %232, %250, %246 : <f32>
                  %253 = ascvec.select %232, %251, %247 : <f32>
                  %254 = arith.muli %arg116, %c2_i32_55 overflow<nsw> : i32
                  %255 = arith.extsi %254 : i32 to i64
                  %256 = arith.addi %arg112, %255 : i64
                  %257 = arith.index_cast %256 : i64 to index
                  %258 = ascvec.loadalign_brc %arg106[%257] : memref<128xf32, 1>, index -> <f32>
                  %259 = ascvec.exp_sub %252, %258, %228 : <f32>, <f32> -> <f32>
                  %260 = ascvec.exp_sub %253, %258, %228 : <f32>, <f32> -> <f32>
                  %261 = ascvec.reduce_sum %259, %228 : <f32>
                  %262 = ascvec.reduce_sum %260, %228 : <f32>
                  %263 = ascvec.add %261, %262, %228 : <f32>
                  %264 = ascvec.duplicate %263, %228 : <f32> -> <f32>
                  %265 = ascvec.squeeze_store_reg %264, %mask : <f32>, !ascvec.mask -> <f32>
                  ascvec.store_unalign %arg108[%arg107], %265 : memref<128xf32, 1>, <f32>
                  %266 = ascvec.cast %259, %228 {reg_layout = 0 : i32} : <f32> -> <bf16>
                  %267 = ascvec.cast %260, %228 {reg_layout = 1 : i32} : <f32> -> <bf16>
                  %268 = ascvec.add %266, %267, %229 : <bf16>
                  %269 = arith.muli %arg116, %c16_i32_48 overflow<nsw> : i32
                  %270 = arith.index_cast %269 : i32 to index
                  ascvec.storealign_strided %arg113[%270], %268, %229, %c33, %c0_54 : memref<4224xbf16, 1>, <bf16>
                } {cannir.loop_unroll = 1 : i64}
                %c0_61 = arith.constant 0 : index
                %c1_62 = arith.constant 1 : index
                scf.for %arg116 = %c0_61 to %c1_62 step %c1_62 {
                  ascvec.store_unalign_post %arg108[%arg107] : memref<128xf32, 1>
                }
                ascvec.mem_bar vst_vld
                %c0_63 = arith.constant 0 : index
                %c1_64 = arith.constant 1 : index
                scf.for %arg116 = %c0_63 to %c1_64 step %c1_64 {
                  %248 = ascvec.loadalign %arg106[%arg107] : memref<128xf32, 1>, index -> <f32>
                  %249 = ascvec.exp_sub %241, %248, %228 : <f32>, <f32> -> <f32>
                  ascvec.storealign %arg114[%arg115], %249, %228 : memref<256xf32, 1>, index, <f32>
                  %250 = ascvec.loadalign %arg108[%arg107] : memref<128xf32, 1>, index -> <f32>
                  %251 = ascvec.madd_two_address %242, %249, %250, %228 : <f32>
                  ascvec.storealign %arg108[%arg107], %251, %228 : memref<128xf32, 1>, index, <f32>
                }
              } {cannir.vf_mode = "simd"}
              ascvec.release_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
              ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
              ascvec.release_buf v, %c1_i64, 0 {cannir.buffer.generated_sync}
              ascvec.sync_intra_arrive v, %c0_i64 {cannir.sync.banks = 3 : i32}
            } else {
              %215 = cannir.datastruct_get %arg22["softmax_scale"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
              %216 = arith.remsi %140, %c2_i64 : i64
              %217 = arith.cmpi ne, %216, %c0_i64 : i64
              %218 = arith.cmpi slt, %216, %c0_i64 : i64
              %219 = arith.andi %217, %218 : i1
              %220 = arith.select %219, %c2_i64, %c0_i64 : i64
              %221 = arith.addi %216, %220 overflow<nsw> : i64
              %222 = arith.muli %221, %c64_i64 overflow<nsw> : i64
              %223 = arith.index_cast %222 : i64 to index
              %224 = arith.trunci %209 : i64 to i32
              %225 = arith.muli %221, %c128_i64 overflow<nsw> : i64
              %226 = arith.remsi %142, %c2_i64 : i64
              %227 = arith.muli %226, %c64_i64 overflow<nsw> : i64
              %228 = arith.addi %225, %227 overflow<nsw> : i64
              %229 = arith.index_cast %228 : i64 to index
              ascvec.sync_intra_wait v, %c0_i64 {cannir.sync.banks = 3 : i32}
              ascvec.get_buf v, %c1_i64, 0 {cannir.buffer.generated_sync}
              ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
              ascvec.get_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
              ascvec.vf(%alloc_19, %223, %alloc_20, %224, %alloc_16, %215, %222, %alloc_17, %alloc_21, %229 : memref<128xf32, 1>, index, memref<128xf32, 1>, i32, memref<4096xf32, 1>, f32, i64, memref<4224xbf16, 1>, memref<256xf32, 1>, index) {
              ^bb0(%arg104: memref<128xf32, 1>, %arg105: index, %arg106: memref<128xf32, 1>, %arg107: i32, %arg108: memref<4096xf32, 1>, %arg109: f32, %arg110: i64, %arg111: memref<4224xbf16, 1>, %arg112: memref<256xf32, 1>, %arg113: index):
                %c33 = arith.constant 33 : index
                %c16_i32_48 = arith.constant 16 : i32
                %c32_i32_49 = arith.constant 32 : i32
                %c128_i32 = arith.constant 128 : i32
                %c1_i32_50 = arith.constant 1 : i32
                %c0_i32_51 = arith.constant 0 : i32
                %c-8388608_i32 = arith.constant -8388608 : i32
                %c0_52 = arith.constant 0 : index
                %c2_i32_53 = arith.constant 2 : i32
                %230 = ascvec.create_mask : !ascvec.mask
                %231 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                %mask, %remaining_out = ascvec.update_mask %c2_i32_53 : !ascvec.mask
                %232 = ascvec.loadalign %arg104[%arg105] : memref<128xf32, 1>, index -> <f32>
                ascvec.mem_bar vld_vst
                %233 = ascvec.loadalign %arg106[%arg105] : memref<128xf32, 1>, index -> <f32>
                %234 = ascvec.duplicate_scalar_nomask %c-8388608_i32 : i32 -> <ui32>
                %235 = ascvec.reinterpret %234 : <ui32> -> <f32>
                %236 = ascvec.arange %c0_i32_51 : i32 -> <i32>
                %237 = ascvec.mul_scalar %236, %c2_i32_53, %230 : <i32>, i32
                %238 = ascvec.add_scalar %237, %c1_i32_50, %230 : <i32>, i32
                %239 = ascvec.cmp_ge_scalar %237, %arg107, %230 : <i32>, i32 -> !ascvec.mask
                %240 = ascvec.cmp_ge_scalar %238, %arg107, %230 : <i32>, i32 -> !ascvec.mask
                ascvec.store_unalign_begin %arg104 : memref<128xf32, 1>
                scf.for %arg114 = %c0_i32_51 to %c32_i32_49 step %c1_i32_50  : i32 {
                  %241 = arith.muli %arg114, %c128_i32 overflow<nsw> : i32
                  %242 = arith.index_cast %241 : i32 to index
                  %result0, %result1 = ascvec.loadalign_deintlv %arg108[%242] {dintlv_width = "b32"} : memref<4096xf32, 1>, index -> <f32>, <f32>
                  %243 = ascvec.mul_scalar %result0, %arg109, %230 : <f32>, f32
                  %244 = ascvec.mul_scalar %result1, %arg109, %230 : <f32>, f32
                  %245 = ascvec.select %235, %243, %239 : <f32>
                  %246 = ascvec.select %235, %244, %240 : <f32>
                  %247 = ascvec.reduce_max %245, %230 : <f32>
                  %248 = ascvec.reduce_max %246, %230 : <f32>
                  %249 = ascvec.max %247, %248, %230 : <f32>
                  %250 = ascvec.duplicate %249, %230 : <f32> -> <f32>
                  %251 = arith.muli %arg114, %c2_i32_53 overflow<nsw> : i32
                  %252 = arith.extsi %251 : i32 to i64
                  %253 = arith.addi %arg110, %252 : i64
                  %254 = arith.index_cast %253 : i64 to index
                  %255 = ascvec.loadalign_brc %arg104[%254] : memref<128xf32, 1>, index -> <f32>
                  %256 = ascvec.max %250, %255, %230 : <f32>
                  %257 = ascvec.squeeze_store_reg %256, %mask : <f32>, !ascvec.mask -> <f32>
                  ascvec.store_unalign %arg104[%arg105], %257 : memref<128xf32, 1>, <f32>
                } {cannir.loop_unroll = 1 : i64}
                %c0_54 = arith.constant 0 : index
                %c1 = arith.constant 1 : index
                scf.for %arg114 = %c0_54 to %c1 step %c1 {
                  ascvec.store_unalign_post %arg104[%arg105] : memref<128xf32, 1>
                }
                ascvec.mem_bar vst_vld
                %c0_55 = arith.constant 0 : index
                %c1_56 = arith.constant 1 : index
                scf.for %arg114 = %c0_55 to %c1_56 step %c1_56 {
                  ascvec.store_unalign_begin %arg106 : memref<128xf32, 1>
                }
                scf.for %arg114 = %c0_i32_51 to %c32_i32_49 step %c1_i32_50  : i32 {
                  %241 = arith.muli %arg114, %c128_i32 overflow<nsw> : i32
                  %242 = arith.index_cast %241 : i32 to index
                  %result0, %result1 = ascvec.loadalign_deintlv %arg108[%242] {dintlv_width = "b32"} : memref<4096xf32, 1>, index -> <f32>, <f32>
                  %243 = ascvec.mul_scalar %result0, %arg109, %230 : <f32>, f32
                  %244 = ascvec.mul_scalar %result1, %arg109, %230 : <f32>, f32
                  %245 = ascvec.select %235, %243, %239 : <f32>
                  %246 = ascvec.select %235, %244, %240 : <f32>
                  %247 = arith.muli %arg114, %c2_i32_53 overflow<nsw> : i32
                  %248 = arith.extsi %247 : i32 to i64
                  %249 = arith.addi %arg110, %248 : i64
                  %250 = arith.index_cast %249 : i64 to index
                  %251 = ascvec.loadalign_brc %arg104[%250] : memref<128xf32, 1>, index -> <f32>
                  %252 = ascvec.exp_sub %245, %251, %230 : <f32>, <f32> -> <f32>
                  %253 = ascvec.exp_sub %246, %251, %230 : <f32>, <f32> -> <f32>
                  %254 = ascvec.reduce_sum %252, %230 : <f32>
                  %255 = ascvec.reduce_sum %253, %230 : <f32>
                  %256 = ascvec.add %254, %255, %230 : <f32>
                  %257 = ascvec.duplicate %256, %230 : <f32> -> <f32>
                  %258 = ascvec.squeeze_store_reg %257, %mask : <f32>, !ascvec.mask -> <f32>
                  ascvec.store_unalign %arg106[%arg105], %258 : memref<128xf32, 1>, <f32>
                  %259 = ascvec.cast %252, %230 {reg_layout = 0 : i32} : <f32> -> <bf16>
                  %260 = ascvec.cast %253, %230 {reg_layout = 1 : i32} : <f32> -> <bf16>
                  %261 = ascvec.add %259, %260, %231 : <bf16>
                  %262 = arith.muli %arg114, %c16_i32_48 overflow<nsw> : i32
                  %263 = arith.index_cast %262 : i32 to index
                  ascvec.storealign_strided %arg111[%263], %261, %231, %c33, %c0_52 : memref<4224xbf16, 1>, <bf16>
                } {cannir.loop_unroll = 1 : i64}
                %c0_57 = arith.constant 0 : index
                %c1_58 = arith.constant 1 : index
                scf.for %arg114 = %c0_57 to %c1_58 step %c1_58 {
                  ascvec.store_unalign_post %arg106[%arg105] : memref<128xf32, 1>
                }
                ascvec.mem_bar vst_vld
                %c0_59 = arith.constant 0 : index
                %c1_60 = arith.constant 1 : index
                scf.for %arg114 = %c0_59 to %c1_60 step %c1_60 {
                  %241 = ascvec.loadalign %arg104[%arg105] : memref<128xf32, 1>, index -> <f32>
                  %242 = ascvec.exp_sub %232, %241, %230 : <f32>, <f32> -> <f32>
                  ascvec.storealign %arg112[%arg113], %242, %230 : memref<256xf32, 1>, index, <f32>
                  %243 = ascvec.loadalign %arg106[%arg105] : memref<128xf32, 1>, index -> <f32>
                  %244 = ascvec.madd_two_address %233, %242, %243, %230 : <f32>
                  ascvec.storealign %arg106[%arg105], %244, %230 : memref<128xf32, 1>, index, <f32>
                }
              } {cannir.vf_mode = "simd"}
              ascvec.release_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
              ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
              ascvec.release_buf v, %c1_i64, 0 {cannir.buffer.generated_sync}
              ascvec.sync_intra_arrive v, %c0_i64 {cannir.sync.banks = 3 : i32}
            }
          }
          %211 = arith.cmpi eq, %51, %c0_i64 : i64
          scf.if %211 {
            %214 = ascscalar.bitcast_memref %alloc_17 byte_offset = %c0_i64 : memref<4224xbf16, 1> -> memref<?xbf16, 1>
            %215 = ascscalar.bitcast_memref %alloc_47 byte_offset = %c0_i64 : memref<8192xbf16, 2> -> memref<?xbf16, 2>
            ascvec.sync_intra_wait mte3, %c6_i64 {cannir.sync.banks = 3 : i32}
            ascvec.get_buf mte3, %c0_i64, 0 {cannir.buffer.generated_sync}
            ascvec.copy_ub2l1_strided %215[%c0], %214[%c0], %c8_i32, %c32_i32, %c1_i32, %c32_i32 : memref<?xbf16, 2>, memref<?xbf16, 1>
            ascvec.release_buf mte3, %c0_i64, 0 {cannir.buffer.generated_sync}
            ascvec.sync_intra_arrive mte3, %c6_i64 {cannir.sync.banks = 3 : i32}
          } else {
            %214 = ascscalar.bitcast_memref %alloc_17 byte_offset = %c0_i64 : memref<4224xbf16, 1> -> memref<?xbf16, 1>
            %215 = ascscalar.bitcast_memref %alloc_47 byte_offset = %c0_i64 : memref<8192xbf16, 2> -> memref<?xbf16, 2>
            ascvec.sync_intra_wait mte3, %c6_i64 {cannir.sync.banks = 3 : i32}
            ascvec.get_buf mte3, %c0_i64, 0 {cannir.buffer.generated_sync}
            ascvec.copy_ub2l1_strided %215[%c512], %214[%c0], %c8_i32, %c32_i32, %c1_i32, %c32_i32 : memref<?xbf16, 2>, memref<?xbf16, 1>
            ascvec.release_buf mte3, %c0_i64, 0 {cannir.buffer.generated_sync}
            ascvec.sync_intra_arrive mte3, %c6_i64 {cannir.sync.banks = 3 : i32}
          }
          %212 = arith.addi %142, %c1_i64 : i64
          %213 = arith.cmpi eq, %212, %144 : i64
          scf.if %213 {
            %214 = arith.cmpi sge, %201, %c0_i64 : i64
            scf.if %214 {
              %216 = arith.remsi %140, %c2_i64 : i64
              %217 = arith.cmpi ne, %216, %c0_i64 : i64
              %218 = arith.cmpi slt, %216, %c0_i64 : i64
              %219 = arith.andi %217, %218 : i1
              %220 = arith.select %219, %c2_i64, %c0_i64 : i64
              %221 = arith.addi %216, %220 overflow<nsw> : i64
              %222 = arith.muli %221, %c64_i64 overflow<nsw> : i64
              %223 = arith.index_cast %222 : i64 to index
              ascvec.get_buf v, %c1_i64, 0 {cannir.buffer.generated_sync}
              ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
              ascvec.vf(%alloc_19, %223, %alloc_20 : memref<128xf32, 1>, index, memref<128xf32, 1>) {
              ^bb0(%arg104: memref<128xf32, 1>, %arg105: index, %arg106: memref<128xf32, 1>):
                %c2139095040_i32 = arith.constant 2139095040 : i32
                %cst = arith.constant -3.40282347E+38 : f32
                %c0_48 = arith.constant 0 : index
                %c1 = arith.constant 1 : index
                scf.for %arg107 = %c0_48 to %c1 step %c1 {
                  %227 = ascvec.create_mask : !ascvec.mask
                  %228 = ascvec.loadalign %arg104[%arg105] : memref<128xf32, 1>, index -> <f32>
                  %result0, %result1 = ascvec.deintlv %228, %228 : <f32>, <f32> -> <f32>, <f32>
                  %229 = ascvec.loadalign %arg106[%arg105] : memref<128xf32, 1>, index -> <f32>
                  %result0_49, %result1_50 = ascvec.deintlv %229, %229 : <f32>, <f32> -> <f32>, <f32>
                  %230 = ascvec.cmp_eq_scalar %result0, %cst, %227 : <f32>, f32 -> !ascvec.mask
                  %231 = ascvec.duplicate_scalar_nomask %c2139095040_i32 : i32 -> <ui32>
                  %232 = ascvec.reinterpret %231 : <ui32> -> <f32>
                  ascvec.storealign %arg104[%arg105], %result0, %227 : memref<128xf32, 1>, index, <f32>
                  %233 = ascvec.select %232, %result0_49, %230 : <f32>
                  ascvec.storealign %arg106[%arg105], %233, %227 : memref<128xf32, 1>, index, <f32>
                }
              } {cannir.vf_mode = "simd"}
              ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
              ascvec.release_buf v, %c1_i64, 0 {cannir.buffer.generated_sync}
              %224 = cannir.datastruct_get %arg22["num_heads_q"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
              %225 = cannir.datastruct_get %arg22["cube_core_count"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
              %226 = arith.cmpi sgt, %59, %c0_i64 : i64
              scf.if %226 {
                %227 = arith.muli %225, %c2_i32 : i32
                %228 = arith.muli %227, %224 : i32
                %229 = arith.muli %228, %c512_i32 : i32
                %230 = arith.muli %201, %c128_i64 : i64
                %231 = arith.extsi %229 : i32 to i64
                %232 = arith.addi %231, %230 : i64
                %233 = arith.addi %232, %52 : i64
                %234 = arith.muli %233, %c32_i64 : i64
                %235 = ascscalar.memref_address %alloc_15 : memref<1xf32>
                %236 = arith.divui %234, %c8_i64 : i64
                %237 = arith.addi %235, %236 : i64
                %238 = ascscalar.bind_gm_ptr %237 : memref<1xf32>
                %239 = arith.muli %221, %c2048_i64 overflow<nsw> : i64
                %240 = ascscalar.memref_address %alloc_19 : memref<128xf32, 1>
                %241 = arith.divui %239, %c8_i64 : i64
                %242 = arith.addi %240, %241 : i64
                %243 = ascscalar.bind_local_ptr %242 : memref<1xf32, 1>
                %244 = arith.index_cast %59 : i64 to index
                %245 = arith.muli %244, %c4 : index
                %246 = arith.cmpi ugt, %59, %c0_i64 : i64
                scf.if %246 {
                  %256 = arith.index_cast %245 : index to i64
                  ascvec.get_buf mte3, %c1_i64, 0 {cannir.buffer.generated_sync}
                  %257 = arith.trunci %256 : i64 to i32
                  ascvec.copy_ub2gm %238[%c0], %243[%c0], %c1_i32, %257, %c0_i64, %257 l2_cache_ctl %c0_i32 : memref<1xf32>, memref<1xf32, 1>
                  ascvec.release_buf mte3, %c1_i64, 0 {cannir.buffer.generated_sync}
                }
                %247 = arith.addi %232, %c64_i64 : i64
                %248 = arith.addi %247, %52 : i64
                %249 = arith.muli %248, %c32_i64 : i64
                %250 = arith.divui %249, %c8_i64 : i64
                %251 = arith.addi %235, %250 : i64
                %252 = ascscalar.bind_gm_ptr %251 : memref<1xf32>
                %253 = ascscalar.memref_address %alloc_20 : memref<128xf32, 1>
                %254 = arith.addi %253, %241 : i64
                %255 = ascscalar.bind_local_ptr %254 : memref<1xf32, 1>
                scf.if %246 {
                  %256 = arith.index_cast %245 : index to i64
                  ascvec.get_buf mte3, %c2_i64, 0 {cannir.buffer.generated_sync}
                  %257 = arith.trunci %256 : i64 to i32
                  ascvec.copy_ub2gm %252[%c0], %255[%c0], %c1_i32, %257, %c0_i64, %257 l2_cache_ctl %c0_i32 : memref<1xf32>, memref<1xf32, 1>
                  ascvec.release_buf mte3, %c2_i64, 0 {cannir.buffer.generated_sync}
                }
              }
            } else {
              %216 = arith.remsi %140, %c2_i64 : i64
              %217 = arith.cmpi ne, %216, %c0_i64 : i64
              %218 = arith.cmpi slt, %216, %c0_i64 : i64
              %219 = arith.andi %217, %218 : i1
              %220 = arith.select %219, %c2_i64, %c0_i64 : i64
              %221 = arith.addi %216, %220 overflow<nsw> : i64
              %222 = arith.muli %221, %c64_i64 overflow<nsw> : i64
              %223 = arith.index_cast %222 : i64 to index
              ascvec.get_buf v, %c1_i64, 0 {cannir.buffer.generated_sync}
              ascvec.vf(%alloc_19, %223, %alloc_20 : memref<128xf32, 1>, index, memref<128xf32, 1>) {
              ^bb0(%arg104: memref<128xf32, 1>, %arg105: index, %arg106: memref<128xf32, 1>):
                %c2139095040_i32 = arith.constant 2139095040 : i32
                %cst = arith.constant -3.40282347E+38 : f32
                %c0_48 = arith.constant 0 : index
                %c1 = arith.constant 1 : index
                scf.for %arg107 = %c0_48 to %c1 step %c1 {
                  %224 = ascvec.create_mask : !ascvec.mask
                  %225 = ascvec.loadalign %arg104[%arg105] : memref<128xf32, 1>, index -> <f32>
                  %result0, %result1 = ascvec.deintlv %225, %225 : <f32>, <f32> -> <f32>, <f32>
                  %226 = ascvec.loadalign %arg106[%arg105] : memref<128xf32, 1>, index -> <f32>
                  %result0_49, %result1_50 = ascvec.deintlv %226, %226 : <f32>, <f32> -> <f32>, <f32>
                  %227 = ascvec.cmp_eq_scalar %result0, %cst, %224 : <f32>, f32 -> !ascvec.mask
                  %228 = ascvec.duplicate_scalar_nomask %c2139095040_i32 : i32 -> <ui32>
                  %229 = ascvec.reinterpret %228 : <ui32> -> <f32>
                  %230 = ascvec.ln %result0_49, %224 : <f32>
                  %231 = ascvec.add %230, %result0, %224 : <f32>
                  %232 = ascvec.select %229, %231, %227 : <f32>
                  ascvec.storealign %arg104[%arg105], %232, %224 : memref<128xf32, 1>, index, <f32>
                  %233 = ascvec.select %229, %result0_49, %227 : <f32>
                  ascvec.storealign %arg106[%arg105], %233, %224 : memref<128xf32, 1>, index, <f32>
                }
              } {cannir.vf_mode = "simd"}
              ascvec.release_buf v, %c1_i64, 0 {cannir.buffer.generated_sync}
            }
            %215 = arith.cmpi slt, %201, %c0_i64 : i64
            scf.if %215 {
              %216 = cannir.datastruct_get %arg22["num_heads_q"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
              %217 = arith.extsi %216 : i32 to i64
              %218 = arith.muli %141, %217 : i64
              %219 = arith.addi %218, %52 : i64
              %220 = arith.remsi %140, %c2_i64 : i64
              %221 = arith.cmpi ne, %220, %c0_i64 : i64
              %222 = arith.cmpi slt, %220, %c0_i64 : i64
              %223 = arith.andi %221, %222 : i1
              %224 = arith.select %223, %c2_i64, %c0_i64 : i64
              %225 = arith.addi %220, %224 overflow<nsw> : i64
              %226 = arith.muli %219, %c32_i64 : i64
              %227 = ascscalar.memref_address %alloc_14 : memref<1xf32>
              %228 = arith.divui %226, %c8_i64 : i64
              %229 = arith.addi %227, %228 : i64
              %230 = ascscalar.bind_gm_ptr %229 : memref<1xf32>
              %231 = arith.muli %225, %c2048_i64 overflow<nsw> : i64
              %232 = ascscalar.memref_address %alloc_19 : memref<128xf32, 1>
              %233 = arith.divui %231, %c8_i64 : i64
              %234 = arith.addi %232, %233 : i64
              %235 = ascscalar.bind_local_ptr %234 : memref<1xf32, 1>
              %236 = arith.index_cast %59 : i64 to index
              %237 = arith.muli %236, %c4 : index
              %238 = arith.cmpi ugt, %59, %c0_i64 : i64
              scf.if %238 {
                %239 = arith.index_cast %237 : index to i64
                ascvec.get_buf mte3, %c1_i64, 0 {cannir.buffer.generated_sync}
                %240 = arith.trunci %239 : i64 to i32
                ascvec.copy_ub2gm %230[%c0], %235[%c0], %c1_i32, %240, %c0_i64, %240 l2_cache_ctl %c0_i32 : memref<1xf32>, memref<1xf32, 1>
                ascvec.release_buf mte3, %c1_i64, 0 {cannir.buffer.generated_sync}
              }
            }
          }
        }
        scf.if %160 {
          %192 = arith.cmpi ne, %136, %c0_i64 : i64
          scf.if %192 {
            %193 = arith.cmpi slt, %132, %135 : i64
            %194 = arith.select %193, %132, %135 : i64
            %195 = arith.muli %133, %c128_i64 : i64
            %196 = arith.subi %194, %195 : i64
            %197 = arith.muli %51, %c64_i64 : i64
            %198 = arith.subi %196, %197 : i64
            %199 = arith.cmpi sgt, %198, %c0_i64 : i64
            %200 = arith.cmpi slt, %198, %c64_i64 : i64
            %201 = arith.select %200, %198, %c64_i64 : i64
            %202 = arith.select %199, %201, %c0_i64 : i64
            %203 = arith.cmpi sge, %202, %c48_i64 : i64
            scf.if %203 {
              scf.for %arg104 = %c0_i64 to %c8_i64 step %c1_i64  : i64 {
                %206 = arith.muli %arg104, %c2_i64 overflow<nsw> : i64
                %207 = arith.addi %206, %c32_i64 overflow<nsw> : i64
                ascvec.get_buf s, %109, 0 {cannir.buffer.generated_sync}
                %208 = arith.index_cast %arg90 : i64 to index
                %209 = arith.index_cast %207 : i64 to index
                %210 = arith.muli %209, %208 : index
                %211 = memref.load %112[%210] : memref<1xi32, 1>
                %212 = arith.extui %211 : i32 to i64
                %213 = arith.addi %206, %c33_i64 overflow<nsw> : i64
                %214 = arith.index_cast %213 : i64 to index
                %215 = arith.muli %214, %208 : index
                %216 = memref.load %112[%215] : memref<1xi32, 1>
                ascvec.release_buf s, %109, 0 {cannir.buffer.generated_sync}
                %217 = arith.extui %216 : i32 to i64
                %218 = arith.cmpi slt, %212, %217 : i64
                %219 = arith.select %218, %212, %217 : i64
                %220 = arith.select %218, %217, %212 : i64
                %221 = arith.muli %219, %c1024_i64 : i64
                %222 = arith.muli %220, %c1024_i64 : i64
                %223 = arith.subi %222, %221 : i64
                %224 = arith.muli %arg104, %c2048_i64 overflow<nsw> : i64
                ascvec.get_buf mte2, %c7_i64, 0 {cannir.buffer.generated_sync}
                "ascvec.copy_gm2ub_plan"(%alloc_26, %alloc_0, %c1_i32, %c0_i32, %224, %221, %c2_i64, %c1024_i64, %223, %c1024_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = 190976 : i64, ub_capacity = 16384 : i64}> : (memref<8192xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                ascvec.release_buf mte2, %c7_i64, 0 {cannir.buffer.generated_sync}
              }
            } else {
              scf.for %arg104 = %c0_i64 to %c8_i64 step %c1_i64  : i64 {
                %206 = arith.muli %arg104, %c2_i64 overflow<nsw> : i64
                %207 = arith.addi %206, %c32_i64 overflow<nsw> : i64
                %208 = arith.cmpi slt, %207, %202 : i64
                %209 = scf.if %208 -> (i32) {
                  ascvec.get_buf s, %109, 0 {cannir.buffer.generated_sync}
                  %216 = arith.index_cast %arg90 : i64 to index
                  %217 = arith.index_cast %207 : i64 to index
                  %218 = arith.muli %217, %216 : index
                  %219 = memref.load %112[%218] : memref<1xi32, 1>
                  ascvec.release_buf s, %109, 0 {cannir.buffer.generated_sync}
                  scf.yield %219 : i32
                } else {
                  scf.yield %c-1_i32 : i32
                }
                %210 = arith.addi %206, %c33_i64 overflow<nsw> : i64
                %211 = arith.cmpi slt, %210, %202 : i64
                %212 = scf.if %211 -> (i32) {
                  ascvec.get_buf s, %109, 0 {cannir.buffer.generated_sync}
                  %216 = arith.index_cast %arg90 : i64 to index
                  %217 = arith.index_cast %210 : i64 to index
                  %218 = arith.muli %217, %216 : index
                  %219 = memref.load %112[%218] : memref<1xi32, 1>
                  ascvec.release_buf s, %109, 0 {cannir.buffer.generated_sync}
                  scf.yield %219 : i32
                } else {
                  scf.yield %c-1_i32 : i32
                }
                %213 = arith.cmpi sge, %209, %c0_i32 : i32
                %214 = arith.cmpi sge, %212, %c0_i32 : i32
                %215 = arith.select %213, %214, %false : i1
                scf.if %215 {
                  %216 = arith.extui %209 : i32 to i64
                  %217 = arith.extui %212 : i32 to i64
                  %218 = arith.cmpi slt, %216, %217 : i64
                  %219 = arith.select %218, %216, %217 : i64
                  %220 = arith.select %218, %217, %216 : i64
                  %221 = arith.muli %219, %c1024_i64 : i64
                  %222 = arith.muli %220, %c1024_i64 : i64
                  %223 = arith.subi %222, %221 : i64
                  %224 = arith.muli %arg104, %c2048_i64 overflow<nsw> : i64
                  ascvec.get_buf mte2, %c7_i64, 0 {cannir.buffer.generated_sync}
                  "ascvec.copy_gm2ub_plan"(%alloc_26, %alloc_0, %c1_i32, %c0_i32, %224, %221, %c2_i64, %c1024_i64, %223, %c1024_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = 190976 : i64, ub_capacity = 16384 : i64}> : (memref<8192xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                  ascvec.release_buf mte2, %c7_i64, 0 {cannir.buffer.generated_sync}
                } else {
                  %216 = arith.select %213, %true, %214 : i1
                  scf.if %216 {
                    %217 = arith.extsi %209 : i32 to i64
                    %218 = arith.extsi %212 : i32 to i64
                    %219 = arith.select %213, %217, %218 : i64
                    %220 = cannir.extract_layout %arg2 : !cannir.tensor<<gm, bf16>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>
                    %221 = cannir.extract_shape %220 : <<(-1)>, <(1)>> -> <(-1)>
                    %222 = cannir.extract_dim %221[0] : !cannir.shape<(-1)>
                    %223 = arith.divui %222, %c512_i64 : i64
                    %224 = arith.minsi %219, %223 : i64
                    %225 = arith.subi %223, %224 : i64
                    %226 = arith.minsi %225, %c1_i64 : i64
                    %227 = arith.cmpi sle, %219, %223 : i64
                    %228 = arith.select %227, %226, %c0_i64 : i64
                    %229 = arith.muli %219, %c1024_i64 : i64
                    %230 = arith.muli %arg104, %c2048_i64 overflow<nsw> : i64
                    %231 = arith.muli %228, %c1024_i64 : i64
                    %232 = arith.cmpi eq, %231, %c0_i64 : i64
                    %233 = arith.cmpi ne, %231, %c0_i64 : i64
                    %234 = arith.extui %233 : i1 to i64
                    %235 = arith.select %232, %c2_i64, %231 : i64
                    ascvec.get_buf mte2, %c7_i64, 0 {cannir.buffer.generated_sync}
                    "ascvec.copy_gm2ub_plan"(%alloc_26, %alloc_0, %c1_i32, %c0_i32, %230, %229, %234, %235, %c0_i64, %235, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = 190976 : i64, ub_capacity = 16384 : i64}> : (memref<8192xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                    ascvec.release_buf mte2, %c7_i64, 0 {cannir.buffer.generated_sync}
                  }
                }
              }
            }
            %204 = arith.cmpi sge, %202, %c64_i64 : i64
            scf.if %204 {
              scf.for %arg104 = %c0_i64 to %c8_i64 step %c1_i64  : i64 {
                %206 = arith.muli %arg104, %c2_i64 overflow<nsw> : i64
                %207 = arith.addi %206, %c48_i64 overflow<nsw> : i64
                ascvec.get_buf s, %109, 0 {cannir.buffer.generated_sync}
                %208 = arith.index_cast %arg90 : i64 to index
                %209 = arith.index_cast %207 : i64 to index
                %210 = arith.muli %209, %208 : index
                %211 = memref.load %112[%210] : memref<1xi32, 1>
                %212 = arith.extui %211 : i32 to i64
                %213 = arith.addi %206, %c49_i64 overflow<nsw> : i64
                %214 = arith.index_cast %213 : i64 to index
                %215 = arith.muli %214, %208 : index
                %216 = memref.load %112[%215] : memref<1xi32, 1>
                ascvec.release_buf s, %109, 0 {cannir.buffer.generated_sync}
                %217 = arith.extui %216 : i32 to i64
                %218 = arith.cmpi slt, %212, %217 : i64
                %219 = arith.select %218, %212, %217 : i64
                %220 = arith.select %218, %217, %212 : i64
                %221 = arith.muli %219, %c1024_i64 : i64
                %222 = arith.muli %220, %c1024_i64 : i64
                %223 = arith.subi %222, %221 : i64
                %224 = arith.muli %arg104, %c2048_i64 overflow<nsw> : i64
                ascvec.get_buf mte2, %c8_i64, 0 {cannir.buffer.generated_sync}
                "ascvec.copy_gm2ub_plan"(%alloc_27, %alloc_0, %c1_i32, %c0_i32, %224, %221, %c2_i64, %c1024_i64, %223, %c1024_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = 207360 : i64, ub_capacity = 16384 : i64}> : (memref<8192xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                ascvec.release_buf mte2, %c8_i64, 0 {cannir.buffer.generated_sync}
              }
            } else {
              scf.for %arg104 = %c0_i64 to %c8_i64 step %c1_i64  : i64 {
                %206 = arith.muli %arg104, %c2_i64 overflow<nsw> : i64
                %207 = arith.addi %206, %c48_i64 overflow<nsw> : i64
                %208 = arith.cmpi slt, %207, %202 : i64
                %209 = scf.if %208 -> (i32) {
                  ascvec.get_buf s, %109, 0 {cannir.buffer.generated_sync}
                  %216 = arith.index_cast %arg90 : i64 to index
                  %217 = arith.index_cast %207 : i64 to index
                  %218 = arith.muli %217, %216 : index
                  %219 = memref.load %112[%218] : memref<1xi32, 1>
                  ascvec.release_buf s, %109, 0 {cannir.buffer.generated_sync}
                  scf.yield %219 : i32
                } else {
                  scf.yield %c-1_i32 : i32
                }
                %210 = arith.addi %206, %c49_i64 overflow<nsw> : i64
                %211 = arith.cmpi slt, %210, %202 : i64
                %212 = scf.if %211 -> (i32) {
                  ascvec.get_buf s, %109, 0 {cannir.buffer.generated_sync}
                  %216 = arith.index_cast %arg90 : i64 to index
                  %217 = arith.index_cast %210 : i64 to index
                  %218 = arith.muli %217, %216 : index
                  %219 = memref.load %112[%218] : memref<1xi32, 1>
                  ascvec.release_buf s, %109, 0 {cannir.buffer.generated_sync}
                  scf.yield %219 : i32
                } else {
                  scf.yield %c-1_i32 : i32
                }
                %213 = arith.cmpi sge, %209, %c0_i32 : i32
                %214 = arith.cmpi sge, %212, %c0_i32 : i32
                %215 = arith.select %213, %214, %false : i1
                scf.if %215 {
                  %216 = arith.extui %209 : i32 to i64
                  %217 = arith.extui %212 : i32 to i64
                  %218 = arith.cmpi slt, %216, %217 : i64
                  %219 = arith.select %218, %216, %217 : i64
                  %220 = arith.select %218, %217, %216 : i64
                  %221 = arith.muli %219, %c1024_i64 : i64
                  %222 = arith.muli %220, %c1024_i64 : i64
                  %223 = arith.subi %222, %221 : i64
                  %224 = arith.muli %arg104, %c2048_i64 overflow<nsw> : i64
                  ascvec.get_buf mte2, %c8_i64, 0 {cannir.buffer.generated_sync}
                  "ascvec.copy_gm2ub_plan"(%alloc_27, %alloc_0, %c1_i32, %c0_i32, %224, %221, %c2_i64, %c1024_i64, %223, %c1024_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = 207360 : i64, ub_capacity = 16384 : i64}> : (memref<8192xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                  ascvec.release_buf mte2, %c8_i64, 0 {cannir.buffer.generated_sync}
                } else {
                  %216 = arith.select %213, %true, %214 : i1
                  scf.if %216 {
                    %217 = arith.extsi %209 : i32 to i64
                    %218 = arith.extsi %212 : i32 to i64
                    %219 = arith.select %213, %217, %218 : i64
                    %220 = cannir.extract_layout %arg2 : !cannir.tensor<<gm, bf16>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>
                    %221 = cannir.extract_shape %220 : <<(-1)>, <(1)>> -> <(-1)>
                    %222 = cannir.extract_dim %221[0] : !cannir.shape<(-1)>
                    %223 = arith.divui %222, %c512_i64 : i64
                    %224 = arith.minsi %219, %223 : i64
                    %225 = arith.subi %223, %224 : i64
                    %226 = arith.minsi %225, %c1_i64 : i64
                    %227 = arith.cmpi sle, %219, %223 : i64
                    %228 = arith.select %227, %226, %c0_i64 : i64
                    %229 = arith.muli %219, %c1024_i64 : i64
                    %230 = arith.muli %arg104, %c2048_i64 overflow<nsw> : i64
                    %231 = arith.muli %228, %c1024_i64 : i64
                    %232 = arith.cmpi eq, %231, %c0_i64 : i64
                    %233 = arith.cmpi ne, %231, %c0_i64 : i64
                    %234 = arith.extui %233 : i1 to i64
                    %235 = arith.select %232, %c2_i64, %231 : i64
                    ascvec.get_buf mte2, %c8_i64, 0 {cannir.buffer.generated_sync}
                    "ascvec.copy_gm2ub_plan"(%alloc_27, %alloc_0, %c1_i32, %c0_i32, %230, %229, %234, %235, %c0_i64, %235, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = 207360 : i64, ub_capacity = 16384 : i64}> : (memref<8192xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                    ascvec.release_buf mte2, %c8_i64, 0 {cannir.buffer.generated_sync}
                  }
                }
              }
            }
            ascvec.get_buf v, %c7_i64, 0 {cannir.buffer.generated_sync}
            ascvec.get_buf v, %c9_i64, 0 {cannir.buffer.generated_sync}
            ascvec.vf(%alloc_26, %alloc_28 : memref<8192xbf16, 1>, memref<8704xbf16, 1>) {
            ^bb0(%arg104: memref<8192xbf16, 1>, %arg105: memref<8704xbf16, 1>):
              %c1_i32_48 = arith.constant 1 : i32
              %c0_i32_49 = arith.constant 0 : i32
              %c6528_i32 = arith.constant 6528 : i32
              %c384_i32 = arith.constant 384 : i32
              %c4352_i32 = arith.constant 4352 : i32
              %c256_i32 = arith.constant 256 : i32
              %c2176_i32 = arith.constant 2176 : i32
              %c128_i32 = arith.constant 128 : i32
              %c17 = arith.constant 17 : index
              %c16_i32_50 = arith.constant 16 : i32
              %c0_51 = arith.constant 0 : index
              %c512_i32_52 = arith.constant 512 : i32
              %206 = ascvec.create_mask elem_bits 16 : !ascvec.mask
              scf.for %arg106 = %c0_i32_49 to %c16_i32_50 step %c1_i32_48  : i32 {
                %207 = arith.muli %arg106, %c512_i32_52 overflow<nsw> : i32
                %208 = arith.index_cast %207 : i32 to index
                %209 = ascvec.loadalign %arg104[%208] : memref<8192xbf16, 1>, index -> <bf16>
                %210 = arith.muli %arg106, %c16_i32_50 overflow<nsw> : i32
                %211 = arith.index_cast %210 : i32 to index
                ascvec.storealign_strided %arg105[%211], %209, %206, %c17, %c0_51 : memref<8704xbf16, 1>, <bf16>
                %212 = arith.addi %207, %c128_i32 overflow<nsw> : i32
                %213 = arith.index_cast %212 : i32 to index
                %214 = ascvec.loadalign %arg104[%213] : memref<8192xbf16, 1>, index -> <bf16>
                %215 = arith.addi %210, %c2176_i32 overflow<nsw> : i32
                %216 = arith.index_cast %215 : i32 to index
                ascvec.storealign_strided %arg105[%216], %214, %206, %c17, %c0_51 : memref<8704xbf16, 1>, <bf16>
                %217 = arith.addi %207, %c256_i32 overflow<nsw> : i32
                %218 = arith.index_cast %217 : i32 to index
                %219 = ascvec.loadalign %arg104[%218] : memref<8192xbf16, 1>, index -> <bf16>
                %220 = arith.addi %210, %c4352_i32 overflow<nsw> : i32
                %221 = arith.index_cast %220 : i32 to index
                ascvec.storealign_strided %arg105[%221], %219, %206, %c17, %c0_51 : memref<8704xbf16, 1>, <bf16>
                %222 = arith.addi %207, %c384_i32 overflow<nsw> : i32
                %223 = arith.index_cast %222 : i32 to index
                %224 = ascvec.loadalign %arg104[%223] : memref<8192xbf16, 1>, index -> <bf16>
                %225 = arith.addi %210, %c6528_i32 overflow<nsw> : i32
                %226 = arith.index_cast %225 : i32 to index
                ascvec.storealign_strided %arg105[%226], %224, %206, %c17, %c0_51 : memref<8704xbf16, 1>, <bf16>
              }
            } {cannir.vf_mode = "simd"}
            ascvec.release_buf v, %c9_i64, 0 {cannir.buffer.generated_sync}
            ascvec.release_buf v, %c7_i64, 0 {cannir.buffer.generated_sync}
            %205 = arith.cmpi eq, %51, %c0_i64 : i64
            ascvec.get_buf mte3, %c9_i64, 0 {cannir.buffer.generated_sync}
            scf.if %205 {
              %206 = ascscalar.bitcast_memref %alloc_28 byte_offset = %c0_i64 : memref<8704xbf16, 1> -> memref<?xbf16, 1>
              %207 = ascscalar.bind_local_ptr %arg67 : memref<1xbf16, 2>
              %208 = arith.index_cast %arg68 : i64 to index
              %209 = arith.divui %208, %c16 : index
              %210 = arith.addi %209, %c512 : index
              ascvec.copy_ub2l1_strided %207[%210], %206[%c0], %c32_i32, %c16_i32, %c1_i32, %c48_i32 : memref<1xbf16, 2>, memref<?xbf16, 1>
            } else {
              %206 = ascscalar.bitcast_memref %alloc_28 byte_offset = %c0_i64 : memref<8704xbf16, 1> -> memref<?xbf16, 1>
              %207 = ascscalar.bind_local_ptr %arg77 : memref<1xbf16, 2>
              %208 = arith.index_cast %arg78 : i64 to index
              %209 = arith.divui %208, %c16 : index
              %210 = arith.addi %209, %c512 : index
              ascvec.copy_ub2l1_strided %207[%210], %206[%c0], %c32_i32, %c16_i32, %c1_i32, %c48_i32 : memref<1xbf16, 2>, memref<?xbf16, 1>
            }
            ascvec.release_buf mte3, %c9_i64, 0 {cannir.buffer.generated_sync}
            ascvec.get_buf v, %c8_i64, 0 {cannir.buffer.generated_sync}
            ascvec.get_buf v, %c10_i64, 0 {cannir.buffer.generated_sync}
            ascvec.vf(%alloc_27, %alloc_29 : memref<8192xbf16, 1>, memref<8704xbf16, 1>) {
            ^bb0(%arg104: memref<8192xbf16, 1>, %arg105: memref<8704xbf16, 1>):
              %c1_i32_48 = arith.constant 1 : i32
              %c0_i32_49 = arith.constant 0 : i32
              %c6528_i32 = arith.constant 6528 : i32
              %c384_i32 = arith.constant 384 : i32
              %c4352_i32 = arith.constant 4352 : i32
              %c256_i32 = arith.constant 256 : i32
              %c2176_i32 = arith.constant 2176 : i32
              %c128_i32 = arith.constant 128 : i32
              %c17 = arith.constant 17 : index
              %c16_i32_50 = arith.constant 16 : i32
              %c0_51 = arith.constant 0 : index
              %c512_i32_52 = arith.constant 512 : i32
              %206 = ascvec.create_mask elem_bits 16 : !ascvec.mask
              scf.for %arg106 = %c0_i32_49 to %c16_i32_50 step %c1_i32_48  : i32 {
                %207 = arith.muli %arg106, %c512_i32_52 overflow<nsw> : i32
                %208 = arith.index_cast %207 : i32 to index
                %209 = ascvec.loadalign %arg104[%208] : memref<8192xbf16, 1>, index -> <bf16>
                %210 = arith.muli %arg106, %c16_i32_50 overflow<nsw> : i32
                %211 = arith.index_cast %210 : i32 to index
                ascvec.storealign_strided %arg105[%211], %209, %206, %c17, %c0_51 : memref<8704xbf16, 1>, <bf16>
                %212 = arith.addi %207, %c128_i32 overflow<nsw> : i32
                %213 = arith.index_cast %212 : i32 to index
                %214 = ascvec.loadalign %arg104[%213] : memref<8192xbf16, 1>, index -> <bf16>
                %215 = arith.addi %210, %c2176_i32 overflow<nsw> : i32
                %216 = arith.index_cast %215 : i32 to index
                ascvec.storealign_strided %arg105[%216], %214, %206, %c17, %c0_51 : memref<8704xbf16, 1>, <bf16>
                %217 = arith.addi %207, %c256_i32 overflow<nsw> : i32
                %218 = arith.index_cast %217 : i32 to index
                %219 = ascvec.loadalign %arg104[%218] : memref<8192xbf16, 1>, index -> <bf16>
                %220 = arith.addi %210, %c4352_i32 overflow<nsw> : i32
                %221 = arith.index_cast %220 : i32 to index
                ascvec.storealign_strided %arg105[%221], %219, %206, %c17, %c0_51 : memref<8704xbf16, 1>, <bf16>
                %222 = arith.addi %207, %c384_i32 overflow<nsw> : i32
                %223 = arith.index_cast %222 : i32 to index
                %224 = ascvec.loadalign %arg104[%223] : memref<8192xbf16, 1>, index -> <bf16>
                %225 = arith.addi %210, %c6528_i32 overflow<nsw> : i32
                %226 = arith.index_cast %225 : i32 to index
                ascvec.storealign_strided %arg105[%226], %224, %206, %c17, %c0_51 : memref<8704xbf16, 1>, <bf16>
              }
            } {cannir.vf_mode = "simd"}
            ascvec.release_buf v, %c10_i64, 0 {cannir.buffer.generated_sync}
            ascvec.release_buf v, %c8_i64, 0 {cannir.buffer.generated_sync}
            scf.if %205 {
              %206 = ascscalar.bitcast_memref %alloc_29 byte_offset = %c0_i64 : memref<8704xbf16, 1> -> memref<?xbf16, 1>
              %207 = ascscalar.bind_local_ptr %arg67 : memref<1xbf16, 2>
              %208 = arith.index_cast %arg68 : i64 to index
              %209 = arith.divui %208, %c16 : index
              %210 = arith.addi %209, %c768 : index
              ascvec.get_buf mte3, %c10_i64, 0 {cannir.buffer.generated_sync}
              ascvec.copy_ub2l1_strided %207[%210], %206[%c0], %c32_i32, %c16_i32, %c1_i32, %c48_i32 : memref<1xbf16, 2>, memref<?xbf16, 1>
              ascvec.release_buf mte3, %c10_i64, 0 {cannir.buffer.generated_sync}
              %211 = arith.cmpi eq, %arg101, %c18_i64 : i64
              %212 = arith.select %211, %c4_i64, %c3_i64 : i64
              %213 = arith.cmpi eq, %arg101, %c19_i64 : i64
              %214 = arith.select %213, %c5_i64, %212 : i64
              ascvec.sync_intra_arrive mte3, %214 {cannir.sync.banks = 1 : i32}
            } else {
              %206 = ascscalar.bitcast_memref %alloc_29 byte_offset = %c0_i64 : memref<8704xbf16, 1> -> memref<?xbf16, 1>
              %207 = ascscalar.bind_local_ptr %arg77 : memref<1xbf16, 2>
              %208 = arith.index_cast %arg78 : i64 to index
              %209 = arith.divui %208, %c16 : index
              %210 = arith.addi %209, %c768 : index
              ascvec.get_buf mte3, %c10_i64, 0 {cannir.buffer.generated_sync}
              ascvec.copy_ub2l1_strided %207[%210], %206[%c0], %c32_i32, %c16_i32, %c1_i32, %c48_i32 : memref<1xbf16, 2>, memref<?xbf16, 1>
              ascvec.release_buf mte3, %c10_i64, 0 {cannir.buffer.generated_sync}
              %211 = arith.cmpi eq, %arg102, %c22_i64 : i64
              %212 = arith.select %211, %c8_i64, %c7_i64 : i64
              %213 = arith.cmpi eq, %arg102, %c23_i64 : i64
              %214 = arith.select %213, %c9_i64, %212 : i64
              ascvec.sync_intra_arrive mte3, %214 {cannir.sync.banks = 2 : i32}
            }
          } else {
            %193 = arith.cmpi slt, %132, %134 : i64
            %194 = arith.select %193, %132, %134 : i64
            %195 = arith.muli %133, %c128_i64 : i64
            %196 = arith.subi %194, %195 : i64
            %197 = arith.muli %51, %c64_i64 : i64
            %198 = arith.subi %196, %197 : i64
            %199 = arith.cmpi sgt, %198, %c0_i64 : i64
            %200 = arith.cmpi slt, %198, %c64_i64 : i64
            %201 = arith.select %200, %198, %c64_i64 : i64
            %202 = arith.select %199, %201, %c0_i64 : i64
            %203 = arith.cmpi sge, %202, %c48_i64 : i64
            scf.if %203 {
              scf.for %arg104 = %c0_i64 to %c8_i64 step %c1_i64  : i64 {
                %206 = arith.muli %arg104, %c2_i64 overflow<nsw> : i64
                %207 = arith.addi %206, %c32_i64 overflow<nsw> : i64
                ascvec.get_buf s, %109, 0 {cannir.buffer.generated_sync}
                %208 = arith.index_cast %arg90 : i64 to index
                %209 = arith.index_cast %207 : i64 to index
                %210 = arith.muli %209, %208 : index
                %211 = memref.load %112[%210] : memref<1xi32, 1>
                %212 = arith.extui %211 : i32 to i64
                %213 = arith.addi %206, %c33_i64 overflow<nsw> : i64
                %214 = arith.index_cast %213 : i64 to index
                %215 = arith.muli %214, %208 : index
                %216 = memref.load %112[%215] : memref<1xi32, 1>
                ascvec.release_buf s, %109, 0 {cannir.buffer.generated_sync}
                %217 = arith.extui %216 : i32 to i64
                %218 = arith.cmpi slt, %212, %217 : i64
                %219 = arith.select %218, %212, %217 : i64
                %220 = arith.select %218, %217, %212 : i64
                %221 = arith.muli %219, %c1024_i64 : i64
                %222 = arith.muli %220, %c1024_i64 : i64
                %223 = arith.subi %222, %221 : i64
                %224 = arith.muli %arg104, %c2048_i64 overflow<nsw> : i64
                ascvec.get_buf mte2, %c7_i64, 0 {cannir.buffer.generated_sync}
                "ascvec.copy_gm2ub_plan"(%alloc_26, %alloc, %c1_i32, %c0_i32, %224, %221, %c2_i64, %c1024_i64, %223, %c1024_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = 190976 : i64, ub_capacity = 16384 : i64}> : (memref<8192xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                ascvec.release_buf mte2, %c7_i64, 0 {cannir.buffer.generated_sync}
              }
            } else {
              scf.for %arg104 = %c0_i64 to %c8_i64 step %c1_i64  : i64 {
                %206 = arith.muli %arg104, %c2_i64 overflow<nsw> : i64
                %207 = arith.addi %206, %c32_i64 overflow<nsw> : i64
                %208 = arith.cmpi slt, %207, %202 : i64
                %209 = scf.if %208 -> (i32) {
                  ascvec.get_buf s, %109, 0 {cannir.buffer.generated_sync}
                  %216 = arith.index_cast %arg90 : i64 to index
                  %217 = arith.index_cast %207 : i64 to index
                  %218 = arith.muli %217, %216 : index
                  %219 = memref.load %112[%218] : memref<1xi32, 1>
                  ascvec.release_buf s, %109, 0 {cannir.buffer.generated_sync}
                  scf.yield %219 : i32
                } else {
                  scf.yield %c-1_i32 : i32
                }
                %210 = arith.addi %206, %c33_i64 overflow<nsw> : i64
                %211 = arith.cmpi slt, %210, %202 : i64
                %212 = scf.if %211 -> (i32) {
                  ascvec.get_buf s, %109, 0 {cannir.buffer.generated_sync}
                  %216 = arith.index_cast %arg90 : i64 to index
                  %217 = arith.index_cast %210 : i64 to index
                  %218 = arith.muli %217, %216 : index
                  %219 = memref.load %112[%218] : memref<1xi32, 1>
                  ascvec.release_buf s, %109, 0 {cannir.buffer.generated_sync}
                  scf.yield %219 : i32
                } else {
                  scf.yield %c-1_i32 : i32
                }
                %213 = arith.cmpi sge, %209, %c0_i32 : i32
                %214 = arith.cmpi sge, %212, %c0_i32 : i32
                %215 = arith.select %213, %214, %false : i1
                scf.if %215 {
                  %216 = arith.extui %209 : i32 to i64
                  %217 = arith.extui %212 : i32 to i64
                  %218 = arith.cmpi slt, %216, %217 : i64
                  %219 = arith.select %218, %216, %217 : i64
                  %220 = arith.select %218, %217, %216 : i64
                  %221 = arith.muli %219, %c1024_i64 : i64
                  %222 = arith.muli %220, %c1024_i64 : i64
                  %223 = arith.subi %222, %221 : i64
                  %224 = arith.muli %arg104, %c2048_i64 overflow<nsw> : i64
                  ascvec.get_buf mte2, %c7_i64, 0 {cannir.buffer.generated_sync}
                  "ascvec.copy_gm2ub_plan"(%alloc_26, %alloc, %c1_i32, %c0_i32, %224, %221, %c2_i64, %c1024_i64, %223, %c1024_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = 190976 : i64, ub_capacity = 16384 : i64}> : (memref<8192xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                  ascvec.release_buf mte2, %c7_i64, 0 {cannir.buffer.generated_sync}
                } else {
                  %216 = arith.select %213, %true, %214 : i1
                  scf.if %216 {
                    %217 = arith.extsi %209 : i32 to i64
                    %218 = arith.extsi %212 : i32 to i64
                    %219 = arith.select %213, %217, %218 : i64
                    %220 = cannir.extract_layout %arg1 : !cannir.tensor<<gm, bf16>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>
                    %221 = cannir.extract_shape %220 : <<(-1)>, <(1)>> -> <(-1)>
                    %222 = cannir.extract_dim %221[0] : !cannir.shape<(-1)>
                    %223 = arith.divui %222, %c512_i64 : i64
                    %224 = arith.minsi %219, %223 : i64
                    %225 = arith.subi %223, %224 : i64
                    %226 = arith.minsi %225, %c1_i64 : i64
                    %227 = arith.cmpi sle, %219, %223 : i64
                    %228 = arith.select %227, %226, %c0_i64 : i64
                    %229 = arith.muli %219, %c1024_i64 : i64
                    %230 = arith.muli %arg104, %c2048_i64 overflow<nsw> : i64
                    %231 = arith.muli %228, %c1024_i64 : i64
                    %232 = arith.cmpi eq, %231, %c0_i64 : i64
                    %233 = arith.cmpi ne, %231, %c0_i64 : i64
                    %234 = arith.extui %233 : i1 to i64
                    %235 = arith.select %232, %c2_i64, %231 : i64
                    ascvec.get_buf mte2, %c7_i64, 0 {cannir.buffer.generated_sync}
                    "ascvec.copy_gm2ub_plan"(%alloc_26, %alloc, %c1_i32, %c0_i32, %230, %229, %234, %235, %c0_i64, %235, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = 190976 : i64, ub_capacity = 16384 : i64}> : (memref<8192xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                    ascvec.release_buf mte2, %c7_i64, 0 {cannir.buffer.generated_sync}
                  }
                }
              }
            }
            %204 = arith.cmpi sge, %202, %c64_i64 : i64
            scf.if %204 {
              scf.for %arg104 = %c0_i64 to %c8_i64 step %c1_i64  : i64 {
                %206 = arith.muli %arg104, %c2_i64 overflow<nsw> : i64
                %207 = arith.addi %206, %c48_i64 overflow<nsw> : i64
                ascvec.get_buf s, %109, 0 {cannir.buffer.generated_sync}
                %208 = arith.index_cast %arg90 : i64 to index
                %209 = arith.index_cast %207 : i64 to index
                %210 = arith.muli %209, %208 : index
                %211 = memref.load %112[%210] : memref<1xi32, 1>
                %212 = arith.extui %211 : i32 to i64
                %213 = arith.addi %206, %c49_i64 overflow<nsw> : i64
                %214 = arith.index_cast %213 : i64 to index
                %215 = arith.muli %214, %208 : index
                %216 = memref.load %112[%215] : memref<1xi32, 1>
                ascvec.release_buf s, %109, 0 {cannir.buffer.generated_sync}
                %217 = arith.extui %216 : i32 to i64
                %218 = arith.cmpi slt, %212, %217 : i64
                %219 = arith.select %218, %212, %217 : i64
                %220 = arith.select %218, %217, %212 : i64
                %221 = arith.muli %219, %c1024_i64 : i64
                %222 = arith.muli %220, %c1024_i64 : i64
                %223 = arith.subi %222, %221 : i64
                %224 = arith.muli %arg104, %c2048_i64 overflow<nsw> : i64
                ascvec.get_buf mte2, %c8_i64, 0 {cannir.buffer.generated_sync}
                "ascvec.copy_gm2ub_plan"(%alloc_27, %alloc, %c1_i32, %c0_i32, %224, %221, %c2_i64, %c1024_i64, %223, %c1024_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = 207360 : i64, ub_capacity = 16384 : i64}> : (memref<8192xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                ascvec.release_buf mte2, %c8_i64, 0 {cannir.buffer.generated_sync}
              }
            } else {
              scf.for %arg104 = %c0_i64 to %c8_i64 step %c1_i64  : i64 {
                %206 = arith.muli %arg104, %c2_i64 overflow<nsw> : i64
                %207 = arith.addi %206, %c48_i64 overflow<nsw> : i64
                %208 = arith.cmpi slt, %207, %202 : i64
                %209 = scf.if %208 -> (i32) {
                  ascvec.get_buf s, %109, 0 {cannir.buffer.generated_sync}
                  %216 = arith.index_cast %arg90 : i64 to index
                  %217 = arith.index_cast %207 : i64 to index
                  %218 = arith.muli %217, %216 : index
                  %219 = memref.load %112[%218] : memref<1xi32, 1>
                  ascvec.release_buf s, %109, 0 {cannir.buffer.generated_sync}
                  scf.yield %219 : i32
                } else {
                  scf.yield %c-1_i32 : i32
                }
                %210 = arith.addi %206, %c49_i64 overflow<nsw> : i64
                %211 = arith.cmpi slt, %210, %202 : i64
                %212 = scf.if %211 -> (i32) {
                  ascvec.get_buf s, %109, 0 {cannir.buffer.generated_sync}
                  %216 = arith.index_cast %arg90 : i64 to index
                  %217 = arith.index_cast %210 : i64 to index
                  %218 = arith.muli %217, %216 : index
                  %219 = memref.load %112[%218] : memref<1xi32, 1>
                  ascvec.release_buf s, %109, 0 {cannir.buffer.generated_sync}
                  scf.yield %219 : i32
                } else {
                  scf.yield %c-1_i32 : i32
                }
                %213 = arith.cmpi sge, %209, %c0_i32 : i32
                %214 = arith.cmpi sge, %212, %c0_i32 : i32
                %215 = arith.select %213, %214, %false : i1
                scf.if %215 {
                  %216 = arith.extui %209 : i32 to i64
                  %217 = arith.extui %212 : i32 to i64
                  %218 = arith.cmpi slt, %216, %217 : i64
                  %219 = arith.select %218, %216, %217 : i64
                  %220 = arith.select %218, %217, %216 : i64
                  %221 = arith.muli %219, %c1024_i64 : i64
                  %222 = arith.muli %220, %c1024_i64 : i64
                  %223 = arith.subi %222, %221 : i64
                  %224 = arith.muli %arg104, %c2048_i64 overflow<nsw> : i64
                  ascvec.get_buf mte2, %c8_i64, 0 {cannir.buffer.generated_sync}
                  "ascvec.copy_gm2ub_plan"(%alloc_27, %alloc, %c1_i32, %c0_i32, %224, %221, %c2_i64, %c1024_i64, %223, %c1024_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = 207360 : i64, ub_capacity = 16384 : i64}> : (memref<8192xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                  ascvec.release_buf mte2, %c8_i64, 0 {cannir.buffer.generated_sync}
                } else {
                  %216 = arith.select %213, %true, %214 : i1
                  scf.if %216 {
                    %217 = arith.extsi %209 : i32 to i64
                    %218 = arith.extsi %212 : i32 to i64
                    %219 = arith.select %213, %217, %218 : i64
                    %220 = cannir.extract_layout %arg1 : !cannir.tensor<<gm, bf16>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>
                    %221 = cannir.extract_shape %220 : <<(-1)>, <(1)>> -> <(-1)>
                    %222 = cannir.extract_dim %221[0] : !cannir.shape<(-1)>
                    %223 = arith.divui %222, %c512_i64 : i64
                    %224 = arith.minsi %219, %223 : i64
                    %225 = arith.subi %223, %224 : i64
                    %226 = arith.minsi %225, %c1_i64 : i64
                    %227 = arith.cmpi sle, %219, %223 : i64
                    %228 = arith.select %227, %226, %c0_i64 : i64
                    %229 = arith.muli %219, %c1024_i64 : i64
                    %230 = arith.muli %arg104, %c2048_i64 overflow<nsw> : i64
                    %231 = arith.muli %228, %c1024_i64 : i64
                    %232 = arith.cmpi eq, %231, %c0_i64 : i64
                    %233 = arith.cmpi ne, %231, %c0_i64 : i64
                    %234 = arith.extui %233 : i1 to i64
                    %235 = arith.select %232, %c2_i64, %231 : i64
                    ascvec.get_buf mte2, %c8_i64, 0 {cannir.buffer.generated_sync}
                    "ascvec.copy_gm2ub_plan"(%alloc_27, %alloc, %c1_i32, %c0_i32, %230, %229, %234, %235, %c0_i64, %235, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = 207360 : i64, ub_capacity = 16384 : i64}> : (memref<8192xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                    ascvec.release_buf mte2, %c8_i64, 0 {cannir.buffer.generated_sync}
                  }
                }
              }
            }
            ascvec.get_buf v, %c7_i64, 0 {cannir.buffer.generated_sync}
            ascvec.get_buf v, %c9_i64, 0 {cannir.buffer.generated_sync}
            ascvec.vf(%alloc_26, %alloc_28 : memref<8192xbf16, 1>, memref<8704xbf16, 1>) {
            ^bb0(%arg104: memref<8192xbf16, 1>, %arg105: memref<8704xbf16, 1>):
              %c1_i32_48 = arith.constant 1 : i32
              %c0_i32_49 = arith.constant 0 : i32
              %c6528_i32 = arith.constant 6528 : i32
              %c384_i32 = arith.constant 384 : i32
              %c4352_i32 = arith.constant 4352 : i32
              %c256_i32 = arith.constant 256 : i32
              %c2176_i32 = arith.constant 2176 : i32
              %c128_i32 = arith.constant 128 : i32
              %c17 = arith.constant 17 : index
              %c16_i32_50 = arith.constant 16 : i32
              %c0_51 = arith.constant 0 : index
              %c512_i32_52 = arith.constant 512 : i32
              %206 = ascvec.create_mask elem_bits 16 : !ascvec.mask
              scf.for %arg106 = %c0_i32_49 to %c16_i32_50 step %c1_i32_48  : i32 {
                %207 = arith.muli %arg106, %c512_i32_52 overflow<nsw> : i32
                %208 = arith.index_cast %207 : i32 to index
                %209 = ascvec.loadalign %arg104[%208] : memref<8192xbf16, 1>, index -> <bf16>
                %210 = arith.muli %arg106, %c16_i32_50 overflow<nsw> : i32
                %211 = arith.index_cast %210 : i32 to index
                ascvec.storealign_strided %arg105[%211], %209, %206, %c17, %c0_51 : memref<8704xbf16, 1>, <bf16>
                %212 = arith.addi %207, %c128_i32 overflow<nsw> : i32
                %213 = arith.index_cast %212 : i32 to index
                %214 = ascvec.loadalign %arg104[%213] : memref<8192xbf16, 1>, index -> <bf16>
                %215 = arith.addi %210, %c2176_i32 overflow<nsw> : i32
                %216 = arith.index_cast %215 : i32 to index
                ascvec.storealign_strided %arg105[%216], %214, %206, %c17, %c0_51 : memref<8704xbf16, 1>, <bf16>
                %217 = arith.addi %207, %c256_i32 overflow<nsw> : i32
                %218 = arith.index_cast %217 : i32 to index
                %219 = ascvec.loadalign %arg104[%218] : memref<8192xbf16, 1>, index -> <bf16>
                %220 = arith.addi %210, %c4352_i32 overflow<nsw> : i32
                %221 = arith.index_cast %220 : i32 to index
                ascvec.storealign_strided %arg105[%221], %219, %206, %c17, %c0_51 : memref<8704xbf16, 1>, <bf16>
                %222 = arith.addi %207, %c384_i32 overflow<nsw> : i32
                %223 = arith.index_cast %222 : i32 to index
                %224 = ascvec.loadalign %arg104[%223] : memref<8192xbf16, 1>, index -> <bf16>
                %225 = arith.addi %210, %c6528_i32 overflow<nsw> : i32
                %226 = arith.index_cast %225 : i32 to index
                ascvec.storealign_strided %arg105[%226], %224, %206, %c17, %c0_51 : memref<8704xbf16, 1>, <bf16>
              }
            } {cannir.vf_mode = "simd"}
            ascvec.release_buf v, %c9_i64, 0 {cannir.buffer.generated_sync}
            ascvec.release_buf v, %c7_i64, 0 {cannir.buffer.generated_sync}
            %205 = arith.cmpi eq, %51, %c0_i64 : i64
            ascvec.get_buf mte3, %c9_i64, 0 {cannir.buffer.generated_sync}
            scf.if %205 {
              %206 = ascscalar.bitcast_memref %alloc_28 byte_offset = %c0_i64 : memref<8704xbf16, 1> -> memref<?xbf16, 1>
              %207 = ascscalar.bind_local_ptr %arg67 : memref<1xbf16, 2>
              %208 = arith.index_cast %arg68 : i64 to index
              %209 = arith.divui %208, %c16 : index
              %210 = arith.addi %209, %c512 : index
              ascvec.copy_ub2l1_strided %207[%210], %206[%c0], %c32_i32, %c16_i32, %c1_i32, %c48_i32 : memref<1xbf16, 2>, memref<?xbf16, 1>
            } else {
              %206 = ascscalar.bitcast_memref %alloc_28 byte_offset = %c0_i64 : memref<8704xbf16, 1> -> memref<?xbf16, 1>
              %207 = ascscalar.bind_local_ptr %arg77 : memref<1xbf16, 2>
              %208 = arith.index_cast %arg78 : i64 to index
              %209 = arith.divui %208, %c16 : index
              %210 = arith.addi %209, %c512 : index
              ascvec.copy_ub2l1_strided %207[%210], %206[%c0], %c32_i32, %c16_i32, %c1_i32, %c48_i32 : memref<1xbf16, 2>, memref<?xbf16, 1>
            }
            ascvec.release_buf mte3, %c9_i64, 0 {cannir.buffer.generated_sync}
            ascvec.get_buf v, %c8_i64, 0 {cannir.buffer.generated_sync}
            ascvec.get_buf v, %c10_i64, 0 {cannir.buffer.generated_sync}
            ascvec.vf(%alloc_27, %alloc_29 : memref<8192xbf16, 1>, memref<8704xbf16, 1>) {
            ^bb0(%arg104: memref<8192xbf16, 1>, %arg105: memref<8704xbf16, 1>):
              %c1_i32_48 = arith.constant 1 : i32
              %c0_i32_49 = arith.constant 0 : i32
              %c6528_i32 = arith.constant 6528 : i32
              %c384_i32 = arith.constant 384 : i32
              %c4352_i32 = arith.constant 4352 : i32
              %c256_i32 = arith.constant 256 : i32
              %c2176_i32 = arith.constant 2176 : i32
              %c128_i32 = arith.constant 128 : i32
              %c17 = arith.constant 17 : index
              %c16_i32_50 = arith.constant 16 : i32
              %c0_51 = arith.constant 0 : index
              %c512_i32_52 = arith.constant 512 : i32
              %206 = ascvec.create_mask elem_bits 16 : !ascvec.mask
              scf.for %arg106 = %c0_i32_49 to %c16_i32_50 step %c1_i32_48  : i32 {
                %207 = arith.muli %arg106, %c512_i32_52 overflow<nsw> : i32
                %208 = arith.index_cast %207 : i32 to index
                %209 = ascvec.loadalign %arg104[%208] : memref<8192xbf16, 1>, index -> <bf16>
                %210 = arith.muli %arg106, %c16_i32_50 overflow<nsw> : i32
                %211 = arith.index_cast %210 : i32 to index
                ascvec.storealign_strided %arg105[%211], %209, %206, %c17, %c0_51 : memref<8704xbf16, 1>, <bf16>
                %212 = arith.addi %207, %c128_i32 overflow<nsw> : i32
                %213 = arith.index_cast %212 : i32 to index
                %214 = ascvec.loadalign %arg104[%213] : memref<8192xbf16, 1>, index -> <bf16>
                %215 = arith.addi %210, %c2176_i32 overflow<nsw> : i32
                %216 = arith.index_cast %215 : i32 to index
                ascvec.storealign_strided %arg105[%216], %214, %206, %c17, %c0_51 : memref<8704xbf16, 1>, <bf16>
                %217 = arith.addi %207, %c256_i32 overflow<nsw> : i32
                %218 = arith.index_cast %217 : i32 to index
                %219 = ascvec.loadalign %arg104[%218] : memref<8192xbf16, 1>, index -> <bf16>
                %220 = arith.addi %210, %c4352_i32 overflow<nsw> : i32
                %221 = arith.index_cast %220 : i32 to index
                ascvec.storealign_strided %arg105[%221], %219, %206, %c17, %c0_51 : memref<8704xbf16, 1>, <bf16>
                %222 = arith.addi %207, %c384_i32 overflow<nsw> : i32
                %223 = arith.index_cast %222 : i32 to index
                %224 = ascvec.loadalign %arg104[%223] : memref<8192xbf16, 1>, index -> <bf16>
                %225 = arith.addi %210, %c6528_i32 overflow<nsw> : i32
                %226 = arith.index_cast %225 : i32 to index
                ascvec.storealign_strided %arg105[%226], %224, %206, %c17, %c0_51 : memref<8704xbf16, 1>, <bf16>
              }
            } {cannir.vf_mode = "simd"}
            ascvec.release_buf v, %c10_i64, 0 {cannir.buffer.generated_sync}
            ascvec.release_buf v, %c8_i64, 0 {cannir.buffer.generated_sync}
            scf.if %205 {
              %206 = ascscalar.bitcast_memref %alloc_29 byte_offset = %c0_i64 : memref<8704xbf16, 1> -> memref<?xbf16, 1>
              %207 = ascscalar.bind_local_ptr %arg67 : memref<1xbf16, 2>
              %208 = arith.index_cast %arg68 : i64 to index
              %209 = arith.divui %208, %c16 : index
              %210 = arith.addi %209, %c768 : index
              ascvec.get_buf mte3, %c10_i64, 0 {cannir.buffer.generated_sync}
              ascvec.copy_ub2l1_strided %207[%210], %206[%c0], %c32_i32, %c16_i32, %c1_i32, %c48_i32 : memref<1xbf16, 2>, memref<?xbf16, 1>
              ascvec.release_buf mte3, %c10_i64, 0 {cannir.buffer.generated_sync}
              %211 = arith.cmpi eq, %arg101, %c18_i64 : i64
              %212 = arith.select %211, %c4_i64, %c3_i64 : i64
              %213 = arith.cmpi eq, %arg101, %c19_i64 : i64
              %214 = arith.select %213, %c5_i64, %212 : i64
              ascvec.sync_intra_arrive mte3, %214 {cannir.sync.banks = 1 : i32}
            } else {
              %206 = ascscalar.bitcast_memref %alloc_29 byte_offset = %c0_i64 : memref<8704xbf16, 1> -> memref<?xbf16, 1>
              %207 = ascscalar.bind_local_ptr %arg77 : memref<1xbf16, 2>
              %208 = arith.index_cast %arg78 : i64 to index
              %209 = arith.divui %208, %c16 : index
              %210 = arith.addi %209, %c768 : index
              ascvec.get_buf mte3, %c10_i64, 0 {cannir.buffer.generated_sync}
              ascvec.copy_ub2l1_strided %207[%210], %206[%c0], %c32_i32, %c16_i32, %c1_i32, %c48_i32 : memref<1xbf16, 2>, memref<?xbf16, 1>
              ascvec.release_buf mte3, %c10_i64, 0 {cannir.buffer.generated_sync}
              %211 = arith.cmpi eq, %arg102, %c22_i64 : i64
              %212 = arith.select %211, %c8_i64, %c7_i64 : i64
              %213 = arith.cmpi eq, %arg102, %c23_i64 : i64
              %214 = arith.select %213, %c9_i64, %212 : i64
              ascvec.sync_intra_arrive mte3, %214 {cannir.sync.banks = 2 : i32}
            }
          }
        }
        %164 = arith.cmpi sge, %arg97, %c4_i64 : i64
        %165 = arith.cmpi sge, %154, %c0_i64 : i64
        %166 = arith.select %164, %165, %false : i1
        scf.if %166 {
          %192 = arith.cmpi eq, %153, %29 : i64
          %193 = arith.cmpi sgt, %5, %c0_i64 : i64
          %194 = arith.select %192, %193, %false : i1
          %195 = arith.cmpi eq, %153, %35 : i64
          %196 = arith.cmpi sgt, %9, %c0_i64 : i64
          %197 = arith.select %195, %196, %false : i1
          %198 = arith.extui %193 : i1 to i64
          %199 = arith.addi %13, %198 : i64
          %200 = arith.select %197, %199, %c-1_i64 : i64
          %201 = arith.select %194, %13, %200 : i64
          %202 = arith.addi %154, %c1_i64 : i64
          %203 = arith.cmpi eq, %202, %155 : i64
          %204 = arith.cmpi eq, %154, %c0_i64 : i64
          scf.if %204 {
            ascvec.sync_intra_wait v, %c1_i64 {cannir.sync.banks = 3 : i32}
            ascvec.get_buf v, %c5_i64, 0 {cannir.buffer.generated_sync}
            ascvec.vf(%alloc_24 : memref<16384xf32, 1>) {
            ^bb0(%arg104: memref<16384xf32, 1>):
              %c0_48 = arith.constant 0 : index
              %c1_i32_49 = arith.constant 1 : i32
              %c0_50 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              scf.for %arg105 = %c0_50 to %c1 step %c1 {
                %mask, %remaining_out = ascvec.update_mask %c1_i32_49 : !ascvec.mask
                %210 = ascvec.loadalign %arg104[%c0_48] : memref<16384xf32, 1>, index -> <f32>
                ascvec.storealign %arg104[%c0_48], %210, %mask : memref<16384xf32, 1>, index, <f32>
              }
            } {cannir.vf_mode = "simd"}
            ascvec.release_buf v, %c5_i64, 0 {cannir.buffer.generated_sync}
            %209 = arith.cmpi ne, %202, %155 : i64
            scf.if %209 {
              ascvec.sync_intra_arrive v, %c1_i64 {cannir.sync.banks = 3 : i32}
            }
            scf.if %203 {
              %210 = arith.cmpi sge, %201, %c0_i64 : i64
              scf.if %210 {
                ascvec.get_buf v, %c5_i64, 0 {cannir.buffer.generated_sync}
                ascvec.vf(%152, %alloc_20, %alloc_24 : i64, memref<128xf32, 1>, memref<16384xf32, 1>) {
                ^bb0(%arg104: i64, %arg105: memref<128xf32, 1>, %arg106: memref<16384xf32, 1>):
                  %c1_i32_48 = arith.constant 1 : i32
                  %c32_i32_49 = arith.constant 32 : i32
                  %c0_i32_50 = arith.constant 0 : i32
                  %c192_i32 = arith.constant 192 : i32
                  %c128_i32 = arith.constant 128 : i32
                  %c64_i32 = arith.constant 64 : i32
                  %c256_i32 = arith.constant 256 : i32
                  %c64_i64_51 = arith.constant 64 : i64
                  %c0_i64_52 = arith.constant 0 : i64
                  %c2_i64_53 = arith.constant 2 : i64
                  %cst = arith.constant 1.000000e+00 : f32
                  %212 = ascvec.create_mask : !ascvec.mask
                  scf.for %arg107 = %c0_i32_50 to %c32_i32_49 step %c1_i32_48  : i32 {
                    %213 = ascvec.duplicate_scalar_nomask %cst : f32 -> <f32>
                    %214 = arith.remsi %arg104, %c2_i64_53 : i64
                    %215 = arith.cmpi ne, %214, %c0_i64_52 : i64
                    %216 = arith.cmpi slt, %214, %c0_i64_52 : i64
                    %217 = arith.andi %215, %216 : i1
                    %218 = arith.select %217, %c2_i64_53, %c0_i64_52 : i64
                    %219 = arith.addi %214, %218 overflow<nsw> : i64
                    %220 = arith.muli %219, %c64_i64_51 overflow<nsw> : i64
                    %221 = arith.extsi %arg107 : i32 to i64
                    %222 = arith.addi %220, %221 overflow<nsw> : i64
                    %223 = arith.index_cast %222 : i64 to index
                    %224 = ascvec.loadalign_brc %arg105[%223] : memref<128xf32, 1>, index -> <f32>
                    %225 = ascvec.div %213, %224, %212 : <f32>
                    %226 = arith.muli %arg107, %c256_i32 overflow<nsw> : i32
                    %227 = arith.index_cast %226 : i32 to index
                    %228 = ascvec.loadalign %arg106[%227] : memref<16384xf32, 1>, index -> <f32>
                    %229 = ascvec.mul %228, %225, %212 : <f32>
                    ascvec.storealign %arg106[%227], %229, %212 : memref<16384xf32, 1>, index, <f32>
                    %230 = arith.addi %226, %c64_i32 overflow<nsw> : i32
                    %231 = arith.index_cast %230 : i32 to index
                    %232 = ascvec.loadalign %arg106[%231] : memref<16384xf32, 1>, index -> <f32>
                    %233 = ascvec.mul %232, %225, %212 : <f32>
                    ascvec.storealign %arg106[%231], %233, %212 : memref<16384xf32, 1>, index, <f32>
                    %234 = arith.addi %226, %c128_i32 overflow<nsw> : i32
                    %235 = arith.index_cast %234 : i32 to index
                    %236 = ascvec.loadalign %arg106[%235] : memref<16384xf32, 1>, index -> <f32>
                    %237 = ascvec.mul %236, %225, %212 : <f32>
                    ascvec.storealign %arg106[%235], %237, %212 : memref<16384xf32, 1>, index, <f32>
                    %238 = arith.addi %226, %c192_i32 overflow<nsw> : i32
                    %239 = arith.index_cast %238 : i32 to index
                    %240 = ascvec.loadalign %arg106[%239] : memref<16384xf32, 1>, index -> <f32>
                    %241 = ascvec.mul %240, %225, %212 : <f32>
                    ascvec.storealign %arg106[%239], %241, %212 : memref<16384xf32, 1>, index, <f32>
                  } {cannir.loop_unroll = 1 : i64}
                } {cannir.vf_mode = "simd"}
                ascvec.release_buf v, %c5_i64, 0 {cannir.buffer.generated_sync}
                %211 = arith.cmpi sle, %59, %c0_i64 : i64
                scf.if %211 {
                  ascvec.sync_intra_arrive v, %c1_i64 {cannir.sync.banks = 3 : i32}
                }
              } else {
                ascvec.get_buf v, %c5_i64, 0 {cannir.buffer.generated_sync}
                %211 = ascscalar.bitcast_memref %alloc_24 byte_offset = %c0_i64 : memref<16384xf32, 1> -> memref<?xbf16, 1>
                ascvec.vf(%152, %alloc_20, %alloc_24, %211 : i64, memref<128xf32, 1>, memref<16384xf32, 1>, memref<?xbf16, 1>) {
                ^bb0(%arg104: i64, %arg105: memref<128xf32, 1>, %arg106: memref<16384xf32, 1>, %arg107: memref<?xbf16, 1>):
                  %c1_i32_48 = arith.constant 1 : i32
                  %c32_i32_49 = arith.constant 32 : i32
                  %c0_i32_50 = arith.constant 0 : i32
                  %c192_i32 = arith.constant 192 : i32
                  %c128_i32 = arith.constant 128 : i32
                  %c64_i32 = arith.constant 64 : i32
                  %c16384 = arith.constant 16384 : index
                  %c512_i32_51 = arith.constant 512 : i32
                  %c256_i32 = arith.constant 256 : i32
                  %c64_i64_52 = arith.constant 64 : i64
                  %c0_i64_53 = arith.constant 0 : i64
                  %c2_i64_54 = arith.constant 2 : i64
                  %cst = arith.constant 1.000000e+00 : f32
                  %213 = ascvec.create_mask : !ascvec.mask
                  scf.for %arg108 = %c0_i32_50 to %c32_i32_49 step %c1_i32_48  : i32 {
                    %214 = ascvec.duplicate_scalar_nomask %cst : f32 -> <f32>
                    %215 = arith.remsi %arg104, %c2_i64_54 : i64
                    %216 = arith.cmpi ne, %215, %c0_i64_53 : i64
                    %217 = arith.cmpi slt, %215, %c0_i64_53 : i64
                    %218 = arith.andi %216, %217 : i1
                    %219 = arith.select %218, %c2_i64_54, %c0_i64_53 : i64
                    %220 = arith.addi %215, %219 overflow<nsw> : i64
                    %221 = arith.muli %220, %c64_i64_52 overflow<nsw> : i64
                    %222 = arith.extsi %arg108 : i32 to i64
                    %223 = arith.addi %221, %222 overflow<nsw> : i64
                    %224 = arith.index_cast %223 : i64 to index
                    %225 = ascvec.loadalign_brc %arg105[%224] : memref<128xf32, 1>, index -> <f32>
                    %226 = ascvec.div %214, %225, %213 : <f32>
                    %227 = arith.muli %arg108, %c256_i32 overflow<nsw> : i32
                    %228 = arith.index_cast %227 : i32 to index
                    %229 = ascvec.loadalign %arg106[%228] : memref<16384xf32, 1>, index -> <f32>
                    %230 = ascvec.mul %229, %226, %213 : <f32>
                    %231 = ascvec.cast %230, %213 : <f32> -> <bf16>
                    %232 = arith.muli %arg108, %c512_i32_51 overflow<nsw> : i32
                    %233 = arith.index_cast %232 : i32 to index
                    %234 = arith.addi %233, %c16384 : index
                    ascvec.storealign_pack %arg107[%234], %231, %213 {mode = "b32_to_b16"} : memref<?xbf16, 1>, index, <bf16>
                    %235 = arith.addi %227, %c64_i32 overflow<nsw> : i32
                    %236 = arith.index_cast %235 : i32 to index
                    %237 = ascvec.loadalign %arg106[%236] : memref<16384xf32, 1>, index -> <f32>
                    %238 = ascvec.mul %237, %226, %213 : <f32>
                    %239 = ascvec.cast %238, %213 : <f32> -> <bf16>
                    %240 = arith.addi %232, %c64_i32 overflow<nsw> : i32
                    %241 = arith.index_cast %240 : i32 to index
                    %242 = arith.addi %241, %c16384 : index
                    ascvec.storealign_pack %arg107[%242], %239, %213 {mode = "b32_to_b16"} : memref<?xbf16, 1>, index, <bf16>
                    %243 = arith.addi %227, %c128_i32 overflow<nsw> : i32
                    %244 = arith.index_cast %243 : i32 to index
                    %245 = ascvec.loadalign %arg106[%244] : memref<16384xf32, 1>, index -> <f32>
                    %246 = ascvec.mul %245, %226, %213 : <f32>
                    %247 = ascvec.cast %246, %213 : <f32> -> <bf16>
                    %248 = arith.addi %232, %c128_i32 overflow<nsw> : i32
                    %249 = arith.index_cast %248 : i32 to index
                    %250 = arith.addi %249, %c16384 : index
                    ascvec.storealign_pack %arg107[%250], %247, %213 {mode = "b32_to_b16"} : memref<?xbf16, 1>, index, <bf16>
                    %251 = arith.addi %227, %c192_i32 overflow<nsw> : i32
                    %252 = arith.index_cast %251 : i32 to index
                    %253 = ascvec.loadalign %arg106[%252] : memref<16384xf32, 1>, index -> <f32>
                    %254 = ascvec.mul %253, %226, %213 : <f32>
                    %255 = ascvec.cast %254, %213 : <f32> -> <bf16>
                    %256 = arith.addi %232, %c192_i32 overflow<nsw> : i32
                    %257 = arith.index_cast %256 : i32 to index
                    %258 = arith.addi %257, %c16384 : index
                    ascvec.storealign_pack %arg107[%258], %255, %213 {mode = "b32_to_b16"} : memref<?xbf16, 1>, index, <bf16>
                  } {cannir.loop_unroll = 1 : i64}
                } {cannir.vf_mode = "simd"}
                ascvec.release_buf v, %c5_i64, 0 {cannir.buffer.generated_sync}
                %212 = arith.cmpi sle, %59, %c0_i64 : i64
                scf.if %212 {
                  ascvec.sync_intra_arrive v, %c1_i64 {cannir.sync.banks = 3 : i32}
                }
              }
            }
          } else {
            scf.if %203 {
              %209 = arith.cmpi sge, %201, %c0_i64 : i64
              scf.if %209 {
                ascvec.sync_intra_wait v, %c1_i64 {cannir.sync.banks = 3 : i32}
                ascvec.get_buf v, %c5_i64, 0 {cannir.buffer.generated_sync}
                ascvec.vf(%152, %154, %alloc_21, %alloc_20, %alloc_24, %alloc_24 : i64, i64, memref<256xf32, 1>, memref<128xf32, 1>, memref<16384xf32, 1>, memref<16384xf32, 1>) {
                ^bb0(%arg104: i64, %arg105: i64, %arg106: memref<256xf32, 1>, %arg107: memref<128xf32, 1>, %arg108: memref<16384xf32, 1>, %arg109: memref<16384xf32, 1>):
                  %c1_i32_48 = arith.constant 1 : i32
                  %c32_i32_49 = arith.constant 32 : i32
                  %c0_i32_50 = arith.constant 0 : i32
                  %c192_i32 = arith.constant 192 : i32
                  %c128_i32 = arith.constant 128 : i32
                  %c64_i32 = arith.constant 64 : i32
                  %c8192 = arith.constant 8192 : index
                  %c256_i32 = arith.constant 256 : i32
                  %cst = arith.constant 1.000000e+00 : f32
                  %c2_i32_51 = arith.constant 2 : i32
                  %c64_i64_52 = arith.constant 64 : i64
                  %c128_i64_53 = arith.constant 128 : i64
                  %c0_i64_54 = arith.constant 0 : i64
                  %c2_i64_55 = arith.constant 2 : i64
                  %211 = ascvec.create_mask : !ascvec.mask
                  scf.for %arg110 = %c0_i32_50 to %c32_i32_49 step %c1_i32_48  : i32 {
                    %212 = arith.remsi %arg104, %c2_i64_55 : i64
                    %213 = arith.cmpi ne, %212, %c0_i64_54 : i64
                    %214 = arith.cmpi slt, %212, %c0_i64_54 : i64
                    %215 = arith.andi %213, %214 : i1
                    %216 = arith.select %215, %c2_i64_55, %c0_i64_54 : i64
                    %217 = arith.addi %212, %216 overflow<nsw> : i64
                    %218 = arith.muli %217, %c128_i64_53 overflow<nsw> : i64
                    %219 = arith.remsi %arg105, %c2_i64_55 : i64
                    %220 = arith.muli %219, %c64_i64_52 overflow<nsw> : i64
                    %221 = arith.addi %218, %220 overflow<nsw> : i64
                    %222 = arith.muli %arg110, %c2_i32_51 overflow<nsw> : i32
                    %223 = arith.extsi %222 : i32 to i64
                    %224 = arith.addi %221, %223 overflow<nsw> : i64
                    %225 = arith.index_cast %224 : i64 to index
                    %226 = ascvec.loadalign_brc %arg106[%225] : memref<256xf32, 1>, index -> <f32>
                    %227 = ascvec.duplicate_scalar_nomask %cst : f32 -> <f32>
                    %228 = arith.muli %217, %c64_i64_52 overflow<nsw> : i64
                    %229 = arith.extsi %arg110 : i32 to i64
                    %230 = arith.addi %228, %229 overflow<nsw> : i64
                    %231 = arith.index_cast %230 : i64 to index
                    %232 = ascvec.loadalign_brc %arg107[%231] : memref<128xf32, 1>, index -> <f32>
                    %233 = ascvec.div %227, %232, %211 : <f32>
                    %234 = arith.muli %arg110, %c256_i32 overflow<nsw> : i32
                    %235 = arith.index_cast %234 : i32 to index
                    %236 = arith.addi %235, %c8192 : index
                    %237 = ascvec.loadalign %arg108[%236] : memref<16384xf32, 1>, index -> <f32>
                    %238 = ascvec.loadalign %arg109[%235] : memref<16384xf32, 1>, index -> <f32>
                    %239 = ascvec.mul %238, %226, %211 : <f32>
                    %240 = ascvec.add %239, %237, %211 : <f32>
                    %241 = ascvec.mul %240, %233, %211 : <f32>
                    ascvec.storealign %arg109[%235], %241, %211 : memref<16384xf32, 1>, index, <f32>
                    %242 = arith.addi %234, %c64_i32 overflow<nsw> : i32
                    %243 = arith.index_cast %242 : i32 to index
                    %244 = arith.addi %243, %c8192 : index
                    %245 = ascvec.loadalign %arg108[%244] : memref<16384xf32, 1>, index -> <f32>
                    %246 = ascvec.loadalign %arg109[%243] : memref<16384xf32, 1>, index -> <f32>
                    %247 = ascvec.mul %246, %226, %211 : <f32>
                    %248 = ascvec.add %247, %245, %211 : <f32>
                    %249 = ascvec.mul %248, %233, %211 : <f32>
                    ascvec.storealign %arg109[%243], %249, %211 : memref<16384xf32, 1>, index, <f32>
                    %250 = arith.addi %234, %c128_i32 overflow<nsw> : i32
                    %251 = arith.index_cast %250 : i32 to index
                    %252 = arith.addi %251, %c8192 : index
                    %253 = ascvec.loadalign %arg108[%252] : memref<16384xf32, 1>, index -> <f32>
                    %254 = ascvec.loadalign %arg109[%251] : memref<16384xf32, 1>, index -> <f32>
                    %255 = ascvec.mul %254, %226, %211 : <f32>
                    %256 = ascvec.add %255, %253, %211 : <f32>
                    %257 = ascvec.mul %256, %233, %211 : <f32>
                    ascvec.storealign %arg109[%251], %257, %211 : memref<16384xf32, 1>, index, <f32>
                    %258 = arith.addi %234, %c192_i32 overflow<nsw> : i32
                    %259 = arith.index_cast %258 : i32 to index
                    %260 = arith.addi %259, %c8192 : index
                    %261 = ascvec.loadalign %arg108[%260] : memref<16384xf32, 1>, index -> <f32>
                    %262 = ascvec.loadalign %arg109[%259] : memref<16384xf32, 1>, index -> <f32>
                    %263 = ascvec.mul %262, %226, %211 : <f32>
                    %264 = ascvec.add %263, %261, %211 : <f32>
                    %265 = ascvec.mul %264, %233, %211 : <f32>
                    ascvec.storealign %arg109[%259], %265, %211 : memref<16384xf32, 1>, index, <f32>
                  } {cannir.loop_unroll = 1 : i64}
                } {cannir.vf_mode = "simd"}
                ascvec.release_buf v, %c5_i64, 0 {cannir.buffer.generated_sync}
                %210 = arith.cmpi sle, %59, %c0_i64 : i64
                scf.if %210 {
                  ascvec.sync_intra_arrive v, %c1_i64 {cannir.sync.banks = 3 : i32}
                }
              } else {
                ascvec.sync_intra_wait v, %c1_i64 {cannir.sync.banks = 3 : i32}
                ascvec.get_buf v, %c5_i64, 0 {cannir.buffer.generated_sync}
                %210 = ascscalar.bitcast_memref %alloc_24 byte_offset = %c0_i64 : memref<16384xf32, 1> -> memref<?xbf16, 1>
                ascvec.vf(%152, %154, %alloc_21, %alloc_20, %alloc_24, %alloc_24, %210 : i64, i64, memref<256xf32, 1>, memref<128xf32, 1>, memref<16384xf32, 1>, memref<16384xf32, 1>, memref<?xbf16, 1>) {
                ^bb0(%arg104: i64, %arg105: i64, %arg106: memref<256xf32, 1>, %arg107: memref<128xf32, 1>, %arg108: memref<16384xf32, 1>, %arg109: memref<16384xf32, 1>, %arg110: memref<?xbf16, 1>):
                  %c1_i32_48 = arith.constant 1 : i32
                  %c32_i32_49 = arith.constant 32 : i32
                  %c0_i32_50 = arith.constant 0 : i32
                  %c192_i32 = arith.constant 192 : i32
                  %c128_i32 = arith.constant 128 : i32
                  %c64_i32 = arith.constant 64 : i32
                  %c16384 = arith.constant 16384 : index
                  %c512_i32_51 = arith.constant 512 : i32
                  %c8192 = arith.constant 8192 : index
                  %c256_i32 = arith.constant 256 : i32
                  %cst = arith.constant 1.000000e+00 : f32
                  %c2_i32_52 = arith.constant 2 : i32
                  %c64_i64_53 = arith.constant 64 : i64
                  %c128_i64_54 = arith.constant 128 : i64
                  %c0_i64_55 = arith.constant 0 : i64
                  %c2_i64_56 = arith.constant 2 : i64
                  %212 = ascvec.create_mask : !ascvec.mask
                  scf.for %arg111 = %c0_i32_50 to %c32_i32_49 step %c1_i32_48  : i32 {
                    %213 = arith.remsi %arg104, %c2_i64_56 : i64
                    %214 = arith.cmpi ne, %213, %c0_i64_55 : i64
                    %215 = arith.cmpi slt, %213, %c0_i64_55 : i64
                    %216 = arith.andi %214, %215 : i1
                    %217 = arith.select %216, %c2_i64_56, %c0_i64_55 : i64
                    %218 = arith.addi %213, %217 overflow<nsw> : i64
                    %219 = arith.muli %218, %c128_i64_54 overflow<nsw> : i64
                    %220 = arith.remsi %arg105, %c2_i64_56 : i64
                    %221 = arith.muli %220, %c64_i64_53 overflow<nsw> : i64
                    %222 = arith.addi %219, %221 overflow<nsw> : i64
                    %223 = arith.muli %arg111, %c2_i32_52 overflow<nsw> : i32
                    %224 = arith.extsi %223 : i32 to i64
                    %225 = arith.addi %222, %224 overflow<nsw> : i64
                    %226 = arith.index_cast %225 : i64 to index
                    %227 = ascvec.loadalign_brc %arg106[%226] : memref<256xf32, 1>, index -> <f32>
                    %228 = ascvec.duplicate_scalar_nomask %cst : f32 -> <f32>
                    %229 = arith.muli %218, %c64_i64_53 overflow<nsw> : i64
                    %230 = arith.extsi %arg111 : i32 to i64
                    %231 = arith.addi %229, %230 overflow<nsw> : i64
                    %232 = arith.index_cast %231 : i64 to index
                    %233 = ascvec.loadalign_brc %arg107[%232] : memref<128xf32, 1>, index -> <f32>
                    %234 = ascvec.div %228, %233, %212 : <f32>
                    %235 = arith.muli %arg111, %c256_i32 overflow<nsw> : i32
                    %236 = arith.index_cast %235 : i32 to index
                    %237 = arith.addi %236, %c8192 : index
                    %238 = ascvec.loadalign %arg108[%237] : memref<16384xf32, 1>, index -> <f32>
                    %239 = ascvec.loadalign %arg109[%236] : memref<16384xf32, 1>, index -> <f32>
                    %240 = ascvec.mul %239, %227, %212 : <f32>
                    %241 = ascvec.add %240, %238, %212 : <f32>
                    %242 = ascvec.mul %241, %234, %212 : <f32>
                    %243 = ascvec.cast %242, %212 : <f32> -> <bf16>
                    %244 = arith.muli %arg111, %c512_i32_51 overflow<nsw> : i32
                    %245 = arith.index_cast %244 : i32 to index
                    %246 = arith.addi %245, %c16384 : index
                    ascvec.storealign_pack %arg110[%246], %243, %212 {mode = "b32_to_b16"} : memref<?xbf16, 1>, index, <bf16>
                    %247 = arith.addi %235, %c64_i32 overflow<nsw> : i32
                    %248 = arith.index_cast %247 : i32 to index
                    %249 = arith.addi %248, %c8192 : index
                    %250 = ascvec.loadalign %arg108[%249] : memref<16384xf32, 1>, index -> <f32>
                    %251 = ascvec.loadalign %arg109[%248] : memref<16384xf32, 1>, index -> <f32>
                    %252 = ascvec.mul %251, %227, %212 : <f32>
                    %253 = ascvec.add %252, %250, %212 : <f32>
                    %254 = ascvec.mul %253, %234, %212 : <f32>
                    %255 = ascvec.cast %254, %212 : <f32> -> <bf16>
                    %256 = arith.addi %244, %c64_i32 overflow<nsw> : i32
                    %257 = arith.index_cast %256 : i32 to index
                    %258 = arith.addi %257, %c16384 : index
                    ascvec.storealign_pack %arg110[%258], %255, %212 {mode = "b32_to_b16"} : memref<?xbf16, 1>, index, <bf16>
                    %259 = arith.addi %235, %c128_i32 overflow<nsw> : i32
                    %260 = arith.index_cast %259 : i32 to index
                    %261 = arith.addi %260, %c8192 : index
                    %262 = ascvec.loadalign %arg108[%261] : memref<16384xf32, 1>, index -> <f32>
                    %263 = ascvec.loadalign %arg109[%260] : memref<16384xf32, 1>, index -> <f32>
                    %264 = ascvec.mul %263, %227, %212 : <f32>
                    %265 = ascvec.add %264, %262, %212 : <f32>
                    %266 = ascvec.mul %265, %234, %212 : <f32>
                    %267 = ascvec.cast %266, %212 : <f32> -> <bf16>
                    %268 = arith.addi %244, %c128_i32 overflow<nsw> : i32
                    %269 = arith.index_cast %268 : i32 to index
                    %270 = arith.addi %269, %c16384 : index
                    ascvec.storealign_pack %arg110[%270], %267, %212 {mode = "b32_to_b16"} : memref<?xbf16, 1>, index, <bf16>
                    %271 = arith.addi %235, %c192_i32 overflow<nsw> : i32
                    %272 = arith.index_cast %271 : i32 to index
                    %273 = arith.addi %272, %c8192 : index
                    %274 = ascvec.loadalign %arg108[%273] : memref<16384xf32, 1>, index -> <f32>
                    %275 = ascvec.loadalign %arg109[%272] : memref<16384xf32, 1>, index -> <f32>
                    %276 = ascvec.mul %275, %227, %212 : <f32>
                    %277 = ascvec.add %276, %274, %212 : <f32>
                    %278 = ascvec.mul %277, %234, %212 : <f32>
                    %279 = ascvec.cast %278, %212 : <f32> -> <bf16>
                    %280 = arith.addi %244, %c192_i32 overflow<nsw> : i32
                    %281 = arith.index_cast %280 : i32 to index
                    %282 = arith.addi %281, %c16384 : index
                    ascvec.storealign_pack %arg110[%282], %279, %212 {mode = "b32_to_b16"} : memref<?xbf16, 1>, index, <bf16>
                  } {cannir.loop_unroll = 1 : i64}
                } {cannir.vf_mode = "simd"}
                ascvec.release_buf v, %c5_i64, 0 {cannir.buffer.generated_sync}
                %211 = arith.cmpi sle, %59, %c0_i64 : i64
                scf.if %211 {
                  ascvec.sync_intra_arrive v, %c1_i64 {cannir.sync.banks = 3 : i32}
                }
              }
            } else {
              %209 = arith.cmpi sge, %201, %c0_i64 : i64
              scf.if %209 {
                ascvec.sync_intra_wait v, %c1_i64 {cannir.sync.banks = 3 : i32}
                ascvec.get_buf v, %c5_i64, 0 {cannir.buffer.generated_sync}
                ascvec.vf(%152, %154, %alloc_21, %alloc_24, %alloc_24 : i64, i64, memref<256xf32, 1>, memref<16384xf32, 1>, memref<16384xf32, 1>) {
                ^bb0(%arg104: i64, %arg105: i64, %arg106: memref<256xf32, 1>, %arg107: memref<16384xf32, 1>, %arg108: memref<16384xf32, 1>):
                  %c1_i32_48 = arith.constant 1 : i32
                  %c32_i32_49 = arith.constant 32 : i32
                  %c0_i32_50 = arith.constant 0 : i32
                  %c192_i32 = arith.constant 192 : i32
                  %c128_i32 = arith.constant 128 : i32
                  %c64_i32 = arith.constant 64 : i32
                  %c8192 = arith.constant 8192 : index
                  %c256_i32 = arith.constant 256 : i32
                  %c2_i32_51 = arith.constant 2 : i32
                  %c64_i64_52 = arith.constant 64 : i64
                  %c128_i64_53 = arith.constant 128 : i64
                  %c0_i64_54 = arith.constant 0 : i64
                  %c2_i64_55 = arith.constant 2 : i64
                  %210 = ascvec.create_mask : !ascvec.mask
                  scf.for %arg109 = %c0_i32_50 to %c32_i32_49 step %c1_i32_48  : i32 {
                    %211 = arith.remsi %arg104, %c2_i64_55 : i64
                    %212 = arith.cmpi ne, %211, %c0_i64_54 : i64
                    %213 = arith.cmpi slt, %211, %c0_i64_54 : i64
                    %214 = arith.andi %212, %213 : i1
                    %215 = arith.select %214, %c2_i64_55, %c0_i64_54 : i64
                    %216 = arith.addi %211, %215 overflow<nsw> : i64
                    %217 = arith.muli %216, %c128_i64_53 overflow<nsw> : i64
                    %218 = arith.remsi %arg105, %c2_i64_55 : i64
                    %219 = arith.muli %218, %c64_i64_52 overflow<nsw> : i64
                    %220 = arith.addi %217, %219 overflow<nsw> : i64
                    %221 = arith.muli %arg109, %c2_i32_51 overflow<nsw> : i32
                    %222 = arith.extsi %221 : i32 to i64
                    %223 = arith.addi %220, %222 overflow<nsw> : i64
                    %224 = arith.index_cast %223 : i64 to index
                    %225 = ascvec.loadalign_brc %arg106[%224] : memref<256xf32, 1>, index -> <f32>
                    %226 = arith.muli %arg109, %c256_i32 overflow<nsw> : i32
                    %227 = arith.index_cast %226 : i32 to index
                    %228 = arith.addi %227, %c8192 : index
                    %229 = ascvec.loadalign %arg107[%228] : memref<16384xf32, 1>, index -> <f32>
                    %230 = ascvec.loadalign %arg108[%227] : memref<16384xf32, 1>, index -> <f32>
                    %231 = ascvec.mul %230, %225, %210 : <f32>
                    %232 = ascvec.add %231, %229, %210 : <f32>
                    ascvec.storealign %arg108[%227], %232, %210 : memref<16384xf32, 1>, index, <f32>
                    %233 = arith.addi %226, %c64_i32 overflow<nsw> : i32
                    %234 = arith.index_cast %233 : i32 to index
                    %235 = arith.addi %234, %c8192 : index
                    %236 = ascvec.loadalign %arg107[%235] : memref<16384xf32, 1>, index -> <f32>
                    %237 = ascvec.loadalign %arg108[%234] : memref<16384xf32, 1>, index -> <f32>
                    %238 = ascvec.mul %237, %225, %210 : <f32>
                    %239 = ascvec.add %238, %236, %210 : <f32>
                    ascvec.storealign %arg108[%234], %239, %210 : memref<16384xf32, 1>, index, <f32>
                    %240 = arith.addi %226, %c128_i32 overflow<nsw> : i32
                    %241 = arith.index_cast %240 : i32 to index
                    %242 = arith.addi %241, %c8192 : index
                    %243 = ascvec.loadalign %arg107[%242] : memref<16384xf32, 1>, index -> <f32>
                    %244 = ascvec.loadalign %arg108[%241] : memref<16384xf32, 1>, index -> <f32>
                    %245 = ascvec.mul %244, %225, %210 : <f32>
                    %246 = ascvec.add %245, %243, %210 : <f32>
                    ascvec.storealign %arg108[%241], %246, %210 : memref<16384xf32, 1>, index, <f32>
                    %247 = arith.addi %226, %c192_i32 overflow<nsw> : i32
                    %248 = arith.index_cast %247 : i32 to index
                    %249 = arith.addi %248, %c8192 : index
                    %250 = ascvec.loadalign %arg107[%249] : memref<16384xf32, 1>, index -> <f32>
                    %251 = ascvec.loadalign %arg108[%248] : memref<16384xf32, 1>, index -> <f32>
                    %252 = ascvec.mul %251, %225, %210 : <f32>
                    %253 = ascvec.add %252, %250, %210 : <f32>
                    ascvec.storealign %arg108[%248], %253, %210 : memref<16384xf32, 1>, index, <f32>
                  } {cannir.loop_unroll = 1 : i64}
                } {cannir.vf_mode = "simd"}
                ascvec.release_buf v, %c5_i64, 0 {cannir.buffer.generated_sync}
                ascvec.sync_intra_arrive v, %c1_i64 {cannir.sync.banks = 3 : i32}
              } else {
                ascvec.sync_intra_wait v, %c1_i64 {cannir.sync.banks = 3 : i32}
                ascvec.get_buf v, %c5_i64, 0 {cannir.buffer.generated_sync}
                ascvec.vf(%152, %154, %alloc_21, %alloc_24, %alloc_24 : i64, i64, memref<256xf32, 1>, memref<16384xf32, 1>, memref<16384xf32, 1>) {
                ^bb0(%arg104: i64, %arg105: i64, %arg106: memref<256xf32, 1>, %arg107: memref<16384xf32, 1>, %arg108: memref<16384xf32, 1>):
                  %c1_i32_48 = arith.constant 1 : i32
                  %c32_i32_49 = arith.constant 32 : i32
                  %c0_i32_50 = arith.constant 0 : i32
                  %c192_i32 = arith.constant 192 : i32
                  %c128_i32 = arith.constant 128 : i32
                  %c64_i32 = arith.constant 64 : i32
                  %c8192 = arith.constant 8192 : index
                  %c256_i32 = arith.constant 256 : i32
                  %c2_i32_51 = arith.constant 2 : i32
                  %c64_i64_52 = arith.constant 64 : i64
                  %c128_i64_53 = arith.constant 128 : i64
                  %c0_i64_54 = arith.constant 0 : i64
                  %c2_i64_55 = arith.constant 2 : i64
                  %210 = ascvec.create_mask : !ascvec.mask
                  scf.for %arg109 = %c0_i32_50 to %c32_i32_49 step %c1_i32_48  : i32 {
                    %211 = arith.remsi %arg104, %c2_i64_55 : i64
                    %212 = arith.cmpi ne, %211, %c0_i64_54 : i64
                    %213 = arith.cmpi slt, %211, %c0_i64_54 : i64
                    %214 = arith.andi %212, %213 : i1
                    %215 = arith.select %214, %c2_i64_55, %c0_i64_54 : i64
                    %216 = arith.addi %211, %215 overflow<nsw> : i64
                    %217 = arith.muli %216, %c128_i64_53 overflow<nsw> : i64
                    %218 = arith.remsi %arg105, %c2_i64_55 : i64
                    %219 = arith.muli %218, %c64_i64_52 overflow<nsw> : i64
                    %220 = arith.addi %217, %219 overflow<nsw> : i64
                    %221 = arith.muli %arg109, %c2_i32_51 overflow<nsw> : i32
                    %222 = arith.extsi %221 : i32 to i64
                    %223 = arith.addi %220, %222 overflow<nsw> : i64
                    %224 = arith.index_cast %223 : i64 to index
                    %225 = ascvec.loadalign_brc %arg106[%224] : memref<256xf32, 1>, index -> <f32>
                    %226 = arith.muli %arg109, %c256_i32 overflow<nsw> : i32
                    %227 = arith.index_cast %226 : i32 to index
                    %228 = arith.addi %227, %c8192 : index
                    %229 = ascvec.loadalign %arg107[%228] : memref<16384xf32, 1>, index -> <f32>
                    %230 = ascvec.loadalign %arg108[%227] : memref<16384xf32, 1>, index -> <f32>
                    %231 = ascvec.mul %230, %225, %210 : <f32>
                    %232 = ascvec.add %231, %229, %210 : <f32>
                    ascvec.storealign %arg108[%227], %232, %210 : memref<16384xf32, 1>, index, <f32>
                    %233 = arith.addi %226, %c64_i32 overflow<nsw> : i32
                    %234 = arith.index_cast %233 : i32 to index
                    %235 = arith.addi %234, %c8192 : index
                    %236 = ascvec.loadalign %arg107[%235] : memref<16384xf32, 1>, index -> <f32>
                    %237 = ascvec.loadalign %arg108[%234] : memref<16384xf32, 1>, index -> <f32>
                    %238 = ascvec.mul %237, %225, %210 : <f32>
                    %239 = ascvec.add %238, %236, %210 : <f32>
                    ascvec.storealign %arg108[%234], %239, %210 : memref<16384xf32, 1>, index, <f32>
                    %240 = arith.addi %226, %c128_i32 overflow<nsw> : i32
                    %241 = arith.index_cast %240 : i32 to index
                    %242 = arith.addi %241, %c8192 : index
                    %243 = ascvec.loadalign %arg107[%242] : memref<16384xf32, 1>, index -> <f32>
                    %244 = ascvec.loadalign %arg108[%241] : memref<16384xf32, 1>, index -> <f32>
                    %245 = ascvec.mul %244, %225, %210 : <f32>
                    %246 = ascvec.add %245, %243, %210 : <f32>
                    ascvec.storealign %arg108[%241], %246, %210 : memref<16384xf32, 1>, index, <f32>
                    %247 = arith.addi %226, %c192_i32 overflow<nsw> : i32
                    %248 = arith.index_cast %247 : i32 to index
                    %249 = arith.addi %248, %c8192 : index
                    %250 = ascvec.loadalign %arg107[%249] : memref<16384xf32, 1>, index -> <f32>
                    %251 = ascvec.loadalign %arg108[%248] : memref<16384xf32, 1>, index -> <f32>
                    %252 = ascvec.mul %251, %225, %210 : <f32>
                    %253 = ascvec.add %252, %250, %210 : <f32>
                    ascvec.storealign %arg108[%248], %253, %210 : memref<16384xf32, 1>, index, <f32>
                  } {cannir.loop_unroll = 1 : i64}
                } {cannir.vf_mode = "simd"}
                ascvec.release_buf v, %c5_i64, 0 {cannir.buffer.generated_sync}
                ascvec.sync_intra_arrive v, %c1_i64 {cannir.sync.banks = 3 : i32}
              }
            }
          }
          %205 = arith.cmpi sge, %201, %c0_i64 : i64
          %206 = arith.select %203, %205, %false : i1
          scf.if %206 {
            %209 = cannir.datastruct_get %arg22["num_heads_q"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
            %210 = arith.cmpi sgt, %59, %c0_i64 : i64
            scf.if %210 {
              %211 = arith.extsi %209 : i32 to i64
              %212 = arith.muli %201, %211 : i64
              %213 = arith.addi %212, %52 : i64
              %214 = arith.muli %213, %c16384_i64 : i64
              %215 = ascscalar.memref_address %alloc_15 : memref<1xf32>
              %216 = arith.divui %214, %c8_i64 : i64
              %217 = arith.addi %215, %216 : i64
              %218 = ascscalar.bind_gm_ptr %217 : memref<1xf32>
              %219 = ascscalar.bitcast_memref %alloc_24 byte_offset = %c0_i64 : memref<16384xf32, 1> -> memref<?xf32, 1>
              %220 = arith.cmpi ugt, %59, %c0_i64 : i64
              scf.if %220 {
                ascvec.get_buf mte3, %c5_i64, 0 {cannir.buffer.generated_sync}
                %221 = arith.trunci %59 : i64 to i32
                ascvec.copy_ub2gm %218[%c0], %219[%c0], %221, %c1024_i32, %c2048_i64, %c1024_i32 l2_cache_ctl %c0_i32 : memref<1xf32>, memref<?xf32, 1>
                ascvec.release_buf mte3, %c5_i64, 0 {cannir.buffer.generated_sync}
              }
              ascvec.get_buf mte3, %c5_i64, 0 {cannir.buffer.handoff_dependency}
              ascvec.release_buf mte3, %c5_i64, 0 {cannir.buffer.handoff_dependency}
              ascvec.sync_intra_arrive mte3, %c1_i64 {cannir.sync.banks = 3 : i32}
            }
          }
          %207 = arith.cmpi slt, %201, %c0_i64 : i64
          %208 = arith.andi %207, %203 : i1
          scf.if %208 {
            %209 = arith.cmpi sgt, %59, %c0_i64 : i64
            scf.if %209 {
              %210 = cannir.datastruct_get %arg22["num_heads_q"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
              %211 = arith.extsi %210 : i32 to i64
              %212 = arith.muli %153, %211 : i64
              %213 = arith.muli %212, %c8192_i64 : i64
              %214 = arith.muli %51, %c262144_i64 : i64
              %215 = arith.addi %213, %214 : i64
              %216 = ascscalar.memref_address %alloc_13 : memref<1xbf16>
              %217 = arith.divui %215, %c8_i64 : i64
              %218 = arith.addi %216, %217 : i64
              %219 = ascscalar.bind_gm_ptr %218 : memref<1xbf16>
              %220 = ascscalar.memref_address %alloc_24 : memref<16384xf32, 1>
              %221 = arith.addi %220, %c32768_i64 : i64
              %222 = ascscalar.bind_local_ptr %221 : memref<1xbf16, 1>
              %223 = arith.cmpi ugt, %59, %c0_i64 : i64
              scf.if %223 {
                ascvec.get_buf mte3, %c5_i64, 0 {cannir.buffer.generated_sync}
                %224 = arith.trunci %59 : i64 to i32
                ascvec.copy_ub2gm %219[%c0], %222[%c0], %224, %c512_i32, %c1024_i64, %c1024_i32 l2_cache_ctl %c4_i32 : memref<1xbf16>, memref<1xbf16, 1>
                ascvec.release_buf mte3, %c5_i64, 0 {cannir.buffer.generated_sync}
              }
              ascvec.get_buf mte3, %c5_i64, 0 {cannir.buffer.handoff_dependency}
              ascvec.release_buf mte3, %c5_i64, 0 {cannir.buffer.handoff_dependency}
              ascvec.sync_intra_arrive mte3, %c1_i64 {cannir.sync.banks = 3 : i32}
            }
          }
          scf.if %204 {
            ascvec.sync_intra_wait v, %c2_i64 {cannir.sync.banks = 3 : i32}
            ascvec.get_buf v, %c6_i64, 0 {cannir.buffer.generated_sync}
            ascvec.vf(%alloc_25 : memref<16384xf32, 1>) {
            ^bb0(%arg104: memref<16384xf32, 1>):
              %c0_48 = arith.constant 0 : index
              %c1_i32_49 = arith.constant 1 : i32
              %c0_50 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              scf.for %arg105 = %c0_50 to %c1 step %c1 {
                %mask, %remaining_out = ascvec.update_mask %c1_i32_49 : !ascvec.mask
                %210 = ascvec.loadalign %arg104[%c0_48] : memref<16384xf32, 1>, index -> <f32>
                ascvec.storealign %arg104[%c0_48], %210, %mask : memref<16384xf32, 1>, index, <f32>
              }
            } {cannir.vf_mode = "simd"}
            ascvec.release_buf v, %c6_i64, 0 {cannir.buffer.generated_sync}
            %209 = arith.cmpi ne, %202, %155 : i64
            scf.if %209 {
              ascvec.sync_intra_arrive v, %c2_i64 {cannir.sync.banks = 3 : i32}
            }
            scf.if %203 {
              scf.if %205 {
                ascvec.get_buf v, %c6_i64, 0 {cannir.buffer.generated_sync}
                ascvec.vf(%152, %alloc_20, %alloc_25 : i64, memref<128xf32, 1>, memref<16384xf32, 1>) {
                ^bb0(%arg104: i64, %arg105: memref<128xf32, 1>, %arg106: memref<16384xf32, 1>):
                  %c1_i32_48 = arith.constant 1 : i32
                  %c32_i32_49 = arith.constant 32 : i32
                  %c0_i32_50 = arith.constant 0 : i32
                  %c192_i32 = arith.constant 192 : i32
                  %c128_i32 = arith.constant 128 : i32
                  %c64_i32 = arith.constant 64 : i32
                  %c256_i32 = arith.constant 256 : i32
                  %c64_i64_51 = arith.constant 64 : i64
                  %c0_i64_52 = arith.constant 0 : i64
                  %c2_i64_53 = arith.constant 2 : i64
                  %cst = arith.constant 1.000000e+00 : f32
                  %211 = ascvec.create_mask : !ascvec.mask
                  scf.for %arg107 = %c0_i32_50 to %c32_i32_49 step %c1_i32_48  : i32 {
                    %212 = ascvec.duplicate_scalar_nomask %cst : f32 -> <f32>
                    %213 = arith.remsi %arg104, %c2_i64_53 : i64
                    %214 = arith.cmpi ne, %213, %c0_i64_52 : i64
                    %215 = arith.cmpi slt, %213, %c0_i64_52 : i64
                    %216 = arith.andi %214, %215 : i1
                    %217 = arith.select %216, %c2_i64_53, %c0_i64_52 : i64
                    %218 = arith.addi %213, %217 overflow<nsw> : i64
                    %219 = arith.muli %218, %c64_i64_51 overflow<nsw> : i64
                    %220 = arith.extsi %arg107 : i32 to i64
                    %221 = arith.addi %219, %220 overflow<nsw> : i64
                    %222 = arith.index_cast %221 : i64 to index
                    %223 = ascvec.loadalign_brc %arg105[%222] : memref<128xf32, 1>, index -> <f32>
                    %224 = ascvec.div %212, %223, %211 : <f32>
                    %225 = arith.muli %arg107, %c256_i32 overflow<nsw> : i32
                    %226 = arith.index_cast %225 : i32 to index
                    %227 = ascvec.loadalign %arg106[%226] : memref<16384xf32, 1>, index -> <f32>
                    %228 = ascvec.mul %227, %224, %211 : <f32>
                    ascvec.storealign %arg106[%226], %228, %211 : memref<16384xf32, 1>, index, <f32>
                    %229 = arith.addi %225, %c64_i32 overflow<nsw> : i32
                    %230 = arith.index_cast %229 : i32 to index
                    %231 = ascvec.loadalign %arg106[%230] : memref<16384xf32, 1>, index -> <f32>
                    %232 = ascvec.mul %231, %224, %211 : <f32>
                    ascvec.storealign %arg106[%230], %232, %211 : memref<16384xf32, 1>, index, <f32>
                    %233 = arith.addi %225, %c128_i32 overflow<nsw> : i32
                    %234 = arith.index_cast %233 : i32 to index
                    %235 = ascvec.loadalign %arg106[%234] : memref<16384xf32, 1>, index -> <f32>
                    %236 = ascvec.mul %235, %224, %211 : <f32>
                    ascvec.storealign %arg106[%234], %236, %211 : memref<16384xf32, 1>, index, <f32>
                    %237 = arith.addi %225, %c192_i32 overflow<nsw> : i32
                    %238 = arith.index_cast %237 : i32 to index
                    %239 = ascvec.loadalign %arg106[%238] : memref<16384xf32, 1>, index -> <f32>
                    %240 = ascvec.mul %239, %224, %211 : <f32>
                    ascvec.storealign %arg106[%238], %240, %211 : memref<16384xf32, 1>, index, <f32>
                  } {cannir.loop_unroll = 1 : i64}
                } {cannir.vf_mode = "simd"}
                ascvec.release_buf v, %c6_i64, 0 {cannir.buffer.generated_sync}
                %210 = arith.cmpi sle, %59, %c0_i64 : i64
                scf.if %210 {
                  ascvec.sync_intra_arrive v, %c2_i64 {cannir.sync.banks = 3 : i32}
                }
              } else {
                ascvec.get_buf v, %c6_i64, 0 {cannir.buffer.generated_sync}
                %210 = ascscalar.bitcast_memref %alloc_25 byte_offset = %c0_i64 : memref<16384xf32, 1> -> memref<?xbf16, 1>
                ascvec.vf(%152, %alloc_20, %alloc_25, %210 : i64, memref<128xf32, 1>, memref<16384xf32, 1>, memref<?xbf16, 1>) {
                ^bb0(%arg104: i64, %arg105: memref<128xf32, 1>, %arg106: memref<16384xf32, 1>, %arg107: memref<?xbf16, 1>):
                  %c1_i32_48 = arith.constant 1 : i32
                  %c32_i32_49 = arith.constant 32 : i32
                  %c0_i32_50 = arith.constant 0 : i32
                  %c192_i32 = arith.constant 192 : i32
                  %c128_i32 = arith.constant 128 : i32
                  %c64_i32 = arith.constant 64 : i32
                  %c16384 = arith.constant 16384 : index
                  %c512_i32_51 = arith.constant 512 : i32
                  %c256_i32 = arith.constant 256 : i32
                  %c64_i64_52 = arith.constant 64 : i64
                  %c0_i64_53 = arith.constant 0 : i64
                  %c2_i64_54 = arith.constant 2 : i64
                  %cst = arith.constant 1.000000e+00 : f32
                  %212 = ascvec.create_mask : !ascvec.mask
                  scf.for %arg108 = %c0_i32_50 to %c32_i32_49 step %c1_i32_48  : i32 {
                    %213 = ascvec.duplicate_scalar_nomask %cst : f32 -> <f32>
                    %214 = arith.remsi %arg104, %c2_i64_54 : i64
                    %215 = arith.cmpi ne, %214, %c0_i64_53 : i64
                    %216 = arith.cmpi slt, %214, %c0_i64_53 : i64
                    %217 = arith.andi %215, %216 : i1
                    %218 = arith.select %217, %c2_i64_54, %c0_i64_53 : i64
                    %219 = arith.addi %214, %218 overflow<nsw> : i64
                    %220 = arith.muli %219, %c64_i64_52 overflow<nsw> : i64
                    %221 = arith.extsi %arg108 : i32 to i64
                    %222 = arith.addi %220, %221 overflow<nsw> : i64
                    %223 = arith.index_cast %222 : i64 to index
                    %224 = ascvec.loadalign_brc %arg105[%223] : memref<128xf32, 1>, index -> <f32>
                    %225 = ascvec.div %213, %224, %212 : <f32>
                    %226 = arith.muli %arg108, %c256_i32 overflow<nsw> : i32
                    %227 = arith.index_cast %226 : i32 to index
                    %228 = ascvec.loadalign %arg106[%227] : memref<16384xf32, 1>, index -> <f32>
                    %229 = ascvec.mul %228, %225, %212 : <f32>
                    %230 = ascvec.cast %229, %212 : <f32> -> <bf16>
                    %231 = arith.muli %arg108, %c512_i32_51 overflow<nsw> : i32
                    %232 = arith.index_cast %231 : i32 to index
                    %233 = arith.addi %232, %c16384 : index
                    ascvec.storealign_pack %arg107[%233], %230, %212 {mode = "b32_to_b16"} : memref<?xbf16, 1>, index, <bf16>
                    %234 = arith.addi %226, %c64_i32 overflow<nsw> : i32
                    %235 = arith.index_cast %234 : i32 to index
                    %236 = ascvec.loadalign %arg106[%235] : memref<16384xf32, 1>, index -> <f32>
                    %237 = ascvec.mul %236, %225, %212 : <f32>
                    %238 = ascvec.cast %237, %212 : <f32> -> <bf16>
                    %239 = arith.addi %231, %c64_i32 overflow<nsw> : i32
                    %240 = arith.index_cast %239 : i32 to index
                    %241 = arith.addi %240, %c16384 : index
                    ascvec.storealign_pack %arg107[%241], %238, %212 {mode = "b32_to_b16"} : memref<?xbf16, 1>, index, <bf16>
                    %242 = arith.addi %226, %c128_i32 overflow<nsw> : i32
                    %243 = arith.index_cast %242 : i32 to index
                    %244 = ascvec.loadalign %arg106[%243] : memref<16384xf32, 1>, index -> <f32>
                    %245 = ascvec.mul %244, %225, %212 : <f32>
                    %246 = ascvec.cast %245, %212 : <f32> -> <bf16>
                    %247 = arith.addi %231, %c128_i32 overflow<nsw> : i32
                    %248 = arith.index_cast %247 : i32 to index
                    %249 = arith.addi %248, %c16384 : index
                    ascvec.storealign_pack %arg107[%249], %246, %212 {mode = "b32_to_b16"} : memref<?xbf16, 1>, index, <bf16>
                    %250 = arith.addi %226, %c192_i32 overflow<nsw> : i32
                    %251 = arith.index_cast %250 : i32 to index
                    %252 = ascvec.loadalign %arg106[%251] : memref<16384xf32, 1>, index -> <f32>
                    %253 = ascvec.mul %252, %225, %212 : <f32>
                    %254 = ascvec.cast %253, %212 : <f32> -> <bf16>
                    %255 = arith.addi %231, %c192_i32 overflow<nsw> : i32
                    %256 = arith.index_cast %255 : i32 to index
                    %257 = arith.addi %256, %c16384 : index
                    ascvec.storealign_pack %arg107[%257], %254, %212 {mode = "b32_to_b16"} : memref<?xbf16, 1>, index, <bf16>
                  } {cannir.loop_unroll = 1 : i64}
                } {cannir.vf_mode = "simd"}
                ascvec.release_buf v, %c6_i64, 0 {cannir.buffer.generated_sync}
                %211 = arith.cmpi sle, %59, %c0_i64 : i64
                scf.if %211 {
                  ascvec.sync_intra_arrive v, %c2_i64 {cannir.sync.banks = 3 : i32}
                }
              }
            }
          } else {
            scf.if %203 {
              scf.if %205 {
                ascvec.sync_intra_wait v, %c2_i64 {cannir.sync.banks = 3 : i32}
                ascvec.get_buf v, %c6_i64, 0 {cannir.buffer.generated_sync}
                ascvec.vf(%152, %154, %alloc_21, %alloc_20, %alloc_25, %alloc_25 : i64, i64, memref<256xf32, 1>, memref<128xf32, 1>, memref<16384xf32, 1>, memref<16384xf32, 1>) {
                ^bb0(%arg104: i64, %arg105: i64, %arg106: memref<256xf32, 1>, %arg107: memref<128xf32, 1>, %arg108: memref<16384xf32, 1>, %arg109: memref<16384xf32, 1>):
                  %c1_i32_48 = arith.constant 1 : i32
                  %c32_i32_49 = arith.constant 32 : i32
                  %c0_i32_50 = arith.constant 0 : i32
                  %c192_i32 = arith.constant 192 : i32
                  %c128_i32 = arith.constant 128 : i32
                  %c64_i32 = arith.constant 64 : i32
                  %c8192 = arith.constant 8192 : index
                  %c256_i32 = arith.constant 256 : i32
                  %cst = arith.constant 1.000000e+00 : f32
                  %c2_i32_51 = arith.constant 2 : i32
                  %c64_i64_52 = arith.constant 64 : i64
                  %c128_i64_53 = arith.constant 128 : i64
                  %c0_i64_54 = arith.constant 0 : i64
                  %c2_i64_55 = arith.constant 2 : i64
                  %210 = ascvec.create_mask : !ascvec.mask
                  scf.for %arg110 = %c0_i32_50 to %c32_i32_49 step %c1_i32_48  : i32 {
                    %211 = arith.remsi %arg104, %c2_i64_55 : i64
                    %212 = arith.cmpi ne, %211, %c0_i64_54 : i64
                    %213 = arith.cmpi slt, %211, %c0_i64_54 : i64
                    %214 = arith.andi %212, %213 : i1
                    %215 = arith.select %214, %c2_i64_55, %c0_i64_54 : i64
                    %216 = arith.addi %211, %215 overflow<nsw> : i64
                    %217 = arith.muli %216, %c128_i64_53 overflow<nsw> : i64
                    %218 = arith.remsi %arg105, %c2_i64_55 : i64
                    %219 = arith.muli %218, %c64_i64_52 overflow<nsw> : i64
                    %220 = arith.addi %217, %219 overflow<nsw> : i64
                    %221 = arith.muli %arg110, %c2_i32_51 overflow<nsw> : i32
                    %222 = arith.extsi %221 : i32 to i64
                    %223 = arith.addi %220, %222 overflow<nsw> : i64
                    %224 = arith.index_cast %223 : i64 to index
                    %225 = ascvec.loadalign_brc %arg106[%224] : memref<256xf32, 1>, index -> <f32>
                    %226 = ascvec.duplicate_scalar_nomask %cst : f32 -> <f32>
                    %227 = arith.muli %216, %c64_i64_52 overflow<nsw> : i64
                    %228 = arith.extsi %arg110 : i32 to i64
                    %229 = arith.addi %227, %228 overflow<nsw> : i64
                    %230 = arith.index_cast %229 : i64 to index
                    %231 = ascvec.loadalign_brc %arg107[%230] : memref<128xf32, 1>, index -> <f32>
                    %232 = ascvec.div %226, %231, %210 : <f32>
                    %233 = arith.muli %arg110, %c256_i32 overflow<nsw> : i32
                    %234 = arith.index_cast %233 : i32 to index
                    %235 = arith.addi %234, %c8192 : index
                    %236 = ascvec.loadalign %arg108[%235] : memref<16384xf32, 1>, index -> <f32>
                    %237 = ascvec.loadalign %arg109[%234] : memref<16384xf32, 1>, index -> <f32>
                    %238 = ascvec.mul %237, %225, %210 : <f32>
                    %239 = ascvec.add %238, %236, %210 : <f32>
                    %240 = ascvec.mul %239, %232, %210 : <f32>
                    ascvec.storealign %arg109[%234], %240, %210 : memref<16384xf32, 1>, index, <f32>
                    %241 = arith.addi %233, %c64_i32 overflow<nsw> : i32
                    %242 = arith.index_cast %241 : i32 to index
                    %243 = arith.addi %242, %c8192 : index
                    %244 = ascvec.loadalign %arg108[%243] : memref<16384xf32, 1>, index -> <f32>
                    %245 = ascvec.loadalign %arg109[%242] : memref<16384xf32, 1>, index -> <f32>
                    %246 = ascvec.mul %245, %225, %210 : <f32>
                    %247 = ascvec.add %246, %244, %210 : <f32>
                    %248 = ascvec.mul %247, %232, %210 : <f32>
                    ascvec.storealign %arg109[%242], %248, %210 : memref<16384xf32, 1>, index, <f32>
                    %249 = arith.addi %233, %c128_i32 overflow<nsw> : i32
                    %250 = arith.index_cast %249 : i32 to index
                    %251 = arith.addi %250, %c8192 : index
                    %252 = ascvec.loadalign %arg108[%251] : memref<16384xf32, 1>, index -> <f32>
                    %253 = ascvec.loadalign %arg109[%250] : memref<16384xf32, 1>, index -> <f32>
                    %254 = ascvec.mul %253, %225, %210 : <f32>
                    %255 = ascvec.add %254, %252, %210 : <f32>
                    %256 = ascvec.mul %255, %232, %210 : <f32>
                    ascvec.storealign %arg109[%250], %256, %210 : memref<16384xf32, 1>, index, <f32>
                    %257 = arith.addi %233, %c192_i32 overflow<nsw> : i32
                    %258 = arith.index_cast %257 : i32 to index
                    %259 = arith.addi %258, %c8192 : index
                    %260 = ascvec.loadalign %arg108[%259] : memref<16384xf32, 1>, index -> <f32>
                    %261 = ascvec.loadalign %arg109[%258] : memref<16384xf32, 1>, index -> <f32>
                    %262 = ascvec.mul %261, %225, %210 : <f32>
                    %263 = ascvec.add %262, %260, %210 : <f32>
                    %264 = ascvec.mul %263, %232, %210 : <f32>
                    ascvec.storealign %arg109[%258], %264, %210 : memref<16384xf32, 1>, index, <f32>
                  } {cannir.loop_unroll = 1 : i64}
                } {cannir.vf_mode = "simd"}
                ascvec.release_buf v, %c6_i64, 0 {cannir.buffer.generated_sync}
                %209 = arith.cmpi sle, %59, %c0_i64 : i64
                scf.if %209 {
                  ascvec.sync_intra_arrive v, %c2_i64 {cannir.sync.banks = 3 : i32}
                }
              } else {
                ascvec.sync_intra_wait v, %c2_i64 {cannir.sync.banks = 3 : i32}
                ascvec.get_buf v, %c6_i64, 0 {cannir.buffer.generated_sync}
                %209 = ascscalar.bitcast_memref %alloc_25 byte_offset = %c0_i64 : memref<16384xf32, 1> -> memref<?xbf16, 1>
                ascvec.vf(%152, %154, %alloc_21, %alloc_20, %alloc_25, %alloc_25, %209 : i64, i64, memref<256xf32, 1>, memref<128xf32, 1>, memref<16384xf32, 1>, memref<16384xf32, 1>, memref<?xbf16, 1>) {
                ^bb0(%arg104: i64, %arg105: i64, %arg106: memref<256xf32, 1>, %arg107: memref<128xf32, 1>, %arg108: memref<16384xf32, 1>, %arg109: memref<16384xf32, 1>, %arg110: memref<?xbf16, 1>):
                  %c1_i32_48 = arith.constant 1 : i32
                  %c32_i32_49 = arith.constant 32 : i32
                  %c0_i32_50 = arith.constant 0 : i32
                  %c192_i32 = arith.constant 192 : i32
                  %c128_i32 = arith.constant 128 : i32
                  %c64_i32 = arith.constant 64 : i32
                  %c16384 = arith.constant 16384 : index
                  %c512_i32_51 = arith.constant 512 : i32
                  %c8192 = arith.constant 8192 : index
                  %c256_i32 = arith.constant 256 : i32
                  %cst = arith.constant 1.000000e+00 : f32
                  %c2_i32_52 = arith.constant 2 : i32
                  %c64_i64_53 = arith.constant 64 : i64
                  %c128_i64_54 = arith.constant 128 : i64
                  %c0_i64_55 = arith.constant 0 : i64
                  %c2_i64_56 = arith.constant 2 : i64
                  %211 = ascvec.create_mask : !ascvec.mask
                  scf.for %arg111 = %c0_i32_50 to %c32_i32_49 step %c1_i32_48  : i32 {
                    %212 = arith.remsi %arg104, %c2_i64_56 : i64
                    %213 = arith.cmpi ne, %212, %c0_i64_55 : i64
                    %214 = arith.cmpi slt, %212, %c0_i64_55 : i64
                    %215 = arith.andi %213, %214 : i1
                    %216 = arith.select %215, %c2_i64_56, %c0_i64_55 : i64
                    %217 = arith.addi %212, %216 overflow<nsw> : i64
                    %218 = arith.muli %217, %c128_i64_54 overflow<nsw> : i64
                    %219 = arith.remsi %arg105, %c2_i64_56 : i64
                    %220 = arith.muli %219, %c64_i64_53 overflow<nsw> : i64
                    %221 = arith.addi %218, %220 overflow<nsw> : i64
                    %222 = arith.muli %arg111, %c2_i32_52 overflow<nsw> : i32
                    %223 = arith.extsi %222 : i32 to i64
                    %224 = arith.addi %221, %223 overflow<nsw> : i64
                    %225 = arith.index_cast %224 : i64 to index
                    %226 = ascvec.loadalign_brc %arg106[%225] : memref<256xf32, 1>, index -> <f32>
                    %227 = ascvec.duplicate_scalar_nomask %cst : f32 -> <f32>
                    %228 = arith.muli %217, %c64_i64_53 overflow<nsw> : i64
                    %229 = arith.extsi %arg111 : i32 to i64
                    %230 = arith.addi %228, %229 overflow<nsw> : i64
                    %231 = arith.index_cast %230 : i64 to index
                    %232 = ascvec.loadalign_brc %arg107[%231] : memref<128xf32, 1>, index -> <f32>
                    %233 = ascvec.div %227, %232, %211 : <f32>
                    %234 = arith.muli %arg111, %c256_i32 overflow<nsw> : i32
                    %235 = arith.index_cast %234 : i32 to index
                    %236 = arith.addi %235, %c8192 : index
                    %237 = ascvec.loadalign %arg108[%236] : memref<16384xf32, 1>, index -> <f32>
                    %238 = ascvec.loadalign %arg109[%235] : memref<16384xf32, 1>, index -> <f32>
                    %239 = ascvec.mul %238, %226, %211 : <f32>
                    %240 = ascvec.add %239, %237, %211 : <f32>
                    %241 = ascvec.mul %240, %233, %211 : <f32>
                    %242 = ascvec.cast %241, %211 : <f32> -> <bf16>
                    %243 = arith.muli %arg111, %c512_i32_51 overflow<nsw> : i32
                    %244 = arith.index_cast %243 : i32 to index
                    %245 = arith.addi %244, %c16384 : index
                    ascvec.storealign_pack %arg110[%245], %242, %211 {mode = "b32_to_b16"} : memref<?xbf16, 1>, index, <bf16>
                    %246 = arith.addi %234, %c64_i32 overflow<nsw> : i32
                    %247 = arith.index_cast %246 : i32 to index
                    %248 = arith.addi %247, %c8192 : index
                    %249 = ascvec.loadalign %arg108[%248] : memref<16384xf32, 1>, index -> <f32>
                    %250 = ascvec.loadalign %arg109[%247] : memref<16384xf32, 1>, index -> <f32>
                    %251 = ascvec.mul %250, %226, %211 : <f32>
                    %252 = ascvec.add %251, %249, %211 : <f32>
                    %253 = ascvec.mul %252, %233, %211 : <f32>
                    %254 = ascvec.cast %253, %211 : <f32> -> <bf16>
                    %255 = arith.addi %243, %c64_i32 overflow<nsw> : i32
                    %256 = arith.index_cast %255 : i32 to index
                    %257 = arith.addi %256, %c16384 : index
                    ascvec.storealign_pack %arg110[%257], %254, %211 {mode = "b32_to_b16"} : memref<?xbf16, 1>, index, <bf16>
                    %258 = arith.addi %234, %c128_i32 overflow<nsw> : i32
                    %259 = arith.index_cast %258 : i32 to index
                    %260 = arith.addi %259, %c8192 : index
                    %261 = ascvec.loadalign %arg108[%260] : memref<16384xf32, 1>, index -> <f32>
                    %262 = ascvec.loadalign %arg109[%259] : memref<16384xf32, 1>, index -> <f32>
                    %263 = ascvec.mul %262, %226, %211 : <f32>
                    %264 = ascvec.add %263, %261, %211 : <f32>
                    %265 = ascvec.mul %264, %233, %211 : <f32>
                    %266 = ascvec.cast %265, %211 : <f32> -> <bf16>
                    %267 = arith.addi %243, %c128_i32 overflow<nsw> : i32
                    %268 = arith.index_cast %267 : i32 to index
                    %269 = arith.addi %268, %c16384 : index
                    ascvec.storealign_pack %arg110[%269], %266, %211 {mode = "b32_to_b16"} : memref<?xbf16, 1>, index, <bf16>
                    %270 = arith.addi %234, %c192_i32 overflow<nsw> : i32
                    %271 = arith.index_cast %270 : i32 to index
                    %272 = arith.addi %271, %c8192 : index
                    %273 = ascvec.loadalign %arg108[%272] : memref<16384xf32, 1>, index -> <f32>
                    %274 = ascvec.loadalign %arg109[%271] : memref<16384xf32, 1>, index -> <f32>
                    %275 = ascvec.mul %274, %226, %211 : <f32>
                    %276 = ascvec.add %275, %273, %211 : <f32>
                    %277 = ascvec.mul %276, %233, %211 : <f32>
                    %278 = ascvec.cast %277, %211 : <f32> -> <bf16>
                    %279 = arith.addi %243, %c192_i32 overflow<nsw> : i32
                    %280 = arith.index_cast %279 : i32 to index
                    %281 = arith.addi %280, %c16384 : index
                    ascvec.storealign_pack %arg110[%281], %278, %211 {mode = "b32_to_b16"} : memref<?xbf16, 1>, index, <bf16>
                  } {cannir.loop_unroll = 1 : i64}
                } {cannir.vf_mode = "simd"}
                ascvec.release_buf v, %c6_i64, 0 {cannir.buffer.generated_sync}
                %210 = arith.cmpi sle, %59, %c0_i64 : i64
                scf.if %210 {
                  ascvec.sync_intra_arrive v, %c2_i64 {cannir.sync.banks = 3 : i32}
                }
              }
            } else {
              scf.if %205 {
                ascvec.sync_intra_wait v, %c2_i64 {cannir.sync.banks = 3 : i32}
                ascvec.get_buf v, %c6_i64, 0 {cannir.buffer.generated_sync}
                ascvec.vf(%152, %154, %alloc_21, %alloc_25, %alloc_25 : i64, i64, memref<256xf32, 1>, memref<16384xf32, 1>, memref<16384xf32, 1>) {
                ^bb0(%arg104: i64, %arg105: i64, %arg106: memref<256xf32, 1>, %arg107: memref<16384xf32, 1>, %arg108: memref<16384xf32, 1>):
                  %c1_i32_48 = arith.constant 1 : i32
                  %c32_i32_49 = arith.constant 32 : i32
                  %c0_i32_50 = arith.constant 0 : i32
                  %c192_i32 = arith.constant 192 : i32
                  %c128_i32 = arith.constant 128 : i32
                  %c64_i32 = arith.constant 64 : i32
                  %c8192 = arith.constant 8192 : index
                  %c256_i32 = arith.constant 256 : i32
                  %c2_i32_51 = arith.constant 2 : i32
                  %c64_i64_52 = arith.constant 64 : i64
                  %c128_i64_53 = arith.constant 128 : i64
                  %c0_i64_54 = arith.constant 0 : i64
                  %c2_i64_55 = arith.constant 2 : i64
                  %209 = ascvec.create_mask : !ascvec.mask
                  scf.for %arg109 = %c0_i32_50 to %c32_i32_49 step %c1_i32_48  : i32 {
                    %210 = arith.remsi %arg104, %c2_i64_55 : i64
                    %211 = arith.cmpi ne, %210, %c0_i64_54 : i64
                    %212 = arith.cmpi slt, %210, %c0_i64_54 : i64
                    %213 = arith.andi %211, %212 : i1
                    %214 = arith.select %213, %c2_i64_55, %c0_i64_54 : i64
                    %215 = arith.addi %210, %214 overflow<nsw> : i64
                    %216 = arith.muli %215, %c128_i64_53 overflow<nsw> : i64
                    %217 = arith.remsi %arg105, %c2_i64_55 : i64
                    %218 = arith.muli %217, %c64_i64_52 overflow<nsw> : i64
                    %219 = arith.addi %216, %218 overflow<nsw> : i64
                    %220 = arith.muli %arg109, %c2_i32_51 overflow<nsw> : i32
                    %221 = arith.extsi %220 : i32 to i64
                    %222 = arith.addi %219, %221 overflow<nsw> : i64
                    %223 = arith.index_cast %222 : i64 to index
                    %224 = ascvec.loadalign_brc %arg106[%223] : memref<256xf32, 1>, index -> <f32>
                    %225 = arith.muli %arg109, %c256_i32 overflow<nsw> : i32
                    %226 = arith.index_cast %225 : i32 to index
                    %227 = arith.addi %226, %c8192 : index
                    %228 = ascvec.loadalign %arg107[%227] : memref<16384xf32, 1>, index -> <f32>
                    %229 = ascvec.loadalign %arg108[%226] : memref<16384xf32, 1>, index -> <f32>
                    %230 = ascvec.mul %229, %224, %209 : <f32>
                    %231 = ascvec.add %230, %228, %209 : <f32>
                    ascvec.storealign %arg108[%226], %231, %209 : memref<16384xf32, 1>, index, <f32>
                    %232 = arith.addi %225, %c64_i32 overflow<nsw> : i32
                    %233 = arith.index_cast %232 : i32 to index
                    %234 = arith.addi %233, %c8192 : index
                    %235 = ascvec.loadalign %arg107[%234] : memref<16384xf32, 1>, index -> <f32>
                    %236 = ascvec.loadalign %arg108[%233] : memref<16384xf32, 1>, index -> <f32>
                    %237 = ascvec.mul %236, %224, %209 : <f32>
                    %238 = ascvec.add %237, %235, %209 : <f32>
                    ascvec.storealign %arg108[%233], %238, %209 : memref<16384xf32, 1>, index, <f32>
                    %239 = arith.addi %225, %c128_i32 overflow<nsw> : i32
                    %240 = arith.index_cast %239 : i32 to index
                    %241 = arith.addi %240, %c8192 : index
                    %242 = ascvec.loadalign %arg107[%241] : memref<16384xf32, 1>, index -> <f32>
                    %243 = ascvec.loadalign %arg108[%240] : memref<16384xf32, 1>, index -> <f32>
                    %244 = ascvec.mul %243, %224, %209 : <f32>
                    %245 = ascvec.add %244, %242, %209 : <f32>
                    ascvec.storealign %arg108[%240], %245, %209 : memref<16384xf32, 1>, index, <f32>
                    %246 = arith.addi %225, %c192_i32 overflow<nsw> : i32
                    %247 = arith.index_cast %246 : i32 to index
                    %248 = arith.addi %247, %c8192 : index
                    %249 = ascvec.loadalign %arg107[%248] : memref<16384xf32, 1>, index -> <f32>
                    %250 = ascvec.loadalign %arg108[%247] : memref<16384xf32, 1>, index -> <f32>
                    %251 = ascvec.mul %250, %224, %209 : <f32>
                    %252 = ascvec.add %251, %249, %209 : <f32>
                    ascvec.storealign %arg108[%247], %252, %209 : memref<16384xf32, 1>, index, <f32>
                  } {cannir.loop_unroll = 1 : i64}
                } {cannir.vf_mode = "simd"}
                ascvec.release_buf v, %c6_i64, 0 {cannir.buffer.generated_sync}
                ascvec.sync_intra_arrive v, %c2_i64 {cannir.sync.banks = 3 : i32}
              } else {
                ascvec.sync_intra_wait v, %c2_i64 {cannir.sync.banks = 3 : i32}
                ascvec.get_buf v, %c6_i64, 0 {cannir.buffer.generated_sync}
                ascvec.vf(%152, %154, %alloc_21, %alloc_25, %alloc_25 : i64, i64, memref<256xf32, 1>, memref<16384xf32, 1>, memref<16384xf32, 1>) {
                ^bb0(%arg104: i64, %arg105: i64, %arg106: memref<256xf32, 1>, %arg107: memref<16384xf32, 1>, %arg108: memref<16384xf32, 1>):
                  %c1_i32_48 = arith.constant 1 : i32
                  %c32_i32_49 = arith.constant 32 : i32
                  %c0_i32_50 = arith.constant 0 : i32
                  %c192_i32 = arith.constant 192 : i32
                  %c128_i32 = arith.constant 128 : i32
                  %c64_i32 = arith.constant 64 : i32
                  %c8192 = arith.constant 8192 : index
                  %c256_i32 = arith.constant 256 : i32
                  %c2_i32_51 = arith.constant 2 : i32
                  %c64_i64_52 = arith.constant 64 : i64
                  %c128_i64_53 = arith.constant 128 : i64
                  %c0_i64_54 = arith.constant 0 : i64
                  %c2_i64_55 = arith.constant 2 : i64
                  %209 = ascvec.create_mask : !ascvec.mask
                  scf.for %arg109 = %c0_i32_50 to %c32_i32_49 step %c1_i32_48  : i32 {
                    %210 = arith.remsi %arg104, %c2_i64_55 : i64
                    %211 = arith.cmpi ne, %210, %c0_i64_54 : i64
                    %212 = arith.cmpi slt, %210, %c0_i64_54 : i64
                    %213 = arith.andi %211, %212 : i1
                    %214 = arith.select %213, %c2_i64_55, %c0_i64_54 : i64
                    %215 = arith.addi %210, %214 overflow<nsw> : i64
                    %216 = arith.muli %215, %c128_i64_53 overflow<nsw> : i64
                    %217 = arith.remsi %arg105, %c2_i64_55 : i64
                    %218 = arith.muli %217, %c64_i64_52 overflow<nsw> : i64
                    %219 = arith.addi %216, %218 overflow<nsw> : i64
                    %220 = arith.muli %arg109, %c2_i32_51 overflow<nsw> : i32
                    %221 = arith.extsi %220 : i32 to i64
                    %222 = arith.addi %219, %221 overflow<nsw> : i64
                    %223 = arith.index_cast %222 : i64 to index
                    %224 = ascvec.loadalign_brc %arg106[%223] : memref<256xf32, 1>, index -> <f32>
                    %225 = arith.muli %arg109, %c256_i32 overflow<nsw> : i32
                    %226 = arith.index_cast %225 : i32 to index
                    %227 = arith.addi %226, %c8192 : index
                    %228 = ascvec.loadalign %arg107[%227] : memref<16384xf32, 1>, index -> <f32>
                    %229 = ascvec.loadalign %arg108[%226] : memref<16384xf32, 1>, index -> <f32>
                    %230 = ascvec.mul %229, %224, %209 : <f32>
                    %231 = ascvec.add %230, %228, %209 : <f32>
                    ascvec.storealign %arg108[%226], %231, %209 : memref<16384xf32, 1>, index, <f32>
                    %232 = arith.addi %225, %c64_i32 overflow<nsw> : i32
                    %233 = arith.index_cast %232 : i32 to index
                    %234 = arith.addi %233, %c8192 : index
                    %235 = ascvec.loadalign %arg107[%234] : memref<16384xf32, 1>, index -> <f32>
                    %236 = ascvec.loadalign %arg108[%233] : memref<16384xf32, 1>, index -> <f32>
                    %237 = ascvec.mul %236, %224, %209 : <f32>
                    %238 = ascvec.add %237, %235, %209 : <f32>
                    ascvec.storealign %arg108[%233], %238, %209 : memref<16384xf32, 1>, index, <f32>
                    %239 = arith.addi %225, %c128_i32 overflow<nsw> : i32
                    %240 = arith.index_cast %239 : i32 to index
                    %241 = arith.addi %240, %c8192 : index
                    %242 = ascvec.loadalign %arg107[%241] : memref<16384xf32, 1>, index -> <f32>
                    %243 = ascvec.loadalign %arg108[%240] : memref<16384xf32, 1>, index -> <f32>
                    %244 = ascvec.mul %243, %224, %209 : <f32>
                    %245 = ascvec.add %244, %242, %209 : <f32>
                    ascvec.storealign %arg108[%240], %245, %209 : memref<16384xf32, 1>, index, <f32>
                    %246 = arith.addi %225, %c192_i32 overflow<nsw> : i32
                    %247 = arith.index_cast %246 : i32 to index
                    %248 = arith.addi %247, %c8192 : index
                    %249 = ascvec.loadalign %arg107[%248] : memref<16384xf32, 1>, index -> <f32>
                    %250 = ascvec.loadalign %arg108[%247] : memref<16384xf32, 1>, index -> <f32>
                    %251 = ascvec.mul %250, %224, %209 : <f32>
                    %252 = ascvec.add %251, %249, %209 : <f32>
                    ascvec.storealign %arg108[%247], %252, %209 : memref<16384xf32, 1>, index, <f32>
                  } {cannir.loop_unroll = 1 : i64}
                } {cannir.vf_mode = "simd"}
                ascvec.release_buf v, %c6_i64, 0 {cannir.buffer.generated_sync}
                ascvec.sync_intra_arrive v, %c2_i64 {cannir.sync.banks = 3 : i32}
              }
            }
          }
          scf.if %206 {
            %209 = cannir.datastruct_get %arg22["num_heads_q"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
            %210 = arith.cmpi sgt, %59, %c0_i64 : i64
            scf.if %210 {
              %211 = arith.extsi %209 : i32 to i64
              %212 = arith.muli %201, %211 : i64
              %213 = arith.addi %212, %52 : i64
              %214 = arith.muli %213, %c16384_i64 : i64
              %215 = arith.addi %214, %c8192_i64 : i64
              %216 = ascscalar.memref_address %alloc_15 : memref<1xf32>
              %217 = arith.divui %215, %c8_i64 : i64
              %218 = arith.addi %216, %217 : i64
              %219 = ascscalar.bind_gm_ptr %218 : memref<1xf32>
              %220 = ascscalar.bitcast_memref %alloc_25 byte_offset = %c0_i64 : memref<16384xf32, 1> -> memref<?xf32, 1>
              %221 = arith.cmpi ugt, %59, %c0_i64 : i64
              scf.if %221 {
                ascvec.get_buf mte3, %c6_i64, 0 {cannir.buffer.generated_sync}
                %222 = arith.trunci %59 : i64 to i32
                ascvec.copy_ub2gm %219[%c0], %220[%c0], %222, %c1024_i32, %c2048_i64, %c1024_i32 l2_cache_ctl %c0_i32 : memref<1xf32>, memref<?xf32, 1>
                ascvec.release_buf mte3, %c6_i64, 0 {cannir.buffer.generated_sync}
              }
              ascvec.get_buf mte3, %c6_i64, 0 {cannir.buffer.handoff_dependency}
              ascvec.release_buf mte3, %c6_i64, 0 {cannir.buffer.handoff_dependency}
              ascvec.sync_intra_arrive mte3, %c2_i64 {cannir.sync.banks = 3 : i32}
            }
          }
          scf.if %208 {
            %209 = arith.cmpi sgt, %59, %c0_i64 : i64
            scf.if %209 {
              %210 = cannir.datastruct_get %arg22["num_heads_q"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
              %211 = arith.extsi %210 : i32 to i64
              %212 = arith.muli %153, %211 : i64
              %213 = arith.muli %212, %c8192_i64 : i64
              %214 = arith.muli %51, %c16384_i64 : i64
              %215 = arith.addi %214, %c256_i64 : i64
              %216 = arith.muli %215, %c16_i64 : i64
              %217 = arith.addi %213, %216 : i64
              %218 = ascscalar.memref_address %alloc_13 : memref<1xbf16>
              %219 = arith.divui %217, %c8_i64 : i64
              %220 = arith.addi %218, %219 : i64
              %221 = ascscalar.bind_gm_ptr %220 : memref<1xbf16>
              %222 = ascscalar.memref_address %alloc_25 : memref<16384xf32, 1>
              %223 = arith.addi %222, %c32768_i64 : i64
              %224 = ascscalar.bind_local_ptr %223 : memref<1xbf16, 1>
              %225 = arith.cmpi ugt, %59, %c0_i64 : i64
              scf.if %225 {
                ascvec.get_buf mte3, %c6_i64, 0 {cannir.buffer.generated_sync}
                %226 = arith.trunci %59 : i64 to i32
                ascvec.copy_ub2gm %221[%c0], %224[%c0], %226, %c512_i32, %c1024_i64, %c1024_i32 l2_cache_ctl %c4_i32 : memref<1xbf16>, memref<1xbf16, 1>
                ascvec.release_buf mte3, %c6_i64, 0 {cannir.buffer.generated_sync}
              }
              ascvec.get_buf mte3, %c6_i64, 0 {cannir.buffer.handoff_dependency}
              ascvec.release_buf mte3, %c6_i64, 0 {cannir.buffer.handoff_dependency}
              ascvec.sync_intra_arrive mte3, %c2_i64 {cannir.sync.banks = 3 : i32}
            }
          }
        }
        %167 = arith.select %160, %c0_i64, %arg68 : i64
        %168 = arith.select %160, %c32768_i64, %arg69 : i64
        %169 = arith.select %160, %c1_i64, %arg70 : i64
        %170 = arith.select %160, %c32768_i64, %arg71 : i64
        %171 = arith.select %160, %c1_i64, %arg72 : i64
        %172 = arith.select %160, %c0_i64, %arg74 : i64
        %173 = arith.select %160, %c0_i64, %arg76 : i64
        %174 = arith.select %160, %c0_i64, %arg78 : i64
        %175 = arith.select %160, %c32768_i64, %arg79 : i64
        %176 = arith.select %160, %c1_i64, %arg80 : i64
        %177 = arith.select %160, %c32768_i64, %arg81 : i64
        %178 = arith.select %160, %c1_i64, %arg82 : i64
        %179 = arith.select %160, %c0_i64, %arg84 : i64
        %180 = arith.select %160, %c0_i64, %arg86 : i64
        %181 = arith.select %160, %c0_i64, %arg88 : i64
        %182 = arith.select %160, %c64_i64, %arg89 : i64
        %183 = arith.select %160, %c1_i64, %arg90 : i64
        %184 = arith.select %160, %c64_i64, %arg91 : i64
        %185 = arith.select %160, %c1_i64, %arg92 : i64
        %186 = arith.select %160, %c0_i64, %arg94 : i64
        %187 = arith.select %160, %c0_i64, %arg96 : i64
        %188:14 = scf.if %160 -> (i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) {
          %192 = arith.cmpi eq, %arg100, %c1_i64 : i64
          %193 = ascscalar.memref_address %alloc_46 : memref<32768xbf16, 2>
          %194 = ascscalar.memref_address %alloc_45 : memref<32768xbf16, 2>
          %195 = arith.select %192, %193, %194 : i64
          %196 = arith.cmpi eq, %arg100, %c0_i64 : i64
          %197 = arith.select %196, %90, %195 : i64
          %198 = arith.select %192, %c18_i64, %c19_i64 : i64
          %199 = arith.select %196, %c17_i64, %198 : i64
          %200 = arith.addi %arg100, %c1_i64 : i64
          %201 = arith.remui %200, %c3_i64 : i64
          %202 = ascscalar.memref_address %alloc_43 : memref<32768xbf16, 2>
          %203 = ascscalar.memref_address %alloc_42 : memref<32768xbf16, 2>
          %204 = arith.select %192, %202, %203 : i64
          %205 = arith.select %196, %91, %204 : i64
          %206 = arith.select %192, %c22_i64, %c23_i64 : i64
          %207 = arith.select %196, %c21_i64, %206 : i64
          %208 = arith.cmpi eq, %arg99, %c0_i64 : i64
          %209 = ascscalar.memref_address %alloc_31 : memref<64xi32, 1>
          %210 = arith.select %208, %92, %209 : i64
          %211 = arith.select %208, %c12_i64, %c13_i64 : i64
          %212 = arith.addi %arg99, %c1_i64 : i64
          %213 = arith.remui %212, %c2_i64 : i64
          scf.yield %197, %197, %197, %205, %205, %205, %210, %210, %210, %213, %201, %199, %207, %211 : i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64
        } else {
          scf.yield %arg67, %arg73, %arg75, %arg77, %arg83, %arg85, %arg87, %arg93, %arg95, %arg99, %arg100, %arg101, %arg102, %arg103 : i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64
        }
        %189 = cannir.array_load %36[%c0] : <1xi64>
        %190 = arith.addi %189, %c1_i64 : i64
        cannir.array_store %190, %36[%c0] : <1xi64>
        %191 = arith.addi %arg97, %c1_i64 : i64
        scf.yield %188#0, %167, %168, %169, %170, %171, %188#1, %172, %188#2, %173, %188#3, %174, %175, %176, %177, %178, %188#4, %179, %188#5, %180, %188#6, %181, %182, %183, %184, %185, %188#7, %186, %188#8, %187, %191, %126, %188#9, %188#10, %188#11, %188#12, %188#13 : i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64
      } {cannir.view_payload}
      %101 = arith.addi %arg57, %99#10 : i64
      %102 = arith.cmpi eq, %99#11, %c0_i64 : i64
      %103 = arith.extui %102 : i1 to i64
      %104 = arith.addi %arg58, %103 : i64
      %105 = arith.select %96, %arg57, %101 : i64
      %106 = arith.select %96, %arg58, %104 : i64
      %107 = arith.addi %arg59, %c1_i64 : i64
      scf.yield %99#0, %100#0, %100#1, %100#2, %100#3, %100#4, %100#5, %100#6, %100#7, %100#8, %100#9, %100#10, %100#11, %100#12, %100#13, %100#14, %100#15, %100#16, %100#17, %100#18, %100#19, %100#20, %100#21, %100#22, %100#23, %100#24, %100#25, %100#26, %100#27, %100#28, %100#29, %100#30, %105, %106, %107, %100#31, %100#32, %100#33, %100#34, %100#35, %100#36 : i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64
    } {cannir.view_payload}
    %94 = memref.load %alloc_8[%c900] : memref<1xi32>
    %95 = arith.cmpi ne, %94, %c0_i32 : i32
    scf.if %95 {
      ascvec.sync_all
      ascvec.sync_block_arrive mte3, %c0_i64
      ascvec.sync_block_arrive mte3, %c1_i64 {mode = 0 : i32}
      ascvec.sync_block_wait s, %c1_i64 {mode = 0 : i32}
      ascvec.sync_block_wait s, %c2_i64
      %96 = arith.muli %0, %c2_i64 : i64
      %97 = arith.addi %96, %51 : i64
      %98 = arith.muli %97, %c8_i64 : i64
      %99 = arith.addi %98, %c324_i64 : i64
      %100 = arith.index_cast %99 : i64 to index
      %101 = memref.load %alloc_8[%100] : memref<1xi32>
      %102 = arith.cmpi ne, %101, %c0_i32 : i32
      scf.if %102 {
        %103 = arith.addi %98, %c325_i64 : i64
        %104 = arith.index_cast %103 : i64 to index
        %105 = memref.load %alloc_8[%104] : memref<1xi32>
        %106 = arith.extsi %105 : i32 to i64
        %107 = cannir.datastruct_get %arg22["query_rows_per_batch"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
        %108 = arith.extsi %107 : i32 to i64
        %109 = arith.muli %106, %108 : i64
        %110 = arith.addi %98, %c326_i64 : i64
        %111 = arith.index_cast %110 : i64 to index
        %112 = memref.load %alloc_8[%111] : memref<1xi32>
        %113 = arith.extsi %112 : i32 to i64
        %114 = arith.addi %109, %113 : i64
        %115 = cannir.datastruct_get %arg22["batch_mode"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
        %116 = arith.cmpi eq, %115, %c2_i32 : i32
        %117 = scf.if %116 -> (i64) {
          %143 = arith.index_cast %105 : i32 to index
          %144 = memref.load %alloc_10[%143] : memref<1xi32>
          %145 = arith.extsi %144 : i32 to i64
          %146 = memref.load %alloc_8[%111] : memref<1xi32>
          %147 = arith.extsi %146 : i32 to i64
          %148 = arith.addi %145, %147 : i64
          scf.yield %148 : i64
        } else {
          scf.yield %114 : i64
        }
        %118 = arith.addi %98, %c327_i64 : i64
        %119 = arith.index_cast %118 : i64 to index
        %120 = memref.load %alloc_8[%119] : memref<1xi32>
        %121 = arith.extsi %120 : i32 to i64
        %122 = arith.addi %98, %c328_i64 : i64
        %123 = arith.index_cast %122 : i64 to index
        %124 = memref.load %alloc_8[%123] : memref<1xi32>
        %125 = arith.extsi %124 : i32 to i64
        %126 = arith.addi %98, %c329_i64 : i64
        %127 = arith.index_cast %126 : i64 to index
        %128 = memref.load %alloc_8[%127] : memref<1xi32>
        %129 = arith.extsi %128 : i32 to i64
        %130 = arith.addi %98, %c330_i64 : i64
        %131 = arith.index_cast %130 : i64 to index
        %132 = memref.load %alloc_8[%131] : memref<1xi32>
        %133 = arith.extsi %132 : i32 to i64
        %134 = cannir.datastruct_get %arg22["cube_core_count"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
        %135 = arith.muli %134, %c2_i32 : i32
        %136 = cannir.datastruct_get %arg22["num_heads_q"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
        %137 = arith.muli %135, %136 : i32
        %138 = arith.muli %137, %c512_i32 : i32
        %139 = arith.muli %121, %c128_i64 : i64
        %140 = arith.extsi %138 : i32 to i64
        %141 = arith.addi %140, %139 : i64
        %142 = scf.for %arg24 = %c0_i64 to %133 step %c16_i64 iter_args(%arg25 = %c0_i64) -> (i64)  : i64 {
          %143 = arith.subi %133, %arg24 : i64
          %144 = arith.minsi %143, %c16_i64 : i64
          %145 = arith.addi %129, %arg24 : i64
          %146 = ascscalar.bitcast_memref %alloc_32 byte_offset = %c0_i64 : memref<9216xf32, 1> -> memref<?xf32, 1>
          %147 = arith.muli %141, %c4_i64 : i64
          %148 = arith.muli %145, %c4_i64 : i64
          %149 = arith.addi %147, %148 : i64
          %150 = arith.muli %144, %c4_i64 : i64
          %151 = arith.cmpi eq, %150, %c0_i64 : i64
          %152 = arith.select %151, %c0_i64, %125 : i64
          %153 = arith.select %151, %c4_i64, %150 : i64
          ascvec.get_buf mte2, %c0_i64, 0 {cannir.buffer.generated_sync}
          "ascvec.copy_gm2ub_plan"(%146, %alloc_15, %c0_i32, %c0_i32, %c0_i64, %149, %152, %153, %c512_i64, %c512_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 0 : i64, ub_capacity = 36864 : i64}> : (memref<?xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
          %154 = arith.addi %145, %c64_i64 : i64
          %155 = arith.muli %154, %c4_i64 : i64
          %156 = arith.addi %147, %155 : i64
          "ascvec.copy_gm2ub_plan"(%146, %alloc_15, %c0_i32, %c0_i32, %c256_i64, %156, %152, %153, %c512_i64, %c512_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 0 : i64, ub_capacity = 36864 : i64}> : (memref<?xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
          ascvec.release_buf mte2, %c0_i64, 0 {cannir.buffer.generated_sync}
          %157 = arith.trunci %144 : i64 to i32
          ascvec.vf(%157, %alloc_37, %alloc_38 : i32, memref<64xf32, 1>, memref<64xf32, 1>) {
          ^bb0(%arg26: i32, %arg27: memref<64xf32, 1>, %arg28: memref<64xf32, 1>):
            %cst = arith.constant 0.000000e+00 : f32
            %c0_48 = arith.constant 0 : index
            %cst_49 = arith.constant -3.40282347E+38 : f32
            %c0_50 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            scf.for %arg29 = %c0_50 to %c1 step %c1 {
              %mask, %remaining_out = ascvec.update_mask %arg26 : !ascvec.mask
              %201 = ascvec.duplicate_scalar_nomask %cst_49 : f32 -> <f32>
              ascvec.storealign %arg27[%c0_48], %201, %mask : memref<64xf32, 1>, index, <f32>
              %202 = ascvec.duplicate_scalar_nomask %cst : f32 -> <f32>
              ascvec.storealign %arg28[%c0_48], %202, %mask : memref<64xf32, 1>, index, <f32>
            }
          } {cannir.vf_mode = "simd"}
          scf.for %arg26 = %c0_i64 to %125 step %c1_i64  : i64 {
            %201 = arith.muli %arg26, %c128_i64 : i64
            %202 = arith.index_cast %201 : i64 to index
            %203 = arith.muli %arg26, %c64_i64 : i64
            %204 = arith.index_cast %203 : i64 to index
            ascvec.get_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
            ascvec.vf(%157, %alloc_32, %202, %alloc_37, %alloc_39, %204 : i32, memref<9216xf32, 1>, index, memref<64xf32, 1>, memref<4608xf32, 1>, index) {
            ^bb0(%arg27: i32, %arg28: memref<9216xf32, 1>, %arg29: index, %arg30: memref<64xf32, 1>, %arg31: memref<4608xf32, 1>, %arg32: index):
              %c0_48 = arith.constant 0 : index
              %c0_49 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              scf.for %arg33 = %c0_49 to %c1 step %c1 {
                %mask, %remaining_out = ascvec.update_mask %arg27 : !ascvec.mask
                %205 = ascvec.loadalign %arg28[%arg29] : memref<9216xf32, 1>, index -> <f32>
                %206 = ascvec.loadalign %arg30[%c0_48] : memref<64xf32, 1>, index -> <f32>
                %207 = ascvec.max %206, %205, %mask : <f32>
                ascvec.storealign %arg30[%c0_48], %207, %mask : memref<64xf32, 1>, index, <f32>
                ascvec.storealign %arg31[%arg32], %205, %mask : memref<4608xf32, 1>, index, <f32>
              }
            } {cannir.vf_mode = "simd"}
            ascvec.release_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
          }
          scf.for %arg26 = %c0_i64 to %125 step %c1_i64  : i64 {
            %201 = arith.muli %arg26, %c64_i64 : i64
            %202 = arith.index_cast %201 : i64 to index
            %203 = arith.muli %arg26, %c128_i64 : i64
            %204 = arith.addi %203, %c64_i64 : i64
            %205 = arith.index_cast %204 : i64 to index
            ascvec.get_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
            ascvec.vf(%157, %alloc_39, %202, %alloc_37, %alloc_32, %205, %alloc_38 : i32, memref<4608xf32, 1>, index, memref<64xf32, 1>, memref<9216xf32, 1>, index, memref<64xf32, 1>) {
            ^bb0(%arg27: i32, %arg28: memref<4608xf32, 1>, %arg29: index, %arg30: memref<64xf32, 1>, %arg31: memref<9216xf32, 1>, %arg32: index, %arg33: memref<64xf32, 1>):
              %cst = arith.constant 0.000000e+00 : f32
              %cst_48 = arith.constant -3.40282347E+38 : f32
              %c0_49 = arith.constant 0 : index
              %c0_50 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              scf.for %arg34 = %c0_50 to %c1 step %c1 {
                %mask, %remaining_out = ascvec.update_mask %arg27 : !ascvec.mask
                %206 = ascvec.loadalign %arg28[%arg29] : memref<4608xf32, 1>, index -> <f32>
                %207 = ascvec.cmp_eq_scalar %206, %cst_48, %mask : <f32>, f32 -> !ascvec.mask
                %208 = ascvec.loadalign %arg30[%c0_49] : memref<64xf32, 1>, index -> <f32>
                %209 = ascvec.exp_sub %206, %208, %mask : <f32>, <f32> -> <f32>
                %210 = ascvec.loadalign %arg31[%arg32] : memref<9216xf32, 1>, index -> <f32>
                %211 = ascvec.mul %209, %210, %mask : <f32>
                %212 = ascvec.duplicate_scalar_nomask %cst : f32 -> <f32>
                %213 = ascvec.select %212, %211, %207 : <f32>
                %214 = ascvec.loadalign %arg33[%c0_49] : memref<64xf32, 1>, index -> <f32>
                %215 = ascvec.add %214, %213, %mask : <f32>
                ascvec.storealign %arg33[%c0_49], %215, %mask : memref<64xf32, 1>, index, <f32>
                ascvec.storealign %arg28[%arg29], %213, %mask : memref<4608xf32, 1>, index, <f32>
              }
            } {cannir.vf_mode = "simd"}
            ascvec.release_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
          }
          %158 = cannir.datastruct_get %arg22["num_heads_q"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
          %159 = arith.extsi %158 : i32 to i64
          %160 = arith.muli %121, %159 : i64
          %161 = arith.addi %160, %145 : i64
          %162 = arith.cmpi eq, %arg25, %c0_i64 : i64
          %163 = ascscalar.memref_address %alloc_33 : memref<8192xf32, 1>
          %164 = ascscalar.memref_address %alloc_34 : memref<8192xf32, 1>
          %165 = arith.select %162, %163, %164 : i64
          %166 = arith.select %162, %c1_i64, %c2_i64 : i64
          %167 = arith.addi %arg25, %c1_i64 overflow<nsw> : i64
          %168 = arith.remui %167, %c2_i64 : i64
          %169 = arith.muli %161, %c2048_i64 : i64
          %170 = arith.muli %144, %c2048_i64 : i64
          %171 = arith.cmpi eq, %170, %c0_i64 : i64
          %172 = arith.cmpi ne, %170, %c0_i64 : i64
          %173 = arith.extui %172 : i1 to i64
          %174 = arith.select %171, %c4_i64, %170 : i64
          %175 = ascscalar.bind_local_ptr %165 : memref<1xf32, 1>
          ascvec.get_buf mte2, %166, 0 {cannir.buffer.generated_sync}
          "ascvec.copy_gm2ub_plan"(%175, %alloc_15, %c0_i32, %c0_i32, %c0_i64, %169, %173, %174, %c0_i64, %174, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = -1 : i64, ub_capacity = 4 : i64}> : (memref<1xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
          ascvec.release_buf mte2, %166, 0 {cannir.buffer.generated_sync}
          %176 = ascscalar.bind_local_ptr %165 : memref<8192xf32, 1>
          ascvec.get_buf v, %166, 0 {cannir.buffer.generated_sync}
          ascvec.vf(%alloc_38, %alloc_39, %176, %alloc_35, %144 : memref<64xf32, 1>, memref<4608xf32, 1>, memref<8192xf32, 1>, memref<8192xf32, 1>, i64) {
          ^bb0(%arg26: memref<64xf32, 1>, %arg27: memref<4608xf32, 1>, %arg28: memref<8192xf32, 1>, %arg29: memref<8192xf32, 1>, %arg30: i64):
            %c1_i64_48 = arith.constant 1 : i64
            %c0_i64_49 = arith.constant 0 : i64
            %c448_i64 = arith.constant 448 : i64
            %c384_i64 = arith.constant 384 : i64
            %c320_i64 = arith.constant 320 : i64
            %c256_i64_50 = arith.constant 256 : i64
            %c192_i64 = arith.constant 192 : i64
            %c128_i64_51 = arith.constant 128 : i64
            %c64_i64_52 = arith.constant 64 : i64
            %c512_i64_53 = arith.constant 512 : i64
            %cst = arith.constant 1.000000e+00 : f32
            %cst_54 = arith.constant 0.000000e+00 : f32
            %201 = ascvec.create_mask : !ascvec.mask
            scf.for %arg31 = %c0_i64_49 to %arg30 step %c1_i64_48  : i64 {
              %202 = arith.index_cast %arg31 : i64 to index
              %203 = ascvec.loadalign_brc %arg26[%202] : memref<64xf32, 1>, index -> <f32>
              %204 = ascvec.cmp_eq_scalar %203, %cst_54, %201 : <f32>, f32 -> !ascvec.mask
              %205 = ascvec.duplicate_scalar_nomask %cst : f32 -> <f32>
              %206 = ascvec.select %205, %203, %204 : <f32>
              %207 = ascvec.loadalign_brc %arg27[%202] : memref<4608xf32, 1>, index -> <f32>
              %208 = ascvec.div %207, %206, %201 : <f32>
              %209 = arith.muli %arg31, %c512_i64_53 : i64
              %210 = arith.index_cast %209 : i64 to index
              %211 = ascvec.loadalign %arg28[%210] : memref<8192xf32, 1>, index -> <f32>
              %212 = ascvec.mul %211, %208, %201 : <f32>
              ascvec.storealign %arg29[%210], %212, %201 : memref<8192xf32, 1>, index, <f32>
              %213 = arith.addi %209, %c64_i64_52 : i64
              %214 = arith.index_cast %213 : i64 to index
              %215 = ascvec.loadalign %arg28[%214] : memref<8192xf32, 1>, index -> <f32>
              %216 = ascvec.mul %215, %208, %201 : <f32>
              ascvec.storealign %arg29[%214], %216, %201 : memref<8192xf32, 1>, index, <f32>
              %217 = arith.addi %209, %c128_i64_51 : i64
              %218 = arith.index_cast %217 : i64 to index
              %219 = ascvec.loadalign %arg28[%218] : memref<8192xf32, 1>, index -> <f32>
              %220 = ascvec.mul %219, %208, %201 : <f32>
              ascvec.storealign %arg29[%218], %220, %201 : memref<8192xf32, 1>, index, <f32>
              %221 = arith.addi %209, %c192_i64 : i64
              %222 = arith.index_cast %221 : i64 to index
              %223 = ascvec.loadalign %arg28[%222] : memref<8192xf32, 1>, index -> <f32>
              %224 = ascvec.mul %223, %208, %201 : <f32>
              ascvec.storealign %arg29[%222], %224, %201 : memref<8192xf32, 1>, index, <f32>
              %225 = arith.addi %209, %c256_i64_50 : i64
              %226 = arith.index_cast %225 : i64 to index
              %227 = ascvec.loadalign %arg28[%226] : memref<8192xf32, 1>, index -> <f32>
              %228 = ascvec.mul %227, %208, %201 : <f32>
              ascvec.storealign %arg29[%226], %228, %201 : memref<8192xf32, 1>, index, <f32>
              %229 = arith.addi %209, %c320_i64 : i64
              %230 = arith.index_cast %229 : i64 to index
              %231 = ascvec.loadalign %arg28[%230] : memref<8192xf32, 1>, index -> <f32>
              %232 = ascvec.mul %231, %208, %201 : <f32>
              ascvec.storealign %arg29[%230], %232, %201 : memref<8192xf32, 1>, index, <f32>
              %233 = arith.addi %209, %c384_i64 : i64
              %234 = arith.index_cast %233 : i64 to index
              %235 = ascvec.loadalign %arg28[%234] : memref<8192xf32, 1>, index -> <f32>
              %236 = ascvec.mul %235, %208, %201 : <f32>
              ascvec.storealign %arg29[%234], %236, %201 : memref<8192xf32, 1>, index, <f32>
              %237 = arith.addi %209, %c448_i64 : i64
              %238 = arith.index_cast %237 : i64 to index
              %239 = ascvec.loadalign %arg28[%238] : memref<8192xf32, 1>, index -> <f32>
              %240 = ascvec.mul %239, %208, %201 : <f32>
              ascvec.storealign %arg29[%238], %240, %201 : memref<8192xf32, 1>, index, <f32>
            }
          } {cannir.vf_mode = "simd"}
          ascvec.release_buf v, %166, 0 {cannir.buffer.generated_sync}
          %177 = scf.for %arg26 = %c1_i64 to %125 step %c1_i64 iter_args(%arg27 = %168) -> (i64)  : i64 {
            %201 = arith.addi %121, %arg26 : i64
            %202 = cannir.datastruct_get %arg22["num_heads_q"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
            %203 = arith.extsi %202 : i32 to i64
            %204 = arith.muli %201, %203 : i64
            %205 = arith.addi %204, %145 : i64
            %206 = arith.cmpi eq, %arg27, %c0_i64 : i64
            %207 = arith.select %206, %163, %164 : i64
            %208 = arith.select %206, %c1_i64, %c2_i64 : i64
            %209 = arith.addi %arg27, %c1_i64 overflow<nsw> : i64
            %210 = arith.remui %209, %c2_i64 : i64
            %211 = arith.muli %205, %c2048_i64 : i64
            %212 = ascscalar.bind_local_ptr %207 : memref<1xf32, 1>
            ascvec.get_buf mte2, %208, 0 {cannir.buffer.generated_sync}
            "ascvec.copy_gm2ub_plan"(%212, %alloc_15, %c0_i32, %c0_i32, %c0_i64, %211, %173, %174, %c0_i64, %174, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = -1 : i64, ub_capacity = 4 : i64}> : (memref<1xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
            ascvec.release_buf mte2, %208, 0 {cannir.buffer.generated_sync}
            %213 = ascscalar.bind_local_ptr %207 : memref<8192xf32, 1>
            ascvec.get_buf v, %208, 0 {cannir.buffer.generated_sync}
            ascvec.vf(%alloc_38, %arg26, %alloc_39, %213, %alloc_35, %144 : memref<64xf32, 1>, i64, memref<4608xf32, 1>, memref<8192xf32, 1>, memref<8192xf32, 1>, i64) {
            ^bb0(%arg28: memref<64xf32, 1>, %arg29: i64, %arg30: memref<4608xf32, 1>, %arg31: memref<8192xf32, 1>, %arg32: memref<8192xf32, 1>, %arg33: i64):
              %c1_i64_48 = arith.constant 1 : i64
              %c0_i64_49 = arith.constant 0 : i64
              %c448_i64 = arith.constant 448 : i64
              %c384_i64 = arith.constant 384 : i64
              %c320_i64 = arith.constant 320 : i64
              %c256_i64_50 = arith.constant 256 : i64
              %c192_i64 = arith.constant 192 : i64
              %c128_i64_51 = arith.constant 128 : i64
              %c512_i64_52 = arith.constant 512 : i64
              %c64_i64_53 = arith.constant 64 : i64
              %cst = arith.constant 1.000000e+00 : f32
              %cst_54 = arith.constant 0.000000e+00 : f32
              %214 = ascvec.create_mask : !ascvec.mask
              scf.for %arg34 = %c0_i64_49 to %arg33 step %c1_i64_48  : i64 {
                %215 = arith.index_cast %arg34 : i64 to index
                %216 = ascvec.loadalign_brc %arg28[%215] : memref<64xf32, 1>, index -> <f32>
                %217 = ascvec.cmp_eq_scalar %216, %cst_54, %214 : <f32>, f32 -> !ascvec.mask
                %218 = ascvec.duplicate_scalar_nomask %cst : f32 -> <f32>
                %219 = ascvec.select %218, %216, %217 : <f32>
                %220 = arith.muli %arg29, %c64_i64_53 : i64
                %221 = arith.addi %220, %arg34 : i64
                %222 = arith.index_cast %221 : i64 to index
                %223 = ascvec.loadalign_brc %arg30[%222] : memref<4608xf32, 1>, index -> <f32>
                %224 = ascvec.div %223, %219, %214 : <f32>
                %225 = arith.muli %arg34, %c512_i64_52 : i64
                %226 = arith.index_cast %225 : i64 to index
                %227 = ascvec.loadalign %arg31[%226] : memref<8192xf32, 1>, index -> <f32>
                %228 = ascvec.mul %227, %224, %214 : <f32>
                %229 = ascvec.loadalign %arg32[%226] : memref<8192xf32, 1>, index -> <f32>
                %230 = ascvec.add %229, %228, %214 : <f32>
                ascvec.storealign %arg32[%226], %230, %214 : memref<8192xf32, 1>, index, <f32>
                %231 = arith.addi %225, %c64_i64_53 : i64
                %232 = arith.index_cast %231 : i64 to index
                %233 = ascvec.loadalign %arg31[%232] : memref<8192xf32, 1>, index -> <f32>
                %234 = ascvec.mul %233, %224, %214 : <f32>
                %235 = ascvec.loadalign %arg32[%232] : memref<8192xf32, 1>, index -> <f32>
                %236 = ascvec.add %235, %234, %214 : <f32>
                ascvec.storealign %arg32[%232], %236, %214 : memref<8192xf32, 1>, index, <f32>
                %237 = arith.addi %225, %c128_i64_51 : i64
                %238 = arith.index_cast %237 : i64 to index
                %239 = ascvec.loadalign %arg31[%238] : memref<8192xf32, 1>, index -> <f32>
                %240 = ascvec.mul %239, %224, %214 : <f32>
                %241 = ascvec.loadalign %arg32[%238] : memref<8192xf32, 1>, index -> <f32>
                %242 = ascvec.add %241, %240, %214 : <f32>
                ascvec.storealign %arg32[%238], %242, %214 : memref<8192xf32, 1>, index, <f32>
                %243 = arith.addi %225, %c192_i64 : i64
                %244 = arith.index_cast %243 : i64 to index
                %245 = ascvec.loadalign %arg31[%244] : memref<8192xf32, 1>, index -> <f32>
                %246 = ascvec.mul %245, %224, %214 : <f32>
                %247 = ascvec.loadalign %arg32[%244] : memref<8192xf32, 1>, index -> <f32>
                %248 = ascvec.add %247, %246, %214 : <f32>
                ascvec.storealign %arg32[%244], %248, %214 : memref<8192xf32, 1>, index, <f32>
                %249 = arith.addi %225, %c256_i64_50 : i64
                %250 = arith.index_cast %249 : i64 to index
                %251 = ascvec.loadalign %arg31[%250] : memref<8192xf32, 1>, index -> <f32>
                %252 = ascvec.mul %251, %224, %214 : <f32>
                %253 = ascvec.loadalign %arg32[%250] : memref<8192xf32, 1>, index -> <f32>
                %254 = ascvec.add %253, %252, %214 : <f32>
                ascvec.storealign %arg32[%250], %254, %214 : memref<8192xf32, 1>, index, <f32>
                %255 = arith.addi %225, %c320_i64 : i64
                %256 = arith.index_cast %255 : i64 to index
                %257 = ascvec.loadalign %arg31[%256] : memref<8192xf32, 1>, index -> <f32>
                %258 = ascvec.mul %257, %224, %214 : <f32>
                %259 = ascvec.loadalign %arg32[%256] : memref<8192xf32, 1>, index -> <f32>
                %260 = ascvec.add %259, %258, %214 : <f32>
                ascvec.storealign %arg32[%256], %260, %214 : memref<8192xf32, 1>, index, <f32>
                %261 = arith.addi %225, %c384_i64 : i64
                %262 = arith.index_cast %261 : i64 to index
                %263 = ascvec.loadalign %arg31[%262] : memref<8192xf32, 1>, index -> <f32>
                %264 = ascvec.mul %263, %224, %214 : <f32>
                %265 = ascvec.loadalign %arg32[%262] : memref<8192xf32, 1>, index -> <f32>
                %266 = ascvec.add %265, %264, %214 : <f32>
                ascvec.storealign %arg32[%262], %266, %214 : memref<8192xf32, 1>, index, <f32>
                %267 = arith.addi %225, %c448_i64 : i64
                %268 = arith.index_cast %267 : i64 to index
                %269 = ascvec.loadalign %arg31[%268] : memref<8192xf32, 1>, index -> <f32>
                %270 = ascvec.mul %269, %224, %214 : <f32>
                %271 = ascvec.loadalign %arg32[%268] : memref<8192xf32, 1>, index -> <f32>
                %272 = ascvec.add %271, %270, %214 : <f32>
                ascvec.storealign %arg32[%268], %272, %214 : memref<8192xf32, 1>, index, <f32>
              }
            } {cannir.vf_mode = "simd"}
            ascvec.release_buf v, %208, 0 {cannir.buffer.generated_sync}
            scf.yield %210 : i64
          }
          ascvec.get_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
          ascvec.get_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
          ascvec.vf(%alloc_35, %alloc_36, %144, %157, %alloc_38, %alloc_37, %alloc_40 : memref<8192xf32, 1>, memref<8192xbf16, 1>, i64, i32, memref<64xf32, 1>, memref<64xf32, 1>, memref<64xf32, 1>) {
          ^bb0(%arg26: memref<8192xf32, 1>, %arg27: memref<8192xbf16, 1>, %arg28: i64, %arg29: i32, %arg30: memref<64xf32, 1>, %arg31: memref<64xf32, 1>, %arg32: memref<64xf32, 1>):
            %c2139095040_i32 = arith.constant 2139095040 : i32
            %cst = arith.constant 0.000000e+00 : f32
            %c1_i64_48 = arith.constant 1 : i64
            %c0_i64_49 = arith.constant 0 : i64
            %c448_i64 = arith.constant 448 : i64
            %c384_i64 = arith.constant 384 : i64
            %c320_i64 = arith.constant 320 : i64
            %c256_i64_50 = arith.constant 256 : i64
            %c192_i64 = arith.constant 192 : i64
            %c128_i64_51 = arith.constant 128 : i64
            %c64_i64_52 = arith.constant 64 : i64
            %c0_53 = arith.constant 0 : index
            %c512_i64_54 = arith.constant 512 : i64
            %201 = ascvec.create_mask : !ascvec.mask
            scf.for %arg33 = %c0_i64_49 to %arg28 step %c1_i64_48  : i64 {
              %202 = arith.muli %arg33, %c512_i64_54 : i64
              %203 = arith.index_cast %202 : i64 to index
              %204 = ascvec.loadalign %arg26[%203] : memref<8192xf32, 1>, index -> <f32>
              %205 = ascvec.cast %204, %201 : <f32> -> <bf16>
              ascvec.storealign_pack %arg27[%203], %205, %201 {mode = "b32_to_b16"} : memref<8192xbf16, 1>, index, <bf16>
              %206 = arith.addi %202, %c64_i64_52 : i64
              %207 = arith.index_cast %206 : i64 to index
              %208 = ascvec.loadalign %arg26[%207] : memref<8192xf32, 1>, index -> <f32>
              %209 = ascvec.cast %208, %201 : <f32> -> <bf16>
              ascvec.storealign_pack %arg27[%207], %209, %201 {mode = "b32_to_b16"} : memref<8192xbf16, 1>, index, <bf16>
              %210 = arith.addi %202, %c128_i64_51 : i64
              %211 = arith.index_cast %210 : i64 to index
              %212 = ascvec.loadalign %arg26[%211] : memref<8192xf32, 1>, index -> <f32>
              %213 = ascvec.cast %212, %201 : <f32> -> <bf16>
              ascvec.storealign_pack %arg27[%211], %213, %201 {mode = "b32_to_b16"} : memref<8192xbf16, 1>, index, <bf16>
              %214 = arith.addi %202, %c192_i64 : i64
              %215 = arith.index_cast %214 : i64 to index
              %216 = ascvec.loadalign %arg26[%215] : memref<8192xf32, 1>, index -> <f32>
              %217 = ascvec.cast %216, %201 : <f32> -> <bf16>
              ascvec.storealign_pack %arg27[%215], %217, %201 {mode = "b32_to_b16"} : memref<8192xbf16, 1>, index, <bf16>
              %218 = arith.addi %202, %c256_i64_50 : i64
              %219 = arith.index_cast %218 : i64 to index
              %220 = ascvec.loadalign %arg26[%219] : memref<8192xf32, 1>, index -> <f32>
              %221 = ascvec.cast %220, %201 : <f32> -> <bf16>
              ascvec.storealign_pack %arg27[%219], %221, %201 {mode = "b32_to_b16"} : memref<8192xbf16, 1>, index, <bf16>
              %222 = arith.addi %202, %c320_i64 : i64
              %223 = arith.index_cast %222 : i64 to index
              %224 = ascvec.loadalign %arg26[%223] : memref<8192xf32, 1>, index -> <f32>
              %225 = ascvec.cast %224, %201 : <f32> -> <bf16>
              ascvec.storealign_pack %arg27[%223], %225, %201 {mode = "b32_to_b16"} : memref<8192xbf16, 1>, index, <bf16>
              %226 = arith.addi %202, %c384_i64 : i64
              %227 = arith.index_cast %226 : i64 to index
              %228 = ascvec.loadalign %arg26[%227] : memref<8192xf32, 1>, index -> <f32>
              %229 = ascvec.cast %228, %201 : <f32> -> <bf16>
              ascvec.storealign_pack %arg27[%227], %229, %201 {mode = "b32_to_b16"} : memref<8192xbf16, 1>, index, <bf16>
              %230 = arith.addi %202, %c448_i64 : i64
              %231 = arith.index_cast %230 : i64 to index
              %232 = ascvec.loadalign %arg26[%231] : memref<8192xf32, 1>, index -> <f32>
              %233 = ascvec.cast %232, %201 : <f32> -> <bf16>
              ascvec.storealign_pack %arg27[%231], %233, %201 {mode = "b32_to_b16"} : memref<8192xbf16, 1>, index, <bf16>
            }
            %c0_55 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            scf.for %arg33 = %c0_55 to %c1 step %c1 {
              %mask, %remaining_out = ascvec.update_mask %arg29 : !ascvec.mask
              %202 = ascvec.loadalign %arg30[%c0_53] : memref<64xf32, 1>, index -> <f32>
              %203 = ascvec.ln %202, %mask : <f32>
              %204 = ascvec.loadalign %arg31[%c0_53] : memref<64xf32, 1>, index -> <f32>
              %205 = ascvec.add %203, %204, %mask : <f32>
              %206 = ascvec.cmp_eq_scalar %202, %cst, %mask : <f32>, f32 -> !ascvec.mask
              %207 = ascvec.duplicate_scalar_nomask %c2139095040_i32 : i32 -> <ui32>
              %208 = ascvec.reinterpret %207 : <ui32> -> <f32>
              %209 = ascvec.select %208, %205, %206 : <f32>
              ascvec.storealign %arg32[%c0_53], %209, %mask : memref<64xf32, 1>, index, <f32>
            }
          } {cannir.vf_mode = "simd"}
          ascvec.release_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
          ascvec.release_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
          %178 = cannir.datastruct_get %arg22["num_heads_q"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
          %179 = arith.extsi %178 : i32 to i64
          %180 = arith.muli %117, %179 : i64
          %181 = arith.addi %180, %145 : i64
          %182 = arith.muli %181, %c8192_i64 : i64
          %183 = ascscalar.memref_address %alloc_13 : memref<1xbf16>
          %184 = arith.divui %182, %c8_i64 : i64
          %185 = arith.addi %183, %184 : i64
          %186 = ascscalar.bind_gm_ptr %185 : memref<1xbf16>
          %187 = ascscalar.bitcast_memref %alloc_36 byte_offset = %c0_i64 : memref<8192xbf16, 1> -> memref<?xbf16, 1>
          %188 = arith.cmpi ugt, %144, %c0_i64 : i64
          scf.if %188 {
            ascvec.get_buf mte3, %c3_i64, 0 {cannir.buffer.generated_sync}
            ascvec.copy_ub2gm %186[%c0], %187[%c0], %157, %c1024_i32, %c1024_i64, %c1024_i32 l2_cache_ctl %c0_i32 : memref<1xbf16>, memref<?xbf16, 1>
            ascvec.release_buf mte3, %c3_i64, 0 {cannir.buffer.generated_sync}
          }
          %189 = cannir.datastruct_get %arg22["num_heads_q"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
          %190 = arith.extsi %189 : i32 to i64
          %191 = arith.muli %117, %190 : i64
          %192 = arith.addi %191, %145 : i64
          %193 = arith.muli %192, %c32_i64 : i64
          %194 = ascscalar.memref_address %alloc_14 : memref<1xf32>
          %195 = arith.divui %193, %c8_i64 : i64
          %196 = arith.addi %194, %195 : i64
          %197 = ascscalar.bind_gm_ptr %196 : memref<1xf32>
          %198 = ascscalar.bitcast_memref %alloc_40 byte_offset = %c0_i64 : memref<64xf32, 1> -> memref<?xf32, 1>
          %199 = arith.index_cast %144 : i64 to index
          %200 = arith.muli %199, %c4 : index
          scf.if %188 {
            %201 = arith.index_cast %200 : index to i64
            ascvec.get_buf mte3, %c4_i64, 0 {cannir.buffer.generated_sync}
            %202 = arith.trunci %201 : i64 to i32
            ascvec.copy_ub2gm %197[%c0], %198[%c0], %c1_i32, %202, %c0_i64, %202 l2_cache_ctl %c0_i32 : memref<1xf32>, memref<?xf32, 1>
            ascvec.release_buf mte3, %c4_i64, 0 {cannir.buffer.generated_sync}
          }
          scf.yield %177 : i64
        }
      }
    }
    ascvec.sync_intra_wait mte3, %c3_i64
    ascvec.sync_intra_wait mte3, %c4_i64
    ascvec.sync_intra_wait mte3, %c5_i64
    ascvec.sync_intra_wait mte3, %c6_i64
    ascvec.sync_intra_wait mte3, %c7_i64
    ascvec.sync_intra_wait mte3, %c8_i64
    ascvec.sync_intra_wait mte3, %c9_i64
    return
  }
  func.func @kernel_SparseN64Kernel___call___0_aic(%arg0: !cannir.tensor<<gm, bf16>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg1: !cannir.tensor<<gm, bf16>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg2: !cannir.tensor<<gm, bf16>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg3: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg4: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg5: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg6: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg7: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg8: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg9: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg10: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg11: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg12: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg13: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg14: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg15: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg16: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg17: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg18: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg19: !cannir.tensor<<gm, f32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg20: !cannir.tensor<<gm, bf16>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg21: !cannir.tensor<<gm, f32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg22: !cannir.datastruct<"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>, %arg23: !cannir.tensor<<gm, f32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>) attributes {cannir.buffer.auto_sync, cannir.buffer.sync_analyzed, cannir.buffer.sync_verified, cannir.buffer.tensor_state, cannir.core = "aic", cannir.cube_state_mode = 1 : i32, cannir.cube_state_planned, cannir.device, cannir.explicit_ring_buffer_cursors, cannir.kernel_group = "kernel_SparseN64Kernel___call___0", cannir.resource.ids_prepared} {
    %c900 = arith.constant 900 : index
    %c8384 = arith.constant 8384 : index
    %c192 = arith.constant 192 : index
    %c8320 = arith.constant 8320 : index
    %c128 = arith.constant 128 : index
    %c8256 = arith.constant 8256 : index
    %c64 = arith.constant 64 : index
    %c8192 = arith.constant 8192 : index
    %c4096 = arith.constant 4096 : index
    %c28_i32 = arith.constant 28 : i32
    %c24_i32 = arith.constant 24 : i32
    %c20_i32 = arith.constant 20 : i32
    %c12_i32 = arith.constant 12 : i32
    %c1024 = arith.constant 1024 : index
    %c8_i32 = arith.constant 8 : i32
    %c16_i64 = arith.constant 16 : i64
    %c9_i64 = arith.constant 9 : i64
    %c8_i64 = arith.constant 8 : i64
    %c7_i64 = arith.constant 7 : i64
    %c6_i64 = arith.constant 6 : i64
    %c5_i64 = arith.constant 5 : i64
    %c4_i64 = arith.constant 4 : i64
    %c3_i64 = arith.constant 3 : i64
    %c2_i64 = arith.constant 2 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %c10_i64 = arith.constant 10 : i64
    %c14_i64 = arith.constant 14 : i64
    %c15_i64 = arith.constant 15 : i64
    %c17_i64 = arith.constant 17 : i64
    %c18_i64 = arith.constant 18 : i64
    %c19_i64 = arith.constant 19 : i64
    %c20_i64 = arith.constant 20 : i64
    %c21_i64 = arith.constant 21 : i64
    %c22_i64 = arith.constant 22 : i64
    %c23_i64 = arith.constant 23 : i64
    %c24_i64 = arith.constant 24 : i64
    %c25_i64 = arith.constant 25 : i64
    %c127_i64 = arith.constant 127 : i64
    %c15 = arith.constant 15 : index
    %c384 = arith.constant 384 : index
    %c16 = arith.constant 16 : index
    %c384_i32 = arith.constant 384 : i32
    %c256_i32 = arith.constant 256 : i32
    %c16_i32 = arith.constant 16 : i32
    %c64_i32 = arith.constant 64 : i32
    %c128_i32 = arith.constant 128 : i32
    %c0 = arith.constant 0 : index
    %c256_i64 = arith.constant 256 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %c384_i64 = arith.constant 384 : i64
    %c64_i64 = arith.constant 64 : i64
    %c32768_i64 = arith.constant 32768 : i64
    %c512_i64 = arith.constant 512 : i64
    %c128_i64 = arith.constant 128 : i64
    %c4_i32 = arith.constant 4 : i32
    %c3_i32 = arith.constant 3 : i32
    %c1_i32 = arith.constant 1 : i32
    %false = arith.constant false
    %true = arith.constant true
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %alloc = memref.alloc() {cannir.arg_index = 0 : i64} : memref<1xbf16>
    %alloc_0 = memref.alloc() {cannir.arg_index = 9 : i64} : memref<1xi32>
    %alloc_1 = memref.alloc() {cannir.arg_index = 10 : i64} : memref<1xi32>
    %alloc_2 = memref.alloc() {cannir.arg_index = 11 : i64} : memref<1xi32>
    %alloc_3 = memref.alloc() {cannir.arg_index = 12 : i64} : memref<1xi32>
    %alloc_4 = memref.alloc() {cannir.arg_index = 13 : i64} : memref<1xi32>
    %alloc_5 = memref.alloc() {cannir.arg_index = 14 : i64} : memref<1xi32>
    %alloc_6 = memref.alloc() {cannir.arg_index = 15 : i64} : memref<1xi32>
    %alloc_7 = memref.alloc() {cannir.arg_index = 16 : i64} : memref<1xi32>
    %alloc_8 = memref.alloc() {cannir.arg_index = 17 : i64} : memref<1xi32>
    %alloc_9 = memref.alloc() {cannir.arg_index = 18 : i64} : memref<1xi32>
    %alloc_10 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<4096xf32, 1>
    %alloc_11 = memref.alloc() {cannir.byte_offset = 59904 : i64} : memref<16384xf32, 1>
    %alloc_12 = memref.alloc() {cannir.byte_offset = 125440 : i64} : memref<16384xf32, 1>
    %alloc_13 = memref.alloc() {cannir.byte_offset = 131072 : i64} : memref<8192xf32, 5>
    %alloc_14 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<8192xbf16, 3>
    %alloc_15 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<8192xbf16, 4>
    %alloc_16 = memref.alloc() {cannir.byte_offset = 16384 : i64} : memref<8192xbf16, 3>
    %alloc_17 = memref.alloc() {cannir.byte_offset = 16384 : i64} : memref<8192xbf16, 4>
    %alloc_18 = memref.alloc() {cannir.byte_offset = 32768 : i64} : memref<8192xbf16, 3>
    %alloc_19 = memref.alloc() {cannir.byte_offset = 32768 : i64} : memref<8192xbf16, 4>
    %alloc_20 = memref.alloc() {cannir.byte_offset = 49152 : i64} : memref<8192xbf16, 3>
    %alloc_21 = memref.alloc() {cannir.byte_offset = 49152 : i64} : memref<8192xbf16, 4>
    %alloc_22 = memref.alloc() {cannir.byte_offset = 327680 : i64} : memref<32768xbf16, 2>
    %alloc_23 = memref.alloc() {cannir.byte_offset = 458752 : i64} : memref<32768xbf16, 2>
    %alloc_24 = memref.alloc() {cannir.byte_offset = 393216 : i64} : memref<32768xbf16, 2>
    %alloc_25 = memref.alloc() {cannir.byte_offset = 114688 : i64} : memref<32768xbf16, 2>
    %alloc_26 = memref.alloc() {cannir.byte_offset = 245760 : i64} : memref<32768xbf16, 2>
    %alloc_27 = memref.alloc() {cannir.byte_offset = 180224 : i64} : memref<32768xbf16, 2>
    %alloc_28 = memref.alloc() {cannir.byte_offset = 311296 : i64} : memref<8192xbf16, 2>
    %alloc_29 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<4096xf32, 5>
    %alloc_30 = memref.alloc() {cannir.byte_offset = 16384 : i64} : memref<4096xf32, 5>
    %alloc_31 = memref.alloc() {cannir.byte_offset = 32768 : i64} : memref<4096xf32, 5>
    %alloc_32 = memref.alloc() {cannir.byte_offset = 49152 : i64} : memref<4096xf32, 5>
    %alloc_33 = memref.alloc() {cannir.byte_offset = 65536 : i64} : memref<4096xf32, 5>
    %alloc_34 = memref.alloc() {cannir.byte_offset = 81920 : i64} : memref<4096xf32, 5>
    %alloc_35 = memref.alloc() {cannir.byte_offset = 114688 : i64} : memref<4096xf32, 5>
    %alloc_36 = memref.alloc() {cannir.byte_offset = 98304 : i64} : memref<4096xf32, 5>
    %alloc_37 = memref.alloc() {cannir.byte_offset = 98304 : i64} : memref<8192xbf16, 2>
    %alloc_38 = memref.alloc() {cannir.byte_offset = 49152 : i64} : memref<24576xbf16, 2>
    %alloc_39 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<24576xbf16, 2>
    asccube.sync_intra_arrive mte1, %c9_i64
    asccube.sync_intra_arrive mte1, %c25_i64
    asccube.sync_intra_arrive mte1, %c8_i64
    asccube.sync_intra_arrive mte1, %c24_i64
    asccube.sync_intra_arrive mte1, %c7_i64
    asccube.sync_intra_arrive mte1, %c23_i64
    asccube.sync_intra_arrive mte1, %c6_i64
    asccube.sync_intra_arrive mte1, %c22_i64
    asccube.sync_intra_arrive mte1, %c5_i64
    asccube.sync_intra_arrive mte1, %c21_i64
    asccube.sync_intra_arrive mte1, %c4_i64
    asccube.sync_intra_arrive mte1, %c20_i64
    asccube.sync_intra_arrive mte1, %c3_i64
    asccube.sync_intra_arrive mte1, %c19_i64
    %0 = ascscalar.get_block_idx
    %1 = arith.muli %0, %c9_i64 : i64
    %2 = arith.addi %1, %c3_i64 : i64
    %3 = arith.index_cast %2 : i64 to index
    %4 = memref.load %alloc_5[%3] : memref<1xi32>
    %5 = arith.extsi %4 : i32 to i64
    %6 = arith.addi %1, %c6_i64 : i64
    %7 = arith.index_cast %6 : i64 to index
    %8 = memref.load %alloc_5[%7] : memref<1xi32>
    %9 = arith.extsi %8 : i32 to i64
    %10 = arith.addi %1, %c1_i64 : i64
    %11 = arith.addi %1, %c4_i64 : i64
    %12 = arith.addi %1, %c2_i64 : i64
    %13 = arith.addi %1, %c5_i64 : i64
    %14 = arith.index_cast %1 : i64 to index
    %15 = memref.load %alloc_5[%14] : memref<1xi32>
    %16 = arith.cmpi ne, %15, %c0_i32 : i32
    %17 = scf.if %16 -> (i64) {
      %24 = arith.index_cast %10 : i64 to index
      %25 = memref.load %alloc_5[%24] : memref<1xi32>
      %26 = arith.extsi %25 : i32 to i64
      %27 = arith.index_cast %12 : i64 to index
      %28 = memref.load %alloc_5[%27] : memref<1xi32>
      %29 = arith.extsi %28 : i32 to i64
      %30 = arith.index_cast %11 : i64 to index
      %31 = memref.load %alloc_5[%30] : memref<1xi32>
      %32 = arith.extsi %31 : i32 to i64
      %33 = arith.index_cast %13 : i64 to index
      %34 = memref.load %alloc_5[%33] : memref<1xi32>
      %35 = arith.extsi %34 : i32 to i64
      %36 = arith.cmpi sgt, %9, %c0_i64 : i64
      %37 = arith.extui %36 : i1 to i64
      %38 = arith.addi %35, %37 : i64
      %39 = cannir.datastruct_get %arg22["batch_count"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
      %40 = arith.extsi %39 : i32 to i64
      %41 = arith.cmpi slt, %32, %40 : i64
      %42 = arith.addi %32, %c1_i64 : i64
      %43 = arith.select %41, %42, %40 : i64
      %44 = scf.for %arg24 = %26 to %43 step %c1_i64 iter_args(%arg25 = %c0_i64) -> (i64)  : i64 {
        %45 = cannir.datastruct_get %arg22["query_rows_per_batch"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
        %46 = arith.extsi %45 : i32 to i64
        %47 = cannir.datastruct_get %arg22["batch_mode"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
        %48 = arith.cmpi eq, %47, %c2_i32 : i32
        %49 = scf.if %48 -> (i64) {
          %58 = arith.addi %arg24, %c1_i64 : i64
          %59 = arith.index_cast %58 : i64 to index
          %60 = memref.load %alloc_7[%59] : memref<1xi32>
          %61 = arith.index_cast %arg24 : i64 to index
          %62 = memref.load %alloc_7[%61] : memref<1xi32>
          %63 = arith.subi %60, %62 : i32
          %64 = arith.extsi %63 : i32 to i64
          scf.yield %64 : i64
        } else {
          scf.yield %46 : i64
        }
        %50 = arith.cmpi eq, %arg24, %26 : i64
        %51 = arith.select %50, %29, %c0_i64 : i64
        %52 = arith.cmpi eq, %arg24, %32 : i64
        %53 = arith.select %52, %38, %49 : i64
        %54 = arith.cmpi sgt, %53, %51 : i64
        %55 = arith.subi %53, %51 : i64
        %56 = arith.select %54, %55, %c0_i64 : i64
        %57 = arith.addi %arg25, %56 : i64
        scf.yield %57 : i64
      }
      scf.yield %44 : i64
    } else {
      scf.yield %c0_i64 : i64
    }
    %18 = arith.cmpi sgt, %17, %c0_i64 : i64
    %19 = arith.extui %18 : i1 to i64
    %20:12 = scf.while (%arg24 = %c0_i64, %arg25 = %c0_i64, %arg26 = %c0_i64, %arg27 = %c0_i64, %arg28 = %c0_i64, %arg29 = %c0_i64, %arg30 = %c0_i64, %arg31 = %c0_i64, %arg32 = %c0_i64, %arg33 = %c0_i64, %arg34 = %c0_i64, %arg35 = %c0_i64, %arg36 = %19) : (i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> (i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) {
      %24 = arith.cmpi ne, %arg36, %c0_i64 : i64
      scf.condition(%24) %arg24, %arg25, %arg26, %arg27, %arg28, %arg29, %arg30, %arg31, %arg32, %arg33, %arg34, %arg35 : i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64
    } do {
    ^bb0(%arg24: i64, %arg25: i64, %arg26: i64, %arg27: i64, %arg28: i64, %arg29: i64, %arg30: i64, %arg31: i64, %arg32: i64, %arg33: i64, %arg34: i64, %arg35: i64):
      %24 = memref.load %alloc_5[%14] : memref<1xi32>
      %25 = arith.cmpi ne, %24, %c0_i32 : i32
      %26:5 = scf.if %25 -> (i64, i64, i64, i64, i64) {
        %48 = arith.index_cast %10 : i64 to index
        %49 = memref.load %alloc_5[%48] : memref<1xi32>
        %50 = arith.extsi %49 : i32 to i64
        %51 = arith.index_cast %12 : i64 to index
        %52 = memref.load %alloc_5[%51] : memref<1xi32>
        %53 = arith.extsi %52 : i32 to i64
        %54 = arith.index_cast %11 : i64 to index
        %55 = memref.load %alloc_5[%54] : memref<1xi32>
        %56 = arith.extsi %55 : i32 to i64
        %57 = arith.index_cast %13 : i64 to index
        %58 = memref.load %alloc_5[%57] : memref<1xi32>
        %59 = arith.extsi %58 : i32 to i64
        %60 = arith.cmpi sgt, %9, %c0_i64 : i64
        %61 = arith.extui %60 : i1 to i64
        %62 = arith.addi %59, %61 : i64
        %63 = cannir.datastruct_get %arg22["batch_mode"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
        %64 = arith.cmpi eq, %63, %c2_i32 : i32
        %65:5 = scf.if %64 -> (i64, i64, i64, i64, i64) {
          %66 = arith.index_cast %49 : i32 to index
          %67 = memref.load %alloc_7[%66] : memref<1xi32>
          %68 = arith.extsi %67 : i32 to i64
          %69 = arith.addi %68, %53 : i64
          %70 = cannir.datastruct_get %arg22["batch_count"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
          %71 = arith.extsi %70 : i32 to i64
          %72 = arith.index_cast %70 : i32 to index
          %73 = memref.load %alloc_7[%72] : memref<1xi32>
          %74 = arith.extsi %73 : i32 to i64
          %75 = arith.cmpi slt, %56, %71 : i64
          %76 = scf.if %75 -> (i64) {
            %80 = arith.index_cast %55 : i32 to index
            %81 = memref.load %alloc_7[%80] : memref<1xi32>
            %82 = arith.extsi %81 : i32 to i64
            %83 = arith.addi %56, %c1_i64 : i64
            %84 = arith.index_cast %83 : i64 to index
            %85 = memref.load %alloc_7[%84] : memref<1xi32>
            %86 = arith.extsi %85 : i32 to i64
            %87 = arith.subi %86, %82 : i64
            %88 = arith.cmpi slt, %62, %87 : i64
            %89 = arith.select %88, %62, %87 : i64
            %90 = arith.addi %82, %89 : i64
            scf.yield %90 : i64
          } else {
            scf.yield %74 : i64
          }
          %77 = arith.addi %69, %arg35 : i64
          %78 = arith.cmpi sge, %arg35, %c0_i64 : i64
          %79:5 = scf.if %78 -> (i64, i64, i64, i64, i64) {
            %80 = arith.cmpi slt, %77, %76 : i64
            %81 = arith.select %80, %77, %c0_i64 : i64
            %82 = arith.extui %80 : i1 to i64
            %83:3 = scf.if %80 -> (i64, i64, i64) {
              %84 = cannir.datastruct_get %arg22["batch_count"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
              %85 = arith.extsi %84 : i32 to i64
              %86 = arith.cmpi slt, %56, %85 : i64
              %87 = arith.subi %85, %c1_i64 : i64
              %88 = arith.select %86, %56, %87 : i64
              %89 = memref.load %alloc_7[%66] : memref<1xi32>
              %90 = arith.extsi %89 : i32 to i64
              %91 = arith.addi %50, %c1_i64 : i64
              %92 = arith.index_cast %91 : i64 to index
              %93 = memref.load %alloc_7[%92] : memref<1xi32>
              %94 = arith.extsi %93 : i32 to i64
              %95 = arith.cmpi slt, %50, %88 : i64
              %96:3 = scf.if %95 -> (i64, i64, i64) {
                %102 = memref.load %alloc_7[%c0] : memref<1xi32>
                %103 = arith.extsi %102 : i32 to i64
                %104 = cannir.datastruct_get %arg22["batch_count"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
                %105 = arith.extsi %104 : i32 to i64
                %106 = arith.index_cast %104 : i32 to index
                %107 = memref.load %alloc_7[%106] : memref<1xi32>
                %108 = arith.extsi %107 : i32 to i64
                %109 = arith.subi %108, %103 : i64
                %110 = arith.remsi %109, %105 : i64
                %111 = arith.cmpi ne, %110, %c0_i64 : i64
                %112 = arith.cmpi slt, %110, %c0_i64 : i64
                %113 = arith.cmpi slt, %105, %c0_i64 : i64
                %114 = arith.xori %112, %113 : i1
                %115 = arith.andi %111, %114 : i1
                %116 = arith.divsi %109, %105 : i64
                %117 = arith.extui %115 : i1 to i64
                %118 = arith.subi %116, %117 : i64
                %119 = arith.cmpi sgt, %118, %c0_i64 : i64
                %120 = arith.select %119, %118, %c1_i64 : i64
                %121 = arith.subi %77, %103 : i64
                %122 = arith.remsi %121, %120 : i64
                %123 = arith.cmpi ne, %122, %c0_i64 : i64
                %124 = arith.cmpi slt, %122, %c0_i64 : i64
                %125 = arith.andi %123, %124 : i1
                %126 = arith.divsi %121, %120 : i64
                %127 = arith.extui %125 : i1 to i64
                %128 = arith.subi %126, %127 : i64
                %129 = arith.cmpi slt, %128, %50 : i64
                %130 = arith.select %129, %50, %128 : i64
                %131 = arith.cmpi sgt, %130, %88 : i64
                %132 = arith.select %131, %88, %130 : i64
                %133 = arith.index_cast %132 : i64 to index
                %134 = memref.load %alloc_7[%133] : memref<1xi32>
                %135 = arith.extsi %134 : i32 to i64
                %136 = arith.addi %132, %c1_i64 : i64
                %137 = arith.index_cast %136 : i64 to index
                %138 = memref.load %alloc_7[%137] : memref<1xi32>
                %139 = arith.extsi %138 : i32 to i64
                %140 = arith.cmpi slt, %77, %135 : i64
                %141 = arith.cmpi sge, %77, %139 : i64
                %142 = arith.select %140, %true, %141 : i1
                %143:3 = scf.if %142 -> (i64, i64, i64) {
                  %144:2 = scf.while (%arg36 = %50, %arg37 = %88) : (i64, i64) -> (i64, i64) {
                    %152 = arith.cmpi slt, %arg36, %arg37 : i64
                    scf.condition(%152) %arg36, %arg37 : i64, i64
                  } do {
                  ^bb0(%arg36: i64, %arg37: i64):
                    %152 = arith.addi %arg36, %arg37 : i64
                    %153 = arith.remsi %152, %c2_i64 : i64
                    %154 = arith.cmpi ne, %153, %c0_i64 : i64
                    %155 = arith.cmpi slt, %153, %c0_i64 : i64
                    %156 = arith.andi %154, %155 : i1
                    %157 = arith.divsi %152, %c2_i64 : i64
                    %158 = arith.extui %156 : i1 to i64
                    %159 = arith.subi %157, %158 : i64
                    %160 = arith.addi %159, %c1_i64 : i64
                    %161 = arith.index_cast %160 : i64 to index
                    %162 = memref.load %alloc_7[%161] : memref<1xi32>
                    %163 = arith.extsi %162 : i32 to i64
                    %164 = arith.cmpi sle, %163, %77 : i64
                    %165 = arith.select %164, %160, %arg36 : i64
                    %166 = arith.select %164, %arg37, %159 : i64
                    scf.yield %165, %166 : i64, i64
                  }
                  %145 = arith.index_cast %144#0 : i64 to index
                  %146 = memref.load %alloc_7[%145] : memref<1xi32>
                  %147 = arith.extsi %146 : i32 to i64
                  %148 = arith.addi %144#0, %c1_i64 : i64
                  %149 = arith.index_cast %148 : i64 to index
                  %150 = memref.load %alloc_7[%149] : memref<1xi32>
                  %151 = arith.extsi %150 : i32 to i64
                  scf.yield %144#0, %147, %151 : i64, i64, i64
                } else {
                  scf.yield %132, %135, %139 : i64, i64, i64
                }
                scf.yield %143#0, %143#1, %143#2 : i64, i64, i64
              } else {
                scf.yield %50, %90, %94 : i64, i64, i64
              }
              %97 = arith.subi %77, %96#1 : i64
              %98 = arith.subi %96#2, %96#1 : i64
              %99 = cannir.datastruct_get %arg22["has_seqused_q"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
              %100 = arith.cmpi ne, %99, %c0_i32 : i32
              %101 = scf.if %100 -> (i64) {
                %102 = arith.index_cast %96#0 : i64 to index
                %103 = memref.load %alloc_6[%102] : memref<1xi32>
                %104 = arith.extsi %103 : i32 to i64
                scf.yield %104 : i64
              } else {
                scf.yield %98 : i64
              }
              scf.yield %96#0, %97, %101 : i64, i64, i64
            } else {
              scf.yield %c0_i64, %c0_i64, %c0_i64 : i64, i64, i64
            }
            scf.yield %81, %83#0, %83#1, %83#2, %82 : i64, i64, i64, i64, i64
          } else {
            scf.yield %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64 : i64, i64, i64, i64, i64
          }
          scf.yield %79#0, %79#1, %79#2, %79#3, %79#4 : i64, i64, i64, i64, i64
        } else {
          %66 = cannir.datastruct_get %arg22["query_rows_per_batch"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
          %67 = arith.extsi %66 : i32 to i64
          %68 = arith.muli %50, %67 : i64
          %69 = arith.addi %68, %53 : i64
          %70 = arith.muli %56, %67 : i64
          %71 = arith.cmpi slt, %62, %67 : i64
          %72 = arith.select %71, %62, %67 : i64
          %73 = arith.addi %70, %72 : i64
          %74 = arith.addi %69, %arg35 : i64
          %75 = arith.cmpi sge, %arg35, %c0_i64 : i64
          %76:5 = scf.if %75 -> (i64, i64, i64, i64, i64) {
            %77 = arith.cmpi slt, %74, %73 : i64
            %78 = arith.select %77, %74, %c0_i64 : i64
            %79 = arith.extui %77 : i1 to i64
            %80:3 = scf.if %77 -> (i64, i64, i64) {
              %81 = arith.remsi %74, %67 : i64
              %82 = arith.cmpi ne, %81, %c0_i64 : i64
              %83 = arith.cmpi slt, %81, %c0_i64 : i64
              %84 = arith.cmpi slt, %67, %c0_i64 : i64
              %85 = arith.xori %83, %84 : i1
              %86 = arith.andi %82, %85 : i1
              %87 = arith.divsi %74, %67 : i64
              %88 = arith.extui %86 : i1 to i64
              %89 = arith.subi %87, %88 : i64
              %90 = arith.muli %89, %67 : i64
              %91 = arith.subi %74, %90 : i64
              %92 = cannir.datastruct_get %arg22["has_seqused_q"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
              %93 = arith.cmpi ne, %92, %c0_i32 : i32
              %94 = scf.if %93 -> (i64) {
                %95 = arith.index_cast %89 : i64 to index
                %96 = memref.load %alloc_6[%95] : memref<1xi32>
                %97 = arith.extsi %96 : i32 to i64
                scf.yield %97 : i64
              } else {
                scf.yield %67 : i64
              }
              scf.yield %89, %91, %94 : i64, i64, i64
            } else {
              scf.yield %c0_i64, %c0_i64, %c0_i64 : i64, i64, i64
            }
            scf.yield %78, %80#0, %80#1, %80#2, %79 : i64, i64, i64, i64, i64
          } else {
            scf.yield %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64 : i64, i64, i64, i64, i64
          }
          scf.yield %76#0, %76#1, %76#2, %76#3, %76#4 : i64, i64, i64, i64, i64
        }
        scf.yield %65#0, %65#1, %65#2, %65#3, %65#4 : i64, i64, i64, i64, i64
      } else {
        scf.yield %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64 : i64, i64, i64, i64, i64
      }
      %27 = arith.cmpi ne, %26#4, %c0_i64 : i64
      %28 = arith.cmpi slt, %26#2, %26#3 : i64
      %29 = arith.select %27, %28, %false : i1
      %30:9 = scf.if %29 -> (i64, i64, i64, i64, i64, i64, i64, i64, i64) {
        %48 = cannir.datastruct_get %arg22["ori_kv_capacity"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
        %49 = arith.extsi %48 : i32 to i64
        %50 = arith.muli %26#1, %49 : i64
        %51 = cannir.datastruct_get %arg22["cmp_kv_capacity"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
        %52 = arith.extsi %51 : i32 to i64
        %53 = arith.muli %26#1, %52 : i64
        %54 = cannir.datastruct_get %arg22["kv_mode"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
        %55 = arith.cmpi eq, %54, %c1_i32 : i32
        %56:4 = scf.if %55 -> (i64, i64, i64, i64) {
          scf.yield %c0_i64, %c0_i64, %49, %52 : i64, i64, i64, i64
        } else {
          %103 = cannir.datastruct_get %arg22["batch_mode"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
          %104 = arith.cmpi eq, %103, %c2_i32 : i32
          %105:4 = scf.if %104 -> (i64, i64, i64, i64) {
            %106 = arith.index_cast %26#1 : i64 to index
            %107 = memref.load %alloc_8[%106] : memref<1xi32>
            %108 = arith.extsi %107 : i32 to i64
            %109 = arith.addi %26#1, %c1_i64 : i64
            %110 = arith.index_cast %109 : i64 to index
            %111 = memref.load %alloc_8[%110] : memref<1xi32>
            %112 = arith.extsi %111 : i32 to i64
            %113 = arith.subi %112, %108 : i64
            %114 = cannir.datastruct_get %arg22["has_cmp"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
            %115 = arith.cmpi ne, %114, %c0_i32 : i32
            %116:2 = scf.if %115 -> (i64, i64) {
              %117 = memref.load %alloc_9[%106] : memref<1xi32>
              %118 = arith.extsi %117 : i32 to i64
              %119 = memref.load %alloc_9[%110] : memref<1xi32>
              %120 = arith.extsi %119 : i32 to i64
              %121 = arith.subi %120, %118 : i64
              scf.yield %118, %121 : i64, i64
            } else {
              scf.yield %53, %52 : i64, i64
            }
            scf.yield %108, %113, %116#0, %116#1 : i64, i64, i64, i64
          } else {
            scf.yield %50, %49, %53, %52 : i64, i64, i64, i64
          }
          scf.yield %105#0, %105#2, %105#1, %105#3 : i64, i64, i64, i64
        }
        %57 = cannir.datastruct_get %arg22["has_cmp"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
        %58 = arith.cmpi eq, %57, %c0_i32 : i32
        %59 = arith.select %58, %c0_i64, %56#1 : i64
        %60 = arith.select %58, %c0_i64, %56#3 : i64
        %61 = cannir.datastruct_get %arg22["has_seqused_ori_kv"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
        %62 = arith.cmpi ne, %61, %c0_i32 : i32
        %63 = scf.if %62 -> (i64) {
          %103 = arith.index_cast %26#1 : i64 to index
          %104 = memref.load %alloc_2[%103] : memref<1xi32>
          %105 = arith.extsi %104 : i32 to i64
          %106 = arith.cmpi sgt, %105, %c0_i64 : i64
          %107 = arith.cmpi slt, %105, %56#2 : i64
          %108 = arith.select %107, %105, %56#2 : i64
          %109 = arith.select %106, %108, %c0_i64 : i64
          scf.yield %109 : i64
        } else {
          scf.yield %56#2 : i64
        }
        %64 = arith.cmpi ne, %57, %c0_i32 : i32
        %65 = scf.if %64 -> (i1) {
          %103 = cannir.datastruct_get %arg22["has_seqused_cmp_kv"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
          %104 = arith.cmpi ne, %103, %c0_i32 : i32
          scf.yield %104 : i1
        } else {
          scf.yield %false : i1
        }
        %66 = scf.if %65 -> (i64) {
          %103 = arith.index_cast %26#1 : i64 to index
          %104 = memref.load %alloc_3[%103] : memref<1xi32>
          %105 = arith.extsi %104 : i32 to i64
          %106 = arith.cmpi sgt, %105, %c0_i64 : i64
          %107 = arith.cmpi slt, %105, %60 : i64
          %108 = arith.select %107, %105, %60 : i64
          %109 = arith.select %106, %108, %c0_i64 : i64
          scf.yield %109 : i64
        } else {
          scf.yield %60 : i64
        }
        %67 = cannir.datastruct_get %arg22["ori_mask_mode"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
        %68 = arith.cmpi eq, %67, %c3_i32 : i32
        %69 = arith.subi %63, %26#3 : i64
        %70 = arith.addi %69, %26#2 : i64
        %71 = arith.addi %70, %c1_i64 : i64
        %72 = arith.cmpi sgt, %71, %c0_i64 : i64
        %73 = arith.cmpi slt, %71, %63 : i64
        %74 = arith.select %73, %71, %63 : i64
        %75 = arith.select %72, %74, %c0_i64 : i64
        %76 = arith.select %68, %75, %63 : i64
        %77 = arith.cmpi eq, %67, %c4_i32 : i32
        %78:2 = scf.if %77 -> (i64, i64) {
          %103 = cannir.datastruct_get %arg22["ori_win_left"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
          %104 = arith.cmpi sge, %103, %c0_i32 : i32
          %105 = scf.if %104 -> (i64) {
            %109 = cannir.datastruct_get %arg22["ori_win_left"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
            %110 = arith.extsi %109 : i32 to i64
            %111 = arith.subi %70, %110 : i64
            %112 = arith.cmpi sgt, %111, %c0_i64 : i64
            %113 = arith.cmpi slt, %111, %63 : i64
            %114 = arith.select %113, %111, %63 : i64
            %115 = arith.select %112, %114, %c0_i64 : i64
            scf.yield %115 : i64
          } else {
            scf.yield %c0_i64 : i64
          }
          %106 = cannir.datastruct_get %arg22["ori_win_right"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
          %107 = arith.cmpi sge, %106, %c0_i32 : i32
          %108 = scf.if %107 -> (i64) {
            %109 = cannir.datastruct_get %arg22["ori_win_right"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
            %110 = arith.extsi %109 : i32 to i64
            %111 = arith.addi %70, %110 : i64
            %112 = arith.addi %111, %c1_i64 : i64
            %113 = arith.cmpi sgt, %112, %c0_i64 : i64
            %114 = arith.cmpi slt, %112, %63 : i64
            %115 = arith.select %114, %112, %63 : i64
            %116 = arith.select %113, %115, %c0_i64 : i64
            scf.yield %116 : i64
          } else {
            scf.yield %76 : i64
          }
          scf.yield %105, %108 : i64, i64
        } else {
          scf.yield %c0_i64, %76 : i64, i64
        }
        %79 = arith.cmpi sgt, %78#1, %78#0 : i64
        %80 = arith.subi %78#1, %78#0 : i64
        %81 = arith.select %79, %80, %c0_i64 : i64
        %82 = cannir.datastruct_get %arg22["is_ori_dense"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
        %83 = arith.cmpi eq, %82, %c0_i32 : i32
        %84 = scf.if %83 -> (i64) {
          %103 = cannir.datastruct_get %arg22["ori_sparse_index_width"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
          %104 = arith.extsi %103 : i32 to i64
          %105 = arith.cmpi slt, %81, %104 : i64
          %106 = arith.select %105, %81, %104 : i64
          %107 = cannir.datastruct_get %arg22["has_ori_topk_length"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
          %108 = arith.cmpi ne, %107, %c0_i32 : i32
          %109 = scf.if %108 -> (i64) {
            %110 = arith.index_cast %26#0 : i64 to index
            %111 = memref.load %alloc_0[%110] : memref<1xi32>
            %112 = arith.extsi %111 : i32 to i64
            %113 = arith.cmpi sgt, %112, %c0_i64 : i64
            %114 = arith.cmpi slt, %106, %112 : i64
            %115 = arith.select %114, %106, %112 : i64
            %116 = arith.select %113, %115, %c0_i64 : i64
            scf.yield %116 : i64
          } else {
            scf.yield %106 : i64
          }
          scf.yield %109 : i64
        } else {
          scf.yield %81 : i64
        }
        %85 = scf.if %64 -> (i64) {
          %103 = cannir.datastruct_get %arg22["cmp_mask_mode"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
          %104 = arith.cmpi eq, %103, %c3_i32 : i32
          %105 = scf.if %104 -> (i64) {
            %113 = cannir.datastruct_get %arg22["cmp_ratio"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
            %114 = arith.extsi %113 : i32 to i64
            %115 = arith.muli %66, %114 : i64
            %116 = arith.cmpi ne, %113, %c1_i32 : i32
            %117 = scf.if %116 -> (i64) {
              %134 = arith.index_cast %26#1 : i64 to index
              %135 = memref.load %alloc_4[%134] : memref<1xi32>
              %136 = arith.extsi %135 : i32 to i64
              %137 = arith.addi %115, %136 : i64
              scf.yield %137 : i64
            } else {
              scf.yield %115 : i64
            }
            %118 = arith.subi %117, %26#3 : i64
            %119 = arith.addi %118, %26#2 : i64
            %120 = arith.addi %119, %c1_i64 : i64
            %121 = arith.remsi %120, %114 : i64
            %122 = arith.cmpi ne, %121, %c0_i64 : i64
            %123 = arith.cmpi slt, %121, %c0_i64 : i64
            %124 = arith.cmpi slt, %114, %c0_i64 : i64
            %125 = arith.xori %123, %124 : i1
            %126 = arith.andi %122, %125 : i1
            %127 = arith.divsi %120, %114 : i64
            %128 = arith.extui %126 : i1 to i64
            %129 = arith.subi %127, %128 : i64
            %130 = arith.cmpi sgt, %129, %c0_i64 : i64
            %131 = arith.cmpi slt, %129, %66 : i64
            %132 = arith.select %131, %129, %66 : i64
            %133 = arith.select %130, %132, %c0_i64 : i64
            scf.yield %133 : i64
          } else {
            scf.yield %66 : i64
          }
          %106 = cannir.datastruct_get %arg22["cmp_sparse_index_width"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
          %107 = arith.extsi %106 : i32 to i64
          %108 = arith.cmpi slt, %105, %107 : i64
          %109 = arith.select %108, %105, %107 : i64
          %110 = cannir.datastruct_get %arg22["has_cmp_topk_length"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
          %111 = arith.cmpi ne, %110, %c0_i32 : i32
          %112 = scf.if %111 -> (i64) {
            %113 = arith.index_cast %26#0 : i64 to index
            %114 = memref.load %alloc_1[%113] : memref<1xi32>
            %115 = arith.extsi %114 : i32 to i64
            %116 = arith.cmpi sgt, %115, %c0_i64 : i64
            %117 = arith.cmpi slt, %109, %115 : i64
            %118 = arith.select %117, %109, %115 : i64
            %119 = arith.select %116, %118, %c0_i64 : i64
            scf.yield %119 : i64
          } else {
            scf.yield %109 : i64
          }
          scf.yield %112 : i64
        } else {
          scf.yield %c0_i64 : i64
        }
        %86 = arith.addi %84, %c127_i64 : i64
        %87 = arith.remsi %86, %c128_i64 : i64
        %88 = arith.cmpi ne, %87, %c0_i64 : i64
        %89 = arith.cmpi slt, %87, %c0_i64 : i64
        %90 = arith.andi %88, %89 : i1
        %91 = arith.divsi %86, %c128_i64 : i64
        %92 = arith.extui %90 : i1 to i64
        %93 = arith.subi %91, %92 : i64
        %94 = arith.addi %85, %c127_i64 : i64
        %95 = arith.remsi %94, %c128_i64 : i64
        %96 = arith.cmpi ne, %95, %c0_i64 : i64
        %97 = arith.cmpi slt, %95, %c0_i64 : i64
        %98 = arith.andi %96, %97 : i1
        %99 = arith.divsi %94, %c128_i64 : i64
        %100 = arith.extui %98 : i1 to i64
        %101 = arith.subi %99, %100 : i64
        %102 = arith.addi %93, %101 : i64
        scf.yield %56#0, %59, %56#2, %60, %78#0, %84, %85, %93, %102 : i64, i64, i64, i64, i64, i64, i64, i64, i64
      } else {
        scf.yield %arg25, %arg26, %arg27, %arg28, %arg29, %arg30, %arg31, %arg32, %arg33 : i64, i64, i64, i64, i64, i64, i64, i64, i64
      }
      %31 = arith.cmpi eq, %arg35, %c0_i64 : i64
      %32 = arith.select %31, %5, %c0_i64 : i64
      %33 = arith.addi %arg35, %c1_i64 : i64
      %34 = arith.cmpi eq, %33, %17 : i64
      %35 = arith.cmpi sgt, %9, %c0_i64 : i64
      %36 = arith.select %34, %35, %false : i1
      %37 = arith.minsi %30#8, %9 : i64
      %38 = arith.select %36, %37, %30#8 : i64
      %39 = arith.subi %38, %32 : i64
      %40 = arith.maxsi %39, %c0_i64 : i64
      %41 = arith.cmpi sgt, %40, %c0_i64 : i64
      %42 = arith.extui %41 : i1 to i64
      %43 = arith.cmpi eq, %42, %c0_i64 : i64
      %44 = arith.select %43, %33, %arg35 : i64
      %45 = arith.cmpi slt, %44, %17 : i64
      %46 = arith.extui %43 : i1 to i64
      %47 = arith.select %45, %46, %c0_i64 : i64
      scf.yield %26#0, %30#0, %30#1, %30#2, %30#3, %30#4, %30#5, %30#6, %30#7, %40, %42, %44, %47 : i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64
    }
    %21 = arith.cmpi ne, %20#10, %c0_i64 : i64
    scf.if %21 {
      %24 = cannir.datastruct_get %arg22["num_heads_q"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
      %25 = arith.extsi %24 : i32 to i64
      %26 = arith.muli %20#0, %25 : i64
      %27 = arith.muli %26, %c512_i64 : i64
      %28 = ascscalar.bitcast_memref %alloc_39 byte_offset = %c0_i64 : memref<24576xbf16, 2> -> memref<?xbf16, 2>
      %29 = arith.divui %27, %c16_i64 : i64
      %30 = arith.muli %29, %c16_i64 : i64
      asccube.get_buf mte2, %c0_i64, 0 {cannir.buffer.generated_sync}
      %31 = arith.index_cast %30 : i64 to index
      asccube.copy_gm2l1 %28[%c0], %alloc[%31], %c64_i32, %c384_i32, %c1024_i64, %c1_i32, %c0_i64, %c0_i64, %c64_i32 l2_cache_ctl %c4_i32 {nd2nz = true} : memref<?xbf16, 2>, memref<1xbf16>
      asccube.release_buf mte2, %c0_i64, 0 {cannir.buffer.generated_sync}
      %32 = cannir.datastruct_get %arg22["num_heads_q"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
      %33 = arith.extsi %32 : i32 to i64
      %34 = arith.muli %20#0, %33 : i64
      %35 = arith.muli %34, %c512_i64 : i64
      %36 = ascscalar.bitcast_memref %alloc_37 byte_offset = %c0_i64 : memref<8192xbf16, 2> -> memref<?xbf16, 2>
      %37 = arith.divui %35, %c16_i64 : i64
      %38 = arith.muli %37, %c16_i64 : i64
      %39 = arith.index_cast %38 : i64 to index
      %40 = arith.addi %39, %c384 : index
      asccube.get_buf mte2, %c2_i64, 0 {cannir.buffer.generated_sync}
      asccube.copy_gm2l1 %36[%c0], %alloc[%40], %c64_i32, %c128_i32, %c1024_i64, %c1_i32, %c0_i64, %c0_i64, %c64_i32 l2_cache_ctl %c4_i32 {nd2nz = true} : memref<?xbf16, 2>, memref<1xbf16>
      asccube.release_buf mte2, %c2_i64, 0 {cannir.buffer.generated_sync}
      asccube.set_mmad_direction m
      %41 = ascscalar.bitcast_memref %alloc_14 byte_offset = %c0_i64 : memref<8192xbf16, 3> -> memref<?xbf16, 3>
      asccube.get_buf mte1, %c3_i64, 0 {cannir.buffer.generated_sync}
      asccube.get_buf mte1, %c0_i64, 0 {cannir.buffer.generated_sync}
      asccube.copy_l12l0a %41[%c0], %28[%c0], %c0_i32, %c0_i32, %c4_i32, %c4_i32, %c4_i32, %c4_i32 : memref<?xbf16, 3>, memref<?xbf16, 2>
      asccube.release_buf mte1, %c0_i64, 0 {cannir.buffer.generated_sync}
      asccube.release_buf mte1, %c3_i64, 0 {cannir.buffer.generated_sync}
      %42 = ascscalar.bitcast_memref %alloc_15 byte_offset = %c0_i64 : memref<8192xbf16, 4> -> memref<?xbf16, 4>
      %43 = ascscalar.bitcast_memref %alloc_25 byte_offset = %c0_i64 : memref<32768xbf16, 2> -> memref<?xbf16, 2>
      asccube.sync_intra_wait mte1, %c3_i64 {cannir.sync.banks = 1 : i32}
      asccube.get_buf mte1, %c7_i64, 0 {cannir.buffer.generated_sync}
      asccube.copy_l12l0b %42[%c0], %43[%c0], %c0_i32, %c0_i32, %c4_i32, %c4_i32, %c4_i32, %c8_i32 : memref<?xbf16, 4>, memref<?xbf16, 2>
      asccube.release_buf mte1, %c7_i64, 0 {cannir.buffer.generated_sync}
      %44 = ascscalar.bitcast_memref %alloc_22 byte_offset = %c0_i64 : memref<32768xbf16, 2> -> memref<?xbf16, 2>
      asccube.sync_intra_wait mte1, %c23_i64 {cannir.sync.banks = 2 : i32}
      asccube.get_buf mte1, %c7_i64, 0 {cannir.buffer.generated_sync}
      asccube.copy_l12l0b %42[%c1024], %44[%c0], %c0_i32, %c0_i32, %c4_i32, %c4_i32, %c4_i32, %c8_i32 : memref<?xbf16, 4>, memref<?xbf16, 2>
      asccube.release_buf mte1, %c7_i64, 0 {cannir.buffer.generated_sync}
      %45 = ascscalar.bitcast_memref %alloc_13 byte_offset = %c0_i64 : memref<8192xf32, 5> -> memref<?xf32, 5>
      asccube.get_buf m, %c3_i64, 0 {cannir.buffer.generated_sync}
      asccube.get_buf m, %c7_i64, 0 {cannir.buffer.generated_sync}
      asccube.mmad %45[%c0], %41, %42, %c64_i32, %c64_i32, %c128_i32, %c2_i32 {c_matrix_init_val = true, mmad_direction = 0 : i32} : memref<?xf32, 5>, memref<?xbf16, 3>, memref<?xbf16, 4>
      asccube.release_buf m, %c7_i64, 0 {cannir.buffer.generated_sync}
      asccube.release_buf m, %c3_i64, 0 {cannir.buffer.generated_sync}
      %46 = ascscalar.bitcast_memref %alloc_16 byte_offset = %c0_i64 : memref<8192xbf16, 3> -> memref<?xbf16, 3>
      asccube.get_buf mte1, %c4_i64, 0 {cannir.buffer.generated_sync}
      asccube.get_buf mte1, %c0_i64, 0 {cannir.buffer.generated_sync}
      asccube.copy_l12l0a %46[%c0], %28[%c0], %c0_i32, %c4_i32, %c4_i32, %c4_i32, %c4_i32, %c4_i32 : memref<?xbf16, 3>, memref<?xbf16, 2>
      asccube.release_buf mte1, %c0_i64, 0 {cannir.buffer.generated_sync}
      asccube.release_buf mte1, %c4_i64, 0 {cannir.buffer.generated_sync}
      %47 = ascscalar.bitcast_memref %alloc_17 byte_offset = %c0_i64 : memref<8192xbf16, 4> -> memref<?xbf16, 4>
      asccube.get_buf mte1, %c8_i64, 0 {cannir.buffer.generated_sync}
      asccube.copy_l12l0b %47[%c0], %43[%c0], %c0_i32, %c4_i32, %c4_i32, %c4_i32, %c4_i32, %c8_i32 : memref<?xbf16, 4>, memref<?xbf16, 2>
      asccube.copy_l12l0b %47[%c1024], %44[%c0], %c0_i32, %c4_i32, %c4_i32, %c4_i32, %c4_i32, %c8_i32 : memref<?xbf16, 4>, memref<?xbf16, 2>
      asccube.release_buf mte1, %c8_i64, 0 {cannir.buffer.generated_sync}
      asccube.get_buf m, %c4_i64, 0 {cannir.buffer.generated_sync}
      asccube.get_buf m, %c8_i64, 0 {cannir.buffer.generated_sync}
      asccube.mmad %45[%c0], %46, %47, %c64_i32, %c64_i32, %c128_i32, %c2_i32 {c_matrix_init_val = false, mmad_direction = 0 : i32} : memref<?xf32, 5>, memref<?xbf16, 3>, memref<?xbf16, 4>
      asccube.release_buf m, %c8_i64, 0 {cannir.buffer.generated_sync}
      asccube.release_buf m, %c4_i64, 0 {cannir.buffer.generated_sync}
      %48 = ascscalar.bitcast_memref %alloc_18 byte_offset = %c0_i64 : memref<8192xbf16, 3> -> memref<?xbf16, 3>
      asccube.get_buf mte1, %c5_i64, 0 {cannir.buffer.generated_sync}
      asccube.get_buf mte1, %c0_i64, 0 {cannir.buffer.generated_sync}
      asccube.copy_l12l0a %48[%c0], %28[%c0], %c0_i32, %c8_i32, %c4_i32, %c4_i32, %c4_i32, %c4_i32 : memref<?xbf16, 3>, memref<?xbf16, 2>
      asccube.release_buf mte1, %c0_i64, 0 {cannir.buffer.generated_sync}
      asccube.release_buf mte1, %c5_i64, 0 {cannir.buffer.generated_sync}
      %49 = ascscalar.bitcast_memref %alloc_19 byte_offset = %c0_i64 : memref<8192xbf16, 4> -> memref<?xbf16, 4>
      asccube.get_buf mte1, %c9_i64, 0 {cannir.buffer.generated_sync}
      asccube.copy_l12l0b %49[%c0], %43[%c0], %c0_i32, %c8_i32, %c4_i32, %c4_i32, %c4_i32, %c8_i32 : memref<?xbf16, 4>, memref<?xbf16, 2>
      asccube.copy_l12l0b %49[%c1024], %44[%c0], %c0_i32, %c8_i32, %c4_i32, %c4_i32, %c4_i32, %c8_i32 : memref<?xbf16, 4>, memref<?xbf16, 2>
      asccube.release_buf mte1, %c9_i64, 0 {cannir.buffer.generated_sync}
      asccube.get_buf m, %c5_i64, 0 {cannir.buffer.generated_sync}
      asccube.get_buf m, %c9_i64, 0 {cannir.buffer.generated_sync}
      asccube.mmad %45[%c0], %48, %49, %c64_i32, %c64_i32, %c128_i32, %c2_i32 {c_matrix_init_val = false, mmad_direction = 0 : i32} : memref<?xf32, 5>, memref<?xbf16, 3>, memref<?xbf16, 4>
      asccube.release_buf m, %c9_i64, 0 {cannir.buffer.generated_sync}
      asccube.release_buf m, %c5_i64, 0 {cannir.buffer.generated_sync}
      %50 = ascscalar.bitcast_memref %alloc_20 byte_offset = %c0_i64 : memref<8192xbf16, 3> -> memref<?xbf16, 3>
      asccube.get_buf mte1, %c6_i64, 0 {cannir.buffer.generated_sync}
      asccube.get_buf mte1, %c0_i64, 0 {cannir.buffer.generated_sync}
      asccube.copy_l12l0a %50[%c0], %28[%c0], %c0_i32, %c12_i32, %c4_i32, %c4_i32, %c4_i32, %c4_i32 : memref<?xbf16, 3>, memref<?xbf16, 2>
      asccube.release_buf mte1, %c0_i64, 0 {cannir.buffer.generated_sync}
      asccube.release_buf mte1, %c6_i64, 0 {cannir.buffer.generated_sync}
      %51 = ascscalar.bitcast_memref %alloc_21 byte_offset = %c0_i64 : memref<8192xbf16, 4> -> memref<?xbf16, 4>
      asccube.get_buf mte1, %c10_i64, 0 {cannir.buffer.generated_sync}
      asccube.copy_l12l0b %51[%c0], %43[%c0], %c0_i32, %c12_i32, %c4_i32, %c4_i32, %c4_i32, %c8_i32 : memref<?xbf16, 4>, memref<?xbf16, 2>
      asccube.copy_l12l0b %51[%c1024], %44[%c0], %c0_i32, %c12_i32, %c4_i32, %c4_i32, %c4_i32, %c8_i32 : memref<?xbf16, 4>, memref<?xbf16, 2>
      asccube.release_buf mte1, %c10_i64, 0 {cannir.buffer.generated_sync}
      asccube.get_buf m, %c6_i64, 0 {cannir.buffer.generated_sync}
      asccube.get_buf m, %c10_i64, 0 {cannir.buffer.generated_sync}
      asccube.mmad %45[%c0], %50, %51, %c64_i32, %c64_i32, %c128_i32, %c2_i32 {c_matrix_init_val = false, mmad_direction = 0 : i32} : memref<?xf32, 5>, memref<?xbf16, 3>, memref<?xbf16, 4>
      asccube.release_buf m, %c10_i64, 0 {cannir.buffer.generated_sync}
      asccube.release_buf m, %c6_i64, 0 {cannir.buffer.generated_sync}
      asccube.get_buf mte1, %c3_i64, 0 {cannir.buffer.generated_sync}
      asccube.get_buf mte1, %c0_i64, 0 {cannir.buffer.generated_sync}
      asccube.copy_l12l0a %41[%c0], %28[%c0], %c0_i32, %c16_i32, %c4_i32, %c4_i32, %c4_i32, %c4_i32 : memref<?xbf16, 3>, memref<?xbf16, 2>
      asccube.release_buf mte1, %c0_i64, 0 {cannir.buffer.generated_sync}
      asccube.release_buf mte1, %c3_i64, 0 {cannir.buffer.generated_sync}
      asccube.get_buf mte1, %c7_i64, 0 {cannir.buffer.generated_sync}
      asccube.copy_l12l0b %42[%c0], %43[%c0], %c0_i32, %c16_i32, %c4_i32, %c4_i32, %c4_i32, %c8_i32 : memref<?xbf16, 4>, memref<?xbf16, 2>
      asccube.copy_l12l0b %42[%c1024], %44[%c0], %c0_i32, %c16_i32, %c4_i32, %c4_i32, %c4_i32, %c8_i32 : memref<?xbf16, 4>, memref<?xbf16, 2>
      asccube.release_buf mte1, %c7_i64, 0 {cannir.buffer.generated_sync}
      asccube.get_buf m, %c3_i64, 0 {cannir.buffer.generated_sync}
      asccube.get_buf m, %c7_i64, 0 {cannir.buffer.generated_sync}
      asccube.mmad %45[%c0], %41, %42, %c64_i32, %c64_i32, %c128_i32, %c2_i32 {c_matrix_init_val = false, mmad_direction = 0 : i32} : memref<?xf32, 5>, memref<?xbf16, 3>, memref<?xbf16, 4>
      asccube.release_buf m, %c7_i64, 0 {cannir.buffer.generated_sync}
      asccube.release_buf m, %c3_i64, 0 {cannir.buffer.generated_sync}
      asccube.get_buf mte1, %c4_i64, 0 {cannir.buffer.generated_sync}
      asccube.get_buf mte1, %c0_i64, 0 {cannir.buffer.generated_sync}
      asccube.copy_l12l0a %46[%c0], %28[%c0], %c0_i32, %c20_i32, %c4_i32, %c4_i32, %c4_i32, %c4_i32 : memref<?xbf16, 3>, memref<?xbf16, 2>
      asccube.release_buf mte1, %c0_i64, 0 {cannir.buffer.generated_sync}
      asccube.release_buf mte1, %c4_i64, 0 {cannir.buffer.generated_sync}
      asccube.get_buf mte1, %c8_i64, 0 {cannir.buffer.generated_sync}
      asccube.copy_l12l0b %47[%c0], %43[%c0], %c0_i32, %c20_i32, %c4_i32, %c4_i32, %c4_i32, %c8_i32 : memref<?xbf16, 4>, memref<?xbf16, 2>
      asccube.copy_l12l0b %47[%c1024], %44[%c0], %c0_i32, %c20_i32, %c4_i32, %c4_i32, %c4_i32, %c8_i32 : memref<?xbf16, 4>, memref<?xbf16, 2>
      asccube.release_buf mte1, %c8_i64, 0 {cannir.buffer.generated_sync}
      asccube.get_buf m, %c4_i64, 0 {cannir.buffer.generated_sync}
      asccube.get_buf m, %c8_i64, 0 {cannir.buffer.generated_sync}
      asccube.mmad %45[%c0], %46, %47, %c64_i32, %c64_i32, %c128_i32, %c2_i32 {c_matrix_init_val = false, mmad_direction = 0 : i32} : memref<?xf32, 5>, memref<?xbf16, 3>, memref<?xbf16, 4>
      asccube.release_buf m, %c8_i64, 0 {cannir.buffer.generated_sync}
      asccube.release_buf m, %c4_i64, 0 {cannir.buffer.generated_sync}
      asccube.get_buf mte1, %c5_i64, 0 {cannir.buffer.generated_sync}
      asccube.get_buf mte1, %c2_i64, 0 {cannir.buffer.generated_sync}
      asccube.copy_l12l0a %48[%c0], %36[%c0], %c0_i32, %c0_i32, %c4_i32, %c4_i32, %c4_i32, %c4_i32 : memref<?xbf16, 3>, memref<?xbf16, 2>
      asccube.release_buf mte1, %c2_i64, 0 {cannir.buffer.generated_sync}
      asccube.release_buf mte1, %c5_i64, 0 {cannir.buffer.generated_sync}
      asccube.get_buf mte1, %c9_i64, 0 {cannir.buffer.generated_sync}
      asccube.copy_l12l0b %49[%c0], %43[%c0], %c0_i32, %c24_i32, %c4_i32, %c4_i32, %c4_i32, %c8_i32 : memref<?xbf16, 4>, memref<?xbf16, 2>
      asccube.copy_l12l0b %49[%c1024], %44[%c0], %c0_i32, %c24_i32, %c4_i32, %c4_i32, %c4_i32, %c8_i32 : memref<?xbf16, 4>, memref<?xbf16, 2>
      asccube.release_buf mte1, %c9_i64, 0 {cannir.buffer.generated_sync}
      asccube.get_buf m, %c5_i64, 0 {cannir.buffer.generated_sync}
      asccube.get_buf m, %c9_i64, 0 {cannir.buffer.generated_sync}
      asccube.mmad %45[%c0], %48, %49, %c64_i32, %c64_i32, %c128_i32, %c2_i32 {c_matrix_init_val = false, mmad_direction = 0 : i32} : memref<?xf32, 5>, memref<?xbf16, 3>, memref<?xbf16, 4>
      asccube.release_buf m, %c9_i64, 0 {cannir.buffer.generated_sync}
      asccube.release_buf m, %c5_i64, 0 {cannir.buffer.generated_sync}
      asccube.get_buf mte1, %c6_i64, 0 {cannir.buffer.generated_sync}
      asccube.get_buf mte1, %c2_i64, 0 {cannir.buffer.generated_sync}
      asccube.copy_l12l0a %50[%c0], %36[%c0], %c0_i32, %c4_i32, %c4_i32, %c4_i32, %c4_i32, %c4_i32 : memref<?xbf16, 3>, memref<?xbf16, 2>
      asccube.release_buf mte1, %c2_i64, 0 {cannir.buffer.generated_sync}
      asccube.release_buf mte1, %c6_i64, 0 {cannir.buffer.generated_sync}
      asccube.get_buf mte1, %c10_i64, 0 {cannir.buffer.generated_sync}
      asccube.copy_l12l0b %51[%c0], %43[%c0], %c0_i32, %c28_i32, %c4_i32, %c4_i32, %c4_i32, %c8_i32 : memref<?xbf16, 4>, memref<?xbf16, 2>
      asccube.copy_l12l0b %51[%c1024], %44[%c0], %c0_i32, %c28_i32, %c4_i32, %c4_i32, %c4_i32, %c8_i32 : memref<?xbf16, 4>, memref<?xbf16, 2>
      asccube.release_buf mte1, %c10_i64, 0 {cannir.buffer.generated_sync}
      asccube.get_buf m, %c6_i64, 0 {cannir.buffer.generated_sync}
      asccube.get_buf m, %c10_i64, 0 {cannir.buffer.generated_sync}
      asccube.mmad %45[%c0], %50, %51, %c64_i32, %c64_i32, %c128_i32, %c3_i32 {c_matrix_init_val = false, mmad_direction = 0 : i32} : memref<?xf32, 5>, memref<?xbf16, 3>, memref<?xbf16, 4>
      asccube.release_buf m, %c10_i64, 0 {cannir.buffer.generated_sync}
      asccube.release_buf m, %c6_i64, 0 {cannir.buffer.generated_sync}
      %52 = ascscalar.bitcast_memref %alloc_10 byte_offset = %c0_i64 : memref<4096xf32, 1> -> memref<?xf32, 1>
      asccube.sync_intra_wait fixpipe, %c0_i64 {cannir.sync.banks = 3 : i32}
      asccube.sync_intra_wait fixpipe, %c16_i64 {cannir.sync.banks = 3 : i32}
      asccube.copy_l0c2ub %52[%c0], %45[%c0], %c128_i32, %c64_i32, %c128_i32, %c64_i32, %c3_i32 sub_block_id %c0_i32 {dual_dst_ctl = 1 : i32, nz2nd_en = true, relu_en = false} : memref<?xf32, 1>, memref<?xf32, 5>
      asccube.sync_intra_arrive fixpipe, %c0_i64 {cannir.sync.banks = 3 : i32}
      asccube.sync_intra_arrive fixpipe, %c16_i64 {cannir.sync.banks = 3 : i32}
      %53 = ascscalar.memref_address %alloc_25 : memref<32768xbf16, 2>
      %54 = ascscalar.memref_address %alloc_22 : memref<32768xbf16, 2>
      %55 = ascscalar.memref_address %alloc_39 : memref<24576xbf16, 2>
      %56:45 = scf.while (%arg24 = %53, %arg25 = %c0_i64, %arg26 = %c32768_i64, %arg27 = %c1_i64, %arg28 = %c32768_i64, %arg29 = %c1_i64, %arg30 = %53, %arg31 = %c0_i64, %arg32 = %53, %arg33 = %c0_i64, %arg34 = %54, %arg35 = %c0_i64, %arg36 = %c32768_i64, %arg37 = %c1_i64, %arg38 = %c32768_i64, %arg39 = %c1_i64, %arg40 = %54, %arg41 = %c0_i64, %arg42 = %54, %arg43 = %c0_i64, %arg44 = %20#9, %arg45 = %20#11, %arg46 = %55, %arg47 = %c0_i64, %arg48 = %c64_i64, %arg49 = %c384_i64, %arg50 = %c24_i64, %arg51 = %c4_i64, %arg52 = %c1024_i64, %arg53 = %c256_i64, %arg54 = %55, %arg55 = %c0_i64, %arg56 = %55, %arg57 = %c0_i64, %arg58 = %c0_i64, %arg59 = %c1_i64, %arg60 = %c1_i64, %arg61 = %c1_i64, %arg62 = %c0_i64, %arg63 = %c0_i64, %arg64 = %c0_i64, %arg65 = %c17_i64, %arg66 = %c21_i64, %arg67 = %c14_i64, %arg68 = %c14_i64, %arg69 = %c14_i64) : (i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> (i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) {
        %57 = arith.cmpi ne, %arg59, %c0_i64 : i64
        scf.condition(%57) %arg24, %arg25, %arg26, %arg27, %arg28, %arg29, %arg30, %arg31, %arg32, %arg33, %arg34, %arg35, %arg36, %arg37, %arg38, %arg39, %arg40, %arg41, %arg42, %arg43, %arg44, %arg45, %arg46, %arg47, %arg48, %arg49, %arg50, %arg51, %arg52, %arg53, %arg54, %arg55, %arg56, %arg57, %arg58, %arg60, %arg61, %arg62, %arg63, %arg64, %arg65, %arg66, %arg67, %arg68, %arg69 : i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64
      } do {
      ^bb0(%arg24: i64, %arg25: i64, %arg26: i64, %arg27: i64, %arg28: i64, %arg29: i64, %arg30: i64, %arg31: i64, %arg32: i64, %arg33: i64, %arg34: i64, %arg35: i64, %arg36: i64, %arg37: i64, %arg38: i64, %arg39: i64, %arg40: i64, %arg41: i64, %arg42: i64, %arg43: i64, %arg44: i64, %arg45: i64, %arg46: i64, %arg47: i64, %arg48: i64, %arg49: i64, %arg50: i64, %arg51: i64, %arg52: i64, %arg53: i64, %arg54: i64, %arg55: i64, %arg56: i64, %arg57: i64, %arg58: i64, %arg59: i64, %arg60: i64, %arg61: i64, %arg62: i64, %arg63: i64, %arg64: i64, %arg65: i64, %arg66: i64, %arg67: i64, %arg68: i64):
        %57 = arith.cmpi eq, %arg66, %c15_i64 : i64
        %58 = arith.extui %57 : i1 to i64
        %59 = arith.addi %arg45, %c1_i64 : i64
        %60 = arith.cmpi slt, %59, %17 : i64
        %61 = arith.extui %60 : i1 to i64
        %62:12 = scf.while (%arg69 = %c0_i64, %arg70 = %c0_i64, %arg71 = %c0_i64, %arg72 = %c0_i64, %arg73 = %c0_i64, %arg74 = %c0_i64, %arg75 = %c0_i64, %arg76 = %c0_i64, %arg77 = %c0_i64, %arg78 = %c0_i64, %arg79 = %c0_i64, %arg80 = %59, %arg81 = %61) : (i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> (i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) {
          %563 = arith.cmpi ne, %arg81, %c0_i64 : i64
          scf.condition(%563) %arg69, %arg70, %arg71, %arg72, %arg73, %arg74, %arg75, %arg76, %arg77, %arg78, %arg79, %arg80 : i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64
        } do {
        ^bb0(%arg69: i64, %arg70: i64, %arg71: i64, %arg72: i64, %arg73: i64, %arg74: i64, %arg75: i64, %arg76: i64, %arg77: i64, %arg78: i64, %arg79: i64, %arg80: i64):
          %563 = memref.load %alloc_5[%14] : memref<1xi32>
          %564 = arith.cmpi ne, %563, %c0_i32 : i32
          %565:5 = scf.if %564 -> (i64, i64, i64, i64, i64) {
            %587 = arith.index_cast %10 : i64 to index
            %588 = memref.load %alloc_5[%587] : memref<1xi32>
            %589 = arith.extsi %588 : i32 to i64
            %590 = arith.index_cast %12 : i64 to index
            %591 = memref.load %alloc_5[%590] : memref<1xi32>
            %592 = arith.extsi %591 : i32 to i64
            %593 = arith.index_cast %11 : i64 to index
            %594 = memref.load %alloc_5[%593] : memref<1xi32>
            %595 = arith.extsi %594 : i32 to i64
            %596 = arith.index_cast %13 : i64 to index
            %597 = memref.load %alloc_5[%596] : memref<1xi32>
            %598 = arith.extsi %597 : i32 to i64
            %599 = arith.cmpi sgt, %9, %c0_i64 : i64
            %600 = arith.extui %599 : i1 to i64
            %601 = arith.addi %598, %600 : i64
            %602 = cannir.datastruct_get %arg22["batch_mode"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
            %603 = arith.cmpi eq, %602, %c2_i32 : i32
            %604:5 = scf.if %603 -> (i64, i64, i64, i64, i64) {
              %605 = arith.index_cast %588 : i32 to index
              %606 = memref.load %alloc_7[%605] : memref<1xi32>
              %607 = arith.extsi %606 : i32 to i64
              %608 = arith.addi %607, %592 : i64
              %609 = cannir.datastruct_get %arg22["batch_count"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
              %610 = arith.extsi %609 : i32 to i64
              %611 = arith.index_cast %609 : i32 to index
              %612 = memref.load %alloc_7[%611] : memref<1xi32>
              %613 = arith.extsi %612 : i32 to i64
              %614 = arith.cmpi slt, %595, %610 : i64
              %615 = scf.if %614 -> (i64) {
                %619 = arith.index_cast %594 : i32 to index
                %620 = memref.load %alloc_7[%619] : memref<1xi32>
                %621 = arith.extsi %620 : i32 to i64
                %622 = arith.addi %595, %c1_i64 : i64
                %623 = arith.index_cast %622 : i64 to index
                %624 = memref.load %alloc_7[%623] : memref<1xi32>
                %625 = arith.extsi %624 : i32 to i64
                %626 = arith.subi %625, %621 : i64
                %627 = arith.cmpi slt, %601, %626 : i64
                %628 = arith.select %627, %601, %626 : i64
                %629 = arith.addi %621, %628 : i64
                scf.yield %629 : i64
              } else {
                scf.yield %613 : i64
              }
              %616 = arith.addi %608, %arg80 : i64
              %617 = arith.cmpi sge, %arg80, %c0_i64 : i64
              %618:5 = scf.if %617 -> (i64, i64, i64, i64, i64) {
                %619 = arith.cmpi slt, %616, %615 : i64
                %620 = arith.select %619, %616, %c0_i64 : i64
                %621 = arith.extui %619 : i1 to i64
                %622:3 = scf.if %619 -> (i64, i64, i64) {
                  %623 = cannir.datastruct_get %arg22["batch_count"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
                  %624 = arith.extsi %623 : i32 to i64
                  %625 = arith.cmpi slt, %595, %624 : i64
                  %626 = arith.subi %624, %c1_i64 : i64
                  %627 = arith.select %625, %595, %626 : i64
                  %628 = memref.load %alloc_7[%605] : memref<1xi32>
                  %629 = arith.extsi %628 : i32 to i64
                  %630 = arith.addi %589, %c1_i64 : i64
                  %631 = arith.index_cast %630 : i64 to index
                  %632 = memref.load %alloc_7[%631] : memref<1xi32>
                  %633 = arith.extsi %632 : i32 to i64
                  %634 = arith.cmpi slt, %589, %627 : i64
                  %635:3 = scf.if %634 -> (i64, i64, i64) {
                    %641 = memref.load %alloc_7[%c0] : memref<1xi32>
                    %642 = arith.extsi %641 : i32 to i64
                    %643 = cannir.datastruct_get %arg22["batch_count"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
                    %644 = arith.extsi %643 : i32 to i64
                    %645 = arith.index_cast %643 : i32 to index
                    %646 = memref.load %alloc_7[%645] : memref<1xi32>
                    %647 = arith.extsi %646 : i32 to i64
                    %648 = arith.subi %647, %642 : i64
                    %649 = arith.remsi %648, %644 : i64
                    %650 = arith.cmpi ne, %649, %c0_i64 : i64
                    %651 = arith.cmpi slt, %649, %c0_i64 : i64
                    %652 = arith.cmpi slt, %644, %c0_i64 : i64
                    %653 = arith.xori %651, %652 : i1
                    %654 = arith.andi %650, %653 : i1
                    %655 = arith.divsi %648, %644 : i64
                    %656 = arith.extui %654 : i1 to i64
                    %657 = arith.subi %655, %656 : i64
                    %658 = arith.cmpi sgt, %657, %c0_i64 : i64
                    %659 = arith.select %658, %657, %c1_i64 : i64
                    %660 = arith.subi %616, %642 : i64
                    %661 = arith.remsi %660, %659 : i64
                    %662 = arith.cmpi ne, %661, %c0_i64 : i64
                    %663 = arith.cmpi slt, %661, %c0_i64 : i64
                    %664 = arith.andi %662, %663 : i1
                    %665 = arith.divsi %660, %659 : i64
                    %666 = arith.extui %664 : i1 to i64
                    %667 = arith.subi %665, %666 : i64
                    %668 = arith.cmpi slt, %667, %589 : i64
                    %669 = arith.select %668, %589, %667 : i64
                    %670 = arith.cmpi sgt, %669, %627 : i64
                    %671 = arith.select %670, %627, %669 : i64
                    %672 = arith.index_cast %671 : i64 to index
                    %673 = memref.load %alloc_7[%672] : memref<1xi32>
                    %674 = arith.extsi %673 : i32 to i64
                    %675 = arith.addi %671, %c1_i64 : i64
                    %676 = arith.index_cast %675 : i64 to index
                    %677 = memref.load %alloc_7[%676] : memref<1xi32>
                    %678 = arith.extsi %677 : i32 to i64
                    %679 = arith.cmpi slt, %616, %674 : i64
                    %680 = arith.cmpi sge, %616, %678 : i64
                    %681 = arith.select %679, %true, %680 : i1
                    %682:3 = scf.if %681 -> (i64, i64, i64) {
                      %683:2 = scf.while (%arg81 = %589, %arg82 = %627) : (i64, i64) -> (i64, i64) {
                        %691 = arith.cmpi slt, %arg81, %arg82 : i64
                        scf.condition(%691) %arg81, %arg82 : i64, i64
                      } do {
                      ^bb0(%arg81: i64, %arg82: i64):
                        %691 = arith.addi %arg81, %arg82 : i64
                        %692 = arith.remsi %691, %c2_i64 : i64
                        %693 = arith.cmpi ne, %692, %c0_i64 : i64
                        %694 = arith.cmpi slt, %692, %c0_i64 : i64
                        %695 = arith.andi %693, %694 : i1
                        %696 = arith.divsi %691, %c2_i64 : i64
                        %697 = arith.extui %695 : i1 to i64
                        %698 = arith.subi %696, %697 : i64
                        %699 = arith.addi %698, %c1_i64 : i64
                        %700 = arith.index_cast %699 : i64 to index
                        %701 = memref.load %alloc_7[%700] : memref<1xi32>
                        %702 = arith.extsi %701 : i32 to i64
                        %703 = arith.cmpi sle, %702, %616 : i64
                        %704 = arith.select %703, %699, %arg81 : i64
                        %705 = arith.select %703, %arg82, %698 : i64
                        scf.yield %704, %705 : i64, i64
                      }
                      %684 = arith.index_cast %683#0 : i64 to index
                      %685 = memref.load %alloc_7[%684] : memref<1xi32>
                      %686 = arith.extsi %685 : i32 to i64
                      %687 = arith.addi %683#0, %c1_i64 : i64
                      %688 = arith.index_cast %687 : i64 to index
                      %689 = memref.load %alloc_7[%688] : memref<1xi32>
                      %690 = arith.extsi %689 : i32 to i64
                      scf.yield %683#0, %686, %690 : i64, i64, i64
                    } else {
                      scf.yield %671, %674, %678 : i64, i64, i64
                    }
                    scf.yield %682#0, %682#1, %682#2 : i64, i64, i64
                  } else {
                    scf.yield %589, %629, %633 : i64, i64, i64
                  }
                  %636 = arith.subi %616, %635#1 : i64
                  %637 = arith.subi %635#2, %635#1 : i64
                  %638 = cannir.datastruct_get %arg22["has_seqused_q"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
                  %639 = arith.cmpi ne, %638, %c0_i32 : i32
                  %640 = scf.if %639 -> (i64) {
                    %641 = arith.index_cast %635#0 : i64 to index
                    %642 = memref.load %alloc_6[%641] : memref<1xi32>
                    %643 = arith.extsi %642 : i32 to i64
                    scf.yield %643 : i64
                  } else {
                    scf.yield %637 : i64
                  }
                  scf.yield %635#0, %636, %640 : i64, i64, i64
                } else {
                  scf.yield %c0_i64, %c0_i64, %c0_i64 : i64, i64, i64
                }
                scf.yield %620, %622#0, %622#1, %622#2, %621 : i64, i64, i64, i64, i64
              } else {
                scf.yield %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64 : i64, i64, i64, i64, i64
              }
              scf.yield %618#0, %618#1, %618#2, %618#3, %618#4 : i64, i64, i64, i64, i64
            } else {
              %605 = cannir.datastruct_get %arg22["query_rows_per_batch"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
              %606 = arith.extsi %605 : i32 to i64
              %607 = arith.muli %589, %606 : i64
              %608 = arith.addi %607, %592 : i64
              %609 = arith.muli %595, %606 : i64
              %610 = arith.cmpi slt, %601, %606 : i64
              %611 = arith.select %610, %601, %606 : i64
              %612 = arith.addi %609, %611 : i64
              %613 = arith.addi %608, %arg80 : i64
              %614 = arith.cmpi sge, %arg80, %c0_i64 : i64
              %615:5 = scf.if %614 -> (i64, i64, i64, i64, i64) {
                %616 = arith.cmpi slt, %613, %612 : i64
                %617 = arith.select %616, %613, %c0_i64 : i64
                %618 = arith.extui %616 : i1 to i64
                %619:3 = scf.if %616 -> (i64, i64, i64) {
                  %620 = arith.remsi %613, %606 : i64
                  %621 = arith.cmpi ne, %620, %c0_i64 : i64
                  %622 = arith.cmpi slt, %620, %c0_i64 : i64
                  %623 = arith.cmpi slt, %606, %c0_i64 : i64
                  %624 = arith.xori %622, %623 : i1
                  %625 = arith.andi %621, %624 : i1
                  %626 = arith.divsi %613, %606 : i64
                  %627 = arith.extui %625 : i1 to i64
                  %628 = arith.subi %626, %627 : i64
                  %629 = arith.muli %628, %606 : i64
                  %630 = arith.subi %613, %629 : i64
                  %631 = cannir.datastruct_get %arg22["has_seqused_q"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
                  %632 = arith.cmpi ne, %631, %c0_i32 : i32
                  %633 = scf.if %632 -> (i64) {
                    %634 = arith.index_cast %628 : i64 to index
                    %635 = memref.load %alloc_6[%634] : memref<1xi32>
                    %636 = arith.extsi %635 : i32 to i64
                    scf.yield %636 : i64
                  } else {
                    scf.yield %606 : i64
                  }
                  scf.yield %628, %630, %633 : i64, i64, i64
                } else {
                  scf.yield %c0_i64, %c0_i64, %c0_i64 : i64, i64, i64
                }
                scf.yield %617, %619#0, %619#1, %619#2, %618 : i64, i64, i64, i64, i64
              } else {
                scf.yield %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64 : i64, i64, i64, i64, i64
              }
              scf.yield %615#0, %615#1, %615#2, %615#3, %615#4 : i64, i64, i64, i64, i64
            }
            scf.yield %604#0, %604#1, %604#2, %604#3, %604#4 : i64, i64, i64, i64, i64
          } else {
            scf.yield %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64 : i64, i64, i64, i64, i64
          }
          %566 = arith.cmpi ne, %565#4, %c0_i64 : i64
          %567 = arith.cmpi slt, %565#2, %565#3 : i64
          %568 = arith.select %566, %567, %false : i1
          %569:9 = scf.if %568 -> (i64, i64, i64, i64, i64, i64, i64, i64, i64) {
            %587 = cannir.datastruct_get %arg22["ori_kv_capacity"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
            %588 = arith.extsi %587 : i32 to i64
            %589 = arith.muli %565#1, %588 : i64
            %590 = cannir.datastruct_get %arg22["cmp_kv_capacity"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
            %591 = arith.extsi %590 : i32 to i64
            %592 = arith.muli %565#1, %591 : i64
            %593 = cannir.datastruct_get %arg22["kv_mode"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
            %594 = arith.cmpi eq, %593, %c1_i32 : i32
            %595:4 = scf.if %594 -> (i64, i64, i64, i64) {
              scf.yield %c0_i64, %c0_i64, %588, %591 : i64, i64, i64, i64
            } else {
              %642 = cannir.datastruct_get %arg22["batch_mode"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
              %643 = arith.cmpi eq, %642, %c2_i32 : i32
              %644:4 = scf.if %643 -> (i64, i64, i64, i64) {
                %645 = arith.index_cast %565#1 : i64 to index
                %646 = memref.load %alloc_8[%645] : memref<1xi32>
                %647 = arith.extsi %646 : i32 to i64
                %648 = arith.addi %565#1, %c1_i64 : i64
                %649 = arith.index_cast %648 : i64 to index
                %650 = memref.load %alloc_8[%649] : memref<1xi32>
                %651 = arith.extsi %650 : i32 to i64
                %652 = arith.subi %651, %647 : i64
                %653 = cannir.datastruct_get %arg22["has_cmp"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
                %654 = arith.cmpi ne, %653, %c0_i32 : i32
                %655:2 = scf.if %654 -> (i64, i64) {
                  %656 = memref.load %alloc_9[%645] : memref<1xi32>
                  %657 = arith.extsi %656 : i32 to i64
                  %658 = memref.load %alloc_9[%649] : memref<1xi32>
                  %659 = arith.extsi %658 : i32 to i64
                  %660 = arith.subi %659, %657 : i64
                  scf.yield %657, %660 : i64, i64
                } else {
                  scf.yield %592, %591 : i64, i64
                }
                scf.yield %647, %652, %655#0, %655#1 : i64, i64, i64, i64
              } else {
                scf.yield %589, %588, %592, %591 : i64, i64, i64, i64
              }
              scf.yield %644#0, %644#2, %644#1, %644#3 : i64, i64, i64, i64
            }
            %596 = cannir.datastruct_get %arg22["has_cmp"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
            %597 = arith.cmpi eq, %596, %c0_i32 : i32
            %598 = arith.select %597, %c0_i64, %595#1 : i64
            %599 = arith.select %597, %c0_i64, %595#3 : i64
            %600 = cannir.datastruct_get %arg22["has_seqused_ori_kv"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
            %601 = arith.cmpi ne, %600, %c0_i32 : i32
            %602 = scf.if %601 -> (i64) {
              %642 = arith.index_cast %565#1 : i64 to index
              %643 = memref.load %alloc_2[%642] : memref<1xi32>
              %644 = arith.extsi %643 : i32 to i64
              %645 = arith.cmpi sgt, %644, %c0_i64 : i64
              %646 = arith.cmpi slt, %644, %595#2 : i64
              %647 = arith.select %646, %644, %595#2 : i64
              %648 = arith.select %645, %647, %c0_i64 : i64
              scf.yield %648 : i64
            } else {
              scf.yield %595#2 : i64
            }
            %603 = arith.cmpi ne, %596, %c0_i32 : i32
            %604 = scf.if %603 -> (i1) {
              %642 = cannir.datastruct_get %arg22["has_seqused_cmp_kv"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
              %643 = arith.cmpi ne, %642, %c0_i32 : i32
              scf.yield %643 : i1
            } else {
              scf.yield %false : i1
            }
            %605 = scf.if %604 -> (i64) {
              %642 = arith.index_cast %565#1 : i64 to index
              %643 = memref.load %alloc_3[%642] : memref<1xi32>
              %644 = arith.extsi %643 : i32 to i64
              %645 = arith.cmpi sgt, %644, %c0_i64 : i64
              %646 = arith.cmpi slt, %644, %599 : i64
              %647 = arith.select %646, %644, %599 : i64
              %648 = arith.select %645, %647, %c0_i64 : i64
              scf.yield %648 : i64
            } else {
              scf.yield %599 : i64
            }
            %606 = cannir.datastruct_get %arg22["ori_mask_mode"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
            %607 = arith.cmpi eq, %606, %c3_i32 : i32
            %608 = arith.subi %602, %565#3 : i64
            %609 = arith.addi %608, %565#2 : i64
            %610 = arith.addi %609, %c1_i64 : i64
            %611 = arith.cmpi sgt, %610, %c0_i64 : i64
            %612 = arith.cmpi slt, %610, %602 : i64
            %613 = arith.select %612, %610, %602 : i64
            %614 = arith.select %611, %613, %c0_i64 : i64
            %615 = arith.select %607, %614, %602 : i64
            %616 = arith.cmpi eq, %606, %c4_i32 : i32
            %617:2 = scf.if %616 -> (i64, i64) {
              %642 = cannir.datastruct_get %arg22["ori_win_left"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
              %643 = arith.cmpi sge, %642, %c0_i32 : i32
              %644 = scf.if %643 -> (i64) {
                %648 = cannir.datastruct_get %arg22["ori_win_left"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
                %649 = arith.extsi %648 : i32 to i64
                %650 = arith.subi %609, %649 : i64
                %651 = arith.cmpi sgt, %650, %c0_i64 : i64
                %652 = arith.cmpi slt, %650, %602 : i64
                %653 = arith.select %652, %650, %602 : i64
                %654 = arith.select %651, %653, %c0_i64 : i64
                scf.yield %654 : i64
              } else {
                scf.yield %c0_i64 : i64
              }
              %645 = cannir.datastruct_get %arg22["ori_win_right"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
              %646 = arith.cmpi sge, %645, %c0_i32 : i32
              %647 = scf.if %646 -> (i64) {
                %648 = cannir.datastruct_get %arg22["ori_win_right"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
                %649 = arith.extsi %648 : i32 to i64
                %650 = arith.addi %609, %649 : i64
                %651 = arith.addi %650, %c1_i64 : i64
                %652 = arith.cmpi sgt, %651, %c0_i64 : i64
                %653 = arith.cmpi slt, %651, %602 : i64
                %654 = arith.select %653, %651, %602 : i64
                %655 = arith.select %652, %654, %c0_i64 : i64
                scf.yield %655 : i64
              } else {
                scf.yield %615 : i64
              }
              scf.yield %644, %647 : i64, i64
            } else {
              scf.yield %c0_i64, %615 : i64, i64
            }
            %618 = arith.cmpi sgt, %617#1, %617#0 : i64
            %619 = arith.subi %617#1, %617#0 : i64
            %620 = arith.select %618, %619, %c0_i64 : i64
            %621 = cannir.datastruct_get %arg22["is_ori_dense"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
            %622 = arith.cmpi eq, %621, %c0_i32 : i32
            %623 = scf.if %622 -> (i64) {
              %642 = cannir.datastruct_get %arg22["ori_sparse_index_width"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
              %643 = arith.extsi %642 : i32 to i64
              %644 = arith.cmpi slt, %620, %643 : i64
              %645 = arith.select %644, %620, %643 : i64
              %646 = cannir.datastruct_get %arg22["has_ori_topk_length"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
              %647 = arith.cmpi ne, %646, %c0_i32 : i32
              %648 = scf.if %647 -> (i64) {
                %649 = arith.index_cast %565#0 : i64 to index
                %650 = memref.load %alloc_0[%649] : memref<1xi32>
                %651 = arith.extsi %650 : i32 to i64
                %652 = arith.cmpi sgt, %651, %c0_i64 : i64
                %653 = arith.cmpi slt, %645, %651 : i64
                %654 = arith.select %653, %645, %651 : i64
                %655 = arith.select %652, %654, %c0_i64 : i64
                scf.yield %655 : i64
              } else {
                scf.yield %645 : i64
              }
              scf.yield %648 : i64
            } else {
              scf.yield %620 : i64
            }
            %624 = scf.if %603 -> (i64) {
              %642 = cannir.datastruct_get %arg22["cmp_mask_mode"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
              %643 = arith.cmpi eq, %642, %c3_i32 : i32
              %644 = scf.if %643 -> (i64) {
                %652 = cannir.datastruct_get %arg22["cmp_ratio"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
                %653 = arith.extsi %652 : i32 to i64
                %654 = arith.muli %605, %653 : i64
                %655 = arith.cmpi ne, %652, %c1_i32 : i32
                %656 = scf.if %655 -> (i64) {
                  %673 = arith.index_cast %565#1 : i64 to index
                  %674 = memref.load %alloc_4[%673] : memref<1xi32>
                  %675 = arith.extsi %674 : i32 to i64
                  %676 = arith.addi %654, %675 : i64
                  scf.yield %676 : i64
                } else {
                  scf.yield %654 : i64
                }
                %657 = arith.subi %656, %565#3 : i64
                %658 = arith.addi %657, %565#2 : i64
                %659 = arith.addi %658, %c1_i64 : i64
                %660 = arith.remsi %659, %653 : i64
                %661 = arith.cmpi ne, %660, %c0_i64 : i64
                %662 = arith.cmpi slt, %660, %c0_i64 : i64
                %663 = arith.cmpi slt, %653, %c0_i64 : i64
                %664 = arith.xori %662, %663 : i1
                %665 = arith.andi %661, %664 : i1
                %666 = arith.divsi %659, %653 : i64
                %667 = arith.extui %665 : i1 to i64
                %668 = arith.subi %666, %667 : i64
                %669 = arith.cmpi sgt, %668, %c0_i64 : i64
                %670 = arith.cmpi slt, %668, %605 : i64
                %671 = arith.select %670, %668, %605 : i64
                %672 = arith.select %669, %671, %c0_i64 : i64
                scf.yield %672 : i64
              } else {
                scf.yield %605 : i64
              }
              %645 = cannir.datastruct_get %arg22["cmp_sparse_index_width"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
              %646 = arith.extsi %645 : i32 to i64
              %647 = arith.cmpi slt, %644, %646 : i64
              %648 = arith.select %647, %644, %646 : i64
              %649 = cannir.datastruct_get %arg22["has_cmp_topk_length"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
              %650 = arith.cmpi ne, %649, %c0_i32 : i32
              %651 = scf.if %650 -> (i64) {
                %652 = arith.index_cast %565#0 : i64 to index
                %653 = memref.load %alloc_1[%652] : memref<1xi32>
                %654 = arith.extsi %653 : i32 to i64
                %655 = arith.cmpi sgt, %654, %c0_i64 : i64
                %656 = arith.cmpi slt, %648, %654 : i64
                %657 = arith.select %656, %648, %654 : i64
                %658 = arith.select %655, %657, %c0_i64 : i64
                scf.yield %658 : i64
              } else {
                scf.yield %648 : i64
              }
              scf.yield %651 : i64
            } else {
              scf.yield %c0_i64 : i64
            }
            %625 = arith.addi %623, %c127_i64 : i64
            %626 = arith.remsi %625, %c128_i64 : i64
            %627 = arith.cmpi ne, %626, %c0_i64 : i64
            %628 = arith.cmpi slt, %626, %c0_i64 : i64
            %629 = arith.andi %627, %628 : i1
            %630 = arith.divsi %625, %c128_i64 : i64
            %631 = arith.extui %629 : i1 to i64
            %632 = arith.subi %630, %631 : i64
            %633 = arith.addi %624, %c127_i64 : i64
            %634 = arith.remsi %633, %c128_i64 : i64
            %635 = arith.cmpi ne, %634, %c0_i64 : i64
            %636 = arith.cmpi slt, %634, %c0_i64 : i64
            %637 = arith.andi %635, %636 : i1
            %638 = arith.divsi %633, %c128_i64 : i64
            %639 = arith.extui %637 : i1 to i64
            %640 = arith.subi %638, %639 : i64
            %641 = arith.addi %632, %640 : i64
            scf.yield %595#0, %598, %595#2, %599, %617#0, %623, %624, %632, %641 : i64, i64, i64, i64, i64, i64, i64, i64, i64
          } else {
            scf.yield %arg70, %arg71, %arg72, %arg73, %arg74, %arg75, %arg76, %arg77, %arg78 : i64, i64, i64, i64, i64, i64, i64, i64, i64
          }
          %570 = arith.cmpi eq, %arg80, %c0_i64 : i64
          %571 = arith.select %570, %5, %c0_i64 : i64
          %572 = arith.addi %arg80, %c1_i64 : i64
          %573 = arith.cmpi eq, %572, %17 : i64
          %574 = arith.cmpi sgt, %9, %c0_i64 : i64
          %575 = arith.select %573, %574, %false : i1
          %576 = arith.minsi %569#8, %9 : i64
          %577 = arith.select %575, %576, %569#8 : i64
          %578 = arith.subi %577, %571 : i64
          %579 = arith.maxsi %578, %c0_i64 : i64
          %580 = arith.cmpi sgt, %579, %c0_i64 : i64
          %581 = arith.extui %580 : i1 to i64
          %582 = arith.cmpi eq, %581, %c0_i64 : i64
          %583 = arith.select %582, %572, %arg80 : i64
          %584 = arith.cmpi slt, %583, %17 : i64
          %585 = arith.extui %582 : i1 to i64
          %586 = arith.select %584, %585, %c0_i64 : i64
          scf.yield %565#0, %569#0, %569#1, %569#2, %569#3, %569#4, %569#5, %569#6, %569#7, %579, %581, %583, %586 : i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64
        }
        %63 = arith.cmpi ne, %62#10, %c0_i64 : i64
        %64 = arith.select %63, %c0_i64, %arg47 : i64
        %65 = arith.select %63, %c64_i64, %arg48 : i64
        %66 = arith.select %63, %c384_i64, %arg49 : i64
        %67 = arith.select %63, %c24_i64, %arg50 : i64
        %68 = arith.select %63, %c4_i64, %arg51 : i64
        %69 = arith.select %63, %c1024_i64, %arg52 : i64
        %70 = arith.select %63, %c256_i64, %arg53 : i64
        %71 = arith.select %63, %c0_i64, %arg55 : i64
        %72 = arith.select %63, %c0_i64, %arg57 : i64
        %73:7 = scf.if %63 -> (i64, i64, i64, i64, i64, i64, i64) {
          %563 = cannir.datastruct_get %arg22["num_heads_q"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
          %564 = arith.extsi %563 : i32 to i64
          %565 = arith.muli %62#0, %564 : i64
          %566 = arith.muli %565, %c512_i64 : i64
          %567 = arith.cmpi eq, %arg59, %c0_i64 : i64
          %568 = ascscalar.memref_address %alloc_38 : memref<24576xbf16, 2>
          %569 = arith.select %567, %55, %568 : i64
          %570 = arith.select %567, %c14_i64, %c15_i64 : i64
          %571 = arith.cmpi ne, %arg59, %c0_i64 : i64
          %572 = arith.extui %571 : i1 to i64
          %573 = arith.addi %arg59, %c1_i64 overflow<nsw> : i64
          %574 = arith.remui %573, %c2_i64 : i64
          %575 = ascscalar.bind_local_ptr %569 : memref<1xbf16, 2>
          %576 = arith.divui %566, %c16_i64 : i64
          %577 = arith.muli %576, %c16_i64 : i64
          asccube.get_buf mte2, %572, 0 {cannir.buffer.generated_sync}
          %578 = arith.index_cast %577 : i64 to index
          asccube.copy_gm2l1 %575[%c0], %alloc[%578], %c64_i32, %c384_i32, %c1024_i64, %c1_i32, %c0_i64, %c0_i64, %c64_i32 l2_cache_ctl %c4_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
          asccube.release_buf mte2, %572, 0 {cannir.buffer.generated_sync}
          scf.yield %569, %569, %569, %574, %570, %570, %570 : i64, i64, i64, i64, i64, i64, i64
        } else {
          scf.yield %arg46, %arg54, %arg56, %arg59, %arg66, %arg67, %arg68 : i64, i64, i64, i64, i64, i64, i64
        }
        %74 = arith.cmpi eq, %73#4, %c15_i64 : i64
        %75 = arith.extui %74 : i1 to i64
        %76 = arith.subi %arg44, %c1_i64 : i64
        %77:24 = scf.for %arg69 = %c0_i64 to %76 step %c1_i64 iter_args(%arg70 = %arg24, %arg71 = %arg25, %arg72 = %arg26, %arg73 = %arg27, %arg74 = %arg28, %arg75 = %arg29, %arg76 = %arg30, %arg77 = %arg31, %arg78 = %arg32, %arg79 = %arg33, %arg80 = %arg34, %arg81 = %arg35, %arg82 = %arg36, %arg83 = %arg37, %arg84 = %arg38, %arg85 = %arg39, %arg86 = %arg40, %arg87 = %arg41, %arg88 = %arg42, %arg89 = %arg43, %arg90 = %arg60, %arg91 = %arg61, %arg92 = %arg64, %arg93 = %arg65) -> (i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64)  : i64 {
          %563 = arith.cmpi eq, %arg90, %c1_i64 : i64
          %564 = ascscalar.memref_address %alloc_27 : memref<32768xbf16, 2>
          %565 = ascscalar.memref_address %alloc_26 : memref<32768xbf16, 2>
          %566 = arith.select %563, %564, %565 : i64
          %567 = arith.cmpi eq, %arg90, %c0_i64 : i64
          %568 = arith.select %567, %53, %566 : i64
          %569 = arith.select %563, %c18_i64, %c19_i64 : i64
          %570 = arith.select %567, %c17_i64, %569 : i64
          %571 = arith.addi %arg90, %c1_i64 overflow<nsw> : i64
          %572 = arith.remui %571, %c3_i64 : i64
          %573 = ascscalar.memref_address %alloc_24 : memref<32768xbf16, 2>
          %574 = ascscalar.memref_address %alloc_23 : memref<32768xbf16, 2>
          %575 = arith.select %563, %573, %574 : i64
          %576 = arith.select %567, %54, %575 : i64
          %577 = arith.select %563, %c22_i64, %c23_i64 : i64
          %578 = arith.select %567, %c21_i64, %577 : i64
          asccube.set_mmad_direction m
          %579 = arith.cmpi eq, %arg91, %c2_i64 : i64
          %580 = ascscalar.memref_address %alloc_18 : memref<8192xbf16, 3>
          %581 = ascscalar.memref_address %alloc_20 : memref<8192xbf16, 3>
          %582 = arith.select %579, %580, %581 : i64
          %583 = arith.cmpi eq, %arg91, %c1_i64 : i64
          %584 = ascscalar.memref_address %alloc_16 : memref<8192xbf16, 3>
          %585 = arith.select %583, %584, %582 : i64
          %586 = arith.cmpi eq, %arg91, %c0_i64 : i64
          %587 = ascscalar.memref_address %alloc_14 : memref<8192xbf16, 3>
          %588 = arith.select %586, %587, %585 : i64
          %589 = arith.select %579, %c5_i64, %c6_i64 : i64
          %590 = arith.select %583, %c4_i64, %589 : i64
          %591 = arith.select %586, %c3_i64, %590 : i64
          %592 = arith.addi %arg91, %c1_i64 overflow<nsw> : i64
          %593 = arith.remui %592, %c4_i64 : i64
          %594 = arith.muli %arg48, %arg49 overflow<nsw> : i64
          %595 = arith.remsi %594, %c64_i64 : i64
          %596 = arith.cmpi ne, %595, %c0_i64 : i64
          %597 = arith.cmpi slt, %595, %c0_i64 : i64
          %598 = arith.andi %596, %597 : i1
          %599 = arith.divsi %594, %c64_i64 : i64
          %600 = arith.extui %598 : i1 to i64
          %601 = arith.subi %599, %600 : i64
          %602 = arith.divsi %601, %c64_i64 : i64
          %603 = arith.minsi %602, %c0_i64 : i64
          %604 = arith.muli %603, %c64_i64 : i64
          %605 = arith.subi %601, %604 : i64
          %606 = arith.minsi %605, %c64_i64 : i64
          %607 = arith.cmpi sge, %602, %c0_i64 : i64
          %608 = arith.select %607, %606, %c0_i64 : i64
          %609 = ascscalar.bind_local_ptr %588 : memref<1xbf16, 3>
          %610 = ascscalar.bind_local_ptr %arg46 : memref<1xbf16, 2>
          %611 = arith.index_cast %arg47 : i64 to index
          %612 = arith.divui %611, %c16 : index
          %613 = arith.index_cast %608 : i64 to index
          %614 = arith.addi %613, %c15 : index
          %615 = arith.divui %614, %c16 : index
          %616 = arith.index_cast %615 : index to i64
          %617 = arith.cmpi ugt, %616, %c0_i64 : i64
          scf.if %617 {
            asccube.get_buf mte1, %591, 0 {cannir.buffer.generated_sync}
            asccube.get_buf mte1, %58, 0 {cannir.buffer.generated_sync}
            %1325 = arith.trunci %616 : i64 to i32
            asccube.copy_l12l0a %609[%c0], %610[%612], %c0_i32, %c0_i32, %c4_i32, %1325, %c4_i32, %c4_i32 : memref<1xbf16, 3>, memref<1xbf16, 2>
            asccube.release_buf mte1, %58, 0 {cannir.buffer.generated_sync}
            asccube.release_buf mte1, %591, 0 {cannir.buffer.generated_sync}
          }
          %618 = arith.cmpi eq, %arg62, %c2_i64 : i64
          %619 = ascscalar.memref_address %alloc_19 : memref<8192xbf16, 4>
          %620 = ascscalar.memref_address %alloc_21 : memref<8192xbf16, 4>
          %621 = arith.select %618, %619, %620 : i64
          %622 = arith.cmpi eq, %arg62, %c1_i64 : i64
          %623 = ascscalar.memref_address %alloc_17 : memref<8192xbf16, 4>
          %624 = arith.select %622, %623, %621 : i64
          %625 = arith.cmpi eq, %arg62, %c0_i64 : i64
          %626 = ascscalar.memref_address %alloc_15 : memref<8192xbf16, 4>
          %627 = arith.select %625, %626, %624 : i64
          %628 = arith.select %618, %c9_i64, %c10_i64 : i64
          %629 = arith.select %622, %c8_i64, %628 : i64
          %630 = arith.select %625, %c7_i64, %629 : i64
          %631 = arith.addi %arg62, %c1_i64 overflow<nsw> : i64
          %632 = arith.remui %631, %c4_i64 : i64
          %633 = ascscalar.bind_local_ptr %627 : memref<1xbf16, 4>
          %634 = ascscalar.bind_local_ptr %568 : memref<1xbf16, 2>
          %635 = arith.cmpi eq, %570, %c18_i64 : i64
          %636 = arith.select %635, %c4_i64, %c3_i64 : i64
          %637 = arith.cmpi eq, %570, %c19_i64 : i64
          %638 = arith.select %637, %c5_i64, %636 : i64
          asccube.sync_intra_wait mte1, %638 {cannir.sync.banks = 1 : i32}
          asccube.get_buf mte1, %630, 0 {cannir.buffer.generated_sync}
          asccube.copy_l12l0b %633[%c0], %634[%c0], %c0_i32, %c0_i32, %c4_i32, %c4_i32, %c4_i32, %c8_i32 : memref<1xbf16, 4>, memref<1xbf16, 2>
          asccube.release_buf mte1, %630, 0 {cannir.buffer.generated_sync}
          %639 = ascscalar.bind_local_ptr %576 : memref<1xbf16, 2>
          %640 = arith.cmpi eq, %578, %c22_i64 : i64
          %641 = arith.select %640, %c8_i64, %c7_i64 : i64
          %642 = arith.cmpi eq, %578, %c23_i64 : i64
          %643 = arith.select %642, %c9_i64, %641 : i64
          %644 = arith.addi %643, %c16_i64 overflow<nsw> : i64
          asccube.sync_intra_wait mte1, %644 {cannir.sync.banks = 2 : i32}
          asccube.get_buf mte1, %630, 0 {cannir.buffer.generated_sync}
          asccube.copy_l12l0b %633[%c1024], %639[%c0], %c0_i32, %c0_i32, %c4_i32, %c4_i32, %c4_i32, %c8_i32 : memref<1xbf16, 4>, memref<1xbf16, 2>
          asccube.release_buf mte1, %630, 0 {cannir.buffer.generated_sync}
          asccube.get_buf m, %591, 0 {cannir.buffer.generated_sync}
          asccube.get_buf m, %630, 0 {cannir.buffer.generated_sync}
          asccube.mmad %45[%c0], %609, %633, %c64_i32, %c64_i32, %c128_i32, %c2_i32 {c_matrix_init_val = true, mmad_direction = 0 : i32} : memref<?xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
          asccube.release_buf m, %630, 0 {cannir.buffer.generated_sync}
          asccube.release_buf m, %591, 0 {cannir.buffer.generated_sync}
          %645 = arith.cmpi eq, %593, %c2_i64 : i64
          %646 = arith.select %645, %580, %581 : i64
          %647 = arith.cmpi eq, %593, %c1_i64 : i64
          %648 = arith.select %647, %584, %646 : i64
          %649 = arith.cmpi eq, %593, %c0_i64 : i64
          %650 = arith.select %649, %587, %648 : i64
          %651 = arith.select %645, %c5_i64, %c6_i64 : i64
          %652 = arith.select %647, %c4_i64, %651 : i64
          %653 = arith.select %649, %c3_i64, %652 : i64
          %654 = arith.addi %593, %c1_i64 overflow<nsw> : i64
          %655 = arith.remui %654, %c4_i64 : i64
          %656 = arith.minsi %602, %c1_i64 : i64
          %657 = arith.muli %656, %c64_i64 : i64
          %658 = arith.subi %601, %657 : i64
          %659 = arith.minsi %658, %c64_i64 : i64
          %660 = arith.cmpi sge, %602, %c1_i64 : i64
          %661 = arith.select %660, %659, %c0_i64 : i64
          %662 = ascscalar.bind_local_ptr %650 : memref<1xbf16, 3>
          %663 = arith.index_cast %661 : i64 to index
          %664 = arith.addi %663, %c15 : index
          %665 = arith.divui %664, %c16 : index
          %666 = arith.index_cast %665 : index to i64
          %667 = arith.cmpi ugt, %666, %c0_i64 : i64
          scf.if %667 {
            asccube.get_buf mte1, %653, 0 {cannir.buffer.generated_sync}
            asccube.get_buf mte1, %58, 0 {cannir.buffer.generated_sync}
            %1325 = arith.trunci %666 : i64 to i32
            asccube.copy_l12l0a %662[%c0], %610[%612], %c0_i32, %c4_i32, %c4_i32, %1325, %c4_i32, %c4_i32 : memref<1xbf16, 3>, memref<1xbf16, 2>
            asccube.release_buf mte1, %58, 0 {cannir.buffer.generated_sync}
            asccube.release_buf mte1, %653, 0 {cannir.buffer.generated_sync}
          }
          %668 = arith.cmpi eq, %632, %c2_i64 : i64
          %669 = arith.select %668, %619, %620 : i64
          %670 = arith.cmpi eq, %632, %c1_i64 : i64
          %671 = arith.select %670, %623, %669 : i64
          %672 = arith.cmpi eq, %632, %c0_i64 : i64
          %673 = arith.select %672, %626, %671 : i64
          %674 = arith.select %668, %c9_i64, %c10_i64 : i64
          %675 = arith.select %670, %c8_i64, %674 : i64
          %676 = arith.select %672, %c7_i64, %675 : i64
          %677 = arith.addi %632, %c1_i64 overflow<nsw> : i64
          %678 = arith.remui %677, %c4_i64 : i64
          %679 = ascscalar.bind_local_ptr %673 : memref<1xbf16, 4>
          asccube.get_buf mte1, %676, 0 {cannir.buffer.generated_sync}
          asccube.copy_l12l0b %679[%c0], %634[%c0], %c0_i32, %c4_i32, %c4_i32, %c4_i32, %c4_i32, %c8_i32 : memref<1xbf16, 4>, memref<1xbf16, 2>
          asccube.copy_l12l0b %679[%c1024], %639[%c0], %c0_i32, %c4_i32, %c4_i32, %c4_i32, %c4_i32, %c8_i32 : memref<1xbf16, 4>, memref<1xbf16, 2>
          asccube.release_buf mte1, %676, 0 {cannir.buffer.generated_sync}
          asccube.get_buf m, %653, 0 {cannir.buffer.generated_sync}
          asccube.get_buf m, %676, 0 {cannir.buffer.generated_sync}
          asccube.mmad %45[%c0], %662, %679, %c64_i32, %c64_i32, %c128_i32, %c2_i32 {c_matrix_init_val = false, mmad_direction = 0 : i32} : memref<?xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
          asccube.release_buf m, %676, 0 {cannir.buffer.generated_sync}
          asccube.release_buf m, %653, 0 {cannir.buffer.generated_sync}
          %680 = arith.cmpi eq, %655, %c2_i64 : i64
          %681 = arith.select %680, %580, %581 : i64
          %682 = arith.cmpi eq, %655, %c1_i64 : i64
          %683 = arith.select %682, %584, %681 : i64
          %684 = arith.cmpi eq, %655, %c0_i64 : i64
          %685 = arith.select %684, %587, %683 : i64
          %686 = arith.select %680, %c5_i64, %c6_i64 : i64
          %687 = arith.select %682, %c4_i64, %686 : i64
          %688 = arith.select %684, %c3_i64, %687 : i64
          %689 = arith.addi %655, %c1_i64 overflow<nsw> : i64
          %690 = arith.remui %689, %c4_i64 : i64
          %691 = arith.minsi %602, %c2_i64 : i64
          %692 = arith.muli %691, %c64_i64 : i64
          %693 = arith.subi %601, %692 : i64
          %694 = arith.minsi %693, %c64_i64 : i64
          %695 = arith.cmpi sge, %602, %c2_i64 : i64
          %696 = arith.select %695, %694, %c0_i64 : i64
          %697 = ascscalar.bind_local_ptr %685 : memref<1xbf16, 3>
          %698 = arith.index_cast %696 : i64 to index
          %699 = arith.addi %698, %c15 : index
          %700 = arith.divui %699, %c16 : index
          %701 = arith.index_cast %700 : index to i64
          %702 = arith.cmpi ugt, %701, %c0_i64 : i64
          scf.if %702 {
            asccube.get_buf mte1, %688, 0 {cannir.buffer.generated_sync}
            asccube.get_buf mte1, %58, 0 {cannir.buffer.generated_sync}
            %1325 = arith.trunci %701 : i64 to i32
            asccube.copy_l12l0a %697[%c0], %610[%612], %c0_i32, %c8_i32, %c4_i32, %1325, %c4_i32, %c4_i32 : memref<1xbf16, 3>, memref<1xbf16, 2>
            asccube.release_buf mte1, %58, 0 {cannir.buffer.generated_sync}
            asccube.release_buf mte1, %688, 0 {cannir.buffer.generated_sync}
          }
          %703 = arith.cmpi eq, %678, %c2_i64 : i64
          %704 = arith.select %703, %619, %620 : i64
          %705 = arith.cmpi eq, %678, %c1_i64 : i64
          %706 = arith.select %705, %623, %704 : i64
          %707 = arith.cmpi eq, %678, %c0_i64 : i64
          %708 = arith.select %707, %626, %706 : i64
          %709 = arith.select %703, %c9_i64, %c10_i64 : i64
          %710 = arith.select %705, %c8_i64, %709 : i64
          %711 = arith.select %707, %c7_i64, %710 : i64
          %712 = arith.addi %678, %c1_i64 overflow<nsw> : i64
          %713 = arith.remui %712, %c4_i64 : i64
          %714 = ascscalar.bind_local_ptr %708 : memref<1xbf16, 4>
          asccube.get_buf mte1, %711, 0 {cannir.buffer.generated_sync}
          asccube.copy_l12l0b %714[%c0], %634[%c0], %c0_i32, %c8_i32, %c4_i32, %c4_i32, %c4_i32, %c8_i32 : memref<1xbf16, 4>, memref<1xbf16, 2>
          asccube.copy_l12l0b %714[%c1024], %639[%c0], %c0_i32, %c8_i32, %c4_i32, %c4_i32, %c4_i32, %c8_i32 : memref<1xbf16, 4>, memref<1xbf16, 2>
          asccube.release_buf mte1, %711, 0 {cannir.buffer.generated_sync}
          asccube.get_buf m, %688, 0 {cannir.buffer.generated_sync}
          asccube.get_buf m, %711, 0 {cannir.buffer.generated_sync}
          asccube.mmad %45[%c0], %697, %714, %c64_i32, %c64_i32, %c128_i32, %c2_i32 {c_matrix_init_val = false, mmad_direction = 0 : i32} : memref<?xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
          asccube.release_buf m, %711, 0 {cannir.buffer.generated_sync}
          asccube.release_buf m, %688, 0 {cannir.buffer.generated_sync}
          %715 = arith.cmpi eq, %690, %c2_i64 : i64
          %716 = arith.select %715, %580, %581 : i64
          %717 = arith.cmpi eq, %690, %c1_i64 : i64
          %718 = arith.select %717, %584, %716 : i64
          %719 = arith.cmpi eq, %690, %c0_i64 : i64
          %720 = arith.select %719, %587, %718 : i64
          %721 = arith.select %715, %c5_i64, %c6_i64 : i64
          %722 = arith.select %717, %c4_i64, %721 : i64
          %723 = arith.select %719, %c3_i64, %722 : i64
          %724 = arith.addi %690, %c1_i64 overflow<nsw> : i64
          %725 = arith.remui %724, %c4_i64 : i64
          %726 = arith.minsi %602, %c3_i64 : i64
          %727 = arith.muli %726, %c64_i64 : i64
          %728 = arith.subi %601, %727 : i64
          %729 = arith.minsi %728, %c64_i64 : i64
          %730 = arith.cmpi sge, %602, %c3_i64 : i64
          %731 = arith.select %730, %729, %c0_i64 : i64
          %732 = ascscalar.bind_local_ptr %720 : memref<1xbf16, 3>
          %733 = arith.index_cast %731 : i64 to index
          %734 = arith.addi %733, %c15 : index
          %735 = arith.divui %734, %c16 : index
          %736 = arith.index_cast %735 : index to i64
          %737 = arith.cmpi ugt, %736, %c0_i64 : i64
          scf.if %737 {
            asccube.get_buf mte1, %723, 0 {cannir.buffer.generated_sync}
            asccube.get_buf mte1, %58, 0 {cannir.buffer.generated_sync}
            %1325 = arith.trunci %736 : i64 to i32
            asccube.copy_l12l0a %732[%c0], %610[%612], %c0_i32, %c12_i32, %c4_i32, %1325, %c4_i32, %c4_i32 : memref<1xbf16, 3>, memref<1xbf16, 2>
            asccube.release_buf mte1, %58, 0 {cannir.buffer.generated_sync}
            asccube.release_buf mte1, %723, 0 {cannir.buffer.generated_sync}
          }
          %738 = arith.cmpi eq, %713, %c2_i64 : i64
          %739 = arith.select %738, %619, %620 : i64
          %740 = arith.cmpi eq, %713, %c1_i64 : i64
          %741 = arith.select %740, %623, %739 : i64
          %742 = arith.cmpi eq, %713, %c0_i64 : i64
          %743 = arith.select %742, %626, %741 : i64
          %744 = arith.select %738, %c9_i64, %c10_i64 : i64
          %745 = arith.select %740, %c8_i64, %744 : i64
          %746 = arith.select %742, %c7_i64, %745 : i64
          %747 = arith.addi %713, %c1_i64 overflow<nsw> : i64
          %748 = arith.remui %747, %c4_i64 : i64
          %749 = ascscalar.bind_local_ptr %743 : memref<1xbf16, 4>
          asccube.get_buf mte1, %746, 0 {cannir.buffer.generated_sync}
          asccube.copy_l12l0b %749[%c0], %634[%c0], %c0_i32, %c12_i32, %c4_i32, %c4_i32, %c4_i32, %c8_i32 : memref<1xbf16, 4>, memref<1xbf16, 2>
          asccube.copy_l12l0b %749[%c1024], %639[%c0], %c0_i32, %c12_i32, %c4_i32, %c4_i32, %c4_i32, %c8_i32 : memref<1xbf16, 4>, memref<1xbf16, 2>
          asccube.release_buf mte1, %746, 0 {cannir.buffer.generated_sync}
          asccube.get_buf m, %723, 0 {cannir.buffer.generated_sync}
          asccube.get_buf m, %746, 0 {cannir.buffer.generated_sync}
          asccube.mmad %45[%c0], %732, %749, %c64_i32, %c64_i32, %c128_i32, %c2_i32 {c_matrix_init_val = false, mmad_direction = 0 : i32} : memref<?xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
          asccube.release_buf m, %746, 0 {cannir.buffer.generated_sync}
          asccube.release_buf m, %723, 0 {cannir.buffer.generated_sync}
          %750 = arith.cmpi eq, %725, %c2_i64 : i64
          %751 = arith.select %750, %580, %581 : i64
          %752 = arith.cmpi eq, %725, %c1_i64 : i64
          %753 = arith.select %752, %584, %751 : i64
          %754 = arith.cmpi eq, %725, %c0_i64 : i64
          %755 = arith.select %754, %587, %753 : i64
          %756 = arith.select %750, %c5_i64, %c6_i64 : i64
          %757 = arith.select %752, %c4_i64, %756 : i64
          %758 = arith.select %754, %c3_i64, %757 : i64
          %759 = arith.addi %725, %c1_i64 overflow<nsw> : i64
          %760 = arith.remui %759, %c4_i64 : i64
          %761 = arith.minsi %602, %c4_i64 : i64
          %762 = arith.muli %761, %c64_i64 : i64
          %763 = arith.subi %601, %762 : i64
          %764 = arith.minsi %763, %c64_i64 : i64
          %765 = arith.cmpi sge, %602, %c4_i64 : i64
          %766 = arith.select %765, %764, %c0_i64 : i64
          %767 = ascscalar.bind_local_ptr %755 : memref<1xbf16, 3>
          %768 = arith.index_cast %766 : i64 to index
          %769 = arith.addi %768, %c15 : index
          %770 = arith.divui %769, %c16 : index
          %771 = arith.index_cast %770 : index to i64
          %772 = arith.cmpi ugt, %771, %c0_i64 : i64
          scf.if %772 {
            asccube.get_buf mte1, %758, 0 {cannir.buffer.generated_sync}
            asccube.get_buf mte1, %58, 0 {cannir.buffer.generated_sync}
            %1325 = arith.trunci %771 : i64 to i32
            asccube.copy_l12l0a %767[%c0], %610[%612], %c0_i32, %c16_i32, %c4_i32, %1325, %c4_i32, %c4_i32 : memref<1xbf16, 3>, memref<1xbf16, 2>
            asccube.release_buf mte1, %58, 0 {cannir.buffer.generated_sync}
            asccube.release_buf mte1, %758, 0 {cannir.buffer.generated_sync}
          }
          %773 = arith.cmpi eq, %748, %c2_i64 : i64
          %774 = arith.select %773, %619, %620 : i64
          %775 = arith.cmpi eq, %748, %c1_i64 : i64
          %776 = arith.select %775, %623, %774 : i64
          %777 = arith.cmpi eq, %748, %c0_i64 : i64
          %778 = arith.select %777, %626, %776 : i64
          %779 = arith.select %773, %c9_i64, %c10_i64 : i64
          %780 = arith.select %775, %c8_i64, %779 : i64
          %781 = arith.select %777, %c7_i64, %780 : i64
          %782 = arith.addi %748, %c1_i64 overflow<nsw> : i64
          %783 = arith.remui %782, %c4_i64 : i64
          %784 = ascscalar.bind_local_ptr %778 : memref<1xbf16, 4>
          asccube.get_buf mte1, %781, 0 {cannir.buffer.generated_sync}
          asccube.copy_l12l0b %784[%c0], %634[%c0], %c0_i32, %c16_i32, %c4_i32, %c4_i32, %c4_i32, %c8_i32 : memref<1xbf16, 4>, memref<1xbf16, 2>
          asccube.copy_l12l0b %784[%c1024], %639[%c0], %c0_i32, %c16_i32, %c4_i32, %c4_i32, %c4_i32, %c8_i32 : memref<1xbf16, 4>, memref<1xbf16, 2>
          asccube.release_buf mte1, %781, 0 {cannir.buffer.generated_sync}
          asccube.get_buf m, %758, 0 {cannir.buffer.generated_sync}
          asccube.get_buf m, %781, 0 {cannir.buffer.generated_sync}
          asccube.mmad %45[%c0], %767, %784, %c64_i32, %c64_i32, %c128_i32, %c2_i32 {c_matrix_init_val = false, mmad_direction = 0 : i32} : memref<?xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
          asccube.release_buf m, %781, 0 {cannir.buffer.generated_sync}
          asccube.release_buf m, %758, 0 {cannir.buffer.generated_sync}
          %785 = arith.cmpi eq, %760, %c2_i64 : i64
          %786 = arith.select %785, %580, %581 : i64
          %787 = arith.cmpi eq, %760, %c1_i64 : i64
          %788 = arith.select %787, %584, %786 : i64
          %789 = arith.cmpi eq, %760, %c0_i64 : i64
          %790 = arith.select %789, %587, %788 : i64
          %791 = arith.select %785, %c5_i64, %c6_i64 : i64
          %792 = arith.select %787, %c4_i64, %791 : i64
          %793 = arith.select %789, %c3_i64, %792 : i64
          %794 = arith.addi %760, %c1_i64 overflow<nsw> : i64
          %795 = arith.remui %794, %c4_i64 : i64
          %796 = arith.minsi %602, %c5_i64 : i64
          %797 = arith.muli %796, %c64_i64 : i64
          %798 = arith.subi %601, %797 : i64
          %799 = arith.minsi %798, %c64_i64 : i64
          %800 = arith.cmpi sge, %602, %c5_i64 : i64
          %801 = arith.select %800, %799, %c0_i64 : i64
          %802 = ascscalar.bind_local_ptr %790 : memref<1xbf16, 3>
          %803 = arith.index_cast %801 : i64 to index
          %804 = arith.addi %803, %c15 : index
          %805 = arith.divui %804, %c16 : index
          %806 = arith.index_cast %805 : index to i64
          %807 = arith.cmpi ugt, %806, %c0_i64 : i64
          scf.if %807 {
            asccube.get_buf mte1, %793, 0 {cannir.buffer.generated_sync}
            asccube.get_buf mte1, %58, 0 {cannir.buffer.generated_sync}
            %1325 = arith.trunci %806 : i64 to i32
            asccube.copy_l12l0a %802[%c0], %610[%612], %c0_i32, %c20_i32, %c4_i32, %1325, %c4_i32, %c4_i32 : memref<1xbf16, 3>, memref<1xbf16, 2>
            asccube.release_buf mte1, %58, 0 {cannir.buffer.generated_sync}
            asccube.release_buf mte1, %793, 0 {cannir.buffer.generated_sync}
          }
          %808 = arith.cmpi eq, %783, %c2_i64 : i64
          %809 = arith.select %808, %619, %620 : i64
          %810 = arith.cmpi eq, %783, %c1_i64 : i64
          %811 = arith.select %810, %623, %809 : i64
          %812 = arith.cmpi eq, %783, %c0_i64 : i64
          %813 = arith.select %812, %626, %811 : i64
          %814 = arith.select %808, %c9_i64, %c10_i64 : i64
          %815 = arith.select %810, %c8_i64, %814 : i64
          %816 = arith.select %812, %c7_i64, %815 : i64
          %817 = arith.addi %783, %c1_i64 overflow<nsw> : i64
          %818 = arith.remui %817, %c4_i64 : i64
          %819 = ascscalar.bind_local_ptr %813 : memref<1xbf16, 4>
          asccube.get_buf mte1, %816, 0 {cannir.buffer.generated_sync}
          asccube.copy_l12l0b %819[%c0], %634[%c0], %c0_i32, %c20_i32, %c4_i32, %c4_i32, %c4_i32, %c8_i32 : memref<1xbf16, 4>, memref<1xbf16, 2>
          asccube.copy_l12l0b %819[%c1024], %639[%c0], %c0_i32, %c20_i32, %c4_i32, %c4_i32, %c4_i32, %c8_i32 : memref<1xbf16, 4>, memref<1xbf16, 2>
          asccube.release_buf mte1, %816, 0 {cannir.buffer.generated_sync}
          asccube.get_buf m, %793, 0 {cannir.buffer.generated_sync}
          asccube.get_buf m, %816, 0 {cannir.buffer.generated_sync}
          asccube.mmad %45[%c0], %802, %819, %c64_i32, %c64_i32, %c128_i32, %c2_i32 {c_matrix_init_val = false, mmad_direction = 0 : i32} : memref<?xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
          asccube.release_buf m, %816, 0 {cannir.buffer.generated_sync}
          asccube.release_buf m, %793, 0 {cannir.buffer.generated_sync}
          %820 = arith.cmpi eq, %795, %c2_i64 : i64
          %821 = arith.select %820, %580, %581 : i64
          %822 = arith.cmpi eq, %795, %c1_i64 : i64
          %823 = arith.select %822, %584, %821 : i64
          %824 = arith.cmpi eq, %795, %c0_i64 : i64
          %825 = arith.select %824, %587, %823 : i64
          %826 = arith.select %820, %c5_i64, %c6_i64 : i64
          %827 = arith.select %822, %c4_i64, %826 : i64
          %828 = arith.select %824, %c3_i64, %827 : i64
          %829 = arith.addi %795, %c1_i64 overflow<nsw> : i64
          %830 = arith.remui %829, %c4_i64 : i64
          %831 = ascscalar.bind_local_ptr %825 : memref<1xbf16, 3>
          asccube.get_buf mte1, %828, 0 {cannir.buffer.generated_sync}
          asccube.copy_l12l0a %831[%c0], %36[%c0], %c0_i32, %c0_i32, %c4_i32, %c4_i32, %c4_i32, %c4_i32 : memref<1xbf16, 3>, memref<?xbf16, 2>
          asccube.release_buf mte1, %828, 0 {cannir.buffer.generated_sync}
          %832 = arith.cmpi eq, %818, %c2_i64 : i64
          %833 = arith.select %832, %619, %620 : i64
          %834 = arith.cmpi eq, %818, %c1_i64 : i64
          %835 = arith.select %834, %623, %833 : i64
          %836 = arith.cmpi eq, %818, %c0_i64 : i64
          %837 = arith.select %836, %626, %835 : i64
          %838 = arith.select %832, %c9_i64, %c10_i64 : i64
          %839 = arith.select %834, %c8_i64, %838 : i64
          %840 = arith.select %836, %c7_i64, %839 : i64
          %841 = arith.addi %818, %c1_i64 overflow<nsw> : i64
          %842 = arith.remui %841, %c4_i64 : i64
          %843 = ascscalar.bind_local_ptr %837 : memref<1xbf16, 4>
          asccube.get_buf mte1, %840, 0 {cannir.buffer.generated_sync}
          asccube.copy_l12l0b %843[%c0], %634[%c0], %c0_i32, %c24_i32, %c4_i32, %c4_i32, %c4_i32, %c8_i32 : memref<1xbf16, 4>, memref<1xbf16, 2>
          asccube.copy_l12l0b %843[%c1024], %639[%c0], %c0_i32, %c24_i32, %c4_i32, %c4_i32, %c4_i32, %c8_i32 : memref<1xbf16, 4>, memref<1xbf16, 2>
          asccube.release_buf mte1, %840, 0 {cannir.buffer.generated_sync}
          asccube.get_buf m, %828, 0 {cannir.buffer.generated_sync}
          asccube.get_buf m, %840, 0 {cannir.buffer.generated_sync}
          asccube.mmad %45[%c0], %831, %843, %c64_i32, %c64_i32, %c128_i32, %c2_i32 {c_matrix_init_val = false, mmad_direction = 0 : i32} : memref<?xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
          asccube.release_buf m, %840, 0 {cannir.buffer.generated_sync}
          asccube.release_buf m, %828, 0 {cannir.buffer.generated_sync}
          %844 = arith.cmpi eq, %830, %c2_i64 : i64
          %845 = arith.select %844, %580, %581 : i64
          %846 = arith.cmpi eq, %830, %c1_i64 : i64
          %847 = arith.select %846, %584, %845 : i64
          %848 = arith.cmpi eq, %830, %c0_i64 : i64
          %849 = arith.select %848, %587, %847 : i64
          %850 = arith.select %844, %c5_i64, %c6_i64 : i64
          %851 = arith.select %846, %c4_i64, %850 : i64
          %852 = arith.select %848, %c3_i64, %851 : i64
          %853 = arith.addi %830, %c1_i64 overflow<nsw> : i64
          %854 = arith.remui %853, %c4_i64 : i64
          %855 = ascscalar.bind_local_ptr %849 : memref<1xbf16, 3>
          asccube.get_buf mte1, %852, 0 {cannir.buffer.generated_sync}
          asccube.get_buf mte1, %c2_i64, 0 {cannir.buffer.generated_sync}
          asccube.copy_l12l0a %855[%c0], %36[%c0], %c0_i32, %c4_i32, %c4_i32, %c4_i32, %c4_i32, %c4_i32 : memref<1xbf16, 3>, memref<?xbf16, 2>
          asccube.release_buf mte1, %c2_i64, 0 {cannir.buffer.generated_sync}
          asccube.release_buf mte1, %852, 0 {cannir.buffer.generated_sync}
          %856 = arith.cmpi eq, %842, %c2_i64 : i64
          %857 = arith.select %856, %619, %620 : i64
          %858 = arith.cmpi eq, %842, %c1_i64 : i64
          %859 = arith.select %858, %623, %857 : i64
          %860 = arith.cmpi eq, %842, %c0_i64 : i64
          %861 = arith.select %860, %626, %859 : i64
          %862 = arith.select %856, %c9_i64, %c10_i64 : i64
          %863 = arith.select %858, %c8_i64, %862 : i64
          %864 = arith.select %860, %c7_i64, %863 : i64
          %865 = arith.addi %842, %c1_i64 overflow<nsw> : i64
          %866 = arith.remui %865, %c4_i64 : i64
          %867 = ascscalar.bind_local_ptr %861 : memref<1xbf16, 4>
          asccube.get_buf mte1, %864, 0 {cannir.buffer.generated_sync}
          asccube.copy_l12l0b %867[%c0], %634[%c0], %c0_i32, %c28_i32, %c4_i32, %c4_i32, %c4_i32, %c8_i32 : memref<1xbf16, 4>, memref<1xbf16, 2>
          asccube.copy_l12l0b %867[%c1024], %639[%c0], %c0_i32, %c28_i32, %c4_i32, %c4_i32, %c4_i32, %c8_i32 : memref<1xbf16, 4>, memref<1xbf16, 2>
          asccube.release_buf mte1, %864, 0 {cannir.buffer.generated_sync}
          asccube.get_buf m, %852, 0 {cannir.buffer.generated_sync}
          asccube.get_buf m, %864, 0 {cannir.buffer.generated_sync}
          asccube.mmad %45[%c0], %855, %867, %c64_i32, %c64_i32, %c128_i32, %c3_i32 {c_matrix_init_val = false, mmad_direction = 0 : i32} : memref<?xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
          asccube.release_buf m, %864, 0 {cannir.buffer.generated_sync}
          asccube.release_buf m, %852, 0 {cannir.buffer.generated_sync}
          asccube.sync_intra_wait fixpipe, %c0_i64 {cannir.sync.banks = 3 : i32}
          asccube.sync_intra_wait fixpipe, %c16_i64 {cannir.sync.banks = 3 : i32}
          asccube.copy_l0c2ub %52[%c0], %45[%c0], %c128_i32, %c64_i32, %c128_i32, %c64_i32, %c3_i32 sub_block_id %c0_i32 {dual_dst_ctl = 1 : i32, nz2nd_en = true, relu_en = false} : memref<?xf32, 1>, memref<?xf32, 5>
          asccube.sync_intra_arrive fixpipe, %c0_i64 {cannir.sync.banks = 3 : i32}
          asccube.sync_intra_arrive fixpipe, %c16_i64 {cannir.sync.banks = 3 : i32}
          asccube.set_mmad_direction n
          %868 = arith.cmpi eq, %854, %c2_i64 : i64
          %869 = arith.select %868, %580, %581 : i64
          %870 = arith.cmpi eq, %854, %c1_i64 : i64
          %871 = arith.select %870, %584, %869 : i64
          %872 = arith.cmpi eq, %854, %c0_i64 : i64
          %873 = arith.select %872, %587, %871 : i64
          %874 = arith.select %868, %c5_i64, %c6_i64 : i64
          %875 = arith.select %870, %c4_i64, %874 : i64
          %876 = arith.select %872, %c3_i64, %875 : i64
          %877 = arith.addi %854, %c1_i64 overflow<nsw> : i64
          %878 = arith.remui %877, %c4_i64 : i64
          %879 = arith.cmpi eq, %866, %c2_i64 : i64
          %880 = arith.select %879, %619, %620 : i64
          %881 = arith.cmpi eq, %866, %c1_i64 : i64
          %882 = arith.select %881, %623, %880 : i64
          %883 = arith.cmpi eq, %866, %c0_i64 : i64
          %884 = arith.select %883, %626, %882 : i64
          %885 = arith.select %879, %c9_i64, %c10_i64 : i64
          %886 = arith.select %881, %c8_i64, %885 : i64
          %887 = arith.select %883, %c7_i64, %886 : i64
          %888 = arith.addi %866, %c1_i64 overflow<nsw> : i64
          %889 = arith.remui %888, %c4_i64 : i64
          %890 = arith.remsi %arg72, %c64_i64 : i64
          %891 = arith.cmpi ne, %890, %c0_i64 : i64
          %892 = arith.cmpi slt, %890, %c0_i64 : i64
          %893 = arith.andi %891, %892 : i1
          %894 = arith.divsi %arg72, %c64_i64 : i64
          %895 = arith.extui %893 : i1 to i64
          %896 = arith.subi %894, %895 : i64
          %897 = arith.divsi %896, %c64_i64 : i64
          %898 = arith.minsi %897, %c0_i64 : i64
          %899 = arith.muli %898, %c64_i64 : i64
          %900 = arith.subi %896, %899 : i64
          %901 = arith.minsi %900, %c64_i64 : i64
          %902 = arith.cmpi sge, %897, %c0_i64 : i64
          %903 = arith.select %902, %901, %c0_i64 : i64
          %904 = ascscalar.bind_local_ptr %884 : memref<1xbf16, 4>
          %905 = ascscalar.bind_local_ptr %arg70 : memref<1xbf16, 2>
          %906 = arith.index_cast %arg71 : i64 to index
          %907 = arith.divui %906, %c16 : index
          %908 = arith.index_cast %903 : i64 to index
          %909 = arith.addi %908, %c15 : index
          %910 = arith.divui %909, %c16 : index
          %911 = arith.index_cast %910 : index to i64
          %912 = arith.cmpi ugt, %911, %c0_i64 : i64
          scf.if %912 {
            asccube.get_buf mte1, %887, 0 {cannir.buffer.generated_sync}
            %1325 = arith.trunci %911 : i64 to i32
            asccube.copy_l12l0b %904[%c0], %905[%907], %c0_i32, %c0_i32, %c4_i32, %1325, %c4_i32, %c4_i32 {transpose = true} : memref<1xbf16, 4>, memref<1xbf16, 2>
            asccube.release_buf mte1, %887, 0 {cannir.buffer.generated_sync}
          }
          %913 = arith.remsi %arg82, %c64_i64 : i64
          %914 = arith.cmpi ne, %913, %c0_i64 : i64
          %915 = arith.cmpi slt, %913, %c0_i64 : i64
          %916 = arith.andi %914, %915 : i1
          %917 = arith.divsi %arg82, %c64_i64 : i64
          %918 = arith.extui %916 : i1 to i64
          %919 = arith.subi %917, %918 : i64
          %920 = arith.divsi %919, %c64_i64 : i64
          %921 = arith.minsi %920, %c0_i64 : i64
          %922 = arith.muli %921, %c64_i64 : i64
          %923 = arith.subi %919, %922 : i64
          %924 = arith.minsi %923, %c64_i64 : i64
          %925 = arith.cmpi sge, %920, %c0_i64 : i64
          %926 = arith.select %925, %924, %c0_i64 : i64
          %927 = ascscalar.bind_local_ptr %arg80 : memref<1xbf16, 2>
          %928 = arith.index_cast %arg81 : i64 to index
          %929 = arith.divui %928, %c16 : index
          %930 = arith.index_cast %926 : i64 to index
          %931 = arith.addi %930, %c15 : index
          %932 = arith.divui %931, %c16 : index
          %933 = arith.index_cast %932 : index to i64
          %934 = arith.cmpi ugt, %933, %c0_i64 : i64
          scf.if %934 {
            asccube.get_buf mte1, %887, 0 {cannir.buffer.generated_sync}
            %1325 = arith.trunci %933 : i64 to i32
            asccube.copy_l12l0b %904[%c4096], %927[%929], %c0_i32, %c0_i32, %c4_i32, %1325, %c4_i32, %c4_i32 {transpose = true} : memref<1xbf16, 4>, memref<1xbf16, 2>
            asccube.release_buf mte1, %887, 0 {cannir.buffer.generated_sync}
          }
          %935 = ascscalar.bind_local_ptr %873 : memref<1xbf16, 3>
          %936 = ascscalar.bitcast_memref %alloc_28 byte_offset = %c0_i64 : memref<8192xbf16, 2> -> memref<?xbf16, 2>
          asccube.sync_intra_wait mte1, %c6_i64 {cannir.sync.banks = 3 : i32}
          asccube.sync_intra_wait mte1, %c22_i64 {cannir.sync.banks = 3 : i32}
          asccube.get_buf mte1, %876, 0 {cannir.buffer.generated_sync}
          asccube.copy_l12l0a %935[%c0], %936[%c0], %c0_i32, %c0_i32, %c4_i32, %c8_i32, %c4_i32, %c4_i32 : memref<1xbf16, 3>, memref<?xbf16, 2>
          asccube.release_buf mte1, %876, 0 {cannir.buffer.generated_sync}
          asccube.sync_intra_arrive mte1, %c6_i64 {cannir.sync.banks = 3 : i32}
          asccube.sync_intra_arrive mte1, %c22_i64 {cannir.sync.banks = 3 : i32}
          %937 = arith.cmpi eq, %arg63, %c6_i64 : i64
          %938 = ascscalar.memref_address %alloc_36 : memref<4096xf32, 5>
          %939 = ascscalar.memref_address %alloc_35 : memref<4096xf32, 5>
          %940 = arith.select %937, %938, %939 : i64
          %941 = arith.cmpi eq, %arg63, %c5_i64 : i64
          %942 = ascscalar.memref_address %alloc_34 : memref<4096xf32, 5>
          %943 = arith.select %941, %942, %940 : i64
          %944 = arith.cmpi eq, %arg63, %c4_i64 : i64
          %945 = ascscalar.memref_address %alloc_33 : memref<4096xf32, 5>
          %946 = arith.select %944, %945, %943 : i64
          %947 = arith.cmpi eq, %arg63, %c3_i64 : i64
          %948 = ascscalar.memref_address %alloc_32 : memref<4096xf32, 5>
          %949 = arith.select %947, %948, %946 : i64
          %950 = arith.cmpi eq, %arg63, %c2_i64 : i64
          %951 = ascscalar.memref_address %alloc_31 : memref<4096xf32, 5>
          %952 = arith.select %950, %951, %949 : i64
          %953 = arith.cmpi eq, %arg63, %c1_i64 : i64
          %954 = ascscalar.memref_address %alloc_30 : memref<4096xf32, 5>
          %955 = arith.select %953, %954, %952 : i64
          %956 = arith.cmpi eq, %arg63, %c0_i64 : i64
          %957 = ascscalar.memref_address %alloc_29 : memref<4096xf32, 5>
          %958 = arith.select %956, %957, %955 : i64
          %959 = arith.addi %arg63, %c1_i64 overflow<nsw> : i64
          %960 = arith.remui %959, %c8_i64 : i64
          %961 = ascscalar.bind_local_ptr %958 : memref<1xf32, 5>
          asccube.get_buf m, %876, 0 {cannir.buffer.generated_sync}
          asccube.get_buf m, %887, 0 {cannir.buffer.generated_sync}
          asccube.mmad %961[%c0], %935, %904, %c64_i32, %c128_i32, %c64_i32, %c3_i32 {c_matrix_init_val = true, mmad_direction = 1 : i32} : memref<1xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
          asccube.release_buf m, %887, 0 {cannir.buffer.generated_sync}
          asccube.release_buf m, %876, 0 {cannir.buffer.generated_sync}
          %962 = arith.cmpi eq, %arg69, %c0_i64 : i64
          scf.if %962 {
            %1325 = ascscalar.bitcast_memref %alloc_11 byte_offset = %c0_i64 : memref<16384xf32, 1> -> memref<?xf32, 1>
            asccube.sync_intra_wait fixpipe, %c1_i64 {cannir.sync.banks = 3 : i32}
            asccube.sync_intra_wait fixpipe, %c17_i64 {cannir.sync.banks = 3 : i32}
            asccube.copy_l0c2ub %1325[%c0], %961[%c0], %c64_i32, %c64_i32, %c256_i32, %c64_i32, %c3_i32 sub_block_id %c0_i32 {dual_dst_ctl = 1 : i32, nz2nd_en = true, relu_en = false} : memref<?xf32, 1>, memref<1xf32, 5>
          } else {
            %1325 = ascscalar.bitcast_memref %alloc_11 byte_offset = %c0_i64 : memref<16384xf32, 1> -> memref<?xf32, 1>
            asccube.sync_intra_wait fixpipe, %c1_i64 {cannir.sync.banks = 3 : i32}
            asccube.sync_intra_wait fixpipe, %c17_i64 {cannir.sync.banks = 3 : i32}
            asccube.copy_l0c2ub %1325[%c8192], %961[%c0], %c64_i32, %c64_i32, %c256_i32, %c64_i32, %c3_i32 sub_block_id %c0_i32 {dual_dst_ctl = 1 : i32, nz2nd_en = true, relu_en = false} : memref<?xf32, 1>, memref<1xf32, 5>
          }
          %963 = arith.cmpi eq, %889, %c2_i64 : i64
          %964 = arith.select %963, %619, %620 : i64
          %965 = arith.cmpi eq, %889, %c1_i64 : i64
          %966 = arith.select %965, %623, %964 : i64
          %967 = arith.cmpi eq, %889, %c0_i64 : i64
          %968 = arith.select %967, %626, %966 : i64
          %969 = arith.select %963, %c9_i64, %c10_i64 : i64
          %970 = arith.select %965, %c8_i64, %969 : i64
          %971 = arith.select %967, %c7_i64, %970 : i64
          %972 = arith.addi %889, %c1_i64 overflow<nsw> : i64
          %973 = arith.remui %972, %c4_i64 : i64
          %974 = arith.minsi %897, %c1_i64 : i64
          %975 = arith.muli %974, %c64_i64 : i64
          %976 = arith.subi %896, %975 : i64
          %977 = arith.minsi %976, %c64_i64 : i64
          %978 = arith.cmpi sge, %897, %c1_i64 : i64
          %979 = arith.select %978, %977, %c0_i64 : i64
          %980 = ascscalar.bind_local_ptr %968 : memref<1xbf16, 4>
          %981 = arith.index_cast %979 : i64 to index
          %982 = arith.addi %981, %c15 : index
          %983 = arith.divui %982, %c16 : index
          %984 = arith.index_cast %983 : index to i64
          %985 = arith.cmpi ugt, %984, %c0_i64 : i64
          scf.if %985 {
            asccube.get_buf mte1, %971, 0 {cannir.buffer.generated_sync}
            %1325 = arith.trunci %984 : i64 to i32
            asccube.copy_l12l0b %980[%c0], %905[%907], %c0_i32, %c4_i32, %c4_i32, %1325, %c4_i32, %c4_i32 {transpose = true} : memref<1xbf16, 4>, memref<1xbf16, 2>
            asccube.release_buf mte1, %971, 0 {cannir.buffer.generated_sync}
          }
          %986 = arith.minsi %920, %c1_i64 : i64
          %987 = arith.muli %986, %c64_i64 : i64
          %988 = arith.subi %919, %987 : i64
          %989 = arith.minsi %988, %c64_i64 : i64
          %990 = arith.cmpi sge, %920, %c1_i64 : i64
          %991 = arith.select %990, %989, %c0_i64 : i64
          %992 = arith.index_cast %991 : i64 to index
          %993 = arith.addi %992, %c15 : index
          %994 = arith.divui %993, %c16 : index
          %995 = arith.index_cast %994 : index to i64
          %996 = arith.cmpi ugt, %995, %c0_i64 : i64
          scf.if %996 {
            asccube.get_buf mte1, %971, 0 {cannir.buffer.generated_sync}
            %1325 = arith.trunci %995 : i64 to i32
            asccube.copy_l12l0b %980[%c4096], %927[%929], %c0_i32, %c4_i32, %c4_i32, %1325, %c4_i32, %c4_i32 {transpose = true} : memref<1xbf16, 4>, memref<1xbf16, 2>
            asccube.release_buf mte1, %971, 0 {cannir.buffer.generated_sync}
          }
          %997 = arith.cmpi eq, %960, %c6_i64 : i64
          %998 = arith.select %997, %938, %939 : i64
          %999 = arith.cmpi eq, %960, %c5_i64 : i64
          %1000 = arith.select %999, %942, %998 : i64
          %1001 = arith.cmpi eq, %960, %c4_i64 : i64
          %1002 = arith.select %1001, %945, %1000 : i64
          %1003 = arith.cmpi eq, %960, %c3_i64 : i64
          %1004 = arith.select %1003, %948, %1002 : i64
          %1005 = arith.cmpi eq, %960, %c2_i64 : i64
          %1006 = arith.select %1005, %951, %1004 : i64
          %1007 = arith.cmpi eq, %960, %c1_i64 : i64
          %1008 = arith.select %1007, %954, %1006 : i64
          %1009 = arith.cmpi eq, %960, %c0_i64 : i64
          %1010 = arith.select %1009, %957, %1008 : i64
          %1011 = arith.addi %960, %c1_i64 overflow<nsw> : i64
          %1012 = arith.remui %1011, %c8_i64 : i64
          %1013 = ascscalar.bind_local_ptr %1010 : memref<1xf32, 5>
          asccube.get_buf m, %876, 0 {cannir.buffer.generated_sync}
          asccube.get_buf m, %971, 0 {cannir.buffer.generated_sync}
          asccube.mmad %1013[%c0], %935, %980, %c64_i32, %c128_i32, %c64_i32, %c3_i32 {c_matrix_init_val = true, mmad_direction = 1 : i32} : memref<1xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
          asccube.release_buf m, %971, 0 {cannir.buffer.generated_sync}
          asccube.release_buf m, %876, 0 {cannir.buffer.generated_sync}
          scf.if %962 {
            %1325 = ascscalar.bitcast_memref %alloc_11 byte_offset = %c0_i64 : memref<16384xf32, 1> -> memref<?xf32, 1>
            asccube.copy_l0c2ub %1325[%c64], %1013[%c0], %c64_i32, %c64_i32, %c256_i32, %c64_i32, %c3_i32 sub_block_id %c0_i32 {dual_dst_ctl = 1 : i32, nz2nd_en = true, relu_en = false} : memref<?xf32, 1>, memref<1xf32, 5>
          } else {
            %1325 = ascscalar.bitcast_memref %alloc_11 byte_offset = %c0_i64 : memref<16384xf32, 1> -> memref<?xf32, 1>
            asccube.copy_l0c2ub %1325[%c8256], %1013[%c0], %c64_i32, %c64_i32, %c256_i32, %c64_i32, %c3_i32 sub_block_id %c0_i32 {dual_dst_ctl = 1 : i32, nz2nd_en = true, relu_en = false} : memref<?xf32, 1>, memref<1xf32, 5>
          }
          %1014 = arith.cmpi eq, %973, %c2_i64 : i64
          %1015 = arith.select %1014, %619, %620 : i64
          %1016 = arith.cmpi eq, %973, %c1_i64 : i64
          %1017 = arith.select %1016, %623, %1015 : i64
          %1018 = arith.cmpi eq, %973, %c0_i64 : i64
          %1019 = arith.select %1018, %626, %1017 : i64
          %1020 = arith.select %1014, %c9_i64, %c10_i64 : i64
          %1021 = arith.select %1016, %c8_i64, %1020 : i64
          %1022 = arith.select %1018, %c7_i64, %1021 : i64
          %1023 = arith.addi %973, %c1_i64 overflow<nsw> : i64
          %1024 = arith.remui %1023, %c4_i64 : i64
          %1025 = arith.minsi %897, %c2_i64 : i64
          %1026 = arith.muli %1025, %c64_i64 : i64
          %1027 = arith.subi %896, %1026 : i64
          %1028 = arith.minsi %1027, %c64_i64 : i64
          %1029 = arith.cmpi sge, %897, %c2_i64 : i64
          %1030 = arith.select %1029, %1028, %c0_i64 : i64
          %1031 = ascscalar.bind_local_ptr %1019 : memref<1xbf16, 4>
          %1032 = arith.index_cast %1030 : i64 to index
          %1033 = arith.addi %1032, %c15 : index
          %1034 = arith.divui %1033, %c16 : index
          %1035 = arith.index_cast %1034 : index to i64
          %1036 = arith.cmpi ugt, %1035, %c0_i64 : i64
          scf.if %1036 {
            asccube.get_buf mte1, %1022, 0 {cannir.buffer.generated_sync}
            %1325 = arith.trunci %1035 : i64 to i32
            asccube.copy_l12l0b %1031[%c0], %905[%907], %c0_i32, %c8_i32, %c4_i32, %1325, %c4_i32, %c4_i32 {transpose = true} : memref<1xbf16, 4>, memref<1xbf16, 2>
            asccube.release_buf mte1, %1022, 0 {cannir.buffer.generated_sync}
          }
          %1037 = arith.minsi %920, %c2_i64 : i64
          %1038 = arith.muli %1037, %c64_i64 : i64
          %1039 = arith.subi %919, %1038 : i64
          %1040 = arith.minsi %1039, %c64_i64 : i64
          %1041 = arith.cmpi sge, %920, %c2_i64 : i64
          %1042 = arith.select %1041, %1040, %c0_i64 : i64
          %1043 = arith.index_cast %1042 : i64 to index
          %1044 = arith.addi %1043, %c15 : index
          %1045 = arith.divui %1044, %c16 : index
          %1046 = arith.index_cast %1045 : index to i64
          %1047 = arith.cmpi ugt, %1046, %c0_i64 : i64
          scf.if %1047 {
            asccube.get_buf mte1, %1022, 0 {cannir.buffer.generated_sync}
            %1325 = arith.trunci %1046 : i64 to i32
            asccube.copy_l12l0b %1031[%c4096], %927[%929], %c0_i32, %c8_i32, %c4_i32, %1325, %c4_i32, %c4_i32 {transpose = true} : memref<1xbf16, 4>, memref<1xbf16, 2>
            asccube.release_buf mte1, %1022, 0 {cannir.buffer.generated_sync}
          }
          %1048 = arith.cmpi eq, %1012, %c6_i64 : i64
          %1049 = arith.select %1048, %938, %939 : i64
          %1050 = arith.cmpi eq, %1012, %c5_i64 : i64
          %1051 = arith.select %1050, %942, %1049 : i64
          %1052 = arith.cmpi eq, %1012, %c4_i64 : i64
          %1053 = arith.select %1052, %945, %1051 : i64
          %1054 = arith.cmpi eq, %1012, %c3_i64 : i64
          %1055 = arith.select %1054, %948, %1053 : i64
          %1056 = arith.cmpi eq, %1012, %c2_i64 : i64
          %1057 = arith.select %1056, %951, %1055 : i64
          %1058 = arith.cmpi eq, %1012, %c1_i64 : i64
          %1059 = arith.select %1058, %954, %1057 : i64
          %1060 = arith.cmpi eq, %1012, %c0_i64 : i64
          %1061 = arith.select %1060, %957, %1059 : i64
          %1062 = arith.addi %1012, %c1_i64 overflow<nsw> : i64
          %1063 = arith.remui %1062, %c8_i64 : i64
          %1064 = ascscalar.bind_local_ptr %1061 : memref<1xf32, 5>
          asccube.get_buf m, %876, 0 {cannir.buffer.generated_sync}
          asccube.get_buf m, %1022, 0 {cannir.buffer.generated_sync}
          asccube.mmad %1064[%c0], %935, %1031, %c64_i32, %c128_i32, %c64_i32, %c3_i32 {c_matrix_init_val = true, mmad_direction = 1 : i32} : memref<1xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
          asccube.release_buf m, %1022, 0 {cannir.buffer.generated_sync}
          asccube.release_buf m, %876, 0 {cannir.buffer.generated_sync}
          scf.if %962 {
            %1325 = ascscalar.bitcast_memref %alloc_11 byte_offset = %c0_i64 : memref<16384xf32, 1> -> memref<?xf32, 1>
            asccube.copy_l0c2ub %1325[%c128], %1064[%c0], %c64_i32, %c64_i32, %c256_i32, %c64_i32, %c3_i32 sub_block_id %c0_i32 {dual_dst_ctl = 1 : i32, nz2nd_en = true, relu_en = false} : memref<?xf32, 1>, memref<1xf32, 5>
          } else {
            %1325 = ascscalar.bitcast_memref %alloc_11 byte_offset = %c0_i64 : memref<16384xf32, 1> -> memref<?xf32, 1>
            asccube.copy_l0c2ub %1325[%c8320], %1064[%c0], %c64_i32, %c64_i32, %c256_i32, %c64_i32, %c3_i32 sub_block_id %c0_i32 {dual_dst_ctl = 1 : i32, nz2nd_en = true, relu_en = false} : memref<?xf32, 1>, memref<1xf32, 5>
          }
          %1065 = arith.cmpi eq, %1024, %c2_i64 : i64
          %1066 = arith.select %1065, %619, %620 : i64
          %1067 = arith.cmpi eq, %1024, %c1_i64 : i64
          %1068 = arith.select %1067, %623, %1066 : i64
          %1069 = arith.cmpi eq, %1024, %c0_i64 : i64
          %1070 = arith.select %1069, %626, %1068 : i64
          %1071 = arith.select %1065, %c9_i64, %c10_i64 : i64
          %1072 = arith.select %1067, %c8_i64, %1071 : i64
          %1073 = arith.select %1069, %c7_i64, %1072 : i64
          %1074 = arith.addi %1024, %c1_i64 overflow<nsw> : i64
          %1075 = arith.remui %1074, %c4_i64 : i64
          %1076 = arith.minsi %897, %c3_i64 : i64
          %1077 = arith.muli %1076, %c64_i64 : i64
          %1078 = arith.subi %896, %1077 : i64
          %1079 = arith.minsi %1078, %c64_i64 : i64
          %1080 = arith.cmpi sge, %897, %c3_i64 : i64
          %1081 = arith.select %1080, %1079, %c0_i64 : i64
          %1082 = ascscalar.bind_local_ptr %1070 : memref<1xbf16, 4>
          %1083 = arith.index_cast %1081 : i64 to index
          %1084 = arith.addi %1083, %c15 : index
          %1085 = arith.divui %1084, %c16 : index
          %1086 = arith.index_cast %1085 : index to i64
          %1087 = arith.cmpi ugt, %1086, %c0_i64 : i64
          scf.if %1087 {
            asccube.get_buf mte1, %1073, 0 {cannir.buffer.generated_sync}
            %1325 = arith.trunci %1086 : i64 to i32
            asccube.copy_l12l0b %1082[%c0], %905[%907], %c0_i32, %c12_i32, %c4_i32, %1325, %c4_i32, %c4_i32 {transpose = true} : memref<1xbf16, 4>, memref<1xbf16, 2>
            asccube.release_buf mte1, %1073, 0 {cannir.buffer.generated_sync}
          }
          %1088 = arith.minsi %920, %c3_i64 : i64
          %1089 = arith.muli %1088, %c64_i64 : i64
          %1090 = arith.subi %919, %1089 : i64
          %1091 = arith.minsi %1090, %c64_i64 : i64
          %1092 = arith.cmpi sge, %920, %c3_i64 : i64
          %1093 = arith.select %1092, %1091, %c0_i64 : i64
          %1094 = arith.index_cast %1093 : i64 to index
          %1095 = arith.addi %1094, %c15 : index
          %1096 = arith.divui %1095, %c16 : index
          %1097 = arith.index_cast %1096 : index to i64
          %1098 = arith.cmpi ugt, %1097, %c0_i64 : i64
          scf.if %1098 {
            asccube.get_buf mte1, %1073, 0 {cannir.buffer.generated_sync}
            %1325 = arith.trunci %1097 : i64 to i32
            asccube.copy_l12l0b %1082[%c4096], %927[%929], %c0_i32, %c12_i32, %c4_i32, %1325, %c4_i32, %c4_i32 {transpose = true} : memref<1xbf16, 4>, memref<1xbf16, 2>
            asccube.release_buf mte1, %1073, 0 {cannir.buffer.generated_sync}
          }
          %1099 = arith.cmpi eq, %1063, %c6_i64 : i64
          %1100 = arith.select %1099, %938, %939 : i64
          %1101 = arith.cmpi eq, %1063, %c5_i64 : i64
          %1102 = arith.select %1101, %942, %1100 : i64
          %1103 = arith.cmpi eq, %1063, %c4_i64 : i64
          %1104 = arith.select %1103, %945, %1102 : i64
          %1105 = arith.cmpi eq, %1063, %c3_i64 : i64
          %1106 = arith.select %1105, %948, %1104 : i64
          %1107 = arith.cmpi eq, %1063, %c2_i64 : i64
          %1108 = arith.select %1107, %951, %1106 : i64
          %1109 = arith.cmpi eq, %1063, %c1_i64 : i64
          %1110 = arith.select %1109, %954, %1108 : i64
          %1111 = arith.cmpi eq, %1063, %c0_i64 : i64
          %1112 = arith.select %1111, %957, %1110 : i64
          %1113 = arith.addi %1063, %c1_i64 overflow<nsw> : i64
          %1114 = arith.remui %1113, %c8_i64 : i64
          %1115 = ascscalar.bind_local_ptr %1112 : memref<1xf32, 5>
          asccube.get_buf m, %876, 0 {cannir.buffer.generated_sync}
          asccube.get_buf m, %1073, 0 {cannir.buffer.generated_sync}
          asccube.mmad %1115[%c0], %935, %1082, %c64_i32, %c128_i32, %c64_i32, %c3_i32 {c_matrix_init_val = true, mmad_direction = 1 : i32} : memref<1xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
          asccube.release_buf m, %1073, 0 {cannir.buffer.generated_sync}
          asccube.release_buf m, %876, 0 {cannir.buffer.generated_sync}
          scf.if %962 {
            %1325 = ascscalar.bitcast_memref %alloc_11 byte_offset = %c0_i64 : memref<16384xf32, 1> -> memref<?xf32, 1>
            asccube.copy_l0c2ub %1325[%c192], %1115[%c0], %c64_i32, %c64_i32, %c256_i32, %c64_i32, %c3_i32 sub_block_id %c0_i32 {dual_dst_ctl = 1 : i32, nz2nd_en = true, relu_en = false} : memref<?xf32, 1>, memref<1xf32, 5>
            asccube.sync_intra_arrive fixpipe, %c1_i64 {cannir.sync.banks = 3 : i32}
            asccube.sync_intra_arrive fixpipe, %c17_i64 {cannir.sync.banks = 3 : i32}
          } else {
            %1325 = ascscalar.bitcast_memref %alloc_11 byte_offset = %c0_i64 : memref<16384xf32, 1> -> memref<?xf32, 1>
            asccube.copy_l0c2ub %1325[%c8384], %1115[%c0], %c64_i32, %c64_i32, %c256_i32, %c64_i32, %c3_i32 sub_block_id %c0_i32 {dual_dst_ctl = 1 : i32, nz2nd_en = true, relu_en = false} : memref<?xf32, 1>, memref<1xf32, 5>
            asccube.sync_intra_arrive fixpipe, %c1_i64 {cannir.sync.banks = 3 : i32}
            asccube.sync_intra_arrive fixpipe, %c17_i64 {cannir.sync.banks = 3 : i32}
          }
          %1116 = arith.cmpi eq, %1075, %c2_i64 : i64
          %1117 = arith.select %1116, %619, %620 : i64
          %1118 = arith.cmpi eq, %1075, %c1_i64 : i64
          %1119 = arith.select %1118, %623, %1117 : i64
          %1120 = arith.cmpi eq, %1075, %c0_i64 : i64
          %1121 = arith.select %1120, %626, %1119 : i64
          %1122 = arith.select %1116, %c9_i64, %c10_i64 : i64
          %1123 = arith.select %1118, %c8_i64, %1122 : i64
          %1124 = arith.select %1120, %c7_i64, %1123 : i64
          %1125 = arith.addi %1075, %c1_i64 overflow<nsw> : i64
          %1126 = arith.remui %1125, %c4_i64 : i64
          %1127 = arith.minsi %897, %c4_i64 : i64
          %1128 = arith.muli %1127, %c64_i64 : i64
          %1129 = arith.subi %896, %1128 : i64
          %1130 = arith.minsi %1129, %c64_i64 : i64
          %1131 = arith.cmpi sge, %897, %c4_i64 : i64
          %1132 = arith.select %1131, %1130, %c0_i64 : i64
          %1133 = ascscalar.bind_local_ptr %1121 : memref<1xbf16, 4>
          %1134 = arith.index_cast %1132 : i64 to index
          %1135 = arith.addi %1134, %c15 : index
          %1136 = arith.divui %1135, %c16 : index
          %1137 = arith.index_cast %1136 : index to i64
          %1138 = arith.cmpi ugt, %1137, %c0_i64 : i64
          scf.if %1138 {
            asccube.get_buf mte1, %1124, 0 {cannir.buffer.generated_sync}
            %1325 = arith.trunci %1137 : i64 to i32
            asccube.copy_l12l0b %1133[%c0], %905[%907], %c0_i32, %c16_i32, %c4_i32, %1325, %c4_i32, %c4_i32 {transpose = true} : memref<1xbf16, 4>, memref<1xbf16, 2>
            asccube.release_buf mte1, %1124, 0 {cannir.buffer.generated_sync}
          }
          %1139 = arith.minsi %920, %c4_i64 : i64
          %1140 = arith.muli %1139, %c64_i64 : i64
          %1141 = arith.subi %919, %1140 : i64
          %1142 = arith.minsi %1141, %c64_i64 : i64
          %1143 = arith.cmpi sge, %920, %c4_i64 : i64
          %1144 = arith.select %1143, %1142, %c0_i64 : i64
          %1145 = arith.index_cast %1144 : i64 to index
          %1146 = arith.addi %1145, %c15 : index
          %1147 = arith.divui %1146, %c16 : index
          %1148 = arith.index_cast %1147 : index to i64
          %1149 = arith.cmpi ugt, %1148, %c0_i64 : i64
          scf.if %1149 {
            asccube.get_buf mte1, %1124, 0 {cannir.buffer.generated_sync}
            %1325 = arith.trunci %1148 : i64 to i32
            asccube.copy_l12l0b %1133[%c4096], %927[%929], %c0_i32, %c16_i32, %c4_i32, %1325, %c4_i32, %c4_i32 {transpose = true} : memref<1xbf16, 4>, memref<1xbf16, 2>
            asccube.release_buf mte1, %1124, 0 {cannir.buffer.generated_sync}
          }
          %1150 = arith.cmpi eq, %1114, %c6_i64 : i64
          %1151 = arith.select %1150, %938, %939 : i64
          %1152 = arith.cmpi eq, %1114, %c5_i64 : i64
          %1153 = arith.select %1152, %942, %1151 : i64
          %1154 = arith.cmpi eq, %1114, %c4_i64 : i64
          %1155 = arith.select %1154, %945, %1153 : i64
          %1156 = arith.cmpi eq, %1114, %c3_i64 : i64
          %1157 = arith.select %1156, %948, %1155 : i64
          %1158 = arith.cmpi eq, %1114, %c2_i64 : i64
          %1159 = arith.select %1158, %951, %1157 : i64
          %1160 = arith.cmpi eq, %1114, %c1_i64 : i64
          %1161 = arith.select %1160, %954, %1159 : i64
          %1162 = arith.cmpi eq, %1114, %c0_i64 : i64
          %1163 = arith.select %1162, %957, %1161 : i64
          %1164 = arith.addi %1114, %c1_i64 overflow<nsw> : i64
          %1165 = arith.remui %1164, %c8_i64 : i64
          %1166 = ascscalar.bind_local_ptr %1163 : memref<1xf32, 5>
          asccube.get_buf m, %876, 0 {cannir.buffer.generated_sync}
          asccube.get_buf m, %1124, 0 {cannir.buffer.generated_sync}
          asccube.mmad %1166[%c0], %935, %1133, %c64_i32, %c128_i32, %c64_i32, %c3_i32 {c_matrix_init_val = true, mmad_direction = 1 : i32} : memref<1xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
          asccube.release_buf m, %1124, 0 {cannir.buffer.generated_sync}
          asccube.release_buf m, %876, 0 {cannir.buffer.generated_sync}
          scf.if %962 {
            %1325 = ascscalar.bitcast_memref %alloc_12 byte_offset = %c0_i64 : memref<16384xf32, 1> -> memref<?xf32, 1>
            asccube.sync_intra_wait fixpipe, %c2_i64 {cannir.sync.banks = 3 : i32}
            asccube.sync_intra_wait fixpipe, %c18_i64 {cannir.sync.banks = 3 : i32}
            asccube.copy_l0c2ub %1325[%c0], %1166[%c0], %c64_i32, %c64_i32, %c256_i32, %c64_i32, %c3_i32 sub_block_id %c0_i32 {dual_dst_ctl = 1 : i32, nz2nd_en = true, relu_en = false} : memref<?xf32, 1>, memref<1xf32, 5>
          } else {
            %1325 = ascscalar.bitcast_memref %alloc_12 byte_offset = %c0_i64 : memref<16384xf32, 1> -> memref<?xf32, 1>
            asccube.sync_intra_wait fixpipe, %c2_i64 {cannir.sync.banks = 3 : i32}
            asccube.sync_intra_wait fixpipe, %c18_i64 {cannir.sync.banks = 3 : i32}
            asccube.copy_l0c2ub %1325[%c8192], %1166[%c0], %c64_i32, %c64_i32, %c256_i32, %c64_i32, %c3_i32 sub_block_id %c0_i32 {dual_dst_ctl = 1 : i32, nz2nd_en = true, relu_en = false} : memref<?xf32, 1>, memref<1xf32, 5>
          }
          %1167 = arith.cmpi eq, %1126, %c2_i64 : i64
          %1168 = arith.select %1167, %619, %620 : i64
          %1169 = arith.cmpi eq, %1126, %c1_i64 : i64
          %1170 = arith.select %1169, %623, %1168 : i64
          %1171 = arith.cmpi eq, %1126, %c0_i64 : i64
          %1172 = arith.select %1171, %626, %1170 : i64
          %1173 = arith.select %1167, %c9_i64, %c10_i64 : i64
          %1174 = arith.select %1169, %c8_i64, %1173 : i64
          %1175 = arith.select %1171, %c7_i64, %1174 : i64
          %1176 = arith.addi %1126, %c1_i64 overflow<nsw> : i64
          %1177 = arith.remui %1176, %c4_i64 : i64
          %1178 = arith.minsi %897, %c5_i64 : i64
          %1179 = arith.muli %1178, %c64_i64 : i64
          %1180 = arith.subi %896, %1179 : i64
          %1181 = arith.minsi %1180, %c64_i64 : i64
          %1182 = arith.cmpi sge, %897, %c5_i64 : i64
          %1183 = arith.select %1182, %1181, %c0_i64 : i64
          %1184 = ascscalar.bind_local_ptr %1172 : memref<1xbf16, 4>
          %1185 = arith.index_cast %1183 : i64 to index
          %1186 = arith.addi %1185, %c15 : index
          %1187 = arith.divui %1186, %c16 : index
          %1188 = arith.index_cast %1187 : index to i64
          %1189 = arith.cmpi ugt, %1188, %c0_i64 : i64
          scf.if %1189 {
            asccube.get_buf mte1, %1175, 0 {cannir.buffer.generated_sync}
            %1325 = arith.trunci %1188 : i64 to i32
            asccube.copy_l12l0b %1184[%c0], %905[%907], %c0_i32, %c20_i32, %c4_i32, %1325, %c4_i32, %c4_i32 {transpose = true} : memref<1xbf16, 4>, memref<1xbf16, 2>
            asccube.release_buf mte1, %1175, 0 {cannir.buffer.generated_sync}
          }
          %1190 = arith.minsi %920, %c5_i64 : i64
          %1191 = arith.muli %1190, %c64_i64 : i64
          %1192 = arith.subi %919, %1191 : i64
          %1193 = arith.minsi %1192, %c64_i64 : i64
          %1194 = arith.cmpi sge, %920, %c5_i64 : i64
          %1195 = arith.select %1194, %1193, %c0_i64 : i64
          %1196 = arith.index_cast %1195 : i64 to index
          %1197 = arith.addi %1196, %c15 : index
          %1198 = arith.divui %1197, %c16 : index
          %1199 = arith.index_cast %1198 : index to i64
          %1200 = arith.cmpi ugt, %1199, %c0_i64 : i64
          scf.if %1200 {
            asccube.get_buf mte1, %1175, 0 {cannir.buffer.generated_sync}
            %1325 = arith.trunci %1199 : i64 to i32
            asccube.copy_l12l0b %1184[%c4096], %927[%929], %c0_i32, %c20_i32, %c4_i32, %1325, %c4_i32, %c4_i32 {transpose = true} : memref<1xbf16, 4>, memref<1xbf16, 2>
            asccube.release_buf mte1, %1175, 0 {cannir.buffer.generated_sync}
          }
          %1201 = arith.cmpi eq, %1165, %c6_i64 : i64
          %1202 = arith.select %1201, %938, %939 : i64
          %1203 = arith.cmpi eq, %1165, %c5_i64 : i64
          %1204 = arith.select %1203, %942, %1202 : i64
          %1205 = arith.cmpi eq, %1165, %c4_i64 : i64
          %1206 = arith.select %1205, %945, %1204 : i64
          %1207 = arith.cmpi eq, %1165, %c3_i64 : i64
          %1208 = arith.select %1207, %948, %1206 : i64
          %1209 = arith.cmpi eq, %1165, %c2_i64 : i64
          %1210 = arith.select %1209, %951, %1208 : i64
          %1211 = arith.cmpi eq, %1165, %c1_i64 : i64
          %1212 = arith.select %1211, %954, %1210 : i64
          %1213 = arith.cmpi eq, %1165, %c0_i64 : i64
          %1214 = arith.select %1213, %957, %1212 : i64
          %1215 = arith.addi %1165, %c1_i64 overflow<nsw> : i64
          %1216 = arith.remui %1215, %c8_i64 : i64
          %1217 = ascscalar.bind_local_ptr %1214 : memref<1xf32, 5>
          asccube.get_buf m, %876, 0 {cannir.buffer.generated_sync}
          asccube.get_buf m, %1175, 0 {cannir.buffer.generated_sync}
          asccube.mmad %1217[%c0], %935, %1184, %c64_i32, %c128_i32, %c64_i32, %c3_i32 {c_matrix_init_val = true, mmad_direction = 1 : i32} : memref<1xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
          asccube.release_buf m, %1175, 0 {cannir.buffer.generated_sync}
          asccube.release_buf m, %876, 0 {cannir.buffer.generated_sync}
          scf.if %962 {
            %1325 = ascscalar.bitcast_memref %alloc_12 byte_offset = %c0_i64 : memref<16384xf32, 1> -> memref<?xf32, 1>
            asccube.copy_l0c2ub %1325[%c64], %1217[%c0], %c64_i32, %c64_i32, %c256_i32, %c64_i32, %c3_i32 sub_block_id %c0_i32 {dual_dst_ctl = 1 : i32, nz2nd_en = true, relu_en = false} : memref<?xf32, 1>, memref<1xf32, 5>
          } else {
            %1325 = ascscalar.bitcast_memref %alloc_12 byte_offset = %c0_i64 : memref<16384xf32, 1> -> memref<?xf32, 1>
            asccube.copy_l0c2ub %1325[%c8256], %1217[%c0], %c64_i32, %c64_i32, %c256_i32, %c64_i32, %c3_i32 sub_block_id %c0_i32 {dual_dst_ctl = 1 : i32, nz2nd_en = true, relu_en = false} : memref<?xf32, 1>, memref<1xf32, 5>
          }
          %1218 = arith.cmpi eq, %1177, %c2_i64 : i64
          %1219 = arith.select %1218, %619, %620 : i64
          %1220 = arith.cmpi eq, %1177, %c1_i64 : i64
          %1221 = arith.select %1220, %623, %1219 : i64
          %1222 = arith.cmpi eq, %1177, %c0_i64 : i64
          %1223 = arith.select %1222, %626, %1221 : i64
          %1224 = arith.select %1218, %c9_i64, %c10_i64 : i64
          %1225 = arith.select %1220, %c8_i64, %1224 : i64
          %1226 = arith.select %1222, %c7_i64, %1225 : i64
          %1227 = arith.addi %1177, %c1_i64 overflow<nsw> : i64
          %1228 = arith.remui %1227, %c4_i64 : i64
          %1229 = arith.minsi %897, %c6_i64 : i64
          %1230 = arith.muli %1229, %c64_i64 : i64
          %1231 = arith.subi %896, %1230 : i64
          %1232 = arith.minsi %1231, %c64_i64 : i64
          %1233 = arith.cmpi sge, %897, %c6_i64 : i64
          %1234 = arith.select %1233, %1232, %c0_i64 : i64
          %1235 = ascscalar.bind_local_ptr %1223 : memref<1xbf16, 4>
          %1236 = arith.index_cast %1234 : i64 to index
          %1237 = arith.addi %1236, %c15 : index
          %1238 = arith.divui %1237, %c16 : index
          %1239 = arith.index_cast %1238 : index to i64
          %1240 = arith.cmpi ugt, %1239, %c0_i64 : i64
          scf.if %1240 {
            asccube.get_buf mte1, %1226, 0 {cannir.buffer.generated_sync}
            %1325 = arith.trunci %1239 : i64 to i32
            asccube.copy_l12l0b %1235[%c0], %905[%907], %c0_i32, %c24_i32, %c4_i32, %1325, %c4_i32, %c4_i32 {transpose = true} : memref<1xbf16, 4>, memref<1xbf16, 2>
            asccube.release_buf mte1, %1226, 0 {cannir.buffer.generated_sync}
          }
          %1241 = arith.minsi %920, %c6_i64 : i64
          %1242 = arith.muli %1241, %c64_i64 : i64
          %1243 = arith.subi %919, %1242 : i64
          %1244 = arith.minsi %1243, %c64_i64 : i64
          %1245 = arith.cmpi sge, %920, %c6_i64 : i64
          %1246 = arith.select %1245, %1244, %c0_i64 : i64
          %1247 = arith.index_cast %1246 : i64 to index
          %1248 = arith.addi %1247, %c15 : index
          %1249 = arith.divui %1248, %c16 : index
          %1250 = arith.index_cast %1249 : index to i64
          %1251 = arith.cmpi ugt, %1250, %c0_i64 : i64
          scf.if %1251 {
            asccube.get_buf mte1, %1226, 0 {cannir.buffer.generated_sync}
            %1325 = arith.trunci %1250 : i64 to i32
            asccube.copy_l12l0b %1235[%c4096], %927[%929], %c0_i32, %c24_i32, %c4_i32, %1325, %c4_i32, %c4_i32 {transpose = true} : memref<1xbf16, 4>, memref<1xbf16, 2>
            asccube.release_buf mte1, %1226, 0 {cannir.buffer.generated_sync}
          }
          %1252 = arith.cmpi eq, %1216, %c6_i64 : i64
          %1253 = arith.select %1252, %938, %939 : i64
          %1254 = arith.cmpi eq, %1216, %c5_i64 : i64
          %1255 = arith.select %1254, %942, %1253 : i64
          %1256 = arith.cmpi eq, %1216, %c4_i64 : i64
          %1257 = arith.select %1256, %945, %1255 : i64
          %1258 = arith.cmpi eq, %1216, %c3_i64 : i64
          %1259 = arith.select %1258, %948, %1257 : i64
          %1260 = arith.cmpi eq, %1216, %c2_i64 : i64
          %1261 = arith.select %1260, %951, %1259 : i64
          %1262 = arith.cmpi eq, %1216, %c1_i64 : i64
          %1263 = arith.select %1262, %954, %1261 : i64
          %1264 = arith.cmpi eq, %1216, %c0_i64 : i64
          %1265 = arith.select %1264, %957, %1263 : i64
          %1266 = arith.addi %1216, %c1_i64 overflow<nsw> : i64
          %1267 = arith.remui %1266, %c8_i64 : i64
          %1268 = ascscalar.bind_local_ptr %1265 : memref<1xf32, 5>
          asccube.get_buf m, %876, 0 {cannir.buffer.generated_sync}
          asccube.get_buf m, %1226, 0 {cannir.buffer.generated_sync}
          asccube.mmad %1268[%c0], %935, %1235, %c64_i32, %c128_i32, %c64_i32, %c3_i32 {c_matrix_init_val = true, mmad_direction = 1 : i32} : memref<1xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
          asccube.release_buf m, %1226, 0 {cannir.buffer.generated_sync}
          asccube.release_buf m, %876, 0 {cannir.buffer.generated_sync}
          scf.if %962 {
            %1325 = ascscalar.bitcast_memref %alloc_12 byte_offset = %c0_i64 : memref<16384xf32, 1> -> memref<?xf32, 1>
            asccube.copy_l0c2ub %1325[%c128], %1268[%c0], %c64_i32, %c64_i32, %c256_i32, %c64_i32, %c3_i32 sub_block_id %c0_i32 {dual_dst_ctl = 1 : i32, nz2nd_en = true, relu_en = false} : memref<?xf32, 1>, memref<1xf32, 5>
          } else {
            %1325 = ascscalar.bitcast_memref %alloc_12 byte_offset = %c0_i64 : memref<16384xf32, 1> -> memref<?xf32, 1>
            asccube.copy_l0c2ub %1325[%c8320], %1268[%c0], %c64_i32, %c64_i32, %c256_i32, %c64_i32, %c3_i32 sub_block_id %c0_i32 {dual_dst_ctl = 1 : i32, nz2nd_en = true, relu_en = false} : memref<?xf32, 1>, memref<1xf32, 5>
          }
          %1269 = arith.cmpi eq, %1228, %c2_i64 : i64
          %1270 = arith.select %1269, %619, %620 : i64
          %1271 = arith.cmpi eq, %1228, %c1_i64 : i64
          %1272 = arith.select %1271, %623, %1270 : i64
          %1273 = arith.cmpi eq, %1228, %c0_i64 : i64
          %1274 = arith.select %1273, %626, %1272 : i64
          %1275 = arith.select %1269, %c9_i64, %c10_i64 : i64
          %1276 = arith.select %1271, %c8_i64, %1275 : i64
          %1277 = arith.select %1273, %c7_i64, %1276 : i64
          %1278 = arith.minsi %897, %c7_i64 : i64
          %1279 = arith.muli %1278, %c64_i64 : i64
          %1280 = arith.subi %896, %1279 : i64
          %1281 = arith.minsi %1280, %c64_i64 : i64
          %1282 = arith.cmpi sge, %897, %c7_i64 : i64
          %1283 = arith.select %1282, %1281, %c0_i64 : i64
          %1284 = ascscalar.bind_local_ptr %1274 : memref<1xbf16, 4>
          %1285 = arith.index_cast %1283 : i64 to index
          %1286 = arith.addi %1285, %c15 : index
          %1287 = arith.divui %1286, %c16 : index
          %1288 = arith.index_cast %1287 : index to i64
          %1289 = arith.cmpi ugt, %1288, %c0_i64 : i64
          scf.if %1289 {
            asccube.get_buf mte1, %1277, 0 {cannir.buffer.generated_sync}
            %1325 = arith.trunci %1288 : i64 to i32
            asccube.copy_l12l0b %1284[%c0], %905[%907], %c0_i32, %c28_i32, %c4_i32, %1325, %c4_i32, %c4_i32 {transpose = true} : memref<1xbf16, 4>, memref<1xbf16, 2>
            asccube.release_buf mte1, %1277, 0 {cannir.buffer.generated_sync}
          }
          %1290 = arith.cmpi eq, %arg92, %c18_i64 : i64
          %1291 = arith.select %1290, %c4_i64, %c3_i64 : i64
          %1292 = arith.cmpi eq, %arg92, %c19_i64 : i64
          %1293 = arith.select %1292, %c5_i64, %1291 : i64
          asccube.sync_intra_arrive mte1, %1293 {cannir.sync.banks = 1 : i32}
          %1294 = arith.minsi %920, %c7_i64 : i64
          %1295 = arith.muli %1294, %c64_i64 : i64
          %1296 = arith.subi %919, %1295 : i64
          %1297 = arith.minsi %1296, %c64_i64 : i64
          %1298 = arith.cmpi sge, %920, %c7_i64 : i64
          %1299 = arith.select %1298, %1297, %c0_i64 : i64
          %1300 = arith.index_cast %1299 : i64 to index
          %1301 = arith.addi %1300, %c15 : index
          %1302 = arith.divui %1301, %c16 : index
          %1303 = arith.index_cast %1302 : index to i64
          %1304 = arith.cmpi ugt, %1303, %c0_i64 : i64
          scf.if %1304 {
            asccube.get_buf mte1, %1277, 0 {cannir.buffer.generated_sync}
            %1325 = arith.trunci %1303 : i64 to i32
            asccube.copy_l12l0b %1284[%c4096], %927[%929], %c0_i32, %c28_i32, %c4_i32, %1325, %c4_i32, %c4_i32 {transpose = true} : memref<1xbf16, 4>, memref<1xbf16, 2>
            asccube.release_buf mte1, %1277, 0 {cannir.buffer.generated_sync}
          }
          %1305 = arith.cmpi eq, %arg93, %c22_i64 : i64
          %1306 = arith.select %1305, %c8_i64, %c7_i64 : i64
          %1307 = arith.cmpi eq, %arg93, %c23_i64 : i64
          %1308 = arith.select %1307, %c9_i64, %1306 : i64
          %1309 = arith.addi %1308, %c16_i64 overflow<nsw> : i64
          asccube.sync_intra_arrive mte1, %1309 {cannir.sync.banks = 2 : i32}
          %1310 = arith.cmpi eq, %1267, %c6_i64 : i64
          %1311 = arith.select %1310, %938, %939 : i64
          %1312 = arith.cmpi eq, %1267, %c5_i64 : i64
          %1313 = arith.select %1312, %942, %1311 : i64
          %1314 = arith.cmpi eq, %1267, %c4_i64 : i64
          %1315 = arith.select %1314, %945, %1313 : i64
          %1316 = arith.cmpi eq, %1267, %c3_i64 : i64
          %1317 = arith.select %1316, %948, %1315 : i64
          %1318 = arith.cmpi eq, %1267, %c2_i64 : i64
          %1319 = arith.select %1318, %951, %1317 : i64
          %1320 = arith.cmpi eq, %1267, %c1_i64 : i64
          %1321 = arith.select %1320, %954, %1319 : i64
          %1322 = arith.cmpi eq, %1267, %c0_i64 : i64
          %1323 = arith.select %1322, %957, %1321 : i64
          %1324 = ascscalar.bind_local_ptr %1323 : memref<1xf32, 5>
          asccube.get_buf m, %876, 0 {cannir.buffer.generated_sync}
          asccube.get_buf m, %1277, 0 {cannir.buffer.generated_sync}
          asccube.mmad %1324[%c0], %935, %1284, %c64_i32, %c128_i32, %c64_i32, %c3_i32 {c_matrix_init_val = true, mmad_direction = 1 : i32} : memref<1xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
          asccube.release_buf m, %1277, 0 {cannir.buffer.generated_sync}
          asccube.release_buf m, %876, 0 {cannir.buffer.generated_sync}
          scf.if %962 {
            %1325 = ascscalar.bitcast_memref %alloc_12 byte_offset = %c0_i64 : memref<16384xf32, 1> -> memref<?xf32, 1>
            asccube.copy_l0c2ub %1325[%c192], %1324[%c0], %c64_i32, %c64_i32, %c256_i32, %c64_i32, %c3_i32 sub_block_id %c0_i32 {dual_dst_ctl = 1 : i32, nz2nd_en = true, relu_en = false} : memref<?xf32, 1>, memref<1xf32, 5>
            asccube.sync_intra_arrive fixpipe, %c2_i64 {cannir.sync.banks = 3 : i32}
            asccube.sync_intra_arrive fixpipe, %c18_i64 {cannir.sync.banks = 3 : i32}
          } else {
            %1325 = ascscalar.bitcast_memref %alloc_12 byte_offset = %c0_i64 : memref<16384xf32, 1> -> memref<?xf32, 1>
            asccube.copy_l0c2ub %1325[%c8384], %1324[%c0], %c64_i32, %c64_i32, %c256_i32, %c64_i32, %c3_i32 sub_block_id %c0_i32 {dual_dst_ctl = 1 : i32, nz2nd_en = true, relu_en = false} : memref<?xf32, 1>, memref<1xf32, 5>
            asccube.sync_intra_arrive fixpipe, %c2_i64 {cannir.sync.banks = 3 : i32}
            asccube.sync_intra_arrive fixpipe, %c18_i64 {cannir.sync.banks = 3 : i32}
          }
          scf.yield %568, %c0_i64, %c32768_i64, %c1_i64, %c32768_i64, %c1_i64, %568, %c0_i64, %568, %c0_i64, %576, %c0_i64, %c32768_i64, %c1_i64, %c32768_i64, %c1_i64, %576, %c0_i64, %576, %c0_i64, %572, %878, %570, %578 : i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64
        } {cannir.view_payload}
        %78 = arith.select %63, %c0_i64, %77#1 : i64
        %79 = arith.select %63, %c32768_i64, %77#2 : i64
        %80 = arith.select %63, %c1_i64, %77#3 : i64
        %81 = arith.select %63, %c32768_i64, %77#4 : i64
        %82 = arith.select %63, %c1_i64, %77#5 : i64
        %83 = arith.select %63, %c0_i64, %77#7 : i64
        %84 = arith.select %63, %c0_i64, %77#9 : i64
        %85 = arith.select %63, %c0_i64, %77#11 : i64
        %86 = arith.select %63, %c32768_i64, %77#12 : i64
        %87 = arith.select %63, %c1_i64, %77#13 : i64
        %88 = arith.select %63, %c32768_i64, %77#14 : i64
        %89 = arith.select %63, %c1_i64, %77#15 : i64
        %90 = arith.select %63, %c0_i64, %77#17 : i64
        %91 = arith.select %63, %c0_i64, %77#19 : i64
        %92:11 = scf.if %63 -> (i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) {
          %563 = cannir.datastruct_get %arg22["num_heads_q"] : <"HighPerfSmlaConstInfo", ["query_row_count", "ori_kv_capacity", "cmp_kv_capacity", "cmp_sparse_index_width", "has_cmp_topk_length", "num_heads_q", "ori_sparse_index_width", "ori_win_left", "ori_win_right", "has_ori_topk_length", "has_seqused_ori_kv", "has_seqused_cmp_kv", "has_cmp", "batch_mode", "batch_count", "query_rows_per_batch", "cube_core_count", "has_seqused_q", "ori_mask_mode", "cmp_mask_mode", "cmp_ratio", "is_ori_dense", "kv_mode", "ori_page_size", "cmp_page_size", "ori_page_stride_rows", "cmp_page_stride_rows", "ori_block_table_width", "cmp_block_table_width", "ori_page_size_reciprocal", "cmp_page_size_reciprocal", "ori_page_size_log2", "cmp_page_size_log2", "ori_page_size_is_power_of_two", "cmp_page_size_is_power_of_two", "softmax_scale"], [i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ui32, ui32, i32, i32, i32, i32, f32], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140], 144, 4>
          %564 = arith.extsi %563 : i32 to i64
          %565 = arith.muli %62#0, %564 : i64
          %566 = arith.muli %565, %c512_i64 : i64
          %567 = arith.divui %566, %c16_i64 : i64
          %568 = arith.muli %567, %c16_i64 : i64
          %569 = arith.index_cast %568 : i64 to index
          %570 = arith.addi %569, %c384 : index
          asccube.get_buf mte2, %c2_i64, 0 {cannir.buffer.generated_sync}
          asccube.copy_gm2l1 %36[%c0], %alloc[%570], %c64_i32, %c128_i32, %c1024_i64, %c1_i32, %c0_i64, %c0_i64, %c64_i32 l2_cache_ctl %c4_i32 {nd2nz = true} : memref<?xbf16, 2>, memref<1xbf16>
          asccube.release_buf mte2, %c2_i64, 0 {cannir.buffer.generated_sync}
          %571 = arith.cmpi eq, %77#20, %c1_i64 : i64
          %572 = ascscalar.memref_address %alloc_27 : memref<32768xbf16, 2>
          %573 = ascscalar.memref_address %alloc_26 : memref<32768xbf16, 2>
          %574 = arith.select %571, %572, %573 : i64
          %575 = arith.cmpi eq, %77#20, %c0_i64 : i64
          %576 = arith.select %575, %53, %574 : i64
          %577 = arith.select %571, %c18_i64, %c19_i64 : i64
          %578 = arith.select %575, %c17_i64, %577 : i64
          %579 = arith.addi %77#20, %c1_i64 overflow<nsw> : i64
          %580 = arith.remui %579, %c3_i64 : i64
          %581 = ascscalar.memref_address %alloc_24 : memref<32768xbf16, 2>
          %582 = ascscalar.memref_address %alloc_23 : memref<32768xbf16, 2>
          %583 = arith.select %571, %581, %582 : i64
          %584 = arith.select %575, %54, %583 : i64
          %585 = arith.select %571, %c22_i64, %c23_i64 : i64
          %586 = arith.select %575, %c21_i64, %585 : i64
          asccube.set_mmad_direction m
          %587 = arith.cmpi eq, %77#21, %c2_i64 : i64
          %588 = ascscalar.memref_address %alloc_18 : memref<8192xbf16, 3>
          %589 = ascscalar.memref_address %alloc_20 : memref<8192xbf16, 3>
          %590 = arith.select %587, %588, %589 : i64
          %591 = arith.cmpi eq, %77#21, %c1_i64 : i64
          %592 = ascscalar.memref_address %alloc_16 : memref<8192xbf16, 3>
          %593 = arith.select %591, %592, %590 : i64
          %594 = arith.cmpi eq, %77#21, %c0_i64 : i64
          %595 = ascscalar.memref_address %alloc_14 : memref<8192xbf16, 3>
          %596 = arith.select %594, %595, %593 : i64
          %597 = arith.select %587, %c5_i64, %c6_i64 : i64
          %598 = arith.select %591, %c4_i64, %597 : i64
          %599 = arith.select %594, %c3_i64, %598 : i64
          %600 = arith.addi %77#21, %c1_i64 overflow<nsw> : i64
          %601 = arith.remui %600, %c4_i64 : i64
          %602 = arith.muli %65, %66 overflow<nsw> : i64
          %603 = arith.remsi %602, %c64_i64 : i64
          %604 = arith.cmpi ne, %603, %c0_i64 : i64
          %605 = arith.cmpi slt, %603, %c0_i64 : i64
          %606 = arith.andi %604, %605 : i1
          %607 = arith.divsi %602, %c64_i64 : i64
          %608 = arith.extui %606 : i1 to i64
          %609 = arith.subi %607, %608 : i64
          %610 = arith.divsi %609, %c64_i64 : i64
          %611 = arith.minsi %610, %c0_i64 : i64
          %612 = arith.muli %611, %c64_i64 : i64
          %613 = arith.subi %609, %612 : i64
          %614 = arith.minsi %613, %c64_i64 : i64
          %615 = arith.cmpi sge, %610, %c0_i64 : i64
          %616 = arith.select %615, %614, %c0_i64 : i64
          %617 = ascscalar.bind_local_ptr %596 : memref<1xbf16, 3>
          %618 = ascscalar.bind_local_ptr %73#0 : memref<1xbf16, 2>
          %619 = arith.index_cast %64 : i64 to index
          %620 = arith.divui %619, %c16 : index
          %621 = arith.index_cast %616 : i64 to index
          %622 = arith.addi %621, %c15 : index
          %623 = arith.divui %622, %c16 : index
          %624 = arith.index_cast %623 : index to i64
          %625 = arith.cmpi ugt, %624, %c0_i64 : i64
          scf.if %625 {
            asccube.get_buf mte1, %599, 0 {cannir.buffer.generated_sync}
            asccube.get_buf mte1, %75, 0 {cannir.buffer.generated_sync}
            %876 = arith.trunci %624 : i64 to i32
            asccube.copy_l12l0a %617[%c0], %618[%620], %c0_i32, %c0_i32, %c4_i32, %876, %c4_i32, %c4_i32 : memref<1xbf16, 3>, memref<1xbf16, 2>
            asccube.release_buf mte1, %75, 0 {cannir.buffer.generated_sync}
            asccube.release_buf mte1, %599, 0 {cannir.buffer.generated_sync}
          }
          %626 = arith.cmpi eq, %arg62, %c2_i64 : i64
          %627 = ascscalar.memref_address %alloc_19 : memref<8192xbf16, 4>
          %628 = ascscalar.memref_address %alloc_21 : memref<8192xbf16, 4>
          %629 = arith.select %626, %627, %628 : i64
          %630 = arith.cmpi eq, %arg62, %c1_i64 : i64
          %631 = ascscalar.memref_address %alloc_17 : memref<8192xbf16, 4>
          %632 = arith.select %630, %631, %629 : i64
          %633 = arith.cmpi eq, %arg62, %c0_i64 : i64
          %634 = ascscalar.memref_address %alloc_15 : memref<8192xbf16, 4>
          %635 = arith.select %633, %634, %632 : i64
          %636 = arith.select %626, %c9_i64, %c10_i64 : i64
          %637 = arith.select %630, %c8_i64, %636 : i64
          %638 = arith.select %633, %c7_i64, %637 : i64
          %639 = arith.addi %arg62, %c1_i64 overflow<nsw> : i64
          %640 = arith.remui %639, %c4_i64 : i64
          %641 = ascscalar.bind_local_ptr %635 : memref<1xbf16, 4>
          %642 = ascscalar.bind_local_ptr %576 : memref<1xbf16, 2>
          %643 = arith.cmpi eq, %578, %c18_i64 : i64
          %644 = arith.select %643, %c4_i64, %c3_i64 : i64
          %645 = arith.cmpi eq, %578, %c19_i64 : i64
          %646 = arith.select %645, %c5_i64, %644 : i64
          asccube.sync_intra_wait mte1, %646 {cannir.sync.banks = 1 : i32}
          asccube.get_buf mte1, %638, 0 {cannir.buffer.generated_sync}
          asccube.copy_l12l0b %641[%c0], %642[%c0], %c0_i32, %c0_i32, %c4_i32, %c4_i32, %c4_i32, %c8_i32 : memref<1xbf16, 4>, memref<1xbf16, 2>
          asccube.release_buf mte1, %638, 0 {cannir.buffer.generated_sync}
          %647 = ascscalar.bind_local_ptr %584 : memref<1xbf16, 2>
          %648 = arith.cmpi eq, %586, %c22_i64 : i64
          %649 = arith.select %648, %c8_i64, %c7_i64 : i64
          %650 = arith.cmpi eq, %586, %c23_i64 : i64
          %651 = arith.select %650, %c9_i64, %649 : i64
          %652 = arith.addi %651, %c16_i64 overflow<nsw> : i64
          asccube.sync_intra_wait mte1, %652 {cannir.sync.banks = 2 : i32}
          asccube.get_buf mte1, %638, 0 {cannir.buffer.generated_sync}
          asccube.copy_l12l0b %641[%c1024], %647[%c0], %c0_i32, %c0_i32, %c4_i32, %c4_i32, %c4_i32, %c8_i32 : memref<1xbf16, 4>, memref<1xbf16, 2>
          asccube.release_buf mte1, %638, 0 {cannir.buffer.generated_sync}
          asccube.get_buf m, %599, 0 {cannir.buffer.generated_sync}
          asccube.get_buf m, %638, 0 {cannir.buffer.generated_sync}
          asccube.mmad %45[%c0], %617, %641, %c64_i32, %c64_i32, %c128_i32, %c2_i32 {c_matrix_init_val = true, mmad_direction = 0 : i32} : memref<?xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
          asccube.release_buf m, %638, 0 {cannir.buffer.generated_sync}
          asccube.release_buf m, %599, 0 {cannir.buffer.generated_sync}
          %653 = arith.cmpi eq, %601, %c2_i64 : i64
          %654 = arith.select %653, %588, %589 : i64
          %655 = arith.cmpi eq, %601, %c1_i64 : i64
          %656 = arith.select %655, %592, %654 : i64
          %657 = arith.cmpi eq, %601, %c0_i64 : i64
          %658 = arith.select %657, %595, %656 : i64
          %659 = arith.select %653, %c5_i64, %c6_i64 : i64
          %660 = arith.select %655, %c4_i64, %659 : i64
          %661 = arith.select %657, %c3_i64, %660 : i64
          %662 = arith.addi %601, %c1_i64 overflow<nsw> : i64
          %663 = arith.remui %662, %c4_i64 : i64
          %664 = arith.minsi %610, %c1_i64 : i64
          %665 = arith.muli %664, %c64_i64 : i64
          %666 = arith.subi %609, %665 : i64
          %667 = arith.minsi %666, %c64_i64 : i64
          %668 = arith.cmpi sge, %610, %c1_i64 : i64
          %669 = arith.select %668, %667, %c0_i64 : i64
          %670 = ascscalar.bind_local_ptr %658 : memref<1xbf16, 3>
          %671 = arith.index_cast %669 : i64 to index
          %672 = arith.addi %671, %c15 : index
          %673 = arith.divui %672, %c16 : index
          %674 = arith.index_cast %673 : index to i64
          %675 = arith.cmpi ugt, %674, %c0_i64 : i64
          scf.if %675 {
            asccube.get_buf mte1, %661, 0 {cannir.buffer.generated_sync}
            asccube.get_buf mte1, %75, 0 {cannir.buffer.generated_sync}
            %876 = arith.trunci %674 : i64 to i32
            asccube.copy_l12l0a %670[%c0], %618[%620], %c0_i32, %c4_i32, %c4_i32, %876, %c4_i32, %c4_i32 : memref<1xbf16, 3>, memref<1xbf16, 2>
            asccube.release_buf mte1, %75, 0 {cannir.buffer.generated_sync}
            asccube.release_buf mte1, %661, 0 {cannir.buffer.generated_sync}
          }
          %676 = arith.cmpi eq, %640, %c2_i64 : i64
          %677 = arith.select %676, %627, %628 : i64
          %678 = arith.cmpi eq, %640, %c1_i64 : i64
          %679 = arith.select %678, %631, %677 : i64
          %680 = arith.cmpi eq, %640, %c0_i64 : i64
          %681 = arith.select %680, %634, %679 : i64
          %682 = arith.select %676, %c9_i64, %c10_i64 : i64
          %683 = arith.select %678, %c8_i64, %682 : i64
          %684 = arith.select %680, %c7_i64, %683 : i64
          %685 = arith.addi %640, %c1_i64 overflow<nsw> : i64
          %686 = arith.remui %685, %c4_i64 : i64
          %687 = ascscalar.bind_local_ptr %681 : memref<1xbf16, 4>
          asccube.get_buf mte1, %684, 0 {cannir.buffer.generated_sync}
          asccube.copy_l12l0b %687[%c0], %642[%c0], %c0_i32, %c4_i32, %c4_i32, %c4_i32, %c4_i32, %c8_i32 : memref<1xbf16, 4>, memref<1xbf16, 2>
          asccube.copy_l12l0b %687[%c1024], %647[%c0], %c0_i32, %c4_i32, %c4_i32, %c4_i32, %c4_i32, %c8_i32 : memref<1xbf16, 4>, memref<1xbf16, 2>
          asccube.release_buf mte1, %684, 0 {cannir.buffer.generated_sync}
          asccube.get_buf m, %661, 0 {cannir.buffer.generated_sync}
          asccube.get_buf m, %684, 0 {cannir.buffer.generated_sync}
          asccube.mmad %45[%c0], %670, %687, %c64_i32, %c64_i32, %c128_i32, %c2_i32 {c_matrix_init_val = false, mmad_direction = 0 : i32} : memref<?xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
          asccube.release_buf m, %684, 0 {cannir.buffer.generated_sync}
          asccube.release_buf m, %661, 0 {cannir.buffer.generated_sync}
          %688 = arith.cmpi eq, %663, %c2_i64 : i64
          %689 = arith.select %688, %588, %589 : i64
          %690 = arith.cmpi eq, %663, %c1_i64 : i64
          %691 = arith.select %690, %592, %689 : i64
          %692 = arith.cmpi eq, %663, %c0_i64 : i64
          %693 = arith.select %692, %595, %691 : i64
          %694 = arith.select %688, %c5_i64, %c6_i64 : i64
          %695 = arith.select %690, %c4_i64, %694 : i64
          %696 = arith.select %692, %c3_i64, %695 : i64
          %697 = arith.addi %663, %c1_i64 overflow<nsw> : i64
          %698 = arith.remui %697, %c4_i64 : i64
          %699 = arith.minsi %610, %c2_i64 : i64
          %700 = arith.muli %699, %c64_i64 : i64
          %701 = arith.subi %609, %700 : i64
          %702 = arith.minsi %701, %c64_i64 : i64
          %703 = arith.cmpi sge, %610, %c2_i64 : i64
          %704 = arith.select %703, %702, %c0_i64 : i64
          %705 = ascscalar.bind_local_ptr %693 : memref<1xbf16, 3>
          %706 = arith.index_cast %704 : i64 to index
          %707 = arith.addi %706, %c15 : index
          %708 = arith.divui %707, %c16 : index
          %709 = arith.index_cast %708 : index to i64
          %710 = arith.cmpi ugt, %709, %c0_i64 : i64
          scf.if %710 {
            asccube.get_buf mte1, %696, 0 {cannir.buffer.generated_sync}
            asccube.get_buf mte1, %75, 0 {cannir.buffer.generated_sync}
            %876 = arith.trunci %709 : i64 to i32
            asccube.copy_l12l0a %705[%c0], %618[%620], %c0_i32, %c8_i32, %c4_i32, %876, %c4_i32, %c4_i32 : memref<1xbf16, 3>, memref<1xbf16, 2>
            asccube.release_buf mte1, %75, 0 {cannir.buffer.generated_sync}
            asccube.release_buf mte1, %696, 0 {cannir.buffer.generated_sync}
          }
          %711 = arith.cmpi eq, %686, %c2_i64 : i64
          %712 = arith.select %711, %627, %628 : i64
          %713 = arith.cmpi eq, %686, %c1_i64 : i64
          %714 = arith.select %713, %631, %712 : i64
          %715 = arith.cmpi eq, %686, %c0_i64 : i64
          %716 = arith.select %715, %634, %714 : i64
          %717 = arith.select %711, %c9_i64, %c10_i64 : i64
          %718 = arith.select %713, %c8_i64, %717 : i64
          %719 = arith.select %715, %c7_i64, %718 : i64
          %720 = arith.addi %686, %c1_i64 overflow<nsw> : i64
          %721 = arith.remui %720, %c4_i64 : i64
          %722 = ascscalar.bind_local_ptr %716 : memref<1xbf16, 4>
          asccube.get_buf mte1, %719, 0 {cannir.buffer.generated_sync}
          asccube.copy_l12l0b %722[%c0], %642[%c0], %c0_i32, %c8_i32, %c4_i32, %c4_i32, %c4_i32, %c8_i32 : memref<1xbf16, 4>, memref<1xbf16, 2>
          asccube.copy_l12l0b %722[%c1024], %647[%c0], %c0_i32, %c8_i32, %c4_i32, %c4_i32, %c4_i32, %c8_i32 : memref<1xbf16, 4>, memref<1xbf16, 2>
          asccube.release_buf mte1, %719, 0 {cannir.buffer.generated_sync}
          asccube.get_buf m, %696, 0 {cannir.buffer.generated_sync}
          asccube.get_buf m, %719, 0 {cannir.buffer.generated_sync}
          asccube.mmad %45[%c0], %705, %722, %c64_i32, %c64_i32, %c128_i32, %c2_i32 {c_matrix_init_val = false, mmad_direction = 0 : i32} : memref<?xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
          asccube.release_buf m, %719, 0 {cannir.buffer.generated_sync}
          asccube.release_buf m, %696, 0 {cannir.buffer.generated_sync}
          %723 = arith.cmpi eq, %698, %c2_i64 : i64
          %724 = arith.select %723, %588, %589 : i64
          %725 = arith.cmpi eq, %698, %c1_i64 : i64
          %726 = arith.select %725, %592, %724 : i64
          %727 = arith.cmpi eq, %698, %c0_i64 : i64
          %728 = arith.select %727, %595, %726 : i64
          %729 = arith.select %723, %c5_i64, %c6_i64 : i64
          %730 = arith.select %725, %c4_i64, %729 : i64
          %731 = arith.select %727, %c3_i64, %730 : i64
          %732 = arith.addi %698, %c1_i64 overflow<nsw> : i64
          %733 = arith.remui %732, %c4_i64 : i64
          %734 = arith.minsi %610, %c3_i64 : i64
          %735 = arith.muli %734, %c64_i64 : i64
          %736 = arith.subi %609, %735 : i64
          %737 = arith.minsi %736, %c64_i64 : i64
          %738 = arith.cmpi sge, %610, %c3_i64 : i64
          %739 = arith.select %738, %737, %c0_i64 : i64
          %740 = ascscalar.bind_local_ptr %728 : memref<1xbf16, 3>
          %741 = arith.index_cast %739 : i64 to index
          %742 = arith.addi %741, %c15 : index
          %743 = arith.divui %742, %c16 : index
          %744 = arith.index_cast %743 : index to i64
          %745 = arith.cmpi ugt, %744, %c0_i64 : i64
          scf.if %745 {
            asccube.get_buf mte1, %731, 0 {cannir.buffer.generated_sync}
            asccube.get_buf mte1, %75, 0 {cannir.buffer.generated_sync}
            %876 = arith.trunci %744 : i64 to i32
            asccube.copy_l12l0a %740[%c0], %618[%620], %c0_i32, %c12_i32, %c4_i32, %876, %c4_i32, %c4_i32 : memref<1xbf16, 3>, memref<1xbf16, 2>
            asccube.release_buf mte1, %75, 0 {cannir.buffer.generated_sync}
            asccube.release_buf mte1, %731, 0 {cannir.buffer.generated_sync}
          }
          %746 = arith.cmpi eq, %721, %c2_i64 : i64
          %747 = arith.select %746, %627, %628 : i64
          %748 = arith.cmpi eq, %721, %c1_i64 : i64
          %749 = arith.select %748, %631, %747 : i64
          %750 = arith.cmpi eq, %721, %c0_i64 : i64
          %751 = arith.select %750, %634, %749 : i64
          %752 = arith.select %746, %c9_i64, %c10_i64 : i64
          %753 = arith.select %748, %c8_i64, %752 : i64
          %754 = arith.select %750, %c7_i64, %753 : i64
          %755 = arith.addi %721, %c1_i64 overflow<nsw> : i64
          %756 = arith.remui %755, %c4_i64 : i64
          %757 = ascscalar.bind_local_ptr %751 : memref<1xbf16, 4>
          asccube.get_buf mte1, %754, 0 {cannir.buffer.generated_sync}
          asccube.copy_l12l0b %757[%c0], %642[%c0], %c0_i32, %c12_i32, %c4_i32, %c4_i32, %c4_i32, %c8_i32 : memref<1xbf16, 4>, memref<1xbf16, 2>
          asccube.copy_l12l0b %757[%c1024], %647[%c0], %c0_i32, %c12_i32, %c4_i32, %c4_i32, %c4_i32, %c8_i32 : memref<1xbf16, 4>, memref<1xbf16, 2>
          asccube.release_buf mte1, %754, 0 {cannir.buffer.generated_sync}
          asccube.get_buf m, %731, 0 {cannir.buffer.generated_sync}
          asccube.get_buf m, %754, 0 {cannir.buffer.generated_sync}
          asccube.mmad %45[%c0], %740, %757, %c64_i32, %c64_i32, %c128_i32, %c2_i32 {c_matrix_init_val = false, mmad_direction = 0 : i32} : memref<?xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
          asccube.release_buf m, %754, 0 {cannir.buffer.generated_sync}
          asccube.release_buf m, %731, 0 {cannir.buffer.generated_sync}
          %758 = arith.cmpi eq, %733, %c2_i64 : i64
          %759 = arith.select %758, %588, %589 : i64
          %760 = arith.cmpi eq, %733, %c1_i64 : i64
          %761 = arith.select %760, %592, %759 : i64
          %762 = arith.cmpi eq, %733, %c0_i64 : i64
          %763 = arith.select %762, %595, %761 : i64
          %764 = arith.select %758, %c5_i64, %c6_i64 : i64
          %765 = arith.select %760, %c4_i64, %764 : i64
          %766 = arith.select %762, %c3_i64, %765 : i64
          %767 = arith.addi %733, %c1_i64 overflow<nsw> : i64
          %768 = arith.remui %767, %c4_i64 : i64
          %769 = arith.minsi %610, %c4_i64 : i64
          %770 = arith.muli %769, %c64_i64 : i64
          %771 = arith.subi %609, %770 : i64
          %772 = arith.minsi %771, %c64_i64 : i64
          %773 = arith.cmpi sge, %610, %c4_i64 : i64
          %774 = arith.select %773, %772, %c0_i64 : i64
          %775 = ascscalar.bind_local_ptr %763 : memref<1xbf16, 3>
          %776 = arith.index_cast %774 : i64 to index
          %777 = arith.addi %776, %c15 : index
          %778 = arith.divui %777, %c16 : index
          %779 = arith.index_cast %778 : index to i64
          %780 = arith.cmpi ugt, %779, %c0_i64 : i64
          scf.if %780 {
            asccube.get_buf mte1, %766, 0 {cannir.buffer.generated_sync}
            asccube.get_buf mte1, %75, 0 {cannir.buffer.generated_sync}
            %876 = arith.trunci %779 : i64 to i32
            asccube.copy_l12l0a %775[%c0], %618[%620], %c0_i32, %c16_i32, %c4_i32, %876, %c4_i32, %c4_i32 : memref<1xbf16, 3>, memref<1xbf16, 2>
            asccube.release_buf mte1, %75, 0 {cannir.buffer.generated_sync}
            asccube.release_buf mte1, %766, 0 {cannir.buffer.generated_sync}
          }
          %781 = arith.cmpi eq, %756, %c2_i64 : i64
          %782 = arith.select %781, %627, %628 : i64
          %783 = arith.cmpi eq, %756, %c1_i64 : i64
          %784 = arith.select %783, %631, %782 : i64
          %785 = arith.cmpi eq, %756, %c0_i64 : i64
          %786 = arith.select %785, %634, %784 : i64
          %787 = arith.select %781, %c9_i64, %c10_i64 : i64
          %788 = arith.select %783, %c8_i64, %787 : i64
          %789 = arith.select %785, %c7_i64, %788 : i64
          %790 = arith.addi %756, %c1_i64 overflow<nsw> : i64
          %791 = arith.remui %790, %c4_i64 : i64
          %792 = ascscalar.bind_local_ptr %786 : memref<1xbf16, 4>
          asccube.get_buf mte1, %789, 0 {cannir.buffer.generated_sync}
          asccube.copy_l12l0b %792[%c0], %642[%c0], %c0_i32, %c16_i32, %c4_i32, %c4_i32, %c4_i32, %c8_i32 : memref<1xbf16, 4>, memref<1xbf16, 2>
          asccube.copy_l12l0b %792[%c1024], %647[%c0], %c0_i32, %c16_i32, %c4_i32, %c4_i32, %c4_i32, %c8_i32 : memref<1xbf16, 4>, memref<1xbf16, 2>
          asccube.release_buf mte1, %789, 0 {cannir.buffer.generated_sync}
          asccube.get_buf m, %766, 0 {cannir.buffer.generated_sync}
          asccube.get_buf m, %789, 0 {cannir.buffer.generated_sync}
          asccube.mmad %45[%c0], %775, %792, %c64_i32, %c64_i32, %c128_i32, %c2_i32 {c_matrix_init_val = false, mmad_direction = 0 : i32} : memref<?xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
          asccube.release_buf m, %789, 0 {cannir.buffer.generated_sync}
          asccube.release_buf m, %766, 0 {cannir.buffer.generated_sync}
          %793 = arith.cmpi eq, %768, %c2_i64 : i64
          %794 = arith.select %793, %588, %589 : i64
          %795 = arith.cmpi eq, %768, %c1_i64 : i64
          %796 = arith.select %795, %592, %794 : i64
          %797 = arith.cmpi eq, %768, %c0_i64 : i64
          %798 = arith.select %797, %595, %796 : i64
          %799 = arith.select %793, %c5_i64, %c6_i64 : i64
          %800 = arith.select %795, %c4_i64, %799 : i64
          %801 = arith.select %797, %c3_i64, %800 : i64
          %802 = arith.addi %768, %c1_i64 overflow<nsw> : i64
          %803 = arith.remui %802, %c4_i64 : i64
          %804 = arith.minsi %610, %c5_i64 : i64
          %805 = arith.muli %804, %c64_i64 : i64
          %806 = arith.subi %609, %805 : i64
          %807 = arith.minsi %806, %c64_i64 : i64
          %808 = arith.cmpi sge, %610, %c5_i64 : i64
          %809 = arith.select %808, %807, %c0_i64 : i64
          %810 = ascscalar.bind_local_ptr %798 : memref<1xbf16, 3>
          %811 = arith.index_cast %809 : i64 to index
          %812 = arith.addi %811, %c15 : index
          %813 = arith.divui %812, %c16 : index
          %814 = arith.index_cast %813 : index to i64
          %815 = arith.cmpi ugt, %814, %c0_i64 : i64
          scf.if %815 {
            asccube.get_buf mte1, %801, 0 {cannir.buffer.generated_sync}
            asccube.get_buf mte1, %75, 0 {cannir.buffer.generated_sync}
            %876 = arith.trunci %814 : i64 to i32
            asccube.copy_l12l0a %810[%c0], %618[%620], %c0_i32, %c20_i32, %c4_i32, %876, %c4_i32, %c4_i32 : memref<1xbf16, 3>, memref<1xbf16, 2>
            asccube.release_buf mte1, %75, 0 {cannir.buffer.generated_sync}
            asccube.release_buf mte1, %801, 0 {cannir.buffer.generated_sync}
          }
          %816 = arith.cmpi eq, %791, %c2_i64 : i64
          %817 = arith.select %816, %627, %628 : i64
          %818 = arith.cmpi eq, %791, %c1_i64 : i64
          %819 = arith.select %818, %631, %817 : i64
          %820 = arith.cmpi eq, %791, %c0_i64 : i64
          %821 = arith.select %820, %634, %819 : i64
          %822 = arith.select %816, %c9_i64, %c10_i64 : i64
          %823 = arith.select %818, %c8_i64, %822 : i64
          %824 = arith.select %820, %c7_i64, %823 : i64
          %825 = arith.addi %791, %c1_i64 overflow<nsw> : i64
          %826 = arith.remui %825, %c4_i64 : i64
          %827 = ascscalar.bind_local_ptr %821 : memref<1xbf16, 4>
          asccube.get_buf mte1, %824, 0 {cannir.buffer.generated_sync}
          asccube.copy_l12l0b %827[%c0], %642[%c0], %c0_i32, %c20_i32, %c4_i32, %c4_i32, %c4_i32, %c8_i32 : memref<1xbf16, 4>, memref<1xbf16, 2>
          asccube.copy_l12l0b %827[%c1024], %647[%c0], %c0_i32, %c20_i32, %c4_i32, %c4_i32, %c4_i32, %c8_i32 : memref<1xbf16, 4>, memref<1xbf16, 2>
          asccube.release_buf mte1, %824, 0 {cannir.buffer.generated_sync}
          asccube.get_buf m, %801, 0 {cannir.buffer.generated_sync}
          asccube.get_buf m, %824, 0 {cannir.buffer.generated_sync}
          asccube.mmad %45[%c0], %810, %827, %c64_i32, %c64_i32, %c128_i32, %c2_i32 {c_matrix_init_val = false, mmad_direction = 0 : i32} : memref<?xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
          asccube.release_buf m, %824, 0 {cannir.buffer.generated_sync}
          asccube.release_buf m, %801, 0 {cannir.buffer.generated_sync}
          %828 = arith.cmpi eq, %803, %c2_i64 : i64
          %829 = arith.select %828, %588, %589 : i64
          %830 = arith.cmpi eq, %803, %c1_i64 : i64
          %831 = arith.select %830, %592, %829 : i64
          %832 = arith.cmpi eq, %803, %c0_i64 : i64
          %833 = arith.select %832, %595, %831 : i64
          %834 = arith.select %828, %c5_i64, %c6_i64 : i64
          %835 = arith.select %830, %c4_i64, %834 : i64
          %836 = arith.select %832, %c3_i64, %835 : i64
          %837 = arith.addi %803, %c1_i64 overflow<nsw> : i64
          %838 = arith.remui %837, %c4_i64 : i64
          %839 = ascscalar.bind_local_ptr %833 : memref<1xbf16, 3>
          asccube.get_buf mte1, %836, 0 {cannir.buffer.generated_sync}
          asccube.get_buf mte1, %c2_i64, 0 {cannir.buffer.generated_sync}
          asccube.copy_l12l0a %839[%c0], %36[%c0], %c0_i32, %c0_i32, %c4_i32, %c4_i32, %c4_i32, %c4_i32 : memref<1xbf16, 3>, memref<?xbf16, 2>
          asccube.release_buf mte1, %c2_i64, 0 {cannir.buffer.generated_sync}
          asccube.release_buf mte1, %836, 0 {cannir.buffer.generated_sync}
          %840 = arith.cmpi eq, %826, %c2_i64 : i64
          %841 = arith.select %840, %627, %628 : i64
          %842 = arith.cmpi eq, %826, %c1_i64 : i64
          %843 = arith.select %842, %631, %841 : i64
          %844 = arith.cmpi eq, %826, %c0_i64 : i64
          %845 = arith.select %844, %634, %843 : i64
          %846 = arith.select %840, %c9_i64, %c10_i64 : i64
          %847 = arith.select %842, %c8_i64, %846 : i64
          %848 = arith.select %844, %c7_i64, %847 : i64
          %849 = arith.addi %826, %c1_i64 overflow<nsw> : i64
          %850 = arith.remui %849, %c4_i64 : i64
          %851 = ascscalar.bind_local_ptr %845 : memref<1xbf16, 4>
          asccube.get_buf mte1, %848, 0 {cannir.buffer.generated_sync}
          asccube.copy_l12l0b %851[%c0], %642[%c0], %c0_i32, %c24_i32, %c4_i32, %c4_i32, %c4_i32, %c8_i32 : memref<1xbf16, 4>, memref<1xbf16, 2>
          asccube.copy_l12l0b %851[%c1024], %647[%c0], %c0_i32, %c24_i32, %c4_i32, %c4_i32, %c4_i32, %c8_i32 : memref<1xbf16, 4>, memref<1xbf16, 2>
          asccube.release_buf mte1, %848, 0 {cannir.buffer.generated_sync}
          asccube.get_buf m, %836, 0 {cannir.buffer.generated_sync}
          asccube.get_buf m, %848, 0 {cannir.buffer.generated_sync}
          asccube.mmad %45[%c0], %839, %851, %c64_i32, %c64_i32, %c128_i32, %c2_i32 {c_matrix_init_val = false, mmad_direction = 0 : i32} : memref<?xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
          asccube.release_buf m, %848, 0 {cannir.buffer.generated_sync}
          asccube.release_buf m, %836, 0 {cannir.buffer.generated_sync}
          %852 = arith.cmpi eq, %838, %c2_i64 : i64
          %853 = arith.select %852, %588, %589 : i64
          %854 = arith.cmpi eq, %838, %c1_i64 : i64
          %855 = arith.select %854, %592, %853 : i64
          %856 = arith.cmpi eq, %838, %c0_i64 : i64
          %857 = arith.select %856, %595, %855 : i64
          %858 = arith.select %852, %c5_i64, %c6_i64 : i64
          %859 = arith.select %854, %c4_i64, %858 : i64
          %860 = arith.select %856, %c3_i64, %859 : i64
          %861 = arith.addi %838, %c1_i64 overflow<nsw> : i64
          %862 = arith.remui %861, %c4_i64 : i64
          %863 = ascscalar.bind_local_ptr %857 : memref<1xbf16, 3>
          asccube.get_buf mte1, %860, 0 {cannir.buffer.generated_sync}
          asccube.get_buf mte1, %c2_i64, 0 {cannir.buffer.generated_sync}
          asccube.copy_l12l0a %863[%c0], %36[%c0], %c0_i32, %c4_i32, %c4_i32, %c4_i32, %c4_i32, %c4_i32 : memref<1xbf16, 3>, memref<?xbf16, 2>
          asccube.release_buf mte1, %c2_i64, 0 {cannir.buffer.generated_sync}
          asccube.release_buf mte1, %860, 0 {cannir.buffer.generated_sync}
          %864 = arith.cmpi eq, %850, %c2_i64 : i64
          %865 = arith.select %864, %627, %628 : i64
          %866 = arith.cmpi eq, %850, %c1_i64 : i64
          %867 = arith.select %866, %631, %865 : i64
          %868 = arith.cmpi eq, %850, %c0_i64 : i64
          %869 = arith.select %868, %634, %867 : i64
          %870 = arith.select %864, %c9_i64, %c10_i64 : i64
          %871 = arith.select %866, %c8_i64, %870 : i64
          %872 = arith.select %868, %c7_i64, %871 : i64
          %873 = arith.addi %850, %c1_i64 overflow<nsw> : i64
          %874 = arith.remui %873, %c4_i64 : i64
          %875 = ascscalar.bind_local_ptr %869 : memref<1xbf16, 4>
          asccube.get_buf mte1, %872, 0 {cannir.buffer.generated_sync}
          asccube.copy_l12l0b %875[%c0], %642[%c0], %c0_i32, %c28_i32, %c4_i32, %c4_i32, %c4_i32, %c8_i32 : memref<1xbf16, 4>, memref<1xbf16, 2>
          asccube.copy_l12l0b %875[%c1024], %647[%c0], %c0_i32, %c28_i32, %c4_i32, %c4_i32, %c4_i32, %c8_i32 : memref<1xbf16, 4>, memref<1xbf16, 2>
          asccube.release_buf mte1, %872, 0 {cannir.buffer.generated_sync}
          asccube.get_buf m, %860, 0 {cannir.buffer.generated_sync}
          asccube.get_buf m, %872, 0 {cannir.buffer.generated_sync}
          asccube.mmad %45[%c0], %863, %875, %c64_i32, %c64_i32, %c128_i32, %c3_i32 {c_matrix_init_val = false, mmad_direction = 0 : i32} : memref<?xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
          asccube.release_buf m, %872, 0 {cannir.buffer.generated_sync}
          asccube.release_buf m, %860, 0 {cannir.buffer.generated_sync}
          asccube.sync_intra_wait fixpipe, %c0_i64 {cannir.sync.banks = 3 : i32}
          asccube.sync_intra_wait fixpipe, %c16_i64 {cannir.sync.banks = 3 : i32}
          asccube.copy_l0c2ub %52[%c0], %45[%c0], %c128_i32, %c64_i32, %c128_i32, %c64_i32, %c3_i32 sub_block_id %c0_i32 {dual_dst_ctl = 1 : i32, nz2nd_en = true, relu_en = false} : memref<?xf32, 1>, memref<?xf32, 5>
          asccube.sync_intra_arrive fixpipe, %c0_i64 {cannir.sync.banks = 3 : i32}
          asccube.sync_intra_arrive fixpipe, %c16_i64 {cannir.sync.banks = 3 : i32}
          scf.yield %576, %576, %576, %584, %584, %584, %580, %862, %874, %578, %586 : i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64
        } else {
          scf.yield %77#0, %77#6, %77#8, %77#10, %77#16, %77#18, %77#20, %77#21, %arg62, %77#22, %77#23 : i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64
        }
        asccube.set_mmad_direction n
        %93 = arith.cmpi eq, %92#7, %c2_i64 : i64
        %94 = ascscalar.memref_address %alloc_18 : memref<8192xbf16, 3>
        %95 = ascscalar.memref_address %alloc_20 : memref<8192xbf16, 3>
        %96 = arith.select %93, %94, %95 : i64
        %97 = arith.cmpi eq, %92#7, %c1_i64 : i64
        %98 = ascscalar.memref_address %alloc_16 : memref<8192xbf16, 3>
        %99 = arith.select %97, %98, %96 : i64
        %100 = arith.cmpi eq, %92#7, %c0_i64 : i64
        %101 = ascscalar.memref_address %alloc_14 : memref<8192xbf16, 3>
        %102 = arith.select %100, %101, %99 : i64
        %103 = arith.select %93, %c5_i64, %c6_i64 : i64
        %104 = arith.select %97, %c4_i64, %103 : i64
        %105 = arith.select %100, %c3_i64, %104 : i64
        %106 = arith.addi %92#7, %c1_i64 overflow<nsw> : i64
        %107 = arith.remui %106, %c4_i64 : i64
        %108 = arith.cmpi eq, %92#8, %c2_i64 : i64
        %109 = ascscalar.memref_address %alloc_19 : memref<8192xbf16, 4>
        %110 = ascscalar.memref_address %alloc_21 : memref<8192xbf16, 4>
        %111 = arith.select %108, %109, %110 : i64
        %112 = arith.cmpi eq, %92#8, %c1_i64 : i64
        %113 = ascscalar.memref_address %alloc_17 : memref<8192xbf16, 4>
        %114 = arith.select %112, %113, %111 : i64
        %115 = arith.cmpi eq, %92#8, %c0_i64 : i64
        %116 = ascscalar.memref_address %alloc_15 : memref<8192xbf16, 4>
        %117 = arith.select %115, %116, %114 : i64
        %118 = arith.select %108, %c9_i64, %c10_i64 : i64
        %119 = arith.select %112, %c8_i64, %118 : i64
        %120 = arith.select %115, %c7_i64, %119 : i64
        %121 = arith.addi %92#8, %c1_i64 overflow<nsw> : i64
        %122 = arith.remui %121, %c4_i64 : i64
        %123 = arith.remsi %77#2, %c64_i64 : i64
        %124 = arith.cmpi ne, %123, %c0_i64 : i64
        %125 = arith.cmpi slt, %123, %c0_i64 : i64
        %126 = arith.andi %124, %125 : i1
        %127 = arith.divsi %77#2, %c64_i64 : i64
        %128 = arith.extui %126 : i1 to i64
        %129 = arith.subi %127, %128 : i64
        %130 = arith.divsi %129, %c64_i64 : i64
        %131 = arith.minsi %130, %c0_i64 : i64
        %132 = arith.muli %131, %c64_i64 : i64
        %133 = arith.subi %129, %132 : i64
        %134 = arith.minsi %133, %c64_i64 : i64
        %135 = arith.cmpi sge, %130, %c0_i64 : i64
        %136 = arith.select %135, %134, %c0_i64 : i64
        %137 = ascscalar.bind_local_ptr %117 : memref<1xbf16, 4>
        %138 = ascscalar.bind_local_ptr %77#0 : memref<1xbf16, 2>
        %139 = arith.index_cast %77#1 : i64 to index
        %140 = arith.divui %139, %c16 : index
        %141 = arith.index_cast %136 : i64 to index
        %142 = arith.addi %141, %c15 : index
        %143 = arith.divui %142, %c16 : index
        %144 = arith.index_cast %143 : index to i64
        %145 = arith.cmpi ugt, %144, %c0_i64 : i64
        scf.if %145 {
          asccube.get_buf mte1, %120, 0 {cannir.buffer.generated_sync}
          %563 = arith.trunci %144 : i64 to i32
          asccube.copy_l12l0b %137[%c0], %138[%140], %c0_i32, %c0_i32, %c4_i32, %563, %c4_i32, %c4_i32 {transpose = true} : memref<1xbf16, 4>, memref<1xbf16, 2>
          asccube.release_buf mte1, %120, 0 {cannir.buffer.generated_sync}
        }
        %146 = arith.remsi %77#12, %c64_i64 : i64
        %147 = arith.cmpi ne, %146, %c0_i64 : i64
        %148 = arith.cmpi slt, %146, %c0_i64 : i64
        %149 = arith.andi %147, %148 : i1
        %150 = arith.divsi %77#12, %c64_i64 : i64
        %151 = arith.extui %149 : i1 to i64
        %152 = arith.subi %150, %151 : i64
        %153 = arith.divsi %152, %c64_i64 : i64
        %154 = arith.minsi %153, %c0_i64 : i64
        %155 = arith.muli %154, %c64_i64 : i64
        %156 = arith.subi %152, %155 : i64
        %157 = arith.minsi %156, %c64_i64 : i64
        %158 = arith.cmpi sge, %153, %c0_i64 : i64
        %159 = arith.select %158, %157, %c0_i64 : i64
        %160 = ascscalar.bind_local_ptr %77#10 : memref<1xbf16, 2>
        %161 = arith.index_cast %77#11 : i64 to index
        %162 = arith.divui %161, %c16 : index
        %163 = arith.index_cast %159 : i64 to index
        %164 = arith.addi %163, %c15 : index
        %165 = arith.divui %164, %c16 : index
        %166 = arith.index_cast %165 : index to i64
        %167 = arith.cmpi ugt, %166, %c0_i64 : i64
        scf.if %167 {
          asccube.get_buf mte1, %120, 0 {cannir.buffer.generated_sync}
          %563 = arith.trunci %166 : i64 to i32
          asccube.copy_l12l0b %137[%c4096], %160[%162], %c0_i32, %c0_i32, %c4_i32, %563, %c4_i32, %c4_i32 {transpose = true} : memref<1xbf16, 4>, memref<1xbf16, 2>
          asccube.release_buf mte1, %120, 0 {cannir.buffer.generated_sync}
        }
        %168 = ascscalar.bind_local_ptr %102 : memref<1xbf16, 3>
        %169 = ascscalar.bitcast_memref %alloc_28 byte_offset = %c0_i64 : memref<8192xbf16, 2> -> memref<?xbf16, 2>
        asccube.sync_intra_wait mte1, %c6_i64 {cannir.sync.banks = 3 : i32}
        asccube.sync_intra_wait mte1, %c22_i64 {cannir.sync.banks = 3 : i32}
        asccube.get_buf mte1, %105, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0a %168[%c0], %169[%c0], %c0_i32, %c0_i32, %c4_i32, %c8_i32, %c4_i32, %c4_i32 : memref<1xbf16, 3>, memref<?xbf16, 2>
        asccube.release_buf mte1, %105, 0 {cannir.buffer.generated_sync}
        asccube.sync_intra_arrive mte1, %c6_i64 {cannir.sync.banks = 3 : i32}
        asccube.sync_intra_arrive mte1, %c22_i64 {cannir.sync.banks = 3 : i32}
        %170 = arith.cmpi eq, %arg63, %c6_i64 : i64
        %171 = ascscalar.memref_address %alloc_36 : memref<4096xf32, 5>
        %172 = ascscalar.memref_address %alloc_35 : memref<4096xf32, 5>
        %173 = arith.select %170, %171, %172 : i64
        %174 = arith.cmpi eq, %arg63, %c5_i64 : i64
        %175 = ascscalar.memref_address %alloc_34 : memref<4096xf32, 5>
        %176 = arith.select %174, %175, %173 : i64
        %177 = arith.cmpi eq, %arg63, %c4_i64 : i64
        %178 = ascscalar.memref_address %alloc_33 : memref<4096xf32, 5>
        %179 = arith.select %177, %178, %176 : i64
        %180 = arith.cmpi eq, %arg63, %c3_i64 : i64
        %181 = ascscalar.memref_address %alloc_32 : memref<4096xf32, 5>
        %182 = arith.select %180, %181, %179 : i64
        %183 = arith.cmpi eq, %arg63, %c2_i64 : i64
        %184 = ascscalar.memref_address %alloc_31 : memref<4096xf32, 5>
        %185 = arith.select %183, %184, %182 : i64
        %186 = arith.cmpi eq, %arg63, %c1_i64 : i64
        %187 = ascscalar.memref_address %alloc_30 : memref<4096xf32, 5>
        %188 = arith.select %186, %187, %185 : i64
        %189 = arith.cmpi eq, %arg63, %c0_i64 : i64
        %190 = ascscalar.memref_address %alloc_29 : memref<4096xf32, 5>
        %191 = arith.select %189, %190, %188 : i64
        %192 = arith.addi %arg63, %c1_i64 overflow<nsw> : i64
        %193 = arith.remui %192, %c8_i64 : i64
        %194 = ascscalar.bind_local_ptr %191 : memref<1xf32, 5>
        asccube.get_buf m, %105, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %120, 0 {cannir.buffer.generated_sync}
        asccube.mmad %194[%c0], %168, %137, %c64_i32, %c128_i32, %c64_i32, %c3_i32 {c_matrix_init_val = true, mmad_direction = 1 : i32} : memref<1xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
        asccube.release_buf m, %120, 0 {cannir.buffer.generated_sync}
        asccube.release_buf m, %105, 0 {cannir.buffer.generated_sync}
        %195 = arith.cmpi eq, %76, %c0_i64 : i64
        scf.if %195 {
          %563 = ascscalar.bitcast_memref %alloc_11 byte_offset = %c0_i64 : memref<16384xf32, 1> -> memref<?xf32, 1>
          asccube.sync_intra_wait fixpipe, %c1_i64 {cannir.sync.banks = 3 : i32}
          asccube.sync_intra_wait fixpipe, %c17_i64 {cannir.sync.banks = 3 : i32}
          asccube.copy_l0c2ub %563[%c0], %194[%c0], %c64_i32, %c64_i32, %c256_i32, %c64_i32, %c3_i32 sub_block_id %c0_i32 {dual_dst_ctl = 1 : i32, nz2nd_en = true, relu_en = false} : memref<?xf32, 1>, memref<1xf32, 5>
        } else {
          %563 = ascscalar.bitcast_memref %alloc_11 byte_offset = %c0_i64 : memref<16384xf32, 1> -> memref<?xf32, 1>
          asccube.sync_intra_wait fixpipe, %c1_i64 {cannir.sync.banks = 3 : i32}
          asccube.sync_intra_wait fixpipe, %c17_i64 {cannir.sync.banks = 3 : i32}
          asccube.copy_l0c2ub %563[%c8192], %194[%c0], %c64_i32, %c64_i32, %c256_i32, %c64_i32, %c3_i32 sub_block_id %c0_i32 {dual_dst_ctl = 1 : i32, nz2nd_en = true, relu_en = false} : memref<?xf32, 1>, memref<1xf32, 5>
        }
        %196 = arith.cmpi eq, %122, %c2_i64 : i64
        %197 = arith.select %196, %109, %110 : i64
        %198 = arith.cmpi eq, %122, %c1_i64 : i64
        %199 = arith.select %198, %113, %197 : i64
        %200 = arith.cmpi eq, %122, %c0_i64 : i64
        %201 = arith.select %200, %116, %199 : i64
        %202 = arith.select %196, %c9_i64, %c10_i64 : i64
        %203 = arith.select %198, %c8_i64, %202 : i64
        %204 = arith.select %200, %c7_i64, %203 : i64
        %205 = arith.addi %122, %c1_i64 overflow<nsw> : i64
        %206 = arith.remui %205, %c4_i64 : i64
        %207 = arith.minsi %130, %c1_i64 : i64
        %208 = arith.muli %207, %c64_i64 : i64
        %209 = arith.subi %129, %208 : i64
        %210 = arith.minsi %209, %c64_i64 : i64
        %211 = arith.cmpi sge, %130, %c1_i64 : i64
        %212 = arith.select %211, %210, %c0_i64 : i64
        %213 = ascscalar.bind_local_ptr %201 : memref<1xbf16, 4>
        %214 = arith.index_cast %212 : i64 to index
        %215 = arith.addi %214, %c15 : index
        %216 = arith.divui %215, %c16 : index
        %217 = arith.index_cast %216 : index to i64
        %218 = arith.cmpi ugt, %217, %c0_i64 : i64
        scf.if %218 {
          asccube.get_buf mte1, %204, 0 {cannir.buffer.generated_sync}
          %563 = arith.trunci %217 : i64 to i32
          asccube.copy_l12l0b %213[%c0], %138[%140], %c0_i32, %c4_i32, %c4_i32, %563, %c4_i32, %c4_i32 {transpose = true} : memref<1xbf16, 4>, memref<1xbf16, 2>
          asccube.release_buf mte1, %204, 0 {cannir.buffer.generated_sync}
        }
        %219 = arith.minsi %153, %c1_i64 : i64
        %220 = arith.muli %219, %c64_i64 : i64
        %221 = arith.subi %152, %220 : i64
        %222 = arith.minsi %221, %c64_i64 : i64
        %223 = arith.cmpi sge, %153, %c1_i64 : i64
        %224 = arith.select %223, %222, %c0_i64 : i64
        %225 = arith.index_cast %224 : i64 to index
        %226 = arith.addi %225, %c15 : index
        %227 = arith.divui %226, %c16 : index
        %228 = arith.index_cast %227 : index to i64
        %229 = arith.cmpi ugt, %228, %c0_i64 : i64
        scf.if %229 {
          asccube.get_buf mte1, %204, 0 {cannir.buffer.generated_sync}
          %563 = arith.trunci %228 : i64 to i32
          asccube.copy_l12l0b %213[%c4096], %160[%162], %c0_i32, %c4_i32, %c4_i32, %563, %c4_i32, %c4_i32 {transpose = true} : memref<1xbf16, 4>, memref<1xbf16, 2>
          asccube.release_buf mte1, %204, 0 {cannir.buffer.generated_sync}
        }
        %230 = arith.cmpi eq, %193, %c6_i64 : i64
        %231 = arith.select %230, %171, %172 : i64
        %232 = arith.cmpi eq, %193, %c5_i64 : i64
        %233 = arith.select %232, %175, %231 : i64
        %234 = arith.cmpi eq, %193, %c4_i64 : i64
        %235 = arith.select %234, %178, %233 : i64
        %236 = arith.cmpi eq, %193, %c3_i64 : i64
        %237 = arith.select %236, %181, %235 : i64
        %238 = arith.cmpi eq, %193, %c2_i64 : i64
        %239 = arith.select %238, %184, %237 : i64
        %240 = arith.cmpi eq, %193, %c1_i64 : i64
        %241 = arith.select %240, %187, %239 : i64
        %242 = arith.cmpi eq, %193, %c0_i64 : i64
        %243 = arith.select %242, %190, %241 : i64
        %244 = arith.addi %193, %c1_i64 overflow<nsw> : i64
        %245 = arith.remui %244, %c8_i64 : i64
        %246 = ascscalar.bind_local_ptr %243 : memref<1xf32, 5>
        asccube.get_buf m, %105, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %204, 0 {cannir.buffer.generated_sync}
        asccube.mmad %246[%c0], %168, %213, %c64_i32, %c128_i32, %c64_i32, %c3_i32 {c_matrix_init_val = true, mmad_direction = 1 : i32} : memref<1xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
        asccube.release_buf m, %204, 0 {cannir.buffer.generated_sync}
        asccube.release_buf m, %105, 0 {cannir.buffer.generated_sync}
        scf.if %195 {
          %563 = ascscalar.bitcast_memref %alloc_11 byte_offset = %c0_i64 : memref<16384xf32, 1> -> memref<?xf32, 1>
          asccube.copy_l0c2ub %563[%c64], %246[%c0], %c64_i32, %c64_i32, %c256_i32, %c64_i32, %c3_i32 sub_block_id %c0_i32 {dual_dst_ctl = 1 : i32, nz2nd_en = true, relu_en = false} : memref<?xf32, 1>, memref<1xf32, 5>
        } else {
          %563 = ascscalar.bitcast_memref %alloc_11 byte_offset = %c0_i64 : memref<16384xf32, 1> -> memref<?xf32, 1>
          asccube.copy_l0c2ub %563[%c8256], %246[%c0], %c64_i32, %c64_i32, %c256_i32, %c64_i32, %c3_i32 sub_block_id %c0_i32 {dual_dst_ctl = 1 : i32, nz2nd_en = true, relu_en = false} : memref<?xf32, 1>, memref<1xf32, 5>
        }
        %247 = arith.cmpi eq, %206, %c2_i64 : i64
        %248 = arith.select %247, %109, %110 : i64
        %249 = arith.cmpi eq, %206, %c1_i64 : i64
        %250 = arith.select %249, %113, %248 : i64
        %251 = arith.cmpi eq, %206, %c0_i64 : i64
        %252 = arith.select %251, %116, %250 : i64
        %253 = arith.select %247, %c9_i64, %c10_i64 : i64
        %254 = arith.select %249, %c8_i64, %253 : i64
        %255 = arith.select %251, %c7_i64, %254 : i64
        %256 = arith.addi %206, %c1_i64 overflow<nsw> : i64
        %257 = arith.remui %256, %c4_i64 : i64
        %258 = arith.minsi %130, %c2_i64 : i64
        %259 = arith.muli %258, %c64_i64 : i64
        %260 = arith.subi %129, %259 : i64
        %261 = arith.minsi %260, %c64_i64 : i64
        %262 = arith.cmpi sge, %130, %c2_i64 : i64
        %263 = arith.select %262, %261, %c0_i64 : i64
        %264 = ascscalar.bind_local_ptr %252 : memref<1xbf16, 4>
        %265 = arith.index_cast %263 : i64 to index
        %266 = arith.addi %265, %c15 : index
        %267 = arith.divui %266, %c16 : index
        %268 = arith.index_cast %267 : index to i64
        %269 = arith.cmpi ugt, %268, %c0_i64 : i64
        scf.if %269 {
          asccube.get_buf mte1, %255, 0 {cannir.buffer.generated_sync}
          %563 = arith.trunci %268 : i64 to i32
          asccube.copy_l12l0b %264[%c0], %138[%140], %c0_i32, %c8_i32, %c4_i32, %563, %c4_i32, %c4_i32 {transpose = true} : memref<1xbf16, 4>, memref<1xbf16, 2>
          asccube.release_buf mte1, %255, 0 {cannir.buffer.generated_sync}
        }
        %270 = arith.minsi %153, %c2_i64 : i64
        %271 = arith.muli %270, %c64_i64 : i64
        %272 = arith.subi %152, %271 : i64
        %273 = arith.minsi %272, %c64_i64 : i64
        %274 = arith.cmpi sge, %153, %c2_i64 : i64
        %275 = arith.select %274, %273, %c0_i64 : i64
        %276 = arith.index_cast %275 : i64 to index
        %277 = arith.addi %276, %c15 : index
        %278 = arith.divui %277, %c16 : index
        %279 = arith.index_cast %278 : index to i64
        %280 = arith.cmpi ugt, %279, %c0_i64 : i64
        scf.if %280 {
          asccube.get_buf mte1, %255, 0 {cannir.buffer.generated_sync}
          %563 = arith.trunci %279 : i64 to i32
          asccube.copy_l12l0b %264[%c4096], %160[%162], %c0_i32, %c8_i32, %c4_i32, %563, %c4_i32, %c4_i32 {transpose = true} : memref<1xbf16, 4>, memref<1xbf16, 2>
          asccube.release_buf mte1, %255, 0 {cannir.buffer.generated_sync}
        }
        %281 = arith.cmpi eq, %245, %c6_i64 : i64
        %282 = arith.select %281, %171, %172 : i64
        %283 = arith.cmpi eq, %245, %c5_i64 : i64
        %284 = arith.select %283, %175, %282 : i64
        %285 = arith.cmpi eq, %245, %c4_i64 : i64
        %286 = arith.select %285, %178, %284 : i64
        %287 = arith.cmpi eq, %245, %c3_i64 : i64
        %288 = arith.select %287, %181, %286 : i64
        %289 = arith.cmpi eq, %245, %c2_i64 : i64
        %290 = arith.select %289, %184, %288 : i64
        %291 = arith.cmpi eq, %245, %c1_i64 : i64
        %292 = arith.select %291, %187, %290 : i64
        %293 = arith.cmpi eq, %245, %c0_i64 : i64
        %294 = arith.select %293, %190, %292 : i64
        %295 = arith.addi %245, %c1_i64 overflow<nsw> : i64
        %296 = arith.remui %295, %c8_i64 : i64
        %297 = ascscalar.bind_local_ptr %294 : memref<1xf32, 5>
        asccube.get_buf m, %105, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %255, 0 {cannir.buffer.generated_sync}
        asccube.mmad %297[%c0], %168, %264, %c64_i32, %c128_i32, %c64_i32, %c3_i32 {c_matrix_init_val = true, mmad_direction = 1 : i32} : memref<1xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
        asccube.release_buf m, %255, 0 {cannir.buffer.generated_sync}
        asccube.release_buf m, %105, 0 {cannir.buffer.generated_sync}
        scf.if %195 {
          %563 = ascscalar.bitcast_memref %alloc_11 byte_offset = %c0_i64 : memref<16384xf32, 1> -> memref<?xf32, 1>
          asccube.copy_l0c2ub %563[%c128], %297[%c0], %c64_i32, %c64_i32, %c256_i32, %c64_i32, %c3_i32 sub_block_id %c0_i32 {dual_dst_ctl = 1 : i32, nz2nd_en = true, relu_en = false} : memref<?xf32, 1>, memref<1xf32, 5>
        } else {
          %563 = ascscalar.bitcast_memref %alloc_11 byte_offset = %c0_i64 : memref<16384xf32, 1> -> memref<?xf32, 1>
          asccube.copy_l0c2ub %563[%c8320], %297[%c0], %c64_i32, %c64_i32, %c256_i32, %c64_i32, %c3_i32 sub_block_id %c0_i32 {dual_dst_ctl = 1 : i32, nz2nd_en = true, relu_en = false} : memref<?xf32, 1>, memref<1xf32, 5>
        }
        %298 = arith.cmpi eq, %257, %c2_i64 : i64
        %299 = arith.select %298, %109, %110 : i64
        %300 = arith.cmpi eq, %257, %c1_i64 : i64
        %301 = arith.select %300, %113, %299 : i64
        %302 = arith.cmpi eq, %257, %c0_i64 : i64
        %303 = arith.select %302, %116, %301 : i64
        %304 = arith.select %298, %c9_i64, %c10_i64 : i64
        %305 = arith.select %300, %c8_i64, %304 : i64
        %306 = arith.select %302, %c7_i64, %305 : i64
        %307 = arith.addi %257, %c1_i64 overflow<nsw> : i64
        %308 = arith.remui %307, %c4_i64 : i64
        %309 = arith.minsi %130, %c3_i64 : i64
        %310 = arith.muli %309, %c64_i64 : i64
        %311 = arith.subi %129, %310 : i64
        %312 = arith.minsi %311, %c64_i64 : i64
        %313 = arith.cmpi sge, %130, %c3_i64 : i64
        %314 = arith.select %313, %312, %c0_i64 : i64
        %315 = ascscalar.bind_local_ptr %303 : memref<1xbf16, 4>
        %316 = arith.index_cast %314 : i64 to index
        %317 = arith.addi %316, %c15 : index
        %318 = arith.divui %317, %c16 : index
        %319 = arith.index_cast %318 : index to i64
        %320 = arith.cmpi ugt, %319, %c0_i64 : i64
        scf.if %320 {
          asccube.get_buf mte1, %306, 0 {cannir.buffer.generated_sync}
          %563 = arith.trunci %319 : i64 to i32
          asccube.copy_l12l0b %315[%c0], %138[%140], %c0_i32, %c12_i32, %c4_i32, %563, %c4_i32, %c4_i32 {transpose = true} : memref<1xbf16, 4>, memref<1xbf16, 2>
          asccube.release_buf mte1, %306, 0 {cannir.buffer.generated_sync}
        }
        %321 = arith.minsi %153, %c3_i64 : i64
        %322 = arith.muli %321, %c64_i64 : i64
        %323 = arith.subi %152, %322 : i64
        %324 = arith.minsi %323, %c64_i64 : i64
        %325 = arith.cmpi sge, %153, %c3_i64 : i64
        %326 = arith.select %325, %324, %c0_i64 : i64
        %327 = arith.index_cast %326 : i64 to index
        %328 = arith.addi %327, %c15 : index
        %329 = arith.divui %328, %c16 : index
        %330 = arith.index_cast %329 : index to i64
        %331 = arith.cmpi ugt, %330, %c0_i64 : i64
        scf.if %331 {
          asccube.get_buf mte1, %306, 0 {cannir.buffer.generated_sync}
          %563 = arith.trunci %330 : i64 to i32
          asccube.copy_l12l0b %315[%c4096], %160[%162], %c0_i32, %c12_i32, %c4_i32, %563, %c4_i32, %c4_i32 {transpose = true} : memref<1xbf16, 4>, memref<1xbf16, 2>
          asccube.release_buf mte1, %306, 0 {cannir.buffer.generated_sync}
        }
        %332 = arith.cmpi eq, %296, %c6_i64 : i64
        %333 = arith.select %332, %171, %172 : i64
        %334 = arith.cmpi eq, %296, %c5_i64 : i64
        %335 = arith.select %334, %175, %333 : i64
        %336 = arith.cmpi eq, %296, %c4_i64 : i64
        %337 = arith.select %336, %178, %335 : i64
        %338 = arith.cmpi eq, %296, %c3_i64 : i64
        %339 = arith.select %338, %181, %337 : i64
        %340 = arith.cmpi eq, %296, %c2_i64 : i64
        %341 = arith.select %340, %184, %339 : i64
        %342 = arith.cmpi eq, %296, %c1_i64 : i64
        %343 = arith.select %342, %187, %341 : i64
        %344 = arith.cmpi eq, %296, %c0_i64 : i64
        %345 = arith.select %344, %190, %343 : i64
        %346 = arith.addi %296, %c1_i64 overflow<nsw> : i64
        %347 = arith.remui %346, %c8_i64 : i64
        %348 = ascscalar.bind_local_ptr %345 : memref<1xf32, 5>
        asccube.get_buf m, %105, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %306, 0 {cannir.buffer.generated_sync}
        asccube.mmad %348[%c0], %168, %315, %c64_i32, %c128_i32, %c64_i32, %c3_i32 {c_matrix_init_val = true, mmad_direction = 1 : i32} : memref<1xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
        asccube.release_buf m, %306, 0 {cannir.buffer.generated_sync}
        asccube.release_buf m, %105, 0 {cannir.buffer.generated_sync}
        scf.if %195 {
          %563 = ascscalar.bitcast_memref %alloc_11 byte_offset = %c0_i64 : memref<16384xf32, 1> -> memref<?xf32, 1>
          asccube.copy_l0c2ub %563[%c192], %348[%c0], %c64_i32, %c64_i32, %c256_i32, %c64_i32, %c3_i32 sub_block_id %c0_i32 {dual_dst_ctl = 1 : i32, nz2nd_en = true, relu_en = false} : memref<?xf32, 1>, memref<1xf32, 5>
          asccube.sync_intra_arrive fixpipe, %c1_i64 {cannir.sync.banks = 3 : i32}
          asccube.sync_intra_arrive fixpipe, %c17_i64 {cannir.sync.banks = 3 : i32}
        } else {
          %563 = ascscalar.bitcast_memref %alloc_11 byte_offset = %c0_i64 : memref<16384xf32, 1> -> memref<?xf32, 1>
          asccube.copy_l0c2ub %563[%c8384], %348[%c0], %c64_i32, %c64_i32, %c256_i32, %c64_i32, %c3_i32 sub_block_id %c0_i32 {dual_dst_ctl = 1 : i32, nz2nd_en = true, relu_en = false} : memref<?xf32, 1>, memref<1xf32, 5>
          asccube.sync_intra_arrive fixpipe, %c1_i64 {cannir.sync.banks = 3 : i32}
          asccube.sync_intra_arrive fixpipe, %c17_i64 {cannir.sync.banks = 3 : i32}
        }
        %349 = arith.cmpi eq, %308, %c2_i64 : i64
        %350 = arith.select %349, %109, %110 : i64
        %351 = arith.cmpi eq, %308, %c1_i64 : i64
        %352 = arith.select %351, %113, %350 : i64
        %353 = arith.cmpi eq, %308, %c0_i64 : i64
        %354 = arith.select %353, %116, %352 : i64
        %355 = arith.select %349, %c9_i64, %c10_i64 : i64
        %356 = arith.select %351, %c8_i64, %355 : i64
        %357 = arith.select %353, %c7_i64, %356 : i64
        %358 = arith.addi %308, %c1_i64 overflow<nsw> : i64
        %359 = arith.remui %358, %c4_i64 : i64
        %360 = arith.minsi %130, %c4_i64 : i64
        %361 = arith.muli %360, %c64_i64 : i64
        %362 = arith.subi %129, %361 : i64
        %363 = arith.minsi %362, %c64_i64 : i64
        %364 = arith.cmpi sge, %130, %c4_i64 : i64
        %365 = arith.select %364, %363, %c0_i64 : i64
        %366 = ascscalar.bind_local_ptr %354 : memref<1xbf16, 4>
        %367 = arith.index_cast %365 : i64 to index
        %368 = arith.addi %367, %c15 : index
        %369 = arith.divui %368, %c16 : index
        %370 = arith.index_cast %369 : index to i64
        %371 = arith.cmpi ugt, %370, %c0_i64 : i64
        scf.if %371 {
          asccube.get_buf mte1, %357, 0 {cannir.buffer.generated_sync}
          %563 = arith.trunci %370 : i64 to i32
          asccube.copy_l12l0b %366[%c0], %138[%140], %c0_i32, %c16_i32, %c4_i32, %563, %c4_i32, %c4_i32 {transpose = true} : memref<1xbf16, 4>, memref<1xbf16, 2>
          asccube.release_buf mte1, %357, 0 {cannir.buffer.generated_sync}
        }
        %372 = arith.minsi %153, %c4_i64 : i64
        %373 = arith.muli %372, %c64_i64 : i64
        %374 = arith.subi %152, %373 : i64
        %375 = arith.minsi %374, %c64_i64 : i64
        %376 = arith.cmpi sge, %153, %c4_i64 : i64
        %377 = arith.select %376, %375, %c0_i64 : i64
        %378 = arith.index_cast %377 : i64 to index
        %379 = arith.addi %378, %c15 : index
        %380 = arith.divui %379, %c16 : index
        %381 = arith.index_cast %380 : index to i64
        %382 = arith.cmpi ugt, %381, %c0_i64 : i64
        scf.if %382 {
          asccube.get_buf mte1, %357, 0 {cannir.buffer.generated_sync}
          %563 = arith.trunci %381 : i64 to i32
          asccube.copy_l12l0b %366[%c4096], %160[%162], %c0_i32, %c16_i32, %c4_i32, %563, %c4_i32, %c4_i32 {transpose = true} : memref<1xbf16, 4>, memref<1xbf16, 2>
          asccube.release_buf mte1, %357, 0 {cannir.buffer.generated_sync}
        }
        %383 = arith.cmpi eq, %347, %c6_i64 : i64
        %384 = arith.select %383, %171, %172 : i64
        %385 = arith.cmpi eq, %347, %c5_i64 : i64
        %386 = arith.select %385, %175, %384 : i64
        %387 = arith.cmpi eq, %347, %c4_i64 : i64
        %388 = arith.select %387, %178, %386 : i64
        %389 = arith.cmpi eq, %347, %c3_i64 : i64
        %390 = arith.select %389, %181, %388 : i64
        %391 = arith.cmpi eq, %347, %c2_i64 : i64
        %392 = arith.select %391, %184, %390 : i64
        %393 = arith.cmpi eq, %347, %c1_i64 : i64
        %394 = arith.select %393, %187, %392 : i64
        %395 = arith.cmpi eq, %347, %c0_i64 : i64
        %396 = arith.select %395, %190, %394 : i64
        %397 = arith.addi %347, %c1_i64 overflow<nsw> : i64
        %398 = arith.remui %397, %c8_i64 : i64
        %399 = ascscalar.bind_local_ptr %396 : memref<1xf32, 5>
        asccube.get_buf m, %105, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %357, 0 {cannir.buffer.generated_sync}
        asccube.mmad %399[%c0], %168, %366, %c64_i32, %c128_i32, %c64_i32, %c3_i32 {c_matrix_init_val = true, mmad_direction = 1 : i32} : memref<1xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
        asccube.release_buf m, %357, 0 {cannir.buffer.generated_sync}
        asccube.release_buf m, %105, 0 {cannir.buffer.generated_sync}
        scf.if %195 {
          %563 = ascscalar.bitcast_memref %alloc_12 byte_offset = %c0_i64 : memref<16384xf32, 1> -> memref<?xf32, 1>
          asccube.sync_intra_wait fixpipe, %c2_i64 {cannir.sync.banks = 3 : i32}
          asccube.sync_intra_wait fixpipe, %c18_i64 {cannir.sync.banks = 3 : i32}
          asccube.copy_l0c2ub %563[%c0], %399[%c0], %c64_i32, %c64_i32, %c256_i32, %c64_i32, %c3_i32 sub_block_id %c0_i32 {dual_dst_ctl = 1 : i32, nz2nd_en = true, relu_en = false} : memref<?xf32, 1>, memref<1xf32, 5>
        } else {
          %563 = ascscalar.bitcast_memref %alloc_12 byte_offset = %c0_i64 : memref<16384xf32, 1> -> memref<?xf32, 1>
          asccube.sync_intra_wait fixpipe, %c2_i64 {cannir.sync.banks = 3 : i32}
          asccube.sync_intra_wait fixpipe, %c18_i64 {cannir.sync.banks = 3 : i32}
          asccube.copy_l0c2ub %563[%c8192], %399[%c0], %c64_i32, %c64_i32, %c256_i32, %c64_i32, %c3_i32 sub_block_id %c0_i32 {dual_dst_ctl = 1 : i32, nz2nd_en = true, relu_en = false} : memref<?xf32, 1>, memref<1xf32, 5>
        }
        %400 = arith.cmpi eq, %359, %c2_i64 : i64
        %401 = arith.select %400, %109, %110 : i64
        %402 = arith.cmpi eq, %359, %c1_i64 : i64
        %403 = arith.select %402, %113, %401 : i64
        %404 = arith.cmpi eq, %359, %c0_i64 : i64
        %405 = arith.select %404, %116, %403 : i64
        %406 = arith.select %400, %c9_i64, %c10_i64 : i64
        %407 = arith.select %402, %c8_i64, %406 : i64
        %408 = arith.select %404, %c7_i64, %407 : i64
        %409 = arith.addi %359, %c1_i64 overflow<nsw> : i64
        %410 = arith.remui %409, %c4_i64 : i64
        %411 = arith.minsi %130, %c5_i64 : i64
        %412 = arith.muli %411, %c64_i64 : i64
        %413 = arith.subi %129, %412 : i64
        %414 = arith.minsi %413, %c64_i64 : i64
        %415 = arith.cmpi sge, %130, %c5_i64 : i64
        %416 = arith.select %415, %414, %c0_i64 : i64
        %417 = ascscalar.bind_local_ptr %405 : memref<1xbf16, 4>
        %418 = arith.index_cast %416 : i64 to index
        %419 = arith.addi %418, %c15 : index
        %420 = arith.divui %419, %c16 : index
        %421 = arith.index_cast %420 : index to i64
        %422 = arith.cmpi ugt, %421, %c0_i64 : i64
        scf.if %422 {
          asccube.get_buf mte1, %408, 0 {cannir.buffer.generated_sync}
          %563 = arith.trunci %421 : i64 to i32
          asccube.copy_l12l0b %417[%c0], %138[%140], %c0_i32, %c20_i32, %c4_i32, %563, %c4_i32, %c4_i32 {transpose = true} : memref<1xbf16, 4>, memref<1xbf16, 2>
          asccube.release_buf mte1, %408, 0 {cannir.buffer.generated_sync}
        }
        %423 = arith.minsi %153, %c5_i64 : i64
        %424 = arith.muli %423, %c64_i64 : i64
        %425 = arith.subi %152, %424 : i64
        %426 = arith.minsi %425, %c64_i64 : i64
        %427 = arith.cmpi sge, %153, %c5_i64 : i64
        %428 = arith.select %427, %426, %c0_i64 : i64
        %429 = arith.index_cast %428 : i64 to index
        %430 = arith.addi %429, %c15 : index
        %431 = arith.divui %430, %c16 : index
        %432 = arith.index_cast %431 : index to i64
        %433 = arith.cmpi ugt, %432, %c0_i64 : i64
        scf.if %433 {
          asccube.get_buf mte1, %408, 0 {cannir.buffer.generated_sync}
          %563 = arith.trunci %432 : i64 to i32
          asccube.copy_l12l0b %417[%c4096], %160[%162], %c0_i32, %c20_i32, %c4_i32, %563, %c4_i32, %c4_i32 {transpose = true} : memref<1xbf16, 4>, memref<1xbf16, 2>
          asccube.release_buf mte1, %408, 0 {cannir.buffer.generated_sync}
        }
        %434 = arith.cmpi eq, %398, %c6_i64 : i64
        %435 = arith.select %434, %171, %172 : i64
        %436 = arith.cmpi eq, %398, %c5_i64 : i64
        %437 = arith.select %436, %175, %435 : i64
        %438 = arith.cmpi eq, %398, %c4_i64 : i64
        %439 = arith.select %438, %178, %437 : i64
        %440 = arith.cmpi eq, %398, %c3_i64 : i64
        %441 = arith.select %440, %181, %439 : i64
        %442 = arith.cmpi eq, %398, %c2_i64 : i64
        %443 = arith.select %442, %184, %441 : i64
        %444 = arith.cmpi eq, %398, %c1_i64 : i64
        %445 = arith.select %444, %187, %443 : i64
        %446 = arith.cmpi eq, %398, %c0_i64 : i64
        %447 = arith.select %446, %190, %445 : i64
        %448 = arith.addi %398, %c1_i64 overflow<nsw> : i64
        %449 = arith.remui %448, %c8_i64 : i64
        %450 = ascscalar.bind_local_ptr %447 : memref<1xf32, 5>
        asccube.get_buf m, %105, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %408, 0 {cannir.buffer.generated_sync}
        asccube.mmad %450[%c0], %168, %417, %c64_i32, %c128_i32, %c64_i32, %c3_i32 {c_matrix_init_val = true, mmad_direction = 1 : i32} : memref<1xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
        asccube.release_buf m, %408, 0 {cannir.buffer.generated_sync}
        asccube.release_buf m, %105, 0 {cannir.buffer.generated_sync}
        scf.if %195 {
          %563 = ascscalar.bitcast_memref %alloc_12 byte_offset = %c0_i64 : memref<16384xf32, 1> -> memref<?xf32, 1>
          asccube.copy_l0c2ub %563[%c64], %450[%c0], %c64_i32, %c64_i32, %c256_i32, %c64_i32, %c3_i32 sub_block_id %c0_i32 {dual_dst_ctl = 1 : i32, nz2nd_en = true, relu_en = false} : memref<?xf32, 1>, memref<1xf32, 5>
        } else {
          %563 = ascscalar.bitcast_memref %alloc_12 byte_offset = %c0_i64 : memref<16384xf32, 1> -> memref<?xf32, 1>
          asccube.copy_l0c2ub %563[%c8256], %450[%c0], %c64_i32, %c64_i32, %c256_i32, %c64_i32, %c3_i32 sub_block_id %c0_i32 {dual_dst_ctl = 1 : i32, nz2nd_en = true, relu_en = false} : memref<?xf32, 1>, memref<1xf32, 5>
        }
        %451 = arith.cmpi eq, %410, %c2_i64 : i64
        %452 = arith.select %451, %109, %110 : i64
        %453 = arith.cmpi eq, %410, %c1_i64 : i64
        %454 = arith.select %453, %113, %452 : i64
        %455 = arith.cmpi eq, %410, %c0_i64 : i64
        %456 = arith.select %455, %116, %454 : i64
        %457 = arith.select %451, %c9_i64, %c10_i64 : i64
        %458 = arith.select %453, %c8_i64, %457 : i64
        %459 = arith.select %455, %c7_i64, %458 : i64
        %460 = arith.addi %410, %c1_i64 overflow<nsw> : i64
        %461 = arith.remui %460, %c4_i64 : i64
        %462 = arith.minsi %130, %c6_i64 : i64
        %463 = arith.muli %462, %c64_i64 : i64
        %464 = arith.subi %129, %463 : i64
        %465 = arith.minsi %464, %c64_i64 : i64
        %466 = arith.cmpi sge, %130, %c6_i64 : i64
        %467 = arith.select %466, %465, %c0_i64 : i64
        %468 = ascscalar.bind_local_ptr %456 : memref<1xbf16, 4>
        %469 = arith.index_cast %467 : i64 to index
        %470 = arith.addi %469, %c15 : index
        %471 = arith.divui %470, %c16 : index
        %472 = arith.index_cast %471 : index to i64
        %473 = arith.cmpi ugt, %472, %c0_i64 : i64
        scf.if %473 {
          asccube.get_buf mte1, %459, 0 {cannir.buffer.generated_sync}
          %563 = arith.trunci %472 : i64 to i32
          asccube.copy_l12l0b %468[%c0], %138[%140], %c0_i32, %c24_i32, %c4_i32, %563, %c4_i32, %c4_i32 {transpose = true} : memref<1xbf16, 4>, memref<1xbf16, 2>
          asccube.release_buf mte1, %459, 0 {cannir.buffer.generated_sync}
        }
        %474 = arith.minsi %153, %c6_i64 : i64
        %475 = arith.muli %474, %c64_i64 : i64
        %476 = arith.subi %152, %475 : i64
        %477 = arith.minsi %476, %c64_i64 : i64
        %478 = arith.cmpi sge, %153, %c6_i64 : i64
        %479 = arith.select %478, %477, %c0_i64 : i64
        %480 = arith.index_cast %479 : i64 to index
        %481 = arith.addi %480, %c15 : index
        %482 = arith.divui %481, %c16 : index
        %483 = arith.index_cast %482 : index to i64
        %484 = arith.cmpi ugt, %483, %c0_i64 : i64
        scf.if %484 {
          asccube.get_buf mte1, %459, 0 {cannir.buffer.generated_sync}
          %563 = arith.trunci %483 : i64 to i32
          asccube.copy_l12l0b %468[%c4096], %160[%162], %c0_i32, %c24_i32, %c4_i32, %563, %c4_i32, %c4_i32 {transpose = true} : memref<1xbf16, 4>, memref<1xbf16, 2>
          asccube.release_buf mte1, %459, 0 {cannir.buffer.generated_sync}
        }
        %485 = arith.cmpi eq, %449, %c6_i64 : i64
        %486 = arith.select %485, %171, %172 : i64
        %487 = arith.cmpi eq, %449, %c5_i64 : i64
        %488 = arith.select %487, %175, %486 : i64
        %489 = arith.cmpi eq, %449, %c4_i64 : i64
        %490 = arith.select %489, %178, %488 : i64
        %491 = arith.cmpi eq, %449, %c3_i64 : i64
        %492 = arith.select %491, %181, %490 : i64
        %493 = arith.cmpi eq, %449, %c2_i64 : i64
        %494 = arith.select %493, %184, %492 : i64
        %495 = arith.cmpi eq, %449, %c1_i64 : i64
        %496 = arith.select %495, %187, %494 : i64
        %497 = arith.cmpi eq, %449, %c0_i64 : i64
        %498 = arith.select %497, %190, %496 : i64
        %499 = arith.addi %449, %c1_i64 overflow<nsw> : i64
        %500 = arith.remui %499, %c8_i64 : i64
        %501 = ascscalar.bind_local_ptr %498 : memref<1xf32, 5>
        asccube.get_buf m, %105, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %459, 0 {cannir.buffer.generated_sync}
        asccube.mmad %501[%c0], %168, %468, %c64_i32, %c128_i32, %c64_i32, %c3_i32 {c_matrix_init_val = true, mmad_direction = 1 : i32} : memref<1xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
        asccube.release_buf m, %459, 0 {cannir.buffer.generated_sync}
        asccube.release_buf m, %105, 0 {cannir.buffer.generated_sync}
        scf.if %195 {
          %563 = ascscalar.bitcast_memref %alloc_12 byte_offset = %c0_i64 : memref<16384xf32, 1> -> memref<?xf32, 1>
          asccube.copy_l0c2ub %563[%c128], %501[%c0], %c64_i32, %c64_i32, %c256_i32, %c64_i32, %c3_i32 sub_block_id %c0_i32 {dual_dst_ctl = 1 : i32, nz2nd_en = true, relu_en = false} : memref<?xf32, 1>, memref<1xf32, 5>
        } else {
          %563 = ascscalar.bitcast_memref %alloc_12 byte_offset = %c0_i64 : memref<16384xf32, 1> -> memref<?xf32, 1>
          asccube.copy_l0c2ub %563[%c8320], %501[%c0], %c64_i32, %c64_i32, %c256_i32, %c64_i32, %c3_i32 sub_block_id %c0_i32 {dual_dst_ctl = 1 : i32, nz2nd_en = true, relu_en = false} : memref<?xf32, 1>, memref<1xf32, 5>
        }
        %502 = arith.cmpi eq, %461, %c2_i64 : i64
        %503 = arith.select %502, %109, %110 : i64
        %504 = arith.cmpi eq, %461, %c1_i64 : i64
        %505 = arith.select %504, %113, %503 : i64
        %506 = arith.cmpi eq, %461, %c0_i64 : i64
        %507 = arith.select %506, %116, %505 : i64
        %508 = arith.select %502, %c9_i64, %c10_i64 : i64
        %509 = arith.select %504, %c8_i64, %508 : i64
        %510 = arith.select %506, %c7_i64, %509 : i64
        %511 = arith.addi %461, %c1_i64 overflow<nsw> : i64
        %512 = arith.remui %511, %c4_i64 : i64
        %513 = arith.minsi %130, %c7_i64 : i64
        %514 = arith.muli %513, %c64_i64 : i64
        %515 = arith.subi %129, %514 : i64
        %516 = arith.minsi %515, %c64_i64 : i64
        %517 = arith.cmpi sge, %130, %c7_i64 : i64
        %518 = arith.select %517, %516, %c0_i64 : i64
        %519 = ascscalar.bind_local_ptr %507 : memref<1xbf16, 4>
        %520 = arith.index_cast %518 : i64 to index
        %521 = arith.addi %520, %c15 : index
        %522 = arith.divui %521, %c16 : index
        %523 = arith.index_cast %522 : index to i64
        %524 = arith.cmpi ugt, %523, %c0_i64 : i64
        scf.if %524 {
          asccube.get_buf mte1, %510, 0 {cannir.buffer.generated_sync}
          %563 = arith.trunci %523 : i64 to i32
          asccube.copy_l12l0b %519[%c0], %138[%140], %c0_i32, %c28_i32, %c4_i32, %563, %c4_i32, %c4_i32 {transpose = true} : memref<1xbf16, 4>, memref<1xbf16, 2>
          asccube.release_buf mte1, %510, 0 {cannir.buffer.generated_sync}
        }
        %525 = arith.cmpi eq, %77#22, %c18_i64 : i64
        %526 = arith.select %525, %c4_i64, %c3_i64 : i64
        %527 = arith.cmpi eq, %77#22, %c19_i64 : i64
        %528 = arith.select %527, %c5_i64, %526 : i64
        asccube.sync_intra_arrive mte1, %528 {cannir.sync.banks = 1 : i32}
        %529 = arith.minsi %153, %c7_i64 : i64
        %530 = arith.muli %529, %c64_i64 : i64
        %531 = arith.subi %152, %530 : i64
        %532 = arith.minsi %531, %c64_i64 : i64
        %533 = arith.cmpi sge, %153, %c7_i64 : i64
        %534 = arith.select %533, %532, %c0_i64 : i64
        %535 = arith.index_cast %534 : i64 to index
        %536 = arith.addi %535, %c15 : index
        %537 = arith.divui %536, %c16 : index
        %538 = arith.index_cast %537 : index to i64
        %539 = arith.cmpi ugt, %538, %c0_i64 : i64
        scf.if %539 {
          asccube.get_buf mte1, %510, 0 {cannir.buffer.generated_sync}
          %563 = arith.trunci %538 : i64 to i32
          asccube.copy_l12l0b %519[%c4096], %160[%162], %c0_i32, %c28_i32, %c4_i32, %563, %c4_i32, %c4_i32 {transpose = true} : memref<1xbf16, 4>, memref<1xbf16, 2>
          asccube.release_buf mte1, %510, 0 {cannir.buffer.generated_sync}
        }
        %540 = arith.cmpi eq, %77#23, %c22_i64 : i64
        %541 = arith.select %540, %c8_i64, %c7_i64 : i64
        %542 = arith.cmpi eq, %77#23, %c23_i64 : i64
        %543 = arith.select %542, %c9_i64, %541 : i64
        %544 = arith.addi %543, %c16_i64 overflow<nsw> : i64
        asccube.sync_intra_arrive mte1, %544 {cannir.sync.banks = 2 : i32}
        %545 = arith.cmpi eq, %500, %c6_i64 : i64
        %546 = arith.select %545, %171, %172 : i64
        %547 = arith.cmpi eq, %500, %c5_i64 : i64
        %548 = arith.select %547, %175, %546 : i64
        %549 = arith.cmpi eq, %500, %c4_i64 : i64
        %550 = arith.select %549, %178, %548 : i64
        %551 = arith.cmpi eq, %500, %c3_i64 : i64
        %552 = arith.select %551, %181, %550 : i64
        %553 = arith.cmpi eq, %500, %c2_i64 : i64
        %554 = arith.select %553, %184, %552 : i64
        %555 = arith.cmpi eq, %500, %c1_i64 : i64
        %556 = arith.select %555, %187, %554 : i64
        %557 = arith.cmpi eq, %500, %c0_i64 : i64
        %558 = arith.select %557, %190, %556 : i64
        %559 = arith.addi %500, %c1_i64 overflow<nsw> : i64
        %560 = arith.remui %559, %c8_i64 : i64
        %561 = ascscalar.bind_local_ptr %558 : memref<1xf32, 5>
        asccube.get_buf m, %105, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %510, 0 {cannir.buffer.generated_sync}
        asccube.mmad %561[%c0], %168, %519, %c64_i32, %c128_i32, %c64_i32, %c3_i32 {c_matrix_init_val = true, mmad_direction = 1 : i32} : memref<1xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
        asccube.release_buf m, %510, 0 {cannir.buffer.generated_sync}
        asccube.release_buf m, %105, 0 {cannir.buffer.generated_sync}
        scf.if %195 {
          %563 = ascscalar.bitcast_memref %alloc_12 byte_offset = %c0_i64 : memref<16384xf32, 1> -> memref<?xf32, 1>
          asccube.copy_l0c2ub %563[%c192], %561[%c0], %c64_i32, %c64_i32, %c256_i32, %c64_i32, %c3_i32 sub_block_id %c0_i32 {dual_dst_ctl = 1 : i32, nz2nd_en = true, relu_en = false} : memref<?xf32, 1>, memref<1xf32, 5>
          asccube.sync_intra_arrive fixpipe, %c2_i64 {cannir.sync.banks = 3 : i32}
          asccube.sync_intra_arrive fixpipe, %c18_i64 {cannir.sync.banks = 3 : i32}
        } else {
          %563 = ascscalar.bitcast_memref %alloc_12 byte_offset = %c0_i64 : memref<16384xf32, 1> -> memref<?xf32, 1>
          asccube.copy_l0c2ub %563[%c8384], %561[%c0], %c64_i32, %c64_i32, %c256_i32, %c64_i32, %c3_i32 sub_block_id %c0_i32 {dual_dst_ctl = 1 : i32, nz2nd_en = true, relu_en = false} : memref<?xf32, 1>, memref<1xf32, 5>
          asccube.sync_intra_arrive fixpipe, %c2_i64 {cannir.sync.banks = 3 : i32}
          asccube.sync_intra_arrive fixpipe, %c18_i64 {cannir.sync.banks = 3 : i32}
        }
        %562 = arith.addi %arg58, %c1_i64 : i64
        scf.yield %92#0, %78, %79, %80, %81, %82, %92#1, %83, %92#2, %84, %92#3, %85, %86, %87, %88, %89, %92#4, %90, %92#5, %91, %62#9, %62#11, %73#0, %64, %65, %66, %67, %68, %69, %70, %73#1, %71, %73#2, %72, %562, %62#10, %73#3, %92#6, %107, %512, %560, %92#9, %92#10, %73#4, %73#5, %73#6 : i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64
      }
    }
    %22 = memref.load %alloc_5[%c900] : memref<1xi32>
    %23 = arith.cmpi ne, %22, %c0_i32 : i32
    scf.if %23 {
      asccube.sync_all
      asccube.sync_block_wait s, %c0_i64
      asccube.sync_block_arrive fixpipe, %c1_i64 {mode = 0 : i32}
      asccube.sync_block_wait s, %c1_i64 {mode = 0 : i32}
      asccube.sync_block_arrive mte3, %c2_i64
    }
    asccube.sync_intra_wait fixpipe, %c0_i64
    asccube.sync_intra_wait fixpipe, %c16_i64
    asccube.sync_intra_wait fixpipe, %c1_i64
    asccube.sync_intra_wait fixpipe, %c17_i64
    asccube.sync_intra_wait fixpipe, %c2_i64
    asccube.sync_intra_wait fixpipe, %c18_i64
    return
  }
}
