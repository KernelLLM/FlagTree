module {
  func.func @run_qr_rope_09955ed81558b7972789964d6257885b(%arg0: !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg1: !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg2: !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg3: !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg4: !cannir.tensor<<gm, bf16>, <<(-1, -1, 512)>, <(-1, 512, 1)>>, <<(-1, -1, 512)>, <(-1, 512, 1)>>, identity>, %arg5: !cannir.tensor<<gm, f32>, <<(1, -1)>, <(-1, 1)>>, <<(1, -1)>, <(-1, 1)>>, identity>, %arg6: !cannir.tensor<<gm, f32>, <<(1, 512)>, <(512, 1)>>, <<(1, 512)>, <(512, 1)>>, identity>, %arg7: !cannir.tensor<<gm, bf16>, <<(-1, -1, 1, -1)>, <(-1, -1, -1, 1)>>, <<(-1, -1, 1, -1)>, <(-1, -1, -1, 1)>>, identity>, %arg8: !cannir.tensor<<gm, i64>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg9: !cannir.tensor<<gm, bf16>, <<(-1, 64)>, <(64, 1)>>, <<(-1, 64)>, <(64, 1)>>, identity>, %arg10: !cannir.tensor<<gm, bf16>, <<(-1, 64)>, <(64, 1)>>, <<(-1, 64)>, <(64, 1)>>, identity>, %arg11: !cannir.tensor<<gm, bf16>, <<(8, -1, -1)>, <(-1, -1, 1)>>, <<(8, -1, -1)>, <(-1, -1, 1)>>, identity>, %arg12: !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg13: !cannir.tensor<<gm, bf16>, <<(8, -1)>, <(-1, 1)>>, <<(8, -1)>, <(-1, 1)>>, identity>, %arg14: !cannir.tensor<<gm, f32>, <<(32, -1)>, <(-1, 1)>>, <<(32, -1)>, <(-1, 1)>>, identity>, %arg15: !cannir.tensor<<gm, bf16>, <<(8, -1)>, <(-1, 1)>>, <<(8, -1)>, <(-1, 1)>>, identity>, %arg16: !cannir.tensor<<gm, bf16>, <<(8, -1)>, <(-1, 1)>>, <<(8, -1)>, <(-1, 1)>>, identity>, %arg17: !cannir.tensor<<gm, bf16>, <<(8, -1)>, <(-1, 1)>>, <<(8, -1)>, <(-1, 1)>>, identity>, %arg18: f32, %arg19: f32, %arg20: f32, %arg21: i64, %arg22: i64, %arg23: i64) attributes {cannbotdsl.contract_digest = "09955ed81558b7972789964d6257885b0ca2386804b77e81ac35fe470c0847ef", cannbotdsl.public_slot_count = 24 : i64, cannbotdsl.slot_dtype_bits = [16, 16, 16, 16, 16, 32, 32, 16, 64, 16, 16, 16, 16, 16, 32, 16, 16, 16, 32, 32, 32, 64, 64, 64], cannbotdsl.slot_dtype_codes = [4, 4, 4, 4, 4, 2, 2, 4, 0, 4, 4, 4, 4, 4, 2, 4, 4, 4, 2, 2, 2, 0, 0, 0], cannbotdsl.slot_dtype_lanes = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], cannbotdsl.slot_flags = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], cannbotdsl.slot_kinds = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2], cannbotdsl.slot_type_digests = ["022b75a13987d7aef004e8fde8311ec1df5da13b3074eb5361860dc8ef2ecc8c", "b5c570c7c0159fc7a1075fa80de10b11b86c087e2cd3b2b946c60eb5f900ca19", "4d18a67066e55c86778cb78c908a80cc34547a2314d6ad8a79685f46673a5f89", "19b7e7050b7764b1c88f44198d5c4406c2e0c098dda1763d7bfe93f39fe0ae8a", "fd9e7bf46e5fdebb23db2cd9b9963e60289355042c04954fef2defa9b5be4007", "c8dd513e68d0aedae9a8e8521f33690b2fc63ba3dec5c2466e442b3a6bdebcad", "7b962932eb0067670c47bf914f64cc35656e412a09850be9b2bc297c5b918fc3", "fb792c582dd4611121a7cb3b62874165486549686f3034fbf3fe80e2ce472c27", "5ae7da10e83ac3f5ad55bbe47c978e5c8bcb0ad1d2171af81149eef24de96b7a", "6b6a90f195a5f29ca792af7b1733c8cafee2bfdeb6c69f295bfd0b7b7ca2d0e2", "6b6a90f195a5f29ca792af7b1733c8cafee2bfdeb6c69f295bfd0b7b7ca2d0e2", "13c55ee263fb1de89af4a17e85a06e57e0a80322131b4a24877e6239a1cd9823", "94630781bac8f1d566d1cf60e3b7a788d8ab31b7d3bd29e3f0948835334a7389", "04b43b0d54692facb90dd4907abdcbea2a855209f41c8d7117c4541d045feadb", "71457ab3f054c0e3c98727cb61d4f911f9bfd02151d9e1818a29adf3358c77f3", "e193cc2cd4f1cc826799f79ad2e88cbae4821553fdf9ffb7bfb8f20a2744fda9", "d086e48705d567aaf08e28892867093bc7951876e29b4e73f1528d2f336addf1", "f25adc241f574dff5e15cbbf47b60c522ade173908cfe64a7a59f8bcceebd153", "4a5363e2ec1eb8cefee037ab0b231653c07f9b44a406ef92ed9ba3e6f5541dbf", "4a5363e2ec1eb8cefee037ab0b231653c07f9b44a406ef92ed9ba3e6f5541dbf", "4a5363e2ec1eb8cefee037ab0b231653c07f9b44a406ef92ed9ba3e6f5541dbf", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515"], cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.host} {
    cannir.kernel_launch @kernel_MlaPrologKernel_run_qr_rope_0(%arg23) [%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18, %arg19, %arg20, %arg21, %arg22] {cannbotdsl.profile = {indices = array<i64: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17>, name = "MlaPrologKernel_run_qr_rope", op_type = "MlaPrologKernel_run_qr_rope", roles = array<i64: 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1>}} : !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, !cannir.tensor<<gm, bf16>, <<(-1, -1, 512)>, <(-1, 512, 1)>>, <<(-1, -1, 512)>, <(-1, 512, 1)>>, identity>, !cannir.tensor<<gm, f32>, <<(1, -1)>, <(-1, 1)>>, <<(1, -1)>, <(-1, 1)>>, identity>, !cannir.tensor<<gm, f32>, <<(1, 512)>, <(512, 1)>>, <<(1, 512)>, <(512, 1)>>, identity>, !cannir.tensor<<gm, bf16>, <<(-1, -1, 1, -1)>, <(-1, -1, -1, 1)>>, <<(-1, -1, 1, -1)>, <(-1, -1, -1, 1)>>, identity>, !cannir.tensor<<gm, i64>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, !cannir.tensor<<gm, bf16>, <<(-1, 64)>, <(64, 1)>>, <<(-1, 64)>, <(64, 1)>>, identity>, !cannir.tensor<<gm, bf16>, <<(-1, 64)>, <(64, 1)>>, <<(-1, 64)>, <(64, 1)>>, identity>, !cannir.tensor<<gm, bf16>, <<(8, -1, -1)>, <(-1, -1, 1)>>, <<(8, -1, -1)>, <(-1, -1, 1)>>, identity>, !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, !cannir.tensor<<gm, bf16>, <<(8, -1)>, <(-1, 1)>>, <<(8, -1)>, <(-1, 1)>>, identity>, !cannir.tensor<<gm, f32>, <<(32, -1)>, <(-1, 1)>>, <<(32, -1)>, <(-1, 1)>>, identity>, !cannir.tensor<<gm, bf16>, <<(8, -1)>, <(-1, 1)>>, <<(8, -1)>, <(-1, 1)>>, identity>, !cannir.tensor<<gm, bf16>, <<(8, -1)>, <(-1, 1)>>, <<(8, -1)>, <(-1, 1)>>, identity>, !cannir.tensor<<gm, bf16>, <<(8, -1)>, <(-1, 1)>>, <<(8, -1)>, <(-1, 1)>>, identity>, f32, f32, f32, i64, i64
    return
  }
  func.func @kernel_MlaPrologKernel_run_qr_rope_0_aiv(%arg0: !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg1: !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg2: !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg3: !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg4: !cannir.tensor<<gm, bf16>, <<(-1, -1, 512)>, <(-1, 512, 1)>>, <<(-1, -1, 512)>, <(-1, 512, 1)>>, identity>, %arg5: !cannir.tensor<<gm, f32>, <<(1, -1)>, <(-1, 1)>>, <<(1, -1)>, <(-1, 1)>>, identity>, %arg6: !cannir.tensor<<gm, f32>, <<(1, 512)>, <(512, 1)>>, <<(1, 512)>, <(512, 1)>>, identity>, %arg7: !cannir.tensor<<gm, bf16>, <<(-1, -1, 1, -1)>, <(-1, -1, -1, 1)>>, <<(-1, -1, 1, -1)>, <(-1, -1, -1, 1)>>, identity>, %arg8: !cannir.tensor<<gm, i64>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg9: !cannir.tensor<<gm, bf16>, <<(-1, 64)>, <(64, 1)>>, <<(-1, 64)>, <(64, 1)>>, identity>, %arg10: !cannir.tensor<<gm, bf16>, <<(-1, 64)>, <(64, 1)>>, <<(-1, 64)>, <(64, 1)>>, identity>, %arg11: !cannir.tensor<<gm, bf16>, <<(8, -1, -1)>, <(-1, -1, 1)>>, <<(8, -1, -1)>, <(-1, -1, 1)>>, identity>, %arg12: !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg13: !cannir.tensor<<gm, bf16>, <<(8, -1)>, <(-1, 1)>>, <<(8, -1)>, <(-1, 1)>>, identity>, %arg14: !cannir.tensor<<gm, f32>, <<(32, -1)>, <(-1, 1)>>, <<(32, -1)>, <(-1, 1)>>, identity>, %arg15: !cannir.tensor<<gm, bf16>, <<(8, -1)>, <(-1, 1)>>, <<(8, -1)>, <(-1, 1)>>, identity>, %arg16: !cannir.tensor<<gm, bf16>, <<(8, -1)>, <(-1, 1)>>, <<(8, -1)>, <(-1, 1)>>, identity>, %arg17: !cannir.tensor<<gm, bf16>, <<(8, -1)>, <(-1, 1)>>, <<(8, -1)>, <(-1, 1)>>, identity>, %arg18: f32, %arg19: f32, %arg20: f32, %arg21: i64, %arg22: i64) attributes {cannir.buffer.auto_sync, cannir.buffer.sync_analyzed, cannir.buffer.sync_verified, cannir.buffer.tensor_state, cannir.core = "aiv", cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.device, cannir.explicit_ring_buffer_cursors, cannir.kernel_group = "kernel_MlaPrologKernel_run_qr_rope_0", cannir.resource.ids_prepared} {
    %c512_i64 = arith.constant 512 : i64
    %c0_i32 = arith.constant 0 : i32
    %c16384_i64 = arith.constant 16384 : i64
    %c128_i32 = arith.constant 128 : i32
    %c0 = arith.constant 0 : index
    %c256_i64 = arith.constant 256 : i64
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 1.000000e+00 : f32
    %c8192_i64 = arith.constant 8192 : i64
    %c24576_i64 = arith.constant 24576 : i64
    %c64_i32 = arith.constant 64 : i32
    %c192_i32 = arith.constant 192 : i32
    %c256_i32 = arith.constant 256 : i32
    %c320_i32 = arith.constant 320 : i32
    %c384_i32 = arith.constant 384 : i32
    %c448_i32 = arith.constant 448 : i32
    %c512_i32 = arith.constant 512 : i32
    %c576_i32 = arith.constant 576 : i32
    %c640_i32 = arith.constant 640 : i32
    %c704_i32 = arith.constant 704 : i32
    %c768_i32 = arith.constant 768 : i32
    %c832_i32 = arith.constant 832 : i32
    %c896_i32 = arith.constant 896 : i32
    %c960_i32 = arith.constant 960 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %c1088_i32 = arith.constant 1088 : i32
    %c1152_i32 = arith.constant 1152 : i32
    %c1216_i32 = arith.constant 1216 : i32
    %c1280_i32 = arith.constant 1280 : i32
    %c1344_i32 = arith.constant 1344 : i32
    %c1408_i32 = arith.constant 1408 : i32
    %c1472_i32 = arith.constant 1472 : i32
    %c1536_i32 = arith.constant 1536 : i32
    %c1600_i32 = arith.constant 1600 : i32
    %c1664_i32 = arith.constant 1664 : i32
    %c1728_i32 = arith.constant 1728 : i32
    %c1792_i32 = arith.constant 1792 : i32
    %c1856_i32 = arith.constant 1856 : i32
    %c1920_i32 = arith.constant 1920 : i32
    %c1984_i32 = arith.constant 1984 : i32
    %c2048_i64 = arith.constant 2048 : i64
    %c1152_i64 = arith.constant 1152 : i64
    %c128_i64 = arith.constant 128 : i64
    %c576_i64 = arith.constant 576 : i64
    %c2 = arith.constant 2 : index
    %c64_i64 = arith.constant 64 : i64
    %c36_i64 = arith.constant 36 : i64
    %c32_i64 = arith.constant 32 : i64
    %c24_i64 = arith.constant 24 : i64
    %c16_i64 = arith.constant 16 : i64
    %c15_i64 = arith.constant 15 : i64
    %c14_i64 = arith.constant 14 : i64
    %c13_i64 = arith.constant 13 : i64
    %c8_i64 = arith.constant 8 : i64
    %c6_i64 = arith.constant 6 : i64
    %c5_i64 = arith.constant 5 : i64
    %c4_i64 = arith.constant 4 : i64
    %c3_i64 = arith.constant 3 : i64
    %c2_i64 = arith.constant 2 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %alloc = memref.alloc() {cannir.arg_index = 0 : i64} : memref<1xbf16>
    %alloc_0 = memref.alloc() {cannir.arg_index = 5 : i64} : memref<1xf32>
    %alloc_1 = memref.alloc() {cannir.arg_index = 6 : i64} : memref<512xf32>
    %alloc_2 = memref.alloc() {cannir.arg_index = 7 : i64} : memref<1xbf16>
    %alloc_3 = memref.alloc() {cannir.arg_index = 8 : i64} : memref<1xi64>
    %alloc_4 = memref.alloc() {cannir.arg_index = 9 : i64} : memref<1xbf16>
    %alloc_5 = memref.alloc() {cannir.arg_index = 10 : i64} : memref<1xbf16>
    %alloc_6 = memref.alloc() {cannir.arg_index = 11 : i64} : memref<1xbf16>
    %alloc_7 = memref.alloc() {cannir.arg_index = 12 : i64} : memref<1xbf16>
    %alloc_8 = memref.alloc() {cannir.arg_index = 13 : i64} : memref<1xbf16>
    %alloc_9 = memref.alloc() {cannir.arg_index = 14 : i64} : memref<1xf32>
    %alloc_10 = memref.alloc() {cannir.arg_index = 15 : i64} : memref<1xbf16>
    %alloc_11 = memref.alloc() {cannir.arg_index = 16 : i64} : memref<1xbf16>
    %alloc_12 = memref.alloc() {cannir.arg_index = 17 : i64} : memref<1xbf16>
    %alloc_13 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<8192xbf16, 1>
    %alloc_14 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<8192xf32, 1>
    %alloc_15 = memref.alloc() {cannir.byte_offset = 32768 : i64} : memref<8192xf32, 1>
    %alloc_16 = memref.alloc() {cannir.byte_offset = 65536 : i64} : memref<2048xf32, 1>
    %alloc_17 = memref.alloc() {cannir.byte_offset = 73728 : i64} : memref<2048xf32, 1>
    %alloc_18 = memref.alloc() {cannir.byte_offset = 81920 : i64} : memref<2048xf32, 1>
    %alloc_19 = memref.alloc() {cannir.byte_offset = 106496 : i64} : memref<2048xbf16, 1>
    %alloc_20 = memref.alloc() {cannir.byte_offset = 110592 : i64} : memref<2048xbf16, 1>
    %alloc_21 = memref.alloc() {cannir.byte_offset = 114688 : i64} : memref<8xf32, 1>
    %alloc_22 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<576xbf16, 1>
    %alloc_23 = memref.alloc() {cannir.byte_offset = 1152 : i64} : memref<576xbf16, 1>
    %alloc_24 = memref.alloc() {cannir.byte_offset = 2304 : i64} : memref<576xf32, 1>
    %alloc_25 = memref.alloc() {cannir.byte_offset = 4608 : i64} : memref<576xf32, 1>
    %alloc_26 = memref.alloc() {cannir.byte_offset = 6912 : i64} : memref<512xf32, 1>
    %alloc_27 = memref.alloc() {cannir.byte_offset = 8960 : i64} : memref<576xbf16, 1>
    %alloc_28 = memref.alloc() {cannir.byte_offset = 10112 : i64} : memref<576xbf16, 1>
    %alloc_29 = memref.alloc() {cannir.byte_offset = 14912 : i64} : memref<64xf32, 1>
    %alloc_30 = memref.alloc() {cannir.byte_offset = 15200 : i64} : memref<64xbf16, 1>
    %alloc_31 = memref.alloc() {cannir.byte_offset = 15328 : i64} : memref<64xbf16, 1>
    %alloc_32 = memref.alloc() {cannir.byte_offset = 15456 : i64} : memref<128xf32, 1>
    %alloc_33 = memref.alloc() {cannir.byte_offset = 15968 : i64} : memref<128xf32, 1>
    %alloc_34 = memref.alloc() {cannir.byte_offset = 16480 : i64} : memref<128xf32, 1>
    %alloc_35 = memref.alloc() {cannir.byte_offset = 16992 : i64} : memref<64xf32, 1>
    %alloc_36 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<64xbf16, 1>
    %alloc_37 = memref.alloc() {cannir.byte_offset = 128 : i64} : memref<64xbf16, 1>
    %alloc_38 = memref.alloc() {cannir.byte_offset = 256 : i64} : memref<64xf32, 1>
    %alloc_39 = memref.alloc() {cannir.byte_offset = 512 : i64} : memref<64xbf16, 1>
    %alloc_40 = memref.alloc() {cannir.byte_offset = 640 : i64} : memref<64xbf16, 1>
    %alloc_41 = memref.alloc() {cannir.byte_offset = 768 : i64} : memref<64xbf16, 1>
    %alloc_42 = memref.alloc() {cannir.byte_offset = 896 : i64} : memref<64xbf16, 1>
    %alloc_43 = memref.alloc() {cannir.byte_offset = 1024 : i64} : memref<128xf32, 1>
    %alloc_44 = memref.alloc() {cannir.byte_offset = 1536 : i64} : memref<128xf32, 1>
    %alloc_45 = memref.alloc() {cannir.byte_offset = 2048 : i64} : memref<128xf32, 1>
    %alloc_46 = memref.alloc() {cannir.byte_offset = 2560 : i64} : memref<64xf32, 1>
    %0 = ascscalar.get_block_idx
    %1 = ascscalar.get_block_num
    %2 = ascscalar.get_subblock_id
    %3 = cannir.extract_layout %arg0 : !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
    %4 = cannir.extract_shape %3 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, -1)>
    %5 = cannir.extract_dim %4[0] : !cannir.shape<(-1, -1)>
    %6 = cannir.extract_dim %4[1] : !cannir.shape<(-1, -1)>
    %7 = cannir.extract_layout %arg1 : !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
    %8 = cannir.extract_shape %7 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, -1)>
    %9 = cannir.extract_dim %8[0] : !cannir.shape<(-1, -1)>
    %10 = cannir.extract_layout %arg3 : !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
    %11 = cannir.extract_shape %10 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, -1)>
    %12 = cannir.extract_dim %11[0] : !cannir.shape<(-1, -1)>
    %13 = cannir.extract_layout %arg4 : !cannir.tensor<<gm, bf16>, <<(-1, -1, 512)>, <(-1, 512, 1)>>, <<(-1, -1, 512)>, <(-1, 512, 1)>>, identity>
    %14 = cannir.extract_shape %13 : <<(-1, -1, 512)>, <(-1, 512, 1)>> -> <(-1, -1, 512)>
    %15 = cannir.extract_dim %14[0] : !cannir.shape<(-1, -1, 512)>
    %16 = cannir.extract_dim %14[1] : !cannir.shape<(-1, -1, 512)>
    %17 = arith.addi %16, %12 : i64
    %18 = arith.subi %17, %c512_i64 : i64
    %19 = arith.muli %0, %c2_i64 : i64
    %20 = arith.addi %19, %2 : i64
    %21 = arith.cmpi ne, %5, %c8_i64 : i64
    scf.if %21 {
      %34 = arith.muli %1, %c2_i64 : i64
      scf.for %arg23 = %20 to %c8_i64 step %34  : i64 {
        %35 = arith.cmpi slt, %arg23, %5 : i64
        scf.if %35 {
          %36 = ascscalar.bitcast_memref %alloc_13 byte_offset = %c0_i64 : memref<8192xbf16, 1> -> memref<?xbf16, 1>
          %37 = arith.minsi %arg23, %5 : i64
          %38 = arith.subi %5, %37 : i64
          %39 = arith.minsi %38, %c1_i64 : i64
          %40 = arith.cmpi sle, %arg23, %5 : i64
          %41 = arith.select %40, %39, %c0_i64 : i64
          %42 = cannir.extract_physical_layout %arg0 : !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
          %43 = cannir.extract_stride %42 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
          %44 = cannir.extract_dim %43[0] : !cannir.stride<(-1, 1)>
          %45 = arith.muli %44, %c2_i64 : i64
          %46 = arith.maxsi %6, %c1_i64 : i64
          %47 = arith.divsi %6, %46 : i64
          %48 = arith.minsi %47, %c0_i64 : i64
          %49 = arith.muli %48, %6 : i64
          %50 = arith.subi %6, %49 : i64
          %51 = arith.minsi %6, %50 : i64
          %52 = arith.cmpi sge, %47, %c0_i64 : i64
          %53 = arith.select %52, %51, %c0_i64 : i64
          %54 = arith.muli %arg23, %44 : i64
          %55 = arith.muli %54, %c2_i64 : i64
          %56 = arith.muli %53, %c2_i64 : i64
          %57 = arith.cmpi eq, %56, %c0_i64 : i64
          %58 = arith.select %57, %c0_i64, %41 : i64
          %59 = arith.select %57, %c2_i64, %56 : i64
          ascvec.get_buf mte2, %c0_i64, 0 {cannir.buffer.generated_sync}
          "ascvec.copy_gm2ub_plan"(%36, %alloc, %c0_i32, %c0_i32, %c0_i64, %55, %58, %59, %45, %c16384_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = 0 : i64, ub_capacity = 16384 : i64}> : (memref<?xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
          ascvec.release_buf mte2, %c0_i64, 0 {cannir.buffer.generated_sync}
          %60 = cannir.extract_physical_layout %arg13 : !cannir.tensor<<gm, bf16>, <<(8, -1)>, <(-1, 1)>>, <<(8, -1)>, <(-1, 1)>>, identity>
          %61 = cannir.extract_stride %60 : <<(8, -1)>, <(-1, 1)>> -> <(-1, 1)>
          %62 = cannir.extract_dim %61[0] : !cannir.stride<(-1, 1)>
          %63 = arith.muli %arg23, %62 : i64
          %64 = arith.muli %63, %c16_i64 : i64
          %65 = ascscalar.memref_address %alloc_8 : memref<1xbf16>
          %66 = arith.divui %64, %c8_i64 : i64
          %67 = arith.addi %65, %66 : i64
          %68 = ascscalar.bind_gm_ptr %67 : memref<1xbf16>
          %69 = arith.index_cast %6 : i64 to index
          %70 = cannir.extract_layout %arg13 : !cannir.tensor<<gm, bf16>, <<(8, -1)>, <(-1, 1)>>, <<(8, -1)>, <(-1, 1)>>, identity>
          %71 = cannir.extract_stride %70 : <<(8, -1)>, <(-1, 1)>> -> <(-1, 1)>
          %72 = cannir.extract_dim %71[0] : !cannir.stride<(-1, 1)>
          %73 = arith.index_cast %72 : i64 to index
          %74 = arith.muli %69, %c2 : index
          %75 = arith.muli %73, %c2 : index
          %76 = arith.cmpi ugt, %6, %c0_i64 : i64
          scf.if %76 {
            %77 = arith.index_cast %74 : index to i64
            %78 = arith.index_cast %75 : index to i64
            ascvec.get_buf mte3, %c0_i64, 0 {cannir.buffer.generated_sync}
            %79 = arith.trunci %77 : i64 to i32
            ascvec.copy_ub2gm %68[%c0], %36[%c0], %c1_i32, %79, %78, %79 l2_cache_ctl %c0_i32 : memref<1xbf16>, memref<?xbf16, 1>
            ascvec.release_buf mte3, %c0_i64, 0 {cannir.buffer.generated_sync}
          }
        } else {
          ascvec.get_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
          ascvec.vf(%alloc_13 : memref<8192xbf16, 1>) {
          ^bb0(%arg24: memref<8192xbf16, 1>):
            %c8064 = arith.constant 8064 : index
            %c7936 = arith.constant 7936 : index
            %c7808 = arith.constant 7808 : index
            %c7680 = arith.constant 7680 : index
            %c7552 = arith.constant 7552 : index
            %c7424 = arith.constant 7424 : index
            %c7296 = arith.constant 7296 : index
            %c7168 = arith.constant 7168 : index
            %c7040 = arith.constant 7040 : index
            %c6912 = arith.constant 6912 : index
            %c6784 = arith.constant 6784 : index
            %c6656 = arith.constant 6656 : index
            %c6528 = arith.constant 6528 : index
            %c6400 = arith.constant 6400 : index
            %c6272 = arith.constant 6272 : index
            %c6144 = arith.constant 6144 : index
            %c6016 = arith.constant 6016 : index
            %c5888 = arith.constant 5888 : index
            %c5760 = arith.constant 5760 : index
            %c5632 = arith.constant 5632 : index
            %c5504 = arith.constant 5504 : index
            %c5376 = arith.constant 5376 : index
            %c5248 = arith.constant 5248 : index
            %c5120 = arith.constant 5120 : index
            %c4992 = arith.constant 4992 : index
            %c4864 = arith.constant 4864 : index
            %c4736 = arith.constant 4736 : index
            %c4608 = arith.constant 4608 : index
            %c4480 = arith.constant 4480 : index
            %c4352 = arith.constant 4352 : index
            %c4224 = arith.constant 4224 : index
            %c4096 = arith.constant 4096 : index
            %c3968 = arith.constant 3968 : index
            %c3840 = arith.constant 3840 : index
            %c3712 = arith.constant 3712 : index
            %c3584 = arith.constant 3584 : index
            %c3456 = arith.constant 3456 : index
            %c3328 = arith.constant 3328 : index
            %c3200 = arith.constant 3200 : index
            %c3072 = arith.constant 3072 : index
            %c2944 = arith.constant 2944 : index
            %c2816 = arith.constant 2816 : index
            %c2688 = arith.constant 2688 : index
            %c2560 = arith.constant 2560 : index
            %c2432 = arith.constant 2432 : index
            %c2304 = arith.constant 2304 : index
            %c2176 = arith.constant 2176 : index
            %c2048 = arith.constant 2048 : index
            %c1920 = arith.constant 1920 : index
            %c1792 = arith.constant 1792 : index
            %c1664 = arith.constant 1664 : index
            %c1536 = arith.constant 1536 : index
            %c1408 = arith.constant 1408 : index
            %c1280 = arith.constant 1280 : index
            %c1152 = arith.constant 1152 : index
            %c1024 = arith.constant 1024 : index
            %c896 = arith.constant 896 : index
            %c768 = arith.constant 768 : index
            %c640 = arith.constant 640 : index
            %c512 = arith.constant 512 : index
            %c384 = arith.constant 384 : index
            %c256 = arith.constant 256 : index
            %c128 = arith.constant 128 : index
            %c0_47 = arith.constant 0 : index
            %cst_48 = arith.constant 0.000000e+00 : bf16
            %c128_i32_49 = arith.constant 128 : i32
            %c0_50 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            scf.for %arg25 = %c0_50 to %c1 step %c1 {
              %mask, %remaining_out = ascvec.update_mask %c128_i32_49 elem_bits 16 : !ascvec.mask
              %54 = ascvec.duplicate_scalar %cst_48, %mask : bf16 -> <bf16>
              ascvec.storealign %arg24[%c0_47], %54, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg24[%c128], %54, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg24[%c256], %54, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg24[%c384], %54, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg24[%c512], %54, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg24[%c640], %54, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg24[%c768], %54, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg24[%c896], %54, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg24[%c1024], %54, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg24[%c1152], %54, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg24[%c1280], %54, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg24[%c1408], %54, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg24[%c1536], %54, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg24[%c1664], %54, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg24[%c1792], %54, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg24[%c1920], %54, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg24[%c2048], %54, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg24[%c2176], %54, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg24[%c2304], %54, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg24[%c2432], %54, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg24[%c2560], %54, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg24[%c2688], %54, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg24[%c2816], %54, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg24[%c2944], %54, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg24[%c3072], %54, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg24[%c3200], %54, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg24[%c3328], %54, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg24[%c3456], %54, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg24[%c3584], %54, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg24[%c3712], %54, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg24[%c3840], %54, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg24[%c3968], %54, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg24[%c4096], %54, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg24[%c4224], %54, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg24[%c4352], %54, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg24[%c4480], %54, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg24[%c4608], %54, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg24[%c4736], %54, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg24[%c4864], %54, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg24[%c4992], %54, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg24[%c5120], %54, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg24[%c5248], %54, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg24[%c5376], %54, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg24[%c5504], %54, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg24[%c5632], %54, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg24[%c5760], %54, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg24[%c5888], %54, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg24[%c6016], %54, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg24[%c6144], %54, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg24[%c6272], %54, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg24[%c6400], %54, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg24[%c6528], %54, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg24[%c6656], %54, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg24[%c6784], %54, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg24[%c6912], %54, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg24[%c7040], %54, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg24[%c7168], %54, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg24[%c7296], %54, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg24[%c7424], %54, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg24[%c7552], %54, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg24[%c7680], %54, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg24[%c7808], %54, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg24[%c7936], %54, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg24[%c8064], %54, %mask : memref<8192xbf16, 1>, index, <bf16>
            }
            ascvec.mem_bar vst_vld
          } {cannir.vf_mode = "simd"}
          ascvec.release_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
          %36 = cannir.extract_physical_layout %arg13 : !cannir.tensor<<gm, bf16>, <<(8, -1)>, <(-1, 1)>>, <<(8, -1)>, <(-1, 1)>>, identity>
          %37 = cannir.extract_stride %36 : <<(8, -1)>, <(-1, 1)>> -> <(-1, 1)>
          %38 = cannir.extract_dim %37[0] : !cannir.stride<(-1, 1)>
          %39 = arith.muli %arg23, %38 : i64
          %40 = arith.muli %39, %c16_i64 : i64
          %41 = ascscalar.memref_address %alloc_8 : memref<1xbf16>
          %42 = arith.divui %40, %c8_i64 : i64
          %43 = arith.addi %41, %42 : i64
          %44 = ascscalar.bind_gm_ptr %43 : memref<1xbf16>
          %45 = ascscalar.bitcast_memref %alloc_13 byte_offset = %c0_i64 : memref<8192xbf16, 1> -> memref<?xbf16, 1>
          %46 = arith.index_cast %6 : i64 to index
          %47 = cannir.extract_layout %arg13 : !cannir.tensor<<gm, bf16>, <<(8, -1)>, <(-1, 1)>>, <<(8, -1)>, <(-1, 1)>>, identity>
          %48 = cannir.extract_stride %47 : <<(8, -1)>, <(-1, 1)>> -> <(-1, 1)>
          %49 = cannir.extract_dim %48[0] : !cannir.stride<(-1, 1)>
          %50 = arith.index_cast %49 : i64 to index
          %51 = arith.muli %46, %c2 : index
          %52 = arith.muli %50, %c2 : index
          %53 = arith.cmpi ugt, %6, %c0_i64 : i64
          scf.if %53 {
            %54 = arith.index_cast %51 : index to i64
            %55 = arith.index_cast %52 : index to i64
            ascvec.get_buf mte3, %c0_i64, 0 {cannir.buffer.generated_sync}
            %56 = arith.trunci %54 : i64 to i32
            ascvec.copy_ub2gm %44[%c0], %45[%c0], %c1_i32, %56, %55, %56 l2_cache_ctl %c0_i32 : memref<1xbf16>, memref<?xbf16, 1>
            ascvec.release_buf mte3, %c0_i64, 0 {cannir.buffer.generated_sync}
          }
        }
      }
      ascvec.sync_all
      ascvec.sync_block_arrive mte3, %c13_i64
      ascvec.sync_block_arrive mte3, %c14_i64 {mode = 0 : i32}
      ascvec.sync_block_wait s, %c14_i64 {mode = 0 : i32}
      ascvec.sync_block_wait s, %c15_i64
    }
    ascvec.sync_all
    ascvec.sync_block_arrive mte3, %c13_i64
    ascvec.sync_block_arrive mte3, %c14_i64 {mode = 0 : i32}
    ascvec.sync_block_wait s, %c14_i64 {mode = 0 : i32}
    ascvec.sync_block_wait s, %c15_i64
    %22 = arith.cmpi slt, %20, %5 : i64
    %23 = scf.if %22 -> (i64) {
      %34 = arith.muli %1, %c2_i64 : i64
      %35 = arith.sitofp %9 : i64 to f32
      %36 = arith.divf %cst, %35 : f32
      %37 = ascscalar.bitcast_memref %alloc_18 byte_offset = %c0_i64 : memref<2048xf32, 1> -> memref<?xf32, 1>
      %38 = cannir.extract_layout %arg5 : !cannir.tensor<<gm, f32>, <<(1, -1)>, <(-1, 1)>>, <<(1, -1)>, <(-1, 1)>>, identity>
      %39 = cannir.extract_shape %38 : <<(1, -1)>, <(-1, 1)>> -> <(1, -1)>
      %40 = cannir.extract_dim %39[1] : !cannir.shape<(1, -1)>
      %41 = arith.maxsi %9, %c1_i64 : i64
      %42 = arith.divsi %40, %41 : i64
      %43 = arith.minsi %42, %c0_i64 : i64
      %44 = arith.muli %43, %9 : i64
      %45 = arith.subi %40, %44 : i64
      %46 = arith.minsi %9, %45 : i64
      %47 = arith.cmpi sge, %42, %c0_i64 : i64
      %48 = arith.select %47, %46, %c0_i64 : i64
      %49 = arith.muli %48, %c4_i64 : i64
      %50 = arith.cmpi eq, %49, %c0_i64 : i64
      %51 = arith.cmpi ne, %49, %c0_i64 : i64
      %52 = arith.extui %51 : i1 to i64
      %53 = arith.select %50, %c4_i64, %49 : i64
      ascvec.get_buf mte2, %c2_i64, 0 {cannir.buffer.generated_sync}
      "ascvec.copy_gm2ub_plan"(%37, %alloc_0, %c0_i32, %c0_i32, %c0_i64, %c0_i64, %52, %53, %c0_i64, %53, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 81920 : i64, ub_capacity = 8192 : i64}> : (memref<?xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
      ascvec.release_buf mte2, %c2_i64, 0 {cannir.buffer.generated_sync}
      %54 = scf.for %arg23 = %20 to %5 step %34 iter_args(%arg24 = %c0_i64) -> (i64)  : i64 {
        %55 = arith.cmpi eq, %arg24, %c0_i64 : i64
        %56 = ascscalar.memref_address %alloc_14 : memref<8192xf32, 1>
        %57 = ascscalar.memref_address %alloc_15 : memref<8192xf32, 1>
        %58 = arith.select %55, %56, %57 : i64
        %59 = arith.cmpi ne, %arg24, %c0_i64 : i64
        %60 = arith.extui %59 : i1 to i64
        %61 = arith.addi %arg24, %c1_i64 overflow<nsw> : i64
        %62 = arith.remui %61, %c2_i64 : i64
        %63 = arith.minsi %arg23, %c32_i64 : i64
        %64 = arith.subi %c32_i64, %63 : i64
        %65 = arith.minsi %64, %c1_i64 : i64
        %66 = arith.cmpi sle, %arg23, %c32_i64 : i64
        %67 = arith.select %66, %65, %c0_i64 : i64
        %68 = cannir.extract_physical_layout %arg14 : !cannir.tensor<<gm, f32>, <<(32, -1)>, <(-1, 1)>>, <<(32, -1)>, <(-1, 1)>>, identity>
        %69 = cannir.extract_stride %68 : <<(32, -1)>, <(-1, 1)>> -> <(-1, 1)>
        %70 = cannir.extract_dim %69[0] : !cannir.stride<(-1, 1)>
        %71 = arith.muli %70, %c4_i64 : i64
        %72 = cannir.extract_layout %arg14 : !cannir.tensor<<gm, f32>, <<(32, -1)>, <(-1, 1)>>, <<(32, -1)>, <(-1, 1)>>, identity>
        %73 = cannir.extract_shape %72 : <<(32, -1)>, <(-1, 1)>> -> <(32, -1)>
        %74 = cannir.extract_dim %73[1] : !cannir.shape<(32, -1)>
        %75 = arith.divsi %74, %41 : i64
        %76 = arith.minsi %75, %c0_i64 : i64
        %77 = arith.muli %76, %9 : i64
        %78 = arith.subi %74, %77 : i64
        %79 = arith.minsi %9, %78 : i64
        %80 = arith.cmpi sge, %75, %c0_i64 : i64
        %81 = arith.select %80, %79, %c0_i64 : i64
        %82 = arith.muli %arg23, %70 : i64
        %83 = arith.muli %82, %c4_i64 : i64
        %84 = arith.muli %81, %c4_i64 : i64
        %85 = arith.cmpi eq, %84, %c0_i64 : i64
        %86 = arith.select %85, %c0_i64, %67 : i64
        %87 = arith.select %85, %c4_i64, %84 : i64
        %88 = ascscalar.bind_local_ptr %58 : memref<1xf32, 1>
        ascvec.get_buf mte2, %60, 0 {cannir.buffer.generated_sync}
        "ascvec.copy_gm2ub_plan"(%88, %alloc_9, %c0_i32, %c0_i32, %c0_i64, %83, %86, %87, %71, %c8192_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = -1 : i64, ub_capacity = 4 : i64}> : (memref<1xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
        %89 = arith.addi %arg23, %c8_i64 : i64
        %90 = arith.minsi %89, %c32_i64 : i64
        %91 = arith.subi %c32_i64, %90 : i64
        %92 = arith.minsi %91, %c1_i64 : i64
        %93 = arith.cmpi sle, %89, %c32_i64 : i64
        %94 = arith.select %93, %92, %c0_i64 : i64
        %95 = arith.muli %89, %70 : i64
        %96 = arith.muli %95, %c4_i64 : i64
        %97 = arith.select %85, %c0_i64, %94 : i64
        "ascvec.copy_gm2ub_plan"(%88, %alloc_9, %c0_i32, %c0_i32, %c8192_i64, %96, %97, %87, %71, %c8192_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = -1 : i64, ub_capacity = 4 : i64}> : (memref<1xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
        %98 = arith.addi %arg23, %c16_i64 : i64
        %99 = arith.minsi %98, %c32_i64 : i64
        %100 = arith.subi %c32_i64, %99 : i64
        %101 = arith.minsi %100, %c1_i64 : i64
        %102 = arith.cmpi sle, %98, %c32_i64 : i64
        %103 = arith.select %102, %101, %c0_i64 : i64
        %104 = arith.muli %98, %70 : i64
        %105 = arith.muli %104, %c4_i64 : i64
        %106 = arith.select %85, %c0_i64, %103 : i64
        "ascvec.copy_gm2ub_plan"(%88, %alloc_9, %c0_i32, %c0_i32, %c16384_i64, %105, %106, %87, %71, %c8192_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = -1 : i64, ub_capacity = 4 : i64}> : (memref<1xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
        %107 = arith.addi %arg23, %c24_i64 : i64
        %108 = arith.minsi %107, %c32_i64 : i64
        %109 = arith.subi %c32_i64, %108 : i64
        %110 = arith.minsi %109, %c1_i64 : i64
        %111 = arith.cmpi sle, %107, %c32_i64 : i64
        %112 = arith.select %111, %110, %c0_i64 : i64
        %113 = arith.muli %107, %70 : i64
        %114 = arith.muli %113, %c4_i64 : i64
        %115 = arith.select %85, %c0_i64, %112 : i64
        "ascvec.copy_gm2ub_plan"(%88, %alloc_9, %c0_i32, %c0_i32, %c24576_i64, %114, %115, %87, %71, %c8192_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = -1 : i64, ub_capacity = 4 : i64}> : (memref<1xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
        ascvec.release_buf mte2, %60, 0 {cannir.buffer.generated_sync}
        %116 = ascscalar.bind_local_ptr %58 : memref<8192xf32, 1>
        %117 = ascscalar.memref_address %alloc_16 : memref<2048xf32, 1>
        %118 = ascscalar.memref_address %alloc_17 : memref<2048xf32, 1>
        %119 = arith.select %55, %117, %118 : i64
        %120 = ascscalar.bind_local_ptr %119 : memref<2048xf32, 1>
        %121 = arith.trunci %9 : i64 to i32
        ascvec.get_buf v, %60, 0 {cannir.buffer.generated_sync}
        ascvec.vf(%116, %120, %121 : memref<8192xf32, 1>, memref<2048xf32, 1>, i32) {
        ^bb0(%arg25: memref<8192xf32, 1>, %arg26: memref<2048xf32, 1>, %arg27: i32):
          %c64_i32_47 = arith.constant 64 : i32
          %c0_i32_48 = arith.constant 0 : i32
          %c6144_i32 = arith.constant 6144 : i32
          %c4096_i32 = arith.constant 4096 : i32
          %c2048_i32 = arith.constant 2048 : i32
          scf.for %arg28 = %c0_i32_48 to %arg27 step %c64_i32_47  : i32 {
            %254 = arith.index_cast %arg28 : i32 to index
            %255 = ascvec.loadalign %arg25[%254] : memref<8192xf32, 1>, index -> <f32>
            %256 = arith.addi %arg28, %c2048_i32 : i32
            %257 = arith.index_cast %256 : i32 to index
            %258 = ascvec.loadalign %arg25[%257] : memref<8192xf32, 1>, index -> <f32>
            %259 = ascvec.create_mask : !ascvec.mask
            %260 = ascvec.add %255, %258, %259 : <f32>
            %261 = arith.addi %arg28, %c4096_i32 : i32
            %262 = arith.index_cast %261 : i32 to index
            %263 = ascvec.loadalign %arg25[%262] : memref<8192xf32, 1>, index -> <f32>
            %264 = arith.addi %arg28, %c6144_i32 : i32
            %265 = arith.index_cast %264 : i32 to index
            %266 = ascvec.loadalign %arg25[%265] : memref<8192xf32, 1>, index -> <f32>
            %267 = ascvec.add %263, %266, %259 : <f32>
            %268 = ascvec.add %260, %267, %259 : <f32>
            %269 = ascvec.cast %268, %259 : <f32> -> <bf16>
            %270 = ascvec.cast %269, %259 {rounding = 6 : i32} : <bf16> -> <f32>
            ascvec.storealign %arg26[%254], %270, %259 : memref<2048xf32, 1>, index, <f32>
          }
          ascvec.mem_bar vst_vld
        } {cannir.vf_mode = "simd"}
        ascvec.release_buf v, %60, 0 {cannir.buffer.generated_sync}
        %122 = arith.maxsi %121, %c0_i32 : i32
        %123 = arith.minsi %122, %c64_i32 : i32
        %124 = arith.subi %121, %c64_i32 : i32
        %125 = arith.maxsi %124, %c0_i32 : i32
        %126 = arith.minsi %125, %c64_i32 : i32
        %127 = arith.subi %121, %c128_i32 : i32
        %128 = arith.maxsi %127, %c0_i32 : i32
        %129 = arith.minsi %128, %c64_i32 : i32
        %130 = arith.subi %121, %c192_i32 : i32
        %131 = arith.maxsi %130, %c0_i32 : i32
        %132 = arith.minsi %131, %c64_i32 : i32
        %133 = arith.subi %121, %c256_i32 : i32
        %134 = arith.maxsi %133, %c0_i32 : i32
        %135 = arith.minsi %134, %c64_i32 : i32
        %136 = arith.subi %121, %c320_i32 : i32
        %137 = arith.maxsi %136, %c0_i32 : i32
        %138 = arith.minsi %137, %c64_i32 : i32
        %139 = arith.subi %121, %c384_i32 : i32
        %140 = arith.maxsi %139, %c0_i32 : i32
        %141 = arith.minsi %140, %c64_i32 : i32
        %142 = arith.subi %121, %c448_i32 : i32
        %143 = arith.maxsi %142, %c0_i32 : i32
        %144 = arith.minsi %143, %c64_i32 : i32
        %145 = arith.subi %121, %c512_i32 : i32
        %146 = arith.maxsi %145, %c0_i32 : i32
        %147 = arith.minsi %146, %c64_i32 : i32
        %148 = arith.subi %121, %c576_i32 : i32
        %149 = arith.maxsi %148, %c0_i32 : i32
        %150 = arith.minsi %149, %c64_i32 : i32
        %151 = arith.subi %121, %c640_i32 : i32
        %152 = arith.maxsi %151, %c0_i32 : i32
        %153 = arith.minsi %152, %c64_i32 : i32
        %154 = arith.subi %121, %c704_i32 : i32
        %155 = arith.maxsi %154, %c0_i32 : i32
        %156 = arith.minsi %155, %c64_i32 : i32
        %157 = arith.subi %121, %c768_i32 : i32
        %158 = arith.maxsi %157, %c0_i32 : i32
        %159 = arith.minsi %158, %c64_i32 : i32
        %160 = arith.subi %121, %c832_i32 : i32
        %161 = arith.maxsi %160, %c0_i32 : i32
        %162 = arith.minsi %161, %c64_i32 : i32
        %163 = arith.subi %121, %c896_i32 : i32
        %164 = arith.maxsi %163, %c0_i32 : i32
        %165 = arith.minsi %164, %c64_i32 : i32
        %166 = arith.subi %121, %c960_i32 : i32
        %167 = arith.maxsi %166, %c0_i32 : i32
        %168 = arith.minsi %167, %c64_i32 : i32
        %169 = arith.subi %121, %c1024_i32 : i32
        %170 = arith.maxsi %169, %c0_i32 : i32
        %171 = arith.minsi %170, %c64_i32 : i32
        %172 = arith.subi %121, %c1088_i32 : i32
        %173 = arith.maxsi %172, %c0_i32 : i32
        %174 = arith.minsi %173, %c64_i32 : i32
        %175 = arith.subi %121, %c1152_i32 : i32
        %176 = arith.maxsi %175, %c0_i32 : i32
        %177 = arith.minsi %176, %c64_i32 : i32
        %178 = arith.subi %121, %c1216_i32 : i32
        %179 = arith.maxsi %178, %c0_i32 : i32
        %180 = arith.minsi %179, %c64_i32 : i32
        %181 = arith.subi %121, %c1280_i32 : i32
        %182 = arith.maxsi %181, %c0_i32 : i32
        %183 = arith.minsi %182, %c64_i32 : i32
        %184 = arith.subi %121, %c1344_i32 : i32
        %185 = arith.maxsi %184, %c0_i32 : i32
        %186 = arith.minsi %185, %c64_i32 : i32
        %187 = arith.subi %121, %c1408_i32 : i32
        %188 = arith.maxsi %187, %c0_i32 : i32
        %189 = arith.minsi %188, %c64_i32 : i32
        %190 = arith.subi %121, %c1472_i32 : i32
        %191 = arith.maxsi %190, %c0_i32 : i32
        %192 = arith.minsi %191, %c64_i32 : i32
        %193 = arith.subi %121, %c1536_i32 : i32
        %194 = arith.maxsi %193, %c0_i32 : i32
        %195 = arith.minsi %194, %c64_i32 : i32
        %196 = arith.subi %121, %c1600_i32 : i32
        %197 = arith.maxsi %196, %c0_i32 : i32
        %198 = arith.minsi %197, %c64_i32 : i32
        %199 = arith.subi %121, %c1664_i32 : i32
        %200 = arith.maxsi %199, %c0_i32 : i32
        %201 = arith.minsi %200, %c64_i32 : i32
        %202 = arith.subi %121, %c1728_i32 : i32
        %203 = arith.maxsi %202, %c0_i32 : i32
        %204 = arith.minsi %203, %c64_i32 : i32
        %205 = arith.subi %121, %c1792_i32 : i32
        %206 = arith.maxsi %205, %c0_i32 : i32
        %207 = arith.minsi %206, %c64_i32 : i32
        %208 = arith.subi %121, %c1856_i32 : i32
        %209 = arith.maxsi %208, %c0_i32 : i32
        %210 = arith.minsi %209, %c64_i32 : i32
        %211 = arith.subi %121, %c1920_i32 : i32
        %212 = arith.maxsi %211, %c0_i32 : i32
        %213 = arith.minsi %212, %c64_i32 : i32
        %214 = arith.subi %121, %c1984_i32 : i32
        %215 = arith.maxsi %214, %c0_i32 : i32
        %216 = arith.minsi %215, %c64_i32 : i32
        ascvec.vf(%123, %120, %126, %129, %132, %135, %138, %141, %144, %147, %150, %153, %156, %159, %162, %165, %168, %171, %174, %177, %180, %183, %186, %189, %192, %195, %198, %201, %204, %207, %210, %213, %216, %36, %arg18, %alloc_21 : i32, memref<2048xf32, 1>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, f32, f32, memref<8xf32, 1>) {
        ^bb0(%arg25: i32, %arg26: memref<2048xf32, 1>, %arg27: i32, %arg28: i32, %arg29: i32, %arg30: i32, %arg31: i32, %arg32: i32, %arg33: i32, %arg34: i32, %arg35: i32, %arg36: i32, %arg37: i32, %arg38: i32, %arg39: i32, %arg40: i32, %arg41: i32, %arg42: i32, %arg43: i32, %arg44: i32, %arg45: i32, %arg46: i32, %arg47: i32, %arg48: i32, %arg49: i32, %arg50: i32, %arg51: i32, %arg52: i32, %arg53: i32, %arg54: i32, %arg55: i32, %arg56: i32, %arg57: i32, %arg58: f32, %arg59: f32, %arg60: memref<8xf32, 1>):
          %c1984 = arith.constant 1984 : index
          %c1920 = arith.constant 1920 : index
          %c1856 = arith.constant 1856 : index
          %c1792 = arith.constant 1792 : index
          %c1728 = arith.constant 1728 : index
          %c1664 = arith.constant 1664 : index
          %c1600 = arith.constant 1600 : index
          %c1536 = arith.constant 1536 : index
          %c1472 = arith.constant 1472 : index
          %c1408 = arith.constant 1408 : index
          %c1344 = arith.constant 1344 : index
          %c1280 = arith.constant 1280 : index
          %c1216 = arith.constant 1216 : index
          %c1152 = arith.constant 1152 : index
          %c1088 = arith.constant 1088 : index
          %c1024 = arith.constant 1024 : index
          %c960 = arith.constant 960 : index
          %c896 = arith.constant 896 : index
          %c832 = arith.constant 832 : index
          %c768 = arith.constant 768 : index
          %c704 = arith.constant 704 : index
          %c640 = arith.constant 640 : index
          %c576 = arith.constant 576 : index
          %c512 = arith.constant 512 : index
          %c448 = arith.constant 448 : index
          %c384 = arith.constant 384 : index
          %c320 = arith.constant 320 : index
          %c256 = arith.constant 256 : index
          %c192 = arith.constant 192 : index
          %c128 = arith.constant 128 : index
          %c64 = arith.constant 64 : index
          %c0_47 = arith.constant 0 : index
          %cst_48 = arith.constant 0.000000e+00 : f32
          %c1_i32_49 = arith.constant 1 : i32
          %c0_50 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          scf.for %arg61 = %c0_50 to %c1 step %c1 {
            %mask, %remaining_out = ascvec.update_mask %c1_i32_49 : !ascvec.mask
            %254 = ascvec.duplicate_scalar %cst_48, %mask : f32 -> <f32>
            %mask_51, %remaining_out_52 = ascvec.update_mask %arg25 : !ascvec.mask
            %255 = ascvec.loadalign %arg26[%c0_47] : memref<2048xf32, 1>, index -> <f32>
            %256 = ascvec.mul %255, %255, %mask_51 : <f32>
            %257 = ascvec.reduce_sum %256, %mask_51 : <f32>
            %258 = ascvec.add %254, %257, %mask : <f32>
            %mask_53, %remaining_out_54 = ascvec.update_mask %arg27 : !ascvec.mask
            %259 = ascvec.loadalign %arg26[%c64] : memref<2048xf32, 1>, index -> <f32>
            %260 = ascvec.mul %259, %259, %mask_53 : <f32>
            %261 = ascvec.reduce_sum %260, %mask_53 : <f32>
            %262 = ascvec.add %258, %261, %mask : <f32>
            %mask_55, %remaining_out_56 = ascvec.update_mask %arg28 : !ascvec.mask
            %263 = ascvec.loadalign %arg26[%c128] : memref<2048xf32, 1>, index -> <f32>
            %264 = ascvec.mul %263, %263, %mask_55 : <f32>
            %265 = ascvec.reduce_sum %264, %mask_55 : <f32>
            %266 = ascvec.add %262, %265, %mask : <f32>
            %mask_57, %remaining_out_58 = ascvec.update_mask %arg29 : !ascvec.mask
            %267 = ascvec.loadalign %arg26[%c192] : memref<2048xf32, 1>, index -> <f32>
            %268 = ascvec.mul %267, %267, %mask_57 : <f32>
            %269 = ascvec.reduce_sum %268, %mask_57 : <f32>
            %270 = ascvec.add %266, %269, %mask : <f32>
            %mask_59, %remaining_out_60 = ascvec.update_mask %arg30 : !ascvec.mask
            %271 = ascvec.loadalign %arg26[%c256] : memref<2048xf32, 1>, index -> <f32>
            %272 = ascvec.mul %271, %271, %mask_59 : <f32>
            %273 = ascvec.reduce_sum %272, %mask_59 : <f32>
            %274 = ascvec.add %270, %273, %mask : <f32>
            %mask_61, %remaining_out_62 = ascvec.update_mask %arg31 : !ascvec.mask
            %275 = ascvec.loadalign %arg26[%c320] : memref<2048xf32, 1>, index -> <f32>
            %276 = ascvec.mul %275, %275, %mask_61 : <f32>
            %277 = ascvec.reduce_sum %276, %mask_61 : <f32>
            %278 = ascvec.add %274, %277, %mask : <f32>
            %mask_63, %remaining_out_64 = ascvec.update_mask %arg32 : !ascvec.mask
            %279 = ascvec.loadalign %arg26[%c384] : memref<2048xf32, 1>, index -> <f32>
            %280 = ascvec.mul %279, %279, %mask_63 : <f32>
            %281 = ascvec.reduce_sum %280, %mask_63 : <f32>
            %282 = ascvec.add %278, %281, %mask : <f32>
            %mask_65, %remaining_out_66 = ascvec.update_mask %arg33 : !ascvec.mask
            %283 = ascvec.loadalign %arg26[%c448] : memref<2048xf32, 1>, index -> <f32>
            %284 = ascvec.mul %283, %283, %mask_65 : <f32>
            %285 = ascvec.reduce_sum %284, %mask_65 : <f32>
            %286 = ascvec.add %282, %285, %mask : <f32>
            %mask_67, %remaining_out_68 = ascvec.update_mask %arg34 : !ascvec.mask
            %287 = ascvec.loadalign %arg26[%c512] : memref<2048xf32, 1>, index -> <f32>
            %288 = ascvec.mul %287, %287, %mask_67 : <f32>
            %289 = ascvec.reduce_sum %288, %mask_67 : <f32>
            %290 = ascvec.add %286, %289, %mask : <f32>
            %mask_69, %remaining_out_70 = ascvec.update_mask %arg35 : !ascvec.mask
            %291 = ascvec.loadalign %arg26[%c576] : memref<2048xf32, 1>, index -> <f32>
            %292 = ascvec.mul %291, %291, %mask_69 : <f32>
            %293 = ascvec.reduce_sum %292, %mask_69 : <f32>
            %294 = ascvec.add %290, %293, %mask : <f32>
            %mask_71, %remaining_out_72 = ascvec.update_mask %arg36 : !ascvec.mask
            %295 = ascvec.loadalign %arg26[%c640] : memref<2048xf32, 1>, index -> <f32>
            %296 = ascvec.mul %295, %295, %mask_71 : <f32>
            %297 = ascvec.reduce_sum %296, %mask_71 : <f32>
            %298 = ascvec.add %294, %297, %mask : <f32>
            %mask_73, %remaining_out_74 = ascvec.update_mask %arg37 : !ascvec.mask
            %299 = ascvec.loadalign %arg26[%c704] : memref<2048xf32, 1>, index -> <f32>
            %300 = ascvec.mul %299, %299, %mask_73 : <f32>
            %301 = ascvec.reduce_sum %300, %mask_73 : <f32>
            %302 = ascvec.add %298, %301, %mask : <f32>
            %mask_75, %remaining_out_76 = ascvec.update_mask %arg38 : !ascvec.mask
            %303 = ascvec.loadalign %arg26[%c768] : memref<2048xf32, 1>, index -> <f32>
            %304 = ascvec.mul %303, %303, %mask_75 : <f32>
            %305 = ascvec.reduce_sum %304, %mask_75 : <f32>
            %306 = ascvec.add %302, %305, %mask : <f32>
            %mask_77, %remaining_out_78 = ascvec.update_mask %arg39 : !ascvec.mask
            %307 = ascvec.loadalign %arg26[%c832] : memref<2048xf32, 1>, index -> <f32>
            %308 = ascvec.mul %307, %307, %mask_77 : <f32>
            %309 = ascvec.reduce_sum %308, %mask_77 : <f32>
            %310 = ascvec.add %306, %309, %mask : <f32>
            %mask_79, %remaining_out_80 = ascvec.update_mask %arg40 : !ascvec.mask
            %311 = ascvec.loadalign %arg26[%c896] : memref<2048xf32, 1>, index -> <f32>
            %312 = ascvec.mul %311, %311, %mask_79 : <f32>
            %313 = ascvec.reduce_sum %312, %mask_79 : <f32>
            %314 = ascvec.add %310, %313, %mask : <f32>
            %mask_81, %remaining_out_82 = ascvec.update_mask %arg41 : !ascvec.mask
            %315 = ascvec.loadalign %arg26[%c960] : memref<2048xf32, 1>, index -> <f32>
            %316 = ascvec.mul %315, %315, %mask_81 : <f32>
            %317 = ascvec.reduce_sum %316, %mask_81 : <f32>
            %318 = ascvec.add %314, %317, %mask : <f32>
            %mask_83, %remaining_out_84 = ascvec.update_mask %arg42 : !ascvec.mask
            %319 = ascvec.loadalign %arg26[%c1024] : memref<2048xf32, 1>, index -> <f32>
            %320 = ascvec.mul %319, %319, %mask_83 : <f32>
            %321 = ascvec.reduce_sum %320, %mask_83 : <f32>
            %322 = ascvec.add %318, %321, %mask : <f32>
            %mask_85, %remaining_out_86 = ascvec.update_mask %arg43 : !ascvec.mask
            %323 = ascvec.loadalign %arg26[%c1088] : memref<2048xf32, 1>, index -> <f32>
            %324 = ascvec.mul %323, %323, %mask_85 : <f32>
            %325 = ascvec.reduce_sum %324, %mask_85 : <f32>
            %326 = ascvec.add %322, %325, %mask : <f32>
            %mask_87, %remaining_out_88 = ascvec.update_mask %arg44 : !ascvec.mask
            %327 = ascvec.loadalign %arg26[%c1152] : memref<2048xf32, 1>, index -> <f32>
            %328 = ascvec.mul %327, %327, %mask_87 : <f32>
            %329 = ascvec.reduce_sum %328, %mask_87 : <f32>
            %330 = ascvec.add %326, %329, %mask : <f32>
            %mask_89, %remaining_out_90 = ascvec.update_mask %arg45 : !ascvec.mask
            %331 = ascvec.loadalign %arg26[%c1216] : memref<2048xf32, 1>, index -> <f32>
            %332 = ascvec.mul %331, %331, %mask_89 : <f32>
            %333 = ascvec.reduce_sum %332, %mask_89 : <f32>
            %334 = ascvec.add %330, %333, %mask : <f32>
            %mask_91, %remaining_out_92 = ascvec.update_mask %arg46 : !ascvec.mask
            %335 = ascvec.loadalign %arg26[%c1280] : memref<2048xf32, 1>, index -> <f32>
            %336 = ascvec.mul %335, %335, %mask_91 : <f32>
            %337 = ascvec.reduce_sum %336, %mask_91 : <f32>
            %338 = ascvec.add %334, %337, %mask : <f32>
            %mask_93, %remaining_out_94 = ascvec.update_mask %arg47 : !ascvec.mask
            %339 = ascvec.loadalign %arg26[%c1344] : memref<2048xf32, 1>, index -> <f32>
            %340 = ascvec.mul %339, %339, %mask_93 : <f32>
            %341 = ascvec.reduce_sum %340, %mask_93 : <f32>
            %342 = ascvec.add %338, %341, %mask : <f32>
            %mask_95, %remaining_out_96 = ascvec.update_mask %arg48 : !ascvec.mask
            %343 = ascvec.loadalign %arg26[%c1408] : memref<2048xf32, 1>, index -> <f32>
            %344 = ascvec.mul %343, %343, %mask_95 : <f32>
            %345 = ascvec.reduce_sum %344, %mask_95 : <f32>
            %346 = ascvec.add %342, %345, %mask : <f32>
            %mask_97, %remaining_out_98 = ascvec.update_mask %arg49 : !ascvec.mask
            %347 = ascvec.loadalign %arg26[%c1472] : memref<2048xf32, 1>, index -> <f32>
            %348 = ascvec.mul %347, %347, %mask_97 : <f32>
            %349 = ascvec.reduce_sum %348, %mask_97 : <f32>
            %350 = ascvec.add %346, %349, %mask : <f32>
            %mask_99, %remaining_out_100 = ascvec.update_mask %arg50 : !ascvec.mask
            %351 = ascvec.loadalign %arg26[%c1536] : memref<2048xf32, 1>, index -> <f32>
            %352 = ascvec.mul %351, %351, %mask_99 : <f32>
            %353 = ascvec.reduce_sum %352, %mask_99 : <f32>
            %354 = ascvec.add %350, %353, %mask : <f32>
            %mask_101, %remaining_out_102 = ascvec.update_mask %arg51 : !ascvec.mask
            %355 = ascvec.loadalign %arg26[%c1600] : memref<2048xf32, 1>, index -> <f32>
            %356 = ascvec.mul %355, %355, %mask_101 : <f32>
            %357 = ascvec.reduce_sum %356, %mask_101 : <f32>
            %358 = ascvec.add %354, %357, %mask : <f32>
            %mask_103, %remaining_out_104 = ascvec.update_mask %arg52 : !ascvec.mask
            %359 = ascvec.loadalign %arg26[%c1664] : memref<2048xf32, 1>, index -> <f32>
            %360 = ascvec.mul %359, %359, %mask_103 : <f32>
            %361 = ascvec.reduce_sum %360, %mask_103 : <f32>
            %362 = ascvec.add %358, %361, %mask : <f32>
            %mask_105, %remaining_out_106 = ascvec.update_mask %arg53 : !ascvec.mask
            %363 = ascvec.loadalign %arg26[%c1728] : memref<2048xf32, 1>, index -> <f32>
            %364 = ascvec.mul %363, %363, %mask_105 : <f32>
            %365 = ascvec.reduce_sum %364, %mask_105 : <f32>
            %366 = ascvec.add %362, %365, %mask : <f32>
            %mask_107, %remaining_out_108 = ascvec.update_mask %arg54 : !ascvec.mask
            %367 = ascvec.loadalign %arg26[%c1792] : memref<2048xf32, 1>, index -> <f32>
            %368 = ascvec.mul %367, %367, %mask_107 : <f32>
            %369 = ascvec.reduce_sum %368, %mask_107 : <f32>
            %370 = ascvec.add %366, %369, %mask : <f32>
            %mask_109, %remaining_out_110 = ascvec.update_mask %arg55 : !ascvec.mask
            %371 = ascvec.loadalign %arg26[%c1856] : memref<2048xf32, 1>, index -> <f32>
            %372 = ascvec.mul %371, %371, %mask_109 : <f32>
            %373 = ascvec.reduce_sum %372, %mask_109 : <f32>
            %374 = ascvec.add %370, %373, %mask : <f32>
            %mask_111, %remaining_out_112 = ascvec.update_mask %arg56 : !ascvec.mask
            %375 = ascvec.loadalign %arg26[%c1920] : memref<2048xf32, 1>, index -> <f32>
            %376 = ascvec.mul %375, %375, %mask_111 : <f32>
            %377 = ascvec.reduce_sum %376, %mask_111 : <f32>
            %378 = ascvec.add %374, %377, %mask : <f32>
            %mask_113, %remaining_out_114 = ascvec.update_mask %arg57 : !ascvec.mask
            %379 = ascvec.loadalign %arg26[%c1984] : memref<2048xf32, 1>, index -> <f32>
            %380 = ascvec.mul %379, %379, %mask_113 : <f32>
            %381 = ascvec.reduce_sum %380, %mask_113 : <f32>
            %382 = ascvec.add %378, %381, %mask : <f32>
            %383 = ascvec.mul_scalar %382, %arg58, %mask : <f32>, f32
            %384 = ascvec.add_scalar %383, %arg59, %mask : <f32>, f32
            %385 = ascvec.sqrt %384, %mask : <f32>
            ascvec.storealign_1st %arg60[%c0_47], %385 : memref<8xf32, 1>, index, <f32>
          }
          ascvec.mem_bar vst_vld
        } {cannir.vf_mode = "simd"}
        %217 = ascscalar.memref_address %alloc_19 : memref<2048xbf16, 1>
        %218 = ascscalar.memref_address %alloc_20 : memref<2048xbf16, 1>
        %219 = arith.select %55, %217, %218 : i64
        %220 = ascscalar.bind_local_ptr %219 : memref<2048xbf16, 1>
        %221 = arith.select %55, %c3_i64, %c4_i64 : i64
        ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
        ascvec.get_buf v, %221, 0 {cannir.buffer.generated_sync}
        ascvec.vf(%alloc_21, %120, %alloc_18, %arg19, %220, %121 : memref<8xf32, 1>, memref<2048xf32, 1>, memref<2048xf32, 1>, f32, memref<2048xbf16, 1>, i32) {
        ^bb0(%arg25: memref<8xf32, 1>, %arg26: memref<2048xf32, 1>, %arg27: memref<2048xf32, 1>, %arg28: f32, %arg29: memref<2048xbf16, 1>, %arg30: i32):
          %c64_i32_47 = arith.constant 64 : i32
          %c0_i32_48 = arith.constant 0 : i32
          %c0_49 = arith.constant 0 : index
          %254 = ascvec.loadalign_brc %arg25[%c0_49] : memref<8xf32, 1>, index -> <f32>
          scf.for %arg31 = %c0_i32_48 to %arg30 step %c64_i32_47  : i32 {
            %255 = arith.index_cast %arg31 : i32 to index
            %256 = ascvec.loadalign %arg26[%255] : memref<2048xf32, 1>, index -> <f32>
            %257 = ascvec.create_mask : !ascvec.mask
            %258 = ascvec.div %256, %254, %257 {precision_mode = true} : <f32>
            %259 = ascvec.loadalign %arg27[%255] : memref<2048xf32, 1>, index -> <f32>
            %260 = ascvec.mul %258, %259, %257 : <f32>
            %261 = ascvec.mul_scalar %260, %arg28, %257 : <f32>, f32
            %262 = ascvec.cast %261, %257 : <f32> -> <bf16>
            ascvec.storealign_pack %arg29[%255], %262, %257 {mode = "b32_to_b16"} : memref<2048xbf16, 1>, index, <bf16>
          }
          ascvec.mem_bar vst_vld
        } {cannir.vf_mode = "simd"}
        ascvec.release_buf v, %221, 0 {cannir.buffer.generated_sync}
        ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
        %222 = cannir.extract_physical_layout %arg17 : !cannir.tensor<<gm, bf16>, <<(8, -1)>, <(-1, 1)>>, <<(8, -1)>, <(-1, 1)>>, identity>
        %223 = cannir.extract_stride %222 : <<(8, -1)>, <(-1, 1)>> -> <(-1, 1)>
        %224 = cannir.extract_dim %223[0] : !cannir.stride<(-1, 1)>
        %225 = arith.muli %arg23, %224 : i64
        %226 = arith.muli %225, %c16_i64 : i64
        %227 = ascscalar.memref_address %alloc_12 : memref<1xbf16>
        %228 = arith.divui %226, %c8_i64 : i64
        %229 = arith.addi %227, %228 : i64
        %230 = ascscalar.bind_gm_ptr %229 : memref<1xbf16>
        %231 = ascscalar.bind_local_ptr %219 : memref<1xbf16, 1>
        %232 = arith.index_cast %9 : i64 to index
        %233 = cannir.extract_layout %arg17 : !cannir.tensor<<gm, bf16>, <<(8, -1)>, <(-1, 1)>>, <<(8, -1)>, <(-1, 1)>>, identity>
        %234 = cannir.extract_stride %233 : <<(8, -1)>, <(-1, 1)>> -> <(-1, 1)>
        %235 = cannir.extract_dim %234[0] : !cannir.stride<(-1, 1)>
        %236 = arith.index_cast %235 : i64 to index
        %237 = arith.muli %232, %c2 : index
        %238 = arith.muli %236, %c2 : index
        %239 = arith.cmpi ugt, %9, %c0_i64 : i64
        scf.if %239 {
          %254 = arith.index_cast %237 : index to i64
          %255 = arith.index_cast %238 : index to i64
          ascvec.get_buf mte3, %221, 0 {cannir.buffer.generated_sync}
          %256 = arith.trunci %254 : i64 to i32
          ascvec.copy_ub2gm %230[%c0], %231[%c0], %c1_i32, %256, %255, %256 l2_cache_ctl %c0_i32 : memref<1xbf16>, memref<1xbf16, 1>
          ascvec.release_buf mte3, %221, 0 {cannir.buffer.generated_sync}
        }
        %240 = cannir.extract_physical_layout %arg12 : !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
        %241 = cannir.extract_stride %240 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
        %242 = cannir.extract_dim %241[0] : !cannir.stride<(-1, 1)>
        %243 = arith.muli %arg23, %242 : i64
        %244 = arith.muli %243, %c16_i64 : i64
        %245 = ascscalar.memref_address %alloc_7 : memref<1xbf16>
        %246 = arith.divui %244, %c8_i64 : i64
        %247 = arith.addi %245, %246 : i64
        %248 = ascscalar.bind_gm_ptr %247 : memref<1xbf16>
        %249 = cannir.extract_layout %arg12 : !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
        %250 = cannir.extract_stride %249 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
        %251 = cannir.extract_dim %250[0] : !cannir.stride<(-1, 1)>
        %252 = arith.index_cast %251 : i64 to index
        %253 = arith.muli %252, %c2 : index
        scf.if %239 {
          %254 = arith.index_cast %237 : index to i64
          %255 = arith.index_cast %253 : index to i64
          ascvec.get_buf mte3, %221, 0 {cannir.buffer.generated_sync}
          %256 = arith.trunci %254 : i64 to i32
          ascvec.copy_ub2gm %248[%c0], %231[%c0], %c1_i32, %256, %255, %256 l2_cache_ctl %c0_i32 : memref<1xbf16>, memref<1xbf16, 1>
          ascvec.release_buf mte3, %221, 0 {cannir.buffer.generated_sync}
        }
        scf.yield %62 : i64
      }
      scf.yield %54 : i64
    } else {
      scf.yield %c0_i64 : i64
    }
    %24 = arith.addi %5, %20 : i64
    %25 = arith.muli %1, %c2_i64 : i64
    %26 = scf.for %arg23 = %24 to %c8_i64 step %25 iter_args(%arg24 = %23) -> (i64)  : i64 {
      %34 = arith.cmpi eq, %arg24, %c0_i64 : i64
      %35 = ascscalar.memref_address %alloc_19 : memref<2048xbf16, 1>
      %36 = ascscalar.memref_address %alloc_20 : memref<2048xbf16, 1>
      %37 = arith.select %34, %35, %36 : i64
      %38 = ascscalar.bind_local_ptr %37 : memref<2048xbf16, 1>
      %39 = arith.select %34, %c3_i64, %c4_i64 : i64
      %40 = arith.addi %arg24, %c1_i64 overflow<nsw> : i64
      %41 = arith.remui %40, %c2_i64 : i64
      ascvec.get_buf v, %39, 0 {cannir.buffer.generated_sync}
      ascvec.vf(%38 : memref<2048xbf16, 1>) {
      ^bb0(%arg25: memref<2048xbf16, 1>):
        %c1920 = arith.constant 1920 : index
        %c1792 = arith.constant 1792 : index
        %c1664 = arith.constant 1664 : index
        %c1536 = arith.constant 1536 : index
        %c1408 = arith.constant 1408 : index
        %c1280 = arith.constant 1280 : index
        %c1152 = arith.constant 1152 : index
        %c1024 = arith.constant 1024 : index
        %c896 = arith.constant 896 : index
        %c768 = arith.constant 768 : index
        %c640 = arith.constant 640 : index
        %c512 = arith.constant 512 : index
        %c384 = arith.constant 384 : index
        %c256 = arith.constant 256 : index
        %c128 = arith.constant 128 : index
        %c0_47 = arith.constant 0 : index
        %cst_48 = arith.constant 0.000000e+00 : bf16
        %c128_i32_49 = arith.constant 128 : i32
        %c0_50 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        scf.for %arg26 = %c0_50 to %c1 step %c1 {
          %mask, %remaining_out = ascvec.update_mask %c128_i32_49 elem_bits 16 : !ascvec.mask
          %60 = ascvec.duplicate_scalar %cst_48, %mask : bf16 -> <bf16>
          ascvec.storealign %arg25[%c0_47], %60, %mask : memref<2048xbf16, 1>, index, <bf16>
          ascvec.storealign %arg25[%c128], %60, %mask : memref<2048xbf16, 1>, index, <bf16>
          ascvec.storealign %arg25[%c256], %60, %mask : memref<2048xbf16, 1>, index, <bf16>
          ascvec.storealign %arg25[%c384], %60, %mask : memref<2048xbf16, 1>, index, <bf16>
          ascvec.storealign %arg25[%c512], %60, %mask : memref<2048xbf16, 1>, index, <bf16>
          ascvec.storealign %arg25[%c640], %60, %mask : memref<2048xbf16, 1>, index, <bf16>
          ascvec.storealign %arg25[%c768], %60, %mask : memref<2048xbf16, 1>, index, <bf16>
          ascvec.storealign %arg25[%c896], %60, %mask : memref<2048xbf16, 1>, index, <bf16>
          ascvec.storealign %arg25[%c1024], %60, %mask : memref<2048xbf16, 1>, index, <bf16>
          ascvec.storealign %arg25[%c1152], %60, %mask : memref<2048xbf16, 1>, index, <bf16>
          ascvec.storealign %arg25[%c1280], %60, %mask : memref<2048xbf16, 1>, index, <bf16>
          ascvec.storealign %arg25[%c1408], %60, %mask : memref<2048xbf16, 1>, index, <bf16>
          ascvec.storealign %arg25[%c1536], %60, %mask : memref<2048xbf16, 1>, index, <bf16>
          ascvec.storealign %arg25[%c1664], %60, %mask : memref<2048xbf16, 1>, index, <bf16>
          ascvec.storealign %arg25[%c1792], %60, %mask : memref<2048xbf16, 1>, index, <bf16>
          ascvec.storealign %arg25[%c1920], %60, %mask : memref<2048xbf16, 1>, index, <bf16>
        }
        ascvec.mem_bar vst_vld
      } {cannir.vf_mode = "simd"}
      ascvec.release_buf v, %39, 0 {cannir.buffer.generated_sync}
      %42 = cannir.extract_physical_layout %arg17 : !cannir.tensor<<gm, bf16>, <<(8, -1)>, <(-1, 1)>>, <<(8, -1)>, <(-1, 1)>>, identity>
      %43 = cannir.extract_stride %42 : <<(8, -1)>, <(-1, 1)>> -> <(-1, 1)>
      %44 = cannir.extract_dim %43[0] : !cannir.stride<(-1, 1)>
      %45 = arith.muli %arg23, %44 : i64
      %46 = arith.muli %45, %c16_i64 : i64
      %47 = ascscalar.memref_address %alloc_12 : memref<1xbf16>
      %48 = arith.divui %46, %c8_i64 : i64
      %49 = arith.addi %47, %48 : i64
      %50 = ascscalar.bind_gm_ptr %49 : memref<1xbf16>
      %51 = ascscalar.bind_local_ptr %37 : memref<1xbf16, 1>
      %52 = arith.index_cast %9 : i64 to index
      %53 = cannir.extract_layout %arg17 : !cannir.tensor<<gm, bf16>, <<(8, -1)>, <(-1, 1)>>, <<(8, -1)>, <(-1, 1)>>, identity>
      %54 = cannir.extract_stride %53 : <<(8, -1)>, <(-1, 1)>> -> <(-1, 1)>
      %55 = cannir.extract_dim %54[0] : !cannir.stride<(-1, 1)>
      %56 = arith.index_cast %55 : i64 to index
      %57 = arith.muli %52, %c2 : index
      %58 = arith.muli %56, %c2 : index
      %59 = arith.cmpi ugt, %9, %c0_i64 : i64
      scf.if %59 {
        %60 = arith.index_cast %57 : index to i64
        %61 = arith.index_cast %58 : index to i64
        ascvec.get_buf mte3, %39, 0 {cannir.buffer.generated_sync}
        %62 = arith.trunci %60 : i64 to i32
        ascvec.copy_ub2gm %50[%c0], %51[%c0], %c1_i32, %62, %61, %62 l2_cache_ctl %c0_i32 : memref<1xbf16>, memref<1xbf16, 1>
        ascvec.release_buf mte3, %39, 0 {cannir.buffer.generated_sync}
      }
      scf.yield %41 : i64
    }
    ascvec.sync_all
    ascvec.sync_block_arrive mte3, %c13_i64
    ascvec.sync_block_arrive mte3, %c14_i64 {mode = 0 : i32}
    ascvec.sync_block_wait s, %c14_i64 {mode = 0 : i32}
    ascvec.sync_block_wait s, %c15_i64
    %27 = arith.cmpi eq, %2, %c0_i64 : i64
    scf.if %27 {
      %34 = arith.subi %12, %c512_i64 : i64
      %35 = ascscalar.bitcast_memref %alloc_26 byte_offset = %c0_i64 : memref<512xf32, 1> -> memref<?xf32, 1>
      ascvec.get_buf mte2, %c2_i64, 0 {cannir.buffer.generated_sync}
      "ascvec.copy_gm2ub_plan"(%35, %alloc_1, %c0_i32, %c0_i32, %c0_i64, %c0_i64, %c1_i64, %c2048_i64, %c0_i64, %c2048_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 2048 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 6912 : i64, ub_capacity = 2048 : i64}> : (memref<?xf32, 1>, memref<512xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
      ascvec.release_buf mte2, %c2_i64, 0 {cannir.buffer.generated_sync}
      %36 = scf.for %arg23 = %0 to %5 step %1 iter_args(%arg24 = %c0_i64) -> (i64)  : i64 {
        %37 = arith.cmpi eq, %arg24, %c0_i64 : i64
        %38 = ascscalar.memref_address %alloc_22 : memref<576xbf16, 1>
        %39 = ascscalar.memref_address %alloc_23 : memref<576xbf16, 1>
        %40 = arith.select %37, %38, %39 : i64
        %41 = arith.cmpi ne, %arg24, %c0_i64 : i64
        %42 = arith.extui %41 : i1 to i64
        %43 = arith.addi %arg24, %c1_i64 overflow<nsw> : i64
        %44 = arith.remui %43, %c2_i64 : i64
        %45 = arith.minsi %arg23, %c8_i64 : i64
        %46 = arith.subi %c8_i64, %45 : i64
        %47 = arith.minsi %46, %c1_i64 : i64
        %48 = arith.cmpi sle, %arg23, %c8_i64 : i64
        %49 = arith.select %48, %47, %c0_i64 : i64
        %50 = cannir.extract_physical_layout %arg15 : !cannir.tensor<<gm, bf16>, <<(8, -1)>, <(-1, 1)>>, <<(8, -1)>, <(-1, 1)>>, identity>
        %51 = cannir.extract_stride %50 : <<(8, -1)>, <(-1, 1)>> -> <(-1, 1)>
        %52 = cannir.extract_dim %51[0] : !cannir.stride<(-1, 1)>
        %53 = arith.muli %52, %c2_i64 : i64
        %54 = cannir.extract_layout %arg15 : !cannir.tensor<<gm, bf16>, <<(8, -1)>, <(-1, 1)>>, <<(8, -1)>, <(-1, 1)>>, identity>
        %55 = cannir.extract_shape %54 : <<(8, -1)>, <(-1, 1)>> -> <(8, -1)>
        %56 = cannir.extract_dim %55[1] : !cannir.shape<(8, -1)>
        %57 = arith.maxsi %12, %c1_i64 : i64
        %58 = arith.divsi %56, %57 : i64
        %59 = arith.minsi %58, %c0_i64 : i64
        %60 = arith.muli %59, %12 : i64
        %61 = arith.subi %56, %60 : i64
        %62 = arith.minsi %12, %61 : i64
        %63 = arith.cmpi sge, %58, %c0_i64 : i64
        %64 = arith.select %63, %62, %c0_i64 : i64
        %65 = arith.muli %arg23, %52 : i64
        %66 = arith.muli %65, %c2_i64 : i64
        %67 = arith.muli %64, %c2_i64 : i64
        %68 = arith.cmpi eq, %67, %c0_i64 : i64
        %69 = arith.select %68, %c0_i64, %49 : i64
        %70 = arith.select %68, %c2_i64, %67 : i64
        %71 = ascscalar.bind_local_ptr %40 : memref<1xbf16, 1>
        ascvec.get_buf mte2, %42, 0 {cannir.buffer.generated_sync}
        "ascvec.copy_gm2ub_plan"(%71, %alloc_10, %c0_i32, %c0_i32, %c0_i64, %66, %69, %70, %53, %c1152_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
        ascvec.release_buf mte2, %42, 0 {cannir.buffer.generated_sync}
        %72 = ascscalar.bind_local_ptr %40 : memref<576xbf16, 1>
        %73 = ascscalar.memref_address %alloc_24 : memref<576xf32, 1>
        %74 = ascscalar.memref_address %alloc_25 : memref<576xf32, 1>
        %75 = arith.select %37, %73, %74 : i64
        %76 = ascscalar.bind_local_ptr %75 : memref<576xf32, 1>
        %77 = arith.trunci %12 : i64 to i32
        ascvec.get_buf v, %42, 0 {cannir.buffer.generated_sync}
        ascvec.vf(%72, %76, %77 : memref<576xbf16, 1>, memref<576xf32, 1>, i32) {
        ^bb0(%arg25: memref<576xbf16, 1>, %arg26: memref<576xf32, 1>, %arg27: i32):
          %c64_i32_47 = arith.constant 64 : i32
          %c0_i32_48 = arith.constant 0 : i32
          scf.for %arg28 = %c0_i32_48 to %arg27 step %c64_i32_47  : i32 {
            %115 = arith.index_cast %arg28 : i32 to index
            %116 = ascvec.loadalign_unpack %arg25[%115] {mode = "b16_to_b32"} : memref<576xbf16, 1>, index -> <bf16>
            %117 = ascvec.create_mask : !ascvec.mask
            %118 = ascvec.cast %116, %117 {rounding = 6 : i32} : <bf16> -> <f32>
            ascvec.storealign %arg26[%115], %118, %117 : memref<576xf32, 1>, index, <f32>
          }
          ascvec.mem_bar vst_vld
        } {cannir.vf_mode = "simd"}
        ascvec.release_buf v, %42, 0 {cannir.buffer.generated_sync}
        ascvec.vf(%76, %arg18, %alloc_29 : memref<576xf32, 1>, f32, memref<64xf32, 1>) {
        ^bb0(%arg25: memref<576xf32, 1>, %arg26: f32, %arg27: memref<64xf32, 1>):
          %c448 = arith.constant 448 : index
          %c384 = arith.constant 384 : index
          %c320 = arith.constant 320 : index
          %c256 = arith.constant 256 : index
          %c192 = arith.constant 192 : index
          %c128 = arith.constant 128 : index
          %c64 = arith.constant 64 : index
          %cst_47 = arith.constant 0.001953125 : f32
          %c0_48 = arith.constant 0 : index
          %cst_49 = arith.constant 0.000000e+00 : f32
          %c1_i32_50 = arith.constant 1 : i32
          %c0_51 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          scf.for %arg28 = %c0_51 to %c1 step %c1 {
            %mask, %remaining_out = ascvec.update_mask %c1_i32_50 : !ascvec.mask
            %115 = ascvec.duplicate_scalar %cst_49, %mask : f32 -> <f32>
            %116 = ascvec.loadalign %arg25[%c0_48] : memref<576xf32, 1>, index -> <f32>
            %117 = ascvec.create_mask : !ascvec.mask
            %118 = ascvec.mul %116, %116, %117 : <f32>
            %119 = ascvec.reduce_sum %118, %117 : <f32>
            %120 = ascvec.add %115, %119, %mask : <f32>
            %121 = ascvec.loadalign %arg25[%c64] : memref<576xf32, 1>, index -> <f32>
            %122 = ascvec.mul %121, %121, %117 : <f32>
            %123 = ascvec.reduce_sum %122, %117 : <f32>
            %124 = ascvec.add %120, %123, %mask : <f32>
            %125 = ascvec.loadalign %arg25[%c128] : memref<576xf32, 1>, index -> <f32>
            %126 = ascvec.mul %125, %125, %117 : <f32>
            %127 = ascvec.reduce_sum %126, %117 : <f32>
            %128 = ascvec.add %124, %127, %mask : <f32>
            %129 = ascvec.loadalign %arg25[%c192] : memref<576xf32, 1>, index -> <f32>
            %130 = ascvec.mul %129, %129, %117 : <f32>
            %131 = ascvec.reduce_sum %130, %117 : <f32>
            %132 = ascvec.add %128, %131, %mask : <f32>
            %133 = ascvec.loadalign %arg25[%c256] : memref<576xf32, 1>, index -> <f32>
            %134 = ascvec.mul %133, %133, %117 : <f32>
            %135 = ascvec.reduce_sum %134, %117 : <f32>
            %136 = ascvec.add %132, %135, %mask : <f32>
            %137 = ascvec.loadalign %arg25[%c320] : memref<576xf32, 1>, index -> <f32>
            %138 = ascvec.mul %137, %137, %117 : <f32>
            %139 = ascvec.reduce_sum %138, %117 : <f32>
            %140 = ascvec.add %136, %139, %mask : <f32>
            %141 = ascvec.loadalign %arg25[%c384] : memref<576xf32, 1>, index -> <f32>
            %142 = ascvec.mul %141, %141, %117 : <f32>
            %143 = ascvec.reduce_sum %142, %117 : <f32>
            %144 = ascvec.add %140, %143, %mask : <f32>
            %145 = ascvec.loadalign %arg25[%c448] : memref<576xf32, 1>, index -> <f32>
            %146 = ascvec.mul %145, %145, %117 : <f32>
            %147 = ascvec.reduce_sum %146, %117 : <f32>
            %148 = ascvec.add %144, %147, %mask : <f32>
            %149 = ascvec.mul_scalar %148, %cst_47, %mask : <f32>, f32
            %150 = ascvec.add_scalar %149, %arg26, %mask : <f32>, f32
            %151 = ascvec.sqrt %150, %mask : <f32>
            ascvec.storealign_1st %arg27[%c0_48], %151 : memref<64xf32, 1>, index, <f32>
          }
          ascvec.mem_bar vst_vld
        } {cannir.vf_mode = "simd"}
        %78 = cannir.extract_layout %arg9 : !cannir.tensor<<gm, bf16>, <<(-1, 64)>, <(64, 1)>>, <<(-1, 64)>, <(64, 1)>>, identity>
        %79 = cannir.extract_shape %78 : <<(-1, 64)>, <(64, 1)>> -> <(-1, 64)>
        %80 = cannir.extract_dim %79[0] : !cannir.shape<(-1, 64)>
        %81 = arith.minsi %arg23, %80 : i64
        %82 = arith.subi %80, %81 : i64
        %83 = arith.minsi %82, %c1_i64 : i64
        %84 = arith.cmpi sle, %arg23, %80 : i64
        %85 = arith.select %84, %83, %c0_i64 : i64
        %86 = arith.muli %arg23, %c128_i64 : i64
        %87 = arith.muli %85, %c128_i64 : i64
        %88 = arith.cmpi eq, %87, %c0_i64 : i64
        %89 = arith.cmpi ne, %87, %c0_i64 : i64
        %90 = arith.extui %89 : i1 to i64
        %91 = arith.select %88, %c2_i64, %87 : i64
        %92 = ascscalar.bitcast_memref %alloc_30 byte_offset = %c0_i64 : memref<64xbf16, 1> -> memref<?xbf16, 1>
        ascvec.get_buf mte2, %c5_i64, 0 {cannir.buffer.generated_sync}
        "ascvec.copy_gm2ub_plan"(%92, %alloc_4, %c0_i32, %c0_i32, %c0_i64, %86, %90, %91, %c0_i64, %91, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = 15200 : i64, ub_capacity = 128 : i64}> : (memref<?xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
        ascvec.release_buf mte2, %c5_i64, 0 {cannir.buffer.generated_sync}
        ascvec.get_buf v, %c5_i64, 0 {cannir.buffer.generated_sync}
        ascvec.vf(%alloc_30, %alloc_32 : memref<64xbf16, 1>, memref<128xf32, 1>) {
        ^bb0(%arg25: memref<64xbf16, 1>, %arg26: memref<128xf32, 1>):
          %c0_47 = arith.constant 0 : index
          %c0_48 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          scf.for %arg27 = %c0_48 to %c1 step %c1 {
            %115 = ascvec.loadalign_unpack %arg25[%c0_47] {mode = "b16_to_b32"} : memref<64xbf16, 1>, index -> <bf16>
            %116 = ascvec.create_mask : !ascvec.mask
            %117 = ascvec.cast %115, %116 {rounding = 6 : i32} : <bf16> -> <f32>
            ascvec.storealign %arg26[%c0_47], %117, %116 : memref<128xf32, 1>, index, <f32>
          }
          ascvec.mem_bar vst_vld
        } {cannir.vf_mode = "simd"}
        ascvec.release_buf v, %c5_i64, 0 {cannir.buffer.generated_sync}
        %93 = cannir.extract_layout %arg10 : !cannir.tensor<<gm, bf16>, <<(-1, 64)>, <(64, 1)>>, <<(-1, 64)>, <(64, 1)>>, identity>
        %94 = cannir.extract_shape %93 : <<(-1, 64)>, <(64, 1)>> -> <(-1, 64)>
        %95 = cannir.extract_dim %94[0] : !cannir.shape<(-1, 64)>
        %96 = arith.minsi %arg23, %95 : i64
        %97 = arith.subi %95, %96 : i64
        %98 = arith.minsi %97, %c1_i64 : i64
        %99 = arith.cmpi sle, %arg23, %95 : i64
        %100 = arith.select %99, %98, %c0_i64 : i64
        %101 = arith.muli %100, %c128_i64 : i64
        %102 = arith.cmpi eq, %101, %c0_i64 : i64
        %103 = arith.cmpi ne, %101, %c0_i64 : i64
        %104 = arith.extui %103 : i1 to i64
        %105 = arith.select %102, %c2_i64, %101 : i64
        %106 = ascscalar.bitcast_memref %alloc_31 byte_offset = %c0_i64 : memref<64xbf16, 1> -> memref<?xbf16, 1>
        ascvec.get_buf mte2, %c6_i64, 0 {cannir.buffer.generated_sync}
        "ascvec.copy_gm2ub_plan"(%106, %alloc_5, %c0_i32, %c0_i32, %c0_i64, %86, %104, %105, %c0_i64, %105, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = 15328 : i64, ub_capacity = 128 : i64}> : (memref<?xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
        ascvec.release_buf mte2, %c6_i64, 0 {cannir.buffer.generated_sync}
        ascvec.get_buf v, %c6_i64, 0 {cannir.buffer.generated_sync}
        ascvec.vf(%alloc_31, %alloc_33 : memref<64xbf16, 1>, memref<128xf32, 1>) {
        ^bb0(%arg25: memref<64xbf16, 1>, %arg26: memref<128xf32, 1>):
          %c0_47 = arith.constant 0 : index
          %c0_48 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          scf.for %arg27 = %c0_48 to %c1 step %c1 {
            %115 = ascvec.loadalign_unpack %arg25[%c0_47] {mode = "b16_to_b32"} : memref<64xbf16, 1>, index -> <bf16>
            %116 = ascvec.create_mask : !ascvec.mask
            %117 = ascvec.cast %115, %116 {rounding = 6 : i32} : <bf16> -> <f32>
            ascvec.storealign %arg26[%c0_47], %117, %116 : memref<128xf32, 1>, index, <f32>
          }
          ascvec.mem_bar vst_vld
        } {cannir.vf_mode = "simd"}
        ascvec.release_buf v, %c6_i64, 0 {cannir.buffer.generated_sync}
        ascvec.vf(%76, %alloc_34 : memref<576xf32, 1>, memref<128xf32, 1>) {
        ^bb0(%arg25: memref<576xf32, 1>, %arg26: memref<128xf32, 1>):
          %c0_47 = arith.constant 0 : index
          %c512 = arith.constant 512 : index
          %c0_48 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          scf.for %arg27 = %c0_48 to %c1 step %c1 {
            %115 = ascvec.loadalign %arg25[%c512] : memref<576xf32, 1>, index -> <f32>
            %116 = ascvec.create_mask : !ascvec.mask
            ascvec.storealign %arg26[%c0_47], %115, %116 : memref<128xf32, 1>, index, <f32>
          }
          ascvec.mem_bar vst_vld
        } {cannir.vf_mode = "simd"}
        %107 = arith.cmpi eq, %arg22, %c1_i64 : i64
        scf.if %107 {
          ascvec.vf(%alloc_33, %alloc_32, %alloc_34, %alloc_35 : memref<128xf32, 1>, memref<128xf32, 1>, memref<128xf32, 1>, memref<64xf32, 1>) {
          ^bb0(%arg25: memref<128xf32, 1>, %arg26: memref<128xf32, 1>, %arg27: memref<128xf32, 1>, %arg28: memref<64xf32, 1>):
            %c32 = arith.constant 32 : index
            %c0_47 = arith.constant 0 : index
            %c32_i32 = arith.constant 32 : i32
            %c0_48 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            scf.for %arg29 = %c0_48 to %c1 step %c1 {
              %mask, %remaining_out = ascvec.update_mask %c32_i32 : !ascvec.mask
              %115 = ascvec.loadalign %arg25[%c0_47] : memref<128xf32, 1>, index -> <f32>
              %116 = ascvec.loadalign %arg25[%c32] : memref<128xf32, 1>, index -> <f32>
              %117 = ascvec.loadalign %arg26[%c0_47] : memref<128xf32, 1>, index -> <f32>
              %118 = ascvec.loadalign %arg26[%c32] : memref<128xf32, 1>, index -> <f32>
              %result0, %result1 = ascvec.loadalign_deintlv %arg27[%c0_47] {dintlv_width = "b32"} : memref<128xf32, 1>, index -> <f32>, <f32>
              %119 = ascvec.mul %result0, %115, %mask : <f32>
              %120 = ascvec.mul %result1, %117, %mask : <f32>
              %121 = ascvec.sub %119, %120, %mask : <f32>
              %122 = ascvec.mul %result1, %116, %mask : <f32>
              %123 = ascvec.mul %result0, %118, %mask : <f32>
              %124 = ascvec.add %122, %123, %mask : <f32>
              ascvec.storealign %arg28[%c0_47], %121, %mask : memref<64xf32, 1>, index, <f32>
              ascvec.storealign %arg28[%c32], %124, %mask : memref<64xf32, 1>, index, <f32>
            }
            ascvec.mem_bar vst_vld
          } {cannir.vf_mode = "simd"}
        } else {
          ascvec.vf(%alloc_33, %alloc_32, %alloc_34, %alloc_35 : memref<128xf32, 1>, memref<128xf32, 1>, memref<128xf32, 1>, memref<64xf32, 1>) {
          ^bb0(%arg25: memref<128xf32, 1>, %arg26: memref<128xf32, 1>, %arg27: memref<128xf32, 1>, %arg28: memref<64xf32, 1>):
            %c32 = arith.constant 32 : index
            %c0_47 = arith.constant 0 : index
            %c32_i32 = arith.constant 32 : i32
            %c0_48 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            scf.for %arg29 = %c0_48 to %c1 step %c1 {
              %mask, %remaining_out = ascvec.update_mask %c32_i32 : !ascvec.mask
              %115 = ascvec.loadalign %arg25[%c0_47] : memref<128xf32, 1>, index -> <f32>
              %116 = ascvec.loadalign %arg25[%c32] : memref<128xf32, 1>, index -> <f32>
              %117 = ascvec.loadalign %arg26[%c0_47] : memref<128xf32, 1>, index -> <f32>
              %118 = ascvec.loadalign %arg26[%c32] : memref<128xf32, 1>, index -> <f32>
              %119 = ascvec.loadalign %arg27[%c0_47] : memref<128xf32, 1>, index -> <f32>
              %120 = ascvec.loadalign %arg27[%c32] : memref<128xf32, 1>, index -> <f32>
              %121 = ascvec.mul %119, %115, %mask : <f32>
              %122 = ascvec.mul %120, %117, %mask : <f32>
              %123 = ascvec.sub %121, %122, %mask : <f32>
              %124 = ascvec.mul %120, %116, %mask : <f32>
              %125 = ascvec.mul %119, %118, %mask : <f32>
              %126 = ascvec.add %124, %125, %mask : <f32>
              ascvec.storealign %arg28[%c0_47], %123, %mask : memref<64xf32, 1>, index, <f32>
              ascvec.storealign %arg28[%c32], %126, %mask : memref<64xf32, 1>, index, <f32>
            }
            ascvec.mem_bar vst_vld
          } {cannir.vf_mode = "simd"}
        }
        %108 = ascscalar.memref_address %alloc_27 : memref<576xbf16, 1>
        %109 = ascscalar.memref_address %alloc_28 : memref<576xbf16, 1>
        %110 = arith.select %37, %108, %109 : i64
        %111 = ascscalar.bind_local_ptr %110 : memref<576xbf16, 1>
        %112 = arith.select %37, %c3_i64, %c4_i64 : i64
        %113 = arith.trunci %34 : i64 to i32
        ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
        ascvec.get_buf v, %112, 0 {cannir.buffer.generated_sync}
        ascvec.vf(%113, %alloc_29, %76, %alloc_26, %arg20, %111, %alloc_35 : i32, memref<64xf32, 1>, memref<576xf32, 1>, memref<512xf32, 1>, f32, memref<576xbf16, 1>, memref<64xf32, 1>) {
        ^bb0(%arg25: i32, %arg26: memref<64xf32, 1>, %arg27: memref<576xf32, 1>, %arg28: memref<512xf32, 1>, %arg29: f32, %arg30: memref<576xbf16, 1>, %arg31: memref<64xf32, 1>):
          %c512 = arith.constant 512 : index
          %c448 = arith.constant 448 : index
          %c384 = arith.constant 384 : index
          %c320 = arith.constant 320 : index
          %c256 = arith.constant 256 : index
          %c192 = arith.constant 192 : index
          %c128 = arith.constant 128 : index
          %c64 = arith.constant 64 : index
          %c0_47 = arith.constant 0 : index
          %c0_48 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          scf.for %arg32 = %c0_48 to %c1 step %c1 {
            %mask, %remaining_out = ascvec.update_mask %arg25 : !ascvec.mask
            %115 = ascvec.loadalign_brc %arg26[%c0_47] : memref<64xf32, 1>, index -> <f32>
            %116 = ascvec.loadalign %arg27[%c0_47] : memref<576xf32, 1>, index -> <f32>
            %117 = ascvec.create_mask : !ascvec.mask
            %118 = ascvec.div %116, %115, %117 {precision_mode = true} : <f32>
            %119 = ascvec.loadalign %arg28[%c0_47] : memref<512xf32, 1>, index -> <f32>
            %120 = ascvec.mul %118, %119, %117 : <f32>
            %121 = ascvec.mul_scalar %120, %arg29, %117 : <f32>, f32
            %122 = ascvec.cast %121, %117 : <f32> -> <bf16>
            ascvec.storealign_pack %arg30[%c0_47], %122, %117 {mode = "b32_to_b16"} : memref<576xbf16, 1>, index, <bf16>
            %123 = ascvec.loadalign %arg27[%c64] : memref<576xf32, 1>, index -> <f32>
            %124 = ascvec.div %123, %115, %117 {precision_mode = true} : <f32>
            %125 = ascvec.loadalign %arg28[%c64] : memref<512xf32, 1>, index -> <f32>
            %126 = ascvec.mul %124, %125, %117 : <f32>
            %127 = ascvec.mul_scalar %126, %arg29, %117 : <f32>, f32
            %128 = ascvec.cast %127, %117 : <f32> -> <bf16>
            ascvec.storealign_pack %arg30[%c64], %128, %117 {mode = "b32_to_b16"} : memref<576xbf16, 1>, index, <bf16>
            %129 = ascvec.loadalign %arg27[%c128] : memref<576xf32, 1>, index -> <f32>
            %130 = ascvec.div %129, %115, %117 {precision_mode = true} : <f32>
            %131 = ascvec.loadalign %arg28[%c128] : memref<512xf32, 1>, index -> <f32>
            %132 = ascvec.mul %130, %131, %117 : <f32>
            %133 = ascvec.mul_scalar %132, %arg29, %117 : <f32>, f32
            %134 = ascvec.cast %133, %117 : <f32> -> <bf16>
            ascvec.storealign_pack %arg30[%c128], %134, %117 {mode = "b32_to_b16"} : memref<576xbf16, 1>, index, <bf16>
            %135 = ascvec.loadalign %arg27[%c192] : memref<576xf32, 1>, index -> <f32>
            %136 = ascvec.div %135, %115, %117 {precision_mode = true} : <f32>
            %137 = ascvec.loadalign %arg28[%c192] : memref<512xf32, 1>, index -> <f32>
            %138 = ascvec.mul %136, %137, %117 : <f32>
            %139 = ascvec.mul_scalar %138, %arg29, %117 : <f32>, f32
            %140 = ascvec.cast %139, %117 : <f32> -> <bf16>
            ascvec.storealign_pack %arg30[%c192], %140, %117 {mode = "b32_to_b16"} : memref<576xbf16, 1>, index, <bf16>
            %141 = ascvec.loadalign %arg27[%c256] : memref<576xf32, 1>, index -> <f32>
            %142 = ascvec.div %141, %115, %117 {precision_mode = true} : <f32>
            %143 = ascvec.loadalign %arg28[%c256] : memref<512xf32, 1>, index -> <f32>
            %144 = ascvec.mul %142, %143, %117 : <f32>
            %145 = ascvec.mul_scalar %144, %arg29, %117 : <f32>, f32
            %146 = ascvec.cast %145, %117 : <f32> -> <bf16>
            ascvec.storealign_pack %arg30[%c256], %146, %117 {mode = "b32_to_b16"} : memref<576xbf16, 1>, index, <bf16>
            %147 = ascvec.loadalign %arg27[%c320] : memref<576xf32, 1>, index -> <f32>
            %148 = ascvec.div %147, %115, %117 {precision_mode = true} : <f32>
            %149 = ascvec.loadalign %arg28[%c320] : memref<512xf32, 1>, index -> <f32>
            %150 = ascvec.mul %148, %149, %117 : <f32>
            %151 = ascvec.mul_scalar %150, %arg29, %117 : <f32>, f32
            %152 = ascvec.cast %151, %117 : <f32> -> <bf16>
            ascvec.storealign_pack %arg30[%c320], %152, %117 {mode = "b32_to_b16"} : memref<576xbf16, 1>, index, <bf16>
            %153 = ascvec.loadalign %arg27[%c384] : memref<576xf32, 1>, index -> <f32>
            %154 = ascvec.div %153, %115, %117 {precision_mode = true} : <f32>
            %155 = ascvec.loadalign %arg28[%c384] : memref<512xf32, 1>, index -> <f32>
            %156 = ascvec.mul %154, %155, %117 : <f32>
            %157 = ascvec.mul_scalar %156, %arg29, %117 : <f32>, f32
            %158 = ascvec.cast %157, %117 : <f32> -> <bf16>
            ascvec.storealign_pack %arg30[%c384], %158, %117 {mode = "b32_to_b16"} : memref<576xbf16, 1>, index, <bf16>
            %159 = ascvec.loadalign %arg27[%c448] : memref<576xf32, 1>, index -> <f32>
            %160 = ascvec.div %159, %115, %117 {precision_mode = true} : <f32>
            %161 = ascvec.loadalign %arg28[%c448] : memref<512xf32, 1>, index -> <f32>
            %162 = ascvec.mul %160, %161, %117 : <f32>
            %163 = ascvec.mul_scalar %162, %arg29, %117 : <f32>, f32
            %164 = ascvec.cast %163, %117 : <f32> -> <bf16>
            ascvec.storealign_pack %arg30[%c448], %164, %117 {mode = "b32_to_b16"} : memref<576xbf16, 1>, index, <bf16>
            %165 = ascvec.loadalign %arg31[%c0_47] : memref<64xf32, 1>, index -> <f32>
            %166 = ascvec.cast %165, %117 : <f32> -> <bf16>
            ascvec.storealign_pack %arg30[%c512], %166, %mask {mode = "b32_to_b16"} : memref<576xbf16, 1>, index, <bf16>
          }
          ascvec.mem_bar vst_vld
        } {cannir.vf_mode = "simd"}
        ascvec.release_buf v, %112, 0 {cannir.buffer.generated_sync}
        ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
        %114 = arith.cmpi eq, %arg21, %c0_i64 : i64
        scf.if %114 {
          %115 = arith.index_cast %arg23 : i64 to index
          %116 = memref.load %alloc_3[%115] : memref<1xi64>
          %117 = arith.cmpi sge, %116, %c0_i64 : i64
          scf.if %117 {
            %118 = cannir.extract_layout %arg7 : !cannir.tensor<<gm, bf16>, <<(-1, -1, 1, -1)>, <(-1, -1, -1, 1)>>, <<(-1, -1, 1, -1)>, <(-1, -1, -1, 1)>>, identity>
            %119 = cannir.extract_shape %118 : <<(-1, -1, 1, -1)>, <(-1, -1, -1, 1)>> -> <(-1, -1, 1, -1)>
            %120 = cannir.extract_dim %119[3] : !cannir.shape<(-1, -1, 1, -1)>
            %121 = arith.muli %116, %120 : i64
            %122 = arith.muli %121, %c16_i64 : i64
            %123 = ascscalar.memref_address %alloc_2 : memref<1xbf16>
            %124 = arith.divui %122, %c8_i64 : i64
            %125 = arith.addi %123, %124 : i64
            %126 = ascscalar.bind_gm_ptr %125 : memref<1xbf16>
            %127 = ascscalar.bind_local_ptr %110 : memref<1xbf16, 1>
            %128 = arith.index_cast %120 : i64 to index
            %129 = arith.muli %128, %c2 : index
            %130 = arith.cmpi ugt, %120, %c0_i64 : i64
            scf.if %130 {
              %131 = arith.index_cast %129 : index to i64
              ascvec.get_buf mte3, %112, 0 {cannir.buffer.generated_sync}
              %132 = arith.trunci %131 : i64 to i32
              ascvec.copy_ub2gm %126[%c0], %127[%c0], %c1_i32, %132, %131, %132 l2_cache_ctl %c0_i32 : memref<1xbf16>, memref<1xbf16, 1>
              ascvec.release_buf mte3, %112, 0 {cannir.buffer.generated_sync}
            }
          }
        } else {
          %115 = arith.index_cast %arg23 : i64 to index
          %116 = memref.load %alloc_3[%115] : memref<1xi64>
          %117 = arith.cmpi sge, %116, %c0_i64 : i64
          scf.if %117 {
            %118 = cannir.extract_layout %arg7 : !cannir.tensor<<gm, bf16>, <<(-1, -1, 1, -1)>, <(-1, -1, -1, 1)>>, <<(-1, -1, 1, -1)>, <(-1, -1, -1, 1)>>, identity>
            %119 = cannir.extract_shape %118 : <<(-1, -1, 1, -1)>, <(-1, -1, -1, 1)>> -> <(-1, -1, 1, -1)>
            %120 = cannir.extract_dim %119[3] : !cannir.shape<(-1, -1, 1, -1)>
            %121 = arith.remsi %120, %c16_i64 : i64
            %122 = arith.cmpi ne, %121, %c0_i64 : i64
            %123 = arith.cmpi slt, %121, %c0_i64 : i64
            %124 = arith.andi %122, %123 : i1
            %125 = arith.divsi %120, %c16_i64 : i64
            %126 = arith.extui %124 : i1 to i64
            %127 = arith.subi %125, %126 : i64
            %128 = cannir.extract_dim %119[1] : !cannir.shape<(-1, -1, 1, -1)>
            %129 = arith.remsi %116, %128 : i64
            %130 = arith.cmpi ne, %129, %c0_i64 : i64
            %131 = arith.cmpi slt, %129, %c0_i64 : i64
            %132 = arith.cmpi slt, %128, %c0_i64 : i64
            %133 = arith.xori %131, %132 : i1
            %134 = arith.andi %130, %133 : i1
            %135 = arith.divsi %116, %128 : i64
            %136 = arith.extui %134 : i1 to i64
            %137 = arith.subi %135, %136 : i64
            %138 = arith.select %134, %128, %c0_i64 : i64
            %139 = arith.addi %129, %138 : i64
            scf.for %arg25 = %c0_i64 to %127 step %c1_i64  : i64 {
              %140 = arith.muli %137, %127 : i64
              %141 = arith.addi %140, %arg25 : i64
              %142 = arith.muli %141, %128 : i64
              %143 = arith.addi %142, %139 : i64
              %144 = arith.muli %143, %c256_i64 : i64
              %145 = ascscalar.memref_address %alloc_2 : memref<1xbf16>
              %146 = arith.divui %144, %c8_i64 : i64
              %147 = arith.addi %145, %146 : i64
              %148 = ascscalar.bind_gm_ptr %147 : memref<1xbf16>
              %149 = arith.muli %arg25, %c256_i64 : i64
              %150 = arith.divui %149, %c8_i64 : i64
              %151 = arith.addi %110, %150 : i64
              %152 = ascscalar.bind_local_ptr %151 : memref<1xbf16, 1>
              %153 = arith.minsi %arg25, %c36_i64 : i64
              %154 = arith.muli %153, %c16_i64 : i64
              %155 = arith.subi %c576_i64, %154 : i64
              %156 = arith.minsi %155, %c16_i64 : i64
              %157 = arith.cmpi sle, %arg25, %c36_i64 : i64
              %158 = arith.select %157, %156, %c0_i64 : i64
              %159 = arith.index_cast %158 : i64 to index
              %160 = arith.muli %159, %c2 : index
              %161 = arith.cmpi ugt, %158, %c0_i64 : i64
              scf.if %161 {
                %162 = arith.index_cast %160 : index to i64
                ascvec.get_buf mte3, %112, 0 {cannir.buffer.generated_sync}
                %163 = arith.trunci %162 : i64 to i32
                ascvec.copy_ub2gm %148[%c0], %152[%c0], %c1_i32, %163, %c32_i64, %c1152_i32 l2_cache_ctl %c0_i32 : memref<1xbf16>, memref<1xbf16, 1>
                ascvec.release_buf mte3, %112, 0 {cannir.buffer.generated_sync}
              }
            }
          }
        }
        scf.yield %44 : i64
      }
    }
    ascvec.sync_all
    ascvec.sync_block_arrive mte3, %c13_i64
    ascvec.sync_block_arrive mte3, %c14_i64 {mode = 0 : i32}
    ascvec.sync_block_wait s, %c14_i64 {mode = 0 : i32}
    ascvec.sync_block_wait s, %c15_i64
    %28 = cannir.extract_layout %arg11 : !cannir.tensor<<gm, bf16>, <<(8, -1, -1)>, <(-1, -1, 1)>>, <<(8, -1, -1)>, <(-1, -1, 1)>>, identity>
    %29 = cannir.extract_shape %28 : <<(8, -1, -1)>, <(-1, -1, 1)>> -> <(8, -1, -1)>
    %30 = cannir.extract_dim %29[1] : !cannir.shape<(8, -1, -1)>
    %31 = cannir.extract_dim %29[2] : !cannir.shape<(8, -1, -1)>
    %32 = arith.muli %30, %31 : i64
    %33 = arith.cmpi ne, %12, %c512_i64 : i64
    scf.if %33 {
      %34 = arith.muli %5, %15 : i64
      %35 = arith.cmpi slt, %20, %34 : i64
      scf.if %35 {
        %36 = scf.for %arg23 = %20 to %34 step %25 iter_args(%arg24 = %c0_i64) -> (i64)  : i64 {
          %37 = arith.remsi %arg23, %15 : i64
          %38 = arith.cmpi ne, %37, %c0_i64 : i64
          %39 = arith.cmpi slt, %37, %c0_i64 : i64
          %40 = arith.cmpi slt, %15, %c0_i64 : i64
          %41 = arith.xori %39, %40 : i1
          %42 = arith.andi %38, %41 : i1
          %43 = arith.divsi %arg23, %15 : i64
          %44 = arith.extui %42 : i1 to i64
          %45 = arith.subi %43, %44 : i64
          %46 = arith.select %42, %15, %c0_i64 : i64
          %47 = arith.addi %37, %46 : i64
          %48 = cannir.extract_layout %arg9 : !cannir.tensor<<gm, bf16>, <<(-1, 64)>, <(64, 1)>>, <<(-1, 64)>, <(64, 1)>>, identity>
          %49 = cannir.extract_shape %48 : <<(-1, 64)>, <(64, 1)>> -> <(-1, 64)>
          %50 = cannir.extract_dim %49[0] : !cannir.shape<(-1, 64)>
          %51 = arith.minsi %45, %50 : i64
          %52 = arith.subi %50, %51 : i64
          %53 = arith.minsi %52, %c1_i64 : i64
          %54 = arith.cmpi sle, %45, %50 : i64
          %55 = arith.select %54, %53, %c0_i64 : i64
          %56 = arith.muli %45, %c128_i64 : i64
          %57 = arith.muli %55, %c128_i64 : i64
          %58 = arith.cmpi eq, %57, %c0_i64 : i64
          %59 = arith.cmpi ne, %57, %c0_i64 : i64
          %60 = arith.extui %59 : i1 to i64
          %61 = arith.select %58, %c2_i64, %57 : i64
          %62 = ascscalar.bitcast_memref %alloc_41 byte_offset = %c0_i64 : memref<64xbf16, 1> -> memref<?xbf16, 1>
          ascvec.get_buf mte2, %c4_i64, 0 {cannir.buffer.generated_sync}
          "ascvec.copy_gm2ub_plan"(%62, %alloc_4, %c0_i32, %c0_i32, %c0_i64, %56, %60, %61, %c0_i64, %61, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = 768 : i64, ub_capacity = 128 : i64}> : (memref<?xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
          ascvec.release_buf mte2, %c4_i64, 0 {cannir.buffer.generated_sync}
          ascvec.get_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
          ascvec.vf(%alloc_41, %alloc_43 : memref<64xbf16, 1>, memref<128xf32, 1>) {
          ^bb0(%arg25: memref<64xbf16, 1>, %arg26: memref<128xf32, 1>):
            %c0_47 = arith.constant 0 : index
            %c0_48 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            scf.for %arg27 = %c0_48 to %c1 step %c1 {
              %150 = ascvec.loadalign_unpack %arg25[%c0_47] {mode = "b16_to_b32"} : memref<64xbf16, 1>, index -> <bf16>
              %151 = ascvec.create_mask : !ascvec.mask
              %152 = ascvec.cast %150, %151 {rounding = 6 : i32} : <bf16> -> <f32>
              ascvec.storealign %arg26[%c0_47], %152, %151 : memref<128xf32, 1>, index, <f32>
            }
            ascvec.mem_bar vst_vld
          } {cannir.vf_mode = "simd"}
          ascvec.release_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
          %63 = cannir.extract_layout %arg10 : !cannir.tensor<<gm, bf16>, <<(-1, 64)>, <(64, 1)>>, <<(-1, 64)>, <(64, 1)>>, identity>
          %64 = cannir.extract_shape %63 : <<(-1, 64)>, <(64, 1)>> -> <(-1, 64)>
          %65 = cannir.extract_dim %64[0] : !cannir.shape<(-1, 64)>
          %66 = arith.minsi %45, %65 : i64
          %67 = arith.subi %65, %66 : i64
          %68 = arith.minsi %67, %c1_i64 : i64
          %69 = arith.cmpi sle, %45, %65 : i64
          %70 = arith.select %69, %68, %c0_i64 : i64
          %71 = arith.muli %70, %c128_i64 : i64
          %72 = arith.cmpi eq, %71, %c0_i64 : i64
          %73 = arith.cmpi ne, %71, %c0_i64 : i64
          %74 = arith.extui %73 : i1 to i64
          %75 = arith.select %72, %c2_i64, %71 : i64
          %76 = ascscalar.bitcast_memref %alloc_42 byte_offset = %c0_i64 : memref<64xbf16, 1> -> memref<?xbf16, 1>
          ascvec.get_buf mte2, %c5_i64, 0 {cannir.buffer.generated_sync}
          "ascvec.copy_gm2ub_plan"(%76, %alloc_5, %c0_i32, %c0_i32, %c0_i64, %56, %74, %75, %c0_i64, %75, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = 896 : i64, ub_capacity = 128 : i64}> : (memref<?xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
          ascvec.release_buf mte2, %c5_i64, 0 {cannir.buffer.generated_sync}
          ascvec.get_buf v, %c5_i64, 0 {cannir.buffer.generated_sync}
          ascvec.vf(%alloc_42, %alloc_44 : memref<64xbf16, 1>, memref<128xf32, 1>) {
          ^bb0(%arg25: memref<64xbf16, 1>, %arg26: memref<128xf32, 1>):
            %c0_47 = arith.constant 0 : index
            %c0_48 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            scf.for %arg27 = %c0_48 to %c1 step %c1 {
              %150 = ascvec.loadalign_unpack %arg25[%c0_47] {mode = "b16_to_b32"} : memref<64xbf16, 1>, index -> <bf16>
              %151 = ascvec.create_mask : !ascvec.mask
              %152 = ascvec.cast %150, %151 {rounding = 6 : i32} : <bf16> -> <f32>
              ascvec.storealign %arg26[%c0_47], %152, %151 : memref<128xf32, 1>, index, <f32>
            }
            ascvec.mem_bar vst_vld
          } {cannir.vf_mode = "simd"}
          ascvec.release_buf v, %c5_i64, 0 {cannir.buffer.generated_sync}
          %77 = arith.muli %47, %18 : i64
          %78 = arith.addi %77, %16 : i64
          %79 = arith.muli %47, %12 : i64
          %80 = arith.addi %79, %c512_i64 : i64
          %81 = arith.cmpi eq, %arg24, %c0_i64 : i64
          %82 = ascscalar.memref_address %alloc_36 : memref<64xbf16, 1>
          %83 = ascscalar.memref_address %alloc_37 : memref<64xbf16, 1>
          %84 = arith.select %81, %82, %83 : i64
          %85 = arith.cmpi ne, %arg24, %c0_i64 : i64
          %86 = arith.extui %85 : i1 to i64
          %87 = arith.addi %arg24, %c1_i64 overflow<nsw> : i64
          %88 = arith.remui %87, %c2_i64 : i64
          %89 = arith.remsi %78, %c64_i64 : i64
          %90 = arith.cmpi ne, %89, %c0_i64 : i64
          %91 = arith.cmpi slt, %89, %c0_i64 : i64
          %92 = arith.andi %90, %91 : i1
          %93 = arith.divsi %78, %c64_i64 : i64
          %94 = arith.extui %92 : i1 to i64
          %95 = arith.subi %93, %94 : i64
          %96 = arith.minsi %45, %c8_i64 : i64
          %97 = arith.subi %c8_i64, %96 : i64
          %98 = arith.minsi %97, %c1_i64 : i64
          %99 = arith.cmpi sle, %45, %c8_i64 : i64
          %100 = arith.select %99, %98, %c0_i64 : i64
          %101 = cannir.extract_physical_layout %arg16 : !cannir.tensor<<gm, bf16>, <<(8, -1)>, <(-1, 1)>>, <<(8, -1)>, <(-1, 1)>>, identity>
          %102 = cannir.extract_stride %101 : <<(8, -1)>, <(-1, 1)>> -> <(-1, 1)>
          %103 = cannir.extract_dim %102[0] : !cannir.stride<(-1, 1)>
          %104 = arith.muli %103, %c2_i64 : i64
          %105 = cannir.extract_layout %arg16 : !cannir.tensor<<gm, bf16>, <<(8, -1)>, <(-1, 1)>>, <<(8, -1)>, <(-1, 1)>>, identity>
          %106 = cannir.extract_shape %105 : <<(8, -1)>, <(-1, 1)>> -> <(8, -1)>
          %107 = cannir.extract_dim %106[1] : !cannir.shape<(8, -1)>
          %108 = arith.divsi %107, %c64_i64 : i64
          %109 = arith.minsi %95, %108 : i64
          %110 = arith.muli %109, %c64_i64 : i64
          %111 = arith.subi %107, %110 : i64
          %112 = arith.minsi %111, %c64_i64 : i64
          %113 = arith.cmpi sle, %95, %108 : i64
          %114 = arith.select %113, %112, %c0_i64 : i64
          %115 = arith.muli %45, %103 : i64
          %116 = arith.muli %115, %c2_i64 : i64
          %117 = arith.muli %95, %c128_i64 : i64
          %118 = arith.addi %116, %117 : i64
          %119 = arith.muli %114, %c2_i64 : i64
          %120 = arith.cmpi eq, %119, %c0_i64 : i64
          %121 = arith.select %120, %c0_i64, %100 : i64
          %122 = arith.select %120, %c2_i64, %119 : i64
          %123 = ascscalar.bind_local_ptr %84 : memref<1xbf16, 1>
          ascvec.get_buf mte2, %86, 0 {cannir.buffer.generated_sync}
          "ascvec.copy_gm2ub_plan"(%123, %alloc_11, %c0_i32, %c0_i32, %c0_i64, %118, %121, %122, %104, %c128_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
          ascvec.release_buf mte2, %86, 0 {cannir.buffer.generated_sync}
          %124 = ascscalar.bind_local_ptr %84 : memref<64xbf16, 1>
          ascvec.get_buf v, %86, 0 {cannir.buffer.generated_sync}
          ascvec.vf(%124, %alloc_38 : memref<64xbf16, 1>, memref<64xf32, 1>) {
          ^bb0(%arg25: memref<64xbf16, 1>, %arg26: memref<64xf32, 1>):
            %c0_47 = arith.constant 0 : index
            %c0_48 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            scf.for %arg27 = %c0_48 to %c1 step %c1 {
              %150 = ascvec.loadalign_unpack %arg25[%c0_47] {mode = "b16_to_b32"} : memref<64xbf16, 1>, index -> <bf16>
              %151 = ascvec.create_mask : !ascvec.mask
              %152 = ascvec.cast %150, %151 {rounding = 6 : i32} : <bf16> -> <f32>
              ascvec.storealign %arg26[%c0_47], %152, %151 : memref<64xf32, 1>, index, <f32>
            }
            ascvec.mem_bar vst_vld
          } {cannir.vf_mode = "simd"}
          ascvec.release_buf v, %86, 0 {cannir.buffer.generated_sync}
          ascvec.vf(%alloc_38, %alloc_45 : memref<64xf32, 1>, memref<128xf32, 1>) {
          ^bb0(%arg25: memref<64xf32, 1>, %arg26: memref<128xf32, 1>):
            %c0_47 = arith.constant 0 : index
            %c0_48 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            scf.for %arg27 = %c0_48 to %c1 step %c1 {
              %150 = ascvec.loadalign %arg25[%c0_47] : memref<64xf32, 1>, index -> <f32>
              %151 = ascvec.create_mask : !ascvec.mask
              ascvec.storealign %arg26[%c0_47], %150, %151 : memref<128xf32, 1>, index, <f32>
            }
            ascvec.mem_bar vst_vld
          } {cannir.vf_mode = "simd"}
          %125 = arith.cmpi eq, %arg22, %c1_i64 : i64
          scf.if %125 {
            ascvec.vf(%alloc_44, %alloc_43, %alloc_45, %alloc_46 : memref<128xf32, 1>, memref<128xf32, 1>, memref<128xf32, 1>, memref<64xf32, 1>) {
            ^bb0(%arg25: memref<128xf32, 1>, %arg26: memref<128xf32, 1>, %arg27: memref<128xf32, 1>, %arg28: memref<64xf32, 1>):
              %c32 = arith.constant 32 : index
              %c0_47 = arith.constant 0 : index
              %c32_i32 = arith.constant 32 : i32
              %c0_48 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              scf.for %arg29 = %c0_48 to %c1 step %c1 {
                %mask, %remaining_out = ascvec.update_mask %c32_i32 : !ascvec.mask
                %150 = ascvec.loadalign %arg25[%c0_47] : memref<128xf32, 1>, index -> <f32>
                %151 = ascvec.loadalign %arg25[%c32] : memref<128xf32, 1>, index -> <f32>
                %152 = ascvec.loadalign %arg26[%c0_47] : memref<128xf32, 1>, index -> <f32>
                %153 = ascvec.loadalign %arg26[%c32] : memref<128xf32, 1>, index -> <f32>
                %result0, %result1 = ascvec.loadalign_deintlv %arg27[%c0_47] {dintlv_width = "b32"} : memref<128xf32, 1>, index -> <f32>, <f32>
                %154 = ascvec.mul %result0, %150, %mask : <f32>
                %155 = ascvec.mul %result1, %152, %mask : <f32>
                %156 = ascvec.sub %154, %155, %mask : <f32>
                %157 = ascvec.mul %result1, %151, %mask : <f32>
                %158 = ascvec.mul %result0, %153, %mask : <f32>
                %159 = ascvec.add %157, %158, %mask : <f32>
                ascvec.storealign %arg28[%c0_47], %156, %mask : memref<64xf32, 1>, index, <f32>
                ascvec.storealign %arg28[%c32], %159, %mask : memref<64xf32, 1>, index, <f32>
              }
              ascvec.mem_bar vst_vld
            } {cannir.vf_mode = "simd"}
          } else {
            ascvec.vf(%alloc_44, %alloc_43, %alloc_45, %alloc_46 : memref<128xf32, 1>, memref<128xf32, 1>, memref<128xf32, 1>, memref<64xf32, 1>) {
            ^bb0(%arg25: memref<128xf32, 1>, %arg26: memref<128xf32, 1>, %arg27: memref<128xf32, 1>, %arg28: memref<64xf32, 1>):
              %c32 = arith.constant 32 : index
              %c0_47 = arith.constant 0 : index
              %c32_i32 = arith.constant 32 : i32
              %c0_48 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              scf.for %arg29 = %c0_48 to %c1 step %c1 {
                %mask, %remaining_out = ascvec.update_mask %c32_i32 : !ascvec.mask
                %150 = ascvec.loadalign %arg25[%c0_47] : memref<128xf32, 1>, index -> <f32>
                %151 = ascvec.loadalign %arg25[%c32] : memref<128xf32, 1>, index -> <f32>
                %152 = ascvec.loadalign %arg26[%c0_47] : memref<128xf32, 1>, index -> <f32>
                %153 = ascvec.loadalign %arg26[%c32] : memref<128xf32, 1>, index -> <f32>
                %154 = ascvec.loadalign %arg27[%c0_47] : memref<128xf32, 1>, index -> <f32>
                %155 = ascvec.loadalign %arg27[%c32] : memref<128xf32, 1>, index -> <f32>
                %156 = ascvec.mul %154, %150, %mask : <f32>
                %157 = ascvec.mul %155, %152, %mask : <f32>
                %158 = ascvec.sub %156, %157, %mask : <f32>
                %159 = ascvec.mul %155, %151, %mask : <f32>
                %160 = ascvec.mul %154, %153, %mask : <f32>
                %161 = ascvec.add %159, %160, %mask : <f32>
                ascvec.storealign %arg28[%c0_47], %158, %mask : memref<64xf32, 1>, index, <f32>
                ascvec.storealign %arg28[%c32], %161, %mask : memref<64xf32, 1>, index, <f32>
              }
              ascvec.mem_bar vst_vld
            } {cannir.vf_mode = "simd"}
          }
          %126 = ascscalar.memref_address %alloc_39 : memref<64xbf16, 1>
          %127 = ascscalar.memref_address %alloc_40 : memref<64xbf16, 1>
          %128 = arith.select %81, %126, %127 : i64
          %129 = ascscalar.bind_local_ptr %128 : memref<64xbf16, 1>
          %130 = arith.select %81, %c2_i64, %c3_i64 : i64
          ascvec.get_buf v, %130, 0 {cannir.buffer.generated_sync}
          ascvec.vf(%alloc_46, %129 : memref<64xf32, 1>, memref<64xbf16, 1>) {
          ^bb0(%arg25: memref<64xf32, 1>, %arg26: memref<64xbf16, 1>):
            %c0_47 = arith.constant 0 : index
            %c0_48 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            scf.for %arg27 = %c0_48 to %c1 step %c1 {
              %150 = ascvec.loadalign %arg25[%c0_47] : memref<64xf32, 1>, index -> <f32>
              %151 = ascvec.create_mask : !ascvec.mask
              %152 = ascvec.cast %150, %151 : <f32> -> <bf16>
              ascvec.storealign_pack %arg26[%c0_47], %152, %151 {mode = "b32_to_b16"} : memref<64xbf16, 1>, index, <bf16>
            }
            ascvec.mem_bar vst_vld
          } {cannir.vf_mode = "simd"}
          ascvec.release_buf v, %130, 0 {cannir.buffer.generated_sync}
          %131 = arith.remsi %80, %c64_i64 : i64
          %132 = arith.cmpi ne, %131, %c0_i64 : i64
          %133 = arith.cmpi slt, %131, %c0_i64 : i64
          %134 = arith.andi %132, %133 : i1
          %135 = arith.divsi %80, %c64_i64 : i64
          %136 = arith.extui %134 : i1 to i64
          %137 = arith.subi %135, %136 : i64
          %138 = arith.muli %45, %32 : i64
          %139 = arith.muli %137, %c64_i64 : i64
          %140 = arith.addi %138, %139 : i64
          %141 = arith.muli %140, %c16_i64 : i64
          %142 = ascscalar.memref_address %alloc_6 : memref<1xbf16>
          %143 = arith.divui %141, %c8_i64 : i64
          %144 = arith.addi %142, %143 : i64
          %145 = ascscalar.bind_gm_ptr %144 : memref<1xbf16>
          %146 = ascscalar.bind_local_ptr %128 : memref<1xbf16, 1>
          %147 = arith.index_cast %32 : i64 to index
          %148 = arith.muli %147, %c2 : index
          %149 = arith.index_cast %148 : index to i64
          ascvec.get_buf mte3, %130, 0 {cannir.buffer.generated_sync}
          ascvec.copy_ub2gm %145[%c0], %146[%c0], %c1_i32, %c128_i32, %149, %c128_i32 l2_cache_ctl %c0_i32 : memref<1xbf16>, memref<1xbf16, 1>
          ascvec.release_buf mte3, %130, 0 {cannir.buffer.generated_sync}
          scf.yield %88 : i64
        }
      }
    }
    return
  }
  func.func @kernel_MlaPrologKernel_run_qr_rope_0_aic(%arg0: !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg1: !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg2: !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg3: !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg4: !cannir.tensor<<gm, bf16>, <<(-1, -1, 512)>, <(-1, 512, 1)>>, <<(-1, -1, 512)>, <(-1, 512, 1)>>, identity>, %arg5: !cannir.tensor<<gm, f32>, <<(1, -1)>, <(-1, 1)>>, <<(1, -1)>, <(-1, 1)>>, identity>, %arg6: !cannir.tensor<<gm, f32>, <<(1, 512)>, <(512, 1)>>, <<(1, 512)>, <(512, 1)>>, identity>, %arg7: !cannir.tensor<<gm, bf16>, <<(-1, -1, 1, -1)>, <(-1, -1, -1, 1)>>, <<(-1, -1, 1, -1)>, <(-1, -1, -1, 1)>>, identity>, %arg8: !cannir.tensor<<gm, i64>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg9: !cannir.tensor<<gm, bf16>, <<(-1, 64)>, <(64, 1)>>, <<(-1, 64)>, <(64, 1)>>, identity>, %arg10: !cannir.tensor<<gm, bf16>, <<(-1, 64)>, <(64, 1)>>, <<(-1, 64)>, <(64, 1)>>, identity>, %arg11: !cannir.tensor<<gm, bf16>, <<(8, -1, -1)>, <(-1, -1, 1)>>, <<(8, -1, -1)>, <(-1, -1, 1)>>, identity>, %arg12: !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg13: !cannir.tensor<<gm, bf16>, <<(8, -1)>, <(-1, 1)>>, <<(8, -1)>, <(-1, 1)>>, identity>, %arg14: !cannir.tensor<<gm, f32>, <<(32, -1)>, <(-1, 1)>>, <<(32, -1)>, <(-1, 1)>>, identity>, %arg15: !cannir.tensor<<gm, bf16>, <<(8, -1)>, <(-1, 1)>>, <<(8, -1)>, <(-1, 1)>>, identity>, %arg16: !cannir.tensor<<gm, bf16>, <<(8, -1)>, <(-1, 1)>>, <<(8, -1)>, <(-1, 1)>>, identity>, %arg17: !cannir.tensor<<gm, bf16>, <<(8, -1)>, <(-1, 1)>>, <<(8, -1)>, <(-1, 1)>>, identity>, %arg18: f32, %arg19: f32, %arg20: f32, %arg21: i64, %arg22: i64) attributes {cannir.buffer.auto_sync, cannir.buffer.sync_analyzed, cannir.buffer.sync_verified, cannir.buffer.tensor_state, cannir.core = "aic", cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.device, cannir.explicit_ring_buffer_cursors, cannir.kernel_group = "kernel_MlaPrologKernel_run_qr_rope_0", cannir.resource.ids_prepared} {
    %c6_i32 = arith.constant 6 : i32
    %c96_i32 = arith.constant 96 : i32
    %c4_i32 = arith.constant 4 : i32
    %c8_i32 = arith.constant 8 : i32
    %c16_i32 = arith.constant 16 : i32
    %c512_i64 = arith.constant 512 : i64
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c0 = arith.constant 0 : index
    %c256_i64 = arith.constant 256 : i64
    %c1_i32 = arith.constant 1 : i32
    %c64_i32 = arith.constant 64 : i32
    %c2048_i64 = arith.constant 2048 : i64
    %c128_i64 = arith.constant 128 : i64
    %c96_i64 = arith.constant 96 : i64
    %c2 = arith.constant 2 : index
    %c1024_i64 = arith.constant 1024 : i64
    %c64_i64 = arith.constant 64 : i64
    %c32_i64 = arith.constant 32 : i64
    %c16_i64 = arith.constant 16 : i64
    %c15_i64 = arith.constant 15 : i64
    %c14_i64 = arith.constant 14 : i64
    %c13_i64 = arith.constant 13 : i64
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
    %alloc = memref.alloc() {cannir.arg_index = 1 : i64} : memref<1xbf16>
    %alloc_0 = memref.alloc() {cannir.arg_index = 2 : i64} : memref<1xbf16>
    %alloc_1 = memref.alloc() {cannir.arg_index = 3 : i64} : memref<1xbf16>
    %alloc_2 = memref.alloc() {cannir.arg_index = 4 : i64} : memref<1xbf16>
    %alloc_3 = memref.alloc() {cannir.arg_index = 11 : i64} : memref<1xbf16>
    %alloc_4 = memref.alloc() {cannir.arg_index = 13 : i64} : memref<1xbf16>
    %alloc_5 = memref.alloc() {cannir.arg_index = 14 : i64} : memref<1xf32>
    %alloc_6 = memref.alloc() {cannir.arg_index = 15 : i64} : memref<1xbf16>
    %alloc_7 = memref.alloc() {cannir.arg_index = 16 : i64} : memref<1xbf16>
    %alloc_8 = memref.alloc() {cannir.arg_index = 17 : i64} : memref<1xbf16>
    %alloc_9 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<1024xf32, 5>
    %alloc_10 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<2048xbf16, 3>
    %alloc_11 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<8192xbf16, 4>
    %alloc_12 = memref.alloc() {cannir.byte_offset = 4096 : i64} : memref<2048xbf16, 3>
    %alloc_13 = memref.alloc() {cannir.byte_offset = 16384 : i64} : memref<8192xbf16, 4>
    %alloc_14 = memref.alloc() {cannir.byte_offset = 4096 : i64} : memref<1024xf32, 5>
    %alloc_15 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<1024xbf16, 3>
    %alloc_16 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<8192xbf16, 4>
    %alloc_17 = memref.alloc() {cannir.byte_offset = 20480 : i64} : memref<8192xbf16, 2>
    %alloc_18 = memref.alloc() {cannir.byte_offset = 4096 : i64} : memref<8192xbf16, 2>
    %alloc_19 = memref.alloc() {cannir.byte_offset = 2048 : i64} : memref<1024xbf16, 2>
    %alloc_20 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<1024xbf16, 2>
    %alloc_21 = memref.alloc() {cannir.byte_offset = 8192 : i64} : memref<2048xf32, 5>
    %alloc_22 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<2048xf32, 5>
    %alloc_23 = memref.alloc() {cannir.byte_offset = 24576 : i64} : memref<12288xbf16, 4>
    %alloc_24 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<12288xbf16, 4>
    %alloc_25 = memref.alloc() {cannir.byte_offset = 32768 : i64} : memref<12288xbf16, 2>
    %alloc_26 = memref.alloc() {cannir.byte_offset = 8192 : i64} : memref<12288xbf16, 2>
    %alloc_27 = memref.alloc() {cannir.byte_offset = 4096 : i64} : memref<2048xbf16, 3>
    %alloc_28 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<2048xbf16, 3>
    %alloc_29 = memref.alloc() {cannir.byte_offset = 4096 : i64} : memref<2048xbf16, 2>
    %alloc_30 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<2048xbf16, 2>
    %alloc_31 = memref.alloc() {cannir.byte_offset = 6144 : i64} : memref<1536xf32, 5>
    %alloc_32 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<1536xf32, 5>
    %alloc_33 = memref.alloc() {cannir.byte_offset = 16384 : i64} : memref<8192xbf16, 4>
    %alloc_34 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<8192xbf16, 4>
    %alloc_35 = memref.alloc() {cannir.byte_offset = 4096 : i64} : memref<2048xbf16, 3>
    %alloc_36 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<2048xbf16, 3>
    %alloc_37 = memref.alloc() {cannir.byte_offset = 49152 : i64} : memref<16384xbf16, 2>
    %alloc_38 = memref.alloc() {cannir.byte_offset = 16384 : i64} : memref<16384xbf16, 2>
    %alloc_39 = memref.alloc() {cannir.byte_offset = 8192 : i64} : memref<4096xbf16, 2>
    %alloc_40 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<4096xbf16, 2>
    %alloc_41 = memref.alloc() {cannir.byte_offset = 4096 : i64} : memref<1024xf32, 5>
    %alloc_42 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<1024xf32, 5>
    %alloc_43 = memref.alloc() {cannir.byte_offset = 49152 : i64} : memref<16384xbf16, 2>
    %alloc_44 = memref.alloc() {cannir.byte_offset = 16384 : i64} : memref<16384xbf16, 2>
    %alloc_45 = memref.alloc() {cannir.byte_offset = 8192 : i64} : memref<4096xbf16, 2>
    %alloc_46 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<4096xbf16, 2>
    %0 = ascscalar.get_block_idx
    %1 = ascscalar.get_block_num
    %2 = cannir.extract_layout %arg0 : !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
    %3 = cannir.extract_shape %2 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, -1)>
    %4 = cannir.extract_dim %3[0] : !cannir.shape<(-1, -1)>
    %5 = cannir.extract_dim %3[1] : !cannir.shape<(-1, -1)>
    %6 = cannir.extract_layout %arg1 : !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
    %7 = cannir.extract_shape %6 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, -1)>
    %8 = cannir.extract_dim %7[0] : !cannir.shape<(-1, -1)>
    %9 = cannir.extract_layout %arg3 : !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
    %10 = cannir.extract_shape %9 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, -1)>
    %11 = cannir.extract_dim %10[0] : !cannir.shape<(-1, -1)>
    %12 = cannir.extract_layout %arg4 : !cannir.tensor<<gm, bf16>, <<(-1, -1, 512)>, <(-1, 512, 1)>>, <<(-1, -1, 512)>, <(-1, 512, 1)>>, identity>
    %13 = cannir.extract_shape %12 : <<(-1, -1, 512)>, <(-1, 512, 1)>> -> <(-1, -1, 512)>
    %14 = cannir.extract_dim %13[0] : !cannir.shape<(-1, -1, 512)>
    %15 = cannir.extract_dim %13[1] : !cannir.shape<(-1, -1, 512)>
    %16 = arith.addi %15, %11 : i64
    %17 = arith.subi %16, %c512_i64 : i64
    %18 = cannir.extract_layout %arg2 : !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
    %19 = cannir.extract_shape %18 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, -1)>
    %20 = cannir.extract_dim %19[0] : !cannir.shape<(-1, -1)>
    %21 = arith.cmpi ne, %4, %c8_i64 : i64
    scf.if %21 {
      asccube.sync_all
      asccube.sync_block_wait s, %c13_i64
      asccube.sync_block_arrive fixpipe, %c14_i64 {mode = 0 : i32}
      asccube.sync_block_wait s, %c14_i64 {mode = 0 : i32}
      asccube.sync_block_arrive mte3, %c15_i64
    }
    %22 = arith.remsi %8, %c64_i64 : i64
    %23 = arith.cmpi ne, %22, %c0_i64 : i64
    %24 = arith.cmpi slt, %22, %c0_i64 : i64
    %25 = arith.andi %23, %24 : i1
    %26 = arith.divsi %8, %c64_i64 : i64
    %27 = arith.extui %25 : i1 to i64
    %28 = arith.subi %26, %27 : i64
    %29 = arith.remsi %5, %c256_i64 : i64
    %30 = arith.cmpi ne, %29, %c0_i64 : i64
    %31 = arith.cmpi slt, %29, %c0_i64 : i64
    %32 = arith.andi %30, %31 : i1
    %33 = arith.divsi %5, %c256_i64 : i64
    %34 = arith.extui %32 : i1 to i64
    %35 = arith.subi %33, %34 : i64
    %36 = scf.for %arg23 = %0 to %28 step %1 iter_args(%arg24 = %c0_i64) -> (i64)  : i64 {
      %66 = arith.remsi %arg23, %28 : i64
      %67 = arith.cmpi ne, %66, %c0_i64 : i64
      %68 = arith.cmpi slt, %66, %c0_i64 : i64
      %69 = arith.cmpi slt, %28, %c0_i64 : i64
      %70 = arith.xori %68, %69 : i1
      %71 = arith.andi %67, %70 : i1
      %72 = arith.divsi %arg23, %28 : i64
      %73 = arith.extui %71 : i1 to i64
      %74 = arith.subi %72, %73 : i64
      %75 = arith.select %71, %28, %c0_i64 : i64
      %76 = arith.addi %66, %75 : i64
      %77 = arith.remsi %35, %c4_i64 : i64
      %78 = arith.cmpi ne, %77, %c0_i64 : i64
      %79 = arith.cmpi slt, %77, %c0_i64 : i64
      %80 = arith.andi %78, %79 : i1
      %81 = arith.divsi %35, %c4_i64 : i64
      %82 = arith.extui %80 : i1 to i64
      %83 = arith.subi %81, %82 : i64
      %84 = scf.for %arg25 = %c0_i64 to %83 step %c1_i64 iter_args(%arg26 = %arg24) -> (i64)  : i64 {
        %142 = arith.cmpi eq, %arg26, %c0_i64 : i64
        %143 = ascscalar.memref_address %alloc_46 : memref<4096xbf16, 2>
        %144 = ascscalar.memref_address %alloc_45 : memref<4096xbf16, 2>
        %145 = arith.select %142, %143, %144 : i64
        %146 = arith.cmpi ne, %arg26, %c0_i64 : i64
        %147 = arith.extui %146 : i1 to i64
        %148 = arith.addi %arg26, %c1_i64 overflow<nsw> : i64
        %149 = arith.remui %148, %c2_i64 : i64
        %150 = ascscalar.bind_local_ptr %145 : memref<1xbf16, 2>
        %151 = arith.muli %74, %c8_i64 : i64
        %152 = cannir.extract_physical_layout %arg13 : !cannir.tensor<<gm, bf16>, <<(8, -1)>, <(-1, 1)>>, <<(8, -1)>, <(-1, 1)>>, identity>
        %153 = cannir.extract_stride %152 : <<(8, -1)>, <(-1, 1)>> -> <(-1, 1)>
        %154 = cannir.extract_dim %153[0] : !cannir.stride<(-1, 1)>
        %155 = arith.muli %151, %154 : i64
        %156 = arith.index_cast %155 : i64 to index
        %157 = arith.muli %arg25, %c256_i64 : i64
        %158 = arith.index_cast %157 : i64 to index
        %159 = arith.addi %156, %158 : index
        %160 = arith.minsi %74, %c1_i64 : i64
        %161 = arith.muli %160, %c8_i64 : i64
        %162 = arith.subi %c8_i64, %161 : i64
        %163 = arith.minsi %162, %c8_i64 : i64
        %164 = arith.cmpi sle, %74, %c1_i64 : i64
        %165 = arith.select %164, %163, %c0_i64 : i64
        %166 = cannir.extract_layout %arg13 : !cannir.tensor<<gm, bf16>, <<(8, -1)>, <(-1, 1)>>, <<(8, -1)>, <(-1, 1)>>, identity>
        %167 = cannir.extract_shape %166 : <<(8, -1)>, <(-1, 1)>> -> <(8, -1)>
        %168 = cannir.extract_dim %167[1] : !cannir.shape<(8, -1)>
        %169 = arith.divsi %168, %c256_i64 : i64
        %170 = arith.minsi %arg25, %169 : i64
        %171 = arith.muli %170, %c256_i64 : i64
        %172 = arith.subi %168, %171 : i64
        %173 = arith.minsi %172, %c256_i64 : i64
        %174 = arith.cmpi sle, %arg25, %169 : i64
        %175 = arith.select %174, %173, %c0_i64 : i64
        %176 = cannir.extract_stride %166 : <<(8, -1)>, <(-1, 1)>> -> <(-1, 1)>
        %177 = cannir.extract_dim %176[0] : !cannir.stride<(-1, 1)>
        %178 = arith.index_cast %177 : i64 to index
        %179 = arith.muli %178, %c2 : index
        %180 = arith.index_cast %179 : index to i64
        asccube.get_buf mte2, %147, 0 {cannir.buffer.generated_sync}
        %181 = arith.cmpi ugt, %165, %c0_i64 : i64
        %182 = arith.cmpi ugt, %175, %c0_i64 : i64
        %183 = arith.andi %181, %182 : i1
        scf.if %183 {
          %225 = arith.trunci %165 : i64 to i32
          %226 = arith.trunci %175 : i64 to i32
          asccube.copy_gm2l1 %150[%c0], %alloc_4[%159], %225, %226, %180, %c1_i32, %c0_i64, %c0_i64, %c16_i32 l2_cache_ctl %c1_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
        }
        asccube.release_buf mte2, %147, 0 {cannir.buffer.generated_sync}
        %184 = ascscalar.memref_address %alloc_44 : memref<16384xbf16, 2>
        %185 = ascscalar.memref_address %alloc_43 : memref<16384xbf16, 2>
        %186 = arith.select %142, %184, %185 : i64
        %187 = arith.select %142, %c2_i64, %c3_i64 : i64
        %188 = ascscalar.bind_local_ptr %186 : memref<1xbf16, 2>
        %189 = arith.muli %76, %c4_i64 : i64
        %190 = cannir.extract_physical_layout %arg1 : !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
        %191 = cannir.extract_stride %190 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
        %192 = cannir.extract_dim %191[0] : !cannir.stride<(-1, 1)>
        %193 = arith.muli %189, %192 : i64
        %194 = arith.muli %arg25, %c16_i64 : i64
        %195 = arith.addi %193, %194 : i64
        %196 = arith.muli %195, %c16_i64 : i64
        %197 = arith.minsi %76, %26 : i64
        %198 = arith.muli %197, %c64_i64 : i64
        %199 = arith.subi %8, %198 : i64
        %200 = arith.minsi %199, %c64_i64 : i64
        %201 = arith.cmpi sle, %76, %26 : i64
        %202 = arith.select %201, %200, %c0_i64 : i64
        %203 = cannir.extract_dim %7[1] : !cannir.shape<(-1, -1)>
        %204 = arith.divsi %203, %c256_i64 : i64
        %205 = arith.minsi %arg25, %204 : i64
        %206 = arith.muli %205, %c256_i64 : i64
        %207 = arith.subi %203, %206 : i64
        %208 = arith.minsi %207, %c256_i64 : i64
        %209 = arith.cmpi sle, %arg25, %204 : i64
        %210 = arith.select %209, %208, %c0_i64 : i64
        %211 = cannir.extract_stride %6 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
        %212 = cannir.extract_dim %211[0] : !cannir.stride<(-1, 1)>
        %213 = arith.index_cast %212 : i64 to index
        %214 = arith.muli %213, %c2 : index
        %215 = arith.index_cast %214 : index to i64
        asccube.get_buf mte2, %187, 0 {cannir.buffer.generated_sync}
        %216 = arith.cmpi ugt, %202, %c0_i64 : i64
        %217 = arith.cmpi ugt, %210, %c0_i64 : i64
        %218 = arith.andi %216, %217 : i1
        scf.if %218 {
          %225 = arith.index_cast %196 : i64 to index
          %226 = arith.trunci %202 : i64 to i32
          %227 = arith.trunci %210 : i64 to i32
          asccube.copy_gm2l1 %188[%c0], %alloc[%225], %226, %227, %215, %c1_i32, %c0_i64, %c0_i64, %c64_i32 l2_cache_ctl %c0_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
        }
        asccube.release_buf mte2, %187, 0 {cannir.buffer.generated_sync}
        %219 = ascscalar.bitcast_memref %alloc_10 byte_offset = %c0_i64 : memref<2048xbf16, 3> -> memref<?xbf16, 3>
        asccube.get_buf mte1, %c4_i64, 0 {cannir.buffer.generated_sync}
        asccube.get_buf mte1, %147, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0a %219[%c0], %150[%c0], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c1_i32, %c1_i32 : memref<?xbf16, 3>, memref<1xbf16, 2>
        asccube.release_buf mte1, %147, 0 {cannir.buffer.generated_sync}
        asccube.release_buf mte1, %c4_i64, 0 {cannir.buffer.generated_sync}
        %220 = ascscalar.bitcast_memref %alloc_11 byte_offset = %c0_i64 : memref<8192xbf16, 4> -> memref<?xbf16, 4>
        asccube.get_buf mte1, %c6_i64, 0 {cannir.buffer.generated_sync}
        asccube.get_buf mte1, %187, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0b %220[%c0], %188[%c0], %c0_i32, %c0_i32, %c4_i32, %c8_i32, %c4_i32, %c4_i32 : memref<?xbf16, 4>, memref<1xbf16, 2>
        asccube.release_buf mte1, %187, 0 {cannir.buffer.generated_sync}
        asccube.release_buf mte1, %c6_i64, 0 {cannir.buffer.generated_sync}
        %221 = arith.cmpi eq, %arg25, %c0_i64 : i64
        %222 = ascscalar.bitcast_memref %alloc_9 byte_offset = %c0_i64 : memref<1024xf32, 5> -> memref<?xf32, 5>
        asccube.get_buf m, %c8_i64, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %c4_i64, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %c6_i64, 0 {cannir.buffer.generated_sync}
        scf.if %221 {
          asccube.mmad %222[%c0], %219, %220, %c8_i32, %c128_i32, %c64_i32, %c0_i32 {c_matrix_init_val = true} : memref<?xf32, 5>, memref<?xbf16, 3>, memref<?xbf16, 4>
        } else {
          asccube.mmad %222[%c0], %219, %220, %c8_i32, %c128_i32, %c64_i32, %c0_i32 {c_matrix_init_val = false} : memref<?xf32, 5>, memref<?xbf16, 3>, memref<?xbf16, 4>
        }
        asccube.release_buf m, %c6_i64, 0 {cannir.buffer.generated_sync}
        asccube.release_buf m, %c4_i64, 0 {cannir.buffer.generated_sync}
        %223 = ascscalar.bitcast_memref %alloc_12 byte_offset = %c0_i64 : memref<2048xbf16, 3> -> memref<?xbf16, 3>
        asccube.get_buf mte1, %c5_i64, 0 {cannir.buffer.generated_sync}
        asccube.get_buf mte1, %147, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0a %223[%c0], %150[%c0], %c0_i32, %c8_i32, %c1_i32, %c8_i32, %c1_i32, %c1_i32 : memref<?xbf16, 3>, memref<1xbf16, 2>
        asccube.release_buf mte1, %147, 0 {cannir.buffer.generated_sync}
        asccube.release_buf mte1, %c5_i64, 0 {cannir.buffer.generated_sync}
        %224 = ascscalar.bitcast_memref %alloc_13 byte_offset = %c0_i64 : memref<8192xbf16, 4> -> memref<?xbf16, 4>
        asccube.get_buf mte1, %c7_i64, 0 {cannir.buffer.generated_sync}
        asccube.get_buf mte1, %187, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0b %224[%c0], %188[%c0], %c0_i32, %c8_i32, %c4_i32, %c8_i32, %c4_i32, %c4_i32 : memref<?xbf16, 4>, memref<1xbf16, 2>
        asccube.release_buf mte1, %187, 0 {cannir.buffer.generated_sync}
        asccube.release_buf mte1, %c7_i64, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %c5_i64, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %c7_i64, 0 {cannir.buffer.generated_sync}
        asccube.mmad %222[%c0], %223, %224, %c8_i32, %c128_i32, %c64_i32, %c0_i32 {c_matrix_init_val = false} : memref<?xf32, 5>, memref<?xbf16, 3>, memref<?xbf16, 4>
        asccube.release_buf m, %c7_i64, 0 {cannir.buffer.generated_sync}
        asccube.release_buf m, %c5_i64, 0 {cannir.buffer.generated_sync}
        asccube.release_buf m, %c8_i64, 0 {cannir.buffer.generated_sync}
        scf.yield %149 : i64
      }
      %85 = ascscalar.bitcast_memref %alloc_9 byte_offset = %c0_i64 : memref<1024xf32, 5> -> memref<?xf32, 5>
      %86 = arith.muli %74, %c8_i64 : i64
      %87 = cannir.extract_physical_layout %arg14 : !cannir.tensor<<gm, f32>, <<(32, -1)>, <(-1, 1)>>, <<(32, -1)>, <(-1, 1)>>, identity>
      %88 = cannir.extract_stride %87 : <<(32, -1)>, <(-1, 1)>> -> <(-1, 1)>
      %89 = cannir.extract_dim %88[0] : !cannir.stride<(-1, 1)>
      %90 = arith.muli %86, %89 : i64
      %91 = arith.index_cast %90 : i64 to index
      %92 = arith.muli %76, %c64_i64 : i64
      %93 = arith.index_cast %92 : i64 to index
      %94 = arith.addi %91, %93 : index
      %95 = cannir.extract_layout %arg14 : !cannir.tensor<<gm, f32>, <<(32, -1)>, <(-1, 1)>>, <<(32, -1)>, <(-1, 1)>>, identity>
      %96 = cannir.extract_stride %95 : <<(32, -1)>, <(-1, 1)>> -> <(-1, 1)>
      %97 = cannir.extract_dim %96[0] : !cannir.stride<(-1, 1)>
      asccube.get_buf fixpipe, %c8_i64, 0 {cannir.buffer.generated_sync}
      %98 = arith.trunci %97 : i64 to i32
      asccube.copy_l0c2gm %alloc_5[%94], %85[%c0], %c64_i32, %c8_i32, %98, %c16_i32, %c0_i32 l2_cache_ctl %c0_i32 {nz2nd_en = true, relu_en = false} : memref<1xf32>, memref<?xf32, 5>
      asccube.release_buf fixpipe, %c8_i64, 0 {cannir.buffer.generated_sync}
      %99 = arith.muli %35, %c2_i64 : i64
      %100 = arith.remsi %99, %c4_i64 : i64
      %101 = arith.cmpi ne, %100, %c0_i64 : i64
      %102 = arith.cmpi slt, %100, %c0_i64 : i64
      %103 = arith.andi %101, %102 : i1
      %104 = arith.divsi %99, %c4_i64 : i64
      %105 = arith.extui %103 : i1 to i64
      %106 = arith.subi %104, %105 : i64
      %107 = scf.for %arg25 = %83 to %106 step %c1_i64 iter_args(%arg26 = %84) -> (i64)  : i64 {
        %142 = arith.cmpi eq, %arg26, %c0_i64 : i64
        %143 = ascscalar.memref_address %alloc_46 : memref<4096xbf16, 2>
        %144 = ascscalar.memref_address %alloc_45 : memref<4096xbf16, 2>
        %145 = arith.select %142, %143, %144 : i64
        %146 = arith.cmpi ne, %arg26, %c0_i64 : i64
        %147 = arith.extui %146 : i1 to i64
        %148 = arith.addi %arg26, %c1_i64 overflow<nsw> : i64
        %149 = arith.remui %148, %c2_i64 : i64
        %150 = ascscalar.bind_local_ptr %145 : memref<1xbf16, 2>
        %151 = cannir.extract_physical_layout %arg13 : !cannir.tensor<<gm, bf16>, <<(8, -1)>, <(-1, 1)>>, <<(8, -1)>, <(-1, 1)>>, identity>
        %152 = cannir.extract_stride %151 : <<(8, -1)>, <(-1, 1)>> -> <(-1, 1)>
        %153 = cannir.extract_dim %152[0] : !cannir.stride<(-1, 1)>
        %154 = arith.muli %86, %153 : i64
        %155 = arith.index_cast %154 : i64 to index
        %156 = arith.muli %arg25, %c256_i64 : i64
        %157 = arith.index_cast %156 : i64 to index
        %158 = arith.addi %155, %157 : index
        %159 = arith.minsi %74, %c1_i64 : i64
        %160 = arith.muli %159, %c8_i64 : i64
        %161 = arith.subi %c8_i64, %160 : i64
        %162 = arith.minsi %161, %c8_i64 : i64
        %163 = arith.cmpi sle, %74, %c1_i64 : i64
        %164 = arith.select %163, %162, %c0_i64 : i64
        %165 = cannir.extract_layout %arg13 : !cannir.tensor<<gm, bf16>, <<(8, -1)>, <(-1, 1)>>, <<(8, -1)>, <(-1, 1)>>, identity>
        %166 = cannir.extract_shape %165 : <<(8, -1)>, <(-1, 1)>> -> <(8, -1)>
        %167 = cannir.extract_dim %166[1] : !cannir.shape<(8, -1)>
        %168 = arith.divsi %167, %c256_i64 : i64
        %169 = arith.minsi %arg25, %168 : i64
        %170 = arith.muli %169, %c256_i64 : i64
        %171 = arith.subi %167, %170 : i64
        %172 = arith.minsi %171, %c256_i64 : i64
        %173 = arith.cmpi sle, %arg25, %168 : i64
        %174 = arith.select %173, %172, %c0_i64 : i64
        %175 = cannir.extract_stride %165 : <<(8, -1)>, <(-1, 1)>> -> <(-1, 1)>
        %176 = cannir.extract_dim %175[0] : !cannir.stride<(-1, 1)>
        %177 = arith.index_cast %176 : i64 to index
        %178 = arith.muli %177, %c2 : index
        %179 = arith.index_cast %178 : index to i64
        asccube.get_buf mte2, %147, 0 {cannir.buffer.generated_sync}
        %180 = arith.cmpi ugt, %164, %c0_i64 : i64
        %181 = arith.cmpi ugt, %174, %c0_i64 : i64
        %182 = arith.andi %180, %181 : i1
        scf.if %182 {
          %224 = arith.trunci %164 : i64 to i32
          %225 = arith.trunci %174 : i64 to i32
          asccube.copy_gm2l1 %150[%c0], %alloc_4[%158], %224, %225, %179, %c1_i32, %c0_i64, %c0_i64, %c16_i32 l2_cache_ctl %c1_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
        }
        asccube.release_buf mte2, %147, 0 {cannir.buffer.generated_sync}
        %183 = ascscalar.memref_address %alloc_44 : memref<16384xbf16, 2>
        %184 = ascscalar.memref_address %alloc_43 : memref<16384xbf16, 2>
        %185 = arith.select %142, %183, %184 : i64
        %186 = arith.select %142, %c2_i64, %c3_i64 : i64
        %187 = ascscalar.bind_local_ptr %185 : memref<1xbf16, 2>
        %188 = arith.muli %76, %c4_i64 : i64
        %189 = cannir.extract_physical_layout %arg1 : !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
        %190 = cannir.extract_stride %189 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
        %191 = cannir.extract_dim %190[0] : !cannir.stride<(-1, 1)>
        %192 = arith.muli %188, %191 : i64
        %193 = arith.muli %arg25, %c16_i64 : i64
        %194 = arith.addi %192, %193 : i64
        %195 = arith.muli %194, %c16_i64 : i64
        %196 = arith.minsi %76, %26 : i64
        %197 = arith.muli %196, %c64_i64 : i64
        %198 = arith.subi %8, %197 : i64
        %199 = arith.minsi %198, %c64_i64 : i64
        %200 = arith.cmpi sle, %76, %26 : i64
        %201 = arith.select %200, %199, %c0_i64 : i64
        %202 = cannir.extract_dim %7[1] : !cannir.shape<(-1, -1)>
        %203 = arith.divsi %202, %c256_i64 : i64
        %204 = arith.minsi %arg25, %203 : i64
        %205 = arith.muli %204, %c256_i64 : i64
        %206 = arith.subi %202, %205 : i64
        %207 = arith.minsi %206, %c256_i64 : i64
        %208 = arith.cmpi sle, %arg25, %203 : i64
        %209 = arith.select %208, %207, %c0_i64 : i64
        %210 = cannir.extract_stride %6 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
        %211 = cannir.extract_dim %210[0] : !cannir.stride<(-1, 1)>
        %212 = arith.index_cast %211 : i64 to index
        %213 = arith.muli %212, %c2 : index
        %214 = arith.index_cast %213 : index to i64
        asccube.get_buf mte2, %186, 0 {cannir.buffer.generated_sync}
        %215 = arith.cmpi ugt, %201, %c0_i64 : i64
        %216 = arith.cmpi ugt, %209, %c0_i64 : i64
        %217 = arith.andi %215, %216 : i1
        scf.if %217 {
          %224 = arith.index_cast %195 : i64 to index
          %225 = arith.trunci %201 : i64 to i32
          %226 = arith.trunci %209 : i64 to i32
          asccube.copy_gm2l1 %187[%c0], %alloc[%224], %225, %226, %214, %c1_i32, %c0_i64, %c0_i64, %c64_i32 l2_cache_ctl %c0_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
        }
        asccube.release_buf mte2, %186, 0 {cannir.buffer.generated_sync}
        %218 = ascscalar.bitcast_memref %alloc_10 byte_offset = %c0_i64 : memref<2048xbf16, 3> -> memref<?xbf16, 3>
        asccube.get_buf mte1, %c4_i64, 0 {cannir.buffer.generated_sync}
        asccube.get_buf mte1, %147, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0a %218[%c0], %150[%c0], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c1_i32, %c1_i32 : memref<?xbf16, 3>, memref<1xbf16, 2>
        asccube.release_buf mte1, %147, 0 {cannir.buffer.generated_sync}
        asccube.release_buf mte1, %c4_i64, 0 {cannir.buffer.generated_sync}
        %219 = ascscalar.bitcast_memref %alloc_11 byte_offset = %c0_i64 : memref<8192xbf16, 4> -> memref<?xbf16, 4>
        asccube.get_buf mte1, %c6_i64, 0 {cannir.buffer.generated_sync}
        asccube.get_buf mte1, %186, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0b %219[%c0], %187[%c0], %c0_i32, %c0_i32, %c4_i32, %c8_i32, %c4_i32, %c4_i32 : memref<?xbf16, 4>, memref<1xbf16, 2>
        asccube.release_buf mte1, %186, 0 {cannir.buffer.generated_sync}
        asccube.release_buf mte1, %c6_i64, 0 {cannir.buffer.generated_sync}
        %220 = arith.cmpi eq, %arg25, %83 : i64
        %221 = ascscalar.bitcast_memref %alloc_14 byte_offset = %c0_i64 : memref<1024xf32, 5> -> memref<?xf32, 5>
        asccube.get_buf m, %c9_i64, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %c4_i64, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %c6_i64, 0 {cannir.buffer.generated_sync}
        scf.if %220 {
          asccube.mmad %221[%c0], %218, %219, %c8_i32, %c128_i32, %c64_i32, %c0_i32 {c_matrix_init_val = true} : memref<?xf32, 5>, memref<?xbf16, 3>, memref<?xbf16, 4>
        } else {
          asccube.mmad %221[%c0], %218, %219, %c8_i32, %c128_i32, %c64_i32, %c0_i32 {c_matrix_init_val = false} : memref<?xf32, 5>, memref<?xbf16, 3>, memref<?xbf16, 4>
        }
        asccube.release_buf m, %c6_i64, 0 {cannir.buffer.generated_sync}
        asccube.release_buf m, %c4_i64, 0 {cannir.buffer.generated_sync}
        %222 = ascscalar.bitcast_memref %alloc_12 byte_offset = %c0_i64 : memref<2048xbf16, 3> -> memref<?xbf16, 3>
        asccube.get_buf mte1, %c5_i64, 0 {cannir.buffer.generated_sync}
        asccube.get_buf mte1, %147, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0a %222[%c0], %150[%c0], %c0_i32, %c8_i32, %c1_i32, %c8_i32, %c1_i32, %c1_i32 : memref<?xbf16, 3>, memref<1xbf16, 2>
        asccube.release_buf mte1, %147, 0 {cannir.buffer.generated_sync}
        asccube.release_buf mte1, %c5_i64, 0 {cannir.buffer.generated_sync}
        %223 = ascscalar.bitcast_memref %alloc_13 byte_offset = %c0_i64 : memref<8192xbf16, 4> -> memref<?xbf16, 4>
        asccube.get_buf mte1, %c7_i64, 0 {cannir.buffer.generated_sync}
        asccube.get_buf mte1, %186, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0b %223[%c0], %187[%c0], %c0_i32, %c8_i32, %c4_i32, %c8_i32, %c4_i32, %c4_i32 : memref<?xbf16, 4>, memref<1xbf16, 2>
        asccube.release_buf mte1, %186, 0 {cannir.buffer.generated_sync}
        asccube.release_buf mte1, %c7_i64, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %c5_i64, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %c7_i64, 0 {cannir.buffer.generated_sync}
        asccube.mmad %221[%c0], %222, %223, %c8_i32, %c128_i32, %c64_i32, %c0_i32 {c_matrix_init_val = false} : memref<?xf32, 5>, memref<?xbf16, 3>, memref<?xbf16, 4>
        asccube.release_buf m, %c7_i64, 0 {cannir.buffer.generated_sync}
        asccube.release_buf m, %c5_i64, 0 {cannir.buffer.generated_sync}
        asccube.release_buf m, %c9_i64, 0 {cannir.buffer.generated_sync}
        scf.yield %149 : i64
      }
      %108 = arith.addi %74, %c1_i64 : i64
      %109 = ascscalar.bitcast_memref %alloc_14 byte_offset = %c0_i64 : memref<1024xf32, 5> -> memref<?xf32, 5>
      %110 = arith.muli %108, %c8_i64 : i64
      %111 = arith.muli %110, %89 : i64
      %112 = arith.index_cast %111 : i64 to index
      %113 = arith.addi %112, %93 : index
      asccube.get_buf fixpipe, %c9_i64, 0 {cannir.buffer.generated_sync}
      asccube.copy_l0c2gm %alloc_5[%113], %109[%c0], %c64_i32, %c8_i32, %98, %c16_i32, %c0_i32 l2_cache_ctl %c0_i32 {nz2nd_en = true, relu_en = false} : memref<1xf32>, memref<?xf32, 5>
      asccube.release_buf fixpipe, %c9_i64, 0 {cannir.buffer.generated_sync}
      %114 = arith.muli %35, %c3_i64 : i64
      %115 = arith.remsi %114, %c4_i64 : i64
      %116 = arith.cmpi ne, %115, %c0_i64 : i64
      %117 = arith.cmpi slt, %115, %c0_i64 : i64
      %118 = arith.andi %116, %117 : i1
      %119 = arith.divsi %114, %c4_i64 : i64
      %120 = arith.extui %118 : i1 to i64
      %121 = arith.subi %119, %120 : i64
      %122 = scf.for %arg25 = %106 to %121 step %c1_i64 iter_args(%arg26 = %107) -> (i64)  : i64 {
        %142 = arith.cmpi eq, %arg26, %c0_i64 : i64
        %143 = ascscalar.memref_address %alloc_46 : memref<4096xbf16, 2>
        %144 = ascscalar.memref_address %alloc_45 : memref<4096xbf16, 2>
        %145 = arith.select %142, %143, %144 : i64
        %146 = arith.cmpi ne, %arg26, %c0_i64 : i64
        %147 = arith.extui %146 : i1 to i64
        %148 = arith.addi %arg26, %c1_i64 overflow<nsw> : i64
        %149 = arith.remui %148, %c2_i64 : i64
        %150 = ascscalar.bind_local_ptr %145 : memref<1xbf16, 2>
        %151 = cannir.extract_physical_layout %arg13 : !cannir.tensor<<gm, bf16>, <<(8, -1)>, <(-1, 1)>>, <<(8, -1)>, <(-1, 1)>>, identity>
        %152 = cannir.extract_stride %151 : <<(8, -1)>, <(-1, 1)>> -> <(-1, 1)>
        %153 = cannir.extract_dim %152[0] : !cannir.stride<(-1, 1)>
        %154 = arith.muli %86, %153 : i64
        %155 = arith.index_cast %154 : i64 to index
        %156 = arith.muli %arg25, %c256_i64 : i64
        %157 = arith.index_cast %156 : i64 to index
        %158 = arith.addi %155, %157 : index
        %159 = arith.minsi %74, %c1_i64 : i64
        %160 = arith.muli %159, %c8_i64 : i64
        %161 = arith.subi %c8_i64, %160 : i64
        %162 = arith.minsi %161, %c8_i64 : i64
        %163 = arith.cmpi sle, %74, %c1_i64 : i64
        %164 = arith.select %163, %162, %c0_i64 : i64
        %165 = cannir.extract_layout %arg13 : !cannir.tensor<<gm, bf16>, <<(8, -1)>, <(-1, 1)>>, <<(8, -1)>, <(-1, 1)>>, identity>
        %166 = cannir.extract_shape %165 : <<(8, -1)>, <(-1, 1)>> -> <(8, -1)>
        %167 = cannir.extract_dim %166[1] : !cannir.shape<(8, -1)>
        %168 = arith.divsi %167, %c256_i64 : i64
        %169 = arith.minsi %arg25, %168 : i64
        %170 = arith.muli %169, %c256_i64 : i64
        %171 = arith.subi %167, %170 : i64
        %172 = arith.minsi %171, %c256_i64 : i64
        %173 = arith.cmpi sle, %arg25, %168 : i64
        %174 = arith.select %173, %172, %c0_i64 : i64
        %175 = cannir.extract_stride %165 : <<(8, -1)>, <(-1, 1)>> -> <(-1, 1)>
        %176 = cannir.extract_dim %175[0] : !cannir.stride<(-1, 1)>
        %177 = arith.index_cast %176 : i64 to index
        %178 = arith.muli %177, %c2 : index
        %179 = arith.index_cast %178 : index to i64
        asccube.get_buf mte2, %147, 0 {cannir.buffer.generated_sync}
        %180 = arith.cmpi ugt, %164, %c0_i64 : i64
        %181 = arith.cmpi ugt, %174, %c0_i64 : i64
        %182 = arith.andi %180, %181 : i1
        scf.if %182 {
          %223 = arith.trunci %164 : i64 to i32
          %224 = arith.trunci %174 : i64 to i32
          asccube.copy_gm2l1 %150[%c0], %alloc_4[%158], %223, %224, %179, %c1_i32, %c0_i64, %c0_i64, %c16_i32 l2_cache_ctl %c1_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
        }
        asccube.release_buf mte2, %147, 0 {cannir.buffer.generated_sync}
        %183 = ascscalar.memref_address %alloc_44 : memref<16384xbf16, 2>
        %184 = ascscalar.memref_address %alloc_43 : memref<16384xbf16, 2>
        %185 = arith.select %142, %183, %184 : i64
        %186 = arith.select %142, %c2_i64, %c3_i64 : i64
        %187 = ascscalar.bind_local_ptr %185 : memref<1xbf16, 2>
        %188 = arith.muli %76, %c4_i64 : i64
        %189 = cannir.extract_physical_layout %arg1 : !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
        %190 = cannir.extract_stride %189 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
        %191 = cannir.extract_dim %190[0] : !cannir.stride<(-1, 1)>
        %192 = arith.muli %188, %191 : i64
        %193 = arith.muli %arg25, %c16_i64 : i64
        %194 = arith.addi %192, %193 : i64
        %195 = arith.muli %194, %c16_i64 : i64
        %196 = arith.minsi %76, %26 : i64
        %197 = arith.muli %196, %c64_i64 : i64
        %198 = arith.subi %8, %197 : i64
        %199 = arith.minsi %198, %c64_i64 : i64
        %200 = arith.cmpi sle, %76, %26 : i64
        %201 = arith.select %200, %199, %c0_i64 : i64
        %202 = cannir.extract_dim %7[1] : !cannir.shape<(-1, -1)>
        %203 = arith.divsi %202, %c256_i64 : i64
        %204 = arith.minsi %arg25, %203 : i64
        %205 = arith.muli %204, %c256_i64 : i64
        %206 = arith.subi %202, %205 : i64
        %207 = arith.minsi %206, %c256_i64 : i64
        %208 = arith.cmpi sle, %arg25, %203 : i64
        %209 = arith.select %208, %207, %c0_i64 : i64
        %210 = cannir.extract_stride %6 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
        %211 = cannir.extract_dim %210[0] : !cannir.stride<(-1, 1)>
        %212 = arith.index_cast %211 : i64 to index
        %213 = arith.muli %212, %c2 : index
        %214 = arith.index_cast %213 : index to i64
        asccube.get_buf mte2, %186, 0 {cannir.buffer.generated_sync}
        %215 = arith.cmpi ugt, %201, %c0_i64 : i64
        %216 = arith.cmpi ugt, %209, %c0_i64 : i64
        %217 = arith.andi %215, %216 : i1
        scf.if %217 {
          %223 = arith.index_cast %195 : i64 to index
          %224 = arith.trunci %201 : i64 to i32
          %225 = arith.trunci %209 : i64 to i32
          asccube.copy_gm2l1 %187[%c0], %alloc[%223], %224, %225, %214, %c1_i32, %c0_i64, %c0_i64, %c64_i32 l2_cache_ctl %c0_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
        }
        asccube.release_buf mte2, %186, 0 {cannir.buffer.generated_sync}
        %218 = ascscalar.bitcast_memref %alloc_10 byte_offset = %c0_i64 : memref<2048xbf16, 3> -> memref<?xbf16, 3>
        asccube.get_buf mte1, %c4_i64, 0 {cannir.buffer.generated_sync}
        asccube.get_buf mte1, %147, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0a %218[%c0], %150[%c0], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c1_i32, %c1_i32 : memref<?xbf16, 3>, memref<1xbf16, 2>
        asccube.release_buf mte1, %147, 0 {cannir.buffer.generated_sync}
        asccube.release_buf mte1, %c4_i64, 0 {cannir.buffer.generated_sync}
        %219 = ascscalar.bitcast_memref %alloc_11 byte_offset = %c0_i64 : memref<8192xbf16, 4> -> memref<?xbf16, 4>
        asccube.get_buf mte1, %c6_i64, 0 {cannir.buffer.generated_sync}
        asccube.get_buf mte1, %186, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0b %219[%c0], %187[%c0], %c0_i32, %c0_i32, %c4_i32, %c8_i32, %c4_i32, %c4_i32 : memref<?xbf16, 4>, memref<1xbf16, 2>
        asccube.release_buf mte1, %186, 0 {cannir.buffer.generated_sync}
        asccube.release_buf mte1, %c6_i64, 0 {cannir.buffer.generated_sync}
        %220 = arith.cmpi eq, %arg25, %106 : i64
        asccube.get_buf m, %c8_i64, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %c4_i64, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %c6_i64, 0 {cannir.buffer.generated_sync}
        scf.if %220 {
          asccube.mmad %85[%c0], %218, %219, %c8_i32, %c128_i32, %c64_i32, %c0_i32 {c_matrix_init_val = true} : memref<?xf32, 5>, memref<?xbf16, 3>, memref<?xbf16, 4>
        } else {
          asccube.mmad %85[%c0], %218, %219, %c8_i32, %c128_i32, %c64_i32, %c0_i32 {c_matrix_init_val = false} : memref<?xf32, 5>, memref<?xbf16, 3>, memref<?xbf16, 4>
        }
        asccube.release_buf m, %c6_i64, 0 {cannir.buffer.generated_sync}
        asccube.release_buf m, %c4_i64, 0 {cannir.buffer.generated_sync}
        %221 = ascscalar.bitcast_memref %alloc_12 byte_offset = %c0_i64 : memref<2048xbf16, 3> -> memref<?xbf16, 3>
        asccube.get_buf mte1, %c5_i64, 0 {cannir.buffer.generated_sync}
        asccube.get_buf mte1, %147, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0a %221[%c0], %150[%c0], %c0_i32, %c8_i32, %c1_i32, %c8_i32, %c1_i32, %c1_i32 : memref<?xbf16, 3>, memref<1xbf16, 2>
        asccube.release_buf mte1, %147, 0 {cannir.buffer.generated_sync}
        asccube.release_buf mte1, %c5_i64, 0 {cannir.buffer.generated_sync}
        %222 = ascscalar.bitcast_memref %alloc_13 byte_offset = %c0_i64 : memref<8192xbf16, 4> -> memref<?xbf16, 4>
        asccube.get_buf mte1, %c7_i64, 0 {cannir.buffer.generated_sync}
        asccube.get_buf mte1, %186, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0b %222[%c0], %187[%c0], %c0_i32, %c8_i32, %c4_i32, %c8_i32, %c4_i32, %c4_i32 : memref<?xbf16, 4>, memref<1xbf16, 2>
        asccube.release_buf mte1, %186, 0 {cannir.buffer.generated_sync}
        asccube.release_buf mte1, %c7_i64, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %c5_i64, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %c7_i64, 0 {cannir.buffer.generated_sync}
        asccube.mmad %85[%c0], %221, %222, %c8_i32, %c128_i32, %c64_i32, %c0_i32 {c_matrix_init_val = false} : memref<?xf32, 5>, memref<?xbf16, 3>, memref<?xbf16, 4>
        asccube.release_buf m, %c7_i64, 0 {cannir.buffer.generated_sync}
        asccube.release_buf m, %c5_i64, 0 {cannir.buffer.generated_sync}
        asccube.release_buf m, %c8_i64, 0 {cannir.buffer.generated_sync}
        scf.yield %149 : i64
      }
      %123 = arith.addi %74, %c2_i64 : i64
      %124 = arith.muli %123, %c8_i64 : i64
      %125 = arith.muli %124, %89 : i64
      %126 = arith.index_cast %125 : i64 to index
      %127 = arith.addi %126, %93 : index
      asccube.get_buf fixpipe, %c8_i64, 0 {cannir.buffer.generated_sync}
      asccube.copy_l0c2gm %alloc_5[%127], %85[%c0], %c64_i32, %c8_i32, %98, %c16_i32, %c0_i32 l2_cache_ctl %c0_i32 {nz2nd_en = true, relu_en = false} : memref<1xf32>, memref<?xf32, 5>
      asccube.release_buf fixpipe, %c8_i64, 0 {cannir.buffer.generated_sync}
      %128 = arith.muli %35, %c4_i64 : i64
      %129 = arith.remsi %128, %c4_i64 : i64
      %130 = arith.cmpi ne, %129, %c0_i64 : i64
      %131 = arith.cmpi slt, %129, %c0_i64 : i64
      %132 = arith.andi %130, %131 : i1
      %133 = arith.divsi %128, %c4_i64 : i64
      %134 = arith.extui %132 : i1 to i64
      %135 = arith.subi %133, %134 : i64
      %136 = scf.for %arg25 = %121 to %135 step %c1_i64 iter_args(%arg26 = %122) -> (i64)  : i64 {
        %142 = arith.cmpi eq, %arg26, %c0_i64 : i64
        %143 = ascscalar.memref_address %alloc_46 : memref<4096xbf16, 2>
        %144 = ascscalar.memref_address %alloc_45 : memref<4096xbf16, 2>
        %145 = arith.select %142, %143, %144 : i64
        %146 = arith.cmpi ne, %arg26, %c0_i64 : i64
        %147 = arith.extui %146 : i1 to i64
        %148 = arith.addi %arg26, %c1_i64 overflow<nsw> : i64
        %149 = arith.remui %148, %c2_i64 : i64
        %150 = ascscalar.bind_local_ptr %145 : memref<1xbf16, 2>
        %151 = cannir.extract_physical_layout %arg13 : !cannir.tensor<<gm, bf16>, <<(8, -1)>, <(-1, 1)>>, <<(8, -1)>, <(-1, 1)>>, identity>
        %152 = cannir.extract_stride %151 : <<(8, -1)>, <(-1, 1)>> -> <(-1, 1)>
        %153 = cannir.extract_dim %152[0] : !cannir.stride<(-1, 1)>
        %154 = arith.muli %86, %153 : i64
        %155 = arith.index_cast %154 : i64 to index
        %156 = arith.muli %arg25, %c256_i64 : i64
        %157 = arith.index_cast %156 : i64 to index
        %158 = arith.addi %155, %157 : index
        %159 = arith.minsi %74, %c1_i64 : i64
        %160 = arith.muli %159, %c8_i64 : i64
        %161 = arith.subi %c8_i64, %160 : i64
        %162 = arith.minsi %161, %c8_i64 : i64
        %163 = arith.cmpi sle, %74, %c1_i64 : i64
        %164 = arith.select %163, %162, %c0_i64 : i64
        %165 = cannir.extract_layout %arg13 : !cannir.tensor<<gm, bf16>, <<(8, -1)>, <(-1, 1)>>, <<(8, -1)>, <(-1, 1)>>, identity>
        %166 = cannir.extract_shape %165 : <<(8, -1)>, <(-1, 1)>> -> <(8, -1)>
        %167 = cannir.extract_dim %166[1] : !cannir.shape<(8, -1)>
        %168 = arith.divsi %167, %c256_i64 : i64
        %169 = arith.minsi %arg25, %168 : i64
        %170 = arith.muli %169, %c256_i64 : i64
        %171 = arith.subi %167, %170 : i64
        %172 = arith.minsi %171, %c256_i64 : i64
        %173 = arith.cmpi sle, %arg25, %168 : i64
        %174 = arith.select %173, %172, %c0_i64 : i64
        %175 = cannir.extract_stride %165 : <<(8, -1)>, <(-1, 1)>> -> <(-1, 1)>
        %176 = cannir.extract_dim %175[0] : !cannir.stride<(-1, 1)>
        %177 = arith.index_cast %176 : i64 to index
        %178 = arith.muli %177, %c2 : index
        %179 = arith.index_cast %178 : index to i64
        asccube.get_buf mte2, %147, 0 {cannir.buffer.generated_sync}
        %180 = arith.cmpi ugt, %164, %c0_i64 : i64
        %181 = arith.cmpi ugt, %174, %c0_i64 : i64
        %182 = arith.andi %180, %181 : i1
        scf.if %182 {
          %223 = arith.trunci %164 : i64 to i32
          %224 = arith.trunci %174 : i64 to i32
          asccube.copy_gm2l1 %150[%c0], %alloc_4[%158], %223, %224, %179, %c1_i32, %c0_i64, %c0_i64, %c16_i32 l2_cache_ctl %c1_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
        }
        asccube.release_buf mte2, %147, 0 {cannir.buffer.generated_sync}
        %183 = ascscalar.memref_address %alloc_44 : memref<16384xbf16, 2>
        %184 = ascscalar.memref_address %alloc_43 : memref<16384xbf16, 2>
        %185 = arith.select %142, %183, %184 : i64
        %186 = arith.select %142, %c2_i64, %c3_i64 : i64
        %187 = ascscalar.bind_local_ptr %185 : memref<1xbf16, 2>
        %188 = arith.muli %76, %c4_i64 : i64
        %189 = cannir.extract_physical_layout %arg1 : !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
        %190 = cannir.extract_stride %189 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
        %191 = cannir.extract_dim %190[0] : !cannir.stride<(-1, 1)>
        %192 = arith.muli %188, %191 : i64
        %193 = arith.muli %arg25, %c16_i64 : i64
        %194 = arith.addi %192, %193 : i64
        %195 = arith.muli %194, %c16_i64 : i64
        %196 = arith.minsi %76, %26 : i64
        %197 = arith.muli %196, %c64_i64 : i64
        %198 = arith.subi %8, %197 : i64
        %199 = arith.minsi %198, %c64_i64 : i64
        %200 = arith.cmpi sle, %76, %26 : i64
        %201 = arith.select %200, %199, %c0_i64 : i64
        %202 = cannir.extract_dim %7[1] : !cannir.shape<(-1, -1)>
        %203 = arith.divsi %202, %c256_i64 : i64
        %204 = arith.minsi %arg25, %203 : i64
        %205 = arith.muli %204, %c256_i64 : i64
        %206 = arith.subi %202, %205 : i64
        %207 = arith.minsi %206, %c256_i64 : i64
        %208 = arith.cmpi sle, %arg25, %203 : i64
        %209 = arith.select %208, %207, %c0_i64 : i64
        %210 = cannir.extract_stride %6 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
        %211 = cannir.extract_dim %210[0] : !cannir.stride<(-1, 1)>
        %212 = arith.index_cast %211 : i64 to index
        %213 = arith.muli %212, %c2 : index
        %214 = arith.index_cast %213 : index to i64
        asccube.get_buf mte2, %186, 0 {cannir.buffer.generated_sync}
        %215 = arith.cmpi ugt, %201, %c0_i64 : i64
        %216 = arith.cmpi ugt, %209, %c0_i64 : i64
        %217 = arith.andi %215, %216 : i1
        scf.if %217 {
          %223 = arith.index_cast %195 : i64 to index
          %224 = arith.trunci %201 : i64 to i32
          %225 = arith.trunci %209 : i64 to i32
          asccube.copy_gm2l1 %187[%c0], %alloc[%223], %224, %225, %214, %c1_i32, %c0_i64, %c0_i64, %c64_i32 l2_cache_ctl %c0_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
        }
        asccube.release_buf mte2, %186, 0 {cannir.buffer.generated_sync}
        %218 = ascscalar.bitcast_memref %alloc_10 byte_offset = %c0_i64 : memref<2048xbf16, 3> -> memref<?xbf16, 3>
        asccube.get_buf mte1, %c4_i64, 0 {cannir.buffer.generated_sync}
        asccube.get_buf mte1, %147, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0a %218[%c0], %150[%c0], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c1_i32, %c1_i32 : memref<?xbf16, 3>, memref<1xbf16, 2>
        asccube.release_buf mte1, %147, 0 {cannir.buffer.generated_sync}
        asccube.release_buf mte1, %c4_i64, 0 {cannir.buffer.generated_sync}
        %219 = ascscalar.bitcast_memref %alloc_11 byte_offset = %c0_i64 : memref<8192xbf16, 4> -> memref<?xbf16, 4>
        asccube.get_buf mte1, %c6_i64, 0 {cannir.buffer.generated_sync}
        asccube.get_buf mte1, %186, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0b %219[%c0], %187[%c0], %c0_i32, %c0_i32, %c4_i32, %c8_i32, %c4_i32, %c4_i32 : memref<?xbf16, 4>, memref<1xbf16, 2>
        asccube.release_buf mte1, %186, 0 {cannir.buffer.generated_sync}
        asccube.release_buf mte1, %c6_i64, 0 {cannir.buffer.generated_sync}
        %220 = arith.cmpi eq, %arg25, %121 : i64
        asccube.get_buf m, %c9_i64, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %c4_i64, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %c6_i64, 0 {cannir.buffer.generated_sync}
        scf.if %220 {
          asccube.mmad %109[%c0], %218, %219, %c8_i32, %c128_i32, %c64_i32, %c0_i32 {c_matrix_init_val = true} : memref<?xf32, 5>, memref<?xbf16, 3>, memref<?xbf16, 4>
        } else {
          asccube.mmad %109[%c0], %218, %219, %c8_i32, %c128_i32, %c64_i32, %c0_i32 {c_matrix_init_val = false} : memref<?xf32, 5>, memref<?xbf16, 3>, memref<?xbf16, 4>
        }
        asccube.release_buf m, %c6_i64, 0 {cannir.buffer.generated_sync}
        asccube.release_buf m, %c4_i64, 0 {cannir.buffer.generated_sync}
        %221 = ascscalar.bitcast_memref %alloc_12 byte_offset = %c0_i64 : memref<2048xbf16, 3> -> memref<?xbf16, 3>
        asccube.get_buf mte1, %c5_i64, 0 {cannir.buffer.generated_sync}
        asccube.get_buf mte1, %147, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0a %221[%c0], %150[%c0], %c0_i32, %c8_i32, %c1_i32, %c8_i32, %c1_i32, %c1_i32 : memref<?xbf16, 3>, memref<1xbf16, 2>
        asccube.release_buf mte1, %147, 0 {cannir.buffer.generated_sync}
        asccube.release_buf mte1, %c5_i64, 0 {cannir.buffer.generated_sync}
        %222 = ascscalar.bitcast_memref %alloc_13 byte_offset = %c0_i64 : memref<8192xbf16, 4> -> memref<?xbf16, 4>
        asccube.get_buf mte1, %c7_i64, 0 {cannir.buffer.generated_sync}
        asccube.get_buf mte1, %186, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0b %222[%c0], %187[%c0], %c0_i32, %c8_i32, %c4_i32, %c8_i32, %c4_i32, %c4_i32 : memref<?xbf16, 4>, memref<1xbf16, 2>
        asccube.release_buf mte1, %186, 0 {cannir.buffer.generated_sync}
        asccube.release_buf mte1, %c7_i64, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %c5_i64, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %c7_i64, 0 {cannir.buffer.generated_sync}
        asccube.mmad %109[%c0], %221, %222, %c8_i32, %c128_i32, %c64_i32, %c0_i32 {c_matrix_init_val = false} : memref<?xf32, 5>, memref<?xbf16, 3>, memref<?xbf16, 4>
        asccube.release_buf m, %c7_i64, 0 {cannir.buffer.generated_sync}
        asccube.release_buf m, %c5_i64, 0 {cannir.buffer.generated_sync}
        asccube.release_buf m, %c9_i64, 0 {cannir.buffer.generated_sync}
        scf.yield %149 : i64
      }
      %137 = arith.addi %74, %c3_i64 : i64
      %138 = arith.muli %137, %c8_i64 : i64
      %139 = arith.muli %138, %89 : i64
      %140 = arith.index_cast %139 : i64 to index
      %141 = arith.addi %140, %93 : index
      asccube.get_buf fixpipe, %c9_i64, 0 {cannir.buffer.generated_sync}
      asccube.copy_l0c2gm %alloc_5[%141], %109[%c0], %c64_i32, %c8_i32, %98, %c16_i32, %c0_i32 l2_cache_ctl %c0_i32 {nz2nd_en = true, relu_en = false} : memref<1xf32>, memref<?xf32, 5>
      asccube.release_buf fixpipe, %c9_i64, 0 {cannir.buffer.generated_sync}
      scf.yield %136 : i64
    }
    asccube.sync_all
    asccube.sync_block_wait s, %c13_i64
    asccube.sync_block_arrive fixpipe, %c14_i64 {mode = 0 : i32}
    asccube.sync_block_wait s, %c14_i64 {mode = 0 : i32}
    asccube.sync_block_arrive mte3, %c15_i64
    %37 = arith.remsi %11, %c64_i64 : i64
    %38 = arith.cmpi ne, %37, %c0_i64 : i64
    %39 = arith.cmpi slt, %37, %c0_i64 : i64
    %40 = arith.andi %38, %39 : i1
    %41 = arith.divsi %11, %c64_i64 : i64
    %42 = arith.extui %40 : i1 to i64
    %43 = arith.subi %41, %42 : i64
    %44:3 = scf.for %arg23 = %0 to %43 step %1 iter_args(%arg24 = %c0_i64, %arg25 = %c0_i64, %arg26 = %c0_i64) -> (i64, i64, i64)  : i64 {
      %66 = arith.remsi %arg23, %43 : i64
      %67 = arith.cmpi ne, %66, %c0_i64 : i64
      %68 = arith.cmpi slt, %66, %c0_i64 : i64
      %69 = arith.cmpi slt, %43, %c0_i64 : i64
      %70 = arith.xori %68, %69 : i1
      %71 = arith.andi %67, %70 : i1
      %72 = arith.divsi %arg23, %43 : i64
      %73 = arith.extui %71 : i1 to i64
      %74 = arith.subi %72, %73 : i64
      %75 = arith.select %71, %43, %c0_i64 : i64
      %76 = arith.addi %66, %75 : i64
      %77 = arith.cmpi eq, %arg26, %c0_i64 : i64
      %78 = ascscalar.memref_address %alloc_42 : memref<1024xf32, 5>
      %79 = ascscalar.memref_address %alloc_41 : memref<1024xf32, 5>
      %80 = arith.select %77, %78, %79 : i64
      %81 = arith.select %77, %c8_i64, %c9_i64 : i64
      %82 = arith.addi %arg26, %c1_i64 overflow<nsw> : i64
      %83 = arith.remui %82, %c2_i64 : i64
      %84:2 = scf.for %arg27 = %c0_i64 to %35 step %c1_i64 iter_args(%arg28 = %arg24, %arg29 = %arg25) -> (i64, i64)  : i64 {
        %99 = arith.cmpi eq, %arg28, %c0_i64 : i64
        %100 = ascscalar.memref_address %alloc_40 : memref<4096xbf16, 2>
        %101 = ascscalar.memref_address %alloc_39 : memref<4096xbf16, 2>
        %102 = arith.select %99, %100, %101 : i64
        %103 = arith.cmpi ne, %arg28, %c0_i64 : i64
        %104 = arith.extui %103 : i1 to i64
        %105 = arith.addi %arg28, %c1_i64 overflow<nsw> : i64
        %106 = arith.remui %105, %c2_i64 : i64
        %107 = ascscalar.bind_local_ptr %102 : memref<1xbf16, 2>
        %108 = arith.muli %74, %c8_i64 : i64
        %109 = cannir.extract_physical_layout %arg13 : !cannir.tensor<<gm, bf16>, <<(8, -1)>, <(-1, 1)>>, <<(8, -1)>, <(-1, 1)>>, identity>
        %110 = cannir.extract_stride %109 : <<(8, -1)>, <(-1, 1)>> -> <(-1, 1)>
        %111 = cannir.extract_dim %110[0] : !cannir.stride<(-1, 1)>
        %112 = arith.muli %108, %111 : i64
        %113 = arith.index_cast %112 : i64 to index
        %114 = arith.muli %arg27, %c256_i64 : i64
        %115 = arith.index_cast %114 : i64 to index
        %116 = arith.addi %113, %115 : index
        %117 = arith.minsi %74, %c1_i64 : i64
        %118 = arith.muli %117, %c8_i64 : i64
        %119 = arith.subi %c8_i64, %118 : i64
        %120 = arith.minsi %119, %c8_i64 : i64
        %121 = arith.cmpi sle, %74, %c1_i64 : i64
        %122 = arith.select %121, %120, %c0_i64 : i64
        %123 = cannir.extract_layout %arg13 : !cannir.tensor<<gm, bf16>, <<(8, -1)>, <(-1, 1)>>, <<(8, -1)>, <(-1, 1)>>, identity>
        %124 = cannir.extract_shape %123 : <<(8, -1)>, <(-1, 1)>> -> <(8, -1)>
        %125 = cannir.extract_dim %124[1] : !cannir.shape<(8, -1)>
        %126 = arith.divsi %125, %c256_i64 : i64
        %127 = arith.minsi %arg27, %126 : i64
        %128 = arith.muli %127, %c256_i64 : i64
        %129 = arith.subi %125, %128 : i64
        %130 = arith.minsi %129, %c256_i64 : i64
        %131 = arith.cmpi sle, %arg27, %126 : i64
        %132 = arith.select %131, %130, %c0_i64 : i64
        %133 = cannir.extract_stride %123 : <<(8, -1)>, <(-1, 1)>> -> <(-1, 1)>
        %134 = cannir.extract_dim %133[0] : !cannir.stride<(-1, 1)>
        %135 = arith.index_cast %134 : i64 to index
        %136 = arith.muli %135, %c2 : index
        %137 = arith.index_cast %136 : index to i64
        asccube.get_buf mte2, %104, 0 {cannir.buffer.generated_sync}
        %138 = arith.cmpi ugt, %122, %c0_i64 : i64
        %139 = arith.cmpi ugt, %132, %c0_i64 : i64
        %140 = arith.andi %138, %139 : i1
        scf.if %140 {
          %200 = arith.trunci %122 : i64 to i32
          %201 = arith.trunci %132 : i64 to i32
          asccube.copy_gm2l1 %107[%c0], %alloc_4[%116], %200, %201, %137, %c1_i32, %c0_i64, %c0_i64, %c16_i32 l2_cache_ctl %c1_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
        }
        asccube.release_buf mte2, %104, 0 {cannir.buffer.generated_sync}
        %141 = ascscalar.memref_address %alloc_38 : memref<16384xbf16, 2>
        %142 = ascscalar.memref_address %alloc_37 : memref<16384xbf16, 2>
        %143 = arith.select %99, %141, %142 : i64
        %144 = arith.select %99, %c2_i64, %c3_i64 : i64
        %145 = ascscalar.bind_local_ptr %143 : memref<1xbf16, 2>
        %146 = arith.muli %76, %c4_i64 : i64
        %147 = cannir.extract_physical_layout %arg3 : !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
        %148 = cannir.extract_stride %147 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
        %149 = cannir.extract_dim %148[0] : !cannir.stride<(-1, 1)>
        %150 = arith.muli %146, %149 : i64
        %151 = arith.muli %arg27, %c16_i64 : i64
        %152 = arith.addi %150, %151 : i64
        %153 = arith.muli %152, %c16_i64 : i64
        %154 = arith.minsi %76, %41 : i64
        %155 = arith.muli %154, %c64_i64 : i64
        %156 = arith.subi %11, %155 : i64
        %157 = arith.minsi %156, %c64_i64 : i64
        %158 = arith.cmpi sle, %76, %41 : i64
        %159 = arith.select %158, %157, %c0_i64 : i64
        %160 = cannir.extract_dim %10[1] : !cannir.shape<(-1, -1)>
        %161 = arith.divsi %160, %c256_i64 : i64
        %162 = arith.minsi %arg27, %161 : i64
        %163 = arith.muli %162, %c256_i64 : i64
        %164 = arith.subi %160, %163 : i64
        %165 = arith.minsi %164, %c256_i64 : i64
        %166 = arith.cmpi sle, %arg27, %161 : i64
        %167 = arith.select %166, %165, %c0_i64 : i64
        %168 = cannir.extract_stride %9 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
        %169 = cannir.extract_dim %168[0] : !cannir.stride<(-1, 1)>
        %170 = arith.index_cast %169 : i64 to index
        %171 = arith.muli %170, %c2 : index
        %172 = arith.index_cast %171 : index to i64
        asccube.get_buf mte2, %144, 0 {cannir.buffer.generated_sync}
        %173 = arith.cmpi ugt, %159, %c0_i64 : i64
        %174 = arith.cmpi ugt, %167, %c0_i64 : i64
        %175 = arith.andi %173, %174 : i1
        scf.if %175 {
          %200 = arith.index_cast %153 : i64 to index
          %201 = arith.trunci %159 : i64 to i32
          %202 = arith.trunci %167 : i64 to i32
          asccube.copy_gm2l1 %145[%c0], %alloc_1[%200], %201, %202, %172, %c1_i32, %c0_i64, %c0_i64, %c64_i32 l2_cache_ctl %c0_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
        }
        asccube.release_buf mte2, %144, 0 {cannir.buffer.generated_sync}
        %176 = arith.cmpi eq, %arg29, %c0_i64 : i64
        %177 = ascscalar.memref_address %alloc_36 : memref<2048xbf16, 3>
        %178 = ascscalar.memref_address %alloc_35 : memref<2048xbf16, 3>
        %179 = arith.select %176, %177, %178 : i64
        %180 = arith.select %176, %c4_i64, %c5_i64 : i64
        %181 = arith.addi %arg29, %c1_i64 overflow<nsw> : i64
        %182 = arith.remui %181, %c2_i64 : i64
        %183 = ascscalar.bind_local_ptr %179 : memref<1xbf16, 3>
        asccube.get_buf mte1, %180, 0 {cannir.buffer.generated_sync}
        asccube.get_buf mte1, %104, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0a %183[%c0], %107[%c0], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c1_i32, %c1_i32 : memref<1xbf16, 3>, memref<1xbf16, 2>
        asccube.release_buf mte1, %104, 0 {cannir.buffer.generated_sync}
        asccube.release_buf mte1, %180, 0 {cannir.buffer.generated_sync}
        %184 = ascscalar.memref_address %alloc_34 : memref<8192xbf16, 4>
        %185 = ascscalar.memref_address %alloc_33 : memref<8192xbf16, 4>
        %186 = arith.select %176, %184, %185 : i64
        %187 = arith.select %176, %c6_i64, %c7_i64 : i64
        %188 = ascscalar.bind_local_ptr %186 : memref<1xbf16, 4>
        asccube.get_buf mte1, %187, 0 {cannir.buffer.generated_sync}
        asccube.get_buf mte1, %144, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0b %188[%c0], %145[%c0], %c0_i32, %c0_i32, %c4_i32, %c8_i32, %c4_i32, %c4_i32 : memref<1xbf16, 4>, memref<1xbf16, 2>
        asccube.release_buf mte1, %144, 0 {cannir.buffer.generated_sync}
        asccube.release_buf mte1, %187, 0 {cannir.buffer.generated_sync}
        %189 = arith.cmpi eq, %arg27, %c0_i64 : i64
        %190 = ascscalar.bind_local_ptr %80 : memref<1xf32, 5>
        asccube.get_buf m, %81, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %180, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %187, 0 {cannir.buffer.generated_sync}
        scf.if %189 {
          asccube.mmad %190[%c0], %183, %188, %c8_i32, %c128_i32, %c64_i32, %c0_i32 {c_matrix_init_val = true} : memref<1xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
        } else {
          asccube.mmad %190[%c0], %183, %188, %c8_i32, %c128_i32, %c64_i32, %c0_i32 {c_matrix_init_val = false} : memref<1xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
        }
        asccube.release_buf m, %187, 0 {cannir.buffer.generated_sync}
        asccube.release_buf m, %180, 0 {cannir.buffer.generated_sync}
        %191 = arith.cmpi eq, %182, %c0_i64 : i64
        %192 = arith.select %191, %177, %178 : i64
        %193 = arith.select %191, %c4_i64, %c5_i64 : i64
        %194 = arith.addi %182, %c1_i64 overflow<nsw> : i64
        %195 = arith.remui %194, %c2_i64 : i64
        %196 = ascscalar.bind_local_ptr %192 : memref<1xbf16, 3>
        asccube.get_buf mte1, %193, 0 {cannir.buffer.generated_sync}
        asccube.get_buf mte1, %104, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0a %196[%c0], %107[%c0], %c0_i32, %c8_i32, %c1_i32, %c8_i32, %c1_i32, %c1_i32 : memref<1xbf16, 3>, memref<1xbf16, 2>
        asccube.release_buf mte1, %104, 0 {cannir.buffer.generated_sync}
        asccube.release_buf mte1, %193, 0 {cannir.buffer.generated_sync}
        %197 = arith.select %191, %184, %185 : i64
        %198 = arith.select %191, %c6_i64, %c7_i64 : i64
        %199 = ascscalar.bind_local_ptr %197 : memref<1xbf16, 4>
        asccube.get_buf mte1, %198, 0 {cannir.buffer.generated_sync}
        asccube.get_buf mte1, %144, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0b %199[%c0], %145[%c0], %c0_i32, %c8_i32, %c4_i32, %c8_i32, %c4_i32, %c4_i32 : memref<1xbf16, 4>, memref<1xbf16, 2>
        asccube.release_buf mte1, %144, 0 {cannir.buffer.generated_sync}
        asccube.release_buf mte1, %198, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %193, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %198, 0 {cannir.buffer.generated_sync}
        asccube.mmad %190[%c0], %196, %199, %c8_i32, %c128_i32, %c64_i32, %c0_i32 {c_matrix_init_val = false} : memref<1xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
        asccube.release_buf m, %198, 0 {cannir.buffer.generated_sync}
        asccube.release_buf m, %193, 0 {cannir.buffer.generated_sync}
        asccube.release_buf m, %81, 0 {cannir.buffer.generated_sync}
        scf.yield %106, %195 : i64, i64
      }
      %85 = ascscalar.bind_local_ptr %80 : memref<1xf32, 5>
      %86 = arith.muli %74, %c8_i64 : i64
      %87 = cannir.extract_physical_layout %arg15 : !cannir.tensor<<gm, bf16>, <<(8, -1)>, <(-1, 1)>>, <<(8, -1)>, <(-1, 1)>>, identity>
      %88 = cannir.extract_stride %87 : <<(8, -1)>, <(-1, 1)>> -> <(-1, 1)>
      %89 = cannir.extract_dim %88[0] : !cannir.stride<(-1, 1)>
      %90 = arith.muli %86, %89 : i64
      %91 = arith.index_cast %90 : i64 to index
      %92 = arith.muli %76, %c64_i64 : i64
      %93 = arith.index_cast %92 : i64 to index
      %94 = arith.addi %91, %93 : index
      %95 = cannir.extract_layout %arg15 : !cannir.tensor<<gm, bf16>, <<(8, -1)>, <(-1, 1)>>, <<(8, -1)>, <(-1, 1)>>, identity>
      %96 = cannir.extract_stride %95 : <<(8, -1)>, <(-1, 1)>> -> <(-1, 1)>
      %97 = cannir.extract_dim %96[0] : !cannir.stride<(-1, 1)>
      asccube.get_buf fixpipe, %81, 0 {cannir.buffer.generated_sync}
      %98 = arith.trunci %97 : i64 to i32
      asccube.copy_l0c2gm %alloc_6[%94], %85[%c0], %c64_i32, %c8_i32, %98, %c16_i32, %c0_i32 l2_cache_ctl %c0_i32 {nz2nd_en = true, relu_en = false} : memref<1xbf16>, memref<1xf32, 5>
      asccube.release_buf fixpipe, %81, 0 {cannir.buffer.generated_sync}
      scf.yield %84#0, %84#1, %83 : i64, i64, i64
    }
    %45 = arith.muli %14, %15 : i64
    asccube.sync_all
    asccube.sync_block_wait s, %c13_i64
    asccube.sync_block_arrive fixpipe, %c14_i64 {mode = 0 : i32}
    asccube.sync_block_wait s, %c14_i64 {mode = 0 : i32}
    asccube.sync_block_arrive mte3, %c15_i64
    %46 = arith.remsi %20, %c96_i64 : i64
    %47 = arith.cmpi ne, %46, %c0_i64 : i64
    %48 = arith.cmpi slt, %46, %c0_i64 : i64
    %49 = arith.andi %47, %48 : i1
    %50 = arith.divsi %20, %c96_i64 : i64
    %51 = arith.extui %49 : i1 to i64
    %52 = arith.subi %50, %51 : i64
    %53:2 = scf.for %arg23 = %0 to %52 step %1 iter_args(%arg24 = %c0_i64, %arg25 = %c0_i64) -> (i64, i64)  : i64 {
      %66 = arith.muli %arg23, %c96_i64 : i64
      %67 = arith.cmpi eq, %arg25, %c0_i64 : i64
      %68 = ascscalar.memref_address %alloc_32 : memref<1536xf32, 5>
      %69 = ascscalar.memref_address %alloc_31 : memref<1536xf32, 5>
      %70 = arith.select %67, %68, %69 : i64
      %71 = arith.select %67, %c8_i64, %c9_i64 : i64
      %72 = arith.addi %arg25, %c1_i64 overflow<nsw> : i64
      %73 = arith.remui %72, %c2_i64 : i64
      %74 = arith.remsi %8, %c128_i64 : i64
      %75 = arith.cmpi ne, %74, %c0_i64 : i64
      %76 = arith.cmpi slt, %74, %c0_i64 : i64
      %77 = arith.andi %75, %76 : i1
      %78 = arith.divsi %8, %c128_i64 : i64
      %79 = arith.extui %77 : i1 to i64
      %80 = arith.subi %78, %79 : i64
      %81 = scf.for %arg26 = %c0_i64 to %80 step %c1_i64 iter_args(%arg27 = %arg24) -> (i64)  : i64 {
        %96 = arith.cmpi eq, %arg27, %c0_i64 : i64
        %97 = ascscalar.memref_address %alloc_30 : memref<2048xbf16, 2>
        %98 = ascscalar.memref_address %alloc_29 : memref<2048xbf16, 2>
        %99 = arith.select %96, %97, %98 : i64
        %100 = arith.cmpi ne, %arg27, %c0_i64 : i64
        %101 = arith.extui %100 : i1 to i64
        %102 = arith.addi %arg27, %c1_i64 overflow<nsw> : i64
        %103 = arith.remui %102, %c2_i64 : i64
        %104 = ascscalar.bind_local_ptr %99 : memref<1xbf16, 2>
        %105 = arith.muli %arg26, %c128_i64 : i64
        %106 = cannir.extract_layout %arg17 : !cannir.tensor<<gm, bf16>, <<(8, -1)>, <(-1, 1)>>, <<(8, -1)>, <(-1, 1)>>, identity>
        %107 = cannir.extract_shape %106 : <<(8, -1)>, <(-1, 1)>> -> <(8, -1)>
        %108 = cannir.extract_dim %107[1] : !cannir.shape<(8, -1)>
        %109 = arith.divsi %108, %c128_i64 : i64
        %110 = arith.minsi %arg26, %109 : i64
        %111 = arith.muli %110, %c128_i64 : i64
        %112 = arith.subi %108, %111 : i64
        %113 = arith.minsi %112, %c128_i64 : i64
        %114 = arith.cmpi sle, %arg26, %109 : i64
        %115 = arith.select %114, %113, %c0_i64 : i64
        %116 = cannir.extract_stride %106 : <<(8, -1)>, <(-1, 1)>> -> <(-1, 1)>
        %117 = cannir.extract_dim %116[0] : !cannir.stride<(-1, 1)>
        %118 = arith.index_cast %117 : i64 to index
        %119 = arith.muli %118, %c2 : index
        %120 = arith.index_cast %119 : index to i64
        asccube.get_buf mte2, %101, 0 {cannir.buffer.generated_sync}
        %121 = arith.cmpi ugt, %115, %c0_i64 : i64
        scf.if %121 {
          %176 = arith.index_cast %105 : i64 to index
          %177 = arith.trunci %115 : i64 to i32
          asccube.copy_gm2l1 %104[%c0], %alloc_8[%176], %c8_i32, %177, %120, %c1_i32, %c0_i64, %c0_i64, %c16_i32 l2_cache_ctl %c1_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
        }
        asccube.release_buf mte2, %101, 0 {cannir.buffer.generated_sync}
        %122 = ascscalar.memref_address %alloc_28 : memref<2048xbf16, 3>
        %123 = ascscalar.memref_address %alloc_27 : memref<2048xbf16, 3>
        %124 = arith.select %96, %122, %123 : i64
        %125 = arith.select %96, %c4_i64, %c5_i64 : i64
        %126 = ascscalar.bind_local_ptr %124 : memref<1xbf16, 3>
        asccube.get_buf mte1, %125, 0 {cannir.buffer.generated_sync}
        asccube.get_buf mte1, %101, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0a %126[%c0], %104[%c0], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c1_i32, %c1_i32 : memref<1xbf16, 3>, memref<1xbf16, 2>
        asccube.release_buf mte1, %101, 0 {cannir.buffer.generated_sync}
        asccube.release_buf mte1, %125, 0 {cannir.buffer.generated_sync}
        %127 = ascscalar.memref_address %alloc_26 : memref<12288xbf16, 2>
        %128 = ascscalar.memref_address %alloc_25 : memref<12288xbf16, 2>
        %129 = arith.select %96, %127, %128 : i64
        %130 = arith.select %96, %c2_i64, %c3_i64 : i64
        %131 = arith.remsi %66, %c96_i64 : i64
        %132 = arith.cmpi ne, %131, %c0_i64 : i64
        %133 = arith.cmpi slt, %131, %c0_i64 : i64
        %134 = arith.andi %132, %133 : i1
        %135 = arith.divsi %66, %c96_i64 : i64
        %136 = arith.extui %134 : i1 to i64
        %137 = arith.subi %135, %136 : i64
        %138 = ascscalar.bind_local_ptr %129 : memref<1xbf16, 2>
        %139 = arith.muli %137, %c6_i64 : i64
        %140 = cannir.extract_physical_layout %arg2 : !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
        %141 = cannir.extract_stride %140 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
        %142 = cannir.extract_dim %141[0] : !cannir.stride<(-1, 1)>
        %143 = arith.muli %139, %142 : i64
        %144 = arith.muli %arg26, %c8_i64 : i64
        %145 = arith.addi %143, %144 : i64
        %146 = arith.muli %145, %c16_i64 : i64
        %147 = arith.minsi %137, %50 : i64
        %148 = arith.muli %147, %c96_i64 : i64
        %149 = arith.subi %20, %148 : i64
        %150 = arith.minsi %149, %c96_i64 : i64
        %151 = arith.cmpi sle, %137, %50 : i64
        %152 = arith.select %151, %150, %c0_i64 : i64
        %153 = cannir.extract_dim %19[1] : !cannir.shape<(-1, -1)>
        %154 = arith.divsi %153, %c128_i64 : i64
        %155 = arith.minsi %arg26, %154 : i64
        %156 = arith.muli %155, %c128_i64 : i64
        %157 = arith.subi %153, %156 : i64
        %158 = arith.minsi %157, %c128_i64 : i64
        %159 = arith.cmpi sle, %arg26, %154 : i64
        %160 = arith.select %159, %158, %c0_i64 : i64
        %161 = cannir.extract_stride %18 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
        %162 = cannir.extract_dim %161[0] : !cannir.stride<(-1, 1)>
        %163 = arith.index_cast %162 : i64 to index
        %164 = arith.muli %163, %c2 : index
        %165 = arith.index_cast %164 : index to i64
        asccube.get_buf mte2, %130, 0 {cannir.buffer.generated_sync}
        %166 = arith.cmpi ugt, %152, %c0_i64 : i64
        %167 = arith.cmpi ugt, %160, %c0_i64 : i64
        %168 = arith.andi %166, %167 : i1
        scf.if %168 {
          %176 = arith.index_cast %146 : i64 to index
          %177 = arith.trunci %152 : i64 to i32
          %178 = arith.trunci %160 : i64 to i32
          asccube.copy_gm2l1 %138[%c0], %alloc_0[%176], %177, %178, %165, %c1_i32, %c0_i64, %c0_i64, %c96_i32 l2_cache_ctl %c0_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
        }
        asccube.release_buf mte2, %130, 0 {cannir.buffer.generated_sync}
        %169 = ascscalar.memref_address %alloc_24 : memref<12288xbf16, 4>
        %170 = ascscalar.memref_address %alloc_23 : memref<12288xbf16, 4>
        %171 = arith.select %96, %169, %170 : i64
        %172 = arith.select %96, %c6_i64, %c7_i64 : i64
        %173 = ascscalar.bind_local_ptr %171 : memref<1xbf16, 4>
        asccube.get_buf mte1, %172, 0 {cannir.buffer.generated_sync}
        asccube.get_buf mte1, %130, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0b %173[%c0], %138[%c0], %c0_i32, %c0_i32, %c6_i32, %c8_i32, %c6_i32, %c6_i32 : memref<1xbf16, 4>, memref<1xbf16, 2>
        asccube.release_buf mte1, %130, 0 {cannir.buffer.generated_sync}
        asccube.release_buf mte1, %172, 0 {cannir.buffer.generated_sync}
        %174 = arith.cmpi eq, %arg26, %c0_i64 : i64
        %175 = ascscalar.bind_local_ptr %70 : memref<1xf32, 5>
        asccube.get_buf m, %71, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %125, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %172, 0 {cannir.buffer.generated_sync}
        scf.if %174 {
          asccube.mmad %175[%c0], %126, %173, %c8_i32, %c128_i32, %c96_i32, %c0_i32 {c_matrix_init_val = true} : memref<1xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
        } else {
          asccube.mmad %175[%c0], %126, %173, %c8_i32, %c128_i32, %c96_i32, %c0_i32 {c_matrix_init_val = false} : memref<1xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
        }
        asccube.release_buf m, %172, 0 {cannir.buffer.generated_sync}
        asccube.release_buf m, %125, 0 {cannir.buffer.generated_sync}
        asccube.release_buf m, %71, 0 {cannir.buffer.generated_sync}
        scf.yield %103 : i64
      }
      %82 = arith.remsi %66, %c96_i64 : i64
      %83 = arith.cmpi ne, %82, %c0_i64 : i64
      %84 = arith.cmpi slt, %82, %c0_i64 : i64
      %85 = arith.andi %83, %84 : i1
      %86 = arith.divsi %66, %c96_i64 : i64
      %87 = arith.extui %85 : i1 to i64
      %88 = arith.subi %86, %87 : i64
      %89 = ascscalar.bind_local_ptr %70 : memref<1xf32, 5>
      %90 = arith.muli %88, %c96_i64 : i64
      %91 = cannir.extract_layout %arg16 : !cannir.tensor<<gm, bf16>, <<(8, -1)>, <(-1, 1)>>, <<(8, -1)>, <(-1, 1)>>, identity>
      %92 = cannir.extract_stride %91 : <<(8, -1)>, <(-1, 1)>> -> <(-1, 1)>
      %93 = cannir.extract_dim %92[0] : !cannir.stride<(-1, 1)>
      asccube.get_buf fixpipe, %71, 0 {cannir.buffer.generated_sync}
      %94 = arith.trunci %93 : i64 to i32
      %95 = arith.index_cast %90 : i64 to index
      asccube.copy_l0c2gm %alloc_7[%95], %89[%c0], %c96_i32, %c8_i32, %94, %c16_i32, %c0_i32 l2_cache_ctl %c0_i32 {nz2nd_en = true, relu_en = false} : memref<1xbf16>, memref<1xf32, 5>
      asccube.release_buf fixpipe, %71, 0 {cannir.buffer.generated_sync}
      scf.yield %81, %73 : i64, i64
    }
    %54 = arith.muli %52, %c96_i64 : i64
    %55 = arith.subi %20, %54 : i64
    %56 = arith.cmpi eq, %55, %c64_i64 : i64
    %57:2 = scf.if %56 -> (i64, i64) {
      %66 = arith.remsi %52, %1 : i64
      %67 = arith.cmpi ne, %66, %c0_i64 : i64
      %68 = arith.cmpi slt, %66, %c0_i64 : i64
      %69 = arith.cmpi slt, %1, %c0_i64 : i64
      %70 = arith.xori %68, %69 : i1
      %71 = arith.andi %67, %70 : i1
      %72 = arith.select %71, %1, %c0_i64 : i64
      %73 = arith.addi %66, %72 : i64
      %74 = arith.cmpi eq, %73, %0 : i64
      %75:2 = scf.if %74 -> (i64, i64) {
        %76 = arith.cmpi eq, %53#1, %c0_i64 : i64
        %77 = ascscalar.memref_address %alloc_32 : memref<1536xf32, 5>
        %78 = ascscalar.memref_address %alloc_31 : memref<1536xf32, 5>
        %79 = arith.select %76, %77, %78 : i64
        %80 = arith.select %76, %c8_i64, %c9_i64 : i64
        %81 = arith.addi %53#1, %c1_i64 overflow<nsw> : i64
        %82 = arith.remui %81, %c2_i64 : i64
        %83 = arith.remsi %8, %c128_i64 : i64
        %84 = arith.cmpi ne, %83, %c0_i64 : i64
        %85 = arith.cmpi slt, %83, %c0_i64 : i64
        %86 = arith.andi %84, %85 : i1
        %87 = arith.divsi %8, %c128_i64 : i64
        %88 = arith.extui %86 : i1 to i64
        %89 = arith.subi %87, %88 : i64
        %90 = scf.for %arg23 = %c0_i64 to %89 step %c1_i64 iter_args(%arg24 = %53#0) -> (i64)  : i64 {
          %105 = arith.cmpi eq, %arg24, %c0_i64 : i64
          %106 = ascscalar.memref_address %alloc_30 : memref<2048xbf16, 2>
          %107 = ascscalar.memref_address %alloc_29 : memref<2048xbf16, 2>
          %108 = arith.select %105, %106, %107 : i64
          %109 = arith.cmpi ne, %arg24, %c0_i64 : i64
          %110 = arith.extui %109 : i1 to i64
          %111 = arith.addi %arg24, %c1_i64 overflow<nsw> : i64
          %112 = arith.remui %111, %c2_i64 : i64
          %113 = ascscalar.bind_local_ptr %108 : memref<1xbf16, 2>
          %114 = arith.muli %arg23, %c128_i64 : i64
          %115 = cannir.extract_layout %arg17 : !cannir.tensor<<gm, bf16>, <<(8, -1)>, <(-1, 1)>>, <<(8, -1)>, <(-1, 1)>>, identity>
          %116 = cannir.extract_shape %115 : <<(8, -1)>, <(-1, 1)>> -> <(8, -1)>
          %117 = cannir.extract_dim %116[1] : !cannir.shape<(8, -1)>
          %118 = arith.divsi %117, %c128_i64 : i64
          %119 = arith.minsi %arg23, %118 : i64
          %120 = arith.muli %119, %c128_i64 : i64
          %121 = arith.subi %117, %120 : i64
          %122 = arith.minsi %121, %c128_i64 : i64
          %123 = arith.cmpi sle, %arg23, %118 : i64
          %124 = arith.select %123, %122, %c0_i64 : i64
          %125 = cannir.extract_stride %115 : <<(8, -1)>, <(-1, 1)>> -> <(-1, 1)>
          %126 = cannir.extract_dim %125[0] : !cannir.stride<(-1, 1)>
          %127 = arith.index_cast %126 : i64 to index
          %128 = arith.muli %127, %c2 : index
          %129 = arith.index_cast %128 : index to i64
          asccube.get_buf mte2, %110, 0 {cannir.buffer.generated_sync}
          %130 = arith.cmpi ugt, %124, %c0_i64 : i64
          scf.if %130 {
            %186 = arith.index_cast %114 : i64 to index
            %187 = arith.trunci %124 : i64 to i32
            asccube.copy_gm2l1 %113[%c0], %alloc_8[%186], %c8_i32, %187, %129, %c1_i32, %c0_i64, %c0_i64, %c16_i32 l2_cache_ctl %c1_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
          }
          asccube.release_buf mte2, %110, 0 {cannir.buffer.generated_sync}
          %131 = ascscalar.memref_address %alloc_28 : memref<2048xbf16, 3>
          %132 = ascscalar.memref_address %alloc_27 : memref<2048xbf16, 3>
          %133 = arith.select %105, %131, %132 : i64
          %134 = arith.select %105, %c4_i64, %c5_i64 : i64
          %135 = ascscalar.bind_local_ptr %133 : memref<1xbf16, 3>
          asccube.get_buf mte1, %134, 0 {cannir.buffer.generated_sync}
          asccube.get_buf mte1, %110, 0 {cannir.buffer.generated_sync}
          asccube.copy_l12l0a %135[%c0], %113[%c0], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c1_i32, %c1_i32 : memref<1xbf16, 3>, memref<1xbf16, 2>
          asccube.release_buf mte1, %110, 0 {cannir.buffer.generated_sync}
          asccube.release_buf mte1, %134, 0 {cannir.buffer.generated_sync}
          %136 = ascscalar.memref_address %alloc_26 : memref<12288xbf16, 2>
          %137 = ascscalar.memref_address %alloc_25 : memref<12288xbf16, 2>
          %138 = arith.select %105, %136, %137 : i64
          %139 = arith.select %105, %c2_i64, %c3_i64 : i64
          %140 = arith.remsi %54, %c64_i64 : i64
          %141 = arith.cmpi ne, %140, %c0_i64 : i64
          %142 = arith.cmpi slt, %140, %c0_i64 : i64
          %143 = arith.andi %141, %142 : i1
          %144 = arith.divsi %54, %c64_i64 : i64
          %145 = arith.extui %143 : i1 to i64
          %146 = arith.subi %144, %145 : i64
          %147 = ascscalar.bind_local_ptr %138 : memref<1xbf16, 2>
          %148 = arith.muli %146, %c4_i64 : i64
          %149 = cannir.extract_physical_layout %arg2 : !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
          %150 = cannir.extract_stride %149 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
          %151 = cannir.extract_dim %150[0] : !cannir.stride<(-1, 1)>
          %152 = arith.muli %148, %151 : i64
          %153 = arith.muli %arg23, %c8_i64 : i64
          %154 = arith.addi %152, %153 : i64
          %155 = arith.muli %154, %c16_i64 : i64
          %156 = arith.divsi %20, %c64_i64 : i64
          %157 = arith.minsi %146, %156 : i64
          %158 = arith.muli %157, %c64_i64 : i64
          %159 = arith.subi %20, %158 : i64
          %160 = arith.minsi %159, %c64_i64 : i64
          %161 = arith.cmpi sle, %146, %156 : i64
          %162 = arith.select %161, %160, %c0_i64 : i64
          %163 = cannir.extract_dim %19[1] : !cannir.shape<(-1, -1)>
          %164 = arith.divsi %163, %c128_i64 : i64
          %165 = arith.minsi %arg23, %164 : i64
          %166 = arith.muli %165, %c128_i64 : i64
          %167 = arith.subi %163, %166 : i64
          %168 = arith.minsi %167, %c128_i64 : i64
          %169 = arith.cmpi sle, %arg23, %164 : i64
          %170 = arith.select %169, %168, %c0_i64 : i64
          %171 = cannir.extract_stride %18 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
          %172 = cannir.extract_dim %171[0] : !cannir.stride<(-1, 1)>
          %173 = arith.index_cast %172 : i64 to index
          %174 = arith.muli %173, %c2 : index
          %175 = arith.index_cast %174 : index to i64
          asccube.get_buf mte2, %139, 0 {cannir.buffer.generated_sync}
          %176 = arith.cmpi ugt, %162, %c0_i64 : i64
          %177 = arith.cmpi ugt, %170, %c0_i64 : i64
          %178 = arith.andi %176, %177 : i1
          scf.if %178 {
            %186 = arith.index_cast %155 : i64 to index
            %187 = arith.trunci %162 : i64 to i32
            %188 = arith.trunci %170 : i64 to i32
            asccube.copy_gm2l1 %147[%c0], %alloc_0[%186], %187, %188, %175, %c1_i32, %c0_i64, %c0_i64, %c96_i32 l2_cache_ctl %c0_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
          }
          asccube.release_buf mte2, %139, 0 {cannir.buffer.generated_sync}
          %179 = ascscalar.memref_address %alloc_24 : memref<12288xbf16, 4>
          %180 = ascscalar.memref_address %alloc_23 : memref<12288xbf16, 4>
          %181 = arith.select %105, %179, %180 : i64
          %182 = arith.select %105, %c6_i64, %c7_i64 : i64
          %183 = ascscalar.bind_local_ptr %181 : memref<1xbf16, 4>
          asccube.get_buf mte1, %182, 0 {cannir.buffer.generated_sync}
          asccube.get_buf mte1, %139, 0 {cannir.buffer.generated_sync}
          asccube.copy_l12l0b %183[%c0], %147[%c0], %c0_i32, %c0_i32, %c6_i32, %c8_i32, %c6_i32, %c6_i32 : memref<1xbf16, 4>, memref<1xbf16, 2>
          asccube.release_buf mte1, %139, 0 {cannir.buffer.generated_sync}
          asccube.release_buf mte1, %182, 0 {cannir.buffer.generated_sync}
          %184 = arith.cmpi eq, %arg23, %c0_i64 : i64
          %185 = ascscalar.bind_local_ptr %79 : memref<1xf32, 5>
          asccube.get_buf m, %80, 0 {cannir.buffer.generated_sync}
          asccube.get_buf m, %134, 0 {cannir.buffer.generated_sync}
          asccube.get_buf m, %182, 0 {cannir.buffer.generated_sync}
          scf.if %184 {
            asccube.mmad %185[%c0], %135, %183, %c8_i32, %c128_i32, %c96_i32, %c0_i32 {c_matrix_init_val = true} : memref<1xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
          } else {
            asccube.mmad %185[%c0], %135, %183, %c8_i32, %c128_i32, %c96_i32, %c0_i32 {c_matrix_init_val = false} : memref<1xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
          }
          asccube.release_buf m, %182, 0 {cannir.buffer.generated_sync}
          asccube.release_buf m, %134, 0 {cannir.buffer.generated_sync}
          asccube.release_buf m, %80, 0 {cannir.buffer.generated_sync}
          scf.yield %112 : i64
        }
        %91 = arith.remsi %54, %c96_i64 : i64
        %92 = arith.cmpi ne, %91, %c0_i64 : i64
        %93 = arith.cmpi slt, %91, %c0_i64 : i64
        %94 = arith.andi %92, %93 : i1
        %95 = arith.divsi %54, %c96_i64 : i64
        %96 = arith.extui %94 : i1 to i64
        %97 = arith.subi %95, %96 : i64
        %98 = ascscalar.bind_local_ptr %79 : memref<1xf32, 5>
        %99 = arith.muli %97, %c96_i64 : i64
        %100 = cannir.extract_layout %arg16 : !cannir.tensor<<gm, bf16>, <<(8, -1)>, <(-1, 1)>>, <<(8, -1)>, <(-1, 1)>>, identity>
        %101 = cannir.extract_stride %100 : <<(8, -1)>, <(-1, 1)>> -> <(-1, 1)>
        %102 = cannir.extract_dim %101[0] : !cannir.stride<(-1, 1)>
        asccube.get_buf fixpipe, %80, 0 {cannir.buffer.generated_sync}
        %103 = arith.trunci %102 : i64 to i32
        %104 = arith.index_cast %99 : i64 to index
        asccube.copy_l0c2gm %alloc_7[%104], %98[%c0], %c96_i32, %c8_i32, %103, %c16_i32, %c0_i32 l2_cache_ctl %c0_i32 {nz2nd_en = true, relu_en = false} : memref<1xbf16>, memref<1xf32, 5>
        asccube.release_buf fixpipe, %80, 0 {cannir.buffer.generated_sync}
        scf.yield %90, %82 : i64, i64
      } else {
        scf.yield %53#0, %53#1 : i64, i64
      }
      scf.yield %75#0, %75#1 : i64, i64
    } else {
      scf.yield %53#0, %53#1 : i64, i64
    }
    %58 = arith.cmpi eq, %55, %c32_i64 : i64
    scf.if %58 {
      %66 = arith.remsi %52, %1 : i64
      %67 = arith.cmpi ne, %66, %c0_i64 : i64
      %68 = arith.cmpi slt, %66, %c0_i64 : i64
      %69 = arith.cmpi slt, %1, %c0_i64 : i64
      %70 = arith.xori %68, %69 : i1
      %71 = arith.andi %67, %70 : i1
      %72 = arith.select %71, %1, %c0_i64 : i64
      %73 = arith.addi %66, %72 : i64
      %74 = arith.cmpi eq, %73, %0 : i64
      scf.if %74 {
        %75 = arith.cmpi eq, %57#1, %c0_i64 : i64
        %76 = ascscalar.memref_address %alloc_32 : memref<1536xf32, 5>
        %77 = ascscalar.memref_address %alloc_31 : memref<1536xf32, 5>
        %78 = arith.select %75, %76, %77 : i64
        %79 = arith.select %75, %c8_i64, %c9_i64 : i64
        %80 = arith.remsi %8, %c128_i64 : i64
        %81 = arith.cmpi ne, %80, %c0_i64 : i64
        %82 = arith.cmpi slt, %80, %c0_i64 : i64
        %83 = arith.andi %81, %82 : i1
        %84 = arith.divsi %8, %c128_i64 : i64
        %85 = arith.extui %83 : i1 to i64
        %86 = arith.subi %84, %85 : i64
        %87 = scf.for %arg23 = %c0_i64 to %86 step %c1_i64 iter_args(%arg24 = %57#0) -> (i64)  : i64 {
          %102 = arith.cmpi eq, %arg24, %c0_i64 : i64
          %103 = ascscalar.memref_address %alloc_30 : memref<2048xbf16, 2>
          %104 = ascscalar.memref_address %alloc_29 : memref<2048xbf16, 2>
          %105 = arith.select %102, %103, %104 : i64
          %106 = arith.cmpi ne, %arg24, %c0_i64 : i64
          %107 = arith.extui %106 : i1 to i64
          %108 = arith.addi %arg24, %c1_i64 overflow<nsw> : i64
          %109 = arith.remui %108, %c2_i64 : i64
          %110 = ascscalar.bind_local_ptr %105 : memref<1xbf16, 2>
          %111 = arith.muli %arg23, %c128_i64 : i64
          %112 = cannir.extract_layout %arg17 : !cannir.tensor<<gm, bf16>, <<(8, -1)>, <(-1, 1)>>, <<(8, -1)>, <(-1, 1)>>, identity>
          %113 = cannir.extract_shape %112 : <<(8, -1)>, <(-1, 1)>> -> <(8, -1)>
          %114 = cannir.extract_dim %113[1] : !cannir.shape<(8, -1)>
          %115 = arith.divsi %114, %c128_i64 : i64
          %116 = arith.minsi %arg23, %115 : i64
          %117 = arith.muli %116, %c128_i64 : i64
          %118 = arith.subi %114, %117 : i64
          %119 = arith.minsi %118, %c128_i64 : i64
          %120 = arith.cmpi sle, %arg23, %115 : i64
          %121 = arith.select %120, %119, %c0_i64 : i64
          %122 = cannir.extract_stride %112 : <<(8, -1)>, <(-1, 1)>> -> <(-1, 1)>
          %123 = cannir.extract_dim %122[0] : !cannir.stride<(-1, 1)>
          %124 = arith.index_cast %123 : i64 to index
          %125 = arith.muli %124, %c2 : index
          %126 = arith.index_cast %125 : index to i64
          asccube.get_buf mte2, %107, 0 {cannir.buffer.generated_sync}
          %127 = arith.cmpi ugt, %121, %c0_i64 : i64
          scf.if %127 {
            %183 = arith.index_cast %111 : i64 to index
            %184 = arith.trunci %121 : i64 to i32
            asccube.copy_gm2l1 %110[%c0], %alloc_8[%183], %c8_i32, %184, %126, %c1_i32, %c0_i64, %c0_i64, %c16_i32 l2_cache_ctl %c1_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
          }
          asccube.release_buf mte2, %107, 0 {cannir.buffer.generated_sync}
          %128 = ascscalar.memref_address %alloc_28 : memref<2048xbf16, 3>
          %129 = ascscalar.memref_address %alloc_27 : memref<2048xbf16, 3>
          %130 = arith.select %102, %128, %129 : i64
          %131 = arith.select %102, %c4_i64, %c5_i64 : i64
          %132 = ascscalar.bind_local_ptr %130 : memref<1xbf16, 3>
          asccube.get_buf mte1, %131, 0 {cannir.buffer.generated_sync}
          asccube.get_buf mte1, %107, 0 {cannir.buffer.generated_sync}
          asccube.copy_l12l0a %132[%c0], %110[%c0], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c1_i32, %c1_i32 : memref<1xbf16, 3>, memref<1xbf16, 2>
          asccube.release_buf mte1, %107, 0 {cannir.buffer.generated_sync}
          asccube.release_buf mte1, %131, 0 {cannir.buffer.generated_sync}
          %133 = ascscalar.memref_address %alloc_26 : memref<12288xbf16, 2>
          %134 = ascscalar.memref_address %alloc_25 : memref<12288xbf16, 2>
          %135 = arith.select %102, %133, %134 : i64
          %136 = arith.select %102, %c2_i64, %c3_i64 : i64
          %137 = arith.remsi %54, %c32_i64 : i64
          %138 = arith.cmpi ne, %137, %c0_i64 : i64
          %139 = arith.cmpi slt, %137, %c0_i64 : i64
          %140 = arith.andi %138, %139 : i1
          %141 = arith.divsi %54, %c32_i64 : i64
          %142 = arith.extui %140 : i1 to i64
          %143 = arith.subi %141, %142 : i64
          %144 = ascscalar.bind_local_ptr %135 : memref<1xbf16, 2>
          %145 = arith.muli %143, %c2_i64 : i64
          %146 = cannir.extract_physical_layout %arg2 : !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
          %147 = cannir.extract_stride %146 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
          %148 = cannir.extract_dim %147[0] : !cannir.stride<(-1, 1)>
          %149 = arith.muli %145, %148 : i64
          %150 = arith.muli %arg23, %c8_i64 : i64
          %151 = arith.addi %149, %150 : i64
          %152 = arith.muli %151, %c16_i64 : i64
          %153 = arith.divsi %20, %c32_i64 : i64
          %154 = arith.minsi %143, %153 : i64
          %155 = arith.muli %154, %c32_i64 : i64
          %156 = arith.subi %20, %155 : i64
          %157 = arith.minsi %156, %c32_i64 : i64
          %158 = arith.cmpi sle, %143, %153 : i64
          %159 = arith.select %158, %157, %c0_i64 : i64
          %160 = cannir.extract_dim %19[1] : !cannir.shape<(-1, -1)>
          %161 = arith.divsi %160, %c128_i64 : i64
          %162 = arith.minsi %arg23, %161 : i64
          %163 = arith.muli %162, %c128_i64 : i64
          %164 = arith.subi %160, %163 : i64
          %165 = arith.minsi %164, %c128_i64 : i64
          %166 = arith.cmpi sle, %arg23, %161 : i64
          %167 = arith.select %166, %165, %c0_i64 : i64
          %168 = cannir.extract_stride %18 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
          %169 = cannir.extract_dim %168[0] : !cannir.stride<(-1, 1)>
          %170 = arith.index_cast %169 : i64 to index
          %171 = arith.muli %170, %c2 : index
          %172 = arith.index_cast %171 : index to i64
          asccube.get_buf mte2, %136, 0 {cannir.buffer.generated_sync}
          %173 = arith.cmpi ugt, %159, %c0_i64 : i64
          %174 = arith.cmpi ugt, %167, %c0_i64 : i64
          %175 = arith.andi %173, %174 : i1
          scf.if %175 {
            %183 = arith.index_cast %152 : i64 to index
            %184 = arith.trunci %159 : i64 to i32
            %185 = arith.trunci %167 : i64 to i32
            asccube.copy_gm2l1 %144[%c0], %alloc_0[%183], %184, %185, %172, %c1_i32, %c0_i64, %c0_i64, %c96_i32 l2_cache_ctl %c0_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
          }
          asccube.release_buf mte2, %136, 0 {cannir.buffer.generated_sync}
          %176 = ascscalar.memref_address %alloc_24 : memref<12288xbf16, 4>
          %177 = ascscalar.memref_address %alloc_23 : memref<12288xbf16, 4>
          %178 = arith.select %102, %176, %177 : i64
          %179 = arith.select %102, %c6_i64, %c7_i64 : i64
          %180 = ascscalar.bind_local_ptr %178 : memref<1xbf16, 4>
          asccube.get_buf mte1, %179, 0 {cannir.buffer.generated_sync}
          asccube.get_buf mte1, %136, 0 {cannir.buffer.generated_sync}
          asccube.copy_l12l0b %180[%c0], %144[%c0], %c0_i32, %c0_i32, %c6_i32, %c8_i32, %c6_i32, %c6_i32 : memref<1xbf16, 4>, memref<1xbf16, 2>
          asccube.release_buf mte1, %136, 0 {cannir.buffer.generated_sync}
          asccube.release_buf mte1, %179, 0 {cannir.buffer.generated_sync}
          %181 = arith.cmpi eq, %arg23, %c0_i64 : i64
          %182 = ascscalar.bind_local_ptr %78 : memref<1xf32, 5>
          asccube.get_buf m, %79, 0 {cannir.buffer.generated_sync}
          asccube.get_buf m, %131, 0 {cannir.buffer.generated_sync}
          asccube.get_buf m, %179, 0 {cannir.buffer.generated_sync}
          scf.if %181 {
            asccube.mmad %182[%c0], %132, %180, %c8_i32, %c128_i32, %c96_i32, %c0_i32 {c_matrix_init_val = true} : memref<1xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
          } else {
            asccube.mmad %182[%c0], %132, %180, %c8_i32, %c128_i32, %c96_i32, %c0_i32 {c_matrix_init_val = false} : memref<1xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
          }
          asccube.release_buf m, %179, 0 {cannir.buffer.generated_sync}
          asccube.release_buf m, %131, 0 {cannir.buffer.generated_sync}
          asccube.release_buf m, %79, 0 {cannir.buffer.generated_sync}
          scf.yield %109 : i64
        }
        %88 = arith.remsi %54, %c96_i64 : i64
        %89 = arith.cmpi ne, %88, %c0_i64 : i64
        %90 = arith.cmpi slt, %88, %c0_i64 : i64
        %91 = arith.andi %89, %90 : i1
        %92 = arith.divsi %54, %c96_i64 : i64
        %93 = arith.extui %91 : i1 to i64
        %94 = arith.subi %92, %93 : i64
        %95 = ascscalar.bind_local_ptr %78 : memref<1xf32, 5>
        %96 = arith.muli %94, %c96_i64 : i64
        %97 = cannir.extract_layout %arg16 : !cannir.tensor<<gm, bf16>, <<(8, -1)>, <(-1, 1)>>, <<(8, -1)>, <(-1, 1)>>, identity>
        %98 = cannir.extract_stride %97 : <<(8, -1)>, <(-1, 1)>> -> <(-1, 1)>
        %99 = cannir.extract_dim %98[0] : !cannir.stride<(-1, 1)>
        asccube.get_buf fixpipe, %79, 0 {cannir.buffer.generated_sync}
        %100 = arith.trunci %99 : i64 to i32
        %101 = arith.index_cast %96 : i64 to index
        asccube.copy_l0c2gm %alloc_7[%101], %95[%c0], %c96_i32, %c8_i32, %100, %c16_i32, %c0_i32 l2_cache_ctl %c0_i32 {nz2nd_en = true, relu_en = false} : memref<1xbf16>, memref<1xf32, 5>
        asccube.release_buf fixpipe, %79, 0 {cannir.buffer.generated_sync}
      }
    }
    asccube.sync_all
    asccube.sync_block_wait s, %c13_i64
    asccube.sync_block_arrive fixpipe, %c14_i64 {mode = 0 : i32}
    asccube.sync_block_wait s, %c14_i64 {mode = 0 : i32}
    asccube.sync_block_arrive mte3, %c15_i64
    %59 = cannir.extract_layout %arg11 : !cannir.tensor<<gm, bf16>, <<(8, -1, -1)>, <(-1, -1, 1)>>, <<(8, -1, -1)>, <(-1, -1, 1)>>, identity>
    %60 = cannir.extract_shape %59 : <<(8, -1, -1)>, <(-1, -1, 1)>> -> <(8, -1, -1)>
    %61 = cannir.extract_dim %60[1] : !cannir.shape<(8, -1, -1)>
    %62 = cannir.extract_dim %60[2] : !cannir.shape<(8, -1, -1)>
    %63 = arith.muli %61, %62 : i64
    %64 = arith.muli %14, %c4_i64 : i64
    %65:2 = scf.for %arg23 = %0 to %64 step %1 iter_args(%arg24 = %c0_i64, %arg25 = %c0_i64) -> (i64, i64)  : i64 {
      %66 = arith.remsi %arg23, %c4_i64 : i64
      %67 = arith.cmpi ne, %66, %c0_i64 : i64
      %68 = arith.cmpi slt, %66, %c0_i64 : i64
      %69 = arith.andi %67, %68 : i1
      %70 = arith.select %69, %c4_i64, %c0_i64 : i64
      %71 = arith.addi %66, %70 overflow<nsw> : i64
      %72 = arith.divsi %arg23, %c4_i64 : i64
      %73 = arith.extui %69 : i1 to i64
      %74 = arith.subi %72, %73 : i64
      %75 = arith.remsi %74, %14 : i64
      %76 = arith.cmpi ne, %75, %c0_i64 : i64
      %77 = arith.cmpi slt, %75, %c0_i64 : i64
      %78 = arith.cmpi slt, %14, %c0_i64 : i64
      %79 = arith.xori %77, %78 : i1
      %80 = arith.andi %76, %79 : i1
      %81 = arith.select %80, %14, %c0_i64 : i64
      %82 = arith.addi %75, %81 : i64
      %83 = arith.remsi %arg23, %64 : i64
      %84 = arith.cmpi ne, %83, %c0_i64 : i64
      %85 = arith.cmpi slt, %83, %c0_i64 : i64
      %86 = arith.cmpi slt, %64, %c0_i64 : i64
      %87 = arith.xori %85, %86 : i1
      %88 = arith.andi %84, %87 : i1
      %89 = arith.divsi %arg23, %64 : i64
      %90 = arith.extui %88 : i1 to i64
      %91 = arith.subi %89, %90 : i64
      %92 = arith.muli %91, %c8_i64 : i64
      %93 = arith.muli %82, %17 : i64
      %94 = arith.muli %82, %11 : i64
      %95 = arith.cmpi eq, %arg25, %c0_i64 : i64
      %96 = ascscalar.memref_address %alloc_22 : memref<2048xf32, 5>
      %97 = ascscalar.memref_address %alloc_21 : memref<2048xf32, 5>
      %98 = arith.select %95, %96, %97 : i64
      %99 = arith.select %95, %c6_i64, %c7_i64 : i64
      %100 = arith.addi %arg25, %c1_i64 overflow<nsw> : i64
      %101 = arith.remui %100, %c2_i64 : i64
      %102 = arith.remsi %15, %c64_i64 : i64
      %103 = arith.cmpi ne, %102, %c0_i64 : i64
      %104 = arith.cmpi slt, %102, %c0_i64 : i64
      %105 = arith.andi %103, %104 : i1
      %106 = arith.divsi %15, %c64_i64 : i64
      %107 = arith.extui %105 : i1 to i64
      %108 = arith.subi %106, %107 : i64
      %109 = scf.for %arg26 = %c0_i64 to %108 step %c1_i64 iter_args(%arg27 = %arg24) -> (i64)  : i64 {
        %118 = arith.cmpi eq, %arg27, %c0_i64 : i64
        %119 = ascscalar.memref_address %alloc_20 : memref<1024xbf16, 2>
        %120 = ascscalar.memref_address %alloc_19 : memref<1024xbf16, 2>
        %121 = arith.select %118, %119, %120 : i64
        %122 = arith.cmpi ne, %arg27, %c0_i64 : i64
        %123 = arith.extui %122 : i1 to i64
        %124 = arith.addi %arg27, %c1_i64 overflow<nsw> : i64
        %125 = arith.remui %124, %c2_i64 : i64
        %126 = arith.muli %arg26, %c64_i64 : i64
        %127 = arith.addi %93, %126 : i64
        %128 = arith.addi %arg26, %c1_i64 : i64
        %129 = arith.muli %128, %c64_i64 : i64
        %130 = arith.addi %93, %129 : i64
        %131 = arith.subi %130, %127 : i64
        %132 = ascscalar.bind_local_ptr %121 : memref<1xbf16, 2>
        %133 = cannir.extract_physical_layout %arg16 : !cannir.tensor<<gm, bf16>, <<(8, -1)>, <(-1, 1)>>, <<(8, -1)>, <(-1, 1)>>, identity>
        %134 = cannir.extract_stride %133 : <<(8, -1)>, <(-1, 1)>> -> <(-1, 1)>
        %135 = cannir.extract_dim %134[0] : !cannir.stride<(-1, 1)>
        %136 = arith.muli %92, %135 : i64
        %137 = arith.index_cast %136 : i64 to index
        %138 = arith.index_cast %127 : i64 to index
        %139 = arith.addi %137, %138 : index
        %140 = cannir.extract_layout %arg16 : !cannir.tensor<<gm, bf16>, <<(8, -1)>, <(-1, 1)>>, <<(8, -1)>, <(-1, 1)>>, identity>
        %141 = cannir.extract_stride %140 : <<(8, -1)>, <(-1, 1)>> -> <(-1, 1)>
        %142 = cannir.extract_dim %141[0] : !cannir.stride<(-1, 1)>
        %143 = arith.index_cast %142 : i64 to index
        %144 = arith.muli %143, %c2 : index
        %145 = arith.index_cast %144 : index to i64
        asccube.get_buf mte2, %123, 0 {cannir.buffer.generated_sync}
        %146 = arith.cmpi ugt, %131, %c0_i64 : i64
        scf.if %146 {
          %177 = arith.trunci %131 : i64 to i32
          asccube.copy_gm2l1 %132[%c0], %alloc_7[%139], %c8_i32, %177, %145, %c1_i32, %c0_i64, %c0_i64, %c16_i32 l2_cache_ctl %c1_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
        }
        asccube.release_buf mte2, %123, 0 {cannir.buffer.generated_sync}
        %147 = ascscalar.bitcast_memref %alloc_15 byte_offset = %c0_i64 : memref<1024xbf16, 3> -> memref<?xbf16, 3>
        asccube.get_buf mte1, %c4_i64, 0 {cannir.buffer.generated_sync}
        asccube.get_buf mte1, %123, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0a %147[%c0], %132[%c0], %c0_i32, %c0_i32, %c1_i32, %c4_i32, %c1_i32, %c1_i32 : memref<?xbf16, 3>, memref<1xbf16, 2>
        asccube.release_buf mte1, %123, 0 {cannir.buffer.generated_sync}
        asccube.release_buf mte1, %c4_i64, 0 {cannir.buffer.generated_sync}
        %148 = ascscalar.memref_address %alloc_18 : memref<8192xbf16, 2>
        %149 = ascscalar.memref_address %alloc_17 : memref<8192xbf16, 2>
        %150 = arith.select %118, %148, %149 : i64
        %151 = arith.select %118, %c2_i64, %c3_i64 : i64
        %152 = arith.muli %82, %15 : i64
        %153 = arith.addi %152, %126 : i64
        %154 = arith.remsi %153, %c64_i64 : i64
        %155 = arith.cmpi ne, %154, %c0_i64 : i64
        %156 = arith.cmpi slt, %154, %c0_i64 : i64
        %157 = arith.andi %155, %156 : i1
        %158 = arith.divsi %153, %c64_i64 : i64
        %159 = arith.extui %157 : i1 to i64
        %160 = arith.subi %158, %159 : i64
        %161 = ascscalar.bind_local_ptr %150 : memref<1xbf16, 2>
        %162 = arith.muli %160, %c2048_i64 : i64
        %163 = arith.muli %71, %c8_i64 overflow<nsw> : i64
        %164 = arith.addi %162, %163 : i64
        %165 = arith.muli %164, %c16_i64 : i64
        %166 = arith.divsi %45, %c64_i64 : i64
        %167 = arith.minsi %160, %166 : i64
        %168 = arith.muli %167, %c64_i64 : i64
        %169 = arith.subi %45, %168 : i64
        %170 = arith.minsi %169, %c64_i64 : i64
        %171 = arith.cmpi sle, %160, %166 : i64
        %172 = arith.select %171, %170, %c0_i64 : i64
        asccube.get_buf mte2, %151, 0 {cannir.buffer.generated_sync}
        %173 = arith.cmpi ugt, %172, %c0_i64 : i64
        scf.if %173 {
          %177 = arith.index_cast %165 : i64 to index
          %178 = arith.trunci %172 : i64 to i32
          asccube.copy_gm2l1 %161[%c0], %alloc_2[%177], %178, %c128_i32, %c1024_i64, %c1_i32, %c0_i64, %c0_i64, %c64_i32 l2_cache_ctl %c0_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
        }
        asccube.release_buf mte2, %151, 0 {cannir.buffer.generated_sync}
        %174 = ascscalar.bitcast_memref %alloc_16 byte_offset = %c0_i64 : memref<8192xbf16, 4> -> memref<?xbf16, 4>
        asccube.get_buf mte1, %c5_i64, 0 {cannir.buffer.generated_sync}
        asccube.get_buf mte1, %151, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0b %174[%c0], %161[%c0], %c0_i32, %c0_i32, %c4_i32, %c8_i32, %c4_i32, %c8_i32 {transpose = true} : memref<?xbf16, 4>, memref<1xbf16, 2>
        asccube.release_buf mte1, %151, 0 {cannir.buffer.generated_sync}
        asccube.release_buf mte1, %c5_i64, 0 {cannir.buffer.generated_sync}
        %175 = arith.cmpi eq, %arg26, %c0_i64 : i64
        %176 = ascscalar.bind_local_ptr %98 : memref<1xf32, 5>
        asccube.get_buf m, %99, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %c4_i64, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %c5_i64, 0 {cannir.buffer.generated_sync}
        scf.if %175 {
          asccube.mmad %176[%c0], %147, %174, %c8_i32, %c64_i32, %c128_i32, %c0_i32 {c_matrix_init_val = true} : memref<1xf32, 5>, memref<?xbf16, 3>, memref<?xbf16, 4>
        } else {
          asccube.mmad %176[%c0], %147, %174, %c8_i32, %c64_i32, %c128_i32, %c0_i32 {c_matrix_init_val = false} : memref<1xf32, 5>, memref<?xbf16, 3>, memref<?xbf16, 4>
        }
        asccube.release_buf m, %c5_i64, 0 {cannir.buffer.generated_sync}
        asccube.release_buf m, %c4_i64, 0 {cannir.buffer.generated_sync}
        asccube.release_buf m, %99, 0 {cannir.buffer.generated_sync}
        scf.yield %125 : i64
      }
      %110 = arith.muli %71, %c128_i64 overflow<nsw> : i64
      %111 = arith.addi %94, %110 : i64
      %112 = ascscalar.bind_local_ptr %98 : memref<1xf32, 5>
      %113 = arith.muli %92, %63 : i64
      %114 = arith.index_cast %113 : i64 to index
      %115 = arith.index_cast %111 : i64 to index
      %116 = arith.addi %114, %115 : index
      asccube.get_buf fixpipe, %99, 0 {cannir.buffer.generated_sync}
      %117 = arith.trunci %63 : i64 to i32
      asccube.copy_l0c2gm %alloc_3[%116], %112[%c0], %c128_i32, %c8_i32, %117, %c16_i32, %c0_i32 l2_cache_ctl %c0_i32 {nz2nd_en = true, relu_en = false} : memref<1xbf16>, memref<1xf32, 5>
      asccube.release_buf fixpipe, %99, 0 {cannir.buffer.generated_sync}
      scf.yield %109, %101 : i64, i64
    }
    return
  }
}
