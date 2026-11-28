module {
  func.func @run_m3_qr_980525913000de9f38c0eab0feaaf6ac(%arg0: !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg1: !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg2: !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg3: !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg4: !cannir.tensor<<gm, bf16>, <<(-1, -1, 512)>, <(-1, 512, 1)>>, <<(-1, -1, 512)>, <(-1, 512, 1)>>, identity>, %arg5: !cannir.tensor<<gm, f32>, <<(1, -1)>, <(-1, 1)>>, <<(1, -1)>, <(-1, 1)>>, identity>, %arg6: !cannir.tensor<<gm, f32>, <<(1, 512)>, <(512, 1)>>, <<(1, 512)>, <(512, 1)>>, identity>, %arg7: !cannir.tensor<<gm, ui8>, <<(-1, -1, 1, -1)>, <(-1, -1, -1, 1)>>, <<(-1, -1, 1, -1)>, <(-1, -1, -1, 1)>>, identity>, %arg8: !cannir.tensor<<gm, i64>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg9: !cannir.tensor<<gm, f32>, <<(1)>, <(1)>>, <<(1)>, <(1)>>, identity>, %arg10: !cannir.tensor<<gm, ui8>, <<(8, -1, -1)>, <(-1, -1, 1)>>, <<(8, -1, -1)>, <(-1, -1, 1)>>, identity>, %arg11: !cannir.tensor<<gm, f32>, <<(1)>, <(1)>>, <<(1)>, <(1)>>, identity>, %arg12: !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg13: !cannir.tensor<<gm, bf16>, <<(8, -1)>, <(-1, 1)>>, <<(8, -1)>, <(-1, 1)>>, identity>, %arg14: !cannir.tensor<<gm, f32>, <<(32, -1)>, <(-1, 1)>>, <<(32, -1)>, <(-1, 1)>>, identity>, %arg15: !cannir.tensor<<gm, bf16>, <<(8, -1)>, <(-1, 1)>>, <<(8, -1)>, <(-1, 1)>>, identity>, %arg16: !cannir.tensor<<gm, bf16>, <<(8, -1)>, <(-1, 1)>>, <<(8, -1)>, <(-1, 1)>>, identity>, %arg17: !cannir.tensor<<gm, f32>, <<(8, -1, -1)>, <(-1, -1, 1)>>, <<(8, -1, -1)>, <(-1, -1, 1)>>, identity>, %arg18: !cannir.tensor<<gm, f32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg19: !cannir.tensor<<gm, bf16>, <<(8, -1)>, <(-1, 1)>>, <<(8, -1)>, <(-1, 1)>>, identity>, %arg20: f32, %arg21: f32, %arg22: f32, %arg23: i64, %arg24: i64, %arg25: i64) attributes {cannbotdsl.contract_digest = "980525913000de9f38c0eab0feaaf6ac77c4552c6ca0a80a721678e5f0407b2c", cannbotdsl.public_slot_count = 26 : i64, cannbotdsl.slot_dtype_bits = [16, 16, 16, 16, 16, 32, 32, 8, 64, 32, 8, 32, 16, 16, 32, 16, 16, 32, 32, 16, 32, 32, 32, 64, 64, 64], cannbotdsl.slot_dtype_codes = [4, 4, 4, 4, 4, 2, 2, 1, 0, 2, 1, 2, 4, 4, 2, 4, 4, 2, 2, 4, 2, 2, 2, 0, 0, 0], cannbotdsl.slot_dtype_lanes = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], cannbotdsl.slot_flags = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], cannbotdsl.slot_kinds = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2], cannbotdsl.slot_type_digests = ["022b75a13987d7aef004e8fde8311ec1df5da13b3074eb5361860dc8ef2ecc8c", "b5c570c7c0159fc7a1075fa80de10b11b86c087e2cd3b2b946c60eb5f900ca19", "4d18a67066e55c86778cb78c908a80cc34547a2314d6ad8a79685f46673a5f89", "19b7e7050b7764b1c88f44198d5c4406c2e0c098dda1763d7bfe93f39fe0ae8a", "fd9e7bf46e5fdebb23db2cd9b9963e60289355042c04954fef2defa9b5be4007", "c8dd513e68d0aedae9a8e8521f33690b2fc63ba3dec5c2466e442b3a6bdebcad", "7b962932eb0067670c47bf914f64cc35656e412a09850be9b2bc297c5b918fc3", "277ce9bf46fc3cd46f3b9869de0350834ae00b3b3b136ec2cf276e01fd038628", "5ae7da10e83ac3f5ad55bbe47c978e5c8bcb0ad1d2171af81149eef24de96b7a", "b134b54071cdca5322ca9f5d39ee9881e5a8f90b566373e7bf45d235577c8f89", "4ce33545a16a5cfdc2711eac7f6dddb62c5b7f62dc29f4712cb761c9250029c1", "b134b54071cdca5322ca9f5d39ee9881e5a8f90b566373e7bf45d235577c8f89", "94630781bac8f1d566d1cf60e3b7a788d8ab31b7d3bd29e3f0948835334a7389", "04b43b0d54692facb90dd4907abdcbea2a855209f41c8d7117c4541d045feadb", "71457ab3f054c0e3c98727cb61d4f911f9bfd02151d9e1818a29adf3358c77f3", "e193cc2cd4f1cc826799f79ad2e88cbae4821553fdf9ffb7bfb8f20a2744fda9", "d086e48705d567aaf08e28892867093bc7951876e29b4e73f1528d2f336addf1", "2fa234002b306cf43ef7d6e1f36b1b982ae5bcf7410147ed1841efecc1d4324f", "02a82f6a9122bb99530a0ca4aa992d37b966da526006a2c53383ba9046eda13a", "f25adc241f574dff5e15cbbf47b60c522ade173908cfe64a7a59f8bcceebd153", "4a5363e2ec1eb8cefee037ab0b231653c07f9b44a406ef92ed9ba3e6f5541dbf", "4a5363e2ec1eb8cefee037ab0b231653c07f9b44a406ef92ed9ba3e6f5541dbf", "4a5363e2ec1eb8cefee037ab0b231653c07f9b44a406ef92ed9ba3e6f5541dbf", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515"], cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.host} {
    cannir.kernel_launch @kernel_MlaPrologKernel_run_m3_qr_0(%arg25) [%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18, %arg19, %arg20, %arg21, %arg22, %arg23, %arg24] {cannbotdsl.profile = {indices = array<i64: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19>, name = "MlaPrologKernel_run_m3_qr", op_type = "MlaPrologKernel_run_m3_qr", roles = array<i64: 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1>}} : !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, !cannir.tensor<<gm, bf16>, <<(-1, -1, 512)>, <(-1, 512, 1)>>, <<(-1, -1, 512)>, <(-1, 512, 1)>>, identity>, !cannir.tensor<<gm, f32>, <<(1, -1)>, <(-1, 1)>>, <<(1, -1)>, <(-1, 1)>>, identity>, !cannir.tensor<<gm, f32>, <<(1, 512)>, <(512, 1)>>, <<(1, 512)>, <(512, 1)>>, identity>, !cannir.tensor<<gm, ui8>, <<(-1, -1, 1, -1)>, <(-1, -1, -1, 1)>>, <<(-1, -1, 1, -1)>, <(-1, -1, -1, 1)>>, identity>, !cannir.tensor<<gm, i64>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, !cannir.tensor<<gm, f32>, <<(1)>, <(1)>>, <<(1)>, <(1)>>, identity>, !cannir.tensor<<gm, ui8>, <<(8, -1, -1)>, <(-1, -1, 1)>>, <<(8, -1, -1)>, <(-1, -1, 1)>>, identity>, !cannir.tensor<<gm, f32>, <<(1)>, <(1)>>, <<(1)>, <(1)>>, identity>, !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, !cannir.tensor<<gm, bf16>, <<(8, -1)>, <(-1, 1)>>, <<(8, -1)>, <(-1, 1)>>, identity>, !cannir.tensor<<gm, f32>, <<(32, -1)>, <(-1, 1)>>, <<(32, -1)>, <(-1, 1)>>, identity>, !cannir.tensor<<gm, bf16>, <<(8, -1)>, <(-1, 1)>>, <<(8, -1)>, <(-1, 1)>>, identity>, !cannir.tensor<<gm, bf16>, <<(8, -1)>, <(-1, 1)>>, <<(8, -1)>, <(-1, 1)>>, identity>, !cannir.tensor<<gm, f32>, <<(8, -1, -1)>, <(-1, -1, 1)>>, <<(8, -1, -1)>, <(-1, -1, 1)>>, identity>, !cannir.tensor<<gm, f32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, !cannir.tensor<<gm, bf16>, <<(8, -1)>, <(-1, 1)>>, <<(8, -1)>, <(-1, 1)>>, identity>, f32, f32, f32, i64, i64
    return
  }
  func.func @kernel_MlaPrologKernel_run_m3_qr_0_aiv(%arg0: !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg1: !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg2: !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg3: !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg4: !cannir.tensor<<gm, bf16>, <<(-1, -1, 512)>, <(-1, 512, 1)>>, <<(-1, -1, 512)>, <(-1, 512, 1)>>, identity>, %arg5: !cannir.tensor<<gm, f32>, <<(1, -1)>, <(-1, 1)>>, <<(1, -1)>, <(-1, 1)>>, identity>, %arg6: !cannir.tensor<<gm, f32>, <<(1, 512)>, <(512, 1)>>, <<(1, 512)>, <(512, 1)>>, identity>, %arg7: !cannir.tensor<<gm, ui8>, <<(-1, -1, 1, -1)>, <(-1, -1, -1, 1)>>, <<(-1, -1, 1, -1)>, <(-1, -1, -1, 1)>>, identity>, %arg8: !cannir.tensor<<gm, i64>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg9: !cannir.tensor<<gm, f32>, <<(1)>, <(1)>>, <<(1)>, <(1)>>, identity>, %arg10: !cannir.tensor<<gm, ui8>, <<(8, -1, -1)>, <(-1, -1, 1)>>, <<(8, -1, -1)>, <(-1, -1, 1)>>, identity>, %arg11: !cannir.tensor<<gm, f32>, <<(1)>, <(1)>>, <<(1)>, <(1)>>, identity>, %arg12: !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg13: !cannir.tensor<<gm, bf16>, <<(8, -1)>, <(-1, 1)>>, <<(8, -1)>, <(-1, 1)>>, identity>, %arg14: !cannir.tensor<<gm, f32>, <<(32, -1)>, <(-1, 1)>>, <<(32, -1)>, <(-1, 1)>>, identity>, %arg15: !cannir.tensor<<gm, bf16>, <<(8, -1)>, <(-1, 1)>>, <<(8, -1)>, <(-1, 1)>>, identity>, %arg16: !cannir.tensor<<gm, bf16>, <<(8, -1)>, <(-1, 1)>>, <<(8, -1)>, <(-1, 1)>>, identity>, %arg17: !cannir.tensor<<gm, f32>, <<(8, -1, -1)>, <(-1, -1, 1)>>, <<(8, -1, -1)>, <(-1, -1, 1)>>, identity>, %arg18: !cannir.tensor<<gm, f32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg19: !cannir.tensor<<gm, bf16>, <<(8, -1)>, <(-1, 1)>>, <<(8, -1)>, <(-1, 1)>>, identity>, %arg20: f32, %arg21: f32, %arg22: f32, %arg23: i64, %arg24: i64) attributes {cannir.buffer.auto_sync, cannir.buffer.sync_analyzed, cannir.buffer.sync_verified, cannir.buffer.tensor_state, cannir.core = "aiv", cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.device, cannir.explicit_ring_buffer_cursors, cannir.kernel_group = "kernel_MlaPrologKernel_run_m3_qr_0", cannir.resource.ids_prepared} {
    %c4_i32 = arith.constant 4 : i32
    %c512_i64 = arith.constant 512 : i64
    %c0_i32 = arith.constant 0 : i32
    %c16384_i64 = arith.constant 16384 : i64
    %c128_i32 = arith.constant 128 : i32
    %c0 = arith.constant 0 : index
    %c256_i64 = arith.constant 256 : i64
    %c1_i32 = arith.constant 1 : i32
    %false = arith.constant false
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
    %c576_i64 = arith.constant 576 : i64
    %c128_i64 = arith.constant 128 : i64
    %c2304_i64 = arith.constant 2304 : i64
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c64_i64 = arith.constant 64 : i64
    %c32_i64 = arith.constant 32 : i64
    %c24_i64 = arith.constant 24 : i64
    %c18_i64 = arith.constant 18 : i64
    %c16_i64 = arith.constant 16 : i64
    %c15_i64 = arith.constant 15 : i64
    %c14_i64 = arith.constant 14 : i64
    %c13_i64 = arith.constant 13 : i64
    %c8_i64 = arith.constant 8 : i64
    %c5_i64 = arith.constant 5 : i64
    %c4_i64 = arith.constant 4 : i64
    %c3_i64 = arith.constant 3 : i64
    %c2_i64 = arith.constant 2 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %alloc = memref.alloc() {cannir.arg_index = 0 : i64} : memref<1xbf16>
    %alloc_0 = memref.alloc() {cannir.arg_index = 5 : i64} : memref<1xf32>
    %alloc_1 = memref.alloc() {cannir.arg_index = 6 : i64} : memref<512xf32>
    %alloc_2 = memref.alloc() {cannir.arg_index = 7 : i64} : memref<1xui8>
    %alloc_3 = memref.alloc() {cannir.arg_index = 8 : i64} : memref<1xi64>
    %alloc_4 = memref.alloc() {cannir.arg_index = 9 : i64} : memref<1xf32>
    %alloc_5 = memref.alloc() {cannir.arg_index = 10 : i64} : memref<1xui8>
    %alloc_6 = memref.alloc() {cannir.arg_index = 11 : i64} : memref<1xf32>
    %alloc_7 = memref.alloc() {cannir.arg_index = 12 : i64} : memref<1xbf16>
    %alloc_8 = memref.alloc() {cannir.arg_index = 13 : i64} : memref<1xbf16>
    %alloc_9 = memref.alloc() {cannir.arg_index = 14 : i64} : memref<1xf32>
    %alloc_10 = memref.alloc() {cannir.arg_index = 15 : i64} : memref<1xbf16>
    %alloc_11 = memref.alloc() {cannir.arg_index = 16 : i64} : memref<1xbf16>
    %alloc_12 = memref.alloc() {cannir.arg_index = 17 : i64} : memref<1xf32>
    %alloc_13 = memref.alloc() {cannir.arg_index = 18 : i64} : memref<1xf32>
    %alloc_14 = memref.alloc() {cannir.arg_index = 19 : i64} : memref<1xbf16>
    %alloc_15 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<8192xbf16, 1>
    %alloc_16 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<8192xf32, 1>
    %alloc_17 = memref.alloc() {cannir.byte_offset = 32768 : i64} : memref<8192xf32, 1>
    %alloc_18 = memref.alloc() {cannir.byte_offset = 65536 : i64} : memref<2048xf32, 1>
    %alloc_19 = memref.alloc() {cannir.byte_offset = 73728 : i64} : memref<2048xf32, 1>
    %alloc_20 = memref.alloc() {cannir.byte_offset = 81920 : i64} : memref<2048xf32, 1>
    %alloc_21 = memref.alloc() {cannir.byte_offset = 106496 : i64} : memref<2048xbf16, 1>
    %alloc_22 = memref.alloc() {cannir.byte_offset = 110592 : i64} : memref<2048xbf16, 1>
    %alloc_23 = memref.alloc() {cannir.byte_offset = 114688 : i64} : memref<8xf32, 1>
    %alloc_24 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<576xbf16, 1>
    %alloc_25 = memref.alloc() {cannir.byte_offset = 1152 : i64} : memref<576xbf16, 1>
    %alloc_26 = memref.alloc() {cannir.byte_offset = 2304 : i64} : memref<576xf32, 1>
    %alloc_27 = memref.alloc() {cannir.byte_offset = 4608 : i64} : memref<576xf32, 1>
    %alloc_28 = memref.alloc() {cannir.byte_offset = 6912 : i64} : memref<512xf32, 1>
    %alloc_29 = memref.alloc() {cannir.byte_offset = 13760 : i64} : memref<576xui8, 1>
    %alloc_30 = memref.alloc() {cannir.byte_offset = 14336 : i64} : memref<576xui8, 1>
    %alloc_31 = memref.alloc() {cannir.byte_offset = 14912 : i64} : memref<64xf32, 1>
    %alloc_32 = memref.alloc() {cannir.byte_offset = 15168 : i64} : memref<8xf32, 1>
    %alloc_33 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<64xbf16, 1>
    %alloc_34 = memref.alloc() {cannir.byte_offset = 128 : i64} : memref<64xbf16, 1>
    %alloc_35 = memref.alloc() {cannir.byte_offset = 256 : i64} : memref<64xf32, 1>
    %alloc_36 = memref.alloc() {cannir.byte_offset = 512 : i64} : memref<64xf32, 1>
    %alloc_37 = memref.alloc() {cannir.byte_offset = 768 : i64} : memref<64xf32, 1>
    %alloc_38 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<576xf32, 1>
    %alloc_39 = memref.alloc() {cannir.byte_offset = 2304 : i64} : memref<576xf32, 1>
    %alloc_40 = memref.alloc() {cannir.byte_offset = 4608 : i64} : memref<64xf32, 1>
    %alloc_41 = memref.alloc() {cannir.byte_offset = 4864 : i64} : memref<64xf32, 1>
    %alloc_42 = memref.alloc() {cannir.byte_offset = 5120 : i64} : memref<8xf32, 1>
    %alloc_43 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<576xf32, 1>
    %alloc_44 = memref.alloc() {cannir.byte_offset = 2304 : i64} : memref<576xf32, 1>
    %alloc_45 = memref.alloc() {cannir.byte_offset = 4608 : i64} : memref<64xf32, 1>
    %alloc_46 = memref.alloc() {cannir.byte_offset = 4864 : i64} : memref<576xui8, 1>
    %alloc_47 = memref.alloc() {cannir.byte_offset = 5440 : i64} : memref<576xui8, 1>
    %alloc_48 = memref.alloc() {cannir.byte_offset = 6016 : i64} : memref<8xf32, 1>
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
      %42 = arith.muli %1, %c2_i64 : i64
      scf.for %arg25 = %20 to %c8_i64 step %42  : i64 {
        %43 = arith.cmpi slt, %arg25, %5 : i64
        scf.if %43 {
          %44 = ascscalar.bitcast_memref %alloc_15 byte_offset = %c0_i64 : memref<8192xbf16, 1> -> memref<?xbf16, 1>
          %45 = arith.minsi %arg25, %5 : i64
          %46 = arith.subi %5, %45 : i64
          %47 = arith.minsi %46, %c1_i64 : i64
          %48 = arith.cmpi sle, %arg25, %5 : i64
          %49 = arith.select %48, %47, %c0_i64 : i64
          %50 = cannir.extract_physical_layout %arg0 : !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
          %51 = cannir.extract_stride %50 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
          %52 = cannir.extract_dim %51[0] : !cannir.stride<(-1, 1)>
          %53 = arith.muli %52, %c2_i64 : i64
          %54 = arith.maxsi %6, %c1_i64 : i64
          %55 = arith.divsi %6, %54 : i64
          %56 = arith.minsi %55, %c0_i64 : i64
          %57 = arith.muli %56, %6 : i64
          %58 = arith.subi %6, %57 : i64
          %59 = arith.minsi %6, %58 : i64
          %60 = arith.cmpi sge, %55, %c0_i64 : i64
          %61 = arith.select %60, %59, %c0_i64 : i64
          %62 = arith.muli %arg25, %52 : i64
          %63 = arith.muli %62, %c2_i64 : i64
          %64 = arith.muli %61, %c2_i64 : i64
          %65 = arith.cmpi eq, %64, %c0_i64 : i64
          %66 = arith.select %65, %c0_i64, %49 : i64
          %67 = arith.select %65, %c2_i64, %64 : i64
          ascvec.get_buf mte2, %c0_i64, 0 {cannir.buffer.generated_sync}
          "ascvec.copy_gm2ub_plan"(%44, %alloc, %c0_i32, %c0_i32, %c0_i64, %63, %66, %67, %53, %c16384_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = 0 : i64, ub_capacity = 16384 : i64}> : (memref<?xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
          ascvec.release_buf mte2, %c0_i64, 0 {cannir.buffer.generated_sync}
          %68 = cannir.extract_physical_layout %arg13 : !cannir.tensor<<gm, bf16>, <<(8, -1)>, <(-1, 1)>>, <<(8, -1)>, <(-1, 1)>>, identity>
          %69 = cannir.extract_stride %68 : <<(8, -1)>, <(-1, 1)>> -> <(-1, 1)>
          %70 = cannir.extract_dim %69[0] : !cannir.stride<(-1, 1)>
          %71 = arith.muli %arg25, %70 : i64
          %72 = arith.muli %71, %c16_i64 : i64
          %73 = ascscalar.memref_address %alloc_8 : memref<1xbf16>
          %74 = arith.divui %72, %c8_i64 : i64
          %75 = arith.addi %73, %74 : i64
          %76 = ascscalar.bind_gm_ptr %75 : memref<1xbf16>
          %77 = arith.index_cast %6 : i64 to index
          %78 = cannir.extract_layout %arg13 : !cannir.tensor<<gm, bf16>, <<(8, -1)>, <(-1, 1)>>, <<(8, -1)>, <(-1, 1)>>, identity>
          %79 = cannir.extract_stride %78 : <<(8, -1)>, <(-1, 1)>> -> <(-1, 1)>
          %80 = cannir.extract_dim %79[0] : !cannir.stride<(-1, 1)>
          %81 = arith.index_cast %80 : i64 to index
          %82 = arith.muli %77, %c2 : index
          %83 = arith.muli %81, %c2 : index
          %84 = arith.cmpi ugt, %6, %c0_i64 : i64
          scf.if %84 {
            %85 = arith.index_cast %82 : index to i64
            %86 = arith.index_cast %83 : index to i64
            ascvec.get_buf mte3, %c0_i64, 0 {cannir.buffer.generated_sync}
            %87 = arith.trunci %85 : i64 to i32
            ascvec.copy_ub2gm %76[%c0], %44[%c0], %c1_i32, %87, %86, %87 l2_cache_ctl %c0_i32 : memref<1xbf16>, memref<?xbf16, 1>
            ascvec.release_buf mte3, %c0_i64, 0 {cannir.buffer.generated_sync}
          }
        } else {
          ascvec.get_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
          ascvec.vf(%alloc_15 : memref<8192xbf16, 1>) {
          ^bb0(%arg26: memref<8192xbf16, 1>):
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
            %c0_49 = arith.constant 0 : index
            %cst_50 = arith.constant 0.000000e+00 : bf16
            %c128_i32_51 = arith.constant 128 : i32
            %c0_52 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            scf.for %arg27 = %c0_52 to %c1 step %c1 {
              %mask, %remaining_out = ascvec.update_mask %c128_i32_51 elem_bits 16 : !ascvec.mask
              %62 = ascvec.duplicate_scalar %cst_50, %mask : bf16 -> <bf16>
              ascvec.storealign %arg26[%c0_49], %62, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c128], %62, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c256], %62, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c384], %62, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c512], %62, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c640], %62, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c768], %62, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c896], %62, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c1024], %62, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c1152], %62, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c1280], %62, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c1408], %62, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c1536], %62, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c1664], %62, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c1792], %62, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c1920], %62, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c2048], %62, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c2176], %62, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c2304], %62, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c2432], %62, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c2560], %62, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c2688], %62, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c2816], %62, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c2944], %62, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c3072], %62, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c3200], %62, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c3328], %62, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c3456], %62, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c3584], %62, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c3712], %62, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c3840], %62, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c3968], %62, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c4096], %62, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c4224], %62, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c4352], %62, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c4480], %62, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c4608], %62, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c4736], %62, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c4864], %62, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c4992], %62, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c5120], %62, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c5248], %62, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c5376], %62, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c5504], %62, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c5632], %62, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c5760], %62, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c5888], %62, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c6016], %62, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c6144], %62, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c6272], %62, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c6400], %62, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c6528], %62, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c6656], %62, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c6784], %62, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c6912], %62, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c7040], %62, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c7168], %62, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c7296], %62, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c7424], %62, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c7552], %62, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c7680], %62, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c7808], %62, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c7936], %62, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c8064], %62, %mask : memref<8192xbf16, 1>, index, <bf16>
            }
            ascvec.mem_bar vst_vld
          } {cannir.vf_mode = "simd"}
          ascvec.release_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
          %44 = cannir.extract_physical_layout %arg13 : !cannir.tensor<<gm, bf16>, <<(8, -1)>, <(-1, 1)>>, <<(8, -1)>, <(-1, 1)>>, identity>
          %45 = cannir.extract_stride %44 : <<(8, -1)>, <(-1, 1)>> -> <(-1, 1)>
          %46 = cannir.extract_dim %45[0] : !cannir.stride<(-1, 1)>
          %47 = arith.muli %arg25, %46 : i64
          %48 = arith.muli %47, %c16_i64 : i64
          %49 = ascscalar.memref_address %alloc_8 : memref<1xbf16>
          %50 = arith.divui %48, %c8_i64 : i64
          %51 = arith.addi %49, %50 : i64
          %52 = ascscalar.bind_gm_ptr %51 : memref<1xbf16>
          %53 = ascscalar.bitcast_memref %alloc_15 byte_offset = %c0_i64 : memref<8192xbf16, 1> -> memref<?xbf16, 1>
          %54 = arith.index_cast %6 : i64 to index
          %55 = cannir.extract_layout %arg13 : !cannir.tensor<<gm, bf16>, <<(8, -1)>, <(-1, 1)>>, <<(8, -1)>, <(-1, 1)>>, identity>
          %56 = cannir.extract_stride %55 : <<(8, -1)>, <(-1, 1)>> -> <(-1, 1)>
          %57 = cannir.extract_dim %56[0] : !cannir.stride<(-1, 1)>
          %58 = arith.index_cast %57 : i64 to index
          %59 = arith.muli %54, %c2 : index
          %60 = arith.muli %58, %c2 : index
          %61 = arith.cmpi ugt, %6, %c0_i64 : i64
          scf.if %61 {
            %62 = arith.index_cast %59 : index to i64
            %63 = arith.index_cast %60 : index to i64
            ascvec.get_buf mte3, %c0_i64, 0 {cannir.buffer.generated_sync}
            %64 = arith.trunci %62 : i64 to i32
            ascvec.copy_ub2gm %52[%c0], %53[%c0], %c1_i32, %64, %63, %64 l2_cache_ctl %c0_i32 : memref<1xbf16>, memref<?xbf16, 1>
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
      %42 = arith.muli %1, %c2_i64 : i64
      %43 = arith.sitofp %9 : i64 to f32
      %44 = arith.divf %cst, %43 : f32
      %45 = ascscalar.bitcast_memref %alloc_20 byte_offset = %c0_i64 : memref<2048xf32, 1> -> memref<?xf32, 1>
      %46 = cannir.extract_layout %arg5 : !cannir.tensor<<gm, f32>, <<(1, -1)>, <(-1, 1)>>, <<(1, -1)>, <(-1, 1)>>, identity>
      %47 = cannir.extract_shape %46 : <<(1, -1)>, <(-1, 1)>> -> <(1, -1)>
      %48 = cannir.extract_dim %47[1] : !cannir.shape<(1, -1)>
      %49 = arith.maxsi %9, %c1_i64 : i64
      %50 = arith.divsi %48, %49 : i64
      %51 = arith.minsi %50, %c0_i64 : i64
      %52 = arith.muli %51, %9 : i64
      %53 = arith.subi %48, %52 : i64
      %54 = arith.minsi %9, %53 : i64
      %55 = arith.cmpi sge, %50, %c0_i64 : i64
      %56 = arith.select %55, %54, %c0_i64 : i64
      %57 = arith.muli %56, %c4_i64 : i64
      %58 = arith.cmpi eq, %57, %c0_i64 : i64
      %59 = arith.cmpi ne, %57, %c0_i64 : i64
      %60 = arith.extui %59 : i1 to i64
      %61 = arith.select %58, %c4_i64, %57 : i64
      ascvec.get_buf mte2, %c2_i64, 0 {cannir.buffer.generated_sync}
      "ascvec.copy_gm2ub_plan"(%45, %alloc_0, %c0_i32, %c0_i32, %c0_i64, %c0_i64, %60, %61, %c0_i64, %61, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 81920 : i64, ub_capacity = 8192 : i64}> : (memref<?xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
      ascvec.release_buf mte2, %c2_i64, 0 {cannir.buffer.generated_sync}
      %62 = scf.for %arg25 = %20 to %5 step %42 iter_args(%arg26 = %c0_i64) -> (i64)  : i64 {
        %63 = arith.cmpi eq, %arg26, %c0_i64 : i64
        %64 = ascscalar.memref_address %alloc_16 : memref<8192xf32, 1>
        %65 = ascscalar.memref_address %alloc_17 : memref<8192xf32, 1>
        %66 = arith.select %63, %64, %65 : i64
        %67 = arith.cmpi ne, %arg26, %c0_i64 : i64
        %68 = arith.extui %67 : i1 to i64
        %69 = arith.addi %arg26, %c1_i64 overflow<nsw> : i64
        %70 = arith.remui %69, %c2_i64 : i64
        %71 = arith.minsi %arg25, %c32_i64 : i64
        %72 = arith.subi %c32_i64, %71 : i64
        %73 = arith.minsi %72, %c1_i64 : i64
        %74 = arith.cmpi sle, %arg25, %c32_i64 : i64
        %75 = arith.select %74, %73, %c0_i64 : i64
        %76 = cannir.extract_physical_layout %arg14 : !cannir.tensor<<gm, f32>, <<(32, -1)>, <(-1, 1)>>, <<(32, -1)>, <(-1, 1)>>, identity>
        %77 = cannir.extract_stride %76 : <<(32, -1)>, <(-1, 1)>> -> <(-1, 1)>
        %78 = cannir.extract_dim %77[0] : !cannir.stride<(-1, 1)>
        %79 = arith.muli %78, %c4_i64 : i64
        %80 = cannir.extract_layout %arg14 : !cannir.tensor<<gm, f32>, <<(32, -1)>, <(-1, 1)>>, <<(32, -1)>, <(-1, 1)>>, identity>
        %81 = cannir.extract_shape %80 : <<(32, -1)>, <(-1, 1)>> -> <(32, -1)>
        %82 = cannir.extract_dim %81[1] : !cannir.shape<(32, -1)>
        %83 = arith.divsi %82, %49 : i64
        %84 = arith.minsi %83, %c0_i64 : i64
        %85 = arith.muli %84, %9 : i64
        %86 = arith.subi %82, %85 : i64
        %87 = arith.minsi %9, %86 : i64
        %88 = arith.cmpi sge, %83, %c0_i64 : i64
        %89 = arith.select %88, %87, %c0_i64 : i64
        %90 = arith.muli %arg25, %78 : i64
        %91 = arith.muli %90, %c4_i64 : i64
        %92 = arith.muli %89, %c4_i64 : i64
        %93 = arith.cmpi eq, %92, %c0_i64 : i64
        %94 = arith.select %93, %c0_i64, %75 : i64
        %95 = arith.select %93, %c4_i64, %92 : i64
        %96 = ascscalar.bind_local_ptr %66 : memref<1xf32, 1>
        ascvec.get_buf mte2, %68, 0 {cannir.buffer.generated_sync}
        "ascvec.copy_gm2ub_plan"(%96, %alloc_9, %c0_i32, %c0_i32, %c0_i64, %91, %94, %95, %79, %c8192_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = -1 : i64, ub_capacity = 4 : i64}> : (memref<1xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
        %97 = arith.addi %arg25, %c8_i64 : i64
        %98 = arith.minsi %97, %c32_i64 : i64
        %99 = arith.subi %c32_i64, %98 : i64
        %100 = arith.minsi %99, %c1_i64 : i64
        %101 = arith.cmpi sle, %97, %c32_i64 : i64
        %102 = arith.select %101, %100, %c0_i64 : i64
        %103 = arith.muli %97, %78 : i64
        %104 = arith.muli %103, %c4_i64 : i64
        %105 = arith.select %93, %c0_i64, %102 : i64
        "ascvec.copy_gm2ub_plan"(%96, %alloc_9, %c0_i32, %c0_i32, %c8192_i64, %104, %105, %95, %79, %c8192_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = -1 : i64, ub_capacity = 4 : i64}> : (memref<1xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
        %106 = arith.addi %arg25, %c16_i64 : i64
        %107 = arith.minsi %106, %c32_i64 : i64
        %108 = arith.subi %c32_i64, %107 : i64
        %109 = arith.minsi %108, %c1_i64 : i64
        %110 = arith.cmpi sle, %106, %c32_i64 : i64
        %111 = arith.select %110, %109, %c0_i64 : i64
        %112 = arith.muli %106, %78 : i64
        %113 = arith.muli %112, %c4_i64 : i64
        %114 = arith.select %93, %c0_i64, %111 : i64
        "ascvec.copy_gm2ub_plan"(%96, %alloc_9, %c0_i32, %c0_i32, %c16384_i64, %113, %114, %95, %79, %c8192_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = -1 : i64, ub_capacity = 4 : i64}> : (memref<1xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
        %115 = arith.addi %arg25, %c24_i64 : i64
        %116 = arith.minsi %115, %c32_i64 : i64
        %117 = arith.subi %c32_i64, %116 : i64
        %118 = arith.minsi %117, %c1_i64 : i64
        %119 = arith.cmpi sle, %115, %c32_i64 : i64
        %120 = arith.select %119, %118, %c0_i64 : i64
        %121 = arith.muli %115, %78 : i64
        %122 = arith.muli %121, %c4_i64 : i64
        %123 = arith.select %93, %c0_i64, %120 : i64
        "ascvec.copy_gm2ub_plan"(%96, %alloc_9, %c0_i32, %c0_i32, %c24576_i64, %122, %123, %95, %79, %c8192_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = -1 : i64, ub_capacity = 4 : i64}> : (memref<1xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
        ascvec.release_buf mte2, %68, 0 {cannir.buffer.generated_sync}
        %124 = ascscalar.bind_local_ptr %66 : memref<8192xf32, 1>
        %125 = ascscalar.memref_address %alloc_18 : memref<2048xf32, 1>
        %126 = ascscalar.memref_address %alloc_19 : memref<2048xf32, 1>
        %127 = arith.select %63, %125, %126 : i64
        %128 = ascscalar.bind_local_ptr %127 : memref<2048xf32, 1>
        %129 = arith.trunci %9 : i64 to i32
        ascvec.get_buf v, %68, 0 {cannir.buffer.generated_sync}
        ascvec.vf(%124, %128, %129 : memref<8192xf32, 1>, memref<2048xf32, 1>, i32) {
        ^bb0(%arg27: memref<8192xf32, 1>, %arg28: memref<2048xf32, 1>, %arg29: i32):
          %c64_i32_49 = arith.constant 64 : i32
          %c0_i32_50 = arith.constant 0 : i32
          %c6144_i32 = arith.constant 6144 : i32
          %c4096_i32 = arith.constant 4096 : i32
          %c2048_i32 = arith.constant 2048 : i32
          scf.for %arg30 = %c0_i32_50 to %arg29 step %c64_i32_49  : i32 {
            %262 = arith.index_cast %arg30 : i32 to index
            %263 = ascvec.loadalign %arg27[%262] : memref<8192xf32, 1>, index -> <f32>
            %264 = arith.addi %arg30, %c2048_i32 : i32
            %265 = arith.index_cast %264 : i32 to index
            %266 = ascvec.loadalign %arg27[%265] : memref<8192xf32, 1>, index -> <f32>
            %267 = ascvec.create_mask : !ascvec.mask
            %268 = ascvec.add %263, %266, %267 : <f32>
            %269 = arith.addi %arg30, %c4096_i32 : i32
            %270 = arith.index_cast %269 : i32 to index
            %271 = ascvec.loadalign %arg27[%270] : memref<8192xf32, 1>, index -> <f32>
            %272 = arith.addi %arg30, %c6144_i32 : i32
            %273 = arith.index_cast %272 : i32 to index
            %274 = ascvec.loadalign %arg27[%273] : memref<8192xf32, 1>, index -> <f32>
            %275 = ascvec.add %271, %274, %267 : <f32>
            %276 = ascvec.add %268, %275, %267 : <f32>
            %277 = ascvec.cast %276, %267 : <f32> -> <bf16>
            %278 = ascvec.cast %277, %267 {rounding = 6 : i32} : <bf16> -> <f32>
            ascvec.storealign %arg28[%262], %278, %267 : memref<2048xf32, 1>, index, <f32>
          }
          ascvec.mem_bar vst_vld
        } {cannir.vf_mode = "simd"}
        ascvec.release_buf v, %68, 0 {cannir.buffer.generated_sync}
        %130 = arith.maxsi %129, %c0_i32 : i32
        %131 = arith.minsi %130, %c64_i32 : i32
        %132 = arith.subi %129, %c64_i32 : i32
        %133 = arith.maxsi %132, %c0_i32 : i32
        %134 = arith.minsi %133, %c64_i32 : i32
        %135 = arith.subi %129, %c128_i32 : i32
        %136 = arith.maxsi %135, %c0_i32 : i32
        %137 = arith.minsi %136, %c64_i32 : i32
        %138 = arith.subi %129, %c192_i32 : i32
        %139 = arith.maxsi %138, %c0_i32 : i32
        %140 = arith.minsi %139, %c64_i32 : i32
        %141 = arith.subi %129, %c256_i32 : i32
        %142 = arith.maxsi %141, %c0_i32 : i32
        %143 = arith.minsi %142, %c64_i32 : i32
        %144 = arith.subi %129, %c320_i32 : i32
        %145 = arith.maxsi %144, %c0_i32 : i32
        %146 = arith.minsi %145, %c64_i32 : i32
        %147 = arith.subi %129, %c384_i32 : i32
        %148 = arith.maxsi %147, %c0_i32 : i32
        %149 = arith.minsi %148, %c64_i32 : i32
        %150 = arith.subi %129, %c448_i32 : i32
        %151 = arith.maxsi %150, %c0_i32 : i32
        %152 = arith.minsi %151, %c64_i32 : i32
        %153 = arith.subi %129, %c512_i32 : i32
        %154 = arith.maxsi %153, %c0_i32 : i32
        %155 = arith.minsi %154, %c64_i32 : i32
        %156 = arith.subi %129, %c576_i32 : i32
        %157 = arith.maxsi %156, %c0_i32 : i32
        %158 = arith.minsi %157, %c64_i32 : i32
        %159 = arith.subi %129, %c640_i32 : i32
        %160 = arith.maxsi %159, %c0_i32 : i32
        %161 = arith.minsi %160, %c64_i32 : i32
        %162 = arith.subi %129, %c704_i32 : i32
        %163 = arith.maxsi %162, %c0_i32 : i32
        %164 = arith.minsi %163, %c64_i32 : i32
        %165 = arith.subi %129, %c768_i32 : i32
        %166 = arith.maxsi %165, %c0_i32 : i32
        %167 = arith.minsi %166, %c64_i32 : i32
        %168 = arith.subi %129, %c832_i32 : i32
        %169 = arith.maxsi %168, %c0_i32 : i32
        %170 = arith.minsi %169, %c64_i32 : i32
        %171 = arith.subi %129, %c896_i32 : i32
        %172 = arith.maxsi %171, %c0_i32 : i32
        %173 = arith.minsi %172, %c64_i32 : i32
        %174 = arith.subi %129, %c960_i32 : i32
        %175 = arith.maxsi %174, %c0_i32 : i32
        %176 = arith.minsi %175, %c64_i32 : i32
        %177 = arith.subi %129, %c1024_i32 : i32
        %178 = arith.maxsi %177, %c0_i32 : i32
        %179 = arith.minsi %178, %c64_i32 : i32
        %180 = arith.subi %129, %c1088_i32 : i32
        %181 = arith.maxsi %180, %c0_i32 : i32
        %182 = arith.minsi %181, %c64_i32 : i32
        %183 = arith.subi %129, %c1152_i32 : i32
        %184 = arith.maxsi %183, %c0_i32 : i32
        %185 = arith.minsi %184, %c64_i32 : i32
        %186 = arith.subi %129, %c1216_i32 : i32
        %187 = arith.maxsi %186, %c0_i32 : i32
        %188 = arith.minsi %187, %c64_i32 : i32
        %189 = arith.subi %129, %c1280_i32 : i32
        %190 = arith.maxsi %189, %c0_i32 : i32
        %191 = arith.minsi %190, %c64_i32 : i32
        %192 = arith.subi %129, %c1344_i32 : i32
        %193 = arith.maxsi %192, %c0_i32 : i32
        %194 = arith.minsi %193, %c64_i32 : i32
        %195 = arith.subi %129, %c1408_i32 : i32
        %196 = arith.maxsi %195, %c0_i32 : i32
        %197 = arith.minsi %196, %c64_i32 : i32
        %198 = arith.subi %129, %c1472_i32 : i32
        %199 = arith.maxsi %198, %c0_i32 : i32
        %200 = arith.minsi %199, %c64_i32 : i32
        %201 = arith.subi %129, %c1536_i32 : i32
        %202 = arith.maxsi %201, %c0_i32 : i32
        %203 = arith.minsi %202, %c64_i32 : i32
        %204 = arith.subi %129, %c1600_i32 : i32
        %205 = arith.maxsi %204, %c0_i32 : i32
        %206 = arith.minsi %205, %c64_i32 : i32
        %207 = arith.subi %129, %c1664_i32 : i32
        %208 = arith.maxsi %207, %c0_i32 : i32
        %209 = arith.minsi %208, %c64_i32 : i32
        %210 = arith.subi %129, %c1728_i32 : i32
        %211 = arith.maxsi %210, %c0_i32 : i32
        %212 = arith.minsi %211, %c64_i32 : i32
        %213 = arith.subi %129, %c1792_i32 : i32
        %214 = arith.maxsi %213, %c0_i32 : i32
        %215 = arith.minsi %214, %c64_i32 : i32
        %216 = arith.subi %129, %c1856_i32 : i32
        %217 = arith.maxsi %216, %c0_i32 : i32
        %218 = arith.minsi %217, %c64_i32 : i32
        %219 = arith.subi %129, %c1920_i32 : i32
        %220 = arith.maxsi %219, %c0_i32 : i32
        %221 = arith.minsi %220, %c64_i32 : i32
        %222 = arith.subi %129, %c1984_i32 : i32
        %223 = arith.maxsi %222, %c0_i32 : i32
        %224 = arith.minsi %223, %c64_i32 : i32
        ascvec.vf(%131, %128, %134, %137, %140, %143, %146, %149, %152, %155, %158, %161, %164, %167, %170, %173, %176, %179, %182, %185, %188, %191, %194, %197, %200, %203, %206, %209, %212, %215, %218, %221, %224, %44, %arg20, %alloc_23 : i32, memref<2048xf32, 1>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, f32, f32, memref<8xf32, 1>) {
        ^bb0(%arg27: i32, %arg28: memref<2048xf32, 1>, %arg29: i32, %arg30: i32, %arg31: i32, %arg32: i32, %arg33: i32, %arg34: i32, %arg35: i32, %arg36: i32, %arg37: i32, %arg38: i32, %arg39: i32, %arg40: i32, %arg41: i32, %arg42: i32, %arg43: i32, %arg44: i32, %arg45: i32, %arg46: i32, %arg47: i32, %arg48: i32, %arg49: i32, %arg50: i32, %arg51: i32, %arg52: i32, %arg53: i32, %arg54: i32, %arg55: i32, %arg56: i32, %arg57: i32, %arg58: i32, %arg59: i32, %arg60: f32, %arg61: f32, %arg62: memref<8xf32, 1>):
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
          %c0_49 = arith.constant 0 : index
          %cst_50 = arith.constant 0.000000e+00 : f32
          %c1_i32_51 = arith.constant 1 : i32
          %c0_52 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          scf.for %arg63 = %c0_52 to %c1 step %c1 {
            %mask, %remaining_out = ascvec.update_mask %c1_i32_51 : !ascvec.mask
            %262 = ascvec.duplicate_scalar %cst_50, %mask : f32 -> <f32>
            %mask_53, %remaining_out_54 = ascvec.update_mask %arg27 : !ascvec.mask
            %263 = ascvec.loadalign %arg28[%c0_49] : memref<2048xf32, 1>, index -> <f32>
            %264 = ascvec.mul %263, %263, %mask_53 : <f32>
            %265 = ascvec.reduce_sum %264, %mask_53 : <f32>
            %266 = ascvec.add %262, %265, %mask : <f32>
            %mask_55, %remaining_out_56 = ascvec.update_mask %arg29 : !ascvec.mask
            %267 = ascvec.loadalign %arg28[%c64] : memref<2048xf32, 1>, index -> <f32>
            %268 = ascvec.mul %267, %267, %mask_55 : <f32>
            %269 = ascvec.reduce_sum %268, %mask_55 : <f32>
            %270 = ascvec.add %266, %269, %mask : <f32>
            %mask_57, %remaining_out_58 = ascvec.update_mask %arg30 : !ascvec.mask
            %271 = ascvec.loadalign %arg28[%c128] : memref<2048xf32, 1>, index -> <f32>
            %272 = ascvec.mul %271, %271, %mask_57 : <f32>
            %273 = ascvec.reduce_sum %272, %mask_57 : <f32>
            %274 = ascvec.add %270, %273, %mask : <f32>
            %mask_59, %remaining_out_60 = ascvec.update_mask %arg31 : !ascvec.mask
            %275 = ascvec.loadalign %arg28[%c192] : memref<2048xf32, 1>, index -> <f32>
            %276 = ascvec.mul %275, %275, %mask_59 : <f32>
            %277 = ascvec.reduce_sum %276, %mask_59 : <f32>
            %278 = ascvec.add %274, %277, %mask : <f32>
            %mask_61, %remaining_out_62 = ascvec.update_mask %arg32 : !ascvec.mask
            %279 = ascvec.loadalign %arg28[%c256] : memref<2048xf32, 1>, index -> <f32>
            %280 = ascvec.mul %279, %279, %mask_61 : <f32>
            %281 = ascvec.reduce_sum %280, %mask_61 : <f32>
            %282 = ascvec.add %278, %281, %mask : <f32>
            %mask_63, %remaining_out_64 = ascvec.update_mask %arg33 : !ascvec.mask
            %283 = ascvec.loadalign %arg28[%c320] : memref<2048xf32, 1>, index -> <f32>
            %284 = ascvec.mul %283, %283, %mask_63 : <f32>
            %285 = ascvec.reduce_sum %284, %mask_63 : <f32>
            %286 = ascvec.add %282, %285, %mask : <f32>
            %mask_65, %remaining_out_66 = ascvec.update_mask %arg34 : !ascvec.mask
            %287 = ascvec.loadalign %arg28[%c384] : memref<2048xf32, 1>, index -> <f32>
            %288 = ascvec.mul %287, %287, %mask_65 : <f32>
            %289 = ascvec.reduce_sum %288, %mask_65 : <f32>
            %290 = ascvec.add %286, %289, %mask : <f32>
            %mask_67, %remaining_out_68 = ascvec.update_mask %arg35 : !ascvec.mask
            %291 = ascvec.loadalign %arg28[%c448] : memref<2048xf32, 1>, index -> <f32>
            %292 = ascvec.mul %291, %291, %mask_67 : <f32>
            %293 = ascvec.reduce_sum %292, %mask_67 : <f32>
            %294 = ascvec.add %290, %293, %mask : <f32>
            %mask_69, %remaining_out_70 = ascvec.update_mask %arg36 : !ascvec.mask
            %295 = ascvec.loadalign %arg28[%c512] : memref<2048xf32, 1>, index -> <f32>
            %296 = ascvec.mul %295, %295, %mask_69 : <f32>
            %297 = ascvec.reduce_sum %296, %mask_69 : <f32>
            %298 = ascvec.add %294, %297, %mask : <f32>
            %mask_71, %remaining_out_72 = ascvec.update_mask %arg37 : !ascvec.mask
            %299 = ascvec.loadalign %arg28[%c576] : memref<2048xf32, 1>, index -> <f32>
            %300 = ascvec.mul %299, %299, %mask_71 : <f32>
            %301 = ascvec.reduce_sum %300, %mask_71 : <f32>
            %302 = ascvec.add %298, %301, %mask : <f32>
            %mask_73, %remaining_out_74 = ascvec.update_mask %arg38 : !ascvec.mask
            %303 = ascvec.loadalign %arg28[%c640] : memref<2048xf32, 1>, index -> <f32>
            %304 = ascvec.mul %303, %303, %mask_73 : <f32>
            %305 = ascvec.reduce_sum %304, %mask_73 : <f32>
            %306 = ascvec.add %302, %305, %mask : <f32>
            %mask_75, %remaining_out_76 = ascvec.update_mask %arg39 : !ascvec.mask
            %307 = ascvec.loadalign %arg28[%c704] : memref<2048xf32, 1>, index -> <f32>
            %308 = ascvec.mul %307, %307, %mask_75 : <f32>
            %309 = ascvec.reduce_sum %308, %mask_75 : <f32>
            %310 = ascvec.add %306, %309, %mask : <f32>
            %mask_77, %remaining_out_78 = ascvec.update_mask %arg40 : !ascvec.mask
            %311 = ascvec.loadalign %arg28[%c768] : memref<2048xf32, 1>, index -> <f32>
            %312 = ascvec.mul %311, %311, %mask_77 : <f32>
            %313 = ascvec.reduce_sum %312, %mask_77 : <f32>
            %314 = ascvec.add %310, %313, %mask : <f32>
            %mask_79, %remaining_out_80 = ascvec.update_mask %arg41 : !ascvec.mask
            %315 = ascvec.loadalign %arg28[%c832] : memref<2048xf32, 1>, index -> <f32>
            %316 = ascvec.mul %315, %315, %mask_79 : <f32>
            %317 = ascvec.reduce_sum %316, %mask_79 : <f32>
            %318 = ascvec.add %314, %317, %mask : <f32>
            %mask_81, %remaining_out_82 = ascvec.update_mask %arg42 : !ascvec.mask
            %319 = ascvec.loadalign %arg28[%c896] : memref<2048xf32, 1>, index -> <f32>
            %320 = ascvec.mul %319, %319, %mask_81 : <f32>
            %321 = ascvec.reduce_sum %320, %mask_81 : <f32>
            %322 = ascvec.add %318, %321, %mask : <f32>
            %mask_83, %remaining_out_84 = ascvec.update_mask %arg43 : !ascvec.mask
            %323 = ascvec.loadalign %arg28[%c960] : memref<2048xf32, 1>, index -> <f32>
            %324 = ascvec.mul %323, %323, %mask_83 : <f32>
            %325 = ascvec.reduce_sum %324, %mask_83 : <f32>
            %326 = ascvec.add %322, %325, %mask : <f32>
            %mask_85, %remaining_out_86 = ascvec.update_mask %arg44 : !ascvec.mask
            %327 = ascvec.loadalign %arg28[%c1024] : memref<2048xf32, 1>, index -> <f32>
            %328 = ascvec.mul %327, %327, %mask_85 : <f32>
            %329 = ascvec.reduce_sum %328, %mask_85 : <f32>
            %330 = ascvec.add %326, %329, %mask : <f32>
            %mask_87, %remaining_out_88 = ascvec.update_mask %arg45 : !ascvec.mask
            %331 = ascvec.loadalign %arg28[%c1088] : memref<2048xf32, 1>, index -> <f32>
            %332 = ascvec.mul %331, %331, %mask_87 : <f32>
            %333 = ascvec.reduce_sum %332, %mask_87 : <f32>
            %334 = ascvec.add %330, %333, %mask : <f32>
            %mask_89, %remaining_out_90 = ascvec.update_mask %arg46 : !ascvec.mask
            %335 = ascvec.loadalign %arg28[%c1152] : memref<2048xf32, 1>, index -> <f32>
            %336 = ascvec.mul %335, %335, %mask_89 : <f32>
            %337 = ascvec.reduce_sum %336, %mask_89 : <f32>
            %338 = ascvec.add %334, %337, %mask : <f32>
            %mask_91, %remaining_out_92 = ascvec.update_mask %arg47 : !ascvec.mask
            %339 = ascvec.loadalign %arg28[%c1216] : memref<2048xf32, 1>, index -> <f32>
            %340 = ascvec.mul %339, %339, %mask_91 : <f32>
            %341 = ascvec.reduce_sum %340, %mask_91 : <f32>
            %342 = ascvec.add %338, %341, %mask : <f32>
            %mask_93, %remaining_out_94 = ascvec.update_mask %arg48 : !ascvec.mask
            %343 = ascvec.loadalign %arg28[%c1280] : memref<2048xf32, 1>, index -> <f32>
            %344 = ascvec.mul %343, %343, %mask_93 : <f32>
            %345 = ascvec.reduce_sum %344, %mask_93 : <f32>
            %346 = ascvec.add %342, %345, %mask : <f32>
            %mask_95, %remaining_out_96 = ascvec.update_mask %arg49 : !ascvec.mask
            %347 = ascvec.loadalign %arg28[%c1344] : memref<2048xf32, 1>, index -> <f32>
            %348 = ascvec.mul %347, %347, %mask_95 : <f32>
            %349 = ascvec.reduce_sum %348, %mask_95 : <f32>
            %350 = ascvec.add %346, %349, %mask : <f32>
            %mask_97, %remaining_out_98 = ascvec.update_mask %arg50 : !ascvec.mask
            %351 = ascvec.loadalign %arg28[%c1408] : memref<2048xf32, 1>, index -> <f32>
            %352 = ascvec.mul %351, %351, %mask_97 : <f32>
            %353 = ascvec.reduce_sum %352, %mask_97 : <f32>
            %354 = ascvec.add %350, %353, %mask : <f32>
            %mask_99, %remaining_out_100 = ascvec.update_mask %arg51 : !ascvec.mask
            %355 = ascvec.loadalign %arg28[%c1472] : memref<2048xf32, 1>, index -> <f32>
            %356 = ascvec.mul %355, %355, %mask_99 : <f32>
            %357 = ascvec.reduce_sum %356, %mask_99 : <f32>
            %358 = ascvec.add %354, %357, %mask : <f32>
            %mask_101, %remaining_out_102 = ascvec.update_mask %arg52 : !ascvec.mask
            %359 = ascvec.loadalign %arg28[%c1536] : memref<2048xf32, 1>, index -> <f32>
            %360 = ascvec.mul %359, %359, %mask_101 : <f32>
            %361 = ascvec.reduce_sum %360, %mask_101 : <f32>
            %362 = ascvec.add %358, %361, %mask : <f32>
            %mask_103, %remaining_out_104 = ascvec.update_mask %arg53 : !ascvec.mask
            %363 = ascvec.loadalign %arg28[%c1600] : memref<2048xf32, 1>, index -> <f32>
            %364 = ascvec.mul %363, %363, %mask_103 : <f32>
            %365 = ascvec.reduce_sum %364, %mask_103 : <f32>
            %366 = ascvec.add %362, %365, %mask : <f32>
            %mask_105, %remaining_out_106 = ascvec.update_mask %arg54 : !ascvec.mask
            %367 = ascvec.loadalign %arg28[%c1664] : memref<2048xf32, 1>, index -> <f32>
            %368 = ascvec.mul %367, %367, %mask_105 : <f32>
            %369 = ascvec.reduce_sum %368, %mask_105 : <f32>
            %370 = ascvec.add %366, %369, %mask : <f32>
            %mask_107, %remaining_out_108 = ascvec.update_mask %arg55 : !ascvec.mask
            %371 = ascvec.loadalign %arg28[%c1728] : memref<2048xf32, 1>, index -> <f32>
            %372 = ascvec.mul %371, %371, %mask_107 : <f32>
            %373 = ascvec.reduce_sum %372, %mask_107 : <f32>
            %374 = ascvec.add %370, %373, %mask : <f32>
            %mask_109, %remaining_out_110 = ascvec.update_mask %arg56 : !ascvec.mask
            %375 = ascvec.loadalign %arg28[%c1792] : memref<2048xf32, 1>, index -> <f32>
            %376 = ascvec.mul %375, %375, %mask_109 : <f32>
            %377 = ascvec.reduce_sum %376, %mask_109 : <f32>
            %378 = ascvec.add %374, %377, %mask : <f32>
            %mask_111, %remaining_out_112 = ascvec.update_mask %arg57 : !ascvec.mask
            %379 = ascvec.loadalign %arg28[%c1856] : memref<2048xf32, 1>, index -> <f32>
            %380 = ascvec.mul %379, %379, %mask_111 : <f32>
            %381 = ascvec.reduce_sum %380, %mask_111 : <f32>
            %382 = ascvec.add %378, %381, %mask : <f32>
            %mask_113, %remaining_out_114 = ascvec.update_mask %arg58 : !ascvec.mask
            %383 = ascvec.loadalign %arg28[%c1920] : memref<2048xf32, 1>, index -> <f32>
            %384 = ascvec.mul %383, %383, %mask_113 : <f32>
            %385 = ascvec.reduce_sum %384, %mask_113 : <f32>
            %386 = ascvec.add %382, %385, %mask : <f32>
            %mask_115, %remaining_out_116 = ascvec.update_mask %arg59 : !ascvec.mask
            %387 = ascvec.loadalign %arg28[%c1984] : memref<2048xf32, 1>, index -> <f32>
            %388 = ascvec.mul %387, %387, %mask_115 : <f32>
            %389 = ascvec.reduce_sum %388, %mask_115 : <f32>
            %390 = ascvec.add %386, %389, %mask : <f32>
            %391 = ascvec.mul_scalar %390, %arg60, %mask : <f32>, f32
            %392 = ascvec.add_scalar %391, %arg61, %mask : <f32>, f32
            %393 = ascvec.sqrt %392, %mask : <f32>
            ascvec.storealign_1st %arg62[%c0_49], %393 : memref<8xf32, 1>, index, <f32>
          }
          ascvec.mem_bar vst_vld
        } {cannir.vf_mode = "simd"}
        %225 = ascscalar.memref_address %alloc_21 : memref<2048xbf16, 1>
        %226 = ascscalar.memref_address %alloc_22 : memref<2048xbf16, 1>
        %227 = arith.select %63, %225, %226 : i64
        %228 = ascscalar.bind_local_ptr %227 : memref<2048xbf16, 1>
        %229 = arith.select %63, %c3_i64, %c4_i64 : i64
        ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
        ascvec.get_buf v, %229, 0 {cannir.buffer.generated_sync}
        ascvec.vf(%alloc_23, %128, %alloc_20, %arg21, %228, %129 : memref<8xf32, 1>, memref<2048xf32, 1>, memref<2048xf32, 1>, f32, memref<2048xbf16, 1>, i32) {
        ^bb0(%arg27: memref<8xf32, 1>, %arg28: memref<2048xf32, 1>, %arg29: memref<2048xf32, 1>, %arg30: f32, %arg31: memref<2048xbf16, 1>, %arg32: i32):
          %c64_i32_49 = arith.constant 64 : i32
          %c0_i32_50 = arith.constant 0 : i32
          %c0_51 = arith.constant 0 : index
          %262 = ascvec.loadalign_brc %arg27[%c0_51] : memref<8xf32, 1>, index -> <f32>
          scf.for %arg33 = %c0_i32_50 to %arg32 step %c64_i32_49  : i32 {
            %263 = arith.index_cast %arg33 : i32 to index
            %264 = ascvec.loadalign %arg28[%263] : memref<2048xf32, 1>, index -> <f32>
            %265 = ascvec.create_mask : !ascvec.mask
            %266 = ascvec.div %264, %262, %265 {precision_mode = true} : <f32>
            %267 = ascvec.loadalign %arg29[%263] : memref<2048xf32, 1>, index -> <f32>
            %268 = ascvec.mul %266, %267, %265 : <f32>
            %269 = ascvec.mul_scalar %268, %arg30, %265 : <f32>, f32
            %270 = ascvec.cast %269, %265 : <f32> -> <bf16>
            ascvec.storealign_pack %arg31[%263], %270, %265 {mode = "b32_to_b16"} : memref<2048xbf16, 1>, index, <bf16>
          }
          ascvec.mem_bar vst_vld
        } {cannir.vf_mode = "simd"}
        ascvec.release_buf v, %229, 0 {cannir.buffer.generated_sync}
        ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
        %230 = cannir.extract_physical_layout %arg19 : !cannir.tensor<<gm, bf16>, <<(8, -1)>, <(-1, 1)>>, <<(8, -1)>, <(-1, 1)>>, identity>
        %231 = cannir.extract_stride %230 : <<(8, -1)>, <(-1, 1)>> -> <(-1, 1)>
        %232 = cannir.extract_dim %231[0] : !cannir.stride<(-1, 1)>
        %233 = arith.muli %arg25, %232 : i64
        %234 = arith.muli %233, %c16_i64 : i64
        %235 = ascscalar.memref_address %alloc_14 : memref<1xbf16>
        %236 = arith.divui %234, %c8_i64 : i64
        %237 = arith.addi %235, %236 : i64
        %238 = ascscalar.bind_gm_ptr %237 : memref<1xbf16>
        %239 = ascscalar.bind_local_ptr %227 : memref<1xbf16, 1>
        %240 = arith.index_cast %9 : i64 to index
        %241 = cannir.extract_layout %arg19 : !cannir.tensor<<gm, bf16>, <<(8, -1)>, <(-1, 1)>>, <<(8, -1)>, <(-1, 1)>>, identity>
        %242 = cannir.extract_stride %241 : <<(8, -1)>, <(-1, 1)>> -> <(-1, 1)>
        %243 = cannir.extract_dim %242[0] : !cannir.stride<(-1, 1)>
        %244 = arith.index_cast %243 : i64 to index
        %245 = arith.muli %240, %c2 : index
        %246 = arith.muli %244, %c2 : index
        %247 = arith.cmpi ugt, %9, %c0_i64 : i64
        scf.if %247 {
          %262 = arith.index_cast %245 : index to i64
          %263 = arith.index_cast %246 : index to i64
          ascvec.get_buf mte3, %229, 0 {cannir.buffer.generated_sync}
          %264 = arith.trunci %262 : i64 to i32
          ascvec.copy_ub2gm %238[%c0], %239[%c0], %c1_i32, %264, %263, %264 l2_cache_ctl %c0_i32 : memref<1xbf16>, memref<1xbf16, 1>
          ascvec.release_buf mte3, %229, 0 {cannir.buffer.generated_sync}
        }
        %248 = cannir.extract_physical_layout %arg12 : !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
        %249 = cannir.extract_stride %248 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
        %250 = cannir.extract_dim %249[0] : !cannir.stride<(-1, 1)>
        %251 = arith.muli %arg25, %250 : i64
        %252 = arith.muli %251, %c16_i64 : i64
        %253 = ascscalar.memref_address %alloc_7 : memref<1xbf16>
        %254 = arith.divui %252, %c8_i64 : i64
        %255 = arith.addi %253, %254 : i64
        %256 = ascscalar.bind_gm_ptr %255 : memref<1xbf16>
        %257 = cannir.extract_layout %arg12 : !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
        %258 = cannir.extract_stride %257 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
        %259 = cannir.extract_dim %258[0] : !cannir.stride<(-1, 1)>
        %260 = arith.index_cast %259 : i64 to index
        %261 = arith.muli %260, %c2 : index
        scf.if %247 {
          %262 = arith.index_cast %245 : index to i64
          %263 = arith.index_cast %261 : index to i64
          ascvec.get_buf mte3, %229, 0 {cannir.buffer.generated_sync}
          %264 = arith.trunci %262 : i64 to i32
          ascvec.copy_ub2gm %256[%c0], %239[%c0], %c1_i32, %264, %263, %264 l2_cache_ctl %c0_i32 : memref<1xbf16>, memref<1xbf16, 1>
          ascvec.release_buf mte3, %229, 0 {cannir.buffer.generated_sync}
        }
        scf.yield %70 : i64
      }
      scf.yield %62 : i64
    } else {
      scf.yield %c0_i64 : i64
    }
    %24 = arith.addi %5, %20 : i64
    %25 = arith.muli %1, %c2_i64 : i64
    %26 = scf.for %arg25 = %24 to %c8_i64 step %25 iter_args(%arg26 = %23) -> (i64)  : i64 {
      %42 = arith.cmpi eq, %arg26, %c0_i64 : i64
      %43 = ascscalar.memref_address %alloc_21 : memref<2048xbf16, 1>
      %44 = ascscalar.memref_address %alloc_22 : memref<2048xbf16, 1>
      %45 = arith.select %42, %43, %44 : i64
      %46 = ascscalar.bind_local_ptr %45 : memref<2048xbf16, 1>
      %47 = arith.select %42, %c3_i64, %c4_i64 : i64
      %48 = arith.addi %arg26, %c1_i64 overflow<nsw> : i64
      %49 = arith.remui %48, %c2_i64 : i64
      ascvec.get_buf v, %47, 0 {cannir.buffer.generated_sync}
      ascvec.vf(%46 : memref<2048xbf16, 1>) {
      ^bb0(%arg27: memref<2048xbf16, 1>):
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
        %c0_49 = arith.constant 0 : index
        %cst_50 = arith.constant 0.000000e+00 : bf16
        %c128_i32_51 = arith.constant 128 : i32
        %c0_52 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        scf.for %arg28 = %c0_52 to %c1 step %c1 {
          %mask, %remaining_out = ascvec.update_mask %c128_i32_51 elem_bits 16 : !ascvec.mask
          %68 = ascvec.duplicate_scalar %cst_50, %mask : bf16 -> <bf16>
          ascvec.storealign %arg27[%c0_49], %68, %mask : memref<2048xbf16, 1>, index, <bf16>
          ascvec.storealign %arg27[%c128], %68, %mask : memref<2048xbf16, 1>, index, <bf16>
          ascvec.storealign %arg27[%c256], %68, %mask : memref<2048xbf16, 1>, index, <bf16>
          ascvec.storealign %arg27[%c384], %68, %mask : memref<2048xbf16, 1>, index, <bf16>
          ascvec.storealign %arg27[%c512], %68, %mask : memref<2048xbf16, 1>, index, <bf16>
          ascvec.storealign %arg27[%c640], %68, %mask : memref<2048xbf16, 1>, index, <bf16>
          ascvec.storealign %arg27[%c768], %68, %mask : memref<2048xbf16, 1>, index, <bf16>
          ascvec.storealign %arg27[%c896], %68, %mask : memref<2048xbf16, 1>, index, <bf16>
          ascvec.storealign %arg27[%c1024], %68, %mask : memref<2048xbf16, 1>, index, <bf16>
          ascvec.storealign %arg27[%c1152], %68, %mask : memref<2048xbf16, 1>, index, <bf16>
          ascvec.storealign %arg27[%c1280], %68, %mask : memref<2048xbf16, 1>, index, <bf16>
          ascvec.storealign %arg27[%c1408], %68, %mask : memref<2048xbf16, 1>, index, <bf16>
          ascvec.storealign %arg27[%c1536], %68, %mask : memref<2048xbf16, 1>, index, <bf16>
          ascvec.storealign %arg27[%c1664], %68, %mask : memref<2048xbf16, 1>, index, <bf16>
          ascvec.storealign %arg27[%c1792], %68, %mask : memref<2048xbf16, 1>, index, <bf16>
          ascvec.storealign %arg27[%c1920], %68, %mask : memref<2048xbf16, 1>, index, <bf16>
        }
        ascvec.mem_bar vst_vld
      } {cannir.vf_mode = "simd"}
      ascvec.release_buf v, %47, 0 {cannir.buffer.generated_sync}
      %50 = cannir.extract_physical_layout %arg19 : !cannir.tensor<<gm, bf16>, <<(8, -1)>, <(-1, 1)>>, <<(8, -1)>, <(-1, 1)>>, identity>
      %51 = cannir.extract_stride %50 : <<(8, -1)>, <(-1, 1)>> -> <(-1, 1)>
      %52 = cannir.extract_dim %51[0] : !cannir.stride<(-1, 1)>
      %53 = arith.muli %arg25, %52 : i64
      %54 = arith.muli %53, %c16_i64 : i64
      %55 = ascscalar.memref_address %alloc_14 : memref<1xbf16>
      %56 = arith.divui %54, %c8_i64 : i64
      %57 = arith.addi %55, %56 : i64
      %58 = ascscalar.bind_gm_ptr %57 : memref<1xbf16>
      %59 = ascscalar.bind_local_ptr %45 : memref<1xbf16, 1>
      %60 = arith.index_cast %9 : i64 to index
      %61 = cannir.extract_layout %arg19 : !cannir.tensor<<gm, bf16>, <<(8, -1)>, <(-1, 1)>>, <<(8, -1)>, <(-1, 1)>>, identity>
      %62 = cannir.extract_stride %61 : <<(8, -1)>, <(-1, 1)>> -> <(-1, 1)>
      %63 = cannir.extract_dim %62[0] : !cannir.stride<(-1, 1)>
      %64 = arith.index_cast %63 : i64 to index
      %65 = arith.muli %60, %c2 : index
      %66 = arith.muli %64, %c2 : index
      %67 = arith.cmpi ugt, %9, %c0_i64 : i64
      scf.if %67 {
        %68 = arith.index_cast %65 : index to i64
        %69 = arith.index_cast %66 : index to i64
        ascvec.get_buf mte3, %47, 0 {cannir.buffer.generated_sync}
        %70 = arith.trunci %68 : i64 to i32
        ascvec.copy_ub2gm %58[%c0], %59[%c0], %c1_i32, %70, %69, %70 l2_cache_ctl %c0_i32 : memref<1xbf16>, memref<1xbf16, 1>
        ascvec.release_buf mte3, %47, 0 {cannir.buffer.generated_sync}
      }
      scf.yield %49 : i64
    }
    ascvec.sync_all
    ascvec.sync_block_arrive mte3, %c13_i64
    ascvec.sync_block_arrive mte3, %c14_i64 {mode = 0 : i32}
    ascvec.sync_block_wait s, %c14_i64 {mode = 0 : i32}
    ascvec.sync_block_wait s, %c15_i64
    %27 = arith.cmpi eq, %2, %c0_i64 : i64
    scf.if %27 {
      %42 = arith.subi %12, %c512_i64 : i64
      %43 = ascscalar.bitcast_memref %alloc_28 byte_offset = %c0_i64 : memref<512xf32, 1> -> memref<?xf32, 1>
      "ascvec.copy_gm2ub_plan"(%43, %alloc_1, %c0_i32, %c0_i32, %c0_i64, %c0_i64, %c1_i64, %c2048_i64, %c0_i64, %c2048_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 2048 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 6912 : i64, ub_capacity = 2048 : i64}> : (memref<?xf32, 1>, memref<512xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
      ascvec.get_buf mte2, %c5_i64, 0 {cannir.buffer.generated_sync}
      "ascvec.copy_gm2ub_plan"(%alloc_32, %alloc_4, %c0_i32, %c0_i32, %c0_i64, %c0_i64, %c1_i64, %c4_i64, %c0_i64, %c4_i64, %c0_i64, %c0_i64, %c1_i64, %c4_i64, %c32_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 4 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 15168 : i64, ub_capacity = 32 : i64}> : (memref<8xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
      ascvec.release_buf mte2, %c5_i64, 0 {cannir.buffer.generated_sync}
      %44 = scf.for %arg25 = %0 to %5 step %1 iter_args(%arg26 = %c0_i64) -> (i64)  : i64 {
        %45 = arith.cmpi eq, %arg26, %c0_i64 : i64
        %46 = ascscalar.memref_address %alloc_24 : memref<576xbf16, 1>
        %47 = ascscalar.memref_address %alloc_25 : memref<576xbf16, 1>
        %48 = arith.select %45, %46, %47 : i64
        %49 = arith.cmpi ne, %arg26, %c0_i64 : i64
        %50 = arith.extui %49 : i1 to i64
        %51 = arith.addi %arg26, %c1_i64 overflow<nsw> : i64
        %52 = arith.remui %51, %c2_i64 : i64
        %53 = arith.minsi %arg25, %c8_i64 : i64
        %54 = arith.subi %c8_i64, %53 : i64
        %55 = arith.minsi %54, %c1_i64 : i64
        %56 = arith.cmpi sle, %arg25, %c8_i64 : i64
        %57 = arith.select %56, %55, %c0_i64 : i64
        %58 = cannir.extract_physical_layout %arg15 : !cannir.tensor<<gm, bf16>, <<(8, -1)>, <(-1, 1)>>, <<(8, -1)>, <(-1, 1)>>, identity>
        %59 = cannir.extract_stride %58 : <<(8, -1)>, <(-1, 1)>> -> <(-1, 1)>
        %60 = cannir.extract_dim %59[0] : !cannir.stride<(-1, 1)>
        %61 = arith.muli %60, %c2_i64 : i64
        %62 = cannir.extract_layout %arg15 : !cannir.tensor<<gm, bf16>, <<(8, -1)>, <(-1, 1)>>, <<(8, -1)>, <(-1, 1)>>, identity>
        %63 = cannir.extract_shape %62 : <<(8, -1)>, <(-1, 1)>> -> <(8, -1)>
        %64 = cannir.extract_dim %63[1] : !cannir.shape<(8, -1)>
        %65 = arith.maxsi %12, %c1_i64 : i64
        %66 = arith.divsi %64, %65 : i64
        %67 = arith.minsi %66, %c0_i64 : i64
        %68 = arith.muli %67, %12 : i64
        %69 = arith.subi %64, %68 : i64
        %70 = arith.minsi %12, %69 : i64
        %71 = arith.cmpi sge, %66, %c0_i64 : i64
        %72 = arith.select %71, %70, %c0_i64 : i64
        %73 = arith.muli %arg25, %60 : i64
        %74 = arith.muli %73, %c2_i64 : i64
        %75 = arith.muli %72, %c2_i64 : i64
        %76 = arith.cmpi eq, %75, %c0_i64 : i64
        %77 = arith.select %76, %c0_i64, %57 : i64
        %78 = arith.select %76, %c2_i64, %75 : i64
        %79 = ascscalar.bind_local_ptr %48 : memref<1xbf16, 1>
        ascvec.get_buf mte2, %50, 0 {cannir.buffer.generated_sync}
        "ascvec.copy_gm2ub_plan"(%79, %alloc_10, %c0_i32, %c0_i32, %c0_i64, %74, %77, %78, %61, %c1152_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
        ascvec.release_buf mte2, %50, 0 {cannir.buffer.generated_sync}
        %80 = ascscalar.bind_local_ptr %48 : memref<576xbf16, 1>
        %81 = ascscalar.memref_address %alloc_26 : memref<576xf32, 1>
        %82 = ascscalar.memref_address %alloc_27 : memref<576xf32, 1>
        %83 = arith.select %45, %81, %82 : i64
        %84 = ascscalar.bind_local_ptr %83 : memref<576xf32, 1>
        %85 = arith.trunci %12 : i64 to i32
        ascvec.get_buf v, %50, 0 {cannir.buffer.generated_sync}
        ascvec.vf(%80, %84, %85 : memref<576xbf16, 1>, memref<576xf32, 1>, i32) {
        ^bb0(%arg27: memref<576xbf16, 1>, %arg28: memref<576xf32, 1>, %arg29: i32):
          %c64_i32_49 = arith.constant 64 : i32
          %c0_i32_50 = arith.constant 0 : i32
          scf.for %arg30 = %c0_i32_50 to %arg29 step %c64_i32_49  : i32 {
            %93 = arith.index_cast %arg30 : i32 to index
            %94 = ascvec.loadalign_unpack %arg27[%93] {mode = "b16_to_b32"} : memref<576xbf16, 1>, index -> <bf16>
            %95 = ascvec.create_mask : !ascvec.mask
            %96 = ascvec.cast %94, %95 {rounding = 6 : i32} : <bf16> -> <f32>
            ascvec.storealign %arg28[%93], %96, %95 : memref<576xf32, 1>, index, <f32>
          }
          ascvec.mem_bar vst_vld
        } {cannir.vf_mode = "simd"}
        ascvec.release_buf v, %50, 0 {cannir.buffer.generated_sync}
        ascvec.vf(%84, %arg20, %alloc_31 : memref<576xf32, 1>, f32, memref<64xf32, 1>) {
        ^bb0(%arg27: memref<576xf32, 1>, %arg28: f32, %arg29: memref<64xf32, 1>):
          %c448 = arith.constant 448 : index
          %c384 = arith.constant 384 : index
          %c320 = arith.constant 320 : index
          %c256 = arith.constant 256 : index
          %c192 = arith.constant 192 : index
          %c128 = arith.constant 128 : index
          %c64 = arith.constant 64 : index
          %cst_49 = arith.constant 0.001953125 : f32
          %c0_50 = arith.constant 0 : index
          %cst_51 = arith.constant 0.000000e+00 : f32
          %c1_i32_52 = arith.constant 1 : i32
          %c0_53 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          scf.for %arg30 = %c0_53 to %c1 step %c1 {
            %mask, %remaining_out = ascvec.update_mask %c1_i32_52 : !ascvec.mask
            %93 = ascvec.duplicate_scalar %cst_51, %mask : f32 -> <f32>
            %94 = ascvec.loadalign %arg27[%c0_50] : memref<576xf32, 1>, index -> <f32>
            %95 = ascvec.create_mask : !ascvec.mask
            %96 = ascvec.mul %94, %94, %95 : <f32>
            %97 = ascvec.reduce_sum %96, %95 : <f32>
            %98 = ascvec.add %93, %97, %mask : <f32>
            %99 = ascvec.loadalign %arg27[%c64] : memref<576xf32, 1>, index -> <f32>
            %100 = ascvec.mul %99, %99, %95 : <f32>
            %101 = ascvec.reduce_sum %100, %95 : <f32>
            %102 = ascvec.add %98, %101, %mask : <f32>
            %103 = ascvec.loadalign %arg27[%c128] : memref<576xf32, 1>, index -> <f32>
            %104 = ascvec.mul %103, %103, %95 : <f32>
            %105 = ascvec.reduce_sum %104, %95 : <f32>
            %106 = ascvec.add %102, %105, %mask : <f32>
            %107 = ascvec.loadalign %arg27[%c192] : memref<576xf32, 1>, index -> <f32>
            %108 = ascvec.mul %107, %107, %95 : <f32>
            %109 = ascvec.reduce_sum %108, %95 : <f32>
            %110 = ascvec.add %106, %109, %mask : <f32>
            %111 = ascvec.loadalign %arg27[%c256] : memref<576xf32, 1>, index -> <f32>
            %112 = ascvec.mul %111, %111, %95 : <f32>
            %113 = ascvec.reduce_sum %112, %95 : <f32>
            %114 = ascvec.add %110, %113, %mask : <f32>
            %115 = ascvec.loadalign %arg27[%c320] : memref<576xf32, 1>, index -> <f32>
            %116 = ascvec.mul %115, %115, %95 : <f32>
            %117 = ascvec.reduce_sum %116, %95 : <f32>
            %118 = ascvec.add %114, %117, %mask : <f32>
            %119 = ascvec.loadalign %arg27[%c384] : memref<576xf32, 1>, index -> <f32>
            %120 = ascvec.mul %119, %119, %95 : <f32>
            %121 = ascvec.reduce_sum %120, %95 : <f32>
            %122 = ascvec.add %118, %121, %mask : <f32>
            %123 = ascvec.loadalign %arg27[%c448] : memref<576xf32, 1>, index -> <f32>
            %124 = ascvec.mul %123, %123, %95 : <f32>
            %125 = ascvec.reduce_sum %124, %95 : <f32>
            %126 = ascvec.add %122, %125, %mask : <f32>
            %127 = ascvec.mul_scalar %126, %cst_49, %mask : <f32>, f32
            %128 = ascvec.add_scalar %127, %arg28, %mask : <f32>, f32
            %129 = ascvec.sqrt %128, %mask : <f32>
            ascvec.storealign_1st %arg29[%c0_50], %129 : memref<64xf32, 1>, index, <f32>
          }
          ascvec.mem_bar vst_vld
        } {cannir.vf_mode = "simd"}
        %86 = ascscalar.memref_address %alloc_29 : memref<576xui8, 1>
        %87 = ascscalar.memref_address %alloc_30 : memref<576xui8, 1>
        %88 = arith.select %45, %86, %87 : i64
        %89 = ascscalar.bind_local_ptr %88 : memref<576xui8, 1>
        %90 = arith.select %45, %c3_i64, %c4_i64 : i64
        %91 = arith.trunci %42 : i64 to i32
        ascvec.get_buf v, %c5_i64, 0 {cannir.buffer.generated_sync}
        ascvec.get_buf v, %90, 0 {cannir.buffer.generated_sync}
        ascvec.vf(%alloc_31, %alloc_32, %84, %alloc_28, %arg22, %89, %91 : memref<64xf32, 1>, memref<8xf32, 1>, memref<576xf32, 1>, memref<512xf32, 1>, f32, memref<576xui8, 1>, i32) {
        ^bb0(%arg27: memref<64xf32, 1>, %arg28: memref<8xf32, 1>, %arg29: memref<576xf32, 1>, %arg30: memref<512xf32, 1>, %arg31: f32, %arg32: memref<576xui8, 1>, %arg33: i32):
          %c512 = arith.constant 512 : index
          %c448 = arith.constant 448 : index
          %c384 = arith.constant 384 : index
          %c320 = arith.constant 320 : index
          %c256 = arith.constant 256 : index
          %c192 = arith.constant 192 : index
          %c128 = arith.constant 128 : index
          %c64 = arith.constant 64 : index
          %cst_49 = arith.constant 3.276800e+04 : f32
          %cst_50 = arith.constant -3.276800e+04 : f32
          %c0_51 = arith.constant 0 : index
          %c0_52 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          scf.for %arg34 = %c0_52 to %c1 step %c1 {
            %93 = ascvec.loadalign_brc %arg27[%c0_51] : memref<64xf32, 1>, index -> <f32>
            %94 = ascvec.loadalign_brc %arg28[%c0_51] : memref<8xf32, 1>, index -> <f32>
            %95 = ascvec.loadalign %arg29[%c0_51] : memref<576xf32, 1>, index -> <f32>
            %96 = ascvec.create_mask : !ascvec.mask
            %97 = ascvec.div %95, %93, %96 {precision_mode = true} : <f32>
            %98 = ascvec.loadalign %arg30[%c0_51] : memref<512xf32, 1>, index -> <f32>
            %99 = ascvec.mul %97, %98, %96 : <f32>
            %100 = ascvec.mul_scalar %99, %arg31, %96 : <f32>, f32
            %101 = ascvec.mul %100, %94, %96 : <f32>
            %102 = ascvec.max_scalar %101, %cst_50, %96 : <f32>, f32
            %103 = ascvec.min_scalar %102, %cst_49, %96 : <f32>, f32
            %104 = ascvec.cast %103, %96 {reg_layout = 0 : i32, rounding = 1 : i32} : <f32> -> <!ascdtype.hifloat8>
            %105 = ascvec.reinterpret %104 : <!ascdtype.hifloat8> -> <f8E4M3FN>
            ascvec.storealign_pack %arg32[%c0_51], %105, %96 {mode = "b32_to_b8"} : memref<576xui8, 1>, index, <f8E4M3FN>
            %106 = ascvec.loadalign %arg29[%c64] : memref<576xf32, 1>, index -> <f32>
            %107 = ascvec.div %106, %93, %96 {precision_mode = true} : <f32>
            %108 = ascvec.loadalign %arg30[%c64] : memref<512xf32, 1>, index -> <f32>
            %109 = ascvec.mul %107, %108, %96 : <f32>
            %110 = ascvec.mul_scalar %109, %arg31, %96 : <f32>, f32
            %111 = ascvec.mul %110, %94, %96 : <f32>
            %112 = ascvec.max_scalar %111, %cst_50, %96 : <f32>, f32
            %113 = ascvec.min_scalar %112, %cst_49, %96 : <f32>, f32
            %114 = ascvec.cast %113, %96 {reg_layout = 0 : i32, rounding = 1 : i32} : <f32> -> <!ascdtype.hifloat8>
            %115 = ascvec.reinterpret %114 : <!ascdtype.hifloat8> -> <f8E4M3FN>
            ascvec.storealign_pack %arg32[%c64], %115, %96 {mode = "b32_to_b8"} : memref<576xui8, 1>, index, <f8E4M3FN>
            %116 = ascvec.loadalign %arg29[%c128] : memref<576xf32, 1>, index -> <f32>
            %117 = ascvec.div %116, %93, %96 {precision_mode = true} : <f32>
            %118 = ascvec.loadalign %arg30[%c128] : memref<512xf32, 1>, index -> <f32>
            %119 = ascvec.mul %117, %118, %96 : <f32>
            %120 = ascvec.mul_scalar %119, %arg31, %96 : <f32>, f32
            %121 = ascvec.mul %120, %94, %96 : <f32>
            %122 = ascvec.max_scalar %121, %cst_50, %96 : <f32>, f32
            %123 = ascvec.min_scalar %122, %cst_49, %96 : <f32>, f32
            %124 = ascvec.cast %123, %96 {reg_layout = 0 : i32, rounding = 1 : i32} : <f32> -> <!ascdtype.hifloat8>
            %125 = ascvec.reinterpret %124 : <!ascdtype.hifloat8> -> <f8E4M3FN>
            ascvec.storealign_pack %arg32[%c128], %125, %96 {mode = "b32_to_b8"} : memref<576xui8, 1>, index, <f8E4M3FN>
            %126 = ascvec.loadalign %arg29[%c192] : memref<576xf32, 1>, index -> <f32>
            %127 = ascvec.div %126, %93, %96 {precision_mode = true} : <f32>
            %128 = ascvec.loadalign %arg30[%c192] : memref<512xf32, 1>, index -> <f32>
            %129 = ascvec.mul %127, %128, %96 : <f32>
            %130 = ascvec.mul_scalar %129, %arg31, %96 : <f32>, f32
            %131 = ascvec.mul %130, %94, %96 : <f32>
            %132 = ascvec.max_scalar %131, %cst_50, %96 : <f32>, f32
            %133 = ascvec.min_scalar %132, %cst_49, %96 : <f32>, f32
            %134 = ascvec.cast %133, %96 {reg_layout = 0 : i32, rounding = 1 : i32} : <f32> -> <!ascdtype.hifloat8>
            %135 = ascvec.reinterpret %134 : <!ascdtype.hifloat8> -> <f8E4M3FN>
            ascvec.storealign_pack %arg32[%c192], %135, %96 {mode = "b32_to_b8"} : memref<576xui8, 1>, index, <f8E4M3FN>
            %136 = ascvec.loadalign %arg29[%c256] : memref<576xf32, 1>, index -> <f32>
            %137 = ascvec.div %136, %93, %96 {precision_mode = true} : <f32>
            %138 = ascvec.loadalign %arg30[%c256] : memref<512xf32, 1>, index -> <f32>
            %139 = ascvec.mul %137, %138, %96 : <f32>
            %140 = ascvec.mul_scalar %139, %arg31, %96 : <f32>, f32
            %141 = ascvec.mul %140, %94, %96 : <f32>
            %142 = ascvec.max_scalar %141, %cst_50, %96 : <f32>, f32
            %143 = ascvec.min_scalar %142, %cst_49, %96 : <f32>, f32
            %144 = ascvec.cast %143, %96 {reg_layout = 0 : i32, rounding = 1 : i32} : <f32> -> <!ascdtype.hifloat8>
            %145 = ascvec.reinterpret %144 : <!ascdtype.hifloat8> -> <f8E4M3FN>
            ascvec.storealign_pack %arg32[%c256], %145, %96 {mode = "b32_to_b8"} : memref<576xui8, 1>, index, <f8E4M3FN>
            %146 = ascvec.loadalign %arg29[%c320] : memref<576xf32, 1>, index -> <f32>
            %147 = ascvec.div %146, %93, %96 {precision_mode = true} : <f32>
            %148 = ascvec.loadalign %arg30[%c320] : memref<512xf32, 1>, index -> <f32>
            %149 = ascvec.mul %147, %148, %96 : <f32>
            %150 = ascvec.mul_scalar %149, %arg31, %96 : <f32>, f32
            %151 = ascvec.mul %150, %94, %96 : <f32>
            %152 = ascvec.max_scalar %151, %cst_50, %96 : <f32>, f32
            %153 = ascvec.min_scalar %152, %cst_49, %96 : <f32>, f32
            %154 = ascvec.cast %153, %96 {reg_layout = 0 : i32, rounding = 1 : i32} : <f32> -> <!ascdtype.hifloat8>
            %155 = ascvec.reinterpret %154 : <!ascdtype.hifloat8> -> <f8E4M3FN>
            ascvec.storealign_pack %arg32[%c320], %155, %96 {mode = "b32_to_b8"} : memref<576xui8, 1>, index, <f8E4M3FN>
            %156 = ascvec.loadalign %arg29[%c384] : memref<576xf32, 1>, index -> <f32>
            %157 = ascvec.div %156, %93, %96 {precision_mode = true} : <f32>
            %158 = ascvec.loadalign %arg30[%c384] : memref<512xf32, 1>, index -> <f32>
            %159 = ascvec.mul %157, %158, %96 : <f32>
            %160 = ascvec.mul_scalar %159, %arg31, %96 : <f32>, f32
            %161 = ascvec.mul %160, %94, %96 : <f32>
            %162 = ascvec.max_scalar %161, %cst_50, %96 : <f32>, f32
            %163 = ascvec.min_scalar %162, %cst_49, %96 : <f32>, f32
            %164 = ascvec.cast %163, %96 {reg_layout = 0 : i32, rounding = 1 : i32} : <f32> -> <!ascdtype.hifloat8>
            %165 = ascvec.reinterpret %164 : <!ascdtype.hifloat8> -> <f8E4M3FN>
            ascvec.storealign_pack %arg32[%c384], %165, %96 {mode = "b32_to_b8"} : memref<576xui8, 1>, index, <f8E4M3FN>
            %166 = ascvec.loadalign %arg29[%c448] : memref<576xf32, 1>, index -> <f32>
            %167 = ascvec.div %166, %93, %96 {precision_mode = true} : <f32>
            %168 = ascvec.loadalign %arg30[%c448] : memref<512xf32, 1>, index -> <f32>
            %169 = ascvec.mul %167, %168, %96 : <f32>
            %170 = ascvec.mul_scalar %169, %arg31, %96 : <f32>, f32
            %171 = ascvec.mul %170, %94, %96 : <f32>
            %172 = ascvec.max_scalar %171, %cst_50, %96 : <f32>, f32
            %173 = ascvec.min_scalar %172, %cst_49, %96 : <f32>, f32
            %174 = ascvec.cast %173, %96 {reg_layout = 0 : i32, rounding = 1 : i32} : <f32> -> <!ascdtype.hifloat8>
            %175 = ascvec.reinterpret %174 : <!ascdtype.hifloat8> -> <f8E4M3FN>
            ascvec.storealign_pack %arg32[%c448], %175, %96 {mode = "b32_to_b8"} : memref<576xui8, 1>, index, <f8E4M3FN>
            %mask, %remaining_out = ascvec.update_mask %arg33 : !ascvec.mask
            %176 = ascvec.loadalign %arg29[%c512] : memref<576xf32, 1>, index -> <f32>
            %177 = ascvec.mul %176, %94, %96 : <f32>
            %178 = ascvec.max_scalar %177, %cst_50, %96 : <f32>, f32
            %179 = ascvec.min_scalar %178, %cst_49, %96 : <f32>, f32
            %180 = ascvec.cast %179, %96 {reg_layout = 0 : i32, rounding = 1 : i32} : <f32> -> <!ascdtype.hifloat8>
            %181 = ascvec.reinterpret %180 : <!ascdtype.hifloat8> -> <f8E4M3FN>
            ascvec.storealign_pack %arg32[%c512], %181, %mask {mode = "b32_to_b8"} : memref<576xui8, 1>, index, <f8E4M3FN>
          }
          ascvec.mem_bar vst_vld
        } {cannir.vf_mode = "simd"}
        ascvec.release_buf v, %90, 0 {cannir.buffer.generated_sync}
        ascvec.release_buf v, %c5_i64, 0 {cannir.buffer.generated_sync}
        %92 = arith.cmpi eq, %arg23, %c0_i64 : i64
        scf.if %92 {
          %93 = arith.index_cast %arg25 : i64 to index
          %94 = memref.load %alloc_3[%93] : memref<1xi64>
          %95 = arith.cmpi sge, %94, %c0_i64 : i64
          scf.if %95 {
            %96 = cannir.extract_layout %arg7 : !cannir.tensor<<gm, ui8>, <<(-1, -1, 1, -1)>, <(-1, -1, -1, 1)>>, <<(-1, -1, 1, -1)>, <(-1, -1, -1, 1)>>, identity>
            %97 = cannir.extract_shape %96 : <<(-1, -1, 1, -1)>, <(-1, -1, -1, 1)>> -> <(-1, -1, 1, -1)>
            %98 = cannir.extract_dim %97[3] : !cannir.shape<(-1, -1, 1, -1)>
            %99 = arith.muli %94, %98 : i64
            %100 = arith.muli %99, %c8_i64 : i64
            %101 = ascscalar.memref_address %alloc_2 : memref<1xui8>
            %102 = arith.divui %100, %c8_i64 : i64
            %103 = arith.addi %101, %102 : i64
            %104 = ascscalar.bind_gm_ptr %103 : memref<1xui8>
            %105 = ascscalar.bind_local_ptr %88 : memref<1xui8, 1>
            %106 = arith.cmpi ugt, %98, %c0_i64 : i64
            scf.if %106 {
              ascvec.get_buf mte3, %90, 0 {cannir.buffer.generated_sync}
              %107 = arith.trunci %98 : i64 to i32
              ascvec.copy_ub2gm %104[%c0], %105[%c0], %c1_i32, %107, %98, %107 l2_cache_ctl %c0_i32 : memref<1xui8>, memref<1xui8, 1>
              ascvec.release_buf mte3, %90, 0 {cannir.buffer.generated_sync}
            }
          }
        } else {
          %93 = arith.index_cast %arg25 : i64 to index
          %94 = memref.load %alloc_3[%93] : memref<1xi64>
          %95 = arith.cmpi sge, %94, %c0_i64 : i64
          scf.if %95 {
            %96 = cannir.extract_layout %arg7 : !cannir.tensor<<gm, ui8>, <<(-1, -1, 1, -1)>, <(-1, -1, -1, 1)>>, <<(-1, -1, 1, -1)>, <(-1, -1, -1, 1)>>, identity>
            %97 = cannir.extract_shape %96 : <<(-1, -1, 1, -1)>, <(-1, -1, -1, 1)>> -> <(-1, -1, 1, -1)>
            %98 = cannir.extract_dim %97[3] : !cannir.shape<(-1, -1, 1, -1)>
            %99 = arith.remsi %98, %c32_i64 : i64
            %100 = arith.cmpi ne, %99, %c0_i64 : i64
            %101 = arith.cmpi slt, %99, %c0_i64 : i64
            %102 = arith.andi %100, %101 : i1
            %103 = arith.divsi %98, %c32_i64 : i64
            %104 = arith.extui %102 : i1 to i64
            %105 = arith.subi %103, %104 : i64
            %106 = cannir.extract_dim %97[1] : !cannir.shape<(-1, -1, 1, -1)>
            %107 = arith.remsi %94, %106 : i64
            %108 = arith.cmpi ne, %107, %c0_i64 : i64
            %109 = arith.cmpi slt, %107, %c0_i64 : i64
            %110 = arith.cmpi slt, %106, %c0_i64 : i64
            %111 = arith.xori %109, %110 : i1
            %112 = arith.andi %108, %111 : i1
            %113 = arith.divsi %94, %106 : i64
            %114 = arith.extui %112 : i1 to i64
            %115 = arith.subi %113, %114 : i64
            %116 = arith.select %112, %106, %c0_i64 : i64
            %117 = arith.addi %107, %116 : i64
            scf.for %arg27 = %c0_i64 to %105 step %c1_i64  : i64 {
              %118 = arith.muli %115, %105 : i64
              %119 = arith.addi %118, %arg27 : i64
              %120 = arith.muli %119, %106 : i64
              %121 = arith.addi %120, %117 : i64
              %122 = arith.muli %121, %c256_i64 : i64
              %123 = ascscalar.memref_address %alloc_2 : memref<1xui8>
              %124 = arith.divui %122, %c8_i64 : i64
              %125 = arith.addi %123, %124 : i64
              %126 = ascscalar.bind_gm_ptr %125 : memref<1xui8>
              %127 = arith.muli %arg27, %c256_i64 : i64
              %128 = arith.divui %127, %c8_i64 : i64
              %129 = arith.addi %88, %128 : i64
              %130 = ascscalar.bind_local_ptr %129 : memref<1xui8, 1>
              %131 = arith.minsi %arg27, %c18_i64 : i64
              %132 = arith.muli %131, %c32_i64 : i64
              %133 = arith.subi %c576_i64, %132 : i64
              %134 = arith.minsi %133, %c32_i64 : i64
              %135 = arith.cmpi sle, %arg27, %c18_i64 : i64
              %136 = arith.select %135, %134, %c0_i64 : i64
              %137 = arith.cmpi ugt, %136, %c0_i64 : i64
              scf.if %137 {
                ascvec.get_buf mte3, %90, 0 {cannir.buffer.generated_sync}
                %138 = arith.trunci %136 : i64 to i32
                ascvec.copy_ub2gm %126[%c0], %130[%c0], %c1_i32, %138, %c32_i64, %c576_i32 l2_cache_ctl %c0_i32 : memref<1xui8>, memref<1xui8, 1>
                ascvec.release_buf mte3, %90, 0 {cannir.buffer.generated_sync}
              }
            }
          }
        }
        scf.yield %52 : i64
      }
    }
    ascvec.sync_all
    ascvec.sync_block_arrive mte3, %c13_i64
    ascvec.sync_block_arrive mte3, %c14_i64 {mode = 0 : i32}
    ascvec.sync_block_wait s, %c14_i64 {mode = 0 : i32}
    ascvec.sync_block_wait s, %c15_i64
    %28 = cannir.extract_layout %arg10 : !cannir.tensor<<gm, ui8>, <<(8, -1, -1)>, <(-1, -1, 1)>>, <<(8, -1, -1)>, <(-1, -1, 1)>>, identity>
    %29 = cannir.extract_shape %28 : <<(8, -1, -1)>, <(-1, -1, 1)>> -> <(8, -1, -1)>
    %30 = cannir.extract_dim %29[1] : !cannir.shape<(8, -1, -1)>
    %31 = cannir.extract_dim %29[2] : !cannir.shape<(8, -1, -1)>
    %32 = arith.muli %30, %31 : i64
    %33 = cannir.extract_layout %arg17 : !cannir.tensor<<gm, f32>, <<(8, -1, -1)>, <(-1, -1, 1)>>, <<(8, -1, -1)>, <(-1, -1, 1)>>, identity>
    %34 = cannir.extract_shape %33 : <<(8, -1, -1)>, <(-1, -1, 1)>> -> <(8, -1, -1)>
    %35 = cannir.extract_dim %34[1] : !cannir.shape<(8, -1, -1)>
    %36 = cannir.extract_dim %34[2] : !cannir.shape<(8, -1, -1)>
    %37 = arith.muli %35, %36 : i64
    %38 = arith.cmpi ne, %12, %c512_i64 : i64
    %39 = arith.select %38, %22, %false : i1
    scf.if %39 {
      %42:2 = scf.for %arg25 = %20 to %5 step %25 iter_args(%arg26 = %c0_i64, %arg27 = %c0_i64) -> (i64, i64)  : i64 {
        %43:2 = scf.for %arg28 = %c0_i64 to %15 step %c1_i64 iter_args(%arg29 = %arg26, %arg30 = %arg27) -> (i64, i64)  : i64 {
          %44 = arith.muli %arg28, %18 : i64
          %45 = arith.addi %44, %16 : i64
          %46 = arith.cmpi eq, %arg29, %c0_i64 : i64
          %47 = ascscalar.memref_address %alloc_33 : memref<64xbf16, 1>
          %48 = ascscalar.memref_address %alloc_34 : memref<64xbf16, 1>
          %49 = arith.select %46, %47, %48 : i64
          %50 = arith.cmpi ne, %arg29, %c0_i64 : i64
          %51 = arith.extui %50 : i1 to i64
          %52 = arith.addi %arg29, %c1_i64 overflow<nsw> : i64
          %53 = arith.remui %52, %c2_i64 : i64
          %54 = arith.remsi %45, %c64_i64 : i64
          %55 = arith.cmpi ne, %54, %c0_i64 : i64
          %56 = arith.cmpi slt, %54, %c0_i64 : i64
          %57 = arith.andi %55, %56 : i1
          %58 = arith.divsi %45, %c64_i64 : i64
          %59 = arith.extui %57 : i1 to i64
          %60 = arith.subi %58, %59 : i64
          %61 = arith.minsi %arg25, %c8_i64 : i64
          %62 = arith.subi %c8_i64, %61 : i64
          %63 = arith.minsi %62, %c1_i64 : i64
          %64 = arith.cmpi sle, %arg25, %c8_i64 : i64
          %65 = arith.select %64, %63, %c0_i64 : i64
          %66 = cannir.extract_physical_layout %arg16 : !cannir.tensor<<gm, bf16>, <<(8, -1)>, <(-1, 1)>>, <<(8, -1)>, <(-1, 1)>>, identity>
          %67 = cannir.extract_stride %66 : <<(8, -1)>, <(-1, 1)>> -> <(-1, 1)>
          %68 = cannir.extract_dim %67[0] : !cannir.stride<(-1, 1)>
          %69 = arith.muli %68, %c2_i64 : i64
          %70 = cannir.extract_layout %arg16 : !cannir.tensor<<gm, bf16>, <<(8, -1)>, <(-1, 1)>>, <<(8, -1)>, <(-1, 1)>>, identity>
          %71 = cannir.extract_shape %70 : <<(8, -1)>, <(-1, 1)>> -> <(8, -1)>
          %72 = cannir.extract_dim %71[1] : !cannir.shape<(8, -1)>
          %73 = arith.divsi %72, %c64_i64 : i64
          %74 = arith.minsi %60, %73 : i64
          %75 = arith.muli %74, %c64_i64 : i64
          %76 = arith.subi %72, %75 : i64
          %77 = arith.minsi %76, %c64_i64 : i64
          %78 = arith.cmpi sle, %60, %73 : i64
          %79 = arith.select %78, %77, %c0_i64 : i64
          %80 = arith.muli %arg25, %68 : i64
          %81 = arith.muli %80, %c2_i64 : i64
          %82 = arith.muli %60, %c128_i64 : i64
          %83 = arith.addi %81, %82 : i64
          %84 = arith.muli %79, %c2_i64 : i64
          %85 = arith.cmpi eq, %84, %c0_i64 : i64
          %86 = arith.select %85, %c0_i64, %65 : i64
          %87 = arith.select %85, %c2_i64, %84 : i64
          %88 = ascscalar.bind_local_ptr %49 : memref<1xbf16, 1>
          ascvec.get_buf mte2, %51, 0 {cannir.buffer.generated_sync}
          "ascvec.copy_gm2ub_plan"(%88, %alloc_11, %c0_i32, %c0_i32, %c0_i64, %83, %86, %87, %69, %c128_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
          ascvec.release_buf mte2, %51, 0 {cannir.buffer.generated_sync}
          %89 = ascscalar.bind_local_ptr %49 : memref<64xbf16, 1>
          ascvec.get_buf v, %51, 0 {cannir.buffer.generated_sync}
          ascvec.vf(%89, %alloc_35 : memref<64xbf16, 1>, memref<64xf32, 1>) {
          ^bb0(%arg31: memref<64xbf16, 1>, %arg32: memref<64xf32, 1>):
            %c0_49 = arith.constant 0 : index
            %c0_50 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            scf.for %arg33 = %c0_50 to %c1 step %c1 {
              %91 = ascvec.loadalign_unpack %arg31[%c0_49] {mode = "b16_to_b32"} : memref<64xbf16, 1>, index -> <bf16>
              %92 = ascvec.create_mask : !ascvec.mask
              %93 = ascvec.cast %91, %92 {rounding = 6 : i32} : <bf16> -> <f32>
              ascvec.storealign %arg32[%c0_49], %93, %92 : memref<64xf32, 1>, index, <f32>
            }
            ascvec.mem_bar vst_vld
          } {cannir.vf_mode = "simd"}
          ascvec.release_buf v, %51, 0 {cannir.buffer.generated_sync}
          %90 = scf.if %38 -> (i64) {
            %91 = arith.cmpi eq, %arg30, %c0_i64 : i64
            %92 = ascscalar.memref_address %alloc_36 : memref<64xf32, 1>
            %93 = ascscalar.memref_address %alloc_37 : memref<64xf32, 1>
            %94 = arith.select %91, %92, %93 : i64
            %95 = ascscalar.bind_local_ptr %94 : memref<64xf32, 1>
            %96 = arith.select %91, %c2_i64, %c3_i64 : i64
            %97 = arith.addi %arg30, %c1_i64 : i64
            %98 = arith.remui %97, %c2_i64 : i64
            ascvec.get_buf v, %96, 0 {cannir.buffer.generated_sync}
            ascvec.vf(%alloc_35, %95 : memref<64xf32, 1>, memref<64xf32, 1>) {
            ^bb0(%arg31: memref<64xf32, 1>, %arg32: memref<64xf32, 1>):
              %c0_49 = arith.constant 0 : index
              %c0_50 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              scf.for %arg33 = %c0_50 to %c1 step %c1 {
                %120 = ascvec.loadalign %arg31[%c0_49] : memref<64xf32, 1>, index -> <f32>
                %121 = ascvec.create_mask : !ascvec.mask
                ascvec.storealign %arg32[%c0_49], %120, %121 : memref<64xf32, 1>, index, <f32>
              }
              ascvec.mem_bar vst_vld
            } {cannir.vf_mode = "simd"}
            ascvec.release_buf v, %96, 0 {cannir.buffer.generated_sync}
            %99 = arith.muli %arg28, %12 : i64
            %100 = arith.addi %99, %c512_i64 : i64
            %101 = arith.remsi %100, %c64_i64 : i64
            %102 = arith.cmpi ne, %101, %c0_i64 : i64
            %103 = arith.cmpi slt, %101, %c0_i64 : i64
            %104 = arith.andi %102, %103 : i1
            %105 = arith.divsi %100, %c64_i64 : i64
            %106 = arith.extui %104 : i1 to i64
            %107 = arith.subi %105, %106 : i64
            %108 = arith.muli %arg25, %37 : i64
            %109 = arith.muli %107, %c64_i64 : i64
            %110 = arith.addi %108, %109 : i64
            %111 = arith.muli %110, %c32_i64 : i64
            %112 = ascscalar.memref_address %alloc_12 : memref<1xf32>
            %113 = arith.divui %111, %c8_i64 : i64
            %114 = arith.addi %112, %113 : i64
            %115 = ascscalar.bind_gm_ptr %114 : memref<1xf32>
            %116 = ascscalar.bind_local_ptr %94 : memref<1xf32, 1>
            %117 = arith.index_cast %37 : i64 to index
            %118 = arith.muli %117, %c4 : index
            %119 = arith.index_cast %118 : index to i64
            ascvec.get_buf mte3, %96, 0 {cannir.buffer.generated_sync}
            ascvec.copy_ub2gm %115[%c0], %116[%c0], %c1_i32, %c256_i32, %119, %c256_i32 l2_cache_ctl %c0_i32 : memref<1xf32>, memref<1xf32, 1>
            ascvec.release_buf mte3, %96, 0 {cannir.buffer.generated_sync}
            scf.yield %98 : i64
          } else {
            scf.yield %arg30 : i64
          }
          scf.yield %53, %90 : i64, i64
        }
        scf.yield %43#0, %43#1 : i64, i64
      }
    }
    ascvec.sync_all
    ascvec.sync_block_arrive mte3, %c13_i64
    ascvec.sync_block_arrive mte3, %c14_i64 {mode = 0 : i32}
    ascvec.sync_block_wait s, %c14_i64 {mode = 0 : i32}
    ascvec.sync_block_wait s, %c15_i64
    scf.if %27 {
      %42 = arith.muli %5, %15 : i64
      ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
      ascvec.vf(%alloc_40, %alloc_42 : memref<64xf32, 1>, memref<8xf32, 1>) {
      ^bb0(%arg25: memref<64xf32, 1>, %arg26: memref<8xf32, 1>):
        %c0_49 = arith.constant 0 : index
        %cst_50 = arith.constant 0.000000e+00 : f32
        %c0_51 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        scf.for %arg27 = %c0_51 to %c1 step %c1 {
          %50 = ascvec.create_mask : !ascvec.mask
          %51 = ascvec.duplicate_scalar %cst_50, %50 : f32 -> <f32>
          ascvec.storealign %arg25[%c0_49], %51, %50 : memref<64xf32, 1>, index, <f32>
          ascvec.storealign_1st %arg26[%c0_49], %51 : memref<8xf32, 1>, index, <f32>
        }
        ascvec.mem_bar vst_vld
      } {cannir.vf_mode = "simd"}
      ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
      %43:2 = scf.for %arg25 = %0 to %42 step %1 iter_args(%arg26 = %c0_i64, %arg27 = %c1_i64) -> (i64, i64)  : i64 {
        %50 = arith.remsi %arg25, %15 : i64
        %51 = arith.cmpi ne, %50, %c0_i64 : i64
        %52 = arith.cmpi slt, %50, %c0_i64 : i64
        %53 = arith.cmpi slt, %15, %c0_i64 : i64
        %54 = arith.xori %52, %53 : i1
        %55 = arith.andi %51, %54 : i1
        %56 = arith.divsi %arg25, %15 : i64
        %57 = arith.extui %55 : i1 to i64
        %58 = arith.subi %56, %57 : i64
        %59 = arith.select %55, %15, %c0_i64 : i64
        %60 = arith.addi %50, %59 : i64
        %61 = arith.cmpi eq, %arg26, %c0_i64 : i64
        %62 = ascscalar.memref_address %alloc_40 : memref<64xf32, 1>
        %63 = ascscalar.memref_address %alloc_41 : memref<64xf32, 1>
        %64 = arith.select %61, %62, %63 : i64
        %65 = ascscalar.bind_local_ptr %64 : memref<64xf32, 1>
        %66 = arith.addi %arg26, %c1_i64 overflow<nsw> : i64
        %67 = arith.remui %66, %c2_i64 : i64
        %68 = arith.cmpi eq, %arg27, %c0_i64 : i64
        %69 = arith.select %68, %62, %63 : i64
        %70 = ascscalar.bind_local_ptr %69 : memref<64xf32, 1>
        %71 = arith.addi %arg27, %c1_i64 overflow<nsw> : i64
        %72 = arith.remui %71, %c2_i64 : i64
        %73 = ascscalar.memref_address %alloc_38 : memref<576xf32, 1>
        %74 = ascscalar.memref_address %alloc_39 : memref<576xf32, 1>
        %75 = arith.select %61, %73, %74 : i64
        %76 = arith.cmpi ne, %arg26, %c0_i64 : i64
        %77 = arith.extui %76 : i1 to i64
        %78 = arith.minsi %58, %c8_i64 : i64
        %79 = arith.subi %c8_i64, %78 : i64
        %80 = arith.minsi %79, %c1_i64 : i64
        %81 = arith.cmpi sle, %58, %c8_i64 : i64
        %82 = arith.select %81, %80, %c0_i64 : i64
        %83 = arith.muli %37, %c4_i64 : i64
        %84 = arith.maxsi %12, %c1_i64 : i64
        %85 = arith.divsi %37, %84 : i64
        %86 = arith.minsi %60, %85 : i64
        %87 = arith.muli %86, %12 : i64
        %88 = arith.subi %37, %87 : i64
        %89 = arith.minsi %12, %88 : i64
        %90 = arith.cmpi sle, %60, %85 : i64
        %91 = arith.select %90, %89, %c0_i64 : i64
        %92 = arith.muli %58, %37 : i64
        %93 = arith.muli %92, %c4_i64 : i64
        %94 = arith.muli %60, %12 : i64
        %95 = arith.muli %94, %c4_i64 : i64
        %96 = arith.addi %93, %95 : i64
        %97 = arith.muli %91, %c4_i64 : i64
        %98 = arith.cmpi eq, %97, %c0_i64 : i64
        %99 = arith.select %98, %c0_i64, %82 : i64
        %100 = arith.select %98, %c4_i64, %97 : i64
        %101 = ascscalar.bind_local_ptr %75 : memref<1xf32, 1>
        ascvec.get_buf mte2, %77, 0 {cannir.buffer.generated_sync}
        "ascvec.copy_gm2ub_plan"(%101, %alloc_12, %c0_i32, %c0_i32, %c0_i64, %96, %99, %100, %83, %c2304_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = -1 : i64, ub_capacity = 4 : i64}> : (memref<1xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
        ascvec.release_buf mte2, %77, 0 {cannir.buffer.generated_sync}
        %102 = ascscalar.bind_local_ptr %75 : memref<576xf32, 1>
        %103 = arith.trunci %12 : i64 to i32
        %104 = arith.maxsi %103, %c0_i32 : i32
        %105 = arith.minsi %104, %c64_i32 : i32
        %106 = arith.subi %103, %c64_i32 : i32
        %107 = arith.maxsi %106, %c0_i32 : i32
        %108 = arith.minsi %107, %c64_i32 : i32
        %109 = arith.subi %103, %c128_i32 : i32
        %110 = arith.maxsi %109, %c0_i32 : i32
        %111 = arith.minsi %110, %c64_i32 : i32
        %112 = arith.subi %103, %c192_i32 : i32
        %113 = arith.maxsi %112, %c0_i32 : i32
        %114 = arith.minsi %113, %c64_i32 : i32
        %115 = arith.subi %103, %c256_i32 : i32
        %116 = arith.maxsi %115, %c0_i32 : i32
        %117 = arith.minsi %116, %c64_i32 : i32
        %118 = arith.subi %103, %c320_i32 : i32
        %119 = arith.maxsi %118, %c0_i32 : i32
        %120 = arith.minsi %119, %c64_i32 : i32
        %121 = arith.subi %103, %c384_i32 : i32
        %122 = arith.maxsi %121, %c0_i32 : i32
        %123 = arith.minsi %122, %c64_i32 : i32
        %124 = arith.subi %103, %c448_i32 : i32
        %125 = arith.maxsi %124, %c0_i32 : i32
        %126 = arith.minsi %125, %c64_i32 : i32
        %127 = arith.subi %103, %c512_i32 : i32
        %128 = arith.maxsi %127, %c0_i32 : i32
        %129 = arith.minsi %128, %c64_i32 : i32
        ascvec.get_buf v, %77, 0 {cannir.buffer.generated_sync}
        ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
        ascvec.vf(%105, %102, %108, %111, %114, %117, %120, %123, %126, %129, %alloc_42 : i32, memref<576xf32, 1>, i32, i32, i32, i32, i32, i32, i32, i32, memref<8xf32, 1>) {
        ^bb0(%arg28: i32, %arg29: memref<576xf32, 1>, %arg30: i32, %arg31: i32, %arg32: i32, %arg33: i32, %arg34: i32, %arg35: i32, %arg36: i32, %arg37: i32, %arg38: memref<8xf32, 1>):
          %c2_49 = arith.constant 2 : index
          %c512 = arith.constant 512 : index
          %c448 = arith.constant 448 : index
          %c384 = arith.constant 384 : index
          %c320 = arith.constant 320 : index
          %c256 = arith.constant 256 : index
          %c192 = arith.constant 192 : index
          %c128 = arith.constant 128 : index
          %c64 = arith.constant 64 : index
          %c0_50 = arith.constant 0 : index
          %cst_51 = arith.constant 0.000000e+00 : f32
          %c1_i32_52 = arith.constant 1 : i32
          %c0_53 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          scf.for %arg39 = %c0_53 to %c1 step %c1 {
            %mask, %remaining_out = ascvec.update_mask %c1_i32_52 : !ascvec.mask
            %130 = ascvec.duplicate_scalar %cst_51, %mask : f32 -> <f32>
            %mask_54, %remaining_out_55 = ascvec.update_mask %arg28 : !ascvec.mask
            %131 = ascvec.loadalign %arg29[%c0_50] : memref<576xf32, 1>, index -> <f32>
            %132 = ascvec.abs %131, %mask_54 : <f32>
            %133 = ascvec.reduce_max %132, %mask_54 : <f32>
            %134 = ascvec.max %130, %133, %mask : <f32>
            %mask_56, %remaining_out_57 = ascvec.update_mask %arg30 : !ascvec.mask
            %135 = ascvec.loadalign %arg29[%c64] : memref<576xf32, 1>, index -> <f32>
            %136 = ascvec.abs %135, %mask_56 : <f32>
            %137 = ascvec.reduce_max %136, %mask_56 : <f32>
            %138 = ascvec.max %134, %137, %mask : <f32>
            %mask_58, %remaining_out_59 = ascvec.update_mask %arg31 : !ascvec.mask
            %139 = ascvec.loadalign %arg29[%c128] : memref<576xf32, 1>, index -> <f32>
            %140 = ascvec.abs %139, %mask_58 : <f32>
            %141 = ascvec.reduce_max %140, %mask_58 : <f32>
            %142 = ascvec.max %138, %141, %mask : <f32>
            %mask_60, %remaining_out_61 = ascvec.update_mask %arg32 : !ascvec.mask
            %143 = ascvec.loadalign %arg29[%c192] : memref<576xf32, 1>, index -> <f32>
            %144 = ascvec.abs %143, %mask_60 : <f32>
            %145 = ascvec.reduce_max %144, %mask_60 : <f32>
            %146 = ascvec.max %142, %145, %mask : <f32>
            %mask_62, %remaining_out_63 = ascvec.update_mask %arg33 : !ascvec.mask
            %147 = ascvec.loadalign %arg29[%c256] : memref<576xf32, 1>, index -> <f32>
            %148 = ascvec.abs %147, %mask_62 : <f32>
            %149 = ascvec.reduce_max %148, %mask_62 : <f32>
            %150 = ascvec.max %146, %149, %mask : <f32>
            %mask_64, %remaining_out_65 = ascvec.update_mask %arg34 : !ascvec.mask
            %151 = ascvec.loadalign %arg29[%c320] : memref<576xf32, 1>, index -> <f32>
            %152 = ascvec.abs %151, %mask_64 : <f32>
            %153 = ascvec.reduce_max %152, %mask_64 : <f32>
            %154 = ascvec.max %150, %153, %mask : <f32>
            %mask_66, %remaining_out_67 = ascvec.update_mask %arg35 : !ascvec.mask
            %155 = ascvec.loadalign %arg29[%c384] : memref<576xf32, 1>, index -> <f32>
            %156 = ascvec.abs %155, %mask_66 : <f32>
            %157 = ascvec.reduce_max %156, %mask_66 : <f32>
            %158 = ascvec.max %154, %157, %mask : <f32>
            %mask_68, %remaining_out_69 = ascvec.update_mask %arg36 : !ascvec.mask
            %159 = ascvec.loadalign %arg29[%c448] : memref<576xf32, 1>, index -> <f32>
            %160 = ascvec.abs %159, %mask_68 : <f32>
            %161 = ascvec.reduce_max %160, %mask_68 : <f32>
            %162 = ascvec.max %158, %161, %mask : <f32>
            %mask_70, %remaining_out_71 = ascvec.update_mask %arg37 : !ascvec.mask
            %163 = ascvec.loadalign %arg29[%c512] : memref<576xf32, 1>, index -> <f32>
            %164 = ascvec.abs %163, %mask_70 : <f32>
            %165 = ascvec.reduce_max %164, %mask_70 : <f32>
            %166 = ascvec.max %162, %165, %mask : <f32>
            ascvec.storealign_1st %arg38[%c2_49], %166 : memref<8xf32, 1>, index, <f32>
          }
          ascvec.mem_bar vst_vld
        } {cannir.vf_mode = "simd"}
        ascvec.release_buf v, %77, 0 {cannir.buffer.generated_sync}
        ascvec.vf(%65, %alloc_42, %70 : memref<64xf32, 1>, memref<8xf32, 1>, memref<64xf32, 1>) {
        ^bb0(%arg28: memref<64xf32, 1>, %arg29: memref<8xf32, 1>, %arg30: memref<64xf32, 1>):
          %c2_49 = arith.constant 2 : index
          %c0_50 = arith.constant 0 : index
          %c1_i32_51 = arith.constant 1 : i32
          %c0_52 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          scf.for %arg31 = %c0_52 to %c1 step %c1 {
            %mask, %remaining_out = ascvec.update_mask %c1_i32_51 : !ascvec.mask
            %130 = ascvec.loadalign %arg28[%c0_50] : memref<64xf32, 1>, index -> <f32>
            %131 = ascvec.loadalign_brc %arg29[%c2_49] : memref<8xf32, 1>, index -> <f32>
            %132 = ascvec.max %130, %131, %mask : <f32>
            ascvec.storealign %arg30[%c0_50], %132, %mask : memref<64xf32, 1>, index, <f32>
            ascvec.storealign_1st %arg29[%c0_50], %132 : memref<8xf32, 1>, index, <f32>
          }
          ascvec.mem_bar vst_vld
        } {cannir.vf_mode = "simd"}
        ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
        scf.yield %67, %72 : i64, i64
      }
      %44 = arith.muli %0, %c32_i64 : i64
      %45 = ascscalar.memref_address %alloc_13 : memref<1xf32>
      %46 = arith.divui %44, %c8_i64 : i64
      %47 = arith.addi %45, %46 : i64
      %48 = ascscalar.bind_gm_ptr %47 : memref<1xf32>
      %49 = ascscalar.bitcast_memref %alloc_42 byte_offset = %c0_i64 : memref<8xf32, 1> -> memref<?xf32, 1>
      ascvec.get_buf mte3, %c2_i64, 0 {cannir.buffer.generated_sync}
      ascvec.copy_ub2gm %48[%c0], %49[%c0], %c1_i32, %c4_i32, %c0_i64, %c4_i32 l2_cache_ctl %c0_i32 : memref<1xf32>, memref<?xf32, 1>
      ascvec.release_buf mte3, %c2_i64, 0 {cannir.buffer.generated_sync}
    }
    ascvec.sync_all
    ascvec.sync_block_arrive mte3, %c13_i64
    ascvec.sync_block_arrive mte3, %c14_i64 {mode = 0 : i32}
    ascvec.sync_block_wait s, %c14_i64 {mode = 0 : i32}
    ascvec.sync_block_wait s, %c15_i64
    %40 = arith.muli %5, %15 : i64
    %41 = arith.cmpi slt, %20, %40 : i64
    scf.if %41 {
      %42 = arith.addi %0, %2 : i64
      %43 = arith.cmpi eq, %42, %c0_i64 : i64
      ascvec.vf(%alloc_48 : memref<8xf32, 1>) {
      ^bb0(%arg25: memref<8xf32, 1>):
        %c2_49 = arith.constant 2 : index
        %cst_50 = arith.constant 0.000000e+00 : f32
        %c1_i32_51 = arith.constant 1 : i32
        %c0_52 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        scf.for %arg26 = %c0_52 to %c1 step %c1 {
          %mask, %remaining_out = ascvec.update_mask %c1_i32_51 : !ascvec.mask
          %45 = ascvec.duplicate_scalar %cst_50, %mask : f32 -> <f32>
          ascvec.storealign_1st %arg25[%c2_49], %45 : memref<8xf32, 1>, index, <f32>
        }
        ascvec.mem_bar vst_vld
      } {cannir.vf_mode = "simd"}
      scf.for %arg25 = %c0_i64 to %1 step %c64_i64  : i64 {
        %45 = ascscalar.bitcast_memref %alloc_45 byte_offset = %c0_i64 : memref<64xf32, 1> -> memref<?xf32, 1>
        %46 = arith.remsi %arg25, %c64_i64 : i64
        %47 = arith.cmpi ne, %46, %c0_i64 : i64
        %48 = arith.cmpi slt, %46, %c0_i64 : i64
        %49 = arith.andi %47, %48 : i1
        %50 = arith.divsi %arg25, %c64_i64 : i64
        %51 = arith.extui %49 : i1 to i64
        %52 = arith.subi %50, %51 : i64
        %53 = cannir.extract_layout %arg18 : !cannir.tensor<<gm, f32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>
        %54 = cannir.extract_shape %53 : <<(-1)>, <(1)>> -> <(-1)>
        %55 = cannir.extract_dim %54[0] : !cannir.shape<(-1)>
        %56 = arith.divsi %55, %c64_i64 : i64
        %57 = arith.minsi %52, %56 : i64
        %58 = arith.muli %57, %c64_i64 : i64
        %59 = arith.subi %55, %58 : i64
        %60 = arith.minsi %59, %c64_i64 : i64
        %61 = arith.cmpi sle, %52, %56 : i64
        %62 = arith.select %61, %60, %c0_i64 : i64
        %63 = arith.muli %52, %c256_i64 : i64
        %64 = arith.muli %62, %c4_i64 : i64
        %65 = arith.cmpi eq, %64, %c0_i64 : i64
        %66 = arith.cmpi ne, %64, %c0_i64 : i64
        %67 = arith.extui %66 : i1 to i64
        %68 = arith.select %65, %c4_i64, %64 : i64
        ascvec.get_buf mte2, %c2_i64, 0 {cannir.buffer.generated_sync}
        "ascvec.copy_gm2ub_plan"(%45, %alloc_13, %c0_i32, %c0_i32, %c0_i64, %63, %67, %68, %c0_i64, %68, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 4608 : i64, ub_capacity = 256 : i64}> : (memref<?xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
        ascvec.release_buf mte2, %c2_i64, 0 {cannir.buffer.generated_sync}
        %69 = arith.trunci %1 : i64 to i32
        %70 = arith.trunci %arg25 : i64 to i32
        %71 = arith.subi %69, %70 : i32
        %72 = arith.minsi %71, %c64_i32 : i32
        ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
        ascvec.vf(%72, %alloc_48, %alloc_45 : i32, memref<8xf32, 1>, memref<64xf32, 1>) {
        ^bb0(%arg26: i32, %arg27: memref<8xf32, 1>, %arg28: memref<64xf32, 1>):
          %c2_49 = arith.constant 2 : index
          %c0_50 = arith.constant 0 : index
          %c1_i32_51 = arith.constant 1 : i32
          %c0_52 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          scf.for %arg29 = %c0_52 to %c1 step %c1 {
            %mask, %remaining_out = ascvec.update_mask %c1_i32_51 : !ascvec.mask
            %mask_53, %remaining_out_54 = ascvec.update_mask %arg26 : !ascvec.mask
            %73 = ascvec.loadalign_brc %arg27[%c2_49] : memref<8xf32, 1>, index -> <f32>
            %74 = ascvec.loadalign %arg28[%c0_50] : memref<64xf32, 1>, index -> <f32>
            %75 = ascvec.reduce_max %74, %mask_53 : <f32>
            %76 = ascvec.max %73, %75, %mask : <f32>
            ascvec.storealign_1st %arg27[%c2_49], %76 : memref<8xf32, 1>, index, <f32>
          }
          ascvec.mem_bar vst_vld
        } {cannir.vf_mode = "simd"}
        ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
      }
      ascvec.get_buf v, %c5_i64, 0 {cannir.buffer.generated_sync}
      ascvec.vf(%alloc_48 : memref<8xf32, 1>) {
      ^bb0(%arg25: memref<8xf32, 1>):
        %c1 = arith.constant 1 : index
        %c2_49 = arith.constant 2 : index
        %cst_50 = arith.constant 1.000000e+00 : f32
        %cst_51 = arith.constant 0.000000e+00 : f32
        %cst_52 = arith.constant 3.276800e+04 : f32
        %c0_53 = arith.constant 0 : index
        %c1_i32_54 = arith.constant 1 : i32
        %c0_55 = arith.constant 0 : index
        %c1_56 = arith.constant 1 : index
        scf.for %arg26 = %c0_55 to %c1_56 step %c1_56 {
          %mask, %remaining_out = ascvec.update_mask %c1_i32_54 : !ascvec.mask
          %45 = ascvec.loadalign_brc %arg25[%c2_49] : memref<8xf32, 1>, index -> <f32>
          %46 = ascvec.duplicate_scalar %cst_52, %mask : f32 -> <f32>
          %47 = ascvec.div %45, %46, %mask {precision_mode = true} : <f32>
          %48 = ascvec.duplicate_scalar %cst_51, %mask : f32 -> <f32>
          %49 = ascvec.cmp_eq %45, %48, %mask : <f32>, <f32> -> !ascvec.mask
          %50 = ascvec.duplicate_scalar %cst_50, %mask : f32 -> <f32>
          %51 = ascvec.select %50, %47, %49 : <f32>
          ascvec.storealign_1st %arg25[%c0_53], %47 : memref<8xf32, 1>, index, <f32>
          ascvec.storealign_1st %arg25[%c1], %51 : memref<8xf32, 1>, index, <f32>
          ascvec.storealign_1st %arg25[%c2_49], %45 : memref<8xf32, 1>, index, <f32>
        }
        ascvec.mem_bar vst_vld
      } {cannir.vf_mode = "simd"}
      ascvec.release_buf v, %c5_i64, 0 {cannir.buffer.generated_sync}
      scf.if %43 {
        %45 = ascscalar.bitcast_memref %alloc_48 byte_offset = %c0_i64 : memref<8xf32, 1> -> memref<?xf32, 1>
        ascvec.get_buf mte3, %c5_i64, 0 {cannir.buffer.generated_sync}
        ascvec.copy_ub2gm %alloc_6[%c0], %45[%c0], %c1_i32, %c4_i32, %c0_i64, %c4_i32 l2_cache_ctl %c0_i32 : memref<1xf32>, memref<?xf32, 1>
        ascvec.release_buf mte3, %c5_i64, 0 {cannir.buffer.generated_sync}
      }
      %44 = scf.for %arg25 = %20 to %40 step %25 iter_args(%arg26 = %c0_i64) -> (i64)  : i64 {
        %45 = arith.remsi %arg25, %15 : i64
        %46 = arith.cmpi ne, %45, %c0_i64 : i64
        %47 = arith.cmpi slt, %45, %c0_i64 : i64
        %48 = arith.cmpi slt, %15, %c0_i64 : i64
        %49 = arith.xori %47, %48 : i1
        %50 = arith.andi %46, %49 : i1
        %51 = arith.divsi %arg25, %15 : i64
        %52 = arith.extui %50 : i1 to i64
        %53 = arith.subi %51, %52 : i64
        %54 = arith.select %50, %15, %c0_i64 : i64
        %55 = arith.addi %45, %54 : i64
        %56 = arith.cmpi eq, %arg26, %c0_i64 : i64
        %57 = ascscalar.memref_address %alloc_43 : memref<576xf32, 1>
        %58 = ascscalar.memref_address %alloc_44 : memref<576xf32, 1>
        %59 = arith.select %56, %57, %58 : i64
        %60 = arith.cmpi ne, %arg26, %c0_i64 : i64
        %61 = arith.extui %60 : i1 to i64
        %62 = arith.addi %arg26, %c1_i64 overflow<nsw> : i64
        %63 = arith.remui %62, %c2_i64 : i64
        %64 = arith.minsi %53, %c8_i64 : i64
        %65 = arith.subi %c8_i64, %64 : i64
        %66 = arith.minsi %65, %c1_i64 : i64
        %67 = arith.cmpi sle, %53, %c8_i64 : i64
        %68 = arith.select %67, %66, %c0_i64 : i64
        %69 = arith.muli %37, %c4_i64 : i64
        %70 = arith.maxsi %12, %c1_i64 : i64
        %71 = arith.divsi %37, %70 : i64
        %72 = arith.minsi %55, %71 : i64
        %73 = arith.muli %72, %12 : i64
        %74 = arith.subi %37, %73 : i64
        %75 = arith.minsi %12, %74 : i64
        %76 = arith.cmpi sle, %55, %71 : i64
        %77 = arith.select %76, %75, %c0_i64 : i64
        %78 = arith.muli %53, %37 : i64
        %79 = arith.muli %78, %c4_i64 : i64
        %80 = arith.muli %55, %12 : i64
        %81 = arith.muli %80, %c4_i64 : i64
        %82 = arith.addi %79, %81 : i64
        %83 = arith.muli %77, %c4_i64 : i64
        %84 = arith.cmpi eq, %83, %c0_i64 : i64
        %85 = arith.select %84, %c0_i64, %68 : i64
        %86 = arith.select %84, %c4_i64, %83 : i64
        %87 = ascscalar.bind_local_ptr %59 : memref<1xf32, 1>
        ascvec.get_buf mte2, %61, 0 {cannir.buffer.generated_sync}
        "ascvec.copy_gm2ub_plan"(%87, %alloc_12, %c0_i32, %c0_i32, %c0_i64, %82, %85, %86, %69, %c2304_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = -1 : i64, ub_capacity = 4 : i64}> : (memref<1xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
        ascvec.release_buf mte2, %61, 0 {cannir.buffer.generated_sync}
        %88 = ascscalar.bind_local_ptr %59 : memref<576xf32, 1>
        %89 = ascscalar.memref_address %alloc_46 : memref<576xui8, 1>
        %90 = ascscalar.memref_address %alloc_47 : memref<576xui8, 1>
        %91 = arith.select %56, %89, %90 : i64
        %92 = ascscalar.bind_local_ptr %91 : memref<576xui8, 1>
        %93 = arith.select %56, %c3_i64, %c4_i64 : i64
        %94 = arith.trunci %12 : i64 to i32
        ascvec.get_buf v, %61, 0 {cannir.buffer.generated_sync}
        ascvec.get_buf v, %93, 0 {cannir.buffer.generated_sync}
        ascvec.vf(%alloc_48, %88, %92, %94 : memref<8xf32, 1>, memref<576xf32, 1>, memref<576xui8, 1>, i32) {
        ^bb0(%arg27: memref<8xf32, 1>, %arg28: memref<576xf32, 1>, %arg29: memref<576xui8, 1>, %arg30: i32):
          %c2_49 = arith.constant 2 : index
          %c64_i32_50 = arith.constant 64 : i32
          %c0_i32_51 = arith.constant 0 : i32
          %cst_52 = arith.constant 3.276800e+04 : f32
          %cst_53 = arith.constant -3.276800e+04 : f32
          %cst_54 = arith.constant 0.000000e+00 : f32
          %c1 = arith.constant 1 : index
          %104 = ascvec.loadalign_brc %arg27[%c1] : memref<8xf32, 1>, index -> <f32>
          %105 = ascvec.loadalign_brc %arg27[%c2_49] : memref<8xf32, 1>, index -> <f32>
          %106 = ascvec.create_mask : !ascvec.mask
          %107 = ascvec.duplicate_scalar %cst_54, %106 : f32 -> <f32>
          %108 = ascvec.cmp_eq %105, %107, %106 : <f32>, <f32> -> !ascvec.mask
          scf.for %arg31 = %c0_i32_51 to %arg30 step %c64_i32_50  : i32 {
            %109 = arith.index_cast %arg31 : i32 to index
            %110 = ascvec.loadalign %arg28[%109] : memref<576xf32, 1>, index -> <f32>
            %111 = ascvec.div %110, %104, %106 {precision_mode = true} : <f32>
            %112 = ascvec.select %107, %111, %108 : <f32>
            %113 = ascvec.max_scalar %112, %cst_53, %106 : <f32>, f32
            %114 = ascvec.min_scalar %113, %cst_52, %106 : <f32>, f32
            %115 = ascvec.cast %114, %106 {reg_layout = 0 : i32, rounding = 1 : i32} : <f32> -> <!ascdtype.hifloat8>
            %116 = ascvec.reinterpret %115 : <!ascdtype.hifloat8> -> <f8E4M3FN>
            ascvec.storealign_pack %arg29[%109], %116, %106 {mode = "b32_to_b8"} : memref<576xui8, 1>, index, <f8E4M3FN>
          }
          ascvec.mem_bar vst_vld
        } {cannir.vf_mode = "simd"}
        ascvec.release_buf v, %93, 0 {cannir.buffer.generated_sync}
        ascvec.release_buf v, %61, 0 {cannir.buffer.generated_sync}
        %95 = arith.muli %53, %32 : i64
        %96 = arith.addi %95, %80 : i64
        %97 = arith.muli %96, %c8_i64 : i64
        %98 = ascscalar.memref_address %alloc_5 : memref<1xui8>
        %99 = arith.divui %97, %c8_i64 : i64
        %100 = arith.addi %98, %99 : i64
        %101 = ascscalar.bind_gm_ptr %100 : memref<1xui8>
        %102 = ascscalar.bind_local_ptr %91 : memref<1xui8, 1>
        %103 = arith.cmpi ugt, %12, %c0_i64 : i64
        scf.if %103 {
          ascvec.get_buf mte3, %93, 0 {cannir.buffer.generated_sync}
          ascvec.copy_ub2gm %101[%c0], %102[%c0], %c1_i32, %94, %32, %94 l2_cache_ctl %c0_i32 : memref<1xui8>, memref<1xui8, 1>
          ascvec.release_buf mte3, %93, 0 {cannir.buffer.generated_sync}
        }
        scf.yield %63 : i64
      }
    }
    return
  }
  func.func @kernel_MlaPrologKernel_run_m3_qr_0_aic(%arg0: !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg1: !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg2: !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg3: !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg4: !cannir.tensor<<gm, bf16>, <<(-1, -1, 512)>, <(-1, 512, 1)>>, <<(-1, -1, 512)>, <(-1, 512, 1)>>, identity>, %arg5: !cannir.tensor<<gm, f32>, <<(1, -1)>, <(-1, 1)>>, <<(1, -1)>, <(-1, 1)>>, identity>, %arg6: !cannir.tensor<<gm, f32>, <<(1, 512)>, <(512, 1)>>, <<(1, 512)>, <(512, 1)>>, identity>, %arg7: !cannir.tensor<<gm, ui8>, <<(-1, -1, 1, -1)>, <(-1, -1, -1, 1)>>, <<(-1, -1, 1, -1)>, <(-1, -1, -1, 1)>>, identity>, %arg8: !cannir.tensor<<gm, i64>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg9: !cannir.tensor<<gm, f32>, <<(1)>, <(1)>>, <<(1)>, <(1)>>, identity>, %arg10: !cannir.tensor<<gm, ui8>, <<(8, -1, -1)>, <(-1, -1, 1)>>, <<(8, -1, -1)>, <(-1, -1, 1)>>, identity>, %arg11: !cannir.tensor<<gm, f32>, <<(1)>, <(1)>>, <<(1)>, <(1)>>, identity>, %arg12: !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg13: !cannir.tensor<<gm, bf16>, <<(8, -1)>, <(-1, 1)>>, <<(8, -1)>, <(-1, 1)>>, identity>, %arg14: !cannir.tensor<<gm, f32>, <<(32, -1)>, <(-1, 1)>>, <<(32, -1)>, <(-1, 1)>>, identity>, %arg15: !cannir.tensor<<gm, bf16>, <<(8, -1)>, <(-1, 1)>>, <<(8, -1)>, <(-1, 1)>>, identity>, %arg16: !cannir.tensor<<gm, bf16>, <<(8, -1)>, <(-1, 1)>>, <<(8, -1)>, <(-1, 1)>>, identity>, %arg17: !cannir.tensor<<gm, f32>, <<(8, -1, -1)>, <(-1, -1, 1)>>, <<(8, -1, -1)>, <(-1, -1, 1)>>, identity>, %arg18: !cannir.tensor<<gm, f32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg19: !cannir.tensor<<gm, bf16>, <<(8, -1)>, <(-1, 1)>>, <<(8, -1)>, <(-1, 1)>>, identity>, %arg20: f32, %arg21: f32, %arg22: f32, %arg23: i64, %arg24: i64) attributes {cannir.buffer.auto_sync, cannir.buffer.sync_analyzed, cannir.buffer.sync_verified, cannir.buffer.tensor_state, cannir.core = "aic", cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.device, cannir.explicit_ring_buffer_cursors, cannir.kernel_group = "kernel_MlaPrologKernel_run_m3_qr_0", cannir.resource.ids_prepared} {
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
    %c96_i64 = arith.constant 96 : i64
    %c128_i64 = arith.constant 128 : i64
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
    %alloc_3 = memref.alloc() {cannir.arg_index = 13 : i64} : memref<1xbf16>
    %alloc_4 = memref.alloc() {cannir.arg_index = 14 : i64} : memref<1xf32>
    %alloc_5 = memref.alloc() {cannir.arg_index = 15 : i64} : memref<1xbf16>
    %alloc_6 = memref.alloc() {cannir.arg_index = 16 : i64} : memref<1xbf16>
    %alloc_7 = memref.alloc() {cannir.arg_index = 17 : i64} : memref<1xf32>
    %alloc_8 = memref.alloc() {cannir.arg_index = 19 : i64} : memref<1xbf16>
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
    %36 = scf.for %arg25 = %0 to %28 step %1 iter_args(%arg26 = %c0_i64) -> (i64)  : i64 {
      %66 = arith.remsi %arg25, %28 : i64
      %67 = arith.cmpi ne, %66, %c0_i64 : i64
      %68 = arith.cmpi slt, %66, %c0_i64 : i64
      %69 = arith.cmpi slt, %28, %c0_i64 : i64
      %70 = arith.xori %68, %69 : i1
      %71 = arith.andi %67, %70 : i1
      %72 = arith.divsi %arg25, %28 : i64
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
      %84 = scf.for %arg27 = %c0_i64 to %83 step %c1_i64 iter_args(%arg28 = %arg26) -> (i64)  : i64 {
        %142 = arith.cmpi eq, %arg28, %c0_i64 : i64
        %143 = ascscalar.memref_address %alloc_46 : memref<4096xbf16, 2>
        %144 = ascscalar.memref_address %alloc_45 : memref<4096xbf16, 2>
        %145 = arith.select %142, %143, %144 : i64
        %146 = arith.cmpi ne, %arg28, %c0_i64 : i64
        %147 = arith.extui %146 : i1 to i64
        %148 = arith.addi %arg28, %c1_i64 overflow<nsw> : i64
        %149 = arith.remui %148, %c2_i64 : i64
        %150 = ascscalar.bind_local_ptr %145 : memref<1xbf16, 2>
        %151 = arith.muli %74, %c8_i64 : i64
        %152 = cannir.extract_physical_layout %arg13 : !cannir.tensor<<gm, bf16>, <<(8, -1)>, <(-1, 1)>>, <<(8, -1)>, <(-1, 1)>>, identity>
        %153 = cannir.extract_stride %152 : <<(8, -1)>, <(-1, 1)>> -> <(-1, 1)>
        %154 = cannir.extract_dim %153[0] : !cannir.stride<(-1, 1)>
        %155 = arith.muli %151, %154 : i64
        %156 = arith.index_cast %155 : i64 to index
        %157 = arith.muli %arg27, %c256_i64 : i64
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
        %170 = arith.minsi %arg27, %169 : i64
        %171 = arith.muli %170, %c256_i64 : i64
        %172 = arith.subi %168, %171 : i64
        %173 = arith.minsi %172, %c256_i64 : i64
        %174 = arith.cmpi sle, %arg27, %169 : i64
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
          asccube.copy_gm2l1 %150[%c0], %alloc_3[%159], %225, %226, %180, %c1_i32, %c0_i64, %c0_i64, %c16_i32 l2_cache_ctl %c1_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
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
        %194 = arith.muli %arg27, %c16_i64 : i64
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
        %205 = arith.minsi %arg27, %204 : i64
        %206 = arith.muli %205, %c256_i64 : i64
        %207 = arith.subi %203, %206 : i64
        %208 = arith.minsi %207, %c256_i64 : i64
        %209 = arith.cmpi sle, %arg27, %204 : i64
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
        %221 = arith.cmpi eq, %arg27, %c0_i64 : i64
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
      asccube.copy_l0c2gm %alloc_4[%94], %85[%c0], %c64_i32, %c8_i32, %98, %c16_i32, %c0_i32 l2_cache_ctl %c0_i32 {nz2nd_en = true, relu_en = false} : memref<1xf32>, memref<?xf32, 5>
      asccube.release_buf fixpipe, %c8_i64, 0 {cannir.buffer.generated_sync}
      %99 = arith.muli %35, %c2_i64 : i64
      %100 = arith.remsi %99, %c4_i64 : i64
      %101 = arith.cmpi ne, %100, %c0_i64 : i64
      %102 = arith.cmpi slt, %100, %c0_i64 : i64
      %103 = arith.andi %101, %102 : i1
      %104 = arith.divsi %99, %c4_i64 : i64
      %105 = arith.extui %103 : i1 to i64
      %106 = arith.subi %104, %105 : i64
      %107 = scf.for %arg27 = %83 to %106 step %c1_i64 iter_args(%arg28 = %84) -> (i64)  : i64 {
        %142 = arith.cmpi eq, %arg28, %c0_i64 : i64
        %143 = ascscalar.memref_address %alloc_46 : memref<4096xbf16, 2>
        %144 = ascscalar.memref_address %alloc_45 : memref<4096xbf16, 2>
        %145 = arith.select %142, %143, %144 : i64
        %146 = arith.cmpi ne, %arg28, %c0_i64 : i64
        %147 = arith.extui %146 : i1 to i64
        %148 = arith.addi %arg28, %c1_i64 overflow<nsw> : i64
        %149 = arith.remui %148, %c2_i64 : i64
        %150 = ascscalar.bind_local_ptr %145 : memref<1xbf16, 2>
        %151 = cannir.extract_physical_layout %arg13 : !cannir.tensor<<gm, bf16>, <<(8, -1)>, <(-1, 1)>>, <<(8, -1)>, <(-1, 1)>>, identity>
        %152 = cannir.extract_stride %151 : <<(8, -1)>, <(-1, 1)>> -> <(-1, 1)>
        %153 = cannir.extract_dim %152[0] : !cannir.stride<(-1, 1)>
        %154 = arith.muli %86, %153 : i64
        %155 = arith.index_cast %154 : i64 to index
        %156 = arith.muli %arg27, %c256_i64 : i64
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
        %169 = arith.minsi %arg27, %168 : i64
        %170 = arith.muli %169, %c256_i64 : i64
        %171 = arith.subi %167, %170 : i64
        %172 = arith.minsi %171, %c256_i64 : i64
        %173 = arith.cmpi sle, %arg27, %168 : i64
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
          asccube.copy_gm2l1 %150[%c0], %alloc_3[%158], %224, %225, %179, %c1_i32, %c0_i64, %c0_i64, %c16_i32 l2_cache_ctl %c1_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
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
        %193 = arith.muli %arg27, %c16_i64 : i64
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
        %204 = arith.minsi %arg27, %203 : i64
        %205 = arith.muli %204, %c256_i64 : i64
        %206 = arith.subi %202, %205 : i64
        %207 = arith.minsi %206, %c256_i64 : i64
        %208 = arith.cmpi sle, %arg27, %203 : i64
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
        %220 = arith.cmpi eq, %arg27, %83 : i64
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
      asccube.copy_l0c2gm %alloc_4[%113], %109[%c0], %c64_i32, %c8_i32, %98, %c16_i32, %c0_i32 l2_cache_ctl %c0_i32 {nz2nd_en = true, relu_en = false} : memref<1xf32>, memref<?xf32, 5>
      asccube.release_buf fixpipe, %c9_i64, 0 {cannir.buffer.generated_sync}
      %114 = arith.muli %35, %c3_i64 : i64
      %115 = arith.remsi %114, %c4_i64 : i64
      %116 = arith.cmpi ne, %115, %c0_i64 : i64
      %117 = arith.cmpi slt, %115, %c0_i64 : i64
      %118 = arith.andi %116, %117 : i1
      %119 = arith.divsi %114, %c4_i64 : i64
      %120 = arith.extui %118 : i1 to i64
      %121 = arith.subi %119, %120 : i64
      %122 = scf.for %arg27 = %106 to %121 step %c1_i64 iter_args(%arg28 = %107) -> (i64)  : i64 {
        %142 = arith.cmpi eq, %arg28, %c0_i64 : i64
        %143 = ascscalar.memref_address %alloc_46 : memref<4096xbf16, 2>
        %144 = ascscalar.memref_address %alloc_45 : memref<4096xbf16, 2>
        %145 = arith.select %142, %143, %144 : i64
        %146 = arith.cmpi ne, %arg28, %c0_i64 : i64
        %147 = arith.extui %146 : i1 to i64
        %148 = arith.addi %arg28, %c1_i64 overflow<nsw> : i64
        %149 = arith.remui %148, %c2_i64 : i64
        %150 = ascscalar.bind_local_ptr %145 : memref<1xbf16, 2>
        %151 = cannir.extract_physical_layout %arg13 : !cannir.tensor<<gm, bf16>, <<(8, -1)>, <(-1, 1)>>, <<(8, -1)>, <(-1, 1)>>, identity>
        %152 = cannir.extract_stride %151 : <<(8, -1)>, <(-1, 1)>> -> <(-1, 1)>
        %153 = cannir.extract_dim %152[0] : !cannir.stride<(-1, 1)>
        %154 = arith.muli %86, %153 : i64
        %155 = arith.index_cast %154 : i64 to index
        %156 = arith.muli %arg27, %c256_i64 : i64
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
        %169 = arith.minsi %arg27, %168 : i64
        %170 = arith.muli %169, %c256_i64 : i64
        %171 = arith.subi %167, %170 : i64
        %172 = arith.minsi %171, %c256_i64 : i64
        %173 = arith.cmpi sle, %arg27, %168 : i64
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
          asccube.copy_gm2l1 %150[%c0], %alloc_3[%158], %223, %224, %179, %c1_i32, %c0_i64, %c0_i64, %c16_i32 l2_cache_ctl %c1_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
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
        %193 = arith.muli %arg27, %c16_i64 : i64
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
        %204 = arith.minsi %arg27, %203 : i64
        %205 = arith.muli %204, %c256_i64 : i64
        %206 = arith.subi %202, %205 : i64
        %207 = arith.minsi %206, %c256_i64 : i64
        %208 = arith.cmpi sle, %arg27, %203 : i64
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
        %220 = arith.cmpi eq, %arg27, %106 : i64
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
      asccube.copy_l0c2gm %alloc_4[%127], %85[%c0], %c64_i32, %c8_i32, %98, %c16_i32, %c0_i32 l2_cache_ctl %c0_i32 {nz2nd_en = true, relu_en = false} : memref<1xf32>, memref<?xf32, 5>
      asccube.release_buf fixpipe, %c8_i64, 0 {cannir.buffer.generated_sync}
      %128 = arith.muli %35, %c4_i64 : i64
      %129 = arith.remsi %128, %c4_i64 : i64
      %130 = arith.cmpi ne, %129, %c0_i64 : i64
      %131 = arith.cmpi slt, %129, %c0_i64 : i64
      %132 = arith.andi %130, %131 : i1
      %133 = arith.divsi %128, %c4_i64 : i64
      %134 = arith.extui %132 : i1 to i64
      %135 = arith.subi %133, %134 : i64
      %136 = scf.for %arg27 = %121 to %135 step %c1_i64 iter_args(%arg28 = %122) -> (i64)  : i64 {
        %142 = arith.cmpi eq, %arg28, %c0_i64 : i64
        %143 = ascscalar.memref_address %alloc_46 : memref<4096xbf16, 2>
        %144 = ascscalar.memref_address %alloc_45 : memref<4096xbf16, 2>
        %145 = arith.select %142, %143, %144 : i64
        %146 = arith.cmpi ne, %arg28, %c0_i64 : i64
        %147 = arith.extui %146 : i1 to i64
        %148 = arith.addi %arg28, %c1_i64 overflow<nsw> : i64
        %149 = arith.remui %148, %c2_i64 : i64
        %150 = ascscalar.bind_local_ptr %145 : memref<1xbf16, 2>
        %151 = cannir.extract_physical_layout %arg13 : !cannir.tensor<<gm, bf16>, <<(8, -1)>, <(-1, 1)>>, <<(8, -1)>, <(-1, 1)>>, identity>
        %152 = cannir.extract_stride %151 : <<(8, -1)>, <(-1, 1)>> -> <(-1, 1)>
        %153 = cannir.extract_dim %152[0] : !cannir.stride<(-1, 1)>
        %154 = arith.muli %86, %153 : i64
        %155 = arith.index_cast %154 : i64 to index
        %156 = arith.muli %arg27, %c256_i64 : i64
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
        %169 = arith.minsi %arg27, %168 : i64
        %170 = arith.muli %169, %c256_i64 : i64
        %171 = arith.subi %167, %170 : i64
        %172 = arith.minsi %171, %c256_i64 : i64
        %173 = arith.cmpi sle, %arg27, %168 : i64
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
          asccube.copy_gm2l1 %150[%c0], %alloc_3[%158], %223, %224, %179, %c1_i32, %c0_i64, %c0_i64, %c16_i32 l2_cache_ctl %c1_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
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
        %193 = arith.muli %arg27, %c16_i64 : i64
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
        %204 = arith.minsi %arg27, %203 : i64
        %205 = arith.muli %204, %c256_i64 : i64
        %206 = arith.subi %202, %205 : i64
        %207 = arith.minsi %206, %c256_i64 : i64
        %208 = arith.cmpi sle, %arg27, %203 : i64
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
        %220 = arith.cmpi eq, %arg27, %121 : i64
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
      asccube.copy_l0c2gm %alloc_4[%141], %109[%c0], %c64_i32, %c8_i32, %98, %c16_i32, %c0_i32 l2_cache_ctl %c0_i32 {nz2nd_en = true, relu_en = false} : memref<1xf32>, memref<?xf32, 5>
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
    %44:3 = scf.for %arg25 = %0 to %43 step %1 iter_args(%arg26 = %c0_i64, %arg27 = %c0_i64, %arg28 = %c0_i64) -> (i64, i64, i64)  : i64 {
      %66 = arith.remsi %arg25, %43 : i64
      %67 = arith.cmpi ne, %66, %c0_i64 : i64
      %68 = arith.cmpi slt, %66, %c0_i64 : i64
      %69 = arith.cmpi slt, %43, %c0_i64 : i64
      %70 = arith.xori %68, %69 : i1
      %71 = arith.andi %67, %70 : i1
      %72 = arith.divsi %arg25, %43 : i64
      %73 = arith.extui %71 : i1 to i64
      %74 = arith.subi %72, %73 : i64
      %75 = arith.select %71, %43, %c0_i64 : i64
      %76 = arith.addi %66, %75 : i64
      %77 = arith.cmpi eq, %arg28, %c0_i64 : i64
      %78 = ascscalar.memref_address %alloc_42 : memref<1024xf32, 5>
      %79 = ascscalar.memref_address %alloc_41 : memref<1024xf32, 5>
      %80 = arith.select %77, %78, %79 : i64
      %81 = arith.select %77, %c8_i64, %c9_i64 : i64
      %82 = arith.addi %arg28, %c1_i64 overflow<nsw> : i64
      %83 = arith.remui %82, %c2_i64 : i64
      %84:2 = scf.for %arg29 = %c0_i64 to %35 step %c1_i64 iter_args(%arg30 = %arg26, %arg31 = %arg27) -> (i64, i64)  : i64 {
        %99 = arith.cmpi eq, %arg30, %c0_i64 : i64
        %100 = ascscalar.memref_address %alloc_40 : memref<4096xbf16, 2>
        %101 = ascscalar.memref_address %alloc_39 : memref<4096xbf16, 2>
        %102 = arith.select %99, %100, %101 : i64
        %103 = arith.cmpi ne, %arg30, %c0_i64 : i64
        %104 = arith.extui %103 : i1 to i64
        %105 = arith.addi %arg30, %c1_i64 overflow<nsw> : i64
        %106 = arith.remui %105, %c2_i64 : i64
        %107 = ascscalar.bind_local_ptr %102 : memref<1xbf16, 2>
        %108 = arith.muli %74, %c8_i64 : i64
        %109 = cannir.extract_physical_layout %arg13 : !cannir.tensor<<gm, bf16>, <<(8, -1)>, <(-1, 1)>>, <<(8, -1)>, <(-1, 1)>>, identity>
        %110 = cannir.extract_stride %109 : <<(8, -1)>, <(-1, 1)>> -> <(-1, 1)>
        %111 = cannir.extract_dim %110[0] : !cannir.stride<(-1, 1)>
        %112 = arith.muli %108, %111 : i64
        %113 = arith.index_cast %112 : i64 to index
        %114 = arith.muli %arg29, %c256_i64 : i64
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
        %127 = arith.minsi %arg29, %126 : i64
        %128 = arith.muli %127, %c256_i64 : i64
        %129 = arith.subi %125, %128 : i64
        %130 = arith.minsi %129, %c256_i64 : i64
        %131 = arith.cmpi sle, %arg29, %126 : i64
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
          asccube.copy_gm2l1 %107[%c0], %alloc_3[%116], %200, %201, %137, %c1_i32, %c0_i64, %c0_i64, %c16_i32 l2_cache_ctl %c1_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
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
        %151 = arith.muli %arg29, %c16_i64 : i64
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
        %162 = arith.minsi %arg29, %161 : i64
        %163 = arith.muli %162, %c256_i64 : i64
        %164 = arith.subi %160, %163 : i64
        %165 = arith.minsi %164, %c256_i64 : i64
        %166 = arith.cmpi sle, %arg29, %161 : i64
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
        %176 = arith.cmpi eq, %arg31, %c0_i64 : i64
        %177 = ascscalar.memref_address %alloc_36 : memref<2048xbf16, 3>
        %178 = ascscalar.memref_address %alloc_35 : memref<2048xbf16, 3>
        %179 = arith.select %176, %177, %178 : i64
        %180 = arith.select %176, %c4_i64, %c5_i64 : i64
        %181 = arith.addi %arg31, %c1_i64 overflow<nsw> : i64
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
        %189 = arith.cmpi eq, %arg29, %c0_i64 : i64
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
      asccube.copy_l0c2gm %alloc_5[%94], %85[%c0], %c64_i32, %c8_i32, %98, %c16_i32, %c0_i32 l2_cache_ctl %c0_i32 {nz2nd_en = true, relu_en = false} : memref<1xbf16>, memref<1xf32, 5>
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
    %53:2 = scf.for %arg25 = %0 to %52 step %1 iter_args(%arg26 = %c0_i64, %arg27 = %c0_i64) -> (i64, i64)  : i64 {
      %66 = arith.muli %arg25, %c96_i64 : i64
      %67 = arith.cmpi eq, %arg27, %c0_i64 : i64
      %68 = ascscalar.memref_address %alloc_32 : memref<1536xf32, 5>
      %69 = ascscalar.memref_address %alloc_31 : memref<1536xf32, 5>
      %70 = arith.select %67, %68, %69 : i64
      %71 = arith.select %67, %c8_i64, %c9_i64 : i64
      %72 = arith.addi %arg27, %c1_i64 overflow<nsw> : i64
      %73 = arith.remui %72, %c2_i64 : i64
      %74 = arith.remsi %8, %c128_i64 : i64
      %75 = arith.cmpi ne, %74, %c0_i64 : i64
      %76 = arith.cmpi slt, %74, %c0_i64 : i64
      %77 = arith.andi %75, %76 : i1
      %78 = arith.divsi %8, %c128_i64 : i64
      %79 = arith.extui %77 : i1 to i64
      %80 = arith.subi %78, %79 : i64
      %81 = scf.for %arg28 = %c0_i64 to %80 step %c1_i64 iter_args(%arg29 = %arg26) -> (i64)  : i64 {
        %96 = arith.cmpi eq, %arg29, %c0_i64 : i64
        %97 = ascscalar.memref_address %alloc_30 : memref<2048xbf16, 2>
        %98 = ascscalar.memref_address %alloc_29 : memref<2048xbf16, 2>
        %99 = arith.select %96, %97, %98 : i64
        %100 = arith.cmpi ne, %arg29, %c0_i64 : i64
        %101 = arith.extui %100 : i1 to i64
        %102 = arith.addi %arg29, %c1_i64 overflow<nsw> : i64
        %103 = arith.remui %102, %c2_i64 : i64
        %104 = ascscalar.bind_local_ptr %99 : memref<1xbf16, 2>
        %105 = arith.muli %arg28, %c128_i64 : i64
        %106 = cannir.extract_layout %arg19 : !cannir.tensor<<gm, bf16>, <<(8, -1)>, <(-1, 1)>>, <<(8, -1)>, <(-1, 1)>>, identity>
        %107 = cannir.extract_shape %106 : <<(8, -1)>, <(-1, 1)>> -> <(8, -1)>
        %108 = cannir.extract_dim %107[1] : !cannir.shape<(8, -1)>
        %109 = arith.divsi %108, %c128_i64 : i64
        %110 = arith.minsi %arg28, %109 : i64
        %111 = arith.muli %110, %c128_i64 : i64
        %112 = arith.subi %108, %111 : i64
        %113 = arith.minsi %112, %c128_i64 : i64
        %114 = arith.cmpi sle, %arg28, %109 : i64
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
        %144 = arith.muli %arg28, %c8_i64 : i64
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
        %155 = arith.minsi %arg28, %154 : i64
        %156 = arith.muli %155, %c128_i64 : i64
        %157 = arith.subi %153, %156 : i64
        %158 = arith.minsi %157, %c128_i64 : i64
        %159 = arith.cmpi sle, %arg28, %154 : i64
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
        %174 = arith.cmpi eq, %arg28, %c0_i64 : i64
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
      asccube.copy_l0c2gm %alloc_6[%95], %89[%c0], %c96_i32, %c8_i32, %94, %c16_i32, %c0_i32 l2_cache_ctl %c0_i32 {nz2nd_en = true, relu_en = false} : memref<1xbf16>, memref<1xf32, 5>
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
        %90 = scf.for %arg25 = %c0_i64 to %89 step %c1_i64 iter_args(%arg26 = %53#0) -> (i64)  : i64 {
          %105 = arith.cmpi eq, %arg26, %c0_i64 : i64
          %106 = ascscalar.memref_address %alloc_30 : memref<2048xbf16, 2>
          %107 = ascscalar.memref_address %alloc_29 : memref<2048xbf16, 2>
          %108 = arith.select %105, %106, %107 : i64
          %109 = arith.cmpi ne, %arg26, %c0_i64 : i64
          %110 = arith.extui %109 : i1 to i64
          %111 = arith.addi %arg26, %c1_i64 overflow<nsw> : i64
          %112 = arith.remui %111, %c2_i64 : i64
          %113 = ascscalar.bind_local_ptr %108 : memref<1xbf16, 2>
          %114 = arith.muli %arg25, %c128_i64 : i64
          %115 = cannir.extract_layout %arg19 : !cannir.tensor<<gm, bf16>, <<(8, -1)>, <(-1, 1)>>, <<(8, -1)>, <(-1, 1)>>, identity>
          %116 = cannir.extract_shape %115 : <<(8, -1)>, <(-1, 1)>> -> <(8, -1)>
          %117 = cannir.extract_dim %116[1] : !cannir.shape<(8, -1)>
          %118 = arith.divsi %117, %c128_i64 : i64
          %119 = arith.minsi %arg25, %118 : i64
          %120 = arith.muli %119, %c128_i64 : i64
          %121 = arith.subi %117, %120 : i64
          %122 = arith.minsi %121, %c128_i64 : i64
          %123 = arith.cmpi sle, %arg25, %118 : i64
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
          %153 = arith.muli %arg25, %c8_i64 : i64
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
          %165 = arith.minsi %arg25, %164 : i64
          %166 = arith.muli %165, %c128_i64 : i64
          %167 = arith.subi %163, %166 : i64
          %168 = arith.minsi %167, %c128_i64 : i64
          %169 = arith.cmpi sle, %arg25, %164 : i64
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
          %184 = arith.cmpi eq, %arg25, %c0_i64 : i64
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
        asccube.copy_l0c2gm %alloc_6[%104], %98[%c0], %c96_i32, %c8_i32, %103, %c16_i32, %c0_i32 l2_cache_ctl %c0_i32 {nz2nd_en = true, relu_en = false} : memref<1xbf16>, memref<1xf32, 5>
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
        %87 = scf.for %arg25 = %c0_i64 to %86 step %c1_i64 iter_args(%arg26 = %57#0) -> (i64)  : i64 {
          %102 = arith.cmpi eq, %arg26, %c0_i64 : i64
          %103 = ascscalar.memref_address %alloc_30 : memref<2048xbf16, 2>
          %104 = ascscalar.memref_address %alloc_29 : memref<2048xbf16, 2>
          %105 = arith.select %102, %103, %104 : i64
          %106 = arith.cmpi ne, %arg26, %c0_i64 : i64
          %107 = arith.extui %106 : i1 to i64
          %108 = arith.addi %arg26, %c1_i64 overflow<nsw> : i64
          %109 = arith.remui %108, %c2_i64 : i64
          %110 = ascscalar.bind_local_ptr %105 : memref<1xbf16, 2>
          %111 = arith.muli %arg25, %c128_i64 : i64
          %112 = cannir.extract_layout %arg19 : !cannir.tensor<<gm, bf16>, <<(8, -1)>, <(-1, 1)>>, <<(8, -1)>, <(-1, 1)>>, identity>
          %113 = cannir.extract_shape %112 : <<(8, -1)>, <(-1, 1)>> -> <(8, -1)>
          %114 = cannir.extract_dim %113[1] : !cannir.shape<(8, -1)>
          %115 = arith.divsi %114, %c128_i64 : i64
          %116 = arith.minsi %arg25, %115 : i64
          %117 = arith.muli %116, %c128_i64 : i64
          %118 = arith.subi %114, %117 : i64
          %119 = arith.minsi %118, %c128_i64 : i64
          %120 = arith.cmpi sle, %arg25, %115 : i64
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
          %150 = arith.muli %arg25, %c8_i64 : i64
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
          %162 = arith.minsi %arg25, %161 : i64
          %163 = arith.muli %162, %c128_i64 : i64
          %164 = arith.subi %160, %163 : i64
          %165 = arith.minsi %164, %c128_i64 : i64
          %166 = arith.cmpi sle, %arg25, %161 : i64
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
          %181 = arith.cmpi eq, %arg25, %c0_i64 : i64
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
        asccube.copy_l0c2gm %alloc_6[%101], %95[%c0], %c96_i32, %c8_i32, %100, %c16_i32, %c0_i32 l2_cache_ctl %c0_i32 {nz2nd_en = true, relu_en = false} : memref<1xbf16>, memref<1xf32, 5>
        asccube.release_buf fixpipe, %79, 0 {cannir.buffer.generated_sync}
      }
    }
    asccube.sync_all
    asccube.sync_block_wait s, %c13_i64
    asccube.sync_block_arrive fixpipe, %c14_i64 {mode = 0 : i32}
    asccube.sync_block_wait s, %c14_i64 {mode = 0 : i32}
    asccube.sync_block_arrive mte3, %c15_i64
    %59 = cannir.extract_layout %arg17 : !cannir.tensor<<gm, f32>, <<(8, -1, -1)>, <(-1, -1, 1)>>, <<(8, -1, -1)>, <(-1, -1, 1)>>, identity>
    %60 = cannir.extract_shape %59 : <<(8, -1, -1)>, <(-1, -1, 1)>> -> <(8, -1, -1)>
    %61 = cannir.extract_dim %60[1] : !cannir.shape<(8, -1, -1)>
    %62 = cannir.extract_dim %60[2] : !cannir.shape<(8, -1, -1)>
    %63 = arith.muli %61, %62 : i64
    %64 = arith.muli %14, %c4_i64 : i64
    %65:2 = scf.for %arg25 = %0 to %64 step %1 iter_args(%arg26 = %c0_i64, %arg27 = %c0_i64) -> (i64, i64)  : i64 {
      %66 = arith.remsi %arg25, %c4_i64 : i64
      %67 = arith.cmpi ne, %66, %c0_i64 : i64
      %68 = arith.cmpi slt, %66, %c0_i64 : i64
      %69 = arith.andi %67, %68 : i1
      %70 = arith.select %69, %c4_i64, %c0_i64 : i64
      %71 = arith.addi %66, %70 overflow<nsw> : i64
      %72 = arith.divsi %arg25, %c4_i64 : i64
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
      %83 = arith.remsi %arg25, %64 : i64
      %84 = arith.cmpi ne, %83, %c0_i64 : i64
      %85 = arith.cmpi slt, %83, %c0_i64 : i64
      %86 = arith.cmpi slt, %64, %c0_i64 : i64
      %87 = arith.xori %85, %86 : i1
      %88 = arith.andi %84, %87 : i1
      %89 = arith.divsi %arg25, %64 : i64
      %90 = arith.extui %88 : i1 to i64
      %91 = arith.subi %89, %90 : i64
      %92 = arith.muli %91, %c8_i64 : i64
      %93 = arith.muli %82, %17 : i64
      %94 = arith.muli %82, %11 : i64
      %95 = arith.cmpi eq, %arg27, %c0_i64 : i64
      %96 = ascscalar.memref_address %alloc_22 : memref<2048xf32, 5>
      %97 = ascscalar.memref_address %alloc_21 : memref<2048xf32, 5>
      %98 = arith.select %95, %96, %97 : i64
      %99 = arith.select %95, %c6_i64, %c7_i64 : i64
      %100 = arith.addi %arg27, %c1_i64 overflow<nsw> : i64
      %101 = arith.remui %100, %c2_i64 : i64
      %102 = arith.remsi %15, %c64_i64 : i64
      %103 = arith.cmpi ne, %102, %c0_i64 : i64
      %104 = arith.cmpi slt, %102, %c0_i64 : i64
      %105 = arith.andi %103, %104 : i1
      %106 = arith.divsi %15, %c64_i64 : i64
      %107 = arith.extui %105 : i1 to i64
      %108 = arith.subi %106, %107 : i64
      %109 = scf.for %arg28 = %c0_i64 to %108 step %c1_i64 iter_args(%arg29 = %arg26) -> (i64)  : i64 {
        %118 = arith.cmpi eq, %arg29, %c0_i64 : i64
        %119 = ascscalar.memref_address %alloc_20 : memref<1024xbf16, 2>
        %120 = ascscalar.memref_address %alloc_19 : memref<1024xbf16, 2>
        %121 = arith.select %118, %119, %120 : i64
        %122 = arith.cmpi ne, %arg29, %c0_i64 : i64
        %123 = arith.extui %122 : i1 to i64
        %124 = arith.addi %arg29, %c1_i64 overflow<nsw> : i64
        %125 = arith.remui %124, %c2_i64 : i64
        %126 = arith.muli %arg28, %c64_i64 : i64
        %127 = arith.addi %93, %126 : i64
        %128 = arith.addi %arg28, %c1_i64 : i64
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
          asccube.copy_gm2l1 %132[%c0], %alloc_6[%139], %c8_i32, %177, %145, %c1_i32, %c0_i64, %c0_i64, %c16_i32 l2_cache_ctl %c1_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
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
        %175 = arith.cmpi eq, %arg28, %c0_i64 : i64
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
      asccube.copy_l0c2gm %alloc_7[%116], %112[%c0], %c128_i32, %c8_i32, %117, %c16_i32, %c0_i32 l2_cache_ctl %c0_i32 {nz2nd_en = true, relu_en = false} : memref<1xf32>, memref<1xf32, 5>
      asccube.release_buf fixpipe, %99, 0 {cannir.buffer.generated_sync}
      scf.yield %109, %101 : i64, i64
    }
    asccube.sync_all
    asccube.sync_block_wait s, %c13_i64
    asccube.sync_block_arrive fixpipe, %c14_i64 {mode = 0 : i32}
    asccube.sync_block_wait s, %c14_i64 {mode = 0 : i32}
    asccube.sync_block_arrive mte3, %c15_i64
    asccube.sync_all
    asccube.sync_block_wait s, %c13_i64
    asccube.sync_block_arrive fixpipe, %c14_i64 {mode = 0 : i32}
    asccube.sync_block_wait s, %c14_i64 {mode = 0 : i32}
    asccube.sync_block_arrive mte3, %c15_i64
    return
  }
}
