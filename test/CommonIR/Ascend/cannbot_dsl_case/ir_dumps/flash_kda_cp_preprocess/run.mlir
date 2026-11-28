module {
  func.func @run_10d7177016a4856a28f194648b3756e9(%arg0: !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>, %arg1: !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>, %arg2: !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>, %arg3: !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 1)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 1)>, <(-1, -1, -1, 1)>>, identity>, %arg4: !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>, %arg5: !cannir.tensor<<gm, f32>, <<(-1, 2, 64, 64)>, <(8192, 4096, 64, 1)>>, <<(-1, 2, 64, 64)>, <(8192, 4096, 64, 1)>>, identity>, %arg6: !cannir.tensor<<gm, f32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg7: !cannir.tensor<<gm, f32>, <<(-1, 128)>, <(128, 1)>>, <<(-1, 128)>, <(128, 1)>>, identity>, %arg8: !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>, %arg9: !cannir.tensor<<gm, f32>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>, %arg10: !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>, %arg11: !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>, %arg12: !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 64, 64)>, <(-1, -1, 4096, 64, 1)>>, <<(-1, -1, -1, 64, 64)>, <(-1, -1, 4096, 64, 1)>>, identity>, %arg13: !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>, %arg14: !cannir.tensor<<gm, f32>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, identity>, %arg15: !cannir.tensor<<gm, f32>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, identity>, %arg16: !cannir.tensor<<gm, f16>, <<(16, 16)>, <(16, 1)>>, <<(16, 16)>, <(16, 1)>>, identity>, %arg17: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg18: i64, %arg19: i64, %arg20: f32, %arg21: f32, %arg22: i64) attributes {cannbotdsl.contract_digest = "10d7177016a4856a28f194648b3756e953375a45cbe6b20ed37fbeb68372aa09", cannbotdsl.public_slot_count = 23 : i64, cannbotdsl.slot_dtype_bits = [16, 16, 16, 16, 16, 32, 32, 32, 16, 32, 16, 16, 16, 16, 32, 32, 16, 32, 64, 64, 32, 32, 64], cannbotdsl.slot_dtype_codes = [4, 4, 4, 4, 4, 2, 2, 2, 4, 2, 4, 4, 4, 4, 2, 2, 2, 0, 0, 0, 2, 2, 0], cannbotdsl.slot_dtype_lanes = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], cannbotdsl.slot_flags = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], cannbotdsl.slot_kinds = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2], cannbotdsl.slot_type_digests = ["4a1c69d2ffc87d66669b1e4496ecac91f7d4973a54afad124d76b586258c12b1", "8fe4bd9c465df919d8abcb9d884336d77bd020b58b8676edd0bb7d89a7f70b7e", "dece239ab58b88079dcd449de844a681d422db3073506e9331755926ed93c9b5", "fbdf5e05258e65c7d4ac3fc4a0ad4fd176ee824c75c04ae8ea52663d47c3f9db", "d564f0f510d1f40c384e475c31ae55066d4d89bb5756077c61d9a58d79a794d3", "b31ece3585500b507d5a8d7b97d291a09567e8d6c56916d08e8e237e678238d6", "a05ebd8b3d177a4571fe8f4fd5b23e273c533df6e9b02ccadb3c402089f77ad1", "55f9d4577234f976a0c1a4d4e18d91486f24ad29a64d613a2aa1351a969a9283", "8ae7865ed05b48a13663d67ae7fbda02f2bfb23ec15e2bd62d52c6cc5f769d46", "1960f968740fb9ea4f503863f89c134d1c1888cdbaef6f2adbd598a191355ce3", "95677d90b1022c942b5e0aea92adf9600c5c260fe67d0052617fd711ddbcfb8f", "e8532f0b6dc2ff14140b6559036f38df8a747fc5f896f86e09e6f47a9d9a2dd2", "35f7a44a9b6e5cba2bd13365089124842961be1707ab7912323647c2d8f0b9e4", "3c265f70ec96ee7efb7896aed46bf0eba796541ff055f367840b3196bd0c8aa3", "5ae2adc2c479dc1b99c70666e575927277c2a02ec4793f1a2bbaee13c14b352a", "5ae2adc2c479dc1b99c70666e575927277c2a02ec4793f1a2bbaee13c14b352a", "a9d6bd41bbceabc46c2acae522a59dbc980761fc6b64a50ed67e332fcc91aec6", "df1782892fb0fe07e22c2b9245d6bd16a4aa9b9bed10c33a9432045882421aba", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "4a5363e2ec1eb8cefee037ab0b231653c07f9b44a406ef92ed9ba3e6f5541dbf", "4a5363e2ec1eb8cefee037ab0b231653c07f9b44a406ef92ed9ba3e6f5541dbf", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515"], cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.host} {
    cannir.kernel_launch @kernel_FlashKdaCpPreprocessKernel___call___0(%arg22) [%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18, %arg19, %arg20, %arg21] {cannbotdsl.profile = {indices = array<i64: 0, 1, 2, 3, 4, 6, 7, 17, 14, 15, 9, 10, 8, 13, 11, 12>, name = "flash_kda_cp_preprocess", op_type = "FlashKDACPPreprocess", roles = array<i64: 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1>}} : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>, !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>, !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>, !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 1)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 1)>, <(-1, -1, -1, 1)>>, identity>, !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>, !cannir.tensor<<gm, f32>, <<(-1, 2, 64, 64)>, <(8192, 4096, 64, 1)>>, <<(-1, 2, 64, 64)>, <(8192, 4096, 64, 1)>>, identity>, !cannir.tensor<<gm, f32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, !cannir.tensor<<gm, f32>, <<(-1, 128)>, <(128, 1)>>, <<(-1, 128)>, <(128, 1)>>, identity>, !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>, !cannir.tensor<<gm, f32>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>, !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>, !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>, !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 64, 64)>, <(-1, -1, 4096, 64, 1)>>, <<(-1, -1, -1, 64, 64)>, <(-1, -1, 4096, 64, 1)>>, identity>, !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>, !cannir.tensor<<gm, f32>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, identity>, !cannir.tensor<<gm, f32>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, identity>, !cannir.tensor<<gm, f16>, <<(16, 16)>, <(16, 1)>>, <<(16, 16)>, <(16, 1)>>, identity>, !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, i64, i64, f32, f32
    return
  }
  func.func @kernel_FlashKdaCpPreprocessKernel___call___0_aiv(%arg0: !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>, %arg1: !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>, %arg2: !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>, %arg3: !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 1)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 1)>, <(-1, -1, -1, 1)>>, identity>, %arg4: !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>, %arg5: !cannir.tensor<<gm, f32>, <<(-1, 2, 64, 64)>, <(8192, 4096, 64, 1)>>, <<(-1, 2, 64, 64)>, <(8192, 4096, 64, 1)>>, identity>, %arg6: !cannir.tensor<<gm, f32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg7: !cannir.tensor<<gm, f32>, <<(-1, 128)>, <(128, 1)>>, <<(-1, 128)>, <(128, 1)>>, identity>, %arg8: !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>, %arg9: !cannir.tensor<<gm, f32>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>, %arg10: !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>, %arg11: !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>, %arg12: !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 64, 64)>, <(-1, -1, 4096, 64, 1)>>, <<(-1, -1, -1, 64, 64)>, <(-1, -1, 4096, 64, 1)>>, identity>, %arg13: !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>, %arg14: !cannir.tensor<<gm, f32>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, identity>, %arg15: !cannir.tensor<<gm, f32>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, identity>, %arg16: !cannir.tensor<<gm, f16>, <<(16, 16)>, <(16, 1)>>, <<(16, 16)>, <(16, 1)>>, identity>, %arg17: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg18: i64, %arg19: i64, %arg20: f32, %arg21: f32) attributes {cannir.buffer.auto_sync, cannir.buffer.sync_analyzed, cannir.buffer.sync_verified, cannir.buffer.tensor_state, cannir.core = "aiv", cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.device, cannir.explicit_ring_buffer_cursors, cannir.kernel_group = "kernel_FlashKdaCpPreprocessKernel___call___0", cannir.resource.ids_prepared} {
    %c32768_i32 = arith.constant 32768 : i32
    %c4096_i32 = arith.constant 4096 : i32
    %c8192_i32 = arith.constant 8192 : i32
    %c16384_i32 = arith.constant 16384 : i32
    %c512 = arith.constant 512 : index
    %c4_i32 = arith.constant 4 : i32
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c6 = arith.constant 6 : index
    %c5 = arith.constant 5 : index
    %c3 = arith.constant 3 : index
    %c1262764338_i32 = arith.constant 1262764338 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %false = arith.constant false
    %c1_i32 = arith.constant 1 : i32
    %c9_i32 = arith.constant 9 : i32
    %c0 = arith.constant 0 : index
    %c256_i32 = arith.constant 256 : i32
    %c64_i32 = arith.constant 64 : i32
    %c32_i32 = arith.constant 32 : i32
    %c128_i32 = arith.constant 128 : i32
    %c16_i32 = arith.constant 16 : i32
    %c128_i64 = arith.constant 128 : i64
    %c256_i64 = arith.constant 256 : i64
    %c512_i64 = arith.constant 512 : i64
    %c32768_i64 = arith.constant 32768 : i64
    %c16384_i64 = arith.constant 16384 : i64
    %c1024 = arith.constant 1024 : index
    %c16 = arith.constant 16 : index
    %c2 = arith.constant 2 : index
    %c8192_i64 = arith.constant 8192 : i64
    %c4096_i64 = arith.constant 4096 : i64
    %c8 = arith.constant 8 : index
    %c2048_i64 = arith.constant 2048 : i64
    %c262144_i64 = arith.constant 262144 : i64
    %c256 = arith.constant 256 : index
    %c65_i32 = arith.constant 65 : i32
    %c64_i64 = arith.constant 64 : i64
    %c63_i64 = arith.constant 63 : i64
    %c40_i64 = arith.constant 40 : i64
    %c39_i64 = arith.constant 39 : i64
    %c32_i64 = arith.constant 32 : i64
    %c21_i64 = arith.constant 21 : i64
    %c20_i64 = arith.constant 20 : i64
    %c19_i64 = arith.constant 19 : i64
    %c18_i64 = arith.constant 18 : i64
    %c17_i64 = arith.constant 17 : i64
    %c16_i64 = arith.constant 16 : i64
    %c15_i64 = arith.constant 15 : i64
    %c14_i64 = arith.constant 14 : i64
    %c13_i64 = arith.constant 13 : i64
    %c12_i64 = arith.constant 12 : i64
    %c11_i64 = arith.constant 11 : i64
    %c10_i64 = arith.constant 10 : i64
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
    %alloc = memref.alloc() {cannir.arg_index = 0 : i64} : memref<1xbf16>
    %alloc_0 = memref.alloc() {cannir.arg_index = 1 : i64} : memref<1xbf16>
    %alloc_1 = memref.alloc() {cannir.arg_index = 2 : i64} : memref<1xbf16>
    %alloc_2 = memref.alloc() {cannir.arg_index = 3 : i64} : memref<1xbf16>
    %alloc_3 = memref.alloc() {cannir.arg_index = 4 : i64} : memref<1xbf16>
    %alloc_4 = memref.alloc() {cannir.arg_index = 5 : i64} : memref<1xf32>
    %alloc_5 = memref.alloc() {cannir.arg_index = 6 : i64} : memref<1xf32>
    %alloc_6 = memref.alloc() {cannir.arg_index = 7 : i64} : memref<1xf32>
    %alloc_7 = memref.alloc() {cannir.arg_index = 8 : i64} : memref<1xbf16>
    %alloc_8 = memref.alloc() {cannir.arg_index = 9 : i64} : memref<1xf32>
    %alloc_9 = memref.alloc() {cannir.arg_index = 10 : i64} : memref<1xbf16>
    %alloc_10 = memref.alloc() {cannir.arg_index = 11 : i64} : memref<1xbf16>
    %alloc_11 = memref.alloc() {cannir.arg_index = 12 : i64} : memref<1xbf16>
    %alloc_12 = memref.alloc() {cannir.arg_index = 13 : i64} : memref<1xbf16>
    %alloc_13 = memref.alloc() {cannir.arg_index = 14 : i64} : memref<1xf32>
    %alloc_14 = memref.alloc() {cannir.arg_index = 15 : i64} : memref<1xf32>
    %alloc_15 = memref.alloc() {cannir.arg_index = 17 : i64} : memref<1xi32>
    %alloc_16 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<8192xbf16, 1>
    %alloc_17 = memref.alloc() {cannir.byte_offset = 16384 : i64} : memref<8192xbf16, 1>
    %alloc_18 = memref.alloc() {cannir.byte_offset = 32768 : i64} : memref<4096xf32, 1>
    %alloc_19 = memref.alloc() {cannir.byte_offset = 49152 : i64} : memref<4096xbf16, 1>
    %alloc_20 = memref.alloc() {cannir.byte_offset = 57344 : i64} : memref<4096xf32, 1>
    %alloc_21 = memref.alloc() {cannir.byte_offset = 73728 : i64} : memref<64xf32, 1>
    %alloc_22 = memref.alloc() {cannir.byte_offset = 73984 : i64} : memref<1xf32, 1>
    %alloc_23 = memref.alloc() {cannir.byte_offset = 74016 : i64} : memref<64xbf16, 1>
    %alloc_24 = memref.alloc() {cannir.byte_offset = 74144 : i64} : memref<64xf32, 1>
    %alloc_25 = memref.alloc() {cannir.byte_offset = 74400 : i64} : memref<4096xbf16, 1>
    %alloc_26 = memref.alloc() {cannir.byte_offset = 82592 : i64} : memref<4096xbf16, 1>
    %alloc_27 = memref.alloc() {cannir.byte_offset = 90784 : i64} : memref<4096xbf16, 1>
    %alloc_28 = memref.alloc() {cannir.byte_offset = 98976 : i64} : memref<4096xf32, 1>
    %alloc_29 = memref.alloc() {cannir.byte_offset = 115360 : i64} : memref<64xf32, 1>
    %alloc_30 = memref.alloc() {cannir.byte_offset = 115616 : i64} : memref<64xf32, 1>
    %alloc_31 = memref.alloc() {cannir.byte_offset = 115872 : i64} : memref<4096xf32, 1>
    %alloc_32 = memref.alloc() {cannir.byte_offset = 132256 : i64} : memref<4096xf32, 1>
    %alloc_33 = memref.alloc() {cannir.byte_offset = 148640 : i64} : memref<2048xf32, 1>
    %alloc_34 = memref.alloc() {cannir.byte_offset = 156832 : i64} : memref<2048xf32, 1>
    %alloc_35 = memref.alloc() {cannir.byte_offset = 165024 : i64} : memref<4160xbf16, 1>
    %alloc_36 = memref.alloc() {cannir.byte_offset = 173344 : i64} : memref<4160xbf16, 1>
    %alloc_37 = memref.alloc() {cannir.byte_offset = 181664 : i64} : memref<4096xbf16, 1>
    %alloc_38 = memref.alloc() {cannir.byte_offset = 189856 : i64} : memref<2048xf16, 1>
    %alloc_39 = memref.alloc() {cannir.byte_offset = 193952 : i64} : memref<2048xbf16, 1>
    %alloc_40 = memref.alloc() {cannir.byte_offset = 198048 : i64} : memref<4096xf32, 1>
    %alloc_41 = memref.alloc() {cannir.byte_offset = 214432 : i64} : memref<4096xbf16, 1>
    %alloc_42 = memref.alloc() {cannir.byte_offset = 222624 : i64} : memref<4096xbf16, 1>
    %alloc_43 = memref.alloc() {cannir.byte_offset = 230816 : i64} : memref<2048xf32, 1>
    %alloc_44 = memref.alloc() {cannir.byte_offset = 239008 : i64} : memref<2112xf16, 1>
    %alloc_45 = memref.alloc() {cannir.byte_offset = 243232 : i64} : memref<1024xbf16, 1>
    %alloc_46 = memref.alloc() {cannir.byte_offset = 245280 : i64} : memref<1024xbf16, 1>
    %alloc_47 = memref.alloc() {cannir.byte_offset = 247328 : i64} : memref<1088xbf16, 1>
    %alloc_48 = memref.alloc() {cannir.byte_offset = 249504 : i64} : memref<1088xbf16, 1>
    %alloc_49 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<8192xf32, 1>
    %alloc_50 = memref.alloc() {cannir.byte_offset = 32768 : i64} : memref<8192xf32, 1>
    %alloc_51 = memref.alloc() {cannir.byte_offset = 65536 : i64} : memref<64xf32, 1>
    %alloc_52 = memref.alloc() {cannir.byte_offset = 65792 : i64} : memref<8192xf32, 1>
    %alloc_53 = memref.alloc() {cannir.byte_offset = 98560 : i64} : memref<8192xf32, 1>
    %alloc_54 = memref.alloc() {cannir.byte_offset = 131328 : i64} : memref<4096xbf16, 1>
    %alloc_55 = memref.alloc() {cannir.byte_offset = 139520 : i64} : memref<4096xf32, 1>
    %alloc_56 = memref.alloc() {cannir.byte_offset = 155904 : i64} : memref<4096xf32, 1>
    %alloc_57 = memref.alloc() {cannir.byte_offset = 98304 : i64} : memref<16384xf32, 2>
    %alloc_58 = memref.alloc() {cannir.byte_offset = 163840 : i64} : memref<16384xf32, 2>
    %alloc_59 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<8192xf32, 2>
    %alloc_60 = memref.alloc() {cannir.byte_offset = 32768 : i64} : memref<8192xf32, 2>
    %alloc_61 = memref.alloc() {cannir.byte_offset = 65536 : i64} : memref<8192xf32, 2>
    %alloc_62 = memref.alloc() {cannir.byte_offset = 77824 : i64} : memref<8192xbf16, 2>
    %alloc_63 = memref.alloc() {cannir.byte_offset = 94208 : i64} : memref<8192xbf16, 2>
    %alloc_64 = memref.alloc() {cannir.byte_offset = 32768 : i64} : memref<4096xf16, 2>
    %alloc_65 = memref.alloc() {cannir.byte_offset = 73728 : i64} : memref<2048xbf16, 2>
    %alloc_66 = memref.alloc() {cannir.byte_offset = 69632 : i64} : memref<2048xbf16, 2>
    %alloc_67 = memref.alloc() {cannir.byte_offset = 65536 : i64} : memref<2048xbf16, 2>
    %0 = ascscalar.get_block_idx
    %1 = ascscalar.get_block_num
    %2 = memref.load %alloc_15[%c2] : memref<1xi32>
    %3 = memref.load %alloc_15[%c3] : memref<1xi32>
    %4 = memref.load %alloc_15[%c5] : memref<1xi32>
    %5 = arith.extsi %4 : i32 to i64
    %6 = memref.load %alloc_15[%c6] : memref<1xi32>
    %7 = arith.extsi %6 : i32 to i64
    %8 = memref.load %alloc_15[%c0] : memref<1xi32>
    %9 = arith.cmpi eq, %8, %c1262764338_i32 : i32
    %10 = scf.if %9 -> (i1) {
      %11 = memref.load %alloc_15[%c1] : memref<1xi32>
      %12 = arith.cmpi eq, %11, %c2_i32 : i32
      %13 = scf.if %12 -> (i1) {
        %14 = arith.cmpi eq, %2, %c0_i32 : i32
        %15 = scf.if %14 -> (i1) {
          %16 = arith.extsi %3 : i32 to i64
          %17 = arith.cmpi eq, %16, %1 : i64
          %18 = scf.if %17 -> (i1) {
            %19 = cannir.extract_layout %arg1 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
            %20 = cannir.extract_shape %19 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 128)>
            %21 = cannir.extract_dim %20[1] : !cannir.shape<(-1, -1, -1, 128)>
            %22 = arith.cmpi eq, %7, %21 : i64
            %23 = arith.muli %5, %7 : i64
            %24 = arith.cmpi eq, %23, %arg18 : i64
            %25 = arith.select %22, %24, %false : i1
            scf.yield %25 : i1
          } else {
            scf.yield %false : i1
          }
          scf.yield %18 : i1
        } else {
          scf.yield %false : i1
        }
        scf.yield %15 : i1
      } else {
        scf.yield %false : i1
      }
      scf.yield %13 : i1
    } else {
      scf.yield %false : i1
    }
    scf.if %10 {
      ascvec.sync_intra_arrive v, %c15_i64
      ascvec.sync_intra_arrive v, %c14_i64
      ascvec.sync_intra_arrive v, %c4_i64
      ascvec.sync_intra_arrive v, %c3_i64
      ascvec.sync_intra_arrive v, %c2_i64
      %11 = memref.load %alloc_15[%c4] : memref<1xi32>
      scf.for %arg22 = %c0_i32 to %11 step %c1_i32  : i32 {
        %12 = arith.addi %arg22, %c9_i32 : i32
        %13 = arith.index_cast %12 : i32 to index
        %14 = memref.load %alloc_15[%13] : memref<1xi32>
        %15 = arith.addi %14, %c1_i32 : i32
        %16 = arith.index_cast %15 : i32 to index
        %17 = memref.load %alloc_15[%16] : memref<1xi32>
        %18 = arith.addi %14, %c2_i32 : i32
        %19 = arith.addi %18, %17 : i32
        %20 = arith.addi %19, %17 : i32
        %21 = arith.addi %20, %17 : i32
        %22 = arith.addi %21, %17 : i32
        %23 = arith.addi %22, %c1_i32 : i32
        %24 = arith.addi %23, %17 : i32
        %25 = arith.addi %24, %17 : i32
        %26 = arith.addi %25, %17 : i32
        %27 = arith.addi %26, %c1_i32 : i32
        %28 = arith.muli %0, %c2_i64 : i64
        %29 = arith.extsi %27 : i32 to i64
        %30 = arith.addi %29, %28 : i64
        %31 = arith.index_cast %30 : i64 to index
        %32 = memref.load %alloc_15[%31] : memref<1xi32>
        %33 = arith.addi %30, %c1_i64 : i64
        %34 = arith.index_cast %33 : i64 to index
        %35 = memref.load %alloc_15[%34] : memref<1xi32>
        scf.for %arg23 = %c0_i32 to %17 step %c1_i32  : i32 {
          %49 = arith.addi %25, %arg23 : i32
          %50 = arith.index_cast %49 : i32 to index
          %51 = memref.load %alloc_15[%50] : memref<1xi32>
          %52 = arith.addi %49, %c1_i32 : i32
          %53 = arith.index_cast %52 : i32 to index
          %54 = memref.load %alloc_15[%53] : memref<1xi32>
          %55 = arith.cmpi slt, %32, %51 : i32
          %56 = arith.select %55, %51, %32 : i32
          %57 = arith.cmpi sgt, %35, %54 : i32
          %58 = arith.select %57, %54, %35 : i32
          %59 = arith.cmpi slt, %56, %58 : i32
          %60 = arith.subi %56, %51 : i32
          %61 = arith.subi %58, %51 : i32
          %62 = arith.extsi %60 : i32 to i64
          %63 = arith.extsi %61 : i32 to i64
          %64 = arith.select %59, %62, %c0_i64 : i64
          %65 = arith.select %59, %63, %c0_i64 : i64
          %66 = arith.addi %18, %arg23 : i32
          %67 = arith.index_cast %66 : i32 to index
          %68 = memref.load %alloc_15[%67] : memref<1xi32>
          %69 = arith.addi %19, %arg23 : i32
          %70 = arith.index_cast %69 : i32 to index
          %71 = memref.load %alloc_15[%70] : memref<1xi32>
          %72 = arith.addi %20, %arg23 : i32
          %73 = arith.index_cast %72 : i32 to index
          %74 = memref.load %alloc_15[%73] : memref<1xi32>
          %75 = arith.addi %21, %arg23 : i32
          %76 = arith.index_cast %75 : i32 to index
          %77 = memref.load %alloc_15[%76] : memref<1xi32>
          %78 = arith.extsi %77 : i32 to i64
          %79 = arith.addi %23, %arg23 : i32
          %80 = arith.index_cast %79 : i32 to index
          %81 = memref.load %alloc_15[%80] : memref<1xi32>
          %82 = arith.addi %24, %arg23 : i32
          %83 = arith.index_cast %82 : i32 to index
          %84 = memref.load %alloc_15[%83] : memref<1xi32>
          %85 = arith.addi %78, %c63_i64 : i64
          %86 = arith.remsi %85, %c64_i64 : i64
          %87 = arith.cmpi ne, %86, %c0_i64 : i64
          %88 = arith.cmpi slt, %86, %c0_i64 : i64
          %89 = arith.andi %87, %88 : i1
          %90 = arith.divsi %85, %c64_i64 : i64
          %91 = arith.extui %89 : i1 to i64
          %92 = arith.subi %90, %91 : i64
          %93 = arith.subi %92, %c1_i64 : i64
          %94 = arith.muli %93, %c64_i64 : i64
          %95 = arith.subi %78, %94 : i64
          %96 = arith.extsi %74 : i32 to i64
          %97 = arith.extsi %71 : i32 to i64
          %98 = arith.extsi %68 : i32 to i64
          %99 = arith.addi %81, %84 : i32
          %100 = arith.extsi %99 : i32 to i64
          %101 = arith.cmpi eq, %100, %92 : i64
          %102 = ascscalar.get_subblock_id
          %103 = arith.cmpi slt, %64, %65 : i64
          scf.if %103 {
            %104 = ascscalar.bitcast_memref %alloc_16 byte_offset = %c0_i64 : memref<8192xbf16, 1> -> memref<?xbf16, 1>
            %105 = ascscalar.bitcast_memref %alloc_17 byte_offset = %c0_i64 : memref<8192xbf16, 1> -> memref<?xbf16, 1>
            %106 = ascscalar.bitcast_memref %alloc_19 byte_offset = %c0_i64 : memref<4096xbf16, 1> -> memref<?xbf16, 1>
            %107 = ascscalar.bitcast_memref %alloc_21 byte_offset = %c0_i64 : memref<64xf32, 1> -> memref<?xf32, 1>
            %108 = ascscalar.bitcast_memref %alloc_22 byte_offset = %c0_i64 : memref<1xf32, 1> -> memref<?xf32, 1>
            %109 = ascscalar.bitcast_memref %alloc_23 byte_offset = %c0_i64 : memref<64xbf16, 1> -> memref<?xbf16, 1>
            %110 = ascscalar.bitcast_memref %alloc_27 byte_offset = %c0_i64 : memref<4096xbf16, 1> -> memref<?xbf16, 1>
            %111 = cannir.alloc_array : <2xi64>
            %112 = cannir.alloc_array : <2xi64>
            %113:6 = scf.for %arg24 = %64 to %65 step %c1_i64 iter_args(%arg25 = %c0_i64, %arg26 = %c0_i64, %arg27 = %c0_i64, %arg28 = %c0_i64, %arg29 = %c0_i64, %arg30 = %c0_i64) -> (i64, i64, i64, i64, i64, i64)  : i64 {
              %118 = arith.cmpi sge, %arg25, %c1_i64 : i64
              %119 = arith.subi %arg25, %c1_i64 : i64
              %120 = arith.cmpi slt, %119, %arg25 : i64
              %121 = arith.select %118, %120, %false : i1
              %122:2 = scf.if %121 -> (i64, i64) {
                %297 = arith.remsi %119, %c2_i64 : i64
                %298 = arith.index_cast %297 : i64 to index
                %299 = cannir.array_load %111[%298] : <2xi64>
                %300 = arith.extsi %84 : i32 to i64
                %301 = arith.remsi %299, %300 : i64
                %302 = arith.cmpi ne, %301, %c0_i64 : i64
                %303 = arith.cmpi slt, %301, %c0_i64 : i64
                %304 = arith.cmpi slt, %300, %c0_i64 : i64
                %305 = arith.xori %303, %304 : i1
                %306 = arith.andi %302, %305 : i1
                %307 = arith.divsi %299, %300 : i64
                %308 = arith.extui %306 : i1 to i64
                %309 = arith.subi %307, %308 : i64
                %310 = arith.muli %309, %300 : i64
                %311 = arith.subi %299, %310 : i64
                %312 = arith.extsi %81 : i32 to i64
                %313 = arith.addi %312, %311 : i64
                %314 = cannir.extract_layout %arg1 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                %315 = cannir.extract_shape %314 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 128)>
                %316 = cannir.extract_dim %315[1] : !cannir.shape<(-1, -1, -1, 128)>
                %317 = arith.maxsi %316, %c1_i64 : i64
                %318 = arith.remsi %309, %317 : i64
                %319 = arith.cmpi slt, %318, %c0_i64 : i64
                %320 = arith.select %319, %317, %c0_i64 : i64
                %321 = arith.addi %318, %320 : i64
                %322 = arith.remsi %299, %c2_i64 : i64
                %323 = arith.cmpi ne, %322, %c0_i64 : i64
                %324 = arith.cmpi slt, %322, %c0_i64 : i64
                %325 = arith.andi %323, %324 : i1
                %326 = arith.select %325, %c2_i64, %c0_i64 : i64
                %327 = arith.addi %322, %326 overflow<nsw> : i64
                %328 = cannir.array_load %112[%298] : <2xi64>
                %329 = arith.muli %102, %c32_i64 : i64
                ascvec.sync_intra_wait v, %c4_i64 {cannir.sync.banks = 3 : i32}
                ascvec.vf(%alloc_43, %329, %alloc_24, %alloc_38 : memref<2048xf32, 1>, i64, memref<64xf32, 1>, memref<2048xf16, 1>) {
                ^bb0(%arg31: memref<2048xf32, 1>, %arg32: i64, %arg33: memref<64xf32, 1>, %arg34: memref<2048xf16, 1>):
                  %c1_i32_68 = arith.constant 1 : i32
                  %c32_i32_69 = arith.constant 32 : i32
                  %c0_i32_70 = arith.constant 0 : i32
                  %cst = arith.constant -1.000000e+00 : f32
                  %cst_71 = arith.constant 0.000000e+00 : f32
                  %c64_i32_72 = arith.constant 64 : i32
                  %mask, %remaining_out = ascvec.update_mask %c64_i32_72 : !ascvec.mask
                  %367 = ascvec.duplicate_scalar %cst_71, %mask : f32 -> <f32>
                  scf.for %arg35 = %c0_i32_70 to %c32_i32_69 step %c1_i32_68  : i32 {
                    %368 = arith.muli %arg35, %c64_i32_72 overflow<nsw> : i32
                    %369 = arith.index_cast %368 : i32 to index
                    %370 = ascvec.loadalign %arg31[%369] : memref<2048xf32, 1>, index -> <f32>
                    %371 = arith.trunci %arg32 : i64 to i32
                    %372 = arith.addi %371, %arg35 : i32
                    %373 = arith.index_cast %372 : i32 to index
                    %374 = ascvec.loadalign_brc %arg33[%373] : memref<64xf32, 1>, index -> <f32>
                    %375 = ascvec.mul %370, %374, %mask : <f32>
                    %mask_73, %remaining_out_74 = ascvec.update_mask %372 : !ascvec.mask
                    %376 = ascvec.select %375, %367, %mask_73 : <f32>
                    %377 = ascvec.mul_scalar %376, %cst, %mask : <f32>, f32
                    %378 = ascvec.cast %377, %mask : <f32> -> <f16>
                    ascvec.storealign_pack %arg34[%369], %378, %mask {mode = "b32_to_b16"} : memref<2048xf16, 1>, index, <f16>
                  } {cannir.loop_unroll = 4 : i64}
                } {cannir.vf_mode = "simd"}
                ascvec.sync_intra_arrive v, %c4_i64 {cannir.sync.banks = 3 : i32}
                ascvec.get_buf v, %c14_i64, 0 {cannir.buffer.generated_sync}
                ascvec.vf(%alloc_38, %alloc_44 : memref<2048xf16, 1>, memref<2112xf16, 1>) {
                ^bb0(%arg31: memref<2048xf16, 1>, %arg32: memref<2112xf16, 1>):
                  %c1_i32_68 = arith.constant 1 : i32
                  %c32_i32_69 = arith.constant 32 : i32
                  %c0_i32_70 = arith.constant 0 : i32
                  %c33 = arith.constant 33 : index
                  %c16_i32_71 = arith.constant 16 : i32
                  %c0_72 = arith.constant 0 : index
                  %c64_i32_73 = arith.constant 64 : i32
                  scf.for %arg33 = %c0_i32_70 to %c32_i32_69 step %c1_i32_68  : i32 {
                    %mask, %remaining_out = ascvec.update_mask %c64_i32_73 elem_bits 16 : !ascvec.mask
                    %367 = arith.muli %arg33, %c64_i32_73 overflow<nsw> : i32
                    %368 = arith.index_cast %367 : i32 to index
                    %369 = ascvec.loadalign %arg31[%368] : memref<2048xf16, 1>, index -> <f16>
                    %370 = arith.muli %arg33, %c16_i32_71 overflow<nsw> : i32
                    %371 = arith.index_cast %370 : i32 to index
                    ascvec.storealign_strided %arg32[%371], %369, %mask, %c33, %c0_72 : memref<2112xf16, 1>, <f16>
                  }
                } {cannir.vf_mode = "simd"}
                ascvec.release_buf v, %c14_i64, 0 {cannir.buffer.generated_sync}
                %330 = ascscalar.bitcast_memref %alloc_44 byte_offset = %c0_i64 : memref<2112xf16, 1> -> memref<?xf16, 1>
                %331 = ascscalar.bitcast_memref %alloc_64 byte_offset = %c0_i64 : memref<4096xf16, 2> -> memref<?xf16, 2>
                %332 = arith.index_cast %329 : i64 to index
                %333 = arith.muli %332, %c16 : index
                ascvec.sync_intra_wait mte3, %c1_i64 {cannir.sync.banks = 3 : i32}
                ascvec.get_buf mte3, %c14_i64, 0 {cannir.buffer.generated_sync}
                ascvec.copy_ub2l1_strided %331[%333], %330[%c0], %c4_i32, %c32_i32, %c1_i32, %c32_i32 : memref<?xf16, 2>, memref<?xf16, 1>
                ascvec.release_buf mte3, %c14_i64, 0 {cannir.buffer.generated_sync}
                ascvec.sync_intra_arrive mte3, %c1_i64 {cannir.sync.banks = 3 : i32}
                %334 = arith.cmpi eq, %327, %c0_i64 : i64
                %335 = ascscalar.memref_address %alloc_41 : memref<4096xbf16, 1>
                %336 = ascscalar.memref_address %alloc_42 : memref<4096xbf16, 1>
                %337 = arith.select %334, %335, %336 : i64
                %338 = ascscalar.bind_local_ptr %337 : memref<4096xbf16, 1>
                %339 = arith.cmpi slt, %328, %c64_i64 : i64
                %340 = scf.if %339 -> (i64) {
                  %367 = arith.divsi %78, %c64_i64 : i64
                  %368 = arith.minsi %313, %367 : i64
                  %369 = arith.muli %368, %c64_i64 : i64
                  %370 = arith.subi %78, %369 : i64
                  %371 = arith.minsi %370, %c64_i64 : i64
                  %372 = arith.cmpi sle, %313, %367 : i64
                  %373 = arith.select %372, %371, %c0_i64 : i64
                  %374 = arith.trunci %373 : i64 to i32
                  ascvec.vf(%alloc_27, %374 : memref<4096xbf16, 1>, i32) {
                  ^bb0(%arg31: memref<4096xbf16, 1>, %arg32: i32):
                    %c1_i32_68 = arith.constant 1 : i32
                    %cst = arith.constant 0.000000e+00 : bf16
                    %c64_i32_69 = arith.constant 64 : i32
                    %mask, %remaining_out = ascvec.update_mask %c64_i32_69 elem_bits 16 : !ascvec.mask
                    %415 = ascvec.duplicate_scalar %cst, %mask : bf16 -> <bf16>
                    scf.for %arg33 = %arg32 to %c64_i32_69 step %c1_i32_68  : i32 {
                      %416 = arith.muli %arg33, %c64_i32_69 : i32
                      %417 = arith.index_cast %416 : i32 to index
                      ascvec.storealign %arg31[%417], %415, %mask : memref<4096xbf16, 1>, index, <bf16>
                    } {cannir.loop_unroll = 4 : i64}
                    ascvec.mem_bar vst_vld
                  } {cannir.vf_mode = "simd"}
                  ascvec.sync_notify {dst_pipe = 5 : i32, event_id = 0 : i32, src_pipe = 2 : i32}
                  ascvec.sync_wait {dst_pipe = 5 : i32, event_id = 0 : i32, src_pipe = 2 : i32}
                  %375 = cannir.extract_physical_layout %arg1 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                  %376 = cannir.extract_stride %375 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                  %377 = cannir.extract_dim %376[2] : !cannir.stride<(-1, -1, -1, 1)>
                  %378 = arith.muli %377, %c2_i64 : i64
                  %379 = cannir.extract_dim %376[0] : !cannir.stride<(-1, -1, -1, 1)>
                  %380 = arith.muli %97, %379 : i64
                  %381 = arith.muli %380, %c2_i64 : i64
                  %382 = cannir.extract_dim %376[1] : !cannir.stride<(-1, -1, -1, 1)>
                  %383 = arith.muli %321, %382 : i64
                  %384 = arith.muli %383, %c2_i64 : i64
                  %385 = arith.addi %381, %384 : i64
                  %386 = arith.muli %102, %c128_i64 : i64
                  %387 = arith.addi %385, %386 : i64
                  %388 = arith.muli %313, %c64_i64 : i64
                  %389 = arith.addi %388, %96 : i64
                  %390 = arith.muli %389, %377 : i64
                  %391 = arith.muli %390, %c2_i64 : i64
                  %392 = arith.addi %387, %391 : i64
                  "ascvec.copy_gm2ub_plan"(%110, %alloc_0, %c0_i32, %c0_i32, %c0_i64, %392, %373, %c128_i64, %378, %c128_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = 90784 : i64, ub_capacity = 8192 : i64}> : (memref<?xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                  ascvec.sync_notify {dst_pipe = 2 : i32, event_id = 1 : i32, src_pipe = 5 : i32}
                  ascvec.get_buf v, %c12_i64, 0 {cannir.buffer.generated_sync}
                  ascvec.vf(%alloc_24, %alloc_28, %338, %alloc_37 : memref<64xf32, 1>, memref<4096xf32, 1>, memref<4096xbf16, 1>, memref<4096xbf16, 1>) {
                  ^bb0(%arg31: memref<64xf32, 1>, %arg32: memref<4096xf32, 1>, %arg33: memref<4096xbf16, 1>, %arg34: memref<4096xbf16, 1>):
                    %c1_i32_68 = arith.constant 1 : i32
                    %c0_i32_69 = arith.constant 0 : i32
                    %cst = arith.constant -1.000000e+00 : f32
                    %c64_i32_70 = arith.constant 64 : i32
                    %mask, %remaining_out = ascvec.update_mask %c64_i32_70 : !ascvec.mask
                    scf.for %arg35 = %c0_i32_69 to %c64_i32_70 step %c1_i32_68  : i32 {
                      %415 = arith.muli %arg35, %c64_i32_70 overflow<nsw> : i32
                      %416 = arith.index_cast %arg35 : i32 to index
                      %417 = ascvec.loadalign_brc %arg31[%416] : memref<64xf32, 1>, index -> <f32>
                      %418 = ascvec.mul_scalar %417, %cst, %mask : <f32>, f32
                      %419 = arith.index_cast %415 : i32 to index
                      %420 = ascvec.loadalign %arg32[%419] : memref<4096xf32, 1>, index -> <f32>
                      %421 = ascvec.loadalign_unpack %arg33[%419] {mode = "b16_to_b32"} : memref<4096xbf16, 1>, index -> <bf16>
                      %422 = ascvec.cast %421, %mask {rounding = 6 : i32} : <bf16> -> <f32>
                      %423 = ascvec.exp %420, %mask : <f32>
                      %424 = ascvec.mul %422, %423, %mask : <f32>
                      %425 = ascvec.mul %424, %418, %mask : <f32>
                      %426 = ascvec.cast %425, %mask : <f32> -> <bf16>
                      ascvec.storealign_pack %arg34[%419], %426, %mask {mode = "b32_to_b16"} : memref<4096xbf16, 1>, index, <bf16>
                    } {cannir.loop_unroll = 4 : i64}
                  } {cannir.vf_mode = "simd"}
                  %393 = arith.cmpi eq, %arg30, %c0_i64 : i64
                  %394 = ascscalar.memref_address %alloc_35 : memref<4160xbf16, 1>
                  %395 = ascscalar.memref_address %alloc_36 : memref<4160xbf16, 1>
                  %396 = arith.select %393, %394, %395 : i64
                  %397 = ascscalar.bind_local_ptr %396 : memref<4160xbf16, 1>
                  %398 = arith.select %393, %c10_i64, %c11_i64 : i64
                  %399 = arith.addi %arg30, %c1_i64 overflow<nsw> : i64
                  %400 = arith.remui %399, %c2_i64 : i64
                  ascvec.get_buf v, %398, 0 {cannir.buffer.generated_sync}
                  ascvec.vf(%alloc_37, %397 : memref<4096xbf16, 1>, memref<4160xbf16, 1>) {
                  ^bb0(%arg31: memref<4096xbf16, 1>, %arg32: memref<4160xbf16, 1>):
                    %c1_i32_68 = arith.constant 1 : i32
                    %c0_i32_69 = arith.constant 0 : i32
                    %c65 = arith.constant 65 : index
                    %c16_i32_70 = arith.constant 16 : i32
                    %c0_71 = arith.constant 0 : index
                    %c64_i32_72 = arith.constant 64 : i32
                    scf.for %arg33 = %c0_i32_69 to %c64_i32_72 step %c1_i32_68  : i32 {
                      %mask, %remaining_out = ascvec.update_mask %c64_i32_72 elem_bits 16 : !ascvec.mask
                      %415 = arith.muli %arg33, %c64_i32_72 overflow<nsw> : i32
                      %416 = arith.index_cast %415 : i32 to index
                      %417 = ascvec.loadalign %arg31[%416] : memref<4096xbf16, 1>, index -> <bf16>
                      %418 = arith.muli %arg33, %c16_i32_70 overflow<nsw> : i32
                      %419 = arith.index_cast %418 : i32 to index
                      ascvec.storealign_strided %arg32[%419], %417, %mask, %c65, %c0_71 : memref<4160xbf16, 1>, <bf16>
                    }
                  } {cannir.vf_mode = "simd"}
                  ascvec.release_buf v, %398, 0 {cannir.buffer.generated_sync}
                  ascvec.release_buf v, %c12_i64, 0 {cannir.buffer.generated_sync}
                  %401 = ascscalar.bind_local_ptr %396 : memref<1xbf16, 1>
                  %402 = ascscalar.bitcast_memref %alloc_63 byte_offset = %c0_i64 : memref<8192xbf16, 2> -> memref<?xbf16, 2>
                  %403 = arith.muli %102, %c64_i64 : i64
                  %404 = arith.index_cast %403 : i64 to index
                  %405 = arith.divui %404, %c16 : index
                  %406 = arith.muli %405, %c1024 : index
                  ascvec.sync_intra_wait mte3, %c9_i64 {cannir.sync.banks = 3 : i32}
                  ascvec.get_buf mte3, %398, 0 {cannir.buffer.generated_sync}
                  ascvec.copy_ub2l1_strided %402[%406], %401[%c0], %c4_i32, %c64_i32, %c1_i32, %c0_i32 : memref<?xbf16, 2>, memref<1xbf16, 1>
                  ascvec.release_buf mte3, %398, 0 {cannir.buffer.generated_sync}
                  ascvec.sync_intra_arrive mte3, %c9_i64 {cannir.sync.banks = 3 : i32}
                  ascvec.sync_wait {dst_pipe = 2 : i32, event_id = 1 : i32, src_pipe = 5 : i32}
                  ascvec.vf(%alloc_24, %alloc_27, %alloc_37 : memref<64xf32, 1>, memref<4096xbf16, 1>, memref<4096xbf16, 1>) {
                  ^bb0(%arg31: memref<64xf32, 1>, %arg32: memref<4096xbf16, 1>, %arg33: memref<4096xbf16, 1>):
                    %c1_i32_68 = arith.constant 1 : i32
                    %c0_i32_69 = arith.constant 0 : i32
                    %c64_i32_70 = arith.constant 64 : i32
                    %mask, %remaining_out = ascvec.update_mask %c64_i32_70 : !ascvec.mask
                    scf.for %arg34 = %c0_i32_69 to %c64_i32_70 step %c1_i32_68  : i32 {
                      %415 = arith.muli %arg34, %c64_i32_70 overflow<nsw> : i32
                      %416 = arith.index_cast %arg34 : i32 to index
                      %417 = ascvec.loadalign_brc %arg31[%416] : memref<64xf32, 1>, index -> <f32>
                      %418 = arith.index_cast %415 : i32 to index
                      %419 = ascvec.loadalign_unpack %arg32[%418] {mode = "b16_to_b32"} : memref<4096xbf16, 1>, index -> <bf16>
                      %420 = ascvec.cast %419, %mask {rounding = 6 : i32} : <bf16> -> <f32>
                      %421 = ascvec.mul %420, %417, %mask : <f32>
                      %422 = ascvec.cast %421, %mask : <f32> -> <bf16>
                      ascvec.storealign_pack %arg33[%418], %422, %mask {mode = "b32_to_b16"} : memref<4096xbf16, 1>, index, <bf16>
                    } {cannir.loop_unroll = 4 : i64}
                  } {cannir.vf_mode = "simd"}
                  %407 = arith.cmpi eq, %400, %c0_i64 : i64
                  %408 = arith.select %407, %394, %395 : i64
                  %409 = ascscalar.bind_local_ptr %408 : memref<4160xbf16, 1>
                  %410 = arith.select %407, %c10_i64, %c11_i64 : i64
                  %411 = arith.addi %400, %c1_i64 overflow<nsw> : i64
                  %412 = arith.remui %411, %c2_i64 : i64
                  ascvec.get_buf v, %410, 0 {cannir.buffer.generated_sync}
                  ascvec.vf(%alloc_37, %409 : memref<4096xbf16, 1>, memref<4160xbf16, 1>) {
                  ^bb0(%arg31: memref<4096xbf16, 1>, %arg32: memref<4160xbf16, 1>):
                    %c1_i32_68 = arith.constant 1 : i32
                    %c0_i32_69 = arith.constant 0 : i32
                    %c65 = arith.constant 65 : index
                    %c16_i32_70 = arith.constant 16 : i32
                    %c0_71 = arith.constant 0 : index
                    %c64_i32_72 = arith.constant 64 : i32
                    scf.for %arg33 = %c0_i32_69 to %c64_i32_72 step %c1_i32_68  : i32 {
                      %mask, %remaining_out = ascvec.update_mask %c64_i32_72 elem_bits 16 : !ascvec.mask
                      %415 = arith.muli %arg33, %c64_i32_72 overflow<nsw> : i32
                      %416 = arith.index_cast %415 : i32 to index
                      %417 = ascvec.loadalign %arg31[%416] : memref<4096xbf16, 1>, index -> <bf16>
                      %418 = arith.muli %arg33, %c16_i32_70 overflow<nsw> : i32
                      %419 = arith.index_cast %418 : i32 to index
                      ascvec.storealign_strided %arg32[%419], %417, %mask, %c65, %c0_71 : memref<4160xbf16, 1>, <bf16>
                    }
                  } {cannir.vf_mode = "simd"}
                  ascvec.release_buf v, %410, 0 {cannir.buffer.generated_sync}
                  %413 = ascscalar.bind_local_ptr %408 : memref<1xbf16, 1>
                  %414 = ascscalar.bitcast_memref %alloc_62 byte_offset = %c0_i64 : memref<8192xbf16, 2> -> memref<?xbf16, 2>
                  ascvec.sync_intra_wait mte3, %c8_i64 {cannir.sync.banks = 3 : i32}
                  ascvec.get_buf mte3, %410, 0 {cannir.buffer.generated_sync}
                  ascvec.copy_ub2l1_strided %414[%406], %413[%c0], %c4_i32, %c64_i32, %c1_i32, %c0_i32 : memref<?xbf16, 2>, memref<1xbf16, 1>
                  ascvec.release_buf mte3, %410, 0 {cannir.buffer.generated_sync}
                  ascvec.sync_intra_arrive mte3, %c8_i64 {cannir.sync.banks = 3 : i32}
                  scf.yield %412 : i64
                } else {
                  %367 = arith.divsi %78, %c64_i64 : i64
                  %368 = arith.minsi %313, %367 : i64
                  %369 = arith.muli %368, %c64_i64 : i64
                  %370 = arith.subi %78, %369 : i64
                  %371 = arith.minsi %370, %c64_i64 : i64
                  %372 = arith.cmpi sle, %313, %367 : i64
                  %373 = arith.select %372, %371, %c0_i64 : i64
                  %374 = cannir.extract_physical_layout %arg1 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                  %375 = cannir.extract_stride %374 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                  %376 = cannir.extract_dim %375[2] : !cannir.stride<(-1, -1, -1, 1)>
                  %377 = arith.muli %376, %c2_i64 : i64
                  %378 = cannir.extract_dim %375[0] : !cannir.stride<(-1, -1, -1, 1)>
                  %379 = arith.muli %97, %378 : i64
                  %380 = arith.muli %379, %c2_i64 : i64
                  %381 = cannir.extract_dim %375[1] : !cannir.stride<(-1, -1, -1, 1)>
                  %382 = arith.muli %321, %381 : i64
                  %383 = arith.muli %382, %c2_i64 : i64
                  %384 = arith.addi %380, %383 : i64
                  %385 = arith.muli %102, %c128_i64 : i64
                  %386 = arith.addi %384, %385 : i64
                  %387 = arith.muli %313, %c64_i64 : i64
                  %388 = arith.addi %387, %96 : i64
                  %389 = arith.muli %388, %376 : i64
                  %390 = arith.muli %389, %c2_i64 : i64
                  %391 = arith.addi %386, %390 : i64
                  ascvec.get_buf mte2, %c7_i64, 0 {cannir.buffer.generated_sync}
                  "ascvec.copy_gm2ub_plan"(%110, %alloc_0, %c0_i32, %c0_i32, %c0_i64, %391, %373, %c128_i64, %377, %c128_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = 90784 : i64, ub_capacity = 8192 : i64}> : (memref<?xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                  ascvec.release_buf mte2, %c7_i64, 0 {cannir.buffer.generated_sync}
                  ascvec.sync_notify {dst_pipe = 2 : i32, event_id = 1 : i32, src_pipe = 5 : i32}
                  ascvec.get_buf v, %c12_i64, 0 {cannir.buffer.generated_sync}
                  ascvec.vf(%alloc_24, %alloc_28, %338, %alloc_37 : memref<64xf32, 1>, memref<4096xf32, 1>, memref<4096xbf16, 1>, memref<4096xbf16, 1>) {
                  ^bb0(%arg31: memref<64xf32, 1>, %arg32: memref<4096xf32, 1>, %arg33: memref<4096xbf16, 1>, %arg34: memref<4096xbf16, 1>):
                    %c1_i32_68 = arith.constant 1 : i32
                    %c0_i32_69 = arith.constant 0 : i32
                    %cst = arith.constant -1.000000e+00 : f32
                    %c64_i32_70 = arith.constant 64 : i32
                    %mask, %remaining_out = ascvec.update_mask %c64_i32_70 : !ascvec.mask
                    scf.for %arg35 = %c0_i32_69 to %c64_i32_70 step %c1_i32_68  : i32 {
                      %414 = arith.muli %arg35, %c64_i32_70 overflow<nsw> : i32
                      %415 = arith.index_cast %arg35 : i32 to index
                      %416 = ascvec.loadalign_brc %arg31[%415] : memref<64xf32, 1>, index -> <f32>
                      %417 = ascvec.mul_scalar %416, %cst, %mask : <f32>, f32
                      %418 = arith.index_cast %414 : i32 to index
                      %419 = ascvec.loadalign %arg32[%418] : memref<4096xf32, 1>, index -> <f32>
                      %420 = ascvec.loadalign_unpack %arg33[%418] {mode = "b16_to_b32"} : memref<4096xbf16, 1>, index -> <bf16>
                      %421 = ascvec.cast %420, %mask {rounding = 6 : i32} : <bf16> -> <f32>
                      %422 = ascvec.exp %419, %mask : <f32>
                      %423 = ascvec.mul %421, %422, %mask : <f32>
                      %424 = ascvec.mul %423, %417, %mask : <f32>
                      %425 = ascvec.cast %424, %mask : <f32> -> <bf16>
                      ascvec.storealign_pack %arg34[%418], %425, %mask {mode = "b32_to_b16"} : memref<4096xbf16, 1>, index, <bf16>
                    } {cannir.loop_unroll = 4 : i64}
                  } {cannir.vf_mode = "simd"}
                  %392 = arith.cmpi eq, %arg30, %c0_i64 : i64
                  %393 = ascscalar.memref_address %alloc_35 : memref<4160xbf16, 1>
                  %394 = ascscalar.memref_address %alloc_36 : memref<4160xbf16, 1>
                  %395 = arith.select %392, %393, %394 : i64
                  %396 = ascscalar.bind_local_ptr %395 : memref<4160xbf16, 1>
                  %397 = arith.select %392, %c10_i64, %c11_i64 : i64
                  %398 = arith.addi %arg30, %c1_i64 overflow<nsw> : i64
                  %399 = arith.remui %398, %c2_i64 : i64
                  ascvec.get_buf v, %397, 0 {cannir.buffer.generated_sync}
                  ascvec.vf(%alloc_37, %396 : memref<4096xbf16, 1>, memref<4160xbf16, 1>) {
                  ^bb0(%arg31: memref<4096xbf16, 1>, %arg32: memref<4160xbf16, 1>):
                    %c1_i32_68 = arith.constant 1 : i32
                    %c0_i32_69 = arith.constant 0 : i32
                    %c65 = arith.constant 65 : index
                    %c16_i32_70 = arith.constant 16 : i32
                    %c0_71 = arith.constant 0 : index
                    %c64_i32_72 = arith.constant 64 : i32
                    scf.for %arg33 = %c0_i32_69 to %c64_i32_72 step %c1_i32_68  : i32 {
                      %mask, %remaining_out = ascvec.update_mask %c64_i32_72 elem_bits 16 : !ascvec.mask
                      %414 = arith.muli %arg33, %c64_i32_72 overflow<nsw> : i32
                      %415 = arith.index_cast %414 : i32 to index
                      %416 = ascvec.loadalign %arg31[%415] : memref<4096xbf16, 1>, index -> <bf16>
                      %417 = arith.muli %arg33, %c16_i32_70 overflow<nsw> : i32
                      %418 = arith.index_cast %417 : i32 to index
                      ascvec.storealign_strided %arg32[%418], %416, %mask, %c65, %c0_71 : memref<4160xbf16, 1>, <bf16>
                    }
                  } {cannir.vf_mode = "simd"}
                  ascvec.release_buf v, %397, 0 {cannir.buffer.generated_sync}
                  ascvec.release_buf v, %c12_i64, 0 {cannir.buffer.generated_sync}
                  %400 = ascscalar.bind_local_ptr %395 : memref<1xbf16, 1>
                  %401 = ascscalar.bitcast_memref %alloc_63 byte_offset = %c0_i64 : memref<8192xbf16, 2> -> memref<?xbf16, 2>
                  %402 = arith.muli %102, %c64_i64 : i64
                  %403 = arith.index_cast %402 : i64 to index
                  %404 = arith.divui %403, %c16 : index
                  %405 = arith.muli %404, %c1024 : index
                  ascvec.sync_intra_wait mte3, %c9_i64 {cannir.sync.banks = 3 : i32}
                  ascvec.get_buf mte3, %397, 0 {cannir.buffer.generated_sync}
                  ascvec.copy_ub2l1_strided %401[%405], %400[%c0], %c4_i32, %c64_i32, %c1_i32, %c0_i32 : memref<?xbf16, 2>, memref<1xbf16, 1>
                  ascvec.release_buf mte3, %397, 0 {cannir.buffer.generated_sync}
                  ascvec.sync_intra_arrive mte3, %c9_i64 {cannir.sync.banks = 3 : i32}
                  ascvec.sync_wait {dst_pipe = 2 : i32, event_id = 1 : i32, src_pipe = 5 : i32}
                  ascvec.vf(%alloc_24, %alloc_27, %alloc_37 : memref<64xf32, 1>, memref<4096xbf16, 1>, memref<4096xbf16, 1>) {
                  ^bb0(%arg31: memref<64xf32, 1>, %arg32: memref<4096xbf16, 1>, %arg33: memref<4096xbf16, 1>):
                    %c1_i32_68 = arith.constant 1 : i32
                    %c0_i32_69 = arith.constant 0 : i32
                    %c64_i32_70 = arith.constant 64 : i32
                    %mask, %remaining_out = ascvec.update_mask %c64_i32_70 : !ascvec.mask
                    scf.for %arg34 = %c0_i32_69 to %c64_i32_70 step %c1_i32_68  : i32 {
                      %414 = arith.muli %arg34, %c64_i32_70 overflow<nsw> : i32
                      %415 = arith.index_cast %arg34 : i32 to index
                      %416 = ascvec.loadalign_brc %arg31[%415] : memref<64xf32, 1>, index -> <f32>
                      %417 = arith.index_cast %414 : i32 to index
                      %418 = ascvec.loadalign_unpack %arg32[%417] {mode = "b16_to_b32"} : memref<4096xbf16, 1>, index -> <bf16>
                      %419 = ascvec.cast %418, %mask {rounding = 6 : i32} : <bf16> -> <f32>
                      %420 = ascvec.mul %419, %416, %mask : <f32>
                      %421 = ascvec.cast %420, %mask : <f32> -> <bf16>
                      ascvec.storealign_pack %arg33[%417], %421, %mask {mode = "b32_to_b16"} : memref<4096xbf16, 1>, index, <bf16>
                    } {cannir.loop_unroll = 4 : i64}
                  } {cannir.vf_mode = "simd"}
                  %406 = arith.cmpi eq, %399, %c0_i64 : i64
                  %407 = arith.select %406, %393, %394 : i64
                  %408 = ascscalar.bind_local_ptr %407 : memref<4160xbf16, 1>
                  %409 = arith.select %406, %c10_i64, %c11_i64 : i64
                  %410 = arith.addi %399, %c1_i64 overflow<nsw> : i64
                  %411 = arith.remui %410, %c2_i64 : i64
                  ascvec.get_buf v, %409, 0 {cannir.buffer.generated_sync}
                  ascvec.vf(%alloc_37, %408 : memref<4096xbf16, 1>, memref<4160xbf16, 1>) {
                  ^bb0(%arg31: memref<4096xbf16, 1>, %arg32: memref<4160xbf16, 1>):
                    %c1_i32_68 = arith.constant 1 : i32
                    %c0_i32_69 = arith.constant 0 : i32
                    %c65 = arith.constant 65 : index
                    %c16_i32_70 = arith.constant 16 : i32
                    %c0_71 = arith.constant 0 : index
                    %c64_i32_72 = arith.constant 64 : i32
                    scf.for %arg33 = %c0_i32_69 to %c64_i32_72 step %c1_i32_68  : i32 {
                      %mask, %remaining_out = ascvec.update_mask %c64_i32_72 elem_bits 16 : !ascvec.mask
                      %414 = arith.muli %arg33, %c64_i32_72 overflow<nsw> : i32
                      %415 = arith.index_cast %414 : i32 to index
                      %416 = ascvec.loadalign %arg31[%415] : memref<4096xbf16, 1>, index -> <bf16>
                      %417 = arith.muli %arg33, %c16_i32_70 overflow<nsw> : i32
                      %418 = arith.index_cast %417 : i32 to index
                      ascvec.storealign_strided %arg32[%418], %416, %mask, %c65, %c0_71 : memref<4160xbf16, 1>, <bf16>
                    }
                  } {cannir.vf_mode = "simd"}
                  ascvec.release_buf v, %409, 0 {cannir.buffer.generated_sync}
                  %412 = ascscalar.bind_local_ptr %407 : memref<1xbf16, 1>
                  %413 = ascscalar.bitcast_memref %alloc_62 byte_offset = %c0_i64 : memref<8192xbf16, 2> -> memref<?xbf16, 2>
                  ascvec.sync_intra_wait mte3, %c8_i64 {cannir.sync.banks = 3 : i32}
                  ascvec.get_buf mte3, %409, 0 {cannir.buffer.generated_sync}
                  ascvec.copy_ub2l1_strided %413[%405], %412[%c0], %c4_i32, %c64_i32, %c1_i32, %c0_i32 : memref<?xbf16, 2>, memref<1xbf16, 1>
                  ascvec.release_buf mte3, %409, 0 {cannir.buffer.generated_sync}
                  ascvec.sync_intra_arrive mte3, %c8_i64 {cannir.sync.banks = 3 : i32}
                  scf.yield %411 : i64
                }
                %341 = arith.addi %arg28, %c1_i64 overflow<nsw> : i64
                %342 = arith.remui %341, %c2_i64 : i64
                %343 = arith.addi %342, %c1_i64 overflow<nsw> : i64
                %344 = arith.remui %343, %c2_i64 : i64
                %345 = arith.addi %344, %c1_i64 overflow<nsw> : i64
                %346 = arith.remui %345, %c2_i64 : i64
                %347 = arith.addi %346, %c1_i64 overflow<nsw> : i64
                %348 = arith.remui %347, %c2_i64 : i64
                %349 = arith.addi %348, %c1_i64 overflow<nsw> : i64
                %350 = arith.remui %349, %c2_i64 : i64
                %351 = arith.addi %350, %c1_i64 overflow<nsw> : i64
                %352 = arith.remui %351, %c2_i64 : i64
                %353 = arith.addi %352, %c1_i64 overflow<nsw> : i64
                %354 = arith.remui %353, %c2_i64 : i64
                %355 = arith.addi %354, %c1_i64 overflow<nsw> : i64
                %356 = arith.remui %355, %c2_i64 : i64
                %357 = arith.addi %356, %c1_i64 overflow<nsw> : i64
                %358 = arith.remui %357, %c2_i64 : i64
                %359 = arith.addi %358, %c1_i64 overflow<nsw> : i64
                %360 = arith.remui %359, %c2_i64 : i64
                %361 = arith.addi %360, %c1_i64 overflow<nsw> : i64
                %362 = arith.remui %361, %c2_i64 : i64
                %363 = arith.addi %362, %c1_i64 overflow<nsw> : i64
                %364 = arith.remui %363, %c2_i64 : i64
                %365 = arith.addi %364, %c1_i64 overflow<nsw> : i64
                %366 = arith.remui %365, %c2_i64 : i64
                scf.yield %366, %340 : i64, i64
              } else {
                scf.yield %arg28, %arg30 : i64, i64
              }
              %123 = arith.remsi %arg25, %c2_i64 : i64
              %124 = arith.cmpi ne, %123, %c0_i64 : i64
              %125 = arith.cmpi slt, %123, %c0_i64 : i64
              %126 = arith.andi %124, %125 : i1
              %127 = arith.select %126, %c2_i64, %c0_i64 : i64
              %128 = arith.addi %123, %127 overflow<nsw> : i64
              %129 = arith.extsi %84 : i32 to i64
              %130 = arith.remsi %arg24, %129 : i64
              %131 = arith.cmpi ne, %130, %c0_i64 : i64
              %132 = arith.cmpi slt, %130, %c0_i64 : i64
              %133 = arith.cmpi slt, %129, %c0_i64 : i64
              %134 = arith.xori %132, %133 : i1
              %135 = arith.andi %131, %134 : i1
              %136 = arith.divsi %arg24, %129 : i64
              %137 = arith.extui %135 : i1 to i64
              %138 = arith.subi %136, %137 : i64
              %139 = arith.muli %138, %129 : i64
              %140 = arith.subi %arg24, %139 : i64
              %141 = arith.extsi %81 : i32 to i64
              %142 = arith.addi %141, %140 : i64
              %143 = cannir.extract_layout %arg1 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
              %144 = cannir.extract_shape %143 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 128)>
              %145 = cannir.extract_dim %144[1] : !cannir.shape<(-1, -1, -1, 128)>
              %146 = arith.maxsi %145, %c1_i64 : i64
              %147 = arith.remsi %138, %146 : i64
              %148 = arith.cmpi slt, %147, %c0_i64 : i64
              %149 = arith.select %148, %146, %c0_i64 : i64
              %150 = arith.addi %147, %149 : i64
              %151 = arith.remsi %arg24, %c2_i64 : i64
              %152 = arith.cmpi ne, %151, %c0_i64 : i64
              %153 = arith.cmpi slt, %151, %c0_i64 : i64
              %154 = arith.andi %152, %153 : i1
              %155 = arith.select %154, %c2_i64, %c0_i64 : i64
              %156 = arith.addi %151, %155 overflow<nsw> : i64
              %157 = cannir.extract_layout %arg0 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
              %158 = cannir.extract_shape %157 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 128)>
              %159 = cannir.extract_dim %158[1] : !cannir.shape<(-1, -1, -1, 128)>
              %160 = arith.remsi %145, %159 : i64
              %161 = arith.cmpi ne, %160, %c0_i64 : i64
              %162 = arith.cmpi slt, %160, %c0_i64 : i64
              %163 = arith.cmpi slt, %159, %c0_i64 : i64
              %164 = arith.xori %162, %163 : i1
              %165 = arith.andi %161, %164 : i1
              %166 = arith.divsi %145, %159 : i64
              %167 = arith.extui %165 : i1 to i64
              %168 = arith.subi %166, %167 : i64
              %169 = arith.remsi %150, %168 : i64
              %170 = arith.cmpi ne, %169, %c0_i64 : i64
              %171 = arith.cmpi slt, %169, %c0_i64 : i64
              %172 = arith.cmpi slt, %168, %c0_i64 : i64
              %173 = arith.xori %171, %172 : i1
              %174 = arith.andi %170, %173 : i1
              %175 = arith.divsi %150, %168 : i64
              %176 = arith.extui %174 : i1 to i64
              %177 = arith.subi %175, %176 : i64
              %178 = arith.addi %140, %c1_i64 : i64
              %179 = arith.cmpi eq, %178, %129 : i64
              %180 = arith.select %101, %179, %false : i1
              %181 = arith.select %180, %95, %c64_i64 : i64
              %182 = arith.index_cast %128 : i64 to index
              cannir.array_store %arg24, %111[%182] : <2xi64>
              cannir.array_store %181, %112[%182] : <2xi64>
              %183 = arith.cmpi eq, %102, %c0_i64 : i64
              scf.if %183 {
                %297 = arith.cmpi slt, %181, %c64_i64 : i64
                scf.if %297 {
                  %298 = arith.divsi %78, %c64_i64 : i64
                  %299 = arith.minsi %142, %298 : i64
                  %300 = arith.muli %299, %c64_i64 : i64
                  %301 = arith.subi %78, %300 : i64
                  %302 = arith.minsi %301, %c64_i64 : i64
                  %303 = arith.cmpi sle, %142, %298 : i64
                  %304 = arith.select %303, %302, %c0_i64 : i64
                  %305 = arith.trunci %304 : i64 to i32
                  ascvec.vf(%alloc_16, %305 : memref<8192xbf16, 1>, i32) {
                  ^bb0(%arg31: memref<8192xbf16, 1>, %arg32: i32):
                    %c1_i32_68 = arith.constant 1 : i32
                    %c64_i32_69 = arith.constant 64 : i32
                    %cst = arith.constant 0.000000e+00 : bf16
                    %c128_i32_70 = arith.constant 128 : i32
                    %mask, %remaining_out = ascvec.update_mask %c128_i32_70 elem_bits 16 : !ascvec.mask
                    %328 = ascvec.duplicate_scalar %cst, %mask : bf16 -> <bf16>
                    scf.for %arg33 = %arg32 to %c64_i32_69 step %c1_i32_68  : i32 {
                      %329 = arith.muli %arg33, %c128_i32_70 : i32
                      %330 = arith.index_cast %329 : i32 to index
                      ascvec.storealign %arg31[%330], %328, %mask : memref<8192xbf16, 1>, index, <bf16>
                    } {cannir.loop_unroll = 4 : i64}
                    ascvec.mem_bar vst_vld
                  } {cannir.vf_mode = "simd"}
                  ascvec.sync_notify {dst_pipe = 5 : i32, event_id = 0 : i32, src_pipe = 2 : i32}
                  ascvec.sync_wait {dst_pipe = 5 : i32, event_id = 0 : i32, src_pipe = 2 : i32}
                  %306 = cannir.extract_physical_layout %arg2 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                  %307 = cannir.extract_stride %306 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                  %308 = cannir.extract_dim %307[2] : !cannir.stride<(-1, -1, -1, 1)>
                  %309 = arith.muli %308, %c2_i64 : i64
                  %310 = cannir.extract_dim %307[0] : !cannir.stride<(-1, -1, -1, 1)>
                  %311 = arith.muli %97, %310 : i64
                  %312 = arith.muli %311, %c2_i64 : i64
                  %313 = cannir.extract_dim %307[1] : !cannir.stride<(-1, -1, -1, 1)>
                  %314 = arith.muli %177, %313 : i64
                  %315 = arith.muli %314, %c2_i64 : i64
                  %316 = arith.addi %312, %315 : i64
                  %317 = arith.muli %142, %c64_i64 : i64
                  %318 = arith.addi %317, %96 : i64
                  %319 = arith.muli %318, %308 : i64
                  %320 = arith.muli %319, %c2_i64 : i64
                  %321 = arith.addi %316, %320 : i64
                  "ascvec.copy_gm2ub_plan"(%104, %alloc_1, %c0_i32, %c0_i32, %c0_i64, %321, %304, %c256_i64, %309, %c256_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = 0 : i64, ub_capacity = 16384 : i64}> : (memref<?xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                  ascvec.sync_notify {dst_pipe = 2 : i32, event_id = 0 : i32, src_pipe = 5 : i32}
                  ascvec.sync_wait {dst_pipe = 2 : i32, event_id = 0 : i32, src_pipe = 5 : i32}
                  ascvec.vf(%alloc_16 : memref<8192xbf16, 1>) {
                  ^bb0(%arg31: memref<8192xbf16, 1>):
                    %c1_i32_68 = arith.constant 1 : i32
                    %c0_i32_69 = arith.constant 0 : i32
                    %cst = arith.constant 9.99999997E-7 : f32
                    %c128_i32_70 = arith.constant 128 : i32
                    %cst_71 = arith.constant 1.000000e+00 : f32
                    %c64_i32_72 = arith.constant 64 : i32
                    %mask, %remaining_out = ascvec.update_mask %c64_i32_72 : !ascvec.mask
                    %328 = ascvec.duplicate_scalar %cst_71, %mask : f32 -> <f32>
                    scf.for %arg32 = %c0_i32_69 to %c64_i32_72 step %c1_i32_68  : i32 {
                      %329 = arith.muli %arg32, %c128_i32_70 overflow<nsw> : i32
                      %330 = arith.index_cast %329 : i32 to index
                      %331 = ascvec.loadalign_unpack %arg31[%330] {mode = "b16_to_b32"} : memref<8192xbf16, 1>, index -> <bf16>
                      %332 = arith.addi %329, %c64_i32_72 overflow<nsw> : i32
                      %333 = arith.index_cast %332 : i32 to index
                      %334 = ascvec.loadalign_unpack %arg31[%333] {mode = "b16_to_b32"} : memref<8192xbf16, 1>, index -> <bf16>
                      %335 = ascvec.cast %331, %mask {rounding = 6 : i32} : <bf16> -> <f32>
                      %336 = ascvec.cast %334, %mask {rounding = 6 : i32} : <bf16> -> <f32>
                      %337 = ascvec.mul %335, %335, %mask : <f32>
                      %338 = ascvec.mul %336, %336, %mask : <f32>
                      %339 = ascvec.add %337, %338, %mask : <f32>
                      %340 = ascvec.reduce_sum %339, %mask : <f32>
                      %341 = ascvec.duplicate %340, %mask : <f32> -> <f32>
                      %342 = ascvec.add_scalar %341, %cst, %mask : <f32>, f32
                      %343 = ascvec.sqrt %342, %mask : <f32>
                      %344 = ascvec.div %328, %343, %mask : <f32>
                      %345 = ascvec.mul %335, %344, %mask : <f32>
                      %346 = ascvec.mul %336, %344, %mask : <f32>
                      %347 = ascvec.cast %345, %mask : <f32> -> <bf16>
                      %348 = ascvec.cast %346, %mask : <f32> -> <bf16>
                      ascvec.storealign_pack %arg31[%330], %347, %mask {mode = "b32_to_b16"} : memref<8192xbf16, 1>, index, <bf16>
                      ascvec.storealign_pack %arg31[%333], %348, %mask {mode = "b32_to_b16"} : memref<8192xbf16, 1>, index, <bf16>
                    } {cannir.loop_unroll = 4 : i64}
                    ascvec.mem_bar vst_vld
                  } {cannir.vf_mode = "simd"}
                  ascvec.vf(%alloc_25, %alloc_16 : memref<4096xbf16, 1>, memref<8192xbf16, 1>) {
                  ^bb0(%arg31: memref<4096xbf16, 1>, %arg32: memref<8192xbf16, 1>):
                    %c0_68 = arith.constant 0 : index
                    %c128 = arith.constant 128 : index
                    %c1_69 = arith.constant 1 : index
                    %c64 = arith.constant 64 : index
                    %c64_i32_70 = arith.constant 64 : i32
                    scf.for %arg33 = %c0_68 to %c64 step %c1_69 {
                      %mask, %remaining_out = ascvec.update_mask %c64_i32_70 elem_bits 16 : !ascvec.mask
                      %328 = arith.remui %arg33, %c64 : index
                      %329 = arith.muli %328, %c64 : index
                      %330 = arith.muli %328, %c128 : index
                      %331 = ascvec.loadalign %arg32[%330] : memref<8192xbf16, 1>, index -> <bf16>
                      ascvec.storealign %arg31[%329], %331, %mask : memref<4096xbf16, 1>, index, <bf16>
                    }
                    ascvec.mem_bar vst_vld
                  } {cannir.vf_mode = "simd"}
                  ascvec.vf(%alloc_16, %alloc_18 : memref<8192xbf16, 1>, memref<4096xf32, 1>) {
                  ^bb0(%arg31: memref<8192xbf16, 1>, %arg32: memref<4096xf32, 1>):
                    %c1_i32_68 = arith.constant 1 : i32
                    %c0_i32_69 = arith.constant 0 : i32
                    %c64 = arith.constant 64 : index
                    %c128_i32_70 = arith.constant 128 : i32
                    %c64_i32_71 = arith.constant 64 : i32
                    scf.for %arg33 = %c0_i32_69 to %c64_i32_71 step %c1_i32_68  : i32 {
                      %mask, %remaining_out = ascvec.update_mask %c64_i32_71 : !ascvec.mask
                      %328 = arith.muli %arg33, %c128_i32_70 overflow<nsw> : i32
                      %329 = arith.index_cast %328 : i32 to index
                      %330 = arith.addi %329, %c64 : index
                      %331 = ascvec.loadalign_unpack %arg31[%330] {mode = "b16_to_b32"} : memref<8192xbf16, 1>, index -> <bf16>
                      %332 = ascvec.cast %331, %mask {rounding = 6 : i32} : <bf16> -> <f32>
                      %333 = arith.muli %arg33, %c64_i32_71 overflow<nsw> : i32
                      %334 = arith.index_cast %333 : i32 to index
                      ascvec.storealign %arg32[%334], %332, %mask : memref<4096xf32, 1>, index, <f32>
                    }
                  } {cannir.vf_mode = "simd"}
                  ascvec.sync_notify {dst_pipe = 6 : i32, event_id = 2 : i32, src_pipe = 2 : i32}
                  ascvec.sync_wait {dst_pipe = 6 : i32, event_id = 2 : i32, src_pipe = 2 : i32}
                  %322 = arith.muli %0, %c262144_i64 : i64
                  %323 = ascscalar.memref_address %alloc_4 : memref<1xf32>
                  %324 = arith.divui %322, %c8_i64 : i64
                  %325 = arith.addi %323, %324 : i64
                  %326 = ascscalar.bind_gm_ptr %325 : memref<1xf32>
                  %327 = ascscalar.bitcast_memref %alloc_18 byte_offset = %c0_i64 : memref<4096xf32, 1> -> memref<?xf32, 1>
                  ascvec.copy_ub2gm %326[%c0], %327[%c0], %c1_i32, %c16384_i32, %c0_i64, %c16384_i32 l2_cache_ctl %c0_i32 : memref<1xf32>, memref<?xf32, 1>
                  ascvec.sync_all
                  ascvec.sync_block_arrive mte3, %c10_i64 {mode = 1 : i32}
                } else {
                  %298 = arith.divsi %78, %c64_i64 : i64
                  %299 = arith.minsi %142, %298 : i64
                  %300 = arith.muli %299, %c64_i64 : i64
                  %301 = arith.subi %78, %300 : i64
                  %302 = arith.minsi %301, %c64_i64 : i64
                  %303 = arith.cmpi sle, %142, %298 : i64
                  %304 = arith.select %303, %302, %c0_i64 : i64
                  %305 = cannir.extract_physical_layout %arg2 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                  %306 = cannir.extract_stride %305 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                  %307 = cannir.extract_dim %306[2] : !cannir.stride<(-1, -1, -1, 1)>
                  %308 = arith.muli %307, %c2_i64 : i64
                  %309 = cannir.extract_dim %306[0] : !cannir.stride<(-1, -1, -1, 1)>
                  %310 = arith.muli %97, %309 : i64
                  %311 = arith.muli %310, %c2_i64 : i64
                  %312 = cannir.extract_dim %306[1] : !cannir.stride<(-1, -1, -1, 1)>
                  %313 = arith.muli %177, %312 : i64
                  %314 = arith.muli %313, %c2_i64 : i64
                  %315 = arith.addi %311, %314 : i64
                  %316 = arith.muli %142, %c64_i64 : i64
                  %317 = arith.addi %316, %96 : i64
                  %318 = arith.muli %317, %307 : i64
                  %319 = arith.muli %318, %c2_i64 : i64
                  %320 = arith.addi %315, %319 : i64
                  "ascvec.copy_gm2ub_plan"(%104, %alloc_1, %c0_i32, %c0_i32, %c0_i64, %320, %304, %c256_i64, %308, %c256_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = 0 : i64, ub_capacity = 16384 : i64}> : (memref<?xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                  ascvec.sync_notify {dst_pipe = 2 : i32, event_id = 0 : i32, src_pipe = 5 : i32}
                  ascvec.sync_wait {dst_pipe = 2 : i32, event_id = 0 : i32, src_pipe = 5 : i32}
                  ascvec.vf(%alloc_16 : memref<8192xbf16, 1>) {
                  ^bb0(%arg31: memref<8192xbf16, 1>):
                    %c1_i32_68 = arith.constant 1 : i32
                    %c0_i32_69 = arith.constant 0 : i32
                    %cst = arith.constant 9.99999997E-7 : f32
                    %c128_i32_70 = arith.constant 128 : i32
                    %cst_71 = arith.constant 1.000000e+00 : f32
                    %c64_i32_72 = arith.constant 64 : i32
                    %mask, %remaining_out = ascvec.update_mask %c64_i32_72 : !ascvec.mask
                    %327 = ascvec.duplicate_scalar %cst_71, %mask : f32 -> <f32>
                    scf.for %arg32 = %c0_i32_69 to %c64_i32_72 step %c1_i32_68  : i32 {
                      %328 = arith.muli %arg32, %c128_i32_70 overflow<nsw> : i32
                      %329 = arith.index_cast %328 : i32 to index
                      %330 = ascvec.loadalign_unpack %arg31[%329] {mode = "b16_to_b32"} : memref<8192xbf16, 1>, index -> <bf16>
                      %331 = arith.addi %328, %c64_i32_72 overflow<nsw> : i32
                      %332 = arith.index_cast %331 : i32 to index
                      %333 = ascvec.loadalign_unpack %arg31[%332] {mode = "b16_to_b32"} : memref<8192xbf16, 1>, index -> <bf16>
                      %334 = ascvec.cast %330, %mask {rounding = 6 : i32} : <bf16> -> <f32>
                      %335 = ascvec.cast %333, %mask {rounding = 6 : i32} : <bf16> -> <f32>
                      %336 = ascvec.mul %334, %334, %mask : <f32>
                      %337 = ascvec.mul %335, %335, %mask : <f32>
                      %338 = ascvec.add %336, %337, %mask : <f32>
                      %339 = ascvec.reduce_sum %338, %mask : <f32>
                      %340 = ascvec.duplicate %339, %mask : <f32> -> <f32>
                      %341 = ascvec.add_scalar %340, %cst, %mask : <f32>, f32
                      %342 = ascvec.sqrt %341, %mask : <f32>
                      %343 = ascvec.div %327, %342, %mask : <f32>
                      %344 = ascvec.mul %334, %343, %mask : <f32>
                      %345 = ascvec.mul %335, %343, %mask : <f32>
                      %346 = ascvec.cast %344, %mask : <f32> -> <bf16>
                      %347 = ascvec.cast %345, %mask : <f32> -> <bf16>
                      ascvec.storealign_pack %arg31[%329], %346, %mask {mode = "b32_to_b16"} : memref<8192xbf16, 1>, index, <bf16>
                      ascvec.storealign_pack %arg31[%332], %347, %mask {mode = "b32_to_b16"} : memref<8192xbf16, 1>, index, <bf16>
                    } {cannir.loop_unroll = 4 : i64}
                    ascvec.mem_bar vst_vld
                  } {cannir.vf_mode = "simd"}
                  ascvec.vf(%alloc_25, %alloc_16 : memref<4096xbf16, 1>, memref<8192xbf16, 1>) {
                  ^bb0(%arg31: memref<4096xbf16, 1>, %arg32: memref<8192xbf16, 1>):
                    %c0_68 = arith.constant 0 : index
                    %c128 = arith.constant 128 : index
                    %c1_69 = arith.constant 1 : index
                    %c64 = arith.constant 64 : index
                    %c64_i32_70 = arith.constant 64 : i32
                    scf.for %arg33 = %c0_68 to %c64 step %c1_69 {
                      %mask, %remaining_out = ascvec.update_mask %c64_i32_70 elem_bits 16 : !ascvec.mask
                      %327 = arith.remui %arg33, %c64 : index
                      %328 = arith.muli %327, %c64 : index
                      %329 = arith.muli %327, %c128 : index
                      %330 = ascvec.loadalign %arg32[%329] : memref<8192xbf16, 1>, index -> <bf16>
                      ascvec.storealign %arg31[%328], %330, %mask : memref<4096xbf16, 1>, index, <bf16>
                    }
                    ascvec.mem_bar vst_vld
                  } {cannir.vf_mode = "simd"}
                  ascvec.vf(%alloc_16, %alloc_18 : memref<8192xbf16, 1>, memref<4096xf32, 1>) {
                  ^bb0(%arg31: memref<8192xbf16, 1>, %arg32: memref<4096xf32, 1>):
                    %c1_i32_68 = arith.constant 1 : i32
                    %c0_i32_69 = arith.constant 0 : i32
                    %c64 = arith.constant 64 : index
                    %c128_i32_70 = arith.constant 128 : i32
                    %c64_i32_71 = arith.constant 64 : i32
                    scf.for %arg33 = %c0_i32_69 to %c64_i32_71 step %c1_i32_68  : i32 {
                      %mask, %remaining_out = ascvec.update_mask %c64_i32_71 : !ascvec.mask
                      %327 = arith.muli %arg33, %c128_i32_70 overflow<nsw> : i32
                      %328 = arith.index_cast %327 : i32 to index
                      %329 = arith.addi %328, %c64 : index
                      %330 = ascvec.loadalign_unpack %arg31[%329] {mode = "b16_to_b32"} : memref<8192xbf16, 1>, index -> <bf16>
                      %331 = ascvec.cast %330, %mask {rounding = 6 : i32} : <bf16> -> <f32>
                      %332 = arith.muli %arg33, %c64_i32_71 overflow<nsw> : i32
                      %333 = arith.index_cast %332 : i32 to index
                      ascvec.storealign %arg32[%333], %331, %mask : memref<4096xf32, 1>, index, <f32>
                    }
                  } {cannir.vf_mode = "simd"}
                  ascvec.sync_notify {dst_pipe = 6 : i32, event_id = 2 : i32, src_pipe = 2 : i32}
                  ascvec.sync_wait {dst_pipe = 6 : i32, event_id = 2 : i32, src_pipe = 2 : i32}
                  %321 = arith.muli %0, %c262144_i64 : i64
                  %322 = ascscalar.memref_address %alloc_4 : memref<1xf32>
                  %323 = arith.divui %321, %c8_i64 : i64
                  %324 = arith.addi %322, %323 : i64
                  %325 = ascscalar.bind_gm_ptr %324 : memref<1xf32>
                  %326 = ascscalar.bitcast_memref %alloc_18 byte_offset = %c0_i64 : memref<4096xf32, 1> -> memref<?xf32, 1>
                  ascvec.copy_ub2gm %325[%c0], %326[%c0], %c1_i32, %c16384_i32, %c0_i64, %c16384_i32 l2_cache_ctl %c0_i32 : memref<1xf32>, memref<?xf32, 1>
                  ascvec.sync_all
                  ascvec.sync_block_arrive mte3, %c10_i64 {mode = 1 : i32}
                }
              } else {
                %297 = arith.cmpi slt, %181, %c64_i64 : i64
                scf.if %297 {
                  %298 = arith.divsi %78, %c64_i64 : i64
                  %299 = arith.minsi %142, %298 : i64
                  %300 = arith.muli %299, %c64_i64 : i64
                  %301 = arith.subi %78, %300 : i64
                  %302 = arith.minsi %301, %c64_i64 : i64
                  %303 = arith.cmpi sle, %142, %298 : i64
                  %304 = arith.select %303, %302, %c0_i64 : i64
                  %305 = arith.trunci %304 : i64 to i32
                  ascvec.vf(%alloc_17, %305 : memref<8192xbf16, 1>, i32) {
                  ^bb0(%arg31: memref<8192xbf16, 1>, %arg32: i32):
                    %c1_i32_68 = arith.constant 1 : i32
                    %c64_i32_69 = arith.constant 64 : i32
                    %cst = arith.constant 0.000000e+00 : bf16
                    %c128_i32_70 = arith.constant 128 : i32
                    %mask, %remaining_out = ascvec.update_mask %c128_i32_70 elem_bits 16 : !ascvec.mask
                    %330 = ascvec.duplicate_scalar %cst, %mask : bf16 -> <bf16>
                    scf.for %arg33 = %arg32 to %c64_i32_69 step %c1_i32_68  : i32 {
                      %331 = arith.muli %arg33, %c128_i32_70 : i32
                      %332 = arith.index_cast %331 : i32 to index
                      ascvec.storealign %arg31[%332], %330, %mask : memref<8192xbf16, 1>, index, <bf16>
                    } {cannir.loop_unroll = 4 : i64}
                    ascvec.mem_bar vst_vld
                  } {cannir.vf_mode = "simd"}
                  ascvec.sync_notify {dst_pipe = 5 : i32, event_id = 0 : i32, src_pipe = 2 : i32}
                  ascvec.sync_wait {dst_pipe = 5 : i32, event_id = 0 : i32, src_pipe = 2 : i32}
                  %306 = cannir.extract_physical_layout %arg0 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                  %307 = cannir.extract_stride %306 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                  %308 = cannir.extract_dim %307[2] : !cannir.stride<(-1, -1, -1, 1)>
                  %309 = arith.muli %308, %c2_i64 : i64
                  %310 = cannir.extract_dim %307[0] : !cannir.stride<(-1, -1, -1, 1)>
                  %311 = arith.muli %97, %310 : i64
                  %312 = arith.muli %311, %c2_i64 : i64
                  %313 = cannir.extract_dim %307[1] : !cannir.stride<(-1, -1, -1, 1)>
                  %314 = arith.muli %177, %313 : i64
                  %315 = arith.muli %314, %c2_i64 : i64
                  %316 = arith.addi %312, %315 : i64
                  %317 = arith.muli %142, %c64_i64 : i64
                  %318 = arith.addi %317, %96 : i64
                  %319 = arith.muli %318, %308 : i64
                  %320 = arith.muli %319, %c2_i64 : i64
                  %321 = arith.addi %316, %320 : i64
                  "ascvec.copy_gm2ub_plan"(%105, %alloc, %c0_i32, %c0_i32, %c0_i64, %321, %304, %c256_i64, %309, %c256_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = 16384 : i64, ub_capacity = 16384 : i64}> : (memref<?xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                  ascvec.sync_notify {dst_pipe = 2 : i32, event_id = 4 : i32, src_pipe = 5 : i32}
                  ascvec.sync_wait {dst_pipe = 2 : i32, event_id = 4 : i32, src_pipe = 5 : i32}
                  ascvec.vf(%alloc_17 : memref<8192xbf16, 1>) {
                  ^bb0(%arg31: memref<8192xbf16, 1>):
                    %c1_i32_68 = arith.constant 1 : i32
                    %c0_i32_69 = arith.constant 0 : i32
                    %cst = arith.constant 9.99999997E-7 : f32
                    %c128_i32_70 = arith.constant 128 : i32
                    %cst_71 = arith.constant 1.000000e+00 : f32
                    %c64_i32_72 = arith.constant 64 : i32
                    %mask, %remaining_out = ascvec.update_mask %c64_i32_72 : !ascvec.mask
                    %330 = ascvec.duplicate_scalar %cst_71, %mask : f32 -> <f32>
                    scf.for %arg32 = %c0_i32_69 to %c64_i32_72 step %c1_i32_68  : i32 {
                      %331 = arith.muli %arg32, %c128_i32_70 overflow<nsw> : i32
                      %332 = arith.index_cast %331 : i32 to index
                      %333 = ascvec.loadalign_unpack %arg31[%332] {mode = "b16_to_b32"} : memref<8192xbf16, 1>, index -> <bf16>
                      %334 = arith.addi %331, %c64_i32_72 overflow<nsw> : i32
                      %335 = arith.index_cast %334 : i32 to index
                      %336 = ascvec.loadalign_unpack %arg31[%335] {mode = "b16_to_b32"} : memref<8192xbf16, 1>, index -> <bf16>
                      %337 = ascvec.cast %333, %mask {rounding = 6 : i32} : <bf16> -> <f32>
                      %338 = ascvec.cast %336, %mask {rounding = 6 : i32} : <bf16> -> <f32>
                      %339 = ascvec.mul %337, %337, %mask : <f32>
                      %340 = ascvec.mul %338, %338, %mask : <f32>
                      %341 = ascvec.add %339, %340, %mask : <f32>
                      %342 = ascvec.reduce_sum %341, %mask : <f32>
                      %343 = ascvec.duplicate %342, %mask : <f32> -> <f32>
                      %344 = ascvec.add_scalar %343, %cst, %mask : <f32>, f32
                      %345 = ascvec.sqrt %344, %mask : <f32>
                      %346 = ascvec.div %330, %345, %mask : <f32>
                      %347 = ascvec.mul %337, %346, %mask : <f32>
                      %348 = ascvec.mul %338, %346, %mask : <f32>
                      %349 = ascvec.cast %347, %mask : <f32> -> <bf16>
                      %350 = ascvec.cast %348, %mask : <f32> -> <bf16>
                      ascvec.storealign_pack %arg31[%332], %349, %mask {mode = "b32_to_b16"} : memref<8192xbf16, 1>, index, <bf16>
                      ascvec.storealign_pack %arg31[%335], %350, %mask {mode = "b32_to_b16"} : memref<8192xbf16, 1>, index, <bf16>
                    } {cannir.loop_unroll = 4 : i64}
                    ascvec.mem_bar vst_vld
                  } {cannir.vf_mode = "simd"}
                  ascvec.vf(%alloc_26, %alloc_17 : memref<4096xbf16, 1>, memref<8192xbf16, 1>) {
                  ^bb0(%arg31: memref<4096xbf16, 1>, %arg32: memref<8192xbf16, 1>):
                    %c64 = arith.constant 64 : index
                    %c0_68 = arith.constant 0 : index
                    %c128 = arith.constant 128 : index
                    %c1_69 = arith.constant 1 : index
                    %c64_i32_70 = arith.constant 64 : i32
                    scf.for %arg33 = %c0_68 to %c64 step %c1_69 {
                      %mask, %remaining_out = ascvec.update_mask %c64_i32_70 elem_bits 16 : !ascvec.mask
                      %330 = arith.remui %arg33, %c64 : index
                      %331 = arith.muli %330, %c64 : index
                      %332 = arith.muli %330, %c128 : index
                      %333 = arith.addi %332, %c64 : index
                      %334 = ascvec.loadalign %arg32[%333] : memref<8192xbf16, 1>, index -> <bf16>
                      ascvec.storealign %arg31[%331], %334, %mask : memref<4096xbf16, 1>, index, <bf16>
                    }
                    ascvec.mem_bar vst_vld
                  } {cannir.vf_mode = "simd"}
                  ascvec.vf(%alloc_17, %alloc_18 : memref<8192xbf16, 1>, memref<4096xf32, 1>) {
                  ^bb0(%arg31: memref<8192xbf16, 1>, %arg32: memref<4096xf32, 1>):
                    %c1_i32_68 = arith.constant 1 : i32
                    %c0_i32_69 = arith.constant 0 : i32
                    %c128_i32_70 = arith.constant 128 : i32
                    %c64_i32_71 = arith.constant 64 : i32
                    scf.for %arg33 = %c0_i32_69 to %c64_i32_71 step %c1_i32_68  : i32 {
                      %mask, %remaining_out = ascvec.update_mask %c64_i32_71 : !ascvec.mask
                      %330 = arith.muli %arg33, %c128_i32_70 overflow<nsw> : i32
                      %331 = arith.index_cast %330 : i32 to index
                      %332 = ascvec.loadalign_unpack %arg31[%331] {mode = "b16_to_b32"} : memref<8192xbf16, 1>, index -> <bf16>
                      %333 = ascvec.cast %332, %mask {rounding = 6 : i32} : <bf16> -> <f32>
                      %334 = arith.muli %arg33, %c64_i32_71 overflow<nsw> : i32
                      %335 = arith.index_cast %334 : i32 to index
                      ascvec.storealign %arg32[%335], %333, %mask : memref<4096xf32, 1>, index, <f32>
                    }
                  } {cannir.vf_mode = "simd"}
                  ascvec.sync_notify {dst_pipe = 6 : i32, event_id = 2 : i32, src_pipe = 2 : i32}
                  ascvec.sync_wait {dst_pipe = 6 : i32, event_id = 2 : i32, src_pipe = 2 : i32}
                  %322 = arith.muli %0, %c8192_i64 : i64
                  %323 = arith.addi %322, %c4096_i64 : i64
                  %324 = arith.muli %323, %c32_i64 : i64
                  %325 = ascscalar.memref_address %alloc_4 : memref<1xf32>
                  %326 = arith.divui %324, %c8_i64 : i64
                  %327 = arith.addi %325, %326 : i64
                  %328 = ascscalar.bind_gm_ptr %327 : memref<1xf32>
                  %329 = ascscalar.bitcast_memref %alloc_18 byte_offset = %c0_i64 : memref<4096xf32, 1> -> memref<?xf32, 1>
                  ascvec.copy_ub2gm %328[%c0], %329[%c0], %c1_i32, %c16384_i32, %c0_i64, %c16384_i32 l2_cache_ctl %c0_i32 : memref<1xf32>, memref<?xf32, 1>
                  ascvec.sync_all
                  ascvec.sync_block_arrive mte3, %c10_i64 {mode = 1 : i32}
                } else {
                  %298 = arith.divsi %78, %c64_i64 : i64
                  %299 = arith.minsi %142, %298 : i64
                  %300 = arith.muli %299, %c64_i64 : i64
                  %301 = arith.subi %78, %300 : i64
                  %302 = arith.minsi %301, %c64_i64 : i64
                  %303 = arith.cmpi sle, %142, %298 : i64
                  %304 = arith.select %303, %302, %c0_i64 : i64
                  %305 = cannir.extract_physical_layout %arg0 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                  %306 = cannir.extract_stride %305 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                  %307 = cannir.extract_dim %306[2] : !cannir.stride<(-1, -1, -1, 1)>
                  %308 = arith.muli %307, %c2_i64 : i64
                  %309 = cannir.extract_dim %306[0] : !cannir.stride<(-1, -1, -1, 1)>
                  %310 = arith.muli %97, %309 : i64
                  %311 = arith.muli %310, %c2_i64 : i64
                  %312 = cannir.extract_dim %306[1] : !cannir.stride<(-1, -1, -1, 1)>
                  %313 = arith.muli %177, %312 : i64
                  %314 = arith.muli %313, %c2_i64 : i64
                  %315 = arith.addi %311, %314 : i64
                  %316 = arith.muli %142, %c64_i64 : i64
                  %317 = arith.addi %316, %96 : i64
                  %318 = arith.muli %317, %307 : i64
                  %319 = arith.muli %318, %c2_i64 : i64
                  %320 = arith.addi %315, %319 : i64
                  "ascvec.copy_gm2ub_plan"(%105, %alloc, %c0_i32, %c0_i32, %c0_i64, %320, %304, %c256_i64, %308, %c256_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = 16384 : i64, ub_capacity = 16384 : i64}> : (memref<?xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                  ascvec.sync_all
                  ascvec.sync_notify {dst_pipe = 2 : i32, event_id = 4 : i32, src_pipe = 5 : i32}
                  ascvec.sync_wait {dst_pipe = 2 : i32, event_id = 4 : i32, src_pipe = 5 : i32}
                  ascvec.vf(%alloc_17 : memref<8192xbf16, 1>) {
                  ^bb0(%arg31: memref<8192xbf16, 1>):
                    %c1_i32_68 = arith.constant 1 : i32
                    %c0_i32_69 = arith.constant 0 : i32
                    %cst = arith.constant 9.99999997E-7 : f32
                    %c128_i32_70 = arith.constant 128 : i32
                    %cst_71 = arith.constant 1.000000e+00 : f32
                    %c64_i32_72 = arith.constant 64 : i32
                    %mask, %remaining_out = ascvec.update_mask %c64_i32_72 : !ascvec.mask
                    %329 = ascvec.duplicate_scalar %cst_71, %mask : f32 -> <f32>
                    scf.for %arg32 = %c0_i32_69 to %c64_i32_72 step %c1_i32_68  : i32 {
                      %330 = arith.muli %arg32, %c128_i32_70 overflow<nsw> : i32
                      %331 = arith.index_cast %330 : i32 to index
                      %332 = ascvec.loadalign_unpack %arg31[%331] {mode = "b16_to_b32"} : memref<8192xbf16, 1>, index -> <bf16>
                      %333 = arith.addi %330, %c64_i32_72 overflow<nsw> : i32
                      %334 = arith.index_cast %333 : i32 to index
                      %335 = ascvec.loadalign_unpack %arg31[%334] {mode = "b16_to_b32"} : memref<8192xbf16, 1>, index -> <bf16>
                      %336 = ascvec.cast %332, %mask {rounding = 6 : i32} : <bf16> -> <f32>
                      %337 = ascvec.cast %335, %mask {rounding = 6 : i32} : <bf16> -> <f32>
                      %338 = ascvec.mul %336, %336, %mask : <f32>
                      %339 = ascvec.mul %337, %337, %mask : <f32>
                      %340 = ascvec.add %338, %339, %mask : <f32>
                      %341 = ascvec.reduce_sum %340, %mask : <f32>
                      %342 = ascvec.duplicate %341, %mask : <f32> -> <f32>
                      %343 = ascvec.add_scalar %342, %cst, %mask : <f32>, f32
                      %344 = ascvec.sqrt %343, %mask : <f32>
                      %345 = ascvec.div %329, %344, %mask : <f32>
                      %346 = ascvec.mul %336, %345, %mask : <f32>
                      %347 = ascvec.mul %337, %345, %mask : <f32>
                      %348 = ascvec.cast %346, %mask : <f32> -> <bf16>
                      %349 = ascvec.cast %347, %mask : <f32> -> <bf16>
                      ascvec.storealign_pack %arg31[%331], %348, %mask {mode = "b32_to_b16"} : memref<8192xbf16, 1>, index, <bf16>
                      ascvec.storealign_pack %arg31[%334], %349, %mask {mode = "b32_to_b16"} : memref<8192xbf16, 1>, index, <bf16>
                    } {cannir.loop_unroll = 4 : i64}
                    ascvec.mem_bar vst_vld
                  } {cannir.vf_mode = "simd"}
                  ascvec.vf(%alloc_26, %alloc_17 : memref<4096xbf16, 1>, memref<8192xbf16, 1>) {
                  ^bb0(%arg31: memref<4096xbf16, 1>, %arg32: memref<8192xbf16, 1>):
                    %c64 = arith.constant 64 : index
                    %c0_68 = arith.constant 0 : index
                    %c128 = arith.constant 128 : index
                    %c1_69 = arith.constant 1 : index
                    %c64_i32_70 = arith.constant 64 : i32
                    scf.for %arg33 = %c0_68 to %c64 step %c1_69 {
                      %mask, %remaining_out = ascvec.update_mask %c64_i32_70 elem_bits 16 : !ascvec.mask
                      %329 = arith.remui %arg33, %c64 : index
                      %330 = arith.muli %329, %c64 : index
                      %331 = arith.muli %329, %c128 : index
                      %332 = arith.addi %331, %c64 : index
                      %333 = ascvec.loadalign %arg32[%332] : memref<8192xbf16, 1>, index -> <bf16>
                      ascvec.storealign %arg31[%330], %333, %mask : memref<4096xbf16, 1>, index, <bf16>
                    }
                    ascvec.mem_bar vst_vld
                  } {cannir.vf_mode = "simd"}
                  ascvec.vf(%alloc_17, %alloc_18 : memref<8192xbf16, 1>, memref<4096xf32, 1>) {
                  ^bb0(%arg31: memref<8192xbf16, 1>, %arg32: memref<4096xf32, 1>):
                    %c1_i32_68 = arith.constant 1 : i32
                    %c0_i32_69 = arith.constant 0 : i32
                    %c128_i32_70 = arith.constant 128 : i32
                    %c64_i32_71 = arith.constant 64 : i32
                    scf.for %arg33 = %c0_i32_69 to %c64_i32_71 step %c1_i32_68  : i32 {
                      %mask, %remaining_out = ascvec.update_mask %c64_i32_71 : !ascvec.mask
                      %329 = arith.muli %arg33, %c128_i32_70 overflow<nsw> : i32
                      %330 = arith.index_cast %329 : i32 to index
                      %331 = ascvec.loadalign_unpack %arg31[%330] {mode = "b16_to_b32"} : memref<8192xbf16, 1>, index -> <bf16>
                      %332 = ascvec.cast %331, %mask {rounding = 6 : i32} : <bf16> -> <f32>
                      %333 = arith.muli %arg33, %c64_i32_71 overflow<nsw> : i32
                      %334 = arith.index_cast %333 : i32 to index
                      ascvec.storealign %arg32[%334], %332, %mask : memref<4096xf32, 1>, index, <f32>
                    }
                  } {cannir.vf_mode = "simd"}
                  ascvec.sync_notify {dst_pipe = 6 : i32, event_id = 2 : i32, src_pipe = 2 : i32}
                  ascvec.sync_wait {dst_pipe = 6 : i32, event_id = 2 : i32, src_pipe = 2 : i32}
                  %321 = arith.muli %0, %c8192_i64 : i64
                  %322 = arith.addi %321, %c4096_i64 : i64
                  %323 = arith.muli %322, %c32_i64 : i64
                  %324 = ascscalar.memref_address %alloc_4 : memref<1xf32>
                  %325 = arith.divui %323, %c8_i64 : i64
                  %326 = arith.addi %324, %325 : i64
                  %327 = ascscalar.bind_gm_ptr %326 : memref<1xf32>
                  %328 = ascscalar.bitcast_memref %alloc_18 byte_offset = %c0_i64 : memref<4096xf32, 1> -> memref<?xf32, 1>
                  ascvec.copy_ub2gm %327[%c0], %328[%c0], %c1_i32, %c16384_i32, %c0_i64, %c16384_i32 l2_cache_ctl %c0_i32 : memref<1xf32>, memref<?xf32, 1>
                  ascvec.sync_all
                  ascvec.sync_block_arrive mte3, %c10_i64 {mode = 1 : i32}
                }
              }
              %184 = arith.divsi %78, %c64_i64 : i64
              %185 = arith.minsi %142, %184 : i64
              %186 = arith.muli %185, %c64_i64 : i64
              %187 = arith.subi %78, %186 : i64
              %188 = arith.minsi %187, %c64_i64 : i64
              %189 = arith.cmpi sle, %142, %184 : i64
              %190 = arith.select %189, %188, %c0_i64 : i64
              %191 = cannir.extract_physical_layout %arg4 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
              %192 = cannir.extract_stride %191 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
              %193 = cannir.extract_dim %192[2] : !cannir.stride<(-1, -1, -1, 1)>
              %194 = arith.muli %193, %c2_i64 : i64
              %195 = cannir.extract_dim %192[0] : !cannir.stride<(-1, -1, -1, 1)>
              %196 = arith.muli %97, %195 : i64
              %197 = arith.muli %196, %c2_i64 : i64
              %198 = cannir.extract_dim %192[1] : !cannir.stride<(-1, -1, -1, 1)>
              %199 = arith.muli %150, %198 : i64
              %200 = arith.muli %199, %c2_i64 : i64
              %201 = arith.addi %197, %200 : i64
              %202 = arith.muli %102, %c128_i64 : i64
              %203 = arith.addi %201, %202 : i64
              %204 = arith.muli %142, %c64_i64 : i64
              %205 = arith.addi %204, %96 : i64
              %206 = arith.muli %205, %193 : i64
              %207 = arith.muli %206, %c2_i64 : i64
              %208 = arith.addi %203, %207 : i64
              "ascvec.copy_gm2ub_plan"(%106, %alloc_3, %c0_i32, %c0_i32, %c0_i64, %208, %190, %c128_i64, %194, %c128_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = 49152 : i64, ub_capacity = 8192 : i64}> : (memref<?xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
              %209 = arith.muli %150, %c512_i64 : i64
              %210 = arith.muli %102, %c256_i64 : i64
              %211 = arith.addi %209, %210 : i64
              "ascvec.copy_gm2ub_plan"(%107, %alloc_6, %c0_i32, %c0_i32, %c0_i64, %211, %c1_i64, %c256_i64, %c0_i64, %c256_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 73728 : i64, ub_capacity = 256 : i64}> : (memref<?xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
              %212 = cannir.extract_layout %arg6 : !cannir.tensor<<gm, f32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>
              %213 = cannir.extract_shape %212 : <<(-1)>, <(1)>> -> <(-1)>
              %214 = cannir.extract_dim %213[0] : !cannir.shape<(-1)>
              %215 = arith.minsi %150, %214 : i64
              %216 = arith.subi %214, %215 : i64
              %217 = arith.minsi %216, %c1_i64 : i64
              %218 = arith.cmpi sle, %150, %214 : i64
              %219 = arith.select %218, %217, %c0_i64 : i64
              %220 = arith.muli %150, %c4_i64 : i64
              %221 = arith.muli %219, %c4_i64 : i64
              %222 = arith.cmpi eq, %221, %c0_i64 : i64
              %223 = arith.cmpi ne, %221, %c0_i64 : i64
              %224 = arith.extui %223 : i1 to i64
              %225 = arith.select %222, %c4_i64, %221 : i64
              "ascvec.copy_gm2ub_plan"(%108, %alloc_5, %c0_i32, %c0_i32, %c0_i64, %220, %224, %225, %c0_i64, %225, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 73984 : i64, ub_capacity = 4 : i64}> : (memref<?xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
              ascvec.sync_notify {dst_pipe = 2 : i32, event_id = 0 : i32, src_pipe = 5 : i32}
              ascvec.sync_wait {dst_pipe = 2 : i32, event_id = 0 : i32, src_pipe = 5 : i32}
              %226 = arith.cmpi slt, %181, %c64_i64 : i64
              %227 = arith.select %226, %181, %c64_i64 : i64
              scf.if %226 {
                ascvec.vf(%alloc_20, %181 : memref<4096xf32, 1>, i64) {
                ^bb0(%arg31: memref<4096xf32, 1>, %arg32: i64):
                  %c1_i64_68 = arith.constant 1 : i64
                  %c64_i64_69 = arith.constant 64 : i64
                  %cst = arith.constant 0.000000e+00 : f32
                  %c64_i32_70 = arith.constant 64 : i32
                  %mask, %remaining_out = ascvec.update_mask %c64_i32_70 : !ascvec.mask
                  %297 = ascvec.duplicate_scalar %cst, %mask : f32 -> <f32>
                  scf.for %arg33 = %arg32 to %c64_i64_69 step %c1_i64_68  : i64 {
                    %298 = arith.muli %arg33, %c64_i64_69 : i64
                    %299 = arith.index_cast %298 : i64 to index
                    ascvec.storealign %arg31[%299], %297, %mask : memref<4096xf32, 1>, index, <f32>
                  } {cannir.loop_unroll = 4 : i64}
                  ascvec.mem_bar vst_vld
                } {cannir.vf_mode = "simd"}
              }
              ascvec.vf(%arg21, %alloc_22, %alloc_19, %alloc_21, %alloc_20, %227 : f32, memref<1xf32, 1>, memref<4096xbf16, 1>, memref<64xf32, 1>, memref<4096xf32, 1>, i64) {
              ^bb0(%arg31: f32, %arg32: memref<1xf32, 1>, %arg33: memref<4096xbf16, 1>, %arg34: memref<64xf32, 1>, %arg35: memref<4096xf32, 1>, %arg36: i64):
                %c1_i64_68 = arith.constant 1 : i64
                %c0_i64_69 = arith.constant 0 : i64
                %c64_i64_70 = arith.constant 64 : i64
                %cst = arith.constant -1.000000e+00 : f32
                %c0_71 = arith.constant 0 : index
                %cst_72 = arith.constant 1.000000e+00 : f32
                %c64_i32_73 = arith.constant 64 : i32
                %mask, %remaining_out = ascvec.update_mask %c64_i32_73 : !ascvec.mask
                %297 = ascvec.duplicate_scalar %arg31, %mask : f32 -> <f32>
                %298 = ascvec.duplicate_scalar %cst_72, %mask : f32 -> <f32>
                %299 = ascvec.loadalign_brc %arg32[%c0_71] : memref<1xf32, 1>, index -> <f32>
                %300 = ascvec.exp %299, %mask : <f32>
                %301 = ascvec.mul_scalar %300, %cst, %mask : <f32>, f32
                scf.for %arg37 = %c0_i64_69 to %arg36 step %c1_i64_68  : i64 {
                  %302 = arith.muli %arg37, %c64_i64_70 : i64
                  %303 = arith.index_cast %302 : i64 to index
                  %304 = ascvec.loadalign_unpack %arg33[%303] {mode = "b16_to_b32"} : memref<4096xbf16, 1>, index -> <bf16>
                  %305 = ascvec.cast %304, %mask {rounding = 6 : i32} : <bf16> -> <f32>
                  %306 = ascvec.loadalign %arg34[%c0_71] : memref<64xf32, 1>, index -> <f32>
                  %307 = ascvec.add %305, %306, %mask : <f32>
                  %308 = ascvec.mul %301, %307, %mask : <f32>
                  %309 = ascvec.exp %308, %mask : <f32>
                  %310 = ascvec.add %298, %309, %mask : <f32>
                  %311 = ascvec.div %297, %310, %mask : <f32>
                  ascvec.storealign %arg35[%303], %311, %mask : memref<4096xf32, 1>, index, <f32>
                } {cannir.loop_unroll = 4 : i64}
                ascvec.mem_bar vst_vld
              } {cannir.vf_mode = "simd"}
              ascvec.sync_all
              ascvec.vf(%alloc_20, %alloc_40 : memref<4096xf32, 1>, memref<4096xf32, 1>) {
              ^bb0(%arg31: memref<4096xf32, 1>, %arg32: memref<4096xf32, 1>):
                %c1_i32_68 = arith.constant 1 : i32
                %c0_69 = arith.constant 0 : index
                %c64_i32_70 = arith.constant 64 : i32
                %mask, %remaining_out = ascvec.update_mask %c64_i32_70 : !ascvec.mask
                %297 = ascvec.loadalign %arg31[%c0_69] : memref<4096xf32, 1>, index -> <f32>
                ascvec.storealign %arg32[%c0_69], %297, %mask : memref<4096xf32, 1>, index, <f32>
                %298 = scf.for %arg33 = %c1_i32_68 to %c64_i32_70 step %c1_i32_68 iter_args(%arg34 = %297) -> (!ascvec.vreg<f32>)  : i32 {
                  %299 = arith.muli %arg33, %c64_i32_70 overflow<nsw> : i32
                  %300 = arith.index_cast %299 : i32 to index
                  %301 = ascvec.loadalign %arg31[%300] : memref<4096xf32, 1>, index -> <f32>
                  %302 = ascvec.add %arg34, %301, %mask : <f32>
                  ascvec.storealign %arg32[%300], %302, %mask : memref<4096xf32, 1>, index, <f32>
                  scf.yield %302 : !ascvec.vreg<f32>
                }
              } {cannir.vf_mode = "simd"}
              scf.if %183 {
                ascvec.sync_block_wait mte2, %c10_i64 {mode = 1 : i32}
                %297 = arith.muli %0, %c32768_i64 : i64
                %298 = arith.addi %297, %c16384_i64 : i64
                "ascvec.copy_gm2ub_plan"(%alloc_18, %alloc_4, %c0_i32, %c0_i32, %c0_i64, %298, %c1_i64, %c16384_i64, %c0_i64, %c16384_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 32768 : i64, ub_capacity = 16384 : i64}> : (memref<4096xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                ascvec.sync_all
                ascvec.sync_block_arrive mte2, %c11_i64 {mode = 1 : i32}
                ascvec.sync_block_wait mte3, %c11_i64 {mode = 1 : i32}
                ascvec.sync_notify {dst_pipe = 2 : i32, event_id = 3 : i32, src_pipe = 5 : i32}
                ascvec.sync_wait {dst_pipe = 2 : i32, event_id = 3 : i32, src_pipe = 5 : i32}
                ascvec.vf(%alloc_18, %alloc_26 : memref<4096xf32, 1>, memref<4096xbf16, 1>) {
                ^bb0(%arg31: memref<4096xf32, 1>, %arg32: memref<4096xbf16, 1>):
                  %c1_i32_68 = arith.constant 1 : i32
                  %c0_i32_69 = arith.constant 0 : i32
                  %c64_i32_70 = arith.constant 64 : i32
                  scf.for %arg33 = %c0_i32_69 to %c64_i32_70 step %c1_i32_68  : i32 {
                    %mask, %remaining_out = ascvec.update_mask %c64_i32_70 : !ascvec.mask
                    %299 = arith.muli %arg33, %c64_i32_70 overflow<nsw> : i32
                    %300 = arith.index_cast %299 : i32 to index
                    %301 = ascvec.loadalign %arg31[%300] : memref<4096xf32, 1>, index -> <f32>
                    %302 = ascvec.cast %301, %mask : <f32> -> <bf16>
                    ascvec.storealign_pack %arg32[%300], %302, %mask {mode = "b32_to_b16"} : memref<4096xbf16, 1>, index, <bf16>
                  }
                } {cannir.vf_mode = "simd"}
              } else {
                ascvec.sync_block_wait mte2, %c10_i64 {mode = 1 : i32}
                %297 = arith.muli %0, %c32768_i64 : i64
                "ascvec.copy_gm2ub_plan"(%alloc_18, %alloc_4, %c0_i32, %c0_i32, %c0_i64, %297, %c1_i64, %c16384_i64, %c0_i64, %c16384_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 32768 : i64, ub_capacity = 16384 : i64}> : (memref<4096xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                ascvec.sync_all
                ascvec.sync_block_arrive mte2, %c11_i64 {mode = 1 : i32}
                ascvec.sync_block_wait mte3, %c11_i64 {mode = 1 : i32}
                ascvec.sync_notify {dst_pipe = 2 : i32, event_id = 3 : i32, src_pipe = 5 : i32}
                ascvec.sync_wait {dst_pipe = 2 : i32, event_id = 3 : i32, src_pipe = 5 : i32}
                ascvec.vf(%alloc_18, %alloc_25 : memref<4096xf32, 1>, memref<4096xbf16, 1>) {
                ^bb0(%arg31: memref<4096xf32, 1>, %arg32: memref<4096xbf16, 1>):
                  %c1_i32_68 = arith.constant 1 : i32
                  %c0_i32_69 = arith.constant 0 : i32
                  %c64_i32_70 = arith.constant 64 : i32
                  scf.for %arg33 = %c0_i32_69 to %c64_i32_70 step %c1_i32_68  : i32 {
                    %mask, %remaining_out = ascvec.update_mask %c64_i32_70 : !ascvec.mask
                    %298 = arith.muli %arg33, %c64_i32_70 overflow<nsw> : i32
                    %299 = arith.index_cast %298 : i32 to index
                    %300 = ascvec.loadalign %arg31[%299] : memref<4096xf32, 1>, index -> <f32>
                    %301 = ascvec.cast %300, %mask : <f32> -> <bf16>
                    ascvec.storealign_pack %arg32[%299], %301, %mask {mode = "b32_to_b16"} : memref<4096xbf16, 1>, index, <bf16>
                  }
                } {cannir.vf_mode = "simd"}
              }
              %228 = arith.cmpi eq, %156, %c0_i64 : i64
              %229 = ascscalar.memref_address %alloc_41 : memref<4096xbf16, 1>
              %230 = ascscalar.memref_address %alloc_42 : memref<4096xbf16, 1>
              %231 = arith.select %228, %229, %230 : i64
              %232 = ascscalar.bind_local_ptr %231 : memref<4096xbf16, 1>
              %233 = ascscalar.memref_address %alloc_31 : memref<4096xf32, 1>
              %234 = ascscalar.memref_address %alloc_32 : memref<4096xf32, 1>
              %235 = arith.select %228, %233, %234 : i64
              %236 = ascscalar.bind_local_ptr %235 : memref<4096xf32, 1>
              %237 = arith.cmpi eq, %arg29, %c0_i64 : i64
              %238 = ascscalar.memref_address %alloc_29 : memref<64xf32, 1>
              %239 = ascscalar.memref_address %alloc_30 : memref<64xf32, 1>
              %240 = arith.select %237, %238, %239 : i64
              %241 = ascscalar.bind_local_ptr %240 : memref<64xf32, 1>
              %242 = arith.select %237, %c8_i64, %c9_i64 : i64
              %243 = arith.addi %arg29, %c1_i64 overflow<nsw> : i64
              %244 = arith.remui %243, %c2_i64 : i64
              ascvec.get_buf v, %242, 0 {cannir.buffer.generated_sync}
              ascvec.vf(%alloc_40, %241 : memref<4096xf32, 1>, memref<64xf32, 1>) {
              ^bb0(%arg31: memref<4096xf32, 1>, %arg32: memref<64xf32, 1>):
                %c4032 = arith.constant 4032 : index
                %c0_68 = arith.constant 0 : index
                %c64_i32_69 = arith.constant 64 : i32
                %c0_70 = arith.constant 0 : index
                %c1_71 = arith.constant 1 : index
                scf.for %arg33 = %c0_70 to %c1_71 step %c1_71 {
                  %mask, %remaining_out = ascvec.update_mask %c64_i32_69 : !ascvec.mask
                  %297 = ascvec.loadalign %arg31[%c4032] : memref<4096xf32, 1>, index -> <f32>
                  ascvec.storealign %arg32[%c0_68], %297, %mask : memref<64xf32, 1>, index, <f32>
                }
              } {cannir.vf_mode = "simd"}
              ascvec.release_buf v, %242, 0 {cannir.buffer.generated_sync}
              ascvec.copy_ub2ub %alloc_28[%c0], %alloc_40[%c0], %c16384_i32 : memref<4096xf32, 1>, memref<4096xf32, 1>
              ascvec.vf(%alloc_28, %alloc_40 : memref<4096xf32, 1>, memref<4096xf32, 1>) {
              ^bb0(%arg31: memref<4096xf32, 1>, %arg32: memref<4096xf32, 1>):
              } {cannir.vf_mode = "simd"}
              ascvec.copy_ub2ub %236[%c0], %alloc_40[%c0], %c16384_i32 : memref<4096xf32, 1>, memref<4096xf32, 1>
              ascvec.vf(%236, %alloc_40 : memref<4096xf32, 1>, memref<4096xf32, 1>) {
              ^bb0(%arg31: memref<4096xf32, 1>, %arg32: memref<4096xf32, 1>):
              } {cannir.vf_mode = "simd"}
              ascvec.copy_ub2ub %232[%c0], %alloc_26[%c0], %c8192_i32 : memref<4096xbf16, 1>, memref<4096xbf16, 1>
              ascvec.vf(%232, %alloc_26 : memref<4096xbf16, 1>, memref<4096xbf16, 1>) {
              ^bb0(%arg31: memref<4096xbf16, 1>, %arg32: memref<4096xbf16, 1>):
              } {cannir.vf_mode = "simd"}
              ascvec.get_buf v, %c12_i64, 0 {cannir.buffer.generated_sync}
              ascvec.vf(%alloc_28, %alloc_25, %arg20, %alloc_37 : memref<4096xf32, 1>, memref<4096xbf16, 1>, f32, memref<4096xbf16, 1>) {
              ^bb0(%arg31: memref<4096xf32, 1>, %arg32: memref<4096xbf16, 1>, %arg33: f32, %arg34: memref<4096xbf16, 1>):
                %c1_i32_68 = arith.constant 1 : i32
                %c0_i32_69 = arith.constant 0 : i32
                %c64_i32_70 = arith.constant 64 : i32
                %mask, %remaining_out = ascvec.update_mask %c64_i32_70 : !ascvec.mask
                scf.for %arg35 = %c0_i32_69 to %c64_i32_70 step %c1_i32_68  : i32 {
                  %297 = arith.muli %arg35, %c64_i32_70 overflow<nsw> : i32
                  %298 = arith.index_cast %297 : i32 to index
                  %299 = ascvec.loadalign %arg31[%298] : memref<4096xf32, 1>, index -> <f32>
                  %300 = ascvec.exp %299, %mask : <f32>
                  %301 = ascvec.loadalign_unpack %arg32[%298] {mode = "b16_to_b32"} : memref<4096xbf16, 1>, index -> <bf16>
                  %302 = ascvec.cast %301, %mask {rounding = 6 : i32} : <bf16> -> <f32>
                  %303 = ascvec.mul %302, %300, %mask : <f32>
                  %304 = ascvec.mul_scalar %303, %arg33, %mask : <f32>, f32
                  %305 = ascvec.cast %304, %mask : <f32> -> <bf16>
                  ascvec.storealign_pack %arg34[%298], %305, %mask {mode = "b32_to_b16"} : memref<4096xbf16, 1>, index, <bf16>
                } {cannir.loop_unroll = 4 : i64}
              } {cannir.vf_mode = "simd"}
              ascvec.release_buf v, %c12_i64, 0 {cannir.buffer.generated_sync}
              %245 = cannir.extract_physical_layout %arg10 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
              %246 = cannir.extract_stride %245 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
              %247 = cannir.extract_dim %246[0] : !cannir.stride<(-1, -1, -1, 1)>
              %248 = arith.muli %97, %247 : i64
              %249 = cannir.extract_dim %246[1] : !cannir.stride<(-1, -1, -1, 1)>
              %250 = arith.muli %150, %249 : i64
              %251 = arith.addi %248, %250 : i64
              %252 = cannir.extract_dim %246[2] : !cannir.stride<(-1, -1, -1, 1)>
              %253 = arith.muli %205, %252 : i64
              %254 = arith.addi %251, %253 : i64
              %255 = arith.muli %102, %c64_i64 : i64
              %256 = arith.addi %254, %255 : i64
              %257 = arith.muli %256, %c16_i64 : i64
              %258 = ascscalar.memref_address %alloc_9 : memref<1xbf16>
              %259 = arith.divui %257, %c8_i64 : i64
              %260 = arith.addi %258, %259 : i64
              %261 = ascscalar.bind_gm_ptr %260 : memref<1xbf16>
              %262 = ascscalar.bitcast_memref %alloc_37 byte_offset = %c0_i64 : memref<4096xbf16, 1> -> memref<?xbf16, 1>
              %263 = cannir.extract_layout %arg10 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
              %264 = cannir.extract_stride %263 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
              %265 = cannir.extract_dim %264[2] : !cannir.stride<(-1, -1, -1, 1)>
              %266 = arith.index_cast %265 : i64 to index
              %267 = arith.muli %266, %c2 : index
              %268 = arith.cmpi ugt, %190, %c0_i64 : i64
              scf.if %268 {
                %297 = arith.index_cast %267 : index to i64
                ascvec.get_buf mte3, %c12_i64, 0 {cannir.buffer.generated_sync}
                %298 = arith.trunci %190 : i64 to i32
                ascvec.copy_ub2gm %261[%c0], %262[%c0], %298, %c128_i32, %297, %c128_i32 l2_cache_ctl %c0_i32 : memref<1xbf16>, memref<?xbf16, 1>
                ascvec.release_buf mte3, %c12_i64, 0 {cannir.buffer.generated_sync}
              }
              ascvec.vf(%alloc_40, %alloc_26, %alloc_45 : memref<4096xf32, 1>, memref<4096xbf16, 1>, memref<1024xbf16, 1>) {
              ^bb0(%arg31: memref<4096xf32, 1>, %arg32: memref<4096xbf16, 1>, %arg33: memref<1024xbf16, 1>):
                %c1_i32_68 = arith.constant 1 : i32
                %c16_i32_69 = arith.constant 16 : i32
                %c0_i32_70 = arith.constant 0 : i32
                %c0_71 = arith.constant 0 : index
                %c64_i32_72 = arith.constant 64 : i32
                %mask, %remaining_out = ascvec.update_mask %c64_i32_72 : !ascvec.mask
                %297 = ascvec.loadalign %arg31[%c0_71] : memref<4096xf32, 1>, index -> <f32>
                scf.for %arg34 = %c0_i32_70 to %c16_i32_69 step %c1_i32_68  : i32 {
                  %298 = arith.muli %arg34, %c64_i32_72 overflow<nsw> : i32
                  %299 = arith.index_cast %298 : i32 to index
                  %300 = ascvec.loadalign_unpack %arg32[%299] {mode = "b16_to_b32"} : memref<4096xbf16, 1>, index -> <bf16>
                  %301 = ascvec.cast %300, %mask {rounding = 6 : i32} : <bf16> -> <f32>
                  %302 = ascvec.loadalign %arg31[%299] : memref<4096xf32, 1>, index -> <f32>
                  %303 = ascvec.sub %302, %297, %mask : <f32>
                  %304 = ascvec.exp %303, %mask : <f32>
                  %305 = ascvec.mul %301, %304, %mask : <f32>
                  %306 = ascvec.cast %305, %mask : <f32> -> <bf16>
                  ascvec.storealign_pack %arg33[%299], %306, %mask {mode = "b32_to_b16"} : memref<1024xbf16, 1>, index, <bf16>
                } {cannir.loop_unroll = 2 : i64}
              } {cannir.vf_mode = "simd"}
              ascvec.get_buf v, %c15_i64, 0 {cannir.buffer.generated_sync}
              ascvec.vf(%alloc_45, %alloc_47 : memref<1024xbf16, 1>, memref<1088xbf16, 1>) {
              ^bb0(%arg31: memref<1024xbf16, 1>, %arg32: memref<1088xbf16, 1>):
                %c1_i32_68 = arith.constant 1 : i32
                %c0_i32_69 = arith.constant 0 : i32
                %c17 = arith.constant 17 : index
                %c16_i32_70 = arith.constant 16 : i32
                %c0_71 = arith.constant 0 : index
                %c64_i32_72 = arith.constant 64 : i32
                scf.for %arg33 = %c0_i32_69 to %c16_i32_70 step %c1_i32_68  : i32 {
                  %mask, %remaining_out = ascvec.update_mask %c64_i32_72 elem_bits 16 : !ascvec.mask
                  %297 = arith.muli %arg33, %c64_i32_72 overflow<nsw> : i32
                  %298 = arith.index_cast %297 : i32 to index
                  %299 = ascvec.loadalign %arg31[%298] : memref<1024xbf16, 1>, index -> <bf16>
                  %300 = arith.muli %arg33, %c16_i32_70 overflow<nsw> : i32
                  %301 = arith.index_cast %300 : i32 to index
                  ascvec.storealign_strided %arg32[%301], %299, %mask, %c17, %c0_71 : memref<1088xbf16, 1>, <bf16>
                }
              } {cannir.vf_mode = "simd"}
              ascvec.release_buf v, %c15_i64, 0 {cannir.buffer.generated_sync}
              %269 = ascscalar.bitcast_memref %alloc_47 byte_offset = %c0_i64 : memref<1088xbf16, 1> -> memref<?xbf16, 1>
              %270 = ascscalar.bitcast_memref %alloc_67 byte_offset = %c0_i64 : memref<2048xbf16, 2> -> memref<?xbf16, 2>
              %271 = arith.index_cast %255 : i64 to index
              %272 = arith.divui %271, %c16 : index
              %273 = arith.muli %272, %c256 : index
              ascvec.sync_intra_wait mte3, %c5_i64 {cannir.sync.banks = 3 : i32}
              ascvec.get_buf mte3, %c15_i64, 0 {cannir.buffer.generated_sync}
              ascvec.copy_ub2l1_strided %270[%273], %269[%c0], %c4_i32, %c16_i32, %c1_i32, %c0_i32 : memref<?xbf16, 2>, memref<?xbf16, 1>
              ascvec.release_buf mte3, %c15_i64, 0 {cannir.buffer.generated_sync}
              ascvec.sync_intra_arrive mte3, %c5_i64 {cannir.sync.banks = 3 : i32}
              ascvec.vf(%alloc_40, %alloc_26, %alloc_46 : memref<4096xf32, 1>, memref<4096xbf16, 1>, memref<1024xbf16, 1>) {
              ^bb0(%arg31: memref<4096xf32, 1>, %arg32: memref<4096xbf16, 1>, %arg33: memref<1024xbf16, 1>):
                %c1_i32_68 = arith.constant 1 : i32
                %c16_i32_69 = arith.constant 16 : i32
                %c0_i32_70 = arith.constant 0 : i32
                %c0_71 = arith.constant 0 : index
                %c64_i32_72 = arith.constant 64 : i32
                %mask, %remaining_out = ascvec.update_mask %c64_i32_72 : !ascvec.mask
                %297 = ascvec.loadalign %arg31[%c0_71] : memref<4096xf32, 1>, index -> <f32>
                scf.for %arg34 = %c0_i32_70 to %c16_i32_69 step %c1_i32_68  : i32 {
                  %298 = arith.muli %arg34, %c64_i32_72 overflow<nsw> : i32
                  %299 = arith.index_cast %298 : i32 to index
                  %300 = ascvec.loadalign_unpack %arg32[%299] {mode = "b16_to_b32"} : memref<4096xbf16, 1>, index -> <bf16>
                  %301 = ascvec.cast %300, %mask {rounding = 6 : i32} : <bf16> -> <f32>
                  %302 = ascvec.loadalign %arg31[%299] : memref<4096xf32, 1>, index -> <f32>
                  %303 = ascvec.sub %297, %302, %mask : <f32>
                  %304 = ascvec.exp %303, %mask : <f32>
                  %305 = ascvec.mul %301, %304, %mask : <f32>
                  %306 = ascvec.cast %305, %mask : <f32> -> <bf16>
                  ascvec.storealign_pack %arg33[%299], %306, %mask {mode = "b32_to_b16"} : memref<1024xbf16, 1>, index, <bf16>
                } {cannir.loop_unroll = 2 : i64}
              } {cannir.vf_mode = "simd"}
              ascvec.get_buf v, %c16_i64, 0 {cannir.buffer.generated_sync}
              ascvec.vf(%alloc_46, %alloc_48 : memref<1024xbf16, 1>, memref<1088xbf16, 1>) {
              ^bb0(%arg31: memref<1024xbf16, 1>, %arg32: memref<1088xbf16, 1>):
                %c1_i32_68 = arith.constant 1 : i32
                %c0_i32_69 = arith.constant 0 : i32
                %c17 = arith.constant 17 : index
                %c16_i32_70 = arith.constant 16 : i32
                %c0_71 = arith.constant 0 : index
                %c64_i32_72 = arith.constant 64 : i32
                scf.for %arg33 = %c0_i32_69 to %c16_i32_70 step %c1_i32_68  : i32 {
                  %mask, %remaining_out = ascvec.update_mask %c64_i32_72 elem_bits 16 : !ascvec.mask
                  %297 = arith.muli %arg33, %c64_i32_72 overflow<nsw> : i32
                  %298 = arith.index_cast %297 : i32 to index
                  %299 = ascvec.loadalign %arg31[%298] : memref<1024xbf16, 1>, index -> <bf16>
                  %300 = arith.muli %arg33, %c16_i32_70 overflow<nsw> : i32
                  %301 = arith.index_cast %300 : i32 to index
                  ascvec.storealign_strided %arg32[%301], %299, %mask, %c17, %c0_71 : memref<1088xbf16, 1>, <bf16>
                }
              } {cannir.vf_mode = "simd"}
              ascvec.release_buf v, %c16_i64, 0 {cannir.buffer.generated_sync}
              %274 = ascscalar.bitcast_memref %alloc_48 byte_offset = %c0_i64 : memref<1088xbf16, 1> -> memref<?xbf16, 1>
              %275 = ascscalar.bitcast_memref %alloc_66 byte_offset = %c0_i64 : memref<2048xbf16, 2> -> memref<?xbf16, 2>
              ascvec.sync_intra_wait mte3, %c6_i64 {cannir.sync.banks = 3 : i32}
              ascvec.get_buf mte3, %c16_i64, 0 {cannir.buffer.generated_sync}
              ascvec.copy_ub2l1_strided %275[%273], %274[%c0], %c4_i32, %c16_i32, %c1_i32, %c0_i32 : memref<?xbf16, 2>, memref<?xbf16, 1>
              ascvec.release_buf mte3, %c16_i64, 0 {cannir.buffer.generated_sync}
              ascvec.sync_intra_arrive mte3, %c6_i64 {cannir.sync.banks = 3 : i32}
              ascvec.vf(%alloc_40, %alloc_25, %arg20, %alloc_45 : memref<4096xf32, 1>, memref<4096xbf16, 1>, f32, memref<1024xbf16, 1>) {
              ^bb0(%arg31: memref<4096xf32, 1>, %arg32: memref<4096xbf16, 1>, %arg33: f32, %arg34: memref<1024xbf16, 1>):
                %c1_i32_68 = arith.constant 1 : i32
                %c16_i32_69 = arith.constant 16 : i32
                %c0_i32_70 = arith.constant 0 : i32
                %c0_71 = arith.constant 0 : index
                %c64_i32_72 = arith.constant 64 : i32
                %mask, %remaining_out = ascvec.update_mask %c64_i32_72 : !ascvec.mask
                %297 = ascvec.loadalign %arg31[%c0_71] : memref<4096xf32, 1>, index -> <f32>
                scf.for %arg35 = %c0_i32_70 to %c16_i32_69 step %c1_i32_68  : i32 {
                  %298 = arith.muli %arg35, %c64_i32_72 overflow<nsw> : i32
                  %299 = arith.index_cast %298 : i32 to index
                  %300 = ascvec.loadalign_unpack %arg32[%299] {mode = "b16_to_b32"} : memref<4096xbf16, 1>, index -> <bf16>
                  %301 = ascvec.cast %300, %mask {rounding = 6 : i32} : <bf16> -> <f32>
                  %302 = ascvec.loadalign %arg31[%299] : memref<4096xf32, 1>, index -> <f32>
                  %303 = ascvec.sub %302, %297, %mask : <f32>
                  %304 = ascvec.exp %303, %mask : <f32>
                  %305 = ascvec.mul %301, %304, %mask : <f32>
                  %306 = ascvec.mul_scalar %305, %arg33, %mask : <f32>, f32
                  %307 = ascvec.cast %306, %mask : <f32> -> <bf16>
                  ascvec.storealign_pack %arg34[%299], %307, %mask {mode = "b32_to_b16"} : memref<1024xbf16, 1>, index, <bf16>
                } {cannir.loop_unroll = 2 : i64}
              } {cannir.vf_mode = "simd"}
              ascvec.get_buf v, %c15_i64, 0 {cannir.buffer.generated_sync}
              ascvec.vf(%alloc_45, %alloc_47 : memref<1024xbf16, 1>, memref<1088xbf16, 1>) {
              ^bb0(%arg31: memref<1024xbf16, 1>, %arg32: memref<1088xbf16, 1>):
                %c1_i32_68 = arith.constant 1 : i32
                %c0_i32_69 = arith.constant 0 : i32
                %c17 = arith.constant 17 : index
                %c16_i32_70 = arith.constant 16 : i32
                %c0_71 = arith.constant 0 : index
                %c64_i32_72 = arith.constant 64 : i32
                scf.for %arg33 = %c0_i32_69 to %c16_i32_70 step %c1_i32_68  : i32 {
                  %mask, %remaining_out = ascvec.update_mask %c64_i32_72 elem_bits 16 : !ascvec.mask
                  %297 = arith.muli %arg33, %c64_i32_72 overflow<nsw> : i32
                  %298 = arith.index_cast %297 : i32 to index
                  %299 = ascvec.loadalign %arg31[%298] : memref<1024xbf16, 1>, index -> <bf16>
                  %300 = arith.muli %arg33, %c16_i32_70 overflow<nsw> : i32
                  %301 = arith.index_cast %300 : i32 to index
                  ascvec.storealign_strided %arg32[%301], %299, %mask, %c17, %c0_71 : memref<1088xbf16, 1>, <bf16>
                }
              } {cannir.vf_mode = "simd"}
              ascvec.release_buf v, %c15_i64, 0 {cannir.buffer.generated_sync}
              %276 = ascscalar.bitcast_memref %alloc_65 byte_offset = %c0_i64 : memref<2048xbf16, 2> -> memref<?xbf16, 2>
              ascvec.sync_intra_wait mte3, %c7_i64 {cannir.sync.banks = 3 : i32}
              ascvec.get_buf mte3, %c15_i64, 0 {cannir.buffer.generated_sync}
              ascvec.copy_ub2l1_strided %276[%273], %269[%c0], %c4_i32, %c16_i32, %c1_i32, %c0_i32 : memref<?xbf16, 2>, memref<?xbf16, 1>
              ascvec.release_buf mte3, %c15_i64, 0 {cannir.buffer.generated_sync}
              ascvec.sync_intra_arrive mte3, %c7_i64 {cannir.sync.banks = 3 : i32}
              ascvec.vf(%alloc_40, %alloc_26, %alloc_46 : memref<4096xf32, 1>, memref<4096xbf16, 1>, memref<1024xbf16, 1>) {
              ^bb0(%arg31: memref<4096xf32, 1>, %arg32: memref<4096xbf16, 1>, %arg33: memref<1024xbf16, 1>):
                %c1_i32_68 = arith.constant 1 : i32
                %c0_i32_69 = arith.constant 0 : i32
                %c16_i32_70 = arith.constant 16 : i32
                %c1024_71 = arith.constant 1024 : index
                %c64_i32_72 = arith.constant 64 : i32
                %mask, %remaining_out = ascvec.update_mask %c64_i32_72 : !ascvec.mask
                %297 = ascvec.loadalign %arg31[%c1024_71] : memref<4096xf32, 1>, index -> <f32>
                scf.for %arg34 = %c0_i32_69 to %c16_i32_70 step %c1_i32_68  : i32 {
                  %298 = arith.muli %arg34, %c64_i32_72 overflow<nsw> : i32
                  %299 = arith.addi %arg34, %c16_i32_70 overflow<nsw> : i32
                  %300 = arith.muli %299, %c64_i32_72 overflow<nsw> : i32
                  %301 = arith.index_cast %298 : i32 to index
                  %302 = arith.addi %301, %c1024_71 : index
                  %303 = ascvec.loadalign_unpack %arg32[%302] {mode = "b16_to_b32"} : memref<4096xbf16, 1>, index -> <bf16>
                  %304 = ascvec.cast %303, %mask {rounding = 6 : i32} : <bf16> -> <f32>
                  %305 = arith.index_cast %300 : i32 to index
                  %306 = ascvec.loadalign %arg31[%305] : memref<4096xf32, 1>, index -> <f32>
                  %307 = ascvec.sub %306, %297, %mask : <f32>
                  %308 = ascvec.exp %307, %mask : <f32>
                  %309 = ascvec.mul %304, %308, %mask : <f32>
                  %310 = ascvec.cast %309, %mask : <f32> -> <bf16>
                  ascvec.storealign_pack %arg33[%301], %310, %mask {mode = "b32_to_b16"} : memref<1024xbf16, 1>, index, <bf16>
                } {cannir.loop_unroll = 2 : i64}
              } {cannir.vf_mode = "simd"}
              ascvec.get_buf v, %c16_i64, 0 {cannir.buffer.generated_sync}
              ascvec.vf(%alloc_46, %alloc_48 : memref<1024xbf16, 1>, memref<1088xbf16, 1>) {
              ^bb0(%arg31: memref<1024xbf16, 1>, %arg32: memref<1088xbf16, 1>):
                %c1_i32_68 = arith.constant 1 : i32
                %c0_i32_69 = arith.constant 0 : i32
                %c17 = arith.constant 17 : index
                %c16_i32_70 = arith.constant 16 : i32
                %c0_71 = arith.constant 0 : index
                %c64_i32_72 = arith.constant 64 : i32
                scf.for %arg33 = %c0_i32_69 to %c16_i32_70 step %c1_i32_68  : i32 {
                  %mask, %remaining_out = ascvec.update_mask %c64_i32_72 elem_bits 16 : !ascvec.mask
                  %297 = arith.muli %arg33, %c64_i32_72 overflow<nsw> : i32
                  %298 = arith.index_cast %297 : i32 to index
                  %299 = ascvec.loadalign %arg31[%298] : memref<1024xbf16, 1>, index -> <bf16>
                  %300 = arith.muli %arg33, %c16_i32_70 overflow<nsw> : i32
                  %301 = arith.index_cast %300 : i32 to index
                  ascvec.storealign_strided %arg32[%301], %299, %mask, %c17, %c0_71 : memref<1088xbf16, 1>, <bf16>
                }
              } {cannir.vf_mode = "simd"}
              ascvec.release_buf v, %c16_i64, 0 {cannir.buffer.generated_sync}
              ascvec.sync_intra_wait mte3, %c5_i64 {cannir.sync.banks = 3 : i32}
              ascvec.get_buf mte3, %c16_i64, 0 {cannir.buffer.generated_sync}
              ascvec.copy_ub2l1_strided %270[%273], %274[%c0], %c4_i32, %c16_i32, %c1_i32, %c0_i32 : memref<?xbf16, 2>, memref<?xbf16, 1>
              ascvec.release_buf mte3, %c16_i64, 0 {cannir.buffer.generated_sync}
              ascvec.sync_intra_arrive mte3, %c5_i64 {cannir.sync.banks = 3 : i32}
              ascvec.vf(%alloc_40, %alloc_26, %alloc_45 : memref<4096xf32, 1>, memref<4096xbf16, 1>, memref<1024xbf16, 1>) {
              ^bb0(%arg31: memref<4096xf32, 1>, %arg32: memref<4096xbf16, 1>, %arg33: memref<1024xbf16, 1>):
                %c1024_68 = arith.constant 1024 : index
                %c1_i32_69 = arith.constant 1 : i32
                %c16_i32_70 = arith.constant 16 : i32
                %c0_i32_71 = arith.constant 0 : i32
                %c64_i32_72 = arith.constant 64 : i32
                %mask, %remaining_out = ascvec.update_mask %c64_i32_72 : !ascvec.mask
                %297 = ascvec.loadalign %arg31[%c1024_68] : memref<4096xf32, 1>, index -> <f32>
                scf.for %arg34 = %c0_i32_71 to %c16_i32_70 step %c1_i32_69  : i32 {
                  %298 = arith.muli %arg34, %c64_i32_72 overflow<nsw> : i32
                  %299 = arith.index_cast %298 : i32 to index
                  %300 = ascvec.loadalign_unpack %arg32[%299] {mode = "b16_to_b32"} : memref<4096xbf16, 1>, index -> <bf16>
                  %301 = ascvec.cast %300, %mask {rounding = 6 : i32} : <bf16> -> <f32>
                  %302 = ascvec.loadalign %arg31[%299] : memref<4096xf32, 1>, index -> <f32>
                  %303 = ascvec.sub %297, %302, %mask : <f32>
                  %304 = ascvec.exp %303, %mask : <f32>
                  %305 = ascvec.mul %301, %304, %mask : <f32>
                  %306 = ascvec.cast %305, %mask : <f32> -> <bf16>
                  ascvec.storealign_pack %arg33[%299], %306, %mask {mode = "b32_to_b16"} : memref<1024xbf16, 1>, index, <bf16>
                } {cannir.loop_unroll = 2 : i64}
              } {cannir.vf_mode = "simd"}
              ascvec.get_buf v, %c15_i64, 0 {cannir.buffer.generated_sync}
              ascvec.vf(%alloc_45, %alloc_47 : memref<1024xbf16, 1>, memref<1088xbf16, 1>) {
              ^bb0(%arg31: memref<1024xbf16, 1>, %arg32: memref<1088xbf16, 1>):
                %c1_i32_68 = arith.constant 1 : i32
                %c0_i32_69 = arith.constant 0 : i32
                %c17 = arith.constant 17 : index
                %c16_i32_70 = arith.constant 16 : i32
                %c0_71 = arith.constant 0 : index
                %c64_i32_72 = arith.constant 64 : i32
                scf.for %arg33 = %c0_i32_69 to %c16_i32_70 step %c1_i32_68  : i32 {
                  %mask, %remaining_out = ascvec.update_mask %c64_i32_72 elem_bits 16 : !ascvec.mask
                  %297 = arith.muli %arg33, %c64_i32_72 overflow<nsw> : i32
                  %298 = arith.index_cast %297 : i32 to index
                  %299 = ascvec.loadalign %arg31[%298] : memref<1024xbf16, 1>, index -> <bf16>
                  %300 = arith.muli %arg33, %c16_i32_70 overflow<nsw> : i32
                  %301 = arith.index_cast %300 : i32 to index
                  ascvec.storealign_strided %arg32[%301], %299, %mask, %c17, %c0_71 : memref<1088xbf16, 1>, <bf16>
                }
              } {cannir.vf_mode = "simd"}
              ascvec.release_buf v, %c15_i64, 0 {cannir.buffer.generated_sync}
              ascvec.sync_intra_wait mte3, %c6_i64 {cannir.sync.banks = 3 : i32}
              ascvec.get_buf mte3, %c15_i64, 0 {cannir.buffer.generated_sync}
              ascvec.copy_ub2l1_strided %275[%273], %269[%c0], %c4_i32, %c16_i32, %c1_i32, %c0_i32 : memref<?xbf16, 2>, memref<?xbf16, 1>
              ascvec.release_buf mte3, %c15_i64, 0 {cannir.buffer.generated_sync}
              ascvec.sync_intra_arrive mte3, %c6_i64 {cannir.sync.banks = 3 : i32}
              ascvec.vf(%alloc_40, %alloc_25, %arg20, %alloc_46 : memref<4096xf32, 1>, memref<4096xbf16, 1>, f32, memref<1024xbf16, 1>) {
              ^bb0(%arg31: memref<4096xf32, 1>, %arg32: memref<4096xbf16, 1>, %arg33: f32, %arg34: memref<1024xbf16, 1>):
                %c1_i32_68 = arith.constant 1 : i32
                %c0_i32_69 = arith.constant 0 : i32
                %c16_i32_70 = arith.constant 16 : i32
                %c1024_71 = arith.constant 1024 : index
                %c64_i32_72 = arith.constant 64 : i32
                %mask, %remaining_out = ascvec.update_mask %c64_i32_72 : !ascvec.mask
                %297 = ascvec.loadalign %arg31[%c1024_71] : memref<4096xf32, 1>, index -> <f32>
                scf.for %arg35 = %c0_i32_69 to %c16_i32_70 step %c1_i32_68  : i32 {
                  %298 = arith.muli %arg35, %c64_i32_72 overflow<nsw> : i32
                  %299 = arith.addi %arg35, %c16_i32_70 overflow<nsw> : i32
                  %300 = arith.muli %299, %c64_i32_72 overflow<nsw> : i32
                  %301 = arith.index_cast %298 : i32 to index
                  %302 = arith.addi %301, %c1024_71 : index
                  %303 = ascvec.loadalign_unpack %arg32[%302] {mode = "b16_to_b32"} : memref<4096xbf16, 1>, index -> <bf16>
                  %304 = ascvec.cast %303, %mask {rounding = 6 : i32} : <bf16> -> <f32>
                  %305 = arith.index_cast %300 : i32 to index
                  %306 = ascvec.loadalign %arg31[%305] : memref<4096xf32, 1>, index -> <f32>
                  %307 = ascvec.sub %306, %297, %mask : <f32>
                  %308 = ascvec.exp %307, %mask : <f32>
                  %309 = ascvec.mul %304, %308, %mask : <f32>
                  %310 = ascvec.mul_scalar %309, %arg33, %mask : <f32>, f32
                  %311 = ascvec.cast %310, %mask : <f32> -> <bf16>
                  ascvec.storealign_pack %arg34[%301], %311, %mask {mode = "b32_to_b16"} : memref<1024xbf16, 1>, index, <bf16>
                } {cannir.loop_unroll = 2 : i64}
              } {cannir.vf_mode = "simd"}
              ascvec.get_buf v, %c16_i64, 0 {cannir.buffer.generated_sync}
              ascvec.vf(%alloc_46, %alloc_48 : memref<1024xbf16, 1>, memref<1088xbf16, 1>) {
              ^bb0(%arg31: memref<1024xbf16, 1>, %arg32: memref<1088xbf16, 1>):
                %c1_i32_68 = arith.constant 1 : i32
                %c0_i32_69 = arith.constant 0 : i32
                %c17 = arith.constant 17 : index
                %c16_i32_70 = arith.constant 16 : i32
                %c0_71 = arith.constant 0 : index
                %c64_i32_72 = arith.constant 64 : i32
                scf.for %arg33 = %c0_i32_69 to %c16_i32_70 step %c1_i32_68  : i32 {
                  %mask, %remaining_out = ascvec.update_mask %c64_i32_72 elem_bits 16 : !ascvec.mask
                  %297 = arith.muli %arg33, %c64_i32_72 overflow<nsw> : i32
                  %298 = arith.index_cast %297 : i32 to index
                  %299 = ascvec.loadalign %arg31[%298] : memref<1024xbf16, 1>, index -> <bf16>
                  %300 = arith.muli %arg33, %c16_i32_70 overflow<nsw> : i32
                  %301 = arith.index_cast %300 : i32 to index
                  ascvec.storealign_strided %arg32[%301], %299, %mask, %c17, %c0_71 : memref<1088xbf16, 1>, <bf16>
                }
              } {cannir.vf_mode = "simd"}
              ascvec.release_buf v, %c16_i64, 0 {cannir.buffer.generated_sync}
              ascvec.sync_intra_wait mte3, %c7_i64 {cannir.sync.banks = 3 : i32}
              ascvec.get_buf mte3, %c16_i64, 0 {cannir.buffer.generated_sync}
              ascvec.copy_ub2l1_strided %276[%273], %274[%c0], %c4_i32, %c16_i32, %c1_i32, %c0_i32 : memref<?xbf16, 2>, memref<?xbf16, 1>
              ascvec.release_buf mte3, %c16_i64, 0 {cannir.buffer.generated_sync}
              ascvec.sync_intra_arrive mte3, %c7_i64 {cannir.sync.banks = 3 : i32}
              ascvec.vf(%alloc_40, %alloc_26, %alloc_45 : memref<4096xf32, 1>, memref<4096xbf16, 1>, memref<1024xbf16, 1>) {
              ^bb0(%arg31: memref<4096xf32, 1>, %arg32: memref<4096xbf16, 1>, %arg33: memref<1024xbf16, 1>):
                %c1_i32_68 = arith.constant 1 : i32
                %c0_i32_69 = arith.constant 0 : i32
                %c16_i32_70 = arith.constant 16 : i32
                %c1024_71 = arith.constant 1024 : index
                %c64_i32_72 = arith.constant 64 : i32
                %mask, %remaining_out = ascvec.update_mask %c64_i32_72 : !ascvec.mask
                %297 = ascvec.loadalign %arg31[%c1024_71] : memref<4096xf32, 1>, index -> <f32>
                scf.for %arg34 = %c0_i32_69 to %c16_i32_70 step %c1_i32_68  : i32 {
                  %298 = arith.muli %arg34, %c64_i32_72 overflow<nsw> : i32
                  %299 = arith.addi %arg34, %c16_i32_70 overflow<nsw> : i32
                  %300 = arith.muli %299, %c64_i32_72 overflow<nsw> : i32
                  %301 = arith.index_cast %298 : i32 to index
                  %302 = arith.addi %301, %c1024_71 : index
                  %303 = ascvec.loadalign_unpack %arg32[%302] {mode = "b16_to_b32"} : memref<4096xbf16, 1>, index -> <bf16>
                  %304 = ascvec.cast %303, %mask {rounding = 6 : i32} : <bf16> -> <f32>
                  %305 = arith.index_cast %300 : i32 to index
                  %306 = ascvec.loadalign %arg31[%305] : memref<4096xf32, 1>, index -> <f32>
                  %307 = ascvec.sub %297, %306, %mask : <f32>
                  %308 = ascvec.exp %307, %mask : <f32>
                  %309 = ascvec.mul %304, %308, %mask : <f32>
                  %310 = ascvec.cast %309, %mask : <f32> -> <bf16>
                  ascvec.storealign_pack %arg33[%301], %310, %mask {mode = "b32_to_b16"} : memref<1024xbf16, 1>, index, <bf16>
                } {cannir.loop_unroll = 2 : i64}
              } {cannir.vf_mode = "simd"}
              ascvec.get_buf v, %c15_i64, 0 {cannir.buffer.generated_sync}
              ascvec.vf(%alloc_45, %alloc_47 : memref<1024xbf16, 1>, memref<1088xbf16, 1>) {
              ^bb0(%arg31: memref<1024xbf16, 1>, %arg32: memref<1088xbf16, 1>):
                %c1_i32_68 = arith.constant 1 : i32
                %c0_i32_69 = arith.constant 0 : i32
                %c17 = arith.constant 17 : index
                %c16_i32_70 = arith.constant 16 : i32
                %c0_71 = arith.constant 0 : index
                %c64_i32_72 = arith.constant 64 : i32
                scf.for %arg33 = %c0_i32_69 to %c16_i32_70 step %c1_i32_68  : i32 {
                  %mask, %remaining_out = ascvec.update_mask %c64_i32_72 elem_bits 16 : !ascvec.mask
                  %297 = arith.muli %arg33, %c64_i32_72 overflow<nsw> : i32
                  %298 = arith.index_cast %297 : i32 to index
                  %299 = ascvec.loadalign %arg31[%298] : memref<1024xbf16, 1>, index -> <bf16>
                  %300 = arith.muli %arg33, %c16_i32_70 overflow<nsw> : i32
                  %301 = arith.index_cast %300 : i32 to index
                  ascvec.storealign_strided %arg32[%301], %299, %mask, %c17, %c0_71 : memref<1088xbf16, 1>, <bf16>
                }
              } {cannir.vf_mode = "simd"}
              ascvec.release_buf v, %c15_i64, 0 {cannir.buffer.generated_sync}
              ascvec.sync_intra_wait mte3, %c5_i64 {cannir.sync.banks = 3 : i32}
              ascvec.get_buf mte3, %c15_i64, 0 {cannir.buffer.generated_sync}
              ascvec.copy_ub2l1_strided %270[%273], %269[%c0], %c4_i32, %c16_i32, %c1_i32, %c0_i32 : memref<?xbf16, 2>, memref<?xbf16, 1>
              ascvec.release_buf mte3, %c15_i64, 0 {cannir.buffer.generated_sync}
              ascvec.sync_intra_arrive mte3, %c5_i64 {cannir.sync.banks = 3 : i32}
              ascvec.vf(%alloc_40, %alloc_26, %alloc_46 : memref<4096xf32, 1>, memref<4096xbf16, 1>, memref<1024xbf16, 1>) {
              ^bb0(%arg31: memref<4096xf32, 1>, %arg32: memref<4096xbf16, 1>, %arg33: memref<1024xbf16, 1>):
                %c1_i32_68 = arith.constant 1 : i32
                %c16_i32_69 = arith.constant 16 : i32
                %c0_i32_70 = arith.constant 0 : i32
                %c32_i32_71 = arith.constant 32 : i32
                %c2048 = arith.constant 2048 : index
                %c64_i32_72 = arith.constant 64 : i32
                %mask, %remaining_out = ascvec.update_mask %c64_i32_72 : !ascvec.mask
                %297 = ascvec.loadalign %arg31[%c2048] : memref<4096xf32, 1>, index -> <f32>
                scf.for %arg34 = %c0_i32_70 to %c16_i32_69 step %c1_i32_68  : i32 {
                  %298 = arith.muli %arg34, %c64_i32_72 overflow<nsw> : i32
                  %299 = arith.addi %arg34, %c32_i32_71 overflow<nsw> : i32
                  %300 = arith.muli %299, %c64_i32_72 overflow<nsw> : i32
                  %301 = arith.index_cast %298 : i32 to index
                  %302 = arith.addi %301, %c2048 : index
                  %303 = ascvec.loadalign_unpack %arg32[%302] {mode = "b16_to_b32"} : memref<4096xbf16, 1>, index -> <bf16>
                  %304 = ascvec.cast %303, %mask {rounding = 6 : i32} : <bf16> -> <f32>
                  %305 = arith.index_cast %300 : i32 to index
                  %306 = ascvec.loadalign %arg31[%305] : memref<4096xf32, 1>, index -> <f32>
                  %307 = ascvec.sub %306, %297, %mask : <f32>
                  %308 = ascvec.exp %307, %mask : <f32>
                  %309 = ascvec.mul %304, %308, %mask : <f32>
                  %310 = ascvec.cast %309, %mask : <f32> -> <bf16>
                  ascvec.storealign_pack %arg33[%301], %310, %mask {mode = "b32_to_b16"} : memref<1024xbf16, 1>, index, <bf16>
                } {cannir.loop_unroll = 2 : i64}
              } {cannir.vf_mode = "simd"}
              ascvec.get_buf v, %c16_i64, 0 {cannir.buffer.generated_sync}
              ascvec.vf(%alloc_46, %alloc_48 : memref<1024xbf16, 1>, memref<1088xbf16, 1>) {
              ^bb0(%arg31: memref<1024xbf16, 1>, %arg32: memref<1088xbf16, 1>):
                %c1_i32_68 = arith.constant 1 : i32
                %c0_i32_69 = arith.constant 0 : i32
                %c17 = arith.constant 17 : index
                %c16_i32_70 = arith.constant 16 : i32
                %c0_71 = arith.constant 0 : index
                %c64_i32_72 = arith.constant 64 : i32
                scf.for %arg33 = %c0_i32_69 to %c16_i32_70 step %c1_i32_68  : i32 {
                  %mask, %remaining_out = ascvec.update_mask %c64_i32_72 elem_bits 16 : !ascvec.mask
                  %297 = arith.muli %arg33, %c64_i32_72 overflow<nsw> : i32
                  %298 = arith.index_cast %297 : i32 to index
                  %299 = ascvec.loadalign %arg31[%298] : memref<1024xbf16, 1>, index -> <bf16>
                  %300 = arith.muli %arg33, %c16_i32_70 overflow<nsw> : i32
                  %301 = arith.index_cast %300 : i32 to index
                  ascvec.storealign_strided %arg32[%301], %299, %mask, %c17, %c0_71 : memref<1088xbf16, 1>, <bf16>
                }
              } {cannir.vf_mode = "simd"}
              ascvec.release_buf v, %c16_i64, 0 {cannir.buffer.generated_sync}
              ascvec.sync_intra_wait mte3, %c6_i64 {cannir.sync.banks = 3 : i32}
              ascvec.get_buf mte3, %c16_i64, 0 {cannir.buffer.generated_sync}
              ascvec.copy_ub2l1_strided %275[%273], %274[%c0], %c4_i32, %c16_i32, %c1_i32, %c0_i32 : memref<?xbf16, 2>, memref<?xbf16, 1>
              ascvec.release_buf mte3, %c16_i64, 0 {cannir.buffer.generated_sync}
              ascvec.sync_intra_arrive mte3, %c6_i64 {cannir.sync.banks = 3 : i32}
              ascvec.vf(%alloc_40, %alloc_26, %alloc_45 : memref<4096xf32, 1>, memref<4096xbf16, 1>, memref<1024xbf16, 1>) {
              ^bb0(%arg31: memref<4096xf32, 1>, %arg32: memref<4096xbf16, 1>, %arg33: memref<1024xbf16, 1>):
                %c2048 = arith.constant 2048 : index
                %c1_i32_68 = arith.constant 1 : i32
                %c16_i32_69 = arith.constant 16 : i32
                %c0_i32_70 = arith.constant 0 : i32
                %c64_i32_71 = arith.constant 64 : i32
                %mask, %remaining_out = ascvec.update_mask %c64_i32_71 : !ascvec.mask
                %297 = ascvec.loadalign %arg31[%c2048] : memref<4096xf32, 1>, index -> <f32>
                scf.for %arg34 = %c0_i32_70 to %c16_i32_69 step %c1_i32_68  : i32 {
                  %298 = arith.muli %arg34, %c64_i32_71 overflow<nsw> : i32
                  %299 = arith.index_cast %298 : i32 to index
                  %300 = ascvec.loadalign_unpack %arg32[%299] {mode = "b16_to_b32"} : memref<4096xbf16, 1>, index -> <bf16>
                  %301 = ascvec.cast %300, %mask {rounding = 6 : i32} : <bf16> -> <f32>
                  %302 = ascvec.loadalign %arg31[%299] : memref<4096xf32, 1>, index -> <f32>
                  %303 = ascvec.sub %297, %302, %mask : <f32>
                  %304 = ascvec.exp %303, %mask : <f32>
                  %305 = ascvec.mul %301, %304, %mask : <f32>
                  %306 = ascvec.cast %305, %mask : <f32> -> <bf16>
                  ascvec.storealign_pack %arg33[%299], %306, %mask {mode = "b32_to_b16"} : memref<1024xbf16, 1>, index, <bf16>
                } {cannir.loop_unroll = 2 : i64}
              } {cannir.vf_mode = "simd"}
              ascvec.get_buf v, %c15_i64, 0 {cannir.buffer.generated_sync}
              ascvec.vf(%alloc_45, %alloc_47 : memref<1024xbf16, 1>, memref<1088xbf16, 1>) {
              ^bb0(%arg31: memref<1024xbf16, 1>, %arg32: memref<1088xbf16, 1>):
                %c1_i32_68 = arith.constant 1 : i32
                %c0_i32_69 = arith.constant 0 : i32
                %c17 = arith.constant 17 : index
                %c16_i32_70 = arith.constant 16 : i32
                %c0_71 = arith.constant 0 : index
                %c64_i32_72 = arith.constant 64 : i32
                scf.for %arg33 = %c0_i32_69 to %c16_i32_70 step %c1_i32_68  : i32 {
                  %mask, %remaining_out = ascvec.update_mask %c64_i32_72 elem_bits 16 : !ascvec.mask
                  %297 = arith.muli %arg33, %c64_i32_72 overflow<nsw> : i32
                  %298 = arith.index_cast %297 : i32 to index
                  %299 = ascvec.loadalign %arg31[%298] : memref<1024xbf16, 1>, index -> <bf16>
                  %300 = arith.muli %arg33, %c16_i32_70 overflow<nsw> : i32
                  %301 = arith.index_cast %300 : i32 to index
                  ascvec.storealign_strided %arg32[%301], %299, %mask, %c17, %c0_71 : memref<1088xbf16, 1>, <bf16>
                }
              } {cannir.vf_mode = "simd"}
              ascvec.release_buf v, %c15_i64, 0 {cannir.buffer.generated_sync}
              ascvec.sync_intra_wait mte3, %c7_i64 {cannir.sync.banks = 3 : i32}
              ascvec.get_buf mte3, %c15_i64, 0 {cannir.buffer.generated_sync}
              ascvec.copy_ub2l1_strided %276[%273], %269[%c0], %c4_i32, %c16_i32, %c1_i32, %c0_i32 : memref<?xbf16, 2>, memref<?xbf16, 1>
              ascvec.release_buf mte3, %c15_i64, 0 {cannir.buffer.generated_sync}
              ascvec.sync_intra_arrive mte3, %c7_i64 {cannir.sync.banks = 3 : i32}
              ascvec.vf(%alloc_40, %alloc_25, %arg20, %alloc_46 : memref<4096xf32, 1>, memref<4096xbf16, 1>, f32, memref<1024xbf16, 1>) {
              ^bb0(%arg31: memref<4096xf32, 1>, %arg32: memref<4096xbf16, 1>, %arg33: f32, %arg34: memref<1024xbf16, 1>):
                %c1_i32_68 = arith.constant 1 : i32
                %c16_i32_69 = arith.constant 16 : i32
                %c0_i32_70 = arith.constant 0 : i32
                %c32_i32_71 = arith.constant 32 : i32
                %c2048 = arith.constant 2048 : index
                %c64_i32_72 = arith.constant 64 : i32
                %mask, %remaining_out = ascvec.update_mask %c64_i32_72 : !ascvec.mask
                %297 = ascvec.loadalign %arg31[%c2048] : memref<4096xf32, 1>, index -> <f32>
                scf.for %arg35 = %c0_i32_70 to %c16_i32_69 step %c1_i32_68  : i32 {
                  %298 = arith.muli %arg35, %c64_i32_72 overflow<nsw> : i32
                  %299 = arith.addi %arg35, %c32_i32_71 overflow<nsw> : i32
                  %300 = arith.muli %299, %c64_i32_72 overflow<nsw> : i32
                  %301 = arith.index_cast %298 : i32 to index
                  %302 = arith.addi %301, %c2048 : index
                  %303 = ascvec.loadalign_unpack %arg32[%302] {mode = "b16_to_b32"} : memref<4096xbf16, 1>, index -> <bf16>
                  %304 = ascvec.cast %303, %mask {rounding = 6 : i32} : <bf16> -> <f32>
                  %305 = arith.index_cast %300 : i32 to index
                  %306 = ascvec.loadalign %arg31[%305] : memref<4096xf32, 1>, index -> <f32>
                  %307 = ascvec.sub %306, %297, %mask : <f32>
                  %308 = ascvec.exp %307, %mask : <f32>
                  %309 = ascvec.mul %304, %308, %mask : <f32>
                  %310 = ascvec.mul_scalar %309, %arg33, %mask : <f32>, f32
                  %311 = ascvec.cast %310, %mask : <f32> -> <bf16>
                  ascvec.storealign_pack %arg34[%301], %311, %mask {mode = "b32_to_b16"} : memref<1024xbf16, 1>, index, <bf16>
                } {cannir.loop_unroll = 2 : i64}
              } {cannir.vf_mode = "simd"}
              ascvec.get_buf v, %c16_i64, 0 {cannir.buffer.generated_sync}
              ascvec.vf(%alloc_46, %alloc_48 : memref<1024xbf16, 1>, memref<1088xbf16, 1>) {
              ^bb0(%arg31: memref<1024xbf16, 1>, %arg32: memref<1088xbf16, 1>):
                %c1_i32_68 = arith.constant 1 : i32
                %c0_i32_69 = arith.constant 0 : i32
                %c17 = arith.constant 17 : index
                %c16_i32_70 = arith.constant 16 : i32
                %c0_71 = arith.constant 0 : index
                %c64_i32_72 = arith.constant 64 : i32
                scf.for %arg33 = %c0_i32_69 to %c16_i32_70 step %c1_i32_68  : i32 {
                  %mask, %remaining_out = ascvec.update_mask %c64_i32_72 elem_bits 16 : !ascvec.mask
                  %297 = arith.muli %arg33, %c64_i32_72 overflow<nsw> : i32
                  %298 = arith.index_cast %297 : i32 to index
                  %299 = ascvec.loadalign %arg31[%298] : memref<1024xbf16, 1>, index -> <bf16>
                  %300 = arith.muli %arg33, %c16_i32_70 overflow<nsw> : i32
                  %301 = arith.index_cast %300 : i32 to index
                  ascvec.storealign_strided %arg32[%301], %299, %mask, %c17, %c0_71 : memref<1088xbf16, 1>, <bf16>
                }
              } {cannir.vf_mode = "simd"}
              ascvec.release_buf v, %c16_i64, 0 {cannir.buffer.generated_sync}
              ascvec.sync_intra_wait mte3, %c5_i64 {cannir.sync.banks = 3 : i32}
              ascvec.get_buf mte3, %c16_i64, 0 {cannir.buffer.generated_sync}
              ascvec.copy_ub2l1_strided %270[%273], %274[%c0], %c4_i32, %c16_i32, %c1_i32, %c0_i32 : memref<?xbf16, 2>, memref<?xbf16, 1>
              ascvec.release_buf mte3, %c16_i64, 0 {cannir.buffer.generated_sync}
              ascvec.sync_intra_arrive mte3, %c5_i64 {cannir.sync.banks = 3 : i32}
              ascvec.vf(%alloc_40, %alloc_26, %alloc_45 : memref<4096xf32, 1>, memref<4096xbf16, 1>, memref<1024xbf16, 1>) {
              ^bb0(%arg31: memref<4096xf32, 1>, %arg32: memref<4096xbf16, 1>, %arg33: memref<1024xbf16, 1>):
                %c2048 = arith.constant 2048 : index
                %c1_i32_68 = arith.constant 1 : i32
                %c0_i32_69 = arith.constant 0 : i32
                %c1024_70 = arith.constant 1024 : index
                %c16_i32_71 = arith.constant 16 : i32
                %c64_i32_72 = arith.constant 64 : i32
                %mask, %remaining_out = ascvec.update_mask %c64_i32_72 : !ascvec.mask
                %297 = ascvec.loadalign %arg31[%c2048] : memref<4096xf32, 1>, index -> <f32>
                scf.for %arg34 = %c0_i32_69 to %c16_i32_71 step %c1_i32_68  : i32 {
                  %298 = arith.muli %arg34, %c64_i32_72 overflow<nsw> : i32
                  %299 = arith.addi %arg34, %c16_i32_71 overflow<nsw> : i32
                  %300 = arith.muli %299, %c64_i32_72 overflow<nsw> : i32
                  %301 = arith.index_cast %298 : i32 to index
                  %302 = arith.addi %301, %c1024_70 : index
                  %303 = ascvec.loadalign_unpack %arg32[%302] {mode = "b16_to_b32"} : memref<4096xbf16, 1>, index -> <bf16>
                  %304 = ascvec.cast %303, %mask {rounding = 6 : i32} : <bf16> -> <f32>
                  %305 = arith.index_cast %300 : i32 to index
                  %306 = ascvec.loadalign %arg31[%305] : memref<4096xf32, 1>, index -> <f32>
                  %307 = ascvec.sub %297, %306, %mask : <f32>
                  %308 = ascvec.exp %307, %mask : <f32>
                  %309 = ascvec.mul %304, %308, %mask : <f32>
                  %310 = ascvec.cast %309, %mask : <f32> -> <bf16>
                  ascvec.storealign_pack %arg33[%301], %310, %mask {mode = "b32_to_b16"} : memref<1024xbf16, 1>, index, <bf16>
                } {cannir.loop_unroll = 2 : i64}
              } {cannir.vf_mode = "simd"}
              ascvec.get_buf v, %c15_i64, 0 {cannir.buffer.generated_sync}
              ascvec.vf(%alloc_45, %alloc_47 : memref<1024xbf16, 1>, memref<1088xbf16, 1>) {
              ^bb0(%arg31: memref<1024xbf16, 1>, %arg32: memref<1088xbf16, 1>):
                %c1_i32_68 = arith.constant 1 : i32
                %c0_i32_69 = arith.constant 0 : i32
                %c17 = arith.constant 17 : index
                %c16_i32_70 = arith.constant 16 : i32
                %c0_71 = arith.constant 0 : index
                %c64_i32_72 = arith.constant 64 : i32
                scf.for %arg33 = %c0_i32_69 to %c16_i32_70 step %c1_i32_68  : i32 {
                  %mask, %remaining_out = ascvec.update_mask %c64_i32_72 elem_bits 16 : !ascvec.mask
                  %297 = arith.muli %arg33, %c64_i32_72 overflow<nsw> : i32
                  %298 = arith.index_cast %297 : i32 to index
                  %299 = ascvec.loadalign %arg31[%298] : memref<1024xbf16, 1>, index -> <bf16>
                  %300 = arith.muli %arg33, %c16_i32_70 overflow<nsw> : i32
                  %301 = arith.index_cast %300 : i32 to index
                  ascvec.storealign_strided %arg32[%301], %299, %mask, %c17, %c0_71 : memref<1088xbf16, 1>, <bf16>
                }
              } {cannir.vf_mode = "simd"}
              ascvec.release_buf v, %c15_i64, 0 {cannir.buffer.generated_sync}
              ascvec.sync_intra_wait mte3, %c6_i64 {cannir.sync.banks = 3 : i32}
              ascvec.get_buf mte3, %c15_i64, 0 {cannir.buffer.generated_sync}
              ascvec.copy_ub2l1_strided %275[%273], %269[%c0], %c4_i32, %c16_i32, %c1_i32, %c0_i32 : memref<?xbf16, 2>, memref<?xbf16, 1>
              ascvec.release_buf mte3, %c15_i64, 0 {cannir.buffer.generated_sync}
              ascvec.sync_intra_arrive mte3, %c6_i64 {cannir.sync.banks = 3 : i32}
              ascvec.vf(%alloc_40, %alloc_26, %alloc_46 : memref<4096xf32, 1>, memref<4096xbf16, 1>, memref<1024xbf16, 1>) {
              ^bb0(%arg31: memref<4096xf32, 1>, %arg32: memref<4096xbf16, 1>, %arg33: memref<1024xbf16, 1>):
                %c1_i32_68 = arith.constant 1 : i32
                %c16_i32_69 = arith.constant 16 : i32
                %c0_i32_70 = arith.constant 0 : i32
                %c32_i32_71 = arith.constant 32 : i32
                %c2048 = arith.constant 2048 : index
                %c64_i32_72 = arith.constant 64 : i32
                %mask, %remaining_out = ascvec.update_mask %c64_i32_72 : !ascvec.mask
                %297 = ascvec.loadalign %arg31[%c2048] : memref<4096xf32, 1>, index -> <f32>
                scf.for %arg34 = %c0_i32_70 to %c16_i32_69 step %c1_i32_68  : i32 {
                  %298 = arith.muli %arg34, %c64_i32_72 overflow<nsw> : i32
                  %299 = arith.addi %arg34, %c32_i32_71 overflow<nsw> : i32
                  %300 = arith.muli %299, %c64_i32_72 overflow<nsw> : i32
                  %301 = arith.index_cast %298 : i32 to index
                  %302 = arith.addi %301, %c2048 : index
                  %303 = ascvec.loadalign_unpack %arg32[%302] {mode = "b16_to_b32"} : memref<4096xbf16, 1>, index -> <bf16>
                  %304 = ascvec.cast %303, %mask {rounding = 6 : i32} : <bf16> -> <f32>
                  %305 = arith.index_cast %300 : i32 to index
                  %306 = ascvec.loadalign %arg31[%305] : memref<4096xf32, 1>, index -> <f32>
                  %307 = ascvec.sub %297, %306, %mask : <f32>
                  %308 = ascvec.exp %307, %mask : <f32>
                  %309 = ascvec.mul %304, %308, %mask : <f32>
                  %310 = ascvec.cast %309, %mask : <f32> -> <bf16>
                  ascvec.storealign_pack %arg33[%301], %310, %mask {mode = "b32_to_b16"} : memref<1024xbf16, 1>, index, <bf16>
                } {cannir.loop_unroll = 2 : i64}
              } {cannir.vf_mode = "simd"}
              ascvec.get_buf v, %c16_i64, 0 {cannir.buffer.generated_sync}
              ascvec.vf(%alloc_46, %alloc_48 : memref<1024xbf16, 1>, memref<1088xbf16, 1>) {
              ^bb0(%arg31: memref<1024xbf16, 1>, %arg32: memref<1088xbf16, 1>):
                %c1_i32_68 = arith.constant 1 : i32
                %c0_i32_69 = arith.constant 0 : i32
                %c17 = arith.constant 17 : index
                %c16_i32_70 = arith.constant 16 : i32
                %c0_71 = arith.constant 0 : index
                %c64_i32_72 = arith.constant 64 : i32
                scf.for %arg33 = %c0_i32_69 to %c16_i32_70 step %c1_i32_68  : i32 {
                  %mask, %remaining_out = ascvec.update_mask %c64_i32_72 elem_bits 16 : !ascvec.mask
                  %297 = arith.muli %arg33, %c64_i32_72 overflow<nsw> : i32
                  %298 = arith.index_cast %297 : i32 to index
                  %299 = ascvec.loadalign %arg31[%298] : memref<1024xbf16, 1>, index -> <bf16>
                  %300 = arith.muli %arg33, %c16_i32_70 overflow<nsw> : i32
                  %301 = arith.index_cast %300 : i32 to index
                  ascvec.storealign_strided %arg32[%301], %299, %mask, %c17, %c0_71 : memref<1088xbf16, 1>, <bf16>
                }
              } {cannir.vf_mode = "simd"}
              ascvec.release_buf v, %c16_i64, 0 {cannir.buffer.generated_sync}
              ascvec.sync_intra_wait mte3, %c7_i64 {cannir.sync.banks = 3 : i32}
              ascvec.get_buf mte3, %c16_i64, 0 {cannir.buffer.generated_sync}
              ascvec.copy_ub2l1_strided %276[%273], %274[%c0], %c4_i32, %c16_i32, %c1_i32, %c0_i32 : memref<?xbf16, 2>, memref<?xbf16, 1>
              ascvec.release_buf mte3, %c16_i64, 0 {cannir.buffer.generated_sync}
              ascvec.sync_intra_arrive mte3, %c7_i64 {cannir.sync.banks = 3 : i32}
              ascvec.vf(%alloc_40, %alloc_26, %alloc_45 : memref<4096xf32, 1>, memref<4096xbf16, 1>, memref<1024xbf16, 1>) {
              ^bb0(%arg31: memref<4096xf32, 1>, %arg32: memref<4096xbf16, 1>, %arg33: memref<1024xbf16, 1>):
                %c1_i32_68 = arith.constant 1 : i32
                %c16_i32_69 = arith.constant 16 : i32
                %c0_i32_70 = arith.constant 0 : i32
                %c48_i32 = arith.constant 48 : i32
                %c3072 = arith.constant 3072 : index
                %c64_i32_71 = arith.constant 64 : i32
                %mask, %remaining_out = ascvec.update_mask %c64_i32_71 : !ascvec.mask
                %297 = ascvec.loadalign %arg31[%c3072] : memref<4096xf32, 1>, index -> <f32>
                scf.for %arg34 = %c0_i32_70 to %c16_i32_69 step %c1_i32_68  : i32 {
                  %298 = arith.muli %arg34, %c64_i32_71 overflow<nsw> : i32
                  %299 = arith.addi %arg34, %c48_i32 overflow<nsw> : i32
                  %300 = arith.muli %299, %c64_i32_71 overflow<nsw> : i32
                  %301 = arith.index_cast %298 : i32 to index
                  %302 = arith.addi %301, %c3072 : index
                  %303 = ascvec.loadalign_unpack %arg32[%302] {mode = "b16_to_b32"} : memref<4096xbf16, 1>, index -> <bf16>
                  %304 = ascvec.cast %303, %mask {rounding = 6 : i32} : <bf16> -> <f32>
                  %305 = arith.index_cast %300 : i32 to index
                  %306 = ascvec.loadalign %arg31[%305] : memref<4096xf32, 1>, index -> <f32>
                  %307 = ascvec.sub %306, %297, %mask : <f32>
                  %308 = ascvec.exp %307, %mask : <f32>
                  %309 = ascvec.mul %304, %308, %mask : <f32>
                  %310 = ascvec.cast %309, %mask : <f32> -> <bf16>
                  ascvec.storealign_pack %arg33[%301], %310, %mask {mode = "b32_to_b16"} : memref<1024xbf16, 1>, index, <bf16>
                } {cannir.loop_unroll = 2 : i64}
              } {cannir.vf_mode = "simd"}
              ascvec.get_buf v, %c15_i64, 0 {cannir.buffer.generated_sync}
              ascvec.vf(%alloc_45, %alloc_47 : memref<1024xbf16, 1>, memref<1088xbf16, 1>) {
              ^bb0(%arg31: memref<1024xbf16, 1>, %arg32: memref<1088xbf16, 1>):
                %c1_i32_68 = arith.constant 1 : i32
                %c0_i32_69 = arith.constant 0 : i32
                %c17 = arith.constant 17 : index
                %c16_i32_70 = arith.constant 16 : i32
                %c0_71 = arith.constant 0 : index
                %c64_i32_72 = arith.constant 64 : i32
                scf.for %arg33 = %c0_i32_69 to %c16_i32_70 step %c1_i32_68  : i32 {
                  %mask, %remaining_out = ascvec.update_mask %c64_i32_72 elem_bits 16 : !ascvec.mask
                  %297 = arith.muli %arg33, %c64_i32_72 overflow<nsw> : i32
                  %298 = arith.index_cast %297 : i32 to index
                  %299 = ascvec.loadalign %arg31[%298] : memref<1024xbf16, 1>, index -> <bf16>
                  %300 = arith.muli %arg33, %c16_i32_70 overflow<nsw> : i32
                  %301 = arith.index_cast %300 : i32 to index
                  ascvec.storealign_strided %arg32[%301], %299, %mask, %c17, %c0_71 : memref<1088xbf16, 1>, <bf16>
                }
              } {cannir.vf_mode = "simd"}
              ascvec.release_buf v, %c15_i64, 0 {cannir.buffer.generated_sync}
              ascvec.sync_intra_wait mte3, %c5_i64 {cannir.sync.banks = 3 : i32}
              ascvec.get_buf mte3, %c15_i64, 0 {cannir.buffer.generated_sync}
              ascvec.copy_ub2l1_strided %270[%273], %269[%c0], %c4_i32, %c16_i32, %c1_i32, %c0_i32 : memref<?xbf16, 2>, memref<?xbf16, 1>
              ascvec.release_buf mte3, %c15_i64, 0 {cannir.buffer.generated_sync}
              ascvec.sync_intra_arrive mte3, %c5_i64 {cannir.sync.banks = 3 : i32}
              ascvec.vf(%alloc_40, %alloc_26, %alloc_46 : memref<4096xf32, 1>, memref<4096xbf16, 1>, memref<1024xbf16, 1>) {
              ^bb0(%arg31: memref<4096xf32, 1>, %arg32: memref<4096xbf16, 1>, %arg33: memref<1024xbf16, 1>):
                %c3072 = arith.constant 3072 : index
                %c1_i32_68 = arith.constant 1 : i32
                %c16_i32_69 = arith.constant 16 : i32
                %c0_i32_70 = arith.constant 0 : i32
                %c64_i32_71 = arith.constant 64 : i32
                %mask, %remaining_out = ascvec.update_mask %c64_i32_71 : !ascvec.mask
                %297 = ascvec.loadalign %arg31[%c3072] : memref<4096xf32, 1>, index -> <f32>
                scf.for %arg34 = %c0_i32_70 to %c16_i32_69 step %c1_i32_68  : i32 {
                  %298 = arith.muli %arg34, %c64_i32_71 overflow<nsw> : i32
                  %299 = arith.index_cast %298 : i32 to index
                  %300 = ascvec.loadalign_unpack %arg32[%299] {mode = "b16_to_b32"} : memref<4096xbf16, 1>, index -> <bf16>
                  %301 = ascvec.cast %300, %mask {rounding = 6 : i32} : <bf16> -> <f32>
                  %302 = ascvec.loadalign %arg31[%299] : memref<4096xf32, 1>, index -> <f32>
                  %303 = ascvec.sub %297, %302, %mask : <f32>
                  %304 = ascvec.exp %303, %mask : <f32>
                  %305 = ascvec.mul %301, %304, %mask : <f32>
                  %306 = ascvec.cast %305, %mask : <f32> -> <bf16>
                  ascvec.storealign_pack %arg33[%299], %306, %mask {mode = "b32_to_b16"} : memref<1024xbf16, 1>, index, <bf16>
                } {cannir.loop_unroll = 2 : i64}
              } {cannir.vf_mode = "simd"}
              ascvec.get_buf v, %c16_i64, 0 {cannir.buffer.generated_sync}
              ascvec.vf(%alloc_46, %alloc_48 : memref<1024xbf16, 1>, memref<1088xbf16, 1>) {
              ^bb0(%arg31: memref<1024xbf16, 1>, %arg32: memref<1088xbf16, 1>):
                %c1_i32_68 = arith.constant 1 : i32
                %c0_i32_69 = arith.constant 0 : i32
                %c17 = arith.constant 17 : index
                %c16_i32_70 = arith.constant 16 : i32
                %c0_71 = arith.constant 0 : index
                %c64_i32_72 = arith.constant 64 : i32
                scf.for %arg33 = %c0_i32_69 to %c16_i32_70 step %c1_i32_68  : i32 {
                  %mask, %remaining_out = ascvec.update_mask %c64_i32_72 elem_bits 16 : !ascvec.mask
                  %297 = arith.muli %arg33, %c64_i32_72 overflow<nsw> : i32
                  %298 = arith.index_cast %297 : i32 to index
                  %299 = ascvec.loadalign %arg31[%298] : memref<1024xbf16, 1>, index -> <bf16>
                  %300 = arith.muli %arg33, %c16_i32_70 overflow<nsw> : i32
                  %301 = arith.index_cast %300 : i32 to index
                  ascvec.storealign_strided %arg32[%301], %299, %mask, %c17, %c0_71 : memref<1088xbf16, 1>, <bf16>
                }
              } {cannir.vf_mode = "simd"}
              ascvec.release_buf v, %c16_i64, 0 {cannir.buffer.generated_sync}
              ascvec.sync_intra_wait mte3, %c6_i64 {cannir.sync.banks = 3 : i32}
              ascvec.get_buf mte3, %c16_i64, 0 {cannir.buffer.generated_sync}
              ascvec.copy_ub2l1_strided %275[%273], %274[%c0], %c4_i32, %c16_i32, %c1_i32, %c0_i32 : memref<?xbf16, 2>, memref<?xbf16, 1>
              ascvec.release_buf mte3, %c16_i64, 0 {cannir.buffer.generated_sync}
              ascvec.sync_intra_arrive mte3, %c6_i64 {cannir.sync.banks = 3 : i32}
              ascvec.vf(%alloc_40, %alloc_25, %arg20, %alloc_45 : memref<4096xf32, 1>, memref<4096xbf16, 1>, f32, memref<1024xbf16, 1>) {
              ^bb0(%arg31: memref<4096xf32, 1>, %arg32: memref<4096xbf16, 1>, %arg33: f32, %arg34: memref<1024xbf16, 1>):
                %c1_i32_68 = arith.constant 1 : i32
                %c16_i32_69 = arith.constant 16 : i32
                %c0_i32_70 = arith.constant 0 : i32
                %c48_i32 = arith.constant 48 : i32
                %c3072 = arith.constant 3072 : index
                %c64_i32_71 = arith.constant 64 : i32
                %mask, %remaining_out = ascvec.update_mask %c64_i32_71 : !ascvec.mask
                %297 = ascvec.loadalign %arg31[%c3072] : memref<4096xf32, 1>, index -> <f32>
                scf.for %arg35 = %c0_i32_70 to %c16_i32_69 step %c1_i32_68  : i32 {
                  %298 = arith.muli %arg35, %c64_i32_71 overflow<nsw> : i32
                  %299 = arith.addi %arg35, %c48_i32 overflow<nsw> : i32
                  %300 = arith.muli %299, %c64_i32_71 overflow<nsw> : i32
                  %301 = arith.index_cast %298 : i32 to index
                  %302 = arith.addi %301, %c3072 : index
                  %303 = ascvec.loadalign_unpack %arg32[%302] {mode = "b16_to_b32"} : memref<4096xbf16, 1>, index -> <bf16>
                  %304 = ascvec.cast %303, %mask {rounding = 6 : i32} : <bf16> -> <f32>
                  %305 = arith.index_cast %300 : i32 to index
                  %306 = ascvec.loadalign %arg31[%305] : memref<4096xf32, 1>, index -> <f32>
                  %307 = ascvec.sub %306, %297, %mask : <f32>
                  %308 = ascvec.exp %307, %mask : <f32>
                  %309 = ascvec.mul %304, %308, %mask : <f32>
                  %310 = ascvec.mul_scalar %309, %arg33, %mask : <f32>, f32
                  %311 = ascvec.cast %310, %mask : <f32> -> <bf16>
                  ascvec.storealign_pack %arg34[%301], %311, %mask {mode = "b32_to_b16"} : memref<1024xbf16, 1>, index, <bf16>
                } {cannir.loop_unroll = 2 : i64}
              } {cannir.vf_mode = "simd"}
              ascvec.get_buf v, %c15_i64, 0 {cannir.buffer.generated_sync}
              ascvec.vf(%alloc_45, %alloc_47 : memref<1024xbf16, 1>, memref<1088xbf16, 1>) {
              ^bb0(%arg31: memref<1024xbf16, 1>, %arg32: memref<1088xbf16, 1>):
                %c1_i32_68 = arith.constant 1 : i32
                %c0_i32_69 = arith.constant 0 : i32
                %c17 = arith.constant 17 : index
                %c16_i32_70 = arith.constant 16 : i32
                %c0_71 = arith.constant 0 : index
                %c64_i32_72 = arith.constant 64 : i32
                scf.for %arg33 = %c0_i32_69 to %c16_i32_70 step %c1_i32_68  : i32 {
                  %mask, %remaining_out = ascvec.update_mask %c64_i32_72 elem_bits 16 : !ascvec.mask
                  %297 = arith.muli %arg33, %c64_i32_72 overflow<nsw> : i32
                  %298 = arith.index_cast %297 : i32 to index
                  %299 = ascvec.loadalign %arg31[%298] : memref<1024xbf16, 1>, index -> <bf16>
                  %300 = arith.muli %arg33, %c16_i32_70 overflow<nsw> : i32
                  %301 = arith.index_cast %300 : i32 to index
                  ascvec.storealign_strided %arg32[%301], %299, %mask, %c17, %c0_71 : memref<1088xbf16, 1>, <bf16>
                }
              } {cannir.vf_mode = "simd"}
              ascvec.release_buf v, %c15_i64, 0 {cannir.buffer.generated_sync}
              ascvec.sync_intra_wait mte3, %c7_i64 {cannir.sync.banks = 3 : i32}
              ascvec.get_buf mte3, %c15_i64, 0 {cannir.buffer.generated_sync}
              ascvec.copy_ub2l1_strided %276[%273], %269[%c0], %c4_i32, %c16_i32, %c1_i32, %c0_i32 : memref<?xbf16, 2>, memref<?xbf16, 1>
              ascvec.release_buf mte3, %c15_i64, 0 {cannir.buffer.generated_sync}
              ascvec.sync_intra_arrive mte3, %c7_i64 {cannir.sync.banks = 3 : i32}
              ascvec.vf(%alloc_40, %alloc_26, %alloc_46 : memref<4096xf32, 1>, memref<4096xbf16, 1>, memref<1024xbf16, 1>) {
              ^bb0(%arg31: memref<4096xf32, 1>, %arg32: memref<4096xbf16, 1>, %arg33: memref<1024xbf16, 1>):
                %c3072 = arith.constant 3072 : index
                %c1_i32_68 = arith.constant 1 : i32
                %c0_i32_69 = arith.constant 0 : i32
                %c1024_70 = arith.constant 1024 : index
                %c16_i32_71 = arith.constant 16 : i32
                %c64_i32_72 = arith.constant 64 : i32
                %mask, %remaining_out = ascvec.update_mask %c64_i32_72 : !ascvec.mask
                %297 = ascvec.loadalign %arg31[%c3072] : memref<4096xf32, 1>, index -> <f32>
                scf.for %arg34 = %c0_i32_69 to %c16_i32_71 step %c1_i32_68  : i32 {
                  %298 = arith.muli %arg34, %c64_i32_72 overflow<nsw> : i32
                  %299 = arith.addi %arg34, %c16_i32_71 overflow<nsw> : i32
                  %300 = arith.muli %299, %c64_i32_72 overflow<nsw> : i32
                  %301 = arith.index_cast %298 : i32 to index
                  %302 = arith.addi %301, %c1024_70 : index
                  %303 = ascvec.loadalign_unpack %arg32[%302] {mode = "b16_to_b32"} : memref<4096xbf16, 1>, index -> <bf16>
                  %304 = ascvec.cast %303, %mask {rounding = 6 : i32} : <bf16> -> <f32>
                  %305 = arith.index_cast %300 : i32 to index
                  %306 = ascvec.loadalign %arg31[%305] : memref<4096xf32, 1>, index -> <f32>
                  %307 = ascvec.sub %297, %306, %mask : <f32>
                  %308 = ascvec.exp %307, %mask : <f32>
                  %309 = ascvec.mul %304, %308, %mask : <f32>
                  %310 = ascvec.cast %309, %mask : <f32> -> <bf16>
                  ascvec.storealign_pack %arg33[%301], %310, %mask {mode = "b32_to_b16"} : memref<1024xbf16, 1>, index, <bf16>
                } {cannir.loop_unroll = 2 : i64}
              } {cannir.vf_mode = "simd"}
              ascvec.get_buf v, %c16_i64, 0 {cannir.buffer.generated_sync}
              ascvec.vf(%alloc_46, %alloc_48 : memref<1024xbf16, 1>, memref<1088xbf16, 1>) {
              ^bb0(%arg31: memref<1024xbf16, 1>, %arg32: memref<1088xbf16, 1>):
                %c1_i32_68 = arith.constant 1 : i32
                %c0_i32_69 = arith.constant 0 : i32
                %c17 = arith.constant 17 : index
                %c16_i32_70 = arith.constant 16 : i32
                %c0_71 = arith.constant 0 : index
                %c64_i32_72 = arith.constant 64 : i32
                scf.for %arg33 = %c0_i32_69 to %c16_i32_70 step %c1_i32_68  : i32 {
                  %mask, %remaining_out = ascvec.update_mask %c64_i32_72 elem_bits 16 : !ascvec.mask
                  %297 = arith.muli %arg33, %c64_i32_72 overflow<nsw> : i32
                  %298 = arith.index_cast %297 : i32 to index
                  %299 = ascvec.loadalign %arg31[%298] : memref<1024xbf16, 1>, index -> <bf16>
                  %300 = arith.muli %arg33, %c16_i32_70 overflow<nsw> : i32
                  %301 = arith.index_cast %300 : i32 to index
                  ascvec.storealign_strided %arg32[%301], %299, %mask, %c17, %c0_71 : memref<1088xbf16, 1>, <bf16>
                }
              } {cannir.vf_mode = "simd"}
              ascvec.release_buf v, %c16_i64, 0 {cannir.buffer.generated_sync}
              ascvec.sync_intra_wait mte3, %c5_i64 {cannir.sync.banks = 3 : i32}
              ascvec.get_buf mte3, %c16_i64, 0 {cannir.buffer.generated_sync}
              ascvec.copy_ub2l1_strided %270[%273], %274[%c0], %c4_i32, %c16_i32, %c1_i32, %c0_i32 : memref<?xbf16, 2>, memref<?xbf16, 1>
              ascvec.release_buf mte3, %c16_i64, 0 {cannir.buffer.generated_sync}
              ascvec.sync_intra_arrive mte3, %c5_i64 {cannir.sync.banks = 3 : i32}
              ascvec.vf(%alloc_40, %alloc_26, %alloc_45 : memref<4096xf32, 1>, memref<4096xbf16, 1>, memref<1024xbf16, 1>) {
              ^bb0(%arg31: memref<4096xf32, 1>, %arg32: memref<4096xbf16, 1>, %arg33: memref<1024xbf16, 1>):
                %c3072 = arith.constant 3072 : index
                %c1_i32_68 = arith.constant 1 : i32
                %c16_i32_69 = arith.constant 16 : i32
                %c0_i32_70 = arith.constant 0 : i32
                %c2048 = arith.constant 2048 : index
                %c32_i32_71 = arith.constant 32 : i32
                %c64_i32_72 = arith.constant 64 : i32
                %mask, %remaining_out = ascvec.update_mask %c64_i32_72 : !ascvec.mask
                %297 = ascvec.loadalign %arg31[%c3072] : memref<4096xf32, 1>, index -> <f32>
                scf.for %arg34 = %c0_i32_70 to %c16_i32_69 step %c1_i32_68  : i32 {
                  %298 = arith.muli %arg34, %c64_i32_72 overflow<nsw> : i32
                  %299 = arith.addi %arg34, %c32_i32_71 overflow<nsw> : i32
                  %300 = arith.muli %299, %c64_i32_72 overflow<nsw> : i32
                  %301 = arith.index_cast %298 : i32 to index
                  %302 = arith.addi %301, %c2048 : index
                  %303 = ascvec.loadalign_unpack %arg32[%302] {mode = "b16_to_b32"} : memref<4096xbf16, 1>, index -> <bf16>
                  %304 = ascvec.cast %303, %mask {rounding = 6 : i32} : <bf16> -> <f32>
                  %305 = arith.index_cast %300 : i32 to index
                  %306 = ascvec.loadalign %arg31[%305] : memref<4096xf32, 1>, index -> <f32>
                  %307 = ascvec.sub %297, %306, %mask : <f32>
                  %308 = ascvec.exp %307, %mask : <f32>
                  %309 = ascvec.mul %304, %308, %mask : <f32>
                  %310 = ascvec.cast %309, %mask : <f32> -> <bf16>
                  ascvec.storealign_pack %arg33[%301], %310, %mask {mode = "b32_to_b16"} : memref<1024xbf16, 1>, index, <bf16>
                } {cannir.loop_unroll = 2 : i64}
              } {cannir.vf_mode = "simd"}
              ascvec.get_buf v, %c15_i64, 0 {cannir.buffer.generated_sync}
              ascvec.vf(%alloc_45, %alloc_47 : memref<1024xbf16, 1>, memref<1088xbf16, 1>) {
              ^bb0(%arg31: memref<1024xbf16, 1>, %arg32: memref<1088xbf16, 1>):
                %c1_i32_68 = arith.constant 1 : i32
                %c0_i32_69 = arith.constant 0 : i32
                %c17 = arith.constant 17 : index
                %c16_i32_70 = arith.constant 16 : i32
                %c0_71 = arith.constant 0 : index
                %c64_i32_72 = arith.constant 64 : i32
                scf.for %arg33 = %c0_i32_69 to %c16_i32_70 step %c1_i32_68  : i32 {
                  %mask, %remaining_out = ascvec.update_mask %c64_i32_72 elem_bits 16 : !ascvec.mask
                  %297 = arith.muli %arg33, %c64_i32_72 overflow<nsw> : i32
                  %298 = arith.index_cast %297 : i32 to index
                  %299 = ascvec.loadalign %arg31[%298] : memref<1024xbf16, 1>, index -> <bf16>
                  %300 = arith.muli %arg33, %c16_i32_70 overflow<nsw> : i32
                  %301 = arith.index_cast %300 : i32 to index
                  ascvec.storealign_strided %arg32[%301], %299, %mask, %c17, %c0_71 : memref<1088xbf16, 1>, <bf16>
                }
              } {cannir.vf_mode = "simd"}
              ascvec.release_buf v, %c15_i64, 0 {cannir.buffer.generated_sync}
              ascvec.sync_intra_wait mte3, %c6_i64 {cannir.sync.banks = 3 : i32}
              ascvec.get_buf mte3, %c15_i64, 0 {cannir.buffer.generated_sync}
              ascvec.copy_ub2l1_strided %275[%273], %269[%c0], %c4_i32, %c16_i32, %c1_i32, %c0_i32 : memref<?xbf16, 2>, memref<?xbf16, 1>
              ascvec.release_buf mte3, %c15_i64, 0 {cannir.buffer.generated_sync}
              ascvec.sync_intra_arrive mte3, %c6_i64 {cannir.sync.banks = 3 : i32}
              ascvec.vf(%alloc_40, %alloc_26, %alloc_46 : memref<4096xf32, 1>, memref<4096xbf16, 1>, memref<1024xbf16, 1>) {
              ^bb0(%arg31: memref<4096xf32, 1>, %arg32: memref<4096xbf16, 1>, %arg33: memref<1024xbf16, 1>):
                %c1_i32_68 = arith.constant 1 : i32
                %c16_i32_69 = arith.constant 16 : i32
                %c0_i32_70 = arith.constant 0 : i32
                %c48_i32 = arith.constant 48 : i32
                %c3072 = arith.constant 3072 : index
                %c64_i32_71 = arith.constant 64 : i32
                %mask, %remaining_out = ascvec.update_mask %c64_i32_71 : !ascvec.mask
                %297 = ascvec.loadalign %arg31[%c3072] : memref<4096xf32, 1>, index -> <f32>
                scf.for %arg34 = %c0_i32_70 to %c16_i32_69 step %c1_i32_68  : i32 {
                  %298 = arith.muli %arg34, %c64_i32_71 overflow<nsw> : i32
                  %299 = arith.addi %arg34, %c48_i32 overflow<nsw> : i32
                  %300 = arith.muli %299, %c64_i32_71 overflow<nsw> : i32
                  %301 = arith.index_cast %298 : i32 to index
                  %302 = arith.addi %301, %c3072 : index
                  %303 = ascvec.loadalign_unpack %arg32[%302] {mode = "b16_to_b32"} : memref<4096xbf16, 1>, index -> <bf16>
                  %304 = ascvec.cast %303, %mask {rounding = 6 : i32} : <bf16> -> <f32>
                  %305 = arith.index_cast %300 : i32 to index
                  %306 = ascvec.loadalign %arg31[%305] : memref<4096xf32, 1>, index -> <f32>
                  %307 = ascvec.sub %297, %306, %mask : <f32>
                  %308 = ascvec.exp %307, %mask : <f32>
                  %309 = ascvec.mul %304, %308, %mask : <f32>
                  %310 = ascvec.cast %309, %mask : <f32> -> <bf16>
                  ascvec.storealign_pack %arg33[%301], %310, %mask {mode = "b32_to_b16"} : memref<1024xbf16, 1>, index, <bf16>
                } {cannir.loop_unroll = 2 : i64}
              } {cannir.vf_mode = "simd"}
              ascvec.get_buf v, %c16_i64, 0 {cannir.buffer.generated_sync}
              ascvec.vf(%alloc_46, %alloc_48 : memref<1024xbf16, 1>, memref<1088xbf16, 1>) {
              ^bb0(%arg31: memref<1024xbf16, 1>, %arg32: memref<1088xbf16, 1>):
                %c1_i32_68 = arith.constant 1 : i32
                %c0_i32_69 = arith.constant 0 : i32
                %c17 = arith.constant 17 : index
                %c16_i32_70 = arith.constant 16 : i32
                %c0_71 = arith.constant 0 : index
                %c64_i32_72 = arith.constant 64 : i32
                scf.for %arg33 = %c0_i32_69 to %c16_i32_70 step %c1_i32_68  : i32 {
                  %mask, %remaining_out = ascvec.update_mask %c64_i32_72 elem_bits 16 : !ascvec.mask
                  %297 = arith.muli %arg33, %c64_i32_72 overflow<nsw> : i32
                  %298 = arith.index_cast %297 : i32 to index
                  %299 = ascvec.loadalign %arg31[%298] : memref<1024xbf16, 1>, index -> <bf16>
                  %300 = arith.muli %arg33, %c16_i32_70 overflow<nsw> : i32
                  %301 = arith.index_cast %300 : i32 to index
                  ascvec.storealign_strided %arg32[%301], %299, %mask, %c17, %c0_71 : memref<1088xbf16, 1>, <bf16>
                }
              } {cannir.vf_mode = "simd"}
              ascvec.release_buf v, %c16_i64, 0 {cannir.buffer.generated_sync}
              ascvec.sync_intra_wait mte3, %c7_i64 {cannir.sync.banks = 3 : i32}
              ascvec.get_buf mte3, %c16_i64, 0 {cannir.buffer.generated_sync}
              ascvec.copy_ub2l1_strided %276[%273], %274[%c0], %c4_i32, %c16_i32, %c1_i32, %c0_i32 : memref<?xbf16, 2>, memref<?xbf16, 1>
              ascvec.release_buf mte3, %c16_i64, 0 {cannir.buffer.generated_sync}
              ascvec.sync_intra_arrive mte3, %c7_i64 {cannir.sync.banks = 3 : i32}
              %277 = cannir.extract_physical_layout %arg3 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 1)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 1)>, <(-1, -1, -1, 1)>>, identity>
              %278 = cannir.extract_stride %277 : <<(-1, -1, -1, 1)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
              %279 = cannir.extract_dim %278[2] : !cannir.stride<(-1, -1, -1, 1)>
              %280 = arith.muli %279, %c2_i64 : i64
              %281 = cannir.extract_dim %278[0] : !cannir.stride<(-1, -1, -1, 1)>
              %282 = arith.muli %97, %281 : i64
              %283 = arith.muli %282, %c2_i64 : i64
              %284 = cannir.extract_dim %278[1] : !cannir.stride<(-1, -1, -1, 1)>
              %285 = arith.muli %150, %284 : i64
              %286 = arith.muli %285, %c2_i64 : i64
              %287 = arith.addi %283, %286 : i64
              %288 = arith.muli %205, %279 : i64
              %289 = arith.muli %288, %c2_i64 : i64
              %290 = arith.addi %287, %289 : i64
              "ascvec.copy_gm2ub_plan"(%109, %alloc_2, %c0_i32, %c0_i32, %c0_i64, %290, %190, %c2_i64, %280, %c2_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = 74016 : i64, ub_capacity = 128 : i64}> : (memref<?xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
              ascvec.sync_notify {dst_pipe = 2 : i32, event_id = 0 : i32, src_pipe = 5 : i32}
              ascvec.sync_wait {dst_pipe = 2 : i32, event_id = 0 : i32, src_pipe = 5 : i32}
              ascvec.vf(%alloc_23, %alloc_24 : memref<64xbf16, 1>, memref<64xf32, 1>) {
              ^bb0(%arg31: memref<64xbf16, 1>, %arg32: memref<64xf32, 1>):
                %c1_i32_68 = arith.constant 1 : i32
                %c64_i32_69 = arith.constant 64 : i32
                %c0_i32_70 = arith.constant 0 : i32
                scf.for %arg33 = %c0_i32_70 to %c64_i32_69 step %c1_i32_68  : i32 {
                  %297 = arith.index_cast %arg33 : i32 to index
                  %298 = ascvec.loadalign_brc %arg31[%297] : memref<64xbf16, 1>, index -> <bf16>
                  %299 = ascvec.create_mask : !ascvec.mask
                  %300 = ascvec.cast %298, %299 {rounding = 6 : i32} : <bf16> -> <f32>
                  ascvec.storealign_1st %arg32[%297], %300 : memref<64xf32, 1>, index, <f32>
                }
              } {cannir.vf_mode = "simd"}
              ascvec.vf(%alloc_24, %181, %226 : memref<64xf32, 1>, i64, i1) {
              ^bb0(%arg31: memref<64xf32, 1>, %arg32: i64, %arg33: i1):
                %cst = arith.constant 0.000000e+00 : f32
                %cst_68 = arith.constant -1.000000e+00 : f32
                %c0_69 = arith.constant 0 : index
                %cst_70 = arith.constant 1.000000e+00 : f32
                %c64_i32_71 = arith.constant 64 : i32
                %mask, %remaining_out = ascvec.update_mask %c64_i32_71 : !ascvec.mask
                %297 = ascvec.duplicate_scalar %cst_70, %mask : f32 -> <f32>
                %298 = ascvec.loadalign %arg31[%c0_69] : memref<64xf32, 1>, index -> <f32>
                %299 = ascvec.mul_scalar %298, %cst_68, %mask : <f32>, f32
                %300 = ascvec.exp %299, %mask : <f32>
                %301 = ascvec.add %297, %300, %mask : <f32>
                %302 = ascvec.div %297, %301, %mask : <f32>
                scf.if %arg33 {
                  %303 = ascvec.duplicate_scalar %cst, %mask : f32 -> <f32>
                  %304 = arith.trunci %arg32 : i64 to i32
                  %c0_72 = arith.constant 0 : index
                  %c1_73 = arith.constant 1 : index
                  scf.for %arg34 = %c0_72 to %c1_73 step %c1_73 {
                    %mask_74, %remaining_out_75 = ascvec.update_mask %304 : !ascvec.mask
                    %305 = ascvec.mask_xor %mask, %mask_74, %mask
                    ascvec.storealign %arg31[%c0_69], %302, %mask_74 : memref<64xf32, 1>, index, <f32>
                    ascvec.storealign %arg31[%c0_69], %303, %305 : memref<64xf32, 1>, index, <f32>
                  }
                } else {
                  %c0_72 = arith.constant 0 : index
                  %c1_73 = arith.constant 1 : index
                  scf.for %arg34 = %c0_72 to %c1_73 step %c1_73 {
                    ascvec.storealign %arg31[%c0_69], %302, %mask : memref<64xf32, 1>, index, <f32>
                  }
                }
              } {cannir.vf_mode = "simd"}
              %291 = ascscalar.memref_address %alloc_33 : memref<2048xf32, 1>
              %292 = ascscalar.memref_address %alloc_34 : memref<2048xf32, 1>
              %293 = arith.addi %arg25, %c1_i64 : i64
              %294 = arith.cmpi slt, %119, %293 : i64
              %295 = arith.select %118, %294, %false : i1
              %296:2 = scf.if %295 -> (i64, i64) {
                %297 = arith.remsi %119, %c2_i64 : i64
                %298 = arith.index_cast %297 : i64 to index
                %299 = cannir.array_load %111[%298] : <2xi64>
                %300 = arith.remsi %299, %129 : i64
                %301 = arith.cmpi ne, %300, %c0_i64 : i64
                %302 = arith.cmpi slt, %300, %c0_i64 : i64
                %303 = arith.xori %302, %133 : i1
                %304 = arith.andi %301, %303 : i1
                %305 = arith.divsi %299, %129 : i64
                %306 = arith.extui %304 : i1 to i64
                %307 = arith.subi %305, %306 : i64
                %308 = arith.muli %307, %129 : i64
                %309 = arith.subi %299, %308 : i64
                %310 = arith.addi %141, %309 : i64
                %311 = arith.remsi %307, %146 : i64
                %312 = arith.cmpi slt, %311, %c0_i64 : i64
                %313 = arith.select %312, %146, %c0_i64 : i64
                %314 = arith.addi %311, %313 : i64
                %315 = arith.remsi %299, %c2_i64 : i64
                %316 = arith.cmpi ne, %315, %c0_i64 : i64
                %317 = arith.cmpi slt, %315, %c0_i64 : i64
                %318 = arith.andi %316, %317 : i1
                %319 = arith.select %318, %c2_i64, %c0_i64 : i64
                %320 = arith.addi %315, %319 overflow<nsw> : i64
                %321 = arith.cmpi eq, %arg26, %c0_i64 : i64
                %322 = arith.select %321, %291, %292 : i64
                %323 = ascscalar.bind_local_ptr %322 : memref<2048xf32, 1>
                %324 = arith.select %321, %c39_i64, %c40_i64 : i64
                %325 = arith.addi %arg26, %c1_i64 overflow<nsw> : i64
                %326 = arith.remui %325, %c2_i64 : i64
                %327 = arith.cmpi eq, %324, %c40_i64 : i64
                %328 = arith.select %327, %c3_i64, %c2_i64 : i64
                ascvec.sync_intra_wait v, %328 {cannir.sync.banks = 3 : i32}
                ascvec.get_buf v, %c13_i64, 0 {cannir.buffer.generated_sync}
                ascvec.vf(%102, %323, %alloc_39 : i64, memref<2048xf32, 1>, memref<2048xbf16, 1>) {
                ^bb0(%arg31: i64, %arg32: memref<2048xf32, 1>, %arg33: memref<2048xbf16, 1>):
                  %c0_i32_68 = arith.constant 0 : i32
                  %c1_i32_69 = arith.constant 1 : i32
                  %c32_i32_70 = arith.constant 32 : i32
                  %cst = arith.constant 0.000000e+00 : f32
                  %c64_i32_71 = arith.constant 64 : i32
                  %mask, %remaining_out = ascvec.update_mask %c64_i32_71 : !ascvec.mask
                  %404 = ascvec.duplicate_scalar %cst, %mask : f32 -> <f32>
                  scf.for %arg34 = %c0_i32_68 to %c32_i32_70 step %c1_i32_69  : i32 {
                    %405 = arith.muli %arg34, %c64_i32_71 overflow<nsw> : i32
                    %406 = arith.trunci %arg31 : i64 to i32
                    %407 = arith.muli %406, %c32_i32_70 : i32
                    %408 = arith.addi %407, %arg34 : i32
                    %409 = arith.addi %408, %c1_i32_69 : i32
                    %mask_72, %remaining_out_73 = ascvec.update_mask %409 : !ascvec.mask
                    %410 = arith.index_cast %405 : i32 to index
                    %411 = ascvec.loadalign %arg32[%410] : memref<2048xf32, 1>, index -> <f32>
                    %412 = ascvec.select %411, %404, %mask_72 : <f32>
                    %413 = ascvec.cast %412, %mask : <f32> -> <bf16>
                    ascvec.storealign_pack %arg33[%410], %413, %mask {mode = "b32_to_b16"} : memref<2048xbf16, 1>, index, <bf16>
                  } {cannir.loop_unroll = 4 : i64}
                } {cannir.vf_mode = "simd"}
                ascvec.release_buf v, %c13_i64, 0 {cannir.buffer.generated_sync}
                ascvec.sync_intra_arrive v, %328 {cannir.sync.banks = 3 : i32}
                %329 = cannir.extract_physical_layout %arg12 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 64, 64)>, <(-1, -1, 4096, 64, 1)>>, <<(-1, -1, -1, 64, 64)>, <(-1, -1, 4096, 64, 1)>>, identity>
                %330 = cannir.extract_stride %329 : <<(-1, -1, -1, 64, 64)>, <(-1, -1, 4096, 64, 1)>> -> <(-1, -1, 4096, 64, 1)>
                %331 = cannir.extract_dim %330[0] : !cannir.stride<(-1, -1, 4096, 64, 1)>
                %332 = arith.muli %98, %331 : i64
                %333 = cannir.extract_dim %330[1] : !cannir.stride<(-1, -1, 4096, 64, 1)>
                %334 = arith.muli %314, %333 : i64
                %335 = arith.addi %332, %334 : i64
                %336 = arith.muli %310, %c4096_i64 : i64
                %337 = arith.addi %335, %336 : i64
                %338 = arith.muli %102, %c2048_i64 : i64
                %339 = arith.addi %337, %338 : i64
                %340 = arith.muli %339, %c16_i64 : i64
                %341 = ascscalar.memref_address %alloc_11 : memref<1xbf16>
                %342 = arith.divui %340, %c8_i64 : i64
                %343 = arith.addi %341, %342 : i64
                %344 = ascscalar.bind_gm_ptr %343 : memref<1xbf16>
                %345 = ascscalar.bitcast_memref %alloc_39 byte_offset = %c0_i64 : memref<2048xbf16, 1> -> memref<?xbf16, 1>
                ascvec.get_buf mte3, %c13_i64, 0 {cannir.buffer.generated_sync}
                ascvec.copy_ub2gm %344[%c0], %345[%c0], %c1_i32, %c4096_i32, %c0_i64, %c4096_i32 l2_cache_ctl %c0_i32 : memref<1xbf16>, memref<?xbf16, 1>
                ascvec.release_buf mte3, %c13_i64, 0 {cannir.buffer.generated_sync}
                %346 = arith.select %321, %238, %239 : i64
                %347 = ascscalar.bind_local_ptr %346 : memref<64xf32, 1>
                %348 = arith.select %321, %c8_i64, %c9_i64 : i64
                %349 = arith.cmpi eq, %320, %c0_i64 : i64
                %350 = arith.select %349, %229, %230 : i64
                %351 = ascscalar.bind_local_ptr %350 : memref<4096xbf16, 1>
                %352 = arith.select %349, %233, %234 : i64
                %353 = ascscalar.bind_local_ptr %352 : memref<4096xf32, 1>
                ascvec.get_buf v, %348, 0 {cannir.buffer.generated_sync}
                ascvec.get_buf v, %c12_i64, 0 {cannir.buffer.generated_sync}
                ascvec.vf(%347, %351, %353, %alloc_37 : memref<64xf32, 1>, memref<4096xbf16, 1>, memref<4096xf32, 1>, memref<4096xbf16, 1>) {
                ^bb0(%arg31: memref<64xf32, 1>, %arg32: memref<4096xbf16, 1>, %arg33: memref<4096xf32, 1>, %arg34: memref<4096xbf16, 1>):
                  %c1_i32_68 = arith.constant 1 : i32
                  %c0_i32_69 = arith.constant 0 : i32
                  %c0_70 = arith.constant 0 : index
                  %c64_i32_71 = arith.constant 64 : i32
                  %mask, %remaining_out = ascvec.update_mask %c64_i32_71 : !ascvec.mask
                  %404 = ascvec.loadalign %arg31[%c0_70] : memref<64xf32, 1>, index -> <f32>
                  scf.for %arg35 = %c0_i32_69 to %c64_i32_71 step %c1_i32_68  : i32 {
                    %405 = arith.muli %arg35, %c64_i32_71 overflow<nsw> : i32
                    %406 = arith.index_cast %405 : i32 to index
                    %407 = ascvec.loadalign_unpack %arg32[%406] {mode = "b16_to_b32"} : memref<4096xbf16, 1>, index -> <bf16>
                    %408 = ascvec.cast %407, %mask {rounding = 6 : i32} : <bf16> -> <f32>
                    %409 = ascvec.loadalign %arg33[%406] : memref<4096xf32, 1>, index -> <f32>
                    %410 = ascvec.sub %404, %409, %mask : <f32>
                    %411 = ascvec.exp %410, %mask : <f32>
                    %412 = ascvec.mul %408, %411, %mask : <f32>
                    %413 = ascvec.cast %412, %mask : <f32> -> <bf16>
                    ascvec.storealign_pack %arg34[%406], %413, %mask {mode = "b32_to_b16"} : memref<4096xbf16, 1>, index, <bf16>
                  } {cannir.loop_unroll = 4 : i64}
                } {cannir.vf_mode = "simd"}
                ascvec.release_buf v, %c12_i64, 0 {cannir.buffer.generated_sync}
                ascvec.release_buf v, %348, 0 {cannir.buffer.generated_sync}
                %354 = arith.minsi %310, %184 : i64
                %355 = arith.muli %354, %c64_i64 : i64
                %356 = arith.subi %78, %355 : i64
                %357 = arith.minsi %356, %c64_i64 : i64
                %358 = arith.cmpi sle, %310, %184 : i64
                %359 = arith.select %358, %357, %c0_i64 : i64
                %360 = cannir.extract_physical_layout %arg8 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                %361 = cannir.extract_stride %360 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                %362 = cannir.extract_dim %361[0] : !cannir.stride<(-1, -1, -1, 1)>
                %363 = arith.muli %97, %362 : i64
                %364 = cannir.extract_dim %361[1] : !cannir.stride<(-1, -1, -1, 1)>
                %365 = arith.muli %314, %364 : i64
                %366 = arith.addi %363, %365 : i64
                %367 = arith.muli %310, %c64_i64 : i64
                %368 = arith.addi %367, %96 : i64
                %369 = cannir.extract_dim %361[2] : !cannir.stride<(-1, -1, -1, 1)>
                %370 = arith.muli %368, %369 : i64
                %371 = arith.addi %366, %370 : i64
                %372 = arith.addi %371, %255 : i64
                %373 = arith.muli %372, %c16_i64 : i64
                %374 = ascscalar.memref_address %alloc_7 : memref<1xbf16>
                %375 = arith.divui %373, %c8_i64 : i64
                %376 = arith.addi %374, %375 : i64
                %377 = ascscalar.bind_gm_ptr %376 : memref<1xbf16>
                %378 = cannir.extract_layout %arg8 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                %379 = cannir.extract_stride %378 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                %380 = cannir.extract_dim %379[2] : !cannir.stride<(-1, -1, -1, 1)>
                %381 = arith.index_cast %380 : i64 to index
                %382 = arith.muli %381, %c2 : index
                %383 = arith.cmpi ugt, %359, %c0_i64 : i64
                scf.if %383 {
                  %404 = arith.index_cast %382 : index to i64
                  ascvec.get_buf mte3, %c12_i64, 0 {cannir.buffer.generated_sync}
                  %405 = arith.trunci %359 : i64 to i32
                  ascvec.copy_ub2gm %377[%c0], %262[%c0], %405, %c128_i32, %404, %c128_i32 l2_cache_ctl %c0_i32 : memref<1xbf16>, memref<?xbf16, 1>
                  ascvec.release_buf mte3, %c12_i64, 0 {cannir.buffer.generated_sync}
                }
                ascvec.get_buf v, %348, 0 {cannir.buffer.generated_sync}
                ascvec.vf(%347 : memref<64xf32, 1>) {
                ^bb0(%arg31: memref<64xf32, 1>):
                  %c0_68 = arith.constant 0 : index
                  %c64_i32_69 = arith.constant 64 : i32
                  %c0_70 = arith.constant 0 : index
                  %c1_71 = arith.constant 1 : index
                  scf.for %arg32 = %c0_70 to %c1_71 step %c1_71 {
                    %mask, %remaining_out = ascvec.update_mask %c64_i32_69 : !ascvec.mask
                    %404 = ascvec.loadalign %arg31[%c0_68] : memref<64xf32, 1>, index -> <f32>
                    %405 = ascvec.exp %404, %mask : <f32>
                    ascvec.storealign %arg31[%c0_68], %405, %mask : memref<64xf32, 1>, index, <f32>
                  }
                } {cannir.vf_mode = "simd"}
                ascvec.release_buf v, %348, 0 {cannir.buffer.generated_sync}
                %384 = cannir.extract_physical_layout %arg9 : !cannir.tensor<<gm, f32>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                %385 = cannir.extract_stride %384 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, 128, 1)>
                %386 = cannir.extract_dim %385[0] : !cannir.stride<(-1, -1, 128, 1)>
                %387 = arith.muli %98, %386 : i64
                %388 = cannir.extract_dim %385[1] : !cannir.stride<(-1, -1, 128, 1)>
                %389 = arith.muli %314, %388 : i64
                %390 = arith.addi %387, %389 : i64
                %391 = arith.muli %310, %c128_i64 : i64
                %392 = arith.addi %390, %391 : i64
                %393 = arith.addi %392, %255 : i64
                %394 = arith.muli %393, %c32_i64 : i64
                %395 = ascscalar.memref_address %alloc_8 : memref<1xf32>
                %396 = arith.divui %394, %c8_i64 : i64
                %397 = arith.addi %395, %396 : i64
                %398 = ascscalar.bind_gm_ptr %397 : memref<1xf32>
                %399 = ascscalar.bind_local_ptr %346 : memref<1xf32, 1>
                ascvec.get_buf mte3, %348, 0 {cannir.buffer.generated_sync}
                ascvec.copy_ub2gm %398[%c0], %399[%c0], %c1_i32, %c256_i32, %c0_i64, %c256_i32 l2_cache_ctl %c0_i32 : memref<1xf32>, memref<1xf32, 1>
                ascvec.release_buf mte3, %348, 0 {cannir.buffer.generated_sync}
                %400 = arith.addi %arg27, %c1_i64 overflow<nsw> : i64
                %401 = arith.remui %400, %c2_i64 : i64
                %402 = arith.addi %401, %c1_i64 overflow<nsw> : i64
                %403 = arith.remui %402, %c2_i64 : i64
                scf.yield %326, %403 : i64, i64
              } else {
                scf.yield %arg26, %arg27 : i64, i64
              }
              scf.yield %293, %296#0, %296#1, %122#0, %244, %122#1 : i64, i64, i64, i64, i64, i64
            }
            %114 = arith.cmpi sge, %113#0, %c1_i64 : i64
            %115 = arith.subi %113#0, %c1_i64 : i64
            %116 = arith.cmpi slt, %115, %113#0 : i64
            %117 = arith.select %114, %116, %false : i1
            scf.if %117 {
              %118 = arith.remsi %115, %c2_i64 : i64
              %119 = arith.index_cast %118 : i64 to index
              %120 = cannir.array_load %111[%119] : <2xi64>
              %121 = arith.extsi %84 : i32 to i64
              %122 = arith.remsi %120, %121 : i64
              %123 = arith.cmpi ne, %122, %c0_i64 : i64
              %124 = arith.cmpi slt, %122, %c0_i64 : i64
              %125 = arith.cmpi slt, %121, %c0_i64 : i64
              %126 = arith.xori %124, %125 : i1
              %127 = arith.andi %123, %126 : i1
              %128 = arith.divsi %120, %121 : i64
              %129 = arith.extui %127 : i1 to i64
              %130 = arith.subi %128, %129 : i64
              %131 = arith.muli %130, %121 : i64
              %132 = arith.subi %120, %131 : i64
              %133 = arith.extsi %81 : i32 to i64
              %134 = arith.addi %133, %132 : i64
              %135 = cannir.extract_layout %arg1 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
              %136 = cannir.extract_shape %135 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 128)>
              %137 = cannir.extract_dim %136[1] : !cannir.shape<(-1, -1, -1, 128)>
              %138 = arith.maxsi %137, %c1_i64 : i64
              %139 = arith.remsi %130, %138 : i64
              %140 = arith.cmpi slt, %139, %c0_i64 : i64
              %141 = arith.select %140, %138, %c0_i64 : i64
              %142 = arith.addi %139, %141 : i64
              %143 = arith.remsi %120, %c2_i64 : i64
              %144 = arith.cmpi ne, %143, %c0_i64 : i64
              %145 = arith.cmpi slt, %143, %c0_i64 : i64
              %146 = arith.andi %144, %145 : i1
              %147 = arith.select %146, %c2_i64, %c0_i64 : i64
              %148 = arith.addi %143, %147 overflow<nsw> : i64
              %149 = cannir.array_load %112[%119] : <2xi64>
              %150 = arith.muli %102, %c32_i64 : i64
              ascvec.sync_intra_wait v, %c4_i64 {cannir.sync.banks = 3 : i32}
              ascvec.vf(%alloc_43, %150, %alloc_24, %alloc_38 : memref<2048xf32, 1>, i64, memref<64xf32, 1>, memref<2048xf16, 1>) {
              ^bb0(%arg24: memref<2048xf32, 1>, %arg25: i64, %arg26: memref<64xf32, 1>, %arg27: memref<2048xf16, 1>):
                %c1_i32_68 = arith.constant 1 : i32
                %c32_i32_69 = arith.constant 32 : i32
                %c0_i32_70 = arith.constant 0 : i32
                %cst = arith.constant -1.000000e+00 : f32
                %cst_71 = arith.constant 0.000000e+00 : f32
                %c64_i32_72 = arith.constant 64 : i32
                %mask, %remaining_out = ascvec.update_mask %c64_i32_72 : !ascvec.mask
                %269 = ascvec.duplicate_scalar %cst_71, %mask : f32 -> <f32>
                scf.for %arg28 = %c0_i32_70 to %c32_i32_69 step %c1_i32_68  : i32 {
                  %270 = arith.muli %arg28, %c64_i32_72 overflow<nsw> : i32
                  %271 = arith.index_cast %270 : i32 to index
                  %272 = ascvec.loadalign %arg24[%271] : memref<2048xf32, 1>, index -> <f32>
                  %273 = arith.trunci %arg25 : i64 to i32
                  %274 = arith.addi %273, %arg28 : i32
                  %275 = arith.index_cast %274 : i32 to index
                  %276 = ascvec.loadalign_brc %arg26[%275] : memref<64xf32, 1>, index -> <f32>
                  %277 = ascvec.mul %272, %276, %mask : <f32>
                  %mask_73, %remaining_out_74 = ascvec.update_mask %274 : !ascvec.mask
                  %278 = ascvec.select %277, %269, %mask_73 : <f32>
                  %279 = ascvec.mul_scalar %278, %cst, %mask : <f32>, f32
                  %280 = ascvec.cast %279, %mask : <f32> -> <f16>
                  ascvec.storealign_pack %arg27[%271], %280, %mask {mode = "b32_to_b16"} : memref<2048xf16, 1>, index, <f16>
                } {cannir.loop_unroll = 4 : i64}
              } {cannir.vf_mode = "simd"}
              ascvec.sync_intra_arrive v, %c4_i64 {cannir.sync.banks = 3 : i32}
              ascvec.get_buf v, %c14_i64, 0 {cannir.buffer.generated_sync}
              ascvec.vf(%alloc_38, %alloc_44 : memref<2048xf16, 1>, memref<2112xf16, 1>) {
              ^bb0(%arg24: memref<2048xf16, 1>, %arg25: memref<2112xf16, 1>):
                %c1_i32_68 = arith.constant 1 : i32
                %c32_i32_69 = arith.constant 32 : i32
                %c0_i32_70 = arith.constant 0 : i32
                %c33 = arith.constant 33 : index
                %c16_i32_71 = arith.constant 16 : i32
                %c0_72 = arith.constant 0 : index
                %c64_i32_73 = arith.constant 64 : i32
                scf.for %arg26 = %c0_i32_70 to %c32_i32_69 step %c1_i32_68  : i32 {
                  %mask, %remaining_out = ascvec.update_mask %c64_i32_73 elem_bits 16 : !ascvec.mask
                  %269 = arith.muli %arg26, %c64_i32_73 overflow<nsw> : i32
                  %270 = arith.index_cast %269 : i32 to index
                  %271 = ascvec.loadalign %arg24[%270] : memref<2048xf16, 1>, index -> <f16>
                  %272 = arith.muli %arg26, %c16_i32_71 overflow<nsw> : i32
                  %273 = arith.index_cast %272 : i32 to index
                  ascvec.storealign_strided %arg25[%273], %271, %mask, %c33, %c0_72 : memref<2112xf16, 1>, <f16>
                }
              } {cannir.vf_mode = "simd"}
              ascvec.release_buf v, %c14_i64, 0 {cannir.buffer.generated_sync}
              %151 = ascscalar.bitcast_memref %alloc_44 byte_offset = %c0_i64 : memref<2112xf16, 1> -> memref<?xf16, 1>
              %152 = ascscalar.bitcast_memref %alloc_64 byte_offset = %c0_i64 : memref<4096xf16, 2> -> memref<?xf16, 2>
              %153 = arith.index_cast %150 : i64 to index
              %154 = arith.muli %153, %c16 : index
              ascvec.sync_intra_wait mte3, %c1_i64 {cannir.sync.banks = 3 : i32}
              ascvec.get_buf mte3, %c14_i64, 0 {cannir.buffer.generated_sync}
              ascvec.copy_ub2l1_strided %152[%154], %151[%c0], %c4_i32, %c32_i32, %c1_i32, %c32_i32 : memref<?xf16, 2>, memref<?xf16, 1>
              ascvec.release_buf mte3, %c14_i64, 0 {cannir.buffer.generated_sync}
              ascvec.sync_intra_arrive mte3, %c1_i64 {cannir.sync.banks = 3 : i32}
              %155 = arith.cmpi eq, %148, %c0_i64 : i64
              %156 = ascscalar.memref_address %alloc_41 : memref<4096xbf16, 1>
              %157 = ascscalar.memref_address %alloc_42 : memref<4096xbf16, 1>
              %158 = arith.select %155, %156, %157 : i64
              %159 = ascscalar.bind_local_ptr %158 : memref<4096xbf16, 1>
              %160 = arith.cmpi slt, %149, %c64_i64 : i64
              scf.if %160 {
                %269 = arith.divsi %78, %c64_i64 : i64
                %270 = arith.minsi %134, %269 : i64
                %271 = arith.muli %270, %c64_i64 : i64
                %272 = arith.subi %78, %271 : i64
                %273 = arith.minsi %272, %c64_i64 : i64
                %274 = arith.cmpi sle, %134, %269 : i64
                %275 = arith.select %274, %273, %c0_i64 : i64
                %276 = arith.trunci %275 : i64 to i32
                ascvec.vf(%alloc_27, %276 : memref<4096xbf16, 1>, i32) {
                ^bb0(%arg24: memref<4096xbf16, 1>, %arg25: i32):
                  %c1_i32_68 = arith.constant 1 : i32
                  %cst = arith.constant 0.000000e+00 : bf16
                  %c64_i32_69 = arith.constant 64 : i32
                  %mask, %remaining_out = ascvec.update_mask %c64_i32_69 elem_bits 16 : !ascvec.mask
                  %315 = ascvec.duplicate_scalar %cst, %mask : bf16 -> <bf16>
                  scf.for %arg26 = %arg25 to %c64_i32_69 step %c1_i32_68  : i32 {
                    %316 = arith.muli %arg26, %c64_i32_69 : i32
                    %317 = arith.index_cast %316 : i32 to index
                    ascvec.storealign %arg24[%317], %315, %mask : memref<4096xbf16, 1>, index, <bf16>
                  } {cannir.loop_unroll = 4 : i64}
                  ascvec.mem_bar vst_vld
                } {cannir.vf_mode = "simd"}
                ascvec.sync_notify {dst_pipe = 5 : i32, event_id = 0 : i32, src_pipe = 2 : i32}
                ascvec.sync_wait {dst_pipe = 5 : i32, event_id = 0 : i32, src_pipe = 2 : i32}
                %277 = cannir.extract_physical_layout %arg1 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                %278 = cannir.extract_stride %277 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                %279 = cannir.extract_dim %278[2] : !cannir.stride<(-1, -1, -1, 1)>
                %280 = arith.muli %279, %c2_i64 : i64
                %281 = cannir.extract_dim %278[0] : !cannir.stride<(-1, -1, -1, 1)>
                %282 = arith.muli %97, %281 : i64
                %283 = arith.muli %282, %c2_i64 : i64
                %284 = cannir.extract_dim %278[1] : !cannir.stride<(-1, -1, -1, 1)>
                %285 = arith.muli %142, %284 : i64
                %286 = arith.muli %285, %c2_i64 : i64
                %287 = arith.addi %283, %286 : i64
                %288 = arith.muli %102, %c128_i64 : i64
                %289 = arith.addi %287, %288 : i64
                %290 = arith.muli %134, %c64_i64 : i64
                %291 = arith.addi %290, %96 : i64
                %292 = arith.muli %291, %279 : i64
                %293 = arith.muli %292, %c2_i64 : i64
                %294 = arith.addi %289, %293 : i64
                "ascvec.copy_gm2ub_plan"(%110, %alloc_0, %c0_i32, %c0_i32, %c0_i64, %294, %275, %c128_i64, %280, %c128_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = 90784 : i64, ub_capacity = 8192 : i64}> : (memref<?xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                ascvec.sync_notify {dst_pipe = 2 : i32, event_id = 1 : i32, src_pipe = 5 : i32}
                ascvec.get_buf v, %c12_i64, 0 {cannir.buffer.generated_sync}
                ascvec.vf(%alloc_24, %alloc_28, %159, %alloc_37 : memref<64xf32, 1>, memref<4096xf32, 1>, memref<4096xbf16, 1>, memref<4096xbf16, 1>) {
                ^bb0(%arg24: memref<64xf32, 1>, %arg25: memref<4096xf32, 1>, %arg26: memref<4096xbf16, 1>, %arg27: memref<4096xbf16, 1>):
                  %c1_i32_68 = arith.constant 1 : i32
                  %c0_i32_69 = arith.constant 0 : i32
                  %cst = arith.constant -1.000000e+00 : f32
                  %c64_i32_70 = arith.constant 64 : i32
                  %mask, %remaining_out = ascvec.update_mask %c64_i32_70 : !ascvec.mask
                  scf.for %arg28 = %c0_i32_69 to %c64_i32_70 step %c1_i32_68  : i32 {
                    %315 = arith.muli %arg28, %c64_i32_70 overflow<nsw> : i32
                    %316 = arith.index_cast %arg28 : i32 to index
                    %317 = ascvec.loadalign_brc %arg24[%316] : memref<64xf32, 1>, index -> <f32>
                    %318 = ascvec.mul_scalar %317, %cst, %mask : <f32>, f32
                    %319 = arith.index_cast %315 : i32 to index
                    %320 = ascvec.loadalign %arg25[%319] : memref<4096xf32, 1>, index -> <f32>
                    %321 = ascvec.loadalign_unpack %arg26[%319] {mode = "b16_to_b32"} : memref<4096xbf16, 1>, index -> <bf16>
                    %322 = ascvec.cast %321, %mask {rounding = 6 : i32} : <bf16> -> <f32>
                    %323 = ascvec.exp %320, %mask : <f32>
                    %324 = ascvec.mul %322, %323, %mask : <f32>
                    %325 = ascvec.mul %324, %318, %mask : <f32>
                    %326 = ascvec.cast %325, %mask : <f32> -> <bf16>
                    ascvec.storealign_pack %arg27[%319], %326, %mask {mode = "b32_to_b16"} : memref<4096xbf16, 1>, index, <bf16>
                  } {cannir.loop_unroll = 4 : i64}
                } {cannir.vf_mode = "simd"}
                %295 = arith.cmpi eq, %113#5, %c0_i64 : i64
                %296 = ascscalar.memref_address %alloc_35 : memref<4160xbf16, 1>
                %297 = ascscalar.memref_address %alloc_36 : memref<4160xbf16, 1>
                %298 = arith.select %295, %296, %297 : i64
                %299 = ascscalar.bind_local_ptr %298 : memref<4160xbf16, 1>
                %300 = arith.select %295, %c10_i64, %c11_i64 : i64
                %301 = arith.addi %113#5, %c1_i64 overflow<nsw> : i64
                %302 = arith.remui %301, %c2_i64 : i64
                ascvec.get_buf v, %300, 0 {cannir.buffer.generated_sync}
                ascvec.vf(%alloc_37, %299 : memref<4096xbf16, 1>, memref<4160xbf16, 1>) {
                ^bb0(%arg24: memref<4096xbf16, 1>, %arg25: memref<4160xbf16, 1>):
                  %c1_i32_68 = arith.constant 1 : i32
                  %c0_i32_69 = arith.constant 0 : i32
                  %c65 = arith.constant 65 : index
                  %c16_i32_70 = arith.constant 16 : i32
                  %c0_71 = arith.constant 0 : index
                  %c64_i32_72 = arith.constant 64 : i32
                  scf.for %arg26 = %c0_i32_69 to %c64_i32_72 step %c1_i32_68  : i32 {
                    %mask, %remaining_out = ascvec.update_mask %c64_i32_72 elem_bits 16 : !ascvec.mask
                    %315 = arith.muli %arg26, %c64_i32_72 overflow<nsw> : i32
                    %316 = arith.index_cast %315 : i32 to index
                    %317 = ascvec.loadalign %arg24[%316] : memref<4096xbf16, 1>, index -> <bf16>
                    %318 = arith.muli %arg26, %c16_i32_70 overflow<nsw> : i32
                    %319 = arith.index_cast %318 : i32 to index
                    ascvec.storealign_strided %arg25[%319], %317, %mask, %c65, %c0_71 : memref<4160xbf16, 1>, <bf16>
                  }
                } {cannir.vf_mode = "simd"}
                ascvec.release_buf v, %300, 0 {cannir.buffer.generated_sync}
                ascvec.release_buf v, %c12_i64, 0 {cannir.buffer.generated_sync}
                %303 = ascscalar.bind_local_ptr %298 : memref<1xbf16, 1>
                %304 = ascscalar.bitcast_memref %alloc_63 byte_offset = %c0_i64 : memref<8192xbf16, 2> -> memref<?xbf16, 2>
                %305 = arith.muli %102, %c64_i64 : i64
                %306 = arith.index_cast %305 : i64 to index
                %307 = arith.divui %306, %c16 : index
                %308 = arith.muli %307, %c1024 : index
                ascvec.sync_intra_wait mte3, %c9_i64 {cannir.sync.banks = 3 : i32}
                ascvec.get_buf mte3, %300, 0 {cannir.buffer.generated_sync}
                ascvec.copy_ub2l1_strided %304[%308], %303[%c0], %c4_i32, %c64_i32, %c1_i32, %c0_i32 : memref<?xbf16, 2>, memref<1xbf16, 1>
                ascvec.release_buf mte3, %300, 0 {cannir.buffer.generated_sync}
                ascvec.sync_intra_arrive mte3, %c9_i64 {cannir.sync.banks = 3 : i32}
                ascvec.sync_wait {dst_pipe = 2 : i32, event_id = 1 : i32, src_pipe = 5 : i32}
                ascvec.get_buf v, %c7_i64, 0 {cannir.buffer.generated_sync}
                ascvec.vf(%alloc_24, %alloc_27, %alloc_37 : memref<64xf32, 1>, memref<4096xbf16, 1>, memref<4096xbf16, 1>) {
                ^bb0(%arg24: memref<64xf32, 1>, %arg25: memref<4096xbf16, 1>, %arg26: memref<4096xbf16, 1>):
                  %c1_i32_68 = arith.constant 1 : i32
                  %c0_i32_69 = arith.constant 0 : i32
                  %c64_i32_70 = arith.constant 64 : i32
                  %mask, %remaining_out = ascvec.update_mask %c64_i32_70 : !ascvec.mask
                  scf.for %arg27 = %c0_i32_69 to %c64_i32_70 step %c1_i32_68  : i32 {
                    %315 = arith.muli %arg27, %c64_i32_70 overflow<nsw> : i32
                    %316 = arith.index_cast %arg27 : i32 to index
                    %317 = ascvec.loadalign_brc %arg24[%316] : memref<64xf32, 1>, index -> <f32>
                    %318 = arith.index_cast %315 : i32 to index
                    %319 = ascvec.loadalign_unpack %arg25[%318] {mode = "b16_to_b32"} : memref<4096xbf16, 1>, index -> <bf16>
                    %320 = ascvec.cast %319, %mask {rounding = 6 : i32} : <bf16> -> <f32>
                    %321 = ascvec.mul %320, %317, %mask : <f32>
                    %322 = ascvec.cast %321, %mask : <f32> -> <bf16>
                    ascvec.storealign_pack %arg26[%318], %322, %mask {mode = "b32_to_b16"} : memref<4096xbf16, 1>, index, <bf16>
                  } {cannir.loop_unroll = 4 : i64}
                } {cannir.vf_mode = "simd"}
                ascvec.release_buf v, %c7_i64, 0 {cannir.buffer.generated_sync}
                %309 = arith.cmpi eq, %302, %c0_i64 : i64
                %310 = arith.select %309, %296, %297 : i64
                %311 = ascscalar.bind_local_ptr %310 : memref<4160xbf16, 1>
                %312 = arith.select %309, %c10_i64, %c11_i64 : i64
                ascvec.get_buf v, %312, 0 {cannir.buffer.generated_sync}
                ascvec.vf(%alloc_37, %311 : memref<4096xbf16, 1>, memref<4160xbf16, 1>) {
                ^bb0(%arg24: memref<4096xbf16, 1>, %arg25: memref<4160xbf16, 1>):
                  %c1_i32_68 = arith.constant 1 : i32
                  %c0_i32_69 = arith.constant 0 : i32
                  %c65 = arith.constant 65 : index
                  %c16_i32_70 = arith.constant 16 : i32
                  %c0_71 = arith.constant 0 : index
                  %c64_i32_72 = arith.constant 64 : i32
                  scf.for %arg26 = %c0_i32_69 to %c64_i32_72 step %c1_i32_68  : i32 {
                    %mask, %remaining_out = ascvec.update_mask %c64_i32_72 elem_bits 16 : !ascvec.mask
                    %315 = arith.muli %arg26, %c64_i32_72 overflow<nsw> : i32
                    %316 = arith.index_cast %315 : i32 to index
                    %317 = ascvec.loadalign %arg24[%316] : memref<4096xbf16, 1>, index -> <bf16>
                    %318 = arith.muli %arg26, %c16_i32_70 overflow<nsw> : i32
                    %319 = arith.index_cast %318 : i32 to index
                    ascvec.storealign_strided %arg25[%319], %317, %mask, %c65, %c0_71 : memref<4160xbf16, 1>, <bf16>
                  }
                } {cannir.vf_mode = "simd"}
                ascvec.release_buf v, %312, 0 {cannir.buffer.generated_sync}
                %313 = ascscalar.bind_local_ptr %310 : memref<1xbf16, 1>
                %314 = ascscalar.bitcast_memref %alloc_62 byte_offset = %c0_i64 : memref<8192xbf16, 2> -> memref<?xbf16, 2>
                ascvec.sync_intra_wait mte3, %c8_i64 {cannir.sync.banks = 3 : i32}
                ascvec.get_buf mte3, %312, 0 {cannir.buffer.generated_sync}
                ascvec.copy_ub2l1_strided %314[%308], %313[%c0], %c4_i32, %c64_i32, %c1_i32, %c0_i32 : memref<?xbf16, 2>, memref<1xbf16, 1>
                ascvec.release_buf mte3, %312, 0 {cannir.buffer.generated_sync}
                ascvec.sync_intra_arrive mte3, %c8_i64 {cannir.sync.banks = 3 : i32}
              } else {
                %269 = arith.divsi %78, %c64_i64 : i64
                %270 = arith.minsi %134, %269 : i64
                %271 = arith.muli %270, %c64_i64 : i64
                %272 = arith.subi %78, %271 : i64
                %273 = arith.minsi %272, %c64_i64 : i64
                %274 = arith.cmpi sle, %134, %269 : i64
                %275 = arith.select %274, %273, %c0_i64 : i64
                %276 = cannir.extract_physical_layout %arg1 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                %277 = cannir.extract_stride %276 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                %278 = cannir.extract_dim %277[2] : !cannir.stride<(-1, -1, -1, 1)>
                %279 = arith.muli %278, %c2_i64 : i64
                %280 = cannir.extract_dim %277[0] : !cannir.stride<(-1, -1, -1, 1)>
                %281 = arith.muli %97, %280 : i64
                %282 = arith.muli %281, %c2_i64 : i64
                %283 = cannir.extract_dim %277[1] : !cannir.stride<(-1, -1, -1, 1)>
                %284 = arith.muli %142, %283 : i64
                %285 = arith.muli %284, %c2_i64 : i64
                %286 = arith.addi %282, %285 : i64
                %287 = arith.muli %102, %c128_i64 : i64
                %288 = arith.addi %286, %287 : i64
                %289 = arith.muli %134, %c64_i64 : i64
                %290 = arith.addi %289, %96 : i64
                %291 = arith.muli %290, %278 : i64
                %292 = arith.muli %291, %c2_i64 : i64
                %293 = arith.addi %288, %292 : i64
                ascvec.get_buf mte2, %c7_i64, 0 {cannir.buffer.generated_sync}
                "ascvec.copy_gm2ub_plan"(%110, %alloc_0, %c0_i32, %c0_i32, %c0_i64, %293, %275, %c128_i64, %279, %c128_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = 90784 : i64, ub_capacity = 8192 : i64}> : (memref<?xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                ascvec.release_buf mte2, %c7_i64, 0 {cannir.buffer.generated_sync}
                ascvec.sync_notify {dst_pipe = 2 : i32, event_id = 1 : i32, src_pipe = 5 : i32}
                ascvec.get_buf v, %c12_i64, 0 {cannir.buffer.generated_sync}
                ascvec.vf(%alloc_24, %alloc_28, %159, %alloc_37 : memref<64xf32, 1>, memref<4096xf32, 1>, memref<4096xbf16, 1>, memref<4096xbf16, 1>) {
                ^bb0(%arg24: memref<64xf32, 1>, %arg25: memref<4096xf32, 1>, %arg26: memref<4096xbf16, 1>, %arg27: memref<4096xbf16, 1>):
                  %c1_i32_68 = arith.constant 1 : i32
                  %c0_i32_69 = arith.constant 0 : i32
                  %cst = arith.constant -1.000000e+00 : f32
                  %c64_i32_70 = arith.constant 64 : i32
                  %mask, %remaining_out = ascvec.update_mask %c64_i32_70 : !ascvec.mask
                  scf.for %arg28 = %c0_i32_69 to %c64_i32_70 step %c1_i32_68  : i32 {
                    %314 = arith.muli %arg28, %c64_i32_70 overflow<nsw> : i32
                    %315 = arith.index_cast %arg28 : i32 to index
                    %316 = ascvec.loadalign_brc %arg24[%315] : memref<64xf32, 1>, index -> <f32>
                    %317 = ascvec.mul_scalar %316, %cst, %mask : <f32>, f32
                    %318 = arith.index_cast %314 : i32 to index
                    %319 = ascvec.loadalign %arg25[%318] : memref<4096xf32, 1>, index -> <f32>
                    %320 = ascvec.loadalign_unpack %arg26[%318] {mode = "b16_to_b32"} : memref<4096xbf16, 1>, index -> <bf16>
                    %321 = ascvec.cast %320, %mask {rounding = 6 : i32} : <bf16> -> <f32>
                    %322 = ascvec.exp %319, %mask : <f32>
                    %323 = ascvec.mul %321, %322, %mask : <f32>
                    %324 = ascvec.mul %323, %317, %mask : <f32>
                    %325 = ascvec.cast %324, %mask : <f32> -> <bf16>
                    ascvec.storealign_pack %arg27[%318], %325, %mask {mode = "b32_to_b16"} : memref<4096xbf16, 1>, index, <bf16>
                  } {cannir.loop_unroll = 4 : i64}
                } {cannir.vf_mode = "simd"}
                %294 = arith.cmpi eq, %113#5, %c0_i64 : i64
                %295 = ascscalar.memref_address %alloc_35 : memref<4160xbf16, 1>
                %296 = ascscalar.memref_address %alloc_36 : memref<4160xbf16, 1>
                %297 = arith.select %294, %295, %296 : i64
                %298 = ascscalar.bind_local_ptr %297 : memref<4160xbf16, 1>
                %299 = arith.select %294, %c10_i64, %c11_i64 : i64
                %300 = arith.addi %113#5, %c1_i64 overflow<nsw> : i64
                %301 = arith.remui %300, %c2_i64 : i64
                ascvec.get_buf v, %299, 0 {cannir.buffer.generated_sync}
                ascvec.vf(%alloc_37, %298 : memref<4096xbf16, 1>, memref<4160xbf16, 1>) {
                ^bb0(%arg24: memref<4096xbf16, 1>, %arg25: memref<4160xbf16, 1>):
                  %c1_i32_68 = arith.constant 1 : i32
                  %c0_i32_69 = arith.constant 0 : i32
                  %c65 = arith.constant 65 : index
                  %c16_i32_70 = arith.constant 16 : i32
                  %c0_71 = arith.constant 0 : index
                  %c64_i32_72 = arith.constant 64 : i32
                  scf.for %arg26 = %c0_i32_69 to %c64_i32_72 step %c1_i32_68  : i32 {
                    %mask, %remaining_out = ascvec.update_mask %c64_i32_72 elem_bits 16 : !ascvec.mask
                    %314 = arith.muli %arg26, %c64_i32_72 overflow<nsw> : i32
                    %315 = arith.index_cast %314 : i32 to index
                    %316 = ascvec.loadalign %arg24[%315] : memref<4096xbf16, 1>, index -> <bf16>
                    %317 = arith.muli %arg26, %c16_i32_70 overflow<nsw> : i32
                    %318 = arith.index_cast %317 : i32 to index
                    ascvec.storealign_strided %arg25[%318], %316, %mask, %c65, %c0_71 : memref<4160xbf16, 1>, <bf16>
                  }
                } {cannir.vf_mode = "simd"}
                ascvec.release_buf v, %299, 0 {cannir.buffer.generated_sync}
                ascvec.release_buf v, %c12_i64, 0 {cannir.buffer.generated_sync}
                %302 = ascscalar.bind_local_ptr %297 : memref<1xbf16, 1>
                %303 = ascscalar.bitcast_memref %alloc_63 byte_offset = %c0_i64 : memref<8192xbf16, 2> -> memref<?xbf16, 2>
                %304 = arith.muli %102, %c64_i64 : i64
                %305 = arith.index_cast %304 : i64 to index
                %306 = arith.divui %305, %c16 : index
                %307 = arith.muli %306, %c1024 : index
                ascvec.sync_intra_wait mte3, %c9_i64 {cannir.sync.banks = 3 : i32}
                ascvec.get_buf mte3, %299, 0 {cannir.buffer.generated_sync}
                ascvec.copy_ub2l1_strided %303[%307], %302[%c0], %c4_i32, %c64_i32, %c1_i32, %c0_i32 : memref<?xbf16, 2>, memref<1xbf16, 1>
                ascvec.release_buf mte3, %299, 0 {cannir.buffer.generated_sync}
                ascvec.sync_intra_arrive mte3, %c9_i64 {cannir.sync.banks = 3 : i32}
                ascvec.sync_wait {dst_pipe = 2 : i32, event_id = 1 : i32, src_pipe = 5 : i32}
                ascvec.get_buf v, %c7_i64, 0 {cannir.buffer.generated_sync}
                ascvec.vf(%alloc_24, %alloc_27, %alloc_37 : memref<64xf32, 1>, memref<4096xbf16, 1>, memref<4096xbf16, 1>) {
                ^bb0(%arg24: memref<64xf32, 1>, %arg25: memref<4096xbf16, 1>, %arg26: memref<4096xbf16, 1>):
                  %c1_i32_68 = arith.constant 1 : i32
                  %c0_i32_69 = arith.constant 0 : i32
                  %c64_i32_70 = arith.constant 64 : i32
                  %mask, %remaining_out = ascvec.update_mask %c64_i32_70 : !ascvec.mask
                  scf.for %arg27 = %c0_i32_69 to %c64_i32_70 step %c1_i32_68  : i32 {
                    %314 = arith.muli %arg27, %c64_i32_70 overflow<nsw> : i32
                    %315 = arith.index_cast %arg27 : i32 to index
                    %316 = ascvec.loadalign_brc %arg24[%315] : memref<64xf32, 1>, index -> <f32>
                    %317 = arith.index_cast %314 : i32 to index
                    %318 = ascvec.loadalign_unpack %arg25[%317] {mode = "b16_to_b32"} : memref<4096xbf16, 1>, index -> <bf16>
                    %319 = ascvec.cast %318, %mask {rounding = 6 : i32} : <bf16> -> <f32>
                    %320 = ascvec.mul %319, %316, %mask : <f32>
                    %321 = ascvec.cast %320, %mask : <f32> -> <bf16>
                    ascvec.storealign_pack %arg26[%317], %321, %mask {mode = "b32_to_b16"} : memref<4096xbf16, 1>, index, <bf16>
                  } {cannir.loop_unroll = 4 : i64}
                } {cannir.vf_mode = "simd"}
                ascvec.release_buf v, %c7_i64, 0 {cannir.buffer.generated_sync}
                %308 = arith.cmpi eq, %301, %c0_i64 : i64
                %309 = arith.select %308, %295, %296 : i64
                %310 = ascscalar.bind_local_ptr %309 : memref<4160xbf16, 1>
                %311 = arith.select %308, %c10_i64, %c11_i64 : i64
                ascvec.get_buf v, %311, 0 {cannir.buffer.generated_sync}
                ascvec.vf(%alloc_37, %310 : memref<4096xbf16, 1>, memref<4160xbf16, 1>) {
                ^bb0(%arg24: memref<4096xbf16, 1>, %arg25: memref<4160xbf16, 1>):
                  %c1_i32_68 = arith.constant 1 : i32
                  %c0_i32_69 = arith.constant 0 : i32
                  %c65 = arith.constant 65 : index
                  %c16_i32_70 = arith.constant 16 : i32
                  %c0_71 = arith.constant 0 : index
                  %c64_i32_72 = arith.constant 64 : i32
                  scf.for %arg26 = %c0_i32_69 to %c64_i32_72 step %c1_i32_68  : i32 {
                    %mask, %remaining_out = ascvec.update_mask %c64_i32_72 elem_bits 16 : !ascvec.mask
                    %314 = arith.muli %arg26, %c64_i32_72 overflow<nsw> : i32
                    %315 = arith.index_cast %314 : i32 to index
                    %316 = ascvec.loadalign %arg24[%315] : memref<4096xbf16, 1>, index -> <bf16>
                    %317 = arith.muli %arg26, %c16_i32_70 overflow<nsw> : i32
                    %318 = arith.index_cast %317 : i32 to index
                    ascvec.storealign_strided %arg25[%318], %316, %mask, %c65, %c0_71 : memref<4160xbf16, 1>, <bf16>
                  }
                } {cannir.vf_mode = "simd"}
                ascvec.release_buf v, %311, 0 {cannir.buffer.generated_sync}
                %312 = ascscalar.bind_local_ptr %309 : memref<1xbf16, 1>
                %313 = ascscalar.bitcast_memref %alloc_62 byte_offset = %c0_i64 : memref<8192xbf16, 2> -> memref<?xbf16, 2>
                ascvec.sync_intra_wait mte3, %c8_i64 {cannir.sync.banks = 3 : i32}
                ascvec.get_buf mte3, %311, 0 {cannir.buffer.generated_sync}
                ascvec.copy_ub2l1_strided %313[%307], %312[%c0], %c4_i32, %c64_i32, %c1_i32, %c0_i32 : memref<?xbf16, 2>, memref<1xbf16, 1>
                ascvec.release_buf mte3, %311, 0 {cannir.buffer.generated_sync}
                ascvec.sync_intra_arrive mte3, %c8_i64 {cannir.sync.banks = 3 : i32}
              }
              %161 = cannir.array_load %111[%119] : <2xi64>
              %162 = arith.remsi %161, %121 : i64
              %163 = arith.cmpi ne, %162, %c0_i64 : i64
              %164 = arith.cmpi slt, %162, %c0_i64 : i64
              %165 = arith.xori %164, %125 : i1
              %166 = arith.andi %163, %165 : i1
              %167 = arith.divsi %161, %121 : i64
              %168 = arith.extui %166 : i1 to i64
              %169 = arith.subi %167, %168 : i64
              %170 = arith.muli %169, %121 : i64
              %171 = arith.subi %161, %170 : i64
              %172 = arith.addi %133, %171 : i64
              %173 = arith.remsi %169, %138 : i64
              %174 = arith.cmpi slt, %173, %c0_i64 : i64
              %175 = arith.select %174, %138, %c0_i64 : i64
              %176 = arith.addi %173, %175 : i64
              %177 = arith.remsi %161, %c2_i64 : i64
              %178 = arith.cmpi ne, %177, %c0_i64 : i64
              %179 = arith.cmpi slt, %177, %c0_i64 : i64
              %180 = arith.andi %178, %179 : i1
              %181 = arith.select %180, %c2_i64, %c0_i64 : i64
              %182 = arith.addi %177, %181 overflow<nsw> : i64
              %183 = arith.cmpi eq, %113#1, %c0_i64 : i64
              %184 = ascscalar.memref_address %alloc_33 : memref<2048xf32, 1>
              %185 = ascscalar.memref_address %alloc_34 : memref<2048xf32, 1>
              %186 = arith.select %183, %184, %185 : i64
              %187 = ascscalar.bind_local_ptr %186 : memref<2048xf32, 1>
              %188 = arith.select %183, %c39_i64, %c40_i64 : i64
              %189 = arith.cmpi eq, %188, %c40_i64 : i64
              %190 = arith.select %189, %c3_i64, %c2_i64 : i64
              ascvec.sync_intra_wait v, %190 {cannir.sync.banks = 3 : i32}
              ascvec.get_buf v, %c13_i64, 0 {cannir.buffer.generated_sync}
              ascvec.vf(%102, %187, %alloc_39 : i64, memref<2048xf32, 1>, memref<2048xbf16, 1>) {
              ^bb0(%arg24: i64, %arg25: memref<2048xf32, 1>, %arg26: memref<2048xbf16, 1>):
                %c0_i32_68 = arith.constant 0 : i32
                %c1_i32_69 = arith.constant 1 : i32
                %c32_i32_70 = arith.constant 32 : i32
                %cst = arith.constant 0.000000e+00 : f32
                %c64_i32_71 = arith.constant 64 : i32
                %mask, %remaining_out = ascvec.update_mask %c64_i32_71 : !ascvec.mask
                %269 = ascvec.duplicate_scalar %cst, %mask : f32 -> <f32>
                scf.for %arg27 = %c0_i32_68 to %c32_i32_70 step %c1_i32_69  : i32 {
                  %270 = arith.muli %arg27, %c64_i32_71 overflow<nsw> : i32
                  %271 = arith.trunci %arg24 : i64 to i32
                  %272 = arith.muli %271, %c32_i32_70 : i32
                  %273 = arith.addi %272, %arg27 : i32
                  %274 = arith.addi %273, %c1_i32_69 : i32
                  %mask_72, %remaining_out_73 = ascvec.update_mask %274 : !ascvec.mask
                  %275 = arith.index_cast %270 : i32 to index
                  %276 = ascvec.loadalign %arg25[%275] : memref<2048xf32, 1>, index -> <f32>
                  %277 = ascvec.select %276, %269, %mask_72 : <f32>
                  %278 = ascvec.cast %277, %mask : <f32> -> <bf16>
                  ascvec.storealign_pack %arg26[%275], %278, %mask {mode = "b32_to_b16"} : memref<2048xbf16, 1>, index, <bf16>
                } {cannir.loop_unroll = 4 : i64}
              } {cannir.vf_mode = "simd"}
              ascvec.release_buf v, %c13_i64, 0 {cannir.buffer.generated_sync}
              ascvec.sync_intra_arrive v, %190 {cannir.sync.banks = 3 : i32}
              %191 = cannir.extract_physical_layout %arg12 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 64, 64)>, <(-1, -1, 4096, 64, 1)>>, <<(-1, -1, -1, 64, 64)>, <(-1, -1, 4096, 64, 1)>>, identity>
              %192 = cannir.extract_stride %191 : <<(-1, -1, -1, 64, 64)>, <(-1, -1, 4096, 64, 1)>> -> <(-1, -1, 4096, 64, 1)>
              %193 = cannir.extract_dim %192[0] : !cannir.stride<(-1, -1, 4096, 64, 1)>
              %194 = arith.muli %98, %193 : i64
              %195 = cannir.extract_dim %192[1] : !cannir.stride<(-1, -1, 4096, 64, 1)>
              %196 = arith.muli %176, %195 : i64
              %197 = arith.addi %194, %196 : i64
              %198 = arith.muli %172, %c4096_i64 : i64
              %199 = arith.addi %197, %198 : i64
              %200 = arith.muli %102, %c2048_i64 : i64
              %201 = arith.addi %199, %200 : i64
              %202 = arith.muli %201, %c16_i64 : i64
              %203 = ascscalar.memref_address %alloc_11 : memref<1xbf16>
              %204 = arith.divui %202, %c8_i64 : i64
              %205 = arith.addi %203, %204 : i64
              %206 = ascscalar.bind_gm_ptr %205 : memref<1xbf16>
              %207 = ascscalar.bitcast_memref %alloc_39 byte_offset = %c0_i64 : memref<2048xbf16, 1> -> memref<?xbf16, 1>
              ascvec.get_buf mte3, %c13_i64, 0 {cannir.buffer.generated_sync}
              ascvec.copy_ub2gm %206[%c0], %207[%c0], %c1_i32, %c4096_i32, %c0_i64, %c4096_i32 l2_cache_ctl %c0_i32 : memref<1xbf16>, memref<?xbf16, 1>
              ascvec.release_buf mte3, %c13_i64, 0 {cannir.buffer.generated_sync}
              %208 = ascscalar.memref_address %alloc_29 : memref<64xf32, 1>
              %209 = ascscalar.memref_address %alloc_30 : memref<64xf32, 1>
              %210 = arith.select %183, %208, %209 : i64
              %211 = ascscalar.bind_local_ptr %210 : memref<64xf32, 1>
              %212 = arith.select %183, %c8_i64, %c9_i64 : i64
              %213 = arith.cmpi eq, %182, %c0_i64 : i64
              %214 = arith.select %213, %156, %157 : i64
              %215 = ascscalar.bind_local_ptr %214 : memref<4096xbf16, 1>
              %216 = ascscalar.memref_address %alloc_31 : memref<4096xf32, 1>
              %217 = ascscalar.memref_address %alloc_32 : memref<4096xf32, 1>
              %218 = arith.select %213, %216, %217 : i64
              %219 = ascscalar.bind_local_ptr %218 : memref<4096xf32, 1>
              ascvec.get_buf v, %212, 0 {cannir.buffer.generated_sync}
              ascvec.get_buf v, %c12_i64, 0 {cannir.buffer.generated_sync}
              ascvec.vf(%211, %215, %219, %alloc_37 : memref<64xf32, 1>, memref<4096xbf16, 1>, memref<4096xf32, 1>, memref<4096xbf16, 1>) {
              ^bb0(%arg24: memref<64xf32, 1>, %arg25: memref<4096xbf16, 1>, %arg26: memref<4096xf32, 1>, %arg27: memref<4096xbf16, 1>):
                %c1_i32_68 = arith.constant 1 : i32
                %c0_i32_69 = arith.constant 0 : i32
                %c0_70 = arith.constant 0 : index
                %c64_i32_71 = arith.constant 64 : i32
                %mask, %remaining_out = ascvec.update_mask %c64_i32_71 : !ascvec.mask
                %269 = ascvec.loadalign %arg24[%c0_70] : memref<64xf32, 1>, index -> <f32>
                scf.for %arg28 = %c0_i32_69 to %c64_i32_71 step %c1_i32_68  : i32 {
                  %270 = arith.muli %arg28, %c64_i32_71 overflow<nsw> : i32
                  %271 = arith.index_cast %270 : i32 to index
                  %272 = ascvec.loadalign_unpack %arg25[%271] {mode = "b16_to_b32"} : memref<4096xbf16, 1>, index -> <bf16>
                  %273 = ascvec.cast %272, %mask {rounding = 6 : i32} : <bf16> -> <f32>
                  %274 = ascvec.loadalign %arg26[%271] : memref<4096xf32, 1>, index -> <f32>
                  %275 = ascvec.sub %269, %274, %mask : <f32>
                  %276 = ascvec.exp %275, %mask : <f32>
                  %277 = ascvec.mul %273, %276, %mask : <f32>
                  %278 = ascvec.cast %277, %mask : <f32> -> <bf16>
                  ascvec.storealign_pack %arg27[%271], %278, %mask {mode = "b32_to_b16"} : memref<4096xbf16, 1>, index, <bf16>
                } {cannir.loop_unroll = 4 : i64}
              } {cannir.vf_mode = "simd"}
              ascvec.release_buf v, %c12_i64, 0 {cannir.buffer.generated_sync}
              ascvec.release_buf v, %212, 0 {cannir.buffer.generated_sync}
              %220 = arith.divsi %78, %c64_i64 : i64
              %221 = arith.minsi %172, %220 : i64
              %222 = arith.muli %221, %c64_i64 : i64
              %223 = arith.subi %78, %222 : i64
              %224 = arith.minsi %223, %c64_i64 : i64
              %225 = arith.cmpi sle, %172, %220 : i64
              %226 = arith.select %225, %224, %c0_i64 : i64
              %227 = cannir.extract_physical_layout %arg8 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
              %228 = cannir.extract_stride %227 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
              %229 = cannir.extract_dim %228[0] : !cannir.stride<(-1, -1, -1, 1)>
              %230 = arith.muli %97, %229 : i64
              %231 = cannir.extract_dim %228[1] : !cannir.stride<(-1, -1, -1, 1)>
              %232 = arith.muli %176, %231 : i64
              %233 = arith.addi %230, %232 : i64
              %234 = arith.muli %172, %c64_i64 : i64
              %235 = arith.addi %234, %96 : i64
              %236 = cannir.extract_dim %228[2] : !cannir.stride<(-1, -1, -1, 1)>
              %237 = arith.muli %235, %236 : i64
              %238 = arith.addi %233, %237 : i64
              %239 = arith.muli %102, %c64_i64 : i64
              %240 = arith.addi %238, %239 : i64
              %241 = arith.muli %240, %c16_i64 : i64
              %242 = ascscalar.memref_address %alloc_7 : memref<1xbf16>
              %243 = arith.divui %241, %c8_i64 : i64
              %244 = arith.addi %242, %243 : i64
              %245 = ascscalar.bind_gm_ptr %244 : memref<1xbf16>
              %246 = ascscalar.bitcast_memref %alloc_37 byte_offset = %c0_i64 : memref<4096xbf16, 1> -> memref<?xbf16, 1>
              %247 = cannir.extract_layout %arg8 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
              %248 = cannir.extract_stride %247 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
              %249 = cannir.extract_dim %248[2] : !cannir.stride<(-1, -1, -1, 1)>
              %250 = arith.index_cast %249 : i64 to index
              %251 = arith.muli %250, %c2 : index
              %252 = arith.cmpi ugt, %226, %c0_i64 : i64
              scf.if %252 {
                %269 = arith.index_cast %251 : index to i64
                ascvec.get_buf mte3, %c12_i64, 0 {cannir.buffer.generated_sync}
                %270 = arith.trunci %226 : i64 to i32
                ascvec.copy_ub2gm %245[%c0], %246[%c0], %270, %c128_i32, %269, %c128_i32 l2_cache_ctl %c0_i32 : memref<1xbf16>, memref<?xbf16, 1>
                ascvec.release_buf mte3, %c12_i64, 0 {cannir.buffer.generated_sync}
              }
              ascvec.get_buf v, %212, 0 {cannir.buffer.generated_sync}
              ascvec.vf(%211 : memref<64xf32, 1>) {
              ^bb0(%arg24: memref<64xf32, 1>):
                %c0_68 = arith.constant 0 : index
                %c64_i32_69 = arith.constant 64 : i32
                %c0_70 = arith.constant 0 : index
                %c1_71 = arith.constant 1 : index
                scf.for %arg25 = %c0_70 to %c1_71 step %c1_71 {
                  %mask, %remaining_out = ascvec.update_mask %c64_i32_69 : !ascvec.mask
                  %269 = ascvec.loadalign %arg24[%c0_68] : memref<64xf32, 1>, index -> <f32>
                  %270 = ascvec.exp %269, %mask : <f32>
                  ascvec.storealign %arg24[%c0_68], %270, %mask : memref<64xf32, 1>, index, <f32>
                }
              } {cannir.vf_mode = "simd"}
              ascvec.release_buf v, %212, 0 {cannir.buffer.generated_sync}
              %253 = cannir.extract_physical_layout %arg9 : !cannir.tensor<<gm, f32>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
              %254 = cannir.extract_stride %253 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, 128, 1)>
              %255 = cannir.extract_dim %254[0] : !cannir.stride<(-1, -1, 128, 1)>
              %256 = arith.muli %98, %255 : i64
              %257 = cannir.extract_dim %254[1] : !cannir.stride<(-1, -1, 128, 1)>
              %258 = arith.muli %176, %257 : i64
              %259 = arith.addi %256, %258 : i64
              %260 = arith.muli %172, %c128_i64 : i64
              %261 = arith.addi %259, %260 : i64
              %262 = arith.addi %261, %239 : i64
              %263 = arith.muli %262, %c32_i64 : i64
              %264 = ascscalar.memref_address %alloc_8 : memref<1xf32>
              %265 = arith.divui %263, %c8_i64 : i64
              %266 = arith.addi %264, %265 : i64
              %267 = ascscalar.bind_gm_ptr %266 : memref<1xf32>
              %268 = ascscalar.bind_local_ptr %210 : memref<1xf32, 1>
              ascvec.get_buf mte3, %212, 0 {cannir.buffer.generated_sync}
              ascvec.copy_ub2gm %267[%c0], %268[%c0], %c1_i32, %c256_i32, %c0_i64, %c256_i32 l2_cache_ctl %c0_i32 : memref<1xf32>, memref<1xf32, 1>
              ascvec.release_buf mte3, %212, 0 {cannir.buffer.generated_sync}
            }
          }
        }
        ascvec.sync_all
        ascvec.sync_block_arrive mte3, %c0_i64
        ascvec.sync_block_wait s, %c8_i64
        %36 = arith.addi %26, %c65_i32 : i32
        %37 = arith.addi %36, %17 : i32
        %38 = arith.addi %37, %17 : i32
        %39 = arith.addi %38, %c1_i32 : i32
        %40 = arith.extsi %39 : i32 to i64
        %41 = arith.addi %40, %28 : i64
        %42 = arith.index_cast %41 : i64 to index
        %43 = memref.load %alloc_15[%42] : memref<1xi32>
        %44 = arith.addi %41, %c1_i64 : i64
        %45 = arith.index_cast %44 : i64 to index
        %46 = memref.load %alloc_15[%45] : memref<1xi32>
        scf.for %arg23 = %c0_i32 to %17 step %c1_i32  : i32 {
          %49 = arith.addi %37, %arg23 : i32
          %50 = arith.index_cast %49 : i32 to index
          %51 = memref.load %alloc_15[%50] : memref<1xi32>
          %52 = arith.addi %49, %c1_i32 : i32
          %53 = arith.index_cast %52 : i32 to index
          %54 = memref.load %alloc_15[%53] : memref<1xi32>
          %55 = arith.cmpi slt, %43, %51 : i32
          %56 = arith.select %55, %51, %43 : i32
          %57 = arith.cmpi sgt, %46, %54 : i32
          %58 = arith.select %57, %54, %46 : i32
          %59 = arith.cmpi slt, %56, %58 : i32
          %60 = arith.subi %56, %51 : i32
          %61 = arith.subi %58, %51 : i32
          %62 = arith.extsi %60 : i32 to i64
          %63 = arith.extsi %61 : i32 to i64
          %64 = arith.select %59, %62, %c0_i64 : i64
          %65 = arith.select %59, %63, %c0_i64 : i64
          %66 = arith.addi %36, %arg23 : i32
          %67 = arith.index_cast %66 : i32 to index
          %68 = memref.load %alloc_15[%67] : memref<1xi32>
          %69 = arith.addi %18, %arg23 : i32
          %70 = arith.index_cast %69 : i32 to index
          %71 = memref.load %alloc_15[%70] : memref<1xi32>
          %72 = arith.addi %19, %arg23 : i32
          %73 = arith.index_cast %72 : i32 to index
          %74 = memref.load %alloc_15[%73] : memref<1xi32>
          %75 = arith.addi %20, %arg23 : i32
          %76 = arith.index_cast %75 : i32 to index
          %77 = memref.load %alloc_15[%76] : memref<1xi32>
          %78 = arith.addi %21, %arg23 : i32
          %79 = arith.index_cast %78 : i32 to index
          %80 = memref.load %alloc_15[%79] : memref<1xi32>
          %81 = arith.extsi %80 : i32 to i64
          %82 = arith.addi %23, %arg23 : i32
          %83 = arith.index_cast %82 : i32 to index
          %84 = memref.load %alloc_15[%83] : memref<1xi32>
          %85 = arith.addi %24, %arg23 : i32
          %86 = arith.index_cast %85 : i32 to index
          %87 = memref.load %alloc_15[%86] : memref<1xi32>
          %88 = arith.extsi %77 : i32 to i64
          %89 = arith.extsi %68 : i32 to i64
          %90 = arith.addi %64, %89 : i64
          %91 = arith.subi %90, %c1_i64 : i64
          %92 = arith.remsi %91, %89 : i64
          %93 = arith.cmpi ne, %92, %c0_i64 : i64
          %94 = arith.cmpi slt, %92, %c0_i64 : i64
          %95 = arith.cmpi slt, %89, %c0_i64 : i64
          %96 = arith.xori %94, %95 : i1
          %97 = arith.andi %93, %96 : i1
          %98 = arith.divsi %91, %89 : i64
          %99 = arith.extui %97 : i1 to i64
          %100 = arith.subi %98, %99 : i64
          %101 = arith.addi %65, %89 : i64
          %102 = arith.subi %101, %c1_i64 : i64
          %103 = arith.remsi %102, %89 : i64
          %104 = arith.cmpi ne, %103, %c0_i64 : i64
          %105 = arith.cmpi slt, %103, %c0_i64 : i64
          %106 = arith.xori %105, %95 : i1
          %107 = arith.andi %104, %106 : i1
          %108 = arith.divsi %102, %89 : i64
          %109 = arith.extui %107 : i1 to i64
          %110 = arith.subi %108, %109 : i64
          %111 = arith.cmpi slt, %100, %110 : i64
          scf.if %111 {
            %112 = arith.extsi %74 : i32 to i64
            %113 = ascscalar.get_subblock_id
            %114 = ascscalar.bitcast_memref %alloc_51 byte_offset = %c0_i64 : memref<64xf32, 1> -> memref<?xf32, 1>
            %115 = ascscalar.bitcast_memref %alloc_54 byte_offset = %c0_i64 : memref<4096xbf16, 1> -> memref<?xbf16, 1>
            %116 = arith.addi %84, %87 : i32
            scf.for %arg24 = %100 to %110 step %c1_i64  : i64 {
              %117 = arith.extsi %71 : i32 to i64
              %118 = arith.muli %117, %7 : i64
              %119 = arith.addi %118, %arg24 : i64
              scf.for %arg25 = %84 to %116 step %c1_i32  : i32 {
                %120 = arith.extsi %arg25 : i32 to i64
                %121 = arith.muli %arg25, %c64_i32 : i32
                %122 = arith.extsi %121 : i32 to i64
                %123 = arith.subi %81, %122 : i64
                %124 = arith.minsi %123, %c64_i64 : i64
                ascvec.get_buf v, %c20_i64, 0 {cannir.buffer.generated_sync}
                ascvec.vf(%alloc_54 : memref<4096xbf16, 1>) {
                ^bb0(%arg26: memref<4096xbf16, 1>):
                  %c1_i32_68 = arith.constant 1 : i32
                  %c32_i32_69 = arith.constant 32 : i32
                  %c0_i32_70 = arith.constant 0 : i32
                  %c128_i32_71 = arith.constant 128 : i32
                  %cst = arith.constant 0.000000e+00 : bf16
                  %140 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                  %141 = ascvec.duplicate_scalar %cst, %140 : bf16 -> <bf16>
                  scf.for %arg27 = %c0_i32_70 to %c32_i32_69 step %c1_i32_68  : i32 {
                    %142 = arith.muli %arg27, %c128_i32_71 overflow<nsw> : i32
                    %143 = arith.index_cast %142 : i32 to index
                    ascvec.storealign %arg26[%143], %141, %140 : memref<4096xbf16, 1>, index, <bf16>
                  }
                  ascvec.mem_bar vst_vld
                } {cannir.vf_mode = "simd"}
                ascvec.release_buf v, %c20_i64, 0 {cannir.buffer.generated_sync}
                %125 = arith.muli %113, %c32_i64 : i64
                %126 = arith.cmpi sgt, %124, %125 : i64
                scf.if %126 {
                  %140 = arith.divsi %81, %c64_i64 : i64
                  %141 = arith.minsi %120, %140 : i64
                  %142 = arith.muli %141, %c64_i64 : i64
                  %143 = arith.subi %81, %142 : i64
                  %144 = arith.minsi %143, %c64_i64 : i64
                  %145 = arith.cmpi sle, %120, %140 : i64
                  %146 = arith.select %145, %144, %c0_i64 : i64
                  %147 = arith.divsi %146, %c32_i64 : i64
                  %148 = arith.minsi %113, %147 : i64
                  %149 = arith.muli %148, %c32_i64 : i64
                  %150 = arith.subi %146, %149 : i64
                  %151 = arith.minsi %150, %c32_i64 : i64
                  %152 = arith.cmpi sle, %113, %147 : i64
                  %153 = arith.select %152, %151, %c0_i64 : i64
                  %154 = cannir.extract_physical_layout %arg8 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                  %155 = cannir.extract_stride %154 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                  %156 = cannir.extract_dim %155[2] : !cannir.stride<(-1, -1, -1, 1)>
                  %157 = arith.muli %156, %c2_i64 : i64
                  %158 = cannir.extract_dim %155[0] : !cannir.stride<(-1, -1, -1, 1)>
                  %159 = arith.muli %112, %158 : i64
                  %160 = arith.muli %159, %c2_i64 : i64
                  %161 = cannir.extract_dim %155[1] : !cannir.stride<(-1, -1, -1, 1)>
                  %162 = arith.muli %arg24, %161 : i64
                  %163 = arith.muli %162, %c2_i64 : i64
                  %164 = arith.addi %160, %163 : i64
                  %165 = arith.muli %120, %c64_i64 : i64
                  %166 = arith.addi %125, %165 : i64
                  %167 = arith.addi %166, %88 : i64
                  %168 = arith.muli %167, %156 : i64
                  %169 = arith.muli %168, %c2_i64 : i64
                  %170 = arith.addi %164, %169 : i64
                  ascvec.get_buf mte2, %c20_i64, 0 {cannir.buffer.generated_sync}
                  "ascvec.copy_gm2ub_plan"(%115, %alloc_7, %c0_i32, %c0_i32, %c0_i64, %170, %153, %c256_i64, %157, %c256_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = 131328 : i64, ub_capacity = 8192 : i64}> : (memref<?xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                  ascvec.release_buf mte2, %c20_i64, 0 {cannir.buffer.generated_sync}
                }
                ascvec.get_buf v, %c20_i64, 0 {cannir.buffer.generated_sync}
                ascvec.vf(%alloc_54, %alloc_55 : memref<4096xbf16, 1>, memref<4096xf32, 1>) {
                ^bb0(%arg26: memref<4096xbf16, 1>, %arg27: memref<4096xf32, 1>):
                  %c1_i32_68 = arith.constant 1 : i32
                  %c32_i32_69 = arith.constant 32 : i32
                  %c64_i32_70 = arith.constant 64 : i32
                  %c0_i32_71 = arith.constant 0 : i32
                  %c128_i32_72 = arith.constant 128 : i32
                  scf.for %arg28 = %c0_i32_71 to %c32_i32_69 step %c1_i32_68  : i32 {
                    scf.for %arg29 = %c0_i32_71 to %c128_i32_72 step %c64_i32_70  : i32 {
                      %140 = arith.subi %c128_i32_72, %arg29 overflow<nsw> : i32
                      %mask, %remaining_out = ascvec.update_mask %140 : !ascvec.mask
                      %141 = arith.muli %arg28, %c128_i32_72 overflow<nsw> : i32
                      %142 = arith.addi %141, %arg29 overflow<nsw> : i32
                      %143 = arith.index_cast %142 : i32 to index
                      %144 = ascvec.loadalign_unpack %arg26[%143] {mode = "b16_to_b32"} : memref<4096xbf16, 1>, index -> <bf16>
                      %145 = ascvec.cast %144, %mask {rounding = 6 : i32} : <bf16> -> <f32>
                      ascvec.storealign %arg27[%143], %145, %mask : memref<4096xf32, 1>, index, <f32>
                    }
                  }
                } {cannir.vf_mode = "simd"}
                ascvec.release_buf v, %c20_i64, 0 {cannir.buffer.generated_sync}
                ascvec.get_buf v, %c21_i64, 0 {cannir.buffer.generated_sync}
                ascvec.vf(%alloc_55, %alloc_56 : memref<4096xf32, 1>, memref<4096xf32, 1>) {
                ^bb0(%arg26: memref<4096xf32, 1>, %arg27: memref<4096xf32, 1>):
                  %c1_i32_68 = arith.constant 1 : i32
                  %c32_i32_69 = arith.constant 32 : i32
                  %c64_i32_70 = arith.constant 64 : i32
                  %c0_i32_71 = arith.constant 0 : i32
                  %c32 = arith.constant 32 : index
                  %c256_i32_72 = arith.constant 256 : i32
                  %c8_i32 = arith.constant 8 : i32
                  %c0_73 = arith.constant 0 : index
                  %c128_i32_74 = arith.constant 128 : i32
                  scf.for %arg28 = %c0_i32_71 to %c32_i32_69 step %c1_i32_68  : i32 {
                    %140 = scf.for %arg29 = %c0_i32_71 to %c128_i32_74 step %c64_i32_70 iter_args(%arg30 = %c128_i32_74) -> (i32)  : i32 {
                      %mask, %remaining_out = ascvec.update_mask %arg30 : !ascvec.mask
                      %141 = arith.muli %arg28, %c128_i32_74 overflow<nsw> : i32
                      %142 = arith.addi %141, %arg29 overflow<nsw> : i32
                      %143 = arith.index_cast %142 : i32 to index
                      %144 = ascvec.loadalign %arg26[%143] : memref<4096xf32, 1>, index -> <f32>
                      %145 = arith.muli %arg28, %c8_i32 overflow<nsw> : i32
                      %146 = arith.divsi %arg29, %c8_i32 : i32
                      %147 = arith.muli %146, %c256_i32_72 overflow<nsw> : i32
                      %148 = arith.addi %145, %147 overflow<nsw> : i32
                      %149 = arith.index_cast %148 : i32 to index
                      ascvec.storealign_strided %arg27[%149], %144, %mask, %c32, %c0_73 : memref<4096xf32, 1>, <f32>
                      scf.yield %remaining_out : i32
                    }
                  }
                } {cannir.vf_mode = "simd"}
                ascvec.release_buf v, %c21_i64, 0 {cannir.buffer.generated_sync}
                %127 = ascscalar.bitcast_memref %alloc_56 byte_offset = %c0_i64 : memref<4096xf32, 1> -> memref<?xf32, 1>
                %128 = ascscalar.bitcast_memref %alloc_59 byte_offset = %c0_i64 : memref<8192xf32, 2> -> memref<?xf32, 2>
                %129 = arith.index_cast %125 : i64 to index
                %130 = arith.muli %129, %c8 : index
                ascvec.sync_intra_wait mte3, %c10_i64 {cannir.sync.banks = 3 : i32}
                ascvec.get_buf mte3, %c21_i64, 0 {cannir.buffer.generated_sync}
                ascvec.copy_ub2l1_strided %128[%130], %127[%c0], %c16_i32, %c32_i32, %c0_i32, %c32_i32 : memref<?xf32, 2>, memref<?xf32, 1>
                ascvec.release_buf mte3, %c21_i64, 0 {cannir.buffer.generated_sync}
                ascvec.sync_intra_arrive mte3, %c10_i64 {cannir.sync.banks = 3 : i32}
                ascvec.get_buf v, %c20_i64, 0 {cannir.buffer.generated_sync}
                ascvec.vf(%alloc_54 : memref<4096xbf16, 1>) {
                ^bb0(%arg26: memref<4096xbf16, 1>):
                  %c1_i32_68 = arith.constant 1 : i32
                  %c32_i32_69 = arith.constant 32 : i32
                  %c0_i32_70 = arith.constant 0 : i32
                  %c128_i32_71 = arith.constant 128 : i32
                  %cst = arith.constant 0.000000e+00 : bf16
                  %140 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                  %141 = ascvec.duplicate_scalar %cst, %140 : bf16 -> <bf16>
                  scf.for %arg27 = %c0_i32_70 to %c32_i32_69 step %c1_i32_68  : i32 {
                    %142 = arith.muli %arg27, %c128_i32_71 overflow<nsw> : i32
                    %143 = arith.index_cast %142 : i32 to index
                    ascvec.storealign %arg26[%143], %141, %140 : memref<4096xbf16, 1>, index, <bf16>
                  }
                  ascvec.mem_bar vst_vld
                } {cannir.vf_mode = "simd"}
                ascvec.release_buf v, %c20_i64, 0 {cannir.buffer.generated_sync}
                scf.if %126 {
                  %140 = arith.divsi %81, %c64_i64 : i64
                  %141 = arith.minsi %120, %140 : i64
                  %142 = arith.muli %141, %c64_i64 : i64
                  %143 = arith.subi %81, %142 : i64
                  %144 = arith.minsi %143, %c64_i64 : i64
                  %145 = arith.cmpi sle, %120, %140 : i64
                  %146 = arith.select %145, %144, %c0_i64 : i64
                  %147 = arith.divsi %146, %c32_i64 : i64
                  %148 = arith.minsi %113, %147 : i64
                  %149 = arith.muli %148, %c32_i64 : i64
                  %150 = arith.subi %146, %149 : i64
                  %151 = arith.minsi %150, %c32_i64 : i64
                  %152 = arith.cmpi sle, %113, %147 : i64
                  %153 = arith.select %152, %151, %c0_i64 : i64
                  %154 = cannir.extract_physical_layout %arg13 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                  %155 = cannir.extract_stride %154 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                  %156 = cannir.extract_dim %155[2] : !cannir.stride<(-1, -1, -1, 1)>
                  %157 = arith.muli %156, %c2_i64 : i64
                  %158 = cannir.extract_dim %155[0] : !cannir.stride<(-1, -1, -1, 1)>
                  %159 = arith.muli %112, %158 : i64
                  %160 = arith.muli %159, %c2_i64 : i64
                  %161 = cannir.extract_dim %155[1] : !cannir.stride<(-1, -1, -1, 1)>
                  %162 = arith.muli %arg24, %161 : i64
                  %163 = arith.muli %162, %c2_i64 : i64
                  %164 = arith.addi %160, %163 : i64
                  %165 = arith.muli %120, %c64_i64 : i64
                  %166 = arith.addi %125, %165 : i64
                  %167 = arith.addi %166, %88 : i64
                  %168 = arith.muli %167, %156 : i64
                  %169 = arith.muli %168, %c2_i64 : i64
                  %170 = arith.addi %164, %169 : i64
                  ascvec.get_buf mte2, %c20_i64, 0 {cannir.buffer.generated_sync}
                  "ascvec.copy_gm2ub_plan"(%115, %alloc_12, %c0_i32, %c0_i32, %c0_i64, %170, %153, %c256_i64, %157, %c256_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = 131328 : i64, ub_capacity = 8192 : i64}> : (memref<?xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                  ascvec.release_buf mte2, %c20_i64, 0 {cannir.buffer.generated_sync}
                }
                ascvec.get_buf v, %c20_i64, 0 {cannir.buffer.generated_sync}
                ascvec.vf(%alloc_54, %alloc_55 : memref<4096xbf16, 1>, memref<4096xf32, 1>) {
                ^bb0(%arg26: memref<4096xbf16, 1>, %arg27: memref<4096xf32, 1>):
                  %c1_i32_68 = arith.constant 1 : i32
                  %c32_i32_69 = arith.constant 32 : i32
                  %c64_i32_70 = arith.constant 64 : i32
                  %c0_i32_71 = arith.constant 0 : i32
                  %c128_i32_72 = arith.constant 128 : i32
                  scf.for %arg28 = %c0_i32_71 to %c32_i32_69 step %c1_i32_68  : i32 {
                    scf.for %arg29 = %c0_i32_71 to %c128_i32_72 step %c64_i32_70  : i32 {
                      %140 = arith.subi %c128_i32_72, %arg29 overflow<nsw> : i32
                      %mask, %remaining_out = ascvec.update_mask %140 : !ascvec.mask
                      %141 = arith.muli %arg28, %c128_i32_72 overflow<nsw> : i32
                      %142 = arith.addi %141, %arg29 overflow<nsw> : i32
                      %143 = arith.index_cast %142 : i32 to index
                      %144 = ascvec.loadalign_unpack %arg26[%143] {mode = "b16_to_b32"} : memref<4096xbf16, 1>, index -> <bf16>
                      %145 = ascvec.cast %144, %mask {rounding = 6 : i32} : <bf16> -> <f32>
                      ascvec.storealign %arg27[%143], %145, %mask : memref<4096xf32, 1>, index, <f32>
                    }
                  }
                } {cannir.vf_mode = "simd"}
                ascvec.release_buf v, %c20_i64, 0 {cannir.buffer.generated_sync}
                ascvec.get_buf v, %c21_i64, 0 {cannir.buffer.generated_sync}
                ascvec.vf(%alloc_55, %alloc_56 : memref<4096xf32, 1>, memref<4096xf32, 1>) {
                ^bb0(%arg26: memref<4096xf32, 1>, %arg27: memref<4096xf32, 1>):
                  %c1_i32_68 = arith.constant 1 : i32
                  %c32_i32_69 = arith.constant 32 : i32
                  %c64_i32_70 = arith.constant 64 : i32
                  %c0_i32_71 = arith.constant 0 : i32
                  %c32 = arith.constant 32 : index
                  %c256_i32_72 = arith.constant 256 : i32
                  %c8_i32 = arith.constant 8 : i32
                  %c0_73 = arith.constant 0 : index
                  %c128_i32_74 = arith.constant 128 : i32
                  scf.for %arg28 = %c0_i32_71 to %c32_i32_69 step %c1_i32_68  : i32 {
                    %140 = scf.for %arg29 = %c0_i32_71 to %c128_i32_74 step %c64_i32_70 iter_args(%arg30 = %c128_i32_74) -> (i32)  : i32 {
                      %mask, %remaining_out = ascvec.update_mask %arg30 : !ascvec.mask
                      %141 = arith.muli %arg28, %c128_i32_74 overflow<nsw> : i32
                      %142 = arith.addi %141, %arg29 overflow<nsw> : i32
                      %143 = arith.index_cast %142 : i32 to index
                      %144 = ascvec.loadalign %arg26[%143] : memref<4096xf32, 1>, index -> <f32>
                      %145 = arith.muli %arg28, %c8_i32 overflow<nsw> : i32
                      %146 = arith.divsi %arg29, %c8_i32 : i32
                      %147 = arith.muli %146, %c256_i32_72 overflow<nsw> : i32
                      %148 = arith.addi %145, %147 overflow<nsw> : i32
                      %149 = arith.index_cast %148 : i32 to index
                      ascvec.storealign_strided %arg27[%149], %144, %mask, %c32, %c0_73 : memref<4096xf32, 1>, <f32>
                      scf.yield %remaining_out : i32
                    }
                  }
                } {cannir.vf_mode = "simd"}
                ascvec.release_buf v, %c21_i64, 0 {cannir.buffer.generated_sync}
                %131 = ascscalar.bitcast_memref %alloc_60 byte_offset = %c0_i64 : memref<8192xf32, 2> -> memref<?xf32, 2>
                ascvec.sync_intra_wait mte3, %c11_i64 {cannir.sync.banks = 3 : i32}
                ascvec.get_buf mte3, %c21_i64, 0 {cannir.buffer.generated_sync}
                ascvec.copy_ub2l1_strided %131[%130], %127[%c0], %c16_i32, %c32_i32, %c0_i32, %c32_i32 : memref<?xf32, 2>, memref<?xf32, 1>
                ascvec.release_buf mte3, %c21_i64, 0 {cannir.buffer.generated_sync}
                ascvec.sync_intra_arrive mte3, %c11_i64 {cannir.sync.banks = 3 : i32}
                ascvec.get_buf v, %c20_i64, 0 {cannir.buffer.generated_sync}
                ascvec.vf(%alloc_54 : memref<4096xbf16, 1>) {
                ^bb0(%arg26: memref<4096xbf16, 1>):
                  %c1_i32_68 = arith.constant 1 : i32
                  %c32_i32_69 = arith.constant 32 : i32
                  %c0_i32_70 = arith.constant 0 : i32
                  %c128_i32_71 = arith.constant 128 : i32
                  %cst = arith.constant 0.000000e+00 : bf16
                  %140 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                  %141 = ascvec.duplicate_scalar %cst, %140 : bf16 -> <bf16>
                  scf.for %arg27 = %c0_i32_70 to %c32_i32_69 step %c1_i32_68  : i32 {
                    %142 = arith.muli %arg27, %c128_i32_71 overflow<nsw> : i32
                    %143 = arith.index_cast %142 : i32 to index
                    ascvec.storealign %arg26[%143], %141, %140 : memref<4096xbf16, 1>, index, <bf16>
                  }
                  ascvec.mem_bar vst_vld
                } {cannir.vf_mode = "simd"}
                ascvec.release_buf v, %c20_i64, 0 {cannir.buffer.generated_sync}
                scf.if %126 {
                  %140 = arith.divsi %81, %c64_i64 : i64
                  %141 = arith.minsi %120, %140 : i64
                  %142 = arith.muli %141, %c64_i64 : i64
                  %143 = arith.subi %81, %142 : i64
                  %144 = arith.minsi %143, %c64_i64 : i64
                  %145 = arith.cmpi sle, %120, %140 : i64
                  %146 = arith.select %145, %144, %c0_i64 : i64
                  %147 = arith.divsi %146, %c32_i64 : i64
                  %148 = arith.minsi %113, %147 : i64
                  %149 = arith.muli %148, %c32_i64 : i64
                  %150 = arith.subi %146, %149 : i64
                  %151 = arith.minsi %150, %c32_i64 : i64
                  %152 = arith.cmpi sle, %113, %147 : i64
                  %153 = arith.select %152, %151, %c0_i64 : i64
                  %154 = cannir.extract_physical_layout %arg11 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                  %155 = cannir.extract_stride %154 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                  %156 = cannir.extract_dim %155[2] : !cannir.stride<(-1, -1, -1, 1)>
                  %157 = arith.muli %156, %c2_i64 : i64
                  %158 = cannir.extract_dim %155[0] : !cannir.stride<(-1, -1, -1, 1)>
                  %159 = arith.muli %112, %158 : i64
                  %160 = arith.muli %159, %c2_i64 : i64
                  %161 = cannir.extract_dim %155[1] : !cannir.stride<(-1, -1, -1, 1)>
                  %162 = arith.muli %arg24, %161 : i64
                  %163 = arith.muli %162, %c2_i64 : i64
                  %164 = arith.addi %160, %163 : i64
                  %165 = arith.muli %120, %c64_i64 : i64
                  %166 = arith.addi %125, %165 : i64
                  %167 = arith.addi %166, %88 : i64
                  %168 = arith.muli %167, %156 : i64
                  %169 = arith.muli %168, %c2_i64 : i64
                  %170 = arith.addi %164, %169 : i64
                  ascvec.get_buf mte2, %c20_i64, 0 {cannir.buffer.generated_sync}
                  "ascvec.copy_gm2ub_plan"(%115, %alloc_10, %c0_i32, %c0_i32, %c0_i64, %170, %153, %c256_i64, %157, %c256_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = 131328 : i64, ub_capacity = 8192 : i64}> : (memref<?xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                  ascvec.release_buf mte2, %c20_i64, 0 {cannir.buffer.generated_sync}
                }
                ascvec.get_buf v, %c20_i64, 0 {cannir.buffer.generated_sync}
                ascvec.vf(%alloc_54, %alloc_55 : memref<4096xbf16, 1>, memref<4096xf32, 1>) {
                ^bb0(%arg26: memref<4096xbf16, 1>, %arg27: memref<4096xf32, 1>):
                  %c1_i32_68 = arith.constant 1 : i32
                  %c32_i32_69 = arith.constant 32 : i32
                  %c64_i32_70 = arith.constant 64 : i32
                  %c0_i32_71 = arith.constant 0 : i32
                  %c128_i32_72 = arith.constant 128 : i32
                  scf.for %arg28 = %c0_i32_71 to %c32_i32_69 step %c1_i32_68  : i32 {
                    scf.for %arg29 = %c0_i32_71 to %c128_i32_72 step %c64_i32_70  : i32 {
                      %140 = arith.subi %c128_i32_72, %arg29 overflow<nsw> : i32
                      %mask, %remaining_out = ascvec.update_mask %140 : !ascvec.mask
                      %141 = arith.muli %arg28, %c128_i32_72 overflow<nsw> : i32
                      %142 = arith.addi %141, %arg29 overflow<nsw> : i32
                      %143 = arith.index_cast %142 : i32 to index
                      %144 = ascvec.loadalign_unpack %arg26[%143] {mode = "b16_to_b32"} : memref<4096xbf16, 1>, index -> <bf16>
                      %145 = ascvec.cast %144, %mask {rounding = 6 : i32} : <bf16> -> <f32>
                      ascvec.storealign %arg27[%143], %145, %mask : memref<4096xf32, 1>, index, <f32>
                    }
                  }
                } {cannir.vf_mode = "simd"}
                ascvec.release_buf v, %c20_i64, 0 {cannir.buffer.generated_sync}
                ascvec.get_buf v, %c21_i64, 0 {cannir.buffer.generated_sync}
                ascvec.vf(%alloc_55, %alloc_56 : memref<4096xf32, 1>, memref<4096xf32, 1>) {
                ^bb0(%arg26: memref<4096xf32, 1>, %arg27: memref<4096xf32, 1>):
                  %c1_i32_68 = arith.constant 1 : i32
                  %c32_i32_69 = arith.constant 32 : i32
                  %c64_i32_70 = arith.constant 64 : i32
                  %c0_i32_71 = arith.constant 0 : i32
                  %c32 = arith.constant 32 : index
                  %c256_i32_72 = arith.constant 256 : i32
                  %c8_i32 = arith.constant 8 : i32
                  %c0_73 = arith.constant 0 : index
                  %c128_i32_74 = arith.constant 128 : i32
                  scf.for %arg28 = %c0_i32_71 to %c32_i32_69 step %c1_i32_68  : i32 {
                    %140 = scf.for %arg29 = %c0_i32_71 to %c128_i32_74 step %c64_i32_70 iter_args(%arg30 = %c128_i32_74) -> (i32)  : i32 {
                      %mask, %remaining_out = ascvec.update_mask %arg30 : !ascvec.mask
                      %141 = arith.muli %arg28, %c128_i32_74 overflow<nsw> : i32
                      %142 = arith.addi %141, %arg29 overflow<nsw> : i32
                      %143 = arith.index_cast %142 : i32 to index
                      %144 = ascvec.loadalign %arg26[%143] : memref<4096xf32, 1>, index -> <f32>
                      %145 = arith.muli %arg28, %c8_i32 overflow<nsw> : i32
                      %146 = arith.divsi %arg29, %c8_i32 : i32
                      %147 = arith.muli %146, %c256_i32_72 overflow<nsw> : i32
                      %148 = arith.addi %145, %147 overflow<nsw> : i32
                      %149 = arith.index_cast %148 : i32 to index
                      ascvec.storealign_strided %arg27[%149], %144, %mask, %c32, %c0_73 : memref<4096xf32, 1>, <f32>
                      scf.yield %remaining_out : i32
                    }
                  }
                } {cannir.vf_mode = "simd"}
                ascvec.release_buf v, %c21_i64, 0 {cannir.buffer.generated_sync}
                %132 = ascscalar.bitcast_memref %alloc_61 byte_offset = %c0_i64 : memref<8192xf32, 2> -> memref<?xf32, 2>
                ascvec.sync_intra_wait mte3, %c12_i64 {cannir.sync.banks = 3 : i32}
                ascvec.get_buf mte3, %c21_i64, 0 {cannir.buffer.generated_sync}
                ascvec.copy_ub2l1_strided %132[%130], %127[%c0], %c16_i32, %c32_i32, %c0_i32, %c32_i32 : memref<?xf32, 2>, memref<?xf32, 1>
                ascvec.release_buf mte3, %c21_i64, 0 {cannir.buffer.generated_sync}
                ascvec.sync_intra_arrive mte3, %c12_i64 {cannir.sync.banks = 3 : i32}
                %133 = arith.cmpi eq, %arg25, %84 : i32
                %134 = arith.addi %arg25, %c1_i32 : i32
                %135 = arith.cmpi eq, %134, %116 : i32
                %136 = arith.cmpi eq, %84, %c0_i32 : i32
                %137 = arith.andi %136, %133 : i1
                %138 = arith.select %137, %135, %false : i1
                scf.if %138 {
                  %140 = arith.cmpi eq, %113, %c0_i64 : i64
                  scf.if %140 {
                    %141 = cannir.extract_layout %arg9 : !cannir.tensor<<gm, f32>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                    %142 = cannir.extract_shape %141 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, -1, 128)>
                    %143 = cannir.extract_dim %142[2] : !cannir.shape<(-1, -1, -1, 128)>
                    %144 = arith.minsi %120, %143 : i64
                    %145 = arith.subi %143, %144 : i64
                    %146 = arith.minsi %145, %c1_i64 : i64
                    %147 = arith.cmpi sle, %120, %143 : i64
                    %148 = arith.select %147, %146, %c0_i64 : i64
                    %149 = cannir.extract_physical_layout %arg9 : !cannir.tensor<<gm, f32>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                    %150 = cannir.extract_stride %149 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, 128, 1)>
                    %151 = cannir.extract_dim %150[0] : !cannir.stride<(-1, -1, 128, 1)>
                    %152 = arith.muli %117, %151 : i64
                    %153 = arith.muli %152, %c4_i64 : i64
                    %154 = cannir.extract_dim %150[1] : !cannir.stride<(-1, -1, 128, 1)>
                    %155 = arith.muli %arg24, %154 : i64
                    %156 = arith.muli %155, %c4_i64 : i64
                    %157 = arith.addi %153, %156 : i64
                    %158 = arith.muli %120, %c512_i64 : i64
                    %159 = arith.addi %157, %158 : i64
                    ascvec.get_buf mte2, %c17_i64, 0 {cannir.buffer.generated_sync}
                    "ascvec.copy_gm2ub_plan"(%114, %alloc_8, %c0_i32, %c0_i32, %c0_i64, %159, %148, %c256_i64, %c512_i64, %c256_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 65536 : i64, ub_capacity = 256 : i64}> : (memref<?xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                    ascvec.release_buf mte2, %c17_i64, 0 {cannir.buffer.generated_sync}
                    ascvec.sync_notify {dst_pipe = 2 : i32, event_id = 0 : i32, src_pipe = 5 : i32}
                    ascvec.sync_wait {dst_pipe = 2 : i32, event_id = 0 : i32, src_pipe = 5 : i32}
                    ascvec.sync_intra_wait v, %c14_i64 {cannir.sync.banks = 3 : i32}
                    ascvec.get_buf v, %c17_i64, 0 {cannir.buffer.generated_sync}
                    ascvec.vf(%alloc_49, %alloc_51 : memref<8192xf32, 1>, memref<64xf32, 1>) {
                    ^bb0(%arg26: memref<8192xf32, 1>, %arg27: memref<64xf32, 1>):
                      %c64_i32_68 = arith.constant 64 : i32
                      %c0_i32_69 = arith.constant 0 : i32
                      %c1_i32_70 = arith.constant 1 : i32
                      %c128_i32_71 = arith.constant 128 : i32
                      %186 = ascvec.create_mask : !ascvec.mask
                      scf.for %arg28 = %c0_i32_69 to %c64_i32_68 step %c1_i32_70  : i32 {
                        %187 = arith.muli %arg28, %c128_i32_71 overflow<nsw> : i32
                        %188 = arith.index_cast %187 : i32 to index
                        %189 = ascvec.loadalign %arg26[%188] : memref<8192xf32, 1>, index -> <f32>
                        %190 = arith.index_cast %arg28 : i32 to index
                        %191 = ascvec.loadalign_brc %arg27[%190] : memref<64xf32, 1>, index -> <f32>
                        %192 = ascvec.add %189, %191, %186 : <f32>
                        %193 = arith.addi %arg28, %c1_i32_70 overflow<nsw> : i32
                        %mask, %remaining_out = ascvec.update_mask %193 : !ascvec.mask
                        %mask_72, %remaining_out_73 = ascvec.update_mask %arg28 : !ascvec.mask
                        %194 = ascvec.select %192, %189, %mask : <f32>
                        %195 = ascvec.select %189, %194, %mask_72 : <f32>
                        ascvec.storealign %arg26[%188], %195, %186 : memref<8192xf32, 1>, index, <f32>
                      }
                      ascvec.mem_bar vst_vld
                    } {cannir.vf_mode = "simd"}
                    ascvec.release_buf v, %c17_i64, 0 {cannir.buffer.generated_sync}
                    ascvec.get_buf v, %c18_i64, 0 {cannir.buffer.generated_sync}
                    ascvec.vf(%alloc_49, %alloc_52 : memref<8192xf32, 1>, memref<8192xf32, 1>) {
                    ^bb0(%arg26: memref<8192xf32, 1>, %arg27: memref<8192xf32, 1>):
                      %c1_i32_68 = arith.constant 1 : i32
                      %c64_i32_69 = arith.constant 64 : i32
                      %c0_i32_70 = arith.constant 0 : i32
                      %c128_i32_71 = arith.constant 128 : i32
                      %c32_i32_72 = arith.constant 32 : i32
                      %mask, %remaining_out = ascvec.update_mask %c32_i32_72 : !ascvec.mask
                      scf.for %arg28 = %c0_i32_70 to %c64_i32_69 step %c1_i32_68  : i32 {
                        %186 = arith.muli %arg28, %c128_i32_71 overflow<nsw> : i32
                        scf.for %arg29 = %c0_i32_70 to %c128_i32_71 step %c32_i32_72  : i32 {
                          %187 = arith.addi %186, %arg29 overflow<nsw> : i32
                          %188 = arith.index_cast %187 : i32 to index
                          %189 = ascvec.loadalign %arg26[%188] : memref<8192xf32, 1>, index -> <f32>
                          ascvec.storealign %arg27[%188], %189, %mask : memref<8192xf32, 1>, index, <f32>
                        } {cannir.loop_unroll = 4 : i64}
                      } {cannir.loop_unroll = 4 : i64}
                      ascvec.mem_bar vst_vld
                    } {cannir.vf_mode = "simd"}
                    ascvec.release_buf v, %c18_i64, 0 {cannir.buffer.generated_sync}
                    ascvec.sync_intra_arrive v, %c14_i64 {cannir.sync.banks = 3 : i32}
                    %160 = cannir.extract_layout %arg14 : !cannir.tensor<<gm, f32>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, identity>
                    %161 = cannir.extract_shape %160 : <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>> -> <(-1, -1, 128, 128)>
                    %162 = cannir.extract_dim %161[1] : !cannir.shape<(-1, -1, 128, 128)>
                    %163 = arith.remsi %119, %162 : i64
                    %164 = arith.cmpi ne, %163, %c0_i64 : i64
                    %165 = arith.cmpi slt, %163, %c0_i64 : i64
                    %166 = arith.cmpi slt, %162, %c0_i64 : i64
                    %167 = arith.xori %165, %166 : i1
                    %168 = arith.andi %164, %167 : i1
                    %169 = arith.divsi %119, %162 : i64
                    %170 = arith.extui %168 : i1 to i64
                    %171 = arith.subi %169, %170 : i64
                    %172 = arith.muli %171, %162 : i64
                    %173 = arith.subi %119, %172 : i64
                    %174 = cannir.extract_physical_layout %arg14 : !cannir.tensor<<gm, f32>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, identity>
                    %175 = cannir.extract_stride %174 : <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>> -> <(-1, 16384, 128, 1)>
                    %176 = cannir.extract_dim %175[0] : !cannir.stride<(-1, 16384, 128, 1)>
                    %177 = arith.muli %171, %176 : i64
                    %178 = arith.muli %173, %c16384_i64 : i64
                    %179 = arith.addi %177, %178 : i64
                    %180 = arith.muli %179, %c32_i64 : i64
                    %181 = ascscalar.memref_address %alloc_13 : memref<1xf32>
                    %182 = arith.divui %180, %c8_i64 : i64
                    %183 = arith.addi %181, %182 : i64
                    %184 = ascscalar.bind_gm_ptr %183 : memref<1xf32>
                    %185 = ascscalar.bitcast_memref %alloc_52 byte_offset = %c0_i64 : memref<8192xf32, 1> -> memref<?xf32, 1>
                    ascvec.get_buf mte3, %c18_i64, 0 {cannir.buffer.generated_sync}
                    ascvec.copy_ub2gm %184[%c0], %185[%c0], %c1_i32, %c32768_i32, %c0_i64, %c32768_i32 l2_cache_ctl %c0_i32 : memref<1xf32>, memref<?xf32, 1>
                    ascvec.release_buf mte3, %c18_i64, 0 {cannir.buffer.generated_sync}
                  } else {
                    %141 = cannir.extract_layout %arg9 : !cannir.tensor<<gm, f32>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                    %142 = cannir.extract_shape %141 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, -1, 128)>
                    %143 = cannir.extract_dim %142[2] : !cannir.shape<(-1, -1, -1, 128)>
                    %144 = arith.minsi %120, %143 : i64
                    %145 = arith.subi %143, %144 : i64
                    %146 = arith.minsi %145, %c1_i64 : i64
                    %147 = arith.cmpi sle, %120, %143 : i64
                    %148 = arith.select %147, %146, %c0_i64 : i64
                    %149 = cannir.extract_physical_layout %arg9 : !cannir.tensor<<gm, f32>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                    %150 = cannir.extract_stride %149 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, 128, 1)>
                    %151 = cannir.extract_dim %150[0] : !cannir.stride<(-1, -1, 128, 1)>
                    %152 = arith.muli %117, %151 : i64
                    %153 = arith.muli %152, %c4_i64 : i64
                    %154 = cannir.extract_dim %150[1] : !cannir.stride<(-1, -1, 128, 1)>
                    %155 = arith.muli %arg24, %154 : i64
                    %156 = arith.muli %155, %c4_i64 : i64
                    %157 = arith.addi %153, %156 : i64
                    %158 = arith.muli %120, %c512_i64 : i64
                    %159 = arith.addi %157, %158 : i64
                    %160 = arith.addi %159, %c256_i64 : i64
                    ascvec.get_buf mte2, %c17_i64, 0 {cannir.buffer.generated_sync}
                    "ascvec.copy_gm2ub_plan"(%114, %alloc_8, %c0_i32, %c0_i32, %c0_i64, %160, %148, %c256_i64, %c512_i64, %c256_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 65536 : i64, ub_capacity = 256 : i64}> : (memref<?xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                    ascvec.release_buf mte2, %c17_i64, 0 {cannir.buffer.generated_sync}
                    ascvec.sync_notify {dst_pipe = 2 : i32, event_id = 0 : i32, src_pipe = 5 : i32}
                    ascvec.sync_wait {dst_pipe = 2 : i32, event_id = 0 : i32, src_pipe = 5 : i32}
                    ascvec.sync_intra_wait v, %c14_i64 {cannir.sync.banks = 3 : i32}
                    ascvec.get_buf v, %c17_i64, 0 {cannir.buffer.generated_sync}
                    ascvec.vf(%alloc_49, %alloc_51 : memref<8192xf32, 1>, memref<64xf32, 1>) {
                    ^bb0(%arg26: memref<8192xf32, 1>, %arg27: memref<64xf32, 1>):
                      %c0_i32_68 = arith.constant 0 : i32
                      %c1_i32_69 = arith.constant 1 : i32
                      %c64_i32_70 = arith.constant 64 : i32
                      %c128_i32_71 = arith.constant 128 : i32
                      %188 = ascvec.create_mask : !ascvec.mask
                      scf.for %arg28 = %c0_i32_68 to %c64_i32_70 step %c1_i32_69  : i32 {
                        %189 = arith.muli %arg28, %c128_i32_71 overflow<nsw> : i32
                        %190 = arith.addi %189, %c64_i32_70 overflow<nsw> : i32
                        %191 = arith.index_cast %190 : i32 to index
                        %192 = ascvec.loadalign %arg26[%191] : memref<8192xf32, 1>, index -> <f32>
                        %193 = arith.index_cast %arg28 : i32 to index
                        %194 = ascvec.loadalign_brc %arg27[%193] : memref<64xf32, 1>, index -> <f32>
                        %195 = ascvec.add %192, %194, %188 : <f32>
                        %196 = arith.addi %arg28, %c1_i32_69 overflow<nsw> : i32
                        %mask, %remaining_out = ascvec.update_mask %196 : !ascvec.mask
                        %mask_72, %remaining_out_73 = ascvec.update_mask %arg28 : !ascvec.mask
                        %197 = ascvec.select %195, %192, %mask : <f32>
                        %198 = ascvec.select %192, %197, %mask_72 : <f32>
                        ascvec.storealign %arg26[%191], %198, %188 : memref<8192xf32, 1>, index, <f32>
                      }
                      ascvec.mem_bar vst_vld
                    } {cannir.vf_mode = "simd"}
                    ascvec.release_buf v, %c17_i64, 0 {cannir.buffer.generated_sync}
                    ascvec.get_buf v, %c18_i64, 0 {cannir.buffer.generated_sync}
                    ascvec.vf(%alloc_49, %alloc_52 : memref<8192xf32, 1>, memref<8192xf32, 1>) {
                    ^bb0(%arg26: memref<8192xf32, 1>, %arg27: memref<8192xf32, 1>):
                      %c1_i32_68 = arith.constant 1 : i32
                      %c64_i32_69 = arith.constant 64 : i32
                      %c0_i32_70 = arith.constant 0 : i32
                      %c128_i32_71 = arith.constant 128 : i32
                      %c32_i32_72 = arith.constant 32 : i32
                      %mask, %remaining_out = ascvec.update_mask %c32_i32_72 : !ascvec.mask
                      scf.for %arg28 = %c0_i32_70 to %c64_i32_69 step %c1_i32_68  : i32 {
                        %188 = arith.muli %arg28, %c128_i32_71 overflow<nsw> : i32
                        scf.for %arg29 = %c0_i32_70 to %c128_i32_71 step %c32_i32_72  : i32 {
                          %189 = arith.addi %188, %arg29 overflow<nsw> : i32
                          %190 = arith.index_cast %189 : i32 to index
                          %191 = ascvec.loadalign %arg26[%190] : memref<8192xf32, 1>, index -> <f32>
                          ascvec.storealign %arg27[%190], %191, %mask : memref<8192xf32, 1>, index, <f32>
                        } {cannir.loop_unroll = 4 : i64}
                      } {cannir.loop_unroll = 4 : i64}
                      ascvec.mem_bar vst_vld
                    } {cannir.vf_mode = "simd"}
                    ascvec.release_buf v, %c18_i64, 0 {cannir.buffer.generated_sync}
                    ascvec.sync_intra_arrive v, %c14_i64 {cannir.sync.banks = 3 : i32}
                    %161 = cannir.extract_layout %arg14 : !cannir.tensor<<gm, f32>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, identity>
                    %162 = cannir.extract_shape %161 : <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>> -> <(-1, -1, 128, 128)>
                    %163 = cannir.extract_dim %162[1] : !cannir.shape<(-1, -1, 128, 128)>
                    %164 = arith.remsi %119, %163 : i64
                    %165 = arith.cmpi ne, %164, %c0_i64 : i64
                    %166 = arith.cmpi slt, %164, %c0_i64 : i64
                    %167 = arith.cmpi slt, %163, %c0_i64 : i64
                    %168 = arith.xori %166, %167 : i1
                    %169 = arith.andi %165, %168 : i1
                    %170 = arith.divsi %119, %163 : i64
                    %171 = arith.extui %169 : i1 to i64
                    %172 = arith.subi %170, %171 : i64
                    %173 = arith.muli %172, %163 : i64
                    %174 = arith.subi %119, %173 : i64
                    %175 = cannir.extract_physical_layout %arg14 : !cannir.tensor<<gm, f32>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, identity>
                    %176 = cannir.extract_stride %175 : <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>> -> <(-1, 16384, 128, 1)>
                    %177 = cannir.extract_dim %176[0] : !cannir.stride<(-1, 16384, 128, 1)>
                    %178 = arith.muli %172, %177 : i64
                    %179 = arith.muli %174, %c16384_i64 : i64
                    %180 = arith.addi %178, %179 : i64
                    %181 = arith.addi %180, %c8192_i64 : i64
                    %182 = arith.muli %181, %c32_i64 : i64
                    %183 = ascscalar.memref_address %alloc_13 : memref<1xf32>
                    %184 = arith.divui %182, %c8_i64 : i64
                    %185 = arith.addi %183, %184 : i64
                    %186 = ascscalar.bind_gm_ptr %185 : memref<1xf32>
                    %187 = ascscalar.bitcast_memref %alloc_52 byte_offset = %c0_i64 : memref<8192xf32, 1> -> memref<?xf32, 1>
                    ascvec.get_buf mte3, %c18_i64, 0 {cannir.buffer.generated_sync}
                    ascvec.copy_ub2gm %186[%c0], %187[%c0], %c1_i32, %c32768_i32, %c0_i64, %c32768_i32 l2_cache_ctl %c0_i32 : memref<1xf32>, memref<?xf32, 1>
                    ascvec.release_buf mte3, %c18_i64, 0 {cannir.buffer.generated_sync}
                  }
                } else {
                  %140 = arith.select %136, %133, %false : i1
                  scf.if %140 {
                    %141 = arith.cmpi eq, %113, %c0_i64 : i64
                    scf.if %141 {
                      %142 = cannir.extract_layout %arg9 : !cannir.tensor<<gm, f32>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                      %143 = cannir.extract_shape %142 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, -1, 128)>
                      %144 = cannir.extract_dim %143[2] : !cannir.shape<(-1, -1, -1, 128)>
                      %145 = arith.minsi %120, %144 : i64
                      %146 = arith.subi %144, %145 : i64
                      %147 = arith.minsi %146, %c1_i64 : i64
                      %148 = arith.cmpi sle, %120, %144 : i64
                      %149 = arith.select %148, %147, %c0_i64 : i64
                      %150 = cannir.extract_physical_layout %arg9 : !cannir.tensor<<gm, f32>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                      %151 = cannir.extract_stride %150 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, 128, 1)>
                      %152 = cannir.extract_dim %151[0] : !cannir.stride<(-1, -1, 128, 1)>
                      %153 = arith.muli %117, %152 : i64
                      %154 = arith.muli %153, %c4_i64 : i64
                      %155 = cannir.extract_dim %151[1] : !cannir.stride<(-1, -1, 128, 1)>
                      %156 = arith.muli %arg24, %155 : i64
                      %157 = arith.muli %156, %c4_i64 : i64
                      %158 = arith.addi %154, %157 : i64
                      %159 = arith.muli %120, %c512_i64 : i64
                      %160 = arith.addi %158, %159 : i64
                      ascvec.get_buf mte2, %c17_i64, 0 {cannir.buffer.generated_sync}
                      "ascvec.copy_gm2ub_plan"(%114, %alloc_8, %c0_i32, %c0_i32, %c0_i64, %160, %149, %c256_i64, %c512_i64, %c256_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 65536 : i64, ub_capacity = 256 : i64}> : (memref<?xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                      ascvec.release_buf mte2, %c17_i64, 0 {cannir.buffer.generated_sync}
                      ascvec.sync_notify {dst_pipe = 2 : i32, event_id = 0 : i32, src_pipe = 5 : i32}
                      ascvec.sync_wait {dst_pipe = 2 : i32, event_id = 0 : i32, src_pipe = 5 : i32}
                      ascvec.sync_intra_wait v, %c14_i64 {cannir.sync.banks = 3 : i32}
                      ascvec.get_buf v, %c17_i64, 0 {cannir.buffer.generated_sync}
                      ascvec.vf(%alloc_49, %alloc_51 : memref<8192xf32, 1>, memref<64xf32, 1>) {
                      ^bb0(%arg26: memref<8192xf32, 1>, %arg27: memref<64xf32, 1>):
                        %c64_i32_68 = arith.constant 64 : i32
                        %c0_i32_69 = arith.constant 0 : i32
                        %c1_i32_70 = arith.constant 1 : i32
                        %c128_i32_71 = arith.constant 128 : i32
                        %163 = ascvec.create_mask : !ascvec.mask
                        scf.for %arg28 = %c0_i32_69 to %c64_i32_68 step %c1_i32_70  : i32 {
                          %164 = arith.muli %arg28, %c128_i32_71 overflow<nsw> : i32
                          %165 = arith.index_cast %164 : i32 to index
                          %166 = ascvec.loadalign %arg26[%165] : memref<8192xf32, 1>, index -> <f32>
                          %167 = arith.index_cast %arg28 : i32 to index
                          %168 = ascvec.loadalign_brc %arg27[%167] : memref<64xf32, 1>, index -> <f32>
                          %169 = ascvec.add %166, %168, %163 : <f32>
                          %170 = arith.addi %arg28, %c1_i32_70 overflow<nsw> : i32
                          %mask, %remaining_out = ascvec.update_mask %170 : !ascvec.mask
                          %mask_72, %remaining_out_73 = ascvec.update_mask %arg28 : !ascvec.mask
                          %171 = ascvec.select %169, %166, %mask : <f32>
                          %172 = ascvec.select %166, %171, %mask_72 : <f32>
                          ascvec.storealign %arg26[%165], %172, %163 : memref<8192xf32, 1>, index, <f32>
                        }
                        ascvec.mem_bar vst_vld
                      } {cannir.vf_mode = "simd"}
                      ascvec.release_buf v, %c17_i64, 0 {cannir.buffer.generated_sync}
                      ascvec.get_buf v, %c19_i64, 0 {cannir.buffer.generated_sync}
                      ascvec.vf(%alloc_49, %alloc_53 : memref<8192xf32, 1>, memref<8192xf32, 1>) {
                      ^bb0(%arg26: memref<8192xf32, 1>, %arg27: memref<8192xf32, 1>):
                        %c1_i32_68 = arith.constant 1 : i32
                        %c64_i32_69 = arith.constant 64 : i32
                        %c0_i32_70 = arith.constant 0 : i32
                        %c64 = arith.constant 64 : index
                        %c512_i32 = arith.constant 512 : i32
                        %c8_i32 = arith.constant 8 : i32
                        %c0_71 = arith.constant 0 : index
                        %c128_i32_72 = arith.constant 128 : i32
                        scf.for %arg28 = %c0_i32_70 to %c64_i32_69 step %c1_i32_68  : i32 {
                          %163 = scf.for %arg29 = %c0_i32_70 to %c128_i32_72 step %c64_i32_69 iter_args(%arg30 = %c128_i32_72) -> (i32)  : i32 {
                            %mask, %remaining_out = ascvec.update_mask %arg30 : !ascvec.mask
                            %164 = arith.muli %arg28, %c128_i32_72 overflow<nsw> : i32
                            %165 = arith.addi %164, %arg29 overflow<nsw> : i32
                            %166 = arith.index_cast %165 : i32 to index
                            %167 = ascvec.loadalign %arg26[%166] : memref<8192xf32, 1>, index -> <f32>
                            %168 = arith.muli %arg28, %c8_i32 overflow<nsw> : i32
                            %169 = arith.divsi %arg29, %c8_i32 : i32
                            %170 = arith.muli %169, %c512_i32 overflow<nsw> : i32
                            %171 = arith.addi %168, %170 overflow<nsw> : i32
                            %172 = arith.index_cast %171 : i32 to index
                            ascvec.storealign_strided %arg27[%172], %167, %mask, %c64, %c0_71 : memref<8192xf32, 1>, <f32>
                            scf.yield %remaining_out : i32
                          }
                        }
                      } {cannir.vf_mode = "simd"}
                      ascvec.release_buf v, %c19_i64, 0 {cannir.buffer.generated_sync}
                      ascvec.sync_intra_arrive v, %c14_i64 {cannir.sync.banks = 3 : i32}
                      %161 = ascscalar.bitcast_memref %alloc_53 byte_offset = %c0_i64 : memref<8192xf32, 1> -> memref<?xf32, 1>
                      %162 = ascscalar.bitcast_memref %alloc_58 byte_offset = %c0_i64 : memref<16384xf32, 2> -> memref<?xf32, 2>
                      ascvec.sync_intra_wait mte3, %c0_i64 {cannir.sync.banks = 3 : i32}
                      ascvec.get_buf mte3, %c19_i64, 0 {cannir.buffer.generated_sync}
                      ascvec.copy_ub2l1_strided %162[%c0], %161[%c0], %c16_i32, %c64_i32, %c0_i32, %c64_i32 : memref<?xf32, 2>, memref<?xf32, 1>
                      ascvec.release_buf mte3, %c19_i64, 0 {cannir.buffer.generated_sync}
                      ascvec.sync_intra_arrive mte3, %c0_i64 {cannir.sync.banks = 3 : i32}
                    } else {
                      %142 = cannir.extract_layout %arg9 : !cannir.tensor<<gm, f32>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                      %143 = cannir.extract_shape %142 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, -1, 128)>
                      %144 = cannir.extract_dim %143[2] : !cannir.shape<(-1, -1, -1, 128)>
                      %145 = arith.minsi %120, %144 : i64
                      %146 = arith.subi %144, %145 : i64
                      %147 = arith.minsi %146, %c1_i64 : i64
                      %148 = arith.cmpi sle, %120, %144 : i64
                      %149 = arith.select %148, %147, %c0_i64 : i64
                      %150 = cannir.extract_physical_layout %arg9 : !cannir.tensor<<gm, f32>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                      %151 = cannir.extract_stride %150 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, 128, 1)>
                      %152 = cannir.extract_dim %151[0] : !cannir.stride<(-1, -1, 128, 1)>
                      %153 = arith.muli %117, %152 : i64
                      %154 = arith.muli %153, %c4_i64 : i64
                      %155 = cannir.extract_dim %151[1] : !cannir.stride<(-1, -1, 128, 1)>
                      %156 = arith.muli %arg24, %155 : i64
                      %157 = arith.muli %156, %c4_i64 : i64
                      %158 = arith.addi %154, %157 : i64
                      %159 = arith.muli %120, %c512_i64 : i64
                      %160 = arith.addi %158, %159 : i64
                      %161 = arith.addi %160, %c256_i64 : i64
                      ascvec.get_buf mte2, %c17_i64, 0 {cannir.buffer.generated_sync}
                      "ascvec.copy_gm2ub_plan"(%114, %alloc_8, %c0_i32, %c0_i32, %c0_i64, %161, %149, %c256_i64, %c512_i64, %c256_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 65536 : i64, ub_capacity = 256 : i64}> : (memref<?xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                      ascvec.release_buf mte2, %c17_i64, 0 {cannir.buffer.generated_sync}
                      ascvec.sync_notify {dst_pipe = 2 : i32, event_id = 0 : i32, src_pipe = 5 : i32}
                      ascvec.sync_wait {dst_pipe = 2 : i32, event_id = 0 : i32, src_pipe = 5 : i32}
                      ascvec.sync_intra_wait v, %c14_i64 {cannir.sync.banks = 3 : i32}
                      ascvec.get_buf v, %c17_i64, 0 {cannir.buffer.generated_sync}
                      ascvec.vf(%alloc_49, %alloc_51 : memref<8192xf32, 1>, memref<64xf32, 1>) {
                      ^bb0(%arg26: memref<8192xf32, 1>, %arg27: memref<64xf32, 1>):
                        %c0_i32_68 = arith.constant 0 : i32
                        %c1_i32_69 = arith.constant 1 : i32
                        %c64_i32_70 = arith.constant 64 : i32
                        %c128_i32_71 = arith.constant 128 : i32
                        %164 = ascvec.create_mask : !ascvec.mask
                        scf.for %arg28 = %c0_i32_68 to %c64_i32_70 step %c1_i32_69  : i32 {
                          %165 = arith.muli %arg28, %c128_i32_71 overflow<nsw> : i32
                          %166 = arith.addi %165, %c64_i32_70 overflow<nsw> : i32
                          %167 = arith.index_cast %166 : i32 to index
                          %168 = ascvec.loadalign %arg26[%167] : memref<8192xf32, 1>, index -> <f32>
                          %169 = arith.index_cast %arg28 : i32 to index
                          %170 = ascvec.loadalign_brc %arg27[%169] : memref<64xf32, 1>, index -> <f32>
                          %171 = ascvec.add %168, %170, %164 : <f32>
                          %172 = arith.addi %arg28, %c1_i32_69 overflow<nsw> : i32
                          %mask, %remaining_out = ascvec.update_mask %172 : !ascvec.mask
                          %mask_72, %remaining_out_73 = ascvec.update_mask %arg28 : !ascvec.mask
                          %173 = ascvec.select %171, %168, %mask : <f32>
                          %174 = ascvec.select %168, %173, %mask_72 : <f32>
                          ascvec.storealign %arg26[%167], %174, %164 : memref<8192xf32, 1>, index, <f32>
                        }
                        ascvec.mem_bar vst_vld
                      } {cannir.vf_mode = "simd"}
                      ascvec.release_buf v, %c17_i64, 0 {cannir.buffer.generated_sync}
                      ascvec.get_buf v, %c19_i64, 0 {cannir.buffer.generated_sync}
                      ascvec.vf(%alloc_49, %alloc_53 : memref<8192xf32, 1>, memref<8192xf32, 1>) {
                      ^bb0(%arg26: memref<8192xf32, 1>, %arg27: memref<8192xf32, 1>):
                        %c1_i32_68 = arith.constant 1 : i32
                        %c64_i32_69 = arith.constant 64 : i32
                        %c0_i32_70 = arith.constant 0 : i32
                        %c64 = arith.constant 64 : index
                        %c512_i32 = arith.constant 512 : i32
                        %c8_i32 = arith.constant 8 : i32
                        %c0_71 = arith.constant 0 : index
                        %c128_i32_72 = arith.constant 128 : i32
                        scf.for %arg28 = %c0_i32_70 to %c64_i32_69 step %c1_i32_68  : i32 {
                          %164 = scf.for %arg29 = %c0_i32_70 to %c128_i32_72 step %c64_i32_69 iter_args(%arg30 = %c128_i32_72) -> (i32)  : i32 {
                            %mask, %remaining_out = ascvec.update_mask %arg30 : !ascvec.mask
                            %165 = arith.muli %arg28, %c128_i32_72 overflow<nsw> : i32
                            %166 = arith.addi %165, %arg29 overflow<nsw> : i32
                            %167 = arith.index_cast %166 : i32 to index
                            %168 = ascvec.loadalign %arg26[%167] : memref<8192xf32, 1>, index -> <f32>
                            %169 = arith.muli %arg28, %c8_i32 overflow<nsw> : i32
                            %170 = arith.divsi %arg29, %c8_i32 : i32
                            %171 = arith.muli %170, %c512_i32 overflow<nsw> : i32
                            %172 = arith.addi %169, %171 overflow<nsw> : i32
                            %173 = arith.index_cast %172 : i32 to index
                            ascvec.storealign_strided %arg27[%173], %168, %mask, %c64, %c0_71 : memref<8192xf32, 1>, <f32>
                            scf.yield %remaining_out : i32
                          }
                        }
                      } {cannir.vf_mode = "simd"}
                      ascvec.release_buf v, %c19_i64, 0 {cannir.buffer.generated_sync}
                      ascvec.sync_intra_arrive v, %c14_i64 {cannir.sync.banks = 3 : i32}
                      %162 = ascscalar.bitcast_memref %alloc_53 byte_offset = %c0_i64 : memref<8192xf32, 1> -> memref<?xf32, 1>
                      %163 = ascscalar.bitcast_memref %alloc_58 byte_offset = %c0_i64 : memref<16384xf32, 2> -> memref<?xf32, 2>
                      ascvec.sync_intra_wait mte3, %c0_i64 {cannir.sync.banks = 3 : i32}
                      ascvec.get_buf mte3, %c19_i64, 0 {cannir.buffer.generated_sync}
                      ascvec.copy_ub2l1_strided %163[%c512], %162[%c0], %c16_i32, %c64_i32, %c0_i32, %c64_i32 : memref<?xf32, 2>, memref<?xf32, 1>
                      ascvec.release_buf mte3, %c19_i64, 0 {cannir.buffer.generated_sync}
                      ascvec.sync_intra_arrive mte3, %c0_i64 {cannir.sync.banks = 3 : i32}
                    }
                    ascvec.sync_block_arrive mte3, %c10_i64
                  } else {
                    %141 = arith.cmpi eq, %113, %c0_i64 : i64
                    scf.if %141 {
                      %142 = cannir.extract_layout %arg9 : !cannir.tensor<<gm, f32>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                      %143 = cannir.extract_shape %142 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, -1, 128)>
                      %144 = cannir.extract_dim %143[2] : !cannir.shape<(-1, -1, -1, 128)>
                      %145 = arith.minsi %120, %144 : i64
                      %146 = arith.subi %144, %145 : i64
                      %147 = arith.minsi %146, %c1_i64 : i64
                      %148 = arith.cmpi sle, %120, %144 : i64
                      %149 = arith.select %148, %147, %c0_i64 : i64
                      %150 = cannir.extract_physical_layout %arg9 : !cannir.tensor<<gm, f32>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                      %151 = cannir.extract_stride %150 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, 128, 1)>
                      %152 = cannir.extract_dim %151[0] : !cannir.stride<(-1, -1, 128, 1)>
                      %153 = arith.muli %117, %152 : i64
                      %154 = arith.muli %153, %c4_i64 : i64
                      %155 = cannir.extract_dim %151[1] : !cannir.stride<(-1, -1, 128, 1)>
                      %156 = arith.muli %arg24, %155 : i64
                      %157 = arith.muli %156, %c4_i64 : i64
                      %158 = arith.addi %154, %157 : i64
                      %159 = arith.muli %120, %c512_i64 : i64
                      %160 = arith.addi %158, %159 : i64
                      ascvec.get_buf mte2, %c17_i64, 0 {cannir.buffer.generated_sync}
                      "ascvec.copy_gm2ub_plan"(%114, %alloc_8, %c0_i32, %c0_i32, %c0_i64, %160, %149, %c256_i64, %c512_i64, %c256_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 65536 : i64, ub_capacity = 256 : i64}> : (memref<?xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                      ascvec.release_buf mte2, %c17_i64, 0 {cannir.buffer.generated_sync}
                      ascvec.sync_notify {dst_pipe = 2 : i32, event_id = 0 : i32, src_pipe = 5 : i32}
                      ascvec.sync_wait {dst_pipe = 2 : i32, event_id = 0 : i32, src_pipe = 5 : i32}
                      ascvec.sync_intra_wait v, %c14_i64 {cannir.sync.banks = 3 : i32}
                      ascvec.get_buf v, %c17_i64, 0 {cannir.buffer.generated_sync}
                      ascvec.vf(%alloc_49, %alloc_51 : memref<8192xf32, 1>, memref<64xf32, 1>) {
                      ^bb0(%arg26: memref<8192xf32, 1>, %arg27: memref<64xf32, 1>):
                        %c64_i32_68 = arith.constant 64 : i32
                        %c0_i32_69 = arith.constant 0 : i32
                        %c1_i32_70 = arith.constant 1 : i32
                        %c128_i32_71 = arith.constant 128 : i32
                        %163 = ascvec.create_mask : !ascvec.mask
                        scf.for %arg28 = %c0_i32_69 to %c64_i32_68 step %c1_i32_70  : i32 {
                          %164 = arith.muli %arg28, %c128_i32_71 overflow<nsw> : i32
                          %165 = arith.index_cast %164 : i32 to index
                          %166 = ascvec.loadalign %arg26[%165] : memref<8192xf32, 1>, index -> <f32>
                          %167 = arith.index_cast %arg28 : i32 to index
                          %168 = ascvec.loadalign_brc %arg27[%167] : memref<64xf32, 1>, index -> <f32>
                          %169 = ascvec.add %166, %168, %163 : <f32>
                          %170 = arith.addi %arg28, %c1_i32_70 overflow<nsw> : i32
                          %mask, %remaining_out = ascvec.update_mask %170 : !ascvec.mask
                          %mask_72, %remaining_out_73 = ascvec.update_mask %arg28 : !ascvec.mask
                          %171 = ascvec.select %169, %166, %mask : <f32>
                          %172 = ascvec.select %166, %171, %mask_72 : <f32>
                          ascvec.storealign %arg26[%165], %172, %163 : memref<8192xf32, 1>, index, <f32>
                        }
                        ascvec.mem_bar vst_vld
                      } {cannir.vf_mode = "simd"}
                      ascvec.release_buf v, %c17_i64, 0 {cannir.buffer.generated_sync}
                      ascvec.get_buf v, %c19_i64, 0 {cannir.buffer.generated_sync}
                      ascvec.vf(%alloc_49, %alloc_53 : memref<8192xf32, 1>, memref<8192xf32, 1>) {
                      ^bb0(%arg26: memref<8192xf32, 1>, %arg27: memref<8192xf32, 1>):
                        %c1_i32_68 = arith.constant 1 : i32
                        %c64_i32_69 = arith.constant 64 : i32
                        %c0_i32_70 = arith.constant 0 : i32
                        %c64 = arith.constant 64 : index
                        %c512_i32 = arith.constant 512 : i32
                        %c8_i32 = arith.constant 8 : i32
                        %c0_71 = arith.constant 0 : index
                        %c128_i32_72 = arith.constant 128 : i32
                        scf.for %arg28 = %c0_i32_70 to %c64_i32_69 step %c1_i32_68  : i32 {
                          %163 = scf.for %arg29 = %c0_i32_70 to %c128_i32_72 step %c64_i32_69 iter_args(%arg30 = %c128_i32_72) -> (i32)  : i32 {
                            %mask, %remaining_out = ascvec.update_mask %arg30 : !ascvec.mask
                            %164 = arith.muli %arg28, %c128_i32_72 overflow<nsw> : i32
                            %165 = arith.addi %164, %arg29 overflow<nsw> : i32
                            %166 = arith.index_cast %165 : i32 to index
                            %167 = ascvec.loadalign %arg26[%166] : memref<8192xf32, 1>, index -> <f32>
                            %168 = arith.muli %arg28, %c8_i32 overflow<nsw> : i32
                            %169 = arith.divsi %arg29, %c8_i32 : i32
                            %170 = arith.muli %169, %c512_i32 overflow<nsw> : i32
                            %171 = arith.addi %168, %170 overflow<nsw> : i32
                            %172 = arith.index_cast %171 : i32 to index
                            ascvec.storealign_strided %arg27[%172], %167, %mask, %c64, %c0_71 : memref<8192xf32, 1>, <f32>
                            scf.yield %remaining_out : i32
                          }
                        }
                      } {cannir.vf_mode = "simd"}
                      ascvec.release_buf v, %c19_i64, 0 {cannir.buffer.generated_sync}
                      ascvec.sync_intra_arrive v, %c14_i64 {cannir.sync.banks = 3 : i32}
                      %161 = ascscalar.bitcast_memref %alloc_53 byte_offset = %c0_i64 : memref<8192xf32, 1> -> memref<?xf32, 1>
                      %162 = ascscalar.bitcast_memref %alloc_57 byte_offset = %c0_i64 : memref<16384xf32, 2> -> memref<?xf32, 2>
                      ascvec.sync_intra_wait mte3, %c13_i64 {cannir.sync.banks = 3 : i32}
                      ascvec.get_buf mte3, %c19_i64, 0 {cannir.buffer.generated_sync}
                      ascvec.copy_ub2l1_strided %162[%c0], %161[%c0], %c16_i32, %c64_i32, %c0_i32, %c64_i32 : memref<?xf32, 2>, memref<?xf32, 1>
                      ascvec.release_buf mte3, %c19_i64, 0 {cannir.buffer.generated_sync}
                      ascvec.sync_intra_arrive mte3, %c13_i64 {cannir.sync.banks = 3 : i32}
                    } else {
                      %142 = cannir.extract_layout %arg9 : !cannir.tensor<<gm, f32>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                      %143 = cannir.extract_shape %142 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, -1, 128)>
                      %144 = cannir.extract_dim %143[2] : !cannir.shape<(-1, -1, -1, 128)>
                      %145 = arith.minsi %120, %144 : i64
                      %146 = arith.subi %144, %145 : i64
                      %147 = arith.minsi %146, %c1_i64 : i64
                      %148 = arith.cmpi sle, %120, %144 : i64
                      %149 = arith.select %148, %147, %c0_i64 : i64
                      %150 = cannir.extract_physical_layout %arg9 : !cannir.tensor<<gm, f32>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                      %151 = cannir.extract_stride %150 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, 128, 1)>
                      %152 = cannir.extract_dim %151[0] : !cannir.stride<(-1, -1, 128, 1)>
                      %153 = arith.muli %117, %152 : i64
                      %154 = arith.muli %153, %c4_i64 : i64
                      %155 = cannir.extract_dim %151[1] : !cannir.stride<(-1, -1, 128, 1)>
                      %156 = arith.muli %arg24, %155 : i64
                      %157 = arith.muli %156, %c4_i64 : i64
                      %158 = arith.addi %154, %157 : i64
                      %159 = arith.muli %120, %c512_i64 : i64
                      %160 = arith.addi %158, %159 : i64
                      %161 = arith.addi %160, %c256_i64 : i64
                      ascvec.get_buf mte2, %c17_i64, 0 {cannir.buffer.generated_sync}
                      "ascvec.copy_gm2ub_plan"(%114, %alloc_8, %c0_i32, %c0_i32, %c0_i64, %161, %149, %c256_i64, %c512_i64, %c256_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 65536 : i64, ub_capacity = 256 : i64}> : (memref<?xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                      ascvec.release_buf mte2, %c17_i64, 0 {cannir.buffer.generated_sync}
                      ascvec.sync_notify {dst_pipe = 2 : i32, event_id = 0 : i32, src_pipe = 5 : i32}
                      ascvec.sync_wait {dst_pipe = 2 : i32, event_id = 0 : i32, src_pipe = 5 : i32}
                      ascvec.sync_intra_wait v, %c14_i64 {cannir.sync.banks = 3 : i32}
                      ascvec.get_buf v, %c17_i64, 0 {cannir.buffer.generated_sync}
                      ascvec.vf(%alloc_49, %alloc_51 : memref<8192xf32, 1>, memref<64xf32, 1>) {
                      ^bb0(%arg26: memref<8192xf32, 1>, %arg27: memref<64xf32, 1>):
                        %c0_i32_68 = arith.constant 0 : i32
                        %c1_i32_69 = arith.constant 1 : i32
                        %c64_i32_70 = arith.constant 64 : i32
                        %c128_i32_71 = arith.constant 128 : i32
                        %164 = ascvec.create_mask : !ascvec.mask
                        scf.for %arg28 = %c0_i32_68 to %c64_i32_70 step %c1_i32_69  : i32 {
                          %165 = arith.muli %arg28, %c128_i32_71 overflow<nsw> : i32
                          %166 = arith.addi %165, %c64_i32_70 overflow<nsw> : i32
                          %167 = arith.index_cast %166 : i32 to index
                          %168 = ascvec.loadalign %arg26[%167] : memref<8192xf32, 1>, index -> <f32>
                          %169 = arith.index_cast %arg28 : i32 to index
                          %170 = ascvec.loadalign_brc %arg27[%169] : memref<64xf32, 1>, index -> <f32>
                          %171 = ascvec.add %168, %170, %164 : <f32>
                          %172 = arith.addi %arg28, %c1_i32_69 overflow<nsw> : i32
                          %mask, %remaining_out = ascvec.update_mask %172 : !ascvec.mask
                          %mask_72, %remaining_out_73 = ascvec.update_mask %arg28 : !ascvec.mask
                          %173 = ascvec.select %171, %168, %mask : <f32>
                          %174 = ascvec.select %168, %173, %mask_72 : <f32>
                          ascvec.storealign %arg26[%167], %174, %164 : memref<8192xf32, 1>, index, <f32>
                        }
                        ascvec.mem_bar vst_vld
                      } {cannir.vf_mode = "simd"}
                      ascvec.release_buf v, %c17_i64, 0 {cannir.buffer.generated_sync}
                      ascvec.get_buf v, %c19_i64, 0 {cannir.buffer.generated_sync}
                      ascvec.vf(%alloc_49, %alloc_53 : memref<8192xf32, 1>, memref<8192xf32, 1>) {
                      ^bb0(%arg26: memref<8192xf32, 1>, %arg27: memref<8192xf32, 1>):
                        %c1_i32_68 = arith.constant 1 : i32
                        %c64_i32_69 = arith.constant 64 : i32
                        %c0_i32_70 = arith.constant 0 : i32
                        %c64 = arith.constant 64 : index
                        %c512_i32 = arith.constant 512 : i32
                        %c8_i32 = arith.constant 8 : i32
                        %c0_71 = arith.constant 0 : index
                        %c128_i32_72 = arith.constant 128 : i32
                        scf.for %arg28 = %c0_i32_70 to %c64_i32_69 step %c1_i32_68  : i32 {
                          %164 = scf.for %arg29 = %c0_i32_70 to %c128_i32_72 step %c64_i32_69 iter_args(%arg30 = %c128_i32_72) -> (i32)  : i32 {
                            %mask, %remaining_out = ascvec.update_mask %arg30 : !ascvec.mask
                            %165 = arith.muli %arg28, %c128_i32_72 overflow<nsw> : i32
                            %166 = arith.addi %165, %arg29 overflow<nsw> : i32
                            %167 = arith.index_cast %166 : i32 to index
                            %168 = ascvec.loadalign %arg26[%167] : memref<8192xf32, 1>, index -> <f32>
                            %169 = arith.muli %arg28, %c8_i32 overflow<nsw> : i32
                            %170 = arith.divsi %arg29, %c8_i32 : i32
                            %171 = arith.muli %170, %c512_i32 overflow<nsw> : i32
                            %172 = arith.addi %169, %171 overflow<nsw> : i32
                            %173 = arith.index_cast %172 : i32 to index
                            ascvec.storealign_strided %arg27[%173], %168, %mask, %c64, %c0_71 : memref<8192xf32, 1>, <f32>
                            scf.yield %remaining_out : i32
                          }
                        }
                      } {cannir.vf_mode = "simd"}
                      ascvec.release_buf v, %c19_i64, 0 {cannir.buffer.generated_sync}
                      ascvec.sync_intra_arrive v, %c14_i64 {cannir.sync.banks = 3 : i32}
                      %162 = ascscalar.bitcast_memref %alloc_53 byte_offset = %c0_i64 : memref<8192xf32, 1> -> memref<?xf32, 1>
                      %163 = ascscalar.bitcast_memref %alloc_57 byte_offset = %c0_i64 : memref<16384xf32, 2> -> memref<?xf32, 2>
                      ascvec.sync_intra_wait mte3, %c13_i64 {cannir.sync.banks = 3 : i32}
                      ascvec.get_buf mte3, %c19_i64, 0 {cannir.buffer.generated_sync}
                      ascvec.copy_ub2l1_strided %163[%c512], %162[%c0], %c16_i32, %c64_i32, %c0_i32, %c64_i32 : memref<?xf32, 2>, memref<?xf32, 1>
                      ascvec.release_buf mte3, %c19_i64, 0 {cannir.buffer.generated_sync}
                      ascvec.sync_intra_arrive mte3, %c13_i64 {cannir.sync.banks = 3 : i32}
                    }
                  }
                }
                %139 = arith.select %136, %133, %false : i1
                scf.if %139 {
                  scf.if %135 {
                    %140 = arith.cmpi eq, %113, %c0_i64 : i64
                    scf.if %140 {
                      ascvec.sync_intra_wait v, %c15_i64 {cannir.sync.banks = 3 : i32}
                      ascvec.get_buf v, %c18_i64, 0 {cannir.buffer.generated_sync}
                      ascvec.vf(%alloc_50, %alloc_52 : memref<8192xf32, 1>, memref<8192xf32, 1>) {
                      ^bb0(%arg26: memref<8192xf32, 1>, %arg27: memref<8192xf32, 1>):
                        %c1_i32_68 = arith.constant 1 : i32
                        %c64_i32_69 = arith.constant 64 : i32
                        %c0_i32_70 = arith.constant 0 : i32
                        %c128_i32_71 = arith.constant 128 : i32
                        %c32_i32_72 = arith.constant 32 : i32
                        %mask, %remaining_out = ascvec.update_mask %c32_i32_72 : !ascvec.mask
                        scf.for %arg28 = %c0_i32_70 to %c64_i32_69 step %c1_i32_68  : i32 {
                          %167 = arith.muli %arg28, %c128_i32_71 overflow<nsw> : i32
                          scf.for %arg29 = %c0_i32_70 to %c128_i32_71 step %c32_i32_72  : i32 {
                            %168 = arith.addi %167, %arg29 overflow<nsw> : i32
                            %169 = arith.index_cast %168 : i32 to index
                            %170 = ascvec.loadalign %arg26[%169] : memref<8192xf32, 1>, index -> <f32>
                            ascvec.storealign %arg27[%169], %170, %mask : memref<8192xf32, 1>, index, <f32>
                          } {cannir.loop_unroll = 4 : i64}
                        } {cannir.loop_unroll = 4 : i64}
                        ascvec.mem_bar vst_vld
                      } {cannir.vf_mode = "simd"}
                      ascvec.release_buf v, %c18_i64, 0 {cannir.buffer.generated_sync}
                      ascvec.sync_intra_arrive v, %c15_i64 {cannir.sync.banks = 3 : i32}
                      %141 = cannir.extract_layout %arg15 : !cannir.tensor<<gm, f32>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, identity>
                      %142 = cannir.extract_shape %141 : <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>> -> <(-1, -1, 128, 128)>
                      %143 = cannir.extract_dim %142[1] : !cannir.shape<(-1, -1, 128, 128)>
                      %144 = arith.remsi %119, %143 : i64
                      %145 = arith.cmpi ne, %144, %c0_i64 : i64
                      %146 = arith.cmpi slt, %144, %c0_i64 : i64
                      %147 = arith.cmpi slt, %143, %c0_i64 : i64
                      %148 = arith.xori %146, %147 : i1
                      %149 = arith.andi %145, %148 : i1
                      %150 = arith.divsi %119, %143 : i64
                      %151 = arith.extui %149 : i1 to i64
                      %152 = arith.subi %150, %151 : i64
                      %153 = arith.muli %152, %143 : i64
                      %154 = arith.subi %119, %153 : i64
                      %155 = cannir.extract_physical_layout %arg15 : !cannir.tensor<<gm, f32>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, identity>
                      %156 = cannir.extract_stride %155 : <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>> -> <(-1, 16384, 128, 1)>
                      %157 = cannir.extract_dim %156[0] : !cannir.stride<(-1, 16384, 128, 1)>
                      %158 = arith.muli %152, %157 : i64
                      %159 = arith.muli %154, %c16384_i64 : i64
                      %160 = arith.addi %158, %159 : i64
                      %161 = arith.muli %160, %c32_i64 : i64
                      %162 = ascscalar.memref_address %alloc_14 : memref<1xf32>
                      %163 = arith.divui %161, %c8_i64 : i64
                      %164 = arith.addi %162, %163 : i64
                      %165 = ascscalar.bind_gm_ptr %164 : memref<1xf32>
                      %166 = ascscalar.bitcast_memref %alloc_52 byte_offset = %c0_i64 : memref<8192xf32, 1> -> memref<?xf32, 1>
                      ascvec.get_buf mte3, %c18_i64, 0 {cannir.buffer.generated_sync}
                      ascvec.copy_ub2gm %165[%c0], %166[%c0], %c1_i32, %c32768_i32, %c0_i64, %c32768_i32 l2_cache_ctl %c0_i32 : memref<1xf32>, memref<?xf32, 1>
                      ascvec.release_buf mte3, %c18_i64, 0 {cannir.buffer.generated_sync}
                    } else {
                      ascvec.sync_intra_wait v, %c15_i64 {cannir.sync.banks = 3 : i32}
                      ascvec.get_buf v, %c18_i64, 0 {cannir.buffer.generated_sync}
                      ascvec.vf(%alloc_50, %alloc_52 : memref<8192xf32, 1>, memref<8192xf32, 1>) {
                      ^bb0(%arg26: memref<8192xf32, 1>, %arg27: memref<8192xf32, 1>):
                        %c1_i32_68 = arith.constant 1 : i32
                        %c64_i32_69 = arith.constant 64 : i32
                        %c0_i32_70 = arith.constant 0 : i32
                        %c128_i32_71 = arith.constant 128 : i32
                        %c32_i32_72 = arith.constant 32 : i32
                        %mask, %remaining_out = ascvec.update_mask %c32_i32_72 : !ascvec.mask
                        scf.for %arg28 = %c0_i32_70 to %c64_i32_69 step %c1_i32_68  : i32 {
                          %168 = arith.muli %arg28, %c128_i32_71 overflow<nsw> : i32
                          scf.for %arg29 = %c0_i32_70 to %c128_i32_71 step %c32_i32_72  : i32 {
                            %169 = arith.addi %168, %arg29 overflow<nsw> : i32
                            %170 = arith.index_cast %169 : i32 to index
                            %171 = ascvec.loadalign %arg26[%170] : memref<8192xf32, 1>, index -> <f32>
                            ascvec.storealign %arg27[%170], %171, %mask : memref<8192xf32, 1>, index, <f32>
                          } {cannir.loop_unroll = 4 : i64}
                        } {cannir.loop_unroll = 4 : i64}
                        ascvec.mem_bar vst_vld
                      } {cannir.vf_mode = "simd"}
                      ascvec.release_buf v, %c18_i64, 0 {cannir.buffer.generated_sync}
                      ascvec.sync_intra_arrive v, %c15_i64 {cannir.sync.banks = 3 : i32}
                      %141 = cannir.extract_layout %arg15 : !cannir.tensor<<gm, f32>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, identity>
                      %142 = cannir.extract_shape %141 : <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>> -> <(-1, -1, 128, 128)>
                      %143 = cannir.extract_dim %142[1] : !cannir.shape<(-1, -1, 128, 128)>
                      %144 = arith.remsi %119, %143 : i64
                      %145 = arith.cmpi ne, %144, %c0_i64 : i64
                      %146 = arith.cmpi slt, %144, %c0_i64 : i64
                      %147 = arith.cmpi slt, %143, %c0_i64 : i64
                      %148 = arith.xori %146, %147 : i1
                      %149 = arith.andi %145, %148 : i1
                      %150 = arith.divsi %119, %143 : i64
                      %151 = arith.extui %149 : i1 to i64
                      %152 = arith.subi %150, %151 : i64
                      %153 = arith.muli %152, %143 : i64
                      %154 = arith.subi %119, %153 : i64
                      %155 = cannir.extract_physical_layout %arg15 : !cannir.tensor<<gm, f32>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, identity>
                      %156 = cannir.extract_stride %155 : <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>> -> <(-1, 16384, 128, 1)>
                      %157 = cannir.extract_dim %156[0] : !cannir.stride<(-1, 16384, 128, 1)>
                      %158 = arith.muli %152, %157 : i64
                      %159 = arith.muli %154, %c16384_i64 : i64
                      %160 = arith.addi %158, %159 : i64
                      %161 = arith.addi %160, %c8192_i64 : i64
                      %162 = arith.muli %161, %c32_i64 : i64
                      %163 = ascscalar.memref_address %alloc_14 : memref<1xf32>
                      %164 = arith.divui %162, %c8_i64 : i64
                      %165 = arith.addi %163, %164 : i64
                      %166 = ascscalar.bind_gm_ptr %165 : memref<1xf32>
                      %167 = ascscalar.bitcast_memref %alloc_52 byte_offset = %c0_i64 : memref<8192xf32, 1> -> memref<?xf32, 1>
                      ascvec.get_buf mte3, %c18_i64, 0 {cannir.buffer.generated_sync}
                      ascvec.copy_ub2gm %166[%c0], %167[%c0], %c1_i32, %c32768_i32, %c0_i64, %c32768_i32 l2_cache_ctl %c0_i32 : memref<1xf32>, memref<?xf32, 1>
                      ascvec.release_buf mte3, %c18_i64, 0 {cannir.buffer.generated_sync}
                    }
                  }
                } else {
                  scf.if %135 {
                    %140 = arith.cmpi eq, %113, %c0_i64 : i64
                    scf.if %140 {
                      ascvec.sync_intra_wait v, %c15_i64 {cannir.sync.banks = 3 : i32}
                      ascvec.get_buf v, %c18_i64, 0 {cannir.buffer.generated_sync}
                      ascvec.vf(%alloc_50, %alloc_52 : memref<8192xf32, 1>, memref<8192xf32, 1>) {
                      ^bb0(%arg26: memref<8192xf32, 1>, %arg27: memref<8192xf32, 1>):
                        %c1_i32_68 = arith.constant 1 : i32
                        %c64_i32_69 = arith.constant 64 : i32
                        %c0_i32_70 = arith.constant 0 : i32
                        %c128_i32_71 = arith.constant 128 : i32
                        %c32_i32_72 = arith.constant 32 : i32
                        %mask, %remaining_out = ascvec.update_mask %c32_i32_72 : !ascvec.mask
                        scf.for %arg28 = %c0_i32_70 to %c64_i32_69 step %c1_i32_68  : i32 {
                          %167 = arith.muli %arg28, %c128_i32_71 overflow<nsw> : i32
                          scf.for %arg29 = %c0_i32_70 to %c128_i32_71 step %c32_i32_72  : i32 {
                            %168 = arith.addi %167, %arg29 overflow<nsw> : i32
                            %169 = arith.index_cast %168 : i32 to index
                            %170 = ascvec.loadalign %arg26[%169] : memref<8192xf32, 1>, index -> <f32>
                            ascvec.storealign %arg27[%169], %170, %mask : memref<8192xf32, 1>, index, <f32>
                          } {cannir.loop_unroll = 4 : i64}
                        } {cannir.loop_unroll = 4 : i64}
                        ascvec.mem_bar vst_vld
                      } {cannir.vf_mode = "simd"}
                      ascvec.release_buf v, %c18_i64, 0 {cannir.buffer.generated_sync}
                      ascvec.sync_intra_arrive v, %c15_i64 {cannir.sync.banks = 3 : i32}
                      %141 = cannir.extract_layout %arg15 : !cannir.tensor<<gm, f32>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, identity>
                      %142 = cannir.extract_shape %141 : <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>> -> <(-1, -1, 128, 128)>
                      %143 = cannir.extract_dim %142[1] : !cannir.shape<(-1, -1, 128, 128)>
                      %144 = arith.remsi %119, %143 : i64
                      %145 = arith.cmpi ne, %144, %c0_i64 : i64
                      %146 = arith.cmpi slt, %144, %c0_i64 : i64
                      %147 = arith.cmpi slt, %143, %c0_i64 : i64
                      %148 = arith.xori %146, %147 : i1
                      %149 = arith.andi %145, %148 : i1
                      %150 = arith.divsi %119, %143 : i64
                      %151 = arith.extui %149 : i1 to i64
                      %152 = arith.subi %150, %151 : i64
                      %153 = arith.muli %152, %143 : i64
                      %154 = arith.subi %119, %153 : i64
                      %155 = cannir.extract_physical_layout %arg15 : !cannir.tensor<<gm, f32>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, identity>
                      %156 = cannir.extract_stride %155 : <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>> -> <(-1, 16384, 128, 1)>
                      %157 = cannir.extract_dim %156[0] : !cannir.stride<(-1, 16384, 128, 1)>
                      %158 = arith.muli %152, %157 : i64
                      %159 = arith.muli %154, %c16384_i64 : i64
                      %160 = arith.addi %158, %159 : i64
                      %161 = arith.muli %160, %c32_i64 : i64
                      %162 = ascscalar.memref_address %alloc_14 : memref<1xf32>
                      %163 = arith.divui %161, %c8_i64 : i64
                      %164 = arith.addi %162, %163 : i64
                      %165 = ascscalar.bind_gm_ptr %164 : memref<1xf32>
                      %166 = ascscalar.bitcast_memref %alloc_52 byte_offset = %c0_i64 : memref<8192xf32, 1> -> memref<?xf32, 1>
                      ascvec.get_buf mte3, %c18_i64, 0 {cannir.buffer.generated_sync}
                      ascvec.copy_ub2gm %165[%c0], %166[%c0], %c1_i32, %c32768_i32, %c0_i64, %c32768_i32 l2_cache_ctl %c0_i32 : memref<1xf32>, memref<?xf32, 1>
                      ascvec.release_buf mte3, %c18_i64, 0 {cannir.buffer.generated_sync}
                    } else {
                      ascvec.sync_intra_wait v, %c15_i64 {cannir.sync.banks = 3 : i32}
                      ascvec.get_buf v, %c18_i64, 0 {cannir.buffer.generated_sync}
                      ascvec.vf(%alloc_50, %alloc_52 : memref<8192xf32, 1>, memref<8192xf32, 1>) {
                      ^bb0(%arg26: memref<8192xf32, 1>, %arg27: memref<8192xf32, 1>):
                        %c1_i32_68 = arith.constant 1 : i32
                        %c64_i32_69 = arith.constant 64 : i32
                        %c0_i32_70 = arith.constant 0 : i32
                        %c128_i32_71 = arith.constant 128 : i32
                        %c32_i32_72 = arith.constant 32 : i32
                        %mask, %remaining_out = ascvec.update_mask %c32_i32_72 : !ascvec.mask
                        scf.for %arg28 = %c0_i32_70 to %c64_i32_69 step %c1_i32_68  : i32 {
                          %168 = arith.muli %arg28, %c128_i32_71 overflow<nsw> : i32
                          scf.for %arg29 = %c0_i32_70 to %c128_i32_71 step %c32_i32_72  : i32 {
                            %169 = arith.addi %168, %arg29 overflow<nsw> : i32
                            %170 = arith.index_cast %169 : i32 to index
                            %171 = ascvec.loadalign %arg26[%170] : memref<8192xf32, 1>, index -> <f32>
                            ascvec.storealign %arg27[%170], %171, %mask : memref<8192xf32, 1>, index, <f32>
                          } {cannir.loop_unroll = 4 : i64}
                        } {cannir.loop_unroll = 4 : i64}
                        ascvec.mem_bar vst_vld
                      } {cannir.vf_mode = "simd"}
                      ascvec.release_buf v, %c18_i64, 0 {cannir.buffer.generated_sync}
                      ascvec.sync_intra_arrive v, %c15_i64 {cannir.sync.banks = 3 : i32}
                      %141 = cannir.extract_layout %arg15 : !cannir.tensor<<gm, f32>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, identity>
                      %142 = cannir.extract_shape %141 : <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>> -> <(-1, -1, 128, 128)>
                      %143 = cannir.extract_dim %142[1] : !cannir.shape<(-1, -1, 128, 128)>
                      %144 = arith.remsi %119, %143 : i64
                      %145 = arith.cmpi ne, %144, %c0_i64 : i64
                      %146 = arith.cmpi slt, %144, %c0_i64 : i64
                      %147 = arith.cmpi slt, %143, %c0_i64 : i64
                      %148 = arith.xori %146, %147 : i1
                      %149 = arith.andi %145, %148 : i1
                      %150 = arith.divsi %119, %143 : i64
                      %151 = arith.extui %149 : i1 to i64
                      %152 = arith.subi %150, %151 : i64
                      %153 = arith.muli %152, %143 : i64
                      %154 = arith.subi %119, %153 : i64
                      %155 = cannir.extract_physical_layout %arg15 : !cannir.tensor<<gm, f32>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, identity>
                      %156 = cannir.extract_stride %155 : <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>> -> <(-1, 16384, 128, 1)>
                      %157 = cannir.extract_dim %156[0] : !cannir.stride<(-1, 16384, 128, 1)>
                      %158 = arith.muli %152, %157 : i64
                      %159 = arith.muli %154, %c16384_i64 : i64
                      %160 = arith.addi %158, %159 : i64
                      %161 = arith.addi %160, %c8192_i64 : i64
                      %162 = arith.muli %161, %c32_i64 : i64
                      %163 = ascscalar.memref_address %alloc_14 : memref<1xf32>
                      %164 = arith.divui %162, %c8_i64 : i64
                      %165 = arith.addi %163, %164 : i64
                      %166 = ascscalar.bind_gm_ptr %165 : memref<1xf32>
                      %167 = ascscalar.bitcast_memref %alloc_52 byte_offset = %c0_i64 : memref<8192xf32, 1> -> memref<?xf32, 1>
                      ascvec.get_buf mte3, %c18_i64, 0 {cannir.buffer.generated_sync}
                      ascvec.copy_ub2gm %166[%c0], %167[%c0], %c1_i32, %c32768_i32, %c0_i64, %c32768_i32 l2_cache_ctl %c0_i32 : memref<1xf32>, memref<?xf32, 1>
                      ascvec.release_buf mte3, %c18_i64, 0 {cannir.buffer.generated_sync}
                    }
                  }
                }
              }
            }
          }
        }
        %47 = arith.addi %arg22, %c1_i32 : i32
        %48 = arith.cmpi slt, %47, %11 : i32
        scf.if %48 {
          ascvec.sync_all
          ascvec.sync_block_arrive mte3, %c1_i64
          ascvec.sync_block_wait s, %c9_i64
        }
      }
      ascvec.sync_intra_wait mte3, %c0_i64
      ascvec.sync_intra_wait mte3, %c1_i64
      ascvec.sync_intra_wait mte3, %c5_i64
      ascvec.sync_intra_wait mte3, %c6_i64
      ascvec.sync_intra_wait mte3, %c7_i64
      ascvec.sync_intra_wait mte3, %c8_i64
      ascvec.sync_intra_wait mte3, %c9_i64
      ascvec.sync_intra_wait mte3, %c10_i64
      ascvec.sync_intra_wait mte3, %c11_i64
      ascvec.sync_intra_wait mte3, %c12_i64
      ascvec.sync_intra_wait mte3, %c13_i64
    }
    return
  }
  func.func @kernel_FlashKdaCpPreprocessKernel___call___0_aic(%arg0: !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>, %arg1: !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>, %arg2: !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>, %arg3: !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 1)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 1)>, <(-1, -1, -1, 1)>>, identity>, %arg4: !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>, %arg5: !cannir.tensor<<gm, f32>, <<(-1, 2, 64, 64)>, <(8192, 4096, 64, 1)>>, <<(-1, 2, 64, 64)>, <(8192, 4096, 64, 1)>>, identity>, %arg6: !cannir.tensor<<gm, f32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg7: !cannir.tensor<<gm, f32>, <<(-1, 128)>, <(128, 1)>>, <<(-1, 128)>, <(128, 1)>>, identity>, %arg8: !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>, %arg9: !cannir.tensor<<gm, f32>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>, %arg10: !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>, %arg11: !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>, %arg12: !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 64, 64)>, <(-1, -1, 4096, 64, 1)>>, <<(-1, -1, -1, 64, 64)>, <(-1, -1, 4096, 64, 1)>>, identity>, %arg13: !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>, %arg14: !cannir.tensor<<gm, f32>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, identity>, %arg15: !cannir.tensor<<gm, f32>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, identity>, %arg16: !cannir.tensor<<gm, f16>, <<(16, 16)>, <(16, 1)>>, <<(16, 16)>, <(16, 1)>>, identity>, %arg17: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg18: i64, %arg19: i64, %arg20: f32, %arg21: f32) attributes {cannir.buffer.auto_sync, cannir.buffer.sync_analyzed, cannir.buffer.sync_verified, cannir.buffer.tensor_state, cannir.core = "aic", cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.device, cannir.explicit_ring_buffer_cursors, cannir.kernel_group = "kernel_FlashKdaCpPreprocessKernel___call___0", cannir.resource.ids_prepared} {
    %c768 = arith.constant 768 : index
    %c512 = arith.constant 512 : index
    %c3584 = arith.constant 3584 : index
    %c3840 = arith.constant 3840 : index
    %c1280 = arith.constant 1280 : index
    %c2560 = arith.constant 2560 : index
    %c2816 = arith.constant 2816 : index
    %c5_i32 = arith.constant 5 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %c4_i32 = arith.constant 4 : i32
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c6 = arith.constant 6 : index
    %c5 = arith.constant 5 : index
    %c3 = arith.constant 3 : index
    %c1262764338_i32 = arith.constant 1262764338 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %false = arith.constant false
    %c1_i32 = arith.constant 1 : i32
    %c9_i32 = arith.constant 9 : i32
    %c0 = arith.constant 0 : index
    %c256_i32 = arith.constant 256 : i32
    %c64_i32 = arith.constant 64 : i32
    %c128_i32 = arith.constant 128 : i32
    %c16_i32 = arith.constant 16 : i32
    %c3_i32 = arith.constant 3 : i32
    %c512_i64 = arith.constant 512 : i64
    %c16384_i64 = arith.constant 16384 : i64
    %c1024 = arith.constant 1024 : index
    %c2048 = arith.constant 2048 : index
    %c3072 = arith.constant 3072 : index
    %c8_i32 = arith.constant 8 : i32
    %true = arith.constant true
    %c2 = arith.constant 2 : index
    %c80_i64 = arith.constant 80 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %c2048_i64 = arith.constant 2048 : i64
    %c3072_i64 = arith.constant 3072 : i64
    %c256 = arith.constant 256 : index
    %c65_i32 = arith.constant 65 : i32
    %c64_i64 = arith.constant 64 : i64
    %c63_i64 = arith.constant 63 : i64
    %c40_i64 = arith.constant 40 : i64
    %c39_i64 = arith.constant 39 : i64
    %c32_i64 = arith.constant 32 : i64
    %c31_i64 = arith.constant 31 : i64
    %c30_i64 = arith.constant 30 : i64
    %c29_i64 = arith.constant 29 : i64
    %c28_i64 = arith.constant 28 : i64
    %c27_i64 = arith.constant 27 : i64
    %c26_i64 = arith.constant 26 : i64
    %c25_i64 = arith.constant 25 : i64
    %c24_i64 = arith.constant 24 : i64
    %c23_i64 = arith.constant 23 : i64
    %c22_i64 = arith.constant 22 : i64
    %c21_i64 = arith.constant 21 : i64
    %c20_i64 = arith.constant 20 : i64
    %c19_i64 = arith.constant 19 : i64
    %c18_i64 = arith.constant 18 : i64
    %c17_i64 = arith.constant 17 : i64
    %c16_i64 = arith.constant 16 : i64
    %c15_i64 = arith.constant 15 : i64
    %c14_i64 = arith.constant 14 : i64
    %c13_i64 = arith.constant 13 : i64
    %c12_i64 = arith.constant 12 : i64
    %c11_i64 = arith.constant 11 : i64
    %c10_i64 = arith.constant 10 : i64
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
    %c5120_i64 = arith.constant 5120 : i64
    %c6144_i64 = arith.constant 6144 : i64
    %c10240_i64 = arith.constant 10240 : i64
    %c7168_i64 = arith.constant 7168 : i64
    %c11264_i64 = arith.constant 11264 : i64
    %c15360_i64 = arith.constant 15360 : i64
    %alloc = memref.alloc() {cannir.arg_index = 11 : i64} : memref<1xbf16>
    %alloc_0 = memref.alloc() {cannir.arg_index = 13 : i64} : memref<1xbf16>
    %alloc_1 = memref.alloc() {cannir.arg_index = 14 : i64} : memref<1xf32>
    %alloc_2 = memref.alloc() {cannir.arg_index = 15 : i64} : memref<1xf32>
    %alloc_3 = memref.alloc() {cannir.arg_index = 16 : i64} : memref<256xf16>
    %alloc_4 = memref.alloc() {cannir.arg_index = 17 : i64} : memref<1xi32>
    %alloc_5 = memref.alloc() {cannir.byte_offset = 148640 : i64} : memref<2048xf32, 1>
    %alloc_6 = memref.alloc() {cannir.byte_offset = 156832 : i64} : memref<2048xf32, 1>
    %alloc_7 = memref.alloc() {cannir.byte_offset = 230816 : i64} : memref<2048xf32, 1>
    %alloc_8 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<8192xf32, 1>
    %alloc_9 = memref.alloc() {cannir.byte_offset = 32768 : i64} : memref<8192xf32, 1>
    %alloc_10 = memref.alloc() {cannir.byte_offset = 40960 : i64} : memref<4096xf16, 2>
    %alloc_11 = memref.alloc() {cannir.byte_offset = 49152 : i64} : memref<4096xf16, 2>
    %alloc_12 = memref.alloc() {cannir.byte_offset = 65536 : i64} : memref<4096xf32, 5>
    %alloc_13 = memref.alloc() {cannir.byte_offset = 81920 : i64} : memref<4096xf32, 5>
    %alloc_14 = memref.alloc() {cannir.byte_offset = 32768 : i64} : memref<2048xbf16, 3>
    %alloc_15 = memref.alloc() {cannir.byte_offset = 32768 : i64} : memref<2048xbf16, 4>
    %alloc_16 = memref.alloc() {cannir.byte_offset = 36864 : i64} : memref<2048xbf16, 3>
    %alloc_17 = memref.alloc() {cannir.byte_offset = 36864 : i64} : memref<2048xbf16, 4>
    %alloc_18 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<16384xf32, 5>
    %alloc_19 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<16384xf32, 3>
    %alloc_20 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<16384xf32, 4>
    %alloc_21 = memref.alloc() {cannir.byte_offset = 65536 : i64} : memref<16384xf32, 5>
    %alloc_22 = memref.alloc() {cannir.byte_offset = 294912 : i64} : memref<16384xf32, 2>
    %alloc_23 = memref.alloc() {cannir.byte_offset = 229376 : i64} : memref<16384xf32, 2>
    %alloc_24 = memref.alloc() {cannir.byte_offset = 98304 : i64} : memref<16384xf32, 2>
    %alloc_25 = memref.alloc() {cannir.byte_offset = 163840 : i64} : memref<16384xf32, 2>
    %alloc_26 = memref.alloc() {cannir.byte_offset = 425984 : i64} : memref<16384xf32, 2>
    %alloc_27 = memref.alloc() {cannir.byte_offset = 360448 : i64} : memref<16384xf32, 2>
    %alloc_28 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<8192xf32, 2>
    %alloc_29 = memref.alloc() {cannir.byte_offset = 32768 : i64} : memref<8192xf32, 2>
    %alloc_30 = memref.alloc() {cannir.byte_offset = 65536 : i64} : memref<8192xf32, 2>
    %alloc_31 = memref.alloc() {cannir.byte_offset = 32768 : i64} : memref<8192xf32, 5>
    %alloc_32 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<8192xf32, 5>
    %alloc_33 = memref.alloc() {cannir.byte_offset = 77824 : i64} : memref<8192xbf16, 2>
    %alloc_34 = memref.alloc() {cannir.byte_offset = 16384 : i64} : memref<8192xbf16, 4>
    %alloc_35 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<8192xbf16, 4>
    %alloc_36 = memref.alloc() {cannir.byte_offset = 94208 : i64} : memref<8192xbf16, 2>
    %alloc_37 = memref.alloc() {cannir.byte_offset = 57344 : i64} : memref<4096xbf16, 2>
    %alloc_38 = memref.alloc() {cannir.byte_offset = 16384 : i64} : memref<8192xbf16, 3>
    %alloc_39 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<8192xbf16, 3>
    %alloc_40 = memref.alloc() {cannir.byte_offset = 16384 : i64} : memref<4096xf16, 2>
    %alloc_41 = memref.alloc() {cannir.byte_offset = 8192 : i64} : memref<4096xf16, 2>
    %alloc_42 = memref.alloc() {cannir.byte_offset = 49152 : i64} : memref<4096xf16, 4>
    %alloc_43 = memref.alloc() {cannir.byte_offset = 40960 : i64} : memref<4096xf16, 4>
    %alloc_44 = memref.alloc() {cannir.byte_offset = 49152 : i64} : memref<4096xf16, 3>
    %alloc_45 = memref.alloc() {cannir.byte_offset = 40960 : i64} : memref<4096xf16, 3>
    %alloc_46 = memref.alloc() {cannir.byte_offset = 24576 : i64} : memref<4096xf16, 2>
    %alloc_47 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<4096xf16, 2>
    %alloc_48 = memref.alloc() {cannir.byte_offset = 32768 : i64} : memref<4096xf16, 2>
    %alloc_49 = memref.alloc() {cannir.byte_offset = 73728 : i64} : memref<2048xbf16, 2>
    %alloc_50 = memref.alloc() {cannir.byte_offset = 69632 : i64} : memref<2048xbf16, 2>
    %alloc_51 = memref.alloc() {cannir.byte_offset = 65536 : i64} : memref<2048xbf16, 2>
    %0 = ascscalar.get_block_idx
    %1 = ascscalar.get_block_num
    %2 = memref.load %alloc_4[%c2] : memref<1xi32>
    %3 = memref.load %alloc_4[%c3] : memref<1xi32>
    %4 = memref.load %alloc_4[%c5] : memref<1xi32>
    %5 = arith.extsi %4 : i32 to i64
    %6 = memref.load %alloc_4[%c6] : memref<1xi32>
    %7 = arith.extsi %6 : i32 to i64
    %8 = memref.load %alloc_4[%c0] : memref<1xi32>
    %9 = arith.cmpi eq, %8, %c1262764338_i32 : i32
    %10 = scf.if %9 -> (i1) {
      %11 = memref.load %alloc_4[%c1] : memref<1xi32>
      %12 = arith.cmpi eq, %11, %c2_i32 : i32
      %13 = scf.if %12 -> (i1) {
        %14 = arith.cmpi eq, %2, %c0_i32 : i32
        %15 = scf.if %14 -> (i1) {
          %16 = arith.extsi %3 : i32 to i64
          %17 = arith.cmpi eq, %16, %1 : i64
          %18 = scf.if %17 -> (i1) {
            %19 = cannir.extract_layout %arg1 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
            %20 = cannir.extract_shape %19 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 128)>
            %21 = cannir.extract_dim %20[1] : !cannir.shape<(-1, -1, -1, 128)>
            %22 = arith.cmpi eq, %7, %21 : i64
            %23 = arith.muli %5, %7 : i64
            %24 = arith.cmpi eq, %23, %arg18 : i64
            %25 = arith.select %22, %24, %false : i1
            scf.yield %25 : i1
          } else {
            scf.yield %false : i1
          }
          scf.yield %18 : i1
        } else {
          scf.yield %false : i1
        }
        scf.yield %15 : i1
      } else {
        scf.yield %false : i1
      }
      scf.yield %13 : i1
    } else {
      scf.yield %false : i1
    }
    scf.if %10 {
      asccube.sync_intra_arrive mte1, %c13_i64
      asccube.sync_intra_arrive mte1, %c29_i64
      asccube.sync_intra_arrive mte1, %c12_i64
      asccube.sync_intra_arrive mte1, %c28_i64
      asccube.sync_intra_arrive mte1, %c11_i64
      asccube.sync_intra_arrive mte1, %c27_i64
      asccube.sync_intra_arrive mte1, %c10_i64
      asccube.sync_intra_arrive mte1, %c26_i64
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
      asccube.sync_intra_arrive mte1, %c1_i64
      asccube.sync_intra_arrive mte1, %c17_i64
      asccube.sync_intra_arrive mte1, %c0_i64
      asccube.sync_intra_arrive mte1, %c16_i64
      %11 = memref.load %alloc_4[%c4] : memref<1xi32>
      scf.for %arg22 = %c0_i32 to %11 step %c1_i32  : i32 {
        %12 = arith.addi %arg22, %c9_i32 : i32
        %13 = arith.index_cast %12 : i32 to index
        %14 = memref.load %alloc_4[%13] : memref<1xi32>
        %15 = arith.addi %14, %c1_i32 : i32
        %16 = arith.index_cast %15 : i32 to index
        %17 = memref.load %alloc_4[%16] : memref<1xi32>
        %18 = arith.addi %14, %c2_i32 : i32
        %19 = arith.addi %18, %17 : i32
        %20 = arith.addi %19, %17 : i32
        %21 = arith.addi %20, %17 : i32
        %22 = arith.addi %21, %17 : i32
        %23 = arith.addi %22, %c1_i32 : i32
        %24 = arith.addi %23, %17 : i32
        %25 = arith.addi %24, %17 : i32
        %26 = arith.addi %25, %17 : i32
        %27 = arith.addi %26, %c1_i32 : i32
        %28 = arith.muli %0, %c2_i64 : i64
        %29 = arith.extsi %27 : i32 to i64
        %30 = arith.addi %29, %28 : i64
        %31 = arith.index_cast %30 : i64 to index
        %32 = memref.load %alloc_4[%31] : memref<1xi32>
        %33 = arith.addi %30, %c1_i64 : i64
        %34 = arith.index_cast %33 : i64 to index
        %35 = memref.load %alloc_4[%34] : memref<1xi32>
        scf.for %arg23 = %c0_i32 to %17 step %c1_i32  : i32 {
          %49 = arith.addi %25, %arg23 : i32
          %50 = arith.index_cast %49 : i32 to index
          %51 = memref.load %alloc_4[%50] : memref<1xi32>
          %52 = arith.addi %49, %c1_i32 : i32
          %53 = arith.index_cast %52 : i32 to index
          %54 = memref.load %alloc_4[%53] : memref<1xi32>
          %55 = arith.cmpi slt, %32, %51 : i32
          %56 = arith.select %55, %51, %32 : i32
          %57 = arith.cmpi sgt, %35, %54 : i32
          %58 = arith.select %57, %54, %35 : i32
          %59 = arith.cmpi slt, %56, %58 : i32
          %60 = arith.subi %56, %51 : i32
          %61 = arith.subi %58, %51 : i32
          %62 = arith.extsi %60 : i32 to i64
          %63 = arith.extsi %61 : i32 to i64
          %64 = arith.select %59, %62, %c0_i64 : i64
          %65 = arith.select %59, %63, %c0_i64 : i64
          %66 = arith.addi %19, %arg23 : i32
          %67 = arith.index_cast %66 : i32 to index
          %68 = memref.load %alloc_4[%67] : memref<1xi32>
          %69 = arith.addi %20, %arg23 : i32
          %70 = arith.index_cast %69 : i32 to index
          %71 = memref.load %alloc_4[%70] : memref<1xi32>
          %72 = arith.addi %21, %arg23 : i32
          %73 = arith.index_cast %72 : i32 to index
          %74 = memref.load %alloc_4[%73] : memref<1xi32>
          %75 = arith.extsi %74 : i32 to i64
          %76 = arith.addi %23, %arg23 : i32
          %77 = arith.index_cast %76 : i32 to index
          %78 = memref.load %alloc_4[%77] : memref<1xi32>
          %79 = arith.addi %24, %arg23 : i32
          %80 = arith.index_cast %79 : i32 to index
          %81 = memref.load %alloc_4[%80] : memref<1xi32>
          %82 = arith.addi %75, %c63_i64 : i64
          %83 = arith.remsi %82, %c64_i64 : i64
          %84 = arith.cmpi ne, %83, %c0_i64 : i64
          %85 = arith.cmpi slt, %83, %c0_i64 : i64
          %86 = arith.andi %84, %85 : i1
          %87 = arith.divsi %82, %c64_i64 : i64
          %88 = arith.extui %86 : i1 to i64
          %89 = arith.subi %87, %88 : i64
          %90 = arith.subi %89, %c1_i64 : i64
          %91 = arith.muli %90, %c64_i64 : i64
          %92 = arith.subi %75, %91 : i64
          %93 = arith.extsi %71 : i32 to i64
          %94 = arith.extsi %68 : i32 to i64
          %95 = arith.addi %78, %81 : i32
          %96 = arith.extsi %95 : i32 to i64
          %97 = arith.cmpi eq, %96, %89 : i64
          %98 = arith.cmpi slt, %64, %65 : i64
          scf.if %98 {
            asccube.get_buf mte2, %c4_i64, 0 {cannir.buffer.generated_sync}
            asccube.fill_l1 %alloc_10[%c0], %c0_i32, %c256_i32, %c1_i32, %c0_i32 : memref<4096xf16, 2>, i32
            asccube.release_buf mte2, %c4_i64, 0 {cannir.buffer.generated_sync}
            asccube.sync_pipe mte2
            %99 = ascscalar.bitcast_memref %alloc_10 byte_offset = %c0_i64 : memref<4096xf16, 2> -> memref<?xf16, 2>
            asccube.get_buf mte2, %c4_i64, 0 {cannir.buffer.generated_sync}
            asccube.copy_gm2l1 %99[%c0], %alloc_3[%c0], %c16_i32, %c16_i32, %c32_i64, %c4_i32, %c0_i64, %c80_i64, %c64_i32 l2_cache_ctl %c0_i32 {nd2nz = true} : memref<?xf16, 2>, memref<256xf16>
            asccube.release_buf mte2, %c4_i64, 0 {cannir.buffer.generated_sync}
            %100 = cannir.alloc_array : <2xi64>
            %101 = cannir.alloc_array : <2xi64>
            %102:6 = scf.for %arg24 = %64 to %65 step %c1_i64 iter_args(%arg25 = %c0_i64, %arg26 = %c0_i64, %arg27 = %c0_i64, %arg28 = %c0_i64, %arg29 = %c0_i64, %arg30 = %c0_i64) -> (i64, i64, i64, i64, i64, i64)  : i64 {
              %107 = arith.cmpi sge, %arg25, %c1_i64 : i64
              %108 = arith.subi %arg25, %c1_i64 : i64
              %109 = arith.cmpi slt, %108, %arg25 : i64
              %110 = arith.select %107, %109, %false : i1
              %111:2 = scf.if %110 -> (i64, i64) {
                %204 = arith.remsi %108, %c2_i64 : i64
                %205 = arith.index_cast %204 : i64 to index
                %206 = cannir.array_load %101[%205] : <2xi64>
                %207 = ascscalar.bitcast_memref %alloc_48 byte_offset = %c0_i64 : memref<4096xf16, 2> -> memref<?xf16, 2>
                %208 = arith.cmpi slt, %206, %c64_i64 : i64
                %209 = scf.if %208 -> (i64) {
                  %339 = arith.addi %arg30, %c1_i64 overflow<nsw> : i64
                  %340 = arith.remui %339, %c2_i64 : i64
                  %341 = arith.addi %340, %c1_i64 overflow<nsw> : i64
                  %342 = arith.remui %341, %c2_i64 : i64
                  scf.yield %342 : i64
                } else {
                  %339 = arith.addi %arg30, %c1_i64 overflow<nsw> : i64
                  %340 = arith.remui %339, %c2_i64 : i64
                  %341 = arith.addi %340, %c1_i64 overflow<nsw> : i64
                  %342 = arith.remui %341, %c2_i64 : i64
                  scf.yield %342 : i64
                }
                asccube.get_buf mte2, %c5_i64, 0 {cannir.buffer.generated_sync}
                asccube.fill_l1 %alloc_11[%c0], %c0_i32, %c256_i32, %c1_i32, %c0_i32 : memref<4096xf16, 2>, i32
                asccube.release_buf mte2, %c5_i64, 0 {cannir.buffer.generated_sync}
                %210 = arith.cmpi eq, %arg28, %c0_i64 : i64
                %211 = ascscalar.memref_address %alloc_45 : memref<4096xf16, 3>
                %212 = ascscalar.memref_address %alloc_44 : memref<4096xf16, 3>
                %213 = arith.select %210, %211, %212 : i64
                %214 = arith.select %210, %c17_i64, %c18_i64 : i64
                %215 = arith.addi %arg28, %c1_i64 overflow<nsw> : i64
                %216 = arith.remui %215, %c2_i64 : i64
                %217 = ascscalar.memref_address %alloc_43 : memref<4096xf16, 4>
                %218 = ascscalar.memref_address %alloc_42 : memref<4096xf16, 4>
                %219 = arith.select %210, %217, %218 : i64
                %220 = arith.select %210, %c19_i64, %c20_i64 : i64
                %221 = ascscalar.bind_local_ptr %213 : memref<1xf16, 3>
                asccube.sync_intra_wait mte1, %c1_i64 {cannir.sync.banks = 3 : i32}
                asccube.sync_intra_wait mte1, %c17_i64 {cannir.sync.banks = 3 : i32}
                asccube.get_buf mte1, %214, 0 {cannir.buffer.generated_sync}
                asccube.copy_l12l0a %221[%c0], %207[%c0], %c0_i32, %c0_i32, %c4_i32, %c4_i32, %c4_i32, %c4_i32 : memref<1xf16, 3>, memref<?xf16, 2>
                asccube.release_buf mte1, %214, 0 {cannir.buffer.generated_sync}
                %222 = ascscalar.bind_local_ptr %219 : memref<1xf16, 4>
                asccube.get_buf mte1, %220, 0 {cannir.buffer.generated_sync}
                asccube.get_buf mte1, %c4_i64, 0 {cannir.buffer.generated_sync}
                asccube.copy_l12l0b %222[%c0], %99[%c0], %c0_i32, %c0_i32, %c4_i32, %c4_i32, %c4_i32, %c4_i32 : memref<1xf16, 4>, memref<?xf16, 2>
                asccube.release_buf mte1, %220, 0 {cannir.buffer.generated_sync}
                %223 = arith.cmpi eq, %216, %c0_i64 : i64
                %224 = arith.select %223, %211, %212 : i64
                %225 = arith.select %223, %c17_i64, %c18_i64 : i64
                %226 = arith.addi %216, %c1_i64 overflow<nsw> : i64
                %227 = arith.remui %226, %c2_i64 : i64
                %228 = arith.select %223, %217, %218 : i64
                %229 = arith.select %223, %c19_i64, %c20_i64 : i64
                %230 = ascscalar.bind_local_ptr %224 : memref<1xf16, 3>
                asccube.get_buf mte1, %225, 0 {cannir.buffer.generated_sync}
                asccube.copy_l12l0a %230[%c0], %99[%c0], %c0_i32, %c0_i32, %c4_i32, %c4_i32, %c4_i32, %c4_i32 : memref<1xf16, 3>, memref<?xf16, 2>
                asccube.release_buf mte1, %225, 0 {cannir.buffer.generated_sync}
                %231 = ascscalar.bind_local_ptr %228 : memref<1xf16, 4>
                asccube.get_buf mte1, %229, 0 {cannir.buffer.generated_sync}
                asccube.copy_l12l0b %231[%c0], %99[%c0], %c0_i32, %c0_i32, %c4_i32, %c4_i32, %c4_i32, %c4_i32 : memref<1xf16, 4>, memref<?xf16, 2>
                asccube.release_buf mte1, %c4_i64, 0 {cannir.buffer.generated_sync}
                asccube.release_buf mte1, %229, 0 {cannir.buffer.generated_sync}
                %232 = ascscalar.bitcast_memref %alloc_12 byte_offset = %c0_i64 : memref<4096xf32, 5> -> memref<?xf32, 5>
                asccube.get_buf m, %c21_i64, 0 {cannir.buffer.generated_sync}
                asccube.get_buf m, %214, 0 {cannir.buffer.generated_sync}
                asccube.get_buf m, %220, 0 {cannir.buffer.generated_sync}
                asccube.get_buf m, %225, 0 {cannir.buffer.generated_sync}
                asccube.get_buf m, %229, 0 {cannir.buffer.generated_sync}
                asccube.mmad %232[%c0], %221, %222, %c64_i32, %c64_i32, %c64_i32, %c2_i32 {c_matrix_init_val = true} : memref<?xf32, 5>, memref<1xf16, 3>, memref<1xf16, 4>
                asccube.mmad %232[%c0], %230, %231, %c64_i32, %c64_i32, %c64_i32, %c3_i32 {c_matrix_init_val = false} : memref<?xf32, 5>, memref<1xf16, 3>, memref<1xf16, 4>
                asccube.release_buf m, %229, 0 {cannir.buffer.generated_sync}
                asccube.release_buf m, %225, 0 {cannir.buffer.generated_sync}
                asccube.release_buf m, %220, 0 {cannir.buffer.generated_sync}
                asccube.release_buf m, %214, 0 {cannir.buffer.generated_sync}
                asccube.release_buf m, %c21_i64, 0 {cannir.buffer.generated_sync}
                %233 = ascscalar.bitcast_memref %alloc_46 byte_offset = %c0_i64 : memref<4096xf16, 2> -> memref<?xf16, 2>
                asccube.get_buf fixpipe, %c3_i64, 0 {cannir.buffer.generated_sync}
                asccube.get_buf fixpipe, %c21_i64, 0 {cannir.buffer.generated_sync}
                asccube.copy_l0c2l1 %233[%c0], %232[%c0], %c64_i32, %c64_i32, %c1024_i32, %c64_i32, %c3_i32 {relu_en = false} : memref<?xf16, 2>, memref<?xf32, 5>
                asccube.release_buf fixpipe, %c21_i64, 0 {cannir.buffer.generated_sync}
                asccube.release_buf fixpipe, %c3_i64, 0 {cannir.buffer.generated_sync}
                %234 = arith.cmpi eq, %227, %c0_i64 : i64
                %235 = arith.select %234, %211, %212 : i64
                %236 = arith.select %234, %c17_i64, %c18_i64 : i64
                %237 = arith.addi %227, %c1_i64 overflow<nsw> : i64
                %238 = arith.remui %237, %c2_i64 : i64
                %239 = arith.select %234, %217, %218 : i64
                %240 = arith.select %234, %c19_i64, %c20_i64 : i64
                %241 = ascscalar.bind_local_ptr %235 : memref<1xf16, 3>
                %242 = ascscalar.bitcast_memref %alloc_11 byte_offset = %c0_i64 : memref<4096xf16, 2> -> memref<?xf16, 2>
                asccube.get_buf mte1, %236, 0 {cannir.buffer.generated_sync}
                asccube.get_buf mte1, %c5_i64, 0 {cannir.buffer.generated_sync}
                asccube.copy_l12l0a %241[%c0], %242[%c0], %c0_i32, %c0_i32, %c4_i32, %c4_i32, %c4_i32, %c4_i32 : memref<1xf16, 3>, memref<?xf16, 2>
                %243 = ascscalar.bind_local_ptr %239 : memref<1xf16, 4>
                asccube.get_buf mte1, %240, 0 {cannir.buffer.generated_sync}
                asccube.copy_l12l0b %243[%c0], %242[%c0], %c0_i32, %c0_i32, %c4_i32, %c4_i32, %c4_i32, %c4_i32 : memref<1xf16, 4>, memref<?xf16, 2>
                asccube.release_buf mte1, %c5_i64, 0 {cannir.buffer.generated_sync}
                asccube.copy_l12l0a %241[%c0], %207[%c0], %c0_i32, %c0_i32, %c1_i32, %c4_i32, %c5_i32, %c5_i32 : memref<1xf16, 3>, memref<?xf16, 2>
                asccube.release_buf mte1, %236, 0 {cannir.buffer.generated_sync}
                asccube.copy_l12l0b %243[%c0], %207[%c0], %c0_i32, %c0_i32, %c1_i32, %c4_i32, %c5_i32, %c4_i32 {cannir.l1_l0b_transpose_repeat, transpose = true} : memref<1xf16, 4>, memref<?xf16, 2>
                asccube.release_buf mte1, %240, 0 {cannir.buffer.generated_sync}
                asccube.sync_intra_arrive mte1, %c1_i64 {cannir.sync.banks = 3 : i32}
                asccube.sync_intra_arrive mte1, %c17_i64 {cannir.sync.banks = 3 : i32}
                %244 = ascscalar.bitcast_memref %alloc_13 byte_offset = %c0_i64 : memref<4096xf32, 5> -> memref<?xf32, 5>
                asccube.get_buf m, %c22_i64, 0 {cannir.buffer.generated_sync}
                asccube.get_buf m, %236, 0 {cannir.buffer.generated_sync}
                asccube.get_buf m, %240, 0 {cannir.buffer.generated_sync}
                asccube.mmad %244[%c0], %241, %243, %c64_i32, %c64_i32, %c64_i32, %c3_i32 {c_matrix_init_val = true} : memref<?xf32, 5>, memref<1xf16, 3>, memref<1xf16, 4>
                asccube.release_buf m, %240, 0 {cannir.buffer.generated_sync}
                asccube.release_buf m, %236, 0 {cannir.buffer.generated_sync}
                asccube.release_buf m, %c22_i64, 0 {cannir.buffer.generated_sync}
                %245 = ascscalar.bitcast_memref %alloc_40 byte_offset = %c0_i64 : memref<4096xf16, 2> -> memref<?xf16, 2>
                asccube.get_buf fixpipe, %c2_i64, 0 {cannir.buffer.generated_sync}
                asccube.get_buf fixpipe, %c22_i64, 0 {cannir.buffer.generated_sync}
                asccube.copy_l0c2l1 %245[%c0], %244[%c0], %c64_i32, %c64_i32, %c1024_i32, %c64_i32, %c3_i32 {relu_en = false} : memref<?xf16, 2>, memref<?xf32, 5>
                asccube.release_buf fixpipe, %c22_i64, 0 {cannir.buffer.generated_sync}
                asccube.release_buf fixpipe, %c2_i64, 0 {cannir.buffer.generated_sync}
                %246 = arith.cmpi eq, %238, %c0_i64 : i64
                %247 = arith.select %246, %211, %212 : i64
                %248 = arith.select %246, %c17_i64, %c18_i64 : i64
                %249 = arith.addi %238, %c1_i64 overflow<nsw> : i64
                %250 = arith.remui %249, %c2_i64 : i64
                %251 = arith.select %246, %217, %218 : i64
                %252 = arith.select %246, %c19_i64, %c20_i64 : i64
                %253 = ascscalar.bind_local_ptr %247 : memref<1xf16, 3>
                asccube.get_buf mte1, %248, 0 {cannir.buffer.generated_sync}
                asccube.get_buf mte1, %c3_i64, 0 {cannir.buffer.generated_sync}
                asccube.copy_l12l0a %253[%c0], %233[%c0], %c0_i32, %c0_i32, %c1_i32, %c4_i32, %c5_i32, %c5_i32 : memref<1xf16, 3>, memref<?xf16, 2>
                asccube.release_buf mte1, %248, 0 {cannir.buffer.generated_sync}
                %254 = ascscalar.bind_local_ptr %251 : memref<1xf16, 4>
                asccube.get_buf mte1, %252, 0 {cannir.buffer.generated_sync}
                asccube.get_buf mte1, %c4_i64, 0 {cannir.buffer.generated_sync}
                asccube.copy_l12l0b %254[%c0], %99[%c0], %c0_i32, %c0_i32, %c1_i32, %c4_i32, %c5_i32, %c4_i32 {cannir.l1_l0b_transpose_repeat, transpose = true} : memref<1xf16, 4>, memref<?xf16, 2>
                asccube.release_buf mte1, %c4_i64, 0 {cannir.buffer.generated_sync}
                asccube.release_buf mte1, %252, 0 {cannir.buffer.generated_sync}
                %255 = arith.cmpi eq, %250, %c0_i64 : i64
                %256 = arith.select %255, %211, %212 : i64
                %257 = arith.select %255, %c17_i64, %c18_i64 : i64
                %258 = arith.addi %250, %c1_i64 overflow<nsw> : i64
                %259 = arith.remui %258, %c2_i64 : i64
                %260 = arith.select %255, %217, %218 : i64
                %261 = arith.select %255, %c19_i64, %c20_i64 : i64
                %262 = ascscalar.bind_local_ptr %256 : memref<1xf16, 3>
                asccube.get_buf mte1, %257, 0 {cannir.buffer.generated_sync}
                asccube.copy_l12l0a %262[%c0], %233[%c0], %c0_i32, %c0_i32, %c1_i32, %c4_i32, %c5_i32, %c5_i32 : memref<1xf16, 3>, memref<?xf16, 2>
                asccube.release_buf mte1, %c3_i64, 0 {cannir.buffer.generated_sync}
                asccube.release_buf mte1, %257, 0 {cannir.buffer.generated_sync}
                %263 = ascscalar.bind_local_ptr %260 : memref<1xf16, 4>
                asccube.get_buf mte1, %261, 0 {cannir.buffer.generated_sync}
                asccube.get_buf mte1, %c2_i64, 0 {cannir.buffer.generated_sync}
                asccube.copy_l12l0b %263[%c0], %245[%c0], %c0_i32, %c0_i32, %c1_i32, %c4_i32, %c5_i32, %c4_i32 {cannir.l1_l0b_transpose_repeat, transpose = true} : memref<1xf16, 4>, memref<?xf16, 2>
                asccube.release_buf mte1, %c2_i64, 0 {cannir.buffer.generated_sync}
                asccube.release_buf mte1, %261, 0 {cannir.buffer.generated_sync}
                asccube.get_buf m, %c21_i64, 0 {cannir.buffer.generated_sync}
                asccube.get_buf m, %248, 0 {cannir.buffer.generated_sync}
                asccube.get_buf m, %252, 0 {cannir.buffer.generated_sync}
                asccube.get_buf m, %257, 0 {cannir.buffer.generated_sync}
                asccube.get_buf m, %261, 0 {cannir.buffer.generated_sync}
                asccube.mmad %232[%c0], %253, %254, %c64_i32, %c64_i32, %c64_i32, %c2_i32 {c_matrix_init_val = true} : memref<?xf32, 5>, memref<1xf16, 3>, memref<1xf16, 4>
                asccube.mmad %232[%c0], %262, %263, %c64_i32, %c64_i32, %c64_i32, %c3_i32 {c_matrix_init_val = false} : memref<?xf32, 5>, memref<1xf16, 3>, memref<1xf16, 4>
                asccube.release_buf m, %261, 0 {cannir.buffer.generated_sync}
                asccube.release_buf m, %257, 0 {cannir.buffer.generated_sync}
                asccube.release_buf m, %252, 0 {cannir.buffer.generated_sync}
                asccube.release_buf m, %248, 0 {cannir.buffer.generated_sync}
                asccube.release_buf m, %c21_i64, 0 {cannir.buffer.generated_sync}
                %264 = ascscalar.bitcast_memref %alloc_41 byte_offset = %c0_i64 : memref<4096xf16, 2> -> memref<?xf16, 2>
                asccube.get_buf fixpipe, %c1_i64, 0 {cannir.buffer.generated_sync}
                asccube.get_buf fixpipe, %c21_i64, 0 {cannir.buffer.generated_sync}
                asccube.copy_l0c2l1 %264[%c0], %232[%c0], %c64_i32, %c64_i32, %c1024_i32, %c64_i32, %c3_i32 {relu_en = false} : memref<?xf16, 2>, memref<?xf32, 5>
                asccube.release_buf fixpipe, %c21_i64, 0 {cannir.buffer.generated_sync}
                asccube.release_buf fixpipe, %c1_i64, 0 {cannir.buffer.generated_sync}
                %265 = arith.cmpi eq, %259, %c0_i64 : i64
                %266 = arith.select %265, %211, %212 : i64
                %267 = arith.select %265, %c17_i64, %c18_i64 : i64
                %268 = arith.addi %259, %c1_i64 overflow<nsw> : i64
                %269 = arith.remui %268, %c2_i64 : i64
                %270 = arith.select %265, %217, %218 : i64
                %271 = arith.select %265, %c19_i64, %c20_i64 : i64
                %272 = ascscalar.bind_local_ptr %266 : memref<1xf16, 3>
                asccube.get_buf mte1, %267, 0 {cannir.buffer.generated_sync}
                asccube.copy_l12l0a %272[%c0], %245[%c0], %c0_i32, %c0_i32, %c1_i32, %c4_i32, %c5_i32, %c5_i32 : memref<1xf16, 3>, memref<?xf16, 2>
                asccube.release_buf mte1, %267, 0 {cannir.buffer.generated_sync}
                %273 = ascscalar.bind_local_ptr %270 : memref<1xf16, 4>
                asccube.get_buf mte1, %271, 0 {cannir.buffer.generated_sync}
                asccube.copy_l12l0b %273[%c0], %245[%c0], %c0_i32, %c0_i32, %c1_i32, %c4_i32, %c5_i32, %c4_i32 {cannir.l1_l0b_transpose_repeat, transpose = true} : memref<1xf16, 4>, memref<?xf16, 2>
                asccube.release_buf mte1, %271, 0 {cannir.buffer.generated_sync}
                asccube.get_buf m, %c22_i64, 0 {cannir.buffer.generated_sync}
                asccube.get_buf m, %267, 0 {cannir.buffer.generated_sync}
                asccube.get_buf m, %271, 0 {cannir.buffer.generated_sync}
                asccube.mmad %244[%c0], %272, %273, %c64_i32, %c64_i32, %c64_i32, %c3_i32 {c_matrix_init_val = true} : memref<?xf32, 5>, memref<1xf16, 3>, memref<1xf16, 4>
                asccube.release_buf m, %271, 0 {cannir.buffer.generated_sync}
                asccube.release_buf m, %267, 0 {cannir.buffer.generated_sync}
                asccube.release_buf m, %c22_i64, 0 {cannir.buffer.generated_sync}
                %274 = ascscalar.bitcast_memref %alloc_47 byte_offset = %c0_i64 : memref<4096xf16, 2> -> memref<?xf16, 2>
                asccube.get_buf fixpipe, %c0_i64, 0 {cannir.buffer.generated_sync}
                asccube.get_buf fixpipe, %c22_i64, 0 {cannir.buffer.generated_sync}
                asccube.copy_l0c2l1 %274[%c0], %244[%c0], %c64_i32, %c64_i32, %c1024_i32, %c64_i32, %c3_i32 {relu_en = false} : memref<?xf16, 2>, memref<?xf32, 5>
                asccube.release_buf fixpipe, %c22_i64, 0 {cannir.buffer.generated_sync}
                asccube.release_buf fixpipe, %c0_i64, 0 {cannir.buffer.generated_sync}
                %275 = arith.cmpi eq, %269, %c0_i64 : i64
                %276 = arith.select %275, %211, %212 : i64
                %277 = arith.select %275, %c17_i64, %c18_i64 : i64
                %278 = arith.addi %269, %c1_i64 overflow<nsw> : i64
                %279 = arith.remui %278, %c2_i64 : i64
                %280 = arith.select %275, %217, %218 : i64
                %281 = arith.select %275, %c19_i64, %c20_i64 : i64
                %282 = ascscalar.bind_local_ptr %276 : memref<1xf16, 3>
                asccube.get_buf mte1, %277, 0 {cannir.buffer.generated_sync}
                asccube.get_buf mte1, %c1_i64, 0 {cannir.buffer.generated_sync}
                asccube.copy_l12l0a %282[%c0], %264[%c0], %c0_i32, %c0_i32, %c1_i32, %c4_i32, %c5_i32, %c5_i32 : memref<1xf16, 3>, memref<?xf16, 2>
                asccube.release_buf mte1, %c1_i64, 0 {cannir.buffer.generated_sync}
                asccube.release_buf mte1, %277, 0 {cannir.buffer.generated_sync}
                %283 = ascscalar.bind_local_ptr %280 : memref<1xf16, 4>
                asccube.get_buf mte1, %281, 0 {cannir.buffer.generated_sync}
                asccube.get_buf mte1, %c0_i64, 0 {cannir.buffer.generated_sync}
                asccube.copy_l12l0b %283[%c0], %274[%c0], %c0_i32, %c0_i32, %c1_i32, %c4_i32, %c5_i32, %c4_i32 {cannir.l1_l0b_transpose_repeat, transpose = true} : memref<1xf16, 4>, memref<?xf16, 2>
                asccube.release_buf mte1, %c0_i64, 0 {cannir.buffer.generated_sync}
                asccube.release_buf mte1, %281, 0 {cannir.buffer.generated_sync}
                asccube.get_buf m, %c21_i64, 0 {cannir.buffer.generated_sync}
                asccube.get_buf m, %277, 0 {cannir.buffer.generated_sync}
                asccube.get_buf m, %281, 0 {cannir.buffer.generated_sync}
                asccube.mmad %232[%c0], %282, %283, %c64_i32, %c64_i32, %c64_i32, %c3_i32 {c_matrix_init_val = false} : memref<?xf32, 5>, memref<1xf16, 3>, memref<1xf16, 4>
                asccube.release_buf m, %281, 0 {cannir.buffer.generated_sync}
                asccube.release_buf m, %277, 0 {cannir.buffer.generated_sync}
                asccube.release_buf m, %c21_i64, 0 {cannir.buffer.generated_sync}
                asccube.get_buf fixpipe, %c1_i64, 0 {cannir.buffer.generated_sync}
                asccube.get_buf fixpipe, %c21_i64, 0 {cannir.buffer.generated_sync}
                asccube.copy_l0c2l1 %264[%c0], %232[%c0], %c64_i32, %c64_i32, %c1024_i32, %c64_i32, %c3_i32 {relu_en = false} : memref<?xf16, 2>, memref<?xf32, 5>
                asccube.release_buf fixpipe, %c21_i64, 0 {cannir.buffer.generated_sync}
                asccube.release_buf fixpipe, %c1_i64, 0 {cannir.buffer.generated_sync}
                %284 = arith.cmpi eq, %279, %c0_i64 : i64
                %285 = arith.select %284, %211, %212 : i64
                %286 = arith.select %284, %c17_i64, %c18_i64 : i64
                %287 = arith.addi %279, %c1_i64 overflow<nsw> : i64
                %288 = arith.remui %287, %c2_i64 : i64
                %289 = arith.select %284, %217, %218 : i64
                %290 = arith.select %284, %c19_i64, %c20_i64 : i64
                %291 = ascscalar.bind_local_ptr %285 : memref<1xf16, 3>
                asccube.get_buf mte1, %286, 0 {cannir.buffer.generated_sync}
                asccube.copy_l12l0a %291[%c0], %274[%c0], %c0_i32, %c0_i32, %c1_i32, %c4_i32, %c5_i32, %c5_i32 : memref<1xf16, 3>, memref<?xf16, 2>
                asccube.release_buf mte1, %286, 0 {cannir.buffer.generated_sync}
                %292 = ascscalar.bind_local_ptr %289 : memref<1xf16, 4>
                asccube.get_buf mte1, %290, 0 {cannir.buffer.generated_sync}
                asccube.copy_l12l0b %292[%c0], %274[%c0], %c0_i32, %c0_i32, %c1_i32, %c4_i32, %c5_i32, %c4_i32 {cannir.l1_l0b_transpose_repeat, transpose = true} : memref<1xf16, 4>, memref<?xf16, 2>
                asccube.release_buf mte1, %290, 0 {cannir.buffer.generated_sync}
                asccube.get_buf m, %c22_i64, 0 {cannir.buffer.generated_sync}
                asccube.get_buf m, %286, 0 {cannir.buffer.generated_sync}
                asccube.get_buf m, %290, 0 {cannir.buffer.generated_sync}
                asccube.mmad %244[%c0], %291, %292, %c64_i32, %c64_i32, %c64_i32, %c3_i32 {c_matrix_init_val = true} : memref<?xf32, 5>, memref<1xf16, 3>, memref<1xf16, 4>
                asccube.release_buf m, %290, 0 {cannir.buffer.generated_sync}
                asccube.release_buf m, %286, 0 {cannir.buffer.generated_sync}
                asccube.release_buf m, %c22_i64, 0 {cannir.buffer.generated_sync}
                asccube.get_buf fixpipe, %c2_i64, 0 {cannir.buffer.generated_sync}
                asccube.get_buf fixpipe, %c22_i64, 0 {cannir.buffer.generated_sync}
                asccube.copy_l0c2l1 %245[%c0], %244[%c0], %c64_i32, %c64_i32, %c1024_i32, %c64_i32, %c3_i32 {relu_en = false} : memref<?xf16, 2>, memref<?xf32, 5>
                asccube.release_buf fixpipe, %c22_i64, 0 {cannir.buffer.generated_sync}
                asccube.release_buf fixpipe, %c2_i64, 0 {cannir.buffer.generated_sync}
                %293 = arith.cmpi eq, %288, %c0_i64 : i64
                %294 = arith.select %293, %211, %212 : i64
                %295 = arith.select %293, %c17_i64, %c18_i64 : i64
                %296 = arith.addi %288, %c1_i64 overflow<nsw> : i64
                %297 = arith.remui %296, %c2_i64 : i64
                %298 = arith.select %293, %217, %218 : i64
                %299 = arith.select %293, %c19_i64, %c20_i64 : i64
                %300 = ascscalar.bind_local_ptr %294 : memref<1xf16, 3>
                asccube.get_buf mte1, %295, 0 {cannir.buffer.generated_sync}
                asccube.get_buf mte1, %c1_i64, 0 {cannir.buffer.generated_sync}
                asccube.copy_l12l0a %300[%c0], %264[%c0], %c0_i32, %c0_i32, %c1_i32, %c4_i32, %c5_i32, %c5_i32 : memref<1xf16, 3>, memref<?xf16, 2>
                asccube.release_buf mte1, %c1_i64, 0 {cannir.buffer.generated_sync}
                asccube.release_buf mte1, %295, 0 {cannir.buffer.generated_sync}
                %301 = ascscalar.bind_local_ptr %298 : memref<1xf16, 4>
                asccube.get_buf mte1, %299, 0 {cannir.buffer.generated_sync}
                asccube.get_buf mte1, %c2_i64, 0 {cannir.buffer.generated_sync}
                asccube.copy_l12l0b %301[%c0], %245[%c0], %c0_i32, %c0_i32, %c1_i32, %c4_i32, %c5_i32, %c4_i32 {cannir.l1_l0b_transpose_repeat, transpose = true} : memref<1xf16, 4>, memref<?xf16, 2>
                asccube.release_buf mte1, %c2_i64, 0 {cannir.buffer.generated_sync}
                asccube.release_buf mte1, %299, 0 {cannir.buffer.generated_sync}
                asccube.get_buf m, %c21_i64, 0 {cannir.buffer.generated_sync}
                asccube.get_buf m, %295, 0 {cannir.buffer.generated_sync}
                asccube.get_buf m, %299, 0 {cannir.buffer.generated_sync}
                asccube.mmad %232[%c0], %300, %301, %c64_i32, %c64_i32, %c64_i32, %c3_i32 {c_matrix_init_val = false} : memref<?xf32, 5>, memref<1xf16, 3>, memref<1xf16, 4>
                asccube.release_buf m, %299, 0 {cannir.buffer.generated_sync}
                asccube.release_buf m, %295, 0 {cannir.buffer.generated_sync}
                asccube.release_buf m, %c21_i64, 0 {cannir.buffer.generated_sync}
                asccube.get_buf fixpipe, %c1_i64, 0 {cannir.buffer.generated_sync}
                asccube.get_buf fixpipe, %c21_i64, 0 {cannir.buffer.generated_sync}
                asccube.copy_l0c2l1 %264[%c0], %232[%c0], %c64_i32, %c64_i32, %c1024_i32, %c64_i32, %c3_i32 {relu_en = false} : memref<?xf16, 2>, memref<?xf32, 5>
                asccube.release_buf fixpipe, %c21_i64, 0 {cannir.buffer.generated_sync}
                asccube.release_buf fixpipe, %c1_i64, 0 {cannir.buffer.generated_sync}
                %302 = arith.cmpi eq, %297, %c0_i64 : i64
                %303 = arith.select %302, %211, %212 : i64
                %304 = arith.select %302, %c17_i64, %c18_i64 : i64
                %305 = arith.addi %297, %c1_i64 overflow<nsw> : i64
                %306 = arith.remui %305, %c2_i64 : i64
                %307 = arith.select %302, %217, %218 : i64
                %308 = arith.select %302, %c19_i64, %c20_i64 : i64
                %309 = ascscalar.bind_local_ptr %303 : memref<1xf16, 3>
                asccube.get_buf mte1, %304, 0 {cannir.buffer.generated_sync}
                asccube.copy_l12l0a %309[%c0], %242[%c0], %c0_i32, %c0_i32, %c4_i32, %c4_i32, %c4_i32, %c4_i32 : memref<1xf16, 3>, memref<?xf16, 2>
                %310 = ascscalar.bind_local_ptr %307 : memref<1xf16, 4>
                asccube.get_buf mte1, %308, 0 {cannir.buffer.generated_sync}
                asccube.copy_l12l0b %310[%c0], %242[%c0], %c0_i32, %c0_i32, %c4_i32, %c4_i32, %c4_i32, %c4_i32 : memref<1xf16, 4>, memref<?xf16, 2>
                asccube.copy_l12l0a %309[%c256], %233[%c0], %c1_i32, %c0_i32, %c1_i32, %c1_i32, %c4_i32, %c4_i32 : memref<1xf16, 3>, memref<?xf16, 2>
                asccube.copy_l12l0a %309[%c2816], %233[%c0], %c3_i32, %c2_i32, %c1_i32, %c1_i32, %c4_i32, %c4_i32 : memref<1xf16, 3>, memref<?xf16, 2>
                asccube.release_buf mte1, %304, 0 {cannir.buffer.generated_sync}
                asccube.get_buf mte1, %c1_i64, 0 {cannir.buffer.generated_sync}
                asccube.copy_l12l0b %310[%c0], %264[%c0], %c0_i32, %c0_i32, %c1_i32, %c1_i32, %c4_i32, %c4_i32 {transpose = true} : memref<1xf16, 4>, memref<?xf16, 2>
                asccube.copy_l12l0b %310[%c2560], %264[%c0], %c2_i32, %c2_i32, %c1_i32, %c1_i32, %c4_i32, %c4_i32 {transpose = true} : memref<1xf16, 4>, memref<?xf16, 2>
                asccube.release_buf mte1, %c1_i64, 0 {cannir.buffer.generated_sync}
                asccube.release_buf mte1, %308, 0 {cannir.buffer.generated_sync}
                asccube.get_buf m, %c22_i64, 0 {cannir.buffer.generated_sync}
                asccube.get_buf m, %304, 0 {cannir.buffer.generated_sync}
                asccube.get_buf m, %308, 0 {cannir.buffer.generated_sync}
                asccube.mmad %244[%c0], %309, %310, %c64_i32, %c64_i32, %c64_i32, %c3_i32 {c_matrix_init_val = true} : memref<?xf32, 5>, memref<1xf16, 3>, memref<1xf16, 4>
                asccube.release_buf m, %308, 0 {cannir.buffer.generated_sync}
                asccube.release_buf m, %304, 0 {cannir.buffer.generated_sync}
                asccube.release_buf m, %c22_i64, 0 {cannir.buffer.generated_sync}
                asccube.get_buf fixpipe, %c2_i64, 0 {cannir.buffer.generated_sync}
                asccube.get_buf fixpipe, %c22_i64, 0 {cannir.buffer.generated_sync}
                asccube.copy_l0c2l1 %245[%c0], %244[%c0], %c64_i32, %c64_i32, %c1024_i32, %c64_i32, %c3_i32 {relu_en = false} : memref<?xf16, 2>, memref<?xf32, 5>
                asccube.release_buf fixpipe, %c22_i64, 0 {cannir.buffer.generated_sync}
                asccube.release_buf fixpipe, %c2_i64, 0 {cannir.buffer.generated_sync}
                %311 = arith.cmpi eq, %306, %c0_i64 : i64
                %312 = arith.select %311, %211, %212 : i64
                %313 = arith.select %311, %c17_i64, %c18_i64 : i64
                %314 = arith.addi %306, %c1_i64 overflow<nsw> : i64
                %315 = arith.remui %314, %c2_i64 : i64
                %316 = arith.select %311, %217, %218 : i64
                %317 = arith.select %311, %c19_i64, %c20_i64 : i64
                %318 = ascscalar.bind_local_ptr %312 : memref<1xf16, 3>
                asccube.get_buf mte1, %313, 0 {cannir.buffer.generated_sync}
                asccube.copy_l12l0a %318[%c0], %242[%c0], %c0_i32, %c0_i32, %c4_i32, %c4_i32, %c4_i32, %c4_i32 : memref<1xf16, 3>, memref<?xf16, 2>
                %319 = ascscalar.bind_local_ptr %316 : memref<1xf16, 4>
                asccube.get_buf mte1, %317, 0 {cannir.buffer.generated_sync}
                asccube.copy_l12l0b %319[%c0], %242[%c0], %c0_i32, %c0_i32, %c4_i32, %c4_i32, %c4_i32, %c4_i32 : memref<1xf16, 4>, memref<?xf16, 2>
                asccube.get_buf mte1, %c1_i64, 0 {cannir.buffer.generated_sync}
                asccube.copy_l12l0a %318[%c1280], %264[%c0], %c1_i32, %c1_i32, %c1_i32, %c1_i32, %c4_i32, %c4_i32 : memref<1xf16, 3>, memref<?xf16, 2>
                asccube.copy_l12l0a %318[%c3840], %264[%c0], %c3_i32, %c3_i32, %c1_i32, %c1_i32, %c4_i32, %c4_i32 : memref<1xf16, 3>, memref<?xf16, 2>
                asccube.release_buf mte1, %c1_i64, 0 {cannir.buffer.generated_sync}
                asccube.release_buf mte1, %313, 0 {cannir.buffer.generated_sync}
                asccube.get_buf mte1, %c2_i64, 0 {cannir.buffer.generated_sync}
                asccube.copy_l12l0b %319[%c1024], %245[%c0], %c1_i32, %c0_i32, %c1_i32, %c1_i32, %c4_i32, %c4_i32 {transpose = true} : memref<1xf16, 4>, memref<?xf16, 2>
                asccube.copy_l12l0b %319[%c3584], %245[%c0], %c3_i32, %c2_i32, %c1_i32, %c1_i32, %c4_i32, %c4_i32 {transpose = true} : memref<1xf16, 4>, memref<?xf16, 2>
                asccube.release_buf mte1, %c2_i64, 0 {cannir.buffer.generated_sync}
                asccube.release_buf mte1, %317, 0 {cannir.buffer.generated_sync}
                asccube.get_buf m, %c21_i64, 0 {cannir.buffer.generated_sync}
                asccube.get_buf m, %313, 0 {cannir.buffer.generated_sync}
                asccube.get_buf m, %317, 0 {cannir.buffer.generated_sync}
                asccube.mmad %232[%c0], %318, %319, %c64_i32, %c64_i32, %c64_i32, %c3_i32 {c_matrix_init_val = false} : memref<?xf32, 5>, memref<1xf16, 3>, memref<1xf16, 4>
                asccube.release_buf m, %317, 0 {cannir.buffer.generated_sync}
                asccube.release_buf m, %313, 0 {cannir.buffer.generated_sync}
                asccube.release_buf m, %c21_i64, 0 {cannir.buffer.generated_sync}
                asccube.get_buf fixpipe, %c1_i64, 0 {cannir.buffer.generated_sync}
                asccube.get_buf fixpipe, %c21_i64, 0 {cannir.buffer.generated_sync}
                asccube.copy_l0c2l1 %264[%c0], %232[%c0], %c64_i32, %c64_i32, %c1024_i32, %c64_i32, %c3_i32 {relu_en = false} : memref<?xf16, 2>, memref<?xf32, 5>
                asccube.release_buf fixpipe, %c21_i64, 0 {cannir.buffer.generated_sync}
                asccube.release_buf fixpipe, %c1_i64, 0 {cannir.buffer.generated_sync}
                %320 = arith.cmpi eq, %315, %c0_i64 : i64
                %321 = arith.select %320, %211, %212 : i64
                %322 = arith.select %320, %c17_i64, %c18_i64 : i64
                %323 = arith.addi %315, %c1_i64 overflow<nsw> : i64
                %324 = arith.remui %323, %c2_i64 : i64
                %325 = arith.select %320, %217, %218 : i64
                %326 = arith.select %320, %c19_i64, %c20_i64 : i64
                %327 = ascscalar.bind_local_ptr %321 : memref<1xf16, 3>
                asccube.get_buf mte1, %322, 0 {cannir.buffer.generated_sync}
                asccube.get_buf mte1, %c5_i64, 0 {cannir.buffer.generated_sync}
                asccube.copy_l12l0a %327[%c0], %242[%c0], %c0_i32, %c0_i32, %c4_i32, %c4_i32, %c4_i32, %c4_i32 : memref<1xf16, 3>, memref<?xf16, 2>
                %328 = ascscalar.bind_local_ptr %325 : memref<1xf16, 4>
                asccube.get_buf mte1, %326, 0 {cannir.buffer.generated_sync}
                asccube.copy_l12l0b %328[%c0], %242[%c0], %c0_i32, %c0_i32, %c4_i32, %c4_i32, %c4_i32, %c4_i32 : memref<1xf16, 4>, memref<?xf16, 2>
                asccube.copy_l12l0a %327[%c0], %233[%c0], %c0_i32, %c0_i32, %c4_i32, %c4_i32, %c4_i32, %c4_i32 : memref<1xf16, 3>, memref<?xf16, 2>
                asccube.release_buf mte1, %322, 0 {cannir.buffer.generated_sync}
                asccube.get_buf mte1, %c1_i64, 0 {cannir.buffer.generated_sync}
                asccube.copy_l12l0b %328[%c0], %264[%c0], %c0_i32, %c0_i32, %c1_i32, %c2_i32, %c4_i32, %c4_i32 {transpose = true} : memref<1xf16, 4>, memref<?xf16, 2>
                asccube.copy_l12l0b %328[%c1024], %264[%c256], %c0_i32, %c0_i32, %c1_i32, %c2_i32, %c4_i32, %c4_i32 {transpose = true} : memref<1xf16, 4>, memref<?xf16, 2>
                asccube.release_buf mte1, %326, 0 {cannir.buffer.generated_sync}
                %329 = arith.cmpi eq, %324, %c0_i64 : i64
                %330 = arith.select %329, %211, %212 : i64
                %331 = arith.select %329, %c17_i64, %c18_i64 : i64
                %332 = arith.addi %324, %c1_i64 overflow<nsw> : i64
                %333 = arith.remui %332, %c2_i64 : i64
                %334 = arith.select %329, %217, %218 : i64
                %335 = arith.select %329, %c19_i64, %c20_i64 : i64
                %336 = ascscalar.bind_local_ptr %330 : memref<1xf16, 3>
                asccube.get_buf mte1, %331, 0 {cannir.buffer.generated_sync}
                asccube.copy_l12l0a %336[%c0], %242[%c0], %c0_i32, %c0_i32, %c4_i32, %c4_i32, %c4_i32, %c4_i32 : memref<1xf16, 3>, memref<?xf16, 2>
                %337 = ascscalar.bind_local_ptr %334 : memref<1xf16, 4>
                asccube.get_buf mte1, %335, 0 {cannir.buffer.generated_sync}
                asccube.copy_l12l0b %337[%c0], %242[%c0], %c0_i32, %c0_i32, %c4_i32, %c4_i32, %c4_i32, %c4_i32 : memref<1xf16, 4>, memref<?xf16, 2>
                asccube.release_buf mte1, %c5_i64, 0 {cannir.buffer.generated_sync}
                asccube.release_buf mte1, %335, 0 {cannir.buffer.generated_sync}
                asccube.copy_l12l0a %336[%c0], %264[%c0], %c0_i32, %c0_i32, %c4_i32, %c4_i32, %c4_i32, %c4_i32 : memref<1xf16, 3>, memref<?xf16, 2>
                asccube.release_buf mte1, %c1_i64, 0 {cannir.buffer.generated_sync}
                asccube.release_buf mte1, %331, 0 {cannir.buffer.generated_sync}
                asccube.get_buf m, %c22_i64, 0 {cannir.buffer.generated_sync}
                asccube.get_buf m, %322, 0 {cannir.buffer.generated_sync}
                asccube.get_buf m, %326, 0 {cannir.buffer.generated_sync}
                asccube.mmad %244[%c0], %327, %328, %c64_i32, %c64_i32, %c64_i32, %c3_i32 {c_matrix_init_val = true} : memref<?xf32, 5>, memref<1xf16, 3>, memref<1xf16, 4>
                asccube.release_buf m, %326, 0 {cannir.buffer.generated_sync}
                asccube.release_buf m, %322, 0 {cannir.buffer.generated_sync}
                asccube.release_buf m, %c22_i64, 0 {cannir.buffer.generated_sync}
                asccube.get_buf fixpipe, %c2_i64, 0 {cannir.buffer.generated_sync}
                asccube.get_buf fixpipe, %c22_i64, 0 {cannir.buffer.generated_sync}
                asccube.copy_l0c2l1 %245[%c0], %244[%c0], %c64_i32, %c64_i32, %c1024_i32, %c64_i32, %c3_i32 {relu_en = false} : memref<?xf16, 2>, memref<?xf32, 5>
                asccube.release_buf fixpipe, %c22_i64, 0 {cannir.buffer.generated_sync}
                asccube.release_buf fixpipe, %c2_i64, 0 {cannir.buffer.generated_sync}
                asccube.get_buf mte1, %335, 0 {cannir.buffer.generated_sync}
                asccube.get_buf mte1, %c2_i64, 0 {cannir.buffer.generated_sync}
                asccube.copy_l12l0b %337[%c2048], %245[%c512], %c0_i32, %c0_i32, %c1_i32, %c2_i32, %c4_i32, %c4_i32 {transpose = true} : memref<1xf16, 4>, memref<?xf16, 2>
                asccube.copy_l12l0b %337[%c3072], %245[%c768], %c0_i32, %c0_i32, %c1_i32, %c2_i32, %c4_i32, %c4_i32 {transpose = true} : memref<1xf16, 4>, memref<?xf16, 2>
                asccube.release_buf mte1, %c2_i64, 0 {cannir.buffer.generated_sync}
                asccube.release_buf mte1, %335, 0 {cannir.buffer.generated_sync}
                asccube.get_buf m, %c21_i64, 0 {cannir.buffer.generated_sync}
                asccube.get_buf m, %331, 0 {cannir.buffer.generated_sync}
                asccube.get_buf m, %335, 0 {cannir.buffer.generated_sync}
                asccube.mmad %232[%c0], %336, %337, %c64_i32, %c64_i32, %c64_i32, %c3_i32 {c_matrix_init_val = false} : memref<?xf32, 5>, memref<1xf16, 3>, memref<1xf16, 4>
                asccube.release_buf m, %335, 0 {cannir.buffer.generated_sync}
                asccube.release_buf m, %331, 0 {cannir.buffer.generated_sync}
                asccube.release_buf m, %c21_i64, 0 {cannir.buffer.generated_sync}
                %338 = ascscalar.bitcast_memref %alloc_37 byte_offset = %c0_i64 : memref<4096xbf16, 2> -> memref<?xbf16, 2>
                asccube.get_buf fixpipe, %c21_i64, 0 {cannir.buffer.generated_sync}
                asccube.copy_l0c2l1 %338[%c0], %232[%c0], %c64_i32, %c64_i32, %c1024_i32, %c64_i32, %c3_i32 {relu_en = false} : memref<?xbf16, 2>, memref<?xf32, 5>
                asccube.release_buf fixpipe, %c21_i64, 0 {cannir.buffer.generated_sync}
                scf.yield %333, %209 : i64, i64
              } else {
                scf.yield %arg28, %arg30 : i64, i64
              }
              %112 = arith.remsi %arg25, %c2_i64 : i64
              %113 = arith.cmpi ne, %112, %c0_i64 : i64
              %114 = arith.cmpi slt, %112, %c0_i64 : i64
              %115 = arith.andi %113, %114 : i1
              %116 = arith.select %115, %c2_i64, %c0_i64 : i64
              %117 = arith.addi %112, %116 overflow<nsw> : i64
              %118 = arith.extsi %81 : i32 to i64
              %119 = arith.remsi %arg24, %118 : i64
              %120 = arith.cmpi ne, %119, %c0_i64 : i64
              %121 = arith.cmpi slt, %119, %c0_i64 : i64
              %122 = arith.cmpi slt, %118, %c0_i64 : i64
              %123 = arith.xori %121, %122 : i1
              %124 = arith.andi %120, %123 : i1
              %125 = arith.divsi %arg24, %118 : i64
              %126 = arith.extui %124 : i1 to i64
              %127 = arith.subi %125, %126 : i64
              %128 = arith.muli %127, %118 : i64
              %129 = arith.subi %arg24, %128 : i64
              %130 = arith.extsi %78 : i32 to i64
              %131 = cannir.extract_layout %arg1 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
              %132 = cannir.extract_shape %131 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 128)>
              %133 = cannir.extract_dim %132[1] : !cannir.shape<(-1, -1, -1, 128)>
              %134 = arith.maxsi %133, %c1_i64 : i64
              %135 = arith.addi %129, %c1_i64 : i64
              %136 = arith.cmpi eq, %135, %118 : i64
              %137 = arith.select %97, %136, %false : i1
              %138 = arith.select %137, %92, %c64_i64 : i64
              %139 = arith.index_cast %117 : i64 to index
              cannir.array_store %arg24, %100[%139] : <2xi64>
              cannir.array_store %138, %101[%139] : <2xi64>
              %140 = arith.divsi %75, %c64_i64 : i64
              %141 = arith.cmpi eq, %arg29, %c0_i64 : i64
              %142 = arith.addi %arg29, %c1_i64 overflow<nsw> : i64
              %143 = arith.remui %142, %c2_i64 : i64
              %144 = ascscalar.bitcast_memref %alloc_51 byte_offset = %c0_i64 : memref<2048xbf16, 2> -> memref<?xbf16, 2>
              %145 = ascscalar.bitcast_memref %alloc_14 byte_offset = %c0_i64 : memref<2048xbf16, 3> -> memref<?xbf16, 3>
              asccube.sync_intra_wait mte1, %c5_i64 {cannir.sync.banks = 3 : i32}
              asccube.sync_intra_wait mte1, %c21_i64 {cannir.sync.banks = 3 : i32}
              asccube.get_buf mte1, %c13_i64, 0 {cannir.buffer.generated_sync}
              asccube.copy_l12l0a %145[%c0], %144[%c0], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c1_i32, %c1_i32 : memref<?xbf16, 3>, memref<?xbf16, 2>
              asccube.release_buf mte1, %c13_i64, 0 {cannir.buffer.generated_sync}
              asccube.sync_intra_arrive mte1, %c5_i64 {cannir.sync.banks = 3 : i32}
              asccube.sync_intra_arrive mte1, %c21_i64 {cannir.sync.banks = 3 : i32}
              %146 = ascscalar.bitcast_memref %alloc_50 byte_offset = %c0_i64 : memref<2048xbf16, 2> -> memref<?xbf16, 2>
              %147 = ascscalar.bitcast_memref %alloc_15 byte_offset = %c0_i64 : memref<2048xbf16, 4> -> memref<?xbf16, 4>
              asccube.sync_intra_wait mte1, %c6_i64 {cannir.sync.banks = 3 : i32}
              asccube.sync_intra_wait mte1, %c22_i64 {cannir.sync.banks = 3 : i32}
              asccube.get_buf mte1, %c15_i64, 0 {cannir.buffer.generated_sync}
              asccube.copy_l12l0b %147[%c0], %146[%c0], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c1_i32, %c1_i32 : memref<?xbf16, 4>, memref<?xbf16, 2>
              asccube.release_buf mte1, %c15_i64, 0 {cannir.buffer.generated_sync}
              asccube.sync_intra_arrive mte1, %c6_i64 {cannir.sync.banks = 3 : i32}
              asccube.sync_intra_arrive mte1, %c22_i64 {cannir.sync.banks = 3 : i32}
              %148 = ascscalar.bitcast_memref %alloc_12 byte_offset = %c0_i64 : memref<4096xf32, 5> -> memref<?xf32, 5>
              asccube.get_buf m, %c21_i64, 0 {cannir.buffer.generated_sync}
              asccube.get_buf m, %c13_i64, 0 {cannir.buffer.generated_sync}
              asccube.get_buf m, %c15_i64, 0 {cannir.buffer.generated_sync}
              asccube.mmad %148[%c0], %145, %147, %c16_i32, %c128_i32, %c16_i32, %c0_i32 {c_matrix_init_val = true} : memref<?xf32, 5>, memref<?xbf16, 3>, memref<?xbf16, 4>
              asccube.release_buf m, %c15_i64, 0 {cannir.buffer.generated_sync}
              asccube.release_buf m, %c13_i64, 0 {cannir.buffer.generated_sync}
              asccube.release_buf m, %c21_i64, 0 {cannir.buffer.generated_sync}
              %149 = ascscalar.bitcast_memref %alloc_49 byte_offset = %c0_i64 : memref<2048xbf16, 2> -> memref<?xbf16, 2>
              %150 = ascscalar.bitcast_memref %alloc_16 byte_offset = %c0_i64 : memref<2048xbf16, 3> -> memref<?xbf16, 3>
              asccube.sync_intra_wait mte1, %c7_i64 {cannir.sync.banks = 3 : i32}
              asccube.sync_intra_wait mte1, %c23_i64 {cannir.sync.banks = 3 : i32}
              asccube.get_buf mte1, %c14_i64, 0 {cannir.buffer.generated_sync}
              asccube.copy_l12l0a %150[%c0], %149[%c0], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c1_i32, %c1_i32 : memref<?xbf16, 3>, memref<?xbf16, 2>
              asccube.release_buf mte1, %c14_i64, 0 {cannir.buffer.generated_sync}
              asccube.sync_intra_arrive mte1, %c7_i64 {cannir.sync.banks = 3 : i32}
              asccube.sync_intra_arrive mte1, %c23_i64 {cannir.sync.banks = 3 : i32}
              %151 = ascscalar.bitcast_memref %alloc_13 byte_offset = %c0_i64 : memref<4096xf32, 5> -> memref<?xf32, 5>
              asccube.get_buf m, %c22_i64, 0 {cannir.buffer.generated_sync}
              asccube.get_buf m, %c14_i64, 0 {cannir.buffer.generated_sync}
              asccube.get_buf m, %c15_i64, 0 {cannir.buffer.generated_sync}
              asccube.mmad %151[%c0], %150, %147, %c16_i32, %c128_i32, %c16_i32, %c0_i32 {c_matrix_init_val = true} : memref<?xf32, 5>, memref<?xbf16, 3>, memref<?xbf16, 4>
              asccube.release_buf m, %c15_i64, 0 {cannir.buffer.generated_sync}
              asccube.release_buf m, %c14_i64, 0 {cannir.buffer.generated_sync}
              asccube.release_buf m, %c22_i64, 0 {cannir.buffer.generated_sync}
              asccube.sync_intra_wait mte1, %c5_i64 {cannir.sync.banks = 3 : i32}
              asccube.sync_intra_wait mte1, %c21_i64 {cannir.sync.banks = 3 : i32}
              asccube.get_buf mte1, %c13_i64, 0 {cannir.buffer.generated_sync}
              asccube.copy_l12l0a %145[%c0], %144[%c0], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c1_i32, %c1_i32 : memref<?xbf16, 3>, memref<?xbf16, 2>
              asccube.release_buf mte1, %c13_i64, 0 {cannir.buffer.generated_sync}
              asccube.sync_intra_arrive mte1, %c5_i64 {cannir.sync.banks = 3 : i32}
              asccube.sync_intra_arrive mte1, %c21_i64 {cannir.sync.banks = 3 : i32}
              %152 = ascscalar.bitcast_memref %alloc_17 byte_offset = %c0_i64 : memref<2048xbf16, 4> -> memref<?xbf16, 4>
              asccube.sync_intra_wait mte1, %c6_i64 {cannir.sync.banks = 3 : i32}
              asccube.sync_intra_wait mte1, %c22_i64 {cannir.sync.banks = 3 : i32}
              asccube.get_buf mte1, %c16_i64, 0 {cannir.buffer.generated_sync}
              asccube.copy_l12l0b %152[%c0], %146[%c0], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c1_i32, %c1_i32 : memref<?xbf16, 4>, memref<?xbf16, 2>
              asccube.release_buf mte1, %c16_i64, 0 {cannir.buffer.generated_sync}
              asccube.sync_intra_arrive mte1, %c6_i64 {cannir.sync.banks = 3 : i32}
              asccube.sync_intra_arrive mte1, %c22_i64 {cannir.sync.banks = 3 : i32}
              %153 = ascscalar.memref_address %alloc_12 : memref<4096xf32, 5>
              %154 = arith.addi %153, %c1024_i64 : i64
              %155 = ascscalar.bind_local_ptr %154 : memref<1xf32, 5>
              asccube.get_buf m, %c13_i64, 0 {cannir.buffer.generated_sync}
              asccube.get_buf m, %c16_i64, 0 {cannir.buffer.generated_sync}
              asccube.mmad %155[%c0], %145, %152, %c16_i32, %c128_i32, %c16_i32, %c0_i32 {c_matrix_init_val = true} : memref<1xf32, 5>, memref<?xbf16, 3>, memref<?xbf16, 4>
              asccube.release_buf m, %c16_i64, 0 {cannir.buffer.generated_sync}
              asccube.release_buf m, %c13_i64, 0 {cannir.buffer.generated_sync}
              asccube.sync_intra_wait mte1, %c7_i64 {cannir.sync.banks = 3 : i32}
              asccube.sync_intra_wait mte1, %c23_i64 {cannir.sync.banks = 3 : i32}
              asccube.get_buf mte1, %c14_i64, 0 {cannir.buffer.generated_sync}
              asccube.copy_l12l0a %150[%c0], %149[%c0], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c1_i32, %c1_i32 : memref<?xbf16, 3>, memref<?xbf16, 2>
              asccube.release_buf mte1, %c14_i64, 0 {cannir.buffer.generated_sync}
              asccube.sync_intra_arrive mte1, %c7_i64 {cannir.sync.banks = 3 : i32}
              asccube.sync_intra_arrive mte1, %c23_i64 {cannir.sync.banks = 3 : i32}
              %156 = ascscalar.memref_address %alloc_13 : memref<4096xf32, 5>
              %157 = arith.addi %156, %c1024_i64 : i64
              %158 = ascscalar.bind_local_ptr %157 : memref<1xf32, 5>
              asccube.get_buf m, %c14_i64, 0 {cannir.buffer.generated_sync}
              asccube.get_buf m, %c16_i64, 0 {cannir.buffer.generated_sync}
              asccube.mmad %158[%c0], %150, %152, %c16_i32, %c128_i32, %c16_i32, %c0_i32 {c_matrix_init_val = true} : memref<1xf32, 5>, memref<?xbf16, 3>, memref<?xbf16, 4>
              asccube.release_buf m, %c16_i64, 0 {cannir.buffer.generated_sync}
              asccube.release_buf m, %c14_i64, 0 {cannir.buffer.generated_sync}
              asccube.sync_intra_wait mte1, %c5_i64 {cannir.sync.banks = 3 : i32}
              asccube.sync_intra_wait mte1, %c21_i64 {cannir.sync.banks = 3 : i32}
              asccube.get_buf mte1, %c15_i64, 0 {cannir.buffer.generated_sync}
              asccube.copy_l12l0b %147[%c0], %144[%c0], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c1_i32, %c1_i32 : memref<?xbf16, 4>, memref<?xbf16, 2>
              asccube.release_buf mte1, %c15_i64, 0 {cannir.buffer.generated_sync}
              asccube.sync_intra_arrive mte1, %c5_i64 {cannir.sync.banks = 3 : i32}
              asccube.sync_intra_arrive mte1, %c21_i64 {cannir.sync.banks = 3 : i32}
              %159 = arith.addi %153, %c5120_i64 : i64
              %160 = ascscalar.bind_local_ptr %159 : memref<1xf32, 5>
              asccube.get_buf m, %c13_i64, 0 {cannir.buffer.generated_sync}
              asccube.get_buf m, %c15_i64, 0 {cannir.buffer.generated_sync}
              asccube.mmad %160[%c0], %145, %147, %c16_i32, %c128_i32, %c16_i32, %c0_i32 {c_matrix_init_val = true} : memref<1xf32, 5>, memref<?xbf16, 3>, memref<?xbf16, 4>
              asccube.release_buf m, %c13_i64, 0 {cannir.buffer.generated_sync}
              %161 = arith.addi %156, %c5120_i64 : i64
              %162 = ascscalar.bind_local_ptr %161 : memref<1xf32, 5>
              asccube.get_buf m, %c14_i64, 0 {cannir.buffer.generated_sync}
              asccube.mmad %162[%c0], %150, %147, %c16_i32, %c128_i32, %c16_i32, %c0_i32 {c_matrix_init_val = true} : memref<1xf32, 5>, memref<?xbf16, 3>, memref<?xbf16, 4>
              asccube.release_buf m, %c15_i64, 0 {cannir.buffer.generated_sync}
              asccube.release_buf m, %c14_i64, 0 {cannir.buffer.generated_sync}
              asccube.sync_intra_wait mte1, %c6_i64 {cannir.sync.banks = 3 : i32}
              asccube.sync_intra_wait mte1, %c22_i64 {cannir.sync.banks = 3 : i32}
              asccube.get_buf mte1, %c13_i64, 0 {cannir.buffer.generated_sync}
              asccube.copy_l12l0a %145[%c0], %146[%c0], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c1_i32, %c1_i32 : memref<?xbf16, 3>, memref<?xbf16, 2>
              asccube.release_buf mte1, %c13_i64, 0 {cannir.buffer.generated_sync}
              asccube.sync_intra_arrive mte1, %c6_i64 {cannir.sync.banks = 3 : i32}
              asccube.sync_intra_arrive mte1, %c22_i64 {cannir.sync.banks = 3 : i32}
              asccube.sync_intra_wait mte1, %c7_i64 {cannir.sync.banks = 3 : i32}
              asccube.sync_intra_wait mte1, %c23_i64 {cannir.sync.banks = 3 : i32}
              asccube.get_buf mte1, %c16_i64, 0 {cannir.buffer.generated_sync}
              asccube.copy_l12l0b %152[%c0], %149[%c0], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c1_i32, %c1_i32 : memref<?xbf16, 4>, memref<?xbf16, 2>
              asccube.release_buf mte1, %c16_i64, 0 {cannir.buffer.generated_sync}
              asccube.sync_intra_arrive mte1, %c7_i64 {cannir.sync.banks = 3 : i32}
              asccube.sync_intra_arrive mte1, %c23_i64 {cannir.sync.banks = 3 : i32}
              %163 = arith.addi %153, %c2048_i64 : i64
              %164 = ascscalar.bind_local_ptr %163 : memref<1xf32, 5>
              asccube.get_buf m, %c13_i64, 0 {cannir.buffer.generated_sync}
              asccube.get_buf m, %c16_i64, 0 {cannir.buffer.generated_sync}
              asccube.mmad %164[%c0], %145, %152, %c16_i32, %c128_i32, %c16_i32, %c0_i32 {c_matrix_init_val = true} : memref<1xf32, 5>, memref<?xbf16, 3>, memref<?xbf16, 4>
              asccube.release_buf m, %c16_i64, 0 {cannir.buffer.generated_sync}
              asccube.release_buf m, %c13_i64, 0 {cannir.buffer.generated_sync}
              asccube.sync_intra_wait mte1, %c5_i64 {cannir.sync.banks = 3 : i32}
              asccube.sync_intra_wait mte1, %c21_i64 {cannir.sync.banks = 3 : i32}
              asccube.get_buf mte1, %c14_i64, 0 {cannir.buffer.generated_sync}
              asccube.copy_l12l0a %150[%c0], %144[%c0], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c1_i32, %c1_i32 : memref<?xbf16, 3>, memref<?xbf16, 2>
              asccube.release_buf mte1, %c14_i64, 0 {cannir.buffer.generated_sync}
              asccube.sync_intra_arrive mte1, %c5_i64 {cannir.sync.banks = 3 : i32}
              asccube.sync_intra_arrive mte1, %c21_i64 {cannir.sync.banks = 3 : i32}
              %165 = arith.addi %156, %c2048_i64 : i64
              %166 = ascscalar.bind_local_ptr %165 : memref<1xf32, 5>
              asccube.get_buf m, %c14_i64, 0 {cannir.buffer.generated_sync}
              asccube.get_buf m, %c16_i64, 0 {cannir.buffer.generated_sync}
              asccube.mmad %166[%c0], %150, %152, %c16_i32, %c128_i32, %c16_i32, %c0_i32 {c_matrix_init_val = true} : memref<1xf32, 5>, memref<?xbf16, 3>, memref<?xbf16, 4>
              asccube.release_buf m, %c16_i64, 0 {cannir.buffer.generated_sync}
              asccube.release_buf m, %c14_i64, 0 {cannir.buffer.generated_sync}
              asccube.sync_intra_wait mte1, %c6_i64 {cannir.sync.banks = 3 : i32}
              asccube.sync_intra_wait mte1, %c22_i64 {cannir.sync.banks = 3 : i32}
              asccube.get_buf mte1, %c15_i64, 0 {cannir.buffer.generated_sync}
              asccube.copy_l12l0b %147[%c0], %146[%c0], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c1_i32, %c1_i32 : memref<?xbf16, 4>, memref<?xbf16, 2>
              asccube.release_buf mte1, %c15_i64, 0 {cannir.buffer.generated_sync}
              asccube.sync_intra_arrive mte1, %c6_i64 {cannir.sync.banks = 3 : i32}
              asccube.sync_intra_arrive mte1, %c22_i64 {cannir.sync.banks = 3 : i32}
              %167 = arith.addi %153, %c6144_i64 : i64
              %168 = ascscalar.bind_local_ptr %167 : memref<1xf32, 5>
              asccube.get_buf m, %c15_i64, 0 {cannir.buffer.generated_sync}
              asccube.mmad %168[%c0], %145, %147, %c16_i32, %c128_i32, %c16_i32, %c0_i32 {c_matrix_init_val = true} : memref<1xf32, 5>, memref<?xbf16, 3>, memref<?xbf16, 4>
              %169 = arith.addi %156, %c6144_i64 : i64
              %170 = ascscalar.bind_local_ptr %169 : memref<1xf32, 5>
              asccube.mmad %170[%c0], %150, %147, %c16_i32, %c128_i32, %c16_i32, %c0_i32 {c_matrix_init_val = true} : memref<1xf32, 5>, memref<?xbf16, 3>, memref<?xbf16, 4>
              asccube.release_buf m, %c15_i64, 0 {cannir.buffer.generated_sync}
              asccube.sync_intra_wait mte1, %c7_i64 {cannir.sync.banks = 3 : i32}
              asccube.sync_intra_wait mte1, %c23_i64 {cannir.sync.banks = 3 : i32}
              asccube.get_buf mte1, %c16_i64, 0 {cannir.buffer.generated_sync}
              asccube.copy_l12l0b %152[%c0], %149[%c0], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c1_i32, %c1_i32 : memref<?xbf16, 4>, memref<?xbf16, 2>
              asccube.release_buf mte1, %c16_i64, 0 {cannir.buffer.generated_sync}
              asccube.sync_intra_arrive mte1, %c7_i64 {cannir.sync.banks = 3 : i32}
              asccube.sync_intra_arrive mte1, %c23_i64 {cannir.sync.banks = 3 : i32}
              %171 = arith.addi %153, %c10240_i64 : i64
              %172 = ascscalar.bind_local_ptr %171 : memref<1xf32, 5>
              asccube.get_buf m, %c13_i64, 0 {cannir.buffer.generated_sync}
              asccube.get_buf m, %c16_i64, 0 {cannir.buffer.generated_sync}
              asccube.mmad %172[%c0], %145, %152, %c16_i32, %c128_i32, %c16_i32, %c0_i32 {c_matrix_init_val = true} : memref<1xf32, 5>, memref<?xbf16, 3>, memref<?xbf16, 4>
              asccube.release_buf m, %c13_i64, 0 {cannir.buffer.generated_sync}
              %173 = arith.addi %156, %c10240_i64 : i64
              %174 = ascscalar.bind_local_ptr %173 : memref<1xf32, 5>
              asccube.get_buf m, %c14_i64, 0 {cannir.buffer.generated_sync}
              asccube.mmad %174[%c0], %150, %152, %c16_i32, %c128_i32, %c16_i32, %c0_i32 {c_matrix_init_val = true} : memref<1xf32, 5>, memref<?xbf16, 3>, memref<?xbf16, 4>
              asccube.release_buf m, %c16_i64, 0 {cannir.buffer.generated_sync}
              asccube.release_buf m, %c14_i64, 0 {cannir.buffer.generated_sync}
              asccube.sync_intra_wait mte1, %c5_i64 {cannir.sync.banks = 3 : i32}
              asccube.sync_intra_wait mte1, %c21_i64 {cannir.sync.banks = 3 : i32}
              asccube.get_buf mte1, %c13_i64, 0 {cannir.buffer.generated_sync}
              asccube.copy_l12l0a %145[%c0], %144[%c0], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c1_i32, %c1_i32 : memref<?xbf16, 3>, memref<?xbf16, 2>
              asccube.release_buf mte1, %c13_i64, 0 {cannir.buffer.generated_sync}
              asccube.sync_intra_arrive mte1, %c5_i64 {cannir.sync.banks = 3 : i32}
              asccube.sync_intra_arrive mte1, %c21_i64 {cannir.sync.banks = 3 : i32}
              asccube.sync_intra_wait mte1, %c6_i64 {cannir.sync.banks = 3 : i32}
              asccube.sync_intra_wait mte1, %c22_i64 {cannir.sync.banks = 3 : i32}
              asccube.get_buf mte1, %c15_i64, 0 {cannir.buffer.generated_sync}
              asccube.copy_l12l0b %147[%c0], %146[%c0], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c1_i32, %c1_i32 : memref<?xbf16, 4>, memref<?xbf16, 2>
              asccube.release_buf mte1, %c15_i64, 0 {cannir.buffer.generated_sync}
              asccube.sync_intra_arrive mte1, %c6_i64 {cannir.sync.banks = 3 : i32}
              asccube.sync_intra_arrive mte1, %c22_i64 {cannir.sync.banks = 3 : i32}
              %175 = arith.addi %153, %c3072_i64 : i64
              %176 = ascscalar.bind_local_ptr %175 : memref<1xf32, 5>
              asccube.get_buf m, %c13_i64, 0 {cannir.buffer.generated_sync}
              asccube.get_buf m, %c15_i64, 0 {cannir.buffer.generated_sync}
              asccube.mmad %176[%c0], %145, %147, %c16_i32, %c128_i32, %c16_i32, %c0_i32 {c_matrix_init_val = true} : memref<1xf32, 5>, memref<?xbf16, 3>, memref<?xbf16, 4>
              asccube.release_buf m, %c15_i64, 0 {cannir.buffer.generated_sync}
              asccube.release_buf m, %c13_i64, 0 {cannir.buffer.generated_sync}
              asccube.sync_intra_wait mte1, %c7_i64 {cannir.sync.banks = 3 : i32}
              asccube.sync_intra_wait mte1, %c23_i64 {cannir.sync.banks = 3 : i32}
              asccube.get_buf mte1, %c14_i64, 0 {cannir.buffer.generated_sync}
              asccube.copy_l12l0a %150[%c0], %149[%c0], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c1_i32, %c1_i32 : memref<?xbf16, 3>, memref<?xbf16, 2>
              asccube.release_buf mte1, %c14_i64, 0 {cannir.buffer.generated_sync}
              asccube.sync_intra_arrive mte1, %c7_i64 {cannir.sync.banks = 3 : i32}
              asccube.sync_intra_arrive mte1, %c23_i64 {cannir.sync.banks = 3 : i32}
              %177 = arith.addi %156, %c3072_i64 : i64
              %178 = ascscalar.bind_local_ptr %177 : memref<1xf32, 5>
              asccube.get_buf m, %c14_i64, 0 {cannir.buffer.generated_sync}
              asccube.get_buf m, %c15_i64, 0 {cannir.buffer.generated_sync}
              asccube.mmad %178[%c0], %150, %147, %c16_i32, %c128_i32, %c16_i32, %c0_i32 {c_matrix_init_val = true} : memref<1xf32, 5>, memref<?xbf16, 3>, memref<?xbf16, 4>
              asccube.release_buf m, %c15_i64, 0 {cannir.buffer.generated_sync}
              asccube.release_buf m, %c14_i64, 0 {cannir.buffer.generated_sync}
              asccube.sync_intra_wait mte1, %c5_i64 {cannir.sync.banks = 3 : i32}
              asccube.sync_intra_wait mte1, %c21_i64 {cannir.sync.banks = 3 : i32}
              asccube.get_buf mte1, %c16_i64, 0 {cannir.buffer.generated_sync}
              asccube.copy_l12l0b %152[%c0], %144[%c0], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c1_i32, %c1_i32 : memref<?xbf16, 4>, memref<?xbf16, 2>
              asccube.release_buf mte1, %c16_i64, 0 {cannir.buffer.generated_sync}
              asccube.sync_intra_arrive mte1, %c5_i64 {cannir.sync.banks = 3 : i32}
              asccube.sync_intra_arrive mte1, %c21_i64 {cannir.sync.banks = 3 : i32}
              %179 = arith.addi %153, %c7168_i64 : i64
              %180 = ascscalar.bind_local_ptr %179 : memref<1xf32, 5>
              asccube.get_buf m, %c16_i64, 0 {cannir.buffer.generated_sync}
              asccube.mmad %180[%c0], %145, %152, %c16_i32, %c128_i32, %c16_i32, %c0_i32 {c_matrix_init_val = true} : memref<1xf32, 5>, memref<?xbf16, 3>, memref<?xbf16, 4>
              %181 = arith.addi %156, %c7168_i64 : i64
              %182 = ascscalar.bind_local_ptr %181 : memref<1xf32, 5>
              asccube.mmad %182[%c0], %150, %152, %c16_i32, %c128_i32, %c16_i32, %c0_i32 {c_matrix_init_val = true} : memref<1xf32, 5>, memref<?xbf16, 3>, memref<?xbf16, 4>
              asccube.release_buf m, %c16_i64, 0 {cannir.buffer.generated_sync}
              asccube.sync_intra_wait mte1, %c6_i64 {cannir.sync.banks = 3 : i32}
              asccube.sync_intra_wait mte1, %c22_i64 {cannir.sync.banks = 3 : i32}
              asccube.get_buf mte1, %c15_i64, 0 {cannir.buffer.generated_sync}
              asccube.copy_l12l0b %147[%c0], %146[%c0], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c1_i32, %c1_i32 : memref<?xbf16, 4>, memref<?xbf16, 2>
              asccube.release_buf mte1, %c15_i64, 0 {cannir.buffer.generated_sync}
              asccube.sync_intra_arrive mte1, %c6_i64 {cannir.sync.banks = 3 : i32}
              asccube.sync_intra_arrive mte1, %c22_i64 {cannir.sync.banks = 3 : i32}
              %183 = arith.addi %153, %c11264_i64 : i64
              %184 = ascscalar.bind_local_ptr %183 : memref<1xf32, 5>
              asccube.get_buf m, %c15_i64, 0 {cannir.buffer.generated_sync}
              asccube.mmad %184[%c0], %145, %147, %c16_i32, %c128_i32, %c16_i32, %c0_i32 {c_matrix_init_val = true} : memref<1xf32, 5>, memref<?xbf16, 3>, memref<?xbf16, 4>
              %185 = arith.addi %156, %c11264_i64 : i64
              %186 = ascscalar.bind_local_ptr %185 : memref<1xf32, 5>
              asccube.mmad %186[%c0], %150, %147, %c16_i32, %c128_i32, %c16_i32, %c0_i32 {c_matrix_init_val = true} : memref<1xf32, 5>, memref<?xbf16, 3>, memref<?xbf16, 4>
              asccube.release_buf m, %c15_i64, 0 {cannir.buffer.generated_sync}
              asccube.sync_intra_wait mte1, %c7_i64 {cannir.sync.banks = 3 : i32}
              asccube.sync_intra_wait mte1, %c23_i64 {cannir.sync.banks = 3 : i32}
              asccube.get_buf mte1, %c16_i64, 0 {cannir.buffer.generated_sync}
              asccube.copy_l12l0b %152[%c0], %149[%c0], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c1_i32, %c1_i32 : memref<?xbf16, 4>, memref<?xbf16, 2>
              asccube.release_buf mte1, %c16_i64, 0 {cannir.buffer.generated_sync}
              asccube.sync_intra_arrive mte1, %c7_i64 {cannir.sync.banks = 3 : i32}
              asccube.sync_intra_arrive mte1, %c23_i64 {cannir.sync.banks = 3 : i32}
              %187 = arith.addi %153, %c15360_i64 : i64
              %188 = ascscalar.bind_local_ptr %187 : memref<1xf32, 5>
              asccube.get_buf m, %c21_i64, 0 {cannir.buffer.generated_sync}
              asccube.get_buf m, %c13_i64, 0 {cannir.buffer.generated_sync}
              asccube.get_buf m, %c16_i64, 0 {cannir.buffer.generated_sync}
              asccube.mmad %188[%c0], %145, %152, %c16_i32, %c128_i32, %c16_i32, %c0_i32 {c_matrix_init_val = true} : memref<1xf32, 5>, memref<?xbf16, 3>, memref<?xbf16, 4>
              asccube.release_buf m, %c13_i64, 0 {cannir.buffer.generated_sync}
              asccube.release_buf m, %c21_i64, 0 {cannir.buffer.generated_sync}
              %189 = arith.addi %156, %c15360_i64 : i64
              %190 = ascscalar.bind_local_ptr %189 : memref<1xf32, 5>
              asccube.get_buf m, %c22_i64, 0 {cannir.buffer.generated_sync}
              asccube.get_buf m, %c14_i64, 0 {cannir.buffer.generated_sync}
              asccube.mmad %190[%c0], %150, %152, %c16_i32, %c128_i32, %c16_i32, %c0_i32 {c_matrix_init_val = true} : memref<1xf32, 5>, memref<?xbf16, 3>, memref<?xbf16, 4>
              asccube.release_buf m, %c16_i64, 0 {cannir.buffer.generated_sync}
              asccube.release_buf m, %c14_i64, 0 {cannir.buffer.generated_sync}
              asccube.release_buf m, %c22_i64, 0 {cannir.buffer.generated_sync}
              %191 = ascscalar.memref_address %alloc_5 : memref<2048xf32, 1>
              %192 = ascscalar.memref_address %alloc_6 : memref<2048xf32, 1>
              %193 = arith.select %141, %191, %192 : i64
              %194 = arith.select %141, %c39_i64, %c40_i64 : i64
              %195 = ascscalar.bitcast_memref %alloc_7 byte_offset = %c0_i64 : memref<2048xf32, 1> -> memref<?xf32, 1>
              asccube.sync_intra_wait fixpipe, %c4_i64 {cannir.sync.banks = 3 : i32}
              asccube.sync_intra_wait fixpipe, %c20_i64 {cannir.sync.banks = 3 : i32}
              asccube.get_buf fixpipe, %c21_i64, 0 {cannir.buffer.generated_sync}
              asccube.copy_l0c2ub %195[%c0], %148[%c0], %c64_i32, %c64_i32, %c64_i32, %c64_i32, %c0_i32 sub_block_id %c0_i32 {dual_dst_ctl = 1 : i32, nz2nd_en = true, relu_en = false} : memref<?xf32, 1>, memref<?xf32, 5>
              asccube.release_buf fixpipe, %c21_i64, 0 {cannir.buffer.generated_sync}
              asccube.sync_intra_arrive fixpipe, %c4_i64 {cannir.sync.banks = 3 : i32}
              asccube.sync_intra_arrive fixpipe, %c20_i64 {cannir.sync.banks = 3 : i32}
              %196 = ascscalar.bind_local_ptr %193 : memref<1xf32, 1>
              %197 = arith.cmpi eq, %194, %c40_i64 : i64
              %198 = arith.select %197, %c3_i64, %c2_i64 : i64
              asccube.sync_intra_wait fixpipe, %198 {cannir.sync.banks = 3 : i32}
              %199 = arith.addi %198, %c16_i64 overflow<nsw> : i64
              asccube.sync_intra_wait fixpipe, %199 {cannir.sync.banks = 3 : i32}
              asccube.get_buf fixpipe, %c22_i64, 0 {cannir.buffer.generated_sync}
              asccube.copy_l0c2ub %196[%c0], %151[%c0], %c64_i32, %c64_i32, %c64_i32, %c64_i32, %c0_i32 sub_block_id %c0_i32 {dual_dst_ctl = 1 : i32, nz2nd_en = true, relu_en = false} : memref<1xf32, 1>, memref<?xf32, 5>
              asccube.release_buf fixpipe, %c22_i64, 0 {cannir.buffer.generated_sync}
              asccube.sync_intra_arrive fixpipe, %198 {cannir.sync.banks = 3 : i32}
              asccube.sync_intra_arrive fixpipe, %199 {cannir.sync.banks = 3 : i32}
              %200 = arith.addi %arg25, %c1_i64 : i64
              %201 = arith.cmpi slt, %108, %200 : i64
              %202 = arith.select %107, %201, %false : i1
              %203:2 = scf.if %202 -> (i64, i64) {
                %204 = arith.remsi %108, %c2_i64 : i64
                %205 = arith.index_cast %204 : i64 to index
                %206 = cannir.array_load %100[%205] : <2xi64>
                %207 = arith.remsi %206, %118 : i64
                %208 = arith.cmpi ne, %207, %c0_i64 : i64
                %209 = arith.cmpi slt, %207, %c0_i64 : i64
                %210 = arith.xori %209, %122 : i1
                %211 = arith.andi %208, %210 : i1
                %212 = arith.divsi %206, %118 : i64
                %213 = arith.extui %211 : i1 to i64
                %214 = arith.subi %212, %213 : i64
                %215 = arith.muli %214, %118 : i64
                %216 = arith.subi %206, %215 : i64
                %217 = arith.addi %130, %216 : i64
                %218 = arith.remsi %214, %134 : i64
                %219 = arith.cmpi slt, %218, %c0_i64 : i64
                %220 = arith.select %219, %134, %c0_i64 : i64
                %221 = arith.addi %218, %220 : i64
                %222 = arith.cmpi eq, %arg26, %c0_i64 : i64
                %223 = arith.addi %arg26, %c1_i64 overflow<nsw> : i64
                %224 = arith.remui %223, %c2_i64 : i64
                %225 = arith.minsi %217, %140 : i64
                %226 = arith.muli %225, %c64_i64 : i64
                %227 = arith.subi %75, %226 : i64
                %228 = arith.minsi %227, %c64_i64 : i64
                %229 = arith.cmpi sle, %217, %140 : i64
                %230 = arith.select %229, %228, %c0_i64 : i64
                %231 = arith.muli %217, %c64_i64 : i64
                %232 = arith.addi %231, %93 : i64
                %233 = arith.cmpi ugt, %230, %c0_i64 : i64
                %234 = ascscalar.memref_address %alloc_39 : memref<8192xbf16, 3>
                %235 = ascscalar.memref_address %alloc_38 : memref<8192xbf16, 3>
                %236 = arith.select %222, %234, %235 : i64
                %237 = arith.select %222, %c7_i64, %c8_i64 : i64
                %238 = ascscalar.bind_local_ptr %236 : memref<1xbf16, 3>
                %239 = ascscalar.bitcast_memref %alloc_37 byte_offset = %c0_i64 : memref<4096xbf16, 2> -> memref<?xbf16, 2>
                asccube.get_buf mte1, %237, 0 {cannir.buffer.generated_sync}
                asccube.copy_l12l0a %238[%c0], %239[%c0], %c0_i32, %c0_i32, %c4_i32, %c4_i32, %c4_i32, %c4_i32 : memref<1xbf16, 3>, memref<?xbf16, 2>
                asccube.release_buf mte1, %237, 0 {cannir.buffer.generated_sync}
                %240 = arith.cmpi eq, %arg27, %c0_i64 : i64
                %241 = ascscalar.memref_address %alloc_35 : memref<8192xbf16, 4>
                %242 = ascscalar.memref_address %alloc_34 : memref<8192xbf16, 4>
                %243 = arith.select %240, %241, %242 : i64
                %244 = arith.select %240, %c9_i64, %c10_i64 : i64
                %245 = arith.addi %arg27, %c1_i64 overflow<nsw> : i64
                %246 = arith.remui %245, %c2_i64 : i64
                %247 = ascscalar.bind_local_ptr %243 : memref<1xbf16, 4>
                %248 = ascscalar.bitcast_memref %alloc_36 byte_offset = %c0_i64 : memref<8192xbf16, 2> -> memref<?xbf16, 2>
                asccube.sync_intra_wait mte1, %c9_i64 {cannir.sync.banks = 3 : i32}
                asccube.sync_intra_wait mte1, %c25_i64 {cannir.sync.banks = 3 : i32}
                asccube.get_buf mte1, %244, 0 {cannir.buffer.generated_sync}
                asccube.copy_l12l0b %247[%c0], %248[%c0], %c0_i32, %c0_i32, %c4_i32, %c8_i32, %c4_i32, %c8_i32 {transpose = true} : memref<1xbf16, 4>, memref<?xbf16, 2>
                asccube.release_buf mte1, %244, 0 {cannir.buffer.generated_sync}
                asccube.sync_intra_arrive mte1, %c9_i64 {cannir.sync.banks = 3 : i32}
                asccube.sync_intra_arrive mte1, %c25_i64 {cannir.sync.banks = 3 : i32}
                %249 = ascscalar.memref_address %alloc_32 : memref<8192xf32, 5>
                %250 = ascscalar.memref_address %alloc_31 : memref<8192xf32, 5>
                %251 = arith.select %240, %249, %250 : i64
                %252 = arith.select %240, %c11_i64, %c12_i64 : i64
                %253 = ascscalar.bind_local_ptr %251 : memref<1xf32, 5>
                asccube.get_buf m, %252, 0 {cannir.buffer.generated_sync}
                asccube.get_buf m, %237, 0 {cannir.buffer.generated_sync}
                asccube.get_buf m, %244, 0 {cannir.buffer.generated_sync}
                asccube.mmad %253[%c0], %238, %247, %c64_i32, %c64_i32, %c128_i32, %c0_i32 {c_matrix_init_val = true} : memref<1xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
                asccube.release_buf m, %244, 0 {cannir.buffer.generated_sync}
                asccube.release_buf m, %252, 0 {cannir.buffer.generated_sync}
                %254 = cannir.extract_physical_layout %arg11 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                %255 = cannir.extract_stride %254 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                %256 = cannir.extract_dim %255[0] : !cannir.stride<(-1, -1, -1, 1)>
                %257 = arith.muli %94, %256 : i64
                %258 = arith.index_cast %257 : i64 to index
                %259 = cannir.extract_dim %255[1] : !cannir.stride<(-1, -1, -1, 1)>
                %260 = arith.muli %221, %259 : i64
                %261 = arith.index_cast %260 : i64 to index
                %262 = arith.addi %258, %261 : index
                %263 = cannir.extract_dim %255[2] : !cannir.stride<(-1, -1, -1, 1)>
                %264 = arith.muli %232, %263 : i64
                %265 = arith.index_cast %264 : i64 to index
                %266 = arith.addi %262, %265 : index
                %267 = cannir.extract_layout %arg11 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                %268 = cannir.extract_stride %267 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                %269 = cannir.extract_dim %268[2] : !cannir.stride<(-1, -1, -1, 1)>
                scf.if %233 {
                  asccube.get_buf fixpipe, %252, 0 {cannir.buffer.generated_sync}
                  %296 = arith.trunci %230 : i64 to i32
                  %297 = arith.trunci %269 : i64 to i32
                  asccube.copy_l0c2gm %alloc[%266], %253[%c0], %c128_i32, %296, %297, %c64_i32, %c0_i32 l2_cache_ctl %c0_i32 {nz2nd_en = true, relu_en = false} : memref<1xbf16>, memref<1xf32, 5>
                  asccube.release_buf fixpipe, %252, 0 {cannir.buffer.generated_sync}
                }
                %270 = arith.cmpi eq, %246, %c0_i64 : i64
                %271 = arith.select %270, %241, %242 : i64
                %272 = arith.select %270, %c9_i64, %c10_i64 : i64
                %273 = arith.addi %246, %c1_i64 overflow<nsw> : i64
                %274 = arith.remui %273, %c2_i64 : i64
                %275 = ascscalar.bind_local_ptr %271 : memref<1xbf16, 4>
                %276 = ascscalar.bitcast_memref %alloc_33 byte_offset = %c0_i64 : memref<8192xbf16, 2> -> memref<?xbf16, 2>
                asccube.sync_intra_wait mte1, %c8_i64 {cannir.sync.banks = 3 : i32}
                asccube.sync_intra_wait mte1, %c24_i64 {cannir.sync.banks = 3 : i32}
                asccube.get_buf mte1, %272, 0 {cannir.buffer.generated_sync}
                asccube.copy_l12l0b %275[%c0], %276[%c0], %c0_i32, %c0_i32, %c4_i32, %c8_i32, %c4_i32, %c8_i32 {transpose = true} : memref<1xbf16, 4>, memref<?xbf16, 2>
                asccube.release_buf mte1, %272, 0 {cannir.buffer.generated_sync}
                asccube.sync_intra_arrive mte1, %c8_i64 {cannir.sync.banks = 3 : i32}
                asccube.sync_intra_arrive mte1, %c24_i64 {cannir.sync.banks = 3 : i32}
                %277 = arith.select %270, %249, %250 : i64
                %278 = arith.select %270, %c11_i64, %c12_i64 : i64
                %279 = ascscalar.bind_local_ptr %277 : memref<1xf32, 5>
                asccube.get_buf m, %278, 0 {cannir.buffer.generated_sync}
                asccube.get_buf m, %272, 0 {cannir.buffer.generated_sync}
                asccube.mmad %279[%c0], %238, %275, %c64_i32, %c64_i32, %c128_i32, %c0_i32 {c_matrix_init_val = true} : memref<1xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
                asccube.release_buf m, %272, 0 {cannir.buffer.generated_sync}
                asccube.release_buf m, %237, 0 {cannir.buffer.generated_sync}
                asccube.release_buf m, %278, 0 {cannir.buffer.generated_sync}
                %280 = cannir.extract_physical_layout %arg13 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                %281 = cannir.extract_stride %280 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                %282 = cannir.extract_dim %281[0] : !cannir.stride<(-1, -1, -1, 1)>
                %283 = arith.muli %94, %282 : i64
                %284 = arith.index_cast %283 : i64 to index
                %285 = cannir.extract_dim %281[1] : !cannir.stride<(-1, -1, -1, 1)>
                %286 = arith.muli %221, %285 : i64
                %287 = arith.index_cast %286 : i64 to index
                %288 = arith.addi %284, %287 : index
                %289 = cannir.extract_dim %281[2] : !cannir.stride<(-1, -1, -1, 1)>
                %290 = arith.muli %232, %289 : i64
                %291 = arith.index_cast %290 : i64 to index
                %292 = arith.addi %288, %291 : index
                %293 = cannir.extract_layout %arg13 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                %294 = cannir.extract_stride %293 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                %295 = cannir.extract_dim %294[2] : !cannir.stride<(-1, -1, -1, 1)>
                scf.if %233 {
                  asccube.get_buf fixpipe, %278, 0 {cannir.buffer.generated_sync}
                  %296 = arith.trunci %230 : i64 to i32
                  %297 = arith.trunci %295 : i64 to i32
                  asccube.copy_l0c2gm %alloc_0[%292], %279[%c0], %c128_i32, %296, %297, %c64_i32, %c0_i32 l2_cache_ctl %c0_i32 {nz2nd_en = true, relu_en = false} : memref<1xbf16>, memref<1xf32, 5>
                  asccube.release_buf fixpipe, %278, 0 {cannir.buffer.generated_sync}
                }
                scf.yield %224, %274 : i64, i64
              } else {
                scf.yield %arg26, %arg27 : i64, i64
              }
              scf.yield %200, %203#0, %203#1, %111#0, %143, %111#1 : i64, i64, i64, i64, i64, i64
            }
            %103 = arith.cmpi sge, %102#0, %c1_i64 : i64
            %104 = arith.subi %102#0, %c1_i64 : i64
            %105 = arith.cmpi slt, %104, %102#0 : i64
            %106 = arith.select %103, %105, %false : i1
            scf.if %106 {
              %107 = arith.remsi %104, %c2_i64 : i64
              %108 = arith.index_cast %107 : i64 to index
              %109 = arith.extsi %81 : i32 to i64
              %110 = arith.cmpi slt, %109, %c0_i64 : i64
              %111 = arith.extsi %78 : i32 to i64
              %112 = cannir.extract_layout %arg1 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
              %113 = cannir.extract_shape %112 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 128)>
              %114 = cannir.extract_dim %113[1] : !cannir.shape<(-1, -1, -1, 128)>
              %115 = arith.maxsi %114, %c1_i64 : i64
              %116 = ascscalar.bitcast_memref %alloc_48 byte_offset = %c0_i64 : memref<4096xf16, 2> -> memref<?xf16, 2>
              asccube.get_buf mte2, %c5_i64, 0 {cannir.buffer.generated_sync}
              asccube.fill_l1 %alloc_11[%c0], %c0_i32, %c256_i32, %c1_i32, %c0_i32 : memref<4096xf16, 2>, i32
              asccube.release_buf mte2, %c5_i64, 0 {cannir.buffer.generated_sync}
              %117 = arith.cmpi eq, %102#3, %c0_i64 : i64
              %118 = ascscalar.memref_address %alloc_45 : memref<4096xf16, 3>
              %119 = ascscalar.memref_address %alloc_44 : memref<4096xf16, 3>
              %120 = arith.select %117, %118, %119 : i64
              %121 = arith.select %117, %c17_i64, %c18_i64 : i64
              %122 = arith.addi %102#3, %c1_i64 overflow<nsw> : i64
              %123 = arith.remui %122, %c2_i64 : i64
              %124 = ascscalar.memref_address %alloc_43 : memref<4096xf16, 4>
              %125 = ascscalar.memref_address %alloc_42 : memref<4096xf16, 4>
              %126 = arith.select %117, %124, %125 : i64
              %127 = arith.select %117, %c19_i64, %c20_i64 : i64
              %128 = ascscalar.bind_local_ptr %120 : memref<1xf16, 3>
              asccube.sync_intra_wait mte1, %c1_i64 {cannir.sync.banks = 3 : i32}
              asccube.sync_intra_wait mte1, %c17_i64 {cannir.sync.banks = 3 : i32}
              asccube.get_buf mte1, %121, 0 {cannir.buffer.generated_sync}
              asccube.copy_l12l0a %128[%c0], %116[%c0], %c0_i32, %c0_i32, %c4_i32, %c4_i32, %c4_i32, %c4_i32 : memref<1xf16, 3>, memref<?xf16, 2>
              asccube.release_buf mte1, %121, 0 {cannir.buffer.generated_sync}
              %129 = ascscalar.bind_local_ptr %126 : memref<1xf16, 4>
              asccube.get_buf mte1, %127, 0 {cannir.buffer.generated_sync}
              asccube.get_buf mte1, %c4_i64, 0 {cannir.buffer.generated_sync}
              asccube.copy_l12l0b %129[%c0], %99[%c0], %c0_i32, %c0_i32, %c4_i32, %c4_i32, %c4_i32, %c4_i32 : memref<1xf16, 4>, memref<?xf16, 2>
              asccube.release_buf mte1, %127, 0 {cannir.buffer.generated_sync}
              %130 = arith.cmpi eq, %123, %c0_i64 : i64
              %131 = arith.select %130, %118, %119 : i64
              %132 = arith.select %130, %c17_i64, %c18_i64 : i64
              %133 = arith.addi %123, %c1_i64 overflow<nsw> : i64
              %134 = arith.remui %133, %c2_i64 : i64
              %135 = arith.select %130, %124, %125 : i64
              %136 = arith.select %130, %c19_i64, %c20_i64 : i64
              %137 = ascscalar.bind_local_ptr %131 : memref<1xf16, 3>
              asccube.get_buf mte1, %132, 0 {cannir.buffer.generated_sync}
              asccube.copy_l12l0a %137[%c0], %99[%c0], %c0_i32, %c0_i32, %c4_i32, %c4_i32, %c4_i32, %c4_i32 : memref<1xf16, 3>, memref<?xf16, 2>
              asccube.release_buf mte1, %132, 0 {cannir.buffer.generated_sync}
              %138 = ascscalar.bind_local_ptr %135 : memref<1xf16, 4>
              asccube.get_buf mte1, %136, 0 {cannir.buffer.generated_sync}
              asccube.copy_l12l0b %138[%c0], %99[%c0], %c0_i32, %c0_i32, %c4_i32, %c4_i32, %c4_i32, %c4_i32 : memref<1xf16, 4>, memref<?xf16, 2>
              asccube.release_buf mte1, %c4_i64, 0 {cannir.buffer.generated_sync}
              asccube.release_buf mte1, %136, 0 {cannir.buffer.generated_sync}
              %139 = ascscalar.bitcast_memref %alloc_12 byte_offset = %c0_i64 : memref<4096xf32, 5> -> memref<?xf32, 5>
              asccube.get_buf m, %c21_i64, 0 {cannir.buffer.generated_sync}
              asccube.get_buf m, %121, 0 {cannir.buffer.generated_sync}
              asccube.get_buf m, %127, 0 {cannir.buffer.generated_sync}
              asccube.get_buf m, %132, 0 {cannir.buffer.generated_sync}
              asccube.get_buf m, %136, 0 {cannir.buffer.generated_sync}
              asccube.mmad %139[%c0], %128, %129, %c64_i32, %c64_i32, %c64_i32, %c2_i32 {c_matrix_init_val = true} : memref<?xf32, 5>, memref<1xf16, 3>, memref<1xf16, 4>
              asccube.mmad %139[%c0], %137, %138, %c64_i32, %c64_i32, %c64_i32, %c3_i32 {c_matrix_init_val = false} : memref<?xf32, 5>, memref<1xf16, 3>, memref<1xf16, 4>
              asccube.release_buf m, %136, 0 {cannir.buffer.generated_sync}
              asccube.release_buf m, %132, 0 {cannir.buffer.generated_sync}
              asccube.release_buf m, %127, 0 {cannir.buffer.generated_sync}
              asccube.release_buf m, %121, 0 {cannir.buffer.generated_sync}
              asccube.release_buf m, %c21_i64, 0 {cannir.buffer.generated_sync}
              %140 = ascscalar.bitcast_memref %alloc_46 byte_offset = %c0_i64 : memref<4096xf16, 2> -> memref<?xf16, 2>
              asccube.get_buf fixpipe, %c3_i64, 0 {cannir.buffer.generated_sync}
              asccube.get_buf fixpipe, %c21_i64, 0 {cannir.buffer.generated_sync}
              asccube.copy_l0c2l1 %140[%c0], %139[%c0], %c64_i32, %c64_i32, %c1024_i32, %c64_i32, %c3_i32 {relu_en = false} : memref<?xf16, 2>, memref<?xf32, 5>
              asccube.release_buf fixpipe, %c21_i64, 0 {cannir.buffer.generated_sync}
              asccube.release_buf fixpipe, %c3_i64, 0 {cannir.buffer.generated_sync}
              %141 = arith.cmpi eq, %134, %c0_i64 : i64
              %142 = arith.select %141, %118, %119 : i64
              %143 = arith.select %141, %c17_i64, %c18_i64 : i64
              %144 = arith.addi %134, %c1_i64 overflow<nsw> : i64
              %145 = arith.remui %144, %c2_i64 : i64
              %146 = arith.select %141, %124, %125 : i64
              %147 = arith.select %141, %c19_i64, %c20_i64 : i64
              %148 = ascscalar.bind_local_ptr %142 : memref<1xf16, 3>
              %149 = ascscalar.bitcast_memref %alloc_11 byte_offset = %c0_i64 : memref<4096xf16, 2> -> memref<?xf16, 2>
              asccube.get_buf mte1, %143, 0 {cannir.buffer.generated_sync}
              asccube.get_buf mte1, %c5_i64, 0 {cannir.buffer.generated_sync}
              asccube.copy_l12l0a %148[%c0], %149[%c0], %c0_i32, %c0_i32, %c4_i32, %c4_i32, %c4_i32, %c4_i32 : memref<1xf16, 3>, memref<?xf16, 2>
              %150 = ascscalar.bind_local_ptr %146 : memref<1xf16, 4>
              asccube.get_buf mte1, %147, 0 {cannir.buffer.generated_sync}
              asccube.copy_l12l0b %150[%c0], %149[%c0], %c0_i32, %c0_i32, %c4_i32, %c4_i32, %c4_i32, %c4_i32 : memref<1xf16, 4>, memref<?xf16, 2>
              asccube.release_buf mte1, %c5_i64, 0 {cannir.buffer.generated_sync}
              asccube.copy_l12l0a %148[%c0], %116[%c0], %c0_i32, %c0_i32, %c1_i32, %c4_i32, %c5_i32, %c5_i32 : memref<1xf16, 3>, memref<?xf16, 2>
              asccube.release_buf mte1, %143, 0 {cannir.buffer.generated_sync}
              asccube.copy_l12l0b %150[%c0], %116[%c0], %c0_i32, %c0_i32, %c1_i32, %c4_i32, %c5_i32, %c4_i32 {cannir.l1_l0b_transpose_repeat, transpose = true} : memref<1xf16, 4>, memref<?xf16, 2>
              asccube.release_buf mte1, %147, 0 {cannir.buffer.generated_sync}
              asccube.sync_intra_arrive mte1, %c1_i64 {cannir.sync.banks = 3 : i32}
              asccube.sync_intra_arrive mte1, %c17_i64 {cannir.sync.banks = 3 : i32}
              %151 = ascscalar.bitcast_memref %alloc_13 byte_offset = %c0_i64 : memref<4096xf32, 5> -> memref<?xf32, 5>
              asccube.get_buf m, %c22_i64, 0 {cannir.buffer.generated_sync}
              asccube.get_buf m, %143, 0 {cannir.buffer.generated_sync}
              asccube.get_buf m, %147, 0 {cannir.buffer.generated_sync}
              asccube.mmad %151[%c0], %148, %150, %c64_i32, %c64_i32, %c64_i32, %c3_i32 {c_matrix_init_val = true} : memref<?xf32, 5>, memref<1xf16, 3>, memref<1xf16, 4>
              asccube.release_buf m, %147, 0 {cannir.buffer.generated_sync}
              asccube.release_buf m, %143, 0 {cannir.buffer.generated_sync}
              asccube.release_buf m, %c22_i64, 0 {cannir.buffer.generated_sync}
              %152 = ascscalar.bitcast_memref %alloc_40 byte_offset = %c0_i64 : memref<4096xf16, 2> -> memref<?xf16, 2>
              asccube.get_buf fixpipe, %c2_i64, 0 {cannir.buffer.generated_sync}
              asccube.get_buf fixpipe, %c22_i64, 0 {cannir.buffer.generated_sync}
              asccube.copy_l0c2l1 %152[%c0], %151[%c0], %c64_i32, %c64_i32, %c1024_i32, %c64_i32, %c3_i32 {relu_en = false} : memref<?xf16, 2>, memref<?xf32, 5>
              asccube.release_buf fixpipe, %c22_i64, 0 {cannir.buffer.generated_sync}
              asccube.release_buf fixpipe, %c2_i64, 0 {cannir.buffer.generated_sync}
              %153 = arith.cmpi eq, %145, %c0_i64 : i64
              %154 = arith.select %153, %118, %119 : i64
              %155 = arith.select %153, %c17_i64, %c18_i64 : i64
              %156 = arith.addi %145, %c1_i64 overflow<nsw> : i64
              %157 = arith.remui %156, %c2_i64 : i64
              %158 = arith.select %153, %124, %125 : i64
              %159 = arith.select %153, %c19_i64, %c20_i64 : i64
              %160 = ascscalar.bind_local_ptr %154 : memref<1xf16, 3>
              asccube.get_buf mte1, %155, 0 {cannir.buffer.generated_sync}
              asccube.get_buf mte1, %c3_i64, 0 {cannir.buffer.generated_sync}
              asccube.copy_l12l0a %160[%c0], %140[%c0], %c0_i32, %c0_i32, %c1_i32, %c4_i32, %c5_i32, %c5_i32 : memref<1xf16, 3>, memref<?xf16, 2>
              asccube.release_buf mte1, %155, 0 {cannir.buffer.generated_sync}
              %161 = ascscalar.bind_local_ptr %158 : memref<1xf16, 4>
              asccube.get_buf mte1, %159, 0 {cannir.buffer.generated_sync}
              asccube.get_buf mte1, %c4_i64, 0 {cannir.buffer.generated_sync}
              asccube.copy_l12l0b %161[%c0], %99[%c0], %c0_i32, %c0_i32, %c1_i32, %c4_i32, %c5_i32, %c4_i32 {cannir.l1_l0b_transpose_repeat, transpose = true} : memref<1xf16, 4>, memref<?xf16, 2>
              asccube.release_buf mte1, %c4_i64, 0 {cannir.buffer.generated_sync}
              asccube.release_buf mte1, %159, 0 {cannir.buffer.generated_sync}
              %162 = arith.cmpi eq, %157, %c0_i64 : i64
              %163 = arith.select %162, %118, %119 : i64
              %164 = arith.select %162, %c17_i64, %c18_i64 : i64
              %165 = arith.addi %157, %c1_i64 overflow<nsw> : i64
              %166 = arith.remui %165, %c2_i64 : i64
              %167 = arith.select %162, %124, %125 : i64
              %168 = arith.select %162, %c19_i64, %c20_i64 : i64
              %169 = ascscalar.bind_local_ptr %163 : memref<1xf16, 3>
              asccube.get_buf mte1, %164, 0 {cannir.buffer.generated_sync}
              asccube.copy_l12l0a %169[%c0], %140[%c0], %c0_i32, %c0_i32, %c1_i32, %c4_i32, %c5_i32, %c5_i32 : memref<1xf16, 3>, memref<?xf16, 2>
              asccube.release_buf mte1, %c3_i64, 0 {cannir.buffer.generated_sync}
              asccube.release_buf mte1, %164, 0 {cannir.buffer.generated_sync}
              %170 = ascscalar.bind_local_ptr %167 : memref<1xf16, 4>
              asccube.get_buf mte1, %168, 0 {cannir.buffer.generated_sync}
              asccube.get_buf mte1, %c2_i64, 0 {cannir.buffer.generated_sync}
              asccube.copy_l12l0b %170[%c0], %152[%c0], %c0_i32, %c0_i32, %c1_i32, %c4_i32, %c5_i32, %c4_i32 {cannir.l1_l0b_transpose_repeat, transpose = true} : memref<1xf16, 4>, memref<?xf16, 2>
              asccube.release_buf mte1, %c2_i64, 0 {cannir.buffer.generated_sync}
              asccube.release_buf mte1, %168, 0 {cannir.buffer.generated_sync}
              asccube.get_buf m, %c21_i64, 0 {cannir.buffer.generated_sync}
              asccube.get_buf m, %155, 0 {cannir.buffer.generated_sync}
              asccube.get_buf m, %159, 0 {cannir.buffer.generated_sync}
              asccube.get_buf m, %164, 0 {cannir.buffer.generated_sync}
              asccube.get_buf m, %168, 0 {cannir.buffer.generated_sync}
              asccube.mmad %139[%c0], %160, %161, %c64_i32, %c64_i32, %c64_i32, %c2_i32 {c_matrix_init_val = true} : memref<?xf32, 5>, memref<1xf16, 3>, memref<1xf16, 4>
              asccube.mmad %139[%c0], %169, %170, %c64_i32, %c64_i32, %c64_i32, %c3_i32 {c_matrix_init_val = false} : memref<?xf32, 5>, memref<1xf16, 3>, memref<1xf16, 4>
              asccube.release_buf m, %168, 0 {cannir.buffer.generated_sync}
              asccube.release_buf m, %164, 0 {cannir.buffer.generated_sync}
              asccube.release_buf m, %159, 0 {cannir.buffer.generated_sync}
              asccube.release_buf m, %155, 0 {cannir.buffer.generated_sync}
              asccube.release_buf m, %c21_i64, 0 {cannir.buffer.generated_sync}
              %171 = ascscalar.bitcast_memref %alloc_41 byte_offset = %c0_i64 : memref<4096xf16, 2> -> memref<?xf16, 2>
              asccube.get_buf fixpipe, %c1_i64, 0 {cannir.buffer.generated_sync}
              asccube.get_buf fixpipe, %c21_i64, 0 {cannir.buffer.generated_sync}
              asccube.copy_l0c2l1 %171[%c0], %139[%c0], %c64_i32, %c64_i32, %c1024_i32, %c64_i32, %c3_i32 {relu_en = false} : memref<?xf16, 2>, memref<?xf32, 5>
              asccube.release_buf fixpipe, %c21_i64, 0 {cannir.buffer.generated_sync}
              asccube.release_buf fixpipe, %c1_i64, 0 {cannir.buffer.generated_sync}
              %172 = arith.cmpi eq, %166, %c0_i64 : i64
              %173 = arith.select %172, %118, %119 : i64
              %174 = arith.select %172, %c17_i64, %c18_i64 : i64
              %175 = arith.addi %166, %c1_i64 overflow<nsw> : i64
              %176 = arith.remui %175, %c2_i64 : i64
              %177 = arith.select %172, %124, %125 : i64
              %178 = arith.select %172, %c19_i64, %c20_i64 : i64
              %179 = ascscalar.bind_local_ptr %173 : memref<1xf16, 3>
              asccube.get_buf mte1, %174, 0 {cannir.buffer.generated_sync}
              asccube.copy_l12l0a %179[%c0], %152[%c0], %c0_i32, %c0_i32, %c1_i32, %c4_i32, %c5_i32, %c5_i32 : memref<1xf16, 3>, memref<?xf16, 2>
              asccube.release_buf mte1, %174, 0 {cannir.buffer.generated_sync}
              %180 = ascscalar.bind_local_ptr %177 : memref<1xf16, 4>
              asccube.get_buf mte1, %178, 0 {cannir.buffer.generated_sync}
              asccube.copy_l12l0b %180[%c0], %152[%c0], %c0_i32, %c0_i32, %c1_i32, %c4_i32, %c5_i32, %c4_i32 {cannir.l1_l0b_transpose_repeat, transpose = true} : memref<1xf16, 4>, memref<?xf16, 2>
              asccube.release_buf mte1, %178, 0 {cannir.buffer.generated_sync}
              asccube.get_buf m, %c22_i64, 0 {cannir.buffer.generated_sync}
              asccube.get_buf m, %174, 0 {cannir.buffer.generated_sync}
              asccube.get_buf m, %178, 0 {cannir.buffer.generated_sync}
              asccube.mmad %151[%c0], %179, %180, %c64_i32, %c64_i32, %c64_i32, %c3_i32 {c_matrix_init_val = true} : memref<?xf32, 5>, memref<1xf16, 3>, memref<1xf16, 4>
              asccube.release_buf m, %178, 0 {cannir.buffer.generated_sync}
              asccube.release_buf m, %174, 0 {cannir.buffer.generated_sync}
              asccube.release_buf m, %c22_i64, 0 {cannir.buffer.generated_sync}
              %181 = ascscalar.bitcast_memref %alloc_47 byte_offset = %c0_i64 : memref<4096xf16, 2> -> memref<?xf16, 2>
              asccube.get_buf fixpipe, %c0_i64, 0 {cannir.buffer.generated_sync}
              asccube.get_buf fixpipe, %c22_i64, 0 {cannir.buffer.generated_sync}
              asccube.copy_l0c2l1 %181[%c0], %151[%c0], %c64_i32, %c64_i32, %c1024_i32, %c64_i32, %c3_i32 {relu_en = false} : memref<?xf16, 2>, memref<?xf32, 5>
              asccube.release_buf fixpipe, %c22_i64, 0 {cannir.buffer.generated_sync}
              asccube.release_buf fixpipe, %c0_i64, 0 {cannir.buffer.generated_sync}
              %182 = arith.cmpi eq, %176, %c0_i64 : i64
              %183 = arith.select %182, %118, %119 : i64
              %184 = arith.select %182, %c17_i64, %c18_i64 : i64
              %185 = arith.addi %176, %c1_i64 overflow<nsw> : i64
              %186 = arith.remui %185, %c2_i64 : i64
              %187 = arith.select %182, %124, %125 : i64
              %188 = arith.select %182, %c19_i64, %c20_i64 : i64
              %189 = ascscalar.bind_local_ptr %183 : memref<1xf16, 3>
              asccube.get_buf mte1, %184, 0 {cannir.buffer.generated_sync}
              asccube.get_buf mte1, %c1_i64, 0 {cannir.buffer.generated_sync}
              asccube.copy_l12l0a %189[%c0], %171[%c0], %c0_i32, %c0_i32, %c1_i32, %c4_i32, %c5_i32, %c5_i32 : memref<1xf16, 3>, memref<?xf16, 2>
              asccube.release_buf mte1, %c1_i64, 0 {cannir.buffer.generated_sync}
              asccube.release_buf mte1, %184, 0 {cannir.buffer.generated_sync}
              %190 = ascscalar.bind_local_ptr %187 : memref<1xf16, 4>
              asccube.get_buf mte1, %188, 0 {cannir.buffer.generated_sync}
              asccube.get_buf mte1, %c0_i64, 0 {cannir.buffer.generated_sync}
              asccube.copy_l12l0b %190[%c0], %181[%c0], %c0_i32, %c0_i32, %c1_i32, %c4_i32, %c5_i32, %c4_i32 {cannir.l1_l0b_transpose_repeat, transpose = true} : memref<1xf16, 4>, memref<?xf16, 2>
              asccube.release_buf mte1, %c0_i64, 0 {cannir.buffer.generated_sync}
              asccube.release_buf mte1, %188, 0 {cannir.buffer.generated_sync}
              asccube.get_buf m, %c21_i64, 0 {cannir.buffer.generated_sync}
              asccube.get_buf m, %184, 0 {cannir.buffer.generated_sync}
              asccube.get_buf m, %188, 0 {cannir.buffer.generated_sync}
              asccube.mmad %139[%c0], %189, %190, %c64_i32, %c64_i32, %c64_i32, %c3_i32 {c_matrix_init_val = false} : memref<?xf32, 5>, memref<1xf16, 3>, memref<1xf16, 4>
              asccube.release_buf m, %188, 0 {cannir.buffer.generated_sync}
              asccube.release_buf m, %184, 0 {cannir.buffer.generated_sync}
              asccube.release_buf m, %c21_i64, 0 {cannir.buffer.generated_sync}
              asccube.get_buf fixpipe, %c1_i64, 0 {cannir.buffer.generated_sync}
              asccube.get_buf fixpipe, %c21_i64, 0 {cannir.buffer.generated_sync}
              asccube.copy_l0c2l1 %171[%c0], %139[%c0], %c64_i32, %c64_i32, %c1024_i32, %c64_i32, %c3_i32 {relu_en = false} : memref<?xf16, 2>, memref<?xf32, 5>
              asccube.release_buf fixpipe, %c21_i64, 0 {cannir.buffer.generated_sync}
              asccube.release_buf fixpipe, %c1_i64, 0 {cannir.buffer.generated_sync}
              %191 = arith.cmpi eq, %186, %c0_i64 : i64
              %192 = arith.select %191, %118, %119 : i64
              %193 = arith.select %191, %c17_i64, %c18_i64 : i64
              %194 = arith.addi %186, %c1_i64 overflow<nsw> : i64
              %195 = arith.remui %194, %c2_i64 : i64
              %196 = arith.select %191, %124, %125 : i64
              %197 = arith.select %191, %c19_i64, %c20_i64 : i64
              %198 = ascscalar.bind_local_ptr %192 : memref<1xf16, 3>
              asccube.get_buf mte1, %193, 0 {cannir.buffer.generated_sync}
              asccube.copy_l12l0a %198[%c0], %181[%c0], %c0_i32, %c0_i32, %c1_i32, %c4_i32, %c5_i32, %c5_i32 : memref<1xf16, 3>, memref<?xf16, 2>
              asccube.release_buf mte1, %193, 0 {cannir.buffer.generated_sync}
              %199 = ascscalar.bind_local_ptr %196 : memref<1xf16, 4>
              asccube.get_buf mte1, %197, 0 {cannir.buffer.generated_sync}
              asccube.copy_l12l0b %199[%c0], %181[%c0], %c0_i32, %c0_i32, %c1_i32, %c4_i32, %c5_i32, %c4_i32 {cannir.l1_l0b_transpose_repeat, transpose = true} : memref<1xf16, 4>, memref<?xf16, 2>
              asccube.release_buf mte1, %197, 0 {cannir.buffer.generated_sync}
              asccube.get_buf m, %c22_i64, 0 {cannir.buffer.generated_sync}
              asccube.get_buf m, %193, 0 {cannir.buffer.generated_sync}
              asccube.get_buf m, %197, 0 {cannir.buffer.generated_sync}
              asccube.mmad %151[%c0], %198, %199, %c64_i32, %c64_i32, %c64_i32, %c3_i32 {c_matrix_init_val = true} : memref<?xf32, 5>, memref<1xf16, 3>, memref<1xf16, 4>
              asccube.release_buf m, %197, 0 {cannir.buffer.generated_sync}
              asccube.release_buf m, %193, 0 {cannir.buffer.generated_sync}
              asccube.release_buf m, %c22_i64, 0 {cannir.buffer.generated_sync}
              asccube.get_buf fixpipe, %c2_i64, 0 {cannir.buffer.generated_sync}
              asccube.get_buf fixpipe, %c22_i64, 0 {cannir.buffer.generated_sync}
              asccube.copy_l0c2l1 %152[%c0], %151[%c0], %c64_i32, %c64_i32, %c1024_i32, %c64_i32, %c3_i32 {relu_en = false} : memref<?xf16, 2>, memref<?xf32, 5>
              asccube.release_buf fixpipe, %c22_i64, 0 {cannir.buffer.generated_sync}
              asccube.release_buf fixpipe, %c2_i64, 0 {cannir.buffer.generated_sync}
              %200 = arith.cmpi eq, %195, %c0_i64 : i64
              %201 = arith.select %200, %118, %119 : i64
              %202 = arith.select %200, %c17_i64, %c18_i64 : i64
              %203 = arith.addi %195, %c1_i64 overflow<nsw> : i64
              %204 = arith.remui %203, %c2_i64 : i64
              %205 = arith.select %200, %124, %125 : i64
              %206 = arith.select %200, %c19_i64, %c20_i64 : i64
              %207 = ascscalar.bind_local_ptr %201 : memref<1xf16, 3>
              asccube.get_buf mte1, %202, 0 {cannir.buffer.generated_sync}
              asccube.get_buf mte1, %c1_i64, 0 {cannir.buffer.generated_sync}
              asccube.copy_l12l0a %207[%c0], %171[%c0], %c0_i32, %c0_i32, %c1_i32, %c4_i32, %c5_i32, %c5_i32 : memref<1xf16, 3>, memref<?xf16, 2>
              asccube.release_buf mte1, %c1_i64, 0 {cannir.buffer.generated_sync}
              asccube.release_buf mte1, %202, 0 {cannir.buffer.generated_sync}
              %208 = ascscalar.bind_local_ptr %205 : memref<1xf16, 4>
              asccube.get_buf mte1, %206, 0 {cannir.buffer.generated_sync}
              asccube.get_buf mte1, %c2_i64, 0 {cannir.buffer.generated_sync}
              asccube.copy_l12l0b %208[%c0], %152[%c0], %c0_i32, %c0_i32, %c1_i32, %c4_i32, %c5_i32, %c4_i32 {cannir.l1_l0b_transpose_repeat, transpose = true} : memref<1xf16, 4>, memref<?xf16, 2>
              asccube.release_buf mte1, %c2_i64, 0 {cannir.buffer.generated_sync}
              asccube.release_buf mte1, %206, 0 {cannir.buffer.generated_sync}
              asccube.get_buf m, %c21_i64, 0 {cannir.buffer.generated_sync}
              asccube.get_buf m, %202, 0 {cannir.buffer.generated_sync}
              asccube.get_buf m, %206, 0 {cannir.buffer.generated_sync}
              asccube.mmad %139[%c0], %207, %208, %c64_i32, %c64_i32, %c64_i32, %c3_i32 {c_matrix_init_val = false} : memref<?xf32, 5>, memref<1xf16, 3>, memref<1xf16, 4>
              asccube.release_buf m, %206, 0 {cannir.buffer.generated_sync}
              asccube.release_buf m, %202, 0 {cannir.buffer.generated_sync}
              asccube.release_buf m, %c21_i64, 0 {cannir.buffer.generated_sync}
              asccube.get_buf fixpipe, %c1_i64, 0 {cannir.buffer.generated_sync}
              asccube.get_buf fixpipe, %c21_i64, 0 {cannir.buffer.generated_sync}
              asccube.copy_l0c2l1 %171[%c0], %139[%c0], %c64_i32, %c64_i32, %c1024_i32, %c64_i32, %c3_i32 {relu_en = false} : memref<?xf16, 2>, memref<?xf32, 5>
              asccube.release_buf fixpipe, %c21_i64, 0 {cannir.buffer.generated_sync}
              asccube.release_buf fixpipe, %c1_i64, 0 {cannir.buffer.generated_sync}
              %209 = arith.cmpi eq, %204, %c0_i64 : i64
              %210 = arith.select %209, %118, %119 : i64
              %211 = arith.select %209, %c17_i64, %c18_i64 : i64
              %212 = arith.addi %204, %c1_i64 overflow<nsw> : i64
              %213 = arith.remui %212, %c2_i64 : i64
              %214 = arith.select %209, %124, %125 : i64
              %215 = arith.select %209, %c19_i64, %c20_i64 : i64
              %216 = ascscalar.bind_local_ptr %210 : memref<1xf16, 3>
              asccube.get_buf mte1, %211, 0 {cannir.buffer.generated_sync}
              asccube.copy_l12l0a %216[%c0], %149[%c0], %c0_i32, %c0_i32, %c4_i32, %c4_i32, %c4_i32, %c4_i32 : memref<1xf16, 3>, memref<?xf16, 2>
              %217 = ascscalar.bind_local_ptr %214 : memref<1xf16, 4>
              asccube.get_buf mte1, %215, 0 {cannir.buffer.generated_sync}
              asccube.copy_l12l0b %217[%c0], %149[%c0], %c0_i32, %c0_i32, %c4_i32, %c4_i32, %c4_i32, %c4_i32 : memref<1xf16, 4>, memref<?xf16, 2>
              asccube.copy_l12l0a %216[%c256], %140[%c0], %c1_i32, %c0_i32, %c1_i32, %c1_i32, %c4_i32, %c4_i32 : memref<1xf16, 3>, memref<?xf16, 2>
              asccube.copy_l12l0a %216[%c2816], %140[%c0], %c3_i32, %c2_i32, %c1_i32, %c1_i32, %c4_i32, %c4_i32 : memref<1xf16, 3>, memref<?xf16, 2>
              asccube.release_buf mte1, %211, 0 {cannir.buffer.generated_sync}
              asccube.get_buf mte1, %c1_i64, 0 {cannir.buffer.generated_sync}
              asccube.copy_l12l0b %217[%c0], %171[%c0], %c0_i32, %c0_i32, %c1_i32, %c1_i32, %c4_i32, %c4_i32 {transpose = true} : memref<1xf16, 4>, memref<?xf16, 2>
              asccube.copy_l12l0b %217[%c2560], %171[%c0], %c2_i32, %c2_i32, %c1_i32, %c1_i32, %c4_i32, %c4_i32 {transpose = true} : memref<1xf16, 4>, memref<?xf16, 2>
              asccube.release_buf mte1, %c1_i64, 0 {cannir.buffer.generated_sync}
              asccube.release_buf mte1, %215, 0 {cannir.buffer.generated_sync}
              asccube.get_buf m, %c22_i64, 0 {cannir.buffer.generated_sync}
              asccube.get_buf m, %211, 0 {cannir.buffer.generated_sync}
              asccube.get_buf m, %215, 0 {cannir.buffer.generated_sync}
              asccube.mmad %151[%c0], %216, %217, %c64_i32, %c64_i32, %c64_i32, %c3_i32 {c_matrix_init_val = true} : memref<?xf32, 5>, memref<1xf16, 3>, memref<1xf16, 4>
              asccube.release_buf m, %215, 0 {cannir.buffer.generated_sync}
              asccube.release_buf m, %211, 0 {cannir.buffer.generated_sync}
              asccube.release_buf m, %c22_i64, 0 {cannir.buffer.generated_sync}
              asccube.get_buf fixpipe, %c2_i64, 0 {cannir.buffer.generated_sync}
              asccube.get_buf fixpipe, %c22_i64, 0 {cannir.buffer.generated_sync}
              asccube.copy_l0c2l1 %152[%c0], %151[%c0], %c64_i32, %c64_i32, %c1024_i32, %c64_i32, %c3_i32 {relu_en = false} : memref<?xf16, 2>, memref<?xf32, 5>
              asccube.release_buf fixpipe, %c22_i64, 0 {cannir.buffer.generated_sync}
              asccube.release_buf fixpipe, %c2_i64, 0 {cannir.buffer.generated_sync}
              %218 = arith.cmpi eq, %213, %c0_i64 : i64
              %219 = arith.select %218, %118, %119 : i64
              %220 = arith.select %218, %c17_i64, %c18_i64 : i64
              %221 = arith.addi %213, %c1_i64 overflow<nsw> : i64
              %222 = arith.remui %221, %c2_i64 : i64
              %223 = arith.select %218, %124, %125 : i64
              %224 = arith.select %218, %c19_i64, %c20_i64 : i64
              %225 = ascscalar.bind_local_ptr %219 : memref<1xf16, 3>
              asccube.get_buf mte1, %220, 0 {cannir.buffer.generated_sync}
              asccube.copy_l12l0a %225[%c0], %149[%c0], %c0_i32, %c0_i32, %c4_i32, %c4_i32, %c4_i32, %c4_i32 : memref<1xf16, 3>, memref<?xf16, 2>
              %226 = ascscalar.bind_local_ptr %223 : memref<1xf16, 4>
              asccube.get_buf mte1, %224, 0 {cannir.buffer.generated_sync}
              asccube.copy_l12l0b %226[%c0], %149[%c0], %c0_i32, %c0_i32, %c4_i32, %c4_i32, %c4_i32, %c4_i32 : memref<1xf16, 4>, memref<?xf16, 2>
              asccube.get_buf mte1, %c1_i64, 0 {cannir.buffer.generated_sync}
              asccube.copy_l12l0a %225[%c1280], %171[%c0], %c1_i32, %c1_i32, %c1_i32, %c1_i32, %c4_i32, %c4_i32 : memref<1xf16, 3>, memref<?xf16, 2>
              asccube.copy_l12l0a %225[%c3840], %171[%c0], %c3_i32, %c3_i32, %c1_i32, %c1_i32, %c4_i32, %c4_i32 : memref<1xf16, 3>, memref<?xf16, 2>
              asccube.release_buf mte1, %c1_i64, 0 {cannir.buffer.generated_sync}
              asccube.release_buf mte1, %220, 0 {cannir.buffer.generated_sync}
              asccube.get_buf mte1, %c2_i64, 0 {cannir.buffer.generated_sync}
              asccube.copy_l12l0b %226[%c1024], %152[%c0], %c1_i32, %c0_i32, %c1_i32, %c1_i32, %c4_i32, %c4_i32 {transpose = true} : memref<1xf16, 4>, memref<?xf16, 2>
              asccube.copy_l12l0b %226[%c3584], %152[%c0], %c3_i32, %c2_i32, %c1_i32, %c1_i32, %c4_i32, %c4_i32 {transpose = true} : memref<1xf16, 4>, memref<?xf16, 2>
              asccube.release_buf mte1, %c2_i64, 0 {cannir.buffer.generated_sync}
              asccube.release_buf mte1, %224, 0 {cannir.buffer.generated_sync}
              asccube.get_buf m, %c21_i64, 0 {cannir.buffer.generated_sync}
              asccube.get_buf m, %220, 0 {cannir.buffer.generated_sync}
              asccube.get_buf m, %224, 0 {cannir.buffer.generated_sync}
              asccube.mmad %139[%c0], %225, %226, %c64_i32, %c64_i32, %c64_i32, %c3_i32 {c_matrix_init_val = false} : memref<?xf32, 5>, memref<1xf16, 3>, memref<1xf16, 4>
              asccube.release_buf m, %224, 0 {cannir.buffer.generated_sync}
              asccube.release_buf m, %220, 0 {cannir.buffer.generated_sync}
              asccube.release_buf m, %c21_i64, 0 {cannir.buffer.generated_sync}
              asccube.get_buf fixpipe, %c1_i64, 0 {cannir.buffer.generated_sync}
              asccube.get_buf fixpipe, %c21_i64, 0 {cannir.buffer.generated_sync}
              asccube.copy_l0c2l1 %171[%c0], %139[%c0], %c64_i32, %c64_i32, %c1024_i32, %c64_i32, %c3_i32 {relu_en = false} : memref<?xf16, 2>, memref<?xf32, 5>
              asccube.release_buf fixpipe, %c21_i64, 0 {cannir.buffer.generated_sync}
              asccube.release_buf fixpipe, %c1_i64, 0 {cannir.buffer.generated_sync}
              %227 = arith.cmpi eq, %222, %c0_i64 : i64
              %228 = arith.select %227, %118, %119 : i64
              %229 = arith.select %227, %c17_i64, %c18_i64 : i64
              %230 = arith.addi %222, %c1_i64 overflow<nsw> : i64
              %231 = arith.remui %230, %c2_i64 : i64
              %232 = arith.select %227, %124, %125 : i64
              %233 = arith.select %227, %c19_i64, %c20_i64 : i64
              %234 = ascscalar.bind_local_ptr %228 : memref<1xf16, 3>
              asccube.get_buf mte1, %229, 0 {cannir.buffer.generated_sync}
              asccube.get_buf mte1, %c5_i64, 0 {cannir.buffer.generated_sync}
              asccube.copy_l12l0a %234[%c0], %149[%c0], %c0_i32, %c0_i32, %c4_i32, %c4_i32, %c4_i32, %c4_i32 : memref<1xf16, 3>, memref<?xf16, 2>
              %235 = ascscalar.bind_local_ptr %232 : memref<1xf16, 4>
              asccube.get_buf mte1, %233, 0 {cannir.buffer.generated_sync}
              asccube.copy_l12l0b %235[%c0], %149[%c0], %c0_i32, %c0_i32, %c4_i32, %c4_i32, %c4_i32, %c4_i32 : memref<1xf16, 4>, memref<?xf16, 2>
              asccube.get_buf mte1, %c3_i64, 0 {cannir.buffer.generated_sync}
              asccube.copy_l12l0a %234[%c0], %140[%c0], %c0_i32, %c0_i32, %c4_i32, %c4_i32, %c4_i32, %c4_i32 : memref<1xf16, 3>, memref<?xf16, 2>
              asccube.release_buf mte1, %c3_i64, 0 {cannir.buffer.generated_sync}
              asccube.release_buf mte1, %229, 0 {cannir.buffer.generated_sync}
              asccube.get_buf mte1, %c1_i64, 0 {cannir.buffer.generated_sync}
              asccube.copy_l12l0b %235[%c0], %171[%c0], %c0_i32, %c0_i32, %c1_i32, %c2_i32, %c4_i32, %c4_i32 {transpose = true} : memref<1xf16, 4>, memref<?xf16, 2>
              asccube.copy_l12l0b %235[%c1024], %171[%c256], %c0_i32, %c0_i32, %c1_i32, %c2_i32, %c4_i32, %c4_i32 {transpose = true} : memref<1xf16, 4>, memref<?xf16, 2>
              asccube.release_buf mte1, %233, 0 {cannir.buffer.generated_sync}
              %236 = arith.cmpi eq, %231, %c0_i64 : i64
              %237 = arith.select %236, %118, %119 : i64
              %238 = arith.select %236, %c17_i64, %c18_i64 : i64
              %239 = arith.select %236, %124, %125 : i64
              %240 = arith.select %236, %c19_i64, %c20_i64 : i64
              %241 = ascscalar.bind_local_ptr %237 : memref<1xf16, 3>
              asccube.get_buf mte1, %238, 0 {cannir.buffer.generated_sync}
              asccube.copy_l12l0a %241[%c0], %149[%c0], %c0_i32, %c0_i32, %c4_i32, %c4_i32, %c4_i32, %c4_i32 : memref<1xf16, 3>, memref<?xf16, 2>
              %242 = ascscalar.bind_local_ptr %239 : memref<1xf16, 4>
              asccube.get_buf mte1, %240, 0 {cannir.buffer.generated_sync}
              asccube.copy_l12l0b %242[%c0], %149[%c0], %c0_i32, %c0_i32, %c4_i32, %c4_i32, %c4_i32, %c4_i32 : memref<1xf16, 4>, memref<?xf16, 2>
              asccube.release_buf mte1, %c5_i64, 0 {cannir.buffer.generated_sync}
              asccube.release_buf mte1, %240, 0 {cannir.buffer.generated_sync}
              asccube.copy_l12l0a %241[%c0], %171[%c0], %c0_i32, %c0_i32, %c4_i32, %c4_i32, %c4_i32, %c4_i32 : memref<1xf16, 3>, memref<?xf16, 2>
              asccube.release_buf mte1, %c1_i64, 0 {cannir.buffer.generated_sync}
              asccube.release_buf mte1, %238, 0 {cannir.buffer.generated_sync}
              asccube.get_buf m, %c22_i64, 0 {cannir.buffer.generated_sync}
              asccube.get_buf m, %229, 0 {cannir.buffer.generated_sync}
              asccube.get_buf m, %233, 0 {cannir.buffer.generated_sync}
              asccube.mmad %151[%c0], %234, %235, %c64_i32, %c64_i32, %c64_i32, %c3_i32 {c_matrix_init_val = true} : memref<?xf32, 5>, memref<1xf16, 3>, memref<1xf16, 4>
              asccube.release_buf m, %233, 0 {cannir.buffer.generated_sync}
              asccube.release_buf m, %229, 0 {cannir.buffer.generated_sync}
              asccube.release_buf m, %c22_i64, 0 {cannir.buffer.generated_sync}
              asccube.get_buf fixpipe, %c2_i64, 0 {cannir.buffer.generated_sync}
              asccube.get_buf fixpipe, %c22_i64, 0 {cannir.buffer.generated_sync}
              asccube.copy_l0c2l1 %152[%c0], %151[%c0], %c64_i32, %c64_i32, %c1024_i32, %c64_i32, %c3_i32 {relu_en = false} : memref<?xf16, 2>, memref<?xf32, 5>
              asccube.release_buf fixpipe, %c22_i64, 0 {cannir.buffer.generated_sync}
              asccube.release_buf fixpipe, %c2_i64, 0 {cannir.buffer.generated_sync}
              asccube.get_buf mte1, %240, 0 {cannir.buffer.generated_sync}
              asccube.get_buf mte1, %c2_i64, 0 {cannir.buffer.generated_sync}
              asccube.copy_l12l0b %242[%c2048], %152[%c512], %c0_i32, %c0_i32, %c1_i32, %c2_i32, %c4_i32, %c4_i32 {transpose = true} : memref<1xf16, 4>, memref<?xf16, 2>
              asccube.copy_l12l0b %242[%c3072], %152[%c768], %c0_i32, %c0_i32, %c1_i32, %c2_i32, %c4_i32, %c4_i32 {transpose = true} : memref<1xf16, 4>, memref<?xf16, 2>
              asccube.release_buf mte1, %c2_i64, 0 {cannir.buffer.generated_sync}
              asccube.release_buf mte1, %240, 0 {cannir.buffer.generated_sync}
              asccube.get_buf m, %c21_i64, 0 {cannir.buffer.generated_sync}
              asccube.get_buf m, %238, 0 {cannir.buffer.generated_sync}
              asccube.get_buf m, %240, 0 {cannir.buffer.generated_sync}
              asccube.mmad %139[%c0], %241, %242, %c64_i32, %c64_i32, %c64_i32, %c3_i32 {c_matrix_init_val = false} : memref<?xf32, 5>, memref<1xf16, 3>, memref<1xf16, 4>
              asccube.release_buf m, %240, 0 {cannir.buffer.generated_sync}
              asccube.release_buf m, %238, 0 {cannir.buffer.generated_sync}
              asccube.release_buf m, %c21_i64, 0 {cannir.buffer.generated_sync}
              %243 = ascscalar.bitcast_memref %alloc_37 byte_offset = %c0_i64 : memref<4096xbf16, 2> -> memref<?xbf16, 2>
              asccube.get_buf fixpipe, %c6_i64, 0 {cannir.buffer.generated_sync}
              asccube.get_buf fixpipe, %c21_i64, 0 {cannir.buffer.generated_sync}
              asccube.copy_l0c2l1 %243[%c0], %139[%c0], %c64_i32, %c64_i32, %c1024_i32, %c64_i32, %c3_i32 {relu_en = false} : memref<?xbf16, 2>, memref<?xf32, 5>
              asccube.release_buf fixpipe, %c21_i64, 0 {cannir.buffer.generated_sync}
              asccube.release_buf fixpipe, %c6_i64, 0 {cannir.buffer.generated_sync}
              %244 = cannir.array_load %100[%108] : <2xi64>
              %245 = arith.remsi %244, %109 : i64
              %246 = arith.cmpi ne, %245, %c0_i64 : i64
              %247 = arith.cmpi slt, %245, %c0_i64 : i64
              %248 = arith.xori %247, %110 : i1
              %249 = arith.andi %246, %248 : i1
              %250 = arith.divsi %244, %109 : i64
              %251 = arith.extui %249 : i1 to i64
              %252 = arith.subi %250, %251 : i64
              %253 = arith.muli %252, %109 : i64
              %254 = arith.subi %244, %253 : i64
              %255 = arith.addi %111, %254 : i64
              %256 = arith.remsi %252, %115 : i64
              %257 = arith.cmpi slt, %256, %c0_i64 : i64
              %258 = arith.select %257, %115, %c0_i64 : i64
              %259 = arith.addi %256, %258 : i64
              %260 = arith.cmpi eq, %102#1, %c0_i64 : i64
              %261 = arith.divsi %75, %c64_i64 : i64
              %262 = arith.minsi %255, %261 : i64
              %263 = arith.muli %262, %c64_i64 : i64
              %264 = arith.subi %75, %263 : i64
              %265 = arith.minsi %264, %c64_i64 : i64
              %266 = arith.cmpi sle, %255, %261 : i64
              %267 = arith.select %266, %265, %c0_i64 : i64
              %268 = arith.muli %255, %c64_i64 : i64
              %269 = arith.addi %268, %93 : i64
              %270 = arith.cmpi ugt, %267, %c0_i64 : i64
              %271 = ascscalar.memref_address %alloc_39 : memref<8192xbf16, 3>
              %272 = ascscalar.memref_address %alloc_38 : memref<8192xbf16, 3>
              %273 = arith.select %260, %271, %272 : i64
              %274 = arith.select %260, %c7_i64, %c8_i64 : i64
              %275 = ascscalar.bind_local_ptr %273 : memref<1xbf16, 3>
              asccube.get_buf mte1, %274, 0 {cannir.buffer.generated_sync}
              asccube.get_buf mte1, %c6_i64, 0 {cannir.buffer.generated_sync}
              asccube.copy_l12l0a %275[%c0], %243[%c0], %c0_i32, %c0_i32, %c4_i32, %c4_i32, %c4_i32, %c4_i32 : memref<1xbf16, 3>, memref<?xbf16, 2>
              asccube.release_buf mte1, %c6_i64, 0 {cannir.buffer.generated_sync}
              asccube.release_buf mte1, %274, 0 {cannir.buffer.generated_sync}
              %276 = arith.cmpi eq, %102#2, %c0_i64 : i64
              %277 = ascscalar.memref_address %alloc_35 : memref<8192xbf16, 4>
              %278 = ascscalar.memref_address %alloc_34 : memref<8192xbf16, 4>
              %279 = arith.select %276, %277, %278 : i64
              %280 = arith.select %276, %c9_i64, %c10_i64 : i64
              %281 = arith.addi %102#2, %c1_i64 overflow<nsw> : i64
              %282 = arith.remui %281, %c2_i64 : i64
              %283 = ascscalar.bind_local_ptr %279 : memref<1xbf16, 4>
              %284 = ascscalar.bitcast_memref %alloc_36 byte_offset = %c0_i64 : memref<8192xbf16, 2> -> memref<?xbf16, 2>
              asccube.sync_intra_wait mte1, %c9_i64 {cannir.sync.banks = 3 : i32}
              asccube.sync_intra_wait mte1, %c25_i64 {cannir.sync.banks = 3 : i32}
              asccube.get_buf mte1, %280, 0 {cannir.buffer.generated_sync}
              asccube.copy_l12l0b %283[%c0], %284[%c0], %c0_i32, %c0_i32, %c4_i32, %c8_i32, %c4_i32, %c8_i32 {transpose = true} : memref<1xbf16, 4>, memref<?xbf16, 2>
              asccube.release_buf mte1, %280, 0 {cannir.buffer.generated_sync}
              asccube.sync_intra_arrive mte1, %c9_i64 {cannir.sync.banks = 3 : i32}
              asccube.sync_intra_arrive mte1, %c25_i64 {cannir.sync.banks = 3 : i32}
              %285 = ascscalar.memref_address %alloc_32 : memref<8192xf32, 5>
              %286 = ascscalar.memref_address %alloc_31 : memref<8192xf32, 5>
              %287 = arith.select %276, %285, %286 : i64
              %288 = arith.select %276, %c11_i64, %c12_i64 : i64
              %289 = ascscalar.bind_local_ptr %287 : memref<1xf32, 5>
              asccube.get_buf m, %288, 0 {cannir.buffer.generated_sync}
              asccube.get_buf m, %274, 0 {cannir.buffer.generated_sync}
              asccube.get_buf m, %280, 0 {cannir.buffer.generated_sync}
              asccube.mmad %289[%c0], %275, %283, %c64_i32, %c64_i32, %c128_i32, %c0_i32 {c_matrix_init_val = true} : memref<1xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
              asccube.release_buf m, %280, 0 {cannir.buffer.generated_sync}
              asccube.release_buf m, %288, 0 {cannir.buffer.generated_sync}
              %290 = cannir.extract_physical_layout %arg11 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
              %291 = cannir.extract_stride %290 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
              %292 = cannir.extract_dim %291[0] : !cannir.stride<(-1, -1, -1, 1)>
              %293 = arith.muli %94, %292 : i64
              %294 = arith.index_cast %293 : i64 to index
              %295 = cannir.extract_dim %291[1] : !cannir.stride<(-1, -1, -1, 1)>
              %296 = arith.muli %259, %295 : i64
              %297 = arith.index_cast %296 : i64 to index
              %298 = arith.addi %294, %297 : index
              %299 = cannir.extract_dim %291[2] : !cannir.stride<(-1, -1, -1, 1)>
              %300 = arith.muli %269, %299 : i64
              %301 = arith.index_cast %300 : i64 to index
              %302 = arith.addi %298, %301 : index
              %303 = cannir.extract_layout %arg11 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
              %304 = cannir.extract_stride %303 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
              %305 = cannir.extract_dim %304[2] : !cannir.stride<(-1, -1, -1, 1)>
              scf.if %270 {
                asccube.get_buf fixpipe, %288, 0 {cannir.buffer.generated_sync}
                %330 = arith.trunci %267 : i64 to i32
                %331 = arith.trunci %305 : i64 to i32
                asccube.copy_l0c2gm %alloc[%302], %289[%c0], %c128_i32, %330, %331, %c64_i32, %c0_i32 l2_cache_ctl %c0_i32 {nz2nd_en = true, relu_en = false} : memref<1xbf16>, memref<1xf32, 5>
                asccube.release_buf fixpipe, %288, 0 {cannir.buffer.generated_sync}
              }
              %306 = arith.cmpi eq, %282, %c0_i64 : i64
              %307 = arith.select %306, %277, %278 : i64
              %308 = arith.select %306, %c9_i64, %c10_i64 : i64
              %309 = ascscalar.bind_local_ptr %307 : memref<1xbf16, 4>
              %310 = ascscalar.bitcast_memref %alloc_33 byte_offset = %c0_i64 : memref<8192xbf16, 2> -> memref<?xbf16, 2>
              asccube.sync_intra_wait mte1, %c8_i64 {cannir.sync.banks = 3 : i32}
              asccube.sync_intra_wait mte1, %c24_i64 {cannir.sync.banks = 3 : i32}
              asccube.get_buf mte1, %308, 0 {cannir.buffer.generated_sync}
              asccube.copy_l12l0b %309[%c0], %310[%c0], %c0_i32, %c0_i32, %c4_i32, %c8_i32, %c4_i32, %c8_i32 {transpose = true} : memref<1xbf16, 4>, memref<?xbf16, 2>
              asccube.release_buf mte1, %308, 0 {cannir.buffer.generated_sync}
              asccube.sync_intra_arrive mte1, %c8_i64 {cannir.sync.banks = 3 : i32}
              asccube.sync_intra_arrive mte1, %c24_i64 {cannir.sync.banks = 3 : i32}
              %311 = arith.select %306, %285, %286 : i64
              %312 = arith.select %306, %c11_i64, %c12_i64 : i64
              %313 = ascscalar.bind_local_ptr %311 : memref<1xf32, 5>
              asccube.get_buf m, %312, 0 {cannir.buffer.generated_sync}
              asccube.get_buf m, %308, 0 {cannir.buffer.generated_sync}
              asccube.mmad %313[%c0], %275, %309, %c64_i32, %c64_i32, %c128_i32, %c0_i32 {c_matrix_init_val = true} : memref<1xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
              asccube.release_buf m, %308, 0 {cannir.buffer.generated_sync}
              asccube.release_buf m, %274, 0 {cannir.buffer.generated_sync}
              asccube.release_buf m, %312, 0 {cannir.buffer.generated_sync}
              %314 = cannir.extract_physical_layout %arg13 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
              %315 = cannir.extract_stride %314 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
              %316 = cannir.extract_dim %315[0] : !cannir.stride<(-1, -1, -1, 1)>
              %317 = arith.muli %94, %316 : i64
              %318 = arith.index_cast %317 : i64 to index
              %319 = cannir.extract_dim %315[1] : !cannir.stride<(-1, -1, -1, 1)>
              %320 = arith.muli %259, %319 : i64
              %321 = arith.index_cast %320 : i64 to index
              %322 = arith.addi %318, %321 : index
              %323 = cannir.extract_dim %315[2] : !cannir.stride<(-1, -1, -1, 1)>
              %324 = arith.muli %269, %323 : i64
              %325 = arith.index_cast %324 : i64 to index
              %326 = arith.addi %322, %325 : index
              %327 = cannir.extract_layout %arg13 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
              %328 = cannir.extract_stride %327 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
              %329 = cannir.extract_dim %328[2] : !cannir.stride<(-1, -1, -1, 1)>
              scf.if %270 {
                asccube.get_buf fixpipe, %312, 0 {cannir.buffer.generated_sync}
                %330 = arith.trunci %267 : i64 to i32
                %331 = arith.trunci %329 : i64 to i32
                asccube.copy_l0c2gm %alloc_0[%326], %313[%c0], %c128_i32, %330, %331, %c64_i32, %c0_i32 l2_cache_ctl %c0_i32 {nz2nd_en = true, relu_en = false} : memref<1xbf16>, memref<1xf32, 5>
                asccube.release_buf fixpipe, %312, 0 {cannir.buffer.generated_sync}
              }
            }
          }
        }
        asccube.sync_all
        asccube.sync_block_wait s, %c0_i64
        asccube.sync_block_arrive fixpipe, %c4_i64 {mode = 0 : i32}
        asccube.sync_block_wait s, %c4_i64 {mode = 0 : i32}
        asccube.sync_block_arrive mte3, %c8_i64
        %36 = arith.addi %26, %c65_i32 : i32
        %37 = arith.addi %36, %17 : i32
        %38 = arith.addi %37, %17 : i32
        %39 = arith.addi %38, %c1_i32 : i32
        %40 = arith.extsi %39 : i32 to i64
        %41 = arith.addi %40, %28 : i64
        %42 = arith.index_cast %41 : i64 to index
        %43 = memref.load %alloc_4[%42] : memref<1xi32>
        %44 = arith.addi %41, %c1_i64 : i64
        %45 = arith.index_cast %44 : i64 to index
        %46 = memref.load %alloc_4[%45] : memref<1xi32>
        scf.for %arg23 = %c0_i32 to %17 step %c1_i32  : i32 {
          %49 = arith.addi %37, %arg23 : i32
          %50 = arith.index_cast %49 : i32 to index
          %51 = memref.load %alloc_4[%50] : memref<1xi32>
          %52 = arith.addi %49, %c1_i32 : i32
          %53 = arith.index_cast %52 : i32 to index
          %54 = memref.load %alloc_4[%53] : memref<1xi32>
          %55 = arith.cmpi slt, %43, %51 : i32
          %56 = arith.select %55, %51, %43 : i32
          %57 = arith.cmpi sgt, %46, %54 : i32
          %58 = arith.select %57, %54, %46 : i32
          %59 = arith.cmpi slt, %56, %58 : i32
          %60 = arith.subi %56, %51 : i32
          %61 = arith.subi %58, %51 : i32
          %62 = arith.extsi %60 : i32 to i64
          %63 = arith.extsi %61 : i32 to i64
          %64 = arith.select %59, %62, %c0_i64 : i64
          %65 = arith.select %59, %63, %c0_i64 : i64
          %66 = arith.addi %36, %arg23 : i32
          %67 = arith.index_cast %66 : i32 to index
          %68 = memref.load %alloc_4[%67] : memref<1xi32>
          %69 = arith.addi %18, %arg23 : i32
          %70 = arith.index_cast %69 : i32 to index
          %71 = memref.load %alloc_4[%70] : memref<1xi32>
          %72 = arith.addi %23, %arg23 : i32
          %73 = arith.index_cast %72 : i32 to index
          %74 = memref.load %alloc_4[%73] : memref<1xi32>
          %75 = arith.addi %24, %arg23 : i32
          %76 = arith.index_cast %75 : i32 to index
          %77 = memref.load %alloc_4[%76] : memref<1xi32>
          %78 = arith.extsi %68 : i32 to i64
          %79 = arith.addi %64, %78 : i64
          %80 = arith.subi %79, %c1_i64 : i64
          %81 = arith.remsi %80, %78 : i64
          %82 = arith.cmpi ne, %81, %c0_i64 : i64
          %83 = arith.cmpi slt, %81, %c0_i64 : i64
          %84 = arith.cmpi slt, %78, %c0_i64 : i64
          %85 = arith.xori %83, %84 : i1
          %86 = arith.andi %82, %85 : i1
          %87 = arith.divsi %80, %78 : i64
          %88 = arith.extui %86 : i1 to i64
          %89 = arith.subi %87, %88 : i64
          %90 = arith.addi %65, %78 : i64
          %91 = arith.subi %90, %c1_i64 : i64
          %92 = arith.remsi %91, %78 : i64
          %93 = arith.cmpi ne, %92, %c0_i64 : i64
          %94 = arith.cmpi slt, %92, %c0_i64 : i64
          %95 = arith.xori %94, %84 : i1
          %96 = arith.andi %93, %95 : i1
          %97 = arith.divsi %91, %78 : i64
          %98 = arith.extui %96 : i1 to i64
          %99 = arith.subi %97, %98 : i64
          %100 = arith.cmpi slt, %89, %99 : i64
          scf.if %100 {
            %101 = arith.addi %74, %77 : i32
            scf.for %arg24 = %89 to %99 step %c1_i64  : i64 {
              %102 = arith.extsi %71 : i32 to i64
              %103 = arith.cmpi sgt, %74, %c0_i32 : i32
              scf.if %103 {
                %104 = ascscalar.bitcast_memref %alloc_27 byte_offset = %c0_i64 : memref<16384xf32, 2> -> memref<?xf32, 2>
                %105 = cannir.extract_physical_layout %arg14 : !cannir.tensor<<gm, f32>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, identity>
                %106 = cannir.extract_stride %105 : <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>> -> <(-1, 16384, 128, 1)>
                %107 = cannir.extract_dim %106[0] : !cannir.stride<(-1, 16384, 128, 1)>
                %108 = arith.muli %102, %107 : i64
                %109 = arith.index_cast %108 : i64 to index
                %110 = arith.muli %arg24, %c16384_i64 : i64
                %111 = arith.index_cast %110 : i64 to index
                %112 = arith.addi %109, %111 : index
                asccube.get_buf mte2, %c25_i64, 0 {cannir.buffer.generated_sync}
                asccube.copy_gm2l1 %104[%c0], %alloc_1[%112], %c128_i32, %c128_i32, %c512_i64, %c1_i32, %c0_i64, %c0_i64, %c128_i32 l2_cache_ctl %c0_i32 {nd2nz = true} : memref<?xf32, 2>, memref<1xf32>
                asccube.release_buf mte2, %c25_i64, 0 {cannir.buffer.generated_sync}
                %113 = ascscalar.bitcast_memref %alloc_26 byte_offset = %c0_i64 : memref<16384xf32, 2> -> memref<?xf32, 2>
                %114 = cannir.extract_physical_layout %arg15 : !cannir.tensor<<gm, f32>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, identity>
                %115 = cannir.extract_stride %114 : <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>> -> <(-1, 16384, 128, 1)>
                %116 = cannir.extract_dim %115[0] : !cannir.stride<(-1, 16384, 128, 1)>
                %117 = arith.muli %102, %116 : i64
                %118 = arith.index_cast %117 : i64 to index
                %119 = arith.addi %118, %111 : index
                asccube.get_buf mte2, %c26_i64, 0 {cannir.buffer.generated_sync}
                asccube.copy_gm2l1 %113[%c0], %alloc_2[%119], %c128_i32, %c128_i32, %c512_i64, %c1_i32, %c0_i64, %c0_i64, %c128_i32 l2_cache_ctl %c0_i32 {nd2nz = true} : memref<?xf32, 2>, memref<1xf32>
                asccube.release_buf mte2, %c26_i64, 0 {cannir.buffer.generated_sync}
              }
              scf.for %arg25 = %74 to %101 step %c1_i32  : i32 {
                %104 = ascscalar.bitcast_memref %alloc_28 byte_offset = %c0_i64 : memref<8192xf32, 2> -> memref<?xf32, 2>
                %105 = ascscalar.bitcast_memref %alloc_29 byte_offset = %c0_i64 : memref<8192xf32, 2> -> memref<?xf32, 2>
                %106 = ascscalar.bitcast_memref %alloc_30 byte_offset = %c0_i64 : memref<8192xf32, 2> -> memref<?xf32, 2>
                %107 = ascscalar.bitcast_memref %alloc_19 byte_offset = %c0_i64 : memref<16384xf32, 3> -> memref<?xf32, 3>
                asccube.sync_intra_wait mte1, %c12_i64 {cannir.sync.banks = 3 : i32}
                asccube.sync_intra_wait mte1, %c28_i64 {cannir.sync.banks = 3 : i32}
                asccube.get_buf mte1, %c27_i64, 0 {cannir.buffer.generated_sync}
                asccube.copy_l12l0a %107[%c0], %106[%c0], %c0_i32, %c0_i32, %c4_i32, %c16_i32, %c4_i32, %c8_i32 {transpose = true} : memref<?xf32, 3>, memref<?xf32, 2>
                asccube.release_buf mte1, %c27_i64, 0 {cannir.buffer.generated_sync}
                asccube.sync_intra_arrive mte1, %c12_i64 {cannir.sync.banks = 3 : i32}
                asccube.sync_intra_arrive mte1, %c28_i64 {cannir.sync.banks = 3 : i32}
                %108 = ascscalar.bitcast_memref %alloc_20 byte_offset = %c0_i64 : memref<16384xf32, 4> -> memref<?xf32, 4>
                asccube.sync_intra_wait mte1, %c10_i64 {cannir.sync.banks = 3 : i32}
                asccube.sync_intra_wait mte1, %c26_i64 {cannir.sync.banks = 3 : i32}
                asccube.get_buf mte1, %c28_i64, 0 {cannir.buffer.generated_sync}
                asccube.copy_l12l0b %108[%c0], %104[%c0], %c0_i32, %c0_i32, %c4_i32, %c16_i32, %c4_i32, %c8_i32 {transpose = true} : memref<?xf32, 4>, memref<?xf32, 2>
                asccube.release_buf mte1, %c28_i64, 0 {cannir.buffer.generated_sync}
                %109 = ascscalar.bitcast_memref %alloc_18 byte_offset = %c0_i64 : memref<16384xf32, 5> -> memref<?xf32, 5>
                asccube.get_buf m, %c29_i64, 0 {cannir.buffer.generated_sync}
                asccube.get_buf m, %c27_i64, 0 {cannir.buffer.generated_sync}
                asccube.get_buf m, %c28_i64, 0 {cannir.buffer.generated_sync}
                asccube.mmad %109[%c0], %107, %108, %c128_i32, %c64_i32, %c128_i32, %c0_i32 {c_matrix_init_val = true} : memref<?xf32, 5>, memref<?xf32, 3>, memref<?xf32, 4>
                asccube.release_buf m, %c28_i64, 0 {cannir.buffer.generated_sync}
                asccube.release_buf m, %c27_i64, 0 {cannir.buffer.generated_sync}
                asccube.release_buf m, %c29_i64, 0 {cannir.buffer.generated_sync}
                %110 = ascscalar.bitcast_memref %alloc_8 byte_offset = %c0_i64 : memref<8192xf32, 1> -> memref<?xf32, 1>
                asccube.sync_intra_wait fixpipe, %c14_i64 {cannir.sync.banks = 3 : i32}
                asccube.sync_intra_wait fixpipe, %c30_i64 {cannir.sync.banks = 3 : i32}
                asccube.get_buf fixpipe, %c29_i64, 0 {cannir.buffer.generated_sync}
                asccube.copy_l0c2ub %110[%c0], %109[%c0], %c128_i32, %c128_i32, %c128_i32, %c128_i32, %c0_i32 sub_block_id %c0_i32 {dual_dst_ctl = 1 : i32, nz2nd_en = true, relu_en = false} : memref<?xf32, 1>, memref<?xf32, 5>
                asccube.release_buf fixpipe, %c29_i64, 0 {cannir.buffer.generated_sync}
                asccube.sync_intra_arrive fixpipe, %c14_i64 {cannir.sync.banks = 3 : i32}
                asccube.sync_intra_arrive fixpipe, %c30_i64 {cannir.sync.banks = 3 : i32}
                %111 = arith.cmpi eq, %arg25, %74 : i32
                %112 = arith.addi %arg25, %c1_i32 : i32
                %113 = arith.cmpi eq, %112, %101 : i32
                %114 = arith.cmpi eq, %74, %c0_i32 : i32
                asccube.sync_intra_wait mte1, %c11_i64 {cannir.sync.banks = 3 : i32}
                asccube.sync_intra_wait mte1, %c27_i64 {cannir.sync.banks = 3 : i32}
                asccube.get_buf mte1, %c27_i64, 0 {cannir.buffer.generated_sync}
                asccube.copy_l12l0a %107[%c0], %105[%c0], %c0_i32, %c0_i32, %c4_i32, %c16_i32, %c4_i32, %c8_i32 {transpose = true} : memref<?xf32, 3>, memref<?xf32, 2>
                asccube.release_buf mte1, %c27_i64, 0 {cannir.buffer.generated_sync}
                asccube.sync_intra_arrive mte1, %c11_i64 {cannir.sync.banks = 3 : i32}
                asccube.sync_intra_arrive mte1, %c27_i64 {cannir.sync.banks = 3 : i32}
                asccube.get_buf mte1, %c28_i64, 0 {cannir.buffer.generated_sync}
                asccube.copy_l12l0b %108[%c0], %104[%c0], %c0_i32, %c0_i32, %c4_i32, %c16_i32, %c4_i32, %c8_i32 {transpose = true} : memref<?xf32, 4>, memref<?xf32, 2>
                asccube.release_buf mte1, %c28_i64, 0 {cannir.buffer.generated_sync}
                asccube.sync_intra_arrive mte1, %c10_i64 {cannir.sync.banks = 3 : i32}
                asccube.sync_intra_arrive mte1, %c26_i64 {cannir.sync.banks = 3 : i32}
                %115 = ascscalar.bitcast_memref %alloc_21 byte_offset = %c0_i64 : memref<16384xf32, 5> -> memref<?xf32, 5>
                asccube.get_buf m, %c30_i64, 0 {cannir.buffer.generated_sync}
                asccube.get_buf m, %c27_i64, 0 {cannir.buffer.generated_sync}
                asccube.get_buf m, %c28_i64, 0 {cannir.buffer.generated_sync}
                asccube.mmad %115[%c0], %107, %108, %c128_i32, %c64_i32, %c128_i32, %c0_i32 {c_matrix_init_val = true} : memref<?xf32, 5>, memref<?xf32, 3>, memref<?xf32, 4>
                asccube.release_buf m, %c28_i64, 0 {cannir.buffer.generated_sync}
                asccube.release_buf m, %c27_i64, 0 {cannir.buffer.generated_sync}
                asccube.release_buf m, %c30_i64, 0 {cannir.buffer.generated_sync}
                %116 = arith.select %103, %111, %false : i1
                scf.if %116 {
                  %120 = ascscalar.bitcast_memref %alloc_27 byte_offset = %c0_i64 : memref<16384xf32, 2> -> memref<?xf32, 2>
                  asccube.get_buf mte1, %c27_i64, 0 {cannir.buffer.generated_sync}
                  asccube.get_buf mte1, %c25_i64, 0 {cannir.buffer.generated_sync}
                  asccube.copy_l12l0a %107[%c0], %120[%c0], %c0_i32, %c0_i32, %c8_i32, %c16_i32, %c8_i32, %c8_i32 : memref<?xf32, 3>, memref<?xf32, 2>
                  asccube.release_buf mte1, %c25_i64, 0 {cannir.buffer.generated_sync}
                  asccube.release_buf mte1, %c27_i64, 0 {cannir.buffer.generated_sync}
                  %121 = ascscalar.bitcast_memref %alloc_24 byte_offset = %c0_i64 : memref<16384xf32, 2> -> memref<?xf32, 2>
                  asccube.sync_intra_wait mte1, %c13_i64 {cannir.sync.banks = 3 : i32}
                  asccube.sync_intra_wait mte1, %c29_i64 {cannir.sync.banks = 3 : i32}
                  asccube.get_buf mte1, %c28_i64, 0 {cannir.buffer.generated_sync}
                  asccube.copy_l12l0b %108[%c0], %121[%c0], %c0_i32, %c0_i32, %c8_i32, %c16_i32, %c8_i32, %c8_i32 {transpose = true} : memref<?xf32, 4>, memref<?xf32, 2>
                  asccube.release_buf mte1, %c28_i64, 0 {cannir.buffer.generated_sync}
                  asccube.get_buf m, %c29_i64, 0 {cannir.buffer.generated_sync}
                  asccube.get_buf m, %c27_i64, 0 {cannir.buffer.generated_sync}
                  asccube.get_buf m, %c28_i64, 0 {cannir.buffer.generated_sync}
                  asccube.mmad %109[%c0], %107, %108, %c128_i32, %c128_i32, %c128_i32, %c0_i32 {c_matrix_init_val = true} : memref<?xf32, 5>, memref<?xf32, 3>, memref<?xf32, 4>
                  asccube.release_buf m, %c28_i64, 0 {cannir.buffer.generated_sync}
                  asccube.release_buf m, %c27_i64, 0 {cannir.buffer.generated_sync}
                  asccube.release_buf m, %c29_i64, 0 {cannir.buffer.generated_sync}
                  %122 = ascscalar.bitcast_memref %alloc_26 byte_offset = %c0_i64 : memref<16384xf32, 2> -> memref<?xf32, 2>
                  asccube.get_buf mte1, %c27_i64, 0 {cannir.buffer.generated_sync}
                  asccube.get_buf mte1, %c26_i64, 0 {cannir.buffer.generated_sync}
                  asccube.copy_l12l0a %107[%c0], %122[%c0], %c0_i32, %c0_i32, %c8_i32, %c16_i32, %c8_i32, %c8_i32 : memref<?xf32, 3>, memref<?xf32, 2>
                  asccube.release_buf mte1, %c26_i64, 0 {cannir.buffer.generated_sync}
                  asccube.release_buf mte1, %c27_i64, 0 {cannir.buffer.generated_sync}
                  asccube.get_buf mte1, %c28_i64, 0 {cannir.buffer.generated_sync}
                  asccube.copy_l12l0b %108[%c0], %121[%c0], %c0_i32, %c0_i32, %c8_i32, %c16_i32, %c8_i32, %c8_i32 {transpose = true} : memref<?xf32, 4>, memref<?xf32, 2>
                  asccube.release_buf mte1, %c28_i64, 0 {cannir.buffer.generated_sync}
                  asccube.sync_intra_arrive mte1, %c13_i64 {cannir.sync.banks = 3 : i32}
                  asccube.sync_intra_arrive mte1, %c29_i64 {cannir.sync.banks = 3 : i32}
                  asccube.get_buf m, %c30_i64, 0 {cannir.buffer.generated_sync}
                  asccube.get_buf m, %c27_i64, 0 {cannir.buffer.generated_sync}
                  asccube.get_buf m, %c28_i64, 0 {cannir.buffer.generated_sync}
                  asccube.mmad %115[%c0], %107, %108, %c128_i32, %c128_i32, %c128_i32, %c0_i32 {c_matrix_init_val = false} : memref<?xf32, 5>, memref<?xf32, 3>, memref<?xf32, 4>
                  asccube.release_buf m, %c28_i64, 0 {cannir.buffer.generated_sync}
                  asccube.release_buf m, %c27_i64, 0 {cannir.buffer.generated_sync}
                  asccube.release_buf m, %c30_i64, 0 {cannir.buffer.generated_sync}
                } else {
                  %120 = arith.cmpi ne, %arg25, %74 : i32
                  %121 = arith.select %103, %true, %120 : i1
                  scf.if %121 {
                    %122 = arith.cmpi eq, %arg25, %c1_i32 : i32
                    %123 = arith.select %114, %122, %false : i1
                    scf.if %123 {
                      asccube.sync_block_wait s, %c10_i64
                      %126 = ascscalar.bitcast_memref %alloc_25 byte_offset = %c0_i64 : memref<16384xf32, 2> -> memref<?xf32, 2>
                      asccube.sync_intra_wait mte1, %c0_i64 {cannir.sync.banks = 3 : i32}
                      asccube.sync_intra_wait mte1, %c16_i64 {cannir.sync.banks = 3 : i32}
                      asccube.get_buf mte1, %c27_i64, 0 {cannir.buffer.generated_sync}
                      asccube.copy_l12l0a %107[%c0], %126[%c0], %c0_i32, %c0_i32, %c8_i32, %c16_i32, %c8_i32, %c8_i32 : memref<?xf32, 3>, memref<?xf32, 2>
                      asccube.release_buf mte1, %c27_i64, 0 {cannir.buffer.generated_sync}
                      asccube.sync_intra_arrive mte1, %c0_i64 {cannir.sync.banks = 3 : i32}
                      asccube.sync_intra_arrive mte1, %c16_i64 {cannir.sync.banks = 3 : i32}
                      %127 = ascscalar.bitcast_memref %alloc_24 byte_offset = %c0_i64 : memref<16384xf32, 2> -> memref<?xf32, 2>
                      asccube.sync_intra_wait mte1, %c13_i64 {cannir.sync.banks = 3 : i32}
                      asccube.sync_intra_wait mte1, %c29_i64 {cannir.sync.banks = 3 : i32}
                      asccube.get_buf mte1, %c28_i64, 0 {cannir.buffer.generated_sync}
                      asccube.copy_l12l0b %108[%c0], %127[%c0], %c0_i32, %c0_i32, %c8_i32, %c16_i32, %c8_i32, %c8_i32 {transpose = true} : memref<?xf32, 4>, memref<?xf32, 2>
                      asccube.release_buf mte1, %c28_i64, 0 {cannir.buffer.generated_sync}
                      asccube.get_buf m, %c29_i64, 0 {cannir.buffer.generated_sync}
                      asccube.get_buf m, %c27_i64, 0 {cannir.buffer.generated_sync}
                      asccube.get_buf m, %c28_i64, 0 {cannir.buffer.generated_sync}
                      asccube.mmad %109[%c0], %107, %108, %c128_i32, %c128_i32, %c128_i32, %c0_i32 {c_matrix_init_val = true} : memref<?xf32, 5>, memref<?xf32, 3>, memref<?xf32, 4>
                      asccube.release_buf m, %c28_i64, 0 {cannir.buffer.generated_sync}
                      asccube.release_buf m, %c27_i64, 0 {cannir.buffer.generated_sync}
                      asccube.release_buf m, %c29_i64, 0 {cannir.buffer.generated_sync}
                    } else {
                      %126 = ascscalar.bitcast_memref %alloc_23 byte_offset = %c0_i64 : memref<16384xf32, 2> -> memref<?xf32, 2>
                      asccube.get_buf mte1, %c27_i64, 0 {cannir.buffer.generated_sync}
                      asccube.copy_l12l0a %107[%c0], %126[%c0], %c0_i32, %c0_i32, %c8_i32, %c16_i32, %c8_i32, %c8_i32 : memref<?xf32, 3>, memref<?xf32, 2>
                      asccube.release_buf mte1, %c27_i64, 0 {cannir.buffer.generated_sync}
                      %127 = ascscalar.bitcast_memref %alloc_24 byte_offset = %c0_i64 : memref<16384xf32, 2> -> memref<?xf32, 2>
                      asccube.sync_intra_wait mte1, %c13_i64 {cannir.sync.banks = 3 : i32}
                      asccube.sync_intra_wait mte1, %c29_i64 {cannir.sync.banks = 3 : i32}
                      asccube.get_buf mte1, %c28_i64, 0 {cannir.buffer.generated_sync}
                      asccube.copy_l12l0b %108[%c0], %127[%c0], %c0_i32, %c0_i32, %c8_i32, %c16_i32, %c8_i32, %c8_i32 {transpose = true} : memref<?xf32, 4>, memref<?xf32, 2>
                      asccube.release_buf mte1, %c28_i64, 0 {cannir.buffer.generated_sync}
                      asccube.get_buf m, %c29_i64, 0 {cannir.buffer.generated_sync}
                      asccube.get_buf m, %c27_i64, 0 {cannir.buffer.generated_sync}
                      asccube.get_buf m, %c28_i64, 0 {cannir.buffer.generated_sync}
                      asccube.mmad %109[%c0], %107, %108, %c128_i32, %c128_i32, %c128_i32, %c0_i32 {c_matrix_init_val = true} : memref<?xf32, 5>, memref<?xf32, 3>, memref<?xf32, 4>
                      asccube.release_buf m, %c28_i64, 0 {cannir.buffer.generated_sync}
                      asccube.release_buf m, %c27_i64, 0 {cannir.buffer.generated_sync}
                      asccube.release_buf m, %c29_i64, 0 {cannir.buffer.generated_sync}
                    }
                    %124 = ascscalar.bitcast_memref %alloc_22 byte_offset = %c0_i64 : memref<16384xf32, 2> -> memref<?xf32, 2>
                    asccube.get_buf mte1, %c27_i64, 0 {cannir.buffer.generated_sync}
                    asccube.copy_l12l0a %107[%c0], %124[%c0], %c0_i32, %c0_i32, %c8_i32, %c16_i32, %c8_i32, %c8_i32 : memref<?xf32, 3>, memref<?xf32, 2>
                    asccube.release_buf mte1, %c27_i64, 0 {cannir.buffer.generated_sync}
                    %125 = ascscalar.bitcast_memref %alloc_24 byte_offset = %c0_i64 : memref<16384xf32, 2> -> memref<?xf32, 2>
                    asccube.get_buf mte1, %c28_i64, 0 {cannir.buffer.generated_sync}
                    asccube.copy_l12l0b %108[%c0], %125[%c0], %c0_i32, %c0_i32, %c8_i32, %c16_i32, %c8_i32, %c8_i32 {transpose = true} : memref<?xf32, 4>, memref<?xf32, 2>
                    asccube.release_buf mte1, %c28_i64, 0 {cannir.buffer.generated_sync}
                    asccube.sync_intra_arrive mte1, %c13_i64 {cannir.sync.banks = 3 : i32}
                    asccube.sync_intra_arrive mte1, %c29_i64 {cannir.sync.banks = 3 : i32}
                    asccube.get_buf m, %c30_i64, 0 {cannir.buffer.generated_sync}
                    asccube.get_buf m, %c27_i64, 0 {cannir.buffer.generated_sync}
                    asccube.get_buf m, %c28_i64, 0 {cannir.buffer.generated_sync}
                    asccube.mmad %115[%c0], %107, %108, %c128_i32, %c128_i32, %c128_i32, %c0_i32 {c_matrix_init_val = false} : memref<?xf32, 5>, memref<?xf32, 3>, memref<?xf32, 4>
                    asccube.release_buf m, %c28_i64, 0 {cannir.buffer.generated_sync}
                    asccube.release_buf m, %c27_i64, 0 {cannir.buffer.generated_sync}
                    asccube.release_buf m, %c30_i64, 0 {cannir.buffer.generated_sync}
                  }
                }
                %117 = arith.cmpi ne, %arg25, %74 : i32
                %118 = arith.select %103, %true, %117 : i1
                scf.if %118 {
                  asccube.get_buf fixpipe, %c29_i64, 0 {cannir.buffer.generated_sync}
                  scf.if %113 {
                    %120 = cannir.extract_physical_layout %arg14 : !cannir.tensor<<gm, f32>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, identity>
                    %121 = cannir.extract_stride %120 : <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>> -> <(-1, 16384, 128, 1)>
                    %122 = cannir.extract_dim %121[0] : !cannir.stride<(-1, 16384, 128, 1)>
                    %123 = arith.muli %102, %122 : i64
                    %124 = arith.index_cast %123 : i64 to index
                    %125 = arith.muli %arg24, %c16384_i64 : i64
                    %126 = arith.index_cast %125 : i64 to index
                    %127 = arith.addi %124, %126 : index
                    asccube.copy_l0c2gm %alloc_1[%127], %109[%c0], %c128_i32, %c128_i32, %c128_i32, %c128_i32, %c0_i32 l2_cache_ctl %c0_i32 {nz2nd_en = true, relu_en = false} : memref<1xf32>, memref<?xf32, 5>
                  } else {
                    %120 = ascscalar.bitcast_memref %alloc_23 byte_offset = %c0_i64 : memref<16384xf32, 2> -> memref<?xf32, 2>
                    asccube.copy_l0c2l1 %120[%c0], %109[%c0], %c128_i32, %c128_i32, %c1024_i32, %c128_i32, %c0_i32 {relu_en = false, split_en = true} : memref<?xf32, 2>, memref<?xf32, 5>
                  }
                  asccube.release_buf fixpipe, %c29_i64, 0 {cannir.buffer.generated_sync}
                }
                %119 = arith.select %114, %111, %false : i1
                scf.if %119 {
                  scf.if %113 {
                    %120 = ascscalar.bitcast_memref %alloc_9 byte_offset = %c0_i64 : memref<8192xf32, 1> -> memref<?xf32, 1>
                    asccube.sync_intra_wait fixpipe, %c15_i64 {cannir.sync.banks = 3 : i32}
                    asccube.sync_intra_wait fixpipe, %c31_i64 {cannir.sync.banks = 3 : i32}
                    asccube.get_buf fixpipe, %c30_i64, 0 {cannir.buffer.generated_sync}
                    asccube.copy_l0c2ub %120[%c0], %115[%c0], %c128_i32, %c128_i32, %c128_i32, %c128_i32, %c0_i32 sub_block_id %c0_i32 {dual_dst_ctl = 1 : i32, nz2nd_en = true, relu_en = false} : memref<?xf32, 1>, memref<?xf32, 5>
                    asccube.release_buf fixpipe, %c30_i64, 0 {cannir.buffer.generated_sync}
                    asccube.sync_intra_arrive fixpipe, %c15_i64 {cannir.sync.banks = 3 : i32}
                    asccube.sync_intra_arrive fixpipe, %c31_i64 {cannir.sync.banks = 3 : i32}
                  } else {
                    %120 = ascscalar.bitcast_memref %alloc_22 byte_offset = %c0_i64 : memref<16384xf32, 2> -> memref<?xf32, 2>
                    asccube.get_buf fixpipe, %c30_i64, 0 {cannir.buffer.generated_sync}
                    asccube.copy_l0c2l1 %120[%c0], %115[%c0], %c128_i32, %c128_i32, %c1024_i32, %c128_i32, %c0_i32 {relu_en = false, split_en = true} : memref<?xf32, 2>, memref<?xf32, 5>
                    asccube.release_buf fixpipe, %c30_i64, 0 {cannir.buffer.generated_sync}
                  }
                } else {
                  scf.if %113 {
                    %120 = ascscalar.bitcast_memref %alloc_9 byte_offset = %c0_i64 : memref<8192xf32, 1> -> memref<?xf32, 1>
                    asccube.sync_intra_wait fixpipe, %c15_i64 {cannir.sync.banks = 3 : i32}
                    asccube.sync_intra_wait fixpipe, %c31_i64 {cannir.sync.banks = 3 : i32}
                    asccube.get_buf fixpipe, %c30_i64, 0 {cannir.buffer.generated_sync}
                    asccube.copy_l0c2ub %120[%c0], %115[%c0], %c128_i32, %c128_i32, %c128_i32, %c128_i32, %c0_i32 sub_block_id %c0_i32 {dual_dst_ctl = 1 : i32, nz2nd_en = true, relu_en = false} : memref<?xf32, 1>, memref<?xf32, 5>
                    asccube.release_buf fixpipe, %c30_i64, 0 {cannir.buffer.generated_sync}
                    asccube.sync_intra_arrive fixpipe, %c15_i64 {cannir.sync.banks = 3 : i32}
                    asccube.sync_intra_arrive fixpipe, %c31_i64 {cannir.sync.banks = 3 : i32}
                  } else {
                    %120 = ascscalar.bitcast_memref %alloc_22 byte_offset = %c0_i64 : memref<16384xf32, 2> -> memref<?xf32, 2>
                    asccube.get_buf fixpipe, %c30_i64, 0 {cannir.buffer.generated_sync}
                    asccube.copy_l0c2l1 %120[%c0], %115[%c0], %c128_i32, %c128_i32, %c1024_i32, %c128_i32, %c0_i32 {relu_en = false, split_en = true} : memref<?xf32, 2>, memref<?xf32, 5>
                    asccube.release_buf fixpipe, %c30_i64, 0 {cannir.buffer.generated_sync}
                  }
                }
              }
            }
          }
        }
        %47 = arith.addi %arg22, %c1_i32 : i32
        %48 = arith.cmpi slt, %47, %11 : i32
        scf.if %48 {
          asccube.sync_all
          asccube.sync_block_wait s, %c1_i64
          asccube.sync_block_arrive fixpipe, %c5_i64 {mode = 0 : i32}
          asccube.sync_block_wait s, %c5_i64 {mode = 0 : i32}
          asccube.sync_block_arrive mte3, %c9_i64
        }
      }
      asccube.sync_intra_wait fixpipe, %c2_i64
      asccube.sync_intra_wait fixpipe, %c18_i64
      asccube.sync_intra_wait fixpipe, %c3_i64
      asccube.sync_intra_wait fixpipe, %c19_i64
      asccube.sync_intra_wait fixpipe, %c4_i64
      asccube.sync_intra_wait fixpipe, %c20_i64
      asccube.sync_intra_wait fixpipe, %c14_i64
      asccube.sync_intra_wait fixpipe, %c30_i64
      asccube.sync_intra_wait fixpipe, %c15_i64
      asccube.sync_intra_wait fixpipe, %c31_i64
    }
    return
  }
}
