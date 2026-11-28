module {
  func.func @run_m2_rope_e4d962d57d1c1ac21b87c776259cffb6(%arg0: !cannir.tensor<<gm, bf16>, <<(-1, 1024)>, <(1024, 1)>>, <<(-1, 1024)>, <(1024, 1)>>, identity>, %arg1: !cannir.tensor<<gm, bf16>, <<(640, 1024)>, <(1024, 1)>>, <<(64, 40, 16, 16)>, <(10240, 256, 16, 1)>>, nd2nz> {cannir.physical_shape_upper_bound = array<i64: 64, 40, 16, 16>}, %arg2: !cannir.tensor<<gm, bf16>, <<(384, 640)>, <(640, 1)>>, <<(40, 24, 16, 16)>, <(6144, 256, 16, 1)>>, nd2nz> {cannir.physical_shape_upper_bound = array<i64: 40, 24, 16, 16>}, %arg3: !cannir.tensor<<gm, bf16>, <<(576, 1024)>, <(1024, 1)>>, <<(64, 36, 16, 16)>, <(9216, 256, 16, 1)>>, nd2nz> {cannir.physical_shape_upper_bound = array<i64: 64, 36, 16, 16>}, %arg4: !cannir.tensor<<gm, bf16>, <<(2, 128, 512)>, <(65536, 512, 1)>>, <<(2, 128, 512)>, <(65536, 512, 1)>>, identity>, %arg5: !cannir.tensor<<gm, f32>, <<(1, 640)>, <(640, 1)>>, <<(1, 640)>, <(640, 1)>>, identity>, %arg6: !cannir.tensor<<gm, f32>, <<(1, 512)>, <(512, 1)>>, <<(1, 512)>, <(512, 1)>>, identity>, %arg7: !cannir.tensor<<gm, f8E4M3FN>, <<(-1, -1, 1, 576)>, <(-1, 576, 576, 1)>>, <<(-1, -1, 1, 576)>, <(-1, 576, 576, 1)>>, identity>, %arg8: !cannir.tensor<<gm, i64>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg9: !cannir.tensor<<gm, f32>, <<(1)>, <(1)>>, <<(1)>, <(1)>>, identity>, %arg10: !cannir.tensor<<gm, bf16>, <<(-1, 64)>, <(64, 1)>>, <<(-1, 64)>, <(64, 1)>>, identity>, %arg11: !cannir.tensor<<gm, bf16>, <<(-1, 64)>, <(64, 1)>>, <<(-1, 64)>, <(64, 1)>>, identity>, %arg12: !cannir.tensor<<gm, f8E4M3FN>, <<(8, 2, 576)>, <(1152, 576, 1)>>, <<(8, 2, 576)>, <(1152, 576, 1)>>, identity>, %arg13: !cannir.tensor<<gm, f32>, <<(-1, 2, 1)>, <(2, 1, 1)>>, <<(-1, 2, 1)>, <(2, 1, 1)>>, identity>, %arg14: !cannir.tensor<<gm, bf16>, <<(8, 1024)>, <(1024, 1)>>, <<(8, 1024)>, <(1024, 1)>>, identity>, %arg15: !cannir.tensor<<gm, bf16>, <<(8, 672)>, <(672, 1)>>, <<(8, 672)>, <(672, 1)>>, identity>, %arg16: !cannir.tensor<<gm, bf16>, <<(8, 576)>, <(576, 1)>>, <<(8, 576)>, <(576, 1)>>, identity>, %arg17: !cannir.tensor<<gm, bf16>, <<(8, 384)>, <(384, 1)>>, <<(8, 384)>, <(384, 1)>>, identity>, %arg18: !cannir.tensor<<gm, f32>, <<(8, 2, 576)>, <(1152, 576, 1)>>, <<(8, 2, 576)>, <(1152, 576, 1)>>, identity>, %arg19: !cannir.tensor<<gm, bf16>, <<(8, 640)>, <(640, 1)>>, <<(8, 640)>, <(640, 1)>>, identity>, %arg20: f32, %arg21: f32, %arg22: f32, %arg23: i64, %arg24: i64, %arg25: i64) attributes {cannbotdsl.contract_digest = "e4d962d57d1c1ac21b87c776259cffb6a10aacd19c478a55b6f94a0f34ff27f3", cannbotdsl.public_slot_count = 26 : i64, cannbotdsl.slot_dtype_bits = [16, 16, 16, 16, 16, 32, 32, 8, 64, 32, 16, 16, 8, 32, 16, 16, 16, 16, 32, 16, 32, 32, 32, 64, 64, 64], cannbotdsl.slot_dtype_codes = [4, 4, 4, 4, 4, 2, 2, 10, 0, 2, 4, 4, 10, 2, 4, 4, 4, 4, 2, 4, 2, 2, 2, 0, 0, 0], cannbotdsl.slot_dtype_lanes = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], cannbotdsl.slot_flags = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], cannbotdsl.slot_kinds = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2], cannbotdsl.slot_type_digests = ["2f12152648aa422671a77bacab9acff4420df5a14464c5e39a59e14b8782bbea", "581f88679ffc97d9b0f852494dc15761661352bc3407e73adccace68b7a1ca11", "5f40ba9feb005fcebe9f20699ffbeaa2b505cb034add8e47d2c39973ea7d7d94", "16c9273b7b8e5eb2c2f9bf216b7eaf76bb4b8b6c63e2002f32fadf24ceba8b86", "acf211e93f52c0b0a29534f094bc8c996dd6988e8cfde7cbbc30b19ec2937547", "e060b4eeab5e7c9d8b67a3153f869de5aad4eee9553a9a98b398bf5577804a47", "7b962932eb0067670c47bf914f64cc35656e412a09850be9b2bc297c5b918fc3", "833aa9f9910b038d7d3a667cdb22a773c7166d5f3f362f230b8bd93de439aaf2", "5ae7da10e83ac3f5ad55bbe47c978e5c8bcb0ad1d2171af81149eef24de96b7a", "b134b54071cdca5322ca9f5d39ee9881e5a8f90b566373e7bf45d235577c8f89", "6b6a90f195a5f29ca792af7b1733c8cafee2bfdeb6c69f295bfd0b7b7ca2d0e2", "6b6a90f195a5f29ca792af7b1733c8cafee2bfdeb6c69f295bfd0b7b7ca2d0e2", "52754354a8c188303e060fa5f0d34e20976a55003d8e0ba2a5e4e2c2cdc65a2e", "1eeded4256175c5337b99a56bcef6c18ad1426f6d5179b130459e661d06a8e53", "8aeb126f8f3c3f165d2516841db0a82c99e1ba495f1076f23cb55e200c479bd2", "06228e85126fccbf1e33641ff0d06b682c62d20f61aaafc43891d07214e515be", "242b167e7f801f9db314d371068618a3ce8621068571e0b424293920276070d9", "0b1a744b30fa25ea0232e0e52d3b5653d1fd59c701e9d62d0f81b2ed3e1bcecd", "14adfafe70bd6c5a30a53fa995a258d1ad13aacc9361552676c62b01ea58e341", "529f05557b642e904963b68de7bd0d051244ee009b053b90be0a92bba9011258", "4a5363e2ec1eb8cefee037ab0b231653c07f9b44a406ef92ed9ba3e6f5541dbf", "4a5363e2ec1eb8cefee037ab0b231653c07f9b44a406ef92ed9ba3e6f5541dbf", "4a5363e2ec1eb8cefee037ab0b231653c07f9b44a406ef92ed9ba3e6f5541dbf", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515"], cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.host} {
    cannir.kernel_launch @kernel_MlaPrologKernel_run_m2_rope_0(%arg25) [%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18, %arg19, %arg20, %arg21, %arg22, %arg23, %arg24] {cannbotdsl.profile = {indices = array<i64: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19>, name = "MlaPrologKernel_run_m2_rope", op_type = "MlaPrologKernel_run_m2_rope", roles = array<i64: 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1>}} : !cannir.tensor<<gm, bf16>, <<(-1, 1024)>, <(1024, 1)>>, <<(-1, 1024)>, <(1024, 1)>>, identity>, !cannir.tensor<<gm, bf16>, <<(640, 1024)>, <(1024, 1)>>, <<(64, 40, 16, 16)>, <(10240, 256, 16, 1)>>, nd2nz>, !cannir.tensor<<gm, bf16>, <<(384, 640)>, <(640, 1)>>, <<(40, 24, 16, 16)>, <(6144, 256, 16, 1)>>, nd2nz>, !cannir.tensor<<gm, bf16>, <<(576, 1024)>, <(1024, 1)>>, <<(64, 36, 16, 16)>, <(9216, 256, 16, 1)>>, nd2nz>, !cannir.tensor<<gm, bf16>, <<(2, 128, 512)>, <(65536, 512, 1)>>, <<(2, 128, 512)>, <(65536, 512, 1)>>, identity>, !cannir.tensor<<gm, f32>, <<(1, 640)>, <(640, 1)>>, <<(1, 640)>, <(640, 1)>>, identity>, !cannir.tensor<<gm, f32>, <<(1, 512)>, <(512, 1)>>, <<(1, 512)>, <(512, 1)>>, identity>, !cannir.tensor<<gm, f8E4M3FN>, <<(-1, -1, 1, 576)>, <(-1, 576, 576, 1)>>, <<(-1, -1, 1, 576)>, <(-1, 576, 576, 1)>>, identity>, !cannir.tensor<<gm, i64>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, !cannir.tensor<<gm, f32>, <<(1)>, <(1)>>, <<(1)>, <(1)>>, identity>, !cannir.tensor<<gm, bf16>, <<(-1, 64)>, <(64, 1)>>, <<(-1, 64)>, <(64, 1)>>, identity>, !cannir.tensor<<gm, bf16>, <<(-1, 64)>, <(64, 1)>>, <<(-1, 64)>, <(64, 1)>>, identity>, !cannir.tensor<<gm, f8E4M3FN>, <<(8, 2, 576)>, <(1152, 576, 1)>>, <<(8, 2, 576)>, <(1152, 576, 1)>>, identity>, !cannir.tensor<<gm, f32>, <<(-1, 2, 1)>, <(2, 1, 1)>>, <<(-1, 2, 1)>, <(2, 1, 1)>>, identity>, !cannir.tensor<<gm, bf16>, <<(8, 1024)>, <(1024, 1)>>, <<(8, 1024)>, <(1024, 1)>>, identity>, !cannir.tensor<<gm, bf16>, <<(8, 672)>, <(672, 1)>>, <<(8, 672)>, <(672, 1)>>, identity>, !cannir.tensor<<gm, bf16>, <<(8, 576)>, <(576, 1)>>, <<(8, 576)>, <(576, 1)>>, identity>, !cannir.tensor<<gm, bf16>, <<(8, 384)>, <(384, 1)>>, <<(8, 384)>, <(384, 1)>>, identity>, !cannir.tensor<<gm, f32>, <<(8, 2, 576)>, <(1152, 576, 1)>>, <<(8, 2, 576)>, <(1152, 576, 1)>>, identity>, !cannir.tensor<<gm, bf16>, <<(8, 640)>, <(640, 1)>>, <<(8, 640)>, <(640, 1)>>, identity>, f32, f32, f32, i64, i64
    return
  }
  func.func @kernel_MlaPrologKernel_run_m2_rope_0_aiv(%arg0: !cannir.tensor<<gm, bf16>, <<(-1, 1024)>, <(1024, 1)>>, <<(-1, 1024)>, <(1024, 1)>>, identity>, %arg1: !cannir.tensor<<gm, bf16>, <<(640, 1024)>, <(1024, 1)>>, <<(64, 40, 16, 16)>, <(10240, 256, 16, 1)>>, nd2nz> {cannir.physical_shape_upper_bound = array<i64: 64, 40, 16, 16>}, %arg2: !cannir.tensor<<gm, bf16>, <<(384, 640)>, <(640, 1)>>, <<(40, 24, 16, 16)>, <(6144, 256, 16, 1)>>, nd2nz> {cannir.physical_shape_upper_bound = array<i64: 40, 24, 16, 16>}, %arg3: !cannir.tensor<<gm, bf16>, <<(576, 1024)>, <(1024, 1)>>, <<(64, 36, 16, 16)>, <(9216, 256, 16, 1)>>, nd2nz> {cannir.physical_shape_upper_bound = array<i64: 64, 36, 16, 16>}, %arg4: !cannir.tensor<<gm, bf16>, <<(2, 128, 512)>, <(65536, 512, 1)>>, <<(2, 128, 512)>, <(65536, 512, 1)>>, identity>, %arg5: !cannir.tensor<<gm, f32>, <<(1, 640)>, <(640, 1)>>, <<(1, 640)>, <(640, 1)>>, identity>, %arg6: !cannir.tensor<<gm, f32>, <<(1, 512)>, <(512, 1)>>, <<(1, 512)>, <(512, 1)>>, identity>, %arg7: !cannir.tensor<<gm, f8E4M3FN>, <<(-1, -1, 1, 576)>, <(-1, 576, 576, 1)>>, <<(-1, -1, 1, 576)>, <(-1, 576, 576, 1)>>, identity>, %arg8: !cannir.tensor<<gm, i64>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg9: !cannir.tensor<<gm, f32>, <<(1)>, <(1)>>, <<(1)>, <(1)>>, identity>, %arg10: !cannir.tensor<<gm, bf16>, <<(-1, 64)>, <(64, 1)>>, <<(-1, 64)>, <(64, 1)>>, identity>, %arg11: !cannir.tensor<<gm, bf16>, <<(-1, 64)>, <(64, 1)>>, <<(-1, 64)>, <(64, 1)>>, identity>, %arg12: !cannir.tensor<<gm, f8E4M3FN>, <<(8, 2, 576)>, <(1152, 576, 1)>>, <<(8, 2, 576)>, <(1152, 576, 1)>>, identity>, %arg13: !cannir.tensor<<gm, f32>, <<(-1, 2, 1)>, <(2, 1, 1)>>, <<(-1, 2, 1)>, <(2, 1, 1)>>, identity>, %arg14: !cannir.tensor<<gm, bf16>, <<(8, 1024)>, <(1024, 1)>>, <<(8, 1024)>, <(1024, 1)>>, identity>, %arg15: !cannir.tensor<<gm, bf16>, <<(8, 672)>, <(672, 1)>>, <<(8, 672)>, <(672, 1)>>, identity>, %arg16: !cannir.tensor<<gm, bf16>, <<(8, 576)>, <(576, 1)>>, <<(8, 576)>, <(576, 1)>>, identity>, %arg17: !cannir.tensor<<gm, bf16>, <<(8, 384)>, <(384, 1)>>, <<(8, 384)>, <(384, 1)>>, identity>, %arg18: !cannir.tensor<<gm, f32>, <<(8, 2, 576)>, <(1152, 576, 1)>>, <<(8, 2, 576)>, <(1152, 576, 1)>>, identity>, %arg19: !cannir.tensor<<gm, bf16>, <<(8, 640)>, <(640, 1)>>, <<(8, 640)>, <(640, 1)>>, identity>, %arg20: f32, %arg21: f32, %arg22: f32, %arg23: i64, %arg24: i64) attributes {cannir.buffer.auto_sync, cannir.buffer.sync_analyzed, cannir.buffer.sync_verified, cannir.buffer.tensor_state, cannir.core = "aiv", cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.device, cannir.explicit_ring_buffer_cursors, cannir.kernel_group = "kernel_MlaPrologKernel_run_m2_rope_0", cannir.resource.ids_prepared} {
    %c4_i32 = arith.constant 4 : i32
    %c256_i32 = arith.constant 256 : i32
    %c1280_i32 = arith.constant 1280 : i32
    %c2048_i32 = arith.constant 2048 : i32
    %c0_i32 = arith.constant 0 : i32
    %c2048_i64 = arith.constant 2048 : i64
    %c16384_i64 = arith.constant 16384 : i64
    %c0 = arith.constant 0 : index
    %c1_i32 = arith.constant 1 : i32
    %c2560_i64 = arith.constant 2560 : i64
    %c1344_i64 = arith.constant 1344 : i64
    %c1280_i64 = arith.constant 1280 : i64
    %c4096_i64 = arith.constant 4096 : i64
    %c1152_i64 = arith.constant 1152 : i64
    %c576_i32 = arith.constant 576 : i32
    %c128_i64 = arith.constant 128 : i64
    %c32_i32 = arith.constant 32 : i32
    %c192_i64 = arith.constant 192 : i64
    %c768_i64 = arith.constant 768 : i64
    %c576_i64 = arith.constant 576 : i64
    %c512_i64 = arith.constant 512 : i64
    %c4608_i64 = arith.constant 4608 : i64
    %c2304_i64 = arith.constant 2304 : i64
    %c10240_i64 = arith.constant 10240 : i64
    %c256_i64 = arith.constant 256 : i64
    %c64_i64 = arith.constant 64 : i64
    %c32_i64 = arith.constant 32 : i64
    %c18_i64 = arith.constant 18 : i64
    %c15_i64 = arith.constant 15 : i64
    %c14_i64 = arith.constant 14 : i64
    %c13_i64 = arith.constant 13 : i64
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
    %alloc_0 = memref.alloc() {cannir.arg_index = 5 : i64} : memref<640xf32>
    %alloc_1 = memref.alloc() {cannir.arg_index = 6 : i64} : memref<512xf32>
    %alloc_2 = memref.alloc() {cannir.arg_index = 7 : i64} : memref<1xf8E4M3FN>
    %alloc_3 = memref.alloc() {cannir.arg_index = 8 : i64} : memref<1xi64>
    %alloc_4 = memref.alloc() {cannir.arg_index = 9 : i64} : memref<1xf32>
    %alloc_5 = memref.alloc() {cannir.arg_index = 10 : i64} : memref<1xbf16>
    %alloc_6 = memref.alloc() {cannir.arg_index = 11 : i64} : memref<1xbf16>
    %alloc_7 = memref.alloc() {cannir.arg_index = 12 : i64} : memref<9216xf8E4M3FN>
    %alloc_8 = memref.alloc() {cannir.arg_index = 13 : i64} : memref<1xf32>
    %alloc_9 = memref.alloc() {cannir.arg_index = 14 : i64} : memref<8192xbf16>
    %alloc_10 = memref.alloc() {cannir.arg_index = 15 : i64} : memref<5376xbf16>
    %alloc_11 = memref.alloc() {cannir.arg_index = 16 : i64} : memref<4608xbf16>
    %alloc_12 = memref.alloc() {cannir.arg_index = 17 : i64} : memref<3072xbf16>
    %alloc_13 = memref.alloc() {cannir.arg_index = 18 : i64} : memref<9216xf32>
    %alloc_14 = memref.alloc() {cannir.arg_index = 19 : i64} : memref<5120xbf16>
    %alloc_15 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<8192xbf16, 1>
    %alloc_16 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<2048xbf16, 1>
    %alloc_17 = memref.alloc() {cannir.byte_offset = 4096 : i64} : memref<2048xbf16, 1>
    %alloc_18 = memref.alloc() {cannir.byte_offset = 8192 : i64} : memref<2048xf32, 1>
    %alloc_19 = memref.alloc() {cannir.byte_offset = 16384 : i64} : memref<2048xf32, 1>
    %alloc_20 = memref.alloc() {cannir.byte_offset = 24576 : i64} : memref<2048xf32, 1>
    %alloc_21 = memref.alloc() {cannir.byte_offset = 49152 : i64} : memref<2048xbf16, 1>
    %alloc_22 = memref.alloc() {cannir.byte_offset = 53248 : i64} : memref<2048xbf16, 1>
    %alloc_23 = memref.alloc() {cannir.byte_offset = 57344 : i64} : memref<8xf32, 1>
    %alloc_24 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<576xbf16, 1>
    %alloc_25 = memref.alloc() {cannir.byte_offset = 1152 : i64} : memref<576xbf16, 1>
    %alloc_26 = memref.alloc() {cannir.byte_offset = 2304 : i64} : memref<576xf32, 1>
    %alloc_27 = memref.alloc() {cannir.byte_offset = 4608 : i64} : memref<576xf32, 1>
    %alloc_28 = memref.alloc() {cannir.byte_offset = 6912 : i64} : memref<512xf32, 1>
    %alloc_29 = memref.alloc() {cannir.byte_offset = 12608 : i64} : memref<576xf8E4M3FN, 1>
    %alloc_30 = memref.alloc() {cannir.byte_offset = 13184 : i64} : memref<576xf8E4M3FN, 1>
    %alloc_31 = memref.alloc() {cannir.byte_offset = 14912 : i64} : memref<64xf32, 1>
    %alloc_32 = memref.alloc() {cannir.byte_offset = 15168 : i64} : memref<8xf32, 1>
    %alloc_33 = memref.alloc() {cannir.byte_offset = 15200 : i64} : memref<64xbf16, 1>
    %alloc_34 = memref.alloc() {cannir.byte_offset = 15328 : i64} : memref<64xbf16, 1>
    %alloc_35 = memref.alloc() {cannir.byte_offset = 15456 : i64} : memref<128xf32, 1>
    %alloc_36 = memref.alloc() {cannir.byte_offset = 15968 : i64} : memref<128xf32, 1>
    %alloc_37 = memref.alloc() {cannir.byte_offset = 16480 : i64} : memref<128xf32, 1>
    %alloc_38 = memref.alloc() {cannir.byte_offset = 16992 : i64} : memref<64xf32, 1>
    %alloc_39 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<64xbf16, 1>
    %alloc_40 = memref.alloc() {cannir.byte_offset = 128 : i64} : memref<64xbf16, 1>
    %alloc_41 = memref.alloc() {cannir.byte_offset = 256 : i64} : memref<64xf32, 1>
    %alloc_42 = memref.alloc() {cannir.byte_offset = 512 : i64} : memref<64xf32, 1>
    %alloc_43 = memref.alloc() {cannir.byte_offset = 768 : i64} : memref<64xf32, 1>
    %alloc_44 = memref.alloc() {cannir.byte_offset = 1024 : i64} : memref<64xbf16, 1>
    %alloc_45 = memref.alloc() {cannir.byte_offset = 1152 : i64} : memref<64xbf16, 1>
    %alloc_46 = memref.alloc() {cannir.byte_offset = 1280 : i64} : memref<128xf32, 1>
    %alloc_47 = memref.alloc() {cannir.byte_offset = 1792 : i64} : memref<128xf32, 1>
    %alloc_48 = memref.alloc() {cannir.byte_offset = 2304 : i64} : memref<128xf32, 1>
    %alloc_49 = memref.alloc() {cannir.byte_offset = 2816 : i64} : memref<64xf32, 1>
    %alloc_50 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<576xf32, 1>
    %alloc_51 = memref.alloc() {cannir.byte_offset = 2304 : i64} : memref<576xf32, 1>
    %alloc_52 = memref.alloc() {cannir.byte_offset = 4608 : i64} : memref<576xf8E4M3FN, 1>
    %alloc_53 = memref.alloc() {cannir.byte_offset = 5184 : i64} : memref<576xf8E4M3FN, 1>
    %alloc_54 = memref.alloc() {cannir.byte_offset = 5760 : i64} : memref<8xf32, 1>
    %0 = ascscalar.get_block_idx
    %1 = ascscalar.get_block_num
    %2 = ascscalar.get_subblock_id
    %3 = cannir.extract_layout %arg0 : !cannir.tensor<<gm, bf16>, <<(-1, 1024)>, <(1024, 1)>>, <<(-1, 1024)>, <(1024, 1)>>, identity>
    %4 = cannir.extract_shape %3 : <<(-1, 1024)>, <(1024, 1)>> -> <(-1, 1024)>
    %5 = cannir.extract_dim %4[0] : !cannir.shape<(-1, 1024)>
    %6 = arith.muli %0, %c2_i64 : i64
    %7 = arith.addi %6, %2 : i64
    %8 = arith.cmpi ne, %5, %c8_i64 : i64
    scf.if %8 {
      %17 = arith.muli %1, %c2_i64 : i64
      scf.for %arg25 = %7 to %c8_i64 step %17  : i64 {
        %18 = arith.cmpi slt, %arg25, %5 : i64
        scf.if %18 {
          %19 = ascscalar.bitcast_memref %alloc_15 byte_offset = %c0_i64 : memref<8192xbf16, 1> -> memref<?xbf16, 1>
          %20 = arith.minsi %arg25, %5 : i64
          %21 = arith.subi %5, %20 : i64
          %22 = arith.minsi %21, %c1_i64 : i64
          %23 = arith.cmpi sle, %arg25, %5 : i64
          %24 = arith.select %23, %22, %c0_i64 : i64
          %25 = arith.muli %arg25, %c2048_i64 : i64
          ascvec.get_buf mte2, %c0_i64, 0 {cannir.buffer.generated_sync}
          "ascvec.copy_gm2ub_plan"(%19, %alloc, %c0_i32, %c0_i32, %c0_i64, %25, %24, %c2048_i64, %c2048_i64, %c16384_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = 0 : i64, ub_capacity = 16384 : i64}> : (memref<?xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
          ascvec.release_buf mte2, %c0_i64, 0 {cannir.buffer.generated_sync}
          %26 = arith.muli %arg25, %c16384_i64 : i64
          %27 = ascscalar.memref_address %alloc_9 : memref<8192xbf16>
          %28 = arith.divui %26, %c8_i64 : i64
          %29 = arith.addi %27, %28 : i64
          %30 = ascscalar.bind_gm_ptr %29 : memref<1xbf16>
          ascvec.get_buf mte3, %c0_i64, 0 {cannir.buffer.generated_sync}
          ascvec.copy_ub2gm %30[%c0], %19[%c0], %c1_i32, %c2048_i32, %c0_i64, %c2048_i32 l2_cache_ctl %c0_i32 : memref<1xbf16>, memref<?xbf16, 1>
          ascvec.release_buf mte3, %c0_i64, 0 {cannir.buffer.generated_sync}
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
            %c0_55 = arith.constant 0 : index
            %cst = arith.constant 0.000000e+00 : bf16
            %c128_i32 = arith.constant 128 : i32
            %c0_56 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            scf.for %arg27 = %c0_56 to %c1 step %c1 {
              %mask, %remaining_out = ascvec.update_mask %c128_i32 elem_bits 16 : !ascvec.mask
              %25 = ascvec.duplicate_scalar %cst, %mask : bf16 -> <bf16>
              ascvec.storealign %arg26[%c0_55], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c128], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c256], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c384], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c512], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c640], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c768], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c896], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c1024], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c1152], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c1280], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c1408], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c1536], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c1664], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c1792], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c1920], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c2048], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c2176], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c2304], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c2432], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c2560], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c2688], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c2816], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c2944], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c3072], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c3200], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c3328], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c3456], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c3584], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c3712], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c3840], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c3968], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c4096], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c4224], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c4352], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c4480], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c4608], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c4736], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c4864], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c4992], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c5120], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c5248], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c5376], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c5504], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c5632], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c5760], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c5888], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c6016], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c6144], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c6272], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c6400], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c6528], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c6656], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c6784], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c6912], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c7040], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c7168], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c7296], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c7424], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c7552], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c7680], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c7808], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c7936], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg26[%c8064], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
            }
            ascvec.mem_bar vst_vld
          } {cannir.vf_mode = "simd"}
          ascvec.release_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
          %19 = arith.muli %arg25, %c16384_i64 : i64
          %20 = ascscalar.memref_address %alloc_9 : memref<8192xbf16>
          %21 = arith.divui %19, %c8_i64 : i64
          %22 = arith.addi %20, %21 : i64
          %23 = ascscalar.bind_gm_ptr %22 : memref<1xbf16>
          %24 = ascscalar.bitcast_memref %alloc_15 byte_offset = %c0_i64 : memref<8192xbf16, 1> -> memref<?xbf16, 1>
          ascvec.get_buf mte3, %c0_i64, 0 {cannir.buffer.generated_sync}
          ascvec.copy_ub2gm %23[%c0], %24[%c0], %c1_i32, %c2048_i32, %c0_i64, %c2048_i32 l2_cache_ctl %c0_i32 : memref<1xbf16>, memref<?xbf16, 1>
          ascvec.release_buf mte3, %c0_i64, 0 {cannir.buffer.generated_sync}
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
    %9 = arith.cmpi slt, %7, %5 : i64
    %10 = scf.if %9 -> (i64) {
      %17 = arith.muli %1, %c2_i64 : i64
      %18 = ascscalar.bitcast_memref %alloc_20 byte_offset = %c0_i64 : memref<2048xf32, 1> -> memref<?xf32, 1>
      ascvec.get_buf mte2, %c2_i64, 0 {cannir.buffer.generated_sync}
      "ascvec.copy_gm2ub_plan"(%18, %alloc_0, %c0_i32, %c0_i32, %c0_i64, %c0_i64, %c1_i64, %c2560_i64, %c0_i64, %c2560_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 2560 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 24576 : i64, ub_capacity = 8192 : i64}> : (memref<?xf32, 1>, memref<640xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
      ascvec.release_buf mte2, %c2_i64, 0 {cannir.buffer.generated_sync}
      %19 = scf.for %arg25 = %7 to %5 step %17 iter_args(%arg26 = %c0_i64) -> (i64)  : i64 {
        %20 = arith.cmpi eq, %arg26, %c0_i64 : i64
        %21 = ascscalar.memref_address %alloc_16 : memref<2048xbf16, 1>
        %22 = ascscalar.memref_address %alloc_17 : memref<2048xbf16, 1>
        %23 = arith.select %20, %21, %22 : i64
        %24 = arith.cmpi ne, %arg26, %c0_i64 : i64
        %25 = arith.extui %24 : i1 to i64
        %26 = arith.addi %arg26, %c1_i64 overflow<nsw> : i64
        %27 = arith.remui %26, %c2_i64 : i64
        %28 = arith.minsi %arg25, %c8_i64 : i64
        %29 = arith.subi %c8_i64, %28 : i64
        %30 = arith.minsi %29, %c1_i64 : i64
        %31 = arith.cmpi sle, %arg25, %c8_i64 : i64
        %32 = arith.select %31, %30, %c0_i64 : i64
        %33 = arith.muli %arg25, %c1344_i64 : i64
        %34 = ascscalar.bind_local_ptr %23 : memref<1xbf16, 1>
        ascvec.get_buf mte2, %25, 0 {cannir.buffer.generated_sync}
        "ascvec.copy_gm2ub_plan"(%34, %alloc_10, %c0_i32, %c0_i32, %c0_i64, %33, %32, %c1280_i64, %c1344_i64, %c4096_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 10752 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<5376xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
        ascvec.release_buf mte2, %25, 0 {cannir.buffer.generated_sync}
        %35 = ascscalar.bind_local_ptr %23 : memref<2048xbf16, 1>
        %36 = ascscalar.memref_address %alloc_18 : memref<2048xf32, 1>
        %37 = ascscalar.memref_address %alloc_19 : memref<2048xf32, 1>
        %38 = arith.select %20, %36, %37 : i64
        %39 = ascscalar.bind_local_ptr %38 : memref<2048xf32, 1>
        ascvec.get_buf v, %25, 0 {cannir.buffer.generated_sync}
        ascvec.vf(%35, %39 : memref<2048xbf16, 1>, memref<2048xf32, 1>) {
        ^bb0(%arg27: memref<2048xbf16, 1>, %arg28: memref<2048xf32, 1>):
          %c64_i32 = arith.constant 64 : i32
          %c640_i32 = arith.constant 640 : i32
          %c0_i32_55 = arith.constant 0 : i32
          scf.for %arg29 = %c0_i32_55 to %c640_i32 step %c64_i32  : i32 {
            %51 = arith.index_cast %arg29 : i32 to index
            %52 = ascvec.loadalign_unpack %arg27[%51] {mode = "b16_to_b32"} : memref<2048xbf16, 1>, index -> <bf16>
            %53 = ascvec.create_mask : !ascvec.mask
            %54 = ascvec.cast %52, %53 {rounding = 6 : i32} : <bf16> -> <f32>
            ascvec.storealign %arg28[%51], %54, %53 : memref<2048xf32, 1>, index, <f32>
          }
          ascvec.mem_bar vst_vld
        } {cannir.vf_mode = "simd"}
        ascvec.release_buf v, %25, 0 {cannir.buffer.generated_sync}
        ascvec.vf(%39, %arg20, %alloc_23 : memref<2048xf32, 1>, f32, memref<8xf32, 1>) {
        ^bb0(%arg27: memref<2048xf32, 1>, %arg28: f32, %arg29: memref<8xf32, 1>):
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
          %cst = arith.constant 1.562500e-03 : f32
          %c0_i32_55 = arith.constant 0 : i32
          %c0_56 = arith.constant 0 : index
          %c64_i32 = arith.constant 64 : i32
          %cst_57 = arith.constant 0.000000e+00 : f32
          %c1_i32_58 = arith.constant 1 : i32
          %c0_59 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          scf.for %arg30 = %c0_59 to %c1 step %c1 {
            %mask, %remaining_out = ascvec.update_mask %c1_i32_58 : !ascvec.mask
            %51 = ascvec.duplicate_scalar %cst_57, %mask : f32 -> <f32>
            %mask_60, %remaining_out_61 = ascvec.update_mask %c64_i32 : !ascvec.mask
            %52 = ascvec.loadalign %arg27[%c0_56] : memref<2048xf32, 1>, index -> <f32>
            %53 = ascvec.mul %52, %52, %mask_60 : <f32>
            %54 = ascvec.reduce_sum %53, %mask_60 : <f32>
            %55 = ascvec.add %51, %54, %mask : <f32>
            %mask_62, %remaining_out_63 = ascvec.update_mask %c64_i32 : !ascvec.mask
            %56 = ascvec.loadalign %arg27[%c64] : memref<2048xf32, 1>, index -> <f32>
            %57 = ascvec.mul %56, %56, %mask_62 : <f32>
            %58 = ascvec.reduce_sum %57, %mask_62 : <f32>
            %59 = ascvec.add %55, %58, %mask : <f32>
            %mask_64, %remaining_out_65 = ascvec.update_mask %c64_i32 : !ascvec.mask
            %60 = ascvec.loadalign %arg27[%c128] : memref<2048xf32, 1>, index -> <f32>
            %61 = ascvec.mul %60, %60, %mask_64 : <f32>
            %62 = ascvec.reduce_sum %61, %mask_64 : <f32>
            %63 = ascvec.add %59, %62, %mask : <f32>
            %mask_66, %remaining_out_67 = ascvec.update_mask %c64_i32 : !ascvec.mask
            %64 = ascvec.loadalign %arg27[%c192] : memref<2048xf32, 1>, index -> <f32>
            %65 = ascvec.mul %64, %64, %mask_66 : <f32>
            %66 = ascvec.reduce_sum %65, %mask_66 : <f32>
            %67 = ascvec.add %63, %66, %mask : <f32>
            %mask_68, %remaining_out_69 = ascvec.update_mask %c64_i32 : !ascvec.mask
            %68 = ascvec.loadalign %arg27[%c256] : memref<2048xf32, 1>, index -> <f32>
            %69 = ascvec.mul %68, %68, %mask_68 : <f32>
            %70 = ascvec.reduce_sum %69, %mask_68 : <f32>
            %71 = ascvec.add %67, %70, %mask : <f32>
            %mask_70, %remaining_out_71 = ascvec.update_mask %c64_i32 : !ascvec.mask
            %72 = ascvec.loadalign %arg27[%c320] : memref<2048xf32, 1>, index -> <f32>
            %73 = ascvec.mul %72, %72, %mask_70 : <f32>
            %74 = ascvec.reduce_sum %73, %mask_70 : <f32>
            %75 = ascvec.add %71, %74, %mask : <f32>
            %mask_72, %remaining_out_73 = ascvec.update_mask %c64_i32 : !ascvec.mask
            %76 = ascvec.loadalign %arg27[%c384] : memref<2048xf32, 1>, index -> <f32>
            %77 = ascvec.mul %76, %76, %mask_72 : <f32>
            %78 = ascvec.reduce_sum %77, %mask_72 : <f32>
            %79 = ascvec.add %75, %78, %mask : <f32>
            %mask_74, %remaining_out_75 = ascvec.update_mask %c64_i32 : !ascvec.mask
            %80 = ascvec.loadalign %arg27[%c448] : memref<2048xf32, 1>, index -> <f32>
            %81 = ascvec.mul %80, %80, %mask_74 : <f32>
            %82 = ascvec.reduce_sum %81, %mask_74 : <f32>
            %83 = ascvec.add %79, %82, %mask : <f32>
            %mask_76, %remaining_out_77 = ascvec.update_mask %c64_i32 : !ascvec.mask
            %84 = ascvec.loadalign %arg27[%c512] : memref<2048xf32, 1>, index -> <f32>
            %85 = ascvec.mul %84, %84, %mask_76 : <f32>
            %86 = ascvec.reduce_sum %85, %mask_76 : <f32>
            %87 = ascvec.add %83, %86, %mask : <f32>
            %mask_78, %remaining_out_79 = ascvec.update_mask %c64_i32 : !ascvec.mask
            %88 = ascvec.loadalign %arg27[%c576] : memref<2048xf32, 1>, index -> <f32>
            %89 = ascvec.mul %88, %88, %mask_78 : <f32>
            %90 = ascvec.reduce_sum %89, %mask_78 : <f32>
            %91 = ascvec.add %87, %90, %mask : <f32>
            %mask_80, %remaining_out_81 = ascvec.update_mask %c0_i32_55 : !ascvec.mask
            %92 = ascvec.loadalign %arg27[%c640] : memref<2048xf32, 1>, index -> <f32>
            %93 = ascvec.mul %92, %92, %mask_80 : <f32>
            %94 = ascvec.reduce_sum %93, %mask_80 : <f32>
            %95 = ascvec.add %91, %94, %mask : <f32>
            %mask_82, %remaining_out_83 = ascvec.update_mask %c0_i32_55 : !ascvec.mask
            %96 = ascvec.loadalign %arg27[%c704] : memref<2048xf32, 1>, index -> <f32>
            %97 = ascvec.mul %96, %96, %mask_82 : <f32>
            %98 = ascvec.reduce_sum %97, %mask_82 : <f32>
            %99 = ascvec.add %95, %98, %mask : <f32>
            %mask_84, %remaining_out_85 = ascvec.update_mask %c0_i32_55 : !ascvec.mask
            %100 = ascvec.loadalign %arg27[%c768] : memref<2048xf32, 1>, index -> <f32>
            %101 = ascvec.mul %100, %100, %mask_84 : <f32>
            %102 = ascvec.reduce_sum %101, %mask_84 : <f32>
            %103 = ascvec.add %99, %102, %mask : <f32>
            %mask_86, %remaining_out_87 = ascvec.update_mask %c0_i32_55 : !ascvec.mask
            %104 = ascvec.loadalign %arg27[%c832] : memref<2048xf32, 1>, index -> <f32>
            %105 = ascvec.mul %104, %104, %mask_86 : <f32>
            %106 = ascvec.reduce_sum %105, %mask_86 : <f32>
            %107 = ascvec.add %103, %106, %mask : <f32>
            %mask_88, %remaining_out_89 = ascvec.update_mask %c0_i32_55 : !ascvec.mask
            %108 = ascvec.loadalign %arg27[%c896] : memref<2048xf32, 1>, index -> <f32>
            %109 = ascvec.mul %108, %108, %mask_88 : <f32>
            %110 = ascvec.reduce_sum %109, %mask_88 : <f32>
            %111 = ascvec.add %107, %110, %mask : <f32>
            %mask_90, %remaining_out_91 = ascvec.update_mask %c0_i32_55 : !ascvec.mask
            %112 = ascvec.loadalign %arg27[%c960] : memref<2048xf32, 1>, index -> <f32>
            %113 = ascvec.mul %112, %112, %mask_90 : <f32>
            %114 = ascvec.reduce_sum %113, %mask_90 : <f32>
            %115 = ascvec.add %111, %114, %mask : <f32>
            %mask_92, %remaining_out_93 = ascvec.update_mask %c0_i32_55 : !ascvec.mask
            %116 = ascvec.loadalign %arg27[%c1024] : memref<2048xf32, 1>, index -> <f32>
            %117 = ascvec.mul %116, %116, %mask_92 : <f32>
            %118 = ascvec.reduce_sum %117, %mask_92 : <f32>
            %119 = ascvec.add %115, %118, %mask : <f32>
            %mask_94, %remaining_out_95 = ascvec.update_mask %c0_i32_55 : !ascvec.mask
            %120 = ascvec.loadalign %arg27[%c1088] : memref<2048xf32, 1>, index -> <f32>
            %121 = ascvec.mul %120, %120, %mask_94 : <f32>
            %122 = ascvec.reduce_sum %121, %mask_94 : <f32>
            %123 = ascvec.add %119, %122, %mask : <f32>
            %mask_96, %remaining_out_97 = ascvec.update_mask %c0_i32_55 : !ascvec.mask
            %124 = ascvec.loadalign %arg27[%c1152] : memref<2048xf32, 1>, index -> <f32>
            %125 = ascvec.mul %124, %124, %mask_96 : <f32>
            %126 = ascvec.reduce_sum %125, %mask_96 : <f32>
            %127 = ascvec.add %123, %126, %mask : <f32>
            %mask_98, %remaining_out_99 = ascvec.update_mask %c0_i32_55 : !ascvec.mask
            %128 = ascvec.loadalign %arg27[%c1216] : memref<2048xf32, 1>, index -> <f32>
            %129 = ascvec.mul %128, %128, %mask_98 : <f32>
            %130 = ascvec.reduce_sum %129, %mask_98 : <f32>
            %131 = ascvec.add %127, %130, %mask : <f32>
            %mask_100, %remaining_out_101 = ascvec.update_mask %c0_i32_55 : !ascvec.mask
            %132 = ascvec.loadalign %arg27[%c1280] : memref<2048xf32, 1>, index -> <f32>
            %133 = ascvec.mul %132, %132, %mask_100 : <f32>
            %134 = ascvec.reduce_sum %133, %mask_100 : <f32>
            %135 = ascvec.add %131, %134, %mask : <f32>
            %mask_102, %remaining_out_103 = ascvec.update_mask %c0_i32_55 : !ascvec.mask
            %136 = ascvec.loadalign %arg27[%c1344] : memref<2048xf32, 1>, index -> <f32>
            %137 = ascvec.mul %136, %136, %mask_102 : <f32>
            %138 = ascvec.reduce_sum %137, %mask_102 : <f32>
            %139 = ascvec.add %135, %138, %mask : <f32>
            %mask_104, %remaining_out_105 = ascvec.update_mask %c0_i32_55 : !ascvec.mask
            %140 = ascvec.loadalign %arg27[%c1408] : memref<2048xf32, 1>, index -> <f32>
            %141 = ascvec.mul %140, %140, %mask_104 : <f32>
            %142 = ascvec.reduce_sum %141, %mask_104 : <f32>
            %143 = ascvec.add %139, %142, %mask : <f32>
            %mask_106, %remaining_out_107 = ascvec.update_mask %c0_i32_55 : !ascvec.mask
            %144 = ascvec.loadalign %arg27[%c1472] : memref<2048xf32, 1>, index -> <f32>
            %145 = ascvec.mul %144, %144, %mask_106 : <f32>
            %146 = ascvec.reduce_sum %145, %mask_106 : <f32>
            %147 = ascvec.add %143, %146, %mask : <f32>
            %mask_108, %remaining_out_109 = ascvec.update_mask %c0_i32_55 : !ascvec.mask
            %148 = ascvec.loadalign %arg27[%c1536] : memref<2048xf32, 1>, index -> <f32>
            %149 = ascvec.mul %148, %148, %mask_108 : <f32>
            %150 = ascvec.reduce_sum %149, %mask_108 : <f32>
            %151 = ascvec.add %147, %150, %mask : <f32>
            %mask_110, %remaining_out_111 = ascvec.update_mask %c0_i32_55 : !ascvec.mask
            %152 = ascvec.loadalign %arg27[%c1600] : memref<2048xf32, 1>, index -> <f32>
            %153 = ascvec.mul %152, %152, %mask_110 : <f32>
            %154 = ascvec.reduce_sum %153, %mask_110 : <f32>
            %155 = ascvec.add %151, %154, %mask : <f32>
            %mask_112, %remaining_out_113 = ascvec.update_mask %c0_i32_55 : !ascvec.mask
            %156 = ascvec.loadalign %arg27[%c1664] : memref<2048xf32, 1>, index -> <f32>
            %157 = ascvec.mul %156, %156, %mask_112 : <f32>
            %158 = ascvec.reduce_sum %157, %mask_112 : <f32>
            %159 = ascvec.add %155, %158, %mask : <f32>
            %mask_114, %remaining_out_115 = ascvec.update_mask %c0_i32_55 : !ascvec.mask
            %160 = ascvec.loadalign %arg27[%c1728] : memref<2048xf32, 1>, index -> <f32>
            %161 = ascvec.mul %160, %160, %mask_114 : <f32>
            %162 = ascvec.reduce_sum %161, %mask_114 : <f32>
            %163 = ascvec.add %159, %162, %mask : <f32>
            %mask_116, %remaining_out_117 = ascvec.update_mask %c0_i32_55 : !ascvec.mask
            %164 = ascvec.loadalign %arg27[%c1792] : memref<2048xf32, 1>, index -> <f32>
            %165 = ascvec.mul %164, %164, %mask_116 : <f32>
            %166 = ascvec.reduce_sum %165, %mask_116 : <f32>
            %167 = ascvec.add %163, %166, %mask : <f32>
            %mask_118, %remaining_out_119 = ascvec.update_mask %c0_i32_55 : !ascvec.mask
            %168 = ascvec.loadalign %arg27[%c1856] : memref<2048xf32, 1>, index -> <f32>
            %169 = ascvec.mul %168, %168, %mask_118 : <f32>
            %170 = ascvec.reduce_sum %169, %mask_118 : <f32>
            %171 = ascvec.add %167, %170, %mask : <f32>
            %mask_120, %remaining_out_121 = ascvec.update_mask %c0_i32_55 : !ascvec.mask
            %172 = ascvec.loadalign %arg27[%c1920] : memref<2048xf32, 1>, index -> <f32>
            %173 = ascvec.mul %172, %172, %mask_120 : <f32>
            %174 = ascvec.reduce_sum %173, %mask_120 : <f32>
            %175 = ascvec.add %171, %174, %mask : <f32>
            %mask_122, %remaining_out_123 = ascvec.update_mask %c0_i32_55 : !ascvec.mask
            %176 = ascvec.loadalign %arg27[%c1984] : memref<2048xf32, 1>, index -> <f32>
            %177 = ascvec.mul %176, %176, %mask_122 : <f32>
            %178 = ascvec.reduce_sum %177, %mask_122 : <f32>
            %179 = ascvec.add %175, %178, %mask : <f32>
            %180 = ascvec.mul_scalar %179, %cst, %mask : <f32>, f32
            %181 = ascvec.add_scalar %180, %arg28, %mask : <f32>, f32
            %182 = ascvec.sqrt %181, %mask : <f32>
            ascvec.storealign_1st %arg29[%c0_56], %182 : memref<8xf32, 1>, index, <f32>
          }
          ascvec.mem_bar vst_vld
        } {cannir.vf_mode = "simd"}
        %40 = ascscalar.memref_address %alloc_21 : memref<2048xbf16, 1>
        %41 = ascscalar.memref_address %alloc_22 : memref<2048xbf16, 1>
        %42 = arith.select %20, %40, %41 : i64
        %43 = ascscalar.bind_local_ptr %42 : memref<2048xbf16, 1>
        %44 = arith.select %20, %c3_i64, %c4_i64 : i64
        ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
        ascvec.get_buf v, %44, 0 {cannir.buffer.generated_sync}
        ascvec.vf(%alloc_23, %39, %alloc_20, %arg21, %43 : memref<8xf32, 1>, memref<2048xf32, 1>, memref<2048xf32, 1>, f32, memref<2048xbf16, 1>) {
        ^bb0(%arg27: memref<8xf32, 1>, %arg28: memref<2048xf32, 1>, %arg29: memref<2048xf32, 1>, %arg30: f32, %arg31: memref<2048xbf16, 1>):
          %c64_i32 = arith.constant 64 : i32
          %c640_i32 = arith.constant 640 : i32
          %c0_i32_55 = arith.constant 0 : i32
          %c0_56 = arith.constant 0 : index
          %51 = ascvec.loadalign_brc %arg27[%c0_56] : memref<8xf32, 1>, index -> <f32>
          scf.for %arg32 = %c0_i32_55 to %c640_i32 step %c64_i32  : i32 {
            %52 = arith.index_cast %arg32 : i32 to index
            %53 = ascvec.loadalign %arg28[%52] : memref<2048xf32, 1>, index -> <f32>
            %54 = ascvec.create_mask : !ascvec.mask
            %55 = ascvec.div %53, %51, %54 {precision_mode = true} : <f32>
            %56 = ascvec.loadalign %arg29[%52] : memref<2048xf32, 1>, index -> <f32>
            %57 = ascvec.mul %55, %56, %54 : <f32>
            %58 = ascvec.mul_scalar %57, %arg30, %54 : <f32>, f32
            %59 = ascvec.cast %58, %54 : <f32> -> <bf16>
            ascvec.storealign_pack %arg31[%52], %59, %54 {mode = "b32_to_b16"} : memref<2048xbf16, 1>, index, <bf16>
          }
          ascvec.mem_bar vst_vld
        } {cannir.vf_mode = "simd"}
        ascvec.release_buf v, %44, 0 {cannir.buffer.generated_sync}
        ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
        %45 = arith.muli %arg25, %c10240_i64 : i64
        %46 = ascscalar.memref_address %alloc_14 : memref<5120xbf16>
        %47 = arith.divui %45, %c8_i64 : i64
        %48 = arith.addi %46, %47 : i64
        %49 = ascscalar.bind_gm_ptr %48 : memref<1xbf16>
        %50 = ascscalar.bind_local_ptr %42 : memref<1xbf16, 1>
        ascvec.get_buf mte3, %44, 0 {cannir.buffer.generated_sync}
        ascvec.copy_ub2gm %49[%c0], %50[%c0], %c1_i32, %c1280_i32, %c0_i64, %c1280_i32 l2_cache_ctl %c0_i32 : memref<1xbf16>, memref<1xbf16, 1>
        ascvec.release_buf mte3, %44, 0 {cannir.buffer.generated_sync}
        scf.yield %27 : i64
      }
      scf.yield %19 : i64
    } else {
      scf.yield %c0_i64 : i64
    }
    %11 = arith.addi %5, %7 : i64
    %12 = arith.muli %1, %c2_i64 : i64
    %13 = scf.for %arg25 = %11 to %c8_i64 step %12 iter_args(%arg26 = %10) -> (i64)  : i64 {
      %17 = arith.cmpi eq, %arg26, %c0_i64 : i64
      %18 = ascscalar.memref_address %alloc_21 : memref<2048xbf16, 1>
      %19 = ascscalar.memref_address %alloc_22 : memref<2048xbf16, 1>
      %20 = arith.select %17, %18, %19 : i64
      %21 = ascscalar.bind_local_ptr %20 : memref<2048xbf16, 1>
      %22 = arith.select %17, %c3_i64, %c4_i64 : i64
      %23 = arith.addi %arg26, %c1_i64 overflow<nsw> : i64
      %24 = arith.remui %23, %c2_i64 : i64
      ascvec.get_buf v, %22, 0 {cannir.buffer.generated_sync}
      ascvec.vf(%21 : memref<2048xbf16, 1>) {
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
        %c0_55 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : bf16
        %c128_i32 = arith.constant 128 : i32
        %c0_56 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        scf.for %arg28 = %c0_56 to %c1 step %c1 {
          %mask, %remaining_out = ascvec.update_mask %c128_i32 elem_bits 16 : !ascvec.mask
          %31 = ascvec.duplicate_scalar %cst, %mask : bf16 -> <bf16>
          ascvec.storealign %arg27[%c0_55], %31, %mask : memref<2048xbf16, 1>, index, <bf16>
          ascvec.storealign %arg27[%c128], %31, %mask : memref<2048xbf16, 1>, index, <bf16>
          ascvec.storealign %arg27[%c256], %31, %mask : memref<2048xbf16, 1>, index, <bf16>
          ascvec.storealign %arg27[%c384], %31, %mask : memref<2048xbf16, 1>, index, <bf16>
          ascvec.storealign %arg27[%c512], %31, %mask : memref<2048xbf16, 1>, index, <bf16>
          ascvec.storealign %arg27[%c640], %31, %mask : memref<2048xbf16, 1>, index, <bf16>
          ascvec.storealign %arg27[%c768], %31, %mask : memref<2048xbf16, 1>, index, <bf16>
          ascvec.storealign %arg27[%c896], %31, %mask : memref<2048xbf16, 1>, index, <bf16>
          ascvec.storealign %arg27[%c1024], %31, %mask : memref<2048xbf16, 1>, index, <bf16>
          ascvec.storealign %arg27[%c1152], %31, %mask : memref<2048xbf16, 1>, index, <bf16>
          ascvec.storealign %arg27[%c1280], %31, %mask : memref<2048xbf16, 1>, index, <bf16>
          ascvec.storealign %arg27[%c1408], %31, %mask : memref<2048xbf16, 1>, index, <bf16>
          ascvec.storealign %arg27[%c1536], %31, %mask : memref<2048xbf16, 1>, index, <bf16>
          ascvec.storealign %arg27[%c1664], %31, %mask : memref<2048xbf16, 1>, index, <bf16>
          ascvec.storealign %arg27[%c1792], %31, %mask : memref<2048xbf16, 1>, index, <bf16>
          ascvec.storealign %arg27[%c1920], %31, %mask : memref<2048xbf16, 1>, index, <bf16>
        }
        ascvec.mem_bar vst_vld
      } {cannir.vf_mode = "simd"}
      ascvec.release_buf v, %22, 0 {cannir.buffer.generated_sync}
      %25 = arith.muli %arg25, %c10240_i64 : i64
      %26 = ascscalar.memref_address %alloc_14 : memref<5120xbf16>
      %27 = arith.divui %25, %c8_i64 : i64
      %28 = arith.addi %26, %27 : i64
      %29 = ascscalar.bind_gm_ptr %28 : memref<1xbf16>
      %30 = ascscalar.bind_local_ptr %20 : memref<1xbf16, 1>
      ascvec.get_buf mte3, %22, 0 {cannir.buffer.generated_sync}
      ascvec.copy_ub2gm %29[%c0], %30[%c0], %c1_i32, %c1280_i32, %c0_i64, %c1280_i32 l2_cache_ctl %c0_i32 : memref<1xbf16>, memref<1xbf16, 1>
      ascvec.release_buf mte3, %22, 0 {cannir.buffer.generated_sync}
      scf.yield %24 : i64
    }
    ascvec.sync_all
    ascvec.sync_block_arrive mte3, %c13_i64
    ascvec.sync_block_arrive mte3, %c14_i64 {mode = 0 : i32}
    ascvec.sync_block_wait s, %c14_i64 {mode = 0 : i32}
    ascvec.sync_block_wait s, %c15_i64
    %14 = arith.cmpi eq, %2, %c0_i64 : i64
    scf.if %14 {
      %17 = ascscalar.bitcast_memref %alloc_28 byte_offset = %c0_i64 : memref<512xf32, 1> -> memref<?xf32, 1>
      "ascvec.copy_gm2ub_plan"(%17, %alloc_1, %c0_i32, %c0_i32, %c0_i64, %c0_i64, %c1_i64, %c2048_i64, %c0_i64, %c2048_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 2048 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 6912 : i64, ub_capacity = 2048 : i64}> : (memref<?xf32, 1>, memref<512xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
      ascvec.get_buf mte2, %c5_i64, 0 {cannir.buffer.generated_sync}
      "ascvec.copy_gm2ub_plan"(%alloc_32, %alloc_4, %c0_i32, %c0_i32, %c0_i64, %c0_i64, %c1_i64, %c4_i64, %c0_i64, %c4_i64, %c0_i64, %c0_i64, %c1_i64, %c4_i64, %c32_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 4 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 15168 : i64, ub_capacity = 32 : i64}> : (memref<8xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
      ascvec.release_buf mte2, %c5_i64, 0 {cannir.buffer.generated_sync}
      %18 = scf.for %arg25 = %0 to %5 step %1 iter_args(%arg26 = %c0_i64) -> (i64)  : i64 {
        %19 = arith.cmpi eq, %arg26, %c0_i64 : i64
        %20 = ascscalar.memref_address %alloc_24 : memref<576xbf16, 1>
        %21 = ascscalar.memref_address %alloc_25 : memref<576xbf16, 1>
        %22 = arith.select %19, %20, %21 : i64
        %23 = arith.cmpi ne, %arg26, %c0_i64 : i64
        %24 = arith.extui %23 : i1 to i64
        %25 = arith.addi %arg26, %c1_i64 overflow<nsw> : i64
        %26 = arith.remui %25, %c2_i64 : i64
        %27 = arith.minsi %arg25, %c8_i64 : i64
        %28 = arith.subi %c8_i64, %27 : i64
        %29 = arith.minsi %28, %c1_i64 : i64
        %30 = arith.cmpi sle, %arg25, %c8_i64 : i64
        %31 = arith.select %30, %29, %c0_i64 : i64
        %32 = arith.muli %arg25, %c1152_i64 : i64
        %33 = arith.muli %31, %c1152_i64 : i64
        %34 = arith.cmpi eq, %33, %c0_i64 : i64
        %35 = arith.cmpi ne, %33, %c0_i64 : i64
        %36 = arith.extui %35 : i1 to i64
        %37 = arith.select %34, %c2_i64, %33 : i64
        %38 = ascscalar.bind_local_ptr %22 : memref<1xbf16, 1>
        ascvec.get_buf mte2, %24, 0 {cannir.buffer.generated_sync}
        "ascvec.copy_gm2ub_plan"(%38, %alloc_11, %c0_i32, %c0_i32, %c0_i64, %32, %36, %37, %c0_i64, %37, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 9216 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<4608xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
        ascvec.release_buf mte2, %24, 0 {cannir.buffer.generated_sync}
        %39 = ascscalar.bind_local_ptr %22 : memref<576xbf16, 1>
        %40 = ascscalar.memref_address %alloc_26 : memref<576xf32, 1>
        %41 = ascscalar.memref_address %alloc_27 : memref<576xf32, 1>
        %42 = arith.select %19, %40, %41 : i64
        %43 = ascscalar.bind_local_ptr %42 : memref<576xf32, 1>
        ascvec.get_buf v, %24, 0 {cannir.buffer.generated_sync}
        ascvec.vf(%39, %43 : memref<576xbf16, 1>, memref<576xf32, 1>) {
        ^bb0(%arg27: memref<576xbf16, 1>, %arg28: memref<576xf32, 1>):
          %c64_i32 = arith.constant 64 : i32
          %c576_i32_55 = arith.constant 576 : i32
          %c0_i32_56 = arith.constant 0 : i32
          scf.for %arg29 = %c0_i32_56 to %c576_i32_55 step %c64_i32  : i32 {
            %80 = arith.index_cast %arg29 : i32 to index
            %81 = ascvec.loadalign_unpack %arg27[%80] {mode = "b16_to_b32"} : memref<576xbf16, 1>, index -> <bf16>
            %82 = ascvec.create_mask : !ascvec.mask
            %83 = ascvec.cast %81, %82 {rounding = 6 : i32} : <bf16> -> <f32>
            ascvec.storealign %arg28[%80], %83, %82 : memref<576xf32, 1>, index, <f32>
          }
          ascvec.mem_bar vst_vld
        } {cannir.vf_mode = "simd"}
        ascvec.release_buf v, %24, 0 {cannir.buffer.generated_sync}
        ascvec.vf(%43, %arg20, %alloc_31 : memref<576xf32, 1>, f32, memref<64xf32, 1>) {
        ^bb0(%arg27: memref<576xf32, 1>, %arg28: f32, %arg29: memref<64xf32, 1>):
          %c448 = arith.constant 448 : index
          %c384 = arith.constant 384 : index
          %c320 = arith.constant 320 : index
          %c256 = arith.constant 256 : index
          %c192 = arith.constant 192 : index
          %c128 = arith.constant 128 : index
          %c64 = arith.constant 64 : index
          %cst = arith.constant 0.001953125 : f32
          %c0_55 = arith.constant 0 : index
          %cst_56 = arith.constant 0.000000e+00 : f32
          %c1_i32_57 = arith.constant 1 : i32
          %c0_58 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          scf.for %arg30 = %c0_58 to %c1 step %c1 {
            %mask, %remaining_out = ascvec.update_mask %c1_i32_57 : !ascvec.mask
            %80 = ascvec.duplicate_scalar %cst_56, %mask : f32 -> <f32>
            %81 = ascvec.loadalign %arg27[%c0_55] : memref<576xf32, 1>, index -> <f32>
            %82 = ascvec.create_mask : !ascvec.mask
            %83 = ascvec.mul %81, %81, %82 : <f32>
            %84 = ascvec.reduce_sum %83, %82 : <f32>
            %85 = ascvec.add %80, %84, %mask : <f32>
            %86 = ascvec.loadalign %arg27[%c64] : memref<576xf32, 1>, index -> <f32>
            %87 = ascvec.mul %86, %86, %82 : <f32>
            %88 = ascvec.reduce_sum %87, %82 : <f32>
            %89 = ascvec.add %85, %88, %mask : <f32>
            %90 = ascvec.loadalign %arg27[%c128] : memref<576xf32, 1>, index -> <f32>
            %91 = ascvec.mul %90, %90, %82 : <f32>
            %92 = ascvec.reduce_sum %91, %82 : <f32>
            %93 = ascvec.add %89, %92, %mask : <f32>
            %94 = ascvec.loadalign %arg27[%c192] : memref<576xf32, 1>, index -> <f32>
            %95 = ascvec.mul %94, %94, %82 : <f32>
            %96 = ascvec.reduce_sum %95, %82 : <f32>
            %97 = ascvec.add %93, %96, %mask : <f32>
            %98 = ascvec.loadalign %arg27[%c256] : memref<576xf32, 1>, index -> <f32>
            %99 = ascvec.mul %98, %98, %82 : <f32>
            %100 = ascvec.reduce_sum %99, %82 : <f32>
            %101 = ascvec.add %97, %100, %mask : <f32>
            %102 = ascvec.loadalign %arg27[%c320] : memref<576xf32, 1>, index -> <f32>
            %103 = ascvec.mul %102, %102, %82 : <f32>
            %104 = ascvec.reduce_sum %103, %82 : <f32>
            %105 = ascvec.add %101, %104, %mask : <f32>
            %106 = ascvec.loadalign %arg27[%c384] : memref<576xf32, 1>, index -> <f32>
            %107 = ascvec.mul %106, %106, %82 : <f32>
            %108 = ascvec.reduce_sum %107, %82 : <f32>
            %109 = ascvec.add %105, %108, %mask : <f32>
            %110 = ascvec.loadalign %arg27[%c448] : memref<576xf32, 1>, index -> <f32>
            %111 = ascvec.mul %110, %110, %82 : <f32>
            %112 = ascvec.reduce_sum %111, %82 : <f32>
            %113 = ascvec.add %109, %112, %mask : <f32>
            %114 = ascvec.mul_scalar %113, %cst, %mask : <f32>, f32
            %115 = ascvec.add_scalar %114, %arg28, %mask : <f32>, f32
            %116 = ascvec.sqrt %115, %mask : <f32>
            ascvec.storealign_1st %arg29[%c0_55], %116 : memref<64xf32, 1>, index, <f32>
          }
          ascvec.mem_bar vst_vld
        } {cannir.vf_mode = "simd"}
        %44 = cannir.extract_layout %arg10 : !cannir.tensor<<gm, bf16>, <<(-1, 64)>, <(64, 1)>>, <<(-1, 64)>, <(64, 1)>>, identity>
        %45 = cannir.extract_shape %44 : <<(-1, 64)>, <(64, 1)>> -> <(-1, 64)>
        %46 = cannir.extract_dim %45[0] : !cannir.shape<(-1, 64)>
        %47 = arith.minsi %arg25, %46 : i64
        %48 = arith.subi %46, %47 : i64
        %49 = arith.minsi %48, %c1_i64 : i64
        %50 = arith.cmpi sle, %arg25, %46 : i64
        %51 = arith.select %50, %49, %c0_i64 : i64
        %52 = arith.muli %arg25, %c128_i64 : i64
        %53 = arith.muli %51, %c128_i64 : i64
        %54 = arith.cmpi eq, %53, %c0_i64 : i64
        %55 = arith.cmpi ne, %53, %c0_i64 : i64
        %56 = arith.extui %55 : i1 to i64
        %57 = arith.select %54, %c2_i64, %53 : i64
        %58 = ascscalar.bitcast_memref %alloc_33 byte_offset = %c0_i64 : memref<64xbf16, 1> -> memref<?xbf16, 1>
        ascvec.get_buf mte2, %c6_i64, 0 {cannir.buffer.generated_sync}
        "ascvec.copy_gm2ub_plan"(%58, %alloc_5, %c0_i32, %c0_i32, %c0_i64, %52, %56, %57, %c0_i64, %57, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = 15200 : i64, ub_capacity = 128 : i64}> : (memref<?xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
        ascvec.release_buf mte2, %c6_i64, 0 {cannir.buffer.generated_sync}
        ascvec.get_buf v, %c6_i64, 0 {cannir.buffer.generated_sync}
        ascvec.vf(%alloc_33, %alloc_35 : memref<64xbf16, 1>, memref<128xf32, 1>) {
        ^bb0(%arg27: memref<64xbf16, 1>, %arg28: memref<128xf32, 1>):
          %c0_55 = arith.constant 0 : index
          %c0_56 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          scf.for %arg29 = %c0_56 to %c1 step %c1 {
            %80 = ascvec.loadalign_unpack %arg27[%c0_55] {mode = "b16_to_b32"} : memref<64xbf16, 1>, index -> <bf16>
            %81 = ascvec.create_mask : !ascvec.mask
            %82 = ascvec.cast %80, %81 {rounding = 6 : i32} : <bf16> -> <f32>
            ascvec.storealign %arg28[%c0_55], %82, %81 : memref<128xf32, 1>, index, <f32>
          }
          ascvec.mem_bar vst_vld
        } {cannir.vf_mode = "simd"}
        ascvec.release_buf v, %c6_i64, 0 {cannir.buffer.generated_sync}
        %59 = cannir.extract_layout %arg11 : !cannir.tensor<<gm, bf16>, <<(-1, 64)>, <(64, 1)>>, <<(-1, 64)>, <(64, 1)>>, identity>
        %60 = cannir.extract_shape %59 : <<(-1, 64)>, <(64, 1)>> -> <(-1, 64)>
        %61 = cannir.extract_dim %60[0] : !cannir.shape<(-1, 64)>
        %62 = arith.minsi %arg25, %61 : i64
        %63 = arith.subi %61, %62 : i64
        %64 = arith.minsi %63, %c1_i64 : i64
        %65 = arith.cmpi sle, %arg25, %61 : i64
        %66 = arith.select %65, %64, %c0_i64 : i64
        %67 = arith.muli %66, %c128_i64 : i64
        %68 = arith.cmpi eq, %67, %c0_i64 : i64
        %69 = arith.cmpi ne, %67, %c0_i64 : i64
        %70 = arith.extui %69 : i1 to i64
        %71 = arith.select %68, %c2_i64, %67 : i64
        %72 = ascscalar.bitcast_memref %alloc_34 byte_offset = %c0_i64 : memref<64xbf16, 1> -> memref<?xbf16, 1>
        ascvec.get_buf mte2, %c7_i64, 0 {cannir.buffer.generated_sync}
        "ascvec.copy_gm2ub_plan"(%72, %alloc_6, %c0_i32, %c0_i32, %c0_i64, %52, %70, %71, %c0_i64, %71, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = 15328 : i64, ub_capacity = 128 : i64}> : (memref<?xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
        ascvec.release_buf mte2, %c7_i64, 0 {cannir.buffer.generated_sync}
        ascvec.get_buf v, %c7_i64, 0 {cannir.buffer.generated_sync}
        ascvec.vf(%alloc_34, %alloc_36 : memref<64xbf16, 1>, memref<128xf32, 1>) {
        ^bb0(%arg27: memref<64xbf16, 1>, %arg28: memref<128xf32, 1>):
          %c0_55 = arith.constant 0 : index
          %c0_56 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          scf.for %arg29 = %c0_56 to %c1 step %c1 {
            %80 = ascvec.loadalign_unpack %arg27[%c0_55] {mode = "b16_to_b32"} : memref<64xbf16, 1>, index -> <bf16>
            %81 = ascvec.create_mask : !ascvec.mask
            %82 = ascvec.cast %80, %81 {rounding = 6 : i32} : <bf16> -> <f32>
            ascvec.storealign %arg28[%c0_55], %82, %81 : memref<128xf32, 1>, index, <f32>
          }
          ascvec.mem_bar vst_vld
        } {cannir.vf_mode = "simd"}
        ascvec.release_buf v, %c7_i64, 0 {cannir.buffer.generated_sync}
        ascvec.vf(%43, %alloc_37 : memref<576xf32, 1>, memref<128xf32, 1>) {
        ^bb0(%arg27: memref<576xf32, 1>, %arg28: memref<128xf32, 1>):
          %c0_55 = arith.constant 0 : index
          %c512 = arith.constant 512 : index
          %c0_56 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          scf.for %arg29 = %c0_56 to %c1 step %c1 {
            %80 = ascvec.loadalign %arg27[%c512] : memref<576xf32, 1>, index -> <f32>
            %81 = ascvec.create_mask : !ascvec.mask
            ascvec.storealign %arg28[%c0_55], %80, %81 : memref<128xf32, 1>, index, <f32>
          }
          ascvec.mem_bar vst_vld
        } {cannir.vf_mode = "simd"}
        %73 = arith.cmpi eq, %arg24, %c1_i64 : i64
        scf.if %73 {
          ascvec.vf(%alloc_36, %alloc_35, %alloc_37, %alloc_38 : memref<128xf32, 1>, memref<128xf32, 1>, memref<128xf32, 1>, memref<64xf32, 1>) {
          ^bb0(%arg27: memref<128xf32, 1>, %arg28: memref<128xf32, 1>, %arg29: memref<128xf32, 1>, %arg30: memref<64xf32, 1>):
            %c32 = arith.constant 32 : index
            %c0_55 = arith.constant 0 : index
            %c32_i32_56 = arith.constant 32 : i32
            %c0_57 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            scf.for %arg31 = %c0_57 to %c1 step %c1 {
              %mask, %remaining_out = ascvec.update_mask %c32_i32_56 : !ascvec.mask
              %80 = ascvec.loadalign %arg27[%c0_55] : memref<128xf32, 1>, index -> <f32>
              %81 = ascvec.loadalign %arg27[%c32] : memref<128xf32, 1>, index -> <f32>
              %82 = ascvec.loadalign %arg28[%c0_55] : memref<128xf32, 1>, index -> <f32>
              %83 = ascvec.loadalign %arg28[%c32] : memref<128xf32, 1>, index -> <f32>
              %result0, %result1 = ascvec.loadalign_deintlv %arg29[%c0_55] {dintlv_width = "b32"} : memref<128xf32, 1>, index -> <f32>, <f32>
              %84 = ascvec.mul %result0, %80, %mask : <f32>
              %85 = ascvec.mul %result1, %82, %mask : <f32>
              %86 = ascvec.sub %84, %85, %mask : <f32>
              %87 = ascvec.mul %result1, %81, %mask : <f32>
              %88 = ascvec.mul %result0, %83, %mask : <f32>
              %89 = ascvec.add %87, %88, %mask : <f32>
              ascvec.storealign %arg30[%c0_55], %86, %mask : memref<64xf32, 1>, index, <f32>
              ascvec.storealign %arg30[%c32], %89, %mask : memref<64xf32, 1>, index, <f32>
            }
            ascvec.mem_bar vst_vld
          } {cannir.vf_mode = "simd"}
        } else {
          ascvec.vf(%alloc_36, %alloc_35, %alloc_37, %alloc_38 : memref<128xf32, 1>, memref<128xf32, 1>, memref<128xf32, 1>, memref<64xf32, 1>) {
          ^bb0(%arg27: memref<128xf32, 1>, %arg28: memref<128xf32, 1>, %arg29: memref<128xf32, 1>, %arg30: memref<64xf32, 1>):
            %c32 = arith.constant 32 : index
            %c0_55 = arith.constant 0 : index
            %c32_i32_56 = arith.constant 32 : i32
            %c0_57 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            scf.for %arg31 = %c0_57 to %c1 step %c1 {
              %mask, %remaining_out = ascvec.update_mask %c32_i32_56 : !ascvec.mask
              %80 = ascvec.loadalign %arg27[%c0_55] : memref<128xf32, 1>, index -> <f32>
              %81 = ascvec.loadalign %arg27[%c32] : memref<128xf32, 1>, index -> <f32>
              %82 = ascvec.loadalign %arg28[%c0_55] : memref<128xf32, 1>, index -> <f32>
              %83 = ascvec.loadalign %arg28[%c32] : memref<128xf32, 1>, index -> <f32>
              %84 = ascvec.loadalign %arg29[%c0_55] : memref<128xf32, 1>, index -> <f32>
              %85 = ascvec.loadalign %arg29[%c32] : memref<128xf32, 1>, index -> <f32>
              %86 = ascvec.mul %84, %80, %mask : <f32>
              %87 = ascvec.mul %85, %82, %mask : <f32>
              %88 = ascvec.sub %86, %87, %mask : <f32>
              %89 = ascvec.mul %85, %81, %mask : <f32>
              %90 = ascvec.mul %84, %83, %mask : <f32>
              %91 = ascvec.add %89, %90, %mask : <f32>
              ascvec.storealign %arg30[%c0_55], %88, %mask : memref<64xf32, 1>, index, <f32>
              ascvec.storealign %arg30[%c32], %91, %mask : memref<64xf32, 1>, index, <f32>
            }
            ascvec.mem_bar vst_vld
          } {cannir.vf_mode = "simd"}
        }
        %74 = ascscalar.memref_address %alloc_29 : memref<576xf8E4M3FN, 1>
        %75 = ascscalar.memref_address %alloc_30 : memref<576xf8E4M3FN, 1>
        %76 = arith.select %19, %74, %75 : i64
        %77 = ascscalar.bind_local_ptr %76 : memref<576xf8E4M3FN, 1>
        %78 = arith.select %19, %c3_i64, %c4_i64 : i64
        ascvec.get_buf v, %c5_i64, 0 {cannir.buffer.generated_sync}
        ascvec.get_buf v, %78, 0 {cannir.buffer.generated_sync}
        ascvec.vf(%alloc_31, %alloc_32, %43, %alloc_28, %arg22, %77, %alloc_38 : memref<64xf32, 1>, memref<8xf32, 1>, memref<576xf32, 1>, memref<512xf32, 1>, f32, memref<576xf8E4M3FN, 1>, memref<64xf32, 1>) {
        ^bb0(%arg27: memref<64xf32, 1>, %arg28: memref<8xf32, 1>, %arg29: memref<576xf32, 1>, %arg30: memref<512xf32, 1>, %arg31: f32, %arg32: memref<576xf8E4M3FN, 1>, %arg33: memref<64xf32, 1>):
          %c512 = arith.constant 512 : index
          %c448 = arith.constant 448 : index
          %c384 = arith.constant 384 : index
          %c320 = arith.constant 320 : index
          %c256 = arith.constant 256 : index
          %c192 = arith.constant 192 : index
          %c128 = arith.constant 128 : index
          %c64 = arith.constant 64 : index
          %c64_i32 = arith.constant 64 : i32
          %cst = arith.constant 4.480000e+02 : f32
          %cst_55 = arith.constant -4.480000e+02 : f32
          %c0_56 = arith.constant 0 : index
          %c0_57 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          scf.for %arg34 = %c0_57 to %c1 step %c1 {
            %80 = ascvec.loadalign_brc %arg27[%c0_56] : memref<64xf32, 1>, index -> <f32>
            %81 = ascvec.loadalign_brc %arg28[%c0_56] : memref<8xf32, 1>, index -> <f32>
            %82 = ascvec.loadalign %arg29[%c0_56] : memref<576xf32, 1>, index -> <f32>
            %83 = ascvec.create_mask : !ascvec.mask
            %84 = ascvec.div %82, %80, %83 {precision_mode = true} : <f32>
            %85 = ascvec.loadalign %arg30[%c0_56] : memref<512xf32, 1>, index -> <f32>
            %86 = ascvec.mul %84, %85, %83 : <f32>
            %87 = ascvec.mul_scalar %86, %arg31, %83 : <f32>, f32
            %88 = ascvec.mul %87, %81, %83 : <f32>
            %89 = ascvec.max_scalar %88, %cst_55, %83 : <f32>, f32
            %90 = ascvec.min_scalar %89, %cst, %83 : <f32>, f32
            %91 = ascvec.cast %90, %83 {reg_layout = 0 : i32} : <f32> -> <f8E4M3FN>
            ascvec.storealign_pack %arg32[%c0_56], %91, %83 {mode = "b32_to_b8"} : memref<576xf8E4M3FN, 1>, index, <f8E4M3FN>
            %92 = ascvec.loadalign %arg29[%c64] : memref<576xf32, 1>, index -> <f32>
            %93 = ascvec.div %92, %80, %83 {precision_mode = true} : <f32>
            %94 = ascvec.loadalign %arg30[%c64] : memref<512xf32, 1>, index -> <f32>
            %95 = ascvec.mul %93, %94, %83 : <f32>
            %96 = ascvec.mul_scalar %95, %arg31, %83 : <f32>, f32
            %97 = ascvec.mul %96, %81, %83 : <f32>
            %98 = ascvec.max_scalar %97, %cst_55, %83 : <f32>, f32
            %99 = ascvec.min_scalar %98, %cst, %83 : <f32>, f32
            %100 = ascvec.cast %99, %83 {reg_layout = 0 : i32} : <f32> -> <f8E4M3FN>
            ascvec.storealign_pack %arg32[%c64], %100, %83 {mode = "b32_to_b8"} : memref<576xf8E4M3FN, 1>, index, <f8E4M3FN>
            %101 = ascvec.loadalign %arg29[%c128] : memref<576xf32, 1>, index -> <f32>
            %102 = ascvec.div %101, %80, %83 {precision_mode = true} : <f32>
            %103 = ascvec.loadalign %arg30[%c128] : memref<512xf32, 1>, index -> <f32>
            %104 = ascvec.mul %102, %103, %83 : <f32>
            %105 = ascvec.mul_scalar %104, %arg31, %83 : <f32>, f32
            %106 = ascvec.mul %105, %81, %83 : <f32>
            %107 = ascvec.max_scalar %106, %cst_55, %83 : <f32>, f32
            %108 = ascvec.min_scalar %107, %cst, %83 : <f32>, f32
            %109 = ascvec.cast %108, %83 {reg_layout = 0 : i32} : <f32> -> <f8E4M3FN>
            ascvec.storealign_pack %arg32[%c128], %109, %83 {mode = "b32_to_b8"} : memref<576xf8E4M3FN, 1>, index, <f8E4M3FN>
            %110 = ascvec.loadalign %arg29[%c192] : memref<576xf32, 1>, index -> <f32>
            %111 = ascvec.div %110, %80, %83 {precision_mode = true} : <f32>
            %112 = ascvec.loadalign %arg30[%c192] : memref<512xf32, 1>, index -> <f32>
            %113 = ascvec.mul %111, %112, %83 : <f32>
            %114 = ascvec.mul_scalar %113, %arg31, %83 : <f32>, f32
            %115 = ascvec.mul %114, %81, %83 : <f32>
            %116 = ascvec.max_scalar %115, %cst_55, %83 : <f32>, f32
            %117 = ascvec.min_scalar %116, %cst, %83 : <f32>, f32
            %118 = ascvec.cast %117, %83 {reg_layout = 0 : i32} : <f32> -> <f8E4M3FN>
            ascvec.storealign_pack %arg32[%c192], %118, %83 {mode = "b32_to_b8"} : memref<576xf8E4M3FN, 1>, index, <f8E4M3FN>
            %119 = ascvec.loadalign %arg29[%c256] : memref<576xf32, 1>, index -> <f32>
            %120 = ascvec.div %119, %80, %83 {precision_mode = true} : <f32>
            %121 = ascvec.loadalign %arg30[%c256] : memref<512xf32, 1>, index -> <f32>
            %122 = ascvec.mul %120, %121, %83 : <f32>
            %123 = ascvec.mul_scalar %122, %arg31, %83 : <f32>, f32
            %124 = ascvec.mul %123, %81, %83 : <f32>
            %125 = ascvec.max_scalar %124, %cst_55, %83 : <f32>, f32
            %126 = ascvec.min_scalar %125, %cst, %83 : <f32>, f32
            %127 = ascvec.cast %126, %83 {reg_layout = 0 : i32} : <f32> -> <f8E4M3FN>
            ascvec.storealign_pack %arg32[%c256], %127, %83 {mode = "b32_to_b8"} : memref<576xf8E4M3FN, 1>, index, <f8E4M3FN>
            %128 = ascvec.loadalign %arg29[%c320] : memref<576xf32, 1>, index -> <f32>
            %129 = ascvec.div %128, %80, %83 {precision_mode = true} : <f32>
            %130 = ascvec.loadalign %arg30[%c320] : memref<512xf32, 1>, index -> <f32>
            %131 = ascvec.mul %129, %130, %83 : <f32>
            %132 = ascvec.mul_scalar %131, %arg31, %83 : <f32>, f32
            %133 = ascvec.mul %132, %81, %83 : <f32>
            %134 = ascvec.max_scalar %133, %cst_55, %83 : <f32>, f32
            %135 = ascvec.min_scalar %134, %cst, %83 : <f32>, f32
            %136 = ascvec.cast %135, %83 {reg_layout = 0 : i32} : <f32> -> <f8E4M3FN>
            ascvec.storealign_pack %arg32[%c320], %136, %83 {mode = "b32_to_b8"} : memref<576xf8E4M3FN, 1>, index, <f8E4M3FN>
            %137 = ascvec.loadalign %arg29[%c384] : memref<576xf32, 1>, index -> <f32>
            %138 = ascvec.div %137, %80, %83 {precision_mode = true} : <f32>
            %139 = ascvec.loadalign %arg30[%c384] : memref<512xf32, 1>, index -> <f32>
            %140 = ascvec.mul %138, %139, %83 : <f32>
            %141 = ascvec.mul_scalar %140, %arg31, %83 : <f32>, f32
            %142 = ascvec.mul %141, %81, %83 : <f32>
            %143 = ascvec.max_scalar %142, %cst_55, %83 : <f32>, f32
            %144 = ascvec.min_scalar %143, %cst, %83 : <f32>, f32
            %145 = ascvec.cast %144, %83 {reg_layout = 0 : i32} : <f32> -> <f8E4M3FN>
            ascvec.storealign_pack %arg32[%c384], %145, %83 {mode = "b32_to_b8"} : memref<576xf8E4M3FN, 1>, index, <f8E4M3FN>
            %146 = ascvec.loadalign %arg29[%c448] : memref<576xf32, 1>, index -> <f32>
            %147 = ascvec.div %146, %80, %83 {precision_mode = true} : <f32>
            %148 = ascvec.loadalign %arg30[%c448] : memref<512xf32, 1>, index -> <f32>
            %149 = ascvec.mul %147, %148, %83 : <f32>
            %150 = ascvec.mul_scalar %149, %arg31, %83 : <f32>, f32
            %151 = ascvec.mul %150, %81, %83 : <f32>
            %152 = ascvec.max_scalar %151, %cst_55, %83 : <f32>, f32
            %153 = ascvec.min_scalar %152, %cst, %83 : <f32>, f32
            %154 = ascvec.cast %153, %83 {reg_layout = 0 : i32} : <f32> -> <f8E4M3FN>
            ascvec.storealign_pack %arg32[%c448], %154, %83 {mode = "b32_to_b8"} : memref<576xf8E4M3FN, 1>, index, <f8E4M3FN>
            %mask, %remaining_out = ascvec.update_mask %c64_i32 : !ascvec.mask
            %155 = ascvec.loadalign %arg33[%c0_56] : memref<64xf32, 1>, index -> <f32>
            %156 = ascvec.mul %155, %81, %83 : <f32>
            %157 = ascvec.max_scalar %156, %cst_55, %83 : <f32>, f32
            %158 = ascvec.min_scalar %157, %cst, %83 : <f32>, f32
            %159 = ascvec.cast %158, %83 {reg_layout = 0 : i32} : <f32> -> <f8E4M3FN>
            ascvec.storealign_pack %arg32[%c512], %159, %mask {mode = "b32_to_b8"} : memref<576xf8E4M3FN, 1>, index, <f8E4M3FN>
          }
          ascvec.mem_bar vst_vld
        } {cannir.vf_mode = "simd"}
        ascvec.release_buf v, %78, 0 {cannir.buffer.generated_sync}
        ascvec.release_buf v, %c5_i64, 0 {cannir.buffer.generated_sync}
        %79 = arith.cmpi eq, %arg23, %c0_i64 : i64
        scf.if %79 {
          %80 = arith.index_cast %arg25 : i64 to index
          %81 = memref.load %alloc_3[%80] : memref<1xi64>
          %82 = arith.cmpi sge, %81, %c0_i64 : i64
          scf.if %82 {
            %83 = arith.muli %81, %c4608_i64 : i64
            %84 = ascscalar.memref_address %alloc_2 : memref<1xf8E4M3FN>
            %85 = arith.divui %83, %c8_i64 : i64
            %86 = arith.addi %84, %85 : i64
            %87 = ascscalar.bind_gm_ptr %86 : memref<1xf8E4M3FN>
            %88 = ascscalar.bind_local_ptr %76 : memref<1xf8E4M3FN, 1>
            ascvec.get_buf mte3, %78, 0 {cannir.buffer.generated_sync}
            ascvec.copy_ub2gm %87[%c0], %88[%c0], %c1_i32, %c576_i32, %c0_i64, %c576_i32 l2_cache_ctl %c0_i32 : memref<1xf8E4M3FN>, memref<1xf8E4M3FN, 1>
            ascvec.release_buf mte3, %78, 0 {cannir.buffer.generated_sync}
          }
        } else {
          %80 = arith.index_cast %arg25 : i64 to index
          %81 = memref.load %alloc_3[%80] : memref<1xi64>
          %82 = arith.cmpi sge, %81, %c0_i64 : i64
          scf.if %82 {
            %83 = cannir.extract_layout %arg7 : !cannir.tensor<<gm, f8E4M3FN>, <<(-1, -1, 1, 576)>, <(-1, 576, 576, 1)>>, <<(-1, -1, 1, 576)>, <(-1, 576, 576, 1)>>, identity>
            %84 = cannir.extract_shape %83 : <<(-1, -1, 1, 576)>, <(-1, 576, 576, 1)>> -> <(-1, -1, 1, 576)>
            %85 = cannir.extract_dim %84[1] : !cannir.shape<(-1, -1, 1, 576)>
            %86 = arith.remsi %81, %85 : i64
            %87 = arith.cmpi ne, %86, %c0_i64 : i64
            %88 = arith.cmpi slt, %86, %c0_i64 : i64
            %89 = arith.cmpi slt, %85, %c0_i64 : i64
            %90 = arith.xori %88, %89 : i1
            %91 = arith.andi %87, %90 : i1
            %92 = arith.divsi %81, %85 : i64
            %93 = arith.extui %91 : i1 to i64
            %94 = arith.subi %92, %93 : i64
            %95 = arith.select %91, %85, %c0_i64 : i64
            %96 = arith.addi %86, %95 : i64
            ascvec.get_buf mte3, %78, 0 {cannir.buffer.generated_sync}
            scf.for %arg27 = %c0_i64 to %c18_i64 step %c1_i64  : i64 {
              %97 = arith.muli %94, %c18_i64 : i64
              %98 = arith.addi %97, %arg27 : i64
              %99 = arith.muli %98, %85 : i64
              %100 = arith.addi %99, %96 : i64
              %101 = arith.muli %100, %c256_i64 : i64
              %102 = ascscalar.memref_address %alloc_2 : memref<1xf8E4M3FN>
              %103 = arith.divui %101, %c8_i64 : i64
              %104 = arith.addi %102, %103 : i64
              %105 = ascscalar.bind_gm_ptr %104 : memref<1xf8E4M3FN>
              %106 = arith.muli %arg27, %c256_i64 overflow<nsw> : i64
              %107 = arith.divui %106, %c8_i64 : i64
              %108 = arith.addi %76, %107 : i64
              %109 = ascscalar.bind_local_ptr %108 : memref<1xf8E4M3FN, 1>
              ascvec.copy_ub2gm %105[%c0], %109[%c0], %c1_i32, %c32_i32, %c0_i64, %c32_i32 l2_cache_ctl %c0_i32 : memref<1xf8E4M3FN>, memref<1xf8E4M3FN, 1>
            }
            ascvec.release_buf mte3, %78, 0 {cannir.buffer.generated_sync}
          }
        }
        scf.yield %26 : i64
      }
    }
    ascvec.sync_all
    ascvec.sync_block_arrive mte3, %c13_i64
    ascvec.sync_block_arrive mte3, %c14_i64 {mode = 0 : i32}
    ascvec.sync_block_wait s, %c14_i64 {mode = 0 : i32}
    ascvec.sync_block_wait s, %c15_i64
    scf.if %9 {
      scf.for %arg25 = %7 to %5 step %12  : i64 {
        %17 = cannir.extract_layout %arg10 : !cannir.tensor<<gm, bf16>, <<(-1, 64)>, <(64, 1)>>, <<(-1, 64)>, <(64, 1)>>, identity>
        %18 = cannir.extract_shape %17 : <<(-1, 64)>, <(64, 1)>> -> <(-1, 64)>
        %19 = cannir.extract_dim %18[0] : !cannir.shape<(-1, 64)>
        %20 = arith.minsi %arg25, %19 : i64
        %21 = arith.subi %19, %20 : i64
        %22 = arith.minsi %21, %c1_i64 : i64
        %23 = arith.cmpi sle, %arg25, %19 : i64
        %24 = arith.select %23, %22, %c0_i64 : i64
        %25 = arith.muli %arg25, %c128_i64 : i64
        %26 = arith.muli %24, %c128_i64 : i64
        %27 = arith.cmpi eq, %26, %c0_i64 : i64
        %28 = arith.cmpi ne, %26, %c0_i64 : i64
        %29 = arith.extui %28 : i1 to i64
        %30 = arith.select %27, %c2_i64, %26 : i64
        %31 = ascscalar.bitcast_memref %alloc_44 byte_offset = %c0_i64 : memref<64xbf16, 1> -> memref<?xbf16, 1>
        ascvec.get_buf mte2, %c4_i64, 0 {cannir.buffer.generated_sync}
        "ascvec.copy_gm2ub_plan"(%31, %alloc_5, %c0_i32, %c0_i32, %c0_i64, %25, %29, %30, %c0_i64, %30, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = 1024 : i64, ub_capacity = 128 : i64}> : (memref<?xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
        ascvec.release_buf mte2, %c4_i64, 0 {cannir.buffer.generated_sync}
        ascvec.get_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
        ascvec.vf(%alloc_44, %alloc_46 : memref<64xbf16, 1>, memref<128xf32, 1>) {
        ^bb0(%arg26: memref<64xbf16, 1>, %arg27: memref<128xf32, 1>):
          %c0_55 = arith.constant 0 : index
          %c0_56 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          scf.for %arg28 = %c0_56 to %c1 step %c1 {
            %47 = ascvec.loadalign_unpack %arg26[%c0_55] {mode = "b16_to_b32"} : memref<64xbf16, 1>, index -> <bf16>
            %48 = ascvec.create_mask : !ascvec.mask
            %49 = ascvec.cast %47, %48 {rounding = 6 : i32} : <bf16> -> <f32>
            ascvec.storealign %arg27[%c0_55], %49, %48 : memref<128xf32, 1>, index, <f32>
          }
          ascvec.mem_bar vst_vld
        } {cannir.vf_mode = "simd"}
        ascvec.release_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
        %32 = cannir.extract_layout %arg11 : !cannir.tensor<<gm, bf16>, <<(-1, 64)>, <(64, 1)>>, <<(-1, 64)>, <(64, 1)>>, identity>
        %33 = cannir.extract_shape %32 : <<(-1, 64)>, <(64, 1)>> -> <(-1, 64)>
        %34 = cannir.extract_dim %33[0] : !cannir.shape<(-1, 64)>
        %35 = arith.minsi %arg25, %34 : i64
        %36 = arith.subi %34, %35 : i64
        %37 = arith.minsi %36, %c1_i64 : i64
        %38 = arith.cmpi sle, %arg25, %34 : i64
        %39 = arith.select %38, %37, %c0_i64 : i64
        %40 = arith.muli %39, %c128_i64 : i64
        %41 = arith.cmpi eq, %40, %c0_i64 : i64
        %42 = arith.cmpi ne, %40, %c0_i64 : i64
        %43 = arith.extui %42 : i1 to i64
        %44 = arith.select %41, %c2_i64, %40 : i64
        %45 = ascscalar.bitcast_memref %alloc_45 byte_offset = %c0_i64 : memref<64xbf16, 1> -> memref<?xbf16, 1>
        ascvec.get_buf mte2, %c5_i64, 0 {cannir.buffer.generated_sync}
        "ascvec.copy_gm2ub_plan"(%45, %alloc_6, %c0_i32, %c0_i32, %c0_i64, %25, %43, %44, %c0_i64, %44, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = 1152 : i64, ub_capacity = 128 : i64}> : (memref<?xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
        ascvec.release_buf mte2, %c5_i64, 0 {cannir.buffer.generated_sync}
        ascvec.get_buf v, %c5_i64, 0 {cannir.buffer.generated_sync}
        ascvec.vf(%alloc_45, %alloc_47 : memref<64xbf16, 1>, memref<128xf32, 1>) {
        ^bb0(%arg26: memref<64xbf16, 1>, %arg27: memref<128xf32, 1>):
          %c0_55 = arith.constant 0 : index
          %c0_56 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          scf.for %arg28 = %c0_56 to %c1 step %c1 {
            %47 = ascvec.loadalign_unpack %arg26[%c0_55] {mode = "b16_to_b32"} : memref<64xbf16, 1>, index -> <bf16>
            %48 = ascvec.create_mask : !ascvec.mask
            %49 = ascvec.cast %47, %48 {rounding = 6 : i32} : <bf16> -> <f32>
            ascvec.storealign %arg27[%c0_55], %49, %48 : memref<128xf32, 1>, index, <f32>
          }
          ascvec.mem_bar vst_vld
        } {cannir.vf_mode = "simd"}
        ascvec.release_buf v, %c5_i64, 0 {cannir.buffer.generated_sync}
        %46 = scf.for %arg26 = %c0_i64 to %c2_i64 step %c1_i64 iter_args(%arg27 = %c0_i64) -> (i64)  : i64 {
          %47 = arith.muli %arg26, %c192_i64 overflow<nsw> : i64
          %48 = arith.addi %47, %c128_i64 overflow<nsw> : i64
          %49 = arith.cmpi eq, %arg27, %c0_i64 : i64
          %50 = ascscalar.memref_address %alloc_39 : memref<64xbf16, 1>
          %51 = ascscalar.memref_address %alloc_40 : memref<64xbf16, 1>
          %52 = arith.select %49, %50, %51 : i64
          %53 = arith.cmpi ne, %arg27, %c0_i64 : i64
          %54 = arith.extui %53 : i1 to i64
          %55 = arith.addi %arg27, %c1_i64 overflow<nsw> : i64
          %56 = arith.remui %55, %c2_i64 : i64
          %57 = arith.divsi %48, %c64_i64 : i64
          %58 = arith.minsi %arg25, %c8_i64 : i64
          %59 = arith.subi %c8_i64, %58 : i64
          %60 = arith.minsi %59, %c1_i64 : i64
          %61 = arith.cmpi sle, %arg25, %c8_i64 : i64
          %62 = arith.select %61, %60, %c0_i64 : i64
          %63 = arith.muli %arg25, %c768_i64 : i64
          %64 = arith.muli %57, %c128_i64 overflow<nsw> : i64
          %65 = arith.addi %63, %64 : i64
          %66 = ascscalar.bind_local_ptr %52 : memref<1xbf16, 1>
          ascvec.get_buf mte2, %54, 0 {cannir.buffer.generated_sync}
          "ascvec.copy_gm2ub_plan"(%66, %alloc_12, %c0_i32, %c0_i32, %c0_i64, %65, %62, %c128_i64, %c768_i64, %c128_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 6144 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<3072xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
          ascvec.release_buf mte2, %54, 0 {cannir.buffer.generated_sync}
          %67 = ascscalar.bind_local_ptr %52 : memref<64xbf16, 1>
          ascvec.get_buf v, %54, 0 {cannir.buffer.generated_sync}
          ascvec.vf(%67, %alloc_41 : memref<64xbf16, 1>, memref<64xf32, 1>) {
          ^bb0(%arg28: memref<64xbf16, 1>, %arg29: memref<64xf32, 1>):
            %c0_55 = arith.constant 0 : index
            %c0_56 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            scf.for %arg30 = %c0_56 to %c1 step %c1 {
              %86 = ascvec.loadalign_unpack %arg28[%c0_55] {mode = "b16_to_b32"} : memref<64xbf16, 1>, index -> <bf16>
              %87 = ascvec.create_mask : !ascvec.mask
              %88 = ascvec.cast %86, %87 {rounding = 6 : i32} : <bf16> -> <f32>
              ascvec.storealign %arg29[%c0_55], %88, %87 : memref<64xf32, 1>, index, <f32>
            }
            ascvec.mem_bar vst_vld
          } {cannir.vf_mode = "simd"}
          ascvec.release_buf v, %54, 0 {cannir.buffer.generated_sync}
          ascvec.vf(%alloc_41, %alloc_48 : memref<64xf32, 1>, memref<128xf32, 1>) {
          ^bb0(%arg28: memref<64xf32, 1>, %arg29: memref<128xf32, 1>):
            %c0_55 = arith.constant 0 : index
            %c0_56 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            scf.for %arg30 = %c0_56 to %c1 step %c1 {
              %86 = ascvec.loadalign %arg28[%c0_55] : memref<64xf32, 1>, index -> <f32>
              %87 = ascvec.create_mask : !ascvec.mask
              ascvec.storealign %arg29[%c0_55], %86, %87 : memref<128xf32, 1>, index, <f32>
            }
            ascvec.mem_bar vst_vld
          } {cannir.vf_mode = "simd"}
          %68 = arith.cmpi eq, %arg24, %c1_i64 : i64
          scf.if %68 {
            ascvec.vf(%alloc_47, %alloc_46, %alloc_48, %alloc_49 : memref<128xf32, 1>, memref<128xf32, 1>, memref<128xf32, 1>, memref<64xf32, 1>) {
            ^bb0(%arg28: memref<128xf32, 1>, %arg29: memref<128xf32, 1>, %arg30: memref<128xf32, 1>, %arg31: memref<64xf32, 1>):
              %c32 = arith.constant 32 : index
              %c0_55 = arith.constant 0 : index
              %c32_i32_56 = arith.constant 32 : i32
              %c0_57 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              scf.for %arg32 = %c0_57 to %c1 step %c1 {
                %mask, %remaining_out = ascvec.update_mask %c32_i32_56 : !ascvec.mask
                %86 = ascvec.loadalign %arg28[%c0_55] : memref<128xf32, 1>, index -> <f32>
                %87 = ascvec.loadalign %arg28[%c32] : memref<128xf32, 1>, index -> <f32>
                %88 = ascvec.loadalign %arg29[%c0_55] : memref<128xf32, 1>, index -> <f32>
                %89 = ascvec.loadalign %arg29[%c32] : memref<128xf32, 1>, index -> <f32>
                %result0, %result1 = ascvec.loadalign_deintlv %arg30[%c0_55] {dintlv_width = "b32"} : memref<128xf32, 1>, index -> <f32>, <f32>
                %90 = ascvec.mul %result0, %86, %mask : <f32>
                %91 = ascvec.mul %result1, %88, %mask : <f32>
                %92 = ascvec.sub %90, %91, %mask : <f32>
                %93 = ascvec.mul %result1, %87, %mask : <f32>
                %94 = ascvec.mul %result0, %89, %mask : <f32>
                %95 = ascvec.add %93, %94, %mask : <f32>
                ascvec.storealign %arg31[%c0_55], %92, %mask : memref<64xf32, 1>, index, <f32>
                ascvec.storealign %arg31[%c32], %95, %mask : memref<64xf32, 1>, index, <f32>
              }
              ascvec.mem_bar vst_vld
            } {cannir.vf_mode = "simd"}
          } else {
            ascvec.vf(%alloc_47, %alloc_46, %alloc_48, %alloc_49 : memref<128xf32, 1>, memref<128xf32, 1>, memref<128xf32, 1>, memref<64xf32, 1>) {
            ^bb0(%arg28: memref<128xf32, 1>, %arg29: memref<128xf32, 1>, %arg30: memref<128xf32, 1>, %arg31: memref<64xf32, 1>):
              %c32 = arith.constant 32 : index
              %c0_55 = arith.constant 0 : index
              %c32_i32_56 = arith.constant 32 : i32
              %c0_57 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              scf.for %arg32 = %c0_57 to %c1 step %c1 {
                %mask, %remaining_out = ascvec.update_mask %c32_i32_56 : !ascvec.mask
                %86 = ascvec.loadalign %arg28[%c0_55] : memref<128xf32, 1>, index -> <f32>
                %87 = ascvec.loadalign %arg28[%c32] : memref<128xf32, 1>, index -> <f32>
                %88 = ascvec.loadalign %arg29[%c0_55] : memref<128xf32, 1>, index -> <f32>
                %89 = ascvec.loadalign %arg29[%c32] : memref<128xf32, 1>, index -> <f32>
                %90 = ascvec.loadalign %arg30[%c0_55] : memref<128xf32, 1>, index -> <f32>
                %91 = ascvec.loadalign %arg30[%c32] : memref<128xf32, 1>, index -> <f32>
                %92 = ascvec.mul %90, %86, %mask : <f32>
                %93 = ascvec.mul %91, %88, %mask : <f32>
                %94 = ascvec.sub %92, %93, %mask : <f32>
                %95 = ascvec.mul %91, %87, %mask : <f32>
                %96 = ascvec.mul %90, %89, %mask : <f32>
                %97 = ascvec.add %95, %96, %mask : <f32>
                ascvec.storealign %arg31[%c0_55], %94, %mask : memref<64xf32, 1>, index, <f32>
                ascvec.storealign %arg31[%c32], %97, %mask : memref<64xf32, 1>, index, <f32>
              }
              ascvec.mem_bar vst_vld
            } {cannir.vf_mode = "simd"}
          }
          %69 = ascscalar.memref_address %alloc_42 : memref<64xf32, 1>
          %70 = ascscalar.memref_address %alloc_43 : memref<64xf32, 1>
          %71 = arith.select %49, %69, %70 : i64
          %72 = ascscalar.bind_local_ptr %71 : memref<64xf32, 1>
          %73 = arith.select %49, %c2_i64, %c3_i64 : i64
          ascvec.get_buf v, %73, 0 {cannir.buffer.generated_sync}
          ascvec.vf(%alloc_49, %72 : memref<64xf32, 1>, memref<64xf32, 1>) {
          ^bb0(%arg28: memref<64xf32, 1>, %arg29: memref<64xf32, 1>):
            %c0_55 = arith.constant 0 : index
            %c0_56 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            scf.for %arg30 = %c0_56 to %c1 step %c1 {
              %86 = ascvec.loadalign %arg28[%c0_55] : memref<64xf32, 1>, index -> <f32>
              %87 = ascvec.create_mask : !ascvec.mask
              ascvec.storealign %arg29[%c0_55], %86, %87 : memref<64xf32, 1>, index, <f32>
            }
            ascvec.mem_bar vst_vld
          } {cannir.vf_mode = "simd"}
          ascvec.release_buf v, %73, 0 {cannir.buffer.generated_sync}
          %74 = arith.muli %arg26, %c576_i64 overflow<nsw> : i64
          %75 = arith.addi %74, %c512_i64 overflow<nsw> : i64
          %76 = arith.divsi %75, %c64_i64 : i64
          %77 = arith.muli %arg25, %c1152_i64 : i64
          %78 = arith.muli %76, %c64_i64 overflow<nsw> : i64
          %79 = arith.addi %77, %78 : i64
          %80 = arith.muli %79, %c32_i64 : i64
          %81 = ascscalar.memref_address %alloc_13 : memref<9216xf32>
          %82 = arith.divui %80, %c8_i64 : i64
          %83 = arith.addi %81, %82 : i64
          %84 = ascscalar.bind_gm_ptr %83 : memref<1xf32>
          %85 = ascscalar.bind_local_ptr %71 : memref<1xf32, 1>
          ascvec.get_buf mte3, %73, 0 {cannir.buffer.generated_sync}
          ascvec.copy_ub2gm %84[%c0], %85[%c0], %c1_i32, %c256_i32, %c0_i64, %c256_i32 l2_cache_ctl %c0_i32 : memref<1xf32>, memref<1xf32, 1>
          ascvec.release_buf mte3, %73, 0 {cannir.buffer.generated_sync}
          scf.yield %56 : i64
        }
      }
    }
    ascvec.sync_all
    ascvec.sync_block_arrive mte3, %c13_i64
    ascvec.sync_block_arrive mte3, %c14_i64 {mode = 0 : i32}
    ascvec.sync_block_wait s, %c14_i64 {mode = 0 : i32}
    ascvec.sync_block_wait s, %c15_i64
    %15 = arith.muli %5, %c2_i64 : i64
    %16 = arith.cmpi slt, %7, %15 : i64
    scf.if %16 {
      %17 = scf.for %arg25 = %7 to %15 step %12 iter_args(%arg26 = %c0_i64) -> (i64)  : i64 {
        %18 = arith.remsi %arg25, %c2_i64 : i64
        %19 = arith.cmpi ne, %18, %c0_i64 : i64
        %20 = arith.cmpi slt, %18, %c0_i64 : i64
        %21 = arith.andi %19, %20 : i1
        %22 = arith.divsi %arg25, %c2_i64 : i64
        %23 = arith.extui %21 : i1 to i64
        %24 = arith.subi %22, %23 : i64
        %25 = arith.select %21, %c2_i64, %c0_i64 : i64
        %26 = arith.addi %18, %25 overflow<nsw> : i64
        %27 = arith.cmpi eq, %arg26, %c0_i64 : i64
        %28 = ascscalar.memref_address %alloc_50 : memref<576xf32, 1>
        %29 = ascscalar.memref_address %alloc_51 : memref<576xf32, 1>
        %30 = arith.select %27, %28, %29 : i64
        %31 = arith.cmpi ne, %arg26, %c0_i64 : i64
        %32 = arith.extui %31 : i1 to i64
        %33 = arith.addi %arg26, %c1_i64 overflow<nsw> : i64
        %34 = arith.remui %33, %c2_i64 : i64
        %35 = arith.minsi %24, %c8_i64 : i64
        %36 = arith.subi %c8_i64, %35 : i64
        %37 = arith.minsi %36, %c1_i64 : i64
        %38 = arith.cmpi sle, %24, %c8_i64 : i64
        %39 = arith.select %38, %37, %c0_i64 : i64
        %40 = arith.muli %24, %c4608_i64 : i64
        %41 = arith.muli %26, %c2304_i64 overflow<nsw> : i64
        %42 = arith.addi %40, %41 : i64
        %43 = ascscalar.bind_local_ptr %30 : memref<1xf32, 1>
        ascvec.get_buf mte2, %32, 0 {cannir.buffer.generated_sync}
        "ascvec.copy_gm2ub_plan"(%43, %alloc_13, %c0_i32, %c0_i32, %c0_i64, %42, %39, %c2304_i64, %c4608_i64, %c2304_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 36864 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = -1 : i64, ub_capacity = 4 : i64}> : (memref<1xf32, 1>, memref<9216xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
        ascvec.release_buf mte2, %32, 0 {cannir.buffer.generated_sync}
        %44 = ascscalar.bind_local_ptr %30 : memref<576xf32, 1>
        ascvec.get_buf v, %32, 0 {cannir.buffer.generated_sync}
        ascvec.get_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
        ascvec.vf(%44, %alloc_54 : memref<576xf32, 1>, memref<8xf32, 1>) {
        ^bb0(%arg27: memref<576xf32, 1>, %arg28: memref<8xf32, 1>):
          %c2 = arith.constant 2 : index
          %c1 = arith.constant 1 : index
          %c512 = arith.constant 512 : index
          %c448 = arith.constant 448 : index
          %c384 = arith.constant 384 : index
          %c320 = arith.constant 320 : index
          %c256 = arith.constant 256 : index
          %c192 = arith.constant 192 : index
          %c128 = arith.constant 128 : index
          %c64 = arith.constant 64 : index
          %cst = arith.constant 1.000000e+00 : f32
          %cst_55 = arith.constant 4.480000e+02 : f32
          %c0_56 = arith.constant 0 : index
          %c64_i32 = arith.constant 64 : i32
          %cst_57 = arith.constant 0.000000e+00 : f32
          %c1_i32_58 = arith.constant 1 : i32
          %c0_59 = arith.constant 0 : index
          %c1_60 = arith.constant 1 : index
          scf.for %arg29 = %c0_59 to %c1_60 step %c1_60 {
            %mask, %remaining_out = ascvec.update_mask %c1_i32_58 : !ascvec.mask
            %65 = ascvec.duplicate_scalar %cst_57, %mask : f32 -> <f32>
            %mask_61, %remaining_out_62 = ascvec.update_mask %c64_i32 : !ascvec.mask
            %66 = ascvec.loadalign %arg27[%c0_56] : memref<576xf32, 1>, index -> <f32>
            %67 = ascvec.abs %66, %mask_61 : <f32>
            %68 = ascvec.reduce_max %67, %mask_61 : <f32>
            %69 = ascvec.max %65, %68, %mask : <f32>
            %mask_63, %remaining_out_64 = ascvec.update_mask %c64_i32 : !ascvec.mask
            %70 = ascvec.loadalign %arg27[%c64] : memref<576xf32, 1>, index -> <f32>
            %71 = ascvec.abs %70, %mask_63 : <f32>
            %72 = ascvec.reduce_max %71, %mask_63 : <f32>
            %73 = ascvec.max %69, %72, %mask : <f32>
            %mask_65, %remaining_out_66 = ascvec.update_mask %c64_i32 : !ascvec.mask
            %74 = ascvec.loadalign %arg27[%c128] : memref<576xf32, 1>, index -> <f32>
            %75 = ascvec.abs %74, %mask_65 : <f32>
            %76 = ascvec.reduce_max %75, %mask_65 : <f32>
            %77 = ascvec.max %73, %76, %mask : <f32>
            %mask_67, %remaining_out_68 = ascvec.update_mask %c64_i32 : !ascvec.mask
            %78 = ascvec.loadalign %arg27[%c192] : memref<576xf32, 1>, index -> <f32>
            %79 = ascvec.abs %78, %mask_67 : <f32>
            %80 = ascvec.reduce_max %79, %mask_67 : <f32>
            %81 = ascvec.max %77, %80, %mask : <f32>
            %mask_69, %remaining_out_70 = ascvec.update_mask %c64_i32 : !ascvec.mask
            %82 = ascvec.loadalign %arg27[%c256] : memref<576xf32, 1>, index -> <f32>
            %83 = ascvec.abs %82, %mask_69 : <f32>
            %84 = ascvec.reduce_max %83, %mask_69 : <f32>
            %85 = ascvec.max %81, %84, %mask : <f32>
            %mask_71, %remaining_out_72 = ascvec.update_mask %c64_i32 : !ascvec.mask
            %86 = ascvec.loadalign %arg27[%c320] : memref<576xf32, 1>, index -> <f32>
            %87 = ascvec.abs %86, %mask_71 : <f32>
            %88 = ascvec.reduce_max %87, %mask_71 : <f32>
            %89 = ascvec.max %85, %88, %mask : <f32>
            %mask_73, %remaining_out_74 = ascvec.update_mask %c64_i32 : !ascvec.mask
            %90 = ascvec.loadalign %arg27[%c384] : memref<576xf32, 1>, index -> <f32>
            %91 = ascvec.abs %90, %mask_73 : <f32>
            %92 = ascvec.reduce_max %91, %mask_73 : <f32>
            %93 = ascvec.max %89, %92, %mask : <f32>
            %mask_75, %remaining_out_76 = ascvec.update_mask %c64_i32 : !ascvec.mask
            %94 = ascvec.loadalign %arg27[%c448] : memref<576xf32, 1>, index -> <f32>
            %95 = ascvec.abs %94, %mask_75 : <f32>
            %96 = ascvec.reduce_max %95, %mask_75 : <f32>
            %97 = ascvec.max %93, %96, %mask : <f32>
            %mask_77, %remaining_out_78 = ascvec.update_mask %c64_i32 : !ascvec.mask
            %98 = ascvec.loadalign %arg27[%c512] : memref<576xf32, 1>, index -> <f32>
            %99 = ascvec.abs %98, %mask_77 : <f32>
            %100 = ascvec.reduce_max %99, %mask_77 : <f32>
            %101 = ascvec.max %97, %100, %mask : <f32>
            %102 = ascvec.duplicate_scalar %cst_55, %mask : f32 -> <f32>
            %103 = ascvec.div %101, %102, %mask {precision_mode = true} : <f32>
            %104 = ascvec.cmp_eq %101, %65, %mask : <f32>, <f32> -> !ascvec.mask
            %105 = ascvec.duplicate_scalar %cst, %mask : f32 -> <f32>
            %106 = ascvec.select %105, %103, %104 : <f32>
            ascvec.storealign_1st %arg28[%c0_56], %103 : memref<8xf32, 1>, index, <f32>
            ascvec.storealign_1st %arg28[%c1], %106 : memref<8xf32, 1>, index, <f32>
            ascvec.storealign_1st %arg28[%c2], %101 : memref<8xf32, 1>, index, <f32>
          }
          ascvec.mem_bar vst_vld
        } {cannir.vf_mode = "simd"}
        %45 = ascscalar.memref_address %alloc_52 : memref<576xf8E4M3FN, 1>
        %46 = ascscalar.memref_address %alloc_53 : memref<576xf8E4M3FN, 1>
        %47 = arith.select %27, %45, %46 : i64
        %48 = ascscalar.bind_local_ptr %47 : memref<576xf8E4M3FN, 1>
        %49 = arith.select %27, %c2_i64, %c3_i64 : i64
        ascvec.get_buf v, %49, 0 {cannir.buffer.generated_sync}
        ascvec.vf(%alloc_54, %44, %48 : memref<8xf32, 1>, memref<576xf32, 1>, memref<576xf8E4M3FN, 1>) {
        ^bb0(%arg27: memref<8xf32, 1>, %arg28: memref<576xf32, 1>, %arg29: memref<576xf8E4M3FN, 1>):
          %c2 = arith.constant 2 : index
          %c64_i32 = arith.constant 64 : i32
          %c576_i32_55 = arith.constant 576 : i32
          %c0_i32_56 = arith.constant 0 : i32
          %cst = arith.constant 4.480000e+02 : f32
          %cst_57 = arith.constant -4.480000e+02 : f32
          %cst_58 = arith.constant 0.000000e+00 : f32
          %c1 = arith.constant 1 : index
          %65 = ascvec.loadalign_brc %arg27[%c1] : memref<8xf32, 1>, index -> <f32>
          %66 = ascvec.loadalign_brc %arg27[%c2] : memref<8xf32, 1>, index -> <f32>
          %67 = ascvec.create_mask : !ascvec.mask
          %68 = ascvec.duplicate_scalar %cst_58, %67 : f32 -> <f32>
          %69 = ascvec.cmp_eq %66, %68, %67 : <f32>, <f32> -> !ascvec.mask
          scf.for %arg30 = %c0_i32_56 to %c576_i32_55 step %c64_i32  : i32 {
            %70 = arith.index_cast %arg30 : i32 to index
            %71 = ascvec.loadalign %arg28[%70] : memref<576xf32, 1>, index -> <f32>
            %72 = ascvec.div %71, %65, %67 {precision_mode = true} : <f32>
            %73 = ascvec.select %68, %72, %69 : <f32>
            %74 = ascvec.max_scalar %73, %cst_57, %67 : <f32>, f32
            %75 = ascvec.min_scalar %74, %cst, %67 : <f32>, f32
            %76 = ascvec.cast %75, %67 {reg_layout = 0 : i32} : <f32> -> <f8E4M3FN>
            ascvec.storealign_pack %arg29[%70], %76, %67 {mode = "b32_to_b8"} : memref<576xf8E4M3FN, 1>, index, <f8E4M3FN>
          }
          ascvec.mem_bar vst_vld
        } {cannir.vf_mode = "simd"}
        ascvec.release_buf v, %49, 0 {cannir.buffer.generated_sync}
        ascvec.release_buf v, %32, 0 {cannir.buffer.generated_sync}
        ascvec.release_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
        %50 = arith.muli %24, %c1152_i64 : i64
        %51 = arith.muli %26, %c576_i64 overflow<nsw> : i64
        %52 = arith.addi %50, %51 : i64
        %53 = arith.muli %52, %c8_i64 : i64
        %54 = ascscalar.memref_address %alloc_7 : memref<9216xf8E4M3FN>
        %55 = arith.divui %53, %c8_i64 : i64
        %56 = arith.addi %54, %55 : i64
        %57 = ascscalar.bind_gm_ptr %56 : memref<1xf8E4M3FN>
        %58 = ascscalar.bind_local_ptr %47 : memref<1xf8E4M3FN, 1>
        ascvec.get_buf mte3, %49, 0 {cannir.buffer.generated_sync}
        ascvec.copy_ub2gm %57[%c0], %58[%c0], %c1_i32, %c576_i32, %c0_i64, %c576_i32 l2_cache_ctl %c0_i32 : memref<1xf8E4M3FN>, memref<1xf8E4M3FN, 1>
        ascvec.release_buf mte3, %49, 0 {cannir.buffer.generated_sync}
        %59 = arith.muli %arg25, %c32_i64 : i64
        %60 = ascscalar.memref_address %alloc_8 : memref<1xf32>
        %61 = arith.divui %59, %c8_i64 : i64
        %62 = arith.addi %60, %61 : i64
        %63 = ascscalar.bind_gm_ptr %62 : memref<1xf32>
        %64 = ascscalar.bitcast_memref %alloc_54 byte_offset = %c0_i64 : memref<8xf32, 1> -> memref<?xf32, 1>
        ascvec.get_buf mte3, %c4_i64, 0 {cannir.buffer.generated_sync}
        ascvec.copy_ub2gm %63[%c0], %64[%c0], %c1_i32, %c4_i32, %c0_i64, %c4_i32 l2_cache_ctl %c0_i32 : memref<1xf32>, memref<?xf32, 1>
        ascvec.release_buf mte3, %c4_i64, 0 {cannir.buffer.generated_sync}
        scf.yield %34 : i64
      }
    }
    return
  }
  func.func @kernel_MlaPrologKernel_run_m2_rope_0_aic(%arg0: !cannir.tensor<<gm, bf16>, <<(-1, 1024)>, <(1024, 1)>>, <<(-1, 1024)>, <(1024, 1)>>, identity>, %arg1: !cannir.tensor<<gm, bf16>, <<(640, 1024)>, <(1024, 1)>>, <<(64, 40, 16, 16)>, <(10240, 256, 16, 1)>>, nd2nz> {cannir.physical_shape_upper_bound = array<i64: 64, 40, 16, 16>}, %arg2: !cannir.tensor<<gm, bf16>, <<(384, 640)>, <(640, 1)>>, <<(40, 24, 16, 16)>, <(6144, 256, 16, 1)>>, nd2nz> {cannir.physical_shape_upper_bound = array<i64: 40, 24, 16, 16>}, %arg3: !cannir.tensor<<gm, bf16>, <<(576, 1024)>, <(1024, 1)>>, <<(64, 36, 16, 16)>, <(9216, 256, 16, 1)>>, nd2nz> {cannir.physical_shape_upper_bound = array<i64: 64, 36, 16, 16>}, %arg4: !cannir.tensor<<gm, bf16>, <<(2, 128, 512)>, <(65536, 512, 1)>>, <<(2, 128, 512)>, <(65536, 512, 1)>>, identity>, %arg5: !cannir.tensor<<gm, f32>, <<(1, 640)>, <(640, 1)>>, <<(1, 640)>, <(640, 1)>>, identity>, %arg6: !cannir.tensor<<gm, f32>, <<(1, 512)>, <(512, 1)>>, <<(1, 512)>, <(512, 1)>>, identity>, %arg7: !cannir.tensor<<gm, f8E4M3FN>, <<(-1, -1, 1, 576)>, <(-1, 576, 576, 1)>>, <<(-1, -1, 1, 576)>, <(-1, 576, 576, 1)>>, identity>, %arg8: !cannir.tensor<<gm, i64>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg9: !cannir.tensor<<gm, f32>, <<(1)>, <(1)>>, <<(1)>, <(1)>>, identity>, %arg10: !cannir.tensor<<gm, bf16>, <<(-1, 64)>, <(64, 1)>>, <<(-1, 64)>, <(64, 1)>>, identity>, %arg11: !cannir.tensor<<gm, bf16>, <<(-1, 64)>, <(64, 1)>>, <<(-1, 64)>, <(64, 1)>>, identity>, %arg12: !cannir.tensor<<gm, f8E4M3FN>, <<(8, 2, 576)>, <(1152, 576, 1)>>, <<(8, 2, 576)>, <(1152, 576, 1)>>, identity>, %arg13: !cannir.tensor<<gm, f32>, <<(-1, 2, 1)>, <(2, 1, 1)>>, <<(-1, 2, 1)>, <(2, 1, 1)>>, identity>, %arg14: !cannir.tensor<<gm, bf16>, <<(8, 1024)>, <(1024, 1)>>, <<(8, 1024)>, <(1024, 1)>>, identity>, %arg15: !cannir.tensor<<gm, bf16>, <<(8, 672)>, <(672, 1)>>, <<(8, 672)>, <(672, 1)>>, identity>, %arg16: !cannir.tensor<<gm, bf16>, <<(8, 576)>, <(576, 1)>>, <<(8, 576)>, <(576, 1)>>, identity>, %arg17: !cannir.tensor<<gm, bf16>, <<(8, 384)>, <(384, 1)>>, <<(8, 384)>, <(384, 1)>>, identity>, %arg18: !cannir.tensor<<gm, f32>, <<(8, 2, 576)>, <(1152, 576, 1)>>, <<(8, 2, 576)>, <(1152, 576, 1)>>, identity>, %arg19: !cannir.tensor<<gm, bf16>, <<(8, 640)>, <(640, 1)>>, <<(8, 640)>, <(640, 1)>>, identity>, %arg20: f32, %arg21: f32, %arg22: f32, %arg23: i64, %arg24: i64) attributes {cannir.buffer.auto_sync, cannir.buffer.sync_analyzed, cannir.buffer.sync_verified, cannir.buffer.tensor_state, cannir.core = "aic", cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.device, cannir.explicit_ring_buffer_cursors, cannir.kernel_group = "kernel_MlaPrologKernel_run_m2_rope_0", cannir.resource.ids_prepared} {
    %c1152_i32 = arith.constant 1152 : i32
    %c4_i32 = arith.constant 4 : i32
    %c384_i32 = arith.constant 384 : i32
    %c96_i32 = arith.constant 96 : i32
    %c6_i32 = arith.constant 6 : i32
    %c3072_i32 = arith.constant 3072 : i32
    %c672_i32 = arith.constant 672 : i32
    %c24_i32 = arith.constant 24 : i32
    %c2_i32 = arith.constant 2 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %c16_i32 = arith.constant 16 : i32
    %c512_i32 = arith.constant 512 : i32
    %c0_i32 = arith.constant 0 : i32
    %c2048_i64 = arith.constant 2048 : i64
    %c128_i32 = arith.constant 128 : i32
    %c0 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    %c512 = arith.constant 512 : index
    %c6144 = arith.constant 6144 : index
    %c1_i32 = arith.constant 1 : i32
    %c1280_i64 = arith.constant 1280 : i64
    %c64_i32 = arith.constant 64 : i32
    %c576_i32 = arith.constant 576 : i32
    %c128_i64 = arith.constant 128 : i64
    %c32_i32 = arith.constant 32 : i32
    %c384_i64 = arith.constant 384 : i64
    %c96_i64 = arith.constant 96 : i64
    %c192_i64 = arith.constant 192 : i64
    %c768_i64 = arith.constant 768 : i64
    %c576_i64 = arith.constant 576 : i64
    %c512_i64 = arith.constant 512 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %c16 = arith.constant 16 : index
    %c10240 = arith.constant 10240 : index
    %c20480_i64 = arith.constant 20480 : i64
    %c15 = arith.constant 15 : index
    %c9216 = arith.constant 9216 : index
    %c18432_i64 = arith.constant 18432 : i64
    %c12288_i64 = arith.constant 12288 : i64
    %c336_i64 = arith.constant 336 : i64
    %c288_i64 = arith.constant 288 : i64
    %c64_i64 = arith.constant 64 : i64
    %c32_i64 = arith.constant 32 : i64
    %c20_i64 = arith.constant 20 : i64
    %c18_i64 = arith.constant 18 : i64
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
    %alloc = memref.alloc() {cannir.arg_index = 1 : i64} : memref<655360xbf16>
    %alloc_0 = memref.alloc() {cannir.arg_index = 2 : i64} : memref<245760xbf16>
    %alloc_1 = memref.alloc() {cannir.arg_index = 3 : i64} : memref<589824xbf16>
    %alloc_2 = memref.alloc() {cannir.arg_index = 4 : i64} : memref<131072xbf16>
    %alloc_3 = memref.alloc() {cannir.arg_index = 14 : i64} : memref<8192xbf16>
    %alloc_4 = memref.alloc() {cannir.arg_index = 15 : i64} : memref<5376xbf16>
    %alloc_5 = memref.alloc() {cannir.arg_index = 16 : i64} : memref<4608xbf16>
    %alloc_6 = memref.alloc() {cannir.arg_index = 17 : i64} : memref<3072xbf16>
    %alloc_7 = memref.alloc() {cannir.arg_index = 18 : i64} : memref<9216xf32>
    %alloc_8 = memref.alloc() {cannir.arg_index = 19 : i64} : memref<5120xbf16>
    %alloc_9 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<1024xbf16, 3>
    %alloc_10 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<8192xbf16, 4>
    %alloc_11 = memref.alloc() {cannir.byte_offset = 20480 : i64} : memref<8192xbf16, 2>
    %alloc_12 = memref.alloc() {cannir.byte_offset = 4096 : i64} : memref<8192xbf16, 2>
    %alloc_13 = memref.alloc() {cannir.byte_offset = 2048 : i64} : memref<1024xbf16, 2>
    %alloc_14 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<1024xbf16, 2>
    %alloc_15 = memref.alloc() {cannir.byte_offset = 8192 : i64} : memref<2048xf32, 5>
    %alloc_16 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<2048xf32, 5>
    %alloc_17 = memref.alloc() {cannir.byte_offset = 24576 : i64} : memref<12288xbf16, 4>
    %alloc_18 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<12288xbf16, 4>
    %alloc_19 = memref.alloc() {cannir.byte_offset = 4096 : i64} : memref<2048xbf16, 3>
    %alloc_20 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<2048xbf16, 3>
    %alloc_21 = memref.alloc() {cannir.byte_offset = 32768 : i64} : memref<12288xbf16, 2>
    %alloc_22 = memref.alloc() {cannir.byte_offset = 8192 : i64} : memref<12288xbf16, 2>
    %alloc_23 = memref.alloc() {cannir.byte_offset = 4096 : i64} : memref<2048xbf16, 2>
    %alloc_24 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<2048xbf16, 2>
    %alloc_25 = memref.alloc() {cannir.byte_offset = 6144 : i64} : memref<1536xf32, 5>
    %alloc_26 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<1536xf32, 5>
    %alloc_27 = memref.alloc() {cannir.byte_offset = 8192 : i64} : memref<4096xbf16, 4>
    %alloc_28 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<4096xbf16, 4>
    %alloc_29 = memref.alloc() {cannir.byte_offset = 4096 : i64} : memref<2048xbf16, 3>
    %alloc_30 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<2048xbf16, 3>
    %alloc_31 = memref.alloc() {cannir.byte_offset = 65536 : i64} : memref<16384xbf16, 2>
    %alloc_32 = memref.alloc() {cannir.byte_offset = 32768 : i64} : memref<16384xbf16, 2>
    %alloc_33 = memref.alloc() {cannir.byte_offset = 16384 : i64} : memref<8192xbf16, 2>
    %alloc_34 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<8192xbf16, 2>
    %alloc_35 = memref.alloc() {cannir.byte_offset = 2048 : i64} : memref<512xf32, 5>
    %alloc_36 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<512xf32, 5>
    %alloc_37 = memref.alloc() {cannir.byte_offset = 8192 : i64} : memref<4096xbf16, 4>
    %alloc_38 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<4096xbf16, 4>
    %alloc_39 = memref.alloc() {cannir.byte_offset = 4096 : i64} : memref<2048xbf16, 3>
    %alloc_40 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<2048xbf16, 3>
    %alloc_41 = memref.alloc() {cannir.byte_offset = 65536 : i64} : memref<16384xbf16, 2>
    %alloc_42 = memref.alloc() {cannir.byte_offset = 32768 : i64} : memref<16384xbf16, 2>
    %alloc_43 = memref.alloc() {cannir.byte_offset = 16384 : i64} : memref<8192xbf16, 2>
    %alloc_44 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<8192xbf16, 2>
    %alloc_45 = memref.alloc() {cannir.byte_offset = 2048 : i64} : memref<512xf32, 5>
    %alloc_46 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<512xf32, 5>
    %0 = ascscalar.get_block_idx
    %1 = ascscalar.get_block_num
    %2 = cannir.extract_layout %arg0 : !cannir.tensor<<gm, bf16>, <<(-1, 1024)>, <(1024, 1)>>, <<(-1, 1024)>, <(1024, 1)>>, identity>
    %3 = cannir.extract_shape %2 : <<(-1, 1024)>, <(1024, 1)>> -> <(-1, 1024)>
    %4 = cannir.extract_dim %3[0] : !cannir.shape<(-1, 1024)>
    %5 = arith.cmpi ne, %4, %c8_i64 : i64
    scf.if %5 {
      asccube.sync_all
      asccube.sync_block_wait s, %c13_i64
      asccube.sync_block_arrive fixpipe, %c14_i64 {mode = 0 : i32}
      asccube.sync_block_wait s, %c14_i64 {mode = 0 : i32}
      asccube.sync_block_arrive mte3, %c15_i64
    }
    %6:3 = scf.for %arg25 = %0 to %c20_i64 step %1 iter_args(%arg26 = %c0_i64, %arg27 = %c0_i64, %arg28 = %c0_i64) -> (i64, i64, i64)  : i64 {
      %10 = arith.remsi %arg25, %c20_i64 : i64
      %11 = arith.cmpi ne, %10, %c0_i64 : i64
      %12 = arith.cmpi slt, %10, %c0_i64 : i64
      %13 = arith.andi %11, %12 : i1
      %14 = arith.divsi %arg25, %c20_i64 : i64
      %15 = arith.extui %13 : i1 to i64
      %16 = arith.subi %14, %15 : i64
      %17 = arith.select %13, %c20_i64, %c0_i64 : i64
      %18 = arith.addi %10, %17 overflow<nsw> : i64
      %19 = arith.cmpi eq, %arg28, %c0_i64 : i64
      %20 = ascscalar.memref_address %alloc_46 : memref<512xf32, 5>
      %21 = ascscalar.memref_address %alloc_45 : memref<512xf32, 5>
      %22 = arith.select %19, %20, %21 : i64
      %23 = arith.select %19, %c8_i64, %c9_i64 : i64
      %24 = arith.addi %arg28, %c1_i64 overflow<nsw> : i64
      %25 = arith.remui %24, %c2_i64 : i64
      %26:2 = scf.for %arg29 = %c0_i64 to %c2_i64 step %c1_i64 iter_args(%arg30 = %arg26, %arg31 = %arg27) -> (i64, i64)  : i64 {
        %33 = arith.cmpi eq, %arg30, %c0_i64 : i64
        %34 = ascscalar.memref_address %alloc_44 : memref<8192xbf16, 2>
        %35 = ascscalar.memref_address %alloc_43 : memref<8192xbf16, 2>
        %36 = arith.select %33, %34, %35 : i64
        %37 = arith.cmpi ne, %arg30, %c0_i64 : i64
        %38 = arith.extui %37 : i1 to i64
        %39 = arith.addi %arg30, %c1_i64 overflow<nsw> : i64
        %40 = arith.remui %39, %c2_i64 : i64
        %41 = ascscalar.bind_local_ptr %36 : memref<1xbf16, 2>
        %42 = arith.muli %16, %c512_i64 : i64
        %43 = arith.muli %arg29, %c32_i64 overflow<nsw> : i64
        %44 = arith.addi %42, %43 : i64
        %45 = arith.muli %44, %c16_i64 : i64
        %46 = arith.minsi %16, %c1_i64 : i64
        %47 = arith.muli %46, %c8_i64 : i64
        %48 = arith.subi %c8_i64, %47 : i64
        %49 = arith.minsi %48, %c8_i64 : i64
        %50 = arith.cmpi sle, %16, %c1_i64 : i64
        %51 = arith.select %50, %49, %c0_i64 : i64
        asccube.get_buf mte2, %38, 0 {cannir.buffer.generated_sync}
        %52 = arith.cmpi ugt, %51, %c0_i64 : i64
        scf.if %52 {
          %109 = arith.index_cast %45 : i64 to index
          %110 = arith.trunci %51 : i64 to i32
          asccube.copy_gm2l1 %41[%c0], %alloc_3[%109], %110, %c512_i32, %c2048_i64, %c1_i32, %c0_i64, %c0_i64, %c16_i32 l2_cache_ctl %c1_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<8192xbf16>
        }
        asccube.release_buf mte2, %38, 0 {cannir.buffer.generated_sync}
        %53 = ascscalar.memref_address %alloc_42 : memref<16384xbf16, 2>
        %54 = ascscalar.memref_address %alloc_41 : memref<16384xbf16, 2>
        %55 = arith.select %33, %53, %54 : i64
        %56 = arith.select %33, %c2_i64, %c3_i64 : i64
        %57 = ascscalar.bind_local_ptr %55 : memref<1xbf16, 2>
        %58 = arith.muli %18, %c32_i64 overflow<nsw> : i64
        %59 = arith.index_cast %58 : i64 to index
        %60 = arith.muli %arg29, %c512_i64 overflow<nsw> : i64
        %61 = arith.index_cast %60 : i64 to index
        %62 = arith.divui %61, %c16 : index
        %63 = arith.muli %62, %c10240 : index
        %64 = arith.divui %59, %c16 : index
        %65 = arith.muli %64, %c256 : index
        %66 = arith.addi %63, %65 : index
        asccube.get_buf mte2, %56, 0 {cannir.buffer.generated_sync}
        asccube.copy_gm2l1_burst %57[%c0], %alloc[%66], %c32_i32, %c1024_i32, %c20480_i64, %c1024_i32 l2_cache_ctl %c0_i32 pad_value %c0_i32 : i32 left_padding %c0_i32 right_padding %c0_i32 : memref<1xbf16, 2>, memref<655360xbf16>
        asccube.release_buf mte2, %56, 0 {cannir.buffer.generated_sync}
        %67 = arith.cmpi eq, %arg31, %c0_i64 : i64
        %68 = ascscalar.memref_address %alloc_40 : memref<2048xbf16, 3>
        %69 = ascscalar.memref_address %alloc_39 : memref<2048xbf16, 3>
        %70 = arith.select %67, %68, %69 : i64
        %71 = arith.select %67, %c4_i64, %c5_i64 : i64
        %72 = arith.addi %arg31, %c1_i64 overflow<nsw> : i64
        %73 = arith.remui %72, %c2_i64 : i64
        %74 = ascscalar.bind_local_ptr %70 : memref<1xbf16, 3>
        asccube.get_buf mte1, %71, 0 {cannir.buffer.generated_sync}
        asccube.get_buf mte1, %38, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0a %74[%c0], %41[%c0], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c1_i32, %c1_i32 : memref<1xbf16, 3>, memref<1xbf16, 2>
        asccube.release_buf mte1, %38, 0 {cannir.buffer.generated_sync}
        asccube.release_buf mte1, %71, 0 {cannir.buffer.generated_sync}
        %75 = ascscalar.memref_address %alloc_38 : memref<4096xbf16, 4>
        %76 = ascscalar.memref_address %alloc_37 : memref<4096xbf16, 4>
        %77 = arith.select %67, %75, %76 : i64
        %78 = arith.select %67, %c6_i64, %c7_i64 : i64
        %79 = ascscalar.bind_local_ptr %77 : memref<1xbf16, 4>
        asccube.get_buf mte1, %78, 0 {cannir.buffer.generated_sync}
        asccube.get_buf mte1, %56, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0b %79[%c0], %57[%c0], %c0_i32, %c0_i32, %c2_i32, %c8_i32, %c2_i32, %c2_i32 : memref<1xbf16, 4>, memref<1xbf16, 2>
        asccube.release_buf mte1, %56, 0 {cannir.buffer.generated_sync}
        asccube.release_buf mte1, %78, 0 {cannir.buffer.generated_sync}
        %80 = arith.cmpi eq, %arg29, %c0_i64 : i64
        %81 = ascscalar.bind_local_ptr %22 : memref<1xf32, 5>
        asccube.get_buf m, %23, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %71, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %78, 0 {cannir.buffer.generated_sync}
        scf.if %80 {
          asccube.mmad %81[%c0], %74, %79, %c8_i32, %c128_i32, %c32_i32, %c0_i32 {c_matrix_init_val = true} : memref<1xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
        } else {
          asccube.mmad %81[%c0], %74, %79, %c8_i32, %c128_i32, %c32_i32, %c0_i32 {c_matrix_init_val = false} : memref<1xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
        }
        asccube.release_buf m, %78, 0 {cannir.buffer.generated_sync}
        asccube.release_buf m, %71, 0 {cannir.buffer.generated_sync}
        %82 = arith.cmpi eq, %73, %c0_i64 : i64
        %83 = arith.select %82, %68, %69 : i64
        %84 = arith.select %82, %c4_i64, %c5_i64 : i64
        %85 = arith.addi %73, %c1_i64 overflow<nsw> : i64
        %86 = arith.remui %85, %c2_i64 : i64
        %87 = ascscalar.bind_local_ptr %83 : memref<1xbf16, 3>
        asccube.get_buf mte1, %84, 0 {cannir.buffer.generated_sync}
        asccube.get_buf mte1, %38, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0a %87[%c0], %41[%c0], %c0_i32, %c8_i32, %c1_i32, %c8_i32, %c1_i32, %c1_i32 : memref<1xbf16, 3>, memref<1xbf16, 2>
        asccube.release_buf mte1, %84, 0 {cannir.buffer.generated_sync}
        %88 = arith.select %82, %75, %76 : i64
        %89 = arith.select %82, %c6_i64, %c7_i64 : i64
        %90 = ascscalar.bind_local_ptr %88 : memref<1xbf16, 4>
        asccube.get_buf mte1, %89, 0 {cannir.buffer.generated_sync}
        asccube.get_buf mte1, %56, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0b %90[%c0], %57[%c0], %c0_i32, %c8_i32, %c2_i32, %c8_i32, %c2_i32, %c2_i32 : memref<1xbf16, 4>, memref<1xbf16, 2>
        asccube.release_buf mte1, %89, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %84, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %89, 0 {cannir.buffer.generated_sync}
        asccube.mmad %81[%c0], %87, %90, %c8_i32, %c128_i32, %c32_i32, %c0_i32 {c_matrix_init_val = false} : memref<1xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
        asccube.release_buf m, %89, 0 {cannir.buffer.generated_sync}
        asccube.release_buf m, %84, 0 {cannir.buffer.generated_sync}
        %91 = arith.cmpi eq, %86, %c0_i64 : i64
        %92 = arith.select %91, %68, %69 : i64
        %93 = arith.select %91, %c4_i64, %c5_i64 : i64
        %94 = arith.addi %86, %c1_i64 overflow<nsw> : i64
        %95 = arith.remui %94, %c2_i64 : i64
        %96 = ascscalar.bind_local_ptr %92 : memref<1xbf16, 3>
        asccube.get_buf mte1, %93, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0a %96[%c0], %41[%c0], %c0_i32, %c16_i32, %c1_i32, %c8_i32, %c1_i32, %c1_i32 : memref<1xbf16, 3>, memref<1xbf16, 2>
        asccube.release_buf mte1, %93, 0 {cannir.buffer.generated_sync}
        %97 = arith.select %91, %75, %76 : i64
        %98 = arith.select %91, %c6_i64, %c7_i64 : i64
        %99 = ascscalar.bind_local_ptr %97 : memref<1xbf16, 4>
        asccube.get_buf mte1, %98, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0b %99[%c0], %57[%c0], %c0_i32, %c16_i32, %c2_i32, %c8_i32, %c2_i32, %c2_i32 : memref<1xbf16, 4>, memref<1xbf16, 2>
        asccube.release_buf mte1, %98, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %93, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %98, 0 {cannir.buffer.generated_sync}
        asccube.mmad %81[%c0], %96, %99, %c8_i32, %c128_i32, %c32_i32, %c0_i32 {c_matrix_init_val = false} : memref<1xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
        asccube.release_buf m, %98, 0 {cannir.buffer.generated_sync}
        asccube.release_buf m, %93, 0 {cannir.buffer.generated_sync}
        %100 = arith.cmpi eq, %95, %c0_i64 : i64
        %101 = arith.select %100, %68, %69 : i64
        %102 = arith.select %100, %c4_i64, %c5_i64 : i64
        %103 = arith.addi %95, %c1_i64 overflow<nsw> : i64
        %104 = arith.remui %103, %c2_i64 : i64
        %105 = ascscalar.bind_local_ptr %101 : memref<1xbf16, 3>
        asccube.get_buf mte1, %102, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0a %105[%c0], %41[%c0], %c0_i32, %c24_i32, %c1_i32, %c8_i32, %c1_i32, %c1_i32 : memref<1xbf16, 3>, memref<1xbf16, 2>
        asccube.release_buf mte1, %38, 0 {cannir.buffer.generated_sync}
        asccube.release_buf mte1, %102, 0 {cannir.buffer.generated_sync}
        %106 = arith.select %100, %75, %76 : i64
        %107 = arith.select %100, %c6_i64, %c7_i64 : i64
        %108 = ascscalar.bind_local_ptr %106 : memref<1xbf16, 4>
        asccube.get_buf mte1, %107, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0b %108[%c0], %57[%c0], %c0_i32, %c24_i32, %c2_i32, %c8_i32, %c2_i32, %c2_i32 : memref<1xbf16, 4>, memref<1xbf16, 2>
        asccube.release_buf mte1, %56, 0 {cannir.buffer.generated_sync}
        asccube.release_buf mte1, %107, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %102, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %107, 0 {cannir.buffer.generated_sync}
        asccube.mmad %81[%c0], %105, %108, %c8_i32, %c128_i32, %c32_i32, %c0_i32 {c_matrix_init_val = false} : memref<1xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
        asccube.release_buf m, %107, 0 {cannir.buffer.generated_sync}
        asccube.release_buf m, %102, 0 {cannir.buffer.generated_sync}
        asccube.release_buf m, %23, 0 {cannir.buffer.generated_sync}
        scf.yield %40, %104 : i64, i64
      }
      %27 = ascscalar.bind_local_ptr %22 : memref<1xf32, 5>
      %28 = arith.muli %16, %c336_i64 : i64
      %29 = arith.muli %18, %c2_i64 overflow<nsw> : i64
      %30 = arith.addi %28, %29 : i64
      %31 = arith.muli %30, %c16_i64 : i64
      asccube.get_buf fixpipe, %23, 0 {cannir.buffer.generated_sync}
      %32 = arith.index_cast %31 : i64 to index
      asccube.copy_l0c2gm %alloc_4[%32], %27[%c0], %c32_i32, %c8_i32, %c672_i32, %c16_i32, %c0_i32 l2_cache_ctl %c0_i32 {nz2nd_en = true, relu_en = false} : memref<5376xbf16>, memref<1xf32, 5>
      asccube.release_buf fixpipe, %23, 0 {cannir.buffer.generated_sync}
      scf.yield %26#0, %26#1, %25 : i64, i64, i64
    }
    asccube.sync_all
    asccube.sync_block_wait s, %c13_i64
    asccube.sync_block_arrive fixpipe, %c14_i64 {mode = 0 : i32}
    asccube.sync_block_wait s, %c14_i64 {mode = 0 : i32}
    asccube.sync_block_arrive mte3, %c15_i64
    %7:3 = scf.for %arg25 = %0 to %c18_i64 step %1 iter_args(%arg26 = %c0_i64, %arg27 = %c0_i64, %arg28 = %c0_i64) -> (i64, i64, i64)  : i64 {
      %10 = arith.remsi %arg25, %c18_i64 : i64
      %11 = arith.cmpi ne, %10, %c0_i64 : i64
      %12 = arith.cmpi slt, %10, %c0_i64 : i64
      %13 = arith.andi %11, %12 : i1
      %14 = arith.divsi %arg25, %c18_i64 : i64
      %15 = arith.extui %13 : i1 to i64
      %16 = arith.subi %14, %15 : i64
      %17 = arith.select %13, %c18_i64, %c0_i64 : i64
      %18 = arith.addi %10, %17 overflow<nsw> : i64
      %19 = arith.cmpi eq, %arg28, %c0_i64 : i64
      %20 = ascscalar.memref_address %alloc_36 : memref<512xf32, 5>
      %21 = ascscalar.memref_address %alloc_35 : memref<512xf32, 5>
      %22 = arith.select %19, %20, %21 : i64
      %23 = arith.select %19, %c8_i64, %c9_i64 : i64
      %24 = arith.addi %arg28, %c1_i64 overflow<nsw> : i64
      %25 = arith.remui %24, %c2_i64 : i64
      %26:2 = scf.for %arg29 = %c0_i64 to %c2_i64 step %c1_i64 iter_args(%arg30 = %arg26, %arg31 = %arg27) -> (i64, i64)  : i64 {
        %33 = arith.cmpi eq, %arg30, %c0_i64 : i64
        %34 = ascscalar.memref_address %alloc_34 : memref<8192xbf16, 2>
        %35 = ascscalar.memref_address %alloc_33 : memref<8192xbf16, 2>
        %36 = arith.select %33, %34, %35 : i64
        %37 = arith.cmpi ne, %arg30, %c0_i64 : i64
        %38 = arith.extui %37 : i1 to i64
        %39 = arith.addi %arg30, %c1_i64 overflow<nsw> : i64
        %40 = arith.remui %39, %c2_i64 : i64
        %41 = ascscalar.bind_local_ptr %36 : memref<1xbf16, 2>
        %42 = arith.muli %16, %c512_i64 : i64
        %43 = arith.muli %arg29, %c32_i64 overflow<nsw> : i64
        %44 = arith.addi %42, %43 : i64
        %45 = arith.muli %44, %c16_i64 : i64
        %46 = arith.minsi %16, %c1_i64 : i64
        %47 = arith.muli %46, %c8_i64 : i64
        %48 = arith.subi %c8_i64, %47 : i64
        %49 = arith.minsi %48, %c8_i64 : i64
        %50 = arith.cmpi sle, %16, %c1_i64 : i64
        %51 = arith.select %50, %49, %c0_i64 : i64
        asccube.get_buf mte2, %38, 0 {cannir.buffer.generated_sync}
        %52 = arith.cmpi ugt, %51, %c0_i64 : i64
        scf.if %52 {
          %109 = arith.index_cast %45 : i64 to index
          %110 = arith.trunci %51 : i64 to i32
          asccube.copy_gm2l1 %41[%c0], %alloc_3[%109], %110, %c512_i32, %c2048_i64, %c1_i32, %c0_i64, %c0_i64, %c16_i32 l2_cache_ctl %c1_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<8192xbf16>
        }
        asccube.release_buf mte2, %38, 0 {cannir.buffer.generated_sync}
        %53 = ascscalar.memref_address %alloc_32 : memref<16384xbf16, 2>
        %54 = ascscalar.memref_address %alloc_31 : memref<16384xbf16, 2>
        %55 = arith.select %33, %53, %54 : i64
        %56 = arith.select %33, %c2_i64, %c3_i64 : i64
        %57 = ascscalar.bind_local_ptr %55 : memref<1xbf16, 2>
        %58 = arith.muli %18, %c32_i64 overflow<nsw> : i64
        %59 = arith.index_cast %58 : i64 to index
        %60 = arith.muli %arg29, %c512_i64 overflow<nsw> : i64
        %61 = arith.index_cast %60 : i64 to index
        %62 = arith.divui %61, %c16 : index
        %63 = arith.muli %62, %c9216 : index
        %64 = arith.divui %59, %c16 : index
        %65 = arith.muli %64, %c256 : index
        %66 = arith.addi %63, %65 : index
        asccube.get_buf mte2, %56, 0 {cannir.buffer.generated_sync}
        asccube.copy_gm2l1_burst %57[%c0], %alloc_1[%66], %c32_i32, %c1024_i32, %c18432_i64, %c1024_i32 l2_cache_ctl %c0_i32 pad_value %c0_i32 : i32 left_padding %c0_i32 right_padding %c0_i32 : memref<1xbf16, 2>, memref<589824xbf16>
        asccube.release_buf mte2, %56, 0 {cannir.buffer.generated_sync}
        %67 = arith.cmpi eq, %arg31, %c0_i64 : i64
        %68 = ascscalar.memref_address %alloc_30 : memref<2048xbf16, 3>
        %69 = ascscalar.memref_address %alloc_29 : memref<2048xbf16, 3>
        %70 = arith.select %67, %68, %69 : i64
        %71 = arith.select %67, %c4_i64, %c5_i64 : i64
        %72 = arith.addi %arg31, %c1_i64 overflow<nsw> : i64
        %73 = arith.remui %72, %c2_i64 : i64
        %74 = ascscalar.bind_local_ptr %70 : memref<1xbf16, 3>
        asccube.get_buf mte1, %71, 0 {cannir.buffer.generated_sync}
        asccube.get_buf mte1, %38, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0a %74[%c0], %41[%c0], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c1_i32, %c1_i32 : memref<1xbf16, 3>, memref<1xbf16, 2>
        asccube.release_buf mte1, %38, 0 {cannir.buffer.generated_sync}
        asccube.release_buf mte1, %71, 0 {cannir.buffer.generated_sync}
        %75 = ascscalar.memref_address %alloc_28 : memref<4096xbf16, 4>
        %76 = ascscalar.memref_address %alloc_27 : memref<4096xbf16, 4>
        %77 = arith.select %67, %75, %76 : i64
        %78 = arith.select %67, %c6_i64, %c7_i64 : i64
        %79 = ascscalar.bind_local_ptr %77 : memref<1xbf16, 4>
        asccube.get_buf mte1, %78, 0 {cannir.buffer.generated_sync}
        asccube.get_buf mte1, %56, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0b %79[%c0], %57[%c0], %c0_i32, %c0_i32, %c2_i32, %c8_i32, %c2_i32, %c2_i32 : memref<1xbf16, 4>, memref<1xbf16, 2>
        asccube.release_buf mte1, %56, 0 {cannir.buffer.generated_sync}
        asccube.release_buf mte1, %78, 0 {cannir.buffer.generated_sync}
        %80 = arith.cmpi eq, %arg29, %c0_i64 : i64
        %81 = ascscalar.bind_local_ptr %22 : memref<1xf32, 5>
        asccube.get_buf m, %23, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %71, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %78, 0 {cannir.buffer.generated_sync}
        scf.if %80 {
          asccube.mmad %81[%c0], %74, %79, %c8_i32, %c128_i32, %c32_i32, %c0_i32 {c_matrix_init_val = true} : memref<1xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
        } else {
          asccube.mmad %81[%c0], %74, %79, %c8_i32, %c128_i32, %c32_i32, %c0_i32 {c_matrix_init_val = false} : memref<1xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
        }
        asccube.release_buf m, %78, 0 {cannir.buffer.generated_sync}
        asccube.release_buf m, %71, 0 {cannir.buffer.generated_sync}
        %82 = arith.cmpi eq, %73, %c0_i64 : i64
        %83 = arith.select %82, %68, %69 : i64
        %84 = arith.select %82, %c4_i64, %c5_i64 : i64
        %85 = arith.addi %73, %c1_i64 overflow<nsw> : i64
        %86 = arith.remui %85, %c2_i64 : i64
        %87 = ascscalar.bind_local_ptr %83 : memref<1xbf16, 3>
        asccube.get_buf mte1, %84, 0 {cannir.buffer.generated_sync}
        asccube.get_buf mte1, %38, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0a %87[%c0], %41[%c0], %c0_i32, %c8_i32, %c1_i32, %c8_i32, %c1_i32, %c1_i32 : memref<1xbf16, 3>, memref<1xbf16, 2>
        asccube.release_buf mte1, %84, 0 {cannir.buffer.generated_sync}
        %88 = arith.select %82, %75, %76 : i64
        %89 = arith.select %82, %c6_i64, %c7_i64 : i64
        %90 = ascscalar.bind_local_ptr %88 : memref<1xbf16, 4>
        asccube.get_buf mte1, %89, 0 {cannir.buffer.generated_sync}
        asccube.get_buf mte1, %56, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0b %90[%c0], %57[%c0], %c0_i32, %c8_i32, %c2_i32, %c8_i32, %c2_i32, %c2_i32 : memref<1xbf16, 4>, memref<1xbf16, 2>
        asccube.release_buf mte1, %89, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %84, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %89, 0 {cannir.buffer.generated_sync}
        asccube.mmad %81[%c0], %87, %90, %c8_i32, %c128_i32, %c32_i32, %c0_i32 {c_matrix_init_val = false} : memref<1xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
        asccube.release_buf m, %89, 0 {cannir.buffer.generated_sync}
        asccube.release_buf m, %84, 0 {cannir.buffer.generated_sync}
        %91 = arith.cmpi eq, %86, %c0_i64 : i64
        %92 = arith.select %91, %68, %69 : i64
        %93 = arith.select %91, %c4_i64, %c5_i64 : i64
        %94 = arith.addi %86, %c1_i64 overflow<nsw> : i64
        %95 = arith.remui %94, %c2_i64 : i64
        %96 = ascscalar.bind_local_ptr %92 : memref<1xbf16, 3>
        asccube.get_buf mte1, %93, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0a %96[%c0], %41[%c0], %c0_i32, %c16_i32, %c1_i32, %c8_i32, %c1_i32, %c1_i32 : memref<1xbf16, 3>, memref<1xbf16, 2>
        asccube.release_buf mte1, %93, 0 {cannir.buffer.generated_sync}
        %97 = arith.select %91, %75, %76 : i64
        %98 = arith.select %91, %c6_i64, %c7_i64 : i64
        %99 = ascscalar.bind_local_ptr %97 : memref<1xbf16, 4>
        asccube.get_buf mte1, %98, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0b %99[%c0], %57[%c0], %c0_i32, %c16_i32, %c2_i32, %c8_i32, %c2_i32, %c2_i32 : memref<1xbf16, 4>, memref<1xbf16, 2>
        asccube.release_buf mte1, %98, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %93, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %98, 0 {cannir.buffer.generated_sync}
        asccube.mmad %81[%c0], %96, %99, %c8_i32, %c128_i32, %c32_i32, %c0_i32 {c_matrix_init_val = false} : memref<1xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
        asccube.release_buf m, %98, 0 {cannir.buffer.generated_sync}
        asccube.release_buf m, %93, 0 {cannir.buffer.generated_sync}
        %100 = arith.cmpi eq, %95, %c0_i64 : i64
        %101 = arith.select %100, %68, %69 : i64
        %102 = arith.select %100, %c4_i64, %c5_i64 : i64
        %103 = arith.addi %95, %c1_i64 overflow<nsw> : i64
        %104 = arith.remui %103, %c2_i64 : i64
        %105 = ascscalar.bind_local_ptr %101 : memref<1xbf16, 3>
        asccube.get_buf mte1, %102, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0a %105[%c0], %41[%c0], %c0_i32, %c24_i32, %c1_i32, %c8_i32, %c1_i32, %c1_i32 : memref<1xbf16, 3>, memref<1xbf16, 2>
        asccube.release_buf mte1, %38, 0 {cannir.buffer.generated_sync}
        asccube.release_buf mte1, %102, 0 {cannir.buffer.generated_sync}
        %106 = arith.select %100, %75, %76 : i64
        %107 = arith.select %100, %c6_i64, %c7_i64 : i64
        %108 = ascscalar.bind_local_ptr %106 : memref<1xbf16, 4>
        asccube.get_buf mte1, %107, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0b %108[%c0], %57[%c0], %c0_i32, %c24_i32, %c2_i32, %c8_i32, %c2_i32, %c2_i32 : memref<1xbf16, 4>, memref<1xbf16, 2>
        asccube.release_buf mte1, %56, 0 {cannir.buffer.generated_sync}
        asccube.release_buf mte1, %107, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %102, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %107, 0 {cannir.buffer.generated_sync}
        asccube.mmad %81[%c0], %105, %108, %c8_i32, %c128_i32, %c32_i32, %c0_i32 {c_matrix_init_val = false} : memref<1xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
        asccube.release_buf m, %107, 0 {cannir.buffer.generated_sync}
        asccube.release_buf m, %102, 0 {cannir.buffer.generated_sync}
        asccube.release_buf m, %23, 0 {cannir.buffer.generated_sync}
        scf.yield %40, %104 : i64, i64
      }
      %27 = ascscalar.bind_local_ptr %22 : memref<1xf32, 5>
      %28 = arith.muli %16, %c288_i64 : i64
      %29 = arith.muli %18, %c2_i64 overflow<nsw> : i64
      %30 = arith.addi %28, %29 : i64
      %31 = arith.muli %30, %c16_i64 : i64
      asccube.get_buf fixpipe, %23, 0 {cannir.buffer.generated_sync}
      %32 = arith.index_cast %31 : i64 to index
      asccube.copy_l0c2gm %alloc_5[%32], %27[%c0], %c32_i32, %c8_i32, %c576_i32, %c16_i32, %c0_i32 l2_cache_ctl %c0_i32 {nz2nd_en = true, relu_en = false} : memref<4608xbf16>, memref<1xf32, 5>
      asccube.release_buf fixpipe, %23, 0 {cannir.buffer.generated_sync}
      scf.yield %26#0, %26#1, %25 : i64, i64, i64
    }
    asccube.sync_all
    asccube.sync_block_wait s, %c13_i64
    asccube.sync_block_arrive fixpipe, %c14_i64 {mode = 0 : i32}
    asccube.sync_block_wait s, %c14_i64 {mode = 0 : i32}
    asccube.sync_block_arrive mte3, %c15_i64
    %8:2 = scf.for %arg25 = %0 to %c4_i64 step %1 iter_args(%arg26 = %c0_i64, %arg27 = %c0_i64) -> (i64, i64)  : i64 {
      %10 = arith.cmpi eq, %arg27, %c0_i64 : i64
      %11 = ascscalar.memref_address %alloc_26 : memref<1536xf32, 5>
      %12 = ascscalar.memref_address %alloc_25 : memref<1536xf32, 5>
      %13 = arith.select %10, %11, %12 : i64
      %14 = arith.select %10, %c8_i64, %c9_i64 : i64
      %15 = arith.addi %arg27, %c1_i64 overflow<nsw> : i64
      %16 = arith.remui %15, %c2_i64 : i64
      %17 = scf.for %arg28 = %c0_i64 to %c5_i64 step %c1_i64 iter_args(%arg29 = %arg26) -> (i64)  : i64 {
        %21 = arith.cmpi eq, %arg29, %c0_i64 : i64
        %22 = ascscalar.memref_address %alloc_24 : memref<2048xbf16, 2>
        %23 = ascscalar.memref_address %alloc_23 : memref<2048xbf16, 2>
        %24 = arith.select %21, %22, %23 : i64
        %25 = arith.cmpi ne, %arg29, %c0_i64 : i64
        %26 = arith.extui %25 : i1 to i64
        %27 = arith.addi %arg29, %c1_i64 overflow<nsw> : i64
        %28 = arith.remui %27, %c2_i64 : i64
        %29 = ascscalar.bind_local_ptr %24 : memref<1xbf16, 2>
        %30 = arith.muli %arg28, %c128_i64 overflow<nsw> : i64
        asccube.get_buf mte2, %26, 0 {cannir.buffer.generated_sync}
        %31 = arith.index_cast %30 : i64 to index
        asccube.copy_gm2l1 %29[%c0], %alloc_8[%31], %c8_i32, %c128_i32, %c1280_i64, %c1_i32, %c0_i64, %c0_i64, %c16_i32 l2_cache_ctl %c1_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<5120xbf16>
        asccube.release_buf mte2, %26, 0 {cannir.buffer.generated_sync}
        %32 = ascscalar.memref_address %alloc_22 : memref<12288xbf16, 2>
        %33 = ascscalar.memref_address %alloc_21 : memref<12288xbf16, 2>
        %34 = arith.select %21, %32, %33 : i64
        %35 = arith.select %21, %c2_i64, %c3_i64 : i64
        %36 = ascscalar.bind_local_ptr %34 : memref<1xbf16, 2>
        %37 = arith.muli %arg25, %c96_i64 : i64
        %38 = arith.index_cast %37 : i64 to index
        %39 = arith.minsi %arg25, %c4_i64 : i64
        %40 = arith.muli %39, %c96_i64 : i64
        %41 = arith.subi %c384_i64, %40 : i64
        %42 = arith.minsi %41, %c96_i64 : i64
        %43 = arith.cmpi sle, %arg25, %c4_i64 : i64
        %44 = arith.select %43, %42, %c0_i64 : i64
        %45 = arith.index_cast %44 : i64 to index
        %46 = arith.divui %31, %c16 : index
        %47 = arith.muli %46, %c6144 : index
        %48 = arith.divui %38, %c16 : index
        %49 = arith.muli %48, %c256 : index
        %50 = arith.addi %47, %49 : index
        %51 = arith.addi %45, %c15 : index
        %52 = arith.divui %51, %c16 : index
        %53 = arith.muli %52, %c512 : index
        %54 = arith.index_cast %53 : index to i64
        asccube.get_buf mte2, %35, 0 {cannir.buffer.generated_sync}
        %55 = arith.cmpi ugt, %54, %c0_i64 : i64
        scf.if %55 {
          %68 = arith.trunci %54 : i64 to i32
          asccube.copy_gm2l1_burst %36[%c0], %alloc_0[%50], %c8_i32, %68, %c12288_i64, %c3072_i32 l2_cache_ctl %c0_i32 pad_value %c0_i32 : i32 left_padding %c0_i32 right_padding %c0_i32 : memref<1xbf16, 2>, memref<245760xbf16>
        }
        asccube.release_buf mte2, %35, 0 {cannir.buffer.generated_sync}
        %56 = ascscalar.memref_address %alloc_20 : memref<2048xbf16, 3>
        %57 = ascscalar.memref_address %alloc_19 : memref<2048xbf16, 3>
        %58 = arith.select %21, %56, %57 : i64
        %59 = arith.select %21, %c4_i64, %c5_i64 : i64
        %60 = ascscalar.bind_local_ptr %58 : memref<1xbf16, 3>
        asccube.get_buf mte1, %59, 0 {cannir.buffer.generated_sync}
        asccube.get_buf mte1, %26, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0a %60[%c0], %29[%c0], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c1_i32, %c1_i32 : memref<1xbf16, 3>, memref<1xbf16, 2>
        asccube.release_buf mte1, %26, 0 {cannir.buffer.generated_sync}
        asccube.release_buf mte1, %59, 0 {cannir.buffer.generated_sync}
        %61 = ascscalar.memref_address %alloc_18 : memref<12288xbf16, 4>
        %62 = ascscalar.memref_address %alloc_17 : memref<12288xbf16, 4>
        %63 = arith.select %21, %61, %62 : i64
        %64 = arith.select %21, %c6_i64, %c7_i64 : i64
        %65 = ascscalar.bind_local_ptr %63 : memref<1xbf16, 4>
        asccube.get_buf mte1, %64, 0 {cannir.buffer.generated_sync}
        asccube.get_buf mte1, %35, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0b %65[%c0], %36[%c0], %c0_i32, %c0_i32, %c6_i32, %c8_i32, %c6_i32, %c6_i32 : memref<1xbf16, 4>, memref<1xbf16, 2>
        asccube.release_buf mte1, %35, 0 {cannir.buffer.generated_sync}
        asccube.release_buf mte1, %64, 0 {cannir.buffer.generated_sync}
        %66 = arith.cmpi eq, %arg28, %c0_i64 : i64
        %67 = ascscalar.bind_local_ptr %13 : memref<1xf32, 5>
        asccube.get_buf m, %14, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %59, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %64, 0 {cannir.buffer.generated_sync}
        scf.if %66 {
          asccube.mmad %67[%c0], %60, %65, %c8_i32, %c128_i32, %c96_i32, %c0_i32 {c_matrix_init_val = true} : memref<1xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
        } else {
          asccube.mmad %67[%c0], %60, %65, %c8_i32, %c128_i32, %c96_i32, %c0_i32 {c_matrix_init_val = false} : memref<1xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
        }
        asccube.release_buf m, %64, 0 {cannir.buffer.generated_sync}
        asccube.release_buf m, %59, 0 {cannir.buffer.generated_sync}
        asccube.release_buf m, %14, 0 {cannir.buffer.generated_sync}
        scf.yield %28 : i64
      }
      %18 = ascscalar.bind_local_ptr %13 : memref<1xf32, 5>
      %19 = arith.muli %arg25, %c96_i64 : i64
      asccube.get_buf fixpipe, %14, 0 {cannir.buffer.generated_sync}
      %20 = arith.index_cast %19 : i64 to index
      asccube.copy_l0c2gm %alloc_6[%20], %18[%c0], %c96_i32, %c8_i32, %c384_i32, %c16_i32, %c0_i32 l2_cache_ctl %c0_i32 {nz2nd_en = true, relu_en = false} : memref<3072xbf16>, memref<1xf32, 5>
      asccube.release_buf fixpipe, %14, 0 {cannir.buffer.generated_sync}
      scf.yield %17, %16 : i64, i64
    }
    asccube.sync_all
    asccube.sync_block_wait s, %c13_i64
    asccube.sync_block_arrive fixpipe, %c14_i64 {mode = 0 : i32}
    asccube.sync_block_wait s, %c14_i64 {mode = 0 : i32}
    asccube.sync_block_arrive mte3, %c15_i64
    %9:2 = scf.for %arg25 = %0 to %c8_i64 step %1 iter_args(%arg26 = %c0_i64, %arg27 = %c0_i64) -> (i64, i64)  : i64 {
      %10 = arith.remsi %arg25, %c4_i64 : i64
      %11 = arith.cmpi ne, %10, %c0_i64 : i64
      %12 = arith.cmpi slt, %10, %c0_i64 : i64
      %13 = arith.andi %11, %12 : i1
      %14 = arith.select %13, %c4_i64, %c0_i64 : i64
      %15 = arith.addi %10, %14 overflow<nsw> : i64
      %16 = arith.divsi %arg25, %c4_i64 : i64
      %17 = arith.extui %13 : i1 to i64
      %18 = arith.subi %16, %17 : i64
      %19 = arith.remsi %18, %c2_i64 : i64
      %20 = arith.cmpi ne, %19, %c0_i64 : i64
      %21 = arith.cmpi slt, %19, %c0_i64 : i64
      %22 = arith.andi %20, %21 : i1
      %23 = arith.select %22, %c2_i64, %c0_i64 : i64
      %24 = arith.addi %19, %23 overflow<nsw> : i64
      %25 = arith.remsi %arg25, %c8_i64 : i64
      %26 = arith.cmpi ne, %25, %c0_i64 : i64
      %27 = arith.cmpi slt, %25, %c0_i64 : i64
      %28 = arith.andi %26, %27 : i1
      %29 = arith.divsi %arg25, %c8_i64 : i64
      %30 = arith.extui %28 : i1 to i64
      %31 = arith.subi %29, %30 : i64
      %32 = arith.muli %24, %c192_i64 overflow<nsw> : i64
      %33 = arith.muli %24, %c576_i64 overflow<nsw> : i64
      %34 = arith.cmpi eq, %arg27, %c0_i64 : i64
      %35 = ascscalar.memref_address %alloc_16 : memref<2048xf32, 5>
      %36 = ascscalar.memref_address %alloc_15 : memref<2048xf32, 5>
      %37 = arith.select %34, %35, %36 : i64
      %38 = arith.select %34, %c6_i64, %c7_i64 : i64
      %39 = arith.addi %arg27, %c1_i64 overflow<nsw> : i64
      %40 = arith.remui %39, %c2_i64 : i64
      %41 = scf.for %arg28 = %c0_i64 to %c2_i64 step %c1_i64 iter_args(%arg29 = %arg26) -> (i64)  : i64 {
        %50 = arith.cmpi eq, %arg29, %c0_i64 : i64
        %51 = ascscalar.memref_address %alloc_14 : memref<1024xbf16, 2>
        %52 = ascscalar.memref_address %alloc_13 : memref<1024xbf16, 2>
        %53 = arith.select %50, %51, %52 : i64
        %54 = arith.cmpi ne, %arg29, %c0_i64 : i64
        %55 = arith.extui %54 : i1 to i64
        %56 = arith.addi %arg29, %c1_i64 overflow<nsw> : i64
        %57 = arith.remui %56, %c2_i64 : i64
        %58 = arith.muli %arg28, %c64_i64 overflow<nsw> : i64
        %59 = arith.addi %32, %58 overflow<nsw> : i64
        %60 = arith.addi %arg28, %c1_i64 overflow<nsw> : i64
        %61 = arith.muli %60, %c64_i64 overflow<nsw> : i64
        %62 = arith.addi %32, %61 overflow<nsw> : i64
        %63 = arith.subi %62, %59 overflow<nsw> : i64
        %64 = ascscalar.bind_local_ptr %53 : memref<1xbf16, 2>
        %65 = arith.muli %31, %c192_i64 : i64
        %66 = arith.divui %59, %c16_i64 : i64
        %67 = arith.addi %65, %66 : i64
        %68 = arith.muli %67, %c16_i64 : i64
        asccube.get_buf mte2, %55, 0 {cannir.buffer.generated_sync}
        %69 = arith.cmpi ugt, %63, %c0_i64 : i64
        scf.if %69 {
          %93 = arith.index_cast %68 : i64 to index
          %94 = arith.trunci %63 : i64 to i32
          asccube.copy_gm2l1 %64[%c0], %alloc_6[%93], %c8_i32, %94, %c768_i64, %c1_i32, %c0_i64, %c0_i64, %c16_i32 l2_cache_ctl %c1_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<3072xbf16>
        }
        asccube.release_buf mte2, %55, 0 {cannir.buffer.generated_sync}
        %70 = ascscalar.bitcast_memref %alloc_9 byte_offset = %c0_i64 : memref<1024xbf16, 3> -> memref<?xbf16, 3>
        asccube.get_buf mte1, %c4_i64, 0 {cannir.buffer.generated_sync}
        asccube.get_buf mte1, %55, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0a %70[%c0], %64[%c0], %c0_i32, %c0_i32, %c1_i32, %c4_i32, %c1_i32, %c1_i32 : memref<?xbf16, 3>, memref<1xbf16, 2>
        asccube.release_buf mte1, %55, 0 {cannir.buffer.generated_sync}
        asccube.release_buf mte1, %c4_i64, 0 {cannir.buffer.generated_sync}
        %71 = ascscalar.memref_address %alloc_12 : memref<8192xbf16, 2>
        %72 = ascscalar.memref_address %alloc_11 : memref<8192xbf16, 2>
        %73 = arith.select %50, %71, %72 : i64
        %74 = arith.select %50, %c2_i64, %c3_i64 : i64
        %75 = arith.muli %24, %c128_i64 overflow<nsw> : i64
        %76 = arith.addi %75, %58 overflow<nsw> : i64
        %77 = arith.remsi %76, %c64_i64 : i64
        %78 = arith.cmpi ne, %77, %c0_i64 : i64
        %79 = arith.cmpi slt, %77, %c0_i64 : i64
        %80 = arith.andi %78, %79 : i1
        %81 = arith.divsi %76, %c64_i64 : i64
        %82 = arith.extui %80 : i1 to i64
        %83 = arith.subi %81, %82 : i64
        %84 = ascscalar.bind_local_ptr %73 : memref<1xbf16, 2>
        %85 = arith.muli %83, %c2048_i64 : i64
        %86 = arith.muli %15, %c8_i64 overflow<nsw> : i64
        %87 = arith.addi %85, %86 : i64
        %88 = arith.muli %87, %c16_i64 : i64
        asccube.get_buf mte2, %74, 0 {cannir.buffer.generated_sync}
        %89 = arith.index_cast %88 : i64 to index
        asccube.copy_gm2l1 %84[%c0], %alloc_2[%89], %c64_i32, %c128_i32, %c1024_i64, %c1_i32, %c0_i64, %c0_i64, %c64_i32 l2_cache_ctl %c0_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<131072xbf16>
        asccube.release_buf mte2, %74, 0 {cannir.buffer.generated_sync}
        %90 = ascscalar.bitcast_memref %alloc_10 byte_offset = %c0_i64 : memref<8192xbf16, 4> -> memref<?xbf16, 4>
        asccube.get_buf mte1, %c5_i64, 0 {cannir.buffer.generated_sync}
        asccube.get_buf mte1, %74, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0b %90[%c0], %84[%c0], %c0_i32, %c0_i32, %c4_i32, %c8_i32, %c4_i32, %c8_i32 {transpose = true} : memref<?xbf16, 4>, memref<1xbf16, 2>
        asccube.release_buf mte1, %74, 0 {cannir.buffer.generated_sync}
        asccube.release_buf mte1, %c5_i64, 0 {cannir.buffer.generated_sync}
        %91 = arith.cmpi eq, %arg28, %c0_i64 : i64
        %92 = ascscalar.bind_local_ptr %37 : memref<1xf32, 5>
        asccube.get_buf m, %38, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %c4_i64, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %c5_i64, 0 {cannir.buffer.generated_sync}
        scf.if %91 {
          asccube.mmad %92[%c0], %70, %90, %c8_i32, %c64_i32, %c128_i32, %c0_i32 {c_matrix_init_val = true} : memref<1xf32, 5>, memref<?xbf16, 3>, memref<?xbf16, 4>
        } else {
          asccube.mmad %92[%c0], %70, %90, %c8_i32, %c64_i32, %c128_i32, %c0_i32 {c_matrix_init_val = false} : memref<1xf32, 5>, memref<?xbf16, 3>, memref<?xbf16, 4>
        }
        asccube.release_buf m, %c5_i64, 0 {cannir.buffer.generated_sync}
        asccube.release_buf m, %c4_i64, 0 {cannir.buffer.generated_sync}
        asccube.release_buf m, %38, 0 {cannir.buffer.generated_sync}
        scf.yield %57 : i64
      }
      %42 = arith.muli %15, %c128_i64 overflow<nsw> : i64
      %43 = arith.addi %33, %42 overflow<nsw> : i64
      %44 = ascscalar.bind_local_ptr %37 : memref<1xf32, 5>
      %45 = arith.muli %31, %c288_i64 : i64
      %46 = arith.divui %43, %c32_i64 : i64
      %47 = arith.addi %45, %46 : i64
      %48 = arith.muli %47, %c32_i64 : i64
      asccube.get_buf fixpipe, %38, 0 {cannir.buffer.generated_sync}
      %49 = arith.index_cast %48 : i64 to index
      asccube.copy_l0c2gm %alloc_7[%49], %44[%c0], %c128_i32, %c8_i32, %c1152_i32, %c16_i32, %c0_i32 l2_cache_ctl %c0_i32 {nz2nd_en = true, relu_en = false} : memref<9216xf32>, memref<1xf32, 5>
      asccube.release_buf fixpipe, %38, 0 {cannir.buffer.generated_sync}
      scf.yield %41, %40 : i64, i64
    }
    asccube.sync_all
    asccube.sync_block_wait s, %c13_i64
    asccube.sync_block_arrive fixpipe, %c14_i64 {mode = 0 : i32}
    asccube.sync_block_wait s, %c14_i64 {mode = 0 : i32}
    asccube.sync_block_arrive mte3, %c15_i64
    return
  }
}
