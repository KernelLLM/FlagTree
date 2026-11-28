module {
  func.func @run_rope_582a60ad7289e2231ad4f8a610fa779b(%arg0: !cannir.tensor<<gm, bf16>, <<(-1, 1024)>, <(1024, 1)>>, <<(-1, 1024)>, <(1024, 1)>>, identity>, %arg1: !cannir.tensor<<gm, bf16>, <<(640, 1024)>, <(1024, 1)>>, <<(64, 40, 16, 16)>, <(10240, 256, 16, 1)>>, nd2nz> {cannir.physical_shape_upper_bound = array<i64: 64, 40, 16, 16>}, %arg2: !cannir.tensor<<gm, bf16>, <<(512, 640)>, <(640, 1)>>, <<(40, 32, 16, 16)>, <(8192, 256, 16, 1)>>, nd2nz> {cannir.physical_shape_upper_bound = array<i64: 40, 32, 16, 16>}, %arg3: !cannir.tensor<<gm, bf16>, <<(576, 1024)>, <(1024, 1)>>, <<(64, 36, 16, 16)>, <(9216, 256, 16, 1)>>, nd2nz> {cannir.physical_shape_upper_bound = array<i64: 64, 36, 16, 16>}, %arg4: !cannir.tensor<<gm, bf16>, <<(2, 192, 512)>, <(98304, 512, 1)>>, <<(2, 192, 512)>, <(98304, 512, 1)>>, identity>, %arg5: !cannir.tensor<<gm, f32>, <<(1, 640)>, <(640, 1)>>, <<(1, 640)>, <(640, 1)>>, identity>, %arg6: !cannir.tensor<<gm, f32>, <<(1, 512)>, <(512, 1)>>, <<(1, 512)>, <(512, 1)>>, identity>, %arg7: !cannir.tensor<<gm, f8E4M3FN>, <<(-1, -1, 1, 656)>, <(-1, 656, 656, 1)>>, <<(-1, -1, 1, 656)>, <(-1, 656, 656, 1)>>, identity>, %arg8: !cannir.tensor<<gm, i64>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg9: !cannir.tensor<<gm, bf16>, <<(-1, 64)>, <(64, 1)>>, <<(-1, 64)>, <(64, 1)>>, identity>, %arg10: !cannir.tensor<<gm, bf16>, <<(-1, 64)>, <(64, 1)>>, <<(-1, 64)>, <(64, 1)>>, identity>, %arg11: !cannir.tensor<<gm, bf16>, <<(8, 2, 576)>, <(1152, 576, 1)>>, <<(8, 2, 576)>, <(1152, 576, 1)>>, identity>, %arg12: !cannir.tensor<<gm, bf16>, <<(8, 1024)>, <(1024, 1)>>, <<(8, 1024)>, <(1024, 1)>>, identity>, %arg13: !cannir.tensor<<gm, bf16>, <<(8, 672)>, <(672, 1)>>, <<(8, 672)>, <(672, 1)>>, identity>, %arg14: !cannir.tensor<<gm, bf16>, <<(8, 576)>, <(576, 1)>>, <<(8, 576)>, <(576, 1)>>, identity>, %arg15: !cannir.tensor<<gm, bf16>, <<(8, 576)>, <(576, 1)>>, <<(8, 576)>, <(576, 1)>>, identity>, %arg16: !cannir.tensor<<gm, bf16>, <<(8, 640)>, <(640, 1)>>, <<(8, 640)>, <(640, 1)>>, identity>, %arg17: f32, %arg18: f32, %arg19: f32, %arg20: i64, %arg21: i64, %arg22: i64) attributes {cannbotdsl.contract_digest = "582a60ad7289e2231ad4f8a610fa779b406158f6575dc8a483331685e4ada9a2", cannbotdsl.public_slot_count = 23 : i64, cannbotdsl.slot_dtype_bits = [16, 16, 16, 16, 16, 32, 32, 8, 64, 16, 16, 16, 16, 16, 16, 16, 16, 32, 32, 32, 64, 64, 64], cannbotdsl.slot_dtype_codes = [4, 4, 4, 4, 4, 2, 2, 10, 0, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 0, 0, 0], cannbotdsl.slot_dtype_lanes = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], cannbotdsl.slot_flags = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], cannbotdsl.slot_kinds = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2], cannbotdsl.slot_type_digests = ["2f12152648aa422671a77bacab9acff4420df5a14464c5e39a59e14b8782bbea", "581f88679ffc97d9b0f852494dc15761661352bc3407e73adccace68b7a1ca11", "c4ceb7acba4f4bd36f2f8f19e3af51ea9acb33e52e8501f015471c25dd3508a4", "16c9273b7b8e5eb2c2f9bf216b7eaf76bb4b8b6c63e2002f32fadf24ceba8b86", "2090b9288347c1ade3a9fcf409a742a8e5de94b94d181b1c4173c8e5196585ed", "e060b4eeab5e7c9d8b67a3153f869de5aad4eee9553a9a98b398bf5577804a47", "7b962932eb0067670c47bf914f64cc35656e412a09850be9b2bc297c5b918fc3", "2ccfeb5cdbb9760194e1f12a665343acf12c9ae5de2abe95be41bf4cb3bb1880", "5ae7da10e83ac3f5ad55bbe47c978e5c8bcb0ad1d2171af81149eef24de96b7a", "6b6a90f195a5f29ca792af7b1733c8cafee2bfdeb6c69f295bfd0b7b7ca2d0e2", "6b6a90f195a5f29ca792af7b1733c8cafee2bfdeb6c69f295bfd0b7b7ca2d0e2", "26a5f7a509b1c79785ffe493a391678e96b0f2b75ae44b18d80dd5e1a5d9c03d", "8aeb126f8f3c3f165d2516841db0a82c99e1ba495f1076f23cb55e200c479bd2", "06228e85126fccbf1e33641ff0d06b682c62d20f61aaafc43891d07214e515be", "242b167e7f801f9db314d371068618a3ce8621068571e0b424293920276070d9", "242b167e7f801f9db314d371068618a3ce8621068571e0b424293920276070d9", "529f05557b642e904963b68de7bd0d051244ee009b053b90be0a92bba9011258", "4a5363e2ec1eb8cefee037ab0b231653c07f9b44a406ef92ed9ba3e6f5541dbf", "4a5363e2ec1eb8cefee037ab0b231653c07f9b44a406ef92ed9ba3e6f5541dbf", "4a5363e2ec1eb8cefee037ab0b231653c07f9b44a406ef92ed9ba3e6f5541dbf", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515"], cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.host} {
    cannir.kernel_launch @kernel_MlaPrologKernel_run_rope_0(%arg22) [%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18, %arg19, %arg20, %arg21] {cannbotdsl.profile = {indices = array<i64: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16>, name = "MlaPrologKernel_run_rope", op_type = "MlaPrologKernel_run_rope", roles = array<i64: 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1>}} : !cannir.tensor<<gm, bf16>, <<(-1, 1024)>, <(1024, 1)>>, <<(-1, 1024)>, <(1024, 1)>>, identity>, !cannir.tensor<<gm, bf16>, <<(640, 1024)>, <(1024, 1)>>, <<(64, 40, 16, 16)>, <(10240, 256, 16, 1)>>, nd2nz>, !cannir.tensor<<gm, bf16>, <<(512, 640)>, <(640, 1)>>, <<(40, 32, 16, 16)>, <(8192, 256, 16, 1)>>, nd2nz>, !cannir.tensor<<gm, bf16>, <<(576, 1024)>, <(1024, 1)>>, <<(64, 36, 16, 16)>, <(9216, 256, 16, 1)>>, nd2nz>, !cannir.tensor<<gm, bf16>, <<(2, 192, 512)>, <(98304, 512, 1)>>, <<(2, 192, 512)>, <(98304, 512, 1)>>, identity>, !cannir.tensor<<gm, f32>, <<(1, 640)>, <(640, 1)>>, <<(1, 640)>, <(640, 1)>>, identity>, !cannir.tensor<<gm, f32>, <<(1, 512)>, <(512, 1)>>, <<(1, 512)>, <(512, 1)>>, identity>, !cannir.tensor<<gm, f8E4M3FN>, <<(-1, -1, 1, 656)>, <(-1, 656, 656, 1)>>, <<(-1, -1, 1, 656)>, <(-1, 656, 656, 1)>>, identity>, !cannir.tensor<<gm, i64>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, !cannir.tensor<<gm, bf16>, <<(-1, 64)>, <(64, 1)>>, <<(-1, 64)>, <(64, 1)>>, identity>, !cannir.tensor<<gm, bf16>, <<(-1, 64)>, <(64, 1)>>, <<(-1, 64)>, <(64, 1)>>, identity>, !cannir.tensor<<gm, bf16>, <<(8, 2, 576)>, <(1152, 576, 1)>>, <<(8, 2, 576)>, <(1152, 576, 1)>>, identity>, !cannir.tensor<<gm, bf16>, <<(8, 1024)>, <(1024, 1)>>, <<(8, 1024)>, <(1024, 1)>>, identity>, !cannir.tensor<<gm, bf16>, <<(8, 672)>, <(672, 1)>>, <<(8, 672)>, <(672, 1)>>, identity>, !cannir.tensor<<gm, bf16>, <<(8, 576)>, <(576, 1)>>, <<(8, 576)>, <(576, 1)>>, identity>, !cannir.tensor<<gm, bf16>, <<(8, 576)>, <(576, 1)>>, <<(8, 576)>, <(576, 1)>>, identity>, !cannir.tensor<<gm, bf16>, <<(8, 640)>, <(640, 1)>>, <<(8, 640)>, <(640, 1)>>, identity>, f32, f32, f32, i64, i64
    return
  }
  func.func @kernel_MlaPrologKernel_run_rope_0_aiv(%arg0: !cannir.tensor<<gm, bf16>, <<(-1, 1024)>, <(1024, 1)>>, <<(-1, 1024)>, <(1024, 1)>>, identity>, %arg1: !cannir.tensor<<gm, bf16>, <<(640, 1024)>, <(1024, 1)>>, <<(64, 40, 16, 16)>, <(10240, 256, 16, 1)>>, nd2nz> {cannir.physical_shape_upper_bound = array<i64: 64, 40, 16, 16>}, %arg2: !cannir.tensor<<gm, bf16>, <<(512, 640)>, <(640, 1)>>, <<(40, 32, 16, 16)>, <(8192, 256, 16, 1)>>, nd2nz> {cannir.physical_shape_upper_bound = array<i64: 40, 32, 16, 16>}, %arg3: !cannir.tensor<<gm, bf16>, <<(576, 1024)>, <(1024, 1)>>, <<(64, 36, 16, 16)>, <(9216, 256, 16, 1)>>, nd2nz> {cannir.physical_shape_upper_bound = array<i64: 64, 36, 16, 16>}, %arg4: !cannir.tensor<<gm, bf16>, <<(2, 192, 512)>, <(98304, 512, 1)>>, <<(2, 192, 512)>, <(98304, 512, 1)>>, identity>, %arg5: !cannir.tensor<<gm, f32>, <<(1, 640)>, <(640, 1)>>, <<(1, 640)>, <(640, 1)>>, identity>, %arg6: !cannir.tensor<<gm, f32>, <<(1, 512)>, <(512, 1)>>, <<(1, 512)>, <(512, 1)>>, identity>, %arg7: !cannir.tensor<<gm, f8E4M3FN>, <<(-1, -1, 1, 656)>, <(-1, 656, 656, 1)>>, <<(-1, -1, 1, 656)>, <(-1, 656, 656, 1)>>, identity>, %arg8: !cannir.tensor<<gm, i64>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg9: !cannir.tensor<<gm, bf16>, <<(-1, 64)>, <(64, 1)>>, <<(-1, 64)>, <(64, 1)>>, identity>, %arg10: !cannir.tensor<<gm, bf16>, <<(-1, 64)>, <(64, 1)>>, <<(-1, 64)>, <(64, 1)>>, identity>, %arg11: !cannir.tensor<<gm, bf16>, <<(8, 2, 576)>, <(1152, 576, 1)>>, <<(8, 2, 576)>, <(1152, 576, 1)>>, identity>, %arg12: !cannir.tensor<<gm, bf16>, <<(8, 1024)>, <(1024, 1)>>, <<(8, 1024)>, <(1024, 1)>>, identity>, %arg13: !cannir.tensor<<gm, bf16>, <<(8, 672)>, <(672, 1)>>, <<(8, 672)>, <(672, 1)>>, identity>, %arg14: !cannir.tensor<<gm, bf16>, <<(8, 576)>, <(576, 1)>>, <<(8, 576)>, <(576, 1)>>, identity>, %arg15: !cannir.tensor<<gm, bf16>, <<(8, 576)>, <(576, 1)>>, <<(8, 576)>, <(576, 1)>>, identity>, %arg16: !cannir.tensor<<gm, bf16>, <<(8, 640)>, <(640, 1)>>, <<(8, 640)>, <(640, 1)>>, identity>, %arg17: f32, %arg18: f32, %arg19: f32, %arg20: i64, %arg21: i64) attributes {cannir.buffer.auto_sync, cannir.buffer.sync_analyzed, cannir.buffer.sync_verified, cannir.buffer.tensor_state, cannir.core = "aiv", cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.device, cannir.explicit_ring_buffer_cursors, cannir.kernel_group = "kernel_MlaPrologKernel_run_rope_0", cannir.resource.ids_prepared} {
    %c1280_i32 = arith.constant 1280 : i32
    %c2048_i32 = arith.constant 2048 : i32
    %c0_i32 = arith.constant 0 : i32
    %c2048_i64 = arith.constant 2048 : i64
    %c16384_i64 = arith.constant 16384 : i64
    %c128_i32 = arith.constant 128 : i32
    %c0 = arith.constant 0 : index
    %c1_i32 = arith.constant 1 : i32
    %c2560_i64 = arith.constant 2560 : i64
    %c1344_i64 = arith.constant 1344 : i64
    %c1280_i64 = arith.constant 1280 : i64
    %c4096_i64 = arith.constant 4096 : i64
    %c1152_i64 = arith.constant 1152 : i64
    %c128_i64 = arith.constant 128 : i64
    %c164_i64 = arith.constant 164 : i64
    %c512_i64 = arith.constant 512 : i64
    %c576_i64 = arith.constant 576 : i64
    %c256_i64 = arith.constant 256 : i64
    %c192_i64 = arith.constant 192 : i64
    %c10240_i64 = arith.constant 10240 : i64
    %c64_i64 = arith.constant 64 : i64
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
    %alloc_0 = memref.alloc() {cannir.arg_index = 5 : i64} : memref<640xf32>
    %alloc_1 = memref.alloc() {cannir.arg_index = 6 : i64} : memref<512xf32>
    %alloc_2 = memref.alloc() {cannir.arg_index = 7 : i64} : memref<1xf8E4M3FN>
    %alloc_3 = memref.alloc() {cannir.arg_index = 8 : i64} : memref<1xi64>
    %alloc_4 = memref.alloc() {cannir.arg_index = 9 : i64} : memref<1xbf16>
    %alloc_5 = memref.alloc() {cannir.arg_index = 10 : i64} : memref<1xbf16>
    %alloc_6 = memref.alloc() {cannir.arg_index = 11 : i64} : memref<9216xbf16>
    %alloc_7 = memref.alloc() {cannir.arg_index = 12 : i64} : memref<8192xbf16>
    %alloc_8 = memref.alloc() {cannir.arg_index = 13 : i64} : memref<5376xbf16>
    %alloc_9 = memref.alloc() {cannir.arg_index = 14 : i64} : memref<4608xbf16>
    %alloc_10 = memref.alloc() {cannir.arg_index = 15 : i64} : memref<4608xbf16>
    %alloc_11 = memref.alloc() {cannir.arg_index = 16 : i64} : memref<5120xbf16>
    %alloc_12 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<8192xbf16, 1>
    %alloc_13 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<2048xbf16, 1>
    %alloc_14 = memref.alloc() {cannir.byte_offset = 4096 : i64} : memref<2048xbf16, 1>
    %alloc_15 = memref.alloc() {cannir.byte_offset = 8192 : i64} : memref<2048xf32, 1>
    %alloc_16 = memref.alloc() {cannir.byte_offset = 16384 : i64} : memref<2048xf32, 1>
    %alloc_17 = memref.alloc() {cannir.byte_offset = 24576 : i64} : memref<2048xf32, 1>
    %alloc_18 = memref.alloc() {cannir.byte_offset = 49152 : i64} : memref<2048xbf16, 1>
    %alloc_19 = memref.alloc() {cannir.byte_offset = 53248 : i64} : memref<2048xbf16, 1>
    %alloc_20 = memref.alloc() {cannir.byte_offset = 57344 : i64} : memref<8xf32, 1>
    %alloc_21 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<576xbf16, 1>
    %alloc_22 = memref.alloc() {cannir.byte_offset = 1152 : i64} : memref<576xbf16, 1>
    %alloc_23 = memref.alloc() {cannir.byte_offset = 2304 : i64} : memref<576xf32, 1>
    %alloc_24 = memref.alloc() {cannir.byte_offset = 4608 : i64} : memref<576xf32, 1>
    %alloc_25 = memref.alloc() {cannir.byte_offset = 6912 : i64} : memref<512xf32, 1>
    %alloc_26 = memref.alloc() {cannir.byte_offset = 11264 : i64} : memref<656xf8E4M3FN, 1>
    %alloc_27 = memref.alloc() {cannir.byte_offset = 11936 : i64} : memref<656xf8E4M3FN, 1>
    %alloc_28 = memref.alloc() {cannir.byte_offset = 14912 : i64} : memref<64xf32, 1>
    %alloc_29 = memref.alloc() {cannir.byte_offset = 15200 : i64} : memref<64xbf16, 1>
    %alloc_30 = memref.alloc() {cannir.byte_offset = 15328 : i64} : memref<64xbf16, 1>
    %alloc_31 = memref.alloc() {cannir.byte_offset = 15456 : i64} : memref<128xf32, 1>
    %alloc_32 = memref.alloc() {cannir.byte_offset = 15968 : i64} : memref<128xf32, 1>
    %alloc_33 = memref.alloc() {cannir.byte_offset = 16480 : i64} : memref<128xf32, 1>
    %alloc_34 = memref.alloc() {cannir.byte_offset = 16992 : i64} : memref<64xf32, 1>
    %alloc_35 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<64xbf16, 1>
    %alloc_36 = memref.alloc() {cannir.byte_offset = 128 : i64} : memref<64xbf16, 1>
    %alloc_37 = memref.alloc() {cannir.byte_offset = 256 : i64} : memref<64xf32, 1>
    %alloc_38 = memref.alloc() {cannir.byte_offset = 512 : i64} : memref<64xbf16, 1>
    %alloc_39 = memref.alloc() {cannir.byte_offset = 640 : i64} : memref<64xbf16, 1>
    %alloc_40 = memref.alloc() {cannir.byte_offset = 768 : i64} : memref<64xbf16, 1>
    %alloc_41 = memref.alloc() {cannir.byte_offset = 896 : i64} : memref<64xbf16, 1>
    %alloc_42 = memref.alloc() {cannir.byte_offset = 1024 : i64} : memref<128xf32, 1>
    %alloc_43 = memref.alloc() {cannir.byte_offset = 1536 : i64} : memref<128xf32, 1>
    %alloc_44 = memref.alloc() {cannir.byte_offset = 2048 : i64} : memref<128xf32, 1>
    %alloc_45 = memref.alloc() {cannir.byte_offset = 2560 : i64} : memref<64xf32, 1>
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
      scf.for %arg22 = %7 to %c8_i64 step %17  : i64 {
        %18 = arith.cmpi slt, %arg22, %5 : i64
        scf.if %18 {
          %19 = ascscalar.bitcast_memref %alloc_12 byte_offset = %c0_i64 : memref<8192xbf16, 1> -> memref<?xbf16, 1>
          %20 = arith.minsi %arg22, %5 : i64
          %21 = arith.subi %5, %20 : i64
          %22 = arith.minsi %21, %c1_i64 : i64
          %23 = arith.cmpi sle, %arg22, %5 : i64
          %24 = arith.select %23, %22, %c0_i64 : i64
          %25 = arith.muli %arg22, %c2048_i64 : i64
          ascvec.get_buf mte2, %c0_i64, 0 {cannir.buffer.generated_sync}
          "ascvec.copy_gm2ub_plan"(%19, %alloc, %c0_i32, %c0_i32, %c0_i64, %25, %24, %c2048_i64, %c2048_i64, %c16384_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = 0 : i64, ub_capacity = 16384 : i64}> : (memref<?xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
          ascvec.release_buf mte2, %c0_i64, 0 {cannir.buffer.generated_sync}
          %26 = arith.muli %arg22, %c16384_i64 : i64
          %27 = ascscalar.memref_address %alloc_7 : memref<8192xbf16>
          %28 = arith.divui %26, %c8_i64 : i64
          %29 = arith.addi %27, %28 : i64
          %30 = ascscalar.bind_gm_ptr %29 : memref<1xbf16>
          ascvec.get_buf mte3, %c0_i64, 0 {cannir.buffer.generated_sync}
          ascvec.copy_ub2gm %30[%c0], %19[%c0], %c1_i32, %c2048_i32, %c0_i64, %c2048_i32 l2_cache_ctl %c0_i32 : memref<1xbf16>, memref<?xbf16, 1>
          ascvec.release_buf mte3, %c0_i64, 0 {cannir.buffer.generated_sync}
        } else {
          ascvec.get_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
          ascvec.vf(%alloc_12 : memref<8192xbf16, 1>) {
          ^bb0(%arg23: memref<8192xbf16, 1>):
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
            %c0_46 = arith.constant 0 : index
            %cst = arith.constant 0.000000e+00 : bf16
            %c128_i32_47 = arith.constant 128 : i32
            %c0_48 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            scf.for %arg24 = %c0_48 to %c1 step %c1 {
              %mask, %remaining_out = ascvec.update_mask %c128_i32_47 elem_bits 16 : !ascvec.mask
              %25 = ascvec.duplicate_scalar %cst, %mask : bf16 -> <bf16>
              ascvec.storealign %arg23[%c0_46], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg23[%c128], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg23[%c256], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg23[%c384], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg23[%c512], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg23[%c640], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg23[%c768], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg23[%c896], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg23[%c1024], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg23[%c1152], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg23[%c1280], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg23[%c1408], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg23[%c1536], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg23[%c1664], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg23[%c1792], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg23[%c1920], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg23[%c2048], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg23[%c2176], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg23[%c2304], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg23[%c2432], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg23[%c2560], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg23[%c2688], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg23[%c2816], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg23[%c2944], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg23[%c3072], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg23[%c3200], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg23[%c3328], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg23[%c3456], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg23[%c3584], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg23[%c3712], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg23[%c3840], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg23[%c3968], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg23[%c4096], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg23[%c4224], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg23[%c4352], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg23[%c4480], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg23[%c4608], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg23[%c4736], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg23[%c4864], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg23[%c4992], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg23[%c5120], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg23[%c5248], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg23[%c5376], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg23[%c5504], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg23[%c5632], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg23[%c5760], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg23[%c5888], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg23[%c6016], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg23[%c6144], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg23[%c6272], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg23[%c6400], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg23[%c6528], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg23[%c6656], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg23[%c6784], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg23[%c6912], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg23[%c7040], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg23[%c7168], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg23[%c7296], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg23[%c7424], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg23[%c7552], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg23[%c7680], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg23[%c7808], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg23[%c7936], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
              ascvec.storealign %arg23[%c8064], %25, %mask : memref<8192xbf16, 1>, index, <bf16>
            }
            ascvec.mem_bar vst_vld
          } {cannir.vf_mode = "simd"}
          ascvec.release_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
          %19 = arith.muli %arg22, %c16384_i64 : i64
          %20 = ascscalar.memref_address %alloc_7 : memref<8192xbf16>
          %21 = arith.divui %19, %c8_i64 : i64
          %22 = arith.addi %20, %21 : i64
          %23 = ascscalar.bind_gm_ptr %22 : memref<1xbf16>
          %24 = ascscalar.bitcast_memref %alloc_12 byte_offset = %c0_i64 : memref<8192xbf16, 1> -> memref<?xbf16, 1>
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
      %18 = ascscalar.bitcast_memref %alloc_17 byte_offset = %c0_i64 : memref<2048xf32, 1> -> memref<?xf32, 1>
      ascvec.get_buf mte2, %c2_i64, 0 {cannir.buffer.generated_sync}
      "ascvec.copy_gm2ub_plan"(%18, %alloc_0, %c0_i32, %c0_i32, %c0_i64, %c0_i64, %c1_i64, %c2560_i64, %c0_i64, %c2560_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 2560 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 24576 : i64, ub_capacity = 8192 : i64}> : (memref<?xf32, 1>, memref<640xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
      ascvec.release_buf mte2, %c2_i64, 0 {cannir.buffer.generated_sync}
      %19 = scf.for %arg22 = %7 to %5 step %17 iter_args(%arg23 = %c0_i64) -> (i64)  : i64 {
        %20 = arith.cmpi eq, %arg23, %c0_i64 : i64
        %21 = ascscalar.memref_address %alloc_13 : memref<2048xbf16, 1>
        %22 = ascscalar.memref_address %alloc_14 : memref<2048xbf16, 1>
        %23 = arith.select %20, %21, %22 : i64
        %24 = arith.cmpi ne, %arg23, %c0_i64 : i64
        %25 = arith.extui %24 : i1 to i64
        %26 = arith.addi %arg23, %c1_i64 overflow<nsw> : i64
        %27 = arith.remui %26, %c2_i64 : i64
        %28 = arith.minsi %arg22, %c8_i64 : i64
        %29 = arith.subi %c8_i64, %28 : i64
        %30 = arith.minsi %29, %c1_i64 : i64
        %31 = arith.cmpi sle, %arg22, %c8_i64 : i64
        %32 = arith.select %31, %30, %c0_i64 : i64
        %33 = arith.muli %arg22, %c1344_i64 : i64
        %34 = ascscalar.bind_local_ptr %23 : memref<1xbf16, 1>
        ascvec.get_buf mte2, %25, 0 {cannir.buffer.generated_sync}
        "ascvec.copy_gm2ub_plan"(%34, %alloc_8, %c0_i32, %c0_i32, %c0_i64, %33, %32, %c1280_i64, %c1344_i64, %c4096_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 10752 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<5376xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
        ascvec.release_buf mte2, %25, 0 {cannir.buffer.generated_sync}
        %35 = ascscalar.bind_local_ptr %23 : memref<2048xbf16, 1>
        %36 = ascscalar.memref_address %alloc_15 : memref<2048xf32, 1>
        %37 = ascscalar.memref_address %alloc_16 : memref<2048xf32, 1>
        %38 = arith.select %20, %36, %37 : i64
        %39 = ascscalar.bind_local_ptr %38 : memref<2048xf32, 1>
        ascvec.get_buf v, %25, 0 {cannir.buffer.generated_sync}
        ascvec.vf(%35, %39 : memref<2048xbf16, 1>, memref<2048xf32, 1>) {
        ^bb0(%arg24: memref<2048xbf16, 1>, %arg25: memref<2048xf32, 1>):
          %c64_i32 = arith.constant 64 : i32
          %c640_i32 = arith.constant 640 : i32
          %c0_i32_46 = arith.constant 0 : i32
          scf.for %arg26 = %c0_i32_46 to %c640_i32 step %c64_i32  : i32 {
            %51 = arith.index_cast %arg26 : i32 to index
            %52 = ascvec.loadalign_unpack %arg24[%51] {mode = "b16_to_b32"} : memref<2048xbf16, 1>, index -> <bf16>
            %53 = ascvec.create_mask : !ascvec.mask
            %54 = ascvec.cast %52, %53 {rounding = 6 : i32} : <bf16> -> <f32>
            ascvec.storealign %arg25[%51], %54, %53 : memref<2048xf32, 1>, index, <f32>
          }
          ascvec.mem_bar vst_vld
        } {cannir.vf_mode = "simd"}
        ascvec.release_buf v, %25, 0 {cannir.buffer.generated_sync}
        ascvec.vf(%39, %arg17, %alloc_20 : memref<2048xf32, 1>, f32, memref<8xf32, 1>) {
        ^bb0(%arg24: memref<2048xf32, 1>, %arg25: f32, %arg26: memref<8xf32, 1>):
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
          %c0_i32_46 = arith.constant 0 : i32
          %c0_47 = arith.constant 0 : index
          %c64_i32 = arith.constant 64 : i32
          %cst_48 = arith.constant 0.000000e+00 : f32
          %c1_i32_49 = arith.constant 1 : i32
          %c0_50 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          scf.for %arg27 = %c0_50 to %c1 step %c1 {
            %mask, %remaining_out = ascvec.update_mask %c1_i32_49 : !ascvec.mask
            %51 = ascvec.duplicate_scalar %cst_48, %mask : f32 -> <f32>
            %mask_51, %remaining_out_52 = ascvec.update_mask %c64_i32 : !ascvec.mask
            %52 = ascvec.loadalign %arg24[%c0_47] : memref<2048xf32, 1>, index -> <f32>
            %53 = ascvec.mul %52, %52, %mask_51 : <f32>
            %54 = ascvec.reduce_sum %53, %mask_51 : <f32>
            %55 = ascvec.add %51, %54, %mask : <f32>
            %mask_53, %remaining_out_54 = ascvec.update_mask %c64_i32 : !ascvec.mask
            %56 = ascvec.loadalign %arg24[%c64] : memref<2048xf32, 1>, index -> <f32>
            %57 = ascvec.mul %56, %56, %mask_53 : <f32>
            %58 = ascvec.reduce_sum %57, %mask_53 : <f32>
            %59 = ascvec.add %55, %58, %mask : <f32>
            %mask_55, %remaining_out_56 = ascvec.update_mask %c64_i32 : !ascvec.mask
            %60 = ascvec.loadalign %arg24[%c128] : memref<2048xf32, 1>, index -> <f32>
            %61 = ascvec.mul %60, %60, %mask_55 : <f32>
            %62 = ascvec.reduce_sum %61, %mask_55 : <f32>
            %63 = ascvec.add %59, %62, %mask : <f32>
            %mask_57, %remaining_out_58 = ascvec.update_mask %c64_i32 : !ascvec.mask
            %64 = ascvec.loadalign %arg24[%c192] : memref<2048xf32, 1>, index -> <f32>
            %65 = ascvec.mul %64, %64, %mask_57 : <f32>
            %66 = ascvec.reduce_sum %65, %mask_57 : <f32>
            %67 = ascvec.add %63, %66, %mask : <f32>
            %mask_59, %remaining_out_60 = ascvec.update_mask %c64_i32 : !ascvec.mask
            %68 = ascvec.loadalign %arg24[%c256] : memref<2048xf32, 1>, index -> <f32>
            %69 = ascvec.mul %68, %68, %mask_59 : <f32>
            %70 = ascvec.reduce_sum %69, %mask_59 : <f32>
            %71 = ascvec.add %67, %70, %mask : <f32>
            %mask_61, %remaining_out_62 = ascvec.update_mask %c64_i32 : !ascvec.mask
            %72 = ascvec.loadalign %arg24[%c320] : memref<2048xf32, 1>, index -> <f32>
            %73 = ascvec.mul %72, %72, %mask_61 : <f32>
            %74 = ascvec.reduce_sum %73, %mask_61 : <f32>
            %75 = ascvec.add %71, %74, %mask : <f32>
            %mask_63, %remaining_out_64 = ascvec.update_mask %c64_i32 : !ascvec.mask
            %76 = ascvec.loadalign %arg24[%c384] : memref<2048xf32, 1>, index -> <f32>
            %77 = ascvec.mul %76, %76, %mask_63 : <f32>
            %78 = ascvec.reduce_sum %77, %mask_63 : <f32>
            %79 = ascvec.add %75, %78, %mask : <f32>
            %mask_65, %remaining_out_66 = ascvec.update_mask %c64_i32 : !ascvec.mask
            %80 = ascvec.loadalign %arg24[%c448] : memref<2048xf32, 1>, index -> <f32>
            %81 = ascvec.mul %80, %80, %mask_65 : <f32>
            %82 = ascvec.reduce_sum %81, %mask_65 : <f32>
            %83 = ascvec.add %79, %82, %mask : <f32>
            %mask_67, %remaining_out_68 = ascvec.update_mask %c64_i32 : !ascvec.mask
            %84 = ascvec.loadalign %arg24[%c512] : memref<2048xf32, 1>, index -> <f32>
            %85 = ascvec.mul %84, %84, %mask_67 : <f32>
            %86 = ascvec.reduce_sum %85, %mask_67 : <f32>
            %87 = ascvec.add %83, %86, %mask : <f32>
            %mask_69, %remaining_out_70 = ascvec.update_mask %c64_i32 : !ascvec.mask
            %88 = ascvec.loadalign %arg24[%c576] : memref<2048xf32, 1>, index -> <f32>
            %89 = ascvec.mul %88, %88, %mask_69 : <f32>
            %90 = ascvec.reduce_sum %89, %mask_69 : <f32>
            %91 = ascvec.add %87, %90, %mask : <f32>
            %mask_71, %remaining_out_72 = ascvec.update_mask %c0_i32_46 : !ascvec.mask
            %92 = ascvec.loadalign %arg24[%c640] : memref<2048xf32, 1>, index -> <f32>
            %93 = ascvec.mul %92, %92, %mask_71 : <f32>
            %94 = ascvec.reduce_sum %93, %mask_71 : <f32>
            %95 = ascvec.add %91, %94, %mask : <f32>
            %mask_73, %remaining_out_74 = ascvec.update_mask %c0_i32_46 : !ascvec.mask
            %96 = ascvec.loadalign %arg24[%c704] : memref<2048xf32, 1>, index -> <f32>
            %97 = ascvec.mul %96, %96, %mask_73 : <f32>
            %98 = ascvec.reduce_sum %97, %mask_73 : <f32>
            %99 = ascvec.add %95, %98, %mask : <f32>
            %mask_75, %remaining_out_76 = ascvec.update_mask %c0_i32_46 : !ascvec.mask
            %100 = ascvec.loadalign %arg24[%c768] : memref<2048xf32, 1>, index -> <f32>
            %101 = ascvec.mul %100, %100, %mask_75 : <f32>
            %102 = ascvec.reduce_sum %101, %mask_75 : <f32>
            %103 = ascvec.add %99, %102, %mask : <f32>
            %mask_77, %remaining_out_78 = ascvec.update_mask %c0_i32_46 : !ascvec.mask
            %104 = ascvec.loadalign %arg24[%c832] : memref<2048xf32, 1>, index -> <f32>
            %105 = ascvec.mul %104, %104, %mask_77 : <f32>
            %106 = ascvec.reduce_sum %105, %mask_77 : <f32>
            %107 = ascvec.add %103, %106, %mask : <f32>
            %mask_79, %remaining_out_80 = ascvec.update_mask %c0_i32_46 : !ascvec.mask
            %108 = ascvec.loadalign %arg24[%c896] : memref<2048xf32, 1>, index -> <f32>
            %109 = ascvec.mul %108, %108, %mask_79 : <f32>
            %110 = ascvec.reduce_sum %109, %mask_79 : <f32>
            %111 = ascvec.add %107, %110, %mask : <f32>
            %mask_81, %remaining_out_82 = ascvec.update_mask %c0_i32_46 : !ascvec.mask
            %112 = ascvec.loadalign %arg24[%c960] : memref<2048xf32, 1>, index -> <f32>
            %113 = ascvec.mul %112, %112, %mask_81 : <f32>
            %114 = ascvec.reduce_sum %113, %mask_81 : <f32>
            %115 = ascvec.add %111, %114, %mask : <f32>
            %mask_83, %remaining_out_84 = ascvec.update_mask %c0_i32_46 : !ascvec.mask
            %116 = ascvec.loadalign %arg24[%c1024] : memref<2048xf32, 1>, index -> <f32>
            %117 = ascvec.mul %116, %116, %mask_83 : <f32>
            %118 = ascvec.reduce_sum %117, %mask_83 : <f32>
            %119 = ascvec.add %115, %118, %mask : <f32>
            %mask_85, %remaining_out_86 = ascvec.update_mask %c0_i32_46 : !ascvec.mask
            %120 = ascvec.loadalign %arg24[%c1088] : memref<2048xf32, 1>, index -> <f32>
            %121 = ascvec.mul %120, %120, %mask_85 : <f32>
            %122 = ascvec.reduce_sum %121, %mask_85 : <f32>
            %123 = ascvec.add %119, %122, %mask : <f32>
            %mask_87, %remaining_out_88 = ascvec.update_mask %c0_i32_46 : !ascvec.mask
            %124 = ascvec.loadalign %arg24[%c1152] : memref<2048xf32, 1>, index -> <f32>
            %125 = ascvec.mul %124, %124, %mask_87 : <f32>
            %126 = ascvec.reduce_sum %125, %mask_87 : <f32>
            %127 = ascvec.add %123, %126, %mask : <f32>
            %mask_89, %remaining_out_90 = ascvec.update_mask %c0_i32_46 : !ascvec.mask
            %128 = ascvec.loadalign %arg24[%c1216] : memref<2048xf32, 1>, index -> <f32>
            %129 = ascvec.mul %128, %128, %mask_89 : <f32>
            %130 = ascvec.reduce_sum %129, %mask_89 : <f32>
            %131 = ascvec.add %127, %130, %mask : <f32>
            %mask_91, %remaining_out_92 = ascvec.update_mask %c0_i32_46 : !ascvec.mask
            %132 = ascvec.loadalign %arg24[%c1280] : memref<2048xf32, 1>, index -> <f32>
            %133 = ascvec.mul %132, %132, %mask_91 : <f32>
            %134 = ascvec.reduce_sum %133, %mask_91 : <f32>
            %135 = ascvec.add %131, %134, %mask : <f32>
            %mask_93, %remaining_out_94 = ascvec.update_mask %c0_i32_46 : !ascvec.mask
            %136 = ascvec.loadalign %arg24[%c1344] : memref<2048xf32, 1>, index -> <f32>
            %137 = ascvec.mul %136, %136, %mask_93 : <f32>
            %138 = ascvec.reduce_sum %137, %mask_93 : <f32>
            %139 = ascvec.add %135, %138, %mask : <f32>
            %mask_95, %remaining_out_96 = ascvec.update_mask %c0_i32_46 : !ascvec.mask
            %140 = ascvec.loadalign %arg24[%c1408] : memref<2048xf32, 1>, index -> <f32>
            %141 = ascvec.mul %140, %140, %mask_95 : <f32>
            %142 = ascvec.reduce_sum %141, %mask_95 : <f32>
            %143 = ascvec.add %139, %142, %mask : <f32>
            %mask_97, %remaining_out_98 = ascvec.update_mask %c0_i32_46 : !ascvec.mask
            %144 = ascvec.loadalign %arg24[%c1472] : memref<2048xf32, 1>, index -> <f32>
            %145 = ascvec.mul %144, %144, %mask_97 : <f32>
            %146 = ascvec.reduce_sum %145, %mask_97 : <f32>
            %147 = ascvec.add %143, %146, %mask : <f32>
            %mask_99, %remaining_out_100 = ascvec.update_mask %c0_i32_46 : !ascvec.mask
            %148 = ascvec.loadalign %arg24[%c1536] : memref<2048xf32, 1>, index -> <f32>
            %149 = ascvec.mul %148, %148, %mask_99 : <f32>
            %150 = ascvec.reduce_sum %149, %mask_99 : <f32>
            %151 = ascvec.add %147, %150, %mask : <f32>
            %mask_101, %remaining_out_102 = ascvec.update_mask %c0_i32_46 : !ascvec.mask
            %152 = ascvec.loadalign %arg24[%c1600] : memref<2048xf32, 1>, index -> <f32>
            %153 = ascvec.mul %152, %152, %mask_101 : <f32>
            %154 = ascvec.reduce_sum %153, %mask_101 : <f32>
            %155 = ascvec.add %151, %154, %mask : <f32>
            %mask_103, %remaining_out_104 = ascvec.update_mask %c0_i32_46 : !ascvec.mask
            %156 = ascvec.loadalign %arg24[%c1664] : memref<2048xf32, 1>, index -> <f32>
            %157 = ascvec.mul %156, %156, %mask_103 : <f32>
            %158 = ascvec.reduce_sum %157, %mask_103 : <f32>
            %159 = ascvec.add %155, %158, %mask : <f32>
            %mask_105, %remaining_out_106 = ascvec.update_mask %c0_i32_46 : !ascvec.mask
            %160 = ascvec.loadalign %arg24[%c1728] : memref<2048xf32, 1>, index -> <f32>
            %161 = ascvec.mul %160, %160, %mask_105 : <f32>
            %162 = ascvec.reduce_sum %161, %mask_105 : <f32>
            %163 = ascvec.add %159, %162, %mask : <f32>
            %mask_107, %remaining_out_108 = ascvec.update_mask %c0_i32_46 : !ascvec.mask
            %164 = ascvec.loadalign %arg24[%c1792] : memref<2048xf32, 1>, index -> <f32>
            %165 = ascvec.mul %164, %164, %mask_107 : <f32>
            %166 = ascvec.reduce_sum %165, %mask_107 : <f32>
            %167 = ascvec.add %163, %166, %mask : <f32>
            %mask_109, %remaining_out_110 = ascvec.update_mask %c0_i32_46 : !ascvec.mask
            %168 = ascvec.loadalign %arg24[%c1856] : memref<2048xf32, 1>, index -> <f32>
            %169 = ascvec.mul %168, %168, %mask_109 : <f32>
            %170 = ascvec.reduce_sum %169, %mask_109 : <f32>
            %171 = ascvec.add %167, %170, %mask : <f32>
            %mask_111, %remaining_out_112 = ascvec.update_mask %c0_i32_46 : !ascvec.mask
            %172 = ascvec.loadalign %arg24[%c1920] : memref<2048xf32, 1>, index -> <f32>
            %173 = ascvec.mul %172, %172, %mask_111 : <f32>
            %174 = ascvec.reduce_sum %173, %mask_111 : <f32>
            %175 = ascvec.add %171, %174, %mask : <f32>
            %mask_113, %remaining_out_114 = ascvec.update_mask %c0_i32_46 : !ascvec.mask
            %176 = ascvec.loadalign %arg24[%c1984] : memref<2048xf32, 1>, index -> <f32>
            %177 = ascvec.mul %176, %176, %mask_113 : <f32>
            %178 = ascvec.reduce_sum %177, %mask_113 : <f32>
            %179 = ascvec.add %175, %178, %mask : <f32>
            %180 = ascvec.mul_scalar %179, %cst, %mask : <f32>, f32
            %181 = ascvec.add_scalar %180, %arg25, %mask : <f32>, f32
            %182 = ascvec.sqrt %181, %mask : <f32>
            ascvec.storealign_1st %arg26[%c0_47], %182 : memref<8xf32, 1>, index, <f32>
          }
          ascvec.mem_bar vst_vld
        } {cannir.vf_mode = "simd"}
        %40 = ascscalar.memref_address %alloc_18 : memref<2048xbf16, 1>
        %41 = ascscalar.memref_address %alloc_19 : memref<2048xbf16, 1>
        %42 = arith.select %20, %40, %41 : i64
        %43 = ascscalar.bind_local_ptr %42 : memref<2048xbf16, 1>
        %44 = arith.select %20, %c3_i64, %c4_i64 : i64
        ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
        ascvec.get_buf v, %44, 0 {cannir.buffer.generated_sync}
        ascvec.vf(%alloc_20, %39, %alloc_17, %arg18, %43 : memref<8xf32, 1>, memref<2048xf32, 1>, memref<2048xf32, 1>, f32, memref<2048xbf16, 1>) {
        ^bb0(%arg24: memref<8xf32, 1>, %arg25: memref<2048xf32, 1>, %arg26: memref<2048xf32, 1>, %arg27: f32, %arg28: memref<2048xbf16, 1>):
          %c64_i32 = arith.constant 64 : i32
          %c640_i32 = arith.constant 640 : i32
          %c0_i32_46 = arith.constant 0 : i32
          %c0_47 = arith.constant 0 : index
          %51 = ascvec.loadalign_brc %arg24[%c0_47] : memref<8xf32, 1>, index -> <f32>
          scf.for %arg29 = %c0_i32_46 to %c640_i32 step %c64_i32  : i32 {
            %52 = arith.index_cast %arg29 : i32 to index
            %53 = ascvec.loadalign %arg25[%52] : memref<2048xf32, 1>, index -> <f32>
            %54 = ascvec.create_mask : !ascvec.mask
            %55 = ascvec.div %53, %51, %54 {precision_mode = true} : <f32>
            %56 = ascvec.loadalign %arg26[%52] : memref<2048xf32, 1>, index -> <f32>
            %57 = ascvec.mul %55, %56, %54 : <f32>
            %58 = ascvec.mul_scalar %57, %arg27, %54 : <f32>, f32
            %59 = ascvec.cast %58, %54 : <f32> -> <bf16>
            ascvec.storealign_pack %arg28[%52], %59, %54 {mode = "b32_to_b16"} : memref<2048xbf16, 1>, index, <bf16>
          }
          ascvec.mem_bar vst_vld
        } {cannir.vf_mode = "simd"}
        ascvec.release_buf v, %44, 0 {cannir.buffer.generated_sync}
        ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
        %45 = arith.muli %arg22, %c10240_i64 : i64
        %46 = ascscalar.memref_address %alloc_11 : memref<5120xbf16>
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
    %13 = scf.for %arg22 = %11 to %c8_i64 step %12 iter_args(%arg23 = %10) -> (i64)  : i64 {
      %17 = arith.cmpi eq, %arg23, %c0_i64 : i64
      %18 = ascscalar.memref_address %alloc_18 : memref<2048xbf16, 1>
      %19 = ascscalar.memref_address %alloc_19 : memref<2048xbf16, 1>
      %20 = arith.select %17, %18, %19 : i64
      %21 = ascscalar.bind_local_ptr %20 : memref<2048xbf16, 1>
      %22 = arith.select %17, %c3_i64, %c4_i64 : i64
      %23 = arith.addi %arg23, %c1_i64 overflow<nsw> : i64
      %24 = arith.remui %23, %c2_i64 : i64
      ascvec.get_buf v, %22, 0 {cannir.buffer.generated_sync}
      ascvec.vf(%21 : memref<2048xbf16, 1>) {
      ^bb0(%arg24: memref<2048xbf16, 1>):
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
        %c0_46 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : bf16
        %c128_i32_47 = arith.constant 128 : i32
        %c0_48 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        scf.for %arg25 = %c0_48 to %c1 step %c1 {
          %mask, %remaining_out = ascvec.update_mask %c128_i32_47 elem_bits 16 : !ascvec.mask
          %31 = ascvec.duplicate_scalar %cst, %mask : bf16 -> <bf16>
          ascvec.storealign %arg24[%c0_46], %31, %mask : memref<2048xbf16, 1>, index, <bf16>
          ascvec.storealign %arg24[%c128], %31, %mask : memref<2048xbf16, 1>, index, <bf16>
          ascvec.storealign %arg24[%c256], %31, %mask : memref<2048xbf16, 1>, index, <bf16>
          ascvec.storealign %arg24[%c384], %31, %mask : memref<2048xbf16, 1>, index, <bf16>
          ascvec.storealign %arg24[%c512], %31, %mask : memref<2048xbf16, 1>, index, <bf16>
          ascvec.storealign %arg24[%c640], %31, %mask : memref<2048xbf16, 1>, index, <bf16>
          ascvec.storealign %arg24[%c768], %31, %mask : memref<2048xbf16, 1>, index, <bf16>
          ascvec.storealign %arg24[%c896], %31, %mask : memref<2048xbf16, 1>, index, <bf16>
          ascvec.storealign %arg24[%c1024], %31, %mask : memref<2048xbf16, 1>, index, <bf16>
          ascvec.storealign %arg24[%c1152], %31, %mask : memref<2048xbf16, 1>, index, <bf16>
          ascvec.storealign %arg24[%c1280], %31, %mask : memref<2048xbf16, 1>, index, <bf16>
          ascvec.storealign %arg24[%c1408], %31, %mask : memref<2048xbf16, 1>, index, <bf16>
          ascvec.storealign %arg24[%c1536], %31, %mask : memref<2048xbf16, 1>, index, <bf16>
          ascvec.storealign %arg24[%c1664], %31, %mask : memref<2048xbf16, 1>, index, <bf16>
          ascvec.storealign %arg24[%c1792], %31, %mask : memref<2048xbf16, 1>, index, <bf16>
          ascvec.storealign %arg24[%c1920], %31, %mask : memref<2048xbf16, 1>, index, <bf16>
        }
        ascvec.mem_bar vst_vld
      } {cannir.vf_mode = "simd"}
      ascvec.release_buf v, %22, 0 {cannir.buffer.generated_sync}
      %25 = arith.muli %arg22, %c10240_i64 : i64
      %26 = ascscalar.memref_address %alloc_11 : memref<5120xbf16>
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
      %17 = ascscalar.bitcast_memref %alloc_25 byte_offset = %c0_i64 : memref<512xf32, 1> -> memref<?xf32, 1>
      ascvec.get_buf mte2, %c2_i64, 0 {cannir.buffer.generated_sync}
      "ascvec.copy_gm2ub_plan"(%17, %alloc_1, %c0_i32, %c0_i32, %c0_i64, %c0_i64, %c1_i64, %c2048_i64, %c0_i64, %c2048_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 2048 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 6912 : i64, ub_capacity = 2048 : i64}> : (memref<?xf32, 1>, memref<512xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
      ascvec.release_buf mte2, %c2_i64, 0 {cannir.buffer.generated_sync}
      %18 = scf.for %arg22 = %0 to %5 step %1 iter_args(%arg23 = %c0_i64) -> (i64)  : i64 {
        %19 = arith.cmpi eq, %arg23, %c0_i64 : i64
        %20 = ascscalar.memref_address %alloc_21 : memref<576xbf16, 1>
        %21 = ascscalar.memref_address %alloc_22 : memref<576xbf16, 1>
        %22 = arith.select %19, %20, %21 : i64
        %23 = arith.cmpi ne, %arg23, %c0_i64 : i64
        %24 = arith.extui %23 : i1 to i64
        %25 = arith.addi %arg23, %c1_i64 overflow<nsw> : i64
        %26 = arith.remui %25, %c2_i64 : i64
        %27 = arith.minsi %arg22, %c8_i64 : i64
        %28 = arith.subi %c8_i64, %27 : i64
        %29 = arith.minsi %28, %c1_i64 : i64
        %30 = arith.cmpi sle, %arg22, %c8_i64 : i64
        %31 = arith.select %30, %29, %c0_i64 : i64
        %32 = arith.muli %arg22, %c1152_i64 : i64
        %33 = arith.muli %31, %c1152_i64 : i64
        %34 = arith.cmpi eq, %33, %c0_i64 : i64
        %35 = arith.cmpi ne, %33, %c0_i64 : i64
        %36 = arith.extui %35 : i1 to i64
        %37 = arith.select %34, %c2_i64, %33 : i64
        %38 = ascscalar.bind_local_ptr %22 : memref<1xbf16, 1>
        ascvec.get_buf mte2, %24, 0 {cannir.buffer.generated_sync}
        "ascvec.copy_gm2ub_plan"(%38, %alloc_9, %c0_i32, %c0_i32, %c0_i64, %32, %36, %37, %c0_i64, %37, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 9216 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<4608xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
        ascvec.release_buf mte2, %24, 0 {cannir.buffer.generated_sync}
        %39 = ascscalar.bind_local_ptr %22 : memref<576xbf16, 1>
        %40 = ascscalar.memref_address %alloc_23 : memref<576xf32, 1>
        %41 = ascscalar.memref_address %alloc_24 : memref<576xf32, 1>
        %42 = arith.select %19, %40, %41 : i64
        %43 = ascscalar.bind_local_ptr %42 : memref<576xf32, 1>
        ascvec.get_buf v, %24, 0 {cannir.buffer.generated_sync}
        ascvec.vf(%39, %43 : memref<576xbf16, 1>, memref<576xf32, 1>) {
        ^bb0(%arg24: memref<576xbf16, 1>, %arg25: memref<576xf32, 1>):
          %c64_i32 = arith.constant 64 : i32
          %c576_i32 = arith.constant 576 : i32
          %c0_i32_46 = arith.constant 0 : i32
          scf.for %arg26 = %c0_i32_46 to %c576_i32 step %c64_i32  : i32 {
            %83 = arith.index_cast %arg26 : i32 to index
            %84 = ascvec.loadalign_unpack %arg24[%83] {mode = "b16_to_b32"} : memref<576xbf16, 1>, index -> <bf16>
            %85 = ascvec.create_mask : !ascvec.mask
            %86 = ascvec.cast %84, %85 {rounding = 6 : i32} : <bf16> -> <f32>
            ascvec.storealign %arg25[%83], %86, %85 : memref<576xf32, 1>, index, <f32>
          }
          ascvec.mem_bar vst_vld
        } {cannir.vf_mode = "simd"}
        ascvec.release_buf v, %24, 0 {cannir.buffer.generated_sync}
        ascvec.vf(%43, %arg17, %alloc_28 : memref<576xf32, 1>, f32, memref<64xf32, 1>) {
        ^bb0(%arg24: memref<576xf32, 1>, %arg25: f32, %arg26: memref<64xf32, 1>):
          %c448 = arith.constant 448 : index
          %c384 = arith.constant 384 : index
          %c320 = arith.constant 320 : index
          %c256 = arith.constant 256 : index
          %c192 = arith.constant 192 : index
          %c128 = arith.constant 128 : index
          %c64 = arith.constant 64 : index
          %cst = arith.constant 0.001953125 : f32
          %c0_46 = arith.constant 0 : index
          %cst_47 = arith.constant 0.000000e+00 : f32
          %c1_i32_48 = arith.constant 1 : i32
          %c0_49 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          scf.for %arg27 = %c0_49 to %c1 step %c1 {
            %mask, %remaining_out = ascvec.update_mask %c1_i32_48 : !ascvec.mask
            %83 = ascvec.duplicate_scalar %cst_47, %mask : f32 -> <f32>
            %84 = ascvec.loadalign %arg24[%c0_46] : memref<576xf32, 1>, index -> <f32>
            %85 = ascvec.create_mask : !ascvec.mask
            %86 = ascvec.mul %84, %84, %85 : <f32>
            %87 = ascvec.reduce_sum %86, %85 : <f32>
            %88 = ascvec.add %83, %87, %mask : <f32>
            %89 = ascvec.loadalign %arg24[%c64] : memref<576xf32, 1>, index -> <f32>
            %90 = ascvec.mul %89, %89, %85 : <f32>
            %91 = ascvec.reduce_sum %90, %85 : <f32>
            %92 = ascvec.add %88, %91, %mask : <f32>
            %93 = ascvec.loadalign %arg24[%c128] : memref<576xf32, 1>, index -> <f32>
            %94 = ascvec.mul %93, %93, %85 : <f32>
            %95 = ascvec.reduce_sum %94, %85 : <f32>
            %96 = ascvec.add %92, %95, %mask : <f32>
            %97 = ascvec.loadalign %arg24[%c192] : memref<576xf32, 1>, index -> <f32>
            %98 = ascvec.mul %97, %97, %85 : <f32>
            %99 = ascvec.reduce_sum %98, %85 : <f32>
            %100 = ascvec.add %96, %99, %mask : <f32>
            %101 = ascvec.loadalign %arg24[%c256] : memref<576xf32, 1>, index -> <f32>
            %102 = ascvec.mul %101, %101, %85 : <f32>
            %103 = ascvec.reduce_sum %102, %85 : <f32>
            %104 = ascvec.add %100, %103, %mask : <f32>
            %105 = ascvec.loadalign %arg24[%c320] : memref<576xf32, 1>, index -> <f32>
            %106 = ascvec.mul %105, %105, %85 : <f32>
            %107 = ascvec.reduce_sum %106, %85 : <f32>
            %108 = ascvec.add %104, %107, %mask : <f32>
            %109 = ascvec.loadalign %arg24[%c384] : memref<576xf32, 1>, index -> <f32>
            %110 = ascvec.mul %109, %109, %85 : <f32>
            %111 = ascvec.reduce_sum %110, %85 : <f32>
            %112 = ascvec.add %108, %111, %mask : <f32>
            %113 = ascvec.loadalign %arg24[%c448] : memref<576xf32, 1>, index -> <f32>
            %114 = ascvec.mul %113, %113, %85 : <f32>
            %115 = ascvec.reduce_sum %114, %85 : <f32>
            %116 = ascvec.add %112, %115, %mask : <f32>
            %117 = ascvec.mul_scalar %116, %cst, %mask : <f32>, f32
            %118 = ascvec.add_scalar %117, %arg25, %mask : <f32>, f32
            %119 = ascvec.sqrt %118, %mask : <f32>
            ascvec.storealign_1st %arg26[%c0_46], %119 : memref<64xf32, 1>, index, <f32>
          }
          ascvec.mem_bar vst_vld
        } {cannir.vf_mode = "simd"}
        %44 = cannir.extract_layout %arg9 : !cannir.tensor<<gm, bf16>, <<(-1, 64)>, <(64, 1)>>, <<(-1, 64)>, <(64, 1)>>, identity>
        %45 = cannir.extract_shape %44 : <<(-1, 64)>, <(64, 1)>> -> <(-1, 64)>
        %46 = cannir.extract_dim %45[0] : !cannir.shape<(-1, 64)>
        %47 = arith.minsi %arg22, %46 : i64
        %48 = arith.subi %46, %47 : i64
        %49 = arith.minsi %48, %c1_i64 : i64
        %50 = arith.cmpi sle, %arg22, %46 : i64
        %51 = arith.select %50, %49, %c0_i64 : i64
        %52 = arith.muli %arg22, %c128_i64 : i64
        %53 = arith.muli %51, %c128_i64 : i64
        %54 = arith.cmpi eq, %53, %c0_i64 : i64
        %55 = arith.cmpi ne, %53, %c0_i64 : i64
        %56 = arith.extui %55 : i1 to i64
        %57 = arith.select %54, %c2_i64, %53 : i64
        %58 = ascscalar.bitcast_memref %alloc_29 byte_offset = %c0_i64 : memref<64xbf16, 1> -> memref<?xbf16, 1>
        ascvec.get_buf mte2, %c3_i64, 0 {cannir.buffer.generated_sync}
        "ascvec.copy_gm2ub_plan"(%58, %alloc_4, %c0_i32, %c0_i32, %c0_i64, %52, %56, %57, %c0_i64, %57, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = 15200 : i64, ub_capacity = 128 : i64}> : (memref<?xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
        ascvec.release_buf mte2, %c3_i64, 0 {cannir.buffer.generated_sync}
        ascvec.get_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
        ascvec.vf(%alloc_29, %alloc_31 : memref<64xbf16, 1>, memref<128xf32, 1>) {
        ^bb0(%arg24: memref<64xbf16, 1>, %arg25: memref<128xf32, 1>):
          %c0_46 = arith.constant 0 : index
          %c0_47 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          scf.for %arg26 = %c0_47 to %c1 step %c1 {
            %83 = ascvec.loadalign_unpack %arg24[%c0_46] {mode = "b16_to_b32"} : memref<64xbf16, 1>, index -> <bf16>
            %84 = ascvec.create_mask : !ascvec.mask
            %85 = ascvec.cast %83, %84 {rounding = 6 : i32} : <bf16> -> <f32>
            ascvec.storealign %arg25[%c0_46], %85, %84 : memref<128xf32, 1>, index, <f32>
          }
          ascvec.mem_bar vst_vld
        } {cannir.vf_mode = "simd"}
        ascvec.release_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
        %59 = cannir.extract_layout %arg10 : !cannir.tensor<<gm, bf16>, <<(-1, 64)>, <(64, 1)>>, <<(-1, 64)>, <(64, 1)>>, identity>
        %60 = cannir.extract_shape %59 : <<(-1, 64)>, <(64, 1)>> -> <(-1, 64)>
        %61 = cannir.extract_dim %60[0] : !cannir.shape<(-1, 64)>
        %62 = arith.minsi %arg22, %61 : i64
        %63 = arith.subi %61, %62 : i64
        %64 = arith.minsi %63, %c1_i64 : i64
        %65 = arith.cmpi sle, %arg22, %61 : i64
        %66 = arith.select %65, %64, %c0_i64 : i64
        %67 = arith.muli %66, %c128_i64 : i64
        %68 = arith.cmpi eq, %67, %c0_i64 : i64
        %69 = arith.cmpi ne, %67, %c0_i64 : i64
        %70 = arith.extui %69 : i1 to i64
        %71 = arith.select %68, %c2_i64, %67 : i64
        %72 = ascscalar.bitcast_memref %alloc_30 byte_offset = %c0_i64 : memref<64xbf16, 1> -> memref<?xbf16, 1>
        ascvec.get_buf mte2, %c4_i64, 0 {cannir.buffer.generated_sync}
        "ascvec.copy_gm2ub_plan"(%72, %alloc_5, %c0_i32, %c0_i32, %c0_i64, %52, %70, %71, %c0_i64, %71, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = 15328 : i64, ub_capacity = 128 : i64}> : (memref<?xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
        ascvec.release_buf mte2, %c4_i64, 0 {cannir.buffer.generated_sync}
        ascvec.get_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
        ascvec.vf(%alloc_30, %alloc_32 : memref<64xbf16, 1>, memref<128xf32, 1>) {
        ^bb0(%arg24: memref<64xbf16, 1>, %arg25: memref<128xf32, 1>):
          %c0_46 = arith.constant 0 : index
          %c0_47 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          scf.for %arg26 = %c0_47 to %c1 step %c1 {
            %83 = ascvec.loadalign_unpack %arg24[%c0_46] {mode = "b16_to_b32"} : memref<64xbf16, 1>, index -> <bf16>
            %84 = ascvec.create_mask : !ascvec.mask
            %85 = ascvec.cast %83, %84 {rounding = 6 : i32} : <bf16> -> <f32>
            ascvec.storealign %arg25[%c0_46], %85, %84 : memref<128xf32, 1>, index, <f32>
          }
          ascvec.mem_bar vst_vld
        } {cannir.vf_mode = "simd"}
        ascvec.release_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
        ascvec.vf(%43, %alloc_33 : memref<576xf32, 1>, memref<128xf32, 1>) {
        ^bb0(%arg24: memref<576xf32, 1>, %arg25: memref<128xf32, 1>):
          %c0_46 = arith.constant 0 : index
          %c512 = arith.constant 512 : index
          %c0_47 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          scf.for %arg26 = %c0_47 to %c1 step %c1 {
            %83 = ascvec.loadalign %arg24[%c512] : memref<576xf32, 1>, index -> <f32>
            %84 = ascvec.create_mask : !ascvec.mask
            ascvec.storealign %arg25[%c0_46], %83, %84 : memref<128xf32, 1>, index, <f32>
          }
          ascvec.mem_bar vst_vld
        } {cannir.vf_mode = "simd"}
        %73 = arith.cmpi eq, %arg21, %c1_i64 : i64
        scf.if %73 {
          ascvec.vf(%alloc_32, %alloc_31, %alloc_33, %alloc_34 : memref<128xf32, 1>, memref<128xf32, 1>, memref<128xf32, 1>, memref<64xf32, 1>) {
          ^bb0(%arg24: memref<128xf32, 1>, %arg25: memref<128xf32, 1>, %arg26: memref<128xf32, 1>, %arg27: memref<64xf32, 1>):
            %c32 = arith.constant 32 : index
            %c0_46 = arith.constant 0 : index
            %c32_i32 = arith.constant 32 : i32
            %c0_47 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            scf.for %arg28 = %c0_47 to %c1 step %c1 {
              %mask, %remaining_out = ascvec.update_mask %c32_i32 : !ascvec.mask
              %83 = ascvec.loadalign %arg24[%c0_46] : memref<128xf32, 1>, index -> <f32>
              %84 = ascvec.loadalign %arg24[%c32] : memref<128xf32, 1>, index -> <f32>
              %85 = ascvec.loadalign %arg25[%c0_46] : memref<128xf32, 1>, index -> <f32>
              %86 = ascvec.loadalign %arg25[%c32] : memref<128xf32, 1>, index -> <f32>
              %result0, %result1 = ascvec.loadalign_deintlv %arg26[%c0_46] {dintlv_width = "b32"} : memref<128xf32, 1>, index -> <f32>, <f32>
              %87 = ascvec.mul %result0, %83, %mask : <f32>
              %88 = ascvec.mul %result1, %85, %mask : <f32>
              %89 = ascvec.sub %87, %88, %mask : <f32>
              %90 = ascvec.mul %result1, %84, %mask : <f32>
              %91 = ascvec.mul %result0, %86, %mask : <f32>
              %92 = ascvec.add %90, %91, %mask : <f32>
              ascvec.storealign %arg27[%c0_46], %89, %mask : memref<64xf32, 1>, index, <f32>
              ascvec.storealign %arg27[%c32], %92, %mask : memref<64xf32, 1>, index, <f32>
            }
            ascvec.mem_bar vst_vld
          } {cannir.vf_mode = "simd"}
        } else {
          ascvec.vf(%alloc_32, %alloc_31, %alloc_33, %alloc_34 : memref<128xf32, 1>, memref<128xf32, 1>, memref<128xf32, 1>, memref<64xf32, 1>) {
          ^bb0(%arg24: memref<128xf32, 1>, %arg25: memref<128xf32, 1>, %arg26: memref<128xf32, 1>, %arg27: memref<64xf32, 1>):
            %c32 = arith.constant 32 : index
            %c0_46 = arith.constant 0 : index
            %c32_i32 = arith.constant 32 : i32
            %c0_47 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            scf.for %arg28 = %c0_47 to %c1 step %c1 {
              %mask, %remaining_out = ascvec.update_mask %c32_i32 : !ascvec.mask
              %83 = ascvec.loadalign %arg24[%c0_46] : memref<128xf32, 1>, index -> <f32>
              %84 = ascvec.loadalign %arg24[%c32] : memref<128xf32, 1>, index -> <f32>
              %85 = ascvec.loadalign %arg25[%c0_46] : memref<128xf32, 1>, index -> <f32>
              %86 = ascvec.loadalign %arg25[%c32] : memref<128xf32, 1>, index -> <f32>
              %87 = ascvec.loadalign %arg26[%c0_46] : memref<128xf32, 1>, index -> <f32>
              %88 = ascvec.loadalign %arg26[%c32] : memref<128xf32, 1>, index -> <f32>
              %89 = ascvec.mul %87, %83, %mask : <f32>
              %90 = ascvec.mul %88, %85, %mask : <f32>
              %91 = ascvec.sub %89, %90, %mask : <f32>
              %92 = ascvec.mul %88, %84, %mask : <f32>
              %93 = ascvec.mul %87, %86, %mask : <f32>
              %94 = ascvec.add %92, %93, %mask : <f32>
              ascvec.storealign %arg27[%c0_46], %91, %mask : memref<64xf32, 1>, index, <f32>
              ascvec.storealign %arg27[%c32], %94, %mask : memref<64xf32, 1>, index, <f32>
            }
            ascvec.mem_bar vst_vld
          } {cannir.vf_mode = "simd"}
        }
        %74 = ascscalar.memref_address %alloc_26 : memref<656xf8E4M3FN, 1>
        %75 = ascscalar.memref_address %alloc_27 : memref<656xf8E4M3FN, 1>
        %76 = arith.select %19, %74, %75 : i64
        %77 = ascscalar.bind_local_ptr %76 : memref<656xf8E4M3FN, 1>
        ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
        %78 = ascscalar.bitcast_memref %77 byte_offset = %c0_i64 : memref<656xf8E4M3FN, 1> -> memref<?xf32, 1>
        %79 = ascscalar.bitcast_memref %77 byte_offset = %c0_i64 : memref<656xf8E4M3FN, 1> -> memref<?xbf16, 1>
        ascvec.vf(%alloc_28, %43, %alloc_25, %arg19, %78, %77, %alloc_34, %79 : memref<64xf32, 1>, memref<576xf32, 1>, memref<512xf32, 1>, f32, memref<?xf32, 1>, memref<656xf8E4M3FN, 1>, memref<64xf32, 1>, memref<?xbf16, 1>) {
        ^bb0(%arg24: memref<64xf32, 1>, %arg25: memref<576xf32, 1>, %arg26: memref<512xf32, 1>, %arg27: f32, %arg28: memref<?xf32, 1>, %arg29: memref<656xf8E4M3FN, 1>, %arg30: memref<64xf32, 1>, %arg31: memref<?xbf16, 1>):
          %c3 = arith.constant 3 : index
          %c448 = arith.constant 448 : index
          %c384 = arith.constant 384 : index
          %c2 = arith.constant 2 : index
          %c320 = arith.constant 320 : index
          %c1 = arith.constant 1 : index
          %c192 = arith.constant 192 : index
          %c128 = arith.constant 128 : index
          %c64 = arith.constant 64 : index
          %c64_i32 = arith.constant 64 : i32
          %c160 = arith.constant 160 : index
          %c4_i32 = arith.constant 4 : i32
          %c256 = arith.constant 256 : index
          %cst = arith.constant 9.99999974E-5 : f32
          %c0_46 = arith.constant 0 : index
          %cst_47 = arith.constant 4.480000e+02 : f32
          %c1_i32_48 = arith.constant 1 : i32
          %mask, %remaining_out = ascvec.update_mask %c1_i32_48 : !ascvec.mask
          %83 = ascvec.duplicate_scalar %cst_47, %mask : f32 -> <f32>
          %84 = ascvec.loadalign_brc %arg24[%c0_46] : memref<64xf32, 1>, index -> <f32>
          %85 = ascvec.duplicate_scalar %cst, %mask : f32 -> <f32>
          %86 = ascvec.loadalign %arg25[%c0_46] : memref<576xf32, 1>, index -> <f32>
          %87 = ascvec.create_mask : !ascvec.mask
          %88 = ascvec.div %86, %84, %87 {precision_mode = true} : <f32>
          %89 = ascvec.loadalign %arg26[%c0_46] : memref<512xf32, 1>, index -> <f32>
          %90 = ascvec.mul %88, %89, %87 : <f32>
          %91 = ascvec.mul_scalar %90, %arg27, %87 : <f32>, f32
          %92 = ascvec.abs %91, %87 : <f32>
          %93 = ascvec.reduce_max %92, %87 : <f32>
          %94 = ascvec.max %85, %93, %mask : <f32>
          %95 = ascvec.loadalign %arg25[%c64] : memref<576xf32, 1>, index -> <f32>
          %96 = ascvec.div %95, %84, %87 {precision_mode = true} : <f32>
          %97 = ascvec.loadalign %arg26[%c64] : memref<512xf32, 1>, index -> <f32>
          %98 = ascvec.mul %96, %97, %87 : <f32>
          %99 = ascvec.mul_scalar %98, %arg27, %87 : <f32>, f32
          %100 = ascvec.abs %99, %87 : <f32>
          %101 = ascvec.reduce_max %100, %87 : <f32>
          %102 = ascvec.max %94, %101, %mask : <f32>
          %103 = ascvec.div %102, %83, %mask {precision_mode = true} : <f32>
          ascvec.storealign_1st %arg24[%c0_46], %103 : memref<64xf32, 1>, index, <f32>
          %104 = ascvec.loadalign %arg25[%c128] : memref<576xf32, 1>, index -> <f32>
          %105 = ascvec.div %104, %84, %87 {precision_mode = true} : <f32>
          %106 = ascvec.loadalign %arg26[%c128] : memref<512xf32, 1>, index -> <f32>
          %107 = ascvec.mul %105, %106, %87 : <f32>
          %108 = ascvec.mul_scalar %107, %arg27, %87 : <f32>, f32
          %109 = ascvec.abs %108, %87 : <f32>
          %110 = ascvec.reduce_max %109, %87 : <f32>
          %111 = ascvec.max %85, %110, %mask : <f32>
          %112 = ascvec.loadalign %arg25[%c192] : memref<576xf32, 1>, index -> <f32>
          %113 = ascvec.div %112, %84, %87 {precision_mode = true} : <f32>
          %114 = ascvec.loadalign %arg26[%c192] : memref<512xf32, 1>, index -> <f32>
          %115 = ascvec.mul %113, %114, %87 : <f32>
          %116 = ascvec.mul_scalar %115, %arg27, %87 : <f32>, f32
          %117 = ascvec.abs %116, %87 : <f32>
          %118 = ascvec.reduce_max %117, %87 : <f32>
          %119 = ascvec.max %111, %118, %mask : <f32>
          %120 = ascvec.div %119, %83, %mask {precision_mode = true} : <f32>
          ascvec.storealign_1st %arg24[%c1], %120 : memref<64xf32, 1>, index, <f32>
          %121 = ascvec.loadalign %arg25[%c256] : memref<576xf32, 1>, index -> <f32>
          %122 = ascvec.div %121, %84, %87 {precision_mode = true} : <f32>
          %123 = ascvec.loadalign %arg26[%c256] : memref<512xf32, 1>, index -> <f32>
          %124 = ascvec.mul %122, %123, %87 : <f32>
          %125 = ascvec.mul_scalar %124, %arg27, %87 : <f32>, f32
          %126 = ascvec.abs %125, %87 : <f32>
          %127 = ascvec.reduce_max %126, %87 : <f32>
          %128 = ascvec.max %85, %127, %mask : <f32>
          %129 = ascvec.loadalign %arg25[%c320] : memref<576xf32, 1>, index -> <f32>
          %130 = ascvec.div %129, %84, %87 {precision_mode = true} : <f32>
          %131 = ascvec.loadalign %arg26[%c320] : memref<512xf32, 1>, index -> <f32>
          %132 = ascvec.mul %130, %131, %87 : <f32>
          %133 = ascvec.mul_scalar %132, %arg27, %87 : <f32>, f32
          %134 = ascvec.abs %133, %87 : <f32>
          %135 = ascvec.reduce_max %134, %87 : <f32>
          %136 = ascvec.max %128, %135, %mask : <f32>
          %137 = ascvec.div %136, %83, %mask {precision_mode = true} : <f32>
          ascvec.storealign_1st %arg24[%c2], %137 : memref<64xf32, 1>, index, <f32>
          %138 = ascvec.loadalign %arg25[%c384] : memref<576xf32, 1>, index -> <f32>
          %139 = ascvec.div %138, %84, %87 {precision_mode = true} : <f32>
          %140 = ascvec.loadalign %arg26[%c384] : memref<512xf32, 1>, index -> <f32>
          %141 = ascvec.mul %139, %140, %87 : <f32>
          %142 = ascvec.mul_scalar %141, %arg27, %87 : <f32>, f32
          %143 = ascvec.abs %142, %87 : <f32>
          %144 = ascvec.reduce_max %143, %87 : <f32>
          %145 = ascvec.max %85, %144, %mask : <f32>
          %146 = ascvec.loadalign %arg25[%c448] : memref<576xf32, 1>, index -> <f32>
          %147 = ascvec.div %146, %84, %87 {precision_mode = true} : <f32>
          %148 = ascvec.loadalign %arg26[%c448] : memref<512xf32, 1>, index -> <f32>
          %149 = ascvec.mul %147, %148, %87 : <f32>
          %150 = ascvec.mul_scalar %149, %arg27, %87 : <f32>, f32
          %151 = ascvec.abs %150, %87 : <f32>
          %152 = ascvec.reduce_max %151, %87 : <f32>
          %153 = ascvec.max %145, %152, %mask : <f32>
          %154 = ascvec.div %153, %83, %mask {precision_mode = true} : <f32>
          ascvec.storealign_1st %arg24[%c3], %154 : memref<64xf32, 1>, index, <f32>
          ascvec.mem_bar vst_vld
          %c0_49 = arith.constant 0 : index
          %c1_50 = arith.constant 1 : index
          scf.for %arg32 = %c0_49 to %c1_50 step %c1_50 {
            %155 = ascvec.loadalign %arg24[%c0_46] : memref<64xf32, 1>, index -> <f32>
            %mask_51, %remaining_out_52 = ascvec.update_mask %c4_i32 : !ascvec.mask
            ascvec.storealign %arg28[%c160], %155, %mask_51 : memref<?xf32, 1>, index, <f32>
            %156 = ascvec.loadalign_brc %arg24[%c0_46] : memref<64xf32, 1>, index -> <f32>
            %157 = ascvec.loadalign %arg25[%c0_46] : memref<576xf32, 1>, index -> <f32>
            %158 = ascvec.div %157, %84, %87 {precision_mode = true} : <f32>
            %159 = ascvec.loadalign %arg26[%c0_46] : memref<512xf32, 1>, index -> <f32>
            %160 = ascvec.mul %158, %159, %87 : <f32>
            %161 = ascvec.mul_scalar %160, %arg27, %87 : <f32>, f32
            %162 = ascvec.div %161, %156, %87 {precision_mode = true} : <f32>
            %163 = ascvec.cast %162, %87 {reg_layout = 0 : i32} : <f32> -> <f8E4M3FN>
            ascvec.storealign_pack %arg29[%c0_46], %163, %87 {mode = "b32_to_b8"} : memref<656xf8E4M3FN, 1>, index, <f8E4M3FN>
            %164 = ascvec.loadalign %arg25[%c64] : memref<576xf32, 1>, index -> <f32>
            %165 = ascvec.div %164, %84, %87 {precision_mode = true} : <f32>
            %166 = ascvec.loadalign %arg26[%c64] : memref<512xf32, 1>, index -> <f32>
            %167 = ascvec.mul %165, %166, %87 : <f32>
            %168 = ascvec.mul_scalar %167, %arg27, %87 : <f32>, f32
            %169 = ascvec.div %168, %156, %87 {precision_mode = true} : <f32>
            %170 = ascvec.cast %169, %87 {reg_layout = 0 : i32} : <f32> -> <f8E4M3FN>
            ascvec.storealign_pack %arg29[%c64], %170, %87 {mode = "b32_to_b8"} : memref<656xf8E4M3FN, 1>, index, <f8E4M3FN>
            %171 = ascvec.loadalign_brc %arg24[%c1] : memref<64xf32, 1>, index -> <f32>
            %172 = ascvec.loadalign %arg25[%c128] : memref<576xf32, 1>, index -> <f32>
            %173 = ascvec.div %172, %84, %87 {precision_mode = true} : <f32>
            %174 = ascvec.loadalign %arg26[%c128] : memref<512xf32, 1>, index -> <f32>
            %175 = ascvec.mul %173, %174, %87 : <f32>
            %176 = ascvec.mul_scalar %175, %arg27, %87 : <f32>, f32
            %177 = ascvec.div %176, %171, %87 {precision_mode = true} : <f32>
            %178 = ascvec.cast %177, %87 {reg_layout = 0 : i32} : <f32> -> <f8E4M3FN>
            ascvec.storealign_pack %arg29[%c128], %178, %87 {mode = "b32_to_b8"} : memref<656xf8E4M3FN, 1>, index, <f8E4M3FN>
            %179 = ascvec.loadalign %arg25[%c192] : memref<576xf32, 1>, index -> <f32>
            %180 = ascvec.div %179, %84, %87 {precision_mode = true} : <f32>
            %181 = ascvec.loadalign %arg26[%c192] : memref<512xf32, 1>, index -> <f32>
            %182 = ascvec.mul %180, %181, %87 : <f32>
            %183 = ascvec.mul_scalar %182, %arg27, %87 : <f32>, f32
            %184 = ascvec.div %183, %171, %87 {precision_mode = true} : <f32>
            %185 = ascvec.cast %184, %87 {reg_layout = 0 : i32} : <f32> -> <f8E4M3FN>
            ascvec.storealign_pack %arg29[%c192], %185, %87 {mode = "b32_to_b8"} : memref<656xf8E4M3FN, 1>, index, <f8E4M3FN>
            %186 = ascvec.loadalign_brc %arg24[%c2] : memref<64xf32, 1>, index -> <f32>
            %187 = ascvec.loadalign %arg25[%c256] : memref<576xf32, 1>, index -> <f32>
            %188 = ascvec.div %187, %84, %87 {precision_mode = true} : <f32>
            %189 = ascvec.loadalign %arg26[%c256] : memref<512xf32, 1>, index -> <f32>
            %190 = ascvec.mul %188, %189, %87 : <f32>
            %191 = ascvec.mul_scalar %190, %arg27, %87 : <f32>, f32
            %192 = ascvec.div %191, %186, %87 {precision_mode = true} : <f32>
            %193 = ascvec.cast %192, %87 {reg_layout = 0 : i32} : <f32> -> <f8E4M3FN>
            ascvec.storealign_pack %arg29[%c256], %193, %87 {mode = "b32_to_b8"} : memref<656xf8E4M3FN, 1>, index, <f8E4M3FN>
            %194 = ascvec.loadalign %arg25[%c320] : memref<576xf32, 1>, index -> <f32>
            %195 = ascvec.div %194, %84, %87 {precision_mode = true} : <f32>
            %196 = ascvec.loadalign %arg26[%c320] : memref<512xf32, 1>, index -> <f32>
            %197 = ascvec.mul %195, %196, %87 : <f32>
            %198 = ascvec.mul_scalar %197, %arg27, %87 : <f32>, f32
            %199 = ascvec.div %198, %186, %87 {precision_mode = true} : <f32>
            %200 = ascvec.cast %199, %87 {reg_layout = 0 : i32} : <f32> -> <f8E4M3FN>
            ascvec.storealign_pack %arg29[%c320], %200, %87 {mode = "b32_to_b8"} : memref<656xf8E4M3FN, 1>, index, <f8E4M3FN>
            %201 = ascvec.loadalign_brc %arg24[%c3] : memref<64xf32, 1>, index -> <f32>
            %202 = ascvec.loadalign %arg25[%c384] : memref<576xf32, 1>, index -> <f32>
            %203 = ascvec.div %202, %84, %87 {precision_mode = true} : <f32>
            %204 = ascvec.loadalign %arg26[%c384] : memref<512xf32, 1>, index -> <f32>
            %205 = ascvec.mul %203, %204, %87 : <f32>
            %206 = ascvec.mul_scalar %205, %arg27, %87 : <f32>, f32
            %207 = ascvec.div %206, %201, %87 {precision_mode = true} : <f32>
            %208 = ascvec.cast %207, %87 {reg_layout = 0 : i32} : <f32> -> <f8E4M3FN>
            ascvec.storealign_pack %arg29[%c384], %208, %87 {mode = "b32_to_b8"} : memref<656xf8E4M3FN, 1>, index, <f8E4M3FN>
            %209 = ascvec.loadalign %arg25[%c448] : memref<576xf32, 1>, index -> <f32>
            %210 = ascvec.div %209, %84, %87 {precision_mode = true} : <f32>
            %211 = ascvec.loadalign %arg26[%c448] : memref<512xf32, 1>, index -> <f32>
            %212 = ascvec.mul %210, %211, %87 : <f32>
            %213 = ascvec.mul_scalar %212, %arg27, %87 : <f32>, f32
            %214 = ascvec.div %213, %201, %87 {precision_mode = true} : <f32>
            %215 = ascvec.cast %214, %87 {reg_layout = 0 : i32} : <f32> -> <f8E4M3FN>
            ascvec.storealign_pack %arg29[%c448], %215, %87 {mode = "b32_to_b8"} : memref<656xf8E4M3FN, 1>, index, <f8E4M3FN>
            %mask_53, %remaining_out_54 = ascvec.update_mask %c64_i32 : !ascvec.mask
            %216 = ascvec.loadalign %arg30[%c0_46] : memref<64xf32, 1>, index -> <f32>
            %217 = ascvec.cast %216, %87 : <f32> -> <bf16>
            ascvec.storealign_pack %arg31[%c256], %217, %mask_53 {mode = "b32_to_b16"} : memref<?xbf16, 1>, index, <bf16>
          }
          ascvec.mem_bar vst_vld
        } {cannir.vf_mode = "simd"}
        ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
        %80 = arith.index_cast %arg22 : i64 to index
        %81 = memref.load %alloc_3[%80] : memref<1xi64>
        %82 = arith.cmpi sge, %81, %c0_i64 : i64
        scf.if %82 {
          %83 = arith.muli %81, %c164_i64 : i64
          ascsimt.vf(%77, %83, %alloc_2 : memref<656xf8E4M3FN, 1>, i64, memref<1xf8E4M3FN>) {
          ^bb0(%arg24: memref<656xf8E4M3FN, 1>, %arg25: i64, %arg26: memref<1xf8E4M3FN>):
            %c164_i32 = arith.constant 164 : i32
            %c0_i64_46 = arith.constant 0 : i64
            %84 = ascsimt.thread_idx {ascsimt.scope = 14 : i64, axis = "x", cannir.vf_threads = 256 : i64} : i32
            %85 = arith.cmpi slt, %84, %c164_i32 {cannir.vf_threads = 256 : i64} : i32
            scf.if %85 {
              %86 = arith.extsi %84 : i32 to i64
              %87 = ascscalar.bitcast_memref %arg24 byte_offset = %c0_i64_46 : memref<656xf8E4M3FN, 1> -> memref<?xf32, 1>
              %88 = arith.index_cast %84 : i32 to index
              %89 = ascsimt.load %87[%88] : memref<?xf32, 1> -> f32
              %90 = arith.addi %arg25, %86 : i64
              %91 = ascscalar.bitcast_memref %arg26 byte_offset = %c0_i64_46 : memref<1xf8E4M3FN> -> memref<?xf32>
              %92 = arith.index_cast %90 : i64 to index
              ascsimt.store %91[%92], %89 : memref<?xf32>, f32
            } {cannir.vf_threads = 256 : i64}
          } {threads = 256 : i64}
        }
        scf.yield %26 : i64
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
      %17 = scf.for %arg22 = %7 to %15 step %12 iter_args(%arg23 = %c0_i64) -> (i64)  : i64 {
        %18 = arith.remsi %arg22, %c2_i64 : i64
        %19 = arith.cmpi ne, %18, %c0_i64 : i64
        %20 = arith.cmpi slt, %18, %c0_i64 : i64
        %21 = arith.andi %19, %20 : i1
        %22 = arith.divsi %arg22, %c2_i64 : i64
        %23 = arith.extui %21 : i1 to i64
        %24 = arith.subi %22, %23 : i64
        %25 = arith.select %21, %c2_i64, %c0_i64 : i64
        %26 = arith.addi %18, %25 overflow<nsw> : i64
        %27 = cannir.extract_layout %arg9 : !cannir.tensor<<gm, bf16>, <<(-1, 64)>, <(64, 1)>>, <<(-1, 64)>, <(64, 1)>>, identity>
        %28 = cannir.extract_shape %27 : <<(-1, 64)>, <(64, 1)>> -> <(-1, 64)>
        %29 = cannir.extract_dim %28[0] : !cannir.shape<(-1, 64)>
        %30 = arith.minsi %24, %29 : i64
        %31 = arith.subi %29, %30 : i64
        %32 = arith.minsi %31, %c1_i64 : i64
        %33 = arith.cmpi sle, %24, %29 : i64
        %34 = arith.select %33, %32, %c0_i64 : i64
        %35 = arith.muli %24, %c128_i64 : i64
        %36 = arith.muli %34, %c128_i64 : i64
        %37 = arith.cmpi eq, %36, %c0_i64 : i64
        %38 = arith.cmpi ne, %36, %c0_i64 : i64
        %39 = arith.extui %38 : i1 to i64
        %40 = arith.select %37, %c2_i64, %36 : i64
        %41 = ascscalar.bitcast_memref %alloc_40 byte_offset = %c0_i64 : memref<64xbf16, 1> -> memref<?xbf16, 1>
        ascvec.get_buf mte2, %c4_i64, 0 {cannir.buffer.generated_sync}
        "ascvec.copy_gm2ub_plan"(%41, %alloc_4, %c0_i32, %c0_i32, %c0_i64, %35, %39, %40, %c0_i64, %40, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = 768 : i64, ub_capacity = 128 : i64}> : (memref<?xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
        ascvec.release_buf mte2, %c4_i64, 0 {cannir.buffer.generated_sync}
        ascvec.get_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
        ascvec.vf(%alloc_40, %alloc_42 : memref<64xbf16, 1>, memref<128xf32, 1>) {
        ^bb0(%arg24: memref<64xbf16, 1>, %arg25: memref<128xf32, 1>):
          %c0_46 = arith.constant 0 : index
          %c0_47 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          scf.for %arg26 = %c0_47 to %c1 step %c1 {
            %106 = ascvec.loadalign_unpack %arg24[%c0_46] {mode = "b16_to_b32"} : memref<64xbf16, 1>, index -> <bf16>
            %107 = ascvec.create_mask : !ascvec.mask
            %108 = ascvec.cast %106, %107 {rounding = 6 : i32} : <bf16> -> <f32>
            ascvec.storealign %arg25[%c0_46], %108, %107 : memref<128xf32, 1>, index, <f32>
          }
          ascvec.mem_bar vst_vld
        } {cannir.vf_mode = "simd"}
        ascvec.release_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
        %42 = cannir.extract_layout %arg10 : !cannir.tensor<<gm, bf16>, <<(-1, 64)>, <(64, 1)>>, <<(-1, 64)>, <(64, 1)>>, identity>
        %43 = cannir.extract_shape %42 : <<(-1, 64)>, <(64, 1)>> -> <(-1, 64)>
        %44 = cannir.extract_dim %43[0] : !cannir.shape<(-1, 64)>
        %45 = arith.minsi %24, %44 : i64
        %46 = arith.subi %44, %45 : i64
        %47 = arith.minsi %46, %c1_i64 : i64
        %48 = arith.cmpi sle, %24, %44 : i64
        %49 = arith.select %48, %47, %c0_i64 : i64
        %50 = arith.muli %49, %c128_i64 : i64
        %51 = arith.cmpi eq, %50, %c0_i64 : i64
        %52 = arith.cmpi ne, %50, %c0_i64 : i64
        %53 = arith.extui %52 : i1 to i64
        %54 = arith.select %51, %c2_i64, %50 : i64
        %55 = ascscalar.bitcast_memref %alloc_41 byte_offset = %c0_i64 : memref<64xbf16, 1> -> memref<?xbf16, 1>
        ascvec.get_buf mte2, %c5_i64, 0 {cannir.buffer.generated_sync}
        "ascvec.copy_gm2ub_plan"(%55, %alloc_5, %c0_i32, %c0_i32, %c0_i64, %35, %53, %54, %c0_i64, %54, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = 896 : i64, ub_capacity = 128 : i64}> : (memref<?xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
        ascvec.release_buf mte2, %c5_i64, 0 {cannir.buffer.generated_sync}
        ascvec.get_buf v, %c5_i64, 0 {cannir.buffer.generated_sync}
        ascvec.vf(%alloc_41, %alloc_43 : memref<64xbf16, 1>, memref<128xf32, 1>) {
        ^bb0(%arg24: memref<64xbf16, 1>, %arg25: memref<128xf32, 1>):
          %c0_46 = arith.constant 0 : index
          %c0_47 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          scf.for %arg26 = %c0_47 to %c1 step %c1 {
            %106 = ascvec.loadalign_unpack %arg24[%c0_46] {mode = "b16_to_b32"} : memref<64xbf16, 1>, index -> <bf16>
            %107 = ascvec.create_mask : !ascvec.mask
            %108 = ascvec.cast %106, %107 {rounding = 6 : i32} : <bf16> -> <f32>
            ascvec.storealign %arg25[%c0_46], %108, %107 : memref<128xf32, 1>, index, <f32>
          }
          ascvec.mem_bar vst_vld
        } {cannir.vf_mode = "simd"}
        ascvec.release_buf v, %c5_i64, 0 {cannir.buffer.generated_sync}
        %56 = arith.muli %26, %c256_i64 overflow<nsw> : i64
        %57 = arith.addi %56, %c192_i64 overflow<nsw> : i64
        %58 = arith.muli %26, %c576_i64 overflow<nsw> : i64
        %59 = arith.addi %58, %c512_i64 overflow<nsw> : i64
        %60 = arith.cmpi eq, %arg23, %c0_i64 : i64
        %61 = ascscalar.memref_address %alloc_35 : memref<64xbf16, 1>
        %62 = ascscalar.memref_address %alloc_36 : memref<64xbf16, 1>
        %63 = arith.select %60, %61, %62 : i64
        %64 = arith.cmpi ne, %arg23, %c0_i64 : i64
        %65 = arith.extui %64 : i1 to i64
        %66 = arith.addi %arg23, %c1_i64 overflow<nsw> : i64
        %67 = arith.remui %66, %c2_i64 : i64
        %68 = arith.remsi %57, %c64_i64 : i64
        %69 = arith.cmpi ne, %68, %c0_i64 : i64
        %70 = arith.cmpi slt, %68, %c0_i64 : i64
        %71 = arith.andi %69, %70 : i1
        %72 = arith.divsi %57, %c64_i64 : i64
        %73 = arith.extui %71 : i1 to i64
        %74 = arith.subi %72, %73 : i64
        %75 = arith.minsi %24, %c8_i64 : i64
        %76 = arith.subi %c8_i64, %75 : i64
        %77 = arith.minsi %76, %c1_i64 : i64
        %78 = arith.cmpi sle, %24, %c8_i64 : i64
        %79 = arith.select %78, %77, %c0_i64 : i64
        %80 = arith.muli %24, %c1152_i64 : i64
        %81 = arith.muli %74, %c128_i64 : i64
        %82 = arith.addi %80, %81 : i64
        %83 = ascscalar.bind_local_ptr %63 : memref<1xbf16, 1>
        ascvec.get_buf mte2, %65, 0 {cannir.buffer.generated_sync}
        "ascvec.copy_gm2ub_plan"(%83, %alloc_10, %c0_i32, %c0_i32, %c0_i64, %82, %79, %c128_i64, %c1152_i64, %c128_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 9216 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<4608xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
        ascvec.release_buf mte2, %65, 0 {cannir.buffer.generated_sync}
        %84 = ascscalar.bind_local_ptr %63 : memref<64xbf16, 1>
        ascvec.get_buf v, %65, 0 {cannir.buffer.generated_sync}
        ascvec.vf(%84, %alloc_37 : memref<64xbf16, 1>, memref<64xf32, 1>) {
        ^bb0(%arg24: memref<64xbf16, 1>, %arg25: memref<64xf32, 1>):
          %c0_46 = arith.constant 0 : index
          %c0_47 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          scf.for %arg26 = %c0_47 to %c1 step %c1 {
            %106 = ascvec.loadalign_unpack %arg24[%c0_46] {mode = "b16_to_b32"} : memref<64xbf16, 1>, index -> <bf16>
            %107 = ascvec.create_mask : !ascvec.mask
            %108 = ascvec.cast %106, %107 {rounding = 6 : i32} : <bf16> -> <f32>
            ascvec.storealign %arg25[%c0_46], %108, %107 : memref<64xf32, 1>, index, <f32>
          }
          ascvec.mem_bar vst_vld
        } {cannir.vf_mode = "simd"}
        ascvec.release_buf v, %65, 0 {cannir.buffer.generated_sync}
        ascvec.vf(%alloc_37, %alloc_44 : memref<64xf32, 1>, memref<128xf32, 1>) {
        ^bb0(%arg24: memref<64xf32, 1>, %arg25: memref<128xf32, 1>):
          %c0_46 = arith.constant 0 : index
          %c0_47 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          scf.for %arg26 = %c0_47 to %c1 step %c1 {
            %106 = ascvec.loadalign %arg24[%c0_46] : memref<64xf32, 1>, index -> <f32>
            %107 = ascvec.create_mask : !ascvec.mask
            ascvec.storealign %arg25[%c0_46], %106, %107 : memref<128xf32, 1>, index, <f32>
          }
          ascvec.mem_bar vst_vld
        } {cannir.vf_mode = "simd"}
        %85 = arith.cmpi eq, %arg21, %c1_i64 : i64
        scf.if %85 {
          ascvec.vf(%alloc_43, %alloc_42, %alloc_44, %alloc_45 : memref<128xf32, 1>, memref<128xf32, 1>, memref<128xf32, 1>, memref<64xf32, 1>) {
          ^bb0(%arg24: memref<128xf32, 1>, %arg25: memref<128xf32, 1>, %arg26: memref<128xf32, 1>, %arg27: memref<64xf32, 1>):
            %c32 = arith.constant 32 : index
            %c0_46 = arith.constant 0 : index
            %c32_i32 = arith.constant 32 : i32
            %c0_47 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            scf.for %arg28 = %c0_47 to %c1 step %c1 {
              %mask, %remaining_out = ascvec.update_mask %c32_i32 : !ascvec.mask
              %106 = ascvec.loadalign %arg24[%c0_46] : memref<128xf32, 1>, index -> <f32>
              %107 = ascvec.loadalign %arg24[%c32] : memref<128xf32, 1>, index -> <f32>
              %108 = ascvec.loadalign %arg25[%c0_46] : memref<128xf32, 1>, index -> <f32>
              %109 = ascvec.loadalign %arg25[%c32] : memref<128xf32, 1>, index -> <f32>
              %result0, %result1 = ascvec.loadalign_deintlv %arg26[%c0_46] {dintlv_width = "b32"} : memref<128xf32, 1>, index -> <f32>, <f32>
              %110 = ascvec.mul %result0, %106, %mask : <f32>
              %111 = ascvec.mul %result1, %108, %mask : <f32>
              %112 = ascvec.sub %110, %111, %mask : <f32>
              %113 = ascvec.mul %result1, %107, %mask : <f32>
              %114 = ascvec.mul %result0, %109, %mask : <f32>
              %115 = ascvec.add %113, %114, %mask : <f32>
              ascvec.storealign %arg27[%c0_46], %112, %mask : memref<64xf32, 1>, index, <f32>
              ascvec.storealign %arg27[%c32], %115, %mask : memref<64xf32, 1>, index, <f32>
            }
            ascvec.mem_bar vst_vld
          } {cannir.vf_mode = "simd"}
        } else {
          ascvec.vf(%alloc_43, %alloc_42, %alloc_44, %alloc_45 : memref<128xf32, 1>, memref<128xf32, 1>, memref<128xf32, 1>, memref<64xf32, 1>) {
          ^bb0(%arg24: memref<128xf32, 1>, %arg25: memref<128xf32, 1>, %arg26: memref<128xf32, 1>, %arg27: memref<64xf32, 1>):
            %c32 = arith.constant 32 : index
            %c0_46 = arith.constant 0 : index
            %c32_i32 = arith.constant 32 : i32
            %c0_47 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            scf.for %arg28 = %c0_47 to %c1 step %c1 {
              %mask, %remaining_out = ascvec.update_mask %c32_i32 : !ascvec.mask
              %106 = ascvec.loadalign %arg24[%c0_46] : memref<128xf32, 1>, index -> <f32>
              %107 = ascvec.loadalign %arg24[%c32] : memref<128xf32, 1>, index -> <f32>
              %108 = ascvec.loadalign %arg25[%c0_46] : memref<128xf32, 1>, index -> <f32>
              %109 = ascvec.loadalign %arg25[%c32] : memref<128xf32, 1>, index -> <f32>
              %110 = ascvec.loadalign %arg26[%c0_46] : memref<128xf32, 1>, index -> <f32>
              %111 = ascvec.loadalign %arg26[%c32] : memref<128xf32, 1>, index -> <f32>
              %112 = ascvec.mul %110, %106, %mask : <f32>
              %113 = ascvec.mul %111, %108, %mask : <f32>
              %114 = ascvec.sub %112, %113, %mask : <f32>
              %115 = ascvec.mul %111, %107, %mask : <f32>
              %116 = ascvec.mul %110, %109, %mask : <f32>
              %117 = ascvec.add %115, %116, %mask : <f32>
              ascvec.storealign %arg27[%c0_46], %114, %mask : memref<64xf32, 1>, index, <f32>
              ascvec.storealign %arg27[%c32], %117, %mask : memref<64xf32, 1>, index, <f32>
            }
            ascvec.mem_bar vst_vld
          } {cannir.vf_mode = "simd"}
        }
        %86 = ascscalar.memref_address %alloc_38 : memref<64xbf16, 1>
        %87 = ascscalar.memref_address %alloc_39 : memref<64xbf16, 1>
        %88 = arith.select %60, %86, %87 : i64
        %89 = ascscalar.bind_local_ptr %88 : memref<64xbf16, 1>
        %90 = arith.select %60, %c2_i64, %c3_i64 : i64
        ascvec.get_buf v, %90, 0 {cannir.buffer.generated_sync}
        ascvec.vf(%alloc_45, %89 : memref<64xf32, 1>, memref<64xbf16, 1>) {
        ^bb0(%arg24: memref<64xf32, 1>, %arg25: memref<64xbf16, 1>):
          %c0_46 = arith.constant 0 : index
          %c0_47 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          scf.for %arg26 = %c0_47 to %c1 step %c1 {
            %106 = ascvec.loadalign %arg24[%c0_46] : memref<64xf32, 1>, index -> <f32>
            %107 = ascvec.create_mask : !ascvec.mask
            %108 = ascvec.cast %106, %107 : <f32> -> <bf16>
            ascvec.storealign_pack %arg25[%c0_46], %108, %107 {mode = "b32_to_b16"} : memref<64xbf16, 1>, index, <bf16>
          }
          ascvec.mem_bar vst_vld
        } {cannir.vf_mode = "simd"}
        ascvec.release_buf v, %90, 0 {cannir.buffer.generated_sync}
        %91 = arith.remsi %59, %c64_i64 : i64
        %92 = arith.cmpi ne, %91, %c0_i64 : i64
        %93 = arith.cmpi slt, %91, %c0_i64 : i64
        %94 = arith.andi %92, %93 : i1
        %95 = arith.divsi %59, %c64_i64 : i64
        %96 = arith.extui %94 : i1 to i64
        %97 = arith.subi %95, %96 : i64
        %98 = arith.muli %97, %c64_i64 : i64
        %99 = arith.addi %80, %98 : i64
        %100 = arith.muli %99, %c16_i64 : i64
        %101 = ascscalar.memref_address %alloc_6 : memref<9216xbf16>
        %102 = arith.divui %100, %c8_i64 : i64
        %103 = arith.addi %101, %102 : i64
        %104 = ascscalar.bind_gm_ptr %103 : memref<1xbf16>
        %105 = ascscalar.bind_local_ptr %88 : memref<1xbf16, 1>
        ascvec.get_buf mte3, %90, 0 {cannir.buffer.generated_sync}
        ascvec.copy_ub2gm %104[%c0], %105[%c0], %c1_i32, %c128_i32, %c0_i64, %c128_i32 l2_cache_ctl %c0_i32 : memref<1xbf16>, memref<1xbf16, 1>
        ascvec.release_buf mte3, %90, 0 {cannir.buffer.generated_sync}
        scf.yield %67 : i64
      }
    }
    return
  }
  func.func @kernel_MlaPrologKernel_run_rope_0_aic(%arg0: !cannir.tensor<<gm, bf16>, <<(-1, 1024)>, <(1024, 1)>>, <<(-1, 1024)>, <(1024, 1)>>, identity>, %arg1: !cannir.tensor<<gm, bf16>, <<(640, 1024)>, <(1024, 1)>>, <<(64, 40, 16, 16)>, <(10240, 256, 16, 1)>>, nd2nz> {cannir.physical_shape_upper_bound = array<i64: 64, 40, 16, 16>}, %arg2: !cannir.tensor<<gm, bf16>, <<(512, 640)>, <(640, 1)>>, <<(40, 32, 16, 16)>, <(8192, 256, 16, 1)>>, nd2nz> {cannir.physical_shape_upper_bound = array<i64: 40, 32, 16, 16>}, %arg3: !cannir.tensor<<gm, bf16>, <<(576, 1024)>, <(1024, 1)>>, <<(64, 36, 16, 16)>, <(9216, 256, 16, 1)>>, nd2nz> {cannir.physical_shape_upper_bound = array<i64: 64, 36, 16, 16>}, %arg4: !cannir.tensor<<gm, bf16>, <<(2, 192, 512)>, <(98304, 512, 1)>>, <<(2, 192, 512)>, <(98304, 512, 1)>>, identity>, %arg5: !cannir.tensor<<gm, f32>, <<(1, 640)>, <(640, 1)>>, <<(1, 640)>, <(640, 1)>>, identity>, %arg6: !cannir.tensor<<gm, f32>, <<(1, 512)>, <(512, 1)>>, <<(1, 512)>, <(512, 1)>>, identity>, %arg7: !cannir.tensor<<gm, f8E4M3FN>, <<(-1, -1, 1, 656)>, <(-1, 656, 656, 1)>>, <<(-1, -1, 1, 656)>, <(-1, 656, 656, 1)>>, identity>, %arg8: !cannir.tensor<<gm, i64>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg9: !cannir.tensor<<gm, bf16>, <<(-1, 64)>, <(64, 1)>>, <<(-1, 64)>, <(64, 1)>>, identity>, %arg10: !cannir.tensor<<gm, bf16>, <<(-1, 64)>, <(64, 1)>>, <<(-1, 64)>, <(64, 1)>>, identity>, %arg11: !cannir.tensor<<gm, bf16>, <<(8, 2, 576)>, <(1152, 576, 1)>>, <<(8, 2, 576)>, <(1152, 576, 1)>>, identity>, %arg12: !cannir.tensor<<gm, bf16>, <<(8, 1024)>, <(1024, 1)>>, <<(8, 1024)>, <(1024, 1)>>, identity>, %arg13: !cannir.tensor<<gm, bf16>, <<(8, 672)>, <(672, 1)>>, <<(8, 672)>, <(672, 1)>>, identity>, %arg14: !cannir.tensor<<gm, bf16>, <<(8, 576)>, <(576, 1)>>, <<(8, 576)>, <(576, 1)>>, identity>, %arg15: !cannir.tensor<<gm, bf16>, <<(8, 576)>, <(576, 1)>>, <<(8, 576)>, <(576, 1)>>, identity>, %arg16: !cannir.tensor<<gm, bf16>, <<(8, 640)>, <(640, 1)>>, <<(8, 640)>, <(640, 1)>>, identity>, %arg17: f32, %arg18: f32, %arg19: f32, %arg20: i64, %arg21: i64) attributes {cannir.buffer.auto_sync, cannir.buffer.sync_analyzed, cannir.buffer.sync_verified, cannir.buffer.tensor_state, cannir.core = "aic", cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.device, cannir.explicit_ring_buffer_cursors, cannir.kernel_group = "kernel_MlaPrologKernel_run_rope_0", cannir.resource.ids_prepared} {
    %c1152_i32 = arith.constant 1152 : i32
    %c4096_i32 = arith.constant 4096 : i32
    %c672_i32 = arith.constant 672 : i32
    %c24_i32 = arith.constant 24 : i32
    %c2_i32 = arith.constant 2 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %c16_i32 = arith.constant 16 : i32
    %c512_i32 = arith.constant 512 : i32
    %c0_i32 = arith.constant 0 : i32
    %c2048_i64 = arith.constant 2048 : i64
    %c16384_i64 = arith.constant 16384 : i64
    %c128_i32 = arith.constant 128 : i32
    %c0 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    %c512 = arith.constant 512 : index
    %c1_i32 = arith.constant 1 : i32
    %c1280_i64 = arith.constant 1280 : i64
    %c64_i32 = arith.constant 64 : i32
    %c1152_i64 = arith.constant 1152 : i64
    %c576_i32 = arith.constant 576 : i32
    %c128_i64 = arith.constant 128 : i64
    %c32_i32 = arith.constant 32 : i32
    %c4_i32 = arith.constant 4 : i32
    %c512_i64 = arith.constant 512 : i64
    %c576_i64 = arith.constant 576 : i64
    %c256_i64 = arith.constant 256 : i64
    %c192_i64 = arith.constant 192 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %c16 = arith.constant 16 : index
    %c10240 = arith.constant 10240 : index
    %c20480_i64 = arith.constant 20480 : i64
    %c15 = arith.constant 15 : index
    %c9216 = arith.constant 9216 : index
    %c18432_i64 = arith.constant 18432 : i64
    %c8192 = arith.constant 8192 : index
    %c336_i64 = arith.constant 336 : i64
    %c288_i64 = arith.constant 288 : i64
    %c32768_i64 = arith.constant 32768 : i64
    %c64_i64 = arith.constant 64 : i64
    %c32_i64 = arith.constant 32 : i64
    %c20_i64 = arith.constant 20 : i64
    %c18_i64 = arith.constant 18 : i64
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
    %c8192_i64 = arith.constant 8192 : i64
    %c24576_i64 = arith.constant 24576 : i64
    %alloc = memref.alloc() {cannir.arg_index = 1 : i64} : memref<655360xbf16>
    %alloc_0 = memref.alloc() {cannir.arg_index = 2 : i64} : memref<327680xbf16>
    %alloc_1 = memref.alloc() {cannir.arg_index = 3 : i64} : memref<589824xbf16>
    %alloc_2 = memref.alloc() {cannir.arg_index = 4 : i64} : memref<196608xbf16>
    %alloc_3 = memref.alloc() {cannir.arg_index = 11 : i64} : memref<9216xbf16>
    %alloc_4 = memref.alloc() {cannir.arg_index = 12 : i64} : memref<8192xbf16>
    %alloc_5 = memref.alloc() {cannir.arg_index = 13 : i64} : memref<5376xbf16>
    %alloc_6 = memref.alloc() {cannir.arg_index = 14 : i64} : memref<4608xbf16>
    %alloc_7 = memref.alloc() {cannir.arg_index = 15 : i64} : memref<4608xbf16>
    %alloc_8 = memref.alloc() {cannir.arg_index = 16 : i64} : memref<5120xbf16>
    %alloc_9 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<1024xbf16, 3>
    %alloc_10 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<8192xbf16, 4>
    %alloc_11 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<1024xbf16, 3>
    %alloc_12 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<8192xbf16, 4>
    %alloc_13 = memref.alloc() {cannir.byte_offset = 20480 : i64} : memref<8192xbf16, 2>
    %alloc_14 = memref.alloc() {cannir.byte_offset = 4096 : i64} : memref<8192xbf16, 2>
    %alloc_15 = memref.alloc() {cannir.byte_offset = 2048 : i64} : memref<1024xbf16, 2>
    %alloc_16 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<1024xbf16, 2>
    %alloc_17 = memref.alloc() {cannir.byte_offset = 8192 : i64} : memref<2048xf32, 5>
    %alloc_18 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<2048xf32, 5>
    %alloc_19 = memref.alloc() {cannir.byte_offset = 69632 : i64} : memref<32768xbf16, 2>
    %alloc_20 = memref.alloc() {cannir.byte_offset = 4096 : i64} : memref<32768xbf16, 2>
    %alloc_21 = memref.alloc() {cannir.byte_offset = 2048 : i64} : memref<1024xbf16, 2>
    %alloc_22 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<1024xbf16, 2>
    %alloc_23 = memref.alloc() {cannir.byte_offset = 32768 : i64} : memref<8192xf32, 5>
    %alloc_24 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<8192xf32, 5>
    %alloc_25 = memref.alloc() {cannir.byte_offset = 32768 : i64} : memref<16384xbf16, 4>
    %alloc_26 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<16384xbf16, 4>
    %alloc_27 = memref.alloc() {cannir.byte_offset = 4096 : i64} : memref<2048xbf16, 3>
    %alloc_28 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<2048xbf16, 3>
    %alloc_29 = memref.alloc() {cannir.byte_offset = 40960 : i64} : memref<16384xbf16, 2>
    %alloc_30 = memref.alloc() {cannir.byte_offset = 8192 : i64} : memref<16384xbf16, 2>
    %alloc_31 = memref.alloc() {cannir.byte_offset = 4096 : i64} : memref<2048xbf16, 2>
    %alloc_32 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<2048xbf16, 2>
    %alloc_33 = memref.alloc() {cannir.byte_offset = 8192 : i64} : memref<2048xf32, 5>
    %alloc_34 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<2048xf32, 5>
    %alloc_35 = memref.alloc() {cannir.byte_offset = 8192 : i64} : memref<4096xbf16, 4>
    %alloc_36 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<4096xbf16, 4>
    %alloc_37 = memref.alloc() {cannir.byte_offset = 4096 : i64} : memref<2048xbf16, 3>
    %alloc_38 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<2048xbf16, 3>
    %alloc_39 = memref.alloc() {cannir.byte_offset = 65536 : i64} : memref<16384xbf16, 2>
    %alloc_40 = memref.alloc() {cannir.byte_offset = 32768 : i64} : memref<16384xbf16, 2>
    %alloc_41 = memref.alloc() {cannir.byte_offset = 16384 : i64} : memref<8192xbf16, 2>
    %alloc_42 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<8192xbf16, 2>
    %alloc_43 = memref.alloc() {cannir.byte_offset = 2048 : i64} : memref<512xf32, 5>
    %alloc_44 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<512xf32, 5>
    %alloc_45 = memref.alloc() {cannir.byte_offset = 8192 : i64} : memref<4096xbf16, 4>
    %alloc_46 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<4096xbf16, 4>
    %alloc_47 = memref.alloc() {cannir.byte_offset = 4096 : i64} : memref<2048xbf16, 3>
    %alloc_48 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<2048xbf16, 3>
    %alloc_49 = memref.alloc() {cannir.byte_offset = 65536 : i64} : memref<16384xbf16, 2>
    %alloc_50 = memref.alloc() {cannir.byte_offset = 32768 : i64} : memref<16384xbf16, 2>
    %alloc_51 = memref.alloc() {cannir.byte_offset = 16384 : i64} : memref<8192xbf16, 2>
    %alloc_52 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<8192xbf16, 2>
    %alloc_53 = memref.alloc() {cannir.byte_offset = 2048 : i64} : memref<512xf32, 5>
    %alloc_54 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<512xf32, 5>
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
    %6:3 = scf.for %arg22 = %0 to %c20_i64 step %1 iter_args(%arg23 = %c0_i64, %arg24 = %c0_i64, %arg25 = %c0_i64) -> (i64, i64, i64)  : i64 {
      %11 = arith.remsi %arg22, %c20_i64 : i64
      %12 = arith.cmpi ne, %11, %c0_i64 : i64
      %13 = arith.cmpi slt, %11, %c0_i64 : i64
      %14 = arith.andi %12, %13 : i1
      %15 = arith.divsi %arg22, %c20_i64 : i64
      %16 = arith.extui %14 : i1 to i64
      %17 = arith.subi %15, %16 : i64
      %18 = arith.select %14, %c20_i64, %c0_i64 : i64
      %19 = arith.addi %11, %18 overflow<nsw> : i64
      %20 = arith.cmpi eq, %arg25, %c0_i64 : i64
      %21 = ascscalar.memref_address %alloc_54 : memref<512xf32, 5>
      %22 = ascscalar.memref_address %alloc_53 : memref<512xf32, 5>
      %23 = arith.select %20, %21, %22 : i64
      %24 = arith.select %20, %c8_i64, %c9_i64 : i64
      %25 = arith.addi %arg25, %c1_i64 overflow<nsw> : i64
      %26 = arith.remui %25, %c2_i64 : i64
      %27:2 = scf.for %arg26 = %c0_i64 to %c2_i64 step %c1_i64 iter_args(%arg27 = %arg23, %arg28 = %arg24) -> (i64, i64)  : i64 {
        %34 = arith.cmpi eq, %arg27, %c0_i64 : i64
        %35 = ascscalar.memref_address %alloc_52 : memref<8192xbf16, 2>
        %36 = ascscalar.memref_address %alloc_51 : memref<8192xbf16, 2>
        %37 = arith.select %34, %35, %36 : i64
        %38 = arith.cmpi ne, %arg27, %c0_i64 : i64
        %39 = arith.extui %38 : i1 to i64
        %40 = arith.addi %arg27, %c1_i64 overflow<nsw> : i64
        %41 = arith.remui %40, %c2_i64 : i64
        %42 = ascscalar.bind_local_ptr %37 : memref<1xbf16, 2>
        %43 = arith.muli %17, %c512_i64 : i64
        %44 = arith.muli %arg26, %c32_i64 overflow<nsw> : i64
        %45 = arith.addi %43, %44 : i64
        %46 = arith.muli %45, %c16_i64 : i64
        %47 = arith.minsi %17, %c1_i64 : i64
        %48 = arith.muli %47, %c8_i64 : i64
        %49 = arith.subi %c8_i64, %48 : i64
        %50 = arith.minsi %49, %c8_i64 : i64
        %51 = arith.cmpi sle, %17, %c1_i64 : i64
        %52 = arith.select %51, %50, %c0_i64 : i64
        asccube.get_buf mte2, %39, 0 {cannir.buffer.generated_sync}
        %53 = arith.cmpi ugt, %52, %c0_i64 : i64
        scf.if %53 {
          %110 = arith.index_cast %46 : i64 to index
          %111 = arith.trunci %52 : i64 to i32
          asccube.copy_gm2l1 %42[%c0], %alloc_4[%110], %111, %c512_i32, %c2048_i64, %c1_i32, %c0_i64, %c0_i64, %c16_i32 l2_cache_ctl %c1_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<8192xbf16>
        }
        asccube.release_buf mte2, %39, 0 {cannir.buffer.generated_sync}
        %54 = ascscalar.memref_address %alloc_50 : memref<16384xbf16, 2>
        %55 = ascscalar.memref_address %alloc_49 : memref<16384xbf16, 2>
        %56 = arith.select %34, %54, %55 : i64
        %57 = arith.select %34, %c2_i64, %c3_i64 : i64
        %58 = ascscalar.bind_local_ptr %56 : memref<1xbf16, 2>
        %59 = arith.muli %19, %c32_i64 overflow<nsw> : i64
        %60 = arith.index_cast %59 : i64 to index
        %61 = arith.muli %arg26, %c512_i64 overflow<nsw> : i64
        %62 = arith.index_cast %61 : i64 to index
        %63 = arith.divui %62, %c16 : index
        %64 = arith.muli %63, %c10240 : index
        %65 = arith.divui %60, %c16 : index
        %66 = arith.muli %65, %c256 : index
        %67 = arith.addi %64, %66 : index
        asccube.get_buf mte2, %57, 0 {cannir.buffer.generated_sync}
        asccube.copy_gm2l1_burst %58[%c0], %alloc[%67], %c32_i32, %c1024_i32, %c20480_i64, %c1024_i32 l2_cache_ctl %c0_i32 pad_value %c0_i32 : i32 left_padding %c0_i32 right_padding %c0_i32 : memref<1xbf16, 2>, memref<655360xbf16>
        asccube.release_buf mte2, %57, 0 {cannir.buffer.generated_sync}
        %68 = arith.cmpi eq, %arg28, %c0_i64 : i64
        %69 = ascscalar.memref_address %alloc_48 : memref<2048xbf16, 3>
        %70 = ascscalar.memref_address %alloc_47 : memref<2048xbf16, 3>
        %71 = arith.select %68, %69, %70 : i64
        %72 = arith.select %68, %c4_i64, %c5_i64 : i64
        %73 = arith.addi %arg28, %c1_i64 overflow<nsw> : i64
        %74 = arith.remui %73, %c2_i64 : i64
        %75 = ascscalar.bind_local_ptr %71 : memref<1xbf16, 3>
        asccube.get_buf mte1, %72, 0 {cannir.buffer.generated_sync}
        asccube.get_buf mte1, %39, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0a %75[%c0], %42[%c0], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c1_i32, %c1_i32 : memref<1xbf16, 3>, memref<1xbf16, 2>
        asccube.release_buf mte1, %39, 0 {cannir.buffer.generated_sync}
        asccube.release_buf mte1, %72, 0 {cannir.buffer.generated_sync}
        %76 = ascscalar.memref_address %alloc_46 : memref<4096xbf16, 4>
        %77 = ascscalar.memref_address %alloc_45 : memref<4096xbf16, 4>
        %78 = arith.select %68, %76, %77 : i64
        %79 = arith.select %68, %c6_i64, %c7_i64 : i64
        %80 = ascscalar.bind_local_ptr %78 : memref<1xbf16, 4>
        asccube.get_buf mte1, %79, 0 {cannir.buffer.generated_sync}
        asccube.get_buf mte1, %57, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0b %80[%c0], %58[%c0], %c0_i32, %c0_i32, %c2_i32, %c8_i32, %c2_i32, %c2_i32 : memref<1xbf16, 4>, memref<1xbf16, 2>
        asccube.release_buf mte1, %57, 0 {cannir.buffer.generated_sync}
        asccube.release_buf mte1, %79, 0 {cannir.buffer.generated_sync}
        %81 = arith.cmpi eq, %arg26, %c0_i64 : i64
        %82 = ascscalar.bind_local_ptr %23 : memref<1xf32, 5>
        asccube.get_buf m, %24, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %72, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %79, 0 {cannir.buffer.generated_sync}
        scf.if %81 {
          asccube.mmad %82[%c0], %75, %80, %c8_i32, %c128_i32, %c32_i32, %c0_i32 {c_matrix_init_val = true} : memref<1xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
        } else {
          asccube.mmad %82[%c0], %75, %80, %c8_i32, %c128_i32, %c32_i32, %c0_i32 {c_matrix_init_val = false} : memref<1xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
        }
        asccube.release_buf m, %79, 0 {cannir.buffer.generated_sync}
        asccube.release_buf m, %72, 0 {cannir.buffer.generated_sync}
        %83 = arith.cmpi eq, %74, %c0_i64 : i64
        %84 = arith.select %83, %69, %70 : i64
        %85 = arith.select %83, %c4_i64, %c5_i64 : i64
        %86 = arith.addi %74, %c1_i64 overflow<nsw> : i64
        %87 = arith.remui %86, %c2_i64 : i64
        %88 = ascscalar.bind_local_ptr %84 : memref<1xbf16, 3>
        asccube.get_buf mte1, %85, 0 {cannir.buffer.generated_sync}
        asccube.get_buf mte1, %39, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0a %88[%c0], %42[%c0], %c0_i32, %c8_i32, %c1_i32, %c8_i32, %c1_i32, %c1_i32 : memref<1xbf16, 3>, memref<1xbf16, 2>
        asccube.release_buf mte1, %85, 0 {cannir.buffer.generated_sync}
        %89 = arith.select %83, %76, %77 : i64
        %90 = arith.select %83, %c6_i64, %c7_i64 : i64
        %91 = ascscalar.bind_local_ptr %89 : memref<1xbf16, 4>
        asccube.get_buf mte1, %90, 0 {cannir.buffer.generated_sync}
        asccube.get_buf mte1, %57, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0b %91[%c0], %58[%c0], %c0_i32, %c8_i32, %c2_i32, %c8_i32, %c2_i32, %c2_i32 : memref<1xbf16, 4>, memref<1xbf16, 2>
        asccube.release_buf mte1, %90, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %85, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %90, 0 {cannir.buffer.generated_sync}
        asccube.mmad %82[%c0], %88, %91, %c8_i32, %c128_i32, %c32_i32, %c0_i32 {c_matrix_init_val = false} : memref<1xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
        asccube.release_buf m, %90, 0 {cannir.buffer.generated_sync}
        asccube.release_buf m, %85, 0 {cannir.buffer.generated_sync}
        %92 = arith.cmpi eq, %87, %c0_i64 : i64
        %93 = arith.select %92, %69, %70 : i64
        %94 = arith.select %92, %c4_i64, %c5_i64 : i64
        %95 = arith.addi %87, %c1_i64 overflow<nsw> : i64
        %96 = arith.remui %95, %c2_i64 : i64
        %97 = ascscalar.bind_local_ptr %93 : memref<1xbf16, 3>
        asccube.get_buf mte1, %94, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0a %97[%c0], %42[%c0], %c0_i32, %c16_i32, %c1_i32, %c8_i32, %c1_i32, %c1_i32 : memref<1xbf16, 3>, memref<1xbf16, 2>
        asccube.release_buf mte1, %94, 0 {cannir.buffer.generated_sync}
        %98 = arith.select %92, %76, %77 : i64
        %99 = arith.select %92, %c6_i64, %c7_i64 : i64
        %100 = ascscalar.bind_local_ptr %98 : memref<1xbf16, 4>
        asccube.get_buf mte1, %99, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0b %100[%c0], %58[%c0], %c0_i32, %c16_i32, %c2_i32, %c8_i32, %c2_i32, %c2_i32 : memref<1xbf16, 4>, memref<1xbf16, 2>
        asccube.release_buf mte1, %99, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %94, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %99, 0 {cannir.buffer.generated_sync}
        asccube.mmad %82[%c0], %97, %100, %c8_i32, %c128_i32, %c32_i32, %c0_i32 {c_matrix_init_val = false} : memref<1xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
        asccube.release_buf m, %99, 0 {cannir.buffer.generated_sync}
        asccube.release_buf m, %94, 0 {cannir.buffer.generated_sync}
        %101 = arith.cmpi eq, %96, %c0_i64 : i64
        %102 = arith.select %101, %69, %70 : i64
        %103 = arith.select %101, %c4_i64, %c5_i64 : i64
        %104 = arith.addi %96, %c1_i64 overflow<nsw> : i64
        %105 = arith.remui %104, %c2_i64 : i64
        %106 = ascscalar.bind_local_ptr %102 : memref<1xbf16, 3>
        asccube.get_buf mte1, %103, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0a %106[%c0], %42[%c0], %c0_i32, %c24_i32, %c1_i32, %c8_i32, %c1_i32, %c1_i32 : memref<1xbf16, 3>, memref<1xbf16, 2>
        asccube.release_buf mte1, %39, 0 {cannir.buffer.generated_sync}
        asccube.release_buf mte1, %103, 0 {cannir.buffer.generated_sync}
        %107 = arith.select %101, %76, %77 : i64
        %108 = arith.select %101, %c6_i64, %c7_i64 : i64
        %109 = ascscalar.bind_local_ptr %107 : memref<1xbf16, 4>
        asccube.get_buf mte1, %108, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0b %109[%c0], %58[%c0], %c0_i32, %c24_i32, %c2_i32, %c8_i32, %c2_i32, %c2_i32 : memref<1xbf16, 4>, memref<1xbf16, 2>
        asccube.release_buf mte1, %57, 0 {cannir.buffer.generated_sync}
        asccube.release_buf mte1, %108, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %103, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %108, 0 {cannir.buffer.generated_sync}
        asccube.mmad %82[%c0], %106, %109, %c8_i32, %c128_i32, %c32_i32, %c0_i32 {c_matrix_init_val = false} : memref<1xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
        asccube.release_buf m, %108, 0 {cannir.buffer.generated_sync}
        asccube.release_buf m, %103, 0 {cannir.buffer.generated_sync}
        asccube.release_buf m, %24, 0 {cannir.buffer.generated_sync}
        scf.yield %41, %105 : i64, i64
      }
      %28 = ascscalar.bind_local_ptr %23 : memref<1xf32, 5>
      %29 = arith.muli %17, %c336_i64 : i64
      %30 = arith.muli %19, %c2_i64 overflow<nsw> : i64
      %31 = arith.addi %29, %30 : i64
      %32 = arith.muli %31, %c16_i64 : i64
      asccube.get_buf fixpipe, %24, 0 {cannir.buffer.generated_sync}
      %33 = arith.index_cast %32 : i64 to index
      asccube.copy_l0c2gm %alloc_5[%33], %28[%c0], %c32_i32, %c8_i32, %c672_i32, %c16_i32, %c0_i32 l2_cache_ctl %c0_i32 {nz2nd_en = true, relu_en = false} : memref<5376xbf16>, memref<1xf32, 5>
      asccube.release_buf fixpipe, %24, 0 {cannir.buffer.generated_sync}
      scf.yield %27#0, %27#1, %26 : i64, i64, i64
    }
    asccube.sync_all
    asccube.sync_block_wait s, %c13_i64
    asccube.sync_block_arrive fixpipe, %c14_i64 {mode = 0 : i32}
    asccube.sync_block_wait s, %c14_i64 {mode = 0 : i32}
    asccube.sync_block_arrive mte3, %c15_i64
    %7:3 = scf.for %arg22 = %0 to %c18_i64 step %1 iter_args(%arg23 = %c0_i64, %arg24 = %c0_i64, %arg25 = %c0_i64) -> (i64, i64, i64)  : i64 {
      %11 = arith.remsi %arg22, %c18_i64 : i64
      %12 = arith.cmpi ne, %11, %c0_i64 : i64
      %13 = arith.cmpi slt, %11, %c0_i64 : i64
      %14 = arith.andi %12, %13 : i1
      %15 = arith.divsi %arg22, %c18_i64 : i64
      %16 = arith.extui %14 : i1 to i64
      %17 = arith.subi %15, %16 : i64
      %18 = arith.select %14, %c18_i64, %c0_i64 : i64
      %19 = arith.addi %11, %18 overflow<nsw> : i64
      %20 = arith.cmpi eq, %arg25, %c0_i64 : i64
      %21 = ascscalar.memref_address %alloc_44 : memref<512xf32, 5>
      %22 = ascscalar.memref_address %alloc_43 : memref<512xf32, 5>
      %23 = arith.select %20, %21, %22 : i64
      %24 = arith.select %20, %c8_i64, %c9_i64 : i64
      %25 = arith.addi %arg25, %c1_i64 overflow<nsw> : i64
      %26 = arith.remui %25, %c2_i64 : i64
      %27:2 = scf.for %arg26 = %c0_i64 to %c2_i64 step %c1_i64 iter_args(%arg27 = %arg23, %arg28 = %arg24) -> (i64, i64)  : i64 {
        %34 = arith.cmpi eq, %arg27, %c0_i64 : i64
        %35 = ascscalar.memref_address %alloc_42 : memref<8192xbf16, 2>
        %36 = ascscalar.memref_address %alloc_41 : memref<8192xbf16, 2>
        %37 = arith.select %34, %35, %36 : i64
        %38 = arith.cmpi ne, %arg27, %c0_i64 : i64
        %39 = arith.extui %38 : i1 to i64
        %40 = arith.addi %arg27, %c1_i64 overflow<nsw> : i64
        %41 = arith.remui %40, %c2_i64 : i64
        %42 = ascscalar.bind_local_ptr %37 : memref<1xbf16, 2>
        %43 = arith.muli %17, %c512_i64 : i64
        %44 = arith.muli %arg26, %c32_i64 overflow<nsw> : i64
        %45 = arith.addi %43, %44 : i64
        %46 = arith.muli %45, %c16_i64 : i64
        %47 = arith.minsi %17, %c1_i64 : i64
        %48 = arith.muli %47, %c8_i64 : i64
        %49 = arith.subi %c8_i64, %48 : i64
        %50 = arith.minsi %49, %c8_i64 : i64
        %51 = arith.cmpi sle, %17, %c1_i64 : i64
        %52 = arith.select %51, %50, %c0_i64 : i64
        asccube.get_buf mte2, %39, 0 {cannir.buffer.generated_sync}
        %53 = arith.cmpi ugt, %52, %c0_i64 : i64
        scf.if %53 {
          %110 = arith.index_cast %46 : i64 to index
          %111 = arith.trunci %52 : i64 to i32
          asccube.copy_gm2l1 %42[%c0], %alloc_4[%110], %111, %c512_i32, %c2048_i64, %c1_i32, %c0_i64, %c0_i64, %c16_i32 l2_cache_ctl %c1_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<8192xbf16>
        }
        asccube.release_buf mte2, %39, 0 {cannir.buffer.generated_sync}
        %54 = ascscalar.memref_address %alloc_40 : memref<16384xbf16, 2>
        %55 = ascscalar.memref_address %alloc_39 : memref<16384xbf16, 2>
        %56 = arith.select %34, %54, %55 : i64
        %57 = arith.select %34, %c2_i64, %c3_i64 : i64
        %58 = ascscalar.bind_local_ptr %56 : memref<1xbf16, 2>
        %59 = arith.muli %19, %c32_i64 overflow<nsw> : i64
        %60 = arith.index_cast %59 : i64 to index
        %61 = arith.muli %arg26, %c512_i64 overflow<nsw> : i64
        %62 = arith.index_cast %61 : i64 to index
        %63 = arith.divui %62, %c16 : index
        %64 = arith.muli %63, %c9216 : index
        %65 = arith.divui %60, %c16 : index
        %66 = arith.muli %65, %c256 : index
        %67 = arith.addi %64, %66 : index
        asccube.get_buf mte2, %57, 0 {cannir.buffer.generated_sync}
        asccube.copy_gm2l1_burst %58[%c0], %alloc_1[%67], %c32_i32, %c1024_i32, %c18432_i64, %c1024_i32 l2_cache_ctl %c0_i32 pad_value %c0_i32 : i32 left_padding %c0_i32 right_padding %c0_i32 : memref<1xbf16, 2>, memref<589824xbf16>
        asccube.release_buf mte2, %57, 0 {cannir.buffer.generated_sync}
        %68 = arith.cmpi eq, %arg28, %c0_i64 : i64
        %69 = ascscalar.memref_address %alloc_38 : memref<2048xbf16, 3>
        %70 = ascscalar.memref_address %alloc_37 : memref<2048xbf16, 3>
        %71 = arith.select %68, %69, %70 : i64
        %72 = arith.select %68, %c4_i64, %c5_i64 : i64
        %73 = arith.addi %arg28, %c1_i64 overflow<nsw> : i64
        %74 = arith.remui %73, %c2_i64 : i64
        %75 = ascscalar.bind_local_ptr %71 : memref<1xbf16, 3>
        asccube.get_buf mte1, %72, 0 {cannir.buffer.generated_sync}
        asccube.get_buf mte1, %39, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0a %75[%c0], %42[%c0], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c1_i32, %c1_i32 : memref<1xbf16, 3>, memref<1xbf16, 2>
        asccube.release_buf mte1, %39, 0 {cannir.buffer.generated_sync}
        asccube.release_buf mte1, %72, 0 {cannir.buffer.generated_sync}
        %76 = ascscalar.memref_address %alloc_36 : memref<4096xbf16, 4>
        %77 = ascscalar.memref_address %alloc_35 : memref<4096xbf16, 4>
        %78 = arith.select %68, %76, %77 : i64
        %79 = arith.select %68, %c6_i64, %c7_i64 : i64
        %80 = ascscalar.bind_local_ptr %78 : memref<1xbf16, 4>
        asccube.get_buf mte1, %79, 0 {cannir.buffer.generated_sync}
        asccube.get_buf mte1, %57, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0b %80[%c0], %58[%c0], %c0_i32, %c0_i32, %c2_i32, %c8_i32, %c2_i32, %c2_i32 : memref<1xbf16, 4>, memref<1xbf16, 2>
        asccube.release_buf mte1, %57, 0 {cannir.buffer.generated_sync}
        asccube.release_buf mte1, %79, 0 {cannir.buffer.generated_sync}
        %81 = arith.cmpi eq, %arg26, %c0_i64 : i64
        %82 = ascscalar.bind_local_ptr %23 : memref<1xf32, 5>
        asccube.get_buf m, %24, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %72, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %79, 0 {cannir.buffer.generated_sync}
        scf.if %81 {
          asccube.mmad %82[%c0], %75, %80, %c8_i32, %c128_i32, %c32_i32, %c0_i32 {c_matrix_init_val = true} : memref<1xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
        } else {
          asccube.mmad %82[%c0], %75, %80, %c8_i32, %c128_i32, %c32_i32, %c0_i32 {c_matrix_init_val = false} : memref<1xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
        }
        asccube.release_buf m, %79, 0 {cannir.buffer.generated_sync}
        asccube.release_buf m, %72, 0 {cannir.buffer.generated_sync}
        %83 = arith.cmpi eq, %74, %c0_i64 : i64
        %84 = arith.select %83, %69, %70 : i64
        %85 = arith.select %83, %c4_i64, %c5_i64 : i64
        %86 = arith.addi %74, %c1_i64 overflow<nsw> : i64
        %87 = arith.remui %86, %c2_i64 : i64
        %88 = ascscalar.bind_local_ptr %84 : memref<1xbf16, 3>
        asccube.get_buf mte1, %85, 0 {cannir.buffer.generated_sync}
        asccube.get_buf mte1, %39, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0a %88[%c0], %42[%c0], %c0_i32, %c8_i32, %c1_i32, %c8_i32, %c1_i32, %c1_i32 : memref<1xbf16, 3>, memref<1xbf16, 2>
        asccube.release_buf mte1, %85, 0 {cannir.buffer.generated_sync}
        %89 = arith.select %83, %76, %77 : i64
        %90 = arith.select %83, %c6_i64, %c7_i64 : i64
        %91 = ascscalar.bind_local_ptr %89 : memref<1xbf16, 4>
        asccube.get_buf mte1, %90, 0 {cannir.buffer.generated_sync}
        asccube.get_buf mte1, %57, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0b %91[%c0], %58[%c0], %c0_i32, %c8_i32, %c2_i32, %c8_i32, %c2_i32, %c2_i32 : memref<1xbf16, 4>, memref<1xbf16, 2>
        asccube.release_buf mte1, %90, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %85, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %90, 0 {cannir.buffer.generated_sync}
        asccube.mmad %82[%c0], %88, %91, %c8_i32, %c128_i32, %c32_i32, %c0_i32 {c_matrix_init_val = false} : memref<1xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
        asccube.release_buf m, %90, 0 {cannir.buffer.generated_sync}
        asccube.release_buf m, %85, 0 {cannir.buffer.generated_sync}
        %92 = arith.cmpi eq, %87, %c0_i64 : i64
        %93 = arith.select %92, %69, %70 : i64
        %94 = arith.select %92, %c4_i64, %c5_i64 : i64
        %95 = arith.addi %87, %c1_i64 overflow<nsw> : i64
        %96 = arith.remui %95, %c2_i64 : i64
        %97 = ascscalar.bind_local_ptr %93 : memref<1xbf16, 3>
        asccube.get_buf mte1, %94, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0a %97[%c0], %42[%c0], %c0_i32, %c16_i32, %c1_i32, %c8_i32, %c1_i32, %c1_i32 : memref<1xbf16, 3>, memref<1xbf16, 2>
        asccube.release_buf mte1, %94, 0 {cannir.buffer.generated_sync}
        %98 = arith.select %92, %76, %77 : i64
        %99 = arith.select %92, %c6_i64, %c7_i64 : i64
        %100 = ascscalar.bind_local_ptr %98 : memref<1xbf16, 4>
        asccube.get_buf mte1, %99, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0b %100[%c0], %58[%c0], %c0_i32, %c16_i32, %c2_i32, %c8_i32, %c2_i32, %c2_i32 : memref<1xbf16, 4>, memref<1xbf16, 2>
        asccube.release_buf mte1, %99, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %94, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %99, 0 {cannir.buffer.generated_sync}
        asccube.mmad %82[%c0], %97, %100, %c8_i32, %c128_i32, %c32_i32, %c0_i32 {c_matrix_init_val = false} : memref<1xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
        asccube.release_buf m, %99, 0 {cannir.buffer.generated_sync}
        asccube.release_buf m, %94, 0 {cannir.buffer.generated_sync}
        %101 = arith.cmpi eq, %96, %c0_i64 : i64
        %102 = arith.select %101, %69, %70 : i64
        %103 = arith.select %101, %c4_i64, %c5_i64 : i64
        %104 = arith.addi %96, %c1_i64 overflow<nsw> : i64
        %105 = arith.remui %104, %c2_i64 : i64
        %106 = ascscalar.bind_local_ptr %102 : memref<1xbf16, 3>
        asccube.get_buf mte1, %103, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0a %106[%c0], %42[%c0], %c0_i32, %c24_i32, %c1_i32, %c8_i32, %c1_i32, %c1_i32 : memref<1xbf16, 3>, memref<1xbf16, 2>
        asccube.release_buf mte1, %39, 0 {cannir.buffer.generated_sync}
        asccube.release_buf mte1, %103, 0 {cannir.buffer.generated_sync}
        %107 = arith.select %101, %76, %77 : i64
        %108 = arith.select %101, %c6_i64, %c7_i64 : i64
        %109 = ascscalar.bind_local_ptr %107 : memref<1xbf16, 4>
        asccube.get_buf mte1, %108, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0b %109[%c0], %58[%c0], %c0_i32, %c24_i32, %c2_i32, %c8_i32, %c2_i32, %c2_i32 : memref<1xbf16, 4>, memref<1xbf16, 2>
        asccube.release_buf mte1, %57, 0 {cannir.buffer.generated_sync}
        asccube.release_buf mte1, %108, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %103, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %108, 0 {cannir.buffer.generated_sync}
        asccube.mmad %82[%c0], %106, %109, %c8_i32, %c128_i32, %c32_i32, %c0_i32 {c_matrix_init_val = false} : memref<1xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
        asccube.release_buf m, %108, 0 {cannir.buffer.generated_sync}
        asccube.release_buf m, %103, 0 {cannir.buffer.generated_sync}
        asccube.release_buf m, %24, 0 {cannir.buffer.generated_sync}
        scf.yield %41, %105 : i64, i64
      }
      %28 = ascscalar.bind_local_ptr %23 : memref<1xf32, 5>
      %29 = arith.muli %17, %c288_i64 : i64
      %30 = arith.muli %19, %c2_i64 overflow<nsw> : i64
      %31 = arith.addi %29, %30 : i64
      %32 = arith.muli %31, %c16_i64 : i64
      asccube.get_buf fixpipe, %24, 0 {cannir.buffer.generated_sync}
      %33 = arith.index_cast %32 : i64 to index
      asccube.copy_l0c2gm %alloc_6[%33], %28[%c0], %c32_i32, %c8_i32, %c576_i32, %c16_i32, %c0_i32 l2_cache_ctl %c0_i32 {nz2nd_en = true, relu_en = false} : memref<4608xbf16>, memref<1xf32, 5>
      asccube.release_buf fixpipe, %24, 0 {cannir.buffer.generated_sync}
      scf.yield %27#0, %27#1, %26 : i64, i64, i64
    }
    %8 = arith.muli %1, %c2_i64 : i64
    asccube.sync_all
    asccube.sync_block_wait s, %c13_i64
    asccube.sync_block_arrive fixpipe, %c14_i64 {mode = 0 : i32}
    asccube.sync_block_wait s, %c14_i64 {mode = 0 : i32}
    asccube.sync_block_arrive mte3, %c15_i64
    %9:2 = scf.for %arg22 = %0 to %c4_i64 step %1 iter_args(%arg23 = %c0_i64, %arg24 = %c0_i64) -> (i64, i64)  : i64 {
      %11 = arith.cmpi eq, %arg24, %c0_i64 : i64
      %12 = ascscalar.memref_address %alloc_34 : memref<2048xf32, 5>
      %13 = ascscalar.memref_address %alloc_33 : memref<2048xf32, 5>
      %14 = arith.select %11, %12, %13 : i64
      %15 = arith.select %11, %c8_i64, %c9_i64 : i64
      %16 = arith.addi %arg24, %c1_i64 overflow<nsw> : i64
      %17 = arith.remui %16, %c2_i64 : i64
      %18 = scf.for %arg25 = %c0_i64 to %c5_i64 step %c1_i64 iter_args(%arg26 = %arg23) -> (i64)  : i64 {
        %22 = arith.cmpi eq, %arg26, %c0_i64 : i64
        %23 = ascscalar.memref_address %alloc_32 : memref<2048xbf16, 2>
        %24 = ascscalar.memref_address %alloc_31 : memref<2048xbf16, 2>
        %25 = arith.select %22, %23, %24 : i64
        %26 = arith.cmpi ne, %arg26, %c0_i64 : i64
        %27 = arith.extui %26 : i1 to i64
        %28 = arith.addi %arg26, %c1_i64 overflow<nsw> : i64
        %29 = arith.remui %28, %c2_i64 : i64
        %30 = ascscalar.bind_local_ptr %25 : memref<1xbf16, 2>
        %31 = arith.muli %arg25, %c128_i64 overflow<nsw> : i64
        asccube.get_buf mte2, %27, 0 {cannir.buffer.generated_sync}
        %32 = arith.index_cast %31 : i64 to index
        asccube.copy_gm2l1 %30[%c0], %alloc_8[%32], %c8_i32, %c128_i32, %c1280_i64, %c1_i32, %c0_i64, %c0_i64, %c16_i32 l2_cache_ctl %c1_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<5120xbf16>
        asccube.release_buf mte2, %27, 0 {cannir.buffer.generated_sync}
        %33 = ascscalar.memref_address %alloc_30 : memref<16384xbf16, 2>
        %34 = ascscalar.memref_address %alloc_29 : memref<16384xbf16, 2>
        %35 = arith.select %22, %33, %34 : i64
        %36 = arith.select %22, %c2_i64, %c3_i64 : i64
        %37 = ascscalar.bind_local_ptr %35 : memref<1xbf16, 2>
        %38 = arith.muli %arg22, %c128_i64 : i64
        %39 = arith.index_cast %38 : i64 to index
        %40 = arith.minsi %arg22, %c4_i64 : i64
        %41 = arith.muli %40, %c128_i64 : i64
        %42 = arith.subi %c512_i64, %41 : i64
        %43 = arith.minsi %42, %c128_i64 : i64
        %44 = arith.cmpi sle, %arg22, %c4_i64 : i64
        %45 = arith.select %44, %43, %c0_i64 : i64
        %46 = arith.index_cast %45 : i64 to index
        %47 = arith.divui %32, %c16 : index
        %48 = arith.muli %47, %c8192 : index
        %49 = arith.divui %39, %c16 : index
        %50 = arith.muli %49, %c256 : index
        %51 = arith.addi %48, %50 : index
        %52 = arith.addi %46, %c15 : index
        %53 = arith.divui %52, %c16 : index
        %54 = arith.muli %53, %c512 : index
        %55 = arith.index_cast %54 : index to i64
        asccube.get_buf mte2, %36, 0 {cannir.buffer.generated_sync}
        %56 = arith.cmpi ugt, %55, %c0_i64 : i64
        scf.if %56 {
          %69 = arith.trunci %55 : i64 to i32
          asccube.copy_gm2l1_burst %37[%c0], %alloc_0[%51], %c8_i32, %69, %c16384_i64, %c4096_i32 l2_cache_ctl %c0_i32 pad_value %c0_i32 : i32 left_padding %c0_i32 right_padding %c0_i32 : memref<1xbf16, 2>, memref<327680xbf16>
        }
        asccube.release_buf mte2, %36, 0 {cannir.buffer.generated_sync}
        %57 = ascscalar.memref_address %alloc_28 : memref<2048xbf16, 3>
        %58 = ascscalar.memref_address %alloc_27 : memref<2048xbf16, 3>
        %59 = arith.select %22, %57, %58 : i64
        %60 = arith.select %22, %c4_i64, %c5_i64 : i64
        %61 = ascscalar.bind_local_ptr %59 : memref<1xbf16, 3>
        asccube.get_buf mte1, %60, 0 {cannir.buffer.generated_sync}
        asccube.get_buf mte1, %27, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0a %61[%c0], %30[%c0], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c1_i32, %c1_i32 : memref<1xbf16, 3>, memref<1xbf16, 2>
        asccube.release_buf mte1, %27, 0 {cannir.buffer.generated_sync}
        asccube.release_buf mte1, %60, 0 {cannir.buffer.generated_sync}
        %62 = ascscalar.memref_address %alloc_26 : memref<16384xbf16, 4>
        %63 = ascscalar.memref_address %alloc_25 : memref<16384xbf16, 4>
        %64 = arith.select %22, %62, %63 : i64
        %65 = arith.select %22, %c6_i64, %c7_i64 : i64
        %66 = ascscalar.bind_local_ptr %64 : memref<1xbf16, 4>
        asccube.get_buf mte1, %65, 0 {cannir.buffer.generated_sync}
        asccube.get_buf mte1, %36, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0b %66[%c0], %37[%c0], %c0_i32, %c0_i32, %c8_i32, %c8_i32, %c8_i32, %c8_i32 : memref<1xbf16, 4>, memref<1xbf16, 2>
        asccube.release_buf mte1, %36, 0 {cannir.buffer.generated_sync}
        asccube.release_buf mte1, %65, 0 {cannir.buffer.generated_sync}
        %67 = arith.cmpi eq, %arg25, %c0_i64 : i64
        %68 = ascscalar.bind_local_ptr %14 : memref<1xf32, 5>
        asccube.get_buf m, %15, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %60, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %65, 0 {cannir.buffer.generated_sync}
        scf.if %67 {
          asccube.mmad %68[%c0], %61, %66, %c8_i32, %c128_i32, %c128_i32, %c0_i32 {c_matrix_init_val = true} : memref<1xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
        } else {
          asccube.mmad %68[%c0], %61, %66, %c8_i32, %c128_i32, %c128_i32, %c0_i32 {c_matrix_init_val = false} : memref<1xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
        }
        asccube.release_buf m, %65, 0 {cannir.buffer.generated_sync}
        asccube.release_buf m, %60, 0 {cannir.buffer.generated_sync}
        asccube.release_buf m, %15, 0 {cannir.buffer.generated_sync}
        scf.yield %29 : i64
      }
      %19 = ascscalar.bind_local_ptr %14 : memref<1xf32, 5>
      %20 = arith.muli %arg22, %c128_i64 : i64
      asccube.get_buf fixpipe, %15, 0 {cannir.buffer.generated_sync}
      %21 = arith.index_cast %20 : i64 to index
      asccube.copy_l0c2gm %alloc_7[%21], %19[%c0], %c128_i32, %c8_i32, %c576_i32, %c16_i32, %c0_i32 l2_cache_ctl %c0_i32 {nz2nd_en = true, relu_en = false} : memref<4608xbf16>, memref<1xf32, 5>
      asccube.release_buf fixpipe, %15, 0 {cannir.buffer.generated_sync}
      scf.yield %18, %17 : i64, i64
    }
    asccube.sync_all
    asccube.sync_block_wait s, %c13_i64
    asccube.sync_block_arrive fixpipe, %c14_i64 {mode = 0 : i32}
    asccube.sync_block_wait s, %c14_i64 {mode = 0 : i32}
    asccube.sync_block_arrive mte3, %c15_i64
    %10 = arith.cmpi sle, %8, %c2_i64 : i64
    scf.if %10 {
      %11:2 = scf.for %arg22 = %0 to %c2_i64 step %1 iter_args(%arg23 = %c0_i64, %arg24 = %c0_i64) -> (i64, i64)  : i64 {
        %12 = arith.remsi %arg22, %c2_i64 : i64
        %13 = arith.cmpi ne, %12, %c0_i64 : i64
        %14 = arith.cmpi slt, %12, %c0_i64 : i64
        %15 = arith.andi %13, %14 : i1
        %16 = arith.select %15, %c2_i64, %c0_i64 : i64
        %17 = arith.addi %12, %16 overflow<nsw> : i64
        %18 = arith.divsi %arg22, %c2_i64 : i64
        %19 = arith.extui %15 : i1 to i64
        %20 = arith.subi %18, %19 : i64
        %21 = arith.cmpi eq, %arg24, %c0_i64 : i64
        %22 = ascscalar.memref_address %alloc_24 : memref<8192xf32, 5>
        %23 = ascscalar.memref_address %alloc_23 : memref<8192xf32, 5>
        %24 = arith.select %21, %22, %23 : i64
        %25 = arith.select %21, %c6_i64, %c7_i64 : i64
        %26 = arith.addi %arg24, %c1_i64 overflow<nsw> : i64
        %27 = arith.remui %26, %c2_i64 : i64
        %28 = scf.for %arg25 = %c0_i64 to %c3_i64 step %c1_i64 iter_args(%arg26 = %arg23) -> (i64)  : i64 {
          %36 = arith.cmpi eq, %arg26, %c0_i64 : i64
          %37 = ascscalar.memref_address %alloc_22 : memref<1024xbf16, 2>
          %38 = ascscalar.memref_address %alloc_21 : memref<1024xbf16, 2>
          %39 = arith.select %36, %37, %38 : i64
          %40 = arith.cmpi ne, %arg26, %c0_i64 : i64
          %41 = arith.extui %40 : i1 to i64
          %42 = arith.addi %arg26, %c1_i64 overflow<nsw> : i64
          %43 = arith.remui %42, %c2_i64 : i64
          %44 = arith.muli %17, %c256_i64 overflow<nsw> : i64
          %45 = arith.muli %arg25, %c64_i64 overflow<nsw> : i64
          %46 = arith.addi %44, %45 overflow<nsw> : i64
          %47 = arith.addi %arg25, %c1_i64 overflow<nsw> : i64
          %48 = arith.muli %47, %c64_i64 overflow<nsw> : i64
          %49 = arith.addi %44, %48 overflow<nsw> : i64
          %50 = arith.subi %49, %46 overflow<nsw> : i64
          %51 = ascscalar.bind_local_ptr %39 : memref<1xbf16, 2>
          %52 = arith.muli %20, %c288_i64 : i64
          %53 = arith.divui %46, %c16_i64 : i64
          %54 = arith.addi %52, %53 : i64
          %55 = arith.muli %54, %c16_i64 : i64
          asccube.get_buf mte2, %41, 0 {cannir.buffer.generated_sync}
          %56 = arith.cmpi ugt, %50, %c0_i64 : i64
          scf.if %56 {
            %83 = arith.index_cast %55 : i64 to index
            %84 = arith.trunci %50 : i64 to i32
            asccube.copy_gm2l1 %51[%c0], %alloc_7[%83], %c8_i32, %84, %c1152_i64, %c1_i32, %c0_i64, %c0_i64, %c16_i32 l2_cache_ctl %c1_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<4608xbf16>
          }
          asccube.release_buf mte2, %41, 0 {cannir.buffer.generated_sync}
          %57 = ascscalar.memref_address %alloc_20 : memref<32768xbf16, 2>
          %58 = ascscalar.memref_address %alloc_19 : memref<32768xbf16, 2>
          %59 = arith.select %36, %57, %58 : i64
          %60 = arith.select %36, %c2_i64, %c3_i64 : i64
          %61 = arith.muli %17, %c192_i64 overflow<nsw> : i64
          %62 = arith.addi %61, %45 overflow<nsw> : i64
          %63 = arith.remsi %62, %c64_i64 : i64
          %64 = arith.cmpi ne, %63, %c0_i64 : i64
          %65 = arith.cmpi slt, %63, %c0_i64 : i64
          %66 = arith.andi %64, %65 : i1
          %67 = arith.divsi %62, %c64_i64 : i64
          %68 = arith.extui %66 : i1 to i64
          %69 = arith.subi %67, %68 : i64
          %70 = ascscalar.bind_local_ptr %59 : memref<1xbf16, 2>
          %71 = arith.muli %69, %c32768_i64 : i64
          asccube.get_buf mte2, %60, 0 {cannir.buffer.generated_sync}
          %72 = arith.index_cast %71 : i64 to index
          asccube.copy_gm2l1 %70[%c0], %alloc_2[%72], %c64_i32, %c512_i32, %c1024_i64, %c1_i32, %c0_i64, %c0_i64, %c64_i32 l2_cache_ctl %c0_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<196608xbf16>
          asccube.release_buf mte2, %60, 0 {cannir.buffer.generated_sync}
          %73 = ascscalar.bitcast_memref %alloc_9 byte_offset = %c0_i64 : memref<1024xbf16, 3> -> memref<?xbf16, 3>
          asccube.get_buf mte1, %c4_i64, 0 {cannir.buffer.generated_sync}
          asccube.get_buf mte1, %41, 0 {cannir.buffer.generated_sync}
          asccube.copy_l12l0a %73[%c0], %51[%c0], %c0_i32, %c0_i32, %c1_i32, %c4_i32, %c1_i32, %c1_i32 : memref<?xbf16, 3>, memref<1xbf16, 2>
          asccube.release_buf mte1, %41, 0 {cannir.buffer.generated_sync}
          asccube.release_buf mte1, %c4_i64, 0 {cannir.buffer.generated_sync}
          %74 = ascscalar.bitcast_memref %alloc_10 byte_offset = %c0_i64 : memref<8192xbf16, 4> -> memref<?xbf16, 4>
          asccube.get_buf mte1, %c5_i64, 0 {cannir.buffer.generated_sync}
          asccube.get_buf mte1, %60, 0 {cannir.buffer.generated_sync}
          asccube.copy_l12l0b %74[%c0], %70[%c0], %c0_i32, %c0_i32, %c4_i32, %c8_i32, %c4_i32, %c8_i32 {transpose = true} : memref<?xbf16, 4>, memref<1xbf16, 2>
          asccube.release_buf mte1, %60, 0 {cannir.buffer.generated_sync}
          asccube.release_buf mte1, %c5_i64, 0 {cannir.buffer.generated_sync}
          %75 = arith.cmpi eq, %arg25, %c0_i64 : i64
          %76 = ascscalar.bind_local_ptr %24 : memref<1xf32, 5>
          asccube.get_buf m, %25, 0 {cannir.buffer.generated_sync}
          asccube.get_buf m, %c4_i64, 0 {cannir.buffer.generated_sync}
          asccube.get_buf m, %c5_i64, 0 {cannir.buffer.generated_sync}
          scf.if %75 {
            asccube.mmad %76[%c0], %73, %74, %c8_i32, %c64_i32, %c128_i32, %c0_i32 {c_matrix_init_val = true} : memref<1xf32, 5>, memref<?xbf16, 3>, memref<?xbf16, 4>
          } else {
            asccube.mmad %76[%c0], %73, %74, %c8_i32, %c64_i32, %c128_i32, %c0_i32 {c_matrix_init_val = false} : memref<1xf32, 5>, memref<?xbf16, 3>, memref<?xbf16, 4>
          }
          asccube.release_buf m, %c5_i64, 0 {cannir.buffer.generated_sync}
          asccube.get_buf mte1, %c5_i64, 0 {cannir.buffer.generated_sync}
          asccube.get_buf mte1, %60, 0 {cannir.buffer.generated_sync}
          asccube.copy_l12l0b %74[%c0], %70[%c0], %c0_i32, %c8_i32, %c4_i32, %c8_i32, %c4_i32, %c8_i32 {transpose = true} : memref<?xbf16, 4>, memref<1xbf16, 2>
          asccube.release_buf mte1, %60, 0 {cannir.buffer.generated_sync}
          asccube.release_buf mte1, %c5_i64, 0 {cannir.buffer.generated_sync}
          %77 = arith.addi %24, %c8192_i64 : i64
          %78 = ascscalar.bind_local_ptr %77 : memref<1xf32, 5>
          asccube.get_buf m, %c5_i64, 0 {cannir.buffer.generated_sync}
          scf.if %75 {
            asccube.mmad %78[%c0], %73, %74, %c8_i32, %c64_i32, %c128_i32, %c0_i32 {c_matrix_init_val = true} : memref<1xf32, 5>, memref<?xbf16, 3>, memref<?xbf16, 4>
          } else {
            asccube.mmad %78[%c0], %73, %74, %c8_i32, %c64_i32, %c128_i32, %c0_i32 {c_matrix_init_val = false} : memref<1xf32, 5>, memref<?xbf16, 3>, memref<?xbf16, 4>
          }
          asccube.release_buf m, %c5_i64, 0 {cannir.buffer.generated_sync}
          asccube.get_buf mte1, %c5_i64, 0 {cannir.buffer.generated_sync}
          asccube.get_buf mte1, %60, 0 {cannir.buffer.generated_sync}
          asccube.copy_l12l0b %74[%c0], %70[%c0], %c0_i32, %c16_i32, %c4_i32, %c8_i32, %c4_i32, %c8_i32 {transpose = true} : memref<?xbf16, 4>, memref<1xbf16, 2>
          asccube.release_buf mte1, %60, 0 {cannir.buffer.generated_sync}
          asccube.release_buf mte1, %c5_i64, 0 {cannir.buffer.generated_sync}
          %79 = arith.addi %24, %c16384_i64 : i64
          %80 = ascscalar.bind_local_ptr %79 : memref<1xf32, 5>
          asccube.get_buf m, %c5_i64, 0 {cannir.buffer.generated_sync}
          scf.if %75 {
            asccube.mmad %80[%c0], %73, %74, %c8_i32, %c64_i32, %c128_i32, %c0_i32 {c_matrix_init_val = true} : memref<1xf32, 5>, memref<?xbf16, 3>, memref<?xbf16, 4>
          } else {
            asccube.mmad %80[%c0], %73, %74, %c8_i32, %c64_i32, %c128_i32, %c0_i32 {c_matrix_init_val = false} : memref<1xf32, 5>, memref<?xbf16, 3>, memref<?xbf16, 4>
          }
          asccube.release_buf m, %c5_i64, 0 {cannir.buffer.generated_sync}
          asccube.get_buf mte1, %c5_i64, 0 {cannir.buffer.generated_sync}
          asccube.get_buf mte1, %60, 0 {cannir.buffer.generated_sync}
          asccube.copy_l12l0b %74[%c0], %70[%c0], %c0_i32, %c24_i32, %c4_i32, %c8_i32, %c4_i32, %c8_i32 {transpose = true} : memref<?xbf16, 4>, memref<1xbf16, 2>
          asccube.release_buf mte1, %60, 0 {cannir.buffer.generated_sync}
          asccube.release_buf mte1, %c5_i64, 0 {cannir.buffer.generated_sync}
          %81 = arith.addi %24, %c24576_i64 : i64
          %82 = ascscalar.bind_local_ptr %81 : memref<1xf32, 5>
          asccube.get_buf m, %c5_i64, 0 {cannir.buffer.generated_sync}
          scf.if %75 {
            asccube.mmad %82[%c0], %73, %74, %c8_i32, %c64_i32, %c128_i32, %c0_i32 {c_matrix_init_val = true} : memref<1xf32, 5>, memref<?xbf16, 3>, memref<?xbf16, 4>
          } else {
            asccube.mmad %82[%c0], %73, %74, %c8_i32, %c64_i32, %c128_i32, %c0_i32 {c_matrix_init_val = false} : memref<1xf32, 5>, memref<?xbf16, 3>, memref<?xbf16, 4>
          }
          asccube.release_buf m, %c5_i64, 0 {cannir.buffer.generated_sync}
          asccube.release_buf m, %c4_i64, 0 {cannir.buffer.generated_sync}
          asccube.release_buf m, %25, 0 {cannir.buffer.generated_sync}
          scf.yield %43 : i64
        }
        %29 = arith.muli %17, %c576_i64 overflow<nsw> : i64
        %30 = ascscalar.bind_local_ptr %24 : memref<1xf32, 5>
        %31 = arith.muli %20, %c576_i64 : i64
        %32 = arith.divui %29, %c16_i64 : i64
        %33 = arith.addi %31, %32 : i64
        %34 = arith.muli %33, %c16_i64 : i64
        asccube.get_buf fixpipe, %25, 0 {cannir.buffer.generated_sync}
        %35 = arith.index_cast %34 : i64 to index
        asccube.copy_l0c2gm %alloc_3[%35], %30[%c0], %c512_i32, %c8_i32, %c1152_i32, %c16_i32, %c0_i32 l2_cache_ctl %c0_i32 {nz2nd_en = true, relu_en = false} : memref<9216xbf16>, memref<1xf32, 5>
        asccube.release_buf fixpipe, %25, 0 {cannir.buffer.generated_sync}
        scf.yield %28, %27 : i64, i64
      }
    } else {
      %11:2 = scf.for %arg22 = %0 to %c8_i64 step %1 iter_args(%arg23 = %c0_i64, %arg24 = %c0_i64) -> (i64, i64)  : i64 {
        %12 = arith.remsi %arg22, %c4_i64 : i64
        %13 = arith.cmpi ne, %12, %c0_i64 : i64
        %14 = arith.cmpi slt, %12, %c0_i64 : i64
        %15 = arith.andi %13, %14 : i1
        %16 = arith.select %15, %c4_i64, %c0_i64 : i64
        %17 = arith.addi %12, %16 overflow<nsw> : i64
        %18 = arith.divsi %arg22, %c4_i64 : i64
        %19 = arith.extui %15 : i1 to i64
        %20 = arith.subi %18, %19 : i64
        %21 = arith.remsi %20, %c2_i64 : i64
        %22 = arith.cmpi ne, %21, %c0_i64 : i64
        %23 = arith.cmpi slt, %21, %c0_i64 : i64
        %24 = arith.andi %22, %23 : i1
        %25 = arith.select %24, %c2_i64, %c0_i64 : i64
        %26 = arith.addi %21, %25 overflow<nsw> : i64
        %27 = arith.remsi %arg22, %c8_i64 : i64
        %28 = arith.cmpi ne, %27, %c0_i64 : i64
        %29 = arith.cmpi slt, %27, %c0_i64 : i64
        %30 = arith.andi %28, %29 : i1
        %31 = arith.divsi %arg22, %c8_i64 : i64
        %32 = arith.extui %30 : i1 to i64
        %33 = arith.subi %31, %32 : i64
        %34 = arith.muli %26, %c256_i64 overflow<nsw> : i64
        %35 = arith.muli %26, %c576_i64 overflow<nsw> : i64
        %36 = arith.cmpi eq, %arg24, %c0_i64 : i64
        %37 = ascscalar.memref_address %alloc_18 : memref<2048xf32, 5>
        %38 = ascscalar.memref_address %alloc_17 : memref<2048xf32, 5>
        %39 = arith.select %36, %37, %38 : i64
        %40 = arith.select %36, %c14_i64, %c15_i64 : i64
        %41 = arith.addi %arg24, %c1_i64 overflow<nsw> : i64
        %42 = arith.remui %41, %c2_i64 : i64
        %43 = scf.for %arg25 = %c0_i64 to %c3_i64 step %c1_i64 iter_args(%arg26 = %arg23) -> (i64)  : i64 {
          %52 = arith.cmpi eq, %arg26, %c0_i64 : i64
          %53 = ascscalar.memref_address %alloc_16 : memref<1024xbf16, 2>
          %54 = ascscalar.memref_address %alloc_15 : memref<1024xbf16, 2>
          %55 = arith.select %52, %53, %54 : i64
          %56 = arith.select %52, %c8_i64, %c9_i64 : i64
          %57 = arith.addi %arg26, %c1_i64 overflow<nsw> : i64
          %58 = arith.remui %57, %c2_i64 : i64
          %59 = arith.muli %arg25, %c64_i64 overflow<nsw> : i64
          %60 = arith.addi %34, %59 overflow<nsw> : i64
          %61 = arith.addi %arg25, %c1_i64 overflow<nsw> : i64
          %62 = arith.muli %61, %c64_i64 overflow<nsw> : i64
          %63 = arith.addi %34, %62 overflow<nsw> : i64
          %64 = arith.subi %63, %60 overflow<nsw> : i64
          %65 = ascscalar.bind_local_ptr %55 : memref<1xbf16, 2>
          %66 = arith.muli %33, %c288_i64 : i64
          %67 = arith.divui %60, %c16_i64 : i64
          %68 = arith.addi %66, %67 : i64
          %69 = arith.muli %68, %c16_i64 : i64
          asccube.get_buf mte2, %56, 0 {cannir.buffer.generated_sync}
          %70 = arith.cmpi ugt, %64, %c0_i64 : i64
          scf.if %70 {
            %94 = arith.index_cast %69 : i64 to index
            %95 = arith.trunci %64 : i64 to i32
            asccube.copy_gm2l1 %65[%c0], %alloc_7[%94], %c8_i32, %95, %c1152_i64, %c1_i32, %c0_i64, %c0_i64, %c16_i32 l2_cache_ctl %c1_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<4608xbf16>
          }
          asccube.release_buf mte2, %56, 0 {cannir.buffer.generated_sync}
          %71 = ascscalar.bitcast_memref %alloc_11 byte_offset = %c0_i64 : memref<1024xbf16, 3> -> memref<?xbf16, 3>
          asccube.get_buf mte1, %c12_i64, 0 {cannir.buffer.generated_sync}
          asccube.get_buf mte1, %56, 0 {cannir.buffer.generated_sync}
          asccube.copy_l12l0a %71[%c0], %65[%c0], %c0_i32, %c0_i32, %c1_i32, %c4_i32, %c1_i32, %c1_i32 : memref<?xbf16, 3>, memref<1xbf16, 2>
          asccube.release_buf mte1, %56, 0 {cannir.buffer.generated_sync}
          asccube.release_buf mte1, %c12_i64, 0 {cannir.buffer.generated_sync}
          %72 = ascscalar.memref_address %alloc_14 : memref<8192xbf16, 2>
          %73 = ascscalar.memref_address %alloc_13 : memref<8192xbf16, 2>
          %74 = arith.select %52, %72, %73 : i64
          %75 = arith.select %52, %c10_i64, %c11_i64 : i64
          %76 = arith.muli %26, %c192_i64 overflow<nsw> : i64
          %77 = arith.addi %76, %59 overflow<nsw> : i64
          %78 = arith.remsi %77, %c64_i64 : i64
          %79 = arith.cmpi ne, %78, %c0_i64 : i64
          %80 = arith.cmpi slt, %78, %c0_i64 : i64
          %81 = arith.andi %79, %80 : i1
          %82 = arith.divsi %77, %c64_i64 : i64
          %83 = arith.extui %81 : i1 to i64
          %84 = arith.subi %82, %83 : i64
          %85 = ascscalar.bind_local_ptr %74 : memref<1xbf16, 2>
          %86 = arith.muli %84, %c2048_i64 : i64
          %87 = arith.muli %17, %c8_i64 overflow<nsw> : i64
          %88 = arith.addi %86, %87 : i64
          %89 = arith.muli %88, %c16_i64 : i64
          asccube.get_buf mte2, %75, 0 {cannir.buffer.generated_sync}
          %90 = arith.index_cast %89 : i64 to index
          asccube.copy_gm2l1 %85[%c0], %alloc_2[%90], %c64_i32, %c128_i32, %c1024_i64, %c1_i32, %c0_i64, %c0_i64, %c64_i32 l2_cache_ctl %c0_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<196608xbf16>
          asccube.release_buf mte2, %75, 0 {cannir.buffer.generated_sync}
          %91 = ascscalar.bitcast_memref %alloc_12 byte_offset = %c0_i64 : memref<8192xbf16, 4> -> memref<?xbf16, 4>
          asccube.get_buf mte1, %c13_i64, 0 {cannir.buffer.generated_sync}
          asccube.get_buf mte1, %75, 0 {cannir.buffer.generated_sync}
          asccube.copy_l12l0b %91[%c0], %85[%c0], %c0_i32, %c0_i32, %c4_i32, %c8_i32, %c4_i32, %c8_i32 {transpose = true} : memref<?xbf16, 4>, memref<1xbf16, 2>
          asccube.release_buf mte1, %75, 0 {cannir.buffer.generated_sync}
          asccube.release_buf mte1, %c13_i64, 0 {cannir.buffer.generated_sync}
          %92 = arith.cmpi eq, %arg25, %c0_i64 : i64
          %93 = ascscalar.bind_local_ptr %39 : memref<1xf32, 5>
          asccube.get_buf m, %40, 0 {cannir.buffer.generated_sync}
          asccube.get_buf m, %c12_i64, 0 {cannir.buffer.generated_sync}
          asccube.get_buf m, %c13_i64, 0 {cannir.buffer.generated_sync}
          scf.if %92 {
            asccube.mmad %93[%c0], %71, %91, %c8_i32, %c64_i32, %c128_i32, %c0_i32 {c_matrix_init_val = true} : memref<1xf32, 5>, memref<?xbf16, 3>, memref<?xbf16, 4>
          } else {
            asccube.mmad %93[%c0], %71, %91, %c8_i32, %c64_i32, %c128_i32, %c0_i32 {c_matrix_init_val = false} : memref<1xf32, 5>, memref<?xbf16, 3>, memref<?xbf16, 4>
          }
          asccube.release_buf m, %c13_i64, 0 {cannir.buffer.generated_sync}
          asccube.release_buf m, %c12_i64, 0 {cannir.buffer.generated_sync}
          asccube.release_buf m, %40, 0 {cannir.buffer.generated_sync}
          scf.yield %58 : i64
        }
        %44 = arith.muli %17, %c128_i64 overflow<nsw> : i64
        %45 = arith.addi %35, %44 overflow<nsw> : i64
        %46 = ascscalar.bind_local_ptr %39 : memref<1xf32, 5>
        %47 = arith.muli %33, %c576_i64 : i64
        %48 = arith.divui %45, %c16_i64 : i64
        %49 = arith.addi %47, %48 : i64
        %50 = arith.muli %49, %c16_i64 : i64
        asccube.get_buf fixpipe, %40, 0 {cannir.buffer.generated_sync}
        %51 = arith.index_cast %50 : i64 to index
        asccube.copy_l0c2gm %alloc_3[%51], %46[%c0], %c128_i32, %c8_i32, %c1152_i32, %c16_i32, %c0_i32 l2_cache_ctl %c0_i32 {nz2nd_en = true, relu_en = false} : memref<9216xbf16>, memref<1xf32, 5>
        asccube.release_buf fixpipe, %40, 0 {cannir.buffer.generated_sync}
        scf.yield %43, %42 : i64, i64
      }
    }
    return
  }
}
