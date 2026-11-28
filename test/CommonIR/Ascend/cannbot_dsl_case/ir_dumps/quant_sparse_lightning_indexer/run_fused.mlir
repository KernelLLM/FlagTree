module {
  func.func @run_fused_846f65dd336116ec1fa2193eb36cdafb(%arg0: i64, %arg1: !cannir.tensor<<gm, ui8>, <<(-1, 1, 64)>, <(-1, 64, 1)>>, <<(-1, 1, 64)>, <(-1, 64, 1)>>, identity>, %arg2: i64, %arg3: !cannir.tensor<<gm, ui8>, <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>>, <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>>, identity>, %arg4: i64, %arg5: !cannir.tensor<<gm, f32>, <<(-1, 32)>, <(32, 1)>>, <<(-1, 32)>, <(32, 1)>>, identity>, %arg6: !cannir.tensor<<gm, !ascdtype.fp8_e8m0>, <<(-1, 2, 2)>, <(4, 2, 1)>>, <<(-1, 2, 2)>, <(4, 2, 1)>>, identity>, %arg7: !cannir.tensor<<gm, i32>, <<(-1, 1, 2048)>, <(2048, 2048, 1)>>, <<(-1, 1, 2048)>, <(2048, 2048, 1)>>, identity>, %arg8: !cannir.tensor<<gm, i32>, <<(-1, 1)>, <(1, 1)>>, <<(-1, 1)>, <(1, 1)>>, identity>, %arg9: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg10: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg11: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg12: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg13: !cannir.tensor<<gm, i32>, <<(-1, 1)>, <(1, 1)>>, <<(-1, 1)>, <(1, 1)>>, identity>, %arg14: !cannir.tensor<<gm, i32>, <<(-1, 1)>, <(1, 1)>>, <<(-1, 1)>, <(1, 1)>>, identity>, %arg15: !cannir.tensor<<gm, ui8>, <<(73728, 64)>, <(64, 1)>>, <<(73728, 64)>, <(64, 1)>>, identity>, %arg16: !cannir.tensor<<gm, !ascdtype.fp4x2_e2m1>, <<(73728, 128)>, <(128, 1)>>, <<(73728, 128)>, <(128, 1)>>, identity>, %arg17: !cannir.tensor<<gm, ui8>, <<(73728, 4)>, <(4, 1)>>, <<(73728, 4)>, <(4, 1)>>, identity>, %arg18: !cannir.tensor<<gm, !ascdtype.fp8_e8m0>, <<(73728, 2, 2)>, <(4, 2, 1)>>, <<(73728, 2, 2)>, <(4, 2, 1)>>, identity>, %arg19: !cannir.tensor<<gm, ui16>, <<(-1, 16384)>, <(16384, 1)>>, <<(-1, 16384)>, <(16384, 1)>>, identity>, %arg20: i64, %arg21: !cannir.tensor<<gm, i32>, <<(-1, 8192)>, <(8192, 1)>>, <<(-1, 8192)>, <(8192, 1)>>, identity>, %arg22: i64, %arg23: i64, %arg24: i64, %arg25: i64, %arg26: i64, %arg27: i64, %arg28: !cannir.tensor<<gm, i32>, <<(128, 8)>, <(8, 1)>>, <<(128, 8)>, <(8, 1)>>, identity>, %arg29: i64, %arg30: i64, %arg31: i64, %arg32: i64, %arg33: i64) attributes {cannbotdsl.contract_digest = "846f65dd336116ec1fa2193eb36cdafb6385bd7050d15f46ceb72499b64971f9", cannbotdsl.public_slot_count = 34 : i64, cannbotdsl.slot_dtype_bits = [64, 8, 64, 8, 64, 32, 8, 32, 32, 32, 32, 32, 32, 32, 32, 8, 8, 8, 8, 16, 64, 32, 64, 64, 64, 64, 64, 64, 32, 64, 64, 64, 64, 64], cannbotdsl.slot_dtype_codes = [0, 1, 0, 1, 0, 2, 14, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 14, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], cannbotdsl.slot_dtype_lanes = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], cannbotdsl.slot_flags = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], cannbotdsl.slot_kinds = [2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2], cannbotdsl.slot_type_digests = ["385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "7f35f932506cf308dea7116c08821f7cc368951c3af2830f09e24287b22218aa", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "413fdef20af3527209c1658dc8700898cde647ffe13c65f93dcef77b804173fe", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "303efa15a10b189586194340f9acf94f49608a8d0ff96167ef4f24d50ee66e03", "f23a2a226c424d9c13d010802f1fd872bf7fbb10f043d43b23f2b47c05c692f9", "af386738adeb9da7641befcca3f8abe025a79b20722348d07d7599f7c98dc720", "262d13f4c64244b3e843cd33c0ef202296d91d37afeffa5a24ffc09c5f638dd1", "340989c846395acc3ce70fa11aac94805b9dd8f66445e397dbeb03af10ef82c9", "e030c76f69ec395f0341ef8f9ab18ddc826e6df4047f270d22f84aa569a4ad93", "340989c846395acc3ce70fa11aac94805b9dd8f66445e397dbeb03af10ef82c9", "e030c76f69ec395f0341ef8f9ab18ddc826e6df4047f270d22f84aa569a4ad93", "262d13f4c64244b3e843cd33c0ef202296d91d37afeffa5a24ffc09c5f638dd1", "262d13f4c64244b3e843cd33c0ef202296d91d37afeffa5a24ffc09c5f638dd1", "8f0aef591096c393599e92ed390f7e4bb282c7bfc9cbb345c96e5408d8cac4d2", "55db2619015518a59e0d2b02314709d9204e45dac4ca4fd1903cf9c5c2a3f447", "1f1f4f046f9dae193b49dea9e88407a98d8222e8ddfeb17b1d570032d1ebb78c", "09cd160a2e712a479120111e1576cd52043581d2739689926f5e847412317e52", "de548474fda71ab77a62ff7f0705f742dcd213b142b85374d3034808e8f91576", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "684cb7f9e0aae0da05d2d3ab2dc34696f464dcc4e357ce8e083a8b410cd0db46", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "cebc6a6e5577b0a5f53b2408bb132f57e156393513d83c69eedf6e9e026ef584", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515"], cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.host} {
    cannir.kernel_launch @kernel_QuantSparseLightningIndexer_0(%arg24) [%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18, %arg19, %arg20, %arg21, %arg22, %arg23, %arg24, %arg25, %arg26, %arg27, %arg28, %arg29, %arg30, %arg31, %arg32, %arg33] {cannbotdsl.profile = {indices = array<i64>, name = "QuantSparseLightningIndexer", op_type = "QuantSparseLightningIndexer", roles = array<i64>}} : i64, !cannir.tensor<<gm, ui8>, <<(-1, 1, 64)>, <(-1, 64, 1)>>, <<(-1, 1, 64)>, <(-1, 64, 1)>>, identity>, i64, !cannir.tensor<<gm, ui8>, <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>>, <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>>, identity>, i64, !cannir.tensor<<gm, f32>, <<(-1, 32)>, <(32, 1)>>, <<(-1, 32)>, <(32, 1)>>, identity>, !cannir.tensor<<gm, !ascdtype.fp8_e8m0>, <<(-1, 2, 2)>, <(4, 2, 1)>>, <<(-1, 2, 2)>, <(4, 2, 1)>>, identity>, !cannir.tensor<<gm, i32>, <<(-1, 1, 2048)>, <(2048, 2048, 1)>>, <<(-1, 1, 2048)>, <(2048, 2048, 1)>>, identity>, !cannir.tensor<<gm, i32>, <<(-1, 1)>, <(1, 1)>>, <<(-1, 1)>, <(1, 1)>>, identity>, !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, !cannir.tensor<<gm, i32>, <<(-1, 1)>, <(1, 1)>>, <<(-1, 1)>, <(1, 1)>>, identity>, !cannir.tensor<<gm, i32>, <<(-1, 1)>, <(1, 1)>>, <<(-1, 1)>, <(1, 1)>>, identity>, !cannir.tensor<<gm, ui8>, <<(73728, 64)>, <(64, 1)>>, <<(73728, 64)>, <(64, 1)>>, identity>, !cannir.tensor<<gm, !ascdtype.fp4x2_e2m1>, <<(73728, 128)>, <(128, 1)>>, <<(73728, 128)>, <(128, 1)>>, identity>, !cannir.tensor<<gm, ui8>, <<(73728, 4)>, <(4, 1)>>, <<(73728, 4)>, <(4, 1)>>, identity>, !cannir.tensor<<gm, !ascdtype.fp8_e8m0>, <<(73728, 2, 2)>, <(4, 2, 1)>>, <<(73728, 2, 2)>, <(4, 2, 1)>>, identity>, !cannir.tensor<<gm, ui16>, <<(-1, 16384)>, <(16384, 1)>>, <<(-1, 16384)>, <(16384, 1)>>, identity>, i64, !cannir.tensor<<gm, i32>, <<(-1, 8192)>, <(8192, 1)>>, <<(-1, 8192)>, <(8192, 1)>>, identity>, i64, i64, i64, i64, i64, i64, !cannir.tensor<<gm, i32>, <<(128, 8)>, <(8, 1)>>, <<(128, 8)>, <(8, 1)>>, identity>, i64, i64, i64, i64, i64
    return
  }
  func.func @kernel_QuantSparseLightningIndexer_0_aiv(%arg0: i64, %arg1: !cannir.tensor<<gm, ui8>, <<(-1, 1, 64)>, <(-1, 64, 1)>>, <<(-1, 1, 64)>, <(-1, 64, 1)>>, identity>, %arg2: i64, %arg3: !cannir.tensor<<gm, ui8>, <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>>, <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>>, identity>, %arg4: i64, %arg5: !cannir.tensor<<gm, f32>, <<(-1, 32)>, <(32, 1)>>, <<(-1, 32)>, <(32, 1)>>, identity>, %arg6: !cannir.tensor<<gm, !ascdtype.fp8_e8m0>, <<(-1, 2, 2)>, <(4, 2, 1)>>, <<(-1, 2, 2)>, <(4, 2, 1)>>, identity>, %arg7: !cannir.tensor<<gm, i32>, <<(-1, 1, 2048)>, <(2048, 2048, 1)>>, <<(-1, 1, 2048)>, <(2048, 2048, 1)>>, identity>, %arg8: !cannir.tensor<<gm, i32>, <<(-1, 1)>, <(1, 1)>>, <<(-1, 1)>, <(1, 1)>>, identity>, %arg9: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg10: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg11: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg12: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg13: !cannir.tensor<<gm, i32>, <<(-1, 1)>, <(1, 1)>>, <<(-1, 1)>, <(1, 1)>>, identity>, %arg14: !cannir.tensor<<gm, i32>, <<(-1, 1)>, <(1, 1)>>, <<(-1, 1)>, <(1, 1)>>, identity>, %arg15: !cannir.tensor<<gm, ui8>, <<(73728, 64)>, <(64, 1)>>, <<(73728, 64)>, <(64, 1)>>, identity>, %arg16: !cannir.tensor<<gm, !ascdtype.fp4x2_e2m1>, <<(73728, 128)>, <(128, 1)>>, <<(73728, 128)>, <(128, 1)>>, identity>, %arg17: !cannir.tensor<<gm, ui8>, <<(73728, 4)>, <(4, 1)>>, <<(73728, 4)>, <(4, 1)>>, identity>, %arg18: !cannir.tensor<<gm, !ascdtype.fp8_e8m0>, <<(73728, 2, 2)>, <(4, 2, 1)>>, <<(73728, 2, 2)>, <(4, 2, 1)>>, identity>, %arg19: !cannir.tensor<<gm, ui16>, <<(-1, 16384)>, <(16384, 1)>>, <<(-1, 16384)>, <(16384, 1)>>, identity>, %arg20: i64, %arg21: !cannir.tensor<<gm, i32>, <<(-1, 8192)>, <(8192, 1)>>, <<(-1, 8192)>, <(8192, 1)>>, identity>, %arg22: i64, %arg23: i64, %arg24: i64, %arg25: i64, %arg26: i64, %arg27: i64, %arg28: !cannir.tensor<<gm, i32>, <<(128, 8)>, <(8, 1)>>, <<(128, 8)>, <(8, 1)>>, identity>, %arg29: i64, %arg30: i64, %arg31: i64, %arg32: i64, %arg33: i64) attributes {cannir.buffer.auto_sync, cannir.buffer.sync_analyzed, cannir.buffer.sync_verified, cannir.buffer.tensor_state, cannir.core = "aiv", cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.device, cannir.explicit_ring_buffer_cursors, cannir.kernel_group = "kernel_QuantSparseLightningIndexer_0", cannir.resource.ids_prepared} {
    %c544_i32 = arith.constant 544 : i32
    %c512_i32 = arith.constant 512 : i32
    %c5_i32 = arith.constant 5 : i32
    %c6_i32 = arith.constant 6 : i32
    %c2048 = arith.constant 2048 : index
    %c16384_i32 = arith.constant 16384 : i32
    %c32768_i32 = arith.constant 32768 : i32
    %c3 = arith.constant 3 : index
    %c6 = arith.constant 6 : index
    %c5 = arith.constant 5 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c7 = arith.constant 7 : index
    %c8 = arith.constant 8 : index
    %c288 = arith.constant 288 : index
    %c0_i32 = arith.constant 0 : i32
    %c32_i64 = arith.constant 32 : i64
    %c1_i32 = arith.constant 1 : i32
    %c64_i32 = arith.constant 64 : i32
    %c64_i64 = arith.constant 64 : i64
    %false = arith.constant false
    %true = arith.constant true
    %c8192_i64 = arith.constant 8192 : i64
    %c63_i32 = arith.constant 63 : i32
    %c0 = arith.constant 0 : index
    %c32_i32 = arith.constant 32 : i32
    %c128_i64 = arith.constant 128 : i64
    %c512_i64 = arith.constant 512 : i64
    %c256_i64 = arith.constant 256 : i64
    %c544_i64 = arith.constant 544 : i64
    %c7_i32 = arith.constant 7 : i32
    %c4_i32 = arith.constant 4 : i32
    %c3_i32 = arith.constant 3 : i32
    %c2_i32 = arith.constant 2 : i32
    %c32768_i64 = arith.constant 32768 : i64
    %c255_i64 = arith.constant 255 : i64
    %c127_i64 = arith.constant 127 : i64
    %c36_i64 = arith.constant 36 : i64
    %c16384_i64 = arith.constant 16384 : i64
    %c65536_i64 = arith.constant 65536 : i64
    %c63_i64 = arith.constant 63 : i64
    %c2 = arith.constant 2 : index
    %c4096_i64 = arith.constant 4096 : i64
    %c262144_i64 = arith.constant 262144 : i64
    %c131072_i64 = arith.constant 131072 : i64
    %c4352_i64 = arith.constant 4352 : i64
    %c-8191_i64 = arith.constant -8191 : i64
    %c16_i64 = arith.constant 16 : i64
    %c14_i64 = arith.constant 14 : i64
    %c10_i64 = arith.constant 10 : i64
    %c8_i64 = arith.constant 8 : i64
    %c7_i64 = arith.constant 7 : i64
    %c6_i64 = arith.constant 6 : i64
    %c5_i64 = arith.constant 5 : i64
    %c4_i64 = arith.constant 4 : i64
    %c3_i64 = arith.constant 3 : i64
    %c2_i64 = arith.constant 2 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %alloc = memref.alloc() {cannir.arg_index = 5 : i64} : memref<1xf32>
    %alloc_0 = memref.alloc() {cannir.arg_index = 7 : i64} : memref<1xi32>
    %alloc_1 = memref.alloc() {cannir.arg_index = 8 : i64} : memref<1xi32>
    %alloc_2 = memref.alloc() {cannir.arg_index = 9 : i64} : memref<1xi32>
    %alloc_3 = memref.alloc() {cannir.arg_index = 10 : i64} : memref<1xi32>
    %alloc_4 = memref.alloc() {cannir.arg_index = 11 : i64} : memref<1xi32>
    %alloc_5 = memref.alloc() {cannir.arg_index = 12 : i64} : memref<1xi32>
    %alloc_6 = memref.alloc() {cannir.arg_index = 14 : i64} : memref<1xi32>
    %alloc_7 = memref.alloc() {cannir.arg_index = 15 : i64} : memref<4718592xui8>
    %alloc_8 = memref.alloc() {cannir.arg_index = 17 : i64} : memref<294912xui8>
    %alloc_9 = memref.alloc() {cannir.arg_index = 19 : i64} : memref<1xui16>
    %alloc_10 = memref.alloc() {cannir.arg_index = 28 : i64} : memref<1024xi32>
    %alloc_11 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<16384xbf16, 1>
    %alloc_12 = memref.alloc() {cannir.byte_offset = 32768 : i64} : memref<17408xui8, 1>
    %alloc_13 = memref.alloc() {cannir.byte_offset = 50176 : i64} : memref<17408xui8, 1>
    %alloc_14 = memref.alloc() {cannir.byte_offset = 67584 : i64} : memref<1024xui32, 1>
    %alloc_15 = memref.alloc() {cannir.byte_offset = 71680 : i64} : memref<32xf32, 1>
    %alloc_16 = memref.alloc() {cannir.byte_offset = 71808 : i64} : memref<32xbf16, 1>
    %alloc_17 = memref.alloc() {cannir.byte_offset = 71872 : i64} : memref<16384xui16, 1>
    %alloc_18 = memref.alloc() {cannir.byte_offset = 104640 : i64} : memref<2048xi32, 1>
    %alloc_19 = memref.alloc() {cannir.byte_offset = 112832 : i64} : memref<512xui16, 1>
    %alloc_20 = memref.alloc() {cannir.byte_offset = 113856 : i64} : memref<24576xui16, 1>
    %alloc_21 = memref.alloc() {cannir.byte_offset = 163008 : i64} : memref<256xui16, 1>
    %alloc_22 = memref.alloc() {cannir.byte_offset = 163520 : i64} : memref<64xi32, 1>
    %alloc_23 = memref.alloc() {cannir.byte_offset = 163776 : i64} : memref<8192xi32, 1>
    %alloc_24 = memref.alloc() {cannir.byte_offset = 196544 : i64} : memref<8192xui16, 1>
    %alloc_25 = memref.alloc() {cannir.byte_offset = 212928 : i64} : memref<8xi32, 1>
    %alloc_26 = memref.alloc() {cannir.byte_offset = 212960 : i64} : memref<2112xi32, 1>
    %alloc_27 = memref.alloc() {cannir.byte_offset = 221408 : i64} : memref<8192xi32, 1>
    %alloc_28 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<24576xui16, 1>
    %alloc_29 = memref.alloc() {cannir.byte_offset = 49152 : i64} : memref<24576xui16, 1>
    %alloc_30 = memref.alloc() {cannir.byte_offset = 98304 : i64} : memref<256xui16, 1>
    %alloc_31 = memref.alloc() {cannir.byte_offset = 98816 : i64} : memref<64xi32, 1>
    %alloc_32 = memref.alloc() {cannir.byte_offset = 99072 : i64} : memref<8192xi32, 1>
    %alloc_33 = memref.alloc() {cannir.byte_offset = 131840 : i64} : memref<8192xui16, 1>
    %alloc_34 = memref.alloc() {cannir.byte_offset = 148224 : i64} : memref<16384xi32, 1>
    %alloc_35 = memref.alloc() {cannir.byte_offset = 213760 : i64} : memref<16384xui16, 1>
    %0 = memref.load %alloc_10[%c288] : memref<1024xi32>
    %1 = arith.cmpi ne, %0, %c0_i32 : i32
    %2 = ascscalar.bind_gm_ptr %arg29 : memref<1xi32>
    %3 = ascscalar.bind_gm_ptr %arg30 : memref<1xui16>
    %4 = arith.muli %arg24, %c2_i64 : i64
    %5 = ascscalar.bind_gm_ptr %arg22 : memref<1xi32>
    %6 = ascscalar.bind_gm_ptr %arg23 : memref<1xui16>
    %7 = ascscalar.get_subblock_id
    %8 = ascscalar.get_block_idx
    %9 = arith.index_cast %8 : i64 to index
    %10 = arith.muli %9, %c8 : index
    %11 = arith.addi %10, %c7 : index
    %12 = memref.load %alloc_10[%11] : memref<1024xi32>
    %13 = arith.extsi %12 : i32 to i64
    %14 = memref.load %alloc_10[%10] : memref<1024xi32>
    %15 = arith.cmpi ne, %14, %c0_i32 : i32
    scf.if %15 {
      ascvec.sync_intra_arrive v, %c0_i64
      %16 = arith.addi %10, %c1 : index
      %17 = memref.load %alloc_10[%16] : memref<1024xi32>
      %18 = arith.extsi %17 : i32 to i64
      %19 = arith.addi %10, %c4 : index
      %20 = memref.load %alloc_10[%19] : memref<1024xi32>
      %21 = arith.extsi %20 : i32 to i64
      %22 = arith.addi %21, %c1_i64 : i64
      %23 = arith.minsi %22, %arg26 : i64
      %24:2 = scf.for %arg34 = %18 to %23 step %c1_i64 iter_args(%arg35 = %13, %arg36 = %c0_i64) -> (i64, i64)  : i64 {
        %25 = arith.index_cast %arg34 : i64 to index
        %26 = memref.load %alloc_2[%25] : memref<1xi32>
        %27 = arith.extsi %26 : i32 to i64
        %28 = arith.addi %arg34, %c1_i64 : i64
        %29 = arith.index_cast %28 : i64 to index
        %30 = memref.load %alloc_2[%29] : memref<1xi32>
        %31 = arith.extsi %30 : i32 to i64
        %32 = memref.load %alloc_3[%25] : memref<1xi32>
        %33 = memref.load %alloc_5[%25] : memref<1xi32>
        %34 = arith.subi %31, %27 : i64
        %35 = arith.cmpi eq, %arg34, %18 : i64
        %36 = scf.if %35 -> (i64) {
          %40 = arith.addi %10, %c2 : index
          %41 = memref.load %alloc_10[%40] : memref<1024xi32>
          %42 = arith.extsi %41 : i32 to i64
          scf.yield %42 : i64
        } else {
          scf.yield %c0_i64 : i64
        }
        %37 = arith.cmpi eq, %arg34, %21 : i64
        %38 = scf.if %37 -> (i64) {
          %40 = arith.addi %10, %c5 : index
          %41 = memref.load %alloc_10[%40] : memref<1024xi32>
          %42 = arith.extsi %41 : i32 to i64
          %43 = arith.addi %10, %c6 : index
          %44 = memref.load %alloc_10[%43] : memref<1024xi32>
          %45 = arith.cmpi sgt, %44, %c0_i32 : i32
          %46 = arith.extui %45 : i1 to i64
          %47 = arith.addi %42, %46 : i64
          scf.yield %47 : i64
        } else {
          scf.yield %34 : i64
        }
        ascvec.vf(%alloc_14 : memref<1024xui32, 1>) {
        ^bb0(%arg37: memref<1024xui32, 1>):
          %c1_i32_36 = arith.constant 1 : i32
          %c16_i32 = arith.constant 16 : i32
          %c64_i32_37 = arith.constant 64 : i32
          %c0_i32_38 = arith.constant 0 : i32
          %40 = ascvec.create_mask : !ascvec.mask
          %41 = ascvec.duplicate_scalar %c0_i32_38, %40 : i32 -> <ui32>
          scf.for %arg38 = %c0_i32_38 to %c16_i32 step %c1_i32_36  : i32 {
            %42 = arith.muli %arg38, %c64_i32_37 overflow<nsw> : i32
            %43 = arith.index_cast %42 : i32 to index
            ascvec.storealign %arg37[%43], %41, %40 : memref<1024xui32, 1>, index, <ui32>
          } {cannir.loop_unroll = 1 : i64}
        } {cannir.vf_mode = "simd"}
        %39:3 = scf.for %arg37 = %36 to %38 step %c1_i64 iter_args(%arg38 = %arg35, %arg39 = %c0_i64, %arg40 = %arg36) -> (i64, i64, i64)  : i64 {
          %40 = arith.addi %27, %arg37 : i64
          %41 = arith.index_cast %40 : i64 to index
          %42 = memref.load %alloc_1[%41] : memref<1xi32>
          %43 = arith.extsi %42 : i32 to i64
          %44 = arith.addi %43, %c63_i64 : i64
          %45 = arith.remsi %44, %c64_i64 : i64
          %46 = arith.cmpi ne, %45, %c0_i64 : i64
          %47 = arith.cmpi slt, %45, %c0_i64 : i64
          %48 = arith.andi %46, %47 : i1
          %49 = arith.divsi %44, %c64_i64 : i64
          %50 = arith.extui %48 : i1 to i64
          %51 = arith.subi %49, %50 : i64
          %52 = arith.cmpi eq, %arg37, %36 : i64
          %53 = arith.select %35, %52, %false : i1
          %54 = scf.if %53 -> (i64) {
            %78 = arith.addi %10, %c3 : index
            %79 = memref.load %alloc_10[%78] : memref<1024xi32>
            %80 = arith.extsi %79 : i32 to i64
            scf.yield %80 : i64
          } else {
            scf.yield %c0_i64 : i64
          }
          %55 = scf.if %37 -> (i1) {
            %78 = arith.addi %10, %c5 : index
            %79 = memref.load %alloc_10[%78] : memref<1024xi32>
            %80 = arith.extsi %79 : i32 to i64
            %81 = arith.cmpi eq, %arg37, %80 : i64
            scf.yield %81 : i1
          } else {
            scf.yield %false : i1
          }
          %56 = scf.if %55 -> (i64) {
            %78 = arith.addi %10, %c6 : index
            %79 = memref.load %alloc_10[%78] : memref<1024xi32>
            %80 = arith.extsi %79 : i32 to i64
            %81 = arith.minsi %51, %80 : i64
            scf.yield %81 : i64
          } else {
            scf.yield %51 : i64
          }
          %57 = arith.extsi %32 : i32 to i64
          %58 = arith.cmpi sge, %arg37, %57 : i64
          %59 = arith.select %58, %c0_i32, %33 : i32
          %60 = arith.cmpi sgt, %59, %c0_i32 : i32
          %61 = arith.cmpi sgt, %54, %c0_i64 : i64
          %62 = arith.cmpi slt, %56, %51 : i64
          %63 = arith.select %61, %true, %62 : i1
          %64 = arith.select %60, %63, %false : i1
          %65 = arith.select %64, %arg38, %40 : i64
          %66 = arith.select %64, %arg22, %arg29 : i64
          %67 = arith.select %64, %arg23, %arg30 : i64
          %68 = arith.cmpi eq, %7, %c0_i64 : i64
          scf.if %68 {
            ascvec.get_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
            ascvec.get_buf v, %c5_i64, 0 {cannir.buffer.generated_sync}
            ascvec.vf(%alloc_23, %alloc_24 : memref<8192xi32, 1>, memref<8192xui16, 1>) {
            ^bb0(%arg41: memref<8192xi32, 1>, %arg42: memref<8192xui16, 1>):
              %c1_i32_36 = arith.constant 1 : i32
              %c128_i32 = arith.constant 128 : i32
              %c0_i32_37 = arith.constant 0 : i32
              %c-128_i16 = arith.constant -128 : i16
              %c-1_i32 = arith.constant -1 : i32
              %c8192_i32 = arith.constant 8192 : i32
              %c64_i32_38 = arith.constant 64 : i32
              scf.for %arg43 = %c0_i32_37 to %c128_i32 step %c1_i32_36  : i32 {
                %88 = arith.muli %arg43, %c64_i32_38 overflow<nsw> : i32
                %89 = arith.subi %c8192_i32, %88 overflow<nsw> : i32
                %mask, %remaining_out = ascvec.update_mask %89 : !ascvec.mask
                %mask_39, %remaining_out_40 = ascvec.update_mask %89 elem_bits 16 : !ascvec.mask
                %mask_41, %remaining_out_42 = ascvec.update_mask %c64_i32_38 elem_bits 16 : !ascvec.mask
                %mask_43, %remaining_out_44 = ascvec.update_mask %c64_i32_38 elem_bits 16 : !ascvec.mask
                %90 = ascvec.mask_and %mask_39, %mask_41, %mask_43
                %91 = ascvec.duplicate_scalar %c-1_i32, %mask : i32 -> <i32>
                %92 = arith.index_cast %88 : i32 to index
                ascvec.storealign %arg41[%92], %91, %mask : memref<8192xi32, 1>, index, <i32>
                %93 = ascvec.duplicate_scalar %c-128_i16, %90 : i16 -> <ui16>
                ascvec.storealign %arg42[%92], %93, %90 : memref<8192xui16, 1>, index, <ui16>
              } {cannir.loop_unroll = 1 : i64}
              ascvec.mem_bar vst_vld
            } {cannir.vf_mode = "simd"}
            ascvec.release_buf v, %c5_i64, 0 {cannir.buffer.generated_sync}
            ascvec.release_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
            %78 = arith.muli %65, %c262144_i64 : i64
            %79 = arith.divui %78, %c8_i64 : i64
            %80 = arith.addi %66, %79 : i64
            %81 = ascscalar.bind_gm_ptr %80 : memref<1xi32>
            %82 = ascscalar.bitcast_memref %alloc_23 byte_offset = %c0_i64 : memref<8192xi32, 1> -> memref<?xi32, 1>
            ascvec.get_buf mte3, %c4_i64, 0 {cannir.buffer.generated_sync}
            ascvec.copy_ub2gm %81[%c0], %82[%c0], %c1_i32, %c32768_i32, %c0_i64, %c32768_i32 l2_cache_ctl %c0_i32 : memref<1xi32>, memref<?xi32, 1>
            ascvec.release_buf mte3, %c4_i64, 0 {cannir.buffer.generated_sync}
            %83 = arith.muli %65, %c131072_i64 : i64
            %84 = arith.divui %83, %c8_i64 : i64
            %85 = arith.addi %67, %84 : i64
            %86 = ascscalar.bind_gm_ptr %85 : memref<1xui16>
            %87 = ascscalar.bitcast_memref %alloc_24 byte_offset = %c0_i64 : memref<8192xui16, 1> -> memref<?xui16, 1>
            ascvec.get_buf mte3, %c5_i64, 0 {cannir.buffer.generated_sync}
            ascvec.copy_ub2gm %86[%c0], %87[%c0], %c1_i32, %c16384_i32, %c0_i64, %c16384_i32 l2_cache_ctl %c0_i32 : memref<1xui16>, memref<?xui16, 1>
            ascvec.release_buf mte3, %c5_i64, 0 {cannir.buffer.generated_sync}
          }
          %69 = arith.addi %arg38, %c1_i64 : i64
          %70 = arith.select %64, %69, %arg38 : i64
          %71 = arith.muli %54, %c64_i64 : i64
          %72 = arith.muli %56, %c64_i64 : i64
          %73 = arith.cmpi sgt, %42, %c0_i32 : i32
          %74 = arith.cmpi sgt, %56, %54 : i64
          %75 = arith.select %74, %60, %false : i1
          %76 = arith.select %73, %75, %false : i1
          %77:2 = scf.if %76 -> (i64, i64) {
            scf.if %68 {
              %103 = arith.trunci %71 : i64 to i32
              %104 = arith.trunci %72 : i64 to i32
              %105 = arith.muli %40, %c8192_i64 : i64
              ascvec.get_buf mte2, %c7_i64, 0 {cannir.buffer.generated_sync}
              "ascvec.copy_gm2ub_plan"(%alloc_26, %alloc_0, %c0_i32, %c0_i32, %c0_i64, %105, %c1_i64, %c8192_i64, %c0_i64, %c8192_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = i32, ub_base = 212960 : i64, ub_capacity = 8448 : i64}> : (memref<2112xi32, 1>, memref<1xi32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
              ascvec.release_buf mte2, %c7_i64, 0 {cannir.buffer.generated_sync}
              %106 = arith.remsi %103, %c64_i32 : i32
              %107 = arith.cmpi ne, %106, %c0_i32 : i32
              %108 = arith.cmpi slt, %106, %c0_i32 : i32
              %109 = arith.andi %107, %108 : i1
              %110 = arith.divsi %103, %c64_i32 : i32
              %111 = arith.extui %109 : i1 to i32
              %112 = arith.subi %110, %111 : i32
              %113 = arith.addi %104, %c63_i32 : i32
              %114 = arith.remsi %113, %c64_i32 : i32
              %115 = arith.cmpi ne, %114, %c0_i32 : i32
              %116 = arith.cmpi slt, %114, %c0_i32 : i32
              %117 = arith.andi %115, %116 : i1
              %118 = arith.divsi %113, %c64_i32 : i32
              %119 = arith.extui %117 : i1 to i32
              %120 = arith.subi %118, %119 : i32
              ascvec.get_buf v, %c7_i64, 0 {cannir.buffer.generated_sync}
              ascvec.vf(%alloc_26, %59, %42, %103, %104, %alloc_18, %112, %120, %alloc_25 : memref<2112xi32, 1>, i32, i32, i32, i32, memref<2048xi32, 1>, i32, i32, memref<8xi32, 1>) {
              ^bb0(%arg41: memref<2112xi32, 1>, %arg42: i32, %arg43: i32, %arg44: i32, %arg45: i32, %arg46: memref<2048xi32, 1>, %arg47: i32, %arg48: i32, %arg49: memref<8xi32, 1>):
                %c1_i32_36 = arith.constant 1 : i32
                %c8_i32 = arith.constant 8 : i32
                %c0_37 = arith.constant 0 : index
                %c0_i32_38 = arith.constant 0 : i32
                %c64_i32_39 = arith.constant 64 : i32
                %mask, %remaining_out = ascvec.update_mask %c64_i32_39 : !ascvec.mask
                %121 = ascvec.duplicate_scalar %c0_i32_38, %mask : i32 -> <i32>
                %122 = scf.for %arg50 = %arg47 to %arg48 step %c1_i32_36 iter_args(%arg51 = %121) -> (!ascvec.vreg<i32>)  : i32 {
                  %123 = arith.muli %arg50, %c64_i32_39 : i32
                  %124 = arith.index_cast %123 : i32 to index
                  %125 = ascvec.loadalign %arg41[%124] : memref<2112xi32, 1>, index -> <i32>
                  %126 = ascvec.duplicate_scalar %arg42, %mask : i32 -> <i32>
                  %127 = ascvec.mul_scalar %125, %c8_i32, %mask : <i32>, i32
                  %128 = ascvec.sub %126, %127, %mask : <i32>
                  %129 = ascvec.max_scalar %128, %c0_i32_38, %mask : <i32>, i32
                  %130 = ascvec.min_scalar %129, %c8_i32, %mask : <i32>, i32
                  %131 = ascvec.arange %123 : i32 -> <i32>
                  %132 = ascvec.cmp_lt_scalar %131, %arg43, %mask : <i32>, i32 -> !ascvec.mask
                  %133 = ascvec.cmp_ge_scalar %131, %arg44, %mask : <i32>, i32 -> !ascvec.mask
                  %134 = ascvec.cmp_lt_scalar %131, %arg45, %mask : <i32>, i32 -> !ascvec.mask
                  %135 = ascvec.mask_and %133, %134, %mask
                  %136 = ascvec.mask_and %132, %135, %mask
                  %137 = ascvec.select %130, %121, %136 : <i32>
                  ascvec.storealign %arg46[%124], %137, %mask : memref<2048xi32, 1>, index, <i32>
                  %138 = ascvec.add %arg51, %137, %mask : <i32>
                  scf.yield %138 : !ascvec.vreg<i32>
                } {cannir.loop_unroll = 1 : i64}
                %c0_40 = arith.constant 0 : index
                %c1_41 = arith.constant 1 : index
                scf.for %arg50 = %c0_40 to %c1_41 step %c1_41 {
                  %123 = ascvec.reduce_sum %122, %mask : <i32>
                  %mask_42, %remaining_out_43 = ascvec.update_mask %c1_i32_36 : !ascvec.mask
                  ascvec.storealign %arg49[%c0_37], %123, %mask_42 : memref<8xi32, 1>, index, <i32>
                }
                ascvec.mem_bar vst_vld
              } {cannir.vf_mode = "simd"}
              ascvec.release_buf v, %c7_i64, 0 {cannir.buffer.generated_sync}
              ascvec.sync_notify {dst_pipe = 1 : i32, event_id = 0 : i32, src_pipe = 2 : i32}
            } else {
              %103 = arith.trunci %71 : i64 to i32
              %104 = arith.trunci %72 : i64 to i32
              %105 = arith.muli %40, %c8192_i64 : i64
              ascvec.get_buf mte2, %c7_i64, 0 {cannir.buffer.generated_sync}
              "ascvec.copy_gm2ub_plan"(%alloc_26, %alloc_0, %c0_i32, %c0_i32, %c0_i64, %105, %c1_i64, %c8192_i64, %c0_i64, %c8192_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = i32, ub_base = 212960 : i64, ub_capacity = 8448 : i64}> : (memref<2112xi32, 1>, memref<1xi32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
              ascvec.release_buf mte2, %c7_i64, 0 {cannir.buffer.generated_sync}
              %106 = arith.remsi %103, %c64_i32 : i32
              %107 = arith.cmpi ne, %106, %c0_i32 : i32
              %108 = arith.cmpi slt, %106, %c0_i32 : i32
              %109 = arith.andi %107, %108 : i1
              %110 = arith.divsi %103, %c64_i32 : i32
              %111 = arith.extui %109 : i1 to i32
              %112 = arith.subi %110, %111 : i32
              %113 = arith.addi %104, %c63_i32 : i32
              %114 = arith.remsi %113, %c64_i32 : i32
              %115 = arith.cmpi ne, %114, %c0_i32 : i32
              %116 = arith.cmpi slt, %114, %c0_i32 : i32
              %117 = arith.andi %115, %116 : i1
              %118 = arith.divsi %113, %c64_i32 : i32
              %119 = arith.extui %117 : i1 to i32
              %120 = arith.subi %118, %119 : i32
              ascvec.get_buf v, %c7_i64, 0 {cannir.buffer.generated_sync}
              ascvec.vf(%alloc_26, %59, %42, %103, %104, %alloc_18, %112, %120 : memref<2112xi32, 1>, i32, i32, i32, i32, memref<2048xi32, 1>, i32, i32) {
              ^bb0(%arg41: memref<2112xi32, 1>, %arg42: i32, %arg43: i32, %arg44: i32, %arg45: i32, %arg46: memref<2048xi32, 1>, %arg47: i32, %arg48: i32):
                %c1_i32_36 = arith.constant 1 : i32
                %c0_i32_37 = arith.constant 0 : i32
                %c8_i32 = arith.constant 8 : i32
                %c64_i32_38 = arith.constant 64 : i32
                %c32_i32_39 = arith.constant 32 : i32
                %mask, %remaining_out = ascvec.update_mask %c32_i32_39 : !ascvec.mask
                scf.for %arg49 = %arg47 to %arg48 step %c1_i32_36  : i32 {
                  %121 = arith.muli %arg49, %c64_i32_38 : i32
                  %122 = arith.addi %121, %c32_i32_39 : i32
                  %123 = arith.index_cast %122 : i32 to index
                  %124 = ascvec.loadalign %arg41[%123] : memref<2112xi32, 1>, index -> <i32>
                  %125 = ascvec.duplicate_scalar %arg42, %mask : i32 -> <i32>
                  %126 = ascvec.mul_scalar %124, %c8_i32, %mask : <i32>, i32
                  %127 = ascvec.sub %125, %126, %mask : <i32>
                  %128 = ascvec.max_scalar %127, %c0_i32_37, %mask : <i32>, i32
                  %129 = ascvec.min_scalar %128, %c8_i32, %mask : <i32>, i32
                  %130 = ascvec.duplicate_scalar %c0_i32_37, %mask : i32 -> <i32>
                  %131 = ascvec.arange %122 : i32 -> <i32>
                  %132 = ascvec.cmp_lt_scalar %131, %arg43, %mask : <i32>, i32 -> !ascvec.mask
                  %133 = ascvec.cmp_ge_scalar %131, %arg44, %mask : <i32>, i32 -> !ascvec.mask
                  %134 = ascvec.cmp_lt_scalar %131, %arg45, %mask : <i32>, i32 -> !ascvec.mask
                  %135 = ascvec.mask_and %133, %134, %mask
                  %136 = ascvec.mask_and %132, %135, %mask
                  %137 = ascvec.select %129, %130, %136 : <i32>
                  ascvec.storealign %arg46[%123], %137, %mask : memref<2048xi32, 1>, index, <i32>
                } {cannir.loop_unroll = 1 : i64}
                ascvec.mem_bar vst_vld
              } {cannir.vf_mode = "simd"}
              ascvec.release_buf v, %c7_i64, 0 {cannir.buffer.generated_sync}
            }
            %78 = arith.muli %40, %c128_i64 : i64
            %79 = cannir.extract_layout %arg5 : !cannir.tensor<<gm, f32>, <<(-1, 32)>, <(32, 1)>>, <<(-1, 32)>, <(32, 1)>>, identity>
            %80 = cannir.extract_shape %79 : <<(-1, 32)>, <(32, 1)>> -> <(-1, 32)>
            %81 = cannir.extract_dim %80[0] : !cannir.shape<(-1, 32)>
            %82 = arith.minsi %40, %81 : i64
            %83 = arith.subi %81, %82 : i64
            %84 = arith.minsi %83, %c1_i64 : i64
            %85 = arith.cmpi sle, %40, %81 : i64
            %86 = arith.select %85, %84, %c0_i64 : i64
            %87 = arith.muli %86, %c128_i64 : i64
            %88 = arith.cmpi eq, %87, %c0_i64 : i64
            %89 = arith.cmpi ne, %87, %c0_i64 : i64
            %90 = arith.extui %89 : i1 to i64
            %91 = arith.select %88, %c4_i64, %87 : i64
            ascvec.get_buf mte2, %c2_i64, 0 {cannir.buffer.generated_sync}
            "ascvec.copy_gm2ub_plan"(%alloc_15, %alloc, %c0_i32, %c0_i32, %c0_i64, %78, %90, %91, %c0_i64, %91, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 71680 : i64, ub_capacity = 128 : i64}> : (memref<32xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
            ascvec.release_buf mte2, %c2_i64, 0 {cannir.buffer.generated_sync}
            ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
            ascvec.vf(%alloc_15, %alloc_16 : memref<32xf32, 1>, memref<32xbf16, 1>) {
            ^bb0(%arg41: memref<32xf32, 1>, %arg42: memref<32xbf16, 1>):
              %c0_36 = arith.constant 0 : index
              %c32_i32_37 = arith.constant 32 : i32
              %c0_38 = arith.constant 0 : index
              %c1_39 = arith.constant 1 : index
              scf.for %arg43 = %c0_38 to %c1_39 step %c1_39 {
                %mask, %remaining_out = ascvec.update_mask %c32_i32_37 : !ascvec.mask
                %103 = ascvec.loadalign %arg41[%c0_36] : memref<32xf32, 1>, index -> <f32>
                %104 = ascvec.cast %103, %mask {reg_layout = 0 : i32} : <f32> -> <bf16>
                %105 = ascvec.reinterpret %104 : <bf16> -> <ui32>
                %106 = ascvec.pack %105 src_elem_bits 32 dst_elem_bits 16 {part = "low"} : <ui32> -> <ui16>
                %107 = ascvec.reinterpret %106 : <ui16> -> <bf16>
                %mask_40, %remaining_out_41 = ascvec.update_mask %c32_i32_37 elem_bits 16 : !ascvec.mask
                ascvec.storealign %arg42[%c0_36], %107, %mask_40 : memref<32xbf16, 1>, index, <bf16>
              }
              ascvec.mem_bar vst_vld
            } {cannir.vf_mode = "simd"}
            ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
            %92 = arith.cmpi eq, %arg39, %c0_i64 : i64
            scf.if %92 {
              ascvec.sync_notify {dst_pipe = 5 : i32, event_id = 2 : i32, src_pipe = 6 : i32}
              ascvec.sync_notify {dst_pipe = 5 : i32, event_id = 3 : i32, src_pipe = 6 : i32}
            }
            ascvec.sync_intra_arrive v, %c4_i64
            ascvec.sync_intra_arrive v, %c5_i64
            %93 = arith.addi %56, %c1_i64 : i64
            %94 = scf.for %arg41 = %54 to %93 step %c1_i64 iter_args(%arg42 = %arg40) -> (i64)  : i64 {
              %103 = arith.cmpi slt, %arg41, %56 : i64
              %104 = arith.addi %54, %arg39 : i64
              %105 = arith.cmpi sge, %arg41, %104 : i64
              %106 = arith.select %103, %105, %false : i1
              scf.if %106 {
                %109 = arith.addi %54, %c4_i64 : i64
                %110 = arith.cmpi sge, %arg41, %109 : i64
                scf.if %110 {
                  %126 = arith.remsi %arg41, %c4_i64 : i64
                  %127 = arith.cmpi ne, %126, %c0_i64 : i64
                  %128 = arith.cmpi slt, %126, %c0_i64 : i64
                  %129 = arith.andi %127, %128 : i1
                  %130 = arith.select %129, %c4_i64, %c0_i64 : i64
                  %131 = arith.addi %126, %130 overflow<nsw> : i64
                  %132 = arith.addi %131, %c10_i64 overflow<nsw> : i64
                  ascvec.sync_intra_wait mte2, %132
                }
                %111 = arith.remsi %arg41, %c2_i64 : i64
                %112 = arith.cmpi ne, %111, %c0_i64 : i64
                %113 = arith.cmpi slt, %111, %c0_i64 : i64
                %114 = arith.andi %112, %113 : i1
                %115 = arith.select %114, %c2_i64, %c0_i64 : i64
                %116 = arith.addi %111, %115 overflow<nsw> : i64
                %117 = arith.cmpi eq, %116, %c0_i64 : i64
                scf.if %117 {
                  %126 = arith.muli %8, %c4_i64 : i64
                  %127 = arith.remsi %arg41, %c4_i64 : i64
                  %128 = arith.cmpi ne, %127, %c0_i64 : i64
                  %129 = arith.cmpi slt, %127, %c0_i64 : i64
                  %130 = arith.andi %128, %129 : i1
                  %131 = arith.select %130, %c4_i64, %c0_i64 : i64
                  %132 = arith.addi %127, %131 overflow<nsw> : i64
                  %133 = arith.addi %126, %132 : i64
                  %134 = arith.muli %133, %c512_i64 : i64
                  %135 = arith.muli %7, %c256_i64 : i64
                  %136 = arith.addi %134, %135 : i64
                  %137 = arith.muli %arg41, %c64_i64 : i64
                  %138 = arith.muli %7, %c32_i64 : i64
                  %139 = arith.addi %137, %138 : i64
                  %140 = memref.load %alloc_4[%25] : memref<1xi32>
                  %141 = arith.extsi %140 : i32 to i64
                  %142 = memref.load %alloc_4[%29] : memref<1xi32>
                  %143 = arith.extsi %142 : i32 to i64
                  %144 = arith.subi %143, %141 : i64
                  ascvec.sync_wait {dst_pipe = 5 : i32, event_id = 2 : i32, src_pipe = 6 : i32}
                  scf.for %arg43 = %c0_i64 to %c32_i64 step %c1_i64  : i64 {
                    %159 = arith.addi %139, %arg43 : i64
                    %160 = arith.cmpi slt, %159, %43 : i64
                    %161 = arith.subi %43, %c1_i64 : i64
                    %162 = arith.select %160, %159, %161 : i64
                    %163 = arith.index_cast %162 : i64 to index
                    %164 = arith.muli %41, %c2048 : index
                    %165 = arith.addi %164, %163 : index
                    %166 = memref.load %alloc_0[%165] : memref<1xi32>
                    %167 = arith.extsi %166 : i32 to i64
                    %168 = arith.muli %167, %c8_i64 : i64
                    %169 = arith.subi %144, %168 : i64
                    %170 = arith.minsi %169, %c8_i64 : i64
                    %171 = arith.cmpi eq, %170, %c8_i64 : i64
                    scf.if %171 {
                      %172 = arith.addi %141, %168 : i64
                      %173 = cannir.extract_layout %arg1 : !cannir.tensor<<gm, ui8>, <<(-1, 1, 64)>, <(-1, 64, 1)>>, <<(-1, 1, 64)>, <(-1, 64, 1)>>, identity>
                      %174 = cannir.extract_stride %173 : <<(-1, 1, 64)>, <(-1, 64, 1)>> -> <(-1, 64, 1)>
                      %175 = cannir.extract_dim %174[0] : !cannir.stride<(-1, 64, 1)>
                      %176 = arith.muli %172, %175 : i64
                      %177 = arith.addi %arg2, %176 : i64
                      %178 = ascscalar.bind_gm_ptr %177 : memref<512xui8>
                      %179 = cannir.extract_layout %arg3 : !cannir.tensor<<gm, ui8>, <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>>, <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>>, identity>
                      %180 = cannir.extract_stride %179 : <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>> -> <(-1, 4, 2, 1)>
                      %181 = cannir.extract_dim %180[0] : !cannir.stride<(-1, 4, 2, 1)>
                      %182 = arith.muli %172, %181 : i64
                      %183 = arith.addi %arg4, %182 : i64
                      %184 = ascscalar.bind_gm_ptr %183 : memref<32xui8>
                      %185 = arith.muli %arg43, %c544_i64 overflow<nsw> : i64
                      "ascvec.copy_gm2ub_plan"(%alloc_12, %178, %c0_i32, %c0_i32, %185, %c0_i64, %c8_i64, %c64_i64, %175, %c64_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = ui8, ub_base = 32768 : i64, ub_capacity = 17408 : i64}> : (memref<17408xui8, 1>, memref<512xui8>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                      %186 = arith.addi %185, %c512_i64 overflow<nsw> : i64
                      "ascvec.copy_gm2ub_plan"(%alloc_12, %184, %c0_i32, %c0_i32, %186, %c0_i64, %c8_i64, %c4_i64, %181, %c4_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = ui8, ub_base = 32768 : i64, ub_capacity = 17408 : i64}> : (memref<17408xui8, 1>, memref<32xui8>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                    } else {
                      %172 = arith.cmpi eq, %170, %c1_i64 : i64
                      scf.if %172 {
                        %179 = arith.addi %141, %168 : i64
                        %180 = cannir.extract_layout %arg1 : !cannir.tensor<<gm, ui8>, <<(-1, 1, 64)>, <(-1, 64, 1)>>, <<(-1, 1, 64)>, <(-1, 64, 1)>>, identity>
                        %181 = cannir.extract_stride %180 : <<(-1, 1, 64)>, <(-1, 64, 1)>> -> <(-1, 64, 1)>
                        %182 = cannir.extract_dim %181[0] : !cannir.stride<(-1, 64, 1)>
                        %183 = arith.muli %179, %182 : i64
                        %184 = arith.addi %arg2, %183 : i64
                        %185 = ascscalar.bind_gm_ptr %184 : memref<64xui8>
                        %186 = cannir.extract_layout %arg3 : !cannir.tensor<<gm, ui8>, <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>>, <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>>, identity>
                        %187 = cannir.extract_stride %186 : <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>> -> <(-1, 4, 2, 1)>
                        %188 = cannir.extract_dim %187[0] : !cannir.stride<(-1, 4, 2, 1)>
                        %189 = arith.muli %179, %188 : i64
                        %190 = arith.addi %arg4, %189 : i64
                        %191 = ascscalar.bind_gm_ptr %190 : memref<4xui8>
                        %192 = arith.muli %arg43, %c4352_i64 overflow<nsw> : i64
                        %193 = ascscalar.memref_address %alloc_12 : memref<17408xui8, 1>
                        %194 = arith.divui %192, %c8_i64 : i64
                        %195 = arith.addi %193, %194 : i64
                        %196 = ascscalar.bind_local_ptr %195 : memref<1xui8, 1>
                        ascvec.copy_gm2ub_nddma %196[%c0], %185[%c0], %c64_i32, %c1_i32, %c1_i32, %c1_i32, %c1_i32, %c1_i64, %182, %c0_i64, %c0_i64, %c0_i64, %c1_i64, %c64_i64, %c0_i64, %c0_i64, %c0_i64 l2_cache_ctl %c0_i32 left_padding(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) right_padding(%c0_i32, %c7_i32, %c0_i32, %c0_i32, %c0_i32) pad_value %c0_i32 : i32 {nearest = true} : memref<1xui8, 1>, memref<64xui8>
                        %197 = arith.muli %arg43, %c544_i64 overflow<nsw> : i64
                        %198 = arith.addi %197, %c512_i64 overflow<nsw> : i64
                        %199 = arith.muli %198, %c8_i64 overflow<nsw> : i64
                        %200 = arith.divui %199, %c8_i64 : i64
                        %201 = arith.addi %193, %200 : i64
                        %202 = ascscalar.bind_local_ptr %201 : memref<1xui8, 1>
                        ascvec.copy_gm2ub_nddma %202[%c0], %191[%c0], %c4_i32, %c1_i32, %c1_i32, %c1_i32, %c1_i32, %c1_i64, %188, %c0_i64, %c0_i64, %c0_i64, %c1_i64, %c4_i64, %c0_i64, %c0_i64, %c0_i64 l2_cache_ctl %c0_i32 left_padding(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) right_padding(%c0_i32, %c7_i32, %c0_i32, %c0_i32, %c0_i32) pad_value %c0_i32 : i32 {nearest = true} : memref<1xui8, 1>, memref<4xui8>
                      }
                      %173 = arith.cmpi eq, %170, %c2_i64 : i64
                      scf.if %173 {
                        %179 = arith.addi %141, %168 : i64
                        %180 = cannir.extract_layout %arg1 : !cannir.tensor<<gm, ui8>, <<(-1, 1, 64)>, <(-1, 64, 1)>>, <<(-1, 1, 64)>, <(-1, 64, 1)>>, identity>
                        %181 = cannir.extract_stride %180 : <<(-1, 1, 64)>, <(-1, 64, 1)>> -> <(-1, 64, 1)>
                        %182 = cannir.extract_dim %181[0] : !cannir.stride<(-1, 64, 1)>
                        %183 = arith.muli %179, %182 : i64
                        %184 = arith.addi %arg2, %183 : i64
                        %185 = ascscalar.bind_gm_ptr %184 : memref<128xui8>
                        %186 = cannir.extract_layout %arg3 : !cannir.tensor<<gm, ui8>, <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>>, <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>>, identity>
                        %187 = cannir.extract_stride %186 : <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>> -> <(-1, 4, 2, 1)>
                        %188 = cannir.extract_dim %187[0] : !cannir.stride<(-1, 4, 2, 1)>
                        %189 = arith.muli %179, %188 : i64
                        %190 = arith.addi %arg4, %189 : i64
                        %191 = ascscalar.bind_gm_ptr %190 : memref<8xui8>
                        %192 = arith.muli %arg43, %c4352_i64 overflow<nsw> : i64
                        %193 = ascscalar.memref_address %alloc_12 : memref<17408xui8, 1>
                        %194 = arith.divui %192, %c8_i64 : i64
                        %195 = arith.addi %193, %194 : i64
                        %196 = ascscalar.bind_local_ptr %195 : memref<1xui8, 1>
                        ascvec.copy_gm2ub_nddma %196[%c0], %185[%c0], %c64_i32, %c2_i32, %c1_i32, %c1_i32, %c1_i32, %c1_i64, %182, %c0_i64, %c0_i64, %c0_i64, %c1_i64, %c64_i64, %c0_i64, %c0_i64, %c0_i64 l2_cache_ctl %c0_i32 left_padding(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) right_padding(%c0_i32, %c6_i32, %c0_i32, %c0_i32, %c0_i32) pad_value %c0_i32 : i32 {nearest = true} : memref<1xui8, 1>, memref<128xui8>
                        %197 = arith.muli %arg43, %c544_i64 overflow<nsw> : i64
                        %198 = arith.addi %197, %c512_i64 overflow<nsw> : i64
                        %199 = arith.muli %198, %c8_i64 overflow<nsw> : i64
                        %200 = arith.divui %199, %c8_i64 : i64
                        %201 = arith.addi %193, %200 : i64
                        %202 = ascscalar.bind_local_ptr %201 : memref<1xui8, 1>
                        ascvec.copy_gm2ub_nddma %202[%c0], %191[%c0], %c4_i32, %c2_i32, %c1_i32, %c1_i32, %c1_i32, %c1_i64, %188, %c0_i64, %c0_i64, %c0_i64, %c1_i64, %c4_i64, %c0_i64, %c0_i64, %c0_i64 l2_cache_ctl %c0_i32 left_padding(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) right_padding(%c0_i32, %c6_i32, %c0_i32, %c0_i32, %c0_i32) pad_value %c0_i32 : i32 {nearest = true} : memref<1xui8, 1>, memref<8xui8>
                      }
                      %174 = arith.cmpi eq, %170, %c3_i64 : i64
                      scf.if %174 {
                        %179 = arith.addi %141, %168 : i64
                        %180 = cannir.extract_layout %arg1 : !cannir.tensor<<gm, ui8>, <<(-1, 1, 64)>, <(-1, 64, 1)>>, <<(-1, 1, 64)>, <(-1, 64, 1)>>, identity>
                        %181 = cannir.extract_stride %180 : <<(-1, 1, 64)>, <(-1, 64, 1)>> -> <(-1, 64, 1)>
                        %182 = cannir.extract_dim %181[0] : !cannir.stride<(-1, 64, 1)>
                        %183 = arith.muli %179, %182 : i64
                        %184 = arith.addi %arg2, %183 : i64
                        %185 = ascscalar.bind_gm_ptr %184 : memref<192xui8>
                        %186 = cannir.extract_layout %arg3 : !cannir.tensor<<gm, ui8>, <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>>, <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>>, identity>
                        %187 = cannir.extract_stride %186 : <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>> -> <(-1, 4, 2, 1)>
                        %188 = cannir.extract_dim %187[0] : !cannir.stride<(-1, 4, 2, 1)>
                        %189 = arith.muli %179, %188 : i64
                        %190 = arith.addi %arg4, %189 : i64
                        %191 = ascscalar.bind_gm_ptr %190 : memref<12xui8>
                        %192 = arith.muli %arg43, %c4352_i64 overflow<nsw> : i64
                        %193 = ascscalar.memref_address %alloc_12 : memref<17408xui8, 1>
                        %194 = arith.divui %192, %c8_i64 : i64
                        %195 = arith.addi %193, %194 : i64
                        %196 = ascscalar.bind_local_ptr %195 : memref<1xui8, 1>
                        ascvec.copy_gm2ub_nddma %196[%c0], %185[%c0], %c64_i32, %c3_i32, %c1_i32, %c1_i32, %c1_i32, %c1_i64, %182, %c0_i64, %c0_i64, %c0_i64, %c1_i64, %c64_i64, %c0_i64, %c0_i64, %c0_i64 l2_cache_ctl %c0_i32 left_padding(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) right_padding(%c0_i32, %c5_i32, %c0_i32, %c0_i32, %c0_i32) pad_value %c0_i32 : i32 {nearest = true} : memref<1xui8, 1>, memref<192xui8>
                        %197 = arith.muli %arg43, %c544_i64 overflow<nsw> : i64
                        %198 = arith.addi %197, %c512_i64 overflow<nsw> : i64
                        %199 = arith.muli %198, %c8_i64 overflow<nsw> : i64
                        %200 = arith.divui %199, %c8_i64 : i64
                        %201 = arith.addi %193, %200 : i64
                        %202 = ascscalar.bind_local_ptr %201 : memref<1xui8, 1>
                        ascvec.copy_gm2ub_nddma %202[%c0], %191[%c0], %c4_i32, %c3_i32, %c1_i32, %c1_i32, %c1_i32, %c1_i64, %188, %c0_i64, %c0_i64, %c0_i64, %c1_i64, %c4_i64, %c0_i64, %c0_i64, %c0_i64 l2_cache_ctl %c0_i32 left_padding(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) right_padding(%c0_i32, %c5_i32, %c0_i32, %c0_i32, %c0_i32) pad_value %c0_i32 : i32 {nearest = true} : memref<1xui8, 1>, memref<12xui8>
                      }
                      %175 = arith.cmpi eq, %170, %c4_i64 : i64
                      scf.if %175 {
                        %179 = arith.addi %141, %168 : i64
                        %180 = cannir.extract_layout %arg1 : !cannir.tensor<<gm, ui8>, <<(-1, 1, 64)>, <(-1, 64, 1)>>, <<(-1, 1, 64)>, <(-1, 64, 1)>>, identity>
                        %181 = cannir.extract_stride %180 : <<(-1, 1, 64)>, <(-1, 64, 1)>> -> <(-1, 64, 1)>
                        %182 = cannir.extract_dim %181[0] : !cannir.stride<(-1, 64, 1)>
                        %183 = arith.muli %179, %182 : i64
                        %184 = arith.addi %arg2, %183 : i64
                        %185 = ascscalar.bind_gm_ptr %184 : memref<256xui8>
                        %186 = cannir.extract_layout %arg3 : !cannir.tensor<<gm, ui8>, <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>>, <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>>, identity>
                        %187 = cannir.extract_stride %186 : <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>> -> <(-1, 4, 2, 1)>
                        %188 = cannir.extract_dim %187[0] : !cannir.stride<(-1, 4, 2, 1)>
                        %189 = arith.muli %179, %188 : i64
                        %190 = arith.addi %arg4, %189 : i64
                        %191 = ascscalar.bind_gm_ptr %190 : memref<16xui8>
                        %192 = arith.muli %arg43, %c4352_i64 overflow<nsw> : i64
                        %193 = ascscalar.memref_address %alloc_12 : memref<17408xui8, 1>
                        %194 = arith.divui %192, %c8_i64 : i64
                        %195 = arith.addi %193, %194 : i64
                        %196 = ascscalar.bind_local_ptr %195 : memref<1xui8, 1>
                        ascvec.copy_gm2ub_nddma %196[%c0], %185[%c0], %c64_i32, %c4_i32, %c1_i32, %c1_i32, %c1_i32, %c1_i64, %182, %c0_i64, %c0_i64, %c0_i64, %c1_i64, %c64_i64, %c0_i64, %c0_i64, %c0_i64 l2_cache_ctl %c0_i32 left_padding(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) right_padding(%c0_i32, %c4_i32, %c0_i32, %c0_i32, %c0_i32) pad_value %c0_i32 : i32 {nearest = true} : memref<1xui8, 1>, memref<256xui8>
                        %197 = arith.muli %arg43, %c544_i64 overflow<nsw> : i64
                        %198 = arith.addi %197, %c512_i64 overflow<nsw> : i64
                        %199 = arith.muli %198, %c8_i64 overflow<nsw> : i64
                        %200 = arith.divui %199, %c8_i64 : i64
                        %201 = arith.addi %193, %200 : i64
                        %202 = ascscalar.bind_local_ptr %201 : memref<1xui8, 1>
                        ascvec.copy_gm2ub_nddma %202[%c0], %191[%c0], %c4_i32, %c4_i32, %c1_i32, %c1_i32, %c1_i32, %c1_i64, %188, %c0_i64, %c0_i64, %c0_i64, %c1_i64, %c4_i64, %c0_i64, %c0_i64, %c0_i64 l2_cache_ctl %c0_i32 left_padding(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) right_padding(%c0_i32, %c4_i32, %c0_i32, %c0_i32, %c0_i32) pad_value %c0_i32 : i32 {nearest = true} : memref<1xui8, 1>, memref<16xui8>
                      }
                      %176 = arith.cmpi eq, %170, %c5_i64 : i64
                      scf.if %176 {
                        %179 = arith.addi %141, %168 : i64
                        %180 = cannir.extract_layout %arg1 : !cannir.tensor<<gm, ui8>, <<(-1, 1, 64)>, <(-1, 64, 1)>>, <<(-1, 1, 64)>, <(-1, 64, 1)>>, identity>
                        %181 = cannir.extract_stride %180 : <<(-1, 1, 64)>, <(-1, 64, 1)>> -> <(-1, 64, 1)>
                        %182 = cannir.extract_dim %181[0] : !cannir.stride<(-1, 64, 1)>
                        %183 = arith.muli %179, %182 : i64
                        %184 = arith.addi %arg2, %183 : i64
                        %185 = ascscalar.bind_gm_ptr %184 : memref<320xui8>
                        %186 = cannir.extract_layout %arg3 : !cannir.tensor<<gm, ui8>, <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>>, <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>>, identity>
                        %187 = cannir.extract_stride %186 : <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>> -> <(-1, 4, 2, 1)>
                        %188 = cannir.extract_dim %187[0] : !cannir.stride<(-1, 4, 2, 1)>
                        %189 = arith.muli %179, %188 : i64
                        %190 = arith.addi %arg4, %189 : i64
                        %191 = ascscalar.bind_gm_ptr %190 : memref<20xui8>
                        %192 = arith.muli %arg43, %c4352_i64 overflow<nsw> : i64
                        %193 = ascscalar.memref_address %alloc_12 : memref<17408xui8, 1>
                        %194 = arith.divui %192, %c8_i64 : i64
                        %195 = arith.addi %193, %194 : i64
                        %196 = ascscalar.bind_local_ptr %195 : memref<1xui8, 1>
                        ascvec.copy_gm2ub_nddma %196[%c0], %185[%c0], %c64_i32, %c5_i32, %c1_i32, %c1_i32, %c1_i32, %c1_i64, %182, %c0_i64, %c0_i64, %c0_i64, %c1_i64, %c64_i64, %c0_i64, %c0_i64, %c0_i64 l2_cache_ctl %c0_i32 left_padding(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) right_padding(%c0_i32, %c3_i32, %c0_i32, %c0_i32, %c0_i32) pad_value %c0_i32 : i32 {nearest = true} : memref<1xui8, 1>, memref<320xui8>
                        %197 = arith.muli %arg43, %c544_i64 overflow<nsw> : i64
                        %198 = arith.addi %197, %c512_i64 overflow<nsw> : i64
                        %199 = arith.muli %198, %c8_i64 overflow<nsw> : i64
                        %200 = arith.divui %199, %c8_i64 : i64
                        %201 = arith.addi %193, %200 : i64
                        %202 = ascscalar.bind_local_ptr %201 : memref<1xui8, 1>
                        ascvec.copy_gm2ub_nddma %202[%c0], %191[%c0], %c4_i32, %c5_i32, %c1_i32, %c1_i32, %c1_i32, %c1_i64, %188, %c0_i64, %c0_i64, %c0_i64, %c1_i64, %c4_i64, %c0_i64, %c0_i64, %c0_i64 l2_cache_ctl %c0_i32 left_padding(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) right_padding(%c0_i32, %c3_i32, %c0_i32, %c0_i32, %c0_i32) pad_value %c0_i32 : i32 {nearest = true} : memref<1xui8, 1>, memref<20xui8>
                      }
                      %177 = arith.cmpi eq, %170, %c6_i64 : i64
                      scf.if %177 {
                        %179 = arith.addi %141, %168 : i64
                        %180 = cannir.extract_layout %arg1 : !cannir.tensor<<gm, ui8>, <<(-1, 1, 64)>, <(-1, 64, 1)>>, <<(-1, 1, 64)>, <(-1, 64, 1)>>, identity>
                        %181 = cannir.extract_stride %180 : <<(-1, 1, 64)>, <(-1, 64, 1)>> -> <(-1, 64, 1)>
                        %182 = cannir.extract_dim %181[0] : !cannir.stride<(-1, 64, 1)>
                        %183 = arith.muli %179, %182 : i64
                        %184 = arith.addi %arg2, %183 : i64
                        %185 = ascscalar.bind_gm_ptr %184 : memref<384xui8>
                        %186 = cannir.extract_layout %arg3 : !cannir.tensor<<gm, ui8>, <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>>, <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>>, identity>
                        %187 = cannir.extract_stride %186 : <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>> -> <(-1, 4, 2, 1)>
                        %188 = cannir.extract_dim %187[0] : !cannir.stride<(-1, 4, 2, 1)>
                        %189 = arith.muli %179, %188 : i64
                        %190 = arith.addi %arg4, %189 : i64
                        %191 = ascscalar.bind_gm_ptr %190 : memref<24xui8>
                        %192 = arith.muli %arg43, %c4352_i64 overflow<nsw> : i64
                        %193 = ascscalar.memref_address %alloc_12 : memref<17408xui8, 1>
                        %194 = arith.divui %192, %c8_i64 : i64
                        %195 = arith.addi %193, %194 : i64
                        %196 = ascscalar.bind_local_ptr %195 : memref<1xui8, 1>
                        ascvec.copy_gm2ub_nddma %196[%c0], %185[%c0], %c64_i32, %c6_i32, %c1_i32, %c1_i32, %c1_i32, %c1_i64, %182, %c0_i64, %c0_i64, %c0_i64, %c1_i64, %c64_i64, %c0_i64, %c0_i64, %c0_i64 l2_cache_ctl %c0_i32 left_padding(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) right_padding(%c0_i32, %c2_i32, %c0_i32, %c0_i32, %c0_i32) pad_value %c0_i32 : i32 {nearest = true} : memref<1xui8, 1>, memref<384xui8>
                        %197 = arith.muli %arg43, %c544_i64 overflow<nsw> : i64
                        %198 = arith.addi %197, %c512_i64 overflow<nsw> : i64
                        %199 = arith.muli %198, %c8_i64 overflow<nsw> : i64
                        %200 = arith.divui %199, %c8_i64 : i64
                        %201 = arith.addi %193, %200 : i64
                        %202 = ascscalar.bind_local_ptr %201 : memref<1xui8, 1>
                        ascvec.copy_gm2ub_nddma %202[%c0], %191[%c0], %c4_i32, %c6_i32, %c1_i32, %c1_i32, %c1_i32, %c1_i64, %188, %c0_i64, %c0_i64, %c0_i64, %c1_i64, %c4_i64, %c0_i64, %c0_i64, %c0_i64 l2_cache_ctl %c0_i32 left_padding(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) right_padding(%c0_i32, %c2_i32, %c0_i32, %c0_i32, %c0_i32) pad_value %c0_i32 : i32 {nearest = true} : memref<1xui8, 1>, memref<24xui8>
                      }
                      %178 = arith.cmpi eq, %170, %c7_i64 : i64
                      scf.if %178 {
                        %179 = arith.addi %141, %168 : i64
                        %180 = cannir.extract_layout %arg1 : !cannir.tensor<<gm, ui8>, <<(-1, 1, 64)>, <(-1, 64, 1)>>, <<(-1, 1, 64)>, <(-1, 64, 1)>>, identity>
                        %181 = cannir.extract_stride %180 : <<(-1, 1, 64)>, <(-1, 64, 1)>> -> <(-1, 64, 1)>
                        %182 = cannir.extract_dim %181[0] : !cannir.stride<(-1, 64, 1)>
                        %183 = arith.muli %179, %182 : i64
                        %184 = arith.addi %arg2, %183 : i64
                        %185 = ascscalar.bind_gm_ptr %184 : memref<448xui8>
                        %186 = cannir.extract_layout %arg3 : !cannir.tensor<<gm, ui8>, <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>>, <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>>, identity>
                        %187 = cannir.extract_stride %186 : <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>> -> <(-1, 4, 2, 1)>
                        %188 = cannir.extract_dim %187[0] : !cannir.stride<(-1, 4, 2, 1)>
                        %189 = arith.muli %179, %188 : i64
                        %190 = arith.addi %arg4, %189 : i64
                        %191 = ascscalar.bind_gm_ptr %190 : memref<28xui8>
                        %192 = arith.muli %arg43, %c4352_i64 overflow<nsw> : i64
                        %193 = ascscalar.memref_address %alloc_12 : memref<17408xui8, 1>
                        %194 = arith.divui %192, %c8_i64 : i64
                        %195 = arith.addi %193, %194 : i64
                        %196 = ascscalar.bind_local_ptr %195 : memref<1xui8, 1>
                        ascvec.copy_gm2ub_nddma %196[%c0], %185[%c0], %c64_i32, %c7_i32, %c1_i32, %c1_i32, %c1_i32, %c1_i64, %182, %c0_i64, %c0_i64, %c0_i64, %c1_i64, %c64_i64, %c0_i64, %c0_i64, %c0_i64 l2_cache_ctl %c0_i32 left_padding(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) right_padding(%c0_i32, %c1_i32, %c0_i32, %c0_i32, %c0_i32) pad_value %c0_i32 : i32 {nearest = true} : memref<1xui8, 1>, memref<448xui8>
                        %197 = arith.muli %arg43, %c544_i64 overflow<nsw> : i64
                        %198 = arith.addi %197, %c512_i64 overflow<nsw> : i64
                        %199 = arith.muli %198, %c8_i64 overflow<nsw> : i64
                        %200 = arith.divui %199, %c8_i64 : i64
                        %201 = arith.addi %193, %200 : i64
                        %202 = ascscalar.bind_local_ptr %201 : memref<1xui8, 1>
                        ascvec.copy_gm2ub_nddma %202[%c0], %191[%c0], %c4_i32, %c7_i32, %c1_i32, %c1_i32, %c1_i32, %c1_i64, %188, %c0_i64, %c0_i64, %c0_i64, %c1_i64, %c4_i64, %c0_i64, %c0_i64, %c0_i64 l2_cache_ctl %c0_i32 left_padding(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) right_padding(%c0_i32, %c1_i32, %c0_i32, %c0_i32, %c0_i32) pad_value %c0_i32 : i32 {nearest = true} : memref<1xui8, 1>, memref<28xui8>
                      }
                    }
                  }
                  ascvec.sync_notify {dst_pipe = 6 : i32, event_id = 2 : i32, src_pipe = 5 : i32}
                  ascvec.sync_wait {dst_pipe = 6 : i32, event_id = 2 : i32, src_pipe = 5 : i32}
                  %145 = arith.muli %136, %c512_i64 : i64
                  %146 = ascscalar.memref_address %alloc_7 : memref<4718592xui8>
                  %147 = arith.divui %145, %c8_i64 : i64
                  %148 = arith.addi %146, %147 : i64
                  %149 = ascscalar.bind_gm_ptr %148 : memref<1xui8>
                  %150 = ascscalar.bitcast_memref %alloc_12 byte_offset = %c0_i64 : memref<17408xui8, 1> -> memref<?xui8, 1>
                  ascvec.copy_ub2gm %149[%c0], %150[%c0], %c32_i32, %c512_i32, %c512_i64, %c544_i32 l2_cache_ctl %c0_i32 : memref<1xui8>, memref<?xui8, 1>
                  %151 = arith.muli %136, %c32_i64 : i64
                  %152 = ascscalar.memref_address %alloc_8 : memref<294912xui8>
                  %153 = arith.divui %151, %c8_i64 : i64
                  %154 = arith.addi %152, %153 : i64
                  %155 = ascscalar.bind_gm_ptr %154 : memref<1xui8>
                  %156 = ascscalar.memref_address %alloc_12 : memref<17408xui8, 1>
                  %157 = arith.addi %156, %c512_i64 : i64
                  %158 = ascscalar.bind_local_ptr %157 : memref<1xui8, 1>
                  ascvec.copy_ub2gm %155[%c0], %158[%c0], %c32_i32, %c32_i32, %c32_i64, %c544_i32 l2_cache_ctl %c0_i32 : memref<1xui8>, memref<1xui8, 1>
                  ascvec.sync_notify {dst_pipe = 5 : i32, event_id = 2 : i32, src_pipe = 6 : i32}
                }
                %118 = arith.cmpi eq, %116, %c1_i64 : i64
                scf.if %118 {
                  %126 = arith.muli %8, %c4_i64 : i64
                  %127 = arith.remsi %arg41, %c4_i64 : i64
                  %128 = arith.cmpi ne, %127, %c0_i64 : i64
                  %129 = arith.cmpi slt, %127, %c0_i64 : i64
                  %130 = arith.andi %128, %129 : i1
                  %131 = arith.select %130, %c4_i64, %c0_i64 : i64
                  %132 = arith.addi %127, %131 overflow<nsw> : i64
                  %133 = arith.addi %126, %132 : i64
                  %134 = arith.muli %133, %c512_i64 : i64
                  %135 = arith.muli %7, %c256_i64 : i64
                  %136 = arith.addi %134, %135 : i64
                  %137 = arith.muli %arg41, %c64_i64 : i64
                  %138 = arith.muli %7, %c32_i64 : i64
                  %139 = arith.addi %137, %138 : i64
                  %140 = memref.load %alloc_4[%25] : memref<1xi32>
                  %141 = arith.extsi %140 : i32 to i64
                  %142 = memref.load %alloc_4[%29] : memref<1xi32>
                  %143 = arith.extsi %142 : i32 to i64
                  %144 = arith.subi %143, %141 : i64
                  ascvec.sync_wait {dst_pipe = 5 : i32, event_id = 3 : i32, src_pipe = 6 : i32}
                  scf.for %arg43 = %c0_i64 to %c32_i64 step %c1_i64  : i64 {
                    %159 = arith.addi %139, %arg43 : i64
                    %160 = arith.cmpi slt, %159, %43 : i64
                    %161 = arith.subi %43, %c1_i64 : i64
                    %162 = arith.select %160, %159, %161 : i64
                    %163 = arith.index_cast %162 : i64 to index
                    %164 = arith.muli %41, %c2048 : index
                    %165 = arith.addi %164, %163 : index
                    %166 = memref.load %alloc_0[%165] : memref<1xi32>
                    %167 = arith.extsi %166 : i32 to i64
                    %168 = arith.muli %167, %c8_i64 : i64
                    %169 = arith.subi %144, %168 : i64
                    %170 = arith.minsi %169, %c8_i64 : i64
                    %171 = arith.cmpi eq, %170, %c8_i64 : i64
                    scf.if %171 {
                      %172 = arith.addi %141, %168 : i64
                      %173 = cannir.extract_layout %arg1 : !cannir.tensor<<gm, ui8>, <<(-1, 1, 64)>, <(-1, 64, 1)>>, <<(-1, 1, 64)>, <(-1, 64, 1)>>, identity>
                      %174 = cannir.extract_stride %173 : <<(-1, 1, 64)>, <(-1, 64, 1)>> -> <(-1, 64, 1)>
                      %175 = cannir.extract_dim %174[0] : !cannir.stride<(-1, 64, 1)>
                      %176 = arith.muli %172, %175 : i64
                      %177 = arith.addi %arg2, %176 : i64
                      %178 = ascscalar.bind_gm_ptr %177 : memref<512xui8>
                      %179 = cannir.extract_layout %arg3 : !cannir.tensor<<gm, ui8>, <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>>, <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>>, identity>
                      %180 = cannir.extract_stride %179 : <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>> -> <(-1, 4, 2, 1)>
                      %181 = cannir.extract_dim %180[0] : !cannir.stride<(-1, 4, 2, 1)>
                      %182 = arith.muli %172, %181 : i64
                      %183 = arith.addi %arg4, %182 : i64
                      %184 = ascscalar.bind_gm_ptr %183 : memref<32xui8>
                      %185 = arith.muli %arg43, %c544_i64 overflow<nsw> : i64
                      "ascvec.copy_gm2ub_plan"(%alloc_13, %178, %c0_i32, %c0_i32, %185, %c0_i64, %c8_i64, %c64_i64, %175, %c64_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = ui8, ub_base = 50176 : i64, ub_capacity = 17408 : i64}> : (memref<17408xui8, 1>, memref<512xui8>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                      %186 = arith.addi %185, %c512_i64 overflow<nsw> : i64
                      "ascvec.copy_gm2ub_plan"(%alloc_13, %184, %c0_i32, %c0_i32, %186, %c0_i64, %c8_i64, %c4_i64, %181, %c4_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = ui8, ub_base = 50176 : i64, ub_capacity = 17408 : i64}> : (memref<17408xui8, 1>, memref<32xui8>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                    } else {
                      %172 = arith.cmpi eq, %170, %c1_i64 : i64
                      scf.if %172 {
                        %179 = arith.addi %141, %168 : i64
                        %180 = cannir.extract_layout %arg1 : !cannir.tensor<<gm, ui8>, <<(-1, 1, 64)>, <(-1, 64, 1)>>, <<(-1, 1, 64)>, <(-1, 64, 1)>>, identity>
                        %181 = cannir.extract_stride %180 : <<(-1, 1, 64)>, <(-1, 64, 1)>> -> <(-1, 64, 1)>
                        %182 = cannir.extract_dim %181[0] : !cannir.stride<(-1, 64, 1)>
                        %183 = arith.muli %179, %182 : i64
                        %184 = arith.addi %arg2, %183 : i64
                        %185 = ascscalar.bind_gm_ptr %184 : memref<64xui8>
                        %186 = cannir.extract_layout %arg3 : !cannir.tensor<<gm, ui8>, <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>>, <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>>, identity>
                        %187 = cannir.extract_stride %186 : <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>> -> <(-1, 4, 2, 1)>
                        %188 = cannir.extract_dim %187[0] : !cannir.stride<(-1, 4, 2, 1)>
                        %189 = arith.muli %179, %188 : i64
                        %190 = arith.addi %arg4, %189 : i64
                        %191 = ascscalar.bind_gm_ptr %190 : memref<4xui8>
                        %192 = arith.muli %arg43, %c4352_i64 overflow<nsw> : i64
                        %193 = ascscalar.memref_address %alloc_13 : memref<17408xui8, 1>
                        %194 = arith.divui %192, %c8_i64 : i64
                        %195 = arith.addi %193, %194 : i64
                        %196 = ascscalar.bind_local_ptr %195 : memref<1xui8, 1>
                        ascvec.copy_gm2ub_nddma %196[%c0], %185[%c0], %c64_i32, %c1_i32, %c1_i32, %c1_i32, %c1_i32, %c1_i64, %182, %c0_i64, %c0_i64, %c0_i64, %c1_i64, %c64_i64, %c0_i64, %c0_i64, %c0_i64 l2_cache_ctl %c0_i32 left_padding(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) right_padding(%c0_i32, %c7_i32, %c0_i32, %c0_i32, %c0_i32) pad_value %c0_i32 : i32 {nearest = true} : memref<1xui8, 1>, memref<64xui8>
                        %197 = arith.muli %arg43, %c544_i64 overflow<nsw> : i64
                        %198 = arith.addi %197, %c512_i64 overflow<nsw> : i64
                        %199 = arith.muli %198, %c8_i64 overflow<nsw> : i64
                        %200 = arith.divui %199, %c8_i64 : i64
                        %201 = arith.addi %193, %200 : i64
                        %202 = ascscalar.bind_local_ptr %201 : memref<1xui8, 1>
                        ascvec.copy_gm2ub_nddma %202[%c0], %191[%c0], %c4_i32, %c1_i32, %c1_i32, %c1_i32, %c1_i32, %c1_i64, %188, %c0_i64, %c0_i64, %c0_i64, %c1_i64, %c4_i64, %c0_i64, %c0_i64, %c0_i64 l2_cache_ctl %c0_i32 left_padding(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) right_padding(%c0_i32, %c7_i32, %c0_i32, %c0_i32, %c0_i32) pad_value %c0_i32 : i32 {nearest = true} : memref<1xui8, 1>, memref<4xui8>
                      }
                      %173 = arith.cmpi eq, %170, %c2_i64 : i64
                      scf.if %173 {
                        %179 = arith.addi %141, %168 : i64
                        %180 = cannir.extract_layout %arg1 : !cannir.tensor<<gm, ui8>, <<(-1, 1, 64)>, <(-1, 64, 1)>>, <<(-1, 1, 64)>, <(-1, 64, 1)>>, identity>
                        %181 = cannir.extract_stride %180 : <<(-1, 1, 64)>, <(-1, 64, 1)>> -> <(-1, 64, 1)>
                        %182 = cannir.extract_dim %181[0] : !cannir.stride<(-1, 64, 1)>
                        %183 = arith.muli %179, %182 : i64
                        %184 = arith.addi %arg2, %183 : i64
                        %185 = ascscalar.bind_gm_ptr %184 : memref<128xui8>
                        %186 = cannir.extract_layout %arg3 : !cannir.tensor<<gm, ui8>, <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>>, <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>>, identity>
                        %187 = cannir.extract_stride %186 : <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>> -> <(-1, 4, 2, 1)>
                        %188 = cannir.extract_dim %187[0] : !cannir.stride<(-1, 4, 2, 1)>
                        %189 = arith.muli %179, %188 : i64
                        %190 = arith.addi %arg4, %189 : i64
                        %191 = ascscalar.bind_gm_ptr %190 : memref<8xui8>
                        %192 = arith.muli %arg43, %c4352_i64 overflow<nsw> : i64
                        %193 = ascscalar.memref_address %alloc_13 : memref<17408xui8, 1>
                        %194 = arith.divui %192, %c8_i64 : i64
                        %195 = arith.addi %193, %194 : i64
                        %196 = ascscalar.bind_local_ptr %195 : memref<1xui8, 1>
                        ascvec.copy_gm2ub_nddma %196[%c0], %185[%c0], %c64_i32, %c2_i32, %c1_i32, %c1_i32, %c1_i32, %c1_i64, %182, %c0_i64, %c0_i64, %c0_i64, %c1_i64, %c64_i64, %c0_i64, %c0_i64, %c0_i64 l2_cache_ctl %c0_i32 left_padding(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) right_padding(%c0_i32, %c6_i32, %c0_i32, %c0_i32, %c0_i32) pad_value %c0_i32 : i32 {nearest = true} : memref<1xui8, 1>, memref<128xui8>
                        %197 = arith.muli %arg43, %c544_i64 overflow<nsw> : i64
                        %198 = arith.addi %197, %c512_i64 overflow<nsw> : i64
                        %199 = arith.muli %198, %c8_i64 overflow<nsw> : i64
                        %200 = arith.divui %199, %c8_i64 : i64
                        %201 = arith.addi %193, %200 : i64
                        %202 = ascscalar.bind_local_ptr %201 : memref<1xui8, 1>
                        ascvec.copy_gm2ub_nddma %202[%c0], %191[%c0], %c4_i32, %c2_i32, %c1_i32, %c1_i32, %c1_i32, %c1_i64, %188, %c0_i64, %c0_i64, %c0_i64, %c1_i64, %c4_i64, %c0_i64, %c0_i64, %c0_i64 l2_cache_ctl %c0_i32 left_padding(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) right_padding(%c0_i32, %c6_i32, %c0_i32, %c0_i32, %c0_i32) pad_value %c0_i32 : i32 {nearest = true} : memref<1xui8, 1>, memref<8xui8>
                      }
                      %174 = arith.cmpi eq, %170, %c3_i64 : i64
                      scf.if %174 {
                        %179 = arith.addi %141, %168 : i64
                        %180 = cannir.extract_layout %arg1 : !cannir.tensor<<gm, ui8>, <<(-1, 1, 64)>, <(-1, 64, 1)>>, <<(-1, 1, 64)>, <(-1, 64, 1)>>, identity>
                        %181 = cannir.extract_stride %180 : <<(-1, 1, 64)>, <(-1, 64, 1)>> -> <(-1, 64, 1)>
                        %182 = cannir.extract_dim %181[0] : !cannir.stride<(-1, 64, 1)>
                        %183 = arith.muli %179, %182 : i64
                        %184 = arith.addi %arg2, %183 : i64
                        %185 = ascscalar.bind_gm_ptr %184 : memref<192xui8>
                        %186 = cannir.extract_layout %arg3 : !cannir.tensor<<gm, ui8>, <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>>, <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>>, identity>
                        %187 = cannir.extract_stride %186 : <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>> -> <(-1, 4, 2, 1)>
                        %188 = cannir.extract_dim %187[0] : !cannir.stride<(-1, 4, 2, 1)>
                        %189 = arith.muli %179, %188 : i64
                        %190 = arith.addi %arg4, %189 : i64
                        %191 = ascscalar.bind_gm_ptr %190 : memref<12xui8>
                        %192 = arith.muli %arg43, %c4352_i64 overflow<nsw> : i64
                        %193 = ascscalar.memref_address %alloc_13 : memref<17408xui8, 1>
                        %194 = arith.divui %192, %c8_i64 : i64
                        %195 = arith.addi %193, %194 : i64
                        %196 = ascscalar.bind_local_ptr %195 : memref<1xui8, 1>
                        ascvec.copy_gm2ub_nddma %196[%c0], %185[%c0], %c64_i32, %c3_i32, %c1_i32, %c1_i32, %c1_i32, %c1_i64, %182, %c0_i64, %c0_i64, %c0_i64, %c1_i64, %c64_i64, %c0_i64, %c0_i64, %c0_i64 l2_cache_ctl %c0_i32 left_padding(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) right_padding(%c0_i32, %c5_i32, %c0_i32, %c0_i32, %c0_i32) pad_value %c0_i32 : i32 {nearest = true} : memref<1xui8, 1>, memref<192xui8>
                        %197 = arith.muli %arg43, %c544_i64 overflow<nsw> : i64
                        %198 = arith.addi %197, %c512_i64 overflow<nsw> : i64
                        %199 = arith.muli %198, %c8_i64 overflow<nsw> : i64
                        %200 = arith.divui %199, %c8_i64 : i64
                        %201 = arith.addi %193, %200 : i64
                        %202 = ascscalar.bind_local_ptr %201 : memref<1xui8, 1>
                        ascvec.copy_gm2ub_nddma %202[%c0], %191[%c0], %c4_i32, %c3_i32, %c1_i32, %c1_i32, %c1_i32, %c1_i64, %188, %c0_i64, %c0_i64, %c0_i64, %c1_i64, %c4_i64, %c0_i64, %c0_i64, %c0_i64 l2_cache_ctl %c0_i32 left_padding(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) right_padding(%c0_i32, %c5_i32, %c0_i32, %c0_i32, %c0_i32) pad_value %c0_i32 : i32 {nearest = true} : memref<1xui8, 1>, memref<12xui8>
                      }
                      %175 = arith.cmpi eq, %170, %c4_i64 : i64
                      scf.if %175 {
                        %179 = arith.addi %141, %168 : i64
                        %180 = cannir.extract_layout %arg1 : !cannir.tensor<<gm, ui8>, <<(-1, 1, 64)>, <(-1, 64, 1)>>, <<(-1, 1, 64)>, <(-1, 64, 1)>>, identity>
                        %181 = cannir.extract_stride %180 : <<(-1, 1, 64)>, <(-1, 64, 1)>> -> <(-1, 64, 1)>
                        %182 = cannir.extract_dim %181[0] : !cannir.stride<(-1, 64, 1)>
                        %183 = arith.muli %179, %182 : i64
                        %184 = arith.addi %arg2, %183 : i64
                        %185 = ascscalar.bind_gm_ptr %184 : memref<256xui8>
                        %186 = cannir.extract_layout %arg3 : !cannir.tensor<<gm, ui8>, <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>>, <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>>, identity>
                        %187 = cannir.extract_stride %186 : <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>> -> <(-1, 4, 2, 1)>
                        %188 = cannir.extract_dim %187[0] : !cannir.stride<(-1, 4, 2, 1)>
                        %189 = arith.muli %179, %188 : i64
                        %190 = arith.addi %arg4, %189 : i64
                        %191 = ascscalar.bind_gm_ptr %190 : memref<16xui8>
                        %192 = arith.muli %arg43, %c4352_i64 overflow<nsw> : i64
                        %193 = ascscalar.memref_address %alloc_13 : memref<17408xui8, 1>
                        %194 = arith.divui %192, %c8_i64 : i64
                        %195 = arith.addi %193, %194 : i64
                        %196 = ascscalar.bind_local_ptr %195 : memref<1xui8, 1>
                        ascvec.copy_gm2ub_nddma %196[%c0], %185[%c0], %c64_i32, %c4_i32, %c1_i32, %c1_i32, %c1_i32, %c1_i64, %182, %c0_i64, %c0_i64, %c0_i64, %c1_i64, %c64_i64, %c0_i64, %c0_i64, %c0_i64 l2_cache_ctl %c0_i32 left_padding(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) right_padding(%c0_i32, %c4_i32, %c0_i32, %c0_i32, %c0_i32) pad_value %c0_i32 : i32 {nearest = true} : memref<1xui8, 1>, memref<256xui8>
                        %197 = arith.muli %arg43, %c544_i64 overflow<nsw> : i64
                        %198 = arith.addi %197, %c512_i64 overflow<nsw> : i64
                        %199 = arith.muli %198, %c8_i64 overflow<nsw> : i64
                        %200 = arith.divui %199, %c8_i64 : i64
                        %201 = arith.addi %193, %200 : i64
                        %202 = ascscalar.bind_local_ptr %201 : memref<1xui8, 1>
                        ascvec.copy_gm2ub_nddma %202[%c0], %191[%c0], %c4_i32, %c4_i32, %c1_i32, %c1_i32, %c1_i32, %c1_i64, %188, %c0_i64, %c0_i64, %c0_i64, %c1_i64, %c4_i64, %c0_i64, %c0_i64, %c0_i64 l2_cache_ctl %c0_i32 left_padding(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) right_padding(%c0_i32, %c4_i32, %c0_i32, %c0_i32, %c0_i32) pad_value %c0_i32 : i32 {nearest = true} : memref<1xui8, 1>, memref<16xui8>
                      }
                      %176 = arith.cmpi eq, %170, %c5_i64 : i64
                      scf.if %176 {
                        %179 = arith.addi %141, %168 : i64
                        %180 = cannir.extract_layout %arg1 : !cannir.tensor<<gm, ui8>, <<(-1, 1, 64)>, <(-1, 64, 1)>>, <<(-1, 1, 64)>, <(-1, 64, 1)>>, identity>
                        %181 = cannir.extract_stride %180 : <<(-1, 1, 64)>, <(-1, 64, 1)>> -> <(-1, 64, 1)>
                        %182 = cannir.extract_dim %181[0] : !cannir.stride<(-1, 64, 1)>
                        %183 = arith.muli %179, %182 : i64
                        %184 = arith.addi %arg2, %183 : i64
                        %185 = ascscalar.bind_gm_ptr %184 : memref<320xui8>
                        %186 = cannir.extract_layout %arg3 : !cannir.tensor<<gm, ui8>, <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>>, <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>>, identity>
                        %187 = cannir.extract_stride %186 : <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>> -> <(-1, 4, 2, 1)>
                        %188 = cannir.extract_dim %187[0] : !cannir.stride<(-1, 4, 2, 1)>
                        %189 = arith.muli %179, %188 : i64
                        %190 = arith.addi %arg4, %189 : i64
                        %191 = ascscalar.bind_gm_ptr %190 : memref<20xui8>
                        %192 = arith.muli %arg43, %c4352_i64 overflow<nsw> : i64
                        %193 = ascscalar.memref_address %alloc_13 : memref<17408xui8, 1>
                        %194 = arith.divui %192, %c8_i64 : i64
                        %195 = arith.addi %193, %194 : i64
                        %196 = ascscalar.bind_local_ptr %195 : memref<1xui8, 1>
                        ascvec.copy_gm2ub_nddma %196[%c0], %185[%c0], %c64_i32, %c5_i32, %c1_i32, %c1_i32, %c1_i32, %c1_i64, %182, %c0_i64, %c0_i64, %c0_i64, %c1_i64, %c64_i64, %c0_i64, %c0_i64, %c0_i64 l2_cache_ctl %c0_i32 left_padding(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) right_padding(%c0_i32, %c3_i32, %c0_i32, %c0_i32, %c0_i32) pad_value %c0_i32 : i32 {nearest = true} : memref<1xui8, 1>, memref<320xui8>
                        %197 = arith.muli %arg43, %c544_i64 overflow<nsw> : i64
                        %198 = arith.addi %197, %c512_i64 overflow<nsw> : i64
                        %199 = arith.muli %198, %c8_i64 overflow<nsw> : i64
                        %200 = arith.divui %199, %c8_i64 : i64
                        %201 = arith.addi %193, %200 : i64
                        %202 = ascscalar.bind_local_ptr %201 : memref<1xui8, 1>
                        ascvec.copy_gm2ub_nddma %202[%c0], %191[%c0], %c4_i32, %c5_i32, %c1_i32, %c1_i32, %c1_i32, %c1_i64, %188, %c0_i64, %c0_i64, %c0_i64, %c1_i64, %c4_i64, %c0_i64, %c0_i64, %c0_i64 l2_cache_ctl %c0_i32 left_padding(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) right_padding(%c0_i32, %c3_i32, %c0_i32, %c0_i32, %c0_i32) pad_value %c0_i32 : i32 {nearest = true} : memref<1xui8, 1>, memref<20xui8>
                      }
                      %177 = arith.cmpi eq, %170, %c6_i64 : i64
                      scf.if %177 {
                        %179 = arith.addi %141, %168 : i64
                        %180 = cannir.extract_layout %arg1 : !cannir.tensor<<gm, ui8>, <<(-1, 1, 64)>, <(-1, 64, 1)>>, <<(-1, 1, 64)>, <(-1, 64, 1)>>, identity>
                        %181 = cannir.extract_stride %180 : <<(-1, 1, 64)>, <(-1, 64, 1)>> -> <(-1, 64, 1)>
                        %182 = cannir.extract_dim %181[0] : !cannir.stride<(-1, 64, 1)>
                        %183 = arith.muli %179, %182 : i64
                        %184 = arith.addi %arg2, %183 : i64
                        %185 = ascscalar.bind_gm_ptr %184 : memref<384xui8>
                        %186 = cannir.extract_layout %arg3 : !cannir.tensor<<gm, ui8>, <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>>, <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>>, identity>
                        %187 = cannir.extract_stride %186 : <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>> -> <(-1, 4, 2, 1)>
                        %188 = cannir.extract_dim %187[0] : !cannir.stride<(-1, 4, 2, 1)>
                        %189 = arith.muli %179, %188 : i64
                        %190 = arith.addi %arg4, %189 : i64
                        %191 = ascscalar.bind_gm_ptr %190 : memref<24xui8>
                        %192 = arith.muli %arg43, %c4352_i64 overflow<nsw> : i64
                        %193 = ascscalar.memref_address %alloc_13 : memref<17408xui8, 1>
                        %194 = arith.divui %192, %c8_i64 : i64
                        %195 = arith.addi %193, %194 : i64
                        %196 = ascscalar.bind_local_ptr %195 : memref<1xui8, 1>
                        ascvec.copy_gm2ub_nddma %196[%c0], %185[%c0], %c64_i32, %c6_i32, %c1_i32, %c1_i32, %c1_i32, %c1_i64, %182, %c0_i64, %c0_i64, %c0_i64, %c1_i64, %c64_i64, %c0_i64, %c0_i64, %c0_i64 l2_cache_ctl %c0_i32 left_padding(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) right_padding(%c0_i32, %c2_i32, %c0_i32, %c0_i32, %c0_i32) pad_value %c0_i32 : i32 {nearest = true} : memref<1xui8, 1>, memref<384xui8>
                        %197 = arith.muli %arg43, %c544_i64 overflow<nsw> : i64
                        %198 = arith.addi %197, %c512_i64 overflow<nsw> : i64
                        %199 = arith.muli %198, %c8_i64 overflow<nsw> : i64
                        %200 = arith.divui %199, %c8_i64 : i64
                        %201 = arith.addi %193, %200 : i64
                        %202 = ascscalar.bind_local_ptr %201 : memref<1xui8, 1>
                        ascvec.copy_gm2ub_nddma %202[%c0], %191[%c0], %c4_i32, %c6_i32, %c1_i32, %c1_i32, %c1_i32, %c1_i64, %188, %c0_i64, %c0_i64, %c0_i64, %c1_i64, %c4_i64, %c0_i64, %c0_i64, %c0_i64 l2_cache_ctl %c0_i32 left_padding(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) right_padding(%c0_i32, %c2_i32, %c0_i32, %c0_i32, %c0_i32) pad_value %c0_i32 : i32 {nearest = true} : memref<1xui8, 1>, memref<24xui8>
                      }
                      %178 = arith.cmpi eq, %170, %c7_i64 : i64
                      scf.if %178 {
                        %179 = arith.addi %141, %168 : i64
                        %180 = cannir.extract_layout %arg1 : !cannir.tensor<<gm, ui8>, <<(-1, 1, 64)>, <(-1, 64, 1)>>, <<(-1, 1, 64)>, <(-1, 64, 1)>>, identity>
                        %181 = cannir.extract_stride %180 : <<(-1, 1, 64)>, <(-1, 64, 1)>> -> <(-1, 64, 1)>
                        %182 = cannir.extract_dim %181[0] : !cannir.stride<(-1, 64, 1)>
                        %183 = arith.muli %179, %182 : i64
                        %184 = arith.addi %arg2, %183 : i64
                        %185 = ascscalar.bind_gm_ptr %184 : memref<448xui8>
                        %186 = cannir.extract_layout %arg3 : !cannir.tensor<<gm, ui8>, <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>>, <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>>, identity>
                        %187 = cannir.extract_stride %186 : <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>> -> <(-1, 4, 2, 1)>
                        %188 = cannir.extract_dim %187[0] : !cannir.stride<(-1, 4, 2, 1)>
                        %189 = arith.muli %179, %188 : i64
                        %190 = arith.addi %arg4, %189 : i64
                        %191 = ascscalar.bind_gm_ptr %190 : memref<28xui8>
                        %192 = arith.muli %arg43, %c4352_i64 overflow<nsw> : i64
                        %193 = ascscalar.memref_address %alloc_13 : memref<17408xui8, 1>
                        %194 = arith.divui %192, %c8_i64 : i64
                        %195 = arith.addi %193, %194 : i64
                        %196 = ascscalar.bind_local_ptr %195 : memref<1xui8, 1>
                        ascvec.copy_gm2ub_nddma %196[%c0], %185[%c0], %c64_i32, %c7_i32, %c1_i32, %c1_i32, %c1_i32, %c1_i64, %182, %c0_i64, %c0_i64, %c0_i64, %c1_i64, %c64_i64, %c0_i64, %c0_i64, %c0_i64 l2_cache_ctl %c0_i32 left_padding(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) right_padding(%c0_i32, %c1_i32, %c0_i32, %c0_i32, %c0_i32) pad_value %c0_i32 : i32 {nearest = true} : memref<1xui8, 1>, memref<448xui8>
                        %197 = arith.muli %arg43, %c544_i64 overflow<nsw> : i64
                        %198 = arith.addi %197, %c512_i64 overflow<nsw> : i64
                        %199 = arith.muli %198, %c8_i64 overflow<nsw> : i64
                        %200 = arith.divui %199, %c8_i64 : i64
                        %201 = arith.addi %193, %200 : i64
                        %202 = ascscalar.bind_local_ptr %201 : memref<1xui8, 1>
                        ascvec.copy_gm2ub_nddma %202[%c0], %191[%c0], %c4_i32, %c7_i32, %c1_i32, %c1_i32, %c1_i32, %c1_i64, %188, %c0_i64, %c0_i64, %c0_i64, %c1_i64, %c4_i64, %c0_i64, %c0_i64, %c0_i64 l2_cache_ctl %c0_i32 left_padding(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) right_padding(%c0_i32, %c1_i32, %c0_i32, %c0_i32, %c0_i32) pad_value %c0_i32 : i32 {nearest = true} : memref<1xui8, 1>, memref<28xui8>
                      }
                    }
                  }
                  ascvec.sync_notify {dst_pipe = 6 : i32, event_id = 2 : i32, src_pipe = 5 : i32}
                  ascvec.sync_wait {dst_pipe = 6 : i32, event_id = 2 : i32, src_pipe = 5 : i32}
                  %145 = arith.muli %136, %c512_i64 : i64
                  %146 = ascscalar.memref_address %alloc_7 : memref<4718592xui8>
                  %147 = arith.divui %145, %c8_i64 : i64
                  %148 = arith.addi %146, %147 : i64
                  %149 = ascscalar.bind_gm_ptr %148 : memref<1xui8>
                  %150 = ascscalar.bitcast_memref %alloc_13 byte_offset = %c0_i64 : memref<17408xui8, 1> -> memref<?xui8, 1>
                  ascvec.copy_ub2gm %149[%c0], %150[%c0], %c32_i32, %c512_i32, %c512_i64, %c544_i32 l2_cache_ctl %c0_i32 : memref<1xui8>, memref<?xui8, 1>
                  %151 = arith.muli %136, %c32_i64 : i64
                  %152 = ascscalar.memref_address %alloc_8 : memref<294912xui8>
                  %153 = arith.divui %151, %c8_i64 : i64
                  %154 = arith.addi %152, %153 : i64
                  %155 = ascscalar.bind_gm_ptr %154 : memref<1xui8>
                  %156 = ascscalar.memref_address %alloc_13 : memref<17408xui8, 1>
                  %157 = arith.addi %156, %c512_i64 : i64
                  %158 = ascscalar.bind_local_ptr %157 : memref<1xui8, 1>
                  ascvec.copy_ub2gm %155[%c0], %158[%c0], %c32_i32, %c32_i32, %c32_i64, %c544_i32 l2_cache_ctl %c0_i32 : memref<1xui8>, memref<1xui8, 1>
                  ascvec.sync_notify {dst_pipe = 5 : i32, event_id = 3 : i32, src_pipe = 6 : i32}
                }
                %119 = arith.remsi %arg41, %c4_i64 : i64
                %120 = arith.cmpi ne, %119, %c0_i64 : i64
                %121 = arith.cmpi slt, %119, %c0_i64 : i64
                %122 = arith.andi %120, %121 : i1
                %123 = arith.select %122, %c4_i64, %c0_i64 : i64
                %124 = arith.addi %119, %123 overflow<nsw> : i64
                %125 = arith.addi %124, %c6_i64 overflow<nsw> : i64
                ascvec.sync_intra_arrive mte3, %125
              }
              %107 = arith.cmpi sgt, %arg41, %54 : i64
              %108 = scf.if %107 -> (i64) {
                %109 = arith.subi %arg41, %c1_i64 : i64
                %110 = arith.remsi %109, %c2_i64 : i64
                %111 = arith.cmpi ne, %110, %c0_i64 : i64
                %112 = arith.cmpi slt, %110, %c0_i64 : i64
                %113 = arith.andi %111, %112 : i1
                %114 = arith.select %113, %c2_i64, %c0_i64 : i64
                %115 = arith.addi %110, %114 overflow<nsw> : i64
                %116 = arith.addi %arg42, %c1_i64 : i64
                %117 = arith.remui %116, %c2_i64 : i64
                %118 = arith.addi %115, %c4_i64 overflow<nsw> : i64
                %119 = arith.muli %115, %c8192_i64 overflow<nsw> : i64
                %120 = arith.index_cast %119 : i64 to index
                %121 = arith.addi %115, %c14_i64 overflow<nsw> : i64
                ascvec.sync_intra_wait v, %121
                %122 = arith.subi %109, %54 : i64
                %123 = arith.muli %109, %c32_i64 : i64
                %124 = arith.trunci %123 : i64 to i32
                %125 = arith.muli %122, %c256_i64 : i64
                %126 = arith.index_cast %125 : i64 to index
                %127 = arith.addi %125, %c128_i64 : i64
                %128 = arith.index_cast %127 : i64 to index
                ascvec.sync_intra_wait v, %c0_i64 {cannir.sync.banks = 3 : i32}
                ascvec.get_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
                ascvec.vf(%109, %7, %alloc_18, %alloc_19, %alloc_16, %alloc_11, %120, %124, %alloc_14, %alloc_17, %126, %128 : i64, i64, memref<2048xi32, 1>, memref<512xui16, 1>, memref<32xbf16, 1>, memref<16384xbf16, 1>, index, i32, memref<1024xui32, 1>, memref<16384xui16, 1>, index, index) {
                ^bb0(%arg43: i64, %arg44: i64, %arg45: memref<2048xi32, 1>, %arg46: memref<512xui16, 1>, %arg47: memref<32xbf16, 1>, %arg48: memref<16384xbf16, 1>, %arg49: index, %arg50: i32, %arg51: memref<1024xui32, 1>, %arg52: memref<16384xui16, 1>, %arg53: index, %arg54: index):
                  %c128 = arith.constant 128 : index
                  %c1024_i32 = arith.constant 1024 : i32
                  %c1_i16 = arith.constant 1 : i16
                  %c0_i16 = arith.constant 0 : i16
                  %c-1_i16 = arith.constant -1 : i16
                  %c-32768_i16 = arith.constant -32768 : i16
                  %c16_i32 = arith.constant 16 : i32
                  %c8_i32 = arith.constant 8 : i32
                  %c3_i32_36 = arith.constant 3 : i32
                  %c2_i32_37 = arith.constant 2 : i32
                  %c256_i32 = arith.constant 256 : i32
                  %cst = arith.constant 0.000000e+00 : bf16
                  %c4_i32_38 = arith.constant 4 : i32
                  %c0_39 = arith.constant 0 : index
                  %c8_i64_40 = arith.constant 8 : i64
                  %c32_i64_41 = arith.constant 32 : i64
                  %c64_i64_42 = arith.constant 64 : i64
                  %c7_i32_43 = arith.constant 7 : i32
                  %c0_i32_44 = arith.constant 0 : i32
                  %c1_i32_45 = arith.constant 1 : i32
                  %c64_i32_46 = arith.constant 64 : i32
                  %c128_i32 = arith.constant 128 : i32
                  %129 = ascvec.create_mask : !ascvec.mask
                  %mask, %remaining_out = ascvec.update_mask %c128_i32 elem_bits 16 : !ascvec.mask
                  %mask_47, %remaining_out_48 = ascvec.update_mask %c64_i32_46 elem_bits 16 : !ascvec.mask
                  %130 = ascvec.duplicate_scalar %c1_i32_45, %129 : i32 -> <ui32>
                  %131 = ascvec.duplicate_scalar %c0_i32_44, %129 : i32 -> <ui32>
                  scf.for %arg55 = %c0_i32_44 to %c4_i32_38 step %c1_i32_45  : i32 {
                    %134 = arith.muli %arg55, %c64_i32_46 overflow<nsw> : i32
                    %135 = ascvec.arange %134 : i32 -> <ui32>
                    %136 = ascvec.duplicate_scalar %c7_i32_43, %129 : i32 -> <ui32>
                    %137 = ascvec.and %135, %136, %129 : <ui32>, <ui32> -> <ui32>
                    %138 = arith.muli %arg43, %c64_i64_42 : i64
                    %139 = arith.muli %arg44, %c32_i64_41 : i64
                    %140 = arith.addi %138, %139 : i64
                    %141 = arith.extsi %arg55 : i32 to i64
                    %142 = arith.muli %141, %c8_i64_40 overflow<nsw> : i64
                    %143 = arith.addi %140, %142 : i64
                    %144 = arith.index_cast %143 : i64 to index
                    %145 = ascvec.loadalign_brc %arg45[%144] {brc_width = "e2b_b32"} : memref<2048xi32, 1>, index -> <ui32>
                    %146 = ascvec.cmp_lt %137, %145, %129 : <ui32>, <ui32> -> !ascvec.mask
                    %147 = ascvec.select %130, %131, %146 : <ui32>
                    %148 = ascvec.pack %147 src_elem_bits 32 dst_elem_bits 16 {part = "low"} : <ui32> -> <ui16>
                    %149 = arith.index_cast %134 : i32 to index
                    ascvec.storealign %arg46[%149], %148, %mask_47 : memref<512xui16, 1>, index, <ui16>
                  } {cannir.loop_unroll = 1 : i64}
                  ascvec.mem_bar vst_vld
                  %132 = ascvec.duplicate_scalar %cst, %mask : bf16 -> <bf16>
                  %133:2 = scf.for %arg55 = %c0_i32_44 to %c8_i32 step %c1_i32_45 iter_args(%arg56 = %132, %arg57 = %132) -> (!ascvec.vreg<bf16>, !ascvec.vreg<bf16>)  : i32 {
                    %134 = arith.muli %arg55, %c4_i32_38 overflow<nsw> : i32
                    %135 = arith.index_cast %134 : i32 to index
                    %136 = ascvec.loadalign_brc %arg47[%135] : memref<32xbf16, 1>, index -> <bf16>
                    %137 = arith.muli %arg55, %c1024_i32 overflow<nsw> : i32
                    %138 = arith.index_cast %137 : i32 to index
                    %139 = arith.addi %arg49, %138 : index
                    %140 = ascvec.loadalign %arg48[%139] : memref<16384xbf16, 1>, index -> <bf16>
                    %141 = ascvec.max_scalar %140, %cst, %mask : <bf16>, bf16
                    %142 = arith.addi %137, %c128_i32 overflow<nsw> : i32
                    %143 = arith.index_cast %142 : i32 to index
                    %144 = arith.addi %arg49, %143 : index
                    %145 = ascvec.loadalign %arg48[%144] : memref<16384xbf16, 1>, index -> <bf16>
                    %146 = ascvec.max_scalar %145, %cst, %mask : <bf16>, bf16
                    %147 = arith.addi %134, %c1_i32_45 overflow<nsw> : i32
                    %148 = arith.index_cast %147 : i32 to index
                    %149 = ascvec.loadalign_brc %arg47[%148] : memref<32xbf16, 1>, index -> <bf16>
                    %150 = arith.muli %147, %c256_i32 overflow<nsw> : i32
                    %151 = arith.index_cast %150 : i32 to index
                    %152 = arith.addi %arg49, %151 : index
                    %153 = ascvec.loadalign %arg48[%152] : memref<16384xbf16, 1>, index -> <bf16>
                    %154 = ascvec.max_scalar %153, %cst, %mask : <bf16>, bf16
                    %155 = arith.addi %150, %c128_i32 overflow<nsw> : i32
                    %156 = arith.index_cast %155 : i32 to index
                    %157 = arith.addi %arg49, %156 : index
                    %158 = ascvec.loadalign %arg48[%157] : memref<16384xbf16, 1>, index -> <bf16>
                    %159 = ascvec.max_scalar %158, %cst, %mask : <bf16>, bf16
                    %160 = arith.addi %134, %c2_i32_37 overflow<nsw> : i32
                    %161 = arith.index_cast %160 : i32 to index
                    %162 = ascvec.loadalign_brc %arg47[%161] : memref<32xbf16, 1>, index -> <bf16>
                    %163 = arith.muli %160, %c256_i32 overflow<nsw> : i32
                    %164 = arith.index_cast %163 : i32 to index
                    %165 = arith.addi %arg49, %164 : index
                    %166 = ascvec.loadalign %arg48[%165] : memref<16384xbf16, 1>, index -> <bf16>
                    %167 = ascvec.max_scalar %166, %cst, %mask : <bf16>, bf16
                    %168 = arith.addi %163, %c128_i32 overflow<nsw> : i32
                    %169 = arith.index_cast %168 : i32 to index
                    %170 = arith.addi %arg49, %169 : index
                    %171 = ascvec.loadalign %arg48[%170] : memref<16384xbf16, 1>, index -> <bf16>
                    %172 = ascvec.max_scalar %171, %cst, %mask : <bf16>, bf16
                    %173 = arith.addi %134, %c3_i32_36 overflow<nsw> : i32
                    %174 = arith.index_cast %173 : i32 to index
                    %175 = ascvec.loadalign_brc %arg47[%174] : memref<32xbf16, 1>, index -> <bf16>
                    %176 = arith.muli %173, %c256_i32 overflow<nsw> : i32
                    %177 = arith.index_cast %176 : i32 to index
                    %178 = arith.addi %arg49, %177 : index
                    %179 = ascvec.loadalign %arg48[%178] : memref<16384xbf16, 1>, index -> <bf16>
                    %180 = ascvec.max_scalar %179, %cst, %mask : <bf16>, bf16
                    %181 = arith.addi %176, %c128_i32 overflow<nsw> : i32
                    %182 = arith.index_cast %181 : i32 to index
                    %183 = arith.addi %arg49, %182 : index
                    %184 = ascvec.loadalign %arg48[%183] : memref<16384xbf16, 1>, index -> <bf16>
                    %185 = ascvec.max_scalar %184, %cst, %mask : <bf16>, bf16
                    %186 = ascvec.madd_two_address %141, %136, %arg56, %mask : <bf16>
                    %187 = ascvec.madd_two_address %146, %136, %arg57, %mask : <bf16>
                    %188 = ascvec.madd_two_address %154, %149, %186, %mask : <bf16>
                    %189 = ascvec.madd_two_address %159, %149, %187, %mask : <bf16>
                    %190 = ascvec.madd_two_address %167, %162, %188, %mask : <bf16>
                    %191 = ascvec.madd_two_address %172, %162, %189, %mask : <bf16>
                    %192 = ascvec.madd_two_address %180, %175, %190, %mask : <bf16>
                    %193 = ascvec.madd_two_address %185, %175, %191, %mask : <bf16>
                    scf.yield %192, %193 : !ascvec.vreg<bf16>, !ascvec.vreg<bf16>
                  } {cannir.loop_unroll = 1 : i64}
                  %c0_49 = arith.constant 0 : index
                  %c1_50 = arith.constant 1 : index
                  scf.for %arg55 = %c0_49 to %c1_50 step %c1_50 {
                    %134 = ascvec.arange %c0_i32_44 : i32 -> <ui32>
                    %135 = ascvec.shr %134, 2, %129 : <ui32> -> <ui32>
                    %136 = ascvec.add_scalar %135, %arg50, %129 : <ui32>, i32
                    %137 = ascvec.add_scalar %136, %c16_i32, %129 : <ui32>, i32
                    %138 = ascvec.gather %arg51[%c0_39], %136, %129 : memref<1024xui32, 1>, <ui32>, !ascvec.mask -> <ui32>
                    %139 = ascvec.xor %134, %138, %129 : <ui32>, <ui32> -> <ui32>
                    %140 = ascvec.gather %arg51[%c0_39], %137, %129 : memref<1024xui32, 1>, <ui32>, !ascvec.mask -> <ui32>
                    %141 = ascvec.xor %134, %140, %129 : <ui32>, <ui32> -> <ui32>
                    %142 = ascvec.reinterpret %133#0 : <bf16> -> <ui32>
                    %143 = ascvec.gather_reg %142, %139 : <ui32>, <ui32> -> <ui32>
                    %144 = ascvec.reinterpret %143 : <ui32> -> <bf16>
                    %145 = ascvec.reinterpret %133#1 : <bf16> -> <ui32>
                    %146 = ascvec.gather_reg %145, %141 : <ui32>, <ui32> -> <ui32>
                    %147 = ascvec.reinterpret %146 : <ui32> -> <bf16>
                    %148 = ascvec.add %144, %132, %mask : <bf16>
                    %149 = ascvec.reinterpret %148 : <bf16> -> <ui16>
                    %150 = ascvec.duplicate_scalar %c-32768_i16, %mask : i16 -> <ui16>
                    %151 = ascvec.and %149, %150, %mask : <ui16>, <ui16> -> <ui16>
                    %152 = ascvec.cmp_eq_scalar %151, %c-32768_i16, %mask : <ui16>, i16 -> !ascvec.mask
                    %153 = ascvec.xor %149, %150, %mask : <ui16>, <ui16> -> <ui16>
                    %154 = ascvec.duplicate_scalar %c-1_i16, %mask : i16 -> <ui16>
                    %155 = ascvec.xor %149, %154, %mask : <ui16>, <ui16> -> <ui16>
                    %156 = ascvec.select %155, %153, %152 : <ui16>
                    %157 = ascvec.duplicate_scalar %c0_i16, %mask : i16 -> <ui16>
                    %158 = ascvec.loadalign %arg46[%c0_39] : memref<512xui16, 1>, index -> <ui16>
                    %159 = ascvec.cmp_eq_scalar %158, %c1_i16, %mask : <ui16>, i16 -> !ascvec.mask
                    %160 = ascvec.select %156, %157, %159 : <ui16>
                    ascvec.storealign %arg52[%arg53], %160, %mask : memref<16384xui16, 1>, index, <ui16>
                    %161 = ascvec.add %147, %132, %mask : <bf16>
                    %162 = ascvec.reinterpret %161 : <bf16> -> <ui16>
                    %163 = ascvec.and %162, %150, %mask : <ui16>, <ui16> -> <ui16>
                    %164 = ascvec.cmp_eq_scalar %163, %c-32768_i16, %mask : <ui16>, i16 -> !ascvec.mask
                    %165 = ascvec.xor %162, %150, %mask : <ui16>, <ui16> -> <ui16>
                    %166 = ascvec.xor %162, %154, %mask : <ui16>, <ui16> -> <ui16>
                    %167 = ascvec.select %166, %165, %164 : <ui16>
                    %168 = ascvec.loadalign %arg46[%c128] : memref<512xui16, 1>, index -> <ui16>
                    %169 = ascvec.cmp_eq_scalar %168, %c1_i16, %mask : <ui16>, i16 -> !ascvec.mask
                    %170 = ascvec.select %167, %157, %169 : <ui16>
                    ascvec.storealign %arg52[%arg54], %170, %mask : memref<16384xui16, 1>, index, <ui16>
                  }
                  ascvec.mem_bar vst_vld
                } {cannir.vf_mode = "simd"}
                ascvec.release_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
                ascvec.sync_intra_arrive v, %c0_i64 {cannir.sync.banks = 3 : i32}
                ascvec.sync_intra_arrive v, %118
                scf.yield %117 : i64
              } else {
                scf.yield %arg42 : i64
              }
              scf.yield %108 : i64
            } {cannir.loop_unroll = 1 : i64}
            %95 = arith.subi %56, %c4_i64 : i64
            %96 = arith.maxsi %54, %95 : i64
            scf.for %arg41 = %96 to %56 step %c1_i64  : i64 {
              %103 = arith.remsi %arg41, %c4_i64 : i64
              %104 = arith.cmpi ne, %103, %c0_i64 : i64
              %105 = arith.cmpi slt, %103, %c0_i64 : i64
              %106 = arith.andi %104, %105 : i1
              %107 = arith.select %106, %c4_i64, %c0_i64 : i64
              %108 = arith.addi %103, %107 overflow<nsw> : i64
              %109 = arith.addi %108, %c10_i64 overflow<nsw> : i64
              ascvec.sync_intra_wait mte2, %109
            } {cannir.loop_unroll = 1 : i64}
            ascvec.sync_wait {dst_pipe = 5 : i32, event_id = 2 : i32, src_pipe = 6 : i32}
            ascvec.sync_wait {dst_pipe = 5 : i32, event_id = 3 : i32, src_pipe = 6 : i32}
            %97 = arith.subi %56, %54 : i64
            ascvec.sync_notify {dst_pipe = 6 : i32, event_id = 0 : i32, src_pipe = 2 : i32}
            ascvec.sync_wait {dst_pipe = 6 : i32, event_id = 0 : i32, src_pipe = 2 : i32}
            scf.for %arg41 = %c0_i64 to %97 step %c1_i64  : i64 {
              %103 = arith.muli %7, %c256_i64 : i64
              %104 = arith.muli %arg41, %c512_i64 : i64
              %105 = arith.addi %103, %104 : i64
              %106 = arith.muli %arg41, %c256_i64 : i64
              %107 = arith.addi %arg41, %c1_i64 : i64
              %108 = arith.muli %107, %c256_i64 : i64
              %109 = arith.subi %108, %106 : i64
              %110 = arith.muli %8, %c16384_i64 : i64
              %111 = arith.addi %110, %105 : i64
              %112 = arith.muli %111, %c16_i64 : i64
              %113 = ascscalar.memref_address %alloc_9 : memref<1xui16>
              %114 = arith.divui %112, %c8_i64 : i64
              %115 = arith.addi %113, %114 : i64
              %116 = ascscalar.bind_gm_ptr %115 : memref<1xui16>
              %117 = arith.muli %arg41, %c4096_i64 : i64
              %118 = ascscalar.memref_address %alloc_17 : memref<16384xui16, 1>
              %119 = arith.divui %117, %c8_i64 : i64
              %120 = arith.addi %118, %119 : i64
              %121 = ascscalar.bind_local_ptr %120 : memref<1xui16, 1>
              %122 = arith.index_cast %109 : i64 to index
              %123 = arith.muli %122, %c2 : index
              %124 = arith.cmpi ugt, %109, %c0_i64 : i64
              scf.if %124 {
                %125 = arith.index_cast %123 : index to i64
                ascvec.get_buf mte3, %c3_i64, 0 {cannir.buffer.generated_sync}
                %126 = arith.trunci %125 : i64 to i32
                ascvec.copy_ub2gm %116[%c0], %121[%c0], %c1_i32, %126, %c32768_i64, %c32768_i32 l2_cache_ctl %c0_i32 : memref<1xui16>, memref<1xui16, 1>
                ascvec.release_buf mte3, %c3_i64, 0 {cannir.buffer.generated_sync}
              }
            } {cannir.loop_unroll = 1 : i64}
            ascvec.sync_intra_arrive mte3, %c6_i64
            ascvec.sync_intra_wait mte2, %c7_i64
            scf.if %68 {
              %103 = cannir.extract_layout %arg19 : !cannir.tensor<<gm, ui16>, <<(-1, 16384)>, <(16384, 1)>>, <<(-1, 16384)>, <(16384, 1)>>, identity>
              %104 = cannir.extract_shape %103 : <<(-1, 16384)>, <(16384, 1)>> -> <(-1, 16384)>
              %105 = cannir.extract_dim %104[0] : !cannir.shape<(-1, 16384)>
              %106 = arith.minsi %8, %105 : i64
              %107 = arith.subi %105, %106 : i64
              %108 = arith.minsi %107, %c1_i64 : i64
              %109 = arith.cmpi sle, %8, %105 : i64
              %110 = arith.select %109, %108, %c0_i64 : i64
              %111 = arith.muli %8, %c32768_i64 : i64
              %112 = arith.muli %110, %c32768_i64 : i64
              %113 = arith.cmpi eq, %112, %c0_i64 : i64
              %114 = arith.cmpi ne, %112, %c0_i64 : i64
              %115 = arith.extui %114 : i1 to i64
              %116 = arith.select %113, %c2_i64, %112 : i64
              ascvec.get_buf mte2, %c3_i64, 0 {cannir.buffer.generated_sync}
              "ascvec.copy_gm2ub_plan"(%alloc_17, %alloc_9, %c0_i32, %c0_i32, %c0_i64, %111, %115, %116, %c0_i64, %116, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = ui16, ub_base = 71872 : i64, ub_capacity = 32768 : i64}> : (memref<16384xui16, 1>, memref<1xui16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
              ascvec.release_buf mte2, %c3_i64, 0 {cannir.buffer.generated_sync}
              ascvec.sync_notify {dst_pipe = 2 : i32, event_id = 0 : i32, src_pipe = 5 : i32}
              ascvec.sync_wait {dst_pipe = 2 : i32, event_id = 0 : i32, src_pipe = 5 : i32}
            }
            %98 = arith.addi %arg37, %c1_i64 : i64
            %99 = arith.cmpi slt, %98, %38 : i64
            %100 = arith.cmpi slt, %98, %57 : i64
            %101 = arith.select %99, %100, %false : i1
            %102 = scf.if %101 -> (i64) {
              %103 = arith.addi %40, %c1_i64 : i64
              %104 = arith.index_cast %103 : i64 to index
              %105 = memref.load %alloc_1[%104] : memref<1xi32>
              %106 = arith.extsi %105 : i32 to i64
              %107 = arith.addi %106, %c63_i64 : i64
              %108 = arith.remsi %107, %c64_i64 : i64
              %109 = arith.cmpi ne, %108, %c0_i64 : i64
              %110 = arith.cmpi slt, %108, %c0_i64 : i64
              %111 = arith.andi %109, %110 : i1
              %112 = arith.divsi %107, %c64_i64 : i64
              %113 = arith.extui %111 : i1 to i64
              %114 = arith.subi %112, %113 : i64
              %115 = scf.if %37 -> (i1) {
                %123 = arith.addi %10, %c5 : index
                %124 = memref.load %alloc_10[%123] : memref<1024xi32>
                %125 = arith.extsi %124 : i32 to i64
                %126 = arith.cmpi eq, %98, %125 : i64
                scf.yield %126 : i1
              } else {
                scf.yield %false : i1
              }
              %116 = scf.if %115 -> (i64) {
                %123 = arith.addi %10, %c6 : index
                %124 = memref.load %alloc_10[%123] : memref<1024xi32>
                %125 = arith.extsi %124 : i32 to i64
                %126 = arith.minsi %114, %125 : i64
                scf.yield %126 : i64
              } else {
                scf.yield %114 : i64
              }
              %117 = arith.cmpi sgt, %105, %c0_i32 : i32
              %118 = arith.cmpi sgt, %116, %c0_i64 : i64
              %119 = arith.cmpi sgt, %33, %c0_i32 : i32
              %120 = arith.select %118, %119, %false : i1
              %121 = arith.select %117, %120, %false : i1
              %122 = scf.if %121 -> (i64) {
                ascvec.sync_notify {dst_pipe = 5 : i32, event_id = 2 : i32, src_pipe = 6 : i32}
                ascvec.sync_notify {dst_pipe = 5 : i32, event_id = 3 : i32, src_pipe = 6 : i32}
                %123 = arith.minsi %116, %c4_i64 : i64
                scf.for %arg41 = %c0_i64 to %123 step %c1_i64  : i64 {
                  %124 = arith.remsi %arg41, %c2_i64 : i64
                  %125 = arith.cmpi eq, %124, %c0_i64 : i64
                  scf.if %125 {
                    %129 = arith.muli %8, %c4_i64 : i64
                    %130 = arith.remsi %arg41, %c4_i64 : i64
                    %131 = arith.addi %129, %130 : i64
                    %132 = arith.muli %131, %c512_i64 : i64
                    %133 = arith.muli %7, %c256_i64 : i64
                    %134 = arith.addi %132, %133 : i64
                    %135 = arith.muli %arg41, %c64_i64 : i64
                    %136 = arith.muli %7, %c32_i64 : i64
                    %137 = arith.addi %135, %136 : i64
                    %138 = memref.load %alloc_4[%25] : memref<1xi32>
                    %139 = arith.extsi %138 : i32 to i64
                    %140 = memref.load %alloc_4[%29] : memref<1xi32>
                    %141 = arith.extsi %140 : i32 to i64
                    %142 = arith.subi %141, %139 : i64
                    ascvec.sync_wait {dst_pipe = 5 : i32, event_id = 2 : i32, src_pipe = 6 : i32}
                    scf.for %arg42 = %c0_i64 to %c32_i64 step %c1_i64  : i64 {
                      %157 = arith.addi %137, %arg42 : i64
                      %158 = arith.cmpi slt, %157, %106 : i64
                      %159 = arith.subi %106, %c1_i64 : i64
                      %160 = arith.select %158, %157, %159 : i64
                      %161 = arith.index_cast %160 : i64 to index
                      %162 = arith.muli %104, %c2048 : index
                      %163 = arith.addi %162, %161 : index
                      %164 = memref.load %alloc_0[%163] : memref<1xi32>
                      %165 = arith.extsi %164 : i32 to i64
                      %166 = arith.muli %165, %c8_i64 : i64
                      %167 = arith.subi %142, %166 : i64
                      %168 = arith.minsi %167, %c8_i64 : i64
                      %169 = arith.cmpi eq, %168, %c8_i64 : i64
                      scf.if %169 {
                        %170 = arith.addi %139, %166 : i64
                        %171 = cannir.extract_layout %arg1 : !cannir.tensor<<gm, ui8>, <<(-1, 1, 64)>, <(-1, 64, 1)>>, <<(-1, 1, 64)>, <(-1, 64, 1)>>, identity>
                        %172 = cannir.extract_stride %171 : <<(-1, 1, 64)>, <(-1, 64, 1)>> -> <(-1, 64, 1)>
                        %173 = cannir.extract_dim %172[0] : !cannir.stride<(-1, 64, 1)>
                        %174 = arith.muli %170, %173 : i64
                        %175 = arith.addi %arg2, %174 : i64
                        %176 = ascscalar.bind_gm_ptr %175 : memref<512xui8>
                        %177 = cannir.extract_layout %arg3 : !cannir.tensor<<gm, ui8>, <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>>, <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>>, identity>
                        %178 = cannir.extract_stride %177 : <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>> -> <(-1, 4, 2, 1)>
                        %179 = cannir.extract_dim %178[0] : !cannir.stride<(-1, 4, 2, 1)>
                        %180 = arith.muli %170, %179 : i64
                        %181 = arith.addi %arg4, %180 : i64
                        %182 = ascscalar.bind_gm_ptr %181 : memref<32xui8>
                        %183 = arith.muli %arg42, %c544_i64 overflow<nsw> : i64
                        "ascvec.copy_gm2ub_plan"(%alloc_12, %176, %c0_i32, %c0_i32, %183, %c0_i64, %c8_i64, %c64_i64, %173, %c64_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = ui8, ub_base = 32768 : i64, ub_capacity = 17408 : i64}> : (memref<17408xui8, 1>, memref<512xui8>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                        %184 = arith.addi %183, %c512_i64 overflow<nsw> : i64
                        "ascvec.copy_gm2ub_plan"(%alloc_12, %182, %c0_i32, %c0_i32, %184, %c0_i64, %c8_i64, %c4_i64, %179, %c4_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = ui8, ub_base = 32768 : i64, ub_capacity = 17408 : i64}> : (memref<17408xui8, 1>, memref<32xui8>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                      } else {
                        %170 = arith.cmpi eq, %168, %c1_i64 : i64
                        scf.if %170 {
                          %177 = arith.addi %139, %166 : i64
                          %178 = cannir.extract_layout %arg1 : !cannir.tensor<<gm, ui8>, <<(-1, 1, 64)>, <(-1, 64, 1)>>, <<(-1, 1, 64)>, <(-1, 64, 1)>>, identity>
                          %179 = cannir.extract_stride %178 : <<(-1, 1, 64)>, <(-1, 64, 1)>> -> <(-1, 64, 1)>
                          %180 = cannir.extract_dim %179[0] : !cannir.stride<(-1, 64, 1)>
                          %181 = arith.muli %177, %180 : i64
                          %182 = arith.addi %arg2, %181 : i64
                          %183 = ascscalar.bind_gm_ptr %182 : memref<64xui8>
                          %184 = cannir.extract_layout %arg3 : !cannir.tensor<<gm, ui8>, <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>>, <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>>, identity>
                          %185 = cannir.extract_stride %184 : <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>> -> <(-1, 4, 2, 1)>
                          %186 = cannir.extract_dim %185[0] : !cannir.stride<(-1, 4, 2, 1)>
                          %187 = arith.muli %177, %186 : i64
                          %188 = arith.addi %arg4, %187 : i64
                          %189 = ascscalar.bind_gm_ptr %188 : memref<4xui8>
                          %190 = arith.muli %arg42, %c4352_i64 overflow<nsw> : i64
                          %191 = ascscalar.memref_address %alloc_12 : memref<17408xui8, 1>
                          %192 = arith.divui %190, %c8_i64 : i64
                          %193 = arith.addi %191, %192 : i64
                          %194 = ascscalar.bind_local_ptr %193 : memref<1xui8, 1>
                          ascvec.copy_gm2ub_nddma %194[%c0], %183[%c0], %c64_i32, %c1_i32, %c1_i32, %c1_i32, %c1_i32, %c1_i64, %180, %c0_i64, %c0_i64, %c0_i64, %c1_i64, %c64_i64, %c0_i64, %c0_i64, %c0_i64 l2_cache_ctl %c0_i32 left_padding(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) right_padding(%c0_i32, %c7_i32, %c0_i32, %c0_i32, %c0_i32) pad_value %c0_i32 : i32 {nearest = true} : memref<1xui8, 1>, memref<64xui8>
                          %195 = arith.muli %arg42, %c544_i64 overflow<nsw> : i64
                          %196 = arith.addi %195, %c512_i64 overflow<nsw> : i64
                          %197 = arith.muli %196, %c8_i64 overflow<nsw> : i64
                          %198 = arith.divui %197, %c8_i64 : i64
                          %199 = arith.addi %191, %198 : i64
                          %200 = ascscalar.bind_local_ptr %199 : memref<1xui8, 1>
                          ascvec.copy_gm2ub_nddma %200[%c0], %189[%c0], %c4_i32, %c1_i32, %c1_i32, %c1_i32, %c1_i32, %c1_i64, %186, %c0_i64, %c0_i64, %c0_i64, %c1_i64, %c4_i64, %c0_i64, %c0_i64, %c0_i64 l2_cache_ctl %c0_i32 left_padding(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) right_padding(%c0_i32, %c7_i32, %c0_i32, %c0_i32, %c0_i32) pad_value %c0_i32 : i32 {nearest = true} : memref<1xui8, 1>, memref<4xui8>
                        }
                        %171 = arith.cmpi eq, %168, %c2_i64 : i64
                        scf.if %171 {
                          %177 = arith.addi %139, %166 : i64
                          %178 = cannir.extract_layout %arg1 : !cannir.tensor<<gm, ui8>, <<(-1, 1, 64)>, <(-1, 64, 1)>>, <<(-1, 1, 64)>, <(-1, 64, 1)>>, identity>
                          %179 = cannir.extract_stride %178 : <<(-1, 1, 64)>, <(-1, 64, 1)>> -> <(-1, 64, 1)>
                          %180 = cannir.extract_dim %179[0] : !cannir.stride<(-1, 64, 1)>
                          %181 = arith.muli %177, %180 : i64
                          %182 = arith.addi %arg2, %181 : i64
                          %183 = ascscalar.bind_gm_ptr %182 : memref<128xui8>
                          %184 = cannir.extract_layout %arg3 : !cannir.tensor<<gm, ui8>, <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>>, <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>>, identity>
                          %185 = cannir.extract_stride %184 : <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>> -> <(-1, 4, 2, 1)>
                          %186 = cannir.extract_dim %185[0] : !cannir.stride<(-1, 4, 2, 1)>
                          %187 = arith.muli %177, %186 : i64
                          %188 = arith.addi %arg4, %187 : i64
                          %189 = ascscalar.bind_gm_ptr %188 : memref<8xui8>
                          %190 = arith.muli %arg42, %c4352_i64 overflow<nsw> : i64
                          %191 = ascscalar.memref_address %alloc_12 : memref<17408xui8, 1>
                          %192 = arith.divui %190, %c8_i64 : i64
                          %193 = arith.addi %191, %192 : i64
                          %194 = ascscalar.bind_local_ptr %193 : memref<1xui8, 1>
                          ascvec.copy_gm2ub_nddma %194[%c0], %183[%c0], %c64_i32, %c2_i32, %c1_i32, %c1_i32, %c1_i32, %c1_i64, %180, %c0_i64, %c0_i64, %c0_i64, %c1_i64, %c64_i64, %c0_i64, %c0_i64, %c0_i64 l2_cache_ctl %c0_i32 left_padding(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) right_padding(%c0_i32, %c6_i32, %c0_i32, %c0_i32, %c0_i32) pad_value %c0_i32 : i32 {nearest = true} : memref<1xui8, 1>, memref<128xui8>
                          %195 = arith.muli %arg42, %c544_i64 overflow<nsw> : i64
                          %196 = arith.addi %195, %c512_i64 overflow<nsw> : i64
                          %197 = arith.muli %196, %c8_i64 overflow<nsw> : i64
                          %198 = arith.divui %197, %c8_i64 : i64
                          %199 = arith.addi %191, %198 : i64
                          %200 = ascscalar.bind_local_ptr %199 : memref<1xui8, 1>
                          ascvec.copy_gm2ub_nddma %200[%c0], %189[%c0], %c4_i32, %c2_i32, %c1_i32, %c1_i32, %c1_i32, %c1_i64, %186, %c0_i64, %c0_i64, %c0_i64, %c1_i64, %c4_i64, %c0_i64, %c0_i64, %c0_i64 l2_cache_ctl %c0_i32 left_padding(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) right_padding(%c0_i32, %c6_i32, %c0_i32, %c0_i32, %c0_i32) pad_value %c0_i32 : i32 {nearest = true} : memref<1xui8, 1>, memref<8xui8>
                        }
                        %172 = arith.cmpi eq, %168, %c3_i64 : i64
                        scf.if %172 {
                          %177 = arith.addi %139, %166 : i64
                          %178 = cannir.extract_layout %arg1 : !cannir.tensor<<gm, ui8>, <<(-1, 1, 64)>, <(-1, 64, 1)>>, <<(-1, 1, 64)>, <(-1, 64, 1)>>, identity>
                          %179 = cannir.extract_stride %178 : <<(-1, 1, 64)>, <(-1, 64, 1)>> -> <(-1, 64, 1)>
                          %180 = cannir.extract_dim %179[0] : !cannir.stride<(-1, 64, 1)>
                          %181 = arith.muli %177, %180 : i64
                          %182 = arith.addi %arg2, %181 : i64
                          %183 = ascscalar.bind_gm_ptr %182 : memref<192xui8>
                          %184 = cannir.extract_layout %arg3 : !cannir.tensor<<gm, ui8>, <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>>, <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>>, identity>
                          %185 = cannir.extract_stride %184 : <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>> -> <(-1, 4, 2, 1)>
                          %186 = cannir.extract_dim %185[0] : !cannir.stride<(-1, 4, 2, 1)>
                          %187 = arith.muli %177, %186 : i64
                          %188 = arith.addi %arg4, %187 : i64
                          %189 = ascscalar.bind_gm_ptr %188 : memref<12xui8>
                          %190 = arith.muli %arg42, %c4352_i64 overflow<nsw> : i64
                          %191 = ascscalar.memref_address %alloc_12 : memref<17408xui8, 1>
                          %192 = arith.divui %190, %c8_i64 : i64
                          %193 = arith.addi %191, %192 : i64
                          %194 = ascscalar.bind_local_ptr %193 : memref<1xui8, 1>
                          ascvec.copy_gm2ub_nddma %194[%c0], %183[%c0], %c64_i32, %c3_i32, %c1_i32, %c1_i32, %c1_i32, %c1_i64, %180, %c0_i64, %c0_i64, %c0_i64, %c1_i64, %c64_i64, %c0_i64, %c0_i64, %c0_i64 l2_cache_ctl %c0_i32 left_padding(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) right_padding(%c0_i32, %c5_i32, %c0_i32, %c0_i32, %c0_i32) pad_value %c0_i32 : i32 {nearest = true} : memref<1xui8, 1>, memref<192xui8>
                          %195 = arith.muli %arg42, %c544_i64 overflow<nsw> : i64
                          %196 = arith.addi %195, %c512_i64 overflow<nsw> : i64
                          %197 = arith.muli %196, %c8_i64 overflow<nsw> : i64
                          %198 = arith.divui %197, %c8_i64 : i64
                          %199 = arith.addi %191, %198 : i64
                          %200 = ascscalar.bind_local_ptr %199 : memref<1xui8, 1>
                          ascvec.copy_gm2ub_nddma %200[%c0], %189[%c0], %c4_i32, %c3_i32, %c1_i32, %c1_i32, %c1_i32, %c1_i64, %186, %c0_i64, %c0_i64, %c0_i64, %c1_i64, %c4_i64, %c0_i64, %c0_i64, %c0_i64 l2_cache_ctl %c0_i32 left_padding(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) right_padding(%c0_i32, %c5_i32, %c0_i32, %c0_i32, %c0_i32) pad_value %c0_i32 : i32 {nearest = true} : memref<1xui8, 1>, memref<12xui8>
                        }
                        %173 = arith.cmpi eq, %168, %c4_i64 : i64
                        scf.if %173 {
                          %177 = arith.addi %139, %166 : i64
                          %178 = cannir.extract_layout %arg1 : !cannir.tensor<<gm, ui8>, <<(-1, 1, 64)>, <(-1, 64, 1)>>, <<(-1, 1, 64)>, <(-1, 64, 1)>>, identity>
                          %179 = cannir.extract_stride %178 : <<(-1, 1, 64)>, <(-1, 64, 1)>> -> <(-1, 64, 1)>
                          %180 = cannir.extract_dim %179[0] : !cannir.stride<(-1, 64, 1)>
                          %181 = arith.muli %177, %180 : i64
                          %182 = arith.addi %arg2, %181 : i64
                          %183 = ascscalar.bind_gm_ptr %182 : memref<256xui8>
                          %184 = cannir.extract_layout %arg3 : !cannir.tensor<<gm, ui8>, <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>>, <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>>, identity>
                          %185 = cannir.extract_stride %184 : <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>> -> <(-1, 4, 2, 1)>
                          %186 = cannir.extract_dim %185[0] : !cannir.stride<(-1, 4, 2, 1)>
                          %187 = arith.muli %177, %186 : i64
                          %188 = arith.addi %arg4, %187 : i64
                          %189 = ascscalar.bind_gm_ptr %188 : memref<16xui8>
                          %190 = arith.muli %arg42, %c4352_i64 overflow<nsw> : i64
                          %191 = ascscalar.memref_address %alloc_12 : memref<17408xui8, 1>
                          %192 = arith.divui %190, %c8_i64 : i64
                          %193 = arith.addi %191, %192 : i64
                          %194 = ascscalar.bind_local_ptr %193 : memref<1xui8, 1>
                          ascvec.copy_gm2ub_nddma %194[%c0], %183[%c0], %c64_i32, %c4_i32, %c1_i32, %c1_i32, %c1_i32, %c1_i64, %180, %c0_i64, %c0_i64, %c0_i64, %c1_i64, %c64_i64, %c0_i64, %c0_i64, %c0_i64 l2_cache_ctl %c0_i32 left_padding(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) right_padding(%c0_i32, %c4_i32, %c0_i32, %c0_i32, %c0_i32) pad_value %c0_i32 : i32 {nearest = true} : memref<1xui8, 1>, memref<256xui8>
                          %195 = arith.muli %arg42, %c544_i64 overflow<nsw> : i64
                          %196 = arith.addi %195, %c512_i64 overflow<nsw> : i64
                          %197 = arith.muli %196, %c8_i64 overflow<nsw> : i64
                          %198 = arith.divui %197, %c8_i64 : i64
                          %199 = arith.addi %191, %198 : i64
                          %200 = ascscalar.bind_local_ptr %199 : memref<1xui8, 1>
                          ascvec.copy_gm2ub_nddma %200[%c0], %189[%c0], %c4_i32, %c4_i32, %c1_i32, %c1_i32, %c1_i32, %c1_i64, %186, %c0_i64, %c0_i64, %c0_i64, %c1_i64, %c4_i64, %c0_i64, %c0_i64, %c0_i64 l2_cache_ctl %c0_i32 left_padding(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) right_padding(%c0_i32, %c4_i32, %c0_i32, %c0_i32, %c0_i32) pad_value %c0_i32 : i32 {nearest = true} : memref<1xui8, 1>, memref<16xui8>
                        }
                        %174 = arith.cmpi eq, %168, %c5_i64 : i64
                        scf.if %174 {
                          %177 = arith.addi %139, %166 : i64
                          %178 = cannir.extract_layout %arg1 : !cannir.tensor<<gm, ui8>, <<(-1, 1, 64)>, <(-1, 64, 1)>>, <<(-1, 1, 64)>, <(-1, 64, 1)>>, identity>
                          %179 = cannir.extract_stride %178 : <<(-1, 1, 64)>, <(-1, 64, 1)>> -> <(-1, 64, 1)>
                          %180 = cannir.extract_dim %179[0] : !cannir.stride<(-1, 64, 1)>
                          %181 = arith.muli %177, %180 : i64
                          %182 = arith.addi %arg2, %181 : i64
                          %183 = ascscalar.bind_gm_ptr %182 : memref<320xui8>
                          %184 = cannir.extract_layout %arg3 : !cannir.tensor<<gm, ui8>, <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>>, <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>>, identity>
                          %185 = cannir.extract_stride %184 : <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>> -> <(-1, 4, 2, 1)>
                          %186 = cannir.extract_dim %185[0] : !cannir.stride<(-1, 4, 2, 1)>
                          %187 = arith.muli %177, %186 : i64
                          %188 = arith.addi %arg4, %187 : i64
                          %189 = ascscalar.bind_gm_ptr %188 : memref<20xui8>
                          %190 = arith.muli %arg42, %c4352_i64 overflow<nsw> : i64
                          %191 = ascscalar.memref_address %alloc_12 : memref<17408xui8, 1>
                          %192 = arith.divui %190, %c8_i64 : i64
                          %193 = arith.addi %191, %192 : i64
                          %194 = ascscalar.bind_local_ptr %193 : memref<1xui8, 1>
                          ascvec.copy_gm2ub_nddma %194[%c0], %183[%c0], %c64_i32, %c5_i32, %c1_i32, %c1_i32, %c1_i32, %c1_i64, %180, %c0_i64, %c0_i64, %c0_i64, %c1_i64, %c64_i64, %c0_i64, %c0_i64, %c0_i64 l2_cache_ctl %c0_i32 left_padding(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) right_padding(%c0_i32, %c3_i32, %c0_i32, %c0_i32, %c0_i32) pad_value %c0_i32 : i32 {nearest = true} : memref<1xui8, 1>, memref<320xui8>
                          %195 = arith.muli %arg42, %c544_i64 overflow<nsw> : i64
                          %196 = arith.addi %195, %c512_i64 overflow<nsw> : i64
                          %197 = arith.muli %196, %c8_i64 overflow<nsw> : i64
                          %198 = arith.divui %197, %c8_i64 : i64
                          %199 = arith.addi %191, %198 : i64
                          %200 = ascscalar.bind_local_ptr %199 : memref<1xui8, 1>
                          ascvec.copy_gm2ub_nddma %200[%c0], %189[%c0], %c4_i32, %c5_i32, %c1_i32, %c1_i32, %c1_i32, %c1_i64, %186, %c0_i64, %c0_i64, %c0_i64, %c1_i64, %c4_i64, %c0_i64, %c0_i64, %c0_i64 l2_cache_ctl %c0_i32 left_padding(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) right_padding(%c0_i32, %c3_i32, %c0_i32, %c0_i32, %c0_i32) pad_value %c0_i32 : i32 {nearest = true} : memref<1xui8, 1>, memref<20xui8>
                        }
                        %175 = arith.cmpi eq, %168, %c6_i64 : i64
                        scf.if %175 {
                          %177 = arith.addi %139, %166 : i64
                          %178 = cannir.extract_layout %arg1 : !cannir.tensor<<gm, ui8>, <<(-1, 1, 64)>, <(-1, 64, 1)>>, <<(-1, 1, 64)>, <(-1, 64, 1)>>, identity>
                          %179 = cannir.extract_stride %178 : <<(-1, 1, 64)>, <(-1, 64, 1)>> -> <(-1, 64, 1)>
                          %180 = cannir.extract_dim %179[0] : !cannir.stride<(-1, 64, 1)>
                          %181 = arith.muli %177, %180 : i64
                          %182 = arith.addi %arg2, %181 : i64
                          %183 = ascscalar.bind_gm_ptr %182 : memref<384xui8>
                          %184 = cannir.extract_layout %arg3 : !cannir.tensor<<gm, ui8>, <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>>, <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>>, identity>
                          %185 = cannir.extract_stride %184 : <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>> -> <(-1, 4, 2, 1)>
                          %186 = cannir.extract_dim %185[0] : !cannir.stride<(-1, 4, 2, 1)>
                          %187 = arith.muli %177, %186 : i64
                          %188 = arith.addi %arg4, %187 : i64
                          %189 = ascscalar.bind_gm_ptr %188 : memref<24xui8>
                          %190 = arith.muli %arg42, %c4352_i64 overflow<nsw> : i64
                          %191 = ascscalar.memref_address %alloc_12 : memref<17408xui8, 1>
                          %192 = arith.divui %190, %c8_i64 : i64
                          %193 = arith.addi %191, %192 : i64
                          %194 = ascscalar.bind_local_ptr %193 : memref<1xui8, 1>
                          ascvec.copy_gm2ub_nddma %194[%c0], %183[%c0], %c64_i32, %c6_i32, %c1_i32, %c1_i32, %c1_i32, %c1_i64, %180, %c0_i64, %c0_i64, %c0_i64, %c1_i64, %c64_i64, %c0_i64, %c0_i64, %c0_i64 l2_cache_ctl %c0_i32 left_padding(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) right_padding(%c0_i32, %c2_i32, %c0_i32, %c0_i32, %c0_i32) pad_value %c0_i32 : i32 {nearest = true} : memref<1xui8, 1>, memref<384xui8>
                          %195 = arith.muli %arg42, %c544_i64 overflow<nsw> : i64
                          %196 = arith.addi %195, %c512_i64 overflow<nsw> : i64
                          %197 = arith.muli %196, %c8_i64 overflow<nsw> : i64
                          %198 = arith.divui %197, %c8_i64 : i64
                          %199 = arith.addi %191, %198 : i64
                          %200 = ascscalar.bind_local_ptr %199 : memref<1xui8, 1>
                          ascvec.copy_gm2ub_nddma %200[%c0], %189[%c0], %c4_i32, %c6_i32, %c1_i32, %c1_i32, %c1_i32, %c1_i64, %186, %c0_i64, %c0_i64, %c0_i64, %c1_i64, %c4_i64, %c0_i64, %c0_i64, %c0_i64 l2_cache_ctl %c0_i32 left_padding(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) right_padding(%c0_i32, %c2_i32, %c0_i32, %c0_i32, %c0_i32) pad_value %c0_i32 : i32 {nearest = true} : memref<1xui8, 1>, memref<24xui8>
                        }
                        %176 = arith.cmpi eq, %168, %c7_i64 : i64
                        scf.if %176 {
                          %177 = arith.addi %139, %166 : i64
                          %178 = cannir.extract_layout %arg1 : !cannir.tensor<<gm, ui8>, <<(-1, 1, 64)>, <(-1, 64, 1)>>, <<(-1, 1, 64)>, <(-1, 64, 1)>>, identity>
                          %179 = cannir.extract_stride %178 : <<(-1, 1, 64)>, <(-1, 64, 1)>> -> <(-1, 64, 1)>
                          %180 = cannir.extract_dim %179[0] : !cannir.stride<(-1, 64, 1)>
                          %181 = arith.muli %177, %180 : i64
                          %182 = arith.addi %arg2, %181 : i64
                          %183 = ascscalar.bind_gm_ptr %182 : memref<448xui8>
                          %184 = cannir.extract_layout %arg3 : !cannir.tensor<<gm, ui8>, <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>>, <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>>, identity>
                          %185 = cannir.extract_stride %184 : <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>> -> <(-1, 4, 2, 1)>
                          %186 = cannir.extract_dim %185[0] : !cannir.stride<(-1, 4, 2, 1)>
                          %187 = arith.muli %177, %186 : i64
                          %188 = arith.addi %arg4, %187 : i64
                          %189 = ascscalar.bind_gm_ptr %188 : memref<28xui8>
                          %190 = arith.muli %arg42, %c4352_i64 overflow<nsw> : i64
                          %191 = ascscalar.memref_address %alloc_12 : memref<17408xui8, 1>
                          %192 = arith.divui %190, %c8_i64 : i64
                          %193 = arith.addi %191, %192 : i64
                          %194 = ascscalar.bind_local_ptr %193 : memref<1xui8, 1>
                          ascvec.copy_gm2ub_nddma %194[%c0], %183[%c0], %c64_i32, %c7_i32, %c1_i32, %c1_i32, %c1_i32, %c1_i64, %180, %c0_i64, %c0_i64, %c0_i64, %c1_i64, %c64_i64, %c0_i64, %c0_i64, %c0_i64 l2_cache_ctl %c0_i32 left_padding(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) right_padding(%c0_i32, %c1_i32, %c0_i32, %c0_i32, %c0_i32) pad_value %c0_i32 : i32 {nearest = true} : memref<1xui8, 1>, memref<448xui8>
                          %195 = arith.muli %arg42, %c544_i64 overflow<nsw> : i64
                          %196 = arith.addi %195, %c512_i64 overflow<nsw> : i64
                          %197 = arith.muli %196, %c8_i64 overflow<nsw> : i64
                          %198 = arith.divui %197, %c8_i64 : i64
                          %199 = arith.addi %191, %198 : i64
                          %200 = ascscalar.bind_local_ptr %199 : memref<1xui8, 1>
                          ascvec.copy_gm2ub_nddma %200[%c0], %189[%c0], %c4_i32, %c7_i32, %c1_i32, %c1_i32, %c1_i32, %c1_i64, %186, %c0_i64, %c0_i64, %c0_i64, %c1_i64, %c4_i64, %c0_i64, %c0_i64, %c0_i64 l2_cache_ctl %c0_i32 left_padding(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) right_padding(%c0_i32, %c1_i32, %c0_i32, %c0_i32, %c0_i32) pad_value %c0_i32 : i32 {nearest = true} : memref<1xui8, 1>, memref<28xui8>
                        }
                      }
                    }
                    ascvec.sync_notify {dst_pipe = 6 : i32, event_id = 2 : i32, src_pipe = 5 : i32}
                    ascvec.sync_wait {dst_pipe = 6 : i32, event_id = 2 : i32, src_pipe = 5 : i32}
                    %143 = arith.muli %134, %c512_i64 : i64
                    %144 = ascscalar.memref_address %alloc_7 : memref<4718592xui8>
                    %145 = arith.divui %143, %c8_i64 : i64
                    %146 = arith.addi %144, %145 : i64
                    %147 = ascscalar.bind_gm_ptr %146 : memref<1xui8>
                    %148 = ascscalar.bitcast_memref %alloc_12 byte_offset = %c0_i64 : memref<17408xui8, 1> -> memref<?xui8, 1>
                    ascvec.copy_ub2gm %147[%c0], %148[%c0], %c32_i32, %c512_i32, %c512_i64, %c544_i32 l2_cache_ctl %c0_i32 : memref<1xui8>, memref<?xui8, 1>
                    %149 = arith.muli %134, %c32_i64 : i64
                    %150 = ascscalar.memref_address %alloc_8 : memref<294912xui8>
                    %151 = arith.divui %149, %c8_i64 : i64
                    %152 = arith.addi %150, %151 : i64
                    %153 = ascscalar.bind_gm_ptr %152 : memref<1xui8>
                    %154 = ascscalar.memref_address %alloc_12 : memref<17408xui8, 1>
                    %155 = arith.addi %154, %c512_i64 : i64
                    %156 = ascscalar.bind_local_ptr %155 : memref<1xui8, 1>
                    ascvec.copy_ub2gm %153[%c0], %156[%c0], %c32_i32, %c32_i32, %c32_i64, %c544_i32 l2_cache_ctl %c0_i32 : memref<1xui8>, memref<1xui8, 1>
                    ascvec.sync_notify {dst_pipe = 5 : i32, event_id = 2 : i32, src_pipe = 6 : i32}
                  }
                  %126 = arith.cmpi eq, %124, %c1_i64 : i64
                  scf.if %126 {
                    %129 = arith.muli %8, %c4_i64 : i64
                    %130 = arith.remsi %arg41, %c4_i64 : i64
                    %131 = arith.addi %129, %130 : i64
                    %132 = arith.muli %131, %c512_i64 : i64
                    %133 = arith.muli %7, %c256_i64 : i64
                    %134 = arith.addi %132, %133 : i64
                    %135 = arith.muli %arg41, %c64_i64 : i64
                    %136 = arith.muli %7, %c32_i64 : i64
                    %137 = arith.addi %135, %136 : i64
                    %138 = memref.load %alloc_4[%25] : memref<1xi32>
                    %139 = arith.extsi %138 : i32 to i64
                    %140 = memref.load %alloc_4[%29] : memref<1xi32>
                    %141 = arith.extsi %140 : i32 to i64
                    %142 = arith.subi %141, %139 : i64
                    ascvec.sync_wait {dst_pipe = 5 : i32, event_id = 3 : i32, src_pipe = 6 : i32}
                    scf.for %arg42 = %c0_i64 to %c32_i64 step %c1_i64  : i64 {
                      %157 = arith.addi %137, %arg42 : i64
                      %158 = arith.cmpi slt, %157, %106 : i64
                      %159 = arith.subi %106, %c1_i64 : i64
                      %160 = arith.select %158, %157, %159 : i64
                      %161 = arith.index_cast %160 : i64 to index
                      %162 = arith.muli %104, %c2048 : index
                      %163 = arith.addi %162, %161 : index
                      %164 = memref.load %alloc_0[%163] : memref<1xi32>
                      %165 = arith.extsi %164 : i32 to i64
                      %166 = arith.muli %165, %c8_i64 : i64
                      %167 = arith.subi %142, %166 : i64
                      %168 = arith.minsi %167, %c8_i64 : i64
                      %169 = arith.cmpi eq, %168, %c8_i64 : i64
                      scf.if %169 {
                        %170 = arith.addi %139, %166 : i64
                        %171 = cannir.extract_layout %arg1 : !cannir.tensor<<gm, ui8>, <<(-1, 1, 64)>, <(-1, 64, 1)>>, <<(-1, 1, 64)>, <(-1, 64, 1)>>, identity>
                        %172 = cannir.extract_stride %171 : <<(-1, 1, 64)>, <(-1, 64, 1)>> -> <(-1, 64, 1)>
                        %173 = cannir.extract_dim %172[0] : !cannir.stride<(-1, 64, 1)>
                        %174 = arith.muli %170, %173 : i64
                        %175 = arith.addi %arg2, %174 : i64
                        %176 = ascscalar.bind_gm_ptr %175 : memref<512xui8>
                        %177 = cannir.extract_layout %arg3 : !cannir.tensor<<gm, ui8>, <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>>, <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>>, identity>
                        %178 = cannir.extract_stride %177 : <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>> -> <(-1, 4, 2, 1)>
                        %179 = cannir.extract_dim %178[0] : !cannir.stride<(-1, 4, 2, 1)>
                        %180 = arith.muli %170, %179 : i64
                        %181 = arith.addi %arg4, %180 : i64
                        %182 = ascscalar.bind_gm_ptr %181 : memref<32xui8>
                        %183 = arith.muli %arg42, %c544_i64 overflow<nsw> : i64
                        "ascvec.copy_gm2ub_plan"(%alloc_13, %176, %c0_i32, %c0_i32, %183, %c0_i64, %c8_i64, %c64_i64, %173, %c64_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = ui8, ub_base = 50176 : i64, ub_capacity = 17408 : i64}> : (memref<17408xui8, 1>, memref<512xui8>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                        %184 = arith.addi %183, %c512_i64 overflow<nsw> : i64
                        "ascvec.copy_gm2ub_plan"(%alloc_13, %182, %c0_i32, %c0_i32, %184, %c0_i64, %c8_i64, %c4_i64, %179, %c4_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = ui8, ub_base = 50176 : i64, ub_capacity = 17408 : i64}> : (memref<17408xui8, 1>, memref<32xui8>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                      } else {
                        %170 = arith.cmpi eq, %168, %c1_i64 : i64
                        scf.if %170 {
                          %177 = arith.addi %139, %166 : i64
                          %178 = cannir.extract_layout %arg1 : !cannir.tensor<<gm, ui8>, <<(-1, 1, 64)>, <(-1, 64, 1)>>, <<(-1, 1, 64)>, <(-1, 64, 1)>>, identity>
                          %179 = cannir.extract_stride %178 : <<(-1, 1, 64)>, <(-1, 64, 1)>> -> <(-1, 64, 1)>
                          %180 = cannir.extract_dim %179[0] : !cannir.stride<(-1, 64, 1)>
                          %181 = arith.muli %177, %180 : i64
                          %182 = arith.addi %arg2, %181 : i64
                          %183 = ascscalar.bind_gm_ptr %182 : memref<64xui8>
                          %184 = cannir.extract_layout %arg3 : !cannir.tensor<<gm, ui8>, <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>>, <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>>, identity>
                          %185 = cannir.extract_stride %184 : <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>> -> <(-1, 4, 2, 1)>
                          %186 = cannir.extract_dim %185[0] : !cannir.stride<(-1, 4, 2, 1)>
                          %187 = arith.muli %177, %186 : i64
                          %188 = arith.addi %arg4, %187 : i64
                          %189 = ascscalar.bind_gm_ptr %188 : memref<4xui8>
                          %190 = arith.muli %arg42, %c4352_i64 overflow<nsw> : i64
                          %191 = ascscalar.memref_address %alloc_13 : memref<17408xui8, 1>
                          %192 = arith.divui %190, %c8_i64 : i64
                          %193 = arith.addi %191, %192 : i64
                          %194 = ascscalar.bind_local_ptr %193 : memref<1xui8, 1>
                          ascvec.copy_gm2ub_nddma %194[%c0], %183[%c0], %c64_i32, %c1_i32, %c1_i32, %c1_i32, %c1_i32, %c1_i64, %180, %c0_i64, %c0_i64, %c0_i64, %c1_i64, %c64_i64, %c0_i64, %c0_i64, %c0_i64 l2_cache_ctl %c0_i32 left_padding(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) right_padding(%c0_i32, %c7_i32, %c0_i32, %c0_i32, %c0_i32) pad_value %c0_i32 : i32 {nearest = true} : memref<1xui8, 1>, memref<64xui8>
                          %195 = arith.muli %arg42, %c544_i64 overflow<nsw> : i64
                          %196 = arith.addi %195, %c512_i64 overflow<nsw> : i64
                          %197 = arith.muli %196, %c8_i64 overflow<nsw> : i64
                          %198 = arith.divui %197, %c8_i64 : i64
                          %199 = arith.addi %191, %198 : i64
                          %200 = ascscalar.bind_local_ptr %199 : memref<1xui8, 1>
                          ascvec.copy_gm2ub_nddma %200[%c0], %189[%c0], %c4_i32, %c1_i32, %c1_i32, %c1_i32, %c1_i32, %c1_i64, %186, %c0_i64, %c0_i64, %c0_i64, %c1_i64, %c4_i64, %c0_i64, %c0_i64, %c0_i64 l2_cache_ctl %c0_i32 left_padding(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) right_padding(%c0_i32, %c7_i32, %c0_i32, %c0_i32, %c0_i32) pad_value %c0_i32 : i32 {nearest = true} : memref<1xui8, 1>, memref<4xui8>
                        }
                        %171 = arith.cmpi eq, %168, %c2_i64 : i64
                        scf.if %171 {
                          %177 = arith.addi %139, %166 : i64
                          %178 = cannir.extract_layout %arg1 : !cannir.tensor<<gm, ui8>, <<(-1, 1, 64)>, <(-1, 64, 1)>>, <<(-1, 1, 64)>, <(-1, 64, 1)>>, identity>
                          %179 = cannir.extract_stride %178 : <<(-1, 1, 64)>, <(-1, 64, 1)>> -> <(-1, 64, 1)>
                          %180 = cannir.extract_dim %179[0] : !cannir.stride<(-1, 64, 1)>
                          %181 = arith.muli %177, %180 : i64
                          %182 = arith.addi %arg2, %181 : i64
                          %183 = ascscalar.bind_gm_ptr %182 : memref<128xui8>
                          %184 = cannir.extract_layout %arg3 : !cannir.tensor<<gm, ui8>, <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>>, <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>>, identity>
                          %185 = cannir.extract_stride %184 : <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>> -> <(-1, 4, 2, 1)>
                          %186 = cannir.extract_dim %185[0] : !cannir.stride<(-1, 4, 2, 1)>
                          %187 = arith.muli %177, %186 : i64
                          %188 = arith.addi %arg4, %187 : i64
                          %189 = ascscalar.bind_gm_ptr %188 : memref<8xui8>
                          %190 = arith.muli %arg42, %c4352_i64 overflow<nsw> : i64
                          %191 = ascscalar.memref_address %alloc_13 : memref<17408xui8, 1>
                          %192 = arith.divui %190, %c8_i64 : i64
                          %193 = arith.addi %191, %192 : i64
                          %194 = ascscalar.bind_local_ptr %193 : memref<1xui8, 1>
                          ascvec.copy_gm2ub_nddma %194[%c0], %183[%c0], %c64_i32, %c2_i32, %c1_i32, %c1_i32, %c1_i32, %c1_i64, %180, %c0_i64, %c0_i64, %c0_i64, %c1_i64, %c64_i64, %c0_i64, %c0_i64, %c0_i64 l2_cache_ctl %c0_i32 left_padding(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) right_padding(%c0_i32, %c6_i32, %c0_i32, %c0_i32, %c0_i32) pad_value %c0_i32 : i32 {nearest = true} : memref<1xui8, 1>, memref<128xui8>
                          %195 = arith.muli %arg42, %c544_i64 overflow<nsw> : i64
                          %196 = arith.addi %195, %c512_i64 overflow<nsw> : i64
                          %197 = arith.muli %196, %c8_i64 overflow<nsw> : i64
                          %198 = arith.divui %197, %c8_i64 : i64
                          %199 = arith.addi %191, %198 : i64
                          %200 = ascscalar.bind_local_ptr %199 : memref<1xui8, 1>
                          ascvec.copy_gm2ub_nddma %200[%c0], %189[%c0], %c4_i32, %c2_i32, %c1_i32, %c1_i32, %c1_i32, %c1_i64, %186, %c0_i64, %c0_i64, %c0_i64, %c1_i64, %c4_i64, %c0_i64, %c0_i64, %c0_i64 l2_cache_ctl %c0_i32 left_padding(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) right_padding(%c0_i32, %c6_i32, %c0_i32, %c0_i32, %c0_i32) pad_value %c0_i32 : i32 {nearest = true} : memref<1xui8, 1>, memref<8xui8>
                        }
                        %172 = arith.cmpi eq, %168, %c3_i64 : i64
                        scf.if %172 {
                          %177 = arith.addi %139, %166 : i64
                          %178 = cannir.extract_layout %arg1 : !cannir.tensor<<gm, ui8>, <<(-1, 1, 64)>, <(-1, 64, 1)>>, <<(-1, 1, 64)>, <(-1, 64, 1)>>, identity>
                          %179 = cannir.extract_stride %178 : <<(-1, 1, 64)>, <(-1, 64, 1)>> -> <(-1, 64, 1)>
                          %180 = cannir.extract_dim %179[0] : !cannir.stride<(-1, 64, 1)>
                          %181 = arith.muli %177, %180 : i64
                          %182 = arith.addi %arg2, %181 : i64
                          %183 = ascscalar.bind_gm_ptr %182 : memref<192xui8>
                          %184 = cannir.extract_layout %arg3 : !cannir.tensor<<gm, ui8>, <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>>, <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>>, identity>
                          %185 = cannir.extract_stride %184 : <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>> -> <(-1, 4, 2, 1)>
                          %186 = cannir.extract_dim %185[0] : !cannir.stride<(-1, 4, 2, 1)>
                          %187 = arith.muli %177, %186 : i64
                          %188 = arith.addi %arg4, %187 : i64
                          %189 = ascscalar.bind_gm_ptr %188 : memref<12xui8>
                          %190 = arith.muli %arg42, %c4352_i64 overflow<nsw> : i64
                          %191 = ascscalar.memref_address %alloc_13 : memref<17408xui8, 1>
                          %192 = arith.divui %190, %c8_i64 : i64
                          %193 = arith.addi %191, %192 : i64
                          %194 = ascscalar.bind_local_ptr %193 : memref<1xui8, 1>
                          ascvec.copy_gm2ub_nddma %194[%c0], %183[%c0], %c64_i32, %c3_i32, %c1_i32, %c1_i32, %c1_i32, %c1_i64, %180, %c0_i64, %c0_i64, %c0_i64, %c1_i64, %c64_i64, %c0_i64, %c0_i64, %c0_i64 l2_cache_ctl %c0_i32 left_padding(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) right_padding(%c0_i32, %c5_i32, %c0_i32, %c0_i32, %c0_i32) pad_value %c0_i32 : i32 {nearest = true} : memref<1xui8, 1>, memref<192xui8>
                          %195 = arith.muli %arg42, %c544_i64 overflow<nsw> : i64
                          %196 = arith.addi %195, %c512_i64 overflow<nsw> : i64
                          %197 = arith.muli %196, %c8_i64 overflow<nsw> : i64
                          %198 = arith.divui %197, %c8_i64 : i64
                          %199 = arith.addi %191, %198 : i64
                          %200 = ascscalar.bind_local_ptr %199 : memref<1xui8, 1>
                          ascvec.copy_gm2ub_nddma %200[%c0], %189[%c0], %c4_i32, %c3_i32, %c1_i32, %c1_i32, %c1_i32, %c1_i64, %186, %c0_i64, %c0_i64, %c0_i64, %c1_i64, %c4_i64, %c0_i64, %c0_i64, %c0_i64 l2_cache_ctl %c0_i32 left_padding(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) right_padding(%c0_i32, %c5_i32, %c0_i32, %c0_i32, %c0_i32) pad_value %c0_i32 : i32 {nearest = true} : memref<1xui8, 1>, memref<12xui8>
                        }
                        %173 = arith.cmpi eq, %168, %c4_i64 : i64
                        scf.if %173 {
                          %177 = arith.addi %139, %166 : i64
                          %178 = cannir.extract_layout %arg1 : !cannir.tensor<<gm, ui8>, <<(-1, 1, 64)>, <(-1, 64, 1)>>, <<(-1, 1, 64)>, <(-1, 64, 1)>>, identity>
                          %179 = cannir.extract_stride %178 : <<(-1, 1, 64)>, <(-1, 64, 1)>> -> <(-1, 64, 1)>
                          %180 = cannir.extract_dim %179[0] : !cannir.stride<(-1, 64, 1)>
                          %181 = arith.muli %177, %180 : i64
                          %182 = arith.addi %arg2, %181 : i64
                          %183 = ascscalar.bind_gm_ptr %182 : memref<256xui8>
                          %184 = cannir.extract_layout %arg3 : !cannir.tensor<<gm, ui8>, <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>>, <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>>, identity>
                          %185 = cannir.extract_stride %184 : <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>> -> <(-1, 4, 2, 1)>
                          %186 = cannir.extract_dim %185[0] : !cannir.stride<(-1, 4, 2, 1)>
                          %187 = arith.muli %177, %186 : i64
                          %188 = arith.addi %arg4, %187 : i64
                          %189 = ascscalar.bind_gm_ptr %188 : memref<16xui8>
                          %190 = arith.muli %arg42, %c4352_i64 overflow<nsw> : i64
                          %191 = ascscalar.memref_address %alloc_13 : memref<17408xui8, 1>
                          %192 = arith.divui %190, %c8_i64 : i64
                          %193 = arith.addi %191, %192 : i64
                          %194 = ascscalar.bind_local_ptr %193 : memref<1xui8, 1>
                          ascvec.copy_gm2ub_nddma %194[%c0], %183[%c0], %c64_i32, %c4_i32, %c1_i32, %c1_i32, %c1_i32, %c1_i64, %180, %c0_i64, %c0_i64, %c0_i64, %c1_i64, %c64_i64, %c0_i64, %c0_i64, %c0_i64 l2_cache_ctl %c0_i32 left_padding(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) right_padding(%c0_i32, %c4_i32, %c0_i32, %c0_i32, %c0_i32) pad_value %c0_i32 : i32 {nearest = true} : memref<1xui8, 1>, memref<256xui8>
                          %195 = arith.muli %arg42, %c544_i64 overflow<nsw> : i64
                          %196 = arith.addi %195, %c512_i64 overflow<nsw> : i64
                          %197 = arith.muli %196, %c8_i64 overflow<nsw> : i64
                          %198 = arith.divui %197, %c8_i64 : i64
                          %199 = arith.addi %191, %198 : i64
                          %200 = ascscalar.bind_local_ptr %199 : memref<1xui8, 1>
                          ascvec.copy_gm2ub_nddma %200[%c0], %189[%c0], %c4_i32, %c4_i32, %c1_i32, %c1_i32, %c1_i32, %c1_i64, %186, %c0_i64, %c0_i64, %c0_i64, %c1_i64, %c4_i64, %c0_i64, %c0_i64, %c0_i64 l2_cache_ctl %c0_i32 left_padding(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) right_padding(%c0_i32, %c4_i32, %c0_i32, %c0_i32, %c0_i32) pad_value %c0_i32 : i32 {nearest = true} : memref<1xui8, 1>, memref<16xui8>
                        }
                        %174 = arith.cmpi eq, %168, %c5_i64 : i64
                        scf.if %174 {
                          %177 = arith.addi %139, %166 : i64
                          %178 = cannir.extract_layout %arg1 : !cannir.tensor<<gm, ui8>, <<(-1, 1, 64)>, <(-1, 64, 1)>>, <<(-1, 1, 64)>, <(-1, 64, 1)>>, identity>
                          %179 = cannir.extract_stride %178 : <<(-1, 1, 64)>, <(-1, 64, 1)>> -> <(-1, 64, 1)>
                          %180 = cannir.extract_dim %179[0] : !cannir.stride<(-1, 64, 1)>
                          %181 = arith.muli %177, %180 : i64
                          %182 = arith.addi %arg2, %181 : i64
                          %183 = ascscalar.bind_gm_ptr %182 : memref<320xui8>
                          %184 = cannir.extract_layout %arg3 : !cannir.tensor<<gm, ui8>, <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>>, <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>>, identity>
                          %185 = cannir.extract_stride %184 : <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>> -> <(-1, 4, 2, 1)>
                          %186 = cannir.extract_dim %185[0] : !cannir.stride<(-1, 4, 2, 1)>
                          %187 = arith.muli %177, %186 : i64
                          %188 = arith.addi %arg4, %187 : i64
                          %189 = ascscalar.bind_gm_ptr %188 : memref<20xui8>
                          %190 = arith.muli %arg42, %c4352_i64 overflow<nsw> : i64
                          %191 = ascscalar.memref_address %alloc_13 : memref<17408xui8, 1>
                          %192 = arith.divui %190, %c8_i64 : i64
                          %193 = arith.addi %191, %192 : i64
                          %194 = ascscalar.bind_local_ptr %193 : memref<1xui8, 1>
                          ascvec.copy_gm2ub_nddma %194[%c0], %183[%c0], %c64_i32, %c5_i32, %c1_i32, %c1_i32, %c1_i32, %c1_i64, %180, %c0_i64, %c0_i64, %c0_i64, %c1_i64, %c64_i64, %c0_i64, %c0_i64, %c0_i64 l2_cache_ctl %c0_i32 left_padding(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) right_padding(%c0_i32, %c3_i32, %c0_i32, %c0_i32, %c0_i32) pad_value %c0_i32 : i32 {nearest = true} : memref<1xui8, 1>, memref<320xui8>
                          %195 = arith.muli %arg42, %c544_i64 overflow<nsw> : i64
                          %196 = arith.addi %195, %c512_i64 overflow<nsw> : i64
                          %197 = arith.muli %196, %c8_i64 overflow<nsw> : i64
                          %198 = arith.divui %197, %c8_i64 : i64
                          %199 = arith.addi %191, %198 : i64
                          %200 = ascscalar.bind_local_ptr %199 : memref<1xui8, 1>
                          ascvec.copy_gm2ub_nddma %200[%c0], %189[%c0], %c4_i32, %c5_i32, %c1_i32, %c1_i32, %c1_i32, %c1_i64, %186, %c0_i64, %c0_i64, %c0_i64, %c1_i64, %c4_i64, %c0_i64, %c0_i64, %c0_i64 l2_cache_ctl %c0_i32 left_padding(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) right_padding(%c0_i32, %c3_i32, %c0_i32, %c0_i32, %c0_i32) pad_value %c0_i32 : i32 {nearest = true} : memref<1xui8, 1>, memref<20xui8>
                        }
                        %175 = arith.cmpi eq, %168, %c6_i64 : i64
                        scf.if %175 {
                          %177 = arith.addi %139, %166 : i64
                          %178 = cannir.extract_layout %arg1 : !cannir.tensor<<gm, ui8>, <<(-1, 1, 64)>, <(-1, 64, 1)>>, <<(-1, 1, 64)>, <(-1, 64, 1)>>, identity>
                          %179 = cannir.extract_stride %178 : <<(-1, 1, 64)>, <(-1, 64, 1)>> -> <(-1, 64, 1)>
                          %180 = cannir.extract_dim %179[0] : !cannir.stride<(-1, 64, 1)>
                          %181 = arith.muli %177, %180 : i64
                          %182 = arith.addi %arg2, %181 : i64
                          %183 = ascscalar.bind_gm_ptr %182 : memref<384xui8>
                          %184 = cannir.extract_layout %arg3 : !cannir.tensor<<gm, ui8>, <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>>, <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>>, identity>
                          %185 = cannir.extract_stride %184 : <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>> -> <(-1, 4, 2, 1)>
                          %186 = cannir.extract_dim %185[0] : !cannir.stride<(-1, 4, 2, 1)>
                          %187 = arith.muli %177, %186 : i64
                          %188 = arith.addi %arg4, %187 : i64
                          %189 = ascscalar.bind_gm_ptr %188 : memref<24xui8>
                          %190 = arith.muli %arg42, %c4352_i64 overflow<nsw> : i64
                          %191 = ascscalar.memref_address %alloc_13 : memref<17408xui8, 1>
                          %192 = arith.divui %190, %c8_i64 : i64
                          %193 = arith.addi %191, %192 : i64
                          %194 = ascscalar.bind_local_ptr %193 : memref<1xui8, 1>
                          ascvec.copy_gm2ub_nddma %194[%c0], %183[%c0], %c64_i32, %c6_i32, %c1_i32, %c1_i32, %c1_i32, %c1_i64, %180, %c0_i64, %c0_i64, %c0_i64, %c1_i64, %c64_i64, %c0_i64, %c0_i64, %c0_i64 l2_cache_ctl %c0_i32 left_padding(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) right_padding(%c0_i32, %c2_i32, %c0_i32, %c0_i32, %c0_i32) pad_value %c0_i32 : i32 {nearest = true} : memref<1xui8, 1>, memref<384xui8>
                          %195 = arith.muli %arg42, %c544_i64 overflow<nsw> : i64
                          %196 = arith.addi %195, %c512_i64 overflow<nsw> : i64
                          %197 = arith.muli %196, %c8_i64 overflow<nsw> : i64
                          %198 = arith.divui %197, %c8_i64 : i64
                          %199 = arith.addi %191, %198 : i64
                          %200 = ascscalar.bind_local_ptr %199 : memref<1xui8, 1>
                          ascvec.copy_gm2ub_nddma %200[%c0], %189[%c0], %c4_i32, %c6_i32, %c1_i32, %c1_i32, %c1_i32, %c1_i64, %186, %c0_i64, %c0_i64, %c0_i64, %c1_i64, %c4_i64, %c0_i64, %c0_i64, %c0_i64 l2_cache_ctl %c0_i32 left_padding(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) right_padding(%c0_i32, %c2_i32, %c0_i32, %c0_i32, %c0_i32) pad_value %c0_i32 : i32 {nearest = true} : memref<1xui8, 1>, memref<24xui8>
                        }
                        %176 = arith.cmpi eq, %168, %c7_i64 : i64
                        scf.if %176 {
                          %177 = arith.addi %139, %166 : i64
                          %178 = cannir.extract_layout %arg1 : !cannir.tensor<<gm, ui8>, <<(-1, 1, 64)>, <(-1, 64, 1)>>, <<(-1, 1, 64)>, <(-1, 64, 1)>>, identity>
                          %179 = cannir.extract_stride %178 : <<(-1, 1, 64)>, <(-1, 64, 1)>> -> <(-1, 64, 1)>
                          %180 = cannir.extract_dim %179[0] : !cannir.stride<(-1, 64, 1)>
                          %181 = arith.muli %177, %180 : i64
                          %182 = arith.addi %arg2, %181 : i64
                          %183 = ascscalar.bind_gm_ptr %182 : memref<448xui8>
                          %184 = cannir.extract_layout %arg3 : !cannir.tensor<<gm, ui8>, <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>>, <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>>, identity>
                          %185 = cannir.extract_stride %184 : <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>> -> <(-1, 4, 2, 1)>
                          %186 = cannir.extract_dim %185[0] : !cannir.stride<(-1, 4, 2, 1)>
                          %187 = arith.muli %177, %186 : i64
                          %188 = arith.addi %arg4, %187 : i64
                          %189 = ascscalar.bind_gm_ptr %188 : memref<28xui8>
                          %190 = arith.muli %arg42, %c4352_i64 overflow<nsw> : i64
                          %191 = ascscalar.memref_address %alloc_13 : memref<17408xui8, 1>
                          %192 = arith.divui %190, %c8_i64 : i64
                          %193 = arith.addi %191, %192 : i64
                          %194 = ascscalar.bind_local_ptr %193 : memref<1xui8, 1>
                          ascvec.copy_gm2ub_nddma %194[%c0], %183[%c0], %c64_i32, %c7_i32, %c1_i32, %c1_i32, %c1_i32, %c1_i64, %180, %c0_i64, %c0_i64, %c0_i64, %c1_i64, %c64_i64, %c0_i64, %c0_i64, %c0_i64 l2_cache_ctl %c0_i32 left_padding(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) right_padding(%c0_i32, %c1_i32, %c0_i32, %c0_i32, %c0_i32) pad_value %c0_i32 : i32 {nearest = true} : memref<1xui8, 1>, memref<448xui8>
                          %195 = arith.muli %arg42, %c544_i64 overflow<nsw> : i64
                          %196 = arith.addi %195, %c512_i64 overflow<nsw> : i64
                          %197 = arith.muli %196, %c8_i64 overflow<nsw> : i64
                          %198 = arith.divui %197, %c8_i64 : i64
                          %199 = arith.addi %191, %198 : i64
                          %200 = ascscalar.bind_local_ptr %199 : memref<1xui8, 1>
                          ascvec.copy_gm2ub_nddma %200[%c0], %189[%c0], %c4_i32, %c7_i32, %c1_i32, %c1_i32, %c1_i32, %c1_i64, %186, %c0_i64, %c0_i64, %c0_i64, %c1_i64, %c4_i64, %c0_i64, %c0_i64, %c0_i64 l2_cache_ctl %c0_i32 left_padding(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) right_padding(%c0_i32, %c1_i32, %c0_i32, %c0_i32, %c0_i32) pad_value %c0_i32 : i32 {nearest = true} : memref<1xui8, 1>, memref<28xui8>
                        }
                      }
                    }
                    ascvec.sync_notify {dst_pipe = 6 : i32, event_id = 2 : i32, src_pipe = 5 : i32}
                    ascvec.sync_wait {dst_pipe = 6 : i32, event_id = 2 : i32, src_pipe = 5 : i32}
                    %143 = arith.muli %134, %c512_i64 : i64
                    %144 = ascscalar.memref_address %alloc_7 : memref<4718592xui8>
                    %145 = arith.divui %143, %c8_i64 : i64
                    %146 = arith.addi %144, %145 : i64
                    %147 = ascscalar.bind_gm_ptr %146 : memref<1xui8>
                    %148 = ascscalar.bitcast_memref %alloc_13 byte_offset = %c0_i64 : memref<17408xui8, 1> -> memref<?xui8, 1>
                    ascvec.copy_ub2gm %147[%c0], %148[%c0], %c32_i32, %c512_i32, %c512_i64, %c544_i32 l2_cache_ctl %c0_i32 : memref<1xui8>, memref<?xui8, 1>
                    %149 = arith.muli %134, %c32_i64 : i64
                    %150 = ascscalar.memref_address %alloc_8 : memref<294912xui8>
                    %151 = arith.divui %149, %c8_i64 : i64
                    %152 = arith.addi %150, %151 : i64
                    %153 = ascscalar.bind_gm_ptr %152 : memref<1xui8>
                    %154 = ascscalar.memref_address %alloc_13 : memref<17408xui8, 1>
                    %155 = arith.addi %154, %c512_i64 : i64
                    %156 = ascscalar.bind_local_ptr %155 : memref<1xui8, 1>
                    ascvec.copy_ub2gm %153[%c0], %156[%c0], %c32_i32, %c32_i32, %c32_i64, %c544_i32 l2_cache_ctl %c0_i32 : memref<1xui8>, memref<1xui8, 1>
                    ascvec.sync_notify {dst_pipe = 5 : i32, event_id = 3 : i32, src_pipe = 6 : i32}
                  }
                  %127 = arith.remsi %arg41, %c4_i64 : i64
                  %128 = arith.addi %127, %c6_i64 overflow<nsw> : i64
                  ascvec.sync_intra_arrive mte3, %128
                } {cannir.loop_unroll = 1 : i64}
                scf.yield %123 : i64
              } else {
                scf.yield %c0_i64 : i64
              }
              scf.yield %122 : i64
            } else {
              scf.yield %c0_i64 : i64
            }
            scf.if %68 {
              ascvec.sync_wait {dst_pipe = 1 : i32, event_id = 0 : i32, src_pipe = 2 : i32}
              %103 = memref.load %alloc_25[%c0] : memref<8xi32, 1>
              %104 = arith.cmpi sgt, %103, %c0_i32 : i32
              scf.if %104 {
                %105 = arith.muli %97, %c512_i64 : i64
                %106 = memref.load %alloc_10[%c288] : memref<1024xi32>
                %107 = arith.cmpi ne, %106, %c0_i32 : i32
                %108 = arith.select %107, %c1_i64, %arg31 : i64
                %109 = arith.remsi %105, %c128_i64 : i64
                %110 = arith.cmpi ne, %109, %c0_i64 : i64
                %111 = arith.cmpi slt, %109, %c0_i64 : i64
                %112 = arith.andi %110, %111 : i1
                %113 = arith.divsi %105, %c128_i64 : i64
                %114 = arith.extui %112 : i1 to i64
                %115 = arith.subi %113, %114 : i64
                ascvec.get_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
                ascvec.vf(%alloc_17, %115 : memref<16384xui16, 1>, i64) {
                ^bb0(%arg41: memref<16384xui16, 1>, %arg42: i64):
                  %c1_i64_36 = arith.constant 1 : i64
                  %c64_i64_37 = arith.constant 64 : i64
                  %c128_i64_38 = arith.constant 128 : i64
                  %c0_i16 = arith.constant 0 : i16
                  %c128_i32 = arith.constant 128 : i32
                  %mask, %remaining_out = ascvec.update_mask %c128_i32 elem_bits 16 : !ascvec.mask
                  %145 = ascvec.duplicate_scalar %c0_i16, %mask : i16 -> <ui16>
                  scf.for %arg43 = %arg42 to %c64_i64_37 step %c1_i64_36  : i64 {
                    %146 = arith.muli %arg43, %c128_i64_38 : i64
                    %147 = arith.index_cast %146 : i64 to index
                    ascvec.storealign %arg41[%147], %145, %mask : memref<16384xui16, 1>, index, <ui16>
                  } {cannir.loop_unroll = 1 : i64}
                  ascvec.mem_bar vst_vld
                } {cannir.vf_mode = "simd"}
                %116 = arith.maxsi %105, %c8192_i64 : i64
                %117 = arith.addi %116, %c255_i64 : i64
                %118 = arith.remsi %117, %c256_i64 : i64
                %119 = arith.cmpi ne, %118, %c0_i64 : i64
                %120 = arith.cmpi slt, %118, %c0_i64 : i64
                %121 = arith.andi %119, %120 : i1
                %122 = arith.divsi %117, %c256_i64 : i64
                %123 = arith.extui %121 : i1 to i64
                %124 = arith.subi %122, %123 : i64
                %125 = arith.addi %116, %c-8191_i64 : i64
                %126 = arith.trunci %125 : i64 to i16
                ascvec.vf(%116, %alloc_17, %124, %alloc_21, %126, %alloc_22 : i64, memref<16384xui16, 1>, i64, memref<256xui16, 1>, i16, memref<64xi32, 1>) {
                ^bb0(%arg41: i64, %arg42: memref<16384xui16, 1>, %arg43: i64, %arg44: memref<256xui16, 1>, %arg45: i16, %arg46: memref<64xi32, 1>):
                  %c128 = arith.constant 128 : index
                  %c128_i16 = arith.constant 128 : i16
                  %c-1_i16 = arith.constant -1 : i16
                  %c1_i32_36 = arith.constant 1 : i32
                  %c1_i64_37 = arith.constant 1 : i64
                  %c0_i64_38 = arith.constant 0 : i64
                  %c0_39 = arith.constant 0 : index
                  %c256_i64_40 = arith.constant 256 : i64
                  %c0_i16 = arith.constant 0 : i16
                  %c128_i32 = arith.constant 128 : i32
                  %mask, %remaining_out = ascvec.update_mask %c128_i32 elem_bits 16 : !ascvec.mask
                  %145 = ascvec.duplicate_scalar %c0_i16, %mask : i16 -> <ui16>
                  %146:2 = scf.for %arg47 = %c0_i64_38 to %arg43 step %c1_i64_37 iter_args(%arg48 = %145, %arg49 = %145) -> (!ascvec.vreg<ui16>, !ascvec.vreg<ui16>)  : i64 {
                    %147 = arith.muli %arg47, %c256_i64_40 : i64
                    %148 = arith.subi %arg41, %147 : i64
                    %149 = arith.trunci %148 : i64 to i32
                    %mask_45, %remaining_out_46 = ascvec.update_mask %149 elem_bits 8 : !ascvec.mask
                    %150 = arith.index_cast %147 : i64 to index
                    %result0, %result1 = ascvec.loadalign_deintlv %arg42[%150] {dintlv_width = "b8"} : memref<16384xui16, 1>, index -> <ui8>, <ui8>
                    %151 = ascvec.cumulative_histogram %arg48, %result1, %mask_45 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                    %152 = ascvec.cumulative_histogram %arg49, %result1, %mask_45 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                    scf.yield %151, %152 : !ascvec.vreg<ui16>, !ascvec.vreg<ui16>
                  } {cannir.loop_unroll = 1 : i64}
                  %c0_41 = arith.constant 0 : index
                  %c1_42 = arith.constant 1 : index
                  scf.for %arg47 = %c0_41 to %c1_42 step %c1_42 {
                    ascvec.storealign %arg44[%c0_39], %146#0, %mask : memref<256xui16, 1>, index, <ui16>
                    ascvec.storealign %arg44[%c128], %146#1, %mask : memref<256xui16, 1>, index, <ui16>
                  }
                  ascvec.mem_bar vst_vld
                  %c0_43 = arith.constant 0 : index
                  %c1_44 = arith.constant 1 : index
                  scf.for %arg47 = %c0_43 to %c1_44 step %c1_44 {
                    %mask_45, %remaining_out_46 = ascvec.update_mask %c128_i32 elem_bits 16 : !ascvec.mask
                    %mask_47, %remaining_out_48 = ascvec.update_mask %c1_i32_36 elem_bits 16 : !ascvec.mask
                    %147 = ascvec.duplicate_scalar %c-1_i16, %mask_45 : i16 -> <ui16>
                    %148 = ascvec.loadalign %arg44[%c0_39] : memref<256xui16, 1>, index -> <ui16>
                    %149 = ascvec.loadalign %arg44[%c128] : memref<256xui16, 1>, index -> <ui16>
                    %150 = ascvec.arange %c0_i16 : i16 -> <ui16>
                    %151 = ascvec.cmp_ge_scalar %148, %arg45, %mask_45 : <ui16>, i16 -> !ascvec.mask
                    %152 = ascvec.select %150, %147, %151 : <ui16>
                    %153 = ascvec.arange %c128_i16 : i16 -> <ui16>
                    %154 = ascvec.cmp_ge_scalar %149, %arg45, %mask_45 : <ui16>, i16 -> !ascvec.mask
                    %155 = ascvec.select %153, %147, %154 : <ui16>
                    %156 = ascvec.reduce_min %152, %mask_45 : <ui16>
                    %157 = ascvec.reduce_min %155, %mask_45 : <ui16>
                    %158 = ascvec.min %156, %157, %mask_47 : <ui16>
                    %159 = ascvec.unpack %158 {part = "lower"} : <ui16> -> <ui32>
                    %160 = ascvec.reinterpret %159 : <ui32> -> <i32>
                    %mask_49, %remaining_out_50 = ascvec.update_mask %c1_i32_36 : !ascvec.mask
                    ascvec.storealign %arg46[%c0_39], %160, %mask_49 : memref<64xi32, 1>, index, <i32>
                  }
                  ascvec.mem_bar vst_vld
                } {cannir.vf_mode = "simd"}
                ascvec.vf(%alloc_22, %alloc_21, %126, %alloc_20, %116, %alloc_17, %124 : memref<64xi32, 1>, memref<256xui16, 1>, i16, memref<24576xui16, 1>, i64, memref<16384xui16, 1>, i64) {
                ^bb0(%arg41: memref<64xi32, 1>, %arg42: memref<256xui16, 1>, %arg43: i16, %arg44: memref<24576xui16, 1>, %arg45: i64, %arg46: memref<16384xui16, 1>, %arg47: i64):
                  %c128 = arith.constant 128 : index
                  %c1_i64_36 = arith.constant 1 : i64
                  %c0_i64_37 = arith.constant 0 : i64
                  %c256_i64_38 = arith.constant 256 : i64
                  %c128_i16 = arith.constant 128 : i16
                  %c0_39 = arith.constant 0 : index
                  %c1_i16 = arith.constant 1 : i16
                  %c0_i16 = arith.constant 0 : i16
                  %c1_i32_40 = arith.constant 1 : i32
                  %c128_i32 = arith.constant 128 : i32
                  %mask, %remaining_out = ascvec.update_mask %c128_i32 elem_bits 16 : !ascvec.mask
                  %mask_41, %remaining_out_42 = ascvec.update_mask %c1_i32_40 elem_bits 16 : !ascvec.mask
                  %145 = ascvec.duplicate_scalar %c0_i16, %mask : i16 -> <ui16>
                  %146 = ascvec.duplicate_scalar %c1_i16, %mask : i16 -> <ui16>
                  %147 = ascvec.loadalign_brc %arg41[%c0_39] {brc_width = "b16"} : memref<64xi32, 1>, index -> <ui32>
                  %148 = ascvec.reinterpret %147 : <ui32> -> <ui16>
                  %149 = ascvec.loadalign_brc %arg41[%c0_39] {brc_width = "b8"} : memref<64xi32, 1>, index -> <ui32>
                  %150 = ascvec.reinterpret %149 : <ui32> -> <ui8>
                  %151 = ascvec.sub %148, %146, %mask : <ui16>
                  %152 = ascvec.cmp_eq_scalar %148, %c0_i16, %mask : <ui16>, i16 -> !ascvec.mask
                  %153 = ascvec.select %145, %151, %152 : <ui16>
                  %154 = ascvec.loadalign %arg42[%c0_39] : memref<256xui16, 1>, index -> <ui16>
                  %155 = ascvec.arange %c0_i16 : i16 -> <ui16>
                  %156 = ascvec.cmp_eq %155, %153, %mask : <ui16>, <ui16> -> !ascvec.mask
                  %157 = ascvec.select %154, %145, %156 : <ui16>
                  %158 = ascvec.reduce_max %157, %mask : <ui16>
                  %159 = ascvec.add %145, %158, %mask : <ui16>
                  %160 = ascvec.loadalign %arg42[%c128] : memref<256xui16, 1>, index -> <ui16>
                  %161 = ascvec.arange %c128_i16 : i16 -> <ui16>
                  %162 = ascvec.cmp_eq %161, %153, %mask : <ui16>, <ui16> -> !ascvec.mask
                  %163 = ascvec.select %160, %145, %162 : <ui16>
                  %164 = ascvec.reduce_max %163, %mask : <ui16>
                  %165 = ascvec.add %159, %164, %mask : <ui16>
                  %166 = ascvec.select %145, %165, %152 : <ui16>
                  %167 = ascvec.duplicate_scalar %arg43, %mask : i16 -> <ui16>
                  %168 = ascvec.sub %167, %166, %mask : <ui16>
                  ascvec.storealign %arg44[%c0_39], %168, %mask_41 : memref<24576xui16, 1>, index, <ui16>
                  %169:2 = scf.for %arg48 = %c0_i64_37 to %arg47 step %c1_i64_36 iter_args(%arg49 = %145, %arg50 = %145) -> (!ascvec.vreg<ui16>, !ascvec.vreg<ui16>)  : i64 {
                    %170 = arith.muli %arg48, %c256_i64_38 : i64
                    %171 = arith.subi %arg45, %170 : i64
                    %172 = arith.trunci %171 : i64 to i32
                    %mask_45, %remaining_out_46 = ascvec.update_mask %172 elem_bits 8 : !ascvec.mask
                    %173 = arith.index_cast %170 : i64 to index
                    %result0, %result1 = ascvec.loadalign_deintlv %arg46[%173] {dintlv_width = "b8"} : memref<16384xui16, 1>, index -> <ui8>, <ui8>
                    %174 = ascvec.cmp_eq %result1, %150, %mask_45 : <ui8>, <ui8> -> !ascvec.mask
                    %175 = ascvec.cumulative_histogram %arg49, %result0, %174 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                    %176 = ascvec.cumulative_histogram %arg50, %result0, %174 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                    scf.yield %175, %176 : !ascvec.vreg<ui16>, !ascvec.vreg<ui16>
                  } {cannir.loop_unroll = 1 : i64}
                  %c0_43 = arith.constant 0 : index
                  %c1_44 = arith.constant 1 : index
                  scf.for %arg48 = %c0_43 to %c1_44 step %c1_44 {
                    ascvec.storealign %arg42[%c0_39], %169#0, %mask : memref<256xui16, 1>, index, <ui16>
                    ascvec.storealign %arg42[%c128], %169#1, %mask : memref<256xui16, 1>, index, <ui16>
                  }
                  ascvec.mem_bar vst_vld
                } {cannir.vf_mode = "simd"}
                %127 = arith.addi %116, %c127_i64 : i64
                %128 = arith.remsi %127, %c128_i64 : i64
                %129 = arith.cmpi ne, %128, %c0_i64 : i64
                %130 = arith.cmpi slt, %128, %c0_i64 : i64
                %131 = arith.andi %129, %130 : i1
                %132 = arith.divsi %127, %c128_i64 : i64
                %133 = arith.extui %131 : i1 to i64
                %134 = arith.subi %132, %133 : i64
                ascvec.vf(%alloc_20, %alloc_21, %alloc_22, %116, %alloc_17, %134 : memref<24576xui16, 1>, memref<256xui16, 1>, memref<64xi32, 1>, i64, memref<16384xui16, 1>, i64) {
                ^bb0(%arg41: memref<24576xui16, 1>, %arg42: memref<256xui16, 1>, %arg43: memref<64xi32, 1>, %arg44: i64, %arg45: memref<16384xui16, 1>, %arg46: i64):
                  %c128 = arith.constant 128 : index
                  %c1_i64_36 = arith.constant 1 : i64
                  %c0_i64_37 = arith.constant 0 : i64
                  %c128_i64_38 = arith.constant 128 : i64
                  %c128_i16 = arith.constant 128 : i16
                  %c0_i16 = arith.constant 0 : i16
                  %c-1_i16 = arith.constant -1 : i16
                  %c0_39 = arith.constant 0 : index
                  %c1_i32_40 = arith.constant 1 : i32
                  %c128_i32 = arith.constant 128 : i32
                  %c0_41 = arith.constant 0 : index
                  %c1_42 = arith.constant 1 : index
                  scf.for %arg47 = %c0_41 to %c1_42 step %c1_42 {
                    %mask_45, %remaining_out_46 = ascvec.update_mask %c128_i32 elem_bits 16 : !ascvec.mask
                    %mask_47, %remaining_out_48 = ascvec.update_mask %c1_i32_40 elem_bits 16 : !ascvec.mask
                    %150 = ascvec.loadalign_brc %arg41[%c0_39] : memref<24576xui16, 1>, index -> <ui16>
                    %151 = ascvec.duplicate_scalar %c-1_i16, %mask_45 : i16 -> <ui16>
                    %152 = ascvec.loadalign %arg42[%c0_39] : memref<256xui16, 1>, index -> <ui16>
                    %153 = ascvec.loadalign %arg42[%c128] : memref<256xui16, 1>, index -> <ui16>
                    %154 = ascvec.arange %c0_i16 : i16 -> <ui16>
                    %155 = ascvec.cmp_ge %152, %150, %mask_45 : <ui16>, <ui16> -> !ascvec.mask
                    %156 = ascvec.select %154, %151, %155 : <ui16>
                    %157 = ascvec.arange %c128_i16 : i16 -> <ui16>
                    %158 = ascvec.cmp_ge %153, %150, %mask_45 : <ui16>, <ui16> -> !ascvec.mask
                    %159 = ascvec.select %157, %151, %158 : <ui16>
                    %160 = ascvec.reduce_min %156, %mask_45 : <ui16>
                    %161 = ascvec.reduce_min %159, %mask_45 : <ui16>
                    %162 = ascvec.min %160, %161, %mask_47 : <ui16>
                    ascvec.storealign %arg41[%c0_39], %162, %mask_47 : memref<24576xui16, 1>, index, <ui16>
                  }
                  ascvec.mem_bar vst_vld
                  %mask, %remaining_out = ascvec.update_mask %c128_i32 elem_bits 16 : !ascvec.mask
                  %145 = ascvec.loadalign_brc %arg43[%c0_39] {brc_width = "b16"} : memref<64xi32, 1>, index -> <ui32>
                  %146 = ascvec.reinterpret %145 : <ui32> -> <ui16>
                  %147 = ascvec.loadalign_brc %arg41[%c0_39] : memref<24576xui16, 1>, index -> <ui16>
                  %148 = ascvec.shl %146, 8, %mask : <ui16> -> <ui16>
                  %149 = ascvec.or %148, %147, %mask : <ui16>, <ui16> -> <ui16>
                  ascvec.store_unalign_begin %arg41 : memref<24576xui16, 1>
                  scf.for %arg47 = %c0_i64_37 to %arg46 step %c1_i64_36  : i64 {
                    %150 = arith.muli %arg47, %c128_i64_38 : i64
                    %151 = arith.subi %arg44, %150 : i64
                    %152 = arith.trunci %151 : i64 to i32
                    %mask_45, %remaining_out_46 = ascvec.update_mask %152 elem_bits 16 : !ascvec.mask
                    %153 = arith.index_cast %150 : i64 to index
                    %154 = ascvec.loadalign %arg45[%153] : memref<16384xui16, 1>, index -> <ui16>
                    %155 = arith.trunci %150 : i64 to i16
                    %156 = ascvec.arange %155 : i16 -> <ui16>
                    %157 = ascvec.cmp_gt %154, %149, %mask_45 : <ui16>, <ui16> -> !ascvec.mask
                    %158 = ascvec.squeeze_store_reg %156, %157 : <ui16>, !ascvec.mask -> <ui16>
                    ascvec.store_unalign %arg41[%c0_39], %158 : memref<24576xui16, 1>, <ui16>
                  } {cannir.loop_unroll = 1 : i64}
                  scf.for %arg47 = %c0_i64_37 to %arg46 step %c1_i64_36  : i64 {
                    %150 = arith.muli %arg47, %c128_i64_38 : i64
                    %151 = arith.subi %arg44, %150 : i64
                    %152 = arith.trunci %151 : i64 to i32
                    %mask_45, %remaining_out_46 = ascvec.update_mask %152 elem_bits 16 : !ascvec.mask
                    %153 = arith.index_cast %150 : i64 to index
                    %154 = ascvec.loadalign %arg45[%153] : memref<16384xui16, 1>, index -> <ui16>
                    %155 = arith.trunci %150 : i64 to i16
                    %156 = ascvec.arange %155 : i16 -> <ui16>
                    %157 = ascvec.cmp_eq %154, %149, %mask_45 : <ui16>, <ui16> -> !ascvec.mask
                    %158 = ascvec.squeeze_store_reg %156, %157 : <ui16>, !ascvec.mask -> <ui16>
                    ascvec.store_unalign %arg41[%c0_39], %158 : memref<24576xui16, 1>, <ui16>
                  } {cannir.loop_unroll = 1 : i64}
                  %c0_43 = arith.constant 0 : index
                  %c1_44 = arith.constant 1 : index
                  scf.for %arg47 = %c0_43 to %c1_44 step %c1_44 {
                    ascvec.store_unalign_post %arg41[%c0_39] : memref<24576xui16, 1>
                  }
                  ascvec.mem_bar vst_vld
                } {cannir.vf_mode = "simd"}
                ascvec.release_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
                ascvec.get_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
                ascvec.vf(%alloc_20, %103, %alloc_23 : memref<24576xui16, 1>, i32, memref<8192xi32, 1>) {
                ^bb0(%arg41: memref<24576xui16, 1>, %arg42: i32, %arg43: memref<8192xi32, 1>):
                  %c1_i32_36 = arith.constant 1 : i32
                  %c128_i32 = arith.constant 128 : i32
                  %c0_i32_37 = arith.constant 0 : i32
                  %c-1_i32 = arith.constant -1 : i32
                  %c8192_i32 = arith.constant 8192 : i32
                  %c64_i32_38 = arith.constant 64 : i32
                  scf.for %arg44 = %c0_i32_37 to %c128_i32 step %c1_i32_36  : i32 {
                    %145 = arith.muli %arg44, %c64_i32_38 overflow<nsw> : i32
                    %146 = arith.subi %c8192_i32, %145 overflow<nsw> : i32
                    %mask, %remaining_out = ascvec.update_mask %146 : !ascvec.mask
                    %147 = arith.index_cast %145 : i32 to index
                    %148 = ascvec.loadalign %arg41[%147] : memref<24576xui16, 1>, index -> <ui16>
                    %149 = ascvec.unpack %148 {part = "lower"} : <ui16> -> <ui32>
                    %150 = ascvec.arange %145 : i32 -> <ui32>
                    %151 = ascvec.cmp_lt_scalar %150, %arg42, %mask : <ui32>, i32 -> !ascvec.mask
                    %152 = ascvec.reinterpret %149 : <ui32> -> <i32>
                    %153 = ascvec.duplicate_scalar %c-1_i32, %mask : i32 -> <i32>
                    %154 = ascvec.select %152, %153, %151 : <i32>
                    ascvec.storealign %arg43[%147], %154, %mask : memref<8192xi32, 1>, index, <i32>
                  } {cannir.loop_unroll = 1 : i64}
                  ascvec.mem_bar vst_vld
                } {cannir.vf_mode = "simd"}
                ascvec.release_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
                %135 = arith.cmpi ne, %108, %c0_i64 : i64
                scf.if %135 {
                  ascvec.get_buf v, %c5_i64, 0 {cannir.buffer.generated_sync}
                  ascvec.vf(%alloc_20, %alloc_17, %103, %alloc_24 : memref<24576xui16, 1>, memref<16384xui16, 1>, i32, memref<8192xui16, 1>) {
                  ^bb0(%arg41: memref<24576xui16, 1>, %arg42: memref<16384xui16, 1>, %arg43: i32, %arg44: memref<8192xui16, 1>):
                    %c1_i32_36 = arith.constant 1 : i32
                    %c64_i32_37 = arith.constant 64 : i32
                    %c0_i32_38 = arith.constant 0 : i32
                    %c-128_i16 = arith.constant -128 : i16
                    %c-1_i16 = arith.constant -1 : i16
                    %c-32768_i16 = arith.constant -32768 : i16
                    %c0_39 = arith.constant 0 : index
                    %c8192_i32 = arith.constant 8192 : i32
                    %c128_i32 = arith.constant 128 : i32
                    scf.for %arg45 = %c0_i32_38 to %c64_i32_37 step %c1_i32_36  : i32 {
                      %150 = arith.muli %arg45, %c128_i32 overflow<nsw> : i32
                      %151 = arith.subi %c8192_i32, %150 overflow<nsw> : i32
                      %mask, %remaining_out = ascvec.update_mask %151 elem_bits 16 : !ascvec.mask
                      %152 = arith.index_cast %150 : i32 to index
                      %153 = ascvec.loadalign %arg41[%152] : memref<24576xui16, 1>, index -> <ui16>
                      %154 = ascvec.gather %arg42[%c0_39], %153, %mask : memref<16384xui16, 1>, <ui16>, !ascvec.mask -> <ui16>
                      %155 = ascvec.duplicate_scalar %c-32768_i16, %mask : i16 -> <ui16>
                      %156 = ascvec.and %154, %155, %mask : <ui16>, <ui16> -> <ui16>
                      %157 = ascvec.cmp_eq %156, %155, %mask : <ui16>, <ui16> -> !ascvec.mask
                      %158 = ascvec.xor %154, %155, %mask : <ui16>, <ui16> -> <ui16>
                      %159 = ascvec.duplicate_scalar %c-1_i16, %mask : i16 -> <ui16>
                      %160 = ascvec.xor %154, %159, %mask : <ui16>, <ui16> -> <ui16>
                      %161 = ascvec.select %158, %160, %157 : <ui16>
                      %162 = arith.trunci %150 : i32 to i16
                      %163 = ascvec.arange %162 : i16 -> <ui16>
                      %164 = arith.trunci %arg43 : i32 to i16
                      %165 = ascvec.cmp_lt_scalar %163, %164, %mask : <ui16>, i16 -> !ascvec.mask
                      %166 = ascvec.duplicate_scalar %c-128_i16, %mask : i16 -> <ui16>
                      %167 = ascvec.select %161, %166, %165 : <ui16>
                      ascvec.storealign %arg44[%152], %167, %mask : memref<8192xui16, 1>, index, <ui16>
                    } {cannir.loop_unroll = 1 : i64}
                    ascvec.mem_bar vst_vld
                  } {cannir.vf_mode = "simd"}
                  ascvec.release_buf v, %c5_i64, 0 {cannir.buffer.generated_sync}
                  %145 = arith.muli %65, %c131072_i64 : i64
                  %146 = arith.divui %145, %c8_i64 : i64
                  %147 = arith.addi %67, %146 : i64
                  %148 = ascscalar.bind_gm_ptr %147 : memref<1xui16>
                  %149 = ascscalar.bitcast_memref %alloc_24 byte_offset = %c0_i64 : memref<8192xui16, 1> -> memref<?xui16, 1>
                  ascvec.get_buf mte3, %c5_i64, 0 {cannir.buffer.generated_sync}
                  ascvec.copy_ub2gm %148[%c0], %149[%c0], %c1_i32, %c16384_i32, %c0_i64, %c16384_i32 l2_cache_ctl %c0_i32 : memref<1xui16>, memref<?xui16, 1>
                  ascvec.release_buf mte3, %c5_i64, 0 {cannir.buffer.generated_sync}
                }
                %136 = scf.if %64 -> (i32) {
                  scf.yield %c0_i32 : i32
                } else {
                  %145 = memref.load %alloc_6[%41] : memref<1xi32>
                  scf.yield %145 : i32
                }
                %137 = arith.muli %54, %c512_i64 : i64
                %138 = arith.trunci %137 : i64 to i32
                %139 = arith.muli %40, %c8192_i64 : i64
                ascvec.get_buf mte2, %c7_i64, 0 {cannir.buffer.generated_sync}
                "ascvec.copy_gm2ub_plan"(%alloc_26, %alloc_0, %c0_i32, %c0_i32, %c0_i64, %139, %c1_i64, %c8192_i64, %c0_i64, %c8192_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = i32, ub_base = 212960 : i64, ub_capacity = 8448 : i64}> : (memref<2112xi32, 1>, memref<1xi32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                ascvec.release_buf mte2, %c7_i64, 0 {cannir.buffer.generated_sync}
                ascvec.get_buf v, %c7_i64, 0 {cannir.buffer.generated_sync}
                ascvec.get_buf v, %c8_i64, 0 {cannir.buffer.generated_sync}
                ascvec.vf(%alloc_23, %103, %138, %alloc_26, %136, %alloc_27 : memref<8192xi32, 1>, i32, i32, memref<2112xi32, 1>, i32, memref<8192xi32, 1>) {
                ^bb0(%arg41: memref<8192xi32, 1>, %arg42: i32, %arg43: i32, %arg44: memref<2112xi32, 1>, %arg45: i32, %arg46: memref<8192xi32, 1>):
                  %c1_i32_36 = arith.constant 1 : i32
                  %c128_i32 = arith.constant 128 : i32
                  %c-1_i32 = arith.constant -1 : i32
                  %c7_i32_37 = arith.constant 7 : i32
                  %c8_i32 = arith.constant 8 : i32
                  %c0_i32_38 = arith.constant 0 : i32
                  %c0_39 = arith.constant 0 : index
                  %c8192_i32 = arith.constant 8192 : i32
                  %c64_i32_40 = arith.constant 64 : i32
                  scf.for %arg47 = %c0_i32_38 to %c128_i32 step %c1_i32_36  : i32 {
                    %145 = arith.muli %arg47, %c64_i32_40 overflow<nsw> : i32
                    %146 = arith.subi %c8192_i32, %145 overflow<nsw> : i32
                    %mask, %remaining_out = ascvec.update_mask %146 : !ascvec.mask
                    %147 = arith.index_cast %145 : i32 to index
                    %148 = ascvec.loadalign %arg41[%147] : memref<8192xi32, 1>, index -> <i32>
                    %149 = ascvec.reinterpret %148 : <i32> -> <ui32>
                    %150 = ascvec.arange %145 : i32 -> <ui32>
                    %151 = ascvec.cmp_lt_scalar %150, %arg42, %mask : <ui32>, i32 -> !ascvec.mask
                    %152 = ascvec.add_scalar %149, %arg43, %mask : <ui32>, i32
                    %153 = ascvec.duplicate_scalar %c0_i32_38, %mask : i32 -> <ui32>
                    %154 = ascvec.select %152, %153, %151 : <ui32>
                    %155 = ascvec.shr %154, 3, %mask : <ui32> -> <ui32>
                    %156 = ascvec.gather %arg44[%c0_39], %155, %mask : memref<2112xi32, 1>, <ui32>, !ascvec.mask -> <i32>
                    %157 = ascvec.mul_scalar %156, %c8_i32, %mask : <i32>, i32
                    %158 = ascvec.duplicate_scalar %c7_i32_37, %mask : i32 -> <ui32>
                    %159 = ascvec.and %154, %158, %mask : <ui32>, <ui32> -> <ui32>
                    %160 = ascvec.reinterpret %159 : <ui32> -> <i32>
                    %161 = ascvec.add %157, %160, %mask : <i32>
                    %162 = ascvec.add_scalar %161, %arg45, %mask : <i32>, i32
                    %163 = ascvec.duplicate_scalar %c-1_i32, %mask : i32 -> <i32>
                    %164 = ascvec.select %162, %163, %151 : <i32>
                    ascvec.storealign %arg46[%147], %164, %mask : memref<8192xi32, 1>, index, <i32>
                  } {cannir.loop_unroll = 1 : i64}
                  ascvec.mem_bar vst_vld
                } {cannir.vf_mode = "simd"}
                ascvec.release_buf v, %c8_i64, 0 {cannir.buffer.generated_sync}
                ascvec.release_buf v, %c7_i64, 0 {cannir.buffer.generated_sync}
                %140 = arith.muli %65, %c262144_i64 : i64
                %141 = arith.divui %140, %c8_i64 : i64
                %142 = arith.addi %66, %141 : i64
                %143 = ascscalar.bind_gm_ptr %142 : memref<1xi32>
                %144 = ascscalar.bitcast_memref %alloc_27 byte_offset = %c0_i64 : memref<8192xi32, 1> -> memref<?xi32, 1>
                ascvec.get_buf mte3, %c8_i64, 0 {cannir.buffer.generated_sync}
                ascvec.copy_ub2gm %143[%c0], %144[%c0], %c1_i32, %c32768_i32, %c0_i64, %c32768_i32 l2_cache_ctl %c0_i32 : memref<1xi32>, memref<?xi32, 1>
                ascvec.release_buf mte3, %c8_i64, 0 {cannir.buffer.generated_sync}
              }
            }
            scf.yield %102, %94 : i64, i64
          } else {
            scf.yield %arg39, %arg40 : i64, i64
          }
          scf.yield %70, %77#0, %77#1 : i64, i64, i64
        } {cannir.loop_unroll = 1 : i64}
        scf.yield %39#0, %39#2 : i64, i64
      } {cannir.loop_unroll = 1 : i64}
    }
    scf.if %1 {
      ascvec.sync_all
      ascvec.sync_block_arrive mte3, %c0_i64
      ascvec.sync_block_arrive mte3, %c1_i64 {mode = 0 : i32}
      ascvec.sync_block_wait s, %c1_i64 {mode = 0 : i32}
      ascvec.sync_block_wait s, %c2_i64
      %16 = arith.muli %8, %c2_i64 : i64
      %17 = arith.addi %16, %c36_i64 : i64
      %18 = arith.addi %17, %7 : i64
      %19 = arith.index_cast %18 : i64 to index
      %20 = arith.muli %19, %c8 : index
      %21 = memref.load %alloc_10[%20] : memref<1024xi32>
      %22 = arith.cmpi ne, %21, %c0_i32 : i32
      scf.if %22 {
        %23 = arith.addi %20, %c1 : index
        %24 = memref.load %alloc_10[%23] : memref<1024xi32>
        %25 = arith.index_cast %24 : i32 to index
        %26 = memref.load %alloc_2[%25] : memref<1xi32>
        %27 = arith.extsi %26 : i32 to i64
        %28 = arith.addi %20, %c2 : index
        %29 = memref.load %alloc_10[%28] : memref<1024xi32>
        %30 = arith.extsi %29 : i32 to i64
        %31 = arith.addi %27, %30 : i64
        %32 = arith.addi %20, %c3 : index
        %33 = memref.load %alloc_10[%32] : memref<1024xi32>
        %34 = arith.extsi %33 : i32 to i64
        %35 = arith.addi %20, %c4 : index
        %36 = memref.load %alloc_10[%35] : memref<1024xi32>
        %37 = arith.extsi %36 : i32 to i64
        %38 = arith.addi %20, %c5 : index
        %39 = memref.load %alloc_10[%38] : memref<1024xi32>
        %40 = arith.extsi %39 : i32 to i64
        %41 = arith.addi %20, %c6 : index
        %42 = memref.load %alloc_10[%41] : memref<1024xi32>
        %43 = arith.extsi %42 : i32 to i64
        %44 = arith.addi %40, %43 : i64
        scf.for %arg34 = %40 to %44 step %c1_i64  : i64 {
          %45 = arith.addi %31, %arg34 : i64
          %46 = arith.cmpi ne, %arg33, %c0_i64 : i64
          %47 = scf.if %46 -> (i32) {
            %48 = ascscalar.bind_gm_ptr %arg33 : memref<1xi32>
            %49 = arith.index_cast %45 : i64 to index
            %50 = memref.load %48[%49] : memref<1xi32>
            scf.yield %50 : i32
          } else {
            scf.yield %c0_i32 : i32
          }
          scf.for %arg35 = %c0_i64 to %37 step %c1_i64  : i64 {
            %48 = arith.subi %37, %arg35 : i64
            %49 = arith.minsi %48, %c1_i64 : i64
            %50 = arith.cmpi sgt, %arg35, %c0_i64 : i64
            %51 = arith.extui %50 : i1 to i64
            ascvec.sync_notify {dst_pipe = 5 : i32, event_id = 5 : i32, src_pipe = 2 : i32}
            ascvec.sync_wait {dst_pipe = 5 : i32, event_id = 5 : i32, src_pipe = 2 : i32}
            scf.if %50 {
              %93 = arith.minsi %45, %arg25 : i64
              %94 = arith.subi %arg25, %93 : i64
              %95 = arith.minsi %94, %c1_i64 : i64
              %96 = arith.cmpi sle, %45, %arg25 : i64
              %97 = arith.select %96, %95, %c0_i64 : i64
              %98 = arith.muli %45, %c32768_i64 : i64
              %99 = arith.muli %97, %c32768_i64 : i64
              %100 = arith.cmpi eq, %99, %c0_i64 : i64
              %101 = arith.cmpi ne, %99, %c0_i64 : i64
              %102 = arith.extui %101 : i1 to i64
              %103 = arith.select %100, %c4_i64, %99 : i64
              "ascvec.copy_gm2ub_plan"(%alloc_34, %2, %c0_i32, %c0_i32, %c0_i64, %98, %102, %103, %c0_i64, %103, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = i32, ub_base = 148224 : i64, ub_capacity = 65536 : i64}> : (memref<16384xi32, 1>, memref<1xi32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
              %104 = arith.muli %45, %c16384_i64 : i64
              %105 = arith.muli %97, %c16384_i64 : i64
              %106 = arith.cmpi eq, %105, %c0_i64 : i64
              %107 = arith.cmpi ne, %105, %c0_i64 : i64
              %108 = arith.extui %107 : i1 to i64
              %109 = arith.select %106, %c2_i64, %105 : i64
              "ascvec.copy_gm2ub_plan"(%alloc_35, %3, %c0_i32, %c0_i32, %c0_i64, %104, %108, %109, %c0_i64, %109, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = ui16, ub_base = 213760 : i64, ub_capacity = 32768 : i64}> : (memref<16384xui16, 1>, memref<1xui16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
            }
            scf.for %arg36 = %c0_i64 to %49 step %c1_i64  : i64 {
              %93 = arith.addi %34, %arg35 : i64
              %94 = arith.addi %93, %arg36 : i64
              %95 = arith.addi %94, %arg34 : i64
              %96 = arith.addi %arg36, %51 : i64
              %97 = arith.minsi %95, %4 : i64
              %98 = arith.subi %4, %97 : i64
              %99 = arith.minsi %98, %c1_i64 : i64
              %100 = arith.cmpi sle, %95, %4 : i64
              %101 = arith.select %100, %99, %c0_i64 : i64
              %102 = arith.muli %95, %c32768_i64 : i64
              %103 = arith.muli %96, %c32768_i64 : i64
              "ascvec.copy_gm2ub_plan"(%alloc_34, %5, %c0_i32, %c0_i32, %103, %102, %101, %c32768_i64, %c32768_i64, %c65536_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = i32, ub_base = 148224 : i64, ub_capacity = 65536 : i64}> : (memref<16384xi32, 1>, memref<1xi32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
              %104 = arith.muli %95, %c16384_i64 : i64
              %105 = arith.muli %96, %c16384_i64 : i64
              "ascvec.copy_gm2ub_plan"(%alloc_35, %6, %c0_i32, %c0_i32, %105, %104, %101, %c16384_i64, %c16384_i64, %c32768_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = ui16, ub_base = 213760 : i64, ub_capacity = 32768 : i64}> : (memref<16384xui16, 1>, memref<1xui16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
            } {cannir.loop_unroll = 1 : i64}
            ascvec.sync_notify {dst_pipe = 2 : i32, event_id = 5 : i32, src_pipe = 5 : i32}
            ascvec.sync_wait {dst_pipe = 2 : i32, event_id = 5 : i32, src_pipe = 5 : i32}
            %52 = arith.addi %arg35, %49 : i64
            %53 = arith.cmpi eq, %52, %37 : i64
            %54 = arith.select %53, %47, %c0_i32 : i32
            %55 = arith.addi %49, %51 : i64
            %56 = arith.muli %55, %c8192_i64 : i64
            %57 = arith.addi %56, %c63_i64 : i64
            %58 = arith.remsi %57, %c64_i64 : i64
            %59 = arith.cmpi ne, %58, %c0_i64 : i64
            %60 = arith.cmpi slt, %58, %c0_i64 : i64
            %61 = arith.andi %59, %60 : i1
            %62 = arith.divsi %57, %c64_i64 : i64
            %63 = arith.extui %61 : i1 to i64
            %64 = arith.subi %62, %63 : i64
            ascvec.vf(%56, %alloc_35, %alloc_34, %alloc_28, %64 : i64, memref<16384xui16, 1>, memref<16384xi32, 1>, memref<24576xui16, 1>, i64) {
            ^bb0(%arg36: i64, %arg37: memref<16384xui16, 1>, %arg38: memref<16384xi32, 1>, %arg39: memref<24576xui16, 1>, %arg40: i64):
              %c1_i64_36 = arith.constant 1 : i64
              %c0_i64_37 = arith.constant 0 : i64
              %c1_i16 = arith.constant 1 : i16
              %c0_i16 = arith.constant 0 : i16
              %c0_i32_38 = arith.constant 0 : i32
              %c1_i32_39 = arith.constant 1 : i32
              %c-1_i16 = arith.constant -1 : i16
              %c-32768_i16 = arith.constant -32768 : i16
              %c64_i64_40 = arith.constant 64 : i64
              %c64_i32_41 = arith.constant 64 : i32
              %c0_42 = arith.constant 0 : index
              %c1_43 = arith.constant 1 : index
              scf.for %arg41 = %c0_42 to %c1_43 step %c1_43 {
                %mask, %remaining_out = ascvec.update_mask %c64_i32_41 elem_bits 16 : !ascvec.mask
                %mask_44, %remaining_out_45 = ascvec.update_mask %c64_i32_41 : !ascvec.mask
              }
              scf.for %arg41 = %c0_i64_37 to %arg40 step %c1_i64_36  : i64 {
                %93 = arith.muli %arg41, %c64_i64_40 : i64
                %94 = arith.subi %arg36, %93 : i64
                %95 = arith.minsi %94, %c64_i64_40 : i64
                %96 = arith.trunci %95 : i64 to i32
                %mask, %remaining_out = ascvec.update_mask %96 elem_bits 16 : !ascvec.mask
                %mask_44, %remaining_out_45 = ascvec.update_mask %96 : !ascvec.mask
                %97 = arith.index_cast %93 : i64 to index
                %98 = ascvec.loadalign %arg37[%97] : memref<16384xui16, 1>, index -> <ui16>
                %99 = ascvec.loadalign %arg38[%97] : memref<16384xi32, 1>, index -> <i32>
                %100 = ascvec.duplicate_scalar %c-32768_i16, %mask : i16 -> <ui16>
                %101 = ascvec.xor %98, %100, %mask : <ui16>, <ui16> -> <ui16>
                %102 = ascvec.duplicate_scalar %c-1_i16, %mask : i16 -> <ui16>
                %103 = ascvec.xor %98, %102, %mask : <ui16>, <ui16> -> <ui16>
                %104 = ascvec.and %98, %100, %mask : <ui16>, <ui16> -> <ui16>
                %105 = ascvec.cmp_eq_scalar %104, %c-32768_i16, %mask : <ui16>, i16 -> !ascvec.mask
                %106 = ascvec.select %103, %101, %105 : <ui16>
                %107 = ascvec.duplicate_scalar %c1_i32_39, %mask_44 : i32 -> <i32>
                %108 = ascvec.duplicate_scalar %c0_i32_38, %mask_44 : i32 -> <i32>
                %109 = ascvec.cmp_ge_scalar %99, %c0_i32_38, %mask_44 : <i32>, i32 -> !ascvec.mask
                %110 = ascvec.select %107, %108, %109 : <i32>
                %111 = ascvec.reinterpret %110 : <i32> -> <ui32>
                %112 = ascvec.pack %111 src_elem_bits 32 dst_elem_bits 16 {part = "low"} : <ui32> -> <ui16>
                %113 = ascvec.duplicate_scalar %c0_i16, %mask : i16 -> <ui16>
                %114 = ascvec.cmp_eq_scalar %112, %c1_i16, %mask : <ui16>, i16 -> !ascvec.mask
                %115 = ascvec.select %106, %113, %114 : <ui16>
                ascvec.storealign %arg39[%97], %115, %mask : memref<24576xui16, 1>, index, <ui16>
              } {cannir.loop_unroll = 1 : i64}
              ascvec.mem_bar vst_vld
            } {cannir.vf_mode = "simd"}
            %65 = arith.addi %56, %c255_i64 : i64
            %66 = arith.remsi %65, %c256_i64 : i64
            %67 = arith.cmpi ne, %66, %c0_i64 : i64
            %68 = arith.cmpi slt, %66, %c0_i64 : i64
            %69 = arith.andi %67, %68 : i1
            %70 = arith.divsi %65, %c256_i64 : i64
            %71 = arith.extui %69 : i1 to i64
            %72 = arith.subi %70, %71 : i64
            %73 = arith.addi %56, %c-8191_i64 : i64
            %74 = arith.trunci %73 : i64 to i16
            ascvec.vf(%56, %alloc_28, %72, %alloc_30, %74, %alloc_31 : i64, memref<24576xui16, 1>, i64, memref<256xui16, 1>, i16, memref<64xi32, 1>) {
            ^bb0(%arg36: i64, %arg37: memref<24576xui16, 1>, %arg38: i64, %arg39: memref<256xui16, 1>, %arg40: i16, %arg41: memref<64xi32, 1>):
              %c128 = arith.constant 128 : index
              %c128_i16 = arith.constant 128 : i16
              %c-1_i16 = arith.constant -1 : i16
              %c1_i32_36 = arith.constant 1 : i32
              %c1_i64_37 = arith.constant 1 : i64
              %c0_i64_38 = arith.constant 0 : i64
              %c0_39 = arith.constant 0 : index
              %c256_i64_40 = arith.constant 256 : i64
              %c0_i16 = arith.constant 0 : i16
              %c128_i32 = arith.constant 128 : i32
              %mask, %remaining_out = ascvec.update_mask %c128_i32 elem_bits 16 : !ascvec.mask
              %93 = ascvec.duplicate_scalar %c0_i16, %mask : i16 -> <ui16>
              %94:2 = scf.for %arg42 = %c0_i64_38 to %arg38 step %c1_i64_37 iter_args(%arg43 = %93, %arg44 = %93) -> (!ascvec.vreg<ui16>, !ascvec.vreg<ui16>)  : i64 {
                %95 = arith.muli %arg42, %c256_i64_40 : i64
                %96 = arith.subi %arg36, %95 : i64
                %97 = arith.trunci %96 : i64 to i32
                %mask_45, %remaining_out_46 = ascvec.update_mask %97 elem_bits 8 : !ascvec.mask
                %98 = arith.index_cast %95 : i64 to index
                %result0, %result1 = ascvec.loadalign_deintlv %arg37[%98] {dintlv_width = "b8"} : memref<24576xui16, 1>, index -> <ui8>, <ui8>
                %99 = ascvec.cumulative_histogram %arg43, %result1, %mask_45 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                %100 = ascvec.cumulative_histogram %arg44, %result1, %mask_45 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                scf.yield %99, %100 : !ascvec.vreg<ui16>, !ascvec.vreg<ui16>
              } {cannir.loop_unroll = 1 : i64}
              %c0_41 = arith.constant 0 : index
              %c1_42 = arith.constant 1 : index
              scf.for %arg42 = %c0_41 to %c1_42 step %c1_42 {
                ascvec.storealign %arg39[%c0_39], %94#0, %mask : memref<256xui16, 1>, index, <ui16>
                ascvec.storealign %arg39[%c128], %94#1, %mask : memref<256xui16, 1>, index, <ui16>
              }
              ascvec.mem_bar vst_vld
              %c0_43 = arith.constant 0 : index
              %c1_44 = arith.constant 1 : index
              scf.for %arg42 = %c0_43 to %c1_44 step %c1_44 {
                %mask_45, %remaining_out_46 = ascvec.update_mask %c128_i32 elem_bits 16 : !ascvec.mask
                %mask_47, %remaining_out_48 = ascvec.update_mask %c1_i32_36 elem_bits 16 : !ascvec.mask
                %95 = ascvec.duplicate_scalar %c-1_i16, %mask_45 : i16 -> <ui16>
                %96 = ascvec.loadalign %arg39[%c0_39] : memref<256xui16, 1>, index -> <ui16>
                %97 = ascvec.loadalign %arg39[%c128] : memref<256xui16, 1>, index -> <ui16>
                %98 = ascvec.arange %c0_i16 : i16 -> <ui16>
                %99 = ascvec.cmp_ge_scalar %96, %arg40, %mask_45 : <ui16>, i16 -> !ascvec.mask
                %100 = ascvec.select %98, %95, %99 : <ui16>
                %101 = ascvec.arange %c128_i16 : i16 -> <ui16>
                %102 = ascvec.cmp_ge_scalar %97, %arg40, %mask_45 : <ui16>, i16 -> !ascvec.mask
                %103 = ascvec.select %101, %95, %102 : <ui16>
                %104 = ascvec.reduce_min %100, %mask_45 : <ui16>
                %105 = ascvec.reduce_min %103, %mask_45 : <ui16>
                %106 = ascvec.min %104, %105, %mask_47 : <ui16>
                %107 = ascvec.unpack %106 {part = "lower"} : <ui16> -> <ui32>
                %108 = ascvec.reinterpret %107 : <ui32> -> <i32>
                %mask_49, %remaining_out_50 = ascvec.update_mask %c1_i32_36 : !ascvec.mask
                ascvec.storealign %arg41[%c0_39], %108, %mask_49 : memref<64xi32, 1>, index, <i32>
              }
              ascvec.mem_bar vst_vld
            } {cannir.vf_mode = "simd"}
            ascvec.vf(%alloc_31, %alloc_30, %74, %alloc_29, %56, %alloc_28, %72 : memref<64xi32, 1>, memref<256xui16, 1>, i16, memref<24576xui16, 1>, i64, memref<24576xui16, 1>, i64) {
            ^bb0(%arg36: memref<64xi32, 1>, %arg37: memref<256xui16, 1>, %arg38: i16, %arg39: memref<24576xui16, 1>, %arg40: i64, %arg41: memref<24576xui16, 1>, %arg42: i64):
              %c128 = arith.constant 128 : index
              %c1_i64_36 = arith.constant 1 : i64
              %c0_i64_37 = arith.constant 0 : i64
              %c256_i64_38 = arith.constant 256 : i64
              %c128_i16 = arith.constant 128 : i16
              %c0_39 = arith.constant 0 : index
              %c1_i16 = arith.constant 1 : i16
              %c0_i16 = arith.constant 0 : i16
              %c1_i32_40 = arith.constant 1 : i32
              %c128_i32 = arith.constant 128 : i32
              %mask, %remaining_out = ascvec.update_mask %c128_i32 elem_bits 16 : !ascvec.mask
              %mask_41, %remaining_out_42 = ascvec.update_mask %c1_i32_40 elem_bits 16 : !ascvec.mask
              %93 = ascvec.duplicate_scalar %c0_i16, %mask : i16 -> <ui16>
              %94 = ascvec.duplicate_scalar %c1_i16, %mask : i16 -> <ui16>
              %95 = ascvec.loadalign_brc %arg36[%c0_39] {brc_width = "b16"} : memref<64xi32, 1>, index -> <ui32>
              %96 = ascvec.reinterpret %95 : <ui32> -> <ui16>
              %97 = ascvec.loadalign_brc %arg36[%c0_39] {brc_width = "b8"} : memref<64xi32, 1>, index -> <ui32>
              %98 = ascvec.reinterpret %97 : <ui32> -> <ui8>
              %99 = ascvec.sub %96, %94, %mask : <ui16>
              %100 = ascvec.cmp_eq_scalar %96, %c0_i16, %mask : <ui16>, i16 -> !ascvec.mask
              %101 = ascvec.select %93, %99, %100 : <ui16>
              %102 = ascvec.loadalign %arg37[%c0_39] : memref<256xui16, 1>, index -> <ui16>
              %103 = ascvec.arange %c0_i16 : i16 -> <ui16>
              %104 = ascvec.cmp_eq %103, %101, %mask : <ui16>, <ui16> -> !ascvec.mask
              %105 = ascvec.select %102, %93, %104 : <ui16>
              %106 = ascvec.reduce_max %105, %mask : <ui16>
              %107 = ascvec.add %93, %106, %mask : <ui16>
              %108 = ascvec.loadalign %arg37[%c128] : memref<256xui16, 1>, index -> <ui16>
              %109 = ascvec.arange %c128_i16 : i16 -> <ui16>
              %110 = ascvec.cmp_eq %109, %101, %mask : <ui16>, <ui16> -> !ascvec.mask
              %111 = ascvec.select %108, %93, %110 : <ui16>
              %112 = ascvec.reduce_max %111, %mask : <ui16>
              %113 = ascvec.add %107, %112, %mask : <ui16>
              %114 = ascvec.select %93, %113, %100 : <ui16>
              %115 = ascvec.duplicate_scalar %arg38, %mask : i16 -> <ui16>
              %116 = ascvec.sub %115, %114, %mask : <ui16>
              ascvec.storealign %arg39[%c0_39], %116, %mask_41 : memref<24576xui16, 1>, index, <ui16>
              %117:2 = scf.for %arg43 = %c0_i64_37 to %arg42 step %c1_i64_36 iter_args(%arg44 = %93, %arg45 = %93) -> (!ascvec.vreg<ui16>, !ascvec.vreg<ui16>)  : i64 {
                %118 = arith.muli %arg43, %c256_i64_38 : i64
                %119 = arith.subi %arg40, %118 : i64
                %120 = arith.trunci %119 : i64 to i32
                %mask_45, %remaining_out_46 = ascvec.update_mask %120 elem_bits 8 : !ascvec.mask
                %121 = arith.index_cast %118 : i64 to index
                %result0, %result1 = ascvec.loadalign_deintlv %arg41[%121] {dintlv_width = "b8"} : memref<24576xui16, 1>, index -> <ui8>, <ui8>
                %122 = ascvec.cmp_eq %result1, %98, %mask_45 : <ui8>, <ui8> -> !ascvec.mask
                %123 = ascvec.cumulative_histogram %arg44, %result0, %122 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                %124 = ascvec.cumulative_histogram %arg45, %result0, %122 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                scf.yield %123, %124 : !ascvec.vreg<ui16>, !ascvec.vreg<ui16>
              } {cannir.loop_unroll = 1 : i64}
              %c0_43 = arith.constant 0 : index
              %c1_44 = arith.constant 1 : index
              scf.for %arg43 = %c0_43 to %c1_44 step %c1_44 {
                ascvec.storealign %arg37[%c0_39], %117#0, %mask : memref<256xui16, 1>, index, <ui16>
                ascvec.storealign %arg37[%c128], %117#1, %mask : memref<256xui16, 1>, index, <ui16>
              }
              ascvec.mem_bar vst_vld
            } {cannir.vf_mode = "simd"}
            %75 = arith.addi %56, %c127_i64 : i64
            %76 = arith.remsi %75, %c128_i64 : i64
            %77 = arith.cmpi ne, %76, %c0_i64 : i64
            %78 = arith.cmpi slt, %76, %c0_i64 : i64
            %79 = arith.andi %77, %78 : i1
            %80 = arith.divsi %75, %c128_i64 : i64
            %81 = arith.extui %79 : i1 to i64
            %82 = arith.subi %80, %81 : i64
            ascvec.vf(%alloc_29, %alloc_30, %alloc_31, %56, %alloc_28, %82 : memref<24576xui16, 1>, memref<256xui16, 1>, memref<64xi32, 1>, i64, memref<24576xui16, 1>, i64) {
            ^bb0(%arg36: memref<24576xui16, 1>, %arg37: memref<256xui16, 1>, %arg38: memref<64xi32, 1>, %arg39: i64, %arg40: memref<24576xui16, 1>, %arg41: i64):
              %c128 = arith.constant 128 : index
              %c1_i64_36 = arith.constant 1 : i64
              %c0_i64_37 = arith.constant 0 : i64
              %c128_i64_38 = arith.constant 128 : i64
              %c128_i16 = arith.constant 128 : i16
              %c0_i16 = arith.constant 0 : i16
              %c-1_i16 = arith.constant -1 : i16
              %c0_39 = arith.constant 0 : index
              %c1_i32_40 = arith.constant 1 : i32
              %c128_i32 = arith.constant 128 : i32
              %c0_41 = arith.constant 0 : index
              %c1_42 = arith.constant 1 : index
              scf.for %arg42 = %c0_41 to %c1_42 step %c1_42 {
                %mask_45, %remaining_out_46 = ascvec.update_mask %c128_i32 elem_bits 16 : !ascvec.mask
                %mask_47, %remaining_out_48 = ascvec.update_mask %c1_i32_40 elem_bits 16 : !ascvec.mask
                %98 = ascvec.loadalign_brc %arg36[%c0_39] : memref<24576xui16, 1>, index -> <ui16>
                %99 = ascvec.duplicate_scalar %c-1_i16, %mask_45 : i16 -> <ui16>
                %100 = ascvec.loadalign %arg37[%c0_39] : memref<256xui16, 1>, index -> <ui16>
                %101 = ascvec.loadalign %arg37[%c128] : memref<256xui16, 1>, index -> <ui16>
                %102 = ascvec.arange %c0_i16 : i16 -> <ui16>
                %103 = ascvec.cmp_ge %100, %98, %mask_45 : <ui16>, <ui16> -> !ascvec.mask
                %104 = ascvec.select %102, %99, %103 : <ui16>
                %105 = ascvec.arange %c128_i16 : i16 -> <ui16>
                %106 = ascvec.cmp_ge %101, %98, %mask_45 : <ui16>, <ui16> -> !ascvec.mask
                %107 = ascvec.select %105, %99, %106 : <ui16>
                %108 = ascvec.reduce_min %104, %mask_45 : <ui16>
                %109 = ascvec.reduce_min %107, %mask_45 : <ui16>
                %110 = ascvec.min %108, %109, %mask_47 : <ui16>
                ascvec.storealign %arg36[%c0_39], %110, %mask_47 : memref<24576xui16, 1>, index, <ui16>
              }
              ascvec.mem_bar vst_vld
              %mask, %remaining_out = ascvec.update_mask %c128_i32 elem_bits 16 : !ascvec.mask
              %93 = ascvec.loadalign_brc %arg38[%c0_39] {brc_width = "b16"} : memref<64xi32, 1>, index -> <ui32>
              %94 = ascvec.reinterpret %93 : <ui32> -> <ui16>
              %95 = ascvec.loadalign_brc %arg36[%c0_39] : memref<24576xui16, 1>, index -> <ui16>
              %96 = ascvec.shl %94, 8, %mask : <ui16> -> <ui16>
              %97 = ascvec.or %96, %95, %mask : <ui16>, <ui16> -> <ui16>
              ascvec.store_unalign_begin %arg36 : memref<24576xui16, 1>
              scf.for %arg42 = %c0_i64_37 to %arg41 step %c1_i64_36  : i64 {
                %98 = arith.muli %arg42, %c128_i64_38 : i64
                %99 = arith.subi %arg39, %98 : i64
                %100 = arith.trunci %99 : i64 to i32
                %mask_45, %remaining_out_46 = ascvec.update_mask %100 elem_bits 16 : !ascvec.mask
                %101 = arith.index_cast %98 : i64 to index
                %102 = ascvec.loadalign %arg40[%101] : memref<24576xui16, 1>, index -> <ui16>
                %103 = arith.trunci %98 : i64 to i16
                %104 = ascvec.arange %103 : i16 -> <ui16>
                %105 = ascvec.cmp_gt %102, %97, %mask_45 : <ui16>, <ui16> -> !ascvec.mask
                %106 = ascvec.squeeze_store_reg %104, %105 : <ui16>, !ascvec.mask -> <ui16>
                ascvec.store_unalign %arg36[%c0_39], %106 : memref<24576xui16, 1>, <ui16>
              } {cannir.loop_unroll = 1 : i64}
              scf.for %arg42 = %c0_i64_37 to %arg41 step %c1_i64_36  : i64 {
                %98 = arith.muli %arg42, %c128_i64_38 : i64
                %99 = arith.subi %arg39, %98 : i64
                %100 = arith.trunci %99 : i64 to i32
                %mask_45, %remaining_out_46 = ascvec.update_mask %100 elem_bits 16 : !ascvec.mask
                %101 = arith.index_cast %98 : i64 to index
                %102 = ascvec.loadalign %arg40[%101] : memref<24576xui16, 1>, index -> <ui16>
                %103 = arith.trunci %98 : i64 to i16
                %104 = ascvec.arange %103 : i16 -> <ui16>
                %105 = ascvec.cmp_eq %102, %97, %mask_45 : <ui16>, <ui16> -> !ascvec.mask
                %106 = ascvec.squeeze_store_reg %104, %105 : <ui16>, !ascvec.mask -> <ui16>
                ascvec.store_unalign %arg36[%c0_39], %106 : memref<24576xui16, 1>, <ui16>
              } {cannir.loop_unroll = 1 : i64}
              %c0_43 = arith.constant 0 : index
              %c1_44 = arith.constant 1 : index
              scf.for %arg42 = %c0_43 to %c1_44 step %c1_44 {
                ascvec.store_unalign_post %arg36[%c0_39] : memref<24576xui16, 1>
              }
              ascvec.mem_bar vst_vld
            } {cannir.vf_mode = "simd"}
            ascvec.get_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
            ascvec.vf(%alloc_29, %alloc_34, %alloc_28, %alloc_35, %54, %alloc_32, %alloc_33 : memref<24576xui16, 1>, memref<16384xi32, 1>, memref<24576xui16, 1>, memref<16384xui16, 1>, i32, memref<8192xi32, 1>, memref<8192xui16, 1>) {
            ^bb0(%arg36: memref<24576xui16, 1>, %arg37: memref<16384xi32, 1>, %arg38: memref<24576xui16, 1>, %arg39: memref<16384xui16, 1>, %arg40: i32, %arg41: memref<8192xi32, 1>, %arg42: memref<8192xui16, 1>):
              %c128_i32 = arith.constant 128 : i32
              %c-128_i16 = arith.constant -128 : i16
              %c0_i32_36 = arith.constant 0 : i32
              %c1_i32_37 = arith.constant 1 : i32
              %c-1_i32 = arith.constant -1 : i32
              %c0_i16 = arith.constant 0 : i16
              %c1_i16 = arith.constant 1 : i16
              %c0_38 = arith.constant 0 : index
              %c8192_i32 = arith.constant 8192 : i32
              %c64_i32_39 = arith.constant 64 : i32
              scf.for %arg43 = %c0_i32_36 to %c128_i32 step %c1_i32_37  : i32 {
                %93 = arith.muli %arg43, %c64_i32_39 overflow<nsw> : i32
                %94 = arith.subi %c8192_i32, %93 overflow<nsw> : i32
                %mask, %remaining_out = ascvec.update_mask %94 : !ascvec.mask
                %mask_40, %remaining_out_41 = ascvec.update_mask %94 elem_bits 16 : !ascvec.mask
                %mask_42, %remaining_out_43 = ascvec.update_mask %c64_i32_39 elem_bits 16 : !ascvec.mask
                %mask_44, %remaining_out_45 = ascvec.update_mask %c64_i32_39 elem_bits 16 : !ascvec.mask
                %95 = ascvec.mask_and %mask_40, %mask_42, %mask_44
                %96 = arith.index_cast %93 : i32 to index
                %97 = ascvec.loadalign %arg36[%96] : memref<24576xui16, 1>, index -> <ui16>
                %98 = ascvec.unpack %97 {part = "lower"} : <ui16> -> <ui32>
                %99 = ascvec.reinterpret %98 : <ui32> -> <ui32>
                %100 = ascvec.gather %arg37[%c0_38], %99, %mask : memref<16384xi32, 1>, <ui32>, !ascvec.mask -> <i32>
                %101 = ascvec.gather %arg38[%c0_38], %97, %95 : memref<24576xui16, 1>, <ui16>, !ascvec.mask -> <ui16>
                %102 = ascvec.gather %arg39[%c0_38], %97, %95 : memref<16384xui16, 1>, <ui16>, !ascvec.mask -> <ui16>
                %103 = ascvec.duplicate_scalar %c1_i16, %95 : i16 -> <ui16>
                %104 = ascvec.duplicate_scalar %c0_i16, %95 : i16 -> <ui16>
                %105 = ascvec.cmp_ge_scalar %101, %c1_i16, %95 : <ui16>, i16 -> !ascvec.mask
                %106 = ascvec.select %103, %104, %105 : <ui16>
                %107 = ascvec.unpack %106 {part = "lower"} : <ui16> -> <ui32>
                %108 = ascvec.duplicate_scalar %c-1_i32, %mask : i32 -> <i32>
                %109 = ascvec.cmp_eq_scalar %107, %c1_i32_37, %mask : <ui32>, i32 -> !ascvec.mask
                %110 = ascvec.select %100, %108, %109 : <i32>
                %111 = ascvec.duplicate_scalar %c1_i32_37, %mask : i32 -> <i32>
                %112 = ascvec.duplicate_scalar %c0_i32_36, %mask : i32 -> <i32>
                %113 = ascvec.cmp_ge_scalar %110, %c0_i32_36, %mask : <i32>, i32 -> !ascvec.mask
                %114 = ascvec.select %111, %112, %113 : <i32>
                %115 = ascvec.reinterpret %114 : <i32> -> <ui32>
                %116 = ascvec.pack %115 src_elem_bits 32 dst_elem_bits 16 {part = "low"} : <ui32> -> <ui16>
                %117 = ascvec.select %110, %108, %113 : <i32>
                %118 = ascvec.duplicate_scalar %c-128_i16, %95 : i16 -> <ui16>
                %119 = ascvec.cmp_eq_scalar %116, %c1_i16, %95 : <ui16>, i16 -> !ascvec.mask
                %120 = ascvec.select %102, %118, %119 : <ui16>
                %121 = ascvec.add_scalar %117, %arg40, %mask : <i32>, i32
                %122 = ascvec.cmp_eq_scalar %114, %c1_i32_37, %mask : <i32>, i32 -> !ascvec.mask
                %123 = ascvec.select %121, %108, %122 : <i32>
                ascvec.storealign %arg41[%96], %123, %mask : memref<8192xi32, 1>, index, <i32>
                ascvec.storealign %arg42[%96], %120, %95 : memref<8192xui16, 1>, index, <ui16>
              } {cannir.loop_unroll = 1 : i64}
              ascvec.mem_bar vst_vld
            } {cannir.vf_mode = "simd"}
            ascvec.release_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
            %83 = arith.muli %45, %c262144_i64 : i64
            %84 = arith.divui %83, %c8_i64 : i64
            %85 = arith.addi %arg29, %84 : i64
            %86 = ascscalar.bind_gm_ptr %85 : memref<1xi32>
            %87 = ascscalar.bitcast_memref %alloc_32 byte_offset = %c0_i64 : memref<8192xi32, 1> -> memref<?xi32, 1>
            ascvec.get_buf mte3, %c0_i64, 0 {cannir.buffer.generated_sync}
            ascvec.copy_ub2gm %86[%c0], %87[%c0], %c1_i32, %c32768_i32, %c0_i64, %c32768_i32 l2_cache_ctl %c0_i32 : memref<1xi32>, memref<?xi32, 1>
            ascvec.release_buf mte3, %c0_i64, 0 {cannir.buffer.generated_sync}
            %88 = arith.muli %45, %c131072_i64 : i64
            %89 = arith.divui %88, %c8_i64 : i64
            %90 = arith.addi %arg30, %89 : i64
            %91 = ascscalar.bind_gm_ptr %90 : memref<1xui16>
            %92 = ascscalar.bitcast_memref %alloc_33 byte_offset = %c0_i64 : memref<8192xui16, 1> -> memref<?xui16, 1>
            ascvec.copy_ub2gm %91[%c0], %92[%c0], %c1_i32, %c16384_i32, %c0_i64, %c16384_i32 l2_cache_ctl %c0_i32 : memref<1xui16>, memref<?xui16, 1>
            ascvec.sync_notify {dst_pipe = 5 : i32, event_id = 5 : i32, src_pipe = 6 : i32}
            ascvec.sync_wait {dst_pipe = 5 : i32, event_id = 5 : i32, src_pipe = 6 : i32}
          } {cannir.loop_unroll = 1 : i64}
        } {cannir.loop_unroll = 1 : i64}
      }
    }
    return
  }
  func.func @kernel_QuantSparseLightningIndexer_0_aic(%arg0: i64, %arg1: !cannir.tensor<<gm, ui8>, <<(-1, 1, 64)>, <(-1, 64, 1)>>, <<(-1, 1, 64)>, <(-1, 64, 1)>>, identity>, %arg2: i64, %arg3: !cannir.tensor<<gm, ui8>, <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>>, <<(-1, 1, 2, 2)>, <(-1, 4, 2, 1)>>, identity>, %arg4: i64, %arg5: !cannir.tensor<<gm, f32>, <<(-1, 32)>, <(32, 1)>>, <<(-1, 32)>, <(32, 1)>>, identity>, %arg6: !cannir.tensor<<gm, !ascdtype.fp8_e8m0>, <<(-1, 2, 2)>, <(4, 2, 1)>>, <<(-1, 2, 2)>, <(4, 2, 1)>>, identity>, %arg7: !cannir.tensor<<gm, i32>, <<(-1, 1, 2048)>, <(2048, 2048, 1)>>, <<(-1, 1, 2048)>, <(2048, 2048, 1)>>, identity>, %arg8: !cannir.tensor<<gm, i32>, <<(-1, 1)>, <(1, 1)>>, <<(-1, 1)>, <(1, 1)>>, identity>, %arg9: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg10: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg11: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg12: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg13: !cannir.tensor<<gm, i32>, <<(-1, 1)>, <(1, 1)>>, <<(-1, 1)>, <(1, 1)>>, identity>, %arg14: !cannir.tensor<<gm, i32>, <<(-1, 1)>, <(1, 1)>>, <<(-1, 1)>, <(1, 1)>>, identity>, %arg15: !cannir.tensor<<gm, ui8>, <<(73728, 64)>, <(64, 1)>>, <<(73728, 64)>, <(64, 1)>>, identity>, %arg16: !cannir.tensor<<gm, !ascdtype.fp4x2_e2m1>, <<(73728, 128)>, <(128, 1)>>, <<(73728, 128)>, <(128, 1)>>, identity>, %arg17: !cannir.tensor<<gm, ui8>, <<(73728, 4)>, <(4, 1)>>, <<(73728, 4)>, <(4, 1)>>, identity>, %arg18: !cannir.tensor<<gm, !ascdtype.fp8_e8m0>, <<(73728, 2, 2)>, <(4, 2, 1)>>, <<(73728, 2, 2)>, <(4, 2, 1)>>, identity>, %arg19: !cannir.tensor<<gm, ui16>, <<(-1, 16384)>, <(16384, 1)>>, <<(-1, 16384)>, <(16384, 1)>>, identity>, %arg20: i64, %arg21: !cannir.tensor<<gm, i32>, <<(-1, 8192)>, <(8192, 1)>>, <<(-1, 8192)>, <(8192, 1)>>, identity>, %arg22: i64, %arg23: i64, %arg24: i64, %arg25: i64, %arg26: i64, %arg27: i64, %arg28: !cannir.tensor<<gm, i32>, <<(128, 8)>, <(8, 1)>>, <<(128, 8)>, <(8, 1)>>, identity>, %arg29: i64, %arg30: i64, %arg31: i64, %arg32: i64, %arg33: i64) attributes {cannir.buffer.auto_sync, cannir.buffer.sync_analyzed, cannir.buffer.sync_verified, cannir.buffer.tensor_state, cannir.core = "aic", cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.device, cannir.explicit_ring_buffer_cursors, cannir.kernel_group = "kernel_QuantSparseLightningIndexer_0", cannir.resource.ids_prepared} {
    %c8192 = arith.constant 8192 : index
    %c512_i32 = arith.constant 512 : i32
    %c3 = arith.constant 3 : index
    %c6 = arith.constant 6 : index
    %c5 = arith.constant 5 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c7 = arith.constant 7 : index
    %c8 = arith.constant 8 : index
    %c288 = arith.constant 288 : index
    %c0_i32 = arith.constant 0 : i32
    %c32_i64 = arith.constant 32 : i64
    %c1_i32 = arith.constant 1 : i32
    %c64_i32 = arith.constant 64 : i32
    %c64_i64 = arith.constant 64 : i64
    %false = arith.constant false
    %true = arith.constant true
    %c128_i32 = arith.constant 128 : i32
    %c8192_i64 = arith.constant 8192 : i64
    %c0 = arith.constant 0 : index
    %c32_i32 = arith.constant 32 : i32
    %c128_i64 = arith.constant 128 : i64
    %c512_i64 = arith.constant 512 : i64
    %c2_i32 = arith.constant 2 : i32
    %c73728_i64 = arith.constant 73728 : i64
    %c30_i64 = arith.constant 30 : i64
    %c26_i64 = arith.constant 26 : i64
    %c256_i32 = arith.constant 256 : i32
    %c63_i64 = arith.constant 63 : i64
    %c2 = arith.constant 2 : index
    %c4096_i64 = arith.constant 4096 : i64
    %c2048_i64 = arith.constant 2048 : i64
    %c23_i64 = arith.constant 23 : i64
    %c22_i64 = arith.constant 22 : i64
    %c21_i64 = arith.constant 21 : i64
    %c20_i64 = arith.constant 20 : i64
    %c16_i64 = arith.constant 16 : i64
    %c14_i64 = arith.constant 14 : i64
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
    %alloc = memref.alloc() {cannir.arg_index = 6 : i64} : memref<1x!ascdtype.fp8_e8m0>
    %alloc_0 = memref.alloc() {cannir.arg_index = 8 : i64} : memref<1xi32>
    %alloc_1 = memref.alloc() {cannir.arg_index = 9 : i64} : memref<1xi32>
    %alloc_2 = memref.alloc() {cannir.arg_index = 10 : i64} : memref<1xi32>
    %alloc_3 = memref.alloc() {cannir.arg_index = 12 : i64} : memref<1xi32>
    %alloc_4 = memref.alloc() {cannir.arg_index = 16 : i64} : memref<9437184x!ascdtype.fp4x2_e2m1>
    %alloc_5 = memref.alloc() {cannir.arg_index = 18 : i64} : memref<294912x!ascdtype.fp8_e8m0>
    %alloc_6 = memref.alloc() {cannir.arg_index = 28 : i64} : memref<1024xi32>
    %alloc_7 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<16384xbf16, 1>
    %alloc_8 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<4096x!ascdtype.fp4x2_e2m1, 3>
    %alloc_9 = memref.alloc() {cannir.byte_offset = 65536 : i64} : memref<16384xf32, 5>
    %alloc_10 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<16384xf32, 5>
    %alloc_11 = memref.alloc() {cannir.byte_offset = 32768 : i64} : memref<65536x!ascdtype.fp4x2_e2m1, 4>
    %alloc_12 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<65536x!ascdtype.fp4x2_e2m1, 4>
    %alloc_13 = memref.alloc() {cannir.byte_offset = 70144 : i64} : memref<2048x!ascdtype.fp8_e8m0, 2>
    %alloc_14 = memref.alloc() {cannir.byte_offset = 68096 : i64} : memref<2048x!ascdtype.fp8_e8m0, 2>
    %alloc_15 = memref.alloc() {cannir.byte_offset = 34816 : i64} : memref<65536x!ascdtype.fp4x2_e2m1, 2>
    %alloc_16 = memref.alloc() {cannir.byte_offset = 2048 : i64} : memref<65536x!ascdtype.fp4x2_e2m1, 2>
    %alloc_17 = memref.alloc() {cannir.byte_offset = 67584 : i64} : memref<128x!ascdtype.fp8_e8m0, 2>
    %alloc_18 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<4096x!ascdtype.fp4x2_e2m1, 2>
    %0 = memref.load %alloc_6[%c288] : memref<1024xi32>
    %1 = arith.cmpi ne, %0, %c0_i32 : i32
    %2 = arith.muli %arg25, %c32_i64 : i64
    %3 = ascscalar.bind_gm_ptr %arg0 : memref<1x!ascdtype.fp4x2_e2m1>
    %4 = ascscalar.get_block_idx
    %5 = arith.index_cast %4 : i64 to index
    %6 = arith.muli %5, %c8 : index
    %7 = arith.addi %6, %c7 : index
    %8 = memref.load %alloc_6[%7] : memref<1024xi32>
    %9 = arith.extsi %8 : i32 to i64
    %10 = memref.load %alloc_6[%6] : memref<1024xi32>
    %11 = arith.cmpi ne, %10, %c0_i32 : i32
    scf.if %11 {
      %12 = arith.addi %6, %c1 : index
      %13 = memref.load %alloc_6[%12] : memref<1024xi32>
      %14 = arith.extsi %13 : i32 to i64
      %15 = arith.addi %6, %c4 : index
      %16 = memref.load %alloc_6[%15] : memref<1024xi32>
      %17 = arith.extsi %16 : i32 to i64
      %18 = arith.addi %17, %c1_i64 : i64
      %19 = arith.minsi %18, %arg26 : i64
      %20:2 = scf.for %arg34 = %14 to %19 step %c1_i64 iter_args(%arg35 = %9, %arg36 = %c0_i64) -> (i64, i64)  : i64 {
        %21 = arith.index_cast %arg34 : i64 to index
        %22 = memref.load %alloc_1[%21] : memref<1xi32>
        %23 = arith.extsi %22 : i32 to i64
        %24 = arith.addi %arg34, %c1_i64 : i64
        %25 = arith.index_cast %24 : i64 to index
        %26 = memref.load %alloc_1[%25] : memref<1xi32>
        %27 = arith.extsi %26 : i32 to i64
        %28 = memref.load %alloc_2[%21] : memref<1xi32>
        %29 = memref.load %alloc_3[%21] : memref<1xi32>
        %30 = arith.subi %27, %23 : i64
        %31 = arith.cmpi eq, %arg34, %14 : i64
        %32 = scf.if %31 -> (i64) {
          %36 = arith.addi %6, %c2 : index
          %37 = memref.load %alloc_6[%36] : memref<1024xi32>
          %38 = arith.extsi %37 : i32 to i64
          scf.yield %38 : i64
        } else {
          scf.yield %c0_i64 : i64
        }
        %33 = arith.cmpi eq, %arg34, %17 : i64
        %34 = scf.if %33 -> (i64) {
          %36 = arith.addi %6, %c5 : index
          %37 = memref.load %alloc_6[%36] : memref<1024xi32>
          %38 = arith.extsi %37 : i32 to i64
          %39 = arith.addi %6, %c6 : index
          %40 = memref.load %alloc_6[%39] : memref<1024xi32>
          %41 = arith.cmpi sgt, %40, %c0_i32 : i32
          %42 = arith.extui %41 : i1 to i64
          %43 = arith.addi %38, %42 : i64
          scf.yield %43 : i64
        } else {
          scf.yield %30 : i64
        }
        %35:3 = scf.for %arg37 = %32 to %34 step %c1_i64 iter_args(%arg38 = %arg35, %arg39 = %c0_i64, %arg40 = %arg36) -> (i64, i64, i64)  : i64 {
          %36 = arith.addi %23, %arg37 : i64
          %37 = arith.index_cast %36 : i64 to index
          %38 = memref.load %alloc_0[%37] : memref<1xi32>
          %39 = arith.extsi %38 : i32 to i64
          %40 = arith.addi %39, %c63_i64 : i64
          %41 = arith.remsi %40, %c64_i64 : i64
          %42 = arith.cmpi ne, %41, %c0_i64 : i64
          %43 = arith.cmpi slt, %41, %c0_i64 : i64
          %44 = arith.andi %42, %43 : i1
          %45 = arith.divsi %40, %c64_i64 : i64
          %46 = arith.extui %44 : i1 to i64
          %47 = arith.subi %45, %46 : i64
          %48 = arith.cmpi eq, %arg37, %32 : i64
          %49 = arith.select %31, %48, %false : i1
          %50 = scf.if %49 -> (i64) {
            %68 = arith.addi %6, %c3 : index
            %69 = memref.load %alloc_6[%68] : memref<1024xi32>
            %70 = arith.extsi %69 : i32 to i64
            scf.yield %70 : i64
          } else {
            scf.yield %c0_i64 : i64
          }
          %51 = scf.if %33 -> (i1) {
            %68 = arith.addi %6, %c5 : index
            %69 = memref.load %alloc_6[%68] : memref<1024xi32>
            %70 = arith.extsi %69 : i32 to i64
            %71 = arith.cmpi eq, %arg37, %70 : i64
            scf.yield %71 : i1
          } else {
            scf.yield %false : i1
          }
          %52 = scf.if %51 -> (i64) {
            %68 = arith.addi %6, %c6 : index
            %69 = memref.load %alloc_6[%68] : memref<1024xi32>
            %70 = arith.extsi %69 : i32 to i64
            %71 = arith.minsi %47, %70 : i64
            scf.yield %71 : i64
          } else {
            scf.yield %47 : i64
          }
          %53 = arith.extsi %28 : i32 to i64
          %54 = arith.cmpi sge, %arg37, %53 : i64
          %55 = arith.select %54, %c0_i32, %29 : i32
          %56 = arith.cmpi sgt, %55, %c0_i32 : i32
          %57 = arith.cmpi sgt, %50, %c0_i64 : i64
          %58 = arith.cmpi slt, %52, %47 : i64
          %59 = arith.select %57, %true, %58 : i1
          %60 = arith.select %56, %59, %false : i1
          %61 = arith.addi %arg38, %c1_i64 : i64
          %62 = arith.select %60, %61, %arg38 : i64
          %63 = arith.cmpi sgt, %38, %c0_i32 : i32
          %64 = arith.cmpi sgt, %52, %50 : i64
          %65 = arith.select %64, %56, %false : i1
          %66 = arith.select %63, %65, %false : i1
          %67:2 = scf.if %66 -> (i64, i64) {
            %68 = ascscalar.bitcast_memref %alloc_18 byte_offset = %c0_i64 : memref<4096x!ascdtype.fp4x2_e2m1, 2> -> memref<?x!ascdtype.fp4x2_e2m1, 2>
            %69 = arith.muli %36, %c4096_i64 : i64
            %70 = arith.divsi %2, %c32_i64 : i64
            %71 = arith.minsi %36, %70 : i64
            %72 = arith.muli %71, %c32_i64 : i64
            %73 = arith.subi %2, %72 : i64
            %74 = arith.minsi %73, %c32_i64 : i64
            %75 = arith.cmpi sle, %36, %70 : i64
            %76 = arith.select %75, %74, %c0_i64 : i64
            asccube.get_buf mte2, %c0_i64, 0 {cannir.buffer.generated_sync}
            %77 = arith.cmpi ugt, %76, %c0_i64 : i64
            scf.if %77 {
              %101 = arith.index_cast %69 : i64 to index
              %102 = arith.trunci %76 : i64 to i32
              asccube.copy_gm2l1 %68[%c0], %3[%101], %102, %c64_i32, %c64_i64, %c1_i32, %c0_i64, %c0_i64, %c32_i32 l2_cache_ctl %c0_i32 {nd2nz = true} : memref<?x!ascdtype.fp4x2_e2m1, 2>, memref<1x!ascdtype.fp4x2_e2m1>
            }
            asccube.release_buf mte2, %c0_i64, 0 {cannir.buffer.generated_sync}
            %78 = cannir.extract_layout %arg6 : !cannir.tensor<<gm, !ascdtype.fp8_e8m0>, <<(-1, 2, 2)>, <(4, 2, 1)>>, <<(-1, 2, 2)>, <(4, 2, 1)>>, identity>
            %79 = cannir.extract_shape %78 : <<(-1, 2, 2)>, <(4, 2, 1)>> -> <(-1, 2, 2)>
            %80 = cannir.extract_dim %79[0] : !cannir.shape<(-1, 2, 2)>
            %81 = arith.divsi %80, %c32_i64 : i64
            %82 = arith.minsi %36, %81 : i64
            %83 = arith.muli %82, %c32_i64 : i64
            %84 = arith.subi %80, %83 : i64
            %85 = arith.minsi %84, %c32_i64 : i64
            %86 = arith.cmpi sle, %36, %81 : i64
            %87 = arith.select %86, %85, %c0_i64 : i64
            %88 = arith.muli %36, %c128_i64 : i64
            %89 = ascscalar.bitcast_memref %alloc_17 byte_offset = %c0_i64 : memref<128x!ascdtype.fp8_e8m0, 2> -> memref<?x!ascdtype.fp8_e8m0, 2>
            asccube.get_buf mte2, %c3_i64, 0 {cannir.buffer.generated_sync}
            %90 = arith.cmpi ugt, %87, %c0_i64 : i64
            scf.if %90 {
              %101 = arith.index_cast %88 : i64 to index
              %102 = arith.trunci %87 : i64 to i32
              asccube.copy_gm2l1 %89[%c0], %alloc[%101], %c2_i32, %102, %c4_i64, %c1_i32, %c0_i64, %c0_i64, %c2_i32 l2_cache_ctl %c0_i32 {dn2nz = true, mx_scale = true} : memref<?x!ascdtype.fp8_e8m0, 2>, memref<1x!ascdtype.fp8_e8m0>
            }
            asccube.release_buf mte2, %c3_i64, 0 {cannir.buffer.generated_sync}
            %91 = ascscalar.bitcast_memref %alloc_8 byte_offset = %c0_i64 : memref<4096x!ascdtype.fp4x2_e2m1, 3> -> memref<?x!ascdtype.fp4x2_e2m1, 3>
            %92 = ascscalar.memref_address %91 : memref<?x!ascdtype.fp4x2_e2m1, 3>
            %93 = ascscalar.bind_local_ptr %92 : memref<?xi8, 6>
            asccube.get_buf mte1, %c6_i64, 0 {cannir.buffer.generated_sync}
            asccube.get_buf mte1, %c0_i64, 0 {cannir.buffer.generated_sync}
            asccube.get_buf mte1, %c3_i64, 0 {cannir.buffer.generated_sync}
            asccube.copy_l12l0a %91[%c0], %68[%c0], %c0_i32, %c0_i32, %c2_i32, %c2_i32, %c2_i32, %c2_i32 : memref<?x!ascdtype.fp4x2_e2m1, 3>, memref<?x!ascdtype.fp4x2_e2m1, 2>
            asccube.copy_l12l0a %93[%c0], %89[%c0], %c0_i32, %c0_i32, %c2_i32, %c2_i32, %c2_i32, %c2_i32 : memref<?xi8, 6>, memref<?x!ascdtype.fp8_e8m0, 2>
            asccube.release_buf mte1, %c3_i64, 0 {cannir.buffer.generated_sync}
            asccube.release_buf mte1, %c0_i64, 0 {cannir.buffer.generated_sync}
            asccube.release_buf mte1, %c6_i64, 0 {cannir.buffer.generated_sync}
            %94 = arith.addi %52, %c1_i64 : i64
            %95 = scf.for %arg41 = %50 to %94 step %c1_i64 iter_args(%arg42 = %arg40) -> (i64)  : i64 {
              %101 = arith.cmpi sgt, %arg41, %50 : i64
              %102 = scf.if %101 -> (i64) {
                %103 = arith.subi %arg41, %c1_i64 : i64
                %104 = arith.remsi %103, %c4_i64 : i64
                %105 = arith.cmpi ne, %104, %c0_i64 : i64
                %106 = arith.cmpi slt, %104, %c0_i64 : i64
                %107 = arith.andi %105, %106 : i1
                %108 = arith.select %107, %c4_i64, %c0_i64 : i64
                %109 = arith.addi %104, %108 overflow<nsw> : i64
                %110 = arith.addi %109, %c6_i64 overflow<nsw> : i64
                asccube.sync_intra_wait mte2, %110
                %111 = arith.addi %109, %c22_i64 overflow<nsw> : i64
                asccube.sync_intra_wait mte2, %111
                %112 = arith.muli %4, %c4_i64 : i64
                %113 = arith.addi %112, %109 : i64
                %114 = arith.muli %113, %c512_i64 : i64
                %115 = arith.subi %c73728_i64, %114 : i64
                %116 = arith.remsi %103, %c2_i64 : i64
                %117 = arith.cmpi ne, %116, %c0_i64 : i64
                %118 = arith.cmpi slt, %116, %c0_i64 : i64
                %119 = arith.andi %117, %118 : i1
                %120 = arith.select %119, %c2_i64, %c0_i64 : i64
                %121 = arith.addi %116, %120 overflow<nsw> : i64
                %122 = arith.cmpi eq, %arg42, %c0_i64 : i64
                %123 = ascscalar.memref_address %alloc_16 : memref<65536x!ascdtype.fp4x2_e2m1, 2>
                %124 = ascscalar.memref_address %alloc_15 : memref<65536x!ascdtype.fp4x2_e2m1, 2>
                %125 = arith.select %122, %123, %124 : i64
                %126 = arith.select %122, %c1_i64, %c2_i64 : i64
                %127 = arith.addi %arg42, %c1_i64 : i64
                %128 = arith.remui %127, %c2_i64 : i64
                %129 = arith.divsi %115, %c512_i64 : i64
                %130 = arith.minsi %129, %c0_i64 : i64
                %131 = arith.muli %130, %c512_i64 : i64
                %132 = arith.subi %115, %131 : i64
                %133 = arith.minsi %132, %c512_i64 : i64
                %134 = arith.cmpi sge, %129, %c0_i64 : i64
                %135 = arith.select %134, %133, %c0_i64 : i64
                %136 = ascscalar.bind_local_ptr %125 : memref<1x!ascdtype.fp4x2_e2m1, 2>
                %137 = arith.divui %114, %c4_i64 : i64
                %138 = arith.muli %137, %c512_i64 : i64
                asccube.get_buf mte2, %126, 0 {cannir.buffer.generated_sync}
                %139 = arith.cmpi ugt, %135, %c0_i64 : i64
                scf.if %139 {
                  %167 = arith.index_cast %138 : i64 to index
                  %168 = arith.trunci %135 : i64 to i32
                  asccube.copy_gm2l1 %136[%c0], %alloc_4[%167], %168, %c64_i32, %c64_i64, %c1_i32, %c0_i64, %c0_i64, %c512_i32 l2_cache_ctl %c0_i32 {nd2nz = true} : memref<1x!ascdtype.fp4x2_e2m1, 2>, memref<9437184x!ascdtype.fp4x2_e2m1>
                }
                asccube.release_buf mte2, %126, 0 {cannir.buffer.generated_sync}
                %140 = ascscalar.memref_address %alloc_14 : memref<2048x!ascdtype.fp8_e8m0, 2>
                %141 = ascscalar.memref_address %alloc_13 : memref<2048x!ascdtype.fp8_e8m0, 2>
                %142 = arith.select %122, %140, %141 : i64
                %143 = arith.select %122, %c4_i64, %c5_i64 : i64
                %144 = arith.muli %113, %c2048_i64 : i64
                %145 = ascscalar.bind_local_ptr %142 : memref<1x!ascdtype.fp8_e8m0, 2>
                asccube.get_buf mte2, %143, 0 {cannir.buffer.generated_sync}
                scf.if %139 {
                  %167 = arith.index_cast %144 : i64 to index
                  %168 = arith.trunci %135 : i64 to i32
                  asccube.copy_gm2l1 %145[%c0], %alloc_5[%167], %c2_i32, %168, %c4_i64, %c1_i32, %c0_i64, %c0_i64, %c2_i32 l2_cache_ctl %c0_i32 {dn2nz = true, mx_scale = true} : memref<1x!ascdtype.fp8_e8m0, 2>, memref<294912x!ascdtype.fp8_e8m0>
                }
                asccube.release_buf mte2, %143, 0 {cannir.buffer.generated_sync}
                %146 = ascscalar.memref_address %alloc_12 : memref<65536x!ascdtype.fp4x2_e2m1, 4>
                %147 = ascscalar.memref_address %alloc_11 : memref<65536x!ascdtype.fp4x2_e2m1, 4>
                %148 = arith.select %122, %146, %147 : i64
                %149 = arith.select %122, %c7_i64, %c8_i64 : i64
                %150 = ascscalar.bind_local_ptr %148 : memref<1x!ascdtype.fp4x2_e2m1, 4>
                %151 = ascscalar.memref_address %150 : memref<1x!ascdtype.fp4x2_e2m1, 4>
                %152 = ascscalar.bind_local_ptr %151 : memref<?xi8, 7>
                asccube.get_buf mte1, %149, 0 {cannir.buffer.generated_sync}
                asccube.get_buf mte1, %126, 0 {cannir.buffer.generated_sync}
                asccube.get_buf mte1, %143, 0 {cannir.buffer.generated_sync}
                asccube.copy_l12l0b %150[%c0], %136[%c0], %c0_i32, %c0_i32, %c32_i32, %c2_i32, %c32_i32, %c32_i32 : memref<1x!ascdtype.fp4x2_e2m1, 4>, memref<1x!ascdtype.fp4x2_e2m1, 2>
                asccube.copy_l12l0b %152[%c0], %145[%c0], %c0_i32, %c0_i32, %c32_i32, %c2_i32, %c2_i32, %c2_i32 : memref<?xi8, 7>, memref<1x!ascdtype.fp8_e8m0, 2>
                asccube.release_buf mte1, %143, 0 {cannir.buffer.generated_sync}
                asccube.release_buf mte1, %126, 0 {cannir.buffer.generated_sync}
                asccube.release_buf mte1, %149, 0 {cannir.buffer.generated_sync}
                %153 = ascscalar.memref_address %alloc_10 : memref<16384xf32, 5>
                %154 = ascscalar.memref_address %alloc_9 : memref<16384xf32, 5>
                %155 = arith.select %122, %153, %154 : i64
                %156 = arith.select %122, %c9_i64, %c10_i64 : i64
                %157 = ascscalar.bind_local_ptr %155 : memref<1xf32, 5>
                asccube.get_buf m, %156, 0 {cannir.buffer.generated_sync}
                asccube.get_buf m, %c6_i64, 0 {cannir.buffer.generated_sync}
                asccube.get_buf m, %149, 0 {cannir.buffer.generated_sync}
                asccube.mmad %157[%c0], %91, %150 scale(%93, %152), %c32_i32, %c128_i32, %c512_i32, %c0_i32 {c_matrix_init_val = true, mx_mode = true} : memref<1xf32, 5>, memref<?x!ascdtype.fp4x2_e2m1, 3>, memref<1x!ascdtype.fp4x2_e2m1, 4>, memref<?xi8, 6>, memref<?xi8, 7>
                asccube.release_buf m, %149, 0 {cannir.buffer.generated_sync}
                asccube.release_buf m, %c6_i64, 0 {cannir.buffer.generated_sync}
                asccube.release_buf m, %156, 0 {cannir.buffer.generated_sync}
                %158 = arith.addi %121, %c4_i64 overflow<nsw> : i64
                asccube.sync_intra_wait fixpipe, %158
                %159 = arith.addi %121, %c20_i64 overflow<nsw> : i64
                asccube.sync_intra_wait fixpipe, %159
                %160 = ascscalar.bitcast_memref %alloc_7 byte_offset = %c0_i64 : memref<16384xbf16, 1> -> memref<?xbf16, 1>
                %161 = arith.muli %121, %c8192_i64 overflow<nsw> : i64
                asccube.sync_intra_wait fixpipe, %c0_i64 {cannir.sync.banks = 3 : i32}
                asccube.sync_intra_wait fixpipe, %c16_i64 {cannir.sync.banks = 3 : i32}
                asccube.get_buf fixpipe, %156, 0 {cannir.buffer.generated_sync}
                %162 = arith.index_cast %161 : i64 to index
                asccube.copy_l0c2ub %160[%162], %157[%c0], %c256_i32, %c32_i32, %c256_i32, %c32_i32, %c0_i32 sub_block_id %c0_i32 {nz2nd_en = true, relu_en = false} : memref<?xbf16, 1>, memref<1xf32, 5>
                asccube.copy_l0c2ub %160[%162], %157[%c8192], %c256_i32, %c32_i32, %c256_i32, %c32_i32, %c0_i32 sub_block_id %c1_i32 {nz2nd_en = true, relu_en = false} : memref<?xbf16, 1>, memref<1xf32, 5>
                asccube.release_buf fixpipe, %156, 0 {cannir.buffer.generated_sync}
                asccube.sync_intra_arrive fixpipe, %c0_i64 {cannir.sync.banks = 3 : i32}
                asccube.sync_intra_arrive fixpipe, %c16_i64 {cannir.sync.banks = 3 : i32}
                %163 = arith.addi %121, %c14_i64 overflow<nsw> : i64
                asccube.sync_intra_arrive fixpipe, %163
                %164 = arith.addi %121, %c30_i64 overflow<nsw> : i64
                asccube.sync_intra_arrive fixpipe, %164
                %165 = arith.addi %109, %c10_i64 overflow<nsw> : i64
                asccube.sync_intra_arrive mte2, %165
                %166 = arith.addi %109, %c26_i64 overflow<nsw> : i64
                asccube.sync_intra_arrive mte2, %166
                scf.yield %128 : i64
              } else {
                scf.yield %arg42 : i64
              }
              scf.yield %102 : i64
            } {cannir.loop_unroll = 1 : i64}
            asccube.sync_intra_wait fixpipe, %c4_i64
            asccube.sync_intra_wait fixpipe, %c20_i64
            asccube.sync_intra_wait fixpipe, %c5_i64
            asccube.sync_intra_wait fixpipe, %c21_i64
            asccube.sync_intra_wait mte2, %c6_i64
            asccube.sync_intra_wait mte2, %c22_i64
            asccube.sync_intra_arrive mte2, %c7_i64
            asccube.sync_intra_arrive mte2, %c23_i64
            %96 = arith.addi %arg37, %c1_i64 : i64
            %97 = arith.cmpi slt, %96, %34 : i64
            %98 = arith.cmpi slt, %96, %53 : i64
            %99 = arith.select %97, %98, %false : i1
            %100 = scf.if %99 -> (i64) {
              %101 = arith.addi %36, %c1_i64 : i64
              %102 = arith.index_cast %101 : i64 to index
              %103 = memref.load %alloc_0[%102] : memref<1xi32>
              %104 = arith.extsi %103 : i32 to i64
              %105 = arith.addi %104, %c63_i64 : i64
              %106 = arith.remsi %105, %c64_i64 : i64
              %107 = arith.cmpi ne, %106, %c0_i64 : i64
              %108 = arith.cmpi slt, %106, %c0_i64 : i64
              %109 = arith.andi %107, %108 : i1
              %110 = arith.divsi %105, %c64_i64 : i64
              %111 = arith.extui %109 : i1 to i64
              %112 = arith.subi %110, %111 : i64
              %113 = scf.if %33 -> (i1) {
                %121 = arith.addi %6, %c5 : index
                %122 = memref.load %alloc_6[%121] : memref<1024xi32>
                %123 = arith.extsi %122 : i32 to i64
                %124 = arith.cmpi eq, %96, %123 : i64
                scf.yield %124 : i1
              } else {
                scf.yield %false : i1
              }
              %114 = scf.if %113 -> (i64) {
                %121 = arith.addi %6, %c6 : index
                %122 = memref.load %alloc_6[%121] : memref<1024xi32>
                %123 = arith.extsi %122 : i32 to i64
                %124 = arith.minsi %112, %123 : i64
                scf.yield %124 : i64
              } else {
                scf.yield %112 : i64
              }
              %115 = arith.cmpi sgt, %103, %c0_i32 : i32
              %116 = arith.cmpi sgt, %114, %c0_i64 : i64
              %117 = arith.cmpi sgt, %29, %c0_i32 : i32
              %118 = arith.select %116, %117, %false : i1
              %119 = arith.select %115, %118, %false : i1
              %120 = scf.if %119 -> (i64) {
                %121 = arith.minsi %114, %c4_i64 : i64
                scf.yield %121 : i64
              } else {
                scf.yield %c0_i64 : i64
              }
              scf.yield %120 : i64
            } else {
              scf.yield %c0_i64 : i64
            }
            scf.yield %100, %95 : i64, i64
          } else {
            scf.yield %arg39, %arg40 : i64, i64
          }
          scf.yield %62, %67#0, %67#1 : i64, i64, i64
        } {cannir.loop_unroll = 1 : i64}
        scf.yield %35#0, %35#2 : i64, i64
      } {cannir.loop_unroll = 1 : i64}
      asccube.sync_intra_wait fixpipe, %c0_i64
      asccube.sync_intra_wait fixpipe, %c16_i64
    }
    scf.if %1 {
      asccube.sync_all
      asccube.sync_block_wait s, %c0_i64
      asccube.sync_block_arrive fixpipe, %c1_i64 {mode = 0 : i32}
      asccube.sync_block_wait s, %c1_i64 {mode = 0 : i32}
      asccube.sync_block_arrive mte3, %c2_i64
    }
    return
  }
}
