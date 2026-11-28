module {
  func.func @run_fused_cv_7f5afaa69048ca3c8fc069613a241c0c(%arg0: i64, %arg1: i64, %arg2: !cannir.tensor<<gm, f32>, <<(-1, 32)>, <(32, 1)>>, <<(-1, 32)>, <(32, 1)>>, identity>, %arg3: !cannir.tensor<<gm, !ascdtype.fp8_e8m0>, <<(-1, 2, 2)>, <(4, 2, 1)>>, <<(-1, 2, 2)>, <(4, 2, 1)>>, identity>, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg8: !cannir.tensor<<gm, i32>, <<(1)>, <(1)>>, <<(1)>, <(1)>>, identity>, %arg9: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg10: !cannir.tensor<<gm, i32>, <<(1)>, <(1)>>, <<(1)>, <(1)>>, identity>, %arg11: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg12: !cannir.tensor<<gm, ui16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg13: !cannir.tensor<<gm, ui16>, <<(-1, 1)>, <(1, 1)>>, <<(-1, 1)>, <(1, 1)>>, identity>, %arg14: i64, %arg15: i64, %arg16: i64, %arg17: i64, %arg18: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg19: i64, %arg20: i64, %arg21: i64, %arg22: i64, %arg23: i64, %arg24: i64, %arg25: i64, %arg26: !cannir.tensor<<gm, i32>, <<(128, 8)>, <(8, 1)>>, <<(128, 8)>, <(8, 1)>>, identity>, %arg27: i64, %arg28: i64, %arg29: i64, %arg30: i64, %arg31: i64, %arg32: i64, %arg33: i64, %arg34: i64) attributes {cannbotdsl.contract_digest = "7f5afaa69048ca3c8fc069613a241c0c94347b212d13fbeda9d49e087f4239f5", cannbotdsl.public_slot_count = 35 : i64, cannbotdsl.slot_dtype_bits = [64, 64, 32, 8, 64, 64, 64, 32, 32, 32, 32, 32, 16, 16, 64, 64, 64, 64, 32, 64, 64, 64, 64, 64, 64, 64, 32, 64, 64, 64, 64, 64, 64, 64, 64], cannbotdsl.slot_dtype_codes = [0, 0, 2, 14, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], cannbotdsl.slot_dtype_lanes = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], cannbotdsl.slot_flags = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], cannbotdsl.slot_kinds = [2, 2, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2], cannbotdsl.slot_type_digests = ["385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "303efa15a10b189586194340f9acf94f49608a8d0ff96167ef4f24d50ee66e03", "f23a2a226c424d9c13d010802f1fd872bf7fbb10f043d43b23f2b47c05c692f9", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "3ab0f664517f4ad5f7ad3356ae93b74a5b3087536c7a220473c3a09bc7daf970", "4b2bde939157f01ee36c4a9b4273cfa42144eb3ab3f724764129d109a5681d48", "e030c76f69ec395f0341ef8f9ab18ddc826e6df4047f270d22f84aa569a4ad93", "4b2bde939157f01ee36c4a9b4273cfa42144eb3ab3f724764129d109a5681d48", "3ab0f664517f4ad5f7ad3356ae93b74a5b3087536c7a220473c3a09bc7daf970", "6496eec66a7bf3a9d4c54c191f0e75da589cc0dbfc6a632b4b46d27558c76145", "378ec3543313de23ded0ca5588951a4643908cb21434a4e8dc8960565eb08191", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "5e94aa106f0584536ad0dd76ba40f0377933e025b281f1fa3d2cccc3b499ba2d", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "cebc6a6e5577b0a5f53b2408bb132f57e156393513d83c69eedf6e9e026ef584", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515"], cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.host} {
    cannir.kernel_launch @kernel_Quant_lightning_indexer_0(%arg19) [%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18, %arg19, %arg20, %arg21, %arg22, %arg23, %arg24, %arg25, %arg26, %arg27, %arg28, %arg29, %arg30, %arg31, %arg32, %arg33, %arg34] {cannbotdsl.profile = {indices = array<i64: 2, 3, 7, 9, 11, 12, 12, 26>, name = "Quant_lightning_indexer", op_type = "Quant_lightning_indexer", roles = array<i64: 0, 0, 0, 0, 0, 0, 1, 0>}} : i64, i64, !cannir.tensor<<gm, f32>, <<(-1, 32)>, <(32, 1)>>, <<(-1, 32)>, <(32, 1)>>, identity>, !cannir.tensor<<gm, !ascdtype.fp8_e8m0>, <<(-1, 2, 2)>, <(4, 2, 1)>>, <<(-1, 2, 2)>, <(4, 2, 1)>>, identity>, i64, i64, i64, !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, !cannir.tensor<<gm, i32>, <<(1)>, <(1)>>, <<(1)>, <(1)>>, identity>, !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, !cannir.tensor<<gm, i32>, <<(1)>, <(1)>>, <<(1)>, <(1)>>, identity>, !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, !cannir.tensor<<gm, ui16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, !cannir.tensor<<gm, ui16>, <<(-1, 1)>, <(1, 1)>>, <<(-1, 1)>, <(1, 1)>>, identity>, i64, i64, i64, i64, !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, i64, i64, i64, i64, i64, i64, i64, !cannir.tensor<<gm, i32>, <<(128, 8)>, <(8, 1)>>, <<(128, 8)>, <(8, 1)>>, identity>, i64, i64, i64, i64, i64, i64, i64, i64
    return
  }
  func.func @kernel_Quant_lightning_indexer_0_aiv(%arg0: i64, %arg1: i64, %arg2: !cannir.tensor<<gm, f32>, <<(-1, 32)>, <(32, 1)>>, <<(-1, 32)>, <(32, 1)>>, identity>, %arg3: !cannir.tensor<<gm, !ascdtype.fp8_e8m0>, <<(-1, 2, 2)>, <(4, 2, 1)>>, <<(-1, 2, 2)>, <(4, 2, 1)>>, identity>, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg8: !cannir.tensor<<gm, i32>, <<(1)>, <(1)>>, <<(1)>, <(1)>>, identity>, %arg9: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg10: !cannir.tensor<<gm, i32>, <<(1)>, <(1)>>, <<(1)>, <(1)>>, identity>, %arg11: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg12: !cannir.tensor<<gm, ui16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg13: !cannir.tensor<<gm, ui16>, <<(-1, 1)>, <(1, 1)>>, <<(-1, 1)>, <(1, 1)>>, identity>, %arg14: i64, %arg15: i64, %arg16: i64, %arg17: i64, %arg18: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg19: i64, %arg20: i64, %arg21: i64, %arg22: i64, %arg23: i64, %arg24: i64, %arg25: i64, %arg26: !cannir.tensor<<gm, i32>, <<(128, 8)>, <(8, 1)>>, <<(128, 8)>, <(8, 1)>>, identity>, %arg27: i64, %arg28: i64, %arg29: i64, %arg30: i64, %arg31: i64, %arg32: i64, %arg33: i64, %arg34: i64) attributes {cannir.buffer.auto_sync, cannir.buffer.sync_analyzed, cannir.buffer.sync_verified, cannir.buffer.tensor_state, cannir.core = "aiv", cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.device, cannir.explicit_ring_buffer_cursors, cannir.kernel_group = "kernel_Quant_lightning_indexer_0", cannir.resource.ids_prepared} {
    %c38144 = arith.constant 38144 : index
    %c59904 = arith.constant 59904 : index
    %c16384_i32 = arith.constant 16384 : i32
    %c32768_i32 = arith.constant 32768 : i32
    %c3 = arith.constant 3 : index
    %c6 = arith.constant 6 : index
    %c5 = arith.constant 5 : index
    %c1 = arith.constant 1 : index
    %c7 = arith.constant 7 : index
    %c8 = arith.constant 8 : index
    %c288 = arith.constant 288 : index
    %c0_i32 = arith.constant 0 : i32
    %c-1_i32 = arith.constant -1 : i32
    %c256_i64 = arith.constant 256 : i64
    %false = arith.constant false
    %true = arith.constant true
    %c8192_i64 = arith.constant 8192 : i64
    %c64_i64 = arith.constant 64 : i64
    %c3_i32 = arith.constant 3 : i32
    %c128_i64 = arith.constant 128 : i64
    %c110592_i64 = arith.constant 110592 : i64
    %c32_i32 = arith.constant 32 : i32
    %c0 = arith.constant 0 : index
    %c128_i32 = arith.constant 128 : i32
    %c16384_i64 = arith.constant 16384 : i64
    %c131072_i64 = arith.constant 131072 : i64
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c906240_i64 = arith.constant 906240 : i64
    %c918528_i64 = arith.constant 918528 : i64
    %c256_i32 = arith.constant 256 : i32
    %c512_i64 = arith.constant 512 : i64
    %c127_i64 = arith.constant 127 : i64
    %c512_i32 = arith.constant 512 : i32
    %c1024_i64 = arith.constant 1024 : i64
    %c255_i64 = arith.constant 255 : i64
    %c8192_i32 = arith.constant 8192 : i32
    %c32768_i64 = arith.constant 32768 : i64
    %c384_i64 = arith.constant 384 : i64
    %c147456_i64 = arith.constant 147456 : i64
    %c65536_i64 = arith.constant 65536 : i64
    %c212992_i64 = arith.constant 212992 : i64
    %c63_i64 = arith.constant 63 : i64
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c-8191_i64 = arith.constant -8191 : i64
    %c262144_i64 = arith.constant 262144 : i64
    %c16639_i64 = arith.constant 16639 : i64
    %c8193_i64 = arith.constant 8193 : i64
    %c16511_i64 = arith.constant 16511 : i64
    %c36_i64 = arith.constant 36 : i64
    %c16_i64 = arith.constant 16 : i64
    %c14_i64 = arith.constant 14 : i64
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
    %c218112_i64 = arith.constant 218112 : i64
    %c98304_i64 = arith.constant 98304 : i64
    %alloc = memref.alloc() {cannir.arg_index = 2 : i64} : memref<1xf32>
    %alloc_0 = memref.alloc() {cannir.arg_index = 7 : i64} : memref<1xi32>
    %alloc_1 = memref.alloc() {cannir.arg_index = 9 : i64} : memref<1xi32>
    %alloc_2 = memref.alloc() {cannir.arg_index = 12 : i64} : memref<1xui16>
    %alloc_3 = memref.alloc() {cannir.arg_index = 26 : i64} : memref<1024xi32>
    %alloc_4 = memref.alloc() {cannir.byte_offset = 0 : i64, cannir.onchip_space_root} : memref<262144xui8, 1>
    %0 = memref.load %alloc_3[%c288] : memref<1024xi32>
    %1 = arith.cmpi ne, %0, %c0_i32 : i32
    %2 = ascscalar.bind_gm_ptr %arg14 : memref<3145728xi32>
    %3 = ascscalar.bind_gm_ptr %arg15 : memref<3145728xui16>
    %4 = ascscalar.get_subblock_id
    %5 = ascscalar.get_block_idx
    %6 = arith.index_cast %5 : i64 to index
    %7 = arith.muli %6, %c8 : index
    %8 = arith.addi %7, %c7 : index
    %9 = memref.load %alloc_3[%8] : memref<1024xi32>
    %10 = arith.extsi %9 : i32 to i64
    %11 = memref.load %alloc_3[%7] : memref<1024xi32>
    %12 = arith.cmpi ne, %11, %c0_i32 : i32
    scf.if %12 {
      ascvec.sync_intra_arrive v, %c1_i64
      ascvec.sync_intra_arrive v, %c0_i64
      %13 = arith.addi %7, %c1 : index
      %14 = memref.load %alloc_3[%13] : memref<1024xi32>
      %15 = arith.extsi %14 : i32 to i64
      %16 = arith.addi %7, %c4 : index
      %17 = memref.load %alloc_3[%16] : memref<1024xi32>
      %18 = arith.extsi %17 : i32 to i64
      %19 = arith.addi %18, %c1_i64 : i64
      %20 = arith.minsi %19, %arg21 : i64
      %21:7 = scf.for %arg35 = %15 to %20 step %c1_i64 iter_args(%arg36 = %c-1_i32, %arg37 = %10, %arg38 = %c0_i64, %arg39 = %c0_i64, %arg40 = %c0_i64, %arg41 = %c0_i64, %arg42 = %c0_i64) -> (i32, i64, i64, i64, i64, i64, i64)  : i64 {
        %22 = arith.index_cast %arg35 : i64 to index
        %23 = memref.load %alloc_0[%22] : memref<1xi32>
        %24 = arith.extsi %23 : i32 to i64
        %25 = arith.addi %arg35, %c1_i64 : i64
        %26 = arith.index_cast %25 : i64 to index
        %27 = memref.load %alloc_0[%26] : memref<1xi32>
        %28 = arith.extsi %27 : i32 to i64
        %29 = arith.subi %28, %24 : i64
        %30 = arith.trunci %29 : i64 to i32
        %31 = memref.load %alloc_1[%22] : memref<1xi32>
        %32 = arith.addi %29, %c5_i64 : i64
        %33 = arith.remsi %32, %c6_i64 : i64
        %34 = arith.cmpi ne, %33, %c0_i64 : i64
        %35 = arith.cmpi slt, %33, %c0_i64 : i64
        %36 = arith.andi %34, %35 : i1
        %37 = arith.divsi %32, %c6_i64 : i64
        %38 = arith.extui %36 : i1 to i64
        %39 = arith.subi %37, %38 : i64
        %40 = arith.cmpi eq, %arg35, %15 : i64
        %41 = scf.if %40 -> (i64) {
          %45 = arith.addi %7, %c2 : index
          %46 = memref.load %alloc_3[%45] : memref<1024xi32>
          %47 = arith.extsi %46 : i32 to i64
          scf.yield %47 : i64
        } else {
          scf.yield %c0_i64 : i64
        }
        %42 = arith.cmpi eq, %arg35, %18 : i64
        %43 = scf.if %42 -> (i64) {
          %45 = arith.addi %7, %c5 : index
          %46 = memref.load %alloc_3[%45] : memref<1024xi32>
          %47 = arith.extsi %46 : i32 to i64
          %48 = arith.addi %7, %c6 : index
          %49 = memref.load %alloc_3[%48] : memref<1024xi32>
          %50 = arith.cmpi sgt, %49, %c0_i32 : i32
          %51 = arith.extui %50 : i1 to i64
          %52 = arith.addi %47, %51 : i64
          scf.yield %52 : i64
        } else {
          scf.yield %39 : i64
        }
        %44:7 = scf.for %arg43 = %41 to %43 step %c1_i64 iter_args(%arg44 = %arg36, %arg45 = %arg37, %arg46 = %arg38, %arg47 = %arg39, %arg48 = %arg40, %arg49 = %arg41, %arg50 = %arg42) -> (i32, i64, i64, i64, i64, i64, i64)  : i64 {
          %45 = arith.muli %arg43, %c6_i64 : i64
          %46 = arith.trunci %45 : i64 to i32
          %47 = arith.extsi %46 : i32 to i64
          %48 = arith.addi %24, %47 : i64
          %49 = arith.trunci %48 : i64 to i32
          %50 = arith.extsi %49 : i32 to i64
          %51 = arith.subi %28, %50 : i64
          %52 = arith.minsi %51, %c6_i64 : i64
          %53 = arith.trunci %52 : i64 to i32
          %54 = arith.subi %30, %46 : i32
          %55 = arith.maxsi %54, %c0_i32 : i32
          %56 = arith.minsi %53, %55 : i32
          %57 = arith.cmpi eq, %56, %c0_i32 : i32
          %58 = arith.select %57, %c0_i32, %31 : i32
          %59 = arith.extsi %58 : i32 to i64
          %60 = arith.addi %59, %c255_i64 : i64
          %61 = arith.remsi %60, %c256_i64 : i64
          %62 = arith.cmpi ne, %61, %c0_i64 : i64
          %63 = arith.cmpi slt, %61, %c0_i64 : i64
          %64 = arith.andi %62, %63 : i1
          %65 = arith.divsi %60, %c256_i64 : i64
          %66 = arith.extui %64 : i1 to i64
          %67 = arith.subi %65, %66 : i64
          %68 = arith.cmpi eq, %arg43, %41 : i64
          %69 = arith.select %40, %68, %false : i1
          %70 = scf.if %69 -> (i64) {
            %97 = arith.addi %7, %c3 : index
            %98 = memref.load %alloc_3[%97] : memref<1024xi32>
            %99 = arith.extsi %98 : i32 to i64
            scf.yield %99 : i64
          } else {
            scf.yield %c0_i64 : i64
          }
          %71 = scf.if %42 -> (i1) {
            %97 = arith.addi %7, %c5 : index
            %98 = memref.load %alloc_3[%97] : memref<1024xi32>
            %99 = arith.extsi %98 : i32 to i64
            %100 = arith.cmpi eq, %arg43, %99 : i64
            scf.yield %100 : i1
          } else {
            scf.yield %false : i1
          }
          %72 = scf.if %71 -> (i64) {
            %97 = arith.addi %7, %c6 : index
            %98 = memref.load %alloc_3[%97] : memref<1024xi32>
            %99 = arith.extsi %98 : i32 to i64
            %100 = arith.minsi %67, %99 : i64
            scf.yield %100 : i64
          } else {
            scf.yield %67 : i64
          }
          %73 = arith.cmpi sgt, %70, %c0_i64 : i64
          %74 = arith.cmpi slt, %72, %67 : i64
          %75 = arith.select %73, %true, %74 : i1
          %76 = arith.muli %arg45, %c6_i64 : i64
          %77 = arith.subi %76, %50 : i64
          %78 = arith.muli %77, %c32768_i64 : i64
          %79 = arith.addi %arg14, %78 : i64
          %80 = arith.muli %77, %c16384_i64 : i64
          %81 = arith.addi %arg15, %80 : i64
          %82 = arith.select %75, %79, %arg28 : i64
          %83 = arith.select %75, %81, %arg29 : i64
          %84 = arith.muli %70, %c256_i64 : i64
          %85 = arith.trunci %84 : i64 to i32
          %86 = arith.cmpi sgt, %58, %85 : i32
          %87 = arith.cmpi sgt, %72, %70 : i64
          %88 = arith.select %86, %87, %false : i1
          %89:5 = scf.if %88 -> (i64, i64, i64, i64, i64) {
            %97 = arith.cmpi ne, %75, %false : i1
            %98 = arith.select %97, %c1_i64, %arg32 : i64
            %99 = arith.subi %72, %70 : i64
            %100 = arith.muli %99, %c256_i64 : i64
            %101 = arith.cmpi sge, %arg44, %c0_i32 : i32
            %102 = arith.cmpi ne, %arg44, %c0_i32 : i32
            %103 = arith.select %101, %102, %false : i1
            scf.if %103 {
              ascvec.sync_intra_wait mte3, %c14_i64
              ascvec.sync_all
              ascvec.sync_intra_arrive mte3, %c14_i64
            }
            ascvec.sync_all
            %104 = arith.addi %arg46, %c1_i64 : i64
            %105 = arith.remui %104, %c2_i64 : i64
            %106 = arith.muli %4, %c3_i64 : i64
            %107 = arith.addi %50, %106 : i64
            %108 = arith.trunci %4 : i64 to i32
            %109 = arith.muli %108, %c3_i32 : i32
            %110 = arith.cmpi sge, %109, %53 : i32
            %111 = arith.select %110, %50, %107 : i64
            %112 = ascscalar.bitcast_memref %alloc_4 byte_offset = %c0_i64 : memref<262144xui8, 1> -> memref<?xf32, 1>
            %113 = cannir.extract_layout %arg2 : !cannir.tensor<<gm, f32>, <<(-1, 32)>, <(32, 1)>>, <<(-1, 32)>, <(32, 1)>>, identity>
            %114 = cannir.extract_shape %113 : <<(-1, 32)>, <(32, 1)>> -> <(-1, 32)>
            %115 = cannir.extract_dim %114[0] : !cannir.shape<(-1, 32)>
            %116 = arith.subi %115, %111 : i64
            %117 = arith.divsi %116, %c3_i64 : i64
            %118 = arith.minsi %117, %c0_i64 : i64
            %119 = arith.muli %118, %c3_i64 : i64
            %120 = arith.subi %116, %119 : i64
            %121 = arith.minsi %120, %c3_i64 : i64
            %122 = arith.cmpi sge, %117, %c0_i64 : i64
            %123 = arith.select %122, %121, %c0_i64 : i64
            %124 = arith.muli %111, %c128_i64 : i64
            %125 = arith.muli %123, %c128_i64 : i64
            %126 = arith.cmpi eq, %125, %c0_i64 : i64
            %127 = arith.cmpi ne, %125, %c0_i64 : i64
            %128 = arith.extui %127 : i1 to i64
            %129 = arith.select %126, %c4_i64, %125 : i64
            ascvec.get_buf mte2, %c3_i64, 0 {cannir.buffer.generated_sync}
            "ascvec.copy_gm2ub_plan"(%112, %alloc, %c0_i32, %c0_i32, %c110592_i64, %124, %128, %129, %c0_i64, %129, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 0 : i64, ub_capacity = 262144 : i64}> : (memref<?xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
            ascvec.release_buf mte2, %c3_i64, 0 {cannir.buffer.generated_sync}
            %130 = ascscalar.memref_address %alloc_4 : memref<262144xui8, 1>
            %131 = arith.addi %130, %c110592_i64 : i64
            %132 = ascscalar.bind_local_ptr %131 : memref<96xf32, 1>
            ascvec.get_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
            %133 = ascscalar.bitcast_memref %alloc_4 byte_offset = %c0_i64 : memref<262144xui8, 1> -> memref<?xbf16, 1>
            ascvec.vf(%132, %133 : memref<96xf32, 1>, memref<?xbf16, 1>) {
            ^bb0(%arg51: memref<96xf32, 1>, %arg52: memref<?xbf16, 1>):
              %c55744 = arith.constant 55744 : index
              %c64 = arith.constant 64 : index
              %c55616 = arith.constant 55616 : index
              %c32 = arith.constant 32 : index
              %c55488 = arith.constant 55488 : index
              %c0_5 = arith.constant 0 : index
              %c32_i32_6 = arith.constant 32 : i32
              %c0_7 = arith.constant 0 : index
              %c1_8 = arith.constant 1 : index
              scf.for %arg53 = %c0_7 to %c1_8 step %c1_8 {
                %mask, %remaining_out = ascvec.update_mask %c32_i32_6 : !ascvec.mask
                %mask_9, %remaining_out_10 = ascvec.update_mask %c32_i32_6 elem_bits 16 : !ascvec.mask
                %143 = ascvec.loadalign %arg51[%c0_5] : memref<96xf32, 1>, index -> <f32>
                %144 = ascvec.cast %143, %mask {reg_layout = 0 : i32} : <f32> -> <bf16>
                %145 = ascvec.reinterpret %144 : <bf16> -> <ui32>
                %146 = ascvec.pack %145 src_elem_bits 32 dst_elem_bits 16 {part = "low"} : <ui32> -> <ui16>
                %147 = ascvec.reinterpret %146 : <ui16> -> <bf16>
                ascvec.storealign %arg52[%c55488], %147, %mask_9 : memref<?xbf16, 1>, index, <bf16>
                %148 = ascvec.loadalign %arg51[%c32] : memref<96xf32, 1>, index -> <f32>
                %149 = ascvec.cast %148, %mask {reg_layout = 0 : i32} : <f32> -> <bf16>
                %150 = ascvec.reinterpret %149 : <bf16> -> <ui32>
                %151 = ascvec.pack %150 src_elem_bits 32 dst_elem_bits 16 {part = "low"} : <ui32> -> <ui16>
                %152 = ascvec.reinterpret %151 : <ui16> -> <bf16>
                ascvec.storealign %arg52[%c55616], %152, %mask_9 : memref<?xbf16, 1>, index, <bf16>
                %153 = ascvec.loadalign %arg51[%c64] : memref<96xf32, 1>, index -> <f32>
                %154 = ascvec.cast %153, %mask {reg_layout = 0 : i32} : <f32> -> <bf16>
                %155 = ascvec.reinterpret %154 : <bf16> -> <ui32>
                %156 = ascvec.pack %155 src_elem_bits 32 dst_elem_bits 16 {part = "low"} : <ui32> -> <ui16>
                %157 = ascvec.reinterpret %156 : <ui16> -> <bf16>
                ascvec.storealign %arg52[%c55744], %157, %mask_9 : memref<?xbf16, 1>, index, <bf16>
              }
              ascvec.mem_bar vst_vld
            } {cannir.vf_mode = "simd"}
            ascvec.release_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
            %134 = ascscalar.bitcast_memref %alloc_4 byte_offset = %c0_i64 : memref<262144xui8, 1> -> memref<?xui16, 1>
            ascvec.vf(%134, %134 : memref<?xui16, 1>, memref<?xui16, 1>) {
            ^bb0(%arg51: memref<?xui16, 1>, %arg52: memref<?xui16, 1>):
              %c59776 = arith.constant 59776 : index
              %c59648 = arith.constant 59648 : index
              %c59520 = arith.constant 59520 : index
              %c59392 = arith.constant 59392 : index
              %c59264 = arith.constant 59264 : index
              %c59136 = arith.constant 59136 : index
              %c59008 = arith.constant 59008 : index
              %c58880 = arith.constant 58880 : index
              %c58752 = arith.constant 58752 : index
              %c58624 = arith.constant 58624 : index
              %c58496 = arith.constant 58496 : index
              %c58368 = arith.constant 58368 : index
              %c0_i16 = arith.constant 0 : i16
              %c128_i32_5 = arith.constant 128 : i32
              %c0_6 = arith.constant 0 : index
              %c1_7 = arith.constant 1 : index
              scf.for %arg53 = %c0_6 to %c1_7 step %c1_7 {
                %mask, %remaining_out = ascvec.update_mask %c128_i32_5 elem_bits 16 : !ascvec.mask
                %143 = ascvec.duplicate_scalar %c0_i16, %mask : i16 -> <ui16>
                ascvec.storealign %arg51[%c58368], %143, %mask : memref<?xui16, 1>, index, <ui16>
                ascvec.storealign %arg52[%c58496], %143, %mask : memref<?xui16, 1>, index, <ui16>
                ascvec.storealign %arg51[%c58624], %143, %mask : memref<?xui16, 1>, index, <ui16>
                ascvec.storealign %arg52[%c58752], %143, %mask : memref<?xui16, 1>, index, <ui16>
                ascvec.storealign %arg51[%c58880], %143, %mask : memref<?xui16, 1>, index, <ui16>
                ascvec.storealign %arg52[%c59008], %143, %mask : memref<?xui16, 1>, index, <ui16>
                ascvec.storealign %arg51[%c59136], %143, %mask : memref<?xui16, 1>, index, <ui16>
                ascvec.storealign %arg52[%c59264], %143, %mask : memref<?xui16, 1>, index, <ui16>
                ascvec.storealign %arg51[%c59392], %143, %mask : memref<?xui16, 1>, index, <ui16>
                ascvec.storealign %arg52[%c59520], %143, %mask : memref<?xui16, 1>, index, <ui16>
                ascvec.storealign %arg51[%c59648], %143, %mask : memref<?xui16, 1>, index, <ui16>
                ascvec.storealign %arg52[%c59776], %143, %mask : memref<?xui16, 1>, index, <ui16>
              }
              ascvec.mem_bar vst_vld
            } {cannir.vf_mode = "simd"}
            %135 = arith.cmpi sge, %100, %c8192_i64 : i64
            %136 = arith.cmpi sle, %100, %c16384_i64 : i64
            %137 = arith.select %135, %136, %false : i1
            %138 = arith.extui %137 : i1 to i32
            ascvec.sync_intra_arrive v, %c8_i64
            ascvec.sync_intra_arrive v, %c9_i64
            %139 = arith.addi %70, %c3_i64 : i64
            %140 = arith.minsi %72, %139 : i64
            %141 = scf.for %arg51 = %70 to %140 step %c1_i64 iter_args(%arg52 = %arg47) -> (i64)  : i64 {
              %143 = arith.addi %arg52, %c1_i64 : i64
              %144 = arith.remui %143, %c4_i64 : i64
              scf.yield %144 : i64
            } {cannir.loop_unroll = 1 : i64}
            %142:4 = scf.for %arg51 = %70 to %72 step %c1_i64 iter_args(%arg52 = %141, %arg53 = %arg48, %arg54 = %arg49, %arg55 = %arg50) -> (i64, i64, i64, i64)  : i64 {
              %143 = arith.addi %arg51, %c3_i64 : i64
              %144 = arith.cmpi slt, %143, %72 : i64
              %145 = scf.if %144 -> (i64) {
                %155 = arith.addi %arg52, %c1_i64 : i64
                %156 = arith.remui %155, %c4_i64 : i64
                scf.yield %156 : i64
              } else {
                scf.yield %arg52 : i64
              }
              %146 = arith.subi %arg51, %70 : i64
              %147 = arith.remsi %146, %c2_i64 : i64
              %148 = arith.cmpi ne, %147, %c0_i64 : i64
              %149 = arith.cmpi slt, %147, %c0_i64 : i64
              %150 = arith.andi %148, %149 : i1
              %151 = arith.select %150, %c2_i64, %c0_i64 : i64
              %152 = arith.addi %147, %151 overflow<nsw> : i64
              %153 = arith.cmpi eq, %152, %c0_i64 : i64
              %154:3 = scf.if %153 -> (i64, i64, i64) {
                %155 = arith.muli %arg51, %c256_i64 : i64
                %156 = arith.cmpi eq, %arg54, %c0_i64 : i64
                %157 = arith.addi %arg54, %c1_i64 : i64
                %158 = arith.remui %157, %c2_i64 : i64
                %159 = arith.addi %arg53, %c1_i64 : i64
                %160 = arith.remui %159, %c4_i64 : i64
                %161 = arith.addi %arg55, %c1_i64 : i64
                %162 = arith.remui %161, %c2_i64 : i64
                %163 = arith.addi %162, %c1_i64 overflow<nsw> : i64
                %164 = arith.remui %163, %c2_i64 : i64
                ascvec.sync_intra_wait v, %c8_i64
                %165 = arith.subi %155, %84 : i64
                %166 = arith.subi %72, %arg51 : i64
                %167 = arith.muli %166, %c256_i64 : i64
                %168 = arith.minsi %167, %c256_i64 : i64
                %169 = arith.cmpi slt, %109, %56 : i32
                %170 = arith.select %169, %c32_i32, %c0_i32 : i32
                %171 = arith.cmpi sge, %109, %56 : i32
                %172 = arith.select %171, %c0_i32, %31 : i32
                %173 = arith.trunci %168 : i64 to i32
                %174 = arith.trunci %155 : i64 to i32
                %175 = arith.subi %172, %174 : i32
                %176 = arith.minsi %173, %175 : i32
                %177 = arith.maxsi %176, %c0_i32 : i32
                %178 = arith.addi %109, %c1_i32 : i32
                %179 = arith.cmpi sge, %178, %56 : i32
                %180 = arith.select %179, %c0_i32, %31 : i32
                %181 = arith.subi %180, %174 : i32
                %182 = arith.minsi %173, %181 : i32
                %183 = arith.maxsi %182, %c0_i32 : i32
                %184 = arith.addi %109, %c2_i32 : i32
                %185 = arith.cmpi sge, %184, %56 : i32
                %186 = arith.select %185, %c0_i32, %31 : i32
                %187 = arith.subi %186, %174 : i32
                %188 = arith.minsi %173, %187 : i32
                %189 = arith.maxsi %188, %c0_i32 : i32
                %190 = arith.cmpi eq, %138, %c0_i32 : i32
                scf.if %190 {
                  %221 = arith.select %156, %c906240_i64, %c918528_i64 : i64
                  %222 = arith.divui %221, %c8_i64 : i64
                  %223 = arith.addi %130, %222 : i64
                  %224 = ascscalar.bind_local_ptr %223 : memref<768xui16, 1>
                  %225 = arith.select %156, %c4_i64, %c5_i64 : i64
                  ascvec.sync_intra_wait v, %c0_i64 {cannir.sync.banks = 3 : i32}
                  ascvec.get_buf v, %225, 0 {cannir.buffer.generated_sync}
                  ascvec.vf(%133, %133, %170, %224 : memref<?xbf16, 1>, memref<?xbf16, 1>, i32, memref<768xui16, 1>) {
                  ^bb0(%arg56: memref<?xbf16, 1>, %arg57: memref<?xbf16, 1>, %arg58: i32, %arg59: memref<768xui16, 1>):
                    %c640 = arith.constant 640 : index
                    %c512 = arith.constant 512 : index
                    %c384 = arith.constant 384 : index
                    %c256 = arith.constant 256 : index
                    %c128 = arith.constant 128 : index
                    %c55744 = arith.constant 55744 : index
                    %c55616 = arith.constant 55616 : index
                    %c-1_i16 = arith.constant -1 : i16
                    %c-32768_i16 = arith.constant -32768 : i16
                    %c1_i32_5 = arith.constant 1 : i32
                    %c0_i32_6 = arith.constant 0 : i32
                    %c64_i32 = arith.constant 64 : i32
                    %c32_i32_7 = arith.constant 32 : i32
                    %c24576_i32 = arith.constant 24576 : i32
                    %c256_i32_8 = arith.constant 256 : i32
                    %cst = arith.constant 0.000000e+00 : bf16
                    %c0_9 = arith.constant 0 : index
                    %c55488 = arith.constant 55488 : index
                    %c128_i32_10 = arith.constant 128 : i32
                    %mask, %remaining_out = ascvec.update_mask %c128_i32_10 elem_bits 16 : !ascvec.mask
                    %226 = ascvec.loadalign %arg56[%c55488] : memref<?xbf16, 1>, index -> <bf16>
                    %227 = ascvec.duplicate_scalar %cst, %mask : bf16 -> <bf16>
                    %228 = ascvec.loadalign %arg56[%c55616] : memref<?xbf16, 1>, index -> <bf16>
                    %229 = ascvec.loadalign %arg56[%c55744] : memref<?xbf16, 1>, index -> <bf16>
                    %230:6 = scf.for %arg60 = %c0_i32_6 to %arg58 step %c1_i32_5 iter_args(%arg61 = %227, %arg62 = %227, %arg63 = %227, %arg64 = %227, %arg65 = %227, %arg66 = %227) -> (!ascvec.vreg<bf16>, !ascvec.vreg<bf16>, !ascvec.vreg<bf16>, !ascvec.vreg<bf16>, !ascvec.vreg<bf16>, !ascvec.vreg<bf16>)  : i32 {
                      %231 = arith.trunci %arg60 : i32 to i16
                      %232 = ascvec.duplicate_scalar %231, %mask : i16 -> <ui16>
                      %233 = ascvec.gather_reg %226, %232 : <bf16>, <ui16> -> <bf16>
                      %234 = ascvec.gather_reg %228, %232 : <bf16>, <ui16> -> <bf16>
                      %235 = ascvec.gather_reg %229, %232 : <bf16>, <ui16> -> <bf16>
                      %236 = arith.muli %arg60, %c256_i32_8 : i32
                      %237 = arith.index_cast %236 : i32 to index
                      %238 = ascvec.loadalign %arg57[%237] : memref<?xbf16, 1>, index -> <bf16>
                      %239 = ascvec.madd_two_address %238, %233, %arg61, %mask : <bf16>
                      %240 = arith.addi %236, %c24576_i32 : i32
                      %241 = arith.index_cast %240 : i32 to index
                      %242 = ascvec.loadalign %arg57[%241] : memref<?xbf16, 1>, index -> <bf16>
                      %243 = ascvec.madd_two_address %242, %233, %arg62, %mask : <bf16>
                      %244 = arith.addi %arg60, %c32_i32_7 : i32
                      %245 = arith.muli %244, %c256_i32_8 : i32
                      %246 = arith.index_cast %245 : i32 to index
                      %247 = ascvec.loadalign %arg57[%246] : memref<?xbf16, 1>, index -> <bf16>
                      %248 = ascvec.madd_two_address %247, %234, %arg63, %mask : <bf16>
                      %249 = arith.addi %245, %c24576_i32 : i32
                      %250 = arith.index_cast %249 : i32 to index
                      %251 = ascvec.loadalign %arg57[%250] : memref<?xbf16, 1>, index -> <bf16>
                      %252 = ascvec.madd_two_address %251, %234, %arg64, %mask : <bf16>
                      %253 = arith.addi %arg60, %c64_i32 : i32
                      %254 = arith.muli %253, %c256_i32_8 : i32
                      %255 = arith.index_cast %254 : i32 to index
                      %256 = ascvec.loadalign %arg57[%255] : memref<?xbf16, 1>, index -> <bf16>
                      %257 = ascvec.madd_two_address %256, %235, %arg65, %mask : <bf16>
                      %258 = arith.addi %254, %c24576_i32 : i32
                      %259 = arith.index_cast %258 : i32 to index
                      %260 = ascvec.loadalign %arg57[%259] : memref<?xbf16, 1>, index -> <bf16>
                      %261 = ascvec.madd_two_address %260, %235, %arg66, %mask : <bf16>
                      scf.yield %239, %243, %248, %252, %257, %261 : !ascvec.vreg<bf16>, !ascvec.vreg<bf16>, !ascvec.vreg<bf16>, !ascvec.vreg<bf16>, !ascvec.vreg<bf16>, !ascvec.vreg<bf16>
                    } {cannir.loop_unroll = 4 : i64}
                    %c0_11 = arith.constant 0 : index
                    %c1_12 = arith.constant 1 : index
                    scf.for %arg60 = %c0_11 to %c1_12 step %c1_12 {
                      %231 = ascvec.add_scalar %230#0, %cst, %mask : <bf16>, bf16
                      %232 = ascvec.reinterpret %231 : <bf16> -> <ui16>
                      %233 = ascvec.duplicate_scalar %c-32768_i16, %mask : i16 -> <ui16>
                      %234 = ascvec.and %232, %233, %mask : <ui16>, <ui16> -> <ui16>
                      %235 = ascvec.cmp_eq_scalar %234, %c-32768_i16, %mask : <ui16>, i16 -> !ascvec.mask
                      %236 = ascvec.xor %232, %233, %mask : <ui16>, <ui16> -> <ui16>
                      %237 = ascvec.duplicate_scalar %c-1_i16, %mask : i16 -> <ui16>
                      %238 = ascvec.xor %232, %237, %mask : <ui16>, <ui16> -> <ui16>
                      %239 = ascvec.select %238, %236, %235 : <ui16>
                      ascvec.storealign %arg59[%c0_9], %239, %mask : memref<768xui16, 1>, index, <ui16>
                      %240 = ascvec.add_scalar %230#1, %cst, %mask : <bf16>, bf16
                      %241 = ascvec.reinterpret %240 : <bf16> -> <ui16>
                      %242 = ascvec.and %241, %233, %mask : <ui16>, <ui16> -> <ui16>
                      %243 = ascvec.cmp_eq_scalar %242, %c-32768_i16, %mask : <ui16>, i16 -> !ascvec.mask
                      %244 = ascvec.xor %241, %233, %mask : <ui16>, <ui16> -> <ui16>
                      %245 = ascvec.xor %241, %237, %mask : <ui16>, <ui16> -> <ui16>
                      %246 = ascvec.select %245, %244, %243 : <ui16>
                      ascvec.storealign %arg59[%c128], %246, %mask : memref<768xui16, 1>, index, <ui16>
                      %247 = ascvec.add_scalar %230#2, %cst, %mask : <bf16>, bf16
                      %248 = ascvec.reinterpret %247 : <bf16> -> <ui16>
                      %249 = ascvec.and %248, %233, %mask : <ui16>, <ui16> -> <ui16>
                      %250 = ascvec.cmp_eq_scalar %249, %c-32768_i16, %mask : <ui16>, i16 -> !ascvec.mask
                      %251 = ascvec.xor %248, %233, %mask : <ui16>, <ui16> -> <ui16>
                      %252 = ascvec.xor %248, %237, %mask : <ui16>, <ui16> -> <ui16>
                      %253 = ascvec.select %252, %251, %250 : <ui16>
                      ascvec.storealign %arg59[%c256], %253, %mask : memref<768xui16, 1>, index, <ui16>
                      %254 = ascvec.add_scalar %230#3, %cst, %mask : <bf16>, bf16
                      %255 = ascvec.reinterpret %254 : <bf16> -> <ui16>
                      %256 = ascvec.and %255, %233, %mask : <ui16>, <ui16> -> <ui16>
                      %257 = ascvec.cmp_eq_scalar %256, %c-32768_i16, %mask : <ui16>, i16 -> !ascvec.mask
                      %258 = ascvec.xor %255, %233, %mask : <ui16>, <ui16> -> <ui16>
                      %259 = ascvec.xor %255, %237, %mask : <ui16>, <ui16> -> <ui16>
                      %260 = ascvec.select %259, %258, %257 : <ui16>
                      ascvec.storealign %arg59[%c384], %260, %mask : memref<768xui16, 1>, index, <ui16>
                      %261 = ascvec.add_scalar %230#4, %cst, %mask : <bf16>, bf16
                      %262 = ascvec.reinterpret %261 : <bf16> -> <ui16>
                      %263 = ascvec.and %262, %233, %mask : <ui16>, <ui16> -> <ui16>
                      %264 = ascvec.cmp_eq_scalar %263, %c-32768_i16, %mask : <ui16>, i16 -> !ascvec.mask
                      %265 = ascvec.xor %262, %233, %mask : <ui16>, <ui16> -> <ui16>
                      %266 = ascvec.xor %262, %237, %mask : <ui16>, <ui16> -> <ui16>
                      %267 = ascvec.select %266, %265, %264 : <ui16>
                      ascvec.storealign %arg59[%c512], %267, %mask : memref<768xui16, 1>, index, <ui16>
                      %268 = ascvec.add_scalar %230#5, %cst, %mask : <bf16>, bf16
                      %269 = ascvec.reinterpret %268 : <bf16> -> <ui16>
                      %270 = ascvec.and %269, %233, %mask : <ui16>, <ui16> -> <ui16>
                      %271 = ascvec.cmp_eq_scalar %270, %c-32768_i16, %mask : <ui16>, i16 -> !ascvec.mask
                      %272 = ascvec.xor %269, %233, %mask : <ui16>, <ui16> -> <ui16>
                      %273 = ascvec.xor %269, %237, %mask : <ui16>, <ui16> -> <ui16>
                      %274 = ascvec.select %273, %272, %271 : <ui16>
                      ascvec.storealign %arg59[%c640], %274, %mask : memref<768xui16, 1>, index, <ui16>
                    }
                    ascvec.mem_bar vst_vld
                  } {cannir.vf_mode = "simd"}
                  ascvec.release_buf v, %225, 0 {cannir.buffer.generated_sync}
                  ascvec.sync_intra_arrive v, %c0_i64 {cannir.sync.banks = 3 : i32}
                } else {
                  %221 = arith.select %156, %c906240_i64, %c918528_i64 : i64
                  %222 = arith.divui %221, %c8_i64 : i64
                  %223 = arith.addi %130, %222 : i64
                  %224 = ascscalar.bind_local_ptr %223 : memref<768xui16, 1>
                  %225 = arith.select %156, %c4_i64, %c5_i64 : i64
                  %226 = arith.cmpi slt, %165, %c512_i64 : i64
                  %227 = arith.extui %226 : i1 to i16
                  %228 = arith.minsi %177, %c128_i32 : i32
                  %229 = arith.maxsi %228, %c0_i32 : i32
                  %230 = arith.subi %177, %c128_i32 : i32
                  %231 = arith.minsi %230, %c128_i32 : i32
                  %232 = arith.maxsi %231, %c0_i32 : i32
                  %233 = arith.minsi %183, %c128_i32 : i32
                  %234 = arith.maxsi %233, %c0_i32 : i32
                  %235 = arith.subi %183, %c128_i32 : i32
                  %236 = arith.minsi %235, %c128_i32 : i32
                  %237 = arith.maxsi %236, %c0_i32 : i32
                  %238 = arith.minsi %189, %c128_i32 : i32
                  %239 = arith.maxsi %238, %c0_i32 : i32
                  %240 = arith.subi %189, %c128_i32 : i32
                  %241 = arith.minsi %240, %c128_i32 : i32
                  %242 = arith.maxsi %241, %c0_i32 : i32
                  ascvec.sync_intra_wait v, %c0_i64 {cannir.sync.banks = 3 : i32}
                  ascvec.get_buf v, %225, 0 {cannir.buffer.generated_sync}
                  ascvec.vf(%227, %133, %133, %170, %224, %134, %229, %232, %234, %237, %239, %242 : i16, memref<?xbf16, 1>, memref<?xbf16, 1>, i32, memref<768xui16, 1>, memref<?xui16, 1>, i32, i32, i32, i32, i32, i32) {
                  ^bb0(%arg56: i16, %arg57: memref<?xbf16, 1>, %arg58: memref<?xbf16, 1>, %arg59: i32, %arg60: memref<768xui16, 1>, %arg61: memref<?xui16, 1>, %arg62: i32, %arg63: i32, %arg64: i32, %arg65: i32, %arg66: i32, %arg67: i32):
                    %c59648 = arith.constant 59648 : index
                    %c59392 = arith.constant 59392 : index
                    %c640 = arith.constant 640 : index
                    %c512 = arith.constant 512 : index
                    %c59136 = arith.constant 59136 : index
                    %c58880 = arith.constant 58880 : index
                    %c384 = arith.constant 384 : index
                    %c256 = arith.constant 256 : index
                    %c58624 = arith.constant 58624 : index
                    %c128 = arith.constant 128 : index
                    %c55744 = arith.constant 55744 : index
                    %c55616 = arith.constant 55616 : index
                    %c58368 = arith.constant 58368 : index
                    %c-1_i16 = arith.constant -1 : i16
                    %c-32768_i16 = arith.constant -32768 : i16
                    %c1_i32_5 = arith.constant 1 : i32
                    %c0_i32_6 = arith.constant 0 : i32
                    %c64_i32 = arith.constant 64 : i32
                    %c32_i32_7 = arith.constant 32 : i32
                    %c24576_i32 = arith.constant 24576 : i32
                    %c256_i32_8 = arith.constant 256 : i32
                    %cst = arith.constant 0.000000e+00 : bf16
                    %c0_9 = arith.constant 0 : index
                    %c55488 = arith.constant 55488 : index
                    %c1_i16 = arith.constant 1 : i16
                    %c0_i16 = arith.constant 0 : i16
                    %c128_i32_10 = arith.constant 128 : i32
                    %mask, %remaining_out = ascvec.update_mask %c128_i32_10 elem_bits 16 : !ascvec.mask
                    %243 = ascvec.duplicate_scalar %c0_i16, %mask : i16 -> <ui16>
                    %244 = ascvec.duplicate_scalar %arg56, %mask : i16 -> <ui16>
                    %245 = ascvec.cmp_eq_scalar %244, %c1_i16, %mask : <ui16>, i16 -> !ascvec.mask
                    %246 = ascvec.loadalign %arg57[%c55488] : memref<?xbf16, 1>, index -> <bf16>
                    %247 = ascvec.duplicate_scalar %cst, %mask : bf16 -> <bf16>
                    %248 = ascvec.loadalign %arg57[%c55616] : memref<?xbf16, 1>, index -> <bf16>
                    %249 = ascvec.loadalign %arg57[%c55744] : memref<?xbf16, 1>, index -> <bf16>
                    %250:6 = scf.for %arg68 = %c0_i32_6 to %arg59 step %c1_i32_5 iter_args(%arg69 = %247, %arg70 = %247, %arg71 = %247, %arg72 = %247, %arg73 = %247, %arg74 = %247) -> (!ascvec.vreg<bf16>, !ascvec.vreg<bf16>, !ascvec.vreg<bf16>, !ascvec.vreg<bf16>, !ascvec.vreg<bf16>, !ascvec.vreg<bf16>)  : i32 {
                      %251 = arith.trunci %arg68 : i32 to i16
                      %252 = ascvec.duplicate_scalar %251, %mask : i16 -> <ui16>
                      %253 = ascvec.gather_reg %246, %252 : <bf16>, <ui16> -> <bf16>
                      %254 = ascvec.gather_reg %248, %252 : <bf16>, <ui16> -> <bf16>
                      %255 = ascvec.gather_reg %249, %252 : <bf16>, <ui16> -> <bf16>
                      %256 = arith.muli %arg68, %c256_i32_8 : i32
                      %257 = arith.index_cast %256 : i32 to index
                      %258 = ascvec.loadalign %arg58[%257] : memref<?xbf16, 1>, index -> <bf16>
                      %259 = ascvec.madd_two_address %258, %253, %arg69, %mask : <bf16>
                      %260 = arith.addi %256, %c24576_i32 : i32
                      %261 = arith.index_cast %260 : i32 to index
                      %262 = ascvec.loadalign %arg58[%261] : memref<?xbf16, 1>, index -> <bf16>
                      %263 = ascvec.madd_two_address %262, %253, %arg70, %mask : <bf16>
                      %264 = arith.addi %arg68, %c32_i32_7 : i32
                      %265 = arith.muli %264, %c256_i32_8 : i32
                      %266 = arith.index_cast %265 : i32 to index
                      %267 = ascvec.loadalign %arg58[%266] : memref<?xbf16, 1>, index -> <bf16>
                      %268 = ascvec.madd_two_address %267, %254, %arg71, %mask : <bf16>
                      %269 = arith.addi %265, %c24576_i32 : i32
                      %270 = arith.index_cast %269 : i32 to index
                      %271 = ascvec.loadalign %arg58[%270] : memref<?xbf16, 1>, index -> <bf16>
                      %272 = ascvec.madd_two_address %271, %254, %arg72, %mask : <bf16>
                      %273 = arith.addi %arg68, %c64_i32 : i32
                      %274 = arith.muli %273, %c256_i32_8 : i32
                      %275 = arith.index_cast %274 : i32 to index
                      %276 = ascvec.loadalign %arg58[%275] : memref<?xbf16, 1>, index -> <bf16>
                      %277 = ascvec.madd_two_address %276, %255, %arg73, %mask : <bf16>
                      %278 = arith.addi %274, %c24576_i32 : i32
                      %279 = arith.index_cast %278 : i32 to index
                      %280 = ascvec.loadalign %arg58[%279] : memref<?xbf16, 1>, index -> <bf16>
                      %281 = ascvec.madd_two_address %280, %255, %arg74, %mask : <bf16>
                      scf.yield %259, %263, %268, %272, %277, %281 : !ascvec.vreg<bf16>, !ascvec.vreg<bf16>, !ascvec.vreg<bf16>, !ascvec.vreg<bf16>, !ascvec.vreg<bf16>, !ascvec.vreg<bf16>
                    } {cannir.loop_unroll = 4 : i64}
                    %c0_11 = arith.constant 0 : index
                    %c1_12 = arith.constant 1 : index
                    scf.for %arg68 = %c0_11 to %c1_12 step %c1_12 {
                      %251 = ascvec.add_scalar %250#0, %cst, %mask : <bf16>, bf16
                      %252 = ascvec.reinterpret %251 : <bf16> -> <ui16>
                      %253 = ascvec.duplicate_scalar %c-32768_i16, %mask : i16 -> <ui16>
                      %254 = ascvec.and %252, %253, %mask : <ui16>, <ui16> -> <ui16>
                      %255 = ascvec.cmp_eq_scalar %254, %c-32768_i16, %mask : <ui16>, i16 -> !ascvec.mask
                      %256 = ascvec.xor %252, %253, %mask : <ui16>, <ui16> -> <ui16>
                      %257 = ascvec.duplicate_scalar %c-1_i16, %mask : i16 -> <ui16>
                      %258 = ascvec.xor %252, %257, %mask : <ui16>, <ui16> -> <ui16>
                      %259 = ascvec.select %258, %256, %255 : <ui16>
                      ascvec.storealign %arg60[%c0_9], %259, %mask : memref<768xui16, 1>, index, <ui16>
                      %260 = ascvec.add_scalar %250#1, %cst, %mask : <bf16>, bf16
                      %261 = ascvec.reinterpret %260 : <bf16> -> <ui16>
                      %262 = ascvec.and %261, %253, %mask : <ui16>, <ui16> -> <ui16>
                      %263 = ascvec.cmp_eq_scalar %262, %c-32768_i16, %mask : <ui16>, i16 -> !ascvec.mask
                      %264 = ascvec.xor %261, %253, %mask : <ui16>, <ui16> -> <ui16>
                      %265 = ascvec.xor %261, %257, %mask : <ui16>, <ui16> -> <ui16>
                      %266 = ascvec.select %265, %264, %263 : <ui16>
                      ascvec.storealign %arg60[%c128], %266, %mask : memref<768xui16, 1>, index, <ui16>
                      %267 = ascvec.loadalign %arg61[%c58368] : memref<?xui16, 1>, index -> <ui16>
                      %268 = ascvec.select %243, %267, %245 : <ui16>
                      %269 = ascvec.loadalign %arg61[%c58624] : memref<?xui16, 1>, index -> <ui16>
                      %270 = ascvec.select %243, %269, %245 : <ui16>
                      %271 = ascvec.shr %259, 8, %mask : <ui16> -> <ui16>
                      %272 = ascvec.pack %271 src_elem_bits 16 dst_elem_bits 8 {part = "low"} : <ui16> -> <ui8>
                      %mask_13, %remaining_out_14 = ascvec.update_mask %arg62 elem_bits 8 : !ascvec.mask
                      %273 = ascvec.cumulative_histogram %268, %272, %mask_13 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                      %274 = ascvec.cumulative_histogram %270, %272, %mask_13 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                      %275 = ascvec.shr %266, 8, %mask : <ui16> -> <ui16>
                      %276 = ascvec.pack %275 src_elem_bits 16 dst_elem_bits 8 {part = "low"} : <ui16> -> <ui8>
                      %mask_15, %remaining_out_16 = ascvec.update_mask %arg63 elem_bits 8 : !ascvec.mask
                      %277 = ascvec.cumulative_histogram %273, %276, %mask_15 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                      %278 = ascvec.cumulative_histogram %274, %276, %mask_15 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                      ascvec.storealign %arg61[%c58368], %277, %mask : memref<?xui16, 1>, index, <ui16>
                      ascvec.storealign %arg61[%c58624], %278, %mask : memref<?xui16, 1>, index, <ui16>
                      %279 = ascvec.add_scalar %250#2, %cst, %mask : <bf16>, bf16
                      %280 = ascvec.reinterpret %279 : <bf16> -> <ui16>
                      %281 = ascvec.and %280, %253, %mask : <ui16>, <ui16> -> <ui16>
                      %282 = ascvec.cmp_eq_scalar %281, %c-32768_i16, %mask : <ui16>, i16 -> !ascvec.mask
                      %283 = ascvec.xor %280, %253, %mask : <ui16>, <ui16> -> <ui16>
                      %284 = ascvec.xor %280, %257, %mask : <ui16>, <ui16> -> <ui16>
                      %285 = ascvec.select %284, %283, %282 : <ui16>
                      ascvec.storealign %arg60[%c256], %285, %mask : memref<768xui16, 1>, index, <ui16>
                      %286 = ascvec.add_scalar %250#3, %cst, %mask : <bf16>, bf16
                      %287 = ascvec.reinterpret %286 : <bf16> -> <ui16>
                      %288 = ascvec.and %287, %253, %mask : <ui16>, <ui16> -> <ui16>
                      %289 = ascvec.cmp_eq_scalar %288, %c-32768_i16, %mask : <ui16>, i16 -> !ascvec.mask
                      %290 = ascvec.xor %287, %253, %mask : <ui16>, <ui16> -> <ui16>
                      %291 = ascvec.xor %287, %257, %mask : <ui16>, <ui16> -> <ui16>
                      %292 = ascvec.select %291, %290, %289 : <ui16>
                      ascvec.storealign %arg60[%c384], %292, %mask : memref<768xui16, 1>, index, <ui16>
                      %293 = ascvec.loadalign %arg61[%c58880] : memref<?xui16, 1>, index -> <ui16>
                      %294 = ascvec.select %243, %293, %245 : <ui16>
                      %295 = ascvec.loadalign %arg61[%c59136] : memref<?xui16, 1>, index -> <ui16>
                      %296 = ascvec.select %243, %295, %245 : <ui16>
                      %297 = ascvec.shr %285, 8, %mask : <ui16> -> <ui16>
                      %298 = ascvec.pack %297 src_elem_bits 16 dst_elem_bits 8 {part = "low"} : <ui16> -> <ui8>
                      %mask_17, %remaining_out_18 = ascvec.update_mask %arg64 elem_bits 8 : !ascvec.mask
                      %299 = ascvec.cumulative_histogram %294, %298, %mask_17 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                      %300 = ascvec.cumulative_histogram %296, %298, %mask_17 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                      %301 = ascvec.shr %292, 8, %mask : <ui16> -> <ui16>
                      %302 = ascvec.pack %301 src_elem_bits 16 dst_elem_bits 8 {part = "low"} : <ui16> -> <ui8>
                      %mask_19, %remaining_out_20 = ascvec.update_mask %arg65 elem_bits 8 : !ascvec.mask
                      %303 = ascvec.cumulative_histogram %299, %302, %mask_19 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                      %304 = ascvec.cumulative_histogram %300, %302, %mask_19 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                      ascvec.storealign %arg61[%c58880], %303, %mask : memref<?xui16, 1>, index, <ui16>
                      ascvec.storealign %arg61[%c59136], %304, %mask : memref<?xui16, 1>, index, <ui16>
                      %305 = ascvec.add_scalar %250#4, %cst, %mask : <bf16>, bf16
                      %306 = ascvec.reinterpret %305 : <bf16> -> <ui16>
                      %307 = ascvec.and %306, %253, %mask : <ui16>, <ui16> -> <ui16>
                      %308 = ascvec.cmp_eq_scalar %307, %c-32768_i16, %mask : <ui16>, i16 -> !ascvec.mask
                      %309 = ascvec.xor %306, %253, %mask : <ui16>, <ui16> -> <ui16>
                      %310 = ascvec.xor %306, %257, %mask : <ui16>, <ui16> -> <ui16>
                      %311 = ascvec.select %310, %309, %308 : <ui16>
                      ascvec.storealign %arg60[%c512], %311, %mask : memref<768xui16, 1>, index, <ui16>
                      %312 = ascvec.add_scalar %250#5, %cst, %mask : <bf16>, bf16
                      %313 = ascvec.reinterpret %312 : <bf16> -> <ui16>
                      %314 = ascvec.and %313, %253, %mask : <ui16>, <ui16> -> <ui16>
                      %315 = ascvec.cmp_eq_scalar %314, %c-32768_i16, %mask : <ui16>, i16 -> !ascvec.mask
                      %316 = ascvec.xor %313, %253, %mask : <ui16>, <ui16> -> <ui16>
                      %317 = ascvec.xor %313, %257, %mask : <ui16>, <ui16> -> <ui16>
                      %318 = ascvec.select %317, %316, %315 : <ui16>
                      ascvec.storealign %arg60[%c640], %318, %mask : memref<768xui16, 1>, index, <ui16>
                      %319 = ascvec.loadalign %arg61[%c59392] : memref<?xui16, 1>, index -> <ui16>
                      %320 = ascvec.select %243, %319, %245 : <ui16>
                      %321 = ascvec.loadalign %arg61[%c59648] : memref<?xui16, 1>, index -> <ui16>
                      %322 = ascvec.select %243, %321, %245 : <ui16>
                      %323 = ascvec.shr %311, 8, %mask : <ui16> -> <ui16>
                      %324 = ascvec.pack %323 src_elem_bits 16 dst_elem_bits 8 {part = "low"} : <ui16> -> <ui8>
                      %mask_21, %remaining_out_22 = ascvec.update_mask %arg66 elem_bits 8 : !ascvec.mask
                      %325 = ascvec.cumulative_histogram %320, %324, %mask_21 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                      %326 = ascvec.cumulative_histogram %322, %324, %mask_21 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                      %327 = ascvec.shr %318, 8, %mask : <ui16> -> <ui16>
                      %328 = ascvec.pack %327 src_elem_bits 16 dst_elem_bits 8 {part = "low"} : <ui16> -> <ui8>
                      %mask_23, %remaining_out_24 = ascvec.update_mask %arg67 elem_bits 8 : !ascvec.mask
                      %329 = ascvec.cumulative_histogram %325, %328, %mask_23 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                      %330 = ascvec.cumulative_histogram %326, %328, %mask_23 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                      ascvec.storealign %arg61[%c59392], %329, %mask : memref<?xui16, 1>, index, <ui16>
                      ascvec.storealign %arg61[%c59648], %330, %mask : memref<?xui16, 1>, index, <ui16>
                    }
                    ascvec.mem_bar vst_vld
                  } {cannir.vf_mode = "simd"}
                  ascvec.release_buf v, %225, 0 {cannir.buffer.generated_sync}
                  ascvec.sync_intra_arrive v, %c0_i64 {cannir.sync.banks = 3 : i32}
                }
                %191 = arith.subi %53, %109 : i32
                %192 = arith.cmpi slt, %191, %c0_i32 : i32
                %193 = arith.select %192, %c0_i32, %191 : i32
                %194 = arith.cmpi sgt, %193, %c3_i32 : i32
                %195 = arith.select %194, %c3_i32, %193 : i32
                %196 = arith.muli %5, %c6_i64 : i64
                %197 = arith.addi %196, %106 : i64
                %198 = arith.select %156, %c906240_i64, %c918528_i64 : i64
                %199 = arith.select %156, %c4_i64, %c5_i64 : i64
                %200 = cannir.extract_physical_layout %arg12 : !cannir.tensor<<gm, ui16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
                %201 = cannir.extract_stride %200 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
                %202 = cannir.extract_dim %201[0] : !cannir.stride<(-1, 1)>
                %203 = arith.muli %197, %202 : i64
                %204 = arith.addi %203, %165 : i64
                %205 = arith.muli %204, %c16_i64 : i64
                %206 = ascscalar.memref_address %alloc_2 : memref<1xui16>
                %207 = arith.divui %205, %c8_i64 : i64
                %208 = arith.addi %206, %207 : i64
                %209 = ascscalar.bind_gm_ptr %208 : memref<1xui16>
                %210 = arith.divui %198, %c8_i64 : i64
                %211 = arith.addi %130, %210 : i64
                %212 = ascscalar.bind_local_ptr %211 : memref<1xui16, 1>
                %213 = arith.index_cast %195 : i32 to index
                %214 = cannir.extract_layout %arg12 : !cannir.tensor<<gm, ui16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
                %215 = cannir.extract_stride %214 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
                %216 = cannir.extract_dim %215[0] : !cannir.stride<(-1, 1)>
                %217 = arith.index_cast %216 : i64 to index
                %218 = arith.muli %217, %c2 : index
                %219 = arith.index_cast %213 : index to i64
                %220 = arith.cmpi ugt, %219, %c0_i64 : i64
                scf.if %220 {
                  %221 = arith.index_cast %218 : index to i64
                  ascvec.get_buf mte3, %199, 0 {cannir.buffer.generated_sync}
                  %222 = arith.trunci %219 : i64 to i32
                  ascvec.copy_ub2gm %209[%c0], %212[%c0], %222, %c512_i32, %221, %c512_i32 l2_cache_ctl %c0_i32 : memref<1xui16>, memref<1xui16, 1>
                  ascvec.release_buf mte3, %199, 0 {cannir.buffer.generated_sync}
                }
                ascvec.sync_intra_arrive v, %c8_i64
                scf.yield %160, %158, %164 : i64, i64, i64
              } else {
                %155 = arith.muli %arg51, %c256_i64 : i64
                %156 = arith.cmpi eq, %arg54, %c0_i64 : i64
                %157 = arith.addi %arg54, %c1_i64 : i64
                %158 = arith.remui %157, %c2_i64 : i64
                %159 = arith.addi %arg53, %c1_i64 : i64
                %160 = arith.remui %159, %c4_i64 : i64
                %161 = arith.addi %arg55, %c1_i64 : i64
                %162 = arith.remui %161, %c2_i64 : i64
                %163 = arith.addi %162, %c1_i64 overflow<nsw> : i64
                %164 = arith.remui %163, %c2_i64 : i64
                ascvec.sync_intra_wait v, %c9_i64
                %165 = arith.subi %155, %84 : i64
                %166 = arith.subi %72, %arg51 : i64
                %167 = arith.muli %166, %c256_i64 : i64
                %168 = arith.minsi %167, %c256_i64 : i64
                %169 = arith.cmpi slt, %109, %56 : i32
                %170 = arith.select %169, %c32_i32, %c0_i32 : i32
                %171 = arith.cmpi sge, %109, %56 : i32
                %172 = arith.select %171, %c0_i32, %31 : i32
                %173 = arith.trunci %168 : i64 to i32
                %174 = arith.trunci %155 : i64 to i32
                %175 = arith.subi %172, %174 : i32
                %176 = arith.minsi %173, %175 : i32
                %177 = arith.maxsi %176, %c0_i32 : i32
                %178 = arith.addi %109, %c1_i32 : i32
                %179 = arith.cmpi sge, %178, %56 : i32
                %180 = arith.select %179, %c0_i32, %31 : i32
                %181 = arith.subi %180, %174 : i32
                %182 = arith.minsi %173, %181 : i32
                %183 = arith.maxsi %182, %c0_i32 : i32
                %184 = arith.addi %109, %c2_i32 : i32
                %185 = arith.cmpi sge, %184, %56 : i32
                %186 = arith.select %185, %c0_i32, %31 : i32
                %187 = arith.subi %186, %174 : i32
                %188 = arith.minsi %173, %187 : i32
                %189 = arith.maxsi %188, %c0_i32 : i32
                %190 = arith.cmpi eq, %138, %c0_i32 : i32
                scf.if %190 {
                  %221 = arith.select %156, %c906240_i64, %c918528_i64 : i64
                  %222 = arith.divui %221, %c8_i64 : i64
                  %223 = arith.addi %130, %222 : i64
                  %224 = ascscalar.bind_local_ptr %223 : memref<768xui16, 1>
                  %225 = arith.select %156, %c4_i64, %c5_i64 : i64
                  ascvec.sync_intra_wait v, %c1_i64 {cannir.sync.banks = 3 : i32}
                  ascvec.get_buf v, %225, 0 {cannir.buffer.generated_sync}
                  ascvec.vf(%133, %133, %170, %224 : memref<?xbf16, 1>, memref<?xbf16, 1>, i32, memref<768xui16, 1>) {
                  ^bb0(%arg56: memref<?xbf16, 1>, %arg57: memref<?xbf16, 1>, %arg58: i32, %arg59: memref<768xui16, 1>):
                    %c640 = arith.constant 640 : index
                    %c512 = arith.constant 512 : index
                    %c384 = arith.constant 384 : index
                    %c256 = arith.constant 256 : index
                    %c55744 = arith.constant 55744 : index
                    %c55616 = arith.constant 55616 : index
                    %c-1_i16 = arith.constant -1 : i16
                    %c-32768_i16 = arith.constant -32768 : i16
                    %c1_i32_5 = arith.constant 1 : i32
                    %c0_i32_6 = arith.constant 0 : i32
                    %c64_i32 = arith.constant 64 : i32
                    %c32_i32_7 = arith.constant 32 : i32
                    %c24576_i32 = arith.constant 24576 : i32
                    %c256_i32_8 = arith.constant 256 : i32
                    %c128 = arith.constant 128 : index
                    %cst = arith.constant 0.000000e+00 : bf16
                    %c0_9 = arith.constant 0 : index
                    %c55488 = arith.constant 55488 : index
                    %c128_i32_10 = arith.constant 128 : i32
                    %mask, %remaining_out = ascvec.update_mask %c128_i32_10 elem_bits 16 : !ascvec.mask
                    %226 = ascvec.loadalign %arg56[%c55488] : memref<?xbf16, 1>, index -> <bf16>
                    %227 = ascvec.duplicate_scalar %cst, %mask : bf16 -> <bf16>
                    %228 = ascvec.loadalign %arg56[%c55616] : memref<?xbf16, 1>, index -> <bf16>
                    %229 = ascvec.loadalign %arg56[%c55744] : memref<?xbf16, 1>, index -> <bf16>
                    %230:6 = scf.for %arg60 = %c0_i32_6 to %arg58 step %c1_i32_5 iter_args(%arg61 = %227, %arg62 = %227, %arg63 = %227, %arg64 = %227, %arg65 = %227, %arg66 = %227) -> (!ascvec.vreg<bf16>, !ascvec.vreg<bf16>, !ascvec.vreg<bf16>, !ascvec.vreg<bf16>, !ascvec.vreg<bf16>, !ascvec.vreg<bf16>)  : i32 {
                      %231 = arith.trunci %arg60 : i32 to i16
                      %232 = ascvec.duplicate_scalar %231, %mask : i16 -> <ui16>
                      %233 = ascvec.gather_reg %226, %232 : <bf16>, <ui16> -> <bf16>
                      %234 = ascvec.gather_reg %228, %232 : <bf16>, <ui16> -> <bf16>
                      %235 = ascvec.gather_reg %229, %232 : <bf16>, <ui16> -> <bf16>
                      %236 = arith.muli %arg60, %c256_i32_8 : i32
                      %237 = arith.index_cast %236 : i32 to index
                      %238 = arith.addi %237, %c128 : index
                      %239 = ascvec.loadalign %arg57[%238] : memref<?xbf16, 1>, index -> <bf16>
                      %240 = ascvec.madd_two_address %239, %233, %arg61, %mask : <bf16>
                      %241 = arith.addi %236, %c24576_i32 : i32
                      %242 = arith.index_cast %241 : i32 to index
                      %243 = arith.addi %242, %c128 : index
                      %244 = ascvec.loadalign %arg57[%243] : memref<?xbf16, 1>, index -> <bf16>
                      %245 = ascvec.madd_two_address %244, %233, %arg62, %mask : <bf16>
                      %246 = arith.addi %arg60, %c32_i32_7 : i32
                      %247 = arith.muli %246, %c256_i32_8 : i32
                      %248 = arith.index_cast %247 : i32 to index
                      %249 = arith.addi %248, %c128 : index
                      %250 = ascvec.loadalign %arg57[%249] : memref<?xbf16, 1>, index -> <bf16>
                      %251 = ascvec.madd_two_address %250, %234, %arg63, %mask : <bf16>
                      %252 = arith.addi %247, %c24576_i32 : i32
                      %253 = arith.index_cast %252 : i32 to index
                      %254 = arith.addi %253, %c128 : index
                      %255 = ascvec.loadalign %arg57[%254] : memref<?xbf16, 1>, index -> <bf16>
                      %256 = ascvec.madd_two_address %255, %234, %arg64, %mask : <bf16>
                      %257 = arith.addi %arg60, %c64_i32 : i32
                      %258 = arith.muli %257, %c256_i32_8 : i32
                      %259 = arith.index_cast %258 : i32 to index
                      %260 = arith.addi %259, %c128 : index
                      %261 = ascvec.loadalign %arg57[%260] : memref<?xbf16, 1>, index -> <bf16>
                      %262 = ascvec.madd_two_address %261, %235, %arg65, %mask : <bf16>
                      %263 = arith.addi %258, %c24576_i32 : i32
                      %264 = arith.index_cast %263 : i32 to index
                      %265 = arith.addi %264, %c128 : index
                      %266 = ascvec.loadalign %arg57[%265] : memref<?xbf16, 1>, index -> <bf16>
                      %267 = ascvec.madd_two_address %266, %235, %arg66, %mask : <bf16>
                      scf.yield %240, %245, %251, %256, %262, %267 : !ascvec.vreg<bf16>, !ascvec.vreg<bf16>, !ascvec.vreg<bf16>, !ascvec.vreg<bf16>, !ascvec.vreg<bf16>, !ascvec.vreg<bf16>
                    } {cannir.loop_unroll = 4 : i64}
                    %c0_11 = arith.constant 0 : index
                    %c1_12 = arith.constant 1 : index
                    scf.for %arg60 = %c0_11 to %c1_12 step %c1_12 {
                      %231 = ascvec.add_scalar %230#0, %cst, %mask : <bf16>, bf16
                      %232 = ascvec.reinterpret %231 : <bf16> -> <ui16>
                      %233 = ascvec.duplicate_scalar %c-32768_i16, %mask : i16 -> <ui16>
                      %234 = ascvec.and %232, %233, %mask : <ui16>, <ui16> -> <ui16>
                      %235 = ascvec.cmp_eq_scalar %234, %c-32768_i16, %mask : <ui16>, i16 -> !ascvec.mask
                      %236 = ascvec.xor %232, %233, %mask : <ui16>, <ui16> -> <ui16>
                      %237 = ascvec.duplicate_scalar %c-1_i16, %mask : i16 -> <ui16>
                      %238 = ascvec.xor %232, %237, %mask : <ui16>, <ui16> -> <ui16>
                      %239 = ascvec.select %238, %236, %235 : <ui16>
                      ascvec.storealign %arg59[%c0_9], %239, %mask : memref<768xui16, 1>, index, <ui16>
                      %240 = ascvec.add_scalar %230#1, %cst, %mask : <bf16>, bf16
                      %241 = ascvec.reinterpret %240 : <bf16> -> <ui16>
                      %242 = ascvec.and %241, %233, %mask : <ui16>, <ui16> -> <ui16>
                      %243 = ascvec.cmp_eq_scalar %242, %c-32768_i16, %mask : <ui16>, i16 -> !ascvec.mask
                      %244 = ascvec.xor %241, %233, %mask : <ui16>, <ui16> -> <ui16>
                      %245 = ascvec.xor %241, %237, %mask : <ui16>, <ui16> -> <ui16>
                      %246 = ascvec.select %245, %244, %243 : <ui16>
                      ascvec.storealign %arg59[%c128], %246, %mask : memref<768xui16, 1>, index, <ui16>
                      %247 = ascvec.add_scalar %230#2, %cst, %mask : <bf16>, bf16
                      %248 = ascvec.reinterpret %247 : <bf16> -> <ui16>
                      %249 = ascvec.and %248, %233, %mask : <ui16>, <ui16> -> <ui16>
                      %250 = ascvec.cmp_eq_scalar %249, %c-32768_i16, %mask : <ui16>, i16 -> !ascvec.mask
                      %251 = ascvec.xor %248, %233, %mask : <ui16>, <ui16> -> <ui16>
                      %252 = ascvec.xor %248, %237, %mask : <ui16>, <ui16> -> <ui16>
                      %253 = ascvec.select %252, %251, %250 : <ui16>
                      ascvec.storealign %arg59[%c256], %253, %mask : memref<768xui16, 1>, index, <ui16>
                      %254 = ascvec.add_scalar %230#3, %cst, %mask : <bf16>, bf16
                      %255 = ascvec.reinterpret %254 : <bf16> -> <ui16>
                      %256 = ascvec.and %255, %233, %mask : <ui16>, <ui16> -> <ui16>
                      %257 = ascvec.cmp_eq_scalar %256, %c-32768_i16, %mask : <ui16>, i16 -> !ascvec.mask
                      %258 = ascvec.xor %255, %233, %mask : <ui16>, <ui16> -> <ui16>
                      %259 = ascvec.xor %255, %237, %mask : <ui16>, <ui16> -> <ui16>
                      %260 = ascvec.select %259, %258, %257 : <ui16>
                      ascvec.storealign %arg59[%c384], %260, %mask : memref<768xui16, 1>, index, <ui16>
                      %261 = ascvec.add_scalar %230#4, %cst, %mask : <bf16>, bf16
                      %262 = ascvec.reinterpret %261 : <bf16> -> <ui16>
                      %263 = ascvec.and %262, %233, %mask : <ui16>, <ui16> -> <ui16>
                      %264 = ascvec.cmp_eq_scalar %263, %c-32768_i16, %mask : <ui16>, i16 -> !ascvec.mask
                      %265 = ascvec.xor %262, %233, %mask : <ui16>, <ui16> -> <ui16>
                      %266 = ascvec.xor %262, %237, %mask : <ui16>, <ui16> -> <ui16>
                      %267 = ascvec.select %266, %265, %264 : <ui16>
                      ascvec.storealign %arg59[%c512], %267, %mask : memref<768xui16, 1>, index, <ui16>
                      %268 = ascvec.add_scalar %230#5, %cst, %mask : <bf16>, bf16
                      %269 = ascvec.reinterpret %268 : <bf16> -> <ui16>
                      %270 = ascvec.and %269, %233, %mask : <ui16>, <ui16> -> <ui16>
                      %271 = ascvec.cmp_eq_scalar %270, %c-32768_i16, %mask : <ui16>, i16 -> !ascvec.mask
                      %272 = ascvec.xor %269, %233, %mask : <ui16>, <ui16> -> <ui16>
                      %273 = ascvec.xor %269, %237, %mask : <ui16>, <ui16> -> <ui16>
                      %274 = ascvec.select %273, %272, %271 : <ui16>
                      ascvec.storealign %arg59[%c640], %274, %mask : memref<768xui16, 1>, index, <ui16>
                    }
                    ascvec.mem_bar vst_vld
                  } {cannir.vf_mode = "simd"}
                  ascvec.release_buf v, %225, 0 {cannir.buffer.generated_sync}
                  ascvec.sync_intra_arrive v, %c1_i64 {cannir.sync.banks = 3 : i32}
                } else {
                  %221 = arith.select %156, %c906240_i64, %c918528_i64 : i64
                  %222 = arith.divui %221, %c8_i64 : i64
                  %223 = arith.addi %130, %222 : i64
                  %224 = ascscalar.bind_local_ptr %223 : memref<768xui16, 1>
                  %225 = arith.select %156, %c4_i64, %c5_i64 : i64
                  %226 = arith.cmpi slt, %165, %c512_i64 : i64
                  %227 = arith.extui %226 : i1 to i16
                  %228 = arith.minsi %177, %c128_i32 : i32
                  %229 = arith.maxsi %228, %c0_i32 : i32
                  %230 = arith.subi %177, %c128_i32 : i32
                  %231 = arith.minsi %230, %c128_i32 : i32
                  %232 = arith.maxsi %231, %c0_i32 : i32
                  %233 = arith.minsi %183, %c128_i32 : i32
                  %234 = arith.maxsi %233, %c0_i32 : i32
                  %235 = arith.subi %183, %c128_i32 : i32
                  %236 = arith.minsi %235, %c128_i32 : i32
                  %237 = arith.maxsi %236, %c0_i32 : i32
                  %238 = arith.minsi %189, %c128_i32 : i32
                  %239 = arith.maxsi %238, %c0_i32 : i32
                  %240 = arith.subi %189, %c128_i32 : i32
                  %241 = arith.minsi %240, %c128_i32 : i32
                  %242 = arith.maxsi %241, %c0_i32 : i32
                  ascvec.sync_intra_wait v, %c1_i64 {cannir.sync.banks = 3 : i32}
                  ascvec.get_buf v, %225, 0 {cannir.buffer.generated_sync}
                  ascvec.vf(%227, %133, %133, %170, %224, %134, %229, %232, %234, %237, %239, %242 : i16, memref<?xbf16, 1>, memref<?xbf16, 1>, i32, memref<768xui16, 1>, memref<?xui16, 1>, i32, i32, i32, i32, i32, i32) {
                  ^bb0(%arg56: i16, %arg57: memref<?xbf16, 1>, %arg58: memref<?xbf16, 1>, %arg59: i32, %arg60: memref<768xui16, 1>, %arg61: memref<?xui16, 1>, %arg62: i32, %arg63: i32, %arg64: i32, %arg65: i32, %arg66: i32, %arg67: i32):
                    %c59776 = arith.constant 59776 : index
                    %c59520 = arith.constant 59520 : index
                    %c640 = arith.constant 640 : index
                    %c512 = arith.constant 512 : index
                    %c59264 = arith.constant 59264 : index
                    %c59008 = arith.constant 59008 : index
                    %c384 = arith.constant 384 : index
                    %c256 = arith.constant 256 : index
                    %c58752 = arith.constant 58752 : index
                    %c55744 = arith.constant 55744 : index
                    %c55616 = arith.constant 55616 : index
                    %c58496 = arith.constant 58496 : index
                    %c-1_i16 = arith.constant -1 : i16
                    %c-32768_i16 = arith.constant -32768 : i16
                    %c1_i32_5 = arith.constant 1 : i32
                    %c0_i32_6 = arith.constant 0 : i32
                    %c64_i32 = arith.constant 64 : i32
                    %c32_i32_7 = arith.constant 32 : i32
                    %c24576_i32 = arith.constant 24576 : i32
                    %c256_i32_8 = arith.constant 256 : i32
                    %c128 = arith.constant 128 : index
                    %cst = arith.constant 0.000000e+00 : bf16
                    %c0_9 = arith.constant 0 : index
                    %c55488 = arith.constant 55488 : index
                    %c1_i16 = arith.constant 1 : i16
                    %c0_i16 = arith.constant 0 : i16
                    %c128_i32_10 = arith.constant 128 : i32
                    %mask, %remaining_out = ascvec.update_mask %c128_i32_10 elem_bits 16 : !ascvec.mask
                    %243 = ascvec.duplicate_scalar %c0_i16, %mask : i16 -> <ui16>
                    %244 = ascvec.duplicate_scalar %arg56, %mask : i16 -> <ui16>
                    %245 = ascvec.cmp_eq_scalar %244, %c1_i16, %mask : <ui16>, i16 -> !ascvec.mask
                    %246 = ascvec.loadalign %arg57[%c55488] : memref<?xbf16, 1>, index -> <bf16>
                    %247 = ascvec.duplicate_scalar %cst, %mask : bf16 -> <bf16>
                    %248 = ascvec.loadalign %arg57[%c55616] : memref<?xbf16, 1>, index -> <bf16>
                    %249 = ascvec.loadalign %arg57[%c55744] : memref<?xbf16, 1>, index -> <bf16>
                    %250:6 = scf.for %arg68 = %c0_i32_6 to %arg59 step %c1_i32_5 iter_args(%arg69 = %247, %arg70 = %247, %arg71 = %247, %arg72 = %247, %arg73 = %247, %arg74 = %247) -> (!ascvec.vreg<bf16>, !ascvec.vreg<bf16>, !ascvec.vreg<bf16>, !ascvec.vreg<bf16>, !ascvec.vreg<bf16>, !ascvec.vreg<bf16>)  : i32 {
                      %251 = arith.trunci %arg68 : i32 to i16
                      %252 = ascvec.duplicate_scalar %251, %mask : i16 -> <ui16>
                      %253 = ascvec.gather_reg %246, %252 : <bf16>, <ui16> -> <bf16>
                      %254 = ascvec.gather_reg %248, %252 : <bf16>, <ui16> -> <bf16>
                      %255 = ascvec.gather_reg %249, %252 : <bf16>, <ui16> -> <bf16>
                      %256 = arith.muli %arg68, %c256_i32_8 : i32
                      %257 = arith.index_cast %256 : i32 to index
                      %258 = arith.addi %257, %c128 : index
                      %259 = ascvec.loadalign %arg58[%258] : memref<?xbf16, 1>, index -> <bf16>
                      %260 = ascvec.madd_two_address %259, %253, %arg69, %mask : <bf16>
                      %261 = arith.addi %256, %c24576_i32 : i32
                      %262 = arith.index_cast %261 : i32 to index
                      %263 = arith.addi %262, %c128 : index
                      %264 = ascvec.loadalign %arg58[%263] : memref<?xbf16, 1>, index -> <bf16>
                      %265 = ascvec.madd_two_address %264, %253, %arg70, %mask : <bf16>
                      %266 = arith.addi %arg68, %c32_i32_7 : i32
                      %267 = arith.muli %266, %c256_i32_8 : i32
                      %268 = arith.index_cast %267 : i32 to index
                      %269 = arith.addi %268, %c128 : index
                      %270 = ascvec.loadalign %arg58[%269] : memref<?xbf16, 1>, index -> <bf16>
                      %271 = ascvec.madd_two_address %270, %254, %arg71, %mask : <bf16>
                      %272 = arith.addi %267, %c24576_i32 : i32
                      %273 = arith.index_cast %272 : i32 to index
                      %274 = arith.addi %273, %c128 : index
                      %275 = ascvec.loadalign %arg58[%274] : memref<?xbf16, 1>, index -> <bf16>
                      %276 = ascvec.madd_two_address %275, %254, %arg72, %mask : <bf16>
                      %277 = arith.addi %arg68, %c64_i32 : i32
                      %278 = arith.muli %277, %c256_i32_8 : i32
                      %279 = arith.index_cast %278 : i32 to index
                      %280 = arith.addi %279, %c128 : index
                      %281 = ascvec.loadalign %arg58[%280] : memref<?xbf16, 1>, index -> <bf16>
                      %282 = ascvec.madd_two_address %281, %255, %arg73, %mask : <bf16>
                      %283 = arith.addi %278, %c24576_i32 : i32
                      %284 = arith.index_cast %283 : i32 to index
                      %285 = arith.addi %284, %c128 : index
                      %286 = ascvec.loadalign %arg58[%285] : memref<?xbf16, 1>, index -> <bf16>
                      %287 = ascvec.madd_two_address %286, %255, %arg74, %mask : <bf16>
                      scf.yield %260, %265, %271, %276, %282, %287 : !ascvec.vreg<bf16>, !ascvec.vreg<bf16>, !ascvec.vreg<bf16>, !ascvec.vreg<bf16>, !ascvec.vreg<bf16>, !ascvec.vreg<bf16>
                    } {cannir.loop_unroll = 4 : i64}
                    %c0_11 = arith.constant 0 : index
                    %c1_12 = arith.constant 1 : index
                    scf.for %arg68 = %c0_11 to %c1_12 step %c1_12 {
                      %251 = ascvec.add_scalar %250#0, %cst, %mask : <bf16>, bf16
                      %252 = ascvec.reinterpret %251 : <bf16> -> <ui16>
                      %253 = ascvec.duplicate_scalar %c-32768_i16, %mask : i16 -> <ui16>
                      %254 = ascvec.and %252, %253, %mask : <ui16>, <ui16> -> <ui16>
                      %255 = ascvec.cmp_eq_scalar %254, %c-32768_i16, %mask : <ui16>, i16 -> !ascvec.mask
                      %256 = ascvec.xor %252, %253, %mask : <ui16>, <ui16> -> <ui16>
                      %257 = ascvec.duplicate_scalar %c-1_i16, %mask : i16 -> <ui16>
                      %258 = ascvec.xor %252, %257, %mask : <ui16>, <ui16> -> <ui16>
                      %259 = ascvec.select %258, %256, %255 : <ui16>
                      ascvec.storealign %arg60[%c0_9], %259, %mask : memref<768xui16, 1>, index, <ui16>
                      %260 = ascvec.add_scalar %250#1, %cst, %mask : <bf16>, bf16
                      %261 = ascvec.reinterpret %260 : <bf16> -> <ui16>
                      %262 = ascvec.and %261, %253, %mask : <ui16>, <ui16> -> <ui16>
                      %263 = ascvec.cmp_eq_scalar %262, %c-32768_i16, %mask : <ui16>, i16 -> !ascvec.mask
                      %264 = ascvec.xor %261, %253, %mask : <ui16>, <ui16> -> <ui16>
                      %265 = ascvec.xor %261, %257, %mask : <ui16>, <ui16> -> <ui16>
                      %266 = ascvec.select %265, %264, %263 : <ui16>
                      ascvec.storealign %arg60[%c128], %266, %mask : memref<768xui16, 1>, index, <ui16>
                      %267 = ascvec.loadalign %arg61[%c58496] : memref<?xui16, 1>, index -> <ui16>
                      %268 = ascvec.select %243, %267, %245 : <ui16>
                      %269 = ascvec.loadalign %arg61[%c58752] : memref<?xui16, 1>, index -> <ui16>
                      %270 = ascvec.select %243, %269, %245 : <ui16>
                      %271 = ascvec.shr %259, 8, %mask : <ui16> -> <ui16>
                      %272 = ascvec.pack %271 src_elem_bits 16 dst_elem_bits 8 {part = "low"} : <ui16> -> <ui8>
                      %mask_13, %remaining_out_14 = ascvec.update_mask %arg62 elem_bits 8 : !ascvec.mask
                      %273 = ascvec.cumulative_histogram %268, %272, %mask_13 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                      %274 = ascvec.cumulative_histogram %270, %272, %mask_13 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                      %275 = ascvec.shr %266, 8, %mask : <ui16> -> <ui16>
                      %276 = ascvec.pack %275 src_elem_bits 16 dst_elem_bits 8 {part = "low"} : <ui16> -> <ui8>
                      %mask_15, %remaining_out_16 = ascvec.update_mask %arg63 elem_bits 8 : !ascvec.mask
                      %277 = ascvec.cumulative_histogram %273, %276, %mask_15 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                      %278 = ascvec.cumulative_histogram %274, %276, %mask_15 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                      ascvec.storealign %arg61[%c58496], %277, %mask : memref<?xui16, 1>, index, <ui16>
                      ascvec.storealign %arg61[%c58752], %278, %mask : memref<?xui16, 1>, index, <ui16>
                      %279 = ascvec.add_scalar %250#2, %cst, %mask : <bf16>, bf16
                      %280 = ascvec.reinterpret %279 : <bf16> -> <ui16>
                      %281 = ascvec.and %280, %253, %mask : <ui16>, <ui16> -> <ui16>
                      %282 = ascvec.cmp_eq_scalar %281, %c-32768_i16, %mask : <ui16>, i16 -> !ascvec.mask
                      %283 = ascvec.xor %280, %253, %mask : <ui16>, <ui16> -> <ui16>
                      %284 = ascvec.xor %280, %257, %mask : <ui16>, <ui16> -> <ui16>
                      %285 = ascvec.select %284, %283, %282 : <ui16>
                      ascvec.storealign %arg60[%c256], %285, %mask : memref<768xui16, 1>, index, <ui16>
                      %286 = ascvec.add_scalar %250#3, %cst, %mask : <bf16>, bf16
                      %287 = ascvec.reinterpret %286 : <bf16> -> <ui16>
                      %288 = ascvec.and %287, %253, %mask : <ui16>, <ui16> -> <ui16>
                      %289 = ascvec.cmp_eq_scalar %288, %c-32768_i16, %mask : <ui16>, i16 -> !ascvec.mask
                      %290 = ascvec.xor %287, %253, %mask : <ui16>, <ui16> -> <ui16>
                      %291 = ascvec.xor %287, %257, %mask : <ui16>, <ui16> -> <ui16>
                      %292 = ascvec.select %291, %290, %289 : <ui16>
                      ascvec.storealign %arg60[%c384], %292, %mask : memref<768xui16, 1>, index, <ui16>
                      %293 = ascvec.loadalign %arg61[%c59008] : memref<?xui16, 1>, index -> <ui16>
                      %294 = ascvec.select %243, %293, %245 : <ui16>
                      %295 = ascvec.loadalign %arg61[%c59264] : memref<?xui16, 1>, index -> <ui16>
                      %296 = ascvec.select %243, %295, %245 : <ui16>
                      %297 = ascvec.shr %285, 8, %mask : <ui16> -> <ui16>
                      %298 = ascvec.pack %297 src_elem_bits 16 dst_elem_bits 8 {part = "low"} : <ui16> -> <ui8>
                      %mask_17, %remaining_out_18 = ascvec.update_mask %arg64 elem_bits 8 : !ascvec.mask
                      %299 = ascvec.cumulative_histogram %294, %298, %mask_17 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                      %300 = ascvec.cumulative_histogram %296, %298, %mask_17 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                      %301 = ascvec.shr %292, 8, %mask : <ui16> -> <ui16>
                      %302 = ascvec.pack %301 src_elem_bits 16 dst_elem_bits 8 {part = "low"} : <ui16> -> <ui8>
                      %mask_19, %remaining_out_20 = ascvec.update_mask %arg65 elem_bits 8 : !ascvec.mask
                      %303 = ascvec.cumulative_histogram %299, %302, %mask_19 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                      %304 = ascvec.cumulative_histogram %300, %302, %mask_19 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                      ascvec.storealign %arg61[%c59008], %303, %mask : memref<?xui16, 1>, index, <ui16>
                      ascvec.storealign %arg61[%c59264], %304, %mask : memref<?xui16, 1>, index, <ui16>
                      %305 = ascvec.add_scalar %250#4, %cst, %mask : <bf16>, bf16
                      %306 = ascvec.reinterpret %305 : <bf16> -> <ui16>
                      %307 = ascvec.and %306, %253, %mask : <ui16>, <ui16> -> <ui16>
                      %308 = ascvec.cmp_eq_scalar %307, %c-32768_i16, %mask : <ui16>, i16 -> !ascvec.mask
                      %309 = ascvec.xor %306, %253, %mask : <ui16>, <ui16> -> <ui16>
                      %310 = ascvec.xor %306, %257, %mask : <ui16>, <ui16> -> <ui16>
                      %311 = ascvec.select %310, %309, %308 : <ui16>
                      ascvec.storealign %arg60[%c512], %311, %mask : memref<768xui16, 1>, index, <ui16>
                      %312 = ascvec.add_scalar %250#5, %cst, %mask : <bf16>, bf16
                      %313 = ascvec.reinterpret %312 : <bf16> -> <ui16>
                      %314 = ascvec.and %313, %253, %mask : <ui16>, <ui16> -> <ui16>
                      %315 = ascvec.cmp_eq_scalar %314, %c-32768_i16, %mask : <ui16>, i16 -> !ascvec.mask
                      %316 = ascvec.xor %313, %253, %mask : <ui16>, <ui16> -> <ui16>
                      %317 = ascvec.xor %313, %257, %mask : <ui16>, <ui16> -> <ui16>
                      %318 = ascvec.select %317, %316, %315 : <ui16>
                      ascvec.storealign %arg60[%c640], %318, %mask : memref<768xui16, 1>, index, <ui16>
                      %319 = ascvec.loadalign %arg61[%c59520] : memref<?xui16, 1>, index -> <ui16>
                      %320 = ascvec.select %243, %319, %245 : <ui16>
                      %321 = ascvec.loadalign %arg61[%c59776] : memref<?xui16, 1>, index -> <ui16>
                      %322 = ascvec.select %243, %321, %245 : <ui16>
                      %323 = ascvec.shr %311, 8, %mask : <ui16> -> <ui16>
                      %324 = ascvec.pack %323 src_elem_bits 16 dst_elem_bits 8 {part = "low"} : <ui16> -> <ui8>
                      %mask_21, %remaining_out_22 = ascvec.update_mask %arg66 elem_bits 8 : !ascvec.mask
                      %325 = ascvec.cumulative_histogram %320, %324, %mask_21 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                      %326 = ascvec.cumulative_histogram %322, %324, %mask_21 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                      %327 = ascvec.shr %318, 8, %mask : <ui16> -> <ui16>
                      %328 = ascvec.pack %327 src_elem_bits 16 dst_elem_bits 8 {part = "low"} : <ui16> -> <ui8>
                      %mask_23, %remaining_out_24 = ascvec.update_mask %arg67 elem_bits 8 : !ascvec.mask
                      %329 = ascvec.cumulative_histogram %325, %328, %mask_23 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                      %330 = ascvec.cumulative_histogram %326, %328, %mask_23 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                      ascvec.storealign %arg61[%c59520], %329, %mask : memref<?xui16, 1>, index, <ui16>
                      ascvec.storealign %arg61[%c59776], %330, %mask : memref<?xui16, 1>, index, <ui16>
                    }
                    ascvec.mem_bar vst_vld
                  } {cannir.vf_mode = "simd"}
                  ascvec.release_buf v, %225, 0 {cannir.buffer.generated_sync}
                  ascvec.sync_intra_arrive v, %c1_i64 {cannir.sync.banks = 3 : i32}
                }
                %191 = arith.subi %53, %109 : i32
                %192 = arith.cmpi slt, %191, %c0_i32 : i32
                %193 = arith.select %192, %c0_i32, %191 : i32
                %194 = arith.cmpi sgt, %193, %c3_i32 : i32
                %195 = arith.select %194, %c3_i32, %193 : i32
                %196 = arith.muli %5, %c6_i64 : i64
                %197 = arith.addi %196, %106 : i64
                %198 = arith.select %156, %c906240_i64, %c918528_i64 : i64
                %199 = arith.select %156, %c4_i64, %c5_i64 : i64
                %200 = cannir.extract_physical_layout %arg12 : !cannir.tensor<<gm, ui16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
                %201 = cannir.extract_stride %200 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
                %202 = cannir.extract_dim %201[0] : !cannir.stride<(-1, 1)>
                %203 = arith.muli %197, %202 : i64
                %204 = arith.addi %203, %165 : i64
                %205 = arith.muli %204, %c16_i64 : i64
                %206 = ascscalar.memref_address %alloc_2 : memref<1xui16>
                %207 = arith.divui %205, %c8_i64 : i64
                %208 = arith.addi %206, %207 : i64
                %209 = ascscalar.bind_gm_ptr %208 : memref<1xui16>
                %210 = arith.divui %198, %c8_i64 : i64
                %211 = arith.addi %130, %210 : i64
                %212 = ascscalar.bind_local_ptr %211 : memref<1xui16, 1>
                %213 = arith.index_cast %195 : i32 to index
                %214 = cannir.extract_layout %arg12 : !cannir.tensor<<gm, ui16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
                %215 = cannir.extract_stride %214 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
                %216 = cannir.extract_dim %215[0] : !cannir.stride<(-1, 1)>
                %217 = arith.index_cast %216 : i64 to index
                %218 = arith.muli %217, %c2 : index
                %219 = arith.index_cast %213 : index to i64
                %220 = arith.cmpi ugt, %219, %c0_i64 : i64
                scf.if %220 {
                  %221 = arith.index_cast %218 : index to i64
                  ascvec.get_buf mte3, %199, 0 {cannir.buffer.generated_sync}
                  %222 = arith.trunci %219 : i64 to i32
                  ascvec.copy_ub2gm %209[%c0], %212[%c0], %222, %c512_i32, %221, %c512_i32 l2_cache_ctl %c0_i32 : memref<1xui16>, memref<1xui16, 1>
                  ascvec.release_buf mte3, %199, 0 {cannir.buffer.generated_sync}
                }
                ascvec.sync_intra_arrive v, %c9_i64
                scf.yield %160, %158, %164 : i64, i64, i64
              }
              scf.yield %145, %154#0, %154#1, %154#2 : i64, i64, i64, i64
            } {cannir.loop_unroll = 1 : i64}
            ascvec.sync_notify {dst_pipe = 5 : i32, event_id = 0 : i32, src_pipe = 6 : i32}
            ascvec.sync_wait {dst_pipe = 5 : i32, event_id = 0 : i32, src_pipe = 6 : i32}
            scf.for %arg51 = %c0_i32 to %c3_i32 step %c1_i32  : i32 {
              %143 = arith.extsi %arg51 : i32 to i64
              %144 = arith.addi %106, %143 : i64
              %145 = arith.trunci %144 : i64 to i32
              %146 = arith.extsi %145 : i32 to i64
              %147 = arith.addi %50, %146 : i64
              %148 = arith.muli %5, %c6_i64 : i64
              %149 = arith.addi %148, %146 : i64
              %150 = arith.cmpi sge, %145, %56 : i32
              %151 = arith.select %150, %c0_i32, %31 : i32
              %152 = arith.subi %151, %85 : i32
              %153 = arith.cmpi slt, %152, %c0_i32 : i32
              %154 = arith.select %153, %c0_i32, %152 : i32
              %155 = arith.trunci %100 : i64 to i32
              %156 = arith.cmpi sgt, %154, %155 : i32
              %157 = arith.select %156, %155, %154 : i32
              %158 = arith.cmpi slt, %145, %53 : i32
              scf.if %158 {
                %159 = arith.cmpi ne, %arg34, %c0_i64 : i64
                %160 = arith.cmpi eq, %75, %false : i1
                %161 = arith.select %159, %160, %false : i1
                %162 = scf.if %161 -> (i32) {
                  %170 = ascscalar.bind_gm_ptr %arg34 : memref<1xi32>
                  %171 = arith.index_cast %147 : i64 to index
                  %172 = memref.load %170[%171] : memref<1xi32>
                  scf.yield %172 : i32
                } else {
                  scf.yield %c0_i32 : i32
                }
                %163 = arith.extsi %157 : i32 to i64
                %164 = arith.addi %85, %162 : i32
                %165 = arith.cmpi eq, %98, %c0_i64 : i64
                %166 = arith.cmpi sge, %163, %c8192_i64 : i64
                %167 = arith.cmpi sle, %163, %c16384_i64 : i64
                %168 = arith.select %166, %167, %false : i1
                %169 = arith.select %165, %168, %false : i1
                scf.if %169 {
                  %170 = arith.addi %163, %c127_i64 : i64
                  %171 = arith.divsi %170, %c128_i64 : i64
                  %172 = cannir.extract_physical_layout %arg12 : !cannir.tensor<<gm, ui16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
                  %173 = cannir.extract_stride %172 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
                  %174 = cannir.extract_dim %173[0] : !cannir.stride<(-1, 1)>
                  %175 = arith.muli %149, %174 : i64
                  %176 = arith.muli %175, %c2_i64 : i64
                  %177 = arith.muli %171, %c256_i64 : i64
                  ascvec.get_buf mte2, %c0_i64, 0 {cannir.buffer.generated_sync}
                  "ascvec.copy_gm2ub_plan"(%134, %alloc_2, %c0_i32, %c0_i32, %c0_i64, %176, %c1_i64, %177, %c0_i64, %177, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = ui16, ub_base = 0 : i64, ub_capacity = 262144 : i64}> : (memref<?xui16, 1>, memref<1xui16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                  ascvec.release_buf mte2, %c0_i64, 0 {cannir.buffer.generated_sync}
                  ascvec.vf {
                    %c128_i32_5 = arith.constant 128 : i32
                    %c0_6 = arith.constant 0 : index
                    %c1_7 = arith.constant 1 : index
                    scf.for %arg52 = %c0_6 to %c1_7 step %c1_7 {
                      %mask, %remaining_out = ascvec.update_mask %c128_i32_5 elem_bits 16 : !ascvec.mask
                    }
                    ascvec.mem_bar vst_vld
                  } {cannir.vf_mode = "simd"}
                  ascvec.get_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
                  scf.if %137 {
                    %185 = arith.muli %arg51, %c512_i32 overflow<nsw> : i32
                    %186 = arith.index_cast %185 : i32 to index
                    %187 = arith.addi %185, %c256_i32 overflow<nsw> : i32
                    %188 = arith.index_cast %187 : i32 to index
                    %189 = arith.addi %163, %c-8191_i64 : i64
                    %190 = arith.trunci %189 : i64 to i16
                    %191 = arith.divsi %163, %c1024_i64 : i64
                    %192 = arith.muli %191, %c4_i64 : i64
                    %193 = arith.addi %163, %c255_i64 : i64
                    %194 = arith.divsi %193, %c256_i64 : i64
                    ascvec.vf(%134, %186, %134, %188, %190, %134, %191, %163, %192, %194, %134, %171 : memref<?xui16, 1>, index, memref<?xui16, 1>, index, i16, memref<?xui16, 1>, i64, i64, i64, i64, memref<?xui16, 1>, i64) {
                    ^bb0(%arg52: memref<?xui16, 1>, %arg53: index, %arg54: memref<?xui16, 1>, %arg55: index, %arg56: i16, %arg57: memref<?xui16, 1>, %arg58: i64, %arg59: i64, %arg60: i64, %arg61: i64, %arg62: memref<?xui16, 1>, %arg63: i64):
                      %c128_i64_5 = arith.constant 128 : i64
                      %c24576 = arith.constant 24576 : index
                      %c1_i64_6 = arith.constant 1 : i64
                      %c0_i64_7 = arith.constant 0 : i64
                      %c768_i64 = arith.constant 768 : i64
                      %c512_i64_8 = arith.constant 512 : i64
                      %c256_i64_9 = arith.constant 256 : i64
                      %c1024_i64_10 = arith.constant 1024 : i64
                      %c256_i32_11 = arith.constant 256 : i32
                      %c127_i16 = arith.constant 127 : i16
                      %c1_i16 = arith.constant 1 : i16
                      %c128_i16 = arith.constant 128 : i16
                      %c0_i16 = arith.constant 0 : i16
                      %c-1_i16 = arith.constant -1 : i16
                      %c1_i32_12 = arith.constant 1 : i32
                      %c58496 = arith.constant 58496 : index
                      %c58368 = arith.constant 58368 : index
                      %c128_i32_13 = arith.constant 128 : i32
                      %mask, %remaining_out = ascvec.update_mask %c128_i32_13 elem_bits 16 : !ascvec.mask
                      %195 = arith.addi %arg53, %c58368 : index
                      %196 = ascvec.loadalign %arg52[%195] : memref<?xui16, 1>, index -> <ui16>
                      %197 = arith.addi %arg53, %c58496 : index
                      %198 = ascvec.loadalign %arg54[%197] : memref<?xui16, 1>, index -> <ui16>
                      %199 = ascvec.add %196, %198, %mask : <ui16>
                      %200 = arith.addi %arg55, %c58368 : index
                      %201 = ascvec.loadalign %arg52[%200] : memref<?xui16, 1>, index -> <ui16>
                      %202 = arith.addi %arg55, %c58496 : index
                      %203 = ascvec.loadalign %arg54[%202] : memref<?xui16, 1>, index -> <ui16>
                      %204 = ascvec.add %201, %203, %mask : <ui16>
                      %mask_14, %remaining_out_15 = ascvec.update_mask %c128_i32_13 elem_bits 16 : !ascvec.mask
                      %mask_16, %remaining_out_17 = ascvec.update_mask %c1_i32_12 elem_bits 16 : !ascvec.mask
                      %205 = ascvec.duplicate_scalar %c-1_i16, %mask_14 : i16 -> <ui16>
                      %206 = ascvec.arange %c0_i16 : i16 -> <ui16>
                      %207 = ascvec.cmp_ge_scalar %199, %arg56, %mask_14 : <ui16>, i16 -> !ascvec.mask
                      %208 = ascvec.select %206, %205, %207 : <ui16>
                      %209 = ascvec.arange %c128_i16 : i16 -> <ui16>
                      %210 = ascvec.cmp_ge_scalar %204, %arg56, %mask_14 : <ui16>, i16 -> !ascvec.mask
                      %211 = ascvec.select %209, %205, %210 : <ui16>
                      %212 = ascvec.min %208, %211, %mask_14 : <ui16>
                      %213 = ascvec.reduce_min %212, %mask_14 : <ui16>
                      %mask_18, %remaining_out_19 = ascvec.update_mask %c128_i32_13 elem_bits 16 : !ascvec.mask
                      %mask_20, %remaining_out_21 = ascvec.update_mask %c1_i32_12 elem_bits 16 : !ascvec.mask
                      %214 = ascvec.duplicate_scalar %c0_i16, %mask_18 : i16 -> <ui16>
                      %215 = ascvec.duplicate_scalar %c1_i16, %mask_18 : i16 -> <ui16>
                      %216 = ascvec.duplicate %213, %mask_18 : <ui16> -> <ui16>
                      %217 = ascvec.shl %216, 8, %mask_18 : <ui16> -> <ui16>
                      %218 = ascvec.or %216, %217, %mask_18 : <ui16>, <ui16> -> <ui16>
                      %219 = ascvec.reinterpret %218 : <ui16> -> <ui8>
                      %220 = ascvec.sub %216, %215, %mask_18 : <ui16>
                      %221 = ascvec.cmp_eq_scalar %216, %c0_i16, %mask_18 : <ui16>, i16 -> !ascvec.mask
                      %222 = ascvec.select %214, %220, %221 : <ui16>
                      %223 = ascvec.duplicate_scalar %c127_i16, %mask_18 : i16 -> <ui16>
                      %224 = ascvec.and %222, %223, %mask_18 : <ui16>, <ui16> -> <ui16>
                      %225 = ascvec.gather_reg %199, %224 : <ui16>, <ui16> -> <ui16>
                      %226 = ascvec.gather_reg %204, %224 : <ui16>, <ui16> -> <ui16>
                      %227 = ascvec.cmp_lt_scalar %222, %c128_i16, %mask_18 : <ui16>, i16 -> !ascvec.mask
                      %228 = ascvec.select %225, %226, %227 : <ui16>
                      %229 = ascvec.select %214, %228, %221 : <ui16>
                      %230 = ascvec.duplicate_scalar %arg56, %mask_18 : i16 -> <ui16>
                      %231 = ascvec.sub %230, %229, %mask_18 : <ui16>
                      %232 = ascvec.duplicate %231, %mask_18 : <ui16> -> <ui16>
                      %mask_22, %remaining_out_23 = ascvec.update_mask %c256_i32_11 elem_bits 8 : !ascvec.mask
                      %233:8 = scf.for %arg64 = %c0_i64_7 to %arg58 step %c1_i64_6 iter_args(%arg65 = %214, %arg66 = %214, %arg67 = %214, %arg68 = %214, %arg69 = %214, %arg70 = %214, %arg71 = %214, %arg72 = %214) -> (!ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>)  : i64 {
                        %251 = arith.muli %arg64, %c1024_i64_10 : i64
                        %252 = arith.index_cast %251 : i64 to index
                        %result0, %result1 = ascvec.loadalign_deintlv %arg57[%252] {dintlv_width = "b8"} : memref<?xui16, 1>, index -> <ui8>, <ui8>
                        %253 = ascvec.cmp_eq %result1, %219, %mask_22 : <ui8>, <ui8> -> !ascvec.mask
                        %254 = ascvec.cumulative_histogram %arg65, %result0, %253 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                        %255 = ascvec.cumulative_histogram %arg66, %result0, %253 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                        %256 = arith.addi %251, %c256_i64_9 : i64
                        %257 = arith.index_cast %256 : i64 to index
                        %result0_32, %result1_33 = ascvec.loadalign_deintlv %arg57[%257] {dintlv_width = "b8"} : memref<?xui16, 1>, index -> <ui8>, <ui8>
                        %258 = ascvec.cmp_eq %result1_33, %219, %mask_22 : <ui8>, <ui8> -> !ascvec.mask
                        %259 = ascvec.cumulative_histogram %arg67, %result0_32, %258 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                        %260 = ascvec.cumulative_histogram %arg68, %result0_32, %258 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                        %261 = arith.addi %251, %c512_i64_8 : i64
                        %262 = arith.index_cast %261 : i64 to index
                        %result0_34, %result1_35 = ascvec.loadalign_deintlv %arg57[%262] {dintlv_width = "b8"} : memref<?xui16, 1>, index -> <ui8>, <ui8>
                        %263 = ascvec.cmp_eq %result1_35, %219, %mask_22 : <ui8>, <ui8> -> !ascvec.mask
                        %264 = ascvec.cumulative_histogram %arg69, %result0_34, %263 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                        %265 = ascvec.cumulative_histogram %arg70, %result0_34, %263 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                        %266 = arith.addi %251, %c768_i64 : i64
                        %267 = arith.index_cast %266 : i64 to index
                        %result0_36, %result1_37 = ascvec.loadalign_deintlv %arg57[%267] {dintlv_width = "b8"} : memref<?xui16, 1>, index -> <ui8>, <ui8>
                        %268 = ascvec.cmp_eq %result1_37, %219, %mask_22 : <ui8>, <ui8> -> !ascvec.mask
                        %269 = ascvec.cumulative_histogram %arg71, %result0_36, %268 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                        %270 = ascvec.cumulative_histogram %arg72, %result0_36, %268 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                        scf.yield %254, %255, %259, %260, %264, %265, %269, %270 : !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>
                      } {cannir.loop_unroll = 1 : i64}
                      %234:2 = scf.for %arg64 = %arg60 to %arg61 step %c1_i64_6 iter_args(%arg65 = %233#0, %arg66 = %233#1) -> (!ascvec.vreg<ui16>, !ascvec.vreg<ui16>)  : i64 {
                        %251 = arith.muli %arg64, %c256_i64_9 : i64
                        %252 = arith.subi %arg59, %251 : i64
                        %253 = arith.trunci %252 : i64 to i32
                        %mask_32, %remaining_out_33 = ascvec.update_mask %253 elem_bits 8 : !ascvec.mask
                        %254 = arith.index_cast %251 : i64 to index
                        %result0, %result1 = ascvec.loadalign_deintlv %arg57[%254] {dintlv_width = "b8"} : memref<?xui16, 1>, index -> <ui8>, <ui8>
                        %255 = ascvec.cmp_eq %result1, %219, %mask_32 : <ui8>, <ui8> -> !ascvec.mask
                        %256 = ascvec.cumulative_histogram %arg65, %result0, %255 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                        %257 = ascvec.cumulative_histogram %arg66, %result0, %255 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                        scf.yield %256, %257 : !ascvec.vreg<ui16>, !ascvec.vreg<ui16>
                      } {cannir.loop_unroll = 1 : i64}
                      %235 = ascvec.add %234#0, %233#2, %mask_18 : <ui16>
                      %236 = ascvec.add %235, %233#4, %mask_18 : <ui16>
                      %237 = ascvec.add %236, %233#6, %mask_18 : <ui16>
                      %238 = ascvec.add %234#1, %233#3, %mask_18 : <ui16>
                      %239 = ascvec.add %238, %233#5, %mask_18 : <ui16>
                      %240 = ascvec.add %239, %233#7, %mask_18 : <ui16>
                      %mask_24, %remaining_out_25 = ascvec.update_mask %c128_i32_13 elem_bits 16 : !ascvec.mask
                      %mask_26, %remaining_out_27 = ascvec.update_mask %c1_i32_12 elem_bits 16 : !ascvec.mask
                      %241 = ascvec.duplicate_scalar %c-1_i16, %mask_24 : i16 -> <ui16>
                      %242 = ascvec.cmp_ge %237, %232, %mask_24 : <ui16>, <ui16> -> !ascvec.mask
                      %243 = ascvec.select %206, %241, %242 : <ui16>
                      %244 = ascvec.cmp_ge %240, %232, %mask_24 : <ui16>, <ui16> -> !ascvec.mask
                      %245 = ascvec.select %209, %241, %244 : <ui16>
                      %246 = ascvec.min %243, %245, %mask_24 : <ui16>
                      %247 = ascvec.reduce_min %246, %mask_24 : <ui16>
                      %mask_28, %remaining_out_29 = ascvec.update_mask %c128_i32_13 elem_bits 16 : !ascvec.mask
                      %248 = ascvec.duplicate %247, %mask_28 : <ui16> -> <ui16>
                      %249 = ascvec.shl %216, 8, %mask_28 : <ui16> -> <ui16>
                      %250 = ascvec.or %249, %248, %mask_28 : <ui16>, <ui16> -> <ui16>
                      ascvec.store_unalign_begin %arg62 : memref<?xui16, 1>
                      scf.for %arg64 = %c0_i64_7 to %arg63 step %c1_i64_6  : i64 {
                        %251 = arith.muli %arg64, %c128_i64_5 : i64
                        %252 = arith.subi %arg59, %251 : i64
                        %253 = arith.trunci %252 : i64 to i32
                        %mask_32, %remaining_out_33 = ascvec.update_mask %253 elem_bits 16 : !ascvec.mask
                        %254 = arith.index_cast %251 : i64 to index
                        %255 = ascvec.loadalign %arg57[%254] : memref<?xui16, 1>, index -> <ui16>
                        %256 = arith.trunci %251 : i64 to i16
                        %257 = ascvec.arange %256 : i16 -> <ui16>
                        %258 = ascvec.cmp_gt %255, %250, %mask_32 : <ui16>, <ui16> -> !ascvec.mask
                        %259 = ascvec.squeeze_store_reg %257, %258 : <ui16>, !ascvec.mask -> <ui16>
                        ascvec.store_unalign %arg62[%c24576], %259 : memref<?xui16, 1>, <ui16>
                      } {cannir.loop_unroll = 1 : i64}
                      scf.for %arg64 = %c0_i64_7 to %arg63 step %c1_i64_6  : i64 {
                        %251 = arith.muli %arg64, %c128_i64_5 : i64
                        %252 = arith.subi %arg59, %251 : i64
                        %253 = arith.trunci %252 : i64 to i32
                        %mask_32, %remaining_out_33 = ascvec.update_mask %253 elem_bits 16 : !ascvec.mask
                        %254 = arith.index_cast %251 : i64 to index
                        %255 = ascvec.loadalign %arg57[%254] : memref<?xui16, 1>, index -> <ui16>
                        %256 = arith.trunci %251 : i64 to i16
                        %257 = ascvec.arange %256 : i16 -> <ui16>
                        %258 = ascvec.cmp_eq %255, %250, %mask_32 : <ui16>, <ui16> -> !ascvec.mask
                        %259 = ascvec.squeeze_store_reg %257, %258 : <ui16>, !ascvec.mask -> <ui16>
                        ascvec.store_unalign %arg62[%c24576], %259 : memref<?xui16, 1>, <ui16>
                      } {cannir.loop_unroll = 1 : i64}
                      %c0_30 = arith.constant 0 : index
                      %c1_31 = arith.constant 1 : index
                      scf.for %arg64 = %c0_30 to %c1_31 step %c1_31 {
                        ascvec.store_unalign_post %arg62[%c24576] : memref<?xui16, 1>
                      }
                      ascvec.mem_bar vst_vld
                    } {cannir.vf_mode = "simd"}
                  } else {
                    %185 = arith.divsi %163, %c1024_i64 : i64
                    %186 = arith.muli %185, %c4_i64 : i64
                    %187 = arith.addi %163, %c255_i64 : i64
                    %188 = arith.divsi %187, %c256_i64 : i64
                    %189 = arith.addi %163, %c-8191_i64 : i64
                    %190 = arith.trunci %189 : i64 to i16
                    ascvec.vf(%134, %185, %163, %186, %188, %190, %134, %171 : memref<?xui16, 1>, i64, i64, i64, i64, i16, memref<?xui16, 1>, i64) {
                    ^bb0(%arg52: memref<?xui16, 1>, %arg53: i64, %arg54: i64, %arg55: i64, %arg56: i64, %arg57: i16, %arg58: memref<?xui16, 1>, %arg59: i64):
                      %c128_i64_5 = arith.constant 128 : i64
                      %c24576 = arith.constant 24576 : index
                      %c127_i16 = arith.constant 127 : i16
                      %c1_i16 = arith.constant 1 : i16
                      %c128_i16 = arith.constant 128 : i16
                      %c-1_i16 = arith.constant -1 : i16
                      %c1_i32_6 = arith.constant 1 : i32
                      %c1_i64_7 = arith.constant 1 : i64
                      %c0_i64_8 = arith.constant 0 : i64
                      %c768_i64 = arith.constant 768 : i64
                      %c512_i64_9 = arith.constant 512 : i64
                      %c256_i64_10 = arith.constant 256 : i64
                      %c1024_i64_11 = arith.constant 1024 : i64
                      %c256_i32_12 = arith.constant 256 : i32
                      %c0_i16 = arith.constant 0 : i16
                      %c128_i32_13 = arith.constant 128 : i32
                      %mask, %remaining_out = ascvec.update_mask %c128_i32_13 elem_bits 16 : !ascvec.mask
                      %191 = ascvec.duplicate_scalar %c0_i16, %mask : i16 -> <ui16>
                      %mask_14, %remaining_out_15 = ascvec.update_mask %c256_i32_12 elem_bits 8 : !ascvec.mask
                      %192:8 = scf.for %arg60 = %c0_i64_8 to %arg53 step %c1_i64_7 iter_args(%arg61 = %191, %arg62 = %191, %arg63 = %191, %arg64 = %191, %arg65 = %191, %arg66 = %191, %arg67 = %191, %arg68 = %191) -> (!ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>)  : i64 {
                        %246 = arith.muli %arg60, %c1024_i64_11 : i64
                        %247 = arith.index_cast %246 : i64 to index
                        %result0, %result1 = ascvec.loadalign_deintlv %arg52[%247] {dintlv_width = "b8"} : memref<?xui16, 1>, index -> <ui8>, <ui8>
                        %248 = ascvec.cumulative_histogram %arg61, %result1, %mask_14 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                        %249 = ascvec.cumulative_histogram %arg62, %result1, %mask_14 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                        %250 = arith.addi %246, %c256_i64_10 : i64
                        %251 = arith.index_cast %250 : i64 to index
                        %result0_34, %result1_35 = ascvec.loadalign_deintlv %arg52[%251] {dintlv_width = "b8"} : memref<?xui16, 1>, index -> <ui8>, <ui8>
                        %252 = ascvec.cumulative_histogram %arg63, %result1_35, %mask_14 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                        %253 = ascvec.cumulative_histogram %arg64, %result1_35, %mask_14 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                        %254 = arith.addi %246, %c512_i64_9 : i64
                        %255 = arith.index_cast %254 : i64 to index
                        %result0_36, %result1_37 = ascvec.loadalign_deintlv %arg52[%255] {dintlv_width = "b8"} : memref<?xui16, 1>, index -> <ui8>, <ui8>
                        %256 = ascvec.cumulative_histogram %arg65, %result1_37, %mask_14 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                        %257 = ascvec.cumulative_histogram %arg66, %result1_37, %mask_14 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                        %258 = arith.addi %246, %c768_i64 : i64
                        %259 = arith.index_cast %258 : i64 to index
                        %result0_38, %result1_39 = ascvec.loadalign_deintlv %arg52[%259] {dintlv_width = "b8"} : memref<?xui16, 1>, index -> <ui8>, <ui8>
                        %260 = ascvec.cumulative_histogram %arg67, %result1_39, %mask_14 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                        %261 = ascvec.cumulative_histogram %arg68, %result1_39, %mask_14 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                        scf.yield %248, %249, %252, %253, %256, %257, %260, %261 : !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>
                      } {cannir.loop_unroll = 1 : i64}
                      %193:2 = scf.for %arg60 = %arg55 to %arg56 step %c1_i64_7 iter_args(%arg61 = %192#0, %arg62 = %192#1) -> (!ascvec.vreg<ui16>, !ascvec.vreg<ui16>)  : i64 {
                        %246 = arith.muli %arg60, %c256_i64_10 : i64
                        %247 = arith.subi %arg54, %246 : i64
                        %248 = arith.trunci %247 : i64 to i32
                        %mask_34, %remaining_out_35 = ascvec.update_mask %248 elem_bits 8 : !ascvec.mask
                        %249 = arith.index_cast %246 : i64 to index
                        %result0, %result1 = ascvec.loadalign_deintlv %arg52[%249] {dintlv_width = "b8"} : memref<?xui16, 1>, index -> <ui8>, <ui8>
                        %250 = ascvec.cumulative_histogram %arg61, %result1, %mask_34 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                        %251 = ascvec.cumulative_histogram %arg62, %result1, %mask_34 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                        scf.yield %250, %251 : !ascvec.vreg<ui16>, !ascvec.vreg<ui16>
                      } {cannir.loop_unroll = 1 : i64}
                      %194 = ascvec.add %193#0, %192#2, %mask : <ui16>
                      %195 = ascvec.add %194, %192#4, %mask : <ui16>
                      %196 = ascvec.add %195, %192#6, %mask : <ui16>
                      %197 = ascvec.add %193#1, %192#3, %mask : <ui16>
                      %198 = ascvec.add %197, %192#5, %mask : <ui16>
                      %199 = ascvec.add %198, %192#7, %mask : <ui16>
                      %mask_16, %remaining_out_17 = ascvec.update_mask %c128_i32_13 elem_bits 16 : !ascvec.mask
                      %mask_18, %remaining_out_19 = ascvec.update_mask %c1_i32_6 elem_bits 16 : !ascvec.mask
                      %200 = ascvec.duplicate_scalar %c-1_i16, %mask_16 : i16 -> <ui16>
                      %201 = ascvec.arange %c0_i16 : i16 -> <ui16>
                      %202 = ascvec.cmp_ge_scalar %196, %arg57, %mask_16 : <ui16>, i16 -> !ascvec.mask
                      %203 = ascvec.select %201, %200, %202 : <ui16>
                      %204 = ascvec.arange %c128_i16 : i16 -> <ui16>
                      %205 = ascvec.cmp_ge_scalar %199, %arg57, %mask_16 : <ui16>, i16 -> !ascvec.mask
                      %206 = ascvec.select %204, %200, %205 : <ui16>
                      %207 = ascvec.min %203, %206, %mask_16 : <ui16>
                      %208 = ascvec.reduce_min %207, %mask_16 : <ui16>
                      %mask_20, %remaining_out_21 = ascvec.update_mask %c128_i32_13 elem_bits 16 : !ascvec.mask
                      %mask_22, %remaining_out_23 = ascvec.update_mask %c1_i32_6 elem_bits 16 : !ascvec.mask
                      %209 = ascvec.duplicate_scalar %c0_i16, %mask_20 : i16 -> <ui16>
                      %210 = ascvec.duplicate_scalar %c1_i16, %mask_20 : i16 -> <ui16>
                      %211 = ascvec.duplicate %208, %mask_20 : <ui16> -> <ui16>
                      %212 = ascvec.shl %211, 8, %mask_20 : <ui16> -> <ui16>
                      %213 = ascvec.or %211, %212, %mask_20 : <ui16>, <ui16> -> <ui16>
                      %214 = ascvec.reinterpret %213 : <ui16> -> <ui8>
                      %215 = ascvec.sub %211, %210, %mask_20 : <ui16>
                      %216 = ascvec.cmp_eq_scalar %211, %c0_i16, %mask_20 : <ui16>, i16 -> !ascvec.mask
                      %217 = ascvec.select %209, %215, %216 : <ui16>
                      %218 = ascvec.duplicate_scalar %c127_i16, %mask_20 : i16 -> <ui16>
                      %219 = ascvec.and %217, %218, %mask_20 : <ui16>, <ui16> -> <ui16>
                      %220 = ascvec.gather_reg %196, %219 : <ui16>, <ui16> -> <ui16>
                      %221 = ascvec.gather_reg %199, %219 : <ui16>, <ui16> -> <ui16>
                      %222 = ascvec.cmp_lt_scalar %217, %c128_i16, %mask_20 : <ui16>, i16 -> !ascvec.mask
                      %223 = ascvec.select %220, %221, %222 : <ui16>
                      %224 = ascvec.select %209, %223, %216 : <ui16>
                      %225 = ascvec.duplicate_scalar %arg57, %mask_20 : i16 -> <ui16>
                      %226 = ascvec.sub %225, %224, %mask_20 : <ui16>
                      %227 = ascvec.duplicate %226, %mask_20 : <ui16> -> <ui16>
                      %mask_24, %remaining_out_25 = ascvec.update_mask %c256_i32_12 elem_bits 8 : !ascvec.mask
                      %228:8 = scf.for %arg60 = %c0_i64_8 to %arg53 step %c1_i64_7 iter_args(%arg61 = %209, %arg62 = %209, %arg63 = %209, %arg64 = %209, %arg65 = %209, %arg66 = %209, %arg67 = %209, %arg68 = %209) -> (!ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>)  : i64 {
                        %246 = arith.muli %arg60, %c1024_i64_11 : i64
                        %247 = arith.index_cast %246 : i64 to index
                        %result0, %result1 = ascvec.loadalign_deintlv %arg52[%247] {dintlv_width = "b8"} : memref<?xui16, 1>, index -> <ui8>, <ui8>
                        %248 = ascvec.cmp_eq %result1, %214, %mask_24 : <ui8>, <ui8> -> !ascvec.mask
                        %249 = ascvec.cumulative_histogram %arg61, %result0, %248 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                        %250 = ascvec.cumulative_histogram %arg62, %result0, %248 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                        %251 = arith.addi %246, %c256_i64_10 : i64
                        %252 = arith.index_cast %251 : i64 to index
                        %result0_34, %result1_35 = ascvec.loadalign_deintlv %arg52[%252] {dintlv_width = "b8"} : memref<?xui16, 1>, index -> <ui8>, <ui8>
                        %253 = ascvec.cmp_eq %result1_35, %214, %mask_24 : <ui8>, <ui8> -> !ascvec.mask
                        %254 = ascvec.cumulative_histogram %arg63, %result0_34, %253 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                        %255 = ascvec.cumulative_histogram %arg64, %result0_34, %253 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                        %256 = arith.addi %246, %c512_i64_9 : i64
                        %257 = arith.index_cast %256 : i64 to index
                        %result0_36, %result1_37 = ascvec.loadalign_deintlv %arg52[%257] {dintlv_width = "b8"} : memref<?xui16, 1>, index -> <ui8>, <ui8>
                        %258 = ascvec.cmp_eq %result1_37, %214, %mask_24 : <ui8>, <ui8> -> !ascvec.mask
                        %259 = ascvec.cumulative_histogram %arg65, %result0_36, %258 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                        %260 = ascvec.cumulative_histogram %arg66, %result0_36, %258 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                        %261 = arith.addi %246, %c768_i64 : i64
                        %262 = arith.index_cast %261 : i64 to index
                        %result0_38, %result1_39 = ascvec.loadalign_deintlv %arg52[%262] {dintlv_width = "b8"} : memref<?xui16, 1>, index -> <ui8>, <ui8>
                        %263 = ascvec.cmp_eq %result1_39, %214, %mask_24 : <ui8>, <ui8> -> !ascvec.mask
                        %264 = ascvec.cumulative_histogram %arg67, %result0_38, %263 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                        %265 = ascvec.cumulative_histogram %arg68, %result0_38, %263 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                        scf.yield %249, %250, %254, %255, %259, %260, %264, %265 : !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>
                      } {cannir.loop_unroll = 1 : i64}
                      %229:2 = scf.for %arg60 = %arg55 to %arg56 step %c1_i64_7 iter_args(%arg61 = %228#0, %arg62 = %228#1) -> (!ascvec.vreg<ui16>, !ascvec.vreg<ui16>)  : i64 {
                        %246 = arith.muli %arg60, %c256_i64_10 : i64
                        %247 = arith.subi %arg54, %246 : i64
                        %248 = arith.trunci %247 : i64 to i32
                        %mask_34, %remaining_out_35 = ascvec.update_mask %248 elem_bits 8 : !ascvec.mask
                        %249 = arith.index_cast %246 : i64 to index
                        %result0, %result1 = ascvec.loadalign_deintlv %arg52[%249] {dintlv_width = "b8"} : memref<?xui16, 1>, index -> <ui8>, <ui8>
                        %250 = ascvec.cmp_eq %result1, %214, %mask_34 : <ui8>, <ui8> -> !ascvec.mask
                        %251 = ascvec.cumulative_histogram %arg61, %result0, %250 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                        %252 = ascvec.cumulative_histogram %arg62, %result0, %250 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                        scf.yield %251, %252 : !ascvec.vreg<ui16>, !ascvec.vreg<ui16>
                      } {cannir.loop_unroll = 1 : i64}
                      %230 = ascvec.add %229#0, %228#2, %mask_20 : <ui16>
                      %231 = ascvec.add %230, %228#4, %mask_20 : <ui16>
                      %232 = ascvec.add %231, %228#6, %mask_20 : <ui16>
                      %233 = ascvec.add %229#1, %228#3, %mask_20 : <ui16>
                      %234 = ascvec.add %233, %228#5, %mask_20 : <ui16>
                      %235 = ascvec.add %234, %228#7, %mask_20 : <ui16>
                      %mask_26, %remaining_out_27 = ascvec.update_mask %c128_i32_13 elem_bits 16 : !ascvec.mask
                      %mask_28, %remaining_out_29 = ascvec.update_mask %c1_i32_6 elem_bits 16 : !ascvec.mask
                      %236 = ascvec.duplicate_scalar %c-1_i16, %mask_26 : i16 -> <ui16>
                      %237 = ascvec.cmp_ge %232, %227, %mask_26 : <ui16>, <ui16> -> !ascvec.mask
                      %238 = ascvec.select %201, %236, %237 : <ui16>
                      %239 = ascvec.cmp_ge %235, %227, %mask_26 : <ui16>, <ui16> -> !ascvec.mask
                      %240 = ascvec.select %204, %236, %239 : <ui16>
                      %241 = ascvec.min %238, %240, %mask_26 : <ui16>
                      %242 = ascvec.reduce_min %241, %mask_26 : <ui16>
                      %mask_30, %remaining_out_31 = ascvec.update_mask %c128_i32_13 elem_bits 16 : !ascvec.mask
                      %243 = ascvec.duplicate %242, %mask_30 : <ui16> -> <ui16>
                      %244 = ascvec.shl %211, 8, %mask_30 : <ui16> -> <ui16>
                      %245 = ascvec.or %244, %243, %mask_30 : <ui16>, <ui16> -> <ui16>
                      ascvec.store_unalign_begin %arg58 : memref<?xui16, 1>
                      scf.for %arg60 = %c0_i64_8 to %arg59 step %c1_i64_7  : i64 {
                        %246 = arith.muli %arg60, %c128_i64_5 : i64
                        %247 = arith.subi %arg54, %246 : i64
                        %248 = arith.trunci %247 : i64 to i32
                        %mask_34, %remaining_out_35 = ascvec.update_mask %248 elem_bits 16 : !ascvec.mask
                        %249 = arith.index_cast %246 : i64 to index
                        %250 = ascvec.loadalign %arg52[%249] : memref<?xui16, 1>, index -> <ui16>
                        %251 = arith.trunci %246 : i64 to i16
                        %252 = ascvec.arange %251 : i16 -> <ui16>
                        %253 = ascvec.cmp_gt %250, %245, %mask_34 : <ui16>, <ui16> -> !ascvec.mask
                        %254 = ascvec.squeeze_store_reg %252, %253 : <ui16>, !ascvec.mask -> <ui16>
                        ascvec.store_unalign %arg58[%c24576], %254 : memref<?xui16, 1>, <ui16>
                      } {cannir.loop_unroll = 1 : i64}
                      scf.for %arg60 = %c0_i64_8 to %arg59 step %c1_i64_7  : i64 {
                        %246 = arith.muli %arg60, %c128_i64_5 : i64
                        %247 = arith.subi %arg54, %246 : i64
                        %248 = arith.trunci %247 : i64 to i32
                        %mask_34, %remaining_out_35 = ascvec.update_mask %248 elem_bits 16 : !ascvec.mask
                        %249 = arith.index_cast %246 : i64 to index
                        %250 = ascvec.loadalign %arg52[%249] : memref<?xui16, 1>, index -> <ui16>
                        %251 = arith.trunci %246 : i64 to i16
                        %252 = ascvec.arange %251 : i16 -> <ui16>
                        %253 = ascvec.cmp_eq %250, %245, %mask_34 : <ui16>, <ui16> -> !ascvec.mask
                        %254 = ascvec.squeeze_store_reg %252, %253 : <ui16>, !ascvec.mask -> <ui16>
                        ascvec.store_unalign %arg58[%c24576], %254 : memref<?xui16, 1>, <ui16>
                      } {cannir.loop_unroll = 1 : i64}
                      %c0_32 = arith.constant 0 : index
                      %c1_33 = arith.constant 1 : index
                      scf.for %arg60 = %c0_32 to %c1_33 step %c1_33 {
                        ascvec.store_unalign_post %arg58[%c24576] : memref<?xui16, 1>
                      }
                      ascvec.mem_bar vst_vld
                    } {cannir.vf_mode = "simd"}
                  }
                  ascvec.release_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
                  %178 = arith.addi %130, %c218112_i64 : i64
                  %179 = ascscalar.bind_local_ptr %178 : memref<8192xi32, 1>
                  ascvec.get_buf v, %c1_i64, 0 {cannir.buffer.generated_sync}
                  ascvec.vf(%134, %164, %157, %179 : memref<?xui16, 1>, i32, i32, memref<8192xi32, 1>) {
                  ^bb0(%arg52: memref<?xui16, 1>, %arg53: i32, %arg54: i32, %arg55: memref<8192xi32, 1>):
                    %c1_i32_5 = arith.constant 1 : i32
                    %c128_i32_6 = arith.constant 128 : i32
                    %c0_i32_7 = arith.constant 0 : i32
                    %c-1_i32_8 = arith.constant -1 : i32
                    %c24576 = arith.constant 24576 : index
                    %c8192_i32_9 = arith.constant 8192 : i32
                    %c64_i32 = arith.constant 64 : i32
                    scf.for %arg56 = %c0_i32_7 to %c128_i32_6 step %c1_i32_5  : i32 {
                      %185 = arith.muli %arg56, %c64_i32 overflow<nsw> : i32
                      %186 = arith.subi %c8192_i32_9, %185 overflow<nsw> : i32
                      %mask, %remaining_out = ascvec.update_mask %186 : !ascvec.mask
                      %187 = arith.index_cast %185 : i32 to index
                      %188 = arith.addi %187, %c24576 : index
                      %189 = ascvec.loadalign %arg52[%188] : memref<?xui16, 1>, index -> <ui16>
                      %190 = ascvec.unpack %189 {part = "lower"} : <ui16> -> <ui32>
                      %191 = ascvec.reinterpret %190 : <ui32> -> <i32>
                      %192 = ascvec.add_scalar %191, %arg53, %mask : <i32>, i32
                      %193 = ascvec.arange %185 : i32 -> <ui32>
                      %194 = ascvec.cmp_lt_scalar %193, %arg54, %mask : <ui32>, i32 -> !ascvec.mask
                      %195 = ascvec.duplicate_scalar %c-1_i32_8, %mask : i32 -> <i32>
                      %196 = ascvec.select %192, %195, %194 : <i32>
                      ascvec.storealign %arg55[%187], %196, %mask : memref<8192xi32, 1>, index, <i32>
                    } {cannir.loop_unroll = 1 : i64}
                    ascvec.mem_bar vst_vld
                  } {cannir.vf_mode = "simd"}
                  ascvec.release_buf v, %c1_i64, 0 {cannir.buffer.generated_sync}
                  %180 = arith.muli %147, %c262144_i64 : i64
                  %181 = arith.divui %180, %c8_i64 : i64
                  %182 = arith.addi %82, %181 : i64
                  %183 = ascscalar.bind_gm_ptr %182 : memref<1xi32>
                  %184 = ascscalar.bind_local_ptr %178 : memref<1xi32, 1>
                  ascvec.get_buf mte3, %c1_i64, 0 {cannir.buffer.generated_sync}
                  ascvec.copy_ub2gm %183[%c0], %184[%c0], %c1_i32, %c32768_i32, %c0_i64, %c32768_i32 l2_cache_ctl %c0_i32 : memref<1xi32>, memref<1xi32, 1>
                  ascvec.release_buf mte3, %c1_i64, 0 {cannir.buffer.generated_sync}
                } else {
                  %170 = arith.cmpi ne, %98, %c0_i64 : i64
                  %171 = arith.select %170, %168, %false : i1
                  scf.if %171 {
                    %172 = arith.addi %163, %c127_i64 : i64
                    %173 = arith.divsi %172, %c128_i64 : i64
                    %174 = cannir.extract_physical_layout %arg12 : !cannir.tensor<<gm, ui16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
                    %175 = cannir.extract_stride %174 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
                    %176 = cannir.extract_dim %175[0] : !cannir.stride<(-1, 1)>
                    %177 = arith.muli %149, %176 : i64
                    %178 = arith.muli %177, %c2_i64 : i64
                    %179 = arith.muli %173, %c256_i64 : i64
                    ascvec.get_buf mte2, %c0_i64, 0 {cannir.buffer.generated_sync}
                    "ascvec.copy_gm2ub_plan"(%134, %alloc_2, %c0_i32, %c0_i32, %c0_i64, %178, %c1_i64, %179, %c0_i64, %179, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = ui16, ub_base = 0 : i64, ub_capacity = 262144 : i64}> : (memref<?xui16, 1>, memref<1xui16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                    ascvec.release_buf mte2, %c0_i64, 0 {cannir.buffer.generated_sync}
                    ascvec.vf {
                      %c128_i32_5 = arith.constant 128 : i32
                      %c0_6 = arith.constant 0 : index
                      %c1_7 = arith.constant 1 : index
                      scf.for %arg52 = %c0_6 to %c1_7 step %c1_7 {
                        %mask, %remaining_out = ascvec.update_mask %c128_i32_5 elem_bits 16 : !ascvec.mask
                      }
                      ascvec.mem_bar vst_vld
                    } {cannir.vf_mode = "simd"}
                    ascvec.get_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
                    scf.if %137 {
                      %194 = arith.muli %arg51, %c512_i32 overflow<nsw> : i32
                      %195 = arith.index_cast %194 : i32 to index
                      %196 = arith.addi %194, %c256_i32 overflow<nsw> : i32
                      %197 = arith.index_cast %196 : i32 to index
                      %198 = arith.addi %163, %c-8191_i64 : i64
                      %199 = arith.trunci %198 : i64 to i16
                      %200 = arith.divsi %163, %c1024_i64 : i64
                      %201 = arith.muli %200, %c4_i64 : i64
                      %202 = arith.addi %163, %c255_i64 : i64
                      %203 = arith.divsi %202, %c256_i64 : i64
                      ascvec.vf(%134, %195, %134, %197, %199, %134, %200, %163, %201, %203, %134, %173 : memref<?xui16, 1>, index, memref<?xui16, 1>, index, i16, memref<?xui16, 1>, i64, i64, i64, i64, memref<?xui16, 1>, i64) {
                      ^bb0(%arg52: memref<?xui16, 1>, %arg53: index, %arg54: memref<?xui16, 1>, %arg55: index, %arg56: i16, %arg57: memref<?xui16, 1>, %arg58: i64, %arg59: i64, %arg60: i64, %arg61: i64, %arg62: memref<?xui16, 1>, %arg63: i64):
                        %c128_i64_5 = arith.constant 128 : i64
                        %c24576 = arith.constant 24576 : index
                        %c1_i64_6 = arith.constant 1 : i64
                        %c0_i64_7 = arith.constant 0 : i64
                        %c768_i64 = arith.constant 768 : i64
                        %c512_i64_8 = arith.constant 512 : i64
                        %c256_i64_9 = arith.constant 256 : i64
                        %c1024_i64_10 = arith.constant 1024 : i64
                        %c256_i32_11 = arith.constant 256 : i32
                        %c127_i16 = arith.constant 127 : i16
                        %c1_i16 = arith.constant 1 : i16
                        %c128_i16 = arith.constant 128 : i16
                        %c0_i16 = arith.constant 0 : i16
                        %c-1_i16 = arith.constant -1 : i16
                        %c1_i32_12 = arith.constant 1 : i32
                        %c58496 = arith.constant 58496 : index
                        %c58368 = arith.constant 58368 : index
                        %c128_i32_13 = arith.constant 128 : i32
                        %mask, %remaining_out = ascvec.update_mask %c128_i32_13 elem_bits 16 : !ascvec.mask
                        %204 = arith.addi %arg53, %c58368 : index
                        %205 = ascvec.loadalign %arg52[%204] : memref<?xui16, 1>, index -> <ui16>
                        %206 = arith.addi %arg53, %c58496 : index
                        %207 = ascvec.loadalign %arg54[%206] : memref<?xui16, 1>, index -> <ui16>
                        %208 = ascvec.add %205, %207, %mask : <ui16>
                        %209 = arith.addi %arg55, %c58368 : index
                        %210 = ascvec.loadalign %arg52[%209] : memref<?xui16, 1>, index -> <ui16>
                        %211 = arith.addi %arg55, %c58496 : index
                        %212 = ascvec.loadalign %arg54[%211] : memref<?xui16, 1>, index -> <ui16>
                        %213 = ascvec.add %210, %212, %mask : <ui16>
                        %mask_14, %remaining_out_15 = ascvec.update_mask %c128_i32_13 elem_bits 16 : !ascvec.mask
                        %mask_16, %remaining_out_17 = ascvec.update_mask %c1_i32_12 elem_bits 16 : !ascvec.mask
                        %214 = ascvec.duplicate_scalar %c-1_i16, %mask_14 : i16 -> <ui16>
                        %215 = ascvec.arange %c0_i16 : i16 -> <ui16>
                        %216 = ascvec.cmp_ge_scalar %208, %arg56, %mask_14 : <ui16>, i16 -> !ascvec.mask
                        %217 = ascvec.select %215, %214, %216 : <ui16>
                        %218 = ascvec.arange %c128_i16 : i16 -> <ui16>
                        %219 = ascvec.cmp_ge_scalar %213, %arg56, %mask_14 : <ui16>, i16 -> !ascvec.mask
                        %220 = ascvec.select %218, %214, %219 : <ui16>
                        %221 = ascvec.min %217, %220, %mask_14 : <ui16>
                        %222 = ascvec.reduce_min %221, %mask_14 : <ui16>
                        %mask_18, %remaining_out_19 = ascvec.update_mask %c128_i32_13 elem_bits 16 : !ascvec.mask
                        %mask_20, %remaining_out_21 = ascvec.update_mask %c1_i32_12 elem_bits 16 : !ascvec.mask
                        %223 = ascvec.duplicate_scalar %c0_i16, %mask_18 : i16 -> <ui16>
                        %224 = ascvec.duplicate_scalar %c1_i16, %mask_18 : i16 -> <ui16>
                        %225 = ascvec.duplicate %222, %mask_18 : <ui16> -> <ui16>
                        %226 = ascvec.shl %225, 8, %mask_18 : <ui16> -> <ui16>
                        %227 = ascvec.or %225, %226, %mask_18 : <ui16>, <ui16> -> <ui16>
                        %228 = ascvec.reinterpret %227 : <ui16> -> <ui8>
                        %229 = ascvec.sub %225, %224, %mask_18 : <ui16>
                        %230 = ascvec.cmp_eq_scalar %225, %c0_i16, %mask_18 : <ui16>, i16 -> !ascvec.mask
                        %231 = ascvec.select %223, %229, %230 : <ui16>
                        %232 = ascvec.duplicate_scalar %c127_i16, %mask_18 : i16 -> <ui16>
                        %233 = ascvec.and %231, %232, %mask_18 : <ui16>, <ui16> -> <ui16>
                        %234 = ascvec.gather_reg %208, %233 : <ui16>, <ui16> -> <ui16>
                        %235 = ascvec.gather_reg %213, %233 : <ui16>, <ui16> -> <ui16>
                        %236 = ascvec.cmp_lt_scalar %231, %c128_i16, %mask_18 : <ui16>, i16 -> !ascvec.mask
                        %237 = ascvec.select %234, %235, %236 : <ui16>
                        %238 = ascvec.select %223, %237, %230 : <ui16>
                        %239 = ascvec.duplicate_scalar %arg56, %mask_18 : i16 -> <ui16>
                        %240 = ascvec.sub %239, %238, %mask_18 : <ui16>
                        %241 = ascvec.duplicate %240, %mask_18 : <ui16> -> <ui16>
                        %mask_22, %remaining_out_23 = ascvec.update_mask %c256_i32_11 elem_bits 8 : !ascvec.mask
                        %242:8 = scf.for %arg64 = %c0_i64_7 to %arg58 step %c1_i64_6 iter_args(%arg65 = %223, %arg66 = %223, %arg67 = %223, %arg68 = %223, %arg69 = %223, %arg70 = %223, %arg71 = %223, %arg72 = %223) -> (!ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>)  : i64 {
                          %260 = arith.muli %arg64, %c1024_i64_10 : i64
                          %261 = arith.index_cast %260 : i64 to index
                          %result0, %result1 = ascvec.loadalign_deintlv %arg57[%261] {dintlv_width = "b8"} : memref<?xui16, 1>, index -> <ui8>, <ui8>
                          %262 = ascvec.cmp_eq %result1, %228, %mask_22 : <ui8>, <ui8> -> !ascvec.mask
                          %263 = ascvec.cumulative_histogram %arg65, %result0, %262 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          %264 = ascvec.cumulative_histogram %arg66, %result0, %262 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          %265 = arith.addi %260, %c256_i64_9 : i64
                          %266 = arith.index_cast %265 : i64 to index
                          %result0_32, %result1_33 = ascvec.loadalign_deintlv %arg57[%266] {dintlv_width = "b8"} : memref<?xui16, 1>, index -> <ui8>, <ui8>
                          %267 = ascvec.cmp_eq %result1_33, %228, %mask_22 : <ui8>, <ui8> -> !ascvec.mask
                          %268 = ascvec.cumulative_histogram %arg67, %result0_32, %267 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          %269 = ascvec.cumulative_histogram %arg68, %result0_32, %267 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          %270 = arith.addi %260, %c512_i64_8 : i64
                          %271 = arith.index_cast %270 : i64 to index
                          %result0_34, %result1_35 = ascvec.loadalign_deintlv %arg57[%271] {dintlv_width = "b8"} : memref<?xui16, 1>, index -> <ui8>, <ui8>
                          %272 = ascvec.cmp_eq %result1_35, %228, %mask_22 : <ui8>, <ui8> -> !ascvec.mask
                          %273 = ascvec.cumulative_histogram %arg69, %result0_34, %272 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          %274 = ascvec.cumulative_histogram %arg70, %result0_34, %272 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          %275 = arith.addi %260, %c768_i64 : i64
                          %276 = arith.index_cast %275 : i64 to index
                          %result0_36, %result1_37 = ascvec.loadalign_deintlv %arg57[%276] {dintlv_width = "b8"} : memref<?xui16, 1>, index -> <ui8>, <ui8>
                          %277 = ascvec.cmp_eq %result1_37, %228, %mask_22 : <ui8>, <ui8> -> !ascvec.mask
                          %278 = ascvec.cumulative_histogram %arg71, %result0_36, %277 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          %279 = ascvec.cumulative_histogram %arg72, %result0_36, %277 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          scf.yield %263, %264, %268, %269, %273, %274, %278, %279 : !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>
                        } {cannir.loop_unroll = 1 : i64}
                        %243:2 = scf.for %arg64 = %arg60 to %arg61 step %c1_i64_6 iter_args(%arg65 = %242#0, %arg66 = %242#1) -> (!ascvec.vreg<ui16>, !ascvec.vreg<ui16>)  : i64 {
                          %260 = arith.muli %arg64, %c256_i64_9 : i64
                          %261 = arith.subi %arg59, %260 : i64
                          %262 = arith.trunci %261 : i64 to i32
                          %mask_32, %remaining_out_33 = ascvec.update_mask %262 elem_bits 8 : !ascvec.mask
                          %263 = arith.index_cast %260 : i64 to index
                          %result0, %result1 = ascvec.loadalign_deintlv %arg57[%263] {dintlv_width = "b8"} : memref<?xui16, 1>, index -> <ui8>, <ui8>
                          %264 = ascvec.cmp_eq %result1, %228, %mask_32 : <ui8>, <ui8> -> !ascvec.mask
                          %265 = ascvec.cumulative_histogram %arg65, %result0, %264 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          %266 = ascvec.cumulative_histogram %arg66, %result0, %264 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          scf.yield %265, %266 : !ascvec.vreg<ui16>, !ascvec.vreg<ui16>
                        } {cannir.loop_unroll = 1 : i64}
                        %244 = ascvec.add %243#0, %242#2, %mask_18 : <ui16>
                        %245 = ascvec.add %244, %242#4, %mask_18 : <ui16>
                        %246 = ascvec.add %245, %242#6, %mask_18 : <ui16>
                        %247 = ascvec.add %243#1, %242#3, %mask_18 : <ui16>
                        %248 = ascvec.add %247, %242#5, %mask_18 : <ui16>
                        %249 = ascvec.add %248, %242#7, %mask_18 : <ui16>
                        %mask_24, %remaining_out_25 = ascvec.update_mask %c128_i32_13 elem_bits 16 : !ascvec.mask
                        %mask_26, %remaining_out_27 = ascvec.update_mask %c1_i32_12 elem_bits 16 : !ascvec.mask
                        %250 = ascvec.duplicate_scalar %c-1_i16, %mask_24 : i16 -> <ui16>
                        %251 = ascvec.cmp_ge %246, %241, %mask_24 : <ui16>, <ui16> -> !ascvec.mask
                        %252 = ascvec.select %215, %250, %251 : <ui16>
                        %253 = ascvec.cmp_ge %249, %241, %mask_24 : <ui16>, <ui16> -> !ascvec.mask
                        %254 = ascvec.select %218, %250, %253 : <ui16>
                        %255 = ascvec.min %252, %254, %mask_24 : <ui16>
                        %256 = ascvec.reduce_min %255, %mask_24 : <ui16>
                        %mask_28, %remaining_out_29 = ascvec.update_mask %c128_i32_13 elem_bits 16 : !ascvec.mask
                        %257 = ascvec.duplicate %256, %mask_28 : <ui16> -> <ui16>
                        %258 = ascvec.shl %225, 8, %mask_28 : <ui16> -> <ui16>
                        %259 = ascvec.or %258, %257, %mask_28 : <ui16>, <ui16> -> <ui16>
                        ascvec.store_unalign_begin %arg62 : memref<?xui16, 1>
                        scf.for %arg64 = %c0_i64_7 to %arg63 step %c1_i64_6  : i64 {
                          %260 = arith.muli %arg64, %c128_i64_5 : i64
                          %261 = arith.subi %arg59, %260 : i64
                          %262 = arith.trunci %261 : i64 to i32
                          %mask_32, %remaining_out_33 = ascvec.update_mask %262 elem_bits 16 : !ascvec.mask
                          %263 = arith.index_cast %260 : i64 to index
                          %264 = ascvec.loadalign %arg57[%263] : memref<?xui16, 1>, index -> <ui16>
                          %265 = arith.trunci %260 : i64 to i16
                          %266 = ascvec.arange %265 : i16 -> <ui16>
                          %267 = ascvec.cmp_gt %264, %259, %mask_32 : <ui16>, <ui16> -> !ascvec.mask
                          %268 = ascvec.squeeze_store_reg %266, %267 : <ui16>, !ascvec.mask -> <ui16>
                          ascvec.store_unalign %arg62[%c24576], %268 : memref<?xui16, 1>, <ui16>
                        } {cannir.loop_unroll = 1 : i64}
                        scf.for %arg64 = %c0_i64_7 to %arg63 step %c1_i64_6  : i64 {
                          %260 = arith.muli %arg64, %c128_i64_5 : i64
                          %261 = arith.subi %arg59, %260 : i64
                          %262 = arith.trunci %261 : i64 to i32
                          %mask_32, %remaining_out_33 = ascvec.update_mask %262 elem_bits 16 : !ascvec.mask
                          %263 = arith.index_cast %260 : i64 to index
                          %264 = ascvec.loadalign %arg57[%263] : memref<?xui16, 1>, index -> <ui16>
                          %265 = arith.trunci %260 : i64 to i16
                          %266 = ascvec.arange %265 : i16 -> <ui16>
                          %267 = ascvec.cmp_eq %264, %259, %mask_32 : <ui16>, <ui16> -> !ascvec.mask
                          %268 = ascvec.squeeze_store_reg %266, %267 : <ui16>, !ascvec.mask -> <ui16>
                          ascvec.store_unalign %arg62[%c24576], %268 : memref<?xui16, 1>, <ui16>
                        } {cannir.loop_unroll = 1 : i64}
                        %c0_30 = arith.constant 0 : index
                        %c1_31 = arith.constant 1 : index
                        scf.for %arg64 = %c0_30 to %c1_31 step %c1_31 {
                          ascvec.store_unalign_post %arg62[%c24576] : memref<?xui16, 1>
                        }
                        ascvec.mem_bar vst_vld
                      } {cannir.vf_mode = "simd"}
                    } else {
                      %194 = arith.divsi %163, %c1024_i64 : i64
                      %195 = arith.muli %194, %c4_i64 : i64
                      %196 = arith.addi %163, %c255_i64 : i64
                      %197 = arith.divsi %196, %c256_i64 : i64
                      %198 = arith.addi %163, %c-8191_i64 : i64
                      %199 = arith.trunci %198 : i64 to i16
                      ascvec.vf(%134, %194, %163, %195, %197, %199, %134, %173 : memref<?xui16, 1>, i64, i64, i64, i64, i16, memref<?xui16, 1>, i64) {
                      ^bb0(%arg52: memref<?xui16, 1>, %arg53: i64, %arg54: i64, %arg55: i64, %arg56: i64, %arg57: i16, %arg58: memref<?xui16, 1>, %arg59: i64):
                        %c128_i64_5 = arith.constant 128 : i64
                        %c24576 = arith.constant 24576 : index
                        %c127_i16 = arith.constant 127 : i16
                        %c1_i16 = arith.constant 1 : i16
                        %c128_i16 = arith.constant 128 : i16
                        %c-1_i16 = arith.constant -1 : i16
                        %c1_i32_6 = arith.constant 1 : i32
                        %c1_i64_7 = arith.constant 1 : i64
                        %c0_i64_8 = arith.constant 0 : i64
                        %c768_i64 = arith.constant 768 : i64
                        %c512_i64_9 = arith.constant 512 : i64
                        %c256_i64_10 = arith.constant 256 : i64
                        %c1024_i64_11 = arith.constant 1024 : i64
                        %c256_i32_12 = arith.constant 256 : i32
                        %c0_i16 = arith.constant 0 : i16
                        %c128_i32_13 = arith.constant 128 : i32
                        %mask, %remaining_out = ascvec.update_mask %c128_i32_13 elem_bits 16 : !ascvec.mask
                        %200 = ascvec.duplicate_scalar %c0_i16, %mask : i16 -> <ui16>
                        %mask_14, %remaining_out_15 = ascvec.update_mask %c256_i32_12 elem_bits 8 : !ascvec.mask
                        %201:8 = scf.for %arg60 = %c0_i64_8 to %arg53 step %c1_i64_7 iter_args(%arg61 = %200, %arg62 = %200, %arg63 = %200, %arg64 = %200, %arg65 = %200, %arg66 = %200, %arg67 = %200, %arg68 = %200) -> (!ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>)  : i64 {
                          %255 = arith.muli %arg60, %c1024_i64_11 : i64
                          %256 = arith.index_cast %255 : i64 to index
                          %result0, %result1 = ascvec.loadalign_deintlv %arg52[%256] {dintlv_width = "b8"} : memref<?xui16, 1>, index -> <ui8>, <ui8>
                          %257 = ascvec.cumulative_histogram %arg61, %result1, %mask_14 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          %258 = ascvec.cumulative_histogram %arg62, %result1, %mask_14 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          %259 = arith.addi %255, %c256_i64_10 : i64
                          %260 = arith.index_cast %259 : i64 to index
                          %result0_34, %result1_35 = ascvec.loadalign_deintlv %arg52[%260] {dintlv_width = "b8"} : memref<?xui16, 1>, index -> <ui8>, <ui8>
                          %261 = ascvec.cumulative_histogram %arg63, %result1_35, %mask_14 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          %262 = ascvec.cumulative_histogram %arg64, %result1_35, %mask_14 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          %263 = arith.addi %255, %c512_i64_9 : i64
                          %264 = arith.index_cast %263 : i64 to index
                          %result0_36, %result1_37 = ascvec.loadalign_deintlv %arg52[%264] {dintlv_width = "b8"} : memref<?xui16, 1>, index -> <ui8>, <ui8>
                          %265 = ascvec.cumulative_histogram %arg65, %result1_37, %mask_14 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          %266 = ascvec.cumulative_histogram %arg66, %result1_37, %mask_14 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          %267 = arith.addi %255, %c768_i64 : i64
                          %268 = arith.index_cast %267 : i64 to index
                          %result0_38, %result1_39 = ascvec.loadalign_deintlv %arg52[%268] {dintlv_width = "b8"} : memref<?xui16, 1>, index -> <ui8>, <ui8>
                          %269 = ascvec.cumulative_histogram %arg67, %result1_39, %mask_14 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          %270 = ascvec.cumulative_histogram %arg68, %result1_39, %mask_14 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          scf.yield %257, %258, %261, %262, %265, %266, %269, %270 : !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>
                        } {cannir.loop_unroll = 1 : i64}
                        %202:2 = scf.for %arg60 = %arg55 to %arg56 step %c1_i64_7 iter_args(%arg61 = %201#0, %arg62 = %201#1) -> (!ascvec.vreg<ui16>, !ascvec.vreg<ui16>)  : i64 {
                          %255 = arith.muli %arg60, %c256_i64_10 : i64
                          %256 = arith.subi %arg54, %255 : i64
                          %257 = arith.trunci %256 : i64 to i32
                          %mask_34, %remaining_out_35 = ascvec.update_mask %257 elem_bits 8 : !ascvec.mask
                          %258 = arith.index_cast %255 : i64 to index
                          %result0, %result1 = ascvec.loadalign_deintlv %arg52[%258] {dintlv_width = "b8"} : memref<?xui16, 1>, index -> <ui8>, <ui8>
                          %259 = ascvec.cumulative_histogram %arg61, %result1, %mask_34 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          %260 = ascvec.cumulative_histogram %arg62, %result1, %mask_34 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          scf.yield %259, %260 : !ascvec.vreg<ui16>, !ascvec.vreg<ui16>
                        } {cannir.loop_unroll = 1 : i64}
                        %203 = ascvec.add %202#0, %201#2, %mask : <ui16>
                        %204 = ascvec.add %203, %201#4, %mask : <ui16>
                        %205 = ascvec.add %204, %201#6, %mask : <ui16>
                        %206 = ascvec.add %202#1, %201#3, %mask : <ui16>
                        %207 = ascvec.add %206, %201#5, %mask : <ui16>
                        %208 = ascvec.add %207, %201#7, %mask : <ui16>
                        %mask_16, %remaining_out_17 = ascvec.update_mask %c128_i32_13 elem_bits 16 : !ascvec.mask
                        %mask_18, %remaining_out_19 = ascvec.update_mask %c1_i32_6 elem_bits 16 : !ascvec.mask
                        %209 = ascvec.duplicate_scalar %c-1_i16, %mask_16 : i16 -> <ui16>
                        %210 = ascvec.arange %c0_i16 : i16 -> <ui16>
                        %211 = ascvec.cmp_ge_scalar %205, %arg57, %mask_16 : <ui16>, i16 -> !ascvec.mask
                        %212 = ascvec.select %210, %209, %211 : <ui16>
                        %213 = ascvec.arange %c128_i16 : i16 -> <ui16>
                        %214 = ascvec.cmp_ge_scalar %208, %arg57, %mask_16 : <ui16>, i16 -> !ascvec.mask
                        %215 = ascvec.select %213, %209, %214 : <ui16>
                        %216 = ascvec.min %212, %215, %mask_16 : <ui16>
                        %217 = ascvec.reduce_min %216, %mask_16 : <ui16>
                        %mask_20, %remaining_out_21 = ascvec.update_mask %c128_i32_13 elem_bits 16 : !ascvec.mask
                        %mask_22, %remaining_out_23 = ascvec.update_mask %c1_i32_6 elem_bits 16 : !ascvec.mask
                        %218 = ascvec.duplicate_scalar %c0_i16, %mask_20 : i16 -> <ui16>
                        %219 = ascvec.duplicate_scalar %c1_i16, %mask_20 : i16 -> <ui16>
                        %220 = ascvec.duplicate %217, %mask_20 : <ui16> -> <ui16>
                        %221 = ascvec.shl %220, 8, %mask_20 : <ui16> -> <ui16>
                        %222 = ascvec.or %220, %221, %mask_20 : <ui16>, <ui16> -> <ui16>
                        %223 = ascvec.reinterpret %222 : <ui16> -> <ui8>
                        %224 = ascvec.sub %220, %219, %mask_20 : <ui16>
                        %225 = ascvec.cmp_eq_scalar %220, %c0_i16, %mask_20 : <ui16>, i16 -> !ascvec.mask
                        %226 = ascvec.select %218, %224, %225 : <ui16>
                        %227 = ascvec.duplicate_scalar %c127_i16, %mask_20 : i16 -> <ui16>
                        %228 = ascvec.and %226, %227, %mask_20 : <ui16>, <ui16> -> <ui16>
                        %229 = ascvec.gather_reg %205, %228 : <ui16>, <ui16> -> <ui16>
                        %230 = ascvec.gather_reg %208, %228 : <ui16>, <ui16> -> <ui16>
                        %231 = ascvec.cmp_lt_scalar %226, %c128_i16, %mask_20 : <ui16>, i16 -> !ascvec.mask
                        %232 = ascvec.select %229, %230, %231 : <ui16>
                        %233 = ascvec.select %218, %232, %225 : <ui16>
                        %234 = ascvec.duplicate_scalar %arg57, %mask_20 : i16 -> <ui16>
                        %235 = ascvec.sub %234, %233, %mask_20 : <ui16>
                        %236 = ascvec.duplicate %235, %mask_20 : <ui16> -> <ui16>
                        %mask_24, %remaining_out_25 = ascvec.update_mask %c256_i32_12 elem_bits 8 : !ascvec.mask
                        %237:8 = scf.for %arg60 = %c0_i64_8 to %arg53 step %c1_i64_7 iter_args(%arg61 = %218, %arg62 = %218, %arg63 = %218, %arg64 = %218, %arg65 = %218, %arg66 = %218, %arg67 = %218, %arg68 = %218) -> (!ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>)  : i64 {
                          %255 = arith.muli %arg60, %c1024_i64_11 : i64
                          %256 = arith.index_cast %255 : i64 to index
                          %result0, %result1 = ascvec.loadalign_deintlv %arg52[%256] {dintlv_width = "b8"} : memref<?xui16, 1>, index -> <ui8>, <ui8>
                          %257 = ascvec.cmp_eq %result1, %223, %mask_24 : <ui8>, <ui8> -> !ascvec.mask
                          %258 = ascvec.cumulative_histogram %arg61, %result0, %257 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          %259 = ascvec.cumulative_histogram %arg62, %result0, %257 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          %260 = arith.addi %255, %c256_i64_10 : i64
                          %261 = arith.index_cast %260 : i64 to index
                          %result0_34, %result1_35 = ascvec.loadalign_deintlv %arg52[%261] {dintlv_width = "b8"} : memref<?xui16, 1>, index -> <ui8>, <ui8>
                          %262 = ascvec.cmp_eq %result1_35, %223, %mask_24 : <ui8>, <ui8> -> !ascvec.mask
                          %263 = ascvec.cumulative_histogram %arg63, %result0_34, %262 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          %264 = ascvec.cumulative_histogram %arg64, %result0_34, %262 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          %265 = arith.addi %255, %c512_i64_9 : i64
                          %266 = arith.index_cast %265 : i64 to index
                          %result0_36, %result1_37 = ascvec.loadalign_deintlv %arg52[%266] {dintlv_width = "b8"} : memref<?xui16, 1>, index -> <ui8>, <ui8>
                          %267 = ascvec.cmp_eq %result1_37, %223, %mask_24 : <ui8>, <ui8> -> !ascvec.mask
                          %268 = ascvec.cumulative_histogram %arg65, %result0_36, %267 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          %269 = ascvec.cumulative_histogram %arg66, %result0_36, %267 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          %270 = arith.addi %255, %c768_i64 : i64
                          %271 = arith.index_cast %270 : i64 to index
                          %result0_38, %result1_39 = ascvec.loadalign_deintlv %arg52[%271] {dintlv_width = "b8"} : memref<?xui16, 1>, index -> <ui8>, <ui8>
                          %272 = ascvec.cmp_eq %result1_39, %223, %mask_24 : <ui8>, <ui8> -> !ascvec.mask
                          %273 = ascvec.cumulative_histogram %arg67, %result0_38, %272 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          %274 = ascvec.cumulative_histogram %arg68, %result0_38, %272 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          scf.yield %258, %259, %263, %264, %268, %269, %273, %274 : !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>
                        } {cannir.loop_unroll = 1 : i64}
                        %238:2 = scf.for %arg60 = %arg55 to %arg56 step %c1_i64_7 iter_args(%arg61 = %237#0, %arg62 = %237#1) -> (!ascvec.vreg<ui16>, !ascvec.vreg<ui16>)  : i64 {
                          %255 = arith.muli %arg60, %c256_i64_10 : i64
                          %256 = arith.subi %arg54, %255 : i64
                          %257 = arith.trunci %256 : i64 to i32
                          %mask_34, %remaining_out_35 = ascvec.update_mask %257 elem_bits 8 : !ascvec.mask
                          %258 = arith.index_cast %255 : i64 to index
                          %result0, %result1 = ascvec.loadalign_deintlv %arg52[%258] {dintlv_width = "b8"} : memref<?xui16, 1>, index -> <ui8>, <ui8>
                          %259 = ascvec.cmp_eq %result1, %223, %mask_34 : <ui8>, <ui8> -> !ascvec.mask
                          %260 = ascvec.cumulative_histogram %arg61, %result0, %259 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          %261 = ascvec.cumulative_histogram %arg62, %result0, %259 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          scf.yield %260, %261 : !ascvec.vreg<ui16>, !ascvec.vreg<ui16>
                        } {cannir.loop_unroll = 1 : i64}
                        %239 = ascvec.add %238#0, %237#2, %mask_20 : <ui16>
                        %240 = ascvec.add %239, %237#4, %mask_20 : <ui16>
                        %241 = ascvec.add %240, %237#6, %mask_20 : <ui16>
                        %242 = ascvec.add %238#1, %237#3, %mask_20 : <ui16>
                        %243 = ascvec.add %242, %237#5, %mask_20 : <ui16>
                        %244 = ascvec.add %243, %237#7, %mask_20 : <ui16>
                        %mask_26, %remaining_out_27 = ascvec.update_mask %c128_i32_13 elem_bits 16 : !ascvec.mask
                        %mask_28, %remaining_out_29 = ascvec.update_mask %c1_i32_6 elem_bits 16 : !ascvec.mask
                        %245 = ascvec.duplicate_scalar %c-1_i16, %mask_26 : i16 -> <ui16>
                        %246 = ascvec.cmp_ge %241, %236, %mask_26 : <ui16>, <ui16> -> !ascvec.mask
                        %247 = ascvec.select %210, %245, %246 : <ui16>
                        %248 = ascvec.cmp_ge %244, %236, %mask_26 : <ui16>, <ui16> -> !ascvec.mask
                        %249 = ascvec.select %213, %245, %248 : <ui16>
                        %250 = ascvec.min %247, %249, %mask_26 : <ui16>
                        %251 = ascvec.reduce_min %250, %mask_26 : <ui16>
                        %mask_30, %remaining_out_31 = ascvec.update_mask %c128_i32_13 elem_bits 16 : !ascvec.mask
                        %252 = ascvec.duplicate %251, %mask_30 : <ui16> -> <ui16>
                        %253 = ascvec.shl %220, 8, %mask_30 : <ui16> -> <ui16>
                        %254 = ascvec.or %253, %252, %mask_30 : <ui16>, <ui16> -> <ui16>
                        ascvec.store_unalign_begin %arg58 : memref<?xui16, 1>
                        scf.for %arg60 = %c0_i64_8 to %arg59 step %c1_i64_7  : i64 {
                          %255 = arith.muli %arg60, %c128_i64_5 : i64
                          %256 = arith.subi %arg54, %255 : i64
                          %257 = arith.trunci %256 : i64 to i32
                          %mask_34, %remaining_out_35 = ascvec.update_mask %257 elem_bits 16 : !ascvec.mask
                          %258 = arith.index_cast %255 : i64 to index
                          %259 = ascvec.loadalign %arg52[%258] : memref<?xui16, 1>, index -> <ui16>
                          %260 = arith.trunci %255 : i64 to i16
                          %261 = ascvec.arange %260 : i16 -> <ui16>
                          %262 = ascvec.cmp_gt %259, %254, %mask_34 : <ui16>, <ui16> -> !ascvec.mask
                          %263 = ascvec.squeeze_store_reg %261, %262 : <ui16>, !ascvec.mask -> <ui16>
                          ascvec.store_unalign %arg58[%c24576], %263 : memref<?xui16, 1>, <ui16>
                        } {cannir.loop_unroll = 1 : i64}
                        scf.for %arg60 = %c0_i64_8 to %arg59 step %c1_i64_7  : i64 {
                          %255 = arith.muli %arg60, %c128_i64_5 : i64
                          %256 = arith.subi %arg54, %255 : i64
                          %257 = arith.trunci %256 : i64 to i32
                          %mask_34, %remaining_out_35 = ascvec.update_mask %257 elem_bits 16 : !ascvec.mask
                          %258 = arith.index_cast %255 : i64 to index
                          %259 = ascvec.loadalign %arg52[%258] : memref<?xui16, 1>, index -> <ui16>
                          %260 = arith.trunci %255 : i64 to i16
                          %261 = ascvec.arange %260 : i16 -> <ui16>
                          %262 = ascvec.cmp_eq %259, %254, %mask_34 : <ui16>, <ui16> -> !ascvec.mask
                          %263 = ascvec.squeeze_store_reg %261, %262 : <ui16>, !ascvec.mask -> <ui16>
                          ascvec.store_unalign %arg58[%c24576], %263 : memref<?xui16, 1>, <ui16>
                        } {cannir.loop_unroll = 1 : i64}
                        %c0_32 = arith.constant 0 : index
                        %c1_33 = arith.constant 1 : index
                        scf.for %arg60 = %c0_32 to %c1_33 step %c1_33 {
                          ascvec.store_unalign_post %arg58[%c24576] : memref<?xui16, 1>
                        }
                        ascvec.mem_bar vst_vld
                      } {cannir.vf_mode = "simd"}
                    }
                    %180 = arith.addi %130, %c218112_i64 : i64
                    %181 = ascscalar.bind_local_ptr %180 : memref<8192xi32, 1>
                    %182 = arith.addi %130, %c98304_i64 : i64
                    %183 = ascscalar.bind_local_ptr %182 : memref<8192xui16, 1>
                    ascvec.get_buf v, %c1_i64, 0 {cannir.buffer.generated_sync}
                    ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                    ascvec.vf(%134, %134, %157, %164, %181, %183 : memref<?xui16, 1>, memref<?xui16, 1>, i32, i32, memref<8192xi32, 1>, memref<8192xui16, 1>) {
                    ^bb0(%arg52: memref<?xui16, 1>, %arg53: memref<?xui16, 1>, %arg54: i32, %arg55: i32, %arg56: memref<8192xi32, 1>, %arg57: memref<8192xui16, 1>):
                      %c1_i32_5 = arith.constant 1 : i32
                      %c128_i32_6 = arith.constant 128 : i32
                      %c0_i32_7 = arith.constant 0 : i32
                      %c-128_i16 = arith.constant -128 : i16
                      %c-1_i32_8 = arith.constant -1 : i32
                      %c-1_i16 = arith.constant -1 : i16
                      %c-32768_i16 = arith.constant -32768 : i16
                      %c0_9 = arith.constant 0 : index
                      %c24576 = arith.constant 24576 : index
                      %c8192_i32_10 = arith.constant 8192 : i32
                      %c64_i32 = arith.constant 64 : i32
                      scf.for %arg58 = %c0_i32_7 to %c128_i32_6 step %c1_i32_5  : i32 {
                        %194 = arith.muli %arg58, %c64_i32 overflow<nsw> : i32
                        %195 = arith.subi %c8192_i32_10, %194 overflow<nsw> : i32
                        %mask, %remaining_out = ascvec.update_mask %195 : !ascvec.mask
                        %mask_11, %remaining_out_12 = ascvec.update_mask %195 elem_bits 16 : !ascvec.mask
                        %196 = arith.index_cast %194 : i32 to index
                        %197 = arith.addi %196, %c24576 : index
                        %198 = ascvec.loadalign %arg52[%197] : memref<?xui16, 1>, index -> <ui16>
                        %199 = ascvec.unpack %198 {part = "lower"} : <ui16> -> <ui32>
                        %200 = ascvec.gather %arg53[%c0_9], %198, %mask_11 : memref<?xui16, 1>, <ui16>, !ascvec.mask -> <ui16>
                        %201 = ascvec.duplicate_scalar %c-32768_i16, %mask_11 : i16 -> <ui16>
                        %202 = ascvec.and %200, %201, %mask_11 : <ui16>, <ui16> -> <ui16>
                        %203 = ascvec.cmp_eq %202, %201, %mask_11 : <ui16>, <ui16> -> !ascvec.mask
                        %204 = ascvec.xor %200, %201, %mask_11 : <ui16>, <ui16> -> <ui16>
                        %205 = ascvec.duplicate_scalar %c-1_i16, %mask_11 : i16 -> <ui16>
                        %206 = ascvec.xor %200, %205, %mask_11 : <ui16>, <ui16> -> <ui16>
                        %207 = ascvec.select %204, %206, %203 : <ui16>
                        %208 = ascvec.arange %194 : i32 -> <ui32>
                        %209 = ascvec.cmp_lt_scalar %208, %arg54, %mask : <ui32>, i32 -> !ascvec.mask
                        %210 = arith.trunci %194 : i32 to i16
                        %211 = ascvec.arange %210 : i16 -> <ui16>
                        %212 = arith.trunci %arg54 : i32 to i16
                        %213 = ascvec.cmp_lt_scalar %211, %212, %mask_11 : <ui16>, i16 -> !ascvec.mask
                        %214 = ascvec.reinterpret %199 : <ui32> -> <i32>
                        %215 = ascvec.add_scalar %214, %arg55, %mask : <i32>, i32
                        %216 = ascvec.duplicate_scalar %c-1_i32_8, %mask : i32 -> <i32>
                        %217 = ascvec.select %215, %216, %209 : <i32>
                        %218 = ascvec.duplicate_scalar %c-128_i16, %mask_11 : i16 -> <ui16>
                        %219 = ascvec.select %207, %218, %213 : <ui16>
                        ascvec.storealign %arg56[%196], %217, %mask : memref<8192xi32, 1>, index, <i32>
                        ascvec.storealign %arg57[%196], %219, %mask_11 : memref<8192xui16, 1>, index, <ui16>
                      } {cannir.loop_unroll = 1 : i64}
                      ascvec.mem_bar vst_vld
                    } {cannir.vf_mode = "simd"}
                    ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                    ascvec.release_buf v, %c1_i64, 0 {cannir.buffer.generated_sync}
                    ascvec.release_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
                    %184 = arith.muli %147, %c262144_i64 : i64
                    %185 = arith.divui %184, %c8_i64 : i64
                    %186 = arith.addi %82, %185 : i64
                    %187 = ascscalar.bind_gm_ptr %186 : memref<1xi32>
                    %188 = ascscalar.bind_local_ptr %180 : memref<1xi32, 1>
                    ascvec.get_buf mte3, %c1_i64, 0 {cannir.buffer.generated_sync}
                    ascvec.copy_ub2gm %187[%c0], %188[%c0], %c1_i32, %c32768_i32, %c0_i64, %c32768_i32 l2_cache_ctl %c0_i32 : memref<1xi32>, memref<1xi32, 1>
                    ascvec.release_buf mte3, %c1_i64, 0 {cannir.buffer.generated_sync}
                    %189 = arith.muli %147, %c131072_i64 : i64
                    %190 = arith.divui %189, %c8_i64 : i64
                    %191 = arith.addi %83, %190 : i64
                    %192 = ascscalar.bind_gm_ptr %191 : memref<1xui16>
                    %193 = ascscalar.bind_local_ptr %182 : memref<1xui16, 1>
                    ascvec.get_buf mte3, %c2_i64, 0 {cannir.buffer.generated_sync}
                    ascvec.copy_ub2gm %192[%c0], %193[%c0], %c1_i32, %c16384_i32, %c0_i64, %c16384_i32 l2_cache_ctl %c0_i32 : memref<1xui16>, memref<1xui16, 1>
                    ascvec.release_buf mte3, %c2_i64, 0 {cannir.buffer.generated_sync}
                  } else {
                    %172 = arith.cmpi sgt, %157, %c8192_i32 : i32
                    %173 = arith.select %172, %c8192_i32, %157 : i32
                    %174 = arith.addi %130, %c218112_i64 : i64
                    %175 = ascscalar.bind_local_ptr %174 : memref<8192xi32, 1>
                    %176 = arith.addi %130, %c98304_i64 : i64
                    %177 = ascscalar.bind_local_ptr %176 : memref<8192xui16, 1>
                    %178 = ascscalar.bitcast_memref %alloc_4 byte_offset = %c0_i64 : memref<262144xui8, 1> -> memref<?xi32, 1>
                    ascvec.vf(%134, %178 : memref<?xui16, 1>, memref<?xi32, 1>) {
                    ^bb0(%arg52: memref<?xui16, 1>, %arg53: memref<?xi32, 1>):
                      %c256_i32_5 = arith.constant 256 : i32
                      %c38144_6 = arith.constant 38144 : index
                      %c1_i32_7 = arith.constant 1 : i32
                      %c59904_8 = arith.constant 59904 : index
                      %c0_i32_9 = arith.constant 0 : i32
                      %c0_i16 = arith.constant 0 : i16
                      %c64_i32 = arith.constant 64 : i32
                      %c128_i32_10 = arith.constant 128 : i32
                      %mask, %remaining_out = ascvec.update_mask %c128_i32_10 elem_bits 16 : !ascvec.mask
                      %mask_11, %remaining_out_12 = ascvec.update_mask %c64_i32 : !ascvec.mask
                      %211 = ascvec.duplicate_scalar %c0_i16, %mask : i16 -> <ui16>
                      %212 = ascvec.duplicate_scalar %c0_i32_9, %mask_11 : i32 -> <i32>
                      scf.for %arg54 = %c0_i32_9 to %c128_i32_10 step %c1_i32_7  : i32 {
                        %213 = arith.muli %arg54, %c128_i32_10 overflow<nsw> : i32
                        %214 = arith.index_cast %213 : i32 to index
                        %215 = arith.addi %214, %c59904_8 : index
                        ascvec.storealign %arg52[%215], %211, %mask : memref<?xui16, 1>, index, <ui16>
                      } {cannir.loop_unroll = 1 : i64}
                      scf.for %arg54 = %c0_i32_9 to %c256_i32_5 step %c1_i32_7  : i32 {
                        %213 = arith.muli %arg54, %c64_i32 overflow<nsw> : i32
                        %214 = arith.index_cast %213 : i32 to index
                        %215 = arith.addi %214, %c38144_6 : index
                        ascvec.storealign %arg53[%215], %212, %mask_11 : memref<?xi32, 1>, index, <i32>
                      } {cannir.loop_unroll = 1 : i64}
                      ascvec.mem_bar vst_vld
                    } {cannir.vf_mode = "simd"}
                    %179 = arith.remsi %163, %c16384_i64 : i64
                    %180 = arith.cmpi ne, %179, %c0_i64 : i64
                    %181 = arith.cmpi slt, %179, %c0_i64 : i64
                    %182 = arith.andi %180, %181 : i1
                    %183 = arith.divsi %163, %c16384_i64 : i64
                    %184 = arith.extui %182 : i1 to i64
                    %185 = arith.subi %183, %184 : i64
                    %186 = arith.select %182, %c16384_i64, %c0_i64 : i64
                    %187 = arith.addi %179, %186 overflow<nsw> : i64
                    scf.for %arg52 = %c0_i64 to %185 step %c1_i64  : i64 {
                      %211 = arith.remsi %arg52, %c2_i64 : i64
                      %212 = arith.cmpi ne, %211, %c0_i64 : i64
                      %213 = arith.cmpi slt, %211, %c0_i64 : i64
                      %214 = arith.andi %212, %213 : i1
                      %215 = arith.select %214, %c2_i64, %c0_i64 : i64
                      %216 = arith.addi %211, %215 overflow<nsw> : i64
                      %217 = arith.subi %c1_i64, %216 overflow<nsw> : i64
                      %218 = arith.muli %arg52, %c16384_i64 : i64
                      %219 = arith.cmpi eq, %arg52, %c0_i64 : i64
                      %220 = arith.select %219, %c0_i64, %c8192_i64 : i64
                      %221 = cannir.extract_physical_layout %arg12 : !cannir.tensor<<gm, ui16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
                      %222 = cannir.extract_stride %221 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
                      %223 = cannir.extract_dim %222[0] : !cannir.stride<(-1, 1)>
                      %224 = arith.muli %149, %223 : i64
                      %225 = arith.muli %224, %c2_i64 : i64
                      %226 = arith.muli %arg52, %c32768_i64 : i64
                      %227 = arith.addi %225, %226 : i64
                      %228 = arith.muli %220, %c2_i64 overflow<nsw> : i64
                      ascvec.get_buf mte2, %c0_i64, 0 {cannir.buffer.generated_sync}
                      "ascvec.copy_gm2ub_plan"(%134, %alloc_2, %c0_i32, %c0_i32, %228, %227, %c1_i64, %c32768_i64, %c0_i64, %c32768_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = ui16, ub_base = 0 : i64, ub_capacity = 262144 : i64}> : (memref<?xui16, 1>, memref<1xui16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                      ascvec.release_buf mte2, %c0_i64, 0 {cannir.buffer.generated_sync}
                      %229 = arith.divsi %220, %c128_i64 : i64
                      ascvec.get_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
                      %230 = arith.muli %216, %c8192_i64 overflow<nsw> : i64
                      %231 = arith.index_cast %230 : i64 to index
                      %232 = arith.addi %231, %c59904 : index
                      ascvec.vf(%134, %232, %134, %229 : memref<?xui16, 1>, index, memref<?xui16, 1>, i64) {
                      ^bb0(%arg53: memref<?xui16, 1>, %arg54: index, %arg55: memref<?xui16, 1>, %arg56: i64):
                        %c1_i64_5 = arith.constant 1 : i64
                        %c0_i64_6 = arith.constant 0 : i64
                        %c128_i64_7 = arith.constant 128 : i64
                        %c128_i32_8 = arith.constant 128 : i32
                        %mask, %remaining_out = ascvec.update_mask %c128_i32_8 elem_bits 16 : !ascvec.mask
                        scf.for %arg57 = %c0_i64_6 to %arg56 step %c1_i64_5  : i64 {
                          %247 = arith.muli %arg57, %c128_i64_7 : i64
                          %248 = arith.index_cast %247 : i64 to index
                          %249 = arith.addi %arg54, %248 : index
                          %250 = ascvec.loadalign %arg53[%249] : memref<?xui16, 1>, index -> <ui16>
                          ascvec.storealign %arg55[%248], %250, %mask : memref<?xui16, 1>, index, <ui16>
                        } {cannir.loop_unroll = 1 : i64}
                        ascvec.mem_bar vst_vld
                      } {cannir.vf_mode = "simd"}
                      %233 = arith.addi %220, %c16384_i64 overflow<nsw> : i64
                      %234 = arith.divsi %233, %c1024_i64 : i64
                      %235 = arith.muli %234, %c4_i64 overflow<nsw> : i64
                      %236 = arith.addi %220, %c16639_i64 overflow<nsw> : i64
                      %237 = arith.divsi %236, %c256_i64 : i64
                      %238 = arith.addi %220, %c8193_i64 overflow<nsw> : i64
                      %239 = arith.trunci %238 : i64 to i16
                      %240 = arith.addi %220, %c16511_i64 overflow<nsw> : i64
                      %241 = arith.divsi %240, %c128_i64 : i64
                      ascvec.vf(%134, %234, %233, %235, %237, %239, %134, %241 : memref<?xui16, 1>, i64, i64, i64, i64, i16, memref<?xui16, 1>, i64) {
                      ^bb0(%arg53: memref<?xui16, 1>, %arg54: i64, %arg55: i64, %arg56: i64, %arg57: i64, %arg58: i16, %arg59: memref<?xui16, 1>, %arg60: i64):
                        %c128_i64_5 = arith.constant 128 : i64
                        %c24576 = arith.constant 24576 : index
                        %c127_i16 = arith.constant 127 : i16
                        %c1_i16 = arith.constant 1 : i16
                        %c128_i16 = arith.constant 128 : i16
                        %c-1_i16 = arith.constant -1 : i16
                        %c1_i32_6 = arith.constant 1 : i32
                        %c1_i64_7 = arith.constant 1 : i64
                        %c0_i64_8 = arith.constant 0 : i64
                        %c768_i64 = arith.constant 768 : i64
                        %c512_i64_9 = arith.constant 512 : i64
                        %c256_i64_10 = arith.constant 256 : i64
                        %c1024_i64_11 = arith.constant 1024 : i64
                        %c256_i32_12 = arith.constant 256 : i32
                        %c0_i16 = arith.constant 0 : i16
                        %c128_i32_13 = arith.constant 128 : i32
                        %mask, %remaining_out = ascvec.update_mask %c128_i32_13 elem_bits 16 : !ascvec.mask
                        %247 = ascvec.duplicate_scalar %c0_i16, %mask : i16 -> <ui16>
                        %mask_14, %remaining_out_15 = ascvec.update_mask %c256_i32_12 elem_bits 8 : !ascvec.mask
                        %248:8 = scf.for %arg61 = %c0_i64_8 to %arg54 step %c1_i64_7 iter_args(%arg62 = %247, %arg63 = %247, %arg64 = %247, %arg65 = %247, %arg66 = %247, %arg67 = %247, %arg68 = %247, %arg69 = %247) -> (!ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>)  : i64 {
                          %302 = arith.muli %arg61, %c1024_i64_11 : i64
                          %303 = arith.index_cast %302 : i64 to index
                          %result0, %result1 = ascvec.loadalign_deintlv %arg53[%303] {dintlv_width = "b8"} : memref<?xui16, 1>, index -> <ui8>, <ui8>
                          %304 = ascvec.cumulative_histogram %arg62, %result1, %mask_14 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          %305 = ascvec.cumulative_histogram %arg63, %result1, %mask_14 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          %306 = arith.addi %302, %c256_i64_10 : i64
                          %307 = arith.index_cast %306 : i64 to index
                          %result0_34, %result1_35 = ascvec.loadalign_deintlv %arg53[%307] {dintlv_width = "b8"} : memref<?xui16, 1>, index -> <ui8>, <ui8>
                          %308 = ascvec.cumulative_histogram %arg64, %result1_35, %mask_14 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          %309 = ascvec.cumulative_histogram %arg65, %result1_35, %mask_14 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          %310 = arith.addi %302, %c512_i64_9 : i64
                          %311 = arith.index_cast %310 : i64 to index
                          %result0_36, %result1_37 = ascvec.loadalign_deintlv %arg53[%311] {dintlv_width = "b8"} : memref<?xui16, 1>, index -> <ui8>, <ui8>
                          %312 = ascvec.cumulative_histogram %arg66, %result1_37, %mask_14 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          %313 = ascvec.cumulative_histogram %arg67, %result1_37, %mask_14 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          %314 = arith.addi %302, %c768_i64 : i64
                          %315 = arith.index_cast %314 : i64 to index
                          %result0_38, %result1_39 = ascvec.loadalign_deintlv %arg53[%315] {dintlv_width = "b8"} : memref<?xui16, 1>, index -> <ui8>, <ui8>
                          %316 = ascvec.cumulative_histogram %arg68, %result1_39, %mask_14 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          %317 = ascvec.cumulative_histogram %arg69, %result1_39, %mask_14 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          scf.yield %304, %305, %308, %309, %312, %313, %316, %317 : !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>
                        } {cannir.loop_unroll = 1 : i64}
                        %249:2 = scf.for %arg61 = %arg56 to %arg57 step %c1_i64_7 iter_args(%arg62 = %248#0, %arg63 = %248#1) -> (!ascvec.vreg<ui16>, !ascvec.vreg<ui16>)  : i64 {
                          %302 = arith.muli %arg61, %c256_i64_10 : i64
                          %303 = arith.subi %arg55, %302 : i64
                          %304 = arith.trunci %303 : i64 to i32
                          %mask_34, %remaining_out_35 = ascvec.update_mask %304 elem_bits 8 : !ascvec.mask
                          %305 = arith.index_cast %302 : i64 to index
                          %result0, %result1 = ascvec.loadalign_deintlv %arg53[%305] {dintlv_width = "b8"} : memref<?xui16, 1>, index -> <ui8>, <ui8>
                          %306 = ascvec.cumulative_histogram %arg62, %result1, %mask_34 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          %307 = ascvec.cumulative_histogram %arg63, %result1, %mask_34 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          scf.yield %306, %307 : !ascvec.vreg<ui16>, !ascvec.vreg<ui16>
                        } {cannir.loop_unroll = 1 : i64}
                        %250 = ascvec.add %249#0, %248#2, %mask : <ui16>
                        %251 = ascvec.add %250, %248#4, %mask : <ui16>
                        %252 = ascvec.add %251, %248#6, %mask : <ui16>
                        %253 = ascvec.add %249#1, %248#3, %mask : <ui16>
                        %254 = ascvec.add %253, %248#5, %mask : <ui16>
                        %255 = ascvec.add %254, %248#7, %mask : <ui16>
                        %mask_16, %remaining_out_17 = ascvec.update_mask %c128_i32_13 elem_bits 16 : !ascvec.mask
                        %mask_18, %remaining_out_19 = ascvec.update_mask %c1_i32_6 elem_bits 16 : !ascvec.mask
                        %256 = ascvec.duplicate_scalar %c-1_i16, %mask_16 : i16 -> <ui16>
                        %257 = ascvec.arange %c0_i16 : i16 -> <ui16>
                        %258 = ascvec.cmp_ge_scalar %252, %arg58, %mask_16 : <ui16>, i16 -> !ascvec.mask
                        %259 = ascvec.select %257, %256, %258 : <ui16>
                        %260 = ascvec.arange %c128_i16 : i16 -> <ui16>
                        %261 = ascvec.cmp_ge_scalar %255, %arg58, %mask_16 : <ui16>, i16 -> !ascvec.mask
                        %262 = ascvec.select %260, %256, %261 : <ui16>
                        %263 = ascvec.min %259, %262, %mask_16 : <ui16>
                        %264 = ascvec.reduce_min %263, %mask_16 : <ui16>
                        %mask_20, %remaining_out_21 = ascvec.update_mask %c128_i32_13 elem_bits 16 : !ascvec.mask
                        %mask_22, %remaining_out_23 = ascvec.update_mask %c1_i32_6 elem_bits 16 : !ascvec.mask
                        %265 = ascvec.duplicate_scalar %c0_i16, %mask_20 : i16 -> <ui16>
                        %266 = ascvec.duplicate_scalar %c1_i16, %mask_20 : i16 -> <ui16>
                        %267 = ascvec.duplicate %264, %mask_20 : <ui16> -> <ui16>
                        %268 = ascvec.shl %267, 8, %mask_20 : <ui16> -> <ui16>
                        %269 = ascvec.or %267, %268, %mask_20 : <ui16>, <ui16> -> <ui16>
                        %270 = ascvec.reinterpret %269 : <ui16> -> <ui8>
                        %271 = ascvec.sub %267, %266, %mask_20 : <ui16>
                        %272 = ascvec.cmp_eq_scalar %267, %c0_i16, %mask_20 : <ui16>, i16 -> !ascvec.mask
                        %273 = ascvec.select %265, %271, %272 : <ui16>
                        %274 = ascvec.duplicate_scalar %c127_i16, %mask_20 : i16 -> <ui16>
                        %275 = ascvec.and %273, %274, %mask_20 : <ui16>, <ui16> -> <ui16>
                        %276 = ascvec.gather_reg %252, %275 : <ui16>, <ui16> -> <ui16>
                        %277 = ascvec.gather_reg %255, %275 : <ui16>, <ui16> -> <ui16>
                        %278 = ascvec.cmp_lt_scalar %273, %c128_i16, %mask_20 : <ui16>, i16 -> !ascvec.mask
                        %279 = ascvec.select %276, %277, %278 : <ui16>
                        %280 = ascvec.select %265, %279, %272 : <ui16>
                        %281 = ascvec.duplicate_scalar %arg58, %mask_20 : i16 -> <ui16>
                        %282 = ascvec.sub %281, %280, %mask_20 : <ui16>
                        %283 = ascvec.duplicate %282, %mask_20 : <ui16> -> <ui16>
                        %mask_24, %remaining_out_25 = ascvec.update_mask %c256_i32_12 elem_bits 8 : !ascvec.mask
                        %284:8 = scf.for %arg61 = %c0_i64_8 to %arg54 step %c1_i64_7 iter_args(%arg62 = %265, %arg63 = %265, %arg64 = %265, %arg65 = %265, %arg66 = %265, %arg67 = %265, %arg68 = %265, %arg69 = %265) -> (!ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>)  : i64 {
                          %302 = arith.muli %arg61, %c1024_i64_11 : i64
                          %303 = arith.index_cast %302 : i64 to index
                          %result0, %result1 = ascvec.loadalign_deintlv %arg53[%303] {dintlv_width = "b8"} : memref<?xui16, 1>, index -> <ui8>, <ui8>
                          %304 = ascvec.cmp_eq %result1, %270, %mask_24 : <ui8>, <ui8> -> !ascvec.mask
                          %305 = ascvec.cumulative_histogram %arg62, %result0, %304 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          %306 = ascvec.cumulative_histogram %arg63, %result0, %304 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          %307 = arith.addi %302, %c256_i64_10 : i64
                          %308 = arith.index_cast %307 : i64 to index
                          %result0_34, %result1_35 = ascvec.loadalign_deintlv %arg53[%308] {dintlv_width = "b8"} : memref<?xui16, 1>, index -> <ui8>, <ui8>
                          %309 = ascvec.cmp_eq %result1_35, %270, %mask_24 : <ui8>, <ui8> -> !ascvec.mask
                          %310 = ascvec.cumulative_histogram %arg64, %result0_34, %309 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          %311 = ascvec.cumulative_histogram %arg65, %result0_34, %309 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          %312 = arith.addi %302, %c512_i64_9 : i64
                          %313 = arith.index_cast %312 : i64 to index
                          %result0_36, %result1_37 = ascvec.loadalign_deintlv %arg53[%313] {dintlv_width = "b8"} : memref<?xui16, 1>, index -> <ui8>, <ui8>
                          %314 = ascvec.cmp_eq %result1_37, %270, %mask_24 : <ui8>, <ui8> -> !ascvec.mask
                          %315 = ascvec.cumulative_histogram %arg66, %result0_36, %314 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          %316 = ascvec.cumulative_histogram %arg67, %result0_36, %314 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          %317 = arith.addi %302, %c768_i64 : i64
                          %318 = arith.index_cast %317 : i64 to index
                          %result0_38, %result1_39 = ascvec.loadalign_deintlv %arg53[%318] {dintlv_width = "b8"} : memref<?xui16, 1>, index -> <ui8>, <ui8>
                          %319 = ascvec.cmp_eq %result1_39, %270, %mask_24 : <ui8>, <ui8> -> !ascvec.mask
                          %320 = ascvec.cumulative_histogram %arg68, %result0_38, %319 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          %321 = ascvec.cumulative_histogram %arg69, %result0_38, %319 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          scf.yield %305, %306, %310, %311, %315, %316, %320, %321 : !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>
                        } {cannir.loop_unroll = 1 : i64}
                        %285:2 = scf.for %arg61 = %arg56 to %arg57 step %c1_i64_7 iter_args(%arg62 = %284#0, %arg63 = %284#1) -> (!ascvec.vreg<ui16>, !ascvec.vreg<ui16>)  : i64 {
                          %302 = arith.muli %arg61, %c256_i64_10 : i64
                          %303 = arith.subi %arg55, %302 : i64
                          %304 = arith.trunci %303 : i64 to i32
                          %mask_34, %remaining_out_35 = ascvec.update_mask %304 elem_bits 8 : !ascvec.mask
                          %305 = arith.index_cast %302 : i64 to index
                          %result0, %result1 = ascvec.loadalign_deintlv %arg53[%305] {dintlv_width = "b8"} : memref<?xui16, 1>, index -> <ui8>, <ui8>
                          %306 = ascvec.cmp_eq %result1, %270, %mask_34 : <ui8>, <ui8> -> !ascvec.mask
                          %307 = ascvec.cumulative_histogram %arg62, %result0, %306 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          %308 = ascvec.cumulative_histogram %arg63, %result0, %306 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          scf.yield %307, %308 : !ascvec.vreg<ui16>, !ascvec.vreg<ui16>
                        } {cannir.loop_unroll = 1 : i64}
                        %286 = ascvec.add %285#0, %284#2, %mask_20 : <ui16>
                        %287 = ascvec.add %286, %284#4, %mask_20 : <ui16>
                        %288 = ascvec.add %287, %284#6, %mask_20 : <ui16>
                        %289 = ascvec.add %285#1, %284#3, %mask_20 : <ui16>
                        %290 = ascvec.add %289, %284#5, %mask_20 : <ui16>
                        %291 = ascvec.add %290, %284#7, %mask_20 : <ui16>
                        %mask_26, %remaining_out_27 = ascvec.update_mask %c128_i32_13 elem_bits 16 : !ascvec.mask
                        %mask_28, %remaining_out_29 = ascvec.update_mask %c1_i32_6 elem_bits 16 : !ascvec.mask
                        %292 = ascvec.duplicate_scalar %c-1_i16, %mask_26 : i16 -> <ui16>
                        %293 = ascvec.cmp_ge %288, %283, %mask_26 : <ui16>, <ui16> -> !ascvec.mask
                        %294 = ascvec.select %257, %292, %293 : <ui16>
                        %295 = ascvec.cmp_ge %291, %283, %mask_26 : <ui16>, <ui16> -> !ascvec.mask
                        %296 = ascvec.select %260, %292, %295 : <ui16>
                        %297 = ascvec.min %294, %296, %mask_26 : <ui16>
                        %298 = ascvec.reduce_min %297, %mask_26 : <ui16>
                        %mask_30, %remaining_out_31 = ascvec.update_mask %c128_i32_13 elem_bits 16 : !ascvec.mask
                        %299 = ascvec.duplicate %298, %mask_30 : <ui16> -> <ui16>
                        %300 = ascvec.shl %267, 8, %mask_30 : <ui16> -> <ui16>
                        %301 = ascvec.or %300, %299, %mask_30 : <ui16>, <ui16> -> <ui16>
                        ascvec.store_unalign_begin %arg59 : memref<?xui16, 1>
                        scf.for %arg61 = %c0_i64_8 to %arg60 step %c1_i64_7  : i64 {
                          %302 = arith.muli %arg61, %c128_i64_5 : i64
                          %303 = arith.subi %arg55, %302 : i64
                          %304 = arith.trunci %303 : i64 to i32
                          %mask_34, %remaining_out_35 = ascvec.update_mask %304 elem_bits 16 : !ascvec.mask
                          %305 = arith.index_cast %302 : i64 to index
                          %306 = ascvec.loadalign %arg53[%305] : memref<?xui16, 1>, index -> <ui16>
                          %307 = arith.trunci %302 : i64 to i16
                          %308 = ascvec.arange %307 : i16 -> <ui16>
                          %309 = ascvec.cmp_gt %306, %301, %mask_34 : <ui16>, <ui16> -> !ascvec.mask
                          %310 = ascvec.squeeze_store_reg %308, %309 : <ui16>, !ascvec.mask -> <ui16>
                          ascvec.store_unalign %arg59[%c24576], %310 : memref<?xui16, 1>, <ui16>
                        } {cannir.loop_unroll = 1 : i64}
                        scf.for %arg61 = %c0_i64_8 to %arg60 step %c1_i64_7  : i64 {
                          %302 = arith.muli %arg61, %c128_i64_5 : i64
                          %303 = arith.subi %arg55, %302 : i64
                          %304 = arith.trunci %303 : i64 to i32
                          %mask_34, %remaining_out_35 = ascvec.update_mask %304 elem_bits 16 : !ascvec.mask
                          %305 = arith.index_cast %302 : i64 to index
                          %306 = ascvec.loadalign %arg53[%305] : memref<?xui16, 1>, index -> <ui16>
                          %307 = arith.trunci %302 : i64 to i16
                          %308 = ascvec.arange %307 : i16 -> <ui16>
                          %309 = ascvec.cmp_eq %306, %301, %mask_34 : <ui16>, <ui16> -> !ascvec.mask
                          %310 = ascvec.squeeze_store_reg %308, %309 : <ui16>, !ascvec.mask -> <ui16>
                          ascvec.store_unalign %arg59[%c24576], %310 : memref<?xui16, 1>, <ui16>
                        } {cannir.loop_unroll = 1 : i64}
                        %c0_32 = arith.constant 0 : index
                        %c1_33 = arith.constant 1 : index
                        scf.for %arg61 = %c0_32 to %c1_33 step %c1_33 {
                          ascvec.store_unalign_post %arg59[%c24576] : memref<?xui16, 1>
                        }
                        ascvec.mem_bar vst_vld
                      } {cannir.vf_mode = "simd"}
                      %242 = arith.muli %217, %c8192_i64 overflow<nsw> : i64
                      %243 = arith.index_cast %242 : i64 to index
                      %244 = arith.addi %243, %c59904 : index
                      %245 = arith.addi %231, %c38144 : index
                      %246 = arith.addi %243, %c38144 : index
                      ascvec.vf(%134, %134, %134, %244, %220, %178, %245, %218, %178, %246 : memref<?xui16, 1>, memref<?xui16, 1>, memref<?xui16, 1>, index, i64, memref<?xi32, 1>, index, i64, memref<?xi32, 1>, index) {
                      ^bb0(%arg53: memref<?xui16, 1>, %arg54: memref<?xui16, 1>, %arg55: memref<?xui16, 1>, %arg56: index, %arg57: i64, %arg58: memref<?xi32, 1>, %arg59: index, %arg60: i64, %arg61: memref<?xi32, 1>, %arg62: index):
                        %c1_i32_5 = arith.constant 1 : i32
                        %c64_i32 = arith.constant 64 : i32
                        %c0_i32_6 = arith.constant 0 : i32
                        %c0_7 = arith.constant 0 : index
                        %c24576 = arith.constant 24576 : index
                        %c8192_i32_8 = arith.constant 8192 : i32
                        %c128_i32_9 = arith.constant 128 : i32
                        scf.for %arg63 = %c0_i32_6 to %c64_i32 step %c1_i32_5  : i32 {
                          %247 = arith.muli %arg63, %c128_i32_9 overflow<nsw> : i32
                          %248 = arith.subi %c8192_i32_8, %247 overflow<nsw> : i32
                          %mask, %remaining_out = ascvec.update_mask %248 elem_bits 16 : !ascvec.mask
                          %249 = arith.index_cast %247 : i32 to index
                          %250 = arith.addi %249, %c24576 : index
                          %251 = ascvec.loadalign %arg53[%250] : memref<?xui16, 1>, index -> <ui16>
                          %252 = ascvec.gather %arg54[%c0_7], %251, %mask : memref<?xui16, 1>, <ui16>, !ascvec.mask -> <ui16>
                          %253 = arith.addi %arg56, %249 : index
                          ascvec.storealign %arg55[%253], %252, %mask : memref<?xui16, 1>, index, <ui16>
                        } {cannir.loop_unroll = 1 : i64}
                        %c0_10 = arith.constant 0 : index
                        %c1_11 = arith.constant 1 : index
                        scf.for %arg63 = %c0_10 to %c1_11 step %c1_11 {
                          %mask, %remaining_out = ascvec.update_mask %c64_i32 : !ascvec.mask
                        }
                        scf.for %arg63 = %c0_i32_6 to %c128_i32_9 step %c1_i32_5  : i32 {
                          %247 = arith.muli %arg63, %c64_i32 overflow<nsw> : i32
                          %248 = arith.subi %c8192_i32_8, %247 overflow<nsw> : i32
                          %mask, %remaining_out = ascvec.update_mask %248 : !ascvec.mask
                          %249 = arith.index_cast %247 : i32 to index
                          %250 = arith.addi %249, %c24576 : index
                          %251 = ascvec.loadalign %arg53[%250] : memref<?xui16, 1>, index -> <ui16>
                          %252 = ascvec.unpack %251 {part = "lower"} : <ui16> -> <ui32>
                          %253 = arith.trunci %arg57 : i64 to i32
                          %254 = ascvec.cmp_ge_scalar %252, %253, %mask : <ui32>, i32 -> !ascvec.mask
                          %255 = ascvec.cmp_lt_scalar %252, %253, %mask : <ui32>, i32 -> !ascvec.mask
                          %256 = ascvec.gather %arg58[%arg59], %252, %255 : memref<?xi32, 1>, <ui32>, !ascvec.mask -> <i32>
                          %257 = ascvec.reinterpret %252 : <ui32> -> <i32>
                          %258 = arith.subi %arg60, %arg57 : i64
                          %259 = arith.trunci %258 : i64 to i32
                          %260 = ascvec.add_scalar %257, %259, %mask : <i32>, i32
                          %261 = ascvec.select %260, %256, %254 : <i32>
                          %262 = arith.addi %arg62, %249 : index
                          ascvec.storealign %arg61[%262], %261, %mask : memref<?xi32, 1>, index, <i32>
                        } {cannir.loop_unroll = 1 : i64}
                        ascvec.mem_bar vst_vld
                      } {cannir.vf_mode = "simd"}
                      ascvec.release_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
                    } {cannir.loop_unroll = 1 : i64}
                    %188 = arith.cmpi sgt, %187, %c0_i64 : i64
                    %189 = arith.extui %188 : i1 to i64
                    scf.if %188 {
                      %211 = arith.remsi %185, %c2_i64 : i64
                      %212 = arith.cmpi ne, %211, %c0_i64 : i64
                      %213 = arith.cmpi slt, %211, %c0_i64 : i64
                      %214 = arith.andi %212, %213 : i1
                      %215 = arith.select %214, %c2_i64, %c0_i64 : i64
                      %216 = arith.addi %211, %215 overflow<nsw> : i64
                      %217 = arith.subi %c1_i64, %216 overflow<nsw> : i64
                      %218 = arith.muli %185, %c16384_i64 : i64
                      %219 = arith.cmpi eq, %185, %c0_i64 : i64
                      %220 = arith.cmpi sge, %187, %c8192_i64 : i64
                      %221 = arith.select %219, %220, %false : i1
                      %222 = arith.select %221, %c0_i64, %c8192_i64 : i64
                      %223 = arith.addi %187, %c127_i64 overflow<nsw> : i64
                      %224 = arith.remsi %223, %c128_i64 : i64
                      %225 = arith.cmpi ne, %224, %c0_i64 : i64
                      %226 = arith.cmpi slt, %224, %c0_i64 : i64
                      %227 = arith.andi %225, %226 : i1
                      %228 = arith.divsi %223, %c128_i64 : i64
                      %229 = arith.extui %227 : i1 to i64
                      %230 = arith.subi %228, %229 : i64
                      %231 = cannir.extract_physical_layout %arg12 : !cannir.tensor<<gm, ui16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
                      %232 = cannir.extract_stride %231 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
                      %233 = cannir.extract_dim %232[0] : !cannir.stride<(-1, 1)>
                      %234 = arith.muli %149, %233 : i64
                      %235 = arith.muli %234, %c2_i64 : i64
                      %236 = arith.muli %185, %c32768_i64 : i64
                      %237 = arith.addi %235, %236 : i64
                      %238 = arith.muli %222, %c2_i64 overflow<nsw> : i64
                      %239 = arith.muli %230, %c256_i64 : i64
                      ascvec.get_buf mte2, %c0_i64, 0 {cannir.buffer.generated_sync}
                      "ascvec.copy_gm2ub_plan"(%134, %alloc_2, %c0_i32, %c0_i32, %238, %237, %c1_i64, %239, %c0_i64, %239, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = ui16, ub_base = 0 : i64, ub_capacity = 262144 : i64}> : (memref<?xui16, 1>, memref<1xui16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                      ascvec.release_buf mte2, %c0_i64, 0 {cannir.buffer.generated_sync}
                      %240 = arith.divsi %222, %c128_i64 : i64
                      ascvec.get_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
                      %241 = arith.muli %216, %c8192_i64 overflow<nsw> : i64
                      %242 = arith.index_cast %241 : i64 to index
                      %243 = arith.addi %242, %c59904 : index
                      ascvec.vf(%134, %243, %134, %240 : memref<?xui16, 1>, index, memref<?xui16, 1>, i64) {
                      ^bb0(%arg52: memref<?xui16, 1>, %arg53: index, %arg54: memref<?xui16, 1>, %arg55: i64):
                        %c1_i64_5 = arith.constant 1 : i64
                        %c0_i64_6 = arith.constant 0 : i64
                        %c128_i64_7 = arith.constant 128 : i64
                        %c128_i32_8 = arith.constant 128 : i32
                        %mask, %remaining_out = ascvec.update_mask %c128_i32_8 elem_bits 16 : !ascvec.mask
                        scf.for %arg56 = %c0_i64_6 to %arg55 step %c1_i64_5  : i64 {
                          %276 = arith.muli %arg56, %c128_i64_7 : i64
                          %277 = arith.index_cast %276 : i64 to index
                          %278 = arith.addi %arg53, %277 : index
                          %279 = ascvec.loadalign %arg52[%278] : memref<?xui16, 1>, index -> <ui16>
                          ascvec.storealign %arg54[%277], %279, %mask : memref<?xui16, 1>, index, <ui16>
                        } {cannir.loop_unroll = 1 : i64}
                        ascvec.mem_bar vst_vld
                      } {cannir.vf_mode = "simd"}
                      %244 = arith.addi %222, %187 overflow<nsw> : i64
                      %245 = arith.remsi %244, %c1024_i64 : i64
                      %246 = arith.cmpi ne, %245, %c0_i64 : i64
                      %247 = arith.cmpi slt, %245, %c0_i64 : i64
                      %248 = arith.andi %246, %247 : i1
                      %249 = arith.divsi %244, %c1024_i64 : i64
                      %250 = arith.extui %248 : i1 to i64
                      %251 = arith.subi %249, %250 : i64
                      %252 = arith.muli %251, %c4_i64 : i64
                      %253 = arith.addi %244, %c255_i64 overflow<nsw> : i64
                      %254 = arith.remsi %253, %c256_i64 : i64
                      %255 = arith.cmpi ne, %254, %c0_i64 : i64
                      %256 = arith.cmpi slt, %254, %c0_i64 : i64
                      %257 = arith.andi %255, %256 : i1
                      %258 = arith.divsi %253, %c256_i64 : i64
                      %259 = arith.extui %257 : i1 to i64
                      %260 = arith.subi %258, %259 : i64
                      %261 = arith.addi %244, %c-8191_i64 overflow<nsw> : i64
                      %262 = arith.trunci %261 : i64 to i16
                      %263 = arith.addi %244, %c127_i64 overflow<nsw> : i64
                      %264 = arith.remsi %263, %c128_i64 : i64
                      %265 = arith.cmpi ne, %264, %c0_i64 : i64
                      %266 = arith.cmpi slt, %264, %c0_i64 : i64
                      %267 = arith.andi %265, %266 : i1
                      %268 = arith.divsi %263, %c128_i64 : i64
                      %269 = arith.extui %267 : i1 to i64
                      %270 = arith.subi %268, %269 : i64
                      ascvec.vf(%134, %251, %244, %252, %260, %262, %134, %270 : memref<?xui16, 1>, i64, i64, i64, i64, i16, memref<?xui16, 1>, i64) {
                      ^bb0(%arg52: memref<?xui16, 1>, %arg53: i64, %arg54: i64, %arg55: i64, %arg56: i64, %arg57: i16, %arg58: memref<?xui16, 1>, %arg59: i64):
                        %c128_i64_5 = arith.constant 128 : i64
                        %c24576 = arith.constant 24576 : index
                        %c127_i16 = arith.constant 127 : i16
                        %c1_i16 = arith.constant 1 : i16
                        %c128_i16 = arith.constant 128 : i16
                        %c-1_i16 = arith.constant -1 : i16
                        %c1_i32_6 = arith.constant 1 : i32
                        %c1_i64_7 = arith.constant 1 : i64
                        %c0_i64_8 = arith.constant 0 : i64
                        %c768_i64 = arith.constant 768 : i64
                        %c512_i64_9 = arith.constant 512 : i64
                        %c256_i64_10 = arith.constant 256 : i64
                        %c1024_i64_11 = arith.constant 1024 : i64
                        %c256_i32_12 = arith.constant 256 : i32
                        %c0_i16 = arith.constant 0 : i16
                        %c128_i32_13 = arith.constant 128 : i32
                        %mask, %remaining_out = ascvec.update_mask %c128_i32_13 elem_bits 16 : !ascvec.mask
                        %276 = ascvec.duplicate_scalar %c0_i16, %mask : i16 -> <ui16>
                        %mask_14, %remaining_out_15 = ascvec.update_mask %c256_i32_12 elem_bits 8 : !ascvec.mask
                        %277:8 = scf.for %arg60 = %c0_i64_8 to %arg53 step %c1_i64_7 iter_args(%arg61 = %276, %arg62 = %276, %arg63 = %276, %arg64 = %276, %arg65 = %276, %arg66 = %276, %arg67 = %276, %arg68 = %276) -> (!ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>)  : i64 {
                          %331 = arith.muli %arg60, %c1024_i64_11 : i64
                          %332 = arith.index_cast %331 : i64 to index
                          %result0, %result1 = ascvec.loadalign_deintlv %arg52[%332] {dintlv_width = "b8"} : memref<?xui16, 1>, index -> <ui8>, <ui8>
                          %333 = ascvec.cumulative_histogram %arg61, %result1, %mask_14 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          %334 = ascvec.cumulative_histogram %arg62, %result1, %mask_14 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          %335 = arith.addi %331, %c256_i64_10 : i64
                          %336 = arith.index_cast %335 : i64 to index
                          %result0_34, %result1_35 = ascvec.loadalign_deintlv %arg52[%336] {dintlv_width = "b8"} : memref<?xui16, 1>, index -> <ui8>, <ui8>
                          %337 = ascvec.cumulative_histogram %arg63, %result1_35, %mask_14 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          %338 = ascvec.cumulative_histogram %arg64, %result1_35, %mask_14 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          %339 = arith.addi %331, %c512_i64_9 : i64
                          %340 = arith.index_cast %339 : i64 to index
                          %result0_36, %result1_37 = ascvec.loadalign_deintlv %arg52[%340] {dintlv_width = "b8"} : memref<?xui16, 1>, index -> <ui8>, <ui8>
                          %341 = ascvec.cumulative_histogram %arg65, %result1_37, %mask_14 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          %342 = ascvec.cumulative_histogram %arg66, %result1_37, %mask_14 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          %343 = arith.addi %331, %c768_i64 : i64
                          %344 = arith.index_cast %343 : i64 to index
                          %result0_38, %result1_39 = ascvec.loadalign_deintlv %arg52[%344] {dintlv_width = "b8"} : memref<?xui16, 1>, index -> <ui8>, <ui8>
                          %345 = ascvec.cumulative_histogram %arg67, %result1_39, %mask_14 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          %346 = ascvec.cumulative_histogram %arg68, %result1_39, %mask_14 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          scf.yield %333, %334, %337, %338, %341, %342, %345, %346 : !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>
                        } {cannir.loop_unroll = 1 : i64}
                        %278:2 = scf.for %arg60 = %arg55 to %arg56 step %c1_i64_7 iter_args(%arg61 = %277#0, %arg62 = %277#1) -> (!ascvec.vreg<ui16>, !ascvec.vreg<ui16>)  : i64 {
                          %331 = arith.muli %arg60, %c256_i64_10 : i64
                          %332 = arith.subi %arg54, %331 : i64
                          %333 = arith.trunci %332 : i64 to i32
                          %mask_34, %remaining_out_35 = ascvec.update_mask %333 elem_bits 8 : !ascvec.mask
                          %334 = arith.index_cast %331 : i64 to index
                          %result0, %result1 = ascvec.loadalign_deintlv %arg52[%334] {dintlv_width = "b8"} : memref<?xui16, 1>, index -> <ui8>, <ui8>
                          %335 = ascvec.cumulative_histogram %arg61, %result1, %mask_34 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          %336 = ascvec.cumulative_histogram %arg62, %result1, %mask_34 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          scf.yield %335, %336 : !ascvec.vreg<ui16>, !ascvec.vreg<ui16>
                        } {cannir.loop_unroll = 1 : i64}
                        %279 = ascvec.add %278#0, %277#2, %mask : <ui16>
                        %280 = ascvec.add %279, %277#4, %mask : <ui16>
                        %281 = ascvec.add %280, %277#6, %mask : <ui16>
                        %282 = ascvec.add %278#1, %277#3, %mask : <ui16>
                        %283 = ascvec.add %282, %277#5, %mask : <ui16>
                        %284 = ascvec.add %283, %277#7, %mask : <ui16>
                        %mask_16, %remaining_out_17 = ascvec.update_mask %c128_i32_13 elem_bits 16 : !ascvec.mask
                        %mask_18, %remaining_out_19 = ascvec.update_mask %c1_i32_6 elem_bits 16 : !ascvec.mask
                        %285 = ascvec.duplicate_scalar %c-1_i16, %mask_16 : i16 -> <ui16>
                        %286 = ascvec.arange %c0_i16 : i16 -> <ui16>
                        %287 = ascvec.cmp_ge_scalar %281, %arg57, %mask_16 : <ui16>, i16 -> !ascvec.mask
                        %288 = ascvec.select %286, %285, %287 : <ui16>
                        %289 = ascvec.arange %c128_i16 : i16 -> <ui16>
                        %290 = ascvec.cmp_ge_scalar %284, %arg57, %mask_16 : <ui16>, i16 -> !ascvec.mask
                        %291 = ascvec.select %289, %285, %290 : <ui16>
                        %292 = ascvec.min %288, %291, %mask_16 : <ui16>
                        %293 = ascvec.reduce_min %292, %mask_16 : <ui16>
                        %mask_20, %remaining_out_21 = ascvec.update_mask %c128_i32_13 elem_bits 16 : !ascvec.mask
                        %mask_22, %remaining_out_23 = ascvec.update_mask %c1_i32_6 elem_bits 16 : !ascvec.mask
                        %294 = ascvec.duplicate_scalar %c0_i16, %mask_20 : i16 -> <ui16>
                        %295 = ascvec.duplicate_scalar %c1_i16, %mask_20 : i16 -> <ui16>
                        %296 = ascvec.duplicate %293, %mask_20 : <ui16> -> <ui16>
                        %297 = ascvec.shl %296, 8, %mask_20 : <ui16> -> <ui16>
                        %298 = ascvec.or %296, %297, %mask_20 : <ui16>, <ui16> -> <ui16>
                        %299 = ascvec.reinterpret %298 : <ui16> -> <ui8>
                        %300 = ascvec.sub %296, %295, %mask_20 : <ui16>
                        %301 = ascvec.cmp_eq_scalar %296, %c0_i16, %mask_20 : <ui16>, i16 -> !ascvec.mask
                        %302 = ascvec.select %294, %300, %301 : <ui16>
                        %303 = ascvec.duplicate_scalar %c127_i16, %mask_20 : i16 -> <ui16>
                        %304 = ascvec.and %302, %303, %mask_20 : <ui16>, <ui16> -> <ui16>
                        %305 = ascvec.gather_reg %281, %304 : <ui16>, <ui16> -> <ui16>
                        %306 = ascvec.gather_reg %284, %304 : <ui16>, <ui16> -> <ui16>
                        %307 = ascvec.cmp_lt_scalar %302, %c128_i16, %mask_20 : <ui16>, i16 -> !ascvec.mask
                        %308 = ascvec.select %305, %306, %307 : <ui16>
                        %309 = ascvec.select %294, %308, %301 : <ui16>
                        %310 = ascvec.duplicate_scalar %arg57, %mask_20 : i16 -> <ui16>
                        %311 = ascvec.sub %310, %309, %mask_20 : <ui16>
                        %312 = ascvec.duplicate %311, %mask_20 : <ui16> -> <ui16>
                        %mask_24, %remaining_out_25 = ascvec.update_mask %c256_i32_12 elem_bits 8 : !ascvec.mask
                        %313:8 = scf.for %arg60 = %c0_i64_8 to %arg53 step %c1_i64_7 iter_args(%arg61 = %294, %arg62 = %294, %arg63 = %294, %arg64 = %294, %arg65 = %294, %arg66 = %294, %arg67 = %294, %arg68 = %294) -> (!ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>)  : i64 {
                          %331 = arith.muli %arg60, %c1024_i64_11 : i64
                          %332 = arith.index_cast %331 : i64 to index
                          %result0, %result1 = ascvec.loadalign_deintlv %arg52[%332] {dintlv_width = "b8"} : memref<?xui16, 1>, index -> <ui8>, <ui8>
                          %333 = ascvec.cmp_eq %result1, %299, %mask_24 : <ui8>, <ui8> -> !ascvec.mask
                          %334 = ascvec.cumulative_histogram %arg61, %result0, %333 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          %335 = ascvec.cumulative_histogram %arg62, %result0, %333 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          %336 = arith.addi %331, %c256_i64_10 : i64
                          %337 = arith.index_cast %336 : i64 to index
                          %result0_34, %result1_35 = ascvec.loadalign_deintlv %arg52[%337] {dintlv_width = "b8"} : memref<?xui16, 1>, index -> <ui8>, <ui8>
                          %338 = ascvec.cmp_eq %result1_35, %299, %mask_24 : <ui8>, <ui8> -> !ascvec.mask
                          %339 = ascvec.cumulative_histogram %arg63, %result0_34, %338 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          %340 = ascvec.cumulative_histogram %arg64, %result0_34, %338 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          %341 = arith.addi %331, %c512_i64_9 : i64
                          %342 = arith.index_cast %341 : i64 to index
                          %result0_36, %result1_37 = ascvec.loadalign_deintlv %arg52[%342] {dintlv_width = "b8"} : memref<?xui16, 1>, index -> <ui8>, <ui8>
                          %343 = ascvec.cmp_eq %result1_37, %299, %mask_24 : <ui8>, <ui8> -> !ascvec.mask
                          %344 = ascvec.cumulative_histogram %arg65, %result0_36, %343 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          %345 = ascvec.cumulative_histogram %arg66, %result0_36, %343 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          %346 = arith.addi %331, %c768_i64 : i64
                          %347 = arith.index_cast %346 : i64 to index
                          %result0_38, %result1_39 = ascvec.loadalign_deintlv %arg52[%347] {dintlv_width = "b8"} : memref<?xui16, 1>, index -> <ui8>, <ui8>
                          %348 = ascvec.cmp_eq %result1_39, %299, %mask_24 : <ui8>, <ui8> -> !ascvec.mask
                          %349 = ascvec.cumulative_histogram %arg67, %result0_38, %348 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          %350 = ascvec.cumulative_histogram %arg68, %result0_38, %348 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          scf.yield %334, %335, %339, %340, %344, %345, %349, %350 : !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>
                        } {cannir.loop_unroll = 1 : i64}
                        %314:2 = scf.for %arg60 = %arg55 to %arg56 step %c1_i64_7 iter_args(%arg61 = %313#0, %arg62 = %313#1) -> (!ascvec.vreg<ui16>, !ascvec.vreg<ui16>)  : i64 {
                          %331 = arith.muli %arg60, %c256_i64_10 : i64
                          %332 = arith.subi %arg54, %331 : i64
                          %333 = arith.trunci %332 : i64 to i32
                          %mask_34, %remaining_out_35 = ascvec.update_mask %333 elem_bits 8 : !ascvec.mask
                          %334 = arith.index_cast %331 : i64 to index
                          %result0, %result1 = ascvec.loadalign_deintlv %arg52[%334] {dintlv_width = "b8"} : memref<?xui16, 1>, index -> <ui8>, <ui8>
                          %335 = ascvec.cmp_eq %result1, %299, %mask_34 : <ui8>, <ui8> -> !ascvec.mask
                          %336 = ascvec.cumulative_histogram %arg61, %result0, %335 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          %337 = ascvec.cumulative_histogram %arg62, %result0, %335 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          scf.yield %336, %337 : !ascvec.vreg<ui16>, !ascvec.vreg<ui16>
                        } {cannir.loop_unroll = 1 : i64}
                        %315 = ascvec.add %314#0, %313#2, %mask_20 : <ui16>
                        %316 = ascvec.add %315, %313#4, %mask_20 : <ui16>
                        %317 = ascvec.add %316, %313#6, %mask_20 : <ui16>
                        %318 = ascvec.add %314#1, %313#3, %mask_20 : <ui16>
                        %319 = ascvec.add %318, %313#5, %mask_20 : <ui16>
                        %320 = ascvec.add %319, %313#7, %mask_20 : <ui16>
                        %mask_26, %remaining_out_27 = ascvec.update_mask %c128_i32_13 elem_bits 16 : !ascvec.mask
                        %mask_28, %remaining_out_29 = ascvec.update_mask %c1_i32_6 elem_bits 16 : !ascvec.mask
                        %321 = ascvec.duplicate_scalar %c-1_i16, %mask_26 : i16 -> <ui16>
                        %322 = ascvec.cmp_ge %317, %312, %mask_26 : <ui16>, <ui16> -> !ascvec.mask
                        %323 = ascvec.select %286, %321, %322 : <ui16>
                        %324 = ascvec.cmp_ge %320, %312, %mask_26 : <ui16>, <ui16> -> !ascvec.mask
                        %325 = ascvec.select %289, %321, %324 : <ui16>
                        %326 = ascvec.min %323, %325, %mask_26 : <ui16>
                        %327 = ascvec.reduce_min %326, %mask_26 : <ui16>
                        %mask_30, %remaining_out_31 = ascvec.update_mask %c128_i32_13 elem_bits 16 : !ascvec.mask
                        %328 = ascvec.duplicate %327, %mask_30 : <ui16> -> <ui16>
                        %329 = ascvec.shl %296, 8, %mask_30 : <ui16> -> <ui16>
                        %330 = ascvec.or %329, %328, %mask_30 : <ui16>, <ui16> -> <ui16>
                        ascvec.store_unalign_begin %arg58 : memref<?xui16, 1>
                        scf.for %arg60 = %c0_i64_8 to %arg59 step %c1_i64_7  : i64 {
                          %331 = arith.muli %arg60, %c128_i64_5 : i64
                          %332 = arith.subi %arg54, %331 : i64
                          %333 = arith.trunci %332 : i64 to i32
                          %mask_34, %remaining_out_35 = ascvec.update_mask %333 elem_bits 16 : !ascvec.mask
                          %334 = arith.index_cast %331 : i64 to index
                          %335 = ascvec.loadalign %arg52[%334] : memref<?xui16, 1>, index -> <ui16>
                          %336 = arith.trunci %331 : i64 to i16
                          %337 = ascvec.arange %336 : i16 -> <ui16>
                          %338 = ascvec.cmp_gt %335, %330, %mask_34 : <ui16>, <ui16> -> !ascvec.mask
                          %339 = ascvec.squeeze_store_reg %337, %338 : <ui16>, !ascvec.mask -> <ui16>
                          ascvec.store_unalign %arg58[%c24576], %339 : memref<?xui16, 1>, <ui16>
                        } {cannir.loop_unroll = 1 : i64}
                        scf.for %arg60 = %c0_i64_8 to %arg59 step %c1_i64_7  : i64 {
                          %331 = arith.muli %arg60, %c128_i64_5 : i64
                          %332 = arith.subi %arg54, %331 : i64
                          %333 = arith.trunci %332 : i64 to i32
                          %mask_34, %remaining_out_35 = ascvec.update_mask %333 elem_bits 16 : !ascvec.mask
                          %334 = arith.index_cast %331 : i64 to index
                          %335 = ascvec.loadalign %arg52[%334] : memref<?xui16, 1>, index -> <ui16>
                          %336 = arith.trunci %331 : i64 to i16
                          %337 = ascvec.arange %336 : i16 -> <ui16>
                          %338 = ascvec.cmp_eq %335, %330, %mask_34 : <ui16>, <ui16> -> !ascvec.mask
                          %339 = ascvec.squeeze_store_reg %337, %338 : <ui16>, !ascvec.mask -> <ui16>
                          ascvec.store_unalign %arg58[%c24576], %339 : memref<?xui16, 1>, <ui16>
                        } {cannir.loop_unroll = 1 : i64}
                        %c0_32 = arith.constant 0 : index
                        %c1_33 = arith.constant 1 : index
                        scf.for %arg60 = %c0_32 to %c1_33 step %c1_33 {
                          ascvec.store_unalign_post %arg58[%c24576] : memref<?xui16, 1>
                        }
                        ascvec.mem_bar vst_vld
                      } {cannir.vf_mode = "simd"}
                      %271 = arith.muli %217, %c8192_i64 overflow<nsw> : i64
                      %272 = arith.index_cast %271 : i64 to index
                      %273 = arith.addi %272, %c59904 : index
                      %274 = arith.addi %242, %c38144 : index
                      %275 = arith.addi %272, %c38144 : index
                      ascvec.vf(%134, %134, %134, %273, %222, %178, %274, %218, %178, %275 : memref<?xui16, 1>, memref<?xui16, 1>, memref<?xui16, 1>, index, i64, memref<?xi32, 1>, index, i64, memref<?xi32, 1>, index) {
                      ^bb0(%arg52: memref<?xui16, 1>, %arg53: memref<?xui16, 1>, %arg54: memref<?xui16, 1>, %arg55: index, %arg56: i64, %arg57: memref<?xi32, 1>, %arg58: index, %arg59: i64, %arg60: memref<?xi32, 1>, %arg61: index):
                        %c1_i32_5 = arith.constant 1 : i32
                        %c64_i32 = arith.constant 64 : i32
                        %c0_i32_6 = arith.constant 0 : i32
                        %c0_7 = arith.constant 0 : index
                        %c24576 = arith.constant 24576 : index
                        %c8192_i32_8 = arith.constant 8192 : i32
                        %c128_i32_9 = arith.constant 128 : i32
                        scf.for %arg62 = %c0_i32_6 to %c64_i32 step %c1_i32_5  : i32 {
                          %276 = arith.muli %arg62, %c128_i32_9 overflow<nsw> : i32
                          %277 = arith.subi %c8192_i32_8, %276 overflow<nsw> : i32
                          %mask, %remaining_out = ascvec.update_mask %277 elem_bits 16 : !ascvec.mask
                          %278 = arith.index_cast %276 : i32 to index
                          %279 = arith.addi %278, %c24576 : index
                          %280 = ascvec.loadalign %arg52[%279] : memref<?xui16, 1>, index -> <ui16>
                          %281 = ascvec.gather %arg53[%c0_7], %280, %mask : memref<?xui16, 1>, <ui16>, !ascvec.mask -> <ui16>
                          %282 = arith.addi %arg55, %278 : index
                          ascvec.storealign %arg54[%282], %281, %mask : memref<?xui16, 1>, index, <ui16>
                        } {cannir.loop_unroll = 1 : i64}
                        %c0_10 = arith.constant 0 : index
                        %c1_11 = arith.constant 1 : index
                        scf.for %arg62 = %c0_10 to %c1_11 step %c1_11 {
                          %mask, %remaining_out = ascvec.update_mask %c64_i32 : !ascvec.mask
                        }
                        scf.for %arg62 = %c0_i32_6 to %c128_i32_9 step %c1_i32_5  : i32 {
                          %276 = arith.muli %arg62, %c64_i32 overflow<nsw> : i32
                          %277 = arith.subi %c8192_i32_8, %276 overflow<nsw> : i32
                          %mask, %remaining_out = ascvec.update_mask %277 : !ascvec.mask
                          %278 = arith.index_cast %276 : i32 to index
                          %279 = arith.addi %278, %c24576 : index
                          %280 = ascvec.loadalign %arg52[%279] : memref<?xui16, 1>, index -> <ui16>
                          %281 = ascvec.unpack %280 {part = "lower"} : <ui16> -> <ui32>
                          %282 = arith.trunci %arg56 : i64 to i32
                          %283 = ascvec.cmp_ge_scalar %281, %282, %mask : <ui32>, i32 -> !ascvec.mask
                          %284 = ascvec.cmp_lt_scalar %281, %282, %mask : <ui32>, i32 -> !ascvec.mask
                          %285 = ascvec.gather %arg57[%arg58], %281, %284 : memref<?xi32, 1>, <ui32>, !ascvec.mask -> <i32>
                          %286 = ascvec.reinterpret %281 : <ui32> -> <i32>
                          %287 = arith.subi %arg59, %arg56 : i64
                          %288 = arith.trunci %287 : i64 to i32
                          %289 = ascvec.add_scalar %286, %288, %mask : <i32>, i32
                          %290 = ascvec.select %289, %285, %283 : <i32>
                          %291 = arith.addi %arg61, %278 : index
                          ascvec.storealign %arg60[%291], %290, %mask : memref<?xi32, 1>, index, <i32>
                        } {cannir.loop_unroll = 1 : i64}
                        ascvec.mem_bar vst_vld
                      } {cannir.vf_mode = "simd"}
                      ascvec.release_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
                    }
                    %190 = arith.addi %185, %189 : i64
                    %191 = arith.remsi %190, %c2_i64 : i64
                    %192 = arith.cmpi ne, %191, %c0_i64 : i64
                    %193 = arith.cmpi slt, %191, %c0_i64 : i64
                    %194 = arith.andi %192, %193 : i1
                    %195 = arith.select %194, %c2_i64, %c0_i64 : i64
                    %196 = arith.addi %191, %195 overflow<nsw> : i64
                    ascvec.get_buf v, %c1_i64, 0 {cannir.buffer.generated_sync}
                    ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                    %197 = arith.muli %196, %c8192_i64 overflow<nsw> : i64
                    %198 = arith.index_cast %197 : i64 to index
                    %199 = arith.addi %198, %c38144 : index
                    %200 = arith.addi %198, %c59904 : index
                    ascvec.vf(%178, %199, %164, %134, %200, %173, %175, %177 : memref<?xi32, 1>, index, i32, memref<?xui16, 1>, index, i32, memref<8192xi32, 1>, memref<8192xui16, 1>) {
                    ^bb0(%arg52: memref<?xi32, 1>, %arg53: index, %arg54: i32, %arg55: memref<?xui16, 1>, %arg56: index, %arg57: i32, %arg58: memref<8192xi32, 1>, %arg59: memref<8192xui16, 1>):
                      %c1_i32_5 = arith.constant 1 : i32
                      %c128_i32_6 = arith.constant 128 : i32
                      %c0_i32_7 = arith.constant 0 : i32
                      %c-128_i16 = arith.constant -128 : i16
                      %c-1_i32_8 = arith.constant -1 : i32
                      %c-1_i16 = arith.constant -1 : i16
                      %c-32768_i16 = arith.constant -32768 : i16
                      %c8192_i32_9 = arith.constant 8192 : i32
                      %c64_i32 = arith.constant 64 : i32
                      scf.for %arg60 = %c0_i32_7 to %c128_i32_6 step %c1_i32_5  : i32 {
                        %211 = arith.muli %arg60, %c64_i32 overflow<nsw> : i32
                        %212 = arith.subi %c8192_i32_9, %211 overflow<nsw> : i32
                        %mask, %remaining_out = ascvec.update_mask %212 : !ascvec.mask
                        %mask_10, %remaining_out_11 = ascvec.update_mask %212 elem_bits 16 : !ascvec.mask
                        %mask_12, %remaining_out_13 = ascvec.update_mask %c64_i32 elem_bits 16 : !ascvec.mask
                        %mask_14, %remaining_out_15 = ascvec.update_mask %c64_i32 elem_bits 16 : !ascvec.mask
                        %213 = ascvec.mask_and %mask_10, %mask_12, %mask_14
                        %214 = arith.index_cast %211 : i32 to index
                        %215 = arith.addi %arg53, %214 : index
                        %216 = ascvec.loadalign %arg52[%215] : memref<?xi32, 1>, index -> <i32>
                        %217 = ascvec.add_scalar %216, %arg54, %mask : <i32>, i32
                        %218 = arith.addi %arg56, %214 : index
                        %219 = ascvec.loadalign %arg55[%218] : memref<?xui16, 1>, index -> <ui16>
                        %220 = ascvec.duplicate_scalar %c-32768_i16, %213 : i16 -> <ui16>
                        %221 = ascvec.and %219, %220, %213 : <ui16>, <ui16> -> <ui16>
                        %222 = ascvec.cmp_eq %221, %220, %213 : <ui16>, <ui16> -> !ascvec.mask
                        %223 = ascvec.xor %219, %220, %213 : <ui16>, <ui16> -> <ui16>
                        %224 = ascvec.duplicate_scalar %c-1_i16, %213 : i16 -> <ui16>
                        %225 = ascvec.xor %219, %224, %213 : <ui16>, <ui16> -> <ui16>
                        %226 = ascvec.select %223, %225, %222 : <ui16>
                        %227 = ascvec.arange %211 : i32 -> <ui32>
                        %228 = ascvec.cmp_lt_scalar %227, %arg57, %mask : <ui32>, i32 -> !ascvec.mask
                        %229 = arith.trunci %211 : i32 to i16
                        %230 = ascvec.arange %229 : i16 -> <ui16>
                        %231 = arith.trunci %arg57 : i32 to i16
                        %232 = ascvec.cmp_lt_scalar %230, %231, %213 : <ui16>, i16 -> !ascvec.mask
                        %233 = ascvec.duplicate_scalar %c-1_i32_8, %mask : i32 -> <i32>
                        %234 = ascvec.select %217, %233, %228 : <i32>
                        %235 = ascvec.duplicate_scalar %c-128_i16, %213 : i16 -> <ui16>
                        %236 = ascvec.select %226, %235, %232 : <ui16>
                        ascvec.storealign %arg58[%214], %234, %mask : memref<8192xi32, 1>, index, <i32>
                        ascvec.storealign %arg59[%214], %236, %213 : memref<8192xui16, 1>, index, <ui16>
                      } {cannir.loop_unroll = 1 : i64}
                      ascvec.mem_bar vst_vld
                    } {cannir.vf_mode = "simd"}
                    ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                    ascvec.release_buf v, %c1_i64, 0 {cannir.buffer.generated_sync}
                    %201 = arith.muli %147, %c262144_i64 : i64
                    %202 = arith.divui %201, %c8_i64 : i64
                    %203 = arith.addi %82, %202 : i64
                    %204 = ascscalar.bind_gm_ptr %203 : memref<1xi32>
                    %205 = ascscalar.bind_local_ptr %174 : memref<1xi32, 1>
                    ascvec.get_buf mte3, %c1_i64, 0 {cannir.buffer.generated_sync}
                    ascvec.copy_ub2gm %204[%c0], %205[%c0], %c1_i32, %c32768_i32, %c0_i64, %c32768_i32 l2_cache_ctl %c0_i32 : memref<1xi32>, memref<1xi32, 1>
                    ascvec.release_buf mte3, %c1_i64, 0 {cannir.buffer.generated_sync}
                    %206 = arith.muli %147, %c131072_i64 : i64
                    %207 = arith.divui %206, %c8_i64 : i64
                    %208 = arith.addi %83, %207 : i64
                    %209 = ascscalar.bind_gm_ptr %208 : memref<1xui16>
                    %210 = ascscalar.bind_local_ptr %176 : memref<1xui16, 1>
                    ascvec.get_buf mte3, %c2_i64, 0 {cannir.buffer.generated_sync}
                    ascvec.copy_ub2gm %209[%c0], %210[%c0], %c1_i32, %c16384_i32, %c0_i64, %c16384_i32 l2_cache_ctl %c0_i32 : memref<1xui16>, memref<1xui16, 1>
                    ascvec.release_buf mte3, %c2_i64, 0 {cannir.buffer.generated_sync}
                  }
                }
              }
            } {cannir.loop_unroll = 1 : i64}
            scf.yield %105, %142#0, %142#1, %142#2, %142#3 : i64, i64, i64, i64, i64
          } else {
            %97 = arith.muli %4, %c3_i64 : i64
            %98 = arith.trunci %97 : i64 to i32
            %99 = arith.extsi %98 : i32 to i64
            %100 = arith.addi %50, %99 : i64
            %101 = arith.cmpi slt, %98, %53 : i32
            scf.if %101 {
              %112 = ascscalar.memref_address %alloc_4 : memref<262144xui8, 1>
              %113 = arith.addi %112, %c218112_i64 : i64
              %114 = ascscalar.bind_local_ptr %113 : memref<8192xi32, 1>
              %115 = arith.addi %112, %c98304_i64 : i64
              %116 = ascscalar.bind_local_ptr %115 : memref<8192xui16, 1>
              ascvec.get_buf v, %c1_i64, 0 {cannir.buffer.generated_sync}
              ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
              ascvec.vf(%114, %116 : memref<8192xi32, 1>, memref<8192xui16, 1>) {
              ^bb0(%arg51: memref<8192xi32, 1>, %arg52: memref<8192xui16, 1>):
                %c1_i32_5 = arith.constant 1 : i32
                %c128_i32_6 = arith.constant 128 : i32
                %c0_i32_7 = arith.constant 0 : i32
                %c-128_i16 = arith.constant -128 : i16
                %c-1_i32_8 = arith.constant -1 : i32
                %c8192_i32_9 = arith.constant 8192 : i32
                %c64_i32 = arith.constant 64 : i32
                scf.for %arg53 = %c0_i32_7 to %c128_i32_6 step %c1_i32_5  : i32 {
                  %127 = arith.muli %arg53, %c64_i32 overflow<nsw> : i32
                  %128 = arith.subi %c8192_i32_9, %127 overflow<nsw> : i32
                  %mask, %remaining_out = ascvec.update_mask %128 : !ascvec.mask
                  %mask_10, %remaining_out_11 = ascvec.update_mask %128 elem_bits 16 : !ascvec.mask
                  %mask_12, %remaining_out_13 = ascvec.update_mask %c64_i32 elem_bits 16 : !ascvec.mask
                  %mask_14, %remaining_out_15 = ascvec.update_mask %c64_i32 elem_bits 16 : !ascvec.mask
                  %129 = ascvec.mask_and %mask_10, %mask_12, %mask_14
                  %130 = ascvec.duplicate_scalar %c-1_i32_8, %mask : i32 -> <i32>
                  %131 = arith.index_cast %127 : i32 to index
                  ascvec.storealign %arg51[%131], %130, %mask : memref<8192xi32, 1>, index, <i32>
                  %132 = ascvec.duplicate_scalar %c-128_i16, %129 : i16 -> <ui16>
                  ascvec.storealign %arg52[%131], %132, %129 : memref<8192xui16, 1>, index, <ui16>
                } {cannir.loop_unroll = 1 : i64}
                ascvec.mem_bar vst_vld
              } {cannir.vf_mode = "simd"}
              ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
              ascvec.release_buf v, %c1_i64, 0 {cannir.buffer.generated_sync}
              %117 = arith.muli %100, %c262144_i64 : i64
              %118 = arith.divui %117, %c8_i64 : i64
              %119 = arith.addi %82, %118 : i64
              %120 = ascscalar.bind_gm_ptr %119 : memref<1xi32>
              %121 = ascscalar.bind_local_ptr %113 : memref<1xi32, 1>
              ascvec.get_buf mte3, %c1_i64, 0 {cannir.buffer.generated_sync}
              ascvec.copy_ub2gm %120[%c0], %121[%c0], %c1_i32, %c32768_i32, %c0_i64, %c32768_i32 l2_cache_ctl %c0_i32 : memref<1xi32>, memref<1xi32, 1>
              ascvec.release_buf mte3, %c1_i64, 0 {cannir.buffer.generated_sync}
              %122 = arith.muli %100, %c131072_i64 : i64
              %123 = arith.divui %122, %c8_i64 : i64
              %124 = arith.addi %83, %123 : i64
              %125 = ascscalar.bind_gm_ptr %124 : memref<1xui16>
              %126 = ascscalar.bind_local_ptr %115 : memref<1xui16, 1>
              ascvec.get_buf mte3, %c2_i64, 0 {cannir.buffer.generated_sync}
              ascvec.copy_ub2gm %125[%c0], %126[%c0], %c1_i32, %c16384_i32, %c0_i64, %c16384_i32 l2_cache_ctl %c0_i32 : memref<1xui16>, memref<1xui16, 1>
              ascvec.release_buf mte3, %c2_i64, 0 {cannir.buffer.generated_sync}
            }
            %102 = arith.addi %97, %c1_i64 : i64
            %103 = arith.trunci %102 : i64 to i32
            %104 = arith.extsi %103 : i32 to i64
            %105 = arith.addi %50, %104 : i64
            %106 = arith.cmpi slt, %103, %53 : i32
            scf.if %106 {
              %112 = ascscalar.memref_address %alloc_4 : memref<262144xui8, 1>
              %113 = arith.addi %112, %c218112_i64 : i64
              %114 = ascscalar.bind_local_ptr %113 : memref<8192xi32, 1>
              %115 = arith.addi %112, %c98304_i64 : i64
              %116 = ascscalar.bind_local_ptr %115 : memref<8192xui16, 1>
              ascvec.get_buf v, %c1_i64, 0 {cannir.buffer.generated_sync}
              ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
              ascvec.vf(%114, %116 : memref<8192xi32, 1>, memref<8192xui16, 1>) {
              ^bb0(%arg51: memref<8192xi32, 1>, %arg52: memref<8192xui16, 1>):
                %c1_i32_5 = arith.constant 1 : i32
                %c128_i32_6 = arith.constant 128 : i32
                %c0_i32_7 = arith.constant 0 : i32
                %c-128_i16 = arith.constant -128 : i16
                %c-1_i32_8 = arith.constant -1 : i32
                %c8192_i32_9 = arith.constant 8192 : i32
                %c64_i32 = arith.constant 64 : i32
                scf.for %arg53 = %c0_i32_7 to %c128_i32_6 step %c1_i32_5  : i32 {
                  %127 = arith.muli %arg53, %c64_i32 overflow<nsw> : i32
                  %128 = arith.subi %c8192_i32_9, %127 overflow<nsw> : i32
                  %mask, %remaining_out = ascvec.update_mask %128 : !ascvec.mask
                  %mask_10, %remaining_out_11 = ascvec.update_mask %128 elem_bits 16 : !ascvec.mask
                  %mask_12, %remaining_out_13 = ascvec.update_mask %c64_i32 elem_bits 16 : !ascvec.mask
                  %mask_14, %remaining_out_15 = ascvec.update_mask %c64_i32 elem_bits 16 : !ascvec.mask
                  %129 = ascvec.mask_and %mask_10, %mask_12, %mask_14
                  %130 = ascvec.duplicate_scalar %c-1_i32_8, %mask : i32 -> <i32>
                  %131 = arith.index_cast %127 : i32 to index
                  ascvec.storealign %arg51[%131], %130, %mask : memref<8192xi32, 1>, index, <i32>
                  %132 = ascvec.duplicate_scalar %c-128_i16, %129 : i16 -> <ui16>
                  ascvec.storealign %arg52[%131], %132, %129 : memref<8192xui16, 1>, index, <ui16>
                } {cannir.loop_unroll = 1 : i64}
                ascvec.mem_bar vst_vld
              } {cannir.vf_mode = "simd"}
              ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
              ascvec.release_buf v, %c1_i64, 0 {cannir.buffer.generated_sync}
              %117 = arith.muli %105, %c262144_i64 : i64
              %118 = arith.divui %117, %c8_i64 : i64
              %119 = arith.addi %82, %118 : i64
              %120 = ascscalar.bind_gm_ptr %119 : memref<1xi32>
              %121 = ascscalar.bind_local_ptr %113 : memref<1xi32, 1>
              ascvec.get_buf mte3, %c1_i64, 0 {cannir.buffer.generated_sync}
              ascvec.copy_ub2gm %120[%c0], %121[%c0], %c1_i32, %c32768_i32, %c0_i64, %c32768_i32 l2_cache_ctl %c0_i32 : memref<1xi32>, memref<1xi32, 1>
              ascvec.release_buf mte3, %c1_i64, 0 {cannir.buffer.generated_sync}
              %122 = arith.muli %105, %c131072_i64 : i64
              %123 = arith.divui %122, %c8_i64 : i64
              %124 = arith.addi %83, %123 : i64
              %125 = ascscalar.bind_gm_ptr %124 : memref<1xui16>
              %126 = ascscalar.bind_local_ptr %115 : memref<1xui16, 1>
              ascvec.get_buf mte3, %c2_i64, 0 {cannir.buffer.generated_sync}
              ascvec.copy_ub2gm %125[%c0], %126[%c0], %c1_i32, %c16384_i32, %c0_i64, %c16384_i32 l2_cache_ctl %c0_i32 : memref<1xui16>, memref<1xui16, 1>
              ascvec.release_buf mte3, %c2_i64, 0 {cannir.buffer.generated_sync}
            }
            %107 = arith.addi %97, %c2_i64 : i64
            %108 = arith.trunci %107 : i64 to i32
            %109 = arith.extsi %108 : i32 to i64
            %110 = arith.addi %50, %109 : i64
            %111 = arith.cmpi slt, %108, %53 : i32
            scf.if %111 {
              %112 = ascscalar.memref_address %alloc_4 : memref<262144xui8, 1>
              %113 = arith.addi %112, %c218112_i64 : i64
              %114 = ascscalar.bind_local_ptr %113 : memref<8192xi32, 1>
              %115 = arith.addi %112, %c98304_i64 : i64
              %116 = ascscalar.bind_local_ptr %115 : memref<8192xui16, 1>
              ascvec.get_buf v, %c1_i64, 0 {cannir.buffer.generated_sync}
              ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
              ascvec.vf(%114, %116 : memref<8192xi32, 1>, memref<8192xui16, 1>) {
              ^bb0(%arg51: memref<8192xi32, 1>, %arg52: memref<8192xui16, 1>):
                %c1_i32_5 = arith.constant 1 : i32
                %c128_i32_6 = arith.constant 128 : i32
                %c0_i32_7 = arith.constant 0 : i32
                %c-128_i16 = arith.constant -128 : i16
                %c-1_i32_8 = arith.constant -1 : i32
                %c8192_i32_9 = arith.constant 8192 : i32
                %c64_i32 = arith.constant 64 : i32
                scf.for %arg53 = %c0_i32_7 to %c128_i32_6 step %c1_i32_5  : i32 {
                  %127 = arith.muli %arg53, %c64_i32 overflow<nsw> : i32
                  %128 = arith.subi %c8192_i32_9, %127 overflow<nsw> : i32
                  %mask, %remaining_out = ascvec.update_mask %128 : !ascvec.mask
                  %mask_10, %remaining_out_11 = ascvec.update_mask %128 elem_bits 16 : !ascvec.mask
                  %mask_12, %remaining_out_13 = ascvec.update_mask %c64_i32 elem_bits 16 : !ascvec.mask
                  %mask_14, %remaining_out_15 = ascvec.update_mask %c64_i32 elem_bits 16 : !ascvec.mask
                  %129 = ascvec.mask_and %mask_10, %mask_12, %mask_14
                  %130 = ascvec.duplicate_scalar %c-1_i32_8, %mask : i32 -> <i32>
                  %131 = arith.index_cast %127 : i32 to index
                  ascvec.storealign %arg51[%131], %130, %mask : memref<8192xi32, 1>, index, <i32>
                  %132 = ascvec.duplicate_scalar %c-128_i16, %129 : i16 -> <ui16>
                  ascvec.storealign %arg52[%131], %132, %129 : memref<8192xui16, 1>, index, <ui16>
                } {cannir.loop_unroll = 1 : i64}
                ascvec.mem_bar vst_vld
              } {cannir.vf_mode = "simd"}
              ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
              ascvec.release_buf v, %c1_i64, 0 {cannir.buffer.generated_sync}
              %117 = arith.muli %110, %c262144_i64 : i64
              %118 = arith.divui %117, %c8_i64 : i64
              %119 = arith.addi %82, %118 : i64
              %120 = ascscalar.bind_gm_ptr %119 : memref<1xi32>
              %121 = ascscalar.bind_local_ptr %113 : memref<1xi32, 1>
              ascvec.get_buf mte3, %c1_i64, 0 {cannir.buffer.generated_sync}
              ascvec.copy_ub2gm %120[%c0], %121[%c0], %c1_i32, %c32768_i32, %c0_i64, %c32768_i32 l2_cache_ctl %c0_i32 : memref<1xi32>, memref<1xi32, 1>
              ascvec.release_buf mte3, %c1_i64, 0 {cannir.buffer.generated_sync}
              %122 = arith.muli %110, %c131072_i64 : i64
              %123 = arith.divui %122, %c8_i64 : i64
              %124 = arith.addi %83, %123 : i64
              %125 = ascscalar.bind_gm_ptr %124 : memref<1xui16>
              %126 = ascscalar.bind_local_ptr %115 : memref<1xui16, 1>
              ascvec.get_buf mte3, %c2_i64, 0 {cannir.buffer.generated_sync}
              ascvec.copy_ub2gm %125[%c0], %126[%c0], %c1_i32, %c16384_i32, %c0_i64, %c16384_i32 l2_cache_ctl %c0_i32 : memref<1xui16>, memref<1xui16, 1>
              ascvec.release_buf mte3, %c2_i64, 0 {cannir.buffer.generated_sync}
            }
            scf.yield %arg46, %arg47, %arg48, %arg49, %arg50 : i64, i64, i64, i64, i64
          }
          %90 = arith.cmpi sgt, %59, %84 : i64
          %91 = arith.select %90, %87, %false : i1
          %92 = arith.select %91, %c0_i32, %arg44 : i32
          %93 = arith.cmpi eq, %70, %c0_i64 : i64
          %94 = arith.select %75, %93, %false : i1
          scf.if %94 {
            %97 = arith.muli %4, %c3_i64 : i64
            %98 = arith.trunci %97 : i64 to i32
            %99 = arith.cmpi sge, %98, %56 : i32
            %100 = arith.cmpi slt, %98, %53 : i32
            %101 = arith.select %99, %100, %false : i1
            scf.if %101 {
              %112 = arith.extsi %98 : i32 to i64
              %113 = arith.addi %50, %112 : i64
              %114 = ascscalar.memref_address %alloc_4 : memref<262144xui8, 1>
              %115 = arith.addi %114, %c218112_i64 : i64
              %116 = ascscalar.bind_local_ptr %115 : memref<8192xi32, 1>
              %117 = arith.addi %114, %c98304_i64 : i64
              %118 = ascscalar.bind_local_ptr %117 : memref<8192xui16, 1>
              ascvec.get_buf v, %c1_i64, 0 {cannir.buffer.generated_sync}
              ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
              ascvec.vf(%116, %118 : memref<8192xi32, 1>, memref<8192xui16, 1>) {
              ^bb0(%arg51: memref<8192xi32, 1>, %arg52: memref<8192xui16, 1>):
                %c1_i32_5 = arith.constant 1 : i32
                %c128_i32_6 = arith.constant 128 : i32
                %c0_i32_7 = arith.constant 0 : i32
                %c-128_i16 = arith.constant -128 : i16
                %c-1_i32_8 = arith.constant -1 : i32
                %c8192_i32_9 = arith.constant 8192 : i32
                %c64_i32 = arith.constant 64 : i32
                scf.for %arg53 = %c0_i32_7 to %c128_i32_6 step %c1_i32_5  : i32 {
                  %129 = arith.muli %arg53, %c64_i32 overflow<nsw> : i32
                  %130 = arith.subi %c8192_i32_9, %129 overflow<nsw> : i32
                  %mask, %remaining_out = ascvec.update_mask %130 : !ascvec.mask
                  %mask_10, %remaining_out_11 = ascvec.update_mask %130 elem_bits 16 : !ascvec.mask
                  %mask_12, %remaining_out_13 = ascvec.update_mask %c64_i32 elem_bits 16 : !ascvec.mask
                  %mask_14, %remaining_out_15 = ascvec.update_mask %c64_i32 elem_bits 16 : !ascvec.mask
                  %131 = ascvec.mask_and %mask_10, %mask_12, %mask_14
                  %132 = ascvec.duplicate_scalar %c-1_i32_8, %mask : i32 -> <i32>
                  %133 = arith.index_cast %129 : i32 to index
                  ascvec.storealign %arg51[%133], %132, %mask : memref<8192xi32, 1>, index, <i32>
                  %134 = ascvec.duplicate_scalar %c-128_i16, %131 : i16 -> <ui16>
                  ascvec.storealign %arg52[%133], %134, %131 : memref<8192xui16, 1>, index, <ui16>
                } {cannir.loop_unroll = 1 : i64}
                ascvec.mem_bar vst_vld
              } {cannir.vf_mode = "simd"}
              ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
              ascvec.release_buf v, %c1_i64, 0 {cannir.buffer.generated_sync}
              %119 = arith.muli %113, %c262144_i64 : i64
              %120 = arith.divui %119, %c8_i64 : i64
              %121 = arith.addi %arg28, %120 : i64
              %122 = ascscalar.bind_gm_ptr %121 : memref<1xi32>
              %123 = ascscalar.bind_local_ptr %115 : memref<1xi32, 1>
              ascvec.get_buf mte3, %c1_i64, 0 {cannir.buffer.generated_sync}
              ascvec.copy_ub2gm %122[%c0], %123[%c0], %c1_i32, %c32768_i32, %c0_i64, %c32768_i32 l2_cache_ctl %c0_i32 : memref<1xi32>, memref<1xi32, 1>
              ascvec.release_buf mte3, %c1_i64, 0 {cannir.buffer.generated_sync}
              %124 = arith.muli %113, %c131072_i64 : i64
              %125 = arith.divui %124, %c8_i64 : i64
              %126 = arith.addi %arg29, %125 : i64
              %127 = ascscalar.bind_gm_ptr %126 : memref<1xui16>
              %128 = ascscalar.bind_local_ptr %117 : memref<1xui16, 1>
              ascvec.get_buf mte3, %c2_i64, 0 {cannir.buffer.generated_sync}
              ascvec.copy_ub2gm %127[%c0], %128[%c0], %c1_i32, %c16384_i32, %c0_i64, %c16384_i32 l2_cache_ctl %c0_i32 : memref<1xui16>, memref<1xui16, 1>
              ascvec.release_buf mte3, %c2_i64, 0 {cannir.buffer.generated_sync}
            }
            %102 = arith.addi %97, %c1_i64 : i64
            %103 = arith.trunci %102 : i64 to i32
            %104 = arith.cmpi sge, %103, %56 : i32
            %105 = arith.cmpi slt, %103, %53 : i32
            %106 = arith.select %104, %105, %false : i1
            scf.if %106 {
              %112 = arith.extsi %103 : i32 to i64
              %113 = arith.addi %50, %112 : i64
              %114 = ascscalar.memref_address %alloc_4 : memref<262144xui8, 1>
              %115 = arith.addi %114, %c218112_i64 : i64
              %116 = ascscalar.bind_local_ptr %115 : memref<8192xi32, 1>
              %117 = arith.addi %114, %c98304_i64 : i64
              %118 = ascscalar.bind_local_ptr %117 : memref<8192xui16, 1>
              ascvec.get_buf v, %c1_i64, 0 {cannir.buffer.generated_sync}
              ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
              ascvec.vf(%116, %118 : memref<8192xi32, 1>, memref<8192xui16, 1>) {
              ^bb0(%arg51: memref<8192xi32, 1>, %arg52: memref<8192xui16, 1>):
                %c1_i32_5 = arith.constant 1 : i32
                %c128_i32_6 = arith.constant 128 : i32
                %c0_i32_7 = arith.constant 0 : i32
                %c-128_i16 = arith.constant -128 : i16
                %c-1_i32_8 = arith.constant -1 : i32
                %c8192_i32_9 = arith.constant 8192 : i32
                %c64_i32 = arith.constant 64 : i32
                scf.for %arg53 = %c0_i32_7 to %c128_i32_6 step %c1_i32_5  : i32 {
                  %129 = arith.muli %arg53, %c64_i32 overflow<nsw> : i32
                  %130 = arith.subi %c8192_i32_9, %129 overflow<nsw> : i32
                  %mask, %remaining_out = ascvec.update_mask %130 : !ascvec.mask
                  %mask_10, %remaining_out_11 = ascvec.update_mask %130 elem_bits 16 : !ascvec.mask
                  %mask_12, %remaining_out_13 = ascvec.update_mask %c64_i32 elem_bits 16 : !ascvec.mask
                  %mask_14, %remaining_out_15 = ascvec.update_mask %c64_i32 elem_bits 16 : !ascvec.mask
                  %131 = ascvec.mask_and %mask_10, %mask_12, %mask_14
                  %132 = ascvec.duplicate_scalar %c-1_i32_8, %mask : i32 -> <i32>
                  %133 = arith.index_cast %129 : i32 to index
                  ascvec.storealign %arg51[%133], %132, %mask : memref<8192xi32, 1>, index, <i32>
                  %134 = ascvec.duplicate_scalar %c-128_i16, %131 : i16 -> <ui16>
                  ascvec.storealign %arg52[%133], %134, %131 : memref<8192xui16, 1>, index, <ui16>
                } {cannir.loop_unroll = 1 : i64}
                ascvec.mem_bar vst_vld
              } {cannir.vf_mode = "simd"}
              ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
              ascvec.release_buf v, %c1_i64, 0 {cannir.buffer.generated_sync}
              %119 = arith.muli %113, %c262144_i64 : i64
              %120 = arith.divui %119, %c8_i64 : i64
              %121 = arith.addi %arg28, %120 : i64
              %122 = ascscalar.bind_gm_ptr %121 : memref<1xi32>
              %123 = ascscalar.bind_local_ptr %115 : memref<1xi32, 1>
              ascvec.get_buf mte3, %c1_i64, 0 {cannir.buffer.generated_sync}
              ascvec.copy_ub2gm %122[%c0], %123[%c0], %c1_i32, %c32768_i32, %c0_i64, %c32768_i32 l2_cache_ctl %c0_i32 : memref<1xi32>, memref<1xi32, 1>
              ascvec.release_buf mte3, %c1_i64, 0 {cannir.buffer.generated_sync}
              %124 = arith.muli %113, %c131072_i64 : i64
              %125 = arith.divui %124, %c8_i64 : i64
              %126 = arith.addi %arg29, %125 : i64
              %127 = ascscalar.bind_gm_ptr %126 : memref<1xui16>
              %128 = ascscalar.bind_local_ptr %117 : memref<1xui16, 1>
              ascvec.get_buf mte3, %c2_i64, 0 {cannir.buffer.generated_sync}
              ascvec.copy_ub2gm %127[%c0], %128[%c0], %c1_i32, %c16384_i32, %c0_i64, %c16384_i32 l2_cache_ctl %c0_i32 : memref<1xui16>, memref<1xui16, 1>
              ascvec.release_buf mte3, %c2_i64, 0 {cannir.buffer.generated_sync}
            }
            %107 = arith.addi %97, %c2_i64 : i64
            %108 = arith.trunci %107 : i64 to i32
            %109 = arith.cmpi sge, %108, %56 : i32
            %110 = arith.cmpi slt, %108, %53 : i32
            %111 = arith.select %109, %110, %false : i1
            scf.if %111 {
              %112 = arith.extsi %108 : i32 to i64
              %113 = arith.addi %50, %112 : i64
              %114 = ascscalar.memref_address %alloc_4 : memref<262144xui8, 1>
              %115 = arith.addi %114, %c218112_i64 : i64
              %116 = ascscalar.bind_local_ptr %115 : memref<8192xi32, 1>
              %117 = arith.addi %114, %c98304_i64 : i64
              %118 = ascscalar.bind_local_ptr %117 : memref<8192xui16, 1>
              ascvec.get_buf v, %c1_i64, 0 {cannir.buffer.generated_sync}
              ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
              ascvec.vf(%116, %118 : memref<8192xi32, 1>, memref<8192xui16, 1>) {
              ^bb0(%arg51: memref<8192xi32, 1>, %arg52: memref<8192xui16, 1>):
                %c1_i32_5 = arith.constant 1 : i32
                %c128_i32_6 = arith.constant 128 : i32
                %c0_i32_7 = arith.constant 0 : i32
                %c-128_i16 = arith.constant -128 : i16
                %c-1_i32_8 = arith.constant -1 : i32
                %c8192_i32_9 = arith.constant 8192 : i32
                %c64_i32 = arith.constant 64 : i32
                scf.for %arg53 = %c0_i32_7 to %c128_i32_6 step %c1_i32_5  : i32 {
                  %129 = arith.muli %arg53, %c64_i32 overflow<nsw> : i32
                  %130 = arith.subi %c8192_i32_9, %129 overflow<nsw> : i32
                  %mask, %remaining_out = ascvec.update_mask %130 : !ascvec.mask
                  %mask_10, %remaining_out_11 = ascvec.update_mask %130 elem_bits 16 : !ascvec.mask
                  %mask_12, %remaining_out_13 = ascvec.update_mask %c64_i32 elem_bits 16 : !ascvec.mask
                  %mask_14, %remaining_out_15 = ascvec.update_mask %c64_i32 elem_bits 16 : !ascvec.mask
                  %131 = ascvec.mask_and %mask_10, %mask_12, %mask_14
                  %132 = ascvec.duplicate_scalar %c-1_i32_8, %mask : i32 -> <i32>
                  %133 = arith.index_cast %129 : i32 to index
                  ascvec.storealign %arg51[%133], %132, %mask : memref<8192xi32, 1>, index, <i32>
                  %134 = ascvec.duplicate_scalar %c-128_i16, %131 : i16 -> <ui16>
                  ascvec.storealign %arg52[%133], %134, %131 : memref<8192xui16, 1>, index, <ui16>
                } {cannir.loop_unroll = 1 : i64}
                ascvec.mem_bar vst_vld
              } {cannir.vf_mode = "simd"}
              ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
              ascvec.release_buf v, %c1_i64, 0 {cannir.buffer.generated_sync}
              %119 = arith.muli %113, %c262144_i64 : i64
              %120 = arith.divui %119, %c8_i64 : i64
              %121 = arith.addi %arg28, %120 : i64
              %122 = ascscalar.bind_gm_ptr %121 : memref<1xi32>
              %123 = ascscalar.bind_local_ptr %115 : memref<1xi32, 1>
              ascvec.get_buf mte3, %c1_i64, 0 {cannir.buffer.generated_sync}
              ascvec.copy_ub2gm %122[%c0], %123[%c0], %c1_i32, %c32768_i32, %c0_i64, %c32768_i32 l2_cache_ctl %c0_i32 : memref<1xi32>, memref<1xi32, 1>
              ascvec.release_buf mte3, %c1_i64, 0 {cannir.buffer.generated_sync}
              %124 = arith.muli %113, %c131072_i64 : i64
              %125 = arith.divui %124, %c8_i64 : i64
              %126 = arith.addi %arg29, %125 : i64
              %127 = ascscalar.bind_gm_ptr %126 : memref<1xui16>
              %128 = ascscalar.bind_local_ptr %117 : memref<1xui16, 1>
              ascvec.get_buf mte3, %c2_i64, 0 {cannir.buffer.generated_sync}
              ascvec.copy_ub2gm %127[%c0], %128[%c0], %c1_i32, %c16384_i32, %c0_i64, %c16384_i32 l2_cache_ctl %c0_i32 : memref<1xui16>, memref<1xui16, 1>
              ascvec.release_buf mte3, %c2_i64, 0 {cannir.buffer.generated_sync}
            }
          }
          %95 = arith.addi %arg45, %c1_i64 : i64
          %96 = arith.select %75, %95, %arg45 : i64
          scf.yield %92, %96, %89#0, %89#1, %89#2, %89#3, %89#4 : i32, i64, i64, i64, i64, i64, i64
        } {cannir.loop_unroll = 1 : i64}
        scf.yield %44#0, %44#1, %44#2, %44#3, %44#4, %44#5, %44#6 : i32, i64, i64, i64, i64, i64, i64
      } {cannir.loop_unroll = 1 : i64}
    }
    scf.if %1 {
      ascvec.sync_all
      ascvec.sync_block_arrive mte3, %c0_i64
      ascvec.sync_block_arrive mte3, %c1_i64 {mode = 0 : i32}
      ascvec.sync_block_wait s, %c1_i64 {mode = 0 : i32}
      ascvec.sync_block_wait s, %c2_i64
      %13 = arith.muli %5, %c2_i64 : i64
      %14 = arith.addi %13, %c36_i64 : i64
      %15 = arith.addi %14, %4 : i64
      %16 = arith.index_cast %15 : i64 to index
      %17 = arith.muli %16, %c8 : index
      %18 = memref.load %alloc_3[%17] : memref<1024xi32>
      %19 = arith.cmpi ne, %18, %c0_i32 : i32
      scf.if %19 {
        %20 = arith.addi %17, %c1 : index
        %21 = memref.load %alloc_3[%20] : memref<1024xi32>
        %22 = arith.index_cast %21 : i32 to index
        %23 = memref.load %alloc_0[%22] : memref<1xi32>
        %24 = arith.extsi %23 : i32 to i64
        %25 = arith.addi %17, %c2 : index
        %26 = memref.load %alloc_3[%25] : memref<1024xi32>
        %27 = arith.extsi %26 : i32 to i64
        %28 = arith.muli %27, %c6_i64 : i64
        %29 = arith.addi %24, %28 : i64
        %30 = arith.addi %17, %c3 : index
        %31 = memref.load %alloc_3[%30] : memref<1024xi32>
        %32 = arith.extsi %31 : i32 to i64
        %33 = arith.addi %17, %c4 : index
        %34 = memref.load %alloc_3[%33] : memref<1024xi32>
        %35 = arith.extsi %34 : i32 to i64
        %36 = arith.addi %17, %c5 : index
        %37 = memref.load %alloc_3[%36] : memref<1024xi32>
        %38 = arith.extsi %37 : i32 to i64
        %39 = arith.addi %17, %c6 : index
        %40 = memref.load %alloc_3[%39] : memref<1024xi32>
        %41 = arith.extsi %40 : i32 to i64
        %42 = arith.addi %38, %41 : i64
        scf.for %arg35 = %38 to %42 step %c1_i64  : i64 {
          %43 = arith.addi %29, %arg35 : i64
          %44 = arith.cmpi ne, %arg34, %c0_i64 : i64
          %45 = scf.if %44 -> (i32) {
            %46 = ascscalar.bind_gm_ptr %arg34 : memref<1xi32>
            %47 = arith.index_cast %43 : i64 to index
            %48 = memref.load %46[%47] : memref<1xi32>
            scf.yield %48 : i32
          } else {
            scf.yield %c0_i32 : i32
          }
          scf.for %arg36 = %c0_i64 to %35 step %c1_i64  : i64 {
            %46 = arith.subi %35, %arg36 : i64
            %47 = arith.minsi %46, %c1_i64 : i64
            %48 = arith.cmpi sgt, %arg36, %c0_i64 : i64
            %49 = arith.extui %48 : i1 to i64
            %50 = ascscalar.bitcast_memref %alloc_4 byte_offset = %c0_i64 : memref<262144xui8, 1> -> memref<?xi32, 1>
            %51 = ascscalar.memref_address %alloc_4 : memref<262144xui8, 1>
            %52 = arith.addi %51, %c147456_i64 : i64
            %53 = ascscalar.bind_local_ptr %52 : memref<16384xi32, 1>
            %54 = ascscalar.bitcast_memref %alloc_4 byte_offset = %c0_i64 : memref<262144xui8, 1> -> memref<?xui16, 1>
            %55 = arith.addi %51, %c212992_i64 : i64
            %56 = ascscalar.bind_local_ptr %55 : memref<16384xui16, 1>
            scf.if %48 {
              %98 = arith.addi %51, %c98304_i64 : i64
              %99 = ascscalar.bind_local_ptr %98 : memref<8192xi32, 1>
              ascvec.get_buf v, %c8_i64, 0 {cannir.buffer.generated_sync}
              ascvec.copy_ub2ub %53[%c0], %99[%c0], %c32768_i32 : memref<16384xi32, 1>, memref<8192xi32, 1>
              ascvec.vf(%53, %99 : memref<16384xi32, 1>, memref<8192xi32, 1>) {
              ^bb0(%arg37: memref<16384xi32, 1>, %arg38: memref<8192xi32, 1>):
              } {cannir.vf_mode = "simd"}
              ascvec.release_buf v, %c8_i64, 0 {cannir.buffer.generated_sync}
              %100 = arith.addi %51, %c131072_i64 : i64
              %101 = ascscalar.bind_local_ptr %100 : memref<8192xui16, 1>
              ascvec.get_buf v, %c9_i64, 0 {cannir.buffer.generated_sync}
              ascvec.copy_ub2ub %56[%c0], %101[%c0], %c16384_i32 : memref<16384xui16, 1>, memref<8192xui16, 1>
              ascvec.vf(%56, %101 : memref<16384xui16, 1>, memref<8192xui16, 1>) {
              ^bb0(%arg37: memref<16384xui16, 1>, %arg38: memref<8192xui16, 1>):
              } {cannir.vf_mode = "simd"}
              ascvec.release_buf v, %c9_i64, 0 {cannir.buffer.generated_sync}
            }
            scf.for %arg37 = %c0_i64 to %47 step %c1_i64  : i64 {
              %98 = arith.addi %32, %arg36 : i64
              %99 = arith.addi %98, %arg37 : i64
              %100 = arith.muli %99, %c6_i64 : i64
              %101 = arith.addi %100, %arg35 : i64
              %102 = arith.addi %arg37, %49 : i64
              %103 = arith.minsi %101, %c384_i64 : i64
              %104 = arith.subi %c384_i64, %103 : i64
              %105 = arith.minsi %104, %c1_i64 : i64
              %106 = arith.cmpi sle, %101, %c384_i64 : i64
              %107 = arith.select %106, %105, %c0_i64 : i64
              %108 = arith.muli %101, %c32768_i64 : i64
              %109 = arith.muli %102, %c32768_i64 : i64
              %110 = arith.addi %109, %c147456_i64 : i64
              ascvec.get_buf mte2, %c8_i64, 0 {cannir.buffer.generated_sync}
              "ascvec.copy_gm2ub_plan"(%50, %2, %c0_i32, %c0_i32, %110, %108, %107, %c32768_i64, %c32768_i64, %c65536_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 12582912 : i64, pad_storage_bits = false, storage_dtype = i32, ub_base = 0 : i64, ub_capacity = 262144 : i64}> : (memref<?xi32, 1>, memref<3145728xi32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
              ascvec.release_buf mte2, %c8_i64, 0 {cannir.buffer.generated_sync}
              %111 = arith.muli %101, %c16384_i64 : i64
              %112 = arith.muli %102, %c16384_i64 : i64
              %113 = arith.addi %112, %c212992_i64 : i64
              ascvec.get_buf mte2, %c9_i64, 0 {cannir.buffer.generated_sync}
              "ascvec.copy_gm2ub_plan"(%54, %3, %c0_i32, %c0_i32, %113, %111, %107, %c16384_i64, %c16384_i64, %c32768_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 6291456 : i64, pad_storage_bits = false, storage_dtype = ui16, ub_base = 0 : i64, ub_capacity = 262144 : i64}> : (memref<?xui16, 1>, memref<3145728xui16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
              ascvec.release_buf mte2, %c9_i64, 0 {cannir.buffer.generated_sync}
            } {cannir.loop_unroll = 1 : i64}
            %57 = arith.addi %arg36, %47 : i64
            %58 = arith.cmpi eq, %57, %35 : i64
            %59 = arith.select %58, %45, %c0_i32 : i32
            %60 = arith.addi %47, %49 : i64
            %61 = arith.muli %60, %c8192_i64 : i64
            %62 = arith.select %58, %arg32, %c1_i64 : i64
            %63 = arith.addi %61, %c63_i64 : i64
            %64 = arith.remsi %63, %c64_i64 : i64
            %65 = arith.cmpi ne, %64, %c0_i64 : i64
            %66 = arith.cmpi slt, %64, %c0_i64 : i64
            %67 = arith.andi %65, %66 : i1
            %68 = arith.divsi %63, %c64_i64 : i64
            %69 = arith.extui %67 : i1 to i64
            %70 = arith.subi %68, %69 : i64
            ascvec.get_buf v, %c9_i64, 0 {cannir.buffer.generated_sync}
            ascvec.vf(%61, %56, %53, %54, %70 : i64, memref<16384xui16, 1>, memref<16384xi32, 1>, memref<?xui16, 1>, i64) {
            ^bb0(%arg37: i64, %arg38: memref<16384xui16, 1>, %arg39: memref<16384xi32, 1>, %arg40: memref<?xui16, 1>, %arg41: i64):
              %c1_i64_5 = arith.constant 1 : i64
              %c0_i64_6 = arith.constant 0 : i64
              %c1_i16 = arith.constant 1 : i16
              %c0_i16 = arith.constant 0 : i16
              %c0_i32_7 = arith.constant 0 : i32
              %c1_i32_8 = arith.constant 1 : i32
              %c-1_i16 = arith.constant -1 : i16
              %c-32768_i16 = arith.constant -32768 : i16
              %c64_i64_9 = arith.constant 64 : i64
              %c64_i32 = arith.constant 64 : i32
              %c0_10 = arith.constant 0 : index
              %c1_11 = arith.constant 1 : index
              scf.for %arg42 = %c0_10 to %c1_11 step %c1_11 {
                %mask, %remaining_out = ascvec.update_mask %c64_i32 elem_bits 16 : !ascvec.mask
                %mask_12, %remaining_out_13 = ascvec.update_mask %c64_i32 : !ascvec.mask
              }
              scf.for %arg42 = %c0_i64_6 to %arg41 step %c1_i64_5  : i64 {
                %98 = arith.muli %arg42, %c64_i64_9 : i64
                %99 = arith.subi %arg37, %98 : i64
                %100 = arith.minsi %99, %c64_i64_9 : i64
                %101 = arith.trunci %100 : i64 to i32
                %mask, %remaining_out = ascvec.update_mask %101 elem_bits 16 : !ascvec.mask
                %mask_12, %remaining_out_13 = ascvec.update_mask %101 : !ascvec.mask
                %102 = arith.index_cast %98 : i64 to index
                %103 = ascvec.loadalign %arg38[%102] : memref<16384xui16, 1>, index -> <ui16>
                %104 = ascvec.loadalign %arg39[%102] : memref<16384xi32, 1>, index -> <i32>
                %105 = ascvec.duplicate_scalar %c-32768_i16, %mask : i16 -> <ui16>
                %106 = ascvec.xor %103, %105, %mask : <ui16>, <ui16> -> <ui16>
                %107 = ascvec.duplicate_scalar %c-1_i16, %mask : i16 -> <ui16>
                %108 = ascvec.xor %103, %107, %mask : <ui16>, <ui16> -> <ui16>
                %109 = ascvec.and %103, %105, %mask : <ui16>, <ui16> -> <ui16>
                %110 = ascvec.cmp_eq_scalar %109, %c-32768_i16, %mask : <ui16>, i16 -> !ascvec.mask
                %111 = ascvec.select %108, %106, %110 : <ui16>
                %112 = ascvec.duplicate_scalar %c1_i32_8, %mask_12 : i32 -> <i32>
                %113 = ascvec.duplicate_scalar %c0_i32_7, %mask_12 : i32 -> <i32>
                %114 = ascvec.cmp_ge_scalar %104, %c0_i32_7, %mask_12 : <i32>, i32 -> !ascvec.mask
                %115 = ascvec.select %112, %113, %114 : <i32>
                %116 = ascvec.reinterpret %115 : <i32> -> <ui32>
                %117 = ascvec.pack %116 src_elem_bits 32 dst_elem_bits 16 {part = "low"} : <ui32> -> <ui16>
                %118 = ascvec.duplicate_scalar %c0_i16, %mask : i16 -> <ui16>
                %119 = ascvec.cmp_eq_scalar %117, %c1_i16, %mask : <ui16>, i16 -> !ascvec.mask
                %120 = ascvec.select %111, %118, %119 : <ui16>
                ascvec.storealign %arg40[%102], %120, %mask : memref<?xui16, 1>, index, <ui16>
              } {cannir.loop_unroll = 1 : i64}
              ascvec.mem_bar vst_vld
            } {cannir.vf_mode = "simd"}
            ascvec.release_buf v, %c9_i64, 0 {cannir.buffer.generated_sync}
            %71 = arith.remsi %61, %c1024_i64 : i64
            %72 = arith.cmpi ne, %71, %c0_i64 : i64
            %73 = arith.cmpi slt, %71, %c0_i64 : i64
            %74 = arith.andi %72, %73 : i1
            %75 = arith.divsi %61, %c1024_i64 : i64
            %76 = arith.extui %74 : i1 to i64
            %77 = arith.subi %75, %76 : i64
            %78 = arith.muli %77, %c4_i64 : i64
            %79 = arith.addi %61, %c255_i64 : i64
            %80 = arith.remsi %79, %c256_i64 : i64
            %81 = arith.cmpi ne, %80, %c0_i64 : i64
            %82 = arith.cmpi slt, %80, %c0_i64 : i64
            %83 = arith.andi %81, %82 : i1
            %84 = arith.divsi %79, %c256_i64 : i64
            %85 = arith.extui %83 : i1 to i64
            %86 = arith.subi %84, %85 : i64
            %87 = arith.addi %61, %c-8191_i64 : i64
            %88 = arith.trunci %87 : i64 to i16
            %89 = arith.addi %61, %c127_i64 : i64
            %90 = arith.remsi %89, %c128_i64 : i64
            %91 = arith.cmpi ne, %90, %c0_i64 : i64
            %92 = arith.cmpi slt, %90, %c0_i64 : i64
            %93 = arith.andi %91, %92 : i1
            %94 = arith.divsi %89, %c128_i64 : i64
            %95 = arith.extui %93 : i1 to i64
            %96 = arith.subi %94, %95 : i64
            ascvec.vf(%54, %77, %61, %78, %86, %88, %54, %96 : memref<?xui16, 1>, i64, i64, i64, i64, i16, memref<?xui16, 1>, i64) {
            ^bb0(%arg37: memref<?xui16, 1>, %arg38: i64, %arg39: i64, %arg40: i64, %arg41: i64, %arg42: i16, %arg43: memref<?xui16, 1>, %arg44: i64):
              %c128_i64_5 = arith.constant 128 : i64
              %c24576 = arith.constant 24576 : index
              %c127_i16 = arith.constant 127 : i16
              %c1_i16 = arith.constant 1 : i16
              %c128_i16 = arith.constant 128 : i16
              %c-1_i16 = arith.constant -1 : i16
              %c1_i32_6 = arith.constant 1 : i32
              %c1_i64_7 = arith.constant 1 : i64
              %c0_i64_8 = arith.constant 0 : i64
              %c768_i64 = arith.constant 768 : i64
              %c512_i64_9 = arith.constant 512 : i64
              %c256_i64_10 = arith.constant 256 : i64
              %c1024_i64_11 = arith.constant 1024 : i64
              %c256_i32_12 = arith.constant 256 : i32
              %c0_i16 = arith.constant 0 : i16
              %c128_i32_13 = arith.constant 128 : i32
              %mask, %remaining_out = ascvec.update_mask %c128_i32_13 elem_bits 16 : !ascvec.mask
              %98 = ascvec.duplicate_scalar %c0_i16, %mask : i16 -> <ui16>
              %mask_14, %remaining_out_15 = ascvec.update_mask %c256_i32_12 elem_bits 8 : !ascvec.mask
              %99:8 = scf.for %arg45 = %c0_i64_8 to %arg38 step %c1_i64_7 iter_args(%arg46 = %98, %arg47 = %98, %arg48 = %98, %arg49 = %98, %arg50 = %98, %arg51 = %98, %arg52 = %98, %arg53 = %98) -> (!ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>)  : i64 {
                %153 = arith.muli %arg45, %c1024_i64_11 : i64
                %154 = arith.index_cast %153 : i64 to index
                %result0, %result1 = ascvec.loadalign_deintlv %arg37[%154] {dintlv_width = "b8"} : memref<?xui16, 1>, index -> <ui8>, <ui8>
                %155 = ascvec.cumulative_histogram %arg46, %result1, %mask_14 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                %156 = ascvec.cumulative_histogram %arg47, %result1, %mask_14 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                %157 = arith.addi %153, %c256_i64_10 : i64
                %158 = arith.index_cast %157 : i64 to index
                %result0_34, %result1_35 = ascvec.loadalign_deintlv %arg37[%158] {dintlv_width = "b8"} : memref<?xui16, 1>, index -> <ui8>, <ui8>
                %159 = ascvec.cumulative_histogram %arg48, %result1_35, %mask_14 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                %160 = ascvec.cumulative_histogram %arg49, %result1_35, %mask_14 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                %161 = arith.addi %153, %c512_i64_9 : i64
                %162 = arith.index_cast %161 : i64 to index
                %result0_36, %result1_37 = ascvec.loadalign_deintlv %arg37[%162] {dintlv_width = "b8"} : memref<?xui16, 1>, index -> <ui8>, <ui8>
                %163 = ascvec.cumulative_histogram %arg50, %result1_37, %mask_14 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                %164 = ascvec.cumulative_histogram %arg51, %result1_37, %mask_14 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                %165 = arith.addi %153, %c768_i64 : i64
                %166 = arith.index_cast %165 : i64 to index
                %result0_38, %result1_39 = ascvec.loadalign_deintlv %arg37[%166] {dintlv_width = "b8"} : memref<?xui16, 1>, index -> <ui8>, <ui8>
                %167 = ascvec.cumulative_histogram %arg52, %result1_39, %mask_14 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                %168 = ascvec.cumulative_histogram %arg53, %result1_39, %mask_14 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                scf.yield %155, %156, %159, %160, %163, %164, %167, %168 : !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>
              } {cannir.loop_unroll = 1 : i64}
              %100:2 = scf.for %arg45 = %arg40 to %arg41 step %c1_i64_7 iter_args(%arg46 = %99#0, %arg47 = %99#1) -> (!ascvec.vreg<ui16>, !ascvec.vreg<ui16>)  : i64 {
                %153 = arith.muli %arg45, %c256_i64_10 : i64
                %154 = arith.subi %arg39, %153 : i64
                %155 = arith.trunci %154 : i64 to i32
                %mask_34, %remaining_out_35 = ascvec.update_mask %155 elem_bits 8 : !ascvec.mask
                %156 = arith.index_cast %153 : i64 to index
                %result0, %result1 = ascvec.loadalign_deintlv %arg37[%156] {dintlv_width = "b8"} : memref<?xui16, 1>, index -> <ui8>, <ui8>
                %157 = ascvec.cumulative_histogram %arg46, %result1, %mask_34 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                %158 = ascvec.cumulative_histogram %arg47, %result1, %mask_34 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                scf.yield %157, %158 : !ascvec.vreg<ui16>, !ascvec.vreg<ui16>
              } {cannir.loop_unroll = 1 : i64}
              %101 = ascvec.add %100#0, %99#2, %mask : <ui16>
              %102 = ascvec.add %101, %99#4, %mask : <ui16>
              %103 = ascvec.add %102, %99#6, %mask : <ui16>
              %104 = ascvec.add %100#1, %99#3, %mask : <ui16>
              %105 = ascvec.add %104, %99#5, %mask : <ui16>
              %106 = ascvec.add %105, %99#7, %mask : <ui16>
              %mask_16, %remaining_out_17 = ascvec.update_mask %c128_i32_13 elem_bits 16 : !ascvec.mask
              %mask_18, %remaining_out_19 = ascvec.update_mask %c1_i32_6 elem_bits 16 : !ascvec.mask
              %107 = ascvec.duplicate_scalar %c-1_i16, %mask_16 : i16 -> <ui16>
              %108 = ascvec.arange %c0_i16 : i16 -> <ui16>
              %109 = ascvec.cmp_ge_scalar %103, %arg42, %mask_16 : <ui16>, i16 -> !ascvec.mask
              %110 = ascvec.select %108, %107, %109 : <ui16>
              %111 = ascvec.arange %c128_i16 : i16 -> <ui16>
              %112 = ascvec.cmp_ge_scalar %106, %arg42, %mask_16 : <ui16>, i16 -> !ascvec.mask
              %113 = ascvec.select %111, %107, %112 : <ui16>
              %114 = ascvec.min %110, %113, %mask_16 : <ui16>
              %115 = ascvec.reduce_min %114, %mask_16 : <ui16>
              %mask_20, %remaining_out_21 = ascvec.update_mask %c128_i32_13 elem_bits 16 : !ascvec.mask
              %mask_22, %remaining_out_23 = ascvec.update_mask %c1_i32_6 elem_bits 16 : !ascvec.mask
              %116 = ascvec.duplicate_scalar %c0_i16, %mask_20 : i16 -> <ui16>
              %117 = ascvec.duplicate_scalar %c1_i16, %mask_20 : i16 -> <ui16>
              %118 = ascvec.duplicate %115, %mask_20 : <ui16> -> <ui16>
              %119 = ascvec.shl %118, 8, %mask_20 : <ui16> -> <ui16>
              %120 = ascvec.or %118, %119, %mask_20 : <ui16>, <ui16> -> <ui16>
              %121 = ascvec.reinterpret %120 : <ui16> -> <ui8>
              %122 = ascvec.sub %118, %117, %mask_20 : <ui16>
              %123 = ascvec.cmp_eq_scalar %118, %c0_i16, %mask_20 : <ui16>, i16 -> !ascvec.mask
              %124 = ascvec.select %116, %122, %123 : <ui16>
              %125 = ascvec.duplicate_scalar %c127_i16, %mask_20 : i16 -> <ui16>
              %126 = ascvec.and %124, %125, %mask_20 : <ui16>, <ui16> -> <ui16>
              %127 = ascvec.gather_reg %103, %126 : <ui16>, <ui16> -> <ui16>
              %128 = ascvec.gather_reg %106, %126 : <ui16>, <ui16> -> <ui16>
              %129 = ascvec.cmp_lt_scalar %124, %c128_i16, %mask_20 : <ui16>, i16 -> !ascvec.mask
              %130 = ascvec.select %127, %128, %129 : <ui16>
              %131 = ascvec.select %116, %130, %123 : <ui16>
              %132 = ascvec.duplicate_scalar %arg42, %mask_20 : i16 -> <ui16>
              %133 = ascvec.sub %132, %131, %mask_20 : <ui16>
              %134 = ascvec.duplicate %133, %mask_20 : <ui16> -> <ui16>
              %mask_24, %remaining_out_25 = ascvec.update_mask %c256_i32_12 elem_bits 8 : !ascvec.mask
              %135:8 = scf.for %arg45 = %c0_i64_8 to %arg38 step %c1_i64_7 iter_args(%arg46 = %116, %arg47 = %116, %arg48 = %116, %arg49 = %116, %arg50 = %116, %arg51 = %116, %arg52 = %116, %arg53 = %116) -> (!ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>)  : i64 {
                %153 = arith.muli %arg45, %c1024_i64_11 : i64
                %154 = arith.index_cast %153 : i64 to index
                %result0, %result1 = ascvec.loadalign_deintlv %arg37[%154] {dintlv_width = "b8"} : memref<?xui16, 1>, index -> <ui8>, <ui8>
                %155 = ascvec.cmp_eq %result1, %121, %mask_24 : <ui8>, <ui8> -> !ascvec.mask
                %156 = ascvec.cumulative_histogram %arg46, %result0, %155 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                %157 = ascvec.cumulative_histogram %arg47, %result0, %155 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                %158 = arith.addi %153, %c256_i64_10 : i64
                %159 = arith.index_cast %158 : i64 to index
                %result0_34, %result1_35 = ascvec.loadalign_deintlv %arg37[%159] {dintlv_width = "b8"} : memref<?xui16, 1>, index -> <ui8>, <ui8>
                %160 = ascvec.cmp_eq %result1_35, %121, %mask_24 : <ui8>, <ui8> -> !ascvec.mask
                %161 = ascvec.cumulative_histogram %arg48, %result0_34, %160 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                %162 = ascvec.cumulative_histogram %arg49, %result0_34, %160 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                %163 = arith.addi %153, %c512_i64_9 : i64
                %164 = arith.index_cast %163 : i64 to index
                %result0_36, %result1_37 = ascvec.loadalign_deintlv %arg37[%164] {dintlv_width = "b8"} : memref<?xui16, 1>, index -> <ui8>, <ui8>
                %165 = ascvec.cmp_eq %result1_37, %121, %mask_24 : <ui8>, <ui8> -> !ascvec.mask
                %166 = ascvec.cumulative_histogram %arg50, %result0_36, %165 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                %167 = ascvec.cumulative_histogram %arg51, %result0_36, %165 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                %168 = arith.addi %153, %c768_i64 : i64
                %169 = arith.index_cast %168 : i64 to index
                %result0_38, %result1_39 = ascvec.loadalign_deintlv %arg37[%169] {dintlv_width = "b8"} : memref<?xui16, 1>, index -> <ui8>, <ui8>
                %170 = ascvec.cmp_eq %result1_39, %121, %mask_24 : <ui8>, <ui8> -> !ascvec.mask
                %171 = ascvec.cumulative_histogram %arg52, %result0_38, %170 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                %172 = ascvec.cumulative_histogram %arg53, %result0_38, %170 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                scf.yield %156, %157, %161, %162, %166, %167, %171, %172 : !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>, !ascvec.vreg<ui16>
              } {cannir.loop_unroll = 1 : i64}
              %136:2 = scf.for %arg45 = %arg40 to %arg41 step %c1_i64_7 iter_args(%arg46 = %135#0, %arg47 = %135#1) -> (!ascvec.vreg<ui16>, !ascvec.vreg<ui16>)  : i64 {
                %153 = arith.muli %arg45, %c256_i64_10 : i64
                %154 = arith.subi %arg39, %153 : i64
                %155 = arith.trunci %154 : i64 to i32
                %mask_34, %remaining_out_35 = ascvec.update_mask %155 elem_bits 8 : !ascvec.mask
                %156 = arith.index_cast %153 : i64 to index
                %result0, %result1 = ascvec.loadalign_deintlv %arg37[%156] {dintlv_width = "b8"} : memref<?xui16, 1>, index -> <ui8>, <ui8>
                %157 = ascvec.cmp_eq %result1, %121, %mask_34 : <ui8>, <ui8> -> !ascvec.mask
                %158 = ascvec.cumulative_histogram %arg46, %result0, %157 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                %159 = ascvec.cumulative_histogram %arg47, %result0, %157 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                scf.yield %158, %159 : !ascvec.vreg<ui16>, !ascvec.vreg<ui16>
              } {cannir.loop_unroll = 1 : i64}
              %137 = ascvec.add %136#0, %135#2, %mask_20 : <ui16>
              %138 = ascvec.add %137, %135#4, %mask_20 : <ui16>
              %139 = ascvec.add %138, %135#6, %mask_20 : <ui16>
              %140 = ascvec.add %136#1, %135#3, %mask_20 : <ui16>
              %141 = ascvec.add %140, %135#5, %mask_20 : <ui16>
              %142 = ascvec.add %141, %135#7, %mask_20 : <ui16>
              %mask_26, %remaining_out_27 = ascvec.update_mask %c128_i32_13 elem_bits 16 : !ascvec.mask
              %mask_28, %remaining_out_29 = ascvec.update_mask %c1_i32_6 elem_bits 16 : !ascvec.mask
              %143 = ascvec.duplicate_scalar %c-1_i16, %mask_26 : i16 -> <ui16>
              %144 = ascvec.cmp_ge %139, %134, %mask_26 : <ui16>, <ui16> -> !ascvec.mask
              %145 = ascvec.select %108, %143, %144 : <ui16>
              %146 = ascvec.cmp_ge %142, %134, %mask_26 : <ui16>, <ui16> -> !ascvec.mask
              %147 = ascvec.select %111, %143, %146 : <ui16>
              %148 = ascvec.min %145, %147, %mask_26 : <ui16>
              %149 = ascvec.reduce_min %148, %mask_26 : <ui16>
              %mask_30, %remaining_out_31 = ascvec.update_mask %c128_i32_13 elem_bits 16 : !ascvec.mask
              %150 = ascvec.duplicate %149, %mask_30 : <ui16> -> <ui16>
              %151 = ascvec.shl %118, 8, %mask_30 : <ui16> -> <ui16>
              %152 = ascvec.or %151, %150, %mask_30 : <ui16>, <ui16> -> <ui16>
              ascvec.store_unalign_begin %arg43 : memref<?xui16, 1>
              scf.for %arg45 = %c0_i64_8 to %arg44 step %c1_i64_7  : i64 {
                %153 = arith.muli %arg45, %c128_i64_5 : i64
                %154 = arith.subi %arg39, %153 : i64
                %155 = arith.trunci %154 : i64 to i32
                %mask_34, %remaining_out_35 = ascvec.update_mask %155 elem_bits 16 : !ascvec.mask
                %156 = arith.index_cast %153 : i64 to index
                %157 = ascvec.loadalign %arg37[%156] : memref<?xui16, 1>, index -> <ui16>
                %158 = arith.trunci %153 : i64 to i16
                %159 = ascvec.arange %158 : i16 -> <ui16>
                %160 = ascvec.cmp_gt %157, %152, %mask_34 : <ui16>, <ui16> -> !ascvec.mask
                %161 = ascvec.squeeze_store_reg %159, %160 : <ui16>, !ascvec.mask -> <ui16>
                ascvec.store_unalign %arg43[%c24576], %161 : memref<?xui16, 1>, <ui16>
              } {cannir.loop_unroll = 1 : i64}
              scf.for %arg45 = %c0_i64_8 to %arg44 step %c1_i64_7  : i64 {
                %153 = arith.muli %arg45, %c128_i64_5 : i64
                %154 = arith.subi %arg39, %153 : i64
                %155 = arith.trunci %154 : i64 to i32
                %mask_34, %remaining_out_35 = ascvec.update_mask %155 elem_bits 16 : !ascvec.mask
                %156 = arith.index_cast %153 : i64 to index
                %157 = ascvec.loadalign %arg37[%156] : memref<?xui16, 1>, index -> <ui16>
                %158 = arith.trunci %153 : i64 to i16
                %159 = ascvec.arange %158 : i16 -> <ui16>
                %160 = ascvec.cmp_eq %157, %152, %mask_34 : <ui16>, <ui16> -> !ascvec.mask
                %161 = ascvec.squeeze_store_reg %159, %160 : <ui16>, !ascvec.mask -> <ui16>
                ascvec.store_unalign %arg43[%c24576], %161 : memref<?xui16, 1>, <ui16>
              } {cannir.loop_unroll = 1 : i64}
              %c0_32 = arith.constant 0 : index
              %c1_33 = arith.constant 1 : index
              scf.for %arg45 = %c0_32 to %c1_33 step %c1_33 {
                ascvec.store_unalign_post %arg43[%c24576] : memref<?xui16, 1>
              }
              ascvec.mem_bar vst_vld
            } {cannir.vf_mode = "simd"}
            %97 = arith.cmpi ne, %62, %c0_i64 : i64
            scf.if %97 {
              %98 = arith.addi %51, %c98304_i64 : i64
              %99 = ascscalar.bind_local_ptr %98 : memref<8192xi32, 1>
              %100 = arith.addi %51, %c131072_i64 : i64
              %101 = ascscalar.bind_local_ptr %100 : memref<8192xui16, 1>
              ascvec.get_buf v, %c8_i64, 0 {cannir.buffer.generated_sync}
              ascvec.get_buf v, %c6_i64, 0 {cannir.buffer.generated_sync}
              ascvec.get_buf v, %c7_i64, 0 {cannir.buffer.generated_sync}
              ascvec.vf(%54, %53, %54, %56, %59, %99, %101 : memref<?xui16, 1>, memref<16384xi32, 1>, memref<?xui16, 1>, memref<16384xui16, 1>, i32, memref<8192xi32, 1>, memref<8192xui16, 1>) {
              ^bb0(%arg37: memref<?xui16, 1>, %arg38: memref<16384xi32, 1>, %arg39: memref<?xui16, 1>, %arg40: memref<16384xui16, 1>, %arg41: i32, %arg42: memref<8192xi32, 1>, %arg43: memref<8192xui16, 1>):
                %c128_i32_5 = arith.constant 128 : i32
                %c-128_i16 = arith.constant -128 : i16
                %c0_i32_6 = arith.constant 0 : i32
                %c1_i32_7 = arith.constant 1 : i32
                %c-1_i32_8 = arith.constant -1 : i32
                %c0_i16 = arith.constant 0 : i16
                %c1_i16 = arith.constant 1 : i16
                %c0_9 = arith.constant 0 : index
                %c24576 = arith.constant 24576 : index
                %c8192_i32_10 = arith.constant 8192 : i32
                %c64_i32 = arith.constant 64 : i32
                scf.for %arg44 = %c0_i32_6 to %c128_i32_5 step %c1_i32_7  : i32 {
                  %102 = arith.muli %arg44, %c64_i32 overflow<nsw> : i32
                  %103 = arith.subi %c8192_i32_10, %102 overflow<nsw> : i32
                  %mask, %remaining_out = ascvec.update_mask %103 : !ascvec.mask
                  %mask_11, %remaining_out_12 = ascvec.update_mask %103 elem_bits 16 : !ascvec.mask
                  %mask_13, %remaining_out_14 = ascvec.update_mask %c64_i32 elem_bits 16 : !ascvec.mask
                  %mask_15, %remaining_out_16 = ascvec.update_mask %c64_i32 elem_bits 16 : !ascvec.mask
                  %104 = ascvec.mask_and %mask_11, %mask_13, %mask_15
                  %105 = arith.index_cast %102 : i32 to index
                  %106 = arith.addi %105, %c24576 : index
                  %107 = ascvec.loadalign %arg37[%106] : memref<?xui16, 1>, index -> <ui16>
                  %108 = ascvec.unpack %107 {part = "lower"} : <ui16> -> <ui32>
                  %109 = ascvec.reinterpret %108 : <ui32> -> <ui32>
                  %110 = ascvec.gather %arg38[%c0_9], %109, %mask : memref<16384xi32, 1>, <ui32>, !ascvec.mask -> <i32>
                  %111 = ascvec.gather %arg39[%c0_9], %107, %104 : memref<?xui16, 1>, <ui16>, !ascvec.mask -> <ui16>
                  %112 = ascvec.gather %arg40[%c0_9], %107, %104 : memref<16384xui16, 1>, <ui16>, !ascvec.mask -> <ui16>
                  %113 = ascvec.duplicate_scalar %c1_i16, %104 : i16 -> <ui16>
                  %114 = ascvec.duplicate_scalar %c0_i16, %104 : i16 -> <ui16>
                  %115 = ascvec.cmp_ge_scalar %111, %c1_i16, %104 : <ui16>, i16 -> !ascvec.mask
                  %116 = ascvec.select %113, %114, %115 : <ui16>
                  %117 = ascvec.unpack %116 {part = "lower"} : <ui16> -> <ui32>
                  %118 = ascvec.duplicate_scalar %c-1_i32_8, %mask : i32 -> <i32>
                  %119 = ascvec.cmp_eq_scalar %117, %c1_i32_7, %mask : <ui32>, i32 -> !ascvec.mask
                  %120 = ascvec.select %110, %118, %119 : <i32>
                  %121 = ascvec.duplicate_scalar %c1_i32_7, %mask : i32 -> <i32>
                  %122 = ascvec.duplicate_scalar %c0_i32_6, %mask : i32 -> <i32>
                  %123 = ascvec.cmp_ge_scalar %120, %c0_i32_6, %mask : <i32>, i32 -> !ascvec.mask
                  %124 = ascvec.select %121, %122, %123 : <i32>
                  %125 = ascvec.reinterpret %124 : <i32> -> <ui32>
                  %126 = ascvec.pack %125 src_elem_bits 32 dst_elem_bits 16 {part = "low"} : <ui32> -> <ui16>
                  %127 = ascvec.select %120, %118, %123 : <i32>
                  %128 = ascvec.duplicate_scalar %c-128_i16, %104 : i16 -> <ui16>
                  %129 = ascvec.cmp_eq_scalar %126, %c1_i16, %104 : <ui16>, i16 -> !ascvec.mask
                  %130 = ascvec.select %112, %128, %129 : <ui16>
                  %131 = ascvec.add_scalar %127, %arg41, %mask : <i32>, i32
                  %132 = ascvec.cmp_eq_scalar %124, %c1_i32_7, %mask : <i32>, i32 -> !ascvec.mask
                  %133 = ascvec.select %131, %118, %132 : <i32>
                  ascvec.storealign %arg42[%105], %133, %mask : memref<8192xi32, 1>, index, <i32>
                  ascvec.storealign %arg43[%105], %130, %104 : memref<8192xui16, 1>, index, <ui16>
                } {cannir.loop_unroll = 1 : i64}
                ascvec.mem_bar vst_vld
              } {cannir.vf_mode = "simd"}
              ascvec.release_buf v, %c7_i64, 0 {cannir.buffer.generated_sync}
              ascvec.release_buf v, %c6_i64, 0 {cannir.buffer.generated_sync}
              ascvec.release_buf v, %c8_i64, 0 {cannir.buffer.generated_sync}
              scf.if %58 {
                %102 = arith.muli %43, %c262144_i64 : i64
                %103 = arith.divui %102, %c8_i64 : i64
                %104 = arith.addi %arg28, %103 : i64
                %105 = ascscalar.bind_gm_ptr %104 : memref<1xi32>
                %106 = ascscalar.bind_local_ptr %98 : memref<1xi32, 1>
                ascvec.get_buf mte3, %c6_i64, 0 {cannir.buffer.generated_sync}
                ascvec.copy_ub2gm %105[%c0], %106[%c0], %c1_i32, %c32768_i32, %c0_i64, %c32768_i32 l2_cache_ctl %c0_i32 : memref<1xi32>, memref<1xi32, 1>
                ascvec.release_buf mte3, %c6_i64, 0 {cannir.buffer.generated_sync}
                %107 = arith.muli %43, %c131072_i64 : i64
                %108 = arith.divui %107, %c8_i64 : i64
                %109 = arith.addi %arg29, %108 : i64
                %110 = ascscalar.bind_gm_ptr %109 : memref<1xui16>
                %111 = ascscalar.bind_local_ptr %100 : memref<1xui16, 1>
                ascvec.get_buf mte3, %c7_i64, 0 {cannir.buffer.generated_sync}
                ascvec.copy_ub2gm %110[%c0], %111[%c0], %c1_i32, %c16384_i32, %c0_i64, %c16384_i32 l2_cache_ctl %c0_i32 : memref<1xui16>, memref<1xui16, 1>
                ascvec.release_buf mte3, %c7_i64, 0 {cannir.buffer.generated_sync}
              }
            } else {
              %98 = arith.addi %51, %c98304_i64 : i64
              %99 = ascscalar.bind_local_ptr %98 : memref<8192xi32, 1>
              ascvec.get_buf v, %c8_i64, 0 {cannir.buffer.generated_sync}
              ascvec.get_buf v, %c6_i64, 0 {cannir.buffer.generated_sync}
              ascvec.vf(%54, %53, %54, %59, %99 : memref<?xui16, 1>, memref<16384xi32, 1>, memref<?xui16, 1>, i32, memref<8192xi32, 1>) {
              ^bb0(%arg37: memref<?xui16, 1>, %arg38: memref<16384xi32, 1>, %arg39: memref<?xui16, 1>, %arg40: i32, %arg41: memref<8192xi32, 1>):
                %c128_i32_5 = arith.constant 128 : i32
                %c0_i32_6 = arith.constant 0 : i32
                %c1_i32_7 = arith.constant 1 : i32
                %c-1_i32_8 = arith.constant -1 : i32
                %c0_i16 = arith.constant 0 : i16
                %c1_i16 = arith.constant 1 : i16
                %c0_9 = arith.constant 0 : index
                %c24576 = arith.constant 24576 : index
                %c8192_i32_10 = arith.constant 8192 : i32
                %c64_i32 = arith.constant 64 : i32
                scf.for %arg42 = %c0_i32_6 to %c128_i32_5 step %c1_i32_7  : i32 {
                  %100 = arith.muli %arg42, %c64_i32 overflow<nsw> : i32
                  %101 = arith.subi %c8192_i32_10, %100 overflow<nsw> : i32
                  %mask, %remaining_out = ascvec.update_mask %101 : !ascvec.mask
                  %mask_11, %remaining_out_12 = ascvec.update_mask %101 elem_bits 16 : !ascvec.mask
                  %mask_13, %remaining_out_14 = ascvec.update_mask %c64_i32 elem_bits 16 : !ascvec.mask
                  %mask_15, %remaining_out_16 = ascvec.update_mask %c64_i32 elem_bits 16 : !ascvec.mask
                  %102 = ascvec.mask_and %mask_11, %mask_13, %mask_15
                  %103 = arith.index_cast %100 : i32 to index
                  %104 = arith.addi %103, %c24576 : index
                  %105 = ascvec.loadalign %arg37[%104] : memref<?xui16, 1>, index -> <ui16>
                  %106 = ascvec.unpack %105 {part = "lower"} : <ui16> -> <ui32>
                  %107 = ascvec.gather %arg38[%c0_9], %106, %mask : memref<16384xi32, 1>, <ui32>, !ascvec.mask -> <i32>
                  %108 = ascvec.gather %arg39[%c0_9], %105, %102 : memref<?xui16, 1>, <ui16>, !ascvec.mask -> <ui16>
                  %109 = ascvec.duplicate_scalar %c1_i16, %102 : i16 -> <ui16>
                  %110 = ascvec.duplicate_scalar %c0_i16, %102 : i16 -> <ui16>
                  %111 = ascvec.cmp_ge_scalar %108, %c1_i16, %102 : <ui16>, i16 -> !ascvec.mask
                  %112 = ascvec.select %109, %110, %111 : <ui16>
                  %113 = ascvec.unpack %112 {part = "lower"} : <ui16> -> <ui32>
                  %114 = ascvec.add_scalar %107, %arg40, %mask : <i32>, i32
                  %115 = ascvec.duplicate_scalar %c-1_i32_8, %mask : i32 -> <i32>
                  %116 = ascvec.cmp_eq_scalar %113, %c1_i32_7, %mask : <ui32>, i32 -> !ascvec.mask
                  %117 = ascvec.select %114, %115, %116 : <i32>
                  ascvec.storealign %arg41[%103], %117, %mask : memref<8192xi32, 1>, index, <i32>
                } {cannir.loop_unroll = 1 : i64}
                ascvec.mem_bar vst_vld
              } {cannir.vf_mode = "simd"}
              ascvec.release_buf v, %c6_i64, 0 {cannir.buffer.generated_sync}
              ascvec.release_buf v, %c8_i64, 0 {cannir.buffer.generated_sync}
              scf.if %58 {
                %100 = arith.muli %43, %c262144_i64 : i64
                %101 = arith.divui %100, %c8_i64 : i64
                %102 = arith.addi %arg28, %101 : i64
                %103 = ascscalar.bind_gm_ptr %102 : memref<1xi32>
                %104 = ascscalar.bind_local_ptr %98 : memref<1xi32, 1>
                ascvec.get_buf mte3, %c6_i64, 0 {cannir.buffer.generated_sync}
                ascvec.copy_ub2gm %103[%c0], %104[%c0], %c1_i32, %c32768_i32, %c0_i64, %c32768_i32 l2_cache_ctl %c0_i32 : memref<1xi32>, memref<1xi32, 1>
                ascvec.release_buf mte3, %c6_i64, 0 {cannir.buffer.generated_sync}
              }
            }
          } {cannir.loop_unroll = 1 : i64}
        } {cannir.loop_unroll = 1 : i64}
      }
    }
    return
  }
  func.func @kernel_Quant_lightning_indexer_0_aic(%arg0: i64, %arg1: i64, %arg2: !cannir.tensor<<gm, f32>, <<(-1, 32)>, <(32, 1)>>, <<(-1, 32)>, <(32, 1)>>, identity>, %arg3: !cannir.tensor<<gm, !ascdtype.fp8_e8m0>, <<(-1, 2, 2)>, <(4, 2, 1)>>, <<(-1, 2, 2)>, <(4, 2, 1)>>, identity>, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg8: !cannir.tensor<<gm, i32>, <<(1)>, <(1)>>, <<(1)>, <(1)>>, identity>, %arg9: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg10: !cannir.tensor<<gm, i32>, <<(1)>, <(1)>>, <<(1)>, <(1)>>, identity>, %arg11: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg12: !cannir.tensor<<gm, ui16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg13: !cannir.tensor<<gm, ui16>, <<(-1, 1)>, <(1, 1)>>, <<(-1, 1)>, <(1, 1)>>, identity>, %arg14: i64, %arg15: i64, %arg16: i64, %arg17: i64, %arg18: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg19: i64, %arg20: i64, %arg21: i64, %arg22: i64, %arg23: i64, %arg24: i64, %arg25: i64, %arg26: !cannir.tensor<<gm, i32>, <<(128, 8)>, <(8, 1)>>, <<(128, 8)>, <(8, 1)>>, identity>, %arg27: i64, %arg28: i64, %arg29: i64, %arg30: i64, %arg31: i64, %arg32: i64, %arg33: i64, %arg34: i64) attributes {cannir.buffer.auto_sync, cannir.buffer.sync_analyzed, cannir.buffer.sync_verified, cannir.buffer.tensor_state, cannir.core = "aic", cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.device, cannir.explicit_ring_buffer_cursors, cannir.kernel_group = "kernel_Quant_lightning_indexer_0", cannir.resource.ids_prepared} {
    %c24704 = arith.constant 24704 : index
    %c24576 = arith.constant 24576 : index
    %c96_i32 = arith.constant 96 : i32
    %c16_i32 = arith.constant 16 : i32
    %c12_i32 = arith.constant 12 : i32
    %c6_i32 = arith.constant 6 : i32
    %c192_i32 = arith.constant 192 : i32
    %c3 = arith.constant 3 : index
    %c6 = arith.constant 6 : index
    %c5 = arith.constant 5 : index
    %c1 = arith.constant 1 : index
    %c7 = arith.constant 7 : index
    %c8 = arith.constant 8 : index
    %c288 = arith.constant 288 : index
    %c0_i32 = arith.constant 0 : i32
    %c-1_i32 = arith.constant -1 : i32
    %c256_i64 = arith.constant 256 : i64
    %false = arith.constant false
    %true = arith.constant true
    %c192_i64 = arith.constant 192 : i64
    %c64_i64 = arith.constant 64 : i64
    %c3_i32 = arith.constant 3 : i32
    %c128_i64 = arith.constant 128 : i64
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c128 = arith.constant 128 : index
    %c128_i32 = arith.constant 128 : i32
    %c131072_i64 = arith.constant 131072 : i64
    %c786432_i64 = arith.constant 786432 : i64
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c256_i32 = arith.constant 256 : i32
    %c64_i32 = arith.constant 64 : i32
    %c512_i64 = arith.constant 512 : i64
    %c255_i64 = arith.constant 255 : i64
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c12288 = arith.constant 12288 : index
    %c32_i64 = arith.constant 32 : i64
    %c30_i64 = arith.constant 30 : i64
    %c25_i64 = arith.constant 25 : i64
    %c24_i64 = arith.constant 24 : i64
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
    %alloc = memref.alloc() {cannir.arg_index = 3 : i64} : memref<1x!ascdtype.fp8_e8m0>
    %alloc_0 = memref.alloc() {cannir.arg_index = 7 : i64} : memref<1xi32>
    %alloc_1 = memref.alloc() {cannir.arg_index = 9 : i64} : memref<1xi32>
    %alloc_2 = memref.alloc() {cannir.arg_index = 11 : i64} : memref<1xi32>
    %alloc_3 = memref.alloc() {cannir.arg_index = 26 : i64} : memref<1024xi32>
    %alloc_4 = memref.alloc() {cannir.byte_offset = 0 : i64, cannir.onchip_space_root} : memref<262144xui8, 1>
    %alloc_5 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<12288x!ascdtype.fp4x2_e2m1, 3>
    %alloc_6 = memref.alloc() {cannir.byte_offset = 6144 : i64} : memref<12288x!ascdtype.fp4x2_e2m1, 3>
    %alloc_7 = memref.alloc() {cannir.byte_offset = 0 : i64, cannir.onchip_space_root} : memref<262144xui8, 5>
    %alloc_8 = memref.alloc() {cannir.byte_offset = 0 : i64, cannir.onchip_space_root} : memref<65536xui8, 4>
    %alloc_9 = memref.alloc() {cannir.byte_offset = 92160 : i64} : memref<1024x!ascdtype.fp8_e8m0, 2>
    %alloc_10 = memref.alloc() {cannir.byte_offset = 93184 : i64} : memref<1024x!ascdtype.fp8_e8m0, 2>
    %alloc_11 = memref.alloc() {cannir.byte_offset = 95232 : i64} : memref<1024x!ascdtype.fp8_e8m0, 2>
    %alloc_12 = memref.alloc() {cannir.byte_offset = 94208 : i64} : memref<1024x!ascdtype.fp8_e8m0, 2>
    %alloc_13 = memref.alloc() {cannir.byte_offset = 24576 : i64} : memref<32768x!ascdtype.fp4x2_e2m1, 2>
    %alloc_14 = memref.alloc() {cannir.byte_offset = 40960 : i64} : memref<32768x!ascdtype.fp4x2_e2m1, 2>
    %alloc_15 = memref.alloc() {cannir.byte_offset = 73728 : i64} : memref<32768x!ascdtype.fp4x2_e2m1, 2>
    %alloc_16 = memref.alloc() {cannir.byte_offset = 57344 : i64} : memref<32768x!ascdtype.fp4x2_e2m1, 2>
    %alloc_17 = memref.alloc() {cannir.byte_offset = 91136 : i64} : memref<768x!ascdtype.fp8_e8m0, 2>
    %alloc_18 = memref.alloc() {cannir.byte_offset = 90112 : i64} : memref<768x!ascdtype.fp8_e8m0, 2>
    %alloc_19 = memref.alloc() {cannir.byte_offset = 12288 : i64} : memref<24576x!ascdtype.fp4x2_e2m1, 2>
    %alloc_20 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<24576x!ascdtype.fp4x2_e2m1, 2>
    %0 = memref.load %alloc_3[%c288] : memref<1024xi32>
    %1 = arith.cmpi ne, %0, %c0_i32 : i32
    %2 = arith.muli %arg20, %c32_i64 : i64
    %3 = ascscalar.bind_gm_ptr %arg0 : memref<1x!ascdtype.fp4x2_e2m1>
    %4 = arith.muli %arg5, %c2_i64 : i64
    %5 = ascscalar.bind_gm_ptr %arg1 : memref<1x!ascdtype.fp4x2_e2m1>
    %6 = ascscalar.bind_gm_ptr %arg4 : memref<1x!ascdtype.fp8_e8m0>
    %7 = ascscalar.get_block_idx
    %8 = arith.index_cast %7 : i64 to index
    %9 = arith.muli %8, %c8 : index
    %10 = arith.addi %9, %c7 : index
    %11 = memref.load %alloc_3[%10] : memref<1024xi32>
    %12 = arith.extsi %11 : i32 to i64
    %13 = memref.load %alloc_3[%9] : memref<1024xi32>
    %14 = arith.cmpi ne, %13, %c0_i32 : i32
    scf.if %14 {
      %15 = arith.addi %9, %c1 : index
      %16 = memref.load %alloc_3[%15] : memref<1024xi32>
      %17 = arith.extsi %16 : i32 to i64
      %18 = arith.addi %9, %c4 : index
      %19 = memref.load %alloc_3[%18] : memref<1024xi32>
      %20 = arith.extsi %19 : i32 to i64
      %21 = arith.addi %20, %c1_i64 : i64
      %22 = arith.minsi %21, %arg21 : i64
      %23:7 = scf.for %arg35 = %17 to %22 step %c1_i64 iter_args(%arg36 = %c-1_i32, %arg37 = %12, %arg38 = %c0_i64, %arg39 = %c0_i64, %arg40 = %c0_i64, %arg41 = %c0_i64, %arg42 = %c0_i64) -> (i32, i64, i64, i64, i64, i64, i64)  : i64 {
        %24 = arith.index_cast %arg35 : i64 to index
        %25 = memref.load %alloc_0[%24] : memref<1xi32>
        %26 = arith.extsi %25 : i32 to i64
        %27 = arith.addi %arg35, %c1_i64 : i64
        %28 = arith.index_cast %27 : i64 to index
        %29 = memref.load %alloc_0[%28] : memref<1xi32>
        %30 = arith.extsi %29 : i32 to i64
        %31 = arith.subi %30, %26 : i64
        %32 = arith.trunci %31 : i64 to i32
        %33 = memref.load %alloc_1[%24] : memref<1xi32>
        %34 = arith.addi %31, %c5_i64 : i64
        %35 = arith.remsi %34, %c6_i64 : i64
        %36 = arith.cmpi ne, %35, %c0_i64 : i64
        %37 = arith.cmpi slt, %35, %c0_i64 : i64
        %38 = arith.andi %36, %37 : i1
        %39 = arith.divsi %34, %c6_i64 : i64
        %40 = arith.extui %38 : i1 to i64
        %41 = arith.subi %39, %40 : i64
        %42 = arith.cmpi eq, %arg35, %17 : i64
        %43 = scf.if %42 -> (i64) {
          %47 = arith.addi %9, %c2 : index
          %48 = memref.load %alloc_3[%47] : memref<1024xi32>
          %49 = arith.extsi %48 : i32 to i64
          scf.yield %49 : i64
        } else {
          scf.yield %c0_i64 : i64
        }
        %44 = arith.cmpi eq, %arg35, %20 : i64
        %45 = scf.if %44 -> (i64) {
          %47 = arith.addi %9, %c5 : index
          %48 = memref.load %alloc_3[%47] : memref<1024xi32>
          %49 = arith.extsi %48 : i32 to i64
          %50 = arith.addi %9, %c6 : index
          %51 = memref.load %alloc_3[%50] : memref<1024xi32>
          %52 = arith.cmpi sgt, %51, %c0_i32 : i32
          %53 = arith.extui %52 : i1 to i64
          %54 = arith.addi %49, %53 : i64
          scf.yield %54 : i64
        } else {
          scf.yield %41 : i64
        }
        %46:7 = scf.for %arg43 = %43 to %45 step %c1_i64 iter_args(%arg44 = %arg36, %arg45 = %arg37, %arg46 = %arg38, %arg47 = %arg39, %arg48 = %arg40, %arg49 = %arg41, %arg50 = %arg42) -> (i32, i64, i64, i64, i64, i64, i64)  : i64 {
          %47 = arith.muli %arg43, %c6_i64 : i64
          %48 = arith.trunci %47 : i64 to i32
          %49 = arith.extsi %48 : i32 to i64
          %50 = arith.addi %26, %49 : i64
          %51 = arith.trunci %50 : i64 to i32
          %52 = arith.extsi %51 : i32 to i64
          %53 = arith.subi %30, %52 : i64
          %54 = arith.minsi %53, %c6_i64 : i64
          %55 = arith.trunci %54 : i64 to i32
          %56 = arith.subi %32, %48 : i32
          %57 = arith.maxsi %56, %c0_i32 : i32
          %58 = arith.minsi %55, %57 : i32
          %59 = arith.cmpi eq, %58, %c0_i32 : i32
          %60 = arith.select %59, %c0_i32, %33 : i32
          %61 = arith.extsi %60 : i32 to i64
          %62 = arith.addi %61, %c255_i64 : i64
          %63 = arith.remsi %62, %c256_i64 : i64
          %64 = arith.cmpi ne, %63, %c0_i64 : i64
          %65 = arith.cmpi slt, %63, %c0_i64 : i64
          %66 = arith.andi %64, %65 : i1
          %67 = arith.divsi %62, %c256_i64 : i64
          %68 = arith.extui %66 : i1 to i64
          %69 = arith.subi %67, %68 : i64
          %70 = arith.cmpi eq, %arg43, %43 : i64
          %71 = arith.select %42, %70, %false : i1
          %72 = scf.if %71 -> (i64) {
            %89 = arith.addi %9, %c3 : index
            %90 = memref.load %alloc_3[%89] : memref<1024xi32>
            %91 = arith.extsi %90 : i32 to i64
            scf.yield %91 : i64
          } else {
            scf.yield %c0_i64 : i64
          }
          %73 = scf.if %44 -> (i1) {
            %89 = arith.addi %9, %c5 : index
            %90 = memref.load %alloc_3[%89] : memref<1024xi32>
            %91 = arith.extsi %90 : i32 to i64
            %92 = arith.cmpi eq, %arg43, %91 : i64
            scf.yield %92 : i1
          } else {
            scf.yield %false : i1
          }
          %74 = scf.if %73 -> (i64) {
            %89 = arith.addi %9, %c6 : index
            %90 = memref.load %alloc_3[%89] : memref<1024xi32>
            %91 = arith.extsi %90 : i32 to i64
            %92 = arith.minsi %69, %91 : i64
            scf.yield %92 : i64
          } else {
            scf.yield %69 : i64
          }
          %75 = arith.cmpi sgt, %72, %c0_i64 : i64
          %76 = arith.cmpi slt, %74, %69 : i64
          %77 = arith.select %75, %true, %76 : i1
          %78 = arith.muli %72, %c256_i64 : i64
          %79 = arith.trunci %78 : i64 to i32
          %80 = arith.cmpi sgt, %60, %79 : i32
          %81 = arith.cmpi sgt, %74, %72 : i64
          %82 = arith.select %80, %81, %false : i1
          %83:5 = scf.if %82 -> (i64, i64, i64, i64, i64) {
            %89 = arith.cmpi sge, %arg44, %c0_i32 : i32
            %90 = arith.cmpi ne, %arg44, %c0_i32 : i32
            %91 = arith.select %89, %90, %false : i1
            scf.if %91 {
              asccube.sync_all
              asccube.sync_intra_arrive fixpipe, %c14_i64
              asccube.sync_intra_arrive fixpipe, %c30_i64
              asccube.sync_intra_wait fixpipe, %c14_i64
              asccube.sync_intra_wait fixpipe, %c30_i64
            }
            %92 = arith.muli %52, %c32_i64 : i64
            %93 = arith.subi %2, %92 : i64
            %94 = arith.cmpi sgt, %93, %c0_i64 : i64
            %95 = arith.select %94, %93, %c0_i64 : i64
            %96 = arith.cmpi slt, %95, %c192_i64 : i64
            %97 = arith.select %96, %95, %c192_i64 : i64
            %98 = arith.cmpi eq, %arg46, %c0_i64 : i64
            %99 = ascscalar.memref_address %alloc_20 : memref<24576x!ascdtype.fp4x2_e2m1, 2>
            %100 = ascscalar.memref_address %alloc_19 : memref<24576x!ascdtype.fp4x2_e2m1, 2>
            %101 = arith.select %98, %99, %100 : i64
            %102 = arith.cmpi ne, %arg46, %c0_i64 : i64
            %103 = arith.extui %102 : i1 to i64
            %104 = arith.addi %arg46, %c1_i64 : i64
            %105 = arith.remui %104, %c2_i64 : i64
            %106 = ascscalar.bind_local_ptr %101 : memref<1x!ascdtype.fp4x2_e2m1, 2>
            %107 = arith.divui %92, %c4_i64 : i64
            %108 = arith.muli %107, %c512_i64 : i64
            asccube.get_buf mte2, %103, 0 {cannir.buffer.generated_sync}
            %109 = arith.cmpi ugt, %97, %c0_i64 : i64
            scf.if %109 {
              %139 = arith.index_cast %108 : i64 to index
              %140 = arith.trunci %97 : i64 to i32
              asccube.copy_gm2l1 %106[%c0], %3[%139], %140, %c64_i32, %c64_i64, %c1_i32, %c0_i64, %c0_i64, %c192_i32 l2_cache_ctl %c0_i32 {nd2nz = true} : memref<1x!ascdtype.fp4x2_e2m1, 2>, memref<1x!ascdtype.fp4x2_e2m1>
            }
            asccube.release_buf mte2, %103, 0 {cannir.buffer.generated_sync}
            %110 = ascscalar.memref_address %alloc_18 : memref<768x!ascdtype.fp8_e8m0, 2>
            %111 = ascscalar.memref_address %alloc_17 : memref<768x!ascdtype.fp8_e8m0, 2>
            %112 = arith.select %98, %110, %111 : i64
            %113 = arith.select %98, %c6_i64, %c7_i64 : i64
            %114 = cannir.extract_layout %arg3 : !cannir.tensor<<gm, !ascdtype.fp8_e8m0>, <<(-1, 2, 2)>, <(4, 2, 1)>>, <<(-1, 2, 2)>, <(4, 2, 1)>>, identity>
            %115 = cannir.extract_shape %114 : <<(-1, 2, 2)>, <(4, 2, 1)>> -> <(-1, 2, 2)>
            %116 = cannir.extract_dim %115[0] : !cannir.shape<(-1, 2, 2)>
            %117 = arith.subi %116, %92 : i64
            %118 = arith.divsi %117, %c192_i64 : i64
            %119 = arith.minsi %118, %c0_i64 : i64
            %120 = arith.muli %119, %c192_i64 : i64
            %121 = arith.subi %117, %120 : i64
            %122 = arith.minsi %121, %c192_i64 : i64
            %123 = arith.cmpi sge, %118, %c0_i64 : i64
            %124 = arith.select %123, %122, %c0_i64 : i64
            %125 = arith.muli %52, %c128_i64 : i64
            %126 = ascscalar.bind_local_ptr %112 : memref<1x!ascdtype.fp8_e8m0, 2>
            asccube.get_buf mte2, %113, 0 {cannir.buffer.generated_sync}
            %127 = arith.cmpi ugt, %124, %c0_i64 : i64
            scf.if %127 {
              %139 = arith.index_cast %125 : i64 to index
              %140 = arith.trunci %124 : i64 to i32
              asccube.copy_gm2l1 %126[%c0], %alloc[%139], %c2_i32, %140, %c4_i64, %c1_i32, %c0_i64, %c0_i64, %c2_i32 l2_cache_ctl %c0_i32 {dn2nz = true, mx_scale = true} : memref<1x!ascdtype.fp8_e8m0, 2>, memref<1x!ascdtype.fp8_e8m0>
            }
            asccube.release_buf mte2, %113, 0 {cannir.buffer.generated_sync}
            %128 = ascscalar.bitcast_memref %alloc_5 byte_offset = %c0_i64 : memref<12288x!ascdtype.fp4x2_e2m1, 3> -> memref<?x!ascdtype.fp4x2_e2m1, 3>
            %129 = ascscalar.memref_address %128 : memref<?x!ascdtype.fp4x2_e2m1, 3>
            %130 = ascscalar.bind_local_ptr %129 : memref<?xi8, 6>
            asccube.get_buf mte1, %c12_i64, 0 {cannir.buffer.generated_sync}
            asccube.get_buf mte1, %103, 0 {cannir.buffer.generated_sync}
            asccube.copy_l12l0a %128[%c0], %106[%c0], %c0_i32, %c0_i32, %c6_i32, %c2_i32, %c12_i32, %c6_i32 : memref<?x!ascdtype.fp4x2_e2m1, 3>, memref<1x!ascdtype.fp4x2_e2m1, 2>
            asccube.get_buf mte1, %113, 0 {cannir.buffer.generated_sync}
            asccube.copy_l12l0a %130[%c0], %126[%c0], %c0_i32, %c0_i32, %c6_i32, %c2_i32, %c2_i32, %c2_i32 : memref<?xi8, 6>, memref<1x!ascdtype.fp8_e8m0, 2>
            asccube.release_buf mte1, %c12_i64, 0 {cannir.buffer.generated_sync}
            %131 = ascscalar.bitcast_memref %alloc_6 byte_offset = %c0_i64 : memref<12288x!ascdtype.fp4x2_e2m1, 3> -> memref<?x!ascdtype.fp4x2_e2m1, 3>
            %132 = ascscalar.memref_address %131 : memref<?x!ascdtype.fp4x2_e2m1, 3>
            %133 = ascscalar.bind_local_ptr %132 : memref<?xi8, 6>
            asccube.get_buf mte1, %c13_i64, 0 {cannir.buffer.generated_sync}
            asccube.copy_l12l0a %131[%c0], %106[%c0], %c6_i32, %c0_i32, %c6_i32, %c2_i32, %c12_i32, %c6_i32 : memref<?x!ascdtype.fp4x2_e2m1, 3>, memref<1x!ascdtype.fp4x2_e2m1, 2>
            asccube.release_buf mte1, %103, 0 {cannir.buffer.generated_sync}
            asccube.copy_l12l0a %133[%c0], %126[%c0], %c6_i32, %c0_i32, %c6_i32, %c2_i32, %c2_i32, %c2_i32 : memref<?xi8, 6>, memref<1x!ascdtype.fp8_e8m0, 2>
            asccube.release_buf mte1, %113, 0 {cannir.buffer.generated_sync}
            asccube.release_buf mte1, %c13_i64, 0 {cannir.buffer.generated_sync}
            %134 = ascscalar.bitcast_memref %alloc_4 byte_offset = %c0_i64 : memref<262144xui8, 1> -> memref<?xbf16, 1>
            %135 = arith.addi %72, %c3_i64 : i64
            %136 = arith.minsi %74, %135 : i64
            %137 = scf.for %arg51 = %72 to %136 step %c1_i64 iter_args(%arg52 = %arg47) -> (i64)  : i64 {
              %139 = arith.cmpi eq, %arg52, %c2_i64 : i64
              %140 = ascscalar.memref_address %alloc_16 : memref<32768x!ascdtype.fp4x2_e2m1, 2>
              %141 = ascscalar.memref_address %alloc_15 : memref<32768x!ascdtype.fp4x2_e2m1, 2>
              %142 = arith.select %139, %140, %141 : i64
              %143 = arith.cmpi eq, %arg52, %c1_i64 : i64
              %144 = ascscalar.memref_address %alloc_14 : memref<32768x!ascdtype.fp4x2_e2m1, 2>
              %145 = arith.select %143, %144, %142 : i64
              %146 = arith.cmpi eq, %arg52, %c0_i64 : i64
              %147 = ascscalar.memref_address %alloc_13 : memref<32768x!ascdtype.fp4x2_e2m1, 2>
              %148 = arith.select %146, %147, %145 : i64
              %149 = arith.select %139, %c4_i64, %c5_i64 : i64
              %150 = arith.select %143, %c3_i64, %149 : i64
              %151 = arith.select %146, %c2_i64, %150 : i64
              %152 = arith.addi %arg52, %c1_i64 : i64
              %153 = arith.remui %152, %c4_i64 : i64
              %154 = ascscalar.memref_address %alloc_12 : memref<1024x!ascdtype.fp8_e8m0, 2>
              %155 = ascscalar.memref_address %alloc_11 : memref<1024x!ascdtype.fp8_e8m0, 2>
              %156 = arith.select %139, %154, %155 : i64
              %157 = ascscalar.memref_address %alloc_10 : memref<1024x!ascdtype.fp8_e8m0, 2>
              %158 = arith.select %143, %157, %156 : i64
              %159 = ascscalar.memref_address %alloc_9 : memref<1024x!ascdtype.fp8_e8m0, 2>
              %160 = arith.select %146, %159, %158 : i64
              %161 = arith.select %139, %c10_i64, %c11_i64 : i64
              %162 = arith.select %143, %c9_i64, %161 : i64
              %163 = arith.select %146, %c8_i64, %162 : i64
              %164 = memref.load %alloc_2[%24] : memref<1xi32>
              %165 = arith.extsi %164 : i32 to i64
              %166 = arith.muli %arg51, %c256_i64 : i64
              %167 = arith.extsi %33 : i32 to i64
              %168 = arith.subi %167, %166 : i64
              %169 = arith.minsi %168, %c256_i64 : i64
              %170 = arith.addi %165, %166 : i64
              %171 = ascscalar.bind_local_ptr %148 : memref<1x!ascdtype.fp4x2_e2m1, 2>
              %172 = arith.muli %170, %4 : i64
              %173 = arith.index_cast %4 : i64 to index
              %174 = arith.divui %173, %c2 : index
              %175 = arith.index_cast %174 : index to i64
              asccube.get_buf mte2, %151, 0 {cannir.buffer.generated_sync}
              %176 = arith.cmpi ugt, %169, %c0_i64 : i64
              scf.if %176 {
                %179 = arith.index_cast %172 : i64 to index
                %180 = arith.trunci %169 : i64 to i32
                asccube.copy_gm2l1 %171[%c0], %5[%179], %180, %c64_i32, %175, %c1_i32, %c0_i64, %c0_i64, %c256_i32 l2_cache_ctl %c0_i32 {nd2nz = true} : memref<1x!ascdtype.fp4x2_e2m1, 2>, memref<1x!ascdtype.fp4x2_e2m1>
              }
              asccube.release_buf mte2, %151, 0 {cannir.buffer.generated_sync}
              %177 = arith.muli %170, %arg6 : i64
              %178 = ascscalar.bind_local_ptr %160 : memref<1x!ascdtype.fp8_e8m0, 2>
              asccube.get_buf mte2, %163, 0 {cannir.buffer.generated_sync}
              scf.if %176 {
                %179 = arith.index_cast %177 : i64 to index
                %180 = arith.trunci %169 : i64 to i32
                asccube.copy_gm2l1 %178[%c0], %6[%179], %c2_i32, %180, %arg6, %c1_i32, %c0_i64, %c0_i64, %c2_i32 l2_cache_ctl %c0_i32 {dn2nz = true, mx_scale = true} : memref<1x!ascdtype.fp8_e8m0, 2>, memref<1x!ascdtype.fp8_e8m0>
              }
              asccube.release_buf mte2, %163, 0 {cannir.buffer.generated_sync}
              scf.yield %153 : i64
            } {cannir.loop_unroll = 1 : i64}
            %138:4 = scf.for %arg51 = %72 to %74 step %c1_i64 iter_args(%arg52 = %137, %arg53 = %arg48, %arg54 = %arg49, %arg55 = %arg50) -> (i64, i64, i64, i64)  : i64 {
              %139 = arith.addi %arg51, %c3_i64 : i64
              %140 = arith.cmpi slt, %139, %74 : i64
              %141 = scf.if %140 -> (i64) {
                %151 = arith.cmpi eq, %arg52, %c2_i64 : i64
                %152 = ascscalar.memref_address %alloc_16 : memref<32768x!ascdtype.fp4x2_e2m1, 2>
                %153 = ascscalar.memref_address %alloc_15 : memref<32768x!ascdtype.fp4x2_e2m1, 2>
                %154 = arith.select %151, %152, %153 : i64
                %155 = arith.cmpi eq, %arg52, %c1_i64 : i64
                %156 = ascscalar.memref_address %alloc_14 : memref<32768x!ascdtype.fp4x2_e2m1, 2>
                %157 = arith.select %155, %156, %154 : i64
                %158 = arith.cmpi eq, %arg52, %c0_i64 : i64
                %159 = ascscalar.memref_address %alloc_13 : memref<32768x!ascdtype.fp4x2_e2m1, 2>
                %160 = arith.select %158, %159, %157 : i64
                %161 = arith.select %151, %c4_i64, %c5_i64 : i64
                %162 = arith.select %155, %c3_i64, %161 : i64
                %163 = arith.select %158, %c2_i64, %162 : i64
                %164 = arith.addi %arg52, %c1_i64 : i64
                %165 = arith.remui %164, %c4_i64 : i64
                %166 = ascscalar.memref_address %alloc_12 : memref<1024x!ascdtype.fp8_e8m0, 2>
                %167 = ascscalar.memref_address %alloc_11 : memref<1024x!ascdtype.fp8_e8m0, 2>
                %168 = arith.select %151, %166, %167 : i64
                %169 = ascscalar.memref_address %alloc_10 : memref<1024x!ascdtype.fp8_e8m0, 2>
                %170 = arith.select %155, %169, %168 : i64
                %171 = ascscalar.memref_address %alloc_9 : memref<1024x!ascdtype.fp8_e8m0, 2>
                %172 = arith.select %158, %171, %170 : i64
                %173 = arith.select %151, %c10_i64, %c11_i64 : i64
                %174 = arith.select %155, %c9_i64, %173 : i64
                %175 = arith.select %158, %c8_i64, %174 : i64
                %176 = memref.load %alloc_2[%24] : memref<1xi32>
                %177 = arith.extsi %176 : i32 to i64
                %178 = arith.muli %139, %c256_i64 : i64
                %179 = arith.extsi %33 : i32 to i64
                %180 = arith.subi %179, %178 : i64
                %181 = arith.minsi %180, %c256_i64 : i64
                %182 = arith.addi %177, %178 : i64
                %183 = ascscalar.bind_local_ptr %160 : memref<1x!ascdtype.fp4x2_e2m1, 2>
                %184 = arith.muli %182, %4 : i64
                %185 = arith.index_cast %4 : i64 to index
                %186 = arith.divui %185, %c2 : index
                %187 = arith.index_cast %186 : index to i64
                asccube.get_buf mte2, %163, 0 {cannir.buffer.generated_sync}
                %188 = arith.cmpi ugt, %181, %c0_i64 : i64
                scf.if %188 {
                  %191 = arith.index_cast %184 : i64 to index
                  %192 = arith.trunci %181 : i64 to i32
                  asccube.copy_gm2l1 %183[%c0], %5[%191], %192, %c64_i32, %187, %c1_i32, %c0_i64, %c0_i64, %c256_i32 l2_cache_ctl %c0_i32 {nd2nz = true} : memref<1x!ascdtype.fp4x2_e2m1, 2>, memref<1x!ascdtype.fp4x2_e2m1>
                }
                asccube.release_buf mte2, %163, 0 {cannir.buffer.generated_sync}
                %189 = arith.muli %182, %arg6 : i64
                %190 = ascscalar.bind_local_ptr %172 : memref<1x!ascdtype.fp8_e8m0, 2>
                asccube.get_buf mte2, %175, 0 {cannir.buffer.generated_sync}
                scf.if %188 {
                  %191 = arith.index_cast %189 : i64 to index
                  %192 = arith.trunci %181 : i64 to i32
                  asccube.copy_gm2l1 %190[%c0], %6[%191], %c2_i32, %192, %arg6, %c1_i32, %c0_i64, %c0_i64, %c2_i32 l2_cache_ctl %c0_i32 {dn2nz = true, mx_scale = true} : memref<1x!ascdtype.fp8_e8m0, 2>, memref<1x!ascdtype.fp8_e8m0>
                }
                asccube.release_buf mte2, %175, 0 {cannir.buffer.generated_sync}
                scf.yield %165 : i64
              } else {
                scf.yield %arg52 : i64
              }
              %142 = arith.subi %arg51, %72 : i64
              %143 = arith.remsi %142, %c2_i64 : i64
              %144 = arith.cmpi ne, %143, %c0_i64 : i64
              %145 = arith.cmpi slt, %143, %c0_i64 : i64
              %146 = arith.andi %144, %145 : i1
              %147 = arith.select %146, %c2_i64, %c0_i64 : i64
              %148 = arith.addi %143, %147 overflow<nsw> : i64
              %149 = arith.cmpi eq, %148, %c0_i64 : i64
              %150:3 = scf.if %149 -> (i64, i64, i64) {
                asccube.sync_intra_wait fixpipe, %c8_i64
                asccube.sync_intra_wait fixpipe, %c24_i64
                %151 = arith.cmpi eq, %arg54, %c0_i64 : i64
                %152 = arith.select %151, %c0_i64, %c131072_i64 : i64
                %153 = arith.select %151, %c14_i64, %c15_i64 : i64
                %154 = arith.addi %arg54, %c1_i64 : i64
                %155 = arith.remui %154, %c2_i64 : i64
                %156 = arith.cmpi eq, %arg53, %c2_i64 : i64
                %157 = ascscalar.memref_address %alloc_16 : memref<32768x!ascdtype.fp4x2_e2m1, 2>
                %158 = ascscalar.memref_address %alloc_15 : memref<32768x!ascdtype.fp4x2_e2m1, 2>
                %159 = arith.select %156, %157, %158 : i64
                %160 = arith.cmpi eq, %arg53, %c1_i64 : i64
                %161 = ascscalar.memref_address %alloc_14 : memref<32768x!ascdtype.fp4x2_e2m1, 2>
                %162 = arith.select %160, %161, %159 : i64
                %163 = arith.cmpi eq, %arg53, %c0_i64 : i64
                %164 = ascscalar.memref_address %alloc_13 : memref<32768x!ascdtype.fp4x2_e2m1, 2>
                %165 = arith.select %163, %164, %162 : i64
                %166 = arith.select %156, %c4_i64, %c5_i64 : i64
                %167 = arith.select %160, %c3_i64, %166 : i64
                %168 = arith.select %163, %c2_i64, %167 : i64
                %169 = arith.addi %arg53, %c1_i64 : i64
                %170 = arith.remui %169, %c4_i64 : i64
                %171 = ascscalar.memref_address %alloc_12 : memref<1024x!ascdtype.fp8_e8m0, 2>
                %172 = ascscalar.memref_address %alloc_11 : memref<1024x!ascdtype.fp8_e8m0, 2>
                %173 = arith.select %156, %171, %172 : i64
                %174 = ascscalar.memref_address %alloc_10 : memref<1024x!ascdtype.fp8_e8m0, 2>
                %175 = arith.select %160, %174, %173 : i64
                %176 = ascscalar.memref_address %alloc_9 : memref<1024x!ascdtype.fp8_e8m0, 2>
                %177 = arith.select %163, %176, %175 : i64
                %178 = arith.select %156, %c10_i64, %c11_i64 : i64
                %179 = arith.select %160, %c9_i64, %178 : i64
                %180 = arith.select %163, %c8_i64, %179 : i64
                %181 = ascscalar.bitcast_memref %alloc_8 byte_offset = %c0_i64 : memref<65536xui8, 4> -> memref<?x!ascdtype.fp4x2_e2m1, 4>
                %182 = ascscalar.bind_local_ptr %165 : memref<1x!ascdtype.fp4x2_e2m1, 2>
                %183 = arith.index_cast %152 : i64 to index
                %184 = arith.divui %183, %c4 : index
                %185 = ascscalar.bind_local_ptr %177 : memref<1x!ascdtype.fp8_e8m0, 2>
                %186 = ascscalar.memref_address %alloc_8 : memref<65536xui8, 4>
                %187 = arith.divui %152, %c8_i64 : i64
                %188 = arith.addi %186, %187 : i64
                %189 = ascscalar.bind_local_ptr %188 : memref<1x!ascdtype.fp4x2_e2m1, 4>
                %190 = ascscalar.memref_address %189 : memref<1x!ascdtype.fp4x2_e2m1, 4>
                %191 = ascscalar.bind_local_ptr %190 : memref<?xi8, 7>
                asccube.get_buf mte1, %153, 0 {cannir.buffer.generated_sync}
                asccube.get_buf mte1, %168, 0 {cannir.buffer.generated_sync}
                asccube.get_buf mte1, %180, 0 {cannir.buffer.generated_sync}
                asccube.copy_l12l0b %181[%184], %182[%c0], %c0_i32, %c0_i32, %c16_i32, %c2_i32, %c16_i32, %c16_i32 : memref<?x!ascdtype.fp4x2_e2m1, 4>, memref<1x!ascdtype.fp4x2_e2m1, 2>
                asccube.copy_l12l0b %191[%c0], %185[%c0], %c0_i32, %c0_i32, %c16_i32, %c2_i32, %c2_i32, %c2_i32 : memref<?xi8, 7>, memref<1x!ascdtype.fp8_e8m0, 2>
                asccube.release_buf mte1, %180, 0 {cannir.buffer.generated_sync}
                asccube.release_buf mte1, %168, 0 {cannir.buffer.generated_sync}
                asccube.release_buf mte1, %153, 0 {cannir.buffer.generated_sync}
                %192 = arith.cmpi eq, %arg55, %c0_i64 : i64
                %193 = arith.select %192, %c0_i64, %c786432_i64 : i64
                %194 = arith.addi %arg55, %c1_i64 : i64
                %195 = arith.remui %194, %c2_i64 : i64
                %196 = ascscalar.memref_address %alloc_7 : memref<262144xui8, 5>
                %197 = arith.divui %193, %c8_i64 : i64
                %198 = arith.addi %196, %197 : i64
                %199 = ascscalar.bind_local_ptr %198 : memref<1xf32, 5>
                asccube.get_buf m, %c12_i64, 0 {cannir.buffer.generated_sync}
                asccube.get_buf m, %153, 0 {cannir.buffer.generated_sync}
                asccube.mmad %199[%c0], %128, %189 scale(%130, %191), %c96_i32, %c128_i32, %c256_i32, %c3_i32 {c_matrix_init_val = true, mx_mode = true} : memref<1xf32, 5>, memref<?x!ascdtype.fp4x2_e2m1, 3>, memref<1x!ascdtype.fp4x2_e2m1, 4>, memref<?xi8, 6>, memref<?xi8, 7>
                asccube.release_buf m, %153, 0 {cannir.buffer.generated_sync}
                asccube.release_buf m, %c12_i64, 0 {cannir.buffer.generated_sync}
                %200 = ascscalar.bitcast_memref %alloc_7 byte_offset = %c0_i64 : memref<262144xui8, 5> -> memref<?xf32, 5>
                %201 = arith.index_cast %193 : i64 to index
                %202 = arith.divui %201, %c32 : index
                asccube.sync_intra_wait fixpipe, %c0_i64 {cannir.sync.banks = 3 : i32}
                asccube.sync_intra_wait fixpipe, %c16_i64 {cannir.sync.banks = 3 : i32}
                asccube.copy_l0c2ub %134[%c0], %200[%202], %c128_i32, %c96_i32, %c256_i32, %c96_i32, %c3_i32 sub_block_id %c0_i32 {nz2nd_en = true, relu_en = true} : memref<?xbf16, 1>, memref<?xf32, 5>
                %203 = arith.addi %202, %c12288 : index
                asccube.copy_l0c2ub %134[%c24576], %200[%203], %c128_i32, %c96_i32, %c256_i32, %c96_i32, %c3_i32 sub_block_id %c0_i32 {nz2nd_en = true, relu_en = true} : memref<?xbf16, 1>, memref<?xf32, 5>
                %204 = arith.cmpi eq, %195, %c0_i64 : i64
                %205 = arith.select %204, %c0_i64, %c786432_i64 : i64
                %206 = arith.addi %195, %c1_i64 overflow<nsw> : i64
                %207 = arith.remui %206, %c2_i64 : i64
                %208 = arith.divui %205, %c8_i64 : i64
                %209 = arith.addi %196, %208 : i64
                %210 = ascscalar.bind_local_ptr %209 : memref<1xf32, 5>
                asccube.get_buf m, %c13_i64, 0 {cannir.buffer.generated_sync}
                asccube.get_buf m, %153, 0 {cannir.buffer.generated_sync}
                asccube.mmad %210[%c0], %131, %189 scale(%133, %191), %c96_i32, %c128_i32, %c256_i32, %c3_i32 {c_matrix_init_val = true, mx_mode = true} : memref<1xf32, 5>, memref<?x!ascdtype.fp4x2_e2m1, 3>, memref<1x!ascdtype.fp4x2_e2m1, 4>, memref<?xi8, 6>, memref<?xi8, 7>
                asccube.release_buf m, %153, 0 {cannir.buffer.generated_sync}
                asccube.release_buf m, %c13_i64, 0 {cannir.buffer.generated_sync}
                %211 = arith.index_cast %205 : i64 to index
                %212 = arith.divui %211, %c32 : index
                asccube.copy_l0c2ub %134[%c0], %200[%212], %c128_i32, %c96_i32, %c256_i32, %c96_i32, %c3_i32 sub_block_id %c1_i32 {nz2nd_en = true, relu_en = true} : memref<?xbf16, 1>, memref<?xf32, 5>
                %213 = arith.addi %212, %c12288 : index
                asccube.copy_l0c2ub %134[%c24576], %200[%213], %c128_i32, %c96_i32, %c256_i32, %c96_i32, %c3_i32 sub_block_id %c1_i32 {nz2nd_en = true, relu_en = true} : memref<?xbf16, 1>, memref<?xf32, 5>
                asccube.sync_intra_arrive fixpipe, %c0_i64 {cannir.sync.banks = 3 : i32}
                asccube.sync_intra_arrive fixpipe, %c16_i64 {cannir.sync.banks = 3 : i32}
                asccube.sync_intra_arrive fixpipe, %c8_i64
                asccube.sync_intra_arrive fixpipe, %c24_i64
                scf.yield %170, %155, %207 : i64, i64, i64
              } else {
                asccube.sync_intra_wait fixpipe, %c9_i64
                asccube.sync_intra_wait fixpipe, %c25_i64
                %151 = arith.cmpi eq, %arg54, %c0_i64 : i64
                %152 = arith.select %151, %c0_i64, %c131072_i64 : i64
                %153 = arith.select %151, %c14_i64, %c15_i64 : i64
                %154 = arith.addi %arg54, %c1_i64 : i64
                %155 = arith.remui %154, %c2_i64 : i64
                %156 = arith.cmpi eq, %arg53, %c2_i64 : i64
                %157 = ascscalar.memref_address %alloc_16 : memref<32768x!ascdtype.fp4x2_e2m1, 2>
                %158 = ascscalar.memref_address %alloc_15 : memref<32768x!ascdtype.fp4x2_e2m1, 2>
                %159 = arith.select %156, %157, %158 : i64
                %160 = arith.cmpi eq, %arg53, %c1_i64 : i64
                %161 = ascscalar.memref_address %alloc_14 : memref<32768x!ascdtype.fp4x2_e2m1, 2>
                %162 = arith.select %160, %161, %159 : i64
                %163 = arith.cmpi eq, %arg53, %c0_i64 : i64
                %164 = ascscalar.memref_address %alloc_13 : memref<32768x!ascdtype.fp4x2_e2m1, 2>
                %165 = arith.select %163, %164, %162 : i64
                %166 = arith.select %156, %c4_i64, %c5_i64 : i64
                %167 = arith.select %160, %c3_i64, %166 : i64
                %168 = arith.select %163, %c2_i64, %167 : i64
                %169 = arith.addi %arg53, %c1_i64 : i64
                %170 = arith.remui %169, %c4_i64 : i64
                %171 = ascscalar.memref_address %alloc_12 : memref<1024x!ascdtype.fp8_e8m0, 2>
                %172 = ascscalar.memref_address %alloc_11 : memref<1024x!ascdtype.fp8_e8m0, 2>
                %173 = arith.select %156, %171, %172 : i64
                %174 = ascscalar.memref_address %alloc_10 : memref<1024x!ascdtype.fp8_e8m0, 2>
                %175 = arith.select %160, %174, %173 : i64
                %176 = ascscalar.memref_address %alloc_9 : memref<1024x!ascdtype.fp8_e8m0, 2>
                %177 = arith.select %163, %176, %175 : i64
                %178 = arith.select %156, %c10_i64, %c11_i64 : i64
                %179 = arith.select %160, %c9_i64, %178 : i64
                %180 = arith.select %163, %c8_i64, %179 : i64
                %181 = ascscalar.bitcast_memref %alloc_8 byte_offset = %c0_i64 : memref<65536xui8, 4> -> memref<?x!ascdtype.fp4x2_e2m1, 4>
                %182 = ascscalar.bind_local_ptr %165 : memref<1x!ascdtype.fp4x2_e2m1, 2>
                %183 = arith.index_cast %152 : i64 to index
                %184 = arith.divui %183, %c4 : index
                %185 = ascscalar.bind_local_ptr %177 : memref<1x!ascdtype.fp8_e8m0, 2>
                %186 = ascscalar.memref_address %alloc_8 : memref<65536xui8, 4>
                %187 = arith.divui %152, %c8_i64 : i64
                %188 = arith.addi %186, %187 : i64
                %189 = ascscalar.bind_local_ptr %188 : memref<1x!ascdtype.fp4x2_e2m1, 4>
                %190 = ascscalar.memref_address %189 : memref<1x!ascdtype.fp4x2_e2m1, 4>
                %191 = ascscalar.bind_local_ptr %190 : memref<?xi8, 7>
                asccube.get_buf mte1, %153, 0 {cannir.buffer.generated_sync}
                asccube.get_buf mte1, %168, 0 {cannir.buffer.generated_sync}
                asccube.get_buf mte1, %180, 0 {cannir.buffer.generated_sync}
                asccube.copy_l12l0b %181[%184], %182[%c0], %c0_i32, %c0_i32, %c16_i32, %c2_i32, %c16_i32, %c16_i32 : memref<?x!ascdtype.fp4x2_e2m1, 4>, memref<1x!ascdtype.fp4x2_e2m1, 2>
                asccube.copy_l12l0b %191[%c0], %185[%c0], %c0_i32, %c0_i32, %c16_i32, %c2_i32, %c2_i32, %c2_i32 : memref<?xi8, 7>, memref<1x!ascdtype.fp8_e8m0, 2>
                asccube.release_buf mte1, %180, 0 {cannir.buffer.generated_sync}
                asccube.release_buf mte1, %168, 0 {cannir.buffer.generated_sync}
                asccube.release_buf mte1, %153, 0 {cannir.buffer.generated_sync}
                %192 = arith.cmpi eq, %arg55, %c0_i64 : i64
                %193 = arith.select %192, %c0_i64, %c786432_i64 : i64
                %194 = arith.addi %arg55, %c1_i64 : i64
                %195 = arith.remui %194, %c2_i64 : i64
                %196 = ascscalar.memref_address %alloc_7 : memref<262144xui8, 5>
                %197 = arith.divui %193, %c8_i64 : i64
                %198 = arith.addi %196, %197 : i64
                %199 = ascscalar.bind_local_ptr %198 : memref<1xf32, 5>
                asccube.get_buf m, %c12_i64, 0 {cannir.buffer.generated_sync}
                asccube.get_buf m, %153, 0 {cannir.buffer.generated_sync}
                asccube.mmad %199[%c0], %128, %189 scale(%130, %191), %c96_i32, %c128_i32, %c256_i32, %c3_i32 {c_matrix_init_val = true, mx_mode = true} : memref<1xf32, 5>, memref<?x!ascdtype.fp4x2_e2m1, 3>, memref<1x!ascdtype.fp4x2_e2m1, 4>, memref<?xi8, 6>, memref<?xi8, 7>
                asccube.release_buf m, %153, 0 {cannir.buffer.generated_sync}
                asccube.release_buf m, %c12_i64, 0 {cannir.buffer.generated_sync}
                %200 = ascscalar.bitcast_memref %alloc_7 byte_offset = %c0_i64 : memref<262144xui8, 5> -> memref<?xf32, 5>
                %201 = arith.index_cast %193 : i64 to index
                %202 = arith.divui %201, %c32 : index
                asccube.sync_intra_wait fixpipe, %c1_i64 {cannir.sync.banks = 3 : i32}
                asccube.sync_intra_wait fixpipe, %c17_i64 {cannir.sync.banks = 3 : i32}
                asccube.copy_l0c2ub %134[%c128], %200[%202], %c128_i32, %c96_i32, %c256_i32, %c96_i32, %c3_i32 sub_block_id %c0_i32 {nz2nd_en = true, relu_en = true} : memref<?xbf16, 1>, memref<?xf32, 5>
                %203 = arith.addi %202, %c12288 : index
                asccube.copy_l0c2ub %134[%c24704], %200[%203], %c128_i32, %c96_i32, %c256_i32, %c96_i32, %c3_i32 sub_block_id %c0_i32 {nz2nd_en = true, relu_en = true} : memref<?xbf16, 1>, memref<?xf32, 5>
                %204 = arith.cmpi eq, %195, %c0_i64 : i64
                %205 = arith.select %204, %c0_i64, %c786432_i64 : i64
                %206 = arith.addi %195, %c1_i64 overflow<nsw> : i64
                %207 = arith.remui %206, %c2_i64 : i64
                %208 = arith.divui %205, %c8_i64 : i64
                %209 = arith.addi %196, %208 : i64
                %210 = ascscalar.bind_local_ptr %209 : memref<1xf32, 5>
                asccube.get_buf m, %c13_i64, 0 {cannir.buffer.generated_sync}
                asccube.get_buf m, %153, 0 {cannir.buffer.generated_sync}
                asccube.mmad %210[%c0], %131, %189 scale(%133, %191), %c96_i32, %c128_i32, %c256_i32, %c3_i32 {c_matrix_init_val = true, mx_mode = true} : memref<1xf32, 5>, memref<?x!ascdtype.fp4x2_e2m1, 3>, memref<1x!ascdtype.fp4x2_e2m1, 4>, memref<?xi8, 6>, memref<?xi8, 7>
                asccube.release_buf m, %153, 0 {cannir.buffer.generated_sync}
                asccube.release_buf m, %c13_i64, 0 {cannir.buffer.generated_sync}
                %211 = arith.index_cast %205 : i64 to index
                %212 = arith.divui %211, %c32 : index
                asccube.copy_l0c2ub %134[%c128], %200[%212], %c128_i32, %c96_i32, %c256_i32, %c96_i32, %c3_i32 sub_block_id %c1_i32 {nz2nd_en = true, relu_en = true} : memref<?xbf16, 1>, memref<?xf32, 5>
                %213 = arith.addi %212, %c12288 : index
                asccube.copy_l0c2ub %134[%c24704], %200[%213], %c128_i32, %c96_i32, %c256_i32, %c96_i32, %c3_i32 sub_block_id %c1_i32 {nz2nd_en = true, relu_en = true} : memref<?xbf16, 1>, memref<?xf32, 5>
                asccube.sync_intra_arrive fixpipe, %c1_i64 {cannir.sync.banks = 3 : i32}
                asccube.sync_intra_arrive fixpipe, %c17_i64 {cannir.sync.banks = 3 : i32}
                asccube.sync_intra_arrive fixpipe, %c9_i64
                asccube.sync_intra_arrive fixpipe, %c25_i64
                scf.yield %170, %155, %207 : i64, i64, i64
              }
              scf.yield %141, %150#0, %150#1, %150#2 : i64, i64, i64, i64
            } {cannir.loop_unroll = 1 : i64}
            asccube.sync_intra_wait fixpipe, %c8_i64
            asccube.sync_intra_wait fixpipe, %c24_i64
            asccube.sync_intra_wait fixpipe, %c9_i64
            asccube.sync_intra_wait fixpipe, %c25_i64
            scf.yield %105, %138#0, %138#1, %138#2, %138#3 : i64, i64, i64, i64, i64
          } else {
            scf.yield %arg46, %arg47, %arg48, %arg49, %arg50 : i64, i64, i64, i64, i64
          }
          %84 = arith.cmpi sgt, %61, %78 : i64
          %85 = arith.select %84, %81, %false : i1
          %86 = arith.select %85, %c0_i32, %arg44 : i32
          %87 = arith.addi %arg45, %c1_i64 : i64
          %88 = arith.select %77, %87, %arg45 : i64
          scf.yield %86, %88, %83#0, %83#1, %83#2, %83#3, %83#4 : i32, i64, i64, i64, i64, i64, i64
        } {cannir.loop_unroll = 1 : i64}
        scf.yield %46#0, %46#1, %46#2, %46#3, %46#4, %46#5, %46#6 : i32, i64, i64, i64, i64, i64, i64
      } {cannir.loop_unroll = 1 : i64}
      asccube.sync_intra_wait fixpipe, %c0_i64
      asccube.sync_intra_wait fixpipe, %c16_i64
      asccube.sync_intra_wait fixpipe, %c1_i64
      asccube.sync_intra_wait fixpipe, %c17_i64
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
