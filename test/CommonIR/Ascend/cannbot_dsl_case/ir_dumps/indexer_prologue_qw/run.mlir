module {
  func.func @run_aea66e421273d96d706faac282b79555(%arg0: !cannir.tensor<<gm, bf16>, <<(-1, 5120)>, <(5120, 1)>>, <<(-1, 5120)>, <(5120, 1)>>, identity>, %arg1: !cannir.tensor<<gm, ui8>, <<(-1, 1280)>, <(1280, 1)>>, <<(-1, 1280)>, <(1280, 1)>>, identity>, %arg2: !cannir.tensor<<gm, ui8>, <<(4096, 1280)>, <(1280, 1)>>, <<(40, 256, 16, 32)>, <(131072, 512, 32, 1)>>, nd2nz> {cannir.physical_shape_upper_bound = array<i64: 40, 256, 16, 32>}, %arg3: !cannir.tensor<<gm, bf16>, <<(32, 5120)>, <(5120, 1)>>, <<(320, 2, 16, 16)>, <(512, 256, 16, 1)>>, nd2nz> {cannir.physical_shape_upper_bound = array<i64: 320, 2, 16, 16>}, %arg4: !cannir.tensor<<gm, ui8>, <<(-1, 20, 2)>, <(40, 2, 1)>>, <<(-1, 20, 2)>, <(40, 2, 1)>>, identity>, %arg5: !cannir.tensor<<gm, ui8>, <<(4096, 20, 2)>, <(40, 2, 1)>>, <<(4096, 20, 2)>, <(40, 2, 1)>>, identity>, %arg6: !cannir.tensor<<gm, f32>, <<(-1, 64)>, <(64, 1)>>, <<(-1, 64)>, <(64, 1)>>, identity>, %arg7: !cannir.tensor<<gm, f32>, <<(-1, 64)>, <(64, 1)>>, <<(-1, 64)>, <(64, 1)>>, identity>, %arg8: !cannir.tensor<<gm, i32>, <<(1088)>, <(1)>>, <<(1088)>, <(1)>>, identity>, %arg9: !cannir.tensor<<gm, ui8>, <<(-1, 2048)>, <(2048, 1)>>, <<(-1, 2048)>, <(2048, 1)>>, identity>, %arg10: !cannir.tensor<<gm, ui8>, <<(-1, 128)>, <(128, 1)>>, <<(-1, 128)>, <(128, 1)>>, identity>, %arg11: !cannir.tensor<<gm, f32>, <<(-1, 32)>, <(32, 1)>>, <<(-1, 32)>, <(32, 1)>>, identity>, %arg12: !cannir.tensor<<gm, f32>, <<(-1, 32)>, <(32, 1)>>, <<(-1, 32)>, <(32, 1)>>, identity>, %arg13: i64, %arg14: f32) attributes {cannbotdsl.contract_digest = "aea66e421273d96d706faac282b79555c8c58716358750e24ffa69a03b795072", cannbotdsl.public_slot_count = 15 : i64, cannbotdsl.slot_dtype_bits = [16, 8, 8, 16, 8, 8, 32, 32, 32, 8, 8, 32, 32, 64, 32], cannbotdsl.slot_dtype_codes = [4, 1, 1, 4, 1, 1, 2, 2, 0, 1, 1, 2, 2, 0, 2], cannbotdsl.slot_dtype_lanes = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], cannbotdsl.slot_flags = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], cannbotdsl.slot_kinds = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2], cannbotdsl.slot_type_digests = ["194f24a1cda9b4ee3ec4a5855574a843122ab43b81074c252dd15ce392acab67", "bcbbac92237ad8d3a124b6399e6bc5c6fda677822e281a6b7d5136d13c739fda", "a2e47c41785abfc44625676d6252c9fc2a09bf6cc66983ff8a38a674ea120cb4", "cbe38aeb74f71307955180a53c7bb45ce0db7c2a81d541d1245b366b73ea6c1d", "b3a07e7087dbe94ad6d73f8079c2701849f789adb5204402fefad708328775e2", "6418f6157332d85547f52f7606ece926ce944c9542430f16a04cba47b5b8a8c3", "71e295a6282dc9c9ad25ae57ed2b959a647a40beab9c4ec6f38d261d6f1e3eb6", "71e295a6282dc9c9ad25ae57ed2b959a647a40beab9c4ec6f38d261d6f1e3eb6", "e90ef642235ca9c8e9a6f6aa795f13fa46ec6a27d72c079969162ee2763481bb", "1f4d689642d0577cfebf6e405c730ddd8252ab89258925a4ea93f3d1e544eac9", "95faee56ef2b42f7565a1590c34614bf6158f4e24e183907b7a9132590c3504b", "3df1003c80d9a71db0af4138b24f7bf5ed383278ec80e197923e607f96da57e0", "93994e2fd77370cbce173d24f75c2ddb44c5acdf3ce26ef19c44316337fe1a9d", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "4a5363e2ec1eb8cefee037ab0b231653c07f9b44a406ef92ed9ba3e6f5541dbf"], cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.host} {
    %0 = cannir.extract_layout %arg12 : !cannir.tensor<<gm, f32>, <<(-1, 32)>, <(32, 1)>>, <<(-1, 32)>, <(32, 1)>>, identity>
    %1 = cannir.extract_shape %0 : <<(-1, 32)>, <(32, 1)>> -> <(-1, 32)>
    %2 = cannir.extract_dim %1[0] : !cannir.shape<(-1, 32)>
    cannir.kernel_launch @kernel_IndexerPrologueQwKernel___call___0(%2) [%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14] {cannbotdsl.profile = {indices = array<i64: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11>, name = "IndexerPrologueQwKernel___call__", op_type = "IndexerPrologueQwKernel___call__", roles = array<i64: 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1>}} : !cannir.tensor<<gm, bf16>, <<(-1, 5120)>, <(5120, 1)>>, <<(-1, 5120)>, <(5120, 1)>>, identity>, !cannir.tensor<<gm, ui8>, <<(-1, 1280)>, <(1280, 1)>>, <<(-1, 1280)>, <(1280, 1)>>, identity>, !cannir.tensor<<gm, ui8>, <<(4096, 1280)>, <(1280, 1)>>, <<(40, 256, 16, 32)>, <(131072, 512, 32, 1)>>, nd2nz>, !cannir.tensor<<gm, bf16>, <<(32, 5120)>, <(5120, 1)>>, <<(320, 2, 16, 16)>, <(512, 256, 16, 1)>>, nd2nz>, !cannir.tensor<<gm, ui8>, <<(-1, 20, 2)>, <(40, 2, 1)>>, <<(-1, 20, 2)>, <(40, 2, 1)>>, identity>, !cannir.tensor<<gm, ui8>, <<(4096, 20, 2)>, <(40, 2, 1)>>, <<(4096, 20, 2)>, <(40, 2, 1)>>, identity>, !cannir.tensor<<gm, f32>, <<(-1, 64)>, <(64, 1)>>, <<(-1, 64)>, <(64, 1)>>, identity>, !cannir.tensor<<gm, f32>, <<(-1, 64)>, <(64, 1)>>, <<(-1, 64)>, <(64, 1)>>, identity>, !cannir.tensor<<gm, i32>, <<(1088)>, <(1)>>, <<(1088)>, <(1)>>, identity>, !cannir.tensor<<gm, ui8>, <<(-1, 2048)>, <(2048, 1)>>, <<(-1, 2048)>, <(2048, 1)>>, identity>, !cannir.tensor<<gm, ui8>, <<(-1, 128)>, <(128, 1)>>, <<(-1, 128)>, <(128, 1)>>, identity>, !cannir.tensor<<gm, f32>, <<(-1, 32)>, <(32, 1)>>, <<(-1, 32)>, <(32, 1)>>, identity>, !cannir.tensor<<gm, f32>, <<(-1, 32)>, <(32, 1)>>, <<(-1, 32)>, <(32, 1)>>, identity>, i64, f32
    return
  }
  func.func @kernel_IndexerPrologueQwKernel___call___0_aiv(%arg0: !cannir.tensor<<gm, bf16>, <<(-1, 5120)>, <(5120, 1)>>, <<(-1, 5120)>, <(5120, 1)>>, identity>, %arg1: !cannir.tensor<<gm, ui8>, <<(-1, 1280)>, <(1280, 1)>>, <<(-1, 1280)>, <(1280, 1)>>, identity>, %arg2: !cannir.tensor<<gm, ui8>, <<(4096, 1280)>, <(1280, 1)>>, <<(40, 256, 16, 32)>, <(131072, 512, 32, 1)>>, nd2nz> {cannir.physical_shape_upper_bound = array<i64: 40, 256, 16, 32>}, %arg3: !cannir.tensor<<gm, bf16>, <<(32, 5120)>, <(5120, 1)>>, <<(320, 2, 16, 16)>, <(512, 256, 16, 1)>>, nd2nz> {cannir.physical_shape_upper_bound = array<i64: 320, 2, 16, 16>}, %arg4: !cannir.tensor<<gm, ui8>, <<(-1, 20, 2)>, <(40, 2, 1)>>, <<(-1, 20, 2)>, <(40, 2, 1)>>, identity>, %arg5: !cannir.tensor<<gm, ui8>, <<(4096, 20, 2)>, <(40, 2, 1)>>, <<(4096, 20, 2)>, <(40, 2, 1)>>, identity>, %arg6: !cannir.tensor<<gm, f32>, <<(-1, 64)>, <(64, 1)>>, <<(-1, 64)>, <(64, 1)>>, identity>, %arg7: !cannir.tensor<<gm, f32>, <<(-1, 64)>, <(64, 1)>>, <<(-1, 64)>, <(64, 1)>>, identity>, %arg8: !cannir.tensor<<gm, i32>, <<(1088)>, <(1)>>, <<(1088)>, <(1)>>, identity>, %arg9: !cannir.tensor<<gm, ui8>, <<(-1, 2048)>, <(2048, 1)>>, <<(-1, 2048)>, <(2048, 1)>>, identity>, %arg10: !cannir.tensor<<gm, ui8>, <<(-1, 128)>, <(128, 1)>>, <<(-1, 128)>, <(128, 1)>>, identity>, %arg11: !cannir.tensor<<gm, f32>, <<(-1, 32)>, <(32, 1)>>, <<(-1, 32)>, <(32, 1)>>, identity>, %arg12: !cannir.tensor<<gm, f32>, <<(-1, 32)>, <(32, 1)>>, <<(-1, 32)>, <(32, 1)>>, identity>, %arg13: i64, %arg14: f32) attributes {cannir.buffer.auto_sync, cannir.buffer.sync_analyzed, cannir.buffer.sync_verified, cannir.buffer.tensor_state, cannir.core = "aiv", cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.device, cannir.explicit_ring_buffer_cursors, cannir.kernel_group = "kernel_IndexerPrologueQwKernel___call___0", cannir.resource.ids_prepared} {
    %c0 = arith.constant 0 : index
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %c2_i64 = arith.constant 2 : i64
    %c4_i64 = arith.constant 4 : i64
    %c8_i64 = arith.constant 8 : i64
    %c25_i64 = arith.constant 25 : i64
    %c26_i64 = arith.constant 26 : i64
    %c16384_i64 = arith.constant 16384 : i64
    %c2048_i64 = arith.constant 2048 : i64
    %c262144_i64 = arith.constant 262144 : i64
    %c2048 = arith.constant 2048 : index
    %c64_i64 = arith.constant 64 : i64
    %c128 = arith.constant 128 : index
    %c64_i32 = arith.constant 64 : i32
    %c4_i32 = arith.constant 4 : i32
    %c256_i64 = arith.constant 256 : i64
    %c32768_i64 = arith.constant 32768 : i64
    %c4352_i64 = arith.constant 4352 : i64
    %c32_i64 = arith.constant 32 : i64
    %c0_i32 = arith.constant 0 : i32
    %c128_i64 = arith.constant 128 : i64
    %c127_i64 = arith.constant 127 : i64
    %alloc = memref.alloc() {cannir.arg_index = 6 : i64} : memref<1xf32>
    %alloc_0 = memref.alloc() {cannir.arg_index = 7 : i64} : memref<1xf32>
    %alloc_1 = memref.alloc() {cannir.arg_index = 8 : i64} : memref<1088xi32>
    %alloc_2 = memref.alloc() {cannir.arg_index = 9 : i64} : memref<1xui8>
    %alloc_3 = memref.alloc() {cannir.arg_index = 10 : i64} : memref<1xui8>
    %alloc_4 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<4096xui8, 1>
    %alloc_5 = memref.alloc() {cannir.byte_offset = 4096 : i64} : memref<256xui8, 1>
    %alloc_6 = memref.alloc() {cannir.byte_offset = 4352 : i64} : memref<256xf32, 1>
    %alloc_7 = memref.alloc() {cannir.byte_offset = 5376 : i64} : memref<256xi32, 1>
    %alloc_8 = memref.alloc() {cannir.byte_offset = 6400 : i64} : memref<1088xi32, 1>
    %alloc_9 = memref.alloc() {cannir.byte_offset = 10752 : i64} : memref<4096xf32, 1>
    %alloc_10 = memref.alloc() {cannir.byte_offset = 27136 : i64} : memref<4096xf32, 1>
    %alloc_11 = memref.alloc() {cannir.byte_offset = 43520 : i64} : memref<8192xf32, 1>
    %alloc_12 = memref.alloc() {cannir.byte_offset = 76288 : i64} : memref<8192xf32, 1>
    ascvec.sync_intra_arrive v, %c1_i64
    ascvec.sync_intra_arrive v, %c0_i64
    %0 = ascscalar.get_subblock_id
    %1 = arith.addi %arg13, %c127_i64 : i64
    %2 = arith.remsi %1, %c128_i64 : i64
    %3 = arith.cmpi ne, %2, %c0_i64 : i64
    %4 = arith.cmpi slt, %2, %c0_i64 : i64
    %5 = arith.andi %3, %4 : i1
    %6 = arith.divsi %1, %c128_i64 : i64
    %7 = arith.extui %5 : i1 to i64
    %8 = arith.subi %6, %7 : i64
    %9 = ascscalar.get_block_idx
    %10 = ascscalar.get_block_num
    %11 = scf.for %arg15 = %9 to %8 step %10 iter_args(%arg16 = %c0_i64) -> (i64)  : i64 {
      ascvec.get_buf mte2, %c2_i64, 0 {cannir.buffer.generated_sync}
      "ascvec.copy_gm2ub_plan"(%alloc_8, %alloc_1, %c0_i32, %c0_i32, %c0_i64, %c0_i64, %c1_i64, %c4352_i64, %c0_i64, %c4352_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 4352 : i64, pad_storage_bits = false, storage_dtype = i32, ub_base = 6400 : i64, ub_capacity = 4352 : i64}> : (memref<1088xi32, 1>, memref<1088xi32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
      ascvec.release_buf mte2, %c2_i64, 0 {cannir.buffer.generated_sync}
      %12 = cannir.extract_layout %arg7 : !cannir.tensor<<gm, f32>, <<(-1, 64)>, <(64, 1)>>, <<(-1, 64)>, <(64, 1)>>, identity>
      %13 = cannir.extract_shape %12 : <<(-1, 64)>, <(64, 1)>> -> <(-1, 64)>
      %14 = cannir.extract_dim %13[0] : !cannir.shape<(-1, 64)>
      %15 = arith.divsi %14, %c128_i64 : i64
      %16 = arith.minsi %arg15, %15 : i64
      %17 = arith.muli %16, %c128_i64 : i64
      %18 = arith.subi %14, %17 : i64
      %19 = arith.minsi %18, %c128_i64 : i64
      %20 = arith.cmpi sle, %arg15, %15 : i64
      %21 = arith.select %20, %19, %c0_i64 : i64
      %22 = arith.divsi %21, %c2_i64 : i64
      %23 = arith.remsi %21, %c2_i64 : i64
      %24 = arith.addi %23, %c1_i64 overflow<nsw> : i64
      %25 = arith.divsi %24, %c2_i64 : i64
      %26 = arith.addi %22, %25 : i64
      %27 = arith.maxsi %26, %c1_i64 : i64
      %28 = arith.divsi %21, %27 : i64
      %29 = arith.minsi %0, %28 : i64
      %30 = arith.muli %29, %26 : i64
      %31 = arith.subi %21, %30 : i64
      %32 = arith.minsi %26, %31 : i64
      %33 = arith.cmpi sle, %0, %28 : i64
      %34 = arith.select %33, %32, %c0_i64 : i64
      %35 = arith.muli %arg15, %c32768_i64 : i64
      %36 = arith.divui %21, %c2_i64 : i64
      %37 = arith.remui %21, %c2_i64 : i64
      %38 = arith.addi %37, %c1_i64 overflow<nsw> : i64
      %39 = arith.divui %38, %c2_i64 : i64
      %40 = arith.addi %36, %39 : i64
      %41 = arith.muli %0, %40 : i64
      %42 = arith.muli %41, %c256_i64 : i64
      %43 = arith.addi %35, %42 : i64
      %44 = arith.muli %34, %c256_i64 : i64
      %45 = arith.cmpi eq, %44, %c0_i64 : i64
      %46 = arith.cmpi ne, %44, %c0_i64 : i64
      %47 = arith.extui %46 : i1 to i64
      %48 = arith.select %45, %c4_i64, %44 : i64
      "ascvec.copy_gm2ub_plan"(%alloc_9, %alloc_0, %c0_i32, %c0_i32, %c0_i64, %43, %47, %48, %c0_i64, %48, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 10752 : i64, ub_capacity = 16384 : i64}> : (memref<4096xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
      %49 = cannir.extract_layout %arg6 : !cannir.tensor<<gm, f32>, <<(-1, 64)>, <(64, 1)>>, <<(-1, 64)>, <(64, 1)>>, identity>
      %50 = cannir.extract_shape %49 : <<(-1, 64)>, <(64, 1)>> -> <(-1, 64)>
      %51 = cannir.extract_dim %50[0] : !cannir.shape<(-1, 64)>
      %52 = arith.divsi %51, %c128_i64 : i64
      %53 = arith.minsi %arg15, %52 : i64
      %54 = arith.muli %53, %c128_i64 : i64
      %55 = arith.subi %51, %54 : i64
      %56 = arith.minsi %55, %c128_i64 : i64
      %57 = arith.cmpi sle, %arg15, %52 : i64
      %58 = arith.select %57, %56, %c0_i64 : i64
      %59 = arith.divsi %58, %c2_i64 : i64
      %60 = arith.remsi %58, %c2_i64 : i64
      %61 = arith.addi %60, %c1_i64 overflow<nsw> : i64
      %62 = arith.divsi %61, %c2_i64 : i64
      %63 = arith.addi %59, %62 : i64
      %64 = arith.maxsi %63, %c1_i64 : i64
      %65 = arith.divsi %58, %64 : i64
      %66 = arith.minsi %0, %65 : i64
      %67 = arith.muli %66, %63 : i64
      %68 = arith.subi %58, %67 : i64
      %69 = arith.minsi %63, %68 : i64
      %70 = arith.cmpi sle, %0, %65 : i64
      %71 = arith.select %70, %69, %c0_i64 : i64
      %72 = arith.divui %58, %c2_i64 : i64
      %73 = arith.remui %58, %c2_i64 : i64
      %74 = arith.addi %73, %c1_i64 overflow<nsw> : i64
      %75 = arith.divui %74, %c2_i64 : i64
      %76 = arith.addi %72, %75 : i64
      %77 = arith.muli %0, %76 : i64
      %78 = arith.muli %77, %c256_i64 : i64
      %79 = arith.addi %35, %78 : i64
      %80 = arith.muli %71, %c256_i64 : i64
      %81 = arith.cmpi eq, %80, %c0_i64 : i64
      %82 = arith.cmpi ne, %80, %c0_i64 : i64
      %83 = arith.extui %82 : i1 to i64
      %84 = arith.select %81, %c4_i64, %80 : i64
      ascvec.get_buf mte2, %c4_i64, 0 {cannir.buffer.generated_sync}
      "ascvec.copy_gm2ub_plan"(%alloc_10, %alloc, %c0_i32, %c0_i32, %c0_i64, %79, %83, %84, %c0_i64, %84, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 27136 : i64, ub_capacity = 16384 : i64}> : (memref<4096xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
      ascvec.release_buf mte2, %c4_i64, 0 {cannir.buffer.generated_sync}
      %85 = arith.addi %arg16, %c1_i64 overflow<nsw> : i64
      %86 = arith.remui %85, %c2_i64 : i64
      %87 = cannir.extract_layout %arg9 : !cannir.tensor<<gm, ui8>, <<(-1, 2048)>, <(2048, 1)>>, <<(-1, 2048)>, <(2048, 1)>>, identity>
      %88 = cannir.extract_shape %87 : <<(-1, 2048)>, <(2048, 1)>> -> <(-1, 2048)>
      %89 = cannir.extract_dim %88[0] : !cannir.shape<(-1, 2048)>
      %90 = arith.divsi %89, %c128_i64 : i64
      %91 = arith.minsi %arg15, %90 : i64
      %92 = arith.muli %91, %c128_i64 : i64
      %93 = arith.subi %89, %92 : i64
      %94 = arith.minsi %93, %c128_i64 : i64
      %95 = arith.cmpi sle, %arg15, %90 : i64
      %96 = arith.select %95, %94, %c0_i64 : i64
      %97 = arith.divsi %96, %c2_i64 : i64
      %98 = arith.remsi %96, %c2_i64 : i64
      %99 = arith.addi %98, %c1_i64 overflow<nsw> : i64
      %100 = arith.divsi %99, %c2_i64 : i64
      %101 = arith.addi %97, %100 : i64
      %102 = arith.maxsi %101, %c1_i64 : i64
      %103 = arith.divsi %96, %102 : i64
      %104 = arith.minsi %0, %103 : i64
      %105 = arith.muli %104, %101 : i64
      %106 = arith.subi %96, %105 : i64
      %107 = arith.minsi %101, %106 : i64
      %108 = arith.cmpi sle, %0, %103 : i64
      %109 = arith.select %108, %107, %c0_i64 : i64
      %110 = scf.for %arg17 = %c0_i64 to %c32_i64 step %c1_i64 iter_args(%arg18 = %c0_i64) -> (i64)  : i64 {
        %111 = arith.cmpi eq, %arg18, %c0_i64 : i64
        %112 = arith.addi %arg18, %c1_i64 overflow<nsw> : i64
        %113 = arith.remui %112, %c2_i64 : i64
        %114 = ascscalar.memref_address %alloc_11 : memref<8192xf32, 1>
        %115 = ascscalar.memref_address %alloc_12 : memref<8192xf32, 1>
        %116 = arith.select %111, %114, %115 : i64
        %117 = arith.select %111, %c25_i64, %c26_i64 : i64
        %118 = arith.cmpi eq, %117, %c26_i64 : i64
        %119 = arith.extui %118 : i1 to i64
        %120 = ascscalar.bind_local_ptr %116 : memref<8192xf32, 1>
        %121 = arith.trunci %109 : i64 to i32
        ascvec.sync_intra_wait v, %119 {cannir.sync.banks = 3 : i32}
        ascvec.get_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
        ascvec.vf(%120, %alloc_6, %alloc_9, %alloc_10, %121 : memref<8192xf32, 1>, memref<256xf32, 1>, memref<4096xf32, 1>, memref<4096xf32, 1>, i32) {
        ^bb0(%arg19: memref<8192xf32, 1>, %arg20: memref<256xf32, 1>, %arg21: memref<4096xf32, 1>, %arg22: memref<4096xf32, 1>, %arg23: i32):
          %c3_i32 = arith.constant 3 : i32
          %c2_i32 = arith.constant 2 : i32
          %c64_i32_13 = arith.constant 64 : i32
          %c4_i32_14 = arith.constant 4 : i32
          %c128_i32 = arith.constant 128 : i32
          %cst = arith.constant -1.000000e+00 : f32
          %cst_15 = arith.constant 1.000000e+00 : f32
          %c63_i32 = arith.constant 63 : i32
          %c32_i32 = arith.constant 32 : i32
          %c1_i32 = arith.constant 1 : i32
          %c0_i32_16 = arith.constant 0 : i32
          %162 = ascvec.create_mask : !ascvec.mask
          %163 = ascvec.create_mask pat_mode "PAT_H" : !ascvec.mask
          %164 = ascvec.arange %c0_i32_16 : i32 -> <i32>
          %165 = ascvec.duplicate_scalar %c1_i32, %162 : i32 -> <i32>
          %166 = ascvec.xor %164, %165, %162 : <i32>, <i32> -> <i32>
          %167 = ascvec.reinterpret %166 : <i32> -> <ui32>
          %168 = ascvec.add_scalar %164, %c32_i32, %162 : <i32>, i32
          %169 = ascvec.duplicate_scalar %c63_i32, %162 : i32 -> <i32>
          %170 = ascvec.and %168, %169, %162 : <i32>, <i32> -> <i32>
          %171 = ascvec.reinterpret %170 : <i32> -> <ui32>
          %172 = ascvec.duplicate_scalar %cst_15, %162 : f32 -> <f32>
          %173 = ascvec.duplicate_scalar %cst, %162 : f32 -> <f32>
          %174 = ascvec.and %164, %165, %162 : <i32>, <i32> -> <i32>
          %175 = ascvec.duplicate_scalar %c0_i32_16, %162 : i32 -> <i32>
          %176 = ascvec.cmp_ne %174, %175, %162 : <i32>, <i32> -> !ascvec.mask
          %177 = ascvec.select %172, %173, %176 : <f32>
          scf.for %arg24 = %c0_i32_16 to %arg23 step %c1_i32  : i32 {
            %178 = arith.muli %arg24, %c128_i32 : i32
            %179 = arith.index_cast %178 : i32 to index
            %180 = ascvec.loadalign %arg19[%179] : memref<8192xf32, 1>, index -> <f32>
            %181 = ascvec.abs %180, %162 : <f32>
            %182 = ascvec.reinterpret %181 : <f32> -> <i32>
            %183 = ascvec.gather_reg %182, %171 : <i32>, <ui32> -> <i32>
            %184 = ascvec.reinterpret %183 : <i32> -> <f32>
            %185 = arith.muli %arg24, %c4_i32_14 : i32
            %186 = ascvec.reduce_max %181, %163 : <f32>
            %187 = arith.index_cast %185 : i32 to index
            ascvec.storealign_1st %arg20[%187], %186 : memref<256xf32, 1>, index, <f32>
            %188 = arith.addi %185, %c1_i32 : i32
            %189 = ascvec.reduce_max %184, %163 : <f32>
            %190 = arith.index_cast %188 : i32 to index
            ascvec.storealign_1st %arg20[%190], %189 : memref<256xf32, 1>, index, <f32>
            %191 = arith.addi %178, %c64_i32_13 : i32
            %192 = arith.index_cast %191 : i32 to index
            %193 = ascvec.loadalign %arg19[%192] : memref<8192xf32, 1>, index -> <f32>
            %194 = arith.muli %arg24, %c64_i32_13 : i32
            %195 = arith.index_cast %194 : i32 to index
            %196 = ascvec.loadalign %arg21[%195] : memref<4096xf32, 1>, index -> <f32>
            %197 = ascvec.loadalign %arg22[%195] : memref<4096xf32, 1>, index -> <f32>
            %198 = ascvec.reinterpret %193 : <f32> -> <i32>
            %199 = ascvec.gather_reg %198, %167 : <i32>, <ui32> -> <i32>
            %200 = ascvec.reinterpret %199 : <i32> -> <f32>
            %201 = ascvec.mul %200, %177, %162 : <f32>
            %202 = ascvec.mul %193, %196, %162 : <f32>
            %203 = ascvec.madd_two_address %201, %197, %202, %162 : <f32>
            ascvec.storealign %arg19[%192], %203, %162 : memref<8192xf32, 1>, index, <f32>
            %204 = ascvec.abs %203, %162 : <f32>
            %205 = ascvec.reinterpret %204 : <f32> -> <i32>
            %206 = ascvec.gather_reg %205, %171 : <i32>, <ui32> -> <i32>
            %207 = ascvec.reinterpret %206 : <i32> -> <f32>
            %208 = arith.addi %185, %c2_i32 : i32
            %209 = ascvec.reduce_max %204, %163 : <f32>
            %210 = arith.index_cast %208 : i32 to index
            ascvec.storealign_1st %arg20[%210], %209 : memref<256xf32, 1>, index, <f32>
            %211 = arith.addi %185, %c3_i32 : i32
            %212 = ascvec.reduce_max %207, %163 : <f32>
            %213 = arith.index_cast %211 : i32 to index
            ascvec.storealign_1st %arg20[%213], %212 : memref<256xf32, 1>, index, <f32>
          }
          ascvec.mem_bar vst_vld
        } {cannir.vf_mode = "simd"}
        ascvec.release_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
        ascvec.get_buf v, %c1_i64, 0 {cannir.buffer.generated_sync}
        ascvec.vf(%alloc_6, %alloc_7, %alloc_5 : memref<256xf32, 1>, memref<256xi32, 1>, memref<256xui8, 1>) {
        ^bb0(%arg19: memref<256xf32, 1>, %arg20: memref<256xi32, 1>, %arg21: memref<256xui8, 1>):
          %c4_i32_13 = arith.constant 4 : i32
          %c-2_i32 = arith.constant -2 : i32
          %c4194304_i32 = arith.constant 4194304 : i32
          %c64_i32_14 = arith.constant 64 : i32
          %c8388607_i32 = arith.constant 8388607 : i32
          %c127_i32 = arith.constant 127 : i32
          %c1_i32 = arith.constant 1 : i32
          %c0_i32_15 = arith.constant 0 : i32
          %162 = ascvec.create_mask : !ascvec.mask
          %163 = ascvec.duplicate_scalar %c0_i32_15, %162 : i32 -> <i32>
          %164 = ascvec.duplicate_scalar %c1_i32, %162 : i32 -> <i32>
          %165 = ascvec.duplicate_scalar %c127_i32, %162 : i32 -> <i32>
          %166 = ascvec.duplicate_scalar %c8388607_i32, %162 : i32 -> <i32>
          scf.for %arg22 = %c0_i32_15 to %c4_i32_13 step %c1_i32  : i32 {
            %167 = arith.muli %arg22, %c64_i32_14 overflow<nsw> : i32
            %168 = arith.index_cast %167 : i32 to index
            %169 = ascvec.loadalign %arg19[%168] : memref<256xf32, 1>, index -> <f32>
            %170 = ascvec.reinterpret %169 : <f32> -> <i32>
            %171 = ascvec.shr %170, 23, %162 : <i32> -> <i32>
            %172 = ascvec.and %170, %166, %162 : <i32>, <i32> -> <i32>
            %173 = ascvec.cmp_gt_scalar %172, %c4194304_i32, %162 : <i32>, i32 -> !ascvec.mask
            %174 = ascvec.select %164, %163, %173 : <i32>
            %175 = ascvec.add_scalar %171, %c-2_i32, %162 : <i32>, i32
            %176 = ascvec.add %175, %174, %162 : <i32>
            %177 = ascvec.max_scalar %176, %c1_i32, %162 : <i32>, i32
            %178 = ascvec.cmp_ne %170, %163, %162 : <i32>, <i32> -> !ascvec.mask
            %179 = ascvec.select %177, %165, %178 : <i32>
            %180 = ascvec.sub %179, %165, %162 : <i32>
            %181 = ascvec.shl %180, 3, %162 : <i32> -> <i32>
            ascvec.storealign %arg20[%168], %181, %162 : memref<256xi32, 1>, index, <i32>
            ascvec.storealign_pack %arg21[%168], %179, %162 {mode = "b32_to_b8"} : memref<256xui8, 1>, index, <i32>
          }
          ascvec.mem_bar vst_vld
        } {cannir.vf_mode = "simd"}
        ascvec.release_buf v, %c1_i64, 0 {cannir.buffer.generated_sync}
        ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
        ascvec.get_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
        ascvec.vf(%alloc_7, %120, %alloc_8, %alloc_4, %121 : memref<256xi32, 1>, memref<8192xf32, 1>, memref<1088xi32, 1>, memref<4096xui8, 1>, i32) {
        ^bb0(%arg19: memref<256xi32, 1>, %arg20: memref<8192xf32, 1>, %arg21: memref<1088xi32, 1>, %arg22: memref<4096xui8, 1>, %arg23: i32):
          %c32_i32 = arith.constant 32 : i32
          %c64_i32_13 = arith.constant 64 : i32
          %c2_i32 = arith.constant 2 : i32
          %cst = arith.constant 0.000000e+00 : f32
          %c1034_i32 = arith.constant 1034 : i32
          %c1026_i32 = arith.constant 1026 : i32
          %c1018_i32 = arith.constant 1018 : i32
          %c1000_i32 = arith.constant 1000 : i32
          %c0_14 = arith.constant 0 : index
          %c4_i32_15 = arith.constant 4 : i32
          %c128_i32 = arith.constant 128 : i32
          %cst_16 = arith.constant 1.000000e+00 : f32
          %cst_17 = arith.constant 1.600000e+01 : f32
          %c1_i32 = arith.constant 1 : i32
          %c1048575_i32 = arith.constant 1048575 : i32
          %c2147483647_i32 = arith.constant 2147483647 : i32
          %c8_i32 = arith.constant 8 : i32
          %c0_i32_18 = arith.constant 0 : i32
          %162 = ascvec.create_mask : !ascvec.mask
          %163 = ascvec.create_mask pat_mode "PAT_H" : !ascvec.mask
          %164 = ascvec.arange %c0_i32_18 : i32 -> <i32>
          %165 = ascvec.duplicate_scalar %c0_i32_18, %162 : i32 -> <i32>
          %166 = ascvec.duplicate_scalar %c8_i32, %162 : i32 -> <i32>
          %167 = ascvec.duplicate_scalar %c2147483647_i32, %162 : i32 -> <i32>
          %168 = ascvec.duplicate_scalar %c1048575_i32, %162 : i32 -> <i32>
          %169 = ascvec.duplicate_scalar %c1_i32, %162 : i32 -> <i32>
          %170 = ascvec.and %164, %169, %162 : <i32>, <i32> -> <i32>
          %171 = ascvec.cmp_ne %170, %165, %162 : <i32>, <i32> -> !ascvec.mask
          %172 = ascvec.duplicate_scalar %cst_17, %162 : f32 -> <f32>
          %173 = ascvec.duplicate_scalar %cst_16, %162 : f32 -> <f32>
          %174 = ascvec.select %172, %173, %171 : <f32>
          %175 = ascvec.shr %164, 5, %162 : <i32> -> <i32>
          scf.for %arg24 = %c0_i32_18 to %arg23 step %c1_i32  : i32 {
            %176 = arith.muli %arg24, %c128_i32 : i32
            %177 = arith.muli %arg24, %c4_i32_15 : i32
            %178 = ascvec.add_scalar %175, %177, %162 : <i32>, i32
            %179 = ascvec.add_scalar %178, %c0_i32_18, %162 : <i32>, i32
            %180 = ascvec.reinterpret %179 : <i32> -> <ui32>
            %181 = ascvec.gather %arg19[%c0_14], %180, %162 : memref<256xi32, 1>, <ui32>, !ascvec.mask -> <i32>
            %182 = arith.index_cast %176 : i32 to index
            %183 = ascvec.loadalign %arg20[%182] : memref<8192xf32, 1>, index -> <f32>
            %184 = ascvec.reinterpret %183 : <f32> -> <i32>
            %185 = ascvec.and %184, %167, %162 : <i32>, <i32> -> <i32>
            %186 = ascvec.shr %185, 20, %162 : <i32> -> <i32>
            %187 = ascvec.sub %186, %181, %162 : <i32>
            %188 = ascvec.max_scalar %187, %c0_i32_18, %162 : <i32>, i32
            %189 = ascvec.reinterpret %188 : <i32> -> <ui32>
            %190 = ascvec.gather %arg21[%c0_14], %189, %162 : memref<1088xi32, 1>, <ui32>, !ascvec.mask -> <i32>
            %191 = ascvec.duplicate_scalar %c1000_i32, %162 : i32 -> <i32>
            %192 = ascvec.cmp_eq %188, %191, %162 : <i32>, <i32> -> !ascvec.mask
            %193 = ascvec.select %169, %165, %192 : <i32>
            %194 = ascvec.duplicate_scalar %c1018_i32, %162 : i32 -> <i32>
            %195 = ascvec.cmp_eq %188, %194, %162 : <i32>, <i32> -> !ascvec.mask
            %196 = ascvec.select %169, %193, %195 : <i32>
            %197 = ascvec.duplicate_scalar %c1026_i32, %162 : i32 -> <i32>
            %198 = ascvec.cmp_eq %188, %197, %162 : <i32>, <i32> -> !ascvec.mask
            %199 = ascvec.select %169, %196, %198 : <i32>
            %200 = ascvec.duplicate_scalar %c1034_i32, %162 : i32 -> <i32>
            %201 = ascvec.cmp_eq %188, %200, %162 : <i32>, <i32> -> !ascvec.mask
            %202 = ascvec.select %169, %199, %201 : <i32>
            %203 = ascvec.sub %190, %169, %162 : <i32>
            %204 = ascvec.and %185, %168, %162 : <i32>, <i32> -> <i32>
            %205 = ascvec.cmp_eq %204, %165, %162 : <i32>, <i32> -> !ascvec.mask
            %206 = ascvec.select %203, %190, %205 : <i32>
            %207 = ascvec.cmp_ne %202, %165, %162 : <i32>, <i32> -> !ascvec.mask
            %208 = ascvec.select %206, %190, %207 : <i32>
            %209 = ascvec.cmp_lt_scalar %183, %cst, %162 : <f32>, f32 -> !ascvec.mask
            %210 = ascvec.select %166, %165, %209 : <i32>
            %211 = ascvec.or %208, %210, %162 : <i32>, <i32> -> <i32>
            %212 = ascvec.cast %211, %162 : <i32> -> <f32>
            %213 = ascvec.mul %212, %174, %162 : <f32>
            %214 = ascvec.pair_reduce_sum %213, %162 : <f32>
            %215 = ascvec.cast %214, %162 : <f32> -> <i32>
            %216 = arith.remsi %176, %c2_i32 : i32
            %217 = arith.cmpi ne, %216, %c0_i32_18 : i32
            %218 = arith.cmpi slt, %216, %c0_i32_18 : i32
            %219 = arith.andi %217, %218 : i1
            %220 = arith.divsi %176, %c2_i32 : i32
            %221 = arith.extui %219 : i1 to i32
            %222 = arith.subi %220, %221 : i32
            %223 = arith.index_cast %222 : i32 to index
            ascvec.storealign_pack %arg22[%223], %215, %163 {mode = "b32_to_b8"} : memref<4096xui8, 1>, index, <i32>
            %224 = ascvec.add_scalar %178, %c2_i32, %162 : <i32>, i32
            %225 = ascvec.reinterpret %224 : <i32> -> <ui32>
            %226 = ascvec.gather %arg19[%c0_14], %225, %162 : memref<256xi32, 1>, <ui32>, !ascvec.mask -> <i32>
            %227 = arith.addi %176, %c64_i32_13 : i32
            %228 = arith.index_cast %227 : i32 to index
            %229 = ascvec.loadalign %arg20[%228] : memref<8192xf32, 1>, index -> <f32>
            %230 = ascvec.reinterpret %229 : <f32> -> <i32>
            %231 = ascvec.and %230, %167, %162 : <i32>, <i32> -> <i32>
            %232 = ascvec.shr %231, 20, %162 : <i32> -> <i32>
            %233 = ascvec.sub %232, %226, %162 : <i32>
            %234 = ascvec.max_scalar %233, %c0_i32_18, %162 : <i32>, i32
            %235 = ascvec.reinterpret %234 : <i32> -> <ui32>
            %236 = ascvec.gather %arg21[%c0_14], %235, %162 : memref<1088xi32, 1>, <ui32>, !ascvec.mask -> <i32>
            %237 = ascvec.cmp_eq %234, %191, %162 : <i32>, <i32> -> !ascvec.mask
            %238 = ascvec.select %169, %165, %237 : <i32>
            %239 = ascvec.cmp_eq %234, %194, %162 : <i32>, <i32> -> !ascvec.mask
            %240 = ascvec.select %169, %238, %239 : <i32>
            %241 = ascvec.cmp_eq %234, %197, %162 : <i32>, <i32> -> !ascvec.mask
            %242 = ascvec.select %169, %240, %241 : <i32>
            %243 = ascvec.cmp_eq %234, %200, %162 : <i32>, <i32> -> !ascvec.mask
            %244 = ascvec.select %169, %242, %243 : <i32>
            %245 = ascvec.sub %236, %169, %162 : <i32>
            %246 = ascvec.and %231, %168, %162 : <i32>, <i32> -> <i32>
            %247 = ascvec.cmp_eq %246, %165, %162 : <i32>, <i32> -> !ascvec.mask
            %248 = ascvec.select %245, %236, %247 : <i32>
            %249 = ascvec.cmp_ne %244, %165, %162 : <i32>, <i32> -> !ascvec.mask
            %250 = ascvec.select %248, %236, %249 : <i32>
            %251 = ascvec.cmp_lt_scalar %229, %cst, %162 : <f32>, f32 -> !ascvec.mask
            %252 = ascvec.select %166, %165, %251 : <i32>
            %253 = ascvec.or %250, %252, %162 : <i32>, <i32> -> <i32>
            %254 = ascvec.cast %253, %162 : <i32> -> <f32>
            %255 = ascvec.mul %254, %174, %162 : <f32>
            %256 = ascvec.pair_reduce_sum %255, %162 : <f32>
            %257 = ascvec.cast %256, %162 : <f32> -> <i32>
            %258 = arith.addi %222, %c32_i32 : i32
            %259 = arith.index_cast %258 : i32 to index
            ascvec.storealign_pack %arg22[%259], %257, %163 {mode = "b32_to_b8"} : memref<4096xui8, 1>, index, <i32>
          }
        } {cannir.vf_mode = "simd"}
        ascvec.release_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
        ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
        ascvec.sync_intra_arrive v, %119 {cannir.sync.banks = 3 : i32}
        %122 = arith.muli %arg15, %c262144_i64 : i64
        %123 = arith.muli %arg17, %c64_i64 overflow<nsw> : i64
        %124 = arith.addi %122, %123 : i64
        %125 = arith.muli %124, %c8_i64 : i64
        %126 = ascscalar.memref_address %alloc_2 : memref<1xui8>
        %127 = arith.divui %125, %c8_i64 : i64
        %128 = arith.addi %126, %127 : i64
        %129 = ascscalar.bind_gm_ptr %128 : memref<1xui8>
        %130 = ascscalar.bitcast_memref %alloc_4 byte_offset = %c0_i64 : memref<4096xui8, 1> -> memref<?xui8, 1>
        %131 = arith.muli %0, %101 : i64
        %132 = arith.index_cast %131 : i64 to index
        %133 = arith.muli %132, %c2048 : index
        %134 = arith.cmpi ugt, %109, %c0_i64 : i64
        scf.if %134 {
          ascvec.get_buf mte3, %c0_i64, 0 {cannir.buffer.generated_sync}
          ascvec.copy_ub2gm %129[%133], %130[%c0], %121, %c64_i32, %c2048_i64, %c64_i32 l2_cache_ctl %c0_i32 : memref<1xui8>, memref<?xui8, 1>
          ascvec.release_buf mte3, %c0_i64, 0 {cannir.buffer.generated_sync}
        }
        %135 = arith.muli %arg15, %c16384_i64 : i64
        %136 = arith.muli %arg17, %c4_i64 overflow<nsw> : i64
        %137 = arith.addi %135, %136 : i64
        %138 = arith.muli %137, %c8_i64 : i64
        %139 = ascscalar.memref_address %alloc_3 : memref<1xui8>
        %140 = arith.divui %138, %c8_i64 : i64
        %141 = arith.addi %139, %140 : i64
        %142 = ascscalar.bind_gm_ptr %141 : memref<1xui8>
        %143 = ascscalar.bitcast_memref %alloc_5 byte_offset = %c0_i64 : memref<256xui8, 1> -> memref<?xui8, 1>
        %144 = cannir.extract_layout %arg10 : !cannir.tensor<<gm, ui8>, <<(-1, 128)>, <(128, 1)>>, <<(-1, 128)>, <(128, 1)>>, identity>
        %145 = cannir.extract_shape %144 : <<(-1, 128)>, <(128, 1)>> -> <(-1, 128)>
        %146 = cannir.extract_dim %145[0] : !cannir.shape<(-1, 128)>
        %147 = arith.divsi %146, %c128_i64 : i64
        %148 = arith.minsi %arg15, %147 : i64
        %149 = arith.muli %148, %c128_i64 : i64
        %150 = arith.subi %146, %149 : i64
        %151 = arith.minsi %150, %c128_i64 : i64
        %152 = arith.cmpi sle, %arg15, %147 : i64
        %153 = arith.select %152, %151, %c0_i64 : i64
        %154 = arith.divsi %153, %c2_i64 : i64
        %155 = arith.remsi %153, %c2_i64 : i64
        %156 = arith.addi %155, %c1_i64 overflow<nsw> : i64
        %157 = arith.divsi %156, %c2_i64 : i64
        %158 = arith.addi %154, %157 : i64
        %159 = arith.muli %0, %158 : i64
        %160 = arith.index_cast %159 : i64 to index
        %161 = arith.muli %160, %c128 : index
        scf.if %134 {
          ascvec.get_buf mte3, %c1_i64, 0 {cannir.buffer.generated_sync}
          ascvec.copy_ub2gm %142[%161], %143[%c0], %121, %c4_i32, %c128_i64, %c4_i32 l2_cache_ctl %c0_i32 : memref<1xui8>, memref<?xui8, 1>
          ascvec.release_buf mte3, %c1_i64, 0 {cannir.buffer.generated_sync}
        }
        scf.yield %113 : i64
      }
      scf.yield %86 : i64
    }
    return
  }
  func.func @kernel_IndexerPrologueQwKernel___call___0_aic(%arg0: !cannir.tensor<<gm, bf16>, <<(-1, 5120)>, <(5120, 1)>>, <<(-1, 5120)>, <(5120, 1)>>, identity>, %arg1: !cannir.tensor<<gm, ui8>, <<(-1, 1280)>, <(1280, 1)>>, <<(-1, 1280)>, <(1280, 1)>>, identity>, %arg2: !cannir.tensor<<gm, ui8>, <<(4096, 1280)>, <(1280, 1)>>, <<(40, 256, 16, 32)>, <(131072, 512, 32, 1)>>, nd2nz> {cannir.physical_shape_upper_bound = array<i64: 40, 256, 16, 32>}, %arg3: !cannir.tensor<<gm, bf16>, <<(32, 5120)>, <(5120, 1)>>, <<(320, 2, 16, 16)>, <(512, 256, 16, 1)>>, nd2nz> {cannir.physical_shape_upper_bound = array<i64: 320, 2, 16, 16>}, %arg4: !cannir.tensor<<gm, ui8>, <<(-1, 20, 2)>, <(40, 2, 1)>>, <<(-1, 20, 2)>, <(40, 2, 1)>>, identity>, %arg5: !cannir.tensor<<gm, ui8>, <<(4096, 20, 2)>, <(40, 2, 1)>>, <<(4096, 20, 2)>, <(40, 2, 1)>>, identity>, %arg6: !cannir.tensor<<gm, f32>, <<(-1, 64)>, <(64, 1)>>, <<(-1, 64)>, <(64, 1)>>, identity>, %arg7: !cannir.tensor<<gm, f32>, <<(-1, 64)>, <(64, 1)>>, <<(-1, 64)>, <(64, 1)>>, identity>, %arg8: !cannir.tensor<<gm, i32>, <<(1088)>, <(1)>>, <<(1088)>, <(1)>>, identity>, %arg9: !cannir.tensor<<gm, ui8>, <<(-1, 2048)>, <(2048, 1)>>, <<(-1, 2048)>, <(2048, 1)>>, identity>, %arg10: !cannir.tensor<<gm, ui8>, <<(-1, 128)>, <(128, 1)>>, <<(-1, 128)>, <(128, 1)>>, identity>, %arg11: !cannir.tensor<<gm, f32>, <<(-1, 32)>, <(32, 1)>>, <<(-1, 32)>, <(32, 1)>>, identity>, %arg12: !cannir.tensor<<gm, f32>, <<(-1, 32)>, <(32, 1)>>, <<(-1, 32)>, <(32, 1)>>, identity>, %arg13: i64, %arg14: f32) attributes {cannir.buffer.auto_sync, cannir.buffer.sync_analyzed, cannir.buffer.sync_verified, cannir.buffer.tensor_state, cannir.core = "aic", cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.device, cannir.explicit_ring_buffer_cursors, cannir.kernel_group = "kernel_IndexerPrologueQwKernel___call___0", cannir.resource.ids_prepared} {
    %c18_i32 = arith.constant 18 : i32
    %c36_i32 = arith.constant 36 : i32
    %c14_i32 = arith.constant 14 : i32
    %c28_i32 = arith.constant 28 : i32
    %c24_i32 = arith.constant 24 : i32
    %c16_i32 = arith.constant 16 : i32
    %c6_i32 = arith.constant 6 : i32
    %c12_i32 = arith.constant 12 : i32
    %c4096_i32 = arith.constant 4096 : i32
    %c80_i32 = arith.constant 80 : i32
    %c5_i32 = arith.constant 5 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %c10_i32 = arith.constant 10 : i32
    %c160_i32 = arith.constant 160 : i32
    %c20_i32 = arith.constant 20 : i32
    %c1280_i32 = arith.constant 1280 : i32
    %c0 = arith.constant 0 : index
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %c2_i64 = arith.constant 2 : i64
    %c3_i64 = arith.constant 3 : i64
    %c4_i64 = arith.constant 4 : i64
    %c5_i64 = arith.constant 5 : i64
    %c6_i64 = arith.constant 6 : i64
    %c7_i64 = arith.constant 7 : i64
    %c8_i64 = arith.constant 8 : i64
    %c9_i64 = arith.constant 9 : i64
    %c10_i64 = arith.constant 10 : i64
    %c11_i64 = arith.constant 11 : i64
    %c12_i64 = arith.constant 12 : i64
    %c13_i64 = arith.constant 13 : i64
    %c14_i64 = arith.constant 14 : i64
    %c15_i64 = arith.constant 15 : i64
    %c16_i64 = arith.constant 16 : i64
    %c17_i64 = arith.constant 17 : i64
    %c18_i64 = arith.constant 18 : i64
    %c19_i64 = arith.constant 19 : i64
    %c20_i64 = arith.constant 20 : i64
    %c25_i64 = arith.constant 25 : i64
    %c26_i64 = arith.constant 26 : i64
    %c2621440 = arith.constant 2621440 : index
    %c10240_i64 = arith.constant 10240 : i64
    %c40960_i64 = arith.constant 40960 : i64
    %c163840_i64 = arith.constant 163840 : i64
    %c131072_i64 = arith.constant 131072 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %c256 = arith.constant 256 : index
    %c512 = arith.constant 512 : index
    %c160_i64 = arith.constant 160 : i64
    %c16 = arith.constant 16 : index
    %c15 = arith.constant 15 : index
    %c1280_i64 = arith.constant 1280 : i64
    %c4096_i64 = arith.constant 4096 : i64
    %c8_i32 = arith.constant 8 : i32
    %c2_i32 = arith.constant 2 : i32
    %c4_i32 = arith.constant 4 : i32
    %c128_i32 = arith.constant 128 : i32
    %c32_i32 = arith.constant 32 : i32
    %c1_i32 = arith.constant 1 : i32
    %c256_i64 = arith.constant 256 : i64
    %c32_i64 = arith.constant 32 : i64
    %c40_i64 = arith.constant 40 : i64
    %c5120_i64 = arith.constant 5120 : i64
    %c0_i32 = arith.constant 0 : i32
    %c128_i64 = arith.constant 128 : i64
    %c127_i64 = arith.constant 127 : i64
    %alloc = memref.alloc() {cannir.arg_index = 0 : i64} : memref<1xbf16>
    %alloc_0 = memref.alloc() {cannir.arg_index = 1 : i64} : memref<1xui8>
    %alloc_1 = memref.alloc() {cannir.arg_index = 2 : i64} : memref<5242880xui8>
    %alloc_2 = memref.alloc() {cannir.arg_index = 3 : i64} : memref<163840xbf16>
    %alloc_3 = memref.alloc() {cannir.arg_index = 4 : i64} : memref<1xui8>
    %alloc_4 = memref.alloc() {cannir.arg_index = 5 : i64} : memref<163840xui8>
    %alloc_5 = memref.alloc() {cannir.arg_index = 11 : i64} : memref<1xf32>
    %alloc_6 = memref.alloc() {cannir.byte_offset = 43520 : i64} : memref<8192xf32, 1>
    %alloc_7 = memref.alloc() {cannir.byte_offset = 76288 : i64} : memref<8192xf32, 1>
    %alloc_8 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<10240xbf16, 3>
    %alloc_9 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<2560xbf16, 4>
    %alloc_10 = memref.alloc() {cannir.byte_offset = 20480 : i64} : memref<16384xf8E4M3FN, 3>
    %alloc_11 = memref.alloc() {cannir.byte_offset = 5120 : i64} : memref<16384xf8E4M3FN, 4>
    %alloc_12 = memref.alloc() {cannir.byte_offset = 36864 : i64} : memref<16384xf8E4M3FN, 3>
    %alloc_13 = memref.alloc() {cannir.byte_offset = 21504 : i64} : memref<16384xf8E4M3FN, 4>
    %alloc_14 = memref.alloc() {cannir.byte_offset = 266240 : i64} : memref<5120x!ascdtype.fp8_e8m0, 2>
    %alloc_15 = memref.alloc() {cannir.byte_offset = 102400 : i64} : memref<163840xf8E4M3FN, 2>
    %alloc_16 = memref.alloc() {cannir.byte_offset = 437760 : i64} : memref<2560x!ascdtype.fp8_e8m0, 2>
    %alloc_17 = memref.alloc() {cannir.byte_offset = 353280 : i64} : memref<81920xui8, 2>
    %alloc_18 = memref.alloc() {cannir.byte_offset = 435200 : i64} : memref<2560x!ascdtype.fp8_e8m0, 2>
    %alloc_19 = memref.alloc() {cannir.byte_offset = 271360 : i64} : memref<81920xui8, 2>
    %alloc_20 = memref.alloc() {cannir.byte_offset = 98304 : i64} : memref<16384xf32, 5>
    %alloc_21 = memref.alloc() {cannir.byte_offset = 32768 : i64} : memref<16384xf32, 5>
    %alloc_22 = memref.alloc() {cannir.byte_offset = 92160 : i64} : memref<5120xbf16, 2>
    %alloc_23 = memref.alloc() {cannir.byte_offset = 81920 : i64} : memref<5120xbf16, 2>
    %alloc_24 = memref.alloc() {cannir.byte_offset = 40960 : i64} : memref<20480xbf16, 2>
    %alloc_25 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<20480xbf16, 2>
    %alloc_26 = memref.alloc() {cannir.byte_offset = 16384 : i64} : memref<4096xf32, 5>
    %alloc_27 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<4096xf32, 5>
    %0 = arith.addi %arg13, %c127_i64 : i64
    %1 = arith.remsi %0, %c128_i64 : i64
    %2 = arith.cmpi ne, %1, %c0_i64 : i64
    %3 = arith.cmpi slt, %1, %c0_i64 : i64
    %4 = arith.andi %2, %3 : i1
    %5 = arith.divsi %0, %c128_i64 : i64
    %6 = arith.extui %4 : i1 to i64
    %7 = arith.subi %5, %6 : i64
    %8 = ascscalar.get_block_idx
    %9 = ascscalar.get_block_num
    %10 = scf.for %arg15 = %8 to %7 step %9 iter_args(%arg16 = %c0_i64) -> (i64)  : i64 {
      %11 = arith.cmpi eq, %arg15, %8 : i64
      scf.if %11 {
        %77 = ascscalar.bitcast_memref %alloc_15 byte_offset = %c0_i64 : memref<163840xf8E4M3FN, 2> -> memref<?xf8E4M3FN, 2>
        %78 = arith.muli %arg15, %c163840_i64 : i64
        %79 = cannir.extract_layout %arg1 : !cannir.tensor<<gm, ui8>, <<(-1, 1280)>, <(1280, 1)>>, <<(-1, 1280)>, <(1280, 1)>>, identity>
        %80 = cannir.extract_shape %79 : <<(-1, 1280)>, <(1280, 1)>> -> <(-1, 1280)>
        %81 = cannir.extract_dim %80[0] : !cannir.shape<(-1, 1280)>
        %82 = arith.divsi %81, %c128_i64 : i64
        %83 = arith.minsi %arg15, %82 : i64
        %84 = arith.muli %83, %c128_i64 : i64
        %85 = arith.subi %81, %84 : i64
        %86 = arith.minsi %85, %c128_i64 : i64
        %87 = arith.cmpi sle, %arg15, %82 : i64
        %88 = arith.select %87, %86, %c0_i64 : i64
        asccube.get_buf mte2, %c8_i64, 0 {cannir.buffer.generated_sync}
        %89 = arith.cmpi ugt, %88, %c0_i64 : i64
        scf.if %89 {
          %103 = arith.index_cast %78 : i64 to index
          %104 = arith.trunci %88 : i64 to i32
          asccube.copy_gm2l1 %77[%c0], %alloc_0[%103], %104, %c1280_i32, %c1280_i64, %c1_i32, %c0_i64, %c0_i64, %c128_i32 l2_cache_ctl %c0_i32 {nd2nz = true} : memref<?xf8E4M3FN, 2>, memref<1xui8>
        }
        asccube.release_buf mte2, %c8_i64, 0 {cannir.buffer.generated_sync}
        %90 = cannir.extract_layout %arg4 : !cannir.tensor<<gm, ui8>, <<(-1, 20, 2)>, <(40, 2, 1)>>, <<(-1, 20, 2)>, <(40, 2, 1)>>, identity>
        %91 = cannir.extract_shape %90 : <<(-1, 20, 2)>, <(40, 2, 1)>> -> <(-1, 20, 2)>
        %92 = cannir.extract_dim %91[0] : !cannir.shape<(-1, 20, 2)>
        %93 = arith.divsi %92, %c128_i64 : i64
        %94 = arith.minsi %arg15, %93 : i64
        %95 = arith.muli %94, %c128_i64 : i64
        %96 = arith.subi %92, %95 : i64
        %97 = arith.minsi %96, %c128_i64 : i64
        %98 = arith.cmpi sle, %arg15, %93 : i64
        %99 = arith.select %98, %97, %c0_i64 : i64
        %100 = arith.muli %arg15, %c5120_i64 : i64
        %101 = ascscalar.bitcast_memref %alloc_14 byte_offset = %c0_i64 : memref<5120x!ascdtype.fp8_e8m0, 2> -> memref<?x!ascdtype.fp8_e8m0, 2>
        asccube.get_buf mte2, %c9_i64, 0 {cannir.buffer.generated_sync}
        %102 = arith.cmpi ugt, %99, %c0_i64 : i64
        scf.if %102 {
          %103 = arith.index_cast %100 : i64 to index
          %104 = arith.trunci %99 : i64 to i32
          asccube.copy_gm2l1 %101[%c0], %alloc_3[%103], %c20_i32, %104, %c40_i64, %c1_i32, %c0_i64, %c0_i64, %c20_i32 l2_cache_ctl %c0_i32 {dn2nz = true, mx_scale = true} : memref<?x!ascdtype.fp8_e8m0, 2>, memref<1xui8>
        }
        asccube.release_buf mte2, %c9_i64, 0 {cannir.buffer.generated_sync}
      }
      %12 = arith.cmpi eq, %arg16, %c0_i64 : i64
      %13 = ascscalar.memref_address %alloc_27 : memref<4096xf32, 5>
      %14 = ascscalar.memref_address %alloc_26 : memref<4096xf32, 5>
      %15 = arith.select %12, %13, %14 : i64
      %16 = arith.select %12, %c6_i64, %c7_i64 : i64
      %17 = arith.addi %arg16, %c1_i64 overflow<nsw> : i64
      %18 = arith.remui %17, %c2_i64 : i64
      %19 = cannir.extract_layout %arg11 : !cannir.tensor<<gm, f32>, <<(-1, 32)>, <(32, 1)>>, <<(-1, 32)>, <(32, 1)>>, identity>
      %20 = cannir.extract_shape %19 : <<(-1, 32)>, <(32, 1)>> -> <(-1, 32)>
      %21 = cannir.extract_dim %20[0] : !cannir.shape<(-1, 32)>
      %22 = arith.divsi %21, %c128_i64 : i64
      %23 = arith.minsi %arg15, %22 : i64
      %24 = arith.muli %23, %c128_i64 : i64
      %25 = arith.subi %21, %24 : i64
      %26 = arith.minsi %25, %c128_i64 : i64
      %27 = arith.cmpi sle, %arg15, %22 : i64
      %28 = arith.select %27, %26, %c0_i64 : i64
      %29 = arith.addi %28, %c15_i64 : i64
      %30 = arith.remsi %29, %c16_i64 : i64
      %31 = arith.cmpi ne, %30, %c0_i64 : i64
      %32 = arith.cmpi slt, %30, %c0_i64 : i64
      %33 = arith.andi %31, %32 : i1
      %34 = arith.divsi %29, %c16_i64 : i64
      %35 = arith.extui %33 : i1 to i64
      %36 = arith.subi %34, %35 : i64
      %37 = arith.muli %36, %c16_i64 : i64
      %38 = cannir.extract_layout %arg9 : !cannir.tensor<<gm, ui8>, <<(-1, 2048)>, <(2048, 1)>>, <<(-1, 2048)>, <(2048, 1)>>, identity>
      %39 = cannir.extract_shape %38 : <<(-1, 2048)>, <(2048, 1)>> -> <(-1, 2048)>
      %40 = cannir.extract_dim %39[0] : !cannir.shape<(-1, 2048)>
      %41 = arith.divsi %40, %c128_i64 : i64
      %42 = arith.minsi %arg15, %41 : i64
      %43 = arith.muli %42, %c128_i64 : i64
      %44 = arith.subi %40, %43 : i64
      %45 = arith.minsi %44, %c128_i64 : i64
      %46 = arith.cmpi sle, %arg15, %41 : i64
      %47 = arith.select %46, %45, %c0_i64 : i64
      %48 = arith.divsi %47, %c2_i64 : i64
      %49 = arith.remsi %47, %c2_i64 : i64
      %50 = arith.addi %49, %c1_i64 overflow<nsw> : i64
      %51 = arith.divsi %50, %c2_i64 : i64
      %52 = arith.addi %48, %51 : i64
      %53 = arith.maxsi %52, %c1_i64 : i64
      %54 = arith.divsi %47, %53 : i64
      %55 = arith.minsi %54, %c0_i64 : i64
      %56 = arith.muli %55, %52 : i64
      %57 = arith.subi %47, %56 : i64
      %58 = arith.minsi %52, %57 : i64
      %59 = arith.cmpi sge, %54, %c0_i64 : i64
      %60 = arith.select %59, %58, %c0_i64 : i64
      %61 = arith.muli %60, %c2_i64 : i64
      %62 = scf.for %arg17 = %c0_i64 to %c32_i64 step %c1_i64 iter_args(%arg18 = %c0_i64) -> (i64)  : i64 {
        %77 = arith.cmpi eq, %arg18, %c0_i64 : i64
        %78 = ascscalar.memref_address %alloc_25 : memref<20480xbf16, 2>
        %79 = ascscalar.memref_address %alloc_24 : memref<20480xbf16, 2>
        %80 = arith.select %77, %78, %79 : i64
        %81 = arith.cmpi ne, %arg18, %c0_i64 : i64
        %82 = arith.extui %81 : i1 to i64
        %83 = arith.addi %arg18, %c1_i64 overflow<nsw> : i64
        %84 = arith.remui %83, %c2_i64 : i64
        %85 = ascscalar.bind_local_ptr %80 : memref<1xbf16, 2>
        %86 = arith.muli %arg15, %c40960_i64 : i64
        %87 = arith.muli %arg17, %c10_i64 overflow<nsw> : i64
        %88 = arith.addi %86, %87 : i64
        %89 = arith.muli %88, %c16_i64 : i64
        %90 = cannir.extract_layout %arg0 : !cannir.tensor<<gm, bf16>, <<(-1, 5120)>, <(5120, 1)>>, <<(-1, 5120)>, <(5120, 1)>>, identity>
        %91 = cannir.extract_shape %90 : <<(-1, 5120)>, <(5120, 1)>> -> <(-1, 5120)>
        %92 = cannir.extract_dim %91[0] : !cannir.shape<(-1, 5120)>
        %93 = arith.divsi %92, %c128_i64 : i64
        %94 = arith.minsi %arg15, %93 : i64
        %95 = arith.muli %94, %c128_i64 : i64
        %96 = arith.subi %92, %95 : i64
        %97 = arith.minsi %96, %c128_i64 : i64
        %98 = arith.cmpi sle, %arg15, %93 : i64
        %99 = arith.select %98, %97, %c0_i64 : i64
        asccube.get_buf mte2, %82, 0 {cannir.buffer.generated_sync}
        %100 = arith.cmpi ugt, %99, %c0_i64 : i64
        scf.if %100 {
          %190 = arith.index_cast %89 : i64 to index
          %191 = arith.trunci %99 : i64 to i32
          asccube.copy_gm2l1 %85[%c0], %alloc[%190], %191, %c160_i32, %c10240_i64, %c1_i32, %c0_i64, %c0_i64, %c128_i32 l2_cache_ctl %c1_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
        }
        asccube.release_buf mte2, %82, 0 {cannir.buffer.generated_sync}
        %101 = ascscalar.memref_address %alloc_23 : memref<5120xbf16, 2>
        %102 = ascscalar.memref_address %alloc_22 : memref<5120xbf16, 2>
        %103 = arith.select %77, %101, %102 : i64
        %104 = arith.select %77, %c2_i64, %c3_i64 : i64
        %105 = ascscalar.bind_local_ptr %103 : memref<1xbf16, 2>
        %106 = arith.muli %arg17, %c160_i64 overflow<nsw> : i64
        %107 = arith.index_cast %106 : i64 to index
        %108 = arith.divui %107, %c16 : index
        %109 = arith.muli %108, %c512 : index
        asccube.get_buf mte2, %104, 0 {cannir.buffer.generated_sync}
        asccube.copy_gm2l1_burst %105[%c0], %alloc_2[%109], %c10_i32, %c1024_i32, %c1024_i64, %c1024_i32 l2_cache_ctl %c1_i32 pad_value %c0_i32 : i32 left_padding %c0_i32 right_padding %c0_i32 : memref<1xbf16, 2>, memref<163840xbf16>
        asccube.release_buf mte2, %104, 0 {cannir.buffer.generated_sync}
        %110 = arith.divsi %37, %c128_i64 : i64
        %111 = arith.minsi %110, %c0_i64 : i64
        %112 = arith.muli %111, %c128_i64 : i64
        %113 = arith.subi %37, %112 : i64
        %114 = arith.minsi %113, %c128_i64 : i64
        %115 = arith.cmpi sge, %110, %c0_i64 : i64
        %116 = arith.select %115, %114, %c0_i64 : i64
        %117 = ascscalar.bitcast_memref %alloc_8 byte_offset = %c0_i64 : memref<10240xbf16, 3> -> memref<?xbf16, 3>
        %118 = arith.index_cast %116 : i64 to index
        %119 = arith.addi %118, %c15 : index
        %120 = arith.divui %119, %c16 : index
        %121 = arith.index_cast %120 : index to i64
        %122 = arith.divsi %37, %c16_i64 : i64
        %123 = arith.remsi %37, %c16_i64 : i64
        %124 = arith.addi %123, %c15_i64 overflow<nsw> : i64
        %125 = arith.divsi %124, %c16_i64 : i64
        %126 = arith.addi %122, %125 : i64
        %127 = arith.muli %126, %c256_i64 : i64
        %128 = arith.index_cast %127 : i64 to index
        %129 = arith.divui %128, %c256 : index
        %130 = arith.index_cast %129 : index to i64
        %131 = arith.cmpi ugt, %121, %c0_i64 : i64
        scf.if %131 {
          asccube.get_buf mte1, %c4_i64, 0 {cannir.buffer.generated_sync}
          asccube.get_buf mte1, %82, 0 {cannir.buffer.generated_sync}
          %190 = arith.trunci %121 : i64 to i32
          %191 = arith.trunci %130 : i64 to i32
          asccube.copy_l12l0a %117[%c0], %85[%c0], %c0_i32, %c0_i32, %190, %c5_i32, %191, %c8_i32 : memref<?xbf16, 3>, memref<1xbf16, 2>
          asccube.release_buf mte1, %82, 0 {cannir.buffer.generated_sync}
          asccube.release_buf mte1, %c4_i64, 0 {cannir.buffer.generated_sync}
        }
        %132 = ascscalar.bitcast_memref %alloc_9 byte_offset = %c0_i64 : memref<2560xbf16, 4> -> memref<?xbf16, 4>
        asccube.get_buf mte1, %c5_i64, 0 {cannir.buffer.generated_sync}
        asccube.get_buf mte1, %104, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0b %132[%c0], %105[%c0], %c0_i32, %c0_i32, %c2_i32, %c5_i32, %c2_i32, %c2_i32 : memref<?xbf16, 4>, memref<1xbf16, 2>
        asccube.release_buf mte1, %104, 0 {cannir.buffer.generated_sync}
        asccube.release_buf mte1, %c5_i64, 0 {cannir.buffer.generated_sync}
        %133 = arith.cmpi eq, %arg17, %c0_i64 : i64
        %134 = ascscalar.bind_local_ptr %15 : memref<1xf32, 5>
        asccube.get_buf m, %16, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %c4_i64, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %c5_i64, 0 {cannir.buffer.generated_sync}
        scf.if %133 {
          asccube.mmad %134[%c0], %117, %132, %c128_i32, %c80_i32, %c32_i32, %c0_i32 {c_matrix_init_val = true} : memref<1xf32, 5>, memref<?xbf16, 3>, memref<?xbf16, 4>
        } else {
          asccube.mmad %134[%c0], %117, %132, %c128_i32, %c80_i32, %c32_i32, %c0_i32 {c_matrix_init_val = false} : memref<1xf32, 5>, memref<?xbf16, 3>, memref<?xbf16, 4>
        }
        asccube.release_buf m, %c5_i64, 0 {cannir.buffer.generated_sync}
        asccube.release_buf m, %c4_i64, 0 {cannir.buffer.generated_sync}
        scf.if %131 {
          asccube.get_buf mte1, %c4_i64, 0 {cannir.buffer.generated_sync}
          asccube.get_buf mte1, %82, 0 {cannir.buffer.generated_sync}
          %190 = arith.trunci %121 : i64 to i32
          %191 = arith.trunci %130 : i64 to i32
          asccube.copy_l12l0a %117[%c0], %85[%c0], %c0_i32, %c5_i32, %190, %c5_i32, %191, %c8_i32 : memref<?xbf16, 3>, memref<1xbf16, 2>
          asccube.release_buf mte1, %82, 0 {cannir.buffer.generated_sync}
          asccube.release_buf mte1, %c4_i64, 0 {cannir.buffer.generated_sync}
        }
        asccube.get_buf mte1, %c5_i64, 0 {cannir.buffer.generated_sync}
        asccube.get_buf mte1, %104, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0b %132[%c0], %105[%c0], %c0_i32, %c5_i32, %c2_i32, %c5_i32, %c2_i32, %c2_i32 : memref<?xbf16, 4>, memref<1xbf16, 2>
        asccube.release_buf mte1, %104, 0 {cannir.buffer.generated_sync}
        asccube.release_buf mte1, %c5_i64, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %c4_i64, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %c5_i64, 0 {cannir.buffer.generated_sync}
        asccube.mmad %134[%c0], %117, %132, %c128_i32, %c80_i32, %c32_i32, %c0_i32 {c_matrix_init_val = false} : memref<1xf32, 5>, memref<?xbf16, 3>, memref<?xbf16, 4>
        asccube.release_buf m, %c5_i64, 0 {cannir.buffer.generated_sync}
        asccube.release_buf m, %c4_i64, 0 {cannir.buffer.generated_sync}
        asccube.release_buf m, %16, 0 {cannir.buffer.generated_sync}
        %135 = ascscalar.memref_address %alloc_21 : memref<16384xf32, 5>
        %136 = ascscalar.memref_address %alloc_20 : memref<16384xf32, 5>
        %137 = arith.select %77, %135, %136 : i64
        %138 = arith.select %77, %c18_i64, %c19_i64 : i64
        %139 = ascscalar.bitcast_memref %alloc_19 byte_offset = %c0_i64 : memref<81920xui8, 2> -> memref<?xui8, 2>
        %140 = arith.muli %arg17, %c128_i64 overflow<nsw> : i64
        %141 = arith.index_cast %140 : i64 to index
        %142 = arith.divui %141, %c16 : index
        %143 = arith.muli %142, %c512 : index
        asccube.get_buf mte2, %c10_i64, 0 {cannir.buffer.generated_sync}
        asccube.copy_gm2l1_burst %139[%c0], %alloc_1[%143], %c20_i32, %c4096_i32, %c131072_i64, %c4096_i32 l2_cache_ctl %c0_i32 pad_value %c0_i32 : i32 left_padding %c0_i32 right_padding %c0_i32 : memref<?xui8, 2>, memref<5242880xui8>
        asccube.release_buf mte2, %c10_i64, 0 {cannir.buffer.generated_sync}
        %144 = arith.muli %arg17, %c5120_i64 overflow<nsw> : i64
        %145 = ascscalar.bitcast_memref %alloc_18 byte_offset = %c0_i64 : memref<2560x!ascdtype.fp8_e8m0, 2> -> memref<?x!ascdtype.fp8_e8m0, 2>
        asccube.get_buf mte2, %c12_i64, 0 {cannir.buffer.generated_sync}
        %146 = arith.index_cast %144 : i64 to index
        asccube.copy_gm2l1 %145[%c0], %alloc_4[%146], %c10_i32, %c128_i32, %c40_i64, %c1_i32, %c0_i64, %c0_i64, %c10_i32 l2_cache_ctl %c0_i32 {dn2nz = true, mx_scale = true} : memref<?x!ascdtype.fp8_e8m0, 2>, memref<163840xui8>
        asccube.release_buf mte2, %c12_i64, 0 {cannir.buffer.generated_sync}
        %147 = ascscalar.bitcast_memref %alloc_17 byte_offset = %c0_i64 : memref<81920xui8, 2> -> memref<?xui8, 2>
        %148 = arith.addi %143, %c2621440 : index
        asccube.get_buf mte2, %c11_i64, 0 {cannir.buffer.generated_sync}
        asccube.copy_gm2l1_burst %147[%c0], %alloc_1[%148], %c20_i32, %c4096_i32, %c131072_i64, %c4096_i32 l2_cache_ctl %c0_i32 pad_value %c0_i32 : i32 left_padding %c0_i32 right_padding %c0_i32 : memref<?xui8, 2>, memref<5242880xui8>
        asccube.release_buf mte2, %c11_i64, 0 {cannir.buffer.generated_sync}
        %149 = arith.addi %144, %c20_i64 overflow<nsw> : i64
        %150 = ascscalar.bitcast_memref %alloc_16 byte_offset = %c0_i64 : memref<2560x!ascdtype.fp8_e8m0, 2> -> memref<?x!ascdtype.fp8_e8m0, 2>
        asccube.get_buf mte2, %c13_i64, 0 {cannir.buffer.generated_sync}
        %151 = arith.index_cast %149 : i64 to index
        asccube.copy_gm2l1 %150[%c0], %alloc_4[%151], %c10_i32, %c128_i32, %c40_i64, %c1_i32, %c0_i64, %c0_i64, %c10_i32 l2_cache_ctl %c0_i32 {dn2nz = true, mx_scale = true} : memref<?x!ascdtype.fp8_e8m0, 2>, memref<163840xui8>
        asccube.release_buf mte2, %c13_i64, 0 {cannir.buffer.generated_sync}
        %152 = ascscalar.bitcast_memref %alloc_10 byte_offset = %c0_i64 : memref<16384xf8E4M3FN, 3> -> memref<?xf8E4M3FN, 3>
        %153 = ascscalar.bitcast_memref %alloc_15 byte_offset = %c0_i64 : memref<163840xf8E4M3FN, 2> -> memref<?xf8E4M3FN, 2>
        %154 = ascscalar.bitcast_memref %alloc_14 byte_offset = %c0_i64 : memref<5120x!ascdtype.fp8_e8m0, 2> -> memref<?x!ascdtype.fp8_e8m0, 2>
        %155 = ascscalar.memref_address %152 : memref<?xf8E4M3FN, 3>
        %156 = ascscalar.bind_local_ptr %155 : memref<?xi8, 6>
        asccube.get_buf mte1, %c14_i64, 0 {cannir.buffer.generated_sync}
        asccube.get_buf mte1, %c8_i64, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0a %152[%c0], %153[%c0], %c0_i32, %c0_i32, %c8_i32, %c4_i32, %c8_i32, %c8_i32 : memref<?xf8E4M3FN, 3>, memref<?xf8E4M3FN, 2>
        asccube.get_buf mte1, %c9_i64, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0a %156[%c0], %154[%c0], %c0_i32, %c0_i32, %c8_i32, %c2_i32, %c20_i32, %c2_i32 : memref<?xi8, 6>, memref<?x!ascdtype.fp8_e8m0, 2>
        asccube.release_buf mte1, %c14_i64, 0 {cannir.buffer.generated_sync}
        %157 = ascscalar.bitcast_memref %alloc_11 byte_offset = %c0_i64 : memref<16384xf8E4M3FN, 4> -> memref<?xf8E4M3FN, 4>
        %158 = ascscalar.bitcast_memref %alloc_19 byte_offset = %c0_i64 : memref<81920xui8, 2> -> memref<?xf8E4M3FN, 2>
        %159 = ascscalar.memref_address %157 : memref<?xf8E4M3FN, 4>
        %160 = ascscalar.bind_local_ptr %159 : memref<?xi8, 7>
        asccube.get_buf mte1, %c16_i64, 0 {cannir.buffer.generated_sync}
        asccube.get_buf mte1, %c10_i64, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0b %157[%c0], %158[%c0], %c0_i32, %c0_i32, %c8_i32, %c4_i32, %c8_i32, %c8_i32 : memref<?xf8E4M3FN, 4>, memref<?xf8E4M3FN, 2>
        asccube.get_buf mte1, %c12_i64, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0b %160[%c0], %145[%c0], %c0_i32, %c0_i32, %c8_i32, %c2_i32, %c10_i32, %c2_i32 : memref<?xi8, 7>, memref<?x!ascdtype.fp8_e8m0, 2>
        asccube.release_buf mte1, %c16_i64, 0 {cannir.buffer.generated_sync}
        %161 = ascscalar.bind_local_ptr %137 : memref<1xf32, 5>
        asccube.get_buf m, %138, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %c14_i64, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %c16_i64, 0 {cannir.buffer.generated_sync}
        asccube.mmad %161[%c0], %152, %157 scale(%156, %160), %c128_i32, %c128_i32, %c128_i32, %c0_i32 {c_matrix_init_val = true, mx_mode = true} : memref<1xf32, 5>, memref<?xf8E4M3FN, 3>, memref<?xf8E4M3FN, 4>, memref<?xi8, 6>, memref<?xi8, 7>
        asccube.release_buf m, %c16_i64, 0 {cannir.buffer.generated_sync}
        asccube.release_buf m, %c14_i64, 0 {cannir.buffer.generated_sync}
        %162 = ascscalar.bitcast_memref %alloc_12 byte_offset = %c0_i64 : memref<16384xf8E4M3FN, 3> -> memref<?xf8E4M3FN, 3>
        %163 = ascscalar.memref_address %162 : memref<?xf8E4M3FN, 3>
        %164 = ascscalar.bind_local_ptr %163 : memref<?xi8, 6>
        asccube.get_buf mte1, %c15_i64, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0a %162[%c0], %153[%c0], %c0_i32, %c4_i32, %c8_i32, %c4_i32, %c8_i32, %c8_i32 : memref<?xf8E4M3FN, 3>, memref<?xf8E4M3FN, 2>
        asccube.copy_l12l0a %164[%c0], %154[%c0], %c0_i32, %c2_i32, %c8_i32, %c2_i32, %c20_i32, %c2_i32 : memref<?xi8, 6>, memref<?x!ascdtype.fp8_e8m0, 2>
        asccube.release_buf mte1, %c15_i64, 0 {cannir.buffer.generated_sync}
        %165 = ascscalar.bitcast_memref %alloc_13 byte_offset = %c0_i64 : memref<16384xf8E4M3FN, 4> -> memref<?xf8E4M3FN, 4>
        %166 = ascscalar.memref_address %165 : memref<?xf8E4M3FN, 4>
        %167 = ascscalar.bind_local_ptr %166 : memref<?xi8, 7>
        asccube.get_buf mte1, %c17_i64, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0b %165[%c0], %158[%c0], %c0_i32, %c4_i32, %c8_i32, %c4_i32, %c8_i32, %c8_i32 : memref<?xf8E4M3FN, 4>, memref<?xf8E4M3FN, 2>
        asccube.copy_l12l0b %167[%c0], %145[%c0], %c0_i32, %c2_i32, %c8_i32, %c2_i32, %c10_i32, %c2_i32 : memref<?xi8, 7>, memref<?x!ascdtype.fp8_e8m0, 2>
        asccube.release_buf mte1, %c17_i64, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %c15_i64, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %c17_i64, 0 {cannir.buffer.generated_sync}
        asccube.mmad %161[%c0], %162, %165 scale(%164, %167), %c128_i32, %c128_i32, %c128_i32, %c0_i32 {c_matrix_init_val = false, mx_mode = true} : memref<1xf32, 5>, memref<?xf8E4M3FN, 3>, memref<?xf8E4M3FN, 4>, memref<?xi8, 6>, memref<?xi8, 7>
        asccube.release_buf m, %c17_i64, 0 {cannir.buffer.generated_sync}
        asccube.release_buf m, %c15_i64, 0 {cannir.buffer.generated_sync}
        asccube.get_buf mte1, %c14_i64, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0a %152[%c0], %153[%c0], %c0_i32, %c8_i32, %c8_i32, %c4_i32, %c8_i32, %c8_i32 : memref<?xf8E4M3FN, 3>, memref<?xf8E4M3FN, 2>
        asccube.copy_l12l0a %156[%c0], %154[%c0], %c0_i32, %c4_i32, %c8_i32, %c2_i32, %c20_i32, %c2_i32 : memref<?xi8, 6>, memref<?x!ascdtype.fp8_e8m0, 2>
        asccube.release_buf mte1, %c14_i64, 0 {cannir.buffer.generated_sync}
        asccube.get_buf mte1, %c16_i64, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0b %157[%c0], %158[%c0], %c0_i32, %c8_i32, %c8_i32, %c4_i32, %c8_i32, %c8_i32 : memref<?xf8E4M3FN, 4>, memref<?xf8E4M3FN, 2>
        asccube.copy_l12l0b %160[%c0], %145[%c0], %c0_i32, %c4_i32, %c8_i32, %c2_i32, %c10_i32, %c2_i32 : memref<?xi8, 7>, memref<?x!ascdtype.fp8_e8m0, 2>
        asccube.release_buf mte1, %c16_i64, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %c14_i64, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %c16_i64, 0 {cannir.buffer.generated_sync}
        asccube.mmad %161[%c0], %152, %157 scale(%156, %160), %c128_i32, %c128_i32, %c128_i32, %c0_i32 {c_matrix_init_val = false, mx_mode = true} : memref<1xf32, 5>, memref<?xf8E4M3FN, 3>, memref<?xf8E4M3FN, 4>, memref<?xi8, 6>, memref<?xi8, 7>
        asccube.release_buf m, %c16_i64, 0 {cannir.buffer.generated_sync}
        asccube.release_buf m, %c14_i64, 0 {cannir.buffer.generated_sync}
        asccube.get_buf mte1, %c15_i64, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0a %162[%c0], %153[%c0], %c0_i32, %c12_i32, %c8_i32, %c4_i32, %c8_i32, %c8_i32 : memref<?xf8E4M3FN, 3>, memref<?xf8E4M3FN, 2>
        asccube.copy_l12l0a %164[%c0], %154[%c0], %c0_i32, %c6_i32, %c8_i32, %c2_i32, %c20_i32, %c2_i32 : memref<?xi8, 6>, memref<?x!ascdtype.fp8_e8m0, 2>
        asccube.release_buf mte1, %c15_i64, 0 {cannir.buffer.generated_sync}
        asccube.get_buf mte1, %c17_i64, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0b %165[%c0], %158[%c0], %c0_i32, %c12_i32, %c8_i32, %c4_i32, %c8_i32, %c8_i32 : memref<?xf8E4M3FN, 4>, memref<?xf8E4M3FN, 2>
        asccube.copy_l12l0b %167[%c0], %145[%c0], %c0_i32, %c6_i32, %c8_i32, %c2_i32, %c10_i32, %c2_i32 : memref<?xi8, 7>, memref<?x!ascdtype.fp8_e8m0, 2>
        asccube.release_buf mte1, %c17_i64, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %c15_i64, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %c17_i64, 0 {cannir.buffer.generated_sync}
        asccube.mmad %161[%c0], %162, %165 scale(%164, %167), %c128_i32, %c128_i32, %c128_i32, %c0_i32 {c_matrix_init_val = false, mx_mode = true} : memref<1xf32, 5>, memref<?xf8E4M3FN, 3>, memref<?xf8E4M3FN, 4>, memref<?xi8, 6>, memref<?xi8, 7>
        asccube.release_buf m, %c17_i64, 0 {cannir.buffer.generated_sync}
        asccube.release_buf m, %c15_i64, 0 {cannir.buffer.generated_sync}
        asccube.get_buf mte1, %c14_i64, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0a %152[%c0], %153[%c0], %c0_i32, %c16_i32, %c8_i32, %c4_i32, %c8_i32, %c8_i32 : memref<?xf8E4M3FN, 3>, memref<?xf8E4M3FN, 2>
        asccube.copy_l12l0a %156[%c0], %154[%c0], %c0_i32, %c8_i32, %c8_i32, %c2_i32, %c20_i32, %c2_i32 : memref<?xi8, 6>, memref<?x!ascdtype.fp8_e8m0, 2>
        asccube.release_buf mte1, %c14_i64, 0 {cannir.buffer.generated_sync}
        asccube.get_buf mte1, %c16_i64, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0b %157[%c0], %158[%c0], %c0_i32, %c16_i32, %c8_i32, %c4_i32, %c8_i32, %c8_i32 : memref<?xf8E4M3FN, 4>, memref<?xf8E4M3FN, 2>
        asccube.release_buf mte1, %c10_i64, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0b %160[%c0], %145[%c0], %c0_i32, %c8_i32, %c8_i32, %c2_i32, %c10_i32, %c2_i32 : memref<?xi8, 7>, memref<?x!ascdtype.fp8_e8m0, 2>
        asccube.release_buf mte1, %c12_i64, 0 {cannir.buffer.generated_sync}
        asccube.release_buf mte1, %c16_i64, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %c14_i64, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %c16_i64, 0 {cannir.buffer.generated_sync}
        asccube.mmad %161[%c0], %152, %157 scale(%156, %160), %c128_i32, %c128_i32, %c128_i32, %c0_i32 {c_matrix_init_val = false, mx_mode = true} : memref<1xf32, 5>, memref<?xf8E4M3FN, 3>, memref<?xf8E4M3FN, 4>, memref<?xi8, 6>, memref<?xi8, 7>
        asccube.release_buf m, %c16_i64, 0 {cannir.buffer.generated_sync}
        asccube.release_buf m, %c14_i64, 0 {cannir.buffer.generated_sync}
        asccube.get_buf mte1, %c15_i64, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0a %162[%c0], %153[%c0], %c0_i32, %c20_i32, %c8_i32, %c4_i32, %c8_i32, %c8_i32 : memref<?xf8E4M3FN, 3>, memref<?xf8E4M3FN, 2>
        asccube.copy_l12l0a %164[%c0], %154[%c0], %c0_i32, %c10_i32, %c8_i32, %c2_i32, %c20_i32, %c2_i32 : memref<?xi8, 6>, memref<?x!ascdtype.fp8_e8m0, 2>
        asccube.release_buf mte1, %c15_i64, 0 {cannir.buffer.generated_sync}
        %168 = ascscalar.bitcast_memref %alloc_17 byte_offset = %c0_i64 : memref<81920xui8, 2> -> memref<?xf8E4M3FN, 2>
        asccube.get_buf mte1, %c17_i64, 0 {cannir.buffer.generated_sync}
        asccube.get_buf mte1, %c11_i64, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0b %165[%c0], %168[%c0], %c0_i32, %c0_i32, %c8_i32, %c4_i32, %c8_i32, %c8_i32 : memref<?xf8E4M3FN, 4>, memref<?xf8E4M3FN, 2>
        asccube.get_buf mte1, %c13_i64, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0b %167[%c0], %150[%c0], %c0_i32, %c0_i32, %c8_i32, %c2_i32, %c10_i32, %c2_i32 : memref<?xi8, 7>, memref<?x!ascdtype.fp8_e8m0, 2>
        asccube.release_buf mte1, %c17_i64, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %c15_i64, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %c17_i64, 0 {cannir.buffer.generated_sync}
        asccube.mmad %161[%c0], %162, %165 scale(%164, %167), %c128_i32, %c128_i32, %c128_i32, %c0_i32 {c_matrix_init_val = false, mx_mode = true} : memref<1xf32, 5>, memref<?xf8E4M3FN, 3>, memref<?xf8E4M3FN, 4>, memref<?xi8, 6>, memref<?xi8, 7>
        asccube.release_buf m, %c17_i64, 0 {cannir.buffer.generated_sync}
        asccube.release_buf m, %c15_i64, 0 {cannir.buffer.generated_sync}
        asccube.get_buf mte1, %c14_i64, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0a %152[%c0], %153[%c0], %c0_i32, %c24_i32, %c8_i32, %c4_i32, %c8_i32, %c8_i32 : memref<?xf8E4M3FN, 3>, memref<?xf8E4M3FN, 2>
        asccube.copy_l12l0a %156[%c0], %154[%c0], %c0_i32, %c12_i32, %c8_i32, %c2_i32, %c20_i32, %c2_i32 : memref<?xi8, 6>, memref<?x!ascdtype.fp8_e8m0, 2>
        asccube.release_buf mte1, %c14_i64, 0 {cannir.buffer.generated_sync}
        asccube.get_buf mte1, %c16_i64, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0b %157[%c0], %168[%c0], %c0_i32, %c4_i32, %c8_i32, %c4_i32, %c8_i32, %c8_i32 : memref<?xf8E4M3FN, 4>, memref<?xf8E4M3FN, 2>
        asccube.copy_l12l0b %160[%c0], %150[%c0], %c0_i32, %c2_i32, %c8_i32, %c2_i32, %c10_i32, %c2_i32 : memref<?xi8, 7>, memref<?x!ascdtype.fp8_e8m0, 2>
        asccube.release_buf mte1, %c16_i64, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %c14_i64, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %c16_i64, 0 {cannir.buffer.generated_sync}
        asccube.mmad %161[%c0], %152, %157 scale(%156, %160), %c128_i32, %c128_i32, %c128_i32, %c0_i32 {c_matrix_init_val = false, mx_mode = true} : memref<1xf32, 5>, memref<?xf8E4M3FN, 3>, memref<?xf8E4M3FN, 4>, memref<?xi8, 6>, memref<?xi8, 7>
        asccube.release_buf m, %c16_i64, 0 {cannir.buffer.generated_sync}
        asccube.release_buf m, %c14_i64, 0 {cannir.buffer.generated_sync}
        asccube.get_buf mte1, %c15_i64, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0a %162[%c0], %153[%c0], %c0_i32, %c28_i32, %c8_i32, %c4_i32, %c8_i32, %c8_i32 : memref<?xf8E4M3FN, 3>, memref<?xf8E4M3FN, 2>
        asccube.copy_l12l0a %164[%c0], %154[%c0], %c0_i32, %c14_i32, %c8_i32, %c2_i32, %c20_i32, %c2_i32 : memref<?xi8, 6>, memref<?x!ascdtype.fp8_e8m0, 2>
        asccube.release_buf mte1, %c15_i64, 0 {cannir.buffer.generated_sync}
        asccube.get_buf mte1, %c17_i64, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0b %165[%c0], %168[%c0], %c0_i32, %c8_i32, %c8_i32, %c4_i32, %c8_i32, %c8_i32 : memref<?xf8E4M3FN, 4>, memref<?xf8E4M3FN, 2>
        asccube.copy_l12l0b %167[%c0], %150[%c0], %c0_i32, %c4_i32, %c8_i32, %c2_i32, %c10_i32, %c2_i32 : memref<?xi8, 7>, memref<?x!ascdtype.fp8_e8m0, 2>
        asccube.release_buf mte1, %c17_i64, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %c15_i64, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %c17_i64, 0 {cannir.buffer.generated_sync}
        asccube.mmad %161[%c0], %162, %165 scale(%164, %167), %c128_i32, %c128_i32, %c128_i32, %c0_i32 {c_matrix_init_val = false, mx_mode = true} : memref<1xf32, 5>, memref<?xf8E4M3FN, 3>, memref<?xf8E4M3FN, 4>, memref<?xi8, 6>, memref<?xi8, 7>
        asccube.release_buf m, %c17_i64, 0 {cannir.buffer.generated_sync}
        asccube.release_buf m, %c15_i64, 0 {cannir.buffer.generated_sync}
        asccube.get_buf mte1, %c14_i64, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0a %152[%c0], %153[%c0], %c0_i32, %c32_i32, %c8_i32, %c4_i32, %c8_i32, %c8_i32 : memref<?xf8E4M3FN, 3>, memref<?xf8E4M3FN, 2>
        asccube.copy_l12l0a %156[%c0], %154[%c0], %c0_i32, %c16_i32, %c8_i32, %c2_i32, %c20_i32, %c2_i32 : memref<?xi8, 6>, memref<?x!ascdtype.fp8_e8m0, 2>
        asccube.release_buf mte1, %c14_i64, 0 {cannir.buffer.generated_sync}
        asccube.get_buf mte1, %c16_i64, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0b %157[%c0], %168[%c0], %c0_i32, %c12_i32, %c8_i32, %c4_i32, %c8_i32, %c8_i32 : memref<?xf8E4M3FN, 4>, memref<?xf8E4M3FN, 2>
        asccube.copy_l12l0b %160[%c0], %150[%c0], %c0_i32, %c6_i32, %c8_i32, %c2_i32, %c10_i32, %c2_i32 : memref<?xi8, 7>, memref<?x!ascdtype.fp8_e8m0, 2>
        asccube.release_buf mte1, %c16_i64, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %c14_i64, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %c16_i64, 0 {cannir.buffer.generated_sync}
        asccube.mmad %161[%c0], %152, %157 scale(%156, %160), %c128_i32, %c128_i32, %c128_i32, %c0_i32 {c_matrix_init_val = false, mx_mode = true} : memref<1xf32, 5>, memref<?xf8E4M3FN, 3>, memref<?xf8E4M3FN, 4>, memref<?xi8, 6>, memref<?xi8, 7>
        asccube.release_buf m, %c16_i64, 0 {cannir.buffer.generated_sync}
        asccube.release_buf m, %c14_i64, 0 {cannir.buffer.generated_sync}
        asccube.get_buf mte1, %c15_i64, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0a %162[%c0], %153[%c0], %c0_i32, %c36_i32, %c8_i32, %c4_i32, %c8_i32, %c8_i32 : memref<?xf8E4M3FN, 3>, memref<?xf8E4M3FN, 2>
        asccube.release_buf mte1, %c8_i64, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0a %164[%c0], %154[%c0], %c0_i32, %c18_i32, %c8_i32, %c2_i32, %c20_i32, %c2_i32 : memref<?xi8, 6>, memref<?x!ascdtype.fp8_e8m0, 2>
        asccube.release_buf mte1, %c9_i64, 0 {cannir.buffer.generated_sync}
        asccube.release_buf mte1, %c15_i64, 0 {cannir.buffer.generated_sync}
        asccube.get_buf mte1, %c17_i64, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0b %165[%c0], %168[%c0], %c0_i32, %c16_i32, %c8_i32, %c4_i32, %c8_i32, %c8_i32 : memref<?xf8E4M3FN, 4>, memref<?xf8E4M3FN, 2>
        asccube.release_buf mte1, %c11_i64, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0b %167[%c0], %150[%c0], %c0_i32, %c8_i32, %c8_i32, %c2_i32, %c10_i32, %c2_i32 : memref<?xi8, 7>, memref<?x!ascdtype.fp8_e8m0, 2>
        asccube.release_buf mte1, %c13_i64, 0 {cannir.buffer.generated_sync}
        asccube.release_buf mte1, %c17_i64, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %c15_i64, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %c17_i64, 0 {cannir.buffer.generated_sync}
        asccube.mmad %161[%c0], %162, %165 scale(%164, %167), %c128_i32, %c128_i32, %c128_i32, %c0_i32 {c_matrix_init_val = false, mx_mode = true} : memref<1xf32, 5>, memref<?xf8E4M3FN, 3>, memref<?xf8E4M3FN, 4>, memref<?xi8, 6>, memref<?xi8, 7>
        asccube.release_buf m, %c17_i64, 0 {cannir.buffer.generated_sync}
        asccube.release_buf m, %c15_i64, 0 {cannir.buffer.generated_sync}
        asccube.release_buf m, %138, 0 {cannir.buffer.generated_sync}
        %169 = ascscalar.memref_address %alloc_6 : memref<8192xf32, 1>
        %170 = ascscalar.memref_address %alloc_7 : memref<8192xf32, 1>
        %171 = arith.select %77, %169, %170 : i64
        %172 = arith.select %77, %c25_i64, %c26_i64 : i64
        %173 = ascscalar.bind_local_ptr %171 : memref<1xf32, 1>
        %174 = arith.divsi %61, %c16_i64 : i64
        %175 = arith.remsi %61, %c16_i64 : i64
        %176 = arith.addi %175, %c15_i64 overflow<nsw> : i64
        %177 = arith.divsi %176, %c16_i64 : i64
        %178 = arith.addi %174, %177 : i64
        %179 = arith.muli %178, %c256_i64 : i64
        %180 = arith.index_cast %179 : i64 to index
        %181 = arith.addi %61, %c1_i64 : i64
        %182 = arith.divui %181, %c2_i64 : i64
        %183 = arith.muli %182, %c2_i64 : i64
        %184 = arith.divui %180, %c16 : index
        %185 = arith.index_cast %184 : index to i64
        %186 = arith.cmpi ugt, %61, %c0_i64 : i64
        %187 = arith.cmpi eq, %172, %c26_i64 : i64
        %188 = arith.extui %187 : i1 to i64
        asccube.sync_intra_wait fixpipe, %188 {cannir.sync.banks = 3 : i32}
        %189 = arith.addi %188, %c16_i64 overflow<nsw> : i64
        asccube.sync_intra_wait fixpipe, %189 {cannir.sync.banks = 3 : i32}
        scf.if %186 {
          asccube.get_buf fixpipe, %138, 0 {cannir.buffer.generated_sync}
          %190 = arith.trunci %183 : i64 to i32
          %191 = arith.trunci %185 : i64 to i32
          asccube.copy_l0c2ub %173[%c0], %161[%c0], %c128_i32, %190, %c128_i32, %191, %c0_i32 sub_block_id %c0_i32 {dual_dst_ctl = 1 : i32, nz2nd_en = true, relu_en = false} : memref<1xf32, 1>, memref<1xf32, 5>
          asccube.release_buf fixpipe, %138, 0 {cannir.buffer.generated_sync}
        }
        asccube.sync_intra_arrive fixpipe, %188 {cannir.sync.banks = 3 : i32}
        asccube.sync_intra_arrive fixpipe, %189 {cannir.sync.banks = 3 : i32}
        scf.yield %84 : i64
      }
      %63 = arith.addi %arg15, %9 : i64
      %64 = arith.cmpi slt, %63, %7 : i64
      scf.if %64 {
        %77 = arith.cmpi ne, %63, %arg15 : i64
        scf.if %77 {
          %78 = ascscalar.bitcast_memref %alloc_15 byte_offset = %c0_i64 : memref<163840xf8E4M3FN, 2> -> memref<?xf8E4M3FN, 2>
          %79 = arith.muli %63, %c163840_i64 : i64
          %80 = cannir.extract_layout %arg1 : !cannir.tensor<<gm, ui8>, <<(-1, 1280)>, <(1280, 1)>>, <<(-1, 1280)>, <(1280, 1)>>, identity>
          %81 = cannir.extract_shape %80 : <<(-1, 1280)>, <(1280, 1)>> -> <(-1, 1280)>
          %82 = cannir.extract_dim %81[0] : !cannir.shape<(-1, 1280)>
          %83 = arith.divsi %82, %c128_i64 : i64
          %84 = arith.minsi %63, %83 : i64
          %85 = arith.muli %84, %c128_i64 : i64
          %86 = arith.subi %82, %85 : i64
          %87 = arith.minsi %86, %c128_i64 : i64
          %88 = arith.cmpi sle, %63, %83 : i64
          %89 = arith.select %88, %87, %c0_i64 : i64
          asccube.get_buf mte2, %c8_i64, 0 {cannir.buffer.generated_sync}
          %90 = arith.cmpi ugt, %89, %c0_i64 : i64
          scf.if %90 {
            %104 = arith.index_cast %79 : i64 to index
            %105 = arith.trunci %89 : i64 to i32
            asccube.copy_gm2l1 %78[%c0], %alloc_0[%104], %105, %c1280_i32, %c1280_i64, %c1_i32, %c0_i64, %c0_i64, %c128_i32 l2_cache_ctl %c0_i32 {nd2nz = true} : memref<?xf8E4M3FN, 2>, memref<1xui8>
          }
          asccube.release_buf mte2, %c8_i64, 0 {cannir.buffer.generated_sync}
          %91 = cannir.extract_layout %arg4 : !cannir.tensor<<gm, ui8>, <<(-1, 20, 2)>, <(40, 2, 1)>>, <<(-1, 20, 2)>, <(40, 2, 1)>>, identity>
          %92 = cannir.extract_shape %91 : <<(-1, 20, 2)>, <(40, 2, 1)>> -> <(-1, 20, 2)>
          %93 = cannir.extract_dim %92[0] : !cannir.shape<(-1, 20, 2)>
          %94 = arith.divsi %93, %c128_i64 : i64
          %95 = arith.minsi %63, %94 : i64
          %96 = arith.muli %95, %c128_i64 : i64
          %97 = arith.subi %93, %96 : i64
          %98 = arith.minsi %97, %c128_i64 : i64
          %99 = arith.cmpi sle, %63, %94 : i64
          %100 = arith.select %99, %98, %c0_i64 : i64
          %101 = arith.muli %63, %c5120_i64 : i64
          %102 = ascscalar.bitcast_memref %alloc_14 byte_offset = %c0_i64 : memref<5120x!ascdtype.fp8_e8m0, 2> -> memref<?x!ascdtype.fp8_e8m0, 2>
          asccube.get_buf mte2, %c9_i64, 0 {cannir.buffer.generated_sync}
          %103 = arith.cmpi ugt, %100, %c0_i64 : i64
          scf.if %103 {
            %104 = arith.index_cast %101 : i64 to index
            %105 = arith.trunci %100 : i64 to i32
            asccube.copy_gm2l1 %102[%c0], %alloc_3[%104], %c20_i32, %105, %c40_i64, %c1_i32, %c0_i64, %c0_i64, %c20_i32 l2_cache_ctl %c0_i32 {dn2nz = true, mx_scale = true} : memref<?x!ascdtype.fp8_e8m0, 2>, memref<1xui8>
          }
          asccube.release_buf mte2, %c9_i64, 0 {cannir.buffer.generated_sync}
        }
      }
      %65 = ascscalar.bind_local_ptr %15 : memref<1xf32, 5>
      %66 = arith.muli %arg15, %c4096_i64 : i64
      %67 = arith.divsi %37, %c16_i64 : i64
      %68 = arith.remsi %37, %c16_i64 : i64
      %69 = arith.addi %68, %c15_i64 overflow<nsw> : i64
      %70 = arith.divsi %69, %c16_i64 : i64
      %71 = arith.addi %67, %70 : i64
      %72 = arith.muli %71, %c256_i64 : i64
      %73 = arith.index_cast %72 : i64 to index
      %74 = arith.divui %73, %c16 : index
      %75 = arith.index_cast %74 : index to i64
      %76 = arith.cmpi ugt, %37, %c0_i64 : i64
      scf.if %76 {
        asccube.get_buf fixpipe, %16, 0 {cannir.buffer.generated_sync}
        %77 = arith.trunci %37 : i64 to i32
        %78 = arith.trunci %75 : i64 to i32
        %79 = arith.index_cast %66 : i64 to index
        asccube.copy_l0c2gm %alloc_5[%79], %65[%c0], %c32_i32, %77, %c32_i32, %78, %c0_i32 deq_scale_val(%arg14) l2_cache_ctl %c1_i32 {nz2nd_en = true, relu_en = false} : memref<1xf32>, memref<1xf32, 5>
        asccube.release_buf fixpipe, %16, 0 {cannir.buffer.generated_sync}
      }
      scf.yield %18 : i64
    }
    asccube.sync_intra_wait fixpipe, %c0_i64
    asccube.sync_intra_wait fixpipe, %c16_i64
    asccube.sync_intra_wait fixpipe, %c1_i64
    asccube.sync_intra_wait fixpipe, %c17_i64
    return
  }
}
