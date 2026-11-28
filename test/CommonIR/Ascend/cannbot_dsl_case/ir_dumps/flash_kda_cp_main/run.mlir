module {
  func.func @run_c8939c982f136061469f01c5ea28760b(%arg0: !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>, %arg1: !cannir.tensor<<gm, f32>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>, %arg2: !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>, %arg3: !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>, %arg4: !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>, %arg5: !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 64, 64)>, <(-1, -1, 4096, 64, 1)>>, <<(-1, -1, -1, 64, 64)>, <(-1, -1, 4096, 64, 1)>>, identity>, %arg6: !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>, %arg7: !cannir.tensor<<gm, f32>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, identity>, %arg8: !cannir.tensor<<gm, f32>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, identity>, %arg9: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg10: i64, %arg11: i64, %arg12: i64) attributes {cannbotdsl.contract_digest = "c8939c982f136061469f01c5ea28760bdcfff8202cd7b2fb3d41c3f12fb2ad61", cannbotdsl.public_slot_count = 13 : i64, cannbotdsl.slot_dtype_bits = [16, 32, 16, 16, 16, 16, 16, 32, 32, 32, 64, 64, 64], cannbotdsl.slot_dtype_codes = [4, 2, 4, 4, 4, 4, 4, 2, 2, 0, 0, 0, 0], cannbotdsl.slot_dtype_lanes = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], cannbotdsl.slot_flags = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], cannbotdsl.slot_kinds = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2], cannbotdsl.slot_type_digests = ["8ae7865ed05b48a13663d67ae7fbda02f2bfb23ec15e2bd62d52c6cc5f769d46", "1960f968740fb9ea4f503863f89c134d1c1888cdbaef6f2adbd598a191355ce3", "95677d90b1022c942b5e0aea92adf9600c5c260fe67d0052617fd711ddbcfb8f", "e8532f0b6dc2ff14140b6559036f38df8a747fc5f896f86e09e6f47a9d9a2dd2", "3c265f70ec96ee7efb7896aed46bf0eba796541ff055f367840b3196bd0c8aa3", "35f7a44a9b6e5cba2bd13365089124842961be1707ab7912323647c2d8f0b9e4", "986a641a700ac7f030409415f59dfd9ca51c73ee18901198c72d3585b7519273", "5ae2adc2c479dc1b99c70666e575927277c2a02ec4793f1a2bbaee13c14b352a", "5ae2adc2c479dc1b99c70666e575927277c2a02ec4793f1a2bbaee13c14b352a", "df1782892fb0fe07e22c2b9245d6bd16a4aa9b9bed10c33a9432045882421aba", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515"], cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.host} {
    cannir.kernel_launch @kernel_FlashKdaCpKernel___call___0(%arg12) [%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11] {cannbotdsl.profile = {indices = array<i64: 0, 1, 2, 3, 4, 5, 7, 9, 6, 8>, name = "flash_kda_cp_main", op_type = "FlashKDACPMain", roles = array<i64: 0, 0, 0, 0, 0, 0, 0, 0, 1, 1>}} : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>, !cannir.tensor<<gm, f32>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>, !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>, !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>, !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>, !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 64, 64)>, <(-1, -1, 4096, 64, 1)>>, <<(-1, -1, -1, 64, 64)>, <(-1, -1, 4096, 64, 1)>>, identity>, !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>, !cannir.tensor<<gm, f32>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, identity>, !cannir.tensor<<gm, f32>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, identity>, !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, i64, i64
    return
  }
  func.func @kernel_FlashKdaCpKernel___call___0_aiv(%arg0: !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>, %arg1: !cannir.tensor<<gm, f32>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>, %arg2: !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>, %arg3: !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>, %arg4: !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>, %arg5: !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 64, 64)>, <(-1, -1, 4096, 64, 1)>>, <<(-1, -1, -1, 64, 64)>, <(-1, -1, 4096, 64, 1)>>, identity>, %arg6: !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>, %arg7: !cannir.tensor<<gm, f32>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, identity>, %arg8: !cannir.tensor<<gm, f32>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, identity>, %arg9: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg10: i64, %arg11: i64) attributes {cannir.buffer.auto_sync, cannir.buffer.sync_analyzed, cannir.buffer.sync_verified, cannir.buffer.tensor_state, cannir.core = "aiv", cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.device, cannir.explicit_ring_buffer_cursors, cannir.kernel_group = "kernel_FlashKdaCpKernel___call___0", cannir.resource.ids_prepared} {
    %c16384_i32 = arith.constant 16384 : i32
    %c256_i32 = arith.constant 256 : i32
    %c8_i32 = arith.constant 8 : i32
    %c32768_i32 = arith.constant 32768 : i32
    %c4 = arith.constant 4 : index
    %c6 = arith.constant 6 : index
    %c5 = arith.constant 5 : index
    %c3 = arith.constant 3 : index
    %c0_i32 = arith.constant 0 : i32
    %false = arith.constant false
    %c1_i32 = arith.constant 1 : i32
    %c9_i32 = arith.constant 9 : i32
    %c2_i32 = arith.constant 2 : i32
    %c64_i32 = arith.constant 64 : i32
    %c63_i64 = arith.constant 63 : i64
    %c64_i64 = arith.constant 64 : i64
    %c65536_i64 = arith.constant 65536 : i64
    %c256_i64 = arith.constant 256 : i64
    %c128_i64 = arith.constant 128 : i64
    %c512_i64 = arith.constant 512 : i64
    %c128_i32 = arith.constant 128 : i32
    %c16_i32 = arith.constant 16 : i32
    %c0 = arith.constant 0 : index
    %c4096_i32 = arith.constant 4096 : i32
    %c32_i32 = arith.constant 32 : i32
    %c4_i32 = arith.constant 4 : i32
    %c8192_i32 = arith.constant 8192 : i32
    %c32768_i64 = arith.constant 32768 : i64
    %c16384_i64 = arith.constant 16384 : i64
    %c8192_i64 = arith.constant 8192 : i64
    %c2 = arith.constant 2 : index
    %c4096_i64 = arith.constant 4096 : i64
    %c65_i32 = arith.constant 65 : i32
    %c1024_i64 = arith.constant 1024 : i64
    %c2048_i64 = arith.constant 2048 : i64
    %c32_i64 = arith.constant 32 : i64
    %c16_i64 = arith.constant 16 : i64
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
    %alloc = memref.alloc() {cannir.arg_index = 1 : i64} : memref<1xf32>
    %alloc_0 = memref.alloc() {cannir.arg_index = 4 : i64} : memref<1xbf16>
    %alloc_1 = memref.alloc() {cannir.arg_index = 7 : i64} : memref<1xf32>
    %alloc_2 = memref.alloc() {cannir.arg_index = 8 : i64} : memref<1xf32>
    %alloc_3 = memref.alloc() {cannir.arg_index = 9 : i64} : memref<1xi32>
    %alloc_4 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<64xf32, 1>
    %alloc_5 = memref.alloc() {cannir.byte_offset = 256 : i64} : memref<64xf32, 1>
    %alloc_6 = memref.alloc() {cannir.byte_offset = 512 : i64} : memref<8192xf32, 1>
    %alloc_7 = memref.alloc() {cannir.byte_offset = 33280 : i64} : memref<4096xf32, 1>
    %alloc_8 = memref.alloc() {cannir.byte_offset = 49664 : i64} : memref<8192xf32, 1>
    %alloc_9 = memref.alloc() {cannir.byte_offset = 82432 : i64} : memref<4096xbf16, 1>
    %alloc_10 = memref.alloc() {cannir.byte_offset = 90624 : i64} : memref<8192xbf16, 1>
    %alloc_11 = memref.alloc() {cannir.byte_offset = 107008 : i64} : memref<8256xbf16, 1>
    %alloc_12 = memref.alloc() {cannir.byte_offset = 123520 : i64} : memref<4224xbf16, 1>
    %alloc_13 = memref.alloc() {cannir.byte_offset = 131968 : i64} : memref<8192xf32, 1>
    %alloc_14 = memref.alloc() {cannir.byte_offset = 164736 : i64} : memref<8192xf32, 1>
    %alloc_15 = memref.alloc() {cannir.byte_offset = 197504 : i64} : memref<4096xbf16, 1>
    %alloc_16 = memref.alloc() {cannir.byte_offset = 205696 : i64} : memref<4096xbf16, 1>
    %alloc_17 = memref.alloc() {cannir.byte_offset = 147456 : i64} : memref<16384xbf16, 2>
    %alloc_18 = memref.alloc() {cannir.byte_offset = 114688 : i64} : memref<16384xbf16, 2>
    %alloc_19 = memref.alloc() {cannir.byte_offset = 196608 : i64} : memref<8192xbf16, 2>
    %alloc_20 = memref.alloc() {cannir.byte_offset = 180224 : i64} : memref<8192xbf16, 2>
    %0 = ascscalar.get_block_idx
    %1 = ascscalar.get_block_num
    %2 = memref.load %alloc_3[%c2] : memref<1xi32>
    %3 = memref.load %alloc_3[%c3] : memref<1xi32>
    %4 = memref.load %alloc_3[%c5] : memref<1xi32>
    %5 = arith.extsi %4 : i32 to i64
    %6 = memref.load %alloc_3[%c6] : memref<1xi32>
    %7 = arith.extsi %6 : i32 to i64
    %8 = arith.cmpi eq, %2, %c0_i32 : i32
    %9 = arith.extsi %3 : i32 to i64
    %10 = arith.cmpi eq, %9, %1 : i64
    %11 = arith.muli %5, %7 : i64
    %12 = arith.cmpi eq, %11, %arg10 : i64
    %13 = arith.select %10, %12, %false : i1
    %14 = arith.select %8, %13, %false : i1
    scf.if %14 {
      ascvec.sync_intra_arrive v, %c5_i64
      ascvec.sync_intra_arrive v, %c4_i64
      %15 = memref.load %alloc_3[%c4] : memref<1xi32>
      scf.for %arg12 = %c0_i32 to %15 step %c1_i32  : i32 {
        %16 = arith.addi %arg12, %c9_i32 : i32
        %17 = arith.index_cast %16 : i32 to index
        %18 = memref.load %alloc_3[%17] : memref<1xi32>
        %19 = arith.addi %18, %c1_i32 : i32
        %20 = arith.index_cast %19 : i32 to index
        %21 = memref.load %alloc_3[%20] : memref<1xi32>
        %22 = arith.addi %18, %c2_i32 : i32
        %23 = arith.addi %22, %21 : i32
        %24 = arith.addi %23, %21 : i32
        %25 = arith.addi %24, %21 : i32
        %26 = arith.addi %25, %21 : i32
        %27 = arith.addi %26, %c1_i32 : i32
        %28 = arith.addi %27, %21 : i32
        %29 = arith.addi %28, %21 : i32
        %30 = arith.addi %29, %21 : i32
        %31 = arith.addi %30, %c65_i32 : i32
        %32 = arith.addi %31, %21 : i32
        %33 = arith.addi %32, %21 : i32
        %34 = arith.addi %33, %c1_i32 : i32
        %35 = arith.muli %0, %c2_i64 : i64
        %36 = arith.extsi %34 : i32 to i64
        %37 = arith.addi %36, %35 : i64
        %38 = arith.index_cast %37 : i64 to index
        %39 = memref.load %alloc_3[%38] : memref<1xi32>
        %40 = arith.addi %37, %c1_i64 : i64
        %41 = arith.index_cast %40 : i64 to index
        %42 = memref.load %alloc_3[%41] : memref<1xi32>
        %43 = arith.index_cast %31 : i32 to index
        %44 = memref.load %alloc_3[%43] : memref<1xi32>
        %45 = ascscalar.bitcast_memref %alloc_13 byte_offset = %c0_i64 : memref<8192xf32, 1> -> memref<?xf32, 1>
        %46 = arith.cmpi eq, %44, %c1_i32 : i32
        scf.if %46 {
          %49:7 = scf.for %arg13 = %c0_i32 to %21 step %c1_i32 iter_args(%arg14 = %c0_i64, %arg15 = %c0_i64, %arg16 = %c0_i64, %arg17 = %c0_i64, %arg18 = %c0_i64, %arg19 = %c0_i64, %arg20 = %c0_i64) -> (i64, i64, i64, i64, i64, i64, i64)  : i32 {
            %50 = arith.addi %32, %arg13 : i32
            %51 = arith.index_cast %50 : i32 to index
            %52 = memref.load %alloc_3[%51] : memref<1xi32>
            %53 = arith.addi %50, %c1_i32 : i32
            %54 = arith.index_cast %53 : i32 to index
            %55 = memref.load %alloc_3[%54] : memref<1xi32>
            %56 = arith.cmpi slt, %39, %52 : i32
            %57 = arith.select %56, %52, %39 : i32
            %58 = arith.cmpi sgt, %42, %55 : i32
            %59 = arith.select %58, %55, %42 : i32
            %60 = arith.cmpi slt, %57, %59 : i32
            %61 = arith.subi %57, %52 : i32
            %62 = arith.subi %59, %52 : i32
            %63 = arith.extsi %61 : i32 to i64
            %64 = arith.extsi %62 : i32 to i64
            %65 = arith.select %60, %63, %c0_i64 : i64
            %66 = arith.select %60, %64, %c0_i64 : i64
            %67 = arith.addi %22, %arg13 : i32
            %68 = arith.index_cast %67 : i32 to index
            %69 = memref.load %alloc_3[%68] : memref<1xi32>
            %70 = arith.addi %23, %arg13 : i32
            %71 = arith.index_cast %70 : i32 to index
            %72 = memref.load %alloc_3[%71] : memref<1xi32>
            %73 = arith.addi %24, %arg13 : i32
            %74 = arith.index_cast %73 : i32 to index
            %75 = memref.load %alloc_3[%74] : memref<1xi32>
            %76 = arith.addi %25, %arg13 : i32
            %77 = arith.index_cast %76 : i32 to index
            %78 = memref.load %alloc_3[%77] : memref<1xi32>
            %79 = arith.extsi %78 : i32 to i64
            %80 = arith.addi %27, %arg13 : i32
            %81 = arith.index_cast %80 : i32 to index
            %82 = memref.load %alloc_3[%81] : memref<1xi32>
            %83 = arith.addi %28, %arg13 : i32
            %84 = arith.index_cast %83 : i32 to index
            %85 = memref.load %alloc_3[%84] : memref<1xi32>
            %86 = arith.addi %79, %c63_i64 : i64
            %87 = arith.remsi %86, %c64_i64 : i64
            %88 = arith.cmpi ne, %87, %c0_i64 : i64
            %89 = arith.cmpi slt, %87, %c0_i64 : i64
            %90 = arith.andi %88, %89 : i1
            %91 = arith.divsi %86, %c64_i64 : i64
            %92 = arith.extui %90 : i1 to i64
            %93 = arith.subi %91, %92 : i64
            %94 = arith.extsi %75 : i32 to i64
            %95 = arith.extsi %72 : i32 to i64
            %96 = ascscalar.get_subblock_id
            %97 = arith.cmpi slt, %0, %1 : i64
            %98:7 = scf.if %97 -> (i64, i64, i64, i64, i64, i64, i64) {
              %99:7 = scf.for %arg21 = %65 to %66 step %c1_i64 iter_args(%arg22 = %arg14, %arg23 = %arg15, %arg24 = %arg16, %arg25 = %arg17, %arg26 = %arg18, %arg27 = %arg19, %arg28 = %arg20) -> (i64, i64, i64, i64, i64, i64, i64)  : i64 {
                %100 = arith.maxsi %7, %c1_i64 : i64
                %101 = arith.remsi %arg21, %100 : i64
                %102 = arith.cmpi slt, %101, %c0_i64 : i64
                %103 = arith.select %102, %100, %c0_i64 : i64
                %104 = arith.addi %101, %103 : i64
                %105 = arith.extsi %69 : i32 to i64
                %106 = arith.addi %82, %85 : i32
                %107 = arith.extsi %106 : i32 to i64
                %108 = arith.cmpi eq, %107, %93 : i64
                %109 = arith.muli %82, %c64_i32 : i32
                %110 = arith.extsi %109 : i32 to i64
                %111 = arith.subi %79, %110 : i64
                %112 = arith.minsi %111, %c64_i64 : i64
                %113 = arith.select %108, %112, %c64_i64 : i64
                %114 = arith.extsi %82 : i32 to i64
                %115 = arith.cmpi eq, %arg22, %c0_i64 : i64
                %116 = arith.addi %arg22, %c1_i64 : i64
                %117 = arith.remui %116, %c2_i64 : i64
                %118 = arith.cmpi ne, %arg22, %c0_i64 : i64
                %119 = arith.extui %118 : i1 to i64
                %120 = arith.addi %arg24, %c1_i64 : i64
                %121 = arith.remui %120, %c2_i64 : i64
                %122 = arith.cmpi eq, %82, %c0_i32 : i32
                %123 = scf.if %122 -> (i64) {
                  %153 = cannir.extract_physical_layout %arg7 : !cannir.tensor<<gm, f32>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, identity>
                  %154 = cannir.extract_stride %153 : <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>> -> <(-1, 16384, 128, 1)>
                  %155 = cannir.extract_dim %154[0] : !cannir.stride<(-1, 16384, 128, 1)>
                  %156 = arith.muli %105, %155 : i64
                  %157 = arith.muli %156, %c4_i64 : i64
                  %158 = arith.muli %104, %c65536_i64 : i64
                  %159 = arith.addi %157, %158 : i64
                  %160 = arith.muli %96, %c256_i64 : i64
                  %161 = arith.addi %159, %160 : i64
                  ascvec.get_buf mte2, %c4_i64, 0 {cannir.buffer.generated_sync}
                  "ascvec.copy_gm2ub_plan"(%45, %alloc_1, %c0_i32, %c0_i32, %c0_i64, %161, %c128_i64, %c256_i64, %c512_i64, %c256_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 131968 : i64, ub_capacity = 32768 : i64}> : (memref<?xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                  ascvec.release_buf mte2, %c4_i64, 0 {cannir.buffer.generated_sync}
                  ascvec.get_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
                  ascvec.copy_ub2ub %alloc_8[%c0], %alloc_13[%c0], %c32768_i32 : memref<8192xf32, 1>, memref<8192xf32, 1>
                  ascvec.vf(%alloc_8, %alloc_13 : memref<8192xf32, 1>, memref<8192xf32, 1>) {
                  ^bb0(%arg29: memref<8192xf32, 1>, %arg30: memref<8192xf32, 1>):
                  } {cannir.vf_mode = "simd"}
                  ascvec.release_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
                  ascvec.vf(%alloc_8, %alloc_10 : memref<8192xf32, 1>, memref<8192xbf16, 1>) {
                  ^bb0(%arg29: memref<8192xf32, 1>, %arg30: memref<8192xbf16, 1>):
                    %c1_i32_21 = arith.constant 1 : i32
                    %c128_i32_22 = arith.constant 128 : i32
                    %c0_i32_23 = arith.constant 0 : i32
                    %c64_i32_24 = arith.constant 64 : i32
                    scf.for %arg31 = %c0_i32_23 to %c128_i32_22 step %c1_i32_21  : i32 {
                      %mask, %remaining_out = ascvec.update_mask %c64_i32_24 : !ascvec.mask
                      %175 = arith.muli %arg31, %c64_i32_24 overflow<nsw> : i32
                      %176 = arith.index_cast %175 : i32 to index
                      %177 = ascvec.loadalign %arg29[%176] : memref<8192xf32, 1>, index -> <f32>
                      %178 = ascvec.cast %177, %mask : <f32> -> <bf16>
                      ascvec.storealign_pack %arg30[%176], %178, %mask {mode = "b32_to_b16"} : memref<8192xbf16, 1>, index, <bf16>
                    }
                  } {cannir.vf_mode = "simd"}
                  ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                  ascvec.vf(%alloc_10, %alloc_11 : memref<8192xbf16, 1>, memref<8256xbf16, 1>) {
                  ^bb0(%arg29: memref<8192xbf16, 1>, %arg30: memref<8256xbf16, 1>):
                    %c1_i32_21 = arith.constant 1 : i32
                    %c128_i32_22 = arith.constant 128 : i32
                    %c0_i32_23 = arith.constant 0 : i32
                    %c129 = arith.constant 129 : index
                    %c16_i32_24 = arith.constant 16 : i32
                    %c0_25 = arith.constant 0 : index
                    %c64_i32_26 = arith.constant 64 : i32
                    scf.for %arg31 = %c0_i32_23 to %c128_i32_22 step %c1_i32_21  : i32 {
                      %mask, %remaining_out = ascvec.update_mask %c64_i32_26 elem_bits 16 : !ascvec.mask
                      %175 = arith.muli %arg31, %c64_i32_26 overflow<nsw> : i32
                      %176 = arith.index_cast %175 : i32 to index
                      %177 = ascvec.loadalign %arg29[%176] : memref<8192xbf16, 1>, index -> <bf16>
                      %178 = arith.muli %arg31, %c16_i32_24 overflow<nsw> : i32
                      %179 = arith.index_cast %178 : i32 to index
                      ascvec.storealign_strided %arg30[%179], %177, %mask, %c129, %c0_25 : memref<8256xbf16, 1>, <bf16>
                    }
                  } {cannir.vf_mode = "simd"}
                  ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                  %162 = arith.cmpi eq, %arg25, %c0_i64 : i64
                  %163 = ascscalar.memref_address %alloc_18 : memref<16384xbf16, 2>
                  %164 = ascscalar.memref_address %alloc_17 : memref<16384xbf16, 2>
                  %165 = arith.select %162, %163, %164 : i64
                  %166 = arith.select %162, %c8_i64, %c9_i64 : i64
                  %167 = arith.addi %arg25, %c1_i64 : i64
                  %168 = arith.remui %167, %c2_i64 : i64
                  %169 = ascscalar.bitcast_memref %alloc_11 byte_offset = %c0_i64 : memref<8256xbf16, 1> -> memref<?xbf16, 1>
                  %170 = ascscalar.bind_local_ptr %165 : memref<1xbf16, 2>
                  %171 = arith.muli %96, %c8192_i64 : i64
                  %172 = arith.cmpi eq, %166, %c9_i64 : i64
                  %173 = arith.extui %172 : i1 to i64
                  ascvec.sync_intra_wait mte3, %173 {cannir.sync.banks = 3 : i32}
                  ascvec.get_buf mte3, %c2_i64, 0 {cannir.buffer.generated_sync}
                  %174 = arith.index_cast %171 : i64 to index
                  ascvec.copy_ub2l1_strided %170[%174], %169[%c0], %c4_i32, %c128_i32, %c1_i32, %c0_i32 : memref<1xbf16, 2>, memref<?xbf16, 1>
                  ascvec.release_buf mte3, %c2_i64, 0 {cannir.buffer.generated_sync}
                  ascvec.sync_intra_arrive mte3, %173 {cannir.sync.banks = 3 : i32}
                  scf.yield %168 : i64
                } else {
                  %153 = cannir.extract_physical_layout %arg8 : !cannir.tensor<<gm, f32>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, identity>
                  %154 = cannir.extract_stride %153 : <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>> -> <(-1, 16384, 128, 1)>
                  %155 = cannir.extract_dim %154[0] : !cannir.stride<(-1, 16384, 128, 1)>
                  %156 = arith.muli %105, %155 : i64
                  %157 = arith.muli %156, %c4_i64 : i64
                  %158 = arith.muli %104, %c65536_i64 : i64
                  %159 = arith.addi %157, %158 : i64
                  %160 = arith.muli %96, %c256_i64 : i64
                  %161 = arith.addi %159, %160 : i64
                  ascvec.get_buf mte2, %c4_i64, 0 {cannir.buffer.generated_sync}
                  "ascvec.copy_gm2ub_plan"(%45, %alloc_2, %c0_i32, %c0_i32, %c0_i64, %161, %c128_i64, %c256_i64, %c512_i64, %c256_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 131968 : i64, ub_capacity = 32768 : i64}> : (memref<?xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                  ascvec.release_buf mte2, %c4_i64, 0 {cannir.buffer.generated_sync}
                  ascvec.get_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
                  ascvec.copy_ub2ub %alloc_8[%c0], %alloc_13[%c0], %c32768_i32 : memref<8192xf32, 1>, memref<8192xf32, 1>
                  ascvec.vf(%alloc_8, %alloc_13 : memref<8192xf32, 1>, memref<8192xf32, 1>) {
                  ^bb0(%arg29: memref<8192xf32, 1>, %arg30: memref<8192xf32, 1>):
                  } {cannir.vf_mode = "simd"}
                  ascvec.release_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
                  ascvec.vf(%alloc_8, %alloc_10 : memref<8192xf32, 1>, memref<8192xbf16, 1>) {
                  ^bb0(%arg29: memref<8192xf32, 1>, %arg30: memref<8192xbf16, 1>):
                    %c1_i32_21 = arith.constant 1 : i32
                    %c128_i32_22 = arith.constant 128 : i32
                    %c0_i32_23 = arith.constant 0 : i32
                    %c64_i32_24 = arith.constant 64 : i32
                    scf.for %arg31 = %c0_i32_23 to %c128_i32_22 step %c1_i32_21  : i32 {
                      %mask, %remaining_out = ascvec.update_mask %c64_i32_24 : !ascvec.mask
                      %175 = arith.muli %arg31, %c64_i32_24 overflow<nsw> : i32
                      %176 = arith.index_cast %175 : i32 to index
                      %177 = ascvec.loadalign %arg29[%176] : memref<8192xf32, 1>, index -> <f32>
                      %178 = ascvec.cast %177, %mask : <f32> -> <bf16>
                      ascvec.storealign_pack %arg30[%176], %178, %mask {mode = "b32_to_b16"} : memref<8192xbf16, 1>, index, <bf16>
                    }
                  } {cannir.vf_mode = "simd"}
                  ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                  ascvec.vf(%alloc_10, %alloc_11 : memref<8192xbf16, 1>, memref<8256xbf16, 1>) {
                  ^bb0(%arg29: memref<8192xbf16, 1>, %arg30: memref<8256xbf16, 1>):
                    %c1_i32_21 = arith.constant 1 : i32
                    %c128_i32_22 = arith.constant 128 : i32
                    %c0_i32_23 = arith.constant 0 : i32
                    %c129 = arith.constant 129 : index
                    %c16_i32_24 = arith.constant 16 : i32
                    %c0_25 = arith.constant 0 : index
                    %c64_i32_26 = arith.constant 64 : i32
                    scf.for %arg31 = %c0_i32_23 to %c128_i32_22 step %c1_i32_21  : i32 {
                      %mask, %remaining_out = ascvec.update_mask %c64_i32_26 elem_bits 16 : !ascvec.mask
                      %175 = arith.muli %arg31, %c64_i32_26 overflow<nsw> : i32
                      %176 = arith.index_cast %175 : i32 to index
                      %177 = ascvec.loadalign %arg29[%176] : memref<8192xbf16, 1>, index -> <bf16>
                      %178 = arith.muli %arg31, %c16_i32_24 overflow<nsw> : i32
                      %179 = arith.index_cast %178 : i32 to index
                      ascvec.storealign_strided %arg30[%179], %177, %mask, %c129, %c0_25 : memref<8256xbf16, 1>, <bf16>
                    }
                  } {cannir.vf_mode = "simd"}
                  ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                  %162 = arith.cmpi eq, %arg25, %c0_i64 : i64
                  %163 = ascscalar.memref_address %alloc_18 : memref<16384xbf16, 2>
                  %164 = ascscalar.memref_address %alloc_17 : memref<16384xbf16, 2>
                  %165 = arith.select %162, %163, %164 : i64
                  %166 = arith.select %162, %c8_i64, %c9_i64 : i64
                  %167 = arith.addi %arg25, %c1_i64 : i64
                  %168 = arith.remui %167, %c2_i64 : i64
                  %169 = ascscalar.bitcast_memref %alloc_11 byte_offset = %c0_i64 : memref<8256xbf16, 1> -> memref<?xbf16, 1>
                  %170 = ascscalar.bind_local_ptr %165 : memref<1xbf16, 2>
                  %171 = arith.muli %96, %c8192_i64 : i64
                  %172 = arith.cmpi eq, %166, %c9_i64 : i64
                  %173 = arith.extui %172 : i1 to i64
                  ascvec.sync_intra_wait mte3, %173 {cannir.sync.banks = 3 : i32}
                  ascvec.get_buf mte3, %c2_i64, 0 {cannir.buffer.generated_sync}
                  %174 = arith.index_cast %171 : i64 to index
                  ascvec.copy_ub2l1_strided %170[%174], %169[%c0], %c4_i32, %c128_i32, %c1_i32, %c0_i32 : memref<1xbf16, 2>, memref<?xbf16, 1>
                  ascvec.release_buf mte3, %c2_i64, 0 {cannir.buffer.generated_sync}
                  ascvec.sync_intra_arrive mte3, %173 {cannir.sync.banks = 3 : i32}
                  scf.yield %168 : i64
                }
                %124 = ascscalar.memref_address %alloc_4 : memref<64xf32, 1>
                %125 = ascscalar.memref_address %alloc_5 : memref<64xf32, 1>
                %126 = arith.select %115, %124, %125 : i64
                %127 = cannir.extract_layout %arg1 : !cannir.tensor<<gm, f32>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                %128 = cannir.extract_shape %127 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, -1, 128)>
                %129 = cannir.extract_dim %128[2] : !cannir.shape<(-1, -1, -1, 128)>
                %130 = arith.minsi %114, %129 : i64
                %131 = arith.subi %129, %130 : i64
                %132 = arith.minsi %131, %c1_i64 : i64
                %133 = arith.cmpi sle, %114, %129 : i64
                %134 = arith.select %133, %132, %c0_i64 : i64
                %135 = cannir.extract_physical_layout %arg1 : !cannir.tensor<<gm, f32>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                %136 = cannir.extract_stride %135 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, 128, 1)>
                %137 = cannir.extract_dim %136[0] : !cannir.stride<(-1, -1, 128, 1)>
                %138 = arith.muli %105, %137 : i64
                %139 = arith.muli %138, %c4_i64 : i64
                %140 = cannir.extract_dim %136[1] : !cannir.stride<(-1, -1, 128, 1)>
                %141 = arith.muli %104, %140 : i64
                %142 = arith.muli %141, %c4_i64 : i64
                %143 = arith.addi %139, %142 : i64
                %144 = arith.muli %114, %c512_i64 : i64
                %145 = arith.addi %143, %144 : i64
                %146 = arith.muli %96, %c256_i64 : i64
                %147 = arith.addi %145, %146 : i64
                %148 = ascscalar.bind_local_ptr %126 : memref<1xf32, 1>
                ascvec.get_buf mte2, %119, 0 {cannir.buffer.generated_sync}
                "ascvec.copy_gm2ub_plan"(%148, %alloc, %c0_i32, %c0_i32, %c0_i64, %147, %134, %c256_i64, %c512_i64, %c256_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = -1 : i64, ub_capacity = 4 : i64}> : (memref<1xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                ascvec.release_buf mte2, %119, 0 {cannir.buffer.generated_sync}
                %149 = arith.muli %96, %c32_i64 : i64
                %150 = arith.cmpi slt, %149, %113 : i64
                %151 = scf.if %150 -> (i64) {
                  %153 = arith.subi %113, %149 : i64
                  %154 = arith.cmpi sgt, %153, %c32_i64 : i64
                  %155 = arith.select %154, %c32_i64, %153 : i64
                  %156 = arith.cmpi eq, %arg27, %c0_i64 : i64
                  %157 = ascscalar.memref_address %alloc_15 : memref<4096xbf16, 1>
                  %158 = ascscalar.memref_address %alloc_16 : memref<4096xbf16, 1>
                  %159 = arith.select %156, %157, %158 : i64
                  %160 = arith.select %156, %c6_i64, %c7_i64 : i64
                  %161 = arith.addi %arg27, %c1_i64 : i64
                  %162 = arith.remui %161, %c2_i64 : i64
                  %163 = cannir.extract_physical_layout %arg4 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                  %164 = cannir.extract_stride %163 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                  %165 = cannir.extract_dim %164[2] : !cannir.stride<(-1, -1, -1, 1)>
                  %166 = arith.muli %165, %c2_i64 : i64
                  %167 = cannir.extract_dim %164[0] : !cannir.stride<(-1, -1, -1, 1)>
                  %168 = arith.muli %95, %167 : i64
                  %169 = arith.muli %168, %c2_i64 : i64
                  %170 = cannir.extract_dim %164[1] : !cannir.stride<(-1, -1, -1, 1)>
                  %171 = arith.muli %104, %170 : i64
                  %172 = arith.muli %171, %c2_i64 : i64
                  %173 = arith.addi %169, %172 : i64
                  %174 = arith.muli %114, %c64_i64 : i64
                  %175 = arith.addi %149, %174 : i64
                  %176 = arith.addi %175, %94 : i64
                  %177 = arith.muli %176, %165 : i64
                  %178 = arith.muli %177, %c2_i64 : i64
                  %179 = arith.addi %173, %178 : i64
                  %180 = ascscalar.bind_local_ptr %159 : memref<1xbf16, 1>
                  ascvec.get_buf mte2, %160, 0 {cannir.buffer.generated_sync}
                  "ascvec.copy_gm2ub_plan"(%180, %alloc_0, %c0_i32, %c0_i32, %c0_i64, %179, %155, %c256_i64, %166, %c256_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                  ascvec.release_buf mte2, %160, 0 {cannir.buffer.generated_sync}
                  scf.yield %162 : i64
                } else {
                  scf.yield %arg27 : i64
                }
                %152:7 = scf.for %arg29 = %c0_i32 to %85 step %c1_i32 iter_args(%arg30 = %117, %arg31 = %arg23, %arg32 = %121, %arg33 = %123, %arg34 = %arg26, %arg35 = %151, %arg36 = %arg28) -> (i64, i64, i64, i64, i64, i64, i64)  : i32 {
                  %153 = arith.addi %82, %arg29 : i32
                  %154 = arith.addi %153, %c1_i32 : i32
                  %155 = arith.extsi %154 : i32 to i64
                  %156 = arith.cmpi eq, %155, %93 : i64
                  %157 = arith.muli %153, %c64_i32 : i32
                  %158 = arith.extsi %157 : i32 to i64
                  %159 = arith.subi %79, %158 : i64
                  %160 = arith.select %156, %159, %c64_i64 : i64
                  %161 = arith.addi %153, %c2_i32 : i32
                  %162 = arith.extsi %161 : i32 to i64
                  %163 = arith.cmpi eq, %162, %93 : i64
                  %164 = arith.muli %154, %c64_i32 : i32
                  %165 = arith.extsi %164 : i32 to i64
                  %166 = arith.subi %79, %165 : i64
                  %167 = arith.select %163, %166, %c64_i64 : i64
                  %168 = arith.cmpi eq, %arg31, %c0_i64 : i64
                  %169 = ascscalar.memref_address %alloc_18 : memref<16384xbf16, 2>
                  %170 = ascscalar.memref_address %alloc_17 : memref<16384xbf16, 2>
                  %171 = arith.addi %arg31, %c1_i64 : i64
                  %172 = arith.remui %171, %c2_i64 : i64
                  %173 = arith.addi %arg34, %c1_i64 : i64
                  %174 = arith.remui %173, %c2_i64 : i64
                  %175 = arith.addi %174, %c1_i64 overflow<nsw> : i64
                  %176 = arith.remui %175, %c2_i64 : i64
                  %177 = arith.cmpi ne, %arg31, %c0_i64 : i64
                  %178 = arith.extui %177 : i1 to i64
                  %179 = arith.addi %arg29, %c1_i32 : i32
                  %180 = arith.cmpi slt, %179, %85 : i32
                  %181:2 = scf.if %180 -> (i64, i64) {
                    %207 = arith.cmpi eq, %arg30, %c0_i64 : i64
                    %208 = arith.cmpi ne, %arg30, %c0_i64 : i64
                    %209 = arith.extui %208 : i1 to i64
                    %210 = arith.addi %arg30, %c1_i64 overflow<nsw> : i64
                    %211 = arith.remui %210, %c2_i64 : i64
                    %212 = arith.select %207, %124, %125 : i64
                    %213 = arith.minsi %155, %129 : i64
                    %214 = arith.subi %129, %213 : i64
                    %215 = arith.minsi %214, %c1_i64 : i64
                    %216 = arith.cmpi sle, %155, %129 : i64
                    %217 = arith.select %216, %215, %c0_i64 : i64
                    %218 = arith.muli %155, %c512_i64 : i64
                    %219 = arith.addi %143, %218 : i64
                    %220 = arith.addi %219, %146 : i64
                    %221 = ascscalar.bind_local_ptr %212 : memref<1xf32, 1>
                    ascvec.get_buf mte2, %209, 0 {cannir.buffer.generated_sync}
                    "ascvec.copy_gm2ub_plan"(%221, %alloc, %c0_i32, %c0_i32, %c0_i64, %220, %217, %c256_i64, %c512_i64, %c256_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = -1 : i64, ub_capacity = 4 : i64}> : (memref<1xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                    ascvec.release_buf mte2, %209, 0 {cannir.buffer.generated_sync}
                    %222 = arith.cmpi slt, %149, %167 : i64
                    %223 = scf.if %222 -> (i64) {
                      %224 = arith.subi %167, %149 : i64
                      %225 = arith.cmpi sgt, %224, %c32_i64 : i64
                      %226 = arith.select %225, %c32_i64, %224 : i64
                      %227 = arith.cmpi eq, %arg35, %c0_i64 : i64
                      %228 = ascscalar.memref_address %alloc_15 : memref<4096xbf16, 1>
                      %229 = ascscalar.memref_address %alloc_16 : memref<4096xbf16, 1>
                      %230 = arith.select %227, %228, %229 : i64
                      %231 = arith.select %227, %c6_i64, %c7_i64 : i64
                      %232 = arith.addi %arg35, %c1_i64 : i64
                      %233 = arith.remui %232, %c2_i64 : i64
                      %234 = cannir.extract_physical_layout %arg4 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                      %235 = cannir.extract_stride %234 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                      %236 = cannir.extract_dim %235[2] : !cannir.stride<(-1, -1, -1, 1)>
                      %237 = arith.muli %236, %c2_i64 : i64
                      %238 = cannir.extract_dim %235[0] : !cannir.stride<(-1, -1, -1, 1)>
                      %239 = arith.muli %95, %238 : i64
                      %240 = arith.muli %239, %c2_i64 : i64
                      %241 = cannir.extract_dim %235[1] : !cannir.stride<(-1, -1, -1, 1)>
                      %242 = arith.muli %104, %241 : i64
                      %243 = arith.muli %242, %c2_i64 : i64
                      %244 = arith.addi %240, %243 : i64
                      %245 = arith.muli %155, %c64_i64 : i64
                      %246 = arith.addi %149, %245 : i64
                      %247 = arith.addi %246, %94 : i64
                      %248 = arith.muli %247, %236 : i64
                      %249 = arith.muli %248, %c2_i64 : i64
                      %250 = arith.addi %244, %249 : i64
                      %251 = ascscalar.bind_local_ptr %230 : memref<1xbf16, 1>
                      ascvec.get_buf mte2, %231, 0 {cannir.buffer.generated_sync}
                      "ascvec.copy_gm2ub_plan"(%251, %alloc_0, %c0_i32, %c0_i32, %c0_i64, %250, %226, %c256_i64, %237, %c256_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                      ascvec.release_buf mte2, %231, 0 {cannir.buffer.generated_sync}
                      scf.yield %233 : i64
                    } else {
                      scf.yield %arg35 : i64
                    }
                    scf.yield %211, %223 : i64, i64
                  } else {
                    scf.yield %arg30, %arg35 : i64, i64
                  }
                  %182 = ascscalar.memref_address %alloc_20 : memref<8192xbf16, 2>
                  %183 = ascscalar.memref_address %alloc_19 : memref<8192xbf16, 2>
                  %184 = arith.select %168, %182, %183 : i64
                  %185 = arith.select %168, %c10_i64, %c11_i64 : i64
                  %186 = arith.subi %160, %149 : i64
                  %187 = arith.cmpi slt, %186, %c0_i64 : i64
                  %188 = arith.select %187, %c0_i64, %186 : i64
                  %189 = arith.cmpi sgt, %188, %c32_i64 : i64
                  %190 = arith.select %189, %c32_i64, %188 : i64
                  %191 = arith.cmpi eq, %190, %c32_i64 : i64
                  ascvec.sync_intra_wait v, %c5_i64 {cannir.sync.banks = 3 : i32}
                  %192 = scf.if %191 -> (i64) {
                    %207 = arith.cmpi eq, %arg36, %c0_i64 : i64
                    %208 = ascscalar.memref_address %alloc_15 : memref<4096xbf16, 1>
                    %209 = ascscalar.memref_address %alloc_16 : memref<4096xbf16, 1>
                    %210 = arith.select %207, %208, %209 : i64
                    %211 = ascscalar.bind_local_ptr %210 : memref<4096xbf16, 1>
                    %212 = arith.select %207, %c6_i64, %c7_i64 : i64
                    %213 = arith.addi %arg36, %c1_i64 : i64
                    %214 = arith.remui %213, %c2_i64 : i64
                    ascvec.get_buf v, %212, 0 {cannir.buffer.generated_sync}
                    ascvec.vf(%211, %alloc_7, %alloc_9 : memref<4096xbf16, 1>, memref<4096xf32, 1>, memref<4096xbf16, 1>) {
                    ^bb0(%arg37: memref<4096xbf16, 1>, %arg38: memref<4096xf32, 1>, %arg39: memref<4096xbf16, 1>):
                      %c128_i32_21 = arith.constant 128 : i32
                      %c4096_i32_22 = arith.constant 4096 : i32
                      %c0_i32_23 = arith.constant 0 : i32
                      %c64_i32_24 = arith.constant 64 : i32
                      %mask, %remaining_out = ascvec.update_mask %c64_i32_24 : !ascvec.mask
                      scf.for %arg40 = %c0_i32_23 to %c4096_i32_22 step %c128_i32_21  : i32 {
                        %215 = arith.index_cast %arg40 : i32 to index
                        %216 = ascvec.loadalign_unpack %arg37[%215] {mode = "b16_to_b32"} : memref<4096xbf16, 1>, index -> <bf16>
                        %217 = arith.addi %arg40, %c64_i32_24 overflow<nsw> : i32
                        %218 = arith.index_cast %217 : i32 to index
                        %219 = ascvec.loadalign_unpack %arg37[%218] {mode = "b16_to_b32"} : memref<4096xbf16, 1>, index -> <bf16>
                        %220 = ascvec.cast %216, %mask {rounding = 6 : i32} : <bf16> -> <f32>
                        %221 = ascvec.cast %219, %mask {rounding = 6 : i32} : <bf16> -> <f32>
                        %222 = ascvec.loadalign %arg38[%215] : memref<4096xf32, 1>, index -> <f32>
                        %223 = ascvec.loadalign %arg38[%218] : memref<4096xf32, 1>, index -> <f32>
                        %224 = ascvec.add %220, %222, %mask : <f32>
                        %225 = ascvec.add %221, %223, %mask : <f32>
                        %226 = ascvec.cast %224, %mask : <f32> -> <bf16>
                        %227 = ascvec.cast %225, %mask : <f32> -> <bf16>
                        ascvec.storealign_pack %arg39[%215], %226, %mask {mode = "b32_to_b16"} : memref<4096xbf16, 1>, index, <bf16>
                        ascvec.storealign_pack %arg39[%218], %227, %mask {mode = "b32_to_b16"} : memref<4096xbf16, 1>, index, <bf16>
                      }
                      ascvec.mem_bar vst_vld
                    } {cannir.vf_mode = "simd"}
                    ascvec.release_buf v, %212, 0 {cannir.buffer.generated_sync}
                    scf.yield %214 : i64
                  } else {
                    ascvec.vf(%alloc_9 : memref<4096xbf16, 1>) {
                    ^bb0(%arg37: memref<4096xbf16, 1>):
                      %c1_i32_21 = arith.constant 1 : i32
                      %c32_i32_22 = arith.constant 32 : i32
                      %c0_i32_23 = arith.constant 0 : i32
                      %cst = arith.constant 0.000000e+00 : bf16
                      %c128_i32_24 = arith.constant 128 : i32
                      %mask, %remaining_out = ascvec.update_mask %c128_i32_24 elem_bits 16 : !ascvec.mask
                      %207 = ascvec.duplicate_scalar %cst, %mask : bf16 -> <bf16>
                      scf.for %arg38 = %c0_i32_23 to %c32_i32_22 step %c1_i32_21  : i32 {
                        %208 = arith.muli %arg38, %c128_i32_24 overflow<nsw> : i32
                        %209 = arith.index_cast %208 : i32 to index
                        ascvec.storealign %arg37[%209], %207, %mask : memref<4096xbf16, 1>, index, <bf16>
                      }
                      ascvec.mem_bar vst_vld
                    } {cannir.vf_mode = "simd"}
                    scf.yield %arg36 : i64
                  }
                  %193 = arith.cmpi sgt, %190, %c0_i64 : i64
                  %194 = arith.cmpi slt, %190, %c32_i64 : i64
                  %195 = arith.select %193, %194, %false : i1
                  %196 = scf.if %195 -> (i64) {
                    %207 = arith.cmpi eq, %192, %c0_i64 : i64
                    %208 = ascscalar.memref_address %alloc_15 : memref<4096xbf16, 1>
                    %209 = ascscalar.memref_address %alloc_16 : memref<4096xbf16, 1>
                    %210 = arith.select %207, %208, %209 : i64
                    %211 = ascscalar.bind_local_ptr %210 : memref<4096xbf16, 1>
                    %212 = arith.select %207, %c6_i64, %c7_i64 : i64
                    %213 = arith.addi %192, %c1_i64 : i64
                    %214 = arith.remui %213, %c2_i64 : i64
                    ascvec.get_buf v, %212, 0 {cannir.buffer.generated_sync}
                    ascvec.vf(%211, %alloc_7, %alloc_9, %190 : memref<4096xbf16, 1>, memref<4096xf32, 1>, memref<4096xbf16, 1>, i64) {
                    ^bb0(%arg37: memref<4096xbf16, 1>, %arg38: memref<4096xf32, 1>, %arg39: memref<4096xbf16, 1>, %arg40: i64):
                      %c1_i64_21 = arith.constant 1 : i64
                      %c0_i64_22 = arith.constant 0 : i64
                      %c64_i32_23 = arith.constant 64 : i32
                      %c0_i32_24 = arith.constant 0 : i32
                      %c128_i64_25 = arith.constant 128 : i64
                      %c128_i32_26 = arith.constant 128 : i32
                      scf.for %arg41 = %c0_i64_22 to %arg40 step %c1_i64_21  : i64 {
                        scf.for %arg42 = %c0_i32_24 to %c128_i32_26 step %c64_i32_23  : i32 {
                          %215 = arith.subi %c128_i32_26, %arg42 overflow<nsw> : i32
                          %mask, %remaining_out = ascvec.update_mask %215 : !ascvec.mask
                          %216 = arith.muli %arg41, %c128_i64_25 : i64
                          %217 = arith.extsi %arg42 : i32 to i64
                          %218 = arith.addi %216, %217 : i64
                          %219 = arith.index_cast %218 : i64 to index
                          %220 = ascvec.loadalign_unpack %arg37[%219] {mode = "b16_to_b32"} : memref<4096xbf16, 1>, index -> <bf16>
                          %221 = ascvec.cast %220, %mask {rounding = 6 : i32} : <bf16> -> <f32>
                          %222 = ascvec.loadalign %arg38[%219] : memref<4096xf32, 1>, index -> <f32>
                          %223 = ascvec.add %221, %222, %mask : <f32>
                          %224 = ascvec.cast %223, %mask : <f32> -> <bf16>
                          ascvec.storealign_pack %arg39[%219], %224, %mask {mode = "b32_to_b16"} : memref<4096xbf16, 1>, index, <bf16>
                        }
                      } {cannir.loop_unroll = 2 : i64}
                      ascvec.mem_bar vst_vld
                    } {cannir.vf_mode = "simd"}
                    ascvec.release_buf v, %212, 0 {cannir.buffer.generated_sync}
                    scf.yield %214 : i64
                  } else {
                    scf.yield %192 : i64
                  }
                  ascvec.sync_intra_arrive v, %c5_i64 {cannir.sync.banks = 3 : i32}
                  ascvec.get_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
                  ascvec.vf(%alloc_9, %alloc_12 : memref<4096xbf16, 1>, memref<4224xbf16, 1>) {
                  ^bb0(%arg37: memref<4096xbf16, 1>, %arg38: memref<4224xbf16, 1>):
                    %c1_i32_21 = arith.constant 1 : i32
                    %c32_i32_22 = arith.constant 32 : i32
                    %c0_i32_23 = arith.constant 0 : i32
                    %c33 = arith.constant 33 : index
                    %c16_i32_24 = arith.constant 16 : i32
                    %c0_25 = arith.constant 0 : index
                    %c128_i32_26 = arith.constant 128 : i32
                    scf.for %arg39 = %c0_i32_23 to %c32_i32_22 step %c1_i32_21  : i32 {
                      %mask, %remaining_out = ascvec.update_mask %c128_i32_26 elem_bits 16 : !ascvec.mask
                      %207 = arith.muli %arg39, %c128_i32_26 overflow<nsw> : i32
                      %208 = arith.index_cast %207 : i32 to index
                      %209 = ascvec.loadalign %arg37[%208] : memref<4096xbf16, 1>, index -> <bf16>
                      %210 = arith.muli %arg39, %c16_i32_24 overflow<nsw> : i32
                      %211 = arith.index_cast %210 : i32 to index
                      ascvec.storealign_strided %arg38[%211], %209, %mask, %c33, %c0_25 : memref<4224xbf16, 1>, <bf16>
                    }
                  } {cannir.vf_mode = "simd"}
                  ascvec.release_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
                  %197 = ascscalar.bitcast_memref %alloc_12 byte_offset = %c0_i64 : memref<4224xbf16, 1> -> memref<?xbf16, 1>
                  %198 = ascscalar.bind_local_ptr %184 : memref<1xbf16, 2>
                  %199 = arith.muli %96, %c512_i64 : i64
                  %200 = arith.cmpi eq, %185, %c11_i64 : i64
                  %201 = arith.select %200, %c3_i64, %c2_i64 : i64
                  ascvec.sync_intra_wait mte3, %201 {cannir.sync.banks = 3 : i32}
                  ascvec.get_buf mte3, %c3_i64, 0 {cannir.buffer.generated_sync}
                  %202 = arith.index_cast %199 : i64 to index
                  ascvec.copy_ub2l1_strided %198[%202], %197[%c0], %c8_i32, %c32_i32, %c1_i32, %c32_i32 : memref<1xbf16, 2>, memref<?xbf16, 1>
                  ascvec.release_buf mte3, %c3_i64, 0 {cannir.buffer.generated_sync}
                  ascvec.sync_intra_arrive mte3, %201 {cannir.sync.banks = 3 : i32}
                  %203 = scf.if %180 -> (i64) {
                    %207 = arith.addi %arg32, %c1_i64 overflow<nsw> : i64
                    %208 = arith.remui %207, %c2_i64 : i64
                    scf.yield %208 : i64
                  } else {
                    scf.yield %arg32 : i64
                  }
                  %204 = arith.select %168, %124, %125 : i64
                  %205 = ascscalar.bind_local_ptr %204 : memref<64xf32, 1>
                  ascvec.sync_intra_wait v, %c4_i64 {cannir.sync.banks = 3 : i32}
                  ascvec.get_buf v, %178, 0 {cannir.buffer.generated_sync}
                  ascvec.vf(%205, %alloc_8, %alloc_6 : memref<64xf32, 1>, memref<8192xf32, 1>, memref<8192xf32, 1>) {
                  ^bb0(%arg37: memref<64xf32, 1>, %arg38: memref<8192xf32, 1>, %arg39: memref<8192xf32, 1>):
                    %c8192_i32_21 = arith.constant 8192 : i32
                    %c0_i32_22 = arith.constant 0 : i32
                    %c0_23 = arith.constant 0 : index
                    %c64_i32_24 = arith.constant 64 : i32
                    %mask, %remaining_out = ascvec.update_mask %c64_i32_24 : !ascvec.mask
                    %207 = ascvec.loadalign %arg37[%c0_23] : memref<64xf32, 1>, index -> <f32>
                    scf.for %arg40 = %c0_i32_22 to %c8192_i32_21 step %c64_i32_24  : i32 {
                      %208 = arith.index_cast %arg40 : i32 to index
                      %209 = ascvec.loadalign %arg38[%208] : memref<8192xf32, 1>, index -> <f32>
                      %210 = ascvec.loadalign %arg39[%208] : memref<8192xf32, 1>, index -> <f32>
                      %211 = ascvec.mul %209, %207, %mask : <f32>
                      %212 = ascvec.add %211, %210, %mask : <f32>
                      ascvec.storealign %arg38[%208], %212, %mask : memref<8192xf32, 1>, index, <f32>
                    } {cannir.loop_unroll = 4 : i64}
                  } {cannir.vf_mode = "simd"}
                  ascvec.release_buf v, %178, 0 {cannir.buffer.generated_sync}
                  ascvec.sync_intra_arrive v, %c4_i64 {cannir.sync.banks = 3 : i32}
                  %206 = scf.if %180 -> (i64) {
                    ascvec.vf(%alloc_8, %alloc_10 : memref<8192xf32, 1>, memref<8192xbf16, 1>) {
                    ^bb0(%arg37: memref<8192xf32, 1>, %arg38: memref<8192xbf16, 1>):
                      %c1_i32_21 = arith.constant 1 : i32
                      %c128_i32_22 = arith.constant 128 : i32
                      %c0_i32_23 = arith.constant 0 : i32
                      %c64_i32_24 = arith.constant 64 : i32
                      scf.for %arg39 = %c0_i32_23 to %c128_i32_22 step %c1_i32_21  : i32 {
                        %mask, %remaining_out = ascvec.update_mask %c64_i32_24 : !ascvec.mask
                        %218 = arith.muli %arg39, %c64_i32_24 overflow<nsw> : i32
                        %219 = arith.index_cast %218 : i32 to index
                        %220 = ascvec.loadalign %arg37[%219] : memref<8192xf32, 1>, index -> <f32>
                        %221 = ascvec.cast %220, %mask : <f32> -> <bf16>
                        ascvec.storealign_pack %arg38[%219], %221, %mask {mode = "b32_to_b16"} : memref<8192xbf16, 1>, index, <bf16>
                      }
                    } {cannir.vf_mode = "simd"}
                    ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                    ascvec.vf(%alloc_10, %alloc_11 : memref<8192xbf16, 1>, memref<8256xbf16, 1>) {
                    ^bb0(%arg37: memref<8192xbf16, 1>, %arg38: memref<8256xbf16, 1>):
                      %c1_i32_21 = arith.constant 1 : i32
                      %c128_i32_22 = arith.constant 128 : i32
                      %c0_i32_23 = arith.constant 0 : i32
                      %c129 = arith.constant 129 : index
                      %c16_i32_24 = arith.constant 16 : i32
                      %c0_25 = arith.constant 0 : index
                      %c64_i32_26 = arith.constant 64 : i32
                      scf.for %arg39 = %c0_i32_23 to %c128_i32_22 step %c1_i32_21  : i32 {
                        %mask, %remaining_out = ascvec.update_mask %c64_i32_26 elem_bits 16 : !ascvec.mask
                        %218 = arith.muli %arg39, %c64_i32_26 overflow<nsw> : i32
                        %219 = arith.index_cast %218 : i32 to index
                        %220 = ascvec.loadalign %arg37[%219] : memref<8192xbf16, 1>, index -> <bf16>
                        %221 = arith.muli %arg39, %c16_i32_24 overflow<nsw> : i32
                        %222 = arith.index_cast %221 : i32 to index
                        ascvec.storealign_strided %arg38[%222], %220, %mask, %c129, %c0_25 : memref<8256xbf16, 1>, <bf16>
                      }
                    } {cannir.vf_mode = "simd"}
                    ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                    %207 = arith.cmpi eq, %arg33, %c0_i64 : i64
                    %208 = arith.select %207, %169, %170 : i64
                    %209 = arith.select %207, %c8_i64, %c9_i64 : i64
                    %210 = arith.addi %arg33, %c1_i64 overflow<nsw> : i64
                    %211 = arith.remui %210, %c2_i64 : i64
                    %212 = ascscalar.bitcast_memref %alloc_11 byte_offset = %c0_i64 : memref<8256xbf16, 1> -> memref<?xbf16, 1>
                    %213 = ascscalar.bind_local_ptr %208 : memref<1xbf16, 2>
                    %214 = arith.muli %96, %c8192_i64 : i64
                    %215 = arith.cmpi eq, %209, %c9_i64 : i64
                    %216 = arith.extui %215 : i1 to i64
                    ascvec.sync_intra_wait mte3, %216 {cannir.sync.banks = 3 : i32}
                    ascvec.get_buf mte3, %c2_i64, 0 {cannir.buffer.generated_sync}
                    %217 = arith.index_cast %214 : i64 to index
                    ascvec.copy_ub2l1_strided %213[%217], %212[%c0], %c4_i32, %c128_i32, %c1_i32, %c0_i32 : memref<1xbf16, 2>, memref<?xbf16, 1>
                    ascvec.release_buf mte3, %c2_i64, 0 {cannir.buffer.generated_sync}
                    ascvec.sync_intra_arrive mte3, %216 {cannir.sync.banks = 3 : i32}
                    scf.yield %211 : i64
                  } else {
                    ascvec.get_buf v, %c5_i64, 0 {cannir.buffer.generated_sync}
                    ascvec.copy_ub2ub %alloc_14[%c0], %alloc_8[%c0], %c32768_i32 : memref<8192xf32, 1>, memref<8192xf32, 1>
                    ascvec.vf(%alloc_14, %alloc_8 : memref<8192xf32, 1>, memref<8192xf32, 1>) {
                    ^bb0(%arg37: memref<8192xf32, 1>, %arg38: memref<8192xf32, 1>):
                    } {cannir.vf_mode = "simd"}
                    ascvec.release_buf v, %c5_i64, 0 {cannir.buffer.generated_sync}
                    %207 = cannir.extract_physical_layout %arg8 : !cannir.tensor<<gm, f32>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, identity>
                    %208 = cannir.extract_stride %207 : <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>> -> <(-1, 16384, 128, 1)>
                    %209 = cannir.extract_dim %208[0] : !cannir.stride<(-1, 16384, 128, 1)>
                    %210 = arith.muli %105, %209 : i64
                    %211 = arith.muli %104, %c16384_i64 : i64
                    %212 = arith.addi %210, %211 : i64
                    %213 = arith.muli %96, %c64_i64 : i64
                    %214 = arith.addi %212, %213 : i64
                    %215 = arith.muli %214, %c32_i64 : i64
                    %216 = ascscalar.memref_address %alloc_2 : memref<1xf32>
                    %217 = arith.divui %215, %c8_i64 : i64
                    %218 = arith.addi %216, %217 : i64
                    %219 = ascscalar.bind_gm_ptr %218 : memref<1xf32>
                    %220 = ascscalar.bitcast_memref %alloc_14 byte_offset = %c0_i64 : memref<8192xf32, 1> -> memref<?xf32, 1>
                    ascvec.get_buf mte3, %c5_i64, 0 {cannir.buffer.generated_sync}
                    ascvec.copy_ub2gm %219[%c0], %220[%c0], %c128_i32, %c256_i32, %c512_i64, %c256_i32 l2_cache_ctl %c0_i32 : memref<1xf32>, memref<?xf32, 1>
                    ascvec.release_buf mte3, %c5_i64, 0 {cannir.buffer.generated_sync}
                    scf.yield %arg33 : i64
                  }
                  scf.yield %181#0, %172, %203, %206, %176, %181#1, %196 : i64, i64, i64, i64, i64, i64, i64
                }
                scf.yield %152#0, %152#1, %152#2, %152#3, %152#4, %152#5, %152#6 : i64, i64, i64, i64, i64, i64, i64
              }
              scf.yield %99#0, %99#1, %99#2, %99#3, %99#4, %99#5, %99#6 : i64, i64, i64, i64, i64, i64, i64
            } else {
              scf.yield %arg14, %arg15, %arg16, %arg17, %arg18, %arg19, %arg20 : i64, i64, i64, i64, i64, i64, i64
            }
            scf.yield %98#0, %98#1, %98#2, %98#3, %98#4, %98#5, %98#6 : i64, i64, i64, i64, i64, i64, i64
          }
        } else {
          %49 = arith.cmpi eq, %44, %c2_i32 : i32
          scf.if %49 {
            %50:7 = scf.for %arg13 = %c0_i32 to %21 step %c1_i32 iter_args(%arg14 = %c0_i64, %arg15 = %c0_i64, %arg16 = %c0_i64, %arg17 = %c0_i64, %arg18 = %c0_i64, %arg19 = %c0_i64, %arg20 = %c0_i64) -> (i64, i64, i64, i64, i64, i64, i64)  : i32 {
              %51 = arith.addi %32, %arg13 : i32
              %52 = arith.index_cast %51 : i32 to index
              %53 = memref.load %alloc_3[%52] : memref<1xi32>
              %54 = arith.addi %51, %c1_i32 : i32
              %55 = arith.index_cast %54 : i32 to index
              %56 = memref.load %alloc_3[%55] : memref<1xi32>
              %57 = arith.cmpi slt, %39, %53 : i32
              %58 = arith.select %57, %53, %39 : i32
              %59 = arith.cmpi sgt, %42, %56 : i32
              %60 = arith.select %59, %56, %42 : i32
              %61 = arith.cmpi slt, %58, %60 : i32
              %62 = arith.subi %58, %53 : i32
              %63 = arith.subi %60, %53 : i32
              %64 = arith.extsi %62 : i32 to i64
              %65 = arith.extsi %63 : i32 to i64
              %66 = arith.select %61, %64, %c0_i64 : i64
              %67 = arith.select %61, %65, %c0_i64 : i64
              %68 = arith.addi %22, %arg13 : i32
              %69 = arith.index_cast %68 : i32 to index
              %70 = memref.load %alloc_3[%69] : memref<1xi32>
              %71 = arith.addi %23, %arg13 : i32
              %72 = arith.index_cast %71 : i32 to index
              %73 = memref.load %alloc_3[%72] : memref<1xi32>
              %74 = arith.addi %24, %arg13 : i32
              %75 = arith.index_cast %74 : i32 to index
              %76 = memref.load %alloc_3[%75] : memref<1xi32>
              %77 = arith.addi %25, %arg13 : i32
              %78 = arith.index_cast %77 : i32 to index
              %79 = memref.load %alloc_3[%78] : memref<1xi32>
              %80 = arith.extsi %79 : i32 to i64
              %81 = arith.addi %27, %arg13 : i32
              %82 = arith.index_cast %81 : i32 to index
              %83 = memref.load %alloc_3[%82] : memref<1xi32>
              %84 = arith.addi %28, %arg13 : i32
              %85 = arith.index_cast %84 : i32 to index
              %86 = memref.load %alloc_3[%85] : memref<1xi32>
              %87 = arith.addi %80, %c63_i64 : i64
              %88 = arith.remsi %87, %c64_i64 : i64
              %89 = arith.cmpi ne, %88, %c0_i64 : i64
              %90 = arith.cmpi slt, %88, %c0_i64 : i64
              %91 = arith.andi %89, %90 : i1
              %92 = arith.divsi %87, %c64_i64 : i64
              %93 = arith.extui %91 : i1 to i64
              %94 = arith.subi %92, %93 : i64
              %95 = arith.extsi %76 : i32 to i64
              %96 = arith.extsi %73 : i32 to i64
              %97 = ascscalar.get_subblock_id
              %98 = arith.cmpi slt, %0, %1 : i64
              %99:7 = scf.if %98 -> (i64, i64, i64, i64, i64, i64, i64) {
                %100:7 = scf.for %arg21 = %66 to %67 step %c1_i64 iter_args(%arg22 = %arg14, %arg23 = %arg15, %arg24 = %arg16, %arg25 = %arg17, %arg26 = %arg18, %arg27 = %arg19, %arg28 = %arg20) -> (i64, i64, i64, i64, i64, i64, i64)  : i64 {
                  %101 = arith.remsi %arg21, %c2_i64 : i64
                  %102 = arith.cmpi slt, %101, %c0_i64 : i64
                  %103 = arith.divsi %arg21, %c2_i64 : i64
                  %104 = arith.extui %102 : i1 to i64
                  %105 = arith.subi %103, %104 : i64
                  %106 = arith.maxsi %7, %c1_i64 : i64
                  %107 = arith.remsi %105, %106 : i64
                  %108 = arith.cmpi slt, %107, %c0_i64 : i64
                  %109 = arith.select %108, %106, %c0_i64 : i64
                  %110 = arith.addi %107, %109 : i64
                  %111 = arith.select %102, %c2_i64, %c0_i64 : i64
                  %112 = arith.addi %101, %111 overflow<nsw> : i64
                  %113 = arith.extsi %70 : i32 to i64
                  %114 = arith.addi %83, %86 : i32
                  %115 = arith.extsi %114 : i32 to i64
                  %116 = arith.cmpi eq, %115, %94 : i64
                  %117 = arith.muli %83, %c64_i32 : i32
                  %118 = arith.extsi %117 : i32 to i64
                  %119 = arith.subi %80, %118 : i64
                  %120 = arith.minsi %119, %c64_i64 : i64
                  %121 = arith.select %116, %120, %c64_i64 : i64
                  %122 = arith.extsi %83 : i32 to i64
                  %123 = arith.cmpi eq, %arg22, %c0_i64 : i64
                  %124 = arith.addi %arg22, %c1_i64 : i64
                  %125 = arith.remui %124, %c2_i64 : i64
                  %126 = arith.cmpi ne, %arg22, %c0_i64 : i64
                  %127 = arith.extui %126 : i1 to i64
                  %128 = arith.addi %arg24, %c1_i64 : i64
                  %129 = arith.remui %128, %c2_i64 : i64
                  %130 = arith.cmpi eq, %83, %c0_i32 : i32
                  %131 = scf.if %130 -> (i64) {
                    %161 = cannir.extract_physical_layout %arg7 : !cannir.tensor<<gm, f32>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, identity>
                    %162 = cannir.extract_stride %161 : <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>> -> <(-1, 16384, 128, 1)>
                    %163 = cannir.extract_dim %162[0] : !cannir.stride<(-1, 16384, 128, 1)>
                    %164 = arith.muli %113, %163 : i64
                    %165 = arith.muli %164, %c4_i64 : i64
                    %166 = arith.muli %110, %c65536_i64 : i64
                    %167 = arith.addi %165, %166 : i64
                    %168 = arith.muli %112, %c32768_i64 overflow<nsw> : i64
                    %169 = arith.addi %167, %168 : i64
                    %170 = arith.muli %97, %c256_i64 : i64
                    %171 = arith.addi %169, %170 : i64
                    ascvec.get_buf mte2, %c4_i64, 0 {cannir.buffer.generated_sync}
                    "ascvec.copy_gm2ub_plan"(%45, %alloc_1, %c0_i32, %c0_i32, %c0_i64, %171, %c64_i64, %c256_i64, %c512_i64, %c256_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 131968 : i64, ub_capacity = 32768 : i64}> : (memref<?xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                    ascvec.release_buf mte2, %c4_i64, 0 {cannir.buffer.generated_sync}
                    ascvec.get_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
                    ascvec.copy_ub2ub %alloc_8[%c0], %alloc_13[%c0], %c16384_i32 : memref<8192xf32, 1>, memref<8192xf32, 1>
                    ascvec.vf(%alloc_8, %alloc_13 : memref<8192xf32, 1>, memref<8192xf32, 1>) {
                    ^bb0(%arg29: memref<8192xf32, 1>, %arg30: memref<8192xf32, 1>):
                    } {cannir.vf_mode = "simd"}
                    ascvec.release_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
                    ascvec.vf(%alloc_8, %alloc_10 : memref<8192xf32, 1>, memref<8192xbf16, 1>) {
                    ^bb0(%arg29: memref<8192xf32, 1>, %arg30: memref<8192xbf16, 1>):
                      %c1_i32_21 = arith.constant 1 : i32
                      %c0_i32_22 = arith.constant 0 : i32
                      %c64_i32_23 = arith.constant 64 : i32
                      scf.for %arg31 = %c0_i32_22 to %c64_i32_23 step %c1_i32_21  : i32 {
                        %mask, %remaining_out = ascvec.update_mask %c64_i32_23 : !ascvec.mask
                        %185 = arith.muli %arg31, %c64_i32_23 overflow<nsw> : i32
                        %186 = arith.index_cast %185 : i32 to index
                        %187 = ascvec.loadalign %arg29[%186] : memref<8192xf32, 1>, index -> <f32>
                        %188 = ascvec.cast %187, %mask : <f32> -> <bf16>
                        ascvec.storealign_pack %arg30[%186], %188, %mask {mode = "b32_to_b16"} : memref<8192xbf16, 1>, index, <bf16>
                      }
                    } {cannir.vf_mode = "simd"}
                    ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                    ascvec.vf(%alloc_10, %alloc_11 : memref<8192xbf16, 1>, memref<8256xbf16, 1>) {
                    ^bb0(%arg29: memref<8192xbf16, 1>, %arg30: memref<8256xbf16, 1>):
                      %c1_i32_21 = arith.constant 1 : i32
                      %c0_i32_22 = arith.constant 0 : i32
                      %c65 = arith.constant 65 : index
                      %c16_i32_23 = arith.constant 16 : i32
                      %c0_24 = arith.constant 0 : index
                      %c64_i32_25 = arith.constant 64 : i32
                      scf.for %arg31 = %c0_i32_22 to %c64_i32_25 step %c1_i32_21  : i32 {
                        %mask, %remaining_out = ascvec.update_mask %c64_i32_25 elem_bits 16 : !ascvec.mask
                        %185 = arith.muli %arg31, %c64_i32_25 overflow<nsw> : i32
                        %186 = arith.index_cast %185 : i32 to index
                        %187 = ascvec.loadalign %arg29[%186] : memref<8192xbf16, 1>, index -> <bf16>
                        %188 = arith.muli %arg31, %c16_i32_23 overflow<nsw> : i32
                        %189 = arith.index_cast %188 : i32 to index
                        ascvec.storealign_strided %arg30[%189], %187, %mask, %c65, %c0_24 : memref<8256xbf16, 1>, <bf16>
                      }
                    } {cannir.vf_mode = "simd"}
                    ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                    %172 = arith.cmpi eq, %arg25, %c0_i64 : i64
                    %173 = ascscalar.memref_address %alloc_18 : memref<16384xbf16, 2>
                    %174 = ascscalar.memref_address %alloc_17 : memref<16384xbf16, 2>
                    %175 = arith.select %172, %173, %174 : i64
                    %176 = arith.select %172, %c8_i64, %c9_i64 : i64
                    %177 = arith.addi %arg25, %c1_i64 : i64
                    %178 = arith.remui %177, %c2_i64 : i64
                    %179 = ascscalar.bitcast_memref %alloc_11 byte_offset = %c0_i64 : memref<8256xbf16, 1> -> memref<?xbf16, 1>
                    %180 = ascscalar.bind_local_ptr %175 : memref<1xbf16, 2>
                    %181 = arith.muli %97, %c4096_i64 : i64
                    %182 = arith.cmpi eq, %176, %c9_i64 : i64
                    %183 = arith.extui %182 : i1 to i64
                    ascvec.sync_intra_wait mte3, %183 {cannir.sync.banks = 3 : i32}
                    ascvec.get_buf mte3, %c2_i64, 0 {cannir.buffer.generated_sync}
                    %184 = arith.index_cast %181 : i64 to index
                    ascvec.copy_ub2l1_strided %180[%184], %179[%c0], %c4_i32, %c64_i32, %c1_i32, %c0_i32 : memref<1xbf16, 2>, memref<?xbf16, 1>
                    ascvec.release_buf mte3, %c2_i64, 0 {cannir.buffer.generated_sync}
                    ascvec.sync_intra_arrive mte3, %183 {cannir.sync.banks = 3 : i32}
                    scf.yield %178 : i64
                  } else {
                    %161 = cannir.extract_physical_layout %arg8 : !cannir.tensor<<gm, f32>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, identity>
                    %162 = cannir.extract_stride %161 : <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>> -> <(-1, 16384, 128, 1)>
                    %163 = cannir.extract_dim %162[0] : !cannir.stride<(-1, 16384, 128, 1)>
                    %164 = arith.muli %113, %163 : i64
                    %165 = arith.muli %164, %c4_i64 : i64
                    %166 = arith.muli %110, %c65536_i64 : i64
                    %167 = arith.addi %165, %166 : i64
                    %168 = arith.muli %112, %c32768_i64 overflow<nsw> : i64
                    %169 = arith.addi %167, %168 : i64
                    %170 = arith.muli %97, %c256_i64 : i64
                    %171 = arith.addi %169, %170 : i64
                    ascvec.get_buf mte2, %c4_i64, 0 {cannir.buffer.generated_sync}
                    "ascvec.copy_gm2ub_plan"(%45, %alloc_2, %c0_i32, %c0_i32, %c0_i64, %171, %c64_i64, %c256_i64, %c512_i64, %c256_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 131968 : i64, ub_capacity = 32768 : i64}> : (memref<?xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                    ascvec.release_buf mte2, %c4_i64, 0 {cannir.buffer.generated_sync}
                    ascvec.get_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
                    ascvec.copy_ub2ub %alloc_8[%c0], %alloc_13[%c0], %c16384_i32 : memref<8192xf32, 1>, memref<8192xf32, 1>
                    ascvec.vf(%alloc_8, %alloc_13 : memref<8192xf32, 1>, memref<8192xf32, 1>) {
                    ^bb0(%arg29: memref<8192xf32, 1>, %arg30: memref<8192xf32, 1>):
                    } {cannir.vf_mode = "simd"}
                    ascvec.release_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
                    ascvec.vf(%alloc_8, %alloc_10 : memref<8192xf32, 1>, memref<8192xbf16, 1>) {
                    ^bb0(%arg29: memref<8192xf32, 1>, %arg30: memref<8192xbf16, 1>):
                      %c1_i32_21 = arith.constant 1 : i32
                      %c0_i32_22 = arith.constant 0 : i32
                      %c64_i32_23 = arith.constant 64 : i32
                      scf.for %arg31 = %c0_i32_22 to %c64_i32_23 step %c1_i32_21  : i32 {
                        %mask, %remaining_out = ascvec.update_mask %c64_i32_23 : !ascvec.mask
                        %185 = arith.muli %arg31, %c64_i32_23 overflow<nsw> : i32
                        %186 = arith.index_cast %185 : i32 to index
                        %187 = ascvec.loadalign %arg29[%186] : memref<8192xf32, 1>, index -> <f32>
                        %188 = ascvec.cast %187, %mask : <f32> -> <bf16>
                        ascvec.storealign_pack %arg30[%186], %188, %mask {mode = "b32_to_b16"} : memref<8192xbf16, 1>, index, <bf16>
                      }
                    } {cannir.vf_mode = "simd"}
                    ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                    ascvec.vf(%alloc_10, %alloc_11 : memref<8192xbf16, 1>, memref<8256xbf16, 1>) {
                    ^bb0(%arg29: memref<8192xbf16, 1>, %arg30: memref<8256xbf16, 1>):
                      %c1_i32_21 = arith.constant 1 : i32
                      %c0_i32_22 = arith.constant 0 : i32
                      %c65 = arith.constant 65 : index
                      %c16_i32_23 = arith.constant 16 : i32
                      %c0_24 = arith.constant 0 : index
                      %c64_i32_25 = arith.constant 64 : i32
                      scf.for %arg31 = %c0_i32_22 to %c64_i32_25 step %c1_i32_21  : i32 {
                        %mask, %remaining_out = ascvec.update_mask %c64_i32_25 elem_bits 16 : !ascvec.mask
                        %185 = arith.muli %arg31, %c64_i32_25 overflow<nsw> : i32
                        %186 = arith.index_cast %185 : i32 to index
                        %187 = ascvec.loadalign %arg29[%186] : memref<8192xbf16, 1>, index -> <bf16>
                        %188 = arith.muli %arg31, %c16_i32_23 overflow<nsw> : i32
                        %189 = arith.index_cast %188 : i32 to index
                        ascvec.storealign_strided %arg30[%189], %187, %mask, %c65, %c0_24 : memref<8256xbf16, 1>, <bf16>
                      }
                    } {cannir.vf_mode = "simd"}
                    ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                    %172 = arith.cmpi eq, %arg25, %c0_i64 : i64
                    %173 = ascscalar.memref_address %alloc_18 : memref<16384xbf16, 2>
                    %174 = ascscalar.memref_address %alloc_17 : memref<16384xbf16, 2>
                    %175 = arith.select %172, %173, %174 : i64
                    %176 = arith.select %172, %c8_i64, %c9_i64 : i64
                    %177 = arith.addi %arg25, %c1_i64 : i64
                    %178 = arith.remui %177, %c2_i64 : i64
                    %179 = ascscalar.bitcast_memref %alloc_11 byte_offset = %c0_i64 : memref<8256xbf16, 1> -> memref<?xbf16, 1>
                    %180 = ascscalar.bind_local_ptr %175 : memref<1xbf16, 2>
                    %181 = arith.muli %97, %c4096_i64 : i64
                    %182 = arith.cmpi eq, %176, %c9_i64 : i64
                    %183 = arith.extui %182 : i1 to i64
                    ascvec.sync_intra_wait mte3, %183 {cannir.sync.banks = 3 : i32}
                    ascvec.get_buf mte3, %c2_i64, 0 {cannir.buffer.generated_sync}
                    %184 = arith.index_cast %181 : i64 to index
                    ascvec.copy_ub2l1_strided %180[%184], %179[%c0], %c4_i32, %c64_i32, %c1_i32, %c0_i32 : memref<1xbf16, 2>, memref<?xbf16, 1>
                    ascvec.release_buf mte3, %c2_i64, 0 {cannir.buffer.generated_sync}
                    ascvec.sync_intra_arrive mte3, %183 {cannir.sync.banks = 3 : i32}
                    scf.yield %178 : i64
                  }
                  %132 = ascscalar.memref_address %alloc_4 : memref<64xf32, 1>
                  %133 = ascscalar.memref_address %alloc_5 : memref<64xf32, 1>
                  %134 = arith.select %123, %132, %133 : i64
                  %135 = cannir.extract_layout %arg1 : !cannir.tensor<<gm, f32>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                  %136 = cannir.extract_shape %135 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, -1, 128)>
                  %137 = cannir.extract_dim %136[2] : !cannir.shape<(-1, -1, -1, 128)>
                  %138 = arith.minsi %122, %137 : i64
                  %139 = arith.subi %137, %138 : i64
                  %140 = arith.minsi %139, %c1_i64 : i64
                  %141 = arith.cmpi sle, %122, %137 : i64
                  %142 = arith.select %141, %140, %c0_i64 : i64
                  %143 = cannir.extract_physical_layout %arg1 : !cannir.tensor<<gm, f32>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                  %144 = cannir.extract_stride %143 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, 128, 1)>
                  %145 = cannir.extract_dim %144[0] : !cannir.stride<(-1, -1, 128, 1)>
                  %146 = arith.muli %113, %145 : i64
                  %147 = arith.muli %146, %c4_i64 : i64
                  %148 = cannir.extract_dim %144[1] : !cannir.stride<(-1, -1, 128, 1)>
                  %149 = arith.muli %110, %148 : i64
                  %150 = arith.muli %149, %c4_i64 : i64
                  %151 = arith.addi %147, %150 : i64
                  %152 = arith.muli %122, %c512_i64 : i64
                  %153 = arith.addi %151, %152 : i64
                  %154 = arith.muli %97, %c256_i64 : i64
                  %155 = arith.addi %153, %154 : i64
                  %156 = ascscalar.bind_local_ptr %134 : memref<1xf32, 1>
                  ascvec.get_buf mte2, %127, 0 {cannir.buffer.generated_sync}
                  "ascvec.copy_gm2ub_plan"(%156, %alloc, %c0_i32, %c0_i32, %c0_i64, %155, %142, %c256_i64, %c512_i64, %c256_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = -1 : i64, ub_capacity = 4 : i64}> : (memref<1xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                  ascvec.release_buf mte2, %127, 0 {cannir.buffer.generated_sync}
                  %157 = arith.muli %97, %c32_i64 : i64
                  %158 = arith.cmpi slt, %157, %121 : i64
                  %159 = scf.if %158 -> (i64) {
                    %161 = arith.subi %121, %157 : i64
                    %162 = arith.cmpi sgt, %161, %c32_i64 : i64
                    %163 = arith.select %162, %c32_i64, %161 : i64
                    %164 = arith.cmpi eq, %arg27, %c0_i64 : i64
                    %165 = ascscalar.memref_address %alloc_15 : memref<4096xbf16, 1>
                    %166 = ascscalar.memref_address %alloc_16 : memref<4096xbf16, 1>
                    %167 = arith.select %164, %165, %166 : i64
                    %168 = arith.select %164, %c6_i64, %c7_i64 : i64
                    %169 = arith.addi %arg27, %c1_i64 : i64
                    %170 = arith.remui %169, %c2_i64 : i64
                    %171 = cannir.extract_physical_layout %arg4 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                    %172 = cannir.extract_stride %171 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                    %173 = cannir.extract_dim %172[2] : !cannir.stride<(-1, -1, -1, 1)>
                    %174 = arith.muli %173, %c2_i64 : i64
                    %175 = cannir.extract_dim %172[0] : !cannir.stride<(-1, -1, -1, 1)>
                    %176 = arith.muli %96, %175 : i64
                    %177 = arith.muli %176, %c2_i64 : i64
                    %178 = cannir.extract_dim %172[1] : !cannir.stride<(-1, -1, -1, 1)>
                    %179 = arith.muli %110, %178 : i64
                    %180 = arith.muli %179, %c2_i64 : i64
                    %181 = arith.addi %177, %180 : i64
                    %182 = arith.muli %112, %c128_i64 overflow<nsw> : i64
                    %183 = arith.addi %181, %182 : i64
                    %184 = arith.muli %122, %c64_i64 : i64
                    %185 = arith.addi %157, %184 : i64
                    %186 = arith.addi %185, %95 : i64
                    %187 = arith.muli %186, %173 : i64
                    %188 = arith.muli %187, %c2_i64 : i64
                    %189 = arith.addi %183, %188 : i64
                    %190 = ascscalar.bind_local_ptr %167 : memref<1xbf16, 1>
                    ascvec.get_buf mte2, %168, 0 {cannir.buffer.generated_sync}
                    "ascvec.copy_gm2ub_plan"(%190, %alloc_0, %c0_i32, %c0_i32, %c0_i64, %189, %163, %c128_i64, %174, %c128_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                    ascvec.release_buf mte2, %168, 0 {cannir.buffer.generated_sync}
                    scf.yield %170 : i64
                  } else {
                    scf.yield %arg27 : i64
                  }
                  %160:7 = scf.for %arg29 = %c0_i32 to %86 step %c1_i32 iter_args(%arg30 = %125, %arg31 = %arg23, %arg32 = %129, %arg33 = %131, %arg34 = %arg26, %arg35 = %159, %arg36 = %arg28) -> (i64, i64, i64, i64, i64, i64, i64)  : i32 {
                    %161 = arith.addi %83, %arg29 : i32
                    %162 = arith.addi %161, %c1_i32 : i32
                    %163 = arith.extsi %162 : i32 to i64
                    %164 = arith.cmpi eq, %163, %94 : i64
                    %165 = arith.muli %161, %c64_i32 : i32
                    %166 = arith.extsi %165 : i32 to i64
                    %167 = arith.subi %80, %166 : i64
                    %168 = arith.select %164, %167, %c64_i64 : i64
                    %169 = arith.addi %161, %c2_i32 : i32
                    %170 = arith.extsi %169 : i32 to i64
                    %171 = arith.cmpi eq, %170, %94 : i64
                    %172 = arith.muli %162, %c64_i32 : i32
                    %173 = arith.extsi %172 : i32 to i64
                    %174 = arith.subi %80, %173 : i64
                    %175 = arith.select %171, %174, %c64_i64 : i64
                    %176 = arith.cmpi eq, %arg31, %c0_i64 : i64
                    %177 = ascscalar.memref_address %alloc_18 : memref<16384xbf16, 2>
                    %178 = ascscalar.memref_address %alloc_17 : memref<16384xbf16, 2>
                    %179 = arith.addi %arg31, %c1_i64 : i64
                    %180 = arith.remui %179, %c2_i64 : i64
                    %181 = arith.addi %arg34, %c1_i64 : i64
                    %182 = arith.remui %181, %c2_i64 : i64
                    %183 = arith.addi %182, %c1_i64 overflow<nsw> : i64
                    %184 = arith.remui %183, %c2_i64 : i64
                    %185 = arith.cmpi ne, %arg31, %c0_i64 : i64
                    %186 = arith.extui %185 : i1 to i64
                    %187 = arith.addi %arg29, %c1_i32 : i32
                    %188 = arith.cmpi slt, %187, %86 : i32
                    %189:2 = scf.if %188 -> (i64, i64) {
                      %215 = arith.cmpi eq, %arg30, %c0_i64 : i64
                      %216 = arith.cmpi ne, %arg30, %c0_i64 : i64
                      %217 = arith.extui %216 : i1 to i64
                      %218 = arith.addi %arg30, %c1_i64 overflow<nsw> : i64
                      %219 = arith.remui %218, %c2_i64 : i64
                      %220 = arith.select %215, %132, %133 : i64
                      %221 = arith.minsi %163, %137 : i64
                      %222 = arith.subi %137, %221 : i64
                      %223 = arith.minsi %222, %c1_i64 : i64
                      %224 = arith.cmpi sle, %163, %137 : i64
                      %225 = arith.select %224, %223, %c0_i64 : i64
                      %226 = arith.muli %163, %c512_i64 : i64
                      %227 = arith.addi %151, %226 : i64
                      %228 = arith.addi %227, %154 : i64
                      %229 = ascscalar.bind_local_ptr %220 : memref<1xf32, 1>
                      ascvec.get_buf mte2, %217, 0 {cannir.buffer.generated_sync}
                      "ascvec.copy_gm2ub_plan"(%229, %alloc, %c0_i32, %c0_i32, %c0_i64, %228, %225, %c256_i64, %c512_i64, %c256_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = -1 : i64, ub_capacity = 4 : i64}> : (memref<1xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                      ascvec.release_buf mte2, %217, 0 {cannir.buffer.generated_sync}
                      %230 = arith.cmpi slt, %157, %175 : i64
                      %231 = scf.if %230 -> (i64) {
                        %232 = arith.subi %175, %157 : i64
                        %233 = arith.cmpi sgt, %232, %c32_i64 : i64
                        %234 = arith.select %233, %c32_i64, %232 : i64
                        %235 = arith.cmpi eq, %arg35, %c0_i64 : i64
                        %236 = ascscalar.memref_address %alloc_15 : memref<4096xbf16, 1>
                        %237 = ascscalar.memref_address %alloc_16 : memref<4096xbf16, 1>
                        %238 = arith.select %235, %236, %237 : i64
                        %239 = arith.select %235, %c6_i64, %c7_i64 : i64
                        %240 = arith.addi %arg35, %c1_i64 : i64
                        %241 = arith.remui %240, %c2_i64 : i64
                        %242 = cannir.extract_physical_layout %arg4 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                        %243 = cannir.extract_stride %242 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                        %244 = cannir.extract_dim %243[2] : !cannir.stride<(-1, -1, -1, 1)>
                        %245 = arith.muli %244, %c2_i64 : i64
                        %246 = cannir.extract_dim %243[0] : !cannir.stride<(-1, -1, -1, 1)>
                        %247 = arith.muli %96, %246 : i64
                        %248 = arith.muli %247, %c2_i64 : i64
                        %249 = cannir.extract_dim %243[1] : !cannir.stride<(-1, -1, -1, 1)>
                        %250 = arith.muli %110, %249 : i64
                        %251 = arith.muli %250, %c2_i64 : i64
                        %252 = arith.addi %248, %251 : i64
                        %253 = arith.muli %112, %c128_i64 overflow<nsw> : i64
                        %254 = arith.addi %252, %253 : i64
                        %255 = arith.muli %163, %c64_i64 : i64
                        %256 = arith.addi %157, %255 : i64
                        %257 = arith.addi %256, %95 : i64
                        %258 = arith.muli %257, %244 : i64
                        %259 = arith.muli %258, %c2_i64 : i64
                        %260 = arith.addi %254, %259 : i64
                        %261 = ascscalar.bind_local_ptr %238 : memref<1xbf16, 1>
                        ascvec.get_buf mte2, %239, 0 {cannir.buffer.generated_sync}
                        "ascvec.copy_gm2ub_plan"(%261, %alloc_0, %c0_i32, %c0_i32, %c0_i64, %260, %234, %c128_i64, %245, %c128_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                        ascvec.release_buf mte2, %239, 0 {cannir.buffer.generated_sync}
                        scf.yield %241 : i64
                      } else {
                        scf.yield %arg35 : i64
                      }
                      scf.yield %219, %231 : i64, i64
                    } else {
                      scf.yield %arg30, %arg35 : i64, i64
                    }
                    %190 = ascscalar.memref_address %alloc_20 : memref<8192xbf16, 2>
                    %191 = ascscalar.memref_address %alloc_19 : memref<8192xbf16, 2>
                    %192 = arith.select %176, %190, %191 : i64
                    %193 = arith.select %176, %c10_i64, %c11_i64 : i64
                    %194 = arith.subi %168, %157 : i64
                    %195 = arith.cmpi slt, %194, %c0_i64 : i64
                    %196 = arith.select %195, %c0_i64, %194 : i64
                    %197 = arith.cmpi sgt, %196, %c32_i64 : i64
                    %198 = arith.select %197, %c32_i64, %196 : i64
                    %199 = arith.cmpi eq, %198, %c32_i64 : i64
                    ascvec.sync_intra_wait v, %c5_i64 {cannir.sync.banks = 3 : i32}
                    %200 = scf.if %199 -> (i64) {
                      %215 = arith.cmpi eq, %arg36, %c0_i64 : i64
                      %216 = ascscalar.memref_address %alloc_15 : memref<4096xbf16, 1>
                      %217 = ascscalar.memref_address %alloc_16 : memref<4096xbf16, 1>
                      %218 = arith.select %215, %216, %217 : i64
                      %219 = ascscalar.bind_local_ptr %218 : memref<4096xbf16, 1>
                      %220 = arith.select %215, %c6_i64, %c7_i64 : i64
                      %221 = arith.addi %arg36, %c1_i64 : i64
                      %222 = arith.remui %221, %c2_i64 : i64
                      ascvec.get_buf v, %220, 0 {cannir.buffer.generated_sync}
                      ascvec.vf(%219, %alloc_7, %alloc_9 : memref<4096xbf16, 1>, memref<4096xf32, 1>, memref<4096xbf16, 1>) {
                      ^bb0(%arg37: memref<4096xbf16, 1>, %arg38: memref<4096xf32, 1>, %arg39: memref<4096xbf16, 1>):
                        %c128_i32_21 = arith.constant 128 : i32
                        %c2048_i32 = arith.constant 2048 : i32
                        %c0_i32_22 = arith.constant 0 : i32
                        %c64_i32_23 = arith.constant 64 : i32
                        %mask, %remaining_out = ascvec.update_mask %c64_i32_23 : !ascvec.mask
                        scf.for %arg40 = %c0_i32_22 to %c2048_i32 step %c128_i32_21  : i32 {
                          %223 = arith.index_cast %arg40 : i32 to index
                          %224 = ascvec.loadalign_unpack %arg37[%223] {mode = "b16_to_b32"} : memref<4096xbf16, 1>, index -> <bf16>
                          %225 = arith.addi %arg40, %c64_i32_23 overflow<nsw> : i32
                          %226 = arith.index_cast %225 : i32 to index
                          %227 = ascvec.loadalign_unpack %arg37[%226] {mode = "b16_to_b32"} : memref<4096xbf16, 1>, index -> <bf16>
                          %228 = ascvec.cast %224, %mask {rounding = 6 : i32} : <bf16> -> <f32>
                          %229 = ascvec.cast %227, %mask {rounding = 6 : i32} : <bf16> -> <f32>
                          %230 = ascvec.loadalign %arg38[%223] : memref<4096xf32, 1>, index -> <f32>
                          %231 = ascvec.loadalign %arg38[%226] : memref<4096xf32, 1>, index -> <f32>
                          %232 = ascvec.add %228, %230, %mask : <f32>
                          %233 = ascvec.add %229, %231, %mask : <f32>
                          %234 = ascvec.cast %232, %mask : <f32> -> <bf16>
                          %235 = ascvec.cast %233, %mask : <f32> -> <bf16>
                          ascvec.storealign_pack %arg39[%223], %234, %mask {mode = "b32_to_b16"} : memref<4096xbf16, 1>, index, <bf16>
                          ascvec.storealign_pack %arg39[%226], %235, %mask {mode = "b32_to_b16"} : memref<4096xbf16, 1>, index, <bf16>
                        }
                        ascvec.mem_bar vst_vld
                      } {cannir.vf_mode = "simd"}
                      ascvec.release_buf v, %220, 0 {cannir.buffer.generated_sync}
                      scf.yield %222 : i64
                    } else {
                      ascvec.vf(%alloc_9 : memref<4096xbf16, 1>) {
                      ^bb0(%arg37: memref<4096xbf16, 1>):
                        %c1_i32_21 = arith.constant 1 : i32
                        %c32_i32_22 = arith.constant 32 : i32
                        %c0_i32_23 = arith.constant 0 : i32
                        %cst = arith.constant 0.000000e+00 : bf16
                        %c64_i32_24 = arith.constant 64 : i32
                        %mask, %remaining_out = ascvec.update_mask %c64_i32_24 elem_bits 16 : !ascvec.mask
                        %215 = ascvec.duplicate_scalar %cst, %mask : bf16 -> <bf16>
                        scf.for %arg38 = %c0_i32_23 to %c32_i32_22 step %c1_i32_21  : i32 {
                          %216 = arith.muli %arg38, %c64_i32_24 overflow<nsw> : i32
                          %217 = arith.index_cast %216 : i32 to index
                          ascvec.storealign %arg37[%217], %215, %mask : memref<4096xbf16, 1>, index, <bf16>
                        }
                        ascvec.mem_bar vst_vld
                      } {cannir.vf_mode = "simd"}
                      scf.yield %arg36 : i64
                    }
                    %201 = arith.cmpi sgt, %198, %c0_i64 : i64
                    %202 = arith.cmpi slt, %198, %c32_i64 : i64
                    %203 = arith.select %201, %202, %false : i1
                    %204 = scf.if %203 -> (i64) {
                      %215 = arith.cmpi eq, %200, %c0_i64 : i64
                      %216 = ascscalar.memref_address %alloc_15 : memref<4096xbf16, 1>
                      %217 = ascscalar.memref_address %alloc_16 : memref<4096xbf16, 1>
                      %218 = arith.select %215, %216, %217 : i64
                      %219 = ascscalar.bind_local_ptr %218 : memref<4096xbf16, 1>
                      %220 = arith.select %215, %c6_i64, %c7_i64 : i64
                      %221 = arith.addi %200, %c1_i64 : i64
                      %222 = arith.remui %221, %c2_i64 : i64
                      ascvec.get_buf v, %220, 0 {cannir.buffer.generated_sync}
                      ascvec.vf(%219, %alloc_7, %alloc_9, %198 : memref<4096xbf16, 1>, memref<4096xf32, 1>, memref<4096xbf16, 1>, i64) {
                      ^bb0(%arg37: memref<4096xbf16, 1>, %arg38: memref<4096xf32, 1>, %arg39: memref<4096xbf16, 1>, %arg40: i64):
                        %c1_i64_21 = arith.constant 1 : i64
                        %c0_i64_22 = arith.constant 0 : i64
                        %c64_i64_23 = arith.constant 64 : i64
                        %c64_i32_24 = arith.constant 64 : i32
                        scf.for %arg41 = %c0_i64_22 to %arg40 step %c1_i64_21  : i64 {
                          %mask, %remaining_out = ascvec.update_mask %c64_i32_24 : !ascvec.mask
                          %223 = arith.muli %arg41, %c64_i64_23 : i64
                          %224 = arith.index_cast %223 : i64 to index
                          %225 = ascvec.loadalign_unpack %arg37[%224] {mode = "b16_to_b32"} : memref<4096xbf16, 1>, index -> <bf16>
                          %226 = ascvec.cast %225, %mask {rounding = 6 : i32} : <bf16> -> <f32>
                          %227 = ascvec.loadalign %arg38[%224] : memref<4096xf32, 1>, index -> <f32>
                          %228 = ascvec.add %226, %227, %mask : <f32>
                          %229 = ascvec.cast %228, %mask : <f32> -> <bf16>
                          ascvec.storealign_pack %arg39[%224], %229, %mask {mode = "b32_to_b16"} : memref<4096xbf16, 1>, index, <bf16>
                        } {cannir.loop_unroll = 2 : i64}
                        ascvec.mem_bar vst_vld
                      } {cannir.vf_mode = "simd"}
                      ascvec.release_buf v, %220, 0 {cannir.buffer.generated_sync}
                      scf.yield %222 : i64
                    } else {
                      scf.yield %200 : i64
                    }
                    ascvec.sync_intra_arrive v, %c5_i64 {cannir.sync.banks = 3 : i32}
                    ascvec.get_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
                    ascvec.vf(%alloc_9, %alloc_12 : memref<4096xbf16, 1>, memref<4224xbf16, 1>) {
                    ^bb0(%arg37: memref<4096xbf16, 1>, %arg38: memref<4224xbf16, 1>):
                      %c1_i32_21 = arith.constant 1 : i32
                      %c32_i32_22 = arith.constant 32 : i32
                      %c0_i32_23 = arith.constant 0 : i32
                      %c33 = arith.constant 33 : index
                      %c16_i32_24 = arith.constant 16 : i32
                      %c0_25 = arith.constant 0 : index
                      %c64_i32_26 = arith.constant 64 : i32
                      scf.for %arg39 = %c0_i32_23 to %c32_i32_22 step %c1_i32_21  : i32 {
                        %mask, %remaining_out = ascvec.update_mask %c64_i32_26 elem_bits 16 : !ascvec.mask
                        %215 = arith.muli %arg39, %c64_i32_26 overflow<nsw> : i32
                        %216 = arith.index_cast %215 : i32 to index
                        %217 = ascvec.loadalign %arg37[%216] : memref<4096xbf16, 1>, index -> <bf16>
                        %218 = arith.muli %arg39, %c16_i32_24 overflow<nsw> : i32
                        %219 = arith.index_cast %218 : i32 to index
                        ascvec.storealign_strided %arg38[%219], %217, %mask, %c33, %c0_25 : memref<4224xbf16, 1>, <bf16>
                      }
                    } {cannir.vf_mode = "simd"}
                    ascvec.release_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
                    %205 = ascscalar.bitcast_memref %alloc_12 byte_offset = %c0_i64 : memref<4224xbf16, 1> -> memref<?xbf16, 1>
                    %206 = ascscalar.bind_local_ptr %192 : memref<1xbf16, 2>
                    %207 = arith.muli %97, %c512_i64 : i64
                    %208 = arith.cmpi eq, %193, %c11_i64 : i64
                    %209 = arith.select %208, %c3_i64, %c2_i64 : i64
                    ascvec.sync_intra_wait mte3, %209 {cannir.sync.banks = 3 : i32}
                    ascvec.get_buf mte3, %c3_i64, 0 {cannir.buffer.generated_sync}
                    %210 = arith.index_cast %207 : i64 to index
                    ascvec.copy_ub2l1_strided %206[%210], %205[%c0], %c4_i32, %c32_i32, %c1_i32, %c32_i32 : memref<1xbf16, 2>, memref<?xbf16, 1>
                    ascvec.release_buf mte3, %c3_i64, 0 {cannir.buffer.generated_sync}
                    ascvec.sync_intra_arrive mte3, %209 {cannir.sync.banks = 3 : i32}
                    %211 = scf.if %188 -> (i64) {
                      %215 = arith.addi %arg32, %c1_i64 overflow<nsw> : i64
                      %216 = arith.remui %215, %c2_i64 : i64
                      scf.yield %216 : i64
                    } else {
                      scf.yield %arg32 : i64
                    }
                    %212 = arith.select %176, %132, %133 : i64
                    %213 = ascscalar.bind_local_ptr %212 : memref<64xf32, 1>
                    ascvec.sync_intra_wait v, %c4_i64 {cannir.sync.banks = 3 : i32}
                    ascvec.get_buf v, %186, 0 {cannir.buffer.generated_sync}
                    ascvec.vf(%213, %alloc_8, %alloc_6 : memref<64xf32, 1>, memref<8192xf32, 1>, memref<8192xf32, 1>) {
                    ^bb0(%arg37: memref<64xf32, 1>, %arg38: memref<8192xf32, 1>, %arg39: memref<8192xf32, 1>):
                      %c4096_i32_21 = arith.constant 4096 : i32
                      %c0_i32_22 = arith.constant 0 : i32
                      %c0_23 = arith.constant 0 : index
                      %c64_i32_24 = arith.constant 64 : i32
                      %mask, %remaining_out = ascvec.update_mask %c64_i32_24 : !ascvec.mask
                      %215 = ascvec.loadalign %arg37[%c0_23] : memref<64xf32, 1>, index -> <f32>
                      scf.for %arg40 = %c0_i32_22 to %c4096_i32_21 step %c64_i32_24  : i32 {
                        %216 = arith.index_cast %arg40 : i32 to index
                        %217 = ascvec.loadalign %arg38[%216] : memref<8192xf32, 1>, index -> <f32>
                        %218 = ascvec.loadalign %arg39[%216] : memref<8192xf32, 1>, index -> <f32>
                        %219 = ascvec.mul %217, %215, %mask : <f32>
                        %220 = ascvec.add %219, %218, %mask : <f32>
                        ascvec.storealign %arg38[%216], %220, %mask : memref<8192xf32, 1>, index, <f32>
                      } {cannir.loop_unroll = 4 : i64}
                    } {cannir.vf_mode = "simd"}
                    ascvec.release_buf v, %186, 0 {cannir.buffer.generated_sync}
                    ascvec.sync_intra_arrive v, %c4_i64 {cannir.sync.banks = 3 : i32}
                    %214 = scf.if %188 -> (i64) {
                      ascvec.vf(%alloc_8, %alloc_10 : memref<8192xf32, 1>, memref<8192xbf16, 1>) {
                      ^bb0(%arg37: memref<8192xf32, 1>, %arg38: memref<8192xbf16, 1>):
                        %c1_i32_21 = arith.constant 1 : i32
                        %c0_i32_22 = arith.constant 0 : i32
                        %c64_i32_23 = arith.constant 64 : i32
                        scf.for %arg39 = %c0_i32_22 to %c64_i32_23 step %c1_i32_21  : i32 {
                          %mask, %remaining_out = ascvec.update_mask %c64_i32_23 : !ascvec.mask
                          %226 = arith.muli %arg39, %c64_i32_23 overflow<nsw> : i32
                          %227 = arith.index_cast %226 : i32 to index
                          %228 = ascvec.loadalign %arg37[%227] : memref<8192xf32, 1>, index -> <f32>
                          %229 = ascvec.cast %228, %mask : <f32> -> <bf16>
                          ascvec.storealign_pack %arg38[%227], %229, %mask {mode = "b32_to_b16"} : memref<8192xbf16, 1>, index, <bf16>
                        }
                      } {cannir.vf_mode = "simd"}
                      ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                      ascvec.vf(%alloc_10, %alloc_11 : memref<8192xbf16, 1>, memref<8256xbf16, 1>) {
                      ^bb0(%arg37: memref<8192xbf16, 1>, %arg38: memref<8256xbf16, 1>):
                        %c1_i32_21 = arith.constant 1 : i32
                        %c0_i32_22 = arith.constant 0 : i32
                        %c65 = arith.constant 65 : index
                        %c16_i32_23 = arith.constant 16 : i32
                        %c0_24 = arith.constant 0 : index
                        %c64_i32_25 = arith.constant 64 : i32
                        scf.for %arg39 = %c0_i32_22 to %c64_i32_25 step %c1_i32_21  : i32 {
                          %mask, %remaining_out = ascvec.update_mask %c64_i32_25 elem_bits 16 : !ascvec.mask
                          %226 = arith.muli %arg39, %c64_i32_25 overflow<nsw> : i32
                          %227 = arith.index_cast %226 : i32 to index
                          %228 = ascvec.loadalign %arg37[%227] : memref<8192xbf16, 1>, index -> <bf16>
                          %229 = arith.muli %arg39, %c16_i32_23 overflow<nsw> : i32
                          %230 = arith.index_cast %229 : i32 to index
                          ascvec.storealign_strided %arg38[%230], %228, %mask, %c65, %c0_24 : memref<8256xbf16, 1>, <bf16>
                        }
                      } {cannir.vf_mode = "simd"}
                      ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                      %215 = arith.cmpi eq, %arg33, %c0_i64 : i64
                      %216 = arith.select %215, %177, %178 : i64
                      %217 = arith.select %215, %c8_i64, %c9_i64 : i64
                      %218 = arith.addi %arg33, %c1_i64 overflow<nsw> : i64
                      %219 = arith.remui %218, %c2_i64 : i64
                      %220 = ascscalar.bitcast_memref %alloc_11 byte_offset = %c0_i64 : memref<8256xbf16, 1> -> memref<?xbf16, 1>
                      %221 = ascscalar.bind_local_ptr %216 : memref<1xbf16, 2>
                      %222 = arith.muli %97, %c4096_i64 : i64
                      %223 = arith.cmpi eq, %217, %c9_i64 : i64
                      %224 = arith.extui %223 : i1 to i64
                      ascvec.sync_intra_wait mte3, %224 {cannir.sync.banks = 3 : i32}
                      ascvec.get_buf mte3, %c2_i64, 0 {cannir.buffer.generated_sync}
                      %225 = arith.index_cast %222 : i64 to index
                      ascvec.copy_ub2l1_strided %221[%225], %220[%c0], %c4_i32, %c64_i32, %c1_i32, %c0_i32 : memref<1xbf16, 2>, memref<?xbf16, 1>
                      ascvec.release_buf mte3, %c2_i64, 0 {cannir.buffer.generated_sync}
                      ascvec.sync_intra_arrive mte3, %224 {cannir.sync.banks = 3 : i32}
                      scf.yield %219 : i64
                    } else {
                      ascvec.get_buf v, %c5_i64, 0 {cannir.buffer.generated_sync}
                      ascvec.copy_ub2ub %alloc_14[%c0], %alloc_8[%c0], %c16384_i32 : memref<8192xf32, 1>, memref<8192xf32, 1>
                      ascvec.vf(%alloc_14, %alloc_8 : memref<8192xf32, 1>, memref<8192xf32, 1>) {
                      ^bb0(%arg37: memref<8192xf32, 1>, %arg38: memref<8192xf32, 1>):
                      } {cannir.vf_mode = "simd"}
                      ascvec.release_buf v, %c5_i64, 0 {cannir.buffer.generated_sync}
                      %215 = cannir.extract_physical_layout %arg8 : !cannir.tensor<<gm, f32>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, identity>
                      %216 = cannir.extract_stride %215 : <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>> -> <(-1, 16384, 128, 1)>
                      %217 = cannir.extract_dim %216[0] : !cannir.stride<(-1, 16384, 128, 1)>
                      %218 = arith.muli %113, %217 : i64
                      %219 = arith.muli %110, %c16384_i64 : i64
                      %220 = arith.addi %218, %219 : i64
                      %221 = arith.muli %112, %c8192_i64 overflow<nsw> : i64
                      %222 = arith.addi %220, %221 : i64
                      %223 = arith.muli %97, %c64_i64 : i64
                      %224 = arith.addi %222, %223 : i64
                      %225 = arith.muli %224, %c32_i64 : i64
                      %226 = ascscalar.memref_address %alloc_2 : memref<1xf32>
                      %227 = arith.divui %225, %c8_i64 : i64
                      %228 = arith.addi %226, %227 : i64
                      %229 = ascscalar.bind_gm_ptr %228 : memref<1xf32>
                      %230 = ascscalar.bitcast_memref %alloc_14 byte_offset = %c0_i64 : memref<8192xf32, 1> -> memref<?xf32, 1>
                      ascvec.get_buf mte3, %c5_i64, 0 {cannir.buffer.generated_sync}
                      ascvec.copy_ub2gm %229[%c0], %230[%c0], %c64_i32, %c256_i32, %c512_i64, %c256_i32 l2_cache_ctl %c0_i32 : memref<1xf32>, memref<?xf32, 1>
                      ascvec.release_buf mte3, %c5_i64, 0 {cannir.buffer.generated_sync}
                      scf.yield %arg33 : i64
                    }
                    scf.yield %189#0, %180, %211, %214, %184, %189#1, %204 : i64, i64, i64, i64, i64, i64, i64
                  }
                  scf.yield %160#0, %160#1, %160#2, %160#3, %160#4, %160#5, %160#6 : i64, i64, i64, i64, i64, i64, i64
                }
                scf.yield %100#0, %100#1, %100#2, %100#3, %100#4, %100#5, %100#6 : i64, i64, i64, i64, i64, i64, i64
              } else {
                scf.yield %arg14, %arg15, %arg16, %arg17, %arg18, %arg19, %arg20 : i64, i64, i64, i64, i64, i64, i64
              }
              scf.yield %99#0, %99#1, %99#2, %99#3, %99#4, %99#5, %99#6 : i64, i64, i64, i64, i64, i64, i64
            }
          } else {
            %50 = arith.cmpi eq, %44, %c4_i32 : i32
            scf.if %50 {
              %51:7 = scf.for %arg13 = %c0_i32 to %21 step %c1_i32 iter_args(%arg14 = %c0_i64, %arg15 = %c0_i64, %arg16 = %c0_i64, %arg17 = %c0_i64, %arg18 = %c0_i64, %arg19 = %c0_i64, %arg20 = %c0_i64) -> (i64, i64, i64, i64, i64, i64, i64)  : i32 {
                %52 = arith.addi %32, %arg13 : i32
                %53 = arith.index_cast %52 : i32 to index
                %54 = memref.load %alloc_3[%53] : memref<1xi32>
                %55 = arith.addi %52, %c1_i32 : i32
                %56 = arith.index_cast %55 : i32 to index
                %57 = memref.load %alloc_3[%56] : memref<1xi32>
                %58 = arith.cmpi slt, %39, %54 : i32
                %59 = arith.select %58, %54, %39 : i32
                %60 = arith.cmpi sgt, %42, %57 : i32
                %61 = arith.select %60, %57, %42 : i32
                %62 = arith.cmpi slt, %59, %61 : i32
                %63 = arith.subi %59, %54 : i32
                %64 = arith.subi %61, %54 : i32
                %65 = arith.extsi %63 : i32 to i64
                %66 = arith.extsi %64 : i32 to i64
                %67 = arith.select %62, %65, %c0_i64 : i64
                %68 = arith.select %62, %66, %c0_i64 : i64
                %69 = arith.addi %22, %arg13 : i32
                %70 = arith.index_cast %69 : i32 to index
                %71 = memref.load %alloc_3[%70] : memref<1xi32>
                %72 = arith.addi %23, %arg13 : i32
                %73 = arith.index_cast %72 : i32 to index
                %74 = memref.load %alloc_3[%73] : memref<1xi32>
                %75 = arith.addi %24, %arg13 : i32
                %76 = arith.index_cast %75 : i32 to index
                %77 = memref.load %alloc_3[%76] : memref<1xi32>
                %78 = arith.addi %25, %arg13 : i32
                %79 = arith.index_cast %78 : i32 to index
                %80 = memref.load %alloc_3[%79] : memref<1xi32>
                %81 = arith.extsi %80 : i32 to i64
                %82 = arith.addi %27, %arg13 : i32
                %83 = arith.index_cast %82 : i32 to index
                %84 = memref.load %alloc_3[%83] : memref<1xi32>
                %85 = arith.addi %28, %arg13 : i32
                %86 = arith.index_cast %85 : i32 to index
                %87 = memref.load %alloc_3[%86] : memref<1xi32>
                %88 = arith.addi %81, %c63_i64 : i64
                %89 = arith.remsi %88, %c64_i64 : i64
                %90 = arith.cmpi ne, %89, %c0_i64 : i64
                %91 = arith.cmpi slt, %89, %c0_i64 : i64
                %92 = arith.andi %90, %91 : i1
                %93 = arith.divsi %88, %c64_i64 : i64
                %94 = arith.extui %92 : i1 to i64
                %95 = arith.subi %93, %94 : i64
                %96 = arith.extsi %77 : i32 to i64
                %97 = arith.extsi %74 : i32 to i64
                %98 = ascscalar.get_subblock_id
                %99 = arith.cmpi slt, %0, %1 : i64
                %100:7 = scf.if %99 -> (i64, i64, i64, i64, i64, i64, i64) {
                  %101:7 = scf.for %arg21 = %67 to %68 step %c1_i64 iter_args(%arg22 = %arg14, %arg23 = %arg15, %arg24 = %arg16, %arg25 = %arg17, %arg26 = %arg18, %arg27 = %arg19, %arg28 = %arg20) -> (i64, i64, i64, i64, i64, i64, i64)  : i64 {
                    %102 = arith.remsi %arg21, %c4_i64 : i64
                    %103 = arith.cmpi slt, %102, %c0_i64 : i64
                    %104 = arith.divsi %arg21, %c4_i64 : i64
                    %105 = arith.extui %103 : i1 to i64
                    %106 = arith.subi %104, %105 : i64
                    %107 = arith.maxsi %7, %c1_i64 : i64
                    %108 = arith.remsi %106, %107 : i64
                    %109 = arith.cmpi slt, %108, %c0_i64 : i64
                    %110 = arith.select %109, %107, %c0_i64 : i64
                    %111 = arith.addi %108, %110 : i64
                    %112 = arith.select %103, %c4_i64, %c0_i64 : i64
                    %113 = arith.addi %102, %112 overflow<nsw> : i64
                    %114 = arith.extsi %71 : i32 to i64
                    %115 = arith.addi %84, %87 : i32
                    %116 = arith.extsi %115 : i32 to i64
                    %117 = arith.cmpi eq, %116, %95 : i64
                    %118 = arith.muli %84, %c64_i32 : i32
                    %119 = arith.extsi %118 : i32 to i64
                    %120 = arith.subi %81, %119 : i64
                    %121 = arith.minsi %120, %c64_i64 : i64
                    %122 = arith.select %117, %121, %c64_i64 : i64
                    %123 = arith.extsi %84 : i32 to i64
                    %124 = arith.cmpi eq, %arg22, %c0_i64 : i64
                    %125 = arith.addi %arg22, %c1_i64 : i64
                    %126 = arith.remui %125, %c2_i64 : i64
                    %127 = arith.cmpi ne, %arg22, %c0_i64 : i64
                    %128 = arith.extui %127 : i1 to i64
                    %129 = arith.addi %arg24, %c1_i64 : i64
                    %130 = arith.remui %129, %c2_i64 : i64
                    %131 = arith.cmpi eq, %84, %c0_i32 : i32
                    %132 = scf.if %131 -> (i64) {
                      %162 = cannir.extract_physical_layout %arg7 : !cannir.tensor<<gm, f32>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, identity>
                      %163 = cannir.extract_stride %162 : <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>> -> <(-1, 16384, 128, 1)>
                      %164 = cannir.extract_dim %163[0] : !cannir.stride<(-1, 16384, 128, 1)>
                      %165 = arith.muli %114, %164 : i64
                      %166 = arith.muli %165, %c4_i64 : i64
                      %167 = arith.muli %111, %c65536_i64 : i64
                      %168 = arith.addi %166, %167 : i64
                      %169 = arith.muli %113, %c16384_i64 overflow<nsw> : i64
                      %170 = arith.addi %168, %169 : i64
                      %171 = arith.muli %98, %c256_i64 : i64
                      %172 = arith.addi %170, %171 : i64
                      ascvec.get_buf mte2, %c4_i64, 0 {cannir.buffer.generated_sync}
                      "ascvec.copy_gm2ub_plan"(%45, %alloc_1, %c0_i32, %c0_i32, %c0_i64, %172, %c32_i64, %c256_i64, %c512_i64, %c256_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 131968 : i64, ub_capacity = 32768 : i64}> : (memref<?xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                      ascvec.release_buf mte2, %c4_i64, 0 {cannir.buffer.generated_sync}
                      ascvec.get_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
                      ascvec.copy_ub2ub %alloc_8[%c0], %alloc_13[%c0], %c8192_i32 : memref<8192xf32, 1>, memref<8192xf32, 1>
                      ascvec.vf(%alloc_8, %alloc_13 : memref<8192xf32, 1>, memref<8192xf32, 1>) {
                      ^bb0(%arg29: memref<8192xf32, 1>, %arg30: memref<8192xf32, 1>):
                      } {cannir.vf_mode = "simd"}
                      ascvec.release_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
                      ascvec.vf(%alloc_8, %alloc_10 : memref<8192xf32, 1>, memref<8192xbf16, 1>) {
                      ^bb0(%arg29: memref<8192xf32, 1>, %arg30: memref<8192xbf16, 1>):
                        %c1_i32_21 = arith.constant 1 : i32
                        %c32_i32_22 = arith.constant 32 : i32
                        %c0_i32_23 = arith.constant 0 : i32
                        %c64_i32_24 = arith.constant 64 : i32
                        scf.for %arg31 = %c0_i32_23 to %c32_i32_22 step %c1_i32_21  : i32 {
                          %mask, %remaining_out = ascvec.update_mask %c64_i32_24 : !ascvec.mask
                          %186 = arith.muli %arg31, %c64_i32_24 overflow<nsw> : i32
                          %187 = arith.index_cast %186 : i32 to index
                          %188 = ascvec.loadalign %arg29[%187] : memref<8192xf32, 1>, index -> <f32>
                          %189 = ascvec.cast %188, %mask : <f32> -> <bf16>
                          ascvec.storealign_pack %arg30[%187], %189, %mask {mode = "b32_to_b16"} : memref<8192xbf16, 1>, index, <bf16>
                        }
                      } {cannir.vf_mode = "simd"}
                      ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                      ascvec.vf(%alloc_10, %alloc_11 : memref<8192xbf16, 1>, memref<8256xbf16, 1>) {
                      ^bb0(%arg29: memref<8192xbf16, 1>, %arg30: memref<8256xbf16, 1>):
                        %c1_i32_21 = arith.constant 1 : i32
                        %c32_i32_22 = arith.constant 32 : i32
                        %c0_i32_23 = arith.constant 0 : i32
                        %c33 = arith.constant 33 : index
                        %c16_i32_24 = arith.constant 16 : i32
                        %c0_25 = arith.constant 0 : index
                        %c64_i32_26 = arith.constant 64 : i32
                        scf.for %arg31 = %c0_i32_23 to %c32_i32_22 step %c1_i32_21  : i32 {
                          %mask, %remaining_out = ascvec.update_mask %c64_i32_26 elem_bits 16 : !ascvec.mask
                          %186 = arith.muli %arg31, %c64_i32_26 overflow<nsw> : i32
                          %187 = arith.index_cast %186 : i32 to index
                          %188 = ascvec.loadalign %arg29[%187] : memref<8192xbf16, 1>, index -> <bf16>
                          %189 = arith.muli %arg31, %c16_i32_24 overflow<nsw> : i32
                          %190 = arith.index_cast %189 : i32 to index
                          ascvec.storealign_strided %arg30[%190], %188, %mask, %c33, %c0_25 : memref<8256xbf16, 1>, <bf16>
                        }
                      } {cannir.vf_mode = "simd"}
                      ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                      %173 = arith.cmpi eq, %arg25, %c0_i64 : i64
                      %174 = ascscalar.memref_address %alloc_18 : memref<16384xbf16, 2>
                      %175 = ascscalar.memref_address %alloc_17 : memref<16384xbf16, 2>
                      %176 = arith.select %173, %174, %175 : i64
                      %177 = arith.select %173, %c8_i64, %c9_i64 : i64
                      %178 = arith.addi %arg25, %c1_i64 : i64
                      %179 = arith.remui %178, %c2_i64 : i64
                      %180 = ascscalar.bitcast_memref %alloc_11 byte_offset = %c0_i64 : memref<8256xbf16, 1> -> memref<?xbf16, 1>
                      %181 = ascscalar.bind_local_ptr %176 : memref<1xbf16, 2>
                      %182 = arith.muli %98, %c2048_i64 : i64
                      %183 = arith.cmpi eq, %177, %c9_i64 : i64
                      %184 = arith.extui %183 : i1 to i64
                      ascvec.sync_intra_wait mte3, %184 {cannir.sync.banks = 3 : i32}
                      ascvec.get_buf mte3, %c2_i64, 0 {cannir.buffer.generated_sync}
                      %185 = arith.index_cast %182 : i64 to index
                      ascvec.copy_ub2l1_strided %181[%185], %180[%c0], %c4_i32, %c32_i32, %c1_i32, %c0_i32 : memref<1xbf16, 2>, memref<?xbf16, 1>
                      ascvec.release_buf mte3, %c2_i64, 0 {cannir.buffer.generated_sync}
                      ascvec.sync_intra_arrive mte3, %184 {cannir.sync.banks = 3 : i32}
                      scf.yield %179 : i64
                    } else {
                      %162 = cannir.extract_physical_layout %arg8 : !cannir.tensor<<gm, f32>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, identity>
                      %163 = cannir.extract_stride %162 : <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>> -> <(-1, 16384, 128, 1)>
                      %164 = cannir.extract_dim %163[0] : !cannir.stride<(-1, 16384, 128, 1)>
                      %165 = arith.muli %114, %164 : i64
                      %166 = arith.muli %165, %c4_i64 : i64
                      %167 = arith.muli %111, %c65536_i64 : i64
                      %168 = arith.addi %166, %167 : i64
                      %169 = arith.muli %113, %c16384_i64 overflow<nsw> : i64
                      %170 = arith.addi %168, %169 : i64
                      %171 = arith.muli %98, %c256_i64 : i64
                      %172 = arith.addi %170, %171 : i64
                      ascvec.get_buf mte2, %c4_i64, 0 {cannir.buffer.generated_sync}
                      "ascvec.copy_gm2ub_plan"(%45, %alloc_2, %c0_i32, %c0_i32, %c0_i64, %172, %c32_i64, %c256_i64, %c512_i64, %c256_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 131968 : i64, ub_capacity = 32768 : i64}> : (memref<?xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                      ascvec.release_buf mte2, %c4_i64, 0 {cannir.buffer.generated_sync}
                      ascvec.get_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
                      ascvec.copy_ub2ub %alloc_8[%c0], %alloc_13[%c0], %c8192_i32 : memref<8192xf32, 1>, memref<8192xf32, 1>
                      ascvec.vf(%alloc_8, %alloc_13 : memref<8192xf32, 1>, memref<8192xf32, 1>) {
                      ^bb0(%arg29: memref<8192xf32, 1>, %arg30: memref<8192xf32, 1>):
                      } {cannir.vf_mode = "simd"}
                      ascvec.release_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
                      ascvec.vf(%alloc_8, %alloc_10 : memref<8192xf32, 1>, memref<8192xbf16, 1>) {
                      ^bb0(%arg29: memref<8192xf32, 1>, %arg30: memref<8192xbf16, 1>):
                        %c1_i32_21 = arith.constant 1 : i32
                        %c32_i32_22 = arith.constant 32 : i32
                        %c0_i32_23 = arith.constant 0 : i32
                        %c64_i32_24 = arith.constant 64 : i32
                        scf.for %arg31 = %c0_i32_23 to %c32_i32_22 step %c1_i32_21  : i32 {
                          %mask, %remaining_out = ascvec.update_mask %c64_i32_24 : !ascvec.mask
                          %186 = arith.muli %arg31, %c64_i32_24 overflow<nsw> : i32
                          %187 = arith.index_cast %186 : i32 to index
                          %188 = ascvec.loadalign %arg29[%187] : memref<8192xf32, 1>, index -> <f32>
                          %189 = ascvec.cast %188, %mask : <f32> -> <bf16>
                          ascvec.storealign_pack %arg30[%187], %189, %mask {mode = "b32_to_b16"} : memref<8192xbf16, 1>, index, <bf16>
                        }
                      } {cannir.vf_mode = "simd"}
                      ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                      ascvec.vf(%alloc_10, %alloc_11 : memref<8192xbf16, 1>, memref<8256xbf16, 1>) {
                      ^bb0(%arg29: memref<8192xbf16, 1>, %arg30: memref<8256xbf16, 1>):
                        %c1_i32_21 = arith.constant 1 : i32
                        %c32_i32_22 = arith.constant 32 : i32
                        %c0_i32_23 = arith.constant 0 : i32
                        %c33 = arith.constant 33 : index
                        %c16_i32_24 = arith.constant 16 : i32
                        %c0_25 = arith.constant 0 : index
                        %c64_i32_26 = arith.constant 64 : i32
                        scf.for %arg31 = %c0_i32_23 to %c32_i32_22 step %c1_i32_21  : i32 {
                          %mask, %remaining_out = ascvec.update_mask %c64_i32_26 elem_bits 16 : !ascvec.mask
                          %186 = arith.muli %arg31, %c64_i32_26 overflow<nsw> : i32
                          %187 = arith.index_cast %186 : i32 to index
                          %188 = ascvec.loadalign %arg29[%187] : memref<8192xbf16, 1>, index -> <bf16>
                          %189 = arith.muli %arg31, %c16_i32_24 overflow<nsw> : i32
                          %190 = arith.index_cast %189 : i32 to index
                          ascvec.storealign_strided %arg30[%190], %188, %mask, %c33, %c0_25 : memref<8256xbf16, 1>, <bf16>
                        }
                      } {cannir.vf_mode = "simd"}
                      ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                      %173 = arith.cmpi eq, %arg25, %c0_i64 : i64
                      %174 = ascscalar.memref_address %alloc_18 : memref<16384xbf16, 2>
                      %175 = ascscalar.memref_address %alloc_17 : memref<16384xbf16, 2>
                      %176 = arith.select %173, %174, %175 : i64
                      %177 = arith.select %173, %c8_i64, %c9_i64 : i64
                      %178 = arith.addi %arg25, %c1_i64 : i64
                      %179 = arith.remui %178, %c2_i64 : i64
                      %180 = ascscalar.bitcast_memref %alloc_11 byte_offset = %c0_i64 : memref<8256xbf16, 1> -> memref<?xbf16, 1>
                      %181 = ascscalar.bind_local_ptr %176 : memref<1xbf16, 2>
                      %182 = arith.muli %98, %c2048_i64 : i64
                      %183 = arith.cmpi eq, %177, %c9_i64 : i64
                      %184 = arith.extui %183 : i1 to i64
                      ascvec.sync_intra_wait mte3, %184 {cannir.sync.banks = 3 : i32}
                      ascvec.get_buf mte3, %c2_i64, 0 {cannir.buffer.generated_sync}
                      %185 = arith.index_cast %182 : i64 to index
                      ascvec.copy_ub2l1_strided %181[%185], %180[%c0], %c4_i32, %c32_i32, %c1_i32, %c0_i32 : memref<1xbf16, 2>, memref<?xbf16, 1>
                      ascvec.release_buf mte3, %c2_i64, 0 {cannir.buffer.generated_sync}
                      ascvec.sync_intra_arrive mte3, %184 {cannir.sync.banks = 3 : i32}
                      scf.yield %179 : i64
                    }
                    %133 = ascscalar.memref_address %alloc_4 : memref<64xf32, 1>
                    %134 = ascscalar.memref_address %alloc_5 : memref<64xf32, 1>
                    %135 = arith.select %124, %133, %134 : i64
                    %136 = cannir.extract_layout %arg1 : !cannir.tensor<<gm, f32>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                    %137 = cannir.extract_shape %136 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, -1, 128)>
                    %138 = cannir.extract_dim %137[2] : !cannir.shape<(-1, -1, -1, 128)>
                    %139 = arith.minsi %123, %138 : i64
                    %140 = arith.subi %138, %139 : i64
                    %141 = arith.minsi %140, %c1_i64 : i64
                    %142 = arith.cmpi sle, %123, %138 : i64
                    %143 = arith.select %142, %141, %c0_i64 : i64
                    %144 = cannir.extract_physical_layout %arg1 : !cannir.tensor<<gm, f32>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                    %145 = cannir.extract_stride %144 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, 128, 1)>
                    %146 = cannir.extract_dim %145[0] : !cannir.stride<(-1, -1, 128, 1)>
                    %147 = arith.muli %114, %146 : i64
                    %148 = arith.muli %147, %c4_i64 : i64
                    %149 = cannir.extract_dim %145[1] : !cannir.stride<(-1, -1, 128, 1)>
                    %150 = arith.muli %111, %149 : i64
                    %151 = arith.muli %150, %c4_i64 : i64
                    %152 = arith.addi %148, %151 : i64
                    %153 = arith.muli %123, %c512_i64 : i64
                    %154 = arith.addi %152, %153 : i64
                    %155 = arith.muli %98, %c256_i64 : i64
                    %156 = arith.addi %154, %155 : i64
                    %157 = ascscalar.bind_local_ptr %135 : memref<1xf32, 1>
                    ascvec.get_buf mte2, %128, 0 {cannir.buffer.generated_sync}
                    "ascvec.copy_gm2ub_plan"(%157, %alloc, %c0_i32, %c0_i32, %c0_i64, %156, %143, %c256_i64, %c512_i64, %c256_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = -1 : i64, ub_capacity = 4 : i64}> : (memref<1xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                    ascvec.release_buf mte2, %128, 0 {cannir.buffer.generated_sync}
                    %158 = arith.muli %98, %c32_i64 : i64
                    %159 = arith.cmpi slt, %158, %122 : i64
                    %160 = scf.if %159 -> (i64) {
                      %162 = arith.subi %122, %158 : i64
                      %163 = arith.cmpi sgt, %162, %c32_i64 : i64
                      %164 = arith.select %163, %c32_i64, %162 : i64
                      %165 = arith.cmpi eq, %arg27, %c0_i64 : i64
                      %166 = ascscalar.memref_address %alloc_15 : memref<4096xbf16, 1>
                      %167 = ascscalar.memref_address %alloc_16 : memref<4096xbf16, 1>
                      %168 = arith.select %165, %166, %167 : i64
                      %169 = arith.select %165, %c6_i64, %c7_i64 : i64
                      %170 = arith.addi %arg27, %c1_i64 : i64
                      %171 = arith.remui %170, %c2_i64 : i64
                      %172 = cannir.extract_physical_layout %arg4 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                      %173 = cannir.extract_stride %172 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                      %174 = cannir.extract_dim %173[2] : !cannir.stride<(-1, -1, -1, 1)>
                      %175 = arith.muli %174, %c2_i64 : i64
                      %176 = cannir.extract_dim %173[0] : !cannir.stride<(-1, -1, -1, 1)>
                      %177 = arith.muli %97, %176 : i64
                      %178 = arith.muli %177, %c2_i64 : i64
                      %179 = cannir.extract_dim %173[1] : !cannir.stride<(-1, -1, -1, 1)>
                      %180 = arith.muli %111, %179 : i64
                      %181 = arith.muli %180, %c2_i64 : i64
                      %182 = arith.addi %178, %181 : i64
                      %183 = arith.muli %113, %c64_i64 overflow<nsw> : i64
                      %184 = arith.addi %182, %183 : i64
                      %185 = arith.muli %123, %c64_i64 : i64
                      %186 = arith.addi %158, %185 : i64
                      %187 = arith.addi %186, %96 : i64
                      %188 = arith.muli %187, %174 : i64
                      %189 = arith.muli %188, %c2_i64 : i64
                      %190 = arith.addi %184, %189 : i64
                      %191 = ascscalar.bind_local_ptr %168 : memref<1xbf16, 1>
                      ascvec.get_buf mte2, %169, 0 {cannir.buffer.generated_sync}
                      "ascvec.copy_gm2ub_plan"(%191, %alloc_0, %c0_i32, %c0_i32, %c0_i64, %190, %164, %c64_i64, %175, %c64_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                      ascvec.release_buf mte2, %169, 0 {cannir.buffer.generated_sync}
                      scf.yield %171 : i64
                    } else {
                      scf.yield %arg27 : i64
                    }
                    %161:7 = scf.for %arg29 = %c0_i32 to %87 step %c1_i32 iter_args(%arg30 = %126, %arg31 = %arg23, %arg32 = %130, %arg33 = %132, %arg34 = %arg26, %arg35 = %160, %arg36 = %arg28) -> (i64, i64, i64, i64, i64, i64, i64)  : i32 {
                      %162 = arith.addi %84, %arg29 : i32
                      %163 = arith.addi %162, %c1_i32 : i32
                      %164 = arith.extsi %163 : i32 to i64
                      %165 = arith.cmpi eq, %164, %95 : i64
                      %166 = arith.muli %162, %c64_i32 : i32
                      %167 = arith.extsi %166 : i32 to i64
                      %168 = arith.subi %81, %167 : i64
                      %169 = arith.select %165, %168, %c64_i64 : i64
                      %170 = arith.addi %162, %c2_i32 : i32
                      %171 = arith.extsi %170 : i32 to i64
                      %172 = arith.cmpi eq, %171, %95 : i64
                      %173 = arith.muli %163, %c64_i32 : i32
                      %174 = arith.extsi %173 : i32 to i64
                      %175 = arith.subi %81, %174 : i64
                      %176 = arith.select %172, %175, %c64_i64 : i64
                      %177 = arith.cmpi eq, %arg31, %c0_i64 : i64
                      %178 = ascscalar.memref_address %alloc_18 : memref<16384xbf16, 2>
                      %179 = ascscalar.memref_address %alloc_17 : memref<16384xbf16, 2>
                      %180 = arith.addi %arg31, %c1_i64 : i64
                      %181 = arith.remui %180, %c2_i64 : i64
                      %182 = arith.addi %arg34, %c1_i64 : i64
                      %183 = arith.remui %182, %c2_i64 : i64
                      %184 = arith.addi %183, %c1_i64 overflow<nsw> : i64
                      %185 = arith.remui %184, %c2_i64 : i64
                      %186 = arith.cmpi ne, %arg31, %c0_i64 : i64
                      %187 = arith.extui %186 : i1 to i64
                      %188 = arith.addi %arg29, %c1_i32 : i32
                      %189 = arith.cmpi slt, %188, %87 : i32
                      %190:2 = scf.if %189 -> (i64, i64) {
                        %216 = arith.cmpi eq, %arg30, %c0_i64 : i64
                        %217 = arith.cmpi ne, %arg30, %c0_i64 : i64
                        %218 = arith.extui %217 : i1 to i64
                        %219 = arith.addi %arg30, %c1_i64 overflow<nsw> : i64
                        %220 = arith.remui %219, %c2_i64 : i64
                        %221 = arith.select %216, %133, %134 : i64
                        %222 = arith.minsi %164, %138 : i64
                        %223 = arith.subi %138, %222 : i64
                        %224 = arith.minsi %223, %c1_i64 : i64
                        %225 = arith.cmpi sle, %164, %138 : i64
                        %226 = arith.select %225, %224, %c0_i64 : i64
                        %227 = arith.muli %164, %c512_i64 : i64
                        %228 = arith.addi %152, %227 : i64
                        %229 = arith.addi %228, %155 : i64
                        %230 = ascscalar.bind_local_ptr %221 : memref<1xf32, 1>
                        ascvec.get_buf mte2, %218, 0 {cannir.buffer.generated_sync}
                        "ascvec.copy_gm2ub_plan"(%230, %alloc, %c0_i32, %c0_i32, %c0_i64, %229, %226, %c256_i64, %c512_i64, %c256_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = -1 : i64, ub_capacity = 4 : i64}> : (memref<1xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                        ascvec.release_buf mte2, %218, 0 {cannir.buffer.generated_sync}
                        %231 = arith.cmpi slt, %158, %176 : i64
                        %232 = scf.if %231 -> (i64) {
                          %233 = arith.subi %176, %158 : i64
                          %234 = arith.cmpi sgt, %233, %c32_i64 : i64
                          %235 = arith.select %234, %c32_i64, %233 : i64
                          %236 = arith.cmpi eq, %arg35, %c0_i64 : i64
                          %237 = ascscalar.memref_address %alloc_15 : memref<4096xbf16, 1>
                          %238 = ascscalar.memref_address %alloc_16 : memref<4096xbf16, 1>
                          %239 = arith.select %236, %237, %238 : i64
                          %240 = arith.select %236, %c6_i64, %c7_i64 : i64
                          %241 = arith.addi %arg35, %c1_i64 : i64
                          %242 = arith.remui %241, %c2_i64 : i64
                          %243 = cannir.extract_physical_layout %arg4 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                          %244 = cannir.extract_stride %243 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                          %245 = cannir.extract_dim %244[2] : !cannir.stride<(-1, -1, -1, 1)>
                          %246 = arith.muli %245, %c2_i64 : i64
                          %247 = cannir.extract_dim %244[0] : !cannir.stride<(-1, -1, -1, 1)>
                          %248 = arith.muli %97, %247 : i64
                          %249 = arith.muli %248, %c2_i64 : i64
                          %250 = cannir.extract_dim %244[1] : !cannir.stride<(-1, -1, -1, 1)>
                          %251 = arith.muli %111, %250 : i64
                          %252 = arith.muli %251, %c2_i64 : i64
                          %253 = arith.addi %249, %252 : i64
                          %254 = arith.muli %113, %c64_i64 overflow<nsw> : i64
                          %255 = arith.addi %253, %254 : i64
                          %256 = arith.muli %164, %c64_i64 : i64
                          %257 = arith.addi %158, %256 : i64
                          %258 = arith.addi %257, %96 : i64
                          %259 = arith.muli %258, %245 : i64
                          %260 = arith.muli %259, %c2_i64 : i64
                          %261 = arith.addi %255, %260 : i64
                          %262 = ascscalar.bind_local_ptr %239 : memref<1xbf16, 1>
                          ascvec.get_buf mte2, %240, 0 {cannir.buffer.generated_sync}
                          "ascvec.copy_gm2ub_plan"(%262, %alloc_0, %c0_i32, %c0_i32, %c0_i64, %261, %235, %c64_i64, %246, %c64_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                          ascvec.release_buf mte2, %240, 0 {cannir.buffer.generated_sync}
                          scf.yield %242 : i64
                        } else {
                          scf.yield %arg35 : i64
                        }
                        scf.yield %220, %232 : i64, i64
                      } else {
                        scf.yield %arg30, %arg35 : i64, i64
                      }
                      %191 = ascscalar.memref_address %alloc_20 : memref<8192xbf16, 2>
                      %192 = ascscalar.memref_address %alloc_19 : memref<8192xbf16, 2>
                      %193 = arith.select %177, %191, %192 : i64
                      %194 = arith.select %177, %c10_i64, %c11_i64 : i64
                      %195 = arith.subi %169, %158 : i64
                      %196 = arith.cmpi slt, %195, %c0_i64 : i64
                      %197 = arith.select %196, %c0_i64, %195 : i64
                      %198 = arith.cmpi sgt, %197, %c32_i64 : i64
                      %199 = arith.select %198, %c32_i64, %197 : i64
                      %200 = arith.cmpi eq, %199, %c32_i64 : i64
                      ascvec.sync_intra_wait v, %c5_i64 {cannir.sync.banks = 3 : i32}
                      %201 = scf.if %200 -> (i64) {
                        %216 = arith.cmpi eq, %arg36, %c0_i64 : i64
                        %217 = ascscalar.memref_address %alloc_15 : memref<4096xbf16, 1>
                        %218 = ascscalar.memref_address %alloc_16 : memref<4096xbf16, 1>
                        %219 = arith.select %216, %217, %218 : i64
                        %220 = ascscalar.bind_local_ptr %219 : memref<4096xbf16, 1>
                        %221 = arith.select %216, %c6_i64, %c7_i64 : i64
                        %222 = arith.addi %arg36, %c1_i64 : i64
                        %223 = arith.remui %222, %c2_i64 : i64
                        ascvec.get_buf v, %221, 0 {cannir.buffer.generated_sync}
                        ascvec.vf(%220, %alloc_7, %alloc_9 : memref<4096xbf16, 1>, memref<4096xf32, 1>, memref<4096xbf16, 1>) {
                        ^bb0(%arg37: memref<4096xbf16, 1>, %arg38: memref<4096xf32, 1>, %arg39: memref<4096xbf16, 1>):
                          %c128_i32_21 = arith.constant 128 : i32
                          %c1024_i32 = arith.constant 1024 : i32
                          %c0_i32_22 = arith.constant 0 : i32
                          %c64_i32_23 = arith.constant 64 : i32
                          %mask, %remaining_out = ascvec.update_mask %c64_i32_23 : !ascvec.mask
                          scf.for %arg40 = %c0_i32_22 to %c1024_i32 step %c128_i32_21  : i32 {
                            %224 = arith.index_cast %arg40 : i32 to index
                            %225 = ascvec.loadalign_unpack %arg37[%224] {mode = "b16_to_b32"} : memref<4096xbf16, 1>, index -> <bf16>
                            %226 = arith.addi %arg40, %c64_i32_23 overflow<nsw> : i32
                            %227 = arith.index_cast %226 : i32 to index
                            %228 = ascvec.loadalign_unpack %arg37[%227] {mode = "b16_to_b32"} : memref<4096xbf16, 1>, index -> <bf16>
                            %229 = ascvec.cast %225, %mask {rounding = 6 : i32} : <bf16> -> <f32>
                            %230 = ascvec.cast %228, %mask {rounding = 6 : i32} : <bf16> -> <f32>
                            %231 = ascvec.loadalign %arg38[%224] : memref<4096xf32, 1>, index -> <f32>
                            %232 = ascvec.loadalign %arg38[%227] : memref<4096xf32, 1>, index -> <f32>
                            %233 = ascvec.add %229, %231, %mask : <f32>
                            %234 = ascvec.add %230, %232, %mask : <f32>
                            %235 = ascvec.cast %233, %mask : <f32> -> <bf16>
                            %236 = ascvec.cast %234, %mask : <f32> -> <bf16>
                            ascvec.storealign_pack %arg39[%224], %235, %mask {mode = "b32_to_b16"} : memref<4096xbf16, 1>, index, <bf16>
                            ascvec.storealign_pack %arg39[%227], %236, %mask {mode = "b32_to_b16"} : memref<4096xbf16, 1>, index, <bf16>
                          }
                          ascvec.mem_bar vst_vld
                        } {cannir.vf_mode = "simd"}
                        ascvec.release_buf v, %221, 0 {cannir.buffer.generated_sync}
                        scf.yield %223 : i64
                      } else {
                        ascvec.vf(%alloc_9 : memref<4096xbf16, 1>) {
                        ^bb0(%arg37: memref<4096xbf16, 1>):
                          %c1_i32_21 = arith.constant 1 : i32
                          %c0_i32_22 = arith.constant 0 : i32
                          %cst = arith.constant 0.000000e+00 : bf16
                          %c32_i32_23 = arith.constant 32 : i32
                          %mask, %remaining_out = ascvec.update_mask %c32_i32_23 elem_bits 16 : !ascvec.mask
                          %216 = ascvec.duplicate_scalar %cst, %mask : bf16 -> <bf16>
                          scf.for %arg38 = %c0_i32_22 to %c32_i32_23 step %c1_i32_21  : i32 {
                            %217 = arith.muli %arg38, %c32_i32_23 overflow<nsw> : i32
                            %218 = arith.index_cast %217 : i32 to index
                            ascvec.storealign %arg37[%218], %216, %mask : memref<4096xbf16, 1>, index, <bf16>
                          }
                          ascvec.mem_bar vst_vld
                        } {cannir.vf_mode = "simd"}
                        scf.yield %arg36 : i64
                      }
                      %202 = arith.cmpi sgt, %199, %c0_i64 : i64
                      %203 = arith.cmpi slt, %199, %c32_i64 : i64
                      %204 = arith.select %202, %203, %false : i1
                      %205 = scf.if %204 -> (i64) {
                        %216 = arith.cmpi eq, %201, %c0_i64 : i64
                        %217 = ascscalar.memref_address %alloc_15 : memref<4096xbf16, 1>
                        %218 = ascscalar.memref_address %alloc_16 : memref<4096xbf16, 1>
                        %219 = arith.select %216, %217, %218 : i64
                        %220 = ascscalar.bind_local_ptr %219 : memref<4096xbf16, 1>
                        %221 = arith.select %216, %c6_i64, %c7_i64 : i64
                        %222 = arith.addi %201, %c1_i64 : i64
                        %223 = arith.remui %222, %c2_i64 : i64
                        ascvec.get_buf v, %221, 0 {cannir.buffer.generated_sync}
                        ascvec.vf(%220, %alloc_7, %alloc_9, %199 : memref<4096xbf16, 1>, memref<4096xf32, 1>, memref<4096xbf16, 1>, i64) {
                        ^bb0(%arg37: memref<4096xbf16, 1>, %arg38: memref<4096xf32, 1>, %arg39: memref<4096xbf16, 1>, %arg40: i64):
                          %c1_i64_21 = arith.constant 1 : i64
                          %c0_i64_22 = arith.constant 0 : i64
                          %c32_i64_23 = arith.constant 32 : i64
                          %c32_i32_24 = arith.constant 32 : i32
                          scf.for %arg41 = %c0_i64_22 to %arg40 step %c1_i64_21  : i64 {
                            %mask, %remaining_out = ascvec.update_mask %c32_i32_24 : !ascvec.mask
                            %224 = arith.muli %arg41, %c32_i64_23 : i64
                            %225 = arith.index_cast %224 : i64 to index
                            %226 = ascvec.loadalign_unpack %arg37[%225] {mode = "b16_to_b32"} : memref<4096xbf16, 1>, index -> <bf16>
                            %227 = ascvec.cast %226, %mask {rounding = 6 : i32} : <bf16> -> <f32>
                            %228 = ascvec.loadalign %arg38[%225] : memref<4096xf32, 1>, index -> <f32>
                            %229 = ascvec.add %227, %228, %mask : <f32>
                            %230 = ascvec.cast %229, %mask : <f32> -> <bf16>
                            ascvec.storealign_pack %arg39[%225], %230, %mask {mode = "b32_to_b16"} : memref<4096xbf16, 1>, index, <bf16>
                          } {cannir.loop_unroll = 2 : i64}
                          ascvec.mem_bar vst_vld
                        } {cannir.vf_mode = "simd"}
                        ascvec.release_buf v, %221, 0 {cannir.buffer.generated_sync}
                        scf.yield %223 : i64
                      } else {
                        scf.yield %201 : i64
                      }
                      ascvec.sync_intra_arrive v, %c5_i64 {cannir.sync.banks = 3 : i32}
                      ascvec.get_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
                      ascvec.vf(%alloc_9, %alloc_12 : memref<4096xbf16, 1>, memref<4224xbf16, 1>) {
                      ^bb0(%arg37: memref<4096xbf16, 1>, %arg38: memref<4224xbf16, 1>):
                        %c1_i32_21 = arith.constant 1 : i32
                        %c0_i32_22 = arith.constant 0 : i32
                        %c33 = arith.constant 33 : index
                        %c16_i32_23 = arith.constant 16 : i32
                        %c0_24 = arith.constant 0 : index
                        %c32_i32_25 = arith.constant 32 : i32
                        scf.for %arg39 = %c0_i32_22 to %c32_i32_25 step %c1_i32_21  : i32 {
                          %mask, %remaining_out = ascvec.update_mask %c32_i32_25 elem_bits 16 : !ascvec.mask
                          %216 = arith.muli %arg39, %c32_i32_25 overflow<nsw> : i32
                          %217 = arith.index_cast %216 : i32 to index
                          %218 = ascvec.loadalign %arg37[%217] : memref<4096xbf16, 1>, index -> <bf16>
                          %219 = arith.muli %arg39, %c16_i32_23 overflow<nsw> : i32
                          %220 = arith.index_cast %219 : i32 to index
                          ascvec.storealign_strided %arg38[%220], %218, %mask, %c33, %c0_24 : memref<4224xbf16, 1>, <bf16>
                        }
                      } {cannir.vf_mode = "simd"}
                      ascvec.release_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
                      %206 = ascscalar.bitcast_memref %alloc_12 byte_offset = %c0_i64 : memref<4224xbf16, 1> -> memref<?xbf16, 1>
                      %207 = ascscalar.bind_local_ptr %193 : memref<1xbf16, 2>
                      %208 = arith.muli %98, %c512_i64 : i64
                      %209 = arith.cmpi eq, %194, %c11_i64 : i64
                      %210 = arith.select %209, %c3_i64, %c2_i64 : i64
                      ascvec.sync_intra_wait mte3, %210 {cannir.sync.banks = 3 : i32}
                      ascvec.get_buf mte3, %c3_i64, 0 {cannir.buffer.generated_sync}
                      %211 = arith.index_cast %208 : i64 to index
                      ascvec.copy_ub2l1_strided %207[%211], %206[%c0], %c2_i32, %c32_i32, %c1_i32, %c32_i32 : memref<1xbf16, 2>, memref<?xbf16, 1>
                      ascvec.release_buf mte3, %c3_i64, 0 {cannir.buffer.generated_sync}
                      ascvec.sync_intra_arrive mte3, %210 {cannir.sync.banks = 3 : i32}
                      %212 = scf.if %189 -> (i64) {
                        %216 = arith.addi %arg32, %c1_i64 overflow<nsw> : i64
                        %217 = arith.remui %216, %c2_i64 : i64
                        scf.yield %217 : i64
                      } else {
                        scf.yield %arg32 : i64
                      }
                      %213 = arith.select %177, %133, %134 : i64
                      %214 = ascscalar.bind_local_ptr %213 : memref<64xf32, 1>
                      ascvec.sync_intra_wait v, %c4_i64 {cannir.sync.banks = 3 : i32}
                      ascvec.get_buf v, %187, 0 {cannir.buffer.generated_sync}
                      ascvec.vf(%214, %alloc_8, %alloc_6 : memref<64xf32, 1>, memref<8192xf32, 1>, memref<8192xf32, 1>) {
                      ^bb0(%arg37: memref<64xf32, 1>, %arg38: memref<8192xf32, 1>, %arg39: memref<8192xf32, 1>):
                        %c2048_i32 = arith.constant 2048 : i32
                        %c0_i32_21 = arith.constant 0 : i32
                        %c0_22 = arith.constant 0 : index
                        %c64_i32_23 = arith.constant 64 : i32
                        %mask, %remaining_out = ascvec.update_mask %c64_i32_23 : !ascvec.mask
                        %216 = ascvec.loadalign %arg37[%c0_22] : memref<64xf32, 1>, index -> <f32>
                        scf.for %arg40 = %c0_i32_21 to %c2048_i32 step %c64_i32_23  : i32 {
                          %217 = arith.index_cast %arg40 : i32 to index
                          %218 = ascvec.loadalign %arg38[%217] : memref<8192xf32, 1>, index -> <f32>
                          %219 = ascvec.loadalign %arg39[%217] : memref<8192xf32, 1>, index -> <f32>
                          %220 = ascvec.mul %218, %216, %mask : <f32>
                          %221 = ascvec.add %220, %219, %mask : <f32>
                          ascvec.storealign %arg38[%217], %221, %mask : memref<8192xf32, 1>, index, <f32>
                        } {cannir.loop_unroll = 4 : i64}
                      } {cannir.vf_mode = "simd"}
                      ascvec.release_buf v, %187, 0 {cannir.buffer.generated_sync}
                      ascvec.sync_intra_arrive v, %c4_i64 {cannir.sync.banks = 3 : i32}
                      %215 = scf.if %189 -> (i64) {
                        ascvec.vf(%alloc_8, %alloc_10 : memref<8192xf32, 1>, memref<8192xbf16, 1>) {
                        ^bb0(%arg37: memref<8192xf32, 1>, %arg38: memref<8192xbf16, 1>):
                          %c1_i32_21 = arith.constant 1 : i32
                          %c32_i32_22 = arith.constant 32 : i32
                          %c0_i32_23 = arith.constant 0 : i32
                          %c64_i32_24 = arith.constant 64 : i32
                          scf.for %arg39 = %c0_i32_23 to %c32_i32_22 step %c1_i32_21  : i32 {
                            %mask, %remaining_out = ascvec.update_mask %c64_i32_24 : !ascvec.mask
                            %227 = arith.muli %arg39, %c64_i32_24 overflow<nsw> : i32
                            %228 = arith.index_cast %227 : i32 to index
                            %229 = ascvec.loadalign %arg37[%228] : memref<8192xf32, 1>, index -> <f32>
                            %230 = ascvec.cast %229, %mask : <f32> -> <bf16>
                            ascvec.storealign_pack %arg38[%228], %230, %mask {mode = "b32_to_b16"} : memref<8192xbf16, 1>, index, <bf16>
                          }
                        } {cannir.vf_mode = "simd"}
                        ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                        ascvec.vf(%alloc_10, %alloc_11 : memref<8192xbf16, 1>, memref<8256xbf16, 1>) {
                        ^bb0(%arg37: memref<8192xbf16, 1>, %arg38: memref<8256xbf16, 1>):
                          %c1_i32_21 = arith.constant 1 : i32
                          %c32_i32_22 = arith.constant 32 : i32
                          %c0_i32_23 = arith.constant 0 : i32
                          %c33 = arith.constant 33 : index
                          %c16_i32_24 = arith.constant 16 : i32
                          %c0_25 = arith.constant 0 : index
                          %c64_i32_26 = arith.constant 64 : i32
                          scf.for %arg39 = %c0_i32_23 to %c32_i32_22 step %c1_i32_21  : i32 {
                            %mask, %remaining_out = ascvec.update_mask %c64_i32_26 elem_bits 16 : !ascvec.mask
                            %227 = arith.muli %arg39, %c64_i32_26 overflow<nsw> : i32
                            %228 = arith.index_cast %227 : i32 to index
                            %229 = ascvec.loadalign %arg37[%228] : memref<8192xbf16, 1>, index -> <bf16>
                            %230 = arith.muli %arg39, %c16_i32_24 overflow<nsw> : i32
                            %231 = arith.index_cast %230 : i32 to index
                            ascvec.storealign_strided %arg38[%231], %229, %mask, %c33, %c0_25 : memref<8256xbf16, 1>, <bf16>
                          }
                        } {cannir.vf_mode = "simd"}
                        ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                        %216 = arith.cmpi eq, %arg33, %c0_i64 : i64
                        %217 = arith.select %216, %178, %179 : i64
                        %218 = arith.select %216, %c8_i64, %c9_i64 : i64
                        %219 = arith.addi %arg33, %c1_i64 overflow<nsw> : i64
                        %220 = arith.remui %219, %c2_i64 : i64
                        %221 = ascscalar.bitcast_memref %alloc_11 byte_offset = %c0_i64 : memref<8256xbf16, 1> -> memref<?xbf16, 1>
                        %222 = ascscalar.bind_local_ptr %217 : memref<1xbf16, 2>
                        %223 = arith.muli %98, %c2048_i64 : i64
                        %224 = arith.cmpi eq, %218, %c9_i64 : i64
                        %225 = arith.extui %224 : i1 to i64
                        ascvec.sync_intra_wait mte3, %225 {cannir.sync.banks = 3 : i32}
                        ascvec.get_buf mte3, %c2_i64, 0 {cannir.buffer.generated_sync}
                        %226 = arith.index_cast %223 : i64 to index
                        ascvec.copy_ub2l1_strided %222[%226], %221[%c0], %c4_i32, %c32_i32, %c1_i32, %c0_i32 : memref<1xbf16, 2>, memref<?xbf16, 1>
                        ascvec.release_buf mte3, %c2_i64, 0 {cannir.buffer.generated_sync}
                        ascvec.sync_intra_arrive mte3, %225 {cannir.sync.banks = 3 : i32}
                        scf.yield %220 : i64
                      } else {
                        ascvec.get_buf v, %c5_i64, 0 {cannir.buffer.generated_sync}
                        ascvec.copy_ub2ub %alloc_14[%c0], %alloc_8[%c0], %c8192_i32 : memref<8192xf32, 1>, memref<8192xf32, 1>
                        ascvec.vf(%alloc_14, %alloc_8 : memref<8192xf32, 1>, memref<8192xf32, 1>) {
                        ^bb0(%arg37: memref<8192xf32, 1>, %arg38: memref<8192xf32, 1>):
                        } {cannir.vf_mode = "simd"}
                        ascvec.release_buf v, %c5_i64, 0 {cannir.buffer.generated_sync}
                        %216 = cannir.extract_physical_layout %arg8 : !cannir.tensor<<gm, f32>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, identity>
                        %217 = cannir.extract_stride %216 : <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>> -> <(-1, 16384, 128, 1)>
                        %218 = cannir.extract_dim %217[0] : !cannir.stride<(-1, 16384, 128, 1)>
                        %219 = arith.muli %114, %218 : i64
                        %220 = arith.muli %111, %c16384_i64 : i64
                        %221 = arith.addi %219, %220 : i64
                        %222 = arith.muli %113, %c4096_i64 overflow<nsw> : i64
                        %223 = arith.addi %221, %222 : i64
                        %224 = arith.muli %98, %c64_i64 : i64
                        %225 = arith.addi %223, %224 : i64
                        %226 = arith.muli %225, %c32_i64 : i64
                        %227 = ascscalar.memref_address %alloc_2 : memref<1xf32>
                        %228 = arith.divui %226, %c8_i64 : i64
                        %229 = arith.addi %227, %228 : i64
                        %230 = ascscalar.bind_gm_ptr %229 : memref<1xf32>
                        %231 = ascscalar.bitcast_memref %alloc_14 byte_offset = %c0_i64 : memref<8192xf32, 1> -> memref<?xf32, 1>
                        ascvec.get_buf mte3, %c5_i64, 0 {cannir.buffer.generated_sync}
                        ascvec.copy_ub2gm %230[%c0], %231[%c0], %c32_i32, %c256_i32, %c512_i64, %c256_i32 l2_cache_ctl %c0_i32 : memref<1xf32>, memref<?xf32, 1>
                        ascvec.release_buf mte3, %c5_i64, 0 {cannir.buffer.generated_sync}
                        scf.yield %arg33 : i64
                      }
                      scf.yield %190#0, %181, %212, %215, %185, %190#1, %205 : i64, i64, i64, i64, i64, i64, i64
                    }
                    scf.yield %161#0, %161#1, %161#2, %161#3, %161#4, %161#5, %161#6 : i64, i64, i64, i64, i64, i64, i64
                  }
                  scf.yield %101#0, %101#1, %101#2, %101#3, %101#4, %101#5, %101#6 : i64, i64, i64, i64, i64, i64, i64
                } else {
                  scf.yield %arg14, %arg15, %arg16, %arg17, %arg18, %arg19, %arg20 : i64, i64, i64, i64, i64, i64, i64
                }
                scf.yield %100#0, %100#1, %100#2, %100#3, %100#4, %100#5, %100#6 : i64, i64, i64, i64, i64, i64, i64
              }
            } else {
              %51:7 = scf.for %arg13 = %c0_i32 to %21 step %c1_i32 iter_args(%arg14 = %c0_i64, %arg15 = %c0_i64, %arg16 = %c0_i64, %arg17 = %c0_i64, %arg18 = %c0_i64, %arg19 = %c0_i64, %arg20 = %c0_i64) -> (i64, i64, i64, i64, i64, i64, i64)  : i32 {
                %52 = arith.addi %32, %arg13 : i32
                %53 = arith.index_cast %52 : i32 to index
                %54 = memref.load %alloc_3[%53] : memref<1xi32>
                %55 = arith.addi %52, %c1_i32 : i32
                %56 = arith.index_cast %55 : i32 to index
                %57 = memref.load %alloc_3[%56] : memref<1xi32>
                %58 = arith.cmpi slt, %39, %54 : i32
                %59 = arith.select %58, %54, %39 : i32
                %60 = arith.cmpi sgt, %42, %57 : i32
                %61 = arith.select %60, %57, %42 : i32
                %62 = arith.cmpi slt, %59, %61 : i32
                %63 = arith.subi %59, %54 : i32
                %64 = arith.subi %61, %54 : i32
                %65 = arith.extsi %63 : i32 to i64
                %66 = arith.extsi %64 : i32 to i64
                %67 = arith.select %62, %65, %c0_i64 : i64
                %68 = arith.select %62, %66, %c0_i64 : i64
                %69 = arith.addi %22, %arg13 : i32
                %70 = arith.index_cast %69 : i32 to index
                %71 = memref.load %alloc_3[%70] : memref<1xi32>
                %72 = arith.addi %23, %arg13 : i32
                %73 = arith.index_cast %72 : i32 to index
                %74 = memref.load %alloc_3[%73] : memref<1xi32>
                %75 = arith.addi %24, %arg13 : i32
                %76 = arith.index_cast %75 : i32 to index
                %77 = memref.load %alloc_3[%76] : memref<1xi32>
                %78 = arith.addi %25, %arg13 : i32
                %79 = arith.index_cast %78 : i32 to index
                %80 = memref.load %alloc_3[%79] : memref<1xi32>
                %81 = arith.extsi %80 : i32 to i64
                %82 = arith.addi %27, %arg13 : i32
                %83 = arith.index_cast %82 : i32 to index
                %84 = memref.load %alloc_3[%83] : memref<1xi32>
                %85 = arith.addi %28, %arg13 : i32
                %86 = arith.index_cast %85 : i32 to index
                %87 = memref.load %alloc_3[%86] : memref<1xi32>
                %88 = arith.addi %81, %c63_i64 : i64
                %89 = arith.remsi %88, %c64_i64 : i64
                %90 = arith.cmpi ne, %89, %c0_i64 : i64
                %91 = arith.cmpi slt, %89, %c0_i64 : i64
                %92 = arith.andi %90, %91 : i1
                %93 = arith.divsi %88, %c64_i64 : i64
                %94 = arith.extui %92 : i1 to i64
                %95 = arith.subi %93, %94 : i64
                %96 = arith.extsi %77 : i32 to i64
                %97 = arith.extsi %74 : i32 to i64
                %98 = ascscalar.get_subblock_id
                %99 = arith.cmpi slt, %0, %1 : i64
                %100:7 = scf.if %99 -> (i64, i64, i64, i64, i64, i64, i64) {
                  %101:7 = scf.for %arg21 = %67 to %68 step %c1_i64 iter_args(%arg22 = %arg14, %arg23 = %arg15, %arg24 = %arg16, %arg25 = %arg17, %arg26 = %arg18, %arg27 = %arg19, %arg28 = %arg20) -> (i64, i64, i64, i64, i64, i64, i64)  : i64 {
                    %102 = arith.remsi %arg21, %c8_i64 : i64
                    %103 = arith.cmpi slt, %102, %c0_i64 : i64
                    %104 = arith.divsi %arg21, %c8_i64 : i64
                    %105 = arith.extui %103 : i1 to i64
                    %106 = arith.subi %104, %105 : i64
                    %107 = arith.maxsi %7, %c1_i64 : i64
                    %108 = arith.remsi %106, %107 : i64
                    %109 = arith.cmpi slt, %108, %c0_i64 : i64
                    %110 = arith.select %109, %107, %c0_i64 : i64
                    %111 = arith.addi %108, %110 : i64
                    %112 = arith.select %103, %c8_i64, %c0_i64 : i64
                    %113 = arith.addi %102, %112 overflow<nsw> : i64
                    %114 = arith.extsi %71 : i32 to i64
                    %115 = arith.addi %84, %87 : i32
                    %116 = arith.extsi %115 : i32 to i64
                    %117 = arith.cmpi eq, %116, %95 : i64
                    %118 = arith.muli %84, %c64_i32 : i32
                    %119 = arith.extsi %118 : i32 to i64
                    %120 = arith.subi %81, %119 : i64
                    %121 = arith.minsi %120, %c64_i64 : i64
                    %122 = arith.select %117, %121, %c64_i64 : i64
                    %123 = arith.extsi %84 : i32 to i64
                    %124 = arith.cmpi eq, %arg22, %c0_i64 : i64
                    %125 = arith.addi %arg22, %c1_i64 : i64
                    %126 = arith.remui %125, %c2_i64 : i64
                    %127 = arith.cmpi ne, %arg22, %c0_i64 : i64
                    %128 = arith.extui %127 : i1 to i64
                    %129 = arith.addi %arg24, %c1_i64 : i64
                    %130 = arith.remui %129, %c2_i64 : i64
                    %131 = arith.cmpi eq, %84, %c0_i32 : i32
                    %132 = scf.if %131 -> (i64) {
                      %162 = cannir.extract_physical_layout %arg7 : !cannir.tensor<<gm, f32>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, identity>
                      %163 = cannir.extract_stride %162 : <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>> -> <(-1, 16384, 128, 1)>
                      %164 = cannir.extract_dim %163[0] : !cannir.stride<(-1, 16384, 128, 1)>
                      %165 = arith.muli %114, %164 : i64
                      %166 = arith.muli %165, %c4_i64 : i64
                      %167 = arith.muli %111, %c65536_i64 : i64
                      %168 = arith.addi %166, %167 : i64
                      %169 = arith.muli %113, %c8192_i64 overflow<nsw> : i64
                      %170 = arith.addi %168, %169 : i64
                      %171 = arith.muli %98, %c256_i64 : i64
                      %172 = arith.addi %170, %171 : i64
                      ascvec.get_buf mte2, %c4_i64, 0 {cannir.buffer.generated_sync}
                      "ascvec.copy_gm2ub_plan"(%45, %alloc_1, %c0_i32, %c0_i32, %c0_i64, %172, %c16_i64, %c256_i64, %c512_i64, %c256_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 131968 : i64, ub_capacity = 32768 : i64}> : (memref<?xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                      ascvec.release_buf mte2, %c4_i64, 0 {cannir.buffer.generated_sync}
                      ascvec.get_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
                      ascvec.copy_ub2ub %alloc_8[%c0], %alloc_13[%c0], %c4096_i32 : memref<8192xf32, 1>, memref<8192xf32, 1>
                      ascvec.vf(%alloc_8, %alloc_13 : memref<8192xf32, 1>, memref<8192xf32, 1>) {
                      ^bb0(%arg29: memref<8192xf32, 1>, %arg30: memref<8192xf32, 1>):
                      } {cannir.vf_mode = "simd"}
                      ascvec.release_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
                      ascvec.vf(%alloc_8, %alloc_10 : memref<8192xf32, 1>, memref<8192xbf16, 1>) {
                      ^bb0(%arg29: memref<8192xf32, 1>, %arg30: memref<8192xbf16, 1>):
                        %c1_i32_21 = arith.constant 1 : i32
                        %c16_i32_22 = arith.constant 16 : i32
                        %c0_i32_23 = arith.constant 0 : i32
                        %c64_i32_24 = arith.constant 64 : i32
                        scf.for %arg31 = %c0_i32_23 to %c16_i32_22 step %c1_i32_21  : i32 {
                          %mask, %remaining_out = ascvec.update_mask %c64_i32_24 : !ascvec.mask
                          %186 = arith.muli %arg31, %c64_i32_24 overflow<nsw> : i32
                          %187 = arith.index_cast %186 : i32 to index
                          %188 = ascvec.loadalign %arg29[%187] : memref<8192xf32, 1>, index -> <f32>
                          %189 = ascvec.cast %188, %mask : <f32> -> <bf16>
                          ascvec.storealign_pack %arg30[%187], %189, %mask {mode = "b32_to_b16"} : memref<8192xbf16, 1>, index, <bf16>
                        }
                      } {cannir.vf_mode = "simd"}
                      ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                      ascvec.vf(%alloc_10, %alloc_11 : memref<8192xbf16, 1>, memref<8256xbf16, 1>) {
                      ^bb0(%arg29: memref<8192xbf16, 1>, %arg30: memref<8256xbf16, 1>):
                        %c1_i32_21 = arith.constant 1 : i32
                        %c0_i32_22 = arith.constant 0 : i32
                        %c17 = arith.constant 17 : index
                        %c16_i32_23 = arith.constant 16 : i32
                        %c0_24 = arith.constant 0 : index
                        %c64_i32_25 = arith.constant 64 : i32
                        scf.for %arg31 = %c0_i32_22 to %c16_i32_23 step %c1_i32_21  : i32 {
                          %mask, %remaining_out = ascvec.update_mask %c64_i32_25 elem_bits 16 : !ascvec.mask
                          %186 = arith.muli %arg31, %c64_i32_25 overflow<nsw> : i32
                          %187 = arith.index_cast %186 : i32 to index
                          %188 = ascvec.loadalign %arg29[%187] : memref<8192xbf16, 1>, index -> <bf16>
                          %189 = arith.muli %arg31, %c16_i32_23 overflow<nsw> : i32
                          %190 = arith.index_cast %189 : i32 to index
                          ascvec.storealign_strided %arg30[%190], %188, %mask, %c17, %c0_24 : memref<8256xbf16, 1>, <bf16>
                        }
                      } {cannir.vf_mode = "simd"}
                      ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                      %173 = arith.cmpi eq, %arg25, %c0_i64 : i64
                      %174 = ascscalar.memref_address %alloc_18 : memref<16384xbf16, 2>
                      %175 = ascscalar.memref_address %alloc_17 : memref<16384xbf16, 2>
                      %176 = arith.select %173, %174, %175 : i64
                      %177 = arith.select %173, %c8_i64, %c9_i64 : i64
                      %178 = arith.addi %arg25, %c1_i64 : i64
                      %179 = arith.remui %178, %c2_i64 : i64
                      %180 = ascscalar.bitcast_memref %alloc_11 byte_offset = %c0_i64 : memref<8256xbf16, 1> -> memref<?xbf16, 1>
                      %181 = ascscalar.bind_local_ptr %176 : memref<1xbf16, 2>
                      %182 = arith.muli %98, %c1024_i64 : i64
                      %183 = arith.cmpi eq, %177, %c9_i64 : i64
                      %184 = arith.extui %183 : i1 to i64
                      ascvec.sync_intra_wait mte3, %184 {cannir.sync.banks = 3 : i32}
                      ascvec.get_buf mte3, %c2_i64, 0 {cannir.buffer.generated_sync}
                      %185 = arith.index_cast %182 : i64 to index
                      ascvec.copy_ub2l1_strided %181[%185], %180[%c0], %c4_i32, %c16_i32, %c1_i32, %c0_i32 : memref<1xbf16, 2>, memref<?xbf16, 1>
                      ascvec.release_buf mte3, %c2_i64, 0 {cannir.buffer.generated_sync}
                      ascvec.sync_intra_arrive mte3, %184 {cannir.sync.banks = 3 : i32}
                      scf.yield %179 : i64
                    } else {
                      %162 = cannir.extract_physical_layout %arg8 : !cannir.tensor<<gm, f32>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, identity>
                      %163 = cannir.extract_stride %162 : <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>> -> <(-1, 16384, 128, 1)>
                      %164 = cannir.extract_dim %163[0] : !cannir.stride<(-1, 16384, 128, 1)>
                      %165 = arith.muli %114, %164 : i64
                      %166 = arith.muli %165, %c4_i64 : i64
                      %167 = arith.muli %111, %c65536_i64 : i64
                      %168 = arith.addi %166, %167 : i64
                      %169 = arith.muli %113, %c8192_i64 overflow<nsw> : i64
                      %170 = arith.addi %168, %169 : i64
                      %171 = arith.muli %98, %c256_i64 : i64
                      %172 = arith.addi %170, %171 : i64
                      ascvec.get_buf mte2, %c4_i64, 0 {cannir.buffer.generated_sync}
                      "ascvec.copy_gm2ub_plan"(%45, %alloc_2, %c0_i32, %c0_i32, %c0_i64, %172, %c16_i64, %c256_i64, %c512_i64, %c256_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 131968 : i64, ub_capacity = 32768 : i64}> : (memref<?xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                      ascvec.release_buf mte2, %c4_i64, 0 {cannir.buffer.generated_sync}
                      ascvec.get_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
                      ascvec.copy_ub2ub %alloc_8[%c0], %alloc_13[%c0], %c4096_i32 : memref<8192xf32, 1>, memref<8192xf32, 1>
                      ascvec.vf(%alloc_8, %alloc_13 : memref<8192xf32, 1>, memref<8192xf32, 1>) {
                      ^bb0(%arg29: memref<8192xf32, 1>, %arg30: memref<8192xf32, 1>):
                      } {cannir.vf_mode = "simd"}
                      ascvec.release_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
                      ascvec.vf(%alloc_8, %alloc_10 : memref<8192xf32, 1>, memref<8192xbf16, 1>) {
                      ^bb0(%arg29: memref<8192xf32, 1>, %arg30: memref<8192xbf16, 1>):
                        %c1_i32_21 = arith.constant 1 : i32
                        %c16_i32_22 = arith.constant 16 : i32
                        %c0_i32_23 = arith.constant 0 : i32
                        %c64_i32_24 = arith.constant 64 : i32
                        scf.for %arg31 = %c0_i32_23 to %c16_i32_22 step %c1_i32_21  : i32 {
                          %mask, %remaining_out = ascvec.update_mask %c64_i32_24 : !ascvec.mask
                          %186 = arith.muli %arg31, %c64_i32_24 overflow<nsw> : i32
                          %187 = arith.index_cast %186 : i32 to index
                          %188 = ascvec.loadalign %arg29[%187] : memref<8192xf32, 1>, index -> <f32>
                          %189 = ascvec.cast %188, %mask : <f32> -> <bf16>
                          ascvec.storealign_pack %arg30[%187], %189, %mask {mode = "b32_to_b16"} : memref<8192xbf16, 1>, index, <bf16>
                        }
                      } {cannir.vf_mode = "simd"}
                      ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                      ascvec.vf(%alloc_10, %alloc_11 : memref<8192xbf16, 1>, memref<8256xbf16, 1>) {
                      ^bb0(%arg29: memref<8192xbf16, 1>, %arg30: memref<8256xbf16, 1>):
                        %c1_i32_21 = arith.constant 1 : i32
                        %c0_i32_22 = arith.constant 0 : i32
                        %c17 = arith.constant 17 : index
                        %c16_i32_23 = arith.constant 16 : i32
                        %c0_24 = arith.constant 0 : index
                        %c64_i32_25 = arith.constant 64 : i32
                        scf.for %arg31 = %c0_i32_22 to %c16_i32_23 step %c1_i32_21  : i32 {
                          %mask, %remaining_out = ascvec.update_mask %c64_i32_25 elem_bits 16 : !ascvec.mask
                          %186 = arith.muli %arg31, %c64_i32_25 overflow<nsw> : i32
                          %187 = arith.index_cast %186 : i32 to index
                          %188 = ascvec.loadalign %arg29[%187] : memref<8192xbf16, 1>, index -> <bf16>
                          %189 = arith.muli %arg31, %c16_i32_23 overflow<nsw> : i32
                          %190 = arith.index_cast %189 : i32 to index
                          ascvec.storealign_strided %arg30[%190], %188, %mask, %c17, %c0_24 : memref<8256xbf16, 1>, <bf16>
                        }
                      } {cannir.vf_mode = "simd"}
                      ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                      %173 = arith.cmpi eq, %arg25, %c0_i64 : i64
                      %174 = ascscalar.memref_address %alloc_18 : memref<16384xbf16, 2>
                      %175 = ascscalar.memref_address %alloc_17 : memref<16384xbf16, 2>
                      %176 = arith.select %173, %174, %175 : i64
                      %177 = arith.select %173, %c8_i64, %c9_i64 : i64
                      %178 = arith.addi %arg25, %c1_i64 : i64
                      %179 = arith.remui %178, %c2_i64 : i64
                      %180 = ascscalar.bitcast_memref %alloc_11 byte_offset = %c0_i64 : memref<8256xbf16, 1> -> memref<?xbf16, 1>
                      %181 = ascscalar.bind_local_ptr %176 : memref<1xbf16, 2>
                      %182 = arith.muli %98, %c1024_i64 : i64
                      %183 = arith.cmpi eq, %177, %c9_i64 : i64
                      %184 = arith.extui %183 : i1 to i64
                      ascvec.sync_intra_wait mte3, %184 {cannir.sync.banks = 3 : i32}
                      ascvec.get_buf mte3, %c2_i64, 0 {cannir.buffer.generated_sync}
                      %185 = arith.index_cast %182 : i64 to index
                      ascvec.copy_ub2l1_strided %181[%185], %180[%c0], %c4_i32, %c16_i32, %c1_i32, %c0_i32 : memref<1xbf16, 2>, memref<?xbf16, 1>
                      ascvec.release_buf mte3, %c2_i64, 0 {cannir.buffer.generated_sync}
                      ascvec.sync_intra_arrive mte3, %184 {cannir.sync.banks = 3 : i32}
                      scf.yield %179 : i64
                    }
                    %133 = ascscalar.memref_address %alloc_4 : memref<64xf32, 1>
                    %134 = ascscalar.memref_address %alloc_5 : memref<64xf32, 1>
                    %135 = arith.select %124, %133, %134 : i64
                    %136 = cannir.extract_layout %arg1 : !cannir.tensor<<gm, f32>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                    %137 = cannir.extract_shape %136 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, -1, 128)>
                    %138 = cannir.extract_dim %137[2] : !cannir.shape<(-1, -1, -1, 128)>
                    %139 = arith.minsi %123, %138 : i64
                    %140 = arith.subi %138, %139 : i64
                    %141 = arith.minsi %140, %c1_i64 : i64
                    %142 = arith.cmpi sle, %123, %138 : i64
                    %143 = arith.select %142, %141, %c0_i64 : i64
                    %144 = cannir.extract_physical_layout %arg1 : !cannir.tensor<<gm, f32>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                    %145 = cannir.extract_stride %144 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, 128, 1)>
                    %146 = cannir.extract_dim %145[0] : !cannir.stride<(-1, -1, 128, 1)>
                    %147 = arith.muli %114, %146 : i64
                    %148 = arith.muli %147, %c4_i64 : i64
                    %149 = cannir.extract_dim %145[1] : !cannir.stride<(-1, -1, 128, 1)>
                    %150 = arith.muli %111, %149 : i64
                    %151 = arith.muli %150, %c4_i64 : i64
                    %152 = arith.addi %148, %151 : i64
                    %153 = arith.muli %123, %c512_i64 : i64
                    %154 = arith.addi %152, %153 : i64
                    %155 = arith.muli %98, %c256_i64 : i64
                    %156 = arith.addi %154, %155 : i64
                    %157 = ascscalar.bind_local_ptr %135 : memref<1xf32, 1>
                    ascvec.get_buf mte2, %128, 0 {cannir.buffer.generated_sync}
                    "ascvec.copy_gm2ub_plan"(%157, %alloc, %c0_i32, %c0_i32, %c0_i64, %156, %143, %c256_i64, %c512_i64, %c256_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = -1 : i64, ub_capacity = 4 : i64}> : (memref<1xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                    ascvec.release_buf mte2, %128, 0 {cannir.buffer.generated_sync}
                    %158 = arith.muli %98, %c32_i64 : i64
                    %159 = arith.cmpi slt, %158, %122 : i64
                    %160 = scf.if %159 -> (i64) {
                      %162 = arith.subi %122, %158 : i64
                      %163 = arith.cmpi sgt, %162, %c32_i64 : i64
                      %164 = arith.select %163, %c32_i64, %162 : i64
                      %165 = arith.cmpi eq, %arg27, %c0_i64 : i64
                      %166 = ascscalar.memref_address %alloc_15 : memref<4096xbf16, 1>
                      %167 = ascscalar.memref_address %alloc_16 : memref<4096xbf16, 1>
                      %168 = arith.select %165, %166, %167 : i64
                      %169 = arith.select %165, %c6_i64, %c7_i64 : i64
                      %170 = arith.addi %arg27, %c1_i64 : i64
                      %171 = arith.remui %170, %c2_i64 : i64
                      %172 = cannir.extract_physical_layout %arg4 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                      %173 = cannir.extract_stride %172 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                      %174 = cannir.extract_dim %173[2] : !cannir.stride<(-1, -1, -1, 1)>
                      %175 = arith.muli %174, %c2_i64 : i64
                      %176 = cannir.extract_dim %173[0] : !cannir.stride<(-1, -1, -1, 1)>
                      %177 = arith.muli %97, %176 : i64
                      %178 = arith.muli %177, %c2_i64 : i64
                      %179 = cannir.extract_dim %173[1] : !cannir.stride<(-1, -1, -1, 1)>
                      %180 = arith.muli %111, %179 : i64
                      %181 = arith.muli %180, %c2_i64 : i64
                      %182 = arith.addi %178, %181 : i64
                      %183 = arith.muli %113, %c32_i64 overflow<nsw> : i64
                      %184 = arith.addi %182, %183 : i64
                      %185 = arith.muli %123, %c64_i64 : i64
                      %186 = arith.addi %158, %185 : i64
                      %187 = arith.addi %186, %96 : i64
                      %188 = arith.muli %187, %174 : i64
                      %189 = arith.muli %188, %c2_i64 : i64
                      %190 = arith.addi %184, %189 : i64
                      %191 = ascscalar.bind_local_ptr %168 : memref<1xbf16, 1>
                      ascvec.get_buf mte2, %169, 0 {cannir.buffer.generated_sync}
                      "ascvec.copy_gm2ub_plan"(%191, %alloc_0, %c0_i32, %c0_i32, %c0_i64, %190, %164, %c32_i64, %175, %c32_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                      ascvec.release_buf mte2, %169, 0 {cannir.buffer.generated_sync}
                      scf.yield %171 : i64
                    } else {
                      scf.yield %arg27 : i64
                    }
                    %161:7 = scf.for %arg29 = %c0_i32 to %87 step %c1_i32 iter_args(%arg30 = %126, %arg31 = %arg23, %arg32 = %130, %arg33 = %132, %arg34 = %arg26, %arg35 = %160, %arg36 = %arg28) -> (i64, i64, i64, i64, i64, i64, i64)  : i32 {
                      %162 = arith.addi %84, %arg29 : i32
                      %163 = arith.addi %162, %c1_i32 : i32
                      %164 = arith.extsi %163 : i32 to i64
                      %165 = arith.cmpi eq, %164, %95 : i64
                      %166 = arith.muli %162, %c64_i32 : i32
                      %167 = arith.extsi %166 : i32 to i64
                      %168 = arith.subi %81, %167 : i64
                      %169 = arith.select %165, %168, %c64_i64 : i64
                      %170 = arith.addi %162, %c2_i32 : i32
                      %171 = arith.extsi %170 : i32 to i64
                      %172 = arith.cmpi eq, %171, %95 : i64
                      %173 = arith.muli %163, %c64_i32 : i32
                      %174 = arith.extsi %173 : i32 to i64
                      %175 = arith.subi %81, %174 : i64
                      %176 = arith.select %172, %175, %c64_i64 : i64
                      %177 = arith.cmpi eq, %arg31, %c0_i64 : i64
                      %178 = ascscalar.memref_address %alloc_18 : memref<16384xbf16, 2>
                      %179 = ascscalar.memref_address %alloc_17 : memref<16384xbf16, 2>
                      %180 = arith.addi %arg31, %c1_i64 : i64
                      %181 = arith.remui %180, %c2_i64 : i64
                      %182 = arith.addi %arg34, %c1_i64 : i64
                      %183 = arith.remui %182, %c2_i64 : i64
                      %184 = arith.addi %183, %c1_i64 overflow<nsw> : i64
                      %185 = arith.remui %184, %c2_i64 : i64
                      %186 = arith.cmpi ne, %arg31, %c0_i64 : i64
                      %187 = arith.extui %186 : i1 to i64
                      %188 = arith.addi %arg29, %c1_i32 : i32
                      %189 = arith.cmpi slt, %188, %87 : i32
                      %190:2 = scf.if %189 -> (i64, i64) {
                        %216 = arith.cmpi eq, %arg30, %c0_i64 : i64
                        %217 = arith.cmpi ne, %arg30, %c0_i64 : i64
                        %218 = arith.extui %217 : i1 to i64
                        %219 = arith.addi %arg30, %c1_i64 overflow<nsw> : i64
                        %220 = arith.remui %219, %c2_i64 : i64
                        %221 = arith.select %216, %133, %134 : i64
                        %222 = arith.minsi %164, %138 : i64
                        %223 = arith.subi %138, %222 : i64
                        %224 = arith.minsi %223, %c1_i64 : i64
                        %225 = arith.cmpi sle, %164, %138 : i64
                        %226 = arith.select %225, %224, %c0_i64 : i64
                        %227 = arith.muli %164, %c512_i64 : i64
                        %228 = arith.addi %152, %227 : i64
                        %229 = arith.addi %228, %155 : i64
                        %230 = ascscalar.bind_local_ptr %221 : memref<1xf32, 1>
                        ascvec.get_buf mte2, %218, 0 {cannir.buffer.generated_sync}
                        "ascvec.copy_gm2ub_plan"(%230, %alloc, %c0_i32, %c0_i32, %c0_i64, %229, %226, %c256_i64, %c512_i64, %c256_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = -1 : i64, ub_capacity = 4 : i64}> : (memref<1xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                        ascvec.release_buf mte2, %218, 0 {cannir.buffer.generated_sync}
                        %231 = arith.cmpi slt, %158, %176 : i64
                        %232 = scf.if %231 -> (i64) {
                          %233 = arith.subi %176, %158 : i64
                          %234 = arith.cmpi sgt, %233, %c32_i64 : i64
                          %235 = arith.select %234, %c32_i64, %233 : i64
                          %236 = arith.cmpi eq, %arg35, %c0_i64 : i64
                          %237 = ascscalar.memref_address %alloc_15 : memref<4096xbf16, 1>
                          %238 = ascscalar.memref_address %alloc_16 : memref<4096xbf16, 1>
                          %239 = arith.select %236, %237, %238 : i64
                          %240 = arith.select %236, %c6_i64, %c7_i64 : i64
                          %241 = arith.addi %arg35, %c1_i64 : i64
                          %242 = arith.remui %241, %c2_i64 : i64
                          %243 = cannir.extract_physical_layout %arg4 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                          %244 = cannir.extract_stride %243 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                          %245 = cannir.extract_dim %244[2] : !cannir.stride<(-1, -1, -1, 1)>
                          %246 = arith.muli %245, %c2_i64 : i64
                          %247 = cannir.extract_dim %244[0] : !cannir.stride<(-1, -1, -1, 1)>
                          %248 = arith.muli %97, %247 : i64
                          %249 = arith.muli %248, %c2_i64 : i64
                          %250 = cannir.extract_dim %244[1] : !cannir.stride<(-1, -1, -1, 1)>
                          %251 = arith.muli %111, %250 : i64
                          %252 = arith.muli %251, %c2_i64 : i64
                          %253 = arith.addi %249, %252 : i64
                          %254 = arith.muli %113, %c32_i64 overflow<nsw> : i64
                          %255 = arith.addi %253, %254 : i64
                          %256 = arith.muli %164, %c64_i64 : i64
                          %257 = arith.addi %158, %256 : i64
                          %258 = arith.addi %257, %96 : i64
                          %259 = arith.muli %258, %245 : i64
                          %260 = arith.muli %259, %c2_i64 : i64
                          %261 = arith.addi %255, %260 : i64
                          %262 = ascscalar.bind_local_ptr %239 : memref<1xbf16, 1>
                          ascvec.get_buf mte2, %240, 0 {cannir.buffer.generated_sync}
                          "ascvec.copy_gm2ub_plan"(%262, %alloc_0, %c0_i32, %c0_i32, %c0_i64, %261, %235, %c32_i64, %246, %c32_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                          ascvec.release_buf mte2, %240, 0 {cannir.buffer.generated_sync}
                          scf.yield %242 : i64
                        } else {
                          scf.yield %arg35 : i64
                        }
                        scf.yield %220, %232 : i64, i64
                      } else {
                        scf.yield %arg30, %arg35 : i64, i64
                      }
                      %191 = ascscalar.memref_address %alloc_20 : memref<8192xbf16, 2>
                      %192 = ascscalar.memref_address %alloc_19 : memref<8192xbf16, 2>
                      %193 = arith.select %177, %191, %192 : i64
                      %194 = arith.select %177, %c10_i64, %c11_i64 : i64
                      %195 = arith.subi %169, %158 : i64
                      %196 = arith.cmpi slt, %195, %c0_i64 : i64
                      %197 = arith.select %196, %c0_i64, %195 : i64
                      %198 = arith.cmpi sgt, %197, %c32_i64 : i64
                      %199 = arith.select %198, %c32_i64, %197 : i64
                      %200 = arith.cmpi eq, %199, %c32_i64 : i64
                      ascvec.sync_intra_wait v, %c5_i64 {cannir.sync.banks = 3 : i32}
                      %201 = scf.if %200 -> (i64) {
                        %216 = arith.cmpi eq, %arg36, %c0_i64 : i64
                        %217 = ascscalar.memref_address %alloc_15 : memref<4096xbf16, 1>
                        %218 = ascscalar.memref_address %alloc_16 : memref<4096xbf16, 1>
                        %219 = arith.select %216, %217, %218 : i64
                        %220 = ascscalar.bind_local_ptr %219 : memref<4096xbf16, 1>
                        %221 = arith.select %216, %c6_i64, %c7_i64 : i64
                        %222 = arith.addi %arg36, %c1_i64 : i64
                        %223 = arith.remui %222, %c2_i64 : i64
                        ascvec.get_buf v, %221, 0 {cannir.buffer.generated_sync}
                        ascvec.vf(%220, %alloc_7, %alloc_9 : memref<4096xbf16, 1>, memref<4096xf32, 1>, memref<4096xbf16, 1>) {
                        ^bb0(%arg37: memref<4096xbf16, 1>, %arg38: memref<4096xf32, 1>, %arg39: memref<4096xbf16, 1>):
                          %c128_i32_21 = arith.constant 128 : i32
                          %c512_i32 = arith.constant 512 : i32
                          %c0_i32_22 = arith.constant 0 : i32
                          %c64_i32_23 = arith.constant 64 : i32
                          %mask, %remaining_out = ascvec.update_mask %c64_i32_23 : !ascvec.mask
                          scf.for %arg40 = %c0_i32_22 to %c512_i32 step %c128_i32_21  : i32 {
                            %224 = arith.index_cast %arg40 : i32 to index
                            %225 = ascvec.loadalign_unpack %arg37[%224] {mode = "b16_to_b32"} : memref<4096xbf16, 1>, index -> <bf16>
                            %226 = arith.addi %arg40, %c64_i32_23 overflow<nsw> : i32
                            %227 = arith.index_cast %226 : i32 to index
                            %228 = ascvec.loadalign_unpack %arg37[%227] {mode = "b16_to_b32"} : memref<4096xbf16, 1>, index -> <bf16>
                            %229 = ascvec.cast %225, %mask {rounding = 6 : i32} : <bf16> -> <f32>
                            %230 = ascvec.cast %228, %mask {rounding = 6 : i32} : <bf16> -> <f32>
                            %231 = ascvec.loadalign %arg38[%224] : memref<4096xf32, 1>, index -> <f32>
                            %232 = ascvec.loadalign %arg38[%227] : memref<4096xf32, 1>, index -> <f32>
                            %233 = ascvec.add %229, %231, %mask : <f32>
                            %234 = ascvec.add %230, %232, %mask : <f32>
                            %235 = ascvec.cast %233, %mask : <f32> -> <bf16>
                            %236 = ascvec.cast %234, %mask : <f32> -> <bf16>
                            ascvec.storealign_pack %arg39[%224], %235, %mask {mode = "b32_to_b16"} : memref<4096xbf16, 1>, index, <bf16>
                            ascvec.storealign_pack %arg39[%227], %236, %mask {mode = "b32_to_b16"} : memref<4096xbf16, 1>, index, <bf16>
                          }
                          ascvec.mem_bar vst_vld
                        } {cannir.vf_mode = "simd"}
                        ascvec.release_buf v, %221, 0 {cannir.buffer.generated_sync}
                        scf.yield %223 : i64
                      } else {
                        ascvec.vf(%alloc_9 : memref<4096xbf16, 1>) {
                        ^bb0(%arg37: memref<4096xbf16, 1>):
                          %c1_i32_21 = arith.constant 1 : i32
                          %c32_i32_22 = arith.constant 32 : i32
                          %c0_i32_23 = arith.constant 0 : i32
                          %cst = arith.constant 0.000000e+00 : bf16
                          %c16_i32_24 = arith.constant 16 : i32
                          %mask, %remaining_out = ascvec.update_mask %c16_i32_24 elem_bits 16 : !ascvec.mask
                          %216 = ascvec.duplicate_scalar %cst, %mask : bf16 -> <bf16>
                          scf.for %arg38 = %c0_i32_23 to %c32_i32_22 step %c1_i32_21  : i32 {
                            %217 = arith.muli %arg38, %c16_i32_24 overflow<nsw> : i32
                            %218 = arith.index_cast %217 : i32 to index
                            ascvec.storealign %arg37[%218], %216, %mask : memref<4096xbf16, 1>, index, <bf16>
                          }
                          ascvec.mem_bar vst_vld
                        } {cannir.vf_mode = "simd"}
                        scf.yield %arg36 : i64
                      }
                      %202 = arith.cmpi sgt, %199, %c0_i64 : i64
                      %203 = arith.cmpi slt, %199, %c32_i64 : i64
                      %204 = arith.select %202, %203, %false : i1
                      %205 = scf.if %204 -> (i64) {
                        %216 = arith.cmpi eq, %201, %c0_i64 : i64
                        %217 = ascscalar.memref_address %alloc_15 : memref<4096xbf16, 1>
                        %218 = ascscalar.memref_address %alloc_16 : memref<4096xbf16, 1>
                        %219 = arith.select %216, %217, %218 : i64
                        %220 = ascscalar.bind_local_ptr %219 : memref<4096xbf16, 1>
                        %221 = arith.select %216, %c6_i64, %c7_i64 : i64
                        %222 = arith.addi %201, %c1_i64 : i64
                        %223 = arith.remui %222, %c2_i64 : i64
                        ascvec.get_buf v, %221, 0 {cannir.buffer.generated_sync}
                        ascvec.vf(%220, %alloc_7, %alloc_9, %199 : memref<4096xbf16, 1>, memref<4096xf32, 1>, memref<4096xbf16, 1>, i64) {
                        ^bb0(%arg37: memref<4096xbf16, 1>, %arg38: memref<4096xf32, 1>, %arg39: memref<4096xbf16, 1>, %arg40: i64):
                          %c1_i64_21 = arith.constant 1 : i64
                          %c0_i64_22 = arith.constant 0 : i64
                          %c16_i64_23 = arith.constant 16 : i64
                          %c16_i32_24 = arith.constant 16 : i32
                          scf.for %arg41 = %c0_i64_22 to %arg40 step %c1_i64_21  : i64 {
                            %mask, %remaining_out = ascvec.update_mask %c16_i32_24 : !ascvec.mask
                            %224 = arith.muli %arg41, %c16_i64_23 : i64
                            %225 = arith.index_cast %224 : i64 to index
                            %226 = ascvec.loadalign_unpack %arg37[%225] {mode = "b16_to_b32"} : memref<4096xbf16, 1>, index -> <bf16>
                            %227 = ascvec.cast %226, %mask {rounding = 6 : i32} : <bf16> -> <f32>
                            %228 = ascvec.loadalign %arg38[%225] : memref<4096xf32, 1>, index -> <f32>
                            %229 = ascvec.add %227, %228, %mask : <f32>
                            %230 = ascvec.cast %229, %mask : <f32> -> <bf16>
                            ascvec.storealign_pack %arg39[%225], %230, %mask {mode = "b32_to_b16"} : memref<4096xbf16, 1>, index, <bf16>
                          } {cannir.loop_unroll = 2 : i64}
                          ascvec.mem_bar vst_vld
                        } {cannir.vf_mode = "simd"}
                        ascvec.release_buf v, %221, 0 {cannir.buffer.generated_sync}
                        scf.yield %223 : i64
                      } else {
                        scf.yield %201 : i64
                      }
                      ascvec.sync_intra_arrive v, %c5_i64 {cannir.sync.banks = 3 : i32}
                      ascvec.get_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
                      ascvec.vf(%alloc_9, %alloc_12 : memref<4096xbf16, 1>, memref<4224xbf16, 1>) {
                      ^bb0(%arg37: memref<4096xbf16, 1>, %arg38: memref<4224xbf16, 1>):
                        %c1_i32_21 = arith.constant 1 : i32
                        %c32_i32_22 = arith.constant 32 : i32
                        %c0_i32_23 = arith.constant 0 : i32
                        %c33 = arith.constant 33 : index
                        %c0_24 = arith.constant 0 : index
                        %c16_i32_25 = arith.constant 16 : i32
                        scf.for %arg39 = %c0_i32_23 to %c32_i32_22 step %c1_i32_21  : i32 {
                          %mask, %remaining_out = ascvec.update_mask %c16_i32_25 elem_bits 16 : !ascvec.mask
                          %216 = arith.muli %arg39, %c16_i32_25 overflow<nsw> : i32
                          %217 = arith.index_cast %216 : i32 to index
                          %218 = ascvec.loadalign %arg37[%217] : memref<4096xbf16, 1>, index -> <bf16>
                          ascvec.storealign_strided %arg38[%217], %218, %mask, %c33, %c0_24 : memref<4224xbf16, 1>, <bf16>
                        }
                      } {cannir.vf_mode = "simd"}
                      ascvec.release_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
                      %206 = ascscalar.bitcast_memref %alloc_12 byte_offset = %c0_i64 : memref<4224xbf16, 1> -> memref<?xbf16, 1>
                      %207 = ascscalar.bind_local_ptr %193 : memref<1xbf16, 2>
                      %208 = arith.muli %98, %c512_i64 : i64
                      %209 = arith.cmpi eq, %194, %c11_i64 : i64
                      %210 = arith.select %209, %c3_i64, %c2_i64 : i64
                      ascvec.sync_intra_wait mte3, %210 {cannir.sync.banks = 3 : i32}
                      ascvec.get_buf mte3, %c3_i64, 0 {cannir.buffer.generated_sync}
                      %211 = arith.index_cast %208 : i64 to index
                      ascvec.copy_ub2l1_strided %207[%211], %206[%c0], %c1_i32, %c32_i32, %c1_i32, %c32_i32 : memref<1xbf16, 2>, memref<?xbf16, 1>
                      ascvec.release_buf mte3, %c3_i64, 0 {cannir.buffer.generated_sync}
                      ascvec.sync_intra_arrive mte3, %210 {cannir.sync.banks = 3 : i32}
                      %212 = scf.if %189 -> (i64) {
                        %216 = arith.addi %arg32, %c1_i64 overflow<nsw> : i64
                        %217 = arith.remui %216, %c2_i64 : i64
                        scf.yield %217 : i64
                      } else {
                        scf.yield %arg32 : i64
                      }
                      %213 = arith.select %177, %133, %134 : i64
                      %214 = ascscalar.bind_local_ptr %213 : memref<64xf32, 1>
                      ascvec.sync_intra_wait v, %c4_i64 {cannir.sync.banks = 3 : i32}
                      ascvec.get_buf v, %187, 0 {cannir.buffer.generated_sync}
                      ascvec.vf(%214, %alloc_8, %alloc_6 : memref<64xf32, 1>, memref<8192xf32, 1>, memref<8192xf32, 1>) {
                      ^bb0(%arg37: memref<64xf32, 1>, %arg38: memref<8192xf32, 1>, %arg39: memref<8192xf32, 1>):
                        %c1024_i32 = arith.constant 1024 : i32
                        %c0_i32_21 = arith.constant 0 : i32
                        %c0_22 = arith.constant 0 : index
                        %c64_i32_23 = arith.constant 64 : i32
                        %mask, %remaining_out = ascvec.update_mask %c64_i32_23 : !ascvec.mask
                        %216 = ascvec.loadalign %arg37[%c0_22] : memref<64xf32, 1>, index -> <f32>
                        scf.for %arg40 = %c0_i32_21 to %c1024_i32 step %c64_i32_23  : i32 {
                          %217 = arith.index_cast %arg40 : i32 to index
                          %218 = ascvec.loadalign %arg38[%217] : memref<8192xf32, 1>, index -> <f32>
                          %219 = ascvec.loadalign %arg39[%217] : memref<8192xf32, 1>, index -> <f32>
                          %220 = ascvec.mul %218, %216, %mask : <f32>
                          %221 = ascvec.add %220, %219, %mask : <f32>
                          ascvec.storealign %arg38[%217], %221, %mask : memref<8192xf32, 1>, index, <f32>
                        } {cannir.loop_unroll = 4 : i64}
                      } {cannir.vf_mode = "simd"}
                      ascvec.release_buf v, %187, 0 {cannir.buffer.generated_sync}
                      ascvec.sync_intra_arrive v, %c4_i64 {cannir.sync.banks = 3 : i32}
                      %215 = scf.if %189 -> (i64) {
                        ascvec.vf(%alloc_8, %alloc_10 : memref<8192xf32, 1>, memref<8192xbf16, 1>) {
                        ^bb0(%arg37: memref<8192xf32, 1>, %arg38: memref<8192xbf16, 1>):
                          %c1_i32_21 = arith.constant 1 : i32
                          %c16_i32_22 = arith.constant 16 : i32
                          %c0_i32_23 = arith.constant 0 : i32
                          %c64_i32_24 = arith.constant 64 : i32
                          scf.for %arg39 = %c0_i32_23 to %c16_i32_22 step %c1_i32_21  : i32 {
                            %mask, %remaining_out = ascvec.update_mask %c64_i32_24 : !ascvec.mask
                            %227 = arith.muli %arg39, %c64_i32_24 overflow<nsw> : i32
                            %228 = arith.index_cast %227 : i32 to index
                            %229 = ascvec.loadalign %arg37[%228] : memref<8192xf32, 1>, index -> <f32>
                            %230 = ascvec.cast %229, %mask : <f32> -> <bf16>
                            ascvec.storealign_pack %arg38[%228], %230, %mask {mode = "b32_to_b16"} : memref<8192xbf16, 1>, index, <bf16>
                          }
                        } {cannir.vf_mode = "simd"}
                        ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                        ascvec.vf(%alloc_10, %alloc_11 : memref<8192xbf16, 1>, memref<8256xbf16, 1>) {
                        ^bb0(%arg37: memref<8192xbf16, 1>, %arg38: memref<8256xbf16, 1>):
                          %c1_i32_21 = arith.constant 1 : i32
                          %c0_i32_22 = arith.constant 0 : i32
                          %c17 = arith.constant 17 : index
                          %c16_i32_23 = arith.constant 16 : i32
                          %c0_24 = arith.constant 0 : index
                          %c64_i32_25 = arith.constant 64 : i32
                          scf.for %arg39 = %c0_i32_22 to %c16_i32_23 step %c1_i32_21  : i32 {
                            %mask, %remaining_out = ascvec.update_mask %c64_i32_25 elem_bits 16 : !ascvec.mask
                            %227 = arith.muli %arg39, %c64_i32_25 overflow<nsw> : i32
                            %228 = arith.index_cast %227 : i32 to index
                            %229 = ascvec.loadalign %arg37[%228] : memref<8192xbf16, 1>, index -> <bf16>
                            %230 = arith.muli %arg39, %c16_i32_23 overflow<nsw> : i32
                            %231 = arith.index_cast %230 : i32 to index
                            ascvec.storealign_strided %arg38[%231], %229, %mask, %c17, %c0_24 : memref<8256xbf16, 1>, <bf16>
                          }
                        } {cannir.vf_mode = "simd"}
                        ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                        %216 = arith.cmpi eq, %arg33, %c0_i64 : i64
                        %217 = arith.select %216, %178, %179 : i64
                        %218 = arith.select %216, %c8_i64, %c9_i64 : i64
                        %219 = arith.addi %arg33, %c1_i64 overflow<nsw> : i64
                        %220 = arith.remui %219, %c2_i64 : i64
                        %221 = ascscalar.bitcast_memref %alloc_11 byte_offset = %c0_i64 : memref<8256xbf16, 1> -> memref<?xbf16, 1>
                        %222 = ascscalar.bind_local_ptr %217 : memref<1xbf16, 2>
                        %223 = arith.muli %98, %c1024_i64 : i64
                        %224 = arith.cmpi eq, %218, %c9_i64 : i64
                        %225 = arith.extui %224 : i1 to i64
                        ascvec.sync_intra_wait mte3, %225 {cannir.sync.banks = 3 : i32}
                        ascvec.get_buf mte3, %c2_i64, 0 {cannir.buffer.generated_sync}
                        %226 = arith.index_cast %223 : i64 to index
                        ascvec.copy_ub2l1_strided %222[%226], %221[%c0], %c4_i32, %c16_i32, %c1_i32, %c0_i32 : memref<1xbf16, 2>, memref<?xbf16, 1>
                        ascvec.release_buf mte3, %c2_i64, 0 {cannir.buffer.generated_sync}
                        ascvec.sync_intra_arrive mte3, %225 {cannir.sync.banks = 3 : i32}
                        scf.yield %220 : i64
                      } else {
                        ascvec.get_buf v, %c5_i64, 0 {cannir.buffer.generated_sync}
                        ascvec.copy_ub2ub %alloc_14[%c0], %alloc_8[%c0], %c4096_i32 : memref<8192xf32, 1>, memref<8192xf32, 1>
                        ascvec.vf(%alloc_14, %alloc_8 : memref<8192xf32, 1>, memref<8192xf32, 1>) {
                        ^bb0(%arg37: memref<8192xf32, 1>, %arg38: memref<8192xf32, 1>):
                        } {cannir.vf_mode = "simd"}
                        ascvec.release_buf v, %c5_i64, 0 {cannir.buffer.generated_sync}
                        %216 = cannir.extract_physical_layout %arg8 : !cannir.tensor<<gm, f32>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, identity>
                        %217 = cannir.extract_stride %216 : <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>> -> <(-1, 16384, 128, 1)>
                        %218 = cannir.extract_dim %217[0] : !cannir.stride<(-1, 16384, 128, 1)>
                        %219 = arith.muli %114, %218 : i64
                        %220 = arith.muli %111, %c16384_i64 : i64
                        %221 = arith.addi %219, %220 : i64
                        %222 = arith.muli %113, %c2048_i64 overflow<nsw> : i64
                        %223 = arith.addi %221, %222 : i64
                        %224 = arith.muli %98, %c64_i64 : i64
                        %225 = arith.addi %223, %224 : i64
                        %226 = arith.muli %225, %c32_i64 : i64
                        %227 = ascscalar.memref_address %alloc_2 : memref<1xf32>
                        %228 = arith.divui %226, %c8_i64 : i64
                        %229 = arith.addi %227, %228 : i64
                        %230 = ascscalar.bind_gm_ptr %229 : memref<1xf32>
                        %231 = ascscalar.bitcast_memref %alloc_14 byte_offset = %c0_i64 : memref<8192xf32, 1> -> memref<?xf32, 1>
                        ascvec.get_buf mte3, %c5_i64, 0 {cannir.buffer.generated_sync}
                        ascvec.copy_ub2gm %230[%c0], %231[%c0], %c16_i32, %c256_i32, %c512_i64, %c256_i32 l2_cache_ctl %c0_i32 : memref<1xf32>, memref<?xf32, 1>
                        ascvec.release_buf mte3, %c5_i64, 0 {cannir.buffer.generated_sync}
                        scf.yield %arg33 : i64
                      }
                      scf.yield %190#0, %181, %212, %215, %185, %190#1, %205 : i64, i64, i64, i64, i64, i64, i64
                    }
                    scf.yield %161#0, %161#1, %161#2, %161#3, %161#4, %161#5, %161#6 : i64, i64, i64, i64, i64, i64, i64
                  }
                  scf.yield %101#0, %101#1, %101#2, %101#3, %101#4, %101#5, %101#6 : i64, i64, i64, i64, i64, i64, i64
                } else {
                  scf.yield %arg14, %arg15, %arg16, %arg17, %arg18, %arg19, %arg20 : i64, i64, i64, i64, i64, i64, i64
                }
                scf.yield %100#0, %100#1, %100#2, %100#3, %100#4, %100#5, %100#6 : i64, i64, i64, i64, i64, i64, i64
              }
            }
          }
        }
        %47 = arith.addi %arg12, %c1_i32 : i32
        %48 = arith.cmpi slt, %47, %15 : i32
        scf.if %48 {
          ascvec.sync_all
          ascvec.sync_block_arrive mte3, %c1_i64
          ascvec.sync_block_wait s, %c9_i64
        }
      }
      ascvec.sync_intra_wait mte3, %c0_i64
      ascvec.sync_intra_wait mte3, %c1_i64
      ascvec.sync_intra_wait mte3, %c2_i64
      ascvec.sync_intra_wait mte3, %c3_i64
    }
    return
  }
  func.func @kernel_FlashKdaCpKernel___call___0_aic(%arg0: !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>, %arg1: !cannir.tensor<<gm, f32>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>, %arg2: !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>, %arg3: !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>, %arg4: !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>, %arg5: !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 64, 64)>, <(-1, -1, 4096, 64, 1)>>, <<(-1, -1, -1, 64, 64)>, <(-1, -1, 4096, 64, 1)>>, identity>, %arg6: !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>, %arg7: !cannir.tensor<<gm, f32>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, identity>, %arg8: !cannir.tensor<<gm, f32>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, identity>, %arg9: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg10: i64, %arg11: i64) attributes {cannir.buffer.auto_sync, cannir.buffer.sync_analyzed, cannir.buffer.sync_verified, cannir.buffer.tensor_state, cannir.core = "aic", cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.device, cannir.explicit_ring_buffer_cursors, cannir.kernel_group = "kernel_FlashKdaCpKernel___call___0", cannir.resource.ids_prepared} {
    %c8_i32 = arith.constant 8 : i32
    %c4 = arith.constant 4 : index
    %c6 = arith.constant 6 : index
    %c5 = arith.constant 5 : index
    %c3 = arith.constant 3 : index
    %c0_i32 = arith.constant 0 : i32
    %false = arith.constant false
    %c1_i32 = arith.constant 1 : i32
    %c9_i32 = arith.constant 9 : i32
    %c2_i32 = arith.constant 2 : i32
    %c64_i32 = arith.constant 64 : i32
    %c63_i64 = arith.constant 63 : i64
    %c64_i64 = arith.constant 64 : i64
    %c128_i64 = arith.constant 128 : i64
    %c128_i32 = arith.constant 128 : i32
    %c16_i32 = arith.constant 16 : i32
    %c0 = arith.constant 0 : index
    %c32_i32 = arith.constant 32 : i32
    %c4_i32 = arith.constant 4 : i32
    %c2 = arith.constant 2 : index
    %c4096_i64 = arith.constant 4096 : i64
    %c65_i32 = arith.constant 65 : i32
    %c1024_i64 = arith.constant 1024 : i64
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
    %alloc_0 = memref.alloc() {cannir.arg_index = 2 : i64} : memref<1xbf16>
    %alloc_1 = memref.alloc() {cannir.arg_index = 3 : i64} : memref<1xbf16>
    %alloc_2 = memref.alloc() {cannir.arg_index = 5 : i64} : memref<1xbf16>
    %alloc_3 = memref.alloc() {cannir.arg_index = 6 : i64} : memref<1xbf16>
    %alloc_4 = memref.alloc() {cannir.arg_index = 9 : i64} : memref<1xi32>
    %alloc_5 = memref.alloc() {cannir.byte_offset = 512 : i64} : memref<8192xf32, 1>
    %alloc_6 = memref.alloc() {cannir.byte_offset = 33280 : i64} : memref<4096xf32, 1>
    %alloc_7 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<8192xf32, 5>
    %alloc_8 = memref.alloc() {cannir.byte_offset = 16384 : i64} : memref<16384xbf16, 4>
    %alloc_9 = memref.alloc() {cannir.byte_offset = 32768 : i64} : memref<8192xf32, 5>
    %alloc_10 = memref.alloc() {cannir.byte_offset = 65536 : i64} : memref<16384xf32, 5>
    %alloc_11 = memref.alloc() {cannir.byte_offset = 32768 : i64} : memref<8192xbf16, 3>
    %alloc_12 = memref.alloc() {cannir.byte_offset = 49152 : i64} : memref<8192xbf16, 4>
    %alloc_13 = memref.alloc() {cannir.byte_offset = 49152 : i64} : memref<4096xbf16, 3>
    %alloc_14 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<8192xbf16, 4>
    %alloc_15 = memref.alloc() {cannir.byte_offset = 147456 : i64} : memref<16384xbf16, 2>
    %alloc_16 = memref.alloc() {cannir.byte_offset = 114688 : i64} : memref<16384xbf16, 2>
    %alloc_17 = memref.alloc() {cannir.byte_offset = 106496 : i64} : memref<4096xbf16, 2>
    %alloc_18 = memref.alloc() {cannir.byte_offset = 98304 : i64} : memref<4096xbf16, 2>
    %alloc_19 = memref.alloc() {cannir.byte_offset = 81920 : i64} : memref<8192xbf16, 2>
    %alloc_20 = memref.alloc() {cannir.byte_offset = 65536 : i64} : memref<8192xbf16, 2>
    %alloc_21 = memref.alloc() {cannir.byte_offset = 196608 : i64} : memref<8192xbf16, 2>
    %alloc_22 = memref.alloc() {cannir.byte_offset = 180224 : i64} : memref<8192xbf16, 2>
    %alloc_23 = memref.alloc() {cannir.byte_offset = 49152 : i64} : memref<8192xbf16, 2>
    %alloc_24 = memref.alloc() {cannir.byte_offset = 32768 : i64} : memref<8192xbf16, 2>
    %alloc_25 = memref.alloc() {cannir.byte_offset = 16384 : i64} : memref<8192xbf16, 2>
    %alloc_26 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<8192xbf16, 2>
    %alloc_27 = memref.alloc() {cannir.byte_offset = 16384 : i64} : memref<8192xbf16, 3>
    %alloc_28 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<8192xbf16, 3>
    %0 = ascscalar.get_block_idx
    %1 = ascscalar.get_block_num
    %2 = memref.load %alloc_4[%c2] : memref<1xi32>
    %3 = memref.load %alloc_4[%c3] : memref<1xi32>
    %4 = memref.load %alloc_4[%c5] : memref<1xi32>
    %5 = arith.extsi %4 : i32 to i64
    %6 = memref.load %alloc_4[%c6] : memref<1xi32>
    %7 = arith.extsi %6 : i32 to i64
    %8 = arith.cmpi eq, %2, %c0_i32 : i32
    %9 = arith.extsi %3 : i32 to i64
    %10 = arith.cmpi eq, %9, %1 : i64
    %11 = arith.muli %5, %7 : i64
    %12 = arith.cmpi eq, %11, %arg10 : i64
    %13 = arith.select %10, %12, %false : i1
    %14 = arith.select %8, %13, %false : i1
    scf.if %14 {
      asccube.sync_intra_arrive mte1, %c3_i64
      asccube.sync_intra_arrive mte1, %c19_i64
      asccube.sync_intra_arrive mte1, %c2_i64
      asccube.sync_intra_arrive mte1, %c18_i64
      asccube.sync_intra_arrive mte1, %c1_i64
      asccube.sync_intra_arrive mte1, %c17_i64
      asccube.sync_intra_arrive mte1, %c0_i64
      asccube.sync_intra_arrive mte1, %c16_i64
      %15 = memref.load %alloc_4[%c4] : memref<1xi32>
      scf.for %arg12 = %c0_i32 to %15 step %c1_i32  : i32 {
        %16 = arith.addi %arg12, %c9_i32 : i32
        %17 = arith.index_cast %16 : i32 to index
        %18 = memref.load %alloc_4[%17] : memref<1xi32>
        %19 = arith.addi %18, %c1_i32 : i32
        %20 = arith.index_cast %19 : i32 to index
        %21 = memref.load %alloc_4[%20] : memref<1xi32>
        %22 = arith.addi %18, %c2_i32 : i32
        %23 = arith.addi %22, %21 : i32
        %24 = arith.addi %23, %21 : i32
        %25 = arith.addi %24, %21 : i32
        %26 = arith.addi %25, %21 : i32
        %27 = arith.addi %26, %c1_i32 : i32
        %28 = arith.addi %27, %21 : i32
        %29 = arith.addi %28, %21 : i32
        %30 = arith.addi %29, %21 : i32
        %31 = arith.addi %30, %c65_i32 : i32
        %32 = arith.addi %31, %21 : i32
        %33 = arith.addi %32, %21 : i32
        %34 = arith.addi %33, %c1_i32 : i32
        %35 = arith.muli %0, %c2_i64 : i64
        %36 = arith.extsi %34 : i32 to i64
        %37 = arith.addi %36, %35 : i64
        %38 = arith.index_cast %37 : i64 to index
        %39 = memref.load %alloc_4[%38] : memref<1xi32>
        %40 = arith.addi %37, %c1_i64 : i64
        %41 = arith.index_cast %40 : i64 to index
        %42 = memref.load %alloc_4[%41] : memref<1xi32>
        %43 = arith.index_cast %31 : i32 to index
        %44 = memref.load %alloc_4[%43] : memref<1xi32>
        %45 = arith.cmpi eq, %44, %c1_i32 : i32
        scf.if %45 {
          %48:7 = scf.for %arg13 = %c0_i32 to %21 step %c1_i32 iter_args(%arg14 = %c0_i64, %arg15 = %c0_i64, %arg16 = %c0_i64, %arg17 = %c0_i64, %arg18 = %c0_i64, %arg19 = %c0_i64, %arg20 = %c0_i64) -> (i64, i64, i64, i64, i64, i64, i64)  : i32 {
            %49 = arith.addi %32, %arg13 : i32
            %50 = arith.index_cast %49 : i32 to index
            %51 = memref.load %alloc_4[%50] : memref<1xi32>
            %52 = arith.addi %49, %c1_i32 : i32
            %53 = arith.index_cast %52 : i32 to index
            %54 = memref.load %alloc_4[%53] : memref<1xi32>
            %55 = arith.cmpi slt, %39, %51 : i32
            %56 = arith.select %55, %51, %39 : i32
            %57 = arith.cmpi sgt, %42, %54 : i32
            %58 = arith.select %57, %54, %42 : i32
            %59 = arith.cmpi slt, %56, %58 : i32
            %60 = arith.subi %56, %51 : i32
            %61 = arith.subi %58, %51 : i32
            %62 = arith.extsi %60 : i32 to i64
            %63 = arith.extsi %61 : i32 to i64
            %64 = arith.select %59, %62, %c0_i64 : i64
            %65 = arith.select %59, %63, %c0_i64 : i64
            %66 = arith.addi %22, %arg13 : i32
            %67 = arith.index_cast %66 : i32 to index
            %68 = memref.load %alloc_4[%67] : memref<1xi32>
            %69 = arith.addi %23, %arg13 : i32
            %70 = arith.index_cast %69 : i32 to index
            %71 = memref.load %alloc_4[%70] : memref<1xi32>
            %72 = arith.addi %24, %arg13 : i32
            %73 = arith.index_cast %72 : i32 to index
            %74 = memref.load %alloc_4[%73] : memref<1xi32>
            %75 = arith.addi %25, %arg13 : i32
            %76 = arith.index_cast %75 : i32 to index
            %77 = memref.load %alloc_4[%76] : memref<1xi32>
            %78 = arith.extsi %77 : i32 to i64
            %79 = arith.addi %27, %arg13 : i32
            %80 = arith.index_cast %79 : i32 to index
            %81 = memref.load %alloc_4[%80] : memref<1xi32>
            %82 = arith.addi %28, %arg13 : i32
            %83 = arith.index_cast %82 : i32 to index
            %84 = memref.load %alloc_4[%83] : memref<1xi32>
            %85 = arith.addi %78, %c63_i64 : i64
            %86 = arith.remsi %85, %c64_i64 : i64
            %87 = arith.cmpi ne, %86, %c0_i64 : i64
            %88 = arith.cmpi slt, %86, %c0_i64 : i64
            %89 = arith.andi %87, %88 : i1
            %90 = arith.divsi %85, %c64_i64 : i64
            %91 = arith.extui %89 : i1 to i64
            %92 = arith.subi %90, %91 : i64
            %93 = arith.extsi %74 : i32 to i64
            %94 = arith.extsi %71 : i32 to i64
            %95 = ascscalar.get_subblock_id
            %96 = arith.cmpi slt, %0, %1 : i64
            %97:7 = scf.if %96 -> (i64, i64, i64, i64, i64, i64, i64) {
              %98:7 = scf.for %arg21 = %64 to %65 step %c1_i64 iter_args(%arg22 = %arg14, %arg23 = %arg15, %arg24 = %arg16, %arg25 = %arg17, %arg26 = %arg18, %arg27 = %arg19, %arg28 = %arg20) -> (i64, i64, i64, i64, i64, i64, i64)  : i64 {
                %99 = arith.maxsi %7, %c1_i64 : i64
                %100 = arith.remsi %arg21, %99 : i64
                %101 = arith.cmpi slt, %100, %c0_i64 : i64
                %102 = arith.select %101, %99, %c0_i64 : i64
                %103 = arith.addi %100, %102 : i64
                %104 = arith.extsi %68 : i32 to i64
                %105 = arith.addi %81, %84 : i32
                %106 = arith.extsi %105 : i32 to i64
                %107 = arith.cmpi eq, %106, %92 : i64
                %108 = arith.muli %81, %c64_i32 : i32
                %109 = arith.extsi %108 : i32 to i64
                %110 = arith.subi %78, %109 : i64
                %111 = arith.minsi %110, %c64_i64 : i64
                %112 = arith.select %107, %111, %c64_i64 : i64
                %113 = arith.extsi %81 : i32 to i64
                %114 = arith.cmpi eq, %arg22, %c0_i64 : i64
                %115 = ascscalar.memref_address %alloc_24 : memref<8192xbf16, 2>
                %116 = ascscalar.memref_address %alloc_23 : memref<8192xbf16, 2>
                %117 = arith.select %114, %115, %116 : i64
                %118 = arith.select %114, %c2_i64, %c3_i64 : i64
                %119 = arith.addi %arg22, %c1_i64 : i64
                %120 = arith.remui %119, %c2_i64 : i64
                %121 = arith.cmpi slt, %112, %c64_i64 : i64
                scf.if %121 {
                  asccube.get_buf mte2, %118, 0 {cannir.buffer.generated_sync}
                  scf.for %arg29 = %c0_i64 to %c8_i64 step %c1_i64  : i64 {
                    %144 = ascscalar.bind_local_ptr %117 : memref<1xbf16, 2>
                    %145 = cannir.extract_physical_layout %arg3 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                    %146 = cannir.extract_stride %145 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                    %147 = cannir.extract_dim %146[0] : !cannir.stride<(-1, -1, -1, 1)>
                    %148 = arith.muli %94, %147 : i64
                    %149 = arith.index_cast %148 : i64 to index
                    %150 = cannir.extract_dim %146[1] : !cannir.stride<(-1, -1, -1, 1)>
                    %151 = arith.muli %103, %150 : i64
                    %152 = arith.index_cast %151 : i64 to index
                    %153 = arith.addi %149, %152 : index
                    %154 = arith.muli %113, %c64_i64 : i64
                    %155 = arith.addi %154, %93 : i64
                    %156 = cannir.extract_dim %146[2] : !cannir.stride<(-1, -1, -1, 1)>
                    %157 = arith.muli %155, %156 : i64
                    %158 = arith.index_cast %157 : i64 to index
                    %159 = arith.addi %153, %158 : index
                    %160 = arith.muli %arg29, %c16_i64 overflow<nsw> : i64
                    %161 = arith.index_cast %160 : i64 to index
                    %162 = arith.addi %159, %161 : index
                    %163 = cannir.extract_layout %arg3 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                    %164 = cannir.extract_stride %163 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                    %165 = cannir.extract_dim %164[2] : !cannir.stride<(-1, -1, -1, 1)>
                    %166 = arith.index_cast %165 : i64 to index
                    %167 = arith.muli %arg29, %c1024_i64 overflow<nsw> : i64
                    %168 = arith.muli %166, %c2 : index
                    %169 = arith.index_cast %168 : index to i64
                    %170 = arith.cmpi ugt, %112, %c0_i64 : i64
                    scf.if %170 {
                      %171 = arith.index_cast %167 : i64 to index
                      %172 = arith.trunci %112 : i64 to i32
                      asccube.copy_gm2l1 %144[%171], %alloc_1[%162], %172, %c16_i32, %169, %c1_i32, %c0_i64, %c0_i64, %c64_i32 l2_cache_ctl %c0_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
                    }
                  }
                  asccube.release_buf mte2, %118, 0 {cannir.buffer.generated_sync}
                } else {
                  %144 = ascscalar.bind_local_ptr %117 : memref<1xbf16, 2>
                  %145 = cannir.extract_physical_layout %arg3 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                  %146 = cannir.extract_stride %145 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                  %147 = cannir.extract_dim %146[0] : !cannir.stride<(-1, -1, -1, 1)>
                  %148 = arith.muli %94, %147 : i64
                  %149 = arith.index_cast %148 : i64 to index
                  %150 = cannir.extract_dim %146[1] : !cannir.stride<(-1, -1, -1, 1)>
                  %151 = arith.muli %103, %150 : i64
                  %152 = arith.index_cast %151 : i64 to index
                  %153 = arith.addi %149, %152 : index
                  %154 = arith.muli %113, %c64_i64 : i64
                  %155 = arith.addi %154, %93 : i64
                  %156 = cannir.extract_dim %146[2] : !cannir.stride<(-1, -1, -1, 1)>
                  %157 = arith.muli %155, %156 : i64
                  %158 = arith.index_cast %157 : i64 to index
                  %159 = arith.addi %153, %158 : index
                  %160 = arith.divsi %78, %c64_i64 : i64
                  %161 = arith.minsi %113, %160 : i64
                  %162 = arith.muli %161, %c64_i64 : i64
                  %163 = arith.subi %78, %162 : i64
                  %164 = arith.minsi %163, %c64_i64 : i64
                  %165 = arith.cmpi sle, %113, %160 : i64
                  %166 = arith.select %165, %164, %c0_i64 : i64
                  %167 = cannir.extract_layout %arg3 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                  %168 = cannir.extract_stride %167 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                  %169 = cannir.extract_dim %168[2] : !cannir.stride<(-1, -1, -1, 1)>
                  %170 = arith.index_cast %169 : i64 to index
                  %171 = arith.muli %170, %c2 : index
                  %172 = arith.index_cast %171 : index to i64
                  asccube.get_buf mte2, %118, 0 {cannir.buffer.generated_sync}
                  %173 = arith.cmpi ugt, %166, %c0_i64 : i64
                  scf.if %173 {
                    %174 = arith.trunci %166 : i64 to i32
                    asccube.copy_gm2l1 %144[%c0], %alloc_1[%159], %174, %c128_i32, %172, %c1_i32, %c0_i64, %c0_i64, %c64_i32 l2_cache_ctl %c0_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
                  }
                  asccube.release_buf mte2, %118, 0 {cannir.buffer.generated_sync}
                }
                %122 = ascscalar.memref_address %alloc_26 : memref<8192xbf16, 2>
                %123 = ascscalar.memref_address %alloc_25 : memref<8192xbf16, 2>
                %124 = arith.select %114, %122, %123 : i64
                %125 = arith.cmpi ne, %arg22, %c0_i64 : i64
                %126 = arith.extui %125 : i1 to i64
                scf.if %121 {
                  asccube.get_buf mte2, %126, 0 {cannir.buffer.generated_sync}
                  scf.for %arg29 = %c0_i64 to %c8_i64 step %c1_i64  : i64 {
                    %144 = ascscalar.bind_local_ptr %124 : memref<1xbf16, 2>
                    %145 = cannir.extract_physical_layout %arg2 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                    %146 = cannir.extract_stride %145 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                    %147 = cannir.extract_dim %146[0] : !cannir.stride<(-1, -1, -1, 1)>
                    %148 = arith.muli %94, %147 : i64
                    %149 = arith.index_cast %148 : i64 to index
                    %150 = cannir.extract_dim %146[1] : !cannir.stride<(-1, -1, -1, 1)>
                    %151 = arith.muli %103, %150 : i64
                    %152 = arith.index_cast %151 : i64 to index
                    %153 = arith.addi %149, %152 : index
                    %154 = arith.muli %113, %c64_i64 : i64
                    %155 = arith.addi %154, %93 : i64
                    %156 = cannir.extract_dim %146[2] : !cannir.stride<(-1, -1, -1, 1)>
                    %157 = arith.muli %155, %156 : i64
                    %158 = arith.index_cast %157 : i64 to index
                    %159 = arith.addi %153, %158 : index
                    %160 = arith.muli %arg29, %c16_i64 overflow<nsw> : i64
                    %161 = arith.index_cast %160 : i64 to index
                    %162 = arith.addi %159, %161 : index
                    %163 = cannir.extract_layout %arg2 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                    %164 = cannir.extract_stride %163 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                    %165 = cannir.extract_dim %164[2] : !cannir.stride<(-1, -1, -1, 1)>
                    %166 = arith.index_cast %165 : i64 to index
                    %167 = arith.muli %arg29, %c1024_i64 overflow<nsw> : i64
                    %168 = arith.muli %166, %c2 : index
                    %169 = arith.index_cast %168 : index to i64
                    %170 = arith.cmpi ugt, %112, %c0_i64 : i64
                    scf.if %170 {
                      %171 = arith.index_cast %167 : i64 to index
                      %172 = arith.trunci %112 : i64 to i32
                      asccube.copy_gm2l1 %144[%171], %alloc_0[%162], %172, %c16_i32, %169, %c1_i32, %c0_i64, %c0_i64, %c64_i32 l2_cache_ctl %c0_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
                    }
                  }
                  asccube.release_buf mte2, %126, 0 {cannir.buffer.generated_sync}
                } else {
                  %144 = ascscalar.bind_local_ptr %124 : memref<1xbf16, 2>
                  %145 = cannir.extract_physical_layout %arg2 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                  %146 = cannir.extract_stride %145 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                  %147 = cannir.extract_dim %146[0] : !cannir.stride<(-1, -1, -1, 1)>
                  %148 = arith.muli %94, %147 : i64
                  %149 = arith.index_cast %148 : i64 to index
                  %150 = cannir.extract_dim %146[1] : !cannir.stride<(-1, -1, -1, 1)>
                  %151 = arith.muli %103, %150 : i64
                  %152 = arith.index_cast %151 : i64 to index
                  %153 = arith.addi %149, %152 : index
                  %154 = arith.muli %113, %c64_i64 : i64
                  %155 = arith.addi %154, %93 : i64
                  %156 = cannir.extract_dim %146[2] : !cannir.stride<(-1, -1, -1, 1)>
                  %157 = arith.muli %155, %156 : i64
                  %158 = arith.index_cast %157 : i64 to index
                  %159 = arith.addi %153, %158 : index
                  %160 = arith.divsi %78, %c64_i64 : i64
                  %161 = arith.minsi %113, %160 : i64
                  %162 = arith.muli %161, %c64_i64 : i64
                  %163 = arith.subi %78, %162 : i64
                  %164 = arith.minsi %163, %c64_i64 : i64
                  %165 = arith.cmpi sle, %113, %160 : i64
                  %166 = arith.select %165, %164, %c0_i64 : i64
                  %167 = cannir.extract_layout %arg2 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                  %168 = cannir.extract_stride %167 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                  %169 = cannir.extract_dim %168[2] : !cannir.stride<(-1, -1, -1, 1)>
                  %170 = arith.index_cast %169 : i64 to index
                  %171 = arith.muli %170, %c2 : index
                  %172 = arith.index_cast %171 : index to i64
                  asccube.get_buf mte2, %126, 0 {cannir.buffer.generated_sync}
                  %173 = arith.cmpi ugt, %166, %c0_i64 : i64
                  scf.if %173 {
                    %174 = arith.trunci %166 : i64 to i32
                    asccube.copy_gm2l1 %144[%c0], %alloc_0[%159], %174, %c128_i32, %172, %c1_i32, %c0_i64, %c0_i64, %c64_i32 l2_cache_ctl %c0_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
                  }
                  asccube.release_buf mte2, %126, 0 {cannir.buffer.generated_sync}
                }
                %127 = arith.cmpi eq, %arg24, %c0_i64 : i64
                %128 = ascscalar.memref_address %alloc_20 : memref<8192xbf16, 2>
                %129 = ascscalar.memref_address %alloc_19 : memref<8192xbf16, 2>
                %130 = arith.select %127, %128, %129 : i64
                %131 = arith.select %127, %c4_i64, %c5_i64 : i64
                %132 = arith.addi %arg24, %c1_i64 : i64
                %133 = arith.remui %132, %c2_i64 : i64
                scf.if %121 {
                  asccube.get_buf mte2, %131, 0 {cannir.buffer.generated_sync}
                  scf.for %arg29 = %c0_i64 to %c8_i64 step %c1_i64  : i64 {
                    %144 = ascscalar.bind_local_ptr %130 : memref<1xbf16, 2>
                    %145 = cannir.extract_physical_layout %arg0 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                    %146 = cannir.extract_stride %145 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                    %147 = cannir.extract_dim %146[0] : !cannir.stride<(-1, -1, -1, 1)>
                    %148 = arith.muli %94, %147 : i64
                    %149 = arith.index_cast %148 : i64 to index
                    %150 = cannir.extract_dim %146[1] : !cannir.stride<(-1, -1, -1, 1)>
                    %151 = arith.muli %103, %150 : i64
                    %152 = arith.index_cast %151 : i64 to index
                    %153 = arith.addi %149, %152 : index
                    %154 = arith.muli %113, %c64_i64 : i64
                    %155 = arith.addi %154, %93 : i64
                    %156 = cannir.extract_dim %146[2] : !cannir.stride<(-1, -1, -1, 1)>
                    %157 = arith.muli %155, %156 : i64
                    %158 = arith.index_cast %157 : i64 to index
                    %159 = arith.addi %153, %158 : index
                    %160 = arith.muli %arg29, %c16_i64 overflow<nsw> : i64
                    %161 = arith.index_cast %160 : i64 to index
                    %162 = arith.addi %159, %161 : index
                    %163 = cannir.extract_layout %arg0 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                    %164 = cannir.extract_stride %163 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                    %165 = cannir.extract_dim %164[2] : !cannir.stride<(-1, -1, -1, 1)>
                    %166 = arith.index_cast %165 : i64 to index
                    %167 = arith.muli %arg29, %c1024_i64 overflow<nsw> : i64
                    %168 = arith.muli %166, %c2 : index
                    %169 = arith.index_cast %168 : index to i64
                    %170 = arith.cmpi ugt, %112, %c0_i64 : i64
                    scf.if %170 {
                      %171 = arith.index_cast %167 : i64 to index
                      %172 = arith.trunci %112 : i64 to i32
                      asccube.copy_gm2l1 %144[%171], %alloc[%162], %172, %c16_i32, %169, %c1_i32, %c0_i64, %c0_i64, %c64_i32 l2_cache_ctl %c1_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
                    }
                  }
                  asccube.release_buf mte2, %131, 0 {cannir.buffer.generated_sync}
                } else {
                  %144 = ascscalar.bind_local_ptr %130 : memref<1xbf16, 2>
                  %145 = cannir.extract_physical_layout %arg0 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                  %146 = cannir.extract_stride %145 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                  %147 = cannir.extract_dim %146[0] : !cannir.stride<(-1, -1, -1, 1)>
                  %148 = arith.muli %94, %147 : i64
                  %149 = arith.index_cast %148 : i64 to index
                  %150 = cannir.extract_dim %146[1] : !cannir.stride<(-1, -1, -1, 1)>
                  %151 = arith.muli %103, %150 : i64
                  %152 = arith.index_cast %151 : i64 to index
                  %153 = arith.addi %149, %152 : index
                  %154 = arith.muli %113, %c64_i64 : i64
                  %155 = arith.addi %154, %93 : i64
                  %156 = cannir.extract_dim %146[2] : !cannir.stride<(-1, -1, -1, 1)>
                  %157 = arith.muli %155, %156 : i64
                  %158 = arith.index_cast %157 : i64 to index
                  %159 = arith.addi %153, %158 : index
                  %160 = arith.divsi %78, %c64_i64 : i64
                  %161 = arith.minsi %113, %160 : i64
                  %162 = arith.muli %161, %c64_i64 : i64
                  %163 = arith.subi %78, %162 : i64
                  %164 = arith.minsi %163, %c64_i64 : i64
                  %165 = arith.cmpi sle, %113, %160 : i64
                  %166 = arith.select %165, %164, %c0_i64 : i64
                  %167 = cannir.extract_layout %arg0 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                  %168 = cannir.extract_stride %167 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                  %169 = cannir.extract_dim %168[2] : !cannir.stride<(-1, -1, -1, 1)>
                  %170 = arith.index_cast %169 : i64 to index
                  %171 = arith.muli %170, %c2 : index
                  %172 = arith.index_cast %171 : index to i64
                  asccube.get_buf mte2, %131, 0 {cannir.buffer.generated_sync}
                  %173 = arith.cmpi ugt, %166, %c0_i64 : i64
                  scf.if %173 {
                    %174 = arith.trunci %166 : i64 to i32
                    asccube.copy_gm2l1 %144[%c0], %alloc[%159], %174, %c128_i32, %172, %c1_i32, %c0_i64, %c0_i64, %c64_i32 l2_cache_ctl %c1_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
                  }
                  asccube.release_buf mte2, %131, 0 {cannir.buffer.generated_sync}
                }
                %134 = ascscalar.memref_address %alloc_18 : memref<4096xbf16, 2>
                %135 = ascscalar.memref_address %alloc_17 : memref<4096xbf16, 2>
                %136 = arith.select %127, %134, %135 : i64
                %137 = arith.select %127, %c6_i64, %c7_i64 : i64
                scf.if %121 {
                  asccube.get_buf mte2, %137, 0 {cannir.buffer.generated_sync}
                  scf.for %arg29 = %c0_i64 to %c4_i64 step %c1_i64  : i64 {
                    %144 = ascscalar.bind_local_ptr %136 : memref<1xbf16, 2>
                    %145 = cannir.extract_physical_layout %arg5 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 64, 64)>, <(-1, -1, 4096, 64, 1)>>, <<(-1, -1, -1, 64, 64)>, <(-1, -1, 4096, 64, 1)>>, identity>
                    %146 = cannir.extract_stride %145 : <<(-1, -1, -1, 64, 64)>, <(-1, -1, 4096, 64, 1)>> -> <(-1, -1, 4096, 64, 1)>
                    %147 = cannir.extract_dim %146[0] : !cannir.stride<(-1, -1, 4096, 64, 1)>
                    %148 = arith.muli %104, %147 : i64
                    %149 = arith.index_cast %148 : i64 to index
                    %150 = cannir.extract_dim %146[1] : !cannir.stride<(-1, -1, 4096, 64, 1)>
                    %151 = arith.muli %103, %150 : i64
                    %152 = arith.index_cast %151 : i64 to index
                    %153 = arith.addi %149, %152 : index
                    %154 = arith.muli %113, %c4096_i64 : i64
                    %155 = arith.index_cast %154 : i64 to index
                    %156 = arith.addi %153, %155 : index
                    %157 = arith.muli %arg29, %c16_i64 overflow<nsw> : i64
                    %158 = arith.index_cast %157 : i64 to index
                    %159 = arith.addi %156, %158 : index
                    %160 = arith.muli %arg29, %c1024_i64 overflow<nsw> : i64
                    %161 = arith.cmpi ugt, %112, %c0_i64 : i64
                    scf.if %161 {
                      %162 = arith.index_cast %160 : i64 to index
                      %163 = arith.trunci %112 : i64 to i32
                      asccube.copy_gm2l1 %144[%162], %alloc_2[%159], %163, %c16_i32, %c128_i64, %c1_i32, %c0_i64, %c0_i64, %c64_i32 l2_cache_ctl %c1_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
                    }
                  }
                  asccube.release_buf mte2, %137, 0 {cannir.buffer.generated_sync}
                } else {
                  %144 = ascscalar.bind_local_ptr %136 : memref<1xbf16, 2>
                  %145 = cannir.extract_physical_layout %arg5 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 64, 64)>, <(-1, -1, 4096, 64, 1)>>, <<(-1, -1, -1, 64, 64)>, <(-1, -1, 4096, 64, 1)>>, identity>
                  %146 = cannir.extract_stride %145 : <<(-1, -1, -1, 64, 64)>, <(-1, -1, 4096, 64, 1)>> -> <(-1, -1, 4096, 64, 1)>
                  %147 = cannir.extract_dim %146[0] : !cannir.stride<(-1, -1, 4096, 64, 1)>
                  %148 = arith.muli %104, %147 : i64
                  %149 = arith.index_cast %148 : i64 to index
                  %150 = cannir.extract_dim %146[1] : !cannir.stride<(-1, -1, 4096, 64, 1)>
                  %151 = arith.muli %103, %150 : i64
                  %152 = arith.index_cast %151 : i64 to index
                  %153 = arith.addi %149, %152 : index
                  %154 = arith.muli %113, %c4096_i64 : i64
                  %155 = arith.index_cast %154 : i64 to index
                  %156 = arith.addi %153, %155 : index
                  asccube.get_buf mte2, %137, 0 {cannir.buffer.generated_sync}
                  asccube.copy_gm2l1 %144[%c0], %alloc_2[%156], %c64_i32, %c64_i32, %c128_i64, %c1_i32, %c0_i64, %c0_i64, %c64_i32 l2_cache_ctl %c1_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
                  asccube.release_buf mte2, %137, 0 {cannir.buffer.generated_sync}
                }
                %138 = arith.cmpi eq, %81, %c0_i32 : i32
                %139 = scf.if %138 -> (i64) {
                  %144 = arith.addi %arg25, %c1_i64 : i64
                  %145 = arith.remui %144, %c2_i64 : i64
                  scf.yield %145 : i64
                } else {
                  %144 = arith.addi %arg25, %c1_i64 : i64
                  %145 = arith.remui %144, %c2_i64 : i64
                  scf.yield %145 : i64
                }
                %140 = arith.muli %95, %c32_i64 : i64
                %141 = arith.cmpi slt, %140, %112 : i64
                %142 = scf.if %141 -> (i64) {
                  %144 = arith.addi %arg27, %c1_i64 : i64
                  %145 = arith.remui %144, %c2_i64 : i64
                  scf.yield %145 : i64
                } else {
                  scf.yield %arg27 : i64
                }
                %143:7 = scf.for %arg29 = %c0_i32 to %84 step %c1_i32 iter_args(%arg30 = %120, %arg31 = %arg23, %arg32 = %133, %arg33 = %139, %arg34 = %arg26, %arg35 = %142, %arg36 = %arg28) -> (i64, i64, i64, i64, i64, i64, i64)  : i32 {
                  %144 = arith.addi %81, %arg29 : i32
                  %145 = arith.addi %144, %c1_i32 : i32
                  %146 = arith.extsi %145 : i32 to i64
                  %147 = arith.cmpi eq, %146, %92 : i64
                  %148 = arith.muli %144, %c64_i32 : i32
                  %149 = arith.extsi %148 : i32 to i64
                  %150 = arith.subi %78, %149 : i64
                  %151 = arith.select %147, %150, %c64_i64 : i64
                  %152 = arith.addi %144, %c2_i32 : i32
                  %153 = arith.extsi %152 : i32 to i64
                  %154 = arith.cmpi eq, %153, %92 : i64
                  %155 = arith.muli %145, %c64_i32 : i32
                  %156 = arith.extsi %155 : i32 to i64
                  %157 = arith.subi %78, %156 : i64
                  %158 = arith.select %154, %157, %c64_i64 : i64
                  %159 = arith.extsi %144 : i32 to i64
                  %160 = arith.cmpi eq, %arg31, %c0_i64 : i64
                  %161 = ascscalar.memref_address %alloc_16 : memref<16384xbf16, 2>
                  %162 = ascscalar.memref_address %alloc_15 : memref<16384xbf16, 2>
                  %163 = arith.select %160, %161, %162 : i64
                  %164 = arith.select %160, %c8_i64, %c9_i64 : i64
                  %165 = arith.addi %arg31, %c1_i64 : i64
                  %166 = arith.remui %165, %c2_i64 : i64
                  %167 = ascscalar.bitcast_memref %alloc_8 byte_offset = %c0_i64 : memref<16384xbf16, 4> -> memref<?xbf16, 4>
                  %168 = ascscalar.bind_local_ptr %163 : memref<1xbf16, 2>
                  %169 = arith.cmpi eq, %164, %c9_i64 : i64
                  %170 = arith.extui %169 : i1 to i64
                  asccube.sync_intra_wait mte1, %170 {cannir.sync.banks = 3 : i32}
                  %171 = arith.addi %170, %c16_i64 overflow<nsw> : i64
                  asccube.sync_intra_wait mte1, %171 {cannir.sync.banks = 3 : i32}
                  asccube.get_buf mte1, %c13_i64, 0 {cannir.buffer.generated_sync}
                  asccube.copy_l12l0b %167[%c0], %168[%c0], %c0_i32, %c0_i32, %c8_i32, %c8_i32, %c8_i32, %c8_i32 : memref<?xbf16, 4>, memref<1xbf16, 2>
                  asccube.release_buf mte1, %c13_i64, 0 {cannir.buffer.generated_sync}
                  asccube.sync_intra_arrive mte1, %170 {cannir.sync.banks = 3 : i32}
                  asccube.sync_intra_arrive mte1, %171 {cannir.sync.banks = 3 : i32}
                  %172 = arith.cmpi eq, %arg34, %c0_i64 : i64
                  %173 = ascscalar.memref_address %alloc_28 : memref<8192xbf16, 3>
                  %174 = ascscalar.memref_address %alloc_27 : memref<8192xbf16, 3>
                  %175 = arith.select %172, %173, %174 : i64
                  %176 = arith.select %172, %c8_i64, %c9_i64 : i64
                  %177 = arith.addi %arg34, %c1_i64 : i64
                  %178 = arith.remui %177, %c2_i64 : i64
                  %179 = arith.select %160, %115, %116 : i64
                  %180 = arith.select %160, %c2_i64, %c3_i64 : i64
                  %181 = ascscalar.bind_local_ptr %175 : memref<1xbf16, 3>
                  %182 = ascscalar.bind_local_ptr %179 : memref<1xbf16, 2>
                  asccube.get_buf mte1, %176, 0 {cannir.buffer.generated_sync}
                  asccube.get_buf mte1, %180, 0 {cannir.buffer.generated_sync}
                  asccube.copy_l12l0a %181[%c0], %182[%c0], %c0_i32, %c0_i32, %c4_i32, %c8_i32, %c4_i32, %c4_i32 : memref<1xbf16, 3>, memref<1xbf16, 2>
                  asccube.release_buf mte1, %180, 0 {cannir.buffer.generated_sync}
                  asccube.release_buf mte1, %176, 0 {cannir.buffer.generated_sync}
                  %183 = ascscalar.bitcast_memref %alloc_7 byte_offset = %c0_i64 : memref<8192xf32, 5> -> memref<?xf32, 5>
                  asccube.get_buf m, %c15_i64, 0 {cannir.buffer.generated_sync}
                  asccube.get_buf m, %176, 0 {cannir.buffer.generated_sync}
                  asccube.get_buf m, %c13_i64, 0 {cannir.buffer.generated_sync}
                  asccube.mmad %183[%c0], %181, %167, %c64_i32, %c128_i32, %c128_i32, %c0_i32 {c_matrix_init_val = true} : memref<?xf32, 5>, memref<1xbf16, 3>, memref<?xbf16, 4>
                  asccube.release_buf m, %176, 0 {cannir.buffer.generated_sync}
                  asccube.release_buf m, %c15_i64, 0 {cannir.buffer.generated_sync}
                  %184 = arith.cmpi eq, %178, %c0_i64 : i64
                  %185 = arith.select %184, %173, %174 : i64
                  %186 = arith.select %184, %c8_i64, %c9_i64 : i64
                  %187 = arith.addi %178, %c1_i64 overflow<nsw> : i64
                  %188 = arith.remui %187, %c2_i64 : i64
                  %189 = arith.select %160, %122, %123 : i64
                  %190 = arith.cmpi ne, %arg31, %c0_i64 : i64
                  %191 = arith.extui %190 : i1 to i64
                  %192 = ascscalar.bind_local_ptr %185 : memref<1xbf16, 3>
                  %193 = ascscalar.bind_local_ptr %189 : memref<1xbf16, 2>
                  asccube.get_buf mte1, %186, 0 {cannir.buffer.generated_sync}
                  asccube.get_buf mte1, %191, 0 {cannir.buffer.generated_sync}
                  asccube.copy_l12l0a %192[%c0], %193[%c0], %c0_i32, %c0_i32, %c4_i32, %c8_i32, %c4_i32, %c4_i32 : memref<1xbf16, 3>, memref<1xbf16, 2>
                  asccube.release_buf mte1, %191, 0 {cannir.buffer.generated_sync}
                  asccube.release_buf mte1, %186, 0 {cannir.buffer.generated_sync}
                  %194 = ascscalar.bitcast_memref %alloc_9 byte_offset = %c0_i64 : memref<8192xf32, 5> -> memref<?xf32, 5>
                  asccube.get_buf m, %c16_i64, 0 {cannir.buffer.generated_sync}
                  asccube.get_buf m, %186, 0 {cannir.buffer.generated_sync}
                  asccube.mmad %194[%c0], %192, %167, %c64_i32, %c128_i32, %c128_i32, %c0_i32 {c_matrix_init_val = true} : memref<?xf32, 5>, memref<1xbf16, 3>, memref<?xbf16, 4>
                  asccube.release_buf m, %c13_i64, 0 {cannir.buffer.generated_sync}
                  asccube.release_buf m, %186, 0 {cannir.buffer.generated_sync}
                  asccube.release_buf m, %c16_i64, 0 {cannir.buffer.generated_sync}
                  %195 = arith.select %160, %128, %129 : i64
                  %196 = arith.select %160, %c4_i64, %c5_i64 : i64
                  %197 = ascscalar.bitcast_memref %alloc_12 byte_offset = %c0_i64 : memref<8192xbf16, 4> -> memref<?xbf16, 4>
                  %198 = ascscalar.bind_local_ptr %195 : memref<1xbf16, 2>
                  asccube.get_buf mte1, %c14_i64, 0 {cannir.buffer.generated_sync}
                  asccube.get_buf mte1, %196, 0 {cannir.buffer.generated_sync}
                  asccube.copy_l12l0b %197[%c0], %198[%c0], %c0_i32, %c0_i32, %c4_i32, %c8_i32, %c4_i32, %c8_i32 {transpose = true} : memref<?xbf16, 4>, memref<1xbf16, 2>
                  asccube.release_buf mte1, %196, 0 {cannir.buffer.generated_sync}
                  asccube.release_buf mte1, %c14_i64, 0 {cannir.buffer.generated_sync}
                  %199 = arith.select %160, %134, %135 : i64
                  %200 = arith.select %160, %c6_i64, %c7_i64 : i64
                  %201 = ascscalar.bitcast_memref %alloc_13 byte_offset = %c0_i64 : memref<4096xbf16, 3> -> memref<?xbf16, 3>
                  %202 = ascscalar.bind_local_ptr %199 : memref<1xbf16, 2>
                  asccube.get_buf mte1, %c11_i64, 0 {cannir.buffer.generated_sync}
                  asccube.get_buf mte1, %200, 0 {cannir.buffer.generated_sync}
                  asccube.copy_l12l0a %201[%c0], %202[%c0], %c0_i32, %c0_i32, %c4_i32, %c4_i32, %c4_i32, %c4_i32 : memref<?xbf16, 3>, memref<1xbf16, 2>
                  asccube.release_buf mte1, %200, 0 {cannir.buffer.generated_sync}
                  asccube.release_buf mte1, %c11_i64, 0 {cannir.buffer.generated_sync}
                  %203 = arith.addi %arg29, %c1_i32 : i32
                  %204 = arith.cmpi slt, %203, %84 : i32
                  %205:2 = scf.if %204 -> (i64, i64) {
                    %251 = arith.cmpi eq, %arg30, %c0_i64 : i64
                    %252 = arith.select %251, %122, %123 : i64
                    %253 = arith.cmpi ne, %arg30, %c0_i64 : i64
                    %254 = arith.extui %253 : i1 to i64
                    %255 = arith.addi %arg30, %c1_i64 overflow<nsw> : i64
                    %256 = arith.remui %255, %c2_i64 : i64
                    %257 = arith.cmpi slt, %158, %c64_i64 : i64
                    scf.if %257 {
                      asccube.get_buf mte2, %254, 0 {cannir.buffer.generated_sync}
                      scf.for %arg37 = %c0_i64 to %c8_i64 step %c1_i64  : i64 {
                        %262 = ascscalar.bind_local_ptr %252 : memref<1xbf16, 2>
                        %263 = cannir.extract_physical_layout %arg2 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                        %264 = cannir.extract_stride %263 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                        %265 = cannir.extract_dim %264[0] : !cannir.stride<(-1, -1, -1, 1)>
                        %266 = arith.muli %94, %265 : i64
                        %267 = arith.index_cast %266 : i64 to index
                        %268 = cannir.extract_dim %264[1] : !cannir.stride<(-1, -1, -1, 1)>
                        %269 = arith.muli %103, %268 : i64
                        %270 = arith.index_cast %269 : i64 to index
                        %271 = arith.addi %267, %270 : index
                        %272 = arith.muli %146, %c64_i64 : i64
                        %273 = arith.addi %272, %93 : i64
                        %274 = cannir.extract_dim %264[2] : !cannir.stride<(-1, -1, -1, 1)>
                        %275 = arith.muli %273, %274 : i64
                        %276 = arith.index_cast %275 : i64 to index
                        %277 = arith.addi %271, %276 : index
                        %278 = arith.muli %arg37, %c16_i64 overflow<nsw> : i64
                        %279 = arith.index_cast %278 : i64 to index
                        %280 = arith.addi %277, %279 : index
                        %281 = cannir.extract_layout %arg2 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                        %282 = cannir.extract_stride %281 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                        %283 = cannir.extract_dim %282[2] : !cannir.stride<(-1, -1, -1, 1)>
                        %284 = arith.index_cast %283 : i64 to index
                        %285 = arith.muli %arg37, %c1024_i64 overflow<nsw> : i64
                        %286 = arith.muli %284, %c2 : index
                        %287 = arith.index_cast %286 : index to i64
                        %288 = arith.cmpi ugt, %158, %c0_i64 : i64
                        scf.if %288 {
                          %289 = arith.index_cast %285 : i64 to index
                          %290 = arith.trunci %158 : i64 to i32
                          asccube.copy_gm2l1 %262[%289], %alloc_0[%280], %290, %c16_i32, %287, %c1_i32, %c0_i64, %c0_i64, %c64_i32 l2_cache_ctl %c0_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
                        }
                      }
                      asccube.release_buf mte2, %254, 0 {cannir.buffer.generated_sync}
                    } else {
                      %262 = ascscalar.bind_local_ptr %252 : memref<1xbf16, 2>
                      %263 = cannir.extract_physical_layout %arg2 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                      %264 = cannir.extract_stride %263 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                      %265 = cannir.extract_dim %264[0] : !cannir.stride<(-1, -1, -1, 1)>
                      %266 = arith.muli %94, %265 : i64
                      %267 = arith.index_cast %266 : i64 to index
                      %268 = cannir.extract_dim %264[1] : !cannir.stride<(-1, -1, -1, 1)>
                      %269 = arith.muli %103, %268 : i64
                      %270 = arith.index_cast %269 : i64 to index
                      %271 = arith.addi %267, %270 : index
                      %272 = arith.muli %146, %c64_i64 : i64
                      %273 = arith.addi %272, %93 : i64
                      %274 = cannir.extract_dim %264[2] : !cannir.stride<(-1, -1, -1, 1)>
                      %275 = arith.muli %273, %274 : i64
                      %276 = arith.index_cast %275 : i64 to index
                      %277 = arith.addi %271, %276 : index
                      %278 = arith.divsi %78, %c64_i64 : i64
                      %279 = arith.minsi %146, %278 : i64
                      %280 = arith.muli %279, %c64_i64 : i64
                      %281 = arith.subi %78, %280 : i64
                      %282 = arith.minsi %281, %c64_i64 : i64
                      %283 = arith.cmpi sle, %146, %278 : i64
                      %284 = arith.select %283, %282, %c0_i64 : i64
                      %285 = cannir.extract_layout %arg2 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                      %286 = cannir.extract_stride %285 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                      %287 = cannir.extract_dim %286[2] : !cannir.stride<(-1, -1, -1, 1)>
                      %288 = arith.index_cast %287 : i64 to index
                      %289 = arith.muli %288, %c2 : index
                      %290 = arith.index_cast %289 : index to i64
                      asccube.get_buf mte2, %254, 0 {cannir.buffer.generated_sync}
                      %291 = arith.cmpi ugt, %284, %c0_i64 : i64
                      scf.if %291 {
                        %292 = arith.trunci %284 : i64 to i32
                        asccube.copy_gm2l1 %262[%c0], %alloc_0[%277], %292, %c128_i32, %290, %c1_i32, %c0_i64, %c0_i64, %c64_i32 l2_cache_ctl %c0_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
                      }
                      asccube.release_buf mte2, %254, 0 {cannir.buffer.generated_sync}
                    }
                    %258 = arith.select %251, %115, %116 : i64
                    %259 = arith.select %251, %c2_i64, %c3_i64 : i64
                    scf.if %257 {
                      asccube.get_buf mte2, %259, 0 {cannir.buffer.generated_sync}
                      scf.for %arg37 = %c0_i64 to %c8_i64 step %c1_i64  : i64 {
                        %262 = ascscalar.bind_local_ptr %258 : memref<1xbf16, 2>
                        %263 = cannir.extract_physical_layout %arg3 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                        %264 = cannir.extract_stride %263 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                        %265 = cannir.extract_dim %264[0] : !cannir.stride<(-1, -1, -1, 1)>
                        %266 = arith.muli %94, %265 : i64
                        %267 = arith.index_cast %266 : i64 to index
                        %268 = cannir.extract_dim %264[1] : !cannir.stride<(-1, -1, -1, 1)>
                        %269 = arith.muli %103, %268 : i64
                        %270 = arith.index_cast %269 : i64 to index
                        %271 = arith.addi %267, %270 : index
                        %272 = arith.muli %146, %c64_i64 : i64
                        %273 = arith.addi %272, %93 : i64
                        %274 = cannir.extract_dim %264[2] : !cannir.stride<(-1, -1, -1, 1)>
                        %275 = arith.muli %273, %274 : i64
                        %276 = arith.index_cast %275 : i64 to index
                        %277 = arith.addi %271, %276 : index
                        %278 = arith.muli %arg37, %c16_i64 overflow<nsw> : i64
                        %279 = arith.index_cast %278 : i64 to index
                        %280 = arith.addi %277, %279 : index
                        %281 = cannir.extract_layout %arg3 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                        %282 = cannir.extract_stride %281 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                        %283 = cannir.extract_dim %282[2] : !cannir.stride<(-1, -1, -1, 1)>
                        %284 = arith.index_cast %283 : i64 to index
                        %285 = arith.muli %arg37, %c1024_i64 overflow<nsw> : i64
                        %286 = arith.muli %284, %c2 : index
                        %287 = arith.index_cast %286 : index to i64
                        %288 = arith.cmpi ugt, %158, %c0_i64 : i64
                        scf.if %288 {
                          %289 = arith.index_cast %285 : i64 to index
                          %290 = arith.trunci %158 : i64 to i32
                          asccube.copy_gm2l1 %262[%289], %alloc_1[%280], %290, %c16_i32, %287, %c1_i32, %c0_i64, %c0_i64, %c64_i32 l2_cache_ctl %c0_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
                        }
                      }
                      asccube.release_buf mte2, %259, 0 {cannir.buffer.generated_sync}
                    } else {
                      %262 = ascscalar.bind_local_ptr %258 : memref<1xbf16, 2>
                      %263 = cannir.extract_physical_layout %arg3 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                      %264 = cannir.extract_stride %263 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                      %265 = cannir.extract_dim %264[0] : !cannir.stride<(-1, -1, -1, 1)>
                      %266 = arith.muli %94, %265 : i64
                      %267 = arith.index_cast %266 : i64 to index
                      %268 = cannir.extract_dim %264[1] : !cannir.stride<(-1, -1, -1, 1)>
                      %269 = arith.muli %103, %268 : i64
                      %270 = arith.index_cast %269 : i64 to index
                      %271 = arith.addi %267, %270 : index
                      %272 = arith.muli %146, %c64_i64 : i64
                      %273 = arith.addi %272, %93 : i64
                      %274 = cannir.extract_dim %264[2] : !cannir.stride<(-1, -1, -1, 1)>
                      %275 = arith.muli %273, %274 : i64
                      %276 = arith.index_cast %275 : i64 to index
                      %277 = arith.addi %271, %276 : index
                      %278 = arith.divsi %78, %c64_i64 : i64
                      %279 = arith.minsi %146, %278 : i64
                      %280 = arith.muli %279, %c64_i64 : i64
                      %281 = arith.subi %78, %280 : i64
                      %282 = arith.minsi %281, %c64_i64 : i64
                      %283 = arith.cmpi sle, %146, %278 : i64
                      %284 = arith.select %283, %282, %c0_i64 : i64
                      %285 = cannir.extract_layout %arg3 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                      %286 = cannir.extract_stride %285 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                      %287 = cannir.extract_dim %286[2] : !cannir.stride<(-1, -1, -1, 1)>
                      %288 = arith.index_cast %287 : i64 to index
                      %289 = arith.muli %288, %c2 : index
                      %290 = arith.index_cast %289 : index to i64
                      asccube.get_buf mte2, %259, 0 {cannir.buffer.generated_sync}
                      %291 = arith.cmpi ugt, %284, %c0_i64 : i64
                      scf.if %291 {
                        %292 = arith.trunci %284 : i64 to i32
                        asccube.copy_gm2l1 %262[%c0], %alloc_1[%277], %292, %c128_i32, %290, %c1_i32, %c0_i64, %c0_i64, %c64_i32 l2_cache_ctl %c0_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
                      }
                      asccube.release_buf mte2, %259, 0 {cannir.buffer.generated_sync}
                    }
                    %260 = arith.cmpi slt, %140, %158 : i64
                    %261 = scf.if %260 -> (i64) {
                      %262 = arith.addi %arg35, %c1_i64 : i64
                      %263 = arith.remui %262, %c2_i64 : i64
                      scf.yield %263 : i64
                    } else {
                      scf.yield %arg35 : i64
                    }
                    scf.yield %256, %261 : i64, i64
                  } else {
                    scf.yield %arg30, %arg35 : i64, i64
                  }
                  %206 = ascscalar.bitcast_memref %alloc_6 byte_offset = %c0_i64 : memref<4096xf32, 1> -> memref<?xf32, 1>
                  asccube.sync_intra_wait fixpipe, %c5_i64 {cannir.sync.banks = 3 : i32}
                  asccube.sync_intra_wait fixpipe, %c21_i64 {cannir.sync.banks = 3 : i32}
                  asccube.get_buf fixpipe, %c15_i64, 0 {cannir.buffer.generated_sync}
                  asccube.copy_l0c2ub %206[%c0], %183[%c0], %c128_i32, %c64_i32, %c128_i32, %c64_i32, %c0_i32 sub_block_id %c0_i32 {dual_dst_ctl = 1 : i32, nz2nd_en = true, relu_en = false} : memref<?xf32, 1>, memref<?xf32, 5>
                  asccube.release_buf fixpipe, %c15_i64, 0 {cannir.buffer.generated_sync}
                  asccube.sync_intra_arrive fixpipe, %c5_i64 {cannir.sync.banks = 3 : i32}
                  asccube.sync_intra_arrive fixpipe, %c21_i64 {cannir.sync.banks = 3 : i32}
                  %207 = ascscalar.memref_address %alloc_22 : memref<8192xbf16, 2>
                  %208 = ascscalar.memref_address %alloc_21 : memref<8192xbf16, 2>
                  %209 = arith.select %160, %207, %208 : i64
                  %210 = arith.select %160, %c10_i64, %c11_i64 : i64
                  %211 = arith.subi %151, %140 : i64
                  %212 = arith.cmpi slt, %211, %c0_i64 : i64
                  %213 = arith.select %212, %c0_i64, %211 : i64
                  %214 = arith.cmpi sgt, %213, %c32_i64 : i64
                  %215 = arith.select %214, %c32_i64, %213 : i64
                  %216 = arith.cmpi eq, %215, %c32_i64 : i64
                  %217 = scf.if %216 -> (i64) {
                    %251 = arith.addi %arg36, %c1_i64 : i64
                    %252 = arith.remui %251, %c2_i64 : i64
                    scf.yield %252 : i64
                  } else {
                    scf.yield %arg36 : i64
                  }
                  %218 = arith.cmpi sgt, %215, %c0_i64 : i64
                  %219 = arith.cmpi slt, %215, %c32_i64 : i64
                  %220 = arith.select %218, %219, %false : i1
                  %221 = scf.if %220 -> (i64) {
                    %251 = arith.addi %217, %c1_i64 : i64
                    %252 = arith.remui %251, %c2_i64 : i64
                    scf.yield %252 : i64
                  } else {
                    scf.yield %217 : i64
                  }
                  %222 = ascscalar.bind_local_ptr %209 : memref<1xbf16, 2>
                  %223 = arith.cmpi eq, %210, %c11_i64 : i64
                  %224 = arith.select %223, %c3_i64, %c2_i64 : i64
                  %225 = ascscalar.bitcast_memref %alloc_11 byte_offset = %c0_i64 : memref<8192xbf16, 3> -> memref<?xbf16, 3>
                  asccube.sync_intra_wait mte1, %224 {cannir.sync.banks = 3 : i32}
                  %226 = arith.addi %224, %c16_i64 overflow<nsw> : i64
                  asccube.sync_intra_wait mte1, %226 {cannir.sync.banks = 3 : i32}
                  asccube.get_buf mte1, %c10_i64, 0 {cannir.buffer.generated_sync}
                  asccube.copy_l12l0a %225[%c0], %222[%c0], %c0_i32, %c0_i32, %c4_i32, %c8_i32, %c4_i32, %c8_i32 {transpose = true} : memref<?xbf16, 3>, memref<1xbf16, 2>
                  asccube.release_buf mte1, %c10_i64, 0 {cannir.buffer.generated_sync}
                  %227 = ascscalar.bitcast_memref %alloc_10 byte_offset = %c0_i64 : memref<16384xf32, 5> -> memref<?xf32, 5>
                  asccube.get_buf m, %c17_i64, 0 {cannir.buffer.generated_sync}
                  asccube.get_buf m, %c10_i64, 0 {cannir.buffer.generated_sync}
                  asccube.get_buf m, %c14_i64, 0 {cannir.buffer.generated_sync}
                  asccube.mmad %227[%c0], %225, %197, %c128_i32, %c64_i32, %c128_i32, %c0_i32 {c_matrix_init_val = true} : memref<?xf32, 5>, memref<?xbf16, 3>, memref<?xbf16, 4>
                  asccube.release_buf m, %c14_i64, 0 {cannir.buffer.generated_sync}
                  asccube.release_buf m, %c10_i64, 0 {cannir.buffer.generated_sync}
                  asccube.release_buf m, %c17_i64, 0 {cannir.buffer.generated_sync}
                  %228 = ascscalar.bitcast_memref %alloc_5 byte_offset = %c0_i64 : memref<8192xf32, 1> -> memref<?xf32, 1>
                  asccube.sync_intra_wait fixpipe, %c4_i64 {cannir.sync.banks = 3 : i32}
                  asccube.sync_intra_wait fixpipe, %c20_i64 {cannir.sync.banks = 3 : i32}
                  asccube.get_buf fixpipe, %c17_i64, 0 {cannir.buffer.generated_sync}
                  asccube.copy_l0c2ub %228[%c0], %227[%c0], %c128_i32, %c128_i32, %c64_i32, %c128_i32, %c0_i32 sub_block_id %c0_i32 {dual_dst_ctl = 2 : i32, nz2nd_en = true, relu_en = false} : memref<?xf32, 1>, memref<?xf32, 5>
                  asccube.release_buf fixpipe, %c17_i64, 0 {cannir.buffer.generated_sync}
                  asccube.sync_intra_arrive fixpipe, %c4_i64 {cannir.sync.banks = 3 : i32}
                  asccube.sync_intra_arrive fixpipe, %c20_i64 {cannir.sync.banks = 3 : i32}
                  %229 = ascscalar.bitcast_memref %alloc_14 byte_offset = %c0_i64 : memref<8192xbf16, 4> -> memref<?xbf16, 4>
                  asccube.get_buf mte1, %c12_i64, 0 {cannir.buffer.generated_sync}
                  asccube.copy_l12l0b %229[%c0], %222[%c0], %c0_i32, %c0_i32, %c4_i32, %c8_i32, %c4_i32, %c8_i32 {transpose = true} : memref<?xbf16, 4>, memref<1xbf16, 2>
                  asccube.release_buf mte1, %c12_i64, 0 {cannir.buffer.generated_sync}
                  asccube.sync_intra_arrive mte1, %224 {cannir.sync.banks = 3 : i32}
                  asccube.sync_intra_arrive mte1, %226 {cannir.sync.banks = 3 : i32}
                  asccube.get_buf m, %c16_i64, 0 {cannir.buffer.generated_sync}
                  asccube.get_buf m, %c11_i64, 0 {cannir.buffer.generated_sync}
                  asccube.get_buf m, %c12_i64, 0 {cannir.buffer.generated_sync}
                  asccube.mmad %194[%c0], %201, %229, %c64_i32, %c64_i32, %c128_i32, %c0_i32 {c_matrix_init_val = false} : memref<?xf32, 5>, memref<?xbf16, 3>, memref<?xbf16, 4>
                  asccube.release_buf m, %c12_i64, 0 {cannir.buffer.generated_sync}
                  asccube.release_buf m, %c11_i64, 0 {cannir.buffer.generated_sync}
                  asccube.release_buf m, %c16_i64, 0 {cannir.buffer.generated_sync}
                  %230 = scf.if %204 -> (i64) {
                    %251 = arith.cmpi eq, %arg32, %c0_i64 : i64
                    %252 = arith.select %251, %128, %129 : i64
                    %253 = arith.select %251, %c4_i64, %c5_i64 : i64
                    %254 = arith.addi %arg32, %c1_i64 overflow<nsw> : i64
                    %255 = arith.remui %254, %c2_i64 : i64
                    %256 = arith.cmpi slt, %158, %c64_i64 : i64
                    scf.if %256 {
                      asccube.get_buf mte2, %253, 0 {cannir.buffer.generated_sync}
                      scf.for %arg37 = %c0_i64 to %c8_i64 step %c1_i64  : i64 {
                        %259 = ascscalar.bind_local_ptr %252 : memref<1xbf16, 2>
                        %260 = cannir.extract_physical_layout %arg0 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                        %261 = cannir.extract_stride %260 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                        %262 = cannir.extract_dim %261[0] : !cannir.stride<(-1, -1, -1, 1)>
                        %263 = arith.muli %94, %262 : i64
                        %264 = arith.index_cast %263 : i64 to index
                        %265 = cannir.extract_dim %261[1] : !cannir.stride<(-1, -1, -1, 1)>
                        %266 = arith.muli %103, %265 : i64
                        %267 = arith.index_cast %266 : i64 to index
                        %268 = arith.addi %264, %267 : index
                        %269 = arith.muli %146, %c64_i64 : i64
                        %270 = arith.addi %269, %93 : i64
                        %271 = cannir.extract_dim %261[2] : !cannir.stride<(-1, -1, -1, 1)>
                        %272 = arith.muli %270, %271 : i64
                        %273 = arith.index_cast %272 : i64 to index
                        %274 = arith.addi %268, %273 : index
                        %275 = arith.muli %arg37, %c16_i64 overflow<nsw> : i64
                        %276 = arith.index_cast %275 : i64 to index
                        %277 = arith.addi %274, %276 : index
                        %278 = cannir.extract_layout %arg0 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                        %279 = cannir.extract_stride %278 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                        %280 = cannir.extract_dim %279[2] : !cannir.stride<(-1, -1, -1, 1)>
                        %281 = arith.index_cast %280 : i64 to index
                        %282 = arith.muli %arg37, %c1024_i64 overflow<nsw> : i64
                        %283 = arith.muli %281, %c2 : index
                        %284 = arith.index_cast %283 : index to i64
                        %285 = arith.cmpi ugt, %158, %c0_i64 : i64
                        scf.if %285 {
                          %286 = arith.index_cast %282 : i64 to index
                          %287 = arith.trunci %158 : i64 to i32
                          asccube.copy_gm2l1 %259[%286], %alloc[%277], %287, %c16_i32, %284, %c1_i32, %c0_i64, %c0_i64, %c64_i32 l2_cache_ctl %c1_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
                        }
                      }
                      asccube.release_buf mte2, %253, 0 {cannir.buffer.generated_sync}
                    } else {
                      %259 = ascscalar.bind_local_ptr %252 : memref<1xbf16, 2>
                      %260 = cannir.extract_physical_layout %arg0 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                      %261 = cannir.extract_stride %260 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                      %262 = cannir.extract_dim %261[0] : !cannir.stride<(-1, -1, -1, 1)>
                      %263 = arith.muli %94, %262 : i64
                      %264 = arith.index_cast %263 : i64 to index
                      %265 = cannir.extract_dim %261[1] : !cannir.stride<(-1, -1, -1, 1)>
                      %266 = arith.muli %103, %265 : i64
                      %267 = arith.index_cast %266 : i64 to index
                      %268 = arith.addi %264, %267 : index
                      %269 = arith.muli %146, %c64_i64 : i64
                      %270 = arith.addi %269, %93 : i64
                      %271 = cannir.extract_dim %261[2] : !cannir.stride<(-1, -1, -1, 1)>
                      %272 = arith.muli %270, %271 : i64
                      %273 = arith.index_cast %272 : i64 to index
                      %274 = arith.addi %268, %273 : index
                      %275 = arith.divsi %78, %c64_i64 : i64
                      %276 = arith.minsi %146, %275 : i64
                      %277 = arith.muli %276, %c64_i64 : i64
                      %278 = arith.subi %78, %277 : i64
                      %279 = arith.minsi %278, %c64_i64 : i64
                      %280 = arith.cmpi sle, %146, %275 : i64
                      %281 = arith.select %280, %279, %c0_i64 : i64
                      %282 = cannir.extract_layout %arg0 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                      %283 = cannir.extract_stride %282 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                      %284 = cannir.extract_dim %283[2] : !cannir.stride<(-1, -1, -1, 1)>
                      %285 = arith.index_cast %284 : i64 to index
                      %286 = arith.muli %285, %c2 : index
                      %287 = arith.index_cast %286 : index to i64
                      asccube.get_buf mte2, %253, 0 {cannir.buffer.generated_sync}
                      %288 = arith.cmpi ugt, %281, %c0_i64 : i64
                      scf.if %288 {
                        %289 = arith.trunci %281 : i64 to i32
                        asccube.copy_gm2l1 %259[%c0], %alloc[%274], %289, %c128_i32, %287, %c1_i32, %c0_i64, %c0_i64, %c64_i32 l2_cache_ctl %c1_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
                      }
                      asccube.release_buf mte2, %253, 0 {cannir.buffer.generated_sync}
                    }
                    %257 = arith.select %251, %134, %135 : i64
                    %258 = arith.select %251, %c6_i64, %c7_i64 : i64
                    scf.if %256 {
                      asccube.get_buf mte2, %258, 0 {cannir.buffer.generated_sync}
                      scf.for %arg37 = %c0_i64 to %c4_i64 step %c1_i64  : i64 {
                        %259 = ascscalar.bind_local_ptr %257 : memref<1xbf16, 2>
                        %260 = cannir.extract_physical_layout %arg5 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 64, 64)>, <(-1, -1, 4096, 64, 1)>>, <<(-1, -1, -1, 64, 64)>, <(-1, -1, 4096, 64, 1)>>, identity>
                        %261 = cannir.extract_stride %260 : <<(-1, -1, -1, 64, 64)>, <(-1, -1, 4096, 64, 1)>> -> <(-1, -1, 4096, 64, 1)>
                        %262 = cannir.extract_dim %261[0] : !cannir.stride<(-1, -1, 4096, 64, 1)>
                        %263 = arith.muli %104, %262 : i64
                        %264 = arith.index_cast %263 : i64 to index
                        %265 = cannir.extract_dim %261[1] : !cannir.stride<(-1, -1, 4096, 64, 1)>
                        %266 = arith.muli %103, %265 : i64
                        %267 = arith.index_cast %266 : i64 to index
                        %268 = arith.addi %264, %267 : index
                        %269 = arith.muli %146, %c4096_i64 : i64
                        %270 = arith.index_cast %269 : i64 to index
                        %271 = arith.addi %268, %270 : index
                        %272 = arith.muli %arg37, %c16_i64 overflow<nsw> : i64
                        %273 = arith.index_cast %272 : i64 to index
                        %274 = arith.addi %271, %273 : index
                        %275 = arith.muli %arg37, %c1024_i64 overflow<nsw> : i64
                        %276 = arith.cmpi ugt, %158, %c0_i64 : i64
                        scf.if %276 {
                          %277 = arith.index_cast %275 : i64 to index
                          %278 = arith.trunci %158 : i64 to i32
                          asccube.copy_gm2l1 %259[%277], %alloc_2[%274], %278, %c16_i32, %c128_i64, %c1_i32, %c0_i64, %c0_i64, %c64_i32 l2_cache_ctl %c1_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
                        }
                      }
                      asccube.release_buf mte2, %258, 0 {cannir.buffer.generated_sync}
                    } else {
                      %259 = ascscalar.bind_local_ptr %257 : memref<1xbf16, 2>
                      %260 = cannir.extract_physical_layout %arg5 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 64, 64)>, <(-1, -1, 4096, 64, 1)>>, <<(-1, -1, -1, 64, 64)>, <(-1, -1, 4096, 64, 1)>>, identity>
                      %261 = cannir.extract_stride %260 : <<(-1, -1, -1, 64, 64)>, <(-1, -1, 4096, 64, 1)>> -> <(-1, -1, 4096, 64, 1)>
                      %262 = cannir.extract_dim %261[0] : !cannir.stride<(-1, -1, 4096, 64, 1)>
                      %263 = arith.muli %104, %262 : i64
                      %264 = arith.index_cast %263 : i64 to index
                      %265 = cannir.extract_dim %261[1] : !cannir.stride<(-1, -1, 4096, 64, 1)>
                      %266 = arith.muli %103, %265 : i64
                      %267 = arith.index_cast %266 : i64 to index
                      %268 = arith.addi %264, %267 : index
                      %269 = arith.muli %146, %c4096_i64 : i64
                      %270 = arith.index_cast %269 : i64 to index
                      %271 = arith.addi %268, %270 : index
                      asccube.get_buf mte2, %258, 0 {cannir.buffer.generated_sync}
                      asccube.copy_gm2l1 %259[%c0], %alloc_2[%271], %c64_i32, %c64_i32, %c128_i64, %c1_i32, %c0_i64, %c0_i64, %c64_i32 l2_cache_ctl %c1_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
                      asccube.release_buf mte2, %258, 0 {cannir.buffer.generated_sync}
                    }
                    scf.yield %255 : i64
                  } else {
                    scf.yield %arg32 : i64
                  }
                  %231 = cannir.extract_physical_layout %arg6 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                  %232 = cannir.extract_stride %231 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                  %233 = cannir.extract_dim %232[0] : !cannir.stride<(-1, -1, -1, 1)>
                  %234 = arith.muli %94, %233 : i64
                  %235 = arith.index_cast %234 : i64 to index
                  %236 = cannir.extract_dim %232[1] : !cannir.stride<(-1, -1, -1, 1)>
                  %237 = arith.muli %103, %236 : i64
                  %238 = arith.index_cast %237 : i64 to index
                  %239 = arith.addi %235, %238 : index
                  %240 = arith.muli %159, %c64_i64 : i64
                  %241 = arith.addi %240, %93 : i64
                  %242 = cannir.extract_dim %232[2] : !cannir.stride<(-1, -1, -1, 1)>
                  %243 = arith.muli %241, %242 : i64
                  %244 = arith.index_cast %243 : i64 to index
                  %245 = arith.addi %239, %244 : index
                  %246 = cannir.extract_layout %arg6 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                  %247 = cannir.extract_stride %246 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                  %248 = cannir.extract_dim %247[2] : !cannir.stride<(-1, -1, -1, 1)>
                  asccube.get_buf fixpipe, %c16_i64, 0 {cannir.buffer.generated_sync}
                  %249 = arith.trunci %248 : i64 to i32
                  asccube.copy_l0c2gm %alloc_3[%245], %194[%c0], %c128_i32, %c64_i32, %249, %c64_i32, %c0_i32 l2_cache_ctl %c4_i32 {nz2nd_en = true, relu_en = false} : memref<1xbf16>, memref<?xf32, 5>
                  asccube.release_buf fixpipe, %c16_i64, 0 {cannir.buffer.generated_sync}
                  %250 = scf.if %204 -> (i64) {
                    %251 = arith.addi %arg33, %c1_i64 overflow<nsw> : i64
                    %252 = arith.remui %251, %c2_i64 : i64
                    scf.yield %252 : i64
                  } else {
                    scf.yield %arg33 : i64
                  }
                  scf.yield %205#0, %166, %230, %250, %188, %205#1, %221 : i64, i64, i64, i64, i64, i64, i64
                }
                scf.yield %143#0, %143#1, %143#2, %143#3, %143#4, %143#5, %143#6 : i64, i64, i64, i64, i64, i64, i64
              }
              scf.yield %98#0, %98#1, %98#2, %98#3, %98#4, %98#5, %98#6 : i64, i64, i64, i64, i64, i64, i64
            } else {
              scf.yield %arg14, %arg15, %arg16, %arg17, %arg18, %arg19, %arg20 : i64, i64, i64, i64, i64, i64, i64
            }
            scf.yield %97#0, %97#1, %97#2, %97#3, %97#4, %97#5, %97#6 : i64, i64, i64, i64, i64, i64, i64
          }
        } else {
          %48 = arith.cmpi eq, %44, %c2_i32 : i32
          scf.if %48 {
            %49:7 = scf.for %arg13 = %c0_i32 to %21 step %c1_i32 iter_args(%arg14 = %c0_i64, %arg15 = %c0_i64, %arg16 = %c0_i64, %arg17 = %c0_i64, %arg18 = %c0_i64, %arg19 = %c0_i64, %arg20 = %c0_i64) -> (i64, i64, i64, i64, i64, i64, i64)  : i32 {
              %50 = arith.addi %32, %arg13 : i32
              %51 = arith.index_cast %50 : i32 to index
              %52 = memref.load %alloc_4[%51] : memref<1xi32>
              %53 = arith.addi %50, %c1_i32 : i32
              %54 = arith.index_cast %53 : i32 to index
              %55 = memref.load %alloc_4[%54] : memref<1xi32>
              %56 = arith.cmpi slt, %39, %52 : i32
              %57 = arith.select %56, %52, %39 : i32
              %58 = arith.cmpi sgt, %42, %55 : i32
              %59 = arith.select %58, %55, %42 : i32
              %60 = arith.cmpi slt, %57, %59 : i32
              %61 = arith.subi %57, %52 : i32
              %62 = arith.subi %59, %52 : i32
              %63 = arith.extsi %61 : i32 to i64
              %64 = arith.extsi %62 : i32 to i64
              %65 = arith.select %60, %63, %c0_i64 : i64
              %66 = arith.select %60, %64, %c0_i64 : i64
              %67 = arith.addi %22, %arg13 : i32
              %68 = arith.index_cast %67 : i32 to index
              %69 = memref.load %alloc_4[%68] : memref<1xi32>
              %70 = arith.addi %23, %arg13 : i32
              %71 = arith.index_cast %70 : i32 to index
              %72 = memref.load %alloc_4[%71] : memref<1xi32>
              %73 = arith.addi %24, %arg13 : i32
              %74 = arith.index_cast %73 : i32 to index
              %75 = memref.load %alloc_4[%74] : memref<1xi32>
              %76 = arith.addi %25, %arg13 : i32
              %77 = arith.index_cast %76 : i32 to index
              %78 = memref.load %alloc_4[%77] : memref<1xi32>
              %79 = arith.extsi %78 : i32 to i64
              %80 = arith.addi %27, %arg13 : i32
              %81 = arith.index_cast %80 : i32 to index
              %82 = memref.load %alloc_4[%81] : memref<1xi32>
              %83 = arith.addi %28, %arg13 : i32
              %84 = arith.index_cast %83 : i32 to index
              %85 = memref.load %alloc_4[%84] : memref<1xi32>
              %86 = arith.addi %79, %c63_i64 : i64
              %87 = arith.remsi %86, %c64_i64 : i64
              %88 = arith.cmpi ne, %87, %c0_i64 : i64
              %89 = arith.cmpi slt, %87, %c0_i64 : i64
              %90 = arith.andi %88, %89 : i1
              %91 = arith.divsi %86, %c64_i64 : i64
              %92 = arith.extui %90 : i1 to i64
              %93 = arith.subi %91, %92 : i64
              %94 = arith.extsi %75 : i32 to i64
              %95 = arith.extsi %72 : i32 to i64
              %96 = ascscalar.get_subblock_id
              %97 = arith.cmpi slt, %0, %1 : i64
              %98:7 = scf.if %97 -> (i64, i64, i64, i64, i64, i64, i64) {
                %99:7 = scf.for %arg21 = %65 to %66 step %c1_i64 iter_args(%arg22 = %arg14, %arg23 = %arg15, %arg24 = %arg16, %arg25 = %arg17, %arg26 = %arg18, %arg27 = %arg19, %arg28 = %arg20) -> (i64, i64, i64, i64, i64, i64, i64)  : i64 {
                  %100 = arith.remsi %arg21, %c2_i64 : i64
                  %101 = arith.cmpi slt, %100, %c0_i64 : i64
                  %102 = arith.divsi %arg21, %c2_i64 : i64
                  %103 = arith.extui %101 : i1 to i64
                  %104 = arith.subi %102, %103 : i64
                  %105 = arith.maxsi %7, %c1_i64 : i64
                  %106 = arith.remsi %104, %105 : i64
                  %107 = arith.cmpi slt, %106, %c0_i64 : i64
                  %108 = arith.select %107, %105, %c0_i64 : i64
                  %109 = arith.addi %106, %108 : i64
                  %110 = arith.select %101, %c2_i64, %c0_i64 : i64
                  %111 = arith.addi %100, %110 overflow<nsw> : i64
                  %112 = arith.extsi %69 : i32 to i64
                  %113 = arith.addi %82, %85 : i32
                  %114 = arith.extsi %113 : i32 to i64
                  %115 = arith.cmpi eq, %114, %93 : i64
                  %116 = arith.muli %82, %c64_i32 : i32
                  %117 = arith.extsi %116 : i32 to i64
                  %118 = arith.subi %79, %117 : i64
                  %119 = arith.minsi %118, %c64_i64 : i64
                  %120 = arith.select %115, %119, %c64_i64 : i64
                  %121 = arith.extsi %82 : i32 to i64
                  %122 = arith.cmpi eq, %arg22, %c0_i64 : i64
                  %123 = ascscalar.memref_address %alloc_24 : memref<8192xbf16, 2>
                  %124 = ascscalar.memref_address %alloc_23 : memref<8192xbf16, 2>
                  %125 = arith.select %122, %123, %124 : i64
                  %126 = arith.select %122, %c2_i64, %c3_i64 : i64
                  %127 = arith.addi %arg22, %c1_i64 : i64
                  %128 = arith.remui %127, %c2_i64 : i64
                  %129 = arith.cmpi slt, %120, %c64_i64 : i64
                  scf.if %129 {
                    asccube.get_buf mte2, %126, 0 {cannir.buffer.generated_sync}
                    scf.for %arg29 = %c0_i64 to %c8_i64 step %c1_i64  : i64 {
                      %152 = ascscalar.bind_local_ptr %125 : memref<1xbf16, 2>
                      %153 = cannir.extract_physical_layout %arg3 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                      %154 = cannir.extract_stride %153 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                      %155 = cannir.extract_dim %154[0] : !cannir.stride<(-1, -1, -1, 1)>
                      %156 = arith.muli %95, %155 : i64
                      %157 = arith.index_cast %156 : i64 to index
                      %158 = cannir.extract_dim %154[1] : !cannir.stride<(-1, -1, -1, 1)>
                      %159 = arith.muli %109, %158 : i64
                      %160 = arith.index_cast %159 : i64 to index
                      %161 = arith.addi %157, %160 : index
                      %162 = arith.muli %121, %c64_i64 : i64
                      %163 = arith.addi %162, %94 : i64
                      %164 = cannir.extract_dim %154[2] : !cannir.stride<(-1, -1, -1, 1)>
                      %165 = arith.muli %163, %164 : i64
                      %166 = arith.index_cast %165 : i64 to index
                      %167 = arith.addi %161, %166 : index
                      %168 = arith.muli %arg29, %c16_i64 overflow<nsw> : i64
                      %169 = arith.index_cast %168 : i64 to index
                      %170 = arith.addi %167, %169 : index
                      %171 = cannir.extract_layout %arg3 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                      %172 = cannir.extract_stride %171 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                      %173 = cannir.extract_dim %172[2] : !cannir.stride<(-1, -1, -1, 1)>
                      %174 = arith.index_cast %173 : i64 to index
                      %175 = arith.muli %arg29, %c1024_i64 overflow<nsw> : i64
                      %176 = arith.muli %174, %c2 : index
                      %177 = arith.index_cast %176 : index to i64
                      %178 = arith.cmpi ugt, %120, %c0_i64 : i64
                      scf.if %178 {
                        %179 = arith.index_cast %175 : i64 to index
                        %180 = arith.trunci %120 : i64 to i32
                        asccube.copy_gm2l1 %152[%179], %alloc_1[%170], %180, %c16_i32, %177, %c1_i32, %c0_i64, %c0_i64, %c64_i32 l2_cache_ctl %c0_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
                      }
                    }
                    asccube.release_buf mte2, %126, 0 {cannir.buffer.generated_sync}
                  } else {
                    %152 = ascscalar.bind_local_ptr %125 : memref<1xbf16, 2>
                    %153 = cannir.extract_physical_layout %arg3 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                    %154 = cannir.extract_stride %153 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                    %155 = cannir.extract_dim %154[0] : !cannir.stride<(-1, -1, -1, 1)>
                    %156 = arith.muli %95, %155 : i64
                    %157 = arith.index_cast %156 : i64 to index
                    %158 = cannir.extract_dim %154[1] : !cannir.stride<(-1, -1, -1, 1)>
                    %159 = arith.muli %109, %158 : i64
                    %160 = arith.index_cast %159 : i64 to index
                    %161 = arith.addi %157, %160 : index
                    %162 = arith.muli %121, %c64_i64 : i64
                    %163 = arith.addi %162, %94 : i64
                    %164 = cannir.extract_dim %154[2] : !cannir.stride<(-1, -1, -1, 1)>
                    %165 = arith.muli %163, %164 : i64
                    %166 = arith.index_cast %165 : i64 to index
                    %167 = arith.addi %161, %166 : index
                    %168 = arith.divsi %79, %c64_i64 : i64
                    %169 = arith.minsi %121, %168 : i64
                    %170 = arith.muli %169, %c64_i64 : i64
                    %171 = arith.subi %79, %170 : i64
                    %172 = arith.minsi %171, %c64_i64 : i64
                    %173 = arith.cmpi sle, %121, %168 : i64
                    %174 = arith.select %173, %172, %c0_i64 : i64
                    %175 = cannir.extract_layout %arg3 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                    %176 = cannir.extract_stride %175 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                    %177 = cannir.extract_dim %176[2] : !cannir.stride<(-1, -1, -1, 1)>
                    %178 = arith.index_cast %177 : i64 to index
                    %179 = arith.muli %178, %c2 : index
                    %180 = arith.index_cast %179 : index to i64
                    asccube.get_buf mte2, %126, 0 {cannir.buffer.generated_sync}
                    %181 = arith.cmpi ugt, %174, %c0_i64 : i64
                    scf.if %181 {
                      %182 = arith.trunci %174 : i64 to i32
                      asccube.copy_gm2l1 %152[%c0], %alloc_1[%167], %182, %c128_i32, %180, %c1_i32, %c0_i64, %c0_i64, %c64_i32 l2_cache_ctl %c0_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
                    }
                    asccube.release_buf mte2, %126, 0 {cannir.buffer.generated_sync}
                  }
                  %130 = ascscalar.memref_address %alloc_26 : memref<8192xbf16, 2>
                  %131 = ascscalar.memref_address %alloc_25 : memref<8192xbf16, 2>
                  %132 = arith.select %122, %130, %131 : i64
                  %133 = arith.cmpi ne, %arg22, %c0_i64 : i64
                  %134 = arith.extui %133 : i1 to i64
                  scf.if %129 {
                    asccube.get_buf mte2, %134, 0 {cannir.buffer.generated_sync}
                    scf.for %arg29 = %c0_i64 to %c8_i64 step %c1_i64  : i64 {
                      %152 = ascscalar.bind_local_ptr %132 : memref<1xbf16, 2>
                      %153 = cannir.extract_physical_layout %arg2 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                      %154 = cannir.extract_stride %153 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                      %155 = cannir.extract_dim %154[0] : !cannir.stride<(-1, -1, -1, 1)>
                      %156 = arith.muli %95, %155 : i64
                      %157 = arith.index_cast %156 : i64 to index
                      %158 = cannir.extract_dim %154[1] : !cannir.stride<(-1, -1, -1, 1)>
                      %159 = arith.muli %109, %158 : i64
                      %160 = arith.index_cast %159 : i64 to index
                      %161 = arith.addi %157, %160 : index
                      %162 = arith.muli %121, %c64_i64 : i64
                      %163 = arith.addi %162, %94 : i64
                      %164 = cannir.extract_dim %154[2] : !cannir.stride<(-1, -1, -1, 1)>
                      %165 = arith.muli %163, %164 : i64
                      %166 = arith.index_cast %165 : i64 to index
                      %167 = arith.addi %161, %166 : index
                      %168 = arith.muli %arg29, %c16_i64 overflow<nsw> : i64
                      %169 = arith.index_cast %168 : i64 to index
                      %170 = arith.addi %167, %169 : index
                      %171 = cannir.extract_layout %arg2 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                      %172 = cannir.extract_stride %171 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                      %173 = cannir.extract_dim %172[2] : !cannir.stride<(-1, -1, -1, 1)>
                      %174 = arith.index_cast %173 : i64 to index
                      %175 = arith.muli %arg29, %c1024_i64 overflow<nsw> : i64
                      %176 = arith.muli %174, %c2 : index
                      %177 = arith.index_cast %176 : index to i64
                      %178 = arith.cmpi ugt, %120, %c0_i64 : i64
                      scf.if %178 {
                        %179 = arith.index_cast %175 : i64 to index
                        %180 = arith.trunci %120 : i64 to i32
                        asccube.copy_gm2l1 %152[%179], %alloc_0[%170], %180, %c16_i32, %177, %c1_i32, %c0_i64, %c0_i64, %c64_i32 l2_cache_ctl %c0_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
                      }
                    }
                    asccube.release_buf mte2, %134, 0 {cannir.buffer.generated_sync}
                  } else {
                    %152 = ascscalar.bind_local_ptr %132 : memref<1xbf16, 2>
                    %153 = cannir.extract_physical_layout %arg2 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                    %154 = cannir.extract_stride %153 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                    %155 = cannir.extract_dim %154[0] : !cannir.stride<(-1, -1, -1, 1)>
                    %156 = arith.muli %95, %155 : i64
                    %157 = arith.index_cast %156 : i64 to index
                    %158 = cannir.extract_dim %154[1] : !cannir.stride<(-1, -1, -1, 1)>
                    %159 = arith.muli %109, %158 : i64
                    %160 = arith.index_cast %159 : i64 to index
                    %161 = arith.addi %157, %160 : index
                    %162 = arith.muli %121, %c64_i64 : i64
                    %163 = arith.addi %162, %94 : i64
                    %164 = cannir.extract_dim %154[2] : !cannir.stride<(-1, -1, -1, 1)>
                    %165 = arith.muli %163, %164 : i64
                    %166 = arith.index_cast %165 : i64 to index
                    %167 = arith.addi %161, %166 : index
                    %168 = arith.divsi %79, %c64_i64 : i64
                    %169 = arith.minsi %121, %168 : i64
                    %170 = arith.muli %169, %c64_i64 : i64
                    %171 = arith.subi %79, %170 : i64
                    %172 = arith.minsi %171, %c64_i64 : i64
                    %173 = arith.cmpi sle, %121, %168 : i64
                    %174 = arith.select %173, %172, %c0_i64 : i64
                    %175 = cannir.extract_layout %arg2 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                    %176 = cannir.extract_stride %175 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                    %177 = cannir.extract_dim %176[2] : !cannir.stride<(-1, -1, -1, 1)>
                    %178 = arith.index_cast %177 : i64 to index
                    %179 = arith.muli %178, %c2 : index
                    %180 = arith.index_cast %179 : index to i64
                    asccube.get_buf mte2, %134, 0 {cannir.buffer.generated_sync}
                    %181 = arith.cmpi ugt, %174, %c0_i64 : i64
                    scf.if %181 {
                      %182 = arith.trunci %174 : i64 to i32
                      asccube.copy_gm2l1 %152[%c0], %alloc_0[%167], %182, %c128_i32, %180, %c1_i32, %c0_i64, %c0_i64, %c64_i32 l2_cache_ctl %c0_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
                    }
                    asccube.release_buf mte2, %134, 0 {cannir.buffer.generated_sync}
                  }
                  %135 = arith.cmpi eq, %arg24, %c0_i64 : i64
                  %136 = ascscalar.memref_address %alloc_20 : memref<8192xbf16, 2>
                  %137 = ascscalar.memref_address %alloc_19 : memref<8192xbf16, 2>
                  %138 = arith.select %135, %136, %137 : i64
                  %139 = arith.select %135, %c4_i64, %c5_i64 : i64
                  %140 = arith.addi %arg24, %c1_i64 : i64
                  %141 = arith.remui %140, %c2_i64 : i64
                  scf.if %129 {
                    asccube.get_buf mte2, %139, 0 {cannir.buffer.generated_sync}
                    scf.for %arg29 = %c0_i64 to %c8_i64 step %c1_i64  : i64 {
                      %152 = ascscalar.bind_local_ptr %138 : memref<1xbf16, 2>
                      %153 = cannir.extract_physical_layout %arg0 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                      %154 = cannir.extract_stride %153 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                      %155 = cannir.extract_dim %154[0] : !cannir.stride<(-1, -1, -1, 1)>
                      %156 = arith.muli %95, %155 : i64
                      %157 = arith.index_cast %156 : i64 to index
                      %158 = cannir.extract_dim %154[1] : !cannir.stride<(-1, -1, -1, 1)>
                      %159 = arith.muli %109, %158 : i64
                      %160 = arith.index_cast %159 : i64 to index
                      %161 = arith.addi %157, %160 : index
                      %162 = arith.muli %121, %c64_i64 : i64
                      %163 = arith.addi %162, %94 : i64
                      %164 = cannir.extract_dim %154[2] : !cannir.stride<(-1, -1, -1, 1)>
                      %165 = arith.muli %163, %164 : i64
                      %166 = arith.index_cast %165 : i64 to index
                      %167 = arith.addi %161, %166 : index
                      %168 = arith.muli %arg29, %c16_i64 overflow<nsw> : i64
                      %169 = arith.index_cast %168 : i64 to index
                      %170 = arith.addi %167, %169 : index
                      %171 = cannir.extract_layout %arg0 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                      %172 = cannir.extract_stride %171 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                      %173 = cannir.extract_dim %172[2] : !cannir.stride<(-1, -1, -1, 1)>
                      %174 = arith.index_cast %173 : i64 to index
                      %175 = arith.muli %arg29, %c1024_i64 overflow<nsw> : i64
                      %176 = arith.muli %174, %c2 : index
                      %177 = arith.index_cast %176 : index to i64
                      %178 = arith.cmpi ugt, %120, %c0_i64 : i64
                      scf.if %178 {
                        %179 = arith.index_cast %175 : i64 to index
                        %180 = arith.trunci %120 : i64 to i32
                        asccube.copy_gm2l1 %152[%179], %alloc[%170], %180, %c16_i32, %177, %c1_i32, %c0_i64, %c0_i64, %c64_i32 l2_cache_ctl %c1_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
                      }
                    }
                    asccube.release_buf mte2, %139, 0 {cannir.buffer.generated_sync}
                  } else {
                    %152 = ascscalar.bind_local_ptr %138 : memref<1xbf16, 2>
                    %153 = cannir.extract_physical_layout %arg0 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                    %154 = cannir.extract_stride %153 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                    %155 = cannir.extract_dim %154[0] : !cannir.stride<(-1, -1, -1, 1)>
                    %156 = arith.muli %95, %155 : i64
                    %157 = arith.index_cast %156 : i64 to index
                    %158 = cannir.extract_dim %154[1] : !cannir.stride<(-1, -1, -1, 1)>
                    %159 = arith.muli %109, %158 : i64
                    %160 = arith.index_cast %159 : i64 to index
                    %161 = arith.addi %157, %160 : index
                    %162 = arith.muli %121, %c64_i64 : i64
                    %163 = arith.addi %162, %94 : i64
                    %164 = cannir.extract_dim %154[2] : !cannir.stride<(-1, -1, -1, 1)>
                    %165 = arith.muli %163, %164 : i64
                    %166 = arith.index_cast %165 : i64 to index
                    %167 = arith.addi %161, %166 : index
                    %168 = arith.divsi %79, %c64_i64 : i64
                    %169 = arith.minsi %121, %168 : i64
                    %170 = arith.muli %169, %c64_i64 : i64
                    %171 = arith.subi %79, %170 : i64
                    %172 = arith.minsi %171, %c64_i64 : i64
                    %173 = arith.cmpi sle, %121, %168 : i64
                    %174 = arith.select %173, %172, %c0_i64 : i64
                    %175 = cannir.extract_layout %arg0 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                    %176 = cannir.extract_stride %175 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                    %177 = cannir.extract_dim %176[2] : !cannir.stride<(-1, -1, -1, 1)>
                    %178 = arith.index_cast %177 : i64 to index
                    %179 = arith.muli %178, %c2 : index
                    %180 = arith.index_cast %179 : index to i64
                    asccube.get_buf mte2, %139, 0 {cannir.buffer.generated_sync}
                    %181 = arith.cmpi ugt, %174, %c0_i64 : i64
                    scf.if %181 {
                      %182 = arith.trunci %174 : i64 to i32
                      asccube.copy_gm2l1 %152[%c0], %alloc[%167], %182, %c128_i32, %180, %c1_i32, %c0_i64, %c0_i64, %c64_i32 l2_cache_ctl %c1_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
                    }
                    asccube.release_buf mte2, %139, 0 {cannir.buffer.generated_sync}
                  }
                  %142 = ascscalar.memref_address %alloc_18 : memref<4096xbf16, 2>
                  %143 = ascscalar.memref_address %alloc_17 : memref<4096xbf16, 2>
                  %144 = arith.select %135, %142, %143 : i64
                  %145 = arith.select %135, %c6_i64, %c7_i64 : i64
                  scf.if %129 {
                    asccube.get_buf mte2, %145, 0 {cannir.buffer.generated_sync}
                    scf.for %arg29 = %c0_i64 to %c4_i64 step %c1_i64  : i64 {
                      %152 = ascscalar.bind_local_ptr %144 : memref<1xbf16, 2>
                      %153 = cannir.extract_physical_layout %arg5 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 64, 64)>, <(-1, -1, 4096, 64, 1)>>, <<(-1, -1, -1, 64, 64)>, <(-1, -1, 4096, 64, 1)>>, identity>
                      %154 = cannir.extract_stride %153 : <<(-1, -1, -1, 64, 64)>, <(-1, -1, 4096, 64, 1)>> -> <(-1, -1, 4096, 64, 1)>
                      %155 = cannir.extract_dim %154[0] : !cannir.stride<(-1, -1, 4096, 64, 1)>
                      %156 = arith.muli %112, %155 : i64
                      %157 = arith.index_cast %156 : i64 to index
                      %158 = cannir.extract_dim %154[1] : !cannir.stride<(-1, -1, 4096, 64, 1)>
                      %159 = arith.muli %109, %158 : i64
                      %160 = arith.index_cast %159 : i64 to index
                      %161 = arith.addi %157, %160 : index
                      %162 = arith.muli %121, %c4096_i64 : i64
                      %163 = arith.index_cast %162 : i64 to index
                      %164 = arith.addi %161, %163 : index
                      %165 = arith.muli %arg29, %c16_i64 overflow<nsw> : i64
                      %166 = arith.index_cast %165 : i64 to index
                      %167 = arith.addi %164, %166 : index
                      %168 = arith.muli %arg29, %c1024_i64 overflow<nsw> : i64
                      %169 = arith.cmpi ugt, %120, %c0_i64 : i64
                      scf.if %169 {
                        %170 = arith.index_cast %168 : i64 to index
                        %171 = arith.trunci %120 : i64 to i32
                        asccube.copy_gm2l1 %152[%170], %alloc_2[%167], %171, %c16_i32, %c128_i64, %c1_i32, %c0_i64, %c0_i64, %c64_i32 l2_cache_ctl %c1_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
                      }
                    }
                    asccube.release_buf mte2, %145, 0 {cannir.buffer.generated_sync}
                  } else {
                    %152 = ascscalar.bind_local_ptr %144 : memref<1xbf16, 2>
                    %153 = cannir.extract_physical_layout %arg5 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 64, 64)>, <(-1, -1, 4096, 64, 1)>>, <<(-1, -1, -1, 64, 64)>, <(-1, -1, 4096, 64, 1)>>, identity>
                    %154 = cannir.extract_stride %153 : <<(-1, -1, -1, 64, 64)>, <(-1, -1, 4096, 64, 1)>> -> <(-1, -1, 4096, 64, 1)>
                    %155 = cannir.extract_dim %154[0] : !cannir.stride<(-1, -1, 4096, 64, 1)>
                    %156 = arith.muli %112, %155 : i64
                    %157 = arith.index_cast %156 : i64 to index
                    %158 = cannir.extract_dim %154[1] : !cannir.stride<(-1, -1, 4096, 64, 1)>
                    %159 = arith.muli %109, %158 : i64
                    %160 = arith.index_cast %159 : i64 to index
                    %161 = arith.addi %157, %160 : index
                    %162 = arith.muli %121, %c4096_i64 : i64
                    %163 = arith.index_cast %162 : i64 to index
                    %164 = arith.addi %161, %163 : index
                    asccube.get_buf mte2, %145, 0 {cannir.buffer.generated_sync}
                    asccube.copy_gm2l1 %152[%c0], %alloc_2[%164], %c64_i32, %c64_i32, %c128_i64, %c1_i32, %c0_i64, %c0_i64, %c64_i32 l2_cache_ctl %c1_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
                    asccube.release_buf mte2, %145, 0 {cannir.buffer.generated_sync}
                  }
                  %146 = arith.cmpi eq, %82, %c0_i32 : i32
                  %147 = scf.if %146 -> (i64) {
                    %152 = arith.addi %arg25, %c1_i64 : i64
                    %153 = arith.remui %152, %c2_i64 : i64
                    scf.yield %153 : i64
                  } else {
                    %152 = arith.addi %arg25, %c1_i64 : i64
                    %153 = arith.remui %152, %c2_i64 : i64
                    scf.yield %153 : i64
                  }
                  %148 = arith.muli %96, %c32_i64 : i64
                  %149 = arith.cmpi slt, %148, %120 : i64
                  %150 = scf.if %149 -> (i64) {
                    %152 = arith.addi %arg27, %c1_i64 : i64
                    %153 = arith.remui %152, %c2_i64 : i64
                    scf.yield %153 : i64
                  } else {
                    scf.yield %arg27 : i64
                  }
                  %151:7 = scf.for %arg29 = %c0_i32 to %85 step %c1_i32 iter_args(%arg30 = %128, %arg31 = %arg23, %arg32 = %141, %arg33 = %147, %arg34 = %arg26, %arg35 = %150, %arg36 = %arg28) -> (i64, i64, i64, i64, i64, i64, i64)  : i32 {
                    %152 = arith.addi %82, %arg29 : i32
                    %153 = arith.addi %152, %c1_i32 : i32
                    %154 = arith.extsi %153 : i32 to i64
                    %155 = arith.cmpi eq, %154, %93 : i64
                    %156 = arith.muli %152, %c64_i32 : i32
                    %157 = arith.extsi %156 : i32 to i64
                    %158 = arith.subi %79, %157 : i64
                    %159 = arith.select %155, %158, %c64_i64 : i64
                    %160 = arith.addi %152, %c2_i32 : i32
                    %161 = arith.extsi %160 : i32 to i64
                    %162 = arith.cmpi eq, %161, %93 : i64
                    %163 = arith.muli %153, %c64_i32 : i32
                    %164 = arith.extsi %163 : i32 to i64
                    %165 = arith.subi %79, %164 : i64
                    %166 = arith.select %162, %165, %c64_i64 : i64
                    %167 = arith.extsi %152 : i32 to i64
                    %168 = arith.cmpi eq, %arg31, %c0_i64 : i64
                    %169 = ascscalar.memref_address %alloc_16 : memref<16384xbf16, 2>
                    %170 = ascscalar.memref_address %alloc_15 : memref<16384xbf16, 2>
                    %171 = arith.select %168, %169, %170 : i64
                    %172 = arith.select %168, %c8_i64, %c9_i64 : i64
                    %173 = arith.addi %arg31, %c1_i64 : i64
                    %174 = arith.remui %173, %c2_i64 : i64
                    %175 = ascscalar.bitcast_memref %alloc_8 byte_offset = %c0_i64 : memref<16384xbf16, 4> -> memref<?xbf16, 4>
                    %176 = ascscalar.bind_local_ptr %171 : memref<1xbf16, 2>
                    %177 = arith.cmpi eq, %172, %c9_i64 : i64
                    %178 = arith.extui %177 : i1 to i64
                    asccube.sync_intra_wait mte1, %178 {cannir.sync.banks = 3 : i32}
                    %179 = arith.addi %178, %c16_i64 overflow<nsw> : i64
                    asccube.sync_intra_wait mte1, %179 {cannir.sync.banks = 3 : i32}
                    asccube.get_buf mte1, %c13_i64, 0 {cannir.buffer.generated_sync}
                    asccube.copy_l12l0b %175[%c0], %176[%c0], %c0_i32, %c0_i32, %c4_i32, %c8_i32, %c4_i32, %c4_i32 : memref<?xbf16, 4>, memref<1xbf16, 2>
                    asccube.release_buf mte1, %c13_i64, 0 {cannir.buffer.generated_sync}
                    asccube.sync_intra_arrive mte1, %178 {cannir.sync.banks = 3 : i32}
                    asccube.sync_intra_arrive mte1, %179 {cannir.sync.banks = 3 : i32}
                    %180 = arith.cmpi eq, %arg34, %c0_i64 : i64
                    %181 = ascscalar.memref_address %alloc_28 : memref<8192xbf16, 3>
                    %182 = ascscalar.memref_address %alloc_27 : memref<8192xbf16, 3>
                    %183 = arith.select %180, %181, %182 : i64
                    %184 = arith.select %180, %c8_i64, %c9_i64 : i64
                    %185 = arith.addi %arg34, %c1_i64 : i64
                    %186 = arith.remui %185, %c2_i64 : i64
                    %187 = arith.select %168, %123, %124 : i64
                    %188 = arith.select %168, %c2_i64, %c3_i64 : i64
                    %189 = ascscalar.bind_local_ptr %183 : memref<1xbf16, 3>
                    %190 = ascscalar.bind_local_ptr %187 : memref<1xbf16, 2>
                    asccube.get_buf mte1, %184, 0 {cannir.buffer.generated_sync}
                    asccube.get_buf mte1, %188, 0 {cannir.buffer.generated_sync}
                    asccube.copy_l12l0a %189[%c0], %190[%c0], %c0_i32, %c0_i32, %c4_i32, %c8_i32, %c4_i32, %c4_i32 : memref<1xbf16, 3>, memref<1xbf16, 2>
                    asccube.release_buf mte1, %188, 0 {cannir.buffer.generated_sync}
                    asccube.release_buf mte1, %184, 0 {cannir.buffer.generated_sync}
                    %191 = ascscalar.bitcast_memref %alloc_7 byte_offset = %c0_i64 : memref<8192xf32, 5> -> memref<?xf32, 5>
                    asccube.get_buf m, %c15_i64, 0 {cannir.buffer.generated_sync}
                    asccube.get_buf m, %184, 0 {cannir.buffer.generated_sync}
                    asccube.get_buf m, %c13_i64, 0 {cannir.buffer.generated_sync}
                    asccube.mmad %191[%c0], %189, %175, %c64_i32, %c128_i32, %c64_i32, %c0_i32 {c_matrix_init_val = true} : memref<?xf32, 5>, memref<1xbf16, 3>, memref<?xbf16, 4>
                    asccube.release_buf m, %184, 0 {cannir.buffer.generated_sync}
                    asccube.release_buf m, %c15_i64, 0 {cannir.buffer.generated_sync}
                    %192 = arith.cmpi eq, %186, %c0_i64 : i64
                    %193 = arith.select %192, %181, %182 : i64
                    %194 = arith.select %192, %c8_i64, %c9_i64 : i64
                    %195 = arith.addi %186, %c1_i64 overflow<nsw> : i64
                    %196 = arith.remui %195, %c2_i64 : i64
                    %197 = arith.select %168, %130, %131 : i64
                    %198 = arith.cmpi ne, %arg31, %c0_i64 : i64
                    %199 = arith.extui %198 : i1 to i64
                    %200 = ascscalar.bind_local_ptr %193 : memref<1xbf16, 3>
                    %201 = ascscalar.bind_local_ptr %197 : memref<1xbf16, 2>
                    asccube.get_buf mte1, %194, 0 {cannir.buffer.generated_sync}
                    asccube.get_buf mte1, %199, 0 {cannir.buffer.generated_sync}
                    asccube.copy_l12l0a %200[%c0], %201[%c0], %c0_i32, %c0_i32, %c4_i32, %c8_i32, %c4_i32, %c4_i32 : memref<1xbf16, 3>, memref<1xbf16, 2>
                    asccube.release_buf mte1, %199, 0 {cannir.buffer.generated_sync}
                    asccube.release_buf mte1, %194, 0 {cannir.buffer.generated_sync}
                    %202 = ascscalar.bitcast_memref %alloc_9 byte_offset = %c0_i64 : memref<8192xf32, 5> -> memref<?xf32, 5>
                    asccube.get_buf m, %c16_i64, 0 {cannir.buffer.generated_sync}
                    asccube.get_buf m, %194, 0 {cannir.buffer.generated_sync}
                    asccube.mmad %202[%c0], %200, %175, %c64_i32, %c128_i32, %c64_i32, %c0_i32 {c_matrix_init_val = true} : memref<?xf32, 5>, memref<1xbf16, 3>, memref<?xbf16, 4>
                    asccube.release_buf m, %c13_i64, 0 {cannir.buffer.generated_sync}
                    asccube.release_buf m, %194, 0 {cannir.buffer.generated_sync}
                    asccube.release_buf m, %c16_i64, 0 {cannir.buffer.generated_sync}
                    %203 = arith.select %168, %136, %137 : i64
                    %204 = arith.select %168, %c4_i64, %c5_i64 : i64
                    %205 = ascscalar.bitcast_memref %alloc_12 byte_offset = %c0_i64 : memref<8192xbf16, 4> -> memref<?xbf16, 4>
                    %206 = ascscalar.bind_local_ptr %203 : memref<1xbf16, 2>
                    asccube.get_buf mte1, %c14_i64, 0 {cannir.buffer.generated_sync}
                    asccube.get_buf mte1, %204, 0 {cannir.buffer.generated_sync}
                    asccube.copy_l12l0b %205[%c0], %206[%c0], %c0_i32, %c0_i32, %c4_i32, %c8_i32, %c4_i32, %c8_i32 {transpose = true} : memref<?xbf16, 4>, memref<1xbf16, 2>
                    asccube.release_buf mte1, %204, 0 {cannir.buffer.generated_sync}
                    asccube.release_buf mte1, %c14_i64, 0 {cannir.buffer.generated_sync}
                    %207 = arith.select %168, %142, %143 : i64
                    %208 = arith.select %168, %c6_i64, %c7_i64 : i64
                    %209 = ascscalar.bitcast_memref %alloc_13 byte_offset = %c0_i64 : memref<4096xbf16, 3> -> memref<?xbf16, 3>
                    %210 = ascscalar.bind_local_ptr %207 : memref<1xbf16, 2>
                    asccube.get_buf mte1, %c11_i64, 0 {cannir.buffer.generated_sync}
                    asccube.get_buf mte1, %208, 0 {cannir.buffer.generated_sync}
                    asccube.copy_l12l0a %209[%c0], %210[%c0], %c0_i32, %c0_i32, %c4_i32, %c4_i32, %c4_i32, %c4_i32 : memref<?xbf16, 3>, memref<1xbf16, 2>
                    asccube.release_buf mte1, %208, 0 {cannir.buffer.generated_sync}
                    asccube.release_buf mte1, %c11_i64, 0 {cannir.buffer.generated_sync}
                    %211 = arith.addi %arg29, %c1_i32 : i32
                    %212 = arith.cmpi slt, %211, %85 : i32
                    %213:2 = scf.if %212 -> (i64, i64) {
                      %262 = arith.cmpi eq, %arg30, %c0_i64 : i64
                      %263 = arith.select %262, %130, %131 : i64
                      %264 = arith.cmpi ne, %arg30, %c0_i64 : i64
                      %265 = arith.extui %264 : i1 to i64
                      %266 = arith.addi %arg30, %c1_i64 overflow<nsw> : i64
                      %267 = arith.remui %266, %c2_i64 : i64
                      %268 = arith.cmpi slt, %166, %c64_i64 : i64
                      scf.if %268 {
                        asccube.get_buf mte2, %265, 0 {cannir.buffer.generated_sync}
                        scf.for %arg37 = %c0_i64 to %c8_i64 step %c1_i64  : i64 {
                          %273 = ascscalar.bind_local_ptr %263 : memref<1xbf16, 2>
                          %274 = cannir.extract_physical_layout %arg2 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                          %275 = cannir.extract_stride %274 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                          %276 = cannir.extract_dim %275[0] : !cannir.stride<(-1, -1, -1, 1)>
                          %277 = arith.muli %95, %276 : i64
                          %278 = arith.index_cast %277 : i64 to index
                          %279 = cannir.extract_dim %275[1] : !cannir.stride<(-1, -1, -1, 1)>
                          %280 = arith.muli %109, %279 : i64
                          %281 = arith.index_cast %280 : i64 to index
                          %282 = arith.addi %278, %281 : index
                          %283 = arith.muli %154, %c64_i64 : i64
                          %284 = arith.addi %283, %94 : i64
                          %285 = cannir.extract_dim %275[2] : !cannir.stride<(-1, -1, -1, 1)>
                          %286 = arith.muli %284, %285 : i64
                          %287 = arith.index_cast %286 : i64 to index
                          %288 = arith.addi %282, %287 : index
                          %289 = arith.muli %arg37, %c16_i64 overflow<nsw> : i64
                          %290 = arith.index_cast %289 : i64 to index
                          %291 = arith.addi %288, %290 : index
                          %292 = cannir.extract_layout %arg2 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                          %293 = cannir.extract_stride %292 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                          %294 = cannir.extract_dim %293[2] : !cannir.stride<(-1, -1, -1, 1)>
                          %295 = arith.index_cast %294 : i64 to index
                          %296 = arith.muli %arg37, %c1024_i64 overflow<nsw> : i64
                          %297 = arith.muli %295, %c2 : index
                          %298 = arith.index_cast %297 : index to i64
                          %299 = arith.cmpi ugt, %166, %c0_i64 : i64
                          scf.if %299 {
                            %300 = arith.index_cast %296 : i64 to index
                            %301 = arith.trunci %166 : i64 to i32
                            asccube.copy_gm2l1 %273[%300], %alloc_0[%291], %301, %c16_i32, %298, %c1_i32, %c0_i64, %c0_i64, %c64_i32 l2_cache_ctl %c0_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
                          }
                        }
                        asccube.release_buf mte2, %265, 0 {cannir.buffer.generated_sync}
                      } else {
                        %273 = ascscalar.bind_local_ptr %263 : memref<1xbf16, 2>
                        %274 = cannir.extract_physical_layout %arg2 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                        %275 = cannir.extract_stride %274 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                        %276 = cannir.extract_dim %275[0] : !cannir.stride<(-1, -1, -1, 1)>
                        %277 = arith.muli %95, %276 : i64
                        %278 = arith.index_cast %277 : i64 to index
                        %279 = cannir.extract_dim %275[1] : !cannir.stride<(-1, -1, -1, 1)>
                        %280 = arith.muli %109, %279 : i64
                        %281 = arith.index_cast %280 : i64 to index
                        %282 = arith.addi %278, %281 : index
                        %283 = arith.muli %154, %c64_i64 : i64
                        %284 = arith.addi %283, %94 : i64
                        %285 = cannir.extract_dim %275[2] : !cannir.stride<(-1, -1, -1, 1)>
                        %286 = arith.muli %284, %285 : i64
                        %287 = arith.index_cast %286 : i64 to index
                        %288 = arith.addi %282, %287 : index
                        %289 = arith.divsi %79, %c64_i64 : i64
                        %290 = arith.minsi %154, %289 : i64
                        %291 = arith.muli %290, %c64_i64 : i64
                        %292 = arith.subi %79, %291 : i64
                        %293 = arith.minsi %292, %c64_i64 : i64
                        %294 = arith.cmpi sle, %154, %289 : i64
                        %295 = arith.select %294, %293, %c0_i64 : i64
                        %296 = cannir.extract_layout %arg2 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                        %297 = cannir.extract_stride %296 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                        %298 = cannir.extract_dim %297[2] : !cannir.stride<(-1, -1, -1, 1)>
                        %299 = arith.index_cast %298 : i64 to index
                        %300 = arith.muli %299, %c2 : index
                        %301 = arith.index_cast %300 : index to i64
                        asccube.get_buf mte2, %265, 0 {cannir.buffer.generated_sync}
                        %302 = arith.cmpi ugt, %295, %c0_i64 : i64
                        scf.if %302 {
                          %303 = arith.trunci %295 : i64 to i32
                          asccube.copy_gm2l1 %273[%c0], %alloc_0[%288], %303, %c128_i32, %301, %c1_i32, %c0_i64, %c0_i64, %c64_i32 l2_cache_ctl %c0_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
                        }
                        asccube.release_buf mte2, %265, 0 {cannir.buffer.generated_sync}
                      }
                      %269 = arith.select %262, %123, %124 : i64
                      %270 = arith.select %262, %c2_i64, %c3_i64 : i64
                      scf.if %268 {
                        asccube.get_buf mte2, %270, 0 {cannir.buffer.generated_sync}
                        scf.for %arg37 = %c0_i64 to %c8_i64 step %c1_i64  : i64 {
                          %273 = ascscalar.bind_local_ptr %269 : memref<1xbf16, 2>
                          %274 = cannir.extract_physical_layout %arg3 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                          %275 = cannir.extract_stride %274 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                          %276 = cannir.extract_dim %275[0] : !cannir.stride<(-1, -1, -1, 1)>
                          %277 = arith.muli %95, %276 : i64
                          %278 = arith.index_cast %277 : i64 to index
                          %279 = cannir.extract_dim %275[1] : !cannir.stride<(-1, -1, -1, 1)>
                          %280 = arith.muli %109, %279 : i64
                          %281 = arith.index_cast %280 : i64 to index
                          %282 = arith.addi %278, %281 : index
                          %283 = arith.muli %154, %c64_i64 : i64
                          %284 = arith.addi %283, %94 : i64
                          %285 = cannir.extract_dim %275[2] : !cannir.stride<(-1, -1, -1, 1)>
                          %286 = arith.muli %284, %285 : i64
                          %287 = arith.index_cast %286 : i64 to index
                          %288 = arith.addi %282, %287 : index
                          %289 = arith.muli %arg37, %c16_i64 overflow<nsw> : i64
                          %290 = arith.index_cast %289 : i64 to index
                          %291 = arith.addi %288, %290 : index
                          %292 = cannir.extract_layout %arg3 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                          %293 = cannir.extract_stride %292 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                          %294 = cannir.extract_dim %293[2] : !cannir.stride<(-1, -1, -1, 1)>
                          %295 = arith.index_cast %294 : i64 to index
                          %296 = arith.muli %arg37, %c1024_i64 overflow<nsw> : i64
                          %297 = arith.muli %295, %c2 : index
                          %298 = arith.index_cast %297 : index to i64
                          %299 = arith.cmpi ugt, %166, %c0_i64 : i64
                          scf.if %299 {
                            %300 = arith.index_cast %296 : i64 to index
                            %301 = arith.trunci %166 : i64 to i32
                            asccube.copy_gm2l1 %273[%300], %alloc_1[%291], %301, %c16_i32, %298, %c1_i32, %c0_i64, %c0_i64, %c64_i32 l2_cache_ctl %c0_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
                          }
                        }
                        asccube.release_buf mte2, %270, 0 {cannir.buffer.generated_sync}
                      } else {
                        %273 = ascscalar.bind_local_ptr %269 : memref<1xbf16, 2>
                        %274 = cannir.extract_physical_layout %arg3 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                        %275 = cannir.extract_stride %274 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                        %276 = cannir.extract_dim %275[0] : !cannir.stride<(-1, -1, -1, 1)>
                        %277 = arith.muli %95, %276 : i64
                        %278 = arith.index_cast %277 : i64 to index
                        %279 = cannir.extract_dim %275[1] : !cannir.stride<(-1, -1, -1, 1)>
                        %280 = arith.muli %109, %279 : i64
                        %281 = arith.index_cast %280 : i64 to index
                        %282 = arith.addi %278, %281 : index
                        %283 = arith.muli %154, %c64_i64 : i64
                        %284 = arith.addi %283, %94 : i64
                        %285 = cannir.extract_dim %275[2] : !cannir.stride<(-1, -1, -1, 1)>
                        %286 = arith.muli %284, %285 : i64
                        %287 = arith.index_cast %286 : i64 to index
                        %288 = arith.addi %282, %287 : index
                        %289 = arith.divsi %79, %c64_i64 : i64
                        %290 = arith.minsi %154, %289 : i64
                        %291 = arith.muli %290, %c64_i64 : i64
                        %292 = arith.subi %79, %291 : i64
                        %293 = arith.minsi %292, %c64_i64 : i64
                        %294 = arith.cmpi sle, %154, %289 : i64
                        %295 = arith.select %294, %293, %c0_i64 : i64
                        %296 = cannir.extract_layout %arg3 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                        %297 = cannir.extract_stride %296 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                        %298 = cannir.extract_dim %297[2] : !cannir.stride<(-1, -1, -1, 1)>
                        %299 = arith.index_cast %298 : i64 to index
                        %300 = arith.muli %299, %c2 : index
                        %301 = arith.index_cast %300 : index to i64
                        asccube.get_buf mte2, %270, 0 {cannir.buffer.generated_sync}
                        %302 = arith.cmpi ugt, %295, %c0_i64 : i64
                        scf.if %302 {
                          %303 = arith.trunci %295 : i64 to i32
                          asccube.copy_gm2l1 %273[%c0], %alloc_1[%288], %303, %c128_i32, %301, %c1_i32, %c0_i64, %c0_i64, %c64_i32 l2_cache_ctl %c0_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
                        }
                        asccube.release_buf mte2, %270, 0 {cannir.buffer.generated_sync}
                      }
                      %271 = arith.cmpi slt, %148, %166 : i64
                      %272 = scf.if %271 -> (i64) {
                        %273 = arith.addi %arg35, %c1_i64 : i64
                        %274 = arith.remui %273, %c2_i64 : i64
                        scf.yield %274 : i64
                      } else {
                        scf.yield %arg35 : i64
                      }
                      scf.yield %267, %272 : i64, i64
                    } else {
                      scf.yield %arg30, %arg35 : i64, i64
                    }
                    %214 = ascscalar.bitcast_memref %alloc_6 byte_offset = %c0_i64 : memref<4096xf32, 1> -> memref<?xf32, 1>
                    asccube.sync_intra_wait fixpipe, %c5_i64 {cannir.sync.banks = 3 : i32}
                    asccube.sync_intra_wait fixpipe, %c21_i64 {cannir.sync.banks = 3 : i32}
                    asccube.get_buf fixpipe, %c15_i64, 0 {cannir.buffer.generated_sync}
                    asccube.copy_l0c2ub %214[%c0], %191[%c0], %c64_i32, %c64_i32, %c64_i32, %c64_i32, %c0_i32 sub_block_id %c0_i32 {dual_dst_ctl = 1 : i32, nz2nd_en = true, relu_en = false} : memref<?xf32, 1>, memref<?xf32, 5>
                    asccube.release_buf fixpipe, %c15_i64, 0 {cannir.buffer.generated_sync}
                    asccube.sync_intra_arrive fixpipe, %c5_i64 {cannir.sync.banks = 3 : i32}
                    asccube.sync_intra_arrive fixpipe, %c21_i64 {cannir.sync.banks = 3 : i32}
                    %215 = ascscalar.memref_address %alloc_22 : memref<8192xbf16, 2>
                    %216 = ascscalar.memref_address %alloc_21 : memref<8192xbf16, 2>
                    %217 = arith.select %168, %215, %216 : i64
                    %218 = arith.select %168, %c10_i64, %c11_i64 : i64
                    %219 = arith.subi %159, %148 : i64
                    %220 = arith.cmpi slt, %219, %c0_i64 : i64
                    %221 = arith.select %220, %c0_i64, %219 : i64
                    %222 = arith.cmpi sgt, %221, %c32_i64 : i64
                    %223 = arith.select %222, %c32_i64, %221 : i64
                    %224 = arith.cmpi eq, %223, %c32_i64 : i64
                    %225 = scf.if %224 -> (i64) {
                      %262 = arith.addi %arg36, %c1_i64 : i64
                      %263 = arith.remui %262, %c2_i64 : i64
                      scf.yield %263 : i64
                    } else {
                      scf.yield %arg36 : i64
                    }
                    %226 = arith.cmpi sgt, %223, %c0_i64 : i64
                    %227 = arith.cmpi slt, %223, %c32_i64 : i64
                    %228 = arith.select %226, %227, %false : i1
                    %229 = scf.if %228 -> (i64) {
                      %262 = arith.addi %225, %c1_i64 : i64
                      %263 = arith.remui %262, %c2_i64 : i64
                      scf.yield %263 : i64
                    } else {
                      scf.yield %225 : i64
                    }
                    %230 = ascscalar.bind_local_ptr %217 : memref<1xbf16, 2>
                    %231 = arith.cmpi eq, %218, %c11_i64 : i64
                    %232 = arith.select %231, %c3_i64, %c2_i64 : i64
                    %233 = ascscalar.bitcast_memref %alloc_11 byte_offset = %c0_i64 : memref<8192xbf16, 3> -> memref<?xbf16, 3>
                    asccube.sync_intra_wait mte1, %232 {cannir.sync.banks = 3 : i32}
                    %234 = arith.addi %232, %c16_i64 overflow<nsw> : i64
                    asccube.sync_intra_wait mte1, %234 {cannir.sync.banks = 3 : i32}
                    asccube.get_buf mte1, %c10_i64, 0 {cannir.buffer.generated_sync}
                    asccube.copy_l12l0a %233[%c0], %230[%c0], %c0_i32, %c0_i32, %c4_i32, %c4_i32, %c4_i32, %c4_i32 {transpose = true} : memref<?xbf16, 3>, memref<1xbf16, 2>
                    asccube.release_buf mte1, %c10_i64, 0 {cannir.buffer.generated_sync}
                    %235 = ascscalar.bitcast_memref %alloc_10 byte_offset = %c0_i64 : memref<16384xf32, 5> -> memref<?xf32, 5>
                    asccube.get_buf m, %c17_i64, 0 {cannir.buffer.generated_sync}
                    asccube.get_buf m, %c10_i64, 0 {cannir.buffer.generated_sync}
                    asccube.get_buf m, %c14_i64, 0 {cannir.buffer.generated_sync}
                    asccube.mmad %235[%c0], %233, %205, %c64_i32, %c64_i32, %c128_i32, %c0_i32 {c_matrix_init_val = true} : memref<?xf32, 5>, memref<?xbf16, 3>, memref<?xbf16, 4>
                    asccube.release_buf m, %c14_i64, 0 {cannir.buffer.generated_sync}
                    asccube.release_buf m, %c10_i64, 0 {cannir.buffer.generated_sync}
                    asccube.release_buf m, %c17_i64, 0 {cannir.buffer.generated_sync}
                    %236 = ascscalar.bitcast_memref %alloc_5 byte_offset = %c0_i64 : memref<8192xf32, 1> -> memref<?xf32, 1>
                    asccube.sync_intra_wait fixpipe, %c4_i64 {cannir.sync.banks = 3 : i32}
                    asccube.sync_intra_wait fixpipe, %c20_i64 {cannir.sync.banks = 3 : i32}
                    asccube.get_buf fixpipe, %c17_i64, 0 {cannir.buffer.generated_sync}
                    asccube.copy_l0c2ub %236[%c0], %235[%c0], %c128_i32, %c64_i32, %c64_i32, %c64_i32, %c0_i32 sub_block_id %c0_i32 {dual_dst_ctl = 2 : i32, nz2nd_en = true, relu_en = false} : memref<?xf32, 1>, memref<?xf32, 5>
                    asccube.release_buf fixpipe, %c17_i64, 0 {cannir.buffer.generated_sync}
                    asccube.sync_intra_arrive fixpipe, %c4_i64 {cannir.sync.banks = 3 : i32}
                    asccube.sync_intra_arrive fixpipe, %c20_i64 {cannir.sync.banks = 3 : i32}
                    %237 = ascscalar.bitcast_memref %alloc_14 byte_offset = %c0_i64 : memref<8192xbf16, 4> -> memref<?xbf16, 4>
                    asccube.get_buf mte1, %c12_i64, 0 {cannir.buffer.generated_sync}
                    asccube.copy_l12l0b %237[%c0], %230[%c0], %c0_i32, %c0_i32, %c4_i32, %c4_i32, %c4_i32, %c4_i32 {transpose = true} : memref<?xbf16, 4>, memref<1xbf16, 2>
                    asccube.release_buf mte1, %c12_i64, 0 {cannir.buffer.generated_sync}
                    asccube.sync_intra_arrive mte1, %232 {cannir.sync.banks = 3 : i32}
                    asccube.sync_intra_arrive mte1, %234 {cannir.sync.banks = 3 : i32}
                    asccube.get_buf m, %c16_i64, 0 {cannir.buffer.generated_sync}
                    asccube.get_buf m, %c11_i64, 0 {cannir.buffer.generated_sync}
                    asccube.get_buf m, %c12_i64, 0 {cannir.buffer.generated_sync}
                    asccube.mmad %202[%c0], %209, %237, %c64_i32, %c64_i32, %c64_i32, %c0_i32 {c_matrix_init_val = false} : memref<?xf32, 5>, memref<?xbf16, 3>, memref<?xbf16, 4>
                    asccube.release_buf m, %c12_i64, 0 {cannir.buffer.generated_sync}
                    asccube.release_buf m, %c11_i64, 0 {cannir.buffer.generated_sync}
                    asccube.release_buf m, %c16_i64, 0 {cannir.buffer.generated_sync}
                    %238 = scf.if %212 -> (i64) {
                      %262 = arith.cmpi eq, %arg32, %c0_i64 : i64
                      %263 = arith.select %262, %136, %137 : i64
                      %264 = arith.select %262, %c4_i64, %c5_i64 : i64
                      %265 = arith.addi %arg32, %c1_i64 overflow<nsw> : i64
                      %266 = arith.remui %265, %c2_i64 : i64
                      %267 = arith.cmpi slt, %166, %c64_i64 : i64
                      scf.if %267 {
                        asccube.get_buf mte2, %264, 0 {cannir.buffer.generated_sync}
                        scf.for %arg37 = %c0_i64 to %c8_i64 step %c1_i64  : i64 {
                          %270 = ascscalar.bind_local_ptr %263 : memref<1xbf16, 2>
                          %271 = cannir.extract_physical_layout %arg0 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                          %272 = cannir.extract_stride %271 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                          %273 = cannir.extract_dim %272[0] : !cannir.stride<(-1, -1, -1, 1)>
                          %274 = arith.muli %95, %273 : i64
                          %275 = arith.index_cast %274 : i64 to index
                          %276 = cannir.extract_dim %272[1] : !cannir.stride<(-1, -1, -1, 1)>
                          %277 = arith.muli %109, %276 : i64
                          %278 = arith.index_cast %277 : i64 to index
                          %279 = arith.addi %275, %278 : index
                          %280 = arith.muli %154, %c64_i64 : i64
                          %281 = arith.addi %280, %94 : i64
                          %282 = cannir.extract_dim %272[2] : !cannir.stride<(-1, -1, -1, 1)>
                          %283 = arith.muli %281, %282 : i64
                          %284 = arith.index_cast %283 : i64 to index
                          %285 = arith.addi %279, %284 : index
                          %286 = arith.muli %arg37, %c16_i64 overflow<nsw> : i64
                          %287 = arith.index_cast %286 : i64 to index
                          %288 = arith.addi %285, %287 : index
                          %289 = cannir.extract_layout %arg0 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                          %290 = cannir.extract_stride %289 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                          %291 = cannir.extract_dim %290[2] : !cannir.stride<(-1, -1, -1, 1)>
                          %292 = arith.index_cast %291 : i64 to index
                          %293 = arith.muli %arg37, %c1024_i64 overflow<nsw> : i64
                          %294 = arith.muli %292, %c2 : index
                          %295 = arith.index_cast %294 : index to i64
                          %296 = arith.cmpi ugt, %166, %c0_i64 : i64
                          scf.if %296 {
                            %297 = arith.index_cast %293 : i64 to index
                            %298 = arith.trunci %166 : i64 to i32
                            asccube.copy_gm2l1 %270[%297], %alloc[%288], %298, %c16_i32, %295, %c1_i32, %c0_i64, %c0_i64, %c64_i32 l2_cache_ctl %c1_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
                          }
                        }
                        asccube.release_buf mte2, %264, 0 {cannir.buffer.generated_sync}
                      } else {
                        %270 = ascscalar.bind_local_ptr %263 : memref<1xbf16, 2>
                        %271 = cannir.extract_physical_layout %arg0 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                        %272 = cannir.extract_stride %271 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                        %273 = cannir.extract_dim %272[0] : !cannir.stride<(-1, -1, -1, 1)>
                        %274 = arith.muli %95, %273 : i64
                        %275 = arith.index_cast %274 : i64 to index
                        %276 = cannir.extract_dim %272[1] : !cannir.stride<(-1, -1, -1, 1)>
                        %277 = arith.muli %109, %276 : i64
                        %278 = arith.index_cast %277 : i64 to index
                        %279 = arith.addi %275, %278 : index
                        %280 = arith.muli %154, %c64_i64 : i64
                        %281 = arith.addi %280, %94 : i64
                        %282 = cannir.extract_dim %272[2] : !cannir.stride<(-1, -1, -1, 1)>
                        %283 = arith.muli %281, %282 : i64
                        %284 = arith.index_cast %283 : i64 to index
                        %285 = arith.addi %279, %284 : index
                        %286 = arith.divsi %79, %c64_i64 : i64
                        %287 = arith.minsi %154, %286 : i64
                        %288 = arith.muli %287, %c64_i64 : i64
                        %289 = arith.subi %79, %288 : i64
                        %290 = arith.minsi %289, %c64_i64 : i64
                        %291 = arith.cmpi sle, %154, %286 : i64
                        %292 = arith.select %291, %290, %c0_i64 : i64
                        %293 = cannir.extract_layout %arg0 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                        %294 = cannir.extract_stride %293 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                        %295 = cannir.extract_dim %294[2] : !cannir.stride<(-1, -1, -1, 1)>
                        %296 = arith.index_cast %295 : i64 to index
                        %297 = arith.muli %296, %c2 : index
                        %298 = arith.index_cast %297 : index to i64
                        asccube.get_buf mte2, %264, 0 {cannir.buffer.generated_sync}
                        %299 = arith.cmpi ugt, %292, %c0_i64 : i64
                        scf.if %299 {
                          %300 = arith.trunci %292 : i64 to i32
                          asccube.copy_gm2l1 %270[%c0], %alloc[%285], %300, %c128_i32, %298, %c1_i32, %c0_i64, %c0_i64, %c64_i32 l2_cache_ctl %c1_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
                        }
                        asccube.release_buf mte2, %264, 0 {cannir.buffer.generated_sync}
                      }
                      %268 = arith.select %262, %142, %143 : i64
                      %269 = arith.select %262, %c6_i64, %c7_i64 : i64
                      scf.if %267 {
                        asccube.get_buf mte2, %269, 0 {cannir.buffer.generated_sync}
                        scf.for %arg37 = %c0_i64 to %c4_i64 step %c1_i64  : i64 {
                          %270 = ascscalar.bind_local_ptr %268 : memref<1xbf16, 2>
                          %271 = cannir.extract_physical_layout %arg5 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 64, 64)>, <(-1, -1, 4096, 64, 1)>>, <<(-1, -1, -1, 64, 64)>, <(-1, -1, 4096, 64, 1)>>, identity>
                          %272 = cannir.extract_stride %271 : <<(-1, -1, -1, 64, 64)>, <(-1, -1, 4096, 64, 1)>> -> <(-1, -1, 4096, 64, 1)>
                          %273 = cannir.extract_dim %272[0] : !cannir.stride<(-1, -1, 4096, 64, 1)>
                          %274 = arith.muli %112, %273 : i64
                          %275 = arith.index_cast %274 : i64 to index
                          %276 = cannir.extract_dim %272[1] : !cannir.stride<(-1, -1, 4096, 64, 1)>
                          %277 = arith.muli %109, %276 : i64
                          %278 = arith.index_cast %277 : i64 to index
                          %279 = arith.addi %275, %278 : index
                          %280 = arith.muli %154, %c4096_i64 : i64
                          %281 = arith.index_cast %280 : i64 to index
                          %282 = arith.addi %279, %281 : index
                          %283 = arith.muli %arg37, %c16_i64 overflow<nsw> : i64
                          %284 = arith.index_cast %283 : i64 to index
                          %285 = arith.addi %282, %284 : index
                          %286 = arith.muli %arg37, %c1024_i64 overflow<nsw> : i64
                          %287 = arith.cmpi ugt, %166, %c0_i64 : i64
                          scf.if %287 {
                            %288 = arith.index_cast %286 : i64 to index
                            %289 = arith.trunci %166 : i64 to i32
                            asccube.copy_gm2l1 %270[%288], %alloc_2[%285], %289, %c16_i32, %c128_i64, %c1_i32, %c0_i64, %c0_i64, %c64_i32 l2_cache_ctl %c1_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
                          }
                        }
                        asccube.release_buf mte2, %269, 0 {cannir.buffer.generated_sync}
                      } else {
                        %270 = ascscalar.bind_local_ptr %268 : memref<1xbf16, 2>
                        %271 = cannir.extract_physical_layout %arg5 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 64, 64)>, <(-1, -1, 4096, 64, 1)>>, <<(-1, -1, -1, 64, 64)>, <(-1, -1, 4096, 64, 1)>>, identity>
                        %272 = cannir.extract_stride %271 : <<(-1, -1, -1, 64, 64)>, <(-1, -1, 4096, 64, 1)>> -> <(-1, -1, 4096, 64, 1)>
                        %273 = cannir.extract_dim %272[0] : !cannir.stride<(-1, -1, 4096, 64, 1)>
                        %274 = arith.muli %112, %273 : i64
                        %275 = arith.index_cast %274 : i64 to index
                        %276 = cannir.extract_dim %272[1] : !cannir.stride<(-1, -1, 4096, 64, 1)>
                        %277 = arith.muli %109, %276 : i64
                        %278 = arith.index_cast %277 : i64 to index
                        %279 = arith.addi %275, %278 : index
                        %280 = arith.muli %154, %c4096_i64 : i64
                        %281 = arith.index_cast %280 : i64 to index
                        %282 = arith.addi %279, %281 : index
                        asccube.get_buf mte2, %269, 0 {cannir.buffer.generated_sync}
                        asccube.copy_gm2l1 %270[%c0], %alloc_2[%282], %c64_i32, %c64_i32, %c128_i64, %c1_i32, %c0_i64, %c0_i64, %c64_i32 l2_cache_ctl %c1_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
                        asccube.release_buf mte2, %269, 0 {cannir.buffer.generated_sync}
                      }
                      scf.yield %266 : i64
                    } else {
                      scf.yield %arg32 : i64
                    }
                    %239 = cannir.extract_physical_layout %arg6 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                    %240 = cannir.extract_stride %239 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                    %241 = cannir.extract_dim %240[0] : !cannir.stride<(-1, -1, -1, 1)>
                    %242 = arith.muli %95, %241 : i64
                    %243 = arith.index_cast %242 : i64 to index
                    %244 = cannir.extract_dim %240[1] : !cannir.stride<(-1, -1, -1, 1)>
                    %245 = arith.muli %109, %244 : i64
                    %246 = arith.index_cast %245 : i64 to index
                    %247 = arith.addi %243, %246 : index
                    %248 = arith.muli %167, %c64_i64 : i64
                    %249 = arith.addi %248, %94 : i64
                    %250 = cannir.extract_dim %240[2] : !cannir.stride<(-1, -1, -1, 1)>
                    %251 = arith.muli %249, %250 : i64
                    %252 = arith.index_cast %251 : i64 to index
                    %253 = arith.addi %247, %252 : index
                    %254 = arith.muli %111, %c64_i64 overflow<nsw> : i64
                    %255 = arith.index_cast %254 : i64 to index
                    %256 = arith.addi %253, %255 : index
                    %257 = cannir.extract_layout %arg6 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                    %258 = cannir.extract_stride %257 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                    %259 = cannir.extract_dim %258[2] : !cannir.stride<(-1, -1, -1, 1)>
                    asccube.get_buf fixpipe, %c16_i64, 0 {cannir.buffer.generated_sync}
                    %260 = arith.trunci %259 : i64 to i32
                    asccube.copy_l0c2gm %alloc_3[%256], %202[%c0], %c64_i32, %c64_i32, %260, %c64_i32, %c0_i32 l2_cache_ctl %c4_i32 {nz2nd_en = true, relu_en = false} : memref<1xbf16>, memref<?xf32, 5>
                    asccube.release_buf fixpipe, %c16_i64, 0 {cannir.buffer.generated_sync}
                    %261 = scf.if %212 -> (i64) {
                      %262 = arith.addi %arg33, %c1_i64 overflow<nsw> : i64
                      %263 = arith.remui %262, %c2_i64 : i64
                      scf.yield %263 : i64
                    } else {
                      scf.yield %arg33 : i64
                    }
                    scf.yield %213#0, %174, %238, %261, %196, %213#1, %229 : i64, i64, i64, i64, i64, i64, i64
                  }
                  scf.yield %151#0, %151#1, %151#2, %151#3, %151#4, %151#5, %151#6 : i64, i64, i64, i64, i64, i64, i64
                }
                scf.yield %99#0, %99#1, %99#2, %99#3, %99#4, %99#5, %99#6 : i64, i64, i64, i64, i64, i64, i64
              } else {
                scf.yield %arg14, %arg15, %arg16, %arg17, %arg18, %arg19, %arg20 : i64, i64, i64, i64, i64, i64, i64
              }
              scf.yield %98#0, %98#1, %98#2, %98#3, %98#4, %98#5, %98#6 : i64, i64, i64, i64, i64, i64, i64
            }
          } else {
            %49 = arith.cmpi eq, %44, %c4_i32 : i32
            scf.if %49 {
              %50:7 = scf.for %arg13 = %c0_i32 to %21 step %c1_i32 iter_args(%arg14 = %c0_i64, %arg15 = %c0_i64, %arg16 = %c0_i64, %arg17 = %c0_i64, %arg18 = %c0_i64, %arg19 = %c0_i64, %arg20 = %c0_i64) -> (i64, i64, i64, i64, i64, i64, i64)  : i32 {
                %51 = arith.addi %32, %arg13 : i32
                %52 = arith.index_cast %51 : i32 to index
                %53 = memref.load %alloc_4[%52] : memref<1xi32>
                %54 = arith.addi %51, %c1_i32 : i32
                %55 = arith.index_cast %54 : i32 to index
                %56 = memref.load %alloc_4[%55] : memref<1xi32>
                %57 = arith.cmpi slt, %39, %53 : i32
                %58 = arith.select %57, %53, %39 : i32
                %59 = arith.cmpi sgt, %42, %56 : i32
                %60 = arith.select %59, %56, %42 : i32
                %61 = arith.cmpi slt, %58, %60 : i32
                %62 = arith.subi %58, %53 : i32
                %63 = arith.subi %60, %53 : i32
                %64 = arith.extsi %62 : i32 to i64
                %65 = arith.extsi %63 : i32 to i64
                %66 = arith.select %61, %64, %c0_i64 : i64
                %67 = arith.select %61, %65, %c0_i64 : i64
                %68 = arith.addi %22, %arg13 : i32
                %69 = arith.index_cast %68 : i32 to index
                %70 = memref.load %alloc_4[%69] : memref<1xi32>
                %71 = arith.addi %23, %arg13 : i32
                %72 = arith.index_cast %71 : i32 to index
                %73 = memref.load %alloc_4[%72] : memref<1xi32>
                %74 = arith.addi %24, %arg13 : i32
                %75 = arith.index_cast %74 : i32 to index
                %76 = memref.load %alloc_4[%75] : memref<1xi32>
                %77 = arith.addi %25, %arg13 : i32
                %78 = arith.index_cast %77 : i32 to index
                %79 = memref.load %alloc_4[%78] : memref<1xi32>
                %80 = arith.extsi %79 : i32 to i64
                %81 = arith.addi %27, %arg13 : i32
                %82 = arith.index_cast %81 : i32 to index
                %83 = memref.load %alloc_4[%82] : memref<1xi32>
                %84 = arith.addi %28, %arg13 : i32
                %85 = arith.index_cast %84 : i32 to index
                %86 = memref.load %alloc_4[%85] : memref<1xi32>
                %87 = arith.addi %80, %c63_i64 : i64
                %88 = arith.remsi %87, %c64_i64 : i64
                %89 = arith.cmpi ne, %88, %c0_i64 : i64
                %90 = arith.cmpi slt, %88, %c0_i64 : i64
                %91 = arith.andi %89, %90 : i1
                %92 = arith.divsi %87, %c64_i64 : i64
                %93 = arith.extui %91 : i1 to i64
                %94 = arith.subi %92, %93 : i64
                %95 = arith.extsi %76 : i32 to i64
                %96 = arith.extsi %73 : i32 to i64
                %97 = ascscalar.get_subblock_id
                %98 = arith.cmpi slt, %0, %1 : i64
                %99:7 = scf.if %98 -> (i64, i64, i64, i64, i64, i64, i64) {
                  %100:7 = scf.for %arg21 = %66 to %67 step %c1_i64 iter_args(%arg22 = %arg14, %arg23 = %arg15, %arg24 = %arg16, %arg25 = %arg17, %arg26 = %arg18, %arg27 = %arg19, %arg28 = %arg20) -> (i64, i64, i64, i64, i64, i64, i64)  : i64 {
                    %101 = arith.remsi %arg21, %c4_i64 : i64
                    %102 = arith.cmpi slt, %101, %c0_i64 : i64
                    %103 = arith.divsi %arg21, %c4_i64 : i64
                    %104 = arith.extui %102 : i1 to i64
                    %105 = arith.subi %103, %104 : i64
                    %106 = arith.maxsi %7, %c1_i64 : i64
                    %107 = arith.remsi %105, %106 : i64
                    %108 = arith.cmpi slt, %107, %c0_i64 : i64
                    %109 = arith.select %108, %106, %c0_i64 : i64
                    %110 = arith.addi %107, %109 : i64
                    %111 = arith.select %102, %c4_i64, %c0_i64 : i64
                    %112 = arith.addi %101, %111 overflow<nsw> : i64
                    %113 = arith.extsi %70 : i32 to i64
                    %114 = arith.addi %83, %86 : i32
                    %115 = arith.extsi %114 : i32 to i64
                    %116 = arith.cmpi eq, %115, %94 : i64
                    %117 = arith.muli %83, %c64_i32 : i32
                    %118 = arith.extsi %117 : i32 to i64
                    %119 = arith.subi %80, %118 : i64
                    %120 = arith.minsi %119, %c64_i64 : i64
                    %121 = arith.select %116, %120, %c64_i64 : i64
                    %122 = arith.extsi %83 : i32 to i64
                    %123 = arith.cmpi eq, %arg22, %c0_i64 : i64
                    %124 = ascscalar.memref_address %alloc_24 : memref<8192xbf16, 2>
                    %125 = ascscalar.memref_address %alloc_23 : memref<8192xbf16, 2>
                    %126 = arith.select %123, %124, %125 : i64
                    %127 = arith.select %123, %c2_i64, %c3_i64 : i64
                    %128 = arith.addi %arg22, %c1_i64 : i64
                    %129 = arith.remui %128, %c2_i64 : i64
                    %130 = arith.cmpi slt, %121, %c64_i64 : i64
                    scf.if %130 {
                      asccube.get_buf mte2, %127, 0 {cannir.buffer.generated_sync}
                      scf.for %arg29 = %c0_i64 to %c8_i64 step %c1_i64  : i64 {
                        %153 = ascscalar.bind_local_ptr %126 : memref<1xbf16, 2>
                        %154 = cannir.extract_physical_layout %arg3 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                        %155 = cannir.extract_stride %154 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                        %156 = cannir.extract_dim %155[0] : !cannir.stride<(-1, -1, -1, 1)>
                        %157 = arith.muli %96, %156 : i64
                        %158 = arith.index_cast %157 : i64 to index
                        %159 = cannir.extract_dim %155[1] : !cannir.stride<(-1, -1, -1, 1)>
                        %160 = arith.muli %110, %159 : i64
                        %161 = arith.index_cast %160 : i64 to index
                        %162 = arith.addi %158, %161 : index
                        %163 = arith.muli %122, %c64_i64 : i64
                        %164 = arith.addi %163, %95 : i64
                        %165 = cannir.extract_dim %155[2] : !cannir.stride<(-1, -1, -1, 1)>
                        %166 = arith.muli %164, %165 : i64
                        %167 = arith.index_cast %166 : i64 to index
                        %168 = arith.addi %162, %167 : index
                        %169 = arith.muli %arg29, %c16_i64 overflow<nsw> : i64
                        %170 = arith.index_cast %169 : i64 to index
                        %171 = arith.addi %168, %170 : index
                        %172 = cannir.extract_layout %arg3 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                        %173 = cannir.extract_stride %172 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                        %174 = cannir.extract_dim %173[2] : !cannir.stride<(-1, -1, -1, 1)>
                        %175 = arith.index_cast %174 : i64 to index
                        %176 = arith.muli %arg29, %c1024_i64 overflow<nsw> : i64
                        %177 = arith.muli %175, %c2 : index
                        %178 = arith.index_cast %177 : index to i64
                        %179 = arith.cmpi ugt, %121, %c0_i64 : i64
                        scf.if %179 {
                          %180 = arith.index_cast %176 : i64 to index
                          %181 = arith.trunci %121 : i64 to i32
                          asccube.copy_gm2l1 %153[%180], %alloc_1[%171], %181, %c16_i32, %178, %c1_i32, %c0_i64, %c0_i64, %c64_i32 l2_cache_ctl %c0_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
                        }
                      }
                      asccube.release_buf mte2, %127, 0 {cannir.buffer.generated_sync}
                    } else {
                      %153 = ascscalar.bind_local_ptr %126 : memref<1xbf16, 2>
                      %154 = cannir.extract_physical_layout %arg3 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                      %155 = cannir.extract_stride %154 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                      %156 = cannir.extract_dim %155[0] : !cannir.stride<(-1, -1, -1, 1)>
                      %157 = arith.muli %96, %156 : i64
                      %158 = arith.index_cast %157 : i64 to index
                      %159 = cannir.extract_dim %155[1] : !cannir.stride<(-1, -1, -1, 1)>
                      %160 = arith.muli %110, %159 : i64
                      %161 = arith.index_cast %160 : i64 to index
                      %162 = arith.addi %158, %161 : index
                      %163 = arith.muli %122, %c64_i64 : i64
                      %164 = arith.addi %163, %95 : i64
                      %165 = cannir.extract_dim %155[2] : !cannir.stride<(-1, -1, -1, 1)>
                      %166 = arith.muli %164, %165 : i64
                      %167 = arith.index_cast %166 : i64 to index
                      %168 = arith.addi %162, %167 : index
                      %169 = arith.divsi %80, %c64_i64 : i64
                      %170 = arith.minsi %122, %169 : i64
                      %171 = arith.muli %170, %c64_i64 : i64
                      %172 = arith.subi %80, %171 : i64
                      %173 = arith.minsi %172, %c64_i64 : i64
                      %174 = arith.cmpi sle, %122, %169 : i64
                      %175 = arith.select %174, %173, %c0_i64 : i64
                      %176 = cannir.extract_layout %arg3 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                      %177 = cannir.extract_stride %176 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                      %178 = cannir.extract_dim %177[2] : !cannir.stride<(-1, -1, -1, 1)>
                      %179 = arith.index_cast %178 : i64 to index
                      %180 = arith.muli %179, %c2 : index
                      %181 = arith.index_cast %180 : index to i64
                      asccube.get_buf mte2, %127, 0 {cannir.buffer.generated_sync}
                      %182 = arith.cmpi ugt, %175, %c0_i64 : i64
                      scf.if %182 {
                        %183 = arith.trunci %175 : i64 to i32
                        asccube.copy_gm2l1 %153[%c0], %alloc_1[%168], %183, %c128_i32, %181, %c1_i32, %c0_i64, %c0_i64, %c64_i32 l2_cache_ctl %c0_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
                      }
                      asccube.release_buf mte2, %127, 0 {cannir.buffer.generated_sync}
                    }
                    %131 = ascscalar.memref_address %alloc_26 : memref<8192xbf16, 2>
                    %132 = ascscalar.memref_address %alloc_25 : memref<8192xbf16, 2>
                    %133 = arith.select %123, %131, %132 : i64
                    %134 = arith.cmpi ne, %arg22, %c0_i64 : i64
                    %135 = arith.extui %134 : i1 to i64
                    scf.if %130 {
                      asccube.get_buf mte2, %135, 0 {cannir.buffer.generated_sync}
                      scf.for %arg29 = %c0_i64 to %c8_i64 step %c1_i64  : i64 {
                        %153 = ascscalar.bind_local_ptr %133 : memref<1xbf16, 2>
                        %154 = cannir.extract_physical_layout %arg2 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                        %155 = cannir.extract_stride %154 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                        %156 = cannir.extract_dim %155[0] : !cannir.stride<(-1, -1, -1, 1)>
                        %157 = arith.muli %96, %156 : i64
                        %158 = arith.index_cast %157 : i64 to index
                        %159 = cannir.extract_dim %155[1] : !cannir.stride<(-1, -1, -1, 1)>
                        %160 = arith.muli %110, %159 : i64
                        %161 = arith.index_cast %160 : i64 to index
                        %162 = arith.addi %158, %161 : index
                        %163 = arith.muli %122, %c64_i64 : i64
                        %164 = arith.addi %163, %95 : i64
                        %165 = cannir.extract_dim %155[2] : !cannir.stride<(-1, -1, -1, 1)>
                        %166 = arith.muli %164, %165 : i64
                        %167 = arith.index_cast %166 : i64 to index
                        %168 = arith.addi %162, %167 : index
                        %169 = arith.muli %arg29, %c16_i64 overflow<nsw> : i64
                        %170 = arith.index_cast %169 : i64 to index
                        %171 = arith.addi %168, %170 : index
                        %172 = cannir.extract_layout %arg2 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                        %173 = cannir.extract_stride %172 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                        %174 = cannir.extract_dim %173[2] : !cannir.stride<(-1, -1, -1, 1)>
                        %175 = arith.index_cast %174 : i64 to index
                        %176 = arith.muli %arg29, %c1024_i64 overflow<nsw> : i64
                        %177 = arith.muli %175, %c2 : index
                        %178 = arith.index_cast %177 : index to i64
                        %179 = arith.cmpi ugt, %121, %c0_i64 : i64
                        scf.if %179 {
                          %180 = arith.index_cast %176 : i64 to index
                          %181 = arith.trunci %121 : i64 to i32
                          asccube.copy_gm2l1 %153[%180], %alloc_0[%171], %181, %c16_i32, %178, %c1_i32, %c0_i64, %c0_i64, %c64_i32 l2_cache_ctl %c0_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
                        }
                      }
                      asccube.release_buf mte2, %135, 0 {cannir.buffer.generated_sync}
                    } else {
                      %153 = ascscalar.bind_local_ptr %133 : memref<1xbf16, 2>
                      %154 = cannir.extract_physical_layout %arg2 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                      %155 = cannir.extract_stride %154 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                      %156 = cannir.extract_dim %155[0] : !cannir.stride<(-1, -1, -1, 1)>
                      %157 = arith.muli %96, %156 : i64
                      %158 = arith.index_cast %157 : i64 to index
                      %159 = cannir.extract_dim %155[1] : !cannir.stride<(-1, -1, -1, 1)>
                      %160 = arith.muli %110, %159 : i64
                      %161 = arith.index_cast %160 : i64 to index
                      %162 = arith.addi %158, %161 : index
                      %163 = arith.muli %122, %c64_i64 : i64
                      %164 = arith.addi %163, %95 : i64
                      %165 = cannir.extract_dim %155[2] : !cannir.stride<(-1, -1, -1, 1)>
                      %166 = arith.muli %164, %165 : i64
                      %167 = arith.index_cast %166 : i64 to index
                      %168 = arith.addi %162, %167 : index
                      %169 = arith.divsi %80, %c64_i64 : i64
                      %170 = arith.minsi %122, %169 : i64
                      %171 = arith.muli %170, %c64_i64 : i64
                      %172 = arith.subi %80, %171 : i64
                      %173 = arith.minsi %172, %c64_i64 : i64
                      %174 = arith.cmpi sle, %122, %169 : i64
                      %175 = arith.select %174, %173, %c0_i64 : i64
                      %176 = cannir.extract_layout %arg2 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                      %177 = cannir.extract_stride %176 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                      %178 = cannir.extract_dim %177[2] : !cannir.stride<(-1, -1, -1, 1)>
                      %179 = arith.index_cast %178 : i64 to index
                      %180 = arith.muli %179, %c2 : index
                      %181 = arith.index_cast %180 : index to i64
                      asccube.get_buf mte2, %135, 0 {cannir.buffer.generated_sync}
                      %182 = arith.cmpi ugt, %175, %c0_i64 : i64
                      scf.if %182 {
                        %183 = arith.trunci %175 : i64 to i32
                        asccube.copy_gm2l1 %153[%c0], %alloc_0[%168], %183, %c128_i32, %181, %c1_i32, %c0_i64, %c0_i64, %c64_i32 l2_cache_ctl %c0_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
                      }
                      asccube.release_buf mte2, %135, 0 {cannir.buffer.generated_sync}
                    }
                    %136 = arith.cmpi eq, %arg24, %c0_i64 : i64
                    %137 = ascscalar.memref_address %alloc_20 : memref<8192xbf16, 2>
                    %138 = ascscalar.memref_address %alloc_19 : memref<8192xbf16, 2>
                    %139 = arith.select %136, %137, %138 : i64
                    %140 = arith.select %136, %c4_i64, %c5_i64 : i64
                    %141 = arith.addi %arg24, %c1_i64 : i64
                    %142 = arith.remui %141, %c2_i64 : i64
                    scf.if %130 {
                      asccube.get_buf mte2, %140, 0 {cannir.buffer.generated_sync}
                      scf.for %arg29 = %c0_i64 to %c8_i64 step %c1_i64  : i64 {
                        %153 = ascscalar.bind_local_ptr %139 : memref<1xbf16, 2>
                        %154 = cannir.extract_physical_layout %arg0 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                        %155 = cannir.extract_stride %154 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                        %156 = cannir.extract_dim %155[0] : !cannir.stride<(-1, -1, -1, 1)>
                        %157 = arith.muli %96, %156 : i64
                        %158 = arith.index_cast %157 : i64 to index
                        %159 = cannir.extract_dim %155[1] : !cannir.stride<(-1, -1, -1, 1)>
                        %160 = arith.muli %110, %159 : i64
                        %161 = arith.index_cast %160 : i64 to index
                        %162 = arith.addi %158, %161 : index
                        %163 = arith.muli %122, %c64_i64 : i64
                        %164 = arith.addi %163, %95 : i64
                        %165 = cannir.extract_dim %155[2] : !cannir.stride<(-1, -1, -1, 1)>
                        %166 = arith.muli %164, %165 : i64
                        %167 = arith.index_cast %166 : i64 to index
                        %168 = arith.addi %162, %167 : index
                        %169 = arith.muli %arg29, %c16_i64 overflow<nsw> : i64
                        %170 = arith.index_cast %169 : i64 to index
                        %171 = arith.addi %168, %170 : index
                        %172 = cannir.extract_layout %arg0 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                        %173 = cannir.extract_stride %172 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                        %174 = cannir.extract_dim %173[2] : !cannir.stride<(-1, -1, -1, 1)>
                        %175 = arith.index_cast %174 : i64 to index
                        %176 = arith.muli %arg29, %c1024_i64 overflow<nsw> : i64
                        %177 = arith.muli %175, %c2 : index
                        %178 = arith.index_cast %177 : index to i64
                        %179 = arith.cmpi ugt, %121, %c0_i64 : i64
                        scf.if %179 {
                          %180 = arith.index_cast %176 : i64 to index
                          %181 = arith.trunci %121 : i64 to i32
                          asccube.copy_gm2l1 %153[%180], %alloc[%171], %181, %c16_i32, %178, %c1_i32, %c0_i64, %c0_i64, %c64_i32 l2_cache_ctl %c1_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
                        }
                      }
                      asccube.release_buf mte2, %140, 0 {cannir.buffer.generated_sync}
                    } else {
                      %153 = ascscalar.bind_local_ptr %139 : memref<1xbf16, 2>
                      %154 = cannir.extract_physical_layout %arg0 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                      %155 = cannir.extract_stride %154 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                      %156 = cannir.extract_dim %155[0] : !cannir.stride<(-1, -1, -1, 1)>
                      %157 = arith.muli %96, %156 : i64
                      %158 = arith.index_cast %157 : i64 to index
                      %159 = cannir.extract_dim %155[1] : !cannir.stride<(-1, -1, -1, 1)>
                      %160 = arith.muli %110, %159 : i64
                      %161 = arith.index_cast %160 : i64 to index
                      %162 = arith.addi %158, %161 : index
                      %163 = arith.muli %122, %c64_i64 : i64
                      %164 = arith.addi %163, %95 : i64
                      %165 = cannir.extract_dim %155[2] : !cannir.stride<(-1, -1, -1, 1)>
                      %166 = arith.muli %164, %165 : i64
                      %167 = arith.index_cast %166 : i64 to index
                      %168 = arith.addi %162, %167 : index
                      %169 = arith.divsi %80, %c64_i64 : i64
                      %170 = arith.minsi %122, %169 : i64
                      %171 = arith.muli %170, %c64_i64 : i64
                      %172 = arith.subi %80, %171 : i64
                      %173 = arith.minsi %172, %c64_i64 : i64
                      %174 = arith.cmpi sle, %122, %169 : i64
                      %175 = arith.select %174, %173, %c0_i64 : i64
                      %176 = cannir.extract_layout %arg0 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                      %177 = cannir.extract_stride %176 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                      %178 = cannir.extract_dim %177[2] : !cannir.stride<(-1, -1, -1, 1)>
                      %179 = arith.index_cast %178 : i64 to index
                      %180 = arith.muli %179, %c2 : index
                      %181 = arith.index_cast %180 : index to i64
                      asccube.get_buf mte2, %140, 0 {cannir.buffer.generated_sync}
                      %182 = arith.cmpi ugt, %175, %c0_i64 : i64
                      scf.if %182 {
                        %183 = arith.trunci %175 : i64 to i32
                        asccube.copy_gm2l1 %153[%c0], %alloc[%168], %183, %c128_i32, %181, %c1_i32, %c0_i64, %c0_i64, %c64_i32 l2_cache_ctl %c1_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
                      }
                      asccube.release_buf mte2, %140, 0 {cannir.buffer.generated_sync}
                    }
                    %143 = ascscalar.memref_address %alloc_18 : memref<4096xbf16, 2>
                    %144 = ascscalar.memref_address %alloc_17 : memref<4096xbf16, 2>
                    %145 = arith.select %136, %143, %144 : i64
                    %146 = arith.select %136, %c6_i64, %c7_i64 : i64
                    scf.if %130 {
                      asccube.get_buf mte2, %146, 0 {cannir.buffer.generated_sync}
                      scf.for %arg29 = %c0_i64 to %c4_i64 step %c1_i64  : i64 {
                        %153 = ascscalar.bind_local_ptr %145 : memref<1xbf16, 2>
                        %154 = cannir.extract_physical_layout %arg5 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 64, 64)>, <(-1, -1, 4096, 64, 1)>>, <<(-1, -1, -1, 64, 64)>, <(-1, -1, 4096, 64, 1)>>, identity>
                        %155 = cannir.extract_stride %154 : <<(-1, -1, -1, 64, 64)>, <(-1, -1, 4096, 64, 1)>> -> <(-1, -1, 4096, 64, 1)>
                        %156 = cannir.extract_dim %155[0] : !cannir.stride<(-1, -1, 4096, 64, 1)>
                        %157 = arith.muli %113, %156 : i64
                        %158 = arith.index_cast %157 : i64 to index
                        %159 = cannir.extract_dim %155[1] : !cannir.stride<(-1, -1, 4096, 64, 1)>
                        %160 = arith.muli %110, %159 : i64
                        %161 = arith.index_cast %160 : i64 to index
                        %162 = arith.addi %158, %161 : index
                        %163 = arith.muli %122, %c4096_i64 : i64
                        %164 = arith.index_cast %163 : i64 to index
                        %165 = arith.addi %162, %164 : index
                        %166 = arith.muli %arg29, %c16_i64 overflow<nsw> : i64
                        %167 = arith.index_cast %166 : i64 to index
                        %168 = arith.addi %165, %167 : index
                        %169 = arith.muli %arg29, %c1024_i64 overflow<nsw> : i64
                        %170 = arith.cmpi ugt, %121, %c0_i64 : i64
                        scf.if %170 {
                          %171 = arith.index_cast %169 : i64 to index
                          %172 = arith.trunci %121 : i64 to i32
                          asccube.copy_gm2l1 %153[%171], %alloc_2[%168], %172, %c16_i32, %c128_i64, %c1_i32, %c0_i64, %c0_i64, %c64_i32 l2_cache_ctl %c1_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
                        }
                      }
                      asccube.release_buf mte2, %146, 0 {cannir.buffer.generated_sync}
                    } else {
                      %153 = ascscalar.bind_local_ptr %145 : memref<1xbf16, 2>
                      %154 = cannir.extract_physical_layout %arg5 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 64, 64)>, <(-1, -1, 4096, 64, 1)>>, <<(-1, -1, -1, 64, 64)>, <(-1, -1, 4096, 64, 1)>>, identity>
                      %155 = cannir.extract_stride %154 : <<(-1, -1, -1, 64, 64)>, <(-1, -1, 4096, 64, 1)>> -> <(-1, -1, 4096, 64, 1)>
                      %156 = cannir.extract_dim %155[0] : !cannir.stride<(-1, -1, 4096, 64, 1)>
                      %157 = arith.muli %113, %156 : i64
                      %158 = arith.index_cast %157 : i64 to index
                      %159 = cannir.extract_dim %155[1] : !cannir.stride<(-1, -1, 4096, 64, 1)>
                      %160 = arith.muli %110, %159 : i64
                      %161 = arith.index_cast %160 : i64 to index
                      %162 = arith.addi %158, %161 : index
                      %163 = arith.muli %122, %c4096_i64 : i64
                      %164 = arith.index_cast %163 : i64 to index
                      %165 = arith.addi %162, %164 : index
                      asccube.get_buf mte2, %146, 0 {cannir.buffer.generated_sync}
                      asccube.copy_gm2l1 %153[%c0], %alloc_2[%165], %c64_i32, %c64_i32, %c128_i64, %c1_i32, %c0_i64, %c0_i64, %c64_i32 l2_cache_ctl %c1_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
                      asccube.release_buf mte2, %146, 0 {cannir.buffer.generated_sync}
                    }
                    %147 = arith.cmpi eq, %83, %c0_i32 : i32
                    %148 = scf.if %147 -> (i64) {
                      %153 = arith.addi %arg25, %c1_i64 : i64
                      %154 = arith.remui %153, %c2_i64 : i64
                      scf.yield %154 : i64
                    } else {
                      %153 = arith.addi %arg25, %c1_i64 : i64
                      %154 = arith.remui %153, %c2_i64 : i64
                      scf.yield %154 : i64
                    }
                    %149 = arith.muli %97, %c32_i64 : i64
                    %150 = arith.cmpi slt, %149, %121 : i64
                    %151 = scf.if %150 -> (i64) {
                      %153 = arith.addi %arg27, %c1_i64 : i64
                      %154 = arith.remui %153, %c2_i64 : i64
                      scf.yield %154 : i64
                    } else {
                      scf.yield %arg27 : i64
                    }
                    %152:7 = scf.for %arg29 = %c0_i32 to %86 step %c1_i32 iter_args(%arg30 = %129, %arg31 = %arg23, %arg32 = %142, %arg33 = %148, %arg34 = %arg26, %arg35 = %151, %arg36 = %arg28) -> (i64, i64, i64, i64, i64, i64, i64)  : i32 {
                      %153 = arith.addi %83, %arg29 : i32
                      %154 = arith.addi %153, %c1_i32 : i32
                      %155 = arith.extsi %154 : i32 to i64
                      %156 = arith.cmpi eq, %155, %94 : i64
                      %157 = arith.muli %153, %c64_i32 : i32
                      %158 = arith.extsi %157 : i32 to i64
                      %159 = arith.subi %80, %158 : i64
                      %160 = arith.select %156, %159, %c64_i64 : i64
                      %161 = arith.addi %153, %c2_i32 : i32
                      %162 = arith.extsi %161 : i32 to i64
                      %163 = arith.cmpi eq, %162, %94 : i64
                      %164 = arith.muli %154, %c64_i32 : i32
                      %165 = arith.extsi %164 : i32 to i64
                      %166 = arith.subi %80, %165 : i64
                      %167 = arith.select %163, %166, %c64_i64 : i64
                      %168 = arith.extsi %153 : i32 to i64
                      %169 = arith.cmpi eq, %arg31, %c0_i64 : i64
                      %170 = ascscalar.memref_address %alloc_16 : memref<16384xbf16, 2>
                      %171 = ascscalar.memref_address %alloc_15 : memref<16384xbf16, 2>
                      %172 = arith.select %169, %170, %171 : i64
                      %173 = arith.select %169, %c8_i64, %c9_i64 : i64
                      %174 = arith.addi %arg31, %c1_i64 : i64
                      %175 = arith.remui %174, %c2_i64 : i64
                      %176 = ascscalar.bitcast_memref %alloc_8 byte_offset = %c0_i64 : memref<16384xbf16, 4> -> memref<?xbf16, 4>
                      %177 = ascscalar.bind_local_ptr %172 : memref<1xbf16, 2>
                      %178 = arith.cmpi eq, %173, %c9_i64 : i64
                      %179 = arith.extui %178 : i1 to i64
                      asccube.sync_intra_wait mte1, %179 {cannir.sync.banks = 3 : i32}
                      %180 = arith.addi %179, %c16_i64 overflow<nsw> : i64
                      asccube.sync_intra_wait mte1, %180 {cannir.sync.banks = 3 : i32}
                      asccube.get_buf mte1, %c13_i64, 0 {cannir.buffer.generated_sync}
                      asccube.copy_l12l0b %176[%c0], %177[%c0], %c0_i32, %c0_i32, %c2_i32, %c8_i32, %c2_i32, %c2_i32 : memref<?xbf16, 4>, memref<1xbf16, 2>
                      asccube.release_buf mte1, %c13_i64, 0 {cannir.buffer.generated_sync}
                      asccube.sync_intra_arrive mte1, %179 {cannir.sync.banks = 3 : i32}
                      asccube.sync_intra_arrive mte1, %180 {cannir.sync.banks = 3 : i32}
                      %181 = arith.cmpi eq, %arg34, %c0_i64 : i64
                      %182 = ascscalar.memref_address %alloc_28 : memref<8192xbf16, 3>
                      %183 = ascscalar.memref_address %alloc_27 : memref<8192xbf16, 3>
                      %184 = arith.select %181, %182, %183 : i64
                      %185 = arith.select %181, %c8_i64, %c9_i64 : i64
                      %186 = arith.addi %arg34, %c1_i64 : i64
                      %187 = arith.remui %186, %c2_i64 : i64
                      %188 = arith.select %169, %124, %125 : i64
                      %189 = arith.select %169, %c2_i64, %c3_i64 : i64
                      %190 = ascscalar.bind_local_ptr %184 : memref<1xbf16, 3>
                      %191 = ascscalar.bind_local_ptr %188 : memref<1xbf16, 2>
                      asccube.get_buf mte1, %185, 0 {cannir.buffer.generated_sync}
                      asccube.get_buf mte1, %189, 0 {cannir.buffer.generated_sync}
                      asccube.copy_l12l0a %190[%c0], %191[%c0], %c0_i32, %c0_i32, %c4_i32, %c8_i32, %c4_i32, %c4_i32 : memref<1xbf16, 3>, memref<1xbf16, 2>
                      asccube.release_buf mte1, %189, 0 {cannir.buffer.generated_sync}
                      asccube.release_buf mte1, %185, 0 {cannir.buffer.generated_sync}
                      %192 = ascscalar.bitcast_memref %alloc_7 byte_offset = %c0_i64 : memref<8192xf32, 5> -> memref<?xf32, 5>
                      asccube.get_buf m, %c15_i64, 0 {cannir.buffer.generated_sync}
                      asccube.get_buf m, %185, 0 {cannir.buffer.generated_sync}
                      asccube.get_buf m, %c13_i64, 0 {cannir.buffer.generated_sync}
                      asccube.mmad %192[%c0], %190, %176, %c64_i32, %c128_i32, %c32_i32, %c0_i32 {c_matrix_init_val = true} : memref<?xf32, 5>, memref<1xbf16, 3>, memref<?xbf16, 4>
                      asccube.release_buf m, %185, 0 {cannir.buffer.generated_sync}
                      asccube.release_buf m, %c15_i64, 0 {cannir.buffer.generated_sync}
                      %193 = arith.cmpi eq, %187, %c0_i64 : i64
                      %194 = arith.select %193, %182, %183 : i64
                      %195 = arith.select %193, %c8_i64, %c9_i64 : i64
                      %196 = arith.addi %187, %c1_i64 overflow<nsw> : i64
                      %197 = arith.remui %196, %c2_i64 : i64
                      %198 = arith.select %169, %131, %132 : i64
                      %199 = arith.cmpi ne, %arg31, %c0_i64 : i64
                      %200 = arith.extui %199 : i1 to i64
                      %201 = ascscalar.bind_local_ptr %194 : memref<1xbf16, 3>
                      %202 = ascscalar.bind_local_ptr %198 : memref<1xbf16, 2>
                      asccube.get_buf mte1, %195, 0 {cannir.buffer.generated_sync}
                      asccube.get_buf mte1, %200, 0 {cannir.buffer.generated_sync}
                      asccube.copy_l12l0a %201[%c0], %202[%c0], %c0_i32, %c0_i32, %c4_i32, %c8_i32, %c4_i32, %c4_i32 : memref<1xbf16, 3>, memref<1xbf16, 2>
                      asccube.release_buf mte1, %200, 0 {cannir.buffer.generated_sync}
                      asccube.release_buf mte1, %195, 0 {cannir.buffer.generated_sync}
                      %203 = ascscalar.bitcast_memref %alloc_9 byte_offset = %c0_i64 : memref<8192xf32, 5> -> memref<?xf32, 5>
                      asccube.get_buf m, %c16_i64, 0 {cannir.buffer.generated_sync}
                      asccube.get_buf m, %195, 0 {cannir.buffer.generated_sync}
                      asccube.mmad %203[%c0], %201, %176, %c64_i32, %c128_i32, %c32_i32, %c0_i32 {c_matrix_init_val = true} : memref<?xf32, 5>, memref<1xbf16, 3>, memref<?xbf16, 4>
                      asccube.release_buf m, %c13_i64, 0 {cannir.buffer.generated_sync}
                      asccube.release_buf m, %195, 0 {cannir.buffer.generated_sync}
                      asccube.release_buf m, %c16_i64, 0 {cannir.buffer.generated_sync}
                      %204 = arith.select %169, %137, %138 : i64
                      %205 = arith.select %169, %c4_i64, %c5_i64 : i64
                      %206 = ascscalar.bitcast_memref %alloc_12 byte_offset = %c0_i64 : memref<8192xbf16, 4> -> memref<?xbf16, 4>
                      %207 = ascscalar.bind_local_ptr %204 : memref<1xbf16, 2>
                      asccube.get_buf mte1, %c14_i64, 0 {cannir.buffer.generated_sync}
                      asccube.get_buf mte1, %205, 0 {cannir.buffer.generated_sync}
                      asccube.copy_l12l0b %206[%c0], %207[%c0], %c0_i32, %c0_i32, %c4_i32, %c8_i32, %c4_i32, %c8_i32 {transpose = true} : memref<?xbf16, 4>, memref<1xbf16, 2>
                      asccube.release_buf mte1, %205, 0 {cannir.buffer.generated_sync}
                      asccube.release_buf mte1, %c14_i64, 0 {cannir.buffer.generated_sync}
                      %208 = arith.select %169, %143, %144 : i64
                      %209 = arith.select %169, %c6_i64, %c7_i64 : i64
                      %210 = ascscalar.bitcast_memref %alloc_13 byte_offset = %c0_i64 : memref<4096xbf16, 3> -> memref<?xbf16, 3>
                      %211 = ascscalar.bind_local_ptr %208 : memref<1xbf16, 2>
                      asccube.get_buf mte1, %c11_i64, 0 {cannir.buffer.generated_sync}
                      asccube.get_buf mte1, %209, 0 {cannir.buffer.generated_sync}
                      asccube.copy_l12l0a %210[%c0], %211[%c0], %c0_i32, %c0_i32, %c4_i32, %c4_i32, %c4_i32, %c4_i32 : memref<?xbf16, 3>, memref<1xbf16, 2>
                      asccube.release_buf mte1, %209, 0 {cannir.buffer.generated_sync}
                      asccube.release_buf mte1, %c11_i64, 0 {cannir.buffer.generated_sync}
                      %212 = arith.addi %arg29, %c1_i32 : i32
                      %213 = arith.cmpi slt, %212, %86 : i32
                      %214:2 = scf.if %213 -> (i64, i64) {
                        %263 = arith.cmpi eq, %arg30, %c0_i64 : i64
                        %264 = arith.select %263, %131, %132 : i64
                        %265 = arith.cmpi ne, %arg30, %c0_i64 : i64
                        %266 = arith.extui %265 : i1 to i64
                        %267 = arith.addi %arg30, %c1_i64 overflow<nsw> : i64
                        %268 = arith.remui %267, %c2_i64 : i64
                        %269 = arith.cmpi slt, %167, %c64_i64 : i64
                        scf.if %269 {
                          asccube.get_buf mte2, %266, 0 {cannir.buffer.generated_sync}
                          scf.for %arg37 = %c0_i64 to %c8_i64 step %c1_i64  : i64 {
                            %274 = ascscalar.bind_local_ptr %264 : memref<1xbf16, 2>
                            %275 = cannir.extract_physical_layout %arg2 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                            %276 = cannir.extract_stride %275 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                            %277 = cannir.extract_dim %276[0] : !cannir.stride<(-1, -1, -1, 1)>
                            %278 = arith.muli %96, %277 : i64
                            %279 = arith.index_cast %278 : i64 to index
                            %280 = cannir.extract_dim %276[1] : !cannir.stride<(-1, -1, -1, 1)>
                            %281 = arith.muli %110, %280 : i64
                            %282 = arith.index_cast %281 : i64 to index
                            %283 = arith.addi %279, %282 : index
                            %284 = arith.muli %155, %c64_i64 : i64
                            %285 = arith.addi %284, %95 : i64
                            %286 = cannir.extract_dim %276[2] : !cannir.stride<(-1, -1, -1, 1)>
                            %287 = arith.muli %285, %286 : i64
                            %288 = arith.index_cast %287 : i64 to index
                            %289 = arith.addi %283, %288 : index
                            %290 = arith.muli %arg37, %c16_i64 overflow<nsw> : i64
                            %291 = arith.index_cast %290 : i64 to index
                            %292 = arith.addi %289, %291 : index
                            %293 = cannir.extract_layout %arg2 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                            %294 = cannir.extract_stride %293 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                            %295 = cannir.extract_dim %294[2] : !cannir.stride<(-1, -1, -1, 1)>
                            %296 = arith.index_cast %295 : i64 to index
                            %297 = arith.muli %arg37, %c1024_i64 overflow<nsw> : i64
                            %298 = arith.muli %296, %c2 : index
                            %299 = arith.index_cast %298 : index to i64
                            %300 = arith.cmpi ugt, %167, %c0_i64 : i64
                            scf.if %300 {
                              %301 = arith.index_cast %297 : i64 to index
                              %302 = arith.trunci %167 : i64 to i32
                              asccube.copy_gm2l1 %274[%301], %alloc_0[%292], %302, %c16_i32, %299, %c1_i32, %c0_i64, %c0_i64, %c64_i32 l2_cache_ctl %c0_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
                            }
                          }
                          asccube.release_buf mte2, %266, 0 {cannir.buffer.generated_sync}
                        } else {
                          %274 = ascscalar.bind_local_ptr %264 : memref<1xbf16, 2>
                          %275 = cannir.extract_physical_layout %arg2 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                          %276 = cannir.extract_stride %275 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                          %277 = cannir.extract_dim %276[0] : !cannir.stride<(-1, -1, -1, 1)>
                          %278 = arith.muli %96, %277 : i64
                          %279 = arith.index_cast %278 : i64 to index
                          %280 = cannir.extract_dim %276[1] : !cannir.stride<(-1, -1, -1, 1)>
                          %281 = arith.muli %110, %280 : i64
                          %282 = arith.index_cast %281 : i64 to index
                          %283 = arith.addi %279, %282 : index
                          %284 = arith.muli %155, %c64_i64 : i64
                          %285 = arith.addi %284, %95 : i64
                          %286 = cannir.extract_dim %276[2] : !cannir.stride<(-1, -1, -1, 1)>
                          %287 = arith.muli %285, %286 : i64
                          %288 = arith.index_cast %287 : i64 to index
                          %289 = arith.addi %283, %288 : index
                          %290 = arith.divsi %80, %c64_i64 : i64
                          %291 = arith.minsi %155, %290 : i64
                          %292 = arith.muli %291, %c64_i64 : i64
                          %293 = arith.subi %80, %292 : i64
                          %294 = arith.minsi %293, %c64_i64 : i64
                          %295 = arith.cmpi sle, %155, %290 : i64
                          %296 = arith.select %295, %294, %c0_i64 : i64
                          %297 = cannir.extract_layout %arg2 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                          %298 = cannir.extract_stride %297 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                          %299 = cannir.extract_dim %298[2] : !cannir.stride<(-1, -1, -1, 1)>
                          %300 = arith.index_cast %299 : i64 to index
                          %301 = arith.muli %300, %c2 : index
                          %302 = arith.index_cast %301 : index to i64
                          asccube.get_buf mte2, %266, 0 {cannir.buffer.generated_sync}
                          %303 = arith.cmpi ugt, %296, %c0_i64 : i64
                          scf.if %303 {
                            %304 = arith.trunci %296 : i64 to i32
                            asccube.copy_gm2l1 %274[%c0], %alloc_0[%289], %304, %c128_i32, %302, %c1_i32, %c0_i64, %c0_i64, %c64_i32 l2_cache_ctl %c0_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
                          }
                          asccube.release_buf mte2, %266, 0 {cannir.buffer.generated_sync}
                        }
                        %270 = arith.select %263, %124, %125 : i64
                        %271 = arith.select %263, %c2_i64, %c3_i64 : i64
                        scf.if %269 {
                          asccube.get_buf mte2, %271, 0 {cannir.buffer.generated_sync}
                          scf.for %arg37 = %c0_i64 to %c8_i64 step %c1_i64  : i64 {
                            %274 = ascscalar.bind_local_ptr %270 : memref<1xbf16, 2>
                            %275 = cannir.extract_physical_layout %arg3 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                            %276 = cannir.extract_stride %275 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                            %277 = cannir.extract_dim %276[0] : !cannir.stride<(-1, -1, -1, 1)>
                            %278 = arith.muli %96, %277 : i64
                            %279 = arith.index_cast %278 : i64 to index
                            %280 = cannir.extract_dim %276[1] : !cannir.stride<(-1, -1, -1, 1)>
                            %281 = arith.muli %110, %280 : i64
                            %282 = arith.index_cast %281 : i64 to index
                            %283 = arith.addi %279, %282 : index
                            %284 = arith.muli %155, %c64_i64 : i64
                            %285 = arith.addi %284, %95 : i64
                            %286 = cannir.extract_dim %276[2] : !cannir.stride<(-1, -1, -1, 1)>
                            %287 = arith.muli %285, %286 : i64
                            %288 = arith.index_cast %287 : i64 to index
                            %289 = arith.addi %283, %288 : index
                            %290 = arith.muli %arg37, %c16_i64 overflow<nsw> : i64
                            %291 = arith.index_cast %290 : i64 to index
                            %292 = arith.addi %289, %291 : index
                            %293 = cannir.extract_layout %arg3 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                            %294 = cannir.extract_stride %293 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                            %295 = cannir.extract_dim %294[2] : !cannir.stride<(-1, -1, -1, 1)>
                            %296 = arith.index_cast %295 : i64 to index
                            %297 = arith.muli %arg37, %c1024_i64 overflow<nsw> : i64
                            %298 = arith.muli %296, %c2 : index
                            %299 = arith.index_cast %298 : index to i64
                            %300 = arith.cmpi ugt, %167, %c0_i64 : i64
                            scf.if %300 {
                              %301 = arith.index_cast %297 : i64 to index
                              %302 = arith.trunci %167 : i64 to i32
                              asccube.copy_gm2l1 %274[%301], %alloc_1[%292], %302, %c16_i32, %299, %c1_i32, %c0_i64, %c0_i64, %c64_i32 l2_cache_ctl %c0_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
                            }
                          }
                          asccube.release_buf mte2, %271, 0 {cannir.buffer.generated_sync}
                        } else {
                          %274 = ascscalar.bind_local_ptr %270 : memref<1xbf16, 2>
                          %275 = cannir.extract_physical_layout %arg3 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                          %276 = cannir.extract_stride %275 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                          %277 = cannir.extract_dim %276[0] : !cannir.stride<(-1, -1, -1, 1)>
                          %278 = arith.muli %96, %277 : i64
                          %279 = arith.index_cast %278 : i64 to index
                          %280 = cannir.extract_dim %276[1] : !cannir.stride<(-1, -1, -1, 1)>
                          %281 = arith.muli %110, %280 : i64
                          %282 = arith.index_cast %281 : i64 to index
                          %283 = arith.addi %279, %282 : index
                          %284 = arith.muli %155, %c64_i64 : i64
                          %285 = arith.addi %284, %95 : i64
                          %286 = cannir.extract_dim %276[2] : !cannir.stride<(-1, -1, -1, 1)>
                          %287 = arith.muli %285, %286 : i64
                          %288 = arith.index_cast %287 : i64 to index
                          %289 = arith.addi %283, %288 : index
                          %290 = arith.divsi %80, %c64_i64 : i64
                          %291 = arith.minsi %155, %290 : i64
                          %292 = arith.muli %291, %c64_i64 : i64
                          %293 = arith.subi %80, %292 : i64
                          %294 = arith.minsi %293, %c64_i64 : i64
                          %295 = arith.cmpi sle, %155, %290 : i64
                          %296 = arith.select %295, %294, %c0_i64 : i64
                          %297 = cannir.extract_layout %arg3 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                          %298 = cannir.extract_stride %297 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                          %299 = cannir.extract_dim %298[2] : !cannir.stride<(-1, -1, -1, 1)>
                          %300 = arith.index_cast %299 : i64 to index
                          %301 = arith.muli %300, %c2 : index
                          %302 = arith.index_cast %301 : index to i64
                          asccube.get_buf mte2, %271, 0 {cannir.buffer.generated_sync}
                          %303 = arith.cmpi ugt, %296, %c0_i64 : i64
                          scf.if %303 {
                            %304 = arith.trunci %296 : i64 to i32
                            asccube.copy_gm2l1 %274[%c0], %alloc_1[%289], %304, %c128_i32, %302, %c1_i32, %c0_i64, %c0_i64, %c64_i32 l2_cache_ctl %c0_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
                          }
                          asccube.release_buf mte2, %271, 0 {cannir.buffer.generated_sync}
                        }
                        %272 = arith.cmpi slt, %149, %167 : i64
                        %273 = scf.if %272 -> (i64) {
                          %274 = arith.addi %arg35, %c1_i64 : i64
                          %275 = arith.remui %274, %c2_i64 : i64
                          scf.yield %275 : i64
                        } else {
                          scf.yield %arg35 : i64
                        }
                        scf.yield %268, %273 : i64, i64
                      } else {
                        scf.yield %arg30, %arg35 : i64, i64
                      }
                      %215 = ascscalar.bitcast_memref %alloc_6 byte_offset = %c0_i64 : memref<4096xf32, 1> -> memref<?xf32, 1>
                      asccube.sync_intra_wait fixpipe, %c5_i64 {cannir.sync.banks = 3 : i32}
                      asccube.sync_intra_wait fixpipe, %c21_i64 {cannir.sync.banks = 3 : i32}
                      asccube.get_buf fixpipe, %c15_i64, 0 {cannir.buffer.generated_sync}
                      asccube.copy_l0c2ub %215[%c0], %192[%c0], %c32_i32, %c64_i32, %c32_i32, %c64_i32, %c0_i32 sub_block_id %c0_i32 {dual_dst_ctl = 1 : i32, nz2nd_en = true, relu_en = false} : memref<?xf32, 1>, memref<?xf32, 5>
                      asccube.release_buf fixpipe, %c15_i64, 0 {cannir.buffer.generated_sync}
                      asccube.sync_intra_arrive fixpipe, %c5_i64 {cannir.sync.banks = 3 : i32}
                      asccube.sync_intra_arrive fixpipe, %c21_i64 {cannir.sync.banks = 3 : i32}
                      %216 = ascscalar.memref_address %alloc_22 : memref<8192xbf16, 2>
                      %217 = ascscalar.memref_address %alloc_21 : memref<8192xbf16, 2>
                      %218 = arith.select %169, %216, %217 : i64
                      %219 = arith.select %169, %c10_i64, %c11_i64 : i64
                      %220 = arith.subi %160, %149 : i64
                      %221 = arith.cmpi slt, %220, %c0_i64 : i64
                      %222 = arith.select %221, %c0_i64, %220 : i64
                      %223 = arith.cmpi sgt, %222, %c32_i64 : i64
                      %224 = arith.select %223, %c32_i64, %222 : i64
                      %225 = arith.cmpi eq, %224, %c32_i64 : i64
                      %226 = scf.if %225 -> (i64) {
                        %263 = arith.addi %arg36, %c1_i64 : i64
                        %264 = arith.remui %263, %c2_i64 : i64
                        scf.yield %264 : i64
                      } else {
                        scf.yield %arg36 : i64
                      }
                      %227 = arith.cmpi sgt, %224, %c0_i64 : i64
                      %228 = arith.cmpi slt, %224, %c32_i64 : i64
                      %229 = arith.select %227, %228, %false : i1
                      %230 = scf.if %229 -> (i64) {
                        %263 = arith.addi %226, %c1_i64 : i64
                        %264 = arith.remui %263, %c2_i64 : i64
                        scf.yield %264 : i64
                      } else {
                        scf.yield %226 : i64
                      }
                      %231 = ascscalar.bind_local_ptr %218 : memref<1xbf16, 2>
                      %232 = arith.cmpi eq, %219, %c11_i64 : i64
                      %233 = arith.select %232, %c3_i64, %c2_i64 : i64
                      %234 = ascscalar.bitcast_memref %alloc_11 byte_offset = %c0_i64 : memref<8192xbf16, 3> -> memref<?xbf16, 3>
                      asccube.sync_intra_wait mte1, %233 {cannir.sync.banks = 3 : i32}
                      %235 = arith.addi %233, %c16_i64 overflow<nsw> : i64
                      asccube.sync_intra_wait mte1, %235 {cannir.sync.banks = 3 : i32}
                      asccube.get_buf mte1, %c10_i64, 0 {cannir.buffer.generated_sync}
                      asccube.copy_l12l0a %234[%c0], %231[%c0], %c0_i32, %c0_i32, %c4_i32, %c2_i32, %c4_i32, %c2_i32 {transpose = true} : memref<?xbf16, 3>, memref<1xbf16, 2>
                      asccube.release_buf mte1, %c10_i64, 0 {cannir.buffer.generated_sync}
                      %236 = ascscalar.bitcast_memref %alloc_10 byte_offset = %c0_i64 : memref<16384xf32, 5> -> memref<?xf32, 5>
                      asccube.get_buf m, %c17_i64, 0 {cannir.buffer.generated_sync}
                      asccube.get_buf m, %c10_i64, 0 {cannir.buffer.generated_sync}
                      asccube.get_buf m, %c14_i64, 0 {cannir.buffer.generated_sync}
                      asccube.mmad %236[%c0], %234, %206, %c32_i32, %c64_i32, %c128_i32, %c0_i32 {c_matrix_init_val = true} : memref<?xf32, 5>, memref<?xbf16, 3>, memref<?xbf16, 4>
                      asccube.release_buf m, %c14_i64, 0 {cannir.buffer.generated_sync}
                      asccube.release_buf m, %c10_i64, 0 {cannir.buffer.generated_sync}
                      asccube.release_buf m, %c17_i64, 0 {cannir.buffer.generated_sync}
                      %237 = ascscalar.bitcast_memref %alloc_5 byte_offset = %c0_i64 : memref<8192xf32, 1> -> memref<?xf32, 1>
                      asccube.sync_intra_wait fixpipe, %c4_i64 {cannir.sync.banks = 3 : i32}
                      asccube.sync_intra_wait fixpipe, %c20_i64 {cannir.sync.banks = 3 : i32}
                      asccube.get_buf fixpipe, %c17_i64, 0 {cannir.buffer.generated_sync}
                      asccube.copy_l0c2ub %237[%c0], %236[%c0], %c128_i32, %c32_i32, %c64_i32, %c32_i32, %c0_i32 sub_block_id %c0_i32 {dual_dst_ctl = 2 : i32, nz2nd_en = true, relu_en = false} : memref<?xf32, 1>, memref<?xf32, 5>
                      asccube.release_buf fixpipe, %c17_i64, 0 {cannir.buffer.generated_sync}
                      asccube.sync_intra_arrive fixpipe, %c4_i64 {cannir.sync.banks = 3 : i32}
                      asccube.sync_intra_arrive fixpipe, %c20_i64 {cannir.sync.banks = 3 : i32}
                      %238 = ascscalar.bitcast_memref %alloc_14 byte_offset = %c0_i64 : memref<8192xbf16, 4> -> memref<?xbf16, 4>
                      asccube.get_buf mte1, %c12_i64, 0 {cannir.buffer.generated_sync}
                      asccube.copy_l12l0b %238[%c0], %231[%c0], %c0_i32, %c0_i32, %c4_i32, %c2_i32, %c4_i32, %c2_i32 {transpose = true} : memref<?xbf16, 4>, memref<1xbf16, 2>
                      asccube.release_buf mte1, %c12_i64, 0 {cannir.buffer.generated_sync}
                      asccube.sync_intra_arrive mte1, %233 {cannir.sync.banks = 3 : i32}
                      asccube.sync_intra_arrive mte1, %235 {cannir.sync.banks = 3 : i32}
                      asccube.get_buf m, %c16_i64, 0 {cannir.buffer.generated_sync}
                      asccube.get_buf m, %c11_i64, 0 {cannir.buffer.generated_sync}
                      asccube.get_buf m, %c12_i64, 0 {cannir.buffer.generated_sync}
                      asccube.mmad %203[%c0], %210, %238, %c64_i32, %c64_i32, %c32_i32, %c0_i32 {c_matrix_init_val = false} : memref<?xf32, 5>, memref<?xbf16, 3>, memref<?xbf16, 4>
                      asccube.release_buf m, %c12_i64, 0 {cannir.buffer.generated_sync}
                      asccube.release_buf m, %c11_i64, 0 {cannir.buffer.generated_sync}
                      asccube.release_buf m, %c16_i64, 0 {cannir.buffer.generated_sync}
                      %239 = scf.if %213 -> (i64) {
                        %263 = arith.cmpi eq, %arg32, %c0_i64 : i64
                        %264 = arith.select %263, %137, %138 : i64
                        %265 = arith.select %263, %c4_i64, %c5_i64 : i64
                        %266 = arith.addi %arg32, %c1_i64 overflow<nsw> : i64
                        %267 = arith.remui %266, %c2_i64 : i64
                        %268 = arith.cmpi slt, %167, %c64_i64 : i64
                        scf.if %268 {
                          asccube.get_buf mte2, %265, 0 {cannir.buffer.generated_sync}
                          scf.for %arg37 = %c0_i64 to %c8_i64 step %c1_i64  : i64 {
                            %271 = ascscalar.bind_local_ptr %264 : memref<1xbf16, 2>
                            %272 = cannir.extract_physical_layout %arg0 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                            %273 = cannir.extract_stride %272 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                            %274 = cannir.extract_dim %273[0] : !cannir.stride<(-1, -1, -1, 1)>
                            %275 = arith.muli %96, %274 : i64
                            %276 = arith.index_cast %275 : i64 to index
                            %277 = cannir.extract_dim %273[1] : !cannir.stride<(-1, -1, -1, 1)>
                            %278 = arith.muli %110, %277 : i64
                            %279 = arith.index_cast %278 : i64 to index
                            %280 = arith.addi %276, %279 : index
                            %281 = arith.muli %155, %c64_i64 : i64
                            %282 = arith.addi %281, %95 : i64
                            %283 = cannir.extract_dim %273[2] : !cannir.stride<(-1, -1, -1, 1)>
                            %284 = arith.muli %282, %283 : i64
                            %285 = arith.index_cast %284 : i64 to index
                            %286 = arith.addi %280, %285 : index
                            %287 = arith.muli %arg37, %c16_i64 overflow<nsw> : i64
                            %288 = arith.index_cast %287 : i64 to index
                            %289 = arith.addi %286, %288 : index
                            %290 = cannir.extract_layout %arg0 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                            %291 = cannir.extract_stride %290 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                            %292 = cannir.extract_dim %291[2] : !cannir.stride<(-1, -1, -1, 1)>
                            %293 = arith.index_cast %292 : i64 to index
                            %294 = arith.muli %arg37, %c1024_i64 overflow<nsw> : i64
                            %295 = arith.muli %293, %c2 : index
                            %296 = arith.index_cast %295 : index to i64
                            %297 = arith.cmpi ugt, %167, %c0_i64 : i64
                            scf.if %297 {
                              %298 = arith.index_cast %294 : i64 to index
                              %299 = arith.trunci %167 : i64 to i32
                              asccube.copy_gm2l1 %271[%298], %alloc[%289], %299, %c16_i32, %296, %c1_i32, %c0_i64, %c0_i64, %c64_i32 l2_cache_ctl %c1_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
                            }
                          }
                          asccube.release_buf mte2, %265, 0 {cannir.buffer.generated_sync}
                        } else {
                          %271 = ascscalar.bind_local_ptr %264 : memref<1xbf16, 2>
                          %272 = cannir.extract_physical_layout %arg0 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                          %273 = cannir.extract_stride %272 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                          %274 = cannir.extract_dim %273[0] : !cannir.stride<(-1, -1, -1, 1)>
                          %275 = arith.muli %96, %274 : i64
                          %276 = arith.index_cast %275 : i64 to index
                          %277 = cannir.extract_dim %273[1] : !cannir.stride<(-1, -1, -1, 1)>
                          %278 = arith.muli %110, %277 : i64
                          %279 = arith.index_cast %278 : i64 to index
                          %280 = arith.addi %276, %279 : index
                          %281 = arith.muli %155, %c64_i64 : i64
                          %282 = arith.addi %281, %95 : i64
                          %283 = cannir.extract_dim %273[2] : !cannir.stride<(-1, -1, -1, 1)>
                          %284 = arith.muli %282, %283 : i64
                          %285 = arith.index_cast %284 : i64 to index
                          %286 = arith.addi %280, %285 : index
                          %287 = arith.divsi %80, %c64_i64 : i64
                          %288 = arith.minsi %155, %287 : i64
                          %289 = arith.muli %288, %c64_i64 : i64
                          %290 = arith.subi %80, %289 : i64
                          %291 = arith.minsi %290, %c64_i64 : i64
                          %292 = arith.cmpi sle, %155, %287 : i64
                          %293 = arith.select %292, %291, %c0_i64 : i64
                          %294 = cannir.extract_layout %arg0 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                          %295 = cannir.extract_stride %294 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                          %296 = cannir.extract_dim %295[2] : !cannir.stride<(-1, -1, -1, 1)>
                          %297 = arith.index_cast %296 : i64 to index
                          %298 = arith.muli %297, %c2 : index
                          %299 = arith.index_cast %298 : index to i64
                          asccube.get_buf mte2, %265, 0 {cannir.buffer.generated_sync}
                          %300 = arith.cmpi ugt, %293, %c0_i64 : i64
                          scf.if %300 {
                            %301 = arith.trunci %293 : i64 to i32
                            asccube.copy_gm2l1 %271[%c0], %alloc[%286], %301, %c128_i32, %299, %c1_i32, %c0_i64, %c0_i64, %c64_i32 l2_cache_ctl %c1_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
                          }
                          asccube.release_buf mte2, %265, 0 {cannir.buffer.generated_sync}
                        }
                        %269 = arith.select %263, %143, %144 : i64
                        %270 = arith.select %263, %c6_i64, %c7_i64 : i64
                        scf.if %268 {
                          asccube.get_buf mte2, %270, 0 {cannir.buffer.generated_sync}
                          scf.for %arg37 = %c0_i64 to %c4_i64 step %c1_i64  : i64 {
                            %271 = ascscalar.bind_local_ptr %269 : memref<1xbf16, 2>
                            %272 = cannir.extract_physical_layout %arg5 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 64, 64)>, <(-1, -1, 4096, 64, 1)>>, <<(-1, -1, -1, 64, 64)>, <(-1, -1, 4096, 64, 1)>>, identity>
                            %273 = cannir.extract_stride %272 : <<(-1, -1, -1, 64, 64)>, <(-1, -1, 4096, 64, 1)>> -> <(-1, -1, 4096, 64, 1)>
                            %274 = cannir.extract_dim %273[0] : !cannir.stride<(-1, -1, 4096, 64, 1)>
                            %275 = arith.muli %113, %274 : i64
                            %276 = arith.index_cast %275 : i64 to index
                            %277 = cannir.extract_dim %273[1] : !cannir.stride<(-1, -1, 4096, 64, 1)>
                            %278 = arith.muli %110, %277 : i64
                            %279 = arith.index_cast %278 : i64 to index
                            %280 = arith.addi %276, %279 : index
                            %281 = arith.muli %155, %c4096_i64 : i64
                            %282 = arith.index_cast %281 : i64 to index
                            %283 = arith.addi %280, %282 : index
                            %284 = arith.muli %arg37, %c16_i64 overflow<nsw> : i64
                            %285 = arith.index_cast %284 : i64 to index
                            %286 = arith.addi %283, %285 : index
                            %287 = arith.muli %arg37, %c1024_i64 overflow<nsw> : i64
                            %288 = arith.cmpi ugt, %167, %c0_i64 : i64
                            scf.if %288 {
                              %289 = arith.index_cast %287 : i64 to index
                              %290 = arith.trunci %167 : i64 to i32
                              asccube.copy_gm2l1 %271[%289], %alloc_2[%286], %290, %c16_i32, %c128_i64, %c1_i32, %c0_i64, %c0_i64, %c64_i32 l2_cache_ctl %c1_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
                            }
                          }
                          asccube.release_buf mte2, %270, 0 {cannir.buffer.generated_sync}
                        } else {
                          %271 = ascscalar.bind_local_ptr %269 : memref<1xbf16, 2>
                          %272 = cannir.extract_physical_layout %arg5 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 64, 64)>, <(-1, -1, 4096, 64, 1)>>, <<(-1, -1, -1, 64, 64)>, <(-1, -1, 4096, 64, 1)>>, identity>
                          %273 = cannir.extract_stride %272 : <<(-1, -1, -1, 64, 64)>, <(-1, -1, 4096, 64, 1)>> -> <(-1, -1, 4096, 64, 1)>
                          %274 = cannir.extract_dim %273[0] : !cannir.stride<(-1, -1, 4096, 64, 1)>
                          %275 = arith.muli %113, %274 : i64
                          %276 = arith.index_cast %275 : i64 to index
                          %277 = cannir.extract_dim %273[1] : !cannir.stride<(-1, -1, 4096, 64, 1)>
                          %278 = arith.muli %110, %277 : i64
                          %279 = arith.index_cast %278 : i64 to index
                          %280 = arith.addi %276, %279 : index
                          %281 = arith.muli %155, %c4096_i64 : i64
                          %282 = arith.index_cast %281 : i64 to index
                          %283 = arith.addi %280, %282 : index
                          asccube.get_buf mte2, %270, 0 {cannir.buffer.generated_sync}
                          asccube.copy_gm2l1 %271[%c0], %alloc_2[%283], %c64_i32, %c64_i32, %c128_i64, %c1_i32, %c0_i64, %c0_i64, %c64_i32 l2_cache_ctl %c1_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
                          asccube.release_buf mte2, %270, 0 {cannir.buffer.generated_sync}
                        }
                        scf.yield %267 : i64
                      } else {
                        scf.yield %arg32 : i64
                      }
                      %240 = cannir.extract_physical_layout %arg6 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                      %241 = cannir.extract_stride %240 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                      %242 = cannir.extract_dim %241[0] : !cannir.stride<(-1, -1, -1, 1)>
                      %243 = arith.muli %96, %242 : i64
                      %244 = arith.index_cast %243 : i64 to index
                      %245 = cannir.extract_dim %241[1] : !cannir.stride<(-1, -1, -1, 1)>
                      %246 = arith.muli %110, %245 : i64
                      %247 = arith.index_cast %246 : i64 to index
                      %248 = arith.addi %244, %247 : index
                      %249 = arith.muli %168, %c64_i64 : i64
                      %250 = arith.addi %249, %95 : i64
                      %251 = cannir.extract_dim %241[2] : !cannir.stride<(-1, -1, -1, 1)>
                      %252 = arith.muli %250, %251 : i64
                      %253 = arith.index_cast %252 : i64 to index
                      %254 = arith.addi %248, %253 : index
                      %255 = arith.muli %112, %c32_i64 overflow<nsw> : i64
                      %256 = arith.index_cast %255 : i64 to index
                      %257 = arith.addi %254, %256 : index
                      %258 = cannir.extract_layout %arg6 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                      %259 = cannir.extract_stride %258 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                      %260 = cannir.extract_dim %259[2] : !cannir.stride<(-1, -1, -1, 1)>
                      asccube.get_buf fixpipe, %c16_i64, 0 {cannir.buffer.generated_sync}
                      %261 = arith.trunci %260 : i64 to i32
                      asccube.copy_l0c2gm %alloc_3[%257], %203[%c0], %c32_i32, %c64_i32, %261, %c64_i32, %c0_i32 l2_cache_ctl %c4_i32 {nz2nd_en = true, relu_en = false} : memref<1xbf16>, memref<?xf32, 5>
                      asccube.release_buf fixpipe, %c16_i64, 0 {cannir.buffer.generated_sync}
                      %262 = scf.if %213 -> (i64) {
                        %263 = arith.addi %arg33, %c1_i64 overflow<nsw> : i64
                        %264 = arith.remui %263, %c2_i64 : i64
                        scf.yield %264 : i64
                      } else {
                        scf.yield %arg33 : i64
                      }
                      scf.yield %214#0, %175, %239, %262, %197, %214#1, %230 : i64, i64, i64, i64, i64, i64, i64
                    }
                    scf.yield %152#0, %152#1, %152#2, %152#3, %152#4, %152#5, %152#6 : i64, i64, i64, i64, i64, i64, i64
                  }
                  scf.yield %100#0, %100#1, %100#2, %100#3, %100#4, %100#5, %100#6 : i64, i64, i64, i64, i64, i64, i64
                } else {
                  scf.yield %arg14, %arg15, %arg16, %arg17, %arg18, %arg19, %arg20 : i64, i64, i64, i64, i64, i64, i64
                }
                scf.yield %99#0, %99#1, %99#2, %99#3, %99#4, %99#5, %99#6 : i64, i64, i64, i64, i64, i64, i64
              }
            } else {
              %50:7 = scf.for %arg13 = %c0_i32 to %21 step %c1_i32 iter_args(%arg14 = %c0_i64, %arg15 = %c0_i64, %arg16 = %c0_i64, %arg17 = %c0_i64, %arg18 = %c0_i64, %arg19 = %c0_i64, %arg20 = %c0_i64) -> (i64, i64, i64, i64, i64, i64, i64)  : i32 {
                %51 = arith.addi %32, %arg13 : i32
                %52 = arith.index_cast %51 : i32 to index
                %53 = memref.load %alloc_4[%52] : memref<1xi32>
                %54 = arith.addi %51, %c1_i32 : i32
                %55 = arith.index_cast %54 : i32 to index
                %56 = memref.load %alloc_4[%55] : memref<1xi32>
                %57 = arith.cmpi slt, %39, %53 : i32
                %58 = arith.select %57, %53, %39 : i32
                %59 = arith.cmpi sgt, %42, %56 : i32
                %60 = arith.select %59, %56, %42 : i32
                %61 = arith.cmpi slt, %58, %60 : i32
                %62 = arith.subi %58, %53 : i32
                %63 = arith.subi %60, %53 : i32
                %64 = arith.extsi %62 : i32 to i64
                %65 = arith.extsi %63 : i32 to i64
                %66 = arith.select %61, %64, %c0_i64 : i64
                %67 = arith.select %61, %65, %c0_i64 : i64
                %68 = arith.addi %22, %arg13 : i32
                %69 = arith.index_cast %68 : i32 to index
                %70 = memref.load %alloc_4[%69] : memref<1xi32>
                %71 = arith.addi %23, %arg13 : i32
                %72 = arith.index_cast %71 : i32 to index
                %73 = memref.load %alloc_4[%72] : memref<1xi32>
                %74 = arith.addi %24, %arg13 : i32
                %75 = arith.index_cast %74 : i32 to index
                %76 = memref.load %alloc_4[%75] : memref<1xi32>
                %77 = arith.addi %25, %arg13 : i32
                %78 = arith.index_cast %77 : i32 to index
                %79 = memref.load %alloc_4[%78] : memref<1xi32>
                %80 = arith.extsi %79 : i32 to i64
                %81 = arith.addi %27, %arg13 : i32
                %82 = arith.index_cast %81 : i32 to index
                %83 = memref.load %alloc_4[%82] : memref<1xi32>
                %84 = arith.addi %28, %arg13 : i32
                %85 = arith.index_cast %84 : i32 to index
                %86 = memref.load %alloc_4[%85] : memref<1xi32>
                %87 = arith.addi %80, %c63_i64 : i64
                %88 = arith.remsi %87, %c64_i64 : i64
                %89 = arith.cmpi ne, %88, %c0_i64 : i64
                %90 = arith.cmpi slt, %88, %c0_i64 : i64
                %91 = arith.andi %89, %90 : i1
                %92 = arith.divsi %87, %c64_i64 : i64
                %93 = arith.extui %91 : i1 to i64
                %94 = arith.subi %92, %93 : i64
                %95 = arith.extsi %76 : i32 to i64
                %96 = arith.extsi %73 : i32 to i64
                %97 = ascscalar.get_subblock_id
                %98 = arith.cmpi slt, %0, %1 : i64
                %99:7 = scf.if %98 -> (i64, i64, i64, i64, i64, i64, i64) {
                  %100:7 = scf.for %arg21 = %66 to %67 step %c1_i64 iter_args(%arg22 = %arg14, %arg23 = %arg15, %arg24 = %arg16, %arg25 = %arg17, %arg26 = %arg18, %arg27 = %arg19, %arg28 = %arg20) -> (i64, i64, i64, i64, i64, i64, i64)  : i64 {
                    %101 = arith.remsi %arg21, %c8_i64 : i64
                    %102 = arith.cmpi slt, %101, %c0_i64 : i64
                    %103 = arith.divsi %arg21, %c8_i64 : i64
                    %104 = arith.extui %102 : i1 to i64
                    %105 = arith.subi %103, %104 : i64
                    %106 = arith.maxsi %7, %c1_i64 : i64
                    %107 = arith.remsi %105, %106 : i64
                    %108 = arith.cmpi slt, %107, %c0_i64 : i64
                    %109 = arith.select %108, %106, %c0_i64 : i64
                    %110 = arith.addi %107, %109 : i64
                    %111 = arith.select %102, %c8_i64, %c0_i64 : i64
                    %112 = arith.addi %101, %111 overflow<nsw> : i64
                    %113 = arith.extsi %70 : i32 to i64
                    %114 = arith.addi %83, %86 : i32
                    %115 = arith.extsi %114 : i32 to i64
                    %116 = arith.cmpi eq, %115, %94 : i64
                    %117 = arith.muli %83, %c64_i32 : i32
                    %118 = arith.extsi %117 : i32 to i64
                    %119 = arith.subi %80, %118 : i64
                    %120 = arith.minsi %119, %c64_i64 : i64
                    %121 = arith.select %116, %120, %c64_i64 : i64
                    %122 = arith.extsi %83 : i32 to i64
                    %123 = arith.cmpi eq, %arg22, %c0_i64 : i64
                    %124 = ascscalar.memref_address %alloc_24 : memref<8192xbf16, 2>
                    %125 = ascscalar.memref_address %alloc_23 : memref<8192xbf16, 2>
                    %126 = arith.select %123, %124, %125 : i64
                    %127 = arith.select %123, %c2_i64, %c3_i64 : i64
                    %128 = arith.addi %arg22, %c1_i64 : i64
                    %129 = arith.remui %128, %c2_i64 : i64
                    %130 = arith.cmpi slt, %121, %c64_i64 : i64
                    scf.if %130 {
                      asccube.get_buf mte2, %127, 0 {cannir.buffer.generated_sync}
                      scf.for %arg29 = %c0_i64 to %c8_i64 step %c1_i64  : i64 {
                        %153 = ascscalar.bind_local_ptr %126 : memref<1xbf16, 2>
                        %154 = cannir.extract_physical_layout %arg3 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                        %155 = cannir.extract_stride %154 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                        %156 = cannir.extract_dim %155[0] : !cannir.stride<(-1, -1, -1, 1)>
                        %157 = arith.muli %96, %156 : i64
                        %158 = arith.index_cast %157 : i64 to index
                        %159 = cannir.extract_dim %155[1] : !cannir.stride<(-1, -1, -1, 1)>
                        %160 = arith.muli %110, %159 : i64
                        %161 = arith.index_cast %160 : i64 to index
                        %162 = arith.addi %158, %161 : index
                        %163 = arith.muli %122, %c64_i64 : i64
                        %164 = arith.addi %163, %95 : i64
                        %165 = cannir.extract_dim %155[2] : !cannir.stride<(-1, -1, -1, 1)>
                        %166 = arith.muli %164, %165 : i64
                        %167 = arith.index_cast %166 : i64 to index
                        %168 = arith.addi %162, %167 : index
                        %169 = arith.muli %arg29, %c16_i64 overflow<nsw> : i64
                        %170 = arith.index_cast %169 : i64 to index
                        %171 = arith.addi %168, %170 : index
                        %172 = cannir.extract_layout %arg3 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                        %173 = cannir.extract_stride %172 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                        %174 = cannir.extract_dim %173[2] : !cannir.stride<(-1, -1, -1, 1)>
                        %175 = arith.index_cast %174 : i64 to index
                        %176 = arith.muli %arg29, %c1024_i64 overflow<nsw> : i64
                        %177 = arith.muli %175, %c2 : index
                        %178 = arith.index_cast %177 : index to i64
                        %179 = arith.cmpi ugt, %121, %c0_i64 : i64
                        scf.if %179 {
                          %180 = arith.index_cast %176 : i64 to index
                          %181 = arith.trunci %121 : i64 to i32
                          asccube.copy_gm2l1 %153[%180], %alloc_1[%171], %181, %c16_i32, %178, %c1_i32, %c0_i64, %c0_i64, %c64_i32 l2_cache_ctl %c0_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
                        }
                      }
                      asccube.release_buf mte2, %127, 0 {cannir.buffer.generated_sync}
                    } else {
                      %153 = ascscalar.bind_local_ptr %126 : memref<1xbf16, 2>
                      %154 = cannir.extract_physical_layout %arg3 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                      %155 = cannir.extract_stride %154 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                      %156 = cannir.extract_dim %155[0] : !cannir.stride<(-1, -1, -1, 1)>
                      %157 = arith.muli %96, %156 : i64
                      %158 = arith.index_cast %157 : i64 to index
                      %159 = cannir.extract_dim %155[1] : !cannir.stride<(-1, -1, -1, 1)>
                      %160 = arith.muli %110, %159 : i64
                      %161 = arith.index_cast %160 : i64 to index
                      %162 = arith.addi %158, %161 : index
                      %163 = arith.muli %122, %c64_i64 : i64
                      %164 = arith.addi %163, %95 : i64
                      %165 = cannir.extract_dim %155[2] : !cannir.stride<(-1, -1, -1, 1)>
                      %166 = arith.muli %164, %165 : i64
                      %167 = arith.index_cast %166 : i64 to index
                      %168 = arith.addi %162, %167 : index
                      %169 = arith.divsi %80, %c64_i64 : i64
                      %170 = arith.minsi %122, %169 : i64
                      %171 = arith.muli %170, %c64_i64 : i64
                      %172 = arith.subi %80, %171 : i64
                      %173 = arith.minsi %172, %c64_i64 : i64
                      %174 = arith.cmpi sle, %122, %169 : i64
                      %175 = arith.select %174, %173, %c0_i64 : i64
                      %176 = cannir.extract_layout %arg3 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                      %177 = cannir.extract_stride %176 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                      %178 = cannir.extract_dim %177[2] : !cannir.stride<(-1, -1, -1, 1)>
                      %179 = arith.index_cast %178 : i64 to index
                      %180 = arith.muli %179, %c2 : index
                      %181 = arith.index_cast %180 : index to i64
                      asccube.get_buf mte2, %127, 0 {cannir.buffer.generated_sync}
                      %182 = arith.cmpi ugt, %175, %c0_i64 : i64
                      scf.if %182 {
                        %183 = arith.trunci %175 : i64 to i32
                        asccube.copy_gm2l1 %153[%c0], %alloc_1[%168], %183, %c128_i32, %181, %c1_i32, %c0_i64, %c0_i64, %c64_i32 l2_cache_ctl %c0_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
                      }
                      asccube.release_buf mte2, %127, 0 {cannir.buffer.generated_sync}
                    }
                    %131 = ascscalar.memref_address %alloc_26 : memref<8192xbf16, 2>
                    %132 = ascscalar.memref_address %alloc_25 : memref<8192xbf16, 2>
                    %133 = arith.select %123, %131, %132 : i64
                    %134 = arith.cmpi ne, %arg22, %c0_i64 : i64
                    %135 = arith.extui %134 : i1 to i64
                    scf.if %130 {
                      asccube.get_buf mte2, %135, 0 {cannir.buffer.generated_sync}
                      scf.for %arg29 = %c0_i64 to %c8_i64 step %c1_i64  : i64 {
                        %153 = ascscalar.bind_local_ptr %133 : memref<1xbf16, 2>
                        %154 = cannir.extract_physical_layout %arg2 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                        %155 = cannir.extract_stride %154 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                        %156 = cannir.extract_dim %155[0] : !cannir.stride<(-1, -1, -1, 1)>
                        %157 = arith.muli %96, %156 : i64
                        %158 = arith.index_cast %157 : i64 to index
                        %159 = cannir.extract_dim %155[1] : !cannir.stride<(-1, -1, -1, 1)>
                        %160 = arith.muli %110, %159 : i64
                        %161 = arith.index_cast %160 : i64 to index
                        %162 = arith.addi %158, %161 : index
                        %163 = arith.muli %122, %c64_i64 : i64
                        %164 = arith.addi %163, %95 : i64
                        %165 = cannir.extract_dim %155[2] : !cannir.stride<(-1, -1, -1, 1)>
                        %166 = arith.muli %164, %165 : i64
                        %167 = arith.index_cast %166 : i64 to index
                        %168 = arith.addi %162, %167 : index
                        %169 = arith.muli %arg29, %c16_i64 overflow<nsw> : i64
                        %170 = arith.index_cast %169 : i64 to index
                        %171 = arith.addi %168, %170 : index
                        %172 = cannir.extract_layout %arg2 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                        %173 = cannir.extract_stride %172 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                        %174 = cannir.extract_dim %173[2] : !cannir.stride<(-1, -1, -1, 1)>
                        %175 = arith.index_cast %174 : i64 to index
                        %176 = arith.muli %arg29, %c1024_i64 overflow<nsw> : i64
                        %177 = arith.muli %175, %c2 : index
                        %178 = arith.index_cast %177 : index to i64
                        %179 = arith.cmpi ugt, %121, %c0_i64 : i64
                        scf.if %179 {
                          %180 = arith.index_cast %176 : i64 to index
                          %181 = arith.trunci %121 : i64 to i32
                          asccube.copy_gm2l1 %153[%180], %alloc_0[%171], %181, %c16_i32, %178, %c1_i32, %c0_i64, %c0_i64, %c64_i32 l2_cache_ctl %c0_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
                        }
                      }
                      asccube.release_buf mte2, %135, 0 {cannir.buffer.generated_sync}
                    } else {
                      %153 = ascscalar.bind_local_ptr %133 : memref<1xbf16, 2>
                      %154 = cannir.extract_physical_layout %arg2 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                      %155 = cannir.extract_stride %154 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                      %156 = cannir.extract_dim %155[0] : !cannir.stride<(-1, -1, -1, 1)>
                      %157 = arith.muli %96, %156 : i64
                      %158 = arith.index_cast %157 : i64 to index
                      %159 = cannir.extract_dim %155[1] : !cannir.stride<(-1, -1, -1, 1)>
                      %160 = arith.muli %110, %159 : i64
                      %161 = arith.index_cast %160 : i64 to index
                      %162 = arith.addi %158, %161 : index
                      %163 = arith.muli %122, %c64_i64 : i64
                      %164 = arith.addi %163, %95 : i64
                      %165 = cannir.extract_dim %155[2] : !cannir.stride<(-1, -1, -1, 1)>
                      %166 = arith.muli %164, %165 : i64
                      %167 = arith.index_cast %166 : i64 to index
                      %168 = arith.addi %162, %167 : index
                      %169 = arith.divsi %80, %c64_i64 : i64
                      %170 = arith.minsi %122, %169 : i64
                      %171 = arith.muli %170, %c64_i64 : i64
                      %172 = arith.subi %80, %171 : i64
                      %173 = arith.minsi %172, %c64_i64 : i64
                      %174 = arith.cmpi sle, %122, %169 : i64
                      %175 = arith.select %174, %173, %c0_i64 : i64
                      %176 = cannir.extract_layout %arg2 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                      %177 = cannir.extract_stride %176 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                      %178 = cannir.extract_dim %177[2] : !cannir.stride<(-1, -1, -1, 1)>
                      %179 = arith.index_cast %178 : i64 to index
                      %180 = arith.muli %179, %c2 : index
                      %181 = arith.index_cast %180 : index to i64
                      asccube.get_buf mte2, %135, 0 {cannir.buffer.generated_sync}
                      %182 = arith.cmpi ugt, %175, %c0_i64 : i64
                      scf.if %182 {
                        %183 = arith.trunci %175 : i64 to i32
                        asccube.copy_gm2l1 %153[%c0], %alloc_0[%168], %183, %c128_i32, %181, %c1_i32, %c0_i64, %c0_i64, %c64_i32 l2_cache_ctl %c0_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
                      }
                      asccube.release_buf mte2, %135, 0 {cannir.buffer.generated_sync}
                    }
                    %136 = arith.cmpi eq, %arg24, %c0_i64 : i64
                    %137 = ascscalar.memref_address %alloc_20 : memref<8192xbf16, 2>
                    %138 = ascscalar.memref_address %alloc_19 : memref<8192xbf16, 2>
                    %139 = arith.select %136, %137, %138 : i64
                    %140 = arith.select %136, %c4_i64, %c5_i64 : i64
                    %141 = arith.addi %arg24, %c1_i64 : i64
                    %142 = arith.remui %141, %c2_i64 : i64
                    scf.if %130 {
                      asccube.get_buf mte2, %140, 0 {cannir.buffer.generated_sync}
                      scf.for %arg29 = %c0_i64 to %c8_i64 step %c1_i64  : i64 {
                        %153 = ascscalar.bind_local_ptr %139 : memref<1xbf16, 2>
                        %154 = cannir.extract_physical_layout %arg0 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                        %155 = cannir.extract_stride %154 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                        %156 = cannir.extract_dim %155[0] : !cannir.stride<(-1, -1, -1, 1)>
                        %157 = arith.muli %96, %156 : i64
                        %158 = arith.index_cast %157 : i64 to index
                        %159 = cannir.extract_dim %155[1] : !cannir.stride<(-1, -1, -1, 1)>
                        %160 = arith.muli %110, %159 : i64
                        %161 = arith.index_cast %160 : i64 to index
                        %162 = arith.addi %158, %161 : index
                        %163 = arith.muli %122, %c64_i64 : i64
                        %164 = arith.addi %163, %95 : i64
                        %165 = cannir.extract_dim %155[2] : !cannir.stride<(-1, -1, -1, 1)>
                        %166 = arith.muli %164, %165 : i64
                        %167 = arith.index_cast %166 : i64 to index
                        %168 = arith.addi %162, %167 : index
                        %169 = arith.muli %arg29, %c16_i64 overflow<nsw> : i64
                        %170 = arith.index_cast %169 : i64 to index
                        %171 = arith.addi %168, %170 : index
                        %172 = cannir.extract_layout %arg0 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                        %173 = cannir.extract_stride %172 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                        %174 = cannir.extract_dim %173[2] : !cannir.stride<(-1, -1, -1, 1)>
                        %175 = arith.index_cast %174 : i64 to index
                        %176 = arith.muli %arg29, %c1024_i64 overflow<nsw> : i64
                        %177 = arith.muli %175, %c2 : index
                        %178 = arith.index_cast %177 : index to i64
                        %179 = arith.cmpi ugt, %121, %c0_i64 : i64
                        scf.if %179 {
                          %180 = arith.index_cast %176 : i64 to index
                          %181 = arith.trunci %121 : i64 to i32
                          asccube.copy_gm2l1 %153[%180], %alloc[%171], %181, %c16_i32, %178, %c1_i32, %c0_i64, %c0_i64, %c64_i32 l2_cache_ctl %c1_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
                        }
                      }
                      asccube.release_buf mte2, %140, 0 {cannir.buffer.generated_sync}
                    } else {
                      %153 = ascscalar.bind_local_ptr %139 : memref<1xbf16, 2>
                      %154 = cannir.extract_physical_layout %arg0 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                      %155 = cannir.extract_stride %154 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                      %156 = cannir.extract_dim %155[0] : !cannir.stride<(-1, -1, -1, 1)>
                      %157 = arith.muli %96, %156 : i64
                      %158 = arith.index_cast %157 : i64 to index
                      %159 = cannir.extract_dim %155[1] : !cannir.stride<(-1, -1, -1, 1)>
                      %160 = arith.muli %110, %159 : i64
                      %161 = arith.index_cast %160 : i64 to index
                      %162 = arith.addi %158, %161 : index
                      %163 = arith.muli %122, %c64_i64 : i64
                      %164 = arith.addi %163, %95 : i64
                      %165 = cannir.extract_dim %155[2] : !cannir.stride<(-1, -1, -1, 1)>
                      %166 = arith.muli %164, %165 : i64
                      %167 = arith.index_cast %166 : i64 to index
                      %168 = arith.addi %162, %167 : index
                      %169 = arith.divsi %80, %c64_i64 : i64
                      %170 = arith.minsi %122, %169 : i64
                      %171 = arith.muli %170, %c64_i64 : i64
                      %172 = arith.subi %80, %171 : i64
                      %173 = arith.minsi %172, %c64_i64 : i64
                      %174 = arith.cmpi sle, %122, %169 : i64
                      %175 = arith.select %174, %173, %c0_i64 : i64
                      %176 = cannir.extract_layout %arg0 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                      %177 = cannir.extract_stride %176 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                      %178 = cannir.extract_dim %177[2] : !cannir.stride<(-1, -1, -1, 1)>
                      %179 = arith.index_cast %178 : i64 to index
                      %180 = arith.muli %179, %c2 : index
                      %181 = arith.index_cast %180 : index to i64
                      asccube.get_buf mte2, %140, 0 {cannir.buffer.generated_sync}
                      %182 = arith.cmpi ugt, %175, %c0_i64 : i64
                      scf.if %182 {
                        %183 = arith.trunci %175 : i64 to i32
                        asccube.copy_gm2l1 %153[%c0], %alloc[%168], %183, %c128_i32, %181, %c1_i32, %c0_i64, %c0_i64, %c64_i32 l2_cache_ctl %c1_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
                      }
                      asccube.release_buf mte2, %140, 0 {cannir.buffer.generated_sync}
                    }
                    %143 = ascscalar.memref_address %alloc_18 : memref<4096xbf16, 2>
                    %144 = ascscalar.memref_address %alloc_17 : memref<4096xbf16, 2>
                    %145 = arith.select %136, %143, %144 : i64
                    %146 = arith.select %136, %c6_i64, %c7_i64 : i64
                    scf.if %130 {
                      asccube.get_buf mte2, %146, 0 {cannir.buffer.generated_sync}
                      scf.for %arg29 = %c0_i64 to %c4_i64 step %c1_i64  : i64 {
                        %153 = ascscalar.bind_local_ptr %145 : memref<1xbf16, 2>
                        %154 = cannir.extract_physical_layout %arg5 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 64, 64)>, <(-1, -1, 4096, 64, 1)>>, <<(-1, -1, -1, 64, 64)>, <(-1, -1, 4096, 64, 1)>>, identity>
                        %155 = cannir.extract_stride %154 : <<(-1, -1, -1, 64, 64)>, <(-1, -1, 4096, 64, 1)>> -> <(-1, -1, 4096, 64, 1)>
                        %156 = cannir.extract_dim %155[0] : !cannir.stride<(-1, -1, 4096, 64, 1)>
                        %157 = arith.muli %113, %156 : i64
                        %158 = arith.index_cast %157 : i64 to index
                        %159 = cannir.extract_dim %155[1] : !cannir.stride<(-1, -1, 4096, 64, 1)>
                        %160 = arith.muli %110, %159 : i64
                        %161 = arith.index_cast %160 : i64 to index
                        %162 = arith.addi %158, %161 : index
                        %163 = arith.muli %122, %c4096_i64 : i64
                        %164 = arith.index_cast %163 : i64 to index
                        %165 = arith.addi %162, %164 : index
                        %166 = arith.muli %arg29, %c16_i64 overflow<nsw> : i64
                        %167 = arith.index_cast %166 : i64 to index
                        %168 = arith.addi %165, %167 : index
                        %169 = arith.muli %arg29, %c1024_i64 overflow<nsw> : i64
                        %170 = arith.cmpi ugt, %121, %c0_i64 : i64
                        scf.if %170 {
                          %171 = arith.index_cast %169 : i64 to index
                          %172 = arith.trunci %121 : i64 to i32
                          asccube.copy_gm2l1 %153[%171], %alloc_2[%168], %172, %c16_i32, %c128_i64, %c1_i32, %c0_i64, %c0_i64, %c64_i32 l2_cache_ctl %c1_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
                        }
                      }
                      asccube.release_buf mte2, %146, 0 {cannir.buffer.generated_sync}
                    } else {
                      %153 = ascscalar.bind_local_ptr %145 : memref<1xbf16, 2>
                      %154 = cannir.extract_physical_layout %arg5 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 64, 64)>, <(-1, -1, 4096, 64, 1)>>, <<(-1, -1, -1, 64, 64)>, <(-1, -1, 4096, 64, 1)>>, identity>
                      %155 = cannir.extract_stride %154 : <<(-1, -1, -1, 64, 64)>, <(-1, -1, 4096, 64, 1)>> -> <(-1, -1, 4096, 64, 1)>
                      %156 = cannir.extract_dim %155[0] : !cannir.stride<(-1, -1, 4096, 64, 1)>
                      %157 = arith.muli %113, %156 : i64
                      %158 = arith.index_cast %157 : i64 to index
                      %159 = cannir.extract_dim %155[1] : !cannir.stride<(-1, -1, 4096, 64, 1)>
                      %160 = arith.muli %110, %159 : i64
                      %161 = arith.index_cast %160 : i64 to index
                      %162 = arith.addi %158, %161 : index
                      %163 = arith.muli %122, %c4096_i64 : i64
                      %164 = arith.index_cast %163 : i64 to index
                      %165 = arith.addi %162, %164 : index
                      asccube.get_buf mte2, %146, 0 {cannir.buffer.generated_sync}
                      asccube.copy_gm2l1 %153[%c0], %alloc_2[%165], %c64_i32, %c64_i32, %c128_i64, %c1_i32, %c0_i64, %c0_i64, %c64_i32 l2_cache_ctl %c1_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
                      asccube.release_buf mte2, %146, 0 {cannir.buffer.generated_sync}
                    }
                    %147 = arith.cmpi eq, %83, %c0_i32 : i32
                    %148 = scf.if %147 -> (i64) {
                      %153 = arith.addi %arg25, %c1_i64 : i64
                      %154 = arith.remui %153, %c2_i64 : i64
                      scf.yield %154 : i64
                    } else {
                      %153 = arith.addi %arg25, %c1_i64 : i64
                      %154 = arith.remui %153, %c2_i64 : i64
                      scf.yield %154 : i64
                    }
                    %149 = arith.muli %97, %c32_i64 : i64
                    %150 = arith.cmpi slt, %149, %121 : i64
                    %151 = scf.if %150 -> (i64) {
                      %153 = arith.addi %arg27, %c1_i64 : i64
                      %154 = arith.remui %153, %c2_i64 : i64
                      scf.yield %154 : i64
                    } else {
                      scf.yield %arg27 : i64
                    }
                    %152:7 = scf.for %arg29 = %c0_i32 to %86 step %c1_i32 iter_args(%arg30 = %129, %arg31 = %arg23, %arg32 = %142, %arg33 = %148, %arg34 = %arg26, %arg35 = %151, %arg36 = %arg28) -> (i64, i64, i64, i64, i64, i64, i64)  : i32 {
                      %153 = arith.addi %83, %arg29 : i32
                      %154 = arith.addi %153, %c1_i32 : i32
                      %155 = arith.extsi %154 : i32 to i64
                      %156 = arith.cmpi eq, %155, %94 : i64
                      %157 = arith.muli %153, %c64_i32 : i32
                      %158 = arith.extsi %157 : i32 to i64
                      %159 = arith.subi %80, %158 : i64
                      %160 = arith.select %156, %159, %c64_i64 : i64
                      %161 = arith.addi %153, %c2_i32 : i32
                      %162 = arith.extsi %161 : i32 to i64
                      %163 = arith.cmpi eq, %162, %94 : i64
                      %164 = arith.muli %154, %c64_i32 : i32
                      %165 = arith.extsi %164 : i32 to i64
                      %166 = arith.subi %80, %165 : i64
                      %167 = arith.select %163, %166, %c64_i64 : i64
                      %168 = arith.extsi %153 : i32 to i64
                      %169 = arith.cmpi eq, %arg31, %c0_i64 : i64
                      %170 = ascscalar.memref_address %alloc_16 : memref<16384xbf16, 2>
                      %171 = ascscalar.memref_address %alloc_15 : memref<16384xbf16, 2>
                      %172 = arith.select %169, %170, %171 : i64
                      %173 = arith.select %169, %c8_i64, %c9_i64 : i64
                      %174 = arith.addi %arg31, %c1_i64 : i64
                      %175 = arith.remui %174, %c2_i64 : i64
                      %176 = ascscalar.bitcast_memref %alloc_8 byte_offset = %c0_i64 : memref<16384xbf16, 4> -> memref<?xbf16, 4>
                      %177 = ascscalar.bind_local_ptr %172 : memref<1xbf16, 2>
                      %178 = arith.cmpi eq, %173, %c9_i64 : i64
                      %179 = arith.extui %178 : i1 to i64
                      asccube.sync_intra_wait mte1, %179 {cannir.sync.banks = 3 : i32}
                      %180 = arith.addi %179, %c16_i64 overflow<nsw> : i64
                      asccube.sync_intra_wait mte1, %180 {cannir.sync.banks = 3 : i32}
                      asccube.get_buf mte1, %c13_i64, 0 {cannir.buffer.generated_sync}
                      asccube.copy_l12l0b %176[%c0], %177[%c0], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c1_i32, %c1_i32 : memref<?xbf16, 4>, memref<1xbf16, 2>
                      asccube.release_buf mte1, %c13_i64, 0 {cannir.buffer.generated_sync}
                      asccube.sync_intra_arrive mte1, %179 {cannir.sync.banks = 3 : i32}
                      asccube.sync_intra_arrive mte1, %180 {cannir.sync.banks = 3 : i32}
                      %181 = arith.cmpi eq, %arg34, %c0_i64 : i64
                      %182 = ascscalar.memref_address %alloc_28 : memref<8192xbf16, 3>
                      %183 = ascscalar.memref_address %alloc_27 : memref<8192xbf16, 3>
                      %184 = arith.select %181, %182, %183 : i64
                      %185 = arith.select %181, %c8_i64, %c9_i64 : i64
                      %186 = arith.addi %arg34, %c1_i64 : i64
                      %187 = arith.remui %186, %c2_i64 : i64
                      %188 = arith.select %169, %124, %125 : i64
                      %189 = arith.select %169, %c2_i64, %c3_i64 : i64
                      %190 = ascscalar.bind_local_ptr %184 : memref<1xbf16, 3>
                      %191 = ascscalar.bind_local_ptr %188 : memref<1xbf16, 2>
                      asccube.get_buf mte1, %185, 0 {cannir.buffer.generated_sync}
                      asccube.get_buf mte1, %189, 0 {cannir.buffer.generated_sync}
                      asccube.copy_l12l0a %190[%c0], %191[%c0], %c0_i32, %c0_i32, %c4_i32, %c8_i32, %c4_i32, %c4_i32 : memref<1xbf16, 3>, memref<1xbf16, 2>
                      asccube.release_buf mte1, %189, 0 {cannir.buffer.generated_sync}
                      asccube.release_buf mte1, %185, 0 {cannir.buffer.generated_sync}
                      %192 = ascscalar.bitcast_memref %alloc_7 byte_offset = %c0_i64 : memref<8192xf32, 5> -> memref<?xf32, 5>
                      asccube.get_buf m, %c15_i64, 0 {cannir.buffer.generated_sync}
                      asccube.get_buf m, %185, 0 {cannir.buffer.generated_sync}
                      asccube.get_buf m, %c13_i64, 0 {cannir.buffer.generated_sync}
                      asccube.mmad %192[%c0], %190, %176, %c64_i32, %c128_i32, %c16_i32, %c0_i32 {c_matrix_init_val = true} : memref<?xf32, 5>, memref<1xbf16, 3>, memref<?xbf16, 4>
                      asccube.release_buf m, %185, 0 {cannir.buffer.generated_sync}
                      asccube.release_buf m, %c15_i64, 0 {cannir.buffer.generated_sync}
                      %193 = arith.cmpi eq, %187, %c0_i64 : i64
                      %194 = arith.select %193, %182, %183 : i64
                      %195 = arith.select %193, %c8_i64, %c9_i64 : i64
                      %196 = arith.addi %187, %c1_i64 overflow<nsw> : i64
                      %197 = arith.remui %196, %c2_i64 : i64
                      %198 = arith.select %169, %131, %132 : i64
                      %199 = arith.cmpi ne, %arg31, %c0_i64 : i64
                      %200 = arith.extui %199 : i1 to i64
                      %201 = ascscalar.bind_local_ptr %194 : memref<1xbf16, 3>
                      %202 = ascscalar.bind_local_ptr %198 : memref<1xbf16, 2>
                      asccube.get_buf mte1, %195, 0 {cannir.buffer.generated_sync}
                      asccube.get_buf mte1, %200, 0 {cannir.buffer.generated_sync}
                      asccube.copy_l12l0a %201[%c0], %202[%c0], %c0_i32, %c0_i32, %c4_i32, %c8_i32, %c4_i32, %c4_i32 : memref<1xbf16, 3>, memref<1xbf16, 2>
                      asccube.release_buf mte1, %200, 0 {cannir.buffer.generated_sync}
                      asccube.release_buf mte1, %195, 0 {cannir.buffer.generated_sync}
                      %203 = ascscalar.bitcast_memref %alloc_9 byte_offset = %c0_i64 : memref<8192xf32, 5> -> memref<?xf32, 5>
                      asccube.get_buf m, %c16_i64, 0 {cannir.buffer.generated_sync}
                      asccube.get_buf m, %195, 0 {cannir.buffer.generated_sync}
                      asccube.mmad %203[%c0], %201, %176, %c64_i32, %c128_i32, %c16_i32, %c0_i32 {c_matrix_init_val = true} : memref<?xf32, 5>, memref<1xbf16, 3>, memref<?xbf16, 4>
                      asccube.release_buf m, %c13_i64, 0 {cannir.buffer.generated_sync}
                      asccube.release_buf m, %195, 0 {cannir.buffer.generated_sync}
                      asccube.release_buf m, %c16_i64, 0 {cannir.buffer.generated_sync}
                      %204 = arith.select %169, %137, %138 : i64
                      %205 = arith.select %169, %c4_i64, %c5_i64 : i64
                      %206 = ascscalar.bitcast_memref %alloc_12 byte_offset = %c0_i64 : memref<8192xbf16, 4> -> memref<?xbf16, 4>
                      %207 = ascscalar.bind_local_ptr %204 : memref<1xbf16, 2>
                      asccube.get_buf mte1, %c14_i64, 0 {cannir.buffer.generated_sync}
                      asccube.get_buf mte1, %205, 0 {cannir.buffer.generated_sync}
                      asccube.copy_l12l0b %206[%c0], %207[%c0], %c0_i32, %c0_i32, %c4_i32, %c8_i32, %c4_i32, %c8_i32 {transpose = true} : memref<?xbf16, 4>, memref<1xbf16, 2>
                      asccube.release_buf mte1, %205, 0 {cannir.buffer.generated_sync}
                      asccube.release_buf mte1, %c14_i64, 0 {cannir.buffer.generated_sync}
                      %208 = arith.select %169, %143, %144 : i64
                      %209 = arith.select %169, %c6_i64, %c7_i64 : i64
                      %210 = ascscalar.bitcast_memref %alloc_13 byte_offset = %c0_i64 : memref<4096xbf16, 3> -> memref<?xbf16, 3>
                      %211 = ascscalar.bind_local_ptr %208 : memref<1xbf16, 2>
                      asccube.get_buf mte1, %c11_i64, 0 {cannir.buffer.generated_sync}
                      asccube.get_buf mte1, %209, 0 {cannir.buffer.generated_sync}
                      asccube.copy_l12l0a %210[%c0], %211[%c0], %c0_i32, %c0_i32, %c4_i32, %c4_i32, %c4_i32, %c4_i32 : memref<?xbf16, 3>, memref<1xbf16, 2>
                      asccube.release_buf mte1, %209, 0 {cannir.buffer.generated_sync}
                      asccube.release_buf mte1, %c11_i64, 0 {cannir.buffer.generated_sync}
                      %212 = arith.addi %arg29, %c1_i32 : i32
                      %213 = arith.cmpi slt, %212, %86 : i32
                      %214:2 = scf.if %213 -> (i64, i64) {
                        %263 = arith.cmpi eq, %arg30, %c0_i64 : i64
                        %264 = arith.select %263, %131, %132 : i64
                        %265 = arith.cmpi ne, %arg30, %c0_i64 : i64
                        %266 = arith.extui %265 : i1 to i64
                        %267 = arith.addi %arg30, %c1_i64 overflow<nsw> : i64
                        %268 = arith.remui %267, %c2_i64 : i64
                        %269 = arith.cmpi slt, %167, %c64_i64 : i64
                        scf.if %269 {
                          asccube.get_buf mte2, %266, 0 {cannir.buffer.generated_sync}
                          scf.for %arg37 = %c0_i64 to %c8_i64 step %c1_i64  : i64 {
                            %274 = ascscalar.bind_local_ptr %264 : memref<1xbf16, 2>
                            %275 = cannir.extract_physical_layout %arg2 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                            %276 = cannir.extract_stride %275 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                            %277 = cannir.extract_dim %276[0] : !cannir.stride<(-1, -1, -1, 1)>
                            %278 = arith.muli %96, %277 : i64
                            %279 = arith.index_cast %278 : i64 to index
                            %280 = cannir.extract_dim %276[1] : !cannir.stride<(-1, -1, -1, 1)>
                            %281 = arith.muli %110, %280 : i64
                            %282 = arith.index_cast %281 : i64 to index
                            %283 = arith.addi %279, %282 : index
                            %284 = arith.muli %155, %c64_i64 : i64
                            %285 = arith.addi %284, %95 : i64
                            %286 = cannir.extract_dim %276[2] : !cannir.stride<(-1, -1, -1, 1)>
                            %287 = arith.muli %285, %286 : i64
                            %288 = arith.index_cast %287 : i64 to index
                            %289 = arith.addi %283, %288 : index
                            %290 = arith.muli %arg37, %c16_i64 overflow<nsw> : i64
                            %291 = arith.index_cast %290 : i64 to index
                            %292 = arith.addi %289, %291 : index
                            %293 = cannir.extract_layout %arg2 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                            %294 = cannir.extract_stride %293 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                            %295 = cannir.extract_dim %294[2] : !cannir.stride<(-1, -1, -1, 1)>
                            %296 = arith.index_cast %295 : i64 to index
                            %297 = arith.muli %arg37, %c1024_i64 overflow<nsw> : i64
                            %298 = arith.muli %296, %c2 : index
                            %299 = arith.index_cast %298 : index to i64
                            %300 = arith.cmpi ugt, %167, %c0_i64 : i64
                            scf.if %300 {
                              %301 = arith.index_cast %297 : i64 to index
                              %302 = arith.trunci %167 : i64 to i32
                              asccube.copy_gm2l1 %274[%301], %alloc_0[%292], %302, %c16_i32, %299, %c1_i32, %c0_i64, %c0_i64, %c64_i32 l2_cache_ctl %c0_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
                            }
                          }
                          asccube.release_buf mte2, %266, 0 {cannir.buffer.generated_sync}
                        } else {
                          %274 = ascscalar.bind_local_ptr %264 : memref<1xbf16, 2>
                          %275 = cannir.extract_physical_layout %arg2 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                          %276 = cannir.extract_stride %275 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                          %277 = cannir.extract_dim %276[0] : !cannir.stride<(-1, -1, -1, 1)>
                          %278 = arith.muli %96, %277 : i64
                          %279 = arith.index_cast %278 : i64 to index
                          %280 = cannir.extract_dim %276[1] : !cannir.stride<(-1, -1, -1, 1)>
                          %281 = arith.muli %110, %280 : i64
                          %282 = arith.index_cast %281 : i64 to index
                          %283 = arith.addi %279, %282 : index
                          %284 = arith.muli %155, %c64_i64 : i64
                          %285 = arith.addi %284, %95 : i64
                          %286 = cannir.extract_dim %276[2] : !cannir.stride<(-1, -1, -1, 1)>
                          %287 = arith.muli %285, %286 : i64
                          %288 = arith.index_cast %287 : i64 to index
                          %289 = arith.addi %283, %288 : index
                          %290 = arith.divsi %80, %c64_i64 : i64
                          %291 = arith.minsi %155, %290 : i64
                          %292 = arith.muli %291, %c64_i64 : i64
                          %293 = arith.subi %80, %292 : i64
                          %294 = arith.minsi %293, %c64_i64 : i64
                          %295 = arith.cmpi sle, %155, %290 : i64
                          %296 = arith.select %295, %294, %c0_i64 : i64
                          %297 = cannir.extract_layout %arg2 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                          %298 = cannir.extract_stride %297 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                          %299 = cannir.extract_dim %298[2] : !cannir.stride<(-1, -1, -1, 1)>
                          %300 = arith.index_cast %299 : i64 to index
                          %301 = arith.muli %300, %c2 : index
                          %302 = arith.index_cast %301 : index to i64
                          asccube.get_buf mte2, %266, 0 {cannir.buffer.generated_sync}
                          %303 = arith.cmpi ugt, %296, %c0_i64 : i64
                          scf.if %303 {
                            %304 = arith.trunci %296 : i64 to i32
                            asccube.copy_gm2l1 %274[%c0], %alloc_0[%289], %304, %c128_i32, %302, %c1_i32, %c0_i64, %c0_i64, %c64_i32 l2_cache_ctl %c0_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
                          }
                          asccube.release_buf mte2, %266, 0 {cannir.buffer.generated_sync}
                        }
                        %270 = arith.select %263, %124, %125 : i64
                        %271 = arith.select %263, %c2_i64, %c3_i64 : i64
                        scf.if %269 {
                          asccube.get_buf mte2, %271, 0 {cannir.buffer.generated_sync}
                          scf.for %arg37 = %c0_i64 to %c8_i64 step %c1_i64  : i64 {
                            %274 = ascscalar.bind_local_ptr %270 : memref<1xbf16, 2>
                            %275 = cannir.extract_physical_layout %arg3 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                            %276 = cannir.extract_stride %275 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                            %277 = cannir.extract_dim %276[0] : !cannir.stride<(-1, -1, -1, 1)>
                            %278 = arith.muli %96, %277 : i64
                            %279 = arith.index_cast %278 : i64 to index
                            %280 = cannir.extract_dim %276[1] : !cannir.stride<(-1, -1, -1, 1)>
                            %281 = arith.muli %110, %280 : i64
                            %282 = arith.index_cast %281 : i64 to index
                            %283 = arith.addi %279, %282 : index
                            %284 = arith.muli %155, %c64_i64 : i64
                            %285 = arith.addi %284, %95 : i64
                            %286 = cannir.extract_dim %276[2] : !cannir.stride<(-1, -1, -1, 1)>
                            %287 = arith.muli %285, %286 : i64
                            %288 = arith.index_cast %287 : i64 to index
                            %289 = arith.addi %283, %288 : index
                            %290 = arith.muli %arg37, %c16_i64 overflow<nsw> : i64
                            %291 = arith.index_cast %290 : i64 to index
                            %292 = arith.addi %289, %291 : index
                            %293 = cannir.extract_layout %arg3 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                            %294 = cannir.extract_stride %293 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                            %295 = cannir.extract_dim %294[2] : !cannir.stride<(-1, -1, -1, 1)>
                            %296 = arith.index_cast %295 : i64 to index
                            %297 = arith.muli %arg37, %c1024_i64 overflow<nsw> : i64
                            %298 = arith.muli %296, %c2 : index
                            %299 = arith.index_cast %298 : index to i64
                            %300 = arith.cmpi ugt, %167, %c0_i64 : i64
                            scf.if %300 {
                              %301 = arith.index_cast %297 : i64 to index
                              %302 = arith.trunci %167 : i64 to i32
                              asccube.copy_gm2l1 %274[%301], %alloc_1[%292], %302, %c16_i32, %299, %c1_i32, %c0_i64, %c0_i64, %c64_i32 l2_cache_ctl %c0_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
                            }
                          }
                          asccube.release_buf mte2, %271, 0 {cannir.buffer.generated_sync}
                        } else {
                          %274 = ascscalar.bind_local_ptr %270 : memref<1xbf16, 2>
                          %275 = cannir.extract_physical_layout %arg3 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                          %276 = cannir.extract_stride %275 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                          %277 = cannir.extract_dim %276[0] : !cannir.stride<(-1, -1, -1, 1)>
                          %278 = arith.muli %96, %277 : i64
                          %279 = arith.index_cast %278 : i64 to index
                          %280 = cannir.extract_dim %276[1] : !cannir.stride<(-1, -1, -1, 1)>
                          %281 = arith.muli %110, %280 : i64
                          %282 = arith.index_cast %281 : i64 to index
                          %283 = arith.addi %279, %282 : index
                          %284 = arith.muli %155, %c64_i64 : i64
                          %285 = arith.addi %284, %95 : i64
                          %286 = cannir.extract_dim %276[2] : !cannir.stride<(-1, -1, -1, 1)>
                          %287 = arith.muli %285, %286 : i64
                          %288 = arith.index_cast %287 : i64 to index
                          %289 = arith.addi %283, %288 : index
                          %290 = arith.divsi %80, %c64_i64 : i64
                          %291 = arith.minsi %155, %290 : i64
                          %292 = arith.muli %291, %c64_i64 : i64
                          %293 = arith.subi %80, %292 : i64
                          %294 = arith.minsi %293, %c64_i64 : i64
                          %295 = arith.cmpi sle, %155, %290 : i64
                          %296 = arith.select %295, %294, %c0_i64 : i64
                          %297 = cannir.extract_layout %arg3 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                          %298 = cannir.extract_stride %297 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                          %299 = cannir.extract_dim %298[2] : !cannir.stride<(-1, -1, -1, 1)>
                          %300 = arith.index_cast %299 : i64 to index
                          %301 = arith.muli %300, %c2 : index
                          %302 = arith.index_cast %301 : index to i64
                          asccube.get_buf mte2, %271, 0 {cannir.buffer.generated_sync}
                          %303 = arith.cmpi ugt, %296, %c0_i64 : i64
                          scf.if %303 {
                            %304 = arith.trunci %296 : i64 to i32
                            asccube.copy_gm2l1 %274[%c0], %alloc_1[%289], %304, %c128_i32, %302, %c1_i32, %c0_i64, %c0_i64, %c64_i32 l2_cache_ctl %c0_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
                          }
                          asccube.release_buf mte2, %271, 0 {cannir.buffer.generated_sync}
                        }
                        %272 = arith.cmpi slt, %149, %167 : i64
                        %273 = scf.if %272 -> (i64) {
                          %274 = arith.addi %arg35, %c1_i64 : i64
                          %275 = arith.remui %274, %c2_i64 : i64
                          scf.yield %275 : i64
                        } else {
                          scf.yield %arg35 : i64
                        }
                        scf.yield %268, %273 : i64, i64
                      } else {
                        scf.yield %arg30, %arg35 : i64, i64
                      }
                      %215 = ascscalar.bitcast_memref %alloc_6 byte_offset = %c0_i64 : memref<4096xf32, 1> -> memref<?xf32, 1>
                      asccube.sync_intra_wait fixpipe, %c5_i64 {cannir.sync.banks = 3 : i32}
                      asccube.sync_intra_wait fixpipe, %c21_i64 {cannir.sync.banks = 3 : i32}
                      asccube.get_buf fixpipe, %c15_i64, 0 {cannir.buffer.generated_sync}
                      asccube.copy_l0c2ub %215[%c0], %192[%c0], %c16_i32, %c64_i32, %c16_i32, %c64_i32, %c0_i32 sub_block_id %c0_i32 {dual_dst_ctl = 1 : i32, nz2nd_en = true, relu_en = false} : memref<?xf32, 1>, memref<?xf32, 5>
                      asccube.release_buf fixpipe, %c15_i64, 0 {cannir.buffer.generated_sync}
                      asccube.sync_intra_arrive fixpipe, %c5_i64 {cannir.sync.banks = 3 : i32}
                      asccube.sync_intra_arrive fixpipe, %c21_i64 {cannir.sync.banks = 3 : i32}
                      %216 = ascscalar.memref_address %alloc_22 : memref<8192xbf16, 2>
                      %217 = ascscalar.memref_address %alloc_21 : memref<8192xbf16, 2>
                      %218 = arith.select %169, %216, %217 : i64
                      %219 = arith.select %169, %c10_i64, %c11_i64 : i64
                      %220 = arith.subi %160, %149 : i64
                      %221 = arith.cmpi slt, %220, %c0_i64 : i64
                      %222 = arith.select %221, %c0_i64, %220 : i64
                      %223 = arith.cmpi sgt, %222, %c32_i64 : i64
                      %224 = arith.select %223, %c32_i64, %222 : i64
                      %225 = arith.cmpi eq, %224, %c32_i64 : i64
                      %226 = scf.if %225 -> (i64) {
                        %263 = arith.addi %arg36, %c1_i64 : i64
                        %264 = arith.remui %263, %c2_i64 : i64
                        scf.yield %264 : i64
                      } else {
                        scf.yield %arg36 : i64
                      }
                      %227 = arith.cmpi sgt, %224, %c0_i64 : i64
                      %228 = arith.cmpi slt, %224, %c32_i64 : i64
                      %229 = arith.select %227, %228, %false : i1
                      %230 = scf.if %229 -> (i64) {
                        %263 = arith.addi %226, %c1_i64 : i64
                        %264 = arith.remui %263, %c2_i64 : i64
                        scf.yield %264 : i64
                      } else {
                        scf.yield %226 : i64
                      }
                      %231 = ascscalar.bind_local_ptr %218 : memref<1xbf16, 2>
                      %232 = arith.cmpi eq, %219, %c11_i64 : i64
                      %233 = arith.select %232, %c3_i64, %c2_i64 : i64
                      %234 = ascscalar.bitcast_memref %alloc_11 byte_offset = %c0_i64 : memref<8192xbf16, 3> -> memref<?xbf16, 3>
                      asccube.sync_intra_wait mte1, %233 {cannir.sync.banks = 3 : i32}
                      %235 = arith.addi %233, %c16_i64 overflow<nsw> : i64
                      asccube.sync_intra_wait mte1, %235 {cannir.sync.banks = 3 : i32}
                      asccube.get_buf mte1, %c10_i64, 0 {cannir.buffer.generated_sync}
                      asccube.copy_l12l0a %234[%c0], %231[%c0], %c0_i32, %c0_i32, %c4_i32, %c1_i32, %c4_i32, %c1_i32 {transpose = true} : memref<?xbf16, 3>, memref<1xbf16, 2>
                      asccube.release_buf mte1, %c10_i64, 0 {cannir.buffer.generated_sync}
                      %236 = ascscalar.bitcast_memref %alloc_10 byte_offset = %c0_i64 : memref<16384xf32, 5> -> memref<?xf32, 5>
                      asccube.get_buf m, %c17_i64, 0 {cannir.buffer.generated_sync}
                      asccube.get_buf m, %c10_i64, 0 {cannir.buffer.generated_sync}
                      asccube.get_buf m, %c14_i64, 0 {cannir.buffer.generated_sync}
                      asccube.mmad %236[%c0], %234, %206, %c16_i32, %c64_i32, %c128_i32, %c0_i32 {c_matrix_init_val = true} : memref<?xf32, 5>, memref<?xbf16, 3>, memref<?xbf16, 4>
                      asccube.release_buf m, %c14_i64, 0 {cannir.buffer.generated_sync}
                      asccube.release_buf m, %c10_i64, 0 {cannir.buffer.generated_sync}
                      asccube.release_buf m, %c17_i64, 0 {cannir.buffer.generated_sync}
                      %237 = ascscalar.bitcast_memref %alloc_5 byte_offset = %c0_i64 : memref<8192xf32, 1> -> memref<?xf32, 1>
                      asccube.sync_intra_wait fixpipe, %c4_i64 {cannir.sync.banks = 3 : i32}
                      asccube.sync_intra_wait fixpipe, %c20_i64 {cannir.sync.banks = 3 : i32}
                      asccube.get_buf fixpipe, %c17_i64, 0 {cannir.buffer.generated_sync}
                      asccube.copy_l0c2ub %237[%c0], %236[%c0], %c128_i32, %c16_i32, %c64_i32, %c16_i32, %c0_i32 sub_block_id %c0_i32 {dual_dst_ctl = 2 : i32, nz2nd_en = true, relu_en = false} : memref<?xf32, 1>, memref<?xf32, 5>
                      asccube.release_buf fixpipe, %c17_i64, 0 {cannir.buffer.generated_sync}
                      asccube.sync_intra_arrive fixpipe, %c4_i64 {cannir.sync.banks = 3 : i32}
                      asccube.sync_intra_arrive fixpipe, %c20_i64 {cannir.sync.banks = 3 : i32}
                      %238 = ascscalar.bitcast_memref %alloc_14 byte_offset = %c0_i64 : memref<8192xbf16, 4> -> memref<?xbf16, 4>
                      asccube.get_buf mte1, %c12_i64, 0 {cannir.buffer.generated_sync}
                      asccube.copy_l12l0b %238[%c0], %231[%c0], %c0_i32, %c0_i32, %c4_i32, %c1_i32, %c4_i32, %c1_i32 {transpose = true} : memref<?xbf16, 4>, memref<1xbf16, 2>
                      asccube.release_buf mte1, %c12_i64, 0 {cannir.buffer.generated_sync}
                      asccube.sync_intra_arrive mte1, %233 {cannir.sync.banks = 3 : i32}
                      asccube.sync_intra_arrive mte1, %235 {cannir.sync.banks = 3 : i32}
                      asccube.get_buf m, %c16_i64, 0 {cannir.buffer.generated_sync}
                      asccube.get_buf m, %c11_i64, 0 {cannir.buffer.generated_sync}
                      asccube.get_buf m, %c12_i64, 0 {cannir.buffer.generated_sync}
                      asccube.mmad %203[%c0], %210, %238, %c64_i32, %c64_i32, %c16_i32, %c0_i32 {c_matrix_init_val = false} : memref<?xf32, 5>, memref<?xbf16, 3>, memref<?xbf16, 4>
                      asccube.release_buf m, %c12_i64, 0 {cannir.buffer.generated_sync}
                      asccube.release_buf m, %c11_i64, 0 {cannir.buffer.generated_sync}
                      asccube.release_buf m, %c16_i64, 0 {cannir.buffer.generated_sync}
                      %239 = scf.if %213 -> (i64) {
                        %263 = arith.cmpi eq, %arg32, %c0_i64 : i64
                        %264 = arith.select %263, %137, %138 : i64
                        %265 = arith.select %263, %c4_i64, %c5_i64 : i64
                        %266 = arith.addi %arg32, %c1_i64 overflow<nsw> : i64
                        %267 = arith.remui %266, %c2_i64 : i64
                        %268 = arith.cmpi slt, %167, %c64_i64 : i64
                        scf.if %268 {
                          asccube.get_buf mte2, %265, 0 {cannir.buffer.generated_sync}
                          scf.for %arg37 = %c0_i64 to %c8_i64 step %c1_i64  : i64 {
                            %271 = ascscalar.bind_local_ptr %264 : memref<1xbf16, 2>
                            %272 = cannir.extract_physical_layout %arg0 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                            %273 = cannir.extract_stride %272 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                            %274 = cannir.extract_dim %273[0] : !cannir.stride<(-1, -1, -1, 1)>
                            %275 = arith.muli %96, %274 : i64
                            %276 = arith.index_cast %275 : i64 to index
                            %277 = cannir.extract_dim %273[1] : !cannir.stride<(-1, -1, -1, 1)>
                            %278 = arith.muli %110, %277 : i64
                            %279 = arith.index_cast %278 : i64 to index
                            %280 = arith.addi %276, %279 : index
                            %281 = arith.muli %155, %c64_i64 : i64
                            %282 = arith.addi %281, %95 : i64
                            %283 = cannir.extract_dim %273[2] : !cannir.stride<(-1, -1, -1, 1)>
                            %284 = arith.muli %282, %283 : i64
                            %285 = arith.index_cast %284 : i64 to index
                            %286 = arith.addi %280, %285 : index
                            %287 = arith.muli %arg37, %c16_i64 overflow<nsw> : i64
                            %288 = arith.index_cast %287 : i64 to index
                            %289 = arith.addi %286, %288 : index
                            %290 = cannir.extract_layout %arg0 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                            %291 = cannir.extract_stride %290 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                            %292 = cannir.extract_dim %291[2] : !cannir.stride<(-1, -1, -1, 1)>
                            %293 = arith.index_cast %292 : i64 to index
                            %294 = arith.muli %arg37, %c1024_i64 overflow<nsw> : i64
                            %295 = arith.muli %293, %c2 : index
                            %296 = arith.index_cast %295 : index to i64
                            %297 = arith.cmpi ugt, %167, %c0_i64 : i64
                            scf.if %297 {
                              %298 = arith.index_cast %294 : i64 to index
                              %299 = arith.trunci %167 : i64 to i32
                              asccube.copy_gm2l1 %271[%298], %alloc[%289], %299, %c16_i32, %296, %c1_i32, %c0_i64, %c0_i64, %c64_i32 l2_cache_ctl %c1_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
                            }
                          }
                          asccube.release_buf mte2, %265, 0 {cannir.buffer.generated_sync}
                        } else {
                          %271 = ascscalar.bind_local_ptr %264 : memref<1xbf16, 2>
                          %272 = cannir.extract_physical_layout %arg0 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                          %273 = cannir.extract_stride %272 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                          %274 = cannir.extract_dim %273[0] : !cannir.stride<(-1, -1, -1, 1)>
                          %275 = arith.muli %96, %274 : i64
                          %276 = arith.index_cast %275 : i64 to index
                          %277 = cannir.extract_dim %273[1] : !cannir.stride<(-1, -1, -1, 1)>
                          %278 = arith.muli %110, %277 : i64
                          %279 = arith.index_cast %278 : i64 to index
                          %280 = arith.addi %276, %279 : index
                          %281 = arith.muli %155, %c64_i64 : i64
                          %282 = arith.addi %281, %95 : i64
                          %283 = cannir.extract_dim %273[2] : !cannir.stride<(-1, -1, -1, 1)>
                          %284 = arith.muli %282, %283 : i64
                          %285 = arith.index_cast %284 : i64 to index
                          %286 = arith.addi %280, %285 : index
                          %287 = arith.divsi %80, %c64_i64 : i64
                          %288 = arith.minsi %155, %287 : i64
                          %289 = arith.muli %288, %c64_i64 : i64
                          %290 = arith.subi %80, %289 : i64
                          %291 = arith.minsi %290, %c64_i64 : i64
                          %292 = arith.cmpi sle, %155, %287 : i64
                          %293 = arith.select %292, %291, %c0_i64 : i64
                          %294 = cannir.extract_layout %arg0 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                          %295 = cannir.extract_stride %294 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                          %296 = cannir.extract_dim %295[2] : !cannir.stride<(-1, -1, -1, 1)>
                          %297 = arith.index_cast %296 : i64 to index
                          %298 = arith.muli %297, %c2 : index
                          %299 = arith.index_cast %298 : index to i64
                          asccube.get_buf mte2, %265, 0 {cannir.buffer.generated_sync}
                          %300 = arith.cmpi ugt, %293, %c0_i64 : i64
                          scf.if %300 {
                            %301 = arith.trunci %293 : i64 to i32
                            asccube.copy_gm2l1 %271[%c0], %alloc[%286], %301, %c128_i32, %299, %c1_i32, %c0_i64, %c0_i64, %c64_i32 l2_cache_ctl %c1_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
                          }
                          asccube.release_buf mte2, %265, 0 {cannir.buffer.generated_sync}
                        }
                        %269 = arith.select %263, %143, %144 : i64
                        %270 = arith.select %263, %c6_i64, %c7_i64 : i64
                        scf.if %268 {
                          asccube.get_buf mte2, %270, 0 {cannir.buffer.generated_sync}
                          scf.for %arg37 = %c0_i64 to %c4_i64 step %c1_i64  : i64 {
                            %271 = ascscalar.bind_local_ptr %269 : memref<1xbf16, 2>
                            %272 = cannir.extract_physical_layout %arg5 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 64, 64)>, <(-1, -1, 4096, 64, 1)>>, <<(-1, -1, -1, 64, 64)>, <(-1, -1, 4096, 64, 1)>>, identity>
                            %273 = cannir.extract_stride %272 : <<(-1, -1, -1, 64, 64)>, <(-1, -1, 4096, 64, 1)>> -> <(-1, -1, 4096, 64, 1)>
                            %274 = cannir.extract_dim %273[0] : !cannir.stride<(-1, -1, 4096, 64, 1)>
                            %275 = arith.muli %113, %274 : i64
                            %276 = arith.index_cast %275 : i64 to index
                            %277 = cannir.extract_dim %273[1] : !cannir.stride<(-1, -1, 4096, 64, 1)>
                            %278 = arith.muli %110, %277 : i64
                            %279 = arith.index_cast %278 : i64 to index
                            %280 = arith.addi %276, %279 : index
                            %281 = arith.muli %155, %c4096_i64 : i64
                            %282 = arith.index_cast %281 : i64 to index
                            %283 = arith.addi %280, %282 : index
                            %284 = arith.muli %arg37, %c16_i64 overflow<nsw> : i64
                            %285 = arith.index_cast %284 : i64 to index
                            %286 = arith.addi %283, %285 : index
                            %287 = arith.muli %arg37, %c1024_i64 overflow<nsw> : i64
                            %288 = arith.cmpi ugt, %167, %c0_i64 : i64
                            scf.if %288 {
                              %289 = arith.index_cast %287 : i64 to index
                              %290 = arith.trunci %167 : i64 to i32
                              asccube.copy_gm2l1 %271[%289], %alloc_2[%286], %290, %c16_i32, %c128_i64, %c1_i32, %c0_i64, %c0_i64, %c64_i32 l2_cache_ctl %c1_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
                            }
                          }
                          asccube.release_buf mte2, %270, 0 {cannir.buffer.generated_sync}
                        } else {
                          %271 = ascscalar.bind_local_ptr %269 : memref<1xbf16, 2>
                          %272 = cannir.extract_physical_layout %arg5 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 64, 64)>, <(-1, -1, 4096, 64, 1)>>, <<(-1, -1, -1, 64, 64)>, <(-1, -1, 4096, 64, 1)>>, identity>
                          %273 = cannir.extract_stride %272 : <<(-1, -1, -1, 64, 64)>, <(-1, -1, 4096, 64, 1)>> -> <(-1, -1, 4096, 64, 1)>
                          %274 = cannir.extract_dim %273[0] : !cannir.stride<(-1, -1, 4096, 64, 1)>
                          %275 = arith.muli %113, %274 : i64
                          %276 = arith.index_cast %275 : i64 to index
                          %277 = cannir.extract_dim %273[1] : !cannir.stride<(-1, -1, 4096, 64, 1)>
                          %278 = arith.muli %110, %277 : i64
                          %279 = arith.index_cast %278 : i64 to index
                          %280 = arith.addi %276, %279 : index
                          %281 = arith.muli %155, %c4096_i64 : i64
                          %282 = arith.index_cast %281 : i64 to index
                          %283 = arith.addi %280, %282 : index
                          asccube.get_buf mte2, %270, 0 {cannir.buffer.generated_sync}
                          asccube.copy_gm2l1 %271[%c0], %alloc_2[%283], %c64_i32, %c64_i32, %c128_i64, %c1_i32, %c0_i64, %c0_i64, %c64_i32 l2_cache_ctl %c1_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
                          asccube.release_buf mte2, %270, 0 {cannir.buffer.generated_sync}
                        }
                        scf.yield %267 : i64
                      } else {
                        scf.yield %arg32 : i64
                      }
                      %240 = cannir.extract_physical_layout %arg6 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                      %241 = cannir.extract_stride %240 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                      %242 = cannir.extract_dim %241[0] : !cannir.stride<(-1, -1, -1, 1)>
                      %243 = arith.muli %96, %242 : i64
                      %244 = arith.index_cast %243 : i64 to index
                      %245 = cannir.extract_dim %241[1] : !cannir.stride<(-1, -1, -1, 1)>
                      %246 = arith.muli %110, %245 : i64
                      %247 = arith.index_cast %246 : i64 to index
                      %248 = arith.addi %244, %247 : index
                      %249 = arith.muli %168, %c64_i64 : i64
                      %250 = arith.addi %249, %95 : i64
                      %251 = cannir.extract_dim %241[2] : !cannir.stride<(-1, -1, -1, 1)>
                      %252 = arith.muli %250, %251 : i64
                      %253 = arith.index_cast %252 : i64 to index
                      %254 = arith.addi %248, %253 : index
                      %255 = arith.muli %112, %c16_i64 overflow<nsw> : i64
                      %256 = arith.index_cast %255 : i64 to index
                      %257 = arith.addi %254, %256 : index
                      %258 = cannir.extract_layout %arg6 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
                      %259 = cannir.extract_stride %258 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                      %260 = cannir.extract_dim %259[2] : !cannir.stride<(-1, -1, -1, 1)>
                      asccube.get_buf fixpipe, %c16_i64, 0 {cannir.buffer.generated_sync}
                      %261 = arith.trunci %260 : i64 to i32
                      asccube.copy_l0c2gm %alloc_3[%257], %203[%c0], %c16_i32, %c64_i32, %261, %c64_i32, %c0_i32 l2_cache_ctl %c4_i32 {nz2nd_en = true, relu_en = false} : memref<1xbf16>, memref<?xf32, 5>
                      asccube.release_buf fixpipe, %c16_i64, 0 {cannir.buffer.generated_sync}
                      %262 = scf.if %213 -> (i64) {
                        %263 = arith.addi %arg33, %c1_i64 overflow<nsw> : i64
                        %264 = arith.remui %263, %c2_i64 : i64
                        scf.yield %264 : i64
                      } else {
                        scf.yield %arg33 : i64
                      }
                      scf.yield %214#0, %175, %239, %262, %197, %214#1, %230 : i64, i64, i64, i64, i64, i64, i64
                    }
                    scf.yield %152#0, %152#1, %152#2, %152#3, %152#4, %152#5, %152#6 : i64, i64, i64, i64, i64, i64, i64
                  }
                  scf.yield %100#0, %100#1, %100#2, %100#3, %100#4, %100#5, %100#6 : i64, i64, i64, i64, i64, i64, i64
                } else {
                  scf.yield %arg14, %arg15, %arg16, %arg17, %arg18, %arg19, %arg20 : i64, i64, i64, i64, i64, i64, i64
                }
                scf.yield %99#0, %99#1, %99#2, %99#3, %99#4, %99#5, %99#6 : i64, i64, i64, i64, i64, i64, i64
              }
            }
          }
        }
        %46 = arith.addi %arg12, %c1_i32 : i32
        %47 = arith.cmpi slt, %46, %15 : i32
        scf.if %47 {
          asccube.sync_all
          asccube.sync_block_wait s, %c1_i64
          asccube.sync_block_arrive fixpipe, %c5_i64 {mode = 0 : i32}
          asccube.sync_block_wait s, %c5_i64 {mode = 0 : i32}
          asccube.sync_block_arrive mte3, %c9_i64
        }
      }
      asccube.sync_intra_wait fixpipe, %c4_i64
      asccube.sync_intra_wait fixpipe, %c20_i64
      asccube.sync_intra_wait fixpipe, %c5_i64
      asccube.sync_intra_wait fixpipe, %c21_i64
    }
    return
  }
}
