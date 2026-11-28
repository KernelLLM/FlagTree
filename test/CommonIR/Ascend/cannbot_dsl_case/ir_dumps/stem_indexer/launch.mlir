module {
  func.func @launch_6894b57926bcc0c5faa1d5cdde54ab5d(%arg0: !cannir.tensor<<gm, bf16>, <<(-1, -1, 2, -1, 1024)>, <(-1, -1, 1024, 2048, 1)>>, <<(-1, -1, 2, -1, 1024)>, <(-1, -1, 1024, 2048, 1)>>, identity>, %arg1: !cannir.tensor<<gm, bf16>, <<(-1, -1, 2, -1, 1024)>, <(-1, -1, 1024, 2048, 1)>>, <<(-1, -1, 2, -1, 1024)>, <(-1, -1, 1024, 2048, 1)>>, identity>, %arg2: !cannir.tensor<<gm, f32>, <<(-1, -1, -1)>, <(-1, -1, 1)>>, <<(-1, -1, -1)>, <(-1, -1, 1)>>, identity>, %arg3: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg4: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg5: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg6: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg7: !cannir.tensor<<gm, i32>, <<(-1, -1, -1, -1)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, -1)>, <(-1, -1, -1, 1)>>, identity>, %arg8: !cannir.tensor<<gm, i32>, <<(-1, -1, -1, 8)>, <(-1, -1, 8, 1)>>, <<(-1, -1, -1, 8)>, <(-1, -1, 8, 1)>>, identity>, %arg9: i64, %arg10: i64, %arg11: f32, %arg12: i64) attributes {cannbotdsl.contract_digest = "6894b57926bcc0c5faa1d5cdde54ab5dbf829bbe11d80c1b1cbe25db3b8b1ba5", cannbotdsl.public_slot_count = 13 : i64, cannbotdsl.slot_dtype_bits = [16, 16, 32, 32, 32, 32, 32, 32, 32, 64, 64, 32, 64], cannbotdsl.slot_dtype_codes = [4, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0], cannbotdsl.slot_dtype_lanes = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], cannbotdsl.slot_flags = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], cannbotdsl.slot_kinds = [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2], cannbotdsl.slot_type_digests = ["546dece128fecf118c2b704ad03de32073081bad30f7f512ba39f8bbf2bc549e", "61d22671f5b92bd0a0fa8efd99962e6c5f8966f18ec28784db1aa6fc92138311", "7a39185bfde37da781ff460a62eb45e142991677a11332a92d7d19cde16757c6", "df1782892fb0fe07e22c2b9245d6bd16a4aa9b9bed10c33a9432045882421aba", "e030c76f69ec395f0341ef8f9ab18ddc826e6df4047f270d22f84aa569a4ad93", "e030c76f69ec395f0341ef8f9ab18ddc826e6df4047f270d22f84aa569a4ad93", "e030c76f69ec395f0341ef8f9ab18ddc826e6df4047f270d22f84aa569a4ad93", "c518e21597b418c4dd9bf193e09f33e05d5ab8a989365bad7fcbabaff97f9b15", "3a3eb9ffdc4cb1a935a25f0f2187a89ef8ca6c87cb61a42d4db01b9c5c498bff", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "4a5363e2ec1eb8cefee037ab0b231653c07f9b44a406ef92ed9ba3e6f5541dbf", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515"], cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.host} {
    cannir.kernel_launch @kernel_StemIndexerKernel___call___0(%arg12) [%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11] {cannbotdsl.profile = {indices = array<i64: 0, 1, 2, 3, 4, 5, 6, 7, 8>, name = "StemIndexerKernel___call__", op_type = "StemIndexerKernel___call__", roles = array<i64: 0, 0, 0, 0, 0, 0, 0, 1, 1>}} : !cannir.tensor<<gm, bf16>, <<(-1, -1, 2, -1, 1024)>, <(-1, -1, 1024, 2048, 1)>>, <<(-1, -1, 2, -1, 1024)>, <(-1, -1, 1024, 2048, 1)>>, identity>, !cannir.tensor<<gm, bf16>, <<(-1, -1, 2, -1, 1024)>, <(-1, -1, 1024, 2048, 1)>>, <<(-1, -1, 2, -1, 1024)>, <(-1, -1, 1024, 2048, 1)>>, identity>, !cannir.tensor<<gm, f32>, <<(-1, -1, -1)>, <(-1, -1, 1)>>, <<(-1, -1, -1)>, <(-1, -1, 1)>>, identity>, !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, !cannir.tensor<<gm, i32>, <<(-1, -1, -1, -1)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, -1)>, <(-1, -1, -1, 1)>>, identity>, !cannir.tensor<<gm, i32>, <<(-1, -1, -1, 8)>, <(-1, -1, 8, 1)>>, <<(-1, -1, -1, 8)>, <(-1, -1, 8, 1)>>, identity>, i64, i64, f32
    return
  }
  func.func @kernel_StemIndexerKernel___call___0_aiv(%arg0: !cannir.tensor<<gm, bf16>, <<(-1, -1, 2, -1, 1024)>, <(-1, -1, 1024, 2048, 1)>>, <<(-1, -1, 2, -1, 1024)>, <(-1, -1, 1024, 2048, 1)>>, identity>, %arg1: !cannir.tensor<<gm, bf16>, <<(-1, -1, 2, -1, 1024)>, <(-1, -1, 1024, 2048, 1)>>, <<(-1, -1, 2, -1, 1024)>, <(-1, -1, 1024, 2048, 1)>>, identity>, %arg2: !cannir.tensor<<gm, f32>, <<(-1, -1, -1)>, <(-1, -1, 1)>>, <<(-1, -1, -1)>, <(-1, -1, 1)>>, identity>, %arg3: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg4: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg5: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg6: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg7: !cannir.tensor<<gm, i32>, <<(-1, -1, -1, -1)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, -1)>, <(-1, -1, -1, 1)>>, identity>, %arg8: !cannir.tensor<<gm, i32>, <<(-1, -1, -1, 8)>, <(-1, -1, 8, 1)>>, <<(-1, -1, -1, 8)>, <(-1, -1, 8, 1)>>, identity>, %arg9: i64, %arg10: i64, %arg11: f32) attributes {cannir.buffer.auto_sync, cannir.buffer.sync_analyzed, cannir.buffer.sync_verified, cannir.buffer.tensor_state, cannir.core = "aiv", cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.device, cannir.explicit_ring_buffer_cursors, cannir.kernel_group = "kernel_StemIndexerKernel___call___0", cannir.resource.ids_prepared} {
    %c1280_i32 = arith.constant 1280 : i32
    %c2_i64 = arith.constant 2 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %c3_i64 = arith.constant 3 : i64
    %c4_i64 = arith.constant 4 : i64
    %c8_i64 = arith.constant 8 : i64
    %c10_i64 = arith.constant 10 : i64
    %c11_i64 = arith.constant 11 : i64
    %c12_i64 = arith.constant 12 : i64
    %c255_i64 = arith.constant 255 : i64
    %c63_i64 = arith.constant 63 : i64
    %c127_i64 = arith.constant 127 : i64
    %c4 = arith.constant 4 : index
    %c1024_i64 = arith.constant 1024 : i64
    %c192_i32 = arith.constant 192 : i32
    %c8_i32 = arith.constant 8 : i32
    %c0 = arith.constant 0 : index
    %c128_i32 = arith.constant 128 : i32
    %c256_i32 = arith.constant 256 : i32
    %c64_i32 = arith.constant 64 : i32
    %c5_i32 = arith.constant 5 : i32
    %c3_i32 = arith.constant 3 : i32
    %c2_i32 = arith.constant 2 : i32
    %c16_i32 = arith.constant 16 : i32
    %c4_i32 = arith.constant 4 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c32_i64 = arith.constant 32 : i64
    %c256_i64 = arith.constant 256 : i64
    %cst = arith.constant 1.000000e-01 : f32
    %cst_0 = arith.constant 3.000000e+01 : f32
    %cst_1 = arith.constant 2.000000e-01 : f32
    %c160_i64 = arith.constant 160 : i64
    %c56_i64 = arith.constant 56 : i64
    %c64_i64 = arith.constant 64 : i64
    %c128_i64 = arith.constant 128 : i64
    %c16_i64 = arith.constant 16 : i64
    %c576_i64 = arith.constant 576 : i64
    %alloc = memref.alloc() {cannir.arg_index = 2 : i64} : memref<1xf32>
    %alloc_2 = memref.alloc() {cannir.arg_index = 3 : i64} : memref<1xi32>
    %alloc_3 = memref.alloc() {cannir.arg_index = 4 : i64} : memref<1xi32>
    %alloc_4 = memref.alloc() {cannir.arg_index = 5 : i64} : memref<1xi32>
    %alloc_5 = memref.alloc() {cannir.arg_index = 6 : i64} : memref<1xi32>
    %alloc_6 = memref.alloc() {cannir.arg_index = 7 : i64} : memref<1xi32>
    %alloc_7 = memref.alloc() {cannir.arg_index = 8 : i64} : memref<1xi32>
    %alloc_8 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<2048xf32, 1>
    %alloc_9 = memref.alloc() {cannir.byte_offset = 8192 : i64} : memref<2048xf32, 1>
    %alloc_10 = memref.alloc() {cannir.byte_offset = 16384 : i64} : memref<2048xf32, 1>
    %alloc_11 = memref.alloc() {cannir.byte_offset = 24576 : i64} : memref<8192xf32, 1>
    %alloc_12 = memref.alloc() {cannir.byte_offset = 57344 : i64} : memref<256xf32, 1>
    %alloc_13 = memref.alloc() {cannir.byte_offset = 58368 : i64} : memref<512xi32, 1>
    %alloc_14 = memref.alloc() {cannir.byte_offset = 60416 : i64} : memref<256xi32, 1>
    %alloc_15 = memref.alloc() {cannir.byte_offset = 61440 : i64} : memref<10240xi32, 1>
    %alloc_16 = memref.alloc() {cannir.byte_offset = 102400 : i64} : memref<32xi32, 1>
    %alloc_17 = memref.alloc() {cannir.byte_offset = 102528 : i64} : memref<8192xui32, 1>
    %alloc_18 = memref.alloc() {cannir.byte_offset = 135296 : i64} : memref<8192xui32, 1>
    %alloc_19 = memref.alloc() {cannir.byte_offset = 168064 : i64} : memref<1024xui16, 1>
    %alloc_20 = memref.alloc() {cannir.byte_offset = 170112 : i64} : memref<256xui16, 1>
    %alloc_21 = memref.alloc() {cannir.byte_offset = 170624 : i64} : memref<256xui16, 1>
    %alloc_22 = memref.alloc() {cannir.byte_offset = 171136 : i64} : memref<256xui16, 1>
    %alloc_23 = memref.alloc() {cannir.byte_offset = 171648 : i64} : memref<256xui16, 1>
    %alloc_24 = memref.alloc() {cannir.byte_offset = 172160 : i64} : memref<528xi32, 1>
    %alloc_25 = memref.alloc() {cannir.byte_offset = 174272 : i64} : memref<528xi32, 1>
    %alloc_26 = memref.alloc() {cannir.byte_offset = 176384 : i64} : memref<528xi32, 1>
    %alloc_27 = memref.alloc() {cannir.byte_offset = 178496 : i64} : memref<528xi32, 1>
    %alloc_28 = memref.alloc() {cannir.byte_offset = 180608 : i64} : memref<1024xui32, 1>
    %alloc_29 = memref.alloc() {cannir.byte_offset = 184704 : i64} : memref<1024xui32, 1>
    ascvec.sync_intra_arrive v, %c2_i64
    ascvec.sync_intra_arrive v, %c1_i64
    ascvec.sync_intra_arrive v, %c0_i64
    %0 = ascscalar.get_block_idx
    %1 = ascscalar.get_subblock_id
    %2 = ascscalar.bitcast_memref %alloc_12 byte_offset = %c0_i64 : memref<256xf32, 1> -> memref<?xf32, 1>
    %3 = memref.load %alloc_2[%c0] : memref<1xi32>
    %4 = arith.extsi %3 : i32 to i64
    %5:2 = scf.for %arg12 = %c0_i64 to %4 step %c1_i64 iter_args(%arg13 = %c0_i64, %arg14 = %c0_i64) -> (i64, i64)  : i64 {
      %6 = arith.muli %arg12, %c576_i64 : i64
      %7 = arith.addi %6, %c16_i64 : i64
      %8 = arith.muli %0, %c16_i64 : i64
      %9 = arith.addi %7, %8 : i64
      %10 = arith.index_cast %9 : i64 to index
      %11 = memref.load %alloc_2[%10] : memref<1xi32>
      %12 = arith.extsi %11 : i32 to i64
      %13 = arith.addi %9, %c1_i64 : i64
      %14 = arith.index_cast %13 : i64 to index
      %15 = memref.load %alloc_2[%14] : memref<1xi32>
      %16 = arith.extsi %15 : i32 to i64
      %17 = arith.addi %9, %c3_i64 : i64
      %18 = arith.index_cast %17 : i64 to index
      %19 = memref.load %alloc_2[%18] : memref<1xi32>
      %20 = arith.extsi %19 : i32 to i64
      %21 = arith.addi %9, %c4_i64 : i64
      %22 = arith.index_cast %21 : i64 to index
      %23 = memref.load %alloc_2[%22] : memref<1xi32>
      %24 = arith.extsi %23 : i32 to i64
      %25 = arith.addi %20, %c1_i64 : i64
      %26 = cannir.extract_layout %arg4 : !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>
      %27 = cannir.extract_shape %26 : <<(-1)>, <(1)>> -> <(-1)>
      %28 = cannir.extract_dim %27[0] : !cannir.shape<(-1)>
      %29 = arith.muli %28, %arg9 : i64
      %30 = arith.minsi %25, %29 : i64
      %31:2 = scf.for %arg15 = %12 to %30 step %c1_i64 iter_args(%arg16 = %arg13, %arg17 = %arg14) -> (i64, i64)  : i64 {
        %32 = arith.remsi %arg15, %arg9 : i64
        %33 = arith.cmpi ne, %32, %c0_i64 : i64
        %34 = arith.cmpi slt, %32, %c0_i64 : i64
        %35 = arith.cmpi slt, %arg9, %c0_i64 : i64
        %36 = arith.xori %34, %35 : i1
        %37 = arith.andi %33, %36 : i1
        %38 = arith.divsi %arg15, %arg9 : i64
        %39 = arith.extui %37 : i1 to i64
        %40 = arith.subi %38, %39 : i64
        %41 = arith.muli %40, %arg9 : i64
        %42 = arith.subi %arg15, %41 : i64
        %43 = arith.index_cast %40 : i64 to index
        %44 = memref.load %alloc_3[%43] : memref<1xi32>
        %45 = arith.extsi %44 : i32 to i64
        %46 = memref.load %alloc_4[%43] : memref<1xi32>
        %47 = arith.extsi %46 : i32 to i64
        %48 = arith.addi %45, %c127_i64 : i64
        %49 = arith.remsi %48, %c128_i64 : i64
        %50 = arith.cmpi ne, %49, %c0_i64 : i64
        %51 = arith.cmpi slt, %49, %c0_i64 : i64
        %52 = arith.andi %50, %51 : i1
        %53 = arith.divsi %48, %c128_i64 : i64
        %54 = arith.extui %52 : i1 to i64
        %55 = arith.subi %53, %54 : i64
        %56 = arith.addi %47, %c127_i64 : i64
        %57 = arith.remsi %56, %c128_i64 : i64
        %58 = arith.cmpi ne, %57, %c0_i64 : i64
        %59 = arith.cmpi slt, %57, %c0_i64 : i64
        %60 = arith.andi %58, %59 : i1
        %61 = arith.divsi %56, %c128_i64 : i64
        %62 = arith.extui %60 : i1 to i64
        %63 = arith.subi %61, %62 : i64
        %64 = arith.cmpi eq, %arg15, %12 : i64
        %65 = arith.select %64, %16, %c0_i64 : i64
        %66 = arith.muli %arg10, %55 : i64
        %67 = arith.addi %66, %c63_i64 : i64
        %68 = arith.remsi %67, %c64_i64 : i64
        %69 = arith.cmpi ne, %68, %c0_i64 : i64
        %70 = arith.cmpi slt, %68, %c0_i64 : i64
        %71 = arith.andi %69, %70 : i1
        %72 = arith.divsi %67, %c64_i64 : i64
        %73 = arith.extui %71 : i1 to i64
        %74 = arith.subi %72, %73 : i64
        %75 = arith.cmpi eq, %arg15, %20 : i64
        %76 = arith.select %75, %24, %74 : i64
        %77:2 = scf.for %arg18 = %65 to %76 step %c1_i64 iter_args(%arg19 = %arg16, %arg20 = %arg17) -> (i64, i64)  : i64 {
          %78 = arith.muli %arg18, %c64_i64 : i64
          %79 = arith.addi %78, %c64_i64 : i64
          %80 = arith.minsi %79, %66 : i64
          %81:2 = scf.for %arg21 = %c0_i64 to %arg10 step %c1_i64 iter_args(%arg22 = %arg19, %arg23 = %arg20) -> (i64, i64)  : i64 {
            %82 = arith.muli %arg21, %55 : i64
            %83 = arith.addi %82, %55 : i64
            %84 = arith.maxsi %78, %82 : i64
            %85 = arith.minsi %80, %83 : i64
            %86 = arith.subi %85, %84 : i64
            %87 = arith.maxsi %86, %c0_i64 : i64
            %88 = arith.cmpi sgt, %87, %c0_i64 : i64
            %89:2 = scf.if %88 -> (i64, i64) {
              %90 = arith.muli %42, %arg10 : i64
              %91 = arith.addi %90, %arg21 : i64
              %92 = arith.subi %84, %82 : i64
              %93 = arith.addi %92, %87 : i64
              %94 = arith.subi %93, %c1_i64 : i64
              %95 = memref.load %alloc_3[%43] : memref<1xi32>
              %96 = arith.extsi %95 : i32 to i64
              %97 = memref.load %alloc_4[%43] : memref<1xi32>
              %98 = arith.extsi %97 : i32 to i64
              %99 = memref.load %alloc_5[%43] : memref<1xi32>
              %100 = arith.extsi %99 : i32 to i64
              %101 = arith.addi %100, %c127_i64 : i64
              %102 = arith.remsi %101, %c128_i64 : i64
              %103 = arith.cmpi ne, %102, %c0_i64 : i64
              %104 = arith.cmpi slt, %102, %c0_i64 : i64
              %105 = arith.andi %103, %104 : i1
              %106 = arith.divsi %101, %c128_i64 : i64
              %107 = arith.extui %105 : i1 to i64
              %108 = arith.subi %106, %107 : i64
              %109 = arith.cmpi eq, %96, %c1_i64 : i64
              %110 = arith.cmpi sge, %100, %98 : i64
              %111 = arith.extui %110 : i1 to i64
              %112 = arith.select %109, %111, %c0_i64 : i64
              %113 = arith.cmpi eq, %112, %c0_i64 : i64
              %114 = arith.subi %63, %55 : i64
              %115 = arith.addi %114, %94 : i64
              %116 = arith.addi %115, %c1_i64 : i64
              %117 = arith.minsi %116, %63 : i64
              %118 = arith.maxsi %117, %c0_i64 : i64
              %119 = arith.select %113, %118, %63 : i64
              %120 = arith.cmpi sge, %108, %c56_i64 : i64
              %121 = scf.if %120 -> (i64) {
                %157 = arith.cmpi slt, %108, %c160_i64 : i64
                %158 = scf.if %157 -> (i64) {
                  %159 = arith.sitofp %108 : i64 to f32
                  %160 = arith.mulf %159, %cst_1 : f32
                  %161 = arith.addf %160, %cst_0 : f32
                  %162 = arith.fptosi %161 : f32 to i64
                  scf.yield %162 : i64
                } else {
                  %159 = arith.sitofp %108 : i64 to f32
                  %160 = arith.mulf %159, %cst : f32
                  %161 = arith.addf %160, %cst_0 : f32
                  %162 = arith.fptosi %161 : f32 to i64
                  scf.yield %162 : i64
                }
                scf.yield %158 : i64
              } else {
                scf.yield %108 : i64
              }
              %122 = arith.minsi %121, %c256_i64 : i64
              %123 = arith.minsi %119, %c4_i64 : i64
              %124 = arith.subi %119, %c4_i64 : i64
              %125 = arith.maxsi %124, %123 : i64
              %126 = arith.addi %123, %119 : i64
              %127 = arith.subi %126, %125 : i64
              %128 = arith.subi %119, %127 : i64
              %129 = arith.maxsi %128, %c0_i64 : i64
              %130 = arith.maxsi %124, %c4_i64 : i64
              %131 = arith.addi %130, %c63_i64 : i64
              %132 = arith.remsi %131, %c64_i64 : i64
              %133 = arith.cmpi ne, %132, %c0_i64 : i64
              %134 = arith.cmpi slt, %132, %c0_i64 : i64
              %135 = arith.andi %133, %134 : i1
              %136 = arith.divsi %131, %c64_i64 : i64
              %137 = arith.extui %135 : i1 to i64
              %138 = arith.subi %136, %137 : i64
              %139 = arith.muli %138, %c64_i64 : i64
              %140 = arith.minsi %63, %139 : i64
              %141 = arith.muli %1, %c32_i64 : i64
              %142 = arith.subi %87, %141 : i64
              %143 = arith.minsi %142, %c32_i64 : i64
              %144 = arith.maxsi %143, %c0_i64 : i64
              %145 = arith.trunci %144 : i64 to i32
              %146 = arith.addi %92, %141 : i64
              %147 = arith.addi %114, %146 : i64
              %148 = arith.addi %147, %c1_i64 : i64
              %149 = arith.select %113, %148, %63 : i64
              %150 = arith.trunci %149 : i64 to i32
              %151 = arith.extui %113 : i1 to i32
              %152 = arith.trunci %63 : i64 to i32
              %153 = arith.trunci %122 : i64 to i32
              ascvec.vf(%150, %152, %153, %145, %151, %alloc_13 : i32, i32, i32, i32, i32, memref<512xi32, 1>) {
              ^bb0(%arg24: i32, %arg25: i32, %arg26: i32, %arg27: i32, %arg28: i32, %arg29: memref<512xi32, 1>):
                %c7_i32 = arith.constant 7 : i32
                %c6_i32 = arith.constant 6 : i32
                %c5_i32_30 = arith.constant 5 : i32
                %c255_i32 = arith.constant 255 : i32
                %c3_i32_31 = arith.constant 3 : i32
                %c2_i32_32 = arith.constant 2 : i32
                %c0_33 = arith.constant 0 : index
                %c16_i32_34 = arith.constant 16 : i32
                %c-4_i32 = arith.constant -4 : i32
                %c4_i32_35 = arith.constant 4 : i32
                %c1_i32_36 = arith.constant 1 : i32
                %c0_i32_37 = arith.constant 0 : i32
                %c32_i32_38 = arith.constant 32 : i32
                %c0_39 = arith.constant 0 : index
                %c1 = arith.constant 1 : index
                scf.for %arg30 = %c0_39 to %c1 step %c1 {
                  %mask, %remaining_out = ascvec.update_mask %c32_i32_38 : !ascvec.mask
                  %157 = ascvec.arange %c0_i32_37 : i32 -> <i32>
                  %158 = ascvec.duplicate_scalar %arg24, %mask : i32 -> <i32>
                  %159 = ascvec.duplicate_scalar %arg25, %mask : i32 -> <i32>
                  %160 = ascvec.duplicate_scalar %arg26, %mask : i32 -> <i32>
                  %161 = ascvec.duplicate_scalar %arg27, %mask : i32 -> <i32>
                  %162 = ascvec.duplicate_scalar %c0_i32_37, %mask : i32 -> <i32>
                  %163 = ascvec.duplicate_scalar %c1_i32_36, %mask : i32 -> <i32>
                  %164 = ascvec.mul_scalar %157, %arg28, %mask : <i32>, i32
                  %165 = ascvec.add %158, %164, %mask : <i32>
                  %166 = ascvec.max_scalar %165, %c0_i32_37, %mask : <i32>, i32
                  %167 = ascvec.min %166, %159, %mask : <i32>
                  %168 = ascvec.cmp_lt %157, %161, %mask : <i32>, <i32> -> !ascvec.mask
                  %169 = ascvec.select %167, %162, %168 : <i32>
                  %170 = ascvec.min_scalar %169, %c4_i32_35, %mask : <i32>, i32
                  %171 = ascvec.add_scalar %169, %c-4_i32, %mask : <i32>, i32
                  %172 = ascvec.max %171, %170, %mask : <i32>
                  %173 = ascvec.sub %172, %170, %mask : <i32>
                  %174 = ascvec.min %160, %173, %mask : <i32>
                  %175 = ascvec.sub %169, %173, %mask : <i32>
                  %176 = ascvec.add %175, %174, %mask : <i32>
                  %177 = ascvec.cmp_eq %174, %173, %mask : <i32>, <i32> -> !ascvec.mask
                  %178 = ascvec.select %163, %162, %177 : <i32>
                  %179 = ascvec.arange %c0_i32_37 : i32 -> <ui32>
                  %180 = ascvec.mul_scalar %179, %c16_i32_34, %mask : <ui32>, i32
                  ascvec.scatter %arg29[%c0_33], %169, %180, %mask : memref<512xi32, 1>, <i32>, <ui32>, !ascvec.mask
                  %181 = ascvec.add_scalar %180, %c1_i32_36, %mask : <ui32>, i32
                  ascvec.scatter %arg29[%c0_33], %174, %181, %mask : memref<512xi32, 1>, <i32>, <ui32>, !ascvec.mask
                  %182 = ascvec.add_scalar %180, %c2_i32_32, %mask : <ui32>, i32
                  ascvec.scatter %arg29[%c0_33], %176, %182, %mask : memref<512xi32, 1>, <i32>, <ui32>, !ascvec.mask
                  %183 = ascvec.add_scalar %180, %c3_i32_31, %mask : <ui32>, i32
                  ascvec.scatter %arg29[%c0_33], %178, %183, %mask : memref<512xi32, 1>, <i32>, <ui32>, !ascvec.mask
                  %184 = ascvec.duplicate_scalar %c255_i32, %mask : i32 -> <i32>
                  %185 = ascvec.shr %169, 0, %mask : <i32> -> <i32>
                  %186 = ascvec.and %185, %184, %mask : <i32>, <i32> -> <i32>
                  %187 = ascvec.add_scalar %180, %c4_i32_35, %mask : <ui32>, i32
                  ascvec.scatter %arg29[%c0_33], %186, %187, %mask : memref<512xi32, 1>, <i32>, <ui32>, !ascvec.mask
                  %188 = ascvec.shr %169, 8, %mask : <i32> -> <i32>
                  %189 = ascvec.and %188, %184, %mask : <i32>, <i32> -> <i32>
                  %190 = ascvec.add_scalar %180, %c5_i32_30, %mask : <ui32>, i32
                  ascvec.scatter %arg29[%c0_33], %189, %190, %mask : memref<512xi32, 1>, <i32>, <ui32>, !ascvec.mask
                  %191 = ascvec.shr %169, 16, %mask : <i32> -> <i32>
                  %192 = ascvec.and %191, %184, %mask : <i32>, <i32> -> <i32>
                  %193 = ascvec.add_scalar %180, %c6_i32, %mask : <ui32>, i32
                  ascvec.scatter %arg29[%c0_33], %192, %193, %mask : memref<512xi32, 1>, <i32>, <ui32>, !ascvec.mask
                  %194 = ascvec.shr %169, 24, %mask : <i32> -> <i32>
                  %195 = ascvec.and %194, %184, %mask : <i32>, <i32> -> <i32>
                  %196 = ascvec.add_scalar %180, %c7_i32, %mask : <ui32>, i32
                  ascvec.scatter %arg29[%c0_33], %195, %196, %mask : memref<512xi32, 1>, <i32>, <ui32>, !ascvec.mask
                }
              } {cannir.vf_mode = "simd"}
              %154 = arith.cmpi sgt, %129, %122 : i64
              %155:2 = scf.if %154 -> (i64, i64) {
                %157 = arith.addi %140, %c255_i64 : i64
                %158 = arith.remsi %157, %c256_i64 : i64
                %159 = arith.cmpi ne, %158, %c0_i64 : i64
                %160 = arith.cmpi slt, %158, %c0_i64 : i64
                %161 = arith.andi %159, %160 : i1
                %162 = arith.divsi %157, %c256_i64 : i64
                %163 = arith.extui %161 : i1 to i64
                %164 = arith.subi %162, %163 : i64
                %165:2 = scf.for %arg24 = %c0_i64 to %164 step %c1_i64 iter_args(%arg25 = %arg22, %arg26 = %arg23) -> (i64, i64)  : i64 {
                  %166 = arith.muli %arg24, %c256_i64 : i64
                  %167 = cannir.extract_layout %arg2 : !cannir.tensor<<gm, f32>, <<(-1, -1, -1)>, <(-1, -1, 1)>>, <<(-1, -1, -1)>, <(-1, -1, 1)>>, identity>
                  %168 = cannir.extract_shape %167 : <<(-1, -1, -1)>, <(-1, -1, 1)>> -> <(-1, -1, -1)>
                  %169 = cannir.extract_dim %168[2] : !cannir.shape<(-1, -1, -1)>
                  %170 = arith.subi %169, %166 : i64
                  %171 = arith.divsi %170, %c256_i64 : i64
                  %172 = arith.minsi %171, %c0_i64 : i64
                  %173 = arith.muli %172, %c256_i64 : i64
                  %174 = arith.subi %170, %173 : i64
                  %175 = arith.minsi %174, %c256_i64 : i64
                  %176 = arith.cmpi sge, %171, %c0_i64 : i64
                  %177 = arith.select %176, %175, %c0_i64 : i64
                  %178 = cannir.extract_physical_layout %arg2 : !cannir.tensor<<gm, f32>, <<(-1, -1, -1)>, <(-1, -1, 1)>>, <<(-1, -1, -1)>, <(-1, -1, 1)>>, identity>
                  %179 = cannir.extract_stride %178 : <<(-1, -1, -1)>, <(-1, -1, 1)>> -> <(-1, -1, 1)>
                  %180 = cannir.extract_dim %179[0] : !cannir.stride<(-1, -1, 1)>
                  %181 = arith.muli %40, %180 : i64
                  %182 = arith.muli %181, %c4_i64 : i64
                  %183 = cannir.extract_dim %179[1] : !cannir.stride<(-1, -1, 1)>
                  %184 = arith.muli %42, %183 : i64
                  %185 = arith.muli %184, %c4_i64 : i64
                  %186 = arith.addi %182, %185 : i64
                  %187 = arith.muli %arg24, %c1024_i64 : i64
                  %188 = arith.addi %186, %187 : i64
                  %189 = arith.muli %177, %c4_i64 : i64
                  %190 = arith.cmpi eq, %189, %c0_i64 : i64
                  %191 = arith.cmpi ne, %189, %c0_i64 : i64
                  %192 = arith.extui %191 : i1 to i64
                  %193 = arith.select %190, %c4_i64, %189 : i64
                  ascvec.get_buf mte2, %c0_i64, 0 {cannir.buffer.generated_sync}
                  "ascvec.copy_gm2ub_plan"(%2, %alloc, %c0_i32, %c0_i32, %c0_i64, %188, %192, %193, %c0_i64, %193, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 57344 : i64, ub_capacity = 1024 : i64}> : (memref<?xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                  ascvec.release_buf mte2, %c0_i64, 0 {cannir.buffer.generated_sync}
                  %194:2 = scf.for %arg27 = %c0_i64 to %c4_i64 step %c1_i64 iter_args(%arg28 = %arg25, %arg29 = %arg26) -> (i64, i64)  : i64 {
                    %195 = arith.muli %arg27, %c64_i64 overflow<nsw> : i64
                    %196 = arith.addi %166, %195 : i64
                    %197 = arith.cmpi slt, %196, %140 : i64
                    %198:2 = scf.if %197 -> (i64, i64) {
                      %199 = arith.addi %arg28, %c1_i64 : i64
                      %200 = arith.remui %199, %c2_i64 : i64
                      %201 = arith.cmpi eq, %arg29, %c1_i64 : i64
                      %202 = ascscalar.memref_address %alloc_9 : memref<2048xf32, 1>
                      %203 = ascscalar.memref_address %alloc_10 : memref<2048xf32, 1>
                      %204 = arith.select %201, %202, %203 : i64
                      %205 = arith.cmpi eq, %arg29, %c0_i64 : i64
                      %206 = ascscalar.memref_address %alloc_8 : memref<2048xf32, 1>
                      %207 = arith.select %205, %206, %204 : i64
                      %208 = arith.select %201, %c11_i64, %c12_i64 : i64
                      %209 = arith.select %205, %c10_i64, %208 : i64
                      %210 = arith.addi %arg29, %c1_i64 : i64
                      %211 = arith.remui %210, %c3_i64 : i64
                      %212 = arith.cmpi eq, %209, %c11_i64 : i64
                      %213 = arith.extui %212 : i1 to i64
                      %214 = arith.cmpi eq, %209, %c12_i64 : i64
                      %215 = arith.select %214, %c2_i64, %213 : i64
                      %216 = ascscalar.bind_local_ptr %207 : memref<2048xf32, 1>
                      ascvec.sync_intra_wait v, %215 {cannir.sync.banks = 3 : i32}
                      ascvec.get_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
                      ascvec.vf(%195, %166, %216, %alloc_12, %alloc_13, %alloc_11 : i64, i64, memref<2048xf32, 1>, memref<256xf32, 1>, memref<512xi32, 1>, memref<8192xf32, 1>) {
                      ^bb0(%arg30: i64, %arg31: i64, %arg32: memref<2048xf32, 1>, %arg33: memref<256xf32, 1>, %arg34: memref<512xi32, 1>, %arg35: memref<8192xf32, 1>):
                        %c7_i32 = arith.constant 7 : i32
                        %c6_i32 = arith.constant 6 : i32
                        %c5_i32_30 = arith.constant 5 : i32
                        %c32_i32_31 = arith.constant 32 : i32
                        %c256_i32_32 = arith.constant 256 : i32
                        %c-4_i32 = arith.constant -4 : i32
                        %c1_i32_33 = arith.constant 1 : i32
                        %c4_i32_34 = arith.constant 4 : i32
                        %c16_i32_35 = arith.constant 16 : i32
                        %c255_i32 = arith.constant 255 : i32
                        %c2143289344_i32 = arith.constant 2143289344 : i32
                        %c0_i32_36 = arith.constant 0 : i32
                        %c-2147483648_i32 = arith.constant -2147483648 : i32
                        %cst_37 = arith.constant 1.562500e-02 : f32
                        %c64_i32_38 = arith.constant 64 : i32
                        %mask, %remaining_out = ascvec.update_mask %c64_i32_38 : !ascvec.mask
                        scf.for %arg36 = %c0_i32_36 to %c32_i32_31 step %c1_i32_33  : i32 {
                          %217 = arith.trunci %arg30 : i64 to i32
                          %218 = arith.extsi %217 : i32 to i64
                          %219 = arith.addi %arg31, %218 : i64
                          %220 = arith.muli %arg36, %c64_i32_38 overflow<nsw> : i32
                          %221 = arith.index_cast %220 : i32 to index
                          %222 = ascvec.loadalign %arg32[%221] : memref<2048xf32, 1>, index -> <f32>
                          %223 = ascvec.mul_scalar %222, %cst_37, %mask : <f32>, f32
                          %224 = arith.index_cast %217 : i32 to index
                          %225 = ascvec.loadalign %arg33[%224] : memref<256xf32, 1>, index -> <f32>
                          %226 = ascvec.add %223, %225, %mask : <f32>
                          %227 = ascvec.reinterpret %226 : <f32> -> <ui32>
                          %228 = ascvec.duplicate_scalar %c-2147483648_i32, %mask : i32 -> <ui32>
                          %229 = ascvec.and %227, %228, %mask : <ui32>, <ui32> -> <ui32>
                          %230 = ascvec.cmp_ne_scalar %229, %c0_i32_36, %mask : <ui32>, i32 -> !ascvec.mask
                          %231 = ascvec.xor %227, %228, %mask : <ui32>, <ui32> -> <ui32>
                          %232 = ascvec.not %227, %mask : <ui32>
                          %233 = ascvec.select %232, %231, %230 : <ui32>
                          %234 = ascvec.duplicate_scalar %c0_i32_36, %mask : i32 -> <ui32>
                          %235 = ascvec.cmp_eq_scalar %227, %c2143289344_i32, %mask : <ui32>, i32 -> !ascvec.mask
                          %236 = ascvec.select %234, %233, %235 : <ui32>
                          %237 = arith.trunci %219 : i64 to i32
                          %238 = ascvec.arange %237 : i32 -> <i32>
                          %239 = ascvec.duplicate_scalar %c255_i32, %mask : i32 -> <ui32>
                          %240 = arith.muli %arg36, %c16_i32_35 overflow<nsw> : i32
                          %241 = arith.addi %240, %c4_i32_34 overflow<nsw> : i32
                          %242 = arith.index_cast %241 : i32 to index
                          %243 = ascvec.loadalign_brc %arg34[%242] {brc_width = "b8"} : memref<512xi32, 1>, index -> <ui32>
                          %244 = ascvec.and %243, %239, %mask : <ui32>, <ui32> -> <ui32>
                          %245 = arith.addi %240, %c5_i32_30 overflow<nsw> : i32
                          %246 = arith.index_cast %245 : i32 to index
                          %247 = ascvec.loadalign_brc %arg34[%246] {brc_width = "b8"} : memref<512xi32, 1>, index -> <ui32>
                          %248 = ascvec.and %247, %239, %mask : <ui32>, <ui32> -> <ui32>
                          %249 = ascvec.shl %248, 8, %mask : <ui32> -> <ui32>
                          %250 = ascvec.add %244, %249, %mask : <ui32>
                          %251 = arith.addi %240, %c6_i32 overflow<nsw> : i32
                          %252 = arith.index_cast %251 : i32 to index
                          %253 = ascvec.loadalign_brc %arg34[%252] {brc_width = "b8"} : memref<512xi32, 1>, index -> <ui32>
                          %254 = ascvec.and %253, %239, %mask : <ui32>, <ui32> -> <ui32>
                          %255 = ascvec.shl %254, 16, %mask : <ui32> -> <ui32>
                          %256 = ascvec.add %250, %255, %mask : <ui32>
                          %257 = arith.addi %240, %c7_i32 overflow<nsw> : i32
                          %258 = arith.index_cast %257 : i32 to index
                          %259 = ascvec.loadalign_brc %arg34[%258] {brc_width = "b8"} : memref<512xi32, 1>, index -> <ui32>
                          %260 = ascvec.and %259, %239, %mask : <ui32>, <ui32> -> <ui32>
                          %261 = ascvec.shl %260, 24, %mask : <ui32> -> <ui32>
                          %262 = ascvec.add %256, %261, %mask : <ui32>
                          %263 = ascvec.reinterpret %262 : <ui32> -> <i32>
                          %264 = ascvec.min_scalar %263, %c4_i32_34, %mask : <i32>, i32
                          %265 = ascvec.add_scalar %263, %c-4_i32, %mask : <i32>, i32
                          %266 = ascvec.max_scalar %265, %c0_i32_36, %mask : <i32>, i32
                          %267 = ascvec.max %266, %264, %mask : <i32>
                          %268 = ascvec.cmp_lt %238, %263, %mask : <i32>, <i32> -> !ascvec.mask
                          %269 = ascvec.cmp_ge %238, %264, %268 : <i32>, <i32> -> !ascvec.mask
                          %270 = ascvec.cmp_lt %238, %267, %269 : <i32>, <i32> -> !ascvec.mask
                          %271 = ascvec.select %236, %234, %270 : <ui32>
                          %272 = arith.muli %arg36, %c256_i32_32 overflow<nsw> : i32
                          %273 = arith.addi %272, %217 : i32
                          %274 = ascvec.reinterpret %271 : <ui32> -> <f32>
                          %275 = arith.index_cast %273 : i32 to index
                          ascvec.storealign %arg35[%275], %274, %mask : memref<8192xf32, 1>, index, <f32>
                        }
                      } {cannir.vf_mode = "simd"}
                      ascvec.release_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
                      ascvec.sync_intra_arrive v, %215 {cannir.sync.banks = 3 : i32}
                      scf.yield %200, %211 : i64, i64
                    } else {
                      ascvec.vf(%195, %alloc_11 : i64, memref<8192xf32, 1>) {
                      ^bb0(%arg30: i64, %arg31: memref<8192xf32, 1>):
                        %c1_i32_30 = arith.constant 1 : i32
                        %c32_i32_31 = arith.constant 32 : i32
                        %c0_i32_32 = arith.constant 0 : i32
                        %c256_i32_33 = arith.constant 256 : i32
                        %cst_34 = arith.constant 0.000000e+00 : f32
                        %c64_i32_35 = arith.constant 64 : i32
                        %mask, %remaining_out = ascvec.update_mask %c64_i32_35 : !ascvec.mask
                        %199 = ascvec.duplicate_scalar %cst_34, %mask : f32 -> <f32>
                        scf.for %arg32 = %c0_i32_32 to %c32_i32_31 step %c1_i32_30  : i32 {
                          %200 = arith.muli %arg32, %c256_i32_33 overflow<nsw> : i32
                          %201 = arith.trunci %arg30 : i64 to i32
                          %202 = arith.addi %200, %201 : i32
                          %203 = arith.index_cast %202 : i32 to index
                          ascvec.storealign %arg31[%203], %199, %mask : memref<8192xf32, 1>, index, <f32>
                        }
                      } {cannir.vf_mode = "simd"}
                      scf.yield %arg28, %arg29 : i64, i64
                    }
                    scf.yield %198#0, %198#1 : i64, i64
                  }
                  ascvec.vf {
                    ascvec.mem_bar vst_vld
                  } {cannir.vf_mode = "simd"}
                  scf.for %arg27 = %c0_i64 to %c32_i64 step %c4_i64  : i64 {
                    %195 = arith.cmpi eq, %arg24, %c0_i64 : i64
                    scf.if %195 {
                      %196 = arith.trunci %arg27 : i64 to i32
                      %197 = arith.muli %196, %c16_i32 overflow<nsw> : i32
                      %198 = arith.addi %197, %c1_i32 overflow<nsw> : i32
                      %199 = arith.index_cast %198 : i32 to index
                      %200 = arith.addi %196, %c1_i32 overflow<nsw> : i32
                      %201 = arith.muli %200, %c16_i32 overflow<nsw> : i32
                      %202 = arith.addi %201, %c1_i32 overflow<nsw> : i32
                      %203 = arith.index_cast %202 : i32 to index
                      %204 = arith.addi %196, %c2_i32 overflow<nsw> : i32
                      %205 = arith.muli %204, %c16_i32 overflow<nsw> : i32
                      %206 = arith.addi %205, %c1_i32 overflow<nsw> : i32
                      %207 = arith.index_cast %206 : i32 to index
                      %208 = arith.addi %196, %c3_i32 overflow<nsw> : i32
                      %209 = arith.muli %208, %c16_i32 overflow<nsw> : i32
                      %210 = arith.addi %209, %c1_i32 overflow<nsw> : i32
                      %211 = arith.index_cast %210 : i32 to index
                      %212 = arith.muli %196, %c8_i32 overflow<nsw> : i32
                      %213 = arith.addi %212, %c1_i32 overflow<nsw> : i32
                      %214 = arith.index_cast %213 : i32 to index
                      %215 = arith.addi %212, %c5_i32 overflow<nsw> : i32
                      %216 = arith.index_cast %215 : i32 to index
                      %217 = arith.muli %200, %c8_i32 overflow<nsw> : i32
                      %218 = arith.addi %217, %c1_i32 overflow<nsw> : i32
                      %219 = arith.index_cast %218 : i32 to index
                      %220 = arith.addi %217, %c5_i32 overflow<nsw> : i32
                      %221 = arith.index_cast %220 : i32 to index
                      %222 = arith.muli %204, %c8_i32 overflow<nsw> : i32
                      %223 = arith.addi %222, %c1_i32 overflow<nsw> : i32
                      %224 = arith.index_cast %223 : i32 to index
                      %225 = arith.addi %222, %c5_i32 overflow<nsw> : i32
                      %226 = arith.index_cast %225 : i32 to index
                      %227 = arith.muli %208, %c8_i32 overflow<nsw> : i32
                      %228 = arith.addi %227, %c1_i32 overflow<nsw> : i32
                      %229 = arith.index_cast %228 : i32 to index
                      %230 = arith.addi %227, %c5_i32 overflow<nsw> : i32
                      %231 = arith.index_cast %230 : i32 to index
                      ascvec.vf(%arg27, %alloc_11, %alloc_19, %alloc_13, %199, %alloc_20, %203, %alloc_21, %207, %alloc_22, %211, %alloc_23, %alloc_14, %214, %216, %219, %221, %224, %226, %229, %231 : i64, memref<8192xf32, 1>, memref<1024xui16, 1>, memref<512xi32, 1>, index, memref<256xui16, 1>, index, memref<256xui16, 1>, index, memref<256xui16, 1>, index, memref<256xui16, 1>, memref<256xi32, 1>, index, index, index, index, index, index, index, index) {
                      ^bb0(%arg28: i64, %arg29: memref<8192xf32, 1>, %arg30: memref<1024xui16, 1>, %arg31: memref<512xi32, 1>, %arg32: index, %arg33: memref<256xui16, 1>, %arg34: index, %arg35: memref<256xui16, 1>, %arg36: index, %arg37: memref<256xui16, 1>, %arg38: index, %arg39: memref<256xui16, 1>, %arg40: memref<256xi32, 1>, %arg41: index, %arg42: index, %arg43: index, %arg44: index, %arg45: index, %arg46: index, %arg47: index, %arg48: index):
                        %c896 = arith.constant 896 : index
                        %c768 = arith.constant 768 : index
                        %c640 = arith.constant 640 : index
                        %c512 = arith.constant 512 : index
                        %c384 = arith.constant 384 : index
                        %c256 = arith.constant 256 : index
                        %c128 = arith.constant 128 : index
                        %c768_i16 = arith.constant 768 : i16
                        %c512_i16 = arith.constant 512 : i16
                        %c256_i16 = arith.constant 256 : i16
                        %c128_i16 = arith.constant 128 : i16
                        %c257_i32 = arith.constant 257 : i32
                        %c1_i32_30 = arith.constant 1 : i32
                        %c4_i32_31 = arith.constant 4 : i32
                        %c0_i32_32 = arith.constant 0 : i32
                        %c0_33 = arith.constant 0 : index
                        %c65535_i32 = arith.constant 65535 : i32
                        %c255_i32 = arith.constant 255 : i32
                        %c1_i16 = arith.constant 1 : i16
                        %c0_i16 = arith.constant 0 : i16
                        %c64_i32_34 = arith.constant 64 : i32
                        %c128_i32_35 = arith.constant 128 : i32
                        %c256_i32_36 = arith.constant 256 : i32
                        %mask, %remaining_out = ascvec.update_mask %c256_i32_36 elem_bits 8 : !ascvec.mask
                        %mask_37, %remaining_out_38 = ascvec.update_mask %c128_i32_35 elem_bits 16 : !ascvec.mask
                        %mask_39, %remaining_out_40 = ascvec.update_mask %c64_i32_34 : !ascvec.mask
                        %288 = ascvec.duplicate_scalar %c0_i16, %mask_37 : i16 -> <ui16>
                        %289 = ascvec.duplicate_scalar %c1_i16, %mask_37 : i16 -> <ui16>
                        %290 = ascvec.duplicate_scalar %c255_i32, %mask_39 : i32 -> <ui32>
                        %291 = ascvec.duplicate_scalar %c65535_i32, %mask_39 : i32 -> <ui32>
                        scf.for %arg49 = %c0_i32_32 to %c4_i32_31 step %c1_i32_30  : i32 {
                          %339 = arith.trunci %arg28 : i64 to i32
                          %340 = arith.addi %339, %arg49 : i32
                          %341 = arith.muli %340, %c256_i32_36 : i32
                          %342 = arith.index_cast %341 : i32 to index
                          %result0, %result1 = ascvec.loadalign_deintlv %arg29[%342] {dintlv_width = "b16"} : memref<8192xf32, 1>, index -> <ui16>, <ui16>
                          %343 = arith.addi %341, %c128_i32_35 : i32
                          %344 = arith.index_cast %343 : i32 to index
                          %result0_42, %result1_43 = ascvec.loadalign_deintlv %arg29[%344] {dintlv_width = "b16"} : memref<8192xf32, 1>, index -> <ui16>, <ui16>
                          %345 = ascvec.reinterpret %result1 : <ui16> -> <ui8>
                          %346 = ascvec.reinterpret %result1_43 : <ui16> -> <ui8>
                          %result0_44, %result1_45 = ascvec.deintlv %345, %346 : <ui8>, <ui8> -> <ui8>, <ui8>
                          %347 = ascvec.cumulative_histogram %288, %result1_45, %mask : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          %348 = ascvec.cumulative_histogram %288, %result1_45, %mask {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          %349 = arith.muli %arg49, %c256_i32_36 overflow<nsw> : i32
                          %350 = arith.index_cast %349 : i32 to index
                          ascvec.storealign %arg30[%350], %347, %mask_37 : memref<1024xui16, 1>, index, <ui16>
                          %351 = arith.addi %349, %c128_i32_35 overflow<nsw> : i32
                          %352 = arith.index_cast %351 : i32 to index
                          ascvec.storealign %arg30[%352], %348, %mask_37 : memref<1024xui16, 1>, index, <ui16>
                        }
                        ascvec.mem_bar vst_vld
                        %292 = ascvec.loadalign_brc %arg31[%arg32] : memref<512xi32, 1>, index -> <i32>
                        %293 = ascvec.max_scalar %292, %c1_i32_30, %mask_39 : <i32>, i32
                        %294 = ascvec.duplicate_scalar %c257_i32, %mask_39 : i32 -> <i32>
                        %295 = ascvec.sub %294, %293, %mask_39 : <i32>
                        %296 = ascvec.reinterpret %295 : <i32> -> <ui16>
                        %297 = ascvec.duplicate %296, %mask_37 : <ui16> -> <ui16>
                        ascvec.store_unalign_begin %arg33 : memref<256xui16, 1>
                        %298 = ascvec.arange %c0_i16 : i16 -> <ui16>
                        %299 = ascvec.loadalign %arg30[%c0_33] : memref<1024xui16, 1>, index -> <ui16>
                        %300 = ascvec.cmp_ge %299, %297, %mask_37 : <ui16>, <ui16> -> !ascvec.mask
                        %301 = ascvec.squeeze_store_reg %298, %300 : <ui16>, !ascvec.mask -> <ui16>
                        ascvec.store_unalign %arg33[%c0_33], %301 : memref<256xui16, 1>, <ui16>
                        %302 = ascvec.arange %c128_i16 : i16 -> <ui16>
                        %303 = ascvec.loadalign %arg30[%c128] : memref<1024xui16, 1>, index -> <ui16>
                        %304 = ascvec.cmp_ge %303, %297, %mask_37 : <ui16>, <ui16> -> !ascvec.mask
                        %305 = ascvec.squeeze_store_reg %302, %304 : <ui16>, !ascvec.mask -> <ui16>
                        ascvec.store_unalign %arg33[%c0_33], %305 : memref<256xui16, 1>, <ui16>
                        ascvec.store_unalign_post %arg33[%c0_33] : memref<256xui16, 1>
                        %306 = ascvec.loadalign_brc %arg31[%arg34] : memref<512xi32, 1>, index -> <i32>
                        %307 = ascvec.max_scalar %306, %c1_i32_30, %mask_39 : <i32>, i32
                        %308 = ascvec.sub %294, %307, %mask_39 : <i32>
                        %309 = ascvec.reinterpret %308 : <i32> -> <ui16>
                        %310 = ascvec.duplicate %309, %mask_37 : <ui16> -> <ui16>
                        ascvec.store_unalign_begin %arg35 : memref<256xui16, 1>
                        %311 = ascvec.loadalign %arg30[%c256] : memref<1024xui16, 1>, index -> <ui16>
                        %312 = ascvec.cmp_ge %311, %310, %mask_37 : <ui16>, <ui16> -> !ascvec.mask
                        %313 = ascvec.squeeze_store_reg %298, %312 : <ui16>, !ascvec.mask -> <ui16>
                        ascvec.store_unalign %arg35[%c0_33], %313 : memref<256xui16, 1>, <ui16>
                        %314 = ascvec.loadalign %arg30[%c384] : memref<1024xui16, 1>, index -> <ui16>
                        %315 = ascvec.cmp_ge %314, %310, %mask_37 : <ui16>, <ui16> -> !ascvec.mask
                        %316 = ascvec.squeeze_store_reg %302, %315 : <ui16>, !ascvec.mask -> <ui16>
                        ascvec.store_unalign %arg35[%c0_33], %316 : memref<256xui16, 1>, <ui16>
                        ascvec.store_unalign_post %arg35[%c0_33] : memref<256xui16, 1>
                        %317 = ascvec.loadalign_brc %arg31[%arg36] : memref<512xi32, 1>, index -> <i32>
                        %318 = ascvec.max_scalar %317, %c1_i32_30, %mask_39 : <i32>, i32
                        %319 = ascvec.sub %294, %318, %mask_39 : <i32>
                        %320 = ascvec.reinterpret %319 : <i32> -> <ui16>
                        %321 = ascvec.duplicate %320, %mask_37 : <ui16> -> <ui16>
                        ascvec.store_unalign_begin %arg37 : memref<256xui16, 1>
                        %322 = ascvec.loadalign %arg30[%c512] : memref<1024xui16, 1>, index -> <ui16>
                        %323 = ascvec.cmp_ge %322, %321, %mask_37 : <ui16>, <ui16> -> !ascvec.mask
                        %324 = ascvec.squeeze_store_reg %298, %323 : <ui16>, !ascvec.mask -> <ui16>
                        ascvec.store_unalign %arg37[%c0_33], %324 : memref<256xui16, 1>, <ui16>
                        %325 = ascvec.loadalign %arg30[%c640] : memref<1024xui16, 1>, index -> <ui16>
                        %326 = ascvec.cmp_ge %325, %321, %mask_37 : <ui16>, <ui16> -> !ascvec.mask
                        %327 = ascvec.squeeze_store_reg %302, %326 : <ui16>, !ascvec.mask -> <ui16>
                        ascvec.store_unalign %arg37[%c0_33], %327 : memref<256xui16, 1>, <ui16>
                        ascvec.store_unalign_post %arg37[%c0_33] : memref<256xui16, 1>
                        %328 = ascvec.loadalign_brc %arg31[%arg38] : memref<512xi32, 1>, index -> <i32>
                        %329 = ascvec.max_scalar %328, %c1_i32_30, %mask_39 : <i32>, i32
                        %330 = ascvec.sub %294, %329, %mask_39 : <i32>
                        %331 = ascvec.reinterpret %330 : <i32> -> <ui16>
                        %332 = ascvec.duplicate %331, %mask_37 : <ui16> -> <ui16>
                        ascvec.store_unalign_begin %arg39 : memref<256xui16, 1>
                        %333 = ascvec.loadalign %arg30[%c768] : memref<1024xui16, 1>, index -> <ui16>
                        %334 = ascvec.cmp_ge %333, %332, %mask_37 : <ui16>, <ui16> -> !ascvec.mask
                        %335 = ascvec.squeeze_store_reg %298, %334 : <ui16>, !ascvec.mask -> <ui16>
                        ascvec.store_unalign %arg39[%c0_33], %335 : memref<256xui16, 1>, <ui16>
                        %336 = ascvec.loadalign %arg30[%c896] : memref<1024xui16, 1>, index -> <ui16>
                        %337 = ascvec.cmp_ge %336, %332, %mask_37 : <ui16>, <ui16> -> !ascvec.mask
                        %338 = ascvec.squeeze_store_reg %302, %337 : <ui16>, !ascvec.mask -> <ui16>
                        ascvec.store_unalign %arg39[%c0_33], %338 : memref<256xui16, 1>, <ui16>
                        ascvec.store_unalign_post %arg39[%c0_33] : memref<256xui16, 1>
                        ascvec.mem_bar vst_vld
                        %c0_41 = arith.constant 0 : index
                        %c1 = arith.constant 1 : index
                        scf.for %arg49 = %c0_41 to %c1 step %c1 {
                          %339 = ascvec.loadalign_brc %arg31[%arg32] : memref<512xi32, 1>, index -> <i32>
                          %340 = ascvec.max_scalar %339, %c1_i32_30, %mask_39 : <i32>, i32
                          %341 = ascvec.sub %294, %340, %mask_39 : <i32>
                          %342 = ascvec.reinterpret %341 : <i32> -> <ui16>
                          %343 = ascvec.duplicate %342, %mask_37 : <ui16> -> <ui16>
                          %344 = ascvec.loadalign_brc %arg33[%c0_33] : memref<256xui16, 1>, index -> <ui16>
                          %345 = ascvec.reinterpret %344 : <ui16> -> <ui32>
                          %346 = ascvec.and %345, %290, %mask_39 : <ui32>, <ui32> -> <ui32>
                          %347 = ascvec.reinterpret %346 : <ui32> -> <i32>
                          ascvec.storealign_1st %arg40[%arg41], %347 : memref<256xi32, 1>, index, <i32>
                          %348 = ascvec.cmp_eq_scalar %344, %c0_i16, %mask_37 : <ui16>, i16 -> !ascvec.mask
                          %349 = ascvec.sub %344, %289, %mask_37 : <ui16>
                          %350 = ascvec.select %288, %349, %348 : <ui16>
                          %351 = ascvec.add %350, %288, %mask_37 : <ui16>
                          %352 = ascvec.gather %arg30[%c0_33], %351, %mask_37 : memref<1024xui16, 1>, <ui16>, !ascvec.mask -> <ui16>
                          %353 = ascvec.select %288, %352, %348 : <ui16>
                          %354 = ascvec.sub %343, %353, %mask_37 : <ui16>
                          %355 = ascvec.reinterpret %354 : <ui16> -> <ui32>
                          %356 = ascvec.and %355, %291, %mask_39 : <ui32>, <ui32> -> <ui32>
                          %357 = ascvec.reinterpret %356 : <ui32> -> <i32>
                          ascvec.storealign_1st %arg40[%arg42], %357 : memref<256xi32, 1>, index, <i32>
                          %358 = ascvec.loadalign_brc %arg31[%arg34] : memref<512xi32, 1>, index -> <i32>
                          %359 = ascvec.max_scalar %358, %c1_i32_30, %mask_39 : <i32>, i32
                          %360 = ascvec.sub %294, %359, %mask_39 : <i32>
                          %361 = ascvec.reinterpret %360 : <i32> -> <ui16>
                          %362 = ascvec.duplicate %361, %mask_37 : <ui16> -> <ui16>
                          %363 = ascvec.loadalign_brc %arg35[%c0_33] : memref<256xui16, 1>, index -> <ui16>
                          %364 = ascvec.reinterpret %363 : <ui16> -> <ui32>
                          %365 = ascvec.and %364, %290, %mask_39 : <ui32>, <ui32> -> <ui32>
                          %366 = ascvec.reinterpret %365 : <ui32> -> <i32>
                          ascvec.storealign_1st %arg40[%arg43], %366 : memref<256xi32, 1>, index, <i32>
                          %367 = ascvec.cmp_eq_scalar %363, %c0_i16, %mask_37 : <ui16>, i16 -> !ascvec.mask
                          %368 = ascvec.sub %363, %289, %mask_37 : <ui16>
                          %369 = ascvec.select %288, %368, %367 : <ui16>
                          %370 = ascvec.duplicate_scalar %c256_i16, %mask_37 : i16 -> <ui16>
                          %371 = ascvec.add %369, %370, %mask_37 : <ui16>
                          %372 = ascvec.gather %arg30[%c0_33], %371, %mask_37 : memref<1024xui16, 1>, <ui16>, !ascvec.mask -> <ui16>
                          %373 = ascvec.select %288, %372, %367 : <ui16>
                          %374 = ascvec.sub %362, %373, %mask_37 : <ui16>
                          %375 = ascvec.reinterpret %374 : <ui16> -> <ui32>
                          %376 = ascvec.and %375, %291, %mask_39 : <ui32>, <ui32> -> <ui32>
                          %377 = ascvec.reinterpret %376 : <ui32> -> <i32>
                          ascvec.storealign_1st %arg40[%arg44], %377 : memref<256xi32, 1>, index, <i32>
                          %378 = ascvec.loadalign_brc %arg31[%arg36] : memref<512xi32, 1>, index -> <i32>
                          %379 = ascvec.max_scalar %378, %c1_i32_30, %mask_39 : <i32>, i32
                          %380 = ascvec.sub %294, %379, %mask_39 : <i32>
                          %381 = ascvec.reinterpret %380 : <i32> -> <ui16>
                          %382 = ascvec.duplicate %381, %mask_37 : <ui16> -> <ui16>
                          %383 = ascvec.loadalign_brc %arg37[%c0_33] : memref<256xui16, 1>, index -> <ui16>
                          %384 = ascvec.reinterpret %383 : <ui16> -> <ui32>
                          %385 = ascvec.and %384, %290, %mask_39 : <ui32>, <ui32> -> <ui32>
                          %386 = ascvec.reinterpret %385 : <ui32> -> <i32>
                          ascvec.storealign_1st %arg40[%arg45], %386 : memref<256xi32, 1>, index, <i32>
                          %387 = ascvec.cmp_eq_scalar %383, %c0_i16, %mask_37 : <ui16>, i16 -> !ascvec.mask
                          %388 = ascvec.sub %383, %289, %mask_37 : <ui16>
                          %389 = ascvec.select %288, %388, %387 : <ui16>
                          %390 = ascvec.duplicate_scalar %c512_i16, %mask_37 : i16 -> <ui16>
                          %391 = ascvec.add %389, %390, %mask_37 : <ui16>
                          %392 = ascvec.gather %arg30[%c0_33], %391, %mask_37 : memref<1024xui16, 1>, <ui16>, !ascvec.mask -> <ui16>
                          %393 = ascvec.select %288, %392, %387 : <ui16>
                          %394 = ascvec.sub %382, %393, %mask_37 : <ui16>
                          %395 = ascvec.reinterpret %394 : <ui16> -> <ui32>
                          %396 = ascvec.and %395, %291, %mask_39 : <ui32>, <ui32> -> <ui32>
                          %397 = ascvec.reinterpret %396 : <ui32> -> <i32>
                          ascvec.storealign_1st %arg40[%arg46], %397 : memref<256xi32, 1>, index, <i32>
                          %398 = ascvec.loadalign_brc %arg31[%arg38] : memref<512xi32, 1>, index -> <i32>
                          %399 = ascvec.max_scalar %398, %c1_i32_30, %mask_39 : <i32>, i32
                          %400 = ascvec.sub %294, %399, %mask_39 : <i32>
                          %401 = ascvec.reinterpret %400 : <i32> -> <ui16>
                          %402 = ascvec.duplicate %401, %mask_37 : <ui16> -> <ui16>
                          %403 = ascvec.loadalign_brc %arg39[%c0_33] : memref<256xui16, 1>, index -> <ui16>
                          %404 = ascvec.reinterpret %403 : <ui16> -> <ui32>
                          %405 = ascvec.and %404, %290, %mask_39 : <ui32>, <ui32> -> <ui32>
                          %406 = ascvec.reinterpret %405 : <ui32> -> <i32>
                          ascvec.storealign_1st %arg40[%arg47], %406 : memref<256xi32, 1>, index, <i32>
                          %407 = ascvec.cmp_eq_scalar %403, %c0_i16, %mask_37 : <ui16>, i16 -> !ascvec.mask
                          %408 = ascvec.sub %403, %289, %mask_37 : <ui16>
                          %409 = ascvec.select %288, %408, %407 : <ui16>
                          %410 = ascvec.duplicate_scalar %c768_i16, %mask_37 : i16 -> <ui16>
                          %411 = ascvec.add %409, %410, %mask_37 : <ui16>
                          %412 = ascvec.gather %arg30[%c0_33], %411, %mask_37 : memref<1024xui16, 1>, <ui16>, !ascvec.mask -> <ui16>
                          %413 = ascvec.select %288, %412, %407 : <ui16>
                          %414 = ascvec.sub %402, %413, %mask_37 : <ui16>
                          %415 = ascvec.reinterpret %414 : <ui16> -> <ui32>
                          %416 = ascvec.and %415, %291, %mask_39 : <ui32>, <ui32> -> <ui32>
                          %417 = ascvec.reinterpret %416 : <ui32> -> <i32>
                          ascvec.storealign_1st %arg40[%arg48], %417 : memref<256xi32, 1>, index, <i32>
                        }
                      } {cannir.vf_mode = "simd"}
                      %232 = arith.addi %212, %c2_i32 overflow<nsw> : i32
                      %233 = arith.index_cast %232 : i32 to index
                      %234 = arith.addi %217, %c2_i32 overflow<nsw> : i32
                      %235 = arith.index_cast %234 : i32 to index
                      %236 = arith.addi %222, %c2_i32 overflow<nsw> : i32
                      %237 = arith.index_cast %236 : i32 to index
                      %238 = arith.addi %227, %c2_i32 overflow<nsw> : i32
                      %239 = arith.index_cast %238 : i32 to index
                      ascvec.vf(%196, %alloc_11, %alloc_14, %alloc_19, %216, %alloc_20, %221, %alloc_21, %226, %alloc_22, %231, %alloc_23, %233, %235, %237, %239 : i32, memref<8192xf32, 1>, memref<256xi32, 1>, memref<1024xui16, 1>, index, memref<256xui16, 1>, index, memref<256xui16, 1>, index, memref<256xui16, 1>, index, memref<256xui16, 1>, index, index, index, index) {
                      ^bb0(%arg28: i32, %arg29: memref<8192xf32, 1>, %arg30: memref<256xi32, 1>, %arg31: memref<1024xui16, 1>, %arg32: index, %arg33: memref<256xui16, 1>, %arg34: index, %arg35: memref<256xui16, 1>, %arg36: index, %arg37: memref<256xui16, 1>, %arg38: index, %arg39: memref<256xui16, 1>, %arg40: index, %arg41: index, %arg42: index, %arg43: index):
                        %c896 = arith.constant 896 : index
                        %c768 = arith.constant 768 : index
                        %c640 = arith.constant 640 : index
                        %c512 = arith.constant 512 : index
                        %c384 = arith.constant 384 : index
                        %c256 = arith.constant 256 : index
                        %c128 = arith.constant 128 : index
                        %c768_i16 = arith.constant 768 : i16
                        %c512_i16 = arith.constant 512 : i16
                        %c256_i16 = arith.constant 256 : i16
                        %c128_i16 = arith.constant 128 : i16
                        %c4_i32_30 = arith.constant 4 : i32
                        %c0_i32_31 = arith.constant 0 : i32
                        %c1_i32_32 = arith.constant 1 : i32
                        %c8_i32_33 = arith.constant 8 : i32
                        %c0_34 = arith.constant 0 : index
                        %c65535_i32 = arith.constant 65535 : i32
                        %c255_i32 = arith.constant 255 : i32
                        %c1_i16 = arith.constant 1 : i16
                        %c0_i16 = arith.constant 0 : i16
                        %c64_i32_35 = arith.constant 64 : i32
                        %c128_i32_36 = arith.constant 128 : i32
                        %c256_i32_37 = arith.constant 256 : i32
                        %mask, %remaining_out = ascvec.update_mask %c256_i32_37 elem_bits 8 : !ascvec.mask
                        %mask_38, %remaining_out_39 = ascvec.update_mask %c128_i32_36 elem_bits 16 : !ascvec.mask
                        %mask_40, %remaining_out_41 = ascvec.update_mask %c64_i32_35 : !ascvec.mask
                        %288 = ascvec.duplicate_scalar %c0_i16, %mask_38 : i16 -> <ui16>
                        %289 = ascvec.duplicate_scalar %c1_i16, %mask_38 : i16 -> <ui16>
                        %290 = ascvec.duplicate_scalar %c255_i32, %mask_40 : i32 -> <ui32>
                        %291 = ascvec.duplicate_scalar %c65535_i32, %mask_40 : i32 -> <ui32>
                        scf.for %arg44 = %c0_i32_31 to %c4_i32_30 step %c1_i32_32  : i32 {
                          %292 = arith.addi %arg28, %arg44 : i32
                          %293 = arith.muli %292, %c256_i32_37 : i32
                          %294 = arith.index_cast %293 : i32 to index
                          %result0, %result1 = ascvec.loadalign_deintlv %arg29[%294] {dintlv_width = "b16"} : memref<8192xf32, 1>, index -> <ui16>, <ui16>
                          %295 = arith.addi %293, %c128_i32_36 : i32
                          %296 = arith.index_cast %295 : i32 to index
                          %result0_45, %result1_46 = ascvec.loadalign_deintlv %arg29[%296] {dintlv_width = "b16"} : memref<8192xf32, 1>, index -> <ui16>, <ui16>
                          %297 = ascvec.reinterpret %result1 : <ui16> -> <ui8>
                          %298 = ascvec.reinterpret %result1_46 : <ui16> -> <ui8>
                          %result0_47, %result1_48 = ascvec.deintlv %297, %298 : <ui8>, <ui8> -> <ui8>, <ui8>
                          %299 = arith.muli %292, %c8_i32_33 : i32
                          %300 = arith.addi %299, %c1_i32_32 : i32
                          %301 = arith.index_cast %300 : i32 to index
                          %302 = ascvec.loadalign_brc %arg30[%301] {brc_width = "b8"} : memref<256xi32, 1>, index -> <ui32>
                          %303 = ascvec.reinterpret %302 : <ui32> -> <ui8>
                          %304 = ascvec.cmp_eq %result1_48, %303, %mask : <ui8>, <ui8> -> !ascvec.mask
                          %305 = ascvec.cumulative_histogram %288, %result0_47, %304 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          %306 = ascvec.cumulative_histogram %288, %result0_47, %304 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          %307 = arith.muli %arg44, %c256_i32_37 overflow<nsw> : i32
                          %308 = arith.index_cast %307 : i32 to index
                          ascvec.storealign %arg31[%308], %305, %mask_38 : memref<1024xui16, 1>, index, <ui16>
                          %309 = arith.addi %307, %c128_i32_36 overflow<nsw> : i32
                          %310 = arith.index_cast %309 : i32 to index
                          ascvec.storealign %arg31[%310], %306, %mask_38 : memref<1024xui16, 1>, index, <ui16>
                        }
                        ascvec.mem_bar vst_vld
                        %c0_42 = arith.constant 0 : index
                        %c1 = arith.constant 1 : index
                        scf.for %arg44 = %c0_42 to %c1 step %c1 {
                          %292 = ascvec.loadalign_brc %arg30[%arg32] : memref<256xi32, 1>, index -> <i32>
                          %293 = ascvec.reinterpret %292 : <i32> -> <ui16>
                          %294 = ascvec.duplicate %293, %mask_38 : <ui16> -> <ui16>
                          ascvec.store_unalign_begin %arg33 : memref<256xui16, 1>
                          %295 = ascvec.arange %c0_i16 : i16 -> <ui16>
                          %296 = ascvec.loadalign %arg31[%c0_34] : memref<1024xui16, 1>, index -> <ui16>
                          %297 = ascvec.cmp_ge %296, %294, %mask_38 : <ui16>, <ui16> -> !ascvec.mask
                          %298 = ascvec.squeeze_store_reg %295, %297 : <ui16>, !ascvec.mask -> <ui16>
                          ascvec.store_unalign %arg33[%c0_34], %298 : memref<256xui16, 1>, <ui16>
                          %299 = ascvec.arange %c128_i16 : i16 -> <ui16>
                          %300 = ascvec.loadalign %arg31[%c128] : memref<1024xui16, 1>, index -> <ui16>
                          %301 = ascvec.cmp_ge %300, %294, %mask_38 : <ui16>, <ui16> -> !ascvec.mask
                          %302 = ascvec.squeeze_store_reg %299, %301 : <ui16>, !ascvec.mask -> <ui16>
                          ascvec.store_unalign %arg33[%c0_34], %302 : memref<256xui16, 1>, <ui16>
                          ascvec.store_unalign_post %arg33[%c0_34] : memref<256xui16, 1>
                          %303 = ascvec.loadalign_brc %arg30[%arg34] : memref<256xi32, 1>, index -> <i32>
                          %304 = ascvec.reinterpret %303 : <i32> -> <ui16>
                          %305 = ascvec.duplicate %304, %mask_38 : <ui16> -> <ui16>
                          ascvec.store_unalign_begin %arg35 : memref<256xui16, 1>
                          %306 = ascvec.loadalign %arg31[%c256] : memref<1024xui16, 1>, index -> <ui16>
                          %307 = ascvec.cmp_ge %306, %305, %mask_38 : <ui16>, <ui16> -> !ascvec.mask
                          %308 = ascvec.squeeze_store_reg %295, %307 : <ui16>, !ascvec.mask -> <ui16>
                          ascvec.store_unalign %arg35[%c0_34], %308 : memref<256xui16, 1>, <ui16>
                          %309 = ascvec.loadalign %arg31[%c384] : memref<1024xui16, 1>, index -> <ui16>
                          %310 = ascvec.cmp_ge %309, %305, %mask_38 : <ui16>, <ui16> -> !ascvec.mask
                          %311 = ascvec.squeeze_store_reg %299, %310 : <ui16>, !ascvec.mask -> <ui16>
                          ascvec.store_unalign %arg35[%c0_34], %311 : memref<256xui16, 1>, <ui16>
                          ascvec.store_unalign_post %arg35[%c0_34] : memref<256xui16, 1>
                          %312 = ascvec.loadalign_brc %arg30[%arg36] : memref<256xi32, 1>, index -> <i32>
                          %313 = ascvec.reinterpret %312 : <i32> -> <ui16>
                          %314 = ascvec.duplicate %313, %mask_38 : <ui16> -> <ui16>
                          ascvec.store_unalign_begin %arg37 : memref<256xui16, 1>
                          %315 = ascvec.loadalign %arg31[%c512] : memref<1024xui16, 1>, index -> <ui16>
                          %316 = ascvec.cmp_ge %315, %314, %mask_38 : <ui16>, <ui16> -> !ascvec.mask
                          %317 = ascvec.squeeze_store_reg %295, %316 : <ui16>, !ascvec.mask -> <ui16>
                          ascvec.store_unalign %arg37[%c0_34], %317 : memref<256xui16, 1>, <ui16>
                          %318 = ascvec.loadalign %arg31[%c640] : memref<1024xui16, 1>, index -> <ui16>
                          %319 = ascvec.cmp_ge %318, %314, %mask_38 : <ui16>, <ui16> -> !ascvec.mask
                          %320 = ascvec.squeeze_store_reg %299, %319 : <ui16>, !ascvec.mask -> <ui16>
                          ascvec.store_unalign %arg37[%c0_34], %320 : memref<256xui16, 1>, <ui16>
                          ascvec.store_unalign_post %arg37[%c0_34] : memref<256xui16, 1>
                          %321 = ascvec.loadalign_brc %arg30[%arg38] : memref<256xi32, 1>, index -> <i32>
                          %322 = ascvec.reinterpret %321 : <i32> -> <ui16>
                          %323 = ascvec.duplicate %322, %mask_38 : <ui16> -> <ui16>
                          ascvec.store_unalign_begin %arg39 : memref<256xui16, 1>
                          %324 = ascvec.loadalign %arg31[%c768] : memref<1024xui16, 1>, index -> <ui16>
                          %325 = ascvec.cmp_ge %324, %323, %mask_38 : <ui16>, <ui16> -> !ascvec.mask
                          %326 = ascvec.squeeze_store_reg %295, %325 : <ui16>, !ascvec.mask -> <ui16>
                          ascvec.store_unalign %arg39[%c0_34], %326 : memref<256xui16, 1>, <ui16>
                          %327 = ascvec.loadalign %arg31[%c896] : memref<1024xui16, 1>, index -> <ui16>
                          %328 = ascvec.cmp_ge %327, %323, %mask_38 : <ui16>, <ui16> -> !ascvec.mask
                          %329 = ascvec.squeeze_store_reg %299, %328 : <ui16>, !ascvec.mask -> <ui16>
                          ascvec.store_unalign %arg39[%c0_34], %329 : memref<256xui16, 1>, <ui16>
                          ascvec.store_unalign_post %arg39[%c0_34] : memref<256xui16, 1>
                        }
                        ascvec.mem_bar vst_vld
                        %c0_43 = arith.constant 0 : index
                        %c1_44 = arith.constant 1 : index
                        scf.for %arg44 = %c0_43 to %c1_44 step %c1_44 {
                          %292 = ascvec.loadalign_brc %arg30[%arg32] : memref<256xi32, 1>, index -> <i32>
                          %293 = ascvec.reinterpret %292 : <i32> -> <ui16>
                          %294 = ascvec.duplicate %293, %mask_38 : <ui16> -> <ui16>
                          %295 = ascvec.loadalign_brc %arg33[%c0_34] : memref<256xui16, 1>, index -> <ui16>
                          %296 = ascvec.reinterpret %295 : <ui16> -> <ui32>
                          %297 = ascvec.and %296, %290, %mask_40 : <ui32>, <ui32> -> <ui32>
                          %298 = ascvec.reinterpret %297 : <ui32> -> <i32>
                          ascvec.storealign_1st %arg30[%arg40], %298 : memref<256xi32, 1>, index, <i32>
                          %299 = ascvec.cmp_eq_scalar %295, %c0_i16, %mask_38 : <ui16>, i16 -> !ascvec.mask
                          %300 = ascvec.sub %295, %289, %mask_38 : <ui16>
                          %301 = ascvec.select %288, %300, %299 : <ui16>
                          %302 = ascvec.add %301, %288, %mask_38 : <ui16>
                          %303 = ascvec.gather %arg31[%c0_34], %302, %mask_38 : memref<1024xui16, 1>, <ui16>, !ascvec.mask -> <ui16>
                          %304 = ascvec.select %288, %303, %299 : <ui16>
                          %305 = ascvec.sub %294, %304, %mask_38 : <ui16>
                          %306 = ascvec.reinterpret %305 : <ui16> -> <ui32>
                          %307 = ascvec.and %306, %291, %mask_40 : <ui32>, <ui32> -> <ui32>
                          %308 = ascvec.reinterpret %307 : <ui32> -> <i32>
                          ascvec.storealign_1st %arg30[%arg32], %308 : memref<256xi32, 1>, index, <i32>
                          %309 = ascvec.loadalign_brc %arg30[%arg34] : memref<256xi32, 1>, index -> <i32>
                          %310 = ascvec.reinterpret %309 : <i32> -> <ui16>
                          %311 = ascvec.duplicate %310, %mask_38 : <ui16> -> <ui16>
                          %312 = ascvec.loadalign_brc %arg35[%c0_34] : memref<256xui16, 1>, index -> <ui16>
                          %313 = ascvec.reinterpret %312 : <ui16> -> <ui32>
                          %314 = ascvec.and %313, %290, %mask_40 : <ui32>, <ui32> -> <ui32>
                          %315 = ascvec.reinterpret %314 : <ui32> -> <i32>
                          ascvec.storealign_1st %arg30[%arg41], %315 : memref<256xi32, 1>, index, <i32>
                          %316 = ascvec.cmp_eq_scalar %312, %c0_i16, %mask_38 : <ui16>, i16 -> !ascvec.mask
                          %317 = ascvec.sub %312, %289, %mask_38 : <ui16>
                          %318 = ascvec.select %288, %317, %316 : <ui16>
                          %319 = ascvec.duplicate_scalar %c256_i16, %mask_38 : i16 -> <ui16>
                          %320 = ascvec.add %318, %319, %mask_38 : <ui16>
                          %321 = ascvec.gather %arg31[%c0_34], %320, %mask_38 : memref<1024xui16, 1>, <ui16>, !ascvec.mask -> <ui16>
                          %322 = ascvec.select %288, %321, %316 : <ui16>
                          %323 = ascvec.sub %311, %322, %mask_38 : <ui16>
                          %324 = ascvec.reinterpret %323 : <ui16> -> <ui32>
                          %325 = ascvec.and %324, %291, %mask_40 : <ui32>, <ui32> -> <ui32>
                          %326 = ascvec.reinterpret %325 : <ui32> -> <i32>
                          ascvec.storealign_1st %arg30[%arg34], %326 : memref<256xi32, 1>, index, <i32>
                          %327 = ascvec.loadalign_brc %arg30[%arg36] : memref<256xi32, 1>, index -> <i32>
                          %328 = ascvec.reinterpret %327 : <i32> -> <ui16>
                          %329 = ascvec.duplicate %328, %mask_38 : <ui16> -> <ui16>
                          %330 = ascvec.loadalign_brc %arg37[%c0_34] : memref<256xui16, 1>, index -> <ui16>
                          %331 = ascvec.reinterpret %330 : <ui16> -> <ui32>
                          %332 = ascvec.and %331, %290, %mask_40 : <ui32>, <ui32> -> <ui32>
                          %333 = ascvec.reinterpret %332 : <ui32> -> <i32>
                          ascvec.storealign_1st %arg30[%arg42], %333 : memref<256xi32, 1>, index, <i32>
                          %334 = ascvec.cmp_eq_scalar %330, %c0_i16, %mask_38 : <ui16>, i16 -> !ascvec.mask
                          %335 = ascvec.sub %330, %289, %mask_38 : <ui16>
                          %336 = ascvec.select %288, %335, %334 : <ui16>
                          %337 = ascvec.duplicate_scalar %c512_i16, %mask_38 : i16 -> <ui16>
                          %338 = ascvec.add %336, %337, %mask_38 : <ui16>
                          %339 = ascvec.gather %arg31[%c0_34], %338, %mask_38 : memref<1024xui16, 1>, <ui16>, !ascvec.mask -> <ui16>
                          %340 = ascvec.select %288, %339, %334 : <ui16>
                          %341 = ascvec.sub %329, %340, %mask_38 : <ui16>
                          %342 = ascvec.reinterpret %341 : <ui16> -> <ui32>
                          %343 = ascvec.and %342, %291, %mask_40 : <ui32>, <ui32> -> <ui32>
                          %344 = ascvec.reinterpret %343 : <ui32> -> <i32>
                          ascvec.storealign_1st %arg30[%arg36], %344 : memref<256xi32, 1>, index, <i32>
                          %345 = ascvec.loadalign_brc %arg30[%arg38] : memref<256xi32, 1>, index -> <i32>
                          %346 = ascvec.reinterpret %345 : <i32> -> <ui16>
                          %347 = ascvec.duplicate %346, %mask_38 : <ui16> -> <ui16>
                          %348 = ascvec.loadalign_brc %arg39[%c0_34] : memref<256xui16, 1>, index -> <ui16>
                          %349 = ascvec.reinterpret %348 : <ui16> -> <ui32>
                          %350 = ascvec.and %349, %290, %mask_40 : <ui32>, <ui32> -> <ui32>
                          %351 = ascvec.reinterpret %350 : <ui32> -> <i32>
                          ascvec.storealign_1st %arg30[%arg43], %351 : memref<256xi32, 1>, index, <i32>
                          %352 = ascvec.cmp_eq_scalar %348, %c0_i16, %mask_38 : <ui16>, i16 -> !ascvec.mask
                          %353 = ascvec.sub %348, %289, %mask_38 : <ui16>
                          %354 = ascvec.select %288, %353, %352 : <ui16>
                          %355 = ascvec.duplicate_scalar %c768_i16, %mask_38 : i16 -> <ui16>
                          %356 = ascvec.add %354, %355, %mask_38 : <ui16>
                          %357 = ascvec.gather %arg31[%c0_34], %356, %mask_38 : memref<1024xui16, 1>, <ui16>, !ascvec.mask -> <ui16>
                          %358 = ascvec.select %288, %357, %352 : <ui16>
                          %359 = ascvec.sub %347, %358, %mask_38 : <ui16>
                          %360 = ascvec.reinterpret %359 : <ui16> -> <ui32>
                          %361 = ascvec.and %360, %291, %mask_40 : <ui32>, <ui32> -> <ui32>
                          %362 = ascvec.reinterpret %361 : <ui32> -> <i32>
                          ascvec.storealign_1st %arg30[%arg38], %362 : memref<256xi32, 1>, index, <i32>
                        }
                      } {cannir.vf_mode = "simd"}
                      %240 = arith.addi %212, %c3_i32 overflow<nsw> : i32
                      %241 = arith.index_cast %240 : i32 to index
                      %242 = arith.addi %217, %c3_i32 overflow<nsw> : i32
                      %243 = arith.index_cast %242 : i32 to index
                      %244 = arith.addi %222, %c3_i32 overflow<nsw> : i32
                      %245 = arith.index_cast %244 : i32 to index
                      %246 = arith.addi %227, %c3_i32 overflow<nsw> : i32
                      %247 = arith.index_cast %246 : i32 to index
                      ascvec.vf(%196, %alloc_11, %alloc_14, %alloc_19, %216, %alloc_20, %221, %alloc_21, %226, %alloc_22, %231, %alloc_23, %241, %243, %245, %247 : i32, memref<8192xf32, 1>, memref<256xi32, 1>, memref<1024xui16, 1>, index, memref<256xui16, 1>, index, memref<256xui16, 1>, index, memref<256xui16, 1>, index, memref<256xui16, 1>, index, index, index, index) {
                      ^bb0(%arg28: i32, %arg29: memref<8192xf32, 1>, %arg30: memref<256xi32, 1>, %arg31: memref<1024xui16, 1>, %arg32: index, %arg33: memref<256xui16, 1>, %arg34: index, %arg35: memref<256xui16, 1>, %arg36: index, %arg37: memref<256xui16, 1>, %arg38: index, %arg39: memref<256xui16, 1>, %arg40: index, %arg41: index, %arg42: index, %arg43: index):
                        %c896 = arith.constant 896 : index
                        %c768 = arith.constant 768 : index
                        %c640 = arith.constant 640 : index
                        %c512 = arith.constant 512 : index
                        %c384 = arith.constant 384 : index
                        %c256 = arith.constant 256 : index
                        %c128 = arith.constant 128 : index
                        %c768_i16 = arith.constant 768 : i16
                        %c512_i16 = arith.constant 512 : i16
                        %c256_i16 = arith.constant 256 : i16
                        %c128_i16 = arith.constant 128 : i16
                        %c4_i32_30 = arith.constant 4 : i32
                        %c0_i32_31 = arith.constant 0 : i32
                        %c2_i32_32 = arith.constant 2 : i32
                        %c1_i32_33 = arith.constant 1 : i32
                        %c8_i32_34 = arith.constant 8 : i32
                        %c0_35 = arith.constant 0 : index
                        %c65535_i32 = arith.constant 65535 : i32
                        %c255_i32 = arith.constant 255 : i32
                        %c1_i16 = arith.constant 1 : i16
                        %c0_i16 = arith.constant 0 : i16
                        %c64_i32_36 = arith.constant 64 : i32
                        %c128_i32_37 = arith.constant 128 : i32
                        %c256_i32_38 = arith.constant 256 : i32
                        %mask, %remaining_out = ascvec.update_mask %c256_i32_38 elem_bits 8 : !ascvec.mask
                        %mask_39, %remaining_out_40 = ascvec.update_mask %c128_i32_37 elem_bits 16 : !ascvec.mask
                        %mask_41, %remaining_out_42 = ascvec.update_mask %c64_i32_36 : !ascvec.mask
                        %288 = ascvec.duplicate_scalar %c0_i16, %mask_39 : i16 -> <ui16>
                        %289 = ascvec.duplicate_scalar %c1_i16, %mask_39 : i16 -> <ui16>
                        %290 = ascvec.duplicate_scalar %c255_i32, %mask_41 : i32 -> <ui32>
                        %291 = ascvec.duplicate_scalar %c65535_i32, %mask_41 : i32 -> <ui32>
                        scf.for %arg44 = %c0_i32_31 to %c4_i32_30 step %c1_i32_33  : i32 {
                          %292 = arith.addi %arg28, %arg44 : i32
                          %293 = arith.muli %292, %c256_i32_38 : i32
                          %294 = arith.index_cast %293 : i32 to index
                          %result0, %result1 = ascvec.loadalign_deintlv %arg29[%294] {dintlv_width = "b16"} : memref<8192xf32, 1>, index -> <ui16>, <ui16>
                          %295 = arith.addi %293, %c128_i32_37 : i32
                          %296 = arith.index_cast %295 : i32 to index
                          %result0_46, %result1_47 = ascvec.loadalign_deintlv %arg29[%296] {dintlv_width = "b16"} : memref<8192xf32, 1>, index -> <ui16>, <ui16>
                          %297 = ascvec.reinterpret %result1 : <ui16> -> <ui8>
                          %298 = ascvec.reinterpret %result1_47 : <ui16> -> <ui8>
                          %result0_48, %result1_49 = ascvec.deintlv %297, %298 : <ui8>, <ui8> -> <ui8>, <ui8>
                          %299 = ascvec.reinterpret %result0 : <ui16> -> <ui8>
                          %300 = ascvec.reinterpret %result0_46 : <ui16> -> <ui8>
                          %result0_50, %result1_51 = ascvec.deintlv %299, %300 : <ui8>, <ui8> -> <ui8>, <ui8>
                          %301 = arith.muli %292, %c8_i32_34 : i32
                          %302 = arith.addi %301, %c1_i32_33 : i32
                          %303 = arith.index_cast %302 : i32 to index
                          %304 = ascvec.loadalign_brc %arg30[%303] {brc_width = "b8"} : memref<256xi32, 1>, index -> <ui32>
                          %305 = ascvec.reinterpret %304 : <ui32> -> <ui8>
                          %306 = ascvec.cmp_eq %result1_49, %305, %mask : <ui8>, <ui8> -> !ascvec.mask
                          %307 = arith.addi %301, %c2_i32_32 : i32
                          %308 = arith.index_cast %307 : i32 to index
                          %309 = ascvec.loadalign_brc %arg30[%308] {brc_width = "b8"} : memref<256xi32, 1>, index -> <ui32>
                          %310 = ascvec.reinterpret %309 : <ui32> -> <ui8>
                          %311 = ascvec.cmp_eq %result0_48, %310, %306 : <ui8>, <ui8> -> !ascvec.mask
                          %312 = ascvec.cumulative_histogram %288, %result1_51, %311 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          %313 = ascvec.cumulative_histogram %288, %result1_51, %311 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          %314 = arith.muli %arg44, %c256_i32_38 overflow<nsw> : i32
                          %315 = arith.index_cast %314 : i32 to index
                          ascvec.storealign %arg31[%315], %312, %mask_39 : memref<1024xui16, 1>, index, <ui16>
                          %316 = arith.addi %314, %c128_i32_37 overflow<nsw> : i32
                          %317 = arith.index_cast %316 : i32 to index
                          ascvec.storealign %arg31[%317], %313, %mask_39 : memref<1024xui16, 1>, index, <ui16>
                        }
                        ascvec.mem_bar vst_vld
                        %c0_43 = arith.constant 0 : index
                        %c1 = arith.constant 1 : index
                        scf.for %arg44 = %c0_43 to %c1 step %c1 {
                          %292 = ascvec.loadalign_brc %arg30[%arg32] : memref<256xi32, 1>, index -> <i32>
                          %293 = ascvec.reinterpret %292 : <i32> -> <ui16>
                          %294 = ascvec.duplicate %293, %mask_39 : <ui16> -> <ui16>
                          ascvec.store_unalign_begin %arg33 : memref<256xui16, 1>
                          %295 = ascvec.arange %c0_i16 : i16 -> <ui16>
                          %296 = ascvec.loadalign %arg31[%c0_35] : memref<1024xui16, 1>, index -> <ui16>
                          %297 = ascvec.cmp_ge %296, %294, %mask_39 : <ui16>, <ui16> -> !ascvec.mask
                          %298 = ascvec.squeeze_store_reg %295, %297 : <ui16>, !ascvec.mask -> <ui16>
                          ascvec.store_unalign %arg33[%c0_35], %298 : memref<256xui16, 1>, <ui16>
                          %299 = ascvec.arange %c128_i16 : i16 -> <ui16>
                          %300 = ascvec.loadalign %arg31[%c128] : memref<1024xui16, 1>, index -> <ui16>
                          %301 = ascvec.cmp_ge %300, %294, %mask_39 : <ui16>, <ui16> -> !ascvec.mask
                          %302 = ascvec.squeeze_store_reg %299, %301 : <ui16>, !ascvec.mask -> <ui16>
                          ascvec.store_unalign %arg33[%c0_35], %302 : memref<256xui16, 1>, <ui16>
                          ascvec.store_unalign_post %arg33[%c0_35] : memref<256xui16, 1>
                          %303 = ascvec.loadalign_brc %arg30[%arg34] : memref<256xi32, 1>, index -> <i32>
                          %304 = ascvec.reinterpret %303 : <i32> -> <ui16>
                          %305 = ascvec.duplicate %304, %mask_39 : <ui16> -> <ui16>
                          ascvec.store_unalign_begin %arg35 : memref<256xui16, 1>
                          %306 = ascvec.loadalign %arg31[%c256] : memref<1024xui16, 1>, index -> <ui16>
                          %307 = ascvec.cmp_ge %306, %305, %mask_39 : <ui16>, <ui16> -> !ascvec.mask
                          %308 = ascvec.squeeze_store_reg %295, %307 : <ui16>, !ascvec.mask -> <ui16>
                          ascvec.store_unalign %arg35[%c0_35], %308 : memref<256xui16, 1>, <ui16>
                          %309 = ascvec.loadalign %arg31[%c384] : memref<1024xui16, 1>, index -> <ui16>
                          %310 = ascvec.cmp_ge %309, %305, %mask_39 : <ui16>, <ui16> -> !ascvec.mask
                          %311 = ascvec.squeeze_store_reg %299, %310 : <ui16>, !ascvec.mask -> <ui16>
                          ascvec.store_unalign %arg35[%c0_35], %311 : memref<256xui16, 1>, <ui16>
                          ascvec.store_unalign_post %arg35[%c0_35] : memref<256xui16, 1>
                          %312 = ascvec.loadalign_brc %arg30[%arg36] : memref<256xi32, 1>, index -> <i32>
                          %313 = ascvec.reinterpret %312 : <i32> -> <ui16>
                          %314 = ascvec.duplicate %313, %mask_39 : <ui16> -> <ui16>
                          ascvec.store_unalign_begin %arg37 : memref<256xui16, 1>
                          %315 = ascvec.loadalign %arg31[%c512] : memref<1024xui16, 1>, index -> <ui16>
                          %316 = ascvec.cmp_ge %315, %314, %mask_39 : <ui16>, <ui16> -> !ascvec.mask
                          %317 = ascvec.squeeze_store_reg %295, %316 : <ui16>, !ascvec.mask -> <ui16>
                          ascvec.store_unalign %arg37[%c0_35], %317 : memref<256xui16, 1>, <ui16>
                          %318 = ascvec.loadalign %arg31[%c640] : memref<1024xui16, 1>, index -> <ui16>
                          %319 = ascvec.cmp_ge %318, %314, %mask_39 : <ui16>, <ui16> -> !ascvec.mask
                          %320 = ascvec.squeeze_store_reg %299, %319 : <ui16>, !ascvec.mask -> <ui16>
                          ascvec.store_unalign %arg37[%c0_35], %320 : memref<256xui16, 1>, <ui16>
                          ascvec.store_unalign_post %arg37[%c0_35] : memref<256xui16, 1>
                          %321 = ascvec.loadalign_brc %arg30[%arg38] : memref<256xi32, 1>, index -> <i32>
                          %322 = ascvec.reinterpret %321 : <i32> -> <ui16>
                          %323 = ascvec.duplicate %322, %mask_39 : <ui16> -> <ui16>
                          ascvec.store_unalign_begin %arg39 : memref<256xui16, 1>
                          %324 = ascvec.loadalign %arg31[%c768] : memref<1024xui16, 1>, index -> <ui16>
                          %325 = ascvec.cmp_ge %324, %323, %mask_39 : <ui16>, <ui16> -> !ascvec.mask
                          %326 = ascvec.squeeze_store_reg %295, %325 : <ui16>, !ascvec.mask -> <ui16>
                          ascvec.store_unalign %arg39[%c0_35], %326 : memref<256xui16, 1>, <ui16>
                          %327 = ascvec.loadalign %arg31[%c896] : memref<1024xui16, 1>, index -> <ui16>
                          %328 = ascvec.cmp_ge %327, %323, %mask_39 : <ui16>, <ui16> -> !ascvec.mask
                          %329 = ascvec.squeeze_store_reg %299, %328 : <ui16>, !ascvec.mask -> <ui16>
                          ascvec.store_unalign %arg39[%c0_35], %329 : memref<256xui16, 1>, <ui16>
                          ascvec.store_unalign_post %arg39[%c0_35] : memref<256xui16, 1>
                        }
                        ascvec.mem_bar vst_vld
                        %c0_44 = arith.constant 0 : index
                        %c1_45 = arith.constant 1 : index
                        scf.for %arg44 = %c0_44 to %c1_45 step %c1_45 {
                          %292 = ascvec.loadalign_brc %arg30[%arg32] : memref<256xi32, 1>, index -> <i32>
                          %293 = ascvec.reinterpret %292 : <i32> -> <ui16>
                          %294 = ascvec.duplicate %293, %mask_39 : <ui16> -> <ui16>
                          %295 = ascvec.loadalign_brc %arg33[%c0_35] : memref<256xui16, 1>, index -> <ui16>
                          %296 = ascvec.reinterpret %295 : <ui16> -> <ui32>
                          %297 = ascvec.and %296, %290, %mask_41 : <ui32>, <ui32> -> <ui32>
                          %298 = ascvec.reinterpret %297 : <ui32> -> <i32>
                          ascvec.storealign_1st %arg30[%arg40], %298 : memref<256xi32, 1>, index, <i32>
                          %299 = ascvec.cmp_eq_scalar %295, %c0_i16, %mask_39 : <ui16>, i16 -> !ascvec.mask
                          %300 = ascvec.sub %295, %289, %mask_39 : <ui16>
                          %301 = ascvec.select %288, %300, %299 : <ui16>
                          %302 = ascvec.add %301, %288, %mask_39 : <ui16>
                          %303 = ascvec.gather %arg31[%c0_35], %302, %mask_39 : memref<1024xui16, 1>, <ui16>, !ascvec.mask -> <ui16>
                          %304 = ascvec.select %288, %303, %299 : <ui16>
                          %305 = ascvec.sub %294, %304, %mask_39 : <ui16>
                          %306 = ascvec.reinterpret %305 : <ui16> -> <ui32>
                          %307 = ascvec.and %306, %291, %mask_41 : <ui32>, <ui32> -> <ui32>
                          %308 = ascvec.reinterpret %307 : <ui32> -> <i32>
                          ascvec.storealign_1st %arg30[%arg32], %308 : memref<256xi32, 1>, index, <i32>
                          %309 = ascvec.loadalign_brc %arg30[%arg34] : memref<256xi32, 1>, index -> <i32>
                          %310 = ascvec.reinterpret %309 : <i32> -> <ui16>
                          %311 = ascvec.duplicate %310, %mask_39 : <ui16> -> <ui16>
                          %312 = ascvec.loadalign_brc %arg35[%c0_35] : memref<256xui16, 1>, index -> <ui16>
                          %313 = ascvec.reinterpret %312 : <ui16> -> <ui32>
                          %314 = ascvec.and %313, %290, %mask_41 : <ui32>, <ui32> -> <ui32>
                          %315 = ascvec.reinterpret %314 : <ui32> -> <i32>
                          ascvec.storealign_1st %arg30[%arg41], %315 : memref<256xi32, 1>, index, <i32>
                          %316 = ascvec.cmp_eq_scalar %312, %c0_i16, %mask_39 : <ui16>, i16 -> !ascvec.mask
                          %317 = ascvec.sub %312, %289, %mask_39 : <ui16>
                          %318 = ascvec.select %288, %317, %316 : <ui16>
                          %319 = ascvec.duplicate_scalar %c256_i16, %mask_39 : i16 -> <ui16>
                          %320 = ascvec.add %318, %319, %mask_39 : <ui16>
                          %321 = ascvec.gather %arg31[%c0_35], %320, %mask_39 : memref<1024xui16, 1>, <ui16>, !ascvec.mask -> <ui16>
                          %322 = ascvec.select %288, %321, %316 : <ui16>
                          %323 = ascvec.sub %311, %322, %mask_39 : <ui16>
                          %324 = ascvec.reinterpret %323 : <ui16> -> <ui32>
                          %325 = ascvec.and %324, %291, %mask_41 : <ui32>, <ui32> -> <ui32>
                          %326 = ascvec.reinterpret %325 : <ui32> -> <i32>
                          ascvec.storealign_1st %arg30[%arg34], %326 : memref<256xi32, 1>, index, <i32>
                          %327 = ascvec.loadalign_brc %arg30[%arg36] : memref<256xi32, 1>, index -> <i32>
                          %328 = ascvec.reinterpret %327 : <i32> -> <ui16>
                          %329 = ascvec.duplicate %328, %mask_39 : <ui16> -> <ui16>
                          %330 = ascvec.loadalign_brc %arg37[%c0_35] : memref<256xui16, 1>, index -> <ui16>
                          %331 = ascvec.reinterpret %330 : <ui16> -> <ui32>
                          %332 = ascvec.and %331, %290, %mask_41 : <ui32>, <ui32> -> <ui32>
                          %333 = ascvec.reinterpret %332 : <ui32> -> <i32>
                          ascvec.storealign_1st %arg30[%arg42], %333 : memref<256xi32, 1>, index, <i32>
                          %334 = ascvec.cmp_eq_scalar %330, %c0_i16, %mask_39 : <ui16>, i16 -> !ascvec.mask
                          %335 = ascvec.sub %330, %289, %mask_39 : <ui16>
                          %336 = ascvec.select %288, %335, %334 : <ui16>
                          %337 = ascvec.duplicate_scalar %c512_i16, %mask_39 : i16 -> <ui16>
                          %338 = ascvec.add %336, %337, %mask_39 : <ui16>
                          %339 = ascvec.gather %arg31[%c0_35], %338, %mask_39 : memref<1024xui16, 1>, <ui16>, !ascvec.mask -> <ui16>
                          %340 = ascvec.select %288, %339, %334 : <ui16>
                          %341 = ascvec.sub %329, %340, %mask_39 : <ui16>
                          %342 = ascvec.reinterpret %341 : <ui16> -> <ui32>
                          %343 = ascvec.and %342, %291, %mask_41 : <ui32>, <ui32> -> <ui32>
                          %344 = ascvec.reinterpret %343 : <ui32> -> <i32>
                          ascvec.storealign_1st %arg30[%arg36], %344 : memref<256xi32, 1>, index, <i32>
                          %345 = ascvec.loadalign_brc %arg30[%arg38] : memref<256xi32, 1>, index -> <i32>
                          %346 = ascvec.reinterpret %345 : <i32> -> <ui16>
                          %347 = ascvec.duplicate %346, %mask_39 : <ui16> -> <ui16>
                          %348 = ascvec.loadalign_brc %arg39[%c0_35] : memref<256xui16, 1>, index -> <ui16>
                          %349 = ascvec.reinterpret %348 : <ui16> -> <ui32>
                          %350 = ascvec.and %349, %290, %mask_41 : <ui32>, <ui32> -> <ui32>
                          %351 = ascvec.reinterpret %350 : <ui32> -> <i32>
                          ascvec.storealign_1st %arg30[%arg43], %351 : memref<256xi32, 1>, index, <i32>
                          %352 = ascvec.cmp_eq_scalar %348, %c0_i16, %mask_39 : <ui16>, i16 -> !ascvec.mask
                          %353 = ascvec.sub %348, %289, %mask_39 : <ui16>
                          %354 = ascvec.select %288, %353, %352 : <ui16>
                          %355 = ascvec.duplicate_scalar %c768_i16, %mask_39 : i16 -> <ui16>
                          %356 = ascvec.add %354, %355, %mask_39 : <ui16>
                          %357 = ascvec.gather %arg31[%c0_35], %356, %mask_39 : memref<1024xui16, 1>, <ui16>, !ascvec.mask -> <ui16>
                          %358 = ascvec.select %288, %357, %352 : <ui16>
                          %359 = ascvec.sub %347, %358, %mask_39 : <ui16>
                          %360 = ascvec.reinterpret %359 : <ui16> -> <ui32>
                          %361 = ascvec.and %360, %291, %mask_41 : <ui32>, <ui32> -> <ui32>
                          %362 = ascvec.reinterpret %361 : <ui32> -> <i32>
                          ascvec.storealign_1st %arg30[%arg38], %362 : memref<256xi32, 1>, index, <i32>
                        }
                      } {cannir.vf_mode = "simd"}
                      %248 = arith.addi %212, %c4_i32 overflow<nsw> : i32
                      %249 = arith.index_cast %248 : i32 to index
                      %250 = arith.addi %217, %c4_i32 overflow<nsw> : i32
                      %251 = arith.index_cast %250 : i32 to index
                      %252 = arith.addi %222, %c4_i32 overflow<nsw> : i32
                      %253 = arith.index_cast %252 : i32 to index
                      %254 = arith.addi %227, %c4_i32 overflow<nsw> : i32
                      %255 = arith.index_cast %254 : i32 to index
                      ascvec.vf(%196, %alloc_11, %alloc_14, %alloc_19, %216, %alloc_20, %221, %alloc_21, %226, %alloc_22, %231, %alloc_23, %249, %251, %253, %255 : i32, memref<8192xf32, 1>, memref<256xi32, 1>, memref<1024xui16, 1>, index, memref<256xui16, 1>, index, memref<256xui16, 1>, index, memref<256xui16, 1>, index, memref<256xui16, 1>, index, index, index, index) {
                      ^bb0(%arg28: i32, %arg29: memref<8192xf32, 1>, %arg30: memref<256xi32, 1>, %arg31: memref<1024xui16, 1>, %arg32: index, %arg33: memref<256xui16, 1>, %arg34: index, %arg35: memref<256xui16, 1>, %arg36: index, %arg37: memref<256xui16, 1>, %arg38: index, %arg39: memref<256xui16, 1>, %arg40: index, %arg41: index, %arg42: index, %arg43: index):
                        %c896 = arith.constant 896 : index
                        %c768 = arith.constant 768 : index
                        %c640 = arith.constant 640 : index
                        %c512 = arith.constant 512 : index
                        %c384 = arith.constant 384 : index
                        %c256 = arith.constant 256 : index
                        %c128 = arith.constant 128 : index
                        %c768_i16 = arith.constant 768 : i16
                        %c512_i16 = arith.constant 512 : i16
                        %c256_i16 = arith.constant 256 : i16
                        %c128_i16 = arith.constant 128 : i16
                        %c4_i32_30 = arith.constant 4 : i32
                        %c0_i32_31 = arith.constant 0 : i32
                        %c3_i32_32 = arith.constant 3 : i32
                        %c2_i32_33 = arith.constant 2 : i32
                        %c1_i32_34 = arith.constant 1 : i32
                        %c8_i32_35 = arith.constant 8 : i32
                        %c0_36 = arith.constant 0 : index
                        %c65535_i32 = arith.constant 65535 : i32
                        %c255_i32 = arith.constant 255 : i32
                        %c1_i16 = arith.constant 1 : i16
                        %c0_i16 = arith.constant 0 : i16
                        %c64_i32_37 = arith.constant 64 : i32
                        %c128_i32_38 = arith.constant 128 : i32
                        %c256_i32_39 = arith.constant 256 : i32
                        %mask, %remaining_out = ascvec.update_mask %c256_i32_39 elem_bits 8 : !ascvec.mask
                        %mask_40, %remaining_out_41 = ascvec.update_mask %c128_i32_38 elem_bits 16 : !ascvec.mask
                        %mask_42, %remaining_out_43 = ascvec.update_mask %c64_i32_37 : !ascvec.mask
                        %288 = ascvec.duplicate_scalar %c0_i16, %mask_40 : i16 -> <ui16>
                        %289 = ascvec.duplicate_scalar %c1_i16, %mask_40 : i16 -> <ui16>
                        %290 = ascvec.duplicate_scalar %c255_i32, %mask_42 : i32 -> <ui32>
                        %291 = ascvec.duplicate_scalar %c65535_i32, %mask_42 : i32 -> <ui32>
                        scf.for %arg44 = %c0_i32_31 to %c4_i32_30 step %c1_i32_34  : i32 {
                          %292 = arith.addi %arg28, %arg44 : i32
                          %293 = arith.muli %292, %c256_i32_39 : i32
                          %294 = arith.index_cast %293 : i32 to index
                          %result0, %result1 = ascvec.loadalign_deintlv %arg29[%294] {dintlv_width = "b16"} : memref<8192xf32, 1>, index -> <ui16>, <ui16>
                          %295 = arith.addi %293, %c128_i32_38 : i32
                          %296 = arith.index_cast %295 : i32 to index
                          %result0_47, %result1_48 = ascvec.loadalign_deintlv %arg29[%296] {dintlv_width = "b16"} : memref<8192xf32, 1>, index -> <ui16>, <ui16>
                          %297 = ascvec.reinterpret %result1 : <ui16> -> <ui8>
                          %298 = ascvec.reinterpret %result1_48 : <ui16> -> <ui8>
                          %result0_49, %result1_50 = ascvec.deintlv %297, %298 : <ui8>, <ui8> -> <ui8>, <ui8>
                          %299 = ascvec.reinterpret %result0 : <ui16> -> <ui8>
                          %300 = ascvec.reinterpret %result0_47 : <ui16> -> <ui8>
                          %result0_51, %result1_52 = ascvec.deintlv %299, %300 : <ui8>, <ui8> -> <ui8>, <ui8>
                          %301 = arith.muli %292, %c8_i32_35 : i32
                          %302 = arith.addi %301, %c1_i32_34 : i32
                          %303 = arith.index_cast %302 : i32 to index
                          %304 = ascvec.loadalign_brc %arg30[%303] {brc_width = "b8"} : memref<256xi32, 1>, index -> <ui32>
                          %305 = ascvec.reinterpret %304 : <ui32> -> <ui8>
                          %306 = ascvec.cmp_eq %result1_50, %305, %mask : <ui8>, <ui8> -> !ascvec.mask
                          %307 = arith.addi %301, %c2_i32_33 : i32
                          %308 = arith.index_cast %307 : i32 to index
                          %309 = ascvec.loadalign_brc %arg30[%308] {brc_width = "b8"} : memref<256xi32, 1>, index -> <ui32>
                          %310 = ascvec.reinterpret %309 : <ui32> -> <ui8>
                          %311 = ascvec.cmp_eq %result0_49, %310, %306 : <ui8>, <ui8> -> !ascvec.mask
                          %312 = arith.addi %301, %c3_i32_32 : i32
                          %313 = arith.index_cast %312 : i32 to index
                          %314 = ascvec.loadalign_brc %arg30[%313] {brc_width = "b8"} : memref<256xi32, 1>, index -> <ui32>
                          %315 = ascvec.reinterpret %314 : <ui32> -> <ui8>
                          %316 = ascvec.cmp_eq %result1_52, %315, %311 : <ui8>, <ui8> -> !ascvec.mask
                          %317 = ascvec.cumulative_histogram %288, %result0_51, %316 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          %318 = ascvec.cumulative_histogram %288, %result0_51, %316 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          %319 = arith.muli %arg44, %c256_i32_39 overflow<nsw> : i32
                          %320 = arith.index_cast %319 : i32 to index
                          ascvec.storealign %arg31[%320], %317, %mask_40 : memref<1024xui16, 1>, index, <ui16>
                          %321 = arith.addi %319, %c128_i32_38 overflow<nsw> : i32
                          %322 = arith.index_cast %321 : i32 to index
                          ascvec.storealign %arg31[%322], %318, %mask_40 : memref<1024xui16, 1>, index, <ui16>
                        }
                        ascvec.mem_bar vst_vld
                        %c0_44 = arith.constant 0 : index
                        %c1 = arith.constant 1 : index
                        scf.for %arg44 = %c0_44 to %c1 step %c1 {
                          %292 = ascvec.loadalign_brc %arg30[%arg32] : memref<256xi32, 1>, index -> <i32>
                          %293 = ascvec.reinterpret %292 : <i32> -> <ui16>
                          %294 = ascvec.duplicate %293, %mask_40 : <ui16> -> <ui16>
                          ascvec.store_unalign_begin %arg33 : memref<256xui16, 1>
                          %295 = ascvec.arange %c0_i16 : i16 -> <ui16>
                          %296 = ascvec.loadalign %arg31[%c0_36] : memref<1024xui16, 1>, index -> <ui16>
                          %297 = ascvec.cmp_ge %296, %294, %mask_40 : <ui16>, <ui16> -> !ascvec.mask
                          %298 = ascvec.squeeze_store_reg %295, %297 : <ui16>, !ascvec.mask -> <ui16>
                          ascvec.store_unalign %arg33[%c0_36], %298 : memref<256xui16, 1>, <ui16>
                          %299 = ascvec.arange %c128_i16 : i16 -> <ui16>
                          %300 = ascvec.loadalign %arg31[%c128] : memref<1024xui16, 1>, index -> <ui16>
                          %301 = ascvec.cmp_ge %300, %294, %mask_40 : <ui16>, <ui16> -> !ascvec.mask
                          %302 = ascvec.squeeze_store_reg %299, %301 : <ui16>, !ascvec.mask -> <ui16>
                          ascvec.store_unalign %arg33[%c0_36], %302 : memref<256xui16, 1>, <ui16>
                          ascvec.store_unalign_post %arg33[%c0_36] : memref<256xui16, 1>
                          %303 = ascvec.loadalign_brc %arg30[%arg34] : memref<256xi32, 1>, index -> <i32>
                          %304 = ascvec.reinterpret %303 : <i32> -> <ui16>
                          %305 = ascvec.duplicate %304, %mask_40 : <ui16> -> <ui16>
                          ascvec.store_unalign_begin %arg35 : memref<256xui16, 1>
                          %306 = ascvec.loadalign %arg31[%c256] : memref<1024xui16, 1>, index -> <ui16>
                          %307 = ascvec.cmp_ge %306, %305, %mask_40 : <ui16>, <ui16> -> !ascvec.mask
                          %308 = ascvec.squeeze_store_reg %295, %307 : <ui16>, !ascvec.mask -> <ui16>
                          ascvec.store_unalign %arg35[%c0_36], %308 : memref<256xui16, 1>, <ui16>
                          %309 = ascvec.loadalign %arg31[%c384] : memref<1024xui16, 1>, index -> <ui16>
                          %310 = ascvec.cmp_ge %309, %305, %mask_40 : <ui16>, <ui16> -> !ascvec.mask
                          %311 = ascvec.squeeze_store_reg %299, %310 : <ui16>, !ascvec.mask -> <ui16>
                          ascvec.store_unalign %arg35[%c0_36], %311 : memref<256xui16, 1>, <ui16>
                          ascvec.store_unalign_post %arg35[%c0_36] : memref<256xui16, 1>
                          %312 = ascvec.loadalign_brc %arg30[%arg36] : memref<256xi32, 1>, index -> <i32>
                          %313 = ascvec.reinterpret %312 : <i32> -> <ui16>
                          %314 = ascvec.duplicate %313, %mask_40 : <ui16> -> <ui16>
                          ascvec.store_unalign_begin %arg37 : memref<256xui16, 1>
                          %315 = ascvec.loadalign %arg31[%c512] : memref<1024xui16, 1>, index -> <ui16>
                          %316 = ascvec.cmp_ge %315, %314, %mask_40 : <ui16>, <ui16> -> !ascvec.mask
                          %317 = ascvec.squeeze_store_reg %295, %316 : <ui16>, !ascvec.mask -> <ui16>
                          ascvec.store_unalign %arg37[%c0_36], %317 : memref<256xui16, 1>, <ui16>
                          %318 = ascvec.loadalign %arg31[%c640] : memref<1024xui16, 1>, index -> <ui16>
                          %319 = ascvec.cmp_ge %318, %314, %mask_40 : <ui16>, <ui16> -> !ascvec.mask
                          %320 = ascvec.squeeze_store_reg %299, %319 : <ui16>, !ascvec.mask -> <ui16>
                          ascvec.store_unalign %arg37[%c0_36], %320 : memref<256xui16, 1>, <ui16>
                          ascvec.store_unalign_post %arg37[%c0_36] : memref<256xui16, 1>
                          %321 = ascvec.loadalign_brc %arg30[%arg38] : memref<256xi32, 1>, index -> <i32>
                          %322 = ascvec.reinterpret %321 : <i32> -> <ui16>
                          %323 = ascvec.duplicate %322, %mask_40 : <ui16> -> <ui16>
                          ascvec.store_unalign_begin %arg39 : memref<256xui16, 1>
                          %324 = ascvec.loadalign %arg31[%c768] : memref<1024xui16, 1>, index -> <ui16>
                          %325 = ascvec.cmp_ge %324, %323, %mask_40 : <ui16>, <ui16> -> !ascvec.mask
                          %326 = ascvec.squeeze_store_reg %295, %325 : <ui16>, !ascvec.mask -> <ui16>
                          ascvec.store_unalign %arg39[%c0_36], %326 : memref<256xui16, 1>, <ui16>
                          %327 = ascvec.loadalign %arg31[%c896] : memref<1024xui16, 1>, index -> <ui16>
                          %328 = ascvec.cmp_ge %327, %323, %mask_40 : <ui16>, <ui16> -> !ascvec.mask
                          %329 = ascvec.squeeze_store_reg %299, %328 : <ui16>, !ascvec.mask -> <ui16>
                          ascvec.store_unalign %arg39[%c0_36], %329 : memref<256xui16, 1>, <ui16>
                          ascvec.store_unalign_post %arg39[%c0_36] : memref<256xui16, 1>
                        }
                        ascvec.mem_bar vst_vld
                        %c0_45 = arith.constant 0 : index
                        %c1_46 = arith.constant 1 : index
                        scf.for %arg44 = %c0_45 to %c1_46 step %c1_46 {
                          %292 = ascvec.loadalign_brc %arg30[%arg32] : memref<256xi32, 1>, index -> <i32>
                          %293 = ascvec.reinterpret %292 : <i32> -> <ui16>
                          %294 = ascvec.duplicate %293, %mask_40 : <ui16> -> <ui16>
                          %295 = ascvec.loadalign_brc %arg33[%c0_36] : memref<256xui16, 1>, index -> <ui16>
                          %296 = ascvec.reinterpret %295 : <ui16> -> <ui32>
                          %297 = ascvec.and %296, %290, %mask_42 : <ui32>, <ui32> -> <ui32>
                          %298 = ascvec.reinterpret %297 : <ui32> -> <i32>
                          ascvec.storealign_1st %arg30[%arg40], %298 : memref<256xi32, 1>, index, <i32>
                          %299 = ascvec.cmp_eq_scalar %295, %c0_i16, %mask_40 : <ui16>, i16 -> !ascvec.mask
                          %300 = ascvec.sub %295, %289, %mask_40 : <ui16>
                          %301 = ascvec.select %288, %300, %299 : <ui16>
                          %302 = ascvec.add %301, %288, %mask_40 : <ui16>
                          %303 = ascvec.gather %arg31[%c0_36], %302, %mask_40 : memref<1024xui16, 1>, <ui16>, !ascvec.mask -> <ui16>
                          %304 = ascvec.select %288, %303, %299 : <ui16>
                          %305 = ascvec.sub %294, %304, %mask_40 : <ui16>
                          %306 = ascvec.reinterpret %305 : <ui16> -> <ui32>
                          %307 = ascvec.and %306, %291, %mask_42 : <ui32>, <ui32> -> <ui32>
                          %308 = ascvec.reinterpret %307 : <ui32> -> <i32>
                          ascvec.storealign_1st %arg30[%arg32], %308 : memref<256xi32, 1>, index, <i32>
                          %309 = ascvec.loadalign_brc %arg30[%arg34] : memref<256xi32, 1>, index -> <i32>
                          %310 = ascvec.reinterpret %309 : <i32> -> <ui16>
                          %311 = ascvec.duplicate %310, %mask_40 : <ui16> -> <ui16>
                          %312 = ascvec.loadalign_brc %arg35[%c0_36] : memref<256xui16, 1>, index -> <ui16>
                          %313 = ascvec.reinterpret %312 : <ui16> -> <ui32>
                          %314 = ascvec.and %313, %290, %mask_42 : <ui32>, <ui32> -> <ui32>
                          %315 = ascvec.reinterpret %314 : <ui32> -> <i32>
                          ascvec.storealign_1st %arg30[%arg41], %315 : memref<256xi32, 1>, index, <i32>
                          %316 = ascvec.cmp_eq_scalar %312, %c0_i16, %mask_40 : <ui16>, i16 -> !ascvec.mask
                          %317 = ascvec.sub %312, %289, %mask_40 : <ui16>
                          %318 = ascvec.select %288, %317, %316 : <ui16>
                          %319 = ascvec.duplicate_scalar %c256_i16, %mask_40 : i16 -> <ui16>
                          %320 = ascvec.add %318, %319, %mask_40 : <ui16>
                          %321 = ascvec.gather %arg31[%c0_36], %320, %mask_40 : memref<1024xui16, 1>, <ui16>, !ascvec.mask -> <ui16>
                          %322 = ascvec.select %288, %321, %316 : <ui16>
                          %323 = ascvec.sub %311, %322, %mask_40 : <ui16>
                          %324 = ascvec.reinterpret %323 : <ui16> -> <ui32>
                          %325 = ascvec.and %324, %291, %mask_42 : <ui32>, <ui32> -> <ui32>
                          %326 = ascvec.reinterpret %325 : <ui32> -> <i32>
                          ascvec.storealign_1st %arg30[%arg34], %326 : memref<256xi32, 1>, index, <i32>
                          %327 = ascvec.loadalign_brc %arg30[%arg36] : memref<256xi32, 1>, index -> <i32>
                          %328 = ascvec.reinterpret %327 : <i32> -> <ui16>
                          %329 = ascvec.duplicate %328, %mask_40 : <ui16> -> <ui16>
                          %330 = ascvec.loadalign_brc %arg37[%c0_36] : memref<256xui16, 1>, index -> <ui16>
                          %331 = ascvec.reinterpret %330 : <ui16> -> <ui32>
                          %332 = ascvec.and %331, %290, %mask_42 : <ui32>, <ui32> -> <ui32>
                          %333 = ascvec.reinterpret %332 : <ui32> -> <i32>
                          ascvec.storealign_1st %arg30[%arg42], %333 : memref<256xi32, 1>, index, <i32>
                          %334 = ascvec.cmp_eq_scalar %330, %c0_i16, %mask_40 : <ui16>, i16 -> !ascvec.mask
                          %335 = ascvec.sub %330, %289, %mask_40 : <ui16>
                          %336 = ascvec.select %288, %335, %334 : <ui16>
                          %337 = ascvec.duplicate_scalar %c512_i16, %mask_40 : i16 -> <ui16>
                          %338 = ascvec.add %336, %337, %mask_40 : <ui16>
                          %339 = ascvec.gather %arg31[%c0_36], %338, %mask_40 : memref<1024xui16, 1>, <ui16>, !ascvec.mask -> <ui16>
                          %340 = ascvec.select %288, %339, %334 : <ui16>
                          %341 = ascvec.sub %329, %340, %mask_40 : <ui16>
                          %342 = ascvec.reinterpret %341 : <ui16> -> <ui32>
                          %343 = ascvec.and %342, %291, %mask_42 : <ui32>, <ui32> -> <ui32>
                          %344 = ascvec.reinterpret %343 : <ui32> -> <i32>
                          ascvec.storealign_1st %arg30[%arg36], %344 : memref<256xi32, 1>, index, <i32>
                          %345 = ascvec.loadalign_brc %arg30[%arg38] : memref<256xi32, 1>, index -> <i32>
                          %346 = ascvec.reinterpret %345 : <i32> -> <ui16>
                          %347 = ascvec.duplicate %346, %mask_40 : <ui16> -> <ui16>
                          %348 = ascvec.loadalign_brc %arg39[%c0_36] : memref<256xui16, 1>, index -> <ui16>
                          %349 = ascvec.reinterpret %348 : <ui16> -> <ui32>
                          %350 = ascvec.and %349, %290, %mask_42 : <ui32>, <ui32> -> <ui32>
                          %351 = ascvec.reinterpret %350 : <ui32> -> <i32>
                          ascvec.storealign_1st %arg30[%arg43], %351 : memref<256xi32, 1>, index, <i32>
                          %352 = ascvec.cmp_eq_scalar %348, %c0_i16, %mask_40 : <ui16>, i16 -> !ascvec.mask
                          %353 = ascvec.sub %348, %289, %mask_40 : <ui16>
                          %354 = ascvec.select %288, %353, %352 : <ui16>
                          %355 = ascvec.duplicate_scalar %c768_i16, %mask_40 : i16 -> <ui16>
                          %356 = ascvec.add %354, %355, %mask_40 : <ui16>
                          %357 = ascvec.gather %arg31[%c0_36], %356, %mask_40 : memref<1024xui16, 1>, <ui16>, !ascvec.mask -> <ui16>
                          %358 = ascvec.select %288, %357, %352 : <ui16>
                          %359 = ascvec.sub %347, %358, %mask_40 : <ui16>
                          %360 = ascvec.reinterpret %359 : <ui16> -> <ui32>
                          %361 = ascvec.and %360, %291, %mask_42 : <ui32>, <ui32> -> <ui32>
                          %362 = ascvec.reinterpret %361 : <ui32> -> <i32>
                          ascvec.storealign_1st %arg30[%arg38], %362 : memref<256xi32, 1>, index, <i32>
                        }
                      } {cannir.vf_mode = "simd"}
                      %256 = arith.muli %196, %c256_i32 overflow<nsw> : i32
                      %257 = arith.index_cast %256 : i32 to index
                      %258 = arith.addi %256, %c64_i32 overflow<nsw> : i32
                      %259 = arith.index_cast %258 : i32 to index
                      %260 = arith.addi %256, %c128_i32 overflow<nsw> : i32
                      %261 = arith.index_cast %260 : i32 to index
                      %262 = arith.addi %256, %c192_i32 overflow<nsw> : i32
                      %263 = arith.index_cast %262 : i32 to index
                      %264 = arith.muli %200, %c256_i32 overflow<nsw> : i32
                      %265 = arith.index_cast %264 : i32 to index
                      %266 = arith.addi %264, %c64_i32 overflow<nsw> : i32
                      %267 = arith.index_cast %266 : i32 to index
                      %268 = arith.addi %264, %c128_i32 overflow<nsw> : i32
                      %269 = arith.index_cast %268 : i32 to index
                      %270 = arith.addi %264, %c192_i32 overflow<nsw> : i32
                      %271 = arith.index_cast %270 : i32 to index
                      %272 = arith.muli %204, %c256_i32 overflow<nsw> : i32
                      %273 = arith.index_cast %272 : i32 to index
                      %274 = arith.addi %272, %c64_i32 overflow<nsw> : i32
                      %275 = arith.index_cast %274 : i32 to index
                      %276 = arith.addi %272, %c128_i32 overflow<nsw> : i32
                      %277 = arith.index_cast %276 : i32 to index
                      %278 = arith.addi %272, %c192_i32 overflow<nsw> : i32
                      %279 = arith.index_cast %278 : i32 to index
                      %280 = arith.muli %208, %c256_i32 overflow<nsw> : i32
                      %281 = arith.index_cast %280 : i32 to index
                      %282 = arith.addi %280, %c64_i32 overflow<nsw> : i32
                      %283 = arith.index_cast %282 : i32 to index
                      %284 = arith.addi %280, %c128_i32 overflow<nsw> : i32
                      %285 = arith.index_cast %284 : i32 to index
                      %286 = arith.addi %280, %c192_i32 overflow<nsw> : i32
                      %287 = arith.index_cast %286 : i32 to index
                      ascvec.vf(%alloc_14, %214, %233, %241, %249, %alloc_24, %alloc_11, %257, %259, %261, %263, %219, %235, %243, %251, %alloc_25, %265, %267, %269, %271, %224, %237, %245, %253, %alloc_26, %273, %275, %277, %279, %229, %239, %247, %255, %alloc_27, %281, %283, %285, %287, %alloc_13, %199, %256, %alloc_28, %alloc_29, %203, %264, %207, %272, %211, %280, %196, %alloc_17, %alloc_18 : memref<256xi32, 1>, index, index, index, index, memref<528xi32, 1>, memref<8192xf32, 1>, index, index, index, index, index, index, index, index, memref<528xi32, 1>, index, index, index, index, index, index, index, index, memref<528xi32, 1>, index, index, index, index, index, index, index, index, memref<528xi32, 1>, index, index, index, index, memref<512xi32, 1>, index, i32, memref<1024xui32, 1>, memref<1024xui32, 1>, index, i32, index, i32, index, i32, i32, memref<8192xui32, 1>, memref<8192xui32, 1>) {
                      ^bb0(%arg28: memref<256xi32, 1>, %arg29: index, %arg30: index, %arg31: index, %arg32: index, %arg33: memref<528xi32, 1>, %arg34: memref<8192xf32, 1>, %arg35: index, %arg36: index, %arg37: index, %arg38: index, %arg39: index, %arg40: index, %arg41: index, %arg42: index, %arg43: memref<528xi32, 1>, %arg44: index, %arg45: index, %arg46: index, %arg47: index, %arg48: index, %arg49: index, %arg50: index, %arg51: index, %arg52: memref<528xi32, 1>, %arg53: index, %arg54: index, %arg55: index, %arg56: index, %arg57: index, %arg58: index, %arg59: index, %arg60: index, %arg61: memref<528xi32, 1>, %arg62: index, %arg63: index, %arg64: index, %arg65: index, %arg66: memref<512xi32, 1>, %arg67: index, %arg68: i32, %arg69: memref<1024xui32, 1>, %arg70: memref<1024xui32, 1>, %arg71: index, %arg72: i32, %arg73: index, %arg74: i32, %arg75: index, %arg76: i32, %arg77: i32, %arg78: memref<8192xui32, 1>, %arg79: memref<8192xui32, 1>):
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
                        %c1_i32_30 = arith.constant 1 : i32
                        %c4_i32_31 = arith.constant 4 : i32
                        %c192_i32_32 = arith.constant 192 : i32
                        %c128_i32_33 = arith.constant 128 : i32
                        %c-256_i32 = arith.constant -256 : i32
                        %c448_i32 = arith.constant 448 : i32
                        %c384_i32 = arith.constant 384 : i32
                        %c320_i32 = arith.constant 320 : i32
                        %c256_i32_34 = arith.constant 256 : i32
                        %c0_35 = arith.constant 0 : index
                        %c255_i32 = arith.constant 255 : i32
                        %c0_i32_36 = arith.constant 0 : i32
                        %c64_i32_37 = arith.constant 64 : i32
                        %mask, %remaining_out = ascvec.update_mask %c64_i32_37 : !ascvec.mask
                        %288 = ascvec.duplicate_scalar %c0_i32_36, %mask : i32 -> <i32>
                        %289 = ascvec.duplicate_scalar %c0_i32_36, %mask : i32 -> <ui32>
                        %290 = ascvec.duplicate_scalar %c255_i32, %mask : i32 -> <ui32>
                        %291 = ascvec.loadalign_brc %arg28[%arg29] {brc_width = "b8"} : memref<256xi32, 1>, index -> <ui32>
                        %292 = ascvec.and %291, %290, %mask : <ui32>, <ui32> -> <ui32>
                        %293 = ascvec.loadalign_brc %arg28[%arg30] {brc_width = "b8"} : memref<256xi32, 1>, index -> <ui32>
                        %294 = ascvec.and %293, %290, %mask : <ui32>, <ui32> -> <ui32>
                        %295 = ascvec.loadalign_brc %arg28[%arg31] {brc_width = "b8"} : memref<256xi32, 1>, index -> <ui32>
                        %296 = ascvec.and %295, %290, %mask : <ui32>, <ui32> -> <ui32>
                        %297 = ascvec.loadalign_brc %arg28[%arg32] {brc_width = "b8"} : memref<256xi32, 1>, index -> <ui32>
                        %298 = ascvec.and %297, %290, %mask : <ui32>, <ui32> -> <ui32>
                        %299 = ascvec.shl %292, 24, %mask : <ui32> -> <ui32>
                        %300 = ascvec.shl %294, 16, %mask : <ui32> -> <ui32>
                        %301 = ascvec.add %299, %300, %mask : <ui32>
                        %302 = ascvec.shl %296, 8, %mask : <ui32> -> <ui32>
                        %303 = ascvec.add %301, %302, %mask : <ui32>
                        %304 = ascvec.add %303, %298, %mask : <ui32>
                        ascvec.store_unalign_begin %arg33 : memref<528xi32, 1>
                        %305 = ascvec.loadalign %arg34[%arg35] : memref<8192xf32, 1>, index -> <f32>
                        %306 = ascvec.reinterpret %305 : <f32> -> <ui32>
                        %307 = ascvec.arange %c256_i32_34 : i32 -> <i32>
                        %308 = ascvec.cmp_gt %306, %304, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %309 = ascvec.squeeze_store_reg %307, %308 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg33[%c0_35], %309 : memref<528xi32, 1>, <i32>
                        %310 = ascvec.loadalign %arg34[%arg36] : memref<8192xf32, 1>, index -> <f32>
                        %311 = ascvec.reinterpret %310 : <f32> -> <ui32>
                        %312 = ascvec.arange %c320_i32 : i32 -> <i32>
                        %313 = ascvec.cmp_gt %311, %304, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %314 = ascvec.squeeze_store_reg %312, %313 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg33[%c0_35], %314 : memref<528xi32, 1>, <i32>
                        %315 = ascvec.loadalign %arg34[%arg37] : memref<8192xf32, 1>, index -> <f32>
                        %316 = ascvec.reinterpret %315 : <f32> -> <ui32>
                        %317 = ascvec.arange %c384_i32 : i32 -> <i32>
                        %318 = ascvec.cmp_gt %316, %304, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %319 = ascvec.squeeze_store_reg %317, %318 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg33[%c0_35], %319 : memref<528xi32, 1>, <i32>
                        %320 = ascvec.loadalign %arg34[%arg38] : memref<8192xf32, 1>, index -> <f32>
                        %321 = ascvec.reinterpret %320 : <f32> -> <ui32>
                        %322 = ascvec.arange %c448_i32 : i32 -> <i32>
                        %323 = ascvec.cmp_gt %321, %304, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %324 = ascvec.squeeze_store_reg %322, %323 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg33[%c0_35], %324 : memref<528xi32, 1>, <i32>
                        %325 = ascvec.loadalign %arg34[%arg35] : memref<8192xf32, 1>, index -> <f32>
                        %326 = ascvec.reinterpret %325 : <f32> -> <ui32>
                        %327 = ascvec.cmp_eq %326, %304, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %328 = ascvec.squeeze_store_reg %307, %327 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg33[%c0_35], %328 : memref<528xi32, 1>, <i32>
                        %329 = ascvec.loadalign %arg34[%arg36] : memref<8192xf32, 1>, index -> <f32>
                        %330 = ascvec.reinterpret %329 : <f32> -> <ui32>
                        %331 = ascvec.cmp_eq %330, %304, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %332 = ascvec.squeeze_store_reg %312, %331 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg33[%c0_35], %332 : memref<528xi32, 1>, <i32>
                        %333 = ascvec.loadalign %arg34[%arg37] : memref<8192xf32, 1>, index -> <f32>
                        %334 = ascvec.reinterpret %333 : <f32> -> <ui32>
                        %335 = ascvec.cmp_eq %334, %304, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %336 = ascvec.squeeze_store_reg %317, %335 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg33[%c0_35], %336 : memref<528xi32, 1>, <i32>
                        %337 = ascvec.loadalign %arg34[%arg38] : memref<8192xf32, 1>, index -> <f32>
                        %338 = ascvec.reinterpret %337 : <f32> -> <ui32>
                        %339 = ascvec.cmp_eq %338, %304, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %340 = ascvec.squeeze_store_reg %322, %339 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg33[%c0_35], %340 : memref<528xi32, 1>, <i32>
                        ascvec.store_unalign_post %arg33[%c0_35] : memref<528xi32, 1>
                        %341 = ascvec.loadalign_brc %arg28[%arg39] {brc_width = "b8"} : memref<256xi32, 1>, index -> <ui32>
                        %342 = ascvec.and %341, %290, %mask : <ui32>, <ui32> -> <ui32>
                        %343 = ascvec.loadalign_brc %arg28[%arg40] {brc_width = "b8"} : memref<256xi32, 1>, index -> <ui32>
                        %344 = ascvec.and %343, %290, %mask : <ui32>, <ui32> -> <ui32>
                        %345 = ascvec.loadalign_brc %arg28[%arg41] {brc_width = "b8"} : memref<256xi32, 1>, index -> <ui32>
                        %346 = ascvec.and %345, %290, %mask : <ui32>, <ui32> -> <ui32>
                        %347 = ascvec.loadalign_brc %arg28[%arg42] {brc_width = "b8"} : memref<256xi32, 1>, index -> <ui32>
                        %348 = ascvec.and %347, %290, %mask : <ui32>, <ui32> -> <ui32>
                        %349 = ascvec.shl %342, 24, %mask : <ui32> -> <ui32>
                        %350 = ascvec.shl %344, 16, %mask : <ui32> -> <ui32>
                        %351 = ascvec.add %349, %350, %mask : <ui32>
                        %352 = ascvec.shl %346, 8, %mask : <ui32> -> <ui32>
                        %353 = ascvec.add %351, %352, %mask : <ui32>
                        %354 = ascvec.add %353, %348, %mask : <ui32>
                        ascvec.store_unalign_begin %arg43 : memref<528xi32, 1>
                        %355 = ascvec.loadalign %arg34[%arg44] : memref<8192xf32, 1>, index -> <f32>
                        %356 = ascvec.reinterpret %355 : <f32> -> <ui32>
                        %357 = ascvec.cmp_gt %356, %354, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %358 = ascvec.squeeze_store_reg %307, %357 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg43[%c0_35], %358 : memref<528xi32, 1>, <i32>
                        %359 = ascvec.loadalign %arg34[%arg45] : memref<8192xf32, 1>, index -> <f32>
                        %360 = ascvec.reinterpret %359 : <f32> -> <ui32>
                        %361 = ascvec.cmp_gt %360, %354, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %362 = ascvec.squeeze_store_reg %312, %361 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg43[%c0_35], %362 : memref<528xi32, 1>, <i32>
                        %363 = ascvec.loadalign %arg34[%arg46] : memref<8192xf32, 1>, index -> <f32>
                        %364 = ascvec.reinterpret %363 : <f32> -> <ui32>
                        %365 = ascvec.cmp_gt %364, %354, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %366 = ascvec.squeeze_store_reg %317, %365 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg43[%c0_35], %366 : memref<528xi32, 1>, <i32>
                        %367 = ascvec.loadalign %arg34[%arg47] : memref<8192xf32, 1>, index -> <f32>
                        %368 = ascvec.reinterpret %367 : <f32> -> <ui32>
                        %369 = ascvec.cmp_gt %368, %354, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %370 = ascvec.squeeze_store_reg %322, %369 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg43[%c0_35], %370 : memref<528xi32, 1>, <i32>
                        %371 = ascvec.loadalign %arg34[%arg44] : memref<8192xf32, 1>, index -> <f32>
                        %372 = ascvec.reinterpret %371 : <f32> -> <ui32>
                        %373 = ascvec.cmp_eq %372, %354, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %374 = ascvec.squeeze_store_reg %307, %373 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg43[%c0_35], %374 : memref<528xi32, 1>, <i32>
                        %375 = ascvec.loadalign %arg34[%arg45] : memref<8192xf32, 1>, index -> <f32>
                        %376 = ascvec.reinterpret %375 : <f32> -> <ui32>
                        %377 = ascvec.cmp_eq %376, %354, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %378 = ascvec.squeeze_store_reg %312, %377 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg43[%c0_35], %378 : memref<528xi32, 1>, <i32>
                        %379 = ascvec.loadalign %arg34[%arg46] : memref<8192xf32, 1>, index -> <f32>
                        %380 = ascvec.reinterpret %379 : <f32> -> <ui32>
                        %381 = ascvec.cmp_eq %380, %354, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %382 = ascvec.squeeze_store_reg %317, %381 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg43[%c0_35], %382 : memref<528xi32, 1>, <i32>
                        %383 = ascvec.loadalign %arg34[%arg47] : memref<8192xf32, 1>, index -> <f32>
                        %384 = ascvec.reinterpret %383 : <f32> -> <ui32>
                        %385 = ascvec.cmp_eq %384, %354, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %386 = ascvec.squeeze_store_reg %322, %385 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg43[%c0_35], %386 : memref<528xi32, 1>, <i32>
                        ascvec.store_unalign_post %arg43[%c0_35] : memref<528xi32, 1>
                        %387 = ascvec.loadalign_brc %arg28[%arg48] {brc_width = "b8"} : memref<256xi32, 1>, index -> <ui32>
                        %388 = ascvec.and %387, %290, %mask : <ui32>, <ui32> -> <ui32>
                        %389 = ascvec.loadalign_brc %arg28[%arg49] {brc_width = "b8"} : memref<256xi32, 1>, index -> <ui32>
                        %390 = ascvec.and %389, %290, %mask : <ui32>, <ui32> -> <ui32>
                        %391 = ascvec.loadalign_brc %arg28[%arg50] {brc_width = "b8"} : memref<256xi32, 1>, index -> <ui32>
                        %392 = ascvec.and %391, %290, %mask : <ui32>, <ui32> -> <ui32>
                        %393 = ascvec.loadalign_brc %arg28[%arg51] {brc_width = "b8"} : memref<256xi32, 1>, index -> <ui32>
                        %394 = ascvec.and %393, %290, %mask : <ui32>, <ui32> -> <ui32>
                        %395 = ascvec.shl %388, 24, %mask : <ui32> -> <ui32>
                        %396 = ascvec.shl %390, 16, %mask : <ui32> -> <ui32>
                        %397 = ascvec.add %395, %396, %mask : <ui32>
                        %398 = ascvec.shl %392, 8, %mask : <ui32> -> <ui32>
                        %399 = ascvec.add %397, %398, %mask : <ui32>
                        %400 = ascvec.add %399, %394, %mask : <ui32>
                        ascvec.store_unalign_begin %arg52 : memref<528xi32, 1>
                        %401 = ascvec.loadalign %arg34[%arg53] : memref<8192xf32, 1>, index -> <f32>
                        %402 = ascvec.reinterpret %401 : <f32> -> <ui32>
                        %403 = ascvec.cmp_gt %402, %400, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %404 = ascvec.squeeze_store_reg %307, %403 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg52[%c0_35], %404 : memref<528xi32, 1>, <i32>
                        %405 = ascvec.loadalign %arg34[%arg54] : memref<8192xf32, 1>, index -> <f32>
                        %406 = ascvec.reinterpret %405 : <f32> -> <ui32>
                        %407 = ascvec.cmp_gt %406, %400, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %408 = ascvec.squeeze_store_reg %312, %407 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg52[%c0_35], %408 : memref<528xi32, 1>, <i32>
                        %409 = ascvec.loadalign %arg34[%arg55] : memref<8192xf32, 1>, index -> <f32>
                        %410 = ascvec.reinterpret %409 : <f32> -> <ui32>
                        %411 = ascvec.cmp_gt %410, %400, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %412 = ascvec.squeeze_store_reg %317, %411 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg52[%c0_35], %412 : memref<528xi32, 1>, <i32>
                        %413 = ascvec.loadalign %arg34[%arg56] : memref<8192xf32, 1>, index -> <f32>
                        %414 = ascvec.reinterpret %413 : <f32> -> <ui32>
                        %415 = ascvec.cmp_gt %414, %400, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %416 = ascvec.squeeze_store_reg %322, %415 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg52[%c0_35], %416 : memref<528xi32, 1>, <i32>
                        %417 = ascvec.loadalign %arg34[%arg53] : memref<8192xf32, 1>, index -> <f32>
                        %418 = ascvec.reinterpret %417 : <f32> -> <ui32>
                        %419 = ascvec.cmp_eq %418, %400, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %420 = ascvec.squeeze_store_reg %307, %419 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg52[%c0_35], %420 : memref<528xi32, 1>, <i32>
                        %421 = ascvec.loadalign %arg34[%arg54] : memref<8192xf32, 1>, index -> <f32>
                        %422 = ascvec.reinterpret %421 : <f32> -> <ui32>
                        %423 = ascvec.cmp_eq %422, %400, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %424 = ascvec.squeeze_store_reg %312, %423 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg52[%c0_35], %424 : memref<528xi32, 1>, <i32>
                        %425 = ascvec.loadalign %arg34[%arg55] : memref<8192xf32, 1>, index -> <f32>
                        %426 = ascvec.reinterpret %425 : <f32> -> <ui32>
                        %427 = ascvec.cmp_eq %426, %400, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %428 = ascvec.squeeze_store_reg %317, %427 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg52[%c0_35], %428 : memref<528xi32, 1>, <i32>
                        %429 = ascvec.loadalign %arg34[%arg56] : memref<8192xf32, 1>, index -> <f32>
                        %430 = ascvec.reinterpret %429 : <f32> -> <ui32>
                        %431 = ascvec.cmp_eq %430, %400, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %432 = ascvec.squeeze_store_reg %322, %431 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg52[%c0_35], %432 : memref<528xi32, 1>, <i32>
                        ascvec.store_unalign_post %arg52[%c0_35] : memref<528xi32, 1>
                        %433 = ascvec.loadalign_brc %arg28[%arg57] {brc_width = "b8"} : memref<256xi32, 1>, index -> <ui32>
                        %434 = ascvec.and %433, %290, %mask : <ui32>, <ui32> -> <ui32>
                        %435 = ascvec.loadalign_brc %arg28[%arg58] {brc_width = "b8"} : memref<256xi32, 1>, index -> <ui32>
                        %436 = ascvec.and %435, %290, %mask : <ui32>, <ui32> -> <ui32>
                        %437 = ascvec.loadalign_brc %arg28[%arg59] {brc_width = "b8"} : memref<256xi32, 1>, index -> <ui32>
                        %438 = ascvec.and %437, %290, %mask : <ui32>, <ui32> -> <ui32>
                        %439 = ascvec.loadalign_brc %arg28[%arg60] {brc_width = "b8"} : memref<256xi32, 1>, index -> <ui32>
                        %440 = ascvec.and %439, %290, %mask : <ui32>, <ui32> -> <ui32>
                        %441 = ascvec.shl %434, 24, %mask : <ui32> -> <ui32>
                        %442 = ascvec.shl %436, 16, %mask : <ui32> -> <ui32>
                        %443 = ascvec.add %441, %442, %mask : <ui32>
                        %444 = ascvec.shl %438, 8, %mask : <ui32> -> <ui32>
                        %445 = ascvec.add %443, %444, %mask : <ui32>
                        %446 = ascvec.add %445, %440, %mask : <ui32>
                        ascvec.store_unalign_begin %arg61 : memref<528xi32, 1>
                        %447 = ascvec.loadalign %arg34[%arg62] : memref<8192xf32, 1>, index -> <f32>
                        %448 = ascvec.reinterpret %447 : <f32> -> <ui32>
                        %449 = ascvec.cmp_gt %448, %446, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %450 = ascvec.squeeze_store_reg %307, %449 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg61[%c0_35], %450 : memref<528xi32, 1>, <i32>
                        %451 = ascvec.loadalign %arg34[%arg63] : memref<8192xf32, 1>, index -> <f32>
                        %452 = ascvec.reinterpret %451 : <f32> -> <ui32>
                        %453 = ascvec.cmp_gt %452, %446, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %454 = ascvec.squeeze_store_reg %312, %453 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg61[%c0_35], %454 : memref<528xi32, 1>, <i32>
                        %455 = ascvec.loadalign %arg34[%arg64] : memref<8192xf32, 1>, index -> <f32>
                        %456 = ascvec.reinterpret %455 : <f32> -> <ui32>
                        %457 = ascvec.cmp_gt %456, %446, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %458 = ascvec.squeeze_store_reg %317, %457 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg61[%c0_35], %458 : memref<528xi32, 1>, <i32>
                        %459 = ascvec.loadalign %arg34[%arg65] : memref<8192xf32, 1>, index -> <f32>
                        %460 = ascvec.reinterpret %459 : <f32> -> <ui32>
                        %461 = ascvec.cmp_gt %460, %446, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %462 = ascvec.squeeze_store_reg %322, %461 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg61[%c0_35], %462 : memref<528xi32, 1>, <i32>
                        %463 = ascvec.loadalign %arg34[%arg62] : memref<8192xf32, 1>, index -> <f32>
                        %464 = ascvec.reinterpret %463 : <f32> -> <ui32>
                        %465 = ascvec.cmp_eq %464, %446, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %466 = ascvec.squeeze_store_reg %307, %465 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg61[%c0_35], %466 : memref<528xi32, 1>, <i32>
                        %467 = ascvec.loadalign %arg34[%arg63] : memref<8192xf32, 1>, index -> <f32>
                        %468 = ascvec.reinterpret %467 : <f32> -> <ui32>
                        %469 = ascvec.cmp_eq %468, %446, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %470 = ascvec.squeeze_store_reg %312, %469 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg61[%c0_35], %470 : memref<528xi32, 1>, <i32>
                        %471 = ascvec.loadalign %arg34[%arg64] : memref<8192xf32, 1>, index -> <f32>
                        %472 = ascvec.reinterpret %471 : <f32> -> <ui32>
                        %473 = ascvec.cmp_eq %472, %446, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %474 = ascvec.squeeze_store_reg %317, %473 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg61[%c0_35], %474 : memref<528xi32, 1>, <i32>
                        %475 = ascvec.loadalign %arg34[%arg65] : memref<8192xf32, 1>, index -> <f32>
                        %476 = ascvec.reinterpret %475 : <f32> -> <ui32>
                        %477 = ascvec.cmp_eq %476, %446, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %478 = ascvec.squeeze_store_reg %322, %477 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg61[%c0_35], %478 : memref<528xi32, 1>, <i32>
                        ascvec.store_unalign_post %arg61[%c0_35] : memref<528xi32, 1>
                        ascvec.mem_bar vst_vld
                        %c0_38 = arith.constant 0 : index
                        %c1 = arith.constant 1 : index
                        scf.for %arg80 = %c0_38 to %c1 step %c1 {
                          %479 = ascvec.loadalign_brc %arg66[%arg67] : memref<512xi32, 1>, index -> <i32>
                          %480 = ascvec.duplicate_scalar %arg68, %mask : i32 -> <i32>
                          %481 = ascvec.arange %c0_i32_36 : i32 -> <i32>
                          %482 = ascvec.cmp_lt %481, %479, %mask : <i32>, <i32> -> !ascvec.mask
                          %483 = ascvec.loadalign %arg33[%c0_35] : memref<528xi32, 1>, index -> <i32>
                          %484 = ascvec.select %483, %288, %482 : <i32>
                          %485 = ascvec.cmp_lt_scalar %484, %c256_i32_34, %482 : <i32>, i32 -> !ascvec.mask
                          %486 = ascvec.cmp_ge_scalar %484, %c256_i32_34, %482 : <i32>, i32 -> !ascvec.mask
                          %487 = ascvec.add_scalar %484, %c-256_i32, %mask : <i32>, i32
                          %488 = ascvec.select %487, %288, %486 : <i32>
                          %489 = ascvec.add %488, %480, %mask : <i32>
                          %490 = ascvec.reinterpret %489 : <i32> -> <ui32>
                          %491 = ascvec.gather %arg34[%c0_35], %490, %mask : memref<8192xf32, 1>, <ui32>, !ascvec.mask -> <f32>
                          %492 = ascvec.reinterpret %491 : <f32> -> <ui32>
                          %493 = ascvec.add_scalar %488, %c0_i32_36, %mask : <i32>, i32
                          %494 = ascvec.reinterpret %493 : <i32> -> <ui32>
                          %495 = ascvec.select %289, %492, %485 : <ui32>
                          %496 = ascvec.select %289, %494, %485 : <ui32>
                          %497 = ascvec.select %495, %289, %482 : <ui32>
                          ascvec.storealign %arg69[%c0_35], %497, %mask : memref<1024xui32, 1>, index, <ui32>
                          %498 = ascvec.select %496, %289, %482 : <ui32>
                          ascvec.storealign %arg70[%c0_35], %498, %mask : memref<1024xui32, 1>, index, <ui32>
                          %499 = ascvec.arange %c64_i32_37 : i32 -> <i32>
                          %500 = ascvec.cmp_lt %499, %479, %mask : <i32>, <i32> -> !ascvec.mask
                          %501 = ascvec.loadalign %arg33[%c64] : memref<528xi32, 1>, index -> <i32>
                          %502 = ascvec.select %501, %288, %500 : <i32>
                          %503 = ascvec.cmp_lt_scalar %502, %c256_i32_34, %500 : <i32>, i32 -> !ascvec.mask
                          %504 = ascvec.cmp_ge_scalar %502, %c256_i32_34, %500 : <i32>, i32 -> !ascvec.mask
                          %505 = ascvec.add_scalar %502, %c-256_i32, %mask : <i32>, i32
                          %506 = ascvec.select %505, %288, %504 : <i32>
                          %507 = ascvec.add %506, %480, %mask : <i32>
                          %508 = ascvec.reinterpret %507 : <i32> -> <ui32>
                          %509 = ascvec.gather %arg34[%c0_35], %508, %mask : memref<8192xf32, 1>, <ui32>, !ascvec.mask -> <f32>
                          %510 = ascvec.reinterpret %509 : <f32> -> <ui32>
                          %511 = ascvec.add_scalar %506, %c0_i32_36, %mask : <i32>, i32
                          %512 = ascvec.reinterpret %511 : <i32> -> <ui32>
                          %513 = ascvec.select %289, %510, %503 : <ui32>
                          %514 = ascvec.select %289, %512, %503 : <ui32>
                          %515 = ascvec.select %513, %289, %500 : <ui32>
                          ascvec.storealign %arg69[%c64], %515, %mask : memref<1024xui32, 1>, index, <ui32>
                          %516 = ascvec.select %514, %289, %500 : <ui32>
                          ascvec.storealign %arg70[%c64], %516, %mask : memref<1024xui32, 1>, index, <ui32>
                          %517 = ascvec.arange %c128_i32_33 : i32 -> <i32>
                          %518 = ascvec.cmp_lt %517, %479, %mask : <i32>, <i32> -> !ascvec.mask
                          %519 = ascvec.loadalign %arg33[%c128] : memref<528xi32, 1>, index -> <i32>
                          %520 = ascvec.select %519, %288, %518 : <i32>
                          %521 = ascvec.cmp_lt_scalar %520, %c256_i32_34, %518 : <i32>, i32 -> !ascvec.mask
                          %522 = ascvec.cmp_ge_scalar %520, %c256_i32_34, %518 : <i32>, i32 -> !ascvec.mask
                          %523 = ascvec.add_scalar %520, %c-256_i32, %mask : <i32>, i32
                          %524 = ascvec.select %523, %288, %522 : <i32>
                          %525 = ascvec.add %524, %480, %mask : <i32>
                          %526 = ascvec.reinterpret %525 : <i32> -> <ui32>
                          %527 = ascvec.gather %arg34[%c0_35], %526, %mask : memref<8192xf32, 1>, <ui32>, !ascvec.mask -> <f32>
                          %528 = ascvec.reinterpret %527 : <f32> -> <ui32>
                          %529 = ascvec.add_scalar %524, %c0_i32_36, %mask : <i32>, i32
                          %530 = ascvec.reinterpret %529 : <i32> -> <ui32>
                          %531 = ascvec.select %289, %528, %521 : <ui32>
                          %532 = ascvec.select %289, %530, %521 : <ui32>
                          %533 = ascvec.select %531, %289, %518 : <ui32>
                          ascvec.storealign %arg69[%c128], %533, %mask : memref<1024xui32, 1>, index, <ui32>
                          %534 = ascvec.select %532, %289, %518 : <ui32>
                          ascvec.storealign %arg70[%c128], %534, %mask : memref<1024xui32, 1>, index, <ui32>
                          %535 = ascvec.arange %c192_i32_32 : i32 -> <i32>
                          %536 = ascvec.cmp_lt %535, %479, %mask : <i32>, <i32> -> !ascvec.mask
                          %537 = ascvec.loadalign %arg33[%c192] : memref<528xi32, 1>, index -> <i32>
                          %538 = ascvec.select %537, %288, %536 : <i32>
                          %539 = ascvec.cmp_lt_scalar %538, %c256_i32_34, %536 : <i32>, i32 -> !ascvec.mask
                          %540 = ascvec.cmp_ge_scalar %538, %c256_i32_34, %536 : <i32>, i32 -> !ascvec.mask
                          %541 = ascvec.add_scalar %538, %c-256_i32, %mask : <i32>, i32
                          %542 = ascvec.select %541, %288, %540 : <i32>
                          %543 = ascvec.add %542, %480, %mask : <i32>
                          %544 = ascvec.reinterpret %543 : <i32> -> <ui32>
                          %545 = ascvec.gather %arg34[%c0_35], %544, %mask : memref<8192xf32, 1>, <ui32>, !ascvec.mask -> <f32>
                          %546 = ascvec.reinterpret %545 : <f32> -> <ui32>
                          %547 = ascvec.add_scalar %542, %c0_i32_36, %mask : <i32>, i32
                          %548 = ascvec.reinterpret %547 : <i32> -> <ui32>
                          %549 = ascvec.select %289, %546, %539 : <ui32>
                          %550 = ascvec.select %289, %548, %539 : <ui32>
                          %551 = ascvec.select %549, %289, %536 : <ui32>
                          ascvec.storealign %arg69[%c192], %551, %mask : memref<1024xui32, 1>, index, <ui32>
                          %552 = ascvec.select %550, %289, %536 : <ui32>
                          ascvec.storealign %arg70[%c192], %552, %mask : memref<1024xui32, 1>, index, <ui32>
                          %553 = ascvec.loadalign_brc %arg66[%arg71] : memref<512xi32, 1>, index -> <i32>
                          %554 = ascvec.duplicate_scalar %arg72, %mask : i32 -> <i32>
                          %555 = ascvec.cmp_lt %481, %553, %mask : <i32>, <i32> -> !ascvec.mask
                          %556 = ascvec.loadalign %arg43[%c0_35] : memref<528xi32, 1>, index -> <i32>
                          %557 = ascvec.select %556, %288, %555 : <i32>
                          %558 = ascvec.cmp_lt_scalar %557, %c256_i32_34, %555 : <i32>, i32 -> !ascvec.mask
                          %559 = ascvec.cmp_ge_scalar %557, %c256_i32_34, %555 : <i32>, i32 -> !ascvec.mask
                          %560 = ascvec.add_scalar %557, %c-256_i32, %mask : <i32>, i32
                          %561 = ascvec.select %560, %288, %559 : <i32>
                          %562 = ascvec.add %561, %554, %mask : <i32>
                          %563 = ascvec.reinterpret %562 : <i32> -> <ui32>
                          %564 = ascvec.gather %arg34[%c0_35], %563, %mask : memref<8192xf32, 1>, <ui32>, !ascvec.mask -> <f32>
                          %565 = ascvec.reinterpret %564 : <f32> -> <ui32>
                          %566 = ascvec.add_scalar %561, %c0_i32_36, %mask : <i32>, i32
                          %567 = ascvec.reinterpret %566 : <i32> -> <ui32>
                          %568 = ascvec.select %289, %565, %558 : <ui32>
                          %569 = ascvec.select %289, %567, %558 : <ui32>
                          %570 = ascvec.select %568, %289, %555 : <ui32>
                          ascvec.storealign %arg69[%c256], %570, %mask : memref<1024xui32, 1>, index, <ui32>
                          %571 = ascvec.select %569, %289, %555 : <ui32>
                          ascvec.storealign %arg70[%c256], %571, %mask : memref<1024xui32, 1>, index, <ui32>
                          %572 = ascvec.cmp_lt %499, %553, %mask : <i32>, <i32> -> !ascvec.mask
                          %573 = ascvec.loadalign %arg43[%c64] : memref<528xi32, 1>, index -> <i32>
                          %574 = ascvec.select %573, %288, %572 : <i32>
                          %575 = ascvec.cmp_lt_scalar %574, %c256_i32_34, %572 : <i32>, i32 -> !ascvec.mask
                          %576 = ascvec.cmp_ge_scalar %574, %c256_i32_34, %572 : <i32>, i32 -> !ascvec.mask
                          %577 = ascvec.add_scalar %574, %c-256_i32, %mask : <i32>, i32
                          %578 = ascvec.select %577, %288, %576 : <i32>
                          %579 = ascvec.add %578, %554, %mask : <i32>
                          %580 = ascvec.reinterpret %579 : <i32> -> <ui32>
                          %581 = ascvec.gather %arg34[%c0_35], %580, %mask : memref<8192xf32, 1>, <ui32>, !ascvec.mask -> <f32>
                          %582 = ascvec.reinterpret %581 : <f32> -> <ui32>
                          %583 = ascvec.add_scalar %578, %c0_i32_36, %mask : <i32>, i32
                          %584 = ascvec.reinterpret %583 : <i32> -> <ui32>
                          %585 = ascvec.select %289, %582, %575 : <ui32>
                          %586 = ascvec.select %289, %584, %575 : <ui32>
                          %587 = ascvec.select %585, %289, %572 : <ui32>
                          ascvec.storealign %arg69[%c320], %587, %mask : memref<1024xui32, 1>, index, <ui32>
                          %588 = ascvec.select %586, %289, %572 : <ui32>
                          ascvec.storealign %arg70[%c320], %588, %mask : memref<1024xui32, 1>, index, <ui32>
                          %589 = ascvec.cmp_lt %517, %553, %mask : <i32>, <i32> -> !ascvec.mask
                          %590 = ascvec.loadalign %arg43[%c128] : memref<528xi32, 1>, index -> <i32>
                          %591 = ascvec.select %590, %288, %589 : <i32>
                          %592 = ascvec.cmp_lt_scalar %591, %c256_i32_34, %589 : <i32>, i32 -> !ascvec.mask
                          %593 = ascvec.cmp_ge_scalar %591, %c256_i32_34, %589 : <i32>, i32 -> !ascvec.mask
                          %594 = ascvec.add_scalar %591, %c-256_i32, %mask : <i32>, i32
                          %595 = ascvec.select %594, %288, %593 : <i32>
                          %596 = ascvec.add %595, %554, %mask : <i32>
                          %597 = ascvec.reinterpret %596 : <i32> -> <ui32>
                          %598 = ascvec.gather %arg34[%c0_35], %597, %mask : memref<8192xf32, 1>, <ui32>, !ascvec.mask -> <f32>
                          %599 = ascvec.reinterpret %598 : <f32> -> <ui32>
                          %600 = ascvec.add_scalar %595, %c0_i32_36, %mask : <i32>, i32
                          %601 = ascvec.reinterpret %600 : <i32> -> <ui32>
                          %602 = ascvec.select %289, %599, %592 : <ui32>
                          %603 = ascvec.select %289, %601, %592 : <ui32>
                          %604 = ascvec.select %602, %289, %589 : <ui32>
                          ascvec.storealign %arg69[%c384], %604, %mask : memref<1024xui32, 1>, index, <ui32>
                          %605 = ascvec.select %603, %289, %589 : <ui32>
                          ascvec.storealign %arg70[%c384], %605, %mask : memref<1024xui32, 1>, index, <ui32>
                          %606 = ascvec.cmp_lt %535, %553, %mask : <i32>, <i32> -> !ascvec.mask
                          %607 = ascvec.loadalign %arg43[%c192] : memref<528xi32, 1>, index -> <i32>
                          %608 = ascvec.select %607, %288, %606 : <i32>
                          %609 = ascvec.cmp_lt_scalar %608, %c256_i32_34, %606 : <i32>, i32 -> !ascvec.mask
                          %610 = ascvec.cmp_ge_scalar %608, %c256_i32_34, %606 : <i32>, i32 -> !ascvec.mask
                          %611 = ascvec.add_scalar %608, %c-256_i32, %mask : <i32>, i32
                          %612 = ascvec.select %611, %288, %610 : <i32>
                          %613 = ascvec.add %612, %554, %mask : <i32>
                          %614 = ascvec.reinterpret %613 : <i32> -> <ui32>
                          %615 = ascvec.gather %arg34[%c0_35], %614, %mask : memref<8192xf32, 1>, <ui32>, !ascvec.mask -> <f32>
                          %616 = ascvec.reinterpret %615 : <f32> -> <ui32>
                          %617 = ascvec.add_scalar %612, %c0_i32_36, %mask : <i32>, i32
                          %618 = ascvec.reinterpret %617 : <i32> -> <ui32>
                          %619 = ascvec.select %289, %616, %609 : <ui32>
                          %620 = ascvec.select %289, %618, %609 : <ui32>
                          %621 = ascvec.select %619, %289, %606 : <ui32>
                          ascvec.storealign %arg69[%c448], %621, %mask : memref<1024xui32, 1>, index, <ui32>
                          %622 = ascvec.select %620, %289, %606 : <ui32>
                          ascvec.storealign %arg70[%c448], %622, %mask : memref<1024xui32, 1>, index, <ui32>
                          %623 = ascvec.loadalign_brc %arg66[%arg73] : memref<512xi32, 1>, index -> <i32>
                          %624 = ascvec.duplicate_scalar %arg74, %mask : i32 -> <i32>
                          %625 = ascvec.cmp_lt %481, %623, %mask : <i32>, <i32> -> !ascvec.mask
                          %626 = ascvec.loadalign %arg52[%c0_35] : memref<528xi32, 1>, index -> <i32>
                          %627 = ascvec.select %626, %288, %625 : <i32>
                          %628 = ascvec.cmp_lt_scalar %627, %c256_i32_34, %625 : <i32>, i32 -> !ascvec.mask
                          %629 = ascvec.cmp_ge_scalar %627, %c256_i32_34, %625 : <i32>, i32 -> !ascvec.mask
                          %630 = ascvec.add_scalar %627, %c-256_i32, %mask : <i32>, i32
                          %631 = ascvec.select %630, %288, %629 : <i32>
                          %632 = ascvec.add %631, %624, %mask : <i32>
                          %633 = ascvec.reinterpret %632 : <i32> -> <ui32>
                          %634 = ascvec.gather %arg34[%c0_35], %633, %mask : memref<8192xf32, 1>, <ui32>, !ascvec.mask -> <f32>
                          %635 = ascvec.reinterpret %634 : <f32> -> <ui32>
                          %636 = ascvec.add_scalar %631, %c0_i32_36, %mask : <i32>, i32
                          %637 = ascvec.reinterpret %636 : <i32> -> <ui32>
                          %638 = ascvec.select %289, %635, %628 : <ui32>
                          %639 = ascvec.select %289, %637, %628 : <ui32>
                          %640 = ascvec.select %638, %289, %625 : <ui32>
                          ascvec.storealign %arg69[%c512], %640, %mask : memref<1024xui32, 1>, index, <ui32>
                          %641 = ascvec.select %639, %289, %625 : <ui32>
                          ascvec.storealign %arg70[%c512], %641, %mask : memref<1024xui32, 1>, index, <ui32>
                          %642 = ascvec.cmp_lt %499, %623, %mask : <i32>, <i32> -> !ascvec.mask
                          %643 = ascvec.loadalign %arg52[%c64] : memref<528xi32, 1>, index -> <i32>
                          %644 = ascvec.select %643, %288, %642 : <i32>
                          %645 = ascvec.cmp_lt_scalar %644, %c256_i32_34, %642 : <i32>, i32 -> !ascvec.mask
                          %646 = ascvec.cmp_ge_scalar %644, %c256_i32_34, %642 : <i32>, i32 -> !ascvec.mask
                          %647 = ascvec.add_scalar %644, %c-256_i32, %mask : <i32>, i32
                          %648 = ascvec.select %647, %288, %646 : <i32>
                          %649 = ascvec.add %648, %624, %mask : <i32>
                          %650 = ascvec.reinterpret %649 : <i32> -> <ui32>
                          %651 = ascvec.gather %arg34[%c0_35], %650, %mask : memref<8192xf32, 1>, <ui32>, !ascvec.mask -> <f32>
                          %652 = ascvec.reinterpret %651 : <f32> -> <ui32>
                          %653 = ascvec.add_scalar %648, %c0_i32_36, %mask : <i32>, i32
                          %654 = ascvec.reinterpret %653 : <i32> -> <ui32>
                          %655 = ascvec.select %289, %652, %645 : <ui32>
                          %656 = ascvec.select %289, %654, %645 : <ui32>
                          %657 = ascvec.select %655, %289, %642 : <ui32>
                          ascvec.storealign %arg69[%c576], %657, %mask : memref<1024xui32, 1>, index, <ui32>
                          %658 = ascvec.select %656, %289, %642 : <ui32>
                          ascvec.storealign %arg70[%c576], %658, %mask : memref<1024xui32, 1>, index, <ui32>
                          %659 = ascvec.cmp_lt %517, %623, %mask : <i32>, <i32> -> !ascvec.mask
                          %660 = ascvec.loadalign %arg52[%c128] : memref<528xi32, 1>, index -> <i32>
                          %661 = ascvec.select %660, %288, %659 : <i32>
                          %662 = ascvec.cmp_lt_scalar %661, %c256_i32_34, %659 : <i32>, i32 -> !ascvec.mask
                          %663 = ascvec.cmp_ge_scalar %661, %c256_i32_34, %659 : <i32>, i32 -> !ascvec.mask
                          %664 = ascvec.add_scalar %661, %c-256_i32, %mask : <i32>, i32
                          %665 = ascvec.select %664, %288, %663 : <i32>
                          %666 = ascvec.add %665, %624, %mask : <i32>
                          %667 = ascvec.reinterpret %666 : <i32> -> <ui32>
                          %668 = ascvec.gather %arg34[%c0_35], %667, %mask : memref<8192xf32, 1>, <ui32>, !ascvec.mask -> <f32>
                          %669 = ascvec.reinterpret %668 : <f32> -> <ui32>
                          %670 = ascvec.add_scalar %665, %c0_i32_36, %mask : <i32>, i32
                          %671 = ascvec.reinterpret %670 : <i32> -> <ui32>
                          %672 = ascvec.select %289, %669, %662 : <ui32>
                          %673 = ascvec.select %289, %671, %662 : <ui32>
                          %674 = ascvec.select %672, %289, %659 : <ui32>
                          ascvec.storealign %arg69[%c640], %674, %mask : memref<1024xui32, 1>, index, <ui32>
                          %675 = ascvec.select %673, %289, %659 : <ui32>
                          ascvec.storealign %arg70[%c640], %675, %mask : memref<1024xui32, 1>, index, <ui32>
                          %676 = ascvec.cmp_lt %535, %623, %mask : <i32>, <i32> -> !ascvec.mask
                          %677 = ascvec.loadalign %arg52[%c192] : memref<528xi32, 1>, index -> <i32>
                          %678 = ascvec.select %677, %288, %676 : <i32>
                          %679 = ascvec.cmp_lt_scalar %678, %c256_i32_34, %676 : <i32>, i32 -> !ascvec.mask
                          %680 = ascvec.cmp_ge_scalar %678, %c256_i32_34, %676 : <i32>, i32 -> !ascvec.mask
                          %681 = ascvec.add_scalar %678, %c-256_i32, %mask : <i32>, i32
                          %682 = ascvec.select %681, %288, %680 : <i32>
                          %683 = ascvec.add %682, %624, %mask : <i32>
                          %684 = ascvec.reinterpret %683 : <i32> -> <ui32>
                          %685 = ascvec.gather %arg34[%c0_35], %684, %mask : memref<8192xf32, 1>, <ui32>, !ascvec.mask -> <f32>
                          %686 = ascvec.reinterpret %685 : <f32> -> <ui32>
                          %687 = ascvec.add_scalar %682, %c0_i32_36, %mask : <i32>, i32
                          %688 = ascvec.reinterpret %687 : <i32> -> <ui32>
                          %689 = ascvec.select %289, %686, %679 : <ui32>
                          %690 = ascvec.select %289, %688, %679 : <ui32>
                          %691 = ascvec.select %689, %289, %676 : <ui32>
                          ascvec.storealign %arg69[%c704], %691, %mask : memref<1024xui32, 1>, index, <ui32>
                          %692 = ascvec.select %690, %289, %676 : <ui32>
                          ascvec.storealign %arg70[%c704], %692, %mask : memref<1024xui32, 1>, index, <ui32>
                          %693 = ascvec.loadalign_brc %arg66[%arg75] : memref<512xi32, 1>, index -> <i32>
                          %694 = ascvec.duplicate_scalar %arg76, %mask : i32 -> <i32>
                          %695 = ascvec.cmp_lt %481, %693, %mask : <i32>, <i32> -> !ascvec.mask
                          %696 = ascvec.loadalign %arg61[%c0_35] : memref<528xi32, 1>, index -> <i32>
                          %697 = ascvec.select %696, %288, %695 : <i32>
                          %698 = ascvec.cmp_lt_scalar %697, %c256_i32_34, %695 : <i32>, i32 -> !ascvec.mask
                          %699 = ascvec.cmp_ge_scalar %697, %c256_i32_34, %695 : <i32>, i32 -> !ascvec.mask
                          %700 = ascvec.add_scalar %697, %c-256_i32, %mask : <i32>, i32
                          %701 = ascvec.select %700, %288, %699 : <i32>
                          %702 = ascvec.add %701, %694, %mask : <i32>
                          %703 = ascvec.reinterpret %702 : <i32> -> <ui32>
                          %704 = ascvec.gather %arg34[%c0_35], %703, %mask : memref<8192xf32, 1>, <ui32>, !ascvec.mask -> <f32>
                          %705 = ascvec.reinterpret %704 : <f32> -> <ui32>
                          %706 = ascvec.add_scalar %701, %c0_i32_36, %mask : <i32>, i32
                          %707 = ascvec.reinterpret %706 : <i32> -> <ui32>
                          %708 = ascvec.select %289, %705, %698 : <ui32>
                          %709 = ascvec.select %289, %707, %698 : <ui32>
                          %710 = ascvec.select %708, %289, %695 : <ui32>
                          ascvec.storealign %arg69[%c768], %710, %mask : memref<1024xui32, 1>, index, <ui32>
                          %711 = ascvec.select %709, %289, %695 : <ui32>
                          ascvec.storealign %arg70[%c768], %711, %mask : memref<1024xui32, 1>, index, <ui32>
                          %712 = ascvec.cmp_lt %499, %693, %mask : <i32>, <i32> -> !ascvec.mask
                          %713 = ascvec.loadalign %arg61[%c64] : memref<528xi32, 1>, index -> <i32>
                          %714 = ascvec.select %713, %288, %712 : <i32>
                          %715 = ascvec.cmp_lt_scalar %714, %c256_i32_34, %712 : <i32>, i32 -> !ascvec.mask
                          %716 = ascvec.cmp_ge_scalar %714, %c256_i32_34, %712 : <i32>, i32 -> !ascvec.mask
                          %717 = ascvec.add_scalar %714, %c-256_i32, %mask : <i32>, i32
                          %718 = ascvec.select %717, %288, %716 : <i32>
                          %719 = ascvec.add %718, %694, %mask : <i32>
                          %720 = ascvec.reinterpret %719 : <i32> -> <ui32>
                          %721 = ascvec.gather %arg34[%c0_35], %720, %mask : memref<8192xf32, 1>, <ui32>, !ascvec.mask -> <f32>
                          %722 = ascvec.reinterpret %721 : <f32> -> <ui32>
                          %723 = ascvec.add_scalar %718, %c0_i32_36, %mask : <i32>, i32
                          %724 = ascvec.reinterpret %723 : <i32> -> <ui32>
                          %725 = ascvec.select %289, %722, %715 : <ui32>
                          %726 = ascvec.select %289, %724, %715 : <ui32>
                          %727 = ascvec.select %725, %289, %712 : <ui32>
                          ascvec.storealign %arg69[%c832], %727, %mask : memref<1024xui32, 1>, index, <ui32>
                          %728 = ascvec.select %726, %289, %712 : <ui32>
                          ascvec.storealign %arg70[%c832], %728, %mask : memref<1024xui32, 1>, index, <ui32>
                          %729 = ascvec.cmp_lt %517, %693, %mask : <i32>, <i32> -> !ascvec.mask
                          %730 = ascvec.loadalign %arg61[%c128] : memref<528xi32, 1>, index -> <i32>
                          %731 = ascvec.select %730, %288, %729 : <i32>
                          %732 = ascvec.cmp_lt_scalar %731, %c256_i32_34, %729 : <i32>, i32 -> !ascvec.mask
                          %733 = ascvec.cmp_ge_scalar %731, %c256_i32_34, %729 : <i32>, i32 -> !ascvec.mask
                          %734 = ascvec.add_scalar %731, %c-256_i32, %mask : <i32>, i32
                          %735 = ascvec.select %734, %288, %733 : <i32>
                          %736 = ascvec.add %735, %694, %mask : <i32>
                          %737 = ascvec.reinterpret %736 : <i32> -> <ui32>
                          %738 = ascvec.gather %arg34[%c0_35], %737, %mask : memref<8192xf32, 1>, <ui32>, !ascvec.mask -> <f32>
                          %739 = ascvec.reinterpret %738 : <f32> -> <ui32>
                          %740 = ascvec.add_scalar %735, %c0_i32_36, %mask : <i32>, i32
                          %741 = ascvec.reinterpret %740 : <i32> -> <ui32>
                          %742 = ascvec.select %289, %739, %732 : <ui32>
                          %743 = ascvec.select %289, %741, %732 : <ui32>
                          %744 = ascvec.select %742, %289, %729 : <ui32>
                          ascvec.storealign %arg69[%c896], %744, %mask : memref<1024xui32, 1>, index, <ui32>
                          %745 = ascvec.select %743, %289, %729 : <ui32>
                          ascvec.storealign %arg70[%c896], %745, %mask : memref<1024xui32, 1>, index, <ui32>
                          %746 = ascvec.cmp_lt %535, %693, %mask : <i32>, <i32> -> !ascvec.mask
                          %747 = ascvec.loadalign %arg61[%c192] : memref<528xi32, 1>, index -> <i32>
                          %748 = ascvec.select %747, %288, %746 : <i32>
                          %749 = ascvec.cmp_lt_scalar %748, %c256_i32_34, %746 : <i32>, i32 -> !ascvec.mask
                          %750 = ascvec.cmp_ge_scalar %748, %c256_i32_34, %746 : <i32>, i32 -> !ascvec.mask
                          %751 = ascvec.add_scalar %748, %c-256_i32, %mask : <i32>, i32
                          %752 = ascvec.select %751, %288, %750 : <i32>
                          %753 = ascvec.add %752, %694, %mask : <i32>
                          %754 = ascvec.reinterpret %753 : <i32> -> <ui32>
                          %755 = ascvec.gather %arg34[%c0_35], %754, %mask : memref<8192xf32, 1>, <ui32>, !ascvec.mask -> <f32>
                          %756 = ascvec.reinterpret %755 : <f32> -> <ui32>
                          %757 = ascvec.add_scalar %752, %c0_i32_36, %mask : <i32>, i32
                          %758 = ascvec.reinterpret %757 : <i32> -> <ui32>
                          %759 = ascvec.select %289, %756, %749 : <ui32>
                          %760 = ascvec.select %289, %758, %749 : <ui32>
                          %761 = ascvec.select %759, %289, %746 : <ui32>
                          ascvec.storealign %arg69[%c960], %761, %mask : memref<1024xui32, 1>, index, <ui32>
                          %762 = ascvec.select %760, %289, %746 : <ui32>
                          ascvec.storealign %arg70[%c960], %762, %mask : memref<1024xui32, 1>, index, <ui32>
                        }
                        ascvec.mem_bar vst_vld
                        scf.for %arg80 = %c0_i32_36 to %c4_i32_31 step %c1_i32_30  : i32 {
                          %479 = arith.addi %arg77, %arg80 : i32
                          %480 = arith.muli %479, %c256_i32_34 : i32
                          %481 = arith.muli %arg80, %c256_i32_34 overflow<nsw> : i32
                          %482 = arith.index_cast %481 : i32 to index
                          %483 = ascvec.loadalign %arg69[%482] : memref<1024xui32, 1>, index -> <ui32>
                          %484 = arith.index_cast %480 : i32 to index
                          ascvec.storealign %arg78[%484], %483, %mask : memref<8192xui32, 1>, index, <ui32>
                          %485 = ascvec.loadalign %arg70[%482] : memref<1024xui32, 1>, index -> <ui32>
                          ascvec.storealign %arg79[%484], %485, %mask : memref<8192xui32, 1>, index, <ui32>
                          %486 = arith.addi %480, %c64_i32_37 : i32
                          %487 = arith.addi %481, %c64_i32_37 overflow<nsw> : i32
                          %488 = arith.index_cast %487 : i32 to index
                          %489 = ascvec.loadalign %arg69[%488] : memref<1024xui32, 1>, index -> <ui32>
                          %490 = arith.index_cast %486 : i32 to index
                          ascvec.storealign %arg78[%490], %489, %mask : memref<8192xui32, 1>, index, <ui32>
                          %491 = ascvec.loadalign %arg70[%488] : memref<1024xui32, 1>, index -> <ui32>
                          ascvec.storealign %arg79[%490], %491, %mask : memref<8192xui32, 1>, index, <ui32>
                          %492 = arith.addi %480, %c128_i32_33 : i32
                          %493 = arith.addi %481, %c128_i32_33 overflow<nsw> : i32
                          %494 = arith.index_cast %493 : i32 to index
                          %495 = ascvec.loadalign %arg69[%494] : memref<1024xui32, 1>, index -> <ui32>
                          %496 = arith.index_cast %492 : i32 to index
                          ascvec.storealign %arg78[%496], %495, %mask : memref<8192xui32, 1>, index, <ui32>
                          %497 = ascvec.loadalign %arg70[%494] : memref<1024xui32, 1>, index -> <ui32>
                          ascvec.storealign %arg79[%496], %497, %mask : memref<8192xui32, 1>, index, <ui32>
                          %498 = arith.addi %480, %c192_i32_32 : i32
                          %499 = arith.addi %481, %c192_i32_32 overflow<nsw> : i32
                          %500 = arith.index_cast %499 : i32 to index
                          %501 = ascvec.loadalign %arg69[%500] : memref<1024xui32, 1>, index -> <ui32>
                          %502 = arith.index_cast %498 : i32 to index
                          ascvec.storealign %arg78[%502], %501, %mask : memref<8192xui32, 1>, index, <ui32>
                          %503 = ascvec.loadalign %arg70[%500] : memref<1024xui32, 1>, index -> <ui32>
                          ascvec.storealign %arg79[%502], %503, %mask : memref<8192xui32, 1>, index, <ui32>
                        }
                      } {cannir.vf_mode = "simd"}
                    } else {
                      %196 = arith.trunci %arg27 : i64 to i32
                      %197 = arith.muli %196, %c16_i32 overflow<nsw> : i32
                      %198 = arith.addi %197, %c1_i32 overflow<nsw> : i32
                      %199 = arith.index_cast %198 : i32 to index
                      %200 = arith.addi %196, %c1_i32 overflow<nsw> : i32
                      %201 = arith.muli %200, %c16_i32 overflow<nsw> : i32
                      %202 = arith.addi %201, %c1_i32 overflow<nsw> : i32
                      %203 = arith.index_cast %202 : i32 to index
                      %204 = arith.addi %196, %c2_i32 overflow<nsw> : i32
                      %205 = arith.muli %204, %c16_i32 overflow<nsw> : i32
                      %206 = arith.addi %205, %c1_i32 overflow<nsw> : i32
                      %207 = arith.index_cast %206 : i32 to index
                      %208 = arith.addi %196, %c3_i32 overflow<nsw> : i32
                      %209 = arith.muli %208, %c16_i32 overflow<nsw> : i32
                      %210 = arith.addi %209, %c1_i32 overflow<nsw> : i32
                      %211 = arith.index_cast %210 : i32 to index
                      %212 = arith.muli %196, %c8_i32 overflow<nsw> : i32
                      %213 = arith.addi %212, %c1_i32 overflow<nsw> : i32
                      %214 = arith.index_cast %213 : i32 to index
                      %215 = arith.addi %212, %c5_i32 overflow<nsw> : i32
                      %216 = arith.index_cast %215 : i32 to index
                      %217 = arith.muli %200, %c8_i32 overflow<nsw> : i32
                      %218 = arith.addi %217, %c1_i32 overflow<nsw> : i32
                      %219 = arith.index_cast %218 : i32 to index
                      %220 = arith.addi %217, %c5_i32 overflow<nsw> : i32
                      %221 = arith.index_cast %220 : i32 to index
                      %222 = arith.muli %204, %c8_i32 overflow<nsw> : i32
                      %223 = arith.addi %222, %c1_i32 overflow<nsw> : i32
                      %224 = arith.index_cast %223 : i32 to index
                      %225 = arith.addi %222, %c5_i32 overflow<nsw> : i32
                      %226 = arith.index_cast %225 : i32 to index
                      %227 = arith.muli %208, %c8_i32 overflow<nsw> : i32
                      %228 = arith.addi %227, %c1_i32 overflow<nsw> : i32
                      %229 = arith.index_cast %228 : i32 to index
                      %230 = arith.addi %227, %c5_i32 overflow<nsw> : i32
                      %231 = arith.index_cast %230 : i32 to index
                      ascvec.vf(%arg27, %alloc_17, %alloc_11, %alloc_19, %alloc_13, %199, %alloc_20, %203, %alloc_21, %207, %alloc_22, %211, %alloc_23, %alloc_14, %214, %216, %219, %221, %224, %226, %229, %231 : i64, memref<8192xui32, 1>, memref<8192xf32, 1>, memref<1024xui16, 1>, memref<512xi32, 1>, index, memref<256xui16, 1>, index, memref<256xui16, 1>, index, memref<256xui16, 1>, index, memref<256xui16, 1>, memref<256xi32, 1>, index, index, index, index, index, index, index, index) {
                      ^bb0(%arg28: i64, %arg29: memref<8192xui32, 1>, %arg30: memref<8192xf32, 1>, %arg31: memref<1024xui16, 1>, %arg32: memref<512xi32, 1>, %arg33: index, %arg34: memref<256xui16, 1>, %arg35: index, %arg36: memref<256xui16, 1>, %arg37: index, %arg38: memref<256xui16, 1>, %arg39: index, %arg40: memref<256xui16, 1>, %arg41: memref<256xi32, 1>, %arg42: index, %arg43: index, %arg44: index, %arg45: index, %arg46: index, %arg47: index, %arg48: index, %arg49: index):
                        %c896 = arith.constant 896 : index
                        %c768 = arith.constant 768 : index
                        %c640 = arith.constant 640 : index
                        %c512 = arith.constant 512 : index
                        %c384 = arith.constant 384 : index
                        %c256 = arith.constant 256 : index
                        %c128 = arith.constant 128 : index
                        %c768_i16 = arith.constant 768 : i16
                        %c512_i16 = arith.constant 512 : i16
                        %c256_i16 = arith.constant 256 : i16
                        %c128_i16 = arith.constant 128 : i16
                        %c513_i32 = arith.constant 513 : i32
                        %c1_i32_30 = arith.constant 1 : i32
                        %c4_i32_31 = arith.constant 4 : i32
                        %c0_i32_32 = arith.constant 0 : i32
                        %c0_33 = arith.constant 0 : index
                        %c65535_i32 = arith.constant 65535 : i32
                        %c255_i32 = arith.constant 255 : i32
                        %c1_i16 = arith.constant 1 : i16
                        %c0_i16 = arith.constant 0 : i16
                        %c64_i32_34 = arith.constant 64 : i32
                        %c128_i32_35 = arith.constant 128 : i32
                        %c256_i32_36 = arith.constant 256 : i32
                        %mask, %remaining_out = ascvec.update_mask %c256_i32_36 elem_bits 8 : !ascvec.mask
                        %mask_37, %remaining_out_38 = ascvec.update_mask %c128_i32_35 elem_bits 16 : !ascvec.mask
                        %mask_39, %remaining_out_40 = ascvec.update_mask %c64_i32_34 : !ascvec.mask
                        %289 = ascvec.duplicate_scalar %c0_i16, %mask_37 : i16 -> <ui16>
                        %290 = ascvec.duplicate_scalar %c1_i16, %mask_37 : i16 -> <ui16>
                        %291 = ascvec.duplicate_scalar %c255_i32, %mask_39 : i32 -> <ui32>
                        %292 = ascvec.duplicate_scalar %c65535_i32, %mask_39 : i32 -> <ui32>
                        scf.for %arg50 = %c0_i32_32 to %c4_i32_31 step %c1_i32_30  : i32 {
                          %340 = arith.trunci %arg28 : i64 to i32
                          %341 = arith.addi %340, %arg50 : i32
                          %342 = arith.muli %341, %c256_i32_36 : i32
                          %343 = arith.index_cast %342 : i32 to index
                          %result0, %result1 = ascvec.loadalign_deintlv %arg29[%343] {dintlv_width = "b16"} : memref<8192xui32, 1>, index -> <ui16>, <ui16>
                          %344 = arith.addi %342, %c128_i32_35 : i32
                          %345 = arith.index_cast %344 : i32 to index
                          %result0_42, %result1_43 = ascvec.loadalign_deintlv %arg29[%345] {dintlv_width = "b16"} : memref<8192xui32, 1>, index -> <ui16>, <ui16>
                          %346 = ascvec.reinterpret %result1 : <ui16> -> <ui8>
                          %347 = ascvec.reinterpret %result1_43 : <ui16> -> <ui8>
                          %result0_44, %result1_45 = ascvec.deintlv %346, %347 : <ui8>, <ui8> -> <ui8>, <ui8>
                          %348 = ascvec.cumulative_histogram %289, %result1_45, %mask : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          %349 = ascvec.cumulative_histogram %289, %result1_45, %mask {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          %result0_46, %result1_47 = ascvec.loadalign_deintlv %arg30[%343] {dintlv_width = "b16"} : memref<8192xf32, 1>, index -> <ui16>, <ui16>
                          %result0_48, %result1_49 = ascvec.loadalign_deintlv %arg30[%345] {dintlv_width = "b16"} : memref<8192xf32, 1>, index -> <ui16>, <ui16>
                          %350 = ascvec.reinterpret %result1_47 : <ui16> -> <ui8>
                          %351 = ascvec.reinterpret %result1_49 : <ui16> -> <ui8>
                          %result0_50, %result1_51 = ascvec.deintlv %350, %351 : <ui8>, <ui8> -> <ui8>, <ui8>
                          %352 = ascvec.cumulative_histogram %348, %result1_51, %mask : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          %353 = ascvec.cumulative_histogram %349, %result1_51, %mask {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          %354 = arith.muli %arg50, %c256_i32_36 overflow<nsw> : i32
                          %355 = arith.index_cast %354 : i32 to index
                          ascvec.storealign %arg31[%355], %352, %mask_37 : memref<1024xui16, 1>, index, <ui16>
                          %356 = arith.addi %354, %c128_i32_35 overflow<nsw> : i32
                          %357 = arith.index_cast %356 : i32 to index
                          ascvec.storealign %arg31[%357], %353, %mask_37 : memref<1024xui16, 1>, index, <ui16>
                        }
                        ascvec.mem_bar vst_vld
                        %293 = ascvec.loadalign_brc %arg32[%arg33] : memref<512xi32, 1>, index -> <i32>
                        %294 = ascvec.max_scalar %293, %c1_i32_30, %mask_39 : <i32>, i32
                        %295 = ascvec.duplicate_scalar %c513_i32, %mask_39 : i32 -> <i32>
                        %296 = ascvec.sub %295, %294, %mask_39 : <i32>
                        %297 = ascvec.reinterpret %296 : <i32> -> <ui16>
                        %298 = ascvec.duplicate %297, %mask_37 : <ui16> -> <ui16>
                        ascvec.store_unalign_begin %arg34 : memref<256xui16, 1>
                        %299 = ascvec.arange %c0_i16 : i16 -> <ui16>
                        %300 = ascvec.loadalign %arg31[%c0_33] : memref<1024xui16, 1>, index -> <ui16>
                        %301 = ascvec.cmp_ge %300, %298, %mask_37 : <ui16>, <ui16> -> !ascvec.mask
                        %302 = ascvec.squeeze_store_reg %299, %301 : <ui16>, !ascvec.mask -> <ui16>
                        ascvec.store_unalign %arg34[%c0_33], %302 : memref<256xui16, 1>, <ui16>
                        %303 = ascvec.arange %c128_i16 : i16 -> <ui16>
                        %304 = ascvec.loadalign %arg31[%c128] : memref<1024xui16, 1>, index -> <ui16>
                        %305 = ascvec.cmp_ge %304, %298, %mask_37 : <ui16>, <ui16> -> !ascvec.mask
                        %306 = ascvec.squeeze_store_reg %303, %305 : <ui16>, !ascvec.mask -> <ui16>
                        ascvec.store_unalign %arg34[%c0_33], %306 : memref<256xui16, 1>, <ui16>
                        ascvec.store_unalign_post %arg34[%c0_33] : memref<256xui16, 1>
                        %307 = ascvec.loadalign_brc %arg32[%arg35] : memref<512xi32, 1>, index -> <i32>
                        %308 = ascvec.max_scalar %307, %c1_i32_30, %mask_39 : <i32>, i32
                        %309 = ascvec.sub %295, %308, %mask_39 : <i32>
                        %310 = ascvec.reinterpret %309 : <i32> -> <ui16>
                        %311 = ascvec.duplicate %310, %mask_37 : <ui16> -> <ui16>
                        ascvec.store_unalign_begin %arg36 : memref<256xui16, 1>
                        %312 = ascvec.loadalign %arg31[%c256] : memref<1024xui16, 1>, index -> <ui16>
                        %313 = ascvec.cmp_ge %312, %311, %mask_37 : <ui16>, <ui16> -> !ascvec.mask
                        %314 = ascvec.squeeze_store_reg %299, %313 : <ui16>, !ascvec.mask -> <ui16>
                        ascvec.store_unalign %arg36[%c0_33], %314 : memref<256xui16, 1>, <ui16>
                        %315 = ascvec.loadalign %arg31[%c384] : memref<1024xui16, 1>, index -> <ui16>
                        %316 = ascvec.cmp_ge %315, %311, %mask_37 : <ui16>, <ui16> -> !ascvec.mask
                        %317 = ascvec.squeeze_store_reg %303, %316 : <ui16>, !ascvec.mask -> <ui16>
                        ascvec.store_unalign %arg36[%c0_33], %317 : memref<256xui16, 1>, <ui16>
                        ascvec.store_unalign_post %arg36[%c0_33] : memref<256xui16, 1>
                        %318 = ascvec.loadalign_brc %arg32[%arg37] : memref<512xi32, 1>, index -> <i32>
                        %319 = ascvec.max_scalar %318, %c1_i32_30, %mask_39 : <i32>, i32
                        %320 = ascvec.sub %295, %319, %mask_39 : <i32>
                        %321 = ascvec.reinterpret %320 : <i32> -> <ui16>
                        %322 = ascvec.duplicate %321, %mask_37 : <ui16> -> <ui16>
                        ascvec.store_unalign_begin %arg38 : memref<256xui16, 1>
                        %323 = ascvec.loadalign %arg31[%c512] : memref<1024xui16, 1>, index -> <ui16>
                        %324 = ascvec.cmp_ge %323, %322, %mask_37 : <ui16>, <ui16> -> !ascvec.mask
                        %325 = ascvec.squeeze_store_reg %299, %324 : <ui16>, !ascvec.mask -> <ui16>
                        ascvec.store_unalign %arg38[%c0_33], %325 : memref<256xui16, 1>, <ui16>
                        %326 = ascvec.loadalign %arg31[%c640] : memref<1024xui16, 1>, index -> <ui16>
                        %327 = ascvec.cmp_ge %326, %322, %mask_37 : <ui16>, <ui16> -> !ascvec.mask
                        %328 = ascvec.squeeze_store_reg %303, %327 : <ui16>, !ascvec.mask -> <ui16>
                        ascvec.store_unalign %arg38[%c0_33], %328 : memref<256xui16, 1>, <ui16>
                        ascvec.store_unalign_post %arg38[%c0_33] : memref<256xui16, 1>
                        %329 = ascvec.loadalign_brc %arg32[%arg39] : memref<512xi32, 1>, index -> <i32>
                        %330 = ascvec.max_scalar %329, %c1_i32_30, %mask_39 : <i32>, i32
                        %331 = ascvec.sub %295, %330, %mask_39 : <i32>
                        %332 = ascvec.reinterpret %331 : <i32> -> <ui16>
                        %333 = ascvec.duplicate %332, %mask_37 : <ui16> -> <ui16>
                        ascvec.store_unalign_begin %arg40 : memref<256xui16, 1>
                        %334 = ascvec.loadalign %arg31[%c768] : memref<1024xui16, 1>, index -> <ui16>
                        %335 = ascvec.cmp_ge %334, %333, %mask_37 : <ui16>, <ui16> -> !ascvec.mask
                        %336 = ascvec.squeeze_store_reg %299, %335 : <ui16>, !ascvec.mask -> <ui16>
                        ascvec.store_unalign %arg40[%c0_33], %336 : memref<256xui16, 1>, <ui16>
                        %337 = ascvec.loadalign %arg31[%c896] : memref<1024xui16, 1>, index -> <ui16>
                        %338 = ascvec.cmp_ge %337, %333, %mask_37 : <ui16>, <ui16> -> !ascvec.mask
                        %339 = ascvec.squeeze_store_reg %303, %338 : <ui16>, !ascvec.mask -> <ui16>
                        ascvec.store_unalign %arg40[%c0_33], %339 : memref<256xui16, 1>, <ui16>
                        ascvec.store_unalign_post %arg40[%c0_33] : memref<256xui16, 1>
                        ascvec.mem_bar vst_vld
                        %c0_41 = arith.constant 0 : index
                        %c1 = arith.constant 1 : index
                        scf.for %arg50 = %c0_41 to %c1 step %c1 {
                          %340 = ascvec.loadalign_brc %arg32[%arg33] : memref<512xi32, 1>, index -> <i32>
                          %341 = ascvec.max_scalar %340, %c1_i32_30, %mask_39 : <i32>, i32
                          %342 = ascvec.sub %295, %341, %mask_39 : <i32>
                          %343 = ascvec.reinterpret %342 : <i32> -> <ui16>
                          %344 = ascvec.duplicate %343, %mask_37 : <ui16> -> <ui16>
                          %345 = ascvec.loadalign_brc %arg34[%c0_33] : memref<256xui16, 1>, index -> <ui16>
                          %346 = ascvec.reinterpret %345 : <ui16> -> <ui32>
                          %347 = ascvec.and %346, %291, %mask_39 : <ui32>, <ui32> -> <ui32>
                          %348 = ascvec.reinterpret %347 : <ui32> -> <i32>
                          ascvec.storealign_1st %arg41[%arg42], %348 : memref<256xi32, 1>, index, <i32>
                          %349 = ascvec.cmp_eq_scalar %345, %c0_i16, %mask_37 : <ui16>, i16 -> !ascvec.mask
                          %350 = ascvec.sub %345, %290, %mask_37 : <ui16>
                          %351 = ascvec.select %289, %350, %349 : <ui16>
                          %352 = ascvec.add %351, %289, %mask_37 : <ui16>
                          %353 = ascvec.gather %arg31[%c0_33], %352, %mask_37 : memref<1024xui16, 1>, <ui16>, !ascvec.mask -> <ui16>
                          %354 = ascvec.select %289, %353, %349 : <ui16>
                          %355 = ascvec.sub %344, %354, %mask_37 : <ui16>
                          %356 = ascvec.reinterpret %355 : <ui16> -> <ui32>
                          %357 = ascvec.and %356, %292, %mask_39 : <ui32>, <ui32> -> <ui32>
                          %358 = ascvec.reinterpret %357 : <ui32> -> <i32>
                          ascvec.storealign_1st %arg41[%arg43], %358 : memref<256xi32, 1>, index, <i32>
                          %359 = ascvec.loadalign_brc %arg32[%arg35] : memref<512xi32, 1>, index -> <i32>
                          %360 = ascvec.max_scalar %359, %c1_i32_30, %mask_39 : <i32>, i32
                          %361 = ascvec.sub %295, %360, %mask_39 : <i32>
                          %362 = ascvec.reinterpret %361 : <i32> -> <ui16>
                          %363 = ascvec.duplicate %362, %mask_37 : <ui16> -> <ui16>
                          %364 = ascvec.loadalign_brc %arg36[%c0_33] : memref<256xui16, 1>, index -> <ui16>
                          %365 = ascvec.reinterpret %364 : <ui16> -> <ui32>
                          %366 = ascvec.and %365, %291, %mask_39 : <ui32>, <ui32> -> <ui32>
                          %367 = ascvec.reinterpret %366 : <ui32> -> <i32>
                          ascvec.storealign_1st %arg41[%arg44], %367 : memref<256xi32, 1>, index, <i32>
                          %368 = ascvec.cmp_eq_scalar %364, %c0_i16, %mask_37 : <ui16>, i16 -> !ascvec.mask
                          %369 = ascvec.sub %364, %290, %mask_37 : <ui16>
                          %370 = ascvec.select %289, %369, %368 : <ui16>
                          %371 = ascvec.duplicate_scalar %c256_i16, %mask_37 : i16 -> <ui16>
                          %372 = ascvec.add %370, %371, %mask_37 : <ui16>
                          %373 = ascvec.gather %arg31[%c0_33], %372, %mask_37 : memref<1024xui16, 1>, <ui16>, !ascvec.mask -> <ui16>
                          %374 = ascvec.select %289, %373, %368 : <ui16>
                          %375 = ascvec.sub %363, %374, %mask_37 : <ui16>
                          %376 = ascvec.reinterpret %375 : <ui16> -> <ui32>
                          %377 = ascvec.and %376, %292, %mask_39 : <ui32>, <ui32> -> <ui32>
                          %378 = ascvec.reinterpret %377 : <ui32> -> <i32>
                          ascvec.storealign_1st %arg41[%arg45], %378 : memref<256xi32, 1>, index, <i32>
                          %379 = ascvec.loadalign_brc %arg32[%arg37] : memref<512xi32, 1>, index -> <i32>
                          %380 = ascvec.max_scalar %379, %c1_i32_30, %mask_39 : <i32>, i32
                          %381 = ascvec.sub %295, %380, %mask_39 : <i32>
                          %382 = ascvec.reinterpret %381 : <i32> -> <ui16>
                          %383 = ascvec.duplicate %382, %mask_37 : <ui16> -> <ui16>
                          %384 = ascvec.loadalign_brc %arg38[%c0_33] : memref<256xui16, 1>, index -> <ui16>
                          %385 = ascvec.reinterpret %384 : <ui16> -> <ui32>
                          %386 = ascvec.and %385, %291, %mask_39 : <ui32>, <ui32> -> <ui32>
                          %387 = ascvec.reinterpret %386 : <ui32> -> <i32>
                          ascvec.storealign_1st %arg41[%arg46], %387 : memref<256xi32, 1>, index, <i32>
                          %388 = ascvec.cmp_eq_scalar %384, %c0_i16, %mask_37 : <ui16>, i16 -> !ascvec.mask
                          %389 = ascvec.sub %384, %290, %mask_37 : <ui16>
                          %390 = ascvec.select %289, %389, %388 : <ui16>
                          %391 = ascvec.duplicate_scalar %c512_i16, %mask_37 : i16 -> <ui16>
                          %392 = ascvec.add %390, %391, %mask_37 : <ui16>
                          %393 = ascvec.gather %arg31[%c0_33], %392, %mask_37 : memref<1024xui16, 1>, <ui16>, !ascvec.mask -> <ui16>
                          %394 = ascvec.select %289, %393, %388 : <ui16>
                          %395 = ascvec.sub %383, %394, %mask_37 : <ui16>
                          %396 = ascvec.reinterpret %395 : <ui16> -> <ui32>
                          %397 = ascvec.and %396, %292, %mask_39 : <ui32>, <ui32> -> <ui32>
                          %398 = ascvec.reinterpret %397 : <ui32> -> <i32>
                          ascvec.storealign_1st %arg41[%arg47], %398 : memref<256xi32, 1>, index, <i32>
                          %399 = ascvec.loadalign_brc %arg32[%arg39] : memref<512xi32, 1>, index -> <i32>
                          %400 = ascvec.max_scalar %399, %c1_i32_30, %mask_39 : <i32>, i32
                          %401 = ascvec.sub %295, %400, %mask_39 : <i32>
                          %402 = ascvec.reinterpret %401 : <i32> -> <ui16>
                          %403 = ascvec.duplicate %402, %mask_37 : <ui16> -> <ui16>
                          %404 = ascvec.loadalign_brc %arg40[%c0_33] : memref<256xui16, 1>, index -> <ui16>
                          %405 = ascvec.reinterpret %404 : <ui16> -> <ui32>
                          %406 = ascvec.and %405, %291, %mask_39 : <ui32>, <ui32> -> <ui32>
                          %407 = ascvec.reinterpret %406 : <ui32> -> <i32>
                          ascvec.storealign_1st %arg41[%arg48], %407 : memref<256xi32, 1>, index, <i32>
                          %408 = ascvec.cmp_eq_scalar %404, %c0_i16, %mask_37 : <ui16>, i16 -> !ascvec.mask
                          %409 = ascvec.sub %404, %290, %mask_37 : <ui16>
                          %410 = ascvec.select %289, %409, %408 : <ui16>
                          %411 = ascvec.duplicate_scalar %c768_i16, %mask_37 : i16 -> <ui16>
                          %412 = ascvec.add %410, %411, %mask_37 : <ui16>
                          %413 = ascvec.gather %arg31[%c0_33], %412, %mask_37 : memref<1024xui16, 1>, <ui16>, !ascvec.mask -> <ui16>
                          %414 = ascvec.select %289, %413, %408 : <ui16>
                          %415 = ascvec.sub %403, %414, %mask_37 : <ui16>
                          %416 = ascvec.reinterpret %415 : <ui16> -> <ui32>
                          %417 = ascvec.and %416, %292, %mask_39 : <ui32>, <ui32> -> <ui32>
                          %418 = ascvec.reinterpret %417 : <ui32> -> <i32>
                          ascvec.storealign_1st %arg41[%arg49], %418 : memref<256xi32, 1>, index, <i32>
                        }
                      } {cannir.vf_mode = "simd"}
                      %232 = arith.addi %212, %c2_i32 overflow<nsw> : i32
                      %233 = arith.index_cast %232 : i32 to index
                      %234 = arith.addi %217, %c2_i32 overflow<nsw> : i32
                      %235 = arith.index_cast %234 : i32 to index
                      %236 = arith.addi %222, %c2_i32 overflow<nsw> : i32
                      %237 = arith.index_cast %236 : i32 to index
                      %238 = arith.addi %227, %c2_i32 overflow<nsw> : i32
                      %239 = arith.index_cast %238 : i32 to index
                      ascvec.vf(%196, %alloc_17, %alloc_14, %alloc_11, %alloc_19, %216, %alloc_20, %221, %alloc_21, %226, %alloc_22, %231, %alloc_23, %233, %235, %237, %239 : i32, memref<8192xui32, 1>, memref<256xi32, 1>, memref<8192xf32, 1>, memref<1024xui16, 1>, index, memref<256xui16, 1>, index, memref<256xui16, 1>, index, memref<256xui16, 1>, index, memref<256xui16, 1>, index, index, index, index) {
                      ^bb0(%arg28: i32, %arg29: memref<8192xui32, 1>, %arg30: memref<256xi32, 1>, %arg31: memref<8192xf32, 1>, %arg32: memref<1024xui16, 1>, %arg33: index, %arg34: memref<256xui16, 1>, %arg35: index, %arg36: memref<256xui16, 1>, %arg37: index, %arg38: memref<256xui16, 1>, %arg39: index, %arg40: memref<256xui16, 1>, %arg41: index, %arg42: index, %arg43: index, %arg44: index):
                        %c896 = arith.constant 896 : index
                        %c768 = arith.constant 768 : index
                        %c640 = arith.constant 640 : index
                        %c512 = arith.constant 512 : index
                        %c384 = arith.constant 384 : index
                        %c256 = arith.constant 256 : index
                        %c128 = arith.constant 128 : index
                        %c768_i16 = arith.constant 768 : i16
                        %c512_i16 = arith.constant 512 : i16
                        %c256_i16 = arith.constant 256 : i16
                        %c128_i16 = arith.constant 128 : i16
                        %c4_i32_30 = arith.constant 4 : i32
                        %c0_i32_31 = arith.constant 0 : i32
                        %c1_i32_32 = arith.constant 1 : i32
                        %c8_i32_33 = arith.constant 8 : i32
                        %c0_34 = arith.constant 0 : index
                        %c65535_i32 = arith.constant 65535 : i32
                        %c255_i32 = arith.constant 255 : i32
                        %c1_i16 = arith.constant 1 : i16
                        %c0_i16 = arith.constant 0 : i16
                        %c64_i32_35 = arith.constant 64 : i32
                        %c128_i32_36 = arith.constant 128 : i32
                        %c256_i32_37 = arith.constant 256 : i32
                        %mask, %remaining_out = ascvec.update_mask %c256_i32_37 elem_bits 8 : !ascvec.mask
                        %mask_38, %remaining_out_39 = ascvec.update_mask %c128_i32_36 elem_bits 16 : !ascvec.mask
                        %mask_40, %remaining_out_41 = ascvec.update_mask %c64_i32_35 : !ascvec.mask
                        %289 = ascvec.duplicate_scalar %c0_i16, %mask_38 : i16 -> <ui16>
                        %290 = ascvec.duplicate_scalar %c1_i16, %mask_38 : i16 -> <ui16>
                        %291 = ascvec.duplicate_scalar %c255_i32, %mask_40 : i32 -> <ui32>
                        %292 = ascvec.duplicate_scalar %c65535_i32, %mask_40 : i32 -> <ui32>
                        scf.for %arg45 = %c0_i32_31 to %c4_i32_30 step %c1_i32_32  : i32 {
                          %293 = arith.addi %arg28, %arg45 : i32
                          %294 = arith.muli %293, %c256_i32_37 : i32
                          %295 = arith.index_cast %294 : i32 to index
                          %result0, %result1 = ascvec.loadalign_deintlv %arg29[%295] {dintlv_width = "b16"} : memref<8192xui32, 1>, index -> <ui16>, <ui16>
                          %296 = arith.addi %294, %c128_i32_36 : i32
                          %297 = arith.index_cast %296 : i32 to index
                          %result0_45, %result1_46 = ascvec.loadalign_deintlv %arg29[%297] {dintlv_width = "b16"} : memref<8192xui32, 1>, index -> <ui16>, <ui16>
                          %298 = ascvec.reinterpret %result1 : <ui16> -> <ui8>
                          %299 = ascvec.reinterpret %result1_46 : <ui16> -> <ui8>
                          %result0_47, %result1_48 = ascvec.deintlv %298, %299 : <ui8>, <ui8> -> <ui8>, <ui8>
                          %300 = arith.muli %293, %c8_i32_33 : i32
                          %301 = arith.addi %300, %c1_i32_32 : i32
                          %302 = arith.index_cast %301 : i32 to index
                          %303 = ascvec.loadalign_brc %arg30[%302] {brc_width = "b8"} : memref<256xi32, 1>, index -> <ui32>
                          %304 = ascvec.reinterpret %303 : <ui32> -> <ui8>
                          %305 = ascvec.cmp_eq %result1_48, %304, %mask : <ui8>, <ui8> -> !ascvec.mask
                          %306 = ascvec.cumulative_histogram %289, %result0_47, %305 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          %307 = ascvec.cumulative_histogram %289, %result0_47, %305 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          %result0_49, %result1_50 = ascvec.loadalign_deintlv %arg31[%295] {dintlv_width = "b16"} : memref<8192xf32, 1>, index -> <ui16>, <ui16>
                          %result0_51, %result1_52 = ascvec.loadalign_deintlv %arg31[%297] {dintlv_width = "b16"} : memref<8192xf32, 1>, index -> <ui16>, <ui16>
                          %308 = ascvec.reinterpret %result1_50 : <ui16> -> <ui8>
                          %309 = ascvec.reinterpret %result1_52 : <ui16> -> <ui8>
                          %result0_53, %result1_54 = ascvec.deintlv %308, %309 : <ui8>, <ui8> -> <ui8>, <ui8>
                          %310 = ascvec.cmp_eq %result1_54, %304, %mask : <ui8>, <ui8> -> !ascvec.mask
                          %311 = ascvec.cumulative_histogram %306, %result0_53, %310 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          %312 = ascvec.cumulative_histogram %307, %result0_53, %310 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          %313 = arith.muli %arg45, %c256_i32_37 overflow<nsw> : i32
                          %314 = arith.index_cast %313 : i32 to index
                          ascvec.storealign %arg32[%314], %311, %mask_38 : memref<1024xui16, 1>, index, <ui16>
                          %315 = arith.addi %313, %c128_i32_36 overflow<nsw> : i32
                          %316 = arith.index_cast %315 : i32 to index
                          ascvec.storealign %arg32[%316], %312, %mask_38 : memref<1024xui16, 1>, index, <ui16>
                        }
                        ascvec.mem_bar vst_vld
                        %c0_42 = arith.constant 0 : index
                        %c1 = arith.constant 1 : index
                        scf.for %arg45 = %c0_42 to %c1 step %c1 {
                          %293 = ascvec.loadalign_brc %arg30[%arg33] : memref<256xi32, 1>, index -> <i32>
                          %294 = ascvec.reinterpret %293 : <i32> -> <ui16>
                          %295 = ascvec.duplicate %294, %mask_38 : <ui16> -> <ui16>
                          ascvec.store_unalign_begin %arg34 : memref<256xui16, 1>
                          %296 = ascvec.arange %c0_i16 : i16 -> <ui16>
                          %297 = ascvec.loadalign %arg32[%c0_34] : memref<1024xui16, 1>, index -> <ui16>
                          %298 = ascvec.cmp_ge %297, %295, %mask_38 : <ui16>, <ui16> -> !ascvec.mask
                          %299 = ascvec.squeeze_store_reg %296, %298 : <ui16>, !ascvec.mask -> <ui16>
                          ascvec.store_unalign %arg34[%c0_34], %299 : memref<256xui16, 1>, <ui16>
                          %300 = ascvec.arange %c128_i16 : i16 -> <ui16>
                          %301 = ascvec.loadalign %arg32[%c128] : memref<1024xui16, 1>, index -> <ui16>
                          %302 = ascvec.cmp_ge %301, %295, %mask_38 : <ui16>, <ui16> -> !ascvec.mask
                          %303 = ascvec.squeeze_store_reg %300, %302 : <ui16>, !ascvec.mask -> <ui16>
                          ascvec.store_unalign %arg34[%c0_34], %303 : memref<256xui16, 1>, <ui16>
                          ascvec.store_unalign_post %arg34[%c0_34] : memref<256xui16, 1>
                          %304 = ascvec.loadalign_brc %arg30[%arg35] : memref<256xi32, 1>, index -> <i32>
                          %305 = ascvec.reinterpret %304 : <i32> -> <ui16>
                          %306 = ascvec.duplicate %305, %mask_38 : <ui16> -> <ui16>
                          ascvec.store_unalign_begin %arg36 : memref<256xui16, 1>
                          %307 = ascvec.loadalign %arg32[%c256] : memref<1024xui16, 1>, index -> <ui16>
                          %308 = ascvec.cmp_ge %307, %306, %mask_38 : <ui16>, <ui16> -> !ascvec.mask
                          %309 = ascvec.squeeze_store_reg %296, %308 : <ui16>, !ascvec.mask -> <ui16>
                          ascvec.store_unalign %arg36[%c0_34], %309 : memref<256xui16, 1>, <ui16>
                          %310 = ascvec.loadalign %arg32[%c384] : memref<1024xui16, 1>, index -> <ui16>
                          %311 = ascvec.cmp_ge %310, %306, %mask_38 : <ui16>, <ui16> -> !ascvec.mask
                          %312 = ascvec.squeeze_store_reg %300, %311 : <ui16>, !ascvec.mask -> <ui16>
                          ascvec.store_unalign %arg36[%c0_34], %312 : memref<256xui16, 1>, <ui16>
                          ascvec.store_unalign_post %arg36[%c0_34] : memref<256xui16, 1>
                          %313 = ascvec.loadalign_brc %arg30[%arg37] : memref<256xi32, 1>, index -> <i32>
                          %314 = ascvec.reinterpret %313 : <i32> -> <ui16>
                          %315 = ascvec.duplicate %314, %mask_38 : <ui16> -> <ui16>
                          ascvec.store_unalign_begin %arg38 : memref<256xui16, 1>
                          %316 = ascvec.loadalign %arg32[%c512] : memref<1024xui16, 1>, index -> <ui16>
                          %317 = ascvec.cmp_ge %316, %315, %mask_38 : <ui16>, <ui16> -> !ascvec.mask
                          %318 = ascvec.squeeze_store_reg %296, %317 : <ui16>, !ascvec.mask -> <ui16>
                          ascvec.store_unalign %arg38[%c0_34], %318 : memref<256xui16, 1>, <ui16>
                          %319 = ascvec.loadalign %arg32[%c640] : memref<1024xui16, 1>, index -> <ui16>
                          %320 = ascvec.cmp_ge %319, %315, %mask_38 : <ui16>, <ui16> -> !ascvec.mask
                          %321 = ascvec.squeeze_store_reg %300, %320 : <ui16>, !ascvec.mask -> <ui16>
                          ascvec.store_unalign %arg38[%c0_34], %321 : memref<256xui16, 1>, <ui16>
                          ascvec.store_unalign_post %arg38[%c0_34] : memref<256xui16, 1>
                          %322 = ascvec.loadalign_brc %arg30[%arg39] : memref<256xi32, 1>, index -> <i32>
                          %323 = ascvec.reinterpret %322 : <i32> -> <ui16>
                          %324 = ascvec.duplicate %323, %mask_38 : <ui16> -> <ui16>
                          ascvec.store_unalign_begin %arg40 : memref<256xui16, 1>
                          %325 = ascvec.loadalign %arg32[%c768] : memref<1024xui16, 1>, index -> <ui16>
                          %326 = ascvec.cmp_ge %325, %324, %mask_38 : <ui16>, <ui16> -> !ascvec.mask
                          %327 = ascvec.squeeze_store_reg %296, %326 : <ui16>, !ascvec.mask -> <ui16>
                          ascvec.store_unalign %arg40[%c0_34], %327 : memref<256xui16, 1>, <ui16>
                          %328 = ascvec.loadalign %arg32[%c896] : memref<1024xui16, 1>, index -> <ui16>
                          %329 = ascvec.cmp_ge %328, %324, %mask_38 : <ui16>, <ui16> -> !ascvec.mask
                          %330 = ascvec.squeeze_store_reg %300, %329 : <ui16>, !ascvec.mask -> <ui16>
                          ascvec.store_unalign %arg40[%c0_34], %330 : memref<256xui16, 1>, <ui16>
                          ascvec.store_unalign_post %arg40[%c0_34] : memref<256xui16, 1>
                        }
                        ascvec.mem_bar vst_vld
                        %c0_43 = arith.constant 0 : index
                        %c1_44 = arith.constant 1 : index
                        scf.for %arg45 = %c0_43 to %c1_44 step %c1_44 {
                          %293 = ascvec.loadalign_brc %arg30[%arg33] : memref<256xi32, 1>, index -> <i32>
                          %294 = ascvec.reinterpret %293 : <i32> -> <ui16>
                          %295 = ascvec.duplicate %294, %mask_38 : <ui16> -> <ui16>
                          %296 = ascvec.loadalign_brc %arg34[%c0_34] : memref<256xui16, 1>, index -> <ui16>
                          %297 = ascvec.reinterpret %296 : <ui16> -> <ui32>
                          %298 = ascvec.and %297, %291, %mask_40 : <ui32>, <ui32> -> <ui32>
                          %299 = ascvec.reinterpret %298 : <ui32> -> <i32>
                          ascvec.storealign_1st %arg30[%arg41], %299 : memref<256xi32, 1>, index, <i32>
                          %300 = ascvec.cmp_eq_scalar %296, %c0_i16, %mask_38 : <ui16>, i16 -> !ascvec.mask
                          %301 = ascvec.sub %296, %290, %mask_38 : <ui16>
                          %302 = ascvec.select %289, %301, %300 : <ui16>
                          %303 = ascvec.add %302, %289, %mask_38 : <ui16>
                          %304 = ascvec.gather %arg32[%c0_34], %303, %mask_38 : memref<1024xui16, 1>, <ui16>, !ascvec.mask -> <ui16>
                          %305 = ascvec.select %289, %304, %300 : <ui16>
                          %306 = ascvec.sub %295, %305, %mask_38 : <ui16>
                          %307 = ascvec.reinterpret %306 : <ui16> -> <ui32>
                          %308 = ascvec.and %307, %292, %mask_40 : <ui32>, <ui32> -> <ui32>
                          %309 = ascvec.reinterpret %308 : <ui32> -> <i32>
                          ascvec.storealign_1st %arg30[%arg33], %309 : memref<256xi32, 1>, index, <i32>
                          %310 = ascvec.loadalign_brc %arg30[%arg35] : memref<256xi32, 1>, index -> <i32>
                          %311 = ascvec.reinterpret %310 : <i32> -> <ui16>
                          %312 = ascvec.duplicate %311, %mask_38 : <ui16> -> <ui16>
                          %313 = ascvec.loadalign_brc %arg36[%c0_34] : memref<256xui16, 1>, index -> <ui16>
                          %314 = ascvec.reinterpret %313 : <ui16> -> <ui32>
                          %315 = ascvec.and %314, %291, %mask_40 : <ui32>, <ui32> -> <ui32>
                          %316 = ascvec.reinterpret %315 : <ui32> -> <i32>
                          ascvec.storealign_1st %arg30[%arg42], %316 : memref<256xi32, 1>, index, <i32>
                          %317 = ascvec.cmp_eq_scalar %313, %c0_i16, %mask_38 : <ui16>, i16 -> !ascvec.mask
                          %318 = ascvec.sub %313, %290, %mask_38 : <ui16>
                          %319 = ascvec.select %289, %318, %317 : <ui16>
                          %320 = ascvec.duplicate_scalar %c256_i16, %mask_38 : i16 -> <ui16>
                          %321 = ascvec.add %319, %320, %mask_38 : <ui16>
                          %322 = ascvec.gather %arg32[%c0_34], %321, %mask_38 : memref<1024xui16, 1>, <ui16>, !ascvec.mask -> <ui16>
                          %323 = ascvec.select %289, %322, %317 : <ui16>
                          %324 = ascvec.sub %312, %323, %mask_38 : <ui16>
                          %325 = ascvec.reinterpret %324 : <ui16> -> <ui32>
                          %326 = ascvec.and %325, %292, %mask_40 : <ui32>, <ui32> -> <ui32>
                          %327 = ascvec.reinterpret %326 : <ui32> -> <i32>
                          ascvec.storealign_1st %arg30[%arg35], %327 : memref<256xi32, 1>, index, <i32>
                          %328 = ascvec.loadalign_brc %arg30[%arg37] : memref<256xi32, 1>, index -> <i32>
                          %329 = ascvec.reinterpret %328 : <i32> -> <ui16>
                          %330 = ascvec.duplicate %329, %mask_38 : <ui16> -> <ui16>
                          %331 = ascvec.loadalign_brc %arg38[%c0_34] : memref<256xui16, 1>, index -> <ui16>
                          %332 = ascvec.reinterpret %331 : <ui16> -> <ui32>
                          %333 = ascvec.and %332, %291, %mask_40 : <ui32>, <ui32> -> <ui32>
                          %334 = ascvec.reinterpret %333 : <ui32> -> <i32>
                          ascvec.storealign_1st %arg30[%arg43], %334 : memref<256xi32, 1>, index, <i32>
                          %335 = ascvec.cmp_eq_scalar %331, %c0_i16, %mask_38 : <ui16>, i16 -> !ascvec.mask
                          %336 = ascvec.sub %331, %290, %mask_38 : <ui16>
                          %337 = ascvec.select %289, %336, %335 : <ui16>
                          %338 = ascvec.duplicate_scalar %c512_i16, %mask_38 : i16 -> <ui16>
                          %339 = ascvec.add %337, %338, %mask_38 : <ui16>
                          %340 = ascvec.gather %arg32[%c0_34], %339, %mask_38 : memref<1024xui16, 1>, <ui16>, !ascvec.mask -> <ui16>
                          %341 = ascvec.select %289, %340, %335 : <ui16>
                          %342 = ascvec.sub %330, %341, %mask_38 : <ui16>
                          %343 = ascvec.reinterpret %342 : <ui16> -> <ui32>
                          %344 = ascvec.and %343, %292, %mask_40 : <ui32>, <ui32> -> <ui32>
                          %345 = ascvec.reinterpret %344 : <ui32> -> <i32>
                          ascvec.storealign_1st %arg30[%arg37], %345 : memref<256xi32, 1>, index, <i32>
                          %346 = ascvec.loadalign_brc %arg30[%arg39] : memref<256xi32, 1>, index -> <i32>
                          %347 = ascvec.reinterpret %346 : <i32> -> <ui16>
                          %348 = ascvec.duplicate %347, %mask_38 : <ui16> -> <ui16>
                          %349 = ascvec.loadalign_brc %arg40[%c0_34] : memref<256xui16, 1>, index -> <ui16>
                          %350 = ascvec.reinterpret %349 : <ui16> -> <ui32>
                          %351 = ascvec.and %350, %291, %mask_40 : <ui32>, <ui32> -> <ui32>
                          %352 = ascvec.reinterpret %351 : <ui32> -> <i32>
                          ascvec.storealign_1st %arg30[%arg44], %352 : memref<256xi32, 1>, index, <i32>
                          %353 = ascvec.cmp_eq_scalar %349, %c0_i16, %mask_38 : <ui16>, i16 -> !ascvec.mask
                          %354 = ascvec.sub %349, %290, %mask_38 : <ui16>
                          %355 = ascvec.select %289, %354, %353 : <ui16>
                          %356 = ascvec.duplicate_scalar %c768_i16, %mask_38 : i16 -> <ui16>
                          %357 = ascvec.add %355, %356, %mask_38 : <ui16>
                          %358 = ascvec.gather %arg32[%c0_34], %357, %mask_38 : memref<1024xui16, 1>, <ui16>, !ascvec.mask -> <ui16>
                          %359 = ascvec.select %289, %358, %353 : <ui16>
                          %360 = ascvec.sub %348, %359, %mask_38 : <ui16>
                          %361 = ascvec.reinterpret %360 : <ui16> -> <ui32>
                          %362 = ascvec.and %361, %292, %mask_40 : <ui32>, <ui32> -> <ui32>
                          %363 = ascvec.reinterpret %362 : <ui32> -> <i32>
                          ascvec.storealign_1st %arg30[%arg39], %363 : memref<256xi32, 1>, index, <i32>
                        }
                      } {cannir.vf_mode = "simd"}
                      %240 = arith.addi %212, %c3_i32 overflow<nsw> : i32
                      %241 = arith.index_cast %240 : i32 to index
                      %242 = arith.addi %217, %c3_i32 overflow<nsw> : i32
                      %243 = arith.index_cast %242 : i32 to index
                      %244 = arith.addi %222, %c3_i32 overflow<nsw> : i32
                      %245 = arith.index_cast %244 : i32 to index
                      %246 = arith.addi %227, %c3_i32 overflow<nsw> : i32
                      %247 = arith.index_cast %246 : i32 to index
                      ascvec.vf(%196, %alloc_17, %alloc_14, %alloc_11, %alloc_19, %216, %alloc_20, %221, %alloc_21, %226, %alloc_22, %231, %alloc_23, %241, %243, %245, %247 : i32, memref<8192xui32, 1>, memref<256xi32, 1>, memref<8192xf32, 1>, memref<1024xui16, 1>, index, memref<256xui16, 1>, index, memref<256xui16, 1>, index, memref<256xui16, 1>, index, memref<256xui16, 1>, index, index, index, index) {
                      ^bb0(%arg28: i32, %arg29: memref<8192xui32, 1>, %arg30: memref<256xi32, 1>, %arg31: memref<8192xf32, 1>, %arg32: memref<1024xui16, 1>, %arg33: index, %arg34: memref<256xui16, 1>, %arg35: index, %arg36: memref<256xui16, 1>, %arg37: index, %arg38: memref<256xui16, 1>, %arg39: index, %arg40: memref<256xui16, 1>, %arg41: index, %arg42: index, %arg43: index, %arg44: index):
                        %c896 = arith.constant 896 : index
                        %c768 = arith.constant 768 : index
                        %c640 = arith.constant 640 : index
                        %c512 = arith.constant 512 : index
                        %c384 = arith.constant 384 : index
                        %c256 = arith.constant 256 : index
                        %c128 = arith.constant 128 : index
                        %c768_i16 = arith.constant 768 : i16
                        %c512_i16 = arith.constant 512 : i16
                        %c256_i16 = arith.constant 256 : i16
                        %c128_i16 = arith.constant 128 : i16
                        %c4_i32_30 = arith.constant 4 : i32
                        %c0_i32_31 = arith.constant 0 : i32
                        %c2_i32_32 = arith.constant 2 : i32
                        %c1_i32_33 = arith.constant 1 : i32
                        %c8_i32_34 = arith.constant 8 : i32
                        %c0_35 = arith.constant 0 : index
                        %c65535_i32 = arith.constant 65535 : i32
                        %c255_i32 = arith.constant 255 : i32
                        %c1_i16 = arith.constant 1 : i16
                        %c0_i16 = arith.constant 0 : i16
                        %c64_i32_36 = arith.constant 64 : i32
                        %c128_i32_37 = arith.constant 128 : i32
                        %c256_i32_38 = arith.constant 256 : i32
                        %mask, %remaining_out = ascvec.update_mask %c256_i32_38 elem_bits 8 : !ascvec.mask
                        %mask_39, %remaining_out_40 = ascvec.update_mask %c128_i32_37 elem_bits 16 : !ascvec.mask
                        %mask_41, %remaining_out_42 = ascvec.update_mask %c64_i32_36 : !ascvec.mask
                        %289 = ascvec.duplicate_scalar %c0_i16, %mask_39 : i16 -> <ui16>
                        %290 = ascvec.duplicate_scalar %c1_i16, %mask_39 : i16 -> <ui16>
                        %291 = ascvec.duplicate_scalar %c255_i32, %mask_41 : i32 -> <ui32>
                        %292 = ascvec.duplicate_scalar %c65535_i32, %mask_41 : i32 -> <ui32>
                        scf.for %arg45 = %c0_i32_31 to %c4_i32_30 step %c1_i32_33  : i32 {
                          %293 = arith.addi %arg28, %arg45 : i32
                          %294 = arith.muli %293, %c256_i32_38 : i32
                          %295 = arith.index_cast %294 : i32 to index
                          %result0, %result1 = ascvec.loadalign_deintlv %arg29[%295] {dintlv_width = "b16"} : memref<8192xui32, 1>, index -> <ui16>, <ui16>
                          %296 = arith.addi %294, %c128_i32_37 : i32
                          %297 = arith.index_cast %296 : i32 to index
                          %result0_46, %result1_47 = ascvec.loadalign_deintlv %arg29[%297] {dintlv_width = "b16"} : memref<8192xui32, 1>, index -> <ui16>, <ui16>
                          %298 = ascvec.reinterpret %result1 : <ui16> -> <ui8>
                          %299 = ascvec.reinterpret %result1_47 : <ui16> -> <ui8>
                          %result0_48, %result1_49 = ascvec.deintlv %298, %299 : <ui8>, <ui8> -> <ui8>, <ui8>
                          %300 = ascvec.reinterpret %result0 : <ui16> -> <ui8>
                          %301 = ascvec.reinterpret %result0_46 : <ui16> -> <ui8>
                          %result0_50, %result1_51 = ascvec.deintlv %300, %301 : <ui8>, <ui8> -> <ui8>, <ui8>
                          %302 = arith.muli %293, %c8_i32_34 : i32
                          %303 = arith.addi %302, %c1_i32_33 : i32
                          %304 = arith.index_cast %303 : i32 to index
                          %305 = ascvec.loadalign_brc %arg30[%304] {brc_width = "b8"} : memref<256xi32, 1>, index -> <ui32>
                          %306 = ascvec.reinterpret %305 : <ui32> -> <ui8>
                          %307 = ascvec.cmp_eq %result1_49, %306, %mask : <ui8>, <ui8> -> !ascvec.mask
                          %308 = arith.addi %302, %c2_i32_32 : i32
                          %309 = arith.index_cast %308 : i32 to index
                          %310 = ascvec.loadalign_brc %arg30[%309] {brc_width = "b8"} : memref<256xi32, 1>, index -> <ui32>
                          %311 = ascvec.reinterpret %310 : <ui32> -> <ui8>
                          %312 = ascvec.cmp_eq %result0_48, %311, %307 : <ui8>, <ui8> -> !ascvec.mask
                          %313 = ascvec.cumulative_histogram %289, %result1_51, %312 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          %314 = ascvec.cumulative_histogram %289, %result1_51, %312 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          %result0_52, %result1_53 = ascvec.loadalign_deintlv %arg31[%295] {dintlv_width = "b16"} : memref<8192xf32, 1>, index -> <ui16>, <ui16>
                          %result0_54, %result1_55 = ascvec.loadalign_deintlv %arg31[%297] {dintlv_width = "b16"} : memref<8192xf32, 1>, index -> <ui16>, <ui16>
                          %315 = ascvec.reinterpret %result1_53 : <ui16> -> <ui8>
                          %316 = ascvec.reinterpret %result1_55 : <ui16> -> <ui8>
                          %result0_56, %result1_57 = ascvec.deintlv %315, %316 : <ui8>, <ui8> -> <ui8>, <ui8>
                          %317 = ascvec.reinterpret %result0_52 : <ui16> -> <ui8>
                          %318 = ascvec.reinterpret %result0_54 : <ui16> -> <ui8>
                          %result0_58, %result1_59 = ascvec.deintlv %317, %318 : <ui8>, <ui8> -> <ui8>, <ui8>
                          %319 = ascvec.cmp_eq %result1_57, %306, %mask : <ui8>, <ui8> -> !ascvec.mask
                          %320 = ascvec.cmp_eq %result0_56, %311, %319 : <ui8>, <ui8> -> !ascvec.mask
                          %321 = ascvec.cumulative_histogram %313, %result1_59, %320 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          %322 = ascvec.cumulative_histogram %314, %result1_59, %320 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          %323 = arith.muli %arg45, %c256_i32_38 overflow<nsw> : i32
                          %324 = arith.index_cast %323 : i32 to index
                          ascvec.storealign %arg32[%324], %321, %mask_39 : memref<1024xui16, 1>, index, <ui16>
                          %325 = arith.addi %323, %c128_i32_37 overflow<nsw> : i32
                          %326 = arith.index_cast %325 : i32 to index
                          ascvec.storealign %arg32[%326], %322, %mask_39 : memref<1024xui16, 1>, index, <ui16>
                        }
                        ascvec.mem_bar vst_vld
                        %c0_43 = arith.constant 0 : index
                        %c1 = arith.constant 1 : index
                        scf.for %arg45 = %c0_43 to %c1 step %c1 {
                          %293 = ascvec.loadalign_brc %arg30[%arg33] : memref<256xi32, 1>, index -> <i32>
                          %294 = ascvec.reinterpret %293 : <i32> -> <ui16>
                          %295 = ascvec.duplicate %294, %mask_39 : <ui16> -> <ui16>
                          ascvec.store_unalign_begin %arg34 : memref<256xui16, 1>
                          %296 = ascvec.arange %c0_i16 : i16 -> <ui16>
                          %297 = ascvec.loadalign %arg32[%c0_35] : memref<1024xui16, 1>, index -> <ui16>
                          %298 = ascvec.cmp_ge %297, %295, %mask_39 : <ui16>, <ui16> -> !ascvec.mask
                          %299 = ascvec.squeeze_store_reg %296, %298 : <ui16>, !ascvec.mask -> <ui16>
                          ascvec.store_unalign %arg34[%c0_35], %299 : memref<256xui16, 1>, <ui16>
                          %300 = ascvec.arange %c128_i16 : i16 -> <ui16>
                          %301 = ascvec.loadalign %arg32[%c128] : memref<1024xui16, 1>, index -> <ui16>
                          %302 = ascvec.cmp_ge %301, %295, %mask_39 : <ui16>, <ui16> -> !ascvec.mask
                          %303 = ascvec.squeeze_store_reg %300, %302 : <ui16>, !ascvec.mask -> <ui16>
                          ascvec.store_unalign %arg34[%c0_35], %303 : memref<256xui16, 1>, <ui16>
                          ascvec.store_unalign_post %arg34[%c0_35] : memref<256xui16, 1>
                          %304 = ascvec.loadalign_brc %arg30[%arg35] : memref<256xi32, 1>, index -> <i32>
                          %305 = ascvec.reinterpret %304 : <i32> -> <ui16>
                          %306 = ascvec.duplicate %305, %mask_39 : <ui16> -> <ui16>
                          ascvec.store_unalign_begin %arg36 : memref<256xui16, 1>
                          %307 = ascvec.loadalign %arg32[%c256] : memref<1024xui16, 1>, index -> <ui16>
                          %308 = ascvec.cmp_ge %307, %306, %mask_39 : <ui16>, <ui16> -> !ascvec.mask
                          %309 = ascvec.squeeze_store_reg %296, %308 : <ui16>, !ascvec.mask -> <ui16>
                          ascvec.store_unalign %arg36[%c0_35], %309 : memref<256xui16, 1>, <ui16>
                          %310 = ascvec.loadalign %arg32[%c384] : memref<1024xui16, 1>, index -> <ui16>
                          %311 = ascvec.cmp_ge %310, %306, %mask_39 : <ui16>, <ui16> -> !ascvec.mask
                          %312 = ascvec.squeeze_store_reg %300, %311 : <ui16>, !ascvec.mask -> <ui16>
                          ascvec.store_unalign %arg36[%c0_35], %312 : memref<256xui16, 1>, <ui16>
                          ascvec.store_unalign_post %arg36[%c0_35] : memref<256xui16, 1>
                          %313 = ascvec.loadalign_brc %arg30[%arg37] : memref<256xi32, 1>, index -> <i32>
                          %314 = ascvec.reinterpret %313 : <i32> -> <ui16>
                          %315 = ascvec.duplicate %314, %mask_39 : <ui16> -> <ui16>
                          ascvec.store_unalign_begin %arg38 : memref<256xui16, 1>
                          %316 = ascvec.loadalign %arg32[%c512] : memref<1024xui16, 1>, index -> <ui16>
                          %317 = ascvec.cmp_ge %316, %315, %mask_39 : <ui16>, <ui16> -> !ascvec.mask
                          %318 = ascvec.squeeze_store_reg %296, %317 : <ui16>, !ascvec.mask -> <ui16>
                          ascvec.store_unalign %arg38[%c0_35], %318 : memref<256xui16, 1>, <ui16>
                          %319 = ascvec.loadalign %arg32[%c640] : memref<1024xui16, 1>, index -> <ui16>
                          %320 = ascvec.cmp_ge %319, %315, %mask_39 : <ui16>, <ui16> -> !ascvec.mask
                          %321 = ascvec.squeeze_store_reg %300, %320 : <ui16>, !ascvec.mask -> <ui16>
                          ascvec.store_unalign %arg38[%c0_35], %321 : memref<256xui16, 1>, <ui16>
                          ascvec.store_unalign_post %arg38[%c0_35] : memref<256xui16, 1>
                          %322 = ascvec.loadalign_brc %arg30[%arg39] : memref<256xi32, 1>, index -> <i32>
                          %323 = ascvec.reinterpret %322 : <i32> -> <ui16>
                          %324 = ascvec.duplicate %323, %mask_39 : <ui16> -> <ui16>
                          ascvec.store_unalign_begin %arg40 : memref<256xui16, 1>
                          %325 = ascvec.loadalign %arg32[%c768] : memref<1024xui16, 1>, index -> <ui16>
                          %326 = ascvec.cmp_ge %325, %324, %mask_39 : <ui16>, <ui16> -> !ascvec.mask
                          %327 = ascvec.squeeze_store_reg %296, %326 : <ui16>, !ascvec.mask -> <ui16>
                          ascvec.store_unalign %arg40[%c0_35], %327 : memref<256xui16, 1>, <ui16>
                          %328 = ascvec.loadalign %arg32[%c896] : memref<1024xui16, 1>, index -> <ui16>
                          %329 = ascvec.cmp_ge %328, %324, %mask_39 : <ui16>, <ui16> -> !ascvec.mask
                          %330 = ascvec.squeeze_store_reg %300, %329 : <ui16>, !ascvec.mask -> <ui16>
                          ascvec.store_unalign %arg40[%c0_35], %330 : memref<256xui16, 1>, <ui16>
                          ascvec.store_unalign_post %arg40[%c0_35] : memref<256xui16, 1>
                        }
                        ascvec.mem_bar vst_vld
                        %c0_44 = arith.constant 0 : index
                        %c1_45 = arith.constant 1 : index
                        scf.for %arg45 = %c0_44 to %c1_45 step %c1_45 {
                          %293 = ascvec.loadalign_brc %arg30[%arg33] : memref<256xi32, 1>, index -> <i32>
                          %294 = ascvec.reinterpret %293 : <i32> -> <ui16>
                          %295 = ascvec.duplicate %294, %mask_39 : <ui16> -> <ui16>
                          %296 = ascvec.loadalign_brc %arg34[%c0_35] : memref<256xui16, 1>, index -> <ui16>
                          %297 = ascvec.reinterpret %296 : <ui16> -> <ui32>
                          %298 = ascvec.and %297, %291, %mask_41 : <ui32>, <ui32> -> <ui32>
                          %299 = ascvec.reinterpret %298 : <ui32> -> <i32>
                          ascvec.storealign_1st %arg30[%arg41], %299 : memref<256xi32, 1>, index, <i32>
                          %300 = ascvec.cmp_eq_scalar %296, %c0_i16, %mask_39 : <ui16>, i16 -> !ascvec.mask
                          %301 = ascvec.sub %296, %290, %mask_39 : <ui16>
                          %302 = ascvec.select %289, %301, %300 : <ui16>
                          %303 = ascvec.add %302, %289, %mask_39 : <ui16>
                          %304 = ascvec.gather %arg32[%c0_35], %303, %mask_39 : memref<1024xui16, 1>, <ui16>, !ascvec.mask -> <ui16>
                          %305 = ascvec.select %289, %304, %300 : <ui16>
                          %306 = ascvec.sub %295, %305, %mask_39 : <ui16>
                          %307 = ascvec.reinterpret %306 : <ui16> -> <ui32>
                          %308 = ascvec.and %307, %292, %mask_41 : <ui32>, <ui32> -> <ui32>
                          %309 = ascvec.reinterpret %308 : <ui32> -> <i32>
                          ascvec.storealign_1st %arg30[%arg33], %309 : memref<256xi32, 1>, index, <i32>
                          %310 = ascvec.loadalign_brc %arg30[%arg35] : memref<256xi32, 1>, index -> <i32>
                          %311 = ascvec.reinterpret %310 : <i32> -> <ui16>
                          %312 = ascvec.duplicate %311, %mask_39 : <ui16> -> <ui16>
                          %313 = ascvec.loadalign_brc %arg36[%c0_35] : memref<256xui16, 1>, index -> <ui16>
                          %314 = ascvec.reinterpret %313 : <ui16> -> <ui32>
                          %315 = ascvec.and %314, %291, %mask_41 : <ui32>, <ui32> -> <ui32>
                          %316 = ascvec.reinterpret %315 : <ui32> -> <i32>
                          ascvec.storealign_1st %arg30[%arg42], %316 : memref<256xi32, 1>, index, <i32>
                          %317 = ascvec.cmp_eq_scalar %313, %c0_i16, %mask_39 : <ui16>, i16 -> !ascvec.mask
                          %318 = ascvec.sub %313, %290, %mask_39 : <ui16>
                          %319 = ascvec.select %289, %318, %317 : <ui16>
                          %320 = ascvec.duplicate_scalar %c256_i16, %mask_39 : i16 -> <ui16>
                          %321 = ascvec.add %319, %320, %mask_39 : <ui16>
                          %322 = ascvec.gather %arg32[%c0_35], %321, %mask_39 : memref<1024xui16, 1>, <ui16>, !ascvec.mask -> <ui16>
                          %323 = ascvec.select %289, %322, %317 : <ui16>
                          %324 = ascvec.sub %312, %323, %mask_39 : <ui16>
                          %325 = ascvec.reinterpret %324 : <ui16> -> <ui32>
                          %326 = ascvec.and %325, %292, %mask_41 : <ui32>, <ui32> -> <ui32>
                          %327 = ascvec.reinterpret %326 : <ui32> -> <i32>
                          ascvec.storealign_1st %arg30[%arg35], %327 : memref<256xi32, 1>, index, <i32>
                          %328 = ascvec.loadalign_brc %arg30[%arg37] : memref<256xi32, 1>, index -> <i32>
                          %329 = ascvec.reinterpret %328 : <i32> -> <ui16>
                          %330 = ascvec.duplicate %329, %mask_39 : <ui16> -> <ui16>
                          %331 = ascvec.loadalign_brc %arg38[%c0_35] : memref<256xui16, 1>, index -> <ui16>
                          %332 = ascvec.reinterpret %331 : <ui16> -> <ui32>
                          %333 = ascvec.and %332, %291, %mask_41 : <ui32>, <ui32> -> <ui32>
                          %334 = ascvec.reinterpret %333 : <ui32> -> <i32>
                          ascvec.storealign_1st %arg30[%arg43], %334 : memref<256xi32, 1>, index, <i32>
                          %335 = ascvec.cmp_eq_scalar %331, %c0_i16, %mask_39 : <ui16>, i16 -> !ascvec.mask
                          %336 = ascvec.sub %331, %290, %mask_39 : <ui16>
                          %337 = ascvec.select %289, %336, %335 : <ui16>
                          %338 = ascvec.duplicate_scalar %c512_i16, %mask_39 : i16 -> <ui16>
                          %339 = ascvec.add %337, %338, %mask_39 : <ui16>
                          %340 = ascvec.gather %arg32[%c0_35], %339, %mask_39 : memref<1024xui16, 1>, <ui16>, !ascvec.mask -> <ui16>
                          %341 = ascvec.select %289, %340, %335 : <ui16>
                          %342 = ascvec.sub %330, %341, %mask_39 : <ui16>
                          %343 = ascvec.reinterpret %342 : <ui16> -> <ui32>
                          %344 = ascvec.and %343, %292, %mask_41 : <ui32>, <ui32> -> <ui32>
                          %345 = ascvec.reinterpret %344 : <ui32> -> <i32>
                          ascvec.storealign_1st %arg30[%arg37], %345 : memref<256xi32, 1>, index, <i32>
                          %346 = ascvec.loadalign_brc %arg30[%arg39] : memref<256xi32, 1>, index -> <i32>
                          %347 = ascvec.reinterpret %346 : <i32> -> <ui16>
                          %348 = ascvec.duplicate %347, %mask_39 : <ui16> -> <ui16>
                          %349 = ascvec.loadalign_brc %arg40[%c0_35] : memref<256xui16, 1>, index -> <ui16>
                          %350 = ascvec.reinterpret %349 : <ui16> -> <ui32>
                          %351 = ascvec.and %350, %291, %mask_41 : <ui32>, <ui32> -> <ui32>
                          %352 = ascvec.reinterpret %351 : <ui32> -> <i32>
                          ascvec.storealign_1st %arg30[%arg44], %352 : memref<256xi32, 1>, index, <i32>
                          %353 = ascvec.cmp_eq_scalar %349, %c0_i16, %mask_39 : <ui16>, i16 -> !ascvec.mask
                          %354 = ascvec.sub %349, %290, %mask_39 : <ui16>
                          %355 = ascvec.select %289, %354, %353 : <ui16>
                          %356 = ascvec.duplicate_scalar %c768_i16, %mask_39 : i16 -> <ui16>
                          %357 = ascvec.add %355, %356, %mask_39 : <ui16>
                          %358 = ascvec.gather %arg32[%c0_35], %357, %mask_39 : memref<1024xui16, 1>, <ui16>, !ascvec.mask -> <ui16>
                          %359 = ascvec.select %289, %358, %353 : <ui16>
                          %360 = ascvec.sub %348, %359, %mask_39 : <ui16>
                          %361 = ascvec.reinterpret %360 : <ui16> -> <ui32>
                          %362 = ascvec.and %361, %292, %mask_41 : <ui32>, <ui32> -> <ui32>
                          %363 = ascvec.reinterpret %362 : <ui32> -> <i32>
                          ascvec.storealign_1st %arg30[%arg39], %363 : memref<256xi32, 1>, index, <i32>
                        }
                      } {cannir.vf_mode = "simd"}
                      %248 = arith.addi %212, %c4_i32 overflow<nsw> : i32
                      %249 = arith.index_cast %248 : i32 to index
                      %250 = arith.addi %217, %c4_i32 overflow<nsw> : i32
                      %251 = arith.index_cast %250 : i32 to index
                      %252 = arith.addi %222, %c4_i32 overflow<nsw> : i32
                      %253 = arith.index_cast %252 : i32 to index
                      %254 = arith.addi %227, %c4_i32 overflow<nsw> : i32
                      %255 = arith.index_cast %254 : i32 to index
                      ascvec.vf(%196, %alloc_17, %alloc_14, %alloc_11, %alloc_19, %216, %alloc_20, %221, %alloc_21, %226, %alloc_22, %231, %alloc_23, %249, %251, %253, %255 : i32, memref<8192xui32, 1>, memref<256xi32, 1>, memref<8192xf32, 1>, memref<1024xui16, 1>, index, memref<256xui16, 1>, index, memref<256xui16, 1>, index, memref<256xui16, 1>, index, memref<256xui16, 1>, index, index, index, index) {
                      ^bb0(%arg28: i32, %arg29: memref<8192xui32, 1>, %arg30: memref<256xi32, 1>, %arg31: memref<8192xf32, 1>, %arg32: memref<1024xui16, 1>, %arg33: index, %arg34: memref<256xui16, 1>, %arg35: index, %arg36: memref<256xui16, 1>, %arg37: index, %arg38: memref<256xui16, 1>, %arg39: index, %arg40: memref<256xui16, 1>, %arg41: index, %arg42: index, %arg43: index, %arg44: index):
                        %c896 = arith.constant 896 : index
                        %c768 = arith.constant 768 : index
                        %c640 = arith.constant 640 : index
                        %c512 = arith.constant 512 : index
                        %c384 = arith.constant 384 : index
                        %c256 = arith.constant 256 : index
                        %c128 = arith.constant 128 : index
                        %c768_i16 = arith.constant 768 : i16
                        %c512_i16 = arith.constant 512 : i16
                        %c256_i16 = arith.constant 256 : i16
                        %c128_i16 = arith.constant 128 : i16
                        %c4_i32_30 = arith.constant 4 : i32
                        %c0_i32_31 = arith.constant 0 : i32
                        %c3_i32_32 = arith.constant 3 : i32
                        %c2_i32_33 = arith.constant 2 : i32
                        %c1_i32_34 = arith.constant 1 : i32
                        %c8_i32_35 = arith.constant 8 : i32
                        %c0_36 = arith.constant 0 : index
                        %c65535_i32 = arith.constant 65535 : i32
                        %c255_i32 = arith.constant 255 : i32
                        %c1_i16 = arith.constant 1 : i16
                        %c0_i16 = arith.constant 0 : i16
                        %c64_i32_37 = arith.constant 64 : i32
                        %c128_i32_38 = arith.constant 128 : i32
                        %c256_i32_39 = arith.constant 256 : i32
                        %mask, %remaining_out = ascvec.update_mask %c256_i32_39 elem_bits 8 : !ascvec.mask
                        %mask_40, %remaining_out_41 = ascvec.update_mask %c128_i32_38 elem_bits 16 : !ascvec.mask
                        %mask_42, %remaining_out_43 = ascvec.update_mask %c64_i32_37 : !ascvec.mask
                        %289 = ascvec.duplicate_scalar %c0_i16, %mask_40 : i16 -> <ui16>
                        %290 = ascvec.duplicate_scalar %c1_i16, %mask_40 : i16 -> <ui16>
                        %291 = ascvec.duplicate_scalar %c255_i32, %mask_42 : i32 -> <ui32>
                        %292 = ascvec.duplicate_scalar %c65535_i32, %mask_42 : i32 -> <ui32>
                        scf.for %arg45 = %c0_i32_31 to %c4_i32_30 step %c1_i32_34  : i32 {
                          %293 = arith.addi %arg28, %arg45 : i32
                          %294 = arith.muli %293, %c256_i32_39 : i32
                          %295 = arith.index_cast %294 : i32 to index
                          %result0, %result1 = ascvec.loadalign_deintlv %arg29[%295] {dintlv_width = "b16"} : memref<8192xui32, 1>, index -> <ui16>, <ui16>
                          %296 = arith.addi %294, %c128_i32_38 : i32
                          %297 = arith.index_cast %296 : i32 to index
                          %result0_47, %result1_48 = ascvec.loadalign_deintlv %arg29[%297] {dintlv_width = "b16"} : memref<8192xui32, 1>, index -> <ui16>, <ui16>
                          %298 = ascvec.reinterpret %result1 : <ui16> -> <ui8>
                          %299 = ascvec.reinterpret %result1_48 : <ui16> -> <ui8>
                          %result0_49, %result1_50 = ascvec.deintlv %298, %299 : <ui8>, <ui8> -> <ui8>, <ui8>
                          %300 = ascvec.reinterpret %result0 : <ui16> -> <ui8>
                          %301 = ascvec.reinterpret %result0_47 : <ui16> -> <ui8>
                          %result0_51, %result1_52 = ascvec.deintlv %300, %301 : <ui8>, <ui8> -> <ui8>, <ui8>
                          %302 = arith.muli %293, %c8_i32_35 : i32
                          %303 = arith.addi %302, %c1_i32_34 : i32
                          %304 = arith.index_cast %303 : i32 to index
                          %305 = ascvec.loadalign_brc %arg30[%304] {brc_width = "b8"} : memref<256xi32, 1>, index -> <ui32>
                          %306 = ascvec.reinterpret %305 : <ui32> -> <ui8>
                          %307 = ascvec.cmp_eq %result1_50, %306, %mask : <ui8>, <ui8> -> !ascvec.mask
                          %308 = arith.addi %302, %c2_i32_33 : i32
                          %309 = arith.index_cast %308 : i32 to index
                          %310 = ascvec.loadalign_brc %arg30[%309] {brc_width = "b8"} : memref<256xi32, 1>, index -> <ui32>
                          %311 = ascvec.reinterpret %310 : <ui32> -> <ui8>
                          %312 = ascvec.cmp_eq %result0_49, %311, %307 : <ui8>, <ui8> -> !ascvec.mask
                          %313 = arith.addi %302, %c3_i32_32 : i32
                          %314 = arith.index_cast %313 : i32 to index
                          %315 = ascvec.loadalign_brc %arg30[%314] {brc_width = "b8"} : memref<256xi32, 1>, index -> <ui32>
                          %316 = ascvec.reinterpret %315 : <ui32> -> <ui8>
                          %317 = ascvec.cmp_eq %result1_52, %316, %312 : <ui8>, <ui8> -> !ascvec.mask
                          %318 = ascvec.cumulative_histogram %289, %result0_51, %317 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          %319 = ascvec.cumulative_histogram %289, %result0_51, %317 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          %result0_53, %result1_54 = ascvec.loadalign_deintlv %arg31[%295] {dintlv_width = "b16"} : memref<8192xf32, 1>, index -> <ui16>, <ui16>
                          %result0_55, %result1_56 = ascvec.loadalign_deintlv %arg31[%297] {dintlv_width = "b16"} : memref<8192xf32, 1>, index -> <ui16>, <ui16>
                          %320 = ascvec.reinterpret %result1_54 : <ui16> -> <ui8>
                          %321 = ascvec.reinterpret %result1_56 : <ui16> -> <ui8>
                          %result0_57, %result1_58 = ascvec.deintlv %320, %321 : <ui8>, <ui8> -> <ui8>, <ui8>
                          %322 = ascvec.reinterpret %result0_53 : <ui16> -> <ui8>
                          %323 = ascvec.reinterpret %result0_55 : <ui16> -> <ui8>
                          %result0_59, %result1_60 = ascvec.deintlv %322, %323 : <ui8>, <ui8> -> <ui8>, <ui8>
                          %324 = ascvec.cmp_eq %result1_58, %306, %mask : <ui8>, <ui8> -> !ascvec.mask
                          %325 = ascvec.cmp_eq %result0_57, %311, %324 : <ui8>, <ui8> -> !ascvec.mask
                          %326 = ascvec.cmp_eq %result1_60, %316, %325 : <ui8>, <ui8> -> !ascvec.mask
                          %327 = ascvec.cumulative_histogram %318, %result0_59, %326 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          %328 = ascvec.cumulative_histogram %319, %result0_59, %326 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                          %329 = arith.muli %arg45, %c256_i32_39 overflow<nsw> : i32
                          %330 = arith.index_cast %329 : i32 to index
                          ascvec.storealign %arg32[%330], %327, %mask_40 : memref<1024xui16, 1>, index, <ui16>
                          %331 = arith.addi %329, %c128_i32_38 overflow<nsw> : i32
                          %332 = arith.index_cast %331 : i32 to index
                          ascvec.storealign %arg32[%332], %328, %mask_40 : memref<1024xui16, 1>, index, <ui16>
                        }
                        ascvec.mem_bar vst_vld
                        %c0_44 = arith.constant 0 : index
                        %c1 = arith.constant 1 : index
                        scf.for %arg45 = %c0_44 to %c1 step %c1 {
                          %293 = ascvec.loadalign_brc %arg30[%arg33] : memref<256xi32, 1>, index -> <i32>
                          %294 = ascvec.reinterpret %293 : <i32> -> <ui16>
                          %295 = ascvec.duplicate %294, %mask_40 : <ui16> -> <ui16>
                          ascvec.store_unalign_begin %arg34 : memref<256xui16, 1>
                          %296 = ascvec.arange %c0_i16 : i16 -> <ui16>
                          %297 = ascvec.loadalign %arg32[%c0_36] : memref<1024xui16, 1>, index -> <ui16>
                          %298 = ascvec.cmp_ge %297, %295, %mask_40 : <ui16>, <ui16> -> !ascvec.mask
                          %299 = ascvec.squeeze_store_reg %296, %298 : <ui16>, !ascvec.mask -> <ui16>
                          ascvec.store_unalign %arg34[%c0_36], %299 : memref<256xui16, 1>, <ui16>
                          %300 = ascvec.arange %c128_i16 : i16 -> <ui16>
                          %301 = ascvec.loadalign %arg32[%c128] : memref<1024xui16, 1>, index -> <ui16>
                          %302 = ascvec.cmp_ge %301, %295, %mask_40 : <ui16>, <ui16> -> !ascvec.mask
                          %303 = ascvec.squeeze_store_reg %300, %302 : <ui16>, !ascvec.mask -> <ui16>
                          ascvec.store_unalign %arg34[%c0_36], %303 : memref<256xui16, 1>, <ui16>
                          ascvec.store_unalign_post %arg34[%c0_36] : memref<256xui16, 1>
                          %304 = ascvec.loadalign_brc %arg30[%arg35] : memref<256xi32, 1>, index -> <i32>
                          %305 = ascvec.reinterpret %304 : <i32> -> <ui16>
                          %306 = ascvec.duplicate %305, %mask_40 : <ui16> -> <ui16>
                          ascvec.store_unalign_begin %arg36 : memref<256xui16, 1>
                          %307 = ascvec.loadalign %arg32[%c256] : memref<1024xui16, 1>, index -> <ui16>
                          %308 = ascvec.cmp_ge %307, %306, %mask_40 : <ui16>, <ui16> -> !ascvec.mask
                          %309 = ascvec.squeeze_store_reg %296, %308 : <ui16>, !ascvec.mask -> <ui16>
                          ascvec.store_unalign %arg36[%c0_36], %309 : memref<256xui16, 1>, <ui16>
                          %310 = ascvec.loadalign %arg32[%c384] : memref<1024xui16, 1>, index -> <ui16>
                          %311 = ascvec.cmp_ge %310, %306, %mask_40 : <ui16>, <ui16> -> !ascvec.mask
                          %312 = ascvec.squeeze_store_reg %300, %311 : <ui16>, !ascvec.mask -> <ui16>
                          ascvec.store_unalign %arg36[%c0_36], %312 : memref<256xui16, 1>, <ui16>
                          ascvec.store_unalign_post %arg36[%c0_36] : memref<256xui16, 1>
                          %313 = ascvec.loadalign_brc %arg30[%arg37] : memref<256xi32, 1>, index -> <i32>
                          %314 = ascvec.reinterpret %313 : <i32> -> <ui16>
                          %315 = ascvec.duplicate %314, %mask_40 : <ui16> -> <ui16>
                          ascvec.store_unalign_begin %arg38 : memref<256xui16, 1>
                          %316 = ascvec.loadalign %arg32[%c512] : memref<1024xui16, 1>, index -> <ui16>
                          %317 = ascvec.cmp_ge %316, %315, %mask_40 : <ui16>, <ui16> -> !ascvec.mask
                          %318 = ascvec.squeeze_store_reg %296, %317 : <ui16>, !ascvec.mask -> <ui16>
                          ascvec.store_unalign %arg38[%c0_36], %318 : memref<256xui16, 1>, <ui16>
                          %319 = ascvec.loadalign %arg32[%c640] : memref<1024xui16, 1>, index -> <ui16>
                          %320 = ascvec.cmp_ge %319, %315, %mask_40 : <ui16>, <ui16> -> !ascvec.mask
                          %321 = ascvec.squeeze_store_reg %300, %320 : <ui16>, !ascvec.mask -> <ui16>
                          ascvec.store_unalign %arg38[%c0_36], %321 : memref<256xui16, 1>, <ui16>
                          ascvec.store_unalign_post %arg38[%c0_36] : memref<256xui16, 1>
                          %322 = ascvec.loadalign_brc %arg30[%arg39] : memref<256xi32, 1>, index -> <i32>
                          %323 = ascvec.reinterpret %322 : <i32> -> <ui16>
                          %324 = ascvec.duplicate %323, %mask_40 : <ui16> -> <ui16>
                          ascvec.store_unalign_begin %arg40 : memref<256xui16, 1>
                          %325 = ascvec.loadalign %arg32[%c768] : memref<1024xui16, 1>, index -> <ui16>
                          %326 = ascvec.cmp_ge %325, %324, %mask_40 : <ui16>, <ui16> -> !ascvec.mask
                          %327 = ascvec.squeeze_store_reg %296, %326 : <ui16>, !ascvec.mask -> <ui16>
                          ascvec.store_unalign %arg40[%c0_36], %327 : memref<256xui16, 1>, <ui16>
                          %328 = ascvec.loadalign %arg32[%c896] : memref<1024xui16, 1>, index -> <ui16>
                          %329 = ascvec.cmp_ge %328, %324, %mask_40 : <ui16>, <ui16> -> !ascvec.mask
                          %330 = ascvec.squeeze_store_reg %300, %329 : <ui16>, !ascvec.mask -> <ui16>
                          ascvec.store_unalign %arg40[%c0_36], %330 : memref<256xui16, 1>, <ui16>
                          ascvec.store_unalign_post %arg40[%c0_36] : memref<256xui16, 1>
                        }
                        ascvec.mem_bar vst_vld
                        %c0_45 = arith.constant 0 : index
                        %c1_46 = arith.constant 1 : index
                        scf.for %arg45 = %c0_45 to %c1_46 step %c1_46 {
                          %293 = ascvec.loadalign_brc %arg30[%arg33] : memref<256xi32, 1>, index -> <i32>
                          %294 = ascvec.reinterpret %293 : <i32> -> <ui16>
                          %295 = ascvec.duplicate %294, %mask_40 : <ui16> -> <ui16>
                          %296 = ascvec.loadalign_brc %arg34[%c0_36] : memref<256xui16, 1>, index -> <ui16>
                          %297 = ascvec.reinterpret %296 : <ui16> -> <ui32>
                          %298 = ascvec.and %297, %291, %mask_42 : <ui32>, <ui32> -> <ui32>
                          %299 = ascvec.reinterpret %298 : <ui32> -> <i32>
                          ascvec.storealign_1st %arg30[%arg41], %299 : memref<256xi32, 1>, index, <i32>
                          %300 = ascvec.cmp_eq_scalar %296, %c0_i16, %mask_40 : <ui16>, i16 -> !ascvec.mask
                          %301 = ascvec.sub %296, %290, %mask_40 : <ui16>
                          %302 = ascvec.select %289, %301, %300 : <ui16>
                          %303 = ascvec.add %302, %289, %mask_40 : <ui16>
                          %304 = ascvec.gather %arg32[%c0_36], %303, %mask_40 : memref<1024xui16, 1>, <ui16>, !ascvec.mask -> <ui16>
                          %305 = ascvec.select %289, %304, %300 : <ui16>
                          %306 = ascvec.sub %295, %305, %mask_40 : <ui16>
                          %307 = ascvec.reinterpret %306 : <ui16> -> <ui32>
                          %308 = ascvec.and %307, %292, %mask_42 : <ui32>, <ui32> -> <ui32>
                          %309 = ascvec.reinterpret %308 : <ui32> -> <i32>
                          ascvec.storealign_1st %arg30[%arg33], %309 : memref<256xi32, 1>, index, <i32>
                          %310 = ascvec.loadalign_brc %arg30[%arg35] : memref<256xi32, 1>, index -> <i32>
                          %311 = ascvec.reinterpret %310 : <i32> -> <ui16>
                          %312 = ascvec.duplicate %311, %mask_40 : <ui16> -> <ui16>
                          %313 = ascvec.loadalign_brc %arg36[%c0_36] : memref<256xui16, 1>, index -> <ui16>
                          %314 = ascvec.reinterpret %313 : <ui16> -> <ui32>
                          %315 = ascvec.and %314, %291, %mask_42 : <ui32>, <ui32> -> <ui32>
                          %316 = ascvec.reinterpret %315 : <ui32> -> <i32>
                          ascvec.storealign_1st %arg30[%arg42], %316 : memref<256xi32, 1>, index, <i32>
                          %317 = ascvec.cmp_eq_scalar %313, %c0_i16, %mask_40 : <ui16>, i16 -> !ascvec.mask
                          %318 = ascvec.sub %313, %290, %mask_40 : <ui16>
                          %319 = ascvec.select %289, %318, %317 : <ui16>
                          %320 = ascvec.duplicate_scalar %c256_i16, %mask_40 : i16 -> <ui16>
                          %321 = ascvec.add %319, %320, %mask_40 : <ui16>
                          %322 = ascvec.gather %arg32[%c0_36], %321, %mask_40 : memref<1024xui16, 1>, <ui16>, !ascvec.mask -> <ui16>
                          %323 = ascvec.select %289, %322, %317 : <ui16>
                          %324 = ascvec.sub %312, %323, %mask_40 : <ui16>
                          %325 = ascvec.reinterpret %324 : <ui16> -> <ui32>
                          %326 = ascvec.and %325, %292, %mask_42 : <ui32>, <ui32> -> <ui32>
                          %327 = ascvec.reinterpret %326 : <ui32> -> <i32>
                          ascvec.storealign_1st %arg30[%arg35], %327 : memref<256xi32, 1>, index, <i32>
                          %328 = ascvec.loadalign_brc %arg30[%arg37] : memref<256xi32, 1>, index -> <i32>
                          %329 = ascvec.reinterpret %328 : <i32> -> <ui16>
                          %330 = ascvec.duplicate %329, %mask_40 : <ui16> -> <ui16>
                          %331 = ascvec.loadalign_brc %arg38[%c0_36] : memref<256xui16, 1>, index -> <ui16>
                          %332 = ascvec.reinterpret %331 : <ui16> -> <ui32>
                          %333 = ascvec.and %332, %291, %mask_42 : <ui32>, <ui32> -> <ui32>
                          %334 = ascvec.reinterpret %333 : <ui32> -> <i32>
                          ascvec.storealign_1st %arg30[%arg43], %334 : memref<256xi32, 1>, index, <i32>
                          %335 = ascvec.cmp_eq_scalar %331, %c0_i16, %mask_40 : <ui16>, i16 -> !ascvec.mask
                          %336 = ascvec.sub %331, %290, %mask_40 : <ui16>
                          %337 = ascvec.select %289, %336, %335 : <ui16>
                          %338 = ascvec.duplicate_scalar %c512_i16, %mask_40 : i16 -> <ui16>
                          %339 = ascvec.add %337, %338, %mask_40 : <ui16>
                          %340 = ascvec.gather %arg32[%c0_36], %339, %mask_40 : memref<1024xui16, 1>, <ui16>, !ascvec.mask -> <ui16>
                          %341 = ascvec.select %289, %340, %335 : <ui16>
                          %342 = ascvec.sub %330, %341, %mask_40 : <ui16>
                          %343 = ascvec.reinterpret %342 : <ui16> -> <ui32>
                          %344 = ascvec.and %343, %292, %mask_42 : <ui32>, <ui32> -> <ui32>
                          %345 = ascvec.reinterpret %344 : <ui32> -> <i32>
                          ascvec.storealign_1st %arg30[%arg37], %345 : memref<256xi32, 1>, index, <i32>
                          %346 = ascvec.loadalign_brc %arg30[%arg39] : memref<256xi32, 1>, index -> <i32>
                          %347 = ascvec.reinterpret %346 : <i32> -> <ui16>
                          %348 = ascvec.duplicate %347, %mask_40 : <ui16> -> <ui16>
                          %349 = ascvec.loadalign_brc %arg40[%c0_36] : memref<256xui16, 1>, index -> <ui16>
                          %350 = ascvec.reinterpret %349 : <ui16> -> <ui32>
                          %351 = ascvec.and %350, %291, %mask_42 : <ui32>, <ui32> -> <ui32>
                          %352 = ascvec.reinterpret %351 : <ui32> -> <i32>
                          ascvec.storealign_1st %arg30[%arg44], %352 : memref<256xi32, 1>, index, <i32>
                          %353 = ascvec.cmp_eq_scalar %349, %c0_i16, %mask_40 : <ui16>, i16 -> !ascvec.mask
                          %354 = ascvec.sub %349, %290, %mask_40 : <ui16>
                          %355 = ascvec.select %289, %354, %353 : <ui16>
                          %356 = ascvec.duplicate_scalar %c768_i16, %mask_40 : i16 -> <ui16>
                          %357 = ascvec.add %355, %356, %mask_40 : <ui16>
                          %358 = ascvec.gather %arg32[%c0_36], %357, %mask_40 : memref<1024xui16, 1>, <ui16>, !ascvec.mask -> <ui16>
                          %359 = ascvec.select %289, %358, %353 : <ui16>
                          %360 = ascvec.sub %348, %359, %mask_40 : <ui16>
                          %361 = ascvec.reinterpret %360 : <ui16> -> <ui32>
                          %362 = ascvec.and %361, %292, %mask_42 : <ui32>, <ui32> -> <ui32>
                          %363 = ascvec.reinterpret %362 : <ui32> -> <i32>
                          ascvec.storealign_1st %arg30[%arg39], %363 : memref<256xi32, 1>, index, <i32>
                        }
                      } {cannir.vf_mode = "simd"}
                      %256 = arith.muli %196, %c256_i32 overflow<nsw> : i32
                      %257 = arith.index_cast %256 : i32 to index
                      %258 = arith.addi %256, %c64_i32 overflow<nsw> : i32
                      %259 = arith.index_cast %258 : i32 to index
                      %260 = arith.addi %256, %c128_i32 overflow<nsw> : i32
                      %261 = arith.index_cast %260 : i32 to index
                      %262 = arith.addi %256, %c192_i32 overflow<nsw> : i32
                      %263 = arith.index_cast %262 : i32 to index
                      %264 = arith.muli %200, %c256_i32 overflow<nsw> : i32
                      %265 = arith.index_cast %264 : i32 to index
                      %266 = arith.addi %264, %c64_i32 overflow<nsw> : i32
                      %267 = arith.index_cast %266 : i32 to index
                      %268 = arith.addi %264, %c128_i32 overflow<nsw> : i32
                      %269 = arith.index_cast %268 : i32 to index
                      %270 = arith.addi %264, %c192_i32 overflow<nsw> : i32
                      %271 = arith.index_cast %270 : i32 to index
                      %272 = arith.muli %204, %c256_i32 overflow<nsw> : i32
                      %273 = arith.index_cast %272 : i32 to index
                      %274 = arith.addi %272, %c64_i32 overflow<nsw> : i32
                      %275 = arith.index_cast %274 : i32 to index
                      %276 = arith.addi %272, %c128_i32 overflow<nsw> : i32
                      %277 = arith.index_cast %276 : i32 to index
                      %278 = arith.addi %272, %c192_i32 overflow<nsw> : i32
                      %279 = arith.index_cast %278 : i32 to index
                      %280 = arith.muli %208, %c256_i32 overflow<nsw> : i32
                      %281 = arith.index_cast %280 : i32 to index
                      %282 = arith.addi %280, %c64_i32 overflow<nsw> : i32
                      %283 = arith.index_cast %282 : i32 to index
                      %284 = arith.addi %280, %c128_i32 overflow<nsw> : i32
                      %285 = arith.index_cast %284 : i32 to index
                      %286 = arith.addi %280, %c192_i32 overflow<nsw> : i32
                      %287 = arith.index_cast %286 : i32 to index
                      %288 = arith.trunci %166 : i64 to i32
                      ascvec.vf(%alloc_14, %214, %233, %241, %249, %alloc_24, %alloc_17, %257, %259, %261, %263, %alloc_11, %219, %235, %243, %251, %alloc_25, %265, %267, %269, %271, %224, %237, %245, %253, %alloc_26, %273, %275, %277, %279, %229, %239, %247, %255, %alloc_27, %281, %283, %285, %287, %alloc_13, %199, %256, %alloc_18, %288, %alloc_28, %alloc_29, %203, %264, %207, %272, %211, %280, %196 : memref<256xi32, 1>, index, index, index, index, memref<528xi32, 1>, memref<8192xui32, 1>, index, index, index, index, memref<8192xf32, 1>, index, index, index, index, memref<528xi32, 1>, index, index, index, index, index, index, index, index, memref<528xi32, 1>, index, index, index, index, index, index, index, index, memref<528xi32, 1>, index, index, index, index, memref<512xi32, 1>, index, i32, memref<8192xui32, 1>, i32, memref<1024xui32, 1>, memref<1024xui32, 1>, index, i32, index, i32, index, i32, i32) {
                      ^bb0(%arg28: memref<256xi32, 1>, %arg29: index, %arg30: index, %arg31: index, %arg32: index, %arg33: memref<528xi32, 1>, %arg34: memref<8192xui32, 1>, %arg35: index, %arg36: index, %arg37: index, %arg38: index, %arg39: memref<8192xf32, 1>, %arg40: index, %arg41: index, %arg42: index, %arg43: index, %arg44: memref<528xi32, 1>, %arg45: index, %arg46: index, %arg47: index, %arg48: index, %arg49: index, %arg50: index, %arg51: index, %arg52: index, %arg53: memref<528xi32, 1>, %arg54: index, %arg55: index, %arg56: index, %arg57: index, %arg58: index, %arg59: index, %arg60: index, %arg61: index, %arg62: memref<528xi32, 1>, %arg63: index, %arg64: index, %arg65: index, %arg66: index, %arg67: memref<512xi32, 1>, %arg68: index, %arg69: i32, %arg70: memref<8192xui32, 1>, %arg71: i32, %arg72: memref<1024xui32, 1>, %arg73: memref<1024xui32, 1>, %arg74: index, %arg75: i32, %arg76: index, %arg77: i32, %arg78: index, %arg79: i32, %arg80: i32):
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
                        %c1_i32_30 = arith.constant 1 : i32
                        %c4_i32_31 = arith.constant 4 : i32
                        %c-256_i32 = arith.constant -256 : i32
                        %c448_i32 = arith.constant 448 : i32
                        %c384_i32 = arith.constant 384 : i32
                        %c320_i32 = arith.constant 320 : i32
                        %c256_i32_32 = arith.constant 256 : i32
                        %c192_i32_33 = arith.constant 192 : i32
                        %c128_i32_34 = arith.constant 128 : i32
                        %c0_35 = arith.constant 0 : index
                        %c255_i32 = arith.constant 255 : i32
                        %c0_i32_36 = arith.constant 0 : i32
                        %c64_i32_37 = arith.constant 64 : i32
                        %mask, %remaining_out = ascvec.update_mask %c64_i32_37 : !ascvec.mask
                        %289 = ascvec.duplicate_scalar %c0_i32_36, %mask : i32 -> <i32>
                        %290 = ascvec.duplicate_scalar %c0_i32_36, %mask : i32 -> <ui32>
                        %291 = ascvec.duplicate_scalar %c255_i32, %mask : i32 -> <ui32>
                        %292 = ascvec.loadalign_brc %arg28[%arg29] {brc_width = "b8"} : memref<256xi32, 1>, index -> <ui32>
                        %293 = ascvec.and %292, %291, %mask : <ui32>, <ui32> -> <ui32>
                        %294 = ascvec.loadalign_brc %arg28[%arg30] {brc_width = "b8"} : memref<256xi32, 1>, index -> <ui32>
                        %295 = ascvec.and %294, %291, %mask : <ui32>, <ui32> -> <ui32>
                        %296 = ascvec.loadalign_brc %arg28[%arg31] {brc_width = "b8"} : memref<256xi32, 1>, index -> <ui32>
                        %297 = ascvec.and %296, %291, %mask : <ui32>, <ui32> -> <ui32>
                        %298 = ascvec.loadalign_brc %arg28[%arg32] {brc_width = "b8"} : memref<256xi32, 1>, index -> <ui32>
                        %299 = ascvec.and %298, %291, %mask : <ui32>, <ui32> -> <ui32>
                        %300 = ascvec.shl %293, 24, %mask : <ui32> -> <ui32>
                        %301 = ascvec.shl %295, 16, %mask : <ui32> -> <ui32>
                        %302 = ascvec.add %300, %301, %mask : <ui32>
                        %303 = ascvec.shl %297, 8, %mask : <ui32> -> <ui32>
                        %304 = ascvec.add %302, %303, %mask : <ui32>
                        %305 = ascvec.add %304, %299, %mask : <ui32>
                        ascvec.store_unalign_begin %arg33 : memref<528xi32, 1>
                        %306 = ascvec.loadalign %arg34[%arg35] : memref<8192xui32, 1>, index -> <ui32>
                        %307 = ascvec.arange %c0_i32_36 : i32 -> <i32>
                        %308 = ascvec.cmp_gt %306, %305, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %309 = ascvec.squeeze_store_reg %307, %308 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg33[%c0_35], %309 : memref<528xi32, 1>, <i32>
                        %310 = ascvec.loadalign %arg34[%arg36] : memref<8192xui32, 1>, index -> <ui32>
                        %311 = ascvec.arange %c64_i32_37 : i32 -> <i32>
                        %312 = ascvec.cmp_gt %310, %305, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %313 = ascvec.squeeze_store_reg %311, %312 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg33[%c0_35], %313 : memref<528xi32, 1>, <i32>
                        %314 = ascvec.loadalign %arg34[%arg37] : memref<8192xui32, 1>, index -> <ui32>
                        %315 = ascvec.arange %c128_i32_34 : i32 -> <i32>
                        %316 = ascvec.cmp_gt %314, %305, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %317 = ascvec.squeeze_store_reg %315, %316 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg33[%c0_35], %317 : memref<528xi32, 1>, <i32>
                        %318 = ascvec.loadalign %arg34[%arg38] : memref<8192xui32, 1>, index -> <ui32>
                        %319 = ascvec.arange %c192_i32_33 : i32 -> <i32>
                        %320 = ascvec.cmp_gt %318, %305, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %321 = ascvec.squeeze_store_reg %319, %320 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg33[%c0_35], %321 : memref<528xi32, 1>, <i32>
                        %322 = ascvec.loadalign %arg39[%arg35] : memref<8192xf32, 1>, index -> <f32>
                        %323 = ascvec.reinterpret %322 : <f32> -> <ui32>
                        %324 = ascvec.arange %c256_i32_32 : i32 -> <i32>
                        %325 = ascvec.cmp_gt %323, %305, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %326 = ascvec.squeeze_store_reg %324, %325 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg33[%c0_35], %326 : memref<528xi32, 1>, <i32>
                        %327 = ascvec.loadalign %arg39[%arg36] : memref<8192xf32, 1>, index -> <f32>
                        %328 = ascvec.reinterpret %327 : <f32> -> <ui32>
                        %329 = ascvec.arange %c320_i32 : i32 -> <i32>
                        %330 = ascvec.cmp_gt %328, %305, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %331 = ascvec.squeeze_store_reg %329, %330 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg33[%c0_35], %331 : memref<528xi32, 1>, <i32>
                        %332 = ascvec.loadalign %arg39[%arg37] : memref<8192xf32, 1>, index -> <f32>
                        %333 = ascvec.reinterpret %332 : <f32> -> <ui32>
                        %334 = ascvec.arange %c384_i32 : i32 -> <i32>
                        %335 = ascvec.cmp_gt %333, %305, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %336 = ascvec.squeeze_store_reg %334, %335 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg33[%c0_35], %336 : memref<528xi32, 1>, <i32>
                        %337 = ascvec.loadalign %arg39[%arg38] : memref<8192xf32, 1>, index -> <f32>
                        %338 = ascvec.reinterpret %337 : <f32> -> <ui32>
                        %339 = ascvec.arange %c448_i32 : i32 -> <i32>
                        %340 = ascvec.cmp_gt %338, %305, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %341 = ascvec.squeeze_store_reg %339, %340 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg33[%c0_35], %341 : memref<528xi32, 1>, <i32>
                        %342 = ascvec.loadalign %arg34[%arg35] : memref<8192xui32, 1>, index -> <ui32>
                        %343 = ascvec.cmp_eq %342, %305, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %344 = ascvec.squeeze_store_reg %307, %343 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg33[%c0_35], %344 : memref<528xi32, 1>, <i32>
                        %345 = ascvec.loadalign %arg34[%arg36] : memref<8192xui32, 1>, index -> <ui32>
                        %346 = ascvec.cmp_eq %345, %305, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %347 = ascvec.squeeze_store_reg %311, %346 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg33[%c0_35], %347 : memref<528xi32, 1>, <i32>
                        %348 = ascvec.loadalign %arg34[%arg37] : memref<8192xui32, 1>, index -> <ui32>
                        %349 = ascvec.cmp_eq %348, %305, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %350 = ascvec.squeeze_store_reg %315, %349 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg33[%c0_35], %350 : memref<528xi32, 1>, <i32>
                        %351 = ascvec.loadalign %arg34[%arg38] : memref<8192xui32, 1>, index -> <ui32>
                        %352 = ascvec.cmp_eq %351, %305, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %353 = ascvec.squeeze_store_reg %319, %352 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg33[%c0_35], %353 : memref<528xi32, 1>, <i32>
                        %354 = ascvec.loadalign %arg39[%arg35] : memref<8192xf32, 1>, index -> <f32>
                        %355 = ascvec.reinterpret %354 : <f32> -> <ui32>
                        %356 = ascvec.cmp_eq %355, %305, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %357 = ascvec.squeeze_store_reg %324, %356 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg33[%c0_35], %357 : memref<528xi32, 1>, <i32>
                        %358 = ascvec.loadalign %arg39[%arg36] : memref<8192xf32, 1>, index -> <f32>
                        %359 = ascvec.reinterpret %358 : <f32> -> <ui32>
                        %360 = ascvec.cmp_eq %359, %305, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %361 = ascvec.squeeze_store_reg %329, %360 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg33[%c0_35], %361 : memref<528xi32, 1>, <i32>
                        %362 = ascvec.loadalign %arg39[%arg37] : memref<8192xf32, 1>, index -> <f32>
                        %363 = ascvec.reinterpret %362 : <f32> -> <ui32>
                        %364 = ascvec.cmp_eq %363, %305, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %365 = ascvec.squeeze_store_reg %334, %364 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg33[%c0_35], %365 : memref<528xi32, 1>, <i32>
                        %366 = ascvec.loadalign %arg39[%arg38] : memref<8192xf32, 1>, index -> <f32>
                        %367 = ascvec.reinterpret %366 : <f32> -> <ui32>
                        %368 = ascvec.cmp_eq %367, %305, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %369 = ascvec.squeeze_store_reg %339, %368 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg33[%c0_35], %369 : memref<528xi32, 1>, <i32>
                        ascvec.store_unalign_post %arg33[%c0_35] : memref<528xi32, 1>
                        %370 = ascvec.loadalign_brc %arg28[%arg40] {brc_width = "b8"} : memref<256xi32, 1>, index -> <ui32>
                        %371 = ascvec.and %370, %291, %mask : <ui32>, <ui32> -> <ui32>
                        %372 = ascvec.loadalign_brc %arg28[%arg41] {brc_width = "b8"} : memref<256xi32, 1>, index -> <ui32>
                        %373 = ascvec.and %372, %291, %mask : <ui32>, <ui32> -> <ui32>
                        %374 = ascvec.loadalign_brc %arg28[%arg42] {brc_width = "b8"} : memref<256xi32, 1>, index -> <ui32>
                        %375 = ascvec.and %374, %291, %mask : <ui32>, <ui32> -> <ui32>
                        %376 = ascvec.loadalign_brc %arg28[%arg43] {brc_width = "b8"} : memref<256xi32, 1>, index -> <ui32>
                        %377 = ascvec.and %376, %291, %mask : <ui32>, <ui32> -> <ui32>
                        %378 = ascvec.shl %371, 24, %mask : <ui32> -> <ui32>
                        %379 = ascvec.shl %373, 16, %mask : <ui32> -> <ui32>
                        %380 = ascvec.add %378, %379, %mask : <ui32>
                        %381 = ascvec.shl %375, 8, %mask : <ui32> -> <ui32>
                        %382 = ascvec.add %380, %381, %mask : <ui32>
                        %383 = ascvec.add %382, %377, %mask : <ui32>
                        ascvec.store_unalign_begin %arg44 : memref<528xi32, 1>
                        %384 = ascvec.loadalign %arg34[%arg45] : memref<8192xui32, 1>, index -> <ui32>
                        %385 = ascvec.cmp_gt %384, %383, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %386 = ascvec.squeeze_store_reg %307, %385 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg44[%c0_35], %386 : memref<528xi32, 1>, <i32>
                        %387 = ascvec.loadalign %arg34[%arg46] : memref<8192xui32, 1>, index -> <ui32>
                        %388 = ascvec.cmp_gt %387, %383, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %389 = ascvec.squeeze_store_reg %311, %388 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg44[%c0_35], %389 : memref<528xi32, 1>, <i32>
                        %390 = ascvec.loadalign %arg34[%arg47] : memref<8192xui32, 1>, index -> <ui32>
                        %391 = ascvec.cmp_gt %390, %383, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %392 = ascvec.squeeze_store_reg %315, %391 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg44[%c0_35], %392 : memref<528xi32, 1>, <i32>
                        %393 = ascvec.loadalign %arg34[%arg48] : memref<8192xui32, 1>, index -> <ui32>
                        %394 = ascvec.cmp_gt %393, %383, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %395 = ascvec.squeeze_store_reg %319, %394 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg44[%c0_35], %395 : memref<528xi32, 1>, <i32>
                        %396 = ascvec.loadalign %arg39[%arg45] : memref<8192xf32, 1>, index -> <f32>
                        %397 = ascvec.reinterpret %396 : <f32> -> <ui32>
                        %398 = ascvec.cmp_gt %397, %383, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %399 = ascvec.squeeze_store_reg %324, %398 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg44[%c0_35], %399 : memref<528xi32, 1>, <i32>
                        %400 = ascvec.loadalign %arg39[%arg46] : memref<8192xf32, 1>, index -> <f32>
                        %401 = ascvec.reinterpret %400 : <f32> -> <ui32>
                        %402 = ascvec.cmp_gt %401, %383, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %403 = ascvec.squeeze_store_reg %329, %402 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg44[%c0_35], %403 : memref<528xi32, 1>, <i32>
                        %404 = ascvec.loadalign %arg39[%arg47] : memref<8192xf32, 1>, index -> <f32>
                        %405 = ascvec.reinterpret %404 : <f32> -> <ui32>
                        %406 = ascvec.cmp_gt %405, %383, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %407 = ascvec.squeeze_store_reg %334, %406 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg44[%c0_35], %407 : memref<528xi32, 1>, <i32>
                        %408 = ascvec.loadalign %arg39[%arg48] : memref<8192xf32, 1>, index -> <f32>
                        %409 = ascvec.reinterpret %408 : <f32> -> <ui32>
                        %410 = ascvec.cmp_gt %409, %383, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %411 = ascvec.squeeze_store_reg %339, %410 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg44[%c0_35], %411 : memref<528xi32, 1>, <i32>
                        %412 = ascvec.loadalign %arg34[%arg45] : memref<8192xui32, 1>, index -> <ui32>
                        %413 = ascvec.cmp_eq %412, %383, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %414 = ascvec.squeeze_store_reg %307, %413 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg44[%c0_35], %414 : memref<528xi32, 1>, <i32>
                        %415 = ascvec.loadalign %arg34[%arg46] : memref<8192xui32, 1>, index -> <ui32>
                        %416 = ascvec.cmp_eq %415, %383, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %417 = ascvec.squeeze_store_reg %311, %416 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg44[%c0_35], %417 : memref<528xi32, 1>, <i32>
                        %418 = ascvec.loadalign %arg34[%arg47] : memref<8192xui32, 1>, index -> <ui32>
                        %419 = ascvec.cmp_eq %418, %383, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %420 = ascvec.squeeze_store_reg %315, %419 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg44[%c0_35], %420 : memref<528xi32, 1>, <i32>
                        %421 = ascvec.loadalign %arg34[%arg48] : memref<8192xui32, 1>, index -> <ui32>
                        %422 = ascvec.cmp_eq %421, %383, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %423 = ascvec.squeeze_store_reg %319, %422 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg44[%c0_35], %423 : memref<528xi32, 1>, <i32>
                        %424 = ascvec.loadalign %arg39[%arg45] : memref<8192xf32, 1>, index -> <f32>
                        %425 = ascvec.reinterpret %424 : <f32> -> <ui32>
                        %426 = ascvec.cmp_eq %425, %383, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %427 = ascvec.squeeze_store_reg %324, %426 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg44[%c0_35], %427 : memref<528xi32, 1>, <i32>
                        %428 = ascvec.loadalign %arg39[%arg46] : memref<8192xf32, 1>, index -> <f32>
                        %429 = ascvec.reinterpret %428 : <f32> -> <ui32>
                        %430 = ascvec.cmp_eq %429, %383, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %431 = ascvec.squeeze_store_reg %329, %430 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg44[%c0_35], %431 : memref<528xi32, 1>, <i32>
                        %432 = ascvec.loadalign %arg39[%arg47] : memref<8192xf32, 1>, index -> <f32>
                        %433 = ascvec.reinterpret %432 : <f32> -> <ui32>
                        %434 = ascvec.cmp_eq %433, %383, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %435 = ascvec.squeeze_store_reg %334, %434 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg44[%c0_35], %435 : memref<528xi32, 1>, <i32>
                        %436 = ascvec.loadalign %arg39[%arg48] : memref<8192xf32, 1>, index -> <f32>
                        %437 = ascvec.reinterpret %436 : <f32> -> <ui32>
                        %438 = ascvec.cmp_eq %437, %383, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %439 = ascvec.squeeze_store_reg %339, %438 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg44[%c0_35], %439 : memref<528xi32, 1>, <i32>
                        ascvec.store_unalign_post %arg44[%c0_35] : memref<528xi32, 1>
                        %440 = ascvec.loadalign_brc %arg28[%arg49] {brc_width = "b8"} : memref<256xi32, 1>, index -> <ui32>
                        %441 = ascvec.and %440, %291, %mask : <ui32>, <ui32> -> <ui32>
                        %442 = ascvec.loadalign_brc %arg28[%arg50] {brc_width = "b8"} : memref<256xi32, 1>, index -> <ui32>
                        %443 = ascvec.and %442, %291, %mask : <ui32>, <ui32> -> <ui32>
                        %444 = ascvec.loadalign_brc %arg28[%arg51] {brc_width = "b8"} : memref<256xi32, 1>, index -> <ui32>
                        %445 = ascvec.and %444, %291, %mask : <ui32>, <ui32> -> <ui32>
                        %446 = ascvec.loadalign_brc %arg28[%arg52] {brc_width = "b8"} : memref<256xi32, 1>, index -> <ui32>
                        %447 = ascvec.and %446, %291, %mask : <ui32>, <ui32> -> <ui32>
                        %448 = ascvec.shl %441, 24, %mask : <ui32> -> <ui32>
                        %449 = ascvec.shl %443, 16, %mask : <ui32> -> <ui32>
                        %450 = ascvec.add %448, %449, %mask : <ui32>
                        %451 = ascvec.shl %445, 8, %mask : <ui32> -> <ui32>
                        %452 = ascvec.add %450, %451, %mask : <ui32>
                        %453 = ascvec.add %452, %447, %mask : <ui32>
                        ascvec.store_unalign_begin %arg53 : memref<528xi32, 1>
                        %454 = ascvec.loadalign %arg34[%arg54] : memref<8192xui32, 1>, index -> <ui32>
                        %455 = ascvec.cmp_gt %454, %453, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %456 = ascvec.squeeze_store_reg %307, %455 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg53[%c0_35], %456 : memref<528xi32, 1>, <i32>
                        %457 = ascvec.loadalign %arg34[%arg55] : memref<8192xui32, 1>, index -> <ui32>
                        %458 = ascvec.cmp_gt %457, %453, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %459 = ascvec.squeeze_store_reg %311, %458 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg53[%c0_35], %459 : memref<528xi32, 1>, <i32>
                        %460 = ascvec.loadalign %arg34[%arg56] : memref<8192xui32, 1>, index -> <ui32>
                        %461 = ascvec.cmp_gt %460, %453, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %462 = ascvec.squeeze_store_reg %315, %461 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg53[%c0_35], %462 : memref<528xi32, 1>, <i32>
                        %463 = ascvec.loadalign %arg34[%arg57] : memref<8192xui32, 1>, index -> <ui32>
                        %464 = ascvec.cmp_gt %463, %453, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %465 = ascvec.squeeze_store_reg %319, %464 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg53[%c0_35], %465 : memref<528xi32, 1>, <i32>
                        %466 = ascvec.loadalign %arg39[%arg54] : memref<8192xf32, 1>, index -> <f32>
                        %467 = ascvec.reinterpret %466 : <f32> -> <ui32>
                        %468 = ascvec.cmp_gt %467, %453, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %469 = ascvec.squeeze_store_reg %324, %468 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg53[%c0_35], %469 : memref<528xi32, 1>, <i32>
                        %470 = ascvec.loadalign %arg39[%arg55] : memref<8192xf32, 1>, index -> <f32>
                        %471 = ascvec.reinterpret %470 : <f32> -> <ui32>
                        %472 = ascvec.cmp_gt %471, %453, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %473 = ascvec.squeeze_store_reg %329, %472 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg53[%c0_35], %473 : memref<528xi32, 1>, <i32>
                        %474 = ascvec.loadalign %arg39[%arg56] : memref<8192xf32, 1>, index -> <f32>
                        %475 = ascvec.reinterpret %474 : <f32> -> <ui32>
                        %476 = ascvec.cmp_gt %475, %453, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %477 = ascvec.squeeze_store_reg %334, %476 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg53[%c0_35], %477 : memref<528xi32, 1>, <i32>
                        %478 = ascvec.loadalign %arg39[%arg57] : memref<8192xf32, 1>, index -> <f32>
                        %479 = ascvec.reinterpret %478 : <f32> -> <ui32>
                        %480 = ascvec.cmp_gt %479, %453, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %481 = ascvec.squeeze_store_reg %339, %480 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg53[%c0_35], %481 : memref<528xi32, 1>, <i32>
                        %482 = ascvec.loadalign %arg34[%arg54] : memref<8192xui32, 1>, index -> <ui32>
                        %483 = ascvec.cmp_eq %482, %453, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %484 = ascvec.squeeze_store_reg %307, %483 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg53[%c0_35], %484 : memref<528xi32, 1>, <i32>
                        %485 = ascvec.loadalign %arg34[%arg55] : memref<8192xui32, 1>, index -> <ui32>
                        %486 = ascvec.cmp_eq %485, %453, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %487 = ascvec.squeeze_store_reg %311, %486 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg53[%c0_35], %487 : memref<528xi32, 1>, <i32>
                        %488 = ascvec.loadalign %arg34[%arg56] : memref<8192xui32, 1>, index -> <ui32>
                        %489 = ascvec.cmp_eq %488, %453, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %490 = ascvec.squeeze_store_reg %315, %489 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg53[%c0_35], %490 : memref<528xi32, 1>, <i32>
                        %491 = ascvec.loadalign %arg34[%arg57] : memref<8192xui32, 1>, index -> <ui32>
                        %492 = ascvec.cmp_eq %491, %453, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %493 = ascvec.squeeze_store_reg %319, %492 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg53[%c0_35], %493 : memref<528xi32, 1>, <i32>
                        %494 = ascvec.loadalign %arg39[%arg54] : memref<8192xf32, 1>, index -> <f32>
                        %495 = ascvec.reinterpret %494 : <f32> -> <ui32>
                        %496 = ascvec.cmp_eq %495, %453, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %497 = ascvec.squeeze_store_reg %324, %496 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg53[%c0_35], %497 : memref<528xi32, 1>, <i32>
                        %498 = ascvec.loadalign %arg39[%arg55] : memref<8192xf32, 1>, index -> <f32>
                        %499 = ascvec.reinterpret %498 : <f32> -> <ui32>
                        %500 = ascvec.cmp_eq %499, %453, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %501 = ascvec.squeeze_store_reg %329, %500 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg53[%c0_35], %501 : memref<528xi32, 1>, <i32>
                        %502 = ascvec.loadalign %arg39[%arg56] : memref<8192xf32, 1>, index -> <f32>
                        %503 = ascvec.reinterpret %502 : <f32> -> <ui32>
                        %504 = ascvec.cmp_eq %503, %453, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %505 = ascvec.squeeze_store_reg %334, %504 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg53[%c0_35], %505 : memref<528xi32, 1>, <i32>
                        %506 = ascvec.loadalign %arg39[%arg57] : memref<8192xf32, 1>, index -> <f32>
                        %507 = ascvec.reinterpret %506 : <f32> -> <ui32>
                        %508 = ascvec.cmp_eq %507, %453, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %509 = ascvec.squeeze_store_reg %339, %508 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg53[%c0_35], %509 : memref<528xi32, 1>, <i32>
                        ascvec.store_unalign_post %arg53[%c0_35] : memref<528xi32, 1>
                        %510 = ascvec.loadalign_brc %arg28[%arg58] {brc_width = "b8"} : memref<256xi32, 1>, index -> <ui32>
                        %511 = ascvec.and %510, %291, %mask : <ui32>, <ui32> -> <ui32>
                        %512 = ascvec.loadalign_brc %arg28[%arg59] {brc_width = "b8"} : memref<256xi32, 1>, index -> <ui32>
                        %513 = ascvec.and %512, %291, %mask : <ui32>, <ui32> -> <ui32>
                        %514 = ascvec.loadalign_brc %arg28[%arg60] {brc_width = "b8"} : memref<256xi32, 1>, index -> <ui32>
                        %515 = ascvec.and %514, %291, %mask : <ui32>, <ui32> -> <ui32>
                        %516 = ascvec.loadalign_brc %arg28[%arg61] {brc_width = "b8"} : memref<256xi32, 1>, index -> <ui32>
                        %517 = ascvec.and %516, %291, %mask : <ui32>, <ui32> -> <ui32>
                        %518 = ascvec.shl %511, 24, %mask : <ui32> -> <ui32>
                        %519 = ascvec.shl %513, 16, %mask : <ui32> -> <ui32>
                        %520 = ascvec.add %518, %519, %mask : <ui32>
                        %521 = ascvec.shl %515, 8, %mask : <ui32> -> <ui32>
                        %522 = ascvec.add %520, %521, %mask : <ui32>
                        %523 = ascvec.add %522, %517, %mask : <ui32>
                        ascvec.store_unalign_begin %arg62 : memref<528xi32, 1>
                        %524 = ascvec.loadalign %arg34[%arg63] : memref<8192xui32, 1>, index -> <ui32>
                        %525 = ascvec.cmp_gt %524, %523, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %526 = ascvec.squeeze_store_reg %307, %525 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg62[%c0_35], %526 : memref<528xi32, 1>, <i32>
                        %527 = ascvec.loadalign %arg34[%arg64] : memref<8192xui32, 1>, index -> <ui32>
                        %528 = ascvec.cmp_gt %527, %523, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %529 = ascvec.squeeze_store_reg %311, %528 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg62[%c0_35], %529 : memref<528xi32, 1>, <i32>
                        %530 = ascvec.loadalign %arg34[%arg65] : memref<8192xui32, 1>, index -> <ui32>
                        %531 = ascvec.cmp_gt %530, %523, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %532 = ascvec.squeeze_store_reg %315, %531 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg62[%c0_35], %532 : memref<528xi32, 1>, <i32>
                        %533 = ascvec.loadalign %arg34[%arg66] : memref<8192xui32, 1>, index -> <ui32>
                        %534 = ascvec.cmp_gt %533, %523, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %535 = ascvec.squeeze_store_reg %319, %534 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg62[%c0_35], %535 : memref<528xi32, 1>, <i32>
                        %536 = ascvec.loadalign %arg39[%arg63] : memref<8192xf32, 1>, index -> <f32>
                        %537 = ascvec.reinterpret %536 : <f32> -> <ui32>
                        %538 = ascvec.cmp_gt %537, %523, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %539 = ascvec.squeeze_store_reg %324, %538 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg62[%c0_35], %539 : memref<528xi32, 1>, <i32>
                        %540 = ascvec.loadalign %arg39[%arg64] : memref<8192xf32, 1>, index -> <f32>
                        %541 = ascvec.reinterpret %540 : <f32> -> <ui32>
                        %542 = ascvec.cmp_gt %541, %523, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %543 = ascvec.squeeze_store_reg %329, %542 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg62[%c0_35], %543 : memref<528xi32, 1>, <i32>
                        %544 = ascvec.loadalign %arg39[%arg65] : memref<8192xf32, 1>, index -> <f32>
                        %545 = ascvec.reinterpret %544 : <f32> -> <ui32>
                        %546 = ascvec.cmp_gt %545, %523, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %547 = ascvec.squeeze_store_reg %334, %546 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg62[%c0_35], %547 : memref<528xi32, 1>, <i32>
                        %548 = ascvec.loadalign %arg39[%arg66] : memref<8192xf32, 1>, index -> <f32>
                        %549 = ascvec.reinterpret %548 : <f32> -> <ui32>
                        %550 = ascvec.cmp_gt %549, %523, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %551 = ascvec.squeeze_store_reg %339, %550 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg62[%c0_35], %551 : memref<528xi32, 1>, <i32>
                        %552 = ascvec.loadalign %arg34[%arg63] : memref<8192xui32, 1>, index -> <ui32>
                        %553 = ascvec.cmp_eq %552, %523, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %554 = ascvec.squeeze_store_reg %307, %553 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg62[%c0_35], %554 : memref<528xi32, 1>, <i32>
                        %555 = ascvec.loadalign %arg34[%arg64] : memref<8192xui32, 1>, index -> <ui32>
                        %556 = ascvec.cmp_eq %555, %523, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %557 = ascvec.squeeze_store_reg %311, %556 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg62[%c0_35], %557 : memref<528xi32, 1>, <i32>
                        %558 = ascvec.loadalign %arg34[%arg65] : memref<8192xui32, 1>, index -> <ui32>
                        %559 = ascvec.cmp_eq %558, %523, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %560 = ascvec.squeeze_store_reg %315, %559 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg62[%c0_35], %560 : memref<528xi32, 1>, <i32>
                        %561 = ascvec.loadalign %arg34[%arg66] : memref<8192xui32, 1>, index -> <ui32>
                        %562 = ascvec.cmp_eq %561, %523, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %563 = ascvec.squeeze_store_reg %319, %562 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg62[%c0_35], %563 : memref<528xi32, 1>, <i32>
                        %564 = ascvec.loadalign %arg39[%arg63] : memref<8192xf32, 1>, index -> <f32>
                        %565 = ascvec.reinterpret %564 : <f32> -> <ui32>
                        %566 = ascvec.cmp_eq %565, %523, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %567 = ascvec.squeeze_store_reg %324, %566 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg62[%c0_35], %567 : memref<528xi32, 1>, <i32>
                        %568 = ascvec.loadalign %arg39[%arg64] : memref<8192xf32, 1>, index -> <f32>
                        %569 = ascvec.reinterpret %568 : <f32> -> <ui32>
                        %570 = ascvec.cmp_eq %569, %523, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %571 = ascvec.squeeze_store_reg %329, %570 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg62[%c0_35], %571 : memref<528xi32, 1>, <i32>
                        %572 = ascvec.loadalign %arg39[%arg65] : memref<8192xf32, 1>, index -> <f32>
                        %573 = ascvec.reinterpret %572 : <f32> -> <ui32>
                        %574 = ascvec.cmp_eq %573, %523, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %575 = ascvec.squeeze_store_reg %334, %574 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg62[%c0_35], %575 : memref<528xi32, 1>, <i32>
                        %576 = ascvec.loadalign %arg39[%arg66] : memref<8192xf32, 1>, index -> <f32>
                        %577 = ascvec.reinterpret %576 : <f32> -> <ui32>
                        %578 = ascvec.cmp_eq %577, %523, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %579 = ascvec.squeeze_store_reg %339, %578 : <i32>, !ascvec.mask -> <i32>
                        ascvec.store_unalign %arg62[%c0_35], %579 : memref<528xi32, 1>, <i32>
                        ascvec.store_unalign_post %arg62[%c0_35] : memref<528xi32, 1>
                        ascvec.mem_bar vst_vld
                        %c0_38 = arith.constant 0 : index
                        %c1 = arith.constant 1 : index
                        scf.for %arg81 = %c0_38 to %c1 step %c1 {
                          %580 = ascvec.loadalign_brc %arg67[%arg68] : memref<512xi32, 1>, index -> <i32>
                          %581 = ascvec.duplicate_scalar %arg69, %mask : i32 -> <i32>
                          %582 = ascvec.cmp_lt %307, %580, %mask : <i32>, <i32> -> !ascvec.mask
                          %583 = ascvec.loadalign %arg33[%c0_35] : memref<528xi32, 1>, index -> <i32>
                          %584 = ascvec.select %583, %289, %582 : <i32>
                          %585 = ascvec.cmp_lt_scalar %584, %c256_i32_32, %582 : <i32>, i32 -> !ascvec.mask
                          %586 = ascvec.cmp_ge_scalar %584, %c256_i32_32, %582 : <i32>, i32 -> !ascvec.mask
                          %587 = ascvec.select %584, %289, %585 : <i32>
                          %588 = ascvec.add %587, %581, %mask : <i32>
                          %589 = ascvec.reinterpret %588 : <i32> -> <ui32>
                          %590 = ascvec.gather %arg34[%c0_35], %589, %mask : memref<8192xui32, 1>, <ui32>, !ascvec.mask -> <ui32>
                          %591 = ascvec.gather %arg70[%c0_35], %589, %mask : memref<8192xui32, 1>, <ui32>, !ascvec.mask -> <ui32>
                          %592 = ascvec.add_scalar %584, %c-256_i32, %mask : <i32>, i32
                          %593 = ascvec.select %592, %289, %586 : <i32>
                          %594 = ascvec.add %593, %581, %mask : <i32>
                          %595 = ascvec.reinterpret %594 : <i32> -> <ui32>
                          %596 = ascvec.gather %arg39[%c0_35], %595, %mask : memref<8192xf32, 1>, <ui32>, !ascvec.mask -> <f32>
                          %597 = ascvec.reinterpret %596 : <f32> -> <ui32>
                          %598 = ascvec.add_scalar %593, %arg71, %mask : <i32>, i32
                          %599 = ascvec.reinterpret %598 : <i32> -> <ui32>
                          %600 = ascvec.select %590, %597, %585 : <ui32>
                          %601 = ascvec.select %591, %599, %585 : <ui32>
                          %602 = ascvec.select %600, %290, %582 : <ui32>
                          ascvec.storealign %arg72[%c0_35], %602, %mask : memref<1024xui32, 1>, index, <ui32>
                          %603 = ascvec.select %601, %290, %582 : <ui32>
                          ascvec.storealign %arg73[%c0_35], %603, %mask : memref<1024xui32, 1>, index, <ui32>
                          %604 = ascvec.cmp_lt %311, %580, %mask : <i32>, <i32> -> !ascvec.mask
                          %605 = ascvec.loadalign %arg33[%c64] : memref<528xi32, 1>, index -> <i32>
                          %606 = ascvec.select %605, %289, %604 : <i32>
                          %607 = ascvec.cmp_lt_scalar %606, %c256_i32_32, %604 : <i32>, i32 -> !ascvec.mask
                          %608 = ascvec.cmp_ge_scalar %606, %c256_i32_32, %604 : <i32>, i32 -> !ascvec.mask
                          %609 = ascvec.select %606, %289, %607 : <i32>
                          %610 = ascvec.add %609, %581, %mask : <i32>
                          %611 = ascvec.reinterpret %610 : <i32> -> <ui32>
                          %612 = ascvec.gather %arg34[%c0_35], %611, %mask : memref<8192xui32, 1>, <ui32>, !ascvec.mask -> <ui32>
                          %613 = ascvec.gather %arg70[%c0_35], %611, %mask : memref<8192xui32, 1>, <ui32>, !ascvec.mask -> <ui32>
                          %614 = ascvec.add_scalar %606, %c-256_i32, %mask : <i32>, i32
                          %615 = ascvec.select %614, %289, %608 : <i32>
                          %616 = ascvec.add %615, %581, %mask : <i32>
                          %617 = ascvec.reinterpret %616 : <i32> -> <ui32>
                          %618 = ascvec.gather %arg39[%c0_35], %617, %mask : memref<8192xf32, 1>, <ui32>, !ascvec.mask -> <f32>
                          %619 = ascvec.reinterpret %618 : <f32> -> <ui32>
                          %620 = ascvec.add_scalar %615, %arg71, %mask : <i32>, i32
                          %621 = ascvec.reinterpret %620 : <i32> -> <ui32>
                          %622 = ascvec.select %612, %619, %607 : <ui32>
                          %623 = ascvec.select %613, %621, %607 : <ui32>
                          %624 = ascvec.select %622, %290, %604 : <ui32>
                          ascvec.storealign %arg72[%c64], %624, %mask : memref<1024xui32, 1>, index, <ui32>
                          %625 = ascvec.select %623, %290, %604 : <ui32>
                          ascvec.storealign %arg73[%c64], %625, %mask : memref<1024xui32, 1>, index, <ui32>
                          %626 = ascvec.cmp_lt %315, %580, %mask : <i32>, <i32> -> !ascvec.mask
                          %627 = ascvec.loadalign %arg33[%c128] : memref<528xi32, 1>, index -> <i32>
                          %628 = ascvec.select %627, %289, %626 : <i32>
                          %629 = ascvec.cmp_lt_scalar %628, %c256_i32_32, %626 : <i32>, i32 -> !ascvec.mask
                          %630 = ascvec.cmp_ge_scalar %628, %c256_i32_32, %626 : <i32>, i32 -> !ascvec.mask
                          %631 = ascvec.select %628, %289, %629 : <i32>
                          %632 = ascvec.add %631, %581, %mask : <i32>
                          %633 = ascvec.reinterpret %632 : <i32> -> <ui32>
                          %634 = ascvec.gather %arg34[%c0_35], %633, %mask : memref<8192xui32, 1>, <ui32>, !ascvec.mask -> <ui32>
                          %635 = ascvec.gather %arg70[%c0_35], %633, %mask : memref<8192xui32, 1>, <ui32>, !ascvec.mask -> <ui32>
                          %636 = ascvec.add_scalar %628, %c-256_i32, %mask : <i32>, i32
                          %637 = ascvec.select %636, %289, %630 : <i32>
                          %638 = ascvec.add %637, %581, %mask : <i32>
                          %639 = ascvec.reinterpret %638 : <i32> -> <ui32>
                          %640 = ascvec.gather %arg39[%c0_35], %639, %mask : memref<8192xf32, 1>, <ui32>, !ascvec.mask -> <f32>
                          %641 = ascvec.reinterpret %640 : <f32> -> <ui32>
                          %642 = ascvec.add_scalar %637, %arg71, %mask : <i32>, i32
                          %643 = ascvec.reinterpret %642 : <i32> -> <ui32>
                          %644 = ascvec.select %634, %641, %629 : <ui32>
                          %645 = ascvec.select %635, %643, %629 : <ui32>
                          %646 = ascvec.select %644, %290, %626 : <ui32>
                          ascvec.storealign %arg72[%c128], %646, %mask : memref<1024xui32, 1>, index, <ui32>
                          %647 = ascvec.select %645, %290, %626 : <ui32>
                          ascvec.storealign %arg73[%c128], %647, %mask : memref<1024xui32, 1>, index, <ui32>
                          %648 = ascvec.cmp_lt %319, %580, %mask : <i32>, <i32> -> !ascvec.mask
                          %649 = ascvec.loadalign %arg33[%c192] : memref<528xi32, 1>, index -> <i32>
                          %650 = ascvec.select %649, %289, %648 : <i32>
                          %651 = ascvec.cmp_lt_scalar %650, %c256_i32_32, %648 : <i32>, i32 -> !ascvec.mask
                          %652 = ascvec.cmp_ge_scalar %650, %c256_i32_32, %648 : <i32>, i32 -> !ascvec.mask
                          %653 = ascvec.select %650, %289, %651 : <i32>
                          %654 = ascvec.add %653, %581, %mask : <i32>
                          %655 = ascvec.reinterpret %654 : <i32> -> <ui32>
                          %656 = ascvec.gather %arg34[%c0_35], %655, %mask : memref<8192xui32, 1>, <ui32>, !ascvec.mask -> <ui32>
                          %657 = ascvec.gather %arg70[%c0_35], %655, %mask : memref<8192xui32, 1>, <ui32>, !ascvec.mask -> <ui32>
                          %658 = ascvec.add_scalar %650, %c-256_i32, %mask : <i32>, i32
                          %659 = ascvec.select %658, %289, %652 : <i32>
                          %660 = ascvec.add %659, %581, %mask : <i32>
                          %661 = ascvec.reinterpret %660 : <i32> -> <ui32>
                          %662 = ascvec.gather %arg39[%c0_35], %661, %mask : memref<8192xf32, 1>, <ui32>, !ascvec.mask -> <f32>
                          %663 = ascvec.reinterpret %662 : <f32> -> <ui32>
                          %664 = ascvec.add_scalar %659, %arg71, %mask : <i32>, i32
                          %665 = ascvec.reinterpret %664 : <i32> -> <ui32>
                          %666 = ascvec.select %656, %663, %651 : <ui32>
                          %667 = ascvec.select %657, %665, %651 : <ui32>
                          %668 = ascvec.select %666, %290, %648 : <ui32>
                          ascvec.storealign %arg72[%c192], %668, %mask : memref<1024xui32, 1>, index, <ui32>
                          %669 = ascvec.select %667, %290, %648 : <ui32>
                          ascvec.storealign %arg73[%c192], %669, %mask : memref<1024xui32, 1>, index, <ui32>
                          %670 = ascvec.loadalign_brc %arg67[%arg74] : memref<512xi32, 1>, index -> <i32>
                          %671 = ascvec.duplicate_scalar %arg75, %mask : i32 -> <i32>
                          %672 = ascvec.cmp_lt %307, %670, %mask : <i32>, <i32> -> !ascvec.mask
                          %673 = ascvec.loadalign %arg44[%c0_35] : memref<528xi32, 1>, index -> <i32>
                          %674 = ascvec.select %673, %289, %672 : <i32>
                          %675 = ascvec.cmp_lt_scalar %674, %c256_i32_32, %672 : <i32>, i32 -> !ascvec.mask
                          %676 = ascvec.cmp_ge_scalar %674, %c256_i32_32, %672 : <i32>, i32 -> !ascvec.mask
                          %677 = ascvec.select %674, %289, %675 : <i32>
                          %678 = ascvec.add %677, %671, %mask : <i32>
                          %679 = ascvec.reinterpret %678 : <i32> -> <ui32>
                          %680 = ascvec.gather %arg34[%c0_35], %679, %mask : memref<8192xui32, 1>, <ui32>, !ascvec.mask -> <ui32>
                          %681 = ascvec.gather %arg70[%c0_35], %679, %mask : memref<8192xui32, 1>, <ui32>, !ascvec.mask -> <ui32>
                          %682 = ascvec.add_scalar %674, %c-256_i32, %mask : <i32>, i32
                          %683 = ascvec.select %682, %289, %676 : <i32>
                          %684 = ascvec.add %683, %671, %mask : <i32>
                          %685 = ascvec.reinterpret %684 : <i32> -> <ui32>
                          %686 = ascvec.gather %arg39[%c0_35], %685, %mask : memref<8192xf32, 1>, <ui32>, !ascvec.mask -> <f32>
                          %687 = ascvec.reinterpret %686 : <f32> -> <ui32>
                          %688 = ascvec.add_scalar %683, %arg71, %mask : <i32>, i32
                          %689 = ascvec.reinterpret %688 : <i32> -> <ui32>
                          %690 = ascvec.select %680, %687, %675 : <ui32>
                          %691 = ascvec.select %681, %689, %675 : <ui32>
                          %692 = ascvec.select %690, %290, %672 : <ui32>
                          ascvec.storealign %arg72[%c256], %692, %mask : memref<1024xui32, 1>, index, <ui32>
                          %693 = ascvec.select %691, %290, %672 : <ui32>
                          ascvec.storealign %arg73[%c256], %693, %mask : memref<1024xui32, 1>, index, <ui32>
                          %694 = ascvec.cmp_lt %311, %670, %mask : <i32>, <i32> -> !ascvec.mask
                          %695 = ascvec.loadalign %arg44[%c64] : memref<528xi32, 1>, index -> <i32>
                          %696 = ascvec.select %695, %289, %694 : <i32>
                          %697 = ascvec.cmp_lt_scalar %696, %c256_i32_32, %694 : <i32>, i32 -> !ascvec.mask
                          %698 = ascvec.cmp_ge_scalar %696, %c256_i32_32, %694 : <i32>, i32 -> !ascvec.mask
                          %699 = ascvec.select %696, %289, %697 : <i32>
                          %700 = ascvec.add %699, %671, %mask : <i32>
                          %701 = ascvec.reinterpret %700 : <i32> -> <ui32>
                          %702 = ascvec.gather %arg34[%c0_35], %701, %mask : memref<8192xui32, 1>, <ui32>, !ascvec.mask -> <ui32>
                          %703 = ascvec.gather %arg70[%c0_35], %701, %mask : memref<8192xui32, 1>, <ui32>, !ascvec.mask -> <ui32>
                          %704 = ascvec.add_scalar %696, %c-256_i32, %mask : <i32>, i32
                          %705 = ascvec.select %704, %289, %698 : <i32>
                          %706 = ascvec.add %705, %671, %mask : <i32>
                          %707 = ascvec.reinterpret %706 : <i32> -> <ui32>
                          %708 = ascvec.gather %arg39[%c0_35], %707, %mask : memref<8192xf32, 1>, <ui32>, !ascvec.mask -> <f32>
                          %709 = ascvec.reinterpret %708 : <f32> -> <ui32>
                          %710 = ascvec.add_scalar %705, %arg71, %mask : <i32>, i32
                          %711 = ascvec.reinterpret %710 : <i32> -> <ui32>
                          %712 = ascvec.select %702, %709, %697 : <ui32>
                          %713 = ascvec.select %703, %711, %697 : <ui32>
                          %714 = ascvec.select %712, %290, %694 : <ui32>
                          ascvec.storealign %arg72[%c320], %714, %mask : memref<1024xui32, 1>, index, <ui32>
                          %715 = ascvec.select %713, %290, %694 : <ui32>
                          ascvec.storealign %arg73[%c320], %715, %mask : memref<1024xui32, 1>, index, <ui32>
                          %716 = ascvec.cmp_lt %315, %670, %mask : <i32>, <i32> -> !ascvec.mask
                          %717 = ascvec.loadalign %arg44[%c128] : memref<528xi32, 1>, index -> <i32>
                          %718 = ascvec.select %717, %289, %716 : <i32>
                          %719 = ascvec.cmp_lt_scalar %718, %c256_i32_32, %716 : <i32>, i32 -> !ascvec.mask
                          %720 = ascvec.cmp_ge_scalar %718, %c256_i32_32, %716 : <i32>, i32 -> !ascvec.mask
                          %721 = ascvec.select %718, %289, %719 : <i32>
                          %722 = ascvec.add %721, %671, %mask : <i32>
                          %723 = ascvec.reinterpret %722 : <i32> -> <ui32>
                          %724 = ascvec.gather %arg34[%c0_35], %723, %mask : memref<8192xui32, 1>, <ui32>, !ascvec.mask -> <ui32>
                          %725 = ascvec.gather %arg70[%c0_35], %723, %mask : memref<8192xui32, 1>, <ui32>, !ascvec.mask -> <ui32>
                          %726 = ascvec.add_scalar %718, %c-256_i32, %mask : <i32>, i32
                          %727 = ascvec.select %726, %289, %720 : <i32>
                          %728 = ascvec.add %727, %671, %mask : <i32>
                          %729 = ascvec.reinterpret %728 : <i32> -> <ui32>
                          %730 = ascvec.gather %arg39[%c0_35], %729, %mask : memref<8192xf32, 1>, <ui32>, !ascvec.mask -> <f32>
                          %731 = ascvec.reinterpret %730 : <f32> -> <ui32>
                          %732 = ascvec.add_scalar %727, %arg71, %mask : <i32>, i32
                          %733 = ascvec.reinterpret %732 : <i32> -> <ui32>
                          %734 = ascvec.select %724, %731, %719 : <ui32>
                          %735 = ascvec.select %725, %733, %719 : <ui32>
                          %736 = ascvec.select %734, %290, %716 : <ui32>
                          ascvec.storealign %arg72[%c384], %736, %mask : memref<1024xui32, 1>, index, <ui32>
                          %737 = ascvec.select %735, %290, %716 : <ui32>
                          ascvec.storealign %arg73[%c384], %737, %mask : memref<1024xui32, 1>, index, <ui32>
                          %738 = ascvec.cmp_lt %319, %670, %mask : <i32>, <i32> -> !ascvec.mask
                          %739 = ascvec.loadalign %arg44[%c192] : memref<528xi32, 1>, index -> <i32>
                          %740 = ascvec.select %739, %289, %738 : <i32>
                          %741 = ascvec.cmp_lt_scalar %740, %c256_i32_32, %738 : <i32>, i32 -> !ascvec.mask
                          %742 = ascvec.cmp_ge_scalar %740, %c256_i32_32, %738 : <i32>, i32 -> !ascvec.mask
                          %743 = ascvec.select %740, %289, %741 : <i32>
                          %744 = ascvec.add %743, %671, %mask : <i32>
                          %745 = ascvec.reinterpret %744 : <i32> -> <ui32>
                          %746 = ascvec.gather %arg34[%c0_35], %745, %mask : memref<8192xui32, 1>, <ui32>, !ascvec.mask -> <ui32>
                          %747 = ascvec.gather %arg70[%c0_35], %745, %mask : memref<8192xui32, 1>, <ui32>, !ascvec.mask -> <ui32>
                          %748 = ascvec.add_scalar %740, %c-256_i32, %mask : <i32>, i32
                          %749 = ascvec.select %748, %289, %742 : <i32>
                          %750 = ascvec.add %749, %671, %mask : <i32>
                          %751 = ascvec.reinterpret %750 : <i32> -> <ui32>
                          %752 = ascvec.gather %arg39[%c0_35], %751, %mask : memref<8192xf32, 1>, <ui32>, !ascvec.mask -> <f32>
                          %753 = ascvec.reinterpret %752 : <f32> -> <ui32>
                          %754 = ascvec.add_scalar %749, %arg71, %mask : <i32>, i32
                          %755 = ascvec.reinterpret %754 : <i32> -> <ui32>
                          %756 = ascvec.select %746, %753, %741 : <ui32>
                          %757 = ascvec.select %747, %755, %741 : <ui32>
                          %758 = ascvec.select %756, %290, %738 : <ui32>
                          ascvec.storealign %arg72[%c448], %758, %mask : memref<1024xui32, 1>, index, <ui32>
                          %759 = ascvec.select %757, %290, %738 : <ui32>
                          ascvec.storealign %arg73[%c448], %759, %mask : memref<1024xui32, 1>, index, <ui32>
                          %760 = ascvec.loadalign_brc %arg67[%arg76] : memref<512xi32, 1>, index -> <i32>
                          %761 = ascvec.duplicate_scalar %arg77, %mask : i32 -> <i32>
                          %762 = ascvec.cmp_lt %307, %760, %mask : <i32>, <i32> -> !ascvec.mask
                          %763 = ascvec.loadalign %arg53[%c0_35] : memref<528xi32, 1>, index -> <i32>
                          %764 = ascvec.select %763, %289, %762 : <i32>
                          %765 = ascvec.cmp_lt_scalar %764, %c256_i32_32, %762 : <i32>, i32 -> !ascvec.mask
                          %766 = ascvec.cmp_ge_scalar %764, %c256_i32_32, %762 : <i32>, i32 -> !ascvec.mask
                          %767 = ascvec.select %764, %289, %765 : <i32>
                          %768 = ascvec.add %767, %761, %mask : <i32>
                          %769 = ascvec.reinterpret %768 : <i32> -> <ui32>
                          %770 = ascvec.gather %arg34[%c0_35], %769, %mask : memref<8192xui32, 1>, <ui32>, !ascvec.mask -> <ui32>
                          %771 = ascvec.gather %arg70[%c0_35], %769, %mask : memref<8192xui32, 1>, <ui32>, !ascvec.mask -> <ui32>
                          %772 = ascvec.add_scalar %764, %c-256_i32, %mask : <i32>, i32
                          %773 = ascvec.select %772, %289, %766 : <i32>
                          %774 = ascvec.add %773, %761, %mask : <i32>
                          %775 = ascvec.reinterpret %774 : <i32> -> <ui32>
                          %776 = ascvec.gather %arg39[%c0_35], %775, %mask : memref<8192xf32, 1>, <ui32>, !ascvec.mask -> <f32>
                          %777 = ascvec.reinterpret %776 : <f32> -> <ui32>
                          %778 = ascvec.add_scalar %773, %arg71, %mask : <i32>, i32
                          %779 = ascvec.reinterpret %778 : <i32> -> <ui32>
                          %780 = ascvec.select %770, %777, %765 : <ui32>
                          %781 = ascvec.select %771, %779, %765 : <ui32>
                          %782 = ascvec.select %780, %290, %762 : <ui32>
                          ascvec.storealign %arg72[%c512], %782, %mask : memref<1024xui32, 1>, index, <ui32>
                          %783 = ascvec.select %781, %290, %762 : <ui32>
                          ascvec.storealign %arg73[%c512], %783, %mask : memref<1024xui32, 1>, index, <ui32>
                          %784 = ascvec.cmp_lt %311, %760, %mask : <i32>, <i32> -> !ascvec.mask
                          %785 = ascvec.loadalign %arg53[%c64] : memref<528xi32, 1>, index -> <i32>
                          %786 = ascvec.select %785, %289, %784 : <i32>
                          %787 = ascvec.cmp_lt_scalar %786, %c256_i32_32, %784 : <i32>, i32 -> !ascvec.mask
                          %788 = ascvec.cmp_ge_scalar %786, %c256_i32_32, %784 : <i32>, i32 -> !ascvec.mask
                          %789 = ascvec.select %786, %289, %787 : <i32>
                          %790 = ascvec.add %789, %761, %mask : <i32>
                          %791 = ascvec.reinterpret %790 : <i32> -> <ui32>
                          %792 = ascvec.gather %arg34[%c0_35], %791, %mask : memref<8192xui32, 1>, <ui32>, !ascvec.mask -> <ui32>
                          %793 = ascvec.gather %arg70[%c0_35], %791, %mask : memref<8192xui32, 1>, <ui32>, !ascvec.mask -> <ui32>
                          %794 = ascvec.add_scalar %786, %c-256_i32, %mask : <i32>, i32
                          %795 = ascvec.select %794, %289, %788 : <i32>
                          %796 = ascvec.add %795, %761, %mask : <i32>
                          %797 = ascvec.reinterpret %796 : <i32> -> <ui32>
                          %798 = ascvec.gather %arg39[%c0_35], %797, %mask : memref<8192xf32, 1>, <ui32>, !ascvec.mask -> <f32>
                          %799 = ascvec.reinterpret %798 : <f32> -> <ui32>
                          %800 = ascvec.add_scalar %795, %arg71, %mask : <i32>, i32
                          %801 = ascvec.reinterpret %800 : <i32> -> <ui32>
                          %802 = ascvec.select %792, %799, %787 : <ui32>
                          %803 = ascvec.select %793, %801, %787 : <ui32>
                          %804 = ascvec.select %802, %290, %784 : <ui32>
                          ascvec.storealign %arg72[%c576], %804, %mask : memref<1024xui32, 1>, index, <ui32>
                          %805 = ascvec.select %803, %290, %784 : <ui32>
                          ascvec.storealign %arg73[%c576], %805, %mask : memref<1024xui32, 1>, index, <ui32>
                          %806 = ascvec.cmp_lt %315, %760, %mask : <i32>, <i32> -> !ascvec.mask
                          %807 = ascvec.loadalign %arg53[%c128] : memref<528xi32, 1>, index -> <i32>
                          %808 = ascvec.select %807, %289, %806 : <i32>
                          %809 = ascvec.cmp_lt_scalar %808, %c256_i32_32, %806 : <i32>, i32 -> !ascvec.mask
                          %810 = ascvec.cmp_ge_scalar %808, %c256_i32_32, %806 : <i32>, i32 -> !ascvec.mask
                          %811 = ascvec.select %808, %289, %809 : <i32>
                          %812 = ascvec.add %811, %761, %mask : <i32>
                          %813 = ascvec.reinterpret %812 : <i32> -> <ui32>
                          %814 = ascvec.gather %arg34[%c0_35], %813, %mask : memref<8192xui32, 1>, <ui32>, !ascvec.mask -> <ui32>
                          %815 = ascvec.gather %arg70[%c0_35], %813, %mask : memref<8192xui32, 1>, <ui32>, !ascvec.mask -> <ui32>
                          %816 = ascvec.add_scalar %808, %c-256_i32, %mask : <i32>, i32
                          %817 = ascvec.select %816, %289, %810 : <i32>
                          %818 = ascvec.add %817, %761, %mask : <i32>
                          %819 = ascvec.reinterpret %818 : <i32> -> <ui32>
                          %820 = ascvec.gather %arg39[%c0_35], %819, %mask : memref<8192xf32, 1>, <ui32>, !ascvec.mask -> <f32>
                          %821 = ascvec.reinterpret %820 : <f32> -> <ui32>
                          %822 = ascvec.add_scalar %817, %arg71, %mask : <i32>, i32
                          %823 = ascvec.reinterpret %822 : <i32> -> <ui32>
                          %824 = ascvec.select %814, %821, %809 : <ui32>
                          %825 = ascvec.select %815, %823, %809 : <ui32>
                          %826 = ascvec.select %824, %290, %806 : <ui32>
                          ascvec.storealign %arg72[%c640], %826, %mask : memref<1024xui32, 1>, index, <ui32>
                          %827 = ascvec.select %825, %290, %806 : <ui32>
                          ascvec.storealign %arg73[%c640], %827, %mask : memref<1024xui32, 1>, index, <ui32>
                          %828 = ascvec.cmp_lt %319, %760, %mask : <i32>, <i32> -> !ascvec.mask
                          %829 = ascvec.loadalign %arg53[%c192] : memref<528xi32, 1>, index -> <i32>
                          %830 = ascvec.select %829, %289, %828 : <i32>
                          %831 = ascvec.cmp_lt_scalar %830, %c256_i32_32, %828 : <i32>, i32 -> !ascvec.mask
                          %832 = ascvec.cmp_ge_scalar %830, %c256_i32_32, %828 : <i32>, i32 -> !ascvec.mask
                          %833 = ascvec.select %830, %289, %831 : <i32>
                          %834 = ascvec.add %833, %761, %mask : <i32>
                          %835 = ascvec.reinterpret %834 : <i32> -> <ui32>
                          %836 = ascvec.gather %arg34[%c0_35], %835, %mask : memref<8192xui32, 1>, <ui32>, !ascvec.mask -> <ui32>
                          %837 = ascvec.gather %arg70[%c0_35], %835, %mask : memref<8192xui32, 1>, <ui32>, !ascvec.mask -> <ui32>
                          %838 = ascvec.add_scalar %830, %c-256_i32, %mask : <i32>, i32
                          %839 = ascvec.select %838, %289, %832 : <i32>
                          %840 = ascvec.add %839, %761, %mask : <i32>
                          %841 = ascvec.reinterpret %840 : <i32> -> <ui32>
                          %842 = ascvec.gather %arg39[%c0_35], %841, %mask : memref<8192xf32, 1>, <ui32>, !ascvec.mask -> <f32>
                          %843 = ascvec.reinterpret %842 : <f32> -> <ui32>
                          %844 = ascvec.add_scalar %839, %arg71, %mask : <i32>, i32
                          %845 = ascvec.reinterpret %844 : <i32> -> <ui32>
                          %846 = ascvec.select %836, %843, %831 : <ui32>
                          %847 = ascvec.select %837, %845, %831 : <ui32>
                          %848 = ascvec.select %846, %290, %828 : <ui32>
                          ascvec.storealign %arg72[%c704], %848, %mask : memref<1024xui32, 1>, index, <ui32>
                          %849 = ascvec.select %847, %290, %828 : <ui32>
                          ascvec.storealign %arg73[%c704], %849, %mask : memref<1024xui32, 1>, index, <ui32>
                          %850 = ascvec.loadalign_brc %arg67[%arg78] : memref<512xi32, 1>, index -> <i32>
                          %851 = ascvec.duplicate_scalar %arg79, %mask : i32 -> <i32>
                          %852 = ascvec.cmp_lt %307, %850, %mask : <i32>, <i32> -> !ascvec.mask
                          %853 = ascvec.loadalign %arg62[%c0_35] : memref<528xi32, 1>, index -> <i32>
                          %854 = ascvec.select %853, %289, %852 : <i32>
                          %855 = ascvec.cmp_lt_scalar %854, %c256_i32_32, %852 : <i32>, i32 -> !ascvec.mask
                          %856 = ascvec.cmp_ge_scalar %854, %c256_i32_32, %852 : <i32>, i32 -> !ascvec.mask
                          %857 = ascvec.select %854, %289, %855 : <i32>
                          %858 = ascvec.add %857, %851, %mask : <i32>
                          %859 = ascvec.reinterpret %858 : <i32> -> <ui32>
                          %860 = ascvec.gather %arg34[%c0_35], %859, %mask : memref<8192xui32, 1>, <ui32>, !ascvec.mask -> <ui32>
                          %861 = ascvec.gather %arg70[%c0_35], %859, %mask : memref<8192xui32, 1>, <ui32>, !ascvec.mask -> <ui32>
                          %862 = ascvec.add_scalar %854, %c-256_i32, %mask : <i32>, i32
                          %863 = ascvec.select %862, %289, %856 : <i32>
                          %864 = ascvec.add %863, %851, %mask : <i32>
                          %865 = ascvec.reinterpret %864 : <i32> -> <ui32>
                          %866 = ascvec.gather %arg39[%c0_35], %865, %mask : memref<8192xf32, 1>, <ui32>, !ascvec.mask -> <f32>
                          %867 = ascvec.reinterpret %866 : <f32> -> <ui32>
                          %868 = ascvec.add_scalar %863, %arg71, %mask : <i32>, i32
                          %869 = ascvec.reinterpret %868 : <i32> -> <ui32>
                          %870 = ascvec.select %860, %867, %855 : <ui32>
                          %871 = ascvec.select %861, %869, %855 : <ui32>
                          %872 = ascvec.select %870, %290, %852 : <ui32>
                          ascvec.storealign %arg72[%c768], %872, %mask : memref<1024xui32, 1>, index, <ui32>
                          %873 = ascvec.select %871, %290, %852 : <ui32>
                          ascvec.storealign %arg73[%c768], %873, %mask : memref<1024xui32, 1>, index, <ui32>
                          %874 = ascvec.cmp_lt %311, %850, %mask : <i32>, <i32> -> !ascvec.mask
                          %875 = ascvec.loadalign %arg62[%c64] : memref<528xi32, 1>, index -> <i32>
                          %876 = ascvec.select %875, %289, %874 : <i32>
                          %877 = ascvec.cmp_lt_scalar %876, %c256_i32_32, %874 : <i32>, i32 -> !ascvec.mask
                          %878 = ascvec.cmp_ge_scalar %876, %c256_i32_32, %874 : <i32>, i32 -> !ascvec.mask
                          %879 = ascvec.select %876, %289, %877 : <i32>
                          %880 = ascvec.add %879, %851, %mask : <i32>
                          %881 = ascvec.reinterpret %880 : <i32> -> <ui32>
                          %882 = ascvec.gather %arg34[%c0_35], %881, %mask : memref<8192xui32, 1>, <ui32>, !ascvec.mask -> <ui32>
                          %883 = ascvec.gather %arg70[%c0_35], %881, %mask : memref<8192xui32, 1>, <ui32>, !ascvec.mask -> <ui32>
                          %884 = ascvec.add_scalar %876, %c-256_i32, %mask : <i32>, i32
                          %885 = ascvec.select %884, %289, %878 : <i32>
                          %886 = ascvec.add %885, %851, %mask : <i32>
                          %887 = ascvec.reinterpret %886 : <i32> -> <ui32>
                          %888 = ascvec.gather %arg39[%c0_35], %887, %mask : memref<8192xf32, 1>, <ui32>, !ascvec.mask -> <f32>
                          %889 = ascvec.reinterpret %888 : <f32> -> <ui32>
                          %890 = ascvec.add_scalar %885, %arg71, %mask : <i32>, i32
                          %891 = ascvec.reinterpret %890 : <i32> -> <ui32>
                          %892 = ascvec.select %882, %889, %877 : <ui32>
                          %893 = ascvec.select %883, %891, %877 : <ui32>
                          %894 = ascvec.select %892, %290, %874 : <ui32>
                          ascvec.storealign %arg72[%c832], %894, %mask : memref<1024xui32, 1>, index, <ui32>
                          %895 = ascvec.select %893, %290, %874 : <ui32>
                          ascvec.storealign %arg73[%c832], %895, %mask : memref<1024xui32, 1>, index, <ui32>
                          %896 = ascvec.cmp_lt %315, %850, %mask : <i32>, <i32> -> !ascvec.mask
                          %897 = ascvec.loadalign %arg62[%c128] : memref<528xi32, 1>, index -> <i32>
                          %898 = ascvec.select %897, %289, %896 : <i32>
                          %899 = ascvec.cmp_lt_scalar %898, %c256_i32_32, %896 : <i32>, i32 -> !ascvec.mask
                          %900 = ascvec.cmp_ge_scalar %898, %c256_i32_32, %896 : <i32>, i32 -> !ascvec.mask
                          %901 = ascvec.select %898, %289, %899 : <i32>
                          %902 = ascvec.add %901, %851, %mask : <i32>
                          %903 = ascvec.reinterpret %902 : <i32> -> <ui32>
                          %904 = ascvec.gather %arg34[%c0_35], %903, %mask : memref<8192xui32, 1>, <ui32>, !ascvec.mask -> <ui32>
                          %905 = ascvec.gather %arg70[%c0_35], %903, %mask : memref<8192xui32, 1>, <ui32>, !ascvec.mask -> <ui32>
                          %906 = ascvec.add_scalar %898, %c-256_i32, %mask : <i32>, i32
                          %907 = ascvec.select %906, %289, %900 : <i32>
                          %908 = ascvec.add %907, %851, %mask : <i32>
                          %909 = ascvec.reinterpret %908 : <i32> -> <ui32>
                          %910 = ascvec.gather %arg39[%c0_35], %909, %mask : memref<8192xf32, 1>, <ui32>, !ascvec.mask -> <f32>
                          %911 = ascvec.reinterpret %910 : <f32> -> <ui32>
                          %912 = ascvec.add_scalar %907, %arg71, %mask : <i32>, i32
                          %913 = ascvec.reinterpret %912 : <i32> -> <ui32>
                          %914 = ascvec.select %904, %911, %899 : <ui32>
                          %915 = ascvec.select %905, %913, %899 : <ui32>
                          %916 = ascvec.select %914, %290, %896 : <ui32>
                          ascvec.storealign %arg72[%c896], %916, %mask : memref<1024xui32, 1>, index, <ui32>
                          %917 = ascvec.select %915, %290, %896 : <ui32>
                          ascvec.storealign %arg73[%c896], %917, %mask : memref<1024xui32, 1>, index, <ui32>
                          %918 = ascvec.cmp_lt %319, %850, %mask : <i32>, <i32> -> !ascvec.mask
                          %919 = ascvec.loadalign %arg62[%c192] : memref<528xi32, 1>, index -> <i32>
                          %920 = ascvec.select %919, %289, %918 : <i32>
                          %921 = ascvec.cmp_lt_scalar %920, %c256_i32_32, %918 : <i32>, i32 -> !ascvec.mask
                          %922 = ascvec.cmp_ge_scalar %920, %c256_i32_32, %918 : <i32>, i32 -> !ascvec.mask
                          %923 = ascvec.select %920, %289, %921 : <i32>
                          %924 = ascvec.add %923, %851, %mask : <i32>
                          %925 = ascvec.reinterpret %924 : <i32> -> <ui32>
                          %926 = ascvec.gather %arg34[%c0_35], %925, %mask : memref<8192xui32, 1>, <ui32>, !ascvec.mask -> <ui32>
                          %927 = ascvec.gather %arg70[%c0_35], %925, %mask : memref<8192xui32, 1>, <ui32>, !ascvec.mask -> <ui32>
                          %928 = ascvec.add_scalar %920, %c-256_i32, %mask : <i32>, i32
                          %929 = ascvec.select %928, %289, %922 : <i32>
                          %930 = ascvec.add %929, %851, %mask : <i32>
                          %931 = ascvec.reinterpret %930 : <i32> -> <ui32>
                          %932 = ascvec.gather %arg39[%c0_35], %931, %mask : memref<8192xf32, 1>, <ui32>, !ascvec.mask -> <f32>
                          %933 = ascvec.reinterpret %932 : <f32> -> <ui32>
                          %934 = ascvec.add_scalar %929, %arg71, %mask : <i32>, i32
                          %935 = ascvec.reinterpret %934 : <i32> -> <ui32>
                          %936 = ascvec.select %926, %933, %921 : <ui32>
                          %937 = ascvec.select %927, %935, %921 : <ui32>
                          %938 = ascvec.select %936, %290, %918 : <ui32>
                          ascvec.storealign %arg72[%c960], %938, %mask : memref<1024xui32, 1>, index, <ui32>
                          %939 = ascvec.select %937, %290, %918 : <ui32>
                          ascvec.storealign %arg73[%c960], %939, %mask : memref<1024xui32, 1>, index, <ui32>
                        }
                        ascvec.mem_bar vst_vld
                        scf.for %arg81 = %c0_i32_36 to %c4_i32_31 step %c1_i32_30  : i32 {
                          %580 = arith.addi %arg80, %arg81 : i32
                          %581 = arith.muli %580, %c256_i32_32 : i32
                          %582 = arith.muli %arg81, %c256_i32_32 overflow<nsw> : i32
                          %583 = arith.index_cast %582 : i32 to index
                          %584 = ascvec.loadalign %arg72[%583] : memref<1024xui32, 1>, index -> <ui32>
                          %585 = arith.index_cast %581 : i32 to index
                          ascvec.storealign %arg34[%585], %584, %mask : memref<8192xui32, 1>, index, <ui32>
                          %586 = ascvec.loadalign %arg73[%583] : memref<1024xui32, 1>, index -> <ui32>
                          ascvec.storealign %arg70[%585], %586, %mask : memref<8192xui32, 1>, index, <ui32>
                          %587 = arith.addi %581, %c64_i32_37 : i32
                          %588 = arith.addi %582, %c64_i32_37 overflow<nsw> : i32
                          %589 = arith.index_cast %588 : i32 to index
                          %590 = ascvec.loadalign %arg72[%589] : memref<1024xui32, 1>, index -> <ui32>
                          %591 = arith.index_cast %587 : i32 to index
                          ascvec.storealign %arg34[%591], %590, %mask : memref<8192xui32, 1>, index, <ui32>
                          %592 = ascvec.loadalign %arg73[%589] : memref<1024xui32, 1>, index -> <ui32>
                          ascvec.storealign %arg70[%591], %592, %mask : memref<8192xui32, 1>, index, <ui32>
                          %593 = arith.addi %581, %c128_i32_34 : i32
                          %594 = arith.addi %582, %c128_i32_34 overflow<nsw> : i32
                          %595 = arith.index_cast %594 : i32 to index
                          %596 = ascvec.loadalign %arg72[%595] : memref<1024xui32, 1>, index -> <ui32>
                          %597 = arith.index_cast %593 : i32 to index
                          ascvec.storealign %arg34[%597], %596, %mask : memref<8192xui32, 1>, index, <ui32>
                          %598 = ascvec.loadalign %arg73[%595] : memref<1024xui32, 1>, index -> <ui32>
                          ascvec.storealign %arg70[%597], %598, %mask : memref<8192xui32, 1>, index, <ui32>
                          %599 = arith.addi %581, %c192_i32_33 : i32
                          %600 = arith.addi %582, %c192_i32_33 overflow<nsw> : i32
                          %601 = arith.index_cast %600 : i32 to index
                          %602 = ascvec.loadalign %arg72[%601] : memref<1024xui32, 1>, index -> <ui32>
                          %603 = arith.index_cast %599 : i32 to index
                          ascvec.storealign %arg34[%603], %602, %mask : memref<8192xui32, 1>, index, <ui32>
                          %604 = ascvec.loadalign %arg73[%601] : memref<1024xui32, 1>, index -> <ui32>
                          ascvec.storealign %arg70[%603], %604, %mask : memref<8192xui32, 1>, index, <ui32>
                        }
                      } {cannir.vf_mode = "simd"}
                    }
                  }
                  scf.yield %194#0, %194#1 : i64, i64
                }
                scf.yield %165#0, %165#1 : i64, i64
              } else {
                scf.yield %arg22, %arg23 : i64, i64
              }
              %156 = arith.cmpi eq, %144, %c32_i64 : i64
              scf.if %156 {
                ascvec.get_buf v, %c1_i64, 0 {cannir.buffer.generated_sync}
                ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                ascvec.vf(%alloc_15, %alloc_13, %alloc_18, %alloc_16 : memref<10240xi32, 1>, memref<512xi32, 1>, memref<8192xui32, 1>, memref<32xi32, 1>) {
                ^bb0(%arg24: memref<10240xi32, 1>, %arg25: memref<512xi32, 1>, %arg26: memref<8192xui32, 1>, %arg27: memref<32xi32, 1>):
                  %c32_i32_30 = arith.constant 32 : i32
                  %c256_i32_31 = arith.constant 256 : i32
                  %c-4_i32 = arith.constant -4 : i32
                  %c4_i32_32 = arith.constant 4 : i32
                  %c3_i32_33 = arith.constant 3 : i32
                  %c2_i32_34 = arith.constant 2 : i32
                  %c16_i32_35 = arith.constant 16 : i32
                  %c1_i32_36 = arith.constant 1 : i32
                  %c5_i32_37 = arith.constant 5 : i32
                  %c0_i32_38 = arith.constant 0 : i32
                  %c320_i32 = arith.constant 320 : i32
                  %c-1_i32 = arith.constant -1 : i32
                  %c64_i32_39 = arith.constant 64 : i32
                  scf.for %arg28 = %c0_i32_38 to %c32_i32_30 step %c1_i32_36  : i32 {
                    %mask, %remaining_out = ascvec.update_mask %c64_i32_39 : !ascvec.mask
                    %194 = ascvec.duplicate_scalar %c-1_i32, %mask : i32 -> <i32>
                    %195 = arith.muli %arg28, %c320_i32 overflow<nsw> : i32
                    scf.for %arg29 = %c0_i32_38 to %c5_i32_37 step %c1_i32_36  : i32 {
                      %223 = arith.muli %arg29, %c64_i32_39 overflow<nsw> : i32
                      %224 = arith.addi %195, %223 overflow<nsw> : i32
                      %225 = arith.index_cast %224 : i32 to index
                      ascvec.storealign %arg24[%225], %194, %mask : memref<10240xi32, 1>, index, <i32>
                    }
                    %196 = arith.muli %arg28, %c16_i32_35 overflow<nsw> : i32
                    %197 = arith.index_cast %196 : i32 to index
                    %198 = ascvec.loadalign_brc %arg25[%197] : memref<512xi32, 1>, index -> <i32>
                    %199 = arith.addi %196, %c1_i32_36 overflow<nsw> : i32
                    %200 = arith.index_cast %199 : i32 to index
                    %201 = ascvec.loadalign_brc %arg25[%200] : memref<512xi32, 1>, index -> <i32>
                    %202 = arith.addi %196, %c2_i32_34 overflow<nsw> : i32
                    %203 = arith.index_cast %202 : i32 to index
                    %204 = ascvec.loadalign_brc %arg25[%203] : memref<512xi32, 1>, index -> <i32>
                    %205 = arith.addi %196, %c3_i32_33 overflow<nsw> : i32
                    %206 = arith.index_cast %205 : i32 to index
                    %207 = ascvec.loadalign_brc %arg25[%206] : memref<512xi32, 1>, index -> <i32>
                    %208 = ascvec.cmp_ne_scalar %207, %c0_i32_38, %mask : <i32>, i32 -> !ascvec.mask
                    %209 = ascvec.cmp_eq_scalar %207, %c0_i32_38, %mask : <i32>, i32 -> !ascvec.mask
                    %210 = ascvec.min_scalar %198, %c4_i32_32, %mask : <i32>, i32
                    %211 = ascvec.add_scalar %198, %c-4_i32, %mask : <i32>, i32
                    %212 = ascvec.max_scalar %211, %c0_i32_38, %mask : <i32>, i32
                    %213 = ascvec.max %212, %210, %mask : <i32>
                    %214 = ascvec.sub %198, %213, %mask : <i32>
                    ascvec.store_unalign_begin %arg24 : memref<10240xi32, 1>
                    scf.for %arg29 = %c0_i32_38 to %c5_i32_37 step %c1_i32_36  : i32 {
                      %223 = arith.muli %arg29, %c64_i32_39 overflow<nsw> : i32
                      %224 = ascvec.arange %223 : i32 -> <i32>
                      %225 = ascvec.cmp_lt %224, %198, %208 : <i32>, <i32> -> !ascvec.mask
                      %226 = ascvec.squeeze_store_reg %224, %225 : <i32>, !ascvec.mask -> <i32>
                      %227 = arith.index_cast %195 : i32 to index
                      ascvec.store_unalign %arg24[%227], %226 : memref<10240xi32, 1>, <i32>
                    }
                    %215 = ascvec.arange %c0_i32_38 : i32 -> <i32>
                    %216 = ascvec.cmp_lt %215, %210, %209 : <i32>, <i32> -> !ascvec.mask
                    %217 = ascvec.squeeze_store_reg %215, %216 : <i32>, !ascvec.mask -> <i32>
                    %218 = arith.index_cast %195 : i32 to index
                    ascvec.store_unalign %arg24[%218], %217 : memref<10240xi32, 1>, <i32>
                    scf.for %arg29 = %c0_i32_38 to %c4_i32_32 step %c1_i32_36  : i32 {
                      %223 = arith.muli %arg29, %c64_i32_39 overflow<nsw> : i32
                      %224 = ascvec.arange %223 : i32 -> <i32>
                      %225 = arith.muli %arg28, %c256_i32_31 overflow<nsw> : i32
                      %226 = arith.addi %225, %223 overflow<nsw> : i32
                      %227 = arith.index_cast %226 : i32 to index
                      %228 = ascvec.loadalign %arg26[%227] : memref<8192xui32, 1>, index -> <ui32>
                      %229 = ascvec.reinterpret %228 : <ui32> -> <i32>
                      %230 = ascvec.cmp_lt %224, %201, %209 : <i32>, <i32> -> !ascvec.mask
                      %231 = ascvec.squeeze_store_reg %229, %230 : <i32>, !ascvec.mask -> <i32>
                      ascvec.store_unalign %arg24[%218], %231 : memref<10240xi32, 1>, <i32>
                    }
                    %219 = ascvec.cmp_lt %215, %214, %209 : <i32>, <i32> -> !ascvec.mask
                    %220 = ascvec.add %215, %213, %mask : <i32>
                    %221 = ascvec.squeeze_store_reg %220, %219 : <i32>, !ascvec.mask -> <i32>
                    ascvec.store_unalign %arg24[%218], %221 : memref<10240xi32, 1>, <i32>
                    ascvec.store_unalign_post %arg24[%218] : memref<10240xi32, 1>
                    %222 = arith.index_cast %arg28 : i32 to index
                    ascvec.storealign_1st %arg27[%222], %204 : memref<32xi32, 1>, index, <i32>
                  }
                } {cannir.vf_mode = "simd"}
                ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                ascvec.release_buf v, %c1_i64, 0 {cannir.buffer.generated_sync}
                %157 = cannir.extract_physical_layout %arg7 : !cannir.tensor<<gm, i32>, <<(-1, -1, -1, -1)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, -1)>, <(-1, -1, -1, 1)>>, identity>
                %158 = cannir.extract_stride %157 : <<(-1, -1, -1, -1)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                %159 = cannir.extract_dim %158[0] : !cannir.stride<(-1, -1, -1, 1)>
                %160 = arith.muli %40, %159 : i64
                %161 = cannir.extract_dim %158[1] : !cannir.stride<(-1, -1, -1, 1)>
                %162 = arith.muli %91, %161 : i64
                %163 = arith.addi %160, %162 : i64
                %164 = cannir.extract_dim %158[2] : !cannir.stride<(-1, -1, -1, 1)>
                %165 = arith.muli %146, %164 : i64
                %166 = arith.addi %163, %165 : i64
                %167 = arith.muli %166, %c32_i64 : i64
                %168 = ascscalar.memref_address %alloc_6 : memref<1xi32>
                %169 = arith.divui %167, %c8_i64 : i64
                %170 = arith.addi %168, %169 : i64
                %171 = ascscalar.bind_gm_ptr %170 : memref<1xi32>
                %172 = ascscalar.bitcast_memref %alloc_15 byte_offset = %c0_i64 : memref<10240xi32, 1> -> memref<?xi32, 1>
                %173 = cannir.extract_layout %arg7 : !cannir.tensor<<gm, i32>, <<(-1, -1, -1, -1)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, -1)>, <(-1, -1, -1, 1)>>, identity>
                %174 = cannir.extract_stride %173 : <<(-1, -1, -1, -1)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                %175 = cannir.extract_dim %174[2] : !cannir.stride<(-1, -1, -1, 1)>
                %176 = arith.index_cast %175 : i64 to index
                %177 = arith.muli %176, %c4 : index
                %178 = arith.index_cast %177 : index to i64
                ascvec.get_buf mte3, %c1_i64, 0 {cannir.buffer.generated_sync}
                ascvec.copy_ub2gm %171[%c0], %172[%c0], %c32_i32, %c1280_i32, %178, %c1280_i32 l2_cache_ctl %c0_i32 : memref<1xi32>, memref<?xi32, 1>
                ascvec.release_buf mte3, %c1_i64, 0 {cannir.buffer.generated_sync}
                %179 = cannir.extract_physical_layout %arg8 : !cannir.tensor<<gm, i32>, <<(-1, -1, -1, 8)>, <(-1, -1, 8, 1)>>, <<(-1, -1, -1, 8)>, <(-1, -1, 8, 1)>>, identity>
                %180 = cannir.extract_stride %179 : <<(-1, -1, -1, 8)>, <(-1, -1, 8, 1)>> -> <(-1, -1, 8, 1)>
                %181 = cannir.extract_dim %180[0] : !cannir.stride<(-1, -1, 8, 1)>
                %182 = arith.muli %40, %181 : i64
                %183 = cannir.extract_dim %180[1] : !cannir.stride<(-1, -1, 8, 1)>
                %184 = arith.muli %91, %183 : i64
                %185 = arith.addi %182, %184 : i64
                %186 = arith.muli %146, %c8_i64 : i64
                %187 = arith.addi %185, %186 : i64
                %188 = arith.muli %187, %c32_i64 : i64
                %189 = ascscalar.memref_address %alloc_7 : memref<1xi32>
                %190 = arith.divui %188, %c8_i64 : i64
                %191 = arith.addi %189, %190 : i64
                %192 = ascscalar.bind_gm_ptr %191 : memref<1xi32>
                %193 = ascscalar.bitcast_memref %alloc_16 byte_offset = %c0_i64 : memref<32xi32, 1> -> memref<?xi32, 1>
                ascvec.get_buf mte3, %c2_i64, 0 {cannir.buffer.generated_sync}
                ascvec.copy_ub2gm %192[%c0], %193[%c0], %c32_i32, %c4_i32, %c32_i64, %c4_i32 l2_cache_ctl %c0_i32 : memref<1xi32>, memref<?xi32, 1>
                ascvec.release_buf mte3, %c2_i64, 0 {cannir.buffer.generated_sync}
              } else {
                scf.for %arg24 = %c0_i64 to %144 step %c1_i64  : i64 {
                  %157 = arith.muli %arg24, %c16_i64 : i64
                  %158 = arith.index_cast %157 : i64 to index
                  %159 = arith.addi %157, %c1_i64 : i64
                  %160 = arith.index_cast %159 : i64 to index
                  %161 = arith.addi %157, %c2_i64 : i64
                  %162 = arith.index_cast %161 : i64 to index
                  %163 = arith.addi %157, %c3_i64 : i64
                  %164 = arith.index_cast %163 : i64 to index
                  ascvec.get_buf v, %c1_i64, 0 {cannir.buffer.generated_sync}
                  ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                  ascvec.vf(%alloc_15, %alloc_13, %158, %160, %162, %164, %arg24, %alloc_18, %alloc_16 : memref<10240xi32, 1>, memref<512xi32, 1>, index, index, index, index, i64, memref<8192xui32, 1>, memref<32xi32, 1>) {
                  ^bb0(%arg25: memref<10240xi32, 1>, %arg26: memref<512xi32, 1>, %arg27: index, %arg28: index, %arg29: index, %arg30: index, %arg31: i64, %arg32: memref<8192xui32, 1>, %arg33: memref<32xi32, 1>):
                    %c256_i64_30 = arith.constant 256 : i64
                    %c-4_i32 = arith.constant -4 : i32
                    %c4_i32_31 = arith.constant 4 : i32
                    %c1_i32_32 = arith.constant 1 : i32
                    %c5_i32_33 = arith.constant 5 : i32
                    %c0_i32_34 = arith.constant 0 : i32
                    %c0_35 = arith.constant 0 : index
                    %c-1_i32 = arith.constant -1 : i32
                    %c64_i32_36 = arith.constant 64 : i32
                    %mask, %remaining_out = ascvec.update_mask %c64_i32_36 : !ascvec.mask
                    %203 = ascvec.duplicate_scalar %c-1_i32, %mask : i32 -> <i32>
                    scf.for %arg34 = %c0_i32_34 to %c5_i32_33 step %c1_i32_32  : i32 {
                      %218 = arith.muli %arg34, %c64_i32_36 overflow<nsw> : i32
                      %219 = arith.index_cast %218 : i32 to index
                      ascvec.storealign %arg25[%219], %203, %mask : memref<10240xi32, 1>, index, <i32>
                    }
                    %204 = ascvec.loadalign_brc %arg26[%arg27] : memref<512xi32, 1>, index -> <i32>
                    %205 = ascvec.loadalign_brc %arg26[%arg28] : memref<512xi32, 1>, index -> <i32>
                    %206 = ascvec.loadalign_brc %arg26[%arg29] : memref<512xi32, 1>, index -> <i32>
                    %207 = ascvec.loadalign_brc %arg26[%arg30] : memref<512xi32, 1>, index -> <i32>
                    %208 = ascvec.cmp_ne_scalar %207, %c0_i32_34, %mask : <i32>, i32 -> !ascvec.mask
                    %209 = ascvec.cmp_eq_scalar %207, %c0_i32_34, %mask : <i32>, i32 -> !ascvec.mask
                    %210 = ascvec.min_scalar %204, %c4_i32_31, %mask : <i32>, i32
                    %211 = ascvec.add_scalar %204, %c-4_i32, %mask : <i32>, i32
                    %212 = ascvec.max_scalar %211, %c0_i32_34, %mask : <i32>, i32
                    %213 = ascvec.max %212, %210, %mask : <i32>
                    %214 = ascvec.sub %204, %213, %mask : <i32>
                    ascvec.store_unalign_begin %arg25 : memref<10240xi32, 1>
                    scf.for %arg34 = %c0_i32_34 to %c5_i32_33 step %c1_i32_32  : i32 {
                      %218 = arith.muli %arg34, %c64_i32_36 overflow<nsw> : i32
                      %219 = ascvec.arange %218 : i32 -> <i32>
                      %220 = ascvec.cmp_lt %219, %204, %208 : <i32>, <i32> -> !ascvec.mask
                      %221 = ascvec.squeeze_store_reg %219, %220 : <i32>, !ascvec.mask -> <i32>
                      ascvec.store_unalign %arg25[%c0_35], %221 : memref<10240xi32, 1>, <i32>
                    }
                    %215 = ascvec.arange %c0_i32_34 : i32 -> <i32>
                    %216 = ascvec.cmp_lt %215, %210, %209 : <i32>, <i32> -> !ascvec.mask
                    %217 = ascvec.squeeze_store_reg %215, %216 : <i32>, !ascvec.mask -> <i32>
                    ascvec.store_unalign %arg25[%c0_35], %217 : memref<10240xi32, 1>, <i32>
                    scf.for %arg34 = %c0_i32_34 to %c4_i32_31 step %c1_i32_32  : i32 {
                      %218 = arith.muli %arg34, %c64_i32_36 overflow<nsw> : i32
                      %219 = ascvec.arange %218 : i32 -> <i32>
                      %220 = arith.muli %arg31, %c256_i64_30 : i64
                      %221 = arith.extsi %218 : i32 to i64
                      %222 = arith.addi %220, %221 : i64
                      %223 = arith.index_cast %222 : i64 to index
                      %224 = ascvec.loadalign %arg32[%223] : memref<8192xui32, 1>, index -> <ui32>
                      %225 = ascvec.reinterpret %224 : <ui32> -> <i32>
                      %226 = ascvec.cmp_lt %219, %205, %209 : <i32>, <i32> -> !ascvec.mask
                      %227 = ascvec.squeeze_store_reg %225, %226 : <i32>, !ascvec.mask -> <i32>
                      ascvec.store_unalign %arg25[%c0_35], %227 : memref<10240xi32, 1>, <i32>
                    }
                    %c0_37 = arith.constant 0 : index
                    %c1 = arith.constant 1 : index
                    scf.for %arg34 = %c0_37 to %c1 step %c1 {
                      %218 = ascvec.cmp_lt %215, %214, %209 : <i32>, <i32> -> !ascvec.mask
                      %219 = ascvec.add %215, %213, %mask : <i32>
                      %220 = ascvec.squeeze_store_reg %219, %218 : <i32>, !ascvec.mask -> <i32>
                      ascvec.store_unalign %arg25[%c0_35], %220 : memref<10240xi32, 1>, <i32>
                      ascvec.store_unalign_post %arg25[%c0_35] : memref<10240xi32, 1>
                      ascvec.storealign_1st %arg33[%c0_35], %206 : memref<32xi32, 1>, index, <i32>
                    }
                  } {cannir.vf_mode = "simd"}
                  ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                  ascvec.release_buf v, %c1_i64, 0 {cannir.buffer.generated_sync}
                  %165 = arith.addi %146, %arg24 : i64
                  %166 = cannir.extract_physical_layout %arg7 : !cannir.tensor<<gm, i32>, <<(-1, -1, -1, -1)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, -1)>, <(-1, -1, -1, 1)>>, identity>
                  %167 = cannir.extract_stride %166 : <<(-1, -1, -1, -1)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                  %168 = cannir.extract_dim %167[0] : !cannir.stride<(-1, -1, -1, 1)>
                  %169 = arith.muli %40, %168 : i64
                  %170 = cannir.extract_dim %167[1] : !cannir.stride<(-1, -1, -1, 1)>
                  %171 = arith.muli %91, %170 : i64
                  %172 = arith.addi %169, %171 : i64
                  %173 = cannir.extract_dim %167[2] : !cannir.stride<(-1, -1, -1, 1)>
                  %174 = arith.muli %165, %173 : i64
                  %175 = arith.addi %172, %174 : i64
                  %176 = arith.muli %175, %c32_i64 : i64
                  %177 = ascscalar.memref_address %alloc_6 : memref<1xi32>
                  %178 = arith.divui %176, %c8_i64 : i64
                  %179 = arith.addi %177, %178 : i64
                  %180 = ascscalar.bind_gm_ptr %179 : memref<1xi32>
                  %181 = ascscalar.bitcast_memref %alloc_15 byte_offset = %c0_i64 : memref<10240xi32, 1> -> memref<?xi32, 1>
                  %182 = cannir.extract_layout %arg7 : !cannir.tensor<<gm, i32>, <<(-1, -1, -1, -1)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, -1)>, <(-1, -1, -1, 1)>>, identity>
                  %183 = cannir.extract_stride %182 : <<(-1, -1, -1, -1)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
                  %184 = cannir.extract_dim %183[2] : !cannir.stride<(-1, -1, -1, 1)>
                  %185 = arith.index_cast %184 : i64 to index
                  %186 = arith.muli %185, %c4 : index
                  %187 = arith.index_cast %186 : index to i64
                  ascvec.get_buf mte3, %c1_i64, 0 {cannir.buffer.generated_sync}
                  ascvec.copy_ub2gm %180[%c0], %181[%c0], %c1_i32, %c1280_i32, %187, %c1280_i32 l2_cache_ctl %c0_i32 : memref<1xi32>, memref<?xi32, 1>
                  ascvec.release_buf mte3, %c1_i64, 0 {cannir.buffer.generated_sync}
                  %188 = cannir.extract_physical_layout %arg8 : !cannir.tensor<<gm, i32>, <<(-1, -1, -1, 8)>, <(-1, -1, 8, 1)>>, <<(-1, -1, -1, 8)>, <(-1, -1, 8, 1)>>, identity>
                  %189 = cannir.extract_stride %188 : <<(-1, -1, -1, 8)>, <(-1, -1, 8, 1)>> -> <(-1, -1, 8, 1)>
                  %190 = cannir.extract_dim %189[0] : !cannir.stride<(-1, -1, 8, 1)>
                  %191 = arith.muli %40, %190 : i64
                  %192 = cannir.extract_dim %189[1] : !cannir.stride<(-1, -1, 8, 1)>
                  %193 = arith.muli %91, %192 : i64
                  %194 = arith.addi %191, %193 : i64
                  %195 = arith.muli %165, %c8_i64 : i64
                  %196 = arith.addi %194, %195 : i64
                  %197 = arith.muli %196, %c32_i64 : i64
                  %198 = ascscalar.memref_address %alloc_7 : memref<1xi32>
                  %199 = arith.divui %197, %c8_i64 : i64
                  %200 = arith.addi %198, %199 : i64
                  %201 = ascscalar.bind_gm_ptr %200 : memref<1xi32>
                  %202 = ascscalar.bitcast_memref %alloc_16 byte_offset = %c0_i64 : memref<32xi32, 1> -> memref<?xi32, 1>
                  ascvec.get_buf mte3, %c2_i64, 0 {cannir.buffer.generated_sync}
                  ascvec.copy_ub2gm %201[%c0], %202[%c0], %c1_i32, %c4_i32, %c0_i64, %c4_i32 l2_cache_ctl %c0_i32 : memref<1xi32>, memref<?xi32, 1>
                  ascvec.release_buf mte3, %c2_i64, 0 {cannir.buffer.generated_sync}
                }
              }
              scf.yield %155#0, %155#1 : i64, i64
            } else {
              scf.yield %arg22, %arg23 : i64, i64
            }
            scf.yield %89#0, %89#1 : i64, i64
          }
          scf.yield %81#0, %81#1 : i64, i64
        }
        scf.yield %77#0, %77#1 : i64, i64
      }
      scf.yield %31#0, %31#1 : i64, i64
    }
    return
  }
  func.func @kernel_StemIndexerKernel___call___0_aic(%arg0: !cannir.tensor<<gm, bf16>, <<(-1, -1, 2, -1, 1024)>, <(-1, -1, 1024, 2048, 1)>>, <<(-1, -1, 2, -1, 1024)>, <(-1, -1, 1024, 2048, 1)>>, identity>, %arg1: !cannir.tensor<<gm, bf16>, <<(-1, -1, 2, -1, 1024)>, <(-1, -1, 1024, 2048, 1)>>, <<(-1, -1, 2, -1, 1024)>, <(-1, -1, 1024, 2048, 1)>>, identity>, %arg2: !cannir.tensor<<gm, f32>, <<(-1, -1, -1)>, <(-1, -1, 1)>>, <<(-1, -1, -1)>, <(-1, -1, 1)>>, identity>, %arg3: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg4: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg5: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg6: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg7: !cannir.tensor<<gm, i32>, <<(-1, -1, -1, -1)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, -1)>, <(-1, -1, -1, 1)>>, identity>, %arg8: !cannir.tensor<<gm, i32>, <<(-1, -1, -1, 8)>, <(-1, -1, 8, 1)>>, <<(-1, -1, -1, 8)>, <(-1, -1, 8, 1)>>, identity>, %arg9: i64, %arg10: i64, %arg11: f32) attributes {cannir.buffer.auto_sync, cannir.buffer.sync_analyzed, cannir.buffer.sync_verified, cannir.buffer.tensor_state, cannir.core = "aic", cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.device, cannir.explicit_ring_buffer_cursors, cannir.kernel_group = "kernel_StemIndexerKernel___call___0", cannir.resource.ids_prepared} {
    %c48_i32 = arith.constant 48 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %c2_i64 = arith.constant 2 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
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
    %c255_i64 = arith.constant 255 : i64
    %c4096_i64 = arith.constant 4096 : i64
    %c63_i64 = arith.constant 63 : i64
    %c127_i64 = arith.constant 127 : i64
    %c1024 = arith.constant 1024 : index
    %c2048_i64 = arith.constant 2048 : i64
    %c0 = arith.constant 0 : index
    %c256_i32 = arith.constant 256 : i32
    %c64_i32 = arith.constant 64 : i32
    %c16_i32 = arith.constant 16 : i32
    %c4_i32 = arith.constant 4 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c256_i64 = arith.constant 256 : i64
    %cst = arith.constant 1.000000e-01 : f32
    %cst_0 = arith.constant 3.000000e+01 : f32
    %cst_1 = arith.constant 2.000000e-01 : f32
    %c160_i64 = arith.constant 160 : i64
    %c56_i64 = arith.constant 56 : i64
    %c64_i64 = arith.constant 64 : i64
    %c128_i64 = arith.constant 128 : i64
    %c16_i64 = arith.constant 16 : i64
    %c576_i64 = arith.constant 576 : i64
    %c17_i64 = arith.constant 17 : i64
    %c18_i64 = arith.constant 18 : i64
    %alloc = memref.alloc() {cannir.arg_index = 0 : i64} : memref<1xbf16>
    %alloc_2 = memref.alloc() {cannir.arg_index = 1 : i64} : memref<1xbf16>
    %alloc_3 = memref.alloc() {cannir.arg_index = 3 : i64} : memref<1xi32>
    %alloc_4 = memref.alloc() {cannir.arg_index = 4 : i64} : memref<1xi32>
    %alloc_5 = memref.alloc() {cannir.arg_index = 5 : i64} : memref<1xi32>
    %alloc_6 = memref.alloc() {cannir.arg_index = 6 : i64} : memref<1xi32>
    %alloc_7 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<2048xf32, 1>
    %alloc_8 = memref.alloc() {cannir.byte_offset = 8192 : i64} : memref<2048xf32, 1>
    %alloc_9 = memref.alloc() {cannir.byte_offset = 16384 : i64} : memref<2048xf32, 1>
    %alloc_10 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<16384xbf16, 3>
    %alloc_11 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<16384xbf16, 4>
    %alloc_12 = memref.alloc() {cannir.byte_offset = 32768 : i64} : memref<16384xbf16, 3>
    %alloc_13 = memref.alloc() {cannir.byte_offset = 32768 : i64} : memref<16384xbf16, 4>
    %alloc_14 = memref.alloc() {cannir.byte_offset = 393216 : i64} : memref<65536xbf16, 2>
    %alloc_15 = memref.alloc() {cannir.byte_offset = 131072 : i64} : memref<65536xbf16, 2>
    %alloc_16 = memref.alloc() {cannir.byte_offset = 262144 : i64} : memref<65536xbf16, 2>
    %alloc_17 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<65536xbf16, 2>
    %alloc_18 = memref.alloc() {cannir.byte_offset = 16384 : i64} : memref<4096xf32, 5>
    %alloc_19 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<4096xf32, 5>
    %0 = ascscalar.get_block_idx
    %1 = memref.load %alloc_3[%c0] : memref<1xi32>
    %2 = arith.extsi %1 : i32 to i64
    %3:2 = scf.for %arg12 = %c0_i64 to %2 step %c1_i64 iter_args(%arg13 = %c0_i64, %arg14 = %c0_i64) -> (i64, i64)  : i64 {
      %4 = arith.muli %arg12, %c576_i64 : i64
      %5 = arith.addi %4, %c16_i64 : i64
      %6 = arith.muli %0, %c16_i64 : i64
      %7 = arith.addi %5, %6 : i64
      %8 = arith.index_cast %7 : i64 to index
      %9 = memref.load %alloc_3[%8] : memref<1xi32>
      %10 = arith.extsi %9 : i32 to i64
      %11 = arith.addi %7, %c1_i64 : i64
      %12 = arith.index_cast %11 : i64 to index
      %13 = memref.load %alloc_3[%12] : memref<1xi32>
      %14 = arith.extsi %13 : i32 to i64
      %15 = arith.addi %7, %c3_i64 : i64
      %16 = arith.index_cast %15 : i64 to index
      %17 = memref.load %alloc_3[%16] : memref<1xi32>
      %18 = arith.extsi %17 : i32 to i64
      %19 = arith.addi %7, %c4_i64 : i64
      %20 = arith.index_cast %19 : i64 to index
      %21 = memref.load %alloc_3[%20] : memref<1xi32>
      %22 = arith.extsi %21 : i32 to i64
      %23 = arith.addi %18, %c1_i64 : i64
      %24 = cannir.extract_layout %arg4 : !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>
      %25 = cannir.extract_shape %24 : <<(-1)>, <(1)>> -> <(-1)>
      %26 = cannir.extract_dim %25[0] : !cannir.shape<(-1)>
      %27 = arith.muli %26, %arg9 : i64
      %28 = arith.minsi %23, %27 : i64
      %29:2 = scf.for %arg15 = %10 to %28 step %c1_i64 iter_args(%arg16 = %arg13, %arg17 = %arg14) -> (i64, i64)  : i64 {
        %30 = arith.remsi %arg15, %arg9 : i64
        %31 = arith.cmpi ne, %30, %c0_i64 : i64
        %32 = arith.cmpi slt, %30, %c0_i64 : i64
        %33 = arith.cmpi slt, %arg9, %c0_i64 : i64
        %34 = arith.xori %32, %33 : i1
        %35 = arith.andi %31, %34 : i1
        %36 = arith.divsi %arg15, %arg9 : i64
        %37 = arith.extui %35 : i1 to i64
        %38 = arith.subi %36, %37 : i64
        %39 = arith.muli %38, %arg9 : i64
        %40 = arith.subi %arg15, %39 : i64
        %41 = arith.index_cast %38 : i64 to index
        %42 = memref.load %alloc_4[%41] : memref<1xi32>
        %43 = arith.extsi %42 : i32 to i64
        %44 = memref.load %alloc_5[%41] : memref<1xi32>
        %45 = arith.extsi %44 : i32 to i64
        %46 = arith.addi %43, %c127_i64 : i64
        %47 = arith.remsi %46, %c128_i64 : i64
        %48 = arith.cmpi ne, %47, %c0_i64 : i64
        %49 = arith.cmpi slt, %47, %c0_i64 : i64
        %50 = arith.andi %48, %49 : i1
        %51 = arith.divsi %46, %c128_i64 : i64
        %52 = arith.extui %50 : i1 to i64
        %53 = arith.subi %51, %52 : i64
        %54 = arith.addi %45, %c127_i64 : i64
        %55 = arith.remsi %54, %c128_i64 : i64
        %56 = arith.cmpi ne, %55, %c0_i64 : i64
        %57 = arith.cmpi slt, %55, %c0_i64 : i64
        %58 = arith.andi %56, %57 : i1
        %59 = arith.divsi %54, %c128_i64 : i64
        %60 = arith.extui %58 : i1 to i64
        %61 = arith.subi %59, %60 : i64
        %62 = arith.cmpi eq, %arg15, %10 : i64
        %63 = arith.select %62, %14, %c0_i64 : i64
        %64 = arith.muli %arg10, %53 : i64
        %65 = arith.addi %64, %c63_i64 : i64
        %66 = arith.remsi %65, %c64_i64 : i64
        %67 = arith.cmpi ne, %66, %c0_i64 : i64
        %68 = arith.cmpi slt, %66, %c0_i64 : i64
        %69 = arith.andi %67, %68 : i1
        %70 = arith.divsi %65, %c64_i64 : i64
        %71 = arith.extui %69 : i1 to i64
        %72 = arith.subi %70, %71 : i64
        %73 = arith.cmpi eq, %arg15, %18 : i64
        %74 = arith.select %73, %22, %72 : i64
        %75:2 = scf.for %arg18 = %63 to %74 step %c1_i64 iter_args(%arg19 = %arg16, %arg20 = %arg17) -> (i64, i64)  : i64 {
          %76 = arith.muli %arg18, %c64_i64 : i64
          %77 = arith.addi %76, %c64_i64 : i64
          %78 = arith.minsi %77, %64 : i64
          %79:2 = scf.for %arg21 = %c0_i64 to %arg10 step %c1_i64 iter_args(%arg22 = %arg19, %arg23 = %arg20) -> (i64, i64)  : i64 {
            %80 = arith.muli %arg21, %53 : i64
            %81 = arith.addi %80, %53 : i64
            %82 = arith.maxsi %76, %80 : i64
            %83 = arith.minsi %78, %81 : i64
            %84 = arith.subi %83, %82 : i64
            %85 = arith.maxsi %84, %c0_i64 : i64
            %86 = arith.cmpi sgt, %85, %c0_i64 : i64
            %87:2 = scf.if %86 -> (i64, i64) {
              %88 = arith.muli %40, %arg10 : i64
              %89 = arith.addi %88, %arg21 : i64
              %90 = arith.subi %82, %80 : i64
              %91 = arith.addi %90, %85 : i64
              %92 = arith.subi %91, %c1_i64 : i64
              %93 = memref.load %alloc_4[%41] : memref<1xi32>
              %94 = arith.extsi %93 : i32 to i64
              %95 = memref.load %alloc_5[%41] : memref<1xi32>
              %96 = arith.extsi %95 : i32 to i64
              %97 = memref.load %alloc_6[%41] : memref<1xi32>
              %98 = arith.extsi %97 : i32 to i64
              %99 = arith.addi %98, %c127_i64 : i64
              %100 = arith.remsi %99, %c128_i64 : i64
              %101 = arith.cmpi ne, %100, %c0_i64 : i64
              %102 = arith.cmpi slt, %100, %c0_i64 : i64
              %103 = arith.andi %101, %102 : i1
              %104 = arith.divsi %99, %c128_i64 : i64
              %105 = arith.extui %103 : i1 to i64
              %106 = arith.subi %104, %105 : i64
              %107 = arith.cmpi eq, %94, %c1_i64 : i64
              %108 = arith.cmpi sge, %98, %96 : i64
              %109 = arith.extui %108 : i1 to i64
              %110 = arith.select %107, %109, %c0_i64 : i64
              %111 = arith.cmpi eq, %110, %c0_i64 : i64
              %112 = arith.subi %61, %53 : i64
              %113 = arith.addi %112, %92 : i64
              %114 = arith.addi %113, %c1_i64 : i64
              %115 = arith.minsi %114, %61 : i64
              %116 = arith.maxsi %115, %c0_i64 : i64
              %117 = arith.select %111, %116, %61 : i64
              %118 = arith.cmpi sge, %106, %c56_i64 : i64
              %119 = scf.if %118 -> (i64) {
                %141 = arith.cmpi slt, %106, %c160_i64 : i64
                %142 = scf.if %141 -> (i64) {
                  %143 = arith.sitofp %106 : i64 to f32
                  %144 = arith.mulf %143, %cst_1 : f32
                  %145 = arith.addf %144, %cst_0 : f32
                  %146 = arith.fptosi %145 : f32 to i64
                  scf.yield %146 : i64
                } else {
                  %143 = arith.sitofp %106 : i64 to f32
                  %144 = arith.mulf %143, %cst : f32
                  %145 = arith.addf %144, %cst_0 : f32
                  %146 = arith.fptosi %145 : f32 to i64
                  scf.yield %146 : i64
                }
                scf.yield %142 : i64
              } else {
                scf.yield %106 : i64
              }
              %120 = arith.minsi %119, %c256_i64 : i64
              %121 = arith.minsi %117, %c4_i64 : i64
              %122 = arith.subi %117, %c4_i64 : i64
              %123 = arith.maxsi %122, %121 : i64
              %124 = arith.addi %121, %117 : i64
              %125 = arith.subi %124, %123 : i64
              %126 = arith.subi %117, %125 : i64
              %127 = arith.maxsi %126, %c0_i64 : i64
              %128 = arith.maxsi %122, %c4_i64 : i64
              %129 = arith.addi %128, %c63_i64 : i64
              %130 = arith.remsi %129, %c64_i64 : i64
              %131 = arith.cmpi ne, %130, %c0_i64 : i64
              %132 = arith.cmpi slt, %130, %c0_i64 : i64
              %133 = arith.andi %131, %132 : i1
              %134 = arith.divsi %129, %c64_i64 : i64
              %135 = arith.extui %133 : i1 to i64
              %136 = arith.subi %134, %135 : i64
              %137 = arith.muli %136, %c64_i64 : i64
              %138 = arith.minsi %61, %137 : i64
              %139 = arith.cmpi sgt, %127, %120 : i64
              %140:2 = scf.if %139 -> (i64, i64) {
                %141 = cannir.extract_layout %arg0 : !cannir.tensor<<gm, bf16>, <<(-1, -1, 2, -1, 1024)>, <(-1, -1, 1024, 2048, 1)>>, <<(-1, -1, 2, -1, 1024)>, <(-1, -1, 1024, 2048, 1)>>, identity>
                %142 = cannir.extract_shape %141 : <<(-1, -1, 2, -1, 1024)>, <(-1, -1, 1024, 2048, 1)>> -> <(-1, -1, 2, -1, 1024)>
                %143 = cannir.extract_dim %142[3] : !cannir.shape<(-1, -1, 2, -1, 1024)>
                %144 = arith.subi %143, %90 : i64
                %145 = ascscalar.bitcast_memref %alloc_17 byte_offset = %c0_i64 : memref<65536xbf16, 2> -> memref<?xbf16, 2>
                %146 = cannir.extract_physical_layout %arg0 : !cannir.tensor<<gm, bf16>, <<(-1, -1, 2, -1, 1024)>, <(-1, -1, 1024, 2048, 1)>>, <<(-1, -1, 2, -1, 1024)>, <(-1, -1, 1024, 2048, 1)>>, identity>
                %147 = cannir.extract_stride %146 : <<(-1, -1, 2, -1, 1024)>, <(-1, -1, 1024, 2048, 1)>> -> <(-1, -1, 1024, 2048, 1)>
                %148 = cannir.extract_dim %147[0] : !cannir.stride<(-1, -1, 1024, 2048, 1)>
                %149 = arith.muli %38, %148 : i64
                %150 = arith.index_cast %149 : i64 to index
                %151 = cannir.extract_dim %147[1] : !cannir.stride<(-1, -1, 1024, 2048, 1)>
                %152 = arith.muli %89, %151 : i64
                %153 = arith.index_cast %152 : i64 to index
                %154 = arith.addi %150, %153 : index
                %155 = arith.muli %90, %c2048_i64 : i64
                %156 = arith.index_cast %155 : i64 to index
                %157 = arith.addi %154, %156 : index
                %158 = arith.divsi %144, %c64_i64 : i64
                %159 = arith.minsi %158, %c0_i64 : i64
                %160 = arith.muli %159, %c64_i64 : i64
                %161 = arith.subi %144, %160 : i64
                %162 = arith.minsi %161, %c64_i64 : i64
                %163 = arith.cmpi sge, %158, %c0_i64 : i64
                %164 = arith.select %163, %162, %c0_i64 : i64
                asccube.get_buf mte2, %c0_i64, 0 {cannir.buffer.generated_sync}
                %165 = arith.cmpi ugt, %164, %c0_i64 : i64
                scf.if %165 {
                  %178 = arith.trunci %164 : i64 to i32
                  asccube.copy_gm2l1 %145[%c0], %alloc[%157], %178, %c1024_i32, %c4096_i64, %c1_i32, %c0_i64, %c0_i64, %c64_i32 l2_cache_ctl %c1_i32 {nd2nz = true} : memref<?xbf16, 2>, memref<1xbf16>
                }
                asccube.release_buf mte2, %c0_i64, 0 {cannir.buffer.generated_sync}
                %166 = ascscalar.bitcast_memref %alloc_15 byte_offset = %c0_i64 : memref<65536xbf16, 2> -> memref<?xbf16, 2>
                %167 = arith.addi %154, %c1024 : index
                %168 = arith.addi %167, %156 : index
                asccube.get_buf mte2, %c1_i64, 0 {cannir.buffer.generated_sync}
                scf.if %165 {
                  %178 = arith.trunci %164 : i64 to i32
                  asccube.copy_gm2l1 %166[%c0], %alloc[%168], %178, %c1024_i32, %c4096_i64, %c1_i32, %c0_i64, %c0_i64, %c64_i32 l2_cache_ctl %c1_i32 {nd2nz = true} : memref<?xbf16, 2>, memref<1xbf16>
                }
                asccube.release_buf mte2, %c1_i64, 0 {cannir.buffer.generated_sync}
                %169 = arith.addi %138, %c255_i64 : i64
                %170 = arith.remsi %169, %c256_i64 : i64
                %171 = arith.cmpi ne, %170, %c0_i64 : i64
                %172 = arith.cmpi slt, %170, %c0_i64 : i64
                %173 = arith.andi %171, %172 : i1
                %174 = arith.divsi %169, %c256_i64 : i64
                %175 = arith.extui %173 : i1 to i64
                %176 = arith.subi %174, %175 : i64
                %177:2 = scf.for %arg24 = %c0_i64 to %176 step %c1_i64 iter_args(%arg25 = %arg22, %arg26 = %arg23) -> (i64, i64)  : i64 {
                  %178 = arith.muli %arg24, %c256_i64 : i64
                  %179:2 = scf.for %arg27 = %c0_i64 to %c4_i64 step %c1_i64 iter_args(%arg28 = %arg25, %arg29 = %arg26) -> (i64, i64)  : i64 {
                    %180 = arith.muli %arg27, %c64_i64 overflow<nsw> : i64
                    %181 = arith.addi %178, %180 : i64
                    %182 = arith.cmpi slt, %181, %138 : i64
                    %183:2 = scf.if %182 -> (i64, i64) {
                      %184 = arith.cmpi eq, %arg28, %c0_i64 : i64
                      %185 = ascscalar.memref_address %alloc_19 : memref<4096xf32, 5>
                      %186 = ascscalar.memref_address %alloc_18 : memref<4096xf32, 5>
                      %187 = arith.select %184, %185, %186 : i64
                      %188 = arith.select %184, %c8_i64, %c9_i64 : i64
                      %189 = arith.addi %arg28, %c1_i64 : i64
                      %190 = arith.remui %189, %c2_i64 : i64
                      %191 = cannir.extract_layout %arg1 : !cannir.tensor<<gm, bf16>, <<(-1, -1, 2, -1, 1024)>, <(-1, -1, 1024, 2048, 1)>>, <<(-1, -1, 2, -1, 1024)>, <(-1, -1, 1024, 2048, 1)>>, identity>
                      %192 = cannir.extract_shape %191 : <<(-1, -1, 2, -1, 1024)>, <(-1, -1, 1024, 2048, 1)>> -> <(-1, -1, 2, -1, 1024)>
                      %193 = cannir.extract_dim %192[3] : !cannir.shape<(-1, -1, 2, -1, 1024)>
                      %194 = arith.subi %193, %181 : i64
                      %195 = ascscalar.bitcast_memref %alloc_16 byte_offset = %c0_i64 : memref<65536xbf16, 2> -> memref<?xbf16, 2>
                      %196 = cannir.extract_physical_layout %arg1 : !cannir.tensor<<gm, bf16>, <<(-1, -1, 2, -1, 1024)>, <(-1, -1, 1024, 2048, 1)>>, <<(-1, -1, 2, -1, 1024)>, <(-1, -1, 1024, 2048, 1)>>, identity>
                      %197 = cannir.extract_stride %196 : <<(-1, -1, 2, -1, 1024)>, <(-1, -1, 1024, 2048, 1)>> -> <(-1, -1, 1024, 2048, 1)>
                      %198 = cannir.extract_dim %197[0] : !cannir.stride<(-1, -1, 1024, 2048, 1)>
                      %199 = arith.muli %38, %198 : i64
                      %200 = arith.index_cast %199 : i64 to index
                      %201 = cannir.extract_dim %197[1] : !cannir.stride<(-1, -1, 1024, 2048, 1)>
                      %202 = arith.muli %40, %201 : i64
                      %203 = arith.index_cast %202 : i64 to index
                      %204 = arith.addi %200, %203 : index
                      %205 = arith.muli %181, %c2048_i64 : i64
                      %206 = arith.index_cast %205 : i64 to index
                      %207 = arith.addi %204, %206 : index
                      %208 = arith.divsi %194, %c64_i64 : i64
                      %209 = arith.minsi %208, %c0_i64 : i64
                      %210 = arith.muli %209, %c64_i64 : i64
                      %211 = arith.subi %194, %210 : i64
                      %212 = arith.minsi %211, %c64_i64 : i64
                      %213 = arith.cmpi sge, %208, %c0_i64 : i64
                      %214 = arith.select %213, %212, %c0_i64 : i64
                      asccube.get_buf mte2, %c2_i64, 0 {cannir.buffer.generated_sync}
                      %215 = arith.cmpi ugt, %214, %c0_i64 : i64
                      scf.if %215 {
                        %241 = arith.trunci %214 : i64 to i32
                        asccube.copy_gm2l1 %195[%c0], %alloc_2[%207], %241, %c1024_i32, %c4096_i64, %c1_i32, %c0_i64, %c0_i64, %c64_i32 l2_cache_ctl %c1_i32 {nd2nz = true} : memref<?xbf16, 2>, memref<1xbf16>
                      }
                      asccube.release_buf mte2, %c2_i64, 0 {cannir.buffer.generated_sync}
                      %216 = ascscalar.bitcast_memref %alloc_10 byte_offset = %c0_i64 : memref<16384xbf16, 3> -> memref<?xbf16, 3>
                      asccube.get_buf mte1, %c4_i64, 0 {cannir.buffer.generated_sync}
                      asccube.get_buf mte1, %c0_i64, 0 {cannir.buffer.generated_sync}
                      asccube.copy_l12l0a %216[%c0], %145[%c0], %c0_i32, %c0_i32, %c4_i32, %c16_i32, %c4_i32, %c4_i32 : memref<?xbf16, 3>, memref<?xbf16, 2>
                      asccube.release_buf mte1, %c4_i64, 0 {cannir.buffer.generated_sync}
                      %217 = ascscalar.bitcast_memref %alloc_11 byte_offset = %c0_i64 : memref<16384xbf16, 4> -> memref<?xbf16, 4>
                      asccube.get_buf mte1, %c6_i64, 0 {cannir.buffer.generated_sync}
                      asccube.get_buf mte1, %c2_i64, 0 {cannir.buffer.generated_sync}
                      asccube.copy_l12l0b %217[%c0], %195[%c0], %c0_i32, %c0_i32, %c4_i32, %c16_i32, %c4_i32, %c4_i32 : memref<?xbf16, 4>, memref<?xbf16, 2>
                      asccube.release_buf mte1, %c6_i64, 0 {cannir.buffer.generated_sync}
                      %218 = ascscalar.bind_local_ptr %187 : memref<1xf32, 5>
                      asccube.get_buf m, %188, 0 {cannir.buffer.generated_sync}
                      asccube.get_buf m, %c4_i64, 0 {cannir.buffer.generated_sync}
                      asccube.get_buf m, %c6_i64, 0 {cannir.buffer.generated_sync}
                      asccube.mmad %218[%c0], %216, %217, %c64_i32, %c256_i32, %c64_i32, %c0_i32 {c_matrix_init_val = true} : memref<1xf32, 5>, memref<?xbf16, 3>, memref<?xbf16, 4>
                      asccube.release_buf m, %c6_i64, 0 {cannir.buffer.generated_sync}
                      asccube.release_buf m, %c4_i64, 0 {cannir.buffer.generated_sync}
                      %219 = ascscalar.bitcast_memref %alloc_12 byte_offset = %c0_i64 : memref<16384xbf16, 3> -> memref<?xbf16, 3>
                      asccube.get_buf mte1, %c5_i64, 0 {cannir.buffer.generated_sync}
                      asccube.copy_l12l0a %219[%c0], %145[%c0], %c0_i32, %c16_i32, %c4_i32, %c16_i32, %c4_i32, %c4_i32 : memref<?xbf16, 3>, memref<?xbf16, 2>
                      asccube.release_buf mte1, %c5_i64, 0 {cannir.buffer.generated_sync}
                      %220 = ascscalar.bitcast_memref %alloc_13 byte_offset = %c0_i64 : memref<16384xbf16, 4> -> memref<?xbf16, 4>
                      asccube.get_buf mte1, %c7_i64, 0 {cannir.buffer.generated_sync}
                      asccube.copy_l12l0b %220[%c0], %195[%c0], %c0_i32, %c16_i32, %c4_i32, %c16_i32, %c4_i32, %c4_i32 : memref<?xbf16, 4>, memref<?xbf16, 2>
                      asccube.release_buf mte1, %c7_i64, 0 {cannir.buffer.generated_sync}
                      asccube.get_buf m, %c5_i64, 0 {cannir.buffer.generated_sync}
                      asccube.get_buf m, %c7_i64, 0 {cannir.buffer.generated_sync}
                      asccube.mmad %218[%c0], %219, %220, %c64_i32, %c256_i32, %c64_i32, %c0_i32 {c_matrix_init_val = false} : memref<1xf32, 5>, memref<?xbf16, 3>, memref<?xbf16, 4>
                      asccube.release_buf m, %c7_i64, 0 {cannir.buffer.generated_sync}
                      asccube.release_buf m, %c5_i64, 0 {cannir.buffer.generated_sync}
                      asccube.get_buf mte1, %c4_i64, 0 {cannir.buffer.generated_sync}
                      asccube.copy_l12l0a %216[%c0], %145[%c0], %c0_i32, %c32_i32, %c4_i32, %c16_i32, %c4_i32, %c4_i32 : memref<?xbf16, 3>, memref<?xbf16, 2>
                      asccube.release_buf mte1, %c4_i64, 0 {cannir.buffer.generated_sync}
                      asccube.get_buf mte1, %c6_i64, 0 {cannir.buffer.generated_sync}
                      asccube.copy_l12l0b %217[%c0], %195[%c0], %c0_i32, %c32_i32, %c4_i32, %c16_i32, %c4_i32, %c4_i32 : memref<?xbf16, 4>, memref<?xbf16, 2>
                      asccube.release_buf mte1, %c6_i64, 0 {cannir.buffer.generated_sync}
                      asccube.get_buf m, %c4_i64, 0 {cannir.buffer.generated_sync}
                      asccube.get_buf m, %c6_i64, 0 {cannir.buffer.generated_sync}
                      asccube.mmad %218[%c0], %216, %217, %c64_i32, %c256_i32, %c64_i32, %c0_i32 {c_matrix_init_val = false} : memref<1xf32, 5>, memref<?xbf16, 3>, memref<?xbf16, 4>
                      asccube.release_buf m, %c6_i64, 0 {cannir.buffer.generated_sync}
                      asccube.release_buf m, %c4_i64, 0 {cannir.buffer.generated_sync}
                      asccube.get_buf mte1, %c5_i64, 0 {cannir.buffer.generated_sync}
                      asccube.copy_l12l0a %219[%c0], %145[%c0], %c0_i32, %c48_i32, %c4_i32, %c16_i32, %c4_i32, %c4_i32 : memref<?xbf16, 3>, memref<?xbf16, 2>
                      asccube.release_buf mte1, %c0_i64, 0 {cannir.buffer.generated_sync}
                      asccube.release_buf mte1, %c5_i64, 0 {cannir.buffer.generated_sync}
                      asccube.get_buf mte1, %c7_i64, 0 {cannir.buffer.generated_sync}
                      asccube.copy_l12l0b %220[%c0], %195[%c0], %c0_i32, %c48_i32, %c4_i32, %c16_i32, %c4_i32, %c4_i32 : memref<?xbf16, 4>, memref<?xbf16, 2>
                      asccube.release_buf mte1, %c2_i64, 0 {cannir.buffer.generated_sync}
                      asccube.release_buf mte1, %c7_i64, 0 {cannir.buffer.generated_sync}
                      asccube.get_buf m, %c5_i64, 0 {cannir.buffer.generated_sync}
                      asccube.get_buf m, %c7_i64, 0 {cannir.buffer.generated_sync}
                      asccube.mmad %218[%c0], %219, %220, %c64_i32, %c256_i32, %c64_i32, %c0_i32 {c_matrix_init_val = false} : memref<1xf32, 5>, memref<?xbf16, 3>, memref<?xbf16, 4>
                      asccube.release_buf m, %c7_i64, 0 {cannir.buffer.generated_sync}
                      asccube.release_buf m, %c5_i64, 0 {cannir.buffer.generated_sync}
                      %221 = ascscalar.bitcast_memref %alloc_14 byte_offset = %c0_i64 : memref<65536xbf16, 2> -> memref<?xbf16, 2>
                      %222 = arith.addi %204, %c1024 : index
                      %223 = arith.addi %222, %206 : index
                      asccube.get_buf mte2, %c3_i64, 0 {cannir.buffer.generated_sync}
                      scf.if %215 {
                        %241 = arith.trunci %214 : i64 to i32
                        asccube.copy_gm2l1 %221[%c0], %alloc_2[%223], %241, %c1024_i32, %c4096_i64, %c1_i32, %c0_i64, %c0_i64, %c64_i32 l2_cache_ctl %c1_i32 {nd2nz = true} : memref<?xbf16, 2>, memref<1xbf16>
                      }
                      asccube.release_buf mte2, %c3_i64, 0 {cannir.buffer.generated_sync}
                      asccube.get_buf mte1, %c4_i64, 0 {cannir.buffer.generated_sync}
                      asccube.get_buf mte1, %c1_i64, 0 {cannir.buffer.generated_sync}
                      asccube.copy_l12l0a %216[%c0], %166[%c0], %c0_i32, %c0_i32, %c4_i32, %c16_i32, %c4_i32, %c4_i32 : memref<?xbf16, 3>, memref<?xbf16, 2>
                      asccube.release_buf mte1, %c4_i64, 0 {cannir.buffer.generated_sync}
                      asccube.get_buf mte1, %c6_i64, 0 {cannir.buffer.generated_sync}
                      asccube.get_buf mte1, %c3_i64, 0 {cannir.buffer.generated_sync}
                      asccube.copy_l12l0b %217[%c0], %221[%c0], %c0_i32, %c0_i32, %c4_i32, %c16_i32, %c4_i32, %c4_i32 : memref<?xbf16, 4>, memref<?xbf16, 2>
                      asccube.release_buf mte1, %c6_i64, 0 {cannir.buffer.generated_sync}
                      asccube.get_buf m, %c4_i64, 0 {cannir.buffer.generated_sync}
                      asccube.get_buf m, %c6_i64, 0 {cannir.buffer.generated_sync}
                      asccube.mmad %218[%c0], %216, %217, %c64_i32, %c256_i32, %c64_i32, %c0_i32 {c_matrix_init_val = false} : memref<1xf32, 5>, memref<?xbf16, 3>, memref<?xbf16, 4>
                      asccube.release_buf m, %c6_i64, 0 {cannir.buffer.generated_sync}
                      asccube.release_buf m, %c4_i64, 0 {cannir.buffer.generated_sync}
                      asccube.get_buf mte1, %c5_i64, 0 {cannir.buffer.generated_sync}
                      asccube.copy_l12l0a %219[%c0], %166[%c0], %c0_i32, %c16_i32, %c4_i32, %c16_i32, %c4_i32, %c4_i32 : memref<?xbf16, 3>, memref<?xbf16, 2>
                      asccube.release_buf mte1, %c5_i64, 0 {cannir.buffer.generated_sync}
                      asccube.get_buf mte1, %c7_i64, 0 {cannir.buffer.generated_sync}
                      asccube.copy_l12l0b %220[%c0], %221[%c0], %c0_i32, %c16_i32, %c4_i32, %c16_i32, %c4_i32, %c4_i32 : memref<?xbf16, 4>, memref<?xbf16, 2>
                      asccube.release_buf mte1, %c7_i64, 0 {cannir.buffer.generated_sync}
                      asccube.get_buf m, %c5_i64, 0 {cannir.buffer.generated_sync}
                      asccube.get_buf m, %c7_i64, 0 {cannir.buffer.generated_sync}
                      asccube.mmad %218[%c0], %219, %220, %c64_i32, %c256_i32, %c64_i32, %c0_i32 {c_matrix_init_val = false} : memref<1xf32, 5>, memref<?xbf16, 3>, memref<?xbf16, 4>
                      asccube.release_buf m, %c7_i64, 0 {cannir.buffer.generated_sync}
                      asccube.release_buf m, %c5_i64, 0 {cannir.buffer.generated_sync}
                      asccube.get_buf mte1, %c4_i64, 0 {cannir.buffer.generated_sync}
                      asccube.copy_l12l0a %216[%c0], %166[%c0], %c0_i32, %c32_i32, %c4_i32, %c16_i32, %c4_i32, %c4_i32 : memref<?xbf16, 3>, memref<?xbf16, 2>
                      asccube.release_buf mte1, %c4_i64, 0 {cannir.buffer.generated_sync}
                      asccube.get_buf mte1, %c6_i64, 0 {cannir.buffer.generated_sync}
                      asccube.copy_l12l0b %217[%c0], %221[%c0], %c0_i32, %c32_i32, %c4_i32, %c16_i32, %c4_i32, %c4_i32 : memref<?xbf16, 4>, memref<?xbf16, 2>
                      asccube.release_buf mte1, %c6_i64, 0 {cannir.buffer.generated_sync}
                      asccube.get_buf m, %c4_i64, 0 {cannir.buffer.generated_sync}
                      asccube.get_buf m, %c6_i64, 0 {cannir.buffer.generated_sync}
                      asccube.mmad %218[%c0], %216, %217, %c64_i32, %c256_i32, %c64_i32, %c0_i32 {c_matrix_init_val = false} : memref<1xf32, 5>, memref<?xbf16, 3>, memref<?xbf16, 4>
                      asccube.release_buf m, %c6_i64, 0 {cannir.buffer.generated_sync}
                      asccube.release_buf m, %c4_i64, 0 {cannir.buffer.generated_sync}
                      asccube.get_buf mte1, %c5_i64, 0 {cannir.buffer.generated_sync}
                      asccube.copy_l12l0a %219[%c0], %166[%c0], %c0_i32, %c48_i32, %c4_i32, %c16_i32, %c4_i32, %c4_i32 : memref<?xbf16, 3>, memref<?xbf16, 2>
                      asccube.release_buf mte1, %c1_i64, 0 {cannir.buffer.generated_sync}
                      asccube.release_buf mte1, %c5_i64, 0 {cannir.buffer.generated_sync}
                      asccube.get_buf mte1, %c7_i64, 0 {cannir.buffer.generated_sync}
                      asccube.copy_l12l0b %220[%c0], %221[%c0], %c0_i32, %c48_i32, %c4_i32, %c16_i32, %c4_i32, %c4_i32 : memref<?xbf16, 4>, memref<?xbf16, 2>
                      asccube.release_buf mte1, %c3_i64, 0 {cannir.buffer.generated_sync}
                      asccube.release_buf mte1, %c7_i64, 0 {cannir.buffer.generated_sync}
                      asccube.get_buf m, %c5_i64, 0 {cannir.buffer.generated_sync}
                      asccube.get_buf m, %c7_i64, 0 {cannir.buffer.generated_sync}
                      asccube.mmad %218[%c0], %219, %220, %c64_i32, %c256_i32, %c64_i32, %c0_i32 {c_matrix_init_val = false} : memref<1xf32, 5>, memref<?xbf16, 3>, memref<?xbf16, 4>
                      asccube.release_buf m, %c7_i64, 0 {cannir.buffer.generated_sync}
                      asccube.release_buf m, %c5_i64, 0 {cannir.buffer.generated_sync}
                      asccube.release_buf m, %188, 0 {cannir.buffer.generated_sync}
                      %224 = arith.cmpi eq, %arg29, %c1_i64 : i64
                      %225 = ascscalar.memref_address %alloc_8 : memref<2048xf32, 1>
                      %226 = ascscalar.memref_address %alloc_9 : memref<2048xf32, 1>
                      %227 = arith.select %224, %225, %226 : i64
                      %228 = arith.cmpi eq, %arg29, %c0_i64 : i64
                      %229 = ascscalar.memref_address %alloc_7 : memref<2048xf32, 1>
                      %230 = arith.select %228, %229, %227 : i64
                      %231 = arith.select %224, %c11_i64, %c12_i64 : i64
                      %232 = arith.select %228, %c10_i64, %231 : i64
                      %233 = arith.addi %arg29, %c1_i64 : i64
                      %234 = arith.remui %233, %c3_i64 : i64
                      %235 = ascscalar.bind_local_ptr %230 : memref<1xf32, 1>
                      %236 = arith.cmpi eq, %232, %c11_i64 : i64
                      %237 = arith.extui %236 : i1 to i64
                      %238 = arith.cmpi eq, %232, %c12_i64 : i64
                      %239 = arith.select %238, %c2_i64, %237 : i64
                      asccube.sync_intra_wait fixpipe, %239 {cannir.sync.banks = 3 : i32}
                      %240 = arith.addi %239, %c16_i64 overflow<nsw> : i64
                      asccube.sync_intra_wait fixpipe, %240 {cannir.sync.banks = 3 : i32}
                      asccube.get_buf fixpipe, %188, 0 {cannir.buffer.generated_sync}
                      asccube.copy_l0c2ub %235[%c0], %218[%c0], %c64_i32, %c64_i32, %c64_i32, %c64_i32, %c0_i32 sub_block_id %c0_i32 {dual_dst_ctl = 1 : i32, nz2nd_en = true, relu_en = false} : memref<1xf32, 1>, memref<1xf32, 5>
                      asccube.release_buf fixpipe, %188, 0 {cannir.buffer.generated_sync}
                      asccube.sync_intra_arrive fixpipe, %239 {cannir.sync.banks = 3 : i32}
                      asccube.sync_intra_arrive fixpipe, %240 {cannir.sync.banks = 3 : i32}
                      scf.yield %190, %234 : i64, i64
                    } else {
                      scf.yield %arg28, %arg29 : i64, i64
                    }
                    scf.yield %183#0, %183#1 : i64, i64
                  }
                  scf.yield %179#0, %179#1 : i64, i64
                }
                scf.yield %177#0, %177#1 : i64, i64
              } else {
                scf.yield %arg22, %arg23 : i64, i64
              }
              scf.yield %140#0, %140#1 : i64, i64
            } else {
              scf.yield %arg22, %arg23 : i64, i64
            }
            scf.yield %87#0, %87#1 : i64, i64
          }
          scf.yield %79#0, %79#1 : i64, i64
        }
        scf.yield %75#0, %75#1 : i64, i64
      }
      scf.yield %29#0, %29#1 : i64, i64
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
