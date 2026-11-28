module {
  func.func @launch_67d386cc8398e59a6f8174883612c61f(%arg0: !cannir.tensor<<gm, bf16>, <<(-1, 128)>, <(128, 1)>>, <<(-1, 128)>, <(128, 1)>>, identity>, %arg1: !cannir.tensor<<gm, bf16>, <<(-1, 256, 128)>, <(32768, 128, 1)>>, <<(-1, 256, 128)>, <(32768, 128, 1)>>, identity>, %arg2: !cannir.tensor<<gm, i32>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg3: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg4: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg5: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg6: !cannir.tensor<<gm, i32>, <<(-1, 2056)>, <(2056, 1)>>, <<(-1, 2056)>, <(2056, 1)>>, identity>, %arg7: !cannir.tensor<<gm, ui16>, <<(1024, -1)>, <(-1, 1)>>, <<(1024, -1)>, <(-1, 1)>>, identity>) attributes {cannbotdsl.contract_digest = "67d386cc8398e59a6f8174883612c61f008400e3eae0043fafcdf206835f8d5e", cannbotdsl.public_slot_count = 8 : i64, cannbotdsl.slot_dtype_bits = [16, 16, 32, 32, 32, 32, 32, 16], cannbotdsl.slot_dtype_codes = [4, 4, 0, 0, 0, 0, 0, 1], cannbotdsl.slot_dtype_lanes = [1, 1, 1, 1, 1, 1, 1, 1], cannbotdsl.slot_flags = [1, 1, 1, 1, 1, 1, 1, 1], cannbotdsl.slot_kinds = [1, 1, 1, 1, 1, 1, 1, 1], cannbotdsl.slot_type_digests = ["7e56b57b7bfb1e88e1aa7cb42b74b26633c63315f25bac94c373b8e10ede1927", "b202eb386fa03b59011640e1b8f8ddd2bb55909dcb73539fcd2d30966a4637cc", "7b0d85c6f597bacfa0d5663410329f4998659f20836564b42e969adc6ede89f6", "3238f8ac021397ee0f6145770b0f9dc942e839b14b8109790620d3b1580bb851", "cd903f583f25a26a2e692a39f7f38690a7b4e37be745d4f2fe027950c556870b", "c4b997d1ce0522bff7b28b9aebf67dfec422cfe688d8d629d39374bd3af36169", "69106053573815fd972f7d2c583083b58b714050876204030301453f99628fab", "1ae7ce176c7e1d96c3e48509c8c690481777266f4f3be1f7c0069927c32534bf"], cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.host} {
    %c32_i64 = arith.constant 32 : i64
    cannir.kernel_launch @kernel_QSAIndexerKernel___call___0(%c32_i64) [%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7] {cannbotdsl.profile = {indices = array<i64: 0, 1, 2, 3, 4, 5, 6, 7, 7>, name = "QSAIndexerKernel___call__", op_type = "QSAIndexerKernel___call__", roles = array<i64: 0, 0, 0, 0, 0, 0, 1, 0, 1>}} : !cannir.tensor<<gm, bf16>, <<(-1, 128)>, <(128, 1)>>, <<(-1, 128)>, <(128, 1)>>, identity>, !cannir.tensor<<gm, bf16>, <<(-1, 256, 128)>, <(32768, 128, 1)>>, <<(-1, 256, 128)>, <(32768, 128, 1)>>, identity>, !cannir.tensor<<gm, i32>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, !cannir.tensor<<gm, i32>, <<(-1, 2056)>, <(2056, 1)>>, <<(-1, 2056)>, <(2056, 1)>>, identity>, !cannir.tensor<<gm, ui16>, <<(1024, -1)>, <(-1, 1)>>, <<(1024, -1)>, <(-1, 1)>>, identity>
    return
  }
  func.func @kernel_QSAIndexerKernel___call___0_aiv(%arg0: !cannir.tensor<<gm, bf16>, <<(-1, 128)>, <(128, 1)>>, <<(-1, 128)>, <(128, 1)>>, identity>, %arg1: !cannir.tensor<<gm, bf16>, <<(-1, 256, 128)>, <(32768, 128, 1)>>, <<(-1, 256, 128)>, <(32768, 128, 1)>>, identity>, %arg2: !cannir.tensor<<gm, i32>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg3: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg4: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg5: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg6: !cannir.tensor<<gm, i32>, <<(-1, 2056)>, <(2056, 1)>>, <<(-1, 2056)>, <(2056, 1)>>, identity>, %arg7: !cannir.tensor<<gm, ui16>, <<(1024, -1)>, <(-1, 1)>>, <<(1024, -1)>, <(-1, 1)>>, identity>) attributes {cannir.buffer.auto_sync, cannir.buffer.sync_analyzed, cannir.buffer.sync_verified, cannir.buffer.tensor_state, cannir.core = "aiv", cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.device, cannir.explicit_ring_buffer_cursors, cannir.kernel_group = "kernel_QSAIndexerKernel___call___0", cannir.resource.ids_prepared} {
    %c8224_i32 = arith.constant 8224 : i32
    %c17368 = arith.constant 17368 : index
    %c17360 = arith.constant 17360 : index
    %c4608 = arith.constant 4608 : index
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %c2_i64 = arith.constant 2 : i64
    %c3_i64 = arith.constant 3 : i64
    %c4_i64 = arith.constant 4 : i64
    %c5_i64 = arith.constant 5 : i64
    %c6_i64 = arith.constant 6 : i64
    %c8_i64 = arith.constant 8 : i64
    %c16_i64 = arith.constant 16 : i64
    %c31_i64 = arith.constant 31 : i64
    %c32_i64 = arith.constant 32 : i64
    %c65792_i64 = arith.constant 65792 : i64
    %c4095_i64 = arith.constant 4095 : i64
    %c255_i64 = arith.constant 255 : i64
    %c2 = arith.constant 2 : index
    %c448_i32 = arith.constant 448 : i32
    %c384_i32 = arith.constant 384 : i32
    %c320_i32 = arith.constant 320 : i32
    %c192_i32 = arith.constant 192 : i32
    %c621824_i64 = arith.constant 621824 : i64
    %c556032_i64 = arith.constant 556032 : i64
    %c8_i32 = arith.constant 8 : i32
    %c512_i64 = arith.constant 512 : i64
    %c512_i32 = arith.constant 512 : i32
    %c4608_i64 = arith.constant 4608 : i64
    %c8192_i64 = arith.constant 8192 : i64
    %c818688_i64 = arith.constant 818688 : i64
    %c687616_i64 = arith.constant 687616 : i64
    %c-1_i64 = arith.constant -1 : i64
    %c0 = arith.constant 0 : index
    %c1024_i64 = arith.constant 1024 : i64
    %c256_i32 = arith.constant 256 : i32
    %c16_i32 = arith.constant 16 : i32
    %c128_i32 = arith.constant 128 : i32
    %c65536_i64 = arith.constant 65536 : i64
    %c1474560_i64 = arith.constant 1474560 : i64
    %c950272_i64 = arith.constant 950272 : i64
    %c128_i64 = arith.constant 128 : i64
    %c4096_i64 = arith.constant 4096 : i64
    %c256_i64 = arith.constant 256 : i64
    %c576_i64 = arith.constant 576 : i64
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %alloc = memref.alloc() {cannir.arg_index = 3 : i64} : memref<1xi32>
    %alloc_0 = memref.alloc() {cannir.arg_index = 4 : i64} : memref<1xi32>
    %alloc_1 = memref.alloc() {cannir.arg_index = 5 : i64} : memref<1xi32>
    %alloc_2 = memref.alloc() {cannir.arg_index = 6 : i64} : memref<1xi32>
    %alloc_3 = memref.alloc() {cannir.arg_index = 7 : i64} : memref<1xui16>
    %alloc_4 = memref.alloc() {cannir.byte_offset = 0 : i64, cannir.onchip_space_root} : memref<262144xui8, 1>
    ascvec.sync_intra_arrive v, %c1_i64
    ascvec.sync_intra_arrive v, %c0_i64
    %0 = ascscalar.get_subblock_id
    %1 = ascscalar.bitcast_memref %alloc_4 byte_offset = %c0_i64 : memref<262144xui8, 1> -> memref<?xui32, 1>
    ascvec.vf(%1 : memref<?xui32, 1>) {
    ^bb0(%arg8: memref<?xui32, 1>):
      %c1_i32_5 = arith.constant 1 : i32
      %c32_i32 = arith.constant 32 : i32
      %c4608_6 = arith.constant 4608 : index
      %c0_i32_7 = arith.constant 0 : i32
      %c64_i32_8 = arith.constant 64 : i32
      %mask, %remaining_out = ascvec.update_mask %c64_i32_8 : !ascvec.mask
      %5 = ascvec.duplicate_scalar %c0_i32_7, %mask : i32 -> <ui32>
      scf.for %arg9 = %c0_i32_7 to %c32_i32 step %c1_i32_5  : i32 {
        %6 = arith.muli %arg9, %c64_i32_8 overflow<nsw> : i32
        %7 = arith.index_cast %6 : i32 to index
        %8 = arith.addi %7, %c4608_6 : index
        ascvec.storealign %arg8[%8], %5, %mask : memref<?xui32, 1>, index, <ui32>
      }
    } {cannir.vf_mode = "simd"}
    %2 = memref.load %alloc_1[%c0] : memref<1xi32>
    %3 = arith.extsi %2 : i32 to i64
    %4:6 = scf.for %arg8 = %c0_i64 to %3 step %c1_i64 iter_args(%arg9 = %c0_i64, %arg10 = %c0_i64, %arg11 = %c0_i64, %arg12 = %c0_i64, %arg13 = %c0_i64, %arg14 = %c0_i64) -> (i64, i64, i64, i64, i64, i64)  : i64 {
      %5 = arith.muli %arg8, %c576_i64 : i64
      %6 = arith.addi %5, %c16_i64 : i64
      %7 = ascscalar.get_block_idx
      %8 = arith.muli %7, %c16_i64 : i64
      %9 = arith.addi %6, %8 : i64
      %10 = arith.index_cast %9 : i64 to index
      %11 = memref.load %alloc_1[%10] : memref<1xi32>
      %12 = arith.extsi %11 : i32 to i64
      %13 = arith.addi %9, %c1_i64 : i64
      %14 = arith.index_cast %13 : i64 to index
      %15 = memref.load %alloc_1[%14] : memref<1xi32>
      %16 = arith.extsi %15 : i32 to i64
      %17 = arith.addi %9, %c3_i64 : i64
      %18 = arith.index_cast %17 : i64 to index
      %19 = memref.load %alloc_1[%18] : memref<1xi32>
      %20 = arith.extsi %19 : i32 to i64
      %21 = arith.addi %9, %c4_i64 : i64
      %22 = arith.index_cast %21 : i64 to index
      %23 = memref.load %alloc_1[%22] : memref<1xi32>
      %24 = arith.extsi %23 : i32 to i64
      %25 = arith.addi %20, %c1_i64 : i64
      %26:6 = scf.for %arg15 = %12 to %25 step %c1_i64 iter_args(%arg16 = %arg9, %arg17 = %arg10, %arg18 = %arg11, %arg19 = %arg12, %arg20 = %arg13, %arg21 = %arg14) -> (i64, i64, i64, i64, i64, i64)  : i64 {
        %27 = arith.index_cast %arg15 : i64 to index
        %28 = memref.load %alloc[%27] : memref<1xi32>
        %29 = arith.extsi %28 : i32 to i64
        %30 = arith.addi %arg15, %c1_i64 : i64
        %31 = arith.index_cast %30 : i64 to index
        %32 = memref.load %alloc[%31] : memref<1xi32>
        %33 = arith.extsi %32 : i32 to i64
        %34 = arith.cmpi eq, %arg15, %12 : i64
        %35 = arith.select %34, %16, %c0_i64 : i64
        %36 = arith.subi %33, %29 : i64
        %37 = arith.addi %36, %c31_i64 : i64
        %38 = arith.remsi %37, %c32_i64 : i64
        %39 = arith.cmpi ne, %38, %c0_i64 : i64
        %40 = arith.cmpi slt, %38, %c0_i64 : i64
        %41 = arith.andi %39, %40 : i1
        %42 = arith.divsi %37, %c32_i64 : i64
        %43 = arith.extui %41 : i1 to i64
        %44 = arith.subi %42, %43 : i64
        %45 = arith.cmpi eq, %arg15, %20 : i64
        %46 = arith.select %45, %24, %44 : i64
        %47:6 = scf.for %arg22 = %35 to %46 step %c1_i64 iter_args(%arg23 = %arg16, %arg24 = %arg17, %arg25 = %arg18, %arg26 = %arg19, %arg27 = %arg20, %arg28 = %arg21) -> (i64, i64, i64, i64, i64, i64)  : i64 {
          %48 = arith.muli %arg22, %c32_i64 : i64
          %49 = arith.addi %29, %48 : i64
          %50 = arith.subi %33, %49 : i64
          %51 = arith.minsi %50, %c32_i64 : i64
          %52 = scf.for %arg29 = %c0_i64 to %51 step %c1_i64 iter_args(%arg30 = %c0_i64) -> (i64)  : i64 {
            %74 = arith.addi %49, %arg29 : i64
            %75 = arith.index_cast %74 : i64 to index
            %76 = memref.load %alloc_0[%75] : memref<1xi32>
            %77 = arith.extsi %76 : i32 to i64
            %78 = arith.addi %77, %c1_i64 : i64
            %79 = arith.remsi %78, %c4_i64 : i64
            %80 = arith.cmpi ne, %79, %c0_i64 : i64
            %81 = arith.cmpi slt, %79, %c0_i64 : i64
            %82 = arith.andi %80, %81 : i1
            %83 = arith.divsi %78, %c4_i64 : i64
            %84 = arith.extui %82 : i1 to i64
            %85 = arith.subi %83, %84 : i64
            %86 = arith.maxsi %arg30, %85 : i64
            scf.yield %86 : i64
          }
          %53 = arith.muli %7, %c2_i64 : i64
          %54 = arith.addi %53, %0 : i64
          %55 = arith.muli %54, %c16_i64 : i64
          %56 = arith.addi %52, %c255_i64 : i64
          %57 = arith.remsi %56, %c256_i64 : i64
          %58 = arith.cmpi ne, %57, %c0_i64 : i64
          %59 = arith.cmpi slt, %57, %c0_i64 : i64
          %60 = arith.andi %58, %59 : i1
          %61 = arith.divsi %56, %c256_i64 : i64
          %62 = arith.extui %60 : i1 to i64
          %63 = arith.subi %61, %62 : i64
          %64 = arith.addi %52, %c4095_i64 : i64
          %65 = arith.remsi %64, %c4096_i64 : i64
          %66 = arith.cmpi ne, %65, %c0_i64 : i64
          %67 = arith.cmpi slt, %65, %c0_i64 : i64
          %68 = arith.andi %66, %67 : i1
          %69 = arith.divsi %64, %c4096_i64 : i64
          %70 = arith.extui %68 : i1 to i64
          %71 = arith.subi %69, %70 : i64
          %72 = arith.cmpi sgt, %63, %c0_i64 : i64
          %73:6 = scf.if %72 -> (i64, i64, i64, i64, i64, i64) {
            %74:3 = scf.for %arg29 = %c0_i64 to %63 step %c1_i64 iter_args(%arg30 = %arg23, %arg31 = %arg24, %arg32 = %arg28) -> (i64, i64, i64)  : i64 {
              %78 = arith.muli %arg29, %c256_i64 : i64
              %79 = arith.cmpi eq, %arg30, %c0_i64 : i64
              %80 = arith.addi %arg30, %c1_i64 : i64
              %81 = arith.remui %80, %c2_i64 : i64
              %82 = arith.addi %arg31, %c1_i64 : i64
              %83 = arith.remui %82, %c2_i64 : i64
              %84 = arith.addi %83, %c1_i64 overflow<nsw> : i64
              %85 = arith.remui %84, %c2_i64 : i64
              %86 = arith.select %79, %c950272_i64, %c1474560_i64 : i64
              %87 = arith.cmpi ne, %arg30, %c0_i64 : i64
              %88 = arith.extui %87 : i1 to i64
              %89 = arith.cmpi eq, %88, %c1_i64 : i64
              %90 = arith.extui %89 : i1 to i64
              %91 = ascscalar.memref_address %alloc_4 : memref<262144xui8, 1>
              %92 = arith.divui %86, %c8_i64 : i64
              %93 = arith.addi %91, %92 : i64
              %94 = ascscalar.bind_local_ptr %93 : memref<16384xf32, 1>
              %95 = arith.cmpi eq, %arg32, %c0_i64 : i64
              %96 = arith.select %95, %c0_i64, %c65536_i64 : i64
              %97 = arith.divui %96, %c8_i64 : i64
              %98 = arith.addi %91, %97 : i64
              %99 = ascscalar.bind_local_ptr %98 : memref<4096xui16, 1>
              %100 = arith.select %95, %c5_i64, %c6_i64 : i64
              %101 = arith.addi %arg32, %c1_i64 : i64
              %102 = arith.remui %101, %c2_i64 : i64
              ascvec.sync_intra_wait v, %90 {cannir.sync.banks = 3 : i32}
              ascvec.get_buf v, %100, 0 {cannir.buffer.generated_sync}
              ascvec.vf(%94, %99 : memref<16384xf32, 1>, memref<4096xui16, 1>) {
              ^bb0(%arg33: memref<16384xf32, 1>, %arg34: memref<4096xui16, 1>):
                %c192_i32_5 = arith.constant 192 : i32
                %c1024_i32 = arith.constant 1024 : i32
                %c16_i32_6 = arith.constant 16 : i32
                %c0_i32_7 = arith.constant 0 : i32
                %c3_i32 = arith.constant 3 : i32
                %c2_i32 = arith.constant 2 : i32
                %c1_i32_8 = arith.constant 1 : i32
                %c256_i32_9 = arith.constant 256 : i32
                %c4_i32 = arith.constant 4 : i32
                %c-32768_i16 = arith.constant -32768 : i16
                %cst = arith.constant 0.000000e+00 : f32
                %c128_i32_10 = arith.constant 128 : i32
                %c64_i32_11 = arith.constant 64 : i32
                %mask, %remaining_out = ascvec.update_mask %c64_i32_11 : !ascvec.mask
                %mask_12, %remaining_out_13 = ascvec.update_mask %c128_i32_10 elem_bits 16 : !ascvec.mask
                %120 = ascvec.duplicate_scalar %cst, %mask : f32 -> <f32>
                %121 = ascvec.duplicate_scalar %c-32768_i16, %mask_12 : i16 -> <ui16>
                scf.for %arg35 = %c0_i32_7 to %c16_i32_6 step %c1_i32_8  : i32 {
                  %122 = arith.muli %arg35, %c4_i32 overflow<nsw> : i32
                  %123 = arith.muli %arg35, %c1024_i32 overflow<nsw> : i32
                  %124 = arith.index_cast %123 : i32 to index
                  %125 = ascvec.loadalign %arg33[%124] : memref<16384xf32, 1>, index -> <f32>
                  %126 = arith.addi %123, %c64_i32_11 overflow<nsw> : i32
                  %127 = arith.index_cast %126 : i32 to index
                  %128 = ascvec.loadalign %arg33[%127] : memref<16384xf32, 1>, index -> <f32>
                  %129 = arith.addi %122, %c1_i32_8 overflow<nsw> : i32
                  %130 = arith.muli %129, %c256_i32_9 overflow<nsw> : i32
                  %131 = arith.index_cast %130 : i32 to index
                  %132 = ascvec.loadalign %arg33[%131] : memref<16384xf32, 1>, index -> <f32>
                  %133 = arith.addi %130, %c64_i32_11 overflow<nsw> : i32
                  %134 = arith.index_cast %133 : i32 to index
                  %135 = ascvec.loadalign %arg33[%134] : memref<16384xf32, 1>, index -> <f32>
                  %136 = arith.addi %122, %c2_i32 overflow<nsw> : i32
                  %137 = arith.muli %136, %c256_i32_9 overflow<nsw> : i32
                  %138 = arith.index_cast %137 : i32 to index
                  %139 = ascvec.loadalign %arg33[%138] : memref<16384xf32, 1>, index -> <f32>
                  %140 = arith.addi %137, %c64_i32_11 overflow<nsw> : i32
                  %141 = arith.index_cast %140 : i32 to index
                  %142 = ascvec.loadalign %arg33[%141] : memref<16384xf32, 1>, index -> <f32>
                  %143 = arith.addi %122, %c3_i32 overflow<nsw> : i32
                  %144 = arith.muli %143, %c256_i32_9 overflow<nsw> : i32
                  %145 = arith.index_cast %144 : i32 to index
                  %146 = ascvec.loadalign %arg33[%145] : memref<16384xf32, 1>, index -> <f32>
                  %147 = arith.addi %144, %c64_i32_11 overflow<nsw> : i32
                  %148 = arith.index_cast %147 : i32 to index
                  %149 = ascvec.loadalign %arg33[%148] : memref<16384xf32, 1>, index -> <f32>
                  %150 = ascvec.max %125, %120, %mask : <f32>
                  %151 = ascvec.max %128, %120, %mask : <f32>
                  %152 = ascvec.max %132, %120, %mask : <f32>
                  %153 = ascvec.max %135, %120, %mask : <f32>
                  %154 = ascvec.max %139, %120, %mask : <f32>
                  %155 = ascvec.max %142, %120, %mask : <f32>
                  %156 = ascvec.max %146, %120, %mask : <f32>
                  %157 = ascvec.max %149, %120, %mask : <f32>
                  %158 = ascvec.add %150, %152, %mask : <f32>
                  %159 = ascvec.add %151, %153, %mask : <f32>
                  %160 = ascvec.add %158, %154, %mask : <f32>
                  %161 = ascvec.add %159, %155, %mask : <f32>
                  %162 = ascvec.add %160, %156, %mask : <f32>
                  %163 = ascvec.add %161, %157, %mask : <f32>
                  %164 = ascvec.cast %162, %mask : <f32> -> <bf16>
                  %165 = ascvec.cast %163, %mask : <f32> -> <bf16>
                  %166 = ascvec.reinterpret %164 : <bf16> -> <ui16>
                  %167 = ascvec.reinterpret %165 : <bf16> -> <ui16>
                  %result0, %result1 = ascvec.deintlv %166, %167 : <ui16>, <ui16> -> <ui16>, <ui16>
                  %168 = ascvec.xor %result0, %121, %mask_12 : <ui16>, <ui16> -> <ui16>
                  %169 = arith.muli %arg35, %c256_i32_9 overflow<nsw> : i32
                  %170 = arith.index_cast %169 : i32 to index
                  ascvec.storealign %arg34[%170], %168, %mask_12 : memref<4096xui16, 1>, index, <ui16>
                  %171 = arith.addi %123, %c128_i32_10 overflow<nsw> : i32
                  %172 = arith.index_cast %171 : i32 to index
                  %173 = ascvec.loadalign %arg33[%172] : memref<16384xf32, 1>, index -> <f32>
                  %174 = arith.addi %123, %c192_i32_5 overflow<nsw> : i32
                  %175 = arith.index_cast %174 : i32 to index
                  %176 = ascvec.loadalign %arg33[%175] : memref<16384xf32, 1>, index -> <f32>
                  %177 = arith.addi %130, %c128_i32_10 overflow<nsw> : i32
                  %178 = arith.index_cast %177 : i32 to index
                  %179 = ascvec.loadalign %arg33[%178] : memref<16384xf32, 1>, index -> <f32>
                  %180 = arith.addi %130, %c192_i32_5 overflow<nsw> : i32
                  %181 = arith.index_cast %180 : i32 to index
                  %182 = ascvec.loadalign %arg33[%181] : memref<16384xf32, 1>, index -> <f32>
                  %183 = arith.addi %137, %c128_i32_10 overflow<nsw> : i32
                  %184 = arith.index_cast %183 : i32 to index
                  %185 = ascvec.loadalign %arg33[%184] : memref<16384xf32, 1>, index -> <f32>
                  %186 = arith.addi %137, %c192_i32_5 overflow<nsw> : i32
                  %187 = arith.index_cast %186 : i32 to index
                  %188 = ascvec.loadalign %arg33[%187] : memref<16384xf32, 1>, index -> <f32>
                  %189 = arith.addi %144, %c128_i32_10 overflow<nsw> : i32
                  %190 = arith.index_cast %189 : i32 to index
                  %191 = ascvec.loadalign %arg33[%190] : memref<16384xf32, 1>, index -> <f32>
                  %192 = arith.addi %144, %c192_i32_5 overflow<nsw> : i32
                  %193 = arith.index_cast %192 : i32 to index
                  %194 = ascvec.loadalign %arg33[%193] : memref<16384xf32, 1>, index -> <f32>
                  %195 = ascvec.max %173, %120, %mask : <f32>
                  %196 = ascvec.max %176, %120, %mask : <f32>
                  %197 = ascvec.max %179, %120, %mask : <f32>
                  %198 = ascvec.max %182, %120, %mask : <f32>
                  %199 = ascvec.max %185, %120, %mask : <f32>
                  %200 = ascvec.max %188, %120, %mask : <f32>
                  %201 = ascvec.max %191, %120, %mask : <f32>
                  %202 = ascvec.max %194, %120, %mask : <f32>
                  %203 = ascvec.add %195, %197, %mask : <f32>
                  %204 = ascvec.add %196, %198, %mask : <f32>
                  %205 = ascvec.add %203, %199, %mask : <f32>
                  %206 = ascvec.add %204, %200, %mask : <f32>
                  %207 = ascvec.add %205, %201, %mask : <f32>
                  %208 = ascvec.add %206, %202, %mask : <f32>
                  %209 = ascvec.cast %207, %mask : <f32> -> <bf16>
                  %210 = ascvec.cast %208, %mask : <f32> -> <bf16>
                  %211 = ascvec.reinterpret %209 : <bf16> -> <ui16>
                  %212 = ascvec.reinterpret %210 : <bf16> -> <ui16>
                  %result0_14, %result1_15 = ascvec.deintlv %211, %212 : <ui16>, <ui16> -> <ui16>, <ui16>
                  %213 = ascvec.xor %result0_14, %121, %mask_12 : <ui16>, <ui16> -> <ui16>
                  %214 = arith.addi %169, %c128_i32_10 overflow<nsw> : i32
                  %215 = arith.index_cast %214 : i32 to index
                  ascvec.storealign %arg34[%215], %213, %mask_12 : memref<4096xui16, 1>, index, <ui16>
                }
              } {cannir.vf_mode = "simd"}
              ascvec.release_buf v, %100, 0 {cannir.buffer.generated_sync}
              ascvec.sync_intra_arrive v, %90 {cannir.sync.banks = 3 : i32}
              %103 = cannir.extract_physical_layout %arg7 : !cannir.tensor<<gm, ui16>, <<(1024, -1)>, <(-1, 1)>>, <<(1024, -1)>, <(-1, 1)>>, identity>
              %104 = cannir.extract_stride %103 : <<(1024, -1)>, <(-1, 1)>> -> <(-1, 1)>
              %105 = cannir.extract_dim %104[0] : !cannir.stride<(-1, 1)>
              %106 = arith.muli %55, %105 : i64
              %107 = arith.addi %106, %78 : i64
              %108 = arith.muli %107, %c16_i64 : i64
              %109 = ascscalar.memref_address %alloc_3 : memref<1xui16>
              %110 = arith.divui %108, %c8_i64 : i64
              %111 = arith.addi %109, %110 : i64
              %112 = ascscalar.bind_gm_ptr %111 : memref<1xui16>
              %113 = ascscalar.bind_local_ptr %98 : memref<1xui16, 1>
              %114 = cannir.extract_layout %arg7 : !cannir.tensor<<gm, ui16>, <<(1024, -1)>, <(-1, 1)>>, <<(1024, -1)>, <(-1, 1)>>, identity>
              %115 = cannir.extract_stride %114 : <<(1024, -1)>, <(-1, 1)>> -> <(-1, 1)>
              %116 = cannir.extract_dim %115[0] : !cannir.stride<(-1, 1)>
              %117 = arith.index_cast %116 : i64 to index
              %118 = arith.muli %117, %c2 : index
              %119 = arith.index_cast %118 : index to i64
              ascvec.get_buf mte3, %100, 0 {cannir.buffer.generated_sync}
              ascvec.copy_ub2gm %112[%c0], %113[%c0], %c16_i32, %c512_i32, %119, %c512_i32 l2_cache_ctl %c0_i32 : memref<1xui16>, memref<1xui16, 1>
              ascvec.release_buf mte3, %100, 0 {cannir.buffer.generated_sync}
              scf.yield %81, %85, %102 : i64, i64, i64
            }
            %75 = arith.muli %71, %c16_i64 : i64
            %76 = scf.for %arg29 = %63 to %75 step %c1_i64 iter_args(%arg30 = %74#2) -> (i64)  : i64 {
              %78 = arith.cmpi eq, %arg30, %c0_i64 : i64
              %79 = arith.select %78, %c0_i64, %c65536_i64 : i64
              %80 = ascscalar.memref_address %alloc_4 : memref<262144xui8, 1>
              %81 = arith.divui %79, %c8_i64 : i64
              %82 = arith.addi %80, %81 : i64
              %83 = ascscalar.bind_local_ptr %82 : memref<4096xui16, 1>
              %84 = arith.select %78, %c5_i64, %c6_i64 : i64
              %85 = arith.addi %arg30, %c1_i64 : i64
              %86 = arith.remui %85, %c2_i64 : i64
              ascvec.get_buf v, %84, 0 {cannir.buffer.generated_sync}
              ascvec.vf(%83 : memref<4096xui16, 1>) {
              ^bb0(%arg31: memref<4096xui16, 1>):
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
                %c0_5 = arith.constant 0 : index
                %c0_i16 = arith.constant 0 : i16
                %c128_i32_6 = arith.constant 128 : i32
                %c0_7 = arith.constant 0 : index
                %c1 = arith.constant 1 : index
                scf.for %arg32 = %c0_7 to %c1 step %c1 {
                  %mask, %remaining_out = ascvec.update_mask %c128_i32_6 elem_bits 16 : !ascvec.mask
                  %105 = ascvec.duplicate_scalar %c0_i16, %mask : i16 -> <ui16>
                  ascvec.storealign %arg31[%c0_5], %105, %mask : memref<4096xui16, 1>, index, <ui16>
                  ascvec.storealign %arg31[%c128], %105, %mask : memref<4096xui16, 1>, index, <ui16>
                  ascvec.storealign %arg31[%c256], %105, %mask : memref<4096xui16, 1>, index, <ui16>
                  ascvec.storealign %arg31[%c384], %105, %mask : memref<4096xui16, 1>, index, <ui16>
                  ascvec.storealign %arg31[%c512], %105, %mask : memref<4096xui16, 1>, index, <ui16>
                  ascvec.storealign %arg31[%c640], %105, %mask : memref<4096xui16, 1>, index, <ui16>
                  ascvec.storealign %arg31[%c768], %105, %mask : memref<4096xui16, 1>, index, <ui16>
                  ascvec.storealign %arg31[%c896], %105, %mask : memref<4096xui16, 1>, index, <ui16>
                  ascvec.storealign %arg31[%c1024], %105, %mask : memref<4096xui16, 1>, index, <ui16>
                  ascvec.storealign %arg31[%c1152], %105, %mask : memref<4096xui16, 1>, index, <ui16>
                  ascvec.storealign %arg31[%c1280], %105, %mask : memref<4096xui16, 1>, index, <ui16>
                  ascvec.storealign %arg31[%c1408], %105, %mask : memref<4096xui16, 1>, index, <ui16>
                  ascvec.storealign %arg31[%c1536], %105, %mask : memref<4096xui16, 1>, index, <ui16>
                  ascvec.storealign %arg31[%c1664], %105, %mask : memref<4096xui16, 1>, index, <ui16>
                  ascvec.storealign %arg31[%c1792], %105, %mask : memref<4096xui16, 1>, index, <ui16>
                  ascvec.storealign %arg31[%c1920], %105, %mask : memref<4096xui16, 1>, index, <ui16>
                  ascvec.storealign %arg31[%c2048], %105, %mask : memref<4096xui16, 1>, index, <ui16>
                  ascvec.storealign %arg31[%c2176], %105, %mask : memref<4096xui16, 1>, index, <ui16>
                  ascvec.storealign %arg31[%c2304], %105, %mask : memref<4096xui16, 1>, index, <ui16>
                  ascvec.storealign %arg31[%c2432], %105, %mask : memref<4096xui16, 1>, index, <ui16>
                  ascvec.storealign %arg31[%c2560], %105, %mask : memref<4096xui16, 1>, index, <ui16>
                  ascvec.storealign %arg31[%c2688], %105, %mask : memref<4096xui16, 1>, index, <ui16>
                  ascvec.storealign %arg31[%c2816], %105, %mask : memref<4096xui16, 1>, index, <ui16>
                  ascvec.storealign %arg31[%c2944], %105, %mask : memref<4096xui16, 1>, index, <ui16>
                  ascvec.storealign %arg31[%c3072], %105, %mask : memref<4096xui16, 1>, index, <ui16>
                  ascvec.storealign %arg31[%c3200], %105, %mask : memref<4096xui16, 1>, index, <ui16>
                  ascvec.storealign %arg31[%c3328], %105, %mask : memref<4096xui16, 1>, index, <ui16>
                  ascvec.storealign %arg31[%c3456], %105, %mask : memref<4096xui16, 1>, index, <ui16>
                  ascvec.storealign %arg31[%c3584], %105, %mask : memref<4096xui16, 1>, index, <ui16>
                  ascvec.storealign %arg31[%c3712], %105, %mask : memref<4096xui16, 1>, index, <ui16>
                  ascvec.storealign %arg31[%c3840], %105, %mask : memref<4096xui16, 1>, index, <ui16>
                  ascvec.storealign %arg31[%c3968], %105, %mask : memref<4096xui16, 1>, index, <ui16>
                }
              } {cannir.vf_mode = "simd"}
              ascvec.release_buf v, %84, 0 {cannir.buffer.generated_sync}
              %87 = arith.muli %arg29, %c256_i64 : i64
              %88 = cannir.extract_physical_layout %arg7 : !cannir.tensor<<gm, ui16>, <<(1024, -1)>, <(-1, 1)>>, <<(1024, -1)>, <(-1, 1)>>, identity>
              %89 = cannir.extract_stride %88 : <<(1024, -1)>, <(-1, 1)>> -> <(-1, 1)>
              %90 = cannir.extract_dim %89[0] : !cannir.stride<(-1, 1)>
              %91 = arith.muli %55, %90 : i64
              %92 = arith.addi %91, %87 : i64
              %93 = arith.muli %92, %c16_i64 : i64
              %94 = ascscalar.memref_address %alloc_3 : memref<1xui16>
              %95 = arith.divui %93, %c8_i64 : i64
              %96 = arith.addi %94, %95 : i64
              %97 = ascscalar.bind_gm_ptr %96 : memref<1xui16>
              %98 = ascscalar.bind_local_ptr %82 : memref<1xui16, 1>
              %99 = cannir.extract_layout %arg7 : !cannir.tensor<<gm, ui16>, <<(1024, -1)>, <(-1, 1)>>, <<(1024, -1)>, <(-1, 1)>>, identity>
              %100 = cannir.extract_stride %99 : <<(1024, -1)>, <(-1, 1)>> -> <(-1, 1)>
              %101 = cannir.extract_dim %100[0] : !cannir.stride<(-1, 1)>
              %102 = arith.index_cast %101 : i64 to index
              %103 = arith.muli %102, %c2 : index
              %104 = arith.index_cast %103 : index to i64
              ascvec.get_buf mte3, %84, 0 {cannir.buffer.generated_sync}
              ascvec.copy_ub2gm %97[%c0], %98[%c0], %c16_i32, %c512_i32, %104, %c512_i32 l2_cache_ctl %c0_i32 : memref<1xui16>, memref<1xui16, 1>
              ascvec.release_buf mte3, %84, 0 {cannir.buffer.generated_sync}
              scf.yield %86 : i64
            }
            ascvec.sync_all
            %77:3 = scf.for %arg29 = %c0_i64 to %c16_i64 step %c2_i64 iter_args(%arg30 = %arg25, %arg31 = %arg26, %arg32 = %arg27) -> (i64, i64, i64)  : i64 {
              %78 = arith.muli %0, %c16_i64 : i64
              %79 = arith.addi %78, %arg29 : i64
              %80 = arith.cmpi slt, %79, %51 : i64
              %81 = scf.if %80 -> (i64) {
                %92 = arith.addi %49, %79 : i64
                %93 = arith.index_cast %92 : i64 to index
                %94 = memref.load %alloc_0[%93] : memref<1xi32>
                %95 = arith.extsi %94 : i32 to i64
                scf.yield %95 : i64
              } else {
                scf.yield %c-1_i64 : i64
              }
              %82 = arith.addi %79, %c1_i64 : i64
              %83 = arith.cmpi slt, %82, %51 : i64
              %84 = scf.if %83 -> (i64) {
                %92 = arith.addi %49, %79 : i64
                %93 = arith.addi %92, %c1_i64 : i64
                %94 = arith.index_cast %93 : i64 to index
                %95 = memref.load %alloc_0[%94] : memref<1xi32>
                %96 = arith.extsi %95 : i32 to i64
                scf.yield %96 : i64
              } else {
                scf.yield %c-1_i64 : i64
              }
              %85 = arith.trunci %81 : i64 to i32
              %86 = arith.trunci %84 : i64 to i32
              %87 = ascscalar.bitcast_memref %alloc_4 byte_offset = %c0_i64 : memref<262144xui8, 1> -> memref<?xi32, 1>
              ascvec.vf(%85, %87, %87, %87, %86 : i32, memref<?xi32, 1>, memref<?xi32, 1>, memref<?xi32, 1>, i32) {
              ^bb0(%arg33: i32, %arg34: memref<?xi32, 1>, %arg35: memref<?xi32, 1>, %arg36: memref<?xi32, 1>, %arg37: i32):
                %c17352 = arith.constant 17352 : index
                %c17336 = arith.constant 17336 : index
                %c17320 = arith.constant 17320 : index
                %c17344 = arith.constant 17344 : index
                %c17328 = arith.constant 17328 : index
                %c0_i32_5 = arith.constant 0 : i32
                %c17312 = arith.constant 17312 : index
                %c1_i32_6 = arith.constant 1 : i32
                %c0_7 = arith.constant 0 : index
                %c1 = arith.constant 1 : index
                scf.for %arg38 = %c0_7 to %c1 step %c1 {
                  %mask, %remaining_out = ascvec.update_mask %c1_i32_6 : !ascvec.mask
                  %92 = ascvec.duplicate_scalar %arg33, %mask : i32 -> <i32>
                  %93 = ascvec.add_scalar %92, %c1_i32_6, %mask : <i32>, i32
                  %94 = ascvec.shr %93, 2, %mask : <i32> -> <i32>
                  ascvec.storealign_1st %arg34[%c17312], %94 : memref<?xi32, 1>, index, <i32>
                  %95 = ascvec.duplicate_scalar %c0_i32_5, %mask : i32 -> <i32>
                  ascvec.storealign_1st %arg35[%c17328], %95 : memref<?xi32, 1>, index, <i32>
                  ascvec.storealign_1st %arg36[%c17344], %93 : memref<?xi32, 1>, index, <i32>
                  %96 = ascvec.duplicate_scalar %arg37, %mask : i32 -> <i32>
                  %97 = ascvec.add_scalar %96, %c1_i32_6, %mask : <i32>, i32
                  %98 = ascvec.shr %97, 2, %mask : <i32> -> <i32>
                  ascvec.storealign_1st %arg34[%c17320], %98 : memref<?xi32, 1>, index, <i32>
                  ascvec.storealign_1st %arg35[%c17336], %95 : memref<?xi32, 1>, index, <i32>
                  ascvec.storealign_1st %arg36[%c17352], %97 : memref<?xi32, 1>, index, <i32>
                }
              } {cannir.vf_mode = "simd"}
              %88 = arith.cmpi sgt, %71, %c0_i64 : i64
              %89 = scf.if %88 -> (i64) {
                %92 = arith.addi %55, %arg29 : i64
                %93 = arith.subi %c1024_i64, %92 : i64
                %94 = arith.cmpi eq, %arg31, %c0_i64 : i64
                %95 = arith.select %94, %c687616_i64, %c818688_i64 : i64
                %96 = arith.select %94, %c3_i64, %c4_i64 : i64
                %97 = arith.addi %arg31, %c1_i64 : i64
                %98 = arith.remui %97, %c2_i64 : i64
                %99 = arith.divsi %93, %c2_i64 : i64
                %100 = arith.minsi %99, %c0_i64 : i64
                %101 = arith.muli %100, %c2_i64 : i64
                %102 = arith.subi %93, %101 : i64
                %103 = arith.minsi %102, %c2_i64 : i64
                %104 = arith.cmpi sge, %99, %c0_i64 : i64
                %105 = arith.select %104, %103, %c0_i64 : i64
                %106 = cannir.extract_physical_layout %arg7 : !cannir.tensor<<gm, ui16>, <<(1024, -1)>, <(-1, 1)>>, <<(1024, -1)>, <(-1, 1)>>, identity>
                %107 = cannir.extract_stride %106 : <<(1024, -1)>, <(-1, 1)>> -> <(-1, 1)>
                %108 = cannir.extract_dim %107[0] : !cannir.stride<(-1, 1)>
                %109 = arith.muli %108, %c2_i64 : i64
                %110 = cannir.extract_layout %arg7 : !cannir.tensor<<gm, ui16>, <<(1024, -1)>, <(-1, 1)>>, <<(1024, -1)>, <(-1, 1)>>, identity>
                %111 = cannir.extract_shape %110 : <<(1024, -1)>, <(-1, 1)>> -> <(1024, -1)>
                %112 = cannir.extract_dim %111[1] : !cannir.shape<(1024, -1)>
                %113 = arith.divsi %112, %c4096_i64 : i64
                %114 = arith.minsi %113, %c0_i64 : i64
                %115 = arith.muli %114, %c4096_i64 : i64
                %116 = arith.subi %112, %115 : i64
                %117 = arith.minsi %116, %c4096_i64 : i64
                %118 = arith.cmpi sge, %113, %c0_i64 : i64
                %119 = arith.select %118, %117, %c0_i64 : i64
                %120 = arith.muli %92, %108 : i64
                %121 = arith.muli %120, %c2_i64 : i64
                %122 = arith.divui %95, %c8_i64 : i64
                %123 = arith.muli %119, %c2_i64 : i64
                %124 = arith.cmpi eq, %123, %c0_i64 : i64
                %125 = arith.select %124, %c0_i64, %105 : i64
                %126 = arith.select %124, %c2_i64, %123 : i64
                %127 = ascscalar.bitcast_memref %alloc_4 byte_offset = %c0_i64 : memref<262144xui8, 1> -> memref<?xui16, 1>
                ascvec.get_buf mte2, %96, 0 {cannir.buffer.generated_sync}
                "ascvec.copy_gm2ub_plan"(%127, %alloc_3, %c0_i32, %c0_i32, %122, %121, %125, %126, %109, %c8192_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = ui16, ub_base = 0 : i64, ub_capacity = 262144 : i64}> : (memref<?xui16, 1>, memref<1xui16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                ascvec.release_buf mte2, %96, 0 {cannir.buffer.generated_sync}
                scf.yield %98 : i64
              } else {
                scf.yield %arg31 : i64
              }
              %90:2 = scf.for %arg33 = %c0_i64 to %71 step %c1_i64 iter_args(%arg34 = %89, %arg35 = %arg32) -> (i64, i64)  : i64 {
                %92 = arith.remsi %arg33, %c2_i64 : i64
                %93 = arith.cmpi ne, %92, %c0_i64 : i64
                %94 = arith.cmpi slt, %92, %c0_i64 : i64
                %95 = arith.andi %93, %94 : i1
                %96 = arith.select %95, %c2_i64, %c0_i64 : i64
                %97 = arith.addi %92, %96 overflow<nsw> : i64
                %98 = arith.addi %arg33, %c1_i64 : i64
                %99 = arith.remsi %98, %c2_i64 : i64
                %100 = arith.cmpi ne, %99, %c0_i64 : i64
                %101 = arith.cmpi slt, %99, %c0_i64 : i64
                %102 = arith.andi %100, %101 : i1
                %103 = arith.select %102, %c2_i64, %c0_i64 : i64
                %104 = arith.addi %99, %103 overflow<nsw> : i64
                %105 = arith.cmpi eq, %arg35, %c0_i64 : i64
                %106 = arith.select %105, %c687616_i64, %c818688_i64 : i64
                %107 = ascscalar.memref_address %alloc_4 : memref<262144xui8, 1>
                %108 = arith.divui %106, %c8_i64 : i64
                %109 = arith.addi %107, %108 : i64
                %110 = ascscalar.bind_local_ptr %109 : memref<8192xui16, 1>
                %111 = arith.select %105, %c3_i64, %c4_i64 : i64
                %112 = arith.addi %arg35, %c1_i64 : i64
                %113 = arith.remui %112, %c2_i64 : i64
                %114 = arith.cmpi slt, %98, %71 : i64
                %115 = scf.if %114 -> (i64) {
                  %120 = arith.addi %55, %arg29 : i64
                  %121 = arith.muli %98, %c4096_i64 : i64
                  %122 = cannir.extract_layout %arg7 : !cannir.tensor<<gm, ui16>, <<(1024, -1)>, <(-1, 1)>>, <<(1024, -1)>, <(-1, 1)>>, identity>
                  %123 = cannir.extract_shape %122 : <<(1024, -1)>, <(-1, 1)>> -> <(1024, -1)>
                  %124 = cannir.extract_dim %123[1] : !cannir.shape<(1024, -1)>
                  %125 = arith.subi %c1024_i64, %120 : i64
                  %126 = arith.subi %124, %121 : i64
                  %127 = arith.cmpi eq, %arg34, %c0_i64 : i64
                  %128 = arith.select %127, %c687616_i64, %c818688_i64 : i64
                  %129 = arith.select %127, %c3_i64, %c4_i64 : i64
                  %130 = arith.addi %arg34, %c1_i64 : i64
                  %131 = arith.remui %130, %c2_i64 : i64
                  %132 = arith.divsi %125, %c2_i64 : i64
                  %133 = arith.minsi %132, %c0_i64 : i64
                  %134 = arith.muli %133, %c2_i64 : i64
                  %135 = arith.subi %125, %134 : i64
                  %136 = arith.minsi %135, %c2_i64 : i64
                  %137 = arith.cmpi sge, %132, %c0_i64 : i64
                  %138 = arith.select %137, %136, %c0_i64 : i64
                  %139 = cannir.extract_physical_layout %arg7 : !cannir.tensor<<gm, ui16>, <<(1024, -1)>, <(-1, 1)>>, <<(1024, -1)>, <(-1, 1)>>, identity>
                  %140 = cannir.extract_stride %139 : <<(1024, -1)>, <(-1, 1)>> -> <(-1, 1)>
                  %141 = cannir.extract_dim %140[0] : !cannir.stride<(-1, 1)>
                  %142 = arith.muli %141, %c2_i64 : i64
                  %143 = arith.divsi %126, %c4096_i64 : i64
                  %144 = arith.minsi %143, %c0_i64 : i64
                  %145 = arith.muli %144, %c4096_i64 : i64
                  %146 = arith.subi %126, %145 : i64
                  %147 = arith.minsi %146, %c4096_i64 : i64
                  %148 = arith.cmpi sge, %143, %c0_i64 : i64
                  %149 = arith.select %148, %147, %c0_i64 : i64
                  %150 = arith.muli %120, %141 : i64
                  %151 = arith.muli %150, %c2_i64 : i64
                  %152 = arith.muli %98, %c8192_i64 : i64
                  %153 = arith.addi %151, %152 : i64
                  %154 = arith.divui %128, %c8_i64 : i64
                  %155 = arith.muli %149, %c2_i64 : i64
                  %156 = arith.cmpi eq, %155, %c0_i64 : i64
                  %157 = arith.select %156, %c0_i64, %138 : i64
                  %158 = arith.select %156, %c2_i64, %155 : i64
                  %159 = ascscalar.bitcast_memref %alloc_4 byte_offset = %c0_i64 : memref<262144xui8, 1> -> memref<?xui16, 1>
                  ascvec.get_buf mte2, %129, 0 {cannir.buffer.generated_sync}
                  "ascvec.copy_gm2ub_plan"(%159, %alloc_3, %c0_i32, %c0_i32, %154, %153, %157, %158, %142, %c8192_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = ui16, ub_base = 0 : i64, ub_capacity = 262144 : i64}> : (memref<?xui16, 1>, memref<1xui16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                  ascvec.release_buf mte2, %129, 0 {cannir.buffer.generated_sync}
                  scf.yield %131 : i64
                } else {
                  scf.yield %arg34 : i64
                }
                %116 = arith.muli %arg33, %c4096_i64 : i64
                %117 = arith.cmpi eq, %arg33, %c0_i64 : i64
                %118 = scf.if %117 -> (i64) {
                  ascvec.get_buf v, %111, 0 {cannir.buffer.generated_sync}
                  %120 = ascscalar.bitcast_memref %alloc_4 byte_offset = %c0_i64 : memref<262144xui8, 1> -> memref<?xui16, 1>
                  ascvec.vf(%87, %110, %120, %120 : memref<?xi32, 1>, memref<8192xui16, 1>, memref<?xui16, 1>, memref<?xui16, 1>) {
                  ^bb0(%arg36: memref<?xi32, 1>, %arg37: memref<8192xui16, 1>, %arg38: memref<?xui16, 1>, %arg39: memref<?xui16, 1>):
                    %c33216 = arith.constant 33216 : index
                    %c33088 = arith.constant 33088 : index
                    %c5248_i32 = arith.constant 5248 : i32
                    %c4224_i32 = arith.constant 4224 : i32
                    %c17320 = arith.constant 17320 : index
                    %c32960 = arith.constant 32960 : index
                    %c640_i32 = arith.constant 640 : i32
                    %c5120_i32 = arith.constant 5120 : i32
                    %c32832 = arith.constant 32832 : index
                    %c1_i32_5 = arith.constant 1 : i32
                    %c16_i32_6 = arith.constant 16 : i32
                    %c512_i32_7 = arith.constant 512 : i32
                    %c4096_i32 = arith.constant 4096 : i32
                    %c0_i32_8 = arith.constant 0 : i32
                    %c64_i32_9 = arith.constant 64 : i32
                    %c17312 = arith.constant 17312 : index
                    %c0_i16 = arith.constant 0 : i16
                    %c128_i32_10 = arith.constant 128 : i32
                    %c256_i32_11 = arith.constant 256 : i32
                    %mask, %remaining_out = ascvec.update_mask %c256_i32_11 elem_bits 8 : !ascvec.mask
                    %mask_12, %remaining_out_13 = ascvec.update_mask %c128_i32_10 elem_bits 16 : !ascvec.mask
                    %121 = ascvec.duplicate_scalar %c0_i16, %mask_12 : i16 -> <ui16>
                    %122 = ascvec.loadalign_brc %arg36[%c17312] : memref<?xi32, 1>, index -> <i32>
                    %mask_14, %remaining_out_15 = ascvec.update_mask %c64_i32_9 : !ascvec.mask
                    %123 = ascvec.duplicate_scalar %c0_i32_8, %mask_14 : i32 -> <i32>
                    %124 = ascvec.sub %122, %123, %mask_14 : <i32>
                    %125 = ascvec.max_scalar %124, %c0_i32_8, %mask_14 : <i32>, i32
                    %126 = ascvec.min_scalar %125, %c4096_i32, %mask_14 : <i32>, i32
                    %127 = ascvec.reinterpret %126 : <i32> -> <ui16>
                    %128 = ascvec.duplicate %127, %mask_12 : <ui16> -> <ui16>
                    %129:2 = scf.for %arg40 = %c0_i32_8 to %c16_i32_6 step %c1_i32_5 iter_args(%arg41 = %121, %arg42 = %121) -> (!ascvec.vreg<ui16>, !ascvec.vreg<ui16>)  : i32 {
                      %138 = arith.muli %arg40, %c256_i32_11 overflow<nsw> : i32
                      %139 = arith.index_cast %138 : i32 to index
                      %140 = ascvec.loadalign %arg37[%139] : memref<8192xui16, 1>, index -> <ui16>
                      %141 = arith.addi %138, %c128_i32_10 overflow<nsw> : i32
                      %142 = arith.index_cast %141 : i32 to index
                      %143 = ascvec.loadalign %arg37[%142] : memref<8192xui16, 1>, index -> <ui16>
                      %144 = arith.trunci %138 : i32 to i16
                      %145 = ascvec.arange %144 : i16 -> <ui16>
                      %146 = arith.trunci %141 : i32 to i16
                      %147 = ascvec.arange %146 : i16 -> <ui16>
                      %148 = ascvec.cmp_lt %145, %128, %mask_12 : <ui16>, <ui16> -> !ascvec.mask
                      %149 = ascvec.select %140, %121, %148 : <ui16>
                      %150 = ascvec.cmp_lt %147, %128, %mask_12 : <ui16>, <ui16> -> !ascvec.mask
                      %151 = ascvec.select %143, %121, %150 : <ui16>
                      %152 = ascvec.reinterpret %149 : <ui16> -> <ui8>
                      %153 = ascvec.reinterpret %151 : <ui16> -> <ui8>
                      %result0, %result1 = ascvec.deintlv %152, %153 : <ui8>, <ui8> -> <ui8>, <ui8>
                      %154 = arith.addi %138, %c512_i32_7 overflow<nsw> : i32
                      %155 = arith.index_cast %154 : i32 to index
                      ascvec.storealign %arg38[%155], %149, %mask_12 : memref<?xui16, 1>, index, <ui16>
                      %156 = arith.addi %138, %c640_i32 overflow<nsw> : i32
                      %157 = arith.index_cast %156 : i32 to index
                      ascvec.storealign %arg38[%157], %151, %mask_12 : memref<?xui16, 1>, index, <ui16>
                      %158 = ascvec.cumulative_histogram %arg41, %result1, %mask : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                      %159 = ascvec.cumulative_histogram %arg42, %result1, %mask {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                      scf.yield %158, %159 : !ascvec.vreg<ui16>, !ascvec.vreg<ui16>
                    }
                    ascvec.storealign %arg39[%c32832], %129#0, %mask_12 : memref<?xui16, 1>, index, <ui16>
                    ascvec.storealign %arg39[%c32960], %129#1, %mask_12 : memref<?xui16, 1>, index, <ui16>
                    %130 = ascvec.loadalign_brc %arg36[%c17320] : memref<?xi32, 1>, index -> <i32>
                    %mask_16, %remaining_out_17 = ascvec.update_mask %c64_i32_9 : !ascvec.mask
                    %131 = ascvec.duplicate_scalar %c0_i32_8, %mask_16 : i32 -> <i32>
                    %132 = ascvec.sub %130, %131, %mask_16 : <i32>
                    %133 = ascvec.max_scalar %132, %c0_i32_8, %mask_16 : <i32>, i32
                    %134 = ascvec.min_scalar %133, %c4096_i32, %mask_16 : <i32>, i32
                    %135 = ascvec.reinterpret %134 : <i32> -> <ui16>
                    %136 = ascvec.duplicate %135, %mask_12 : <ui16> -> <ui16>
                    %137:2 = scf.for %arg40 = %c0_i32_8 to %c16_i32_6 step %c1_i32_5 iter_args(%arg41 = %121, %arg42 = %121) -> (!ascvec.vreg<ui16>, !ascvec.vreg<ui16>)  : i32 {
                      %138 = arith.muli %arg40, %c256_i32_11 overflow<nsw> : i32
                      %139 = arith.addi %138, %c4096_i32 overflow<nsw> : i32
                      %140 = arith.index_cast %139 : i32 to index
                      %141 = ascvec.loadalign %arg37[%140] : memref<8192xui16, 1>, index -> <ui16>
                      %142 = arith.addi %138, %c4224_i32 overflow<nsw> : i32
                      %143 = arith.index_cast %142 : i32 to index
                      %144 = ascvec.loadalign %arg37[%143] : memref<8192xui16, 1>, index -> <ui16>
                      %145 = arith.trunci %138 : i32 to i16
                      %146 = ascvec.arange %145 : i16 -> <ui16>
                      %147 = arith.addi %138, %c128_i32_10 overflow<nsw> : i32
                      %148 = arith.trunci %147 : i32 to i16
                      %149 = ascvec.arange %148 : i16 -> <ui16>
                      %150 = ascvec.cmp_lt %146, %136, %mask_12 : <ui16>, <ui16> -> !ascvec.mask
                      %151 = ascvec.select %141, %121, %150 : <ui16>
                      %152 = ascvec.cmp_lt %149, %136, %mask_12 : <ui16>, <ui16> -> !ascvec.mask
                      %153 = ascvec.select %144, %121, %152 : <ui16>
                      %154 = ascvec.reinterpret %151 : <ui16> -> <ui8>
                      %155 = ascvec.reinterpret %153 : <ui16> -> <ui8>
                      %result0, %result1 = ascvec.deintlv %154, %155 : <ui8>, <ui8> -> <ui8>, <ui8>
                      %156 = arith.addi %138, %c5120_i32 overflow<nsw> : i32
                      %157 = arith.index_cast %156 : i32 to index
                      ascvec.storealign %arg38[%157], %151, %mask_12 : memref<?xui16, 1>, index, <ui16>
                      %158 = arith.addi %138, %c5248_i32 overflow<nsw> : i32
                      %159 = arith.index_cast %158 : i32 to index
                      ascvec.storealign %arg38[%159], %153, %mask_12 : memref<?xui16, 1>, index, <ui16>
                      %160 = ascvec.cumulative_histogram %arg41, %result1, %mask : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                      %161 = ascvec.cumulative_histogram %arg42, %result1, %mask {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                      scf.yield %160, %161 : !ascvec.vreg<ui16>, !ascvec.vreg<ui16>
                    }
                    %c0_18 = arith.constant 0 : index
                    %c1 = arith.constant 1 : index
                    scf.for %arg40 = %c0_18 to %c1 step %c1 {
                      ascvec.storealign %arg39[%c33088], %137#0, %mask_12 : memref<?xui16, 1>, index, <ui16>
                      ascvec.storealign %arg39[%c33216], %137#1, %mask_12 : memref<?xui16, 1>, index, <ui16>
                    }
                  } {cannir.vf_mode = "simd"}
                  ascvec.release_buf v, %111, 0 {cannir.buffer.generated_sync}
                  scf.yield %c4608_i64 : i64
                } else {
                  %120 = arith.trunci %116 : i64 to i32
                  ascvec.get_buf v, %111, 0 {cannir.buffer.generated_sync}
                  %121 = ascscalar.bitcast_memref %alloc_4 byte_offset = %c0_i64 : memref<262144xui8, 1> -> memref<?xui16, 1>
                  ascvec.vf(%87, %120, %121, %121, %121, %110, %87, %121, %121, %121 : memref<?xi32, 1>, i32, memref<?xui16, 1>, memref<?xui16, 1>, memref<?xui16, 1>, memref<8192xui16, 1>, memref<?xi32, 1>, memref<?xui16, 1>, memref<?xui16, 1>, memref<?xui16, 1>) {
                  ^bb0(%arg36: memref<?xi32, 1>, %arg37: i32, %arg38: memref<?xui16, 1>, %arg39: memref<?xui16, 1>, %arg40: memref<?xui16, 1>, %arg41: memref<8192xui16, 1>, %arg42: memref<?xi32, 1>, %arg43: memref<?xui16, 1>, %arg44: memref<?xui16, 1>, %arg45: memref<?xui16, 1>):
                    %c17368_5 = arith.constant 17368 : index
                    %c17320 = arith.constant 17320 : index
                    %c17936 = arith.constant 17936 : index
                    %c34112 = arith.constant 34112 : index
                    %c33600 = arith.constant 33600 : index
                    %c17360_6 = arith.constant 17360 : index
                    %c1_i32_7 = arith.constant 1 : i32
                    %c32_i32 = arith.constant 32 : i32
                    %c13312 = arith.constant 13312 : index
                    %c33856 = arith.constant 33856 : index
                    %c33344 = arith.constant 33344 : index
                    %c4096_i32 = arith.constant 4096 : i32
                    %c0_i32_8 = arith.constant 0 : i32
                    %c64_i32_9 = arith.constant 64 : i32
                    %c17312 = arith.constant 17312 : index
                    %c128_i32_10 = arith.constant 128 : i32
                    %mask, %remaining_out = ascvec.update_mask %c128_i32_10 elem_bits 16 : !ascvec.mask
                    %158 = ascvec.loadalign_brc %arg36[%c17312] : memref<?xi32, 1>, index -> <i32>
                    %mask_11, %remaining_out_12 = ascvec.update_mask %c64_i32_9 : !ascvec.mask
                    %159 = ascvec.duplicate_scalar %arg37, %mask_11 : i32 -> <i32>
                    %160 = ascvec.sub %158, %159, %mask_11 : <i32>
                    %161 = ascvec.max_scalar %160, %c0_i32_8, %mask_11 : <i32>, i32
                    %162 = ascvec.min_scalar %161, %c4096_i32, %mask_11 : <i32>, i32
                    %163 = ascvec.reinterpret %162 : <i32> -> <ui16>
                    %164 = ascvec.duplicate %163, %mask : <ui16> -> <ui16>
                    %165 = ascvec.loadalign_brc %arg38[%c33344] : memref<?xui16, 1>, index -> <ui16>
                    %166 = ascvec.loadalign_brc %arg39[%c33856] : memref<?xui16, 1>, index -> <ui16>
                    %167 = ascvec.shl %165, 8, %mask : <ui16> -> <ui16>
                    %168 = ascvec.add %167, %166, %mask : <ui16>
                    ascvec.store_unalign_begin %arg40 : memref<?xui16, 1>
                    scf.for %arg46 = %c0_i32_8 to %c32_i32 step %c1_i32_7  : i32 {
                      %180 = arith.muli %arg46, %c128_i32_10 overflow<nsw> : i32
                      %181 = arith.index_cast %180 : i32 to index
                      %182 = ascvec.loadalign %arg41[%181] : memref<8192xui16, 1>, index -> <ui16>
                      %183 = arith.trunci %180 : i32 to i16
                      %184 = ascvec.arange %183 : i16 -> <ui16>
                      %185 = ascvec.cmp_lt %184, %164, %mask : <ui16>, <ui16> -> !ascvec.mask
                      %186 = ascvec.cmp_gt %182, %168, %185 : <ui16>, <ui16> -> !ascvec.mask
                      %187 = ascvec.squeeze_store_reg %184, %186 : <ui16>, !ascvec.mask -> <ui16>
                      ascvec.store_unalign %arg40[%c13312], %187 : memref<?xui16, 1>, <ui16>
                    }
                    ascvec.store_unalign_post %arg40[%c13312] : memref<?xui16, 1>
                    ascvec.storealign_squeeze_status %arg42[%c17360_6] : memref<?xi32, 1>
                    %169 = ascvec.loadalign_brc %arg36[%c17320] : memref<?xi32, 1>, index -> <i32>
                    %mask_13, %remaining_out_14 = ascvec.update_mask %c64_i32_9 : !ascvec.mask
                    %170 = ascvec.duplicate_scalar %arg37, %mask_13 : i32 -> <i32>
                    %171 = ascvec.sub %169, %170, %mask_13 : <i32>
                    %172 = ascvec.max_scalar %171, %c0_i32_8, %mask_13 : <i32>, i32
                    %173 = ascvec.min_scalar %172, %c4096_i32, %mask_13 : <i32>, i32
                    %174 = ascvec.reinterpret %173 : <i32> -> <ui16>
                    %175 = ascvec.duplicate %174, %mask : <ui16> -> <ui16>
                    %176 = ascvec.loadalign_brc %arg43[%c33600] : memref<?xui16, 1>, index -> <ui16>
                    %177 = ascvec.loadalign_brc %arg44[%c34112] : memref<?xui16, 1>, index -> <ui16>
                    %178 = ascvec.shl %176, 8, %mask : <ui16> -> <ui16>
                    %179 = ascvec.add %178, %177, %mask : <ui16>
                    ascvec.store_unalign_begin %arg45 : memref<?xui16, 1>
                    scf.for %arg46 = %c0_i32_8 to %c32_i32 step %c1_i32_7  : i32 {
                      %180 = arith.muli %arg46, %c128_i32_10 overflow<nsw> : i32
                      %181 = arith.addi %180, %c4096_i32 overflow<nsw> : i32
                      %182 = arith.index_cast %181 : i32 to index
                      %183 = ascvec.loadalign %arg41[%182] : memref<8192xui16, 1>, index -> <ui16>
                      %184 = arith.trunci %180 : i32 to i16
                      %185 = ascvec.arange %184 : i16 -> <ui16>
                      %186 = ascvec.cmp_lt %185, %175, %mask : <ui16>, <ui16> -> !ascvec.mask
                      %187 = ascvec.cmp_gt %183, %179, %186 : <ui16>, <ui16> -> !ascvec.mask
                      %188 = ascvec.squeeze_store_reg %185, %187 : <ui16>, !ascvec.mask -> <ui16>
                      ascvec.store_unalign %arg45[%c17936], %188 : memref<?xui16, 1>, <ui16>
                    }
                    %c0_15 = arith.constant 0 : index
                    %c1 = arith.constant 1 : index
                    scf.for %arg46 = %c0_15 to %c1 step %c1 {
                      ascvec.store_unalign_post %arg45[%c17936] : memref<?xui16, 1>
                      ascvec.storealign_squeeze_status %arg42[%c17368_5] : memref<?xi32, 1>
                    }
                  } {cannir.vf_mode = "simd"}
                  ascvec.release_buf v, %111, 0 {cannir.buffer.generated_sync}
                  ascvec.sync_all
                  ascvec.get_buf s, %c0_i64, 0 {cannir.buffer.generated_sync}
                  %122 = memref.load %87[%c17360] : memref<?xi32, 1>
                  %123 = arith.extsi %122 : i32 to i64
                  %124 = arith.remsi %123, %c2_i64 : i64
                  %125 = arith.cmpi ne, %124, %c0_i64 : i64
                  %126 = arith.cmpi slt, %124, %c0_i64 : i64
                  %127 = arith.andi %125, %126 : i1
                  %128 = arith.divsi %123, %c2_i64 : i64
                  %129 = arith.extui %127 : i1 to i64
                  %130 = arith.subi %128, %129 : i64
                  %131 = memref.load %87[%c17368] : memref<?xi32, 1>
                  ascvec.release_buf s, %c0_i64, 0 {cannir.buffer.generated_sync}
                  %132 = arith.extsi %131 : i32 to i64
                  %133 = arith.remsi %132, %c2_i64 : i64
                  %134 = arith.cmpi ne, %133, %c0_i64 : i64
                  %135 = arith.cmpi slt, %133, %c0_i64 : i64
                  %136 = arith.andi %134, %135 : i1
                  %137 = arith.divsi %132, %c2_i64 : i64
                  %138 = arith.extui %136 : i1 to i64
                  %139 = arith.subi %137, %138 : i64
                  %140 = arith.maxsi %130, %139 : i64
                  %141 = arith.addi %140, %c255_i64 : i64
                  %142 = arith.remsi %141, %c256_i64 : i64
                  %143 = arith.cmpi ne, %142, %c0_i64 : i64
                  %144 = arith.cmpi slt, %142, %c0_i64 : i64
                  %145 = arith.andi %143, %144 : i1
                  %146 = arith.divsi %141, %c256_i64 : i64
                  %147 = arith.extui %145 : i1 to i64
                  %148 = arith.subi %146, %147 : i64
                  %149 = arith.muli %148, %c256_i64 : i64
                  %150 = arith.addi %149, %c512_i64 : i64
                  %151 = arith.remsi %149, %c256_i64 : i64
                  %152 = arith.cmpi ne, %151, %c0_i64 : i64
                  %153 = arith.cmpi slt, %151, %c0_i64 : i64
                  %154 = arith.andi %152, %153 : i1
                  %155 = arith.divsi %149, %c256_i64 : i64
                  %156 = arith.extui %154 : i1 to i64
                  %157 = arith.subi %155, %156 : i64
                  ascvec.get_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
                  ascvec.get_buf v, %111, 0 {cannir.buffer.generated_sync}
                  ascvec.vf(%87, %121, %121, %110, %121, %157, %121 : memref<?xi32, 1>, memref<?xui16, 1>, memref<?xui16, 1>, memref<8192xui16, 1>, memref<?xui16, 1>, i64, memref<?xui16, 1>) {
                  ^bb0(%arg36: memref<?xi32, 1>, %arg37: memref<?xui16, 1>, %arg38: memref<?xui16, 1>, %arg39: memref<8192xui16, 1>, %arg40: memref<?xui16, 1>, %arg41: i64, %arg42: memref<?xui16, 1>):
                    %c5248_i64 = arith.constant 5248 : i64
                    %c33216 = arith.constant 33216 : index
                    %c33088 = arith.constant 33088 : index
                    %c17368_5 = arith.constant 17368 : index
                    %c640_i64 = arith.constant 640 : i64
                    %c32960 = arith.constant 32960 : index
                    %c5120_i64 = arith.constant 5120 : i64
                    %c4096_i16 = arith.constant 4096 : i16
                    %c17936 = arith.constant 17936 : index
                    %c1_i64_6 = arith.constant 1 : i64
                    %c0_i64_7 = arith.constant 0 : i64
                    %c128_i64_8 = arith.constant 128 : i64
                    %c512_i64_9 = arith.constant 512 : i64
                    %c13312 = arith.constant 13312 : index
                    %c256_i64_10 = arith.constant 256 : i64
                    %c32832 = arith.constant 32832 : index
                    %c64_i32_11 = arith.constant 64 : i32
                    %c0_12 = arith.constant 0 : index
                    %c17360_13 = arith.constant 17360 : index
                    %c0_i16 = arith.constant 0 : i16
                    %c256_i32_14 = arith.constant 256 : i32
                    %c128_i32_15 = arith.constant 128 : i32
                    %mask, %remaining_out = ascvec.update_mask %c128_i32_15 elem_bits 16 : !ascvec.mask
                    %mask_16, %remaining_out_17 = ascvec.update_mask %c256_i32_14 elem_bits 8 : !ascvec.mask
                    %158 = ascvec.duplicate_scalar %c0_i16, %mask : i16 -> <ui16>
                    %159 = ascvec.loadalign_brc %arg36[%c17360_13] : memref<?xi32, 1>, index -> <i32>
                    %mask_18, %remaining_out_19 = ascvec.update_mask %c64_i32_11 : !ascvec.mask
                    %160 = ascvec.shr %159, 1, %mask_18 : <i32> -> <i32>
                    %161 = ascvec.reinterpret %160 : <i32> -> <ui16>
                    %162 = ascvec.duplicate %161, %mask : <ui16> -> <ui16>
                    %163 = ascvec.loadalign %arg37[%c32832] : memref<?xui16, 1>, index -> <ui16>
                    %164 = ascvec.loadalign %arg37[%c32960] : memref<?xui16, 1>, index -> <ui16>
                    %165:2 = scf.for %arg43 = %c0_i64_7 to %arg41 step %c1_i64_6 iter_args(%arg44 = %163, %arg45 = %164) -> (!ascvec.vreg<ui16>, !ascvec.vreg<ui16>)  : i64 {
                      %173 = arith.muli %arg43, %c256_i64_10 : i64
                      %174 = arith.trunci %173 : i64 to i16
                      %175 = ascvec.arange %174 : i16 -> <ui16>
                      %176 = ascvec.cmp_lt %175, %162, %mask : <ui16>, <ui16> -> !ascvec.mask
                      %177 = arith.index_cast %173 : i64 to index
                      %178 = arith.addi %177, %c13312 : index
                      %179 = ascvec.loadalign %arg38[%178] : memref<?xui16, 1>, index -> <ui16>
                      %180 = ascvec.select %179, %158, %176 : <ui16>
                      %181 = ascvec.add_scalar %180, %c0_i16, %mask : <ui16>, i16
                      %182 = ascvec.gather %arg39[%c0_12], %181, %mask : memref<8192xui16, 1>, <ui16>, !ascvec.mask -> <ui16>
                      %183 = ascvec.select %182, %158, %176 : <ui16>
                      %184 = arith.addi %173, %c512_i64_9 : i64
                      %185 = arith.index_cast %184 : i64 to index
                      ascvec.storealign %arg40[%185], %183, %mask : memref<?xui16, 1>, index, <ui16>
                      %186 = arith.addi %173, %c128_i64_8 : i64
                      %187 = arith.trunci %186 : i64 to i16
                      %188 = ascvec.arange %187 : i16 -> <ui16>
                      %189 = ascvec.cmp_lt %188, %162, %mask : <ui16>, <ui16> -> !ascvec.mask
                      %190 = arith.index_cast %186 : i64 to index
                      %191 = arith.addi %190, %c13312 : index
                      %192 = ascvec.loadalign %arg38[%191] : memref<?xui16, 1>, index -> <ui16>
                      %193 = ascvec.select %192, %158, %189 : <ui16>
                      %194 = ascvec.add_scalar %193, %c0_i16, %mask : <ui16>, i16
                      %195 = ascvec.gather %arg39[%c0_12], %194, %mask : memref<8192xui16, 1>, <ui16>, !ascvec.mask -> <ui16>
                      %196 = ascvec.select %195, %158, %189 : <ui16>
                      %197 = arith.addi %173, %c640_i64 : i64
                      %198 = arith.index_cast %197 : i64 to index
                      ascvec.storealign %arg40[%198], %196, %mask : memref<?xui16, 1>, index, <ui16>
                      %199 = ascvec.reinterpret %183 : <ui16> -> <ui8>
                      %200 = ascvec.reinterpret %196 : <ui16> -> <ui8>
                      %result0, %result1 = ascvec.deintlv %199, %200 : <ui8>, <ui8> -> <ui8>, <ui8>
                      %201 = ascvec.cumulative_histogram %arg44, %result1, %mask_16 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                      %202 = ascvec.cumulative_histogram %arg45, %result1, %mask_16 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                      scf.yield %201, %202 : !ascvec.vreg<ui16>, !ascvec.vreg<ui16>
                    }
                    ascvec.storealign %arg37[%c32832], %165#0, %mask : memref<?xui16, 1>, index, <ui16>
                    ascvec.storealign %arg37[%c32960], %165#1, %mask : memref<?xui16, 1>, index, <ui16>
                    %166 = ascvec.loadalign_brc %arg36[%c17368_5] : memref<?xi32, 1>, index -> <i32>
                    %mask_20, %remaining_out_21 = ascvec.update_mask %c64_i32_11 : !ascvec.mask
                    %167 = ascvec.shr %166, 1, %mask_20 : <i32> -> <i32>
                    %168 = ascvec.reinterpret %167 : <i32> -> <ui16>
                    %169 = ascvec.duplicate %168, %mask : <ui16> -> <ui16>
                    %170 = ascvec.loadalign %arg37[%c33088] : memref<?xui16, 1>, index -> <ui16>
                    %171 = ascvec.loadalign %arg37[%c33216] : memref<?xui16, 1>, index -> <ui16>
                    %172:2 = scf.for %arg43 = %c0_i64_7 to %arg41 step %c1_i64_6 iter_args(%arg44 = %170, %arg45 = %171) -> (!ascvec.vreg<ui16>, !ascvec.vreg<ui16>)  : i64 {
                      %173 = arith.muli %arg43, %c256_i64_10 : i64
                      %174 = arith.trunci %173 : i64 to i16
                      %175 = ascvec.arange %174 : i16 -> <ui16>
                      %176 = ascvec.cmp_lt %175, %169, %mask : <ui16>, <ui16> -> !ascvec.mask
                      %177 = arith.index_cast %173 : i64 to index
                      %178 = arith.addi %177, %c17936 : index
                      %179 = ascvec.loadalign %arg42[%178] : memref<?xui16, 1>, index -> <ui16>
                      %180 = ascvec.select %179, %158, %176 : <ui16>
                      %181 = ascvec.add_scalar %180, %c4096_i16, %mask : <ui16>, i16
                      %182 = ascvec.gather %arg39[%c0_12], %181, %mask : memref<8192xui16, 1>, <ui16>, !ascvec.mask -> <ui16>
                      %183 = ascvec.select %182, %158, %176 : <ui16>
                      %184 = arith.addi %173, %c5120_i64 : i64
                      %185 = arith.index_cast %184 : i64 to index
                      ascvec.storealign %arg40[%185], %183, %mask : memref<?xui16, 1>, index, <ui16>
                      %186 = arith.addi %173, %c128_i64_8 : i64
                      %187 = arith.trunci %186 : i64 to i16
                      %188 = ascvec.arange %187 : i16 -> <ui16>
                      %189 = ascvec.cmp_lt %188, %169, %mask : <ui16>, <ui16> -> !ascvec.mask
                      %190 = arith.index_cast %186 : i64 to index
                      %191 = arith.addi %190, %c17936 : index
                      %192 = ascvec.loadalign %arg42[%191] : memref<?xui16, 1>, index -> <ui16>
                      %193 = ascvec.select %192, %158, %189 : <ui16>
                      %194 = ascvec.add_scalar %193, %c4096_i16, %mask : <ui16>, i16
                      %195 = ascvec.gather %arg39[%c0_12], %194, %mask : memref<8192xui16, 1>, <ui16>, !ascvec.mask -> <ui16>
                      %196 = ascvec.select %195, %158, %189 : <ui16>
                      %197 = arith.addi %173, %c5248_i64 : i64
                      %198 = arith.index_cast %197 : i64 to index
                      ascvec.storealign %arg40[%198], %196, %mask : memref<?xui16, 1>, index, <ui16>
                      %199 = ascvec.reinterpret %183 : <ui16> -> <ui8>
                      %200 = ascvec.reinterpret %196 : <ui16> -> <ui8>
                      %result0, %result1 = ascvec.deintlv %199, %200 : <ui8>, <ui8> -> <ui8>, <ui8>
                      %201 = ascvec.cumulative_histogram %arg44, %result1, %mask_16 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                      %202 = ascvec.cumulative_histogram %arg45, %result1, %mask_16 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                      scf.yield %201, %202 : !ascvec.vreg<ui16>, !ascvec.vreg<ui16>
                    }
                    %c0_22 = arith.constant 0 : index
                    %c1 = arith.constant 1 : index
                    scf.for %arg43 = %c0_22 to %c1 step %c1 {
                      ascvec.storealign %arg37[%c33088], %172#0, %mask : memref<?xui16, 1>, index, <ui16>
                      ascvec.storealign %arg37[%c33216], %172#1, %mask : memref<?xui16, 1>, index, <ui16>
                    }
                  } {cannir.vf_mode = "simd"}
                  ascvec.release_buf v, %111, 0 {cannir.buffer.generated_sync}
                  ascvec.release_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
                  scf.yield %150 : i64
                }
                ascvec.vf(%87, %87 : memref<?xi32, 1>, memref<?xi32, 1>) {
                ^bb0(%arg36: memref<?xi32, 1>, %arg37: memref<?xi32, 1>):
                  %c17336 = arith.constant 17336 : index
                  %c17320 = arith.constant 17320 : index
                  %c17328 = arith.constant 17328 : index
                  %c512_i32_5 = arith.constant 512 : i32
                  %c17312 = arith.constant 17312 : index
                  %c1_i32_6 = arith.constant 1 : i32
                  %c0_7 = arith.constant 0 : index
                  %c1 = arith.constant 1 : index
                  scf.for %arg38 = %c0_7 to %c1 step %c1 {
                    %mask, %remaining_out = ascvec.update_mask %c1_i32_6 : !ascvec.mask
                    %120 = ascvec.loadalign_brc %arg36[%c17312] : memref<?xi32, 1>, index -> <i32>
                    %121 = ascvec.min_scalar %120, %c512_i32_5, %mask : <i32>, i32
                    ascvec.storealign_1st %arg37[%c17328], %121 : memref<?xi32, 1>, index, <i32>
                    %122 = ascvec.loadalign_brc %arg36[%c17320] : memref<?xi32, 1>, index -> <i32>
                    %123 = ascvec.min_scalar %122, %c512_i32_5, %mask : <i32>, i32
                    ascvec.storealign_1st %arg37[%c17336], %123 : memref<?xi32, 1>, index, <i32>
                  }
                } {cannir.vf_mode = "simd"}
                %119 = arith.cmpi eq, %98, %71 : i64
                scf.if %119 {
                  scf.if %117 {
                    %120 = ascscalar.bitcast_memref %alloc_4 byte_offset = %c0_i64 : memref<262144xui8, 1> -> memref<?xui16, 1>
                    ascvec.vf(%87, %120, %120, %120, %120 : memref<?xi32, 1>, memref<?xui16, 1>, memref<?xui16, 1>, memref<?xui16, 1>, memref<?xui16, 1>) {
                    ^bb0(%arg36: memref<?xi32, 1>, %arg37: memref<?xui16, 1>, %arg38: memref<?xui16, 1>, %arg39: memref<?xui16, 1>, %arg40: memref<?xui16, 1>):
                      %c34496 = arith.constant 34496 : index
                      %c17336 = arith.constant 17336 : index
                      %c256_i16 = arith.constant 256 : i16
                      %c34368 = arith.constant 34368 : index
                      %c256_i32_5 = arith.constant 256 : i32
                      %c33600 = arith.constant 33600 : index
                      %c2_i32 = arith.constant 2 : i32
                      %c0_i32_6 = arith.constant 0 : i32
                      %c32832 = arith.constant 32832 : index
                      %c33344 = arith.constant 33344 : index
                      %c4097_i32 = arith.constant 4097 : i32
                      %c1_i32_7 = arith.constant 1 : i32
                      %c17328 = arith.constant 17328 : index
                      %c1_i16 = arith.constant 1 : i16
                      %c0_i16 = arith.constant 0 : i16
                      %c64_i32_8 = arith.constant 64 : i32
                      %c128_i32_9 = arith.constant 128 : i32
                      %mask, %remaining_out = ascvec.update_mask %c128_i32_9 elem_bits 16 : !ascvec.mask
                      %mask_10, %remaining_out_11 = ascvec.update_mask %c64_i32_8 : !ascvec.mask
                      %124 = ascvec.duplicate_scalar %c0_i16, %mask : i16 -> <ui16>
                      %125 = ascvec.duplicate_scalar %c1_i16, %mask : i16 -> <ui16>
                      %126 = ascvec.loadalign_brc %arg36[%c17328] : memref<?xi32, 1>, index -> <i32>
                      %127 = ascvec.loadalign_brc %arg36[%c17336] : memref<?xi32, 1>, index -> <i32>
                      %128 = ascvec.max_scalar %126, %c1_i32_7, %mask_10 : <i32>, i32
                      %129 = ascvec.duplicate_scalar %c4097_i32, %mask_10 : i32 -> <i32>
                      %130 = ascvec.sub %129, %128, %mask_10 : <i32>
                      %131 = ascvec.reinterpret %130 : <i32> -> <ui16>
                      %132 = ascvec.duplicate %131, %mask : <ui16> -> <ui16>
                      ascvec.store_unalign_begin %arg37 : memref<?xui16, 1>
                      scf.for %arg41 = %c0_i32_6 to %c2_i32 step %c1_i32_7  : i32 {
                        %137 = arith.muli %arg41, %c128_i32_9 overflow<nsw> : i32
                        %138 = arith.trunci %137 : i32 to i16
                        %139 = ascvec.arange %138 : i16 -> <ui16>
                        %140 = arith.index_cast %137 : i32 to index
                        %141 = arith.addi %140, %c32832 : index
                        %142 = ascvec.loadalign %arg38[%141] : memref<?xui16, 1>, index -> <ui16>
                        %143 = ascvec.cmp_ge %142, %132, %mask : <ui16>, <ui16> -> !ascvec.mask
                        %144 = ascvec.squeeze_store_reg %139, %143 : <ui16>, !ascvec.mask -> <ui16>
                        ascvec.store_unalign %arg37[%c33344], %144 : memref<?xui16, 1>, <ui16>
                      }
                      ascvec.store_unalign_post %arg37[%c33344] : memref<?xui16, 1>
                      %133 = ascvec.max_scalar %127, %c1_i32_7, %mask_10 : <i32>, i32
                      %134 = ascvec.sub %129, %133, %mask_10 : <i32>
                      %135 = ascvec.reinterpret %134 : <i32> -> <ui16>
                      %136 = ascvec.duplicate %135, %mask : <ui16> -> <ui16>
                      ascvec.store_unalign_begin %arg39 : memref<?xui16, 1>
                      scf.for %arg41 = %c0_i32_6 to %c2_i32 step %c1_i32_7  : i32 {
                        %137 = arith.muli %arg41, %c128_i32_9 overflow<nsw> : i32
                        %138 = arith.trunci %137 : i32 to i16
                        %139 = ascvec.arange %138 : i16 -> <ui16>
                        %140 = arith.addi %137, %c256_i32_5 overflow<nsw> : i32
                        %141 = arith.index_cast %140 : i32 to index
                        %142 = arith.addi %141, %c32832 : index
                        %143 = ascvec.loadalign %arg38[%142] : memref<?xui16, 1>, index -> <ui16>
                        %144 = ascvec.cmp_ge %143, %136, %mask : <ui16>, <ui16> -> !ascvec.mask
                        %145 = ascvec.squeeze_store_reg %139, %144 : <ui16>, !ascvec.mask -> <ui16>
                        ascvec.store_unalign %arg39[%c33600], %145 : memref<?xui16, 1>, <ui16>
                      }
                      %c0_12 = arith.constant 0 : index
                      %c1 = arith.constant 1 : index
                      scf.for %arg41 = %c0_12 to %c1 step %c1 {
                        ascvec.store_unalign_post %arg39[%c33600] : memref<?xui16, 1>
                      }
                      ascvec.mem_bar vst_vld
                      %c0_13 = arith.constant 0 : index
                      %c1_14 = arith.constant 1 : index
                      scf.for %arg41 = %c0_13 to %c1_14 step %c1_14 {
                        %137 = ascvec.loadalign_brc %arg37[%c33344] : memref<?xui16, 1>, index -> <ui16>
                        %138 = ascvec.cmp_eq_scalar %137, %c0_i16, %mask : <ui16>, i16 -> !ascvec.mask
                        %139 = ascvec.sub %137, %125, %mask : <ui16>
                        %140 = ascvec.select %124, %139, %138 : <ui16>
                        %141 = ascvec.add %140, %124, %mask : <ui16>
                        %142 = ascvec.gather %arg38[%c32832], %141, %mask : memref<?xui16, 1>, <ui16>, !ascvec.mask -> <ui16>
                        %143 = ascvec.select %124, %142, %138 : <ui16>
                        %144 = ascvec.sub %132, %143, %mask : <ui16>
                        ascvec.storealign %arg40[%c34368], %144, %mask : memref<?xui16, 1>, index, <ui16>
                        %145 = ascvec.loadalign_brc %arg39[%c33600] : memref<?xui16, 1>, index -> <ui16>
                        %146 = ascvec.cmp_eq_scalar %145, %c0_i16, %mask : <ui16>, i16 -> !ascvec.mask
                        %147 = ascvec.sub %145, %125, %mask : <ui16>
                        %148 = ascvec.select %124, %147, %146 : <ui16>
                        %149 = ascvec.duplicate_scalar %c256_i16, %mask : i16 -> <ui16>
                        %150 = ascvec.add %148, %149, %mask : <ui16>
                        %151 = ascvec.gather %arg38[%c32832], %150, %mask : memref<?xui16, 1>, <ui16>, !ascvec.mask -> <ui16>
                        %152 = ascvec.select %124, %151, %146 : <ui16>
                        %153 = ascvec.sub %136, %152, %mask : <ui16>
                        ascvec.storealign %arg40[%c34496], %153, %mask : memref<?xui16, 1>, index, <ui16>
                      }
                    } {cannir.vf_mode = "simd"}
                    ascvec.vf(%120, %120, %120, %120 : memref<?xui16, 1>, memref<?xui16, 1>, memref<?xui16, 1>, memref<?xui16, 1>) {
                    ^bb0(%arg36: memref<?xui16, 1>, %arg37: memref<?xui16, 1>, %arg38: memref<?xui16, 1>, %arg39: memref<?xui16, 1>):
                      %c33216 = arith.constant 33216 : index
                      %c33088 = arith.constant 33088 : index
                      %c32960 = arith.constant 32960 : index
                      %c4608_i32 = arith.constant 4608 : i32
                      %c33600 = arith.constant 33600 : index
                      %c32832 = arith.constant 32832 : index
                      %c1_i32_5 = arith.constant 1 : i32
                      %c18_i32 = arith.constant 18 : i32
                      %c2_i32 = arith.constant 2 : i32
                      %c0_i16 = arith.constant 0 : i16
                      %c33344 = arith.constant 33344 : index
                      %c128_i32_6 = arith.constant 128 : i32
                      %c256_i32_7 = arith.constant 256 : i32
                      %mask, %remaining_out = ascvec.update_mask %c256_i32_7 elem_bits 8 : !ascvec.mask
                      %mask_8, %remaining_out_9 = ascvec.update_mask %c128_i32_6 elem_bits 16 : !ascvec.mask
                      %124 = ascvec.loadalign_brc %arg36[%c33344] : memref<?xui16, 1>, index -> <ui16>
                      %125 = ascvec.reinterpret %124 : <ui16> -> <ui8>
                      %126 = ascvec.duplicate %125, %mask : <ui8> -> <ui8>
                      %127 = ascvec.duplicate_scalar %c0_i16, %mask_8 : i16 -> <ui16>
                      %128:2 = scf.for %arg40 = %c2_i32 to %c18_i32 step %c1_i32_5 iter_args(%arg41 = %127, %arg42 = %127) -> (!ascvec.vreg<ui16>, !ascvec.vreg<ui16>)  : i32 {
                        %133 = arith.muli %arg40, %c256_i32_7 overflow<nsw> : i32
                        %134 = arith.index_cast %133 : i32 to index
                        %result0, %result1 = ascvec.loadalign_deintlv %arg37[%134] {dintlv_width = "b8"} : memref<?xui16, 1>, index -> <ui8>, <ui8>
                        %135 = ascvec.cmp_eq %result1, %126, %mask : <ui8>, <ui8> -> !ascvec.mask
                        %136 = ascvec.cumulative_histogram %arg41, %result0, %135 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                        %137 = ascvec.cumulative_histogram %arg42, %result0, %135 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                        scf.yield %136, %137 : !ascvec.vreg<ui16>, !ascvec.vreg<ui16>
                      }
                      ascvec.storealign %arg38[%c32832], %128#0, %mask_8 : memref<?xui16, 1>, index, <ui16>
                      ascvec.storealign %arg38[%c32960], %128#1, %mask_8 : memref<?xui16, 1>, index, <ui16>
                      %129 = ascvec.loadalign_brc %arg39[%c33600] : memref<?xui16, 1>, index -> <ui16>
                      %130 = ascvec.reinterpret %129 : <ui16> -> <ui8>
                      %131 = ascvec.duplicate %130, %mask : <ui8> -> <ui8>
                      %132:2 = scf.for %arg40 = %c2_i32 to %c18_i32 step %c1_i32_5 iter_args(%arg41 = %127, %arg42 = %127) -> (!ascvec.vreg<ui16>, !ascvec.vreg<ui16>)  : i32 {
                        %133 = arith.muli %arg40, %c256_i32_7 overflow<nsw> : i32
                        %134 = arith.addi %133, %c4608_i32 overflow<nsw> : i32
                        %135 = arith.index_cast %134 : i32 to index
                        %result0, %result1 = ascvec.loadalign_deintlv %arg37[%135] {dintlv_width = "b8"} : memref<?xui16, 1>, index -> <ui8>, <ui8>
                        %136 = ascvec.cmp_eq %result1, %131, %mask : <ui8>, <ui8> -> !ascvec.mask
                        %137 = ascvec.cumulative_histogram %arg41, %result0, %136 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                        %138 = ascvec.cumulative_histogram %arg42, %result0, %136 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                        scf.yield %137, %138 : !ascvec.vreg<ui16>, !ascvec.vreg<ui16>
                      }
                      %c0_10 = arith.constant 0 : index
                      %c1 = arith.constant 1 : index
                      scf.for %arg40 = %c0_10 to %c1 step %c1 {
                        ascvec.storealign %arg38[%c33088], %132#0, %mask_8 : memref<?xui16, 1>, index, <ui16>
                        ascvec.storealign %arg38[%c33216], %132#1, %mask_8 : memref<?xui16, 1>, index, <ui16>
                      }
                    } {cannir.vf_mode = "simd"}
                    ascvec.vf(%120, %120, %120, %120 : memref<?xui16, 1>, memref<?xui16, 1>, memref<?xui16, 1>, memref<?xui16, 1>) {
                    ^bb0(%arg36: memref<?xui16, 1>, %arg37: memref<?xui16, 1>, %arg38: memref<?xui16, 1>, %arg39: memref<?xui16, 1>):
                      %c34496 = arith.constant 34496 : index
                      %c256_i32_5 = arith.constant 256 : i32
                      %c34112 = arith.constant 34112 : index
                      %c1_i32_6 = arith.constant 1 : i32
                      %c2_i32 = arith.constant 2 : i32
                      %c0_i32_7 = arith.constant 0 : i32
                      %c32832 = arith.constant 32832 : index
                      %c33856 = arith.constant 33856 : index
                      %c34368 = arith.constant 34368 : index
                      %c128_i32_8 = arith.constant 128 : i32
                      %mask, %remaining_out = ascvec.update_mask %c128_i32_8 elem_bits 16 : !ascvec.mask
                      %124 = ascvec.loadalign_brc %arg36[%c34368] : memref<?xui16, 1>, index -> <ui16>
                      ascvec.store_unalign_begin %arg37 : memref<?xui16, 1>
                      scf.for %arg40 = %c0_i32_7 to %c2_i32 step %c1_i32_6  : i32 {
                        %126 = arith.muli %arg40, %c128_i32_8 overflow<nsw> : i32
                        %127 = arith.trunci %126 : i32 to i16
                        %128 = ascvec.arange %127 : i16 -> <ui16>
                        %129 = arith.index_cast %126 : i32 to index
                        %130 = arith.addi %129, %c32832 : index
                        %131 = ascvec.loadalign %arg38[%130] : memref<?xui16, 1>, index -> <ui16>
                        %132 = ascvec.cmp_ge %131, %124, %mask : <ui16>, <ui16> -> !ascvec.mask
                        %133 = ascvec.squeeze_store_reg %128, %132 : <ui16>, !ascvec.mask -> <ui16>
                        ascvec.store_unalign %arg37[%c33856], %133 : memref<?xui16, 1>, <ui16>
                      }
                      ascvec.store_unalign_post %arg37[%c33856] : memref<?xui16, 1>
                      %125 = ascvec.loadalign_brc %arg36[%c34496] : memref<?xui16, 1>, index -> <ui16>
                      ascvec.store_unalign_begin %arg39 : memref<?xui16, 1>
                      scf.for %arg40 = %c0_i32_7 to %c2_i32 step %c1_i32_6  : i32 {
                        %126 = arith.muli %arg40, %c128_i32_8 overflow<nsw> : i32
                        %127 = arith.trunci %126 : i32 to i16
                        %128 = ascvec.arange %127 : i16 -> <ui16>
                        %129 = arith.addi %126, %c256_i32_5 overflow<nsw> : i32
                        %130 = arith.index_cast %129 : i32 to index
                        %131 = arith.addi %130, %c32832 : index
                        %132 = ascvec.loadalign %arg38[%131] : memref<?xui16, 1>, index -> <ui16>
                        %133 = ascvec.cmp_ge %132, %125, %mask : <ui16>, <ui16> -> !ascvec.mask
                        %134 = ascvec.squeeze_store_reg %128, %133 : <ui16>, !ascvec.mask -> <ui16>
                        ascvec.store_unalign %arg39[%c34112], %134 : memref<?xui16, 1>, <ui16>
                      }
                      %c0_9 = arith.constant 0 : index
                      %c1 = arith.constant 1 : index
                      scf.for %arg40 = %c0_9 to %c1 step %c1 {
                        ascvec.store_unalign_post %arg39[%c34112] : memref<?xui16, 1>
                      }
                    } {cannir.vf_mode = "simd"}
                    ascvec.vf(%120, %120, %120, %120, %120, %120, %120 : memref<?xui16, 1>, memref<?xui16, 1>, memref<?xui16, 1>, memref<?xui16, 1>, memref<?xui16, 1>, memref<?xui16, 1>, memref<?xui16, 1>) {
                    ^bb0(%arg36: memref<?xui16, 1>, %arg37: memref<?xui16, 1>, %arg38: memref<?xui16, 1>, %arg39: memref<?xui16, 1>, %arg40: memref<?xui16, 1>, %arg41: memref<?xui16, 1>, %arg42: memref<?xui16, 1>):
                      %c4608_i32 = arith.constant 4608 : i32
                      %c27184 = arith.constant 27184 : index
                      %c34112 = arith.constant 34112 : index
                      %c33600 = arith.constant 33600 : index
                      %c1_i32_5 = arith.constant 1 : i32
                      %c36_i32 = arith.constant 36 : i32
                      %c4_i32 = arith.constant 4 : i32
                      %c22560 = arith.constant 22560 : index
                      %c33856 = arith.constant 33856 : index
                      %c33344 = arith.constant 33344 : index
                      %c128_i32_6 = arith.constant 128 : i32
                      %mask, %remaining_out = ascvec.update_mask %c128_i32_6 elem_bits 16 : !ascvec.mask
                      %124 = ascvec.loadalign_brc %arg36[%c33344] : memref<?xui16, 1>, index -> <ui16>
                      %125 = ascvec.loadalign_brc %arg37[%c33856] : memref<?xui16, 1>, index -> <ui16>
                      %126 = ascvec.shl %124, 8, %mask : <ui16> -> <ui16>
                      %127 = ascvec.add %126, %125, %mask : <ui16>
                      ascvec.store_unalign_begin %arg38 : memref<?xui16, 1>
                      scf.for %arg43 = %c4_i32 to %c36_i32 step %c1_i32_5  : i32 {
                        %132 = arith.muli %arg43, %c128_i32_6 overflow<nsw> : i32
                        %133 = arith.index_cast %132 : i32 to index
                        %134 = ascvec.loadalign %arg39[%133] : memref<?xui16, 1>, index -> <ui16>
                        %135 = arith.trunci %132 : i32 to i16
                        %136 = ascvec.arange %135 : i16 -> <ui16>
                        %137 = ascvec.cmp_gt %134, %127, %mask : <ui16>, <ui16> -> !ascvec.mask
                        %138 = ascvec.squeeze_store_reg %136, %137 : <ui16>, !ascvec.mask -> <ui16>
                        ascvec.store_unalign %arg38[%c22560], %138 : memref<?xui16, 1>, <ui16>
                      }
                      scf.for %arg43 = %c4_i32 to %c36_i32 step %c1_i32_5  : i32 {
                        %132 = arith.muli %arg43, %c128_i32_6 overflow<nsw> : i32
                        %133 = arith.index_cast %132 : i32 to index
                        %134 = ascvec.loadalign %arg39[%133] : memref<?xui16, 1>, index -> <ui16>
                        %135 = arith.trunci %132 : i32 to i16
                        %136 = ascvec.arange %135 : i16 -> <ui16>
                        %137 = ascvec.cmp_eq %134, %127, %mask : <ui16>, <ui16> -> !ascvec.mask
                        %138 = ascvec.squeeze_store_reg %136, %137 : <ui16>, !ascvec.mask -> <ui16>
                        ascvec.store_unalign %arg38[%c22560], %138 : memref<?xui16, 1>, <ui16>
                      }
                      ascvec.store_unalign_post %arg38[%c22560] : memref<?xui16, 1>
                      %128 = ascvec.loadalign_brc %arg40[%c33600] : memref<?xui16, 1>, index -> <ui16>
                      %129 = ascvec.loadalign_brc %arg41[%c34112] : memref<?xui16, 1>, index -> <ui16>
                      %130 = ascvec.shl %128, 8, %mask : <ui16> -> <ui16>
                      %131 = ascvec.add %130, %129, %mask : <ui16>
                      ascvec.store_unalign_begin %arg42 : memref<?xui16, 1>
                      scf.for %arg43 = %c4_i32 to %c36_i32 step %c1_i32_5  : i32 {
                        %132 = arith.muli %arg43, %c128_i32_6 overflow<nsw> : i32
                        %133 = arith.addi %132, %c4608_i32 overflow<nsw> : i32
                        %134 = arith.index_cast %133 : i32 to index
                        %135 = ascvec.loadalign %arg39[%134] : memref<?xui16, 1>, index -> <ui16>
                        %136 = arith.trunci %132 : i32 to i16
                        %137 = ascvec.arange %136 : i16 -> <ui16>
                        %138 = ascvec.cmp_gt %135, %131, %mask : <ui16>, <ui16> -> !ascvec.mask
                        %139 = ascvec.squeeze_store_reg %137, %138 : <ui16>, !ascvec.mask -> <ui16>
                        ascvec.store_unalign %arg42[%c27184], %139 : memref<?xui16, 1>, <ui16>
                      }
                      scf.for %arg43 = %c4_i32 to %c36_i32 step %c1_i32_5  : i32 {
                        %132 = arith.muli %arg43, %c128_i32_6 overflow<nsw> : i32
                        %133 = arith.addi %132, %c4608_i32 overflow<nsw> : i32
                        %134 = arith.index_cast %133 : i32 to index
                        %135 = ascvec.loadalign %arg39[%134] : memref<?xui16, 1>, index -> <ui16>
                        %136 = arith.trunci %132 : i32 to i16
                        %137 = ascvec.arange %136 : i16 -> <ui16>
                        %138 = ascvec.cmp_eq %135, %131, %mask : <ui16>, <ui16> -> !ascvec.mask
                        %139 = ascvec.squeeze_store_reg %137, %138 : <ui16>, !ascvec.mask -> <ui16>
                        ascvec.store_unalign %arg42[%c27184], %139 : memref<?xui16, 1>, <ui16>
                      }
                      %c0_7 = arith.constant 0 : index
                      %c1 = arith.constant 1 : index
                      scf.for %arg43 = %c0_7 to %c1 step %c1 {
                        ascvec.store_unalign_post %arg42[%c27184] : memref<?xui16, 1>
                      }
                    } {cannir.vf_mode = "simd"}
                    %121 = arith.muli %104, %c1024_i64 overflow<nsw> : i64
                    %122 = arith.index_cast %121 : i64 to index
                    %123 = arith.addi %122, %c4608 : index
                    ascvec.vf(%87, %120, %1, %123, %120 : memref<?xi32, 1>, memref<?xui16, 1>, memref<?xui32, 1>, index, memref<?xui16, 1>) {
                    ^bb0(%arg36: memref<?xi32, 1>, %arg37: memref<?xui16, 1>, %arg38: memref<?xui32, 1>, %arg39: index, %arg40: memref<?xui16, 1>):
                      %c17336 = arith.constant 17336 : index
                      %c27184 = arith.constant 27184 : index
                      %c1_i32_5 = arith.constant 1 : i32
                      %c8_i32_6 = arith.constant 8 : i32
                      %c22560 = arith.constant 22560 : index
                      %c512_i32_7 = arith.constant 512 : i32
                      %c0_i32_8 = arith.constant 0 : i32
                      %c17328 = arith.constant 17328 : index
                      %c64_i32_9 = arith.constant 64 : i32
                      %mask, %remaining_out = ascvec.update_mask %c64_i32_9 : !ascvec.mask
                      %124 = ascvec.loadalign_brc %arg36[%c17328] : memref<?xi32, 1>, index -> <i32>
                      %125 = ascvec.loadalign_brc %arg36[%c17336] : memref<?xi32, 1>, index -> <i32>
                      %126 = ascvec.duplicate_scalar %c0_i32_8, %mask : i32 -> <ui32>
                      %127 = ascvec.duplicate_scalar %c512_i32_7, %mask : i32 -> <ui32>
                      %128 = ascvec.reinterpret %124 : <i32> -> <ui32>
                      scf.for %arg41 = %c0_i32_8 to %c8_i32_6 step %c1_i32_5  : i32 {
                        %130 = arith.muli %arg41, %c64_i32_9 overflow<nsw> : i32
                        %131 = ascvec.arange %130 : i32 -> <ui32>
                        %132 = ascvec.cmp_lt %131, %128, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %133 = arith.index_cast %130 : i32 to index
                        %134 = arith.addi %133, %c22560 : index
                        %135 = ascvec.loadalign_unpack %arg37[%134] {mode = "b16_to_b32"} : memref<?xui16, 1>, index -> <i32>
                        %136 = ascvec.reinterpret %135 : <i32> -> <ui32>
                        %137 = ascvec.select %136, %126, %132 : <ui32>
                        %138 = ascvec.add %137, %126, %mask : <ui32>
                        %139 = ascvec.sub %138, %127, %mask : <ui32>
                        %140 = ascvec.select %139, %126, %132 : <ui32>
                        %141 = arith.addi %arg39, %133 : index
                        ascvec.storealign %arg38[%141], %140, %mask : memref<?xui32, 1>, index, <ui32>
                      }
                      %129 = ascvec.reinterpret %125 : <i32> -> <ui32>
                      scf.for %arg41 = %c0_i32_8 to %c8_i32_6 step %c1_i32_5  : i32 {
                        %130 = arith.muli %arg41, %c64_i32_9 overflow<nsw> : i32
                        %131 = ascvec.arange %130 : i32 -> <ui32>
                        %132 = ascvec.cmp_lt %131, %129, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %133 = arith.index_cast %130 : i32 to index
                        %134 = arith.addi %133, %c27184 : index
                        %135 = ascvec.loadalign_unpack %arg40[%134] {mode = "b16_to_b32"} : memref<?xui16, 1>, index -> <i32>
                        %136 = ascvec.reinterpret %135 : <i32> -> <ui32>
                        %137 = ascvec.select %136, %126, %132 : <ui32>
                        %138 = ascvec.add %137, %126, %mask : <ui32>
                        %139 = ascvec.sub %138, %127, %mask : <ui32>
                        %140 = arith.addi %130, %c512_i32_7 overflow<nsw> : i32
                        %141 = ascvec.select %139, %126, %132 : <ui32>
                        %142 = arith.index_cast %140 : i32 to index
                        %143 = arith.addi %arg39, %142 : index
                        ascvec.storealign %arg38[%143], %141, %mask : memref<?xui32, 1>, index, <ui32>
                      }
                    } {cannir.vf_mode = "simd"}
                  } else {
                    %120 = arith.remsi %118, %c256_i64 : i64
                    %121 = arith.cmpi ne, %120, %c0_i64 : i64
                    %122 = arith.cmpi slt, %120, %c0_i64 : i64
                    %123 = arith.andi %121, %122 : i1
                    %124 = arith.divsi %118, %c256_i64 : i64
                    %125 = arith.extui %123 : i1 to i64
                    %126 = arith.subi %124, %125 : i64
                    %127 = arith.remsi %118, %c128_i64 : i64
                    %128 = arith.cmpi ne, %127, %c0_i64 : i64
                    %129 = arith.cmpi slt, %127, %c0_i64 : i64
                    %130 = arith.andi %128, %129 : i1
                    %131 = arith.divsi %118, %c128_i64 : i64
                    %132 = arith.extui %130 : i1 to i64
                    %133 = arith.subi %131, %132 : i64
                    %134 = arith.addi %118, %c1_i64 : i64
                    %135 = arith.trunci %134 : i64 to i32
                    %136 = ascscalar.bitcast_memref %alloc_4 byte_offset = %c0_i64 : memref<262144xui8, 1> -> memref<?xui16, 1>
                    ascvec.vf(%87, %135, %136, %136, %136, %136 : memref<?xi32, 1>, i32, memref<?xui16, 1>, memref<?xui16, 1>, memref<?xui16, 1>, memref<?xui16, 1>) {
                    ^bb0(%arg36: memref<?xi32, 1>, %arg37: i32, %arg38: memref<?xui16, 1>, %arg39: memref<?xui16, 1>, %arg40: memref<?xui16, 1>, %arg41: memref<?xui16, 1>):
                      %c34496 = arith.constant 34496 : index
                      %c17336 = arith.constant 17336 : index
                      %c256_i16 = arith.constant 256 : i16
                      %c34368 = arith.constant 34368 : index
                      %c256_i32_5 = arith.constant 256 : i32
                      %c33600 = arith.constant 33600 : index
                      %c2_i32 = arith.constant 2 : i32
                      %c0_i32_6 = arith.constant 0 : i32
                      %c32832 = arith.constant 32832 : index
                      %c33344 = arith.constant 33344 : index
                      %c1_i32_7 = arith.constant 1 : i32
                      %c17328 = arith.constant 17328 : index
                      %c1_i16 = arith.constant 1 : i16
                      %c0_i16 = arith.constant 0 : i16
                      %c64_i32_8 = arith.constant 64 : i32
                      %c128_i32_9 = arith.constant 128 : i32
                      %mask, %remaining_out = ascvec.update_mask %c128_i32_9 elem_bits 16 : !ascvec.mask
                      %mask_10, %remaining_out_11 = ascvec.update_mask %c64_i32_8 : !ascvec.mask
                      %144 = ascvec.duplicate_scalar %c0_i16, %mask : i16 -> <ui16>
                      %145 = ascvec.duplicate_scalar %c1_i16, %mask : i16 -> <ui16>
                      %146 = ascvec.loadalign_brc %arg36[%c17328] : memref<?xi32, 1>, index -> <i32>
                      %147 = ascvec.loadalign_brc %arg36[%c17336] : memref<?xi32, 1>, index -> <i32>
                      %148 = ascvec.max_scalar %146, %c1_i32_7, %mask_10 : <i32>, i32
                      %149 = ascvec.duplicate_scalar %arg37, %mask_10 : i32 -> <i32>
                      %150 = ascvec.sub %149, %148, %mask_10 : <i32>
                      %151 = ascvec.reinterpret %150 : <i32> -> <ui16>
                      %152 = ascvec.duplicate %151, %mask : <ui16> -> <ui16>
                      ascvec.store_unalign_begin %arg38 : memref<?xui16, 1>
                      scf.for %arg42 = %c0_i32_6 to %c2_i32 step %c1_i32_7  : i32 {
                        %157 = arith.muli %arg42, %c128_i32_9 overflow<nsw> : i32
                        %158 = arith.trunci %157 : i32 to i16
                        %159 = ascvec.arange %158 : i16 -> <ui16>
                        %160 = arith.index_cast %157 : i32 to index
                        %161 = arith.addi %160, %c32832 : index
                        %162 = ascvec.loadalign %arg39[%161] : memref<?xui16, 1>, index -> <ui16>
                        %163 = ascvec.cmp_ge %162, %152, %mask : <ui16>, <ui16> -> !ascvec.mask
                        %164 = ascvec.squeeze_store_reg %159, %163 : <ui16>, !ascvec.mask -> <ui16>
                        ascvec.store_unalign %arg38[%c33344], %164 : memref<?xui16, 1>, <ui16>
                      }
                      ascvec.store_unalign_post %arg38[%c33344] : memref<?xui16, 1>
                      %153 = ascvec.max_scalar %147, %c1_i32_7, %mask_10 : <i32>, i32
                      %154 = ascvec.sub %149, %153, %mask_10 : <i32>
                      %155 = ascvec.reinterpret %154 : <i32> -> <ui16>
                      %156 = ascvec.duplicate %155, %mask : <ui16> -> <ui16>
                      ascvec.store_unalign_begin %arg40 : memref<?xui16, 1>
                      scf.for %arg42 = %c0_i32_6 to %c2_i32 step %c1_i32_7  : i32 {
                        %157 = arith.muli %arg42, %c128_i32_9 overflow<nsw> : i32
                        %158 = arith.trunci %157 : i32 to i16
                        %159 = ascvec.arange %158 : i16 -> <ui16>
                        %160 = arith.addi %157, %c256_i32_5 overflow<nsw> : i32
                        %161 = arith.index_cast %160 : i32 to index
                        %162 = arith.addi %161, %c32832 : index
                        %163 = ascvec.loadalign %arg39[%162] : memref<?xui16, 1>, index -> <ui16>
                        %164 = ascvec.cmp_ge %163, %156, %mask : <ui16>, <ui16> -> !ascvec.mask
                        %165 = ascvec.squeeze_store_reg %159, %164 : <ui16>, !ascvec.mask -> <ui16>
                        ascvec.store_unalign %arg40[%c33600], %165 : memref<?xui16, 1>, <ui16>
                      }
                      %c0_12 = arith.constant 0 : index
                      %c1 = arith.constant 1 : index
                      scf.for %arg42 = %c0_12 to %c1 step %c1 {
                        ascvec.store_unalign_post %arg40[%c33600] : memref<?xui16, 1>
                      }
                      ascvec.mem_bar vst_vld
                      %c0_13 = arith.constant 0 : index
                      %c1_14 = arith.constant 1 : index
                      scf.for %arg42 = %c0_13 to %c1_14 step %c1_14 {
                        %157 = ascvec.loadalign_brc %arg38[%c33344] : memref<?xui16, 1>, index -> <ui16>
                        %158 = ascvec.cmp_eq_scalar %157, %c0_i16, %mask : <ui16>, i16 -> !ascvec.mask
                        %159 = ascvec.sub %157, %145, %mask : <ui16>
                        %160 = ascvec.select %144, %159, %158 : <ui16>
                        %161 = ascvec.add %160, %144, %mask : <ui16>
                        %162 = ascvec.gather %arg39[%c32832], %161, %mask : memref<?xui16, 1>, <ui16>, !ascvec.mask -> <ui16>
                        %163 = ascvec.select %144, %162, %158 : <ui16>
                        %164 = ascvec.sub %152, %163, %mask : <ui16>
                        ascvec.storealign %arg41[%c34368], %164, %mask : memref<?xui16, 1>, index, <ui16>
                        %165 = ascvec.loadalign_brc %arg40[%c33600] : memref<?xui16, 1>, index -> <ui16>
                        %166 = ascvec.cmp_eq_scalar %165, %c0_i16, %mask : <ui16>, i16 -> !ascvec.mask
                        %167 = ascvec.sub %165, %145, %mask : <ui16>
                        %168 = ascvec.select %144, %167, %166 : <ui16>
                        %169 = ascvec.duplicate_scalar %c256_i16, %mask : i16 -> <ui16>
                        %170 = ascvec.add %168, %169, %mask : <ui16>
                        %171 = ascvec.gather %arg39[%c32832], %170, %mask : memref<?xui16, 1>, <ui16>, !ascvec.mask -> <ui16>
                        %172 = ascvec.select %144, %171, %166 : <ui16>
                        %173 = ascvec.sub %156, %172, %mask : <ui16>
                        ascvec.storealign %arg41[%c34496], %173, %mask : memref<?xui16, 1>, index, <ui16>
                      }
                    } {cannir.vf_mode = "simd"}
                    ascvec.vf(%136, %136, %126, %136, %136 : memref<?xui16, 1>, memref<?xui16, 1>, i64, memref<?xui16, 1>, memref<?xui16, 1>) {
                    ^bb0(%arg36: memref<?xui16, 1>, %arg37: memref<?xui16, 1>, %arg38: i64, %arg39: memref<?xui16, 1>, %arg40: memref<?xui16, 1>):
                      %c33216 = arith.constant 33216 : index
                      %c33088 = arith.constant 33088 : index
                      %c32960 = arith.constant 32960 : index
                      %c4608_i64_5 = arith.constant 4608 : i64
                      %c33600 = arith.constant 33600 : index
                      %c32832 = arith.constant 32832 : index
                      %c1_i64_6 = arith.constant 1 : i64
                      %c0_i64_7 = arith.constant 0 : i64
                      %c256_i64_8 = arith.constant 256 : i64
                      %c0_i16 = arith.constant 0 : i16
                      %c33344 = arith.constant 33344 : index
                      %c128_i32_9 = arith.constant 128 : i32
                      %c256_i32_10 = arith.constant 256 : i32
                      %mask, %remaining_out = ascvec.update_mask %c256_i32_10 elem_bits 8 : !ascvec.mask
                      %mask_11, %remaining_out_12 = ascvec.update_mask %c128_i32_9 elem_bits 16 : !ascvec.mask
                      %144 = ascvec.loadalign_brc %arg36[%c33344] : memref<?xui16, 1>, index -> <ui16>
                      %145 = ascvec.reinterpret %144 : <ui16> -> <ui8>
                      %146 = ascvec.duplicate %145, %mask : <ui8> -> <ui8>
                      %147 = ascvec.duplicate_scalar %c0_i16, %mask_11 : i16 -> <ui16>
                      %148:2 = scf.for %arg41 = %c0_i64_7 to %arg38 step %c1_i64_6 iter_args(%arg42 = %147, %arg43 = %147) -> (!ascvec.vreg<ui16>, !ascvec.vreg<ui16>)  : i64 {
                        %153 = arith.muli %arg41, %c256_i64_8 : i64
                        %154 = arith.index_cast %153 : i64 to index
                        %result0, %result1 = ascvec.loadalign_deintlv %arg37[%154] {dintlv_width = "b8"} : memref<?xui16, 1>, index -> <ui8>, <ui8>
                        %155 = ascvec.cmp_eq %result1, %146, %mask : <ui8>, <ui8> -> !ascvec.mask
                        %156 = ascvec.cumulative_histogram %arg42, %result0, %155 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                        %157 = ascvec.cumulative_histogram %arg43, %result0, %155 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                        scf.yield %156, %157 : !ascvec.vreg<ui16>, !ascvec.vreg<ui16>
                      }
                      ascvec.storealign %arg39[%c32832], %148#0, %mask_11 : memref<?xui16, 1>, index, <ui16>
                      ascvec.storealign %arg39[%c32960], %148#1, %mask_11 : memref<?xui16, 1>, index, <ui16>
                      %149 = ascvec.loadalign_brc %arg40[%c33600] : memref<?xui16, 1>, index -> <ui16>
                      %150 = ascvec.reinterpret %149 : <ui16> -> <ui8>
                      %151 = ascvec.duplicate %150, %mask : <ui8> -> <ui8>
                      %152:2 = scf.for %arg41 = %c0_i64_7 to %arg38 step %c1_i64_6 iter_args(%arg42 = %147, %arg43 = %147) -> (!ascvec.vreg<ui16>, !ascvec.vreg<ui16>)  : i64 {
                        %153 = arith.muli %arg41, %c256_i64_8 : i64
                        %154 = arith.addi %153, %c4608_i64_5 : i64
                        %155 = arith.index_cast %154 : i64 to index
                        %result0, %result1 = ascvec.loadalign_deintlv %arg37[%155] {dintlv_width = "b8"} : memref<?xui16, 1>, index -> <ui8>, <ui8>
                        %156 = ascvec.cmp_eq %result1, %151, %mask : <ui8>, <ui8> -> !ascvec.mask
                        %157 = ascvec.cumulative_histogram %arg42, %result0, %156 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                        %158 = ascvec.cumulative_histogram %arg43, %result0, %156 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                        scf.yield %157, %158 : !ascvec.vreg<ui16>, !ascvec.vreg<ui16>
                      }
                      %c0_13 = arith.constant 0 : index
                      %c1 = arith.constant 1 : index
                      scf.for %arg41 = %c0_13 to %c1 step %c1 {
                        ascvec.storealign %arg39[%c33088], %152#0, %mask_11 : memref<?xui16, 1>, index, <ui16>
                        ascvec.storealign %arg39[%c33216], %152#1, %mask_11 : memref<?xui16, 1>, index, <ui16>
                      }
                    } {cannir.vf_mode = "simd"}
                    ascvec.vf(%136, %136, %136, %136 : memref<?xui16, 1>, memref<?xui16, 1>, memref<?xui16, 1>, memref<?xui16, 1>) {
                    ^bb0(%arg36: memref<?xui16, 1>, %arg37: memref<?xui16, 1>, %arg38: memref<?xui16, 1>, %arg39: memref<?xui16, 1>):
                      %c34496 = arith.constant 34496 : index
                      %c256_i32_5 = arith.constant 256 : i32
                      %c34112 = arith.constant 34112 : index
                      %c1_i32_6 = arith.constant 1 : i32
                      %c2_i32 = arith.constant 2 : i32
                      %c0_i32_7 = arith.constant 0 : i32
                      %c32832 = arith.constant 32832 : index
                      %c33856 = arith.constant 33856 : index
                      %c34368 = arith.constant 34368 : index
                      %c128_i32_8 = arith.constant 128 : i32
                      %mask, %remaining_out = ascvec.update_mask %c128_i32_8 elem_bits 16 : !ascvec.mask
                      %144 = ascvec.loadalign_brc %arg36[%c34368] : memref<?xui16, 1>, index -> <ui16>
                      ascvec.store_unalign_begin %arg37 : memref<?xui16, 1>
                      scf.for %arg40 = %c0_i32_7 to %c2_i32 step %c1_i32_6  : i32 {
                        %146 = arith.muli %arg40, %c128_i32_8 overflow<nsw> : i32
                        %147 = arith.trunci %146 : i32 to i16
                        %148 = ascvec.arange %147 : i16 -> <ui16>
                        %149 = arith.index_cast %146 : i32 to index
                        %150 = arith.addi %149, %c32832 : index
                        %151 = ascvec.loadalign %arg38[%150] : memref<?xui16, 1>, index -> <ui16>
                        %152 = ascvec.cmp_ge %151, %144, %mask : <ui16>, <ui16> -> !ascvec.mask
                        %153 = ascvec.squeeze_store_reg %148, %152 : <ui16>, !ascvec.mask -> <ui16>
                        ascvec.store_unalign %arg37[%c33856], %153 : memref<?xui16, 1>, <ui16>
                      }
                      ascvec.store_unalign_post %arg37[%c33856] : memref<?xui16, 1>
                      %145 = ascvec.loadalign_brc %arg36[%c34496] : memref<?xui16, 1>, index -> <ui16>
                      ascvec.store_unalign_begin %arg39 : memref<?xui16, 1>
                      scf.for %arg40 = %c0_i32_7 to %c2_i32 step %c1_i32_6  : i32 {
                        %146 = arith.muli %arg40, %c128_i32_8 overflow<nsw> : i32
                        %147 = arith.trunci %146 : i32 to i16
                        %148 = ascvec.arange %147 : i16 -> <ui16>
                        %149 = arith.addi %146, %c256_i32_5 overflow<nsw> : i32
                        %150 = arith.index_cast %149 : i32 to index
                        %151 = arith.addi %150, %c32832 : index
                        %152 = ascvec.loadalign %arg38[%151] : memref<?xui16, 1>, index -> <ui16>
                        %153 = ascvec.cmp_ge %152, %145, %mask : <ui16>, <ui16> -> !ascvec.mask
                        %154 = ascvec.squeeze_store_reg %148, %153 : <ui16>, !ascvec.mask -> <ui16>
                        ascvec.store_unalign %arg39[%c34112], %154 : memref<?xui16, 1>, <ui16>
                      }
                      %c0_9 = arith.constant 0 : index
                      %c1 = arith.constant 1 : index
                      scf.for %arg40 = %c0_9 to %c1 step %c1 {
                        ascvec.store_unalign_post %arg39[%c34112] : memref<?xui16, 1>
                      }
                    } {cannir.vf_mode = "simd"}
                    ascvec.vf(%136, %136, %136, %136, %133, %136, %136, %136 : memref<?xui16, 1>, memref<?xui16, 1>, memref<?xui16, 1>, memref<?xui16, 1>, i64, memref<?xui16, 1>, memref<?xui16, 1>, memref<?xui16, 1>) {
                    ^bb0(%arg36: memref<?xui16, 1>, %arg37: memref<?xui16, 1>, %arg38: memref<?xui16, 1>, %arg39: memref<?xui16, 1>, %arg40: i64, %arg41: memref<?xui16, 1>, %arg42: memref<?xui16, 1>, %arg43: memref<?xui16, 1>):
                      %c4608_i64_5 = arith.constant 4608 : i64
                      %c27184 = arith.constant 27184 : index
                      %c34112 = arith.constant 34112 : index
                      %c33600 = arith.constant 33600 : index
                      %c1_i64_6 = arith.constant 1 : i64
                      %c0_i64_7 = arith.constant 0 : i64
                      %c128_i64_8 = arith.constant 128 : i64
                      %c22560 = arith.constant 22560 : index
                      %c33856 = arith.constant 33856 : index
                      %c33344 = arith.constant 33344 : index
                      %c128_i32_9 = arith.constant 128 : i32
                      %mask, %remaining_out = ascvec.update_mask %c128_i32_9 elem_bits 16 : !ascvec.mask
                      %144 = ascvec.loadalign_brc %arg36[%c33344] : memref<?xui16, 1>, index -> <ui16>
                      %145 = ascvec.loadalign_brc %arg37[%c33856] : memref<?xui16, 1>, index -> <ui16>
                      %146 = ascvec.shl %144, 8, %mask : <ui16> -> <ui16>
                      %147 = ascvec.add %146, %145, %mask : <ui16>
                      ascvec.store_unalign_begin %arg38 : memref<?xui16, 1>
                      scf.for %arg44 = %c0_i64_7 to %arg40 step %c1_i64_6  : i64 {
                        %152 = arith.muli %arg44, %c128_i64_8 : i64
                        %153 = arith.index_cast %152 : i64 to index
                        %154 = ascvec.loadalign %arg39[%153] : memref<?xui16, 1>, index -> <ui16>
                        %155 = arith.trunci %152 : i64 to i16
                        %156 = ascvec.arange %155 : i16 -> <ui16>
                        %157 = ascvec.cmp_gt %154, %147, %mask : <ui16>, <ui16> -> !ascvec.mask
                        %158 = ascvec.squeeze_store_reg %156, %157 : <ui16>, !ascvec.mask -> <ui16>
                        ascvec.store_unalign %arg38[%c22560], %158 : memref<?xui16, 1>, <ui16>
                      }
                      scf.for %arg44 = %c0_i64_7 to %arg40 step %c1_i64_6  : i64 {
                        %152 = arith.muli %arg44, %c128_i64_8 : i64
                        %153 = arith.index_cast %152 : i64 to index
                        %154 = ascvec.loadalign %arg39[%153] : memref<?xui16, 1>, index -> <ui16>
                        %155 = arith.trunci %152 : i64 to i16
                        %156 = ascvec.arange %155 : i16 -> <ui16>
                        %157 = ascvec.cmp_eq %154, %147, %mask : <ui16>, <ui16> -> !ascvec.mask
                        %158 = ascvec.squeeze_store_reg %156, %157 : <ui16>, !ascvec.mask -> <ui16>
                        ascvec.store_unalign %arg38[%c22560], %158 : memref<?xui16, 1>, <ui16>
                      }
                      ascvec.store_unalign_post %arg38[%c22560] : memref<?xui16, 1>
                      %148 = ascvec.loadalign_brc %arg41[%c33600] : memref<?xui16, 1>, index -> <ui16>
                      %149 = ascvec.loadalign_brc %arg42[%c34112] : memref<?xui16, 1>, index -> <ui16>
                      %150 = ascvec.shl %148, 8, %mask : <ui16> -> <ui16>
                      %151 = ascvec.add %150, %149, %mask : <ui16>
                      ascvec.store_unalign_begin %arg43 : memref<?xui16, 1>
                      scf.for %arg44 = %c0_i64_7 to %arg40 step %c1_i64_6  : i64 {
                        %152 = arith.muli %arg44, %c128_i64_8 : i64
                        %153 = arith.addi %152, %c4608_i64_5 : i64
                        %154 = arith.index_cast %153 : i64 to index
                        %155 = ascvec.loadalign %arg39[%154] : memref<?xui16, 1>, index -> <ui16>
                        %156 = arith.trunci %152 : i64 to i16
                        %157 = ascvec.arange %156 : i16 -> <ui16>
                        %158 = ascvec.cmp_gt %155, %151, %mask : <ui16>, <ui16> -> !ascvec.mask
                        %159 = ascvec.squeeze_store_reg %157, %158 : <ui16>, !ascvec.mask -> <ui16>
                        ascvec.store_unalign %arg43[%c27184], %159 : memref<?xui16, 1>, <ui16>
                      }
                      scf.for %arg44 = %c0_i64_7 to %arg40 step %c1_i64_6  : i64 {
                        %152 = arith.muli %arg44, %c128_i64_8 : i64
                        %153 = arith.addi %152, %c4608_i64_5 : i64
                        %154 = arith.index_cast %153 : i64 to index
                        %155 = ascvec.loadalign %arg39[%154] : memref<?xui16, 1>, index -> <ui16>
                        %156 = arith.trunci %152 : i64 to i16
                        %157 = ascvec.arange %156 : i16 -> <ui16>
                        %158 = ascvec.cmp_eq %155, %151, %mask : <ui16>, <ui16> -> !ascvec.mask
                        %159 = ascvec.squeeze_store_reg %157, %158 : <ui16>, !ascvec.mask -> <ui16>
                        ascvec.store_unalign %arg43[%c27184], %159 : memref<?xui16, 1>, <ui16>
                      }
                      %c0_10 = arith.constant 0 : index
                      %c1 = arith.constant 1 : index
                      scf.for %arg44 = %c0_10 to %c1 step %c1 {
                        ascvec.store_unalign_post %arg43[%c27184] : memref<?xui16, 1>
                      }
                    } {cannir.vf_mode = "simd"}
                    %137 = arith.trunci %116 : i64 to i32
                    %138 = arith.muli %97, %c1024_i64 overflow<nsw> : i64
                    %139 = arith.index_cast %138 : i64 to index
                    %140 = arith.addi %139, %c4608 : index
                    %141 = arith.muli %104, %c1024_i64 overflow<nsw> : i64
                    %142 = arith.index_cast %141 : i64 to index
                    %143 = arith.addi %142, %c4608 : index
                    ascvec.vf(%87, %137, %136, %1, %140, %136, %1, %143, %136 : memref<?xi32, 1>, i32, memref<?xui16, 1>, memref<?xui32, 1>, index, memref<?xui16, 1>, memref<?xui32, 1>, index, memref<?xui16, 1>) {
                    ^bb0(%arg36: memref<?xi32, 1>, %arg37: i32, %arg38: memref<?xui16, 1>, %arg39: memref<?xui32, 1>, %arg40: index, %arg41: memref<?xui16, 1>, %arg42: memref<?xui32, 1>, %arg43: index, %arg44: memref<?xui16, 1>):
                      %c17336 = arith.constant 17336 : index
                      %c4624_i16 = arith.constant 4624 : i16
                      %c27184 = arith.constant 27184 : index
                      %c1_i32_5 = arith.constant 1 : i32
                      %c8_i32_6 = arith.constant 8 : i32
                      %c13312 = arith.constant 13312 : index
                      %c0_i16 = arith.constant 0 : i16
                      %c22560 = arith.constant 22560 : index
                      %c512_i32_7 = arith.constant 512 : i32
                      %c0_i32_8 = arith.constant 0 : i32
                      %c17328 = arith.constant 17328 : index
                      %c64_i32_9 = arith.constant 64 : i32
                      %mask, %remaining_out = ascvec.update_mask %c64_i32_9 : !ascvec.mask
                      %144 = ascvec.loadalign_brc %arg36[%c17328] : memref<?xi32, 1>, index -> <i32>
                      %145 = ascvec.loadalign_brc %arg36[%c17336] : memref<?xi32, 1>, index -> <i32>
                      %146 = ascvec.duplicate_scalar %c0_i32_8, %mask : i32 -> <ui32>
                      %147 = ascvec.duplicate_scalar %c512_i32_7, %mask : i32 -> <ui32>
                      %148 = ascvec.duplicate_scalar %arg37, %mask : i32 -> <ui32>
                      %149 = ascvec.reinterpret %144 : <i32> -> <ui32>
                      scf.for %arg45 = %c0_i32_8 to %c8_i32_6 step %c1_i32_5  : i32 {
                        %151 = arith.muli %arg45, %c64_i32_9 overflow<nsw> : i32
                        %152 = ascvec.arange %151 : i32 -> <ui32>
                        %153 = ascvec.cmp_lt %152, %149, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %154 = arith.index_cast %151 : i32 to index
                        %155 = arith.addi %154, %c22560 : index
                        %156 = ascvec.loadalign_unpack %arg38[%155] {mode = "b16_to_b32"} : memref<?xui16, 1>, index -> <i32>
                        %157 = ascvec.reinterpret %156 : <i32> -> <ui32>
                        %158 = ascvec.select %157, %146, %153 : <ui32>
                        %159 = ascvec.cmp_lt %158, %147, %153 : <ui32>, <ui32> -> !ascvec.mask
                        %160 = ascvec.select %158, %146, %159 : <ui32>
                        %161 = ascvec.add %160, %146, %mask : <ui32>
                        %162 = ascvec.gather %arg39[%arg40], %161, %mask : memref<?xui32, 1>, <ui32>, !ascvec.mask -> <ui32>
                        %163 = ascvec.sub %158, %147, %mask : <ui32>
                        %164 = ascvec.select %146, %163, %159 : <ui32>
                        %165 = ascvec.select %164, %146, %153 : <ui32>
                        %166 = ascvec.pack %165 src_elem_bits 32 dst_elem_bits 16 {part = "low"} : <ui32> -> <ui16>
                        %mask_10, %remaining_out_11 = ascvec.update_mask %c64_i32_9 elem_bits 16 : !ascvec.mask
                        %167 = ascvec.add_scalar %166, %c0_i16, %mask_10 : <ui16>, i16
                        %168 = ascvec.gather %arg41[%c13312], %167, %mask_10 : memref<?xui16, 1>, <ui16>, !ascvec.mask -> <ui16>
                        %169 = ascvec.unpack %168 {part = "lower"} : <ui16> -> <ui32>
                        %170 = ascvec.add %169, %148, %mask : <ui32>
                        %171 = ascvec.select %162, %170, %159 : <ui32>
                        %172 = ascvec.select %171, %146, %153 : <ui32>
                        %173 = arith.addi %arg43, %154 : index
                        ascvec.storealign %arg42[%173], %172, %mask : memref<?xui32, 1>, index, <ui32>
                      }
                      %150 = ascvec.reinterpret %145 : <i32> -> <ui32>
                      scf.for %arg45 = %c0_i32_8 to %c8_i32_6 step %c1_i32_5  : i32 {
                        %151 = arith.muli %arg45, %c64_i32_9 overflow<nsw> : i32
                        %152 = ascvec.arange %151 : i32 -> <ui32>
                        %153 = ascvec.cmp_lt %152, %150, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %154 = arith.index_cast %151 : i32 to index
                        %155 = arith.addi %154, %c27184 : index
                        %156 = ascvec.loadalign_unpack %arg44[%155] {mode = "b16_to_b32"} : memref<?xui16, 1>, index -> <i32>
                        %157 = ascvec.reinterpret %156 : <i32> -> <ui32>
                        %158 = ascvec.select %157, %146, %153 : <ui32>
                        %159 = ascvec.cmp_lt %158, %147, %153 : <ui32>, <ui32> -> !ascvec.mask
                        %160 = ascvec.select %158, %146, %159 : <ui32>
                        %161 = ascvec.add %160, %147, %mask : <ui32>
                        %162 = ascvec.gather %arg39[%arg40], %161, %mask : memref<?xui32, 1>, <ui32>, !ascvec.mask -> <ui32>
                        %163 = ascvec.sub %158, %147, %mask : <ui32>
                        %164 = ascvec.select %146, %163, %159 : <ui32>
                        %165 = ascvec.select %164, %146, %153 : <ui32>
                        %166 = ascvec.pack %165 src_elem_bits 32 dst_elem_bits 16 {part = "low"} : <ui32> -> <ui16>
                        %mask_10, %remaining_out_11 = ascvec.update_mask %c64_i32_9 elem_bits 16 : !ascvec.mask
                        %167 = ascvec.add_scalar %166, %c4624_i16, %mask_10 : <ui16>, i16
                        %168 = ascvec.gather %arg41[%c13312], %167, %mask_10 : memref<?xui16, 1>, <ui16>, !ascvec.mask -> <ui16>
                        %169 = ascvec.unpack %168 {part = "lower"} : <ui16> -> <ui32>
                        %170 = ascvec.add %169, %148, %mask : <ui32>
                        %171 = ascvec.select %162, %170, %159 : <ui32>
                        %172 = arith.addi %151, %c512_i32_7 overflow<nsw> : i32
                        %173 = ascvec.select %171, %146, %153 : <ui32>
                        %174 = arith.index_cast %172 : i32 to index
                        %175 = arith.addi %arg43, %174 : index
                        ascvec.storealign %arg42[%175], %173, %mask : memref<?xui32, 1>, index, <ui32>
                      }
                    } {cannir.vf_mode = "simd"}
                  }
                } else {
                  scf.if %117 {
                    %120 = ascscalar.bitcast_memref %alloc_4 byte_offset = %c0_i64 : memref<262144xui8, 1> -> memref<?xui16, 1>
                    ascvec.vf(%87, %120, %120, %120, %120 : memref<?xi32, 1>, memref<?xui16, 1>, memref<?xui16, 1>, memref<?xui16, 1>, memref<?xui16, 1>) {
                    ^bb0(%arg36: memref<?xi32, 1>, %arg37: memref<?xui16, 1>, %arg38: memref<?xui16, 1>, %arg39: memref<?xui16, 1>, %arg40: memref<?xui16, 1>):
                      %c34496 = arith.constant 34496 : index
                      %c17336 = arith.constant 17336 : index
                      %c256_i16 = arith.constant 256 : i16
                      %c34368 = arith.constant 34368 : index
                      %c256_i32_5 = arith.constant 256 : i32
                      %c33600 = arith.constant 33600 : index
                      %c2_i32 = arith.constant 2 : i32
                      %c0_i32_6 = arith.constant 0 : i32
                      %c32832 = arith.constant 32832 : index
                      %c33344 = arith.constant 33344 : index
                      %c4097_i32 = arith.constant 4097 : i32
                      %c1_i32_7 = arith.constant 1 : i32
                      %c17328 = arith.constant 17328 : index
                      %c1_i16 = arith.constant 1 : i16
                      %c0_i16 = arith.constant 0 : i16
                      %c64_i32_8 = arith.constant 64 : i32
                      %c128_i32_9 = arith.constant 128 : i32
                      %mask, %remaining_out = ascvec.update_mask %c128_i32_9 elem_bits 16 : !ascvec.mask
                      %mask_10, %remaining_out_11 = ascvec.update_mask %c64_i32_8 : !ascvec.mask
                      %125 = ascvec.duplicate_scalar %c0_i16, %mask : i16 -> <ui16>
                      %126 = ascvec.duplicate_scalar %c1_i16, %mask : i16 -> <ui16>
                      %127 = ascvec.loadalign_brc %arg36[%c17328] : memref<?xi32, 1>, index -> <i32>
                      %128 = ascvec.loadalign_brc %arg36[%c17336] : memref<?xi32, 1>, index -> <i32>
                      %129 = ascvec.max_scalar %127, %c1_i32_7, %mask_10 : <i32>, i32
                      %130 = ascvec.duplicate_scalar %c4097_i32, %mask_10 : i32 -> <i32>
                      %131 = ascvec.sub %130, %129, %mask_10 : <i32>
                      %132 = ascvec.reinterpret %131 : <i32> -> <ui16>
                      %133 = ascvec.duplicate %132, %mask : <ui16> -> <ui16>
                      ascvec.store_unalign_begin %arg37 : memref<?xui16, 1>
                      scf.for %arg41 = %c0_i32_6 to %c2_i32 step %c1_i32_7  : i32 {
                        %138 = arith.muli %arg41, %c128_i32_9 overflow<nsw> : i32
                        %139 = arith.trunci %138 : i32 to i16
                        %140 = ascvec.arange %139 : i16 -> <ui16>
                        %141 = arith.index_cast %138 : i32 to index
                        %142 = arith.addi %141, %c32832 : index
                        %143 = ascvec.loadalign %arg38[%142] : memref<?xui16, 1>, index -> <ui16>
                        %144 = ascvec.cmp_ge %143, %133, %mask : <ui16>, <ui16> -> !ascvec.mask
                        %145 = ascvec.squeeze_store_reg %140, %144 : <ui16>, !ascvec.mask -> <ui16>
                        ascvec.store_unalign %arg37[%c33344], %145 : memref<?xui16, 1>, <ui16>
                      }
                      ascvec.store_unalign_post %arg37[%c33344] : memref<?xui16, 1>
                      %134 = ascvec.max_scalar %128, %c1_i32_7, %mask_10 : <i32>, i32
                      %135 = ascvec.sub %130, %134, %mask_10 : <i32>
                      %136 = ascvec.reinterpret %135 : <i32> -> <ui16>
                      %137 = ascvec.duplicate %136, %mask : <ui16> -> <ui16>
                      ascvec.store_unalign_begin %arg39 : memref<?xui16, 1>
                      scf.for %arg41 = %c0_i32_6 to %c2_i32 step %c1_i32_7  : i32 {
                        %138 = arith.muli %arg41, %c128_i32_9 overflow<nsw> : i32
                        %139 = arith.trunci %138 : i32 to i16
                        %140 = ascvec.arange %139 : i16 -> <ui16>
                        %141 = arith.addi %138, %c256_i32_5 overflow<nsw> : i32
                        %142 = arith.index_cast %141 : i32 to index
                        %143 = arith.addi %142, %c32832 : index
                        %144 = ascvec.loadalign %arg38[%143] : memref<?xui16, 1>, index -> <ui16>
                        %145 = ascvec.cmp_ge %144, %137, %mask : <ui16>, <ui16> -> !ascvec.mask
                        %146 = ascvec.squeeze_store_reg %140, %145 : <ui16>, !ascvec.mask -> <ui16>
                        ascvec.store_unalign %arg39[%c33600], %146 : memref<?xui16, 1>, <ui16>
                      }
                      %c0_12 = arith.constant 0 : index
                      %c1 = arith.constant 1 : index
                      scf.for %arg41 = %c0_12 to %c1 step %c1 {
                        ascvec.store_unalign_post %arg39[%c33600] : memref<?xui16, 1>
                      }
                      ascvec.mem_bar vst_vld
                      %c0_13 = arith.constant 0 : index
                      %c1_14 = arith.constant 1 : index
                      scf.for %arg41 = %c0_13 to %c1_14 step %c1_14 {
                        %138 = ascvec.loadalign_brc %arg37[%c33344] : memref<?xui16, 1>, index -> <ui16>
                        %139 = ascvec.cmp_eq_scalar %138, %c0_i16, %mask : <ui16>, i16 -> !ascvec.mask
                        %140 = ascvec.sub %138, %126, %mask : <ui16>
                        %141 = ascvec.select %125, %140, %139 : <ui16>
                        %142 = ascvec.add %141, %125, %mask : <ui16>
                        %143 = ascvec.gather %arg38[%c32832], %142, %mask : memref<?xui16, 1>, <ui16>, !ascvec.mask -> <ui16>
                        %144 = ascvec.select %125, %143, %139 : <ui16>
                        %145 = ascvec.sub %133, %144, %mask : <ui16>
                        ascvec.storealign %arg40[%c34368], %145, %mask : memref<?xui16, 1>, index, <ui16>
                        %146 = ascvec.loadalign_brc %arg39[%c33600] : memref<?xui16, 1>, index -> <ui16>
                        %147 = ascvec.cmp_eq_scalar %146, %c0_i16, %mask : <ui16>, i16 -> !ascvec.mask
                        %148 = ascvec.sub %146, %126, %mask : <ui16>
                        %149 = ascvec.select %125, %148, %147 : <ui16>
                        %150 = ascvec.duplicate_scalar %c256_i16, %mask : i16 -> <ui16>
                        %151 = ascvec.add %149, %150, %mask : <ui16>
                        %152 = ascvec.gather %arg38[%c32832], %151, %mask : memref<?xui16, 1>, <ui16>, !ascvec.mask -> <ui16>
                        %153 = ascvec.select %125, %152, %147 : <ui16>
                        %154 = ascvec.sub %137, %153, %mask : <ui16>
                        ascvec.storealign %arg40[%c34496], %154, %mask : memref<?xui16, 1>, index, <ui16>
                      }
                    } {cannir.vf_mode = "simd"}
                    ascvec.vf(%120, %120, %120, %120 : memref<?xui16, 1>, memref<?xui16, 1>, memref<?xui16, 1>, memref<?xui16, 1>) {
                    ^bb0(%arg36: memref<?xui16, 1>, %arg37: memref<?xui16, 1>, %arg38: memref<?xui16, 1>, %arg39: memref<?xui16, 1>):
                      %c33216 = arith.constant 33216 : index
                      %c33088 = arith.constant 33088 : index
                      %c32960 = arith.constant 32960 : index
                      %c4608_i32 = arith.constant 4608 : i32
                      %c33600 = arith.constant 33600 : index
                      %c32832 = arith.constant 32832 : index
                      %c1_i32_5 = arith.constant 1 : i32
                      %c18_i32 = arith.constant 18 : i32
                      %c2_i32 = arith.constant 2 : i32
                      %c0_i16 = arith.constant 0 : i16
                      %c33344 = arith.constant 33344 : index
                      %c128_i32_6 = arith.constant 128 : i32
                      %c256_i32_7 = arith.constant 256 : i32
                      %mask, %remaining_out = ascvec.update_mask %c256_i32_7 elem_bits 8 : !ascvec.mask
                      %mask_8, %remaining_out_9 = ascvec.update_mask %c128_i32_6 elem_bits 16 : !ascvec.mask
                      %125 = ascvec.loadalign_brc %arg36[%c33344] : memref<?xui16, 1>, index -> <ui16>
                      %126 = ascvec.reinterpret %125 : <ui16> -> <ui8>
                      %127 = ascvec.duplicate %126, %mask : <ui8> -> <ui8>
                      %128 = ascvec.duplicate_scalar %c0_i16, %mask_8 : i16 -> <ui16>
                      %129:2 = scf.for %arg40 = %c2_i32 to %c18_i32 step %c1_i32_5 iter_args(%arg41 = %128, %arg42 = %128) -> (!ascvec.vreg<ui16>, !ascvec.vreg<ui16>)  : i32 {
                        %134 = arith.muli %arg40, %c256_i32_7 overflow<nsw> : i32
                        %135 = arith.index_cast %134 : i32 to index
                        %result0, %result1 = ascvec.loadalign_deintlv %arg37[%135] {dintlv_width = "b8"} : memref<?xui16, 1>, index -> <ui8>, <ui8>
                        %136 = ascvec.cmp_eq %result1, %127, %mask : <ui8>, <ui8> -> !ascvec.mask
                        %137 = ascvec.cumulative_histogram %arg41, %result0, %136 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                        %138 = ascvec.cumulative_histogram %arg42, %result0, %136 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                        scf.yield %137, %138 : !ascvec.vreg<ui16>, !ascvec.vreg<ui16>
                      }
                      ascvec.storealign %arg38[%c32832], %129#0, %mask_8 : memref<?xui16, 1>, index, <ui16>
                      ascvec.storealign %arg38[%c32960], %129#1, %mask_8 : memref<?xui16, 1>, index, <ui16>
                      %130 = ascvec.loadalign_brc %arg39[%c33600] : memref<?xui16, 1>, index -> <ui16>
                      %131 = ascvec.reinterpret %130 : <ui16> -> <ui8>
                      %132 = ascvec.duplicate %131, %mask : <ui8> -> <ui8>
                      %133:2 = scf.for %arg40 = %c2_i32 to %c18_i32 step %c1_i32_5 iter_args(%arg41 = %128, %arg42 = %128) -> (!ascvec.vreg<ui16>, !ascvec.vreg<ui16>)  : i32 {
                        %134 = arith.muli %arg40, %c256_i32_7 overflow<nsw> : i32
                        %135 = arith.addi %134, %c4608_i32 overflow<nsw> : i32
                        %136 = arith.index_cast %135 : i32 to index
                        %result0, %result1 = ascvec.loadalign_deintlv %arg37[%136] {dintlv_width = "b8"} : memref<?xui16, 1>, index -> <ui8>, <ui8>
                        %137 = ascvec.cmp_eq %result1, %132, %mask : <ui8>, <ui8> -> !ascvec.mask
                        %138 = ascvec.cumulative_histogram %arg41, %result0, %137 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                        %139 = ascvec.cumulative_histogram %arg42, %result0, %137 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                        scf.yield %138, %139 : !ascvec.vreg<ui16>, !ascvec.vreg<ui16>
                      }
                      %c0_10 = arith.constant 0 : index
                      %c1 = arith.constant 1 : index
                      scf.for %arg40 = %c0_10 to %c1 step %c1 {
                        ascvec.storealign %arg38[%c33088], %133#0, %mask_8 : memref<?xui16, 1>, index, <ui16>
                        ascvec.storealign %arg38[%c33216], %133#1, %mask_8 : memref<?xui16, 1>, index, <ui16>
                      }
                    } {cannir.vf_mode = "simd"}
                    ascvec.vf(%120, %120, %120, %120 : memref<?xui16, 1>, memref<?xui16, 1>, memref<?xui16, 1>, memref<?xui16, 1>) {
                    ^bb0(%arg36: memref<?xui16, 1>, %arg37: memref<?xui16, 1>, %arg38: memref<?xui16, 1>, %arg39: memref<?xui16, 1>):
                      %c34496 = arith.constant 34496 : index
                      %c256_i32_5 = arith.constant 256 : i32
                      %c34112 = arith.constant 34112 : index
                      %c1_i32_6 = arith.constant 1 : i32
                      %c2_i32 = arith.constant 2 : i32
                      %c0_i32_7 = arith.constant 0 : i32
                      %c32832 = arith.constant 32832 : index
                      %c33856 = arith.constant 33856 : index
                      %c34368 = arith.constant 34368 : index
                      %c128_i32_8 = arith.constant 128 : i32
                      %mask, %remaining_out = ascvec.update_mask %c128_i32_8 elem_bits 16 : !ascvec.mask
                      %125 = ascvec.loadalign_brc %arg36[%c34368] : memref<?xui16, 1>, index -> <ui16>
                      ascvec.store_unalign_begin %arg37 : memref<?xui16, 1>
                      scf.for %arg40 = %c0_i32_7 to %c2_i32 step %c1_i32_6  : i32 {
                        %127 = arith.muli %arg40, %c128_i32_8 overflow<nsw> : i32
                        %128 = arith.trunci %127 : i32 to i16
                        %129 = ascvec.arange %128 : i16 -> <ui16>
                        %130 = arith.index_cast %127 : i32 to index
                        %131 = arith.addi %130, %c32832 : index
                        %132 = ascvec.loadalign %arg38[%131] : memref<?xui16, 1>, index -> <ui16>
                        %133 = ascvec.cmp_ge %132, %125, %mask : <ui16>, <ui16> -> !ascvec.mask
                        %134 = ascvec.squeeze_store_reg %129, %133 : <ui16>, !ascvec.mask -> <ui16>
                        ascvec.store_unalign %arg37[%c33856], %134 : memref<?xui16, 1>, <ui16>
                      }
                      ascvec.store_unalign_post %arg37[%c33856] : memref<?xui16, 1>
                      %126 = ascvec.loadalign_brc %arg36[%c34496] : memref<?xui16, 1>, index -> <ui16>
                      ascvec.store_unalign_begin %arg39 : memref<?xui16, 1>
                      scf.for %arg40 = %c0_i32_7 to %c2_i32 step %c1_i32_6  : i32 {
                        %127 = arith.muli %arg40, %c128_i32_8 overflow<nsw> : i32
                        %128 = arith.trunci %127 : i32 to i16
                        %129 = ascvec.arange %128 : i16 -> <ui16>
                        %130 = arith.addi %127, %c256_i32_5 overflow<nsw> : i32
                        %131 = arith.index_cast %130 : i32 to index
                        %132 = arith.addi %131, %c32832 : index
                        %133 = ascvec.loadalign %arg38[%132] : memref<?xui16, 1>, index -> <ui16>
                        %134 = ascvec.cmp_ge %133, %126, %mask : <ui16>, <ui16> -> !ascvec.mask
                        %135 = ascvec.squeeze_store_reg %129, %134 : <ui16>, !ascvec.mask -> <ui16>
                        ascvec.store_unalign %arg39[%c34112], %135 : memref<?xui16, 1>, <ui16>
                      }
                      %c0_9 = arith.constant 0 : index
                      %c1 = arith.constant 1 : index
                      scf.for %arg40 = %c0_9 to %c1 step %c1 {
                        ascvec.store_unalign_post %arg39[%c34112] : memref<?xui16, 1>
                      }
                    } {cannir.vf_mode = "simd"}
                    ascvec.vf(%120, %120, %120, %120, %120, %120, %120 : memref<?xui16, 1>, memref<?xui16, 1>, memref<?xui16, 1>, memref<?xui16, 1>, memref<?xui16, 1>, memref<?xui16, 1>, memref<?xui16, 1>) {
                    ^bb0(%arg36: memref<?xui16, 1>, %arg37: memref<?xui16, 1>, %arg38: memref<?xui16, 1>, %arg39: memref<?xui16, 1>, %arg40: memref<?xui16, 1>, %arg41: memref<?xui16, 1>, %arg42: memref<?xui16, 1>):
                      %c4608_i32 = arith.constant 4608 : i32
                      %c27184 = arith.constant 27184 : index
                      %c34112 = arith.constant 34112 : index
                      %c33600 = arith.constant 33600 : index
                      %c1_i32_5 = arith.constant 1 : i32
                      %c36_i32 = arith.constant 36 : i32
                      %c4_i32 = arith.constant 4 : i32
                      %c22560 = arith.constant 22560 : index
                      %c33856 = arith.constant 33856 : index
                      %c33344 = arith.constant 33344 : index
                      %c128_i32_6 = arith.constant 128 : i32
                      %mask, %remaining_out = ascvec.update_mask %c128_i32_6 elem_bits 16 : !ascvec.mask
                      %125 = ascvec.loadalign_brc %arg36[%c33344] : memref<?xui16, 1>, index -> <ui16>
                      %126 = ascvec.loadalign_brc %arg37[%c33856] : memref<?xui16, 1>, index -> <ui16>
                      %127 = ascvec.shl %125, 8, %mask : <ui16> -> <ui16>
                      %128 = ascvec.add %127, %126, %mask : <ui16>
                      ascvec.store_unalign_begin %arg38 : memref<?xui16, 1>
                      scf.for %arg43 = %c4_i32 to %c36_i32 step %c1_i32_5  : i32 {
                        %133 = arith.muli %arg43, %c128_i32_6 overflow<nsw> : i32
                        %134 = arith.index_cast %133 : i32 to index
                        %135 = ascvec.loadalign %arg39[%134] : memref<?xui16, 1>, index -> <ui16>
                        %136 = arith.trunci %133 : i32 to i16
                        %137 = ascvec.arange %136 : i16 -> <ui16>
                        %138 = ascvec.cmp_gt %135, %128, %mask : <ui16>, <ui16> -> !ascvec.mask
                        %139 = ascvec.squeeze_store_reg %137, %138 : <ui16>, !ascvec.mask -> <ui16>
                        ascvec.store_unalign %arg38[%c22560], %139 : memref<?xui16, 1>, <ui16>
                      }
                      scf.for %arg43 = %c4_i32 to %c36_i32 step %c1_i32_5  : i32 {
                        %133 = arith.muli %arg43, %c128_i32_6 overflow<nsw> : i32
                        %134 = arith.index_cast %133 : i32 to index
                        %135 = ascvec.loadalign %arg39[%134] : memref<?xui16, 1>, index -> <ui16>
                        %136 = arith.trunci %133 : i32 to i16
                        %137 = ascvec.arange %136 : i16 -> <ui16>
                        %138 = ascvec.cmp_eq %135, %128, %mask : <ui16>, <ui16> -> !ascvec.mask
                        %139 = ascvec.squeeze_store_reg %137, %138 : <ui16>, !ascvec.mask -> <ui16>
                        ascvec.store_unalign %arg38[%c22560], %139 : memref<?xui16, 1>, <ui16>
                      }
                      ascvec.store_unalign_post %arg38[%c22560] : memref<?xui16, 1>
                      %129 = ascvec.loadalign_brc %arg40[%c33600] : memref<?xui16, 1>, index -> <ui16>
                      %130 = ascvec.loadalign_brc %arg41[%c34112] : memref<?xui16, 1>, index -> <ui16>
                      %131 = ascvec.shl %129, 8, %mask : <ui16> -> <ui16>
                      %132 = ascvec.add %131, %130, %mask : <ui16>
                      ascvec.store_unalign_begin %arg42 : memref<?xui16, 1>
                      scf.for %arg43 = %c4_i32 to %c36_i32 step %c1_i32_5  : i32 {
                        %133 = arith.muli %arg43, %c128_i32_6 overflow<nsw> : i32
                        %134 = arith.addi %133, %c4608_i32 overflow<nsw> : i32
                        %135 = arith.index_cast %134 : i32 to index
                        %136 = ascvec.loadalign %arg39[%135] : memref<?xui16, 1>, index -> <ui16>
                        %137 = arith.trunci %133 : i32 to i16
                        %138 = ascvec.arange %137 : i16 -> <ui16>
                        %139 = ascvec.cmp_gt %136, %132, %mask : <ui16>, <ui16> -> !ascvec.mask
                        %140 = ascvec.squeeze_store_reg %138, %139 : <ui16>, !ascvec.mask -> <ui16>
                        ascvec.store_unalign %arg42[%c27184], %140 : memref<?xui16, 1>, <ui16>
                      }
                      scf.for %arg43 = %c4_i32 to %c36_i32 step %c1_i32_5  : i32 {
                        %133 = arith.muli %arg43, %c128_i32_6 overflow<nsw> : i32
                        %134 = arith.addi %133, %c4608_i32 overflow<nsw> : i32
                        %135 = arith.index_cast %134 : i32 to index
                        %136 = ascvec.loadalign %arg39[%135] : memref<?xui16, 1>, index -> <ui16>
                        %137 = arith.trunci %133 : i32 to i16
                        %138 = ascvec.arange %137 : i16 -> <ui16>
                        %139 = ascvec.cmp_eq %136, %132, %mask : <ui16>, <ui16> -> !ascvec.mask
                        %140 = ascvec.squeeze_store_reg %138, %139 : <ui16>, !ascvec.mask -> <ui16>
                        ascvec.store_unalign %arg42[%c27184], %140 : memref<?xui16, 1>, <ui16>
                      }
                      %c0_7 = arith.constant 0 : index
                      %c1 = arith.constant 1 : index
                      scf.for %arg43 = %c0_7 to %c1 step %c1 {
                        ascvec.store_unalign_post %arg42[%c27184] : memref<?xui16, 1>
                      }
                    } {cannir.vf_mode = "simd"}
                    %121 = ascscalar.bitcast_memref %alloc_4 byte_offset = %c0_i64 : memref<262144xui8, 1> -> memref<?xf32, 1>
                    ascvec.vf(%87, %120, %120, %121, %120 : memref<?xi32, 1>, memref<?xui16, 1>, memref<?xui16, 1>, memref<?xf32, 1>, memref<?xui16, 1>) {
                    ^bb0(%arg36: memref<?xi32, 1>, %arg37: memref<?xui16, 1>, %arg38: memref<?xui16, 1>, %arg39: memref<?xf32, 1>, %arg40: memref<?xui16, 1>):
                      %c17336 = arith.constant 17336 : index
                      %c512_i32_5 = arith.constant 512 : i32
                      %c27184 = arith.constant 27184 : index
                      %c4608_i16 = arith.constant 4608 : i16
                      %c1_i32_6 = arith.constant 1 : i32
                      %c4_i32 = arith.constant 4 : i32
                      %c0_i32_7 = arith.constant 0 : i32
                      %c15904 = arith.constant 15904 : index
                      %c2_i32 = arith.constant 2 : i32
                      %c22560 = arith.constant 22560 : index
                      %c0_8 = arith.constant 0 : index
                      %c17328 = arith.constant 17328 : index
                      %c0_i16 = arith.constant 0 : i16
                      %c64_i32_9 = arith.constant 64 : i32
                      %c128_i32_10 = arith.constant 128 : i32
                      %mask, %remaining_out = ascvec.update_mask %c128_i32_10 elem_bits 16 : !ascvec.mask
                      %mask_11, %remaining_out_12 = ascvec.update_mask %c64_i32_9 : !ascvec.mask
                      %125 = ascvec.duplicate_scalar %c0_i16, %mask : i16 -> <ui16>
                      %126 = ascvec.loadalign_brc %arg36[%c17328] : memref<?xi32, 1>, index -> <i32>
                      %127 = ascvec.loadalign_brc %arg36[%c17336] : memref<?xi32, 1>, index -> <i32>
                      %128 = ascvec.reinterpret %126 : <i32> -> <ui16>
                      %129 = ascvec.duplicate %128, %mask : <ui16> -> <ui16>
                      scf.for %arg41 = %c0_i32_7 to %c4_i32 step %c1_i32_6  : i32 {
                        %133 = arith.muli %arg41, %c128_i32_10 overflow<nsw> : i32
                        %134 = arith.trunci %133 : i32 to i16
                        %135 = ascvec.arange %134 : i16 -> <ui16>
                        %136 = ascvec.cmp_lt %135, %129, %mask : <ui16>, <ui16> -> !ascvec.mask
                        %137 = arith.index_cast %133 : i32 to index
                        %138 = arith.addi %137, %c22560 : index
                        %139 = ascvec.loadalign %arg37[%138] : memref<?xui16, 1>, index -> <ui16>
                        %140 = ascvec.select %139, %125, %136 : <ui16>
                        %141 = ascvec.add %140, %125, %mask : <ui16>
                        %142 = ascvec.gather %arg38[%c0_8], %141, %mask : memref<?xui16, 1>, <ui16>, !ascvec.mask -> <ui16>
                        %143 = arith.divsi %133, %c2_i32 : i32
                        %144 = ascvec.select %142, %125, %136 : <ui16>
                        %145 = ascvec.reinterpret %144 : <ui16> -> <f32>
                        %146 = arith.index_cast %143 : i32 to index
                        %147 = arith.addi %146, %c15904 : index
                        ascvec.storealign %arg39[%147], %145, %mask_11 : memref<?xf32, 1>, index, <f32>
                      }
                      %130 = ascvec.reinterpret %127 : <i32> -> <ui16>
                      %131 = ascvec.duplicate %130, %mask : <ui16> -> <ui16>
                      %132 = ascvec.duplicate_scalar %c4608_i16, %mask : i16 -> <ui16>
                      scf.for %arg41 = %c0_i32_7 to %c4_i32 step %c1_i32_6  : i32 {
                        %133 = arith.muli %arg41, %c128_i32_10 overflow<nsw> : i32
                        %134 = arith.trunci %133 : i32 to i16
                        %135 = ascvec.arange %134 : i16 -> <ui16>
                        %136 = ascvec.cmp_lt %135, %131, %mask : <ui16>, <ui16> -> !ascvec.mask
                        %137 = arith.index_cast %133 : i32 to index
                        %138 = arith.addi %137, %c27184 : index
                        %139 = ascvec.loadalign %arg40[%138] : memref<?xui16, 1>, index -> <ui16>
                        %140 = ascvec.select %139, %125, %136 : <ui16>
                        %141 = ascvec.add %140, %132, %mask : <ui16>
                        %142 = ascvec.gather %arg38[%c0_8], %141, %mask : memref<?xui16, 1>, <ui16>, !ascvec.mask -> <ui16>
                        %143 = arith.addi %133, %c512_i32_5 overflow<nsw> : i32
                        %144 = arith.divsi %143, %c2_i32 : i32
                        %145 = ascvec.select %142, %125, %136 : <ui16>
                        %146 = ascvec.reinterpret %145 : <ui16> -> <f32>
                        %147 = arith.index_cast %144 : i32 to index
                        %148 = arith.addi %147, %c15904 : index
                        ascvec.storealign %arg39[%148], %146, %mask_11 : memref<?xf32, 1>, index, <f32>
                      }
                    } {cannir.vf_mode = "simd"}
                    %122 = arith.muli %104, %c1024_i64 overflow<nsw> : i64
                    %123 = arith.index_cast %122 : i64 to index
                    %124 = arith.addi %123, %c4608 : index
                    ascvec.vf(%87, %120, %1, %124, %120 : memref<?xi32, 1>, memref<?xui16, 1>, memref<?xui32, 1>, index, memref<?xui16, 1>) {
                    ^bb0(%arg36: memref<?xi32, 1>, %arg37: memref<?xui16, 1>, %arg38: memref<?xui32, 1>, %arg39: index, %arg40: memref<?xui16, 1>):
                      %c17336 = arith.constant 17336 : index
                      %c27184 = arith.constant 27184 : index
                      %c1_i32_5 = arith.constant 1 : i32
                      %c8_i32_6 = arith.constant 8 : i32
                      %c22560 = arith.constant 22560 : index
                      %c512_i32_7 = arith.constant 512 : i32
                      %c0_i32_8 = arith.constant 0 : i32
                      %c17328 = arith.constant 17328 : index
                      %c64_i32_9 = arith.constant 64 : i32
                      %mask, %remaining_out = ascvec.update_mask %c64_i32_9 : !ascvec.mask
                      %125 = ascvec.loadalign_brc %arg36[%c17328] : memref<?xi32, 1>, index -> <i32>
                      %126 = ascvec.loadalign_brc %arg36[%c17336] : memref<?xi32, 1>, index -> <i32>
                      %127 = ascvec.duplicate_scalar %c0_i32_8, %mask : i32 -> <ui32>
                      %128 = ascvec.duplicate_scalar %c512_i32_7, %mask : i32 -> <ui32>
                      %129 = ascvec.reinterpret %125 : <i32> -> <ui32>
                      scf.for %arg41 = %c0_i32_8 to %c8_i32_6 step %c1_i32_5  : i32 {
                        %131 = arith.muli %arg41, %c64_i32_9 overflow<nsw> : i32
                        %132 = ascvec.arange %131 : i32 -> <ui32>
                        %133 = ascvec.cmp_lt %132, %129, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %134 = arith.index_cast %131 : i32 to index
                        %135 = arith.addi %134, %c22560 : index
                        %136 = ascvec.loadalign_unpack %arg37[%135] {mode = "b16_to_b32"} : memref<?xui16, 1>, index -> <i32>
                        %137 = ascvec.reinterpret %136 : <i32> -> <ui32>
                        %138 = ascvec.select %137, %127, %133 : <ui32>
                        %139 = ascvec.add %138, %127, %mask : <ui32>
                        %140 = ascvec.sub %139, %128, %mask : <ui32>
                        %141 = ascvec.select %140, %127, %133 : <ui32>
                        %142 = arith.addi %arg39, %134 : index
                        ascvec.storealign %arg38[%142], %141, %mask : memref<?xui32, 1>, index, <ui32>
                      }
                      %130 = ascvec.reinterpret %126 : <i32> -> <ui32>
                      scf.for %arg41 = %c0_i32_8 to %c8_i32_6 step %c1_i32_5  : i32 {
                        %131 = arith.muli %arg41, %c64_i32_9 overflow<nsw> : i32
                        %132 = ascvec.arange %131 : i32 -> <ui32>
                        %133 = ascvec.cmp_lt %132, %130, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %134 = arith.index_cast %131 : i32 to index
                        %135 = arith.addi %134, %c27184 : index
                        %136 = ascvec.loadalign_unpack %arg40[%135] {mode = "b16_to_b32"} : memref<?xui16, 1>, index -> <i32>
                        %137 = ascvec.reinterpret %136 : <i32> -> <ui32>
                        %138 = ascvec.select %137, %127, %133 : <ui32>
                        %139 = ascvec.add %138, %127, %mask : <ui32>
                        %140 = ascvec.sub %139, %128, %mask : <ui32>
                        %141 = arith.addi %131, %c512_i32_7 overflow<nsw> : i32
                        %142 = ascvec.select %140, %127, %133 : <ui32>
                        %143 = arith.index_cast %141 : i32 to index
                        %144 = arith.addi %arg39, %143 : index
                        ascvec.storealign %arg38[%144], %142, %mask : memref<?xui32, 1>, index, <ui32>
                      }
                    } {cannir.vf_mode = "simd"}
                    ascvec.vf(%121, %120, %120 : memref<?xf32, 1>, memref<?xui16, 1>, memref<?xui16, 1>) {
                    ^bb0(%arg36: memref<?xf32, 1>, %arg37: memref<?xui16, 1>, %arg38: memref<?xui16, 1>):
                      %c33216 = arith.constant 33216 : index
                      %c33088 = arith.constant 33088 : index
                      %c4736_i32 = arith.constant 4736 : i32
                      %c32960 = arith.constant 32960 : index
                      %c4608_i32 = arith.constant 4608 : i32
                      %c512_i32_5 = arith.constant 512 : i32
                      %c32832 = arith.constant 32832 : index
                      %c1_i32_6 = arith.constant 1 : i32
                      %c0_i32_7 = arith.constant 0 : i32
                      %c64_i32_8 = arith.constant 64 : i32
                      %c15904 = arith.constant 15904 : index
                      %c2_i32 = arith.constant 2 : i32
                      %c0_i16 = arith.constant 0 : i16
                      %c128_i32_9 = arith.constant 128 : i32
                      %c256_i32_10 = arith.constant 256 : i32
                      %mask, %remaining_out = ascvec.update_mask %c256_i32_10 elem_bits 8 : !ascvec.mask
                      %mask_11, %remaining_out_12 = ascvec.update_mask %c128_i32_9 elem_bits 16 : !ascvec.mask
                      %125 = ascvec.duplicate_scalar %c0_i16, %mask_11 : i16 -> <ui16>
                      %126:2 = scf.for %arg39 = %c0_i32_7 to %c2_i32 step %c1_i32_6 iter_args(%arg40 = %125, %arg41 = %125) -> (!ascvec.vreg<ui16>, !ascvec.vreg<ui16>)  : i32 {
                        %128 = arith.muli %arg39, %c256_i32_10 overflow<nsw> : i32
                        %129 = arith.divsi %128, %c2_i32 : i32
                        %130 = arith.index_cast %129 : i32 to index
                        %131 = arith.addi %130, %c15904 : index
                        %132 = ascvec.loadalign %arg36[%131] : memref<?xf32, 1>, index -> <f32>
                        %133 = ascvec.reinterpret %132 : <f32> -> <ui16>
                        %134 = arith.addi %129, %c64_i32_8 overflow<nsw> : i32
                        %135 = arith.index_cast %134 : i32 to index
                        %136 = arith.addi %135, %c15904 : index
                        %137 = ascvec.loadalign %arg36[%136] : memref<?xf32, 1>, index -> <f32>
                        %138 = ascvec.reinterpret %137 : <f32> -> <ui16>
                        %139 = arith.index_cast %128 : i32 to index
                        ascvec.storealign %arg37[%139], %133, %mask_11 : memref<?xui16, 1>, index, <ui16>
                        %140 = arith.addi %128, %c128_i32_9 overflow<nsw> : i32
                        %141 = arith.index_cast %140 : i32 to index
                        ascvec.storealign %arg37[%141], %138, %mask_11 : memref<?xui16, 1>, index, <ui16>
                        %142 = ascvec.reinterpret %133 : <ui16> -> <ui8>
                        %143 = ascvec.reinterpret %138 : <ui16> -> <ui8>
                        %result0, %result1 = ascvec.deintlv %142, %143 : <ui8>, <ui8> -> <ui8>, <ui8>
                        %144 = ascvec.cumulative_histogram %arg40, %result1, %mask : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                        %145 = ascvec.cumulative_histogram %arg41, %result1, %mask {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                        scf.yield %144, %145 : !ascvec.vreg<ui16>, !ascvec.vreg<ui16>
                      }
                      %c0_13 = arith.constant 0 : index
                      %c1 = arith.constant 1 : index
                      scf.for %arg39 = %c0_13 to %c1 step %c1 {
                        ascvec.storealign %arg38[%c32832], %126#0, %mask_11 : memref<?xui16, 1>, index, <ui16>
                        ascvec.storealign %arg38[%c32960], %126#1, %mask_11 : memref<?xui16, 1>, index, <ui16>
                      }
                      %127:2 = scf.for %arg39 = %c0_i32_7 to %c2_i32 step %c1_i32_6 iter_args(%arg40 = %125, %arg41 = %125) -> (!ascvec.vreg<ui16>, !ascvec.vreg<ui16>)  : i32 {
                        %128 = arith.muli %arg39, %c256_i32_10 overflow<nsw> : i32
                        %129 = arith.addi %128, %c512_i32_5 overflow<nsw> : i32
                        %130 = arith.divsi %129, %c2_i32 : i32
                        %131 = arith.index_cast %130 : i32 to index
                        %132 = arith.addi %131, %c15904 : index
                        %133 = ascvec.loadalign %arg36[%132] : memref<?xf32, 1>, index -> <f32>
                        %134 = ascvec.reinterpret %133 : <f32> -> <ui16>
                        %135 = arith.addi %130, %c64_i32_8 overflow<nsw> : i32
                        %136 = arith.index_cast %135 : i32 to index
                        %137 = arith.addi %136, %c15904 : index
                        %138 = ascvec.loadalign %arg36[%137] : memref<?xf32, 1>, index -> <f32>
                        %139 = ascvec.reinterpret %138 : <f32> -> <ui16>
                        %140 = arith.addi %128, %c4608_i32 overflow<nsw> : i32
                        %141 = arith.index_cast %140 : i32 to index
                        ascvec.storealign %arg37[%141], %134, %mask_11 : memref<?xui16, 1>, index, <ui16>
                        %142 = arith.addi %128, %c4736_i32 overflow<nsw> : i32
                        %143 = arith.index_cast %142 : i32 to index
                        ascvec.storealign %arg37[%143], %139, %mask_11 : memref<?xui16, 1>, index, <ui16>
                        %144 = ascvec.reinterpret %134 : <ui16> -> <ui8>
                        %145 = ascvec.reinterpret %139 : <ui16> -> <ui8>
                        %result0, %result1 = ascvec.deintlv %144, %145 : <ui8>, <ui8> -> <ui8>, <ui8>
                        %146 = ascvec.cumulative_histogram %arg40, %result1, %mask : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                        %147 = ascvec.cumulative_histogram %arg41, %result1, %mask {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                        scf.yield %146, %147 : !ascvec.vreg<ui16>, !ascvec.vreg<ui16>
                      }
                      %c0_14 = arith.constant 0 : index
                      %c1_15 = arith.constant 1 : index
                      scf.for %arg39 = %c0_14 to %c1_15 step %c1_15 {
                        ascvec.storealign %arg38[%c33088], %127#0, %mask_11 : memref<?xui16, 1>, index, <ui16>
                        ascvec.storealign %arg38[%c33216], %127#1, %mask_11 : memref<?xui16, 1>, index, <ui16>
                      }
                    } {cannir.vf_mode = "simd"}
                  } else {
                    %120 = arith.remsi %118, %c256_i64 : i64
                    %121 = arith.cmpi ne, %120, %c0_i64 : i64
                    %122 = arith.cmpi slt, %120, %c0_i64 : i64
                    %123 = arith.andi %121, %122 : i1
                    %124 = arith.divsi %118, %c256_i64 : i64
                    %125 = arith.extui %123 : i1 to i64
                    %126 = arith.subi %124, %125 : i64
                    %127 = arith.remsi %118, %c128_i64 : i64
                    %128 = arith.cmpi ne, %127, %c0_i64 : i64
                    %129 = arith.cmpi slt, %127, %c0_i64 : i64
                    %130 = arith.andi %128, %129 : i1
                    %131 = arith.divsi %118, %c128_i64 : i64
                    %132 = arith.extui %130 : i1 to i64
                    %133 = arith.subi %131, %132 : i64
                    %134 = arith.addi %118, %c1_i64 : i64
                    %135 = arith.trunci %134 : i64 to i32
                    %136 = ascscalar.bitcast_memref %alloc_4 byte_offset = %c0_i64 : memref<262144xui8, 1> -> memref<?xui16, 1>
                    ascvec.vf(%87, %135, %136, %136, %136, %136 : memref<?xi32, 1>, i32, memref<?xui16, 1>, memref<?xui16, 1>, memref<?xui16, 1>, memref<?xui16, 1>) {
                    ^bb0(%arg36: memref<?xi32, 1>, %arg37: i32, %arg38: memref<?xui16, 1>, %arg39: memref<?xui16, 1>, %arg40: memref<?xui16, 1>, %arg41: memref<?xui16, 1>):
                      %c34496 = arith.constant 34496 : index
                      %c17336 = arith.constant 17336 : index
                      %c256_i16 = arith.constant 256 : i16
                      %c34368 = arith.constant 34368 : index
                      %c256_i32_5 = arith.constant 256 : i32
                      %c33600 = arith.constant 33600 : index
                      %c2_i32 = arith.constant 2 : i32
                      %c0_i32_6 = arith.constant 0 : i32
                      %c32832 = arith.constant 32832 : index
                      %c33344 = arith.constant 33344 : index
                      %c1_i32_7 = arith.constant 1 : i32
                      %c17328 = arith.constant 17328 : index
                      %c1_i16 = arith.constant 1 : i16
                      %c0_i16 = arith.constant 0 : i16
                      %c64_i32_8 = arith.constant 64 : i32
                      %c128_i32_9 = arith.constant 128 : i32
                      %mask, %remaining_out = ascvec.update_mask %c128_i32_9 elem_bits 16 : !ascvec.mask
                      %mask_10, %remaining_out_11 = ascvec.update_mask %c64_i32_8 : !ascvec.mask
                      %145 = ascvec.duplicate_scalar %c0_i16, %mask : i16 -> <ui16>
                      %146 = ascvec.duplicate_scalar %c1_i16, %mask : i16 -> <ui16>
                      %147 = ascvec.loadalign_brc %arg36[%c17328] : memref<?xi32, 1>, index -> <i32>
                      %148 = ascvec.loadalign_brc %arg36[%c17336] : memref<?xi32, 1>, index -> <i32>
                      %149 = ascvec.max_scalar %147, %c1_i32_7, %mask_10 : <i32>, i32
                      %150 = ascvec.duplicate_scalar %arg37, %mask_10 : i32 -> <i32>
                      %151 = ascvec.sub %150, %149, %mask_10 : <i32>
                      %152 = ascvec.reinterpret %151 : <i32> -> <ui16>
                      %153 = ascvec.duplicate %152, %mask : <ui16> -> <ui16>
                      ascvec.store_unalign_begin %arg38 : memref<?xui16, 1>
                      scf.for %arg42 = %c0_i32_6 to %c2_i32 step %c1_i32_7  : i32 {
                        %158 = arith.muli %arg42, %c128_i32_9 overflow<nsw> : i32
                        %159 = arith.trunci %158 : i32 to i16
                        %160 = ascvec.arange %159 : i16 -> <ui16>
                        %161 = arith.index_cast %158 : i32 to index
                        %162 = arith.addi %161, %c32832 : index
                        %163 = ascvec.loadalign %arg39[%162] : memref<?xui16, 1>, index -> <ui16>
                        %164 = ascvec.cmp_ge %163, %153, %mask : <ui16>, <ui16> -> !ascvec.mask
                        %165 = ascvec.squeeze_store_reg %160, %164 : <ui16>, !ascvec.mask -> <ui16>
                        ascvec.store_unalign %arg38[%c33344], %165 : memref<?xui16, 1>, <ui16>
                      }
                      ascvec.store_unalign_post %arg38[%c33344] : memref<?xui16, 1>
                      %154 = ascvec.max_scalar %148, %c1_i32_7, %mask_10 : <i32>, i32
                      %155 = ascvec.sub %150, %154, %mask_10 : <i32>
                      %156 = ascvec.reinterpret %155 : <i32> -> <ui16>
                      %157 = ascvec.duplicate %156, %mask : <ui16> -> <ui16>
                      ascvec.store_unalign_begin %arg40 : memref<?xui16, 1>
                      scf.for %arg42 = %c0_i32_6 to %c2_i32 step %c1_i32_7  : i32 {
                        %158 = arith.muli %arg42, %c128_i32_9 overflow<nsw> : i32
                        %159 = arith.trunci %158 : i32 to i16
                        %160 = ascvec.arange %159 : i16 -> <ui16>
                        %161 = arith.addi %158, %c256_i32_5 overflow<nsw> : i32
                        %162 = arith.index_cast %161 : i32 to index
                        %163 = arith.addi %162, %c32832 : index
                        %164 = ascvec.loadalign %arg39[%163] : memref<?xui16, 1>, index -> <ui16>
                        %165 = ascvec.cmp_ge %164, %157, %mask : <ui16>, <ui16> -> !ascvec.mask
                        %166 = ascvec.squeeze_store_reg %160, %165 : <ui16>, !ascvec.mask -> <ui16>
                        ascvec.store_unalign %arg40[%c33600], %166 : memref<?xui16, 1>, <ui16>
                      }
                      %c0_12 = arith.constant 0 : index
                      %c1 = arith.constant 1 : index
                      scf.for %arg42 = %c0_12 to %c1 step %c1 {
                        ascvec.store_unalign_post %arg40[%c33600] : memref<?xui16, 1>
                      }
                      ascvec.mem_bar vst_vld
                      %c0_13 = arith.constant 0 : index
                      %c1_14 = arith.constant 1 : index
                      scf.for %arg42 = %c0_13 to %c1_14 step %c1_14 {
                        %158 = ascvec.loadalign_brc %arg38[%c33344] : memref<?xui16, 1>, index -> <ui16>
                        %159 = ascvec.cmp_eq_scalar %158, %c0_i16, %mask : <ui16>, i16 -> !ascvec.mask
                        %160 = ascvec.sub %158, %146, %mask : <ui16>
                        %161 = ascvec.select %145, %160, %159 : <ui16>
                        %162 = ascvec.add %161, %145, %mask : <ui16>
                        %163 = ascvec.gather %arg39[%c32832], %162, %mask : memref<?xui16, 1>, <ui16>, !ascvec.mask -> <ui16>
                        %164 = ascvec.select %145, %163, %159 : <ui16>
                        %165 = ascvec.sub %153, %164, %mask : <ui16>
                        ascvec.storealign %arg41[%c34368], %165, %mask : memref<?xui16, 1>, index, <ui16>
                        %166 = ascvec.loadalign_brc %arg40[%c33600] : memref<?xui16, 1>, index -> <ui16>
                        %167 = ascvec.cmp_eq_scalar %166, %c0_i16, %mask : <ui16>, i16 -> !ascvec.mask
                        %168 = ascvec.sub %166, %146, %mask : <ui16>
                        %169 = ascvec.select %145, %168, %167 : <ui16>
                        %170 = ascvec.duplicate_scalar %c256_i16, %mask : i16 -> <ui16>
                        %171 = ascvec.add %169, %170, %mask : <ui16>
                        %172 = ascvec.gather %arg39[%c32832], %171, %mask : memref<?xui16, 1>, <ui16>, !ascvec.mask -> <ui16>
                        %173 = ascvec.select %145, %172, %167 : <ui16>
                        %174 = ascvec.sub %157, %173, %mask : <ui16>
                        ascvec.storealign %arg41[%c34496], %174, %mask : memref<?xui16, 1>, index, <ui16>
                      }
                    } {cannir.vf_mode = "simd"}
                    ascvec.vf(%136, %136, %126, %136, %136 : memref<?xui16, 1>, memref<?xui16, 1>, i64, memref<?xui16, 1>, memref<?xui16, 1>) {
                    ^bb0(%arg36: memref<?xui16, 1>, %arg37: memref<?xui16, 1>, %arg38: i64, %arg39: memref<?xui16, 1>, %arg40: memref<?xui16, 1>):
                      %c33216 = arith.constant 33216 : index
                      %c33088 = arith.constant 33088 : index
                      %c32960 = arith.constant 32960 : index
                      %c4608_i64_5 = arith.constant 4608 : i64
                      %c33600 = arith.constant 33600 : index
                      %c32832 = arith.constant 32832 : index
                      %c1_i64_6 = arith.constant 1 : i64
                      %c0_i64_7 = arith.constant 0 : i64
                      %c256_i64_8 = arith.constant 256 : i64
                      %c0_i16 = arith.constant 0 : i16
                      %c33344 = arith.constant 33344 : index
                      %c128_i32_9 = arith.constant 128 : i32
                      %c256_i32_10 = arith.constant 256 : i32
                      %mask, %remaining_out = ascvec.update_mask %c256_i32_10 elem_bits 8 : !ascvec.mask
                      %mask_11, %remaining_out_12 = ascvec.update_mask %c128_i32_9 elem_bits 16 : !ascvec.mask
                      %145 = ascvec.loadalign_brc %arg36[%c33344] : memref<?xui16, 1>, index -> <ui16>
                      %146 = ascvec.reinterpret %145 : <ui16> -> <ui8>
                      %147 = ascvec.duplicate %146, %mask : <ui8> -> <ui8>
                      %148 = ascvec.duplicate_scalar %c0_i16, %mask_11 : i16 -> <ui16>
                      %149:2 = scf.for %arg41 = %c0_i64_7 to %arg38 step %c1_i64_6 iter_args(%arg42 = %148, %arg43 = %148) -> (!ascvec.vreg<ui16>, !ascvec.vreg<ui16>)  : i64 {
                        %154 = arith.muli %arg41, %c256_i64_8 : i64
                        %155 = arith.index_cast %154 : i64 to index
                        %result0, %result1 = ascvec.loadalign_deintlv %arg37[%155] {dintlv_width = "b8"} : memref<?xui16, 1>, index -> <ui8>, <ui8>
                        %156 = ascvec.cmp_eq %result1, %147, %mask : <ui8>, <ui8> -> !ascvec.mask
                        %157 = ascvec.cumulative_histogram %arg42, %result0, %156 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                        %158 = ascvec.cumulative_histogram %arg43, %result0, %156 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                        scf.yield %157, %158 : !ascvec.vreg<ui16>, !ascvec.vreg<ui16>
                      }
                      ascvec.storealign %arg39[%c32832], %149#0, %mask_11 : memref<?xui16, 1>, index, <ui16>
                      ascvec.storealign %arg39[%c32960], %149#1, %mask_11 : memref<?xui16, 1>, index, <ui16>
                      %150 = ascvec.loadalign_brc %arg40[%c33600] : memref<?xui16, 1>, index -> <ui16>
                      %151 = ascvec.reinterpret %150 : <ui16> -> <ui8>
                      %152 = ascvec.duplicate %151, %mask : <ui8> -> <ui8>
                      %153:2 = scf.for %arg41 = %c0_i64_7 to %arg38 step %c1_i64_6 iter_args(%arg42 = %148, %arg43 = %148) -> (!ascvec.vreg<ui16>, !ascvec.vreg<ui16>)  : i64 {
                        %154 = arith.muli %arg41, %c256_i64_8 : i64
                        %155 = arith.addi %154, %c4608_i64_5 : i64
                        %156 = arith.index_cast %155 : i64 to index
                        %result0, %result1 = ascvec.loadalign_deintlv %arg37[%156] {dintlv_width = "b8"} : memref<?xui16, 1>, index -> <ui8>, <ui8>
                        %157 = ascvec.cmp_eq %result1, %152, %mask : <ui8>, <ui8> -> !ascvec.mask
                        %158 = ascvec.cumulative_histogram %arg42, %result0, %157 : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                        %159 = ascvec.cumulative_histogram %arg43, %result0, %157 {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                        scf.yield %158, %159 : !ascvec.vreg<ui16>, !ascvec.vreg<ui16>
                      }
                      %c0_13 = arith.constant 0 : index
                      %c1 = arith.constant 1 : index
                      scf.for %arg41 = %c0_13 to %c1 step %c1 {
                        ascvec.storealign %arg39[%c33088], %153#0, %mask_11 : memref<?xui16, 1>, index, <ui16>
                        ascvec.storealign %arg39[%c33216], %153#1, %mask_11 : memref<?xui16, 1>, index, <ui16>
                      }
                    } {cannir.vf_mode = "simd"}
                    ascvec.vf(%136, %136, %136, %136 : memref<?xui16, 1>, memref<?xui16, 1>, memref<?xui16, 1>, memref<?xui16, 1>) {
                    ^bb0(%arg36: memref<?xui16, 1>, %arg37: memref<?xui16, 1>, %arg38: memref<?xui16, 1>, %arg39: memref<?xui16, 1>):
                      %c34496 = arith.constant 34496 : index
                      %c256_i32_5 = arith.constant 256 : i32
                      %c34112 = arith.constant 34112 : index
                      %c1_i32_6 = arith.constant 1 : i32
                      %c2_i32 = arith.constant 2 : i32
                      %c0_i32_7 = arith.constant 0 : i32
                      %c32832 = arith.constant 32832 : index
                      %c33856 = arith.constant 33856 : index
                      %c34368 = arith.constant 34368 : index
                      %c128_i32_8 = arith.constant 128 : i32
                      %mask, %remaining_out = ascvec.update_mask %c128_i32_8 elem_bits 16 : !ascvec.mask
                      %145 = ascvec.loadalign_brc %arg36[%c34368] : memref<?xui16, 1>, index -> <ui16>
                      ascvec.store_unalign_begin %arg37 : memref<?xui16, 1>
                      scf.for %arg40 = %c0_i32_7 to %c2_i32 step %c1_i32_6  : i32 {
                        %147 = arith.muli %arg40, %c128_i32_8 overflow<nsw> : i32
                        %148 = arith.trunci %147 : i32 to i16
                        %149 = ascvec.arange %148 : i16 -> <ui16>
                        %150 = arith.index_cast %147 : i32 to index
                        %151 = arith.addi %150, %c32832 : index
                        %152 = ascvec.loadalign %arg38[%151] : memref<?xui16, 1>, index -> <ui16>
                        %153 = ascvec.cmp_ge %152, %145, %mask : <ui16>, <ui16> -> !ascvec.mask
                        %154 = ascvec.squeeze_store_reg %149, %153 : <ui16>, !ascvec.mask -> <ui16>
                        ascvec.store_unalign %arg37[%c33856], %154 : memref<?xui16, 1>, <ui16>
                      }
                      ascvec.store_unalign_post %arg37[%c33856] : memref<?xui16, 1>
                      %146 = ascvec.loadalign_brc %arg36[%c34496] : memref<?xui16, 1>, index -> <ui16>
                      ascvec.store_unalign_begin %arg39 : memref<?xui16, 1>
                      scf.for %arg40 = %c0_i32_7 to %c2_i32 step %c1_i32_6  : i32 {
                        %147 = arith.muli %arg40, %c128_i32_8 overflow<nsw> : i32
                        %148 = arith.trunci %147 : i32 to i16
                        %149 = ascvec.arange %148 : i16 -> <ui16>
                        %150 = arith.addi %147, %c256_i32_5 overflow<nsw> : i32
                        %151 = arith.index_cast %150 : i32 to index
                        %152 = arith.addi %151, %c32832 : index
                        %153 = ascvec.loadalign %arg38[%152] : memref<?xui16, 1>, index -> <ui16>
                        %154 = ascvec.cmp_ge %153, %146, %mask : <ui16>, <ui16> -> !ascvec.mask
                        %155 = ascvec.squeeze_store_reg %149, %154 : <ui16>, !ascvec.mask -> <ui16>
                        ascvec.store_unalign %arg39[%c34112], %155 : memref<?xui16, 1>, <ui16>
                      }
                      %c0_9 = arith.constant 0 : index
                      %c1 = arith.constant 1 : index
                      scf.for %arg40 = %c0_9 to %c1 step %c1 {
                        ascvec.store_unalign_post %arg39[%c34112] : memref<?xui16, 1>
                      }
                    } {cannir.vf_mode = "simd"}
                    ascvec.vf(%136, %136, %136, %136, %133, %136, %136, %136 : memref<?xui16, 1>, memref<?xui16, 1>, memref<?xui16, 1>, memref<?xui16, 1>, i64, memref<?xui16, 1>, memref<?xui16, 1>, memref<?xui16, 1>) {
                    ^bb0(%arg36: memref<?xui16, 1>, %arg37: memref<?xui16, 1>, %arg38: memref<?xui16, 1>, %arg39: memref<?xui16, 1>, %arg40: i64, %arg41: memref<?xui16, 1>, %arg42: memref<?xui16, 1>, %arg43: memref<?xui16, 1>):
                      %c4608_i64_5 = arith.constant 4608 : i64
                      %c27184 = arith.constant 27184 : index
                      %c34112 = arith.constant 34112 : index
                      %c33600 = arith.constant 33600 : index
                      %c1_i64_6 = arith.constant 1 : i64
                      %c0_i64_7 = arith.constant 0 : i64
                      %c128_i64_8 = arith.constant 128 : i64
                      %c22560 = arith.constant 22560 : index
                      %c33856 = arith.constant 33856 : index
                      %c33344 = arith.constant 33344 : index
                      %c128_i32_9 = arith.constant 128 : i32
                      %mask, %remaining_out = ascvec.update_mask %c128_i32_9 elem_bits 16 : !ascvec.mask
                      %145 = ascvec.loadalign_brc %arg36[%c33344] : memref<?xui16, 1>, index -> <ui16>
                      %146 = ascvec.loadalign_brc %arg37[%c33856] : memref<?xui16, 1>, index -> <ui16>
                      %147 = ascvec.shl %145, 8, %mask : <ui16> -> <ui16>
                      %148 = ascvec.add %147, %146, %mask : <ui16>
                      ascvec.store_unalign_begin %arg38 : memref<?xui16, 1>
                      scf.for %arg44 = %c0_i64_7 to %arg40 step %c1_i64_6  : i64 {
                        %153 = arith.muli %arg44, %c128_i64_8 : i64
                        %154 = arith.index_cast %153 : i64 to index
                        %155 = ascvec.loadalign %arg39[%154] : memref<?xui16, 1>, index -> <ui16>
                        %156 = arith.trunci %153 : i64 to i16
                        %157 = ascvec.arange %156 : i16 -> <ui16>
                        %158 = ascvec.cmp_gt %155, %148, %mask : <ui16>, <ui16> -> !ascvec.mask
                        %159 = ascvec.squeeze_store_reg %157, %158 : <ui16>, !ascvec.mask -> <ui16>
                        ascvec.store_unalign %arg38[%c22560], %159 : memref<?xui16, 1>, <ui16>
                      }
                      scf.for %arg44 = %c0_i64_7 to %arg40 step %c1_i64_6  : i64 {
                        %153 = arith.muli %arg44, %c128_i64_8 : i64
                        %154 = arith.index_cast %153 : i64 to index
                        %155 = ascvec.loadalign %arg39[%154] : memref<?xui16, 1>, index -> <ui16>
                        %156 = arith.trunci %153 : i64 to i16
                        %157 = ascvec.arange %156 : i16 -> <ui16>
                        %158 = ascvec.cmp_eq %155, %148, %mask : <ui16>, <ui16> -> !ascvec.mask
                        %159 = ascvec.squeeze_store_reg %157, %158 : <ui16>, !ascvec.mask -> <ui16>
                        ascvec.store_unalign %arg38[%c22560], %159 : memref<?xui16, 1>, <ui16>
                      }
                      ascvec.store_unalign_post %arg38[%c22560] : memref<?xui16, 1>
                      %149 = ascvec.loadalign_brc %arg41[%c33600] : memref<?xui16, 1>, index -> <ui16>
                      %150 = ascvec.loadalign_brc %arg42[%c34112] : memref<?xui16, 1>, index -> <ui16>
                      %151 = ascvec.shl %149, 8, %mask : <ui16> -> <ui16>
                      %152 = ascvec.add %151, %150, %mask : <ui16>
                      ascvec.store_unalign_begin %arg43 : memref<?xui16, 1>
                      scf.for %arg44 = %c0_i64_7 to %arg40 step %c1_i64_6  : i64 {
                        %153 = arith.muli %arg44, %c128_i64_8 : i64
                        %154 = arith.addi %153, %c4608_i64_5 : i64
                        %155 = arith.index_cast %154 : i64 to index
                        %156 = ascvec.loadalign %arg39[%155] : memref<?xui16, 1>, index -> <ui16>
                        %157 = arith.trunci %153 : i64 to i16
                        %158 = ascvec.arange %157 : i16 -> <ui16>
                        %159 = ascvec.cmp_gt %156, %152, %mask : <ui16>, <ui16> -> !ascvec.mask
                        %160 = ascvec.squeeze_store_reg %158, %159 : <ui16>, !ascvec.mask -> <ui16>
                        ascvec.store_unalign %arg43[%c27184], %160 : memref<?xui16, 1>, <ui16>
                      }
                      scf.for %arg44 = %c0_i64_7 to %arg40 step %c1_i64_6  : i64 {
                        %153 = arith.muli %arg44, %c128_i64_8 : i64
                        %154 = arith.addi %153, %c4608_i64_5 : i64
                        %155 = arith.index_cast %154 : i64 to index
                        %156 = ascvec.loadalign %arg39[%155] : memref<?xui16, 1>, index -> <ui16>
                        %157 = arith.trunci %153 : i64 to i16
                        %158 = ascvec.arange %157 : i16 -> <ui16>
                        %159 = ascvec.cmp_eq %156, %152, %mask : <ui16>, <ui16> -> !ascvec.mask
                        %160 = ascvec.squeeze_store_reg %158, %159 : <ui16>, !ascvec.mask -> <ui16>
                        ascvec.store_unalign %arg43[%c27184], %160 : memref<?xui16, 1>, <ui16>
                      }
                      %c0_10 = arith.constant 0 : index
                      %c1 = arith.constant 1 : index
                      scf.for %arg44 = %c0_10 to %c1 step %c1 {
                        ascvec.store_unalign_post %arg43[%c27184] : memref<?xui16, 1>
                      }
                    } {cannir.vf_mode = "simd"}
                    %137 = ascscalar.bitcast_memref %alloc_4 byte_offset = %c0_i64 : memref<262144xui8, 1> -> memref<?xf32, 1>
                    ascvec.vf(%87, %136, %136, %137, %136 : memref<?xi32, 1>, memref<?xui16, 1>, memref<?xui16, 1>, memref<?xf32, 1>, memref<?xui16, 1>) {
                    ^bb0(%arg36: memref<?xi32, 1>, %arg37: memref<?xui16, 1>, %arg38: memref<?xui16, 1>, %arg39: memref<?xf32, 1>, %arg40: memref<?xui16, 1>):
                      %c17336 = arith.constant 17336 : index
                      %c512_i32_5 = arith.constant 512 : i32
                      %c27184 = arith.constant 27184 : index
                      %c4608_i16 = arith.constant 4608 : i16
                      %c1_i32_6 = arith.constant 1 : i32
                      %c4_i32 = arith.constant 4 : i32
                      %c0_i32_7 = arith.constant 0 : i32
                      %c15904 = arith.constant 15904 : index
                      %c2_i32 = arith.constant 2 : i32
                      %c22560 = arith.constant 22560 : index
                      %c0_8 = arith.constant 0 : index
                      %c17328 = arith.constant 17328 : index
                      %c0_i16 = arith.constant 0 : i16
                      %c64_i32_9 = arith.constant 64 : i32
                      %c128_i32_10 = arith.constant 128 : i32
                      %mask, %remaining_out = ascvec.update_mask %c128_i32_10 elem_bits 16 : !ascvec.mask
                      %mask_11, %remaining_out_12 = ascvec.update_mask %c64_i32_9 : !ascvec.mask
                      %145 = ascvec.duplicate_scalar %c0_i16, %mask : i16 -> <ui16>
                      %146 = ascvec.loadalign_brc %arg36[%c17328] : memref<?xi32, 1>, index -> <i32>
                      %147 = ascvec.loadalign_brc %arg36[%c17336] : memref<?xi32, 1>, index -> <i32>
                      %148 = ascvec.reinterpret %146 : <i32> -> <ui16>
                      %149 = ascvec.duplicate %148, %mask : <ui16> -> <ui16>
                      scf.for %arg41 = %c0_i32_7 to %c4_i32 step %c1_i32_6  : i32 {
                        %153 = arith.muli %arg41, %c128_i32_10 overflow<nsw> : i32
                        %154 = arith.trunci %153 : i32 to i16
                        %155 = ascvec.arange %154 : i16 -> <ui16>
                        %156 = ascvec.cmp_lt %155, %149, %mask : <ui16>, <ui16> -> !ascvec.mask
                        %157 = arith.index_cast %153 : i32 to index
                        %158 = arith.addi %157, %c22560 : index
                        %159 = ascvec.loadalign %arg37[%158] : memref<?xui16, 1>, index -> <ui16>
                        %160 = ascvec.select %159, %145, %156 : <ui16>
                        %161 = ascvec.add %160, %145, %mask : <ui16>
                        %162 = ascvec.gather %arg38[%c0_8], %161, %mask : memref<?xui16, 1>, <ui16>, !ascvec.mask -> <ui16>
                        %163 = arith.divsi %153, %c2_i32 : i32
                        %164 = ascvec.select %162, %145, %156 : <ui16>
                        %165 = ascvec.reinterpret %164 : <ui16> -> <f32>
                        %166 = arith.index_cast %163 : i32 to index
                        %167 = arith.addi %166, %c15904 : index
                        ascvec.storealign %arg39[%167], %165, %mask_11 : memref<?xf32, 1>, index, <f32>
                      }
                      %150 = ascvec.reinterpret %147 : <i32> -> <ui16>
                      %151 = ascvec.duplicate %150, %mask : <ui16> -> <ui16>
                      %152 = ascvec.duplicate_scalar %c4608_i16, %mask : i16 -> <ui16>
                      scf.for %arg41 = %c0_i32_7 to %c4_i32 step %c1_i32_6  : i32 {
                        %153 = arith.muli %arg41, %c128_i32_10 overflow<nsw> : i32
                        %154 = arith.trunci %153 : i32 to i16
                        %155 = ascvec.arange %154 : i16 -> <ui16>
                        %156 = ascvec.cmp_lt %155, %151, %mask : <ui16>, <ui16> -> !ascvec.mask
                        %157 = arith.index_cast %153 : i32 to index
                        %158 = arith.addi %157, %c27184 : index
                        %159 = ascvec.loadalign %arg40[%158] : memref<?xui16, 1>, index -> <ui16>
                        %160 = ascvec.select %159, %145, %156 : <ui16>
                        %161 = ascvec.add %160, %152, %mask : <ui16>
                        %162 = ascvec.gather %arg38[%c0_8], %161, %mask : memref<?xui16, 1>, <ui16>, !ascvec.mask -> <ui16>
                        %163 = arith.addi %153, %c512_i32_5 overflow<nsw> : i32
                        %164 = arith.divsi %163, %c2_i32 : i32
                        %165 = ascvec.select %162, %145, %156 : <ui16>
                        %166 = ascvec.reinterpret %165 : <ui16> -> <f32>
                        %167 = arith.index_cast %164 : i32 to index
                        %168 = arith.addi %167, %c15904 : index
                        ascvec.storealign %arg39[%168], %166, %mask_11 : memref<?xf32, 1>, index, <f32>
                      }
                    } {cannir.vf_mode = "simd"}
                    %138 = arith.trunci %116 : i64 to i32
                    %139 = arith.muli %97, %c1024_i64 overflow<nsw> : i64
                    %140 = arith.index_cast %139 : i64 to index
                    %141 = arith.addi %140, %c4608 : index
                    %142 = arith.muli %104, %c1024_i64 overflow<nsw> : i64
                    %143 = arith.index_cast %142 : i64 to index
                    %144 = arith.addi %143, %c4608 : index
                    ascvec.vf(%87, %138, %136, %1, %141, %136, %1, %144, %136 : memref<?xi32, 1>, i32, memref<?xui16, 1>, memref<?xui32, 1>, index, memref<?xui16, 1>, memref<?xui32, 1>, index, memref<?xui16, 1>) {
                    ^bb0(%arg36: memref<?xi32, 1>, %arg37: i32, %arg38: memref<?xui16, 1>, %arg39: memref<?xui32, 1>, %arg40: index, %arg41: memref<?xui16, 1>, %arg42: memref<?xui32, 1>, %arg43: index, %arg44: memref<?xui16, 1>):
                      %c17336 = arith.constant 17336 : index
                      %c4624_i16 = arith.constant 4624 : i16
                      %c27184 = arith.constant 27184 : index
                      %c1_i32_5 = arith.constant 1 : i32
                      %c8_i32_6 = arith.constant 8 : i32
                      %c13312 = arith.constant 13312 : index
                      %c0_i16 = arith.constant 0 : i16
                      %c22560 = arith.constant 22560 : index
                      %c512_i32_7 = arith.constant 512 : i32
                      %c0_i32_8 = arith.constant 0 : i32
                      %c17328 = arith.constant 17328 : index
                      %c64_i32_9 = arith.constant 64 : i32
                      %mask, %remaining_out = ascvec.update_mask %c64_i32_9 : !ascvec.mask
                      %145 = ascvec.loadalign_brc %arg36[%c17328] : memref<?xi32, 1>, index -> <i32>
                      %146 = ascvec.loadalign_brc %arg36[%c17336] : memref<?xi32, 1>, index -> <i32>
                      %147 = ascvec.duplicate_scalar %c0_i32_8, %mask : i32 -> <ui32>
                      %148 = ascvec.duplicate_scalar %c512_i32_7, %mask : i32 -> <ui32>
                      %149 = ascvec.duplicate_scalar %arg37, %mask : i32 -> <ui32>
                      %150 = ascvec.reinterpret %145 : <i32> -> <ui32>
                      scf.for %arg45 = %c0_i32_8 to %c8_i32_6 step %c1_i32_5  : i32 {
                        %152 = arith.muli %arg45, %c64_i32_9 overflow<nsw> : i32
                        %153 = ascvec.arange %152 : i32 -> <ui32>
                        %154 = ascvec.cmp_lt %153, %150, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %155 = arith.index_cast %152 : i32 to index
                        %156 = arith.addi %155, %c22560 : index
                        %157 = ascvec.loadalign_unpack %arg38[%156] {mode = "b16_to_b32"} : memref<?xui16, 1>, index -> <i32>
                        %158 = ascvec.reinterpret %157 : <i32> -> <ui32>
                        %159 = ascvec.select %158, %147, %154 : <ui32>
                        %160 = ascvec.cmp_lt %159, %148, %154 : <ui32>, <ui32> -> !ascvec.mask
                        %161 = ascvec.select %159, %147, %160 : <ui32>
                        %162 = ascvec.add %161, %147, %mask : <ui32>
                        %163 = ascvec.gather %arg39[%arg40], %162, %mask : memref<?xui32, 1>, <ui32>, !ascvec.mask -> <ui32>
                        %164 = ascvec.sub %159, %148, %mask : <ui32>
                        %165 = ascvec.select %147, %164, %160 : <ui32>
                        %166 = ascvec.select %165, %147, %154 : <ui32>
                        %167 = ascvec.pack %166 src_elem_bits 32 dst_elem_bits 16 {part = "low"} : <ui32> -> <ui16>
                        %mask_10, %remaining_out_11 = ascvec.update_mask %c64_i32_9 elem_bits 16 : !ascvec.mask
                        %168 = ascvec.add_scalar %167, %c0_i16, %mask_10 : <ui16>, i16
                        %169 = ascvec.gather %arg41[%c13312], %168, %mask_10 : memref<?xui16, 1>, <ui16>, !ascvec.mask -> <ui16>
                        %170 = ascvec.unpack %169 {part = "lower"} : <ui16> -> <ui32>
                        %171 = ascvec.add %170, %149, %mask : <ui32>
                        %172 = ascvec.select %163, %171, %160 : <ui32>
                        %173 = ascvec.select %172, %147, %154 : <ui32>
                        %174 = arith.addi %arg43, %155 : index
                        ascvec.storealign %arg42[%174], %173, %mask : memref<?xui32, 1>, index, <ui32>
                      }
                      %151 = ascvec.reinterpret %146 : <i32> -> <ui32>
                      scf.for %arg45 = %c0_i32_8 to %c8_i32_6 step %c1_i32_5  : i32 {
                        %152 = arith.muli %arg45, %c64_i32_9 overflow<nsw> : i32
                        %153 = ascvec.arange %152 : i32 -> <ui32>
                        %154 = ascvec.cmp_lt %153, %151, %mask : <ui32>, <ui32> -> !ascvec.mask
                        %155 = arith.index_cast %152 : i32 to index
                        %156 = arith.addi %155, %c27184 : index
                        %157 = ascvec.loadalign_unpack %arg44[%156] {mode = "b16_to_b32"} : memref<?xui16, 1>, index -> <i32>
                        %158 = ascvec.reinterpret %157 : <i32> -> <ui32>
                        %159 = ascvec.select %158, %147, %154 : <ui32>
                        %160 = ascvec.cmp_lt %159, %148, %154 : <ui32>, <ui32> -> !ascvec.mask
                        %161 = ascvec.select %159, %147, %160 : <ui32>
                        %162 = ascvec.add %161, %148, %mask : <ui32>
                        %163 = ascvec.gather %arg39[%arg40], %162, %mask : memref<?xui32, 1>, <ui32>, !ascvec.mask -> <ui32>
                        %164 = ascvec.sub %159, %148, %mask : <ui32>
                        %165 = ascvec.select %147, %164, %160 : <ui32>
                        %166 = ascvec.select %165, %147, %154 : <ui32>
                        %167 = ascvec.pack %166 src_elem_bits 32 dst_elem_bits 16 {part = "low"} : <ui32> -> <ui16>
                        %mask_10, %remaining_out_11 = ascvec.update_mask %c64_i32_9 elem_bits 16 : !ascvec.mask
                        %168 = ascvec.add_scalar %167, %c4624_i16, %mask_10 : <ui16>, i16
                        %169 = ascvec.gather %arg41[%c13312], %168, %mask_10 : memref<?xui16, 1>, <ui16>, !ascvec.mask -> <ui16>
                        %170 = ascvec.unpack %169 {part = "lower"} : <ui16> -> <ui32>
                        %171 = ascvec.add %170, %149, %mask : <ui32>
                        %172 = ascvec.select %163, %171, %160 : <ui32>
                        %173 = arith.addi %152, %c512_i32_7 overflow<nsw> : i32
                        %174 = ascvec.select %172, %147, %154 : <ui32>
                        %175 = arith.index_cast %173 : i32 to index
                        %176 = arith.addi %arg43, %175 : index
                        ascvec.storealign %arg42[%176], %174, %mask : memref<?xui32, 1>, index, <ui32>
                      }
                    } {cannir.vf_mode = "simd"}
                    ascvec.vf(%137, %136, %136 : memref<?xf32, 1>, memref<?xui16, 1>, memref<?xui16, 1>) {
                    ^bb0(%arg36: memref<?xf32, 1>, %arg37: memref<?xui16, 1>, %arg38: memref<?xui16, 1>):
                      %c33216 = arith.constant 33216 : index
                      %c33088 = arith.constant 33088 : index
                      %c4736_i32 = arith.constant 4736 : i32
                      %c32960 = arith.constant 32960 : index
                      %c4608_i32 = arith.constant 4608 : i32
                      %c512_i32_5 = arith.constant 512 : i32
                      %c32832 = arith.constant 32832 : index
                      %c1_i32_6 = arith.constant 1 : i32
                      %c0_i32_7 = arith.constant 0 : i32
                      %c64_i32_8 = arith.constant 64 : i32
                      %c15904 = arith.constant 15904 : index
                      %c2_i32 = arith.constant 2 : i32
                      %c0_i16 = arith.constant 0 : i16
                      %c128_i32_9 = arith.constant 128 : i32
                      %c256_i32_10 = arith.constant 256 : i32
                      %mask, %remaining_out = ascvec.update_mask %c256_i32_10 elem_bits 8 : !ascvec.mask
                      %mask_11, %remaining_out_12 = ascvec.update_mask %c128_i32_9 elem_bits 16 : !ascvec.mask
                      %145 = ascvec.duplicate_scalar %c0_i16, %mask_11 : i16 -> <ui16>
                      %146:2 = scf.for %arg39 = %c0_i32_7 to %c2_i32 step %c1_i32_6 iter_args(%arg40 = %145, %arg41 = %145) -> (!ascvec.vreg<ui16>, !ascvec.vreg<ui16>)  : i32 {
                        %148 = arith.muli %arg39, %c256_i32_10 overflow<nsw> : i32
                        %149 = arith.divsi %148, %c2_i32 : i32
                        %150 = arith.index_cast %149 : i32 to index
                        %151 = arith.addi %150, %c15904 : index
                        %152 = ascvec.loadalign %arg36[%151] : memref<?xf32, 1>, index -> <f32>
                        %153 = ascvec.reinterpret %152 : <f32> -> <ui16>
                        %154 = arith.addi %149, %c64_i32_8 overflow<nsw> : i32
                        %155 = arith.index_cast %154 : i32 to index
                        %156 = arith.addi %155, %c15904 : index
                        %157 = ascvec.loadalign %arg36[%156] : memref<?xf32, 1>, index -> <f32>
                        %158 = ascvec.reinterpret %157 : <f32> -> <ui16>
                        %159 = arith.index_cast %148 : i32 to index
                        ascvec.storealign %arg37[%159], %153, %mask_11 : memref<?xui16, 1>, index, <ui16>
                        %160 = arith.addi %148, %c128_i32_9 overflow<nsw> : i32
                        %161 = arith.index_cast %160 : i32 to index
                        ascvec.storealign %arg37[%161], %158, %mask_11 : memref<?xui16, 1>, index, <ui16>
                        %162 = ascvec.reinterpret %153 : <ui16> -> <ui8>
                        %163 = ascvec.reinterpret %158 : <ui16> -> <ui8>
                        %result0, %result1 = ascvec.deintlv %162, %163 : <ui8>, <ui8> -> <ui8>, <ui8>
                        %164 = ascvec.cumulative_histogram %arg40, %result1, %mask : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                        %165 = ascvec.cumulative_histogram %arg41, %result1, %mask {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                        scf.yield %164, %165 : !ascvec.vreg<ui16>, !ascvec.vreg<ui16>
                      }
                      %c0_13 = arith.constant 0 : index
                      %c1 = arith.constant 1 : index
                      scf.for %arg39 = %c0_13 to %c1 step %c1 {
                        ascvec.storealign %arg38[%c32832], %146#0, %mask_11 : memref<?xui16, 1>, index, <ui16>
                        ascvec.storealign %arg38[%c32960], %146#1, %mask_11 : memref<?xui16, 1>, index, <ui16>
                      }
                      %147:2 = scf.for %arg39 = %c0_i32_7 to %c2_i32 step %c1_i32_6 iter_args(%arg40 = %145, %arg41 = %145) -> (!ascvec.vreg<ui16>, !ascvec.vreg<ui16>)  : i32 {
                        %148 = arith.muli %arg39, %c256_i32_10 overflow<nsw> : i32
                        %149 = arith.addi %148, %c512_i32_5 overflow<nsw> : i32
                        %150 = arith.divsi %149, %c2_i32 : i32
                        %151 = arith.index_cast %150 : i32 to index
                        %152 = arith.addi %151, %c15904 : index
                        %153 = ascvec.loadalign %arg36[%152] : memref<?xf32, 1>, index -> <f32>
                        %154 = ascvec.reinterpret %153 : <f32> -> <ui16>
                        %155 = arith.addi %150, %c64_i32_8 overflow<nsw> : i32
                        %156 = arith.index_cast %155 : i32 to index
                        %157 = arith.addi %156, %c15904 : index
                        %158 = ascvec.loadalign %arg36[%157] : memref<?xf32, 1>, index -> <f32>
                        %159 = ascvec.reinterpret %158 : <f32> -> <ui16>
                        %160 = arith.addi %148, %c4608_i32 overflow<nsw> : i32
                        %161 = arith.index_cast %160 : i32 to index
                        ascvec.storealign %arg37[%161], %154, %mask_11 : memref<?xui16, 1>, index, <ui16>
                        %162 = arith.addi %148, %c4736_i32 overflow<nsw> : i32
                        %163 = arith.index_cast %162 : i32 to index
                        ascvec.storealign %arg37[%163], %159, %mask_11 : memref<?xui16, 1>, index, <ui16>
                        %164 = ascvec.reinterpret %154 : <ui16> -> <ui8>
                        %165 = ascvec.reinterpret %159 : <ui16> -> <ui8>
                        %result0, %result1 = ascvec.deintlv %164, %165 : <ui8>, <ui8> -> <ui8>, <ui8>
                        %166 = ascvec.cumulative_histogram %arg40, %result1, %mask : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                        %167 = ascvec.cumulative_histogram %arg41, %result1, %mask {bin = 1 : i32} : <ui16>, <ui8>, !ascvec.mask -> <ui16>
                        scf.yield %166, %167 : !ascvec.vreg<ui16>, !ascvec.vreg<ui16>
                      }
                      %c0_14 = arith.constant 0 : index
                      %c1_15 = arith.constant 1 : index
                      scf.for %arg39 = %c0_14 to %c1_15 step %c1_15 {
                        ascvec.storealign %arg38[%c33088], %147#0, %mask_11 : memref<?xui16, 1>, index, <ui16>
                        ascvec.storealign %arg38[%c33216], %147#1, %mask_11 : memref<?xui16, 1>, index, <ui16>
                      }
                    } {cannir.vf_mode = "simd"}
                  }
                }
                scf.yield %115, %113 : i64, i64
              }
              %91 = scf.if %88 -> (i64) {
                %92 = arith.remsi %71, %c2_i64 : i64
                %93 = scf.for %arg33 = %c0_i64 to %c2_i64 step %c1_i64 iter_args(%arg34 = %arg30) -> (i64)  : i64 {
                  %94 = arith.addi %79, %arg33 : i64
                  %95 = arith.cmpi slt, %94, %51 : i64
                  %96 = scf.if %95 -> (i64) {
                    %97 = arith.cmpi eq, %arg34, %c0_i64 : i64
                    %98 = arith.select %97, %c556032_i64, %c621824_i64 : i64
                    %99 = ascscalar.memref_address %alloc_4 : memref<262144xui8, 1>
                    %100 = arith.divui %98, %c8_i64 : i64
                    %101 = arith.addi %99, %100 : i64
                    %102 = ascscalar.bind_local_ptr %101 : memref<2056xi32, 1>
                    %103 = arith.select %97, %c1_i64, %c2_i64 : i64
                    %104 = arith.addi %arg34, %c1_i64 : i64
                    %105 = arith.remui %104, %c2_i64 : i64
                    %106 = arith.trunci %arg33 : i64 to i32
                    %107 = arith.muli %106, %c8_i32 overflow<nsw> : i32
                    %108 = arith.index_cast %107 : i32 to index
                    %109 = arith.muli %106, %c512_i32 overflow<nsw> : i32
                    %110 = arith.index_cast %109 : i32 to index
                    %111 = arith.addi %109, %c64_i32 overflow<nsw> : i32
                    %112 = arith.index_cast %111 : i32 to index
                    %113 = arith.addi %109, %c128_i32 overflow<nsw> : i32
                    %114 = arith.index_cast %113 : i32 to index
                    %115 = arith.addi %109, %c192_i32 overflow<nsw> : i32
                    %116 = arith.index_cast %115 : i32 to index
                    %117 = arith.addi %109, %c256_i32 overflow<nsw> : i32
                    %118 = arith.index_cast %117 : i32 to index
                    %119 = arith.addi %109, %c320_i32 overflow<nsw> : i32
                    %120 = arith.index_cast %119 : i32 to index
                    %121 = arith.addi %109, %c384_i32 overflow<nsw> : i32
                    %122 = arith.index_cast %121 : i32 to index
                    %123 = arith.addi %109, %c448_i32 overflow<nsw> : i32
                    %124 = arith.index_cast %123 : i32 to index
                    ascvec.get_buf v, %103, 0 {cannir.buffer.generated_sync}
                    %125 = arith.muli %92, %c1024_i64 overflow<nsw> : i64
                    %126 = arith.index_cast %125 : i64 to index
                    %127 = arith.addi %126, %c4608 : index
                    ascvec.vf(%87, %108, %87, %87, %1, %127, %110, %102, %112, %114, %116, %118, %120, %122, %124 : memref<?xi32, 1>, index, memref<?xi32, 1>, memref<?xi32, 1>, memref<?xui32, 1>, index, index, memref<2056xi32, 1>, index, index, index, index, index, index, index) {
                    ^bb0(%arg35: memref<?xi32, 1>, %arg36: index, %arg37: memref<?xi32, 1>, %arg38: memref<?xi32, 1>, %arg39: memref<?xui32, 1>, %arg40: index, %arg41: index, %arg42: memref<2056xi32, 1>, %arg43: index, %arg44: index, %arg45: index, %arg46: index, %arg47: index, %arg48: index, %arg49: index):
                      %c2048 = arith.constant 2048 : index
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
                      %c2051_i32 = arith.constant 2051 : i32
                      %c2048_i32 = arith.constant 2048 : i32
                      %c8_i32_5 = arith.constant 8 : i32
                      %c1984_i32 = arith.constant 1984 : i32
                      %c1920_i32 = arith.constant 1920 : i32
                      %c1856_i32 = arith.constant 1856 : i32
                      %c1792_i32 = arith.constant 1792 : i32
                      %c1728_i32 = arith.constant 1728 : i32
                      %c1664_i32 = arith.constant 1664 : i32
                      %c1600_i32 = arith.constant 1600 : i32
                      %c1536_i32 = arith.constant 1536 : i32
                      %c1472_i32 = arith.constant 1472 : i32
                      %c1408_i32 = arith.constant 1408 : i32
                      %c1344_i32 = arith.constant 1344 : i32
                      %c1280_i32 = arith.constant 1280 : i32
                      %c1216_i32 = arith.constant 1216 : i32
                      %c1152_i32 = arith.constant 1152 : i32
                      %c1088_i32 = arith.constant 1088 : i32
                      %c1024_i32 = arith.constant 1024 : i32
                      %c960_i32 = arith.constant 960 : i32
                      %c896_i32 = arith.constant 896 : i32
                      %c832_i32 = arith.constant 832 : i32
                      %c768_i32 = arith.constant 768 : i32
                      %c704_i32 = arith.constant 704 : i32
                      %c640_i32 = arith.constant 640 : i32
                      %c576_i32 = arith.constant 576 : i32
                      %c512_i32_6 = arith.constant 512 : i32
                      %c448_i32_7 = arith.constant 448 : i32
                      %c384_i32_8 = arith.constant 384 : i32
                      %c320_i32_9 = arith.constant 320 : i32
                      %c256_i32_10 = arith.constant 256 : i32
                      %c192_i32_11 = arith.constant 192 : i32
                      %c128_i32_12 = arith.constant 128 : i32
                      %c0_13 = arith.constant 0 : index
                      %c3_i32 = arith.constant 3 : i32
                      %c0_i32_14 = arith.constant 0 : i32
                      %c-1_i32 = arith.constant -1 : i32
                      %c17344 = arith.constant 17344 : index
                      %c17328 = arith.constant 17328 : index
                      %c17312 = arith.constant 17312 : index
                      %c64_i32_15 = arith.constant 64 : i32
                      %mask, %remaining_out = ascvec.update_mask %c64_i32_15 : !ascvec.mask
                      %135 = arith.addi %arg36, %c17312 : index
                      %136 = ascvec.loadalign_brc %arg35[%135] : memref<?xi32, 1>, index -> <i32>
                      %137 = arith.addi %arg36, %c17328 : index
                      %138 = ascvec.loadalign_brc %arg37[%137] : memref<?xi32, 1>, index -> <i32>
                      %139 = arith.addi %arg36, %c17344 : index
                      %140 = ascvec.loadalign_brc %arg38[%139] : memref<?xi32, 1>, index -> <i32>
                      %141 = ascvec.shl %138, 2, %mask : <i32> -> <i32>
                      %142 = ascvec.shl %136, 2, %mask : <i32> -> <i32>
                      %143 = ascvec.sub %140, %142, %mask : <i32>
                      %144 = ascvec.add %141, %143, %mask : <i32>
                      %145 = ascvec.duplicate_scalar %c-1_i32, %mask : i32 -> <i32>
                      %146 = ascvec.arange %c0_i32_14 : i32 -> <i32>
                      %147 = ascvec.duplicate_scalar %c3_i32, %mask : i32 -> <i32>
                      %148 = ascvec.and %146, %147, %mask : <i32>, <i32> -> <i32>
                      %149 = arith.addi %arg40, %arg41 : index
                      %c0_16 = arith.constant 0 : index
                      %c1 = arith.constant 1 : index
                      scf.for %arg50 = %c0_16 to %c1 step %c1 {
                        %157 = ascvec.loadalign %arg39[%149] : memref<?xui32, 1>, index -> <ui32>
                        %158 = ascvec.reinterpret %157 : <ui32> -> <i32>
                        %159 = ascvec.shl %158, 2, %mask : <i32> -> <i32>
                        %low, %high = ascvec.intlv %159, %159 : <i32>, <i32> -> <i32>, <i32>
                        %low_31, %high_32 = ascvec.intlv %low, %low : <i32>, <i32> -> <i32>, <i32>
                        %low_33, %high_34 = ascvec.intlv %high, %high : <i32>, <i32> -> <i32>, <i32>
                        %160 = ascvec.cmp_lt %146, %141, %mask : <i32>, <i32> -> !ascvec.mask
                        %161 = ascvec.add %low_31, %148, %mask : <i32>
                        %162 = ascvec.sub %146, %141, %mask : <i32>
                        %163 = ascvec.add %142, %162, %mask : <i32>
                        %164 = ascvec.select %161, %163, %160 : <i32>
                        %165 = ascvec.cmp_lt %146, %144, %mask : <i32>, <i32> -> !ascvec.mask
                        %166 = ascvec.select %164, %145, %165 : <i32>
                        ascvec.storealign %arg42[%c0_13], %166, %mask : memref<2056xi32, 1>, index, <i32>
                        %167 = ascvec.arange %c64_i32_15 : i32 -> <i32>
                        %168 = ascvec.cmp_lt %167, %141, %mask : <i32>, <i32> -> !ascvec.mask
                        %169 = ascvec.add %high_32, %148, %mask : <i32>
                        %170 = ascvec.sub %167, %141, %mask : <i32>
                        %171 = ascvec.add %142, %170, %mask : <i32>
                        %172 = ascvec.select %169, %171, %168 : <i32>
                        %173 = ascvec.cmp_lt %167, %144, %mask : <i32>, <i32> -> !ascvec.mask
                        %174 = ascvec.select %172, %145, %173 : <i32>
                        ascvec.storealign %arg42[%c64], %174, %mask : memref<2056xi32, 1>, index, <i32>
                        %175 = ascvec.arange %c128_i32_12 : i32 -> <i32>
                        %176 = ascvec.cmp_lt %175, %141, %mask : <i32>, <i32> -> !ascvec.mask
                        %177 = ascvec.add %low_33, %148, %mask : <i32>
                        %178 = ascvec.sub %175, %141, %mask : <i32>
                        %179 = ascvec.add %142, %178, %mask : <i32>
                        %180 = ascvec.select %177, %179, %176 : <i32>
                        %181 = ascvec.cmp_lt %175, %144, %mask : <i32>, <i32> -> !ascvec.mask
                        %182 = ascvec.select %180, %145, %181 : <i32>
                        ascvec.storealign %arg42[%c128], %182, %mask : memref<2056xi32, 1>, index, <i32>
                        %183 = ascvec.arange %c192_i32_11 : i32 -> <i32>
                        %184 = ascvec.cmp_lt %183, %141, %mask : <i32>, <i32> -> !ascvec.mask
                        %185 = ascvec.add %high_34, %148, %mask : <i32>
                        %186 = ascvec.sub %183, %141, %mask : <i32>
                        %187 = ascvec.add %142, %186, %mask : <i32>
                        %188 = ascvec.select %185, %187, %184 : <i32>
                        %189 = ascvec.cmp_lt %183, %144, %mask : <i32>, <i32> -> !ascvec.mask
                        %190 = ascvec.select %188, %145, %189 : <i32>
                        ascvec.storealign %arg42[%c192], %190, %mask : memref<2056xi32, 1>, index, <i32>
                      }
                      %150 = arith.addi %arg40, %arg43 : index
                      %c0_17 = arith.constant 0 : index
                      %c1_18 = arith.constant 1 : index
                      scf.for %arg50 = %c0_17 to %c1_18 step %c1_18 {
                        %157 = ascvec.loadalign %arg39[%150] : memref<?xui32, 1>, index -> <ui32>
                        %158 = ascvec.reinterpret %157 : <ui32> -> <i32>
                        %159 = ascvec.shl %158, 2, %mask : <i32> -> <i32>
                        %low, %high = ascvec.intlv %159, %159 : <i32>, <i32> -> <i32>, <i32>
                        %low_31, %high_32 = ascvec.intlv %low, %low : <i32>, <i32> -> <i32>, <i32>
                        %low_33, %high_34 = ascvec.intlv %high, %high : <i32>, <i32> -> <i32>, <i32>
                        %160 = ascvec.arange %c256_i32_10 : i32 -> <i32>
                        %161 = ascvec.cmp_lt %160, %141, %mask : <i32>, <i32> -> !ascvec.mask
                        %162 = ascvec.add %low_31, %148, %mask : <i32>
                        %163 = ascvec.sub %160, %141, %mask : <i32>
                        %164 = ascvec.add %142, %163, %mask : <i32>
                        %165 = ascvec.select %162, %164, %161 : <i32>
                        %166 = ascvec.cmp_lt %160, %144, %mask : <i32>, <i32> -> !ascvec.mask
                        %167 = ascvec.select %165, %145, %166 : <i32>
                        ascvec.storealign %arg42[%c256], %167, %mask : memref<2056xi32, 1>, index, <i32>
                        %168 = ascvec.arange %c320_i32_9 : i32 -> <i32>
                        %169 = ascvec.cmp_lt %168, %141, %mask : <i32>, <i32> -> !ascvec.mask
                        %170 = ascvec.add %high_32, %148, %mask : <i32>
                        %171 = ascvec.sub %168, %141, %mask : <i32>
                        %172 = ascvec.add %142, %171, %mask : <i32>
                        %173 = ascvec.select %170, %172, %169 : <i32>
                        %174 = ascvec.cmp_lt %168, %144, %mask : <i32>, <i32> -> !ascvec.mask
                        %175 = ascvec.select %173, %145, %174 : <i32>
                        ascvec.storealign %arg42[%c320], %175, %mask : memref<2056xi32, 1>, index, <i32>
                        %176 = ascvec.arange %c384_i32_8 : i32 -> <i32>
                        %177 = ascvec.cmp_lt %176, %141, %mask : <i32>, <i32> -> !ascvec.mask
                        %178 = ascvec.add %low_33, %148, %mask : <i32>
                        %179 = ascvec.sub %176, %141, %mask : <i32>
                        %180 = ascvec.add %142, %179, %mask : <i32>
                        %181 = ascvec.select %178, %180, %177 : <i32>
                        %182 = ascvec.cmp_lt %176, %144, %mask : <i32>, <i32> -> !ascvec.mask
                        %183 = ascvec.select %181, %145, %182 : <i32>
                        ascvec.storealign %arg42[%c384], %183, %mask : memref<2056xi32, 1>, index, <i32>
                        %184 = ascvec.arange %c448_i32_7 : i32 -> <i32>
                        %185 = ascvec.cmp_lt %184, %141, %mask : <i32>, <i32> -> !ascvec.mask
                        %186 = ascvec.add %high_34, %148, %mask : <i32>
                        %187 = ascvec.sub %184, %141, %mask : <i32>
                        %188 = ascvec.add %142, %187, %mask : <i32>
                        %189 = ascvec.select %186, %188, %185 : <i32>
                        %190 = ascvec.cmp_lt %184, %144, %mask : <i32>, <i32> -> !ascvec.mask
                        %191 = ascvec.select %189, %145, %190 : <i32>
                        ascvec.storealign %arg42[%c448], %191, %mask : memref<2056xi32, 1>, index, <i32>
                      }
                      %151 = arith.addi %arg40, %arg44 : index
                      %c0_19 = arith.constant 0 : index
                      %c1_20 = arith.constant 1 : index
                      scf.for %arg50 = %c0_19 to %c1_20 step %c1_20 {
                        %157 = ascvec.loadalign %arg39[%151] : memref<?xui32, 1>, index -> <ui32>
                        %158 = ascvec.reinterpret %157 : <ui32> -> <i32>
                        %159 = ascvec.shl %158, 2, %mask : <i32> -> <i32>
                        %low, %high = ascvec.intlv %159, %159 : <i32>, <i32> -> <i32>, <i32>
                        %low_31, %high_32 = ascvec.intlv %low, %low : <i32>, <i32> -> <i32>, <i32>
                        %low_33, %high_34 = ascvec.intlv %high, %high : <i32>, <i32> -> <i32>, <i32>
                        %160 = ascvec.arange %c512_i32_6 : i32 -> <i32>
                        %161 = ascvec.cmp_lt %160, %141, %mask : <i32>, <i32> -> !ascvec.mask
                        %162 = ascvec.add %low_31, %148, %mask : <i32>
                        %163 = ascvec.sub %160, %141, %mask : <i32>
                        %164 = ascvec.add %142, %163, %mask : <i32>
                        %165 = ascvec.select %162, %164, %161 : <i32>
                        %166 = ascvec.cmp_lt %160, %144, %mask : <i32>, <i32> -> !ascvec.mask
                        %167 = ascvec.select %165, %145, %166 : <i32>
                        ascvec.storealign %arg42[%c512], %167, %mask : memref<2056xi32, 1>, index, <i32>
                        %168 = ascvec.arange %c576_i32 : i32 -> <i32>
                        %169 = ascvec.cmp_lt %168, %141, %mask : <i32>, <i32> -> !ascvec.mask
                        %170 = ascvec.add %high_32, %148, %mask : <i32>
                        %171 = ascvec.sub %168, %141, %mask : <i32>
                        %172 = ascvec.add %142, %171, %mask : <i32>
                        %173 = ascvec.select %170, %172, %169 : <i32>
                        %174 = ascvec.cmp_lt %168, %144, %mask : <i32>, <i32> -> !ascvec.mask
                        %175 = ascvec.select %173, %145, %174 : <i32>
                        ascvec.storealign %arg42[%c576], %175, %mask : memref<2056xi32, 1>, index, <i32>
                        %176 = ascvec.arange %c640_i32 : i32 -> <i32>
                        %177 = ascvec.cmp_lt %176, %141, %mask : <i32>, <i32> -> !ascvec.mask
                        %178 = ascvec.add %low_33, %148, %mask : <i32>
                        %179 = ascvec.sub %176, %141, %mask : <i32>
                        %180 = ascvec.add %142, %179, %mask : <i32>
                        %181 = ascvec.select %178, %180, %177 : <i32>
                        %182 = ascvec.cmp_lt %176, %144, %mask : <i32>, <i32> -> !ascvec.mask
                        %183 = ascvec.select %181, %145, %182 : <i32>
                        ascvec.storealign %arg42[%c640], %183, %mask : memref<2056xi32, 1>, index, <i32>
                        %184 = ascvec.arange %c704_i32 : i32 -> <i32>
                        %185 = ascvec.cmp_lt %184, %141, %mask : <i32>, <i32> -> !ascvec.mask
                        %186 = ascvec.add %high_34, %148, %mask : <i32>
                        %187 = ascvec.sub %184, %141, %mask : <i32>
                        %188 = ascvec.add %142, %187, %mask : <i32>
                        %189 = ascvec.select %186, %188, %185 : <i32>
                        %190 = ascvec.cmp_lt %184, %144, %mask : <i32>, <i32> -> !ascvec.mask
                        %191 = ascvec.select %189, %145, %190 : <i32>
                        ascvec.storealign %arg42[%c704], %191, %mask : memref<2056xi32, 1>, index, <i32>
                      }
                      %152 = arith.addi %arg40, %arg45 : index
                      %c0_21 = arith.constant 0 : index
                      %c1_22 = arith.constant 1 : index
                      scf.for %arg50 = %c0_21 to %c1_22 step %c1_22 {
                        %157 = ascvec.loadalign %arg39[%152] : memref<?xui32, 1>, index -> <ui32>
                        %158 = ascvec.reinterpret %157 : <ui32> -> <i32>
                        %159 = ascvec.shl %158, 2, %mask : <i32> -> <i32>
                        %low, %high = ascvec.intlv %159, %159 : <i32>, <i32> -> <i32>, <i32>
                        %low_31, %high_32 = ascvec.intlv %low, %low : <i32>, <i32> -> <i32>, <i32>
                        %low_33, %high_34 = ascvec.intlv %high, %high : <i32>, <i32> -> <i32>, <i32>
                        %160 = ascvec.arange %c768_i32 : i32 -> <i32>
                        %161 = ascvec.cmp_lt %160, %141, %mask : <i32>, <i32> -> !ascvec.mask
                        %162 = ascvec.add %low_31, %148, %mask : <i32>
                        %163 = ascvec.sub %160, %141, %mask : <i32>
                        %164 = ascvec.add %142, %163, %mask : <i32>
                        %165 = ascvec.select %162, %164, %161 : <i32>
                        %166 = ascvec.cmp_lt %160, %144, %mask : <i32>, <i32> -> !ascvec.mask
                        %167 = ascvec.select %165, %145, %166 : <i32>
                        ascvec.storealign %arg42[%c768], %167, %mask : memref<2056xi32, 1>, index, <i32>
                        %168 = ascvec.arange %c832_i32 : i32 -> <i32>
                        %169 = ascvec.cmp_lt %168, %141, %mask : <i32>, <i32> -> !ascvec.mask
                        %170 = ascvec.add %high_32, %148, %mask : <i32>
                        %171 = ascvec.sub %168, %141, %mask : <i32>
                        %172 = ascvec.add %142, %171, %mask : <i32>
                        %173 = ascvec.select %170, %172, %169 : <i32>
                        %174 = ascvec.cmp_lt %168, %144, %mask : <i32>, <i32> -> !ascvec.mask
                        %175 = ascvec.select %173, %145, %174 : <i32>
                        ascvec.storealign %arg42[%c832], %175, %mask : memref<2056xi32, 1>, index, <i32>
                        %176 = ascvec.arange %c896_i32 : i32 -> <i32>
                        %177 = ascvec.cmp_lt %176, %141, %mask : <i32>, <i32> -> !ascvec.mask
                        %178 = ascvec.add %low_33, %148, %mask : <i32>
                        %179 = ascvec.sub %176, %141, %mask : <i32>
                        %180 = ascvec.add %142, %179, %mask : <i32>
                        %181 = ascvec.select %178, %180, %177 : <i32>
                        %182 = ascvec.cmp_lt %176, %144, %mask : <i32>, <i32> -> !ascvec.mask
                        %183 = ascvec.select %181, %145, %182 : <i32>
                        ascvec.storealign %arg42[%c896], %183, %mask : memref<2056xi32, 1>, index, <i32>
                        %184 = ascvec.arange %c960_i32 : i32 -> <i32>
                        %185 = ascvec.cmp_lt %184, %141, %mask : <i32>, <i32> -> !ascvec.mask
                        %186 = ascvec.add %high_34, %148, %mask : <i32>
                        %187 = ascvec.sub %184, %141, %mask : <i32>
                        %188 = ascvec.add %142, %187, %mask : <i32>
                        %189 = ascvec.select %186, %188, %185 : <i32>
                        %190 = ascvec.cmp_lt %184, %144, %mask : <i32>, <i32> -> !ascvec.mask
                        %191 = ascvec.select %189, %145, %190 : <i32>
                        ascvec.storealign %arg42[%c960], %191, %mask : memref<2056xi32, 1>, index, <i32>
                      }
                      %153 = arith.addi %arg40, %arg46 : index
                      %c0_23 = arith.constant 0 : index
                      %c1_24 = arith.constant 1 : index
                      scf.for %arg50 = %c0_23 to %c1_24 step %c1_24 {
                        %157 = ascvec.loadalign %arg39[%153] : memref<?xui32, 1>, index -> <ui32>
                        %158 = ascvec.reinterpret %157 : <ui32> -> <i32>
                        %159 = ascvec.shl %158, 2, %mask : <i32> -> <i32>
                        %low, %high = ascvec.intlv %159, %159 : <i32>, <i32> -> <i32>, <i32>
                        %low_31, %high_32 = ascvec.intlv %low, %low : <i32>, <i32> -> <i32>, <i32>
                        %low_33, %high_34 = ascvec.intlv %high, %high : <i32>, <i32> -> <i32>, <i32>
                        %160 = ascvec.arange %c1024_i32 : i32 -> <i32>
                        %161 = ascvec.cmp_lt %160, %141, %mask : <i32>, <i32> -> !ascvec.mask
                        %162 = ascvec.add %low_31, %148, %mask : <i32>
                        %163 = ascvec.sub %160, %141, %mask : <i32>
                        %164 = ascvec.add %142, %163, %mask : <i32>
                        %165 = ascvec.select %162, %164, %161 : <i32>
                        %166 = ascvec.cmp_lt %160, %144, %mask : <i32>, <i32> -> !ascvec.mask
                        %167 = ascvec.select %165, %145, %166 : <i32>
                        ascvec.storealign %arg42[%c1024], %167, %mask : memref<2056xi32, 1>, index, <i32>
                        %168 = ascvec.arange %c1088_i32 : i32 -> <i32>
                        %169 = ascvec.cmp_lt %168, %141, %mask : <i32>, <i32> -> !ascvec.mask
                        %170 = ascvec.add %high_32, %148, %mask : <i32>
                        %171 = ascvec.sub %168, %141, %mask : <i32>
                        %172 = ascvec.add %142, %171, %mask : <i32>
                        %173 = ascvec.select %170, %172, %169 : <i32>
                        %174 = ascvec.cmp_lt %168, %144, %mask : <i32>, <i32> -> !ascvec.mask
                        %175 = ascvec.select %173, %145, %174 : <i32>
                        ascvec.storealign %arg42[%c1088], %175, %mask : memref<2056xi32, 1>, index, <i32>
                        %176 = ascvec.arange %c1152_i32 : i32 -> <i32>
                        %177 = ascvec.cmp_lt %176, %141, %mask : <i32>, <i32> -> !ascvec.mask
                        %178 = ascvec.add %low_33, %148, %mask : <i32>
                        %179 = ascvec.sub %176, %141, %mask : <i32>
                        %180 = ascvec.add %142, %179, %mask : <i32>
                        %181 = ascvec.select %178, %180, %177 : <i32>
                        %182 = ascvec.cmp_lt %176, %144, %mask : <i32>, <i32> -> !ascvec.mask
                        %183 = ascvec.select %181, %145, %182 : <i32>
                        ascvec.storealign %arg42[%c1152], %183, %mask : memref<2056xi32, 1>, index, <i32>
                        %184 = ascvec.arange %c1216_i32 : i32 -> <i32>
                        %185 = ascvec.cmp_lt %184, %141, %mask : <i32>, <i32> -> !ascvec.mask
                        %186 = ascvec.add %high_34, %148, %mask : <i32>
                        %187 = ascvec.sub %184, %141, %mask : <i32>
                        %188 = ascvec.add %142, %187, %mask : <i32>
                        %189 = ascvec.select %186, %188, %185 : <i32>
                        %190 = ascvec.cmp_lt %184, %144, %mask : <i32>, <i32> -> !ascvec.mask
                        %191 = ascvec.select %189, %145, %190 : <i32>
                        ascvec.storealign %arg42[%c1216], %191, %mask : memref<2056xi32, 1>, index, <i32>
                      }
                      %154 = arith.addi %arg40, %arg47 : index
                      %c0_25 = arith.constant 0 : index
                      %c1_26 = arith.constant 1 : index
                      scf.for %arg50 = %c0_25 to %c1_26 step %c1_26 {
                        %157 = ascvec.loadalign %arg39[%154] : memref<?xui32, 1>, index -> <ui32>
                        %158 = ascvec.reinterpret %157 : <ui32> -> <i32>
                        %159 = ascvec.shl %158, 2, %mask : <i32> -> <i32>
                        %low, %high = ascvec.intlv %159, %159 : <i32>, <i32> -> <i32>, <i32>
                        %low_31, %high_32 = ascvec.intlv %low, %low : <i32>, <i32> -> <i32>, <i32>
                        %low_33, %high_34 = ascvec.intlv %high, %high : <i32>, <i32> -> <i32>, <i32>
                        %160 = ascvec.arange %c1280_i32 : i32 -> <i32>
                        %161 = ascvec.cmp_lt %160, %141, %mask : <i32>, <i32> -> !ascvec.mask
                        %162 = ascvec.add %low_31, %148, %mask : <i32>
                        %163 = ascvec.sub %160, %141, %mask : <i32>
                        %164 = ascvec.add %142, %163, %mask : <i32>
                        %165 = ascvec.select %162, %164, %161 : <i32>
                        %166 = ascvec.cmp_lt %160, %144, %mask : <i32>, <i32> -> !ascvec.mask
                        %167 = ascvec.select %165, %145, %166 : <i32>
                        ascvec.storealign %arg42[%c1280], %167, %mask : memref<2056xi32, 1>, index, <i32>
                        %168 = ascvec.arange %c1344_i32 : i32 -> <i32>
                        %169 = ascvec.cmp_lt %168, %141, %mask : <i32>, <i32> -> !ascvec.mask
                        %170 = ascvec.add %high_32, %148, %mask : <i32>
                        %171 = ascvec.sub %168, %141, %mask : <i32>
                        %172 = ascvec.add %142, %171, %mask : <i32>
                        %173 = ascvec.select %170, %172, %169 : <i32>
                        %174 = ascvec.cmp_lt %168, %144, %mask : <i32>, <i32> -> !ascvec.mask
                        %175 = ascvec.select %173, %145, %174 : <i32>
                        ascvec.storealign %arg42[%c1344], %175, %mask : memref<2056xi32, 1>, index, <i32>
                        %176 = ascvec.arange %c1408_i32 : i32 -> <i32>
                        %177 = ascvec.cmp_lt %176, %141, %mask : <i32>, <i32> -> !ascvec.mask
                        %178 = ascvec.add %low_33, %148, %mask : <i32>
                        %179 = ascvec.sub %176, %141, %mask : <i32>
                        %180 = ascvec.add %142, %179, %mask : <i32>
                        %181 = ascvec.select %178, %180, %177 : <i32>
                        %182 = ascvec.cmp_lt %176, %144, %mask : <i32>, <i32> -> !ascvec.mask
                        %183 = ascvec.select %181, %145, %182 : <i32>
                        ascvec.storealign %arg42[%c1408], %183, %mask : memref<2056xi32, 1>, index, <i32>
                        %184 = ascvec.arange %c1472_i32 : i32 -> <i32>
                        %185 = ascvec.cmp_lt %184, %141, %mask : <i32>, <i32> -> !ascvec.mask
                        %186 = ascvec.add %high_34, %148, %mask : <i32>
                        %187 = ascvec.sub %184, %141, %mask : <i32>
                        %188 = ascvec.add %142, %187, %mask : <i32>
                        %189 = ascvec.select %186, %188, %185 : <i32>
                        %190 = ascvec.cmp_lt %184, %144, %mask : <i32>, <i32> -> !ascvec.mask
                        %191 = ascvec.select %189, %145, %190 : <i32>
                        ascvec.storealign %arg42[%c1472], %191, %mask : memref<2056xi32, 1>, index, <i32>
                      }
                      %155 = arith.addi %arg40, %arg48 : index
                      %c0_27 = arith.constant 0 : index
                      %c1_28 = arith.constant 1 : index
                      scf.for %arg50 = %c0_27 to %c1_28 step %c1_28 {
                        %157 = ascvec.loadalign %arg39[%155] : memref<?xui32, 1>, index -> <ui32>
                        %158 = ascvec.reinterpret %157 : <ui32> -> <i32>
                        %159 = ascvec.shl %158, 2, %mask : <i32> -> <i32>
                        %low, %high = ascvec.intlv %159, %159 : <i32>, <i32> -> <i32>, <i32>
                        %low_31, %high_32 = ascvec.intlv %low, %low : <i32>, <i32> -> <i32>, <i32>
                        %low_33, %high_34 = ascvec.intlv %high, %high : <i32>, <i32> -> <i32>, <i32>
                        %160 = ascvec.arange %c1536_i32 : i32 -> <i32>
                        %161 = ascvec.cmp_lt %160, %141, %mask : <i32>, <i32> -> !ascvec.mask
                        %162 = ascvec.add %low_31, %148, %mask : <i32>
                        %163 = ascvec.sub %160, %141, %mask : <i32>
                        %164 = ascvec.add %142, %163, %mask : <i32>
                        %165 = ascvec.select %162, %164, %161 : <i32>
                        %166 = ascvec.cmp_lt %160, %144, %mask : <i32>, <i32> -> !ascvec.mask
                        %167 = ascvec.select %165, %145, %166 : <i32>
                        ascvec.storealign %arg42[%c1536], %167, %mask : memref<2056xi32, 1>, index, <i32>
                        %168 = ascvec.arange %c1600_i32 : i32 -> <i32>
                        %169 = ascvec.cmp_lt %168, %141, %mask : <i32>, <i32> -> !ascvec.mask
                        %170 = ascvec.add %high_32, %148, %mask : <i32>
                        %171 = ascvec.sub %168, %141, %mask : <i32>
                        %172 = ascvec.add %142, %171, %mask : <i32>
                        %173 = ascvec.select %170, %172, %169 : <i32>
                        %174 = ascvec.cmp_lt %168, %144, %mask : <i32>, <i32> -> !ascvec.mask
                        %175 = ascvec.select %173, %145, %174 : <i32>
                        ascvec.storealign %arg42[%c1600], %175, %mask : memref<2056xi32, 1>, index, <i32>
                        %176 = ascvec.arange %c1664_i32 : i32 -> <i32>
                        %177 = ascvec.cmp_lt %176, %141, %mask : <i32>, <i32> -> !ascvec.mask
                        %178 = ascvec.add %low_33, %148, %mask : <i32>
                        %179 = ascvec.sub %176, %141, %mask : <i32>
                        %180 = ascvec.add %142, %179, %mask : <i32>
                        %181 = ascvec.select %178, %180, %177 : <i32>
                        %182 = ascvec.cmp_lt %176, %144, %mask : <i32>, <i32> -> !ascvec.mask
                        %183 = ascvec.select %181, %145, %182 : <i32>
                        ascvec.storealign %arg42[%c1664], %183, %mask : memref<2056xi32, 1>, index, <i32>
                        %184 = ascvec.arange %c1728_i32 : i32 -> <i32>
                        %185 = ascvec.cmp_lt %184, %141, %mask : <i32>, <i32> -> !ascvec.mask
                        %186 = ascvec.add %high_34, %148, %mask : <i32>
                        %187 = ascvec.sub %184, %141, %mask : <i32>
                        %188 = ascvec.add %142, %187, %mask : <i32>
                        %189 = ascvec.select %186, %188, %185 : <i32>
                        %190 = ascvec.cmp_lt %184, %144, %mask : <i32>, <i32> -> !ascvec.mask
                        %191 = ascvec.select %189, %145, %190 : <i32>
                        ascvec.storealign %arg42[%c1728], %191, %mask : memref<2056xi32, 1>, index, <i32>
                      }
                      %156 = arith.addi %arg40, %arg49 : index
                      %c0_29 = arith.constant 0 : index
                      %c1_30 = arith.constant 1 : index
                      scf.for %arg50 = %c0_29 to %c1_30 step %c1_30 {
                        %157 = ascvec.loadalign %arg39[%156] : memref<?xui32, 1>, index -> <ui32>
                        %158 = ascvec.reinterpret %157 : <ui32> -> <i32>
                        %159 = ascvec.shl %158, 2, %mask : <i32> -> <i32>
                        %low, %high = ascvec.intlv %159, %159 : <i32>, <i32> -> <i32>, <i32>
                        %low_31, %high_32 = ascvec.intlv %low, %low : <i32>, <i32> -> <i32>, <i32>
                        %low_33, %high_34 = ascvec.intlv %high, %high : <i32>, <i32> -> <i32>, <i32>
                        %160 = ascvec.arange %c1792_i32 : i32 -> <i32>
                        %161 = ascvec.cmp_lt %160, %141, %mask : <i32>, <i32> -> !ascvec.mask
                        %162 = ascvec.add %low_31, %148, %mask : <i32>
                        %163 = ascvec.sub %160, %141, %mask : <i32>
                        %164 = ascvec.add %142, %163, %mask : <i32>
                        %165 = ascvec.select %162, %164, %161 : <i32>
                        %166 = ascvec.cmp_lt %160, %144, %mask : <i32>, <i32> -> !ascvec.mask
                        %167 = ascvec.select %165, %145, %166 : <i32>
                        ascvec.storealign %arg42[%c1792], %167, %mask : memref<2056xi32, 1>, index, <i32>
                        %168 = ascvec.arange %c1856_i32 : i32 -> <i32>
                        %169 = ascvec.cmp_lt %168, %141, %mask : <i32>, <i32> -> !ascvec.mask
                        %170 = ascvec.add %high_32, %148, %mask : <i32>
                        %171 = ascvec.sub %168, %141, %mask : <i32>
                        %172 = ascvec.add %142, %171, %mask : <i32>
                        %173 = ascvec.select %170, %172, %169 : <i32>
                        %174 = ascvec.cmp_lt %168, %144, %mask : <i32>, <i32> -> !ascvec.mask
                        %175 = ascvec.select %173, %145, %174 : <i32>
                        ascvec.storealign %arg42[%c1856], %175, %mask : memref<2056xi32, 1>, index, <i32>
                        %176 = ascvec.arange %c1920_i32 : i32 -> <i32>
                        %177 = ascvec.cmp_lt %176, %141, %mask : <i32>, <i32> -> !ascvec.mask
                        %178 = ascvec.add %low_33, %148, %mask : <i32>
                        %179 = ascvec.sub %176, %141, %mask : <i32>
                        %180 = ascvec.add %142, %179, %mask : <i32>
                        %181 = ascvec.select %178, %180, %177 : <i32>
                        %182 = ascvec.cmp_lt %176, %144, %mask : <i32>, <i32> -> !ascvec.mask
                        %183 = ascvec.select %181, %145, %182 : <i32>
                        ascvec.storealign %arg42[%c1920], %183, %mask : memref<2056xi32, 1>, index, <i32>
                        %184 = ascvec.arange %c1984_i32 : i32 -> <i32>
                        %185 = ascvec.cmp_lt %184, %141, %mask : <i32>, <i32> -> !ascvec.mask
                        %186 = ascvec.add %high_34, %148, %mask : <i32>
                        %187 = ascvec.sub %184, %141, %mask : <i32>
                        %188 = ascvec.add %142, %187, %mask : <i32>
                        %189 = ascvec.select %186, %188, %185 : <i32>
                        %190 = ascvec.cmp_lt %184, %144, %mask : <i32>, <i32> -> !ascvec.mask
                        %191 = ascvec.select %189, %145, %190 : <i32>
                        ascvec.storealign %arg42[%c1984], %191, %mask : memref<2056xi32, 1>, index, <i32>
                        %mask_35, %remaining_out_36 = ascvec.update_mask %c8_i32_5 : !ascvec.mask
                        %192 = ascvec.arange %c2048_i32 : i32 -> <i32>
                        %193 = ascvec.sub %192, %141, %mask : <i32>
                        %194 = ascvec.add %142, %193, %mask : <i32>
                        %195 = ascvec.cmp_lt %192, %144, %mask_35 : <i32>, <i32> -> !ascvec.mask
                        %196 = ascvec.select %194, %145, %195 : <i32>
                        %197 = ascvec.cmp_eq_scalar %192, %c2051_i32, %mask_35 : <i32>, i32 -> !ascvec.mask
                        %198 = ascvec.select %144, %196, %197 : <i32>
                        ascvec.storealign %arg42[%c2048], %198, %mask_35 : memref<2056xi32, 1>, index, <i32>
                      }
                    } {cannir.vf_mode = "simd"}
                    ascvec.release_buf v, %103, 0 {cannir.buffer.generated_sync}
                    %128 = arith.addi %49, %94 : i64
                    %129 = arith.muli %128, %c65792_i64 : i64
                    %130 = ascscalar.memref_address %alloc_2 : memref<1xi32>
                    %131 = arith.divui %129, %c8_i64 : i64
                    %132 = arith.addi %130, %131 : i64
                    %133 = ascscalar.bind_gm_ptr %132 : memref<1xi32>
                    %134 = ascscalar.bind_local_ptr %101 : memref<1xi32, 1>
                    ascvec.get_buf mte3, %103, 0 {cannir.buffer.generated_sync}
                    ascvec.copy_ub2gm %133[%c0], %134[%c0], %c1_i32, %c8224_i32, %c0_i64, %c8224_i32 l2_cache_ctl %c0_i32 : memref<1xi32>, memref<1xi32, 1>
                    ascvec.release_buf mte3, %103, 0 {cannir.buffer.generated_sync}
                    scf.yield %105 : i64
                  } else {
                    scf.yield %arg34 : i64
                  }
                  scf.yield %96 : i64
                }
                scf.yield %93 : i64
              } else {
                %92 = scf.for %arg33 = %c0_i64 to %c2_i64 step %c1_i64 iter_args(%arg34 = %arg30) -> (i64)  : i64 {
                  %93 = arith.addi %79, %arg33 : i64
                  %94 = arith.cmpi slt, %93, %51 : i64
                  %95 = scf.if %94 -> (i64) {
                    %96 = arith.cmpi eq, %arg34, %c0_i64 : i64
                    %97 = arith.select %96, %c556032_i64, %c621824_i64 : i64
                    %98 = ascscalar.memref_address %alloc_4 : memref<262144xui8, 1>
                    %99 = arith.divui %97, %c8_i64 : i64
                    %100 = arith.addi %98, %99 : i64
                    %101 = ascscalar.bind_local_ptr %100 : memref<2056xi32, 1>
                    %102 = arith.select %96, %c1_i64, %c2_i64 : i64
                    %103 = arith.addi %arg34, %c1_i64 : i64
                    %104 = arith.remui %103, %c2_i64 : i64
                    %105 = arith.trunci %arg33 : i64 to i32
                    %106 = arith.muli %105, %c8_i32 overflow<nsw> : i32
                    %107 = arith.index_cast %106 : i32 to index
                    %108 = arith.muli %105, %c512_i32 overflow<nsw> : i32
                    %109 = arith.index_cast %108 : i32 to index
                    %110 = arith.addi %108, %c64_i32 overflow<nsw> : i32
                    %111 = arith.index_cast %110 : i32 to index
                    %112 = arith.addi %108, %c128_i32 overflow<nsw> : i32
                    %113 = arith.index_cast %112 : i32 to index
                    %114 = arith.addi %108, %c192_i32 overflow<nsw> : i32
                    %115 = arith.index_cast %114 : i32 to index
                    %116 = arith.addi %108, %c256_i32 overflow<nsw> : i32
                    %117 = arith.index_cast %116 : i32 to index
                    %118 = arith.addi %108, %c320_i32 overflow<nsw> : i32
                    %119 = arith.index_cast %118 : i32 to index
                    %120 = arith.addi %108, %c384_i32 overflow<nsw> : i32
                    %121 = arith.index_cast %120 : i32 to index
                    %122 = arith.addi %108, %c448_i32 overflow<nsw> : i32
                    %123 = arith.index_cast %122 : i32 to index
                    ascvec.get_buf v, %102, 0 {cannir.buffer.generated_sync}
                    ascvec.vf(%87, %107, %87, %87, %1, %109, %101, %111, %113, %115, %117, %119, %121, %123 : memref<?xi32, 1>, index, memref<?xi32, 1>, memref<?xi32, 1>, memref<?xui32, 1>, index, memref<2056xi32, 1>, index, index, index, index, index, index, index) {
                    ^bb0(%arg35: memref<?xi32, 1>, %arg36: index, %arg37: memref<?xi32, 1>, %arg38: memref<?xi32, 1>, %arg39: memref<?xui32, 1>, %arg40: index, %arg41: memref<2056xi32, 1>, %arg42: index, %arg43: index, %arg44: index, %arg45: index, %arg46: index, %arg47: index, %arg48: index):
                      %c2048 = arith.constant 2048 : index
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
                      %c2051_i32 = arith.constant 2051 : i32
                      %c2048_i32 = arith.constant 2048 : i32
                      %c8_i32_5 = arith.constant 8 : i32
                      %c1984_i32 = arith.constant 1984 : i32
                      %c1920_i32 = arith.constant 1920 : i32
                      %c1856_i32 = arith.constant 1856 : i32
                      %c1792_i32 = arith.constant 1792 : i32
                      %c1728_i32 = arith.constant 1728 : i32
                      %c1664_i32 = arith.constant 1664 : i32
                      %c1600_i32 = arith.constant 1600 : i32
                      %c1536_i32 = arith.constant 1536 : i32
                      %c1472_i32 = arith.constant 1472 : i32
                      %c1408_i32 = arith.constant 1408 : i32
                      %c1344_i32 = arith.constant 1344 : i32
                      %c1280_i32 = arith.constant 1280 : i32
                      %c1216_i32 = arith.constant 1216 : i32
                      %c1152_i32 = arith.constant 1152 : i32
                      %c1088_i32 = arith.constant 1088 : i32
                      %c1024_i32 = arith.constant 1024 : i32
                      %c960_i32 = arith.constant 960 : i32
                      %c896_i32 = arith.constant 896 : i32
                      %c832_i32 = arith.constant 832 : i32
                      %c768_i32 = arith.constant 768 : i32
                      %c704_i32 = arith.constant 704 : i32
                      %c640_i32 = arith.constant 640 : i32
                      %c576_i32 = arith.constant 576 : i32
                      %c512_i32_6 = arith.constant 512 : i32
                      %c448_i32_7 = arith.constant 448 : i32
                      %c384_i32_8 = arith.constant 384 : i32
                      %c320_i32_9 = arith.constant 320 : i32
                      %c256_i32_10 = arith.constant 256 : i32
                      %c192_i32_11 = arith.constant 192 : i32
                      %c128_i32_12 = arith.constant 128 : i32
                      %c0_13 = arith.constant 0 : index
                      %c4608_14 = arith.constant 4608 : index
                      %c3_i32 = arith.constant 3 : i32
                      %c0_i32_15 = arith.constant 0 : i32
                      %c-1_i32 = arith.constant -1 : i32
                      %c17344 = arith.constant 17344 : index
                      %c17328 = arith.constant 17328 : index
                      %c17312 = arith.constant 17312 : index
                      %c64_i32_16 = arith.constant 64 : i32
                      %mask, %remaining_out = ascvec.update_mask %c64_i32_16 : !ascvec.mask
                      %131 = arith.addi %arg36, %c17312 : index
                      %132 = ascvec.loadalign_brc %arg35[%131] : memref<?xi32, 1>, index -> <i32>
                      %133 = arith.addi %arg36, %c17328 : index
                      %134 = ascvec.loadalign_brc %arg37[%133] : memref<?xi32, 1>, index -> <i32>
                      %135 = arith.addi %arg36, %c17344 : index
                      %136 = ascvec.loadalign_brc %arg38[%135] : memref<?xi32, 1>, index -> <i32>
                      %137 = ascvec.shl %134, 2, %mask : <i32> -> <i32>
                      %138 = ascvec.shl %132, 2, %mask : <i32> -> <i32>
                      %139 = ascvec.sub %136, %138, %mask : <i32>
                      %140 = ascvec.add %137, %139, %mask : <i32>
                      %141 = ascvec.duplicate_scalar %c-1_i32, %mask : i32 -> <i32>
                      %142 = ascvec.arange %c0_i32_15 : i32 -> <i32>
                      %143 = ascvec.duplicate_scalar %c3_i32, %mask : i32 -> <i32>
                      %144 = ascvec.and %142, %143, %mask : <i32>, <i32> -> <i32>
                      %145 = arith.addi %arg40, %c4608_14 : index
                      %c0_17 = arith.constant 0 : index
                      %c1 = arith.constant 1 : index
                      scf.for %arg49 = %c0_17 to %c1 step %c1 {
                        %153 = ascvec.loadalign %arg39[%145] : memref<?xui32, 1>, index -> <ui32>
                        %154 = ascvec.reinterpret %153 : <ui32> -> <i32>
                        %155 = ascvec.shl %154, 2, %mask : <i32> -> <i32>
                        %low, %high = ascvec.intlv %155, %155 : <i32>, <i32> -> <i32>, <i32>
                        %low_32, %high_33 = ascvec.intlv %low, %low : <i32>, <i32> -> <i32>, <i32>
                        %low_34, %high_35 = ascvec.intlv %high, %high : <i32>, <i32> -> <i32>, <i32>
                        %156 = ascvec.cmp_lt %142, %137, %mask : <i32>, <i32> -> !ascvec.mask
                        %157 = ascvec.add %low_32, %144, %mask : <i32>
                        %158 = ascvec.sub %142, %137, %mask : <i32>
                        %159 = ascvec.add %138, %158, %mask : <i32>
                        %160 = ascvec.select %157, %159, %156 : <i32>
                        %161 = ascvec.cmp_lt %142, %140, %mask : <i32>, <i32> -> !ascvec.mask
                        %162 = ascvec.select %160, %141, %161 : <i32>
                        ascvec.storealign %arg41[%c0_13], %162, %mask : memref<2056xi32, 1>, index, <i32>
                        %163 = ascvec.arange %c64_i32_16 : i32 -> <i32>
                        %164 = ascvec.cmp_lt %163, %137, %mask : <i32>, <i32> -> !ascvec.mask
                        %165 = ascvec.add %high_33, %144, %mask : <i32>
                        %166 = ascvec.sub %163, %137, %mask : <i32>
                        %167 = ascvec.add %138, %166, %mask : <i32>
                        %168 = ascvec.select %165, %167, %164 : <i32>
                        %169 = ascvec.cmp_lt %163, %140, %mask : <i32>, <i32> -> !ascvec.mask
                        %170 = ascvec.select %168, %141, %169 : <i32>
                        ascvec.storealign %arg41[%c64], %170, %mask : memref<2056xi32, 1>, index, <i32>
                        %171 = ascvec.arange %c128_i32_12 : i32 -> <i32>
                        %172 = ascvec.cmp_lt %171, %137, %mask : <i32>, <i32> -> !ascvec.mask
                        %173 = ascvec.add %low_34, %144, %mask : <i32>
                        %174 = ascvec.sub %171, %137, %mask : <i32>
                        %175 = ascvec.add %138, %174, %mask : <i32>
                        %176 = ascvec.select %173, %175, %172 : <i32>
                        %177 = ascvec.cmp_lt %171, %140, %mask : <i32>, <i32> -> !ascvec.mask
                        %178 = ascvec.select %176, %141, %177 : <i32>
                        ascvec.storealign %arg41[%c128], %178, %mask : memref<2056xi32, 1>, index, <i32>
                        %179 = ascvec.arange %c192_i32_11 : i32 -> <i32>
                        %180 = ascvec.cmp_lt %179, %137, %mask : <i32>, <i32> -> !ascvec.mask
                        %181 = ascvec.add %high_35, %144, %mask : <i32>
                        %182 = ascvec.sub %179, %137, %mask : <i32>
                        %183 = ascvec.add %138, %182, %mask : <i32>
                        %184 = ascvec.select %181, %183, %180 : <i32>
                        %185 = ascvec.cmp_lt %179, %140, %mask : <i32>, <i32> -> !ascvec.mask
                        %186 = ascvec.select %184, %141, %185 : <i32>
                        ascvec.storealign %arg41[%c192], %186, %mask : memref<2056xi32, 1>, index, <i32>
                      }
                      %146 = arith.addi %arg42, %c4608_14 : index
                      %c0_18 = arith.constant 0 : index
                      %c1_19 = arith.constant 1 : index
                      scf.for %arg49 = %c0_18 to %c1_19 step %c1_19 {
                        %153 = ascvec.loadalign %arg39[%146] : memref<?xui32, 1>, index -> <ui32>
                        %154 = ascvec.reinterpret %153 : <ui32> -> <i32>
                        %155 = ascvec.shl %154, 2, %mask : <i32> -> <i32>
                        %low, %high = ascvec.intlv %155, %155 : <i32>, <i32> -> <i32>, <i32>
                        %low_32, %high_33 = ascvec.intlv %low, %low : <i32>, <i32> -> <i32>, <i32>
                        %low_34, %high_35 = ascvec.intlv %high, %high : <i32>, <i32> -> <i32>, <i32>
                        %156 = ascvec.arange %c256_i32_10 : i32 -> <i32>
                        %157 = ascvec.cmp_lt %156, %137, %mask : <i32>, <i32> -> !ascvec.mask
                        %158 = ascvec.add %low_32, %144, %mask : <i32>
                        %159 = ascvec.sub %156, %137, %mask : <i32>
                        %160 = ascvec.add %138, %159, %mask : <i32>
                        %161 = ascvec.select %158, %160, %157 : <i32>
                        %162 = ascvec.cmp_lt %156, %140, %mask : <i32>, <i32> -> !ascvec.mask
                        %163 = ascvec.select %161, %141, %162 : <i32>
                        ascvec.storealign %arg41[%c256], %163, %mask : memref<2056xi32, 1>, index, <i32>
                        %164 = ascvec.arange %c320_i32_9 : i32 -> <i32>
                        %165 = ascvec.cmp_lt %164, %137, %mask : <i32>, <i32> -> !ascvec.mask
                        %166 = ascvec.add %high_33, %144, %mask : <i32>
                        %167 = ascvec.sub %164, %137, %mask : <i32>
                        %168 = ascvec.add %138, %167, %mask : <i32>
                        %169 = ascvec.select %166, %168, %165 : <i32>
                        %170 = ascvec.cmp_lt %164, %140, %mask : <i32>, <i32> -> !ascvec.mask
                        %171 = ascvec.select %169, %141, %170 : <i32>
                        ascvec.storealign %arg41[%c320], %171, %mask : memref<2056xi32, 1>, index, <i32>
                        %172 = ascvec.arange %c384_i32_8 : i32 -> <i32>
                        %173 = ascvec.cmp_lt %172, %137, %mask : <i32>, <i32> -> !ascvec.mask
                        %174 = ascvec.add %low_34, %144, %mask : <i32>
                        %175 = ascvec.sub %172, %137, %mask : <i32>
                        %176 = ascvec.add %138, %175, %mask : <i32>
                        %177 = ascvec.select %174, %176, %173 : <i32>
                        %178 = ascvec.cmp_lt %172, %140, %mask : <i32>, <i32> -> !ascvec.mask
                        %179 = ascvec.select %177, %141, %178 : <i32>
                        ascvec.storealign %arg41[%c384], %179, %mask : memref<2056xi32, 1>, index, <i32>
                        %180 = ascvec.arange %c448_i32_7 : i32 -> <i32>
                        %181 = ascvec.cmp_lt %180, %137, %mask : <i32>, <i32> -> !ascvec.mask
                        %182 = ascvec.add %high_35, %144, %mask : <i32>
                        %183 = ascvec.sub %180, %137, %mask : <i32>
                        %184 = ascvec.add %138, %183, %mask : <i32>
                        %185 = ascvec.select %182, %184, %181 : <i32>
                        %186 = ascvec.cmp_lt %180, %140, %mask : <i32>, <i32> -> !ascvec.mask
                        %187 = ascvec.select %185, %141, %186 : <i32>
                        ascvec.storealign %arg41[%c448], %187, %mask : memref<2056xi32, 1>, index, <i32>
                      }
                      %147 = arith.addi %arg43, %c4608_14 : index
                      %c0_20 = arith.constant 0 : index
                      %c1_21 = arith.constant 1 : index
                      scf.for %arg49 = %c0_20 to %c1_21 step %c1_21 {
                        %153 = ascvec.loadalign %arg39[%147] : memref<?xui32, 1>, index -> <ui32>
                        %154 = ascvec.reinterpret %153 : <ui32> -> <i32>
                        %155 = ascvec.shl %154, 2, %mask : <i32> -> <i32>
                        %low, %high = ascvec.intlv %155, %155 : <i32>, <i32> -> <i32>, <i32>
                        %low_32, %high_33 = ascvec.intlv %low, %low : <i32>, <i32> -> <i32>, <i32>
                        %low_34, %high_35 = ascvec.intlv %high, %high : <i32>, <i32> -> <i32>, <i32>
                        %156 = ascvec.arange %c512_i32_6 : i32 -> <i32>
                        %157 = ascvec.cmp_lt %156, %137, %mask : <i32>, <i32> -> !ascvec.mask
                        %158 = ascvec.add %low_32, %144, %mask : <i32>
                        %159 = ascvec.sub %156, %137, %mask : <i32>
                        %160 = ascvec.add %138, %159, %mask : <i32>
                        %161 = ascvec.select %158, %160, %157 : <i32>
                        %162 = ascvec.cmp_lt %156, %140, %mask : <i32>, <i32> -> !ascvec.mask
                        %163 = ascvec.select %161, %141, %162 : <i32>
                        ascvec.storealign %arg41[%c512], %163, %mask : memref<2056xi32, 1>, index, <i32>
                        %164 = ascvec.arange %c576_i32 : i32 -> <i32>
                        %165 = ascvec.cmp_lt %164, %137, %mask : <i32>, <i32> -> !ascvec.mask
                        %166 = ascvec.add %high_33, %144, %mask : <i32>
                        %167 = ascvec.sub %164, %137, %mask : <i32>
                        %168 = ascvec.add %138, %167, %mask : <i32>
                        %169 = ascvec.select %166, %168, %165 : <i32>
                        %170 = ascvec.cmp_lt %164, %140, %mask : <i32>, <i32> -> !ascvec.mask
                        %171 = ascvec.select %169, %141, %170 : <i32>
                        ascvec.storealign %arg41[%c576], %171, %mask : memref<2056xi32, 1>, index, <i32>
                        %172 = ascvec.arange %c640_i32 : i32 -> <i32>
                        %173 = ascvec.cmp_lt %172, %137, %mask : <i32>, <i32> -> !ascvec.mask
                        %174 = ascvec.add %low_34, %144, %mask : <i32>
                        %175 = ascvec.sub %172, %137, %mask : <i32>
                        %176 = ascvec.add %138, %175, %mask : <i32>
                        %177 = ascvec.select %174, %176, %173 : <i32>
                        %178 = ascvec.cmp_lt %172, %140, %mask : <i32>, <i32> -> !ascvec.mask
                        %179 = ascvec.select %177, %141, %178 : <i32>
                        ascvec.storealign %arg41[%c640], %179, %mask : memref<2056xi32, 1>, index, <i32>
                        %180 = ascvec.arange %c704_i32 : i32 -> <i32>
                        %181 = ascvec.cmp_lt %180, %137, %mask : <i32>, <i32> -> !ascvec.mask
                        %182 = ascvec.add %high_35, %144, %mask : <i32>
                        %183 = ascvec.sub %180, %137, %mask : <i32>
                        %184 = ascvec.add %138, %183, %mask : <i32>
                        %185 = ascvec.select %182, %184, %181 : <i32>
                        %186 = ascvec.cmp_lt %180, %140, %mask : <i32>, <i32> -> !ascvec.mask
                        %187 = ascvec.select %185, %141, %186 : <i32>
                        ascvec.storealign %arg41[%c704], %187, %mask : memref<2056xi32, 1>, index, <i32>
                      }
                      %148 = arith.addi %arg44, %c4608_14 : index
                      %c0_22 = arith.constant 0 : index
                      %c1_23 = arith.constant 1 : index
                      scf.for %arg49 = %c0_22 to %c1_23 step %c1_23 {
                        %153 = ascvec.loadalign %arg39[%148] : memref<?xui32, 1>, index -> <ui32>
                        %154 = ascvec.reinterpret %153 : <ui32> -> <i32>
                        %155 = ascvec.shl %154, 2, %mask : <i32> -> <i32>
                        %low, %high = ascvec.intlv %155, %155 : <i32>, <i32> -> <i32>, <i32>
                        %low_32, %high_33 = ascvec.intlv %low, %low : <i32>, <i32> -> <i32>, <i32>
                        %low_34, %high_35 = ascvec.intlv %high, %high : <i32>, <i32> -> <i32>, <i32>
                        %156 = ascvec.arange %c768_i32 : i32 -> <i32>
                        %157 = ascvec.cmp_lt %156, %137, %mask : <i32>, <i32> -> !ascvec.mask
                        %158 = ascvec.add %low_32, %144, %mask : <i32>
                        %159 = ascvec.sub %156, %137, %mask : <i32>
                        %160 = ascvec.add %138, %159, %mask : <i32>
                        %161 = ascvec.select %158, %160, %157 : <i32>
                        %162 = ascvec.cmp_lt %156, %140, %mask : <i32>, <i32> -> !ascvec.mask
                        %163 = ascvec.select %161, %141, %162 : <i32>
                        ascvec.storealign %arg41[%c768], %163, %mask : memref<2056xi32, 1>, index, <i32>
                        %164 = ascvec.arange %c832_i32 : i32 -> <i32>
                        %165 = ascvec.cmp_lt %164, %137, %mask : <i32>, <i32> -> !ascvec.mask
                        %166 = ascvec.add %high_33, %144, %mask : <i32>
                        %167 = ascvec.sub %164, %137, %mask : <i32>
                        %168 = ascvec.add %138, %167, %mask : <i32>
                        %169 = ascvec.select %166, %168, %165 : <i32>
                        %170 = ascvec.cmp_lt %164, %140, %mask : <i32>, <i32> -> !ascvec.mask
                        %171 = ascvec.select %169, %141, %170 : <i32>
                        ascvec.storealign %arg41[%c832], %171, %mask : memref<2056xi32, 1>, index, <i32>
                        %172 = ascvec.arange %c896_i32 : i32 -> <i32>
                        %173 = ascvec.cmp_lt %172, %137, %mask : <i32>, <i32> -> !ascvec.mask
                        %174 = ascvec.add %low_34, %144, %mask : <i32>
                        %175 = ascvec.sub %172, %137, %mask : <i32>
                        %176 = ascvec.add %138, %175, %mask : <i32>
                        %177 = ascvec.select %174, %176, %173 : <i32>
                        %178 = ascvec.cmp_lt %172, %140, %mask : <i32>, <i32> -> !ascvec.mask
                        %179 = ascvec.select %177, %141, %178 : <i32>
                        ascvec.storealign %arg41[%c896], %179, %mask : memref<2056xi32, 1>, index, <i32>
                        %180 = ascvec.arange %c960_i32 : i32 -> <i32>
                        %181 = ascvec.cmp_lt %180, %137, %mask : <i32>, <i32> -> !ascvec.mask
                        %182 = ascvec.add %high_35, %144, %mask : <i32>
                        %183 = ascvec.sub %180, %137, %mask : <i32>
                        %184 = ascvec.add %138, %183, %mask : <i32>
                        %185 = ascvec.select %182, %184, %181 : <i32>
                        %186 = ascvec.cmp_lt %180, %140, %mask : <i32>, <i32> -> !ascvec.mask
                        %187 = ascvec.select %185, %141, %186 : <i32>
                        ascvec.storealign %arg41[%c960], %187, %mask : memref<2056xi32, 1>, index, <i32>
                      }
                      %149 = arith.addi %arg45, %c4608_14 : index
                      %c0_24 = arith.constant 0 : index
                      %c1_25 = arith.constant 1 : index
                      scf.for %arg49 = %c0_24 to %c1_25 step %c1_25 {
                        %153 = ascvec.loadalign %arg39[%149] : memref<?xui32, 1>, index -> <ui32>
                        %154 = ascvec.reinterpret %153 : <ui32> -> <i32>
                        %155 = ascvec.shl %154, 2, %mask : <i32> -> <i32>
                        %low, %high = ascvec.intlv %155, %155 : <i32>, <i32> -> <i32>, <i32>
                        %low_32, %high_33 = ascvec.intlv %low, %low : <i32>, <i32> -> <i32>, <i32>
                        %low_34, %high_35 = ascvec.intlv %high, %high : <i32>, <i32> -> <i32>, <i32>
                        %156 = ascvec.arange %c1024_i32 : i32 -> <i32>
                        %157 = ascvec.cmp_lt %156, %137, %mask : <i32>, <i32> -> !ascvec.mask
                        %158 = ascvec.add %low_32, %144, %mask : <i32>
                        %159 = ascvec.sub %156, %137, %mask : <i32>
                        %160 = ascvec.add %138, %159, %mask : <i32>
                        %161 = ascvec.select %158, %160, %157 : <i32>
                        %162 = ascvec.cmp_lt %156, %140, %mask : <i32>, <i32> -> !ascvec.mask
                        %163 = ascvec.select %161, %141, %162 : <i32>
                        ascvec.storealign %arg41[%c1024], %163, %mask : memref<2056xi32, 1>, index, <i32>
                        %164 = ascvec.arange %c1088_i32 : i32 -> <i32>
                        %165 = ascvec.cmp_lt %164, %137, %mask : <i32>, <i32> -> !ascvec.mask
                        %166 = ascvec.add %high_33, %144, %mask : <i32>
                        %167 = ascvec.sub %164, %137, %mask : <i32>
                        %168 = ascvec.add %138, %167, %mask : <i32>
                        %169 = ascvec.select %166, %168, %165 : <i32>
                        %170 = ascvec.cmp_lt %164, %140, %mask : <i32>, <i32> -> !ascvec.mask
                        %171 = ascvec.select %169, %141, %170 : <i32>
                        ascvec.storealign %arg41[%c1088], %171, %mask : memref<2056xi32, 1>, index, <i32>
                        %172 = ascvec.arange %c1152_i32 : i32 -> <i32>
                        %173 = ascvec.cmp_lt %172, %137, %mask : <i32>, <i32> -> !ascvec.mask
                        %174 = ascvec.add %low_34, %144, %mask : <i32>
                        %175 = ascvec.sub %172, %137, %mask : <i32>
                        %176 = ascvec.add %138, %175, %mask : <i32>
                        %177 = ascvec.select %174, %176, %173 : <i32>
                        %178 = ascvec.cmp_lt %172, %140, %mask : <i32>, <i32> -> !ascvec.mask
                        %179 = ascvec.select %177, %141, %178 : <i32>
                        ascvec.storealign %arg41[%c1152], %179, %mask : memref<2056xi32, 1>, index, <i32>
                        %180 = ascvec.arange %c1216_i32 : i32 -> <i32>
                        %181 = ascvec.cmp_lt %180, %137, %mask : <i32>, <i32> -> !ascvec.mask
                        %182 = ascvec.add %high_35, %144, %mask : <i32>
                        %183 = ascvec.sub %180, %137, %mask : <i32>
                        %184 = ascvec.add %138, %183, %mask : <i32>
                        %185 = ascvec.select %182, %184, %181 : <i32>
                        %186 = ascvec.cmp_lt %180, %140, %mask : <i32>, <i32> -> !ascvec.mask
                        %187 = ascvec.select %185, %141, %186 : <i32>
                        ascvec.storealign %arg41[%c1216], %187, %mask : memref<2056xi32, 1>, index, <i32>
                      }
                      %150 = arith.addi %arg46, %c4608_14 : index
                      %c0_26 = arith.constant 0 : index
                      %c1_27 = arith.constant 1 : index
                      scf.for %arg49 = %c0_26 to %c1_27 step %c1_27 {
                        %153 = ascvec.loadalign %arg39[%150] : memref<?xui32, 1>, index -> <ui32>
                        %154 = ascvec.reinterpret %153 : <ui32> -> <i32>
                        %155 = ascvec.shl %154, 2, %mask : <i32> -> <i32>
                        %low, %high = ascvec.intlv %155, %155 : <i32>, <i32> -> <i32>, <i32>
                        %low_32, %high_33 = ascvec.intlv %low, %low : <i32>, <i32> -> <i32>, <i32>
                        %low_34, %high_35 = ascvec.intlv %high, %high : <i32>, <i32> -> <i32>, <i32>
                        %156 = ascvec.arange %c1280_i32 : i32 -> <i32>
                        %157 = ascvec.cmp_lt %156, %137, %mask : <i32>, <i32> -> !ascvec.mask
                        %158 = ascvec.add %low_32, %144, %mask : <i32>
                        %159 = ascvec.sub %156, %137, %mask : <i32>
                        %160 = ascvec.add %138, %159, %mask : <i32>
                        %161 = ascvec.select %158, %160, %157 : <i32>
                        %162 = ascvec.cmp_lt %156, %140, %mask : <i32>, <i32> -> !ascvec.mask
                        %163 = ascvec.select %161, %141, %162 : <i32>
                        ascvec.storealign %arg41[%c1280], %163, %mask : memref<2056xi32, 1>, index, <i32>
                        %164 = ascvec.arange %c1344_i32 : i32 -> <i32>
                        %165 = ascvec.cmp_lt %164, %137, %mask : <i32>, <i32> -> !ascvec.mask
                        %166 = ascvec.add %high_33, %144, %mask : <i32>
                        %167 = ascvec.sub %164, %137, %mask : <i32>
                        %168 = ascvec.add %138, %167, %mask : <i32>
                        %169 = ascvec.select %166, %168, %165 : <i32>
                        %170 = ascvec.cmp_lt %164, %140, %mask : <i32>, <i32> -> !ascvec.mask
                        %171 = ascvec.select %169, %141, %170 : <i32>
                        ascvec.storealign %arg41[%c1344], %171, %mask : memref<2056xi32, 1>, index, <i32>
                        %172 = ascvec.arange %c1408_i32 : i32 -> <i32>
                        %173 = ascvec.cmp_lt %172, %137, %mask : <i32>, <i32> -> !ascvec.mask
                        %174 = ascvec.add %low_34, %144, %mask : <i32>
                        %175 = ascvec.sub %172, %137, %mask : <i32>
                        %176 = ascvec.add %138, %175, %mask : <i32>
                        %177 = ascvec.select %174, %176, %173 : <i32>
                        %178 = ascvec.cmp_lt %172, %140, %mask : <i32>, <i32> -> !ascvec.mask
                        %179 = ascvec.select %177, %141, %178 : <i32>
                        ascvec.storealign %arg41[%c1408], %179, %mask : memref<2056xi32, 1>, index, <i32>
                        %180 = ascvec.arange %c1472_i32 : i32 -> <i32>
                        %181 = ascvec.cmp_lt %180, %137, %mask : <i32>, <i32> -> !ascvec.mask
                        %182 = ascvec.add %high_35, %144, %mask : <i32>
                        %183 = ascvec.sub %180, %137, %mask : <i32>
                        %184 = ascvec.add %138, %183, %mask : <i32>
                        %185 = ascvec.select %182, %184, %181 : <i32>
                        %186 = ascvec.cmp_lt %180, %140, %mask : <i32>, <i32> -> !ascvec.mask
                        %187 = ascvec.select %185, %141, %186 : <i32>
                        ascvec.storealign %arg41[%c1472], %187, %mask : memref<2056xi32, 1>, index, <i32>
                      }
                      %151 = arith.addi %arg47, %c4608_14 : index
                      %c0_28 = arith.constant 0 : index
                      %c1_29 = arith.constant 1 : index
                      scf.for %arg49 = %c0_28 to %c1_29 step %c1_29 {
                        %153 = ascvec.loadalign %arg39[%151] : memref<?xui32, 1>, index -> <ui32>
                        %154 = ascvec.reinterpret %153 : <ui32> -> <i32>
                        %155 = ascvec.shl %154, 2, %mask : <i32> -> <i32>
                        %low, %high = ascvec.intlv %155, %155 : <i32>, <i32> -> <i32>, <i32>
                        %low_32, %high_33 = ascvec.intlv %low, %low : <i32>, <i32> -> <i32>, <i32>
                        %low_34, %high_35 = ascvec.intlv %high, %high : <i32>, <i32> -> <i32>, <i32>
                        %156 = ascvec.arange %c1536_i32 : i32 -> <i32>
                        %157 = ascvec.cmp_lt %156, %137, %mask : <i32>, <i32> -> !ascvec.mask
                        %158 = ascvec.add %low_32, %144, %mask : <i32>
                        %159 = ascvec.sub %156, %137, %mask : <i32>
                        %160 = ascvec.add %138, %159, %mask : <i32>
                        %161 = ascvec.select %158, %160, %157 : <i32>
                        %162 = ascvec.cmp_lt %156, %140, %mask : <i32>, <i32> -> !ascvec.mask
                        %163 = ascvec.select %161, %141, %162 : <i32>
                        ascvec.storealign %arg41[%c1536], %163, %mask : memref<2056xi32, 1>, index, <i32>
                        %164 = ascvec.arange %c1600_i32 : i32 -> <i32>
                        %165 = ascvec.cmp_lt %164, %137, %mask : <i32>, <i32> -> !ascvec.mask
                        %166 = ascvec.add %high_33, %144, %mask : <i32>
                        %167 = ascvec.sub %164, %137, %mask : <i32>
                        %168 = ascvec.add %138, %167, %mask : <i32>
                        %169 = ascvec.select %166, %168, %165 : <i32>
                        %170 = ascvec.cmp_lt %164, %140, %mask : <i32>, <i32> -> !ascvec.mask
                        %171 = ascvec.select %169, %141, %170 : <i32>
                        ascvec.storealign %arg41[%c1600], %171, %mask : memref<2056xi32, 1>, index, <i32>
                        %172 = ascvec.arange %c1664_i32 : i32 -> <i32>
                        %173 = ascvec.cmp_lt %172, %137, %mask : <i32>, <i32> -> !ascvec.mask
                        %174 = ascvec.add %low_34, %144, %mask : <i32>
                        %175 = ascvec.sub %172, %137, %mask : <i32>
                        %176 = ascvec.add %138, %175, %mask : <i32>
                        %177 = ascvec.select %174, %176, %173 : <i32>
                        %178 = ascvec.cmp_lt %172, %140, %mask : <i32>, <i32> -> !ascvec.mask
                        %179 = ascvec.select %177, %141, %178 : <i32>
                        ascvec.storealign %arg41[%c1664], %179, %mask : memref<2056xi32, 1>, index, <i32>
                        %180 = ascvec.arange %c1728_i32 : i32 -> <i32>
                        %181 = ascvec.cmp_lt %180, %137, %mask : <i32>, <i32> -> !ascvec.mask
                        %182 = ascvec.add %high_35, %144, %mask : <i32>
                        %183 = ascvec.sub %180, %137, %mask : <i32>
                        %184 = ascvec.add %138, %183, %mask : <i32>
                        %185 = ascvec.select %182, %184, %181 : <i32>
                        %186 = ascvec.cmp_lt %180, %140, %mask : <i32>, <i32> -> !ascvec.mask
                        %187 = ascvec.select %185, %141, %186 : <i32>
                        ascvec.storealign %arg41[%c1728], %187, %mask : memref<2056xi32, 1>, index, <i32>
                      }
                      %152 = arith.addi %arg48, %c4608_14 : index
                      %c0_30 = arith.constant 0 : index
                      %c1_31 = arith.constant 1 : index
                      scf.for %arg49 = %c0_30 to %c1_31 step %c1_31 {
                        %153 = ascvec.loadalign %arg39[%152] : memref<?xui32, 1>, index -> <ui32>
                        %154 = ascvec.reinterpret %153 : <ui32> -> <i32>
                        %155 = ascvec.shl %154, 2, %mask : <i32> -> <i32>
                        %low, %high = ascvec.intlv %155, %155 : <i32>, <i32> -> <i32>, <i32>
                        %low_32, %high_33 = ascvec.intlv %low, %low : <i32>, <i32> -> <i32>, <i32>
                        %low_34, %high_35 = ascvec.intlv %high, %high : <i32>, <i32> -> <i32>, <i32>
                        %156 = ascvec.arange %c1792_i32 : i32 -> <i32>
                        %157 = ascvec.cmp_lt %156, %137, %mask : <i32>, <i32> -> !ascvec.mask
                        %158 = ascvec.add %low_32, %144, %mask : <i32>
                        %159 = ascvec.sub %156, %137, %mask : <i32>
                        %160 = ascvec.add %138, %159, %mask : <i32>
                        %161 = ascvec.select %158, %160, %157 : <i32>
                        %162 = ascvec.cmp_lt %156, %140, %mask : <i32>, <i32> -> !ascvec.mask
                        %163 = ascvec.select %161, %141, %162 : <i32>
                        ascvec.storealign %arg41[%c1792], %163, %mask : memref<2056xi32, 1>, index, <i32>
                        %164 = ascvec.arange %c1856_i32 : i32 -> <i32>
                        %165 = ascvec.cmp_lt %164, %137, %mask : <i32>, <i32> -> !ascvec.mask
                        %166 = ascvec.add %high_33, %144, %mask : <i32>
                        %167 = ascvec.sub %164, %137, %mask : <i32>
                        %168 = ascvec.add %138, %167, %mask : <i32>
                        %169 = ascvec.select %166, %168, %165 : <i32>
                        %170 = ascvec.cmp_lt %164, %140, %mask : <i32>, <i32> -> !ascvec.mask
                        %171 = ascvec.select %169, %141, %170 : <i32>
                        ascvec.storealign %arg41[%c1856], %171, %mask : memref<2056xi32, 1>, index, <i32>
                        %172 = ascvec.arange %c1920_i32 : i32 -> <i32>
                        %173 = ascvec.cmp_lt %172, %137, %mask : <i32>, <i32> -> !ascvec.mask
                        %174 = ascvec.add %low_34, %144, %mask : <i32>
                        %175 = ascvec.sub %172, %137, %mask : <i32>
                        %176 = ascvec.add %138, %175, %mask : <i32>
                        %177 = ascvec.select %174, %176, %173 : <i32>
                        %178 = ascvec.cmp_lt %172, %140, %mask : <i32>, <i32> -> !ascvec.mask
                        %179 = ascvec.select %177, %141, %178 : <i32>
                        ascvec.storealign %arg41[%c1920], %179, %mask : memref<2056xi32, 1>, index, <i32>
                        %180 = ascvec.arange %c1984_i32 : i32 -> <i32>
                        %181 = ascvec.cmp_lt %180, %137, %mask : <i32>, <i32> -> !ascvec.mask
                        %182 = ascvec.add %high_35, %144, %mask : <i32>
                        %183 = ascvec.sub %180, %137, %mask : <i32>
                        %184 = ascvec.add %138, %183, %mask : <i32>
                        %185 = ascvec.select %182, %184, %181 : <i32>
                        %186 = ascvec.cmp_lt %180, %140, %mask : <i32>, <i32> -> !ascvec.mask
                        %187 = ascvec.select %185, %141, %186 : <i32>
                        ascvec.storealign %arg41[%c1984], %187, %mask : memref<2056xi32, 1>, index, <i32>
                        %mask_36, %remaining_out_37 = ascvec.update_mask %c8_i32_5 : !ascvec.mask
                        %188 = ascvec.arange %c2048_i32 : i32 -> <i32>
                        %189 = ascvec.sub %188, %137, %mask : <i32>
                        %190 = ascvec.add %138, %189, %mask : <i32>
                        %191 = ascvec.cmp_lt %188, %140, %mask_36 : <i32>, <i32> -> !ascvec.mask
                        %192 = ascvec.select %190, %141, %191 : <i32>
                        %193 = ascvec.cmp_eq_scalar %188, %c2051_i32, %mask_36 : <i32>, i32 -> !ascvec.mask
                        %194 = ascvec.select %140, %192, %193 : <i32>
                        ascvec.storealign %arg41[%c2048], %194, %mask_36 : memref<2056xi32, 1>, index, <i32>
                      }
                    } {cannir.vf_mode = "simd"}
                    ascvec.release_buf v, %102, 0 {cannir.buffer.generated_sync}
                    %124 = arith.addi %49, %93 : i64
                    %125 = arith.muli %124, %c65792_i64 : i64
                    %126 = ascscalar.memref_address %alloc_2 : memref<1xi32>
                    %127 = arith.divui %125, %c8_i64 : i64
                    %128 = arith.addi %126, %127 : i64
                    %129 = ascscalar.bind_gm_ptr %128 : memref<1xi32>
                    %130 = ascscalar.bind_local_ptr %100 : memref<1xi32, 1>
                    ascvec.get_buf mte3, %102, 0 {cannir.buffer.generated_sync}
                    ascvec.copy_ub2gm %129[%c0], %130[%c0], %c1_i32, %c8224_i32, %c0_i64, %c8224_i32 l2_cache_ctl %c0_i32 : memref<1xi32>, memref<1xi32, 1>
                    ascvec.release_buf mte3, %102, 0 {cannir.buffer.generated_sync}
                    scf.yield %104 : i64
                  } else {
                    scf.yield %arg34 : i64
                  }
                  scf.yield %95 : i64
                }
                scf.yield %92 : i64
              }
              scf.yield %91, %90#0, %90#1 : i64, i64, i64
            }
            scf.yield %74#0, %74#1, %77#0, %77#1, %77#2, %76 : i64, i64, i64, i64, i64, i64
          } else {
            %74 = scf.for %arg29 = %c0_i64 to %c16_i64 step %c2_i64 iter_args(%arg30 = %arg25) -> (i64)  : i64 {
              %75 = arith.muli %0, %c16_i64 : i64
              %76 = arith.addi %75, %arg29 : i64
              %77 = arith.cmpi slt, %76, %51 : i64
              %78 = scf.if %77 -> (i64) {
                %86 = arith.addi %49, %76 : i64
                %87 = arith.index_cast %86 : i64 to index
                %88 = memref.load %alloc_0[%87] : memref<1xi32>
                %89 = arith.extsi %88 : i32 to i64
                scf.yield %89 : i64
              } else {
                scf.yield %c-1_i64 : i64
              }
              %79 = arith.addi %76, %c1_i64 : i64
              %80 = arith.cmpi slt, %79, %51 : i64
              %81 = scf.if %80 -> (i64) {
                %86 = arith.addi %49, %76 : i64
                %87 = arith.addi %86, %c1_i64 : i64
                %88 = arith.index_cast %87 : i64 to index
                %89 = memref.load %alloc_0[%88] : memref<1xi32>
                %90 = arith.extsi %89 : i32 to i64
                scf.yield %90 : i64
              } else {
                scf.yield %c-1_i64 : i64
              }
              %82 = arith.trunci %78 : i64 to i32
              %83 = arith.trunci %81 : i64 to i32
              %84 = ascscalar.bitcast_memref %alloc_4 byte_offset = %c0_i64 : memref<262144xui8, 1> -> memref<?xi32, 1>
              ascvec.vf(%82, %84, %84, %84, %83 : i32, memref<?xi32, 1>, memref<?xi32, 1>, memref<?xi32, 1>, i32) {
              ^bb0(%arg31: i32, %arg32: memref<?xi32, 1>, %arg33: memref<?xi32, 1>, %arg34: memref<?xi32, 1>, %arg35: i32):
                %c17352 = arith.constant 17352 : index
                %c17336 = arith.constant 17336 : index
                %c17320 = arith.constant 17320 : index
                %c17344 = arith.constant 17344 : index
                %c17328 = arith.constant 17328 : index
                %c0_i32_5 = arith.constant 0 : i32
                %c17312 = arith.constant 17312 : index
                %c1_i32_6 = arith.constant 1 : i32
                %c0_7 = arith.constant 0 : index
                %c1 = arith.constant 1 : index
                scf.for %arg36 = %c0_7 to %c1 step %c1 {
                  %mask, %remaining_out = ascvec.update_mask %c1_i32_6 : !ascvec.mask
                  %86 = ascvec.duplicate_scalar %arg31, %mask : i32 -> <i32>
                  %87 = ascvec.add_scalar %86, %c1_i32_6, %mask : <i32>, i32
                  %88 = ascvec.shr %87, 2, %mask : <i32> -> <i32>
                  ascvec.storealign_1st %arg32[%c17312], %88 : memref<?xi32, 1>, index, <i32>
                  %89 = ascvec.duplicate_scalar %c0_i32_5, %mask : i32 -> <i32>
                  ascvec.storealign_1st %arg33[%c17328], %89 : memref<?xi32, 1>, index, <i32>
                  ascvec.storealign_1st %arg34[%c17344], %87 : memref<?xi32, 1>, index, <i32>
                  %90 = ascvec.duplicate_scalar %arg35, %mask : i32 -> <i32>
                  %91 = ascvec.add_scalar %90, %c1_i32_6, %mask : <i32>, i32
                  %92 = ascvec.shr %91, 2, %mask : <i32> -> <i32>
                  ascvec.storealign_1st %arg32[%c17320], %92 : memref<?xi32, 1>, index, <i32>
                  ascvec.storealign_1st %arg33[%c17336], %89 : memref<?xi32, 1>, index, <i32>
                  ascvec.storealign_1st %arg34[%c17352], %91 : memref<?xi32, 1>, index, <i32>
                }
              } {cannir.vf_mode = "simd"}
              %85 = scf.for %arg31 = %c0_i64 to %c2_i64 step %c1_i64 iter_args(%arg32 = %arg30) -> (i64)  : i64 {
                %86 = arith.addi %76, %arg31 : i64
                %87 = arith.cmpi slt, %86, %51 : i64
                %88 = scf.if %87 -> (i64) {
                  %89 = arith.cmpi eq, %arg32, %c0_i64 : i64
                  %90 = arith.select %89, %c556032_i64, %c621824_i64 : i64
                  %91 = ascscalar.memref_address %alloc_4 : memref<262144xui8, 1>
                  %92 = arith.divui %90, %c8_i64 : i64
                  %93 = arith.addi %91, %92 : i64
                  %94 = ascscalar.bind_local_ptr %93 : memref<2056xi32, 1>
                  %95 = arith.select %89, %c1_i64, %c2_i64 : i64
                  %96 = arith.addi %arg32, %c1_i64 : i64
                  %97 = arith.remui %96, %c2_i64 : i64
                  %98 = arith.trunci %arg31 : i64 to i32
                  %99 = arith.muli %98, %c8_i32 overflow<nsw> : i32
                  %100 = arith.index_cast %99 : i32 to index
                  %101 = arith.muli %98, %c512_i32 overflow<nsw> : i32
                  %102 = arith.index_cast %101 : i32 to index
                  %103 = arith.addi %101, %c64_i32 overflow<nsw> : i32
                  %104 = arith.index_cast %103 : i32 to index
                  %105 = arith.addi %101, %c128_i32 overflow<nsw> : i32
                  %106 = arith.index_cast %105 : i32 to index
                  %107 = arith.addi %101, %c192_i32 overflow<nsw> : i32
                  %108 = arith.index_cast %107 : i32 to index
                  %109 = arith.addi %101, %c256_i32 overflow<nsw> : i32
                  %110 = arith.index_cast %109 : i32 to index
                  %111 = arith.addi %101, %c320_i32 overflow<nsw> : i32
                  %112 = arith.index_cast %111 : i32 to index
                  %113 = arith.addi %101, %c384_i32 overflow<nsw> : i32
                  %114 = arith.index_cast %113 : i32 to index
                  %115 = arith.addi %101, %c448_i32 overflow<nsw> : i32
                  %116 = arith.index_cast %115 : i32 to index
                  ascvec.get_buf v, %95, 0 {cannir.buffer.generated_sync}
                  ascvec.vf(%84, %100, %84, %84, %1, %102, %94, %104, %106, %108, %110, %112, %114, %116 : memref<?xi32, 1>, index, memref<?xi32, 1>, memref<?xi32, 1>, memref<?xui32, 1>, index, memref<2056xi32, 1>, index, index, index, index, index, index, index) {
                  ^bb0(%arg33: memref<?xi32, 1>, %arg34: index, %arg35: memref<?xi32, 1>, %arg36: memref<?xi32, 1>, %arg37: memref<?xui32, 1>, %arg38: index, %arg39: memref<2056xi32, 1>, %arg40: index, %arg41: index, %arg42: index, %arg43: index, %arg44: index, %arg45: index, %arg46: index):
                    %c2048 = arith.constant 2048 : index
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
                    %c2051_i32 = arith.constant 2051 : i32
                    %c2048_i32 = arith.constant 2048 : i32
                    %c8_i32_5 = arith.constant 8 : i32
                    %c1984_i32 = arith.constant 1984 : i32
                    %c1920_i32 = arith.constant 1920 : i32
                    %c1856_i32 = arith.constant 1856 : i32
                    %c1792_i32 = arith.constant 1792 : i32
                    %c1728_i32 = arith.constant 1728 : i32
                    %c1664_i32 = arith.constant 1664 : i32
                    %c1600_i32 = arith.constant 1600 : i32
                    %c1536_i32 = arith.constant 1536 : i32
                    %c1472_i32 = arith.constant 1472 : i32
                    %c1408_i32 = arith.constant 1408 : i32
                    %c1344_i32 = arith.constant 1344 : i32
                    %c1280_i32 = arith.constant 1280 : i32
                    %c1216_i32 = arith.constant 1216 : i32
                    %c1152_i32 = arith.constant 1152 : i32
                    %c1088_i32 = arith.constant 1088 : i32
                    %c1024_i32 = arith.constant 1024 : i32
                    %c960_i32 = arith.constant 960 : i32
                    %c896_i32 = arith.constant 896 : i32
                    %c832_i32 = arith.constant 832 : i32
                    %c768_i32 = arith.constant 768 : i32
                    %c704_i32 = arith.constant 704 : i32
                    %c640_i32 = arith.constant 640 : i32
                    %c576_i32 = arith.constant 576 : i32
                    %c512_i32_6 = arith.constant 512 : i32
                    %c448_i32_7 = arith.constant 448 : i32
                    %c384_i32_8 = arith.constant 384 : i32
                    %c320_i32_9 = arith.constant 320 : i32
                    %c256_i32_10 = arith.constant 256 : i32
                    %c192_i32_11 = arith.constant 192 : i32
                    %c128_i32_12 = arith.constant 128 : i32
                    %c0_13 = arith.constant 0 : index
                    %c4608_14 = arith.constant 4608 : index
                    %c3_i32 = arith.constant 3 : i32
                    %c0_i32_15 = arith.constant 0 : i32
                    %c-1_i32 = arith.constant -1 : i32
                    %c17344 = arith.constant 17344 : index
                    %c17328 = arith.constant 17328 : index
                    %c17312 = arith.constant 17312 : index
                    %c64_i32_16 = arith.constant 64 : i32
                    %mask, %remaining_out = ascvec.update_mask %c64_i32_16 : !ascvec.mask
                    %124 = arith.addi %arg34, %c17312 : index
                    %125 = ascvec.loadalign_brc %arg33[%124] : memref<?xi32, 1>, index -> <i32>
                    %126 = arith.addi %arg34, %c17328 : index
                    %127 = ascvec.loadalign_brc %arg35[%126] : memref<?xi32, 1>, index -> <i32>
                    %128 = arith.addi %arg34, %c17344 : index
                    %129 = ascvec.loadalign_brc %arg36[%128] : memref<?xi32, 1>, index -> <i32>
                    %130 = ascvec.shl %127, 2, %mask : <i32> -> <i32>
                    %131 = ascvec.shl %125, 2, %mask : <i32> -> <i32>
                    %132 = ascvec.sub %129, %131, %mask : <i32>
                    %133 = ascvec.add %130, %132, %mask : <i32>
                    %134 = ascvec.duplicate_scalar %c-1_i32, %mask : i32 -> <i32>
                    %135 = ascvec.arange %c0_i32_15 : i32 -> <i32>
                    %136 = ascvec.duplicate_scalar %c3_i32, %mask : i32 -> <i32>
                    %137 = ascvec.and %135, %136, %mask : <i32>, <i32> -> <i32>
                    %138 = arith.addi %arg38, %c4608_14 : index
                    %c0_17 = arith.constant 0 : index
                    %c1 = arith.constant 1 : index
                    scf.for %arg47 = %c0_17 to %c1 step %c1 {
                      %146 = ascvec.loadalign %arg37[%138] : memref<?xui32, 1>, index -> <ui32>
                      %147 = ascvec.reinterpret %146 : <ui32> -> <i32>
                      %148 = ascvec.shl %147, 2, %mask : <i32> -> <i32>
                      %low, %high = ascvec.intlv %148, %148 : <i32>, <i32> -> <i32>, <i32>
                      %low_32, %high_33 = ascvec.intlv %low, %low : <i32>, <i32> -> <i32>, <i32>
                      %low_34, %high_35 = ascvec.intlv %high, %high : <i32>, <i32> -> <i32>, <i32>
                      %149 = ascvec.cmp_lt %135, %130, %mask : <i32>, <i32> -> !ascvec.mask
                      %150 = ascvec.add %low_32, %137, %mask : <i32>
                      %151 = ascvec.sub %135, %130, %mask : <i32>
                      %152 = ascvec.add %131, %151, %mask : <i32>
                      %153 = ascvec.select %150, %152, %149 : <i32>
                      %154 = ascvec.cmp_lt %135, %133, %mask : <i32>, <i32> -> !ascvec.mask
                      %155 = ascvec.select %153, %134, %154 : <i32>
                      ascvec.storealign %arg39[%c0_13], %155, %mask : memref<2056xi32, 1>, index, <i32>
                      %156 = ascvec.arange %c64_i32_16 : i32 -> <i32>
                      %157 = ascvec.cmp_lt %156, %130, %mask : <i32>, <i32> -> !ascvec.mask
                      %158 = ascvec.add %high_33, %137, %mask : <i32>
                      %159 = ascvec.sub %156, %130, %mask : <i32>
                      %160 = ascvec.add %131, %159, %mask : <i32>
                      %161 = ascvec.select %158, %160, %157 : <i32>
                      %162 = ascvec.cmp_lt %156, %133, %mask : <i32>, <i32> -> !ascvec.mask
                      %163 = ascvec.select %161, %134, %162 : <i32>
                      ascvec.storealign %arg39[%c64], %163, %mask : memref<2056xi32, 1>, index, <i32>
                      %164 = ascvec.arange %c128_i32_12 : i32 -> <i32>
                      %165 = ascvec.cmp_lt %164, %130, %mask : <i32>, <i32> -> !ascvec.mask
                      %166 = ascvec.add %low_34, %137, %mask : <i32>
                      %167 = ascvec.sub %164, %130, %mask : <i32>
                      %168 = ascvec.add %131, %167, %mask : <i32>
                      %169 = ascvec.select %166, %168, %165 : <i32>
                      %170 = ascvec.cmp_lt %164, %133, %mask : <i32>, <i32> -> !ascvec.mask
                      %171 = ascvec.select %169, %134, %170 : <i32>
                      ascvec.storealign %arg39[%c128], %171, %mask : memref<2056xi32, 1>, index, <i32>
                      %172 = ascvec.arange %c192_i32_11 : i32 -> <i32>
                      %173 = ascvec.cmp_lt %172, %130, %mask : <i32>, <i32> -> !ascvec.mask
                      %174 = ascvec.add %high_35, %137, %mask : <i32>
                      %175 = ascvec.sub %172, %130, %mask : <i32>
                      %176 = ascvec.add %131, %175, %mask : <i32>
                      %177 = ascvec.select %174, %176, %173 : <i32>
                      %178 = ascvec.cmp_lt %172, %133, %mask : <i32>, <i32> -> !ascvec.mask
                      %179 = ascvec.select %177, %134, %178 : <i32>
                      ascvec.storealign %arg39[%c192], %179, %mask : memref<2056xi32, 1>, index, <i32>
                    }
                    %139 = arith.addi %arg40, %c4608_14 : index
                    %c0_18 = arith.constant 0 : index
                    %c1_19 = arith.constant 1 : index
                    scf.for %arg47 = %c0_18 to %c1_19 step %c1_19 {
                      %146 = ascvec.loadalign %arg37[%139] : memref<?xui32, 1>, index -> <ui32>
                      %147 = ascvec.reinterpret %146 : <ui32> -> <i32>
                      %148 = ascvec.shl %147, 2, %mask : <i32> -> <i32>
                      %low, %high = ascvec.intlv %148, %148 : <i32>, <i32> -> <i32>, <i32>
                      %low_32, %high_33 = ascvec.intlv %low, %low : <i32>, <i32> -> <i32>, <i32>
                      %low_34, %high_35 = ascvec.intlv %high, %high : <i32>, <i32> -> <i32>, <i32>
                      %149 = ascvec.arange %c256_i32_10 : i32 -> <i32>
                      %150 = ascvec.cmp_lt %149, %130, %mask : <i32>, <i32> -> !ascvec.mask
                      %151 = ascvec.add %low_32, %137, %mask : <i32>
                      %152 = ascvec.sub %149, %130, %mask : <i32>
                      %153 = ascvec.add %131, %152, %mask : <i32>
                      %154 = ascvec.select %151, %153, %150 : <i32>
                      %155 = ascvec.cmp_lt %149, %133, %mask : <i32>, <i32> -> !ascvec.mask
                      %156 = ascvec.select %154, %134, %155 : <i32>
                      ascvec.storealign %arg39[%c256], %156, %mask : memref<2056xi32, 1>, index, <i32>
                      %157 = ascvec.arange %c320_i32_9 : i32 -> <i32>
                      %158 = ascvec.cmp_lt %157, %130, %mask : <i32>, <i32> -> !ascvec.mask
                      %159 = ascvec.add %high_33, %137, %mask : <i32>
                      %160 = ascvec.sub %157, %130, %mask : <i32>
                      %161 = ascvec.add %131, %160, %mask : <i32>
                      %162 = ascvec.select %159, %161, %158 : <i32>
                      %163 = ascvec.cmp_lt %157, %133, %mask : <i32>, <i32> -> !ascvec.mask
                      %164 = ascvec.select %162, %134, %163 : <i32>
                      ascvec.storealign %arg39[%c320], %164, %mask : memref<2056xi32, 1>, index, <i32>
                      %165 = ascvec.arange %c384_i32_8 : i32 -> <i32>
                      %166 = ascvec.cmp_lt %165, %130, %mask : <i32>, <i32> -> !ascvec.mask
                      %167 = ascvec.add %low_34, %137, %mask : <i32>
                      %168 = ascvec.sub %165, %130, %mask : <i32>
                      %169 = ascvec.add %131, %168, %mask : <i32>
                      %170 = ascvec.select %167, %169, %166 : <i32>
                      %171 = ascvec.cmp_lt %165, %133, %mask : <i32>, <i32> -> !ascvec.mask
                      %172 = ascvec.select %170, %134, %171 : <i32>
                      ascvec.storealign %arg39[%c384], %172, %mask : memref<2056xi32, 1>, index, <i32>
                      %173 = ascvec.arange %c448_i32_7 : i32 -> <i32>
                      %174 = ascvec.cmp_lt %173, %130, %mask : <i32>, <i32> -> !ascvec.mask
                      %175 = ascvec.add %high_35, %137, %mask : <i32>
                      %176 = ascvec.sub %173, %130, %mask : <i32>
                      %177 = ascvec.add %131, %176, %mask : <i32>
                      %178 = ascvec.select %175, %177, %174 : <i32>
                      %179 = ascvec.cmp_lt %173, %133, %mask : <i32>, <i32> -> !ascvec.mask
                      %180 = ascvec.select %178, %134, %179 : <i32>
                      ascvec.storealign %arg39[%c448], %180, %mask : memref<2056xi32, 1>, index, <i32>
                    }
                    %140 = arith.addi %arg41, %c4608_14 : index
                    %c0_20 = arith.constant 0 : index
                    %c1_21 = arith.constant 1 : index
                    scf.for %arg47 = %c0_20 to %c1_21 step %c1_21 {
                      %146 = ascvec.loadalign %arg37[%140] : memref<?xui32, 1>, index -> <ui32>
                      %147 = ascvec.reinterpret %146 : <ui32> -> <i32>
                      %148 = ascvec.shl %147, 2, %mask : <i32> -> <i32>
                      %low, %high = ascvec.intlv %148, %148 : <i32>, <i32> -> <i32>, <i32>
                      %low_32, %high_33 = ascvec.intlv %low, %low : <i32>, <i32> -> <i32>, <i32>
                      %low_34, %high_35 = ascvec.intlv %high, %high : <i32>, <i32> -> <i32>, <i32>
                      %149 = ascvec.arange %c512_i32_6 : i32 -> <i32>
                      %150 = ascvec.cmp_lt %149, %130, %mask : <i32>, <i32> -> !ascvec.mask
                      %151 = ascvec.add %low_32, %137, %mask : <i32>
                      %152 = ascvec.sub %149, %130, %mask : <i32>
                      %153 = ascvec.add %131, %152, %mask : <i32>
                      %154 = ascvec.select %151, %153, %150 : <i32>
                      %155 = ascvec.cmp_lt %149, %133, %mask : <i32>, <i32> -> !ascvec.mask
                      %156 = ascvec.select %154, %134, %155 : <i32>
                      ascvec.storealign %arg39[%c512], %156, %mask : memref<2056xi32, 1>, index, <i32>
                      %157 = ascvec.arange %c576_i32 : i32 -> <i32>
                      %158 = ascvec.cmp_lt %157, %130, %mask : <i32>, <i32> -> !ascvec.mask
                      %159 = ascvec.add %high_33, %137, %mask : <i32>
                      %160 = ascvec.sub %157, %130, %mask : <i32>
                      %161 = ascvec.add %131, %160, %mask : <i32>
                      %162 = ascvec.select %159, %161, %158 : <i32>
                      %163 = ascvec.cmp_lt %157, %133, %mask : <i32>, <i32> -> !ascvec.mask
                      %164 = ascvec.select %162, %134, %163 : <i32>
                      ascvec.storealign %arg39[%c576], %164, %mask : memref<2056xi32, 1>, index, <i32>
                      %165 = ascvec.arange %c640_i32 : i32 -> <i32>
                      %166 = ascvec.cmp_lt %165, %130, %mask : <i32>, <i32> -> !ascvec.mask
                      %167 = ascvec.add %low_34, %137, %mask : <i32>
                      %168 = ascvec.sub %165, %130, %mask : <i32>
                      %169 = ascvec.add %131, %168, %mask : <i32>
                      %170 = ascvec.select %167, %169, %166 : <i32>
                      %171 = ascvec.cmp_lt %165, %133, %mask : <i32>, <i32> -> !ascvec.mask
                      %172 = ascvec.select %170, %134, %171 : <i32>
                      ascvec.storealign %arg39[%c640], %172, %mask : memref<2056xi32, 1>, index, <i32>
                      %173 = ascvec.arange %c704_i32 : i32 -> <i32>
                      %174 = ascvec.cmp_lt %173, %130, %mask : <i32>, <i32> -> !ascvec.mask
                      %175 = ascvec.add %high_35, %137, %mask : <i32>
                      %176 = ascvec.sub %173, %130, %mask : <i32>
                      %177 = ascvec.add %131, %176, %mask : <i32>
                      %178 = ascvec.select %175, %177, %174 : <i32>
                      %179 = ascvec.cmp_lt %173, %133, %mask : <i32>, <i32> -> !ascvec.mask
                      %180 = ascvec.select %178, %134, %179 : <i32>
                      ascvec.storealign %arg39[%c704], %180, %mask : memref<2056xi32, 1>, index, <i32>
                    }
                    %141 = arith.addi %arg42, %c4608_14 : index
                    %c0_22 = arith.constant 0 : index
                    %c1_23 = arith.constant 1 : index
                    scf.for %arg47 = %c0_22 to %c1_23 step %c1_23 {
                      %146 = ascvec.loadalign %arg37[%141] : memref<?xui32, 1>, index -> <ui32>
                      %147 = ascvec.reinterpret %146 : <ui32> -> <i32>
                      %148 = ascvec.shl %147, 2, %mask : <i32> -> <i32>
                      %low, %high = ascvec.intlv %148, %148 : <i32>, <i32> -> <i32>, <i32>
                      %low_32, %high_33 = ascvec.intlv %low, %low : <i32>, <i32> -> <i32>, <i32>
                      %low_34, %high_35 = ascvec.intlv %high, %high : <i32>, <i32> -> <i32>, <i32>
                      %149 = ascvec.arange %c768_i32 : i32 -> <i32>
                      %150 = ascvec.cmp_lt %149, %130, %mask : <i32>, <i32> -> !ascvec.mask
                      %151 = ascvec.add %low_32, %137, %mask : <i32>
                      %152 = ascvec.sub %149, %130, %mask : <i32>
                      %153 = ascvec.add %131, %152, %mask : <i32>
                      %154 = ascvec.select %151, %153, %150 : <i32>
                      %155 = ascvec.cmp_lt %149, %133, %mask : <i32>, <i32> -> !ascvec.mask
                      %156 = ascvec.select %154, %134, %155 : <i32>
                      ascvec.storealign %arg39[%c768], %156, %mask : memref<2056xi32, 1>, index, <i32>
                      %157 = ascvec.arange %c832_i32 : i32 -> <i32>
                      %158 = ascvec.cmp_lt %157, %130, %mask : <i32>, <i32> -> !ascvec.mask
                      %159 = ascvec.add %high_33, %137, %mask : <i32>
                      %160 = ascvec.sub %157, %130, %mask : <i32>
                      %161 = ascvec.add %131, %160, %mask : <i32>
                      %162 = ascvec.select %159, %161, %158 : <i32>
                      %163 = ascvec.cmp_lt %157, %133, %mask : <i32>, <i32> -> !ascvec.mask
                      %164 = ascvec.select %162, %134, %163 : <i32>
                      ascvec.storealign %arg39[%c832], %164, %mask : memref<2056xi32, 1>, index, <i32>
                      %165 = ascvec.arange %c896_i32 : i32 -> <i32>
                      %166 = ascvec.cmp_lt %165, %130, %mask : <i32>, <i32> -> !ascvec.mask
                      %167 = ascvec.add %low_34, %137, %mask : <i32>
                      %168 = ascvec.sub %165, %130, %mask : <i32>
                      %169 = ascvec.add %131, %168, %mask : <i32>
                      %170 = ascvec.select %167, %169, %166 : <i32>
                      %171 = ascvec.cmp_lt %165, %133, %mask : <i32>, <i32> -> !ascvec.mask
                      %172 = ascvec.select %170, %134, %171 : <i32>
                      ascvec.storealign %arg39[%c896], %172, %mask : memref<2056xi32, 1>, index, <i32>
                      %173 = ascvec.arange %c960_i32 : i32 -> <i32>
                      %174 = ascvec.cmp_lt %173, %130, %mask : <i32>, <i32> -> !ascvec.mask
                      %175 = ascvec.add %high_35, %137, %mask : <i32>
                      %176 = ascvec.sub %173, %130, %mask : <i32>
                      %177 = ascvec.add %131, %176, %mask : <i32>
                      %178 = ascvec.select %175, %177, %174 : <i32>
                      %179 = ascvec.cmp_lt %173, %133, %mask : <i32>, <i32> -> !ascvec.mask
                      %180 = ascvec.select %178, %134, %179 : <i32>
                      ascvec.storealign %arg39[%c960], %180, %mask : memref<2056xi32, 1>, index, <i32>
                    }
                    %142 = arith.addi %arg43, %c4608_14 : index
                    %c0_24 = arith.constant 0 : index
                    %c1_25 = arith.constant 1 : index
                    scf.for %arg47 = %c0_24 to %c1_25 step %c1_25 {
                      %146 = ascvec.loadalign %arg37[%142] : memref<?xui32, 1>, index -> <ui32>
                      %147 = ascvec.reinterpret %146 : <ui32> -> <i32>
                      %148 = ascvec.shl %147, 2, %mask : <i32> -> <i32>
                      %low, %high = ascvec.intlv %148, %148 : <i32>, <i32> -> <i32>, <i32>
                      %low_32, %high_33 = ascvec.intlv %low, %low : <i32>, <i32> -> <i32>, <i32>
                      %low_34, %high_35 = ascvec.intlv %high, %high : <i32>, <i32> -> <i32>, <i32>
                      %149 = ascvec.arange %c1024_i32 : i32 -> <i32>
                      %150 = ascvec.cmp_lt %149, %130, %mask : <i32>, <i32> -> !ascvec.mask
                      %151 = ascvec.add %low_32, %137, %mask : <i32>
                      %152 = ascvec.sub %149, %130, %mask : <i32>
                      %153 = ascvec.add %131, %152, %mask : <i32>
                      %154 = ascvec.select %151, %153, %150 : <i32>
                      %155 = ascvec.cmp_lt %149, %133, %mask : <i32>, <i32> -> !ascvec.mask
                      %156 = ascvec.select %154, %134, %155 : <i32>
                      ascvec.storealign %arg39[%c1024], %156, %mask : memref<2056xi32, 1>, index, <i32>
                      %157 = ascvec.arange %c1088_i32 : i32 -> <i32>
                      %158 = ascvec.cmp_lt %157, %130, %mask : <i32>, <i32> -> !ascvec.mask
                      %159 = ascvec.add %high_33, %137, %mask : <i32>
                      %160 = ascvec.sub %157, %130, %mask : <i32>
                      %161 = ascvec.add %131, %160, %mask : <i32>
                      %162 = ascvec.select %159, %161, %158 : <i32>
                      %163 = ascvec.cmp_lt %157, %133, %mask : <i32>, <i32> -> !ascvec.mask
                      %164 = ascvec.select %162, %134, %163 : <i32>
                      ascvec.storealign %arg39[%c1088], %164, %mask : memref<2056xi32, 1>, index, <i32>
                      %165 = ascvec.arange %c1152_i32 : i32 -> <i32>
                      %166 = ascvec.cmp_lt %165, %130, %mask : <i32>, <i32> -> !ascvec.mask
                      %167 = ascvec.add %low_34, %137, %mask : <i32>
                      %168 = ascvec.sub %165, %130, %mask : <i32>
                      %169 = ascvec.add %131, %168, %mask : <i32>
                      %170 = ascvec.select %167, %169, %166 : <i32>
                      %171 = ascvec.cmp_lt %165, %133, %mask : <i32>, <i32> -> !ascvec.mask
                      %172 = ascvec.select %170, %134, %171 : <i32>
                      ascvec.storealign %arg39[%c1152], %172, %mask : memref<2056xi32, 1>, index, <i32>
                      %173 = ascvec.arange %c1216_i32 : i32 -> <i32>
                      %174 = ascvec.cmp_lt %173, %130, %mask : <i32>, <i32> -> !ascvec.mask
                      %175 = ascvec.add %high_35, %137, %mask : <i32>
                      %176 = ascvec.sub %173, %130, %mask : <i32>
                      %177 = ascvec.add %131, %176, %mask : <i32>
                      %178 = ascvec.select %175, %177, %174 : <i32>
                      %179 = ascvec.cmp_lt %173, %133, %mask : <i32>, <i32> -> !ascvec.mask
                      %180 = ascvec.select %178, %134, %179 : <i32>
                      ascvec.storealign %arg39[%c1216], %180, %mask : memref<2056xi32, 1>, index, <i32>
                    }
                    %143 = arith.addi %arg44, %c4608_14 : index
                    %c0_26 = arith.constant 0 : index
                    %c1_27 = arith.constant 1 : index
                    scf.for %arg47 = %c0_26 to %c1_27 step %c1_27 {
                      %146 = ascvec.loadalign %arg37[%143] : memref<?xui32, 1>, index -> <ui32>
                      %147 = ascvec.reinterpret %146 : <ui32> -> <i32>
                      %148 = ascvec.shl %147, 2, %mask : <i32> -> <i32>
                      %low, %high = ascvec.intlv %148, %148 : <i32>, <i32> -> <i32>, <i32>
                      %low_32, %high_33 = ascvec.intlv %low, %low : <i32>, <i32> -> <i32>, <i32>
                      %low_34, %high_35 = ascvec.intlv %high, %high : <i32>, <i32> -> <i32>, <i32>
                      %149 = ascvec.arange %c1280_i32 : i32 -> <i32>
                      %150 = ascvec.cmp_lt %149, %130, %mask : <i32>, <i32> -> !ascvec.mask
                      %151 = ascvec.add %low_32, %137, %mask : <i32>
                      %152 = ascvec.sub %149, %130, %mask : <i32>
                      %153 = ascvec.add %131, %152, %mask : <i32>
                      %154 = ascvec.select %151, %153, %150 : <i32>
                      %155 = ascvec.cmp_lt %149, %133, %mask : <i32>, <i32> -> !ascvec.mask
                      %156 = ascvec.select %154, %134, %155 : <i32>
                      ascvec.storealign %arg39[%c1280], %156, %mask : memref<2056xi32, 1>, index, <i32>
                      %157 = ascvec.arange %c1344_i32 : i32 -> <i32>
                      %158 = ascvec.cmp_lt %157, %130, %mask : <i32>, <i32> -> !ascvec.mask
                      %159 = ascvec.add %high_33, %137, %mask : <i32>
                      %160 = ascvec.sub %157, %130, %mask : <i32>
                      %161 = ascvec.add %131, %160, %mask : <i32>
                      %162 = ascvec.select %159, %161, %158 : <i32>
                      %163 = ascvec.cmp_lt %157, %133, %mask : <i32>, <i32> -> !ascvec.mask
                      %164 = ascvec.select %162, %134, %163 : <i32>
                      ascvec.storealign %arg39[%c1344], %164, %mask : memref<2056xi32, 1>, index, <i32>
                      %165 = ascvec.arange %c1408_i32 : i32 -> <i32>
                      %166 = ascvec.cmp_lt %165, %130, %mask : <i32>, <i32> -> !ascvec.mask
                      %167 = ascvec.add %low_34, %137, %mask : <i32>
                      %168 = ascvec.sub %165, %130, %mask : <i32>
                      %169 = ascvec.add %131, %168, %mask : <i32>
                      %170 = ascvec.select %167, %169, %166 : <i32>
                      %171 = ascvec.cmp_lt %165, %133, %mask : <i32>, <i32> -> !ascvec.mask
                      %172 = ascvec.select %170, %134, %171 : <i32>
                      ascvec.storealign %arg39[%c1408], %172, %mask : memref<2056xi32, 1>, index, <i32>
                      %173 = ascvec.arange %c1472_i32 : i32 -> <i32>
                      %174 = ascvec.cmp_lt %173, %130, %mask : <i32>, <i32> -> !ascvec.mask
                      %175 = ascvec.add %high_35, %137, %mask : <i32>
                      %176 = ascvec.sub %173, %130, %mask : <i32>
                      %177 = ascvec.add %131, %176, %mask : <i32>
                      %178 = ascvec.select %175, %177, %174 : <i32>
                      %179 = ascvec.cmp_lt %173, %133, %mask : <i32>, <i32> -> !ascvec.mask
                      %180 = ascvec.select %178, %134, %179 : <i32>
                      ascvec.storealign %arg39[%c1472], %180, %mask : memref<2056xi32, 1>, index, <i32>
                    }
                    %144 = arith.addi %arg45, %c4608_14 : index
                    %c0_28 = arith.constant 0 : index
                    %c1_29 = arith.constant 1 : index
                    scf.for %arg47 = %c0_28 to %c1_29 step %c1_29 {
                      %146 = ascvec.loadalign %arg37[%144] : memref<?xui32, 1>, index -> <ui32>
                      %147 = ascvec.reinterpret %146 : <ui32> -> <i32>
                      %148 = ascvec.shl %147, 2, %mask : <i32> -> <i32>
                      %low, %high = ascvec.intlv %148, %148 : <i32>, <i32> -> <i32>, <i32>
                      %low_32, %high_33 = ascvec.intlv %low, %low : <i32>, <i32> -> <i32>, <i32>
                      %low_34, %high_35 = ascvec.intlv %high, %high : <i32>, <i32> -> <i32>, <i32>
                      %149 = ascvec.arange %c1536_i32 : i32 -> <i32>
                      %150 = ascvec.cmp_lt %149, %130, %mask : <i32>, <i32> -> !ascvec.mask
                      %151 = ascvec.add %low_32, %137, %mask : <i32>
                      %152 = ascvec.sub %149, %130, %mask : <i32>
                      %153 = ascvec.add %131, %152, %mask : <i32>
                      %154 = ascvec.select %151, %153, %150 : <i32>
                      %155 = ascvec.cmp_lt %149, %133, %mask : <i32>, <i32> -> !ascvec.mask
                      %156 = ascvec.select %154, %134, %155 : <i32>
                      ascvec.storealign %arg39[%c1536], %156, %mask : memref<2056xi32, 1>, index, <i32>
                      %157 = ascvec.arange %c1600_i32 : i32 -> <i32>
                      %158 = ascvec.cmp_lt %157, %130, %mask : <i32>, <i32> -> !ascvec.mask
                      %159 = ascvec.add %high_33, %137, %mask : <i32>
                      %160 = ascvec.sub %157, %130, %mask : <i32>
                      %161 = ascvec.add %131, %160, %mask : <i32>
                      %162 = ascvec.select %159, %161, %158 : <i32>
                      %163 = ascvec.cmp_lt %157, %133, %mask : <i32>, <i32> -> !ascvec.mask
                      %164 = ascvec.select %162, %134, %163 : <i32>
                      ascvec.storealign %arg39[%c1600], %164, %mask : memref<2056xi32, 1>, index, <i32>
                      %165 = ascvec.arange %c1664_i32 : i32 -> <i32>
                      %166 = ascvec.cmp_lt %165, %130, %mask : <i32>, <i32> -> !ascvec.mask
                      %167 = ascvec.add %low_34, %137, %mask : <i32>
                      %168 = ascvec.sub %165, %130, %mask : <i32>
                      %169 = ascvec.add %131, %168, %mask : <i32>
                      %170 = ascvec.select %167, %169, %166 : <i32>
                      %171 = ascvec.cmp_lt %165, %133, %mask : <i32>, <i32> -> !ascvec.mask
                      %172 = ascvec.select %170, %134, %171 : <i32>
                      ascvec.storealign %arg39[%c1664], %172, %mask : memref<2056xi32, 1>, index, <i32>
                      %173 = ascvec.arange %c1728_i32 : i32 -> <i32>
                      %174 = ascvec.cmp_lt %173, %130, %mask : <i32>, <i32> -> !ascvec.mask
                      %175 = ascvec.add %high_35, %137, %mask : <i32>
                      %176 = ascvec.sub %173, %130, %mask : <i32>
                      %177 = ascvec.add %131, %176, %mask : <i32>
                      %178 = ascvec.select %175, %177, %174 : <i32>
                      %179 = ascvec.cmp_lt %173, %133, %mask : <i32>, <i32> -> !ascvec.mask
                      %180 = ascvec.select %178, %134, %179 : <i32>
                      ascvec.storealign %arg39[%c1728], %180, %mask : memref<2056xi32, 1>, index, <i32>
                    }
                    %145 = arith.addi %arg46, %c4608_14 : index
                    %c0_30 = arith.constant 0 : index
                    %c1_31 = arith.constant 1 : index
                    scf.for %arg47 = %c0_30 to %c1_31 step %c1_31 {
                      %146 = ascvec.loadalign %arg37[%145] : memref<?xui32, 1>, index -> <ui32>
                      %147 = ascvec.reinterpret %146 : <ui32> -> <i32>
                      %148 = ascvec.shl %147, 2, %mask : <i32> -> <i32>
                      %low, %high = ascvec.intlv %148, %148 : <i32>, <i32> -> <i32>, <i32>
                      %low_32, %high_33 = ascvec.intlv %low, %low : <i32>, <i32> -> <i32>, <i32>
                      %low_34, %high_35 = ascvec.intlv %high, %high : <i32>, <i32> -> <i32>, <i32>
                      %149 = ascvec.arange %c1792_i32 : i32 -> <i32>
                      %150 = ascvec.cmp_lt %149, %130, %mask : <i32>, <i32> -> !ascvec.mask
                      %151 = ascvec.add %low_32, %137, %mask : <i32>
                      %152 = ascvec.sub %149, %130, %mask : <i32>
                      %153 = ascvec.add %131, %152, %mask : <i32>
                      %154 = ascvec.select %151, %153, %150 : <i32>
                      %155 = ascvec.cmp_lt %149, %133, %mask : <i32>, <i32> -> !ascvec.mask
                      %156 = ascvec.select %154, %134, %155 : <i32>
                      ascvec.storealign %arg39[%c1792], %156, %mask : memref<2056xi32, 1>, index, <i32>
                      %157 = ascvec.arange %c1856_i32 : i32 -> <i32>
                      %158 = ascvec.cmp_lt %157, %130, %mask : <i32>, <i32> -> !ascvec.mask
                      %159 = ascvec.add %high_33, %137, %mask : <i32>
                      %160 = ascvec.sub %157, %130, %mask : <i32>
                      %161 = ascvec.add %131, %160, %mask : <i32>
                      %162 = ascvec.select %159, %161, %158 : <i32>
                      %163 = ascvec.cmp_lt %157, %133, %mask : <i32>, <i32> -> !ascvec.mask
                      %164 = ascvec.select %162, %134, %163 : <i32>
                      ascvec.storealign %arg39[%c1856], %164, %mask : memref<2056xi32, 1>, index, <i32>
                      %165 = ascvec.arange %c1920_i32 : i32 -> <i32>
                      %166 = ascvec.cmp_lt %165, %130, %mask : <i32>, <i32> -> !ascvec.mask
                      %167 = ascvec.add %low_34, %137, %mask : <i32>
                      %168 = ascvec.sub %165, %130, %mask : <i32>
                      %169 = ascvec.add %131, %168, %mask : <i32>
                      %170 = ascvec.select %167, %169, %166 : <i32>
                      %171 = ascvec.cmp_lt %165, %133, %mask : <i32>, <i32> -> !ascvec.mask
                      %172 = ascvec.select %170, %134, %171 : <i32>
                      ascvec.storealign %arg39[%c1920], %172, %mask : memref<2056xi32, 1>, index, <i32>
                      %173 = ascvec.arange %c1984_i32 : i32 -> <i32>
                      %174 = ascvec.cmp_lt %173, %130, %mask : <i32>, <i32> -> !ascvec.mask
                      %175 = ascvec.add %high_35, %137, %mask : <i32>
                      %176 = ascvec.sub %173, %130, %mask : <i32>
                      %177 = ascvec.add %131, %176, %mask : <i32>
                      %178 = ascvec.select %175, %177, %174 : <i32>
                      %179 = ascvec.cmp_lt %173, %133, %mask : <i32>, <i32> -> !ascvec.mask
                      %180 = ascvec.select %178, %134, %179 : <i32>
                      ascvec.storealign %arg39[%c1984], %180, %mask : memref<2056xi32, 1>, index, <i32>
                      %mask_36, %remaining_out_37 = ascvec.update_mask %c8_i32_5 : !ascvec.mask
                      %181 = ascvec.arange %c2048_i32 : i32 -> <i32>
                      %182 = ascvec.sub %181, %130, %mask : <i32>
                      %183 = ascvec.add %131, %182, %mask : <i32>
                      %184 = ascvec.cmp_lt %181, %133, %mask_36 : <i32>, <i32> -> !ascvec.mask
                      %185 = ascvec.select %183, %134, %184 : <i32>
                      %186 = ascvec.cmp_eq_scalar %181, %c2051_i32, %mask_36 : <i32>, i32 -> !ascvec.mask
                      %187 = ascvec.select %133, %185, %186 : <i32>
                      ascvec.storealign %arg39[%c2048], %187, %mask_36 : memref<2056xi32, 1>, index, <i32>
                    }
                  } {cannir.vf_mode = "simd"}
                  ascvec.release_buf v, %95, 0 {cannir.buffer.generated_sync}
                  %117 = arith.addi %49, %86 : i64
                  %118 = arith.muli %117, %c65792_i64 : i64
                  %119 = ascscalar.memref_address %alloc_2 : memref<1xi32>
                  %120 = arith.divui %118, %c8_i64 : i64
                  %121 = arith.addi %119, %120 : i64
                  %122 = ascscalar.bind_gm_ptr %121 : memref<1xi32>
                  %123 = ascscalar.bind_local_ptr %93 : memref<1xi32, 1>
                  ascvec.get_buf mte3, %95, 0 {cannir.buffer.generated_sync}
                  ascvec.copy_ub2gm %122[%c0], %123[%c0], %c1_i32, %c8224_i32, %c0_i64, %c8224_i32 l2_cache_ctl %c0_i32 : memref<1xi32>, memref<1xi32, 1>
                  ascvec.release_buf mte3, %95, 0 {cannir.buffer.generated_sync}
                  scf.yield %97 : i64
                } else {
                  scf.yield %arg32 : i64
                }
                scf.yield %88 : i64
              }
              scf.yield %85 : i64
            }
            scf.yield %arg23, %arg24, %74, %arg26, %arg27, %arg28 : i64, i64, i64, i64, i64, i64
          }
          scf.yield %73#0, %73#1, %73#2, %73#3, %73#4, %73#5 : i64, i64, i64, i64, i64, i64
        }
        scf.yield %47#0, %47#1, %47#2, %47#3, %47#4, %47#5 : i64, i64, i64, i64, i64, i64
      }
      scf.yield %26#0, %26#1, %26#2, %26#3, %26#4, %26#5 : i64, i64, i64, i64, i64, i64
    }
    return
  }
  func.func @kernel_QSAIndexerKernel___call___0_aic(%arg0: !cannir.tensor<<gm, bf16>, <<(-1, 128)>, <(128, 1)>>, <<(-1, 128)>, <(128, 1)>>, identity>, %arg1: !cannir.tensor<<gm, bf16>, <<(-1, 256, 128)>, <(32768, 128, 1)>>, <<(-1, 256, 128)>, <(32768, 128, 1)>>, identity>, %arg2: !cannir.tensor<<gm, i32>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg3: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg4: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg5: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg6: !cannir.tensor<<gm, i32>, <<(-1, 2056)>, <(2056, 1)>>, <<(-1, 2056)>, <(2056, 1)>>, identity>, %arg7: !cannir.tensor<<gm, ui16>, <<(1024, -1)>, <(-1, 1)>>, <<(1024, -1)>, <(-1, 1)>>, identity>) attributes {cannir.buffer.auto_sync, cannir.buffer.sync_analyzed, cannir.buffer.sync_verified, cannir.buffer.tensor_state, cannir.core = "aic", cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.device, cannir.explicit_ring_buffer_cursors, cannir.kernel_group = "kernel_QSAIndexerKernel___call___0", cannir.resource.ids_prepared} {
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %c2_i64 = arith.constant 2 : i64
    %c3_i64 = arith.constant 3 : i64
    %c4_i64 = arith.constant 4 : i64
    %c5_i64 = arith.constant 5 : i64
    %c6_i64 = arith.constant 6 : i64
    %c7_i64 = arith.constant 7 : i64
    %c8_i64 = arith.constant 8 : i64
    %c16_i64 = arith.constant 16 : i64
    %c17_i64 = arith.constant 17 : i64
    %c31_i64 = arith.constant 31 : i64
    %c32_i64 = arith.constant 32 : i64
    %c32768_i64 = arith.constant 32768 : i64
    %c4095_i64 = arith.constant 4095 : i64
    %c255_i64 = arith.constant 255 : i64
    %c32 = arith.constant 32 : index
    %c8_i32 = arith.constant 8 : i32
    %c512_i64 = arith.constant 512 : i64
    %c0 = arith.constant 0 : index
    %c256_i32 = arith.constant 256 : i32
    %c16_i32 = arith.constant 16 : i32
    %c128_i32 = arith.constant 128 : i32
    %c65536_i64 = arith.constant 65536 : i64
    %c1474560_i64 = arith.constant 1474560 : i64
    %c950272_i64 = arith.constant 950272 : i64
    %c128_i64 = arith.constant 128 : i64
    %c4096_i64 = arith.constant 4096 : i64
    %c256_i64 = arith.constant 256 : i64
    %c576_i64 = arith.constant 576 : i64
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %alloc = memref.alloc() {cannir.arg_index = 0 : i64} : memref<1xbf16>
    %alloc_0 = memref.alloc() {cannir.arg_index = 1 : i64} : memref<1xbf16>
    %alloc_1 = memref.alloc() {cannir.arg_index = 2 : i64} : memref<1xi32>
    %alloc_2 = memref.alloc() {cannir.arg_index = 3 : i64} : memref<1xi32>
    %alloc_3 = memref.alloc() {cannir.arg_index = 4 : i64} : memref<1xi32>
    %alloc_4 = memref.alloc() {cannir.arg_index = 5 : i64} : memref<1xi32>
    %alloc_5 = memref.alloc() {cannir.byte_offset = 0 : i64, cannir.onchip_space_root} : memref<262144xui8, 1>
    %alloc_6 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<16384xbf16, 2>
    %alloc_7 = memref.alloc() {cannir.byte_offset = 32768 : i64} : memref<16384xbf16, 4>
    %alloc_8 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<16384xbf16, 4>
    %alloc_9 = memref.alloc() {cannir.byte_offset = 32768 : i64} : memref<16384xbf16, 3>
    %alloc_10 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<16384xbf16, 3>
    %alloc_11 = memref.alloc() {cannir.byte_offset = 131072 : i64} : memref<32768xf32, 5>
    %alloc_12 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<32768xf32, 5>
    %alloc_13 = memref.alloc() {cannir.byte_offset = 98304 : i64} : memref<32768xbf16, 2>
    %alloc_14 = memref.alloc() {cannir.byte_offset = 32768 : i64} : memref<32768xbf16, 2>
    %0 = ascscalar.get_subblock_id
    %1 = memref.load %alloc_4[%c0] : memref<1xi32>
    %2 = arith.extsi %1 : i32 to i64
    %3:6 = scf.for %arg8 = %c0_i64 to %2 step %c1_i64 iter_args(%arg9 = %c0_i64, %arg10 = %c0_i64, %arg11 = %c0_i64, %arg12 = %c0_i64, %arg13 = %c0_i64, %arg14 = %c0_i64) -> (i64, i64, i64, i64, i64, i64)  : i64 {
      %4 = arith.muli %arg8, %c576_i64 : i64
      %5 = arith.addi %4, %c16_i64 : i64
      %6 = ascscalar.get_block_idx
      %7 = arith.muli %6, %c16_i64 : i64
      %8 = arith.addi %5, %7 : i64
      %9 = arith.index_cast %8 : i64 to index
      %10 = memref.load %alloc_4[%9] : memref<1xi32>
      %11 = arith.extsi %10 : i32 to i64
      %12 = arith.addi %8, %c1_i64 : i64
      %13 = arith.index_cast %12 : i64 to index
      %14 = memref.load %alloc_4[%13] : memref<1xi32>
      %15 = arith.extsi %14 : i32 to i64
      %16 = arith.addi %8, %c3_i64 : i64
      %17 = arith.index_cast %16 : i64 to index
      %18 = memref.load %alloc_4[%17] : memref<1xi32>
      %19 = arith.extsi %18 : i32 to i64
      %20 = arith.addi %8, %c4_i64 : i64
      %21 = arith.index_cast %20 : i64 to index
      %22 = memref.load %alloc_4[%21] : memref<1xi32>
      %23 = arith.extsi %22 : i32 to i64
      %24 = arith.addi %19, %c1_i64 : i64
      %25:6 = scf.for %arg15 = %11 to %24 step %c1_i64 iter_args(%arg16 = %arg9, %arg17 = %arg10, %arg18 = %arg11, %arg19 = %arg12, %arg20 = %arg13, %arg21 = %arg14) -> (i64, i64, i64, i64, i64, i64)  : i64 {
        %26 = arith.index_cast %arg15 : i64 to index
        %27 = memref.load %alloc_2[%26] : memref<1xi32>
        %28 = arith.extsi %27 : i32 to i64
        %29 = arith.addi %arg15, %c1_i64 : i64
        %30 = arith.index_cast %29 : i64 to index
        %31 = memref.load %alloc_2[%30] : memref<1xi32>
        %32 = arith.extsi %31 : i32 to i64
        %33 = arith.cmpi eq, %arg15, %11 : i64
        %34 = arith.select %33, %15, %c0_i64 : i64
        %35 = arith.subi %32, %28 : i64
        %36 = arith.addi %35, %c31_i64 : i64
        %37 = arith.remsi %36, %c32_i64 : i64
        %38 = arith.cmpi ne, %37, %c0_i64 : i64
        %39 = arith.cmpi slt, %37, %c0_i64 : i64
        %40 = arith.andi %38, %39 : i1
        %41 = arith.divsi %36, %c32_i64 : i64
        %42 = arith.extui %40 : i1 to i64
        %43 = arith.subi %41, %42 : i64
        %44 = arith.cmpi eq, %arg15, %19 : i64
        %45 = arith.select %44, %23, %43 : i64
        %46:6 = scf.for %arg22 = %34 to %45 step %c1_i64 iter_args(%arg23 = %arg16, %arg24 = %arg17, %arg25 = %arg18, %arg26 = %arg19, %arg27 = %arg20, %arg28 = %arg21) -> (i64, i64, i64, i64, i64, i64)  : i64 {
          %47 = arith.muli %arg22, %c32_i64 : i64
          %48 = arith.addi %28, %47 : i64
          %49 = arith.subi %32, %48 : i64
          %50 = arith.minsi %49, %c32_i64 : i64
          %51 = scf.for %arg29 = %c0_i64 to %50 step %c1_i64 iter_args(%arg30 = %c0_i64) -> (i64)  : i64 {
            %70 = arith.addi %48, %arg29 : i64
            %71 = arith.index_cast %70 : i64 to index
            %72 = memref.load %alloc_3[%71] : memref<1xi32>
            %73 = arith.extsi %72 : i32 to i64
            %74 = arith.addi %73, %c1_i64 : i64
            %75 = arith.remsi %74, %c4_i64 : i64
            %76 = arith.cmpi ne, %75, %c0_i64 : i64
            %77 = arith.cmpi slt, %75, %c0_i64 : i64
            %78 = arith.andi %76, %77 : i1
            %79 = arith.divsi %74, %c4_i64 : i64
            %80 = arith.extui %78 : i1 to i64
            %81 = arith.subi %79, %80 : i64
            %82 = arith.maxsi %arg30, %81 : i64
            scf.yield %82 : i64
          }
          %52 = arith.addi %51, %c255_i64 : i64
          %53 = arith.remsi %52, %c256_i64 : i64
          %54 = arith.cmpi ne, %53, %c0_i64 : i64
          %55 = arith.cmpi slt, %53, %c0_i64 : i64
          %56 = arith.andi %54, %55 : i1
          %57 = arith.divsi %52, %c256_i64 : i64
          %58 = arith.extui %56 : i1 to i64
          %59 = arith.subi %57, %58 : i64
          %60 = arith.addi %51, %c4095_i64 : i64
          %61 = arith.remsi %60, %c4096_i64 : i64
          %62 = arith.cmpi ne, %61, %c0_i64 : i64
          %63 = arith.cmpi slt, %61, %c0_i64 : i64
          %64 = arith.andi %62, %63 : i1
          %65 = arith.divsi %60, %c4096_i64 : i64
          %66 = arith.extui %64 : i1 to i64
          %67 = arith.subi %65, %66 : i64
          %68 = arith.cmpi sgt, %59, %c0_i64 : i64
          %69:6 = scf.if %68 -> (i64, i64, i64, i64, i64, i64) {
            %70 = arith.muli %48, %c4_i64 : i64
            %71 = cannir.extract_layout %arg0 : !cannir.tensor<<gm, bf16>, <<(-1, 128)>, <(128, 1)>>, <<(-1, 128)>, <(128, 1)>>, identity>
            %72 = cannir.extract_shape %71 : <<(-1, 128)>, <(128, 1)>> -> <(-1, 128)>
            %73 = cannir.extract_dim %72[0] : !cannir.shape<(-1, 128)>
            %74 = arith.subi %73, %70 : i64
            %75 = ascscalar.bitcast_memref %alloc_6 byte_offset = %c0_i64 : memref<16384xbf16, 2> -> memref<?xbf16, 2>
            %76 = arith.muli %48, %c512_i64 : i64
            %77 = arith.divsi %74, %c128_i64 : i64
            %78 = arith.minsi %77, %c0_i64 : i64
            %79 = arith.muli %78, %c128_i64 : i64
            %80 = arith.subi %74, %79 : i64
            %81 = arith.minsi %80, %c128_i64 : i64
            %82 = arith.cmpi sge, %77, %c0_i64 : i64
            %83 = arith.select %82, %81, %c0_i64 : i64
            asccube.get_buf mte2, %c0_i64, 0 {cannir.buffer.generated_sync}
            %84 = arith.cmpi ugt, %83, %c0_i64 : i64
            scf.if %84 {
              %89 = arith.index_cast %76 : i64 to index
              %90 = arith.trunci %83 : i64 to i32
              asccube.copy_gm2l1 %75[%c0], %alloc[%89], %90, %c128_i32, %c256_i64, %c1_i32, %c0_i64, %c0_i64, %c128_i32 l2_cache_ctl %c0_i32 {nd2nz = true} : memref<?xbf16, 2>, memref<1xbf16>
            }
            asccube.release_buf mte2, %c0_i64, 0 {cannir.buffer.generated_sync}
            %85:3 = scf.for %arg29 = %c0_i64 to %59 step %c1_i64 iter_args(%arg30 = %arg23, %arg31 = %arg24, %arg32 = %arg28) -> (i64, i64, i64)  : i64 {
              %89 = arith.muli %arg29, %c256_i64 : i64
              %90 = arith.remsi %89, %c256_i64 : i64
              %91 = arith.cmpi ne, %90, %c0_i64 : i64
              %92 = arith.cmpi slt, %90, %c0_i64 : i64
              %93 = arith.andi %91, %92 : i1
              %94 = arith.divsi %89, %c256_i64 : i64
              %95 = arith.extui %93 : i1 to i64
              %96 = arith.subi %94, %95 : i64
              %97 = arith.muli %96, %c256_i64 : i64
              %98 = arith.cmpi slt, %97, %51 : i64
              %99 = scf.if %98 -> (i64) {
                %150 = cannir.extract_layout %arg2 : !cannir.tensor<<gm, i32>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
                %151 = cannir.extract_stride %150 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
                %152 = cannir.extract_dim %151[0] : !cannir.stride<(-1, 1)>
                %153 = arith.index_cast %152 : i64 to index
                %154 = arith.index_cast %96 : i64 to index
                %155 = arith.muli %26, %153 : index
                %156 = arith.addi %155, %154 : index
                %157 = memref.load %alloc_1[%156] : memref<1xi32>
                %158 = arith.extsi %157 : i32 to i64
                scf.yield %158 : i64
              } else {
                scf.yield %c0_i64 : i64
              }
              %100 = arith.cmpi eq, %arg30, %c0_i64 : i64
              %101 = ascscalar.memref_address %alloc_14 : memref<32768xbf16, 2>
              %102 = ascscalar.memref_address %alloc_13 : memref<32768xbf16, 2>
              %103 = arith.select %100, %101, %102 : i64
              %104 = arith.select %100, %c1_i64, %c2_i64 : i64
              %105 = arith.addi %arg30, %c1_i64 : i64
              %106 = arith.remui %105, %c2_i64 : i64
              %107 = ascscalar.bind_local_ptr %103 : memref<1xbf16, 2>
              %108 = arith.muli %99, %c32768_i64 : i64
              asccube.get_buf mte2, %104, 0 {cannir.buffer.generated_sync}
              %109 = arith.index_cast %108 : i64 to index
              asccube.copy_gm2l1 %107[%c0], %alloc_0[%109], %c256_i32, %c128_i32, %c256_i64, %c1_i32, %c0_i64, %c0_i64, %c256_i32 l2_cache_ctl %c0_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
              asccube.release_buf mte2, %104, 0 {cannir.buffer.generated_sync}
              %110 = ascscalar.memref_address %alloc_12 : memref<32768xf32, 5>
              %111 = ascscalar.memref_address %alloc_11 : memref<32768xf32, 5>
              %112 = arith.select %100, %110, %111 : i64
              %113 = arith.select %100, %c7_i64, %c8_i64 : i64
              %114 = arith.cmpi eq, %arg31, %c0_i64 : i64
              %115 = ascscalar.memref_address %alloc_10 : memref<16384xbf16, 3>
              %116 = ascscalar.memref_address %alloc_9 : memref<16384xbf16, 3>
              %117 = arith.select %114, %115, %116 : i64
              %118 = arith.select %114, %c3_i64, %c4_i64 : i64
              %119 = arith.addi %arg31, %c1_i64 : i64
              %120 = arith.remui %119, %c2_i64 : i64
              %121 = ascscalar.memref_address %alloc_8 : memref<16384xbf16, 4>
              %122 = ascscalar.memref_address %alloc_7 : memref<16384xbf16, 4>
              %123 = arith.select %114, %121, %122 : i64
              %124 = arith.select %114, %c5_i64, %c6_i64 : i64
              %125 = ascscalar.bind_local_ptr %117 : memref<1xbf16, 3>
              asccube.get_buf mte1, %118, 0 {cannir.buffer.generated_sync}
              asccube.get_buf mte1, %c0_i64, 0 {cannir.buffer.generated_sync}
              asccube.copy_l12l0a %125[%c0], %75[%c0], %c0_i32, %c0_i32, %c8_i32, %c8_i32, %c8_i32, %c8_i32 : memref<1xbf16, 3>, memref<?xbf16, 2>
              asccube.release_buf mte1, %118, 0 {cannir.buffer.generated_sync}
              %126 = ascscalar.bind_local_ptr %123 : memref<1xbf16, 4>
              asccube.get_buf mte1, %124, 0 {cannir.buffer.generated_sync}
              asccube.get_buf mte1, %104, 0 {cannir.buffer.generated_sync}
              asccube.copy_l12l0b %126[%c0], %107[%c0], %c0_i32, %c0_i32, %c8_i32, %c8_i32, %c16_i32, %c8_i32 : memref<1xbf16, 4>, memref<1xbf16, 2>
              asccube.release_buf mte1, %124, 0 {cannir.buffer.generated_sync}
              %127 = ascscalar.bind_local_ptr %112 : memref<1xf32, 5>
              asccube.get_buf m, %113, 0 {cannir.buffer.generated_sync}
              asccube.get_buf m, %118, 0 {cannir.buffer.generated_sync}
              asccube.get_buf m, %124, 0 {cannir.buffer.generated_sync}
              asccube.mmad %127[%c0], %125, %126, %c128_i32, %c128_i32, %c128_i32, %c0_i32 {c_matrix_init_val = true} : memref<1xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
              asccube.release_buf m, %124, 0 {cannir.buffer.generated_sync}
              asccube.release_buf m, %118, 0 {cannir.buffer.generated_sync}
              %128 = arith.cmpi eq, %120, %c0_i64 : i64
              %129 = arith.select %128, %115, %116 : i64
              %130 = arith.select %128, %c3_i64, %c4_i64 : i64
              %131 = arith.addi %120, %c1_i64 overflow<nsw> : i64
              %132 = arith.remui %131, %c2_i64 : i64
              %133 = arith.select %128, %121, %122 : i64
              %134 = arith.select %128, %c5_i64, %c6_i64 : i64
              %135 = ascscalar.bind_local_ptr %129 : memref<1xbf16, 3>
              asccube.get_buf mte1, %130, 0 {cannir.buffer.generated_sync}
              asccube.copy_l12l0a %135[%c0], %75[%c0], %c0_i32, %c0_i32, %c8_i32, %c8_i32, %c8_i32, %c8_i32 : memref<1xbf16, 3>, memref<?xbf16, 2>
              asccube.release_buf mte1, %c0_i64, 0 {cannir.buffer.generated_sync}
              asccube.release_buf mte1, %130, 0 {cannir.buffer.generated_sync}
              %136 = ascscalar.bind_local_ptr %133 : memref<1xbf16, 4>
              asccube.get_buf mte1, %134, 0 {cannir.buffer.generated_sync}
              asccube.copy_l12l0b %136[%c0], %107[%c0], %c8_i32, %c0_i32, %c8_i32, %c8_i32, %c16_i32, %c8_i32 : memref<1xbf16, 4>, memref<1xbf16, 2>
              asccube.release_buf mte1, %104, 0 {cannir.buffer.generated_sync}
              asccube.release_buf mte1, %134, 0 {cannir.buffer.generated_sync}
              %137 = arith.addi %112, %c65536_i64 : i64
              %138 = ascscalar.bind_local_ptr %137 : memref<1xf32, 5>
              asccube.get_buf m, %130, 0 {cannir.buffer.generated_sync}
              asccube.get_buf m, %134, 0 {cannir.buffer.generated_sync}
              asccube.mmad %138[%c0], %135, %136, %c128_i32, %c128_i32, %c128_i32, %c0_i32 {c_matrix_init_val = true} : memref<1xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
              asccube.release_buf m, %134, 0 {cannir.buffer.generated_sync}
              asccube.release_buf m, %130, 0 {cannir.buffer.generated_sync}
              asccube.release_buf m, %113, 0 {cannir.buffer.generated_sync}
              %139 = arith.select %100, %c950272_i64, %c1474560_i64 : i64
              %140 = arith.cmpi ne, %arg30, %c0_i64 : i64
              %141 = arith.extui %140 : i1 to i64
              %142 = ascscalar.bitcast_memref %alloc_5 byte_offset = %c0_i64 : memref<262144xui8, 1> -> memref<?xf32, 1>
              %143 = arith.index_cast %139 : i64 to index
              %144 = arith.divui %143, %c32 : index
              %145 = arith.cmpi eq, %141, %c1_i64 : i64
              %146 = arith.extui %145 : i1 to i64
              asccube.sync_intra_wait fixpipe, %146 {cannir.sync.banks = 3 : i32}
              %147 = arith.addi %146, %c16_i64 overflow<nsw> : i64
              asccube.sync_intra_wait fixpipe, %147 {cannir.sync.banks = 3 : i32}
              asccube.get_buf fixpipe, %113, 0 {cannir.buffer.generated_sync}
              asccube.copy_l0c2ub %142[%144], %127[%c0], %c256_i32, %c128_i32, %c256_i32, %c128_i32, %c0_i32 sub_block_id %c0_i32 {dual_dst_ctl = 1 : i32, nz2nd_en = true, relu_en = false} : memref<?xf32, 1>, memref<1xf32, 5>
              asccube.release_buf fixpipe, %113, 0 {cannir.buffer.generated_sync}
              asccube.sync_intra_arrive fixpipe, %146 {cannir.sync.banks = 3 : i32}
              asccube.sync_intra_arrive fixpipe, %147 {cannir.sync.banks = 3 : i32}
              %148 = arith.addi %arg32, %c1_i64 : i64
              %149 = arith.remui %148, %c2_i64 : i64
              scf.yield %106, %132, %149 : i64, i64, i64
            }
            %86 = arith.muli %67, %c16_i64 : i64
            %87 = scf.for %arg29 = %59 to %86 step %c1_i64 iter_args(%arg30 = %85#2) -> (i64)  : i64 {
              %89 = arith.addi %arg30, %c1_i64 : i64
              %90 = arith.remui %89, %c2_i64 : i64
              scf.yield %90 : i64
            }
            %88:3 = scf.for %arg29 = %c0_i64 to %c16_i64 step %c2_i64 iter_args(%arg30 = %arg25, %arg31 = %arg26, %arg32 = %arg27) -> (i64, i64, i64)  : i64 {
              %89 = arith.muli %0, %c16_i64 : i64
              %90 = arith.addi %89, %arg29 : i64
              %91 = arith.cmpi sgt, %67, %c0_i64 : i64
              %92 = scf.if %91 -> (i64) {
                %95 = arith.addi %arg31, %c1_i64 : i64
                %96 = arith.remui %95, %c2_i64 : i64
                scf.yield %96 : i64
              } else {
                scf.yield %arg31 : i64
              }
              %93:2 = scf.for %arg33 = %c0_i64 to %67 step %c1_i64 iter_args(%arg34 = %92, %arg35 = %arg32) -> (i64, i64)  : i64 {
                %95 = arith.addi %arg33, %c1_i64 : i64
                %96 = arith.addi %arg35, %c1_i64 : i64
                %97 = arith.remui %96, %c2_i64 : i64
                %98 = arith.cmpi slt, %95, %67 : i64
                %99 = scf.if %98 -> (i64) {
                  %100 = arith.addi %arg34, %c1_i64 : i64
                  %101 = arith.remui %100, %c2_i64 : i64
                  scf.yield %101 : i64
                } else {
                  scf.yield %arg34 : i64
                }
                scf.yield %99, %97 : i64, i64
              }
              %94 = scf.if %91 -> (i64) {
                %95 = scf.for %arg33 = %c0_i64 to %c2_i64 step %c1_i64 iter_args(%arg34 = %arg30) -> (i64)  : i64 {
                  %96 = arith.addi %90, %arg33 : i64
                  %97 = arith.cmpi slt, %96, %50 : i64
                  %98 = scf.if %97 -> (i64) {
                    %99 = arith.addi %arg34, %c1_i64 : i64
                    %100 = arith.remui %99, %c2_i64 : i64
                    scf.yield %100 : i64
                  } else {
                    scf.yield %arg34 : i64
                  }
                  scf.yield %98 : i64
                }
                scf.yield %95 : i64
              } else {
                %95 = scf.for %arg33 = %c0_i64 to %c2_i64 step %c1_i64 iter_args(%arg34 = %arg30) -> (i64)  : i64 {
                  %96 = arith.addi %90, %arg33 : i64
                  %97 = arith.cmpi slt, %96, %50 : i64
                  %98 = scf.if %97 -> (i64) {
                    %99 = arith.addi %arg34, %c1_i64 : i64
                    %100 = arith.remui %99, %c2_i64 : i64
                    scf.yield %100 : i64
                  } else {
                    scf.yield %arg34 : i64
                  }
                  scf.yield %98 : i64
                }
                scf.yield %95 : i64
              }
              scf.yield %94, %93#0, %93#1 : i64, i64, i64
            }
            scf.yield %85#0, %85#1, %88#0, %88#1, %88#2, %87 : i64, i64, i64, i64, i64, i64
          } else {
            %70 = scf.for %arg29 = %c0_i64 to %c16_i64 step %c2_i64 iter_args(%arg30 = %arg25) -> (i64)  : i64 {
              %71 = arith.muli %0, %c16_i64 : i64
              %72 = arith.addi %71, %arg29 : i64
              %73 = scf.for %arg31 = %c0_i64 to %c2_i64 step %c1_i64 iter_args(%arg32 = %arg30) -> (i64)  : i64 {
                %74 = arith.addi %72, %arg31 : i64
                %75 = arith.cmpi slt, %74, %50 : i64
                %76 = scf.if %75 -> (i64) {
                  %77 = arith.addi %arg32, %c1_i64 : i64
                  %78 = arith.remui %77, %c2_i64 : i64
                  scf.yield %78 : i64
                } else {
                  scf.yield %arg32 : i64
                }
                scf.yield %76 : i64
              }
              scf.yield %73 : i64
            }
            scf.yield %arg23, %arg24, %70, %arg26, %arg27, %arg28 : i64, i64, i64, i64, i64, i64
          }
          scf.yield %69#0, %69#1, %69#2, %69#3, %69#4, %69#5 : i64, i64, i64, i64, i64, i64
        }
        scf.yield %46#0, %46#1, %46#2, %46#3, %46#4, %46#5 : i64, i64, i64, i64, i64, i64
      }
      scf.yield %25#0, %25#1, %25#2, %25#3, %25#4, %25#5 : i64, i64, i64, i64, i64, i64
    }
    asccube.sync_intra_wait fixpipe, %c0_i64
    asccube.sync_intra_wait fixpipe, %c16_i64
    asccube.sync_intra_wait fixpipe, %c1_i64
    asccube.sync_intra_wait fixpipe, %c17_i64
    return
  }
}
