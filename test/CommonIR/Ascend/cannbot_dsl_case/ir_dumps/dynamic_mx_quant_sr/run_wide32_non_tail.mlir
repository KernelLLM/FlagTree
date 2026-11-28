module {
  func.func @run_wide32_non_tail_4d2938f675cec53aa2cdedd1be989727(%arg0: !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg1: !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg2: !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: !cannir.tensor<<gm, i64>, <<(2)>, <(1)>>, <<(2)>, <(1)>>, identity>) attributes {cannbotdsl.contract_digest = "4d2938f675cec53aa2cdedd1be9897272897bbec34ad9afa728cc7c9f68efea2", cannbotdsl.public_slot_count = 7 : i64, cannbotdsl.slot_dtype_bits = [16, 8, 8, 64, 64, 64, 64], cannbotdsl.slot_dtype_codes = [4, 1, 1, 0, 0, 0, 0], cannbotdsl.slot_dtype_lanes = [1, 1, 1, 1, 1, 1, 1], cannbotdsl.slot_flags = [1, 1, 1, 1, 1, 1, 1], cannbotdsl.slot_kinds = [1, 1, 1, 2, 2, 2, 1], cannbotdsl.slot_type_digests = ["5d7cdc96dc21b3b739e32aeccf58c1df113bec54701c75107ed650f2544a07da", "171ddd09eb070fb4a6413ae87b2ceecddc233a9a173a0404e96285a4bb8b8659", "891bcd21e721f8d1a666bcca4e8705ec63796c1add8ae000536d0d98f0274d3a", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "4126671a83041647cd98afed881383228865da287ea44573939f32b1d956765b"], cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.host} {
    cannir.kernel_launch @kernel__DynamicMxQuantSrKernel_run_wide32_non_tail_0(%arg5) [%arg0, %arg1, %arg2, %arg3, %arg4, %arg6] {cannbotdsl.profile = {indices = array<i64: 0, 1, 2, 5>, name = "_DynamicMxQuantSrKernel_run_wide32_non_tail", op_type = "_DynamicMxQuantSrKernel_run_wide32_non_tail", roles = array<i64: 0, 1, 1, 0>}} : !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, i64, i64, !cannir.tensor<<gm, i64>, <<(2)>, <(1)>>, <<(2)>, <(1)>>, identity>
    return
  }
  func.func @kernel__DynamicMxQuantSrKernel_run_wide32_non_tail_0(%arg0: !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg1: !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg2: !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg3: i64, %arg4: i64, %arg5: !cannir.tensor<<gm, i64>, <<(2)>, <(1)>>, <<(2)>, <(1)>>, identity>) attributes {cannir.buffer.auto_sync, cannir.buffer.sync_analyzed, cannir.buffer.sync_verified, cannir.buffer.tensor_state, cannir.core = "aiv", cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.device, cannir.explicit_ring_buffer_cursors, cannir.kernel_group = "kernel__DynamicMxQuantSrKernel_run_wide32_non_tail_0", cannir.resource.ids_prepared} {
    %c1024_i32 = arith.constant 1024 : i32
    %c16384_i32 = arith.constant 16384 : i32
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c64_i64 = arith.constant 64 : i64
    %c256_i64 = arith.constant 256 : i64
    %c512_i64 = arith.constant 512 : i64
    %c32_i64 = arith.constant 32 : i64
    %false = arith.constant false
    %c128_i64 = arith.constant 128 : i64
    %c0_i32 = arith.constant 0 : i32
    %c32768_i64 = arith.constant 32768 : i64
    %c16_i64 = arith.constant 16 : i64
    %c4294967296_i64 = arith.constant 4294967296 : i64
    %c32_i32 = arith.constant 32 : i32
    %c16_i32 = arith.constant 16 : i32
    %c64_i32 = arith.constant 64 : i32
    %c16384_i64 = arith.constant 16384 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %c8_i64 = arith.constant 8 : i64
    %c5_i64 = arith.constant 5 : i64
    %c4_i64 = arith.constant 4 : i64
    %c3_i64 = arith.constant 3 : i64
    %c2_i64 = arith.constant 2 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %alloc = memref.alloc() {cannir.arg_index = 0 : i64} : memref<1xbf16>
    %alloc_0 = memref.alloc() {cannir.arg_index = 1 : i64} : memref<1xui8>
    %alloc_1 = memref.alloc() {cannir.arg_index = 2 : i64} : memref<1xui8>
    %alloc_2 = memref.alloc() {cannir.arg_index = 5 : i64} : memref<2xi64>
    %alloc_3 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<16384xbf16, 1>
    %alloc_4 = memref.alloc() {cannir.byte_offset = 32768 : i64} : memref<16384xbf16, 1>
    %alloc_5 = memref.alloc() {cannir.byte_offset = 65536 : i64} : memref<16384xui8, 1>
    %alloc_6 = memref.alloc() {cannir.byte_offset = 81920 : i64} : memref<16384xui8, 1>
    %alloc_7 = memref.alloc() {cannir.byte_offset = 98304 : i64} : memref<1024xui8, 1>
    %alloc_8 = memref.alloc() {cannir.byte_offset = 99328 : i64} : memref<1024xui8, 1>
    %alloc_9 = memref.alloc() {cannir.byte_offset = 100352 : i64} : memref<512xui32, 1>
    %alloc_10 = memref.alloc() {cannir.byte_offset = 102400 : i64} : memref<4096xi32, 1>
    %0 = ascscalar.scalar_ptr %alloc_2[%c0] : memref<2xi64>, index -> <gm, i64>
    %1 = ascscalar.load_dev %0 : !ascscalar.scalar_ptr<gm, i64> -> i64
    %2 = ascscalar.scalar_ptr %alloc_2[%c1] : memref<2xi64>, index -> <gm, i64>
    %3 = ascscalar.load_dev %2 : !ascscalar.scalar_ptr<gm, i64> -> i64
    %4 = cannir.extract_layout %arg0 : !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
    %5 = cannir.extract_shape %4 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, -1)>
    %6 = cannir.extract_dim %5[0] : !cannir.shape<(-1, -1)>
    %7 = arith.remsi %arg3, %c64_i64 : i64
    %8 = arith.cmpi ne, %7, %c0_i64 : i64
    %9 = arith.cmpi slt, %7, %c0_i64 : i64
    %10 = arith.andi %8, %9 : i1
    %11 = arith.divsi %arg3, %c64_i64 : i64
    %12 = arith.extui %10 : i1 to i64
    %13 = arith.subi %11, %12 : i64
    %14 = arith.muli %6, %13 : i64
    %15 = arith.cmpi sge, %14, %c256_i64 : i64
    %16 = scf.if %15 -> (i1) {
      %29 = arith.remsi %arg3, %c512_i64 : i64
      %30 = arith.cmpi ne, %29, %c0_i64 : i64
      %31 = arith.cmpi slt, %29, %c0_i64 : i64
      %32 = arith.andi %30, %31 : i1
      %33 = arith.select %32, %c512_i64, %c0_i64 : i64
      %34 = arith.addi %29, %33 overflow<nsw> : i64
      %35 = arith.cmpi eq, %34, %c0_i64 : i64
      %36 = arith.cmpi eq, %arg4, %c32_i64 : i64
      %37 = arith.select %35, %36, %false : i1
      scf.yield %37 : i1
    } else {
      scf.yield %false : i1
    }
    %17 = scf.if %16 -> (i64) {
      scf.yield %c512_i64 : i64
    } else {
      %29 = scf.if %15 -> (i1) {
        %31 = arith.remsi %arg3, %c256_i64 : i64
        %32 = arith.cmpi ne, %31, %c0_i64 : i64
        %33 = arith.cmpi slt, %31, %c0_i64 : i64
        %34 = arith.andi %32, %33 : i1
        %35 = arith.select %34, %c256_i64, %c0_i64 : i64
        %36 = arith.addi %31, %35 overflow<nsw> : i64
        %37 = arith.cmpi eq, %36, %c0_i64 : i64
        %38 = arith.cmpi eq, %arg4, %c32_i64 : i64
        %39 = arith.select %37, %38, %false : i1
        scf.yield %39 : i1
      } else {
        scf.yield %false : i1
      }
      %30 = scf.if %29 -> (i64) {
        scf.yield %c256_i64 : i64
      } else {
        %31 = arith.remsi %arg3, %c128_i64 : i64
        %32 = arith.cmpi ne, %31, %c0_i64 : i64
        %33 = arith.cmpi slt, %31, %c0_i64 : i64
        %34 = arith.andi %32, %33 : i1
        %35 = arith.select %34, %c128_i64, %c0_i64 : i64
        %36 = arith.addi %31, %35 overflow<nsw> : i64
        %37 = arith.cmpi eq, %36, %c0_i64 : i64
        %38 = arith.select %15, %37, %false : i1
        %39 = arith.select %38, %c128_i64, %c64_i64 : i64
        scf.yield %39 : i64
      }
      scf.yield %30 : i64
    }
    %18 = arith.remsi %arg3, %17 : i64
    %19 = arith.cmpi ne, %18, %c0_i64 : i64
    %20 = arith.cmpi slt, %18, %c0_i64 : i64
    %21 = arith.andi %19, %20 : i1
    %22 = arith.divsi %arg3, %17 : i64
    %23 = arith.extui %21 : i1 to i64
    %24 = arith.subi %22, %23 : i64
    %25 = arith.muli %6, %24 : i64
    %26 = ascscalar.get_block_idx
    %27 = ascscalar.get_block_num
    %28 = scf.for %arg6 = %26 to %25 step %27 iter_args(%arg7 = %c0_i64) -> (i64)  : i64 {
      %29 = arith.remsi %arg6, %24 : i64
      %30 = arith.cmpi ne, %29, %c0_i64 : i64
      %31 = arith.cmpi slt, %29, %c0_i64 : i64
      %32 = arith.cmpi slt, %24, %c0_i64 : i64
      %33 = arith.xori %31, %32 : i1
      %34 = arith.andi %30, %33 : i1
      %35 = arith.divsi %arg6, %24 : i64
      %36 = arith.extui %34 : i1 to i64
      %37 = arith.subi %35, %36 : i64
      %38 = arith.select %34, %24, %c0_i64 : i64
      %39 = arith.addi %29, %38 : i64
      %40 = arith.muli %17, %arg4 : i64
      %41 = arith.cmpi eq, %arg7, %c0_i64 : i64
      %42 = ascscalar.memref_address %alloc_3 : memref<16384xbf16, 1>
      %43 = ascscalar.memref_address %alloc_4 : memref<16384xbf16, 1>
      %44 = arith.select %41, %42, %43 : i64
      %45 = arith.cmpi ne, %arg7, %c0_i64 : i64
      %46 = arith.extui %45 : i1 to i64
      %47 = arith.addi %arg7, %c1_i64 overflow<nsw> : i64
      %48 = arith.remui %47, %c2_i64 : i64
      %49 = arith.minsi %37, %6 : i64
      %50 = arith.subi %6, %49 : i64
      %51 = arith.minsi %50, %c1_i64 : i64
      %52 = arith.cmpi sle, %37, %6 : i64
      %53 = arith.select %52, %51, %c0_i64 : i64
      %54 = cannir.extract_physical_layout %arg0 : !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
      %55 = cannir.extract_stride %54 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
      %56 = cannir.extract_dim %55[0] : !cannir.stride<(-1, 1)>
      %57 = arith.muli %56, %c2_i64 : i64
      %58 = cannir.extract_dim %5[1] : !cannir.shape<(-1, -1)>
      %59 = arith.maxsi %40, %c1_i64 : i64
      %60 = arith.divsi %58, %59 : i64
      %61 = arith.minsi %39, %60 : i64
      %62 = arith.muli %61, %40 : i64
      %63 = arith.subi %58, %62 : i64
      %64 = arith.minsi %40, %63 : i64
      %65 = arith.cmpi sle, %39, %60 : i64
      %66 = arith.select %65, %64, %c0_i64 : i64
      %67 = arith.muli %37, %56 : i64
      %68 = arith.muli %67, %c2_i64 : i64
      %69 = arith.muli %39, %40 : i64
      %70 = arith.muli %69, %c2_i64 : i64
      %71 = arith.addi %68, %70 : i64
      %72 = arith.muli %66, %c2_i64 : i64
      %73 = arith.cmpi eq, %72, %c0_i64 : i64
      %74 = arith.select %73, %c0_i64, %53 : i64
      %75 = arith.select %73, %c2_i64, %72 : i64
      %76 = ascscalar.bind_local_ptr %44 : memref<1xbf16, 1>
      ascvec.get_buf mte2, %46, 0 {cannir.buffer.generated_sync}
      "ascvec.copy_gm2ub_plan"(%76, %alloc, %c0_i32, %c0_i32, %c0_i64, %71, %74, %75, %57, %c32768_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
      ascvec.release_buf mte2, %46, 0 {cannir.buffer.generated_sync}
      %77 = ascscalar.bind_local_ptr %44 : memref<16384xbf16, 1>
      %78 = ascscalar.memref_address %alloc_5 : memref<16384xui8, 1>
      %79 = ascscalar.memref_address %alloc_6 : memref<16384xui8, 1>
      %80 = arith.select %41, %78, %79 : i64
      %81 = ascscalar.bind_local_ptr %80 : memref<16384xui8, 1>
      %82 = arith.select %41, %c2_i64, %c3_i64 : i64
      %83 = ascscalar.memref_address %alloc_7 : memref<1024xui8, 1>
      %84 = ascscalar.memref_address %alloc_8 : memref<1024xui8, 1>
      %85 = arith.select %41, %83, %84 : i64
      %86 = ascscalar.bind_local_ptr %85 : memref<1024xui8, 1>
      %87 = arith.select %41, %c4_i64, %c5_i64 : i64
      %88 = arith.muli %37, %arg4 : i64
      %89 = arith.muli %39, %17 : i64
      %90 = arith.remsi %89, %c16_i64 : i64
      %91 = arith.cmpi ne, %90, %c0_i64 : i64
      %92 = arith.cmpi slt, %90, %c0_i64 : i64
      %93 = arith.andi %91, %92 : i1
      %94 = arith.divsi %89, %c16_i64 : i64
      %95 = arith.extui %93 : i1 to i64
      %96 = arith.subi %94, %95 : i64
      %97 = arith.remsi %arg3, %c16_i64 : i64
      %98 = arith.cmpi ne, %97, %c0_i64 : i64
      %99 = arith.cmpi slt, %97, %c0_i64 : i64
      %100 = arith.andi %98, %99 : i1
      %101 = arith.divsi %arg3, %c16_i64 : i64
      %102 = arith.extui %100 : i1 to i64
      %103 = arith.subi %101, %102 : i64
      %104 = arith.trunci %1 : i64 to i32
      %105 = arith.extui %104 : i32 to i64
      %106 = arith.subi %1, %105 : i64
      %107 = arith.remsi %106, %c4294967296_i64 : i64
      %108 = arith.cmpi ne, %107, %c0_i64 : i64
      %109 = arith.cmpi slt, %107, %c0_i64 : i64
      %110 = arith.andi %108, %109 : i1
      %111 = arith.divsi %106, %c4294967296_i64 : i64
      %112 = arith.extui %110 : i1 to i64
      %113 = arith.subi %111, %112 : i64
      %114 = arith.trunci %113 : i64 to i32
      %115 = arith.trunci %17 : i64 to i32
      %116 = arith.divsi %115, %c16_i32 : i32
      ascvec.vf(%96, %88, %103, %3, %104, %114, %alloc_10, %116 : i64, i64, i64, i64, i32, i32, memref<4096xi32, 1>, i32) {
      ^bb0(%arg8: i64, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: i32, %arg13: i32, %arg14: memref<4096xi32, 1>, %arg15: i32):
        %c128_i32 = arith.constant 128 : i32
        %c1_i32 = arith.constant 1 : i32
        %c96_i32 = arith.constant 96 : i32
        %c64_i32_11 = arith.constant 64 : i32
        %c-1767562579_i32 = arith.constant -1767562579 : i32
        %c-1879881855_i32 = arith.constant -1879881855 : i32
        %c-616729560_i32 = arith.constant -616729560 : i32
        %c-239350328_i32 = arith.constant -239350328 : i32
        %c534103459_i32 = arith.constant 534103459 : i32
        %c1401181199_i32 = arith.constant 1401181199 : i32
        %c1684936478_i32 = arith.constant 1684936478 : i32
        %c-1253254570_i32 = arith.constant -1253254570 : i32
        %c-1459197799_i32 = arith.constant -1459197799 : i32
        %c387276957_i32 = arith.constant 387276957 : i32
        %c-308364780_i32 = arith.constant -308364780 : i32
        %c2027808484_i32 = arith.constant 2027808484 : i32
        %c842468239_i32 = arith.constant 842468239 : i32
        %c-626627285_i32 = arith.constant -626627285 : i32
        %c1993301258_i32 = arith.constant 1993301258 : i32
        %c1013904242_i32 = arith.constant 1013904242 : i32
        %c-1150833019_i32 = arith.constant -1150833019 : i32
        %c-1640531527_i32 = arith.constant -1640531527 : i32
        %c-845247145_i32 = arith.constant -845247145 : i32
        %c-766435501_i32 = arith.constant -766435501 : i32
        %c16383_i32 = arith.constant 16383 : i32
        %c1073741824_i32 = arith.constant 1073741824 : i32
        %c4_i32 = arith.constant 4 : i32
        %c0_i64_12 = arith.constant 0 : i64
        %c4294967296_i64_13 = arith.constant 4294967296 : i64
        %c0_i32_14 = arith.constant 0 : i32
        %c32_i32_15 = arith.constant 32 : i32
        %mask, %remaining_out = ascvec.update_mask %c32_i32_15 : !ascvec.mask
        %201 = ascvec.arange %c0_i32_14 : i32 -> <ui32>
        scf.for %arg16 = %c0_i32_14 to %arg15 step %c1_i32  : i32 {
          %202 = arith.extsi %arg16 : i32 to i64
          %203 = arith.addi %arg8, %202 : i64
          %204 = arith.trunci %arg9 : i64 to i32
          %205 = arith.extui %204 : i32 to i64
          %206 = arith.subi %arg9, %205 : i64
          %207 = arith.remsi %206, %c4294967296_i64_13 : i64
          %208 = arith.cmpi ne, %207, %c0_i64_12 : i64
          %209 = arith.cmpi slt, %207, %c0_i64_12 : i64
          %210 = arith.andi %208, %209 : i1
          %211 = arith.divsi %206, %c4294967296_i64_13 : i64
          %212 = arith.extui %210 : i1 to i64
          %213 = arith.subi %211, %212 : i64
          %214 = arith.trunci %213 : i64 to i32
          %215 = arith.trunci %203 : i64 to i32
          %216 = arith.extui %215 : i32 to i64
          %217 = arith.subi %203, %216 : i64
          %218 = arith.remsi %217, %c4294967296_i64_13 : i64
          %219 = arith.cmpi ne, %218, %c0_i64_12 : i64
          %220 = arith.cmpi slt, %218, %c0_i64_12 : i64
          %221 = arith.andi %219, %220 : i1
          %222 = arith.divsi %217, %c4294967296_i64_13 : i64
          %223 = arith.extui %221 : i1 to i64
          %224 = arith.subi %222, %223 : i64
          %225 = arith.trunci %224 : i64 to i32
          %226 = ascvec.duplicate_scalar %204, %mask : i32 -> <ui32>
          %227 = ascvec.duplicate_scalar %214, %mask : i32 -> <ui32>
          %228 = ascvec.duplicate_scalar %c0_i32_14, %mask : i32 -> <ui32>
          %carry, %dst = ascvec.addc %226, %201, %mask : <ui32>, <ui32> -> !ascvec.mask, <ui32>
          %carry_16, %dst_17 = ascvec.addc %227, %228, %carry, %mask : <ui32>, <ui32> -> !ascvec.mask, <ui32>
          %229 = ascvec.duplicate_scalar %215, %mask : i32 -> <ui32>
          %230 = ascvec.duplicate_scalar %225, %mask : i32 -> <ui32>
          %231 = arith.trunci %arg10 : i64 to i32
          %232 = arith.extui %231 : i32 to i64
          %233 = arith.subi %arg10, %232 : i64
          %234 = arith.remsi %233, %c4294967296_i64_13 : i64
          %235 = arith.cmpi ne, %234, %c0_i64_12 : i64
          %236 = arith.cmpi slt, %234, %c0_i64_12 : i64
          %237 = arith.andi %235, %236 : i1
          %238 = arith.divsi %233, %c4294967296_i64_13 : i64
          %239 = arith.extui %237 : i1 to i64
          %240 = arith.subi %238, %239 : i64
          %241 = arith.trunci %240 : i64 to i32
          %242 = ascvec.duplicate_scalar %231, %mask : i32 -> <ui32>
          %dst0, %dst1 = ascvec.mull %dst, %242, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %243 = ascvec.mul_scalar %dst, %241, %mask : <ui32>, i32
          %244 = ascvec.add %dst1, %243, %mask : <ui32>
          %245 = ascvec.mul_scalar %dst_17, %231, %mask : <ui32>, i32
          %246 = ascvec.add %244, %245, %mask : <ui32>
          %carry_18, %dst_19 = ascvec.addc %dst0, %229, %mask : <ui32>, <ui32> -> !ascvec.mask, <ui32>
          %carry_20, %dst_21 = ascvec.addc %246, %230, %carry_18, %mask : <ui32>, <ui32> -> !ascvec.mask, <ui32>
          %247 = arith.trunci %arg11 : i64 to i32
          %248 = arith.extui %247 : i32 to i64
          %249 = arith.subi %arg11, %248 : i64
          %250 = arith.remsi %249, %c4294967296_i64_13 : i64
          %251 = arith.cmpi ne, %250, %c0_i64_12 : i64
          %252 = arith.cmpi slt, %250, %c0_i64_12 : i64
          %253 = arith.andi %251, %252 : i1
          %254 = arith.divsi %249, %c4294967296_i64_13 : i64
          %255 = arith.extui %253 : i1 to i64
          %256 = arith.subi %254, %255 : i64
          %257 = arith.trunci %256 : i64 to i32
          %258 = arith.divui %247, %c4_i32 : i32
          %259 = arith.remui %257, %c4_i32 : i32
          %260 = arith.muli %259, %c1073741824_i32 : i32
          %261 = arith.addi %258, %260 : i32
          %262 = arith.divui %257, %c4_i32 : i32
          %263 = ascvec.shr %dst_19, 14, %mask : <ui32> -> <ui32>
          %264 = ascvec.shl %dst_21, 18, %mask : <ui32> -> <ui32>
          %265 = ascvec.or %263, %264, %mask : <ui32>, <ui32> -> <ui32>
          %266 = ascvec.shr %dst_21, 14, %mask : <ui32> -> <ui32>
          %267 = ascvec.duplicate_scalar %261, %mask : i32 -> <ui32>
          %268 = ascvec.duplicate_scalar %262, %mask : i32 -> <ui32>
          %carry_22, %dst_23 = ascvec.addc %267, %265, %mask : <ui32>, <ui32> -> !ascvec.mask, <ui32>
          %carry_24, %dst_25 = ascvec.addc %268, %266, %carry_22, %mask : <ui32>, <ui32> -> !ascvec.mask, <ui32>
          %269 = ascvec.duplicate_scalar %c16383_i32, %mask : i32 -> <ui32>
          %270 = ascvec.and %dst_19, %269, %mask : <ui32>, <ui32> -> <ui32>
          %271 = ascvec.duplicate_scalar %c-766435501_i32, %mask : i32 -> <ui32>
          %272 = ascvec.duplicate_scalar %c-845247145_i32, %mask : i32 -> <ui32>
          %dst0_26, %dst1_27 = ascvec.mull %228, %271, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %dst0_28, %dst1_29 = ascvec.mull %dst_23, %272, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %273 = ascvec.xor %dst1_29, %270, %mask : <ui32>, <ui32> -> <ui32>
          %274 = ascvec.xor %dst1_27, %dst_25, %mask : <ui32>, <ui32> -> <ui32>
          %275 = ascvec.duplicate_scalar %arg12, %mask : i32 -> <ui32>
          %276 = ascvec.duplicate_scalar %arg13, %mask : i32 -> <ui32>
          %277 = ascvec.xor %273, %275, %mask : <ui32>, <ui32> -> <ui32>
          %278 = ascvec.xor %274, %276, %mask : <ui32>, <ui32> -> <ui32>
          %dst0_30, %dst1_31 = ascvec.mull %277, %271, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %dst0_32, %dst1_33 = ascvec.mull %278, %272, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %279 = ascvec.xor %dst1_33, %dst0_28, %mask : <ui32>, <ui32> -> <ui32>
          %280 = ascvec.xor %dst1_31, %dst0_26, %mask : <ui32>, <ui32> -> <ui32>
          %281 = arith.addi %arg12, %c-1640531527_i32 : i32
          %282 = ascvec.duplicate_scalar %281, %mask : i32 -> <ui32>
          %283 = arith.addi %arg13, %c-1150833019_i32 : i32
          %284 = ascvec.duplicate_scalar %283, %mask : i32 -> <ui32>
          %285 = ascvec.xor %279, %282, %mask : <ui32>, <ui32> -> <ui32>
          %286 = ascvec.xor %280, %284, %mask : <ui32>, <ui32> -> <ui32>
          %dst0_34, %dst1_35 = ascvec.mull %285, %271, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %dst0_36, %dst1_37 = ascvec.mull %286, %272, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %287 = ascvec.xor %dst1_37, %dst0_32, %mask : <ui32>, <ui32> -> <ui32>
          %288 = ascvec.xor %dst1_35, %dst0_30, %mask : <ui32>, <ui32> -> <ui32>
          %289 = arith.addi %arg12, %c1013904242_i32 : i32
          %290 = ascvec.duplicate_scalar %289, %mask : i32 -> <ui32>
          %291 = arith.addi %arg13, %c1993301258_i32 : i32
          %292 = ascvec.duplicate_scalar %291, %mask : i32 -> <ui32>
          %293 = ascvec.xor %287, %290, %mask : <ui32>, <ui32> -> <ui32>
          %294 = ascvec.xor %288, %292, %mask : <ui32>, <ui32> -> <ui32>
          %dst0_38, %dst1_39 = ascvec.mull %293, %271, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %dst0_40, %dst1_41 = ascvec.mull %294, %272, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %295 = ascvec.xor %dst1_41, %dst0_36, %mask : <ui32>, <ui32> -> <ui32>
          %296 = ascvec.xor %dst1_39, %dst0_34, %mask : <ui32>, <ui32> -> <ui32>
          %297 = arith.addi %arg12, %c-626627285_i32 : i32
          %298 = ascvec.duplicate_scalar %297, %mask : i32 -> <ui32>
          %299 = arith.addi %arg13, %c842468239_i32 : i32
          %300 = ascvec.duplicate_scalar %299, %mask : i32 -> <ui32>
          %301 = ascvec.xor %295, %298, %mask : <ui32>, <ui32> -> <ui32>
          %302 = ascvec.xor %296, %300, %mask : <ui32>, <ui32> -> <ui32>
          %dst0_42, %dst1_43 = ascvec.mull %301, %271, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %dst0_44, %dst1_45 = ascvec.mull %302, %272, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %303 = ascvec.xor %dst1_45, %dst0_40, %mask : <ui32>, <ui32> -> <ui32>
          %304 = ascvec.xor %dst1_43, %dst0_38, %mask : <ui32>, <ui32> -> <ui32>
          %305 = arith.addi %arg12, %c2027808484_i32 : i32
          %306 = ascvec.duplicate_scalar %305, %mask : i32 -> <ui32>
          %307 = arith.addi %arg13, %c-308364780_i32 : i32
          %308 = ascvec.duplicate_scalar %307, %mask : i32 -> <ui32>
          %309 = ascvec.xor %303, %306, %mask : <ui32>, <ui32> -> <ui32>
          %310 = ascvec.xor %304, %308, %mask : <ui32>, <ui32> -> <ui32>
          %dst0_46, %dst1_47 = ascvec.mull %309, %271, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %dst0_48, %dst1_49 = ascvec.mull %310, %272, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %311 = ascvec.xor %dst1_49, %dst0_44, %mask : <ui32>, <ui32> -> <ui32>
          %312 = ascvec.xor %dst1_47, %dst0_42, %mask : <ui32>, <ui32> -> <ui32>
          %313 = arith.addi %arg12, %c387276957_i32 : i32
          %314 = ascvec.duplicate_scalar %313, %mask : i32 -> <ui32>
          %315 = arith.addi %arg13, %c-1459197799_i32 : i32
          %316 = ascvec.duplicate_scalar %315, %mask : i32 -> <ui32>
          %317 = ascvec.xor %311, %314, %mask : <ui32>, <ui32> -> <ui32>
          %318 = ascvec.xor %312, %316, %mask : <ui32>, <ui32> -> <ui32>
          %dst0_50, %dst1_51 = ascvec.mull %317, %271, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %dst0_52, %dst1_53 = ascvec.mull %318, %272, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %319 = ascvec.xor %dst1_53, %dst0_48, %mask : <ui32>, <ui32> -> <ui32>
          %320 = ascvec.xor %dst1_51, %dst0_46, %mask : <ui32>, <ui32> -> <ui32>
          %321 = arith.addi %arg12, %c-1253254570_i32 : i32
          %322 = ascvec.duplicate_scalar %321, %mask : i32 -> <ui32>
          %323 = arith.addi %arg13, %c1684936478_i32 : i32
          %324 = ascvec.duplicate_scalar %323, %mask : i32 -> <ui32>
          %325 = ascvec.xor %319, %322, %mask : <ui32>, <ui32> -> <ui32>
          %326 = ascvec.xor %320, %324, %mask : <ui32>, <ui32> -> <ui32>
          %dst0_54, %dst1_55 = ascvec.mull %325, %271, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %dst0_56, %dst1_57 = ascvec.mull %326, %272, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %327 = ascvec.xor %dst1_57, %dst0_52, %mask : <ui32>, <ui32> -> <ui32>
          %328 = ascvec.xor %dst1_55, %dst0_50, %mask : <ui32>, <ui32> -> <ui32>
          %329 = arith.addi %arg12, %c1401181199_i32 : i32
          %330 = ascvec.duplicate_scalar %329, %mask : i32 -> <ui32>
          %331 = arith.addi %arg13, %c534103459_i32 : i32
          %332 = ascvec.duplicate_scalar %331, %mask : i32 -> <ui32>
          %333 = ascvec.xor %327, %330, %mask : <ui32>, <ui32> -> <ui32>
          %334 = ascvec.xor %328, %332, %mask : <ui32>, <ui32> -> <ui32>
          %dst0_58, %dst1_59 = ascvec.mull %333, %271, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %dst0_60, %dst1_61 = ascvec.mull %334, %272, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %335 = ascvec.xor %dst1_61, %dst0_56, %mask : <ui32>, <ui32> -> <ui32>
          %336 = ascvec.xor %dst1_59, %dst0_54, %mask : <ui32>, <ui32> -> <ui32>
          %337 = arith.addi %arg12, %c-239350328_i32 : i32
          %338 = ascvec.duplicate_scalar %337, %mask : i32 -> <ui32>
          %339 = arith.addi %arg13, %c-616729560_i32 : i32
          %340 = ascvec.duplicate_scalar %339, %mask : i32 -> <ui32>
          %341 = ascvec.xor %335, %338, %mask : <ui32>, <ui32> -> <ui32>
          %342 = ascvec.xor %336, %340, %mask : <ui32>, <ui32> -> <ui32>
          %dst0_62, %dst1_63 = ascvec.mull %341, %271, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %dst0_64, %dst1_65 = ascvec.mull %342, %272, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %343 = ascvec.xor %dst1_65, %dst0_60, %mask : <ui32>, <ui32> -> <ui32>
          %344 = ascvec.xor %dst1_63, %dst0_58, %mask : <ui32>, <ui32> -> <ui32>
          %345 = arith.addi %arg12, %c-1879881855_i32 : i32
          %346 = ascvec.duplicate_scalar %345, %mask : i32 -> <ui32>
          %347 = arith.addi %arg13, %c-1767562579_i32 : i32
          %348 = ascvec.duplicate_scalar %347, %mask : i32 -> <ui32>
          %349 = ascvec.xor %343, %346, %mask : <ui32>, <ui32> -> <ui32>
          %350 = ascvec.xor %344, %348, %mask : <ui32>, <ui32> -> <ui32>
          %351 = arith.muli %arg16, %c128_i32 : i32
          %352 = ascvec.reinterpret %349 : <ui32> -> <i32>
          %353 = arith.index_cast %351 : i32 to index
          ascvec.storealign %arg14[%353], %352, %mask : memref<4096xi32, 1>, index, <i32>
          %354 = arith.addi %351, %c32_i32_15 : i32
          %355 = ascvec.reinterpret %dst0_64 : <ui32> -> <i32>
          %356 = arith.index_cast %354 : i32 to index
          ascvec.storealign %arg14[%356], %355, %mask : memref<4096xi32, 1>, index, <i32>
          %357 = arith.addi %351, %c64_i32_11 : i32
          %358 = ascvec.reinterpret %350 : <ui32> -> <i32>
          %359 = arith.index_cast %357 : i32 to index
          ascvec.storealign %arg14[%359], %358, %mask : memref<4096xi32, 1>, index, <i32>
          %360 = arith.addi %351, %c96_i32 : i32
          %361 = ascvec.reinterpret %dst0_62 : <ui32> -> <i32>
          %362 = arith.index_cast %360 : i32 to index
          ascvec.storealign %arg14[%362], %361, %mask : memref<4096xi32, 1>, index, <i32>
        }
        ascvec.mem_bar vst_vld
      } {cannir.vf_mode = "simd"}
      %117 = arith.divsi %115, %c32_i32 : i32
      %118 = arith.divsi %115, %c64_i32 : i32
      ascvec.get_buf v, %46, 0 {cannir.buffer.generated_sync}
      ascvec.get_buf v, %82, 0 {cannir.buffer.generated_sync}
      ascvec.get_buf v, %87, 0 {cannir.buffer.generated_sync}
      %119 = ascscalar.bitcast_memref %86 byte_offset = %c0_i64 : memref<1024xui8, 1> -> memref<?xui16, 1>
      ascvec.vf(%arg4, %77, %alloc_9, %alloc_10, %81, %117, %119, %118 : i64, memref<16384xbf16, 1>, memref<512xui32, 1>, memref<4096xi32, 1>, memref<16384xui8, 1>, i32, memref<?xui16, 1>, i32) {
      ^bb0(%arg8: i64, %arg9: memref<16384xbf16, 1>, %arg10: memref<512xui32, 1>, %arg11: memref<4096xi32, 1>, %arg12: memref<16384xui8, 1>, %arg13: i32, %arg14: memref<?xui16, 1>, %arg15: i32):
        %c64_i32_11 = arith.constant 64 : i32
        %c256_i32 = arith.constant 256 : i32
        %c3_i32 = arith.constant 3 : i32
        %c2139095040_i32 = arith.constant 2139095040 : i32
        %c127_i32 = arith.constant 127 : i32
        %c126_i32 = arith.constant 126 : i32
        %c65535_i32 = arith.constant 65535 : i32
        %c1015021568_i32 = arith.constant 1015021568 : i32
        %cst = arith.constant 0x4C000000 : f32
        %c62914560_i32 = arith.constant 62914560 : i32
        %c128_i32 = arith.constant 128 : i32
        %c4_i32 = arith.constant 4 : i32
        %c2139160576_i32 = arith.constant 2139160576 : i32
        %c254_i32 = arith.constant 254 : i32
        %c-8_i32 = arith.constant -8 : i32
        %c8_i32 = arith.constant 8 : i32
        %c255_i32 = arith.constant 255 : i32
        %c1_i32 = arith.constant 1 : i32
        %c16_i32_12 = arith.constant 16 : i32
        %c2147483647_i32 = arith.constant 2147483647 : i32
        %c2_i32 = arith.constant 2 : i32
        %c0_i32_13 = arith.constant 0 : i32
        %c32_i32_14 = arith.constant 32 : i32
        %mask, %remaining_out = ascvec.update_mask %c32_i32_14 : !ascvec.mask
        scf.for %arg16 = %c0_i32_13 to %arg13 step %c1_i32  : i32 {
          %201 = ascvec.create_mask : !ascvec.mask
          %202 = ascvec.arange %c0_i32_13 : i32 -> <ui32>
          %203 = ascvec.duplicate_scalar %c0_i32_13, %201 : i32 -> <ui32>
          %204 = scf.for %arg17 = %c0_i32_13 to %c16_i32_12 step %c1_i32 iter_args(%arg18 = %203) -> (!ascvec.vreg<ui32>)  : i32 {
            %222 = arith.muli %arg16, %c32_i32_14 : i32
            %223 = arith.muli %arg17, %c2_i32 overflow<nsw> : i32
            %224 = arith.addi %222, %223 : i32
            %225 = arith.extsi %224 : i32 to i64
            %226 = arith.muli %225, %arg8 : i64
            %227 = arith.index_cast %226 : i64 to index
            %228 = ascvec.loadalign_unpack %arg9[%227] {mode = "b16_to_b32"} : memref<16384xbf16, 1>, index -> <bf16>
            %229 = ascvec.cast %228, %201 {rounding = 6 : i32} : <bf16> -> <f32>
            %230 = ascvec.reinterpret %229 : <f32> -> <ui32>
            %231 = ascvec.duplicate_scalar %c2147483647_i32, %201 : i32 -> <ui32>
            %232 = ascvec.and %230, %231, %201 : <ui32>, <ui32> -> <ui32>
            %233 = ascvec.max %arg18, %232, %201 : <ui32>
            scf.yield %233 : !ascvec.vreg<ui32>
          }
          %205 = ascvec.duplicate_scalar %c32_i32_14, %201 : i32 -> <ui32>
          %206 = ascvec.xor %202, %205, %201 : <ui32>, <ui32> -> <ui32>
          %207 = ascvec.gather_reg %204, %206 : <ui32>, <ui32> -> <ui32>
          %208 = ascvec.max %204, %207, %201 : <ui32>
          %209 = ascvec.shr %208, 23, %mask : <ui32> -> <ui32>
          %210 = ascvec.cmp_ge_scalar %209, %c255_i32, %mask : <ui32>, i32 -> !ascvec.mask
          %211 = ascvec.max_scalar %209, %c8_i32, %mask : <ui32>, i32
          %212 = ascvec.add_scalar %211, %c-8_i32, %mask : <ui32>, i32
          %213 = ascvec.duplicate_scalar %c255_i32, %210 merge %212 : !ascvec.vreg<ui32> {mode = "merging"} : i32 -> <ui32>
          %214 = ascvec.duplicate_scalar %c254_i32, %mask : i32 -> <ui32>
          %215 = ascvec.sub %214, %213, %mask : <ui32>
          %216 = ascvec.cmp_ne_scalar %213, %c0_i32_13, %mask : <ui32>, i32 -> !ascvec.mask
          %217 = ascvec.shl %215, 23, %216 : <ui32> -> <ui32>
          %218 = ascvec.duplicate_scalar %c2139160576_i32, %210 merge %217 : !ascvec.vreg<ui32> {mode = "merging"} : i32 -> <ui32>
          %219 = ascvec.reinterpret %218 : <ui32> -> <f32>
          %220 = arith.muli %arg16, %c32_i32_14 : i32
          %221 = arith.index_cast %220 : i32 to index
          ascvec.storealign %arg10[%221], %213, %mask : memref<512xui32, 1>, index, <ui32>
          scf.for %arg17 = %c0_i32_13 to %c8_i32 step %c1_i32  : i32 {
            %222 = arith.muli %arg16, %c256_i32 : i32
            %223 = arith.muli %arg17, %c32_i32_14 overflow<nsw> : i32
            %224 = arith.addi %222, %223 : i32
            %225 = arith.index_cast %224 : i32 to index
            %226 = ascvec.loadalign %arg11[%225] : memref<4096xi32, 1>, index -> <i32>
            %227 = ascvec.reinterpret %226 : <i32> -> <ui32>
            %228 = ascvec.shl %227, 8, %mask : <ui32> -> <ui32>
            %229 = ascvec.shr %227, 24, %mask : <ui32> -> <ui32>
            %230 = ascvec.or %228, %229, %mask : <ui32>, <ui32> -> <ui32>
            %231 = arith.muli %arg17, %c4_i32 overflow<nsw> : i32
            %232 = arith.addi %220, %231 : i32
            %233 = arith.extsi %232 : i32 to i64
            %234 = arith.muli %233, %arg8 : i64
            %235 = ascvec.shr %227, 16, %mask : <ui32> -> <ui32>
            %236 = arith.index_cast %234 : i64 to index
            %237 = ascvec.loadalign_unpack %arg9[%236] {mode = "b16_to_b32"} : memref<16384xbf16, 1>, index -> <bf16>
            %238 = ascvec.cast %237, %mask {rounding = 6 : i32} : <bf16> -> <f32>
            %239 = ascvec.mul %238, %219, %mask : <f32>
            %240 = ascvec.cast %239, %mask : <f32> -> <bf16>
            %241 = ascvec.cast %240, %mask {rounding = 6 : i32} : <bf16> -> <f32>
            %242 = ascvec.reinterpret %241 : <f32> -> <ui32>
            %243 = ascvec.shr %242, 24, %mask : <ui32> -> <ui32>
            %244 = ascvec.duplicate_scalar %c128_i32, %mask : i32 -> <ui32>
            %245 = ascvec.and %243, %244, %mask : <ui32>, <ui32> -> <ui32>
            %246 = ascvec.duplicate_scalar %c2147483647_i32, %mask : i32 -> <ui32>
            %247 = ascvec.and %242, %246, %mask : <ui32>, <ui32> -> <ui32>
            %248 = ascvec.shr %247, 4, %mask : <ui32> -> <ui32>
            %249 = ascvec.duplicate_scalar %c62914560_i32, %mask : i32 -> <ui32>
            %250 = ascvec.sub %248, %249, %mask : <ui32>
            %251 = ascvec.reinterpret %247 : <ui32> -> <f32>
            %252 = ascvec.mul_scalar %251, %cst, %mask : <f32>, f32
            %253 = ascvec.cast %252, %mask {rounding = 4 : i32} : <f32> -> <i32>
            %254 = ascvec.reinterpret %253 : <i32> -> <ui32>
            %255 = ascvec.cmp_ge_scalar %247, %c1015021568_i32, %mask : <ui32>, i32 -> !ascvec.mask
            %256 = ascvec.select %250, %254, %255 : <ui32>
            %257 = ascvec.duplicate_scalar %c65535_i32, %mask : i32 -> <ui32>
            %258 = ascvec.and %235, %257, %mask : <ui32>, <ui32> -> <ui32>
            %259 = ascvec.add %256, %258, %mask : <ui32>
            %260 = ascvec.shr %259, 16, %mask : <ui32> -> <ui32>
            %261 = ascvec.min_scalar %260, %c126_i32, %mask : <ui32>, i32
            %262 = ascvec.or %261, %245, %mask : <ui32>, <ui32> -> <ui32>
            %263 = ascvec.duplicate_scalar %c127_i32, %mask : i32 -> <ui32>
            %264 = ascvec.cmp_gt_scalar %247, %c2139095040_i32, %mask : <ui32>, i32 -> !ascvec.mask
            %265 = ascvec.select %263, %262, %264 : <ui32>
            ascvec.storealign_pack %arg12[%236], %265, %mask {mode = "b32_to_b8"} : memref<16384xui8, 1>, index, <ui32>
            %266 = arith.addi %232, %c1_i32 : i32
            %267 = arith.extsi %266 : i32 to i64
            %268 = arith.muli %267, %arg8 : i64
            %269 = ascvec.and %230, %257, %mask : <ui32>, <ui32> -> <ui32>
            %270 = arith.index_cast %268 : i64 to index
            %271 = ascvec.loadalign_unpack %arg9[%270] {mode = "b16_to_b32"} : memref<16384xbf16, 1>, index -> <bf16>
            %272 = ascvec.cast %271, %mask {rounding = 6 : i32} : <bf16> -> <f32>
            %273 = ascvec.mul %272, %219, %mask : <f32>
            %274 = ascvec.cast %273, %mask : <f32> -> <bf16>
            %275 = ascvec.cast %274, %mask {rounding = 6 : i32} : <bf16> -> <f32>
            %276 = ascvec.reinterpret %275 : <f32> -> <ui32>
            %277 = ascvec.shr %276, 24, %mask : <ui32> -> <ui32>
            %278 = ascvec.and %277, %244, %mask : <ui32>, <ui32> -> <ui32>
            %279 = ascvec.and %276, %246, %mask : <ui32>, <ui32> -> <ui32>
            %280 = ascvec.shr %279, 4, %mask : <ui32> -> <ui32>
            %281 = ascvec.sub %280, %249, %mask : <ui32>
            %282 = ascvec.reinterpret %279 : <ui32> -> <f32>
            %283 = ascvec.mul_scalar %282, %cst, %mask : <f32>, f32
            %284 = ascvec.cast %283, %mask {rounding = 4 : i32} : <f32> -> <i32>
            %285 = ascvec.reinterpret %284 : <i32> -> <ui32>
            %286 = ascvec.cmp_ge_scalar %279, %c1015021568_i32, %mask : <ui32>, i32 -> !ascvec.mask
            %287 = ascvec.select %281, %285, %286 : <ui32>
            %288 = ascvec.and %269, %257, %mask : <ui32>, <ui32> -> <ui32>
            %289 = ascvec.add %287, %288, %mask : <ui32>
            %290 = ascvec.shr %289, 16, %mask : <ui32> -> <ui32>
            %291 = ascvec.min_scalar %290, %c126_i32, %mask : <ui32>, i32
            %292 = ascvec.or %291, %278, %mask : <ui32>, <ui32> -> <ui32>
            %293 = ascvec.cmp_gt_scalar %279, %c2139095040_i32, %mask : <ui32>, i32 -> !ascvec.mask
            %294 = ascvec.select %263, %292, %293 : <ui32>
            ascvec.storealign_pack %arg12[%270], %294, %mask {mode = "b32_to_b8"} : memref<16384xui8, 1>, index, <ui32>
            %295 = arith.addi %232, %c2_i32 : i32
            %296 = arith.extsi %295 : i32 to i64
            %297 = arith.muli %296, %arg8 : i64
            %298 = ascvec.and %227, %257, %mask : <ui32>, <ui32> -> <ui32>
            %299 = arith.index_cast %297 : i64 to index
            %300 = ascvec.loadalign_unpack %arg9[%299] {mode = "b16_to_b32"} : memref<16384xbf16, 1>, index -> <bf16>
            %301 = ascvec.cast %300, %mask {rounding = 6 : i32} : <bf16> -> <f32>
            %302 = ascvec.mul %301, %219, %mask : <f32>
            %303 = ascvec.cast %302, %mask : <f32> -> <bf16>
            %304 = ascvec.cast %303, %mask {rounding = 6 : i32} : <bf16> -> <f32>
            %305 = ascvec.reinterpret %304 : <f32> -> <ui32>
            %306 = ascvec.shr %305, 24, %mask : <ui32> -> <ui32>
            %307 = ascvec.and %306, %244, %mask : <ui32>, <ui32> -> <ui32>
            %308 = ascvec.and %305, %246, %mask : <ui32>, <ui32> -> <ui32>
            %309 = ascvec.shr %308, 4, %mask : <ui32> -> <ui32>
            %310 = ascvec.sub %309, %249, %mask : <ui32>
            %311 = ascvec.reinterpret %308 : <ui32> -> <f32>
            %312 = ascvec.mul_scalar %311, %cst, %mask : <f32>, f32
            %313 = ascvec.cast %312, %mask {rounding = 4 : i32} : <f32> -> <i32>
            %314 = ascvec.reinterpret %313 : <i32> -> <ui32>
            %315 = ascvec.cmp_ge_scalar %308, %c1015021568_i32, %mask : <ui32>, i32 -> !ascvec.mask
            %316 = ascvec.select %310, %314, %315 : <ui32>
            %317 = ascvec.and %298, %257, %mask : <ui32>, <ui32> -> <ui32>
            %318 = ascvec.add %316, %317, %mask : <ui32>
            %319 = ascvec.shr %318, 16, %mask : <ui32> -> <ui32>
            %320 = ascvec.min_scalar %319, %c126_i32, %mask : <ui32>, i32
            %321 = ascvec.or %320, %307, %mask : <ui32>, <ui32> -> <ui32>
            %322 = ascvec.cmp_gt_scalar %308, %c2139095040_i32, %mask : <ui32>, i32 -> !ascvec.mask
            %323 = ascvec.select %263, %321, %322 : <ui32>
            ascvec.storealign_pack %arg12[%299], %323, %mask {mode = "b32_to_b8"} : memref<16384xui8, 1>, index, <ui32>
            %324 = arith.addi %232, %c3_i32 : i32
            %325 = arith.extsi %324 : i32 to i64
            %326 = arith.muli %325, %arg8 : i64
            %327 = ascvec.shr %230, 16, %mask : <ui32> -> <ui32>
            %328 = arith.index_cast %326 : i64 to index
            %329 = ascvec.loadalign_unpack %arg9[%328] {mode = "b16_to_b32"} : memref<16384xbf16, 1>, index -> <bf16>
            %330 = ascvec.cast %329, %mask {rounding = 6 : i32} : <bf16> -> <f32>
            %331 = ascvec.mul %330, %219, %mask : <f32>
            %332 = ascvec.cast %331, %mask : <f32> -> <bf16>
            %333 = ascvec.cast %332, %mask {rounding = 6 : i32} : <bf16> -> <f32>
            %334 = ascvec.reinterpret %333 : <f32> -> <ui32>
            %335 = ascvec.shr %334, 24, %mask : <ui32> -> <ui32>
            %336 = ascvec.and %335, %244, %mask : <ui32>, <ui32> -> <ui32>
            %337 = ascvec.and %334, %246, %mask : <ui32>, <ui32> -> <ui32>
            %338 = ascvec.shr %337, 4, %mask : <ui32> -> <ui32>
            %339 = ascvec.sub %338, %249, %mask : <ui32>
            %340 = ascvec.reinterpret %337 : <ui32> -> <f32>
            %341 = ascvec.mul_scalar %340, %cst, %mask : <f32>, f32
            %342 = ascvec.cast %341, %mask {rounding = 4 : i32} : <f32> -> <i32>
            %343 = ascvec.reinterpret %342 : <i32> -> <ui32>
            %344 = ascvec.cmp_ge_scalar %337, %c1015021568_i32, %mask : <ui32>, i32 -> !ascvec.mask
            %345 = ascvec.select %339, %343, %344 : <ui32>
            %346 = ascvec.and %327, %257, %mask : <ui32>, <ui32> -> <ui32>
            %347 = ascvec.add %345, %346, %mask : <ui32>
            %348 = ascvec.shr %347, 16, %mask : <ui32> -> <ui32>
            %349 = ascvec.min_scalar %348, %c126_i32, %mask : <ui32>, i32
            %350 = ascvec.or %349, %336, %mask : <ui32>, <ui32> -> <ui32>
            %351 = ascvec.cmp_gt_scalar %337, %c2139095040_i32, %mask : <ui32>, i32 -> !ascvec.mask
            %352 = ascvec.select %263, %350, %351 : <ui32>
            ascvec.storealign_pack %arg12[%328], %352, %mask {mode = "b32_to_b8"} : memref<16384xui8, 1>, index, <ui32>
          }
        }
        ascvec.mem_bar vst_vld
        %mask_15, %remaining_out_16 = ascvec.update_mask %c32_i32_14 elem_bits 16 : !ascvec.mask
        scf.for %arg16 = %c0_i32_13 to %arg15 step %c1_i32  : i32 {
          %201 = arith.muli %arg16, %c64_i32_11 : i32
          %202 = arith.index_cast %201 : i32 to index
          %203 = ascvec.loadalign %arg10[%202] : memref<512xui32, 1>, index -> <ui32>
          %204 = arith.addi %201, %c32_i32_14 : i32
          %205 = arith.index_cast %204 : i32 to index
          %206 = ascvec.loadalign %arg10[%205] : memref<512xui32, 1>, index -> <ui32>
          %207 = ascvec.shl %206, 8, %mask : <ui32> -> <ui32>
          %208 = ascvec.or %203, %207, %mask : <ui32>, <ui32> -> <ui32>
          %209 = ascvec.pack %208 src_elem_bits 32 dst_elem_bits 16 {part = "low"} : <ui32> -> <ui16>
          %210 = arith.extsi %arg16 : i32 to i64
          %211 = arith.muli %210, %arg8 : i64
          %212 = arith.index_cast %211 : i64 to index
          ascvec.storealign %arg14[%212], %209, %mask_15 : memref<?xui16, 1>, index, <ui16>
        }
        ascvec.mem_bar vst_vld
      } {cannir.vf_mode = "simd"}
      ascvec.release_buf v, %87, 0 {cannir.buffer.generated_sync}
      ascvec.release_buf v, %82, 0 {cannir.buffer.generated_sync}
      ascvec.release_buf v, %46, 0 {cannir.buffer.generated_sync}
      %120 = cannir.extract_layout %arg1 : !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
      %121 = cannir.extract_shape %120 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, -1)>
      %122 = cannir.extract_dim %121[0] : !cannir.shape<(-1, -1)>
      %123 = arith.minsi %37, %122 : i64
      %124 = arith.subi %122, %123 : i64
      %125 = arith.minsi %124, %c1_i64 : i64
      %126 = arith.cmpi sle, %37, %122 : i64
      %127 = arith.select %126, %125, %c0_i64 : i64
      %128 = arith.cmpi sgt, %127, %c1_i64 : i64
      %129 = arith.select %128, %c1_i64, %127 : i64
      %130 = cannir.extract_dim %121[1] : !cannir.shape<(-1, -1)>
      %131 = arith.divsi %130, %59 : i64
      %132 = arith.minsi %39, %131 : i64
      %133 = arith.muli %132, %40 : i64
      %134 = arith.subi %130, %133 : i64
      %135 = arith.minsi %40, %134 : i64
      %136 = arith.cmpi sle, %39, %131 : i64
      %137 = arith.select %136, %135, %c0_i64 : i64
      %138 = arith.cmpi sgt, %137, %c16384_i64 : i64
      %139 = arith.select %138, %c16384_i64, %137 : i64
      %140 = cannir.extract_physical_layout %arg1 : !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
      %141 = cannir.extract_stride %140 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
      %142 = cannir.extract_dim %141[0] : !cannir.stride<(-1, 1)>
      %143 = arith.muli %37, %142 : i64
      %144 = arith.addi %143, %69 : i64
      %145 = arith.muli %144, %c8_i64 : i64
      %146 = ascscalar.memref_address %alloc_0 : memref<1xui8>
      %147 = arith.divui %145, %c8_i64 : i64
      %148 = arith.addi %146, %147 : i64
      %149 = ascscalar.bind_gm_ptr %148 : memref<1xui8>
      %150 = ascscalar.bind_local_ptr %80 : memref<1xui8, 1>
      %151 = cannir.extract_stride %120 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
      %152 = cannir.extract_dim %151[0] : !cannir.stride<(-1, 1)>
      %153 = arith.cmpi ugt, %129, %c0_i64 : i64
      %154 = arith.cmpi ugt, %139, %c0_i64 : i64
      %155 = arith.andi %153, %154 : i1
      scf.if %155 {
        ascvec.get_buf mte3, %82, 0 {cannir.buffer.generated_sync}
        %201 = arith.trunci %129 : i64 to i32
        %202 = arith.trunci %139 : i64 to i32
        ascvec.copy_ub2gm %149[%c0], %150[%c0], %201, %202, %152, %c16384_i32 l2_cache_ctl %c0_i32 : memref<1xui8>, memref<1xui8, 1>
        ascvec.release_buf mte3, %82, 0 {cannir.buffer.generated_sync}
      }
      %156 = arith.remsi %40, %c32_i64 : i64
      %157 = arith.cmpi ne, %156, %c0_i64 : i64
      %158 = arith.cmpi slt, %156, %c0_i64 : i64
      %159 = arith.andi %157, %158 : i1
      %160 = arith.divsi %40, %c32_i64 : i64
      %161 = arith.extui %159 : i1 to i64
      %162 = arith.subi %160, %161 : i64
      %163 = cannir.extract_layout %arg2 : !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
      %164 = cannir.extract_shape %163 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, -1)>
      %165 = cannir.extract_dim %164[0] : !cannir.shape<(-1, -1)>
      %166 = arith.minsi %37, %165 : i64
      %167 = arith.subi %165, %166 : i64
      %168 = arith.minsi %167, %c1_i64 : i64
      %169 = arith.cmpi sle, %37, %165 : i64
      %170 = arith.select %169, %168, %c0_i64 : i64
      %171 = arith.cmpi sgt, %170, %c1_i64 : i64
      %172 = arith.select %171, %c1_i64, %170 : i64
      %173 = cannir.extract_dim %164[1] : !cannir.shape<(-1, -1)>
      %174 = arith.maxsi %162, %c1_i64 : i64
      %175 = arith.divsi %173, %174 : i64
      %176 = arith.minsi %39, %175 : i64
      %177 = arith.muli %176, %162 : i64
      %178 = arith.subi %173, %177 : i64
      %179 = arith.minsi %162, %178 : i64
      %180 = arith.cmpi sle, %39, %175 : i64
      %181 = arith.select %180, %179, %c0_i64 : i64
      %182 = arith.cmpi sgt, %181, %c1024_i64 : i64
      %183 = arith.select %182, %c1024_i64, %181 : i64
      %184 = cannir.extract_physical_layout %arg2 : !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
      %185 = cannir.extract_stride %184 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
      %186 = cannir.extract_dim %185[0] : !cannir.stride<(-1, 1)>
      %187 = arith.muli %37, %186 : i64
      %188 = arith.muli %39, %162 : i64
      %189 = arith.addi %187, %188 : i64
      %190 = arith.muli %189, %c8_i64 : i64
      %191 = ascscalar.memref_address %alloc_1 : memref<1xui8>
      %192 = arith.divui %190, %c8_i64 : i64
      %193 = arith.addi %191, %192 : i64
      %194 = ascscalar.bind_gm_ptr %193 : memref<1xui8>
      %195 = ascscalar.bind_local_ptr %85 : memref<1xui8, 1>
      %196 = cannir.extract_stride %163 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
      %197 = cannir.extract_dim %196[0] : !cannir.stride<(-1, 1)>
      %198 = arith.cmpi ugt, %172, %c0_i64 : i64
      %199 = arith.cmpi ugt, %183, %c0_i64 : i64
      %200 = arith.andi %198, %199 : i1
      scf.if %200 {
        ascvec.get_buf mte3, %87, 0 {cannir.buffer.generated_sync}
        %201 = arith.trunci %172 : i64 to i32
        %202 = arith.trunci %183 : i64 to i32
        ascvec.copy_ub2gm %194[%c0], %195[%c0], %201, %202, %197, %c1024_i32 l2_cache_ctl %c0_i32 : memref<1xui8>, memref<1xui8, 1>
        ascvec.release_buf mte3, %87, 0 {cannir.buffer.generated_sync}
      }
      scf.yield %48 : i64
    }
    return
  }
}
