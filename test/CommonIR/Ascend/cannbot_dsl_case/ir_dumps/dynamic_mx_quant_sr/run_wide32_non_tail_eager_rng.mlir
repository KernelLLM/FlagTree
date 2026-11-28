module {
  func.func @run_wide32_non_tail_eager_rng_fa2d3ba77377be0c1dfffc1b92b0babb(%arg0: !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg1: !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg2: !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64) attributes {cannbotdsl.contract_digest = "fa2d3ba77377be0c1dfffc1b92b0babbdc6c8d53adfdea2a0441be0cef39eccd", cannbotdsl.public_slot_count = 8 : i64, cannbotdsl.slot_dtype_bits = [16, 8, 8, 64, 64, 64, 64, 64], cannbotdsl.slot_dtype_codes = [4, 1, 1, 0, 0, 0, 0, 0], cannbotdsl.slot_dtype_lanes = [1, 1, 1, 1, 1, 1, 1, 1], cannbotdsl.slot_flags = [1, 1, 1, 1, 1, 1, 1, 1], cannbotdsl.slot_kinds = [1, 1, 1, 2, 2, 2, 2, 2], cannbotdsl.slot_type_digests = ["5d7cdc96dc21b3b739e32aeccf58c1df113bec54701c75107ed650f2544a07da", "171ddd09eb070fb4a6413ae87b2ceecddc233a9a173a0404e96285a4bb8b8659", "891bcd21e721f8d1a666bcca4e8705ec63796c1add8ae000536d0d98f0274d3a", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515"], cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.host} {
    cannir.kernel_launch @kernel__DynamicMxQuantSrKernel_run_wide32_non_tail_0(%arg5) [%arg0, %arg1, %arg2, %arg3, %arg4, %arg6, %arg7] {cannbotdsl.profile = {indices = array<i64: 0, 1, 2>, name = "_DynamicMxQuantSrKernel_run_wide32_non_tail", op_type = "_DynamicMxQuantSrKernel_run_wide32_non_tail", roles = array<i64: 0, 1, 1>}} : !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, i64, i64, i64, i64
    return
  }
  func.func @kernel__DynamicMxQuantSrKernel_run_wide32_non_tail_0(%arg0: !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg1: !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg2: !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64) attributes {cannir.buffer.auto_sync, cannir.buffer.sync_analyzed, cannir.buffer.sync_verified, cannir.buffer.tensor_state, cannir.core = "aiv", cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.device, cannir.explicit_ring_buffer_cursors, cannir.kernel_group = "kernel__DynamicMxQuantSrKernel_run_wide32_non_tail_0", cannir.resource.ids_prepared} {
    %c1024_i32 = arith.constant 1024 : i32
    %c16384_i32 = arith.constant 16384 : i32
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
    %alloc_2 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<16384xbf16, 1>
    %alloc_3 = memref.alloc() {cannir.byte_offset = 32768 : i64} : memref<16384xbf16, 1>
    %alloc_4 = memref.alloc() {cannir.byte_offset = 65536 : i64} : memref<16384xui8, 1>
    %alloc_5 = memref.alloc() {cannir.byte_offset = 81920 : i64} : memref<16384xui8, 1>
    %alloc_6 = memref.alloc() {cannir.byte_offset = 98304 : i64} : memref<1024xui8, 1>
    %alloc_7 = memref.alloc() {cannir.byte_offset = 99328 : i64} : memref<1024xui8, 1>
    %alloc_8 = memref.alloc() {cannir.byte_offset = 100352 : i64} : memref<512xui32, 1>
    %alloc_9 = memref.alloc() {cannir.byte_offset = 102400 : i64} : memref<4096xi32, 1>
    %0 = cannir.extract_layout %arg0 : !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
    %1 = cannir.extract_shape %0 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, -1)>
    %2 = cannir.extract_dim %1[0] : !cannir.shape<(-1, -1)>
    %3 = arith.remsi %arg3, %c64_i64 : i64
    %4 = arith.cmpi ne, %3, %c0_i64 : i64
    %5 = arith.cmpi slt, %3, %c0_i64 : i64
    %6 = arith.andi %4, %5 : i1
    %7 = arith.divsi %arg3, %c64_i64 : i64
    %8 = arith.extui %6 : i1 to i64
    %9 = arith.subi %7, %8 : i64
    %10 = arith.muli %2, %9 : i64
    %11 = arith.cmpi sge, %10, %c256_i64 : i64
    %12 = scf.if %11 -> (i1) {
      %25 = arith.remsi %arg3, %c512_i64 : i64
      %26 = arith.cmpi ne, %25, %c0_i64 : i64
      %27 = arith.cmpi slt, %25, %c0_i64 : i64
      %28 = arith.andi %26, %27 : i1
      %29 = arith.select %28, %c512_i64, %c0_i64 : i64
      %30 = arith.addi %25, %29 overflow<nsw> : i64
      %31 = arith.cmpi eq, %30, %c0_i64 : i64
      %32 = arith.cmpi eq, %arg4, %c32_i64 : i64
      %33 = arith.select %31, %32, %false : i1
      scf.yield %33 : i1
    } else {
      scf.yield %false : i1
    }
    %13 = scf.if %12 -> (i64) {
      scf.yield %c512_i64 : i64
    } else {
      %25 = scf.if %11 -> (i1) {
        %27 = arith.remsi %arg3, %c256_i64 : i64
        %28 = arith.cmpi ne, %27, %c0_i64 : i64
        %29 = arith.cmpi slt, %27, %c0_i64 : i64
        %30 = arith.andi %28, %29 : i1
        %31 = arith.select %30, %c256_i64, %c0_i64 : i64
        %32 = arith.addi %27, %31 overflow<nsw> : i64
        %33 = arith.cmpi eq, %32, %c0_i64 : i64
        %34 = arith.cmpi eq, %arg4, %c32_i64 : i64
        %35 = arith.select %33, %34, %false : i1
        scf.yield %35 : i1
      } else {
        scf.yield %false : i1
      }
      %26 = scf.if %25 -> (i64) {
        scf.yield %c256_i64 : i64
      } else {
        %27 = arith.remsi %arg3, %c128_i64 : i64
        %28 = arith.cmpi ne, %27, %c0_i64 : i64
        %29 = arith.cmpi slt, %27, %c0_i64 : i64
        %30 = arith.andi %28, %29 : i1
        %31 = arith.select %30, %c128_i64, %c0_i64 : i64
        %32 = arith.addi %27, %31 overflow<nsw> : i64
        %33 = arith.cmpi eq, %32, %c0_i64 : i64
        %34 = arith.select %11, %33, %false : i1
        %35 = arith.select %34, %c128_i64, %c64_i64 : i64
        scf.yield %35 : i64
      }
      scf.yield %26 : i64
    }
    %14 = arith.remsi %arg3, %13 : i64
    %15 = arith.cmpi ne, %14, %c0_i64 : i64
    %16 = arith.cmpi slt, %14, %c0_i64 : i64
    %17 = arith.andi %15, %16 : i1
    %18 = arith.divsi %arg3, %13 : i64
    %19 = arith.extui %17 : i1 to i64
    %20 = arith.subi %18, %19 : i64
    %21 = arith.muli %2, %20 : i64
    %22 = ascscalar.get_block_idx
    %23 = ascscalar.get_block_num
    %24 = scf.for %arg7 = %22 to %21 step %23 iter_args(%arg8 = %c0_i64) -> (i64)  : i64 {
      %25 = arith.remsi %arg7, %20 : i64
      %26 = arith.cmpi ne, %25, %c0_i64 : i64
      %27 = arith.cmpi slt, %25, %c0_i64 : i64
      %28 = arith.cmpi slt, %20, %c0_i64 : i64
      %29 = arith.xori %27, %28 : i1
      %30 = arith.andi %26, %29 : i1
      %31 = arith.divsi %arg7, %20 : i64
      %32 = arith.extui %30 : i1 to i64
      %33 = arith.subi %31, %32 : i64
      %34 = arith.select %30, %20, %c0_i64 : i64
      %35 = arith.addi %25, %34 : i64
      %36 = arith.muli %13, %arg4 : i64
      %37 = arith.cmpi eq, %arg8, %c0_i64 : i64
      %38 = ascscalar.memref_address %alloc_2 : memref<16384xbf16, 1>
      %39 = ascscalar.memref_address %alloc_3 : memref<16384xbf16, 1>
      %40 = arith.select %37, %38, %39 : i64
      %41 = arith.cmpi ne, %arg8, %c0_i64 : i64
      %42 = arith.extui %41 : i1 to i64
      %43 = arith.addi %arg8, %c1_i64 overflow<nsw> : i64
      %44 = arith.remui %43, %c2_i64 : i64
      %45 = arith.minsi %33, %2 : i64
      %46 = arith.subi %2, %45 : i64
      %47 = arith.minsi %46, %c1_i64 : i64
      %48 = arith.cmpi sle, %33, %2 : i64
      %49 = arith.select %48, %47, %c0_i64 : i64
      %50 = cannir.extract_physical_layout %arg0 : !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
      %51 = cannir.extract_stride %50 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
      %52 = cannir.extract_dim %51[0] : !cannir.stride<(-1, 1)>
      %53 = arith.muli %52, %c2_i64 : i64
      %54 = cannir.extract_dim %1[1] : !cannir.shape<(-1, -1)>
      %55 = arith.maxsi %36, %c1_i64 : i64
      %56 = arith.divsi %54, %55 : i64
      %57 = arith.minsi %35, %56 : i64
      %58 = arith.muli %57, %36 : i64
      %59 = arith.subi %54, %58 : i64
      %60 = arith.minsi %36, %59 : i64
      %61 = arith.cmpi sle, %35, %56 : i64
      %62 = arith.select %61, %60, %c0_i64 : i64
      %63 = arith.muli %33, %52 : i64
      %64 = arith.muli %63, %c2_i64 : i64
      %65 = arith.muli %35, %36 : i64
      %66 = arith.muli %65, %c2_i64 : i64
      %67 = arith.addi %64, %66 : i64
      %68 = arith.muli %62, %c2_i64 : i64
      %69 = arith.cmpi eq, %68, %c0_i64 : i64
      %70 = arith.select %69, %c0_i64, %49 : i64
      %71 = arith.select %69, %c2_i64, %68 : i64
      %72 = ascscalar.bind_local_ptr %40 : memref<1xbf16, 1>
      ascvec.get_buf mte2, %42, 0 {cannir.buffer.generated_sync}
      "ascvec.copy_gm2ub_plan"(%72, %alloc, %c0_i32, %c0_i32, %c0_i64, %67, %70, %71, %53, %c32768_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
      ascvec.release_buf mte2, %42, 0 {cannir.buffer.generated_sync}
      %73 = ascscalar.bind_local_ptr %40 : memref<16384xbf16, 1>
      %74 = ascscalar.memref_address %alloc_4 : memref<16384xui8, 1>
      %75 = ascscalar.memref_address %alloc_5 : memref<16384xui8, 1>
      %76 = arith.select %37, %74, %75 : i64
      %77 = ascscalar.bind_local_ptr %76 : memref<16384xui8, 1>
      %78 = arith.select %37, %c2_i64, %c3_i64 : i64
      %79 = ascscalar.memref_address %alloc_6 : memref<1024xui8, 1>
      %80 = ascscalar.memref_address %alloc_7 : memref<1024xui8, 1>
      %81 = arith.select %37, %79, %80 : i64
      %82 = ascscalar.bind_local_ptr %81 : memref<1024xui8, 1>
      %83 = arith.select %37, %c4_i64, %c5_i64 : i64
      %84 = arith.muli %33, %arg4 : i64
      %85 = arith.muli %35, %13 : i64
      %86 = arith.remsi %85, %c16_i64 : i64
      %87 = arith.cmpi ne, %86, %c0_i64 : i64
      %88 = arith.cmpi slt, %86, %c0_i64 : i64
      %89 = arith.andi %87, %88 : i1
      %90 = arith.divsi %85, %c16_i64 : i64
      %91 = arith.extui %89 : i1 to i64
      %92 = arith.subi %90, %91 : i64
      %93 = arith.remsi %arg3, %c16_i64 : i64
      %94 = arith.cmpi ne, %93, %c0_i64 : i64
      %95 = arith.cmpi slt, %93, %c0_i64 : i64
      %96 = arith.andi %94, %95 : i1
      %97 = arith.divsi %arg3, %c16_i64 : i64
      %98 = arith.extui %96 : i1 to i64
      %99 = arith.subi %97, %98 : i64
      %100 = arith.trunci %arg5 : i64 to i32
      %101 = arith.extui %100 : i32 to i64
      %102 = arith.subi %arg5, %101 : i64
      %103 = arith.remsi %102, %c4294967296_i64 : i64
      %104 = arith.cmpi ne, %103, %c0_i64 : i64
      %105 = arith.cmpi slt, %103, %c0_i64 : i64
      %106 = arith.andi %104, %105 : i1
      %107 = arith.divsi %102, %c4294967296_i64 : i64
      %108 = arith.extui %106 : i1 to i64
      %109 = arith.subi %107, %108 : i64
      %110 = arith.trunci %109 : i64 to i32
      %111 = arith.trunci %13 : i64 to i32
      %112 = arith.divsi %111, %c16_i32 : i32
      ascvec.vf(%92, %84, %99, %arg6, %100, %110, %alloc_9, %112 : i64, i64, i64, i64, i32, i32, memref<4096xi32, 1>, i32) {
      ^bb0(%arg9: i64, %arg10: i64, %arg11: i64, %arg12: i64, %arg13: i32, %arg14: i32, %arg15: memref<4096xi32, 1>, %arg16: i32):
        %c128_i32 = arith.constant 128 : i32
        %c1_i32 = arith.constant 1 : i32
        %c96_i32 = arith.constant 96 : i32
        %c64_i32_10 = arith.constant 64 : i32
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
        %c0_i64_11 = arith.constant 0 : i64
        %c4294967296_i64_12 = arith.constant 4294967296 : i64
        %c0_i32_13 = arith.constant 0 : i32
        %c32_i32_14 = arith.constant 32 : i32
        %mask, %remaining_out = ascvec.update_mask %c32_i32_14 : !ascvec.mask
        %197 = ascvec.arange %c0_i32_13 : i32 -> <ui32>
        scf.for %arg17 = %c0_i32_13 to %arg16 step %c1_i32  : i32 {
          %198 = arith.extsi %arg17 : i32 to i64
          %199 = arith.addi %arg9, %198 : i64
          %200 = arith.trunci %arg10 : i64 to i32
          %201 = arith.extui %200 : i32 to i64
          %202 = arith.subi %arg10, %201 : i64
          %203 = arith.remsi %202, %c4294967296_i64_12 : i64
          %204 = arith.cmpi ne, %203, %c0_i64_11 : i64
          %205 = arith.cmpi slt, %203, %c0_i64_11 : i64
          %206 = arith.andi %204, %205 : i1
          %207 = arith.divsi %202, %c4294967296_i64_12 : i64
          %208 = arith.extui %206 : i1 to i64
          %209 = arith.subi %207, %208 : i64
          %210 = arith.trunci %209 : i64 to i32
          %211 = arith.trunci %199 : i64 to i32
          %212 = arith.extui %211 : i32 to i64
          %213 = arith.subi %199, %212 : i64
          %214 = arith.remsi %213, %c4294967296_i64_12 : i64
          %215 = arith.cmpi ne, %214, %c0_i64_11 : i64
          %216 = arith.cmpi slt, %214, %c0_i64_11 : i64
          %217 = arith.andi %215, %216 : i1
          %218 = arith.divsi %213, %c4294967296_i64_12 : i64
          %219 = arith.extui %217 : i1 to i64
          %220 = arith.subi %218, %219 : i64
          %221 = arith.trunci %220 : i64 to i32
          %222 = ascvec.duplicate_scalar %200, %mask : i32 -> <ui32>
          %223 = ascvec.duplicate_scalar %210, %mask : i32 -> <ui32>
          %224 = ascvec.duplicate_scalar %c0_i32_13, %mask : i32 -> <ui32>
          %carry, %dst = ascvec.addc %222, %197, %mask : <ui32>, <ui32> -> !ascvec.mask, <ui32>
          %carry_15, %dst_16 = ascvec.addc %223, %224, %carry, %mask : <ui32>, <ui32> -> !ascvec.mask, <ui32>
          %225 = ascvec.duplicate_scalar %211, %mask : i32 -> <ui32>
          %226 = ascvec.duplicate_scalar %221, %mask : i32 -> <ui32>
          %227 = arith.trunci %arg11 : i64 to i32
          %228 = arith.extui %227 : i32 to i64
          %229 = arith.subi %arg11, %228 : i64
          %230 = arith.remsi %229, %c4294967296_i64_12 : i64
          %231 = arith.cmpi ne, %230, %c0_i64_11 : i64
          %232 = arith.cmpi slt, %230, %c0_i64_11 : i64
          %233 = arith.andi %231, %232 : i1
          %234 = arith.divsi %229, %c4294967296_i64_12 : i64
          %235 = arith.extui %233 : i1 to i64
          %236 = arith.subi %234, %235 : i64
          %237 = arith.trunci %236 : i64 to i32
          %238 = ascvec.duplicate_scalar %227, %mask : i32 -> <ui32>
          %dst0, %dst1 = ascvec.mull %dst, %238, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %239 = ascvec.mul_scalar %dst, %237, %mask : <ui32>, i32
          %240 = ascvec.add %dst1, %239, %mask : <ui32>
          %241 = ascvec.mul_scalar %dst_16, %227, %mask : <ui32>, i32
          %242 = ascvec.add %240, %241, %mask : <ui32>
          %carry_17, %dst_18 = ascvec.addc %dst0, %225, %mask : <ui32>, <ui32> -> !ascvec.mask, <ui32>
          %carry_19, %dst_20 = ascvec.addc %242, %226, %carry_17, %mask : <ui32>, <ui32> -> !ascvec.mask, <ui32>
          %243 = arith.trunci %arg12 : i64 to i32
          %244 = arith.extui %243 : i32 to i64
          %245 = arith.subi %arg12, %244 : i64
          %246 = arith.remsi %245, %c4294967296_i64_12 : i64
          %247 = arith.cmpi ne, %246, %c0_i64_11 : i64
          %248 = arith.cmpi slt, %246, %c0_i64_11 : i64
          %249 = arith.andi %247, %248 : i1
          %250 = arith.divsi %245, %c4294967296_i64_12 : i64
          %251 = arith.extui %249 : i1 to i64
          %252 = arith.subi %250, %251 : i64
          %253 = arith.trunci %252 : i64 to i32
          %254 = arith.divui %243, %c4_i32 : i32
          %255 = arith.remui %253, %c4_i32 : i32
          %256 = arith.muli %255, %c1073741824_i32 : i32
          %257 = arith.addi %254, %256 : i32
          %258 = arith.divui %253, %c4_i32 : i32
          %259 = ascvec.shr %dst_18, 14, %mask : <ui32> -> <ui32>
          %260 = ascvec.shl %dst_20, 18, %mask : <ui32> -> <ui32>
          %261 = ascvec.or %259, %260, %mask : <ui32>, <ui32> -> <ui32>
          %262 = ascvec.shr %dst_20, 14, %mask : <ui32> -> <ui32>
          %263 = ascvec.duplicate_scalar %257, %mask : i32 -> <ui32>
          %264 = ascvec.duplicate_scalar %258, %mask : i32 -> <ui32>
          %carry_21, %dst_22 = ascvec.addc %263, %261, %mask : <ui32>, <ui32> -> !ascvec.mask, <ui32>
          %carry_23, %dst_24 = ascvec.addc %264, %262, %carry_21, %mask : <ui32>, <ui32> -> !ascvec.mask, <ui32>
          %265 = ascvec.duplicate_scalar %c16383_i32, %mask : i32 -> <ui32>
          %266 = ascvec.and %dst_18, %265, %mask : <ui32>, <ui32> -> <ui32>
          %267 = ascvec.duplicate_scalar %c-766435501_i32, %mask : i32 -> <ui32>
          %268 = ascvec.duplicate_scalar %c-845247145_i32, %mask : i32 -> <ui32>
          %dst0_25, %dst1_26 = ascvec.mull %224, %267, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %dst0_27, %dst1_28 = ascvec.mull %dst_22, %268, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %269 = ascvec.xor %dst1_28, %266, %mask : <ui32>, <ui32> -> <ui32>
          %270 = ascvec.xor %dst1_26, %dst_24, %mask : <ui32>, <ui32> -> <ui32>
          %271 = ascvec.duplicate_scalar %arg13, %mask : i32 -> <ui32>
          %272 = ascvec.duplicate_scalar %arg14, %mask : i32 -> <ui32>
          %273 = ascvec.xor %269, %271, %mask : <ui32>, <ui32> -> <ui32>
          %274 = ascvec.xor %270, %272, %mask : <ui32>, <ui32> -> <ui32>
          %dst0_29, %dst1_30 = ascvec.mull %273, %267, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %dst0_31, %dst1_32 = ascvec.mull %274, %268, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %275 = ascvec.xor %dst1_32, %dst0_27, %mask : <ui32>, <ui32> -> <ui32>
          %276 = ascvec.xor %dst1_30, %dst0_25, %mask : <ui32>, <ui32> -> <ui32>
          %277 = arith.addi %arg13, %c-1640531527_i32 : i32
          %278 = ascvec.duplicate_scalar %277, %mask : i32 -> <ui32>
          %279 = arith.addi %arg14, %c-1150833019_i32 : i32
          %280 = ascvec.duplicate_scalar %279, %mask : i32 -> <ui32>
          %281 = ascvec.xor %275, %278, %mask : <ui32>, <ui32> -> <ui32>
          %282 = ascvec.xor %276, %280, %mask : <ui32>, <ui32> -> <ui32>
          %dst0_33, %dst1_34 = ascvec.mull %281, %267, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %dst0_35, %dst1_36 = ascvec.mull %282, %268, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %283 = ascvec.xor %dst1_36, %dst0_31, %mask : <ui32>, <ui32> -> <ui32>
          %284 = ascvec.xor %dst1_34, %dst0_29, %mask : <ui32>, <ui32> -> <ui32>
          %285 = arith.addi %arg13, %c1013904242_i32 : i32
          %286 = ascvec.duplicate_scalar %285, %mask : i32 -> <ui32>
          %287 = arith.addi %arg14, %c1993301258_i32 : i32
          %288 = ascvec.duplicate_scalar %287, %mask : i32 -> <ui32>
          %289 = ascvec.xor %283, %286, %mask : <ui32>, <ui32> -> <ui32>
          %290 = ascvec.xor %284, %288, %mask : <ui32>, <ui32> -> <ui32>
          %dst0_37, %dst1_38 = ascvec.mull %289, %267, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %dst0_39, %dst1_40 = ascvec.mull %290, %268, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %291 = ascvec.xor %dst1_40, %dst0_35, %mask : <ui32>, <ui32> -> <ui32>
          %292 = ascvec.xor %dst1_38, %dst0_33, %mask : <ui32>, <ui32> -> <ui32>
          %293 = arith.addi %arg13, %c-626627285_i32 : i32
          %294 = ascvec.duplicate_scalar %293, %mask : i32 -> <ui32>
          %295 = arith.addi %arg14, %c842468239_i32 : i32
          %296 = ascvec.duplicate_scalar %295, %mask : i32 -> <ui32>
          %297 = ascvec.xor %291, %294, %mask : <ui32>, <ui32> -> <ui32>
          %298 = ascvec.xor %292, %296, %mask : <ui32>, <ui32> -> <ui32>
          %dst0_41, %dst1_42 = ascvec.mull %297, %267, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %dst0_43, %dst1_44 = ascvec.mull %298, %268, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %299 = ascvec.xor %dst1_44, %dst0_39, %mask : <ui32>, <ui32> -> <ui32>
          %300 = ascvec.xor %dst1_42, %dst0_37, %mask : <ui32>, <ui32> -> <ui32>
          %301 = arith.addi %arg13, %c2027808484_i32 : i32
          %302 = ascvec.duplicate_scalar %301, %mask : i32 -> <ui32>
          %303 = arith.addi %arg14, %c-308364780_i32 : i32
          %304 = ascvec.duplicate_scalar %303, %mask : i32 -> <ui32>
          %305 = ascvec.xor %299, %302, %mask : <ui32>, <ui32> -> <ui32>
          %306 = ascvec.xor %300, %304, %mask : <ui32>, <ui32> -> <ui32>
          %dst0_45, %dst1_46 = ascvec.mull %305, %267, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %dst0_47, %dst1_48 = ascvec.mull %306, %268, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %307 = ascvec.xor %dst1_48, %dst0_43, %mask : <ui32>, <ui32> -> <ui32>
          %308 = ascvec.xor %dst1_46, %dst0_41, %mask : <ui32>, <ui32> -> <ui32>
          %309 = arith.addi %arg13, %c387276957_i32 : i32
          %310 = ascvec.duplicate_scalar %309, %mask : i32 -> <ui32>
          %311 = arith.addi %arg14, %c-1459197799_i32 : i32
          %312 = ascvec.duplicate_scalar %311, %mask : i32 -> <ui32>
          %313 = ascvec.xor %307, %310, %mask : <ui32>, <ui32> -> <ui32>
          %314 = ascvec.xor %308, %312, %mask : <ui32>, <ui32> -> <ui32>
          %dst0_49, %dst1_50 = ascvec.mull %313, %267, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %dst0_51, %dst1_52 = ascvec.mull %314, %268, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %315 = ascvec.xor %dst1_52, %dst0_47, %mask : <ui32>, <ui32> -> <ui32>
          %316 = ascvec.xor %dst1_50, %dst0_45, %mask : <ui32>, <ui32> -> <ui32>
          %317 = arith.addi %arg13, %c-1253254570_i32 : i32
          %318 = ascvec.duplicate_scalar %317, %mask : i32 -> <ui32>
          %319 = arith.addi %arg14, %c1684936478_i32 : i32
          %320 = ascvec.duplicate_scalar %319, %mask : i32 -> <ui32>
          %321 = ascvec.xor %315, %318, %mask : <ui32>, <ui32> -> <ui32>
          %322 = ascvec.xor %316, %320, %mask : <ui32>, <ui32> -> <ui32>
          %dst0_53, %dst1_54 = ascvec.mull %321, %267, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %dst0_55, %dst1_56 = ascvec.mull %322, %268, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %323 = ascvec.xor %dst1_56, %dst0_51, %mask : <ui32>, <ui32> -> <ui32>
          %324 = ascvec.xor %dst1_54, %dst0_49, %mask : <ui32>, <ui32> -> <ui32>
          %325 = arith.addi %arg13, %c1401181199_i32 : i32
          %326 = ascvec.duplicate_scalar %325, %mask : i32 -> <ui32>
          %327 = arith.addi %arg14, %c534103459_i32 : i32
          %328 = ascvec.duplicate_scalar %327, %mask : i32 -> <ui32>
          %329 = ascvec.xor %323, %326, %mask : <ui32>, <ui32> -> <ui32>
          %330 = ascvec.xor %324, %328, %mask : <ui32>, <ui32> -> <ui32>
          %dst0_57, %dst1_58 = ascvec.mull %329, %267, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %dst0_59, %dst1_60 = ascvec.mull %330, %268, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %331 = ascvec.xor %dst1_60, %dst0_55, %mask : <ui32>, <ui32> -> <ui32>
          %332 = ascvec.xor %dst1_58, %dst0_53, %mask : <ui32>, <ui32> -> <ui32>
          %333 = arith.addi %arg13, %c-239350328_i32 : i32
          %334 = ascvec.duplicate_scalar %333, %mask : i32 -> <ui32>
          %335 = arith.addi %arg14, %c-616729560_i32 : i32
          %336 = ascvec.duplicate_scalar %335, %mask : i32 -> <ui32>
          %337 = ascvec.xor %331, %334, %mask : <ui32>, <ui32> -> <ui32>
          %338 = ascvec.xor %332, %336, %mask : <ui32>, <ui32> -> <ui32>
          %dst0_61, %dst1_62 = ascvec.mull %337, %267, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %dst0_63, %dst1_64 = ascvec.mull %338, %268, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %339 = ascvec.xor %dst1_64, %dst0_59, %mask : <ui32>, <ui32> -> <ui32>
          %340 = ascvec.xor %dst1_62, %dst0_57, %mask : <ui32>, <ui32> -> <ui32>
          %341 = arith.addi %arg13, %c-1879881855_i32 : i32
          %342 = ascvec.duplicate_scalar %341, %mask : i32 -> <ui32>
          %343 = arith.addi %arg14, %c-1767562579_i32 : i32
          %344 = ascvec.duplicate_scalar %343, %mask : i32 -> <ui32>
          %345 = ascvec.xor %339, %342, %mask : <ui32>, <ui32> -> <ui32>
          %346 = ascvec.xor %340, %344, %mask : <ui32>, <ui32> -> <ui32>
          %347 = arith.muli %arg17, %c128_i32 : i32
          %348 = ascvec.reinterpret %345 : <ui32> -> <i32>
          %349 = arith.index_cast %347 : i32 to index
          ascvec.storealign %arg15[%349], %348, %mask : memref<4096xi32, 1>, index, <i32>
          %350 = arith.addi %347, %c32_i32_14 : i32
          %351 = ascvec.reinterpret %dst0_63 : <ui32> -> <i32>
          %352 = arith.index_cast %350 : i32 to index
          ascvec.storealign %arg15[%352], %351, %mask : memref<4096xi32, 1>, index, <i32>
          %353 = arith.addi %347, %c64_i32_10 : i32
          %354 = ascvec.reinterpret %346 : <ui32> -> <i32>
          %355 = arith.index_cast %353 : i32 to index
          ascvec.storealign %arg15[%355], %354, %mask : memref<4096xi32, 1>, index, <i32>
          %356 = arith.addi %347, %c96_i32 : i32
          %357 = ascvec.reinterpret %dst0_61 : <ui32> -> <i32>
          %358 = arith.index_cast %356 : i32 to index
          ascvec.storealign %arg15[%358], %357, %mask : memref<4096xi32, 1>, index, <i32>
        }
        ascvec.mem_bar vst_vld
      } {cannir.vf_mode = "simd"}
      %113 = arith.divsi %111, %c32_i32 : i32
      %114 = arith.divsi %111, %c64_i32 : i32
      ascvec.get_buf v, %42, 0 {cannir.buffer.generated_sync}
      ascvec.get_buf v, %78, 0 {cannir.buffer.generated_sync}
      ascvec.get_buf v, %83, 0 {cannir.buffer.generated_sync}
      %115 = ascscalar.bitcast_memref %82 byte_offset = %c0_i64 : memref<1024xui8, 1> -> memref<?xui16, 1>
      ascvec.vf(%arg4, %73, %alloc_8, %alloc_9, %77, %113, %115, %114 : i64, memref<16384xbf16, 1>, memref<512xui32, 1>, memref<4096xi32, 1>, memref<16384xui8, 1>, i32, memref<?xui16, 1>, i32) {
      ^bb0(%arg9: i64, %arg10: memref<16384xbf16, 1>, %arg11: memref<512xui32, 1>, %arg12: memref<4096xi32, 1>, %arg13: memref<16384xui8, 1>, %arg14: i32, %arg15: memref<?xui16, 1>, %arg16: i32):
        %c64_i32_10 = arith.constant 64 : i32
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
        %c16_i32_11 = arith.constant 16 : i32
        %c2147483647_i32 = arith.constant 2147483647 : i32
        %c2_i32 = arith.constant 2 : i32
        %c0_i32_12 = arith.constant 0 : i32
        %c32_i32_13 = arith.constant 32 : i32
        %mask, %remaining_out = ascvec.update_mask %c32_i32_13 : !ascvec.mask
        scf.for %arg17 = %c0_i32_12 to %arg14 step %c1_i32  : i32 {
          %197 = ascvec.create_mask : !ascvec.mask
          %198 = ascvec.arange %c0_i32_12 : i32 -> <ui32>
          %199 = ascvec.duplicate_scalar %c0_i32_12, %197 : i32 -> <ui32>
          %200 = scf.for %arg18 = %c0_i32_12 to %c16_i32_11 step %c1_i32 iter_args(%arg19 = %199) -> (!ascvec.vreg<ui32>)  : i32 {
            %218 = arith.muli %arg17, %c32_i32_13 : i32
            %219 = arith.muli %arg18, %c2_i32 overflow<nsw> : i32
            %220 = arith.addi %218, %219 : i32
            %221 = arith.extsi %220 : i32 to i64
            %222 = arith.muli %221, %arg9 : i64
            %223 = arith.index_cast %222 : i64 to index
            %224 = ascvec.loadalign_unpack %arg10[%223] {mode = "b16_to_b32"} : memref<16384xbf16, 1>, index -> <bf16>
            %225 = ascvec.cast %224, %197 {rounding = 6 : i32} : <bf16> -> <f32>
            %226 = ascvec.reinterpret %225 : <f32> -> <ui32>
            %227 = ascvec.duplicate_scalar %c2147483647_i32, %197 : i32 -> <ui32>
            %228 = ascvec.and %226, %227, %197 : <ui32>, <ui32> -> <ui32>
            %229 = ascvec.max %arg19, %228, %197 : <ui32>
            scf.yield %229 : !ascvec.vreg<ui32>
          }
          %201 = ascvec.duplicate_scalar %c32_i32_13, %197 : i32 -> <ui32>
          %202 = ascvec.xor %198, %201, %197 : <ui32>, <ui32> -> <ui32>
          %203 = ascvec.gather_reg %200, %202 : <ui32>, <ui32> -> <ui32>
          %204 = ascvec.max %200, %203, %197 : <ui32>
          %205 = ascvec.shr %204, 23, %mask : <ui32> -> <ui32>
          %206 = ascvec.cmp_ge_scalar %205, %c255_i32, %mask : <ui32>, i32 -> !ascvec.mask
          %207 = ascvec.max_scalar %205, %c8_i32, %mask : <ui32>, i32
          %208 = ascvec.add_scalar %207, %c-8_i32, %mask : <ui32>, i32
          %209 = ascvec.duplicate_scalar %c255_i32, %206 merge %208 : !ascvec.vreg<ui32> {mode = "merging"} : i32 -> <ui32>
          %210 = ascvec.duplicate_scalar %c254_i32, %mask : i32 -> <ui32>
          %211 = ascvec.sub %210, %209, %mask : <ui32>
          %212 = ascvec.cmp_ne_scalar %209, %c0_i32_12, %mask : <ui32>, i32 -> !ascvec.mask
          %213 = ascvec.shl %211, 23, %212 : <ui32> -> <ui32>
          %214 = ascvec.duplicate_scalar %c2139160576_i32, %206 merge %213 : !ascvec.vreg<ui32> {mode = "merging"} : i32 -> <ui32>
          %215 = ascvec.reinterpret %214 : <ui32> -> <f32>
          %216 = arith.muli %arg17, %c32_i32_13 : i32
          %217 = arith.index_cast %216 : i32 to index
          ascvec.storealign %arg11[%217], %209, %mask : memref<512xui32, 1>, index, <ui32>
          scf.for %arg18 = %c0_i32_12 to %c8_i32 step %c1_i32  : i32 {
            %218 = arith.muli %arg17, %c256_i32 : i32
            %219 = arith.muli %arg18, %c32_i32_13 overflow<nsw> : i32
            %220 = arith.addi %218, %219 : i32
            %221 = arith.index_cast %220 : i32 to index
            %222 = ascvec.loadalign %arg12[%221] : memref<4096xi32, 1>, index -> <i32>
            %223 = ascvec.reinterpret %222 : <i32> -> <ui32>
            %224 = ascvec.shl %223, 8, %mask : <ui32> -> <ui32>
            %225 = ascvec.shr %223, 24, %mask : <ui32> -> <ui32>
            %226 = ascvec.or %224, %225, %mask : <ui32>, <ui32> -> <ui32>
            %227 = arith.muli %arg18, %c4_i32 overflow<nsw> : i32
            %228 = arith.addi %216, %227 : i32
            %229 = arith.extsi %228 : i32 to i64
            %230 = arith.muli %229, %arg9 : i64
            %231 = ascvec.shr %223, 16, %mask : <ui32> -> <ui32>
            %232 = arith.index_cast %230 : i64 to index
            %233 = ascvec.loadalign_unpack %arg10[%232] {mode = "b16_to_b32"} : memref<16384xbf16, 1>, index -> <bf16>
            %234 = ascvec.cast %233, %mask {rounding = 6 : i32} : <bf16> -> <f32>
            %235 = ascvec.mul %234, %215, %mask : <f32>
            %236 = ascvec.cast %235, %mask : <f32> -> <bf16>
            %237 = ascvec.cast %236, %mask {rounding = 6 : i32} : <bf16> -> <f32>
            %238 = ascvec.reinterpret %237 : <f32> -> <ui32>
            %239 = ascvec.shr %238, 24, %mask : <ui32> -> <ui32>
            %240 = ascvec.duplicate_scalar %c128_i32, %mask : i32 -> <ui32>
            %241 = ascvec.and %239, %240, %mask : <ui32>, <ui32> -> <ui32>
            %242 = ascvec.duplicate_scalar %c2147483647_i32, %mask : i32 -> <ui32>
            %243 = ascvec.and %238, %242, %mask : <ui32>, <ui32> -> <ui32>
            %244 = ascvec.shr %243, 4, %mask : <ui32> -> <ui32>
            %245 = ascvec.duplicate_scalar %c62914560_i32, %mask : i32 -> <ui32>
            %246 = ascvec.sub %244, %245, %mask : <ui32>
            %247 = ascvec.reinterpret %243 : <ui32> -> <f32>
            %248 = ascvec.mul_scalar %247, %cst, %mask : <f32>, f32
            %249 = ascvec.cast %248, %mask {rounding = 4 : i32} : <f32> -> <i32>
            %250 = ascvec.reinterpret %249 : <i32> -> <ui32>
            %251 = ascvec.cmp_ge_scalar %243, %c1015021568_i32, %mask : <ui32>, i32 -> !ascvec.mask
            %252 = ascvec.select %246, %250, %251 : <ui32>
            %253 = ascvec.duplicate_scalar %c65535_i32, %mask : i32 -> <ui32>
            %254 = ascvec.and %231, %253, %mask : <ui32>, <ui32> -> <ui32>
            %255 = ascvec.add %252, %254, %mask : <ui32>
            %256 = ascvec.shr %255, 16, %mask : <ui32> -> <ui32>
            %257 = ascvec.min_scalar %256, %c126_i32, %mask : <ui32>, i32
            %258 = ascvec.or %257, %241, %mask : <ui32>, <ui32> -> <ui32>
            %259 = ascvec.duplicate_scalar %c127_i32, %mask : i32 -> <ui32>
            %260 = ascvec.cmp_gt_scalar %243, %c2139095040_i32, %mask : <ui32>, i32 -> !ascvec.mask
            %261 = ascvec.select %259, %258, %260 : <ui32>
            ascvec.storealign_pack %arg13[%232], %261, %mask {mode = "b32_to_b8"} : memref<16384xui8, 1>, index, <ui32>
            %262 = arith.addi %228, %c1_i32 : i32
            %263 = arith.extsi %262 : i32 to i64
            %264 = arith.muli %263, %arg9 : i64
            %265 = ascvec.and %226, %253, %mask : <ui32>, <ui32> -> <ui32>
            %266 = arith.index_cast %264 : i64 to index
            %267 = ascvec.loadalign_unpack %arg10[%266] {mode = "b16_to_b32"} : memref<16384xbf16, 1>, index -> <bf16>
            %268 = ascvec.cast %267, %mask {rounding = 6 : i32} : <bf16> -> <f32>
            %269 = ascvec.mul %268, %215, %mask : <f32>
            %270 = ascvec.cast %269, %mask : <f32> -> <bf16>
            %271 = ascvec.cast %270, %mask {rounding = 6 : i32} : <bf16> -> <f32>
            %272 = ascvec.reinterpret %271 : <f32> -> <ui32>
            %273 = ascvec.shr %272, 24, %mask : <ui32> -> <ui32>
            %274 = ascvec.and %273, %240, %mask : <ui32>, <ui32> -> <ui32>
            %275 = ascvec.and %272, %242, %mask : <ui32>, <ui32> -> <ui32>
            %276 = ascvec.shr %275, 4, %mask : <ui32> -> <ui32>
            %277 = ascvec.sub %276, %245, %mask : <ui32>
            %278 = ascvec.reinterpret %275 : <ui32> -> <f32>
            %279 = ascvec.mul_scalar %278, %cst, %mask : <f32>, f32
            %280 = ascvec.cast %279, %mask {rounding = 4 : i32} : <f32> -> <i32>
            %281 = ascvec.reinterpret %280 : <i32> -> <ui32>
            %282 = ascvec.cmp_ge_scalar %275, %c1015021568_i32, %mask : <ui32>, i32 -> !ascvec.mask
            %283 = ascvec.select %277, %281, %282 : <ui32>
            %284 = ascvec.and %265, %253, %mask : <ui32>, <ui32> -> <ui32>
            %285 = ascvec.add %283, %284, %mask : <ui32>
            %286 = ascvec.shr %285, 16, %mask : <ui32> -> <ui32>
            %287 = ascvec.min_scalar %286, %c126_i32, %mask : <ui32>, i32
            %288 = ascvec.or %287, %274, %mask : <ui32>, <ui32> -> <ui32>
            %289 = ascvec.cmp_gt_scalar %275, %c2139095040_i32, %mask : <ui32>, i32 -> !ascvec.mask
            %290 = ascvec.select %259, %288, %289 : <ui32>
            ascvec.storealign_pack %arg13[%266], %290, %mask {mode = "b32_to_b8"} : memref<16384xui8, 1>, index, <ui32>
            %291 = arith.addi %228, %c2_i32 : i32
            %292 = arith.extsi %291 : i32 to i64
            %293 = arith.muli %292, %arg9 : i64
            %294 = ascvec.and %223, %253, %mask : <ui32>, <ui32> -> <ui32>
            %295 = arith.index_cast %293 : i64 to index
            %296 = ascvec.loadalign_unpack %arg10[%295] {mode = "b16_to_b32"} : memref<16384xbf16, 1>, index -> <bf16>
            %297 = ascvec.cast %296, %mask {rounding = 6 : i32} : <bf16> -> <f32>
            %298 = ascvec.mul %297, %215, %mask : <f32>
            %299 = ascvec.cast %298, %mask : <f32> -> <bf16>
            %300 = ascvec.cast %299, %mask {rounding = 6 : i32} : <bf16> -> <f32>
            %301 = ascvec.reinterpret %300 : <f32> -> <ui32>
            %302 = ascvec.shr %301, 24, %mask : <ui32> -> <ui32>
            %303 = ascvec.and %302, %240, %mask : <ui32>, <ui32> -> <ui32>
            %304 = ascvec.and %301, %242, %mask : <ui32>, <ui32> -> <ui32>
            %305 = ascvec.shr %304, 4, %mask : <ui32> -> <ui32>
            %306 = ascvec.sub %305, %245, %mask : <ui32>
            %307 = ascvec.reinterpret %304 : <ui32> -> <f32>
            %308 = ascvec.mul_scalar %307, %cst, %mask : <f32>, f32
            %309 = ascvec.cast %308, %mask {rounding = 4 : i32} : <f32> -> <i32>
            %310 = ascvec.reinterpret %309 : <i32> -> <ui32>
            %311 = ascvec.cmp_ge_scalar %304, %c1015021568_i32, %mask : <ui32>, i32 -> !ascvec.mask
            %312 = ascvec.select %306, %310, %311 : <ui32>
            %313 = ascvec.and %294, %253, %mask : <ui32>, <ui32> -> <ui32>
            %314 = ascvec.add %312, %313, %mask : <ui32>
            %315 = ascvec.shr %314, 16, %mask : <ui32> -> <ui32>
            %316 = ascvec.min_scalar %315, %c126_i32, %mask : <ui32>, i32
            %317 = ascvec.or %316, %303, %mask : <ui32>, <ui32> -> <ui32>
            %318 = ascvec.cmp_gt_scalar %304, %c2139095040_i32, %mask : <ui32>, i32 -> !ascvec.mask
            %319 = ascvec.select %259, %317, %318 : <ui32>
            ascvec.storealign_pack %arg13[%295], %319, %mask {mode = "b32_to_b8"} : memref<16384xui8, 1>, index, <ui32>
            %320 = arith.addi %228, %c3_i32 : i32
            %321 = arith.extsi %320 : i32 to i64
            %322 = arith.muli %321, %arg9 : i64
            %323 = ascvec.shr %226, 16, %mask : <ui32> -> <ui32>
            %324 = arith.index_cast %322 : i64 to index
            %325 = ascvec.loadalign_unpack %arg10[%324] {mode = "b16_to_b32"} : memref<16384xbf16, 1>, index -> <bf16>
            %326 = ascvec.cast %325, %mask {rounding = 6 : i32} : <bf16> -> <f32>
            %327 = ascvec.mul %326, %215, %mask : <f32>
            %328 = ascvec.cast %327, %mask : <f32> -> <bf16>
            %329 = ascvec.cast %328, %mask {rounding = 6 : i32} : <bf16> -> <f32>
            %330 = ascvec.reinterpret %329 : <f32> -> <ui32>
            %331 = ascvec.shr %330, 24, %mask : <ui32> -> <ui32>
            %332 = ascvec.and %331, %240, %mask : <ui32>, <ui32> -> <ui32>
            %333 = ascvec.and %330, %242, %mask : <ui32>, <ui32> -> <ui32>
            %334 = ascvec.shr %333, 4, %mask : <ui32> -> <ui32>
            %335 = ascvec.sub %334, %245, %mask : <ui32>
            %336 = ascvec.reinterpret %333 : <ui32> -> <f32>
            %337 = ascvec.mul_scalar %336, %cst, %mask : <f32>, f32
            %338 = ascvec.cast %337, %mask {rounding = 4 : i32} : <f32> -> <i32>
            %339 = ascvec.reinterpret %338 : <i32> -> <ui32>
            %340 = ascvec.cmp_ge_scalar %333, %c1015021568_i32, %mask : <ui32>, i32 -> !ascvec.mask
            %341 = ascvec.select %335, %339, %340 : <ui32>
            %342 = ascvec.and %323, %253, %mask : <ui32>, <ui32> -> <ui32>
            %343 = ascvec.add %341, %342, %mask : <ui32>
            %344 = ascvec.shr %343, 16, %mask : <ui32> -> <ui32>
            %345 = ascvec.min_scalar %344, %c126_i32, %mask : <ui32>, i32
            %346 = ascvec.or %345, %332, %mask : <ui32>, <ui32> -> <ui32>
            %347 = ascvec.cmp_gt_scalar %333, %c2139095040_i32, %mask : <ui32>, i32 -> !ascvec.mask
            %348 = ascvec.select %259, %346, %347 : <ui32>
            ascvec.storealign_pack %arg13[%324], %348, %mask {mode = "b32_to_b8"} : memref<16384xui8, 1>, index, <ui32>
          }
        }
        ascvec.mem_bar vst_vld
        %mask_14, %remaining_out_15 = ascvec.update_mask %c32_i32_13 elem_bits 16 : !ascvec.mask
        scf.for %arg17 = %c0_i32_12 to %arg16 step %c1_i32  : i32 {
          %197 = arith.muli %arg17, %c64_i32_10 : i32
          %198 = arith.index_cast %197 : i32 to index
          %199 = ascvec.loadalign %arg11[%198] : memref<512xui32, 1>, index -> <ui32>
          %200 = arith.addi %197, %c32_i32_13 : i32
          %201 = arith.index_cast %200 : i32 to index
          %202 = ascvec.loadalign %arg11[%201] : memref<512xui32, 1>, index -> <ui32>
          %203 = ascvec.shl %202, 8, %mask : <ui32> -> <ui32>
          %204 = ascvec.or %199, %203, %mask : <ui32>, <ui32> -> <ui32>
          %205 = ascvec.pack %204 src_elem_bits 32 dst_elem_bits 16 {part = "low"} : <ui32> -> <ui16>
          %206 = arith.extsi %arg17 : i32 to i64
          %207 = arith.muli %206, %arg9 : i64
          %208 = arith.index_cast %207 : i64 to index
          ascvec.storealign %arg15[%208], %205, %mask_14 : memref<?xui16, 1>, index, <ui16>
        }
        ascvec.mem_bar vst_vld
      } {cannir.vf_mode = "simd"}
      ascvec.release_buf v, %83, 0 {cannir.buffer.generated_sync}
      ascvec.release_buf v, %78, 0 {cannir.buffer.generated_sync}
      ascvec.release_buf v, %42, 0 {cannir.buffer.generated_sync}
      %116 = cannir.extract_layout %arg1 : !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
      %117 = cannir.extract_shape %116 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, -1)>
      %118 = cannir.extract_dim %117[0] : !cannir.shape<(-1, -1)>
      %119 = arith.minsi %33, %118 : i64
      %120 = arith.subi %118, %119 : i64
      %121 = arith.minsi %120, %c1_i64 : i64
      %122 = arith.cmpi sle, %33, %118 : i64
      %123 = arith.select %122, %121, %c0_i64 : i64
      %124 = arith.cmpi sgt, %123, %c1_i64 : i64
      %125 = arith.select %124, %c1_i64, %123 : i64
      %126 = cannir.extract_dim %117[1] : !cannir.shape<(-1, -1)>
      %127 = arith.divsi %126, %55 : i64
      %128 = arith.minsi %35, %127 : i64
      %129 = arith.muli %128, %36 : i64
      %130 = arith.subi %126, %129 : i64
      %131 = arith.minsi %36, %130 : i64
      %132 = arith.cmpi sle, %35, %127 : i64
      %133 = arith.select %132, %131, %c0_i64 : i64
      %134 = arith.cmpi sgt, %133, %c16384_i64 : i64
      %135 = arith.select %134, %c16384_i64, %133 : i64
      %136 = cannir.extract_physical_layout %arg1 : !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
      %137 = cannir.extract_stride %136 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
      %138 = cannir.extract_dim %137[0] : !cannir.stride<(-1, 1)>
      %139 = arith.muli %33, %138 : i64
      %140 = arith.addi %139, %65 : i64
      %141 = arith.muli %140, %c8_i64 : i64
      %142 = ascscalar.memref_address %alloc_0 : memref<1xui8>
      %143 = arith.divui %141, %c8_i64 : i64
      %144 = arith.addi %142, %143 : i64
      %145 = ascscalar.bind_gm_ptr %144 : memref<1xui8>
      %146 = ascscalar.bind_local_ptr %76 : memref<1xui8, 1>
      %147 = cannir.extract_stride %116 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
      %148 = cannir.extract_dim %147[0] : !cannir.stride<(-1, 1)>
      %149 = arith.cmpi ugt, %125, %c0_i64 : i64
      %150 = arith.cmpi ugt, %135, %c0_i64 : i64
      %151 = arith.andi %149, %150 : i1
      scf.if %151 {
        ascvec.get_buf mte3, %78, 0 {cannir.buffer.generated_sync}
        %197 = arith.trunci %125 : i64 to i32
        %198 = arith.trunci %135 : i64 to i32
        ascvec.copy_ub2gm %145[%c0], %146[%c0], %197, %198, %148, %c16384_i32 l2_cache_ctl %c0_i32 : memref<1xui8>, memref<1xui8, 1>
        ascvec.release_buf mte3, %78, 0 {cannir.buffer.generated_sync}
      }
      %152 = arith.remsi %36, %c32_i64 : i64
      %153 = arith.cmpi ne, %152, %c0_i64 : i64
      %154 = arith.cmpi slt, %152, %c0_i64 : i64
      %155 = arith.andi %153, %154 : i1
      %156 = arith.divsi %36, %c32_i64 : i64
      %157 = arith.extui %155 : i1 to i64
      %158 = arith.subi %156, %157 : i64
      %159 = cannir.extract_layout %arg2 : !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
      %160 = cannir.extract_shape %159 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, -1)>
      %161 = cannir.extract_dim %160[0] : !cannir.shape<(-1, -1)>
      %162 = arith.minsi %33, %161 : i64
      %163 = arith.subi %161, %162 : i64
      %164 = arith.minsi %163, %c1_i64 : i64
      %165 = arith.cmpi sle, %33, %161 : i64
      %166 = arith.select %165, %164, %c0_i64 : i64
      %167 = arith.cmpi sgt, %166, %c1_i64 : i64
      %168 = arith.select %167, %c1_i64, %166 : i64
      %169 = cannir.extract_dim %160[1] : !cannir.shape<(-1, -1)>
      %170 = arith.maxsi %158, %c1_i64 : i64
      %171 = arith.divsi %169, %170 : i64
      %172 = arith.minsi %35, %171 : i64
      %173 = arith.muli %172, %158 : i64
      %174 = arith.subi %169, %173 : i64
      %175 = arith.minsi %158, %174 : i64
      %176 = arith.cmpi sle, %35, %171 : i64
      %177 = arith.select %176, %175, %c0_i64 : i64
      %178 = arith.cmpi sgt, %177, %c1024_i64 : i64
      %179 = arith.select %178, %c1024_i64, %177 : i64
      %180 = cannir.extract_physical_layout %arg2 : !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
      %181 = cannir.extract_stride %180 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
      %182 = cannir.extract_dim %181[0] : !cannir.stride<(-1, 1)>
      %183 = arith.muli %33, %182 : i64
      %184 = arith.muli %35, %158 : i64
      %185 = arith.addi %183, %184 : i64
      %186 = arith.muli %185, %c8_i64 : i64
      %187 = ascscalar.memref_address %alloc_1 : memref<1xui8>
      %188 = arith.divui %186, %c8_i64 : i64
      %189 = arith.addi %187, %188 : i64
      %190 = ascscalar.bind_gm_ptr %189 : memref<1xui8>
      %191 = ascscalar.bind_local_ptr %81 : memref<1xui8, 1>
      %192 = cannir.extract_stride %159 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
      %193 = cannir.extract_dim %192[0] : !cannir.stride<(-1, 1)>
      %194 = arith.cmpi ugt, %168, %c0_i64 : i64
      %195 = arith.cmpi ugt, %179, %c0_i64 : i64
      %196 = arith.andi %194, %195 : i1
      scf.if %196 {
        ascvec.get_buf mte3, %83, 0 {cannir.buffer.generated_sync}
        %197 = arith.trunci %168 : i64 to i32
        %198 = arith.trunci %179 : i64 to i32
        ascvec.copy_ub2gm %190[%c0], %191[%c0], %197, %198, %193, %c1024_i32 l2_cache_ctl %c0_i32 : memref<1xui8>, memref<1xui8, 1>
        ascvec.release_buf mte3, %83, 0 {cannir.buffer.generated_sync}
      }
      scf.yield %44 : i64
    }
    return
  }
}
