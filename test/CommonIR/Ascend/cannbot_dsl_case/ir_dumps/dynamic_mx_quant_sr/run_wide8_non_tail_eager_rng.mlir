module {
  func.func @run_wide8_non_tail_eager_rng_f5ba2ec71f97bcfafcaa6a65cf78a511(%arg0: !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg1: !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg2: !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64) attributes {cannbotdsl.contract_digest = "f5ba2ec71f97bcfafcaa6a65cf78a511fcd68fe300166772dc14c9775db8348c", cannbotdsl.public_slot_count = 8 : i64, cannbotdsl.slot_dtype_bits = [16, 8, 8, 64, 64, 64, 64, 64], cannbotdsl.slot_dtype_codes = [4, 1, 1, 0, 0, 0, 0, 0], cannbotdsl.slot_dtype_lanes = [1, 1, 1, 1, 1, 1, 1, 1], cannbotdsl.slot_flags = [1, 1, 1, 1, 1, 1, 1, 1], cannbotdsl.slot_kinds = [1, 1, 1, 2, 2, 2, 2, 2], cannbotdsl.slot_type_digests = ["5d7cdc96dc21b3b739e32aeccf58c1df113bec54701c75107ed650f2544a07da", "171ddd09eb070fb4a6413ae87b2ceecddc233a9a173a0404e96285a4bb8b8659", "891bcd21e721f8d1a666bcca4e8705ec63796c1add8ae000536d0d98f0274d3a", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515"], cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.host} {
    cannir.kernel_launch @kernel__DynamicMxQuantSrKernel_run_wide8_non_tail_0(%arg5) [%arg0, %arg1, %arg2, %arg3, %arg4, %arg6, %arg7] {cannbotdsl.profile = {indices = array<i64: 0, 1, 2>, name = "_DynamicMxQuantSrKernel_run_wide8_non_tail", op_type = "_DynamicMxQuantSrKernel_run_wide8_non_tail", roles = array<i64: 0, 1, 1>}} : !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, i64, i64, i64, i64
    return
  }
  func.func @kernel__DynamicMxQuantSrKernel_run_wide8_non_tail_0(%arg0: !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg1: !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg2: !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64) attributes {cannir.buffer.auto_sync, cannir.buffer.sync_analyzed, cannir.buffer.sync_verified, cannir.buffer.tensor_state, cannir.core = "aiv", cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.device, cannir.explicit_ring_buffer_cursors, cannir.kernel_group = "kernel__DynamicMxQuantSrKernel_run_wide8_non_tail_0", cannir.resource.ids_prepared} {
    %c1024_i32 = arith.constant 1024 : i32
    %c16384_i32 = arith.constant 16384 : i32
    %c256_i64 = arith.constant 256 : i64
    %c0_i32 = arith.constant 0 : i32
    %c32768_i64 = arith.constant 32768 : i64
    %c32_i64 = arith.constant 32 : i64
    %c4294967296_i64 = arith.constant 4294967296 : i64
    %c0 = arith.constant 0 : index
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
    %0 = arith.remsi %arg3, %c256_i64 : i64
    %1 = arith.cmpi ne, %0, %c0_i64 : i64
    %2 = arith.cmpi slt, %0, %c0_i64 : i64
    %3 = arith.andi %1, %2 : i1
    %4 = arith.divsi %arg3, %c256_i64 : i64
    %5 = arith.extui %3 : i1 to i64
    %6 = arith.subi %4, %5 : i64
    %7 = cannir.extract_layout %arg0 : !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
    %8 = cannir.extract_shape %7 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, -1)>
    %9 = cannir.extract_dim %8[0] : !cannir.shape<(-1, -1)>
    %10 = arith.muli %9, %6 : i64
    %11 = ascscalar.get_block_idx
    %12 = ascscalar.get_block_num
    %13 = scf.for %arg7 = %11 to %10 step %12 iter_args(%arg8 = %c0_i64) -> (i64)  : i64 {
      %14 = arith.remsi %arg7, %6 : i64
      %15 = arith.cmpi ne, %14, %c0_i64 : i64
      %16 = arith.cmpi slt, %14, %c0_i64 : i64
      %17 = arith.cmpi slt, %6, %c0_i64 : i64
      %18 = arith.xori %16, %17 : i1
      %19 = arith.andi %15, %18 : i1
      %20 = arith.divsi %arg7, %6 : i64
      %21 = arith.extui %19 : i1 to i64
      %22 = arith.subi %20, %21 : i64
      %23 = arith.select %19, %6, %c0_i64 : i64
      %24 = arith.addi %14, %23 : i64
      %25 = arith.muli %arg4, %c256_i64 : i64
      %26 = arith.cmpi eq, %arg8, %c0_i64 : i64
      %27 = ascscalar.memref_address %alloc_2 : memref<16384xbf16, 1>
      %28 = ascscalar.memref_address %alloc_3 : memref<16384xbf16, 1>
      %29 = arith.select %26, %27, %28 : i64
      %30 = arith.cmpi ne, %arg8, %c0_i64 : i64
      %31 = arith.extui %30 : i1 to i64
      %32 = arith.addi %arg8, %c1_i64 overflow<nsw> : i64
      %33 = arith.remui %32, %c2_i64 : i64
      %34 = arith.minsi %22, %9 : i64
      %35 = arith.subi %9, %34 : i64
      %36 = arith.minsi %35, %c1_i64 : i64
      %37 = arith.cmpi sle, %22, %9 : i64
      %38 = arith.select %37, %36, %c0_i64 : i64
      %39 = cannir.extract_physical_layout %arg0 : !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
      %40 = cannir.extract_stride %39 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
      %41 = cannir.extract_dim %40[0] : !cannir.stride<(-1, 1)>
      %42 = arith.muli %41, %c2_i64 : i64
      %43 = cannir.extract_dim %8[1] : !cannir.shape<(-1, -1)>
      %44 = arith.maxsi %25, %c1_i64 : i64
      %45 = arith.divsi %43, %44 : i64
      %46 = arith.minsi %24, %45 : i64
      %47 = arith.muli %46, %25 : i64
      %48 = arith.subi %43, %47 : i64
      %49 = arith.minsi %25, %48 : i64
      %50 = arith.cmpi sle, %24, %45 : i64
      %51 = arith.select %50, %49, %c0_i64 : i64
      %52 = arith.muli %22, %41 : i64
      %53 = arith.muli %52, %c2_i64 : i64
      %54 = arith.muli %24, %25 : i64
      %55 = arith.muli %54, %c2_i64 : i64
      %56 = arith.addi %53, %55 : i64
      %57 = arith.muli %51, %c2_i64 : i64
      %58 = arith.cmpi eq, %57, %c0_i64 : i64
      %59 = arith.select %58, %c0_i64, %38 : i64
      %60 = arith.select %58, %c2_i64, %57 : i64
      %61 = ascscalar.bind_local_ptr %29 : memref<1xbf16, 1>
      ascvec.get_buf mte2, %31, 0 {cannir.buffer.generated_sync}
      "ascvec.copy_gm2ub_plan"(%61, %alloc, %c0_i32, %c0_i32, %c0_i64, %56, %59, %60, %42, %c32768_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
      ascvec.release_buf mte2, %31, 0 {cannir.buffer.generated_sync}
      %62 = ascscalar.bind_local_ptr %29 : memref<16384xbf16, 1>
      %63 = ascscalar.memref_address %alloc_4 : memref<16384xui8, 1>
      %64 = ascscalar.memref_address %alloc_5 : memref<16384xui8, 1>
      %65 = arith.select %26, %63, %64 : i64
      %66 = ascscalar.bind_local_ptr %65 : memref<16384xui8, 1>
      %67 = arith.select %26, %c2_i64, %c3_i64 : i64
      %68 = ascscalar.memref_address %alloc_6 : memref<1024xui8, 1>
      %69 = ascscalar.memref_address %alloc_7 : memref<1024xui8, 1>
      %70 = arith.select %26, %68, %69 : i64
      %71 = ascscalar.bind_local_ptr %70 : memref<1024xui8, 1>
      %72 = arith.select %26, %c4_i64, %c5_i64 : i64
      %73 = arith.muli %22, %arg4 : i64
      %74 = arith.muli %24, %c256_i64 : i64
      %75 = arith.remsi %74, %c32_i64 : i64
      %76 = arith.cmpi ne, %75, %c0_i64 : i64
      %77 = arith.cmpi slt, %75, %c0_i64 : i64
      %78 = arith.andi %76, %77 : i1
      %79 = arith.divsi %74, %c32_i64 : i64
      %80 = arith.extui %78 : i1 to i64
      %81 = arith.subi %79, %80 : i64
      %82 = arith.remsi %arg3, %c32_i64 : i64
      %83 = arith.cmpi ne, %82, %c0_i64 : i64
      %84 = arith.cmpi slt, %82, %c0_i64 : i64
      %85 = arith.andi %83, %84 : i1
      %86 = arith.divsi %arg3, %c32_i64 : i64
      %87 = arith.extui %85 : i1 to i64
      %88 = arith.subi %86, %87 : i64
      %89 = arith.trunci %arg5 : i64 to i32
      %90 = arith.extui %89 : i32 to i64
      %91 = arith.subi %arg5, %90 : i64
      %92 = arith.remsi %91, %c4294967296_i64 : i64
      %93 = arith.cmpi ne, %92, %c0_i64 : i64
      %94 = arith.cmpi slt, %92, %c0_i64 : i64
      %95 = arith.andi %93, %94 : i1
      %96 = arith.divsi %91, %c4294967296_i64 : i64
      %97 = arith.extui %95 : i1 to i64
      %98 = arith.subi %96, %97 : i64
      %99 = arith.trunci %98 : i64 to i32
      ascvec.vf(%81, %73, %88, %arg6, %89, %99, %alloc_9 : i64, i64, i64, i64, i32, i32, memref<4096xi32, 1>) {
      ^bb0(%arg9: i64, %arg10: i64, %arg11: i64, %arg12: i64, %arg13: i32, %arg14: i32, %arg15: memref<4096xi32, 1>):
        %c64_i32 = arith.constant 64 : i32
        %c1_i32 = arith.constant 1 : i32
        %c8_i32 = arith.constant 8 : i32
        %c48_i32 = arith.constant 48 : i32
        %c32_i32 = arith.constant 32 : i32
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
        %c0_i64_10 = arith.constant 0 : i64
        %c4294967296_i64_11 = arith.constant 4294967296 : i64
        %c0_i32_12 = arith.constant 0 : i32
        %c16_i32 = arith.constant 16 : i32
        %mask, %remaining_out = ascvec.update_mask %c16_i32 : !ascvec.mask
        %181 = ascvec.arange %c0_i32_12 : i32 -> <ui32>
        scf.for %arg16 = %c0_i32_12 to %c8_i32 step %c1_i32  : i32 {
          %182 = arith.extsi %arg16 : i32 to i64
          %183 = arith.addi %arg9, %182 : i64
          %184 = arith.trunci %arg10 : i64 to i32
          %185 = arith.extui %184 : i32 to i64
          %186 = arith.subi %arg10, %185 : i64
          %187 = arith.remsi %186, %c4294967296_i64_11 : i64
          %188 = arith.cmpi ne, %187, %c0_i64_10 : i64
          %189 = arith.cmpi slt, %187, %c0_i64_10 : i64
          %190 = arith.andi %188, %189 : i1
          %191 = arith.divsi %186, %c4294967296_i64_11 : i64
          %192 = arith.extui %190 : i1 to i64
          %193 = arith.subi %191, %192 : i64
          %194 = arith.trunci %193 : i64 to i32
          %195 = arith.trunci %183 : i64 to i32
          %196 = arith.extui %195 : i32 to i64
          %197 = arith.subi %183, %196 : i64
          %198 = arith.remsi %197, %c4294967296_i64_11 : i64
          %199 = arith.cmpi ne, %198, %c0_i64_10 : i64
          %200 = arith.cmpi slt, %198, %c0_i64_10 : i64
          %201 = arith.andi %199, %200 : i1
          %202 = arith.divsi %197, %c4294967296_i64_11 : i64
          %203 = arith.extui %201 : i1 to i64
          %204 = arith.subi %202, %203 : i64
          %205 = arith.trunci %204 : i64 to i32
          %206 = ascvec.duplicate_scalar %184, %mask : i32 -> <ui32>
          %207 = ascvec.duplicate_scalar %194, %mask : i32 -> <ui32>
          %208 = ascvec.duplicate_scalar %c0_i32_12, %mask : i32 -> <ui32>
          %carry, %dst = ascvec.addc %206, %181, %mask : <ui32>, <ui32> -> !ascvec.mask, <ui32>
          %carry_13, %dst_14 = ascvec.addc %207, %208, %carry, %mask : <ui32>, <ui32> -> !ascvec.mask, <ui32>
          %209 = ascvec.duplicate_scalar %195, %mask : i32 -> <ui32>
          %210 = ascvec.duplicate_scalar %205, %mask : i32 -> <ui32>
          %211 = arith.trunci %arg11 : i64 to i32
          %212 = arith.extui %211 : i32 to i64
          %213 = arith.subi %arg11, %212 : i64
          %214 = arith.remsi %213, %c4294967296_i64_11 : i64
          %215 = arith.cmpi ne, %214, %c0_i64_10 : i64
          %216 = arith.cmpi slt, %214, %c0_i64_10 : i64
          %217 = arith.andi %215, %216 : i1
          %218 = arith.divsi %213, %c4294967296_i64_11 : i64
          %219 = arith.extui %217 : i1 to i64
          %220 = arith.subi %218, %219 : i64
          %221 = arith.trunci %220 : i64 to i32
          %222 = ascvec.duplicate_scalar %211, %mask : i32 -> <ui32>
          %dst0, %dst1 = ascvec.mull %dst, %222, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %223 = ascvec.mul_scalar %dst, %221, %mask : <ui32>, i32
          %224 = ascvec.add %dst1, %223, %mask : <ui32>
          %225 = ascvec.mul_scalar %dst_14, %211, %mask : <ui32>, i32
          %226 = ascvec.add %224, %225, %mask : <ui32>
          %carry_15, %dst_16 = ascvec.addc %dst0, %209, %mask : <ui32>, <ui32> -> !ascvec.mask, <ui32>
          %carry_17, %dst_18 = ascvec.addc %226, %210, %carry_15, %mask : <ui32>, <ui32> -> !ascvec.mask, <ui32>
          %227 = arith.trunci %arg12 : i64 to i32
          %228 = arith.extui %227 : i32 to i64
          %229 = arith.subi %arg12, %228 : i64
          %230 = arith.remsi %229, %c4294967296_i64_11 : i64
          %231 = arith.cmpi ne, %230, %c0_i64_10 : i64
          %232 = arith.cmpi slt, %230, %c0_i64_10 : i64
          %233 = arith.andi %231, %232 : i1
          %234 = arith.divsi %229, %c4294967296_i64_11 : i64
          %235 = arith.extui %233 : i1 to i64
          %236 = arith.subi %234, %235 : i64
          %237 = arith.trunci %236 : i64 to i32
          %238 = arith.divui %227, %c4_i32 : i32
          %239 = arith.remui %237, %c4_i32 : i32
          %240 = arith.muli %239, %c1073741824_i32 : i32
          %241 = arith.addi %238, %240 : i32
          %242 = arith.divui %237, %c4_i32 : i32
          %243 = ascvec.shr %dst_16, 14, %mask : <ui32> -> <ui32>
          %244 = ascvec.shl %dst_18, 18, %mask : <ui32> -> <ui32>
          %245 = ascvec.or %243, %244, %mask : <ui32>, <ui32> -> <ui32>
          %246 = ascvec.shr %dst_18, 14, %mask : <ui32> -> <ui32>
          %247 = ascvec.duplicate_scalar %241, %mask : i32 -> <ui32>
          %248 = ascvec.duplicate_scalar %242, %mask : i32 -> <ui32>
          %carry_19, %dst_20 = ascvec.addc %247, %245, %mask : <ui32>, <ui32> -> !ascvec.mask, <ui32>
          %carry_21, %dst_22 = ascvec.addc %248, %246, %carry_19, %mask : <ui32>, <ui32> -> !ascvec.mask, <ui32>
          %249 = ascvec.duplicate_scalar %c16383_i32, %mask : i32 -> <ui32>
          %250 = ascvec.and %dst_16, %249, %mask : <ui32>, <ui32> -> <ui32>
          %251 = ascvec.duplicate_scalar %c-766435501_i32, %mask : i32 -> <ui32>
          %252 = ascvec.duplicate_scalar %c-845247145_i32, %mask : i32 -> <ui32>
          %dst0_23, %dst1_24 = ascvec.mull %208, %251, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %dst0_25, %dst1_26 = ascvec.mull %dst_20, %252, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %253 = ascvec.xor %dst1_26, %250, %mask : <ui32>, <ui32> -> <ui32>
          %254 = ascvec.xor %dst1_24, %dst_22, %mask : <ui32>, <ui32> -> <ui32>
          %255 = ascvec.duplicate_scalar %arg13, %mask : i32 -> <ui32>
          %256 = ascvec.duplicate_scalar %arg14, %mask : i32 -> <ui32>
          %257 = ascvec.xor %253, %255, %mask : <ui32>, <ui32> -> <ui32>
          %258 = ascvec.xor %254, %256, %mask : <ui32>, <ui32> -> <ui32>
          %dst0_27, %dst1_28 = ascvec.mull %257, %251, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %dst0_29, %dst1_30 = ascvec.mull %258, %252, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %259 = ascvec.xor %dst1_30, %dst0_25, %mask : <ui32>, <ui32> -> <ui32>
          %260 = ascvec.xor %dst1_28, %dst0_23, %mask : <ui32>, <ui32> -> <ui32>
          %261 = arith.addi %arg13, %c-1640531527_i32 : i32
          %262 = ascvec.duplicate_scalar %261, %mask : i32 -> <ui32>
          %263 = arith.addi %arg14, %c-1150833019_i32 : i32
          %264 = ascvec.duplicate_scalar %263, %mask : i32 -> <ui32>
          %265 = ascvec.xor %259, %262, %mask : <ui32>, <ui32> -> <ui32>
          %266 = ascvec.xor %260, %264, %mask : <ui32>, <ui32> -> <ui32>
          %dst0_31, %dst1_32 = ascvec.mull %265, %251, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %dst0_33, %dst1_34 = ascvec.mull %266, %252, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %267 = ascvec.xor %dst1_34, %dst0_29, %mask : <ui32>, <ui32> -> <ui32>
          %268 = ascvec.xor %dst1_32, %dst0_27, %mask : <ui32>, <ui32> -> <ui32>
          %269 = arith.addi %arg13, %c1013904242_i32 : i32
          %270 = ascvec.duplicate_scalar %269, %mask : i32 -> <ui32>
          %271 = arith.addi %arg14, %c1993301258_i32 : i32
          %272 = ascvec.duplicate_scalar %271, %mask : i32 -> <ui32>
          %273 = ascvec.xor %267, %270, %mask : <ui32>, <ui32> -> <ui32>
          %274 = ascvec.xor %268, %272, %mask : <ui32>, <ui32> -> <ui32>
          %dst0_35, %dst1_36 = ascvec.mull %273, %251, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %dst0_37, %dst1_38 = ascvec.mull %274, %252, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %275 = ascvec.xor %dst1_38, %dst0_33, %mask : <ui32>, <ui32> -> <ui32>
          %276 = ascvec.xor %dst1_36, %dst0_31, %mask : <ui32>, <ui32> -> <ui32>
          %277 = arith.addi %arg13, %c-626627285_i32 : i32
          %278 = ascvec.duplicate_scalar %277, %mask : i32 -> <ui32>
          %279 = arith.addi %arg14, %c842468239_i32 : i32
          %280 = ascvec.duplicate_scalar %279, %mask : i32 -> <ui32>
          %281 = ascvec.xor %275, %278, %mask : <ui32>, <ui32> -> <ui32>
          %282 = ascvec.xor %276, %280, %mask : <ui32>, <ui32> -> <ui32>
          %dst0_39, %dst1_40 = ascvec.mull %281, %251, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %dst0_41, %dst1_42 = ascvec.mull %282, %252, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %283 = ascvec.xor %dst1_42, %dst0_37, %mask : <ui32>, <ui32> -> <ui32>
          %284 = ascvec.xor %dst1_40, %dst0_35, %mask : <ui32>, <ui32> -> <ui32>
          %285 = arith.addi %arg13, %c2027808484_i32 : i32
          %286 = ascvec.duplicate_scalar %285, %mask : i32 -> <ui32>
          %287 = arith.addi %arg14, %c-308364780_i32 : i32
          %288 = ascvec.duplicate_scalar %287, %mask : i32 -> <ui32>
          %289 = ascvec.xor %283, %286, %mask : <ui32>, <ui32> -> <ui32>
          %290 = ascvec.xor %284, %288, %mask : <ui32>, <ui32> -> <ui32>
          %dst0_43, %dst1_44 = ascvec.mull %289, %251, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %dst0_45, %dst1_46 = ascvec.mull %290, %252, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %291 = ascvec.xor %dst1_46, %dst0_41, %mask : <ui32>, <ui32> -> <ui32>
          %292 = ascvec.xor %dst1_44, %dst0_39, %mask : <ui32>, <ui32> -> <ui32>
          %293 = arith.addi %arg13, %c387276957_i32 : i32
          %294 = ascvec.duplicate_scalar %293, %mask : i32 -> <ui32>
          %295 = arith.addi %arg14, %c-1459197799_i32 : i32
          %296 = ascvec.duplicate_scalar %295, %mask : i32 -> <ui32>
          %297 = ascvec.xor %291, %294, %mask : <ui32>, <ui32> -> <ui32>
          %298 = ascvec.xor %292, %296, %mask : <ui32>, <ui32> -> <ui32>
          %dst0_47, %dst1_48 = ascvec.mull %297, %251, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %dst0_49, %dst1_50 = ascvec.mull %298, %252, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %299 = ascvec.xor %dst1_50, %dst0_45, %mask : <ui32>, <ui32> -> <ui32>
          %300 = ascvec.xor %dst1_48, %dst0_43, %mask : <ui32>, <ui32> -> <ui32>
          %301 = arith.addi %arg13, %c-1253254570_i32 : i32
          %302 = ascvec.duplicate_scalar %301, %mask : i32 -> <ui32>
          %303 = arith.addi %arg14, %c1684936478_i32 : i32
          %304 = ascvec.duplicate_scalar %303, %mask : i32 -> <ui32>
          %305 = ascvec.xor %299, %302, %mask : <ui32>, <ui32> -> <ui32>
          %306 = ascvec.xor %300, %304, %mask : <ui32>, <ui32> -> <ui32>
          %dst0_51, %dst1_52 = ascvec.mull %305, %251, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %dst0_53, %dst1_54 = ascvec.mull %306, %252, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %307 = ascvec.xor %dst1_54, %dst0_49, %mask : <ui32>, <ui32> -> <ui32>
          %308 = ascvec.xor %dst1_52, %dst0_47, %mask : <ui32>, <ui32> -> <ui32>
          %309 = arith.addi %arg13, %c1401181199_i32 : i32
          %310 = ascvec.duplicate_scalar %309, %mask : i32 -> <ui32>
          %311 = arith.addi %arg14, %c534103459_i32 : i32
          %312 = ascvec.duplicate_scalar %311, %mask : i32 -> <ui32>
          %313 = ascvec.xor %307, %310, %mask : <ui32>, <ui32> -> <ui32>
          %314 = ascvec.xor %308, %312, %mask : <ui32>, <ui32> -> <ui32>
          %dst0_55, %dst1_56 = ascvec.mull %313, %251, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %dst0_57, %dst1_58 = ascvec.mull %314, %252, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %315 = ascvec.xor %dst1_58, %dst0_53, %mask : <ui32>, <ui32> -> <ui32>
          %316 = ascvec.xor %dst1_56, %dst0_51, %mask : <ui32>, <ui32> -> <ui32>
          %317 = arith.addi %arg13, %c-239350328_i32 : i32
          %318 = ascvec.duplicate_scalar %317, %mask : i32 -> <ui32>
          %319 = arith.addi %arg14, %c-616729560_i32 : i32
          %320 = ascvec.duplicate_scalar %319, %mask : i32 -> <ui32>
          %321 = ascvec.xor %315, %318, %mask : <ui32>, <ui32> -> <ui32>
          %322 = ascvec.xor %316, %320, %mask : <ui32>, <ui32> -> <ui32>
          %dst0_59, %dst1_60 = ascvec.mull %321, %251, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %dst0_61, %dst1_62 = ascvec.mull %322, %252, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %323 = ascvec.xor %dst1_62, %dst0_57, %mask : <ui32>, <ui32> -> <ui32>
          %324 = ascvec.xor %dst1_60, %dst0_55, %mask : <ui32>, <ui32> -> <ui32>
          %325 = arith.addi %arg13, %c-1879881855_i32 : i32
          %326 = ascvec.duplicate_scalar %325, %mask : i32 -> <ui32>
          %327 = arith.addi %arg14, %c-1767562579_i32 : i32
          %328 = ascvec.duplicate_scalar %327, %mask : i32 -> <ui32>
          %329 = ascvec.xor %323, %326, %mask : <ui32>, <ui32> -> <ui32>
          %330 = ascvec.xor %324, %328, %mask : <ui32>, <ui32> -> <ui32>
          %331 = arith.muli %arg16, %c64_i32 overflow<nsw> : i32
          %332 = ascvec.reinterpret %329 : <ui32> -> <i32>
          %333 = arith.index_cast %331 : i32 to index
          ascvec.storealign %arg15[%333], %332, %mask : memref<4096xi32, 1>, index, <i32>
          %334 = arith.addi %331, %c16_i32 overflow<nsw> : i32
          %335 = ascvec.reinterpret %dst0_61 : <ui32> -> <i32>
          %336 = arith.index_cast %334 : i32 to index
          ascvec.storealign %arg15[%336], %335, %mask : memref<4096xi32, 1>, index, <i32>
          %337 = arith.addi %331, %c32_i32 overflow<nsw> : i32
          %338 = ascvec.reinterpret %330 : <ui32> -> <i32>
          %339 = arith.index_cast %337 : i32 to index
          ascvec.storealign %arg15[%339], %338, %mask : memref<4096xi32, 1>, index, <i32>
          %340 = arith.addi %331, %c48_i32 overflow<nsw> : i32
          %341 = ascvec.reinterpret %dst0_59 : <ui32> -> <i32>
          %342 = arith.index_cast %340 : i32 to index
          ascvec.storealign %arg15[%342], %341, %mask : memref<4096xi32, 1>, index, <i32>
        }
        ascvec.mem_bar vst_vld
      } {cannir.vf_mode = "simd"}
      ascvec.get_buf v, %67, 0 {cannir.buffer.generated_sync}
      ascvec.get_buf v, %31, 0 {cannir.buffer.generated_sync}
      ascvec.get_buf v, %72, 0 {cannir.buffer.generated_sync}
      ascvec.vf(%66, %arg4, %62, %alloc_8, %alloc_9, %71 : memref<16384xui8, 1>, i64, memref<16384xbf16, 1>, memref<512xui32, 1>, memref<4096xi32, 1>, memref<1024xui8, 1>) {
      ^bb0(%arg9: memref<16384xui8, 1>, %arg10: i64, %arg11: memref<16384xbf16, 1>, %arg12: memref<512xui32, 1>, %arg13: memref<4096xi32, 1>, %arg14: memref<1024xui8, 1>):
        %c64_i32 = arith.constant 64 : i32
        %c7_i32 = arith.constant 7 : i32
        %c6_i32 = arith.constant 6 : i32
        %c5_i32 = arith.constant 5 : i32
        %c3_i32 = arith.constant 3 : i32
        %c2_i32 = arith.constant 2 : i32
        %c127_i32 = arith.constant 127 : i32
        %c123_i32 = arith.constant 123 : i32
        %c947912704_i32 = arith.constant 947912704 : i32
        %cst = arith.constant 4.2949673E+9 : f32
        %c29360128_i32 = arith.constant 29360128 : i32
        %c128_i32 = arith.constant 128 : i32
        %c65535_i32 = arith.constant 65535 : i32
        %c2139160576_i32 = arith.constant 2139160576 : i32
        %c254_i32 = arith.constant 254 : i32
        %c255_i32 = arith.constant 255 : i32
        %c132120577_i32 = arith.constant 132120577 : i32
        %c-123731969_i32 = arith.constant -123731969 : i32
        %c2139095040_i32 = arith.constant 2139095040 : i32
        %c1_i32 = arith.constant 1 : i32
        %c8_i32 = arith.constant 8 : i32
        %c2147483647_i32 = arith.constant 2147483647 : i32
        %c4_i32 = arith.constant 4 : i32
        %c32_i32 = arith.constant 32 : i32
        %c0_i32_10 = arith.constant 0 : i32
        %c0_11 = arith.constant 0 : index
        %c16_i32 = arith.constant 16 : i32
        %mask, %remaining_out = ascvec.update_mask %c16_i32 : !ascvec.mask
        ascvec.store_unalign_begin %arg9 : memref<16384xui8, 1>
        scf.for %arg15 = %c0_i32_10 to %c8_i32 step %c1_i32  : i32 {
          %181 = ascvec.create_mask : !ascvec.mask
          %182 = ascvec.arange %c0_i32_10 : i32 -> <ui32>
          %183 = ascvec.duplicate_scalar %c0_i32_10, %181 : i32 -> <ui32>
          %184 = scf.for %arg16 = %c0_i32_10 to %c8_i32 step %c1_i32 iter_args(%arg17 = %183) -> (!ascvec.vreg<ui32>)  : i32 {
            %205 = arith.muli %arg15, %c32_i32 overflow<nsw> : i32
            %206 = arith.muli %arg16, %c4_i32 overflow<nsw> : i32
            %207 = arith.addi %205, %206 overflow<nsw> : i32
            %208 = arith.extsi %207 : i32 to i64
            %209 = arith.muli %208, %arg10 : i64
            %210 = arith.index_cast %209 : i64 to index
            %211 = ascvec.loadalign_unpack %arg11[%210] {mode = "b16_to_b32"} : memref<16384xbf16, 1>, index -> <bf16>
            %212 = ascvec.cast %211, %181 {rounding = 6 : i32} : <bf16> -> <f32>
            %213 = ascvec.reinterpret %212 : <f32> -> <ui32>
            %214 = ascvec.duplicate_scalar %c2147483647_i32, %181 : i32 -> <ui32>
            %215 = ascvec.and %213, %214, %181 : <ui32>, <ui32> -> <ui32>
            %216 = ascvec.max %arg17, %215, %181 : <ui32>
            scf.yield %216 : !ascvec.vreg<ui32>
          }
          %185 = ascvec.duplicate_scalar %c16_i32, %181 : i32 -> <ui32>
          %186 = ascvec.xor %182, %185, %181 : <ui32>, <ui32> -> <ui32>
          %187 = ascvec.gather_reg %184, %186 : <ui32>, <ui32> -> <ui32>
          %188 = ascvec.max %184, %187, %181 : <ui32>
          %189 = ascvec.duplicate_scalar %c32_i32, %181 : i32 -> <ui32>
          %190 = ascvec.xor %182, %189, %181 : <ui32>, <ui32> -> <ui32>
          %191 = ascvec.gather_reg %188, %190 : <ui32>, <ui32> -> <ui32>
          %192 = ascvec.max %188, %191, %181 : <ui32>
          %193 = ascvec.cmp_ge_scalar %192, %c2139095040_i32, %mask : <ui32>, i32 -> !ascvec.mask
          %194 = ascvec.add_scalar %192, %c-123731969_i32, %mask : <ui32>, i32
          %195 = ascvec.cmp_gt_scalar %192, %c132120577_i32, %mask : <ui32>, i32 -> !ascvec.mask
          %196 = ascvec.shr %194, 23, %195 : <ui32> -> <ui32>
          %197 = ascvec.duplicate_scalar %c255_i32, %193 merge %196 : !ascvec.vreg<ui32> {mode = "merging"} : i32 -> <ui32>
          %198 = ascvec.duplicate_scalar %c254_i32, %mask : i32 -> <ui32>
          %199 = ascvec.sub %198, %197, %mask : <ui32>
          %200 = ascvec.shl %199, 23, %mask : <ui32> -> <ui32>
          %201 = ascvec.duplicate_scalar %c2139160576_i32, %193 merge %200 : !ascvec.vreg<ui32> {mode = "merging"} : i32 -> <ui32>
          %202 = ascvec.reinterpret %201 : <ui32> -> <f32>
          %203 = arith.muli %arg15, %c16_i32 overflow<nsw> : i32
          %204 = arith.index_cast %203 : i32 to index
          ascvec.storealign %arg12[%204], %197, %mask : memref<512xui32, 1>, index, <ui32>
          scf.for %arg16 = %c0_i32_10 to %c4_i32 step %c1_i32  : i32 {
            %205 = arith.muli %arg15, %c64_i32 overflow<nsw> : i32
            %206 = arith.muli %arg16, %c16_i32 overflow<nsw> : i32
            %207 = arith.addi %205, %206 overflow<nsw> : i32
            %208 = arith.index_cast %207 : i32 to index
            %209 = ascvec.loadalign %arg13[%208] : memref<4096xi32, 1>, index -> <i32>
            %210 = ascvec.reinterpret %209 : <i32> -> <ui32>
            %211 = arith.muli %arg15, %c32_i32 overflow<nsw> : i32
            %212 = arith.muli %arg16, %c8_i32 overflow<nsw> : i32
            %213 = arith.addi %211, %212 overflow<nsw> : i32
            %214 = ascvec.shr %210, 21, %mask : <ui32> -> <ui32>
            %215 = ascvec.shl %210, 11, %mask : <ui32> -> <ui32>
            %216 = ascvec.or %214, %215, %mask : <ui32>, <ui32> -> <ui32>
            %217 = ascvec.shr %210, 29, %mask : <ui32> -> <ui32>
            %218 = ascvec.shl %210, 3, %mask : <ui32> -> <ui32>
            %219 = ascvec.or %217, %218, %mask : <ui32>, <ui32> -> <ui32>
            %220 = ascvec.shr %210, 5, %mask : <ui32> -> <ui32>
            %221 = ascvec.shl %210, 27, %mask : <ui32> -> <ui32>
            %222 = ascvec.or %220, %221, %mask : <ui32>, <ui32> -> <ui32>
            %223 = ascvec.shr %210, 13, %mask : <ui32> -> <ui32>
            %224 = ascvec.shl %210, 19, %mask : <ui32> -> <ui32>
            %225 = ascvec.or %223, %224, %mask : <ui32>, <ui32> -> <ui32>
            %226 = ascvec.shr %210, 24, %mask : <ui32> -> <ui32>
            %227 = ascvec.shl %210, 8, %mask : <ui32> -> <ui32>
            %228 = ascvec.or %226, %227, %mask : <ui32>, <ui32> -> <ui32>
            %229 = ascvec.shr %210, 8, %mask : <ui32> -> <ui32>
            %230 = ascvec.shl %210, 24, %mask : <ui32> -> <ui32>
            %231 = ascvec.or %229, %230, %mask : <ui32>, <ui32> -> <ui32>
            %232 = ascvec.shr %210, 16, %mask : <ui32> -> <ui32>
            %233 = ascvec.shl %210, 16, %mask : <ui32> -> <ui32>
            %234 = ascvec.or %232, %233, %mask : <ui32>, <ui32> -> <ui32>
            %235 = arith.extsi %213 : i32 to i64
            %236 = arith.muli %235, %arg10 : i64
            %237 = ascvec.duplicate_scalar %c65535_i32, %mask : i32 -> <ui32>
            %238 = ascvec.and %216, %237, %mask : <ui32>, <ui32> -> <ui32>
            %239 = arith.index_cast %236 : i64 to index
            %240 = ascvec.loadalign_unpack %arg11[%239] {mode = "b16_to_b32"} : memref<16384xbf16, 1>, index -> <bf16>
            %241 = ascvec.cast %240, %mask {rounding = 6 : i32} : <bf16> -> <f32>
            %242 = ascvec.mul %241, %202, %mask : <f32>
            %243 = ascvec.cast %242, %mask : <f32> -> <bf16>
            %244 = ascvec.cast %243, %mask {rounding = 6 : i32} : <bf16> -> <f32>
            %245 = ascvec.reinterpret %244 : <f32> -> <ui32>
            %246 = ascvec.shr %245, 24, %mask : <ui32> -> <ui32>
            %247 = ascvec.duplicate_scalar %c128_i32, %mask : i32 -> <ui32>
            %248 = ascvec.and %246, %247, %mask : <ui32>, <ui32> -> <ui32>
            %249 = ascvec.duplicate_scalar %c2147483647_i32, %mask : i32 -> <ui32>
            %250 = ascvec.and %245, %249, %mask : <ui32>, <ui32> -> <ui32>
            %251 = ascvec.shr %250, 5, %mask : <ui32> -> <ui32>
            %252 = ascvec.duplicate_scalar %c29360128_i32, %mask : i32 -> <ui32>
            %253 = ascvec.sub %251, %252, %mask : <ui32>
            %254 = ascvec.reinterpret %250 : <ui32> -> <f32>
            %255 = ascvec.mul_scalar %254, %cst, %mask : <f32>, f32
            %256 = ascvec.cast %255, %mask {rounding = 4 : i32} : <f32> -> <i32>
            %257 = ascvec.reinterpret %256 : <i32> -> <ui32>
            %258 = ascvec.cmp_ge_scalar %250, %c947912704_i32, %mask : <ui32>, i32 -> !ascvec.mask
            %259 = ascvec.select %253, %257, %258 : <ui32>
            %260 = ascvec.and %238, %237, %mask : <ui32>, <ui32> -> <ui32>
            %261 = ascvec.add %259, %260, %mask : <ui32>
            %262 = ascvec.shr %261, 16, %mask : <ui32> -> <ui32>
            %263 = ascvec.min_scalar %262, %c123_i32, %mask : <ui32>, i32
            %264 = ascvec.or %263, %248, %mask : <ui32>, <ui32> -> <ui32>
            %265 = ascvec.duplicate_scalar %c127_i32, %mask : i32 -> <ui32>
            %266 = ascvec.cmp_gt_scalar %250, %c2139095040_i32, %mask : <ui32>, i32 -> !ascvec.mask
            %267 = ascvec.select %265, %264, %266 : <ui32>
            %268 = ascvec.pack %267 src_elem_bits 32 dst_elem_bits 16 {part = "low"} : <ui32> -> <ui16>
            %269 = ascvec.pack %268 src_elem_bits 16 dst_elem_bits 8 {part = "low"} : <ui16> -> <ui8>
            %mask_17, %remaining_out_18 = ascvec.update_mask %c16_i32 elem_bits 8 : !ascvec.mask
            %270 = ascvec.squeeze_store_reg %269, %mask_17 : <ui8>, !ascvec.mask -> <ui8>
            ascvec.store_unalign %arg9[%c0_11], %270 : memref<16384xui8, 1>, <ui8>
            %271 = arith.addi %213, %c1_i32 overflow<nsw> : i32
            %272 = arith.extsi %271 : i32 to i64
            %273 = arith.muli %272, %arg10 : i64
            %274 = ascvec.and %228, %237, %mask : <ui32>, <ui32> -> <ui32>
            %275 = arith.index_cast %273 : i64 to index
            %276 = ascvec.loadalign_unpack %arg11[%275] {mode = "b16_to_b32"} : memref<16384xbf16, 1>, index -> <bf16>
            %277 = ascvec.cast %276, %mask {rounding = 6 : i32} : <bf16> -> <f32>
            %278 = ascvec.mul %277, %202, %mask : <f32>
            %279 = ascvec.cast %278, %mask : <f32> -> <bf16>
            %280 = ascvec.cast %279, %mask {rounding = 6 : i32} : <bf16> -> <f32>
            %281 = ascvec.reinterpret %280 : <f32> -> <ui32>
            %282 = ascvec.shr %281, 24, %mask : <ui32> -> <ui32>
            %283 = ascvec.and %282, %247, %mask : <ui32>, <ui32> -> <ui32>
            %284 = ascvec.and %281, %249, %mask : <ui32>, <ui32> -> <ui32>
            %285 = ascvec.shr %284, 5, %mask : <ui32> -> <ui32>
            %286 = ascvec.sub %285, %252, %mask : <ui32>
            %287 = ascvec.reinterpret %284 : <ui32> -> <f32>
            %288 = ascvec.mul_scalar %287, %cst, %mask : <f32>, f32
            %289 = ascvec.cast %288, %mask {rounding = 4 : i32} : <f32> -> <i32>
            %290 = ascvec.reinterpret %289 : <i32> -> <ui32>
            %291 = ascvec.cmp_ge_scalar %284, %c947912704_i32, %mask : <ui32>, i32 -> !ascvec.mask
            %292 = ascvec.select %286, %290, %291 : <ui32>
            %293 = ascvec.and %274, %237, %mask : <ui32>, <ui32> -> <ui32>
            %294 = ascvec.add %292, %293, %mask : <ui32>
            %295 = ascvec.shr %294, 16, %mask : <ui32> -> <ui32>
            %296 = ascvec.min_scalar %295, %c123_i32, %mask : <ui32>, i32
            %297 = ascvec.or %296, %283, %mask : <ui32>, <ui32> -> <ui32>
            %298 = ascvec.cmp_gt_scalar %284, %c2139095040_i32, %mask : <ui32>, i32 -> !ascvec.mask
            %299 = ascvec.select %265, %297, %298 : <ui32>
            %300 = ascvec.pack %299 src_elem_bits 32 dst_elem_bits 16 {part = "low"} : <ui32> -> <ui16>
            %301 = ascvec.pack %300 src_elem_bits 16 dst_elem_bits 8 {part = "low"} : <ui16> -> <ui8>
            %mask_19, %remaining_out_20 = ascvec.update_mask %c16_i32 elem_bits 8 : !ascvec.mask
            %302 = ascvec.squeeze_store_reg %301, %mask_19 : <ui8>, !ascvec.mask -> <ui8>
            ascvec.store_unalign %arg9[%c0_11], %302 : memref<16384xui8, 1>, <ui8>
            %303 = arith.addi %213, %c2_i32 overflow<nsw> : i32
            %304 = arith.extsi %303 : i32 to i64
            %305 = arith.muli %304, %arg10 : i64
            %306 = ascvec.and %219, %237, %mask : <ui32>, <ui32> -> <ui32>
            %307 = arith.index_cast %305 : i64 to index
            %308 = ascvec.loadalign_unpack %arg11[%307] {mode = "b16_to_b32"} : memref<16384xbf16, 1>, index -> <bf16>
            %309 = ascvec.cast %308, %mask {rounding = 6 : i32} : <bf16> -> <f32>
            %310 = ascvec.mul %309, %202, %mask : <f32>
            %311 = ascvec.cast %310, %mask : <f32> -> <bf16>
            %312 = ascvec.cast %311, %mask {rounding = 6 : i32} : <bf16> -> <f32>
            %313 = ascvec.reinterpret %312 : <f32> -> <ui32>
            %314 = ascvec.shr %313, 24, %mask : <ui32> -> <ui32>
            %315 = ascvec.and %314, %247, %mask : <ui32>, <ui32> -> <ui32>
            %316 = ascvec.and %313, %249, %mask : <ui32>, <ui32> -> <ui32>
            %317 = ascvec.shr %316, 5, %mask : <ui32> -> <ui32>
            %318 = ascvec.sub %317, %252, %mask : <ui32>
            %319 = ascvec.reinterpret %316 : <ui32> -> <f32>
            %320 = ascvec.mul_scalar %319, %cst, %mask : <f32>, f32
            %321 = ascvec.cast %320, %mask {rounding = 4 : i32} : <f32> -> <i32>
            %322 = ascvec.reinterpret %321 : <i32> -> <ui32>
            %323 = ascvec.cmp_ge_scalar %316, %c947912704_i32, %mask : <ui32>, i32 -> !ascvec.mask
            %324 = ascvec.select %318, %322, %323 : <ui32>
            %325 = ascvec.and %306, %237, %mask : <ui32>, <ui32> -> <ui32>
            %326 = ascvec.add %324, %325, %mask : <ui32>
            %327 = ascvec.shr %326, 16, %mask : <ui32> -> <ui32>
            %328 = ascvec.min_scalar %327, %c123_i32, %mask : <ui32>, i32
            %329 = ascvec.or %328, %315, %mask : <ui32>, <ui32> -> <ui32>
            %330 = ascvec.cmp_gt_scalar %316, %c2139095040_i32, %mask : <ui32>, i32 -> !ascvec.mask
            %331 = ascvec.select %265, %329, %330 : <ui32>
            %332 = ascvec.pack %331 src_elem_bits 32 dst_elem_bits 16 {part = "low"} : <ui32> -> <ui16>
            %333 = ascvec.pack %332 src_elem_bits 16 dst_elem_bits 8 {part = "low"} : <ui16> -> <ui8>
            %mask_21, %remaining_out_22 = ascvec.update_mask %c16_i32 elem_bits 8 : !ascvec.mask
            %334 = ascvec.squeeze_store_reg %333, %mask_21 : <ui8>, !ascvec.mask -> <ui8>
            ascvec.store_unalign %arg9[%c0_11], %334 : memref<16384xui8, 1>, <ui8>
            %335 = arith.addi %213, %c3_i32 overflow<nsw> : i32
            %336 = arith.extsi %335 : i32 to i64
            %337 = arith.muli %336, %arg10 : i64
            %338 = ascvec.and %210, %237, %mask : <ui32>, <ui32> -> <ui32>
            %339 = arith.index_cast %337 : i64 to index
            %340 = ascvec.loadalign_unpack %arg11[%339] {mode = "b16_to_b32"} : memref<16384xbf16, 1>, index -> <bf16>
            %341 = ascvec.cast %340, %mask {rounding = 6 : i32} : <bf16> -> <f32>
            %342 = ascvec.mul %341, %202, %mask : <f32>
            %343 = ascvec.cast %342, %mask : <f32> -> <bf16>
            %344 = ascvec.cast %343, %mask {rounding = 6 : i32} : <bf16> -> <f32>
            %345 = ascvec.reinterpret %344 : <f32> -> <ui32>
            %346 = ascvec.shr %345, 24, %mask : <ui32> -> <ui32>
            %347 = ascvec.and %346, %247, %mask : <ui32>, <ui32> -> <ui32>
            %348 = ascvec.and %345, %249, %mask : <ui32>, <ui32> -> <ui32>
            %349 = ascvec.shr %348, 5, %mask : <ui32> -> <ui32>
            %350 = ascvec.sub %349, %252, %mask : <ui32>
            %351 = ascvec.reinterpret %348 : <ui32> -> <f32>
            %352 = ascvec.mul_scalar %351, %cst, %mask : <f32>, f32
            %353 = ascvec.cast %352, %mask {rounding = 4 : i32} : <f32> -> <i32>
            %354 = ascvec.reinterpret %353 : <i32> -> <ui32>
            %355 = ascvec.cmp_ge_scalar %348, %c947912704_i32, %mask : <ui32>, i32 -> !ascvec.mask
            %356 = ascvec.select %350, %354, %355 : <ui32>
            %357 = ascvec.and %338, %237, %mask : <ui32>, <ui32> -> <ui32>
            %358 = ascvec.add %356, %357, %mask : <ui32>
            %359 = ascvec.shr %358, 16, %mask : <ui32> -> <ui32>
            %360 = ascvec.min_scalar %359, %c123_i32, %mask : <ui32>, i32
            %361 = ascvec.or %360, %347, %mask : <ui32>, <ui32> -> <ui32>
            %362 = ascvec.cmp_gt_scalar %348, %c2139095040_i32, %mask : <ui32>, i32 -> !ascvec.mask
            %363 = ascvec.select %265, %361, %362 : <ui32>
            %364 = ascvec.pack %363 src_elem_bits 32 dst_elem_bits 16 {part = "low"} : <ui32> -> <ui16>
            %365 = ascvec.pack %364 src_elem_bits 16 dst_elem_bits 8 {part = "low"} : <ui16> -> <ui8>
            %mask_23, %remaining_out_24 = ascvec.update_mask %c16_i32 elem_bits 8 : !ascvec.mask
            %366 = ascvec.squeeze_store_reg %365, %mask_23 : <ui8>, !ascvec.mask -> <ui8>
            ascvec.store_unalign %arg9[%c0_11], %366 : memref<16384xui8, 1>, <ui8>
            %367 = arith.addi %213, %c4_i32 overflow<nsw> : i32
            %368 = arith.extsi %367 : i32 to i64
            %369 = arith.muli %368, %arg10 : i64
            %370 = ascvec.and %222, %237, %mask : <ui32>, <ui32> -> <ui32>
            %371 = arith.index_cast %369 : i64 to index
            %372 = ascvec.loadalign_unpack %arg11[%371] {mode = "b16_to_b32"} : memref<16384xbf16, 1>, index -> <bf16>
            %373 = ascvec.cast %372, %mask {rounding = 6 : i32} : <bf16> -> <f32>
            %374 = ascvec.mul %373, %202, %mask : <f32>
            %375 = ascvec.cast %374, %mask : <f32> -> <bf16>
            %376 = ascvec.cast %375, %mask {rounding = 6 : i32} : <bf16> -> <f32>
            %377 = ascvec.reinterpret %376 : <f32> -> <ui32>
            %378 = ascvec.shr %377, 24, %mask : <ui32> -> <ui32>
            %379 = ascvec.and %378, %247, %mask : <ui32>, <ui32> -> <ui32>
            %380 = ascvec.and %377, %249, %mask : <ui32>, <ui32> -> <ui32>
            %381 = ascvec.shr %380, 5, %mask : <ui32> -> <ui32>
            %382 = ascvec.sub %381, %252, %mask : <ui32>
            %383 = ascvec.reinterpret %380 : <ui32> -> <f32>
            %384 = ascvec.mul_scalar %383, %cst, %mask : <f32>, f32
            %385 = ascvec.cast %384, %mask {rounding = 4 : i32} : <f32> -> <i32>
            %386 = ascvec.reinterpret %385 : <i32> -> <ui32>
            %387 = ascvec.cmp_ge_scalar %380, %c947912704_i32, %mask : <ui32>, i32 -> !ascvec.mask
            %388 = ascvec.select %382, %386, %387 : <ui32>
            %389 = ascvec.and %370, %237, %mask : <ui32>, <ui32> -> <ui32>
            %390 = ascvec.add %388, %389, %mask : <ui32>
            %391 = ascvec.shr %390, 16, %mask : <ui32> -> <ui32>
            %392 = ascvec.min_scalar %391, %c123_i32, %mask : <ui32>, i32
            %393 = ascvec.or %392, %379, %mask : <ui32>, <ui32> -> <ui32>
            %394 = ascvec.cmp_gt_scalar %380, %c2139095040_i32, %mask : <ui32>, i32 -> !ascvec.mask
            %395 = ascvec.select %265, %393, %394 : <ui32>
            %396 = ascvec.pack %395 src_elem_bits 32 dst_elem_bits 16 {part = "low"} : <ui32> -> <ui16>
            %397 = ascvec.pack %396 src_elem_bits 16 dst_elem_bits 8 {part = "low"} : <ui16> -> <ui8>
            %mask_25, %remaining_out_26 = ascvec.update_mask %c16_i32 elem_bits 8 : !ascvec.mask
            %398 = ascvec.squeeze_store_reg %397, %mask_25 : <ui8>, !ascvec.mask -> <ui8>
            ascvec.store_unalign %arg9[%c0_11], %398 : memref<16384xui8, 1>, <ui8>
            %399 = arith.addi %213, %c5_i32 overflow<nsw> : i32
            %400 = arith.extsi %399 : i32 to i64
            %401 = arith.muli %400, %arg10 : i64
            %402 = ascvec.and %231, %237, %mask : <ui32>, <ui32> -> <ui32>
            %403 = arith.index_cast %401 : i64 to index
            %404 = ascvec.loadalign_unpack %arg11[%403] {mode = "b16_to_b32"} : memref<16384xbf16, 1>, index -> <bf16>
            %405 = ascvec.cast %404, %mask {rounding = 6 : i32} : <bf16> -> <f32>
            %406 = ascvec.mul %405, %202, %mask : <f32>
            %407 = ascvec.cast %406, %mask : <f32> -> <bf16>
            %408 = ascvec.cast %407, %mask {rounding = 6 : i32} : <bf16> -> <f32>
            %409 = ascvec.reinterpret %408 : <f32> -> <ui32>
            %410 = ascvec.shr %409, 24, %mask : <ui32> -> <ui32>
            %411 = ascvec.and %410, %247, %mask : <ui32>, <ui32> -> <ui32>
            %412 = ascvec.and %409, %249, %mask : <ui32>, <ui32> -> <ui32>
            %413 = ascvec.shr %412, 5, %mask : <ui32> -> <ui32>
            %414 = ascvec.sub %413, %252, %mask : <ui32>
            %415 = ascvec.reinterpret %412 : <ui32> -> <f32>
            %416 = ascvec.mul_scalar %415, %cst, %mask : <f32>, f32
            %417 = ascvec.cast %416, %mask {rounding = 4 : i32} : <f32> -> <i32>
            %418 = ascvec.reinterpret %417 : <i32> -> <ui32>
            %419 = ascvec.cmp_ge_scalar %412, %c947912704_i32, %mask : <ui32>, i32 -> !ascvec.mask
            %420 = ascvec.select %414, %418, %419 : <ui32>
            %421 = ascvec.and %402, %237, %mask : <ui32>, <ui32> -> <ui32>
            %422 = ascvec.add %420, %421, %mask : <ui32>
            %423 = ascvec.shr %422, 16, %mask : <ui32> -> <ui32>
            %424 = ascvec.min_scalar %423, %c123_i32, %mask : <ui32>, i32
            %425 = ascvec.or %424, %411, %mask : <ui32>, <ui32> -> <ui32>
            %426 = ascvec.cmp_gt_scalar %412, %c2139095040_i32, %mask : <ui32>, i32 -> !ascvec.mask
            %427 = ascvec.select %265, %425, %426 : <ui32>
            %428 = ascvec.pack %427 src_elem_bits 32 dst_elem_bits 16 {part = "low"} : <ui32> -> <ui16>
            %429 = ascvec.pack %428 src_elem_bits 16 dst_elem_bits 8 {part = "low"} : <ui16> -> <ui8>
            %mask_27, %remaining_out_28 = ascvec.update_mask %c16_i32 elem_bits 8 : !ascvec.mask
            %430 = ascvec.squeeze_store_reg %429, %mask_27 : <ui8>, !ascvec.mask -> <ui8>
            ascvec.store_unalign %arg9[%c0_11], %430 : memref<16384xui8, 1>, <ui8>
            %431 = arith.addi %213, %c6_i32 overflow<nsw> : i32
            %432 = arith.extsi %431 : i32 to i64
            %433 = arith.muli %432, %arg10 : i64
            %434 = ascvec.and %225, %237, %mask : <ui32>, <ui32> -> <ui32>
            %435 = arith.index_cast %433 : i64 to index
            %436 = ascvec.loadalign_unpack %arg11[%435] {mode = "b16_to_b32"} : memref<16384xbf16, 1>, index -> <bf16>
            %437 = ascvec.cast %436, %mask {rounding = 6 : i32} : <bf16> -> <f32>
            %438 = ascvec.mul %437, %202, %mask : <f32>
            %439 = ascvec.cast %438, %mask : <f32> -> <bf16>
            %440 = ascvec.cast %439, %mask {rounding = 6 : i32} : <bf16> -> <f32>
            %441 = ascvec.reinterpret %440 : <f32> -> <ui32>
            %442 = ascvec.shr %441, 24, %mask : <ui32> -> <ui32>
            %443 = ascvec.and %442, %247, %mask : <ui32>, <ui32> -> <ui32>
            %444 = ascvec.and %441, %249, %mask : <ui32>, <ui32> -> <ui32>
            %445 = ascvec.shr %444, 5, %mask : <ui32> -> <ui32>
            %446 = ascvec.sub %445, %252, %mask : <ui32>
            %447 = ascvec.reinterpret %444 : <ui32> -> <f32>
            %448 = ascvec.mul_scalar %447, %cst, %mask : <f32>, f32
            %449 = ascvec.cast %448, %mask {rounding = 4 : i32} : <f32> -> <i32>
            %450 = ascvec.reinterpret %449 : <i32> -> <ui32>
            %451 = ascvec.cmp_ge_scalar %444, %c947912704_i32, %mask : <ui32>, i32 -> !ascvec.mask
            %452 = ascvec.select %446, %450, %451 : <ui32>
            %453 = ascvec.and %434, %237, %mask : <ui32>, <ui32> -> <ui32>
            %454 = ascvec.add %452, %453, %mask : <ui32>
            %455 = ascvec.shr %454, 16, %mask : <ui32> -> <ui32>
            %456 = ascvec.min_scalar %455, %c123_i32, %mask : <ui32>, i32
            %457 = ascvec.or %456, %443, %mask : <ui32>, <ui32> -> <ui32>
            %458 = ascvec.cmp_gt_scalar %444, %c2139095040_i32, %mask : <ui32>, i32 -> !ascvec.mask
            %459 = ascvec.select %265, %457, %458 : <ui32>
            %460 = ascvec.pack %459 src_elem_bits 32 dst_elem_bits 16 {part = "low"} : <ui32> -> <ui16>
            %461 = ascvec.pack %460 src_elem_bits 16 dst_elem_bits 8 {part = "low"} : <ui16> -> <ui8>
            %mask_29, %remaining_out_30 = ascvec.update_mask %c16_i32 elem_bits 8 : !ascvec.mask
            %462 = ascvec.squeeze_store_reg %461, %mask_29 : <ui8>, !ascvec.mask -> <ui8>
            ascvec.store_unalign %arg9[%c0_11], %462 : memref<16384xui8, 1>, <ui8>
            %463 = arith.addi %213, %c7_i32 overflow<nsw> : i32
            %464 = arith.extsi %463 : i32 to i64
            %465 = arith.muli %464, %arg10 : i64
            %466 = ascvec.and %234, %237, %mask : <ui32>, <ui32> -> <ui32>
            %467 = arith.index_cast %465 : i64 to index
            %468 = ascvec.loadalign_unpack %arg11[%467] {mode = "b16_to_b32"} : memref<16384xbf16, 1>, index -> <bf16>
            %469 = ascvec.cast %468, %mask {rounding = 6 : i32} : <bf16> -> <f32>
            %470 = ascvec.mul %469, %202, %mask : <f32>
            %471 = ascvec.cast %470, %mask : <f32> -> <bf16>
            %472 = ascvec.cast %471, %mask {rounding = 6 : i32} : <bf16> -> <f32>
            %473 = ascvec.reinterpret %472 : <f32> -> <ui32>
            %474 = ascvec.shr %473, 24, %mask : <ui32> -> <ui32>
            %475 = ascvec.and %474, %247, %mask : <ui32>, <ui32> -> <ui32>
            %476 = ascvec.and %473, %249, %mask : <ui32>, <ui32> -> <ui32>
            %477 = ascvec.shr %476, 5, %mask : <ui32> -> <ui32>
            %478 = ascvec.sub %477, %252, %mask : <ui32>
            %479 = ascvec.reinterpret %476 : <ui32> -> <f32>
            %480 = ascvec.mul_scalar %479, %cst, %mask : <f32>, f32
            %481 = ascvec.cast %480, %mask {rounding = 4 : i32} : <f32> -> <i32>
            %482 = ascvec.reinterpret %481 : <i32> -> <ui32>
            %483 = ascvec.cmp_ge_scalar %476, %c947912704_i32, %mask : <ui32>, i32 -> !ascvec.mask
            %484 = ascvec.select %478, %482, %483 : <ui32>
            %485 = ascvec.and %466, %237, %mask : <ui32>, <ui32> -> <ui32>
            %486 = ascvec.add %484, %485, %mask : <ui32>
            %487 = ascvec.shr %486, 16, %mask : <ui32> -> <ui32>
            %488 = ascvec.min_scalar %487, %c123_i32, %mask : <ui32>, i32
            %489 = ascvec.or %488, %475, %mask : <ui32>, <ui32> -> <ui32>
            %490 = ascvec.cmp_gt_scalar %476, %c2139095040_i32, %mask : <ui32>, i32 -> !ascvec.mask
            %491 = ascvec.select %265, %489, %490 : <ui32>
            %492 = ascvec.pack %491 src_elem_bits 32 dst_elem_bits 16 {part = "low"} : <ui32> -> <ui16>
            %493 = ascvec.pack %492 src_elem_bits 16 dst_elem_bits 8 {part = "low"} : <ui16> -> <ui8>
            %mask_31, %remaining_out_32 = ascvec.update_mask %c16_i32 elem_bits 8 : !ascvec.mask
            %494 = ascvec.squeeze_store_reg %493, %mask_31 : <ui8>, !ascvec.mask -> <ui8>
            ascvec.store_unalign %arg9[%c0_11], %494 : memref<16384xui8, 1>, <ui8>
          }
        }
        %c0_12 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        scf.for %arg15 = %c0_12 to %c1 step %c1 {
          ascvec.store_unalign_post %arg9[%c0_11] : memref<16384xui8, 1>
        }
        ascvec.mem_bar vst_vld
        %mask_13, %remaining_out_14 = ascvec.update_mask %c32_i32 elem_bits 8 : !ascvec.mask
        ascvec.store_unalign_begin %arg14 : memref<1024xui8, 1>
        scf.for %arg15 = %c0_i32_10 to %c4_i32 step %c1_i32  : i32 {
          %181 = arith.muli %arg15, %c32_i32 overflow<nsw> : i32
          %182 = arith.index_cast %181 : i32 to index
          %183 = ascvec.loadalign %arg12[%182] : memref<512xui32, 1>, index -> <ui32>
          %184 = arith.addi %181, %c16_i32 overflow<nsw> : i32
          %185 = arith.index_cast %184 : i32 to index
          %186 = ascvec.loadalign %arg12[%185] : memref<512xui32, 1>, index -> <ui32>
          %187 = ascvec.shl %186, 8, %mask : <ui32> -> <ui32>
          %188 = ascvec.or %183, %187, %mask : <ui32>, <ui32> -> <ui32>
          %189 = ascvec.pack %188 src_elem_bits 32 dst_elem_bits 16 {part = "low"} : <ui32> -> <ui16>
          %190 = ascvec.reinterpret %189 : <ui16> -> <ui8>
          %191 = ascvec.squeeze_store_reg %190, %mask_13 : <ui8>, !ascvec.mask -> <ui8>
          ascvec.store_unalign %arg14[%c0_11], %191 : memref<1024xui8, 1>, <ui8>
        }
        %c0_15 = arith.constant 0 : index
        %c1_16 = arith.constant 1 : index
        scf.for %arg15 = %c0_15 to %c1_16 step %c1_16 {
          ascvec.store_unalign_post %arg14[%c0_11] : memref<1024xui8, 1>
        }
        ascvec.mem_bar vst_vld
      } {cannir.vf_mode = "simd"}
      ascvec.release_buf v, %72, 0 {cannir.buffer.generated_sync}
      ascvec.release_buf v, %31, 0 {cannir.buffer.generated_sync}
      ascvec.release_buf v, %67, 0 {cannir.buffer.generated_sync}
      %100 = cannir.extract_layout %arg1 : !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
      %101 = cannir.extract_shape %100 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, -1)>
      %102 = cannir.extract_dim %101[0] : !cannir.shape<(-1, -1)>
      %103 = arith.minsi %22, %102 : i64
      %104 = arith.subi %102, %103 : i64
      %105 = arith.minsi %104, %c1_i64 : i64
      %106 = arith.cmpi sle, %22, %102 : i64
      %107 = arith.select %106, %105, %c0_i64 : i64
      %108 = arith.cmpi sgt, %107, %c1_i64 : i64
      %109 = arith.select %108, %c1_i64, %107 : i64
      %110 = cannir.extract_dim %101[1] : !cannir.shape<(-1, -1)>
      %111 = arith.divsi %110, %44 : i64
      %112 = arith.minsi %24, %111 : i64
      %113 = arith.muli %112, %25 : i64
      %114 = arith.subi %110, %113 : i64
      %115 = arith.minsi %25, %114 : i64
      %116 = arith.cmpi sle, %24, %111 : i64
      %117 = arith.select %116, %115, %c0_i64 : i64
      %118 = arith.cmpi sgt, %117, %c16384_i64 : i64
      %119 = arith.select %118, %c16384_i64, %117 : i64
      %120 = cannir.extract_physical_layout %arg1 : !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
      %121 = cannir.extract_stride %120 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
      %122 = cannir.extract_dim %121[0] : !cannir.stride<(-1, 1)>
      %123 = arith.muli %22, %122 : i64
      %124 = arith.addi %123, %54 : i64
      %125 = arith.muli %124, %c8_i64 : i64
      %126 = ascscalar.memref_address %alloc_0 : memref<1xui8>
      %127 = arith.divui %125, %c8_i64 : i64
      %128 = arith.addi %126, %127 : i64
      %129 = ascscalar.bind_gm_ptr %128 : memref<1xui8>
      %130 = ascscalar.bind_local_ptr %65 : memref<1xui8, 1>
      %131 = cannir.extract_stride %100 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
      %132 = cannir.extract_dim %131[0] : !cannir.stride<(-1, 1)>
      %133 = arith.cmpi ugt, %109, %c0_i64 : i64
      %134 = arith.cmpi ugt, %119, %c0_i64 : i64
      %135 = arith.andi %133, %134 : i1
      scf.if %135 {
        ascvec.get_buf mte3, %67, 0 {cannir.buffer.generated_sync}
        %181 = arith.trunci %109 : i64 to i32
        %182 = arith.trunci %119 : i64 to i32
        ascvec.copy_ub2gm %129[%c0], %130[%c0], %181, %182, %132, %c16384_i32 l2_cache_ctl %c0_i32 : memref<1xui8>, memref<1xui8, 1>
        ascvec.release_buf mte3, %67, 0 {cannir.buffer.generated_sync}
      }
      %136 = arith.remsi %25, %c32_i64 : i64
      %137 = arith.cmpi ne, %136, %c0_i64 : i64
      %138 = arith.cmpi slt, %136, %c0_i64 : i64
      %139 = arith.andi %137, %138 : i1
      %140 = arith.divsi %25, %c32_i64 : i64
      %141 = arith.extui %139 : i1 to i64
      %142 = arith.subi %140, %141 : i64
      %143 = cannir.extract_layout %arg2 : !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
      %144 = cannir.extract_shape %143 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, -1)>
      %145 = cannir.extract_dim %144[0] : !cannir.shape<(-1, -1)>
      %146 = arith.minsi %22, %145 : i64
      %147 = arith.subi %145, %146 : i64
      %148 = arith.minsi %147, %c1_i64 : i64
      %149 = arith.cmpi sle, %22, %145 : i64
      %150 = arith.select %149, %148, %c0_i64 : i64
      %151 = arith.cmpi sgt, %150, %c1_i64 : i64
      %152 = arith.select %151, %c1_i64, %150 : i64
      %153 = cannir.extract_dim %144[1] : !cannir.shape<(-1, -1)>
      %154 = arith.maxsi %142, %c1_i64 : i64
      %155 = arith.divsi %153, %154 : i64
      %156 = arith.minsi %24, %155 : i64
      %157 = arith.muli %156, %142 : i64
      %158 = arith.subi %153, %157 : i64
      %159 = arith.minsi %142, %158 : i64
      %160 = arith.cmpi sle, %24, %155 : i64
      %161 = arith.select %160, %159, %c0_i64 : i64
      %162 = arith.cmpi sgt, %161, %c1024_i64 : i64
      %163 = arith.select %162, %c1024_i64, %161 : i64
      %164 = cannir.extract_physical_layout %arg2 : !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
      %165 = cannir.extract_stride %164 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
      %166 = cannir.extract_dim %165[0] : !cannir.stride<(-1, 1)>
      %167 = arith.muli %22, %166 : i64
      %168 = arith.muli %24, %142 : i64
      %169 = arith.addi %167, %168 : i64
      %170 = arith.muli %169, %c8_i64 : i64
      %171 = ascscalar.memref_address %alloc_1 : memref<1xui8>
      %172 = arith.divui %170, %c8_i64 : i64
      %173 = arith.addi %171, %172 : i64
      %174 = ascscalar.bind_gm_ptr %173 : memref<1xui8>
      %175 = ascscalar.bind_local_ptr %70 : memref<1xui8, 1>
      %176 = cannir.extract_stride %143 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
      %177 = cannir.extract_dim %176[0] : !cannir.stride<(-1, 1)>
      %178 = arith.cmpi ugt, %152, %c0_i64 : i64
      %179 = arith.cmpi ugt, %163, %c0_i64 : i64
      %180 = arith.andi %178, %179 : i1
      scf.if %180 {
        ascvec.get_buf mte3, %72, 0 {cannir.buffer.generated_sync}
        %181 = arith.trunci %152 : i64 to i32
        %182 = arith.trunci %163 : i64 to i32
        ascvec.copy_ub2gm %174[%c0], %175[%c0], %181, %182, %177, %c1024_i32 l2_cache_ctl %c0_i32 : memref<1xui8>, memref<1xui8, 1>
        ascvec.release_buf mte3, %72, 0 {cannir.buffer.generated_sync}
      }
      scf.yield %33 : i64
    }
    return
  }
}
