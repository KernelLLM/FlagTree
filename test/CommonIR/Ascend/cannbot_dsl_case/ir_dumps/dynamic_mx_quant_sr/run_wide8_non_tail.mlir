module {
  func.func @run_wide8_non_tail_8768dfb40d5d1165cd94678ca098851b(%arg0: !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg1: !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg2: !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: !cannir.tensor<<gm, i64>, <<(2)>, <(1)>>, <<(2)>, <(1)>>, identity>) attributes {cannbotdsl.contract_digest = "8768dfb40d5d1165cd94678ca098851bb79b64da87af61debaf2a345c4afee3e", cannbotdsl.public_slot_count = 7 : i64, cannbotdsl.slot_dtype_bits = [16, 8, 8, 64, 64, 64, 64], cannbotdsl.slot_dtype_codes = [4, 1, 1, 0, 0, 0, 0], cannbotdsl.slot_dtype_lanes = [1, 1, 1, 1, 1, 1, 1], cannbotdsl.slot_flags = [1, 1, 1, 1, 1, 1, 1], cannbotdsl.slot_kinds = [1, 1, 1, 2, 2, 2, 1], cannbotdsl.slot_type_digests = ["5d7cdc96dc21b3b739e32aeccf58c1df113bec54701c75107ed650f2544a07da", "171ddd09eb070fb4a6413ae87b2ceecddc233a9a173a0404e96285a4bb8b8659", "891bcd21e721f8d1a666bcca4e8705ec63796c1add8ae000536d0d98f0274d3a", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "4126671a83041647cd98afed881383228865da287ea44573939f32b1d956765b"], cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.host} {
    cannir.kernel_launch @kernel__DynamicMxQuantSrKernel_run_wide8_non_tail_0(%arg5) [%arg0, %arg1, %arg2, %arg3, %arg4, %arg6] {cannbotdsl.profile = {indices = array<i64: 0, 1, 2, 5>, name = "_DynamicMxQuantSrKernel_run_wide8_non_tail", op_type = "_DynamicMxQuantSrKernel_run_wide8_non_tail", roles = array<i64: 0, 1, 1, 0>}} : !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, i64, i64, !cannir.tensor<<gm, i64>, <<(2)>, <(1)>>, <<(2)>, <(1)>>, identity>
    return
  }
  func.func @kernel__DynamicMxQuantSrKernel_run_wide8_non_tail_0(%arg0: !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg1: !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg2: !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg3: i64, %arg4: i64, %arg5: !cannir.tensor<<gm, i64>, <<(2)>, <(1)>>, <<(2)>, <(1)>>, identity>) attributes {cannir.buffer.auto_sync, cannir.buffer.sync_analyzed, cannir.buffer.sync_verified, cannir.buffer.tensor_state, cannir.core = "aiv", cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.device, cannir.explicit_ring_buffer_cursors, cannir.kernel_group = "kernel__DynamicMxQuantSrKernel_run_wide8_non_tail_0", cannir.resource.ids_prepared} {
    %c1024_i32 = arith.constant 1024 : i32
    %c16384_i32 = arith.constant 16384 : i32
    %c1 = arith.constant 1 : index
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
    %4 = arith.remsi %arg3, %c256_i64 : i64
    %5 = arith.cmpi ne, %4, %c0_i64 : i64
    %6 = arith.cmpi slt, %4, %c0_i64 : i64
    %7 = arith.andi %5, %6 : i1
    %8 = arith.divsi %arg3, %c256_i64 : i64
    %9 = arith.extui %7 : i1 to i64
    %10 = arith.subi %8, %9 : i64
    %11 = cannir.extract_layout %arg0 : !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
    %12 = cannir.extract_shape %11 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, -1)>
    %13 = cannir.extract_dim %12[0] : !cannir.shape<(-1, -1)>
    %14 = arith.muli %13, %10 : i64
    %15 = ascscalar.get_block_idx
    %16 = ascscalar.get_block_num
    %17 = scf.for %arg6 = %15 to %14 step %16 iter_args(%arg7 = %c0_i64) -> (i64)  : i64 {
      %18 = arith.remsi %arg6, %10 : i64
      %19 = arith.cmpi ne, %18, %c0_i64 : i64
      %20 = arith.cmpi slt, %18, %c0_i64 : i64
      %21 = arith.cmpi slt, %10, %c0_i64 : i64
      %22 = arith.xori %20, %21 : i1
      %23 = arith.andi %19, %22 : i1
      %24 = arith.divsi %arg6, %10 : i64
      %25 = arith.extui %23 : i1 to i64
      %26 = arith.subi %24, %25 : i64
      %27 = arith.select %23, %10, %c0_i64 : i64
      %28 = arith.addi %18, %27 : i64
      %29 = arith.muli %arg4, %c256_i64 : i64
      %30 = arith.cmpi eq, %arg7, %c0_i64 : i64
      %31 = ascscalar.memref_address %alloc_3 : memref<16384xbf16, 1>
      %32 = ascscalar.memref_address %alloc_4 : memref<16384xbf16, 1>
      %33 = arith.select %30, %31, %32 : i64
      %34 = arith.cmpi ne, %arg7, %c0_i64 : i64
      %35 = arith.extui %34 : i1 to i64
      %36 = arith.addi %arg7, %c1_i64 overflow<nsw> : i64
      %37 = arith.remui %36, %c2_i64 : i64
      %38 = arith.minsi %26, %13 : i64
      %39 = arith.subi %13, %38 : i64
      %40 = arith.minsi %39, %c1_i64 : i64
      %41 = arith.cmpi sle, %26, %13 : i64
      %42 = arith.select %41, %40, %c0_i64 : i64
      %43 = cannir.extract_physical_layout %arg0 : !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
      %44 = cannir.extract_stride %43 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
      %45 = cannir.extract_dim %44[0] : !cannir.stride<(-1, 1)>
      %46 = arith.muli %45, %c2_i64 : i64
      %47 = cannir.extract_dim %12[1] : !cannir.shape<(-1, -1)>
      %48 = arith.maxsi %29, %c1_i64 : i64
      %49 = arith.divsi %47, %48 : i64
      %50 = arith.minsi %28, %49 : i64
      %51 = arith.muli %50, %29 : i64
      %52 = arith.subi %47, %51 : i64
      %53 = arith.minsi %29, %52 : i64
      %54 = arith.cmpi sle, %28, %49 : i64
      %55 = arith.select %54, %53, %c0_i64 : i64
      %56 = arith.muli %26, %45 : i64
      %57 = arith.muli %56, %c2_i64 : i64
      %58 = arith.muli %28, %29 : i64
      %59 = arith.muli %58, %c2_i64 : i64
      %60 = arith.addi %57, %59 : i64
      %61 = arith.muli %55, %c2_i64 : i64
      %62 = arith.cmpi eq, %61, %c0_i64 : i64
      %63 = arith.select %62, %c0_i64, %42 : i64
      %64 = arith.select %62, %c2_i64, %61 : i64
      %65 = ascscalar.bind_local_ptr %33 : memref<1xbf16, 1>
      ascvec.get_buf mte2, %35, 0 {cannir.buffer.generated_sync}
      "ascvec.copy_gm2ub_plan"(%65, %alloc, %c0_i32, %c0_i32, %c0_i64, %60, %63, %64, %46, %c32768_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
      ascvec.release_buf mte2, %35, 0 {cannir.buffer.generated_sync}
      %66 = ascscalar.bind_local_ptr %33 : memref<16384xbf16, 1>
      %67 = ascscalar.memref_address %alloc_5 : memref<16384xui8, 1>
      %68 = ascscalar.memref_address %alloc_6 : memref<16384xui8, 1>
      %69 = arith.select %30, %67, %68 : i64
      %70 = ascscalar.bind_local_ptr %69 : memref<16384xui8, 1>
      %71 = arith.select %30, %c2_i64, %c3_i64 : i64
      %72 = ascscalar.memref_address %alloc_7 : memref<1024xui8, 1>
      %73 = ascscalar.memref_address %alloc_8 : memref<1024xui8, 1>
      %74 = arith.select %30, %72, %73 : i64
      %75 = ascscalar.bind_local_ptr %74 : memref<1024xui8, 1>
      %76 = arith.select %30, %c4_i64, %c5_i64 : i64
      %77 = arith.muli %26, %arg4 : i64
      %78 = arith.muli %28, %c256_i64 : i64
      %79 = arith.remsi %78, %c32_i64 : i64
      %80 = arith.cmpi ne, %79, %c0_i64 : i64
      %81 = arith.cmpi slt, %79, %c0_i64 : i64
      %82 = arith.andi %80, %81 : i1
      %83 = arith.divsi %78, %c32_i64 : i64
      %84 = arith.extui %82 : i1 to i64
      %85 = arith.subi %83, %84 : i64
      %86 = arith.remsi %arg3, %c32_i64 : i64
      %87 = arith.cmpi ne, %86, %c0_i64 : i64
      %88 = arith.cmpi slt, %86, %c0_i64 : i64
      %89 = arith.andi %87, %88 : i1
      %90 = arith.divsi %arg3, %c32_i64 : i64
      %91 = arith.extui %89 : i1 to i64
      %92 = arith.subi %90, %91 : i64
      %93 = arith.trunci %1 : i64 to i32
      %94 = arith.extui %93 : i32 to i64
      %95 = arith.subi %1, %94 : i64
      %96 = arith.remsi %95, %c4294967296_i64 : i64
      %97 = arith.cmpi ne, %96, %c0_i64 : i64
      %98 = arith.cmpi slt, %96, %c0_i64 : i64
      %99 = arith.andi %97, %98 : i1
      %100 = arith.divsi %95, %c4294967296_i64 : i64
      %101 = arith.extui %99 : i1 to i64
      %102 = arith.subi %100, %101 : i64
      %103 = arith.trunci %102 : i64 to i32
      ascvec.vf(%85, %77, %92, %3, %93, %103, %alloc_10 : i64, i64, i64, i64, i32, i32, memref<4096xi32, 1>) {
      ^bb0(%arg8: i64, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: i32, %arg13: i32, %arg14: memref<4096xi32, 1>):
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
        %c0_i64_11 = arith.constant 0 : i64
        %c4294967296_i64_12 = arith.constant 4294967296 : i64
        %c0_i32_13 = arith.constant 0 : i32
        %c16_i32 = arith.constant 16 : i32
        %mask, %remaining_out = ascvec.update_mask %c16_i32 : !ascvec.mask
        %185 = ascvec.arange %c0_i32_13 : i32 -> <ui32>
        scf.for %arg15 = %c0_i32_13 to %c8_i32 step %c1_i32  : i32 {
          %186 = arith.extsi %arg15 : i32 to i64
          %187 = arith.addi %arg8, %186 : i64
          %188 = arith.trunci %arg9 : i64 to i32
          %189 = arith.extui %188 : i32 to i64
          %190 = arith.subi %arg9, %189 : i64
          %191 = arith.remsi %190, %c4294967296_i64_12 : i64
          %192 = arith.cmpi ne, %191, %c0_i64_11 : i64
          %193 = arith.cmpi slt, %191, %c0_i64_11 : i64
          %194 = arith.andi %192, %193 : i1
          %195 = arith.divsi %190, %c4294967296_i64_12 : i64
          %196 = arith.extui %194 : i1 to i64
          %197 = arith.subi %195, %196 : i64
          %198 = arith.trunci %197 : i64 to i32
          %199 = arith.trunci %187 : i64 to i32
          %200 = arith.extui %199 : i32 to i64
          %201 = arith.subi %187, %200 : i64
          %202 = arith.remsi %201, %c4294967296_i64_12 : i64
          %203 = arith.cmpi ne, %202, %c0_i64_11 : i64
          %204 = arith.cmpi slt, %202, %c0_i64_11 : i64
          %205 = arith.andi %203, %204 : i1
          %206 = arith.divsi %201, %c4294967296_i64_12 : i64
          %207 = arith.extui %205 : i1 to i64
          %208 = arith.subi %206, %207 : i64
          %209 = arith.trunci %208 : i64 to i32
          %210 = ascvec.duplicate_scalar %188, %mask : i32 -> <ui32>
          %211 = ascvec.duplicate_scalar %198, %mask : i32 -> <ui32>
          %212 = ascvec.duplicate_scalar %c0_i32_13, %mask : i32 -> <ui32>
          %carry, %dst = ascvec.addc %210, %185, %mask : <ui32>, <ui32> -> !ascvec.mask, <ui32>
          %carry_14, %dst_15 = ascvec.addc %211, %212, %carry, %mask : <ui32>, <ui32> -> !ascvec.mask, <ui32>
          %213 = ascvec.duplicate_scalar %199, %mask : i32 -> <ui32>
          %214 = ascvec.duplicate_scalar %209, %mask : i32 -> <ui32>
          %215 = arith.trunci %arg10 : i64 to i32
          %216 = arith.extui %215 : i32 to i64
          %217 = arith.subi %arg10, %216 : i64
          %218 = arith.remsi %217, %c4294967296_i64_12 : i64
          %219 = arith.cmpi ne, %218, %c0_i64_11 : i64
          %220 = arith.cmpi slt, %218, %c0_i64_11 : i64
          %221 = arith.andi %219, %220 : i1
          %222 = arith.divsi %217, %c4294967296_i64_12 : i64
          %223 = arith.extui %221 : i1 to i64
          %224 = arith.subi %222, %223 : i64
          %225 = arith.trunci %224 : i64 to i32
          %226 = ascvec.duplicate_scalar %215, %mask : i32 -> <ui32>
          %dst0, %dst1 = ascvec.mull %dst, %226, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %227 = ascvec.mul_scalar %dst, %225, %mask : <ui32>, i32
          %228 = ascvec.add %dst1, %227, %mask : <ui32>
          %229 = ascvec.mul_scalar %dst_15, %215, %mask : <ui32>, i32
          %230 = ascvec.add %228, %229, %mask : <ui32>
          %carry_16, %dst_17 = ascvec.addc %dst0, %213, %mask : <ui32>, <ui32> -> !ascvec.mask, <ui32>
          %carry_18, %dst_19 = ascvec.addc %230, %214, %carry_16, %mask : <ui32>, <ui32> -> !ascvec.mask, <ui32>
          %231 = arith.trunci %arg11 : i64 to i32
          %232 = arith.extui %231 : i32 to i64
          %233 = arith.subi %arg11, %232 : i64
          %234 = arith.remsi %233, %c4294967296_i64_12 : i64
          %235 = arith.cmpi ne, %234, %c0_i64_11 : i64
          %236 = arith.cmpi slt, %234, %c0_i64_11 : i64
          %237 = arith.andi %235, %236 : i1
          %238 = arith.divsi %233, %c4294967296_i64_12 : i64
          %239 = arith.extui %237 : i1 to i64
          %240 = arith.subi %238, %239 : i64
          %241 = arith.trunci %240 : i64 to i32
          %242 = arith.divui %231, %c4_i32 : i32
          %243 = arith.remui %241, %c4_i32 : i32
          %244 = arith.muli %243, %c1073741824_i32 : i32
          %245 = arith.addi %242, %244 : i32
          %246 = arith.divui %241, %c4_i32 : i32
          %247 = ascvec.shr %dst_17, 14, %mask : <ui32> -> <ui32>
          %248 = ascvec.shl %dst_19, 18, %mask : <ui32> -> <ui32>
          %249 = ascvec.or %247, %248, %mask : <ui32>, <ui32> -> <ui32>
          %250 = ascvec.shr %dst_19, 14, %mask : <ui32> -> <ui32>
          %251 = ascvec.duplicate_scalar %245, %mask : i32 -> <ui32>
          %252 = ascvec.duplicate_scalar %246, %mask : i32 -> <ui32>
          %carry_20, %dst_21 = ascvec.addc %251, %249, %mask : <ui32>, <ui32> -> !ascvec.mask, <ui32>
          %carry_22, %dst_23 = ascvec.addc %252, %250, %carry_20, %mask : <ui32>, <ui32> -> !ascvec.mask, <ui32>
          %253 = ascvec.duplicate_scalar %c16383_i32, %mask : i32 -> <ui32>
          %254 = ascvec.and %dst_17, %253, %mask : <ui32>, <ui32> -> <ui32>
          %255 = ascvec.duplicate_scalar %c-766435501_i32, %mask : i32 -> <ui32>
          %256 = ascvec.duplicate_scalar %c-845247145_i32, %mask : i32 -> <ui32>
          %dst0_24, %dst1_25 = ascvec.mull %212, %255, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %dst0_26, %dst1_27 = ascvec.mull %dst_21, %256, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %257 = ascvec.xor %dst1_27, %254, %mask : <ui32>, <ui32> -> <ui32>
          %258 = ascvec.xor %dst1_25, %dst_23, %mask : <ui32>, <ui32> -> <ui32>
          %259 = ascvec.duplicate_scalar %arg12, %mask : i32 -> <ui32>
          %260 = ascvec.duplicate_scalar %arg13, %mask : i32 -> <ui32>
          %261 = ascvec.xor %257, %259, %mask : <ui32>, <ui32> -> <ui32>
          %262 = ascvec.xor %258, %260, %mask : <ui32>, <ui32> -> <ui32>
          %dst0_28, %dst1_29 = ascvec.mull %261, %255, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %dst0_30, %dst1_31 = ascvec.mull %262, %256, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %263 = ascvec.xor %dst1_31, %dst0_26, %mask : <ui32>, <ui32> -> <ui32>
          %264 = ascvec.xor %dst1_29, %dst0_24, %mask : <ui32>, <ui32> -> <ui32>
          %265 = arith.addi %arg12, %c-1640531527_i32 : i32
          %266 = ascvec.duplicate_scalar %265, %mask : i32 -> <ui32>
          %267 = arith.addi %arg13, %c-1150833019_i32 : i32
          %268 = ascvec.duplicate_scalar %267, %mask : i32 -> <ui32>
          %269 = ascvec.xor %263, %266, %mask : <ui32>, <ui32> -> <ui32>
          %270 = ascvec.xor %264, %268, %mask : <ui32>, <ui32> -> <ui32>
          %dst0_32, %dst1_33 = ascvec.mull %269, %255, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %dst0_34, %dst1_35 = ascvec.mull %270, %256, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %271 = ascvec.xor %dst1_35, %dst0_30, %mask : <ui32>, <ui32> -> <ui32>
          %272 = ascvec.xor %dst1_33, %dst0_28, %mask : <ui32>, <ui32> -> <ui32>
          %273 = arith.addi %arg12, %c1013904242_i32 : i32
          %274 = ascvec.duplicate_scalar %273, %mask : i32 -> <ui32>
          %275 = arith.addi %arg13, %c1993301258_i32 : i32
          %276 = ascvec.duplicate_scalar %275, %mask : i32 -> <ui32>
          %277 = ascvec.xor %271, %274, %mask : <ui32>, <ui32> -> <ui32>
          %278 = ascvec.xor %272, %276, %mask : <ui32>, <ui32> -> <ui32>
          %dst0_36, %dst1_37 = ascvec.mull %277, %255, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %dst0_38, %dst1_39 = ascvec.mull %278, %256, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %279 = ascvec.xor %dst1_39, %dst0_34, %mask : <ui32>, <ui32> -> <ui32>
          %280 = ascvec.xor %dst1_37, %dst0_32, %mask : <ui32>, <ui32> -> <ui32>
          %281 = arith.addi %arg12, %c-626627285_i32 : i32
          %282 = ascvec.duplicate_scalar %281, %mask : i32 -> <ui32>
          %283 = arith.addi %arg13, %c842468239_i32 : i32
          %284 = ascvec.duplicate_scalar %283, %mask : i32 -> <ui32>
          %285 = ascvec.xor %279, %282, %mask : <ui32>, <ui32> -> <ui32>
          %286 = ascvec.xor %280, %284, %mask : <ui32>, <ui32> -> <ui32>
          %dst0_40, %dst1_41 = ascvec.mull %285, %255, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %dst0_42, %dst1_43 = ascvec.mull %286, %256, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %287 = ascvec.xor %dst1_43, %dst0_38, %mask : <ui32>, <ui32> -> <ui32>
          %288 = ascvec.xor %dst1_41, %dst0_36, %mask : <ui32>, <ui32> -> <ui32>
          %289 = arith.addi %arg12, %c2027808484_i32 : i32
          %290 = ascvec.duplicate_scalar %289, %mask : i32 -> <ui32>
          %291 = arith.addi %arg13, %c-308364780_i32 : i32
          %292 = ascvec.duplicate_scalar %291, %mask : i32 -> <ui32>
          %293 = ascvec.xor %287, %290, %mask : <ui32>, <ui32> -> <ui32>
          %294 = ascvec.xor %288, %292, %mask : <ui32>, <ui32> -> <ui32>
          %dst0_44, %dst1_45 = ascvec.mull %293, %255, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %dst0_46, %dst1_47 = ascvec.mull %294, %256, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %295 = ascvec.xor %dst1_47, %dst0_42, %mask : <ui32>, <ui32> -> <ui32>
          %296 = ascvec.xor %dst1_45, %dst0_40, %mask : <ui32>, <ui32> -> <ui32>
          %297 = arith.addi %arg12, %c387276957_i32 : i32
          %298 = ascvec.duplicate_scalar %297, %mask : i32 -> <ui32>
          %299 = arith.addi %arg13, %c-1459197799_i32 : i32
          %300 = ascvec.duplicate_scalar %299, %mask : i32 -> <ui32>
          %301 = ascvec.xor %295, %298, %mask : <ui32>, <ui32> -> <ui32>
          %302 = ascvec.xor %296, %300, %mask : <ui32>, <ui32> -> <ui32>
          %dst0_48, %dst1_49 = ascvec.mull %301, %255, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %dst0_50, %dst1_51 = ascvec.mull %302, %256, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %303 = ascvec.xor %dst1_51, %dst0_46, %mask : <ui32>, <ui32> -> <ui32>
          %304 = ascvec.xor %dst1_49, %dst0_44, %mask : <ui32>, <ui32> -> <ui32>
          %305 = arith.addi %arg12, %c-1253254570_i32 : i32
          %306 = ascvec.duplicate_scalar %305, %mask : i32 -> <ui32>
          %307 = arith.addi %arg13, %c1684936478_i32 : i32
          %308 = ascvec.duplicate_scalar %307, %mask : i32 -> <ui32>
          %309 = ascvec.xor %303, %306, %mask : <ui32>, <ui32> -> <ui32>
          %310 = ascvec.xor %304, %308, %mask : <ui32>, <ui32> -> <ui32>
          %dst0_52, %dst1_53 = ascvec.mull %309, %255, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %dst0_54, %dst1_55 = ascvec.mull %310, %256, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %311 = ascvec.xor %dst1_55, %dst0_50, %mask : <ui32>, <ui32> -> <ui32>
          %312 = ascvec.xor %dst1_53, %dst0_48, %mask : <ui32>, <ui32> -> <ui32>
          %313 = arith.addi %arg12, %c1401181199_i32 : i32
          %314 = ascvec.duplicate_scalar %313, %mask : i32 -> <ui32>
          %315 = arith.addi %arg13, %c534103459_i32 : i32
          %316 = ascvec.duplicate_scalar %315, %mask : i32 -> <ui32>
          %317 = ascvec.xor %311, %314, %mask : <ui32>, <ui32> -> <ui32>
          %318 = ascvec.xor %312, %316, %mask : <ui32>, <ui32> -> <ui32>
          %dst0_56, %dst1_57 = ascvec.mull %317, %255, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %dst0_58, %dst1_59 = ascvec.mull %318, %256, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %319 = ascvec.xor %dst1_59, %dst0_54, %mask : <ui32>, <ui32> -> <ui32>
          %320 = ascvec.xor %dst1_57, %dst0_52, %mask : <ui32>, <ui32> -> <ui32>
          %321 = arith.addi %arg12, %c-239350328_i32 : i32
          %322 = ascvec.duplicate_scalar %321, %mask : i32 -> <ui32>
          %323 = arith.addi %arg13, %c-616729560_i32 : i32
          %324 = ascvec.duplicate_scalar %323, %mask : i32 -> <ui32>
          %325 = ascvec.xor %319, %322, %mask : <ui32>, <ui32> -> <ui32>
          %326 = ascvec.xor %320, %324, %mask : <ui32>, <ui32> -> <ui32>
          %dst0_60, %dst1_61 = ascvec.mull %325, %255, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %dst0_62, %dst1_63 = ascvec.mull %326, %256, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %327 = ascvec.xor %dst1_63, %dst0_58, %mask : <ui32>, <ui32> -> <ui32>
          %328 = ascvec.xor %dst1_61, %dst0_56, %mask : <ui32>, <ui32> -> <ui32>
          %329 = arith.addi %arg12, %c-1879881855_i32 : i32
          %330 = ascvec.duplicate_scalar %329, %mask : i32 -> <ui32>
          %331 = arith.addi %arg13, %c-1767562579_i32 : i32
          %332 = ascvec.duplicate_scalar %331, %mask : i32 -> <ui32>
          %333 = ascvec.xor %327, %330, %mask : <ui32>, <ui32> -> <ui32>
          %334 = ascvec.xor %328, %332, %mask : <ui32>, <ui32> -> <ui32>
          %335 = arith.muli %arg15, %c64_i32 overflow<nsw> : i32
          %336 = ascvec.reinterpret %333 : <ui32> -> <i32>
          %337 = arith.index_cast %335 : i32 to index
          ascvec.storealign %arg14[%337], %336, %mask : memref<4096xi32, 1>, index, <i32>
          %338 = arith.addi %335, %c16_i32 overflow<nsw> : i32
          %339 = ascvec.reinterpret %dst0_62 : <ui32> -> <i32>
          %340 = arith.index_cast %338 : i32 to index
          ascvec.storealign %arg14[%340], %339, %mask : memref<4096xi32, 1>, index, <i32>
          %341 = arith.addi %335, %c32_i32 overflow<nsw> : i32
          %342 = ascvec.reinterpret %334 : <ui32> -> <i32>
          %343 = arith.index_cast %341 : i32 to index
          ascvec.storealign %arg14[%343], %342, %mask : memref<4096xi32, 1>, index, <i32>
          %344 = arith.addi %335, %c48_i32 overflow<nsw> : i32
          %345 = ascvec.reinterpret %dst0_60 : <ui32> -> <i32>
          %346 = arith.index_cast %344 : i32 to index
          ascvec.storealign %arg14[%346], %345, %mask : memref<4096xi32, 1>, index, <i32>
        }
        ascvec.mem_bar vst_vld
      } {cannir.vf_mode = "simd"}
      ascvec.get_buf v, %71, 0 {cannir.buffer.generated_sync}
      ascvec.get_buf v, %35, 0 {cannir.buffer.generated_sync}
      ascvec.get_buf v, %76, 0 {cannir.buffer.generated_sync}
      ascvec.vf(%70, %arg4, %66, %alloc_9, %alloc_10, %75 : memref<16384xui8, 1>, i64, memref<16384xbf16, 1>, memref<512xui32, 1>, memref<4096xi32, 1>, memref<1024xui8, 1>) {
      ^bb0(%arg8: memref<16384xui8, 1>, %arg9: i64, %arg10: memref<16384xbf16, 1>, %arg11: memref<512xui32, 1>, %arg12: memref<4096xi32, 1>, %arg13: memref<1024xui8, 1>):
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
        %c0_i32_11 = arith.constant 0 : i32
        %c0_12 = arith.constant 0 : index
        %c16_i32 = arith.constant 16 : i32
        %mask, %remaining_out = ascvec.update_mask %c16_i32 : !ascvec.mask
        ascvec.store_unalign_begin %arg8 : memref<16384xui8, 1>
        scf.for %arg14 = %c0_i32_11 to %c8_i32 step %c1_i32  : i32 {
          %185 = ascvec.create_mask : !ascvec.mask
          %186 = ascvec.arange %c0_i32_11 : i32 -> <ui32>
          %187 = ascvec.duplicate_scalar %c0_i32_11, %185 : i32 -> <ui32>
          %188 = scf.for %arg15 = %c0_i32_11 to %c8_i32 step %c1_i32 iter_args(%arg16 = %187) -> (!ascvec.vreg<ui32>)  : i32 {
            %209 = arith.muli %arg14, %c32_i32 overflow<nsw> : i32
            %210 = arith.muli %arg15, %c4_i32 overflow<nsw> : i32
            %211 = arith.addi %209, %210 overflow<nsw> : i32
            %212 = arith.extsi %211 : i32 to i64
            %213 = arith.muli %212, %arg9 : i64
            %214 = arith.index_cast %213 : i64 to index
            %215 = ascvec.loadalign_unpack %arg10[%214] {mode = "b16_to_b32"} : memref<16384xbf16, 1>, index -> <bf16>
            %216 = ascvec.cast %215, %185 {rounding = 6 : i32} : <bf16> -> <f32>
            %217 = ascvec.reinterpret %216 : <f32> -> <ui32>
            %218 = ascvec.duplicate_scalar %c2147483647_i32, %185 : i32 -> <ui32>
            %219 = ascvec.and %217, %218, %185 : <ui32>, <ui32> -> <ui32>
            %220 = ascvec.max %arg16, %219, %185 : <ui32>
            scf.yield %220 : !ascvec.vreg<ui32>
          }
          %189 = ascvec.duplicate_scalar %c16_i32, %185 : i32 -> <ui32>
          %190 = ascvec.xor %186, %189, %185 : <ui32>, <ui32> -> <ui32>
          %191 = ascvec.gather_reg %188, %190 : <ui32>, <ui32> -> <ui32>
          %192 = ascvec.max %188, %191, %185 : <ui32>
          %193 = ascvec.duplicate_scalar %c32_i32, %185 : i32 -> <ui32>
          %194 = ascvec.xor %186, %193, %185 : <ui32>, <ui32> -> <ui32>
          %195 = ascvec.gather_reg %192, %194 : <ui32>, <ui32> -> <ui32>
          %196 = ascvec.max %192, %195, %185 : <ui32>
          %197 = ascvec.cmp_ge_scalar %196, %c2139095040_i32, %mask : <ui32>, i32 -> !ascvec.mask
          %198 = ascvec.add_scalar %196, %c-123731969_i32, %mask : <ui32>, i32
          %199 = ascvec.cmp_gt_scalar %196, %c132120577_i32, %mask : <ui32>, i32 -> !ascvec.mask
          %200 = ascvec.shr %198, 23, %199 : <ui32> -> <ui32>
          %201 = ascvec.duplicate_scalar %c255_i32, %197 merge %200 : !ascvec.vreg<ui32> {mode = "merging"} : i32 -> <ui32>
          %202 = ascvec.duplicate_scalar %c254_i32, %mask : i32 -> <ui32>
          %203 = ascvec.sub %202, %201, %mask : <ui32>
          %204 = ascvec.shl %203, 23, %mask : <ui32> -> <ui32>
          %205 = ascvec.duplicate_scalar %c2139160576_i32, %197 merge %204 : !ascvec.vreg<ui32> {mode = "merging"} : i32 -> <ui32>
          %206 = ascvec.reinterpret %205 : <ui32> -> <f32>
          %207 = arith.muli %arg14, %c16_i32 overflow<nsw> : i32
          %208 = arith.index_cast %207 : i32 to index
          ascvec.storealign %arg11[%208], %201, %mask : memref<512xui32, 1>, index, <ui32>
          scf.for %arg15 = %c0_i32_11 to %c4_i32 step %c1_i32  : i32 {
            %209 = arith.muli %arg14, %c64_i32 overflow<nsw> : i32
            %210 = arith.muli %arg15, %c16_i32 overflow<nsw> : i32
            %211 = arith.addi %209, %210 overflow<nsw> : i32
            %212 = arith.index_cast %211 : i32 to index
            %213 = ascvec.loadalign %arg12[%212] : memref<4096xi32, 1>, index -> <i32>
            %214 = ascvec.reinterpret %213 : <i32> -> <ui32>
            %215 = arith.muli %arg14, %c32_i32 overflow<nsw> : i32
            %216 = arith.muli %arg15, %c8_i32 overflow<nsw> : i32
            %217 = arith.addi %215, %216 overflow<nsw> : i32
            %218 = ascvec.shr %214, 21, %mask : <ui32> -> <ui32>
            %219 = ascvec.shl %214, 11, %mask : <ui32> -> <ui32>
            %220 = ascvec.or %218, %219, %mask : <ui32>, <ui32> -> <ui32>
            %221 = ascvec.shr %214, 29, %mask : <ui32> -> <ui32>
            %222 = ascvec.shl %214, 3, %mask : <ui32> -> <ui32>
            %223 = ascvec.or %221, %222, %mask : <ui32>, <ui32> -> <ui32>
            %224 = ascvec.shr %214, 5, %mask : <ui32> -> <ui32>
            %225 = ascvec.shl %214, 27, %mask : <ui32> -> <ui32>
            %226 = ascvec.or %224, %225, %mask : <ui32>, <ui32> -> <ui32>
            %227 = ascvec.shr %214, 13, %mask : <ui32> -> <ui32>
            %228 = ascvec.shl %214, 19, %mask : <ui32> -> <ui32>
            %229 = ascvec.or %227, %228, %mask : <ui32>, <ui32> -> <ui32>
            %230 = ascvec.shr %214, 24, %mask : <ui32> -> <ui32>
            %231 = ascvec.shl %214, 8, %mask : <ui32> -> <ui32>
            %232 = ascvec.or %230, %231, %mask : <ui32>, <ui32> -> <ui32>
            %233 = ascvec.shr %214, 8, %mask : <ui32> -> <ui32>
            %234 = ascvec.shl %214, 24, %mask : <ui32> -> <ui32>
            %235 = ascvec.or %233, %234, %mask : <ui32>, <ui32> -> <ui32>
            %236 = ascvec.shr %214, 16, %mask : <ui32> -> <ui32>
            %237 = ascvec.shl %214, 16, %mask : <ui32> -> <ui32>
            %238 = ascvec.or %236, %237, %mask : <ui32>, <ui32> -> <ui32>
            %239 = arith.extsi %217 : i32 to i64
            %240 = arith.muli %239, %arg9 : i64
            %241 = ascvec.duplicate_scalar %c65535_i32, %mask : i32 -> <ui32>
            %242 = ascvec.and %220, %241, %mask : <ui32>, <ui32> -> <ui32>
            %243 = arith.index_cast %240 : i64 to index
            %244 = ascvec.loadalign_unpack %arg10[%243] {mode = "b16_to_b32"} : memref<16384xbf16, 1>, index -> <bf16>
            %245 = ascvec.cast %244, %mask {rounding = 6 : i32} : <bf16> -> <f32>
            %246 = ascvec.mul %245, %206, %mask : <f32>
            %247 = ascvec.cast %246, %mask : <f32> -> <bf16>
            %248 = ascvec.cast %247, %mask {rounding = 6 : i32} : <bf16> -> <f32>
            %249 = ascvec.reinterpret %248 : <f32> -> <ui32>
            %250 = ascvec.shr %249, 24, %mask : <ui32> -> <ui32>
            %251 = ascvec.duplicate_scalar %c128_i32, %mask : i32 -> <ui32>
            %252 = ascvec.and %250, %251, %mask : <ui32>, <ui32> -> <ui32>
            %253 = ascvec.duplicate_scalar %c2147483647_i32, %mask : i32 -> <ui32>
            %254 = ascvec.and %249, %253, %mask : <ui32>, <ui32> -> <ui32>
            %255 = ascvec.shr %254, 5, %mask : <ui32> -> <ui32>
            %256 = ascvec.duplicate_scalar %c29360128_i32, %mask : i32 -> <ui32>
            %257 = ascvec.sub %255, %256, %mask : <ui32>
            %258 = ascvec.reinterpret %254 : <ui32> -> <f32>
            %259 = ascvec.mul_scalar %258, %cst, %mask : <f32>, f32
            %260 = ascvec.cast %259, %mask {rounding = 4 : i32} : <f32> -> <i32>
            %261 = ascvec.reinterpret %260 : <i32> -> <ui32>
            %262 = ascvec.cmp_ge_scalar %254, %c947912704_i32, %mask : <ui32>, i32 -> !ascvec.mask
            %263 = ascvec.select %257, %261, %262 : <ui32>
            %264 = ascvec.and %242, %241, %mask : <ui32>, <ui32> -> <ui32>
            %265 = ascvec.add %263, %264, %mask : <ui32>
            %266 = ascvec.shr %265, 16, %mask : <ui32> -> <ui32>
            %267 = ascvec.min_scalar %266, %c123_i32, %mask : <ui32>, i32
            %268 = ascvec.or %267, %252, %mask : <ui32>, <ui32> -> <ui32>
            %269 = ascvec.duplicate_scalar %c127_i32, %mask : i32 -> <ui32>
            %270 = ascvec.cmp_gt_scalar %254, %c2139095040_i32, %mask : <ui32>, i32 -> !ascvec.mask
            %271 = ascvec.select %269, %268, %270 : <ui32>
            %272 = ascvec.pack %271 src_elem_bits 32 dst_elem_bits 16 {part = "low"} : <ui32> -> <ui16>
            %273 = ascvec.pack %272 src_elem_bits 16 dst_elem_bits 8 {part = "low"} : <ui16> -> <ui8>
            %mask_19, %remaining_out_20 = ascvec.update_mask %c16_i32 elem_bits 8 : !ascvec.mask
            %274 = ascvec.squeeze_store_reg %273, %mask_19 : <ui8>, !ascvec.mask -> <ui8>
            ascvec.store_unalign %arg8[%c0_12], %274 : memref<16384xui8, 1>, <ui8>
            %275 = arith.addi %217, %c1_i32 overflow<nsw> : i32
            %276 = arith.extsi %275 : i32 to i64
            %277 = arith.muli %276, %arg9 : i64
            %278 = ascvec.and %232, %241, %mask : <ui32>, <ui32> -> <ui32>
            %279 = arith.index_cast %277 : i64 to index
            %280 = ascvec.loadalign_unpack %arg10[%279] {mode = "b16_to_b32"} : memref<16384xbf16, 1>, index -> <bf16>
            %281 = ascvec.cast %280, %mask {rounding = 6 : i32} : <bf16> -> <f32>
            %282 = ascvec.mul %281, %206, %mask : <f32>
            %283 = ascvec.cast %282, %mask : <f32> -> <bf16>
            %284 = ascvec.cast %283, %mask {rounding = 6 : i32} : <bf16> -> <f32>
            %285 = ascvec.reinterpret %284 : <f32> -> <ui32>
            %286 = ascvec.shr %285, 24, %mask : <ui32> -> <ui32>
            %287 = ascvec.and %286, %251, %mask : <ui32>, <ui32> -> <ui32>
            %288 = ascvec.and %285, %253, %mask : <ui32>, <ui32> -> <ui32>
            %289 = ascvec.shr %288, 5, %mask : <ui32> -> <ui32>
            %290 = ascvec.sub %289, %256, %mask : <ui32>
            %291 = ascvec.reinterpret %288 : <ui32> -> <f32>
            %292 = ascvec.mul_scalar %291, %cst, %mask : <f32>, f32
            %293 = ascvec.cast %292, %mask {rounding = 4 : i32} : <f32> -> <i32>
            %294 = ascvec.reinterpret %293 : <i32> -> <ui32>
            %295 = ascvec.cmp_ge_scalar %288, %c947912704_i32, %mask : <ui32>, i32 -> !ascvec.mask
            %296 = ascvec.select %290, %294, %295 : <ui32>
            %297 = ascvec.and %278, %241, %mask : <ui32>, <ui32> -> <ui32>
            %298 = ascvec.add %296, %297, %mask : <ui32>
            %299 = ascvec.shr %298, 16, %mask : <ui32> -> <ui32>
            %300 = ascvec.min_scalar %299, %c123_i32, %mask : <ui32>, i32
            %301 = ascvec.or %300, %287, %mask : <ui32>, <ui32> -> <ui32>
            %302 = ascvec.cmp_gt_scalar %288, %c2139095040_i32, %mask : <ui32>, i32 -> !ascvec.mask
            %303 = ascvec.select %269, %301, %302 : <ui32>
            %304 = ascvec.pack %303 src_elem_bits 32 dst_elem_bits 16 {part = "low"} : <ui32> -> <ui16>
            %305 = ascvec.pack %304 src_elem_bits 16 dst_elem_bits 8 {part = "low"} : <ui16> -> <ui8>
            %mask_21, %remaining_out_22 = ascvec.update_mask %c16_i32 elem_bits 8 : !ascvec.mask
            %306 = ascvec.squeeze_store_reg %305, %mask_21 : <ui8>, !ascvec.mask -> <ui8>
            ascvec.store_unalign %arg8[%c0_12], %306 : memref<16384xui8, 1>, <ui8>
            %307 = arith.addi %217, %c2_i32 overflow<nsw> : i32
            %308 = arith.extsi %307 : i32 to i64
            %309 = arith.muli %308, %arg9 : i64
            %310 = ascvec.and %223, %241, %mask : <ui32>, <ui32> -> <ui32>
            %311 = arith.index_cast %309 : i64 to index
            %312 = ascvec.loadalign_unpack %arg10[%311] {mode = "b16_to_b32"} : memref<16384xbf16, 1>, index -> <bf16>
            %313 = ascvec.cast %312, %mask {rounding = 6 : i32} : <bf16> -> <f32>
            %314 = ascvec.mul %313, %206, %mask : <f32>
            %315 = ascvec.cast %314, %mask : <f32> -> <bf16>
            %316 = ascvec.cast %315, %mask {rounding = 6 : i32} : <bf16> -> <f32>
            %317 = ascvec.reinterpret %316 : <f32> -> <ui32>
            %318 = ascvec.shr %317, 24, %mask : <ui32> -> <ui32>
            %319 = ascvec.and %318, %251, %mask : <ui32>, <ui32> -> <ui32>
            %320 = ascvec.and %317, %253, %mask : <ui32>, <ui32> -> <ui32>
            %321 = ascvec.shr %320, 5, %mask : <ui32> -> <ui32>
            %322 = ascvec.sub %321, %256, %mask : <ui32>
            %323 = ascvec.reinterpret %320 : <ui32> -> <f32>
            %324 = ascvec.mul_scalar %323, %cst, %mask : <f32>, f32
            %325 = ascvec.cast %324, %mask {rounding = 4 : i32} : <f32> -> <i32>
            %326 = ascvec.reinterpret %325 : <i32> -> <ui32>
            %327 = ascvec.cmp_ge_scalar %320, %c947912704_i32, %mask : <ui32>, i32 -> !ascvec.mask
            %328 = ascvec.select %322, %326, %327 : <ui32>
            %329 = ascvec.and %310, %241, %mask : <ui32>, <ui32> -> <ui32>
            %330 = ascvec.add %328, %329, %mask : <ui32>
            %331 = ascvec.shr %330, 16, %mask : <ui32> -> <ui32>
            %332 = ascvec.min_scalar %331, %c123_i32, %mask : <ui32>, i32
            %333 = ascvec.or %332, %319, %mask : <ui32>, <ui32> -> <ui32>
            %334 = ascvec.cmp_gt_scalar %320, %c2139095040_i32, %mask : <ui32>, i32 -> !ascvec.mask
            %335 = ascvec.select %269, %333, %334 : <ui32>
            %336 = ascvec.pack %335 src_elem_bits 32 dst_elem_bits 16 {part = "low"} : <ui32> -> <ui16>
            %337 = ascvec.pack %336 src_elem_bits 16 dst_elem_bits 8 {part = "low"} : <ui16> -> <ui8>
            %mask_23, %remaining_out_24 = ascvec.update_mask %c16_i32 elem_bits 8 : !ascvec.mask
            %338 = ascvec.squeeze_store_reg %337, %mask_23 : <ui8>, !ascvec.mask -> <ui8>
            ascvec.store_unalign %arg8[%c0_12], %338 : memref<16384xui8, 1>, <ui8>
            %339 = arith.addi %217, %c3_i32 overflow<nsw> : i32
            %340 = arith.extsi %339 : i32 to i64
            %341 = arith.muli %340, %arg9 : i64
            %342 = ascvec.and %214, %241, %mask : <ui32>, <ui32> -> <ui32>
            %343 = arith.index_cast %341 : i64 to index
            %344 = ascvec.loadalign_unpack %arg10[%343] {mode = "b16_to_b32"} : memref<16384xbf16, 1>, index -> <bf16>
            %345 = ascvec.cast %344, %mask {rounding = 6 : i32} : <bf16> -> <f32>
            %346 = ascvec.mul %345, %206, %mask : <f32>
            %347 = ascvec.cast %346, %mask : <f32> -> <bf16>
            %348 = ascvec.cast %347, %mask {rounding = 6 : i32} : <bf16> -> <f32>
            %349 = ascvec.reinterpret %348 : <f32> -> <ui32>
            %350 = ascvec.shr %349, 24, %mask : <ui32> -> <ui32>
            %351 = ascvec.and %350, %251, %mask : <ui32>, <ui32> -> <ui32>
            %352 = ascvec.and %349, %253, %mask : <ui32>, <ui32> -> <ui32>
            %353 = ascvec.shr %352, 5, %mask : <ui32> -> <ui32>
            %354 = ascvec.sub %353, %256, %mask : <ui32>
            %355 = ascvec.reinterpret %352 : <ui32> -> <f32>
            %356 = ascvec.mul_scalar %355, %cst, %mask : <f32>, f32
            %357 = ascvec.cast %356, %mask {rounding = 4 : i32} : <f32> -> <i32>
            %358 = ascvec.reinterpret %357 : <i32> -> <ui32>
            %359 = ascvec.cmp_ge_scalar %352, %c947912704_i32, %mask : <ui32>, i32 -> !ascvec.mask
            %360 = ascvec.select %354, %358, %359 : <ui32>
            %361 = ascvec.and %342, %241, %mask : <ui32>, <ui32> -> <ui32>
            %362 = ascvec.add %360, %361, %mask : <ui32>
            %363 = ascvec.shr %362, 16, %mask : <ui32> -> <ui32>
            %364 = ascvec.min_scalar %363, %c123_i32, %mask : <ui32>, i32
            %365 = ascvec.or %364, %351, %mask : <ui32>, <ui32> -> <ui32>
            %366 = ascvec.cmp_gt_scalar %352, %c2139095040_i32, %mask : <ui32>, i32 -> !ascvec.mask
            %367 = ascvec.select %269, %365, %366 : <ui32>
            %368 = ascvec.pack %367 src_elem_bits 32 dst_elem_bits 16 {part = "low"} : <ui32> -> <ui16>
            %369 = ascvec.pack %368 src_elem_bits 16 dst_elem_bits 8 {part = "low"} : <ui16> -> <ui8>
            %mask_25, %remaining_out_26 = ascvec.update_mask %c16_i32 elem_bits 8 : !ascvec.mask
            %370 = ascvec.squeeze_store_reg %369, %mask_25 : <ui8>, !ascvec.mask -> <ui8>
            ascvec.store_unalign %arg8[%c0_12], %370 : memref<16384xui8, 1>, <ui8>
            %371 = arith.addi %217, %c4_i32 overflow<nsw> : i32
            %372 = arith.extsi %371 : i32 to i64
            %373 = arith.muli %372, %arg9 : i64
            %374 = ascvec.and %226, %241, %mask : <ui32>, <ui32> -> <ui32>
            %375 = arith.index_cast %373 : i64 to index
            %376 = ascvec.loadalign_unpack %arg10[%375] {mode = "b16_to_b32"} : memref<16384xbf16, 1>, index -> <bf16>
            %377 = ascvec.cast %376, %mask {rounding = 6 : i32} : <bf16> -> <f32>
            %378 = ascvec.mul %377, %206, %mask : <f32>
            %379 = ascvec.cast %378, %mask : <f32> -> <bf16>
            %380 = ascvec.cast %379, %mask {rounding = 6 : i32} : <bf16> -> <f32>
            %381 = ascvec.reinterpret %380 : <f32> -> <ui32>
            %382 = ascvec.shr %381, 24, %mask : <ui32> -> <ui32>
            %383 = ascvec.and %382, %251, %mask : <ui32>, <ui32> -> <ui32>
            %384 = ascvec.and %381, %253, %mask : <ui32>, <ui32> -> <ui32>
            %385 = ascvec.shr %384, 5, %mask : <ui32> -> <ui32>
            %386 = ascvec.sub %385, %256, %mask : <ui32>
            %387 = ascvec.reinterpret %384 : <ui32> -> <f32>
            %388 = ascvec.mul_scalar %387, %cst, %mask : <f32>, f32
            %389 = ascvec.cast %388, %mask {rounding = 4 : i32} : <f32> -> <i32>
            %390 = ascvec.reinterpret %389 : <i32> -> <ui32>
            %391 = ascvec.cmp_ge_scalar %384, %c947912704_i32, %mask : <ui32>, i32 -> !ascvec.mask
            %392 = ascvec.select %386, %390, %391 : <ui32>
            %393 = ascvec.and %374, %241, %mask : <ui32>, <ui32> -> <ui32>
            %394 = ascvec.add %392, %393, %mask : <ui32>
            %395 = ascvec.shr %394, 16, %mask : <ui32> -> <ui32>
            %396 = ascvec.min_scalar %395, %c123_i32, %mask : <ui32>, i32
            %397 = ascvec.or %396, %383, %mask : <ui32>, <ui32> -> <ui32>
            %398 = ascvec.cmp_gt_scalar %384, %c2139095040_i32, %mask : <ui32>, i32 -> !ascvec.mask
            %399 = ascvec.select %269, %397, %398 : <ui32>
            %400 = ascvec.pack %399 src_elem_bits 32 dst_elem_bits 16 {part = "low"} : <ui32> -> <ui16>
            %401 = ascvec.pack %400 src_elem_bits 16 dst_elem_bits 8 {part = "low"} : <ui16> -> <ui8>
            %mask_27, %remaining_out_28 = ascvec.update_mask %c16_i32 elem_bits 8 : !ascvec.mask
            %402 = ascvec.squeeze_store_reg %401, %mask_27 : <ui8>, !ascvec.mask -> <ui8>
            ascvec.store_unalign %arg8[%c0_12], %402 : memref<16384xui8, 1>, <ui8>
            %403 = arith.addi %217, %c5_i32 overflow<nsw> : i32
            %404 = arith.extsi %403 : i32 to i64
            %405 = arith.muli %404, %arg9 : i64
            %406 = ascvec.and %235, %241, %mask : <ui32>, <ui32> -> <ui32>
            %407 = arith.index_cast %405 : i64 to index
            %408 = ascvec.loadalign_unpack %arg10[%407] {mode = "b16_to_b32"} : memref<16384xbf16, 1>, index -> <bf16>
            %409 = ascvec.cast %408, %mask {rounding = 6 : i32} : <bf16> -> <f32>
            %410 = ascvec.mul %409, %206, %mask : <f32>
            %411 = ascvec.cast %410, %mask : <f32> -> <bf16>
            %412 = ascvec.cast %411, %mask {rounding = 6 : i32} : <bf16> -> <f32>
            %413 = ascvec.reinterpret %412 : <f32> -> <ui32>
            %414 = ascvec.shr %413, 24, %mask : <ui32> -> <ui32>
            %415 = ascvec.and %414, %251, %mask : <ui32>, <ui32> -> <ui32>
            %416 = ascvec.and %413, %253, %mask : <ui32>, <ui32> -> <ui32>
            %417 = ascvec.shr %416, 5, %mask : <ui32> -> <ui32>
            %418 = ascvec.sub %417, %256, %mask : <ui32>
            %419 = ascvec.reinterpret %416 : <ui32> -> <f32>
            %420 = ascvec.mul_scalar %419, %cst, %mask : <f32>, f32
            %421 = ascvec.cast %420, %mask {rounding = 4 : i32} : <f32> -> <i32>
            %422 = ascvec.reinterpret %421 : <i32> -> <ui32>
            %423 = ascvec.cmp_ge_scalar %416, %c947912704_i32, %mask : <ui32>, i32 -> !ascvec.mask
            %424 = ascvec.select %418, %422, %423 : <ui32>
            %425 = ascvec.and %406, %241, %mask : <ui32>, <ui32> -> <ui32>
            %426 = ascvec.add %424, %425, %mask : <ui32>
            %427 = ascvec.shr %426, 16, %mask : <ui32> -> <ui32>
            %428 = ascvec.min_scalar %427, %c123_i32, %mask : <ui32>, i32
            %429 = ascvec.or %428, %415, %mask : <ui32>, <ui32> -> <ui32>
            %430 = ascvec.cmp_gt_scalar %416, %c2139095040_i32, %mask : <ui32>, i32 -> !ascvec.mask
            %431 = ascvec.select %269, %429, %430 : <ui32>
            %432 = ascvec.pack %431 src_elem_bits 32 dst_elem_bits 16 {part = "low"} : <ui32> -> <ui16>
            %433 = ascvec.pack %432 src_elem_bits 16 dst_elem_bits 8 {part = "low"} : <ui16> -> <ui8>
            %mask_29, %remaining_out_30 = ascvec.update_mask %c16_i32 elem_bits 8 : !ascvec.mask
            %434 = ascvec.squeeze_store_reg %433, %mask_29 : <ui8>, !ascvec.mask -> <ui8>
            ascvec.store_unalign %arg8[%c0_12], %434 : memref<16384xui8, 1>, <ui8>
            %435 = arith.addi %217, %c6_i32 overflow<nsw> : i32
            %436 = arith.extsi %435 : i32 to i64
            %437 = arith.muli %436, %arg9 : i64
            %438 = ascvec.and %229, %241, %mask : <ui32>, <ui32> -> <ui32>
            %439 = arith.index_cast %437 : i64 to index
            %440 = ascvec.loadalign_unpack %arg10[%439] {mode = "b16_to_b32"} : memref<16384xbf16, 1>, index -> <bf16>
            %441 = ascvec.cast %440, %mask {rounding = 6 : i32} : <bf16> -> <f32>
            %442 = ascvec.mul %441, %206, %mask : <f32>
            %443 = ascvec.cast %442, %mask : <f32> -> <bf16>
            %444 = ascvec.cast %443, %mask {rounding = 6 : i32} : <bf16> -> <f32>
            %445 = ascvec.reinterpret %444 : <f32> -> <ui32>
            %446 = ascvec.shr %445, 24, %mask : <ui32> -> <ui32>
            %447 = ascvec.and %446, %251, %mask : <ui32>, <ui32> -> <ui32>
            %448 = ascvec.and %445, %253, %mask : <ui32>, <ui32> -> <ui32>
            %449 = ascvec.shr %448, 5, %mask : <ui32> -> <ui32>
            %450 = ascvec.sub %449, %256, %mask : <ui32>
            %451 = ascvec.reinterpret %448 : <ui32> -> <f32>
            %452 = ascvec.mul_scalar %451, %cst, %mask : <f32>, f32
            %453 = ascvec.cast %452, %mask {rounding = 4 : i32} : <f32> -> <i32>
            %454 = ascvec.reinterpret %453 : <i32> -> <ui32>
            %455 = ascvec.cmp_ge_scalar %448, %c947912704_i32, %mask : <ui32>, i32 -> !ascvec.mask
            %456 = ascvec.select %450, %454, %455 : <ui32>
            %457 = ascvec.and %438, %241, %mask : <ui32>, <ui32> -> <ui32>
            %458 = ascvec.add %456, %457, %mask : <ui32>
            %459 = ascvec.shr %458, 16, %mask : <ui32> -> <ui32>
            %460 = ascvec.min_scalar %459, %c123_i32, %mask : <ui32>, i32
            %461 = ascvec.or %460, %447, %mask : <ui32>, <ui32> -> <ui32>
            %462 = ascvec.cmp_gt_scalar %448, %c2139095040_i32, %mask : <ui32>, i32 -> !ascvec.mask
            %463 = ascvec.select %269, %461, %462 : <ui32>
            %464 = ascvec.pack %463 src_elem_bits 32 dst_elem_bits 16 {part = "low"} : <ui32> -> <ui16>
            %465 = ascvec.pack %464 src_elem_bits 16 dst_elem_bits 8 {part = "low"} : <ui16> -> <ui8>
            %mask_31, %remaining_out_32 = ascvec.update_mask %c16_i32 elem_bits 8 : !ascvec.mask
            %466 = ascvec.squeeze_store_reg %465, %mask_31 : <ui8>, !ascvec.mask -> <ui8>
            ascvec.store_unalign %arg8[%c0_12], %466 : memref<16384xui8, 1>, <ui8>
            %467 = arith.addi %217, %c7_i32 overflow<nsw> : i32
            %468 = arith.extsi %467 : i32 to i64
            %469 = arith.muli %468, %arg9 : i64
            %470 = ascvec.and %238, %241, %mask : <ui32>, <ui32> -> <ui32>
            %471 = arith.index_cast %469 : i64 to index
            %472 = ascvec.loadalign_unpack %arg10[%471] {mode = "b16_to_b32"} : memref<16384xbf16, 1>, index -> <bf16>
            %473 = ascvec.cast %472, %mask {rounding = 6 : i32} : <bf16> -> <f32>
            %474 = ascvec.mul %473, %206, %mask : <f32>
            %475 = ascvec.cast %474, %mask : <f32> -> <bf16>
            %476 = ascvec.cast %475, %mask {rounding = 6 : i32} : <bf16> -> <f32>
            %477 = ascvec.reinterpret %476 : <f32> -> <ui32>
            %478 = ascvec.shr %477, 24, %mask : <ui32> -> <ui32>
            %479 = ascvec.and %478, %251, %mask : <ui32>, <ui32> -> <ui32>
            %480 = ascvec.and %477, %253, %mask : <ui32>, <ui32> -> <ui32>
            %481 = ascvec.shr %480, 5, %mask : <ui32> -> <ui32>
            %482 = ascvec.sub %481, %256, %mask : <ui32>
            %483 = ascvec.reinterpret %480 : <ui32> -> <f32>
            %484 = ascvec.mul_scalar %483, %cst, %mask : <f32>, f32
            %485 = ascvec.cast %484, %mask {rounding = 4 : i32} : <f32> -> <i32>
            %486 = ascvec.reinterpret %485 : <i32> -> <ui32>
            %487 = ascvec.cmp_ge_scalar %480, %c947912704_i32, %mask : <ui32>, i32 -> !ascvec.mask
            %488 = ascvec.select %482, %486, %487 : <ui32>
            %489 = ascvec.and %470, %241, %mask : <ui32>, <ui32> -> <ui32>
            %490 = ascvec.add %488, %489, %mask : <ui32>
            %491 = ascvec.shr %490, 16, %mask : <ui32> -> <ui32>
            %492 = ascvec.min_scalar %491, %c123_i32, %mask : <ui32>, i32
            %493 = ascvec.or %492, %479, %mask : <ui32>, <ui32> -> <ui32>
            %494 = ascvec.cmp_gt_scalar %480, %c2139095040_i32, %mask : <ui32>, i32 -> !ascvec.mask
            %495 = ascvec.select %269, %493, %494 : <ui32>
            %496 = ascvec.pack %495 src_elem_bits 32 dst_elem_bits 16 {part = "low"} : <ui32> -> <ui16>
            %497 = ascvec.pack %496 src_elem_bits 16 dst_elem_bits 8 {part = "low"} : <ui16> -> <ui8>
            %mask_33, %remaining_out_34 = ascvec.update_mask %c16_i32 elem_bits 8 : !ascvec.mask
            %498 = ascvec.squeeze_store_reg %497, %mask_33 : <ui8>, !ascvec.mask -> <ui8>
            ascvec.store_unalign %arg8[%c0_12], %498 : memref<16384xui8, 1>, <ui8>
          }
        }
        %c0_13 = arith.constant 0 : index
        %c1_14 = arith.constant 1 : index
        scf.for %arg14 = %c0_13 to %c1_14 step %c1_14 {
          ascvec.store_unalign_post %arg8[%c0_12] : memref<16384xui8, 1>
        }
        ascvec.mem_bar vst_vld
        %mask_15, %remaining_out_16 = ascvec.update_mask %c32_i32 elem_bits 8 : !ascvec.mask
        ascvec.store_unalign_begin %arg13 : memref<1024xui8, 1>
        scf.for %arg14 = %c0_i32_11 to %c4_i32 step %c1_i32  : i32 {
          %185 = arith.muli %arg14, %c32_i32 overflow<nsw> : i32
          %186 = arith.index_cast %185 : i32 to index
          %187 = ascvec.loadalign %arg11[%186] : memref<512xui32, 1>, index -> <ui32>
          %188 = arith.addi %185, %c16_i32 overflow<nsw> : i32
          %189 = arith.index_cast %188 : i32 to index
          %190 = ascvec.loadalign %arg11[%189] : memref<512xui32, 1>, index -> <ui32>
          %191 = ascvec.shl %190, 8, %mask : <ui32> -> <ui32>
          %192 = ascvec.or %187, %191, %mask : <ui32>, <ui32> -> <ui32>
          %193 = ascvec.pack %192 src_elem_bits 32 dst_elem_bits 16 {part = "low"} : <ui32> -> <ui16>
          %194 = ascvec.reinterpret %193 : <ui16> -> <ui8>
          %195 = ascvec.squeeze_store_reg %194, %mask_15 : <ui8>, !ascvec.mask -> <ui8>
          ascvec.store_unalign %arg13[%c0_12], %195 : memref<1024xui8, 1>, <ui8>
        }
        %c0_17 = arith.constant 0 : index
        %c1_18 = arith.constant 1 : index
        scf.for %arg14 = %c0_17 to %c1_18 step %c1_18 {
          ascvec.store_unalign_post %arg13[%c0_12] : memref<1024xui8, 1>
        }
        ascvec.mem_bar vst_vld
      } {cannir.vf_mode = "simd"}
      ascvec.release_buf v, %76, 0 {cannir.buffer.generated_sync}
      ascvec.release_buf v, %35, 0 {cannir.buffer.generated_sync}
      ascvec.release_buf v, %71, 0 {cannir.buffer.generated_sync}
      %104 = cannir.extract_layout %arg1 : !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
      %105 = cannir.extract_shape %104 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, -1)>
      %106 = cannir.extract_dim %105[0] : !cannir.shape<(-1, -1)>
      %107 = arith.minsi %26, %106 : i64
      %108 = arith.subi %106, %107 : i64
      %109 = arith.minsi %108, %c1_i64 : i64
      %110 = arith.cmpi sle, %26, %106 : i64
      %111 = arith.select %110, %109, %c0_i64 : i64
      %112 = arith.cmpi sgt, %111, %c1_i64 : i64
      %113 = arith.select %112, %c1_i64, %111 : i64
      %114 = cannir.extract_dim %105[1] : !cannir.shape<(-1, -1)>
      %115 = arith.divsi %114, %48 : i64
      %116 = arith.minsi %28, %115 : i64
      %117 = arith.muli %116, %29 : i64
      %118 = arith.subi %114, %117 : i64
      %119 = arith.minsi %29, %118 : i64
      %120 = arith.cmpi sle, %28, %115 : i64
      %121 = arith.select %120, %119, %c0_i64 : i64
      %122 = arith.cmpi sgt, %121, %c16384_i64 : i64
      %123 = arith.select %122, %c16384_i64, %121 : i64
      %124 = cannir.extract_physical_layout %arg1 : !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
      %125 = cannir.extract_stride %124 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
      %126 = cannir.extract_dim %125[0] : !cannir.stride<(-1, 1)>
      %127 = arith.muli %26, %126 : i64
      %128 = arith.addi %127, %58 : i64
      %129 = arith.muli %128, %c8_i64 : i64
      %130 = ascscalar.memref_address %alloc_0 : memref<1xui8>
      %131 = arith.divui %129, %c8_i64 : i64
      %132 = arith.addi %130, %131 : i64
      %133 = ascscalar.bind_gm_ptr %132 : memref<1xui8>
      %134 = ascscalar.bind_local_ptr %69 : memref<1xui8, 1>
      %135 = cannir.extract_stride %104 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
      %136 = cannir.extract_dim %135[0] : !cannir.stride<(-1, 1)>
      %137 = arith.cmpi ugt, %113, %c0_i64 : i64
      %138 = arith.cmpi ugt, %123, %c0_i64 : i64
      %139 = arith.andi %137, %138 : i1
      scf.if %139 {
        ascvec.get_buf mte3, %71, 0 {cannir.buffer.generated_sync}
        %185 = arith.trunci %113 : i64 to i32
        %186 = arith.trunci %123 : i64 to i32
        ascvec.copy_ub2gm %133[%c0], %134[%c0], %185, %186, %136, %c16384_i32 l2_cache_ctl %c0_i32 : memref<1xui8>, memref<1xui8, 1>
        ascvec.release_buf mte3, %71, 0 {cannir.buffer.generated_sync}
      }
      %140 = arith.remsi %29, %c32_i64 : i64
      %141 = arith.cmpi ne, %140, %c0_i64 : i64
      %142 = arith.cmpi slt, %140, %c0_i64 : i64
      %143 = arith.andi %141, %142 : i1
      %144 = arith.divsi %29, %c32_i64 : i64
      %145 = arith.extui %143 : i1 to i64
      %146 = arith.subi %144, %145 : i64
      %147 = cannir.extract_layout %arg2 : !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
      %148 = cannir.extract_shape %147 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, -1)>
      %149 = cannir.extract_dim %148[0] : !cannir.shape<(-1, -1)>
      %150 = arith.minsi %26, %149 : i64
      %151 = arith.subi %149, %150 : i64
      %152 = arith.minsi %151, %c1_i64 : i64
      %153 = arith.cmpi sle, %26, %149 : i64
      %154 = arith.select %153, %152, %c0_i64 : i64
      %155 = arith.cmpi sgt, %154, %c1_i64 : i64
      %156 = arith.select %155, %c1_i64, %154 : i64
      %157 = cannir.extract_dim %148[1] : !cannir.shape<(-1, -1)>
      %158 = arith.maxsi %146, %c1_i64 : i64
      %159 = arith.divsi %157, %158 : i64
      %160 = arith.minsi %28, %159 : i64
      %161 = arith.muli %160, %146 : i64
      %162 = arith.subi %157, %161 : i64
      %163 = arith.minsi %146, %162 : i64
      %164 = arith.cmpi sle, %28, %159 : i64
      %165 = arith.select %164, %163, %c0_i64 : i64
      %166 = arith.cmpi sgt, %165, %c1024_i64 : i64
      %167 = arith.select %166, %c1024_i64, %165 : i64
      %168 = cannir.extract_physical_layout %arg2 : !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
      %169 = cannir.extract_stride %168 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
      %170 = cannir.extract_dim %169[0] : !cannir.stride<(-1, 1)>
      %171 = arith.muli %26, %170 : i64
      %172 = arith.muli %28, %146 : i64
      %173 = arith.addi %171, %172 : i64
      %174 = arith.muli %173, %c8_i64 : i64
      %175 = ascscalar.memref_address %alloc_1 : memref<1xui8>
      %176 = arith.divui %174, %c8_i64 : i64
      %177 = arith.addi %175, %176 : i64
      %178 = ascscalar.bind_gm_ptr %177 : memref<1xui8>
      %179 = ascscalar.bind_local_ptr %74 : memref<1xui8, 1>
      %180 = cannir.extract_stride %147 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
      %181 = cannir.extract_dim %180[0] : !cannir.stride<(-1, 1)>
      %182 = arith.cmpi ugt, %156, %c0_i64 : i64
      %183 = arith.cmpi ugt, %167, %c0_i64 : i64
      %184 = arith.andi %182, %183 : i1
      scf.if %184 {
        ascvec.get_buf mte3, %76, 0 {cannir.buffer.generated_sync}
        %185 = arith.trunci %156 : i64 to i32
        %186 = arith.trunci %167 : i64 to i32
        ascvec.copy_ub2gm %178[%c0], %179[%c0], %185, %186, %181, %c1024_i32 l2_cache_ctl %c0_i32 : memref<1xui8>, memref<1xui8, 1>
        ascvec.release_buf mte3, %76, 0 {cannir.buffer.generated_sync}
      }
      scf.yield %37 : i64
    }
    return
  }
}
