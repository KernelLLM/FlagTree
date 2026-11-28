module {
  func.func @run_wide_non_tail_eager_rng_f5ba2ec71f97bcfafcaa6a65cf78a511(%arg0: !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg1: !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg2: !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64) attributes {cannbotdsl.contract_digest = "f5ba2ec71f97bcfafcaa6a65cf78a511fcd68fe300166772dc14c9775db8348c", cannbotdsl.public_slot_count = 8 : i64, cannbotdsl.slot_dtype_bits = [16, 8, 8, 64, 64, 64, 64, 64], cannbotdsl.slot_dtype_codes = [4, 1, 1, 0, 0, 0, 0, 0], cannbotdsl.slot_dtype_lanes = [1, 1, 1, 1, 1, 1, 1, 1], cannbotdsl.slot_flags = [1, 1, 1, 1, 1, 1, 1, 1], cannbotdsl.slot_kinds = [1, 1, 1, 2, 2, 2, 2, 2], cannbotdsl.slot_type_digests = ["5d7cdc96dc21b3b739e32aeccf58c1df113bec54701c75107ed650f2544a07da", "171ddd09eb070fb4a6413ae87b2ceecddc233a9a173a0404e96285a4bb8b8659", "891bcd21e721f8d1a666bcca4e8705ec63796c1add8ae000536d0d98f0274d3a", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515"], cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.host} {
    cannir.kernel_launch @kernel__DynamicMxQuantSrKernel_run_wide_non_tail_0(%arg5) [%arg0, %arg1, %arg2, %arg3, %arg4, %arg6, %arg7] {cannbotdsl.profile = {indices = array<i64: 0, 1, 2>, name = "_DynamicMxQuantSrKernel_run_wide_non_tail", op_type = "_DynamicMxQuantSrKernel_run_wide_non_tail", roles = array<i64: 0, 1, 1>}} : !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, i64, i64, i64, i64
    return
  }
  func.func @kernel__DynamicMxQuantSrKernel_run_wide_non_tail_0(%arg0: !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg1: !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg2: !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64) attributes {cannir.buffer.auto_sync, cannir.buffer.sync_analyzed, cannir.buffer.sync_verified, cannir.buffer.tensor_state, cannir.core = "aiv", cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.device, cannir.explicit_ring_buffer_cursors, cannir.kernel_group = "kernel__DynamicMxQuantSrKernel_run_wide_non_tail_0", cannir.resource.ids_prepared} {
    %c1024_i32 = arith.constant 1024 : i32
    %c16384_i32 = arith.constant 16384 : i32
    %c0 = arith.constant 0 : index
    %c64_i64 = arith.constant 64 : i64
    %c256_i64 = arith.constant 256 : i64
    %false = arith.constant false
    %c128_i64 = arith.constant 128 : i64
    %c0_i32 = arith.constant 0 : i32
    %c32768_i64 = arith.constant 32768 : i64
    %c16_i64 = arith.constant 16 : i64
    %c4294967296_i64 = arith.constant 4294967296 : i64
    %c16_i32 = arith.constant 16 : i32
    %c64_i32 = arith.constant 64 : i32
    %c32_i32 = arith.constant 32 : i32
    %c16384_i64 = arith.constant 16384 : i64
    %c32_i64 = arith.constant 32 : i64
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
      %25 = arith.remsi %arg3, %c256_i64 : i64
      %26 = arith.cmpi ne, %25, %c0_i64 : i64
      %27 = arith.cmpi slt, %25, %c0_i64 : i64
      %28 = arith.andi %26, %27 : i1
      %29 = arith.select %28, %c256_i64, %c0_i64 : i64
      %30 = arith.addi %25, %29 overflow<nsw> : i64
      %31 = arith.cmpi eq, %30, %c0_i64 : i64
      %32 = arith.cmpi eq, %arg4, %c64_i64 : i64
      %33 = arith.select %31, %32, %false : i1
      scf.yield %33 : i1
    } else {
      scf.yield %false : i1
    }
    %13 = scf.if %12 -> (i64) {
      scf.yield %c256_i64 : i64
    } else {
      %25 = arith.remsi %arg3, %c128_i64 : i64
      %26 = arith.cmpi ne, %25, %c0_i64 : i64
      %27 = arith.cmpi slt, %25, %c0_i64 : i64
      %28 = arith.andi %26, %27 : i1
      %29 = arith.select %28, %c128_i64, %c0_i64 : i64
      %30 = arith.addi %25, %29 overflow<nsw> : i64
      %31 = arith.cmpi eq, %30, %c0_i64 : i64
      %32 = arith.select %11, %31, %false : i1
      %33 = arith.select %32, %c128_i64, %c64_i64 : i64
      scf.yield %33 : i64
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
      %84 = arith.remsi %arg4, %c64_i64 : i64
      %85 = arith.cmpi ne, %84, %c0_i64 : i64
      %86 = arith.cmpi slt, %84, %c0_i64 : i64
      %87 = arith.andi %85, %86 : i1
      %88 = arith.divsi %arg4, %c64_i64 : i64
      %89 = arith.extui %87 : i1 to i64
      %90 = arith.subi %88, %89 : i64
      scf.for %arg9 = %c0_i64 to %90 step %c1_i64  : i64 {
        %172 = arith.muli %arg9, %c64_i64 : i64
        %173 = arith.muli %33, %arg4 : i64
        %174 = arith.addi %173, %172 : i64
        %175 = arith.muli %35, %13 : i64
        %176 = arith.remsi %175, %c16_i64 : i64
        %177 = arith.cmpi ne, %176, %c0_i64 : i64
        %178 = arith.cmpi slt, %176, %c0_i64 : i64
        %179 = arith.andi %177, %178 : i1
        %180 = arith.divsi %175, %c16_i64 : i64
        %181 = arith.extui %179 : i1 to i64
        %182 = arith.subi %180, %181 : i64
        %183 = arith.remsi %arg3, %c16_i64 : i64
        %184 = arith.cmpi ne, %183, %c0_i64 : i64
        %185 = arith.cmpi slt, %183, %c0_i64 : i64
        %186 = arith.andi %184, %185 : i1
        %187 = arith.divsi %arg3, %c16_i64 : i64
        %188 = arith.extui %186 : i1 to i64
        %189 = arith.subi %187, %188 : i64
        %190 = arith.trunci %arg5 : i64 to i32
        %191 = arith.extui %190 : i32 to i64
        %192 = arith.subi %arg5, %191 : i64
        %193 = arith.remsi %192, %c4294967296_i64 : i64
        %194 = arith.cmpi ne, %193, %c0_i64 : i64
        %195 = arith.cmpi slt, %193, %c0_i64 : i64
        %196 = arith.andi %194, %195 : i1
        %197 = arith.divsi %192, %c4294967296_i64 : i64
        %198 = arith.extui %196 : i1 to i64
        %199 = arith.subi %197, %198 : i64
        %200 = arith.trunci %199 : i64 to i32
        %201 = arith.trunci %13 : i64 to i32
        %202 = arith.divsi %201, %c16_i32 : i32
        ascvec.vf(%182, %174, %189, %arg6, %190, %200, %alloc_9, %202 : i64, i64, i64, i64, i32, i32, memref<4096xi32, 1>, i32) {
        ^bb0(%arg10: i64, %arg11: i64, %arg12: i64, %arg13: i64, %arg14: i32, %arg15: i32, %arg16: memref<4096xi32, 1>, %arg17: i32):
          %c1_i32 = arith.constant 1 : i32
          %c192_i32 = arith.constant 192 : i32
          %c128_i32 = arith.constant 128 : i32
          %c64_i32_10 = arith.constant 64 : i32
          %c256_i32 = arith.constant 256 : i32
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
          %206 = ascvec.create_mask : !ascvec.mask
          %207 = ascvec.arange %c0_i32_13 : i32 -> <ui32>
          scf.for %arg18 = %c0_i32_13 to %arg17 step %c1_i32  : i32 {
            %208 = arith.extsi %arg18 : i32 to i64
            %209 = arith.addi %arg10, %208 : i64
            %210 = arith.trunci %arg11 : i64 to i32
            %211 = arith.extui %210 : i32 to i64
            %212 = arith.subi %arg11, %211 : i64
            %213 = arith.remsi %212, %c4294967296_i64_12 : i64
            %214 = arith.cmpi ne, %213, %c0_i64_11 : i64
            %215 = arith.cmpi slt, %213, %c0_i64_11 : i64
            %216 = arith.andi %214, %215 : i1
            %217 = arith.divsi %212, %c4294967296_i64_12 : i64
            %218 = arith.extui %216 : i1 to i64
            %219 = arith.subi %217, %218 : i64
            %220 = arith.trunci %219 : i64 to i32
            %221 = arith.trunci %209 : i64 to i32
            %222 = arith.extui %221 : i32 to i64
            %223 = arith.subi %209, %222 : i64
            %224 = arith.remsi %223, %c4294967296_i64_12 : i64
            %225 = arith.cmpi ne, %224, %c0_i64_11 : i64
            %226 = arith.cmpi slt, %224, %c0_i64_11 : i64
            %227 = arith.andi %225, %226 : i1
            %228 = arith.divsi %223, %c4294967296_i64_12 : i64
            %229 = arith.extui %227 : i1 to i64
            %230 = arith.subi %228, %229 : i64
            %231 = arith.trunci %230 : i64 to i32
            %232 = ascvec.duplicate_scalar %210, %206 : i32 -> <ui32>
            %233 = ascvec.duplicate_scalar %220, %206 : i32 -> <ui32>
            %234 = ascvec.duplicate_scalar %c0_i32_13, %206 : i32 -> <ui32>
            %carry, %dst = ascvec.addc %232, %207, %206 : <ui32>, <ui32> -> !ascvec.mask, <ui32>
            %carry_14, %dst_15 = ascvec.addc %233, %234, %carry, %206 : <ui32>, <ui32> -> !ascvec.mask, <ui32>
            %235 = ascvec.duplicate_scalar %221, %206 : i32 -> <ui32>
            %236 = ascvec.duplicate_scalar %231, %206 : i32 -> <ui32>
            %237 = arith.trunci %arg12 : i64 to i32
            %238 = arith.extui %237 : i32 to i64
            %239 = arith.subi %arg12, %238 : i64
            %240 = arith.remsi %239, %c4294967296_i64_12 : i64
            %241 = arith.cmpi ne, %240, %c0_i64_11 : i64
            %242 = arith.cmpi slt, %240, %c0_i64_11 : i64
            %243 = arith.andi %241, %242 : i1
            %244 = arith.divsi %239, %c4294967296_i64_12 : i64
            %245 = arith.extui %243 : i1 to i64
            %246 = arith.subi %244, %245 : i64
            %247 = arith.trunci %246 : i64 to i32
            %248 = ascvec.duplicate_scalar %237, %206 : i32 -> <ui32>
            %dst0, %dst1 = ascvec.mull %dst, %248, %206 : <ui32>, <ui32> -> <ui32>, <ui32>
            %249 = ascvec.mul_scalar %dst, %247, %206 : <ui32>, i32
            %250 = ascvec.add %dst1, %249, %206 : <ui32>
            %251 = ascvec.mul_scalar %dst_15, %237, %206 : <ui32>, i32
            %252 = ascvec.add %250, %251, %206 : <ui32>
            %carry_16, %dst_17 = ascvec.addc %dst0, %235, %206 : <ui32>, <ui32> -> !ascvec.mask, <ui32>
            %carry_18, %dst_19 = ascvec.addc %252, %236, %carry_16, %206 : <ui32>, <ui32> -> !ascvec.mask, <ui32>
            %253 = arith.trunci %arg13 : i64 to i32
            %254 = arith.extui %253 : i32 to i64
            %255 = arith.subi %arg13, %254 : i64
            %256 = arith.remsi %255, %c4294967296_i64_12 : i64
            %257 = arith.cmpi ne, %256, %c0_i64_11 : i64
            %258 = arith.cmpi slt, %256, %c0_i64_11 : i64
            %259 = arith.andi %257, %258 : i1
            %260 = arith.divsi %255, %c4294967296_i64_12 : i64
            %261 = arith.extui %259 : i1 to i64
            %262 = arith.subi %260, %261 : i64
            %263 = arith.trunci %262 : i64 to i32
            %264 = arith.divui %253, %c4_i32 : i32
            %265 = arith.remui %263, %c4_i32 : i32
            %266 = arith.muli %265, %c1073741824_i32 : i32
            %267 = arith.addi %264, %266 : i32
            %268 = arith.divui %263, %c4_i32 : i32
            %269 = ascvec.shr %dst_17, 14, %206 : <ui32> -> <ui32>
            %270 = ascvec.shl %dst_19, 18, %206 : <ui32> -> <ui32>
            %271 = ascvec.or %269, %270, %206 : <ui32>, <ui32> -> <ui32>
            %272 = ascvec.shr %dst_19, 14, %206 : <ui32> -> <ui32>
            %273 = ascvec.duplicate_scalar %267, %206 : i32 -> <ui32>
            %274 = ascvec.duplicate_scalar %268, %206 : i32 -> <ui32>
            %carry_20, %dst_21 = ascvec.addc %273, %271, %206 : <ui32>, <ui32> -> !ascvec.mask, <ui32>
            %carry_22, %dst_23 = ascvec.addc %274, %272, %carry_20, %206 : <ui32>, <ui32> -> !ascvec.mask, <ui32>
            %275 = ascvec.duplicate_scalar %c16383_i32, %206 : i32 -> <ui32>
            %276 = ascvec.and %dst_17, %275, %206 : <ui32>, <ui32> -> <ui32>
            %277 = ascvec.duplicate_scalar %c-766435501_i32, %206 : i32 -> <ui32>
            %278 = ascvec.duplicate_scalar %c-845247145_i32, %206 : i32 -> <ui32>
            %dst0_24, %dst1_25 = ascvec.mull %234, %277, %206 : <ui32>, <ui32> -> <ui32>, <ui32>
            %dst0_26, %dst1_27 = ascvec.mull %dst_21, %278, %206 : <ui32>, <ui32> -> <ui32>, <ui32>
            %279 = ascvec.xor %dst1_27, %276, %206 : <ui32>, <ui32> -> <ui32>
            %280 = ascvec.xor %dst1_25, %dst_23, %206 : <ui32>, <ui32> -> <ui32>
            %281 = ascvec.duplicate_scalar %arg14, %206 : i32 -> <ui32>
            %282 = ascvec.duplicate_scalar %arg15, %206 : i32 -> <ui32>
            %283 = ascvec.xor %279, %281, %206 : <ui32>, <ui32> -> <ui32>
            %284 = ascvec.xor %280, %282, %206 : <ui32>, <ui32> -> <ui32>
            %dst0_28, %dst1_29 = ascvec.mull %283, %277, %206 : <ui32>, <ui32> -> <ui32>, <ui32>
            %dst0_30, %dst1_31 = ascvec.mull %284, %278, %206 : <ui32>, <ui32> -> <ui32>, <ui32>
            %285 = ascvec.xor %dst1_31, %dst0_26, %206 : <ui32>, <ui32> -> <ui32>
            %286 = ascvec.xor %dst1_29, %dst0_24, %206 : <ui32>, <ui32> -> <ui32>
            %287 = arith.addi %arg14, %c-1640531527_i32 : i32
            %288 = ascvec.duplicate_scalar %287, %206 : i32 -> <ui32>
            %289 = arith.addi %arg15, %c-1150833019_i32 : i32
            %290 = ascvec.duplicate_scalar %289, %206 : i32 -> <ui32>
            %291 = ascvec.xor %285, %288, %206 : <ui32>, <ui32> -> <ui32>
            %292 = ascvec.xor %286, %290, %206 : <ui32>, <ui32> -> <ui32>
            %dst0_32, %dst1_33 = ascvec.mull %291, %277, %206 : <ui32>, <ui32> -> <ui32>, <ui32>
            %dst0_34, %dst1_35 = ascvec.mull %292, %278, %206 : <ui32>, <ui32> -> <ui32>, <ui32>
            %293 = ascvec.xor %dst1_35, %dst0_30, %206 : <ui32>, <ui32> -> <ui32>
            %294 = ascvec.xor %dst1_33, %dst0_28, %206 : <ui32>, <ui32> -> <ui32>
            %295 = arith.addi %arg14, %c1013904242_i32 : i32
            %296 = ascvec.duplicate_scalar %295, %206 : i32 -> <ui32>
            %297 = arith.addi %arg15, %c1993301258_i32 : i32
            %298 = ascvec.duplicate_scalar %297, %206 : i32 -> <ui32>
            %299 = ascvec.xor %293, %296, %206 : <ui32>, <ui32> -> <ui32>
            %300 = ascvec.xor %294, %298, %206 : <ui32>, <ui32> -> <ui32>
            %dst0_36, %dst1_37 = ascvec.mull %299, %277, %206 : <ui32>, <ui32> -> <ui32>, <ui32>
            %dst0_38, %dst1_39 = ascvec.mull %300, %278, %206 : <ui32>, <ui32> -> <ui32>, <ui32>
            %301 = ascvec.xor %dst1_39, %dst0_34, %206 : <ui32>, <ui32> -> <ui32>
            %302 = ascvec.xor %dst1_37, %dst0_32, %206 : <ui32>, <ui32> -> <ui32>
            %303 = arith.addi %arg14, %c-626627285_i32 : i32
            %304 = ascvec.duplicate_scalar %303, %206 : i32 -> <ui32>
            %305 = arith.addi %arg15, %c842468239_i32 : i32
            %306 = ascvec.duplicate_scalar %305, %206 : i32 -> <ui32>
            %307 = ascvec.xor %301, %304, %206 : <ui32>, <ui32> -> <ui32>
            %308 = ascvec.xor %302, %306, %206 : <ui32>, <ui32> -> <ui32>
            %dst0_40, %dst1_41 = ascvec.mull %307, %277, %206 : <ui32>, <ui32> -> <ui32>, <ui32>
            %dst0_42, %dst1_43 = ascvec.mull %308, %278, %206 : <ui32>, <ui32> -> <ui32>, <ui32>
            %309 = ascvec.xor %dst1_43, %dst0_38, %206 : <ui32>, <ui32> -> <ui32>
            %310 = ascvec.xor %dst1_41, %dst0_36, %206 : <ui32>, <ui32> -> <ui32>
            %311 = arith.addi %arg14, %c2027808484_i32 : i32
            %312 = ascvec.duplicate_scalar %311, %206 : i32 -> <ui32>
            %313 = arith.addi %arg15, %c-308364780_i32 : i32
            %314 = ascvec.duplicate_scalar %313, %206 : i32 -> <ui32>
            %315 = ascvec.xor %309, %312, %206 : <ui32>, <ui32> -> <ui32>
            %316 = ascvec.xor %310, %314, %206 : <ui32>, <ui32> -> <ui32>
            %dst0_44, %dst1_45 = ascvec.mull %315, %277, %206 : <ui32>, <ui32> -> <ui32>, <ui32>
            %dst0_46, %dst1_47 = ascvec.mull %316, %278, %206 : <ui32>, <ui32> -> <ui32>, <ui32>
            %317 = ascvec.xor %dst1_47, %dst0_42, %206 : <ui32>, <ui32> -> <ui32>
            %318 = ascvec.xor %dst1_45, %dst0_40, %206 : <ui32>, <ui32> -> <ui32>
            %319 = arith.addi %arg14, %c387276957_i32 : i32
            %320 = ascvec.duplicate_scalar %319, %206 : i32 -> <ui32>
            %321 = arith.addi %arg15, %c-1459197799_i32 : i32
            %322 = ascvec.duplicate_scalar %321, %206 : i32 -> <ui32>
            %323 = ascvec.xor %317, %320, %206 : <ui32>, <ui32> -> <ui32>
            %324 = ascvec.xor %318, %322, %206 : <ui32>, <ui32> -> <ui32>
            %dst0_48, %dst1_49 = ascvec.mull %323, %277, %206 : <ui32>, <ui32> -> <ui32>, <ui32>
            %dst0_50, %dst1_51 = ascvec.mull %324, %278, %206 : <ui32>, <ui32> -> <ui32>, <ui32>
            %325 = ascvec.xor %dst1_51, %dst0_46, %206 : <ui32>, <ui32> -> <ui32>
            %326 = ascvec.xor %dst1_49, %dst0_44, %206 : <ui32>, <ui32> -> <ui32>
            %327 = arith.addi %arg14, %c-1253254570_i32 : i32
            %328 = ascvec.duplicate_scalar %327, %206 : i32 -> <ui32>
            %329 = arith.addi %arg15, %c1684936478_i32 : i32
            %330 = ascvec.duplicate_scalar %329, %206 : i32 -> <ui32>
            %331 = ascvec.xor %325, %328, %206 : <ui32>, <ui32> -> <ui32>
            %332 = ascvec.xor %326, %330, %206 : <ui32>, <ui32> -> <ui32>
            %dst0_52, %dst1_53 = ascvec.mull %331, %277, %206 : <ui32>, <ui32> -> <ui32>, <ui32>
            %dst0_54, %dst1_55 = ascvec.mull %332, %278, %206 : <ui32>, <ui32> -> <ui32>, <ui32>
            %333 = ascvec.xor %dst1_55, %dst0_50, %206 : <ui32>, <ui32> -> <ui32>
            %334 = ascvec.xor %dst1_53, %dst0_48, %206 : <ui32>, <ui32> -> <ui32>
            %335 = arith.addi %arg14, %c1401181199_i32 : i32
            %336 = ascvec.duplicate_scalar %335, %206 : i32 -> <ui32>
            %337 = arith.addi %arg15, %c534103459_i32 : i32
            %338 = ascvec.duplicate_scalar %337, %206 : i32 -> <ui32>
            %339 = ascvec.xor %333, %336, %206 : <ui32>, <ui32> -> <ui32>
            %340 = ascvec.xor %334, %338, %206 : <ui32>, <ui32> -> <ui32>
            %dst0_56, %dst1_57 = ascvec.mull %339, %277, %206 : <ui32>, <ui32> -> <ui32>, <ui32>
            %dst0_58, %dst1_59 = ascvec.mull %340, %278, %206 : <ui32>, <ui32> -> <ui32>, <ui32>
            %341 = ascvec.xor %dst1_59, %dst0_54, %206 : <ui32>, <ui32> -> <ui32>
            %342 = ascvec.xor %dst1_57, %dst0_52, %206 : <ui32>, <ui32> -> <ui32>
            %343 = arith.addi %arg14, %c-239350328_i32 : i32
            %344 = ascvec.duplicate_scalar %343, %206 : i32 -> <ui32>
            %345 = arith.addi %arg15, %c-616729560_i32 : i32
            %346 = ascvec.duplicate_scalar %345, %206 : i32 -> <ui32>
            %347 = ascvec.xor %341, %344, %206 : <ui32>, <ui32> -> <ui32>
            %348 = ascvec.xor %342, %346, %206 : <ui32>, <ui32> -> <ui32>
            %dst0_60, %dst1_61 = ascvec.mull %347, %277, %206 : <ui32>, <ui32> -> <ui32>, <ui32>
            %dst0_62, %dst1_63 = ascvec.mull %348, %278, %206 : <ui32>, <ui32> -> <ui32>, <ui32>
            %349 = ascvec.xor %dst1_63, %dst0_58, %206 : <ui32>, <ui32> -> <ui32>
            %350 = ascvec.xor %dst1_61, %dst0_56, %206 : <ui32>, <ui32> -> <ui32>
            %351 = arith.addi %arg14, %c-1879881855_i32 : i32
            %352 = ascvec.duplicate_scalar %351, %206 : i32 -> <ui32>
            %353 = arith.addi %arg15, %c-1767562579_i32 : i32
            %354 = ascvec.duplicate_scalar %353, %206 : i32 -> <ui32>
            %355 = ascvec.xor %349, %352, %206 : <ui32>, <ui32> -> <ui32>
            %356 = ascvec.xor %350, %354, %206 : <ui32>, <ui32> -> <ui32>
            %357 = arith.muli %arg18, %c256_i32 : i32
            %358 = ascvec.reinterpret %355 : <ui32> -> <i32>
            %359 = arith.index_cast %357 : i32 to index
            ascvec.storealign %arg16[%359], %358, %206 : memref<4096xi32, 1>, index, <i32>
            %360 = arith.addi %357, %c64_i32_10 : i32
            %361 = ascvec.reinterpret %dst0_62 : <ui32> -> <i32>
            %362 = arith.index_cast %360 : i32 to index
            ascvec.storealign %arg16[%362], %361, %206 : memref<4096xi32, 1>, index, <i32>
            %363 = arith.addi %357, %c128_i32 : i32
            %364 = ascvec.reinterpret %356 : <ui32> -> <i32>
            %365 = arith.index_cast %363 : i32 to index
            ascvec.storealign %arg16[%365], %364, %206 : memref<4096xi32, 1>, index, <i32>
            %366 = arith.addi %357, %c192_i32 : i32
            %367 = ascvec.reinterpret %dst0_60 : <ui32> -> <i32>
            %368 = arith.index_cast %366 : i32 to index
            ascvec.storealign %arg16[%368], %367, %206 : memref<4096xi32, 1>, index, <i32>
          }
          ascvec.mem_bar vst_vld
        } {cannir.vf_mode = "simd"}
        %203 = arith.divsi %201, %c32_i32 : i32
        %204 = arith.divsi %201, %c64_i32 : i32
        ascvec.get_buf v, %42, 0 {cannir.buffer.generated_sync}
        ascvec.get_buf v, %78, 0 {cannir.buffer.generated_sync}
        ascvec.get_buf v, %83, 0 {cannir.buffer.generated_sync}
        %205 = ascscalar.bitcast_memref %82 byte_offset = %c0_i64 : memref<1024xui8, 1> -> memref<?xui16, 1>
        ascvec.vf(%arg4, %172, %73, %alloc_8, %alloc_9, %77, %203, %205, %204 : i64, i64, memref<16384xbf16, 1>, memref<512xui32, 1>, memref<4096xi32, 1>, memref<16384xui8, 1>, i32, memref<?xui16, 1>, i32) {
        ^bb0(%arg10: i64, %arg11: i64, %arg12: memref<16384xbf16, 1>, %arg13: memref<512xui32, 1>, %arg14: memref<4096xi32, 1>, %arg15: memref<16384xui8, 1>, %arg16: i32, %arg17: memref<?xui16, 1>, %arg18: i32):
          %c8_i32 = arith.constant 8 : i32
          %c3_i32 = arith.constant 3 : i32
          %c2_i32 = arith.constant 2 : i32
          %c127_i32 = arith.constant 127 : i32
          %c126_i32 = arith.constant 126 : i32
          %c65535_i32 = arith.constant 65535 : i32
          %c1015021568_i32 = arith.constant 1015021568 : i32
          %cst = arith.constant 0x4C000000 : f32
          %c62914560_i32 = arith.constant 62914560 : i32
          %c128_i32 = arith.constant 128 : i32
          %c4_i32 = arith.constant 4 : i32
          %c512_i32 = arith.constant 512 : i32
          %c64_i32_10 = arith.constant 64 : i32
          %c2139160576_i32 = arith.constant 2139160576 : i32
          %c254_i32 = arith.constant 254 : i32
          %c255_i32 = arith.constant 255 : i32
          %c73400321_i32 = arith.constant 73400321 : i32
          %c-65011713_i32 = arith.constant -65011713 : i32
          %c2139095040_i32 = arith.constant 2139095040 : i32
          %c1_i32 = arith.constant 1 : i32
          %c2147483647_i32 = arith.constant 2147483647 : i32
          %c32_i32_11 = arith.constant 32 : i32
          %c0_i32_12 = arith.constant 0 : i32
          %206 = ascvec.create_mask : !ascvec.mask
          scf.for %arg19 = %c0_i32_12 to %arg16 step %c1_i32  : i32 {
            %207 = ascvec.duplicate_scalar %c0_i32_12, %206 : i32 -> <ui32>
            %208 = scf.for %arg20 = %c0_i32_12 to %c32_i32_11 step %c1_i32 iter_args(%arg21 = %207) -> (!ascvec.vreg<ui32>)  : i32 {
              %221 = arith.muli %arg19, %c32_i32_11 : i32
              %222 = arith.addi %221, %arg20 : i32
              %223 = arith.extsi %222 : i32 to i64
              %224 = arith.muli %223, %arg10 : i64
              %225 = arith.addi %224, %arg11 : i64
              %226 = arith.index_cast %225 : i64 to index
              %227 = ascvec.loadalign_unpack %arg12[%226] {mode = "b16_to_b32"} : memref<16384xbf16, 1>, index -> <bf16>
              %228 = ascvec.cast %227, %206 {rounding = 6 : i32} : <bf16> -> <f32>
              %229 = ascvec.reinterpret %228 : <f32> -> <ui32>
              %230 = ascvec.duplicate_scalar %c2147483647_i32, %206 : i32 -> <ui32>
              %231 = ascvec.and %229, %230, %206 : <ui32>, <ui32> -> <ui32>
              %232 = ascvec.max %arg21, %231, %206 : <ui32>
              scf.yield %232 : !ascvec.vreg<ui32>
            }
            %209 = ascvec.cmp_ge_scalar %208, %c2139095040_i32, %206 : <ui32>, i32 -> !ascvec.mask
            %210 = ascvec.add_scalar %208, %c-65011713_i32, %206 : <ui32>, i32
            %211 = ascvec.cmp_gt_scalar %208, %c73400321_i32, %206 : <ui32>, i32 -> !ascvec.mask
            %212 = ascvec.shr %210, 23, %211 : <ui32> -> <ui32>
            %213 = ascvec.duplicate_scalar %c255_i32, %209 merge %212 : !ascvec.vreg<ui32> {mode = "merging"} : i32 -> <ui32>
            %214 = ascvec.duplicate_scalar %c254_i32, %206 : i32 -> <ui32>
            %215 = ascvec.sub %214, %213, %206 : <ui32>
            %216 = ascvec.shl %215, 23, %206 : <ui32> -> <ui32>
            %217 = ascvec.duplicate_scalar %c2139160576_i32, %209 merge %216 : !ascvec.vreg<ui32> {mode = "merging"} : i32 -> <ui32>
            %218 = ascvec.reinterpret %217 : <ui32> -> <f32>
            %219 = arith.muli %arg19, %c64_i32_10 : i32
            %220 = arith.index_cast %219 : i32 to index
            ascvec.storealign %arg13[%220], %213, %206 : memref<512xui32, 1>, index, <ui32>
            scf.for %arg20 = %c0_i32_12 to %c8_i32 step %c1_i32  : i32 {
              %221 = arith.muli %arg19, %c512_i32 : i32
              %222 = arith.muli %arg20, %c64_i32_10 overflow<nsw> : i32
              %223 = arith.addi %221, %222 : i32
              %224 = arith.index_cast %223 : i32 to index
              %225 = ascvec.loadalign %arg14[%224] : memref<4096xi32, 1>, index -> <i32>
              %226 = ascvec.reinterpret %225 : <i32> -> <ui32>
              %227 = ascvec.shl %226, 8, %206 : <ui32> -> <ui32>
              %228 = ascvec.shr %226, 24, %206 : <ui32> -> <ui32>
              %229 = ascvec.or %227, %228, %206 : <ui32>, <ui32> -> <ui32>
              %230 = arith.muli %arg19, %c32_i32_11 : i32
              %231 = arith.muli %arg20, %c4_i32 overflow<nsw> : i32
              %232 = arith.addi %230, %231 : i32
              %233 = arith.extsi %232 : i32 to i64
              %234 = arith.muli %233, %arg10 : i64
              %235 = arith.addi %234, %arg11 : i64
              %236 = ascvec.shr %226, 16, %206 : <ui32> -> <ui32>
              %237 = arith.index_cast %235 : i64 to index
              %238 = ascvec.loadalign_unpack %arg12[%237] {mode = "b16_to_b32"} : memref<16384xbf16, 1>, index -> <bf16>
              %239 = ascvec.cast %238, %206 {rounding = 6 : i32} : <bf16> -> <f32>
              %240 = ascvec.mul %239, %218, %206 : <f32>
              %241 = ascvec.cast %240, %206 : <f32> -> <bf16>
              %242 = ascvec.cast %241, %206 {rounding = 6 : i32} : <bf16> -> <f32>
              %243 = ascvec.reinterpret %242 : <f32> -> <ui32>
              %244 = ascvec.shr %243, 24, %206 : <ui32> -> <ui32>
              %245 = ascvec.duplicate_scalar %c128_i32, %206 : i32 -> <ui32>
              %246 = ascvec.and %244, %245, %206 : <ui32>, <ui32> -> <ui32>
              %247 = ascvec.duplicate_scalar %c2147483647_i32, %206 : i32 -> <ui32>
              %248 = ascvec.and %243, %247, %206 : <ui32>, <ui32> -> <ui32>
              %249 = ascvec.shr %248, 4, %206 : <ui32> -> <ui32>
              %250 = ascvec.duplicate_scalar %c62914560_i32, %206 : i32 -> <ui32>
              %251 = ascvec.sub %249, %250, %206 : <ui32>
              %252 = ascvec.reinterpret %248 : <ui32> -> <f32>
              %253 = ascvec.mul_scalar %252, %cst, %206 : <f32>, f32
              %254 = ascvec.cast %253, %206 {rounding = 4 : i32} : <f32> -> <i32>
              %255 = ascvec.reinterpret %254 : <i32> -> <ui32>
              %256 = ascvec.cmp_ge_scalar %248, %c1015021568_i32, %206 : <ui32>, i32 -> !ascvec.mask
              %257 = ascvec.select %251, %255, %256 : <ui32>
              %258 = ascvec.duplicate_scalar %c65535_i32, %206 : i32 -> <ui32>
              %259 = ascvec.and %236, %258, %206 : <ui32>, <ui32> -> <ui32>
              %260 = ascvec.add %257, %259, %206 : <ui32>
              %261 = ascvec.shr %260, 16, %206 : <ui32> -> <ui32>
              %262 = ascvec.min_scalar %261, %c126_i32, %206 : <ui32>, i32
              %263 = ascvec.or %262, %246, %206 : <ui32>, <ui32> -> <ui32>
              %264 = ascvec.duplicate_scalar %c127_i32, %206 : i32 -> <ui32>
              %265 = ascvec.cmp_gt_scalar %248, %c2139095040_i32, %206 : <ui32>, i32 -> !ascvec.mask
              %266 = ascvec.select %264, %263, %265 : <ui32>
              ascvec.storealign_pack %arg15[%237], %266, %206 {mode = "b32_to_b8"} : memref<16384xui8, 1>, index, <ui32>
              %267 = arith.addi %232, %c1_i32 : i32
              %268 = arith.extsi %267 : i32 to i64
              %269 = arith.muli %268, %arg10 : i64
              %270 = arith.addi %269, %arg11 : i64
              %271 = ascvec.and %229, %258, %206 : <ui32>, <ui32> -> <ui32>
              %272 = arith.index_cast %270 : i64 to index
              %273 = ascvec.loadalign_unpack %arg12[%272] {mode = "b16_to_b32"} : memref<16384xbf16, 1>, index -> <bf16>
              %274 = ascvec.cast %273, %206 {rounding = 6 : i32} : <bf16> -> <f32>
              %275 = ascvec.mul %274, %218, %206 : <f32>
              %276 = ascvec.cast %275, %206 : <f32> -> <bf16>
              %277 = ascvec.cast %276, %206 {rounding = 6 : i32} : <bf16> -> <f32>
              %278 = ascvec.reinterpret %277 : <f32> -> <ui32>
              %279 = ascvec.shr %278, 24, %206 : <ui32> -> <ui32>
              %280 = ascvec.and %279, %245, %206 : <ui32>, <ui32> -> <ui32>
              %281 = ascvec.and %278, %247, %206 : <ui32>, <ui32> -> <ui32>
              %282 = ascvec.shr %281, 4, %206 : <ui32> -> <ui32>
              %283 = ascvec.sub %282, %250, %206 : <ui32>
              %284 = ascvec.reinterpret %281 : <ui32> -> <f32>
              %285 = ascvec.mul_scalar %284, %cst, %206 : <f32>, f32
              %286 = ascvec.cast %285, %206 {rounding = 4 : i32} : <f32> -> <i32>
              %287 = ascvec.reinterpret %286 : <i32> -> <ui32>
              %288 = ascvec.cmp_ge_scalar %281, %c1015021568_i32, %206 : <ui32>, i32 -> !ascvec.mask
              %289 = ascvec.select %283, %287, %288 : <ui32>
              %290 = ascvec.and %271, %258, %206 : <ui32>, <ui32> -> <ui32>
              %291 = ascvec.add %289, %290, %206 : <ui32>
              %292 = ascvec.shr %291, 16, %206 : <ui32> -> <ui32>
              %293 = ascvec.min_scalar %292, %c126_i32, %206 : <ui32>, i32
              %294 = ascvec.or %293, %280, %206 : <ui32>, <ui32> -> <ui32>
              %295 = ascvec.cmp_gt_scalar %281, %c2139095040_i32, %206 : <ui32>, i32 -> !ascvec.mask
              %296 = ascvec.select %264, %294, %295 : <ui32>
              ascvec.storealign_pack %arg15[%272], %296, %206 {mode = "b32_to_b8"} : memref<16384xui8, 1>, index, <ui32>
              %297 = arith.addi %232, %c2_i32 : i32
              %298 = arith.extsi %297 : i32 to i64
              %299 = arith.muli %298, %arg10 : i64
              %300 = arith.addi %299, %arg11 : i64
              %301 = ascvec.and %226, %258, %206 : <ui32>, <ui32> -> <ui32>
              %302 = arith.index_cast %300 : i64 to index
              %303 = ascvec.loadalign_unpack %arg12[%302] {mode = "b16_to_b32"} : memref<16384xbf16, 1>, index -> <bf16>
              %304 = ascvec.cast %303, %206 {rounding = 6 : i32} : <bf16> -> <f32>
              %305 = ascvec.mul %304, %218, %206 : <f32>
              %306 = ascvec.cast %305, %206 : <f32> -> <bf16>
              %307 = ascvec.cast %306, %206 {rounding = 6 : i32} : <bf16> -> <f32>
              %308 = ascvec.reinterpret %307 : <f32> -> <ui32>
              %309 = ascvec.shr %308, 24, %206 : <ui32> -> <ui32>
              %310 = ascvec.and %309, %245, %206 : <ui32>, <ui32> -> <ui32>
              %311 = ascvec.and %308, %247, %206 : <ui32>, <ui32> -> <ui32>
              %312 = ascvec.shr %311, 4, %206 : <ui32> -> <ui32>
              %313 = ascvec.sub %312, %250, %206 : <ui32>
              %314 = ascvec.reinterpret %311 : <ui32> -> <f32>
              %315 = ascvec.mul_scalar %314, %cst, %206 : <f32>, f32
              %316 = ascvec.cast %315, %206 {rounding = 4 : i32} : <f32> -> <i32>
              %317 = ascvec.reinterpret %316 : <i32> -> <ui32>
              %318 = ascvec.cmp_ge_scalar %311, %c1015021568_i32, %206 : <ui32>, i32 -> !ascvec.mask
              %319 = ascvec.select %313, %317, %318 : <ui32>
              %320 = ascvec.and %301, %258, %206 : <ui32>, <ui32> -> <ui32>
              %321 = ascvec.add %319, %320, %206 : <ui32>
              %322 = ascvec.shr %321, 16, %206 : <ui32> -> <ui32>
              %323 = ascvec.min_scalar %322, %c126_i32, %206 : <ui32>, i32
              %324 = ascvec.or %323, %310, %206 : <ui32>, <ui32> -> <ui32>
              %325 = ascvec.cmp_gt_scalar %311, %c2139095040_i32, %206 : <ui32>, i32 -> !ascvec.mask
              %326 = ascvec.select %264, %324, %325 : <ui32>
              ascvec.storealign_pack %arg15[%302], %326, %206 {mode = "b32_to_b8"} : memref<16384xui8, 1>, index, <ui32>
              %327 = arith.addi %232, %c3_i32 : i32
              %328 = arith.extsi %327 : i32 to i64
              %329 = arith.muli %328, %arg10 : i64
              %330 = arith.addi %329, %arg11 : i64
              %331 = ascvec.shr %229, 16, %206 : <ui32> -> <ui32>
              %332 = arith.index_cast %330 : i64 to index
              %333 = ascvec.loadalign_unpack %arg12[%332] {mode = "b16_to_b32"} : memref<16384xbf16, 1>, index -> <bf16>
              %334 = ascvec.cast %333, %206 {rounding = 6 : i32} : <bf16> -> <f32>
              %335 = ascvec.mul %334, %218, %206 : <f32>
              %336 = ascvec.cast %335, %206 : <f32> -> <bf16>
              %337 = ascvec.cast %336, %206 {rounding = 6 : i32} : <bf16> -> <f32>
              %338 = ascvec.reinterpret %337 : <f32> -> <ui32>
              %339 = ascvec.shr %338, 24, %206 : <ui32> -> <ui32>
              %340 = ascvec.and %339, %245, %206 : <ui32>, <ui32> -> <ui32>
              %341 = ascvec.and %338, %247, %206 : <ui32>, <ui32> -> <ui32>
              %342 = ascvec.shr %341, 4, %206 : <ui32> -> <ui32>
              %343 = ascvec.sub %342, %250, %206 : <ui32>
              %344 = ascvec.reinterpret %341 : <ui32> -> <f32>
              %345 = ascvec.mul_scalar %344, %cst, %206 : <f32>, f32
              %346 = ascvec.cast %345, %206 {rounding = 4 : i32} : <f32> -> <i32>
              %347 = ascvec.reinterpret %346 : <i32> -> <ui32>
              %348 = ascvec.cmp_ge_scalar %341, %c1015021568_i32, %206 : <ui32>, i32 -> !ascvec.mask
              %349 = ascvec.select %343, %347, %348 : <ui32>
              %350 = ascvec.and %331, %258, %206 : <ui32>, <ui32> -> <ui32>
              %351 = ascvec.add %349, %350, %206 : <ui32>
              %352 = ascvec.shr %351, 16, %206 : <ui32> -> <ui32>
              %353 = ascvec.min_scalar %352, %c126_i32, %206 : <ui32>, i32
              %354 = ascvec.or %353, %340, %206 : <ui32>, <ui32> -> <ui32>
              %355 = ascvec.cmp_gt_scalar %341, %c2139095040_i32, %206 : <ui32>, i32 -> !ascvec.mask
              %356 = ascvec.select %264, %354, %355 : <ui32>
              ascvec.storealign_pack %arg15[%332], %356, %206 {mode = "b32_to_b8"} : memref<16384xui8, 1>, index, <ui32>
            }
          }
          ascvec.mem_bar vst_vld
          %mask, %remaining_out = ascvec.update_mask %c64_i32_10 elem_bits 16 : !ascvec.mask
          scf.for %arg19 = %c0_i32_12 to %arg18 step %c1_i32  : i32 {
            %207 = arith.muli %arg19, %c128_i32 : i32
            %208 = arith.index_cast %207 : i32 to index
            %209 = ascvec.loadalign %arg13[%208] : memref<512xui32, 1>, index -> <ui32>
            %210 = arith.addi %207, %c64_i32_10 : i32
            %211 = arith.index_cast %210 : i32 to index
            %212 = ascvec.loadalign %arg13[%211] : memref<512xui32, 1>, index -> <ui32>
            %213 = ascvec.shl %212, 8, %206 : <ui32> -> <ui32>
            %214 = ascvec.or %209, %213, %206 : <ui32>, <ui32> -> <ui32>
            %215 = ascvec.pack %214 src_elem_bits 32 dst_elem_bits 16 {part = "low"} : <ui32> -> <ui16>
            %216 = arith.extsi %arg19 : i32 to i64
            %217 = arith.muli %216, %arg10 : i64
            %218 = arith.addi %217, %arg11 : i64
            %219 = arith.index_cast %218 : i64 to index
            ascvec.storealign %arg17[%219], %215, %mask : memref<?xui16, 1>, index, <ui16>
          }
          ascvec.mem_bar vst_vld
        } {cannir.vf_mode = "simd"}
        ascvec.release_buf v, %83, 0 {cannir.buffer.generated_sync}
        ascvec.release_buf v, %78, 0 {cannir.buffer.generated_sync}
        ascvec.release_buf v, %42, 0 {cannir.buffer.generated_sync}
      }
      %91 = cannir.extract_layout %arg1 : !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
      %92 = cannir.extract_shape %91 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, -1)>
      %93 = cannir.extract_dim %92[0] : !cannir.shape<(-1, -1)>
      %94 = arith.minsi %33, %93 : i64
      %95 = arith.subi %93, %94 : i64
      %96 = arith.minsi %95, %c1_i64 : i64
      %97 = arith.cmpi sle, %33, %93 : i64
      %98 = arith.select %97, %96, %c0_i64 : i64
      %99 = arith.cmpi sgt, %98, %c1_i64 : i64
      %100 = arith.select %99, %c1_i64, %98 : i64
      %101 = cannir.extract_dim %92[1] : !cannir.shape<(-1, -1)>
      %102 = arith.divsi %101, %55 : i64
      %103 = arith.minsi %35, %102 : i64
      %104 = arith.muli %103, %36 : i64
      %105 = arith.subi %101, %104 : i64
      %106 = arith.minsi %36, %105 : i64
      %107 = arith.cmpi sle, %35, %102 : i64
      %108 = arith.select %107, %106, %c0_i64 : i64
      %109 = arith.cmpi sgt, %108, %c16384_i64 : i64
      %110 = arith.select %109, %c16384_i64, %108 : i64
      %111 = cannir.extract_physical_layout %arg1 : !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
      %112 = cannir.extract_stride %111 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
      %113 = cannir.extract_dim %112[0] : !cannir.stride<(-1, 1)>
      %114 = arith.muli %33, %113 : i64
      %115 = arith.addi %114, %65 : i64
      %116 = arith.muli %115, %c8_i64 : i64
      %117 = ascscalar.memref_address %alloc_0 : memref<1xui8>
      %118 = arith.divui %116, %c8_i64 : i64
      %119 = arith.addi %117, %118 : i64
      %120 = ascscalar.bind_gm_ptr %119 : memref<1xui8>
      %121 = ascscalar.bind_local_ptr %76 : memref<1xui8, 1>
      %122 = cannir.extract_stride %91 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
      %123 = cannir.extract_dim %122[0] : !cannir.stride<(-1, 1)>
      %124 = arith.cmpi ugt, %100, %c0_i64 : i64
      %125 = arith.cmpi ugt, %110, %c0_i64 : i64
      %126 = arith.andi %124, %125 : i1
      scf.if %126 {
        ascvec.get_buf mte3, %78, 0 {cannir.buffer.generated_sync}
        %172 = arith.trunci %100 : i64 to i32
        %173 = arith.trunci %110 : i64 to i32
        ascvec.copy_ub2gm %120[%c0], %121[%c0], %172, %173, %123, %c16384_i32 l2_cache_ctl %c0_i32 : memref<1xui8>, memref<1xui8, 1>
        ascvec.release_buf mte3, %78, 0 {cannir.buffer.generated_sync}
      }
      %127 = arith.remsi %36, %c32_i64 : i64
      %128 = arith.cmpi ne, %127, %c0_i64 : i64
      %129 = arith.cmpi slt, %127, %c0_i64 : i64
      %130 = arith.andi %128, %129 : i1
      %131 = arith.divsi %36, %c32_i64 : i64
      %132 = arith.extui %130 : i1 to i64
      %133 = arith.subi %131, %132 : i64
      %134 = cannir.extract_layout %arg2 : !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
      %135 = cannir.extract_shape %134 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, -1)>
      %136 = cannir.extract_dim %135[0] : !cannir.shape<(-1, -1)>
      %137 = arith.minsi %33, %136 : i64
      %138 = arith.subi %136, %137 : i64
      %139 = arith.minsi %138, %c1_i64 : i64
      %140 = arith.cmpi sle, %33, %136 : i64
      %141 = arith.select %140, %139, %c0_i64 : i64
      %142 = arith.cmpi sgt, %141, %c1_i64 : i64
      %143 = arith.select %142, %c1_i64, %141 : i64
      %144 = cannir.extract_dim %135[1] : !cannir.shape<(-1, -1)>
      %145 = arith.maxsi %133, %c1_i64 : i64
      %146 = arith.divsi %144, %145 : i64
      %147 = arith.minsi %35, %146 : i64
      %148 = arith.muli %147, %133 : i64
      %149 = arith.subi %144, %148 : i64
      %150 = arith.minsi %133, %149 : i64
      %151 = arith.cmpi sle, %35, %146 : i64
      %152 = arith.select %151, %150, %c0_i64 : i64
      %153 = arith.cmpi sgt, %152, %c1024_i64 : i64
      %154 = arith.select %153, %c1024_i64, %152 : i64
      %155 = cannir.extract_physical_layout %arg2 : !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
      %156 = cannir.extract_stride %155 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
      %157 = cannir.extract_dim %156[0] : !cannir.stride<(-1, 1)>
      %158 = arith.muli %33, %157 : i64
      %159 = arith.muli %35, %133 : i64
      %160 = arith.addi %158, %159 : i64
      %161 = arith.muli %160, %c8_i64 : i64
      %162 = ascscalar.memref_address %alloc_1 : memref<1xui8>
      %163 = arith.divui %161, %c8_i64 : i64
      %164 = arith.addi %162, %163 : i64
      %165 = ascscalar.bind_gm_ptr %164 : memref<1xui8>
      %166 = ascscalar.bind_local_ptr %81 : memref<1xui8, 1>
      %167 = cannir.extract_stride %134 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
      %168 = cannir.extract_dim %167[0] : !cannir.stride<(-1, 1)>
      %169 = arith.cmpi ugt, %143, %c0_i64 : i64
      %170 = arith.cmpi ugt, %154, %c0_i64 : i64
      %171 = arith.andi %169, %170 : i1
      scf.if %171 {
        ascvec.get_buf mte3, %83, 0 {cannir.buffer.generated_sync}
        %172 = arith.trunci %143 : i64 to i32
        %173 = arith.trunci %154 : i64 to i32
        ascvec.copy_ub2gm %165[%c0], %166[%c0], %172, %173, %168, %c1024_i32 l2_cache_ctl %c0_i32 : memref<1xui8>, memref<1xui8, 1>
        ascvec.release_buf mte3, %83, 0 {cannir.buffer.generated_sync}
      }
      scf.yield %44 : i64
    }
    return
  }
}
