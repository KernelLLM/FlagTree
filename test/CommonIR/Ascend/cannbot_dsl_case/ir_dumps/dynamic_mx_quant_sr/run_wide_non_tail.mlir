module {
  func.func @run_wide_non_tail_8768dfb40d5d1165cd94678ca098851b(%arg0: !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg1: !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg2: !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: !cannir.tensor<<gm, i64>, <<(2)>, <(1)>>, <<(2)>, <(1)>>, identity>) attributes {cannbotdsl.contract_digest = "8768dfb40d5d1165cd94678ca098851bb79b64da87af61debaf2a345c4afee3e", cannbotdsl.public_slot_count = 7 : i64, cannbotdsl.slot_dtype_bits = [16, 8, 8, 64, 64, 64, 64], cannbotdsl.slot_dtype_codes = [4, 1, 1, 0, 0, 0, 0], cannbotdsl.slot_dtype_lanes = [1, 1, 1, 1, 1, 1, 1], cannbotdsl.slot_flags = [1, 1, 1, 1, 1, 1, 1], cannbotdsl.slot_kinds = [1, 1, 1, 2, 2, 2, 1], cannbotdsl.slot_type_digests = ["5d7cdc96dc21b3b739e32aeccf58c1df113bec54701c75107ed650f2544a07da", "171ddd09eb070fb4a6413ae87b2ceecddc233a9a173a0404e96285a4bb8b8659", "891bcd21e721f8d1a666bcca4e8705ec63796c1add8ae000536d0d98f0274d3a", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "4126671a83041647cd98afed881383228865da287ea44573939f32b1d956765b"], cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.host} {
    cannir.kernel_launch @kernel__DynamicMxQuantSrKernel_run_wide_non_tail_0(%arg5) [%arg0, %arg1, %arg2, %arg3, %arg4, %arg6] {cannbotdsl.profile = {indices = array<i64: 0, 1, 2, 5>, name = "_DynamicMxQuantSrKernel_run_wide_non_tail", op_type = "_DynamicMxQuantSrKernel_run_wide_non_tail", roles = array<i64: 0, 1, 1, 0>}} : !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, i64, i64, !cannir.tensor<<gm, i64>, <<(2)>, <(1)>>, <<(2)>, <(1)>>, identity>
    return
  }
  func.func @kernel__DynamicMxQuantSrKernel_run_wide_non_tail_0(%arg0: !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg1: !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg2: !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg3: i64, %arg4: i64, %arg5: !cannir.tensor<<gm, i64>, <<(2)>, <(1)>>, <<(2)>, <(1)>>, identity>) attributes {cannir.buffer.auto_sync, cannir.buffer.sync_analyzed, cannir.buffer.sync_verified, cannir.buffer.tensor_state, cannir.core = "aiv", cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.device, cannir.explicit_ring_buffer_cursors, cannir.kernel_group = "kernel__DynamicMxQuantSrKernel_run_wide_non_tail_0", cannir.resource.ids_prepared} {
    %c1024_i32 = arith.constant 1024 : i32
    %c16384_i32 = arith.constant 16384 : i32
    %c1 = arith.constant 1 : index
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
      %29 = arith.remsi %arg3, %c256_i64 : i64
      %30 = arith.cmpi ne, %29, %c0_i64 : i64
      %31 = arith.cmpi slt, %29, %c0_i64 : i64
      %32 = arith.andi %30, %31 : i1
      %33 = arith.select %32, %c256_i64, %c0_i64 : i64
      %34 = arith.addi %29, %33 overflow<nsw> : i64
      %35 = arith.cmpi eq, %34, %c0_i64 : i64
      %36 = arith.cmpi eq, %arg4, %c64_i64 : i64
      %37 = arith.select %35, %36, %false : i1
      scf.yield %37 : i1
    } else {
      scf.yield %false : i1
    }
    %17 = scf.if %16 -> (i64) {
      scf.yield %c256_i64 : i64
    } else {
      %29 = arith.remsi %arg3, %c128_i64 : i64
      %30 = arith.cmpi ne, %29, %c0_i64 : i64
      %31 = arith.cmpi slt, %29, %c0_i64 : i64
      %32 = arith.andi %30, %31 : i1
      %33 = arith.select %32, %c128_i64, %c0_i64 : i64
      %34 = arith.addi %29, %33 overflow<nsw> : i64
      %35 = arith.cmpi eq, %34, %c0_i64 : i64
      %36 = arith.select %15, %35, %false : i1
      %37 = arith.select %36, %c128_i64, %c64_i64 : i64
      scf.yield %37 : i64
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
      %88 = arith.remsi %arg4, %c64_i64 : i64
      %89 = arith.cmpi ne, %88, %c0_i64 : i64
      %90 = arith.cmpi slt, %88, %c0_i64 : i64
      %91 = arith.andi %89, %90 : i1
      %92 = arith.divsi %arg4, %c64_i64 : i64
      %93 = arith.extui %91 : i1 to i64
      %94 = arith.subi %92, %93 : i64
      scf.for %arg8 = %c0_i64 to %94 step %c1_i64  : i64 {
        %176 = arith.muli %arg8, %c64_i64 : i64
        %177 = arith.muli %37, %arg4 : i64
        %178 = arith.addi %177, %176 : i64
        %179 = arith.muli %39, %17 : i64
        %180 = arith.remsi %179, %c16_i64 : i64
        %181 = arith.cmpi ne, %180, %c0_i64 : i64
        %182 = arith.cmpi slt, %180, %c0_i64 : i64
        %183 = arith.andi %181, %182 : i1
        %184 = arith.divsi %179, %c16_i64 : i64
        %185 = arith.extui %183 : i1 to i64
        %186 = arith.subi %184, %185 : i64
        %187 = arith.remsi %arg3, %c16_i64 : i64
        %188 = arith.cmpi ne, %187, %c0_i64 : i64
        %189 = arith.cmpi slt, %187, %c0_i64 : i64
        %190 = arith.andi %188, %189 : i1
        %191 = arith.divsi %arg3, %c16_i64 : i64
        %192 = arith.extui %190 : i1 to i64
        %193 = arith.subi %191, %192 : i64
        %194 = arith.trunci %1 : i64 to i32
        %195 = arith.extui %194 : i32 to i64
        %196 = arith.subi %1, %195 : i64
        %197 = arith.remsi %196, %c4294967296_i64 : i64
        %198 = arith.cmpi ne, %197, %c0_i64 : i64
        %199 = arith.cmpi slt, %197, %c0_i64 : i64
        %200 = arith.andi %198, %199 : i1
        %201 = arith.divsi %196, %c4294967296_i64 : i64
        %202 = arith.extui %200 : i1 to i64
        %203 = arith.subi %201, %202 : i64
        %204 = arith.trunci %203 : i64 to i32
        %205 = arith.trunci %17 : i64 to i32
        %206 = arith.divsi %205, %c16_i32 : i32
        ascvec.vf(%186, %178, %193, %3, %194, %204, %alloc_10, %206 : i64, i64, i64, i64, i32, i32, memref<4096xi32, 1>, i32) {
        ^bb0(%arg9: i64, %arg10: i64, %arg11: i64, %arg12: i64, %arg13: i32, %arg14: i32, %arg15: memref<4096xi32, 1>, %arg16: i32):
          %c1_i32 = arith.constant 1 : i32
          %c192_i32 = arith.constant 192 : i32
          %c128_i32 = arith.constant 128 : i32
          %c64_i32_11 = arith.constant 64 : i32
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
          %c0_i64_12 = arith.constant 0 : i64
          %c4294967296_i64_13 = arith.constant 4294967296 : i64
          %c0_i32_14 = arith.constant 0 : i32
          %210 = ascvec.create_mask : !ascvec.mask
          %211 = ascvec.arange %c0_i32_14 : i32 -> <ui32>
          scf.for %arg17 = %c0_i32_14 to %arg16 step %c1_i32  : i32 {
            %212 = arith.extsi %arg17 : i32 to i64
            %213 = arith.addi %arg9, %212 : i64
            %214 = arith.trunci %arg10 : i64 to i32
            %215 = arith.extui %214 : i32 to i64
            %216 = arith.subi %arg10, %215 : i64
            %217 = arith.remsi %216, %c4294967296_i64_13 : i64
            %218 = arith.cmpi ne, %217, %c0_i64_12 : i64
            %219 = arith.cmpi slt, %217, %c0_i64_12 : i64
            %220 = arith.andi %218, %219 : i1
            %221 = arith.divsi %216, %c4294967296_i64_13 : i64
            %222 = arith.extui %220 : i1 to i64
            %223 = arith.subi %221, %222 : i64
            %224 = arith.trunci %223 : i64 to i32
            %225 = arith.trunci %213 : i64 to i32
            %226 = arith.extui %225 : i32 to i64
            %227 = arith.subi %213, %226 : i64
            %228 = arith.remsi %227, %c4294967296_i64_13 : i64
            %229 = arith.cmpi ne, %228, %c0_i64_12 : i64
            %230 = arith.cmpi slt, %228, %c0_i64_12 : i64
            %231 = arith.andi %229, %230 : i1
            %232 = arith.divsi %227, %c4294967296_i64_13 : i64
            %233 = arith.extui %231 : i1 to i64
            %234 = arith.subi %232, %233 : i64
            %235 = arith.trunci %234 : i64 to i32
            %236 = ascvec.duplicate_scalar %214, %210 : i32 -> <ui32>
            %237 = ascvec.duplicate_scalar %224, %210 : i32 -> <ui32>
            %238 = ascvec.duplicate_scalar %c0_i32_14, %210 : i32 -> <ui32>
            %carry, %dst = ascvec.addc %236, %211, %210 : <ui32>, <ui32> -> !ascvec.mask, <ui32>
            %carry_15, %dst_16 = ascvec.addc %237, %238, %carry, %210 : <ui32>, <ui32> -> !ascvec.mask, <ui32>
            %239 = ascvec.duplicate_scalar %225, %210 : i32 -> <ui32>
            %240 = ascvec.duplicate_scalar %235, %210 : i32 -> <ui32>
            %241 = arith.trunci %arg11 : i64 to i32
            %242 = arith.extui %241 : i32 to i64
            %243 = arith.subi %arg11, %242 : i64
            %244 = arith.remsi %243, %c4294967296_i64_13 : i64
            %245 = arith.cmpi ne, %244, %c0_i64_12 : i64
            %246 = arith.cmpi slt, %244, %c0_i64_12 : i64
            %247 = arith.andi %245, %246 : i1
            %248 = arith.divsi %243, %c4294967296_i64_13 : i64
            %249 = arith.extui %247 : i1 to i64
            %250 = arith.subi %248, %249 : i64
            %251 = arith.trunci %250 : i64 to i32
            %252 = ascvec.duplicate_scalar %241, %210 : i32 -> <ui32>
            %dst0, %dst1 = ascvec.mull %dst, %252, %210 : <ui32>, <ui32> -> <ui32>, <ui32>
            %253 = ascvec.mul_scalar %dst, %251, %210 : <ui32>, i32
            %254 = ascvec.add %dst1, %253, %210 : <ui32>
            %255 = ascvec.mul_scalar %dst_16, %241, %210 : <ui32>, i32
            %256 = ascvec.add %254, %255, %210 : <ui32>
            %carry_17, %dst_18 = ascvec.addc %dst0, %239, %210 : <ui32>, <ui32> -> !ascvec.mask, <ui32>
            %carry_19, %dst_20 = ascvec.addc %256, %240, %carry_17, %210 : <ui32>, <ui32> -> !ascvec.mask, <ui32>
            %257 = arith.trunci %arg12 : i64 to i32
            %258 = arith.extui %257 : i32 to i64
            %259 = arith.subi %arg12, %258 : i64
            %260 = arith.remsi %259, %c4294967296_i64_13 : i64
            %261 = arith.cmpi ne, %260, %c0_i64_12 : i64
            %262 = arith.cmpi slt, %260, %c0_i64_12 : i64
            %263 = arith.andi %261, %262 : i1
            %264 = arith.divsi %259, %c4294967296_i64_13 : i64
            %265 = arith.extui %263 : i1 to i64
            %266 = arith.subi %264, %265 : i64
            %267 = arith.trunci %266 : i64 to i32
            %268 = arith.divui %257, %c4_i32 : i32
            %269 = arith.remui %267, %c4_i32 : i32
            %270 = arith.muli %269, %c1073741824_i32 : i32
            %271 = arith.addi %268, %270 : i32
            %272 = arith.divui %267, %c4_i32 : i32
            %273 = ascvec.shr %dst_18, 14, %210 : <ui32> -> <ui32>
            %274 = ascvec.shl %dst_20, 18, %210 : <ui32> -> <ui32>
            %275 = ascvec.or %273, %274, %210 : <ui32>, <ui32> -> <ui32>
            %276 = ascvec.shr %dst_20, 14, %210 : <ui32> -> <ui32>
            %277 = ascvec.duplicate_scalar %271, %210 : i32 -> <ui32>
            %278 = ascvec.duplicate_scalar %272, %210 : i32 -> <ui32>
            %carry_21, %dst_22 = ascvec.addc %277, %275, %210 : <ui32>, <ui32> -> !ascvec.mask, <ui32>
            %carry_23, %dst_24 = ascvec.addc %278, %276, %carry_21, %210 : <ui32>, <ui32> -> !ascvec.mask, <ui32>
            %279 = ascvec.duplicate_scalar %c16383_i32, %210 : i32 -> <ui32>
            %280 = ascvec.and %dst_18, %279, %210 : <ui32>, <ui32> -> <ui32>
            %281 = ascvec.duplicate_scalar %c-766435501_i32, %210 : i32 -> <ui32>
            %282 = ascvec.duplicate_scalar %c-845247145_i32, %210 : i32 -> <ui32>
            %dst0_25, %dst1_26 = ascvec.mull %238, %281, %210 : <ui32>, <ui32> -> <ui32>, <ui32>
            %dst0_27, %dst1_28 = ascvec.mull %dst_22, %282, %210 : <ui32>, <ui32> -> <ui32>, <ui32>
            %283 = ascvec.xor %dst1_28, %280, %210 : <ui32>, <ui32> -> <ui32>
            %284 = ascvec.xor %dst1_26, %dst_24, %210 : <ui32>, <ui32> -> <ui32>
            %285 = ascvec.duplicate_scalar %arg13, %210 : i32 -> <ui32>
            %286 = ascvec.duplicate_scalar %arg14, %210 : i32 -> <ui32>
            %287 = ascvec.xor %283, %285, %210 : <ui32>, <ui32> -> <ui32>
            %288 = ascvec.xor %284, %286, %210 : <ui32>, <ui32> -> <ui32>
            %dst0_29, %dst1_30 = ascvec.mull %287, %281, %210 : <ui32>, <ui32> -> <ui32>, <ui32>
            %dst0_31, %dst1_32 = ascvec.mull %288, %282, %210 : <ui32>, <ui32> -> <ui32>, <ui32>
            %289 = ascvec.xor %dst1_32, %dst0_27, %210 : <ui32>, <ui32> -> <ui32>
            %290 = ascvec.xor %dst1_30, %dst0_25, %210 : <ui32>, <ui32> -> <ui32>
            %291 = arith.addi %arg13, %c-1640531527_i32 : i32
            %292 = ascvec.duplicate_scalar %291, %210 : i32 -> <ui32>
            %293 = arith.addi %arg14, %c-1150833019_i32 : i32
            %294 = ascvec.duplicate_scalar %293, %210 : i32 -> <ui32>
            %295 = ascvec.xor %289, %292, %210 : <ui32>, <ui32> -> <ui32>
            %296 = ascvec.xor %290, %294, %210 : <ui32>, <ui32> -> <ui32>
            %dst0_33, %dst1_34 = ascvec.mull %295, %281, %210 : <ui32>, <ui32> -> <ui32>, <ui32>
            %dst0_35, %dst1_36 = ascvec.mull %296, %282, %210 : <ui32>, <ui32> -> <ui32>, <ui32>
            %297 = ascvec.xor %dst1_36, %dst0_31, %210 : <ui32>, <ui32> -> <ui32>
            %298 = ascvec.xor %dst1_34, %dst0_29, %210 : <ui32>, <ui32> -> <ui32>
            %299 = arith.addi %arg13, %c1013904242_i32 : i32
            %300 = ascvec.duplicate_scalar %299, %210 : i32 -> <ui32>
            %301 = arith.addi %arg14, %c1993301258_i32 : i32
            %302 = ascvec.duplicate_scalar %301, %210 : i32 -> <ui32>
            %303 = ascvec.xor %297, %300, %210 : <ui32>, <ui32> -> <ui32>
            %304 = ascvec.xor %298, %302, %210 : <ui32>, <ui32> -> <ui32>
            %dst0_37, %dst1_38 = ascvec.mull %303, %281, %210 : <ui32>, <ui32> -> <ui32>, <ui32>
            %dst0_39, %dst1_40 = ascvec.mull %304, %282, %210 : <ui32>, <ui32> -> <ui32>, <ui32>
            %305 = ascvec.xor %dst1_40, %dst0_35, %210 : <ui32>, <ui32> -> <ui32>
            %306 = ascvec.xor %dst1_38, %dst0_33, %210 : <ui32>, <ui32> -> <ui32>
            %307 = arith.addi %arg13, %c-626627285_i32 : i32
            %308 = ascvec.duplicate_scalar %307, %210 : i32 -> <ui32>
            %309 = arith.addi %arg14, %c842468239_i32 : i32
            %310 = ascvec.duplicate_scalar %309, %210 : i32 -> <ui32>
            %311 = ascvec.xor %305, %308, %210 : <ui32>, <ui32> -> <ui32>
            %312 = ascvec.xor %306, %310, %210 : <ui32>, <ui32> -> <ui32>
            %dst0_41, %dst1_42 = ascvec.mull %311, %281, %210 : <ui32>, <ui32> -> <ui32>, <ui32>
            %dst0_43, %dst1_44 = ascvec.mull %312, %282, %210 : <ui32>, <ui32> -> <ui32>, <ui32>
            %313 = ascvec.xor %dst1_44, %dst0_39, %210 : <ui32>, <ui32> -> <ui32>
            %314 = ascvec.xor %dst1_42, %dst0_37, %210 : <ui32>, <ui32> -> <ui32>
            %315 = arith.addi %arg13, %c2027808484_i32 : i32
            %316 = ascvec.duplicate_scalar %315, %210 : i32 -> <ui32>
            %317 = arith.addi %arg14, %c-308364780_i32 : i32
            %318 = ascvec.duplicate_scalar %317, %210 : i32 -> <ui32>
            %319 = ascvec.xor %313, %316, %210 : <ui32>, <ui32> -> <ui32>
            %320 = ascvec.xor %314, %318, %210 : <ui32>, <ui32> -> <ui32>
            %dst0_45, %dst1_46 = ascvec.mull %319, %281, %210 : <ui32>, <ui32> -> <ui32>, <ui32>
            %dst0_47, %dst1_48 = ascvec.mull %320, %282, %210 : <ui32>, <ui32> -> <ui32>, <ui32>
            %321 = ascvec.xor %dst1_48, %dst0_43, %210 : <ui32>, <ui32> -> <ui32>
            %322 = ascvec.xor %dst1_46, %dst0_41, %210 : <ui32>, <ui32> -> <ui32>
            %323 = arith.addi %arg13, %c387276957_i32 : i32
            %324 = ascvec.duplicate_scalar %323, %210 : i32 -> <ui32>
            %325 = arith.addi %arg14, %c-1459197799_i32 : i32
            %326 = ascvec.duplicate_scalar %325, %210 : i32 -> <ui32>
            %327 = ascvec.xor %321, %324, %210 : <ui32>, <ui32> -> <ui32>
            %328 = ascvec.xor %322, %326, %210 : <ui32>, <ui32> -> <ui32>
            %dst0_49, %dst1_50 = ascvec.mull %327, %281, %210 : <ui32>, <ui32> -> <ui32>, <ui32>
            %dst0_51, %dst1_52 = ascvec.mull %328, %282, %210 : <ui32>, <ui32> -> <ui32>, <ui32>
            %329 = ascvec.xor %dst1_52, %dst0_47, %210 : <ui32>, <ui32> -> <ui32>
            %330 = ascvec.xor %dst1_50, %dst0_45, %210 : <ui32>, <ui32> -> <ui32>
            %331 = arith.addi %arg13, %c-1253254570_i32 : i32
            %332 = ascvec.duplicate_scalar %331, %210 : i32 -> <ui32>
            %333 = arith.addi %arg14, %c1684936478_i32 : i32
            %334 = ascvec.duplicate_scalar %333, %210 : i32 -> <ui32>
            %335 = ascvec.xor %329, %332, %210 : <ui32>, <ui32> -> <ui32>
            %336 = ascvec.xor %330, %334, %210 : <ui32>, <ui32> -> <ui32>
            %dst0_53, %dst1_54 = ascvec.mull %335, %281, %210 : <ui32>, <ui32> -> <ui32>, <ui32>
            %dst0_55, %dst1_56 = ascvec.mull %336, %282, %210 : <ui32>, <ui32> -> <ui32>, <ui32>
            %337 = ascvec.xor %dst1_56, %dst0_51, %210 : <ui32>, <ui32> -> <ui32>
            %338 = ascvec.xor %dst1_54, %dst0_49, %210 : <ui32>, <ui32> -> <ui32>
            %339 = arith.addi %arg13, %c1401181199_i32 : i32
            %340 = ascvec.duplicate_scalar %339, %210 : i32 -> <ui32>
            %341 = arith.addi %arg14, %c534103459_i32 : i32
            %342 = ascvec.duplicate_scalar %341, %210 : i32 -> <ui32>
            %343 = ascvec.xor %337, %340, %210 : <ui32>, <ui32> -> <ui32>
            %344 = ascvec.xor %338, %342, %210 : <ui32>, <ui32> -> <ui32>
            %dst0_57, %dst1_58 = ascvec.mull %343, %281, %210 : <ui32>, <ui32> -> <ui32>, <ui32>
            %dst0_59, %dst1_60 = ascvec.mull %344, %282, %210 : <ui32>, <ui32> -> <ui32>, <ui32>
            %345 = ascvec.xor %dst1_60, %dst0_55, %210 : <ui32>, <ui32> -> <ui32>
            %346 = ascvec.xor %dst1_58, %dst0_53, %210 : <ui32>, <ui32> -> <ui32>
            %347 = arith.addi %arg13, %c-239350328_i32 : i32
            %348 = ascvec.duplicate_scalar %347, %210 : i32 -> <ui32>
            %349 = arith.addi %arg14, %c-616729560_i32 : i32
            %350 = ascvec.duplicate_scalar %349, %210 : i32 -> <ui32>
            %351 = ascvec.xor %345, %348, %210 : <ui32>, <ui32> -> <ui32>
            %352 = ascvec.xor %346, %350, %210 : <ui32>, <ui32> -> <ui32>
            %dst0_61, %dst1_62 = ascvec.mull %351, %281, %210 : <ui32>, <ui32> -> <ui32>, <ui32>
            %dst0_63, %dst1_64 = ascvec.mull %352, %282, %210 : <ui32>, <ui32> -> <ui32>, <ui32>
            %353 = ascvec.xor %dst1_64, %dst0_59, %210 : <ui32>, <ui32> -> <ui32>
            %354 = ascvec.xor %dst1_62, %dst0_57, %210 : <ui32>, <ui32> -> <ui32>
            %355 = arith.addi %arg13, %c-1879881855_i32 : i32
            %356 = ascvec.duplicate_scalar %355, %210 : i32 -> <ui32>
            %357 = arith.addi %arg14, %c-1767562579_i32 : i32
            %358 = ascvec.duplicate_scalar %357, %210 : i32 -> <ui32>
            %359 = ascvec.xor %353, %356, %210 : <ui32>, <ui32> -> <ui32>
            %360 = ascvec.xor %354, %358, %210 : <ui32>, <ui32> -> <ui32>
            %361 = arith.muli %arg17, %c256_i32 : i32
            %362 = ascvec.reinterpret %359 : <ui32> -> <i32>
            %363 = arith.index_cast %361 : i32 to index
            ascvec.storealign %arg15[%363], %362, %210 : memref<4096xi32, 1>, index, <i32>
            %364 = arith.addi %361, %c64_i32_11 : i32
            %365 = ascvec.reinterpret %dst0_63 : <ui32> -> <i32>
            %366 = arith.index_cast %364 : i32 to index
            ascvec.storealign %arg15[%366], %365, %210 : memref<4096xi32, 1>, index, <i32>
            %367 = arith.addi %361, %c128_i32 : i32
            %368 = ascvec.reinterpret %360 : <ui32> -> <i32>
            %369 = arith.index_cast %367 : i32 to index
            ascvec.storealign %arg15[%369], %368, %210 : memref<4096xi32, 1>, index, <i32>
            %370 = arith.addi %361, %c192_i32 : i32
            %371 = ascvec.reinterpret %dst0_61 : <ui32> -> <i32>
            %372 = arith.index_cast %370 : i32 to index
            ascvec.storealign %arg15[%372], %371, %210 : memref<4096xi32, 1>, index, <i32>
          }
          ascvec.mem_bar vst_vld
        } {cannir.vf_mode = "simd"}
        %207 = arith.divsi %205, %c32_i32 : i32
        %208 = arith.divsi %205, %c64_i32 : i32
        ascvec.get_buf v, %46, 0 {cannir.buffer.generated_sync}
        ascvec.get_buf v, %82, 0 {cannir.buffer.generated_sync}
        ascvec.get_buf v, %87, 0 {cannir.buffer.generated_sync}
        %209 = ascscalar.bitcast_memref %86 byte_offset = %c0_i64 : memref<1024xui8, 1> -> memref<?xui16, 1>
        ascvec.vf(%arg4, %176, %77, %alloc_9, %alloc_10, %81, %207, %209, %208 : i64, i64, memref<16384xbf16, 1>, memref<512xui32, 1>, memref<4096xi32, 1>, memref<16384xui8, 1>, i32, memref<?xui16, 1>, i32) {
        ^bb0(%arg9: i64, %arg10: i64, %arg11: memref<16384xbf16, 1>, %arg12: memref<512xui32, 1>, %arg13: memref<4096xi32, 1>, %arg14: memref<16384xui8, 1>, %arg15: i32, %arg16: memref<?xui16, 1>, %arg17: i32):
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
          %c64_i32_11 = arith.constant 64 : i32
          %c2139160576_i32 = arith.constant 2139160576 : i32
          %c254_i32 = arith.constant 254 : i32
          %c255_i32 = arith.constant 255 : i32
          %c73400321_i32 = arith.constant 73400321 : i32
          %c-65011713_i32 = arith.constant -65011713 : i32
          %c2139095040_i32 = arith.constant 2139095040 : i32
          %c1_i32 = arith.constant 1 : i32
          %c2147483647_i32 = arith.constant 2147483647 : i32
          %c32_i32_12 = arith.constant 32 : i32
          %c0_i32_13 = arith.constant 0 : i32
          %210 = ascvec.create_mask : !ascvec.mask
          scf.for %arg18 = %c0_i32_13 to %arg15 step %c1_i32  : i32 {
            %211 = ascvec.duplicate_scalar %c0_i32_13, %210 : i32 -> <ui32>
            %212 = scf.for %arg19 = %c0_i32_13 to %c32_i32_12 step %c1_i32 iter_args(%arg20 = %211) -> (!ascvec.vreg<ui32>)  : i32 {
              %225 = arith.muli %arg18, %c32_i32_12 : i32
              %226 = arith.addi %225, %arg19 : i32
              %227 = arith.extsi %226 : i32 to i64
              %228 = arith.muli %227, %arg9 : i64
              %229 = arith.addi %228, %arg10 : i64
              %230 = arith.index_cast %229 : i64 to index
              %231 = ascvec.loadalign_unpack %arg11[%230] {mode = "b16_to_b32"} : memref<16384xbf16, 1>, index -> <bf16>
              %232 = ascvec.cast %231, %210 {rounding = 6 : i32} : <bf16> -> <f32>
              %233 = ascvec.reinterpret %232 : <f32> -> <ui32>
              %234 = ascvec.duplicate_scalar %c2147483647_i32, %210 : i32 -> <ui32>
              %235 = ascvec.and %233, %234, %210 : <ui32>, <ui32> -> <ui32>
              %236 = ascvec.max %arg20, %235, %210 : <ui32>
              scf.yield %236 : !ascvec.vreg<ui32>
            }
            %213 = ascvec.cmp_ge_scalar %212, %c2139095040_i32, %210 : <ui32>, i32 -> !ascvec.mask
            %214 = ascvec.add_scalar %212, %c-65011713_i32, %210 : <ui32>, i32
            %215 = ascvec.cmp_gt_scalar %212, %c73400321_i32, %210 : <ui32>, i32 -> !ascvec.mask
            %216 = ascvec.shr %214, 23, %215 : <ui32> -> <ui32>
            %217 = ascvec.duplicate_scalar %c255_i32, %213 merge %216 : !ascvec.vreg<ui32> {mode = "merging"} : i32 -> <ui32>
            %218 = ascvec.duplicate_scalar %c254_i32, %210 : i32 -> <ui32>
            %219 = ascvec.sub %218, %217, %210 : <ui32>
            %220 = ascvec.shl %219, 23, %210 : <ui32> -> <ui32>
            %221 = ascvec.duplicate_scalar %c2139160576_i32, %213 merge %220 : !ascvec.vreg<ui32> {mode = "merging"} : i32 -> <ui32>
            %222 = ascvec.reinterpret %221 : <ui32> -> <f32>
            %223 = arith.muli %arg18, %c64_i32_11 : i32
            %224 = arith.index_cast %223 : i32 to index
            ascvec.storealign %arg12[%224], %217, %210 : memref<512xui32, 1>, index, <ui32>
            scf.for %arg19 = %c0_i32_13 to %c8_i32 step %c1_i32  : i32 {
              %225 = arith.muli %arg18, %c512_i32 : i32
              %226 = arith.muli %arg19, %c64_i32_11 overflow<nsw> : i32
              %227 = arith.addi %225, %226 : i32
              %228 = arith.index_cast %227 : i32 to index
              %229 = ascvec.loadalign %arg13[%228] : memref<4096xi32, 1>, index -> <i32>
              %230 = ascvec.reinterpret %229 : <i32> -> <ui32>
              %231 = ascvec.shl %230, 8, %210 : <ui32> -> <ui32>
              %232 = ascvec.shr %230, 24, %210 : <ui32> -> <ui32>
              %233 = ascvec.or %231, %232, %210 : <ui32>, <ui32> -> <ui32>
              %234 = arith.muli %arg18, %c32_i32_12 : i32
              %235 = arith.muli %arg19, %c4_i32 overflow<nsw> : i32
              %236 = arith.addi %234, %235 : i32
              %237 = arith.extsi %236 : i32 to i64
              %238 = arith.muli %237, %arg9 : i64
              %239 = arith.addi %238, %arg10 : i64
              %240 = ascvec.shr %230, 16, %210 : <ui32> -> <ui32>
              %241 = arith.index_cast %239 : i64 to index
              %242 = ascvec.loadalign_unpack %arg11[%241] {mode = "b16_to_b32"} : memref<16384xbf16, 1>, index -> <bf16>
              %243 = ascvec.cast %242, %210 {rounding = 6 : i32} : <bf16> -> <f32>
              %244 = ascvec.mul %243, %222, %210 : <f32>
              %245 = ascvec.cast %244, %210 : <f32> -> <bf16>
              %246 = ascvec.cast %245, %210 {rounding = 6 : i32} : <bf16> -> <f32>
              %247 = ascvec.reinterpret %246 : <f32> -> <ui32>
              %248 = ascvec.shr %247, 24, %210 : <ui32> -> <ui32>
              %249 = ascvec.duplicate_scalar %c128_i32, %210 : i32 -> <ui32>
              %250 = ascvec.and %248, %249, %210 : <ui32>, <ui32> -> <ui32>
              %251 = ascvec.duplicate_scalar %c2147483647_i32, %210 : i32 -> <ui32>
              %252 = ascvec.and %247, %251, %210 : <ui32>, <ui32> -> <ui32>
              %253 = ascvec.shr %252, 4, %210 : <ui32> -> <ui32>
              %254 = ascvec.duplicate_scalar %c62914560_i32, %210 : i32 -> <ui32>
              %255 = ascvec.sub %253, %254, %210 : <ui32>
              %256 = ascvec.reinterpret %252 : <ui32> -> <f32>
              %257 = ascvec.mul_scalar %256, %cst, %210 : <f32>, f32
              %258 = ascvec.cast %257, %210 {rounding = 4 : i32} : <f32> -> <i32>
              %259 = ascvec.reinterpret %258 : <i32> -> <ui32>
              %260 = ascvec.cmp_ge_scalar %252, %c1015021568_i32, %210 : <ui32>, i32 -> !ascvec.mask
              %261 = ascvec.select %255, %259, %260 : <ui32>
              %262 = ascvec.duplicate_scalar %c65535_i32, %210 : i32 -> <ui32>
              %263 = ascvec.and %240, %262, %210 : <ui32>, <ui32> -> <ui32>
              %264 = ascvec.add %261, %263, %210 : <ui32>
              %265 = ascvec.shr %264, 16, %210 : <ui32> -> <ui32>
              %266 = ascvec.min_scalar %265, %c126_i32, %210 : <ui32>, i32
              %267 = ascvec.or %266, %250, %210 : <ui32>, <ui32> -> <ui32>
              %268 = ascvec.duplicate_scalar %c127_i32, %210 : i32 -> <ui32>
              %269 = ascvec.cmp_gt_scalar %252, %c2139095040_i32, %210 : <ui32>, i32 -> !ascvec.mask
              %270 = ascvec.select %268, %267, %269 : <ui32>
              ascvec.storealign_pack %arg14[%241], %270, %210 {mode = "b32_to_b8"} : memref<16384xui8, 1>, index, <ui32>
              %271 = arith.addi %236, %c1_i32 : i32
              %272 = arith.extsi %271 : i32 to i64
              %273 = arith.muli %272, %arg9 : i64
              %274 = arith.addi %273, %arg10 : i64
              %275 = ascvec.and %233, %262, %210 : <ui32>, <ui32> -> <ui32>
              %276 = arith.index_cast %274 : i64 to index
              %277 = ascvec.loadalign_unpack %arg11[%276] {mode = "b16_to_b32"} : memref<16384xbf16, 1>, index -> <bf16>
              %278 = ascvec.cast %277, %210 {rounding = 6 : i32} : <bf16> -> <f32>
              %279 = ascvec.mul %278, %222, %210 : <f32>
              %280 = ascvec.cast %279, %210 : <f32> -> <bf16>
              %281 = ascvec.cast %280, %210 {rounding = 6 : i32} : <bf16> -> <f32>
              %282 = ascvec.reinterpret %281 : <f32> -> <ui32>
              %283 = ascvec.shr %282, 24, %210 : <ui32> -> <ui32>
              %284 = ascvec.and %283, %249, %210 : <ui32>, <ui32> -> <ui32>
              %285 = ascvec.and %282, %251, %210 : <ui32>, <ui32> -> <ui32>
              %286 = ascvec.shr %285, 4, %210 : <ui32> -> <ui32>
              %287 = ascvec.sub %286, %254, %210 : <ui32>
              %288 = ascvec.reinterpret %285 : <ui32> -> <f32>
              %289 = ascvec.mul_scalar %288, %cst, %210 : <f32>, f32
              %290 = ascvec.cast %289, %210 {rounding = 4 : i32} : <f32> -> <i32>
              %291 = ascvec.reinterpret %290 : <i32> -> <ui32>
              %292 = ascvec.cmp_ge_scalar %285, %c1015021568_i32, %210 : <ui32>, i32 -> !ascvec.mask
              %293 = ascvec.select %287, %291, %292 : <ui32>
              %294 = ascvec.and %275, %262, %210 : <ui32>, <ui32> -> <ui32>
              %295 = ascvec.add %293, %294, %210 : <ui32>
              %296 = ascvec.shr %295, 16, %210 : <ui32> -> <ui32>
              %297 = ascvec.min_scalar %296, %c126_i32, %210 : <ui32>, i32
              %298 = ascvec.or %297, %284, %210 : <ui32>, <ui32> -> <ui32>
              %299 = ascvec.cmp_gt_scalar %285, %c2139095040_i32, %210 : <ui32>, i32 -> !ascvec.mask
              %300 = ascvec.select %268, %298, %299 : <ui32>
              ascvec.storealign_pack %arg14[%276], %300, %210 {mode = "b32_to_b8"} : memref<16384xui8, 1>, index, <ui32>
              %301 = arith.addi %236, %c2_i32 : i32
              %302 = arith.extsi %301 : i32 to i64
              %303 = arith.muli %302, %arg9 : i64
              %304 = arith.addi %303, %arg10 : i64
              %305 = ascvec.and %230, %262, %210 : <ui32>, <ui32> -> <ui32>
              %306 = arith.index_cast %304 : i64 to index
              %307 = ascvec.loadalign_unpack %arg11[%306] {mode = "b16_to_b32"} : memref<16384xbf16, 1>, index -> <bf16>
              %308 = ascvec.cast %307, %210 {rounding = 6 : i32} : <bf16> -> <f32>
              %309 = ascvec.mul %308, %222, %210 : <f32>
              %310 = ascvec.cast %309, %210 : <f32> -> <bf16>
              %311 = ascvec.cast %310, %210 {rounding = 6 : i32} : <bf16> -> <f32>
              %312 = ascvec.reinterpret %311 : <f32> -> <ui32>
              %313 = ascvec.shr %312, 24, %210 : <ui32> -> <ui32>
              %314 = ascvec.and %313, %249, %210 : <ui32>, <ui32> -> <ui32>
              %315 = ascvec.and %312, %251, %210 : <ui32>, <ui32> -> <ui32>
              %316 = ascvec.shr %315, 4, %210 : <ui32> -> <ui32>
              %317 = ascvec.sub %316, %254, %210 : <ui32>
              %318 = ascvec.reinterpret %315 : <ui32> -> <f32>
              %319 = ascvec.mul_scalar %318, %cst, %210 : <f32>, f32
              %320 = ascvec.cast %319, %210 {rounding = 4 : i32} : <f32> -> <i32>
              %321 = ascvec.reinterpret %320 : <i32> -> <ui32>
              %322 = ascvec.cmp_ge_scalar %315, %c1015021568_i32, %210 : <ui32>, i32 -> !ascvec.mask
              %323 = ascvec.select %317, %321, %322 : <ui32>
              %324 = ascvec.and %305, %262, %210 : <ui32>, <ui32> -> <ui32>
              %325 = ascvec.add %323, %324, %210 : <ui32>
              %326 = ascvec.shr %325, 16, %210 : <ui32> -> <ui32>
              %327 = ascvec.min_scalar %326, %c126_i32, %210 : <ui32>, i32
              %328 = ascvec.or %327, %314, %210 : <ui32>, <ui32> -> <ui32>
              %329 = ascvec.cmp_gt_scalar %315, %c2139095040_i32, %210 : <ui32>, i32 -> !ascvec.mask
              %330 = ascvec.select %268, %328, %329 : <ui32>
              ascvec.storealign_pack %arg14[%306], %330, %210 {mode = "b32_to_b8"} : memref<16384xui8, 1>, index, <ui32>
              %331 = arith.addi %236, %c3_i32 : i32
              %332 = arith.extsi %331 : i32 to i64
              %333 = arith.muli %332, %arg9 : i64
              %334 = arith.addi %333, %arg10 : i64
              %335 = ascvec.shr %233, 16, %210 : <ui32> -> <ui32>
              %336 = arith.index_cast %334 : i64 to index
              %337 = ascvec.loadalign_unpack %arg11[%336] {mode = "b16_to_b32"} : memref<16384xbf16, 1>, index -> <bf16>
              %338 = ascvec.cast %337, %210 {rounding = 6 : i32} : <bf16> -> <f32>
              %339 = ascvec.mul %338, %222, %210 : <f32>
              %340 = ascvec.cast %339, %210 : <f32> -> <bf16>
              %341 = ascvec.cast %340, %210 {rounding = 6 : i32} : <bf16> -> <f32>
              %342 = ascvec.reinterpret %341 : <f32> -> <ui32>
              %343 = ascvec.shr %342, 24, %210 : <ui32> -> <ui32>
              %344 = ascvec.and %343, %249, %210 : <ui32>, <ui32> -> <ui32>
              %345 = ascvec.and %342, %251, %210 : <ui32>, <ui32> -> <ui32>
              %346 = ascvec.shr %345, 4, %210 : <ui32> -> <ui32>
              %347 = ascvec.sub %346, %254, %210 : <ui32>
              %348 = ascvec.reinterpret %345 : <ui32> -> <f32>
              %349 = ascvec.mul_scalar %348, %cst, %210 : <f32>, f32
              %350 = ascvec.cast %349, %210 {rounding = 4 : i32} : <f32> -> <i32>
              %351 = ascvec.reinterpret %350 : <i32> -> <ui32>
              %352 = ascvec.cmp_ge_scalar %345, %c1015021568_i32, %210 : <ui32>, i32 -> !ascvec.mask
              %353 = ascvec.select %347, %351, %352 : <ui32>
              %354 = ascvec.and %335, %262, %210 : <ui32>, <ui32> -> <ui32>
              %355 = ascvec.add %353, %354, %210 : <ui32>
              %356 = ascvec.shr %355, 16, %210 : <ui32> -> <ui32>
              %357 = ascvec.min_scalar %356, %c126_i32, %210 : <ui32>, i32
              %358 = ascvec.or %357, %344, %210 : <ui32>, <ui32> -> <ui32>
              %359 = ascvec.cmp_gt_scalar %345, %c2139095040_i32, %210 : <ui32>, i32 -> !ascvec.mask
              %360 = ascvec.select %268, %358, %359 : <ui32>
              ascvec.storealign_pack %arg14[%336], %360, %210 {mode = "b32_to_b8"} : memref<16384xui8, 1>, index, <ui32>
            }
          }
          ascvec.mem_bar vst_vld
          %mask, %remaining_out = ascvec.update_mask %c64_i32_11 elem_bits 16 : !ascvec.mask
          scf.for %arg18 = %c0_i32_13 to %arg17 step %c1_i32  : i32 {
            %211 = arith.muli %arg18, %c128_i32 : i32
            %212 = arith.index_cast %211 : i32 to index
            %213 = ascvec.loadalign %arg12[%212] : memref<512xui32, 1>, index -> <ui32>
            %214 = arith.addi %211, %c64_i32_11 : i32
            %215 = arith.index_cast %214 : i32 to index
            %216 = ascvec.loadalign %arg12[%215] : memref<512xui32, 1>, index -> <ui32>
            %217 = ascvec.shl %216, 8, %210 : <ui32> -> <ui32>
            %218 = ascvec.or %213, %217, %210 : <ui32>, <ui32> -> <ui32>
            %219 = ascvec.pack %218 src_elem_bits 32 dst_elem_bits 16 {part = "low"} : <ui32> -> <ui16>
            %220 = arith.extsi %arg18 : i32 to i64
            %221 = arith.muli %220, %arg9 : i64
            %222 = arith.addi %221, %arg10 : i64
            %223 = arith.index_cast %222 : i64 to index
            ascvec.storealign %arg16[%223], %219, %mask : memref<?xui16, 1>, index, <ui16>
          }
          ascvec.mem_bar vst_vld
        } {cannir.vf_mode = "simd"}
        ascvec.release_buf v, %87, 0 {cannir.buffer.generated_sync}
        ascvec.release_buf v, %82, 0 {cannir.buffer.generated_sync}
        ascvec.release_buf v, %46, 0 {cannir.buffer.generated_sync}
      }
      %95 = cannir.extract_layout %arg1 : !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
      %96 = cannir.extract_shape %95 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, -1)>
      %97 = cannir.extract_dim %96[0] : !cannir.shape<(-1, -1)>
      %98 = arith.minsi %37, %97 : i64
      %99 = arith.subi %97, %98 : i64
      %100 = arith.minsi %99, %c1_i64 : i64
      %101 = arith.cmpi sle, %37, %97 : i64
      %102 = arith.select %101, %100, %c0_i64 : i64
      %103 = arith.cmpi sgt, %102, %c1_i64 : i64
      %104 = arith.select %103, %c1_i64, %102 : i64
      %105 = cannir.extract_dim %96[1] : !cannir.shape<(-1, -1)>
      %106 = arith.divsi %105, %59 : i64
      %107 = arith.minsi %39, %106 : i64
      %108 = arith.muli %107, %40 : i64
      %109 = arith.subi %105, %108 : i64
      %110 = arith.minsi %40, %109 : i64
      %111 = arith.cmpi sle, %39, %106 : i64
      %112 = arith.select %111, %110, %c0_i64 : i64
      %113 = arith.cmpi sgt, %112, %c16384_i64 : i64
      %114 = arith.select %113, %c16384_i64, %112 : i64
      %115 = cannir.extract_physical_layout %arg1 : !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
      %116 = cannir.extract_stride %115 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
      %117 = cannir.extract_dim %116[0] : !cannir.stride<(-1, 1)>
      %118 = arith.muli %37, %117 : i64
      %119 = arith.addi %118, %69 : i64
      %120 = arith.muli %119, %c8_i64 : i64
      %121 = ascscalar.memref_address %alloc_0 : memref<1xui8>
      %122 = arith.divui %120, %c8_i64 : i64
      %123 = arith.addi %121, %122 : i64
      %124 = ascscalar.bind_gm_ptr %123 : memref<1xui8>
      %125 = ascscalar.bind_local_ptr %80 : memref<1xui8, 1>
      %126 = cannir.extract_stride %95 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
      %127 = cannir.extract_dim %126[0] : !cannir.stride<(-1, 1)>
      %128 = arith.cmpi ugt, %104, %c0_i64 : i64
      %129 = arith.cmpi ugt, %114, %c0_i64 : i64
      %130 = arith.andi %128, %129 : i1
      scf.if %130 {
        ascvec.get_buf mte3, %82, 0 {cannir.buffer.generated_sync}
        %176 = arith.trunci %104 : i64 to i32
        %177 = arith.trunci %114 : i64 to i32
        ascvec.copy_ub2gm %124[%c0], %125[%c0], %176, %177, %127, %c16384_i32 l2_cache_ctl %c0_i32 : memref<1xui8>, memref<1xui8, 1>
        ascvec.release_buf mte3, %82, 0 {cannir.buffer.generated_sync}
      }
      %131 = arith.remsi %40, %c32_i64 : i64
      %132 = arith.cmpi ne, %131, %c0_i64 : i64
      %133 = arith.cmpi slt, %131, %c0_i64 : i64
      %134 = arith.andi %132, %133 : i1
      %135 = arith.divsi %40, %c32_i64 : i64
      %136 = arith.extui %134 : i1 to i64
      %137 = arith.subi %135, %136 : i64
      %138 = cannir.extract_layout %arg2 : !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
      %139 = cannir.extract_shape %138 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, -1)>
      %140 = cannir.extract_dim %139[0] : !cannir.shape<(-1, -1)>
      %141 = arith.minsi %37, %140 : i64
      %142 = arith.subi %140, %141 : i64
      %143 = arith.minsi %142, %c1_i64 : i64
      %144 = arith.cmpi sle, %37, %140 : i64
      %145 = arith.select %144, %143, %c0_i64 : i64
      %146 = arith.cmpi sgt, %145, %c1_i64 : i64
      %147 = arith.select %146, %c1_i64, %145 : i64
      %148 = cannir.extract_dim %139[1] : !cannir.shape<(-1, -1)>
      %149 = arith.maxsi %137, %c1_i64 : i64
      %150 = arith.divsi %148, %149 : i64
      %151 = arith.minsi %39, %150 : i64
      %152 = arith.muli %151, %137 : i64
      %153 = arith.subi %148, %152 : i64
      %154 = arith.minsi %137, %153 : i64
      %155 = arith.cmpi sle, %39, %150 : i64
      %156 = arith.select %155, %154, %c0_i64 : i64
      %157 = arith.cmpi sgt, %156, %c1024_i64 : i64
      %158 = arith.select %157, %c1024_i64, %156 : i64
      %159 = cannir.extract_physical_layout %arg2 : !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
      %160 = cannir.extract_stride %159 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
      %161 = cannir.extract_dim %160[0] : !cannir.stride<(-1, 1)>
      %162 = arith.muli %37, %161 : i64
      %163 = arith.muli %39, %137 : i64
      %164 = arith.addi %162, %163 : i64
      %165 = arith.muli %164, %c8_i64 : i64
      %166 = ascscalar.memref_address %alloc_1 : memref<1xui8>
      %167 = arith.divui %165, %c8_i64 : i64
      %168 = arith.addi %166, %167 : i64
      %169 = ascscalar.bind_gm_ptr %168 : memref<1xui8>
      %170 = ascscalar.bind_local_ptr %85 : memref<1xui8, 1>
      %171 = cannir.extract_stride %138 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
      %172 = cannir.extract_dim %171[0] : !cannir.stride<(-1, 1)>
      %173 = arith.cmpi ugt, %147, %c0_i64 : i64
      %174 = arith.cmpi ugt, %158, %c0_i64 : i64
      %175 = arith.andi %173, %174 : i1
      scf.if %175 {
        ascvec.get_buf mte3, %87, 0 {cannir.buffer.generated_sync}
        %176 = arith.trunci %147 : i64 to i32
        %177 = arith.trunci %158 : i64 to i32
        ascvec.copy_ub2gm %169[%c0], %170[%c0], %176, %177, %172, %c1024_i32 l2_cache_ctl %c0_i32 : memref<1xui8>, memref<1xui8, 1>
        ascvec.release_buf mte3, %87, 0 {cannir.buffer.generated_sync}
      }
      scf.yield %48 : i64
    }
    return
  }
}
