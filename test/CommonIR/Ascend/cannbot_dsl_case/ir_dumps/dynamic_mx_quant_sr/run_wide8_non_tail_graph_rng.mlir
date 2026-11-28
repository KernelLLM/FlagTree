module {
  func.func @run_wide8_non_tail_graph_rng_11d928bb32c49713e8135dc51d7fff05(%arg0: !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg1: !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg2: !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: !cannir.tensor<<gm, i64>, <<(1)>, <(1)>>, <<(1)>, <(1)>>, identity>, %arg7: !cannir.tensor<<gm, i64>, <<(1)>, <(1)>>, <<(1)>, <(1)>>, identity>, %arg8: i64) attributes {cannbotdsl.contract_digest = "11d928bb32c49713e8135dc51d7fff05f197a8f1fdebf2808f660ad106b6281b", cannbotdsl.public_slot_count = 9 : i64, cannbotdsl.slot_dtype_bits = [16, 8, 8, 64, 64, 64, 64, 64, 64], cannbotdsl.slot_dtype_codes = [4, 1, 1, 0, 0, 0, 0, 0, 0], cannbotdsl.slot_dtype_lanes = [1, 1, 1, 1, 1, 1, 1, 1, 1], cannbotdsl.slot_flags = [1, 1, 1, 1, 1, 1, 1, 1, 1], cannbotdsl.slot_kinds = [1, 1, 1, 2, 2, 2, 1, 1, 2], cannbotdsl.slot_type_digests = ["5d7cdc96dc21b3b739e32aeccf58c1df113bec54701c75107ed650f2544a07da", "171ddd09eb070fb4a6413ae87b2ceecddc233a9a173a0404e96285a4bb8b8659", "891bcd21e721f8d1a666bcca4e8705ec63796c1add8ae000536d0d98f0274d3a", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "a9d876affafdf57be3f0e1a45cb0b293a48851dbda8b74101f5184f5130b1b63", "a9d876affafdf57be3f0e1a45cb0b293a48851dbda8b74101f5184f5130b1b63", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515"], cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.host} {
    cannir.kernel_launch @kernel__DynamicMxQuantSrKernel_run_wide8_non_tail_0(%arg5) [%arg0, %arg1, %arg2, %arg3, %arg4, %arg6, %arg7, %arg8] {cannbotdsl.profile = {indices = array<i64: 0, 1, 2, 5, 6>, name = "_DynamicMxQuantSrKernel_run_wide8_non_tail", op_type = "_DynamicMxQuantSrKernel_run_wide8_non_tail", roles = array<i64: 0, 1, 1, 0, 0>}} : !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, i64, i64, !cannir.tensor<<gm, i64>, <<(1)>, <(1)>>, <<(1)>, <(1)>>, identity>, !cannir.tensor<<gm, i64>, <<(1)>, <(1)>>, <<(1)>, <(1)>>, identity>, i64
    return
  }
  func.func @kernel__DynamicMxQuantSrKernel_run_wide8_non_tail_0(%arg0: !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg1: !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg2: !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg3: i64, %arg4: i64, %arg5: !cannir.tensor<<gm, i64>, <<(1)>, <(1)>>, <<(1)>, <(1)>>, identity>, %arg6: !cannir.tensor<<gm, i64>, <<(1)>, <(1)>>, <<(1)>, <(1)>>, identity>, %arg7: i64) attributes {cannir.buffer.auto_sync, cannir.buffer.sync_analyzed, cannir.buffer.sync_verified, cannir.buffer.tensor_state, cannir.core = "aiv", cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.device, cannir.explicit_ring_buffer_cursors, cannir.kernel_group = "kernel__DynamicMxQuantSrKernel_run_wide8_non_tail_0", cannir.resource.ids_prepared} {
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
    %alloc_2 = memref.alloc() {cannir.arg_index = 5 : i64} : memref<1xi64>
    %alloc_3 = memref.alloc() {cannir.arg_index = 6 : i64} : memref<1xi64>
    %alloc_4 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<16384xbf16, 1>
    %alloc_5 = memref.alloc() {cannir.byte_offset = 32768 : i64} : memref<16384xbf16, 1>
    %alloc_6 = memref.alloc() {cannir.byte_offset = 65536 : i64} : memref<16384xui8, 1>
    %alloc_7 = memref.alloc() {cannir.byte_offset = 81920 : i64} : memref<16384xui8, 1>
    %alloc_8 = memref.alloc() {cannir.byte_offset = 98304 : i64} : memref<1024xui8, 1>
    %alloc_9 = memref.alloc() {cannir.byte_offset = 99328 : i64} : memref<1024xui8, 1>
    %alloc_10 = memref.alloc() {cannir.byte_offset = 100352 : i64} : memref<512xui32, 1>
    %alloc_11 = memref.alloc() {cannir.byte_offset = 102400 : i64} : memref<4096xi32, 1>
    %0 = ascscalar.scalar_ptr %alloc_2[%c0] : memref<1xi64>, index -> <gm, i64>
    %1 = ascscalar.load_dev %0 : !ascscalar.scalar_ptr<gm, i64> -> i64
    %2 = ascscalar.scalar_ptr %alloc_3[%c0] : memref<1xi64>, index -> <gm, i64>
    %3 = ascscalar.load_dev %2 : !ascscalar.scalar_ptr<gm, i64> -> i64
    %4 = arith.addi %3, %arg7 : i64
    %5 = arith.remsi %arg3, %c256_i64 : i64
    %6 = arith.cmpi ne, %5, %c0_i64 : i64
    %7 = arith.cmpi slt, %5, %c0_i64 : i64
    %8 = arith.andi %6, %7 : i1
    %9 = arith.divsi %arg3, %c256_i64 : i64
    %10 = arith.extui %8 : i1 to i64
    %11 = arith.subi %9, %10 : i64
    %12 = cannir.extract_layout %arg0 : !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
    %13 = cannir.extract_shape %12 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, -1)>
    %14 = cannir.extract_dim %13[0] : !cannir.shape<(-1, -1)>
    %15 = arith.muli %14, %11 : i64
    %16 = ascscalar.get_block_idx
    %17 = ascscalar.get_block_num
    %18 = scf.for %arg8 = %16 to %15 step %17 iter_args(%arg9 = %c0_i64) -> (i64)  : i64 {
      %19 = arith.remsi %arg8, %11 : i64
      %20 = arith.cmpi ne, %19, %c0_i64 : i64
      %21 = arith.cmpi slt, %19, %c0_i64 : i64
      %22 = arith.cmpi slt, %11, %c0_i64 : i64
      %23 = arith.xori %21, %22 : i1
      %24 = arith.andi %20, %23 : i1
      %25 = arith.divsi %arg8, %11 : i64
      %26 = arith.extui %24 : i1 to i64
      %27 = arith.subi %25, %26 : i64
      %28 = arith.select %24, %11, %c0_i64 : i64
      %29 = arith.addi %19, %28 : i64
      %30 = arith.muli %arg4, %c256_i64 : i64
      %31 = arith.cmpi eq, %arg9, %c0_i64 : i64
      %32 = ascscalar.memref_address %alloc_4 : memref<16384xbf16, 1>
      %33 = ascscalar.memref_address %alloc_5 : memref<16384xbf16, 1>
      %34 = arith.select %31, %32, %33 : i64
      %35 = arith.cmpi ne, %arg9, %c0_i64 : i64
      %36 = arith.extui %35 : i1 to i64
      %37 = arith.addi %arg9, %c1_i64 overflow<nsw> : i64
      %38 = arith.remui %37, %c2_i64 : i64
      %39 = arith.minsi %27, %14 : i64
      %40 = arith.subi %14, %39 : i64
      %41 = arith.minsi %40, %c1_i64 : i64
      %42 = arith.cmpi sle, %27, %14 : i64
      %43 = arith.select %42, %41, %c0_i64 : i64
      %44 = cannir.extract_physical_layout %arg0 : !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
      %45 = cannir.extract_stride %44 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
      %46 = cannir.extract_dim %45[0] : !cannir.stride<(-1, 1)>
      %47 = arith.muli %46, %c2_i64 : i64
      %48 = cannir.extract_dim %13[1] : !cannir.shape<(-1, -1)>
      %49 = arith.maxsi %30, %c1_i64 : i64
      %50 = arith.divsi %48, %49 : i64
      %51 = arith.minsi %29, %50 : i64
      %52 = arith.muli %51, %30 : i64
      %53 = arith.subi %48, %52 : i64
      %54 = arith.minsi %30, %53 : i64
      %55 = arith.cmpi sle, %29, %50 : i64
      %56 = arith.select %55, %54, %c0_i64 : i64
      %57 = arith.muli %27, %46 : i64
      %58 = arith.muli %57, %c2_i64 : i64
      %59 = arith.muli %29, %30 : i64
      %60 = arith.muli %59, %c2_i64 : i64
      %61 = arith.addi %58, %60 : i64
      %62 = arith.muli %56, %c2_i64 : i64
      %63 = arith.cmpi eq, %62, %c0_i64 : i64
      %64 = arith.select %63, %c0_i64, %43 : i64
      %65 = arith.select %63, %c2_i64, %62 : i64
      %66 = ascscalar.bind_local_ptr %34 : memref<1xbf16, 1>
      ascvec.get_buf mte2, %36, 0 {cannir.buffer.generated_sync}
      "ascvec.copy_gm2ub_plan"(%66, %alloc, %c0_i32, %c0_i32, %c0_i64, %61, %64, %65, %47, %c32768_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
      ascvec.release_buf mte2, %36, 0 {cannir.buffer.generated_sync}
      %67 = ascscalar.bind_local_ptr %34 : memref<16384xbf16, 1>
      %68 = ascscalar.memref_address %alloc_6 : memref<16384xui8, 1>
      %69 = ascscalar.memref_address %alloc_7 : memref<16384xui8, 1>
      %70 = arith.select %31, %68, %69 : i64
      %71 = ascscalar.bind_local_ptr %70 : memref<16384xui8, 1>
      %72 = arith.select %31, %c2_i64, %c3_i64 : i64
      %73 = ascscalar.memref_address %alloc_8 : memref<1024xui8, 1>
      %74 = ascscalar.memref_address %alloc_9 : memref<1024xui8, 1>
      %75 = arith.select %31, %73, %74 : i64
      %76 = ascscalar.bind_local_ptr %75 : memref<1024xui8, 1>
      %77 = arith.select %31, %c4_i64, %c5_i64 : i64
      %78 = arith.muli %27, %arg4 : i64
      %79 = arith.muli %29, %c256_i64 : i64
      %80 = arith.remsi %79, %c32_i64 : i64
      %81 = arith.cmpi ne, %80, %c0_i64 : i64
      %82 = arith.cmpi slt, %80, %c0_i64 : i64
      %83 = arith.andi %81, %82 : i1
      %84 = arith.divsi %79, %c32_i64 : i64
      %85 = arith.extui %83 : i1 to i64
      %86 = arith.subi %84, %85 : i64
      %87 = arith.remsi %arg3, %c32_i64 : i64
      %88 = arith.cmpi ne, %87, %c0_i64 : i64
      %89 = arith.cmpi slt, %87, %c0_i64 : i64
      %90 = arith.andi %88, %89 : i1
      %91 = arith.divsi %arg3, %c32_i64 : i64
      %92 = arith.extui %90 : i1 to i64
      %93 = arith.subi %91, %92 : i64
      %94 = arith.trunci %1 : i64 to i32
      %95 = arith.extui %94 : i32 to i64
      %96 = arith.subi %1, %95 : i64
      %97 = arith.remsi %96, %c4294967296_i64 : i64
      %98 = arith.cmpi ne, %97, %c0_i64 : i64
      %99 = arith.cmpi slt, %97, %c0_i64 : i64
      %100 = arith.andi %98, %99 : i1
      %101 = arith.divsi %96, %c4294967296_i64 : i64
      %102 = arith.extui %100 : i1 to i64
      %103 = arith.subi %101, %102 : i64
      %104 = arith.trunci %103 : i64 to i32
      ascvec.vf(%86, %78, %93, %4, %94, %104, %alloc_11 : i64, i64, i64, i64, i32, i32, memref<4096xi32, 1>) {
      ^bb0(%arg10: i64, %arg11: i64, %arg12: i64, %arg13: i64, %arg14: i32, %arg15: i32, %arg16: memref<4096xi32, 1>):
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
        %c0_i64_12 = arith.constant 0 : i64
        %c4294967296_i64_13 = arith.constant 4294967296 : i64
        %c0_i32_14 = arith.constant 0 : i32
        %c16_i32 = arith.constant 16 : i32
        %mask, %remaining_out = ascvec.update_mask %c16_i32 : !ascvec.mask
        %186 = ascvec.arange %c0_i32_14 : i32 -> <ui32>
        scf.for %arg17 = %c0_i32_14 to %c8_i32 step %c1_i32  : i32 {
          %187 = arith.extsi %arg17 : i32 to i64
          %188 = arith.addi %arg10, %187 : i64
          %189 = arith.trunci %arg11 : i64 to i32
          %190 = arith.extui %189 : i32 to i64
          %191 = arith.subi %arg11, %190 : i64
          %192 = arith.remsi %191, %c4294967296_i64_13 : i64
          %193 = arith.cmpi ne, %192, %c0_i64_12 : i64
          %194 = arith.cmpi slt, %192, %c0_i64_12 : i64
          %195 = arith.andi %193, %194 : i1
          %196 = arith.divsi %191, %c4294967296_i64_13 : i64
          %197 = arith.extui %195 : i1 to i64
          %198 = arith.subi %196, %197 : i64
          %199 = arith.trunci %198 : i64 to i32
          %200 = arith.trunci %188 : i64 to i32
          %201 = arith.extui %200 : i32 to i64
          %202 = arith.subi %188, %201 : i64
          %203 = arith.remsi %202, %c4294967296_i64_13 : i64
          %204 = arith.cmpi ne, %203, %c0_i64_12 : i64
          %205 = arith.cmpi slt, %203, %c0_i64_12 : i64
          %206 = arith.andi %204, %205 : i1
          %207 = arith.divsi %202, %c4294967296_i64_13 : i64
          %208 = arith.extui %206 : i1 to i64
          %209 = arith.subi %207, %208 : i64
          %210 = arith.trunci %209 : i64 to i32
          %211 = ascvec.duplicate_scalar %189, %mask : i32 -> <ui32>
          %212 = ascvec.duplicate_scalar %199, %mask : i32 -> <ui32>
          %213 = ascvec.duplicate_scalar %c0_i32_14, %mask : i32 -> <ui32>
          %carry, %dst = ascvec.addc %211, %186, %mask : <ui32>, <ui32> -> !ascvec.mask, <ui32>
          %carry_15, %dst_16 = ascvec.addc %212, %213, %carry, %mask : <ui32>, <ui32> -> !ascvec.mask, <ui32>
          %214 = ascvec.duplicate_scalar %200, %mask : i32 -> <ui32>
          %215 = ascvec.duplicate_scalar %210, %mask : i32 -> <ui32>
          %216 = arith.trunci %arg12 : i64 to i32
          %217 = arith.extui %216 : i32 to i64
          %218 = arith.subi %arg12, %217 : i64
          %219 = arith.remsi %218, %c4294967296_i64_13 : i64
          %220 = arith.cmpi ne, %219, %c0_i64_12 : i64
          %221 = arith.cmpi slt, %219, %c0_i64_12 : i64
          %222 = arith.andi %220, %221 : i1
          %223 = arith.divsi %218, %c4294967296_i64_13 : i64
          %224 = arith.extui %222 : i1 to i64
          %225 = arith.subi %223, %224 : i64
          %226 = arith.trunci %225 : i64 to i32
          %227 = ascvec.duplicate_scalar %216, %mask : i32 -> <ui32>
          %dst0, %dst1 = ascvec.mull %dst, %227, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %228 = ascvec.mul_scalar %dst, %226, %mask : <ui32>, i32
          %229 = ascvec.add %dst1, %228, %mask : <ui32>
          %230 = ascvec.mul_scalar %dst_16, %216, %mask : <ui32>, i32
          %231 = ascvec.add %229, %230, %mask : <ui32>
          %carry_17, %dst_18 = ascvec.addc %dst0, %214, %mask : <ui32>, <ui32> -> !ascvec.mask, <ui32>
          %carry_19, %dst_20 = ascvec.addc %231, %215, %carry_17, %mask : <ui32>, <ui32> -> !ascvec.mask, <ui32>
          %232 = arith.trunci %arg13 : i64 to i32
          %233 = arith.extui %232 : i32 to i64
          %234 = arith.subi %arg13, %233 : i64
          %235 = arith.remsi %234, %c4294967296_i64_13 : i64
          %236 = arith.cmpi ne, %235, %c0_i64_12 : i64
          %237 = arith.cmpi slt, %235, %c0_i64_12 : i64
          %238 = arith.andi %236, %237 : i1
          %239 = arith.divsi %234, %c4294967296_i64_13 : i64
          %240 = arith.extui %238 : i1 to i64
          %241 = arith.subi %239, %240 : i64
          %242 = arith.trunci %241 : i64 to i32
          %243 = arith.divui %232, %c4_i32 : i32
          %244 = arith.remui %242, %c4_i32 : i32
          %245 = arith.muli %244, %c1073741824_i32 : i32
          %246 = arith.addi %243, %245 : i32
          %247 = arith.divui %242, %c4_i32 : i32
          %248 = ascvec.shr %dst_18, 14, %mask : <ui32> -> <ui32>
          %249 = ascvec.shl %dst_20, 18, %mask : <ui32> -> <ui32>
          %250 = ascvec.or %248, %249, %mask : <ui32>, <ui32> -> <ui32>
          %251 = ascvec.shr %dst_20, 14, %mask : <ui32> -> <ui32>
          %252 = ascvec.duplicate_scalar %246, %mask : i32 -> <ui32>
          %253 = ascvec.duplicate_scalar %247, %mask : i32 -> <ui32>
          %carry_21, %dst_22 = ascvec.addc %252, %250, %mask : <ui32>, <ui32> -> !ascvec.mask, <ui32>
          %carry_23, %dst_24 = ascvec.addc %253, %251, %carry_21, %mask : <ui32>, <ui32> -> !ascvec.mask, <ui32>
          %254 = ascvec.duplicate_scalar %c16383_i32, %mask : i32 -> <ui32>
          %255 = ascvec.and %dst_18, %254, %mask : <ui32>, <ui32> -> <ui32>
          %256 = ascvec.duplicate_scalar %c-766435501_i32, %mask : i32 -> <ui32>
          %257 = ascvec.duplicate_scalar %c-845247145_i32, %mask : i32 -> <ui32>
          %dst0_25, %dst1_26 = ascvec.mull %213, %256, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %dst0_27, %dst1_28 = ascvec.mull %dst_22, %257, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %258 = ascvec.xor %dst1_28, %255, %mask : <ui32>, <ui32> -> <ui32>
          %259 = ascvec.xor %dst1_26, %dst_24, %mask : <ui32>, <ui32> -> <ui32>
          %260 = ascvec.duplicate_scalar %arg14, %mask : i32 -> <ui32>
          %261 = ascvec.duplicate_scalar %arg15, %mask : i32 -> <ui32>
          %262 = ascvec.xor %258, %260, %mask : <ui32>, <ui32> -> <ui32>
          %263 = ascvec.xor %259, %261, %mask : <ui32>, <ui32> -> <ui32>
          %dst0_29, %dst1_30 = ascvec.mull %262, %256, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %dst0_31, %dst1_32 = ascvec.mull %263, %257, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %264 = ascvec.xor %dst1_32, %dst0_27, %mask : <ui32>, <ui32> -> <ui32>
          %265 = ascvec.xor %dst1_30, %dst0_25, %mask : <ui32>, <ui32> -> <ui32>
          %266 = arith.addi %arg14, %c-1640531527_i32 : i32
          %267 = ascvec.duplicate_scalar %266, %mask : i32 -> <ui32>
          %268 = arith.addi %arg15, %c-1150833019_i32 : i32
          %269 = ascvec.duplicate_scalar %268, %mask : i32 -> <ui32>
          %270 = ascvec.xor %264, %267, %mask : <ui32>, <ui32> -> <ui32>
          %271 = ascvec.xor %265, %269, %mask : <ui32>, <ui32> -> <ui32>
          %dst0_33, %dst1_34 = ascvec.mull %270, %256, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %dst0_35, %dst1_36 = ascvec.mull %271, %257, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %272 = ascvec.xor %dst1_36, %dst0_31, %mask : <ui32>, <ui32> -> <ui32>
          %273 = ascvec.xor %dst1_34, %dst0_29, %mask : <ui32>, <ui32> -> <ui32>
          %274 = arith.addi %arg14, %c1013904242_i32 : i32
          %275 = ascvec.duplicate_scalar %274, %mask : i32 -> <ui32>
          %276 = arith.addi %arg15, %c1993301258_i32 : i32
          %277 = ascvec.duplicate_scalar %276, %mask : i32 -> <ui32>
          %278 = ascvec.xor %272, %275, %mask : <ui32>, <ui32> -> <ui32>
          %279 = ascvec.xor %273, %277, %mask : <ui32>, <ui32> -> <ui32>
          %dst0_37, %dst1_38 = ascvec.mull %278, %256, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %dst0_39, %dst1_40 = ascvec.mull %279, %257, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %280 = ascvec.xor %dst1_40, %dst0_35, %mask : <ui32>, <ui32> -> <ui32>
          %281 = ascvec.xor %dst1_38, %dst0_33, %mask : <ui32>, <ui32> -> <ui32>
          %282 = arith.addi %arg14, %c-626627285_i32 : i32
          %283 = ascvec.duplicate_scalar %282, %mask : i32 -> <ui32>
          %284 = arith.addi %arg15, %c842468239_i32 : i32
          %285 = ascvec.duplicate_scalar %284, %mask : i32 -> <ui32>
          %286 = ascvec.xor %280, %283, %mask : <ui32>, <ui32> -> <ui32>
          %287 = ascvec.xor %281, %285, %mask : <ui32>, <ui32> -> <ui32>
          %dst0_41, %dst1_42 = ascvec.mull %286, %256, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %dst0_43, %dst1_44 = ascvec.mull %287, %257, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %288 = ascvec.xor %dst1_44, %dst0_39, %mask : <ui32>, <ui32> -> <ui32>
          %289 = ascvec.xor %dst1_42, %dst0_37, %mask : <ui32>, <ui32> -> <ui32>
          %290 = arith.addi %arg14, %c2027808484_i32 : i32
          %291 = ascvec.duplicate_scalar %290, %mask : i32 -> <ui32>
          %292 = arith.addi %arg15, %c-308364780_i32 : i32
          %293 = ascvec.duplicate_scalar %292, %mask : i32 -> <ui32>
          %294 = ascvec.xor %288, %291, %mask : <ui32>, <ui32> -> <ui32>
          %295 = ascvec.xor %289, %293, %mask : <ui32>, <ui32> -> <ui32>
          %dst0_45, %dst1_46 = ascvec.mull %294, %256, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %dst0_47, %dst1_48 = ascvec.mull %295, %257, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %296 = ascvec.xor %dst1_48, %dst0_43, %mask : <ui32>, <ui32> -> <ui32>
          %297 = ascvec.xor %dst1_46, %dst0_41, %mask : <ui32>, <ui32> -> <ui32>
          %298 = arith.addi %arg14, %c387276957_i32 : i32
          %299 = ascvec.duplicate_scalar %298, %mask : i32 -> <ui32>
          %300 = arith.addi %arg15, %c-1459197799_i32 : i32
          %301 = ascvec.duplicate_scalar %300, %mask : i32 -> <ui32>
          %302 = ascvec.xor %296, %299, %mask : <ui32>, <ui32> -> <ui32>
          %303 = ascvec.xor %297, %301, %mask : <ui32>, <ui32> -> <ui32>
          %dst0_49, %dst1_50 = ascvec.mull %302, %256, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %dst0_51, %dst1_52 = ascvec.mull %303, %257, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %304 = ascvec.xor %dst1_52, %dst0_47, %mask : <ui32>, <ui32> -> <ui32>
          %305 = ascvec.xor %dst1_50, %dst0_45, %mask : <ui32>, <ui32> -> <ui32>
          %306 = arith.addi %arg14, %c-1253254570_i32 : i32
          %307 = ascvec.duplicate_scalar %306, %mask : i32 -> <ui32>
          %308 = arith.addi %arg15, %c1684936478_i32 : i32
          %309 = ascvec.duplicate_scalar %308, %mask : i32 -> <ui32>
          %310 = ascvec.xor %304, %307, %mask : <ui32>, <ui32> -> <ui32>
          %311 = ascvec.xor %305, %309, %mask : <ui32>, <ui32> -> <ui32>
          %dst0_53, %dst1_54 = ascvec.mull %310, %256, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %dst0_55, %dst1_56 = ascvec.mull %311, %257, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %312 = ascvec.xor %dst1_56, %dst0_51, %mask : <ui32>, <ui32> -> <ui32>
          %313 = ascvec.xor %dst1_54, %dst0_49, %mask : <ui32>, <ui32> -> <ui32>
          %314 = arith.addi %arg14, %c1401181199_i32 : i32
          %315 = ascvec.duplicate_scalar %314, %mask : i32 -> <ui32>
          %316 = arith.addi %arg15, %c534103459_i32 : i32
          %317 = ascvec.duplicate_scalar %316, %mask : i32 -> <ui32>
          %318 = ascvec.xor %312, %315, %mask : <ui32>, <ui32> -> <ui32>
          %319 = ascvec.xor %313, %317, %mask : <ui32>, <ui32> -> <ui32>
          %dst0_57, %dst1_58 = ascvec.mull %318, %256, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %dst0_59, %dst1_60 = ascvec.mull %319, %257, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %320 = ascvec.xor %dst1_60, %dst0_55, %mask : <ui32>, <ui32> -> <ui32>
          %321 = ascvec.xor %dst1_58, %dst0_53, %mask : <ui32>, <ui32> -> <ui32>
          %322 = arith.addi %arg14, %c-239350328_i32 : i32
          %323 = ascvec.duplicate_scalar %322, %mask : i32 -> <ui32>
          %324 = arith.addi %arg15, %c-616729560_i32 : i32
          %325 = ascvec.duplicate_scalar %324, %mask : i32 -> <ui32>
          %326 = ascvec.xor %320, %323, %mask : <ui32>, <ui32> -> <ui32>
          %327 = ascvec.xor %321, %325, %mask : <ui32>, <ui32> -> <ui32>
          %dst0_61, %dst1_62 = ascvec.mull %326, %256, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %dst0_63, %dst1_64 = ascvec.mull %327, %257, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %328 = ascvec.xor %dst1_64, %dst0_59, %mask : <ui32>, <ui32> -> <ui32>
          %329 = ascvec.xor %dst1_62, %dst0_57, %mask : <ui32>, <ui32> -> <ui32>
          %330 = arith.addi %arg14, %c-1879881855_i32 : i32
          %331 = ascvec.duplicate_scalar %330, %mask : i32 -> <ui32>
          %332 = arith.addi %arg15, %c-1767562579_i32 : i32
          %333 = ascvec.duplicate_scalar %332, %mask : i32 -> <ui32>
          %334 = ascvec.xor %328, %331, %mask : <ui32>, <ui32> -> <ui32>
          %335 = ascvec.xor %329, %333, %mask : <ui32>, <ui32> -> <ui32>
          %336 = arith.muli %arg17, %c64_i32 overflow<nsw> : i32
          %337 = ascvec.reinterpret %334 : <ui32> -> <i32>
          %338 = arith.index_cast %336 : i32 to index
          ascvec.storealign %arg16[%338], %337, %mask : memref<4096xi32, 1>, index, <i32>
          %339 = arith.addi %336, %c16_i32 overflow<nsw> : i32
          %340 = ascvec.reinterpret %dst0_63 : <ui32> -> <i32>
          %341 = arith.index_cast %339 : i32 to index
          ascvec.storealign %arg16[%341], %340, %mask : memref<4096xi32, 1>, index, <i32>
          %342 = arith.addi %336, %c32_i32 overflow<nsw> : i32
          %343 = ascvec.reinterpret %335 : <ui32> -> <i32>
          %344 = arith.index_cast %342 : i32 to index
          ascvec.storealign %arg16[%344], %343, %mask : memref<4096xi32, 1>, index, <i32>
          %345 = arith.addi %336, %c48_i32 overflow<nsw> : i32
          %346 = ascvec.reinterpret %dst0_61 : <ui32> -> <i32>
          %347 = arith.index_cast %345 : i32 to index
          ascvec.storealign %arg16[%347], %346, %mask : memref<4096xi32, 1>, index, <i32>
        }
        ascvec.mem_bar vst_vld
      } {cannir.vf_mode = "simd"}
      ascvec.get_buf v, %72, 0 {cannir.buffer.generated_sync}
      ascvec.get_buf v, %36, 0 {cannir.buffer.generated_sync}
      ascvec.get_buf v, %77, 0 {cannir.buffer.generated_sync}
      ascvec.vf(%71, %arg4, %67, %alloc_10, %alloc_11, %76 : memref<16384xui8, 1>, i64, memref<16384xbf16, 1>, memref<512xui32, 1>, memref<4096xi32, 1>, memref<1024xui8, 1>) {
      ^bb0(%arg10: memref<16384xui8, 1>, %arg11: i64, %arg12: memref<16384xbf16, 1>, %arg13: memref<512xui32, 1>, %arg14: memref<4096xi32, 1>, %arg15: memref<1024xui8, 1>):
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
        %c0_i32_12 = arith.constant 0 : i32
        %c0_13 = arith.constant 0 : index
        %c16_i32 = arith.constant 16 : i32
        %mask, %remaining_out = ascvec.update_mask %c16_i32 : !ascvec.mask
        ascvec.store_unalign_begin %arg10 : memref<16384xui8, 1>
        scf.for %arg16 = %c0_i32_12 to %c8_i32 step %c1_i32  : i32 {
          %186 = ascvec.create_mask : !ascvec.mask
          %187 = ascvec.arange %c0_i32_12 : i32 -> <ui32>
          %188 = ascvec.duplicate_scalar %c0_i32_12, %186 : i32 -> <ui32>
          %189 = scf.for %arg17 = %c0_i32_12 to %c8_i32 step %c1_i32 iter_args(%arg18 = %188) -> (!ascvec.vreg<ui32>)  : i32 {
            %210 = arith.muli %arg16, %c32_i32 overflow<nsw> : i32
            %211 = arith.muli %arg17, %c4_i32 overflow<nsw> : i32
            %212 = arith.addi %210, %211 overflow<nsw> : i32
            %213 = arith.extsi %212 : i32 to i64
            %214 = arith.muli %213, %arg11 : i64
            %215 = arith.index_cast %214 : i64 to index
            %216 = ascvec.loadalign_unpack %arg12[%215] {mode = "b16_to_b32"} : memref<16384xbf16, 1>, index -> <bf16>
            %217 = ascvec.cast %216, %186 {rounding = 6 : i32} : <bf16> -> <f32>
            %218 = ascvec.reinterpret %217 : <f32> -> <ui32>
            %219 = ascvec.duplicate_scalar %c2147483647_i32, %186 : i32 -> <ui32>
            %220 = ascvec.and %218, %219, %186 : <ui32>, <ui32> -> <ui32>
            %221 = ascvec.max %arg18, %220, %186 : <ui32>
            scf.yield %221 : !ascvec.vreg<ui32>
          }
          %190 = ascvec.duplicate_scalar %c16_i32, %186 : i32 -> <ui32>
          %191 = ascvec.xor %187, %190, %186 : <ui32>, <ui32> -> <ui32>
          %192 = ascvec.gather_reg %189, %191 : <ui32>, <ui32> -> <ui32>
          %193 = ascvec.max %189, %192, %186 : <ui32>
          %194 = ascvec.duplicate_scalar %c32_i32, %186 : i32 -> <ui32>
          %195 = ascvec.xor %187, %194, %186 : <ui32>, <ui32> -> <ui32>
          %196 = ascvec.gather_reg %193, %195 : <ui32>, <ui32> -> <ui32>
          %197 = ascvec.max %193, %196, %186 : <ui32>
          %198 = ascvec.cmp_ge_scalar %197, %c2139095040_i32, %mask : <ui32>, i32 -> !ascvec.mask
          %199 = ascvec.add_scalar %197, %c-123731969_i32, %mask : <ui32>, i32
          %200 = ascvec.cmp_gt_scalar %197, %c132120577_i32, %mask : <ui32>, i32 -> !ascvec.mask
          %201 = ascvec.shr %199, 23, %200 : <ui32> -> <ui32>
          %202 = ascvec.duplicate_scalar %c255_i32, %198 merge %201 : !ascvec.vreg<ui32> {mode = "merging"} : i32 -> <ui32>
          %203 = ascvec.duplicate_scalar %c254_i32, %mask : i32 -> <ui32>
          %204 = ascvec.sub %203, %202, %mask : <ui32>
          %205 = ascvec.shl %204, 23, %mask : <ui32> -> <ui32>
          %206 = ascvec.duplicate_scalar %c2139160576_i32, %198 merge %205 : !ascvec.vreg<ui32> {mode = "merging"} : i32 -> <ui32>
          %207 = ascvec.reinterpret %206 : <ui32> -> <f32>
          %208 = arith.muli %arg16, %c16_i32 overflow<nsw> : i32
          %209 = arith.index_cast %208 : i32 to index
          ascvec.storealign %arg13[%209], %202, %mask : memref<512xui32, 1>, index, <ui32>
          scf.for %arg17 = %c0_i32_12 to %c4_i32 step %c1_i32  : i32 {
            %210 = arith.muli %arg16, %c64_i32 overflow<nsw> : i32
            %211 = arith.muli %arg17, %c16_i32 overflow<nsw> : i32
            %212 = arith.addi %210, %211 overflow<nsw> : i32
            %213 = arith.index_cast %212 : i32 to index
            %214 = ascvec.loadalign %arg14[%213] : memref<4096xi32, 1>, index -> <i32>
            %215 = ascvec.reinterpret %214 : <i32> -> <ui32>
            %216 = arith.muli %arg16, %c32_i32 overflow<nsw> : i32
            %217 = arith.muli %arg17, %c8_i32 overflow<nsw> : i32
            %218 = arith.addi %216, %217 overflow<nsw> : i32
            %219 = ascvec.shr %215, 21, %mask : <ui32> -> <ui32>
            %220 = ascvec.shl %215, 11, %mask : <ui32> -> <ui32>
            %221 = ascvec.or %219, %220, %mask : <ui32>, <ui32> -> <ui32>
            %222 = ascvec.shr %215, 29, %mask : <ui32> -> <ui32>
            %223 = ascvec.shl %215, 3, %mask : <ui32> -> <ui32>
            %224 = ascvec.or %222, %223, %mask : <ui32>, <ui32> -> <ui32>
            %225 = ascvec.shr %215, 5, %mask : <ui32> -> <ui32>
            %226 = ascvec.shl %215, 27, %mask : <ui32> -> <ui32>
            %227 = ascvec.or %225, %226, %mask : <ui32>, <ui32> -> <ui32>
            %228 = ascvec.shr %215, 13, %mask : <ui32> -> <ui32>
            %229 = ascvec.shl %215, 19, %mask : <ui32> -> <ui32>
            %230 = ascvec.or %228, %229, %mask : <ui32>, <ui32> -> <ui32>
            %231 = ascvec.shr %215, 24, %mask : <ui32> -> <ui32>
            %232 = ascvec.shl %215, 8, %mask : <ui32> -> <ui32>
            %233 = ascvec.or %231, %232, %mask : <ui32>, <ui32> -> <ui32>
            %234 = ascvec.shr %215, 8, %mask : <ui32> -> <ui32>
            %235 = ascvec.shl %215, 24, %mask : <ui32> -> <ui32>
            %236 = ascvec.or %234, %235, %mask : <ui32>, <ui32> -> <ui32>
            %237 = ascvec.shr %215, 16, %mask : <ui32> -> <ui32>
            %238 = ascvec.shl %215, 16, %mask : <ui32> -> <ui32>
            %239 = ascvec.or %237, %238, %mask : <ui32>, <ui32> -> <ui32>
            %240 = arith.extsi %218 : i32 to i64
            %241 = arith.muli %240, %arg11 : i64
            %242 = ascvec.duplicate_scalar %c65535_i32, %mask : i32 -> <ui32>
            %243 = ascvec.and %221, %242, %mask : <ui32>, <ui32> -> <ui32>
            %244 = arith.index_cast %241 : i64 to index
            %245 = ascvec.loadalign_unpack %arg12[%244] {mode = "b16_to_b32"} : memref<16384xbf16, 1>, index -> <bf16>
            %246 = ascvec.cast %245, %mask {rounding = 6 : i32} : <bf16> -> <f32>
            %247 = ascvec.mul %246, %207, %mask : <f32>
            %248 = ascvec.cast %247, %mask : <f32> -> <bf16>
            %249 = ascvec.cast %248, %mask {rounding = 6 : i32} : <bf16> -> <f32>
            %250 = ascvec.reinterpret %249 : <f32> -> <ui32>
            %251 = ascvec.shr %250, 24, %mask : <ui32> -> <ui32>
            %252 = ascvec.duplicate_scalar %c128_i32, %mask : i32 -> <ui32>
            %253 = ascvec.and %251, %252, %mask : <ui32>, <ui32> -> <ui32>
            %254 = ascvec.duplicate_scalar %c2147483647_i32, %mask : i32 -> <ui32>
            %255 = ascvec.and %250, %254, %mask : <ui32>, <ui32> -> <ui32>
            %256 = ascvec.shr %255, 5, %mask : <ui32> -> <ui32>
            %257 = ascvec.duplicate_scalar %c29360128_i32, %mask : i32 -> <ui32>
            %258 = ascvec.sub %256, %257, %mask : <ui32>
            %259 = ascvec.reinterpret %255 : <ui32> -> <f32>
            %260 = ascvec.mul_scalar %259, %cst, %mask : <f32>, f32
            %261 = ascvec.cast %260, %mask {rounding = 4 : i32} : <f32> -> <i32>
            %262 = ascvec.reinterpret %261 : <i32> -> <ui32>
            %263 = ascvec.cmp_ge_scalar %255, %c947912704_i32, %mask : <ui32>, i32 -> !ascvec.mask
            %264 = ascvec.select %258, %262, %263 : <ui32>
            %265 = ascvec.and %243, %242, %mask : <ui32>, <ui32> -> <ui32>
            %266 = ascvec.add %264, %265, %mask : <ui32>
            %267 = ascvec.shr %266, 16, %mask : <ui32> -> <ui32>
            %268 = ascvec.min_scalar %267, %c123_i32, %mask : <ui32>, i32
            %269 = ascvec.or %268, %253, %mask : <ui32>, <ui32> -> <ui32>
            %270 = ascvec.duplicate_scalar %c127_i32, %mask : i32 -> <ui32>
            %271 = ascvec.cmp_gt_scalar %255, %c2139095040_i32, %mask : <ui32>, i32 -> !ascvec.mask
            %272 = ascvec.select %270, %269, %271 : <ui32>
            %273 = ascvec.pack %272 src_elem_bits 32 dst_elem_bits 16 {part = "low"} : <ui32> -> <ui16>
            %274 = ascvec.pack %273 src_elem_bits 16 dst_elem_bits 8 {part = "low"} : <ui16> -> <ui8>
            %mask_19, %remaining_out_20 = ascvec.update_mask %c16_i32 elem_bits 8 : !ascvec.mask
            %275 = ascvec.squeeze_store_reg %274, %mask_19 : <ui8>, !ascvec.mask -> <ui8>
            ascvec.store_unalign %arg10[%c0_13], %275 : memref<16384xui8, 1>, <ui8>
            %276 = arith.addi %218, %c1_i32 overflow<nsw> : i32
            %277 = arith.extsi %276 : i32 to i64
            %278 = arith.muli %277, %arg11 : i64
            %279 = ascvec.and %233, %242, %mask : <ui32>, <ui32> -> <ui32>
            %280 = arith.index_cast %278 : i64 to index
            %281 = ascvec.loadalign_unpack %arg12[%280] {mode = "b16_to_b32"} : memref<16384xbf16, 1>, index -> <bf16>
            %282 = ascvec.cast %281, %mask {rounding = 6 : i32} : <bf16> -> <f32>
            %283 = ascvec.mul %282, %207, %mask : <f32>
            %284 = ascvec.cast %283, %mask : <f32> -> <bf16>
            %285 = ascvec.cast %284, %mask {rounding = 6 : i32} : <bf16> -> <f32>
            %286 = ascvec.reinterpret %285 : <f32> -> <ui32>
            %287 = ascvec.shr %286, 24, %mask : <ui32> -> <ui32>
            %288 = ascvec.and %287, %252, %mask : <ui32>, <ui32> -> <ui32>
            %289 = ascvec.and %286, %254, %mask : <ui32>, <ui32> -> <ui32>
            %290 = ascvec.shr %289, 5, %mask : <ui32> -> <ui32>
            %291 = ascvec.sub %290, %257, %mask : <ui32>
            %292 = ascvec.reinterpret %289 : <ui32> -> <f32>
            %293 = ascvec.mul_scalar %292, %cst, %mask : <f32>, f32
            %294 = ascvec.cast %293, %mask {rounding = 4 : i32} : <f32> -> <i32>
            %295 = ascvec.reinterpret %294 : <i32> -> <ui32>
            %296 = ascvec.cmp_ge_scalar %289, %c947912704_i32, %mask : <ui32>, i32 -> !ascvec.mask
            %297 = ascvec.select %291, %295, %296 : <ui32>
            %298 = ascvec.and %279, %242, %mask : <ui32>, <ui32> -> <ui32>
            %299 = ascvec.add %297, %298, %mask : <ui32>
            %300 = ascvec.shr %299, 16, %mask : <ui32> -> <ui32>
            %301 = ascvec.min_scalar %300, %c123_i32, %mask : <ui32>, i32
            %302 = ascvec.or %301, %288, %mask : <ui32>, <ui32> -> <ui32>
            %303 = ascvec.cmp_gt_scalar %289, %c2139095040_i32, %mask : <ui32>, i32 -> !ascvec.mask
            %304 = ascvec.select %270, %302, %303 : <ui32>
            %305 = ascvec.pack %304 src_elem_bits 32 dst_elem_bits 16 {part = "low"} : <ui32> -> <ui16>
            %306 = ascvec.pack %305 src_elem_bits 16 dst_elem_bits 8 {part = "low"} : <ui16> -> <ui8>
            %mask_21, %remaining_out_22 = ascvec.update_mask %c16_i32 elem_bits 8 : !ascvec.mask
            %307 = ascvec.squeeze_store_reg %306, %mask_21 : <ui8>, !ascvec.mask -> <ui8>
            ascvec.store_unalign %arg10[%c0_13], %307 : memref<16384xui8, 1>, <ui8>
            %308 = arith.addi %218, %c2_i32 overflow<nsw> : i32
            %309 = arith.extsi %308 : i32 to i64
            %310 = arith.muli %309, %arg11 : i64
            %311 = ascvec.and %224, %242, %mask : <ui32>, <ui32> -> <ui32>
            %312 = arith.index_cast %310 : i64 to index
            %313 = ascvec.loadalign_unpack %arg12[%312] {mode = "b16_to_b32"} : memref<16384xbf16, 1>, index -> <bf16>
            %314 = ascvec.cast %313, %mask {rounding = 6 : i32} : <bf16> -> <f32>
            %315 = ascvec.mul %314, %207, %mask : <f32>
            %316 = ascvec.cast %315, %mask : <f32> -> <bf16>
            %317 = ascvec.cast %316, %mask {rounding = 6 : i32} : <bf16> -> <f32>
            %318 = ascvec.reinterpret %317 : <f32> -> <ui32>
            %319 = ascvec.shr %318, 24, %mask : <ui32> -> <ui32>
            %320 = ascvec.and %319, %252, %mask : <ui32>, <ui32> -> <ui32>
            %321 = ascvec.and %318, %254, %mask : <ui32>, <ui32> -> <ui32>
            %322 = ascvec.shr %321, 5, %mask : <ui32> -> <ui32>
            %323 = ascvec.sub %322, %257, %mask : <ui32>
            %324 = ascvec.reinterpret %321 : <ui32> -> <f32>
            %325 = ascvec.mul_scalar %324, %cst, %mask : <f32>, f32
            %326 = ascvec.cast %325, %mask {rounding = 4 : i32} : <f32> -> <i32>
            %327 = ascvec.reinterpret %326 : <i32> -> <ui32>
            %328 = ascvec.cmp_ge_scalar %321, %c947912704_i32, %mask : <ui32>, i32 -> !ascvec.mask
            %329 = ascvec.select %323, %327, %328 : <ui32>
            %330 = ascvec.and %311, %242, %mask : <ui32>, <ui32> -> <ui32>
            %331 = ascvec.add %329, %330, %mask : <ui32>
            %332 = ascvec.shr %331, 16, %mask : <ui32> -> <ui32>
            %333 = ascvec.min_scalar %332, %c123_i32, %mask : <ui32>, i32
            %334 = ascvec.or %333, %320, %mask : <ui32>, <ui32> -> <ui32>
            %335 = ascvec.cmp_gt_scalar %321, %c2139095040_i32, %mask : <ui32>, i32 -> !ascvec.mask
            %336 = ascvec.select %270, %334, %335 : <ui32>
            %337 = ascvec.pack %336 src_elem_bits 32 dst_elem_bits 16 {part = "low"} : <ui32> -> <ui16>
            %338 = ascvec.pack %337 src_elem_bits 16 dst_elem_bits 8 {part = "low"} : <ui16> -> <ui8>
            %mask_23, %remaining_out_24 = ascvec.update_mask %c16_i32 elem_bits 8 : !ascvec.mask
            %339 = ascvec.squeeze_store_reg %338, %mask_23 : <ui8>, !ascvec.mask -> <ui8>
            ascvec.store_unalign %arg10[%c0_13], %339 : memref<16384xui8, 1>, <ui8>
            %340 = arith.addi %218, %c3_i32 overflow<nsw> : i32
            %341 = arith.extsi %340 : i32 to i64
            %342 = arith.muli %341, %arg11 : i64
            %343 = ascvec.and %215, %242, %mask : <ui32>, <ui32> -> <ui32>
            %344 = arith.index_cast %342 : i64 to index
            %345 = ascvec.loadalign_unpack %arg12[%344] {mode = "b16_to_b32"} : memref<16384xbf16, 1>, index -> <bf16>
            %346 = ascvec.cast %345, %mask {rounding = 6 : i32} : <bf16> -> <f32>
            %347 = ascvec.mul %346, %207, %mask : <f32>
            %348 = ascvec.cast %347, %mask : <f32> -> <bf16>
            %349 = ascvec.cast %348, %mask {rounding = 6 : i32} : <bf16> -> <f32>
            %350 = ascvec.reinterpret %349 : <f32> -> <ui32>
            %351 = ascvec.shr %350, 24, %mask : <ui32> -> <ui32>
            %352 = ascvec.and %351, %252, %mask : <ui32>, <ui32> -> <ui32>
            %353 = ascvec.and %350, %254, %mask : <ui32>, <ui32> -> <ui32>
            %354 = ascvec.shr %353, 5, %mask : <ui32> -> <ui32>
            %355 = ascvec.sub %354, %257, %mask : <ui32>
            %356 = ascvec.reinterpret %353 : <ui32> -> <f32>
            %357 = ascvec.mul_scalar %356, %cst, %mask : <f32>, f32
            %358 = ascvec.cast %357, %mask {rounding = 4 : i32} : <f32> -> <i32>
            %359 = ascvec.reinterpret %358 : <i32> -> <ui32>
            %360 = ascvec.cmp_ge_scalar %353, %c947912704_i32, %mask : <ui32>, i32 -> !ascvec.mask
            %361 = ascvec.select %355, %359, %360 : <ui32>
            %362 = ascvec.and %343, %242, %mask : <ui32>, <ui32> -> <ui32>
            %363 = ascvec.add %361, %362, %mask : <ui32>
            %364 = ascvec.shr %363, 16, %mask : <ui32> -> <ui32>
            %365 = ascvec.min_scalar %364, %c123_i32, %mask : <ui32>, i32
            %366 = ascvec.or %365, %352, %mask : <ui32>, <ui32> -> <ui32>
            %367 = ascvec.cmp_gt_scalar %353, %c2139095040_i32, %mask : <ui32>, i32 -> !ascvec.mask
            %368 = ascvec.select %270, %366, %367 : <ui32>
            %369 = ascvec.pack %368 src_elem_bits 32 dst_elem_bits 16 {part = "low"} : <ui32> -> <ui16>
            %370 = ascvec.pack %369 src_elem_bits 16 dst_elem_bits 8 {part = "low"} : <ui16> -> <ui8>
            %mask_25, %remaining_out_26 = ascvec.update_mask %c16_i32 elem_bits 8 : !ascvec.mask
            %371 = ascvec.squeeze_store_reg %370, %mask_25 : <ui8>, !ascvec.mask -> <ui8>
            ascvec.store_unalign %arg10[%c0_13], %371 : memref<16384xui8, 1>, <ui8>
            %372 = arith.addi %218, %c4_i32 overflow<nsw> : i32
            %373 = arith.extsi %372 : i32 to i64
            %374 = arith.muli %373, %arg11 : i64
            %375 = ascvec.and %227, %242, %mask : <ui32>, <ui32> -> <ui32>
            %376 = arith.index_cast %374 : i64 to index
            %377 = ascvec.loadalign_unpack %arg12[%376] {mode = "b16_to_b32"} : memref<16384xbf16, 1>, index -> <bf16>
            %378 = ascvec.cast %377, %mask {rounding = 6 : i32} : <bf16> -> <f32>
            %379 = ascvec.mul %378, %207, %mask : <f32>
            %380 = ascvec.cast %379, %mask : <f32> -> <bf16>
            %381 = ascvec.cast %380, %mask {rounding = 6 : i32} : <bf16> -> <f32>
            %382 = ascvec.reinterpret %381 : <f32> -> <ui32>
            %383 = ascvec.shr %382, 24, %mask : <ui32> -> <ui32>
            %384 = ascvec.and %383, %252, %mask : <ui32>, <ui32> -> <ui32>
            %385 = ascvec.and %382, %254, %mask : <ui32>, <ui32> -> <ui32>
            %386 = ascvec.shr %385, 5, %mask : <ui32> -> <ui32>
            %387 = ascvec.sub %386, %257, %mask : <ui32>
            %388 = ascvec.reinterpret %385 : <ui32> -> <f32>
            %389 = ascvec.mul_scalar %388, %cst, %mask : <f32>, f32
            %390 = ascvec.cast %389, %mask {rounding = 4 : i32} : <f32> -> <i32>
            %391 = ascvec.reinterpret %390 : <i32> -> <ui32>
            %392 = ascvec.cmp_ge_scalar %385, %c947912704_i32, %mask : <ui32>, i32 -> !ascvec.mask
            %393 = ascvec.select %387, %391, %392 : <ui32>
            %394 = ascvec.and %375, %242, %mask : <ui32>, <ui32> -> <ui32>
            %395 = ascvec.add %393, %394, %mask : <ui32>
            %396 = ascvec.shr %395, 16, %mask : <ui32> -> <ui32>
            %397 = ascvec.min_scalar %396, %c123_i32, %mask : <ui32>, i32
            %398 = ascvec.or %397, %384, %mask : <ui32>, <ui32> -> <ui32>
            %399 = ascvec.cmp_gt_scalar %385, %c2139095040_i32, %mask : <ui32>, i32 -> !ascvec.mask
            %400 = ascvec.select %270, %398, %399 : <ui32>
            %401 = ascvec.pack %400 src_elem_bits 32 dst_elem_bits 16 {part = "low"} : <ui32> -> <ui16>
            %402 = ascvec.pack %401 src_elem_bits 16 dst_elem_bits 8 {part = "low"} : <ui16> -> <ui8>
            %mask_27, %remaining_out_28 = ascvec.update_mask %c16_i32 elem_bits 8 : !ascvec.mask
            %403 = ascvec.squeeze_store_reg %402, %mask_27 : <ui8>, !ascvec.mask -> <ui8>
            ascvec.store_unalign %arg10[%c0_13], %403 : memref<16384xui8, 1>, <ui8>
            %404 = arith.addi %218, %c5_i32 overflow<nsw> : i32
            %405 = arith.extsi %404 : i32 to i64
            %406 = arith.muli %405, %arg11 : i64
            %407 = ascvec.and %236, %242, %mask : <ui32>, <ui32> -> <ui32>
            %408 = arith.index_cast %406 : i64 to index
            %409 = ascvec.loadalign_unpack %arg12[%408] {mode = "b16_to_b32"} : memref<16384xbf16, 1>, index -> <bf16>
            %410 = ascvec.cast %409, %mask {rounding = 6 : i32} : <bf16> -> <f32>
            %411 = ascvec.mul %410, %207, %mask : <f32>
            %412 = ascvec.cast %411, %mask : <f32> -> <bf16>
            %413 = ascvec.cast %412, %mask {rounding = 6 : i32} : <bf16> -> <f32>
            %414 = ascvec.reinterpret %413 : <f32> -> <ui32>
            %415 = ascvec.shr %414, 24, %mask : <ui32> -> <ui32>
            %416 = ascvec.and %415, %252, %mask : <ui32>, <ui32> -> <ui32>
            %417 = ascvec.and %414, %254, %mask : <ui32>, <ui32> -> <ui32>
            %418 = ascvec.shr %417, 5, %mask : <ui32> -> <ui32>
            %419 = ascvec.sub %418, %257, %mask : <ui32>
            %420 = ascvec.reinterpret %417 : <ui32> -> <f32>
            %421 = ascvec.mul_scalar %420, %cst, %mask : <f32>, f32
            %422 = ascvec.cast %421, %mask {rounding = 4 : i32} : <f32> -> <i32>
            %423 = ascvec.reinterpret %422 : <i32> -> <ui32>
            %424 = ascvec.cmp_ge_scalar %417, %c947912704_i32, %mask : <ui32>, i32 -> !ascvec.mask
            %425 = ascvec.select %419, %423, %424 : <ui32>
            %426 = ascvec.and %407, %242, %mask : <ui32>, <ui32> -> <ui32>
            %427 = ascvec.add %425, %426, %mask : <ui32>
            %428 = ascvec.shr %427, 16, %mask : <ui32> -> <ui32>
            %429 = ascvec.min_scalar %428, %c123_i32, %mask : <ui32>, i32
            %430 = ascvec.or %429, %416, %mask : <ui32>, <ui32> -> <ui32>
            %431 = ascvec.cmp_gt_scalar %417, %c2139095040_i32, %mask : <ui32>, i32 -> !ascvec.mask
            %432 = ascvec.select %270, %430, %431 : <ui32>
            %433 = ascvec.pack %432 src_elem_bits 32 dst_elem_bits 16 {part = "low"} : <ui32> -> <ui16>
            %434 = ascvec.pack %433 src_elem_bits 16 dst_elem_bits 8 {part = "low"} : <ui16> -> <ui8>
            %mask_29, %remaining_out_30 = ascvec.update_mask %c16_i32 elem_bits 8 : !ascvec.mask
            %435 = ascvec.squeeze_store_reg %434, %mask_29 : <ui8>, !ascvec.mask -> <ui8>
            ascvec.store_unalign %arg10[%c0_13], %435 : memref<16384xui8, 1>, <ui8>
            %436 = arith.addi %218, %c6_i32 overflow<nsw> : i32
            %437 = arith.extsi %436 : i32 to i64
            %438 = arith.muli %437, %arg11 : i64
            %439 = ascvec.and %230, %242, %mask : <ui32>, <ui32> -> <ui32>
            %440 = arith.index_cast %438 : i64 to index
            %441 = ascvec.loadalign_unpack %arg12[%440] {mode = "b16_to_b32"} : memref<16384xbf16, 1>, index -> <bf16>
            %442 = ascvec.cast %441, %mask {rounding = 6 : i32} : <bf16> -> <f32>
            %443 = ascvec.mul %442, %207, %mask : <f32>
            %444 = ascvec.cast %443, %mask : <f32> -> <bf16>
            %445 = ascvec.cast %444, %mask {rounding = 6 : i32} : <bf16> -> <f32>
            %446 = ascvec.reinterpret %445 : <f32> -> <ui32>
            %447 = ascvec.shr %446, 24, %mask : <ui32> -> <ui32>
            %448 = ascvec.and %447, %252, %mask : <ui32>, <ui32> -> <ui32>
            %449 = ascvec.and %446, %254, %mask : <ui32>, <ui32> -> <ui32>
            %450 = ascvec.shr %449, 5, %mask : <ui32> -> <ui32>
            %451 = ascvec.sub %450, %257, %mask : <ui32>
            %452 = ascvec.reinterpret %449 : <ui32> -> <f32>
            %453 = ascvec.mul_scalar %452, %cst, %mask : <f32>, f32
            %454 = ascvec.cast %453, %mask {rounding = 4 : i32} : <f32> -> <i32>
            %455 = ascvec.reinterpret %454 : <i32> -> <ui32>
            %456 = ascvec.cmp_ge_scalar %449, %c947912704_i32, %mask : <ui32>, i32 -> !ascvec.mask
            %457 = ascvec.select %451, %455, %456 : <ui32>
            %458 = ascvec.and %439, %242, %mask : <ui32>, <ui32> -> <ui32>
            %459 = ascvec.add %457, %458, %mask : <ui32>
            %460 = ascvec.shr %459, 16, %mask : <ui32> -> <ui32>
            %461 = ascvec.min_scalar %460, %c123_i32, %mask : <ui32>, i32
            %462 = ascvec.or %461, %448, %mask : <ui32>, <ui32> -> <ui32>
            %463 = ascvec.cmp_gt_scalar %449, %c2139095040_i32, %mask : <ui32>, i32 -> !ascvec.mask
            %464 = ascvec.select %270, %462, %463 : <ui32>
            %465 = ascvec.pack %464 src_elem_bits 32 dst_elem_bits 16 {part = "low"} : <ui32> -> <ui16>
            %466 = ascvec.pack %465 src_elem_bits 16 dst_elem_bits 8 {part = "low"} : <ui16> -> <ui8>
            %mask_31, %remaining_out_32 = ascvec.update_mask %c16_i32 elem_bits 8 : !ascvec.mask
            %467 = ascvec.squeeze_store_reg %466, %mask_31 : <ui8>, !ascvec.mask -> <ui8>
            ascvec.store_unalign %arg10[%c0_13], %467 : memref<16384xui8, 1>, <ui8>
            %468 = arith.addi %218, %c7_i32 overflow<nsw> : i32
            %469 = arith.extsi %468 : i32 to i64
            %470 = arith.muli %469, %arg11 : i64
            %471 = ascvec.and %239, %242, %mask : <ui32>, <ui32> -> <ui32>
            %472 = arith.index_cast %470 : i64 to index
            %473 = ascvec.loadalign_unpack %arg12[%472] {mode = "b16_to_b32"} : memref<16384xbf16, 1>, index -> <bf16>
            %474 = ascvec.cast %473, %mask {rounding = 6 : i32} : <bf16> -> <f32>
            %475 = ascvec.mul %474, %207, %mask : <f32>
            %476 = ascvec.cast %475, %mask : <f32> -> <bf16>
            %477 = ascvec.cast %476, %mask {rounding = 6 : i32} : <bf16> -> <f32>
            %478 = ascvec.reinterpret %477 : <f32> -> <ui32>
            %479 = ascvec.shr %478, 24, %mask : <ui32> -> <ui32>
            %480 = ascvec.and %479, %252, %mask : <ui32>, <ui32> -> <ui32>
            %481 = ascvec.and %478, %254, %mask : <ui32>, <ui32> -> <ui32>
            %482 = ascvec.shr %481, 5, %mask : <ui32> -> <ui32>
            %483 = ascvec.sub %482, %257, %mask : <ui32>
            %484 = ascvec.reinterpret %481 : <ui32> -> <f32>
            %485 = ascvec.mul_scalar %484, %cst, %mask : <f32>, f32
            %486 = ascvec.cast %485, %mask {rounding = 4 : i32} : <f32> -> <i32>
            %487 = ascvec.reinterpret %486 : <i32> -> <ui32>
            %488 = ascvec.cmp_ge_scalar %481, %c947912704_i32, %mask : <ui32>, i32 -> !ascvec.mask
            %489 = ascvec.select %483, %487, %488 : <ui32>
            %490 = ascvec.and %471, %242, %mask : <ui32>, <ui32> -> <ui32>
            %491 = ascvec.add %489, %490, %mask : <ui32>
            %492 = ascvec.shr %491, 16, %mask : <ui32> -> <ui32>
            %493 = ascvec.min_scalar %492, %c123_i32, %mask : <ui32>, i32
            %494 = ascvec.or %493, %480, %mask : <ui32>, <ui32> -> <ui32>
            %495 = ascvec.cmp_gt_scalar %481, %c2139095040_i32, %mask : <ui32>, i32 -> !ascvec.mask
            %496 = ascvec.select %270, %494, %495 : <ui32>
            %497 = ascvec.pack %496 src_elem_bits 32 dst_elem_bits 16 {part = "low"} : <ui32> -> <ui16>
            %498 = ascvec.pack %497 src_elem_bits 16 dst_elem_bits 8 {part = "low"} : <ui16> -> <ui8>
            %mask_33, %remaining_out_34 = ascvec.update_mask %c16_i32 elem_bits 8 : !ascvec.mask
            %499 = ascvec.squeeze_store_reg %498, %mask_33 : <ui8>, !ascvec.mask -> <ui8>
            ascvec.store_unalign %arg10[%c0_13], %499 : memref<16384xui8, 1>, <ui8>
          }
        }
        %c0_14 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        scf.for %arg16 = %c0_14 to %c1 step %c1 {
          ascvec.store_unalign_post %arg10[%c0_13] : memref<16384xui8, 1>
        }
        ascvec.mem_bar vst_vld
        %mask_15, %remaining_out_16 = ascvec.update_mask %c32_i32 elem_bits 8 : !ascvec.mask
        ascvec.store_unalign_begin %arg15 : memref<1024xui8, 1>
        scf.for %arg16 = %c0_i32_12 to %c4_i32 step %c1_i32  : i32 {
          %186 = arith.muli %arg16, %c32_i32 overflow<nsw> : i32
          %187 = arith.index_cast %186 : i32 to index
          %188 = ascvec.loadalign %arg13[%187] : memref<512xui32, 1>, index -> <ui32>
          %189 = arith.addi %186, %c16_i32 overflow<nsw> : i32
          %190 = arith.index_cast %189 : i32 to index
          %191 = ascvec.loadalign %arg13[%190] : memref<512xui32, 1>, index -> <ui32>
          %192 = ascvec.shl %191, 8, %mask : <ui32> -> <ui32>
          %193 = ascvec.or %188, %192, %mask : <ui32>, <ui32> -> <ui32>
          %194 = ascvec.pack %193 src_elem_bits 32 dst_elem_bits 16 {part = "low"} : <ui32> -> <ui16>
          %195 = ascvec.reinterpret %194 : <ui16> -> <ui8>
          %196 = ascvec.squeeze_store_reg %195, %mask_15 : <ui8>, !ascvec.mask -> <ui8>
          ascvec.store_unalign %arg15[%c0_13], %196 : memref<1024xui8, 1>, <ui8>
        }
        %c0_17 = arith.constant 0 : index
        %c1_18 = arith.constant 1 : index
        scf.for %arg16 = %c0_17 to %c1_18 step %c1_18 {
          ascvec.store_unalign_post %arg15[%c0_13] : memref<1024xui8, 1>
        }
        ascvec.mem_bar vst_vld
      } {cannir.vf_mode = "simd"}
      ascvec.release_buf v, %77, 0 {cannir.buffer.generated_sync}
      ascvec.release_buf v, %36, 0 {cannir.buffer.generated_sync}
      ascvec.release_buf v, %72, 0 {cannir.buffer.generated_sync}
      %105 = cannir.extract_layout %arg1 : !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
      %106 = cannir.extract_shape %105 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, -1)>
      %107 = cannir.extract_dim %106[0] : !cannir.shape<(-1, -1)>
      %108 = arith.minsi %27, %107 : i64
      %109 = arith.subi %107, %108 : i64
      %110 = arith.minsi %109, %c1_i64 : i64
      %111 = arith.cmpi sle, %27, %107 : i64
      %112 = arith.select %111, %110, %c0_i64 : i64
      %113 = arith.cmpi sgt, %112, %c1_i64 : i64
      %114 = arith.select %113, %c1_i64, %112 : i64
      %115 = cannir.extract_dim %106[1] : !cannir.shape<(-1, -1)>
      %116 = arith.divsi %115, %49 : i64
      %117 = arith.minsi %29, %116 : i64
      %118 = arith.muli %117, %30 : i64
      %119 = arith.subi %115, %118 : i64
      %120 = arith.minsi %30, %119 : i64
      %121 = arith.cmpi sle, %29, %116 : i64
      %122 = arith.select %121, %120, %c0_i64 : i64
      %123 = arith.cmpi sgt, %122, %c16384_i64 : i64
      %124 = arith.select %123, %c16384_i64, %122 : i64
      %125 = cannir.extract_physical_layout %arg1 : !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
      %126 = cannir.extract_stride %125 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
      %127 = cannir.extract_dim %126[0] : !cannir.stride<(-1, 1)>
      %128 = arith.muli %27, %127 : i64
      %129 = arith.addi %128, %59 : i64
      %130 = arith.muli %129, %c8_i64 : i64
      %131 = ascscalar.memref_address %alloc_0 : memref<1xui8>
      %132 = arith.divui %130, %c8_i64 : i64
      %133 = arith.addi %131, %132 : i64
      %134 = ascscalar.bind_gm_ptr %133 : memref<1xui8>
      %135 = ascscalar.bind_local_ptr %70 : memref<1xui8, 1>
      %136 = cannir.extract_stride %105 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
      %137 = cannir.extract_dim %136[0] : !cannir.stride<(-1, 1)>
      %138 = arith.cmpi ugt, %114, %c0_i64 : i64
      %139 = arith.cmpi ugt, %124, %c0_i64 : i64
      %140 = arith.andi %138, %139 : i1
      scf.if %140 {
        ascvec.get_buf mte3, %72, 0 {cannir.buffer.generated_sync}
        %186 = arith.trunci %114 : i64 to i32
        %187 = arith.trunci %124 : i64 to i32
        ascvec.copy_ub2gm %134[%c0], %135[%c0], %186, %187, %137, %c16384_i32 l2_cache_ctl %c0_i32 : memref<1xui8>, memref<1xui8, 1>
        ascvec.release_buf mte3, %72, 0 {cannir.buffer.generated_sync}
      }
      %141 = arith.remsi %30, %c32_i64 : i64
      %142 = arith.cmpi ne, %141, %c0_i64 : i64
      %143 = arith.cmpi slt, %141, %c0_i64 : i64
      %144 = arith.andi %142, %143 : i1
      %145 = arith.divsi %30, %c32_i64 : i64
      %146 = arith.extui %144 : i1 to i64
      %147 = arith.subi %145, %146 : i64
      %148 = cannir.extract_layout %arg2 : !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
      %149 = cannir.extract_shape %148 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, -1)>
      %150 = cannir.extract_dim %149[0] : !cannir.shape<(-1, -1)>
      %151 = arith.minsi %27, %150 : i64
      %152 = arith.subi %150, %151 : i64
      %153 = arith.minsi %152, %c1_i64 : i64
      %154 = arith.cmpi sle, %27, %150 : i64
      %155 = arith.select %154, %153, %c0_i64 : i64
      %156 = arith.cmpi sgt, %155, %c1_i64 : i64
      %157 = arith.select %156, %c1_i64, %155 : i64
      %158 = cannir.extract_dim %149[1] : !cannir.shape<(-1, -1)>
      %159 = arith.maxsi %147, %c1_i64 : i64
      %160 = arith.divsi %158, %159 : i64
      %161 = arith.minsi %29, %160 : i64
      %162 = arith.muli %161, %147 : i64
      %163 = arith.subi %158, %162 : i64
      %164 = arith.minsi %147, %163 : i64
      %165 = arith.cmpi sle, %29, %160 : i64
      %166 = arith.select %165, %164, %c0_i64 : i64
      %167 = arith.cmpi sgt, %166, %c1024_i64 : i64
      %168 = arith.select %167, %c1024_i64, %166 : i64
      %169 = cannir.extract_physical_layout %arg2 : !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
      %170 = cannir.extract_stride %169 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
      %171 = cannir.extract_dim %170[0] : !cannir.stride<(-1, 1)>
      %172 = arith.muli %27, %171 : i64
      %173 = arith.muli %29, %147 : i64
      %174 = arith.addi %172, %173 : i64
      %175 = arith.muli %174, %c8_i64 : i64
      %176 = ascscalar.memref_address %alloc_1 : memref<1xui8>
      %177 = arith.divui %175, %c8_i64 : i64
      %178 = arith.addi %176, %177 : i64
      %179 = ascscalar.bind_gm_ptr %178 : memref<1xui8>
      %180 = ascscalar.bind_local_ptr %75 : memref<1xui8, 1>
      %181 = cannir.extract_stride %148 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
      %182 = cannir.extract_dim %181[0] : !cannir.stride<(-1, 1)>
      %183 = arith.cmpi ugt, %157, %c0_i64 : i64
      %184 = arith.cmpi ugt, %168, %c0_i64 : i64
      %185 = arith.andi %183, %184 : i1
      scf.if %185 {
        ascvec.get_buf mte3, %77, 0 {cannir.buffer.generated_sync}
        %186 = arith.trunci %157 : i64 to i32
        %187 = arith.trunci %168 : i64 to i32
        ascvec.copy_ub2gm %179[%c0], %180[%c0], %186, %187, %182, %c1024_i32 l2_cache_ctl %c0_i32 : memref<1xui8>, memref<1xui8, 1>
        ascvec.release_buf mte3, %77, 0 {cannir.buffer.generated_sync}
      }
      scf.yield %38 : i64
    }
    return
  }
}
