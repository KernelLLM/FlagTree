module {
  func.func @run_wide_non_tail_graph_rng_11d928bb32c49713e8135dc51d7fff05(%arg0: !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg1: !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg2: !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: !cannir.tensor<<gm, i64>, <<(1)>, <(1)>>, <<(1)>, <(1)>>, identity>, %arg7: !cannir.tensor<<gm, i64>, <<(1)>, <(1)>>, <<(1)>, <(1)>>, identity>, %arg8: i64) attributes {cannbotdsl.contract_digest = "11d928bb32c49713e8135dc51d7fff05f197a8f1fdebf2808f660ad106b6281b", cannbotdsl.public_slot_count = 9 : i64, cannbotdsl.slot_dtype_bits = [16, 8, 8, 64, 64, 64, 64, 64, 64], cannbotdsl.slot_dtype_codes = [4, 1, 1, 0, 0, 0, 0, 0, 0], cannbotdsl.slot_dtype_lanes = [1, 1, 1, 1, 1, 1, 1, 1, 1], cannbotdsl.slot_flags = [1, 1, 1, 1, 1, 1, 1, 1, 1], cannbotdsl.slot_kinds = [1, 1, 1, 2, 2, 2, 1, 1, 2], cannbotdsl.slot_type_digests = ["5d7cdc96dc21b3b739e32aeccf58c1df113bec54701c75107ed650f2544a07da", "171ddd09eb070fb4a6413ae87b2ceecddc233a9a173a0404e96285a4bb8b8659", "891bcd21e721f8d1a666bcca4e8705ec63796c1add8ae000536d0d98f0274d3a", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "a9d876affafdf57be3f0e1a45cb0b293a48851dbda8b74101f5184f5130b1b63", "a9d876affafdf57be3f0e1a45cb0b293a48851dbda8b74101f5184f5130b1b63", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515"], cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.host} {
    cannir.kernel_launch @kernel__DynamicMxQuantSrKernel_run_wide_non_tail_0(%arg5) [%arg0, %arg1, %arg2, %arg3, %arg4, %arg6, %arg7, %arg8] {cannbotdsl.profile = {indices = array<i64: 0, 1, 2, 5, 6>, name = "_DynamicMxQuantSrKernel_run_wide_non_tail", op_type = "_DynamicMxQuantSrKernel_run_wide_non_tail", roles = array<i64: 0, 1, 1, 0, 0>}} : !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, i64, i64, !cannir.tensor<<gm, i64>, <<(1)>, <(1)>>, <<(1)>, <(1)>>, identity>, !cannir.tensor<<gm, i64>, <<(1)>, <(1)>>, <<(1)>, <(1)>>, identity>, i64
    return
  }
  func.func @kernel__DynamicMxQuantSrKernel_run_wide_non_tail_0(%arg0: !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg1: !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg2: !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg3: i64, %arg4: i64, %arg5: !cannir.tensor<<gm, i64>, <<(1)>, <(1)>>, <<(1)>, <(1)>>, identity>, %arg6: !cannir.tensor<<gm, i64>, <<(1)>, <(1)>>, <<(1)>, <(1)>>, identity>, %arg7: i64) attributes {cannir.buffer.auto_sync, cannir.buffer.sync_analyzed, cannir.buffer.sync_verified, cannir.buffer.tensor_state, cannir.core = "aiv", cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.device, cannir.explicit_ring_buffer_cursors, cannir.kernel_group = "kernel__DynamicMxQuantSrKernel_run_wide_non_tail_0", cannir.resource.ids_prepared} {
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
    %5 = cannir.extract_layout %arg0 : !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
    %6 = cannir.extract_shape %5 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, -1)>
    %7 = cannir.extract_dim %6[0] : !cannir.shape<(-1, -1)>
    %8 = arith.remsi %arg3, %c64_i64 : i64
    %9 = arith.cmpi ne, %8, %c0_i64 : i64
    %10 = arith.cmpi slt, %8, %c0_i64 : i64
    %11 = arith.andi %9, %10 : i1
    %12 = arith.divsi %arg3, %c64_i64 : i64
    %13 = arith.extui %11 : i1 to i64
    %14 = arith.subi %12, %13 : i64
    %15 = arith.muli %7, %14 : i64
    %16 = arith.cmpi sge, %15, %c256_i64 : i64
    %17 = scf.if %16 -> (i1) {
      %30 = arith.remsi %arg3, %c256_i64 : i64
      %31 = arith.cmpi ne, %30, %c0_i64 : i64
      %32 = arith.cmpi slt, %30, %c0_i64 : i64
      %33 = arith.andi %31, %32 : i1
      %34 = arith.select %33, %c256_i64, %c0_i64 : i64
      %35 = arith.addi %30, %34 overflow<nsw> : i64
      %36 = arith.cmpi eq, %35, %c0_i64 : i64
      %37 = arith.cmpi eq, %arg4, %c64_i64 : i64
      %38 = arith.select %36, %37, %false : i1
      scf.yield %38 : i1
    } else {
      scf.yield %false : i1
    }
    %18 = scf.if %17 -> (i64) {
      scf.yield %c256_i64 : i64
    } else {
      %30 = arith.remsi %arg3, %c128_i64 : i64
      %31 = arith.cmpi ne, %30, %c0_i64 : i64
      %32 = arith.cmpi slt, %30, %c0_i64 : i64
      %33 = arith.andi %31, %32 : i1
      %34 = arith.select %33, %c128_i64, %c0_i64 : i64
      %35 = arith.addi %30, %34 overflow<nsw> : i64
      %36 = arith.cmpi eq, %35, %c0_i64 : i64
      %37 = arith.select %16, %36, %false : i1
      %38 = arith.select %37, %c128_i64, %c64_i64 : i64
      scf.yield %38 : i64
    }
    %19 = arith.remsi %arg3, %18 : i64
    %20 = arith.cmpi ne, %19, %c0_i64 : i64
    %21 = arith.cmpi slt, %19, %c0_i64 : i64
    %22 = arith.andi %20, %21 : i1
    %23 = arith.divsi %arg3, %18 : i64
    %24 = arith.extui %22 : i1 to i64
    %25 = arith.subi %23, %24 : i64
    %26 = arith.muli %7, %25 : i64
    %27 = ascscalar.get_block_idx
    %28 = ascscalar.get_block_num
    %29 = scf.for %arg8 = %27 to %26 step %28 iter_args(%arg9 = %c0_i64) -> (i64)  : i64 {
      %30 = arith.remsi %arg8, %25 : i64
      %31 = arith.cmpi ne, %30, %c0_i64 : i64
      %32 = arith.cmpi slt, %30, %c0_i64 : i64
      %33 = arith.cmpi slt, %25, %c0_i64 : i64
      %34 = arith.xori %32, %33 : i1
      %35 = arith.andi %31, %34 : i1
      %36 = arith.divsi %arg8, %25 : i64
      %37 = arith.extui %35 : i1 to i64
      %38 = arith.subi %36, %37 : i64
      %39 = arith.select %35, %25, %c0_i64 : i64
      %40 = arith.addi %30, %39 : i64
      %41 = arith.muli %18, %arg4 : i64
      %42 = arith.cmpi eq, %arg9, %c0_i64 : i64
      %43 = ascscalar.memref_address %alloc_4 : memref<16384xbf16, 1>
      %44 = ascscalar.memref_address %alloc_5 : memref<16384xbf16, 1>
      %45 = arith.select %42, %43, %44 : i64
      %46 = arith.cmpi ne, %arg9, %c0_i64 : i64
      %47 = arith.extui %46 : i1 to i64
      %48 = arith.addi %arg9, %c1_i64 overflow<nsw> : i64
      %49 = arith.remui %48, %c2_i64 : i64
      %50 = arith.minsi %38, %7 : i64
      %51 = arith.subi %7, %50 : i64
      %52 = arith.minsi %51, %c1_i64 : i64
      %53 = arith.cmpi sle, %38, %7 : i64
      %54 = arith.select %53, %52, %c0_i64 : i64
      %55 = cannir.extract_physical_layout %arg0 : !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
      %56 = cannir.extract_stride %55 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
      %57 = cannir.extract_dim %56[0] : !cannir.stride<(-1, 1)>
      %58 = arith.muli %57, %c2_i64 : i64
      %59 = cannir.extract_dim %6[1] : !cannir.shape<(-1, -1)>
      %60 = arith.maxsi %41, %c1_i64 : i64
      %61 = arith.divsi %59, %60 : i64
      %62 = arith.minsi %40, %61 : i64
      %63 = arith.muli %62, %41 : i64
      %64 = arith.subi %59, %63 : i64
      %65 = arith.minsi %41, %64 : i64
      %66 = arith.cmpi sle, %40, %61 : i64
      %67 = arith.select %66, %65, %c0_i64 : i64
      %68 = arith.muli %38, %57 : i64
      %69 = arith.muli %68, %c2_i64 : i64
      %70 = arith.muli %40, %41 : i64
      %71 = arith.muli %70, %c2_i64 : i64
      %72 = arith.addi %69, %71 : i64
      %73 = arith.muli %67, %c2_i64 : i64
      %74 = arith.cmpi eq, %73, %c0_i64 : i64
      %75 = arith.select %74, %c0_i64, %54 : i64
      %76 = arith.select %74, %c2_i64, %73 : i64
      %77 = ascscalar.bind_local_ptr %45 : memref<1xbf16, 1>
      ascvec.get_buf mte2, %47, 0 {cannir.buffer.generated_sync}
      "ascvec.copy_gm2ub_plan"(%77, %alloc, %c0_i32, %c0_i32, %c0_i64, %72, %75, %76, %58, %c32768_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
      ascvec.release_buf mte2, %47, 0 {cannir.buffer.generated_sync}
      %78 = ascscalar.bind_local_ptr %45 : memref<16384xbf16, 1>
      %79 = ascscalar.memref_address %alloc_6 : memref<16384xui8, 1>
      %80 = ascscalar.memref_address %alloc_7 : memref<16384xui8, 1>
      %81 = arith.select %42, %79, %80 : i64
      %82 = ascscalar.bind_local_ptr %81 : memref<16384xui8, 1>
      %83 = arith.select %42, %c2_i64, %c3_i64 : i64
      %84 = ascscalar.memref_address %alloc_8 : memref<1024xui8, 1>
      %85 = ascscalar.memref_address %alloc_9 : memref<1024xui8, 1>
      %86 = arith.select %42, %84, %85 : i64
      %87 = ascscalar.bind_local_ptr %86 : memref<1024xui8, 1>
      %88 = arith.select %42, %c4_i64, %c5_i64 : i64
      %89 = arith.remsi %arg4, %c64_i64 : i64
      %90 = arith.cmpi ne, %89, %c0_i64 : i64
      %91 = arith.cmpi slt, %89, %c0_i64 : i64
      %92 = arith.andi %90, %91 : i1
      %93 = arith.divsi %arg4, %c64_i64 : i64
      %94 = arith.extui %92 : i1 to i64
      %95 = arith.subi %93, %94 : i64
      scf.for %arg10 = %c0_i64 to %95 step %c1_i64  : i64 {
        %177 = arith.muli %arg10, %c64_i64 : i64
        %178 = arith.muli %38, %arg4 : i64
        %179 = arith.addi %178, %177 : i64
        %180 = arith.muli %40, %18 : i64
        %181 = arith.remsi %180, %c16_i64 : i64
        %182 = arith.cmpi ne, %181, %c0_i64 : i64
        %183 = arith.cmpi slt, %181, %c0_i64 : i64
        %184 = arith.andi %182, %183 : i1
        %185 = arith.divsi %180, %c16_i64 : i64
        %186 = arith.extui %184 : i1 to i64
        %187 = arith.subi %185, %186 : i64
        %188 = arith.remsi %arg3, %c16_i64 : i64
        %189 = arith.cmpi ne, %188, %c0_i64 : i64
        %190 = arith.cmpi slt, %188, %c0_i64 : i64
        %191 = arith.andi %189, %190 : i1
        %192 = arith.divsi %arg3, %c16_i64 : i64
        %193 = arith.extui %191 : i1 to i64
        %194 = arith.subi %192, %193 : i64
        %195 = arith.trunci %1 : i64 to i32
        %196 = arith.extui %195 : i32 to i64
        %197 = arith.subi %1, %196 : i64
        %198 = arith.remsi %197, %c4294967296_i64 : i64
        %199 = arith.cmpi ne, %198, %c0_i64 : i64
        %200 = arith.cmpi slt, %198, %c0_i64 : i64
        %201 = arith.andi %199, %200 : i1
        %202 = arith.divsi %197, %c4294967296_i64 : i64
        %203 = arith.extui %201 : i1 to i64
        %204 = arith.subi %202, %203 : i64
        %205 = arith.trunci %204 : i64 to i32
        %206 = arith.trunci %18 : i64 to i32
        %207 = arith.divsi %206, %c16_i32 : i32
        ascvec.vf(%187, %179, %194, %4, %195, %205, %alloc_11, %207 : i64, i64, i64, i64, i32, i32, memref<4096xi32, 1>, i32) {
        ^bb0(%arg11: i64, %arg12: i64, %arg13: i64, %arg14: i64, %arg15: i32, %arg16: i32, %arg17: memref<4096xi32, 1>, %arg18: i32):
          %c1_i32 = arith.constant 1 : i32
          %c192_i32 = arith.constant 192 : i32
          %c128_i32 = arith.constant 128 : i32
          %c64_i32_12 = arith.constant 64 : i32
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
          %c0_i64_13 = arith.constant 0 : i64
          %c4294967296_i64_14 = arith.constant 4294967296 : i64
          %c0_i32_15 = arith.constant 0 : i32
          %211 = ascvec.create_mask : !ascvec.mask
          %212 = ascvec.arange %c0_i32_15 : i32 -> <ui32>
          scf.for %arg19 = %c0_i32_15 to %arg18 step %c1_i32  : i32 {
            %213 = arith.extsi %arg19 : i32 to i64
            %214 = arith.addi %arg11, %213 : i64
            %215 = arith.trunci %arg12 : i64 to i32
            %216 = arith.extui %215 : i32 to i64
            %217 = arith.subi %arg12, %216 : i64
            %218 = arith.remsi %217, %c4294967296_i64_14 : i64
            %219 = arith.cmpi ne, %218, %c0_i64_13 : i64
            %220 = arith.cmpi slt, %218, %c0_i64_13 : i64
            %221 = arith.andi %219, %220 : i1
            %222 = arith.divsi %217, %c4294967296_i64_14 : i64
            %223 = arith.extui %221 : i1 to i64
            %224 = arith.subi %222, %223 : i64
            %225 = arith.trunci %224 : i64 to i32
            %226 = arith.trunci %214 : i64 to i32
            %227 = arith.extui %226 : i32 to i64
            %228 = arith.subi %214, %227 : i64
            %229 = arith.remsi %228, %c4294967296_i64_14 : i64
            %230 = arith.cmpi ne, %229, %c0_i64_13 : i64
            %231 = arith.cmpi slt, %229, %c0_i64_13 : i64
            %232 = arith.andi %230, %231 : i1
            %233 = arith.divsi %228, %c4294967296_i64_14 : i64
            %234 = arith.extui %232 : i1 to i64
            %235 = arith.subi %233, %234 : i64
            %236 = arith.trunci %235 : i64 to i32
            %237 = ascvec.duplicate_scalar %215, %211 : i32 -> <ui32>
            %238 = ascvec.duplicate_scalar %225, %211 : i32 -> <ui32>
            %239 = ascvec.duplicate_scalar %c0_i32_15, %211 : i32 -> <ui32>
            %carry, %dst = ascvec.addc %237, %212, %211 : <ui32>, <ui32> -> !ascvec.mask, <ui32>
            %carry_16, %dst_17 = ascvec.addc %238, %239, %carry, %211 : <ui32>, <ui32> -> !ascvec.mask, <ui32>
            %240 = ascvec.duplicate_scalar %226, %211 : i32 -> <ui32>
            %241 = ascvec.duplicate_scalar %236, %211 : i32 -> <ui32>
            %242 = arith.trunci %arg13 : i64 to i32
            %243 = arith.extui %242 : i32 to i64
            %244 = arith.subi %arg13, %243 : i64
            %245 = arith.remsi %244, %c4294967296_i64_14 : i64
            %246 = arith.cmpi ne, %245, %c0_i64_13 : i64
            %247 = arith.cmpi slt, %245, %c0_i64_13 : i64
            %248 = arith.andi %246, %247 : i1
            %249 = arith.divsi %244, %c4294967296_i64_14 : i64
            %250 = arith.extui %248 : i1 to i64
            %251 = arith.subi %249, %250 : i64
            %252 = arith.trunci %251 : i64 to i32
            %253 = ascvec.duplicate_scalar %242, %211 : i32 -> <ui32>
            %dst0, %dst1 = ascvec.mull %dst, %253, %211 : <ui32>, <ui32> -> <ui32>, <ui32>
            %254 = ascvec.mul_scalar %dst, %252, %211 : <ui32>, i32
            %255 = ascvec.add %dst1, %254, %211 : <ui32>
            %256 = ascvec.mul_scalar %dst_17, %242, %211 : <ui32>, i32
            %257 = ascvec.add %255, %256, %211 : <ui32>
            %carry_18, %dst_19 = ascvec.addc %dst0, %240, %211 : <ui32>, <ui32> -> !ascvec.mask, <ui32>
            %carry_20, %dst_21 = ascvec.addc %257, %241, %carry_18, %211 : <ui32>, <ui32> -> !ascvec.mask, <ui32>
            %258 = arith.trunci %arg14 : i64 to i32
            %259 = arith.extui %258 : i32 to i64
            %260 = arith.subi %arg14, %259 : i64
            %261 = arith.remsi %260, %c4294967296_i64_14 : i64
            %262 = arith.cmpi ne, %261, %c0_i64_13 : i64
            %263 = arith.cmpi slt, %261, %c0_i64_13 : i64
            %264 = arith.andi %262, %263 : i1
            %265 = arith.divsi %260, %c4294967296_i64_14 : i64
            %266 = arith.extui %264 : i1 to i64
            %267 = arith.subi %265, %266 : i64
            %268 = arith.trunci %267 : i64 to i32
            %269 = arith.divui %258, %c4_i32 : i32
            %270 = arith.remui %268, %c4_i32 : i32
            %271 = arith.muli %270, %c1073741824_i32 : i32
            %272 = arith.addi %269, %271 : i32
            %273 = arith.divui %268, %c4_i32 : i32
            %274 = ascvec.shr %dst_19, 14, %211 : <ui32> -> <ui32>
            %275 = ascvec.shl %dst_21, 18, %211 : <ui32> -> <ui32>
            %276 = ascvec.or %274, %275, %211 : <ui32>, <ui32> -> <ui32>
            %277 = ascvec.shr %dst_21, 14, %211 : <ui32> -> <ui32>
            %278 = ascvec.duplicate_scalar %272, %211 : i32 -> <ui32>
            %279 = ascvec.duplicate_scalar %273, %211 : i32 -> <ui32>
            %carry_22, %dst_23 = ascvec.addc %278, %276, %211 : <ui32>, <ui32> -> !ascvec.mask, <ui32>
            %carry_24, %dst_25 = ascvec.addc %279, %277, %carry_22, %211 : <ui32>, <ui32> -> !ascvec.mask, <ui32>
            %280 = ascvec.duplicate_scalar %c16383_i32, %211 : i32 -> <ui32>
            %281 = ascvec.and %dst_19, %280, %211 : <ui32>, <ui32> -> <ui32>
            %282 = ascvec.duplicate_scalar %c-766435501_i32, %211 : i32 -> <ui32>
            %283 = ascvec.duplicate_scalar %c-845247145_i32, %211 : i32 -> <ui32>
            %dst0_26, %dst1_27 = ascvec.mull %239, %282, %211 : <ui32>, <ui32> -> <ui32>, <ui32>
            %dst0_28, %dst1_29 = ascvec.mull %dst_23, %283, %211 : <ui32>, <ui32> -> <ui32>, <ui32>
            %284 = ascvec.xor %dst1_29, %281, %211 : <ui32>, <ui32> -> <ui32>
            %285 = ascvec.xor %dst1_27, %dst_25, %211 : <ui32>, <ui32> -> <ui32>
            %286 = ascvec.duplicate_scalar %arg15, %211 : i32 -> <ui32>
            %287 = ascvec.duplicate_scalar %arg16, %211 : i32 -> <ui32>
            %288 = ascvec.xor %284, %286, %211 : <ui32>, <ui32> -> <ui32>
            %289 = ascvec.xor %285, %287, %211 : <ui32>, <ui32> -> <ui32>
            %dst0_30, %dst1_31 = ascvec.mull %288, %282, %211 : <ui32>, <ui32> -> <ui32>, <ui32>
            %dst0_32, %dst1_33 = ascvec.mull %289, %283, %211 : <ui32>, <ui32> -> <ui32>, <ui32>
            %290 = ascvec.xor %dst1_33, %dst0_28, %211 : <ui32>, <ui32> -> <ui32>
            %291 = ascvec.xor %dst1_31, %dst0_26, %211 : <ui32>, <ui32> -> <ui32>
            %292 = arith.addi %arg15, %c-1640531527_i32 : i32
            %293 = ascvec.duplicate_scalar %292, %211 : i32 -> <ui32>
            %294 = arith.addi %arg16, %c-1150833019_i32 : i32
            %295 = ascvec.duplicate_scalar %294, %211 : i32 -> <ui32>
            %296 = ascvec.xor %290, %293, %211 : <ui32>, <ui32> -> <ui32>
            %297 = ascvec.xor %291, %295, %211 : <ui32>, <ui32> -> <ui32>
            %dst0_34, %dst1_35 = ascvec.mull %296, %282, %211 : <ui32>, <ui32> -> <ui32>, <ui32>
            %dst0_36, %dst1_37 = ascvec.mull %297, %283, %211 : <ui32>, <ui32> -> <ui32>, <ui32>
            %298 = ascvec.xor %dst1_37, %dst0_32, %211 : <ui32>, <ui32> -> <ui32>
            %299 = ascvec.xor %dst1_35, %dst0_30, %211 : <ui32>, <ui32> -> <ui32>
            %300 = arith.addi %arg15, %c1013904242_i32 : i32
            %301 = ascvec.duplicate_scalar %300, %211 : i32 -> <ui32>
            %302 = arith.addi %arg16, %c1993301258_i32 : i32
            %303 = ascvec.duplicate_scalar %302, %211 : i32 -> <ui32>
            %304 = ascvec.xor %298, %301, %211 : <ui32>, <ui32> -> <ui32>
            %305 = ascvec.xor %299, %303, %211 : <ui32>, <ui32> -> <ui32>
            %dst0_38, %dst1_39 = ascvec.mull %304, %282, %211 : <ui32>, <ui32> -> <ui32>, <ui32>
            %dst0_40, %dst1_41 = ascvec.mull %305, %283, %211 : <ui32>, <ui32> -> <ui32>, <ui32>
            %306 = ascvec.xor %dst1_41, %dst0_36, %211 : <ui32>, <ui32> -> <ui32>
            %307 = ascvec.xor %dst1_39, %dst0_34, %211 : <ui32>, <ui32> -> <ui32>
            %308 = arith.addi %arg15, %c-626627285_i32 : i32
            %309 = ascvec.duplicate_scalar %308, %211 : i32 -> <ui32>
            %310 = arith.addi %arg16, %c842468239_i32 : i32
            %311 = ascvec.duplicate_scalar %310, %211 : i32 -> <ui32>
            %312 = ascvec.xor %306, %309, %211 : <ui32>, <ui32> -> <ui32>
            %313 = ascvec.xor %307, %311, %211 : <ui32>, <ui32> -> <ui32>
            %dst0_42, %dst1_43 = ascvec.mull %312, %282, %211 : <ui32>, <ui32> -> <ui32>, <ui32>
            %dst0_44, %dst1_45 = ascvec.mull %313, %283, %211 : <ui32>, <ui32> -> <ui32>, <ui32>
            %314 = ascvec.xor %dst1_45, %dst0_40, %211 : <ui32>, <ui32> -> <ui32>
            %315 = ascvec.xor %dst1_43, %dst0_38, %211 : <ui32>, <ui32> -> <ui32>
            %316 = arith.addi %arg15, %c2027808484_i32 : i32
            %317 = ascvec.duplicate_scalar %316, %211 : i32 -> <ui32>
            %318 = arith.addi %arg16, %c-308364780_i32 : i32
            %319 = ascvec.duplicate_scalar %318, %211 : i32 -> <ui32>
            %320 = ascvec.xor %314, %317, %211 : <ui32>, <ui32> -> <ui32>
            %321 = ascvec.xor %315, %319, %211 : <ui32>, <ui32> -> <ui32>
            %dst0_46, %dst1_47 = ascvec.mull %320, %282, %211 : <ui32>, <ui32> -> <ui32>, <ui32>
            %dst0_48, %dst1_49 = ascvec.mull %321, %283, %211 : <ui32>, <ui32> -> <ui32>, <ui32>
            %322 = ascvec.xor %dst1_49, %dst0_44, %211 : <ui32>, <ui32> -> <ui32>
            %323 = ascvec.xor %dst1_47, %dst0_42, %211 : <ui32>, <ui32> -> <ui32>
            %324 = arith.addi %arg15, %c387276957_i32 : i32
            %325 = ascvec.duplicate_scalar %324, %211 : i32 -> <ui32>
            %326 = arith.addi %arg16, %c-1459197799_i32 : i32
            %327 = ascvec.duplicate_scalar %326, %211 : i32 -> <ui32>
            %328 = ascvec.xor %322, %325, %211 : <ui32>, <ui32> -> <ui32>
            %329 = ascvec.xor %323, %327, %211 : <ui32>, <ui32> -> <ui32>
            %dst0_50, %dst1_51 = ascvec.mull %328, %282, %211 : <ui32>, <ui32> -> <ui32>, <ui32>
            %dst0_52, %dst1_53 = ascvec.mull %329, %283, %211 : <ui32>, <ui32> -> <ui32>, <ui32>
            %330 = ascvec.xor %dst1_53, %dst0_48, %211 : <ui32>, <ui32> -> <ui32>
            %331 = ascvec.xor %dst1_51, %dst0_46, %211 : <ui32>, <ui32> -> <ui32>
            %332 = arith.addi %arg15, %c-1253254570_i32 : i32
            %333 = ascvec.duplicate_scalar %332, %211 : i32 -> <ui32>
            %334 = arith.addi %arg16, %c1684936478_i32 : i32
            %335 = ascvec.duplicate_scalar %334, %211 : i32 -> <ui32>
            %336 = ascvec.xor %330, %333, %211 : <ui32>, <ui32> -> <ui32>
            %337 = ascvec.xor %331, %335, %211 : <ui32>, <ui32> -> <ui32>
            %dst0_54, %dst1_55 = ascvec.mull %336, %282, %211 : <ui32>, <ui32> -> <ui32>, <ui32>
            %dst0_56, %dst1_57 = ascvec.mull %337, %283, %211 : <ui32>, <ui32> -> <ui32>, <ui32>
            %338 = ascvec.xor %dst1_57, %dst0_52, %211 : <ui32>, <ui32> -> <ui32>
            %339 = ascvec.xor %dst1_55, %dst0_50, %211 : <ui32>, <ui32> -> <ui32>
            %340 = arith.addi %arg15, %c1401181199_i32 : i32
            %341 = ascvec.duplicate_scalar %340, %211 : i32 -> <ui32>
            %342 = arith.addi %arg16, %c534103459_i32 : i32
            %343 = ascvec.duplicate_scalar %342, %211 : i32 -> <ui32>
            %344 = ascvec.xor %338, %341, %211 : <ui32>, <ui32> -> <ui32>
            %345 = ascvec.xor %339, %343, %211 : <ui32>, <ui32> -> <ui32>
            %dst0_58, %dst1_59 = ascvec.mull %344, %282, %211 : <ui32>, <ui32> -> <ui32>, <ui32>
            %dst0_60, %dst1_61 = ascvec.mull %345, %283, %211 : <ui32>, <ui32> -> <ui32>, <ui32>
            %346 = ascvec.xor %dst1_61, %dst0_56, %211 : <ui32>, <ui32> -> <ui32>
            %347 = ascvec.xor %dst1_59, %dst0_54, %211 : <ui32>, <ui32> -> <ui32>
            %348 = arith.addi %arg15, %c-239350328_i32 : i32
            %349 = ascvec.duplicate_scalar %348, %211 : i32 -> <ui32>
            %350 = arith.addi %arg16, %c-616729560_i32 : i32
            %351 = ascvec.duplicate_scalar %350, %211 : i32 -> <ui32>
            %352 = ascvec.xor %346, %349, %211 : <ui32>, <ui32> -> <ui32>
            %353 = ascvec.xor %347, %351, %211 : <ui32>, <ui32> -> <ui32>
            %dst0_62, %dst1_63 = ascvec.mull %352, %282, %211 : <ui32>, <ui32> -> <ui32>, <ui32>
            %dst0_64, %dst1_65 = ascvec.mull %353, %283, %211 : <ui32>, <ui32> -> <ui32>, <ui32>
            %354 = ascvec.xor %dst1_65, %dst0_60, %211 : <ui32>, <ui32> -> <ui32>
            %355 = ascvec.xor %dst1_63, %dst0_58, %211 : <ui32>, <ui32> -> <ui32>
            %356 = arith.addi %arg15, %c-1879881855_i32 : i32
            %357 = ascvec.duplicate_scalar %356, %211 : i32 -> <ui32>
            %358 = arith.addi %arg16, %c-1767562579_i32 : i32
            %359 = ascvec.duplicate_scalar %358, %211 : i32 -> <ui32>
            %360 = ascvec.xor %354, %357, %211 : <ui32>, <ui32> -> <ui32>
            %361 = ascvec.xor %355, %359, %211 : <ui32>, <ui32> -> <ui32>
            %362 = arith.muli %arg19, %c256_i32 : i32
            %363 = ascvec.reinterpret %360 : <ui32> -> <i32>
            %364 = arith.index_cast %362 : i32 to index
            ascvec.storealign %arg17[%364], %363, %211 : memref<4096xi32, 1>, index, <i32>
            %365 = arith.addi %362, %c64_i32_12 : i32
            %366 = ascvec.reinterpret %dst0_64 : <ui32> -> <i32>
            %367 = arith.index_cast %365 : i32 to index
            ascvec.storealign %arg17[%367], %366, %211 : memref<4096xi32, 1>, index, <i32>
            %368 = arith.addi %362, %c128_i32 : i32
            %369 = ascvec.reinterpret %361 : <ui32> -> <i32>
            %370 = arith.index_cast %368 : i32 to index
            ascvec.storealign %arg17[%370], %369, %211 : memref<4096xi32, 1>, index, <i32>
            %371 = arith.addi %362, %c192_i32 : i32
            %372 = ascvec.reinterpret %dst0_62 : <ui32> -> <i32>
            %373 = arith.index_cast %371 : i32 to index
            ascvec.storealign %arg17[%373], %372, %211 : memref<4096xi32, 1>, index, <i32>
          }
          ascvec.mem_bar vst_vld
        } {cannir.vf_mode = "simd"}
        %208 = arith.divsi %206, %c32_i32 : i32
        %209 = arith.divsi %206, %c64_i32 : i32
        ascvec.get_buf v, %47, 0 {cannir.buffer.generated_sync}
        ascvec.get_buf v, %83, 0 {cannir.buffer.generated_sync}
        ascvec.get_buf v, %88, 0 {cannir.buffer.generated_sync}
        %210 = ascscalar.bitcast_memref %87 byte_offset = %c0_i64 : memref<1024xui8, 1> -> memref<?xui16, 1>
        ascvec.vf(%arg4, %177, %78, %alloc_10, %alloc_11, %82, %208, %210, %209 : i64, i64, memref<16384xbf16, 1>, memref<512xui32, 1>, memref<4096xi32, 1>, memref<16384xui8, 1>, i32, memref<?xui16, 1>, i32) {
        ^bb0(%arg11: i64, %arg12: i64, %arg13: memref<16384xbf16, 1>, %arg14: memref<512xui32, 1>, %arg15: memref<4096xi32, 1>, %arg16: memref<16384xui8, 1>, %arg17: i32, %arg18: memref<?xui16, 1>, %arg19: i32):
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
          %c64_i32_12 = arith.constant 64 : i32
          %c2139160576_i32 = arith.constant 2139160576 : i32
          %c254_i32 = arith.constant 254 : i32
          %c255_i32 = arith.constant 255 : i32
          %c73400321_i32 = arith.constant 73400321 : i32
          %c-65011713_i32 = arith.constant -65011713 : i32
          %c2139095040_i32 = arith.constant 2139095040 : i32
          %c1_i32 = arith.constant 1 : i32
          %c2147483647_i32 = arith.constant 2147483647 : i32
          %c32_i32_13 = arith.constant 32 : i32
          %c0_i32_14 = arith.constant 0 : i32
          %211 = ascvec.create_mask : !ascvec.mask
          scf.for %arg20 = %c0_i32_14 to %arg17 step %c1_i32  : i32 {
            %212 = ascvec.duplicate_scalar %c0_i32_14, %211 : i32 -> <ui32>
            %213 = scf.for %arg21 = %c0_i32_14 to %c32_i32_13 step %c1_i32 iter_args(%arg22 = %212) -> (!ascvec.vreg<ui32>)  : i32 {
              %226 = arith.muli %arg20, %c32_i32_13 : i32
              %227 = arith.addi %226, %arg21 : i32
              %228 = arith.extsi %227 : i32 to i64
              %229 = arith.muli %228, %arg11 : i64
              %230 = arith.addi %229, %arg12 : i64
              %231 = arith.index_cast %230 : i64 to index
              %232 = ascvec.loadalign_unpack %arg13[%231] {mode = "b16_to_b32"} : memref<16384xbf16, 1>, index -> <bf16>
              %233 = ascvec.cast %232, %211 {rounding = 6 : i32} : <bf16> -> <f32>
              %234 = ascvec.reinterpret %233 : <f32> -> <ui32>
              %235 = ascvec.duplicate_scalar %c2147483647_i32, %211 : i32 -> <ui32>
              %236 = ascvec.and %234, %235, %211 : <ui32>, <ui32> -> <ui32>
              %237 = ascvec.max %arg22, %236, %211 : <ui32>
              scf.yield %237 : !ascvec.vreg<ui32>
            }
            %214 = ascvec.cmp_ge_scalar %213, %c2139095040_i32, %211 : <ui32>, i32 -> !ascvec.mask
            %215 = ascvec.add_scalar %213, %c-65011713_i32, %211 : <ui32>, i32
            %216 = ascvec.cmp_gt_scalar %213, %c73400321_i32, %211 : <ui32>, i32 -> !ascvec.mask
            %217 = ascvec.shr %215, 23, %216 : <ui32> -> <ui32>
            %218 = ascvec.duplicate_scalar %c255_i32, %214 merge %217 : !ascvec.vreg<ui32> {mode = "merging"} : i32 -> <ui32>
            %219 = ascvec.duplicate_scalar %c254_i32, %211 : i32 -> <ui32>
            %220 = ascvec.sub %219, %218, %211 : <ui32>
            %221 = ascvec.shl %220, 23, %211 : <ui32> -> <ui32>
            %222 = ascvec.duplicate_scalar %c2139160576_i32, %214 merge %221 : !ascvec.vreg<ui32> {mode = "merging"} : i32 -> <ui32>
            %223 = ascvec.reinterpret %222 : <ui32> -> <f32>
            %224 = arith.muli %arg20, %c64_i32_12 : i32
            %225 = arith.index_cast %224 : i32 to index
            ascvec.storealign %arg14[%225], %218, %211 : memref<512xui32, 1>, index, <ui32>
            scf.for %arg21 = %c0_i32_14 to %c8_i32 step %c1_i32  : i32 {
              %226 = arith.muli %arg20, %c512_i32 : i32
              %227 = arith.muli %arg21, %c64_i32_12 overflow<nsw> : i32
              %228 = arith.addi %226, %227 : i32
              %229 = arith.index_cast %228 : i32 to index
              %230 = ascvec.loadalign %arg15[%229] : memref<4096xi32, 1>, index -> <i32>
              %231 = ascvec.reinterpret %230 : <i32> -> <ui32>
              %232 = ascvec.shl %231, 8, %211 : <ui32> -> <ui32>
              %233 = ascvec.shr %231, 24, %211 : <ui32> -> <ui32>
              %234 = ascvec.or %232, %233, %211 : <ui32>, <ui32> -> <ui32>
              %235 = arith.muli %arg20, %c32_i32_13 : i32
              %236 = arith.muli %arg21, %c4_i32 overflow<nsw> : i32
              %237 = arith.addi %235, %236 : i32
              %238 = arith.extsi %237 : i32 to i64
              %239 = arith.muli %238, %arg11 : i64
              %240 = arith.addi %239, %arg12 : i64
              %241 = ascvec.shr %231, 16, %211 : <ui32> -> <ui32>
              %242 = arith.index_cast %240 : i64 to index
              %243 = ascvec.loadalign_unpack %arg13[%242] {mode = "b16_to_b32"} : memref<16384xbf16, 1>, index -> <bf16>
              %244 = ascvec.cast %243, %211 {rounding = 6 : i32} : <bf16> -> <f32>
              %245 = ascvec.mul %244, %223, %211 : <f32>
              %246 = ascvec.cast %245, %211 : <f32> -> <bf16>
              %247 = ascvec.cast %246, %211 {rounding = 6 : i32} : <bf16> -> <f32>
              %248 = ascvec.reinterpret %247 : <f32> -> <ui32>
              %249 = ascvec.shr %248, 24, %211 : <ui32> -> <ui32>
              %250 = ascvec.duplicate_scalar %c128_i32, %211 : i32 -> <ui32>
              %251 = ascvec.and %249, %250, %211 : <ui32>, <ui32> -> <ui32>
              %252 = ascvec.duplicate_scalar %c2147483647_i32, %211 : i32 -> <ui32>
              %253 = ascvec.and %248, %252, %211 : <ui32>, <ui32> -> <ui32>
              %254 = ascvec.shr %253, 4, %211 : <ui32> -> <ui32>
              %255 = ascvec.duplicate_scalar %c62914560_i32, %211 : i32 -> <ui32>
              %256 = ascvec.sub %254, %255, %211 : <ui32>
              %257 = ascvec.reinterpret %253 : <ui32> -> <f32>
              %258 = ascvec.mul_scalar %257, %cst, %211 : <f32>, f32
              %259 = ascvec.cast %258, %211 {rounding = 4 : i32} : <f32> -> <i32>
              %260 = ascvec.reinterpret %259 : <i32> -> <ui32>
              %261 = ascvec.cmp_ge_scalar %253, %c1015021568_i32, %211 : <ui32>, i32 -> !ascvec.mask
              %262 = ascvec.select %256, %260, %261 : <ui32>
              %263 = ascvec.duplicate_scalar %c65535_i32, %211 : i32 -> <ui32>
              %264 = ascvec.and %241, %263, %211 : <ui32>, <ui32> -> <ui32>
              %265 = ascvec.add %262, %264, %211 : <ui32>
              %266 = ascvec.shr %265, 16, %211 : <ui32> -> <ui32>
              %267 = ascvec.min_scalar %266, %c126_i32, %211 : <ui32>, i32
              %268 = ascvec.or %267, %251, %211 : <ui32>, <ui32> -> <ui32>
              %269 = ascvec.duplicate_scalar %c127_i32, %211 : i32 -> <ui32>
              %270 = ascvec.cmp_gt_scalar %253, %c2139095040_i32, %211 : <ui32>, i32 -> !ascvec.mask
              %271 = ascvec.select %269, %268, %270 : <ui32>
              ascvec.storealign_pack %arg16[%242], %271, %211 {mode = "b32_to_b8"} : memref<16384xui8, 1>, index, <ui32>
              %272 = arith.addi %237, %c1_i32 : i32
              %273 = arith.extsi %272 : i32 to i64
              %274 = arith.muli %273, %arg11 : i64
              %275 = arith.addi %274, %arg12 : i64
              %276 = ascvec.and %234, %263, %211 : <ui32>, <ui32> -> <ui32>
              %277 = arith.index_cast %275 : i64 to index
              %278 = ascvec.loadalign_unpack %arg13[%277] {mode = "b16_to_b32"} : memref<16384xbf16, 1>, index -> <bf16>
              %279 = ascvec.cast %278, %211 {rounding = 6 : i32} : <bf16> -> <f32>
              %280 = ascvec.mul %279, %223, %211 : <f32>
              %281 = ascvec.cast %280, %211 : <f32> -> <bf16>
              %282 = ascvec.cast %281, %211 {rounding = 6 : i32} : <bf16> -> <f32>
              %283 = ascvec.reinterpret %282 : <f32> -> <ui32>
              %284 = ascvec.shr %283, 24, %211 : <ui32> -> <ui32>
              %285 = ascvec.and %284, %250, %211 : <ui32>, <ui32> -> <ui32>
              %286 = ascvec.and %283, %252, %211 : <ui32>, <ui32> -> <ui32>
              %287 = ascvec.shr %286, 4, %211 : <ui32> -> <ui32>
              %288 = ascvec.sub %287, %255, %211 : <ui32>
              %289 = ascvec.reinterpret %286 : <ui32> -> <f32>
              %290 = ascvec.mul_scalar %289, %cst, %211 : <f32>, f32
              %291 = ascvec.cast %290, %211 {rounding = 4 : i32} : <f32> -> <i32>
              %292 = ascvec.reinterpret %291 : <i32> -> <ui32>
              %293 = ascvec.cmp_ge_scalar %286, %c1015021568_i32, %211 : <ui32>, i32 -> !ascvec.mask
              %294 = ascvec.select %288, %292, %293 : <ui32>
              %295 = ascvec.and %276, %263, %211 : <ui32>, <ui32> -> <ui32>
              %296 = ascvec.add %294, %295, %211 : <ui32>
              %297 = ascvec.shr %296, 16, %211 : <ui32> -> <ui32>
              %298 = ascvec.min_scalar %297, %c126_i32, %211 : <ui32>, i32
              %299 = ascvec.or %298, %285, %211 : <ui32>, <ui32> -> <ui32>
              %300 = ascvec.cmp_gt_scalar %286, %c2139095040_i32, %211 : <ui32>, i32 -> !ascvec.mask
              %301 = ascvec.select %269, %299, %300 : <ui32>
              ascvec.storealign_pack %arg16[%277], %301, %211 {mode = "b32_to_b8"} : memref<16384xui8, 1>, index, <ui32>
              %302 = arith.addi %237, %c2_i32 : i32
              %303 = arith.extsi %302 : i32 to i64
              %304 = arith.muli %303, %arg11 : i64
              %305 = arith.addi %304, %arg12 : i64
              %306 = ascvec.and %231, %263, %211 : <ui32>, <ui32> -> <ui32>
              %307 = arith.index_cast %305 : i64 to index
              %308 = ascvec.loadalign_unpack %arg13[%307] {mode = "b16_to_b32"} : memref<16384xbf16, 1>, index -> <bf16>
              %309 = ascvec.cast %308, %211 {rounding = 6 : i32} : <bf16> -> <f32>
              %310 = ascvec.mul %309, %223, %211 : <f32>
              %311 = ascvec.cast %310, %211 : <f32> -> <bf16>
              %312 = ascvec.cast %311, %211 {rounding = 6 : i32} : <bf16> -> <f32>
              %313 = ascvec.reinterpret %312 : <f32> -> <ui32>
              %314 = ascvec.shr %313, 24, %211 : <ui32> -> <ui32>
              %315 = ascvec.and %314, %250, %211 : <ui32>, <ui32> -> <ui32>
              %316 = ascvec.and %313, %252, %211 : <ui32>, <ui32> -> <ui32>
              %317 = ascvec.shr %316, 4, %211 : <ui32> -> <ui32>
              %318 = ascvec.sub %317, %255, %211 : <ui32>
              %319 = ascvec.reinterpret %316 : <ui32> -> <f32>
              %320 = ascvec.mul_scalar %319, %cst, %211 : <f32>, f32
              %321 = ascvec.cast %320, %211 {rounding = 4 : i32} : <f32> -> <i32>
              %322 = ascvec.reinterpret %321 : <i32> -> <ui32>
              %323 = ascvec.cmp_ge_scalar %316, %c1015021568_i32, %211 : <ui32>, i32 -> !ascvec.mask
              %324 = ascvec.select %318, %322, %323 : <ui32>
              %325 = ascvec.and %306, %263, %211 : <ui32>, <ui32> -> <ui32>
              %326 = ascvec.add %324, %325, %211 : <ui32>
              %327 = ascvec.shr %326, 16, %211 : <ui32> -> <ui32>
              %328 = ascvec.min_scalar %327, %c126_i32, %211 : <ui32>, i32
              %329 = ascvec.or %328, %315, %211 : <ui32>, <ui32> -> <ui32>
              %330 = ascvec.cmp_gt_scalar %316, %c2139095040_i32, %211 : <ui32>, i32 -> !ascvec.mask
              %331 = ascvec.select %269, %329, %330 : <ui32>
              ascvec.storealign_pack %arg16[%307], %331, %211 {mode = "b32_to_b8"} : memref<16384xui8, 1>, index, <ui32>
              %332 = arith.addi %237, %c3_i32 : i32
              %333 = arith.extsi %332 : i32 to i64
              %334 = arith.muli %333, %arg11 : i64
              %335 = arith.addi %334, %arg12 : i64
              %336 = ascvec.shr %234, 16, %211 : <ui32> -> <ui32>
              %337 = arith.index_cast %335 : i64 to index
              %338 = ascvec.loadalign_unpack %arg13[%337] {mode = "b16_to_b32"} : memref<16384xbf16, 1>, index -> <bf16>
              %339 = ascvec.cast %338, %211 {rounding = 6 : i32} : <bf16> -> <f32>
              %340 = ascvec.mul %339, %223, %211 : <f32>
              %341 = ascvec.cast %340, %211 : <f32> -> <bf16>
              %342 = ascvec.cast %341, %211 {rounding = 6 : i32} : <bf16> -> <f32>
              %343 = ascvec.reinterpret %342 : <f32> -> <ui32>
              %344 = ascvec.shr %343, 24, %211 : <ui32> -> <ui32>
              %345 = ascvec.and %344, %250, %211 : <ui32>, <ui32> -> <ui32>
              %346 = ascvec.and %343, %252, %211 : <ui32>, <ui32> -> <ui32>
              %347 = ascvec.shr %346, 4, %211 : <ui32> -> <ui32>
              %348 = ascvec.sub %347, %255, %211 : <ui32>
              %349 = ascvec.reinterpret %346 : <ui32> -> <f32>
              %350 = ascvec.mul_scalar %349, %cst, %211 : <f32>, f32
              %351 = ascvec.cast %350, %211 {rounding = 4 : i32} : <f32> -> <i32>
              %352 = ascvec.reinterpret %351 : <i32> -> <ui32>
              %353 = ascvec.cmp_ge_scalar %346, %c1015021568_i32, %211 : <ui32>, i32 -> !ascvec.mask
              %354 = ascvec.select %348, %352, %353 : <ui32>
              %355 = ascvec.and %336, %263, %211 : <ui32>, <ui32> -> <ui32>
              %356 = ascvec.add %354, %355, %211 : <ui32>
              %357 = ascvec.shr %356, 16, %211 : <ui32> -> <ui32>
              %358 = ascvec.min_scalar %357, %c126_i32, %211 : <ui32>, i32
              %359 = ascvec.or %358, %345, %211 : <ui32>, <ui32> -> <ui32>
              %360 = ascvec.cmp_gt_scalar %346, %c2139095040_i32, %211 : <ui32>, i32 -> !ascvec.mask
              %361 = ascvec.select %269, %359, %360 : <ui32>
              ascvec.storealign_pack %arg16[%337], %361, %211 {mode = "b32_to_b8"} : memref<16384xui8, 1>, index, <ui32>
            }
          }
          ascvec.mem_bar vst_vld
          %mask, %remaining_out = ascvec.update_mask %c64_i32_12 elem_bits 16 : !ascvec.mask
          scf.for %arg20 = %c0_i32_14 to %arg19 step %c1_i32  : i32 {
            %212 = arith.muli %arg20, %c128_i32 : i32
            %213 = arith.index_cast %212 : i32 to index
            %214 = ascvec.loadalign %arg14[%213] : memref<512xui32, 1>, index -> <ui32>
            %215 = arith.addi %212, %c64_i32_12 : i32
            %216 = arith.index_cast %215 : i32 to index
            %217 = ascvec.loadalign %arg14[%216] : memref<512xui32, 1>, index -> <ui32>
            %218 = ascvec.shl %217, 8, %211 : <ui32> -> <ui32>
            %219 = ascvec.or %214, %218, %211 : <ui32>, <ui32> -> <ui32>
            %220 = ascvec.pack %219 src_elem_bits 32 dst_elem_bits 16 {part = "low"} : <ui32> -> <ui16>
            %221 = arith.extsi %arg20 : i32 to i64
            %222 = arith.muli %221, %arg11 : i64
            %223 = arith.addi %222, %arg12 : i64
            %224 = arith.index_cast %223 : i64 to index
            ascvec.storealign %arg18[%224], %220, %mask : memref<?xui16, 1>, index, <ui16>
          }
          ascvec.mem_bar vst_vld
        } {cannir.vf_mode = "simd"}
        ascvec.release_buf v, %88, 0 {cannir.buffer.generated_sync}
        ascvec.release_buf v, %83, 0 {cannir.buffer.generated_sync}
        ascvec.release_buf v, %47, 0 {cannir.buffer.generated_sync}
      }
      %96 = cannir.extract_layout %arg1 : !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
      %97 = cannir.extract_shape %96 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, -1)>
      %98 = cannir.extract_dim %97[0] : !cannir.shape<(-1, -1)>
      %99 = arith.minsi %38, %98 : i64
      %100 = arith.subi %98, %99 : i64
      %101 = arith.minsi %100, %c1_i64 : i64
      %102 = arith.cmpi sle, %38, %98 : i64
      %103 = arith.select %102, %101, %c0_i64 : i64
      %104 = arith.cmpi sgt, %103, %c1_i64 : i64
      %105 = arith.select %104, %c1_i64, %103 : i64
      %106 = cannir.extract_dim %97[1] : !cannir.shape<(-1, -1)>
      %107 = arith.divsi %106, %60 : i64
      %108 = arith.minsi %40, %107 : i64
      %109 = arith.muli %108, %41 : i64
      %110 = arith.subi %106, %109 : i64
      %111 = arith.minsi %41, %110 : i64
      %112 = arith.cmpi sle, %40, %107 : i64
      %113 = arith.select %112, %111, %c0_i64 : i64
      %114 = arith.cmpi sgt, %113, %c16384_i64 : i64
      %115 = arith.select %114, %c16384_i64, %113 : i64
      %116 = cannir.extract_physical_layout %arg1 : !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
      %117 = cannir.extract_stride %116 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
      %118 = cannir.extract_dim %117[0] : !cannir.stride<(-1, 1)>
      %119 = arith.muli %38, %118 : i64
      %120 = arith.addi %119, %70 : i64
      %121 = arith.muli %120, %c8_i64 : i64
      %122 = ascscalar.memref_address %alloc_0 : memref<1xui8>
      %123 = arith.divui %121, %c8_i64 : i64
      %124 = arith.addi %122, %123 : i64
      %125 = ascscalar.bind_gm_ptr %124 : memref<1xui8>
      %126 = ascscalar.bind_local_ptr %81 : memref<1xui8, 1>
      %127 = cannir.extract_stride %96 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
      %128 = cannir.extract_dim %127[0] : !cannir.stride<(-1, 1)>
      %129 = arith.cmpi ugt, %105, %c0_i64 : i64
      %130 = arith.cmpi ugt, %115, %c0_i64 : i64
      %131 = arith.andi %129, %130 : i1
      scf.if %131 {
        ascvec.get_buf mte3, %83, 0 {cannir.buffer.generated_sync}
        %177 = arith.trunci %105 : i64 to i32
        %178 = arith.trunci %115 : i64 to i32
        ascvec.copy_ub2gm %125[%c0], %126[%c0], %177, %178, %128, %c16384_i32 l2_cache_ctl %c0_i32 : memref<1xui8>, memref<1xui8, 1>
        ascvec.release_buf mte3, %83, 0 {cannir.buffer.generated_sync}
      }
      %132 = arith.remsi %41, %c32_i64 : i64
      %133 = arith.cmpi ne, %132, %c0_i64 : i64
      %134 = arith.cmpi slt, %132, %c0_i64 : i64
      %135 = arith.andi %133, %134 : i1
      %136 = arith.divsi %41, %c32_i64 : i64
      %137 = arith.extui %135 : i1 to i64
      %138 = arith.subi %136, %137 : i64
      %139 = cannir.extract_layout %arg2 : !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
      %140 = cannir.extract_shape %139 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, -1)>
      %141 = cannir.extract_dim %140[0] : !cannir.shape<(-1, -1)>
      %142 = arith.minsi %38, %141 : i64
      %143 = arith.subi %141, %142 : i64
      %144 = arith.minsi %143, %c1_i64 : i64
      %145 = arith.cmpi sle, %38, %141 : i64
      %146 = arith.select %145, %144, %c0_i64 : i64
      %147 = arith.cmpi sgt, %146, %c1_i64 : i64
      %148 = arith.select %147, %c1_i64, %146 : i64
      %149 = cannir.extract_dim %140[1] : !cannir.shape<(-1, -1)>
      %150 = arith.maxsi %138, %c1_i64 : i64
      %151 = arith.divsi %149, %150 : i64
      %152 = arith.minsi %40, %151 : i64
      %153 = arith.muli %152, %138 : i64
      %154 = arith.subi %149, %153 : i64
      %155 = arith.minsi %138, %154 : i64
      %156 = arith.cmpi sle, %40, %151 : i64
      %157 = arith.select %156, %155, %c0_i64 : i64
      %158 = arith.cmpi sgt, %157, %c1024_i64 : i64
      %159 = arith.select %158, %c1024_i64, %157 : i64
      %160 = cannir.extract_physical_layout %arg2 : !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
      %161 = cannir.extract_stride %160 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
      %162 = cannir.extract_dim %161[0] : !cannir.stride<(-1, 1)>
      %163 = arith.muli %38, %162 : i64
      %164 = arith.muli %40, %138 : i64
      %165 = arith.addi %163, %164 : i64
      %166 = arith.muli %165, %c8_i64 : i64
      %167 = ascscalar.memref_address %alloc_1 : memref<1xui8>
      %168 = arith.divui %166, %c8_i64 : i64
      %169 = arith.addi %167, %168 : i64
      %170 = ascscalar.bind_gm_ptr %169 : memref<1xui8>
      %171 = ascscalar.bind_local_ptr %86 : memref<1xui8, 1>
      %172 = cannir.extract_stride %139 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
      %173 = cannir.extract_dim %172[0] : !cannir.stride<(-1, 1)>
      %174 = arith.cmpi ugt, %148, %c0_i64 : i64
      %175 = arith.cmpi ugt, %159, %c0_i64 : i64
      %176 = arith.andi %174, %175 : i1
      scf.if %176 {
        ascvec.get_buf mte3, %88, 0 {cannir.buffer.generated_sync}
        %177 = arith.trunci %148 : i64 to i32
        %178 = arith.trunci %159 : i64 to i32
        ascvec.copy_ub2gm %170[%c0], %171[%c0], %177, %178, %173, %c1024_i32 l2_cache_ctl %c0_i32 : memref<1xui8>, memref<1xui8, 1>
        ascvec.release_buf mte3, %88, 0 {cannir.buffer.generated_sync}
      }
      scf.yield %49 : i64
    }
    return
  }
}
