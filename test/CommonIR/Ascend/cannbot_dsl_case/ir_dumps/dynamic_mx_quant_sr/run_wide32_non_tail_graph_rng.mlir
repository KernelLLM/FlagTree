module {
  func.func @run_wide32_non_tail_graph_rng_95560f874e2c99ddbec8d1d9dd29f65a(%arg0: !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg1: !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg2: !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: !cannir.tensor<<gm, i64>, <<(1)>, <(1)>>, <<(1)>, <(1)>>, identity>, %arg7: !cannir.tensor<<gm, i64>, <<(1)>, <(1)>>, <<(1)>, <(1)>>, identity>, %arg8: i64) attributes {cannbotdsl.contract_digest = "95560f874e2c99ddbec8d1d9dd29f65a94fb0a029492fd2f051a0e160453a868", cannbotdsl.public_slot_count = 9 : i64, cannbotdsl.slot_dtype_bits = [16, 8, 8, 64, 64, 64, 64, 64, 64], cannbotdsl.slot_dtype_codes = [4, 1, 1, 0, 0, 0, 0, 0, 0], cannbotdsl.slot_dtype_lanes = [1, 1, 1, 1, 1, 1, 1, 1, 1], cannbotdsl.slot_flags = [1, 1, 1, 1, 1, 1, 1, 1, 1], cannbotdsl.slot_kinds = [1, 1, 1, 2, 2, 2, 1, 1, 2], cannbotdsl.slot_type_digests = ["5d7cdc96dc21b3b739e32aeccf58c1df113bec54701c75107ed650f2544a07da", "171ddd09eb070fb4a6413ae87b2ceecddc233a9a173a0404e96285a4bb8b8659", "891bcd21e721f8d1a666bcca4e8705ec63796c1add8ae000536d0d98f0274d3a", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "a9d876affafdf57be3f0e1a45cb0b293a48851dbda8b74101f5184f5130b1b63", "a9d876affafdf57be3f0e1a45cb0b293a48851dbda8b74101f5184f5130b1b63", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515"], cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.host} {
    cannir.kernel_launch @kernel__DynamicMxQuantSrKernel_run_wide32_non_tail_0(%arg5) [%arg0, %arg1, %arg2, %arg3, %arg4, %arg6, %arg7, %arg8] {cannbotdsl.profile = {indices = array<i64: 0, 1, 2, 5, 6>, name = "_DynamicMxQuantSrKernel_run_wide32_non_tail", op_type = "_DynamicMxQuantSrKernel_run_wide32_non_tail", roles = array<i64: 0, 1, 1, 0, 0>}} : !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, i64, i64, !cannir.tensor<<gm, i64>, <<(1)>, <(1)>>, <<(1)>, <(1)>>, identity>, !cannir.tensor<<gm, i64>, <<(1)>, <(1)>>, <<(1)>, <(1)>>, identity>, i64
    return
  }
  func.func @kernel__DynamicMxQuantSrKernel_run_wide32_non_tail_0(%arg0: !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg1: !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg2: !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg3: i64, %arg4: i64, %arg5: !cannir.tensor<<gm, i64>, <<(1)>, <(1)>>, <<(1)>, <(1)>>, identity>, %arg6: !cannir.tensor<<gm, i64>, <<(1)>, <(1)>>, <<(1)>, <(1)>>, identity>, %arg7: i64) attributes {cannir.buffer.auto_sync, cannir.buffer.sync_analyzed, cannir.buffer.sync_verified, cannir.buffer.tensor_state, cannir.core = "aiv", cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.device, cannir.explicit_ring_buffer_cursors, cannir.kernel_group = "kernel__DynamicMxQuantSrKernel_run_wide32_non_tail_0", cannir.resource.ids_prepared} {
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
      %30 = arith.remsi %arg3, %c512_i64 : i64
      %31 = arith.cmpi ne, %30, %c0_i64 : i64
      %32 = arith.cmpi slt, %30, %c0_i64 : i64
      %33 = arith.andi %31, %32 : i1
      %34 = arith.select %33, %c512_i64, %c0_i64 : i64
      %35 = arith.addi %30, %34 overflow<nsw> : i64
      %36 = arith.cmpi eq, %35, %c0_i64 : i64
      %37 = arith.cmpi eq, %arg4, %c32_i64 : i64
      %38 = arith.select %36, %37, %false : i1
      scf.yield %38 : i1
    } else {
      scf.yield %false : i1
    }
    %18 = scf.if %17 -> (i64) {
      scf.yield %c512_i64 : i64
    } else {
      %30 = scf.if %16 -> (i1) {
        %32 = arith.remsi %arg3, %c256_i64 : i64
        %33 = arith.cmpi ne, %32, %c0_i64 : i64
        %34 = arith.cmpi slt, %32, %c0_i64 : i64
        %35 = arith.andi %33, %34 : i1
        %36 = arith.select %35, %c256_i64, %c0_i64 : i64
        %37 = arith.addi %32, %36 overflow<nsw> : i64
        %38 = arith.cmpi eq, %37, %c0_i64 : i64
        %39 = arith.cmpi eq, %arg4, %c32_i64 : i64
        %40 = arith.select %38, %39, %false : i1
        scf.yield %40 : i1
      } else {
        scf.yield %false : i1
      }
      %31 = scf.if %30 -> (i64) {
        scf.yield %c256_i64 : i64
      } else {
        %32 = arith.remsi %arg3, %c128_i64 : i64
        %33 = arith.cmpi ne, %32, %c0_i64 : i64
        %34 = arith.cmpi slt, %32, %c0_i64 : i64
        %35 = arith.andi %33, %34 : i1
        %36 = arith.select %35, %c128_i64, %c0_i64 : i64
        %37 = arith.addi %32, %36 overflow<nsw> : i64
        %38 = arith.cmpi eq, %37, %c0_i64 : i64
        %39 = arith.select %16, %38, %false : i1
        %40 = arith.select %39, %c128_i64, %c64_i64 : i64
        scf.yield %40 : i64
      }
      scf.yield %31 : i64
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
      %89 = arith.muli %38, %arg4 : i64
      %90 = arith.muli %40, %18 : i64
      %91 = arith.remsi %90, %c16_i64 : i64
      %92 = arith.cmpi ne, %91, %c0_i64 : i64
      %93 = arith.cmpi slt, %91, %c0_i64 : i64
      %94 = arith.andi %92, %93 : i1
      %95 = arith.divsi %90, %c16_i64 : i64
      %96 = arith.extui %94 : i1 to i64
      %97 = arith.subi %95, %96 : i64
      %98 = arith.remsi %arg3, %c16_i64 : i64
      %99 = arith.cmpi ne, %98, %c0_i64 : i64
      %100 = arith.cmpi slt, %98, %c0_i64 : i64
      %101 = arith.andi %99, %100 : i1
      %102 = arith.divsi %arg3, %c16_i64 : i64
      %103 = arith.extui %101 : i1 to i64
      %104 = arith.subi %102, %103 : i64
      %105 = arith.trunci %1 : i64 to i32
      %106 = arith.extui %105 : i32 to i64
      %107 = arith.subi %1, %106 : i64
      %108 = arith.remsi %107, %c4294967296_i64 : i64
      %109 = arith.cmpi ne, %108, %c0_i64 : i64
      %110 = arith.cmpi slt, %108, %c0_i64 : i64
      %111 = arith.andi %109, %110 : i1
      %112 = arith.divsi %107, %c4294967296_i64 : i64
      %113 = arith.extui %111 : i1 to i64
      %114 = arith.subi %112, %113 : i64
      %115 = arith.trunci %114 : i64 to i32
      %116 = arith.trunci %18 : i64 to i32
      %117 = arith.divsi %116, %c16_i32 : i32
      ascvec.vf(%97, %89, %104, %4, %105, %115, %alloc_11, %117 : i64, i64, i64, i64, i32, i32, memref<4096xi32, 1>, i32) {
      ^bb0(%arg10: i64, %arg11: i64, %arg12: i64, %arg13: i64, %arg14: i32, %arg15: i32, %arg16: memref<4096xi32, 1>, %arg17: i32):
        %c128_i32 = arith.constant 128 : i32
        %c1_i32 = arith.constant 1 : i32
        %c96_i32 = arith.constant 96 : i32
        %c64_i32_12 = arith.constant 64 : i32
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
        %c32_i32_16 = arith.constant 32 : i32
        %mask, %remaining_out = ascvec.update_mask %c32_i32_16 : !ascvec.mask
        %202 = ascvec.arange %c0_i32_15 : i32 -> <ui32>
        scf.for %arg18 = %c0_i32_15 to %arg17 step %c1_i32  : i32 {
          %203 = arith.extsi %arg18 : i32 to i64
          %204 = arith.addi %arg10, %203 : i64
          %205 = arith.trunci %arg11 : i64 to i32
          %206 = arith.extui %205 : i32 to i64
          %207 = arith.subi %arg11, %206 : i64
          %208 = arith.remsi %207, %c4294967296_i64_14 : i64
          %209 = arith.cmpi ne, %208, %c0_i64_13 : i64
          %210 = arith.cmpi slt, %208, %c0_i64_13 : i64
          %211 = arith.andi %209, %210 : i1
          %212 = arith.divsi %207, %c4294967296_i64_14 : i64
          %213 = arith.extui %211 : i1 to i64
          %214 = arith.subi %212, %213 : i64
          %215 = arith.trunci %214 : i64 to i32
          %216 = arith.trunci %204 : i64 to i32
          %217 = arith.extui %216 : i32 to i64
          %218 = arith.subi %204, %217 : i64
          %219 = arith.remsi %218, %c4294967296_i64_14 : i64
          %220 = arith.cmpi ne, %219, %c0_i64_13 : i64
          %221 = arith.cmpi slt, %219, %c0_i64_13 : i64
          %222 = arith.andi %220, %221 : i1
          %223 = arith.divsi %218, %c4294967296_i64_14 : i64
          %224 = arith.extui %222 : i1 to i64
          %225 = arith.subi %223, %224 : i64
          %226 = arith.trunci %225 : i64 to i32
          %227 = ascvec.duplicate_scalar %205, %mask : i32 -> <ui32>
          %228 = ascvec.duplicate_scalar %215, %mask : i32 -> <ui32>
          %229 = ascvec.duplicate_scalar %c0_i32_15, %mask : i32 -> <ui32>
          %carry, %dst = ascvec.addc %227, %202, %mask : <ui32>, <ui32> -> !ascvec.mask, <ui32>
          %carry_17, %dst_18 = ascvec.addc %228, %229, %carry, %mask : <ui32>, <ui32> -> !ascvec.mask, <ui32>
          %230 = ascvec.duplicate_scalar %216, %mask : i32 -> <ui32>
          %231 = ascvec.duplicate_scalar %226, %mask : i32 -> <ui32>
          %232 = arith.trunci %arg12 : i64 to i32
          %233 = arith.extui %232 : i32 to i64
          %234 = arith.subi %arg12, %233 : i64
          %235 = arith.remsi %234, %c4294967296_i64_14 : i64
          %236 = arith.cmpi ne, %235, %c0_i64_13 : i64
          %237 = arith.cmpi slt, %235, %c0_i64_13 : i64
          %238 = arith.andi %236, %237 : i1
          %239 = arith.divsi %234, %c4294967296_i64_14 : i64
          %240 = arith.extui %238 : i1 to i64
          %241 = arith.subi %239, %240 : i64
          %242 = arith.trunci %241 : i64 to i32
          %243 = ascvec.duplicate_scalar %232, %mask : i32 -> <ui32>
          %dst0, %dst1 = ascvec.mull %dst, %243, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %244 = ascvec.mul_scalar %dst, %242, %mask : <ui32>, i32
          %245 = ascvec.add %dst1, %244, %mask : <ui32>
          %246 = ascvec.mul_scalar %dst_18, %232, %mask : <ui32>, i32
          %247 = ascvec.add %245, %246, %mask : <ui32>
          %carry_19, %dst_20 = ascvec.addc %dst0, %230, %mask : <ui32>, <ui32> -> !ascvec.mask, <ui32>
          %carry_21, %dst_22 = ascvec.addc %247, %231, %carry_19, %mask : <ui32>, <ui32> -> !ascvec.mask, <ui32>
          %248 = arith.trunci %arg13 : i64 to i32
          %249 = arith.extui %248 : i32 to i64
          %250 = arith.subi %arg13, %249 : i64
          %251 = arith.remsi %250, %c4294967296_i64_14 : i64
          %252 = arith.cmpi ne, %251, %c0_i64_13 : i64
          %253 = arith.cmpi slt, %251, %c0_i64_13 : i64
          %254 = arith.andi %252, %253 : i1
          %255 = arith.divsi %250, %c4294967296_i64_14 : i64
          %256 = arith.extui %254 : i1 to i64
          %257 = arith.subi %255, %256 : i64
          %258 = arith.trunci %257 : i64 to i32
          %259 = arith.divui %248, %c4_i32 : i32
          %260 = arith.remui %258, %c4_i32 : i32
          %261 = arith.muli %260, %c1073741824_i32 : i32
          %262 = arith.addi %259, %261 : i32
          %263 = arith.divui %258, %c4_i32 : i32
          %264 = ascvec.shr %dst_20, 14, %mask : <ui32> -> <ui32>
          %265 = ascvec.shl %dst_22, 18, %mask : <ui32> -> <ui32>
          %266 = ascvec.or %264, %265, %mask : <ui32>, <ui32> -> <ui32>
          %267 = ascvec.shr %dst_22, 14, %mask : <ui32> -> <ui32>
          %268 = ascvec.duplicate_scalar %262, %mask : i32 -> <ui32>
          %269 = ascvec.duplicate_scalar %263, %mask : i32 -> <ui32>
          %carry_23, %dst_24 = ascvec.addc %268, %266, %mask : <ui32>, <ui32> -> !ascvec.mask, <ui32>
          %carry_25, %dst_26 = ascvec.addc %269, %267, %carry_23, %mask : <ui32>, <ui32> -> !ascvec.mask, <ui32>
          %270 = ascvec.duplicate_scalar %c16383_i32, %mask : i32 -> <ui32>
          %271 = ascvec.and %dst_20, %270, %mask : <ui32>, <ui32> -> <ui32>
          %272 = ascvec.duplicate_scalar %c-766435501_i32, %mask : i32 -> <ui32>
          %273 = ascvec.duplicate_scalar %c-845247145_i32, %mask : i32 -> <ui32>
          %dst0_27, %dst1_28 = ascvec.mull %229, %272, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %dst0_29, %dst1_30 = ascvec.mull %dst_24, %273, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %274 = ascvec.xor %dst1_30, %271, %mask : <ui32>, <ui32> -> <ui32>
          %275 = ascvec.xor %dst1_28, %dst_26, %mask : <ui32>, <ui32> -> <ui32>
          %276 = ascvec.duplicate_scalar %arg14, %mask : i32 -> <ui32>
          %277 = ascvec.duplicate_scalar %arg15, %mask : i32 -> <ui32>
          %278 = ascvec.xor %274, %276, %mask : <ui32>, <ui32> -> <ui32>
          %279 = ascvec.xor %275, %277, %mask : <ui32>, <ui32> -> <ui32>
          %dst0_31, %dst1_32 = ascvec.mull %278, %272, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %dst0_33, %dst1_34 = ascvec.mull %279, %273, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %280 = ascvec.xor %dst1_34, %dst0_29, %mask : <ui32>, <ui32> -> <ui32>
          %281 = ascvec.xor %dst1_32, %dst0_27, %mask : <ui32>, <ui32> -> <ui32>
          %282 = arith.addi %arg14, %c-1640531527_i32 : i32
          %283 = ascvec.duplicate_scalar %282, %mask : i32 -> <ui32>
          %284 = arith.addi %arg15, %c-1150833019_i32 : i32
          %285 = ascvec.duplicate_scalar %284, %mask : i32 -> <ui32>
          %286 = ascvec.xor %280, %283, %mask : <ui32>, <ui32> -> <ui32>
          %287 = ascvec.xor %281, %285, %mask : <ui32>, <ui32> -> <ui32>
          %dst0_35, %dst1_36 = ascvec.mull %286, %272, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %dst0_37, %dst1_38 = ascvec.mull %287, %273, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %288 = ascvec.xor %dst1_38, %dst0_33, %mask : <ui32>, <ui32> -> <ui32>
          %289 = ascvec.xor %dst1_36, %dst0_31, %mask : <ui32>, <ui32> -> <ui32>
          %290 = arith.addi %arg14, %c1013904242_i32 : i32
          %291 = ascvec.duplicate_scalar %290, %mask : i32 -> <ui32>
          %292 = arith.addi %arg15, %c1993301258_i32 : i32
          %293 = ascvec.duplicate_scalar %292, %mask : i32 -> <ui32>
          %294 = ascvec.xor %288, %291, %mask : <ui32>, <ui32> -> <ui32>
          %295 = ascvec.xor %289, %293, %mask : <ui32>, <ui32> -> <ui32>
          %dst0_39, %dst1_40 = ascvec.mull %294, %272, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %dst0_41, %dst1_42 = ascvec.mull %295, %273, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %296 = ascvec.xor %dst1_42, %dst0_37, %mask : <ui32>, <ui32> -> <ui32>
          %297 = ascvec.xor %dst1_40, %dst0_35, %mask : <ui32>, <ui32> -> <ui32>
          %298 = arith.addi %arg14, %c-626627285_i32 : i32
          %299 = ascvec.duplicate_scalar %298, %mask : i32 -> <ui32>
          %300 = arith.addi %arg15, %c842468239_i32 : i32
          %301 = ascvec.duplicate_scalar %300, %mask : i32 -> <ui32>
          %302 = ascvec.xor %296, %299, %mask : <ui32>, <ui32> -> <ui32>
          %303 = ascvec.xor %297, %301, %mask : <ui32>, <ui32> -> <ui32>
          %dst0_43, %dst1_44 = ascvec.mull %302, %272, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %dst0_45, %dst1_46 = ascvec.mull %303, %273, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %304 = ascvec.xor %dst1_46, %dst0_41, %mask : <ui32>, <ui32> -> <ui32>
          %305 = ascvec.xor %dst1_44, %dst0_39, %mask : <ui32>, <ui32> -> <ui32>
          %306 = arith.addi %arg14, %c2027808484_i32 : i32
          %307 = ascvec.duplicate_scalar %306, %mask : i32 -> <ui32>
          %308 = arith.addi %arg15, %c-308364780_i32 : i32
          %309 = ascvec.duplicate_scalar %308, %mask : i32 -> <ui32>
          %310 = ascvec.xor %304, %307, %mask : <ui32>, <ui32> -> <ui32>
          %311 = ascvec.xor %305, %309, %mask : <ui32>, <ui32> -> <ui32>
          %dst0_47, %dst1_48 = ascvec.mull %310, %272, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %dst0_49, %dst1_50 = ascvec.mull %311, %273, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %312 = ascvec.xor %dst1_50, %dst0_45, %mask : <ui32>, <ui32> -> <ui32>
          %313 = ascvec.xor %dst1_48, %dst0_43, %mask : <ui32>, <ui32> -> <ui32>
          %314 = arith.addi %arg14, %c387276957_i32 : i32
          %315 = ascvec.duplicate_scalar %314, %mask : i32 -> <ui32>
          %316 = arith.addi %arg15, %c-1459197799_i32 : i32
          %317 = ascvec.duplicate_scalar %316, %mask : i32 -> <ui32>
          %318 = ascvec.xor %312, %315, %mask : <ui32>, <ui32> -> <ui32>
          %319 = ascvec.xor %313, %317, %mask : <ui32>, <ui32> -> <ui32>
          %dst0_51, %dst1_52 = ascvec.mull %318, %272, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %dst0_53, %dst1_54 = ascvec.mull %319, %273, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %320 = ascvec.xor %dst1_54, %dst0_49, %mask : <ui32>, <ui32> -> <ui32>
          %321 = ascvec.xor %dst1_52, %dst0_47, %mask : <ui32>, <ui32> -> <ui32>
          %322 = arith.addi %arg14, %c-1253254570_i32 : i32
          %323 = ascvec.duplicate_scalar %322, %mask : i32 -> <ui32>
          %324 = arith.addi %arg15, %c1684936478_i32 : i32
          %325 = ascvec.duplicate_scalar %324, %mask : i32 -> <ui32>
          %326 = ascvec.xor %320, %323, %mask : <ui32>, <ui32> -> <ui32>
          %327 = ascvec.xor %321, %325, %mask : <ui32>, <ui32> -> <ui32>
          %dst0_55, %dst1_56 = ascvec.mull %326, %272, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %dst0_57, %dst1_58 = ascvec.mull %327, %273, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %328 = ascvec.xor %dst1_58, %dst0_53, %mask : <ui32>, <ui32> -> <ui32>
          %329 = ascvec.xor %dst1_56, %dst0_51, %mask : <ui32>, <ui32> -> <ui32>
          %330 = arith.addi %arg14, %c1401181199_i32 : i32
          %331 = ascvec.duplicate_scalar %330, %mask : i32 -> <ui32>
          %332 = arith.addi %arg15, %c534103459_i32 : i32
          %333 = ascvec.duplicate_scalar %332, %mask : i32 -> <ui32>
          %334 = ascvec.xor %328, %331, %mask : <ui32>, <ui32> -> <ui32>
          %335 = ascvec.xor %329, %333, %mask : <ui32>, <ui32> -> <ui32>
          %dst0_59, %dst1_60 = ascvec.mull %334, %272, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %dst0_61, %dst1_62 = ascvec.mull %335, %273, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %336 = ascvec.xor %dst1_62, %dst0_57, %mask : <ui32>, <ui32> -> <ui32>
          %337 = ascvec.xor %dst1_60, %dst0_55, %mask : <ui32>, <ui32> -> <ui32>
          %338 = arith.addi %arg14, %c-239350328_i32 : i32
          %339 = ascvec.duplicate_scalar %338, %mask : i32 -> <ui32>
          %340 = arith.addi %arg15, %c-616729560_i32 : i32
          %341 = ascvec.duplicate_scalar %340, %mask : i32 -> <ui32>
          %342 = ascvec.xor %336, %339, %mask : <ui32>, <ui32> -> <ui32>
          %343 = ascvec.xor %337, %341, %mask : <ui32>, <ui32> -> <ui32>
          %dst0_63, %dst1_64 = ascvec.mull %342, %272, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %dst0_65, %dst1_66 = ascvec.mull %343, %273, %mask : <ui32>, <ui32> -> <ui32>, <ui32>
          %344 = ascvec.xor %dst1_66, %dst0_61, %mask : <ui32>, <ui32> -> <ui32>
          %345 = ascvec.xor %dst1_64, %dst0_59, %mask : <ui32>, <ui32> -> <ui32>
          %346 = arith.addi %arg14, %c-1879881855_i32 : i32
          %347 = ascvec.duplicate_scalar %346, %mask : i32 -> <ui32>
          %348 = arith.addi %arg15, %c-1767562579_i32 : i32
          %349 = ascvec.duplicate_scalar %348, %mask : i32 -> <ui32>
          %350 = ascvec.xor %344, %347, %mask : <ui32>, <ui32> -> <ui32>
          %351 = ascvec.xor %345, %349, %mask : <ui32>, <ui32> -> <ui32>
          %352 = arith.muli %arg18, %c128_i32 : i32
          %353 = ascvec.reinterpret %350 : <ui32> -> <i32>
          %354 = arith.index_cast %352 : i32 to index
          ascvec.storealign %arg16[%354], %353, %mask : memref<4096xi32, 1>, index, <i32>
          %355 = arith.addi %352, %c32_i32_16 : i32
          %356 = ascvec.reinterpret %dst0_65 : <ui32> -> <i32>
          %357 = arith.index_cast %355 : i32 to index
          ascvec.storealign %arg16[%357], %356, %mask : memref<4096xi32, 1>, index, <i32>
          %358 = arith.addi %352, %c64_i32_12 : i32
          %359 = ascvec.reinterpret %351 : <ui32> -> <i32>
          %360 = arith.index_cast %358 : i32 to index
          ascvec.storealign %arg16[%360], %359, %mask : memref<4096xi32, 1>, index, <i32>
          %361 = arith.addi %352, %c96_i32 : i32
          %362 = ascvec.reinterpret %dst0_63 : <ui32> -> <i32>
          %363 = arith.index_cast %361 : i32 to index
          ascvec.storealign %arg16[%363], %362, %mask : memref<4096xi32, 1>, index, <i32>
        }
        ascvec.mem_bar vst_vld
      } {cannir.vf_mode = "simd"}
      %118 = arith.divsi %116, %c32_i32 : i32
      %119 = arith.divsi %116, %c64_i32 : i32
      ascvec.get_buf v, %47, 0 {cannir.buffer.generated_sync}
      ascvec.get_buf v, %83, 0 {cannir.buffer.generated_sync}
      ascvec.get_buf v, %88, 0 {cannir.buffer.generated_sync}
      %120 = ascscalar.bitcast_memref %87 byte_offset = %c0_i64 : memref<1024xui8, 1> -> memref<?xui16, 1>
      ascvec.vf(%arg4, %78, %alloc_10, %alloc_11, %82, %118, %120, %119 : i64, memref<16384xbf16, 1>, memref<512xui32, 1>, memref<4096xi32, 1>, memref<16384xui8, 1>, i32, memref<?xui16, 1>, i32) {
      ^bb0(%arg10: i64, %arg11: memref<16384xbf16, 1>, %arg12: memref<512xui32, 1>, %arg13: memref<4096xi32, 1>, %arg14: memref<16384xui8, 1>, %arg15: i32, %arg16: memref<?xui16, 1>, %arg17: i32):
        %c64_i32_12 = arith.constant 64 : i32
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
        %c16_i32_13 = arith.constant 16 : i32
        %c2147483647_i32 = arith.constant 2147483647 : i32
        %c2_i32 = arith.constant 2 : i32
        %c0_i32_14 = arith.constant 0 : i32
        %c32_i32_15 = arith.constant 32 : i32
        %mask, %remaining_out = ascvec.update_mask %c32_i32_15 : !ascvec.mask
        scf.for %arg18 = %c0_i32_14 to %arg15 step %c1_i32  : i32 {
          %202 = ascvec.create_mask : !ascvec.mask
          %203 = ascvec.arange %c0_i32_14 : i32 -> <ui32>
          %204 = ascvec.duplicate_scalar %c0_i32_14, %202 : i32 -> <ui32>
          %205 = scf.for %arg19 = %c0_i32_14 to %c16_i32_13 step %c1_i32 iter_args(%arg20 = %204) -> (!ascvec.vreg<ui32>)  : i32 {
            %223 = arith.muli %arg18, %c32_i32_15 : i32
            %224 = arith.muli %arg19, %c2_i32 overflow<nsw> : i32
            %225 = arith.addi %223, %224 : i32
            %226 = arith.extsi %225 : i32 to i64
            %227 = arith.muli %226, %arg10 : i64
            %228 = arith.index_cast %227 : i64 to index
            %229 = ascvec.loadalign_unpack %arg11[%228] {mode = "b16_to_b32"} : memref<16384xbf16, 1>, index -> <bf16>
            %230 = ascvec.cast %229, %202 {rounding = 6 : i32} : <bf16> -> <f32>
            %231 = ascvec.reinterpret %230 : <f32> -> <ui32>
            %232 = ascvec.duplicate_scalar %c2147483647_i32, %202 : i32 -> <ui32>
            %233 = ascvec.and %231, %232, %202 : <ui32>, <ui32> -> <ui32>
            %234 = ascvec.max %arg20, %233, %202 : <ui32>
            scf.yield %234 : !ascvec.vreg<ui32>
          }
          %206 = ascvec.duplicate_scalar %c32_i32_15, %202 : i32 -> <ui32>
          %207 = ascvec.xor %203, %206, %202 : <ui32>, <ui32> -> <ui32>
          %208 = ascvec.gather_reg %205, %207 : <ui32>, <ui32> -> <ui32>
          %209 = ascvec.max %205, %208, %202 : <ui32>
          %210 = ascvec.shr %209, 23, %mask : <ui32> -> <ui32>
          %211 = ascvec.cmp_ge_scalar %210, %c255_i32, %mask : <ui32>, i32 -> !ascvec.mask
          %212 = ascvec.max_scalar %210, %c8_i32, %mask : <ui32>, i32
          %213 = ascvec.add_scalar %212, %c-8_i32, %mask : <ui32>, i32
          %214 = ascvec.duplicate_scalar %c255_i32, %211 merge %213 : !ascvec.vreg<ui32> {mode = "merging"} : i32 -> <ui32>
          %215 = ascvec.duplicate_scalar %c254_i32, %mask : i32 -> <ui32>
          %216 = ascvec.sub %215, %214, %mask : <ui32>
          %217 = ascvec.cmp_ne_scalar %214, %c0_i32_14, %mask : <ui32>, i32 -> !ascvec.mask
          %218 = ascvec.shl %216, 23, %217 : <ui32> -> <ui32>
          %219 = ascvec.duplicate_scalar %c2139160576_i32, %211 merge %218 : !ascvec.vreg<ui32> {mode = "merging"} : i32 -> <ui32>
          %220 = ascvec.reinterpret %219 : <ui32> -> <f32>
          %221 = arith.muli %arg18, %c32_i32_15 : i32
          %222 = arith.index_cast %221 : i32 to index
          ascvec.storealign %arg12[%222], %214, %mask : memref<512xui32, 1>, index, <ui32>
          scf.for %arg19 = %c0_i32_14 to %c8_i32 step %c1_i32  : i32 {
            %223 = arith.muli %arg18, %c256_i32 : i32
            %224 = arith.muli %arg19, %c32_i32_15 overflow<nsw> : i32
            %225 = arith.addi %223, %224 : i32
            %226 = arith.index_cast %225 : i32 to index
            %227 = ascvec.loadalign %arg13[%226] : memref<4096xi32, 1>, index -> <i32>
            %228 = ascvec.reinterpret %227 : <i32> -> <ui32>
            %229 = ascvec.shl %228, 8, %mask : <ui32> -> <ui32>
            %230 = ascvec.shr %228, 24, %mask : <ui32> -> <ui32>
            %231 = ascvec.or %229, %230, %mask : <ui32>, <ui32> -> <ui32>
            %232 = arith.muli %arg19, %c4_i32 overflow<nsw> : i32
            %233 = arith.addi %221, %232 : i32
            %234 = arith.extsi %233 : i32 to i64
            %235 = arith.muli %234, %arg10 : i64
            %236 = ascvec.shr %228, 16, %mask : <ui32> -> <ui32>
            %237 = arith.index_cast %235 : i64 to index
            %238 = ascvec.loadalign_unpack %arg11[%237] {mode = "b16_to_b32"} : memref<16384xbf16, 1>, index -> <bf16>
            %239 = ascvec.cast %238, %mask {rounding = 6 : i32} : <bf16> -> <f32>
            %240 = ascvec.mul %239, %220, %mask : <f32>
            %241 = ascvec.cast %240, %mask : <f32> -> <bf16>
            %242 = ascvec.cast %241, %mask {rounding = 6 : i32} : <bf16> -> <f32>
            %243 = ascvec.reinterpret %242 : <f32> -> <ui32>
            %244 = ascvec.shr %243, 24, %mask : <ui32> -> <ui32>
            %245 = ascvec.duplicate_scalar %c128_i32, %mask : i32 -> <ui32>
            %246 = ascvec.and %244, %245, %mask : <ui32>, <ui32> -> <ui32>
            %247 = ascvec.duplicate_scalar %c2147483647_i32, %mask : i32 -> <ui32>
            %248 = ascvec.and %243, %247, %mask : <ui32>, <ui32> -> <ui32>
            %249 = ascvec.shr %248, 4, %mask : <ui32> -> <ui32>
            %250 = ascvec.duplicate_scalar %c62914560_i32, %mask : i32 -> <ui32>
            %251 = ascvec.sub %249, %250, %mask : <ui32>
            %252 = ascvec.reinterpret %248 : <ui32> -> <f32>
            %253 = ascvec.mul_scalar %252, %cst, %mask : <f32>, f32
            %254 = ascvec.cast %253, %mask {rounding = 4 : i32} : <f32> -> <i32>
            %255 = ascvec.reinterpret %254 : <i32> -> <ui32>
            %256 = ascvec.cmp_ge_scalar %248, %c1015021568_i32, %mask : <ui32>, i32 -> !ascvec.mask
            %257 = ascvec.select %251, %255, %256 : <ui32>
            %258 = ascvec.duplicate_scalar %c65535_i32, %mask : i32 -> <ui32>
            %259 = ascvec.and %236, %258, %mask : <ui32>, <ui32> -> <ui32>
            %260 = ascvec.add %257, %259, %mask : <ui32>
            %261 = ascvec.shr %260, 16, %mask : <ui32> -> <ui32>
            %262 = ascvec.min_scalar %261, %c126_i32, %mask : <ui32>, i32
            %263 = ascvec.or %262, %246, %mask : <ui32>, <ui32> -> <ui32>
            %264 = ascvec.duplicate_scalar %c127_i32, %mask : i32 -> <ui32>
            %265 = ascvec.cmp_gt_scalar %248, %c2139095040_i32, %mask : <ui32>, i32 -> !ascvec.mask
            %266 = ascvec.select %264, %263, %265 : <ui32>
            ascvec.storealign_pack %arg14[%237], %266, %mask {mode = "b32_to_b8"} : memref<16384xui8, 1>, index, <ui32>
            %267 = arith.addi %233, %c1_i32 : i32
            %268 = arith.extsi %267 : i32 to i64
            %269 = arith.muli %268, %arg10 : i64
            %270 = ascvec.and %231, %258, %mask : <ui32>, <ui32> -> <ui32>
            %271 = arith.index_cast %269 : i64 to index
            %272 = ascvec.loadalign_unpack %arg11[%271] {mode = "b16_to_b32"} : memref<16384xbf16, 1>, index -> <bf16>
            %273 = ascvec.cast %272, %mask {rounding = 6 : i32} : <bf16> -> <f32>
            %274 = ascvec.mul %273, %220, %mask : <f32>
            %275 = ascvec.cast %274, %mask : <f32> -> <bf16>
            %276 = ascvec.cast %275, %mask {rounding = 6 : i32} : <bf16> -> <f32>
            %277 = ascvec.reinterpret %276 : <f32> -> <ui32>
            %278 = ascvec.shr %277, 24, %mask : <ui32> -> <ui32>
            %279 = ascvec.and %278, %245, %mask : <ui32>, <ui32> -> <ui32>
            %280 = ascvec.and %277, %247, %mask : <ui32>, <ui32> -> <ui32>
            %281 = ascvec.shr %280, 4, %mask : <ui32> -> <ui32>
            %282 = ascvec.sub %281, %250, %mask : <ui32>
            %283 = ascvec.reinterpret %280 : <ui32> -> <f32>
            %284 = ascvec.mul_scalar %283, %cst, %mask : <f32>, f32
            %285 = ascvec.cast %284, %mask {rounding = 4 : i32} : <f32> -> <i32>
            %286 = ascvec.reinterpret %285 : <i32> -> <ui32>
            %287 = ascvec.cmp_ge_scalar %280, %c1015021568_i32, %mask : <ui32>, i32 -> !ascvec.mask
            %288 = ascvec.select %282, %286, %287 : <ui32>
            %289 = ascvec.and %270, %258, %mask : <ui32>, <ui32> -> <ui32>
            %290 = ascvec.add %288, %289, %mask : <ui32>
            %291 = ascvec.shr %290, 16, %mask : <ui32> -> <ui32>
            %292 = ascvec.min_scalar %291, %c126_i32, %mask : <ui32>, i32
            %293 = ascvec.or %292, %279, %mask : <ui32>, <ui32> -> <ui32>
            %294 = ascvec.cmp_gt_scalar %280, %c2139095040_i32, %mask : <ui32>, i32 -> !ascvec.mask
            %295 = ascvec.select %264, %293, %294 : <ui32>
            ascvec.storealign_pack %arg14[%271], %295, %mask {mode = "b32_to_b8"} : memref<16384xui8, 1>, index, <ui32>
            %296 = arith.addi %233, %c2_i32 : i32
            %297 = arith.extsi %296 : i32 to i64
            %298 = arith.muli %297, %arg10 : i64
            %299 = ascvec.and %228, %258, %mask : <ui32>, <ui32> -> <ui32>
            %300 = arith.index_cast %298 : i64 to index
            %301 = ascvec.loadalign_unpack %arg11[%300] {mode = "b16_to_b32"} : memref<16384xbf16, 1>, index -> <bf16>
            %302 = ascvec.cast %301, %mask {rounding = 6 : i32} : <bf16> -> <f32>
            %303 = ascvec.mul %302, %220, %mask : <f32>
            %304 = ascvec.cast %303, %mask : <f32> -> <bf16>
            %305 = ascvec.cast %304, %mask {rounding = 6 : i32} : <bf16> -> <f32>
            %306 = ascvec.reinterpret %305 : <f32> -> <ui32>
            %307 = ascvec.shr %306, 24, %mask : <ui32> -> <ui32>
            %308 = ascvec.and %307, %245, %mask : <ui32>, <ui32> -> <ui32>
            %309 = ascvec.and %306, %247, %mask : <ui32>, <ui32> -> <ui32>
            %310 = ascvec.shr %309, 4, %mask : <ui32> -> <ui32>
            %311 = ascvec.sub %310, %250, %mask : <ui32>
            %312 = ascvec.reinterpret %309 : <ui32> -> <f32>
            %313 = ascvec.mul_scalar %312, %cst, %mask : <f32>, f32
            %314 = ascvec.cast %313, %mask {rounding = 4 : i32} : <f32> -> <i32>
            %315 = ascvec.reinterpret %314 : <i32> -> <ui32>
            %316 = ascvec.cmp_ge_scalar %309, %c1015021568_i32, %mask : <ui32>, i32 -> !ascvec.mask
            %317 = ascvec.select %311, %315, %316 : <ui32>
            %318 = ascvec.and %299, %258, %mask : <ui32>, <ui32> -> <ui32>
            %319 = ascvec.add %317, %318, %mask : <ui32>
            %320 = ascvec.shr %319, 16, %mask : <ui32> -> <ui32>
            %321 = ascvec.min_scalar %320, %c126_i32, %mask : <ui32>, i32
            %322 = ascvec.or %321, %308, %mask : <ui32>, <ui32> -> <ui32>
            %323 = ascvec.cmp_gt_scalar %309, %c2139095040_i32, %mask : <ui32>, i32 -> !ascvec.mask
            %324 = ascvec.select %264, %322, %323 : <ui32>
            ascvec.storealign_pack %arg14[%300], %324, %mask {mode = "b32_to_b8"} : memref<16384xui8, 1>, index, <ui32>
            %325 = arith.addi %233, %c3_i32 : i32
            %326 = arith.extsi %325 : i32 to i64
            %327 = arith.muli %326, %arg10 : i64
            %328 = ascvec.shr %231, 16, %mask : <ui32> -> <ui32>
            %329 = arith.index_cast %327 : i64 to index
            %330 = ascvec.loadalign_unpack %arg11[%329] {mode = "b16_to_b32"} : memref<16384xbf16, 1>, index -> <bf16>
            %331 = ascvec.cast %330, %mask {rounding = 6 : i32} : <bf16> -> <f32>
            %332 = ascvec.mul %331, %220, %mask : <f32>
            %333 = ascvec.cast %332, %mask : <f32> -> <bf16>
            %334 = ascvec.cast %333, %mask {rounding = 6 : i32} : <bf16> -> <f32>
            %335 = ascvec.reinterpret %334 : <f32> -> <ui32>
            %336 = ascvec.shr %335, 24, %mask : <ui32> -> <ui32>
            %337 = ascvec.and %336, %245, %mask : <ui32>, <ui32> -> <ui32>
            %338 = ascvec.and %335, %247, %mask : <ui32>, <ui32> -> <ui32>
            %339 = ascvec.shr %338, 4, %mask : <ui32> -> <ui32>
            %340 = ascvec.sub %339, %250, %mask : <ui32>
            %341 = ascvec.reinterpret %338 : <ui32> -> <f32>
            %342 = ascvec.mul_scalar %341, %cst, %mask : <f32>, f32
            %343 = ascvec.cast %342, %mask {rounding = 4 : i32} : <f32> -> <i32>
            %344 = ascvec.reinterpret %343 : <i32> -> <ui32>
            %345 = ascvec.cmp_ge_scalar %338, %c1015021568_i32, %mask : <ui32>, i32 -> !ascvec.mask
            %346 = ascvec.select %340, %344, %345 : <ui32>
            %347 = ascvec.and %328, %258, %mask : <ui32>, <ui32> -> <ui32>
            %348 = ascvec.add %346, %347, %mask : <ui32>
            %349 = ascvec.shr %348, 16, %mask : <ui32> -> <ui32>
            %350 = ascvec.min_scalar %349, %c126_i32, %mask : <ui32>, i32
            %351 = ascvec.or %350, %337, %mask : <ui32>, <ui32> -> <ui32>
            %352 = ascvec.cmp_gt_scalar %338, %c2139095040_i32, %mask : <ui32>, i32 -> !ascvec.mask
            %353 = ascvec.select %264, %351, %352 : <ui32>
            ascvec.storealign_pack %arg14[%329], %353, %mask {mode = "b32_to_b8"} : memref<16384xui8, 1>, index, <ui32>
          }
        }
        ascvec.mem_bar vst_vld
        %mask_16, %remaining_out_17 = ascvec.update_mask %c32_i32_15 elem_bits 16 : !ascvec.mask
        scf.for %arg18 = %c0_i32_14 to %arg17 step %c1_i32  : i32 {
          %202 = arith.muli %arg18, %c64_i32_12 : i32
          %203 = arith.index_cast %202 : i32 to index
          %204 = ascvec.loadalign %arg12[%203] : memref<512xui32, 1>, index -> <ui32>
          %205 = arith.addi %202, %c32_i32_15 : i32
          %206 = arith.index_cast %205 : i32 to index
          %207 = ascvec.loadalign %arg12[%206] : memref<512xui32, 1>, index -> <ui32>
          %208 = ascvec.shl %207, 8, %mask : <ui32> -> <ui32>
          %209 = ascvec.or %204, %208, %mask : <ui32>, <ui32> -> <ui32>
          %210 = ascvec.pack %209 src_elem_bits 32 dst_elem_bits 16 {part = "low"} : <ui32> -> <ui16>
          %211 = arith.extsi %arg18 : i32 to i64
          %212 = arith.muli %211, %arg10 : i64
          %213 = arith.index_cast %212 : i64 to index
          ascvec.storealign %arg16[%213], %210, %mask_16 : memref<?xui16, 1>, index, <ui16>
        }
        ascvec.mem_bar vst_vld
      } {cannir.vf_mode = "simd"}
      ascvec.release_buf v, %88, 0 {cannir.buffer.generated_sync}
      ascvec.release_buf v, %83, 0 {cannir.buffer.generated_sync}
      ascvec.release_buf v, %47, 0 {cannir.buffer.generated_sync}
      %121 = cannir.extract_layout %arg1 : !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
      %122 = cannir.extract_shape %121 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, -1)>
      %123 = cannir.extract_dim %122[0] : !cannir.shape<(-1, -1)>
      %124 = arith.minsi %38, %123 : i64
      %125 = arith.subi %123, %124 : i64
      %126 = arith.minsi %125, %c1_i64 : i64
      %127 = arith.cmpi sle, %38, %123 : i64
      %128 = arith.select %127, %126, %c0_i64 : i64
      %129 = arith.cmpi sgt, %128, %c1_i64 : i64
      %130 = arith.select %129, %c1_i64, %128 : i64
      %131 = cannir.extract_dim %122[1] : !cannir.shape<(-1, -1)>
      %132 = arith.divsi %131, %60 : i64
      %133 = arith.minsi %40, %132 : i64
      %134 = arith.muli %133, %41 : i64
      %135 = arith.subi %131, %134 : i64
      %136 = arith.minsi %41, %135 : i64
      %137 = arith.cmpi sle, %40, %132 : i64
      %138 = arith.select %137, %136, %c0_i64 : i64
      %139 = arith.cmpi sgt, %138, %c16384_i64 : i64
      %140 = arith.select %139, %c16384_i64, %138 : i64
      %141 = cannir.extract_physical_layout %arg1 : !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
      %142 = cannir.extract_stride %141 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
      %143 = cannir.extract_dim %142[0] : !cannir.stride<(-1, 1)>
      %144 = arith.muli %38, %143 : i64
      %145 = arith.addi %144, %70 : i64
      %146 = arith.muli %145, %c8_i64 : i64
      %147 = ascscalar.memref_address %alloc_0 : memref<1xui8>
      %148 = arith.divui %146, %c8_i64 : i64
      %149 = arith.addi %147, %148 : i64
      %150 = ascscalar.bind_gm_ptr %149 : memref<1xui8>
      %151 = ascscalar.bind_local_ptr %81 : memref<1xui8, 1>
      %152 = cannir.extract_stride %121 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
      %153 = cannir.extract_dim %152[0] : !cannir.stride<(-1, 1)>
      %154 = arith.cmpi ugt, %130, %c0_i64 : i64
      %155 = arith.cmpi ugt, %140, %c0_i64 : i64
      %156 = arith.andi %154, %155 : i1
      scf.if %156 {
        ascvec.get_buf mte3, %83, 0 {cannir.buffer.generated_sync}
        %202 = arith.trunci %130 : i64 to i32
        %203 = arith.trunci %140 : i64 to i32
        ascvec.copy_ub2gm %150[%c0], %151[%c0], %202, %203, %153, %c16384_i32 l2_cache_ctl %c0_i32 : memref<1xui8>, memref<1xui8, 1>
        ascvec.release_buf mte3, %83, 0 {cannir.buffer.generated_sync}
      }
      %157 = arith.remsi %41, %c32_i64 : i64
      %158 = arith.cmpi ne, %157, %c0_i64 : i64
      %159 = arith.cmpi slt, %157, %c0_i64 : i64
      %160 = arith.andi %158, %159 : i1
      %161 = arith.divsi %41, %c32_i64 : i64
      %162 = arith.extui %160 : i1 to i64
      %163 = arith.subi %161, %162 : i64
      %164 = cannir.extract_layout %arg2 : !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
      %165 = cannir.extract_shape %164 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, -1)>
      %166 = cannir.extract_dim %165[0] : !cannir.shape<(-1, -1)>
      %167 = arith.minsi %38, %166 : i64
      %168 = arith.subi %166, %167 : i64
      %169 = arith.minsi %168, %c1_i64 : i64
      %170 = arith.cmpi sle, %38, %166 : i64
      %171 = arith.select %170, %169, %c0_i64 : i64
      %172 = arith.cmpi sgt, %171, %c1_i64 : i64
      %173 = arith.select %172, %c1_i64, %171 : i64
      %174 = cannir.extract_dim %165[1] : !cannir.shape<(-1, -1)>
      %175 = arith.maxsi %163, %c1_i64 : i64
      %176 = arith.divsi %174, %175 : i64
      %177 = arith.minsi %40, %176 : i64
      %178 = arith.muli %177, %163 : i64
      %179 = arith.subi %174, %178 : i64
      %180 = arith.minsi %163, %179 : i64
      %181 = arith.cmpi sle, %40, %176 : i64
      %182 = arith.select %181, %180, %c0_i64 : i64
      %183 = arith.cmpi sgt, %182, %c1024_i64 : i64
      %184 = arith.select %183, %c1024_i64, %182 : i64
      %185 = cannir.extract_physical_layout %arg2 : !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
      %186 = cannir.extract_stride %185 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
      %187 = cannir.extract_dim %186[0] : !cannir.stride<(-1, 1)>
      %188 = arith.muli %38, %187 : i64
      %189 = arith.muli %40, %163 : i64
      %190 = arith.addi %188, %189 : i64
      %191 = arith.muli %190, %c8_i64 : i64
      %192 = ascscalar.memref_address %alloc_1 : memref<1xui8>
      %193 = arith.divui %191, %c8_i64 : i64
      %194 = arith.addi %192, %193 : i64
      %195 = ascscalar.bind_gm_ptr %194 : memref<1xui8>
      %196 = ascscalar.bind_local_ptr %86 : memref<1xui8, 1>
      %197 = cannir.extract_stride %164 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
      %198 = cannir.extract_dim %197[0] : !cannir.stride<(-1, 1)>
      %199 = arith.cmpi ugt, %173, %c0_i64 : i64
      %200 = arith.cmpi ugt, %184, %c0_i64 : i64
      %201 = arith.andi %199, %200 : i1
      scf.if %201 {
        ascvec.get_buf mte3, %88, 0 {cannir.buffer.generated_sync}
        %202 = arith.trunci %173 : i64 to i32
        %203 = arith.trunci %184 : i64 to i32
        ascvec.copy_ub2gm %195[%c0], %196[%c0], %202, %203, %198, %c1024_i32 l2_cache_ctl %c0_i32 : memref<1xui8>, memref<1xui8, 1>
        ascvec.release_buf mte3, %88, 0 {cannir.buffer.generated_sync}
      }
      scf.yield %49 : i64
    }
    return
  }
}
