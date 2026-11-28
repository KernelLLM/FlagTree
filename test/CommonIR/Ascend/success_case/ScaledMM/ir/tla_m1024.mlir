// -----// IR Dump Before ConvertHIVMToStandard (convert-hivm-to-std) //----- //
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.mode = "tla", hacc.target = #hacc.target<"Ascend950">, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @scaled_m1024_kernel_mix_aic(%arg0: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf16, #hivm.address_space<gm>>, %arg3: memref<?xf16, #hivm.address_space<gm>>, %arg4: memref<?xf32, #hivm.address_space<gm>>, %arg5: memref<?xf32, #hivm.address_space<gm>>, %arg6: memref<?xf16, #hivm.address_space<gm>>, %arg7: memref<?xf16, #hivm.address_space<gm>>, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, true, true, true, false, false, false]> : vector<11xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, mix_mode = "mix", parallel_mode = "simd"} {
    %c32768_i64 = arith.constant 32768 : i64
    %c65536_i64 = arith.constant 65536 : i64
    %c0_i64 = arith.constant 0 : i64
    %c32_i32 = arith.constant 32 : i32
    %c16_i32 = arith.constant 16 : i32
    %c256_i32 = arith.constant 256 : i32
    %c128_i32 = arith.constant 128 : i32
    %c0_i32 = arith.constant 0 : i32
    %c256 = arith.constant 256 : index
    %c64 = arith.constant 64 : index
    %false = arith.constant false
    %c64_i32 = arith.constant 64 : i32
    %c1048576_i32 = arith.constant 1048576 : i32
    %c524288_i32 = arith.constant 524288 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = hivm.hir.get_block_idx -> i64
    %1 = arith.trunci %0 : i64 to i32
    %2 = arith.remsi %1, %arg8 : i32
    hivm.hir.set_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
    scf.for %arg11 = %2 to %c64_i32 step %c32_i32  : i32 {
      %3 = arith.divsi %arg11, %c16_i32 : i32
      %4 = arith.remsi %arg11, %c16_i32 : i32
      %5 = arith.muli %4, %c256_i32 : i32
      %6 = hivm.hir.pointer_cast(%c0_i64) : memref<128x256xf32, #hivm.address_space<ub>>
      %7 = hivm.hir.pointer_cast(%c65536_i64) : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>
      %8 = hivm.hir.pointer_cast(%c0_i64) : memref<16x8x16x16xf16, #hivm.address_space<cbuf>>
      %9 = hivm.hir.pointer_cast(%c0_i64) : memref<16x16x16x16xf32, #hivm.address_space<cc>>
      %10 = hivm.hir.pointer_cast(%c0_i64) : memref<4x16x16x16xf16, #hivm.address_space<ca>>
      %11 = hivm.hir.pointer_cast(%c0_i64) : memref<4x16x16x16xf16, #hivm.address_space<cb>>
      %12 = hivm.hir.pointer_cast(%c32768_i64) : memref<4x16x16x16xf16, #hivm.address_space<ca>>
      %13 = hivm.hir.pointer_cast(%c32768_i64) : memref<4x16x16x16xf16, #hivm.address_space<cb>>
      hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
      scf.for %arg12 = %c0_i32 to %c32_i32 step %c1_i32  : i32 {
        %15 = arith.muli %3, %c1048576_i32 : i32
        %16 = arith.muli %arg12, %c128_i32 : i32
        %17 = arith.index_cast %15 : i32 to index
        %18 = arith.index_cast %16 : i32 to index
        %19 = arith.addi %17, %18 : index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%19], sizes: [256, 128], strides: [4096, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<256x128xf16, strided<[4096, 1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast : memref<256x128xf16, strided<[4096, 1], offset: ?>, #hivm.address_space<gm>>) outs(%7 : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>)
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID1>]
        %20 = arith.muli %arg12, %c524288_i32 : i32
        %21 = arith.index_cast %20 : i32 to index
        %22 = arith.index_cast %5 : i32 to index
        %23 = arith.addi %21, %22 : index
        %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [%23], sizes: [128, 256], strides: [4096, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<128x256xf16, strided<[4096, 1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
        hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast_0 : memref<128x256xf16, strided<[4096, 1], offset: ?>, #hivm.address_space<gm>>) outs(%8 : memref<16x8x16x16xf16, #hivm.address_space<cbuf>>)
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID1>]
        hivm.hir.l12l0 ins(%7 : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>) outs(%10 : memref<4x16x16x16xf16, #hivm.address_space<ca>>) {k_part_idx = 0, k_part = 64, k_part_ceil = 64, k_part_loop = 2, k_part_actual = 64, m = 256}
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
        hivm.hir.l12l0 ins(%8 : memref<16x8x16x16xf16, #hivm.address_space<cbuf>>) outs(%11 : memref<4x16x16x16xf16, #hivm.address_space<cb>>) {k_part_idx = 0, k_part = 64, k_part_ceil = 64, k_part_loop = 2, k_part_actual = 64, m = 256}
        hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_M>, <EVENT_ID0>]
        %24 = arith.cmpi eq, %arg12, %c0_i32 : i32
        hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_M>, <EVENT_ID0>]
        hivm.hir.mmadL0 ins(%10, %11, %c256, %c64, %c256, %false, %false, %24 : memref<4x16x16x16xf16, #hivm.address_space<ca>>, memref<4x16x16x16xf16, #hivm.address_space<cb>>, index, index, index, i1, i1, i1) outs(%9 : memref<16x16x16x16xf32, #hivm.address_space<cc>>)
        hivm.hir.set_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID1>]
        hivm.hir.l12l0 ins(%7 : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>) outs(%12 : memref<4x16x16x16xf16, #hivm.address_space<ca>>) {k_part_idx = 1, k_part = 64, k_part_ceil = 64, k_part_loop = 2, k_part_actual = 64, m = 256}
        hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.l12l0 ins(%8 : memref<16x8x16x16xf16, #hivm.address_space<cbuf>>) outs(%13 : memref<4x16x16x16xf16, #hivm.address_space<cb>>) {k_part_idx = 1, k_part = 64, k_part_ceil = 64, k_part_loop = 2, k_part_actual = 64, m = 256}
        hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_M>, <EVENT_ID0>]
        hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
        hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_M>, <EVENT_ID0>]
        hivm.hir.mmadL0 ins(%12, %13, %c256, %c64, %c256, %false, %false, %false : memref<4x16x16x16xf16, #hivm.address_space<ca>>, memref<4x16x16x16xf16, #hivm.address_space<cb>>, index, index, index, i1, i1, i1) outs(%9 : memref<16x16x16x16xf32, #hivm.address_space<cc>>)
        hivm.hir.set_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID1>]
      }
      hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
      %14 = arith.cmpi sge, %arg11, %c32_i32 : i32
      scf.if %14 {
        hivm.hir.sync_block_wait[<CUBE>, <PIPE_FIX>, <PIPE_FIX>] flag = 2
      }
      hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
      hivm.hir.fixpipe {c0_pad_en = true, dma_mode = #hivm.dma_mode<nz2nd>} ins(%9 : memref<16x16x16x16xf32, #hivm.address_space<cc>>) outs(%6 : memref<128x256xf32, #hivm.address_space<ub>>) dual_dst_mode = <ROW_SPLIT>
      hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
      hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_FIX>] flag = 0
    }
    hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID1>]
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_FIX>, <PIPE_FIX>] flag = 2
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
  func.func @scaled_m1024_kernel_mix_aiv_scope_0(%arg0: memref<256xf32, #hivm.address_space<ub>>, %arg1: memref<128xf32, #hivm.address_space<ub>>, %arg2: memref<128x256xf32, #hivm.address_space<ub>>, %arg3: memref<128x256xf16, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vector_function, hivm.vf_mode = #hivm.vf_mode<SIMD>, no_inline, vector_mode = "vec"} {
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c1_i32 = arith.constant 1 : i32
    %c128 = arith.constant 128 : index
    %c192 = arith.constant 192 : index
    %res = ave.hir.vload <NORM> %arg0[%c0] : memref<256xf32, #hivm.address_space<ub>> into vector<64xf32>
    %res_0 = ave.hir.vload <NORM> %arg0[%c64] : memref<256xf32, #hivm.address_space<ub>> into vector<64xf32>
    scf.for %arg4 = %c0_i32 to %c128_i32 step %c1_i32  : i32 {
      %0 = arith.index_cast %arg4 : i32 to index
      %res_3 = ave.hir.vload <BRC_B32> %arg1[%0] : memref<128xf32, #hivm.address_space<ub>> into vector<64xf32>
      %res_4 = ave.hir.vload <NORM> %arg2[%0, %c0] : memref<128x256xf32, #hivm.address_space<ub>> into vector<64xf32>
      %1 = ave.hir.pge <ALL> : vector<64xi1>
      %2 = ave.hir.vmul %res_4, %res_3, %1 : vector<64xf32>, vector<64xi1>
      %res_5 = ave.hir.vload <NORM> %arg2[%0, %c64] : memref<128x256xf32, #hivm.address_space<ub>> into vector<64xf32>
      %3 = ave.hir.vmul %res_5, %res_3, %1 : vector<64xf32>, vector<64xi1>
      %4 = ave.hir.vmul %2, %res, %1 : vector<64xf32>, vector<64xi1>
      %5 = ave.hir.vmul %3, %res_0, %1 : vector<64xf32>, vector<64xi1>
      %6 = ave.hir.vtruncf %4, <round>, true, <part_even>, %1 : vector<64xf32>, vector<128xf16>, vector<64xi1>
      %7 = ave.hir.vtruncf %5, <round>, true, <part_even>, %1 : vector<64xf32>, vector<128xf16>, vector<64xi1>
      %res1, %res2 = ave.hir.vdintlv %6, %7 {layout_change = #ave<layout_change SPARSE>} : vector<128xf16>, vector<128xf16>
      %8 = ave.hir.pge <ALL> : vector<128xi1>
      ave.hir.masked_store <NORM_B32> %arg3[%0, %c0], %8, %res1 : memref<128x256xf16, #hivm.address_space<ub>>, vector<128xi1>, vector<128xf16>
    }
    %res_1 = ave.hir.vload <NORM> %arg0[%c128] : memref<256xf32, #hivm.address_space<ub>> into vector<64xf32>
    %res_2 = ave.hir.vload <NORM> %arg0[%c192] : memref<256xf32, #hivm.address_space<ub>> into vector<64xf32>
    scf.for %arg4 = %c0_i32 to %c128_i32 step %c1_i32  : i32 {
      %0 = arith.index_cast %arg4 : i32 to index
      %res_3 = ave.hir.vload <BRC_B32> %arg1[%0] : memref<128xf32, #hivm.address_space<ub>> into vector<64xf32>
      %res_4 = ave.hir.vload <NORM> %arg2[%0, %c128] : memref<128x256xf32, #hivm.address_space<ub>> into vector<64xf32>
      %1 = ave.hir.pge <ALL> : vector<64xi1>
      %2 = ave.hir.vmul %res_4, %res_3, %1 : vector<64xf32>, vector<64xi1>
      %res_5 = ave.hir.vload <NORM> %arg2[%0, %c192] : memref<128x256xf32, #hivm.address_space<ub>> into vector<64xf32>
      %3 = ave.hir.vmul %res_5, %res_3, %1 : vector<64xf32>, vector<64xi1>
      %4 = ave.hir.vmul %2, %res_1, %1 : vector<64xf32>, vector<64xi1>
      %5 = ave.hir.vmul %3, %res_2, %1 : vector<64xf32>, vector<64xi1>
      %6 = ave.hir.vtruncf %4, <round>, true, <part_even>, %1 : vector<64xf32>, vector<128xf16>, vector<64xi1>
      %7 = ave.hir.vtruncf %5, <round>, true, <part_even>, %1 : vector<64xf32>, vector<128xf16>, vector<64xi1>
      %res1, %res2 = ave.hir.vdintlv %6, %7 {layout_change = #ave<layout_change SPARSE>} : vector<128xf16>, vector<128xf16>
      %8 = ave.hir.pge <ALL> : vector<128xi1>
      ave.hir.masked_store <NORM_B32> %arg3[%0, %c128], %8, %res1 : memref<128x256xf16, #hivm.address_space<ub>>, vector<128xi1>, vector<128xf16>
    }
    return
  }
  func.func @scaled_m1024_kernel_mix_aiv(%arg0: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf16, #hivm.address_space<gm>>, %arg3: memref<?xf16, #hivm.address_space<gm>>, %arg4: memref<?xf32, #hivm.address_space<gm>>, %arg5: memref<?xf32, #hivm.address_space<gm>>, %arg6: memref<?xf16, #hivm.address_space<gm>>, %arg7: memref<?xf16, #hivm.address_space<gm>>, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, true, true, true, false, false, false]> : vector<11xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, mix_mode = "mix", parallel_mode = "simd"} {
    %c132608_i64 = arith.constant 132608 : i64
    %c131584_i64 = arith.constant 131584 : i64
    %c131072_i64 = arith.constant 131072 : i64
    %c0_i64 = arith.constant 0 : i64
    %c32_i32 = arith.constant 32 : i32
    %c16_i32 = arith.constant 16 : i32
    %c256_i32 = arith.constant 256 : i32
    %c128_i64 = arith.constant 128 : i64
    %c4096_i64 = arith.constant 4096 : i64
    %c64_i32 = arith.constant 64 : i32
    %0 = hivm.hir.get_block_idx -> i64
    %1 = arith.trunci %0 : i64 to i32
    %2 = arith.remsi %1, %arg8 : i32
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    scf.for %arg11 = %2 to %c64_i32 step %c32_i32  : i32 {
      %3 = arith.divsi %arg11, %c16_i32 : i32
      %4 = arith.muli %3, %c256_i32 : i32
      %5 = arith.remsi %arg11, %c16_i32 : i32
      %6 = arith.muli %5, %c256_i32 : i32
      %7 = hivm.hir.pointer_cast(%c0_i64) : memref<128x256xf32, #hivm.address_space<ub>>
      %8 = hivm.hir.pointer_cast(%c131072_i64) : memref<128xf32, #hivm.address_space<ub>>
      %9 = hivm.hir.pointer_cast(%c131584_i64) : memref<256xf32, #hivm.address_space<ub>>
      %10 = hivm.hir.get_sub_block_idx -> i64
      %11 = arith.muli %10, %c128_i64 : i64
      %12 = arith.extsi %4 : i32 to i64
      %13 = arith.addi %12, %11 : i64
      %14 = arith.index_cast %13 : i64 to index
      %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%14], sizes: [128], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.load ins(%reinterpret_cast : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%8 : memref<128xf32, #hivm.address_space<ub>>) {markDCacheInvalidatePatternVisited = 0 : i32}
      %15 = arith.index_cast %6 : i32 to index
      %reinterpret_cast_0 = memref.reinterpret_cast %arg5 to offset: [%15], sizes: [256], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<256xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.load ins(%reinterpret_cast_0 : memref<256xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%9 : memref<256xf32, #hivm.address_space<ub>>) {markDCacheInvalidatePatternVisited = 0 : i32}
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 0
      %16 = hivm.hir.pointer_cast(%c132608_i64) : memref<128x256xf16, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      func.call @scaled_m1024_kernel_mix_aiv_scope_0(%9, %8, %7, %16) {hivm.vector_function} : (memref<256xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>, memref<128x256xf32, #hivm.address_space<ub>>, memref<128x256xf16, #hivm.address_space<ub>>) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_V>] flag = 2
      %17 = arith.muli %13, %c4096_i64 : i64
      %18 = arith.index_cast %17 : i64 to index
      %19 = arith.addi %18, %15 : index
      %reinterpret_cast_1 = memref.reinterpret_cast %arg6 to offset: [%19], sizes: [128, 256], strides: [4096, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<128x256xf16, strided<[4096, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.store ins(%16 : memref<128x256xf16, #hivm.address_space<ub>>) outs(%reinterpret_cast_1 : memref<128x256xf16, strided<[4096, 1], offset: ?>, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    }
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}


