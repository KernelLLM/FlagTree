// -----// IR Dump Before ConvertHIVMToStandard (convert-hivm-to-std) //----- //
#map = affine_map<()[s0, s1, s2] -> (s0 + s1 * s2)>
#map1 = affine_map<(d0)[s0] -> (d0 + s0)>
#map2 = affine_map<()[s0] -> (s0 * 64)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950">, hivm.module_core_type = #hivm.module_core_type<MIX>, ssbuffer.insertionOptimization} {
  func.func @scaled_mm_kernel_mix_aic(%arg0: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf16, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %arg6: memref<?xf16, #hivm.address_space<gm>> {tt.tensor_kind = 1 : i32}, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false]> : vector<16xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
    %c65536_i64 = arith.constant 65536 : i64
    %c32768_i64 = arith.constant 32768 : i64
    %c163840_i64 = arith.constant 163840 : i64
    %c131072_i64 = arith.constant 131072 : i64
    %c49920_i64 = arith.constant 49920 : i64
    %c98304_i64 = arith.constant 98304 : i64
    %c229376_i64 = arith.constant 229376 : i64
    %c196608_i64 = arith.constant 196608 : i64
    %c3_i64 = arith.constant 3 : i64
    %c-1_i64 = arith.constant -1 : i64
    %c2_i64 = arith.constant 2 : i64
    %c1_i64 = arith.constant 1 : i64
    %c64_i32 = arith.constant 64 : i32
    %c32_i32 = arith.constant 32 : i32
    %c1_i32 = arith.constant 1 : i32
    %false = arith.constant false
    %true = arith.constant true
    %c128 = arith.constant 128 : index
    %c-128_i64 = arith.constant -128 : i64
    %c128_i32 = arith.constant 128 : i32
    %c0_i32 = arith.constant 0 : i32
    %c127_i32 = arith.constant 127 : i32
    %c0 = arith.constant 0 : index
    %c0_i64 = arith.constant 0 : i64
    %alloca = memref.alloca() : memref<1xi64>
    memref.store %c0_i64, %alloca[%c0] : memref<1xi64>
    %alloca_0 = memref.alloca() : memref<1xi64>
    memref.store %c0_i64, %alloca_0[%c0] : memref<1xi64>
    %0 = hivm.hir.pointer_cast(%c0_i64) : memref<8x8x16x16xf32, #hivm.address_space<cc>>
    %1 = hivm.hir.pointer_cast(%c65536_i64) : memref<8x8x16x16xf32, #hivm.address_space<cc>>
    %2 = hivm.hir.pointer_cast(%c32768_i64) : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
    %3 = hivm.hir.pointer_cast(%c163840_i64) : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
    %4 = hivm.hir.pointer_cast(%c0_i64) : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
    %5 = hivm.hir.pointer_cast(%c131072_i64) : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
    %6 = hivm.hir.pointer_cast(%c0_i64) : memref<64x128xf32, #hivm.address_space<ub>>
    %7 = hivm.hir.pointer_cast(%c49920_i64) : memref<64x128xf32, #hivm.address_space<ub>>
    %8 = hivm.hir.pointer_cast(%c98304_i64) : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
    %9 = hivm.hir.pointer_cast(%c229376_i64) : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
    %10 = hivm.hir.pointer_cast(%c65536_i64) : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
    %11 = hivm.hir.pointer_cast(%c196608_i64) : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
    %12 = arith.muli %arg13, %arg14 : i32
    %13 = arith.muli %12, %arg15 : i32
    %14 = hivm.hir.get_block_idx -> i64
    %15 = arith.trunci %14 : i64 to i32
    %16 = arith.addi %13, %c64_i32 : i32
    %17 = arith.cmpi slt, %15, %13 : i32
    %18 = arith.select %17, %16, %15 : i32
    hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID1>]
    scf.for %arg16 = %15 to %18 step %c32_i32  : i32 {
      %19 = memref.load %alloca_0[%c0] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi64>
      %20 = arith.remui %19, %c2_i64 : i64
      %21 = arith.index_cast %20 : i64 to index
      %22 = arith.cmpi eq, %20, %c1_i64 : i64
      %23 = arith.select %22, %1, %0 : memref<8x8x16x16xf32, #hivm.address_space<cc>>
      %24 = arith.index_cast %21 : index to i1
      %25 = arith.xori %24, %true : i1
      %26 = arith.extui %25 : i1 to i64
      hivm.hir.set_ctrl false at ctrl[60]
      hivm.hir.set_ctrl false at ctrl[60]
      hivm.hir.set_ctrl true at ctrl[48]
      hivm.hir.set_ctrl true at ctrl[48]
      hivm.hir.set_ctrl true at ctrl[60]
      hivm.hir.set_ctrl true at ctrl[60]
      %27 = arith.cmpi sge, %arg16, %15 : i32
      %28 = arith.cmpi slt, %arg16, %13 : i32
      %29 = arith.andi %27, %28 : i1
      scf.if %29 {
        %31 = arith.select %22, %3, %2 : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
        %32 = arith.select %22, %5, %4 : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
        %33 = arith.select %22, %7, %6 : memref<64x128xf32, #hivm.address_space<ub>>
        %34 = arith.remsi %arg16, %arg13 : i32
        %35 = arith.addi %arg8, %c127_i32 : i32
        %36 = arith.divsi %35, %c128_i32 : i32
        %37 = arith.divsi %34, %36 : i32
        %38 = arith.remsi %34, %36 : i32
        %39 = arith.muli %37, %c128_i32 : i32
        %40 = arith.extsi %39 : i32 to i64
        %41 = arith.muli %38, %c128_i32 : i32
        %42 = arith.extsi %41 : i32 to i64
        %43 = arith.addi %arg9, %c127_i32 : i32
        %44 = arith.divsi %43, %c128_i32 : i32
        %45 = arith.subi %44, %c1_i32 : i32
        %46 = arith.muli %45, %c128_i32 : i32
        %47 = arith.extsi %arg10 : i32 to i64
        %48 = arith.muli %40, %47 : i64
        %49 = arith.index_cast %arg10 : i32 to index
        %50 = arith.index_cast %39 : i32 to index
        %51 = arith.extsi %46 : i32 to i64
        %52 = arith.addi %48, %51 : i64
        %53 = arith.index_cast %46 : i32 to index
        %54 = affine.apply #map()[%53, %50, %49]
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%54], sizes: [128, 128], strides: [%49, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<128x128xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        %55 = arith.extsi %arg11 : i32 to i64
        %56 = arith.muli %51, %55 : i64
        %57 = arith.index_cast %arg11 : i32 to index
        %58 = arith.addi %56, %42 : i64
        %59 = arith.index_cast %41 : i32 to index
        %60 = affine.apply #map()[%59, %53, %57]
        %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [%60], sizes: [128, 128], strides: [%57, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<128x128xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast : memref<128x128xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%32 : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>)
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
        hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast_1 : memref<128x128xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%31 : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>)
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID1>]
        hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, %26]
        hivm.hir.mmadL1 {already_set_real_mkn, hivm.remain_in_l0c, normalized_in_L0C} ins(%32, %31, %true, %c128, %c128, %c128 : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>, memref<8x8x16x16xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%23 : memref<8x8x16x16xf32, #hivm.address_space<cc>>) sync_related_args(%c0_i64, %c1_i64, %c-1_i64, %c-1_i64, %c-1_i64, %c-1_i64, %c-1_i64 : i64, i64, i64, i64, i64, i64, i64)
        %61 = arith.muli %arg11, %c128_i32 : i32
        %62 = arith.subi %c0_i32, %61 : i32
        %63 = arith.extsi %62 : i32 to i64
        hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
        hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID2>]
        hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID3>]
        %64:2 = scf.for %arg17 = %c1_i32 to %44 step %c1_i32 iter_args(%arg18 = %52, %arg19 = %58) -> (i64, i64)  : i32 {
          %65 = memref.load %alloca[%c0] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi64>
          %66 = arith.remui %65, %c2_i64 : i64
          %67 = arith.index_cast %66 : i64 to index
          %68 = arith.cmpi eq, %66, %c1_i64 : i64
          %69 = arith.select %68, %9, %8 : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
          %70 = arith.select %68, %11, %10 : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
          %71 = arith.index_cast %67 : index to i1
          %72 = arith.select %71, %c2_i64, %c3_i64 : i64
          %73 = arith.xori %71, %true : i1
          %74 = arith.extui %73 : i1 to i64
          %75 = arith.addi %arg18, %c-128_i64 : i64
          %76 = arith.index_cast %75 : i64 to index
          %reinterpret_cast_2 = memref.reinterpret_cast %arg2 to offset: [%76], sizes: [128, 128], strides: [%49, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<128x128xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
          %77 = arith.addi %arg19, %63 : i64
          %78 = arith.index_cast %77 : i64 to index
          %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [%78], sizes: [128, 128], strides: [%57, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<128x128xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %74]
          hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast_2 : memref<128x128xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%70 : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>)
          hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %72]
          hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast_3 : memref<128x128xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%69 : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>)
          hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID1>]
          hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, hivm.remain_in_l0c, normalized_in_L0C} ins(%70, %69, %false, %c128, %c128, %c128 : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>, memref<8x8x16x16xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%23 : memref<8x8x16x16xf32, #hivm.address_space<cc>>) sync_related_args(%c0_i64, %c1_i64, %74, %72, %c-1_i64, %c-1_i64, %c-1_i64 : i64, i64, i64, i64, i64, i64, i64)
          %79 = arith.addi %65, %c1_i64 : i64
          memref.store %79, %alloca[%c0] : memref<1xi64>
          scf.yield %75, %77 : i64, i64
        }
        hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
        hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID2>]
        hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID3>]
        hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
        hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 1
        hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%23 : memref<8x8x16x16xf32, #hivm.address_space<cc>>) outs(%33 : memref<64x128xf32, #hivm.address_space<ub>>) dual_dst_mode = <ROW_SPLIT>
        hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, %26]
        hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 0
      }
      %30 = arith.addi %19, %c1_i64 : i64
      memref.store %30, %alloca_0[%c0] : memref<1xi64>
    } {hivm.max_preload_num = 2 : i32}
    hivm.hir.wait_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 1
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 1
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
  func.func @scaled_mm_kernel_mix_aiv_outlined_vf_0(%arg0: memref<64xf32, #hivm.address_space<ub>>, %arg1: memref<64x128xf32, #hivm.address_space<ub>>, %arg2: memref<128xf32, #hivm.address_space<ub>>, %arg3: memref<64x128xf16, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    scf.for %arg4 = %c0 to %c64 step %c1 {
      %subview = memref.subview %arg0[%arg4] [1] [1] : memref<64xf32, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %res = ave.hir.vload <BRC_B32> %subview[%c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<norm>} : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xf32>
      scf.for %arg5 = %c0 to %c128 step %c64 {
        %subview_0 = memref.subview %arg1[%arg4, %arg5] [1, 64] [1, 1] : memref<64x128xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>
        %subview_1 = memref.subview %arg2[%arg5] [64] [1] : memref<128xf32, #hivm.address_space<ub>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
        %subview_2 = memref.subview %arg3[%arg4, %arg5] [1, 64] [1, 1] : memref<64x128xf16, #hivm.address_space<ub>> to memref<1x64xf16, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>
        %subview_3 = memref.subview %subview_0[0, 0] [1, 64] [1, 1] : memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>> to memref<64xf32, #map1, #hivm.address_space<ub>>
        %res_4 = ave.hir.vload <NORM> %subview_3[%c0] {functionType = #ave.func_dist_type<norm>} : memref<64xf32, #map1, #hivm.address_space<ub>> into vector<64xf32>
        %res_5 = ave.hir.vload <NORM> %subview_1[%c0] {functionType = #ave.func_dist_type<norm>} : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xf32>
        %0 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
        %1 = ave.hir.vmul %res, %res_5, %0 : vector<64xf32>, vector<64xi1>
        %2 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
        %3 = ave.hir.vmul %res_4, %1, %2 : vector<64xf32>, vector<64xi1>
        %4 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
        %5 = ave.hir.vtruncf %3, <rint>, false, <part_even>, %4 {functionType = #ave.func_dist_type<even>} : vector<64xf32>, vector<64xf16>, vector<64xi1>
        %subview_6 = memref.subview %subview_2[0, 0] [1, 64] [1, 1] : memref<1x64xf16, strided<[128, 1], offset: ?>, #hivm.address_space<ub>> to memref<64xf16, #map1, #hivm.address_space<ub>>
        %6 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
        %7 = builtin.unrealized_conversion_cast %5 : vector<64xf16> to vector<128xf16>
        %8 = llvm.bitcast %7 : vector<128xf16> to vector<64xf32>
        ave.hir.masked_store <PK_B32> %subview_6[%c0], %6, %8 {functionType = #ave.func_dist_type<pk>, hivm.is_continuous} : memref<64xf16, #map1, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
      }
    }
    return
  }
  func.func @scaled_mm_kernel_mix_aiv(%arg0: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf16, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %arg6: memref<?xf16, #hivm.address_space<gm>> {tt.tensor_kind = 1 : i32}, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false]> : vector<16xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
    %c83456_i64 = arith.constant 83456 : i64
    %c33536_i64 = arith.constant 33536 : i64
    %c33024_i64 = arith.constant 33024 : i64
    %c82944_i64 = arith.constant 82944 : i64
    %c32768_i64 = arith.constant 32768 : i64
    %c82688_i64 = arith.constant 82688 : i64
    %c49920_i64 = arith.constant 49920 : i64
    %true = arith.constant true
    %c2_i64 = arith.constant 2 : i64
    %c64_i32 = arith.constant 64 : i32
    %c127_i32 = arith.constant 127 : i32
    %c128_i32 = arith.constant 128 : i32
    %c32_i32 = arith.constant 32 : i32
    %c1_i64 = arith.constant 1 : i64
    %c0 = arith.constant 0 : index
    %c0_i64 = arith.constant 0 : i64
    %alloca = memref.alloca() : memref<1xi64>
    memref.store %c0_i64, %alloca[%c0] : memref<1xi64>
    %0 = hivm.hir.pointer_cast(%c33536_i64) : memref<64x128xf16, #hivm.address_space<ub>>
    %1 = hivm.hir.pointer_cast(%c83456_i64) : memref<64x128xf16, #hivm.address_space<ub>>
    %2 = hivm.hir.pointer_cast(%c33024_i64) : memref<128xf32, #hivm.address_space<ub>>
    %3 = hivm.hir.pointer_cast(%c82944_i64) : memref<128xf32, #hivm.address_space<ub>>
    %4 = hivm.hir.pointer_cast(%c32768_i64) : memref<64xf32, #hivm.address_space<ub>>
    %5 = hivm.hir.pointer_cast(%c82688_i64) : memref<64xf32, #hivm.address_space<ub>>
    %6 = hivm.hir.pointer_cast(%c49920_i64) : memref<64x128xf32, #hivm.address_space<ub>>
    %7 = hivm.hir.pointer_cast(%c0_i64) : memref<64x128xf32, #hivm.address_space<ub>>
    %8 = arith.muli %arg13, %arg14 : i32
    %9 = arith.muli %8, %arg15 : i32
    %10 = hivm.hir.get_block_idx -> i64
    %11 = arith.trunci %10 : i64 to i32
    %12 = hivm.hir.get_sub_block_idx -> i64
    %13 = arith.index_cast %12 : i64 to index
    %14 = affine.apply #map2()[%13]
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 1
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 1
    %15 = arith.addi %9, %c64_i32 : i32
    %16 = arith.cmpi slt, %11, %9 : i32
    %17 = arith.select %16, %15, %11 : i32
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
    scf.for %arg16 = %11 to %17 step %c32_i32  : i32 {
      %18 = memref.load %alloca[%c0] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi64>
      %19 = arith.remui %18, %c2_i64 : i64
      %20 = arith.index_cast %19 : i64 to index
      %21 = arith.index_cast %20 : index to i1
      %22 = arith.xori %21, %true : i1
      %23 = arith.extui %22 : i1 to i64
      %24 = arith.subi %arg16, %c32_i32 : i32
      hivm.hir.set_ctrl false at ctrl[60]
      hivm.hir.set_ctrl false at ctrl[60]
      hivm.hir.set_ctrl true at ctrl[48]
      hivm.hir.set_ctrl true at ctrl[48]
      hivm.hir.set_ctrl true at ctrl[60]
      hivm.hir.set_ctrl true at ctrl[60]
      %25 = arith.cmpi sge, %24, %11 : i32
      %26 = arith.cmpi slt, %24, %9 : i32
      %27 = arith.andi %25, %26 : i1
      scf.if %27 {
        %29 = arith.cmpi eq, %19, %c1_i64 : i64
        %30 = arith.select %29, %1, %0 : memref<64x128xf16, #hivm.address_space<ub>>
        %31 = arith.select %29, %3, %2 : memref<128xf32, #hivm.address_space<ub>>
        %32 = arith.select %29, %5, %4 : memref<64xf32, #hivm.address_space<ub>>
        %33 = arith.select %29, %7, %6 : memref<64x128xf32, #hivm.address_space<ub>>
        %34 = arith.remsi %24, %arg13 : i32
        %35 = arith.addi %arg8, %c127_i32 : i32
        %36 = arith.divsi %35, %c128_i32 : i32
        %37 = arith.divsi %34, %36 : i32
        %38 = arith.remsi %34, %36 : i32
        %39 = arith.muli %37, %c128_i32 : i32
        %40 = arith.muli %38, %c128_i32 : i32
        %41 = arith.index_cast %39 : i32 to index
        %42 = arith.index_cast %40 : i32 to index
        %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%41], sizes: [128], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview = memref.subview %reinterpret_cast[%14] [64] [1] : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %reinterpret_cast_0 = memref.reinterpret_cast %arg5 to offset: [%42], sizes: [128], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %23]
        hivm.hir.load ins(%subview : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%32 : memref<64xf32, #hivm.address_space<ub>>) {markDCacheInvalidatePatternVisited = 0 : i32} eviction_policy = <EvictFirst> core_type = <VECTOR>
        hivm.hir.load ins(%reinterpret_cast_0 : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%31 : memref<128xf32, #hivm.address_space<ub>>) {markDCacheInvalidatePatternVisited = 0 : i32} eviction_policy = <EvictFirst> core_type = <VECTOR>
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        %43 = arith.index_cast %arg12 : i32 to index
        %44 = affine.apply #map()[%42, %41, %43]
        %reinterpret_cast_1 = memref.reinterpret_cast %arg6 to offset: [%44], sizes: [128, 128], strides: [%43, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<128x128xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        %subview_2 = memref.subview %reinterpret_cast_1[%14, 0] [64, 128] [1, 1] {to_be_bubbled_slice} : memref<128x128xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<64x128xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 0
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %23]
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        func.call @scaled_mm_kernel_mix_aiv_outlined_vf_0(%32, %33, %31, %30) {hivm.vector_function, no_inline} : (memref<64xf32, #hivm.address_space<ub>>, memref<64x128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>, memref<64x128xf16, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %23]
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 1
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
        hivm.hir.pipe_barrier[<PIPE_MTE3>]
        hivm.hir.store ins(%30 : memref<64x128xf16, #hivm.address_space<ub>>) outs(%subview_2 : memref<64x128xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) {autoblockify.dma, tiled_op}
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %23]
      }
      %28 = arith.addi %18, %c1_i64 : i64
      memref.store %28, %alloca[%c0] : memref<1xi64>
    } {hivm.max_preload_num = 2 : i32}
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}


