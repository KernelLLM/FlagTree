module {
  func.func @run_28a39df095a89f01c24087bcac05d994(%arg0: !cannir.tensor<<gm, f8E4M3FN>, <<(2048, 7104)>, <(7104, 1)>>, <<(2048, 7104)>, <(7104, 1)>>, identity>, %arg1: !cannir.tensor<<gm, ui8>, <<(28416, 256)>, <(256, 1)>>, <<(28416, 256)>, <(256, 1)>>, identity>, %arg2: !cannir.tensor<<gm, !ascdtype.fp8_e8m0>, <<(2048, 112, 2)>, <(224, 2, 1)>>, <<(2048, 112, 2)>, <(224, 2, 1)>>, identity>, %arg3: !cannir.tensor<<gm, !ascdtype.fp8_e8m0>, <<(2048, 112, 2)>, <(224, 2, 1)>>, <<(2048, 112, 2)>, <(224, 2, 1)>>, identity>, %arg4: !cannir.tensor<<gm, f32>, <<(2048)>, <(1)>>, <<(2048)>, <(1)>>, identity>, %arg5: !cannir.tensor<<gm, f16>, <<(2048, 2048)>, <(2048, 1)>>, <<(2048, 2048)>, <(2048, 1)>>, identity>) attributes {cannbotdsl.contract_digest = "28a39df095a89f01c24087bcac05d994de372ab267e5b831d4c32e68f6c26823", cannbotdsl.public_slot_count = 6 : i64, cannbotdsl.slot_dtype_bits = [8, 8, 8, 8, 32, 16], cannbotdsl.slot_dtype_codes = [10, 1, 14, 14, 2, 2], cannbotdsl.slot_dtype_lanes = [1, 1, 1, 1, 1, 1], cannbotdsl.slot_flags = [1, 1, 1, 1, 1, 1], cannbotdsl.slot_kinds = [1, 1, 1, 1, 1, 1], cannbotdsl.slot_type_digests = ["c7a14dd26497020952e7291102dc5c47621187d74dc80d43edddf2235943811b", "c67001e4ecf88bc8322b03e741d8afc2b0fb836eb37b4f3eb3a82c079dd01d6a", "c41f8393debe596a95bb17a73684ae37b56ba0003fc9881bd0ba5ff0e66a75d4", "c41f8393debe596a95bb17a73684ae37b56ba0003fc9881bd0ba5ff0e66a75d4", "6c3d7517d574c41fea4825d514dae0ab479ec646b21ff7575066ae9635b55335", "049a29e5f7678a28747ba5a7f4a6171d45262a1ceb63cda35add3ae9a279b22e"], cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.host} {
    %c32_i64 = arith.constant 32 : i64
    cannir.kernel_launch @kernel_MixQuantKernel___call___0(%c32_i64) [%arg0, %arg1, %arg2, %arg3, %arg4, %arg5] {cannbotdsl.profile = {indices = array<i64: 0, 1, 2, 3, 4, 5>, name = "MixQuantKernel___call__", op_type = "MixQuantKernel___call__", roles = array<i64: 0, 0, 0, 0, 0, 1>}} : !cannir.tensor<<gm, f8E4M3FN>, <<(2048, 7104)>, <(7104, 1)>>, <<(2048, 7104)>, <(7104, 1)>>, identity>, !cannir.tensor<<gm, ui8>, <<(28416, 256)>, <(256, 1)>>, <<(28416, 256)>, <(256, 1)>>, identity>, !cannir.tensor<<gm, !ascdtype.fp8_e8m0>, <<(2048, 112, 2)>, <(224, 2, 1)>>, <<(2048, 112, 2)>, <(224, 2, 1)>>, identity>, !cannir.tensor<<gm, !ascdtype.fp8_e8m0>, <<(2048, 112, 2)>, <(224, 2, 1)>>, <<(2048, 112, 2)>, <(224, 2, 1)>>, identity>, !cannir.tensor<<gm, f32>, <<(2048)>, <(1)>>, <<(2048)>, <(1)>>, identity>, !cannir.tensor<<gm, f16>, <<(2048, 2048)>, <(2048, 1)>>, <<(2048, 2048)>, <(2048, 1)>>, identity>
    return
  }
  func.func @kernel_MixQuantKernel___call___0_aiv(%arg0: !cannir.tensor<<gm, f8E4M3FN>, <<(2048, 7104)>, <(7104, 1)>>, <<(2048, 7104)>, <(7104, 1)>>, identity>, %arg1: !cannir.tensor<<gm, ui8>, <<(28416, 256)>, <(256, 1)>>, <<(28416, 256)>, <(256, 1)>>, identity>, %arg2: !cannir.tensor<<gm, !ascdtype.fp8_e8m0>, <<(2048, 112, 2)>, <(224, 2, 1)>>, <<(2048, 112, 2)>, <(224, 2, 1)>>, identity>, %arg3: !cannir.tensor<<gm, !ascdtype.fp8_e8m0>, <<(2048, 112, 2)>, <(224, 2, 1)>>, <<(2048, 112, 2)>, <(224, 2, 1)>>, identity>, %arg4: !cannir.tensor<<gm, f32>, <<(2048)>, <(1)>>, <<(2048)>, <(1)>>, identity>, %arg5: !cannir.tensor<<gm, f16>, <<(2048, 2048)>, <(2048, 1)>>, <<(2048, 2048)>, <(2048, 1)>>, identity>) attributes {cannir.buffer.auto_sync, cannir.buffer.sync_analyzed, cannir.buffer.sync_verified, cannir.buffer.tensor_state, cannir.core = "aiv", cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.device, cannir.explicit_ring_buffer_cursors, cannir.kernel_group = "kernel_MixQuantKernel___call___0", cannir.resource.ids_prepared} {
    %c128_i32 = arith.constant 128 : i32
    %c0 = arith.constant 0 : index
    %c16_i64 = arith.constant 16 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %c2_i64 = arith.constant 2 : i64
    %c4_i64 = arith.constant 4 : i64
    %c8_i64 = arith.constant 8 : i64
    %c14_i64 = arith.constant 14 : i64
    %c524288_i64 = arith.constant 524288 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %c32 = arith.constant 32 : index
    %c4 = arith.constant 4 : index
    %c32_i64 = arith.constant 32 : i64
    %c16_i32 = arith.constant 16 : i32
    %c1_i32 = arith.constant 1 : i32
    %c1776_i64 = arith.constant 1776 : i64
    %c28416_i64 = arith.constant 28416 : i64
    %c4096_i64 = arith.constant 4096 : i64
    %c32768_i64 = arith.constant 32768 : i64
    %c222_i64 = arith.constant 222 : i64
    %c128_i64 = arith.constant 128 : i64
    %c0_i32 = arith.constant 0 : i32
    %c256_i64 = arith.constant 256 : i64
    %alloc = memref.alloc() {cannir.arg_index = 1 : i64} : memref<7274496xui8>
    %alloc_0 = memref.alloc() {cannir.arg_index = 4 : i64} : memref<2048xf32>
    %alloc_1 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<65536xui8, 1>
    %alloc_2 = memref.alloc() {cannir.byte_offset = 65536 : i64} : memref<131072xf8E4M3FN, 1>
    %alloc_3 = memref.alloc() {cannir.byte_offset = 196608 : i64} : memref<256xf32, 1>
    %alloc_4 = memref.alloc() {cannir.byte_offset = 131072 : i64} : memref<256xf32, 2>
    %alloc_5 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<131072xf8E4M3FN, 2>
    %0 = ascscalar.get_block_idx
    %1 = arith.muli %0, %c2_i64 : i64
    scf.for %arg6 = %c0_i64 to %c2_i64 step %c1_i64  : i64 {
      %2 = arith.addi %1, %arg6 : i64
      %3 = arith.divsi %2, %c8_i64 : i64
      %4 = ascscalar.get_subblock_id
      %5 = ascscalar.bitcast_memref %alloc_3 byte_offset = %c0_i64 : memref<256xf32, 1> -> memref<?xf32, 1>
      %6 = arith.muli %3, %c256_i64 : i64
      %7 = arith.addi %3, %c1_i64 : i64
      %8 = arith.muli %7, %c256_i64 : i64
      %9 = arith.subi %8, %6 : i64
      %10 = arith.muli %3, %c1024_i64 : i64
      %11 = arith.muli %9, %c4_i64 : i64
      %12 = arith.cmpi eq, %11, %c0_i64 : i64
      %13 = arith.cmpi ne, %11, %c0_i64 : i64
      %14 = arith.extui %13 : i1 to i64
      %15 = arith.select %12, %c4_i64, %11 : i64
      ascvec.get_buf mte2, %c2_i64, 0 {cannir.buffer.generated_sync}
      "ascvec.copy_gm2ub_plan"(%5, %alloc_0, %c0_i32, %c0_i32, %c0_i64, %10, %14, %15, %c0_i64, %15, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 8192 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 196608 : i64, ub_capacity = 1024 : i64}> : (memref<?xf32, 1>, memref<2048xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
      ascvec.release_buf mte2, %c2_i64, 0 {cannir.buffer.generated_sync}
      ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
      ascvec.vf(%alloc_3 : memref<256xf32, 1>) {
      ^bb0(%arg7: memref<256xf32, 1>):
        %c64_i32 = arith.constant 64 : i32
        %c256_i32 = arith.constant 256 : i32
        %c0_i32_6 = arith.constant 0 : i32
        %cst = arith.constant 1.562500e-02 : f32
        %28 = ascvec.create_mask : !ascvec.mask
        scf.for %arg8 = %c0_i32_6 to %c256_i32 step %c64_i32  : i32 {
          %29 = arith.index_cast %arg8 : i32 to index
          %30 = ascvec.loadalign %arg7[%29] : memref<256xf32, 1>, index -> <f32>
          %31 = ascvec.mul_scalar %30, %cst, %28 : <f32>, f32
          ascvec.storealign %arg7[%29], %31, %28 : memref<256xf32, 1>, index, <f32>
        }
        ascvec.mem_bar vst_vld
      } {cannir.vf_mode = "simd"}
      ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
      %16 = arith.muli %4, %c128_i64 : i64
      %17 = arith.addi %4, %c1_i64 : i64
      %18 = arith.muli %17, %c128_i64 : i64
      %19 = arith.subi %18, %16 : i64
      %20 = ascscalar.bitcast_memref %alloc_4 byte_offset = %c0_i64 : memref<256xf32, 2> -> memref<?xf32, 2>
      %21 = arith.divui %16, %c32_i64 : i64
      %22 = arith.muli %21, %c32_i64 : i64
      %23 = arith.index_cast %19 : i64 to index
      %24 = arith.muli %23, %c4 : index
      %25 = arith.index_cast %24 : index to i64
      %26 = arith.cmpi ugt, %25, %c0_i64 : i64
      ascvec.sync_intra_wait mte3, %c1_i64 {cannir.sync.banks = 3 : i32}
      scf.if %26 {
        %28 = arith.divui %24, %c32 : index
        %29 = arith.index_cast %28 : index to i64
        %30 = arith.cmpi ugt, %29, %c0_i64 : i64
        scf.if %30 {
          ascvec.get_buf mte3, %c2_i64, 0 {cannir.buffer.generated_sync}
          %31 = arith.index_cast %22 : i64 to index
          %32 = arith.trunci %29 : i64 to i32
          ascvec.copy_ub2l1_strided %20[%31], %5[%31], %c1_i32, %32, %c0_i32, %c0_i32 : memref<?xf32, 2>, memref<?xf32, 1>
          ascvec.release_buf mte3, %c2_i64, 0 {cannir.buffer.generated_sync}
        }
      }
      ascvec.sync_intra_arrive mte3, %c1_i64 {cannir.sync.banks = 3 : i32}
      %27 = scf.for %arg7 = %c0_i64 to %c14_i64 step %c1_i64 iter_args(%arg8 = %c0_i64) -> (i64)  : i64 {
        %28 = arith.addi %arg8, %c1_i64 overflow<nsw> : i64
        %29 = arith.remui %28, %c2_i64 : i64
        %30 = arith.muli %arg7, %c16_i64 overflow<nsw> : i64
        %31 = ascscalar.bitcast_memref %alloc_1 byte_offset = %c0_i64 : memref<65536xui8, 1> -> memref<?xui8, 1>
        %32 = arith.subi %c222_i64, %30 overflow<nsw> : i64
        %33 = arith.cmpi slt, %32, %c16_i64 : i64
        %34 = arith.select %33, %32, %c16_i64 : i64
        %35 = arith.cmpi eq, %34, %c16_i64 : i64
        scf.if %35 {
          %46 = arith.muli %3, %c4096_i64 : i64
          %47 = arith.muli %arg7, %c524288_i64 overflow<nsw> : i64
          %48 = arith.addi %47, %46 : i64
          ascvec.get_buf mte2, %c0_i64, 0 {cannir.buffer.generated_sync}
          "ascvec.copy_gm2ub_plan"(%31, %alloc, %c0_i32, %c0_i32, %c0_i64, %48, %c16_i64, %c4096_i64, %c32768_i64, %c4096_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 7274496 : i64, pad_storage_bits = false, storage_dtype = ui8, ub_base = 0 : i64, ub_capacity = 65536 : i64}> : (memref<?xui8, 1>, memref<7274496xui8>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
          ascvec.release_buf mte2, %c0_i64, 0 {cannir.buffer.generated_sync}
        } else {
          ascvec.get_buf mte2, %c0_i64, 0 {cannir.buffer.generated_sync}
          scf.for %arg9 = %c0_i64 to %34 step %c1_i64  : i64 {
            %46 = arith.addi %30, %arg9 overflow<nsw> : i64
            %47 = arith.muli %46, %c8_i64 overflow<nsw> : i64
            %48 = arith.addi %47, %3 : i64
            %49 = arith.minsi %48, %c1776_i64 : i64
            %50 = arith.muli %49, %c16_i64 : i64
            %51 = arith.subi %c28416_i64, %50 : i64
            %52 = arith.minsi %51, %c16_i64 : i64
            %53 = arith.cmpi sle, %48, %c1776_i64 : i64
            %54 = arith.select %53, %52, %c0_i64 : i64
            %55 = arith.muli %48, %c4096_i64 : i64
            %56 = arith.muli %arg9, %c4096_i64 overflow<nsw> : i64
            %57 = arith.muli %54, %c256_i64 : i64
            %58 = arith.cmpi eq, %57, %c0_i64 : i64
            %59 = arith.cmpi ne, %57, %c0_i64 : i64
            %60 = arith.extui %59 : i1 to i64
            %61 = arith.select %58, %c1_i64, %57 : i64
            "ascvec.copy_gm2ub_plan"(%31, %alloc, %c0_i32, %c0_i32, %56, %55, %60, %61, %c0_i64, %61, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 7274496 : i64, pad_storage_bits = false, storage_dtype = ui8, ub_base = 0 : i64, ub_capacity = 65536 : i64}> : (memref<?xui8, 1>, memref<7274496xui8>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
          }
          ascvec.release_buf mte2, %c0_i64, 0 {cannir.buffer.generated_sync}
        }
        %36 = arith.cmpi eq, %4, %c0_i64 : i64
        %37 = arith.select %36, %c0_i64, %c8_i64 : i64
        %38 = arith.addi %37, %c8_i64 overflow<nsw> : i64
        %39 = arith.muli %37, %c256_i64 overflow<nsw> : i64
        %40 = arith.trunci %34 : i64 to i32
        %41 = arith.trunci %39 : i64 to i32
        ascvec.get_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
        ascvec.get_buf v, %c1_i64, 0 {cannir.buffer.generated_sync}
        ascvec.vf(%37, %38, %alloc_1, %alloc_2, %40, %41 : i64, i64, memref<65536xui8, 1>, memref<131072xf8E4M3FN, 1>, i32, i32) {
        ^bb0(%arg9: i64, %arg10: i64, %arg11: memref<65536xui8, 1>, %arg12: memref<131072xf8E4M3FN, 1>, %arg13: i32, %arg14: i32):
          %c16_i32_6 = arith.constant 16 : i32
          %c0_i32_7 = arith.constant 0 : i32
          %c8192_i32 = arith.constant 8192 : i32
          %c4096_i32 = arith.constant 4096 : i32
          %c1_i32_8 = arith.constant 1 : i32
          %c512_i32 = arith.constant 512 : i32
          %c256_i32 = arith.constant 256 : i32
          %c0_i8 = arith.constant 0 : i8
          %c28_i8 = arith.constant 28 : i8
          %c-128_i8 = arith.constant -128 : i8
          %46 = ascvec.create_mask elem_bits 8 : !ascvec.mask
          %47 = ascvec.duplicate_scalar_nomask %c-128_i8 : i8 -> <ui8>
          %48 = ascvec.duplicate_scalar_nomask %c28_i8 : i8 -> <ui8>
          %49 = ascvec.duplicate_scalar_nomask %c0_i8 : i8 -> <ui8>
          %50 = ascvec.reinterpret %49 : <ui8> -> <f8E4M3FN>
          %51:2 = scf.for %arg15 = %c0_i32_7 to %arg13 step %c1_i32_8 iter_args(%arg16 = %arg14, %arg17 = %c0_i32_7) -> (i32, i32)  : i32 {
            %53 = arith.trunci %arg9 : i64 to i32
            %54 = arith.trunci %arg10 : i64 to i32
            %55:2 = scf.for %arg18 = %53 to %54 step %c1_i32_8 iter_args(%arg19 = %arg16, %arg20 = %arg17) -> (i32, i32)  : i32 {
              %58 = arith.index_cast %arg19 : i32 to index
              %59 = ascvec.loadalign %arg11[%58] : memref<65536xui8, 1>, index -> <ui8>
              %60 = ascvec.shl %59, 4, %46 : <ui8> -> <ui8>
              %61 = ascvec.and %60, %47, %46 : <ui8>, <ui8> -> <ui8>
              %62 = ascvec.shl %59, 2, %46 : <ui8> -> <ui8>
              %63 = ascvec.and %62, %48, %46 : <ui8>, <ui8> -> <ui8>
              %64 = ascvec.or %61, %63, %46 : <ui8>, <ui8> -> <ui8>
              %65 = ascvec.and %59, %47, %46 : <ui8>, <ui8> -> <ui8>
              %66 = ascvec.shr %59, 2, %46 : <ui8> -> <ui8>
              %67 = ascvec.and %66, %48, %46 : <ui8>, <ui8> -> <ui8>
              %68 = ascvec.or %65, %67, %46 : <ui8>, <ui8> -> <ui8>
              %69 = ascvec.reinterpret %64 : <ui8> -> <f8E4M3FN>
              %70 = ascvec.reinterpret %68 : <ui8> -> <f8E4M3FN>
              %71 = arith.index_cast %arg20 : i32 to index
              ascvec.storealign_intlv %arg12[%71], %69, %70 : memref<131072xf8E4M3FN, 1>, <f8E4M3FN>, <f8E4M3FN>
              %72 = arith.addi %arg19, %c256_i32 : i32
              %73 = arith.addi %arg20, %c512_i32 : i32
              scf.yield %72, %73 : i32, i32
            }
            %56 = arith.addi %arg16, %c4096_i32 : i32
            %57 = arith.addi %arg17, %c8192_i32 : i32
            scf.yield %56, %57 : i32, i32
          }
          %52 = scf.for %arg15 = %arg13 to %c16_i32_6 step %c1_i32_8 iter_args(%arg16 = %51#1) -> (i32)  : i32 {
            %53 = arith.trunci %arg9 : i64 to i32
            %54 = arith.trunci %arg10 : i64 to i32
            %55 = scf.for %arg17 = %53 to %54 step %c1_i32_8 iter_args(%arg18 = %arg16) -> (i32)  : i32 {
              %57 = arith.index_cast %arg18 : i32 to index
              ascvec.storealign_intlv %arg12[%57], %50, %50 : memref<131072xf8E4M3FN, 1>, <f8E4M3FN>, <f8E4M3FN>
              %58 = arith.addi %arg18, %c512_i32 : i32
              scf.yield %58 : i32
            }
            %56 = arith.addi %arg16, %c8192_i32 : i32
            scf.yield %56 : i32
          }
          ascvec.mem_bar vst_vld
        } {cannir.vf_mode = "simd"}
        ascvec.release_buf v, %c1_i64, 0 {cannir.buffer.generated_sync}
        ascvec.release_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
        %42 = ascscalar.bitcast_memref %alloc_2 byte_offset = %c0_i64 : memref<131072xf8E4M3FN, 1> -> memref<?xf8E4M3FN, 1>
        %43 = ascscalar.bitcast_memref %alloc_5 byte_offset = %c0_i64 : memref<131072xf8E4M3FN, 2> -> memref<?xf8E4M3FN, 2>
        %44 = arith.index_cast %16 : i64 to index
        %45 = arith.muli %44, %c32 : index
        ascvec.sync_intra_wait mte3, %c0_i64 {cannir.sync.banks = 3 : i32}
        ascvec.get_buf mte3, %c1_i64, 0 {cannir.buffer.generated_sync}
        ascvec.copy_ub2l1_strided %43[%45], %42[%c0], %c16_i32, %c128_i32, %c128_i32, %c128_i32 : memref<?xf8E4M3FN, 2>, memref<?xf8E4M3FN, 1>
        ascvec.release_buf mte3, %c1_i64, 0 {cannir.buffer.generated_sync}
        ascvec.sync_intra_arrive mte3, %c0_i64 {cannir.sync.banks = 3 : i32}
        scf.yield %29 : i64
      }
    }
    ascvec.sync_intra_wait mte3, %c0_i64
    ascvec.sync_intra_wait mte3, %c1_i64
    return
  }
  func.func @kernel_MixQuantKernel___call___0_aic(%arg0: !cannir.tensor<<gm, f8E4M3FN>, <<(2048, 7104)>, <(7104, 1)>>, <<(2048, 7104)>, <(7104, 1)>>, identity>, %arg1: !cannir.tensor<<gm, ui8>, <<(28416, 256)>, <(256, 1)>>, <<(28416, 256)>, <(256, 1)>>, identity>, %arg2: !cannir.tensor<<gm, !ascdtype.fp8_e8m0>, <<(2048, 112, 2)>, <(224, 2, 1)>>, <<(2048, 112, 2)>, <(224, 2, 1)>>, identity>, %arg3: !cannir.tensor<<gm, !ascdtype.fp8_e8m0>, <<(2048, 112, 2)>, <(224, 2, 1)>>, <<(2048, 112, 2)>, <(224, 2, 1)>>, identity>, %arg4: !cannir.tensor<<gm, f32>, <<(2048)>, <(1)>>, <<(2048)>, <(1)>>, identity>, %arg5: !cannir.tensor<<gm, f16>, <<(2048, 2048)>, <(2048, 1)>>, <<(2048, 2048)>, <(2048, 1)>>, identity>) attributes {cannir.buffer.auto_sync, cannir.buffer.sync_analyzed, cannir.buffer.sync_verified, cannir.buffer.tensor_state, cannir.core = "aic", cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.device, cannir.explicit_ring_buffer_cursors, cannir.kernel_group = "kernel_MixQuantKernel___call___0", cannir.resource.ids_prepared} {
    %c2048_i32 = arith.constant 2048 : i32
    %c2_i32 = arith.constant 2 : i32
    %c4_i32 = arith.constant 4 : i32
    %c114688 = arith.constant 114688 : index
    %c128_i32 = arith.constant 128 : i32
    %c8_i32 = arith.constant 8 : i32
    %c32_i32 = arith.constant 32 : i32
    %c0 = arith.constant 0 : index
    %c17_i64 = arith.constant 17 : i64
    %c16_i64 = arith.constant 16 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %c2_i64 = arith.constant 2 : i64
    %c3_i64 = arith.constant 3 : i64
    %c4_i64 = arith.constant 4 : i64
    %c5_i64 = arith.constant 5 : i64
    %c6_i64 = arith.constant 6 : i64
    %c7_i64 = arith.constant 7 : i64
    %c8_i64 = arith.constant 8 : i64
    %c9_i64 = arith.constant 9 : i64
    %c10_i64 = arith.constant 10 : i64
    %c13_i64 = arith.constant 13 : i64
    %c14_i64 = arith.constant 14 : i64
    %c8192 = arith.constant 8192 : index
    %c57344_i64 = arith.constant 57344 : i64
    %c2048_i64 = arith.constant 2048 : i64
    %c2 = arith.constant 2 : index
    %c64 = arith.constant 64 : index
    %c63 = arith.constant 63 : index
    %c31 = arith.constant 31 : index
    %c32 = arith.constant 32 : index
    %c3_i32 = arith.constant 3 : i32
    %cst = arith.constant 6.400000e+01 : f32
    %c55_i64 = arith.constant 55 : i64
    %c16_i32 = arith.constant 16 : i32
    %c512_i32 = arith.constant 512 : i32
    %c1_i32 = arith.constant 1 : i32
    %c7104_i64 = arith.constant 7104 : i64
    %c512_i64 = arith.constant 512 : i64
    %c224_i64 = arith.constant 224 : i64
    %c256_i32 = arith.constant 256 : i32
    %c0_i32 = arith.constant 0 : i32
    %c256_i64 = arith.constant 256 : i64
    %alloc = memref.alloc() {cannir.arg_index = 0 : i64} : memref<14548992xf8E4M3FN>
    %alloc_0 = memref.alloc() {cannir.arg_index = 2 : i64} : memref<458752x!ascdtype.fp8_e8m0>
    %alloc_1 = memref.alloc() {cannir.arg_index = 3 : i64} : memref<458752x!ascdtype.fp8_e8m0>
    %alloc_2 = memref.alloc() {cannir.arg_index = 5 : i64} : memref<4194304xf16>
    %alloc_3 = memref.alloc() {cannir.byte_offset = 131072 : i64} : memref<256xf32, 2>
    %alloc_4 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<256xf32, 8>
    %alloc_5 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<65536xf32, 5>
    %alloc_6 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<131072xf8E4M3FN, 2>
    %alloc_7 = memref.alloc() {cannir.byte_offset = 32768 : i64} : memref<32768xf8E4M3FN, 4>
    %alloc_8 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<32768xf8E4M3FN, 4>
    %alloc_9 = memref.alloc() {cannir.byte_offset = 32768 : i64} : memref<32768xf8E4M3FN, 3>
    %alloc_10 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<32768xf8E4M3FN, 3>
    %alloc_11 = memref.alloc() {cannir.byte_offset = 263168 : i64} : memref<131072xf8E4M3FN, 2>
    %alloc_12 = memref.alloc() {cannir.byte_offset = 132096 : i64} : memref<131072xf8E4M3FN, 2>
    %alloc_13 = memref.alloc() {cannir.byte_offset = 406528 : i64} : memref<4096x!ascdtype.fp8_e8m0, 2>
    %alloc_14 = memref.alloc() {cannir.byte_offset = 402432 : i64} : memref<4096x!ascdtype.fp8_e8m0, 2>
    %alloc_15 = memref.alloc() {cannir.byte_offset = 398336 : i64} : memref<4096x!ascdtype.fp8_e8m0, 2>
    %alloc_16 = memref.alloc() {cannir.byte_offset = 394240 : i64} : memref<4096x!ascdtype.fp8_e8m0, 2>
    asccube.sync_intra_arrive mte1, %c1_i64
    asccube.sync_intra_arrive mte1, %c17_i64
    asccube.sync_intra_arrive mte1, %c0_i64
    asccube.sync_intra_arrive mte1, %c16_i64
    %0 = ascscalar.get_block_idx
    %1 = arith.muli %0, %c2_i64 : i64
    scf.for %arg6 = %c0_i64 to %c2_i64 step %c1_i64  : i64 {
      %2 = arith.addi %1, %arg6 : i64
      %3 = arith.divsi %2, %c8_i64 : i64
      %4 = arith.muli %3, %c8_i64 : i64
      %5 = arith.subi %2, %4 : i64
      %6 = arith.muli %3, %c256_i64 : i64
      asccube.sync_intra_wait mte1, %c1_i64 {cannir.sync.banks = 3 : i32}
      asccube.sync_intra_wait mte1, %c17_i64 {cannir.sync.banks = 3 : i32}
      asccube.get_buf mte1, %c6_i64, 0 {cannir.buffer.generated_sync}
      asccube.copy_l12bt %alloc_4[%c0], %alloc_3[%c0], %c0_i32, %c1_i32, %c32_i32, %c0_i32, %c0_i32 : memref<256xf32, 8>, memref<256xf32, 2>
      asccube.release_buf mte1, %c6_i64, 0 {cannir.buffer.generated_sync}
      asccube.sync_intra_arrive mte1, %c1_i64 {cannir.sync.banks = 3 : i32}
      asccube.sync_intra_arrive mte1, %c17_i64 {cannir.sync.banks = 3 : i32}
      %7 = scf.for %arg7 = %c0_i64 to %c14_i64 step %c1_i64 iter_args(%arg8 = %c0_i64) -> (i64)  : i64 {
        %16 = arith.muli %5, %c256_i64 : i64
        %17 = arith.cmpi eq, %arg8, %c0_i64 : i64
        %18 = ascscalar.memref_address %alloc_16 : memref<4096x!ascdtype.fp8_e8m0, 2>
        %19 = ascscalar.memref_address %alloc_15 : memref<4096x!ascdtype.fp8_e8m0, 2>
        %20 = arith.select %17, %18, %19 : i64
        %21 = arith.select %17, %c2_i64, %c3_i64 : i64
        %22 = arith.addi %arg8, %c1_i64 overflow<nsw> : i64
        %23 = arith.remui %22, %c2_i64 : i64
        %24 = arith.muli %5, %c57344_i64 : i64
        %25 = arith.muli %arg7, %c16_i64 overflow<nsw> : i64
        %26 = arith.addi %24, %25 : i64
        %27 = ascscalar.bind_local_ptr %20 : memref<1x!ascdtype.fp8_e8m0, 2>
        asccube.get_buf mte2, %21, 0 {cannir.buffer.generated_sync}
        %28 = arith.index_cast %26 : i64 to index
        asccube.copy_gm2l1 %27[%c0], %alloc_0[%28], %c8_i32, %c256_i32, %c224_i64, %c1_i32, %c0_i64, %c0_i64, %c8_i32 l2_cache_ctl %c0_i32 {dn2nz = true, mx_scale = true} : memref<1x!ascdtype.fp8_e8m0, 2>, memref<458752x!ascdtype.fp8_e8m0>
        asccube.release_buf mte2, %21, 0 {cannir.buffer.generated_sync}
        %29 = ascscalar.memref_address %alloc_14 : memref<4096x!ascdtype.fp8_e8m0, 2>
        %30 = ascscalar.memref_address %alloc_13 : memref<4096x!ascdtype.fp8_e8m0, 2>
        %31 = arith.select %17, %29, %30 : i64
        %32 = arith.select %17, %c4_i64, %c5_i64 : i64
        %33 = arith.muli %3, %c57344_i64 : i64
        %34 = arith.addi %33, %25 : i64
        %35 = ascscalar.bind_local_ptr %31 : memref<1x!ascdtype.fp8_e8m0, 2>
        asccube.get_buf mte2, %32, 0 {cannir.buffer.generated_sync}
        %36 = arith.index_cast %34 : i64 to index
        asccube.copy_gm2l1 %35[%c0], %alloc_1[%36], %c8_i32, %c256_i32, %c224_i64, %c1_i32, %c0_i64, %c0_i64, %c8_i32 l2_cache_ctl %c0_i32 {dn2nz = true, mx_scale = true} : memref<1x!ascdtype.fp8_e8m0, 2>, memref<458752x!ascdtype.fp8_e8m0>
        asccube.release_buf mte2, %32, 0 {cannir.buffer.generated_sync}
        %37 = arith.muli %arg7, %c512_i64 overflow<nsw> : i64
        %38 = arith.subi %c7104_i64, %37 overflow<nsw> : i64
        %39 = arith.minsi %38, %c512_i64 : i64
        %40 = ascscalar.memref_address %alloc_12 : memref<131072xf8E4M3FN, 2>
        %41 = ascscalar.memref_address %alloc_11 : memref<131072xf8E4M3FN, 2>
        %42 = arith.select %17, %40, %41 : i64
        %43 = arith.cmpi ne, %arg8, %c0_i64 : i64
        %44 = arith.extui %43 : i1 to i64
        %45 = ascscalar.bind_local_ptr %42 : memref<1xf8E4M3FN, 2>
        %46 = arith.divui %16, %c8_i64 : i64
        %47 = arith.muli %46, %c7104_i64 : i64
        %48 = arith.divui %37, %c8_i64 : i64
        %49 = arith.addi %47, %48 : i64
        %50 = arith.muli %49, %c8_i64 : i64
        %51 = arith.index_cast %39 : i64 to index
        asccube.get_buf mte2, %44, 0 {cannir.buffer.generated_sync}
        %52 = arith.cmpi ugt, %39, %c0_i64 : i64
        scf.if %52 {
          %64 = arith.index_cast %50 : i64 to index
          %65 = arith.trunci %39 : i64 to i32
          asccube.copy_gm2l1 %45[%c0], %alloc[%64], %c256_i32, %65, %c7104_i64, %c1_i32, %c0_i64, %c0_i64, %c256_i32 l2_cache_ctl %c0_i32 {nd2nz = true} : memref<1xf8E4M3FN, 2>, memref<14548992xf8E4M3FN>
        }
        asccube.release_buf mte2, %44, 0 {cannir.buffer.generated_sync}
        %53 = arith.addi %51, %c31 : index
        %54 = arith.divui %53, %c32 : index
        %55 = arith.muli %54, %c32 : index
        %56 = arith.addi %51, %c63 : index
        %57 = arith.divui %56, %c64 : index
        %58 = arith.muli %57, %c64 : index
        %59 = arith.cmpi ugt, %58, %55 : index
        scf.if %59 {
          %64 = arith.muli %54, %c8192 : index
          asccube.get_buf mte2, %44, 0 {cannir.buffer.generated_sync}
          asccube.fill_l1 %45[%64], %c0_i32, %c1_i32, %c256_i32, %c0_i32 : memref<1xf8E4M3FN, 2>, i32
          asccube.release_buf mte2, %44, 0 {cannir.buffer.generated_sync}
        }
        %60 = ascscalar.bitcast_memref %alloc_6 byte_offset = %c0_i64 : memref<131072xf8E4M3FN, 2> -> memref<?xf8E4M3FN, 2>
        %61 = ascscalar.bind_local_ptr %42 : memref<131072xf8E4M3FN, 2>
        %62 = arith.cmpi eq, %arg7, %c13_i64 : i64
        scf.if %62 {
          asccube.get_buf mte2, %44, 0 {cannir.buffer.generated_sync}
          asccube.fill_l1 %61[%c114688], %c0_i32, %c1_i32, %c512_i32, %c0_i32 : memref<131072xf8E4M3FN, 2>, i32
          asccube.release_buf mte2, %44, 0 {cannir.buffer.generated_sync}
        }
        asccube.sync_intra_wait mte1, %c0_i64 {cannir.sync.banks = 3 : i32}
        asccube.sync_intra_wait mte1, %c16_i64 {cannir.sync.banks = 3 : i32}
        %63 = scf.for %arg9 = %c0_i64 to %c4_i64 step %c1_i64 iter_args(%arg10 = %c0_i64) -> (i64)  : i64 {
          %64 = arith.muli %arg7, %c4_i64 overflow<nsw> : i64
          %65 = arith.addi %64, %arg9 overflow<nsw> : i64
          %66 = arith.cmpi eq, %arg10, %c0_i64 : i64
          %67 = ascscalar.memref_address %alloc_10 : memref<32768xf8E4M3FN, 3>
          %68 = ascscalar.memref_address %alloc_9 : memref<32768xf8E4M3FN, 3>
          %69 = arith.select %66, %67, %68 : i64
          %70 = arith.select %66, %c7_i64, %c8_i64 : i64
          %71 = arith.addi %arg10, %c1_i64 overflow<nsw> : i64
          %72 = arith.remui %71, %c2_i64 : i64
          %73 = ascscalar.bind_local_ptr %69 : memref<1xf8E4M3FN, 3>
          %74 = arith.muli %arg9, %c4_i64 overflow<nsw> : i64
          %75 = arith.index_cast %74 : i64 to index
          %76 = arith.divui %75, %c2 : index
          %77 = arith.index_cast %76 : index to i64
          %78 = ascscalar.memref_address %73 : memref<1xf8E4M3FN, 3>
          %79 = ascscalar.bind_local_ptr %78 : memref<?xi8, 6>
          asccube.get_buf mte1, %70, 0 {cannir.buffer.generated_sync}
          asccube.get_buf mte1, %44, 0 {cannir.buffer.generated_sync}
          asccube.get_buf mte1, %21, 0 {cannir.buffer.generated_sync}
          %80 = arith.trunci %74 : i64 to i32
          asccube.copy_l12l0a %73[%c0], %45[%c0], %c0_i32, %80, %c16_i32, %c4_i32, %c16_i32, %c16_i32 : memref<1xf8E4M3FN, 3>, memref<1xf8E4M3FN, 2>
          %81 = arith.trunci %77 : i64 to i32
          asccube.copy_l12l0a %79[%c0], %27[%c0], %c0_i32, %81, %c16_i32, %c2_i32, %c8_i32, %c2_i32 : memref<?xi8, 6>, memref<1x!ascdtype.fp8_e8m0, 2>
          asccube.release_buf mte1, %21, 0 {cannir.buffer.generated_sync}
          asccube.release_buf mte1, %44, 0 {cannir.buffer.generated_sync}
          asccube.release_buf mte1, %70, 0 {cannir.buffer.generated_sync}
          %82 = ascscalar.memref_address %alloc_8 : memref<32768xf8E4M3FN, 4>
          %83 = ascscalar.memref_address %alloc_7 : memref<32768xf8E4M3FN, 4>
          %84 = arith.select %66, %82, %83 : i64
          %85 = arith.select %66, %c9_i64, %c10_i64 : i64
          %86 = ascscalar.bind_local_ptr %84 : memref<1xf8E4M3FN, 4>
          %87 = ascscalar.memref_address %86 : memref<1xf8E4M3FN, 4>
          %88 = ascscalar.bind_local_ptr %87 : memref<?xi8, 7>
          asccube.get_buf mte1, %85, 0 {cannir.buffer.generated_sync}
          asccube.copy_l12l0b %86[%c0], %60[%c0], %c0_i32, %80, %c16_i32, %c4_i32, %c16_i32, %c16_i32 : memref<1xf8E4M3FN, 4>, memref<?xf8E4M3FN, 2>
          asccube.get_buf mte1, %32, 0 {cannir.buffer.generated_sync}
          asccube.copy_l12l0b %88[%c0], %35[%c0], %c0_i32, %81, %c16_i32, %c2_i32, %c8_i32, %c2_i32 : memref<?xi8, 7>, memref<1x!ascdtype.fp8_e8m0, 2>
          asccube.release_buf mte1, %32, 0 {cannir.buffer.generated_sync}
          asccube.release_buf mte1, %85, 0 {cannir.buffer.generated_sync}
          %89 = arith.cmpi eq, %65, %c0_i64 : i64
          %90 = arith.cmpi eq, %65, %c55_i64 : i64
          %91 = arith.select %90, %c3_i64, %c2_i64 : i64
          %92 = arith.trunci %91 : i64 to i32
          %93 = ascscalar.bitcast_memref %alloc_5 byte_offset = %c0_i64 : memref<65536xf32, 5> -> memref<?xf32, 5>
          %94 = ascscalar.bitcast_memref %alloc_4 byte_offset = %c0_i64 : memref<256xf32, 8> -> memref<?xf32, 8>
          scf.if %89 {
            asccube.get_buf m, %70, 0 {cannir.buffer.generated_sync}
            asccube.get_buf m, %85, 0 {cannir.buffer.generated_sync}
            asccube.get_buf m, %c6_i64, 0 {cannir.buffer.generated_sync}
            asccube.mmad %93[%c0], %73, %86 scale(%79, %88) bias(%94[%c0] : memref<?xf32, 8>), %c256_i32, %c128_i32, %c256_i32, %92 {c_matrix_init_val = false, mx_mode = true} : memref<?xf32, 5>, memref<1xf8E4M3FN, 3>, memref<1xf8E4M3FN, 4>, memref<?xi8, 6>, memref<?xi8, 7>
            asccube.release_buf m, %c6_i64, 0 {cannir.buffer.generated_sync}
            asccube.release_buf m, %85, 0 {cannir.buffer.generated_sync}
            asccube.release_buf m, %70, 0 {cannir.buffer.generated_sync}
          } else {
            asccube.get_buf m, %70, 0 {cannir.buffer.generated_sync}
            asccube.get_buf m, %85, 0 {cannir.buffer.generated_sync}
            asccube.mmad %93[%c0], %73, %86 scale(%79, %88), %c256_i32, %c128_i32, %c256_i32, %92 {c_matrix_init_val = false, mx_mode = true} : memref<?xf32, 5>, memref<1xf8E4M3FN, 3>, memref<1xf8E4M3FN, 4>, memref<?xi8, 6>, memref<?xi8, 7>
            asccube.release_buf m, %85, 0 {cannir.buffer.generated_sync}
            asccube.release_buf m, %70, 0 {cannir.buffer.generated_sync}
          }
          scf.yield %72 : i64
        }
        asccube.sync_intra_arrive mte1, %c0_i64 {cannir.sync.banks = 3 : i32}
        asccube.sync_intra_arrive mte1, %c16_i64 {cannir.sync.banks = 3 : i32}
        scf.yield %23 : i64
      }
      %8 = arith.muli %5, %c256_i64 : i64
      %9 = ascscalar.bitcast_memref %alloc_5 byte_offset = %c0_i64 : memref<65536xf32, 5> -> memref<?xf32, 5>
      %10 = arith.divui %8, %c16_i64 : i64
      %11 = arith.muli %10, %c2048_i64 : i64
      %12 = arith.divui %6, %c16_i64 : i64
      %13 = arith.addi %11, %12 : i64
      %14 = arith.muli %13, %c16_i64 : i64
      %15 = arith.index_cast %14 : i64 to index
      asccube.copy_l0c2gm %alloc_2[%15], %9[%c0], %c256_i32, %c256_i32, %c2048_i32, %c256_i32, %c3_i32 deq_scale_val(%cst) l2_cache_ctl %c0_i32 {nz2nd_en = true, relu_en = false} : memref<4194304xf16>, memref<?xf32, 5>
    }
    return
  }
}
