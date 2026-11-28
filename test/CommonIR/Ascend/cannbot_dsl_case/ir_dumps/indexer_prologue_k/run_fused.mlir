module {
  func.func @run_fused_cdf78c432c2358da35cdc4a106fd02fc(%arg0: !cannir.tensor<<gm, bf16>, <<(-1, 512)>, <(512, 1)>>, <<(-1, 512)>, <(512, 1)>>, identity>, %arg1: !cannir.tensor<<gm, bf16>, <<(128, 512)>, <(512, 1)>>, <<(32, 8, 16, 16)>, <(2048, 256, 16, 1)>>, nd2nz> {cannir.physical_shape_upper_bound = array<i64: 32, 8, 16, 16>}, %arg2: !cannir.tensor<<gm, f32>, <<(128)>, <(1)>>, <<(128)>, <(1)>>, identity>, %arg3: !cannir.tensor<<gm, f32>, <<(-1, 64)>, <(64, 1)>>, <<(-1, 64)>, <(64, 1)>>, identity>, %arg4: !cannir.tensor<<gm, f32>, <<(-1, 64)>, <(64, 1)>>, <<(-1, 64)>, <(64, 1)>>, identity>, %arg5: !cannir.tensor<<gm, i64>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg6: !cannir.tensor<<gm, ui8>, <<(-1, -1, 1, 544)>, <(-1, -1, 544, 1)>>, <<(-1, -1, 1, 544)>, <(-1, -1, 544, 1)>>, identity>, %arg7: !cannir.tensor<<gm, ui8>, <<(1, 1, 1, 1)>, <(-1, -1, 544, 1)>>, <<(1, 1, 1, 1)>, <(-1, -1, 544, 1)>>, identity>, %arg8: !cannir.tensor<<gm, bf16>, <<(-1, 128)>, <(128, 1)>>, <<(-1, 128)>, <(128, 1)>>, identity>, %arg9: f32, %arg10: i32, %arg11: i32) attributes {cannbotdsl.contract_digest = "cdf78c432c2358da35cdc4a106fd02fc9a169357cf3e7ff4a51eea2ba5b8f5cb", cannbotdsl.public_slot_count = 12 : i64, cannbotdsl.slot_dtype_bits = [16, 16, 32, 32, 32, 64, 8, 8, 16, 32, 32, 32], cannbotdsl.slot_dtype_codes = [4, 4, 2, 2, 2, 0, 1, 1, 4, 2, 0, 0], cannbotdsl.slot_dtype_lanes = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], cannbotdsl.slot_flags = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], cannbotdsl.slot_kinds = [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2], cannbotdsl.slot_type_digests = ["fd71eefd3034fb92294714dd81a453c227277dd7fe4431db2d71d9c93f145f91", "e4da5bf029d8971d607127f9ca2f24263acbf38b983549721e913fdb7826ed28", "f61bdb43db5c16d55e8c5449243f360bac8ee1160d146fe829c02549663a3d81", "4f48935358a894f4695d225a04f44d14f74ad72d1e4104c3cede3cd1e1d8146f", "4f48935358a894f4695d225a04f44d14f74ad72d1e4104c3cede3cd1e1d8146f", "5ae7da10e83ac3f5ad55bbe47c978e5c8bcb0ad1d2171af81149eef24de96b7a", "89d149a355a0a1a38e4b7860f181f0951d9dc39862ff5e0e17b89698172baf71", "92d36351f13ddabe898da3b0783a92b3ebff868227d4c288ed509610f4b75f94", "3801b6a5735c1ecf9a96ef4492d52fb89b5f083e8311bd5ddacb10a191239548", "4a5363e2ec1eb8cefee037ab0b231653c07f9b44a406ef92ed9ba3e6f5541dbf", "7dacf501f968d2328c075490e60b2746783f658fa5c50bfc54eaeb02cb98a3a5", "7dacf501f968d2328c075490e60b2746783f658fa5c50bfc54eaeb02cb98a3a5"], cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.host} {
    %0 = arith.extsi %arg10 : i32 to i64
    cannir.kernel_launch @kernel_kernel_0(%0) [%arg0, %arg1, %arg8] {cannbotdsl.profile = {indices = array<i64: 0, 1, 2>, name = "kernel", op_type = "kernel", roles = array<i64: 0, 0, 1>}} : !cannir.tensor<<gm, bf16>, <<(-1, 512)>, <(512, 1)>>, <<(-1, 512)>, <(512, 1)>>, identity>, !cannir.tensor<<gm, bf16>, <<(128, 512)>, <(512, 1)>>, <<(32, 8, 16, 16)>, <(2048, 256, 16, 1)>>, nd2nz>, !cannir.tensor<<gm, bf16>, <<(-1, 128)>, <(128, 1)>>, <<(-1, 128)>, <(128, 1)>>, identity>
    %1 = arith.extsi %arg11 : i32 to i64
    cannir.kernel_launch @kernel__FusedPostScatterKernel___call___1(%1) [%arg8, %arg3, %arg4, %arg2, %arg5, %arg6, %arg7, %arg9] {cannbotdsl.profile = {indices = array<i64: 0, 1, 2, 3, 4, 5>, name = "_FusedPostScatterKernel___call__", op_type = "_FusedPostScatterKernel___call__", roles = array<i64: 0, 0, 0, 0, 0, 1>}} : !cannir.tensor<<gm, bf16>, <<(-1, 128)>, <(128, 1)>>, <<(-1, 128)>, <(128, 1)>>, identity>, !cannir.tensor<<gm, f32>, <<(-1, 64)>, <(64, 1)>>, <<(-1, 64)>, <(64, 1)>>, identity>, !cannir.tensor<<gm, f32>, <<(-1, 64)>, <(64, 1)>>, <<(-1, 64)>, <(64, 1)>>, identity>, !cannir.tensor<<gm, f32>, <<(128)>, <(1)>>, <<(128)>, <(1)>>, identity>, !cannir.tensor<<gm, i64>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, !cannir.tensor<<gm, ui8>, <<(-1, -1, 1, 544)>, <(-1, -1, 544, 1)>>, <<(-1, -1, 1, 544)>, <(-1, -1, 544, 1)>>, identity>, !cannir.tensor<<gm, ui8>, <<(1, 1, 1, 1)>, <(-1, -1, 544, 1)>>, <<(1, 1, 1, 1)>, <(-1, -1, 544, 1)>>, identity>, f32
    return
  }
  func.func @kernel_kernel_0(%arg0: !cannir.tensor<<gm, bf16>, <<(-1, 512)>, <(512, 1)>>, <<(-1, 512)>, <(512, 1)>>, identity>, %arg1: !cannir.tensor<<gm, bf16>, <<(128, 512)>, <(512, 1)>>, <<(32, 8, 16, 16)>, <(2048, 256, 16, 1)>>, nd2nz> {cannir.physical_shape_upper_bound = array<i64: 32, 8, 16, 16>}, %arg2: !cannir.tensor<<gm, bf16>, <<(-1, 128)>, <(128, 1)>>, <<(-1, 128)>, <(128, 1)>>, identity>) attributes {cannir.buffer.auto_sync, cannir.buffer.sync_analyzed, cannir.buffer.sync_verified, cannir.buffer.tensor_state, cannir.core = "aic", cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.device, cannir.explicit_ring_buffer_cursors, cannir.kernel_group = "kernel_kernel_0", cannir.resource.ids_prepared} {
    %c64_i32 = arith.constant 64 : i32
    %c4_i32 = arith.constant 4 : i32
    %c2048_i32 = arith.constant 2048 : i32
    %c8_i32 = arith.constant 8 : i32
    %c16_i32 = arith.constant 16 : i32
    %c1_i32 = arith.constant 1 : i32
    %c128_i32 = arith.constant 128 : i32
    %c0 = arith.constant 0 : index
    %c16_i64 = arith.constant 16 : i64
    %c0_i32 = arith.constant 0 : i32
    %c16 = arith.constant 16 : index
    %c512_i64 = arith.constant 512 : i64
    %c128_i64 = arith.constant 128 : i64
    %c64_i64 = arith.constant 64 : i64
    %c2048 = arith.constant 2048 : index
    %c256 = arith.constant 256 : index
    %c4096_i64 = arith.constant 4096 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %c8_i64 = arith.constant 8 : i64
    %c7_i64 = arith.constant 7 : i64
    %c6_i64 = arith.constant 6 : i64
    %c5_i64 = arith.constant 5 : i64
    %c4_i64 = arith.constant 4 : i64
    %c3_i64 = arith.constant 3 : i64
    %c2_i64 = arith.constant 2 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %alloc = memref.alloc() {cannir.arg_index = 0 : i64} : memref<1xbf16>
    %alloc_0 = memref.alloc() {cannir.arg_index = 1 : i64} : memref<65536xbf16>
    %alloc_1 = memref.alloc() {cannir.arg_index = 2 : i64} : memref<1xbf16>
    %alloc_2 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<1024xf32, 5>
    %alloc_3 = memref.alloc() {cannir.byte_offset = 16384 : i64} : memref<8192xbf16, 4>
    %alloc_4 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<8192xbf16, 4>
    %alloc_5 = memref.alloc() {cannir.byte_offset = 4096 : i64} : memref<2048xbf16, 3>
    %alloc_6 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<2048xbf16, 3>
    %alloc_7 = memref.alloc() {cannir.byte_offset = 24576 : i64} : memref<8192xbf16, 2>
    %alloc_8 = memref.alloc() {cannir.byte_offset = 8192 : i64} : memref<8192xbf16, 2>
    %alloc_9 = memref.alloc() {cannir.byte_offset = 4096 : i64} : memref<2048xbf16, 2>
    %alloc_10 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<2048xbf16, 2>
    %0 = ascscalar.get_block_idx
    %1 = ascscalar.get_block_num
    %2 = cannir.extract_layout %arg0 : !cannir.tensor<<gm, bf16>, <<(-1, 512)>, <(512, 1)>>, <<(-1, 512)>, <(512, 1)>>, identity>
    %3 = cannir.extract_shape %2 : <<(-1, 512)>, <(512, 1)>> -> <(-1, 512)>
    %4 = cannir.extract_dim %3[0] : !cannir.shape<(-1, 512)>
    %5 = arith.divsi %4, %c16_i64 : i64
    %6 = arith.remsi %4, %c16_i64 : i64
    %7 = arith.cmpi ne, %6, %c0_i64 : i64
    %8 = arith.extui %7 : i1 to i64
    %9 = arith.addi %5, %8 : i64
    %10 = arith.muli %9, %c2_i64 : i64
    scf.for %arg3 = %0 to %10 step %1  : i64 {
      %11 = arith.remsi %arg3, %c2_i64 : i64
      %12 = arith.cmpi ne, %11, %c0_i64 : i64
      %13 = arith.cmpi slt, %11, %c0_i64 : i64
      %14 = arith.andi %12, %13 : i1
      %15 = arith.divsi %arg3, %c2_i64 : i64
      %16 = arith.extui %14 : i1 to i64
      %17 = arith.subi %15, %16 : i64
      %18 = arith.select %14, %c2_i64, %c0_i64 : i64
      %19 = arith.addi %11, %18 overflow<nsw> : i64
      %20 = scf.for %arg4 = %c0_i64 to %c4_i64 step %c1_i64 iter_args(%arg5 = %c0_i64) -> (i64)  : i64 {
        %27 = arith.cmpi eq, %arg5, %c0_i64 : i64
        %28 = ascscalar.memref_address %alloc_10 : memref<2048xbf16, 2>
        %29 = ascscalar.memref_address %alloc_9 : memref<2048xbf16, 2>
        %30 = arith.select %27, %28, %29 : i64
        %31 = arith.cmpi eq, %arg5, %c0_i64 : i64
        %32 = arith.select %31, %c0_i64, %c1_i64 : i64
        %33 = arith.addi %arg5, %c1_i64 overflow<nsw> : i64
        %34 = arith.remui %33, %c2_i64 : i64
        %35 = ascscalar.bind_local_ptr %30 : memref<1xbf16, 2>
        %36 = arith.muli %17, %c512_i64 : i64
        %37 = arith.muli %arg4, %c8_i64 overflow<nsw> : i64
        %38 = arith.addi %36, %37 : i64
        %39 = arith.muli %38, %c16_i64 : i64
        %40 = cannir.extract_layout %arg0 : !cannir.tensor<<gm, bf16>, <<(-1, 512)>, <(512, 1)>>, <<(-1, 512)>, <(512, 1)>>, identity>
        %41 = cannir.extract_shape %40 : <<(-1, 512)>, <(512, 1)>> -> <(-1, 512)>
        %42 = cannir.extract_dim %41[0] : !cannir.shape<(-1, 512)>
        %43 = arith.divsi %42, %c16_i64 : i64
        %44 = arith.minsi %17, %43 : i64
        %45 = arith.muli %44, %c16_i64 : i64
        %46 = arith.subi %42, %45 : i64
        %47 = arith.minsi %46, %c16_i64 : i64
        %48 = arith.cmpi sle, %17, %43 : i64
        %49 = arith.select %48, %47, %c0_i64 : i64
        asccube.get_buf mte2, %32, 0 {cannir.buffer.generated_sync}
        %50 = arith.cmpi ugt, %49, %c0_i64 : i64
        scf.if %50 {
          %86 = arith.index_cast %39 : i64 to index
          %87 = arith.trunci %49 : i64 to i32
          asccube.copy_gm2l1 %35[%c0], %alloc[%86], %87, %c128_i32, %c1024_i64, %c1_i32, %c0_i64, %c0_i64, %c16_i32 l2_cache_ctl %c0_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
        }
        asccube.release_buf mte2, %32, 0 {cannir.buffer.generated_sync}
        %51 = ascscalar.memref_address %alloc_8 : memref<8192xbf16, 2>
        %52 = ascscalar.memref_address %alloc_7 : memref<8192xbf16, 2>
        %53 = arith.select %27, %51, %52 : i64
        %54 = arith.cmpi eq, %arg5, %c0_i64 : i64
        %55 = arith.select %54, %c2_i64, %c3_i64 : i64
        %56 = ascscalar.bind_local_ptr %53 : memref<1xbf16, 2>
        %57 = arith.muli %19, %c64_i64 overflow<nsw> : i64
        %58 = arith.index_cast %57 : i64 to index
        %59 = arith.muli %arg4, %c128_i64 overflow<nsw> : i64
        %60 = arith.index_cast %59 : i64 to index
        %61 = arith.divui %60, %c16 : index
        %62 = arith.muli %61, %c2048 : index
        %63 = arith.divui %58, %c16 : index
        %64 = arith.muli %63, %c256 : index
        %65 = arith.addi %62, %64 : index
        %66 = arith.index_cast %65 : index to i64
        asccube.get_buf mte2, %55, 0 {cannir.buffer.generated_sync}
        %67 = arith.index_cast %66 : i64 to index
        asccube.copy_gm2l1_burst %56[%c0], %alloc_0[%67], %c8_i32, %c2048_i32, %c4096_i64, %c2048_i32 l2_cache_ctl %c0_i32 pad_value %c0_i32 : i32 left_padding %c0_i32 right_padding %c0_i32 : memref<1xbf16, 2>, memref<65536xbf16>
        asccube.release_buf mte2, %55, 0 {cannir.buffer.generated_sync}
        %68 = ascscalar.memref_address %alloc_6 : memref<2048xbf16, 3>
        %69 = ascscalar.memref_address %alloc_5 : memref<2048xbf16, 3>
        %70 = arith.select %27, %68, %69 : i64
        %71 = arith.cmpi eq, %arg5, %c0_i64 : i64
        %72 = arith.select %71, %c4_i64, %c5_i64 : i64
        %73 = ascscalar.bind_local_ptr %70 : memref<1xbf16, 3>
        %74 = ascscalar.bind_local_ptr %30 : memref<1xbf16, 2>
        asccube.get_buf mte1, %72, 0 {cannir.buffer.generated_sync}
        asccube.get_buf mte1, %32, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0a %73[%c0], %74[%c0], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c1_i32, %c1_i32 : memref<1xbf16, 3>, memref<1xbf16, 2>
        asccube.release_buf mte1, %32, 0 {cannir.buffer.generated_sync}
        asccube.release_buf mte1, %72, 0 {cannir.buffer.generated_sync}
        %75 = ascscalar.memref_address %alloc_4 : memref<8192xbf16, 4>
        %76 = ascscalar.memref_address %alloc_3 : memref<8192xbf16, 4>
        %77 = arith.select %27, %75, %76 : i64
        %78 = arith.cmpi eq, %arg5, %c0_i64 : i64
        %79 = arith.select %78, %c6_i64, %c7_i64 : i64
        %80 = ascscalar.bind_local_ptr %77 : memref<1xbf16, 4>
        %81 = ascscalar.bind_local_ptr %53 : memref<1xbf16, 2>
        asccube.get_buf mte1, %79, 0 {cannir.buffer.generated_sync}
        asccube.get_buf mte1, %55, 0 {cannir.buffer.generated_sync}
        asccube.copy_l12l0b %80[%c0], %81[%c0], %c0_i32, %c0_i32, %c4_i32, %c8_i32, %c4_i32, %c4_i32 : memref<1xbf16, 4>, memref<1xbf16, 2>
        asccube.release_buf mte1, %55, 0 {cannir.buffer.generated_sync}
        asccube.release_buf mte1, %79, 0 {cannir.buffer.generated_sync}
        %82 = arith.cmpi eq, %arg4, %c0_i64 : i64
        %83 = ascscalar.bitcast_memref %alloc_2 byte_offset = %c0_i64 : memref<1024xf32, 5> -> memref<?xf32, 5>
        %84 = ascscalar.bind_local_ptr %70 : memref<1xbf16, 3>
        %85 = ascscalar.bind_local_ptr %77 : memref<1xbf16, 4>
        asccube.get_buf m, %c8_i64, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %72, 0 {cannir.buffer.generated_sync}
        asccube.get_buf m, %79, 0 {cannir.buffer.generated_sync}
        scf.if %82 {
          asccube.mmad %83[%c0], %84, %85, %c16_i32, %c128_i32, %c64_i32, %c0_i32 {c_matrix_init_val = true} : memref<?xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
        } else {
          asccube.mmad %83[%c0], %84, %85, %c16_i32, %c128_i32, %c64_i32, %c0_i32 {c_matrix_init_val = false} : memref<?xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
        }
        asccube.release_buf m, %79, 0 {cannir.buffer.generated_sync}
        asccube.release_buf m, %72, 0 {cannir.buffer.generated_sync}
        asccube.release_buf m, %c8_i64, 0 {cannir.buffer.generated_sync}
        scf.yield %34 : i64
      }
      %21 = ascscalar.bitcast_memref %alloc_2 byte_offset = %c0_i64 : memref<1024xf32, 5> -> memref<?xf32, 5>
      %22 = arith.muli %17, %c128_i64 : i64
      %23 = arith.muli %19, %c4_i64 overflow<nsw> : i64
      %24 = arith.addi %22, %23 : i64
      %25 = arith.muli %24, %c16_i64 : i64
      asccube.get_buf fixpipe, %c8_i64, 0 {cannir.buffer.generated_sync}
      %26 = arith.index_cast %25 : i64 to index
      asccube.copy_l0c2gm %alloc_1[%26], %21[%c0], %c64_i32, %c16_i32, %c128_i32, %c16_i32, %c0_i32 l2_cache_ctl %c0_i32 {nz2nd_en = true, relu_en = false} : memref<1xbf16>, memref<?xf32, 5>
      asccube.release_buf fixpipe, %c8_i64, 0 {cannir.buffer.generated_sync}
    }
    return
  }
  func.func @kernel__FusedPostScatterKernel___call___1(%arg0: !cannir.tensor<<gm, bf16>, <<(-1, 128)>, <(128, 1)>>, <<(-1, 128)>, <(128, 1)>>, identity>, %arg1: !cannir.tensor<<gm, f32>, <<(-1, 64)>, <(64, 1)>>, <<(-1, 64)>, <(64, 1)>>, identity>, %arg2: !cannir.tensor<<gm, f32>, <<(-1, 64)>, <(64, 1)>>, <<(-1, 64)>, <(64, 1)>>, identity>, %arg3: !cannir.tensor<<gm, f32>, <<(128)>, <(1)>>, <<(128)>, <(1)>>, identity>, %arg4: !cannir.tensor<<gm, i64>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg5: !cannir.tensor<<gm, ui8>, <<(-1, -1, 1, 544)>, <(-1, -1, 544, 1)>>, <<(-1, -1, 1, 544)>, <(-1, -1, 544, 1)>>, identity>, %arg6: !cannir.tensor<<gm, ui8>, <<(1, 1, 1, 1)>, <(-1, -1, 544, 1)>>, <<(1, 1, 1, 1)>, <(-1, -1, 544, 1)>>, identity>, %arg7: f32) attributes {cannir.buffer.auto_sync, cannir.buffer.sync_analyzed, cannir.buffer.sync_verified, cannir.buffer.tensor_state, cannir.core = "aiv", cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.device, cannir.explicit_ring_buffer_cursors, cannir.kernel_group = "kernel__FusedPostScatterKernel___call___1", cannir.resource.ids_prepared} {
    %c0_i32 = arith.constant 0 : i32
    %c512_i64 = arith.constant 512 : i64
    %c-1_i64 = arith.constant -1 : i64
    %c8_i64 = arith.constant 8 : i64
    %c256_i64 = arith.constant 256 : i64
    %c1_i32 = arith.constant 1 : i32
    %c64_i32 = arith.constant 64 : i32
    %c0 = arith.constant 0 : index
    %c4_i32 = arith.constant 4 : i32
    %c64_i64 = arith.constant 64 : i64
    %c6_i64 = arith.constant 6 : i64
    %c5_i64 = arith.constant 5 : i64
    %c4_i64 = arith.constant 4 : i64
    %c3_i64 = arith.constant 3 : i64
    %c2_i64 = arith.constant 2 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %alloc = memref.alloc() {cannir.arg_index = 0 : i64} : memref<1xbf16>
    %alloc_0 = memref.alloc() {cannir.arg_index = 1 : i64} : memref<1xf32>
    %alloc_1 = memref.alloc() {cannir.arg_index = 2 : i64} : memref<1xf32>
    %alloc_2 = memref.alloc() {cannir.arg_index = 3 : i64} : memref<128xf32>
    %alloc_3 = memref.alloc() {cannir.arg_index = 4 : i64} : memref<1xi64>
    %alloc_4 = memref.alloc() {cannir.arg_index = 5 : i64} : memref<1xui8>
    %alloc_5 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<128xbf16, 1>
    %alloc_6 = memref.alloc() {cannir.byte_offset = 256 : i64} : memref<128xbf16, 1>
    %alloc_7 = memref.alloc() {cannir.byte_offset = 512 : i64} : memref<192xbf16, 1>
    %alloc_8 = memref.alloc() {cannir.byte_offset = 896 : i64} : memref<128xui8, 1>
    %alloc_9 = memref.alloc() {cannir.byte_offset = 1024 : i64} : memref<64xui8, 1>
    %alloc_10 = memref.alloc() {cannir.byte_offset = 1088 : i64} : memref<4xui8, 1>
    %alloc_11 = memref.alloc() {cannir.byte_offset = 1120 : i64} : memref<256xui32, 1>
    %alloc_12 = memref.alloc() {cannir.byte_offset = 2144 : i64} : memref<128xf32, 1>
    %alloc_13 = memref.alloc() {cannir.byte_offset = 2656 : i64} : memref<256xf32, 1>
    %alloc_14 = memref.alloc() {cannir.byte_offset = 3680 : i64} : memref<64xf32, 1>
    %alloc_15 = memref.alloc() {cannir.byte_offset = 3936 : i64} : memref<64xf32, 1>
    %alloc_16 = memref.alloc() {cannir.byte_offset = 4192 : i64} : memref<192xf32, 1>
    ascvec.get_buf mte2, %c4_i64, 0 {cannir.buffer.generated_sync}
    "ascvec.copy_gm2ub_plan"(%alloc_12, %alloc_2, %c0_i32, %c0_i32, %c0_i64, %c0_i64, %c1_i64, %c512_i64, %c0_i64, %c512_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 512 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 2144 : i64, ub_capacity = 512 : i64}> : (memref<128xf32, 1>, memref<128xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
    ascvec.release_buf mte2, %c4_i64, 0 {cannir.buffer.generated_sync}
    %0 = ascscalar.get_block_idx
    %1 = ascscalar.get_block_num
    %2 = cannir.extract_layout %arg0 : !cannir.tensor<<gm, bf16>, <<(-1, 128)>, <(128, 1)>>, <<(-1, 128)>, <(128, 1)>>, identity>
    %3 = cannir.extract_shape %2 : <<(-1, 128)>, <(128, 1)>> -> <(-1, 128)>
    %4 = cannir.extract_dim %3[0] : !cannir.shape<(-1, 128)>
    %5 = scf.for %arg8 = %c0_i64 to %4 step %c1_i64 iter_args(%arg9 = %c0_i64) -> (i64)  : i64 {
      %6 = arith.index_cast %arg8 : i64 to index
      %7 = memref.load %alloc_3[%6] : memref<1xi64>
      %8 = arith.cmpi ne, %7, %c-1_i64 : i64
      %9 = scf.if %8 -> (i64) {
        %10 = cannir.extract_layout %arg5 : !cannir.tensor<<gm, ui8>, <<(-1, -1, 1, 544)>, <(-1, -1, 544, 1)>>, <<(-1, -1, 1, 544)>, <(-1, -1, 544, 1)>>, identity>
        %11 = cannir.extract_shape %10 : <<(-1, -1, 1, 544)>, <(-1, -1, 544, 1)>> -> <(-1, -1, 1, 544)>
        %12 = cannir.extract_dim %11[1] : !cannir.shape<(-1, -1, 1, 544)>
        %13 = arith.muli %12, %c8_i64 : i64
        %14 = arith.remsi %7, %13 : i64
        %15 = arith.cmpi ne, %14, %c0_i64 : i64
        %16 = arith.cmpi slt, %14, %c0_i64 : i64
        %17 = arith.cmpi slt, %13, %c0_i64 : i64
        %18 = arith.xori %16, %17 : i1
        %19 = arith.andi %15, %18 : i1
        %20 = arith.divsi %7, %13 : i64
        %21 = arith.extui %19 : i1 to i64
        %22 = arith.subi %20, %21 : i64
        %23 = arith.select %19, %13, %c0_i64 : i64
        %24 = arith.addi %14, %23 : i64
        %25 = arith.remsi %24, %c8_i64 : i64
        %26 = arith.cmpi ne, %25, %c0_i64 : i64
        %27 = arith.cmpi slt, %25, %c0_i64 : i64
        %28 = arith.andi %26, %27 : i1
        %29 = arith.divsi %24, %c8_i64 : i64
        %30 = arith.extui %28 : i1 to i64
        %31 = arith.subi %29, %30 : i64
        %32 = arith.remsi %13, %c8_i64 : i64
        %33 = arith.cmpi ne, %32, %c0_i64 : i64
        %34 = arith.cmpi slt, %32, %c0_i64 : i64
        %35 = arith.andi %33, %34 : i1
        %36 = arith.divsi %13, %c8_i64 : i64
        %37 = arith.extui %35 : i1 to i64
        %38 = arith.subi %36, %37 : i64
        %39 = arith.muli %22, %38 : i64
        %40 = arith.addi %39, %31 : i64
        %41 = arith.remsi %40, %1 : i64
        %42 = arith.cmpi ne, %41, %c0_i64 : i64
        %43 = arith.cmpi slt, %41, %c0_i64 : i64
        %44 = arith.cmpi slt, %1, %c0_i64 : i64
        %45 = arith.xori %43, %44 : i1
        %46 = arith.andi %42, %45 : i1
        %47 = arith.select %46, %1, %c0_i64 : i64
        %48 = arith.addi %41, %47 : i64
        %49 = arith.cmpi eq, %48, %0 : i64
        %50 = scf.if %49 -> (i64) {
          %51 = arith.cmpi eq, %arg9, %c0_i64 : i64
          %52 = ascscalar.memref_address %alloc_5 : memref<128xbf16, 1>
          %53 = ascscalar.memref_address %alloc_6 : memref<128xbf16, 1>
          %54 = arith.select %51, %52, %53 : i64
          %55 = arith.cmpi ne, %arg9, %c0_i64 : i64
          %56 = arith.extui %55 : i1 to i64
          %57 = arith.addi %arg9, %c1_i64 overflow<nsw> : i64
          %58 = arith.remui %57, %c2_i64 : i64
          %59 = arith.minsi %arg8, %4 : i64
          %60 = arith.subi %4, %59 : i64
          %61 = arith.minsi %60, %c1_i64 : i64
          %62 = arith.cmpi sle, %arg8, %4 : i64
          %63 = arith.select %62, %61, %c0_i64 : i64
          %64 = arith.muli %arg8, %c256_i64 : i64
          %65 = arith.muli %63, %c256_i64 : i64
          %66 = arith.cmpi eq, %65, %c0_i64 : i64
          %67 = arith.cmpi ne, %65, %c0_i64 : i64
          %68 = arith.extui %67 : i1 to i64
          %69 = arith.select %66, %c2_i64, %65 : i64
          %70 = ascscalar.bind_local_ptr %54 : memref<1xbf16, 1>
          ascvec.get_buf mte2, %56, 0 {cannir.buffer.generated_sync}
          "ascvec.copy_gm2ub_plan"(%70, %alloc, %c0_i32, %c0_i32, %c0_i64, %64, %68, %69, %c0_i64, %69, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
          ascvec.release_buf mte2, %56, 0 {cannir.buffer.generated_sync}
          %71 = cannir.extract_layout %arg1 : !cannir.tensor<<gm, f32>, <<(-1, 64)>, <(64, 1)>>, <<(-1, 64)>, <(64, 1)>>, identity>
          %72 = cannir.extract_shape %71 : <<(-1, 64)>, <(64, 1)>> -> <(-1, 64)>
          %73 = cannir.extract_dim %72[0] : !cannir.shape<(-1, 64)>
          %74 = arith.minsi %arg8, %73 : i64
          %75 = arith.subi %73, %74 : i64
          %76 = arith.minsi %75, %c1_i64 : i64
          %77 = arith.cmpi sle, %arg8, %73 : i64
          %78 = arith.select %77, %76, %c0_i64 : i64
          %79 = arith.muli %78, %c256_i64 : i64
          %80 = arith.cmpi eq, %79, %c0_i64 : i64
          %81 = arith.cmpi ne, %79, %c0_i64 : i64
          %82 = arith.extui %81 : i1 to i64
          %83 = arith.select %80, %c4_i64, %79 : i64
          ascvec.get_buf mte2, %c5_i64, 0 {cannir.buffer.generated_sync}
          "ascvec.copy_gm2ub_plan"(%alloc_14, %alloc_0, %c0_i32, %c0_i32, %c0_i64, %64, %82, %83, %c0_i64, %83, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 3680 : i64, ub_capacity = 256 : i64}> : (memref<64xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
          ascvec.release_buf mte2, %c5_i64, 0 {cannir.buffer.generated_sync}
          %84 = cannir.extract_layout %arg2 : !cannir.tensor<<gm, f32>, <<(-1, 64)>, <(64, 1)>>, <<(-1, 64)>, <(64, 1)>>, identity>
          %85 = cannir.extract_shape %84 : <<(-1, 64)>, <(64, 1)>> -> <(-1, 64)>
          %86 = cannir.extract_dim %85[0] : !cannir.shape<(-1, 64)>
          %87 = arith.minsi %arg8, %86 : i64
          %88 = arith.subi %86, %87 : i64
          %89 = arith.minsi %88, %c1_i64 : i64
          %90 = arith.cmpi sle, %arg8, %86 : i64
          %91 = arith.select %90, %89, %c0_i64 : i64
          %92 = arith.muli %91, %c256_i64 : i64
          %93 = arith.cmpi eq, %92, %c0_i64 : i64
          %94 = arith.cmpi ne, %92, %c0_i64 : i64
          %95 = arith.extui %94 : i1 to i64
          %96 = arith.select %93, %c4_i64, %92 : i64
          ascvec.get_buf mte2, %c6_i64, 0 {cannir.buffer.generated_sync}
          "ascvec.copy_gm2ub_plan"(%alloc_15, %alloc_1, %c0_i32, %c0_i32, %c0_i64, %64, %95, %96, %c0_i64, %96, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 3936 : i64, ub_capacity = 256 : i64}> : (memref<64xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
          ascvec.release_buf mte2, %c6_i64, 0 {cannir.buffer.generated_sync}
          %97 = ascscalar.bind_local_ptr %54 : memref<128xbf16, 1>
          ascvec.get_buf v, %56, 0 {cannir.buffer.generated_sync}
          ascvec.get_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
          ascvec.vf(%97, %arg7, %alloc_12, %alloc_7 : memref<128xbf16, 1>, f32, memref<128xf32, 1>, memref<192xbf16, 1>) {
          ^bb0(%arg10: memref<128xbf16, 1>, %arg11: f32, %arg12: memref<128xf32, 1>, %arg13: memref<192xbf16, 1>):
            %cst = arith.constant 1.000000e+00 : f32
            %cst_17 = arith.constant 7.812500e-03 : f32
            %c2_i32 = arith.constant 2 : i32
            %c0_i32_18 = arith.constant 0 : i32
            %c64_i32_19 = arith.constant 64 : i32
            %cst_20 = arith.constant 0.000000e+00 : f32
            %c1_i32_21 = arith.constant 1 : i32
            %131 = ascvec.create_mask : !ascvec.mask
            %mask, %remaining_out = ascvec.update_mask %c1_i32_21 : !ascvec.mask
            %132 = ascvec.duplicate_scalar %cst_20, %mask : f32 -> <f32>
            %133 = scf.for %arg14 = %c0_i32_18 to %c2_i32 step %c1_i32_21 iter_args(%arg15 = %132) -> (!ascvec.vreg<f32>)  : i32 {
              %140 = arith.muli %arg14, %c64_i32_19 overflow<nsw> : i32
              %mask_22, %remaining_out_23 = ascvec.update_mask %c64_i32_19 : !ascvec.mask
              %141 = arith.index_cast %140 : i32 to index
              %142 = ascvec.loadalign_unpack %arg10[%141] {mode = "b16_to_b32"} : memref<128xbf16, 1>, index -> <bf16>
              %143 = ascvec.cast %142, %mask_22 {rounding = 6 : i32} : <bf16> -> <f32>
              %144 = ascvec.mul %143, %143, %mask_22 : <f32>
              %145 = ascvec.reduce_sum %144, %mask_22 : <f32>
              %146 = ascvec.add %arg15, %145, %mask : <f32>
              scf.yield %146 : !ascvec.vreg<f32>
            }
            %134 = ascvec.mul_scalar %133, %cst_17, %mask : <f32>, f32
            %135 = ascvec.add_scalar %134, %arg11, %mask : <f32>, f32
            %136 = ascvec.sqrt %135, %mask : <f32>
            %137 = ascvec.duplicate_scalar %cst, %mask : f32 -> <f32>
            %138 = ascvec.div %137, %136, %mask : <f32>
            %139 = ascvec.duplicate %138, %131 : <f32> -> <f32>
            scf.for %arg14 = %c0_i32_18 to %c2_i32 step %c1_i32_21  : i32 {
              %140 = arith.muli %arg14, %c64_i32_19 overflow<nsw> : i32
              %mask_22, %remaining_out_23 = ascvec.update_mask %c64_i32_19 : !ascvec.mask
              %141 = arith.index_cast %140 : i32 to index
              %142 = ascvec.loadalign_unpack %arg10[%141] {mode = "b16_to_b32"} : memref<128xbf16, 1>, index -> <bf16>
              %143 = ascvec.cast %142, %mask_22 {rounding = 6 : i32} : <bf16> -> <f32>
              %144 = ascvec.loadalign %arg12[%141] : memref<128xf32, 1>, index -> <f32>
              %145 = ascvec.mul %143, %139, %mask_22 : <f32>
              %146 = ascvec.mul %145, %144, %mask_22 : <f32>
              %147 = ascvec.cast %146, %mask_22 : <f32> -> <bf16>
              ascvec.storealign_pack %arg13[%141], %147, %mask_22 {mode = "b32_to_b16"} : memref<192xbf16, 1>, index, <bf16>
            }
            ascvec.mem_bar vst_vld
          } {cannir.vf_mode = "simd"}
          ascvec.release_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
          ascvec.release_buf v, %56, 0 {cannir.buffer.generated_sync}
          ascvec.vf(%alloc_7, %alloc_13 : memref<192xbf16, 1>, memref<256xf32, 1>) {
          ^bb0(%arg10: memref<192xbf16, 1>, %arg11: memref<256xf32, 1>):
            %c64 = arith.constant 64 : index
            %c64_i32_17 = arith.constant 64 : i32
            %c0_18 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            scf.for %arg12 = %c0_18 to %c1 step %c1 {
              %mask, %remaining_out = ascvec.update_mask %c64_i32_17 : !ascvec.mask
              %131 = ascvec.loadalign_unpack %arg10[%c64] {mode = "b16_to_b32"} : memref<192xbf16, 1>, index -> <bf16>
              %132 = ascvec.cast %131, %mask {rounding = 6 : i32} : <bf16> -> <f32>
              ascvec.storealign %arg11[%c64], %132, %mask : memref<256xf32, 1>, index, <f32>
            }
            ascvec.mem_bar vst_vld
          } {cannir.vf_mode = "simd"}
          ascvec.get_buf v, %c6_i64, 0 {cannir.buffer.generated_sync}
          ascvec.get_buf v, %c5_i64, 0 {cannir.buffer.generated_sync}
          ascvec.vf(%alloc_13, %alloc_15, %alloc_14, %alloc_16 : memref<256xf32, 1>, memref<64xf32, 1>, memref<64xf32, 1>, memref<192xf32, 1>) {
          ^bb0(%arg10: memref<256xf32, 1>, %arg11: memref<64xf32, 1>, %arg12: memref<64xf32, 1>, %arg13: memref<192xf32, 1>):
            %c64 = arith.constant 64 : index
            %c0_i32_17 = arith.constant 0 : i32
            %c0_18 = arith.constant 0 : index
            %c64_i32_19 = arith.constant 64 : i32
            %c32_i32 = arith.constant 32 : i32
            %131 = ascvec.create_mask : !ascvec.mask
            %mask, %remaining_out = ascvec.update_mask %c32_i32 : !ascvec.mask
            %mask_20, %remaining_out_21 = ascvec.update_mask %c64_i32_19 : !ascvec.mask
            %result0, %result1 = ascvec.loadalign_deintlv %arg10[%c64] {dintlv_width = "b32"} : memref<256xf32, 1>, index -> <f32>, <f32>
            %result0_22, %result1_23 = ascvec.loadalign_deintlv %arg11[%c0_18] {dintlv_width = "b32"} : memref<64xf32, 1>, index -> <f32>, <f32>
            %result0_24, %result1_25 = ascvec.loadalign_deintlv %arg12[%c0_18] {dintlv_width = "b32"} : memref<64xf32, 1>, index -> <f32>, <f32>
            %132 = ascvec.mul %result0, %result0_22, %mask : <f32>
            %133 = ascvec.mul %result1, %result0_24, %mask : <f32>
            %134 = ascvec.sub %132, %133, %mask : <f32>
            %135 = ascvec.mul %result1, %result1_23, %mask : <f32>
            %136 = ascvec.mul %result0, %result1_25, %mask : <f32>
            %137 = ascvec.add %135, %136, %mask : <f32>
            %low, %high = ascvec.intlv %134, %137 : <f32>, <f32> -> <f32>, <f32>
            ascvec.storealign %arg13[%c64], %low, %mask_20 : memref<192xf32, 1>, index, <f32>
            ascvec.mem_bar vst_vld
            %c0_26 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            scf.for %arg14 = %c0_26 to %c1 step %c1 {
              %mask_27, %remaining_out_28 = ascvec.update_mask %c64_i32_19 : !ascvec.mask
              %mask_29, %remaining_out_30 = ascvec.update_mask %c0_i32_17 : !ascvec.mask
              %138 = ascvec.mask_xor %mask_27, %mask_29, %131
              ascvec.loadunalign_init %arg13[%c64] : memref<192xf32, 1>
              %139 = ascvec.loadunalign %arg13[%c64] : memref<192xf32, 1> -> <f32>
              %140 = ascvec.loadalign %arg10[%c64] : memref<256xf32, 1>, index -> <f32>
              %141 = ascvec.select %139, %140, %138 : <f32>
              ascvec.storealign %arg10[%c64], %141, %mask_27 : memref<256xf32, 1>, index, <f32>
            }
            ascvec.mem_bar vst_vld
          } {cannir.vf_mode = "simd"}
          ascvec.release_buf v, %c5_i64, 0 {cannir.buffer.generated_sync}
          ascvec.release_buf v, %c6_i64, 0 {cannir.buffer.generated_sync}
          ascvec.vf(%alloc_13, %alloc_7 : memref<256xf32, 1>, memref<192xbf16, 1>) {
          ^bb0(%arg10: memref<256xf32, 1>, %arg11: memref<192xbf16, 1>):
            %c64 = arith.constant 64 : index
            %c64_i32_17 = arith.constant 64 : i32
            %c0_18 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            scf.for %arg12 = %c0_18 to %c1 step %c1 {
              %mask, %remaining_out = ascvec.update_mask %c64_i32_17 : !ascvec.mask
              %131 = ascvec.loadalign %arg10[%c64] : memref<256xf32, 1>, index -> <f32>
              %132 = ascvec.cast %131, %mask : <f32> -> <bf16>
              ascvec.storealign_pack %arg11[%c64], %132, %mask {mode = "b32_to_b16"} : memref<192xbf16, 1>, index, <bf16>
            }
          } {cannir.vf_mode = "simd"}
          ascvec.vf(%alloc_7, %alloc_11, %alloc_8 : memref<192xbf16, 1>, memref<256xui32, 1>, memref<128xui8, 1>) {
          ^bb0(%arg10: memref<192xbf16, 1>, %arg11: memref<256xui32, 1>, %arg12: memref<128xui8, 1>):
            %c128_i32 = arith.constant 128 : i32
            %c64_i32_17 = arith.constant 64 : i32
            %cst = arith.constant 2.500000e-01 : f32
            %cst_18 = arith.constant 7.500000e-01 : f32
            %cst_19 = arith.constant 1.250000e+00 : f32
            %c2_i32 = arith.constant 2 : i32
            %cst_20 = arith.constant 1.750000e+00 : f32
            %c3_i32 = arith.constant 3 : i32
            %cst_21 = arith.constant 2.500000e+00 : f32
            %c4_i32_22 = arith.constant 4 : i32
            %cst_23 = arith.constant 3.500000e+00 : f32
            %c5_i32 = arith.constant 5 : i32
            %cst_24 = arith.constant 5.000000e+00 : f32
            %c6_i32 = arith.constant 6 : i32
            %c7_i32 = arith.constant 7 : i32
            %cst_25 = arith.constant 0.166666672 : f32
            %cst_26 = arith.constant 7.0529661E-38 : f32
            %c255_i32 = arith.constant 255 : i32
            %c8388607_i32 = arith.constant 8388607 : i32
            %cst_27 = arith.constant 0.000000e+00 : f32
            %c8_i32 = arith.constant 8 : i32
            %c0_i32_28 = arith.constant 0 : i32
            %c1_i32_29 = arith.constant 1 : i32
            %c16_i32 = arith.constant 16 : i32
            %c32_i32 = arith.constant 32 : i32
            %mask, %remaining_out = ascvec.update_mask %c32_i32 : !ascvec.mask
            %mask_30, %remaining_out_31 = ascvec.update_mask %c16_i32 : !ascvec.mask
            %mask_32, %remaining_out_33 = ascvec.update_mask %c32_i32 : !ascvec.mask
            %mask_34, %remaining_out_35 = ascvec.update_mask %c1_i32_29 : !ascvec.mask
            %131 = ascvec.duplicate_scalar %c1_i32_29, %mask : i32 -> <ui32>
            %132 = ascvec.duplicate_scalar %c0_i32_28, %mask : i32 -> <ui32>
            %133 = ascvec.duplicate_scalar %c8_i32, %mask : i32 -> <ui32>
            %134 = ascvec.duplicate_scalar %cst_27, %mask : f32 -> <f32>
            %135 = ascvec.duplicate_scalar %c8388607_i32, %mask : i32 -> <ui32>
            %136 = ascvec.duplicate_scalar %c255_i32, %mask : i32 -> <ui32>
            scf.for %arg13 = %c0_i32_28 to %c4_i32_22 step %c1_i32_29  : i32 {
              %139 = arith.muli %arg13, %c32_i32 overflow<nsw> : i32
              %140 = arith.index_cast %139 : i32 to index
              %141 = ascvec.loadalign_unpack %arg10[%140] {mode = "b16_to_b32"} : memref<192xbf16, 1>, index -> <bf16>
              %142 = ascvec.cast %141, %mask {rounding = 6 : i32} : <bf16> -> <f32>
              %143 = ascvec.abs %142, %mask : <f32>
              %144 = ascvec.reduce_max %143, %mask : <f32>
              %145 = ascvec.max_scalar %144, %cst_26, %mask : <f32>, f32
              %146 = ascvec.mul_scalar %145, %cst_25, %mask : <f32>, f32
              %147 = ascvec.reinterpret %146 : <f32> -> <ui32>
              %148 = ascvec.shr %147, 23, %mask : <ui32> -> <ui32>
              %149 = ascvec.and %148, %136, %mask : <ui32>, <ui32> -> <ui32>
              %150 = ascvec.and %147, %135, %mask : <ui32>, <ui32> -> <ui32>
              %151 = ascvec.cmp_ne %150, %132, %mask : <ui32>, <ui32> -> !ascvec.mask
              %152 = ascvec.select %131, %132, %151 : <ui32>
              %153 = ascvec.add %149, %152, %mask : <ui32>
              %154 = ascvec.shl %153, 23, %mask : <ui32> -> <ui32>
              %155 = ascvec.reinterpret %154 : <ui32> -> <f32>
              %156 = ascvec.duplicate %155, %mask : <f32> -> <f32>
              %157 = ascvec.div %143, %156, %mask : <f32>
              %158 = ascvec.duplicate_scalar %c7_i32, %mask : i32 -> <ui32>
              %159 = ascvec.duplicate_scalar %c6_i32, %mask : i32 -> <ui32>
              %160 = ascvec.cmp_le_scalar %157, %cst_24, %mask : <f32>, f32 -> !ascvec.mask
              %161 = ascvec.select %159, %158, %160 : <ui32>
              %162 = ascvec.duplicate_scalar %c5_i32, %mask : i32 -> <ui32>
              %163 = ascvec.cmp_le_scalar %157, %cst_23, %mask : <f32>, f32 -> !ascvec.mask
              %164 = ascvec.select %162, %161, %163 : <ui32>
              %165 = ascvec.duplicate_scalar %c4_i32_22, %mask : i32 -> <ui32>
              %166 = ascvec.cmp_le_scalar %157, %cst_21, %mask : <f32>, f32 -> !ascvec.mask
              %167 = ascvec.select %165, %164, %166 : <ui32>
              %168 = ascvec.duplicate_scalar %c3_i32, %mask : i32 -> <ui32>
              %169 = ascvec.cmp_le_scalar %157, %cst_20, %mask : <f32>, f32 -> !ascvec.mask
              %170 = ascvec.select %168, %167, %169 : <ui32>
              %171 = ascvec.duplicate_scalar %c2_i32, %mask : i32 -> <ui32>
              %172 = ascvec.cmp_le_scalar %157, %cst_19, %mask : <f32>, f32 -> !ascvec.mask
              %173 = ascvec.select %171, %170, %172 : <ui32>
              %174 = ascvec.cmp_le_scalar %157, %cst_18, %mask : <f32>, f32 -> !ascvec.mask
              %175 = ascvec.select %131, %173, %174 : <ui32>
              %176 = ascvec.cmp_le_scalar %157, %cst, %mask : <f32>, f32 -> !ascvec.mask
              %177 = ascvec.select %132, %175, %176 : <ui32>
              %178 = ascvec.cmp_lt %142, %134, %mask : <f32>, <f32> -> !ascvec.mask
              %179 = ascvec.select %133, %132, %178 : <ui32>
              %180 = ascvec.add %177, %179, %mask : <ui32>
              %result0, %result1 = ascvec.deintlv %180, %180 : <ui32>, <ui32> -> <ui32>, <ui32>
              %181 = ascvec.shl %result1, 4, %mask_30 : <ui32> -> <ui32>
              %182 = ascvec.or %result0, %181, %mask_30 : <ui32>, <ui32> -> <ui32>
              %183 = arith.muli %arg13, %c64_i32_17 overflow<nsw> : i32
              %184 = arith.index_cast %183 : i32 to index
              ascvec.storealign %arg11[%184], %182, %mask_30 : memref<256xui32, 1>, index, <ui32>
              %185 = ascvec.cast %153, %mask_34 {reg_layout = 0 : i32} : <ui32> -> <ui8>
              %186 = arith.addi %arg13, %c64_i32_17 overflow<nsw> : i32
              %187 = arith.index_cast %186 : i32 to index
              ascvec.storealign_1st %arg12[%187], %185 : memref<128xui8, 1>, index, <ui8>
            }
            %mask_36, %remaining_out_37 = ascvec.update_mask %c16_i32 : !ascvec.mask
            %137 = ascvec.create_mask : !ascvec.mask
            %138 = ascvec.mask_xor %mask_32, %mask_36, %137
            ascvec.mem_bar vst_vld
            scf.for %arg13 = %c0_i32_28 to %c2_i32 step %c1_i32_29  : i32 {
              %139 = arith.muli %arg13, %c2_i32 overflow<nsw> : i32
              %140 = arith.muli %arg13, %c128_i32 overflow<nsw> : i32
              %141 = arith.index_cast %140 : i32 to index
              %142 = ascvec.loadalign %arg11[%141] : memref<256xui32, 1>, index -> <ui32>
              %143 = arith.addi %139, %c1_i32_29 overflow<nsw> : i32
              %144 = arith.muli %143, %c64_i32_17 overflow<nsw> : i32
              %145 = arith.subi %144, %c16_i32 overflow<nsw> : i32
              %146 = arith.index_cast %145 : i32 to index
              %147 = ascvec.loadalign %arg11[%146] : memref<256xui32, 1>, index -> <ui32>
              %148 = ascvec.select %147, %142, %138 : <ui32>
              %149 = arith.muli %arg13, %c32_i32 overflow<nsw> : i32
              %150 = arith.index_cast %149 : i32 to index
              ascvec.storealign_pack %arg12[%150], %148, %mask_32 {mode = "b32_to_b8"} : memref<128xui8, 1>, index, <ui32>
            }
            ascvec.mem_bar vst_vld
          } {cannir.vf_mode = "simd"}
          %98 = arith.select %28, %c8_i64, %c0_i64 : i64
          %99 = arith.addi %25, %98 overflow<nsw> : i64
          %100 = arith.muli %99, %c4_i64 overflow<nsw> : i64
          %101 = arith.addi %100, %c512_i64 overflow<nsw> : i64
          ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
          ascvec.copy_ub2ub %alloc_9[%c0], %alloc_8[%c0], %c64_i32 : memref<64xui8, 1>, memref<128xui8, 1>
          ascvec.vf(%alloc_9, %alloc_8 : memref<64xui8, 1>, memref<128xui8, 1>) {
          ^bb0(%arg10: memref<64xui8, 1>, %arg11: memref<128xui8, 1>):
          } {cannir.vf_mode = "simd"}
          ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
          %102 = cannir.extract_physical_layout %arg5 : !cannir.tensor<<gm, ui8>, <<(-1, -1, 1, 544)>, <(-1, -1, 544, 1)>>, <<(-1, -1, 1, 544)>, <(-1, -1, 544, 1)>>, identity>
          %103 = cannir.extract_stride %102 : <<(-1, -1, 1, 544)>, <(-1, -1, 544, 1)>> -> <(-1, -1, 544, 1)>
          %104 = cannir.extract_dim %103[0] : !cannir.stride<(-1, -1, 544, 1)>
          %105 = arith.muli %22, %104 : i64
          %106 = cannir.extract_dim %103[1] : !cannir.stride<(-1, -1, 544, 1)>
          %107 = arith.muli %31, %106 : i64
          %108 = arith.addi %105, %107 : i64
          %109 = arith.muli %99, %c64_i64 overflow<nsw> : i64
          %110 = arith.addi %108, %109 : i64
          %111 = arith.muli %110, %c8_i64 : i64
          %112 = ascscalar.memref_address %alloc_4 : memref<1xui8>
          %113 = arith.divui %111, %c8_i64 : i64
          %114 = arith.addi %112, %113 : i64
          %115 = ascscalar.bind_gm_ptr %114 : memref<1xui8>
          %116 = ascscalar.bitcast_memref %alloc_9 byte_offset = %c0_i64 : memref<64xui8, 1> -> memref<?xui8, 1>
          ascvec.get_buf mte3, %c2_i64, 0 {cannir.buffer.generated_sync}
          ascvec.copy_ub2gm %115[%c0], %116[%c0], %c1_i32, %c64_i32, %c0_i64, %c64_i32 l2_cache_ctl %c0_i32 : memref<1xui8>, memref<?xui8, 1>
          ascvec.release_buf mte3, %c2_i64, 0 {cannir.buffer.generated_sync}
          %117 = arith.remsi %101, %c4_i64 : i64
          %118 = arith.cmpi ne, %117, %c0_i64 : i64
          %119 = arith.cmpi slt, %117, %c0_i64 : i64
          %120 = arith.andi %118, %119 : i1
          %121 = arith.divsi %101, %c4_i64 : i64
          %122 = arith.extui %120 : i1 to i64
          %123 = arith.subi %121, %122 : i64
          ascvec.get_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
          ascvec.vf(%alloc_10, %alloc_8 : memref<4xui8, 1>, memref<128xui8, 1>) {
          ^bb0(%arg10: memref<4xui8, 1>, %arg11: memref<128xui8, 1>):
            %c64 = arith.constant 64 : index
            %c0_17 = arith.constant 0 : index
            %c4_i32_18 = arith.constant 4 : i32
            %c0_19 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            scf.for %arg12 = %c0_19 to %c1 step %c1 {
              %mask, %remaining_out = ascvec.update_mask %c4_i32_18 elem_bits 8 : !ascvec.mask
              %131 = ascvec.loadalign %arg11[%c64] : memref<128xui8, 1>, index -> <ui8>
              ascvec.storealign %arg10[%c0_17], %131, %mask : memref<4xui8, 1>, index, <ui8>
            }
            ascvec.mem_bar vst_vld
          } {cannir.vf_mode = "simd"}
          ascvec.release_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
          %124 = arith.muli %123, %c4_i64 : i64
          %125 = arith.addi %108, %124 : i64
          %126 = arith.muli %125, %c8_i64 : i64
          %127 = arith.divui %126, %c8_i64 : i64
          %128 = arith.addi %112, %127 : i64
          %129 = ascscalar.bind_gm_ptr %128 : memref<1xui8>
          %130 = ascscalar.bitcast_memref %alloc_10 byte_offset = %c0_i64 : memref<4xui8, 1> -> memref<?xui8, 1>
          ascvec.get_buf mte3, %c3_i64, 0 {cannir.buffer.generated_sync}
          ascvec.copy_ub2gm %129[%c0], %130[%c0], %c1_i32, %c4_i32, %c0_i64, %c4_i32 l2_cache_ctl %c0_i32 : memref<1xui8>, memref<?xui8, 1>
          ascvec.release_buf mte3, %c3_i64, 0 {cannir.buffer.generated_sync}
          scf.yield %58 : i64
        } else {
          scf.yield %arg9 : i64
        }
        scf.yield %50 : i64
      } else {
        scf.yield %arg9 : i64
      }
      scf.yield %9 : i64
    }
    return
  }
}
