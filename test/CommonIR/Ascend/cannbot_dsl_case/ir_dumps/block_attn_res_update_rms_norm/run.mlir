module {
  func.func @run_14f5003016fd209bd008946f49863952(%arg0: !cannir.tensor<<gm, f32>, <<(-1, 7168)>, <(7168, 1)>>, <<(-1, 7168)>, <(7168, 1)>>, identity>, %arg1: !cannir.tensor<<gm, bf16>, <<(-1, 7168)>, <(7168, 1)>>, <<(-1, 7168)>, <(7168, 1)>>, identity>, %arg2: !cannir.tensor<<gm, f32>, <<(1, 7168)>, <(7168, 1)>>, <<(1, 7168)>, <(7168, 1)>>, identity>, %arg3: !cannir.tensor<<gm, f32>, <<(-1, 7168)>, <(7168, 1)>>, <<(-1, 7168)>, <(7168, 1)>>, identity>, %arg4: !cannir.tensor<<gm, f32>, <<(1, -1)>, <(-1, 1)>>, <<(1, -1)>, <(-1, 1)>>, identity>, %arg5: !cannir.tensor<<gm, f32>, <<(1, -1)>, <(-1, 1)>>, <<(1, -1)>, <(-1, 1)>>, identity>, %arg6: !cannir.tensor<<gm, bf16>, <<(-1, 7168)>, <(7168, 1)>>, <<(-1, 7168)>, <(7168, 1)>>, identity>, %arg7: !cannir.tensor<<gm, bf16>, <<(1, 7168)>, <(7168, 1)>>, <<(1, 7168)>, <(7168, 1)>>, identity>, %arg8: i64, %arg9: i64) attributes {cannbotdsl.contract_digest = "14f5003016fd209bd008946f498639523c937fc3c861214e5c70c3823c0f5599", cannbotdsl.public_slot_count = 10 : i64, cannbotdsl.slot_dtype_bits = [32, 16, 32, 32, 32, 32, 16, 16, 64, 64], cannbotdsl.slot_dtype_codes = [2, 4, 2, 2, 2, 2, 4, 4, 0, 0], cannbotdsl.slot_dtype_lanes = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], cannbotdsl.slot_flags = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], cannbotdsl.slot_kinds = [1, 1, 1, 1, 1, 1, 1, 1, 2, 2], cannbotdsl.slot_type_digests = ["3b7f118fe4e8019636aaeae4815cb484c8c98019c5f7aa55395f8b4408a627fd", "6be2f78dc779d80255002d16cd0abd89fa4e5a18e666bc83c51f4a710003fc7e", "2f9cb001c97e965a592a1059a2f1da62bc9fb16c90520c73878c16b4498fec9c", "3b7f118fe4e8019636aaeae4815cb484c8c98019c5f7aa55395f8b4408a627fd", "74e05deb83acbd70d96da07c1472f586a5f859098f1588e8c6f04103ef33f7f4", "74e05deb83acbd70d96da07c1472f586a5f859098f1588e8c6f04103ef33f7f4", "6be2f78dc779d80255002d16cd0abd89fa4e5a18e666bc83c51f4a710003fc7e", "157a251959396ac21979a99e3277a0e4c852b57092aea1b55f5507791bd90954", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515"], cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.host} {
    cannir.kernel_launch @kernel_UpdateKernel___call___0(%arg9) [%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8] {cannbotdsl.profile = {indices = array<i64>, name = "block_attn_res_update_rms_norm", op_type = "BlockAttnResUpdateRMSNorm", roles = array<i64>}} : !cannir.tensor<<gm, f32>, <<(-1, 7168)>, <(7168, 1)>>, <<(-1, 7168)>, <(7168, 1)>>, identity>, !cannir.tensor<<gm, bf16>, <<(-1, 7168)>, <(7168, 1)>>, <<(-1, 7168)>, <(7168, 1)>>, identity>, !cannir.tensor<<gm, f32>, <<(1, 7168)>, <(7168, 1)>>, <<(1, 7168)>, <(7168, 1)>>, identity>, !cannir.tensor<<gm, f32>, <<(-1, 7168)>, <(7168, 1)>>, <<(-1, 7168)>, <(7168, 1)>>, identity>, !cannir.tensor<<gm, f32>, <<(1, -1)>, <(-1, 1)>>, <<(1, -1)>, <(-1, 1)>>, identity>, !cannir.tensor<<gm, f32>, <<(1, -1)>, <(-1, 1)>>, <<(1, -1)>, <(-1, 1)>>, identity>, !cannir.tensor<<gm, bf16>, <<(-1, 7168)>, <(7168, 1)>>, <<(-1, 7168)>, <(7168, 1)>>, identity>, !cannir.tensor<<gm, bf16>, <<(1, 7168)>, <(7168, 1)>>, <<(1, 7168)>, <(7168, 1)>>, identity>, i64
    return
  }
  func.func @kernel_UpdateKernel___call___0(%arg0: !cannir.tensor<<gm, f32>, <<(-1, 7168)>, <(7168, 1)>>, <<(-1, 7168)>, <(7168, 1)>>, identity>, %arg1: !cannir.tensor<<gm, bf16>, <<(-1, 7168)>, <(7168, 1)>>, <<(-1, 7168)>, <(7168, 1)>>, identity>, %arg2: !cannir.tensor<<gm, f32>, <<(1, 7168)>, <(7168, 1)>>, <<(1, 7168)>, <(7168, 1)>>, identity>, %arg3: !cannir.tensor<<gm, f32>, <<(-1, 7168)>, <(7168, 1)>>, <<(-1, 7168)>, <(7168, 1)>>, identity>, %arg4: !cannir.tensor<<gm, f32>, <<(1, -1)>, <(-1, 1)>>, <<(1, -1)>, <(-1, 1)>>, identity>, %arg5: !cannir.tensor<<gm, f32>, <<(1, -1)>, <(-1, 1)>>, <<(1, -1)>, <(-1, 1)>>, identity>, %arg6: !cannir.tensor<<gm, bf16>, <<(-1, 7168)>, <(7168, 1)>>, <<(-1, 7168)>, <(7168, 1)>>, identity>, %arg7: !cannir.tensor<<gm, bf16>, <<(1, 7168)>, <(7168, 1)>>, <<(1, 7168)>, <(7168, 1)>>, identity>, %arg8: i64) attributes {cannir.buffer.auto_sync, cannir.buffer.sync_analyzed, cannir.buffer.sync_verified, cannir.buffer.tensor_state, cannir.core = "aiv", cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.device, cannir.explicit_ring_buffer_cursors, cannir.kernel_group = "kernel_UpdateKernel___call___0", cannir.resource.ids_prepared} {
    %c14336_i32 = arith.constant 14336 : i32
    %c28672_i32 = arith.constant 28672 : i32
    %c0_i32 = arith.constant 0 : i32
    %c28672_i64 = arith.constant 28672 : i64
    %c14336_i64 = arith.constant 14336 : i64
    %c1_i32 = arith.constant 1 : i32
    %c0 = arith.constant 0 : index
    %c32_i64 = arith.constant 32 : i64
    %c229376_i64 = arith.constant 229376 : i64
    %c114688_i64 = arith.constant 114688 : i64
    %c9_i64 = arith.constant 9 : i64
    %c8_i64 = arith.constant 8 : i64
    %c7_i64 = arith.constant 7 : i64
    %c6_i64 = arith.constant 6 : i64
    %c5_i64 = arith.constant 5 : i64
    %c4_i64 = arith.constant 4 : i64
    %c3_i64 = arith.constant 3 : i64
    %c2_i64 = arith.constant 2 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %alloc = memref.alloc() {cannir.arg_index = 0 : i64} : memref<1xf32>
    %alloc_0 = memref.alloc() {cannir.arg_index = 1 : i64} : memref<1xbf16>
    %alloc_1 = memref.alloc() {cannir.arg_index = 2 : i64} : memref<7168xf32>
    %alloc_2 = memref.alloc() {cannir.arg_index = 3 : i64} : memref<1xf32>
    %alloc_3 = memref.alloc() {cannir.arg_index = 4 : i64} : memref<1xf32>
    %alloc_4 = memref.alloc() {cannir.arg_index = 5 : i64} : memref<1xf32>
    %alloc_5 = memref.alloc() {cannir.arg_index = 6 : i64} : memref<1xbf16>
    %alloc_6 = memref.alloc() {cannir.arg_index = 7 : i64} : memref<7168xbf16>
    %alloc_7 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<7168xf32, 1>
    %alloc_8 = memref.alloc() {cannir.byte_offset = 28672 : i64} : memref<7168xf32, 1>
    %alloc_9 = memref.alloc() {cannir.byte_offset = 57344 : i64} : memref<7168xbf16, 1>
    %alloc_10 = memref.alloc() {cannir.byte_offset = 71680 : i64} : memref<7168xbf16, 1>
    %alloc_11 = memref.alloc() {cannir.byte_offset = 86016 : i64} : memref<7168xf32, 1>
    %alloc_12 = memref.alloc() {cannir.byte_offset = 114688 : i64} : memref<7168xf32, 1>
    %alloc_13 = memref.alloc() {cannir.byte_offset = 143360 : i64} : memref<7168xf32, 1>
    %alloc_14 = memref.alloc() {cannir.byte_offset = 172032 : i64} : memref<16xf32, 1>
    %alloc_15 = memref.alloc() {cannir.byte_offset = 172096 : i64} : memref<16xf32, 1>
    %alloc_16 = memref.alloc() {cannir.byte_offset = 172160 : i64} : memref<8xf32, 1>
    %alloc_17 = memref.alloc() {cannir.byte_offset = 172192 : i64} : memref<8xf32, 1>
    %alloc_18 = memref.alloc() {cannir.byte_offset = 172224 : i64} : memref<7168xbf16, 1>
    %0 = cannir.extract_layout %arg0 : !cannir.tensor<<gm, f32>, <<(-1, 7168)>, <(7168, 1)>>, <<(-1, 7168)>, <(7168, 1)>>, identity>
    %1 = cannir.extract_shape %0 : <<(-1, 7168)>, <(7168, 1)>> -> <(-1, 7168)>
    %2 = cannir.extract_dim %1[0] : !cannir.shape<(-1, 7168)>
    %3 = ascscalar.get_block_idx
    %4 = arith.muli %3, %arg8 : i64
    %5 = arith.addi %4, %arg8 : i64
    %6 = arith.cmpi sgt, %5, %2 : i64
    %7 = arith.select %6, %2, %5 : i64
    %8 = arith.cmpi slt, %4, %2 : i64
    scf.if %8 {
      %9 = ascscalar.bitcast_memref %alloc_13 byte_offset = %c0_i64 : memref<7168xf32, 1> -> memref<?xf32, 1>
      %10 = ascscalar.bitcast_memref %alloc_18 byte_offset = %c0_i64 : memref<7168xbf16, 1> -> memref<?xbf16, 1>
      ascvec.get_buf mte2, %c6_i64, 0 {cannir.buffer.generated_sync}
      "ascvec.copy_gm2ub_plan"(%9, %alloc_1, %c0_i32, %c0_i32, %c0_i64, %c0_i64, %c1_i64, %c28672_i64, %c0_i64, %c28672_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 28672 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 143360 : i64, ub_capacity = 28672 : i64}> : (memref<?xf32, 1>, memref<7168xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
      ascvec.release_buf mte2, %c6_i64, 0 {cannir.buffer.generated_sync}
      ascvec.get_buf mte2, %c9_i64, 0 {cannir.buffer.generated_sync}
      "ascvec.copy_gm2ub_plan"(%10, %alloc_6, %c0_i32, %c0_i32, %c0_i64, %c0_i64, %c1_i64, %c14336_i64, %c0_i64, %c14336_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 14336 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = 172224 : i64, ub_capacity = 14336 : i64}> : (memref<?xbf16, 1>, memref<7168xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
      ascvec.release_buf mte2, %c9_i64, 0 {cannir.buffer.generated_sync}
      %11 = scf.for %arg9 = %4 to %7 step %c1_i64 iter_args(%arg10 = %c0_i64) -> (i64)  : i64 {
        %12 = arith.subi %7, %arg9 : i64
        %13 = arith.cmpi sgt, %12, %c1_i64 : i64
        %14 = arith.select %13, %c1_i64, %12 : i64
        %15 = arith.cmpi eq, %arg10, %c0_i64 : i64
        %16 = ascscalar.memref_address %alloc_7 : memref<7168xf32, 1>
        %17 = ascscalar.memref_address %alloc_8 : memref<7168xf32, 1>
        %18 = arith.select %15, %16, %17 : i64
        %19 = arith.cmpi ne, %arg10, %c0_i64 : i64
        %20 = arith.extui %19 : i1 to i64
        %21 = arith.addi %arg10, %c1_i64 overflow<nsw> : i64
        %22 = arith.remui %21, %c2_i64 : i64
        %23 = ascscalar.memref_address %alloc_9 : memref<7168xbf16, 1>
        %24 = ascscalar.memref_address %alloc_10 : memref<7168xbf16, 1>
        %25 = arith.select %15, %23, %24 : i64
        %26 = arith.select %15, %c2_i64, %c3_i64 : i64
        %27 = arith.minsi %14, %c0_i64 : i64
        %28 = arith.subi %14, %27 : i64
        %29 = arith.minsi %28, %c1_i64 : i64
        %30 = arith.cmpi sge, %14, %c0_i64 : i64
        %31 = arith.select %30, %29, %c0_i64 : i64
        %32 = arith.muli %arg9, %c28672_i64 : i64
        %33 = arith.muli %31, %c28672_i64 : i64
        %34 = arith.cmpi eq, %33, %c0_i64 : i64
        %35 = arith.cmpi ne, %33, %c0_i64 : i64
        %36 = arith.extui %35 : i1 to i64
        %37 = arith.select %34, %c4_i64, %33 : i64
        %38 = ascscalar.bind_local_ptr %18 : memref<1xf32, 1>
        ascvec.get_buf mte2, %20, 0 {cannir.buffer.generated_sync}
        "ascvec.copy_gm2ub_plan"(%38, %alloc, %c0_i32, %c0_i32, %c0_i64, %32, %36, %37, %c0_i64, %37, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = -1 : i64, ub_capacity = 4 : i64}> : (memref<1xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
        ascvec.release_buf mte2, %20, 0 {cannir.buffer.generated_sync}
        %39 = arith.muli %arg9, %c14336_i64 : i64
        %40 = arith.muli %31, %c14336_i64 : i64
        %41 = arith.cmpi eq, %40, %c0_i64 : i64
        %42 = arith.cmpi ne, %40, %c0_i64 : i64
        %43 = arith.extui %42 : i1 to i64
        %44 = arith.select %41, %c2_i64, %40 : i64
        %45 = ascscalar.bind_local_ptr %25 : memref<1xbf16, 1>
        ascvec.get_buf mte2, %26, 0 {cannir.buffer.generated_sync}
        "ascvec.copy_gm2ub_plan"(%45, %alloc_0, %c0_i32, %c0_i32, %c0_i64, %39, %43, %44, %c0_i64, %44, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
        ascvec.release_buf mte2, %26, 0 {cannir.buffer.generated_sync}
        %46 = ascscalar.bind_local_ptr %18 : memref<7168xf32, 1>
        %47 = ascscalar.bind_local_ptr %25 : memref<7168xbf16, 1>
        %48 = ascscalar.memref_address %alloc_16 : memref<8xf32, 1>
        %49 = ascscalar.memref_address %alloc_17 : memref<8xf32, 1>
        %50 = arith.select %15, %48, %49 : i64
        %51 = ascscalar.bind_local_ptr %50 : memref<8xf32, 1>
        ascvec.get_buf v, %20, 0 {cannir.buffer.generated_sync}
        ascvec.get_buf v, %26, 0 {cannir.buffer.generated_sync}
        ascvec.get_buf v, %c6_i64, 0 {cannir.buffer.generated_sync}
        ascvec.vf(%46, %47, %alloc_13, %51, %14 : memref<7168xf32, 1>, memref<7168xbf16, 1>, memref<7168xf32, 1>, memref<8xf32, 1>, i64) {
        ^bb0(%arg11: memref<7168xf32, 1>, %arg12: memref<7168xbf16, 1>, %arg13: memref<7168xf32, 1>, %arg14: memref<8xf32, 1>, %arg15: i64):
          %c1_i64_19 = arith.constant 1 : i64
          %c0_i64_20 = arith.constant 0 : i64
          %cst = arith.constant 9.99999997E-7 : f32
          %cst_21 = arith.constant 1.39508935E-4 : f32
          %c56_i32 = arith.constant 56 : i32
          %c0_i32_22 = arith.constant 0 : i32
          %c64_i64 = arith.constant 64 : i64
          %c0_23 = arith.constant 0 : index
          %c128_i32 = arith.constant 128 : i32
          %cst_24 = arith.constant 0.000000e+00 : f32
          %c64_i32 = arith.constant 64 : i32
          %c1_i32_25 = arith.constant 1 : i32
          %80 = ascvec.create_mask : !ascvec.mask
          %mask, %remaining_out = ascvec.update_mask %c1_i32_25 : !ascvec.mask
          %mask_26, %remaining_out_27 = ascvec.update_mask %c64_i32 : !ascvec.mask
          scf.for %arg16 = %c0_i64_20 to %arg15 step %c1_i64_19  : i64 {
            %81 = ascvec.duplicate_scalar %cst_24, %80 : f32 -> <f32>
            %82:4 = scf.for %arg17 = %c0_i32_22 to %c56_i32 step %c1_i32_25 iter_args(%arg18 = %81, %arg19 = %81, %arg20 = %81, %arg21 = %81) -> (!ascvec.vreg<f32>, !ascvec.vreg<f32>, !ascvec.vreg<f32>, !ascvec.vreg<f32>)  : i32 {
              %91 = arith.muli %arg17, %c128_i32 overflow<nsw> : i32
              %92 = arith.extsi %91 : i32 to i64
              %93 = arith.index_cast %91 : i32 to index
              %94 = ascvec.loadalign %arg11[%93] : memref<7168xf32, 1>, index -> <f32>
              %95 = ascvec.loadalign_unpack %arg12[%93] {mode = "b16_to_b32"} : memref<7168xbf16, 1>, index -> <bf16>
              %96 = ascvec.cast %95, %80 {rounding = 6 : i32} : <bf16> -> <f32>
              %97 = ascvec.add %94, %96, %80 : <f32>
              %98 = arith.addi %91, %c64_i32 overflow<nsw> : i32
              %99 = arith.index_cast %98 : i32 to index
              %100 = ascvec.loadalign %arg11[%99] : memref<7168xf32, 1>, index -> <f32>
              %101 = ascvec.loadalign_unpack %arg12[%99] {mode = "b16_to_b32"} : memref<7168xbf16, 1>, index -> <bf16>
              %102 = ascvec.cast %101, %80 {rounding = 6 : i32} : <bf16> -> <f32>
              %103 = ascvec.add %100, %102, %80 : <f32>
              %104 = ascvec.loadalign %arg13[%93] : memref<7168xf32, 1>, index -> <f32>
              %105 = ascvec.loadalign %arg13[%99] : memref<7168xf32, 1>, index -> <f32>
              ascvec.storealign %arg11[%93], %97, %80 : memref<7168xf32, 1>, index, <f32>
              %106 = arith.addi %92, %c64_i64 overflow<nsw> : i64
              %107 = arith.index_cast %106 : i64 to index
              ascvec.storealign %arg11[%107], %103, %80 : memref<7168xf32, 1>, index, <f32>
              %108 = ascvec.mul %97, %104, %80 : <f32>
              %109 = ascvec.add %108, %arg18, %80 : <f32>
              %110 = ascvec.mul %103, %105, %80 : <f32>
              %111 = ascvec.add %110, %arg19, %80 : <f32>
              %112 = ascvec.mul %97, %97, %80 : <f32>
              %113 = ascvec.add %112, %arg20, %80 : <f32>
              %114 = ascvec.mul %103, %103, %80 : <f32>
              %115 = ascvec.add %114, %arg21, %80 : <f32>
              scf.yield %109, %111, %113, %115 : !ascvec.vreg<f32>, !ascvec.vreg<f32>, !ascvec.vreg<f32>, !ascvec.vreg<f32>
            }
            %83 = ascvec.add %82#0, %82#1, %80 : <f32>
            %84 = ascvec.add %82#2, %82#3, %80 : <f32>
            %85 = ascvec.reduce_sum %83, %80 : <f32>
            %86 = ascvec.reduce_sum %84, %80 : <f32>
            %87 = ascvec.mul_scalar %86, %cst_21, %mask : <f32>, f32
            %88 = ascvec.add_scalar %87, %cst, %mask : <f32>, f32
            %89 = ascvec.sqrt %88, %mask : <f32>
            %90 = ascvec.div %85, %89, %mask {precision_mode = true} : <f32>
            ascvec.storealign_1st %arg14[%c0_23], %90 : memref<8xf32, 1>, index, <f32>
          }
          ascvec.mem_bar vst_vld
        } {cannir.vf_mode = "simd"}
        ascvec.release_buf v, %c6_i64, 0 {cannir.buffer.generated_sync}
        ascvec.release_buf v, %20, 0 {cannir.buffer.generated_sync}
        %52 = arith.muli %arg9, %c229376_i64 : i64
        %53 = ascscalar.memref_address %alloc : memref<1xf32>
        %54 = arith.divui %52, %c8_i64 : i64
        %55 = arith.addi %53, %54 : i64
        %56 = ascscalar.bind_gm_ptr %55 : memref<1xf32>
        ascvec.get_buf mte3, %20, 0 {cannir.buffer.generated_sync}
        ascvec.copy_ub2gm %56[%c0], %38[%c0], %c1_i32, %c28672_i32, %c0_i64, %c28672_i32 l2_cache_ctl %c0_i32 : memref<1xf32>, memref<1xf32, 1>
        ascvec.release_buf mte3, %20, 0 {cannir.buffer.generated_sync}
        %57 = ascscalar.memref_address %alloc_11 : memref<7168xf32, 1>
        %58 = ascscalar.memref_address %alloc_12 : memref<7168xf32, 1>
        %59 = arith.select %15, %57, %58 : i64
        %60 = arith.select %15, %c4_i64, %c5_i64 : i64
        %61 = ascscalar.memref_address %alloc_14 : memref<16xf32, 1>
        %62 = ascscalar.memref_address %alloc_15 : memref<16xf32, 1>
        %63 = arith.select %15, %61, %62 : i64
        %64 = arith.select %15, %c7_i64, %c8_i64 : i64
        %65 = ascscalar.bind_local_ptr %59 : memref<1xf32, 1>
        ascvec.get_buf mte2, %60, 0 {cannir.buffer.generated_sync}
        "ascvec.copy_gm2ub_plan"(%65, %alloc_2, %c0_i32, %c0_i32, %c0_i64, %32, %36, %37, %c0_i64, %37, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = -1 : i64, ub_capacity = 4 : i64}> : (memref<1xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
        ascvec.release_buf mte2, %60, 0 {cannir.buffer.generated_sync}
        %66 = arith.muli %arg9, %c4_i64 : i64
        %67 = arith.muli %31, %c4_i64 : i64
        %68 = arith.cmpi eq, %67, %c0_i64 : i64
        %69 = arith.cmpi ne, %67, %c0_i64 : i64
        %70 = arith.extui %69 : i1 to i64
        %71 = arith.select %68, %c4_i64, %67 : i64
        %72 = ascscalar.bind_local_ptr %63 : memref<1xf32, 1>
        ascvec.get_buf mte2, %64, 0 {cannir.buffer.generated_sync}
        "ascvec.copy_gm2ub_plan"(%72, %alloc_3, %c0_i32, %c0_i32, %c0_i64, %66, %70, %71, %c0_i64, %71, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = -1 : i64, ub_capacity = 4 : i64}> : (memref<1xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
        "ascvec.copy_gm2ub_plan"(%72, %alloc_4, %c0_i32, %c0_i32, %c32_i64, %66, %70, %71, %c0_i64, %71, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = -1 : i64, ub_capacity = 4 : i64}> : (memref<1xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
        ascvec.release_buf mte2, %64, 0 {cannir.buffer.generated_sync}
        %73 = ascscalar.bind_local_ptr %59 : memref<7168xf32, 1>
        %74 = ascscalar.bind_local_ptr %63 : memref<16xf32, 1>
        ascvec.get_buf v, %64, 0 {cannir.buffer.generated_sync}
        ascvec.get_buf v, %20, 0 {cannir.buffer.generated_sync}
        ascvec.get_buf v, %60, 0 {cannir.buffer.generated_sync}
        ascvec.vf(%74, %51, %46, %73, %47, %14 : memref<16xf32, 1>, memref<8xf32, 1>, memref<7168xf32, 1>, memref<7168xf32, 1>, memref<7168xbf16, 1>, i64) {
        ^bb0(%arg11: memref<16xf32, 1>, %arg12: memref<8xf32, 1>, %arg13: memref<7168xf32, 1>, %arg14: memref<7168xf32, 1>, %arg15: memref<7168xbf16, 1>, %arg16: i64):
          %c8 = arith.constant 8 : index
          %c1_i64_19 = arith.constant 1 : i64
          %c0_i64_20 = arith.constant 0 : i64
          %cst = arith.constant 9.99999997E-7 : f32
          %cst_21 = arith.constant 1.39508935E-4 : f32
          %c56_i32 = arith.constant 56 : i32
          %c0_i32_22 = arith.constant 0 : i32
          %c64_i64 = arith.constant 64 : i64
          %c128_i32 = arith.constant 128 : i32
          %cst_23 = arith.constant 0.000000e+00 : f32
          %cst_24 = arith.constant 1.000000e+00 : f32
          %c0_25 = arith.constant 0 : index
          %c64_i32 = arith.constant 64 : i32
          %c1_i32_26 = arith.constant 1 : i32
          %80 = ascvec.create_mask : !ascvec.mask
          %mask, %remaining_out = ascvec.update_mask %c1_i32_26 : !ascvec.mask
          %mask_27, %remaining_out_28 = ascvec.update_mask %c64_i32 : !ascvec.mask
          scf.for %arg17 = %c0_i64_20 to %arg16 step %c1_i64_19  : i64 {
            %81 = ascvec.loadalign_brc %arg11[%c0_25] : memref<16xf32, 1>, index -> <f32>
            %82 = ascvec.loadalign_brc %arg11[%c8] : memref<16xf32, 1>, index -> <f32>
            %83 = ascvec.loadalign_brc %arg12[%c0_25] : memref<8xf32, 1>, index -> <f32>
            %84 = ascvec.max %81, %83, %80 : <f32>
            %85 = ascvec.exp_sub %81, %84, %80 : <f32>, <f32> -> <f32>
            %86 = ascvec.exp_sub %83, %84, %80 : <f32>, <f32> -> <f32>
            %87 = ascvec.madd_two_address %82, %85, %86, %80 : <f32>
            %88 = ascvec.duplicate_scalar %cst_24, %80 : f32 -> <f32>
            %89 = ascvec.div %88, %87, %80 {precision_mode = true} : <f32>
            %90 = ascvec.mul %85, %89, %80 : <f32>
            %91 = ascvec.mul %86, %89, %80 : <f32>
            %92 = ascvec.duplicate_scalar %cst_23, %80 : f32 -> <f32>
            %93:2 = scf.for %arg18 = %c0_i32_22 to %c56_i32 step %c1_i32_26 iter_args(%arg19 = %92, %arg20 = %92) -> (!ascvec.vreg<f32>, !ascvec.vreg<f32>)  : i32 {
              %101 = arith.muli %arg18, %c128_i32 overflow<nsw> : i32
              %102 = arith.extsi %101 : i32 to i64
              %103 = arith.index_cast %101 : i32 to index
              %104 = ascvec.loadalign %arg13[%103] : memref<7168xf32, 1>, index -> <f32>
              %105 = arith.addi %102, %c64_i64 overflow<nsw> : i64
              %106 = arith.index_cast %105 : i64 to index
              %107 = ascvec.loadalign %arg13[%106] : memref<7168xf32, 1>, index -> <f32>
              %108 = ascvec.loadalign %arg14[%103] : memref<7168xf32, 1>, index -> <f32>
              %109 = ascvec.loadalign %arg14[%106] : memref<7168xf32, 1>, index -> <f32>
              %110 = ascvec.mul %104, %91, %80 : <f32>
              %111 = ascvec.madd_two_address %108, %90, %110, %80 : <f32>
              %112 = ascvec.mul %107, %91, %80 : <f32>
              %113 = ascvec.madd_two_address %109, %90, %112, %80 : <f32>
              %114 = ascvec.cast %111, %80 : <f32> -> <bf16>
              %115 = ascvec.cast %113, %80 : <f32> -> <bf16>
              ascvec.storealign_pack %arg15[%103], %114, %80 {mode = "b32_to_b16"} : memref<7168xbf16, 1>, index, <bf16>
              ascvec.storealign_pack %arg15[%106], %115, %80 {mode = "b32_to_b16"} : memref<7168xbf16, 1>, index, <bf16>
              %116 = ascvec.cast %114, %80 {rounding = 6 : i32} : <bf16> -> <f32>
              %117 = ascvec.cast %115, %80 {rounding = 6 : i32} : <bf16> -> <f32>
              %118 = ascvec.mul %116, %116, %80 : <f32>
              %119 = ascvec.add %arg19, %118, %80 : <f32>
              %120 = ascvec.mul %117, %117, %80 : <f32>
              %121 = ascvec.add %arg20, %120, %80 : <f32>
              scf.yield %119, %121 : !ascvec.vreg<f32>, !ascvec.vreg<f32>
            }
            %94 = ascvec.add %93#0, %93#1, %80 : <f32>
            %95 = ascvec.reduce_sum %94, %80 : <f32>
            %96 = ascvec.mul_scalar %95, %cst_21, %mask : <f32>, f32
            %97 = ascvec.add_scalar %96, %cst, %mask : <f32>, f32
            %98 = ascvec.sqrt %97, %mask : <f32>
            %99 = ascvec.duplicate_scalar %cst_24, %mask : f32 -> <f32>
            %100 = ascvec.div %99, %98, %mask {precision_mode = true} : <f32>
            ascvec.storealign_1st %arg12[%c0_25], %100 : memref<8xf32, 1>, index, <f32>
          }
          ascvec.mem_bar vst_vld
        } {cannir.vf_mode = "simd"}
        ascvec.release_buf v, %60, 0 {cannir.buffer.generated_sync}
        ascvec.release_buf v, %20, 0 {cannir.buffer.generated_sync}
        ascvec.release_buf v, %64, 0 {cannir.buffer.generated_sync}
        ascvec.get_buf v, %c9_i64, 0 {cannir.buffer.generated_sync}
        ascvec.vf(%51, %47, %alloc_18, %14 : memref<8xf32, 1>, memref<7168xbf16, 1>, memref<7168xbf16, 1>, i64) {
        ^bb0(%arg11: memref<8xf32, 1>, %arg12: memref<7168xbf16, 1>, %arg13: memref<7168xbf16, 1>, %arg14: i64):
          %c1_i64_19 = arith.constant 1 : i64
          %c0_i64_20 = arith.constant 0 : i64
          %c1_i32_21 = arith.constant 1 : i32
          %c112_i32 = arith.constant 112 : i32
          %c0_i32_22 = arith.constant 0 : i32
          %c0_23 = arith.constant 0 : index
          %c64_i32 = arith.constant 64 : i32
          %80 = ascvec.create_mask : !ascvec.mask
          %mask, %remaining_out = ascvec.update_mask %c64_i32 : !ascvec.mask
          scf.for %arg15 = %c0_i64_20 to %arg14 step %c1_i64_19  : i64 {
            %81 = ascvec.loadalign_brc %arg11[%c0_23] : memref<8xf32, 1>, index -> <f32>
            scf.for %arg16 = %c0_i32_22 to %c112_i32 step %c1_i32_21  : i32 {
              %82 = arith.muli %arg16, %c64_i32 overflow<nsw> : i32
              %83 = arith.index_cast %82 : i32 to index
              %84 = ascvec.loadalign_unpack %arg12[%83] {mode = "b16_to_b32"} : memref<7168xbf16, 1>, index -> <bf16>
              %85 = ascvec.cast %84, %80 {rounding = 6 : i32} : <bf16> -> <f32>
              %86 = ascvec.loadalign_unpack %arg13[%83] {mode = "b16_to_b32"} : memref<7168xbf16, 1>, index -> <bf16>
              %87 = ascvec.cast %86, %80 {rounding = 6 : i32} : <bf16> -> <f32>
              %88 = ascvec.mul %85, %81, %80 : <f32>
              %89 = ascvec.mul %88, %87, %80 : <f32>
              %90 = ascvec.cast %89, %80 : <f32> -> <bf16>
              ascvec.storealign_pack %arg12[%83], %90, %80 {mode = "b32_to_b16"} : memref<7168xbf16, 1>, index, <bf16>
            }
          }
          ascvec.mem_bar vst_vld
        } {cannir.vf_mode = "simd"}
        ascvec.release_buf v, %c9_i64, 0 {cannir.buffer.generated_sync}
        ascvec.release_buf v, %26, 0 {cannir.buffer.generated_sync}
        %75 = arith.muli %arg9, %c114688_i64 : i64
        %76 = ascscalar.memref_address %alloc_5 : memref<1xbf16>
        %77 = arith.divui %75, %c8_i64 : i64
        %78 = arith.addi %76, %77 : i64
        %79 = ascscalar.bind_gm_ptr %78 : memref<1xbf16>
        ascvec.get_buf mte3, %26, 0 {cannir.buffer.generated_sync}
        ascvec.copy_ub2gm %79[%c0], %45[%c0], %c1_i32, %c14336_i32, %c0_i64, %c14336_i32 l2_cache_ctl %c0_i32 : memref<1xbf16>, memref<1xbf16, 1>
        ascvec.release_buf mte3, %26, 0 {cannir.buffer.generated_sync}
        scf.yield %22 : i64
      }
    }
    return
  }
}
