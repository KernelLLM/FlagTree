module {
  func.func @run_3cf1c954b3c07ef9c17b724e6da57c3b(%arg0: !cannir.tensor<<gm, bf16>, <<(5, 4, 5121)>, <(20484, 5121, 1)>>, <<(5, 4, 5121)>, <(20484, 5121, 1)>>, identity>, %arg1: !cannir.tensor<<gm, bf16>, <<(5, 4, 5121)>, <(20484, 5121, 1)>>, <<(5, 4, 5121)>, <(20484, 5121, 1)>>, identity>, %arg2: !cannir.tensor<<gm, bf16>, <<(5, 5121)>, <(5121, 1)>>, <<(5, 5121)>, <(5121, 1)>>, identity>, %arg3: !cannir.tensor<<gm, f32>, <<(4, 5121)>, <(5121, 1)>>, <<(4, 5121)>, <(5121, 1)>>, identity>, %arg4: !cannir.tensor<<gm, bf16>, <<(5, 4, 5121)>, <(20484, 5121, 1)>>, <<(5, 4, 5121)>, <(20484, 5121, 1)>>, identity>, %arg5: !cannir.tensor<<gm, ui8>, <<(5)>, <(1)>>, <<(5)>, <(1)>>, identity>) attributes {cannbotdsl.contract_digest = "3cf1c954b3c07ef9c17b724e6da57c3bde93e39aafa9445db0eeebae48fb6afe", cannbotdsl.public_slot_count = 6 : i64, cannbotdsl.slot_dtype_bits = [16, 16, 16, 32, 16, 8], cannbotdsl.slot_dtype_codes = [4, 4, 4, 2, 4, 1], cannbotdsl.slot_dtype_lanes = [1, 1, 1, 1, 1, 1], cannbotdsl.slot_flags = [1, 1, 1, 1, 1, 1], cannbotdsl.slot_kinds = [1, 1, 1, 1, 1, 1], cannbotdsl.slot_type_digests = ["00a73a919b6b33dfd1cd209f2b2b3aa969d8e5498b2e5580f6459246c256da18", "00a73a919b6b33dfd1cd209f2b2b3aa969d8e5498b2e5580f6459246c256da18", "d3509c996c03b13e64eb990f699102ff9a87406ce072cf793cbe84ff4123f302", "bc8226ad59fe02660275eedf6f5f05202eb81203b423859c95bc313f83028bf4", "00a73a919b6b33dfd1cd209f2b2b3aa969d8e5498b2e5580f6459246c256da18", "ee6daadaca906908b4e4a0e5e49d54074cc8ed52fb757db54509fea861018c55"], cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.host} {
    %cst = arith.constant 9.99999997E-7 : f32
    %c20_i64 = arith.constant 20 : i64
    cannir.kernel_launch @kernel_EngramGateKernel___call___0(%c20_i64) [%arg0, %arg1, %arg2, %arg3, %arg4, %cst, %cst, %arg5] {cannbotdsl.profile = {indices = array<i64: 0, 1, 2, 3, 4, 7>, name = "EngramGateKernel___call__", op_type = "EngramGateKernel___call__", roles = array<i64: 0, 0, 0, 0, 1, 0>}} : !cannir.tensor<<gm, bf16>, <<(5, 4, 5121)>, <(20484, 5121, 1)>>, <<(5, 4, 5121)>, <(20484, 5121, 1)>>, identity>, !cannir.tensor<<gm, bf16>, <<(5, 4, 5121)>, <(20484, 5121, 1)>>, <<(5, 4, 5121)>, <(20484, 5121, 1)>>, identity>, !cannir.tensor<<gm, bf16>, <<(5, 5121)>, <(5121, 1)>>, <<(5, 5121)>, <(5121, 1)>>, identity>, !cannir.tensor<<gm, f32>, <<(4, 5121)>, <(5121, 1)>>, <<(4, 5121)>, <(5121, 1)>>, identity>, !cannir.tensor<<gm, bf16>, <<(5, 4, 5121)>, <(20484, 5121, 1)>>, <<(5, 4, 5121)>, <(20484, 5121, 1)>>, identity>, f32, f32, !cannir.tensor<<gm, ui8>, <<(5)>, <(1)>>, <<(5)>, <(1)>>, identity>
    return
  }
  func.func @kernel_EngramGateKernel___call___0(%arg0: !cannir.tensor<<gm, bf16>, <<(5, 4, 5121)>, <(20484, 5121, 1)>>, <<(5, 4, 5121)>, <(20484, 5121, 1)>>, identity>, %arg1: !cannir.tensor<<gm, bf16>, <<(5, 4, 5121)>, <(20484, 5121, 1)>>, <<(5, 4, 5121)>, <(20484, 5121, 1)>>, identity>, %arg2: !cannir.tensor<<gm, bf16>, <<(5, 5121)>, <(5121, 1)>>, <<(5, 5121)>, <(5121, 1)>>, identity>, %arg3: !cannir.tensor<<gm, f32>, <<(4, 5121)>, <(5121, 1)>>, <<(4, 5121)>, <(5121, 1)>>, identity>, %arg4: !cannir.tensor<<gm, bf16>, <<(5, 4, 5121)>, <(20484, 5121, 1)>>, <<(5, 4, 5121)>, <(20484, 5121, 1)>>, identity>, %arg5: f32, %arg6: f32, %arg7: !cannir.tensor<<gm, ui8>, <<(5)>, <(1)>>, <<(5)>, <(1)>>, identity>) attributes {cannir.buffer.auto_sync, cannir.buffer.sync_analyzed, cannir.buffer.sync_verified, cannir.buffer.tensor_state, cannir.core = "aiv", cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.device, cannir.kernel_group = "kernel_EngramGateKernel___call___0", cannir.resource.ids_prepared} {
    %c10242_i32 = arith.constant 10242 : i32
    %c10368_i32 = arith.constant 10368 : i32
    %c20_i64 = arith.constant 20 : i64
    %c0 = arith.constant 0 : index
    %c0_i8 = arith.constant 0 : i8
    %c0_i32 = arith.constant 0 : i32
    %c40968_i64 = arith.constant 40968 : i64
    %c10242_i64 = arith.constant 10242 : i64
    %c20484_i64 = arith.constant 20484 : i64
    %c1_i32 = arith.constant 1 : i32
    %c5121_i64 = arith.constant 5121 : i64
    %c16_i64 = arith.constant 16 : i64
    %c8_i64 = arith.constant 8 : i64
    %c7_i64 = arith.constant 7 : i64
    %c6_i64 = arith.constant 6 : i64
    %c4_i64 = arith.constant 4 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %alloc = memref.alloc() {cannir.arg_index = 0 : i64} : memref<102420xbf16>
    %alloc_0 = memref.alloc() {cannir.arg_index = 1 : i64} : memref<102420xbf16>
    %alloc_1 = memref.alloc() {cannir.arg_index = 2 : i64} : memref<25605xbf16>
    %alloc_2 = memref.alloc() {cannir.arg_index = 3 : i64} : memref<20484xf32>
    %alloc_3 = memref.alloc() {cannir.arg_index = 4 : i64} : memref<102420xbf16>
    %alloc_4 = memref.alloc() {cannir.arg_index = 7 : i64} : memref<5xui8>
    %alloc_5 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<5184xbf16, 1>
    %alloc_6 = memref.alloc() {cannir.byte_offset = 10368 : i64} : memref<5184xbf16, 1>
    %alloc_7 = memref.alloc() {cannir.byte_offset = 20736 : i64} : memref<5184xbf16, 1>
    %alloc_8 = memref.alloc() {cannir.byte_offset = 31104 : i64} : memref<5184xbf16, 1>
    %alloc_9 = memref.alloc() {cannir.byte_offset = 41472 : i64} : memref<5184xbf16, 1>
    %alloc_10 = memref.alloc() {cannir.byte_offset = 51840 : i64} : memref<5184xbf16, 1>
    %alloc_11 = memref.alloc() {cannir.byte_offset = 62208 : i64} : memref<5184xbf16, 1>
    %alloc_12 = memref.alloc() {cannir.byte_offset = 72576 : i64} : memref<5184xbf16, 1>
    %alloc_13 = memref.alloc() {cannir.byte_offset = 82944 : i64} : memref<5184xbf16, 1>
    %alloc_14 = memref.alloc() {cannir.byte_offset = 93312 : i64} : memref<5184xf32, 1>
    %alloc_15 = memref.alloc() {cannir.byte_offset = 114048 : i64} : memref<5184xbf16, 1>
    %alloc_16 = memref.alloc() {cannir.byte_offset = 124416 : i64} : memref<5184xbf16, 1>
    %alloc_17 = memref.alloc() {cannir.byte_offset = 134784 : i64} : memref<5184xf32, 1>
    %0 = ascscalar.get_block_idx
    ascvec.vf(%alloc_6, %alloc_7, %alloc_9, %alloc_10, %alloc_12, %alloc_13 : memref<5184xbf16, 1>, memref<5184xbf16, 1>, memref<5184xbf16, 1>, memref<5184xbf16, 1>, memref<5184xbf16, 1>, memref<5184xbf16, 1>) {
    ^bb0(%arg8: memref<5184xbf16, 1>, %arg9: memref<5184xbf16, 1>, %arg10: memref<5184xbf16, 1>, %arg11: memref<5184xbf16, 1>, %arg12: memref<5184xbf16, 1>, %arg13: memref<5184xbf16, 1>):
      %c0_18 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %c0_19 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg14 = %c0_19 to %c1 step %c1 {
        %31 = ascvec.create_mask : !ascvec.mask
        %32 = ascvec.duplicate_scalar %cst, %31 : f32 -> <f32>
        %33 = ascvec.cast %32, %31 : <f32> -> <bf16>
        ascvec.storealign_pack %arg8[%c0_18], %33, %31 {mode = "b32_to_b16"} : memref<5184xbf16, 1>, index, <bf16>
        ascvec.storealign_pack %arg9[%c0_18], %33, %31 {mode = "b32_to_b16"} : memref<5184xbf16, 1>, index, <bf16>
        ascvec.storealign_pack %arg10[%c0_18], %33, %31 {mode = "b32_to_b16"} : memref<5184xbf16, 1>, index, <bf16>
        ascvec.storealign_pack %arg11[%c0_18], %33, %31 {mode = "b32_to_b16"} : memref<5184xbf16, 1>, index, <bf16>
        ascvec.storealign_pack %arg12[%c0_18], %33, %31 {mode = "b32_to_b16"} : memref<5184xbf16, 1>, index, <bf16>
        ascvec.storealign_pack %arg13[%c0_18], %33, %31 {mode = "b32_to_b16"} : memref<5184xbf16, 1>, index, <bf16>
      }
    } {cannir.vf_mode = "simd"}
    %1 = arith.divsi %0, %c4_i64 : i64
    %2 = arith.muli %1, %c4_i64 : i64
    %3 = arith.subi %0, %2 : i64
    %4 = arith.index_cast %1 : i64 to index
    %5 = memref.load %alloc_4[%4] : memref<5xui8>
    %6 = cannir.scalar_storage_cast %5 : ui8 -> i8
    %7 = arith.cmpi ne, %6, %c0_i8 : i8
    scf.if %7 {
      %31 = arith.muli %1, %c40968_i64 : i64
      %32 = arith.muli %3, %c10242_i64 : i64
      %33 = arith.addi %31, %32 : i64
      ascvec.get_buf mte2, %c0_i64, 0 {cannir.buffer.generated_sync}
      "ascvec.copy_gm2ub_plan"(%alloc_5, %alloc, %c0_i32, %c0_i32, %c0_i64, %33, %c1_i64, %c10242_i64, %c0_i64, %c10242_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 204840 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = 0 : i64, ub_capacity = 10368 : i64}> : (memref<5184xbf16, 1>, memref<102420xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
      ascvec.release_buf mte2, %c0_i64, 0 {cannir.buffer.generated_sync}
      %34 = arith.muli %3, %c20484_i64 : i64
      ascvec.get_buf mte2, %c6_i64, 0 {cannir.buffer.generated_sync}
      "ascvec.copy_gm2ub_plan"(%alloc_14, %alloc_2, %c0_i32, %c0_i32, %c0_i64, %34, %c1_i64, %c20484_i64, %c0_i64, %c20484_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 81936 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 93312 : i64, ub_capacity = 20736 : i64}> : (memref<5184xf32, 1>, memref<20484xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
      ascvec.release_buf mte2, %c6_i64, 0 {cannir.buffer.generated_sync}
    } else {
      %31 = arith.muli %1, %c40968_i64 : i64
      %32 = arith.muli %3, %c10242_i64 : i64
      %33 = arith.addi %31, %32 : i64
      ascvec.get_buf mte2, %c0_i64, 0 {cannir.buffer.generated_sync}
      "ascvec.copy_gm2ub_plan"(%alloc_5, %alloc, %c0_i32, %c0_i32, %c0_i64, %33, %c1_i64, %c10242_i64, %c0_i64, %c10242_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 204840 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = 0 : i64, ub_capacity = 10368 : i64}> : (memref<5184xbf16, 1>, memref<102420xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
      ascvec.release_buf mte2, %c0_i64, 0 {cannir.buffer.generated_sync}
      "ascvec.copy_gm2ub_plan"(%alloc_8, %alloc_0, %c0_i32, %c0_i32, %c0_i64, %33, %c1_i64, %c10242_i64, %c0_i64, %c10242_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 204840 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = 31104 : i64, ub_capacity = 10368 : i64}> : (memref<5184xbf16, 1>, memref<102420xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
      %34 = arith.muli %3, %c20484_i64 : i64
      "ascvec.copy_gm2ub_plan"(%alloc_14, %alloc_2, %c0_i32, %c0_i32, %c0_i64, %34, %c1_i64, %c20484_i64, %c0_i64, %c20484_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 81936 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 93312 : i64, ub_capacity = 20736 : i64}> : (memref<5184xf32, 1>, memref<20484xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
      %35 = arith.muli %1, %c10242_i64 : i64
      ascvec.get_buf mte2, %c4_i64, 0 {cannir.buffer.generated_sync}
      "ascvec.copy_gm2ub_plan"(%alloc_11, %alloc_1, %c0_i32, %c0_i32, %c0_i64, %35, %c1_i64, %c10242_i64, %c0_i64, %c10242_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 51210 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = 62208 : i64, ub_capacity = 10368 : i64}> : (memref<5184xbf16, 1>, memref<25605xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
      ascvec.release_buf mte2, %c4_i64, 0 {cannir.buffer.generated_sync}
    }
    %8 = arith.addi %0, %c20_i64 : i64
    %9 = arith.remsi %0, %c4_i64 : i64
    %10 = arith.cmpi ne, %9, %c0_i64 : i64
    %11 = arith.cmpi slt, %9, %c0_i64 : i64
    %12 = arith.andi %10, %11 : i1
    %13 = arith.extui %12 : i1 to i64
    %14 = arith.subi %1, %13 : i64
    %15 = arith.remsi %8, %c4_i64 : i64
    %16 = arith.cmpi ne, %15, %c0_i64 : i64
    %17 = arith.cmpi slt, %15, %c0_i64 : i64
    %18 = arith.andi %16, %17 : i1
    %19 = arith.divsi %8, %c4_i64 : i64
    %20 = arith.extui %18 : i1 to i64
    %21 = arith.subi %19, %20 : i64
    %22 = arith.muli %14, %c4_i64 : i64
    %23 = arith.subi %0, %22 : i64
    %24 = arith.muli %21, %c4_i64 : i64
    %25 = arith.subi %8, %24 : i64
    %26 = arith.cmpi slt, %8, %c20_i64 : i64
    scf.if %26 {
      %31 = arith.index_cast %21 : i64 to index
      %32 = memref.load %alloc_4[%31] : memref<5xui8>
      %33 = cannir.scalar_storage_cast %32 : ui8 -> i8
      %34 = arith.cmpi ne, %33, %c0_i8 : i8
      scf.if %34 {
        %35 = arith.muli %21, %c40968_i64 : i64
        %36 = arith.muli %25, %c10242_i64 : i64
        %37 = arith.addi %35, %36 : i64
        "ascvec.copy_gm2ub_plan"(%alloc_6, %alloc, %c0_i32, %c0_i32, %c0_i64, %37, %c1_i64, %c10242_i64, %c0_i64, %c10242_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 204840 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = 10368 : i64, ub_capacity = 10368 : i64}> : (memref<5184xbf16, 1>, memref<102420xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
      } else {
        %35 = arith.muli %21, %c10242_i64 : i64
        "ascvec.copy_gm2ub_plan"(%alloc_12, %alloc_1, %c0_i32, %c0_i32, %c0_i64, %35, %c1_i64, %c10242_i64, %c0_i64, %c10242_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 51210 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = 72576 : i64, ub_capacity = 10368 : i64}> : (memref<5184xbf16, 1>, memref<25605xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
        %36 = arith.muli %21, %c40968_i64 : i64
        %37 = arith.muli %25, %c10242_i64 : i64
        %38 = arith.addi %36, %37 : i64
        "ascvec.copy_gm2ub_plan"(%alloc_6, %alloc, %c0_i32, %c0_i32, %c0_i64, %38, %c1_i64, %c10242_i64, %c0_i64, %c10242_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 204840 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = 10368 : i64, ub_capacity = 10368 : i64}> : (memref<5184xbf16, 1>, memref<102420xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
        "ascvec.copy_gm2ub_plan"(%alloc_9, %alloc_0, %c0_i32, %c0_i32, %c0_i64, %38, %c1_i64, %c10242_i64, %c0_i64, %c10242_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 204840 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = 41472 : i64, ub_capacity = 10368 : i64}> : (memref<5184xbf16, 1>, memref<102420xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
      }
    }
    %27 = arith.index_cast %14 : i64 to index
    %28 = memref.load %alloc_4[%27] : memref<5xui8>
    %29 = cannir.scalar_storage_cast %28 : ui8 -> i8
    %30 = arith.cmpi ne, %29, %c0_i8 : i8
    scf.if %30 {
      ascvec.get_buf v, %c7_i64, 0 {cannir.buffer.generated_sync}
      ascvec.get_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
      ascvec.copy_ub2ub %alloc_15[%c0], %alloc_5[%c0], %c10368_i32 : memref<5184xbf16, 1>, memref<5184xbf16, 1>
      ascvec.vf(%alloc_15, %alloc_5 : memref<5184xbf16, 1>, memref<5184xbf16, 1>) {
      ^bb0(%arg8: memref<5184xbf16, 1>, %arg9: memref<5184xbf16, 1>):
      } {cannir.vf_mode = "simd"}
      ascvec.release_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
      ascvec.release_buf v, %c7_i64, 0 {cannir.buffer.generated_sync}
      %31 = arith.muli %14, %c20484_i64 : i64
      %32 = arith.muli %23, %c5121_i64 : i64
      %33 = arith.addi %31, %32 : i64
      %34 = arith.muli %33, %c16_i64 : i64
      %35 = ascscalar.memref_address %alloc_3 : memref<102420xbf16>
      %36 = arith.divui %34, %c8_i64 : i64
      %37 = arith.addi %35, %36 : i64
      %38 = ascscalar.bind_gm_ptr %37 : memref<1xbf16>
      %39 = ascscalar.bitcast_memref %alloc_15 byte_offset = %c0_i64 : memref<5184xbf16, 1> -> memref<?xbf16, 1>
      ascvec.get_buf mte3, %c7_i64, 0 {cannir.buffer.generated_sync}
      ascvec.copy_ub2gm %38[%c0], %39[%c0], %c1_i32, %c10242_i32, %c0_i64, %c10242_i32 l2_cache_ctl %c0_i32 : memref<1xbf16>, memref<?xbf16, 1>
      ascvec.release_buf mte3, %c7_i64, 0 {cannir.buffer.generated_sync}
    } else {
      ascvec.get_buf v, %c6_i64, 0 {cannir.buffer.generated_sync}
      ascvec.get_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
      ascvec.get_buf v, %c7_i64, 0 {cannir.buffer.generated_sync}
      ascvec.vf(%alloc_5, %alloc_17, %alloc_8, %alloc_14, %arg5, %arg6, %alloc_11, %alloc_15 : memref<5184xbf16, 1>, memref<5184xf32, 1>, memref<5184xbf16, 1>, memref<5184xf32, 1>, f32, f32, memref<5184xbf16, 1>, memref<5184xbf16, 1>) {
      ^bb0(%arg8: memref<5184xbf16, 1>, %arg9: memref<5184xf32, 1>, %arg10: memref<5184xbf16, 1>, %arg11: memref<5184xf32, 1>, %arg12: f32, %arg13: f32, %arg14: memref<5184xbf16, 1>, %arg15: memref<5184xbf16, 1>):
        %c5120 = arith.constant 5120 : index
        %cst = arith.constant -1.000000e+00 : f32
        %cst_18 = arith.constant 0.0139740603 : f32
        %cst_19 = arith.constant 1.95274362E-4 : f32
        %cst_20 = arith.constant 1.000000e+00 : f32
        %c80_i32 = arith.constant 80 : i32
        %c0_i32_21 = arith.constant 0 : i32
        %c64_i32 = arith.constant 64 : i32
        %cst_22 = arith.constant 0.000000e+00 : f32
        %c1_i32_23 = arith.constant 1 : i32
        %40 = ascvec.create_mask : !ascvec.mask
        %mask, %remaining_out = ascvec.update_mask %c1_i32_23 : !ascvec.mask
        %41 = ascvec.duplicate_scalar %cst_22, %40 : f32 -> <f32>
        %42:3 = scf.for %arg16 = %c0_i32_21 to %c80_i32 step %c1_i32_23 iter_args(%arg17 = %41, %arg18 = %41, %arg19 = %41) -> (!ascvec.vreg<f32>, !ascvec.vreg<f32>, !ascvec.vreg<f32>)  : i32 {
          %82 = arith.muli %arg16, %c64_i32 overflow<nsw> : i32
          %83 = arith.index_cast %82 : i32 to index
          %84 = ascvec.loadalign_unpack %arg8[%83] {mode = "b16_to_b32"} : memref<5184xbf16, 1>, index -> <bf16>
          %85 = ascvec.cast %84, %40 {rounding = 6 : i32} : <bf16> -> <f32>
          ascvec.storealign %arg9[%83], %85, %40 : memref<5184xf32, 1>, index, <f32>
          %86 = ascvec.loadalign_unpack %arg10[%83] {mode = "b16_to_b32"} : memref<5184xbf16, 1>, index -> <bf16>
          %87 = ascvec.cast %86, %40 {rounding = 6 : i32} : <bf16> -> <f32>
          %88 = ascvec.loadalign %arg11[%83] : memref<5184xf32, 1>, index -> <f32>
          %89 = ascvec.mul %85, %85, %40 : <f32>
          %90 = ascvec.add %arg17, %89, %40 : <f32>
          %91 = ascvec.mul %87, %87, %40 : <f32>
          %92 = ascvec.add %arg18, %91, %40 : <f32>
          %93 = ascvec.mul %85, %88, %40 : <f32>
          %94 = ascvec.mul %93, %87, %40 : <f32>
          %95 = ascvec.add %arg19, %94, %40 : <f32>
          scf.yield %90, %92, %95 : !ascvec.vreg<f32>, !ascvec.vreg<f32>, !ascvec.vreg<f32>
        }
        %43 = ascvec.reduce_sum %42#0, %40 : <f32>
        %44 = ascvec.reduce_sum %42#1, %40 : <f32>
        %45 = ascvec.reduce_sum %42#2, %40 : <f32>
        %mask_24, %remaining_out_25 = ascvec.update_mask %c1_i32_23 : !ascvec.mask
        %46 = ascvec.loadalign_unpack %arg8[%c5120] {mode = "b16_to_b32"} : memref<5184xbf16, 1>, index -> <bf16>
        %47 = ascvec.cast %46, %mask_24 {rounding = 6 : i32} : <bf16> -> <f32>
        %48 = ascvec.loadalign_unpack %arg10[%c5120] {mode = "b16_to_b32"} : memref<5184xbf16, 1>, index -> <bf16>
        %49 = ascvec.cast %48, %mask_24 {rounding = 6 : i32} : <bf16> -> <f32>
        %50 = ascvec.loadalign %arg11[%c5120] : memref<5184xf32, 1>, index -> <f32>
        %51 = ascvec.mul %47, %47, %mask_24 : <f32>
        %52 = ascvec.reduce_sum %51, %mask_24 : <f32>
        %53 = ascvec.add %43, %52, %mask : <f32>
        %54 = ascvec.mul %49, %49, %mask_24 : <f32>
        %55 = ascvec.reduce_sum %54, %mask_24 : <f32>
        %56 = ascvec.add %44, %55, %mask : <f32>
        %57 = ascvec.mul %47, %50, %mask_24 : <f32>
        %58 = ascvec.mul %57, %49, %mask_24 : <f32>
        %59 = ascvec.reduce_sum %58, %mask_24 : <f32>
        %60 = ascvec.add %45, %59, %mask : <f32>
        ascvec.storealign %arg9[%c5120], %47, %mask_24 : memref<5184xf32, 1>, index, <f32>
        %61 = ascvec.duplicate_scalar %cst_20, %mask : f32 -> <f32>
        %62 = ascvec.mul_scalar %53, %cst_19, %mask : <f32>, f32
        %63 = ascvec.add_scalar %62, %arg12, %mask : <f32>, f32
        %64 = ascvec.sqrt %63, %mask : <f32>
        %65 = ascvec.mul_scalar %56, %cst_19, %mask : <f32>, f32
        %66 = ascvec.add_scalar %65, %arg12, %mask : <f32>, f32
        %67 = ascvec.sqrt %66, %mask : <f32>
        %68 = ascvec.mul_scalar %60, %cst_18, %mask : <f32>, f32
        %69 = ascvec.mul %64, %67, %mask : <f32>
        %70 = ascvec.div %68, %69, %mask : <f32>
        %71 = ascvec.abs %70, %mask : <f32>
        %72 = ascvec.max_scalar %71, %arg13, %mask : <f32>, f32
        %73 = ascvec.sqrt %72, %mask : <f32>
        %74 = ascvec.cmp_lt_scalar %70, %cst_22, %mask : <f32>, f32 -> !ascvec.mask
        %75 = ascvec.mul_scalar %73, %cst, %mask : <f32>, f32
        %76 = ascvec.select %75, %73, %74 : <f32>
        %77 = ascvec.mul_scalar %76, %cst, %mask : <f32>, f32
        %78 = ascvec.exp %77, %mask : <f32>
        %79 = ascvec.add %61, %78, %mask : <f32>
        %80 = ascvec.div %61, %79, %mask : <f32>
        %81 = ascvec.duplicate %80, %40 : <f32> -> <f32>
        scf.for %arg16 = %c0_i32_21 to %c80_i32 step %c1_i32_23  : i32 {
          %82 = arith.muli %arg16, %c64_i32 overflow<nsw> : i32
          %83 = arith.index_cast %82 : i32 to index
          %84 = ascvec.loadalign %arg9[%83] : memref<5184xf32, 1>, index -> <f32>
          %85 = ascvec.loadalign_unpack %arg14[%83] {mode = "b16_to_b32"} : memref<5184xbf16, 1>, index -> <bf16>
          %86 = ascvec.cast %85, %40 {rounding = 6 : i32} : <bf16> -> <f32>
          %87 = ascvec.mul %81, %86, %40 : <f32>
          %88 = ascvec.add %84, %87, %40 : <f32>
          %89 = ascvec.cast %88, %40 : <f32> -> <bf16>
          ascvec.storealign_pack %arg15[%83], %89, %40 {mode = "b32_to_b16"} : memref<5184xbf16, 1>, index, <bf16>
        }
        %c0_26 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        scf.for %arg16 = %c0_26 to %c1 step %c1 {
          %mask_27, %remaining_out_28 = ascvec.update_mask %c1_i32_23 : !ascvec.mask
          %82 = ascvec.loadalign %arg9[%c5120] : memref<5184xf32, 1>, index -> <f32>
          %83 = ascvec.loadalign_unpack %arg14[%c5120] {mode = "b16_to_b32"} : memref<5184xbf16, 1>, index -> <bf16>
          %84 = ascvec.cast %83, %mask_27 {rounding = 6 : i32} : <bf16> -> <f32>
          %85 = ascvec.mul %81, %84, %mask_27 : <f32>
          %86 = ascvec.add %82, %85, %mask_27 : <f32>
          %87 = ascvec.cast %86, %mask_27 : <f32> -> <bf16>
          ascvec.storealign_pack %arg15[%c5120], %87, %mask_27 {mode = "b32_to_b16"} : memref<5184xbf16, 1>, index, <bf16>
        }
        ascvec.mem_bar vst_vld
      } {cannir.vf_mode = "simd"}
      ascvec.release_buf v, %c7_i64, 0 {cannir.buffer.generated_sync}
      ascvec.release_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
      ascvec.release_buf v, %c6_i64, 0 {cannir.buffer.generated_sync}
      %31 = arith.muli %14, %c20484_i64 : i64
      %32 = arith.muli %23, %c5121_i64 : i64
      %33 = arith.addi %31, %32 : i64
      %34 = arith.muli %33, %c16_i64 : i64
      %35 = ascscalar.memref_address %alloc_3 : memref<102420xbf16>
      %36 = arith.divui %34, %c8_i64 : i64
      %37 = arith.addi %35, %36 : i64
      %38 = ascscalar.bind_gm_ptr %37 : memref<1xbf16>
      %39 = ascscalar.bitcast_memref %alloc_15 byte_offset = %c0_i64 : memref<5184xbf16, 1> -> memref<?xbf16, 1>
      ascvec.get_buf mte3, %c7_i64, 0 {cannir.buffer.generated_sync}
      ascvec.copy_ub2gm %38[%c0], %39[%c0], %c1_i32, %c10242_i32, %c0_i64, %c10242_i32 l2_cache_ctl %c0_i32 : memref<1xbf16>, memref<?xbf16, 1>
      ascvec.release_buf mte3, %c7_i64, 0 {cannir.buffer.generated_sync}
    }
    scf.if %26 {
      %31 = arith.index_cast %21 : i64 to index
      %32 = memref.load %alloc_4[%31] : memref<5xui8>
      %33 = cannir.scalar_storage_cast %32 : ui8 -> i8
      %34 = arith.cmpi ne, %33, %c0_i8 : i8
      scf.if %34 {
        ascvec.copy_ub2ub %alloc_16[%c0], %alloc_6[%c0], %c10368_i32 : memref<5184xbf16, 1>, memref<5184xbf16, 1>
        ascvec.vf(%alloc_16, %alloc_6 : memref<5184xbf16, 1>, memref<5184xbf16, 1>) {
        ^bb0(%arg8: memref<5184xbf16, 1>, %arg9: memref<5184xbf16, 1>):
        } {cannir.vf_mode = "simd"}
        %35 = arith.muli %21, %c20484_i64 : i64
        %36 = arith.muli %25, %c5121_i64 : i64
        %37 = arith.addi %35, %36 : i64
        %38 = arith.muli %37, %c16_i64 : i64
        %39 = ascscalar.memref_address %alloc_3 : memref<102420xbf16>
        %40 = arith.divui %38, %c8_i64 : i64
        %41 = arith.addi %39, %40 : i64
        %42 = ascscalar.bind_gm_ptr %41 : memref<1xbf16>
        %43 = ascscalar.bitcast_memref %alloc_16 byte_offset = %c0_i64 : memref<5184xbf16, 1> -> memref<?xbf16, 1>
        ascvec.copy_ub2gm %42[%c0], %43[%c0], %c1_i32, %c10242_i32, %c0_i64, %c10242_i32 l2_cache_ctl %c0_i32 : memref<1xbf16>, memref<?xbf16, 1>
      } else {
        ascvec.vf(%alloc_6, %alloc_17, %alloc_9, %alloc_14, %arg5, %arg6, %alloc_12, %alloc_16 : memref<5184xbf16, 1>, memref<5184xf32, 1>, memref<5184xbf16, 1>, memref<5184xf32, 1>, f32, f32, memref<5184xbf16, 1>, memref<5184xbf16, 1>) {
        ^bb0(%arg8: memref<5184xbf16, 1>, %arg9: memref<5184xf32, 1>, %arg10: memref<5184xbf16, 1>, %arg11: memref<5184xf32, 1>, %arg12: f32, %arg13: f32, %arg14: memref<5184xbf16, 1>, %arg15: memref<5184xbf16, 1>):
          %c5120 = arith.constant 5120 : index
          %cst = arith.constant -1.000000e+00 : f32
          %cst_18 = arith.constant 0.0139740603 : f32
          %cst_19 = arith.constant 1.95274362E-4 : f32
          %cst_20 = arith.constant 1.000000e+00 : f32
          %c80_i32 = arith.constant 80 : i32
          %c0_i32_21 = arith.constant 0 : i32
          %c64_i32 = arith.constant 64 : i32
          %cst_22 = arith.constant 0.000000e+00 : f32
          %c1_i32_23 = arith.constant 1 : i32
          %44 = ascvec.create_mask : !ascvec.mask
          %mask, %remaining_out = ascvec.update_mask %c1_i32_23 : !ascvec.mask
          %45 = ascvec.duplicate_scalar %cst_22, %44 : f32 -> <f32>
          %46:3 = scf.for %arg16 = %c0_i32_21 to %c80_i32 step %c1_i32_23 iter_args(%arg17 = %45, %arg18 = %45, %arg19 = %45) -> (!ascvec.vreg<f32>, !ascvec.vreg<f32>, !ascvec.vreg<f32>)  : i32 {
            %86 = arith.muli %arg16, %c64_i32 overflow<nsw> : i32
            %87 = arith.index_cast %86 : i32 to index
            %88 = ascvec.loadalign_unpack %arg8[%87] {mode = "b16_to_b32"} : memref<5184xbf16, 1>, index -> <bf16>
            %89 = ascvec.cast %88, %44 {rounding = 6 : i32} : <bf16> -> <f32>
            ascvec.storealign %arg9[%87], %89, %44 : memref<5184xf32, 1>, index, <f32>
            %90 = ascvec.loadalign_unpack %arg10[%87] {mode = "b16_to_b32"} : memref<5184xbf16, 1>, index -> <bf16>
            %91 = ascvec.cast %90, %44 {rounding = 6 : i32} : <bf16> -> <f32>
            %92 = ascvec.loadalign %arg11[%87] : memref<5184xf32, 1>, index -> <f32>
            %93 = ascvec.mul %89, %89, %44 : <f32>
            %94 = ascvec.add %arg17, %93, %44 : <f32>
            %95 = ascvec.mul %91, %91, %44 : <f32>
            %96 = ascvec.add %arg18, %95, %44 : <f32>
            %97 = ascvec.mul %89, %92, %44 : <f32>
            %98 = ascvec.mul %97, %91, %44 : <f32>
            %99 = ascvec.add %arg19, %98, %44 : <f32>
            scf.yield %94, %96, %99 : !ascvec.vreg<f32>, !ascvec.vreg<f32>, !ascvec.vreg<f32>
          }
          %47 = ascvec.reduce_sum %46#0, %44 : <f32>
          %48 = ascvec.reduce_sum %46#1, %44 : <f32>
          %49 = ascvec.reduce_sum %46#2, %44 : <f32>
          %mask_24, %remaining_out_25 = ascvec.update_mask %c1_i32_23 : !ascvec.mask
          %50 = ascvec.loadalign_unpack %arg8[%c5120] {mode = "b16_to_b32"} : memref<5184xbf16, 1>, index -> <bf16>
          %51 = ascvec.cast %50, %mask_24 {rounding = 6 : i32} : <bf16> -> <f32>
          %52 = ascvec.loadalign_unpack %arg10[%c5120] {mode = "b16_to_b32"} : memref<5184xbf16, 1>, index -> <bf16>
          %53 = ascvec.cast %52, %mask_24 {rounding = 6 : i32} : <bf16> -> <f32>
          %54 = ascvec.loadalign %arg11[%c5120] : memref<5184xf32, 1>, index -> <f32>
          %55 = ascvec.mul %51, %51, %mask_24 : <f32>
          %56 = ascvec.reduce_sum %55, %mask_24 : <f32>
          %57 = ascvec.add %47, %56, %mask : <f32>
          %58 = ascvec.mul %53, %53, %mask_24 : <f32>
          %59 = ascvec.reduce_sum %58, %mask_24 : <f32>
          %60 = ascvec.add %48, %59, %mask : <f32>
          %61 = ascvec.mul %51, %54, %mask_24 : <f32>
          %62 = ascvec.mul %61, %53, %mask_24 : <f32>
          %63 = ascvec.reduce_sum %62, %mask_24 : <f32>
          %64 = ascvec.add %49, %63, %mask : <f32>
          ascvec.storealign %arg9[%c5120], %51, %mask_24 : memref<5184xf32, 1>, index, <f32>
          %65 = ascvec.duplicate_scalar %cst_20, %mask : f32 -> <f32>
          %66 = ascvec.mul_scalar %57, %cst_19, %mask : <f32>, f32
          %67 = ascvec.add_scalar %66, %arg12, %mask : <f32>, f32
          %68 = ascvec.sqrt %67, %mask : <f32>
          %69 = ascvec.mul_scalar %60, %cst_19, %mask : <f32>, f32
          %70 = ascvec.add_scalar %69, %arg12, %mask : <f32>, f32
          %71 = ascvec.sqrt %70, %mask : <f32>
          %72 = ascvec.mul_scalar %64, %cst_18, %mask : <f32>, f32
          %73 = ascvec.mul %68, %71, %mask : <f32>
          %74 = ascvec.div %72, %73, %mask : <f32>
          %75 = ascvec.abs %74, %mask : <f32>
          %76 = ascvec.max_scalar %75, %arg13, %mask : <f32>, f32
          %77 = ascvec.sqrt %76, %mask : <f32>
          %78 = ascvec.cmp_lt_scalar %74, %cst_22, %mask : <f32>, f32 -> !ascvec.mask
          %79 = ascvec.mul_scalar %77, %cst, %mask : <f32>, f32
          %80 = ascvec.select %79, %77, %78 : <f32>
          %81 = ascvec.mul_scalar %80, %cst, %mask : <f32>, f32
          %82 = ascvec.exp %81, %mask : <f32>
          %83 = ascvec.add %65, %82, %mask : <f32>
          %84 = ascvec.div %65, %83, %mask : <f32>
          %85 = ascvec.duplicate %84, %44 : <f32> -> <f32>
          scf.for %arg16 = %c0_i32_21 to %c80_i32 step %c1_i32_23  : i32 {
            %86 = arith.muli %arg16, %c64_i32 overflow<nsw> : i32
            %87 = arith.index_cast %86 : i32 to index
            %88 = ascvec.loadalign %arg9[%87] : memref<5184xf32, 1>, index -> <f32>
            %89 = ascvec.loadalign_unpack %arg14[%87] {mode = "b16_to_b32"} : memref<5184xbf16, 1>, index -> <bf16>
            %90 = ascvec.cast %89, %44 {rounding = 6 : i32} : <bf16> -> <f32>
            %91 = ascvec.mul %85, %90, %44 : <f32>
            %92 = ascvec.add %88, %91, %44 : <f32>
            %93 = ascvec.cast %92, %44 : <f32> -> <bf16>
            ascvec.storealign_pack %arg15[%87], %93, %44 {mode = "b32_to_b16"} : memref<5184xbf16, 1>, index, <bf16>
          }
          %c0_26 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          scf.for %arg16 = %c0_26 to %c1 step %c1 {
            %mask_27, %remaining_out_28 = ascvec.update_mask %c1_i32_23 : !ascvec.mask
            %86 = ascvec.loadalign %arg9[%c5120] : memref<5184xf32, 1>, index -> <f32>
            %87 = ascvec.loadalign_unpack %arg14[%c5120] {mode = "b16_to_b32"} : memref<5184xbf16, 1>, index -> <bf16>
            %88 = ascvec.cast %87, %mask_27 {rounding = 6 : i32} : <bf16> -> <f32>
            %89 = ascvec.mul %85, %88, %mask_27 : <f32>
            %90 = ascvec.add %86, %89, %mask_27 : <f32>
            %91 = ascvec.cast %90, %mask_27 : <f32> -> <bf16>
            ascvec.storealign_pack %arg15[%c5120], %91, %mask_27 {mode = "b32_to_b16"} : memref<5184xbf16, 1>, index, <bf16>
          }
          ascvec.mem_bar vst_vld
        } {cannir.vf_mode = "simd"}
        %35 = arith.muli %21, %c20484_i64 : i64
        %36 = arith.muli %25, %c5121_i64 : i64
        %37 = arith.addi %35, %36 : i64
        %38 = arith.muli %37, %c16_i64 : i64
        %39 = ascscalar.memref_address %alloc_3 : memref<102420xbf16>
        %40 = arith.divui %38, %c8_i64 : i64
        %41 = arith.addi %39, %40 : i64
        %42 = ascscalar.bind_gm_ptr %41 : memref<1xbf16>
        %43 = ascscalar.bitcast_memref %alloc_16 byte_offset = %c0_i64 : memref<5184xbf16, 1> -> memref<?xbf16, 1>
        ascvec.copy_ub2gm %42[%c0], %43[%c0], %c1_i32, %c10242_i32, %c0_i64, %c10242_i32 l2_cache_ctl %c0_i32 : memref<1xbf16>, memref<?xbf16, 1>
      }
    }
    return
  }
}
