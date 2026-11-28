module {
  func.func @run_5fb7a17ab6006e2954ff148c3d99b43b(%arg0: !cannir.tensor<<gm, f32>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, identity>, %arg1: !cannir.tensor<<gm, f16>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>, %arg2: !cannir.tensor<<gm, f16>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>, %arg3: !cannir.tensor<<gm, f32>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>, %arg4: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg5: i64) attributes {cannbotdsl.contract_digest = "5fb7a17ab6006e2954ff148c3d99b43b044d1ac108692b533b4c0e7bb64696ab", cannbotdsl.public_slot_count = 6 : i64, cannbotdsl.slot_dtype_bits = [32, 16, 16, 32, 32, 64], cannbotdsl.slot_dtype_codes = [2, 2, 2, 2, 0, 0], cannbotdsl.slot_dtype_lanes = [1, 1, 1, 1, 1, 1], cannbotdsl.slot_flags = [1, 1, 1, 1, 1, 1], cannbotdsl.slot_kinds = [1, 1, 1, 1, 1, 2], cannbotdsl.slot_type_digests = ["6203b5c2fd572266215af8ab68bc5e7f370a1316e8c9ca34e9b9f659e4e619c9", "8b511af123417f688dd19a281999e1ab5cc687d95dd9b1a7d0732dd6e9251029", "8b511af123417f688dd19a281999e1ab5cc687d95dd9b1a7d0732dd6e9251029", "4c9841ce281cf0606227f9d22c059468a95d1dc5d2fe922a197ce615a948b03e", "a4a79c7180d0853b295fbf90ad8be6fc94481e2439cabb72f1568851b3c40efc", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515"], cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.host} {
    cannir.kernel_launch @kernel_kernel_0(%arg5) [%arg0, %arg1, %arg2, %arg3, %arg4] {cannbotdsl.profile = {indices = array<i64: 0, 0, 1, 2, 3, 4>, name = "kernel", op_type = "kernel", roles = array<i64: 0, 1, 0, 0, 0, 0>}} : !cannir.tensor<<gm, f32>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, identity>, !cannir.tensor<<gm, f16>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>, !cannir.tensor<<gm, f16>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>, !cannir.tensor<<gm, f32>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>, !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>
    return
  }
  func.func @kernel_kernel_0(%arg0: !cannir.tensor<<gm, f32>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, identity>, %arg1: !cannir.tensor<<gm, f16>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>, %arg2: !cannir.tensor<<gm, f16>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>, %arg3: !cannir.tensor<<gm, f32>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>, %arg4: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>) attributes {cannir.buffer.auto_sync, cannir.buffer.sync_analyzed, cannir.buffer.sync_verified, cannir.buffer.tensor_state, cannir.core = "aiv", cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.device, cannir.explicit_ring_buffer_cursors, cannir.kernel_group = "kernel_kernel_0", cannir.resource.ids_prepared} {
    %c65536_i32 = arith.constant 65536 : i32
    %c32768_i32 = arith.constant 32768 : i32
    %c16384_i32 = arith.constant 16384 : i32
    %c16_i64 = arith.constant 16 : i64
    %c0_i32 = arith.constant 0 : i32
    %c128_i64 = arith.constant 128 : i64
    %c32_i64 = arith.constant 32 : i64
    %c65536_i64 = arith.constant 65536 : i64
    %c16384_i64 = arith.constant 16384 : i64
    %c512_i64 = arith.constant 512 : i64
    %c256_i64 = arith.constant 256 : i64
    %c64_i64 = arith.constant 64 : i64
    %c1_i32 = arith.constant 1 : i32
    %c0 = arith.constant 0 : index
    %c32768_i64 = arith.constant 32768 : i64
    %false = arith.constant false
    %c4096_i64 = arith.constant 4096 : i64
    %c8192_i64 = arith.constant 8192 : i64
    %c7_i64 = arith.constant 7 : i64
    %c6_i64 = arith.constant 6 : i64
    %c5_i64 = arith.constant 5 : i64
    %c4_i64 = arith.constant 4 : i64
    %c3_i64 = arith.constant 3 : i64
    %c2_i64 = arith.constant 2 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %c8_i64 = arith.constant 8 : i64
    %alloc = memref.alloc() {cannir.arg_index = 0 : i64} : memref<1xf32>
    %alloc_0 = memref.alloc() {cannir.arg_index = 1 : i64} : memref<1xf16>
    %alloc_1 = memref.alloc() {cannir.arg_index = 2 : i64} : memref<1xf16>
    %alloc_2 = memref.alloc() {cannir.arg_index = 3 : i64} : memref<1xf32>
    %alloc_3 = memref.alloc() {cannir.arg_index = 4 : i64} : memref<1xi32>
    %alloc_4 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<16384xf32, 1>
    %alloc_5 = memref.alloc() {cannir.byte_offset = 65536 : i64} : memref<16384xf32, 1>
    %alloc_6 = memref.alloc() {cannir.byte_offset = 131072 : i64} : memref<128xf16, 1>
    %alloc_7 = memref.alloc() {cannir.byte_offset = 131328 : i64} : memref<128xf16, 1>
    %alloc_8 = memref.alloc() {cannir.byte_offset = 131584 : i64} : memref<128xf16, 1>
    %alloc_9 = memref.alloc() {cannir.byte_offset = 131840 : i64} : memref<128xf16, 1>
    %alloc_10 = memref.alloc() {cannir.byte_offset = 132096 : i64} : memref<128xf32, 1>
    %alloc_11 = memref.alloc() {cannir.byte_offset = 132608 : i64} : memref<128xf32, 1>
    %alloc_12 = memref.alloc() {cannir.byte_offset = 133120 : i64} : memref<128xf32, 1>
    %0 = cannir.extract_layout %arg0 : !cannir.tensor<<gm, f32>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, identity>
    %1 = cannir.extract_shape %0 : <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>> -> <(-1, -1, 128, 128)>
    %2 = cannir.extract_dim %1[0] : !cannir.shape<(-1, -1, 128, 128)>
    %3 = cannir.extract_dim %1[1] : !cannir.shape<(-1, -1, 128, 128)>
    %4 = arith.muli %2, %3 : i64
    %5 = arith.cmpi sle, %4, %c16_i64 : i64
    scf.if %5 {
      %6 = cannir.extract_layout %arg1 : !cannir.tensor<<gm, f16>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
      %7 = cannir.extract_shape %6 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, -1, 128)>
      %8 = cannir.extract_dim %7[1] : !cannir.shape<(-1, -1, -1, 128)>
      %9 = arith.muli %4, %c4_i64 : i64
      %10 = ascscalar.get_block_idx
      %11 = ascscalar.get_block_num
      %12:2 = scf.for %arg5 = %10 to %9 step %11 iter_args(%arg6 = %c0_i64, %arg7 = %c0_i64) -> (i64, i64)  : i64 {
        %13 = arith.remsi %arg5, %c4_i64 : i64
        %14 = arith.cmpi ne, %13, %c0_i64 : i64
        %15 = arith.cmpi slt, %13, %c0_i64 : i64
        %16 = arith.andi %14, %15 : i1
        %17 = arith.divsi %arg5, %c4_i64 : i64
        %18 = arith.extui %16 : i1 to i64
        %19 = arith.subi %17, %18 : i64
        %20 = arith.muli %19, %c4_i64 : i64
        %21 = arith.subi %arg5, %20 : i64
        %22 = arith.maxsi %3, %c1_i64 : i64
        %23 = arith.remsi %19, %22 : i64
        %24 = arith.cmpi slt, %23, %c0_i64 : i64
        %25 = arith.divsi %19, %22 : i64
        %26 = arith.extui %24 : i1 to i64
        %27 = arith.subi %25, %26 : i64
        %28 = arith.maxsi %2, %c1_i64 : i64
        %29 = arith.remsi %27, %28 : i64
        %30 = arith.cmpi slt, %29, %c0_i64 : i64
        %31 = arith.select %30, %28, %c0_i64 : i64
        %32 = arith.addi %29, %31 : i64
        %33 = arith.select %24, %22, %c0_i64 : i64
        %34 = arith.addi %23, %33 : i64
        %35 = arith.index_cast %32 : i64 to index
        %36 = memref.load %alloc_3[%35] : memref<1xi32>
        %37 = arith.extsi %36 : i32 to i64
        %38 = arith.cmpi sgt, %37, %c0_i64 : i64
        %39:2 = scf.if %38 -> (i64, i64) {
          %40 = arith.minsi %21, %c4_i64 : i64
          %41 = arith.muli %40, %c32_i64 : i64
          %42 = arith.subi %c128_i64, %41 : i64
          %43 = arith.minsi %42, %c32_i64 : i64
          %44 = arith.cmpi sle, %21, %c4_i64 : i64
          %45 = arith.select %44, %43, %c0_i64 : i64
          %46 = cannir.extract_physical_layout %arg0 : !cannir.tensor<<gm, f32>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, identity>
          %47 = cannir.extract_stride %46 : <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>> -> <(-1, 16384, 128, 1)>
          %48 = cannir.extract_dim %47[0] : !cannir.stride<(-1, 16384, 128, 1)>
          %49 = arith.muli %32, %48 : i64
          %50 = arith.muli %49, %c4_i64 : i64
          %51 = arith.muli %34, %c65536_i64 : i64
          %52 = arith.addi %50, %51 : i64
          %53 = arith.muli %21, %c16384_i64 : i64
          %54 = arith.addi %52, %53 : i64
          %55 = arith.muli %45, %c512_i64 : i64
          %56 = arith.cmpi eq, %55, %c0_i64 : i64
          %57 = arith.cmpi ne, %55, %c0_i64 : i64
          %58 = arith.extui %57 : i1 to i64
          %59 = arith.select %56, %c4_i64, %55 : i64
          ascvec.get_buf mte2, %c0_i64, 0 {cannir.buffer.generated_sync}
          "ascvec.copy_gm2ub_plan"(%alloc_4, %alloc, %c0_i32, %c0_i32, %c0_i64, %54, %58, %59, %c0_i64, %59, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 0 : i64, ub_capacity = 65536 : i64}> : (memref<16384xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
          ascvec.release_buf mte2, %c0_i64, 0 {cannir.buffer.generated_sync}
          %60 = arith.cmpi eq, %arg6, %c0_i64 : i64
          %61 = ascscalar.memref_address %alloc_6 : memref<128xf16, 1>
          %62 = ascscalar.memref_address %alloc_7 : memref<128xf16, 1>
          %63 = arith.select %60, %61, %62 : i64
          %64 = arith.select %60, %c2_i64, %c3_i64 : i64
          %65 = arith.addi %arg6, %c1_i64 : i64
          %66 = arith.remui %65, %c2_i64 : i64
          %67 = cannir.extract_dim %7[2] : !cannir.shape<(-1, -1, -1, 128)>
          %68 = arith.minsi %34, %67 : i64
          %69 = arith.subi %67, %68 : i64
          %70 = arith.minsi %69, %c1_i64 : i64
          %71 = arith.cmpi sle, %34, %67 : i64
          %72 = arith.select %71, %70, %c0_i64 : i64
          %73 = cannir.extract_physical_layout %arg1 : !cannir.tensor<<gm, f16>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
          %74 = cannir.extract_stride %73 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, 128, 1)>
          %75 = cannir.extract_dim %74[0] : !cannir.stride<(-1, -1, 128, 1)>
          %76 = arith.muli %32, %75 : i64
          %77 = arith.muli %76, %c2_i64 : i64
          %78 = arith.muli %34, %c256_i64 : i64
          %79 = arith.addi %77, %78 : i64
          %80 = arith.muli %21, %c64_i64 : i64
          %81 = arith.addi %79, %80 : i64
          %82 = arith.muli %45, %c2_i64 : i64
          %83 = arith.cmpi eq, %82, %c0_i64 : i64
          %84 = arith.select %83, %c0_i64, %72 : i64
          %85 = arith.select %83, %c2_i64, %82 : i64
          %86 = ascscalar.bind_local_ptr %63 : memref<1xf16, 1>
          ascvec.get_buf mte2, %64, 0 {cannir.buffer.generated_sync}
          "ascvec.copy_gm2ub_plan"(%86, %alloc_0, %c0_i32, %c0_i32, %c0_i64, %81, %84, %85, %c256_i64, %c64_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xf16, 1>, memref<1xf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
          ascvec.release_buf mte2, %64, 0 {cannir.buffer.generated_sync}
          %87 = ascscalar.memref_address %alloc_8 : memref<128xf16, 1>
          %88 = ascscalar.memref_address %alloc_9 : memref<128xf16, 1>
          %89 = arith.select %60, %87, %88 : i64
          %90 = arith.select %60, %c4_i64, %c5_i64 : i64
          %91 = cannir.extract_layout %arg2 : !cannir.tensor<<gm, f16>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
          %92 = cannir.extract_shape %91 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, -1, 128)>
          %93 = cannir.extract_dim %92[2] : !cannir.shape<(-1, -1, -1, 128)>
          %94 = arith.minsi %34, %93 : i64
          %95 = arith.subi %93, %94 : i64
          %96 = arith.minsi %95, %c1_i64 : i64
          %97 = arith.cmpi sle, %34, %93 : i64
          %98 = arith.select %97, %96, %c0_i64 : i64
          %99 = cannir.extract_physical_layout %arg2 : !cannir.tensor<<gm, f16>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
          %100 = cannir.extract_stride %99 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, 128, 1)>
          %101 = cannir.extract_dim %100[0] : !cannir.stride<(-1, -1, 128, 1)>
          %102 = arith.muli %32, %101 : i64
          %103 = arith.muli %102, %c2_i64 : i64
          %104 = arith.addi %103, %78 : i64
          %105 = arith.muli %98, %c256_i64 : i64
          %106 = arith.cmpi eq, %105, %c0_i64 : i64
          %107 = arith.cmpi ne, %105, %c0_i64 : i64
          %108 = arith.extui %107 : i1 to i64
          %109 = arith.select %106, %c2_i64, %105 : i64
          %110 = ascscalar.bind_local_ptr %89 : memref<1xf16, 1>
          ascvec.get_buf mte2, %90, 0 {cannir.buffer.generated_sync}
          "ascvec.copy_gm2ub_plan"(%110, %alloc_1, %c0_i32, %c0_i32, %c0_i64, %104, %108, %109, %c0_i64, %109, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xf16, 1>, memref<1xf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
          ascvec.release_buf mte2, %90, 0 {cannir.buffer.generated_sync}
          %111 = ascscalar.memref_address %alloc_10 : memref<128xf32, 1>
          %112 = ascscalar.memref_address %alloc_11 : memref<128xf32, 1>
          %113 = arith.select %60, %111, %112 : i64
          %114 = arith.select %60, %c6_i64, %c7_i64 : i64
          %115 = cannir.extract_layout %arg3 : !cannir.tensor<<gm, f32>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
          %116 = cannir.extract_shape %115 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, -1, 128)>
          %117 = cannir.extract_dim %116[2] : !cannir.shape<(-1, -1, -1, 128)>
          %118 = arith.minsi %34, %117 : i64
          %119 = arith.subi %117, %118 : i64
          %120 = arith.minsi %119, %c1_i64 : i64
          %121 = arith.cmpi sle, %34, %117 : i64
          %122 = arith.select %121, %120, %c0_i64 : i64
          %123 = cannir.extract_physical_layout %arg3 : !cannir.tensor<<gm, f32>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
          %124 = cannir.extract_stride %123 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, 128, 1)>
          %125 = cannir.extract_dim %124[0] : !cannir.stride<(-1, -1, 128, 1)>
          %126 = arith.muli %32, %125 : i64
          %127 = arith.muli %126, %c4_i64 : i64
          %128 = arith.muli %34, %c512_i64 : i64
          %129 = arith.addi %127, %128 : i64
          %130 = arith.muli %122, %c512_i64 : i64
          %131 = arith.cmpi eq, %130, %c0_i64 : i64
          %132 = arith.cmpi ne, %130, %c0_i64 : i64
          %133 = arith.extui %132 : i1 to i64
          %134 = arith.select %131, %c4_i64, %130 : i64
          %135 = ascscalar.bind_local_ptr %113 : memref<1xf32, 1>
          ascvec.get_buf mte2, %114, 0 {cannir.buffer.generated_sync}
          "ascvec.copy_gm2ub_plan"(%135, %alloc_2, %c0_i32, %c0_i32, %c0_i64, %129, %133, %134, %c0_i64, %134, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = -1 : i64, ub_capacity = 4 : i64}> : (memref<1xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
          ascvec.release_buf mte2, %114, 0 {cannir.buffer.generated_sync}
          %136:2 = scf.for %arg8 = %c0_i64 to %8 step %c1_i64 iter_args(%arg9 = %66, %arg10 = %arg7) -> (i64, i64)  : i64 {
            %147 = arith.cmpi slt, %arg8, %37 : i64
            %148:2 = scf.if %147 -> (i64, i64) {
              %149 = arith.addi %arg8, %c1_i64 : i64
              %150 = arith.cmpi slt, %149, %37 : i64
              %151 = scf.if %150 -> (i64) {
                %164 = arith.cmpi eq, %arg9, %c0_i64 : i64
                %165 = arith.select %164, %61, %62 : i64
                %166 = arith.select %164, %c2_i64, %c3_i64 : i64
                %167 = arith.addi %arg9, %c1_i64 overflow<nsw> : i64
                %168 = arith.remui %167, %c2_i64 : i64
                %169 = cannir.extract_dim %74[1] : !cannir.stride<(-1, -1, 128, 1)>
                %170 = arith.muli %149, %169 : i64
                %171 = arith.muli %170, %c2_i64 : i64
                %172 = arith.addi %77, %171 : i64
                %173 = arith.addi %172, %78 : i64
                %174 = arith.addi %173, %80 : i64
                %175 = ascscalar.bind_local_ptr %165 : memref<1xf16, 1>
                ascvec.get_buf mte2, %166, 0 {cannir.buffer.generated_sync}
                "ascvec.copy_gm2ub_plan"(%175, %alloc_0, %c0_i32, %c0_i32, %c0_i64, %174, %84, %85, %c256_i64, %c64_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xf16, 1>, memref<1xf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                ascvec.release_buf mte2, %166, 0 {cannir.buffer.generated_sync}
                %176 = arith.select %164, %87, %88 : i64
                %177 = arith.select %164, %c4_i64, %c5_i64 : i64
                %178 = cannir.extract_dim %100[1] : !cannir.stride<(-1, -1, 128, 1)>
                %179 = arith.muli %149, %178 : i64
                %180 = arith.muli %179, %c2_i64 : i64
                %181 = arith.addi %103, %180 : i64
                %182 = arith.addi %181, %78 : i64
                %183 = ascscalar.bind_local_ptr %176 : memref<1xf16, 1>
                ascvec.get_buf mte2, %177, 0 {cannir.buffer.generated_sync}
                "ascvec.copy_gm2ub_plan"(%183, %alloc_1, %c0_i32, %c0_i32, %c0_i64, %182, %108, %109, %c0_i64, %109, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xf16, 1>, memref<1xf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                ascvec.release_buf mte2, %177, 0 {cannir.buffer.generated_sync}
                %184 = arith.select %164, %111, %112 : i64
                %185 = arith.select %164, %c6_i64, %c7_i64 : i64
                %186 = cannir.extract_dim %124[1] : !cannir.stride<(-1, -1, 128, 1)>
                %187 = arith.muli %149, %186 : i64
                %188 = arith.muli %187, %c4_i64 : i64
                %189 = arith.addi %127, %188 : i64
                %190 = arith.addi %189, %128 : i64
                %191 = ascscalar.bind_local_ptr %184 : memref<1xf32, 1>
                ascvec.get_buf mte2, %185, 0 {cannir.buffer.generated_sync}
                "ascvec.copy_gm2ub_plan"(%191, %alloc_2, %c0_i32, %c0_i32, %c0_i64, %190, %133, %134, %c0_i64, %134, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = -1 : i64, ub_capacity = 4 : i64}> : (memref<1xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                ascvec.release_buf mte2, %185, 0 {cannir.buffer.generated_sync}
                scf.yield %168 : i64
              } else {
                scf.yield %arg9 : i64
              }
              %152 = arith.cmpi eq, %arg10, %c0_i64 : i64
              %153 = arith.select %152, %61, %62 : i64
              %154 = ascscalar.bind_local_ptr %153 : memref<128xf16, 1>
              %155 = arith.select %152, %c2_i64, %c3_i64 : i64
              %156 = arith.addi %arg10, %c1_i64 : i64
              %157 = arith.remui %156, %c2_i64 : i64
              %158 = arith.select %152, %87, %88 : i64
              %159 = ascscalar.bind_local_ptr %158 : memref<128xf16, 1>
              %160 = arith.select %152, %c4_i64, %c5_i64 : i64
              %161 = arith.select %152, %111, %112 : i64
              %162 = ascscalar.bind_local_ptr %161 : memref<128xf32, 1>
              %163 = arith.select %152, %c6_i64, %c7_i64 : i64
              ascvec.get_buf v, %155, 0 {cannir.buffer.generated_sync}
              ascvec.vf(%154, %alloc_12 : memref<128xf16, 1>, memref<128xf32, 1>) {
              ^bb0(%arg11: memref<128xf16, 1>, %arg12: memref<128xf32, 1>):
                %c0_13 = arith.constant 0 : index
                %c32_i32 = arith.constant 32 : i32
                %c0_14 = arith.constant 0 : index
                %c1 = arith.constant 1 : index
                scf.for %arg13 = %c0_14 to %c1 step %c1 {
                  %mask, %remaining_out = ascvec.update_mask %c32_i32 : !ascvec.mask
                  %164 = ascvec.loadalign_unpack %arg11[%c0_13] {mode = "b16_to_b32"} : memref<128xf16, 1>, index -> <f16>
                  %165 = ascvec.cast %164, %mask {rounding = 6 : i32} : <f16> -> <f32>
                  ascvec.storealign %arg12[%c0_13], %165, %mask : memref<128xf32, 1>, index, <f32>
                }
              } {cannir.vf_mode = "simd"}
              ascvec.release_buf v, %155, 0 {cannir.buffer.generated_sync}
              ascvec.get_buf v, %160, 0 {cannir.buffer.generated_sync}
              ascvec.get_buf v, %163, 0 {cannir.buffer.generated_sync}
              ascvec.get_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
              ascvec.vf(%159, %162, %alloc_12, %alloc_4 : memref<128xf16, 1>, memref<128xf32, 1>, memref<128xf32, 1>, memref<16384xf32, 1>) {
              ^bb0(%arg11: memref<128xf16, 1>, %arg12: memref<128xf32, 1>, %arg13: memref<128xf32, 1>, %arg14: memref<16384xf32, 1>):
                %c64 = arith.constant 64 : index
                %c1_i32_13 = arith.constant 1 : i32
                %c32_i32 = arith.constant 32 : i32
                %c0_i32_14 = arith.constant 0 : i32
                %c128_i32 = arith.constant 128 : i32
                %c0_15 = arith.constant 0 : index
                %c64_i32 = arith.constant 64 : i32
                %mask, %remaining_out = ascvec.update_mask %c64_i32 : !ascvec.mask
                %164 = ascvec.loadalign_unpack %arg11[%c0_15] {mode = "b16_to_b32"} : memref<128xf16, 1>, index -> <f16>
                %165 = ascvec.loadalign_unpack %arg11[%c64] {mode = "b16_to_b32"} : memref<128xf16, 1>, index -> <f16>
                %166 = ascvec.cast %164, %mask {rounding = 6 : i32} : <f16> -> <f32>
                %167 = ascvec.cast %165, %mask {rounding = 6 : i32} : <f16> -> <f32>
                %168 = ascvec.loadalign %arg12[%c0_15] : memref<128xf32, 1>, index -> <f32>
                %169 = ascvec.loadalign %arg12[%c64] : memref<128xf32, 1>, index -> <f32>
                scf.for %arg15 = %c0_i32_14 to %c32_i32 step %c1_i32_13  : i32 {
                  %170 = arith.muli %arg15, %c128_i32 overflow<nsw> : i32
                  %171 = arith.index_cast %arg15 : i32 to index
                  %172 = ascvec.loadalign_brc %arg13[%171] : memref<128xf32, 1>, index -> <f32>
                  %173 = arith.index_cast %170 : i32 to index
                  %174 = ascvec.loadalign %arg14[%173] : memref<16384xf32, 1>, index -> <f32>
                  %175 = arith.addi %170, %c64_i32 overflow<nsw> : i32
                  %176 = arith.index_cast %175 : i32 to index
                  %177 = ascvec.loadalign %arg14[%176] : memref<16384xf32, 1>, index -> <f32>
                  %178 = ascvec.mul %174, %168, %mask : <f32>
                  %179 = ascvec.mul %177, %169, %mask : <f32>
                  %180 = ascvec.copy %172, %mask : <f32>
                  %181 = ascvec.madd_two_address %180, %166, %178, %mask : <f32>
                  %182 = ascvec.madd_two_address %172, %167, %179, %mask : <f32>
                  ascvec.storealign %arg14[%173], %181, %mask : memref<16384xf32, 1>, index, <f32>
                  ascvec.storealign %arg14[%176], %182, %mask : memref<16384xf32, 1>, index, <f32>
                } {cannir.loop_unroll = 4 : i64}
              } {cannir.vf_mode = "simd"}
              ascvec.release_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
              ascvec.release_buf v, %163, 0 {cannir.buffer.generated_sync}
              ascvec.release_buf v, %160, 0 {cannir.buffer.generated_sync}
              scf.yield %151, %157 : i64, i64
            } else {
              scf.yield %arg9, %arg10 : i64, i64
            }
            scf.yield %148#0, %148#1 : i64, i64
          }
          %137 = arith.muli %34, %c16384_i64 : i64
          %138 = arith.addi %49, %137 : i64
          %139 = arith.muli %21, %c4096_i64 : i64
          %140 = arith.addi %138, %139 : i64
          %141 = arith.muli %140, %c32_i64 : i64
          %142 = ascscalar.memref_address %alloc : memref<1xf32>
          %143 = arith.divui %141, %c8_i64 : i64
          %144 = arith.addi %142, %143 : i64
          %145 = ascscalar.bind_gm_ptr %144 : memref<1xf32>
          %146 = ascscalar.bitcast_memref %alloc_4 byte_offset = %c0_i64 : memref<16384xf32, 1> -> memref<?xf32, 1>
          ascvec.get_buf mte3, %c0_i64, 0 {cannir.buffer.generated_sync}
          ascvec.copy_ub2gm %145[%c0], %146[%c0], %c1_i32, %c16384_i32, %c0_i64, %c16384_i32 l2_cache_ctl %c0_i32 : memref<1xf32>, memref<?xf32, 1>
          ascvec.release_buf mte3, %c0_i64, 0 {cannir.buffer.generated_sync}
          scf.yield %136#0, %136#1 : i64, i64
        } else {
          scf.yield %arg6, %arg7 : i64, i64
        }
        scf.yield %39#0, %39#1 : i64, i64
      }
    } else {
      %6 = arith.cmpi sle, %4, %c32_i64 : i64
      scf.if %6 {
        %7 = cannir.extract_layout %arg1 : !cannir.tensor<<gm, f16>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
        %8 = cannir.extract_shape %7 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, -1, 128)>
        %9 = cannir.extract_dim %8[1] : !cannir.shape<(-1, -1, -1, 128)>
        %10 = arith.muli %4, %c2_i64 : i64
        %11 = ascscalar.get_block_idx
        %12 = ascscalar.get_block_num
        %13:2 = scf.for %arg5 = %11 to %10 step %12 iter_args(%arg6 = %c0_i64, %arg7 = %c0_i64) -> (i64, i64)  : i64 {
          %14 = arith.remsi %arg5, %c2_i64 : i64
          %15 = arith.cmpi ne, %14, %c0_i64 : i64
          %16 = arith.cmpi slt, %14, %c0_i64 : i64
          %17 = arith.andi %15, %16 : i1
          %18 = arith.divsi %arg5, %c2_i64 : i64
          %19 = arith.extui %17 : i1 to i64
          %20 = arith.subi %18, %19 : i64
          %21 = arith.muli %20, %c2_i64 : i64
          %22 = arith.subi %arg5, %21 : i64
          %23 = arith.maxsi %3, %c1_i64 : i64
          %24 = arith.remsi %20, %23 : i64
          %25 = arith.cmpi slt, %24, %c0_i64 : i64
          %26 = arith.divsi %20, %23 : i64
          %27 = arith.extui %25 : i1 to i64
          %28 = arith.subi %26, %27 : i64
          %29 = arith.maxsi %2, %c1_i64 : i64
          %30 = arith.remsi %28, %29 : i64
          %31 = arith.cmpi slt, %30, %c0_i64 : i64
          %32 = arith.select %31, %29, %c0_i64 : i64
          %33 = arith.addi %30, %32 : i64
          %34 = arith.select %25, %23, %c0_i64 : i64
          %35 = arith.addi %24, %34 : i64
          %36 = arith.index_cast %33 : i64 to index
          %37 = memref.load %alloc_3[%36] : memref<1xi32>
          %38 = arith.extsi %37 : i32 to i64
          %39 = arith.cmpi sgt, %38, %c0_i64 : i64
          %40:2 = scf.if %39 -> (i64, i64) {
            %41 = arith.minsi %22, %c2_i64 : i64
            %42 = arith.muli %41, %c64_i64 : i64
            %43 = arith.subi %c128_i64, %42 : i64
            %44 = arith.minsi %43, %c64_i64 : i64
            %45 = arith.cmpi sle, %22, %c2_i64 : i64
            %46 = arith.select %45, %44, %c0_i64 : i64
            %47 = cannir.extract_physical_layout %arg0 : !cannir.tensor<<gm, f32>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, identity>
            %48 = cannir.extract_stride %47 : <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>> -> <(-1, 16384, 128, 1)>
            %49 = cannir.extract_dim %48[0] : !cannir.stride<(-1, 16384, 128, 1)>
            %50 = arith.muli %33, %49 : i64
            %51 = arith.muli %50, %c4_i64 : i64
            %52 = arith.muli %35, %c65536_i64 : i64
            %53 = arith.addi %51, %52 : i64
            %54 = arith.muli %22, %c32768_i64 : i64
            %55 = arith.addi %53, %54 : i64
            %56 = arith.muli %46, %c512_i64 : i64
            %57 = arith.cmpi eq, %56, %c0_i64 : i64
            %58 = arith.cmpi ne, %56, %c0_i64 : i64
            %59 = arith.extui %58 : i1 to i64
            %60 = arith.select %57, %c4_i64, %56 : i64
            ascvec.get_buf mte2, %c0_i64, 0 {cannir.buffer.generated_sync}
            "ascvec.copy_gm2ub_plan"(%alloc_4, %alloc, %c0_i32, %c0_i32, %c0_i64, %55, %59, %60, %c0_i64, %60, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 0 : i64, ub_capacity = 65536 : i64}> : (memref<16384xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
            ascvec.release_buf mte2, %c0_i64, 0 {cannir.buffer.generated_sync}
            %61 = arith.cmpi eq, %arg6, %c0_i64 : i64
            %62 = ascscalar.memref_address %alloc_6 : memref<128xf16, 1>
            %63 = ascscalar.memref_address %alloc_7 : memref<128xf16, 1>
            %64 = arith.select %61, %62, %63 : i64
            %65 = arith.select %61, %c2_i64, %c3_i64 : i64
            %66 = arith.addi %arg6, %c1_i64 : i64
            %67 = arith.remui %66, %c2_i64 : i64
            %68 = cannir.extract_dim %8[2] : !cannir.shape<(-1, -1, -1, 128)>
            %69 = arith.minsi %35, %68 : i64
            %70 = arith.subi %68, %69 : i64
            %71 = arith.minsi %70, %c1_i64 : i64
            %72 = arith.cmpi sle, %35, %68 : i64
            %73 = arith.select %72, %71, %c0_i64 : i64
            %74 = cannir.extract_physical_layout %arg1 : !cannir.tensor<<gm, f16>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
            %75 = cannir.extract_stride %74 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, 128, 1)>
            %76 = cannir.extract_dim %75[0] : !cannir.stride<(-1, -1, 128, 1)>
            %77 = arith.muli %33, %76 : i64
            %78 = arith.muli %77, %c2_i64 : i64
            %79 = arith.muli %35, %c256_i64 : i64
            %80 = arith.addi %78, %79 : i64
            %81 = arith.muli %22, %c128_i64 : i64
            %82 = arith.addi %80, %81 : i64
            %83 = arith.muli %46, %c2_i64 : i64
            %84 = arith.cmpi eq, %83, %c0_i64 : i64
            %85 = arith.select %84, %c0_i64, %73 : i64
            %86 = arith.select %84, %c2_i64, %83 : i64
            %87 = ascscalar.bind_local_ptr %64 : memref<1xf16, 1>
            ascvec.get_buf mte2, %65, 0 {cannir.buffer.generated_sync}
            "ascvec.copy_gm2ub_plan"(%87, %alloc_0, %c0_i32, %c0_i32, %c0_i64, %82, %85, %86, %c256_i64, %c128_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xf16, 1>, memref<1xf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
            ascvec.release_buf mte2, %65, 0 {cannir.buffer.generated_sync}
            %88 = ascscalar.memref_address %alloc_8 : memref<128xf16, 1>
            %89 = ascscalar.memref_address %alloc_9 : memref<128xf16, 1>
            %90 = arith.select %61, %88, %89 : i64
            %91 = arith.select %61, %c4_i64, %c5_i64 : i64
            %92 = cannir.extract_layout %arg2 : !cannir.tensor<<gm, f16>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
            %93 = cannir.extract_shape %92 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, -1, 128)>
            %94 = cannir.extract_dim %93[2] : !cannir.shape<(-1, -1, -1, 128)>
            %95 = arith.minsi %35, %94 : i64
            %96 = arith.subi %94, %95 : i64
            %97 = arith.minsi %96, %c1_i64 : i64
            %98 = arith.cmpi sle, %35, %94 : i64
            %99 = arith.select %98, %97, %c0_i64 : i64
            %100 = cannir.extract_physical_layout %arg2 : !cannir.tensor<<gm, f16>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
            %101 = cannir.extract_stride %100 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, 128, 1)>
            %102 = cannir.extract_dim %101[0] : !cannir.stride<(-1, -1, 128, 1)>
            %103 = arith.muli %33, %102 : i64
            %104 = arith.muli %103, %c2_i64 : i64
            %105 = arith.addi %104, %79 : i64
            %106 = arith.muli %99, %c256_i64 : i64
            %107 = arith.cmpi eq, %106, %c0_i64 : i64
            %108 = arith.cmpi ne, %106, %c0_i64 : i64
            %109 = arith.extui %108 : i1 to i64
            %110 = arith.select %107, %c2_i64, %106 : i64
            %111 = ascscalar.bind_local_ptr %90 : memref<1xf16, 1>
            ascvec.get_buf mte2, %91, 0 {cannir.buffer.generated_sync}
            "ascvec.copy_gm2ub_plan"(%111, %alloc_1, %c0_i32, %c0_i32, %c0_i64, %105, %109, %110, %c0_i64, %110, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xf16, 1>, memref<1xf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
            ascvec.release_buf mte2, %91, 0 {cannir.buffer.generated_sync}
            %112 = ascscalar.memref_address %alloc_10 : memref<128xf32, 1>
            %113 = ascscalar.memref_address %alloc_11 : memref<128xf32, 1>
            %114 = arith.select %61, %112, %113 : i64
            %115 = arith.select %61, %c6_i64, %c7_i64 : i64
            %116 = cannir.extract_layout %arg3 : !cannir.tensor<<gm, f32>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
            %117 = cannir.extract_shape %116 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, -1, 128)>
            %118 = cannir.extract_dim %117[2] : !cannir.shape<(-1, -1, -1, 128)>
            %119 = arith.minsi %35, %118 : i64
            %120 = arith.subi %118, %119 : i64
            %121 = arith.minsi %120, %c1_i64 : i64
            %122 = arith.cmpi sle, %35, %118 : i64
            %123 = arith.select %122, %121, %c0_i64 : i64
            %124 = cannir.extract_physical_layout %arg3 : !cannir.tensor<<gm, f32>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
            %125 = cannir.extract_stride %124 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, 128, 1)>
            %126 = cannir.extract_dim %125[0] : !cannir.stride<(-1, -1, 128, 1)>
            %127 = arith.muli %33, %126 : i64
            %128 = arith.muli %127, %c4_i64 : i64
            %129 = arith.muli %35, %c512_i64 : i64
            %130 = arith.addi %128, %129 : i64
            %131 = arith.muli %123, %c512_i64 : i64
            %132 = arith.cmpi eq, %131, %c0_i64 : i64
            %133 = arith.cmpi ne, %131, %c0_i64 : i64
            %134 = arith.extui %133 : i1 to i64
            %135 = arith.select %132, %c4_i64, %131 : i64
            %136 = ascscalar.bind_local_ptr %114 : memref<1xf32, 1>
            ascvec.get_buf mte2, %115, 0 {cannir.buffer.generated_sync}
            "ascvec.copy_gm2ub_plan"(%136, %alloc_2, %c0_i32, %c0_i32, %c0_i64, %130, %134, %135, %c0_i64, %135, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = -1 : i64, ub_capacity = 4 : i64}> : (memref<1xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
            ascvec.release_buf mte2, %115, 0 {cannir.buffer.generated_sync}
            %137:2 = scf.for %arg8 = %c0_i64 to %9 step %c1_i64 iter_args(%arg9 = %67, %arg10 = %arg7) -> (i64, i64)  : i64 {
              %148 = arith.cmpi slt, %arg8, %38 : i64
              %149:2 = scf.if %148 -> (i64, i64) {
                %150 = arith.addi %arg8, %c1_i64 : i64
                %151 = arith.cmpi slt, %150, %38 : i64
                %152 = scf.if %151 -> (i64) {
                  %165 = arith.cmpi eq, %arg9, %c0_i64 : i64
                  %166 = arith.select %165, %62, %63 : i64
                  %167 = arith.select %165, %c2_i64, %c3_i64 : i64
                  %168 = arith.addi %arg9, %c1_i64 overflow<nsw> : i64
                  %169 = arith.remui %168, %c2_i64 : i64
                  %170 = cannir.extract_dim %75[1] : !cannir.stride<(-1, -1, 128, 1)>
                  %171 = arith.muli %150, %170 : i64
                  %172 = arith.muli %171, %c2_i64 : i64
                  %173 = arith.addi %78, %172 : i64
                  %174 = arith.addi %173, %79 : i64
                  %175 = arith.addi %174, %81 : i64
                  %176 = ascscalar.bind_local_ptr %166 : memref<1xf16, 1>
                  ascvec.get_buf mte2, %167, 0 {cannir.buffer.generated_sync}
                  "ascvec.copy_gm2ub_plan"(%176, %alloc_0, %c0_i32, %c0_i32, %c0_i64, %175, %85, %86, %c256_i64, %c128_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xf16, 1>, memref<1xf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                  ascvec.release_buf mte2, %167, 0 {cannir.buffer.generated_sync}
                  %177 = arith.select %165, %88, %89 : i64
                  %178 = arith.select %165, %c4_i64, %c5_i64 : i64
                  %179 = cannir.extract_dim %101[1] : !cannir.stride<(-1, -1, 128, 1)>
                  %180 = arith.muli %150, %179 : i64
                  %181 = arith.muli %180, %c2_i64 : i64
                  %182 = arith.addi %104, %181 : i64
                  %183 = arith.addi %182, %79 : i64
                  %184 = ascscalar.bind_local_ptr %177 : memref<1xf16, 1>
                  ascvec.get_buf mte2, %178, 0 {cannir.buffer.generated_sync}
                  "ascvec.copy_gm2ub_plan"(%184, %alloc_1, %c0_i32, %c0_i32, %c0_i64, %183, %109, %110, %c0_i64, %110, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xf16, 1>, memref<1xf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                  ascvec.release_buf mte2, %178, 0 {cannir.buffer.generated_sync}
                  %185 = arith.select %165, %112, %113 : i64
                  %186 = arith.select %165, %c6_i64, %c7_i64 : i64
                  %187 = cannir.extract_dim %125[1] : !cannir.stride<(-1, -1, 128, 1)>
                  %188 = arith.muli %150, %187 : i64
                  %189 = arith.muli %188, %c4_i64 : i64
                  %190 = arith.addi %128, %189 : i64
                  %191 = arith.addi %190, %129 : i64
                  %192 = ascscalar.bind_local_ptr %185 : memref<1xf32, 1>
                  ascvec.get_buf mte2, %186, 0 {cannir.buffer.generated_sync}
                  "ascvec.copy_gm2ub_plan"(%192, %alloc_2, %c0_i32, %c0_i32, %c0_i64, %191, %134, %135, %c0_i64, %135, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = -1 : i64, ub_capacity = 4 : i64}> : (memref<1xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                  ascvec.release_buf mte2, %186, 0 {cannir.buffer.generated_sync}
                  scf.yield %169 : i64
                } else {
                  scf.yield %arg9 : i64
                }
                %153 = arith.cmpi eq, %arg10, %c0_i64 : i64
                %154 = arith.select %153, %62, %63 : i64
                %155 = ascscalar.bind_local_ptr %154 : memref<128xf16, 1>
                %156 = arith.select %153, %c2_i64, %c3_i64 : i64
                %157 = arith.addi %arg10, %c1_i64 : i64
                %158 = arith.remui %157, %c2_i64 : i64
                %159 = arith.select %153, %88, %89 : i64
                %160 = ascscalar.bind_local_ptr %159 : memref<128xf16, 1>
                %161 = arith.select %153, %c4_i64, %c5_i64 : i64
                %162 = arith.select %153, %112, %113 : i64
                %163 = ascscalar.bind_local_ptr %162 : memref<128xf32, 1>
                %164 = arith.select %153, %c6_i64, %c7_i64 : i64
                ascvec.get_buf v, %156, 0 {cannir.buffer.generated_sync}
                ascvec.vf(%155, %alloc_12 : memref<128xf16, 1>, memref<128xf32, 1>) {
                ^bb0(%arg11: memref<128xf16, 1>, %arg12: memref<128xf32, 1>):
                  %c0_13 = arith.constant 0 : index
                  %c64_i32 = arith.constant 64 : i32
                  %c0_14 = arith.constant 0 : index
                  %c1 = arith.constant 1 : index
                  scf.for %arg13 = %c0_14 to %c1 step %c1 {
                    %mask, %remaining_out = ascvec.update_mask %c64_i32 : !ascvec.mask
                    %165 = ascvec.loadalign_unpack %arg11[%c0_13] {mode = "b16_to_b32"} : memref<128xf16, 1>, index -> <f16>
                    %166 = ascvec.cast %165, %mask {rounding = 6 : i32} : <f16> -> <f32>
                    ascvec.storealign %arg12[%c0_13], %166, %mask : memref<128xf32, 1>, index, <f32>
                  }
                } {cannir.vf_mode = "simd"}
                ascvec.release_buf v, %156, 0 {cannir.buffer.generated_sync}
                ascvec.get_buf v, %161, 0 {cannir.buffer.generated_sync}
                ascvec.get_buf v, %164, 0 {cannir.buffer.generated_sync}
                ascvec.get_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
                ascvec.vf(%160, %163, %alloc_12, %alloc_4 : memref<128xf16, 1>, memref<128xf32, 1>, memref<128xf32, 1>, memref<16384xf32, 1>) {
                ^bb0(%arg11: memref<128xf16, 1>, %arg12: memref<128xf32, 1>, %arg13: memref<128xf32, 1>, %arg14: memref<16384xf32, 1>):
                  %c64 = arith.constant 64 : index
                  %c1_i32_13 = arith.constant 1 : i32
                  %c0_i32_14 = arith.constant 0 : i32
                  %c128_i32 = arith.constant 128 : i32
                  %c0_15 = arith.constant 0 : index
                  %c64_i32 = arith.constant 64 : i32
                  %mask, %remaining_out = ascvec.update_mask %c64_i32 : !ascvec.mask
                  %165 = ascvec.loadalign_unpack %arg11[%c0_15] {mode = "b16_to_b32"} : memref<128xf16, 1>, index -> <f16>
                  %166 = ascvec.loadalign_unpack %arg11[%c64] {mode = "b16_to_b32"} : memref<128xf16, 1>, index -> <f16>
                  %167 = ascvec.cast %165, %mask {rounding = 6 : i32} : <f16> -> <f32>
                  %168 = ascvec.cast %166, %mask {rounding = 6 : i32} : <f16> -> <f32>
                  %169 = ascvec.loadalign %arg12[%c0_15] : memref<128xf32, 1>, index -> <f32>
                  %170 = ascvec.loadalign %arg12[%c64] : memref<128xf32, 1>, index -> <f32>
                  scf.for %arg15 = %c0_i32_14 to %c64_i32 step %c1_i32_13  : i32 {
                    %171 = arith.muli %arg15, %c128_i32 overflow<nsw> : i32
                    %172 = arith.index_cast %arg15 : i32 to index
                    %173 = ascvec.loadalign_brc %arg13[%172] : memref<128xf32, 1>, index -> <f32>
                    %174 = arith.index_cast %171 : i32 to index
                    %175 = ascvec.loadalign %arg14[%174] : memref<16384xf32, 1>, index -> <f32>
                    %176 = arith.addi %171, %c64_i32 overflow<nsw> : i32
                    %177 = arith.index_cast %176 : i32 to index
                    %178 = ascvec.loadalign %arg14[%177] : memref<16384xf32, 1>, index -> <f32>
                    %179 = ascvec.mul %175, %169, %mask : <f32>
                    %180 = ascvec.mul %178, %170, %mask : <f32>
                    %181 = ascvec.copy %173, %mask : <f32>
                    %182 = ascvec.madd_two_address %181, %167, %179, %mask : <f32>
                    %183 = ascvec.madd_two_address %173, %168, %180, %mask : <f32>
                    ascvec.storealign %arg14[%174], %182, %mask : memref<16384xf32, 1>, index, <f32>
                    ascvec.storealign %arg14[%177], %183, %mask : memref<16384xf32, 1>, index, <f32>
                  } {cannir.loop_unroll = 4 : i64}
                } {cannir.vf_mode = "simd"}
                ascvec.release_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
                ascvec.release_buf v, %164, 0 {cannir.buffer.generated_sync}
                ascvec.release_buf v, %161, 0 {cannir.buffer.generated_sync}
                scf.yield %152, %158 : i64, i64
              } else {
                scf.yield %arg9, %arg10 : i64, i64
              }
              scf.yield %149#0, %149#1 : i64, i64
            }
            %138 = arith.muli %35, %c16384_i64 : i64
            %139 = arith.addi %50, %138 : i64
            %140 = arith.muli %22, %c8192_i64 : i64
            %141 = arith.addi %139, %140 : i64
            %142 = arith.muli %141, %c32_i64 : i64
            %143 = ascscalar.memref_address %alloc : memref<1xf32>
            %144 = arith.divui %142, %c8_i64 : i64
            %145 = arith.addi %143, %144 : i64
            %146 = ascscalar.bind_gm_ptr %145 : memref<1xf32>
            %147 = ascscalar.bitcast_memref %alloc_4 byte_offset = %c0_i64 : memref<16384xf32, 1> -> memref<?xf32, 1>
            ascvec.get_buf mte3, %c0_i64, 0 {cannir.buffer.generated_sync}
            ascvec.copy_ub2gm %146[%c0], %147[%c0], %c1_i32, %c32768_i32, %c0_i64, %c32768_i32 l2_cache_ctl %c0_i32 : memref<1xf32>, memref<?xf32, 1>
            ascvec.release_buf mte3, %c0_i64, 0 {cannir.buffer.generated_sync}
            scf.yield %137#0, %137#1 : i64, i64
          } else {
            scf.yield %arg6, %arg7 : i64, i64
          }
          scf.yield %40#0, %40#1 : i64, i64
        }
      } else {
        %7 = arith.remsi %4, %c2_i64 : i64
        %8 = arith.cmpi eq, %7, %c0_i64 : i64
        scf.if %8 {
          %9 = cannir.extract_layout %arg1 : !cannir.tensor<<gm, f16>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
          %10 = cannir.extract_shape %9 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, -1, 128)>
          %11 = cannir.extract_dim %10[1] : !cannir.shape<(-1, -1, -1, 128)>
          %12 = ascscalar.get_block_idx
          %13 = ascscalar.get_block_num
          %14 = arith.cmpi sge, %13, %4 : i64
          scf.if %14 {
            %15 = arith.cmpi slt, %12, %4 : i64
            scf.if %15 {
              %16 = arith.maxsi %3, %c1_i64 : i64
              %17 = arith.remsi %12, %16 : i64
              %18 = arith.cmpi slt, %17, %c0_i64 : i64
              %19 = arith.divsi %12, %16 : i64
              %20 = arith.extui %18 : i1 to i64
              %21 = arith.subi %19, %20 : i64
              %22 = arith.maxsi %2, %c1_i64 : i64
              %23 = arith.remsi %21, %22 : i64
              %24 = arith.cmpi slt, %23, %c0_i64 : i64
              %25 = arith.select %24, %22, %c0_i64 : i64
              %26 = arith.addi %23, %25 : i64
              %27 = arith.select %18, %16, %c0_i64 : i64
              %28 = arith.addi %17, %27 : i64
              %29 = arith.index_cast %26 : i64 to index
              %30 = memref.load %alloc_3[%29] : memref<1xi32>
              %31 = arith.extsi %30 : i32 to i64
              %32 = arith.cmpi sgt, %31, %c0_i64 : i64
              scf.if %32 {
                %33 = cannir.extract_physical_layout %arg0 : !cannir.tensor<<gm, f32>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, identity>
                %34 = cannir.extract_stride %33 : <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>> -> <(-1, 16384, 128, 1)>
                %35 = cannir.extract_dim %34[0] : !cannir.stride<(-1, 16384, 128, 1)>
                %36 = arith.muli %26, %35 : i64
                %37 = arith.muli %36, %c4_i64 : i64
                %38 = arith.muli %28, %c65536_i64 : i64
                %39 = arith.addi %37, %38 : i64
                ascvec.get_buf mte2, %c0_i64, 0 {cannir.buffer.generated_sync}
                "ascvec.copy_gm2ub_plan"(%alloc_4, %alloc, %c0_i32, %c0_i32, %c0_i64, %39, %c1_i64, %c65536_i64, %c0_i64, %c65536_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 0 : i64, ub_capacity = 65536 : i64}> : (memref<16384xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                ascvec.release_buf mte2, %c0_i64, 0 {cannir.buffer.generated_sync}
                %40 = cannir.extract_dim %10[2] : !cannir.shape<(-1, -1, -1, 128)>
                %41 = arith.minsi %28, %40 : i64
                %42 = arith.subi %40, %41 : i64
                %43 = arith.minsi %42, %c1_i64 : i64
                %44 = arith.cmpi sle, %28, %40 : i64
                %45 = arith.select %44, %43, %c0_i64 : i64
                %46 = cannir.extract_physical_layout %arg1 : !cannir.tensor<<gm, f16>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                %47 = cannir.extract_stride %46 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, 128, 1)>
                %48 = cannir.extract_dim %47[0] : !cannir.stride<(-1, -1, 128, 1)>
                %49 = arith.muli %26, %48 : i64
                %50 = arith.muli %49, %c2_i64 : i64
                %51 = arith.muli %28, %c256_i64 : i64
                %52 = arith.addi %50, %51 : i64
                %53 = arith.muli %45, %c256_i64 : i64
                %54 = arith.cmpi eq, %53, %c0_i64 : i64
                %55 = arith.cmpi ne, %53, %c0_i64 : i64
                %56 = arith.extui %55 : i1 to i64
                %57 = arith.select %54, %c2_i64, %53 : i64
                %58 = ascscalar.bitcast_memref %alloc_6 byte_offset = %c0_i64 : memref<128xf16, 1> -> memref<?xf16, 1>
                ascvec.get_buf mte2, %c2_i64, 0 {cannir.buffer.generated_sync}
                "ascvec.copy_gm2ub_plan"(%58, %alloc_0, %c0_i32, %c0_i32, %c0_i64, %52, %56, %57, %c0_i64, %57, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f16, ub_base = 131072 : i64, ub_capacity = 256 : i64}> : (memref<?xf16, 1>, memref<1xf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                ascvec.release_buf mte2, %c2_i64, 0 {cannir.buffer.generated_sync}
                %59 = cannir.extract_layout %arg2 : !cannir.tensor<<gm, f16>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                %60 = cannir.extract_shape %59 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, -1, 128)>
                %61 = cannir.extract_dim %60[2] : !cannir.shape<(-1, -1, -1, 128)>
                %62 = arith.minsi %28, %61 : i64
                %63 = arith.subi %61, %62 : i64
                %64 = arith.minsi %63, %c1_i64 : i64
                %65 = arith.cmpi sle, %28, %61 : i64
                %66 = arith.select %65, %64, %c0_i64 : i64
                %67 = cannir.extract_physical_layout %arg2 : !cannir.tensor<<gm, f16>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                %68 = cannir.extract_stride %67 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, 128, 1)>
                %69 = cannir.extract_dim %68[0] : !cannir.stride<(-1, -1, 128, 1)>
                %70 = arith.muli %26, %69 : i64
                %71 = arith.muli %70, %c2_i64 : i64
                %72 = arith.addi %71, %51 : i64
                %73 = arith.muli %66, %c256_i64 : i64
                %74 = arith.cmpi eq, %73, %c0_i64 : i64
                %75 = arith.cmpi ne, %73, %c0_i64 : i64
                %76 = arith.extui %75 : i1 to i64
                %77 = arith.select %74, %c2_i64, %73 : i64
                %78 = ascscalar.bitcast_memref %alloc_8 byte_offset = %c0_i64 : memref<128xf16, 1> -> memref<?xf16, 1>
                ascvec.get_buf mte2, %c4_i64, 0 {cannir.buffer.generated_sync}
                "ascvec.copy_gm2ub_plan"(%78, %alloc_1, %c0_i32, %c0_i32, %c0_i64, %72, %76, %77, %c0_i64, %77, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f16, ub_base = 131584 : i64, ub_capacity = 256 : i64}> : (memref<?xf16, 1>, memref<1xf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                ascvec.release_buf mte2, %c4_i64, 0 {cannir.buffer.generated_sync}
                %79 = cannir.extract_layout %arg3 : !cannir.tensor<<gm, f32>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                %80 = cannir.extract_shape %79 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, -1, 128)>
                %81 = cannir.extract_dim %80[2] : !cannir.shape<(-1, -1, -1, 128)>
                %82 = arith.minsi %28, %81 : i64
                %83 = arith.subi %81, %82 : i64
                %84 = arith.minsi %83, %c1_i64 : i64
                %85 = arith.cmpi sle, %28, %81 : i64
                %86 = arith.select %85, %84, %c0_i64 : i64
                %87 = cannir.extract_physical_layout %arg3 : !cannir.tensor<<gm, f32>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                %88 = cannir.extract_stride %87 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, 128, 1)>
                %89 = cannir.extract_dim %88[0] : !cannir.stride<(-1, -1, 128, 1)>
                %90 = arith.muli %26, %89 : i64
                %91 = arith.muli %90, %c4_i64 : i64
                %92 = arith.muli %28, %c512_i64 : i64
                %93 = arith.addi %91, %92 : i64
                %94 = arith.muli %86, %c512_i64 : i64
                %95 = arith.cmpi eq, %94, %c0_i64 : i64
                %96 = arith.cmpi ne, %94, %c0_i64 : i64
                %97 = arith.extui %96 : i1 to i64
                %98 = arith.select %95, %c4_i64, %94 : i64
                %99 = ascscalar.bitcast_memref %alloc_10 byte_offset = %c0_i64 : memref<128xf32, 1> -> memref<?xf32, 1>
                ascvec.get_buf mte2, %c6_i64, 0 {cannir.buffer.generated_sync}
                "ascvec.copy_gm2ub_plan"(%99, %alloc_2, %c0_i32, %c0_i32, %c0_i64, %93, %97, %98, %c0_i64, %98, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 132096 : i64, ub_capacity = 512 : i64}> : (memref<?xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                ascvec.release_buf mte2, %c6_i64, 0 {cannir.buffer.generated_sync}
                %100:2 = scf.for %arg5 = %c0_i64 to %11 step %c1_i64 iter_args(%arg6 = %c1_i64, %arg7 = %c0_i64) -> (i64, i64)  : i64 {
                  %109 = arith.cmpi slt, %arg5, %31 : i64
                  %110:2 = scf.if %109 -> (i64, i64) {
                    %111 = arith.addi %arg5, %c1_i64 : i64
                    %112 = arith.cmpi slt, %111, %31 : i64
                    %113 = scf.if %112 -> (i64) {
                      %132 = arith.cmpi eq, %arg6, %c0_i64 : i64
                      %133 = ascscalar.memref_address %alloc_6 : memref<128xf16, 1>
                      %134 = ascscalar.memref_address %alloc_7 : memref<128xf16, 1>
                      %135 = arith.select %132, %133, %134 : i64
                      %136 = arith.select %132, %c2_i64, %c3_i64 : i64
                      %137 = arith.addi %arg6, %c1_i64 overflow<nsw> : i64
                      %138 = arith.remui %137, %c2_i64 : i64
                      %139 = cannir.extract_dim %47[1] : !cannir.stride<(-1, -1, 128, 1)>
                      %140 = arith.muli %111, %139 : i64
                      %141 = arith.muli %140, %c2_i64 : i64
                      %142 = arith.addi %50, %141 : i64
                      %143 = arith.addi %142, %51 : i64
                      %144 = ascscalar.bind_local_ptr %135 : memref<1xf16, 1>
                      ascvec.get_buf mte2, %136, 0 {cannir.buffer.generated_sync}
                      "ascvec.copy_gm2ub_plan"(%144, %alloc_0, %c0_i32, %c0_i32, %c0_i64, %143, %56, %57, %c0_i64, %57, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xf16, 1>, memref<1xf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                      ascvec.release_buf mte2, %136, 0 {cannir.buffer.generated_sync}
                      %145 = ascscalar.memref_address %alloc_8 : memref<128xf16, 1>
                      %146 = ascscalar.memref_address %alloc_9 : memref<128xf16, 1>
                      %147 = arith.select %132, %145, %146 : i64
                      %148 = arith.select %132, %c4_i64, %c5_i64 : i64
                      %149 = cannir.extract_dim %68[1] : !cannir.stride<(-1, -1, 128, 1)>
                      %150 = arith.muli %111, %149 : i64
                      %151 = arith.muli %150, %c2_i64 : i64
                      %152 = arith.addi %71, %151 : i64
                      %153 = arith.addi %152, %51 : i64
                      %154 = ascscalar.bind_local_ptr %147 : memref<1xf16, 1>
                      ascvec.get_buf mte2, %148, 0 {cannir.buffer.generated_sync}
                      "ascvec.copy_gm2ub_plan"(%154, %alloc_1, %c0_i32, %c0_i32, %c0_i64, %153, %76, %77, %c0_i64, %77, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xf16, 1>, memref<1xf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                      ascvec.release_buf mte2, %148, 0 {cannir.buffer.generated_sync}
                      %155 = ascscalar.memref_address %alloc_10 : memref<128xf32, 1>
                      %156 = ascscalar.memref_address %alloc_11 : memref<128xf32, 1>
                      %157 = arith.select %132, %155, %156 : i64
                      %158 = arith.select %132, %c6_i64, %c7_i64 : i64
                      %159 = cannir.extract_dim %88[1] : !cannir.stride<(-1, -1, 128, 1)>
                      %160 = arith.muli %111, %159 : i64
                      %161 = arith.muli %160, %c4_i64 : i64
                      %162 = arith.addi %91, %161 : i64
                      %163 = arith.addi %162, %92 : i64
                      %164 = ascscalar.bind_local_ptr %157 : memref<1xf32, 1>
                      ascvec.get_buf mte2, %158, 0 {cannir.buffer.generated_sync}
                      "ascvec.copy_gm2ub_plan"(%164, %alloc_2, %c0_i32, %c0_i32, %c0_i64, %163, %97, %98, %c0_i64, %98, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = -1 : i64, ub_capacity = 4 : i64}> : (memref<1xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                      ascvec.release_buf mte2, %158, 0 {cannir.buffer.generated_sync}
                      scf.yield %138 : i64
                    } else {
                      scf.yield %arg6 : i64
                    }
                    %114 = arith.cmpi eq, %arg7, %c0_i64 : i64
                    %115 = ascscalar.memref_address %alloc_6 : memref<128xf16, 1>
                    %116 = ascscalar.memref_address %alloc_7 : memref<128xf16, 1>
                    %117 = arith.select %114, %115, %116 : i64
                    %118 = ascscalar.bind_local_ptr %117 : memref<128xf16, 1>
                    %119 = arith.select %114, %c2_i64, %c3_i64 : i64
                    %120 = arith.addi %arg7, %c1_i64 overflow<nsw> : i64
                    %121 = arith.remui %120, %c2_i64 : i64
                    %122 = ascscalar.memref_address %alloc_8 : memref<128xf16, 1>
                    %123 = ascscalar.memref_address %alloc_9 : memref<128xf16, 1>
                    %124 = arith.select %114, %122, %123 : i64
                    %125 = ascscalar.bind_local_ptr %124 : memref<128xf16, 1>
                    %126 = arith.select %114, %c4_i64, %c5_i64 : i64
                    %127 = ascscalar.memref_address %alloc_10 : memref<128xf32, 1>
                    %128 = ascscalar.memref_address %alloc_11 : memref<128xf32, 1>
                    %129 = arith.select %114, %127, %128 : i64
                    %130 = ascscalar.bind_local_ptr %129 : memref<128xf32, 1>
                    %131 = arith.select %114, %c6_i64, %c7_i64 : i64
                    ascvec.get_buf v, %119, 0 {cannir.buffer.generated_sync}
                    ascvec.vf(%118, %alloc_12 : memref<128xf16, 1>, memref<128xf32, 1>) {
                    ^bb0(%arg8: memref<128xf16, 1>, %arg9: memref<128xf32, 1>):
                      %c1_i32_13 = arith.constant 1 : i32
                      %c2_i32 = arith.constant 2 : i32
                      %c0_i32_14 = arith.constant 0 : i32
                      %c128_i32 = arith.constant 128 : i32
                      %c64_i32 = arith.constant 64 : i32
                      scf.for %arg10 = %c0_i32_14 to %c2_i32 step %c1_i32_13  : i32 {
                        %132 = arith.muli %arg10, %c64_i32 overflow<nsw> : i32
                        %133 = arith.subi %c128_i32, %132 overflow<nsw> : i32
                        %mask, %remaining_out = ascvec.update_mask %133 : !ascvec.mask
                        %134 = arith.index_cast %132 : i32 to index
                        %135 = ascvec.loadalign_unpack %arg8[%134] {mode = "b16_to_b32"} : memref<128xf16, 1>, index -> <f16>
                        %136 = ascvec.cast %135, %mask {rounding = 6 : i32} : <f16> -> <f32>
                        ascvec.storealign %arg9[%134], %136, %mask : memref<128xf32, 1>, index, <f32>
                      }
                    } {cannir.vf_mode = "simd"}
                    ascvec.release_buf v, %119, 0 {cannir.buffer.generated_sync}
                    ascvec.get_buf v, %126, 0 {cannir.buffer.generated_sync}
                    ascvec.get_buf v, %131, 0 {cannir.buffer.generated_sync}
                    ascvec.get_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
                    ascvec.vf(%125, %130, %alloc_12, %alloc_4 : memref<128xf16, 1>, memref<128xf32, 1>, memref<128xf32, 1>, memref<16384xf32, 1>) {
                    ^bb0(%arg8: memref<128xf16, 1>, %arg9: memref<128xf32, 1>, %arg10: memref<128xf32, 1>, %arg11: memref<16384xf32, 1>):
                      %c64 = arith.constant 64 : index
                      %c1_i32_13 = arith.constant 1 : i32
                      %c0_i32_14 = arith.constant 0 : i32
                      %c128_i32 = arith.constant 128 : i32
                      %c0_15 = arith.constant 0 : index
                      %c64_i32 = arith.constant 64 : i32
                      %mask, %remaining_out = ascvec.update_mask %c64_i32 : !ascvec.mask
                      %132 = ascvec.loadalign_unpack %arg8[%c0_15] {mode = "b16_to_b32"} : memref<128xf16, 1>, index -> <f16>
                      %133 = ascvec.loadalign_unpack %arg8[%c64] {mode = "b16_to_b32"} : memref<128xf16, 1>, index -> <f16>
                      %134 = ascvec.cast %132, %mask {rounding = 6 : i32} : <f16> -> <f32>
                      %135 = ascvec.cast %133, %mask {rounding = 6 : i32} : <f16> -> <f32>
                      %136 = ascvec.loadalign %arg9[%c0_15] : memref<128xf32, 1>, index -> <f32>
                      %137 = ascvec.loadalign %arg9[%c64] : memref<128xf32, 1>, index -> <f32>
                      scf.for %arg12 = %c0_i32_14 to %c128_i32 step %c1_i32_13  : i32 {
                        %138 = arith.muli %arg12, %c128_i32 overflow<nsw> : i32
                        %139 = arith.index_cast %arg12 : i32 to index
                        %140 = ascvec.loadalign_brc %arg10[%139] : memref<128xf32, 1>, index -> <f32>
                        %141 = arith.index_cast %138 : i32 to index
                        %142 = ascvec.loadalign %arg11[%141] : memref<16384xf32, 1>, index -> <f32>
                        %143 = arith.addi %138, %c64_i32 overflow<nsw> : i32
                        %144 = arith.index_cast %143 : i32 to index
                        %145 = ascvec.loadalign %arg11[%144] : memref<16384xf32, 1>, index -> <f32>
                        %146 = ascvec.mul %142, %136, %mask : <f32>
                        %147 = ascvec.mul %145, %137, %mask : <f32>
                        %148 = ascvec.copy %140, %mask : <f32>
                        %149 = ascvec.madd_two_address %148, %134, %146, %mask : <f32>
                        %150 = ascvec.madd_two_address %140, %135, %147, %mask : <f32>
                        ascvec.storealign %arg11[%141], %149, %mask : memref<16384xf32, 1>, index, <f32>
                        ascvec.storealign %arg11[%144], %150, %mask : memref<16384xf32, 1>, index, <f32>
                      } {cannir.loop_unroll = 4 : i64}
                    } {cannir.vf_mode = "simd"}
                    ascvec.release_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
                    ascvec.release_buf v, %131, 0 {cannir.buffer.generated_sync}
                    ascvec.release_buf v, %126, 0 {cannir.buffer.generated_sync}
                    scf.yield %113, %121 : i64, i64
                  } else {
                    scf.yield %arg6, %arg7 : i64, i64
                  }
                  scf.yield %110#0, %110#1 : i64, i64
                }
                %101 = arith.muli %28, %c16384_i64 : i64
                %102 = arith.addi %36, %101 : i64
                %103 = arith.muli %102, %c32_i64 : i64
                %104 = ascscalar.memref_address %alloc : memref<1xf32>
                %105 = arith.divui %103, %c8_i64 : i64
                %106 = arith.addi %104, %105 : i64
                %107 = ascscalar.bind_gm_ptr %106 : memref<1xf32>
                %108 = ascscalar.bitcast_memref %alloc_4 byte_offset = %c0_i64 : memref<16384xf32, 1> -> memref<?xf32, 1>
                ascvec.get_buf mte3, %c0_i64, 0 {cannir.buffer.generated_sync}
                ascvec.copy_ub2gm %107[%c0], %108[%c0], %c1_i32, %c65536_i32, %c0_i64, %c65536_i32 l2_cache_ctl %c0_i32 : memref<1xf32>, memref<?xf32, 1>
                ascvec.release_buf mte3, %c0_i64, 0 {cannir.buffer.generated_sync}
              }
            }
          } else {
            %15 = arith.cmpi ne, %7, %c0_i64 : i64
            %16 = arith.cmpi slt, %7, %c0_i64 : i64
            %17 = arith.andi %15, %16 : i1
            %18 = arith.divsi %4, %c2_i64 : i64
            %19 = arith.extui %17 : i1 to i64
            %20 = arith.subi %18, %19 : i64
            %21 = arith.cmpi sle, %13, %20 : i64
            scf.if %21 {
              %22:2 = scf.for %arg5 = %12 to %20 step %13 iter_args(%arg6 = %c0_i64, %arg7 = %c0_i64) -> (i64, i64)  : i64 {
                %23 = arith.muli %arg5, %c2_i64 : i64
                %24 = arith.addi %23, %c1_i64 : i64
                %25 = arith.maxsi %3, %c1_i64 : i64
                %26 = arith.remsi %23, %25 : i64
                %27 = arith.cmpi slt, %26, %c0_i64 : i64
                %28 = arith.divsi %23, %25 : i64
                %29 = arith.extui %27 : i1 to i64
                %30 = arith.subi %28, %29 : i64
                %31 = arith.maxsi %2, %c1_i64 : i64
                %32 = arith.remsi %30, %31 : i64
                %33 = arith.cmpi slt, %32, %c0_i64 : i64
                %34 = arith.select %33, %31, %c0_i64 : i64
                %35 = arith.addi %32, %34 : i64
                %36 = arith.select %27, %25, %c0_i64 : i64
                %37 = arith.addi %26, %36 : i64
                %38 = arith.remsi %24, %25 : i64
                %39 = arith.cmpi slt, %38, %c0_i64 : i64
                %40 = arith.divsi %24, %25 : i64
                %41 = arith.extui %39 : i1 to i64
                %42 = arith.subi %40, %41 : i64
                %43 = arith.remsi %42, %31 : i64
                %44 = arith.cmpi slt, %43, %c0_i64 : i64
                %45 = arith.select %44, %31, %c0_i64 : i64
                %46 = arith.addi %43, %45 : i64
                %47 = arith.select %39, %25, %c0_i64 : i64
                %48 = arith.addi %38, %47 : i64
                %49 = arith.index_cast %35 : i64 to index
                %50 = memref.load %alloc_3[%49] : memref<1xi32>
                %51 = arith.extsi %50 : i32 to i64
                %52 = arith.index_cast %46 : i64 to index
                %53 = memref.load %alloc_3[%52] : memref<1xi32>
                %54 = arith.extsi %53 : i32 to i64
                %55 = arith.cmpi sgt, %51, %c0_i64 : i64
                %56:4 = scf.if %55 -> (i64, i64, i64, i64) {
                  %59 = cannir.extract_physical_layout %arg0 : !cannir.tensor<<gm, f32>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, identity>
                  %60 = cannir.extract_stride %59 : <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>> -> <(-1, 16384, 128, 1)>
                  %61 = cannir.extract_dim %60[0] : !cannir.stride<(-1, 16384, 128, 1)>
                  %62 = arith.muli %35, %61 : i64
                  %63 = arith.muli %62, %c4_i64 : i64
                  %64 = arith.muli %37, %c65536_i64 : i64
                  %65 = arith.addi %63, %64 : i64
                  ascvec.get_buf mte2, %c0_i64, 0 {cannir.buffer.generated_sync}
                  "ascvec.copy_gm2ub_plan"(%alloc_4, %alloc, %c0_i32, %c0_i32, %c0_i64, %65, %c1_i64, %c65536_i64, %c0_i64, %c65536_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 0 : i64, ub_capacity = 65536 : i64}> : (memref<16384xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                  ascvec.release_buf mte2, %c0_i64, 0 {cannir.buffer.generated_sync}
                  %66 = arith.cmpi eq, %arg6, %c0_i64 : i64
                  %67 = ascscalar.memref_address %alloc_6 : memref<128xf16, 1>
                  %68 = ascscalar.memref_address %alloc_7 : memref<128xf16, 1>
                  %69 = arith.select %66, %67, %68 : i64
                  %70 = arith.select %66, %c2_i64, %c3_i64 : i64
                  %71 = arith.addi %arg6, %c1_i64 : i64
                  %72 = arith.remui %71, %c2_i64 : i64
                  %73 = cannir.extract_dim %10[2] : !cannir.shape<(-1, -1, -1, 128)>
                  %74 = arith.minsi %37, %73 : i64
                  %75 = arith.subi %73, %74 : i64
                  %76 = arith.minsi %75, %c1_i64 : i64
                  %77 = arith.cmpi sle, %37, %73 : i64
                  %78 = arith.select %77, %76, %c0_i64 : i64
                  %79 = cannir.extract_physical_layout %arg1 : !cannir.tensor<<gm, f16>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                  %80 = cannir.extract_stride %79 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, 128, 1)>
                  %81 = cannir.extract_dim %80[0] : !cannir.stride<(-1, -1, 128, 1)>
                  %82 = arith.muli %35, %81 : i64
                  %83 = arith.muli %82, %c2_i64 : i64
                  %84 = arith.muli %37, %c256_i64 : i64
                  %85 = arith.addi %83, %84 : i64
                  %86 = arith.muli %78, %c256_i64 : i64
                  %87 = arith.cmpi eq, %86, %c0_i64 : i64
                  %88 = arith.cmpi ne, %86, %c0_i64 : i64
                  %89 = arith.extui %88 : i1 to i64
                  %90 = arith.select %87, %c2_i64, %86 : i64
                  %91 = ascscalar.bind_local_ptr %69 : memref<1xf16, 1>
                  ascvec.get_buf mte2, %70, 0 {cannir.buffer.generated_sync}
                  "ascvec.copy_gm2ub_plan"(%91, %alloc_0, %c0_i32, %c0_i32, %c0_i64, %85, %89, %90, %c0_i64, %90, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xf16, 1>, memref<1xf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                  ascvec.release_buf mte2, %70, 0 {cannir.buffer.generated_sync}
                  %92 = ascscalar.memref_address %alloc_8 : memref<128xf16, 1>
                  %93 = ascscalar.memref_address %alloc_9 : memref<128xf16, 1>
                  %94 = arith.select %66, %92, %93 : i64
                  %95 = arith.select %66, %c4_i64, %c5_i64 : i64
                  %96 = cannir.extract_layout %arg2 : !cannir.tensor<<gm, f16>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                  %97 = cannir.extract_shape %96 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, -1, 128)>
                  %98 = cannir.extract_dim %97[2] : !cannir.shape<(-1, -1, -1, 128)>
                  %99 = arith.minsi %37, %98 : i64
                  %100 = arith.subi %98, %99 : i64
                  %101 = arith.minsi %100, %c1_i64 : i64
                  %102 = arith.cmpi sle, %37, %98 : i64
                  %103 = arith.select %102, %101, %c0_i64 : i64
                  %104 = cannir.extract_physical_layout %arg2 : !cannir.tensor<<gm, f16>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                  %105 = cannir.extract_stride %104 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, 128, 1)>
                  %106 = cannir.extract_dim %105[0] : !cannir.stride<(-1, -1, 128, 1)>
                  %107 = arith.muli %35, %106 : i64
                  %108 = arith.muli %107, %c2_i64 : i64
                  %109 = arith.addi %108, %84 : i64
                  %110 = arith.muli %103, %c256_i64 : i64
                  %111 = arith.cmpi eq, %110, %c0_i64 : i64
                  %112 = arith.cmpi ne, %110, %c0_i64 : i64
                  %113 = arith.extui %112 : i1 to i64
                  %114 = arith.select %111, %c2_i64, %110 : i64
                  %115 = ascscalar.bind_local_ptr %94 : memref<1xf16, 1>
                  ascvec.get_buf mte2, %95, 0 {cannir.buffer.generated_sync}
                  "ascvec.copy_gm2ub_plan"(%115, %alloc_1, %c0_i32, %c0_i32, %c0_i64, %109, %113, %114, %c0_i64, %114, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xf16, 1>, memref<1xf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                  ascvec.release_buf mte2, %95, 0 {cannir.buffer.generated_sync}
                  %116 = ascscalar.memref_address %alloc_10 : memref<128xf32, 1>
                  %117 = ascscalar.memref_address %alloc_11 : memref<128xf32, 1>
                  %118 = arith.select %66, %116, %117 : i64
                  %119 = arith.select %66, %c6_i64, %c7_i64 : i64
                  %120 = cannir.extract_layout %arg3 : !cannir.tensor<<gm, f32>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                  %121 = cannir.extract_shape %120 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, -1, 128)>
                  %122 = cannir.extract_dim %121[2] : !cannir.shape<(-1, -1, -1, 128)>
                  %123 = arith.minsi %37, %122 : i64
                  %124 = arith.subi %122, %123 : i64
                  %125 = arith.minsi %124, %c1_i64 : i64
                  %126 = arith.cmpi sle, %37, %122 : i64
                  %127 = arith.select %126, %125, %c0_i64 : i64
                  %128 = cannir.extract_physical_layout %arg3 : !cannir.tensor<<gm, f32>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                  %129 = cannir.extract_stride %128 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, 128, 1)>
                  %130 = cannir.extract_dim %129[0] : !cannir.stride<(-1, -1, 128, 1)>
                  %131 = arith.muli %35, %130 : i64
                  %132 = arith.muli %131, %c4_i64 : i64
                  %133 = arith.muli %37, %c512_i64 : i64
                  %134 = arith.addi %132, %133 : i64
                  %135 = arith.muli %127, %c512_i64 : i64
                  %136 = arith.cmpi eq, %135, %c0_i64 : i64
                  %137 = arith.cmpi ne, %135, %c0_i64 : i64
                  %138 = arith.extui %137 : i1 to i64
                  %139 = arith.select %136, %c4_i64, %135 : i64
                  %140 = ascscalar.bind_local_ptr %118 : memref<1xf32, 1>
                  ascvec.get_buf mte2, %119, 0 {cannir.buffer.generated_sync}
                  "ascvec.copy_gm2ub_plan"(%140, %alloc_2, %c0_i32, %c0_i32, %c0_i64, %134, %138, %139, %c0_i64, %139, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = -1 : i64, ub_capacity = 4 : i64}> : (memref<1xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                  ascvec.release_buf mte2, %119, 0 {cannir.buffer.generated_sync}
                  %141:2 = scf.for %arg8 = %c0_i64 to %11 step %c1_i64 iter_args(%arg9 = %72, %arg10 = %arg7) -> (i64, i64)  : i64 {
                    %150 = arith.cmpi slt, %arg8, %51 : i64
                    %151:2 = scf.if %150 -> (i64, i64) {
                      %152 = arith.addi %arg8, %c1_i64 : i64
                      %153 = arith.cmpi slt, %152, %51 : i64
                      %154 = scf.if %153 -> (i64) {
                        %167 = arith.cmpi eq, %arg9, %c0_i64 : i64
                        %168 = arith.select %167, %67, %68 : i64
                        %169 = arith.select %167, %c2_i64, %c3_i64 : i64
                        %170 = arith.addi %arg9, %c1_i64 overflow<nsw> : i64
                        %171 = arith.remui %170, %c2_i64 : i64
                        %172 = cannir.extract_dim %80[1] : !cannir.stride<(-1, -1, 128, 1)>
                        %173 = arith.muli %152, %172 : i64
                        %174 = arith.muli %173, %c2_i64 : i64
                        %175 = arith.addi %83, %174 : i64
                        %176 = arith.addi %175, %84 : i64
                        %177 = ascscalar.bind_local_ptr %168 : memref<1xf16, 1>
                        ascvec.get_buf mte2, %169, 0 {cannir.buffer.generated_sync}
                        "ascvec.copy_gm2ub_plan"(%177, %alloc_0, %c0_i32, %c0_i32, %c0_i64, %176, %89, %90, %c0_i64, %90, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xf16, 1>, memref<1xf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                        ascvec.release_buf mte2, %169, 0 {cannir.buffer.generated_sync}
                        %178 = arith.select %167, %92, %93 : i64
                        %179 = arith.select %167, %c4_i64, %c5_i64 : i64
                        %180 = cannir.extract_dim %105[1] : !cannir.stride<(-1, -1, 128, 1)>
                        %181 = arith.muli %152, %180 : i64
                        %182 = arith.muli %181, %c2_i64 : i64
                        %183 = arith.addi %108, %182 : i64
                        %184 = arith.addi %183, %84 : i64
                        %185 = ascscalar.bind_local_ptr %178 : memref<1xf16, 1>
                        ascvec.get_buf mte2, %179, 0 {cannir.buffer.generated_sync}
                        "ascvec.copy_gm2ub_plan"(%185, %alloc_1, %c0_i32, %c0_i32, %c0_i64, %184, %113, %114, %c0_i64, %114, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xf16, 1>, memref<1xf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                        ascvec.release_buf mte2, %179, 0 {cannir.buffer.generated_sync}
                        %186 = arith.select %167, %116, %117 : i64
                        %187 = arith.select %167, %c6_i64, %c7_i64 : i64
                        %188 = cannir.extract_dim %129[1] : !cannir.stride<(-1, -1, 128, 1)>
                        %189 = arith.muli %152, %188 : i64
                        %190 = arith.muli %189, %c4_i64 : i64
                        %191 = arith.addi %132, %190 : i64
                        %192 = arith.addi %191, %133 : i64
                        %193 = ascscalar.bind_local_ptr %186 : memref<1xf32, 1>
                        ascvec.get_buf mte2, %187, 0 {cannir.buffer.generated_sync}
                        "ascvec.copy_gm2ub_plan"(%193, %alloc_2, %c0_i32, %c0_i32, %c0_i64, %192, %138, %139, %c0_i64, %139, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = -1 : i64, ub_capacity = 4 : i64}> : (memref<1xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                        ascvec.release_buf mte2, %187, 0 {cannir.buffer.generated_sync}
                        scf.yield %171 : i64
                      } else {
                        %167 = arith.cmpi sgt, %54, %c0_i64 : i64
                        %168 = scf.if %167 -> (i64) {
                          %169 = arith.muli %46, %61 : i64
                          %170 = arith.muli %169, %c4_i64 : i64
                          %171 = arith.muli %48, %c65536_i64 : i64
                          %172 = arith.addi %170, %171 : i64
                          ascvec.get_buf mte2, %c1_i64, 0 {cannir.buffer.generated_sync}
                          "ascvec.copy_gm2ub_plan"(%alloc_5, %alloc, %c0_i32, %c0_i32, %c0_i64, %172, %c1_i64, %c65536_i64, %c0_i64, %c65536_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 65536 : i64, ub_capacity = 65536 : i64}> : (memref<16384xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                          ascvec.release_buf mte2, %c1_i64, 0 {cannir.buffer.generated_sync}
                          %173 = arith.cmpi eq, %arg9, %c0_i64 : i64
                          %174 = arith.select %173, %67, %68 : i64
                          %175 = arith.select %173, %c2_i64, %c3_i64 : i64
                          %176 = arith.addi %arg9, %c1_i64 overflow<nsw> : i64
                          %177 = arith.remui %176, %c2_i64 : i64
                          %178 = arith.minsi %48, %73 : i64
                          %179 = arith.subi %73, %178 : i64
                          %180 = arith.minsi %179, %c1_i64 : i64
                          %181 = arith.cmpi sle, %48, %73 : i64
                          %182 = arith.select %181, %180, %c0_i64 : i64
                          %183 = arith.muli %46, %81 : i64
                          %184 = arith.muli %183, %c2_i64 : i64
                          %185 = arith.muli %48, %c256_i64 : i64
                          %186 = arith.addi %184, %185 : i64
                          %187 = arith.muli %182, %c256_i64 : i64
                          %188 = arith.cmpi eq, %187, %c0_i64 : i64
                          %189 = arith.cmpi ne, %187, %c0_i64 : i64
                          %190 = arith.extui %189 : i1 to i64
                          %191 = arith.select %188, %c2_i64, %187 : i64
                          %192 = ascscalar.bind_local_ptr %174 : memref<1xf16, 1>
                          ascvec.get_buf mte2, %175, 0 {cannir.buffer.generated_sync}
                          "ascvec.copy_gm2ub_plan"(%192, %alloc_0, %c0_i32, %c0_i32, %c0_i64, %186, %190, %191, %c0_i64, %191, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xf16, 1>, memref<1xf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                          ascvec.release_buf mte2, %175, 0 {cannir.buffer.generated_sync}
                          %193 = arith.select %173, %92, %93 : i64
                          %194 = arith.select %173, %c4_i64, %c5_i64 : i64
                          %195 = arith.minsi %48, %98 : i64
                          %196 = arith.subi %98, %195 : i64
                          %197 = arith.minsi %196, %c1_i64 : i64
                          %198 = arith.cmpi sle, %48, %98 : i64
                          %199 = arith.select %198, %197, %c0_i64 : i64
                          %200 = arith.muli %46, %106 : i64
                          %201 = arith.muli %200, %c2_i64 : i64
                          %202 = arith.addi %201, %185 : i64
                          %203 = arith.muli %199, %c256_i64 : i64
                          %204 = arith.cmpi eq, %203, %c0_i64 : i64
                          %205 = arith.cmpi ne, %203, %c0_i64 : i64
                          %206 = arith.extui %205 : i1 to i64
                          %207 = arith.select %204, %c2_i64, %203 : i64
                          %208 = ascscalar.bind_local_ptr %193 : memref<1xf16, 1>
                          ascvec.get_buf mte2, %194, 0 {cannir.buffer.generated_sync}
                          "ascvec.copy_gm2ub_plan"(%208, %alloc_1, %c0_i32, %c0_i32, %c0_i64, %202, %206, %207, %c0_i64, %207, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xf16, 1>, memref<1xf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                          ascvec.release_buf mte2, %194, 0 {cannir.buffer.generated_sync}
                          %209 = arith.select %173, %116, %117 : i64
                          %210 = arith.select %173, %c6_i64, %c7_i64 : i64
                          %211 = arith.minsi %48, %122 : i64
                          %212 = arith.subi %122, %211 : i64
                          %213 = arith.minsi %212, %c1_i64 : i64
                          %214 = arith.cmpi sle, %48, %122 : i64
                          %215 = arith.select %214, %213, %c0_i64 : i64
                          %216 = arith.muli %46, %130 : i64
                          %217 = arith.muli %216, %c4_i64 : i64
                          %218 = arith.muli %48, %c512_i64 : i64
                          %219 = arith.addi %217, %218 : i64
                          %220 = arith.muli %215, %c512_i64 : i64
                          %221 = arith.cmpi eq, %220, %c0_i64 : i64
                          %222 = arith.cmpi ne, %220, %c0_i64 : i64
                          %223 = arith.extui %222 : i1 to i64
                          %224 = arith.select %221, %c4_i64, %220 : i64
                          %225 = ascscalar.bind_local_ptr %209 : memref<1xf32, 1>
                          ascvec.get_buf mte2, %210, 0 {cannir.buffer.generated_sync}
                          "ascvec.copy_gm2ub_plan"(%225, %alloc_2, %c0_i32, %c0_i32, %c0_i64, %219, %223, %224, %c0_i64, %224, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = -1 : i64, ub_capacity = 4 : i64}> : (memref<1xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                          ascvec.release_buf mte2, %210, 0 {cannir.buffer.generated_sync}
                          scf.yield %177 : i64
                        } else {
                          scf.yield %arg9 : i64
                        }
                        scf.yield %168 : i64
                      }
                      %155 = arith.cmpi eq, %arg10, %c0_i64 : i64
                      %156 = arith.select %155, %67, %68 : i64
                      %157 = ascscalar.bind_local_ptr %156 : memref<128xf16, 1>
                      %158 = arith.select %155, %c2_i64, %c3_i64 : i64
                      %159 = arith.addi %arg10, %c1_i64 : i64
                      %160 = arith.remui %159, %c2_i64 : i64
                      %161 = arith.select %155, %92, %93 : i64
                      %162 = ascscalar.bind_local_ptr %161 : memref<128xf16, 1>
                      %163 = arith.select %155, %c4_i64, %c5_i64 : i64
                      %164 = arith.select %155, %116, %117 : i64
                      %165 = ascscalar.bind_local_ptr %164 : memref<128xf32, 1>
                      %166 = arith.select %155, %c6_i64, %c7_i64 : i64
                      ascvec.get_buf v, %158, 0 {cannir.buffer.generated_sync}
                      ascvec.vf(%157, %alloc_12 : memref<128xf16, 1>, memref<128xf32, 1>) {
                      ^bb0(%arg11: memref<128xf16, 1>, %arg12: memref<128xf32, 1>):
                        %c1_i32_13 = arith.constant 1 : i32
                        %c2_i32 = arith.constant 2 : i32
                        %c0_i32_14 = arith.constant 0 : i32
                        %c128_i32 = arith.constant 128 : i32
                        %c64_i32 = arith.constant 64 : i32
                        scf.for %arg13 = %c0_i32_14 to %c2_i32 step %c1_i32_13  : i32 {
                          %167 = arith.muli %arg13, %c64_i32 overflow<nsw> : i32
                          %168 = arith.subi %c128_i32, %167 overflow<nsw> : i32
                          %mask, %remaining_out = ascvec.update_mask %168 : !ascvec.mask
                          %169 = arith.index_cast %167 : i32 to index
                          %170 = ascvec.loadalign_unpack %arg11[%169] {mode = "b16_to_b32"} : memref<128xf16, 1>, index -> <f16>
                          %171 = ascvec.cast %170, %mask {rounding = 6 : i32} : <f16> -> <f32>
                          ascvec.storealign %arg12[%169], %171, %mask : memref<128xf32, 1>, index, <f32>
                        }
                      } {cannir.vf_mode = "simd"}
                      ascvec.release_buf v, %158, 0 {cannir.buffer.generated_sync}
                      ascvec.get_buf v, %163, 0 {cannir.buffer.generated_sync}
                      ascvec.get_buf v, %166, 0 {cannir.buffer.generated_sync}
                      ascvec.get_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
                      ascvec.vf(%162, %165, %alloc_12, %alloc_4 : memref<128xf16, 1>, memref<128xf32, 1>, memref<128xf32, 1>, memref<16384xf32, 1>) {
                      ^bb0(%arg11: memref<128xf16, 1>, %arg12: memref<128xf32, 1>, %arg13: memref<128xf32, 1>, %arg14: memref<16384xf32, 1>):
                        %c64 = arith.constant 64 : index
                        %c1_i32_13 = arith.constant 1 : i32
                        %c0_i32_14 = arith.constant 0 : i32
                        %c128_i32 = arith.constant 128 : i32
                        %c0_15 = arith.constant 0 : index
                        %c64_i32 = arith.constant 64 : i32
                        %mask, %remaining_out = ascvec.update_mask %c64_i32 : !ascvec.mask
                        %167 = ascvec.loadalign_unpack %arg11[%c0_15] {mode = "b16_to_b32"} : memref<128xf16, 1>, index -> <f16>
                        %168 = ascvec.loadalign_unpack %arg11[%c64] {mode = "b16_to_b32"} : memref<128xf16, 1>, index -> <f16>
                        %169 = ascvec.cast %167, %mask {rounding = 6 : i32} : <f16> -> <f32>
                        %170 = ascvec.cast %168, %mask {rounding = 6 : i32} : <f16> -> <f32>
                        %171 = ascvec.loadalign %arg12[%c0_15] : memref<128xf32, 1>, index -> <f32>
                        %172 = ascvec.loadalign %arg12[%c64] : memref<128xf32, 1>, index -> <f32>
                        scf.for %arg15 = %c0_i32_14 to %c128_i32 step %c1_i32_13  : i32 {
                          %173 = arith.muli %arg15, %c128_i32 overflow<nsw> : i32
                          %174 = arith.index_cast %arg15 : i32 to index
                          %175 = ascvec.loadalign_brc %arg13[%174] : memref<128xf32, 1>, index -> <f32>
                          %176 = arith.index_cast %173 : i32 to index
                          %177 = ascvec.loadalign %arg14[%176] : memref<16384xf32, 1>, index -> <f32>
                          %178 = arith.addi %173, %c64_i32 overflow<nsw> : i32
                          %179 = arith.index_cast %178 : i32 to index
                          %180 = ascvec.loadalign %arg14[%179] : memref<16384xf32, 1>, index -> <f32>
                          %181 = ascvec.mul %177, %171, %mask : <f32>
                          %182 = ascvec.mul %180, %172, %mask : <f32>
                          %183 = ascvec.copy %175, %mask : <f32>
                          %184 = ascvec.madd_two_address %183, %169, %181, %mask : <f32>
                          %185 = ascvec.madd_two_address %175, %170, %182, %mask : <f32>
                          ascvec.storealign %arg14[%176], %184, %mask : memref<16384xf32, 1>, index, <f32>
                          ascvec.storealign %arg14[%179], %185, %mask : memref<16384xf32, 1>, index, <f32>
                        } {cannir.loop_unroll = 4 : i64}
                      } {cannir.vf_mode = "simd"}
                      ascvec.release_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
                      ascvec.release_buf v, %166, 0 {cannir.buffer.generated_sync}
                      ascvec.release_buf v, %163, 0 {cannir.buffer.generated_sync}
                      scf.yield %154, %160 : i64, i64
                    } else {
                      scf.yield %arg9, %arg10 : i64, i64
                    }
                    scf.yield %151#0, %151#1 : i64, i64
                  }
                  %142 = arith.muli %37, %c16384_i64 : i64
                  %143 = arith.addi %62, %142 : i64
                  %144 = arith.muli %143, %c32_i64 : i64
                  %145 = ascscalar.memref_address %alloc : memref<1xf32>
                  %146 = arith.divui %144, %c8_i64 : i64
                  %147 = arith.addi %145, %146 : i64
                  %148 = ascscalar.bind_gm_ptr %147 : memref<1xf32>
                  %149 = ascscalar.bitcast_memref %alloc_4 byte_offset = %c0_i64 : memref<16384xf32, 1> -> memref<?xf32, 1>
                  ascvec.get_buf mte3, %c0_i64, 0 {cannir.buffer.generated_sync}
                  ascvec.copy_ub2gm %148[%c0], %149[%c0], %c1_i32, %c65536_i32, %c0_i64, %c65536_i32 l2_cache_ctl %c0_i32 : memref<1xf32>, memref<?xf32, 1>
                  ascvec.release_buf mte3, %c0_i64, 0 {cannir.buffer.generated_sync}
                  scf.yield %141#0, %141#1, %141#0, %141#0 : i64, i64, i64, i64
                } else {
                  scf.yield %arg6, %arg7, %arg6, %arg6 : i64, i64, i64, i64
                }
                %57 = arith.cmpi sgt, %54, %c0_i64 : i64
                %58:2 = scf.if %57 -> (i64, i64) {
                  %59 = arith.cmpi eq, %51, %c0_i64 : i64
                  %60 = scf.if %59 -> (i64) {
                    %74 = cannir.extract_physical_layout %arg0 : !cannir.tensor<<gm, f32>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, identity>
                    %75 = cannir.extract_stride %74 : <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>> -> <(-1, 16384, 128, 1)>
                    %76 = cannir.extract_dim %75[0] : !cannir.stride<(-1, 16384, 128, 1)>
                    %77 = arith.muli %46, %76 : i64
                    %78 = arith.muli %77, %c4_i64 : i64
                    %79 = arith.muli %48, %c65536_i64 : i64
                    %80 = arith.addi %78, %79 : i64
                    ascvec.get_buf mte2, %c1_i64, 0 {cannir.buffer.generated_sync}
                    "ascvec.copy_gm2ub_plan"(%alloc_5, %alloc, %c0_i32, %c0_i32, %c0_i64, %80, %c1_i64, %c65536_i64, %c0_i64, %c65536_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 65536 : i64, ub_capacity = 65536 : i64}> : (memref<16384xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                    ascvec.release_buf mte2, %c1_i64, 0 {cannir.buffer.generated_sync}
                    %81 = arith.cmpi eq, %56#0, %c0_i64 : i64
                    %82 = ascscalar.memref_address %alloc_6 : memref<128xf16, 1>
                    %83 = ascscalar.memref_address %alloc_7 : memref<128xf16, 1>
                    %84 = arith.select %81, %82, %83 : i64
                    %85 = arith.select %81, %c2_i64, %c3_i64 : i64
                    %86 = arith.addi %56#0, %c1_i64 : i64
                    %87 = arith.remui %86, %c2_i64 : i64
                    %88 = cannir.extract_dim %10[2] : !cannir.shape<(-1, -1, -1, 128)>
                    %89 = arith.minsi %48, %88 : i64
                    %90 = arith.subi %88, %89 : i64
                    %91 = arith.minsi %90, %c1_i64 : i64
                    %92 = arith.cmpi sle, %48, %88 : i64
                    %93 = arith.select %92, %91, %c0_i64 : i64
                    %94 = cannir.extract_physical_layout %arg1 : !cannir.tensor<<gm, f16>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                    %95 = cannir.extract_stride %94 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, 128, 1)>
                    %96 = cannir.extract_dim %95[0] : !cannir.stride<(-1, -1, 128, 1)>
                    %97 = arith.muli %46, %96 : i64
                    %98 = arith.muli %97, %c2_i64 : i64
                    %99 = arith.muli %48, %c256_i64 : i64
                    %100 = arith.addi %98, %99 : i64
                    %101 = arith.muli %93, %c256_i64 : i64
                    %102 = arith.cmpi eq, %101, %c0_i64 : i64
                    %103 = arith.cmpi ne, %101, %c0_i64 : i64
                    %104 = arith.extui %103 : i1 to i64
                    %105 = arith.select %102, %c2_i64, %101 : i64
                    %106 = ascscalar.bind_local_ptr %84 : memref<1xf16, 1>
                    ascvec.get_buf mte2, %85, 0 {cannir.buffer.generated_sync}
                    "ascvec.copy_gm2ub_plan"(%106, %alloc_0, %c0_i32, %c0_i32, %c0_i64, %100, %104, %105, %c0_i64, %105, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xf16, 1>, memref<1xf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                    ascvec.release_buf mte2, %85, 0 {cannir.buffer.generated_sync}
                    %107 = arith.cmpi eq, %56#2, %c0_i64 : i64
                    %108 = ascscalar.memref_address %alloc_8 : memref<128xf16, 1>
                    %109 = ascscalar.memref_address %alloc_9 : memref<128xf16, 1>
                    %110 = arith.select %107, %108, %109 : i64
                    %111 = arith.select %107, %c4_i64, %c5_i64 : i64
                    %112 = cannir.extract_layout %arg2 : !cannir.tensor<<gm, f16>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                    %113 = cannir.extract_shape %112 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, -1, 128)>
                    %114 = cannir.extract_dim %113[2] : !cannir.shape<(-1, -1, -1, 128)>
                    %115 = arith.minsi %48, %114 : i64
                    %116 = arith.subi %114, %115 : i64
                    %117 = arith.minsi %116, %c1_i64 : i64
                    %118 = arith.cmpi sle, %48, %114 : i64
                    %119 = arith.select %118, %117, %c0_i64 : i64
                    %120 = cannir.extract_physical_layout %arg2 : !cannir.tensor<<gm, f16>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                    %121 = cannir.extract_stride %120 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, 128, 1)>
                    %122 = cannir.extract_dim %121[0] : !cannir.stride<(-1, -1, 128, 1)>
                    %123 = arith.muli %46, %122 : i64
                    %124 = arith.muli %123, %c2_i64 : i64
                    %125 = arith.addi %124, %99 : i64
                    %126 = arith.muli %119, %c256_i64 : i64
                    %127 = arith.cmpi eq, %126, %c0_i64 : i64
                    %128 = arith.cmpi ne, %126, %c0_i64 : i64
                    %129 = arith.extui %128 : i1 to i64
                    %130 = arith.select %127, %c2_i64, %126 : i64
                    %131 = ascscalar.bind_local_ptr %110 : memref<1xf16, 1>
                    ascvec.get_buf mte2, %111, 0 {cannir.buffer.generated_sync}
                    "ascvec.copy_gm2ub_plan"(%131, %alloc_1, %c0_i32, %c0_i32, %c0_i64, %125, %129, %130, %c0_i64, %130, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xf16, 1>, memref<1xf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                    ascvec.release_buf mte2, %111, 0 {cannir.buffer.generated_sync}
                    %132 = arith.cmpi eq, %56#3, %c0_i64 : i64
                    %133 = ascscalar.memref_address %alloc_10 : memref<128xf32, 1>
                    %134 = ascscalar.memref_address %alloc_11 : memref<128xf32, 1>
                    %135 = arith.select %132, %133, %134 : i64
                    %136 = arith.select %132, %c6_i64, %c7_i64 : i64
                    %137 = cannir.extract_layout %arg3 : !cannir.tensor<<gm, f32>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                    %138 = cannir.extract_shape %137 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, -1, 128)>
                    %139 = cannir.extract_dim %138[2] : !cannir.shape<(-1, -1, -1, 128)>
                    %140 = arith.minsi %48, %139 : i64
                    %141 = arith.subi %139, %140 : i64
                    %142 = arith.minsi %141, %c1_i64 : i64
                    %143 = arith.cmpi sle, %48, %139 : i64
                    %144 = arith.select %143, %142, %c0_i64 : i64
                    %145 = cannir.extract_physical_layout %arg3 : !cannir.tensor<<gm, f32>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                    %146 = cannir.extract_stride %145 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, 128, 1)>
                    %147 = cannir.extract_dim %146[0] : !cannir.stride<(-1, -1, 128, 1)>
                    %148 = arith.muli %46, %147 : i64
                    %149 = arith.muli %148, %c4_i64 : i64
                    %150 = arith.muli %48, %c512_i64 : i64
                    %151 = arith.addi %149, %150 : i64
                    %152 = arith.muli %144, %c512_i64 : i64
                    %153 = arith.cmpi eq, %152, %c0_i64 : i64
                    %154 = arith.cmpi ne, %152, %c0_i64 : i64
                    %155 = arith.extui %154 : i1 to i64
                    %156 = arith.select %153, %c4_i64, %152 : i64
                    %157 = ascscalar.bind_local_ptr %135 : memref<1xf32, 1>
                    ascvec.get_buf mte2, %136, 0 {cannir.buffer.generated_sync}
                    "ascvec.copy_gm2ub_plan"(%157, %alloc_2, %c0_i32, %c0_i32, %c0_i64, %151, %155, %156, %c0_i64, %156, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = -1 : i64, ub_capacity = 4 : i64}> : (memref<1xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                    ascvec.release_buf mte2, %136, 0 {cannir.buffer.generated_sync}
                    scf.yield %87 : i64
                  } else {
                    scf.yield %56#0 : i64
                  }
                  %61:2 = scf.for %arg8 = %c0_i64 to %11 step %c1_i64 iter_args(%arg9 = %60, %arg10 = %56#1) -> (i64, i64)  : i64 {
                    %74 = arith.cmpi slt, %arg8, %54 : i64
                    %75:2 = scf.if %74 -> (i64, i64) {
                      %76 = arith.addi %arg8, %c1_i64 : i64
                      %77 = arith.cmpi slt, %76, %54 : i64
                      %78 = scf.if %77 -> (i64) {
                        %97 = arith.cmpi eq, %arg9, %c0_i64 : i64
                        %98 = ascscalar.memref_address %alloc_6 : memref<128xf16, 1>
                        %99 = ascscalar.memref_address %alloc_7 : memref<128xf16, 1>
                        %100 = arith.select %97, %98, %99 : i64
                        %101 = arith.select %97, %c2_i64, %c3_i64 : i64
                        %102 = arith.addi %arg9, %c1_i64 : i64
                        %103 = arith.remui %102, %c2_i64 : i64
                        %104 = cannir.extract_dim %10[2] : !cannir.shape<(-1, -1, -1, 128)>
                        %105 = arith.minsi %48, %104 : i64
                        %106 = arith.subi %104, %105 : i64
                        %107 = arith.minsi %106, %c1_i64 : i64
                        %108 = arith.cmpi sle, %48, %104 : i64
                        %109 = arith.select %108, %107, %c0_i64 : i64
                        %110 = cannir.extract_physical_layout %arg1 : !cannir.tensor<<gm, f16>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                        %111 = cannir.extract_stride %110 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, 128, 1)>
                        %112 = cannir.extract_dim %111[0] : !cannir.stride<(-1, -1, 128, 1)>
                        %113 = arith.muli %46, %112 : i64
                        %114 = arith.muli %113, %c2_i64 : i64
                        %115 = cannir.extract_dim %111[1] : !cannir.stride<(-1, -1, 128, 1)>
                        %116 = arith.muli %76, %115 : i64
                        %117 = arith.muli %116, %c2_i64 : i64
                        %118 = arith.addi %114, %117 : i64
                        %119 = arith.muli %48, %c256_i64 : i64
                        %120 = arith.addi %118, %119 : i64
                        %121 = arith.muli %109, %c256_i64 : i64
                        %122 = arith.cmpi eq, %121, %c0_i64 : i64
                        %123 = arith.cmpi ne, %121, %c0_i64 : i64
                        %124 = arith.extui %123 : i1 to i64
                        %125 = arith.select %122, %c2_i64, %121 : i64
                        %126 = ascscalar.bind_local_ptr %100 : memref<1xf16, 1>
                        ascvec.get_buf mte2, %101, 0 {cannir.buffer.generated_sync}
                        "ascvec.copy_gm2ub_plan"(%126, %alloc_0, %c0_i32, %c0_i32, %c0_i64, %120, %124, %125, %c0_i64, %125, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xf16, 1>, memref<1xf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                        ascvec.release_buf mte2, %101, 0 {cannir.buffer.generated_sync}
                        %127 = ascscalar.memref_address %alloc_8 : memref<128xf16, 1>
                        %128 = ascscalar.memref_address %alloc_9 : memref<128xf16, 1>
                        %129 = arith.select %97, %127, %128 : i64
                        %130 = arith.select %97, %c4_i64, %c5_i64 : i64
                        %131 = cannir.extract_layout %arg2 : !cannir.tensor<<gm, f16>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                        %132 = cannir.extract_shape %131 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, -1, 128)>
                        %133 = cannir.extract_dim %132[2] : !cannir.shape<(-1, -1, -1, 128)>
                        %134 = arith.minsi %48, %133 : i64
                        %135 = arith.subi %133, %134 : i64
                        %136 = arith.minsi %135, %c1_i64 : i64
                        %137 = arith.cmpi sle, %48, %133 : i64
                        %138 = arith.select %137, %136, %c0_i64 : i64
                        %139 = cannir.extract_physical_layout %arg2 : !cannir.tensor<<gm, f16>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                        %140 = cannir.extract_stride %139 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, 128, 1)>
                        %141 = cannir.extract_dim %140[0] : !cannir.stride<(-1, -1, 128, 1)>
                        %142 = arith.muli %46, %141 : i64
                        %143 = arith.muli %142, %c2_i64 : i64
                        %144 = cannir.extract_dim %140[1] : !cannir.stride<(-1, -1, 128, 1)>
                        %145 = arith.muli %76, %144 : i64
                        %146 = arith.muli %145, %c2_i64 : i64
                        %147 = arith.addi %143, %146 : i64
                        %148 = arith.addi %147, %119 : i64
                        %149 = arith.muli %138, %c256_i64 : i64
                        %150 = arith.cmpi eq, %149, %c0_i64 : i64
                        %151 = arith.cmpi ne, %149, %c0_i64 : i64
                        %152 = arith.extui %151 : i1 to i64
                        %153 = arith.select %150, %c2_i64, %149 : i64
                        %154 = ascscalar.bind_local_ptr %129 : memref<1xf16, 1>
                        ascvec.get_buf mte2, %130, 0 {cannir.buffer.generated_sync}
                        "ascvec.copy_gm2ub_plan"(%154, %alloc_1, %c0_i32, %c0_i32, %c0_i64, %148, %152, %153, %c0_i64, %153, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xf16, 1>, memref<1xf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                        ascvec.release_buf mte2, %130, 0 {cannir.buffer.generated_sync}
                        %155 = ascscalar.memref_address %alloc_10 : memref<128xf32, 1>
                        %156 = ascscalar.memref_address %alloc_11 : memref<128xf32, 1>
                        %157 = arith.select %97, %155, %156 : i64
                        %158 = arith.select %97, %c6_i64, %c7_i64 : i64
                        %159 = cannir.extract_layout %arg3 : !cannir.tensor<<gm, f32>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                        %160 = cannir.extract_shape %159 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, -1, 128)>
                        %161 = cannir.extract_dim %160[2] : !cannir.shape<(-1, -1, -1, 128)>
                        %162 = arith.minsi %48, %161 : i64
                        %163 = arith.subi %161, %162 : i64
                        %164 = arith.minsi %163, %c1_i64 : i64
                        %165 = arith.cmpi sle, %48, %161 : i64
                        %166 = arith.select %165, %164, %c0_i64 : i64
                        %167 = cannir.extract_physical_layout %arg3 : !cannir.tensor<<gm, f32>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                        %168 = cannir.extract_stride %167 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, 128, 1)>
                        %169 = cannir.extract_dim %168[0] : !cannir.stride<(-1, -1, 128, 1)>
                        %170 = arith.muli %46, %169 : i64
                        %171 = arith.muli %170, %c4_i64 : i64
                        %172 = cannir.extract_dim %168[1] : !cannir.stride<(-1, -1, 128, 1)>
                        %173 = arith.muli %76, %172 : i64
                        %174 = arith.muli %173, %c4_i64 : i64
                        %175 = arith.addi %171, %174 : i64
                        %176 = arith.muli %48, %c512_i64 : i64
                        %177 = arith.addi %175, %176 : i64
                        %178 = arith.muli %166, %c512_i64 : i64
                        %179 = arith.cmpi eq, %178, %c0_i64 : i64
                        %180 = arith.cmpi ne, %178, %c0_i64 : i64
                        %181 = arith.extui %180 : i1 to i64
                        %182 = arith.select %179, %c4_i64, %178 : i64
                        %183 = ascscalar.bind_local_ptr %157 : memref<1xf32, 1>
                        ascvec.get_buf mte2, %158, 0 {cannir.buffer.generated_sync}
                        "ascvec.copy_gm2ub_plan"(%183, %alloc_2, %c0_i32, %c0_i32, %c0_i64, %177, %181, %182, %c0_i64, %182, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = -1 : i64, ub_capacity = 4 : i64}> : (memref<1xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                        ascvec.release_buf mte2, %158, 0 {cannir.buffer.generated_sync}
                        scf.yield %103 : i64
                      } else {
                        scf.yield %arg9 : i64
                      }
                      %79 = arith.cmpi eq, %arg10, %c0_i64 : i64
                      %80 = ascscalar.memref_address %alloc_6 : memref<128xf16, 1>
                      %81 = ascscalar.memref_address %alloc_7 : memref<128xf16, 1>
                      %82 = arith.select %79, %80, %81 : i64
                      %83 = ascscalar.bind_local_ptr %82 : memref<128xf16, 1>
                      %84 = arith.select %79, %c2_i64, %c3_i64 : i64
                      %85 = arith.addi %arg10, %c1_i64 : i64
                      %86 = arith.remui %85, %c2_i64 : i64
                      %87 = ascscalar.memref_address %alloc_8 : memref<128xf16, 1>
                      %88 = ascscalar.memref_address %alloc_9 : memref<128xf16, 1>
                      %89 = arith.select %79, %87, %88 : i64
                      %90 = ascscalar.bind_local_ptr %89 : memref<128xf16, 1>
                      %91 = arith.select %79, %c4_i64, %c5_i64 : i64
                      %92 = ascscalar.memref_address %alloc_10 : memref<128xf32, 1>
                      %93 = ascscalar.memref_address %alloc_11 : memref<128xf32, 1>
                      %94 = arith.select %79, %92, %93 : i64
                      %95 = ascscalar.bind_local_ptr %94 : memref<128xf32, 1>
                      %96 = arith.select %79, %c6_i64, %c7_i64 : i64
                      ascvec.get_buf v, %84, 0 {cannir.buffer.generated_sync}
                      ascvec.vf(%83, %alloc_12 : memref<128xf16, 1>, memref<128xf32, 1>) {
                      ^bb0(%arg11: memref<128xf16, 1>, %arg12: memref<128xf32, 1>):
                        %c1_i32_13 = arith.constant 1 : i32
                        %c2_i32 = arith.constant 2 : i32
                        %c0_i32_14 = arith.constant 0 : i32
                        %c128_i32 = arith.constant 128 : i32
                        %c64_i32 = arith.constant 64 : i32
                        scf.for %arg13 = %c0_i32_14 to %c2_i32 step %c1_i32_13  : i32 {
                          %97 = arith.muli %arg13, %c64_i32 overflow<nsw> : i32
                          %98 = arith.subi %c128_i32, %97 overflow<nsw> : i32
                          %mask, %remaining_out = ascvec.update_mask %98 : !ascvec.mask
                          %99 = arith.index_cast %97 : i32 to index
                          %100 = ascvec.loadalign_unpack %arg11[%99] {mode = "b16_to_b32"} : memref<128xf16, 1>, index -> <f16>
                          %101 = ascvec.cast %100, %mask {rounding = 6 : i32} : <f16> -> <f32>
                          ascvec.storealign %arg12[%99], %101, %mask : memref<128xf32, 1>, index, <f32>
                        }
                      } {cannir.vf_mode = "simd"}
                      ascvec.release_buf v, %84, 0 {cannir.buffer.generated_sync}
                      ascvec.get_buf v, %91, 0 {cannir.buffer.generated_sync}
                      ascvec.get_buf v, %96, 0 {cannir.buffer.generated_sync}
                      ascvec.get_buf v, %c1_i64, 0 {cannir.buffer.generated_sync}
                      ascvec.vf(%90, %95, %alloc_12, %alloc_5 : memref<128xf16, 1>, memref<128xf32, 1>, memref<128xf32, 1>, memref<16384xf32, 1>) {
                      ^bb0(%arg11: memref<128xf16, 1>, %arg12: memref<128xf32, 1>, %arg13: memref<128xf32, 1>, %arg14: memref<16384xf32, 1>):
                        %c64 = arith.constant 64 : index
                        %c1_i32_13 = arith.constant 1 : i32
                        %c0_i32_14 = arith.constant 0 : i32
                        %c128_i32 = arith.constant 128 : i32
                        %c0_15 = arith.constant 0 : index
                        %c64_i32 = arith.constant 64 : i32
                        %mask, %remaining_out = ascvec.update_mask %c64_i32 : !ascvec.mask
                        %97 = ascvec.loadalign_unpack %arg11[%c0_15] {mode = "b16_to_b32"} : memref<128xf16, 1>, index -> <f16>
                        %98 = ascvec.loadalign_unpack %arg11[%c64] {mode = "b16_to_b32"} : memref<128xf16, 1>, index -> <f16>
                        %99 = ascvec.cast %97, %mask {rounding = 6 : i32} : <f16> -> <f32>
                        %100 = ascvec.cast %98, %mask {rounding = 6 : i32} : <f16> -> <f32>
                        %101 = ascvec.loadalign %arg12[%c0_15] : memref<128xf32, 1>, index -> <f32>
                        %102 = ascvec.loadalign %arg12[%c64] : memref<128xf32, 1>, index -> <f32>
                        scf.for %arg15 = %c0_i32_14 to %c128_i32 step %c1_i32_13  : i32 {
                          %103 = arith.muli %arg15, %c128_i32 overflow<nsw> : i32
                          %104 = arith.index_cast %arg15 : i32 to index
                          %105 = ascvec.loadalign_brc %arg13[%104] : memref<128xf32, 1>, index -> <f32>
                          %106 = arith.index_cast %103 : i32 to index
                          %107 = ascvec.loadalign %arg14[%106] : memref<16384xf32, 1>, index -> <f32>
                          %108 = arith.addi %103, %c64_i32 overflow<nsw> : i32
                          %109 = arith.index_cast %108 : i32 to index
                          %110 = ascvec.loadalign %arg14[%109] : memref<16384xf32, 1>, index -> <f32>
                          %111 = ascvec.mul %107, %101, %mask : <f32>
                          %112 = ascvec.mul %110, %102, %mask : <f32>
                          %113 = ascvec.copy %105, %mask : <f32>
                          %114 = ascvec.madd_two_address %113, %99, %111, %mask : <f32>
                          %115 = ascvec.madd_two_address %105, %100, %112, %mask : <f32>
                          ascvec.storealign %arg14[%106], %114, %mask : memref<16384xf32, 1>, index, <f32>
                          ascvec.storealign %arg14[%109], %115, %mask : memref<16384xf32, 1>, index, <f32>
                        } {cannir.loop_unroll = 4 : i64}
                      } {cannir.vf_mode = "simd"}
                      ascvec.release_buf v, %c1_i64, 0 {cannir.buffer.generated_sync}
                      ascvec.release_buf v, %96, 0 {cannir.buffer.generated_sync}
                      ascvec.release_buf v, %91, 0 {cannir.buffer.generated_sync}
                      scf.yield %78, %86 : i64, i64
                    } else {
                      scf.yield %arg9, %arg10 : i64, i64
                    }
                    scf.yield %75#0, %75#1 : i64, i64
                  }
                  %62 = cannir.extract_physical_layout %arg0 : !cannir.tensor<<gm, f32>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, identity>
                  %63 = cannir.extract_stride %62 : <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>> -> <(-1, 16384, 128, 1)>
                  %64 = cannir.extract_dim %63[0] : !cannir.stride<(-1, 16384, 128, 1)>
                  %65 = arith.muli %46, %64 : i64
                  %66 = arith.muli %48, %c16384_i64 : i64
                  %67 = arith.addi %65, %66 : i64
                  %68 = arith.muli %67, %c32_i64 : i64
                  %69 = ascscalar.memref_address %alloc : memref<1xf32>
                  %70 = arith.divui %68, %c8_i64 : i64
                  %71 = arith.addi %69, %70 : i64
                  %72 = ascscalar.bind_gm_ptr %71 : memref<1xf32>
                  %73 = ascscalar.bitcast_memref %alloc_5 byte_offset = %c0_i64 : memref<16384xf32, 1> -> memref<?xf32, 1>
                  ascvec.get_buf mte3, %c1_i64, 0 {cannir.buffer.generated_sync}
                  ascvec.copy_ub2gm %72[%c0], %73[%c0], %c1_i32, %c65536_i32, %c0_i64, %c65536_i32 l2_cache_ctl %c0_i32 : memref<1xf32>, memref<?xf32, 1>
                  ascvec.release_buf mte3, %c1_i64, 0 {cannir.buffer.generated_sync}
                  scf.yield %61#0, %61#1 : i64, i64
                } else {
                  scf.yield %56#0, %56#1 : i64, i64
                }
                scf.yield %58#0, %58#1 : i64, i64
              }
            } else {
              %22 = arith.muli %13, %c3_i64 : i64
              %23 = arith.muli %4, %c2_i64 : i64
              %24 = arith.cmpi sle, %22, %23 : i64
              scf.if %24 {
                %25 = arith.subi %23, %22 : i64
                %26 = arith.cmpi slt, %12, %25 : i64
                scf.if %26 {
                  %27 = arith.muli %12, %c2_i64 : i64
                  %28 = arith.addi %27, %c1_i64 : i64
                  %29 = arith.maxsi %3, %c1_i64 : i64
                  %30 = arith.remsi %27, %29 : i64
                  %31 = arith.cmpi slt, %30, %c0_i64 : i64
                  %32 = arith.divsi %27, %29 : i64
                  %33 = arith.extui %31 : i1 to i64
                  %34 = arith.subi %32, %33 : i64
                  %35 = arith.maxsi %2, %c1_i64 : i64
                  %36 = arith.remsi %34, %35 : i64
                  %37 = arith.cmpi slt, %36, %c0_i64 : i64
                  %38 = arith.select %37, %35, %c0_i64 : i64
                  %39 = arith.addi %36, %38 : i64
                  %40 = arith.select %31, %29, %c0_i64 : i64
                  %41 = arith.addi %30, %40 : i64
                  %42 = arith.remsi %28, %29 : i64
                  %43 = arith.cmpi slt, %42, %c0_i64 : i64
                  %44 = arith.divsi %28, %29 : i64
                  %45 = arith.extui %43 : i1 to i64
                  %46 = arith.subi %44, %45 : i64
                  %47 = arith.remsi %46, %35 : i64
                  %48 = arith.cmpi slt, %47, %c0_i64 : i64
                  %49 = arith.select %48, %35, %c0_i64 : i64
                  %50 = arith.addi %47, %49 : i64
                  %51 = arith.select %43, %29, %c0_i64 : i64
                  %52 = arith.addi %42, %51 : i64
                  %53 = arith.index_cast %39 : i64 to index
                  %54 = memref.load %alloc_3[%53] : memref<1xi32>
                  %55 = arith.extsi %54 : i32 to i64
                  %56 = arith.index_cast %50 : i64 to index
                  %57 = memref.load %alloc_3[%56] : memref<1xi32>
                  %58 = arith.extsi %57 : i32 to i64
                  %59 = arith.cmpi sgt, %55, %c0_i64 : i64
                  %60:4 = scf.if %59 -> (i64, i64, i64, i64) {
                    %62 = cannir.extract_physical_layout %arg0 : !cannir.tensor<<gm, f32>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, identity>
                    %63 = cannir.extract_stride %62 : <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>> -> <(-1, 16384, 128, 1)>
                    %64 = cannir.extract_dim %63[0] : !cannir.stride<(-1, 16384, 128, 1)>
                    %65 = arith.muli %39, %64 : i64
                    %66 = arith.muli %65, %c4_i64 : i64
                    %67 = arith.muli %41, %c65536_i64 : i64
                    %68 = arith.addi %66, %67 : i64
                    ascvec.get_buf mte2, %c0_i64, 0 {cannir.buffer.generated_sync}
                    "ascvec.copy_gm2ub_plan"(%alloc_4, %alloc, %c0_i32, %c0_i32, %c0_i64, %68, %c1_i64, %c65536_i64, %c0_i64, %c65536_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 0 : i64, ub_capacity = 65536 : i64}> : (memref<16384xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                    ascvec.release_buf mte2, %c0_i64, 0 {cannir.buffer.generated_sync}
                    %69 = cannir.extract_dim %10[2] : !cannir.shape<(-1, -1, -1, 128)>
                    %70 = arith.minsi %41, %69 : i64
                    %71 = arith.subi %69, %70 : i64
                    %72 = arith.minsi %71, %c1_i64 : i64
                    %73 = arith.cmpi sle, %41, %69 : i64
                    %74 = arith.select %73, %72, %c0_i64 : i64
                    %75 = cannir.extract_physical_layout %arg1 : !cannir.tensor<<gm, f16>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                    %76 = cannir.extract_stride %75 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, 128, 1)>
                    %77 = cannir.extract_dim %76[0] : !cannir.stride<(-1, -1, 128, 1)>
                    %78 = arith.muli %39, %77 : i64
                    %79 = arith.muli %78, %c2_i64 : i64
                    %80 = arith.muli %41, %c256_i64 : i64
                    %81 = arith.addi %79, %80 : i64
                    %82 = arith.muli %74, %c256_i64 : i64
                    %83 = arith.cmpi eq, %82, %c0_i64 : i64
                    %84 = arith.cmpi ne, %82, %c0_i64 : i64
                    %85 = arith.extui %84 : i1 to i64
                    %86 = arith.select %83, %c2_i64, %82 : i64
                    %87 = ascscalar.bitcast_memref %alloc_6 byte_offset = %c0_i64 : memref<128xf16, 1> -> memref<?xf16, 1>
                    ascvec.get_buf mte2, %c2_i64, 0 {cannir.buffer.generated_sync}
                    "ascvec.copy_gm2ub_plan"(%87, %alloc_0, %c0_i32, %c0_i32, %c0_i64, %81, %85, %86, %c0_i64, %86, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f16, ub_base = 131072 : i64, ub_capacity = 256 : i64}> : (memref<?xf16, 1>, memref<1xf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                    ascvec.release_buf mte2, %c2_i64, 0 {cannir.buffer.generated_sync}
                    %88 = cannir.extract_layout %arg2 : !cannir.tensor<<gm, f16>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                    %89 = cannir.extract_shape %88 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, -1, 128)>
                    %90 = cannir.extract_dim %89[2] : !cannir.shape<(-1, -1, -1, 128)>
                    %91 = arith.minsi %41, %90 : i64
                    %92 = arith.subi %90, %91 : i64
                    %93 = arith.minsi %92, %c1_i64 : i64
                    %94 = arith.cmpi sle, %41, %90 : i64
                    %95 = arith.select %94, %93, %c0_i64 : i64
                    %96 = cannir.extract_physical_layout %arg2 : !cannir.tensor<<gm, f16>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                    %97 = cannir.extract_stride %96 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, 128, 1)>
                    %98 = cannir.extract_dim %97[0] : !cannir.stride<(-1, -1, 128, 1)>
                    %99 = arith.muli %39, %98 : i64
                    %100 = arith.muli %99, %c2_i64 : i64
                    %101 = arith.addi %100, %80 : i64
                    %102 = arith.muli %95, %c256_i64 : i64
                    %103 = arith.cmpi eq, %102, %c0_i64 : i64
                    %104 = arith.cmpi ne, %102, %c0_i64 : i64
                    %105 = arith.extui %104 : i1 to i64
                    %106 = arith.select %103, %c2_i64, %102 : i64
                    %107 = ascscalar.bitcast_memref %alloc_8 byte_offset = %c0_i64 : memref<128xf16, 1> -> memref<?xf16, 1>
                    ascvec.get_buf mte2, %c4_i64, 0 {cannir.buffer.generated_sync}
                    "ascvec.copy_gm2ub_plan"(%107, %alloc_1, %c0_i32, %c0_i32, %c0_i64, %101, %105, %106, %c0_i64, %106, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f16, ub_base = 131584 : i64, ub_capacity = 256 : i64}> : (memref<?xf16, 1>, memref<1xf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                    ascvec.release_buf mte2, %c4_i64, 0 {cannir.buffer.generated_sync}
                    %108 = cannir.extract_layout %arg3 : !cannir.tensor<<gm, f32>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                    %109 = cannir.extract_shape %108 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, -1, 128)>
                    %110 = cannir.extract_dim %109[2] : !cannir.shape<(-1, -1, -1, 128)>
                    %111 = arith.minsi %41, %110 : i64
                    %112 = arith.subi %110, %111 : i64
                    %113 = arith.minsi %112, %c1_i64 : i64
                    %114 = arith.cmpi sle, %41, %110 : i64
                    %115 = arith.select %114, %113, %c0_i64 : i64
                    %116 = cannir.extract_physical_layout %arg3 : !cannir.tensor<<gm, f32>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                    %117 = cannir.extract_stride %116 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, 128, 1)>
                    %118 = cannir.extract_dim %117[0] : !cannir.stride<(-1, -1, 128, 1)>
                    %119 = arith.muli %39, %118 : i64
                    %120 = arith.muli %119, %c4_i64 : i64
                    %121 = arith.muli %41, %c512_i64 : i64
                    %122 = arith.addi %120, %121 : i64
                    %123 = arith.muli %115, %c512_i64 : i64
                    %124 = arith.cmpi eq, %123, %c0_i64 : i64
                    %125 = arith.cmpi ne, %123, %c0_i64 : i64
                    %126 = arith.extui %125 : i1 to i64
                    %127 = arith.select %124, %c4_i64, %123 : i64
                    %128 = ascscalar.bitcast_memref %alloc_10 byte_offset = %c0_i64 : memref<128xf32, 1> -> memref<?xf32, 1>
                    ascvec.get_buf mte2, %c6_i64, 0 {cannir.buffer.generated_sync}
                    "ascvec.copy_gm2ub_plan"(%128, %alloc_2, %c0_i32, %c0_i32, %c0_i64, %122, %126, %127, %c0_i64, %127, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 132096 : i64, ub_capacity = 512 : i64}> : (memref<?xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                    ascvec.release_buf mte2, %c6_i64, 0 {cannir.buffer.generated_sync}
                    %129:2 = scf.for %arg5 = %c0_i64 to %11 step %c1_i64 iter_args(%arg6 = %c1_i64, %arg7 = %c0_i64) -> (i64, i64)  : i64 {
                      %138 = arith.cmpi slt, %arg5, %55 : i64
                      %139:2 = scf.if %138 -> (i64, i64) {
                        %140 = arith.addi %arg5, %c1_i64 : i64
                        %141 = arith.cmpi slt, %140, %55 : i64
                        %142 = scf.if %141 -> (i64) {
                          %161 = arith.cmpi eq, %arg6, %c0_i64 : i64
                          %162 = ascscalar.memref_address %alloc_6 : memref<128xf16, 1>
                          %163 = ascscalar.memref_address %alloc_7 : memref<128xf16, 1>
                          %164 = arith.select %161, %162, %163 : i64
                          %165 = arith.select %161, %c2_i64, %c3_i64 : i64
                          %166 = arith.addi %arg6, %c1_i64 overflow<nsw> : i64
                          %167 = arith.remui %166, %c2_i64 : i64
                          %168 = cannir.extract_dim %76[1] : !cannir.stride<(-1, -1, 128, 1)>
                          %169 = arith.muli %140, %168 : i64
                          %170 = arith.muli %169, %c2_i64 : i64
                          %171 = arith.addi %79, %170 : i64
                          %172 = arith.addi %171, %80 : i64
                          %173 = ascscalar.bind_local_ptr %164 : memref<1xf16, 1>
                          ascvec.get_buf mte2, %165, 0 {cannir.buffer.generated_sync}
                          "ascvec.copy_gm2ub_plan"(%173, %alloc_0, %c0_i32, %c0_i32, %c0_i64, %172, %85, %86, %c0_i64, %86, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xf16, 1>, memref<1xf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                          ascvec.release_buf mte2, %165, 0 {cannir.buffer.generated_sync}
                          %174 = ascscalar.memref_address %alloc_8 : memref<128xf16, 1>
                          %175 = ascscalar.memref_address %alloc_9 : memref<128xf16, 1>
                          %176 = arith.select %161, %174, %175 : i64
                          %177 = arith.select %161, %c4_i64, %c5_i64 : i64
                          %178 = cannir.extract_dim %97[1] : !cannir.stride<(-1, -1, 128, 1)>
                          %179 = arith.muli %140, %178 : i64
                          %180 = arith.muli %179, %c2_i64 : i64
                          %181 = arith.addi %100, %180 : i64
                          %182 = arith.addi %181, %80 : i64
                          %183 = ascscalar.bind_local_ptr %176 : memref<1xf16, 1>
                          ascvec.get_buf mte2, %177, 0 {cannir.buffer.generated_sync}
                          "ascvec.copy_gm2ub_plan"(%183, %alloc_1, %c0_i32, %c0_i32, %c0_i64, %182, %105, %106, %c0_i64, %106, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xf16, 1>, memref<1xf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                          ascvec.release_buf mte2, %177, 0 {cannir.buffer.generated_sync}
                          %184 = ascscalar.memref_address %alloc_10 : memref<128xf32, 1>
                          %185 = ascscalar.memref_address %alloc_11 : memref<128xf32, 1>
                          %186 = arith.select %161, %184, %185 : i64
                          %187 = arith.select %161, %c6_i64, %c7_i64 : i64
                          %188 = cannir.extract_dim %117[1] : !cannir.stride<(-1, -1, 128, 1)>
                          %189 = arith.muli %140, %188 : i64
                          %190 = arith.muli %189, %c4_i64 : i64
                          %191 = arith.addi %120, %190 : i64
                          %192 = arith.addi %191, %121 : i64
                          %193 = ascscalar.bind_local_ptr %186 : memref<1xf32, 1>
                          ascvec.get_buf mte2, %187, 0 {cannir.buffer.generated_sync}
                          "ascvec.copy_gm2ub_plan"(%193, %alloc_2, %c0_i32, %c0_i32, %c0_i64, %192, %126, %127, %c0_i64, %127, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = -1 : i64, ub_capacity = 4 : i64}> : (memref<1xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                          ascvec.release_buf mte2, %187, 0 {cannir.buffer.generated_sync}
                          scf.yield %167 : i64
                        } else {
                          %161 = arith.cmpi sgt, %58, %c0_i64 : i64
                          %162 = scf.if %161 -> (i64) {
                            %163 = arith.muli %50, %64 : i64
                            %164 = arith.muli %163, %c4_i64 : i64
                            %165 = arith.muli %52, %c65536_i64 : i64
                            %166 = arith.addi %164, %165 : i64
                            ascvec.get_buf mte2, %c1_i64, 0 {cannir.buffer.generated_sync}
                            "ascvec.copy_gm2ub_plan"(%alloc_5, %alloc, %c0_i32, %c0_i32, %c0_i64, %166, %c1_i64, %c65536_i64, %c0_i64, %c65536_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 65536 : i64, ub_capacity = 65536 : i64}> : (memref<16384xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                            ascvec.release_buf mte2, %c1_i64, 0 {cannir.buffer.generated_sync}
                            %167 = arith.cmpi eq, %arg6, %c0_i64 : i64
                            %168 = ascscalar.memref_address %alloc_6 : memref<128xf16, 1>
                            %169 = ascscalar.memref_address %alloc_7 : memref<128xf16, 1>
                            %170 = arith.select %167, %168, %169 : i64
                            %171 = arith.select %167, %c2_i64, %c3_i64 : i64
                            %172 = arith.addi %arg6, %c1_i64 overflow<nsw> : i64
                            %173 = arith.remui %172, %c2_i64 : i64
                            %174 = arith.minsi %52, %69 : i64
                            %175 = arith.subi %69, %174 : i64
                            %176 = arith.minsi %175, %c1_i64 : i64
                            %177 = arith.cmpi sle, %52, %69 : i64
                            %178 = arith.select %177, %176, %c0_i64 : i64
                            %179 = arith.muli %50, %77 : i64
                            %180 = arith.muli %179, %c2_i64 : i64
                            %181 = arith.muli %52, %c256_i64 : i64
                            %182 = arith.addi %180, %181 : i64
                            %183 = arith.muli %178, %c256_i64 : i64
                            %184 = arith.cmpi eq, %183, %c0_i64 : i64
                            %185 = arith.cmpi ne, %183, %c0_i64 : i64
                            %186 = arith.extui %185 : i1 to i64
                            %187 = arith.select %184, %c2_i64, %183 : i64
                            %188 = ascscalar.bind_local_ptr %170 : memref<1xf16, 1>
                            ascvec.get_buf mte2, %171, 0 {cannir.buffer.generated_sync}
                            "ascvec.copy_gm2ub_plan"(%188, %alloc_0, %c0_i32, %c0_i32, %c0_i64, %182, %186, %187, %c0_i64, %187, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xf16, 1>, memref<1xf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                            ascvec.release_buf mte2, %171, 0 {cannir.buffer.generated_sync}
                            %189 = ascscalar.memref_address %alloc_8 : memref<128xf16, 1>
                            %190 = ascscalar.memref_address %alloc_9 : memref<128xf16, 1>
                            %191 = arith.select %167, %189, %190 : i64
                            %192 = arith.select %167, %c4_i64, %c5_i64 : i64
                            %193 = arith.minsi %52, %90 : i64
                            %194 = arith.subi %90, %193 : i64
                            %195 = arith.minsi %194, %c1_i64 : i64
                            %196 = arith.cmpi sle, %52, %90 : i64
                            %197 = arith.select %196, %195, %c0_i64 : i64
                            %198 = arith.muli %50, %98 : i64
                            %199 = arith.muli %198, %c2_i64 : i64
                            %200 = arith.addi %199, %181 : i64
                            %201 = arith.muli %197, %c256_i64 : i64
                            %202 = arith.cmpi eq, %201, %c0_i64 : i64
                            %203 = arith.cmpi ne, %201, %c0_i64 : i64
                            %204 = arith.extui %203 : i1 to i64
                            %205 = arith.select %202, %c2_i64, %201 : i64
                            %206 = ascscalar.bind_local_ptr %191 : memref<1xf16, 1>
                            ascvec.get_buf mte2, %192, 0 {cannir.buffer.generated_sync}
                            "ascvec.copy_gm2ub_plan"(%206, %alloc_1, %c0_i32, %c0_i32, %c0_i64, %200, %204, %205, %c0_i64, %205, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xf16, 1>, memref<1xf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                            ascvec.release_buf mte2, %192, 0 {cannir.buffer.generated_sync}
                            %207 = ascscalar.memref_address %alloc_10 : memref<128xf32, 1>
                            %208 = ascscalar.memref_address %alloc_11 : memref<128xf32, 1>
                            %209 = arith.select %167, %207, %208 : i64
                            %210 = arith.select %167, %c6_i64, %c7_i64 : i64
                            %211 = arith.minsi %52, %110 : i64
                            %212 = arith.subi %110, %211 : i64
                            %213 = arith.minsi %212, %c1_i64 : i64
                            %214 = arith.cmpi sle, %52, %110 : i64
                            %215 = arith.select %214, %213, %c0_i64 : i64
                            %216 = arith.muli %50, %118 : i64
                            %217 = arith.muli %216, %c4_i64 : i64
                            %218 = arith.muli %52, %c512_i64 : i64
                            %219 = arith.addi %217, %218 : i64
                            %220 = arith.muli %215, %c512_i64 : i64
                            %221 = arith.cmpi eq, %220, %c0_i64 : i64
                            %222 = arith.cmpi ne, %220, %c0_i64 : i64
                            %223 = arith.extui %222 : i1 to i64
                            %224 = arith.select %221, %c4_i64, %220 : i64
                            %225 = ascscalar.bind_local_ptr %209 : memref<1xf32, 1>
                            ascvec.get_buf mte2, %210, 0 {cannir.buffer.generated_sync}
                            "ascvec.copy_gm2ub_plan"(%225, %alloc_2, %c0_i32, %c0_i32, %c0_i64, %219, %223, %224, %c0_i64, %224, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = -1 : i64, ub_capacity = 4 : i64}> : (memref<1xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                            ascvec.release_buf mte2, %210, 0 {cannir.buffer.generated_sync}
                            scf.yield %173 : i64
                          } else {
                            scf.yield %arg6 : i64
                          }
                          scf.yield %162 : i64
                        }
                        %143 = arith.cmpi eq, %arg7, %c0_i64 : i64
                        %144 = ascscalar.memref_address %alloc_6 : memref<128xf16, 1>
                        %145 = ascscalar.memref_address %alloc_7 : memref<128xf16, 1>
                        %146 = arith.select %143, %144, %145 : i64
                        %147 = ascscalar.bind_local_ptr %146 : memref<128xf16, 1>
                        %148 = arith.select %143, %c2_i64, %c3_i64 : i64
                        %149 = arith.addi %arg7, %c1_i64 overflow<nsw> : i64
                        %150 = arith.remui %149, %c2_i64 : i64
                        %151 = ascscalar.memref_address %alloc_8 : memref<128xf16, 1>
                        %152 = ascscalar.memref_address %alloc_9 : memref<128xf16, 1>
                        %153 = arith.select %143, %151, %152 : i64
                        %154 = ascscalar.bind_local_ptr %153 : memref<128xf16, 1>
                        %155 = arith.select %143, %c4_i64, %c5_i64 : i64
                        %156 = ascscalar.memref_address %alloc_10 : memref<128xf32, 1>
                        %157 = ascscalar.memref_address %alloc_11 : memref<128xf32, 1>
                        %158 = arith.select %143, %156, %157 : i64
                        %159 = ascscalar.bind_local_ptr %158 : memref<128xf32, 1>
                        %160 = arith.select %143, %c6_i64, %c7_i64 : i64
                        ascvec.get_buf v, %148, 0 {cannir.buffer.generated_sync}
                        ascvec.vf(%147, %alloc_12 : memref<128xf16, 1>, memref<128xf32, 1>) {
                        ^bb0(%arg8: memref<128xf16, 1>, %arg9: memref<128xf32, 1>):
                          %c1_i32_13 = arith.constant 1 : i32
                          %c2_i32 = arith.constant 2 : i32
                          %c0_i32_14 = arith.constant 0 : i32
                          %c128_i32 = arith.constant 128 : i32
                          %c64_i32 = arith.constant 64 : i32
                          scf.for %arg10 = %c0_i32_14 to %c2_i32 step %c1_i32_13  : i32 {
                            %161 = arith.muli %arg10, %c64_i32 overflow<nsw> : i32
                            %162 = arith.subi %c128_i32, %161 overflow<nsw> : i32
                            %mask, %remaining_out = ascvec.update_mask %162 : !ascvec.mask
                            %163 = arith.index_cast %161 : i32 to index
                            %164 = ascvec.loadalign_unpack %arg8[%163] {mode = "b16_to_b32"} : memref<128xf16, 1>, index -> <f16>
                            %165 = ascvec.cast %164, %mask {rounding = 6 : i32} : <f16> -> <f32>
                            ascvec.storealign %arg9[%163], %165, %mask : memref<128xf32, 1>, index, <f32>
                          }
                        } {cannir.vf_mode = "simd"}
                        ascvec.release_buf v, %148, 0 {cannir.buffer.generated_sync}
                        ascvec.get_buf v, %155, 0 {cannir.buffer.generated_sync}
                        ascvec.get_buf v, %160, 0 {cannir.buffer.generated_sync}
                        ascvec.get_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
                        ascvec.vf(%154, %159, %alloc_12, %alloc_4 : memref<128xf16, 1>, memref<128xf32, 1>, memref<128xf32, 1>, memref<16384xf32, 1>) {
                        ^bb0(%arg8: memref<128xf16, 1>, %arg9: memref<128xf32, 1>, %arg10: memref<128xf32, 1>, %arg11: memref<16384xf32, 1>):
                          %c64 = arith.constant 64 : index
                          %c1_i32_13 = arith.constant 1 : i32
                          %c0_i32_14 = arith.constant 0 : i32
                          %c128_i32 = arith.constant 128 : i32
                          %c0_15 = arith.constant 0 : index
                          %c64_i32 = arith.constant 64 : i32
                          %mask, %remaining_out = ascvec.update_mask %c64_i32 : !ascvec.mask
                          %161 = ascvec.loadalign_unpack %arg8[%c0_15] {mode = "b16_to_b32"} : memref<128xf16, 1>, index -> <f16>
                          %162 = ascvec.loadalign_unpack %arg8[%c64] {mode = "b16_to_b32"} : memref<128xf16, 1>, index -> <f16>
                          %163 = ascvec.cast %161, %mask {rounding = 6 : i32} : <f16> -> <f32>
                          %164 = ascvec.cast %162, %mask {rounding = 6 : i32} : <f16> -> <f32>
                          %165 = ascvec.loadalign %arg9[%c0_15] : memref<128xf32, 1>, index -> <f32>
                          %166 = ascvec.loadalign %arg9[%c64] : memref<128xf32, 1>, index -> <f32>
                          scf.for %arg12 = %c0_i32_14 to %c128_i32 step %c1_i32_13  : i32 {
                            %167 = arith.muli %arg12, %c128_i32 overflow<nsw> : i32
                            %168 = arith.index_cast %arg12 : i32 to index
                            %169 = ascvec.loadalign_brc %arg10[%168] : memref<128xf32, 1>, index -> <f32>
                            %170 = arith.index_cast %167 : i32 to index
                            %171 = ascvec.loadalign %arg11[%170] : memref<16384xf32, 1>, index -> <f32>
                            %172 = arith.addi %167, %c64_i32 overflow<nsw> : i32
                            %173 = arith.index_cast %172 : i32 to index
                            %174 = ascvec.loadalign %arg11[%173] : memref<16384xf32, 1>, index -> <f32>
                            %175 = ascvec.mul %171, %165, %mask : <f32>
                            %176 = ascvec.mul %174, %166, %mask : <f32>
                            %177 = ascvec.copy %169, %mask : <f32>
                            %178 = ascvec.madd_two_address %177, %163, %175, %mask : <f32>
                            %179 = ascvec.madd_two_address %169, %164, %176, %mask : <f32>
                            ascvec.storealign %arg11[%170], %178, %mask : memref<16384xf32, 1>, index, <f32>
                            ascvec.storealign %arg11[%173], %179, %mask : memref<16384xf32, 1>, index, <f32>
                          } {cannir.loop_unroll = 4 : i64}
                        } {cannir.vf_mode = "simd"}
                        ascvec.release_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
                        ascvec.release_buf v, %160, 0 {cannir.buffer.generated_sync}
                        ascvec.release_buf v, %155, 0 {cannir.buffer.generated_sync}
                        scf.yield %142, %150 : i64, i64
                      } else {
                        scf.yield %arg6, %arg7 : i64, i64
                      }
                      scf.yield %139#0, %139#1 : i64, i64
                    }
                    %130 = arith.muli %41, %c16384_i64 : i64
                    %131 = arith.addi %65, %130 : i64
                    %132 = arith.muli %131, %c32_i64 : i64
                    %133 = ascscalar.memref_address %alloc : memref<1xf32>
                    %134 = arith.divui %132, %c8_i64 : i64
                    %135 = arith.addi %133, %134 : i64
                    %136 = ascscalar.bind_gm_ptr %135 : memref<1xf32>
                    %137 = ascscalar.bitcast_memref %alloc_4 byte_offset = %c0_i64 : memref<16384xf32, 1> -> memref<?xf32, 1>
                    ascvec.get_buf mte3, %c0_i64, 0 {cannir.buffer.generated_sync}
                    ascvec.copy_ub2gm %136[%c0], %137[%c0], %c1_i32, %c65536_i32, %c0_i64, %c65536_i32 l2_cache_ctl %c0_i32 : memref<1xf32>, memref<?xf32, 1>
                    ascvec.release_buf mte3, %c0_i64, 0 {cannir.buffer.generated_sync}
                    scf.yield %129#0, %129#1, %129#0, %129#0 : i64, i64, i64, i64
                  } else {
                    scf.yield %c0_i64, %c0_i64, %c0_i64, %c0_i64 : i64, i64, i64, i64
                  }
                  %61 = arith.cmpi sgt, %58, %c0_i64 : i64
                  scf.if %61 {
                    %62 = arith.cmpi eq, %55, %c0_i64 : i64
                    %63 = scf.if %62 -> (i64) {
                      %77 = cannir.extract_physical_layout %arg0 : !cannir.tensor<<gm, f32>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, identity>
                      %78 = cannir.extract_stride %77 : <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>> -> <(-1, 16384, 128, 1)>
                      %79 = cannir.extract_dim %78[0] : !cannir.stride<(-1, 16384, 128, 1)>
                      %80 = arith.muli %50, %79 : i64
                      %81 = arith.muli %80, %c4_i64 : i64
                      %82 = arith.muli %52, %c65536_i64 : i64
                      %83 = arith.addi %81, %82 : i64
                      ascvec.get_buf mte2, %c1_i64, 0 {cannir.buffer.generated_sync}
                      "ascvec.copy_gm2ub_plan"(%alloc_5, %alloc, %c0_i32, %c0_i32, %c0_i64, %83, %c1_i64, %c65536_i64, %c0_i64, %c65536_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 65536 : i64, ub_capacity = 65536 : i64}> : (memref<16384xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                      ascvec.release_buf mte2, %c1_i64, 0 {cannir.buffer.generated_sync}
                      %84 = arith.cmpi eq, %60#0, %c0_i64 : i64
                      %85 = ascscalar.memref_address %alloc_6 : memref<128xf16, 1>
                      %86 = ascscalar.memref_address %alloc_7 : memref<128xf16, 1>
                      %87 = arith.select %84, %85, %86 : i64
                      %88 = arith.select %84, %c2_i64, %c3_i64 : i64
                      %89 = arith.addi %60#0, %c1_i64 overflow<nsw> : i64
                      %90 = arith.remui %89, %c2_i64 : i64
                      %91 = cannir.extract_dim %10[2] : !cannir.shape<(-1, -1, -1, 128)>
                      %92 = arith.minsi %52, %91 : i64
                      %93 = arith.subi %91, %92 : i64
                      %94 = arith.minsi %93, %c1_i64 : i64
                      %95 = arith.cmpi sle, %52, %91 : i64
                      %96 = arith.select %95, %94, %c0_i64 : i64
                      %97 = cannir.extract_physical_layout %arg1 : !cannir.tensor<<gm, f16>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                      %98 = cannir.extract_stride %97 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, 128, 1)>
                      %99 = cannir.extract_dim %98[0] : !cannir.stride<(-1, -1, 128, 1)>
                      %100 = arith.muli %50, %99 : i64
                      %101 = arith.muli %100, %c2_i64 : i64
                      %102 = arith.muli %52, %c256_i64 : i64
                      %103 = arith.addi %101, %102 : i64
                      %104 = arith.muli %96, %c256_i64 : i64
                      %105 = arith.cmpi eq, %104, %c0_i64 : i64
                      %106 = arith.cmpi ne, %104, %c0_i64 : i64
                      %107 = arith.extui %106 : i1 to i64
                      %108 = arith.select %105, %c2_i64, %104 : i64
                      %109 = ascscalar.bind_local_ptr %87 : memref<1xf16, 1>
                      ascvec.get_buf mte2, %88, 0 {cannir.buffer.generated_sync}
                      "ascvec.copy_gm2ub_plan"(%109, %alloc_0, %c0_i32, %c0_i32, %c0_i64, %103, %107, %108, %c0_i64, %108, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xf16, 1>, memref<1xf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                      ascvec.release_buf mte2, %88, 0 {cannir.buffer.generated_sync}
                      %110 = arith.cmpi eq, %60#2, %c0_i64 : i64
                      %111 = ascscalar.memref_address %alloc_8 : memref<128xf16, 1>
                      %112 = ascscalar.memref_address %alloc_9 : memref<128xf16, 1>
                      %113 = arith.select %110, %111, %112 : i64
                      %114 = arith.select %110, %c4_i64, %c5_i64 : i64
                      %115 = cannir.extract_layout %arg2 : !cannir.tensor<<gm, f16>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                      %116 = cannir.extract_shape %115 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, -1, 128)>
                      %117 = cannir.extract_dim %116[2] : !cannir.shape<(-1, -1, -1, 128)>
                      %118 = arith.minsi %52, %117 : i64
                      %119 = arith.subi %117, %118 : i64
                      %120 = arith.minsi %119, %c1_i64 : i64
                      %121 = arith.cmpi sle, %52, %117 : i64
                      %122 = arith.select %121, %120, %c0_i64 : i64
                      %123 = cannir.extract_physical_layout %arg2 : !cannir.tensor<<gm, f16>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                      %124 = cannir.extract_stride %123 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, 128, 1)>
                      %125 = cannir.extract_dim %124[0] : !cannir.stride<(-1, -1, 128, 1)>
                      %126 = arith.muli %50, %125 : i64
                      %127 = arith.muli %126, %c2_i64 : i64
                      %128 = arith.addi %127, %102 : i64
                      %129 = arith.muli %122, %c256_i64 : i64
                      %130 = arith.cmpi eq, %129, %c0_i64 : i64
                      %131 = arith.cmpi ne, %129, %c0_i64 : i64
                      %132 = arith.extui %131 : i1 to i64
                      %133 = arith.select %130, %c2_i64, %129 : i64
                      %134 = ascscalar.bind_local_ptr %113 : memref<1xf16, 1>
                      ascvec.get_buf mte2, %114, 0 {cannir.buffer.generated_sync}
                      "ascvec.copy_gm2ub_plan"(%134, %alloc_1, %c0_i32, %c0_i32, %c0_i64, %128, %132, %133, %c0_i64, %133, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xf16, 1>, memref<1xf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                      ascvec.release_buf mte2, %114, 0 {cannir.buffer.generated_sync}
                      %135 = arith.cmpi eq, %60#3, %c0_i64 : i64
                      %136 = ascscalar.memref_address %alloc_10 : memref<128xf32, 1>
                      %137 = ascscalar.memref_address %alloc_11 : memref<128xf32, 1>
                      %138 = arith.select %135, %136, %137 : i64
                      %139 = arith.select %135, %c6_i64, %c7_i64 : i64
                      %140 = cannir.extract_layout %arg3 : !cannir.tensor<<gm, f32>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                      %141 = cannir.extract_shape %140 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, -1, 128)>
                      %142 = cannir.extract_dim %141[2] : !cannir.shape<(-1, -1, -1, 128)>
                      %143 = arith.minsi %52, %142 : i64
                      %144 = arith.subi %142, %143 : i64
                      %145 = arith.minsi %144, %c1_i64 : i64
                      %146 = arith.cmpi sle, %52, %142 : i64
                      %147 = arith.select %146, %145, %c0_i64 : i64
                      %148 = cannir.extract_physical_layout %arg3 : !cannir.tensor<<gm, f32>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                      %149 = cannir.extract_stride %148 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, 128, 1)>
                      %150 = cannir.extract_dim %149[0] : !cannir.stride<(-1, -1, 128, 1)>
                      %151 = arith.muli %50, %150 : i64
                      %152 = arith.muli %151, %c4_i64 : i64
                      %153 = arith.muli %52, %c512_i64 : i64
                      %154 = arith.addi %152, %153 : i64
                      %155 = arith.muli %147, %c512_i64 : i64
                      %156 = arith.cmpi eq, %155, %c0_i64 : i64
                      %157 = arith.cmpi ne, %155, %c0_i64 : i64
                      %158 = arith.extui %157 : i1 to i64
                      %159 = arith.select %156, %c4_i64, %155 : i64
                      %160 = ascscalar.bind_local_ptr %138 : memref<1xf32, 1>
                      ascvec.get_buf mte2, %139, 0 {cannir.buffer.generated_sync}
                      "ascvec.copy_gm2ub_plan"(%160, %alloc_2, %c0_i32, %c0_i32, %c0_i64, %154, %158, %159, %c0_i64, %159, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = -1 : i64, ub_capacity = 4 : i64}> : (memref<1xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                      ascvec.release_buf mte2, %139, 0 {cannir.buffer.generated_sync}
                      scf.yield %90 : i64
                    } else {
                      scf.yield %60#0 : i64
                    }
                    %64:2 = scf.for %arg5 = %c0_i64 to %11 step %c1_i64 iter_args(%arg6 = %63, %arg7 = %60#1) -> (i64, i64)  : i64 {
                      %77 = arith.cmpi slt, %arg5, %58 : i64
                      %78:2 = scf.if %77 -> (i64, i64) {
                        %79 = arith.addi %arg5, %c1_i64 : i64
                        %80 = arith.cmpi slt, %79, %58 : i64
                        %81 = scf.if %80 -> (i64) {
                          %100 = arith.cmpi eq, %arg6, %c0_i64 : i64
                          %101 = ascscalar.memref_address %alloc_6 : memref<128xf16, 1>
                          %102 = ascscalar.memref_address %alloc_7 : memref<128xf16, 1>
                          %103 = arith.select %100, %101, %102 : i64
                          %104 = arith.select %100, %c2_i64, %c3_i64 : i64
                          %105 = arith.addi %arg6, %c1_i64 overflow<nsw> : i64
                          %106 = arith.remui %105, %c2_i64 : i64
                          %107 = cannir.extract_dim %10[2] : !cannir.shape<(-1, -1, -1, 128)>
                          %108 = arith.minsi %52, %107 : i64
                          %109 = arith.subi %107, %108 : i64
                          %110 = arith.minsi %109, %c1_i64 : i64
                          %111 = arith.cmpi sle, %52, %107 : i64
                          %112 = arith.select %111, %110, %c0_i64 : i64
                          %113 = cannir.extract_physical_layout %arg1 : !cannir.tensor<<gm, f16>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                          %114 = cannir.extract_stride %113 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, 128, 1)>
                          %115 = cannir.extract_dim %114[0] : !cannir.stride<(-1, -1, 128, 1)>
                          %116 = arith.muli %50, %115 : i64
                          %117 = arith.muli %116, %c2_i64 : i64
                          %118 = cannir.extract_dim %114[1] : !cannir.stride<(-1, -1, 128, 1)>
                          %119 = arith.muli %79, %118 : i64
                          %120 = arith.muli %119, %c2_i64 : i64
                          %121 = arith.addi %117, %120 : i64
                          %122 = arith.muli %52, %c256_i64 : i64
                          %123 = arith.addi %121, %122 : i64
                          %124 = arith.muli %112, %c256_i64 : i64
                          %125 = arith.cmpi eq, %124, %c0_i64 : i64
                          %126 = arith.cmpi ne, %124, %c0_i64 : i64
                          %127 = arith.extui %126 : i1 to i64
                          %128 = arith.select %125, %c2_i64, %124 : i64
                          %129 = ascscalar.bind_local_ptr %103 : memref<1xf16, 1>
                          ascvec.get_buf mte2, %104, 0 {cannir.buffer.generated_sync}
                          "ascvec.copy_gm2ub_plan"(%129, %alloc_0, %c0_i32, %c0_i32, %c0_i64, %123, %127, %128, %c0_i64, %128, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xf16, 1>, memref<1xf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                          ascvec.release_buf mte2, %104, 0 {cannir.buffer.generated_sync}
                          %130 = ascscalar.memref_address %alloc_8 : memref<128xf16, 1>
                          %131 = ascscalar.memref_address %alloc_9 : memref<128xf16, 1>
                          %132 = arith.select %100, %130, %131 : i64
                          %133 = arith.select %100, %c4_i64, %c5_i64 : i64
                          %134 = cannir.extract_layout %arg2 : !cannir.tensor<<gm, f16>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                          %135 = cannir.extract_shape %134 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, -1, 128)>
                          %136 = cannir.extract_dim %135[2] : !cannir.shape<(-1, -1, -1, 128)>
                          %137 = arith.minsi %52, %136 : i64
                          %138 = arith.subi %136, %137 : i64
                          %139 = arith.minsi %138, %c1_i64 : i64
                          %140 = arith.cmpi sle, %52, %136 : i64
                          %141 = arith.select %140, %139, %c0_i64 : i64
                          %142 = cannir.extract_physical_layout %arg2 : !cannir.tensor<<gm, f16>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                          %143 = cannir.extract_stride %142 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, 128, 1)>
                          %144 = cannir.extract_dim %143[0] : !cannir.stride<(-1, -1, 128, 1)>
                          %145 = arith.muli %50, %144 : i64
                          %146 = arith.muli %145, %c2_i64 : i64
                          %147 = cannir.extract_dim %143[1] : !cannir.stride<(-1, -1, 128, 1)>
                          %148 = arith.muli %79, %147 : i64
                          %149 = arith.muli %148, %c2_i64 : i64
                          %150 = arith.addi %146, %149 : i64
                          %151 = arith.addi %150, %122 : i64
                          %152 = arith.muli %141, %c256_i64 : i64
                          %153 = arith.cmpi eq, %152, %c0_i64 : i64
                          %154 = arith.cmpi ne, %152, %c0_i64 : i64
                          %155 = arith.extui %154 : i1 to i64
                          %156 = arith.select %153, %c2_i64, %152 : i64
                          %157 = ascscalar.bind_local_ptr %132 : memref<1xf16, 1>
                          ascvec.get_buf mte2, %133, 0 {cannir.buffer.generated_sync}
                          "ascvec.copy_gm2ub_plan"(%157, %alloc_1, %c0_i32, %c0_i32, %c0_i64, %151, %155, %156, %c0_i64, %156, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xf16, 1>, memref<1xf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                          ascvec.release_buf mte2, %133, 0 {cannir.buffer.generated_sync}
                          %158 = ascscalar.memref_address %alloc_10 : memref<128xf32, 1>
                          %159 = ascscalar.memref_address %alloc_11 : memref<128xf32, 1>
                          %160 = arith.select %100, %158, %159 : i64
                          %161 = arith.select %100, %c6_i64, %c7_i64 : i64
                          %162 = cannir.extract_layout %arg3 : !cannir.tensor<<gm, f32>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                          %163 = cannir.extract_shape %162 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, -1, 128)>
                          %164 = cannir.extract_dim %163[2] : !cannir.shape<(-1, -1, -1, 128)>
                          %165 = arith.minsi %52, %164 : i64
                          %166 = arith.subi %164, %165 : i64
                          %167 = arith.minsi %166, %c1_i64 : i64
                          %168 = arith.cmpi sle, %52, %164 : i64
                          %169 = arith.select %168, %167, %c0_i64 : i64
                          %170 = cannir.extract_physical_layout %arg3 : !cannir.tensor<<gm, f32>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                          %171 = cannir.extract_stride %170 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, 128, 1)>
                          %172 = cannir.extract_dim %171[0] : !cannir.stride<(-1, -1, 128, 1)>
                          %173 = arith.muli %50, %172 : i64
                          %174 = arith.muli %173, %c4_i64 : i64
                          %175 = cannir.extract_dim %171[1] : !cannir.stride<(-1, -1, 128, 1)>
                          %176 = arith.muli %79, %175 : i64
                          %177 = arith.muli %176, %c4_i64 : i64
                          %178 = arith.addi %174, %177 : i64
                          %179 = arith.muli %52, %c512_i64 : i64
                          %180 = arith.addi %178, %179 : i64
                          %181 = arith.muli %169, %c512_i64 : i64
                          %182 = arith.cmpi eq, %181, %c0_i64 : i64
                          %183 = arith.cmpi ne, %181, %c0_i64 : i64
                          %184 = arith.extui %183 : i1 to i64
                          %185 = arith.select %182, %c4_i64, %181 : i64
                          %186 = ascscalar.bind_local_ptr %160 : memref<1xf32, 1>
                          ascvec.get_buf mte2, %161, 0 {cannir.buffer.generated_sync}
                          "ascvec.copy_gm2ub_plan"(%186, %alloc_2, %c0_i32, %c0_i32, %c0_i64, %180, %184, %185, %c0_i64, %185, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = -1 : i64, ub_capacity = 4 : i64}> : (memref<1xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                          ascvec.release_buf mte2, %161, 0 {cannir.buffer.generated_sync}
                          scf.yield %106 : i64
                        } else {
                          scf.yield %arg6 : i64
                        }
                        %82 = arith.cmpi eq, %arg7, %c0_i64 : i64
                        %83 = ascscalar.memref_address %alloc_6 : memref<128xf16, 1>
                        %84 = ascscalar.memref_address %alloc_7 : memref<128xf16, 1>
                        %85 = arith.select %82, %83, %84 : i64
                        %86 = ascscalar.bind_local_ptr %85 : memref<128xf16, 1>
                        %87 = arith.select %82, %c2_i64, %c3_i64 : i64
                        %88 = arith.addi %arg7, %c1_i64 overflow<nsw> : i64
                        %89 = arith.remui %88, %c2_i64 : i64
                        %90 = ascscalar.memref_address %alloc_8 : memref<128xf16, 1>
                        %91 = ascscalar.memref_address %alloc_9 : memref<128xf16, 1>
                        %92 = arith.select %82, %90, %91 : i64
                        %93 = ascscalar.bind_local_ptr %92 : memref<128xf16, 1>
                        %94 = arith.select %82, %c4_i64, %c5_i64 : i64
                        %95 = ascscalar.memref_address %alloc_10 : memref<128xf32, 1>
                        %96 = ascscalar.memref_address %alloc_11 : memref<128xf32, 1>
                        %97 = arith.select %82, %95, %96 : i64
                        %98 = ascscalar.bind_local_ptr %97 : memref<128xf32, 1>
                        %99 = arith.select %82, %c6_i64, %c7_i64 : i64
                        ascvec.get_buf v, %87, 0 {cannir.buffer.generated_sync}
                        ascvec.vf(%86, %alloc_12 : memref<128xf16, 1>, memref<128xf32, 1>) {
                        ^bb0(%arg8: memref<128xf16, 1>, %arg9: memref<128xf32, 1>):
                          %c1_i32_13 = arith.constant 1 : i32
                          %c2_i32 = arith.constant 2 : i32
                          %c0_i32_14 = arith.constant 0 : i32
                          %c128_i32 = arith.constant 128 : i32
                          %c64_i32 = arith.constant 64 : i32
                          scf.for %arg10 = %c0_i32_14 to %c2_i32 step %c1_i32_13  : i32 {
                            %100 = arith.muli %arg10, %c64_i32 overflow<nsw> : i32
                            %101 = arith.subi %c128_i32, %100 overflow<nsw> : i32
                            %mask, %remaining_out = ascvec.update_mask %101 : !ascvec.mask
                            %102 = arith.index_cast %100 : i32 to index
                            %103 = ascvec.loadalign_unpack %arg8[%102] {mode = "b16_to_b32"} : memref<128xf16, 1>, index -> <f16>
                            %104 = ascvec.cast %103, %mask {rounding = 6 : i32} : <f16> -> <f32>
                            ascvec.storealign %arg9[%102], %104, %mask : memref<128xf32, 1>, index, <f32>
                          }
                        } {cannir.vf_mode = "simd"}
                        ascvec.release_buf v, %87, 0 {cannir.buffer.generated_sync}
                        ascvec.get_buf v, %94, 0 {cannir.buffer.generated_sync}
                        ascvec.get_buf v, %99, 0 {cannir.buffer.generated_sync}
                        ascvec.get_buf v, %c1_i64, 0 {cannir.buffer.generated_sync}
                        ascvec.vf(%93, %98, %alloc_12, %alloc_5 : memref<128xf16, 1>, memref<128xf32, 1>, memref<128xf32, 1>, memref<16384xf32, 1>) {
                        ^bb0(%arg8: memref<128xf16, 1>, %arg9: memref<128xf32, 1>, %arg10: memref<128xf32, 1>, %arg11: memref<16384xf32, 1>):
                          %c64 = arith.constant 64 : index
                          %c1_i32_13 = arith.constant 1 : i32
                          %c0_i32_14 = arith.constant 0 : i32
                          %c128_i32 = arith.constant 128 : i32
                          %c0_15 = arith.constant 0 : index
                          %c64_i32 = arith.constant 64 : i32
                          %mask, %remaining_out = ascvec.update_mask %c64_i32 : !ascvec.mask
                          %100 = ascvec.loadalign_unpack %arg8[%c0_15] {mode = "b16_to_b32"} : memref<128xf16, 1>, index -> <f16>
                          %101 = ascvec.loadalign_unpack %arg8[%c64] {mode = "b16_to_b32"} : memref<128xf16, 1>, index -> <f16>
                          %102 = ascvec.cast %100, %mask {rounding = 6 : i32} : <f16> -> <f32>
                          %103 = ascvec.cast %101, %mask {rounding = 6 : i32} : <f16> -> <f32>
                          %104 = ascvec.loadalign %arg9[%c0_15] : memref<128xf32, 1>, index -> <f32>
                          %105 = ascvec.loadalign %arg9[%c64] : memref<128xf32, 1>, index -> <f32>
                          scf.for %arg12 = %c0_i32_14 to %c128_i32 step %c1_i32_13  : i32 {
                            %106 = arith.muli %arg12, %c128_i32 overflow<nsw> : i32
                            %107 = arith.index_cast %arg12 : i32 to index
                            %108 = ascvec.loadalign_brc %arg10[%107] : memref<128xf32, 1>, index -> <f32>
                            %109 = arith.index_cast %106 : i32 to index
                            %110 = ascvec.loadalign %arg11[%109] : memref<16384xf32, 1>, index -> <f32>
                            %111 = arith.addi %106, %c64_i32 overflow<nsw> : i32
                            %112 = arith.index_cast %111 : i32 to index
                            %113 = ascvec.loadalign %arg11[%112] : memref<16384xf32, 1>, index -> <f32>
                            %114 = ascvec.mul %110, %104, %mask : <f32>
                            %115 = ascvec.mul %113, %105, %mask : <f32>
                            %116 = ascvec.copy %108, %mask : <f32>
                            %117 = ascvec.madd_two_address %116, %102, %114, %mask : <f32>
                            %118 = ascvec.madd_two_address %108, %103, %115, %mask : <f32>
                            ascvec.storealign %arg11[%109], %117, %mask : memref<16384xf32, 1>, index, <f32>
                            ascvec.storealign %arg11[%112], %118, %mask : memref<16384xf32, 1>, index, <f32>
                          } {cannir.loop_unroll = 4 : i64}
                        } {cannir.vf_mode = "simd"}
                        ascvec.release_buf v, %c1_i64, 0 {cannir.buffer.generated_sync}
                        ascvec.release_buf v, %99, 0 {cannir.buffer.generated_sync}
                        ascvec.release_buf v, %94, 0 {cannir.buffer.generated_sync}
                        scf.yield %81, %89 : i64, i64
                      } else {
                        scf.yield %arg6, %arg7 : i64, i64
                      }
                      scf.yield %78#0, %78#1 : i64, i64
                    }
                    %65 = cannir.extract_physical_layout %arg0 : !cannir.tensor<<gm, f32>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, identity>
                    %66 = cannir.extract_stride %65 : <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>> -> <(-1, 16384, 128, 1)>
                    %67 = cannir.extract_dim %66[0] : !cannir.stride<(-1, 16384, 128, 1)>
                    %68 = arith.muli %50, %67 : i64
                    %69 = arith.muli %52, %c16384_i64 : i64
                    %70 = arith.addi %68, %69 : i64
                    %71 = arith.muli %70, %c32_i64 : i64
                    %72 = ascscalar.memref_address %alloc : memref<1xf32>
                    %73 = arith.divui %71, %c8_i64 : i64
                    %74 = arith.addi %72, %73 : i64
                    %75 = ascscalar.bind_gm_ptr %74 : memref<1xf32>
                    %76 = ascscalar.bitcast_memref %alloc_5 byte_offset = %c0_i64 : memref<16384xf32, 1> -> memref<?xf32, 1>
                    ascvec.get_buf mte3, %c1_i64, 0 {cannir.buffer.generated_sync}
                    ascvec.copy_ub2gm %75[%c0], %76[%c0], %c1_i32, %c65536_i32, %c0_i64, %c65536_i32 l2_cache_ctl %c0_i32 : memref<1xf32>, memref<?xf32, 1>
                    ascvec.release_buf mte3, %c1_i64, 0 {cannir.buffer.generated_sync}
                  }
                } else {
                  %27 = arith.subi %12, %25 : i64
                  %28 = arith.subi %13, %25 : i64
                  %29 = arith.muli %25, %c2_i64 : i64
                  %30 = arith.addi %29, %27 : i64
                  %31 = arith.addi %29, %28 : i64
                  %32 = arith.remsi %27, %c2_i64 : i64
                  %33 = arith.cmpi ne, %32, %c0_i64 : i64
                  %34 = arith.cmpi slt, %32, %c0_i64 : i64
                  %35 = arith.andi %33, %34 : i1
                  %36 = arith.divsi %27, %c2_i64 : i64
                  %37 = arith.extui %35 : i1 to i64
                  %38 = arith.subi %36, %37 : i64
                  %39 = arith.addi %31, %38 : i64
                  %40 = arith.select %35, %c2_i64, %c0_i64 : i64
                  %41 = arith.addi %32, %40 overflow<nsw> : i64
                  %42 = arith.maxsi %3, %c1_i64 : i64
                  %43 = arith.remsi %30, %42 : i64
                  %44 = arith.cmpi slt, %43, %c0_i64 : i64
                  %45 = arith.divsi %30, %42 : i64
                  %46 = arith.extui %44 : i1 to i64
                  %47 = arith.subi %45, %46 : i64
                  %48 = arith.maxsi %2, %c1_i64 : i64
                  %49 = arith.remsi %47, %48 : i64
                  %50 = arith.cmpi slt, %49, %c0_i64 : i64
                  %51 = arith.select %50, %48, %c0_i64 : i64
                  %52 = arith.addi %49, %51 : i64
                  %53 = arith.select %44, %42, %c0_i64 : i64
                  %54 = arith.addi %43, %53 : i64
                  %55 = arith.remsi %39, %42 : i64
                  %56 = arith.cmpi slt, %55, %c0_i64 : i64
                  %57 = arith.divsi %39, %42 : i64
                  %58 = arith.extui %56 : i1 to i64
                  %59 = arith.subi %57, %58 : i64
                  %60 = arith.remsi %59, %48 : i64
                  %61 = arith.cmpi slt, %60, %c0_i64 : i64
                  %62 = arith.select %61, %48, %c0_i64 : i64
                  %63 = arith.addi %60, %62 : i64
                  %64 = arith.select %56, %42, %c0_i64 : i64
                  %65 = arith.addi %55, %64 : i64
                  %66 = arith.index_cast %52 : i64 to index
                  %67 = memref.load %alloc_3[%66] : memref<1xi32>
                  %68 = arith.extsi %67 : i32 to i64
                  %69 = arith.index_cast %63 : i64 to index
                  %70 = memref.load %alloc_3[%69] : memref<1xi32>
                  %71 = arith.extsi %70 : i32 to i64
                  %72 = arith.cmpi sgt, %68, %c0_i64 : i64
                  %73:4 = scf.if %72 -> (i64, i64, i64, i64) {
                    %75 = cannir.extract_physical_layout %arg0 : !cannir.tensor<<gm, f32>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, identity>
                    %76 = cannir.extract_stride %75 : <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>> -> <(-1, 16384, 128, 1)>
                    %77 = cannir.extract_dim %76[0] : !cannir.stride<(-1, 16384, 128, 1)>
                    %78 = arith.muli %52, %77 : i64
                    %79 = arith.muli %78, %c4_i64 : i64
                    %80 = arith.muli %54, %c65536_i64 : i64
                    %81 = arith.addi %79, %80 : i64
                    ascvec.get_buf mte2, %c0_i64, 0 {cannir.buffer.generated_sync}
                    "ascvec.copy_gm2ub_plan"(%alloc_4, %alloc, %c0_i32, %c0_i32, %c0_i64, %81, %c1_i64, %c65536_i64, %c0_i64, %c65536_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 0 : i64, ub_capacity = 65536 : i64}> : (memref<16384xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                    ascvec.release_buf mte2, %c0_i64, 0 {cannir.buffer.generated_sync}
                    %82 = cannir.extract_dim %10[2] : !cannir.shape<(-1, -1, -1, 128)>
                    %83 = arith.minsi %54, %82 : i64
                    %84 = arith.subi %82, %83 : i64
                    %85 = arith.minsi %84, %c1_i64 : i64
                    %86 = arith.cmpi sle, %54, %82 : i64
                    %87 = arith.select %86, %85, %c0_i64 : i64
                    %88 = cannir.extract_physical_layout %arg1 : !cannir.tensor<<gm, f16>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                    %89 = cannir.extract_stride %88 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, 128, 1)>
                    %90 = cannir.extract_dim %89[0] : !cannir.stride<(-1, -1, 128, 1)>
                    %91 = arith.muli %52, %90 : i64
                    %92 = arith.muli %91, %c2_i64 : i64
                    %93 = arith.muli %54, %c256_i64 : i64
                    %94 = arith.addi %92, %93 : i64
                    %95 = arith.muli %87, %c256_i64 : i64
                    %96 = arith.cmpi eq, %95, %c0_i64 : i64
                    %97 = arith.cmpi ne, %95, %c0_i64 : i64
                    %98 = arith.extui %97 : i1 to i64
                    %99 = arith.select %96, %c2_i64, %95 : i64
                    %100 = ascscalar.bitcast_memref %alloc_6 byte_offset = %c0_i64 : memref<128xf16, 1> -> memref<?xf16, 1>
                    ascvec.get_buf mte2, %c2_i64, 0 {cannir.buffer.generated_sync}
                    "ascvec.copy_gm2ub_plan"(%100, %alloc_0, %c0_i32, %c0_i32, %c0_i64, %94, %98, %99, %c0_i64, %99, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f16, ub_base = 131072 : i64, ub_capacity = 256 : i64}> : (memref<?xf16, 1>, memref<1xf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                    ascvec.release_buf mte2, %c2_i64, 0 {cannir.buffer.generated_sync}
                    %101 = cannir.extract_layout %arg2 : !cannir.tensor<<gm, f16>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                    %102 = cannir.extract_shape %101 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, -1, 128)>
                    %103 = cannir.extract_dim %102[2] : !cannir.shape<(-1, -1, -1, 128)>
                    %104 = arith.minsi %54, %103 : i64
                    %105 = arith.subi %103, %104 : i64
                    %106 = arith.minsi %105, %c1_i64 : i64
                    %107 = arith.cmpi sle, %54, %103 : i64
                    %108 = arith.select %107, %106, %c0_i64 : i64
                    %109 = cannir.extract_physical_layout %arg2 : !cannir.tensor<<gm, f16>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                    %110 = cannir.extract_stride %109 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, 128, 1)>
                    %111 = cannir.extract_dim %110[0] : !cannir.stride<(-1, -1, 128, 1)>
                    %112 = arith.muli %52, %111 : i64
                    %113 = arith.muli %112, %c2_i64 : i64
                    %114 = arith.addi %113, %93 : i64
                    %115 = arith.muli %108, %c256_i64 : i64
                    %116 = arith.cmpi eq, %115, %c0_i64 : i64
                    %117 = arith.cmpi ne, %115, %c0_i64 : i64
                    %118 = arith.extui %117 : i1 to i64
                    %119 = arith.select %116, %c2_i64, %115 : i64
                    %120 = ascscalar.bitcast_memref %alloc_8 byte_offset = %c0_i64 : memref<128xf16, 1> -> memref<?xf16, 1>
                    ascvec.get_buf mte2, %c4_i64, 0 {cannir.buffer.generated_sync}
                    "ascvec.copy_gm2ub_plan"(%120, %alloc_1, %c0_i32, %c0_i32, %c0_i64, %114, %118, %119, %c0_i64, %119, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f16, ub_base = 131584 : i64, ub_capacity = 256 : i64}> : (memref<?xf16, 1>, memref<1xf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                    ascvec.release_buf mte2, %c4_i64, 0 {cannir.buffer.generated_sync}
                    %121 = cannir.extract_layout %arg3 : !cannir.tensor<<gm, f32>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                    %122 = cannir.extract_shape %121 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, -1, 128)>
                    %123 = cannir.extract_dim %122[2] : !cannir.shape<(-1, -1, -1, 128)>
                    %124 = arith.minsi %54, %123 : i64
                    %125 = arith.subi %123, %124 : i64
                    %126 = arith.minsi %125, %c1_i64 : i64
                    %127 = arith.cmpi sle, %54, %123 : i64
                    %128 = arith.select %127, %126, %c0_i64 : i64
                    %129 = cannir.extract_physical_layout %arg3 : !cannir.tensor<<gm, f32>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                    %130 = cannir.extract_stride %129 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, 128, 1)>
                    %131 = cannir.extract_dim %130[0] : !cannir.stride<(-1, -1, 128, 1)>
                    %132 = arith.muli %52, %131 : i64
                    %133 = arith.muli %132, %c4_i64 : i64
                    %134 = arith.muli %54, %c512_i64 : i64
                    %135 = arith.addi %133, %134 : i64
                    %136 = arith.muli %128, %c512_i64 : i64
                    %137 = arith.cmpi eq, %136, %c0_i64 : i64
                    %138 = arith.cmpi ne, %136, %c0_i64 : i64
                    %139 = arith.extui %138 : i1 to i64
                    %140 = arith.select %137, %c4_i64, %136 : i64
                    %141 = ascscalar.bitcast_memref %alloc_10 byte_offset = %c0_i64 : memref<128xf32, 1> -> memref<?xf32, 1>
                    ascvec.get_buf mte2, %c6_i64, 0 {cannir.buffer.generated_sync}
                    "ascvec.copy_gm2ub_plan"(%141, %alloc_2, %c0_i32, %c0_i32, %c0_i64, %135, %139, %140, %c0_i64, %140, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 132096 : i64, ub_capacity = 512 : i64}> : (memref<?xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                    ascvec.release_buf mte2, %c6_i64, 0 {cannir.buffer.generated_sync}
                    %142:2 = scf.for %arg5 = %c0_i64 to %11 step %c1_i64 iter_args(%arg6 = %c1_i64, %arg7 = %c0_i64) -> (i64, i64)  : i64 {
                      %151 = arith.cmpi slt, %arg5, %68 : i64
                      %152:2 = scf.if %151 -> (i64, i64) {
                        %153 = arith.addi %arg5, %c1_i64 : i64
                        %154 = arith.cmpi slt, %153, %68 : i64
                        %155 = scf.if %154 -> (i64) {
                          %174 = arith.cmpi eq, %arg6, %c0_i64 : i64
                          %175 = ascscalar.memref_address %alloc_6 : memref<128xf16, 1>
                          %176 = ascscalar.memref_address %alloc_7 : memref<128xf16, 1>
                          %177 = arith.select %174, %175, %176 : i64
                          %178 = arith.select %174, %c2_i64, %c3_i64 : i64
                          %179 = arith.addi %arg6, %c1_i64 overflow<nsw> : i64
                          %180 = arith.remui %179, %c2_i64 : i64
                          %181 = cannir.extract_dim %89[1] : !cannir.stride<(-1, -1, 128, 1)>
                          %182 = arith.muli %153, %181 : i64
                          %183 = arith.muli %182, %c2_i64 : i64
                          %184 = arith.addi %92, %183 : i64
                          %185 = arith.addi %184, %93 : i64
                          %186 = ascscalar.bind_local_ptr %177 : memref<1xf16, 1>
                          ascvec.get_buf mte2, %178, 0 {cannir.buffer.generated_sync}
                          "ascvec.copy_gm2ub_plan"(%186, %alloc_0, %c0_i32, %c0_i32, %c0_i64, %185, %98, %99, %c0_i64, %99, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xf16, 1>, memref<1xf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                          ascvec.release_buf mte2, %178, 0 {cannir.buffer.generated_sync}
                          %187 = ascscalar.memref_address %alloc_8 : memref<128xf16, 1>
                          %188 = ascscalar.memref_address %alloc_9 : memref<128xf16, 1>
                          %189 = arith.select %174, %187, %188 : i64
                          %190 = arith.select %174, %c4_i64, %c5_i64 : i64
                          %191 = cannir.extract_dim %110[1] : !cannir.stride<(-1, -1, 128, 1)>
                          %192 = arith.muli %153, %191 : i64
                          %193 = arith.muli %192, %c2_i64 : i64
                          %194 = arith.addi %113, %193 : i64
                          %195 = arith.addi %194, %93 : i64
                          %196 = ascscalar.bind_local_ptr %189 : memref<1xf16, 1>
                          ascvec.get_buf mte2, %190, 0 {cannir.buffer.generated_sync}
                          "ascvec.copy_gm2ub_plan"(%196, %alloc_1, %c0_i32, %c0_i32, %c0_i64, %195, %118, %119, %c0_i64, %119, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xf16, 1>, memref<1xf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                          ascvec.release_buf mte2, %190, 0 {cannir.buffer.generated_sync}
                          %197 = ascscalar.memref_address %alloc_10 : memref<128xf32, 1>
                          %198 = ascscalar.memref_address %alloc_11 : memref<128xf32, 1>
                          %199 = arith.select %174, %197, %198 : i64
                          %200 = arith.select %174, %c6_i64, %c7_i64 : i64
                          %201 = cannir.extract_dim %130[1] : !cannir.stride<(-1, -1, 128, 1)>
                          %202 = arith.muli %153, %201 : i64
                          %203 = arith.muli %202, %c4_i64 : i64
                          %204 = arith.addi %133, %203 : i64
                          %205 = arith.addi %204, %134 : i64
                          %206 = ascscalar.bind_local_ptr %199 : memref<1xf32, 1>
                          ascvec.get_buf mte2, %200, 0 {cannir.buffer.generated_sync}
                          "ascvec.copy_gm2ub_plan"(%206, %alloc_2, %c0_i32, %c0_i32, %c0_i64, %205, %139, %140, %c0_i64, %140, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = -1 : i64, ub_capacity = 4 : i64}> : (memref<1xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                          ascvec.release_buf mte2, %200, 0 {cannir.buffer.generated_sync}
                          scf.yield %180 : i64
                        } else {
                          %174 = arith.cmpi sgt, %71, %c0_i64 : i64
                          %175 = scf.if %174 -> (i64) {
                            %176 = arith.muli %63, %77 : i64
                            %177 = arith.muli %176, %c4_i64 : i64
                            %178 = arith.muli %65, %c65536_i64 : i64
                            %179 = arith.addi %177, %178 : i64
                            %180 = arith.muli %41, %c32768_i64 overflow<nsw> : i64
                            %181 = arith.addi %179, %180 : i64
                            ascvec.get_buf mte2, %c1_i64, 0 {cannir.buffer.generated_sync}
                            "ascvec.copy_gm2ub_plan"(%alloc_5, %alloc, %c0_i32, %c0_i32, %c0_i64, %181, %c1_i64, %c32768_i64, %c0_i64, %c32768_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 65536 : i64, ub_capacity = 65536 : i64}> : (memref<16384xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                            ascvec.release_buf mte2, %c1_i64, 0 {cannir.buffer.generated_sync}
                            %182 = arith.cmpi eq, %arg6, %c0_i64 : i64
                            %183 = ascscalar.memref_address %alloc_6 : memref<128xf16, 1>
                            %184 = ascscalar.memref_address %alloc_7 : memref<128xf16, 1>
                            %185 = arith.select %182, %183, %184 : i64
                            %186 = arith.select %182, %c2_i64, %c3_i64 : i64
                            %187 = arith.addi %arg6, %c1_i64 overflow<nsw> : i64
                            %188 = arith.remui %187, %c2_i64 : i64
                            %189 = arith.minsi %65, %82 : i64
                            %190 = arith.subi %82, %189 : i64
                            %191 = arith.minsi %190, %c1_i64 : i64
                            %192 = arith.cmpi sle, %65, %82 : i64
                            %193 = arith.select %192, %191, %c0_i64 : i64
                            %194 = arith.muli %63, %90 : i64
                            %195 = arith.muli %194, %c2_i64 : i64
                            %196 = arith.muli %65, %c256_i64 : i64
                            %197 = arith.addi %195, %196 : i64
                            %198 = arith.muli %41, %c128_i64 overflow<nsw> : i64
                            %199 = arith.addi %197, %198 : i64
                            %200 = ascscalar.bind_local_ptr %185 : memref<1xf16, 1>
                            ascvec.get_buf mte2, %186, 0 {cannir.buffer.generated_sync}
                            "ascvec.copy_gm2ub_plan"(%200, %alloc_0, %c0_i32, %c0_i32, %c0_i64, %199, %193, %c128_i64, %c256_i64, %c128_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xf16, 1>, memref<1xf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                            ascvec.release_buf mte2, %186, 0 {cannir.buffer.generated_sync}
                            %201 = ascscalar.memref_address %alloc_8 : memref<128xf16, 1>
                            %202 = ascscalar.memref_address %alloc_9 : memref<128xf16, 1>
                            %203 = arith.select %182, %201, %202 : i64
                            %204 = arith.select %182, %c4_i64, %c5_i64 : i64
                            %205 = arith.minsi %65, %103 : i64
                            %206 = arith.subi %103, %205 : i64
                            %207 = arith.minsi %206, %c1_i64 : i64
                            %208 = arith.cmpi sle, %65, %103 : i64
                            %209 = arith.select %208, %207, %c0_i64 : i64
                            %210 = arith.muli %63, %111 : i64
                            %211 = arith.muli %210, %c2_i64 : i64
                            %212 = arith.addi %211, %196 : i64
                            %213 = arith.muli %209, %c256_i64 : i64
                            %214 = arith.cmpi eq, %213, %c0_i64 : i64
                            %215 = arith.cmpi ne, %213, %c0_i64 : i64
                            %216 = arith.extui %215 : i1 to i64
                            %217 = arith.select %214, %c2_i64, %213 : i64
                            %218 = ascscalar.bind_local_ptr %203 : memref<1xf16, 1>
                            ascvec.get_buf mte2, %204, 0 {cannir.buffer.generated_sync}
                            "ascvec.copy_gm2ub_plan"(%218, %alloc_1, %c0_i32, %c0_i32, %c0_i64, %212, %216, %217, %c0_i64, %217, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xf16, 1>, memref<1xf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                            ascvec.release_buf mte2, %204, 0 {cannir.buffer.generated_sync}
                            %219 = ascscalar.memref_address %alloc_10 : memref<128xf32, 1>
                            %220 = ascscalar.memref_address %alloc_11 : memref<128xf32, 1>
                            %221 = arith.select %182, %219, %220 : i64
                            %222 = arith.select %182, %c6_i64, %c7_i64 : i64
                            %223 = arith.minsi %65, %123 : i64
                            %224 = arith.subi %123, %223 : i64
                            %225 = arith.minsi %224, %c1_i64 : i64
                            %226 = arith.cmpi sle, %65, %123 : i64
                            %227 = arith.select %226, %225, %c0_i64 : i64
                            %228 = arith.muli %63, %131 : i64
                            %229 = arith.muli %228, %c4_i64 : i64
                            %230 = arith.muli %65, %c512_i64 : i64
                            %231 = arith.addi %229, %230 : i64
                            %232 = arith.muli %227, %c512_i64 : i64
                            %233 = arith.cmpi eq, %232, %c0_i64 : i64
                            %234 = arith.cmpi ne, %232, %c0_i64 : i64
                            %235 = arith.extui %234 : i1 to i64
                            %236 = arith.select %233, %c4_i64, %232 : i64
                            %237 = ascscalar.bind_local_ptr %221 : memref<1xf32, 1>
                            ascvec.get_buf mte2, %222, 0 {cannir.buffer.generated_sync}
                            "ascvec.copy_gm2ub_plan"(%237, %alloc_2, %c0_i32, %c0_i32, %c0_i64, %231, %235, %236, %c0_i64, %236, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = -1 : i64, ub_capacity = 4 : i64}> : (memref<1xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                            ascvec.release_buf mte2, %222, 0 {cannir.buffer.generated_sync}
                            scf.yield %188 : i64
                          } else {
                            scf.yield %arg6 : i64
                          }
                          scf.yield %175 : i64
                        }
                        %156 = arith.cmpi eq, %arg7, %c0_i64 : i64
                        %157 = ascscalar.memref_address %alloc_6 : memref<128xf16, 1>
                        %158 = ascscalar.memref_address %alloc_7 : memref<128xf16, 1>
                        %159 = arith.select %156, %157, %158 : i64
                        %160 = ascscalar.bind_local_ptr %159 : memref<128xf16, 1>
                        %161 = arith.select %156, %c2_i64, %c3_i64 : i64
                        %162 = arith.addi %arg7, %c1_i64 overflow<nsw> : i64
                        %163 = arith.remui %162, %c2_i64 : i64
                        %164 = ascscalar.memref_address %alloc_8 : memref<128xf16, 1>
                        %165 = ascscalar.memref_address %alloc_9 : memref<128xf16, 1>
                        %166 = arith.select %156, %164, %165 : i64
                        %167 = ascscalar.bind_local_ptr %166 : memref<128xf16, 1>
                        %168 = arith.select %156, %c4_i64, %c5_i64 : i64
                        %169 = ascscalar.memref_address %alloc_10 : memref<128xf32, 1>
                        %170 = ascscalar.memref_address %alloc_11 : memref<128xf32, 1>
                        %171 = arith.select %156, %169, %170 : i64
                        %172 = ascscalar.bind_local_ptr %171 : memref<128xf32, 1>
                        %173 = arith.select %156, %c6_i64, %c7_i64 : i64
                        ascvec.get_buf v, %161, 0 {cannir.buffer.generated_sync}
                        ascvec.vf(%160, %alloc_12 : memref<128xf16, 1>, memref<128xf32, 1>) {
                        ^bb0(%arg8: memref<128xf16, 1>, %arg9: memref<128xf32, 1>):
                          %c1_i32_13 = arith.constant 1 : i32
                          %c2_i32 = arith.constant 2 : i32
                          %c0_i32_14 = arith.constant 0 : i32
                          %c128_i32 = arith.constant 128 : i32
                          %c64_i32 = arith.constant 64 : i32
                          scf.for %arg10 = %c0_i32_14 to %c2_i32 step %c1_i32_13  : i32 {
                            %174 = arith.muli %arg10, %c64_i32 overflow<nsw> : i32
                            %175 = arith.subi %c128_i32, %174 overflow<nsw> : i32
                            %mask, %remaining_out = ascvec.update_mask %175 : !ascvec.mask
                            %176 = arith.index_cast %174 : i32 to index
                            %177 = ascvec.loadalign_unpack %arg8[%176] {mode = "b16_to_b32"} : memref<128xf16, 1>, index -> <f16>
                            %178 = ascvec.cast %177, %mask {rounding = 6 : i32} : <f16> -> <f32>
                            ascvec.storealign %arg9[%176], %178, %mask : memref<128xf32, 1>, index, <f32>
                          }
                        } {cannir.vf_mode = "simd"}
                        ascvec.release_buf v, %161, 0 {cannir.buffer.generated_sync}
                        ascvec.get_buf v, %168, 0 {cannir.buffer.generated_sync}
                        ascvec.get_buf v, %173, 0 {cannir.buffer.generated_sync}
                        ascvec.get_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
                        ascvec.vf(%167, %172, %alloc_12, %alloc_4 : memref<128xf16, 1>, memref<128xf32, 1>, memref<128xf32, 1>, memref<16384xf32, 1>) {
                        ^bb0(%arg8: memref<128xf16, 1>, %arg9: memref<128xf32, 1>, %arg10: memref<128xf32, 1>, %arg11: memref<16384xf32, 1>):
                          %c64 = arith.constant 64 : index
                          %c1_i32_13 = arith.constant 1 : i32
                          %c0_i32_14 = arith.constant 0 : i32
                          %c128_i32 = arith.constant 128 : i32
                          %c0_15 = arith.constant 0 : index
                          %c64_i32 = arith.constant 64 : i32
                          %mask, %remaining_out = ascvec.update_mask %c64_i32 : !ascvec.mask
                          %174 = ascvec.loadalign_unpack %arg8[%c0_15] {mode = "b16_to_b32"} : memref<128xf16, 1>, index -> <f16>
                          %175 = ascvec.loadalign_unpack %arg8[%c64] {mode = "b16_to_b32"} : memref<128xf16, 1>, index -> <f16>
                          %176 = ascvec.cast %174, %mask {rounding = 6 : i32} : <f16> -> <f32>
                          %177 = ascvec.cast %175, %mask {rounding = 6 : i32} : <f16> -> <f32>
                          %178 = ascvec.loadalign %arg9[%c0_15] : memref<128xf32, 1>, index -> <f32>
                          %179 = ascvec.loadalign %arg9[%c64] : memref<128xf32, 1>, index -> <f32>
                          scf.for %arg12 = %c0_i32_14 to %c128_i32 step %c1_i32_13  : i32 {
                            %180 = arith.muli %arg12, %c128_i32 overflow<nsw> : i32
                            %181 = arith.index_cast %arg12 : i32 to index
                            %182 = ascvec.loadalign_brc %arg10[%181] : memref<128xf32, 1>, index -> <f32>
                            %183 = arith.index_cast %180 : i32 to index
                            %184 = ascvec.loadalign %arg11[%183] : memref<16384xf32, 1>, index -> <f32>
                            %185 = arith.addi %180, %c64_i32 overflow<nsw> : i32
                            %186 = arith.index_cast %185 : i32 to index
                            %187 = ascvec.loadalign %arg11[%186] : memref<16384xf32, 1>, index -> <f32>
                            %188 = ascvec.mul %184, %178, %mask : <f32>
                            %189 = ascvec.mul %187, %179, %mask : <f32>
                            %190 = ascvec.copy %182, %mask : <f32>
                            %191 = ascvec.madd_two_address %190, %176, %188, %mask : <f32>
                            %192 = ascvec.madd_two_address %182, %177, %189, %mask : <f32>
                            ascvec.storealign %arg11[%183], %191, %mask : memref<16384xf32, 1>, index, <f32>
                            ascvec.storealign %arg11[%186], %192, %mask : memref<16384xf32, 1>, index, <f32>
                          } {cannir.loop_unroll = 4 : i64}
                        } {cannir.vf_mode = "simd"}
                        ascvec.release_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
                        ascvec.release_buf v, %173, 0 {cannir.buffer.generated_sync}
                        ascvec.release_buf v, %168, 0 {cannir.buffer.generated_sync}
                        scf.yield %155, %163 : i64, i64
                      } else {
                        scf.yield %arg6, %arg7 : i64, i64
                      }
                      scf.yield %152#0, %152#1 : i64, i64
                    }
                    %143 = arith.muli %54, %c16384_i64 : i64
                    %144 = arith.addi %78, %143 : i64
                    %145 = arith.muli %144, %c32_i64 : i64
                    %146 = ascscalar.memref_address %alloc : memref<1xf32>
                    %147 = arith.divui %145, %c8_i64 : i64
                    %148 = arith.addi %146, %147 : i64
                    %149 = ascscalar.bind_gm_ptr %148 : memref<1xf32>
                    %150 = ascscalar.bitcast_memref %alloc_4 byte_offset = %c0_i64 : memref<16384xf32, 1> -> memref<?xf32, 1>
                    ascvec.get_buf mte3, %c0_i64, 0 {cannir.buffer.generated_sync}
                    ascvec.copy_ub2gm %149[%c0], %150[%c0], %c1_i32, %c65536_i32, %c0_i64, %c65536_i32 l2_cache_ctl %c0_i32 : memref<1xf32>, memref<?xf32, 1>
                    ascvec.release_buf mte3, %c0_i64, 0 {cannir.buffer.generated_sync}
                    scf.yield %142#0, %142#1, %142#0, %142#0 : i64, i64, i64, i64
                  } else {
                    scf.yield %c0_i64, %c0_i64, %c0_i64, %c0_i64 : i64, i64, i64, i64
                  }
                  %74 = arith.cmpi sgt, %71, %c0_i64 : i64
                  scf.if %74 {
                    %75 = arith.cmpi eq, %68, %c0_i64 : i64
                    %76 = scf.if %75 -> (i64) {
                      %92 = cannir.extract_physical_layout %arg0 : !cannir.tensor<<gm, f32>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, identity>
                      %93 = cannir.extract_stride %92 : <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>> -> <(-1, 16384, 128, 1)>
                      %94 = cannir.extract_dim %93[0] : !cannir.stride<(-1, 16384, 128, 1)>
                      %95 = arith.muli %63, %94 : i64
                      %96 = arith.muli %95, %c4_i64 : i64
                      %97 = arith.muli %65, %c65536_i64 : i64
                      %98 = arith.addi %96, %97 : i64
                      %99 = arith.muli %41, %c32768_i64 overflow<nsw> : i64
                      %100 = arith.addi %98, %99 : i64
                      ascvec.get_buf mte2, %c1_i64, 0 {cannir.buffer.generated_sync}
                      "ascvec.copy_gm2ub_plan"(%alloc_5, %alloc, %c0_i32, %c0_i32, %c0_i64, %100, %c1_i64, %c32768_i64, %c0_i64, %c32768_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 65536 : i64, ub_capacity = 65536 : i64}> : (memref<16384xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                      ascvec.release_buf mte2, %c1_i64, 0 {cannir.buffer.generated_sync}
                      %101 = arith.cmpi eq, %73#0, %c0_i64 : i64
                      %102 = ascscalar.memref_address %alloc_6 : memref<128xf16, 1>
                      %103 = ascscalar.memref_address %alloc_7 : memref<128xf16, 1>
                      %104 = arith.select %101, %102, %103 : i64
                      %105 = arith.select %101, %c2_i64, %c3_i64 : i64
                      %106 = arith.addi %73#0, %c1_i64 overflow<nsw> : i64
                      %107 = arith.remui %106, %c2_i64 : i64
                      %108 = cannir.extract_dim %10[2] : !cannir.shape<(-1, -1, -1, 128)>
                      %109 = arith.minsi %65, %108 : i64
                      %110 = arith.subi %108, %109 : i64
                      %111 = arith.minsi %110, %c1_i64 : i64
                      %112 = arith.cmpi sle, %65, %108 : i64
                      %113 = arith.select %112, %111, %c0_i64 : i64
                      %114 = cannir.extract_physical_layout %arg1 : !cannir.tensor<<gm, f16>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                      %115 = cannir.extract_stride %114 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, 128, 1)>
                      %116 = cannir.extract_dim %115[0] : !cannir.stride<(-1, -1, 128, 1)>
                      %117 = arith.muli %63, %116 : i64
                      %118 = arith.muli %117, %c2_i64 : i64
                      %119 = arith.muli %65, %c256_i64 : i64
                      %120 = arith.addi %118, %119 : i64
                      %121 = arith.muli %41, %c128_i64 overflow<nsw> : i64
                      %122 = arith.addi %120, %121 : i64
                      %123 = ascscalar.bind_local_ptr %104 : memref<1xf16, 1>
                      ascvec.get_buf mte2, %105, 0 {cannir.buffer.generated_sync}
                      "ascvec.copy_gm2ub_plan"(%123, %alloc_0, %c0_i32, %c0_i32, %c0_i64, %122, %113, %c128_i64, %c256_i64, %c128_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xf16, 1>, memref<1xf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                      ascvec.release_buf mte2, %105, 0 {cannir.buffer.generated_sync}
                      %124 = arith.cmpi eq, %73#2, %c0_i64 : i64
                      %125 = ascscalar.memref_address %alloc_8 : memref<128xf16, 1>
                      %126 = ascscalar.memref_address %alloc_9 : memref<128xf16, 1>
                      %127 = arith.select %124, %125, %126 : i64
                      %128 = arith.select %124, %c4_i64, %c5_i64 : i64
                      %129 = cannir.extract_layout %arg2 : !cannir.tensor<<gm, f16>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                      %130 = cannir.extract_shape %129 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, -1, 128)>
                      %131 = cannir.extract_dim %130[2] : !cannir.shape<(-1, -1, -1, 128)>
                      %132 = arith.minsi %65, %131 : i64
                      %133 = arith.subi %131, %132 : i64
                      %134 = arith.minsi %133, %c1_i64 : i64
                      %135 = arith.cmpi sle, %65, %131 : i64
                      %136 = arith.select %135, %134, %c0_i64 : i64
                      %137 = cannir.extract_physical_layout %arg2 : !cannir.tensor<<gm, f16>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                      %138 = cannir.extract_stride %137 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, 128, 1)>
                      %139 = cannir.extract_dim %138[0] : !cannir.stride<(-1, -1, 128, 1)>
                      %140 = arith.muli %63, %139 : i64
                      %141 = arith.muli %140, %c2_i64 : i64
                      %142 = arith.addi %141, %119 : i64
                      %143 = arith.muli %136, %c256_i64 : i64
                      %144 = arith.cmpi eq, %143, %c0_i64 : i64
                      %145 = arith.cmpi ne, %143, %c0_i64 : i64
                      %146 = arith.extui %145 : i1 to i64
                      %147 = arith.select %144, %c2_i64, %143 : i64
                      %148 = ascscalar.bind_local_ptr %127 : memref<1xf16, 1>
                      ascvec.get_buf mte2, %128, 0 {cannir.buffer.generated_sync}
                      "ascvec.copy_gm2ub_plan"(%148, %alloc_1, %c0_i32, %c0_i32, %c0_i64, %142, %146, %147, %c0_i64, %147, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xf16, 1>, memref<1xf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                      ascvec.release_buf mte2, %128, 0 {cannir.buffer.generated_sync}
                      %149 = arith.cmpi eq, %73#3, %c0_i64 : i64
                      %150 = ascscalar.memref_address %alloc_10 : memref<128xf32, 1>
                      %151 = ascscalar.memref_address %alloc_11 : memref<128xf32, 1>
                      %152 = arith.select %149, %150, %151 : i64
                      %153 = arith.select %149, %c6_i64, %c7_i64 : i64
                      %154 = cannir.extract_layout %arg3 : !cannir.tensor<<gm, f32>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                      %155 = cannir.extract_shape %154 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, -1, 128)>
                      %156 = cannir.extract_dim %155[2] : !cannir.shape<(-1, -1, -1, 128)>
                      %157 = arith.minsi %65, %156 : i64
                      %158 = arith.subi %156, %157 : i64
                      %159 = arith.minsi %158, %c1_i64 : i64
                      %160 = arith.cmpi sle, %65, %156 : i64
                      %161 = arith.select %160, %159, %c0_i64 : i64
                      %162 = cannir.extract_physical_layout %arg3 : !cannir.tensor<<gm, f32>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                      %163 = cannir.extract_stride %162 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, 128, 1)>
                      %164 = cannir.extract_dim %163[0] : !cannir.stride<(-1, -1, 128, 1)>
                      %165 = arith.muli %63, %164 : i64
                      %166 = arith.muli %165, %c4_i64 : i64
                      %167 = arith.muli %65, %c512_i64 : i64
                      %168 = arith.addi %166, %167 : i64
                      %169 = arith.muli %161, %c512_i64 : i64
                      %170 = arith.cmpi eq, %169, %c0_i64 : i64
                      %171 = arith.cmpi ne, %169, %c0_i64 : i64
                      %172 = arith.extui %171 : i1 to i64
                      %173 = arith.select %170, %c4_i64, %169 : i64
                      %174 = ascscalar.bind_local_ptr %152 : memref<1xf32, 1>
                      ascvec.get_buf mte2, %153, 0 {cannir.buffer.generated_sync}
                      "ascvec.copy_gm2ub_plan"(%174, %alloc_2, %c0_i32, %c0_i32, %c0_i64, %168, %172, %173, %c0_i64, %173, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = -1 : i64, ub_capacity = 4 : i64}> : (memref<1xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                      ascvec.release_buf mte2, %153, 0 {cannir.buffer.generated_sync}
                      scf.yield %107 : i64
                    } else {
                      scf.yield %73#0 : i64
                    }
                    %77:2 = scf.for %arg5 = %c0_i64 to %11 step %c1_i64 iter_args(%arg6 = %76, %arg7 = %73#1) -> (i64, i64)  : i64 {
                      %92 = arith.cmpi slt, %arg5, %71 : i64
                      %93:2 = scf.if %92 -> (i64, i64) {
                        %94 = arith.addi %arg5, %c1_i64 : i64
                        %95 = arith.cmpi slt, %94, %71 : i64
                        %96 = scf.if %95 -> (i64) {
                          %115 = arith.cmpi eq, %arg6, %c0_i64 : i64
                          %116 = ascscalar.memref_address %alloc_6 : memref<128xf16, 1>
                          %117 = ascscalar.memref_address %alloc_7 : memref<128xf16, 1>
                          %118 = arith.select %115, %116, %117 : i64
                          %119 = arith.select %115, %c2_i64, %c3_i64 : i64
                          %120 = arith.addi %arg6, %c1_i64 overflow<nsw> : i64
                          %121 = arith.remui %120, %c2_i64 : i64
                          %122 = cannir.extract_dim %10[2] : !cannir.shape<(-1, -1, -1, 128)>
                          %123 = arith.minsi %65, %122 : i64
                          %124 = arith.subi %122, %123 : i64
                          %125 = arith.minsi %124, %c1_i64 : i64
                          %126 = arith.cmpi sle, %65, %122 : i64
                          %127 = arith.select %126, %125, %c0_i64 : i64
                          %128 = cannir.extract_physical_layout %arg1 : !cannir.tensor<<gm, f16>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                          %129 = cannir.extract_stride %128 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, 128, 1)>
                          %130 = cannir.extract_dim %129[0] : !cannir.stride<(-1, -1, 128, 1)>
                          %131 = arith.muli %63, %130 : i64
                          %132 = arith.muli %131, %c2_i64 : i64
                          %133 = cannir.extract_dim %129[1] : !cannir.stride<(-1, -1, 128, 1)>
                          %134 = arith.muli %94, %133 : i64
                          %135 = arith.muli %134, %c2_i64 : i64
                          %136 = arith.addi %132, %135 : i64
                          %137 = arith.muli %65, %c256_i64 : i64
                          %138 = arith.addi %136, %137 : i64
                          %139 = arith.muli %41, %c128_i64 overflow<nsw> : i64
                          %140 = arith.addi %138, %139 : i64
                          %141 = ascscalar.bind_local_ptr %118 : memref<1xf16, 1>
                          ascvec.get_buf mte2, %119, 0 {cannir.buffer.generated_sync}
                          "ascvec.copy_gm2ub_plan"(%141, %alloc_0, %c0_i32, %c0_i32, %c0_i64, %140, %127, %c128_i64, %c256_i64, %c128_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xf16, 1>, memref<1xf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                          ascvec.release_buf mte2, %119, 0 {cannir.buffer.generated_sync}
                          %142 = ascscalar.memref_address %alloc_8 : memref<128xf16, 1>
                          %143 = ascscalar.memref_address %alloc_9 : memref<128xf16, 1>
                          %144 = arith.select %115, %142, %143 : i64
                          %145 = arith.select %115, %c4_i64, %c5_i64 : i64
                          %146 = cannir.extract_layout %arg2 : !cannir.tensor<<gm, f16>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                          %147 = cannir.extract_shape %146 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, -1, 128)>
                          %148 = cannir.extract_dim %147[2] : !cannir.shape<(-1, -1, -1, 128)>
                          %149 = arith.minsi %65, %148 : i64
                          %150 = arith.subi %148, %149 : i64
                          %151 = arith.minsi %150, %c1_i64 : i64
                          %152 = arith.cmpi sle, %65, %148 : i64
                          %153 = arith.select %152, %151, %c0_i64 : i64
                          %154 = cannir.extract_physical_layout %arg2 : !cannir.tensor<<gm, f16>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                          %155 = cannir.extract_stride %154 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, 128, 1)>
                          %156 = cannir.extract_dim %155[0] : !cannir.stride<(-1, -1, 128, 1)>
                          %157 = arith.muli %63, %156 : i64
                          %158 = arith.muli %157, %c2_i64 : i64
                          %159 = cannir.extract_dim %155[1] : !cannir.stride<(-1, -1, 128, 1)>
                          %160 = arith.muli %94, %159 : i64
                          %161 = arith.muli %160, %c2_i64 : i64
                          %162 = arith.addi %158, %161 : i64
                          %163 = arith.addi %162, %137 : i64
                          %164 = arith.muli %153, %c256_i64 : i64
                          %165 = arith.cmpi eq, %164, %c0_i64 : i64
                          %166 = arith.cmpi ne, %164, %c0_i64 : i64
                          %167 = arith.extui %166 : i1 to i64
                          %168 = arith.select %165, %c2_i64, %164 : i64
                          %169 = ascscalar.bind_local_ptr %144 : memref<1xf16, 1>
                          ascvec.get_buf mte2, %145, 0 {cannir.buffer.generated_sync}
                          "ascvec.copy_gm2ub_plan"(%169, %alloc_1, %c0_i32, %c0_i32, %c0_i64, %163, %167, %168, %c0_i64, %168, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xf16, 1>, memref<1xf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                          ascvec.release_buf mte2, %145, 0 {cannir.buffer.generated_sync}
                          %170 = ascscalar.memref_address %alloc_10 : memref<128xf32, 1>
                          %171 = ascscalar.memref_address %alloc_11 : memref<128xf32, 1>
                          %172 = arith.select %115, %170, %171 : i64
                          %173 = arith.select %115, %c6_i64, %c7_i64 : i64
                          %174 = cannir.extract_layout %arg3 : !cannir.tensor<<gm, f32>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                          %175 = cannir.extract_shape %174 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, -1, 128)>
                          %176 = cannir.extract_dim %175[2] : !cannir.shape<(-1, -1, -1, 128)>
                          %177 = arith.minsi %65, %176 : i64
                          %178 = arith.subi %176, %177 : i64
                          %179 = arith.minsi %178, %c1_i64 : i64
                          %180 = arith.cmpi sle, %65, %176 : i64
                          %181 = arith.select %180, %179, %c0_i64 : i64
                          %182 = cannir.extract_physical_layout %arg3 : !cannir.tensor<<gm, f32>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                          %183 = cannir.extract_stride %182 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, 128, 1)>
                          %184 = cannir.extract_dim %183[0] : !cannir.stride<(-1, -1, 128, 1)>
                          %185 = arith.muli %63, %184 : i64
                          %186 = arith.muli %185, %c4_i64 : i64
                          %187 = cannir.extract_dim %183[1] : !cannir.stride<(-1, -1, 128, 1)>
                          %188 = arith.muli %94, %187 : i64
                          %189 = arith.muli %188, %c4_i64 : i64
                          %190 = arith.addi %186, %189 : i64
                          %191 = arith.muli %65, %c512_i64 : i64
                          %192 = arith.addi %190, %191 : i64
                          %193 = arith.muli %181, %c512_i64 : i64
                          %194 = arith.cmpi eq, %193, %c0_i64 : i64
                          %195 = arith.cmpi ne, %193, %c0_i64 : i64
                          %196 = arith.extui %195 : i1 to i64
                          %197 = arith.select %194, %c4_i64, %193 : i64
                          %198 = ascscalar.bind_local_ptr %172 : memref<1xf32, 1>
                          ascvec.get_buf mte2, %173, 0 {cannir.buffer.generated_sync}
                          "ascvec.copy_gm2ub_plan"(%198, %alloc_2, %c0_i32, %c0_i32, %c0_i64, %192, %196, %197, %c0_i64, %197, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = -1 : i64, ub_capacity = 4 : i64}> : (memref<1xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                          ascvec.release_buf mte2, %173, 0 {cannir.buffer.generated_sync}
                          scf.yield %121 : i64
                        } else {
                          scf.yield %arg6 : i64
                        }
                        %97 = arith.cmpi eq, %arg7, %c0_i64 : i64
                        %98 = ascscalar.memref_address %alloc_6 : memref<128xf16, 1>
                        %99 = ascscalar.memref_address %alloc_7 : memref<128xf16, 1>
                        %100 = arith.select %97, %98, %99 : i64
                        %101 = ascscalar.bind_local_ptr %100 : memref<128xf16, 1>
                        %102 = arith.select %97, %c2_i64, %c3_i64 : i64
                        %103 = arith.addi %arg7, %c1_i64 overflow<nsw> : i64
                        %104 = arith.remui %103, %c2_i64 : i64
                        %105 = ascscalar.memref_address %alloc_8 : memref<128xf16, 1>
                        %106 = ascscalar.memref_address %alloc_9 : memref<128xf16, 1>
                        %107 = arith.select %97, %105, %106 : i64
                        %108 = ascscalar.bind_local_ptr %107 : memref<128xf16, 1>
                        %109 = arith.select %97, %c4_i64, %c5_i64 : i64
                        %110 = ascscalar.memref_address %alloc_10 : memref<128xf32, 1>
                        %111 = ascscalar.memref_address %alloc_11 : memref<128xf32, 1>
                        %112 = arith.select %97, %110, %111 : i64
                        %113 = ascscalar.bind_local_ptr %112 : memref<128xf32, 1>
                        %114 = arith.select %97, %c6_i64, %c7_i64 : i64
                        ascvec.get_buf v, %102, 0 {cannir.buffer.generated_sync}
                        ascvec.vf(%101, %alloc_12 : memref<128xf16, 1>, memref<128xf32, 1>) {
                        ^bb0(%arg8: memref<128xf16, 1>, %arg9: memref<128xf32, 1>):
                          %c0_13 = arith.constant 0 : index
                          %c64_i32 = arith.constant 64 : i32
                          %c0_14 = arith.constant 0 : index
                          %c1 = arith.constant 1 : index
                          scf.for %arg10 = %c0_14 to %c1 step %c1 {
                            %mask, %remaining_out = ascvec.update_mask %c64_i32 : !ascvec.mask
                            %115 = ascvec.loadalign_unpack %arg8[%c0_13] {mode = "b16_to_b32"} : memref<128xf16, 1>, index -> <f16>
                            %116 = ascvec.cast %115, %mask {rounding = 6 : i32} : <f16> -> <f32>
                            ascvec.storealign %arg9[%c0_13], %116, %mask : memref<128xf32, 1>, index, <f32>
                          }
                        } {cannir.vf_mode = "simd"}
                        ascvec.release_buf v, %102, 0 {cannir.buffer.generated_sync}
                        ascvec.get_buf v, %109, 0 {cannir.buffer.generated_sync}
                        ascvec.get_buf v, %114, 0 {cannir.buffer.generated_sync}
                        ascvec.get_buf v, %c1_i64, 0 {cannir.buffer.generated_sync}
                        ascvec.vf(%108, %113, %alloc_12, %alloc_5 : memref<128xf16, 1>, memref<128xf32, 1>, memref<128xf32, 1>, memref<16384xf32, 1>) {
                        ^bb0(%arg8: memref<128xf16, 1>, %arg9: memref<128xf32, 1>, %arg10: memref<128xf32, 1>, %arg11: memref<16384xf32, 1>):
                          %c64 = arith.constant 64 : index
                          %c1_i32_13 = arith.constant 1 : i32
                          %c0_i32_14 = arith.constant 0 : i32
                          %c128_i32 = arith.constant 128 : i32
                          %c0_15 = arith.constant 0 : index
                          %c64_i32 = arith.constant 64 : i32
                          %mask, %remaining_out = ascvec.update_mask %c64_i32 : !ascvec.mask
                          %115 = ascvec.loadalign_unpack %arg8[%c0_15] {mode = "b16_to_b32"} : memref<128xf16, 1>, index -> <f16>
                          %116 = ascvec.loadalign_unpack %arg8[%c64] {mode = "b16_to_b32"} : memref<128xf16, 1>, index -> <f16>
                          %117 = ascvec.cast %115, %mask {rounding = 6 : i32} : <f16> -> <f32>
                          %118 = ascvec.cast %116, %mask {rounding = 6 : i32} : <f16> -> <f32>
                          %119 = ascvec.loadalign %arg9[%c0_15] : memref<128xf32, 1>, index -> <f32>
                          %120 = ascvec.loadalign %arg9[%c64] : memref<128xf32, 1>, index -> <f32>
                          scf.for %arg12 = %c0_i32_14 to %c64_i32 step %c1_i32_13  : i32 {
                            %121 = arith.muli %arg12, %c128_i32 overflow<nsw> : i32
                            %122 = arith.index_cast %arg12 : i32 to index
                            %123 = ascvec.loadalign_brc %arg10[%122] : memref<128xf32, 1>, index -> <f32>
                            %124 = arith.index_cast %121 : i32 to index
                            %125 = ascvec.loadalign %arg11[%124] : memref<16384xf32, 1>, index -> <f32>
                            %126 = arith.addi %121, %c64_i32 overflow<nsw> : i32
                            %127 = arith.index_cast %126 : i32 to index
                            %128 = ascvec.loadalign %arg11[%127] : memref<16384xf32, 1>, index -> <f32>
                            %129 = ascvec.mul %125, %119, %mask : <f32>
                            %130 = ascvec.mul %128, %120, %mask : <f32>
                            %131 = ascvec.copy %123, %mask : <f32>
                            %132 = ascvec.madd_two_address %131, %117, %129, %mask : <f32>
                            %133 = ascvec.madd_two_address %123, %118, %130, %mask : <f32>
                            ascvec.storealign %arg11[%124], %132, %mask : memref<16384xf32, 1>, index, <f32>
                            ascvec.storealign %arg11[%127], %133, %mask : memref<16384xf32, 1>, index, <f32>
                          } {cannir.loop_unroll = 4 : i64}
                        } {cannir.vf_mode = "simd"}
                        ascvec.release_buf v, %c1_i64, 0 {cannir.buffer.generated_sync}
                        ascvec.release_buf v, %114, 0 {cannir.buffer.generated_sync}
                        ascvec.release_buf v, %109, 0 {cannir.buffer.generated_sync}
                        scf.yield %96, %104 : i64, i64
                      } else {
                        scf.yield %arg6, %arg7 : i64, i64
                      }
                      scf.yield %93#0, %93#1 : i64, i64
                    }
                    %78 = cannir.extract_physical_layout %arg0 : !cannir.tensor<<gm, f32>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, identity>
                    %79 = cannir.extract_stride %78 : <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>> -> <(-1, 16384, 128, 1)>
                    %80 = cannir.extract_dim %79[0] : !cannir.stride<(-1, 16384, 128, 1)>
                    %81 = arith.muli %63, %80 : i64
                    %82 = arith.muli %65, %c16384_i64 : i64
                    %83 = arith.addi %81, %82 : i64
                    %84 = arith.muli %41, %c8192_i64 overflow<nsw> : i64
                    %85 = arith.addi %83, %84 : i64
                    %86 = arith.muli %85, %c32_i64 : i64
                    %87 = ascscalar.memref_address %alloc : memref<1xf32>
                    %88 = arith.divui %86, %c8_i64 : i64
                    %89 = arith.addi %87, %88 : i64
                    %90 = ascscalar.bind_gm_ptr %89 : memref<1xf32>
                    %91 = ascscalar.bitcast_memref %alloc_5 byte_offset = %c0_i64 : memref<16384xf32, 1> -> memref<?xf32, 1>
                    ascvec.get_buf mte3, %c1_i64, 0 {cannir.buffer.generated_sync}
                    ascvec.copy_ub2gm %90[%c0], %91[%c0], %c1_i32, %c32768_i32, %c0_i64, %c32768_i32 l2_cache_ctl %c0_i32 : memref<1xf32>, memref<?xf32, 1>
                    ascvec.release_buf mte3, %c1_i64, 0 {cannir.buffer.generated_sync}
                  }
                }
              } else {
                %25 = arith.subi %4, %13 : i64
                %26 = arith.cmpi slt, %12, %25 : i64
                scf.if %26 {
                  %27 = arith.muli %12, %c2_i64 : i64
                  %28 = arith.addi %27, %c1_i64 : i64
                  %29 = arith.maxsi %3, %c1_i64 : i64
                  %30 = arith.remsi %27, %29 : i64
                  %31 = arith.cmpi slt, %30, %c0_i64 : i64
                  %32 = arith.divsi %27, %29 : i64
                  %33 = arith.extui %31 : i1 to i64
                  %34 = arith.subi %32, %33 : i64
                  %35 = arith.maxsi %2, %c1_i64 : i64
                  %36 = arith.remsi %34, %35 : i64
                  %37 = arith.cmpi slt, %36, %c0_i64 : i64
                  %38 = arith.select %37, %35, %c0_i64 : i64
                  %39 = arith.addi %36, %38 : i64
                  %40 = arith.select %31, %29, %c0_i64 : i64
                  %41 = arith.addi %30, %40 : i64
                  %42 = arith.remsi %28, %29 : i64
                  %43 = arith.cmpi slt, %42, %c0_i64 : i64
                  %44 = arith.divsi %28, %29 : i64
                  %45 = arith.extui %43 : i1 to i64
                  %46 = arith.subi %44, %45 : i64
                  %47 = arith.remsi %46, %35 : i64
                  %48 = arith.cmpi slt, %47, %c0_i64 : i64
                  %49 = arith.select %48, %35, %c0_i64 : i64
                  %50 = arith.addi %47, %49 : i64
                  %51 = arith.select %43, %29, %c0_i64 : i64
                  %52 = arith.addi %42, %51 : i64
                  %53 = arith.index_cast %39 : i64 to index
                  %54 = memref.load %alloc_3[%53] : memref<1xi32>
                  %55 = arith.extsi %54 : i32 to i64
                  %56 = arith.index_cast %50 : i64 to index
                  %57 = memref.load %alloc_3[%56] : memref<1xi32>
                  %58 = arith.extsi %57 : i32 to i64
                  %59 = arith.cmpi sgt, %55, %c0_i64 : i64
                  %60:4 = scf.if %59 -> (i64, i64, i64, i64) {
                    %62 = cannir.extract_physical_layout %arg0 : !cannir.tensor<<gm, f32>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, identity>
                    %63 = cannir.extract_stride %62 : <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>> -> <(-1, 16384, 128, 1)>
                    %64 = cannir.extract_dim %63[0] : !cannir.stride<(-1, 16384, 128, 1)>
                    %65 = arith.muli %39, %64 : i64
                    %66 = arith.muli %65, %c4_i64 : i64
                    %67 = arith.muli %41, %c65536_i64 : i64
                    %68 = arith.addi %66, %67 : i64
                    ascvec.get_buf mte2, %c0_i64, 0 {cannir.buffer.generated_sync}
                    "ascvec.copy_gm2ub_plan"(%alloc_4, %alloc, %c0_i32, %c0_i32, %c0_i64, %68, %c1_i64, %c65536_i64, %c0_i64, %c65536_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 0 : i64, ub_capacity = 65536 : i64}> : (memref<16384xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                    ascvec.release_buf mte2, %c0_i64, 0 {cannir.buffer.generated_sync}
                    %69 = cannir.extract_dim %10[2] : !cannir.shape<(-1, -1, -1, 128)>
                    %70 = arith.minsi %41, %69 : i64
                    %71 = arith.subi %69, %70 : i64
                    %72 = arith.minsi %71, %c1_i64 : i64
                    %73 = arith.cmpi sle, %41, %69 : i64
                    %74 = arith.select %73, %72, %c0_i64 : i64
                    %75 = cannir.extract_physical_layout %arg1 : !cannir.tensor<<gm, f16>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                    %76 = cannir.extract_stride %75 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, 128, 1)>
                    %77 = cannir.extract_dim %76[0] : !cannir.stride<(-1, -1, 128, 1)>
                    %78 = arith.muli %39, %77 : i64
                    %79 = arith.muli %78, %c2_i64 : i64
                    %80 = arith.muli %41, %c256_i64 : i64
                    %81 = arith.addi %79, %80 : i64
                    %82 = arith.muli %74, %c256_i64 : i64
                    %83 = arith.cmpi eq, %82, %c0_i64 : i64
                    %84 = arith.cmpi ne, %82, %c0_i64 : i64
                    %85 = arith.extui %84 : i1 to i64
                    %86 = arith.select %83, %c2_i64, %82 : i64
                    %87 = ascscalar.bitcast_memref %alloc_6 byte_offset = %c0_i64 : memref<128xf16, 1> -> memref<?xf16, 1>
                    ascvec.get_buf mte2, %c2_i64, 0 {cannir.buffer.generated_sync}
                    "ascvec.copy_gm2ub_plan"(%87, %alloc_0, %c0_i32, %c0_i32, %c0_i64, %81, %85, %86, %c0_i64, %86, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f16, ub_base = 131072 : i64, ub_capacity = 256 : i64}> : (memref<?xf16, 1>, memref<1xf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                    ascvec.release_buf mte2, %c2_i64, 0 {cannir.buffer.generated_sync}
                    %88 = cannir.extract_layout %arg2 : !cannir.tensor<<gm, f16>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                    %89 = cannir.extract_shape %88 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, -1, 128)>
                    %90 = cannir.extract_dim %89[2] : !cannir.shape<(-1, -1, -1, 128)>
                    %91 = arith.minsi %41, %90 : i64
                    %92 = arith.subi %90, %91 : i64
                    %93 = arith.minsi %92, %c1_i64 : i64
                    %94 = arith.cmpi sle, %41, %90 : i64
                    %95 = arith.select %94, %93, %c0_i64 : i64
                    %96 = cannir.extract_physical_layout %arg2 : !cannir.tensor<<gm, f16>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                    %97 = cannir.extract_stride %96 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, 128, 1)>
                    %98 = cannir.extract_dim %97[0] : !cannir.stride<(-1, -1, 128, 1)>
                    %99 = arith.muli %39, %98 : i64
                    %100 = arith.muli %99, %c2_i64 : i64
                    %101 = arith.addi %100, %80 : i64
                    %102 = arith.muli %95, %c256_i64 : i64
                    %103 = arith.cmpi eq, %102, %c0_i64 : i64
                    %104 = arith.cmpi ne, %102, %c0_i64 : i64
                    %105 = arith.extui %104 : i1 to i64
                    %106 = arith.select %103, %c2_i64, %102 : i64
                    %107 = ascscalar.bitcast_memref %alloc_8 byte_offset = %c0_i64 : memref<128xf16, 1> -> memref<?xf16, 1>
                    ascvec.get_buf mte2, %c4_i64, 0 {cannir.buffer.generated_sync}
                    "ascvec.copy_gm2ub_plan"(%107, %alloc_1, %c0_i32, %c0_i32, %c0_i64, %101, %105, %106, %c0_i64, %106, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f16, ub_base = 131584 : i64, ub_capacity = 256 : i64}> : (memref<?xf16, 1>, memref<1xf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                    ascvec.release_buf mte2, %c4_i64, 0 {cannir.buffer.generated_sync}
                    %108 = cannir.extract_layout %arg3 : !cannir.tensor<<gm, f32>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                    %109 = cannir.extract_shape %108 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, -1, 128)>
                    %110 = cannir.extract_dim %109[2] : !cannir.shape<(-1, -1, -1, 128)>
                    %111 = arith.minsi %41, %110 : i64
                    %112 = arith.subi %110, %111 : i64
                    %113 = arith.minsi %112, %c1_i64 : i64
                    %114 = arith.cmpi sle, %41, %110 : i64
                    %115 = arith.select %114, %113, %c0_i64 : i64
                    %116 = cannir.extract_physical_layout %arg3 : !cannir.tensor<<gm, f32>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                    %117 = cannir.extract_stride %116 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, 128, 1)>
                    %118 = cannir.extract_dim %117[0] : !cannir.stride<(-1, -1, 128, 1)>
                    %119 = arith.muli %39, %118 : i64
                    %120 = arith.muli %119, %c4_i64 : i64
                    %121 = arith.muli %41, %c512_i64 : i64
                    %122 = arith.addi %120, %121 : i64
                    %123 = arith.muli %115, %c512_i64 : i64
                    %124 = arith.cmpi eq, %123, %c0_i64 : i64
                    %125 = arith.cmpi ne, %123, %c0_i64 : i64
                    %126 = arith.extui %125 : i1 to i64
                    %127 = arith.select %124, %c4_i64, %123 : i64
                    %128 = ascscalar.bitcast_memref %alloc_10 byte_offset = %c0_i64 : memref<128xf32, 1> -> memref<?xf32, 1>
                    ascvec.get_buf mte2, %c6_i64, 0 {cannir.buffer.generated_sync}
                    "ascvec.copy_gm2ub_plan"(%128, %alloc_2, %c0_i32, %c0_i32, %c0_i64, %122, %126, %127, %c0_i64, %127, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 132096 : i64, ub_capacity = 512 : i64}> : (memref<?xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                    ascvec.release_buf mte2, %c6_i64, 0 {cannir.buffer.generated_sync}
                    %129:2 = scf.for %arg5 = %c0_i64 to %11 step %c1_i64 iter_args(%arg6 = %c1_i64, %arg7 = %c0_i64) -> (i64, i64)  : i64 {
                      %138 = arith.cmpi slt, %arg5, %55 : i64
                      %139:2 = scf.if %138 -> (i64, i64) {
                        %140 = arith.addi %arg5, %c1_i64 : i64
                        %141 = arith.cmpi slt, %140, %55 : i64
                        %142 = scf.if %141 -> (i64) {
                          %161 = arith.cmpi eq, %arg6, %c0_i64 : i64
                          %162 = ascscalar.memref_address %alloc_6 : memref<128xf16, 1>
                          %163 = ascscalar.memref_address %alloc_7 : memref<128xf16, 1>
                          %164 = arith.select %161, %162, %163 : i64
                          %165 = arith.select %161, %c2_i64, %c3_i64 : i64
                          %166 = arith.addi %arg6, %c1_i64 overflow<nsw> : i64
                          %167 = arith.remui %166, %c2_i64 : i64
                          %168 = cannir.extract_dim %76[1] : !cannir.stride<(-1, -1, 128, 1)>
                          %169 = arith.muli %140, %168 : i64
                          %170 = arith.muli %169, %c2_i64 : i64
                          %171 = arith.addi %79, %170 : i64
                          %172 = arith.addi %171, %80 : i64
                          %173 = ascscalar.bind_local_ptr %164 : memref<1xf16, 1>
                          ascvec.get_buf mte2, %165, 0 {cannir.buffer.generated_sync}
                          "ascvec.copy_gm2ub_plan"(%173, %alloc_0, %c0_i32, %c0_i32, %c0_i64, %172, %85, %86, %c0_i64, %86, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xf16, 1>, memref<1xf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                          ascvec.release_buf mte2, %165, 0 {cannir.buffer.generated_sync}
                          %174 = ascscalar.memref_address %alloc_8 : memref<128xf16, 1>
                          %175 = ascscalar.memref_address %alloc_9 : memref<128xf16, 1>
                          %176 = arith.select %161, %174, %175 : i64
                          %177 = arith.select %161, %c4_i64, %c5_i64 : i64
                          %178 = cannir.extract_dim %97[1] : !cannir.stride<(-1, -1, 128, 1)>
                          %179 = arith.muli %140, %178 : i64
                          %180 = arith.muli %179, %c2_i64 : i64
                          %181 = arith.addi %100, %180 : i64
                          %182 = arith.addi %181, %80 : i64
                          %183 = ascscalar.bind_local_ptr %176 : memref<1xf16, 1>
                          ascvec.get_buf mte2, %177, 0 {cannir.buffer.generated_sync}
                          "ascvec.copy_gm2ub_plan"(%183, %alloc_1, %c0_i32, %c0_i32, %c0_i64, %182, %105, %106, %c0_i64, %106, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xf16, 1>, memref<1xf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                          ascvec.release_buf mte2, %177, 0 {cannir.buffer.generated_sync}
                          %184 = ascscalar.memref_address %alloc_10 : memref<128xf32, 1>
                          %185 = ascscalar.memref_address %alloc_11 : memref<128xf32, 1>
                          %186 = arith.select %161, %184, %185 : i64
                          %187 = arith.select %161, %c6_i64, %c7_i64 : i64
                          %188 = cannir.extract_dim %117[1] : !cannir.stride<(-1, -1, 128, 1)>
                          %189 = arith.muli %140, %188 : i64
                          %190 = arith.muli %189, %c4_i64 : i64
                          %191 = arith.addi %120, %190 : i64
                          %192 = arith.addi %191, %121 : i64
                          %193 = ascscalar.bind_local_ptr %186 : memref<1xf32, 1>
                          ascvec.get_buf mte2, %187, 0 {cannir.buffer.generated_sync}
                          "ascvec.copy_gm2ub_plan"(%193, %alloc_2, %c0_i32, %c0_i32, %c0_i64, %192, %126, %127, %c0_i64, %127, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = -1 : i64, ub_capacity = 4 : i64}> : (memref<1xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                          ascvec.release_buf mte2, %187, 0 {cannir.buffer.generated_sync}
                          scf.yield %167 : i64
                        } else {
                          %161 = arith.cmpi sgt, %58, %c0_i64 : i64
                          %162 = scf.if %161 -> (i64) {
                            %163 = arith.muli %50, %64 : i64
                            %164 = arith.muli %163, %c4_i64 : i64
                            %165 = arith.muli %52, %c65536_i64 : i64
                            %166 = arith.addi %164, %165 : i64
                            ascvec.get_buf mte2, %c1_i64, 0 {cannir.buffer.generated_sync}
                            "ascvec.copy_gm2ub_plan"(%alloc_5, %alloc, %c0_i32, %c0_i32, %c0_i64, %166, %c1_i64, %c65536_i64, %c0_i64, %c65536_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 65536 : i64, ub_capacity = 65536 : i64}> : (memref<16384xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                            ascvec.release_buf mte2, %c1_i64, 0 {cannir.buffer.generated_sync}
                            %167 = arith.cmpi eq, %arg6, %c0_i64 : i64
                            %168 = ascscalar.memref_address %alloc_6 : memref<128xf16, 1>
                            %169 = ascscalar.memref_address %alloc_7 : memref<128xf16, 1>
                            %170 = arith.select %167, %168, %169 : i64
                            %171 = arith.select %167, %c2_i64, %c3_i64 : i64
                            %172 = arith.addi %arg6, %c1_i64 overflow<nsw> : i64
                            %173 = arith.remui %172, %c2_i64 : i64
                            %174 = arith.minsi %52, %69 : i64
                            %175 = arith.subi %69, %174 : i64
                            %176 = arith.minsi %175, %c1_i64 : i64
                            %177 = arith.cmpi sle, %52, %69 : i64
                            %178 = arith.select %177, %176, %c0_i64 : i64
                            %179 = arith.muli %50, %77 : i64
                            %180 = arith.muli %179, %c2_i64 : i64
                            %181 = arith.muli %52, %c256_i64 : i64
                            %182 = arith.addi %180, %181 : i64
                            %183 = arith.muli %178, %c256_i64 : i64
                            %184 = arith.cmpi eq, %183, %c0_i64 : i64
                            %185 = arith.cmpi ne, %183, %c0_i64 : i64
                            %186 = arith.extui %185 : i1 to i64
                            %187 = arith.select %184, %c2_i64, %183 : i64
                            %188 = ascscalar.bind_local_ptr %170 : memref<1xf16, 1>
                            ascvec.get_buf mte2, %171, 0 {cannir.buffer.generated_sync}
                            "ascvec.copy_gm2ub_plan"(%188, %alloc_0, %c0_i32, %c0_i32, %c0_i64, %182, %186, %187, %c0_i64, %187, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xf16, 1>, memref<1xf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                            ascvec.release_buf mte2, %171, 0 {cannir.buffer.generated_sync}
                            %189 = ascscalar.memref_address %alloc_8 : memref<128xf16, 1>
                            %190 = ascscalar.memref_address %alloc_9 : memref<128xf16, 1>
                            %191 = arith.select %167, %189, %190 : i64
                            %192 = arith.select %167, %c4_i64, %c5_i64 : i64
                            %193 = arith.minsi %52, %90 : i64
                            %194 = arith.subi %90, %193 : i64
                            %195 = arith.minsi %194, %c1_i64 : i64
                            %196 = arith.cmpi sle, %52, %90 : i64
                            %197 = arith.select %196, %195, %c0_i64 : i64
                            %198 = arith.muli %50, %98 : i64
                            %199 = arith.muli %198, %c2_i64 : i64
                            %200 = arith.addi %199, %181 : i64
                            %201 = arith.muli %197, %c256_i64 : i64
                            %202 = arith.cmpi eq, %201, %c0_i64 : i64
                            %203 = arith.cmpi ne, %201, %c0_i64 : i64
                            %204 = arith.extui %203 : i1 to i64
                            %205 = arith.select %202, %c2_i64, %201 : i64
                            %206 = ascscalar.bind_local_ptr %191 : memref<1xf16, 1>
                            ascvec.get_buf mte2, %192, 0 {cannir.buffer.generated_sync}
                            "ascvec.copy_gm2ub_plan"(%206, %alloc_1, %c0_i32, %c0_i32, %c0_i64, %200, %204, %205, %c0_i64, %205, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xf16, 1>, memref<1xf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                            ascvec.release_buf mte2, %192, 0 {cannir.buffer.generated_sync}
                            %207 = ascscalar.memref_address %alloc_10 : memref<128xf32, 1>
                            %208 = ascscalar.memref_address %alloc_11 : memref<128xf32, 1>
                            %209 = arith.select %167, %207, %208 : i64
                            %210 = arith.select %167, %c6_i64, %c7_i64 : i64
                            %211 = arith.minsi %52, %110 : i64
                            %212 = arith.subi %110, %211 : i64
                            %213 = arith.minsi %212, %c1_i64 : i64
                            %214 = arith.cmpi sle, %52, %110 : i64
                            %215 = arith.select %214, %213, %c0_i64 : i64
                            %216 = arith.muli %50, %118 : i64
                            %217 = arith.muli %216, %c4_i64 : i64
                            %218 = arith.muli %52, %c512_i64 : i64
                            %219 = arith.addi %217, %218 : i64
                            %220 = arith.muli %215, %c512_i64 : i64
                            %221 = arith.cmpi eq, %220, %c0_i64 : i64
                            %222 = arith.cmpi ne, %220, %c0_i64 : i64
                            %223 = arith.extui %222 : i1 to i64
                            %224 = arith.select %221, %c4_i64, %220 : i64
                            %225 = ascscalar.bind_local_ptr %209 : memref<1xf32, 1>
                            ascvec.get_buf mte2, %210, 0 {cannir.buffer.generated_sync}
                            "ascvec.copy_gm2ub_plan"(%225, %alloc_2, %c0_i32, %c0_i32, %c0_i64, %219, %223, %224, %c0_i64, %224, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = -1 : i64, ub_capacity = 4 : i64}> : (memref<1xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                            ascvec.release_buf mte2, %210, 0 {cannir.buffer.generated_sync}
                            scf.yield %173 : i64
                          } else {
                            scf.yield %arg6 : i64
                          }
                          scf.yield %162 : i64
                        }
                        %143 = arith.cmpi eq, %arg7, %c0_i64 : i64
                        %144 = ascscalar.memref_address %alloc_6 : memref<128xf16, 1>
                        %145 = ascscalar.memref_address %alloc_7 : memref<128xf16, 1>
                        %146 = arith.select %143, %144, %145 : i64
                        %147 = ascscalar.bind_local_ptr %146 : memref<128xf16, 1>
                        %148 = arith.select %143, %c2_i64, %c3_i64 : i64
                        %149 = arith.addi %arg7, %c1_i64 overflow<nsw> : i64
                        %150 = arith.remui %149, %c2_i64 : i64
                        %151 = ascscalar.memref_address %alloc_8 : memref<128xf16, 1>
                        %152 = ascscalar.memref_address %alloc_9 : memref<128xf16, 1>
                        %153 = arith.select %143, %151, %152 : i64
                        %154 = ascscalar.bind_local_ptr %153 : memref<128xf16, 1>
                        %155 = arith.select %143, %c4_i64, %c5_i64 : i64
                        %156 = ascscalar.memref_address %alloc_10 : memref<128xf32, 1>
                        %157 = ascscalar.memref_address %alloc_11 : memref<128xf32, 1>
                        %158 = arith.select %143, %156, %157 : i64
                        %159 = ascscalar.bind_local_ptr %158 : memref<128xf32, 1>
                        %160 = arith.select %143, %c6_i64, %c7_i64 : i64
                        ascvec.get_buf v, %148, 0 {cannir.buffer.generated_sync}
                        ascvec.vf(%147, %alloc_12 : memref<128xf16, 1>, memref<128xf32, 1>) {
                        ^bb0(%arg8: memref<128xf16, 1>, %arg9: memref<128xf32, 1>):
                          %c1_i32_13 = arith.constant 1 : i32
                          %c2_i32 = arith.constant 2 : i32
                          %c0_i32_14 = arith.constant 0 : i32
                          %c128_i32 = arith.constant 128 : i32
                          %c64_i32 = arith.constant 64 : i32
                          scf.for %arg10 = %c0_i32_14 to %c2_i32 step %c1_i32_13  : i32 {
                            %161 = arith.muli %arg10, %c64_i32 overflow<nsw> : i32
                            %162 = arith.subi %c128_i32, %161 overflow<nsw> : i32
                            %mask, %remaining_out = ascvec.update_mask %162 : !ascvec.mask
                            %163 = arith.index_cast %161 : i32 to index
                            %164 = ascvec.loadalign_unpack %arg8[%163] {mode = "b16_to_b32"} : memref<128xf16, 1>, index -> <f16>
                            %165 = ascvec.cast %164, %mask {rounding = 6 : i32} : <f16> -> <f32>
                            ascvec.storealign %arg9[%163], %165, %mask : memref<128xf32, 1>, index, <f32>
                          }
                        } {cannir.vf_mode = "simd"}
                        ascvec.release_buf v, %148, 0 {cannir.buffer.generated_sync}
                        ascvec.get_buf v, %155, 0 {cannir.buffer.generated_sync}
                        ascvec.get_buf v, %160, 0 {cannir.buffer.generated_sync}
                        ascvec.get_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
                        ascvec.vf(%154, %159, %alloc_12, %alloc_4 : memref<128xf16, 1>, memref<128xf32, 1>, memref<128xf32, 1>, memref<16384xf32, 1>) {
                        ^bb0(%arg8: memref<128xf16, 1>, %arg9: memref<128xf32, 1>, %arg10: memref<128xf32, 1>, %arg11: memref<16384xf32, 1>):
                          %c64 = arith.constant 64 : index
                          %c1_i32_13 = arith.constant 1 : i32
                          %c0_i32_14 = arith.constant 0 : i32
                          %c128_i32 = arith.constant 128 : i32
                          %c0_15 = arith.constant 0 : index
                          %c64_i32 = arith.constant 64 : i32
                          %mask, %remaining_out = ascvec.update_mask %c64_i32 : !ascvec.mask
                          %161 = ascvec.loadalign_unpack %arg8[%c0_15] {mode = "b16_to_b32"} : memref<128xf16, 1>, index -> <f16>
                          %162 = ascvec.loadalign_unpack %arg8[%c64] {mode = "b16_to_b32"} : memref<128xf16, 1>, index -> <f16>
                          %163 = ascvec.cast %161, %mask {rounding = 6 : i32} : <f16> -> <f32>
                          %164 = ascvec.cast %162, %mask {rounding = 6 : i32} : <f16> -> <f32>
                          %165 = ascvec.loadalign %arg9[%c0_15] : memref<128xf32, 1>, index -> <f32>
                          %166 = ascvec.loadalign %arg9[%c64] : memref<128xf32, 1>, index -> <f32>
                          scf.for %arg12 = %c0_i32_14 to %c128_i32 step %c1_i32_13  : i32 {
                            %167 = arith.muli %arg12, %c128_i32 overflow<nsw> : i32
                            %168 = arith.index_cast %arg12 : i32 to index
                            %169 = ascvec.loadalign_brc %arg10[%168] : memref<128xf32, 1>, index -> <f32>
                            %170 = arith.index_cast %167 : i32 to index
                            %171 = ascvec.loadalign %arg11[%170] : memref<16384xf32, 1>, index -> <f32>
                            %172 = arith.addi %167, %c64_i32 overflow<nsw> : i32
                            %173 = arith.index_cast %172 : i32 to index
                            %174 = ascvec.loadalign %arg11[%173] : memref<16384xf32, 1>, index -> <f32>
                            %175 = ascvec.mul %171, %165, %mask : <f32>
                            %176 = ascvec.mul %174, %166, %mask : <f32>
                            %177 = ascvec.copy %169, %mask : <f32>
                            %178 = ascvec.madd_two_address %177, %163, %175, %mask : <f32>
                            %179 = ascvec.madd_two_address %169, %164, %176, %mask : <f32>
                            ascvec.storealign %arg11[%170], %178, %mask : memref<16384xf32, 1>, index, <f32>
                            ascvec.storealign %arg11[%173], %179, %mask : memref<16384xf32, 1>, index, <f32>
                          } {cannir.loop_unroll = 4 : i64}
                        } {cannir.vf_mode = "simd"}
                        ascvec.release_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
                        ascvec.release_buf v, %160, 0 {cannir.buffer.generated_sync}
                        ascvec.release_buf v, %155, 0 {cannir.buffer.generated_sync}
                        scf.yield %142, %150 : i64, i64
                      } else {
                        scf.yield %arg6, %arg7 : i64, i64
                      }
                      scf.yield %139#0, %139#1 : i64, i64
                    }
                    %130 = arith.muli %41, %c16384_i64 : i64
                    %131 = arith.addi %65, %130 : i64
                    %132 = arith.muli %131, %c32_i64 : i64
                    %133 = ascscalar.memref_address %alloc : memref<1xf32>
                    %134 = arith.divui %132, %c8_i64 : i64
                    %135 = arith.addi %133, %134 : i64
                    %136 = ascscalar.bind_gm_ptr %135 : memref<1xf32>
                    %137 = ascscalar.bitcast_memref %alloc_4 byte_offset = %c0_i64 : memref<16384xf32, 1> -> memref<?xf32, 1>
                    ascvec.get_buf mte3, %c0_i64, 0 {cannir.buffer.generated_sync}
                    ascvec.copy_ub2gm %136[%c0], %137[%c0], %c1_i32, %c65536_i32, %c0_i64, %c65536_i32 l2_cache_ctl %c0_i32 : memref<1xf32>, memref<?xf32, 1>
                    ascvec.release_buf mte3, %c0_i64, 0 {cannir.buffer.generated_sync}
                    scf.yield %129#0, %129#1, %129#0, %129#0 : i64, i64, i64, i64
                  } else {
                    scf.yield %c0_i64, %c0_i64, %c0_i64, %c0_i64 : i64, i64, i64, i64
                  }
                  %61 = arith.cmpi sgt, %58, %c0_i64 : i64
                  scf.if %61 {
                    %62 = arith.cmpi eq, %55, %c0_i64 : i64
                    %63 = scf.if %62 -> (i64) {
                      %77 = cannir.extract_physical_layout %arg0 : !cannir.tensor<<gm, f32>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, identity>
                      %78 = cannir.extract_stride %77 : <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>> -> <(-1, 16384, 128, 1)>
                      %79 = cannir.extract_dim %78[0] : !cannir.stride<(-1, 16384, 128, 1)>
                      %80 = arith.muli %50, %79 : i64
                      %81 = arith.muli %80, %c4_i64 : i64
                      %82 = arith.muli %52, %c65536_i64 : i64
                      %83 = arith.addi %81, %82 : i64
                      ascvec.get_buf mte2, %c1_i64, 0 {cannir.buffer.generated_sync}
                      "ascvec.copy_gm2ub_plan"(%alloc_5, %alloc, %c0_i32, %c0_i32, %c0_i64, %83, %c1_i64, %c65536_i64, %c0_i64, %c65536_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 65536 : i64, ub_capacity = 65536 : i64}> : (memref<16384xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                      ascvec.release_buf mte2, %c1_i64, 0 {cannir.buffer.generated_sync}
                      %84 = arith.cmpi eq, %60#0, %c0_i64 : i64
                      %85 = ascscalar.memref_address %alloc_6 : memref<128xf16, 1>
                      %86 = ascscalar.memref_address %alloc_7 : memref<128xf16, 1>
                      %87 = arith.select %84, %85, %86 : i64
                      %88 = arith.select %84, %c2_i64, %c3_i64 : i64
                      %89 = arith.addi %60#0, %c1_i64 overflow<nsw> : i64
                      %90 = arith.remui %89, %c2_i64 : i64
                      %91 = cannir.extract_dim %10[2] : !cannir.shape<(-1, -1, -1, 128)>
                      %92 = arith.minsi %52, %91 : i64
                      %93 = arith.subi %91, %92 : i64
                      %94 = arith.minsi %93, %c1_i64 : i64
                      %95 = arith.cmpi sle, %52, %91 : i64
                      %96 = arith.select %95, %94, %c0_i64 : i64
                      %97 = cannir.extract_physical_layout %arg1 : !cannir.tensor<<gm, f16>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                      %98 = cannir.extract_stride %97 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, 128, 1)>
                      %99 = cannir.extract_dim %98[0] : !cannir.stride<(-1, -1, 128, 1)>
                      %100 = arith.muli %50, %99 : i64
                      %101 = arith.muli %100, %c2_i64 : i64
                      %102 = arith.muli %52, %c256_i64 : i64
                      %103 = arith.addi %101, %102 : i64
                      %104 = arith.muli %96, %c256_i64 : i64
                      %105 = arith.cmpi eq, %104, %c0_i64 : i64
                      %106 = arith.cmpi ne, %104, %c0_i64 : i64
                      %107 = arith.extui %106 : i1 to i64
                      %108 = arith.select %105, %c2_i64, %104 : i64
                      %109 = ascscalar.bind_local_ptr %87 : memref<1xf16, 1>
                      ascvec.get_buf mte2, %88, 0 {cannir.buffer.generated_sync}
                      "ascvec.copy_gm2ub_plan"(%109, %alloc_0, %c0_i32, %c0_i32, %c0_i64, %103, %107, %108, %c0_i64, %108, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xf16, 1>, memref<1xf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                      ascvec.release_buf mte2, %88, 0 {cannir.buffer.generated_sync}
                      %110 = arith.cmpi eq, %60#2, %c0_i64 : i64
                      %111 = ascscalar.memref_address %alloc_8 : memref<128xf16, 1>
                      %112 = ascscalar.memref_address %alloc_9 : memref<128xf16, 1>
                      %113 = arith.select %110, %111, %112 : i64
                      %114 = arith.select %110, %c4_i64, %c5_i64 : i64
                      %115 = cannir.extract_layout %arg2 : !cannir.tensor<<gm, f16>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                      %116 = cannir.extract_shape %115 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, -1, 128)>
                      %117 = cannir.extract_dim %116[2] : !cannir.shape<(-1, -1, -1, 128)>
                      %118 = arith.minsi %52, %117 : i64
                      %119 = arith.subi %117, %118 : i64
                      %120 = arith.minsi %119, %c1_i64 : i64
                      %121 = arith.cmpi sle, %52, %117 : i64
                      %122 = arith.select %121, %120, %c0_i64 : i64
                      %123 = cannir.extract_physical_layout %arg2 : !cannir.tensor<<gm, f16>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                      %124 = cannir.extract_stride %123 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, 128, 1)>
                      %125 = cannir.extract_dim %124[0] : !cannir.stride<(-1, -1, 128, 1)>
                      %126 = arith.muli %50, %125 : i64
                      %127 = arith.muli %126, %c2_i64 : i64
                      %128 = arith.addi %127, %102 : i64
                      %129 = arith.muli %122, %c256_i64 : i64
                      %130 = arith.cmpi eq, %129, %c0_i64 : i64
                      %131 = arith.cmpi ne, %129, %c0_i64 : i64
                      %132 = arith.extui %131 : i1 to i64
                      %133 = arith.select %130, %c2_i64, %129 : i64
                      %134 = ascscalar.bind_local_ptr %113 : memref<1xf16, 1>
                      ascvec.get_buf mte2, %114, 0 {cannir.buffer.generated_sync}
                      "ascvec.copy_gm2ub_plan"(%134, %alloc_1, %c0_i32, %c0_i32, %c0_i64, %128, %132, %133, %c0_i64, %133, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xf16, 1>, memref<1xf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                      ascvec.release_buf mte2, %114, 0 {cannir.buffer.generated_sync}
                      %135 = arith.cmpi eq, %60#3, %c0_i64 : i64
                      %136 = ascscalar.memref_address %alloc_10 : memref<128xf32, 1>
                      %137 = ascscalar.memref_address %alloc_11 : memref<128xf32, 1>
                      %138 = arith.select %135, %136, %137 : i64
                      %139 = arith.select %135, %c6_i64, %c7_i64 : i64
                      %140 = cannir.extract_layout %arg3 : !cannir.tensor<<gm, f32>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                      %141 = cannir.extract_shape %140 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, -1, 128)>
                      %142 = cannir.extract_dim %141[2] : !cannir.shape<(-1, -1, -1, 128)>
                      %143 = arith.minsi %52, %142 : i64
                      %144 = arith.subi %142, %143 : i64
                      %145 = arith.minsi %144, %c1_i64 : i64
                      %146 = arith.cmpi sle, %52, %142 : i64
                      %147 = arith.select %146, %145, %c0_i64 : i64
                      %148 = cannir.extract_physical_layout %arg3 : !cannir.tensor<<gm, f32>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                      %149 = cannir.extract_stride %148 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, 128, 1)>
                      %150 = cannir.extract_dim %149[0] : !cannir.stride<(-1, -1, 128, 1)>
                      %151 = arith.muli %50, %150 : i64
                      %152 = arith.muli %151, %c4_i64 : i64
                      %153 = arith.muli %52, %c512_i64 : i64
                      %154 = arith.addi %152, %153 : i64
                      %155 = arith.muli %147, %c512_i64 : i64
                      %156 = arith.cmpi eq, %155, %c0_i64 : i64
                      %157 = arith.cmpi ne, %155, %c0_i64 : i64
                      %158 = arith.extui %157 : i1 to i64
                      %159 = arith.select %156, %c4_i64, %155 : i64
                      %160 = ascscalar.bind_local_ptr %138 : memref<1xf32, 1>
                      ascvec.get_buf mte2, %139, 0 {cannir.buffer.generated_sync}
                      "ascvec.copy_gm2ub_plan"(%160, %alloc_2, %c0_i32, %c0_i32, %c0_i64, %154, %158, %159, %c0_i64, %159, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = -1 : i64, ub_capacity = 4 : i64}> : (memref<1xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                      ascvec.release_buf mte2, %139, 0 {cannir.buffer.generated_sync}
                      scf.yield %90 : i64
                    } else {
                      scf.yield %60#0 : i64
                    }
                    %64:2 = scf.for %arg5 = %c0_i64 to %11 step %c1_i64 iter_args(%arg6 = %63, %arg7 = %60#1) -> (i64, i64)  : i64 {
                      %77 = arith.cmpi slt, %arg5, %58 : i64
                      %78:2 = scf.if %77 -> (i64, i64) {
                        %79 = arith.addi %arg5, %c1_i64 : i64
                        %80 = arith.cmpi slt, %79, %58 : i64
                        %81 = scf.if %80 -> (i64) {
                          %100 = arith.cmpi eq, %arg6, %c0_i64 : i64
                          %101 = ascscalar.memref_address %alloc_6 : memref<128xf16, 1>
                          %102 = ascscalar.memref_address %alloc_7 : memref<128xf16, 1>
                          %103 = arith.select %100, %101, %102 : i64
                          %104 = arith.select %100, %c2_i64, %c3_i64 : i64
                          %105 = arith.addi %arg6, %c1_i64 overflow<nsw> : i64
                          %106 = arith.remui %105, %c2_i64 : i64
                          %107 = cannir.extract_dim %10[2] : !cannir.shape<(-1, -1, -1, 128)>
                          %108 = arith.minsi %52, %107 : i64
                          %109 = arith.subi %107, %108 : i64
                          %110 = arith.minsi %109, %c1_i64 : i64
                          %111 = arith.cmpi sle, %52, %107 : i64
                          %112 = arith.select %111, %110, %c0_i64 : i64
                          %113 = cannir.extract_physical_layout %arg1 : !cannir.tensor<<gm, f16>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                          %114 = cannir.extract_stride %113 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, 128, 1)>
                          %115 = cannir.extract_dim %114[0] : !cannir.stride<(-1, -1, 128, 1)>
                          %116 = arith.muli %50, %115 : i64
                          %117 = arith.muli %116, %c2_i64 : i64
                          %118 = cannir.extract_dim %114[1] : !cannir.stride<(-1, -1, 128, 1)>
                          %119 = arith.muli %79, %118 : i64
                          %120 = arith.muli %119, %c2_i64 : i64
                          %121 = arith.addi %117, %120 : i64
                          %122 = arith.muli %52, %c256_i64 : i64
                          %123 = arith.addi %121, %122 : i64
                          %124 = arith.muli %112, %c256_i64 : i64
                          %125 = arith.cmpi eq, %124, %c0_i64 : i64
                          %126 = arith.cmpi ne, %124, %c0_i64 : i64
                          %127 = arith.extui %126 : i1 to i64
                          %128 = arith.select %125, %c2_i64, %124 : i64
                          %129 = ascscalar.bind_local_ptr %103 : memref<1xf16, 1>
                          ascvec.get_buf mte2, %104, 0 {cannir.buffer.generated_sync}
                          "ascvec.copy_gm2ub_plan"(%129, %alloc_0, %c0_i32, %c0_i32, %c0_i64, %123, %127, %128, %c0_i64, %128, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xf16, 1>, memref<1xf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                          ascvec.release_buf mte2, %104, 0 {cannir.buffer.generated_sync}
                          %130 = ascscalar.memref_address %alloc_8 : memref<128xf16, 1>
                          %131 = ascscalar.memref_address %alloc_9 : memref<128xf16, 1>
                          %132 = arith.select %100, %130, %131 : i64
                          %133 = arith.select %100, %c4_i64, %c5_i64 : i64
                          %134 = cannir.extract_layout %arg2 : !cannir.tensor<<gm, f16>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                          %135 = cannir.extract_shape %134 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, -1, 128)>
                          %136 = cannir.extract_dim %135[2] : !cannir.shape<(-1, -1, -1, 128)>
                          %137 = arith.minsi %52, %136 : i64
                          %138 = arith.subi %136, %137 : i64
                          %139 = arith.minsi %138, %c1_i64 : i64
                          %140 = arith.cmpi sle, %52, %136 : i64
                          %141 = arith.select %140, %139, %c0_i64 : i64
                          %142 = cannir.extract_physical_layout %arg2 : !cannir.tensor<<gm, f16>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                          %143 = cannir.extract_stride %142 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, 128, 1)>
                          %144 = cannir.extract_dim %143[0] : !cannir.stride<(-1, -1, 128, 1)>
                          %145 = arith.muli %50, %144 : i64
                          %146 = arith.muli %145, %c2_i64 : i64
                          %147 = cannir.extract_dim %143[1] : !cannir.stride<(-1, -1, 128, 1)>
                          %148 = arith.muli %79, %147 : i64
                          %149 = arith.muli %148, %c2_i64 : i64
                          %150 = arith.addi %146, %149 : i64
                          %151 = arith.addi %150, %122 : i64
                          %152 = arith.muli %141, %c256_i64 : i64
                          %153 = arith.cmpi eq, %152, %c0_i64 : i64
                          %154 = arith.cmpi ne, %152, %c0_i64 : i64
                          %155 = arith.extui %154 : i1 to i64
                          %156 = arith.select %153, %c2_i64, %152 : i64
                          %157 = ascscalar.bind_local_ptr %132 : memref<1xf16, 1>
                          ascvec.get_buf mte2, %133, 0 {cannir.buffer.generated_sync}
                          "ascvec.copy_gm2ub_plan"(%157, %alloc_1, %c0_i32, %c0_i32, %c0_i64, %151, %155, %156, %c0_i64, %156, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xf16, 1>, memref<1xf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                          ascvec.release_buf mte2, %133, 0 {cannir.buffer.generated_sync}
                          %158 = ascscalar.memref_address %alloc_10 : memref<128xf32, 1>
                          %159 = ascscalar.memref_address %alloc_11 : memref<128xf32, 1>
                          %160 = arith.select %100, %158, %159 : i64
                          %161 = arith.select %100, %c6_i64, %c7_i64 : i64
                          %162 = cannir.extract_layout %arg3 : !cannir.tensor<<gm, f32>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                          %163 = cannir.extract_shape %162 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, -1, 128)>
                          %164 = cannir.extract_dim %163[2] : !cannir.shape<(-1, -1, -1, 128)>
                          %165 = arith.minsi %52, %164 : i64
                          %166 = arith.subi %164, %165 : i64
                          %167 = arith.minsi %166, %c1_i64 : i64
                          %168 = arith.cmpi sle, %52, %164 : i64
                          %169 = arith.select %168, %167, %c0_i64 : i64
                          %170 = cannir.extract_physical_layout %arg3 : !cannir.tensor<<gm, f32>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                          %171 = cannir.extract_stride %170 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, 128, 1)>
                          %172 = cannir.extract_dim %171[0] : !cannir.stride<(-1, -1, 128, 1)>
                          %173 = arith.muli %50, %172 : i64
                          %174 = arith.muli %173, %c4_i64 : i64
                          %175 = cannir.extract_dim %171[1] : !cannir.stride<(-1, -1, 128, 1)>
                          %176 = arith.muli %79, %175 : i64
                          %177 = arith.muli %176, %c4_i64 : i64
                          %178 = arith.addi %174, %177 : i64
                          %179 = arith.muli %52, %c512_i64 : i64
                          %180 = arith.addi %178, %179 : i64
                          %181 = arith.muli %169, %c512_i64 : i64
                          %182 = arith.cmpi eq, %181, %c0_i64 : i64
                          %183 = arith.cmpi ne, %181, %c0_i64 : i64
                          %184 = arith.extui %183 : i1 to i64
                          %185 = arith.select %182, %c4_i64, %181 : i64
                          %186 = ascscalar.bind_local_ptr %160 : memref<1xf32, 1>
                          ascvec.get_buf mte2, %161, 0 {cannir.buffer.generated_sync}
                          "ascvec.copy_gm2ub_plan"(%186, %alloc_2, %c0_i32, %c0_i32, %c0_i64, %180, %184, %185, %c0_i64, %185, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = -1 : i64, ub_capacity = 4 : i64}> : (memref<1xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                          ascvec.release_buf mte2, %161, 0 {cannir.buffer.generated_sync}
                          scf.yield %106 : i64
                        } else {
                          scf.yield %arg6 : i64
                        }
                        %82 = arith.cmpi eq, %arg7, %c0_i64 : i64
                        %83 = ascscalar.memref_address %alloc_6 : memref<128xf16, 1>
                        %84 = ascscalar.memref_address %alloc_7 : memref<128xf16, 1>
                        %85 = arith.select %82, %83, %84 : i64
                        %86 = ascscalar.bind_local_ptr %85 : memref<128xf16, 1>
                        %87 = arith.select %82, %c2_i64, %c3_i64 : i64
                        %88 = arith.addi %arg7, %c1_i64 overflow<nsw> : i64
                        %89 = arith.remui %88, %c2_i64 : i64
                        %90 = ascscalar.memref_address %alloc_8 : memref<128xf16, 1>
                        %91 = ascscalar.memref_address %alloc_9 : memref<128xf16, 1>
                        %92 = arith.select %82, %90, %91 : i64
                        %93 = ascscalar.bind_local_ptr %92 : memref<128xf16, 1>
                        %94 = arith.select %82, %c4_i64, %c5_i64 : i64
                        %95 = ascscalar.memref_address %alloc_10 : memref<128xf32, 1>
                        %96 = ascscalar.memref_address %alloc_11 : memref<128xf32, 1>
                        %97 = arith.select %82, %95, %96 : i64
                        %98 = ascscalar.bind_local_ptr %97 : memref<128xf32, 1>
                        %99 = arith.select %82, %c6_i64, %c7_i64 : i64
                        ascvec.get_buf v, %87, 0 {cannir.buffer.generated_sync}
                        ascvec.vf(%86, %alloc_12 : memref<128xf16, 1>, memref<128xf32, 1>) {
                        ^bb0(%arg8: memref<128xf16, 1>, %arg9: memref<128xf32, 1>):
                          %c1_i32_13 = arith.constant 1 : i32
                          %c2_i32 = arith.constant 2 : i32
                          %c0_i32_14 = arith.constant 0 : i32
                          %c128_i32 = arith.constant 128 : i32
                          %c64_i32 = arith.constant 64 : i32
                          scf.for %arg10 = %c0_i32_14 to %c2_i32 step %c1_i32_13  : i32 {
                            %100 = arith.muli %arg10, %c64_i32 overflow<nsw> : i32
                            %101 = arith.subi %c128_i32, %100 overflow<nsw> : i32
                            %mask, %remaining_out = ascvec.update_mask %101 : !ascvec.mask
                            %102 = arith.index_cast %100 : i32 to index
                            %103 = ascvec.loadalign_unpack %arg8[%102] {mode = "b16_to_b32"} : memref<128xf16, 1>, index -> <f16>
                            %104 = ascvec.cast %103, %mask {rounding = 6 : i32} : <f16> -> <f32>
                            ascvec.storealign %arg9[%102], %104, %mask : memref<128xf32, 1>, index, <f32>
                          }
                        } {cannir.vf_mode = "simd"}
                        ascvec.release_buf v, %87, 0 {cannir.buffer.generated_sync}
                        ascvec.get_buf v, %94, 0 {cannir.buffer.generated_sync}
                        ascvec.get_buf v, %99, 0 {cannir.buffer.generated_sync}
                        ascvec.get_buf v, %c1_i64, 0 {cannir.buffer.generated_sync}
                        ascvec.vf(%93, %98, %alloc_12, %alloc_5 : memref<128xf16, 1>, memref<128xf32, 1>, memref<128xf32, 1>, memref<16384xf32, 1>) {
                        ^bb0(%arg8: memref<128xf16, 1>, %arg9: memref<128xf32, 1>, %arg10: memref<128xf32, 1>, %arg11: memref<16384xf32, 1>):
                          %c64 = arith.constant 64 : index
                          %c1_i32_13 = arith.constant 1 : i32
                          %c0_i32_14 = arith.constant 0 : i32
                          %c128_i32 = arith.constant 128 : i32
                          %c0_15 = arith.constant 0 : index
                          %c64_i32 = arith.constant 64 : i32
                          %mask, %remaining_out = ascvec.update_mask %c64_i32 : !ascvec.mask
                          %100 = ascvec.loadalign_unpack %arg8[%c0_15] {mode = "b16_to_b32"} : memref<128xf16, 1>, index -> <f16>
                          %101 = ascvec.loadalign_unpack %arg8[%c64] {mode = "b16_to_b32"} : memref<128xf16, 1>, index -> <f16>
                          %102 = ascvec.cast %100, %mask {rounding = 6 : i32} : <f16> -> <f32>
                          %103 = ascvec.cast %101, %mask {rounding = 6 : i32} : <f16> -> <f32>
                          %104 = ascvec.loadalign %arg9[%c0_15] : memref<128xf32, 1>, index -> <f32>
                          %105 = ascvec.loadalign %arg9[%c64] : memref<128xf32, 1>, index -> <f32>
                          scf.for %arg12 = %c0_i32_14 to %c128_i32 step %c1_i32_13  : i32 {
                            %106 = arith.muli %arg12, %c128_i32 overflow<nsw> : i32
                            %107 = arith.index_cast %arg12 : i32 to index
                            %108 = ascvec.loadalign_brc %arg10[%107] : memref<128xf32, 1>, index -> <f32>
                            %109 = arith.index_cast %106 : i32 to index
                            %110 = ascvec.loadalign %arg11[%109] : memref<16384xf32, 1>, index -> <f32>
                            %111 = arith.addi %106, %c64_i32 overflow<nsw> : i32
                            %112 = arith.index_cast %111 : i32 to index
                            %113 = ascvec.loadalign %arg11[%112] : memref<16384xf32, 1>, index -> <f32>
                            %114 = ascvec.mul %110, %104, %mask : <f32>
                            %115 = ascvec.mul %113, %105, %mask : <f32>
                            %116 = ascvec.copy %108, %mask : <f32>
                            %117 = ascvec.madd_two_address %116, %102, %114, %mask : <f32>
                            %118 = ascvec.madd_two_address %108, %103, %115, %mask : <f32>
                            ascvec.storealign %arg11[%109], %117, %mask : memref<16384xf32, 1>, index, <f32>
                            ascvec.storealign %arg11[%112], %118, %mask : memref<16384xf32, 1>, index, <f32>
                          } {cannir.loop_unroll = 4 : i64}
                        } {cannir.vf_mode = "simd"}
                        ascvec.release_buf v, %c1_i64, 0 {cannir.buffer.generated_sync}
                        ascvec.release_buf v, %99, 0 {cannir.buffer.generated_sync}
                        ascvec.release_buf v, %94, 0 {cannir.buffer.generated_sync}
                        scf.yield %81, %89 : i64, i64
                      } else {
                        scf.yield %arg6, %arg7 : i64, i64
                      }
                      scf.yield %78#0, %78#1 : i64, i64
                    }
                    %65 = cannir.extract_physical_layout %arg0 : !cannir.tensor<<gm, f32>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, identity>
                    %66 = cannir.extract_stride %65 : <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>> -> <(-1, 16384, 128, 1)>
                    %67 = cannir.extract_dim %66[0] : !cannir.stride<(-1, 16384, 128, 1)>
                    %68 = arith.muli %50, %67 : i64
                    %69 = arith.muli %52, %c16384_i64 : i64
                    %70 = arith.addi %68, %69 : i64
                    %71 = arith.muli %70, %c32_i64 : i64
                    %72 = ascscalar.memref_address %alloc : memref<1xf32>
                    %73 = arith.divui %71, %c8_i64 : i64
                    %74 = arith.addi %72, %73 : i64
                    %75 = ascscalar.bind_gm_ptr %74 : memref<1xf32>
                    %76 = ascscalar.bitcast_memref %alloc_5 byte_offset = %c0_i64 : memref<16384xf32, 1> -> memref<?xf32, 1>
                    ascvec.get_buf mte3, %c1_i64, 0 {cannir.buffer.generated_sync}
                    ascvec.copy_ub2gm %75[%c0], %76[%c0], %c1_i32, %c65536_i32, %c0_i64, %c65536_i32 l2_cache_ctl %c0_i32 : memref<1xf32>, memref<?xf32, 1>
                    ascvec.release_buf mte3, %c1_i64, 0 {cannir.buffer.generated_sync}
                  }
                } else {
                  %27 = arith.muli %25, %c2_i64 : i64
                  %28 = arith.addi %27, %12 : i64
                  %29 = arith.subi %28, %25 : i64
                  %30 = arith.maxsi %3, %c1_i64 : i64
                  %31 = arith.remsi %29, %30 : i64
                  %32 = arith.cmpi slt, %31, %c0_i64 : i64
                  %33 = arith.divsi %29, %30 : i64
                  %34 = arith.extui %32 : i1 to i64
                  %35 = arith.subi %33, %34 : i64
                  %36 = arith.maxsi %2, %c1_i64 : i64
                  %37 = arith.remsi %35, %36 : i64
                  %38 = arith.cmpi slt, %37, %c0_i64 : i64
                  %39 = arith.select %38, %36, %c0_i64 : i64
                  %40 = arith.addi %37, %39 : i64
                  %41 = arith.select %32, %30, %c0_i64 : i64
                  %42 = arith.addi %31, %41 : i64
                  %43 = arith.index_cast %40 : i64 to index
                  %44 = memref.load %alloc_3[%43] : memref<1xi32>
                  %45 = arith.extsi %44 : i32 to i64
                  %46 = arith.cmpi sgt, %45, %c0_i64 : i64
                  scf.if %46 {
                    %47 = cannir.extract_physical_layout %arg0 : !cannir.tensor<<gm, f32>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, identity>
                    %48 = cannir.extract_stride %47 : <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>> -> <(-1, 16384, 128, 1)>
                    %49 = cannir.extract_dim %48[0] : !cannir.stride<(-1, 16384, 128, 1)>
                    %50 = arith.muli %40, %49 : i64
                    %51 = arith.muli %50, %c4_i64 : i64
                    %52 = arith.muli %42, %c65536_i64 : i64
                    %53 = arith.addi %51, %52 : i64
                    ascvec.get_buf mte2, %c0_i64, 0 {cannir.buffer.generated_sync}
                    "ascvec.copy_gm2ub_plan"(%alloc_4, %alloc, %c0_i32, %c0_i32, %c0_i64, %53, %c1_i64, %c65536_i64, %c0_i64, %c65536_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 0 : i64, ub_capacity = 65536 : i64}> : (memref<16384xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                    ascvec.release_buf mte2, %c0_i64, 0 {cannir.buffer.generated_sync}
                    %54 = cannir.extract_dim %10[2] : !cannir.shape<(-1, -1, -1, 128)>
                    %55 = arith.minsi %42, %54 : i64
                    %56 = arith.subi %54, %55 : i64
                    %57 = arith.minsi %56, %c1_i64 : i64
                    %58 = arith.cmpi sle, %42, %54 : i64
                    %59 = arith.select %58, %57, %c0_i64 : i64
                    %60 = cannir.extract_physical_layout %arg1 : !cannir.tensor<<gm, f16>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                    %61 = cannir.extract_stride %60 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, 128, 1)>
                    %62 = cannir.extract_dim %61[0] : !cannir.stride<(-1, -1, 128, 1)>
                    %63 = arith.muli %40, %62 : i64
                    %64 = arith.muli %63, %c2_i64 : i64
                    %65 = arith.muli %42, %c256_i64 : i64
                    %66 = arith.addi %64, %65 : i64
                    %67 = arith.muli %59, %c256_i64 : i64
                    %68 = arith.cmpi eq, %67, %c0_i64 : i64
                    %69 = arith.cmpi ne, %67, %c0_i64 : i64
                    %70 = arith.extui %69 : i1 to i64
                    %71 = arith.select %68, %c2_i64, %67 : i64
                    %72 = ascscalar.bitcast_memref %alloc_6 byte_offset = %c0_i64 : memref<128xf16, 1> -> memref<?xf16, 1>
                    ascvec.get_buf mte2, %c2_i64, 0 {cannir.buffer.generated_sync}
                    "ascvec.copy_gm2ub_plan"(%72, %alloc_0, %c0_i32, %c0_i32, %c0_i64, %66, %70, %71, %c0_i64, %71, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f16, ub_base = 131072 : i64, ub_capacity = 256 : i64}> : (memref<?xf16, 1>, memref<1xf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                    ascvec.release_buf mte2, %c2_i64, 0 {cannir.buffer.generated_sync}
                    %73 = cannir.extract_layout %arg2 : !cannir.tensor<<gm, f16>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                    %74 = cannir.extract_shape %73 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, -1, 128)>
                    %75 = cannir.extract_dim %74[2] : !cannir.shape<(-1, -1, -1, 128)>
                    %76 = arith.minsi %42, %75 : i64
                    %77 = arith.subi %75, %76 : i64
                    %78 = arith.minsi %77, %c1_i64 : i64
                    %79 = arith.cmpi sle, %42, %75 : i64
                    %80 = arith.select %79, %78, %c0_i64 : i64
                    %81 = cannir.extract_physical_layout %arg2 : !cannir.tensor<<gm, f16>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                    %82 = cannir.extract_stride %81 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, 128, 1)>
                    %83 = cannir.extract_dim %82[0] : !cannir.stride<(-1, -1, 128, 1)>
                    %84 = arith.muli %40, %83 : i64
                    %85 = arith.muli %84, %c2_i64 : i64
                    %86 = arith.addi %85, %65 : i64
                    %87 = arith.muli %80, %c256_i64 : i64
                    %88 = arith.cmpi eq, %87, %c0_i64 : i64
                    %89 = arith.cmpi ne, %87, %c0_i64 : i64
                    %90 = arith.extui %89 : i1 to i64
                    %91 = arith.select %88, %c2_i64, %87 : i64
                    %92 = ascscalar.bitcast_memref %alloc_8 byte_offset = %c0_i64 : memref<128xf16, 1> -> memref<?xf16, 1>
                    ascvec.get_buf mte2, %c4_i64, 0 {cannir.buffer.generated_sync}
                    "ascvec.copy_gm2ub_plan"(%92, %alloc_1, %c0_i32, %c0_i32, %c0_i64, %86, %90, %91, %c0_i64, %91, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f16, ub_base = 131584 : i64, ub_capacity = 256 : i64}> : (memref<?xf16, 1>, memref<1xf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                    ascvec.release_buf mte2, %c4_i64, 0 {cannir.buffer.generated_sync}
                    %93 = cannir.extract_layout %arg3 : !cannir.tensor<<gm, f32>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                    %94 = cannir.extract_shape %93 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, -1, 128)>
                    %95 = cannir.extract_dim %94[2] : !cannir.shape<(-1, -1, -1, 128)>
                    %96 = arith.minsi %42, %95 : i64
                    %97 = arith.subi %95, %96 : i64
                    %98 = arith.minsi %97, %c1_i64 : i64
                    %99 = arith.cmpi sle, %42, %95 : i64
                    %100 = arith.select %99, %98, %c0_i64 : i64
                    %101 = cannir.extract_physical_layout %arg3 : !cannir.tensor<<gm, f32>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
                    %102 = cannir.extract_stride %101 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, 128, 1)>
                    %103 = cannir.extract_dim %102[0] : !cannir.stride<(-1, -1, 128, 1)>
                    %104 = arith.muli %40, %103 : i64
                    %105 = arith.muli %104, %c4_i64 : i64
                    %106 = arith.muli %42, %c512_i64 : i64
                    %107 = arith.addi %105, %106 : i64
                    %108 = arith.muli %100, %c512_i64 : i64
                    %109 = arith.cmpi eq, %108, %c0_i64 : i64
                    %110 = arith.cmpi ne, %108, %c0_i64 : i64
                    %111 = arith.extui %110 : i1 to i64
                    %112 = arith.select %109, %c4_i64, %108 : i64
                    %113 = ascscalar.bitcast_memref %alloc_10 byte_offset = %c0_i64 : memref<128xf32, 1> -> memref<?xf32, 1>
                    ascvec.get_buf mte2, %c6_i64, 0 {cannir.buffer.generated_sync}
                    "ascvec.copy_gm2ub_plan"(%113, %alloc_2, %c0_i32, %c0_i32, %c0_i64, %107, %111, %112, %c0_i64, %112, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 132096 : i64, ub_capacity = 512 : i64}> : (memref<?xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                    ascvec.release_buf mte2, %c6_i64, 0 {cannir.buffer.generated_sync}
                    %114:2 = scf.for %arg5 = %c0_i64 to %11 step %c1_i64 iter_args(%arg6 = %c1_i64, %arg7 = %c0_i64) -> (i64, i64)  : i64 {
                      %123 = arith.cmpi slt, %arg5, %45 : i64
                      %124:2 = scf.if %123 -> (i64, i64) {
                        %125 = arith.addi %arg5, %c1_i64 : i64
                        %126 = arith.cmpi slt, %125, %45 : i64
                        %127 = scf.if %126 -> (i64) {
                          %146 = arith.cmpi eq, %arg6, %c0_i64 : i64
                          %147 = ascscalar.memref_address %alloc_6 : memref<128xf16, 1>
                          %148 = ascscalar.memref_address %alloc_7 : memref<128xf16, 1>
                          %149 = arith.select %146, %147, %148 : i64
                          %150 = arith.select %146, %c2_i64, %c3_i64 : i64
                          %151 = arith.addi %arg6, %c1_i64 overflow<nsw> : i64
                          %152 = arith.remui %151, %c2_i64 : i64
                          %153 = cannir.extract_dim %61[1] : !cannir.stride<(-1, -1, 128, 1)>
                          %154 = arith.muli %125, %153 : i64
                          %155 = arith.muli %154, %c2_i64 : i64
                          %156 = arith.addi %64, %155 : i64
                          %157 = arith.addi %156, %65 : i64
                          %158 = ascscalar.bind_local_ptr %149 : memref<1xf16, 1>
                          ascvec.get_buf mte2, %150, 0 {cannir.buffer.generated_sync}
                          "ascvec.copy_gm2ub_plan"(%158, %alloc_0, %c0_i32, %c0_i32, %c0_i64, %157, %70, %71, %c0_i64, %71, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xf16, 1>, memref<1xf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                          ascvec.release_buf mte2, %150, 0 {cannir.buffer.generated_sync}
                          %159 = ascscalar.memref_address %alloc_8 : memref<128xf16, 1>
                          %160 = ascscalar.memref_address %alloc_9 : memref<128xf16, 1>
                          %161 = arith.select %146, %159, %160 : i64
                          %162 = arith.select %146, %c4_i64, %c5_i64 : i64
                          %163 = cannir.extract_dim %82[1] : !cannir.stride<(-1, -1, 128, 1)>
                          %164 = arith.muli %125, %163 : i64
                          %165 = arith.muli %164, %c2_i64 : i64
                          %166 = arith.addi %85, %165 : i64
                          %167 = arith.addi %166, %65 : i64
                          %168 = ascscalar.bind_local_ptr %161 : memref<1xf16, 1>
                          ascvec.get_buf mte2, %162, 0 {cannir.buffer.generated_sync}
                          "ascvec.copy_gm2ub_plan"(%168, %alloc_1, %c0_i32, %c0_i32, %c0_i64, %167, %90, %91, %c0_i64, %91, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xf16, 1>, memref<1xf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                          ascvec.release_buf mte2, %162, 0 {cannir.buffer.generated_sync}
                          %169 = ascscalar.memref_address %alloc_10 : memref<128xf32, 1>
                          %170 = ascscalar.memref_address %alloc_11 : memref<128xf32, 1>
                          %171 = arith.select %146, %169, %170 : i64
                          %172 = arith.select %146, %c6_i64, %c7_i64 : i64
                          %173 = cannir.extract_dim %102[1] : !cannir.stride<(-1, -1, 128, 1)>
                          %174 = arith.muli %125, %173 : i64
                          %175 = arith.muli %174, %c4_i64 : i64
                          %176 = arith.addi %105, %175 : i64
                          %177 = arith.addi %176, %106 : i64
                          %178 = ascscalar.bind_local_ptr %171 : memref<1xf32, 1>
                          ascvec.get_buf mte2, %172, 0 {cannir.buffer.generated_sync}
                          "ascvec.copy_gm2ub_plan"(%178, %alloc_2, %c0_i32, %c0_i32, %c0_i64, %177, %111, %112, %c0_i64, %112, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = -1 : i64, ub_capacity = 4 : i64}> : (memref<1xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                          ascvec.release_buf mte2, %172, 0 {cannir.buffer.generated_sync}
                          scf.yield %152 : i64
                        } else {
                          scf.yield %arg6 : i64
                        }
                        %128 = arith.cmpi eq, %arg7, %c0_i64 : i64
                        %129 = ascscalar.memref_address %alloc_6 : memref<128xf16, 1>
                        %130 = ascscalar.memref_address %alloc_7 : memref<128xf16, 1>
                        %131 = arith.select %128, %129, %130 : i64
                        %132 = ascscalar.bind_local_ptr %131 : memref<128xf16, 1>
                        %133 = arith.select %128, %c2_i64, %c3_i64 : i64
                        %134 = arith.addi %arg7, %c1_i64 overflow<nsw> : i64
                        %135 = arith.remui %134, %c2_i64 : i64
                        %136 = ascscalar.memref_address %alloc_8 : memref<128xf16, 1>
                        %137 = ascscalar.memref_address %alloc_9 : memref<128xf16, 1>
                        %138 = arith.select %128, %136, %137 : i64
                        %139 = ascscalar.bind_local_ptr %138 : memref<128xf16, 1>
                        %140 = arith.select %128, %c4_i64, %c5_i64 : i64
                        %141 = ascscalar.memref_address %alloc_10 : memref<128xf32, 1>
                        %142 = ascscalar.memref_address %alloc_11 : memref<128xf32, 1>
                        %143 = arith.select %128, %141, %142 : i64
                        %144 = ascscalar.bind_local_ptr %143 : memref<128xf32, 1>
                        %145 = arith.select %128, %c6_i64, %c7_i64 : i64
                        ascvec.get_buf v, %133, 0 {cannir.buffer.generated_sync}
                        ascvec.vf(%132, %alloc_12 : memref<128xf16, 1>, memref<128xf32, 1>) {
                        ^bb0(%arg8: memref<128xf16, 1>, %arg9: memref<128xf32, 1>):
                          %c1_i32_13 = arith.constant 1 : i32
                          %c2_i32 = arith.constant 2 : i32
                          %c0_i32_14 = arith.constant 0 : i32
                          %c128_i32 = arith.constant 128 : i32
                          %c64_i32 = arith.constant 64 : i32
                          scf.for %arg10 = %c0_i32_14 to %c2_i32 step %c1_i32_13  : i32 {
                            %146 = arith.muli %arg10, %c64_i32 overflow<nsw> : i32
                            %147 = arith.subi %c128_i32, %146 overflow<nsw> : i32
                            %mask, %remaining_out = ascvec.update_mask %147 : !ascvec.mask
                            %148 = arith.index_cast %146 : i32 to index
                            %149 = ascvec.loadalign_unpack %arg8[%148] {mode = "b16_to_b32"} : memref<128xf16, 1>, index -> <f16>
                            %150 = ascvec.cast %149, %mask {rounding = 6 : i32} : <f16> -> <f32>
                            ascvec.storealign %arg9[%148], %150, %mask : memref<128xf32, 1>, index, <f32>
                          }
                        } {cannir.vf_mode = "simd"}
                        ascvec.release_buf v, %133, 0 {cannir.buffer.generated_sync}
                        ascvec.get_buf v, %140, 0 {cannir.buffer.generated_sync}
                        ascvec.get_buf v, %145, 0 {cannir.buffer.generated_sync}
                        ascvec.get_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
                        ascvec.vf(%139, %144, %alloc_12, %alloc_4 : memref<128xf16, 1>, memref<128xf32, 1>, memref<128xf32, 1>, memref<16384xf32, 1>) {
                        ^bb0(%arg8: memref<128xf16, 1>, %arg9: memref<128xf32, 1>, %arg10: memref<128xf32, 1>, %arg11: memref<16384xf32, 1>):
                          %c64 = arith.constant 64 : index
                          %c1_i32_13 = arith.constant 1 : i32
                          %c0_i32_14 = arith.constant 0 : i32
                          %c128_i32 = arith.constant 128 : i32
                          %c0_15 = arith.constant 0 : index
                          %c64_i32 = arith.constant 64 : i32
                          %mask, %remaining_out = ascvec.update_mask %c64_i32 : !ascvec.mask
                          %146 = ascvec.loadalign_unpack %arg8[%c0_15] {mode = "b16_to_b32"} : memref<128xf16, 1>, index -> <f16>
                          %147 = ascvec.loadalign_unpack %arg8[%c64] {mode = "b16_to_b32"} : memref<128xf16, 1>, index -> <f16>
                          %148 = ascvec.cast %146, %mask {rounding = 6 : i32} : <f16> -> <f32>
                          %149 = ascvec.cast %147, %mask {rounding = 6 : i32} : <f16> -> <f32>
                          %150 = ascvec.loadalign %arg9[%c0_15] : memref<128xf32, 1>, index -> <f32>
                          %151 = ascvec.loadalign %arg9[%c64] : memref<128xf32, 1>, index -> <f32>
                          scf.for %arg12 = %c0_i32_14 to %c128_i32 step %c1_i32_13  : i32 {
                            %152 = arith.muli %arg12, %c128_i32 overflow<nsw> : i32
                            %153 = arith.index_cast %arg12 : i32 to index
                            %154 = ascvec.loadalign_brc %arg10[%153] : memref<128xf32, 1>, index -> <f32>
                            %155 = arith.index_cast %152 : i32 to index
                            %156 = ascvec.loadalign %arg11[%155] : memref<16384xf32, 1>, index -> <f32>
                            %157 = arith.addi %152, %c64_i32 overflow<nsw> : i32
                            %158 = arith.index_cast %157 : i32 to index
                            %159 = ascvec.loadalign %arg11[%158] : memref<16384xf32, 1>, index -> <f32>
                            %160 = ascvec.mul %156, %150, %mask : <f32>
                            %161 = ascvec.mul %159, %151, %mask : <f32>
                            %162 = ascvec.copy %154, %mask : <f32>
                            %163 = ascvec.madd_two_address %162, %148, %160, %mask : <f32>
                            %164 = ascvec.madd_two_address %154, %149, %161, %mask : <f32>
                            ascvec.storealign %arg11[%155], %163, %mask : memref<16384xf32, 1>, index, <f32>
                            ascvec.storealign %arg11[%158], %164, %mask : memref<16384xf32, 1>, index, <f32>
                          } {cannir.loop_unroll = 4 : i64}
                        } {cannir.vf_mode = "simd"}
                        ascvec.release_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
                        ascvec.release_buf v, %145, 0 {cannir.buffer.generated_sync}
                        ascvec.release_buf v, %140, 0 {cannir.buffer.generated_sync}
                        scf.yield %127, %135 : i64, i64
                      } else {
                        scf.yield %arg6, %arg7 : i64, i64
                      }
                      scf.yield %124#0, %124#1 : i64, i64
                    }
                    %115 = arith.muli %42, %c16384_i64 : i64
                    %116 = arith.addi %50, %115 : i64
                    %117 = arith.muli %116, %c32_i64 : i64
                    %118 = ascscalar.memref_address %alloc : memref<1xf32>
                    %119 = arith.divui %117, %c8_i64 : i64
                    %120 = arith.addi %118, %119 : i64
                    %121 = ascscalar.bind_gm_ptr %120 : memref<1xf32>
                    %122 = ascscalar.bitcast_memref %alloc_4 byte_offset = %c0_i64 : memref<16384xf32, 1> -> memref<?xf32, 1>
                    ascvec.get_buf mte3, %c0_i64, 0 {cannir.buffer.generated_sync}
                    ascvec.copy_ub2gm %121[%c0], %122[%c0], %c1_i32, %c65536_i32, %c0_i64, %c65536_i32 l2_cache_ctl %c0_i32 : memref<1xf32>, memref<?xf32, 1>
                    ascvec.release_buf mte3, %c0_i64, 0 {cannir.buffer.generated_sync}
                  }
                }
              }
            }
          }
        } else {
          %9 = cannir.extract_layout %arg1 : !cannir.tensor<<gm, f16>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
          %10 = cannir.extract_shape %9 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, -1, 128)>
          %11 = cannir.extract_dim %10[1] : !cannir.shape<(-1, -1, -1, 128)>
          %12 = ascscalar.get_block_idx
          %13 = ascscalar.get_block_num
          %14 = arith.muli %4, %c2_i64 : i64
          %15 = arith.remsi %14, %13 : i64
          %16 = arith.cmpi ne, %15, %c0_i64 : i64
          %17 = arith.cmpi slt, %15, %c0_i64 : i64
          %18 = arith.cmpi slt, %13, %c0_i64 : i64
          %19 = arith.xori %17, %18 : i1
          %20 = arith.andi %16, %19 : i1
          %21 = arith.divsi %14, %13 : i64
          %22 = arith.extui %20 : i1 to i64
          %23 = arith.subi %21, %22 : i64
          %24 = arith.muli %23, %13 : i64
          %25 = arith.subi %14, %24 : i64
          %26 = arith.remsi %23, %c2_i64 : i64
          %27 = arith.cmpi ne, %26, %c0_i64 : i64
          %28 = arith.cmpi slt, %26, %c0_i64 : i64
          %29 = arith.andi %27, %28 : i1
          %30 = arith.divsi %23, %c2_i64 : i64
          %31 = arith.extui %29 : i1 to i64
          %32 = arith.subi %30, %31 : i64
          %33 = arith.addi %23, %c1_i64 : i64
          %34 = arith.remsi %33, %c2_i64 : i64
          %35 = arith.cmpi ne, %34, %c0_i64 : i64
          %36 = arith.cmpi slt, %34, %c0_i64 : i64
          %37 = arith.andi %35, %36 : i1
          %38 = arith.divsi %33, %c2_i64 : i64
          %39 = arith.extui %37 : i1 to i64
          %40 = arith.subi %38, %39 : i64
          %41 = arith.subi %40, %32 : i64
          %42 = arith.cmpi slt, %12, %25 : i64
          %43 = arith.select %42, %40, %32 : i64
          %44 = arith.select %42, %12, %25 : i64
          %45 = arith.muli %12, %32 : i64
          %46 = arith.muli %44, %41 : i64
          %47 = arith.addi %45, %46 : i64
          %48:2 = scf.for %arg5 = %c0_i64 to %43 step %c1_i64 iter_args(%arg6 = %c0_i64, %arg7 = %c0_i64) -> (i64, i64)  : i64 {
            %60 = arith.addi %47, %arg5 : i64
            %61 = arith.maxsi %3, %c1_i64 : i64
            %62 = arith.remsi %60, %61 : i64
            %63 = arith.cmpi slt, %62, %c0_i64 : i64
            %64 = arith.divsi %60, %61 : i64
            %65 = arith.extui %63 : i1 to i64
            %66 = arith.subi %64, %65 : i64
            %67 = arith.maxsi %2, %c1_i64 : i64
            %68 = arith.remsi %66, %67 : i64
            %69 = arith.cmpi slt, %68, %c0_i64 : i64
            %70 = arith.select %69, %67, %c0_i64 : i64
            %71 = arith.addi %68, %70 : i64
            %72 = arith.select %63, %61, %c0_i64 : i64
            %73 = arith.addi %62, %72 : i64
            %74 = arith.index_cast %71 : i64 to index
            %75 = memref.load %alloc_3[%74] : memref<1xi32>
            %76 = arith.extsi %75 : i32 to i64
            %77 = arith.cmpi sgt, %76, %c0_i64 : i64
            %78:2 = scf.if %77 -> (i64, i64) {
              %79 = cannir.extract_physical_layout %arg0 : !cannir.tensor<<gm, f32>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, identity>
              %80 = cannir.extract_stride %79 : <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>> -> <(-1, 16384, 128, 1)>
              %81 = cannir.extract_dim %80[0] : !cannir.stride<(-1, 16384, 128, 1)>
              %82 = arith.muli %71, %81 : i64
              %83 = arith.muli %82, %c4_i64 : i64
              %84 = arith.muli %73, %c65536_i64 : i64
              %85 = arith.addi %83, %84 : i64
              ascvec.get_buf mte2, %c0_i64, 0 {cannir.buffer.generated_sync}
              "ascvec.copy_gm2ub_plan"(%alloc_4, %alloc, %c0_i32, %c0_i32, %c0_i64, %85, %c1_i64, %c65536_i64, %c0_i64, %c65536_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 0 : i64, ub_capacity = 65536 : i64}> : (memref<16384xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
              ascvec.release_buf mte2, %c0_i64, 0 {cannir.buffer.generated_sync}
              %86 = arith.cmpi eq, %arg6, %c0_i64 : i64
              %87 = ascscalar.memref_address %alloc_6 : memref<128xf16, 1>
              %88 = ascscalar.memref_address %alloc_7 : memref<128xf16, 1>
              %89 = arith.select %86, %87, %88 : i64
              %90 = arith.select %86, %c2_i64, %c3_i64 : i64
              %91 = arith.addi %arg6, %c1_i64 : i64
              %92 = arith.remui %91, %c2_i64 : i64
              %93 = cannir.extract_dim %10[2] : !cannir.shape<(-1, -1, -1, 128)>
              %94 = arith.minsi %73, %93 : i64
              %95 = arith.subi %93, %94 : i64
              %96 = arith.minsi %95, %c1_i64 : i64
              %97 = arith.cmpi sle, %73, %93 : i64
              %98 = arith.select %97, %96, %c0_i64 : i64
              %99 = cannir.extract_physical_layout %arg1 : !cannir.tensor<<gm, f16>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
              %100 = cannir.extract_stride %99 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, 128, 1)>
              %101 = cannir.extract_dim %100[0] : !cannir.stride<(-1, -1, 128, 1)>
              %102 = arith.muli %71, %101 : i64
              %103 = arith.muli %102, %c2_i64 : i64
              %104 = arith.muli %73, %c256_i64 : i64
              %105 = arith.addi %103, %104 : i64
              %106 = arith.muli %98, %c256_i64 : i64
              %107 = arith.cmpi eq, %106, %c0_i64 : i64
              %108 = arith.cmpi ne, %106, %c0_i64 : i64
              %109 = arith.extui %108 : i1 to i64
              %110 = arith.select %107, %c2_i64, %106 : i64
              %111 = ascscalar.bind_local_ptr %89 : memref<1xf16, 1>
              ascvec.get_buf mte2, %90, 0 {cannir.buffer.generated_sync}
              "ascvec.copy_gm2ub_plan"(%111, %alloc_0, %c0_i32, %c0_i32, %c0_i64, %105, %109, %110, %c0_i64, %110, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xf16, 1>, memref<1xf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
              ascvec.release_buf mte2, %90, 0 {cannir.buffer.generated_sync}
              %112 = ascscalar.memref_address %alloc_8 : memref<128xf16, 1>
              %113 = ascscalar.memref_address %alloc_9 : memref<128xf16, 1>
              %114 = arith.select %86, %112, %113 : i64
              %115 = arith.select %86, %c4_i64, %c5_i64 : i64
              %116 = cannir.extract_layout %arg2 : !cannir.tensor<<gm, f16>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
              %117 = cannir.extract_shape %116 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, -1, 128)>
              %118 = cannir.extract_dim %117[2] : !cannir.shape<(-1, -1, -1, 128)>
              %119 = arith.minsi %73, %118 : i64
              %120 = arith.subi %118, %119 : i64
              %121 = arith.minsi %120, %c1_i64 : i64
              %122 = arith.cmpi sle, %73, %118 : i64
              %123 = arith.select %122, %121, %c0_i64 : i64
              %124 = cannir.extract_physical_layout %arg2 : !cannir.tensor<<gm, f16>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
              %125 = cannir.extract_stride %124 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, 128, 1)>
              %126 = cannir.extract_dim %125[0] : !cannir.stride<(-1, -1, 128, 1)>
              %127 = arith.muli %71, %126 : i64
              %128 = arith.muli %127, %c2_i64 : i64
              %129 = arith.addi %128, %104 : i64
              %130 = arith.muli %123, %c256_i64 : i64
              %131 = arith.cmpi eq, %130, %c0_i64 : i64
              %132 = arith.cmpi ne, %130, %c0_i64 : i64
              %133 = arith.extui %132 : i1 to i64
              %134 = arith.select %131, %c2_i64, %130 : i64
              %135 = ascscalar.bind_local_ptr %114 : memref<1xf16, 1>
              ascvec.get_buf mte2, %115, 0 {cannir.buffer.generated_sync}
              "ascvec.copy_gm2ub_plan"(%135, %alloc_1, %c0_i32, %c0_i32, %c0_i64, %129, %133, %134, %c0_i64, %134, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xf16, 1>, memref<1xf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
              ascvec.release_buf mte2, %115, 0 {cannir.buffer.generated_sync}
              %136 = ascscalar.memref_address %alloc_10 : memref<128xf32, 1>
              %137 = ascscalar.memref_address %alloc_11 : memref<128xf32, 1>
              %138 = arith.select %86, %136, %137 : i64
              %139 = arith.select %86, %c6_i64, %c7_i64 : i64
              %140 = cannir.extract_layout %arg3 : !cannir.tensor<<gm, f32>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
              %141 = cannir.extract_shape %140 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, -1, 128)>
              %142 = cannir.extract_dim %141[2] : !cannir.shape<(-1, -1, -1, 128)>
              %143 = arith.minsi %73, %142 : i64
              %144 = arith.subi %142, %143 : i64
              %145 = arith.minsi %144, %c1_i64 : i64
              %146 = arith.cmpi sle, %73, %142 : i64
              %147 = arith.select %146, %145, %c0_i64 : i64
              %148 = cannir.extract_physical_layout %arg3 : !cannir.tensor<<gm, f32>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
              %149 = cannir.extract_stride %148 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, 128, 1)>
              %150 = cannir.extract_dim %149[0] : !cannir.stride<(-1, -1, 128, 1)>
              %151 = arith.muli %71, %150 : i64
              %152 = arith.muli %151, %c4_i64 : i64
              %153 = arith.muli %73, %c512_i64 : i64
              %154 = arith.addi %152, %153 : i64
              %155 = arith.muli %147, %c512_i64 : i64
              %156 = arith.cmpi eq, %155, %c0_i64 : i64
              %157 = arith.cmpi ne, %155, %c0_i64 : i64
              %158 = arith.extui %157 : i1 to i64
              %159 = arith.select %156, %c4_i64, %155 : i64
              %160 = ascscalar.bind_local_ptr %138 : memref<1xf32, 1>
              ascvec.get_buf mte2, %139, 0 {cannir.buffer.generated_sync}
              "ascvec.copy_gm2ub_plan"(%160, %alloc_2, %c0_i32, %c0_i32, %c0_i64, %154, %158, %159, %c0_i64, %159, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = -1 : i64, ub_capacity = 4 : i64}> : (memref<1xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
              ascvec.release_buf mte2, %139, 0 {cannir.buffer.generated_sync}
              %161:2 = scf.for %arg8 = %c0_i64 to %11 step %c1_i64 iter_args(%arg9 = %92, %arg10 = %arg7) -> (i64, i64)  : i64 {
                %170 = arith.cmpi slt, %arg8, %76 : i64
                %171:2 = scf.if %170 -> (i64, i64) {
                  %172 = arith.addi %arg8, %c1_i64 : i64
                  %173 = arith.cmpi slt, %172, %76 : i64
                  %174 = scf.if %173 -> (i64) {
                    %187 = arith.cmpi eq, %arg9, %c0_i64 : i64
                    %188 = arith.select %187, %87, %88 : i64
                    %189 = arith.select %187, %c2_i64, %c3_i64 : i64
                    %190 = arith.addi %arg9, %c1_i64 overflow<nsw> : i64
                    %191 = arith.remui %190, %c2_i64 : i64
                    %192 = cannir.extract_dim %100[1] : !cannir.stride<(-1, -1, 128, 1)>
                    %193 = arith.muli %172, %192 : i64
                    %194 = arith.muli %193, %c2_i64 : i64
                    %195 = arith.addi %103, %194 : i64
                    %196 = arith.addi %195, %104 : i64
                    %197 = ascscalar.bind_local_ptr %188 : memref<1xf16, 1>
                    ascvec.get_buf mte2, %189, 0 {cannir.buffer.generated_sync}
                    "ascvec.copy_gm2ub_plan"(%197, %alloc_0, %c0_i32, %c0_i32, %c0_i64, %196, %109, %110, %c0_i64, %110, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xf16, 1>, memref<1xf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                    ascvec.release_buf mte2, %189, 0 {cannir.buffer.generated_sync}
                    %198 = arith.select %187, %112, %113 : i64
                    %199 = arith.select %187, %c4_i64, %c5_i64 : i64
                    %200 = cannir.extract_dim %125[1] : !cannir.stride<(-1, -1, 128, 1)>
                    %201 = arith.muli %172, %200 : i64
                    %202 = arith.muli %201, %c2_i64 : i64
                    %203 = arith.addi %128, %202 : i64
                    %204 = arith.addi %203, %104 : i64
                    %205 = ascscalar.bind_local_ptr %198 : memref<1xf16, 1>
                    ascvec.get_buf mte2, %199, 0 {cannir.buffer.generated_sync}
                    "ascvec.copy_gm2ub_plan"(%205, %alloc_1, %c0_i32, %c0_i32, %c0_i64, %204, %133, %134, %c0_i64, %134, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xf16, 1>, memref<1xf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                    ascvec.release_buf mte2, %199, 0 {cannir.buffer.generated_sync}
                    %206 = arith.select %187, %136, %137 : i64
                    %207 = arith.select %187, %c6_i64, %c7_i64 : i64
                    %208 = cannir.extract_dim %149[1] : !cannir.stride<(-1, -1, 128, 1)>
                    %209 = arith.muli %172, %208 : i64
                    %210 = arith.muli %209, %c4_i64 : i64
                    %211 = arith.addi %152, %210 : i64
                    %212 = arith.addi %211, %153 : i64
                    %213 = ascscalar.bind_local_ptr %206 : memref<1xf32, 1>
                    ascvec.get_buf mte2, %207, 0 {cannir.buffer.generated_sync}
                    "ascvec.copy_gm2ub_plan"(%213, %alloc_2, %c0_i32, %c0_i32, %c0_i64, %212, %158, %159, %c0_i64, %159, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = -1 : i64, ub_capacity = 4 : i64}> : (memref<1xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                    ascvec.release_buf mte2, %207, 0 {cannir.buffer.generated_sync}
                    scf.yield %191 : i64
                  } else {
                    scf.yield %arg9 : i64
                  }
                  %175 = arith.cmpi eq, %arg10, %c0_i64 : i64
                  %176 = arith.select %175, %87, %88 : i64
                  %177 = ascscalar.bind_local_ptr %176 : memref<128xf16, 1>
                  %178 = arith.select %175, %c2_i64, %c3_i64 : i64
                  %179 = arith.addi %arg10, %c1_i64 : i64
                  %180 = arith.remui %179, %c2_i64 : i64
                  %181 = arith.select %175, %112, %113 : i64
                  %182 = ascscalar.bind_local_ptr %181 : memref<128xf16, 1>
                  %183 = arith.select %175, %c4_i64, %c5_i64 : i64
                  %184 = arith.select %175, %136, %137 : i64
                  %185 = ascscalar.bind_local_ptr %184 : memref<128xf32, 1>
                  %186 = arith.select %175, %c6_i64, %c7_i64 : i64
                  ascvec.get_buf v, %178, 0 {cannir.buffer.generated_sync}
                  ascvec.vf(%177, %alloc_12 : memref<128xf16, 1>, memref<128xf32, 1>) {
                  ^bb0(%arg11: memref<128xf16, 1>, %arg12: memref<128xf32, 1>):
                    %c1_i32_13 = arith.constant 1 : i32
                    %c2_i32 = arith.constant 2 : i32
                    %c0_i32_14 = arith.constant 0 : i32
                    %c128_i32 = arith.constant 128 : i32
                    %c64_i32 = arith.constant 64 : i32
                    scf.for %arg13 = %c0_i32_14 to %c2_i32 step %c1_i32_13  : i32 {
                      %187 = arith.muli %arg13, %c64_i32 overflow<nsw> : i32
                      %188 = arith.subi %c128_i32, %187 overflow<nsw> : i32
                      %mask, %remaining_out = ascvec.update_mask %188 : !ascvec.mask
                      %189 = arith.index_cast %187 : i32 to index
                      %190 = ascvec.loadalign_unpack %arg11[%189] {mode = "b16_to_b32"} : memref<128xf16, 1>, index -> <f16>
                      %191 = ascvec.cast %190, %mask {rounding = 6 : i32} : <f16> -> <f32>
                      ascvec.storealign %arg12[%189], %191, %mask : memref<128xf32, 1>, index, <f32>
                    }
                  } {cannir.vf_mode = "simd"}
                  ascvec.release_buf v, %178, 0 {cannir.buffer.generated_sync}
                  ascvec.get_buf v, %183, 0 {cannir.buffer.generated_sync}
                  ascvec.get_buf v, %186, 0 {cannir.buffer.generated_sync}
                  ascvec.get_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
                  ascvec.vf(%182, %185, %alloc_12, %alloc_4 : memref<128xf16, 1>, memref<128xf32, 1>, memref<128xf32, 1>, memref<16384xf32, 1>) {
                  ^bb0(%arg11: memref<128xf16, 1>, %arg12: memref<128xf32, 1>, %arg13: memref<128xf32, 1>, %arg14: memref<16384xf32, 1>):
                    %c64 = arith.constant 64 : index
                    %c1_i32_13 = arith.constant 1 : i32
                    %c0_i32_14 = arith.constant 0 : i32
                    %c128_i32 = arith.constant 128 : i32
                    %c0_15 = arith.constant 0 : index
                    %c64_i32 = arith.constant 64 : i32
                    %mask, %remaining_out = ascvec.update_mask %c64_i32 : !ascvec.mask
                    %187 = ascvec.loadalign_unpack %arg11[%c0_15] {mode = "b16_to_b32"} : memref<128xf16, 1>, index -> <f16>
                    %188 = ascvec.loadalign_unpack %arg11[%c64] {mode = "b16_to_b32"} : memref<128xf16, 1>, index -> <f16>
                    %189 = ascvec.cast %187, %mask {rounding = 6 : i32} : <f16> -> <f32>
                    %190 = ascvec.cast %188, %mask {rounding = 6 : i32} : <f16> -> <f32>
                    %191 = ascvec.loadalign %arg12[%c0_15] : memref<128xf32, 1>, index -> <f32>
                    %192 = ascvec.loadalign %arg12[%c64] : memref<128xf32, 1>, index -> <f32>
                    scf.for %arg15 = %c0_i32_14 to %c128_i32 step %c1_i32_13  : i32 {
                      %193 = arith.muli %arg15, %c128_i32 overflow<nsw> : i32
                      %194 = arith.index_cast %arg15 : i32 to index
                      %195 = ascvec.loadalign_brc %arg13[%194] : memref<128xf32, 1>, index -> <f32>
                      %196 = arith.index_cast %193 : i32 to index
                      %197 = ascvec.loadalign %arg14[%196] : memref<16384xf32, 1>, index -> <f32>
                      %198 = arith.addi %193, %c64_i32 overflow<nsw> : i32
                      %199 = arith.index_cast %198 : i32 to index
                      %200 = ascvec.loadalign %arg14[%199] : memref<16384xf32, 1>, index -> <f32>
                      %201 = ascvec.mul %197, %191, %mask : <f32>
                      %202 = ascvec.mul %200, %192, %mask : <f32>
                      %203 = ascvec.copy %195, %mask : <f32>
                      %204 = ascvec.madd_two_address %203, %189, %201, %mask : <f32>
                      %205 = ascvec.madd_two_address %195, %190, %202, %mask : <f32>
                      ascvec.storealign %arg14[%196], %204, %mask : memref<16384xf32, 1>, index, <f32>
                      ascvec.storealign %arg14[%199], %205, %mask : memref<16384xf32, 1>, index, <f32>
                    } {cannir.loop_unroll = 4 : i64}
                  } {cannir.vf_mode = "simd"}
                  ascvec.release_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
                  ascvec.release_buf v, %186, 0 {cannir.buffer.generated_sync}
                  ascvec.release_buf v, %183, 0 {cannir.buffer.generated_sync}
                  scf.yield %174, %180 : i64, i64
                } else {
                  scf.yield %arg9, %arg10 : i64, i64
                }
                scf.yield %171#0, %171#1 : i64, i64
              }
              %162 = arith.muli %73, %c16384_i64 : i64
              %163 = arith.addi %82, %162 : i64
              %164 = arith.muli %163, %c32_i64 : i64
              %165 = ascscalar.memref_address %alloc : memref<1xf32>
              %166 = arith.divui %164, %c8_i64 : i64
              %167 = arith.addi %165, %166 : i64
              %168 = ascscalar.bind_gm_ptr %167 : memref<1xf32>
              %169 = ascscalar.bitcast_memref %alloc_4 byte_offset = %c0_i64 : memref<16384xf32, 1> -> memref<?xf32, 1>
              ascvec.get_buf mte3, %c0_i64, 0 {cannir.buffer.generated_sync}
              ascvec.copy_ub2gm %168[%c0], %169[%c0], %c1_i32, %c65536_i32, %c0_i64, %c65536_i32 l2_cache_ctl %c0_i32 : memref<1xf32>, memref<?xf32, 1>
              ascvec.release_buf mte3, %c0_i64, 0 {cannir.buffer.generated_sync}
              scf.yield %161#0, %161#1 : i64, i64
            } else {
              scf.yield %arg6, %arg7 : i64, i64
            }
            scf.yield %78#0, %78#1 : i64, i64
          }
          %49 = arith.muli %13, %32 : i64
          %50 = arith.muli %25, %41 : i64
          %51 = arith.addi %49, %50 : i64
          %52 = arith.select %29, %c2_i64, %c0_i64 : i64
          %53 = arith.addi %26, %52 overflow<nsw> : i64
          %54 = arith.cmpi eq, %53, %c0_i64 : i64
          %55 = arith.select %54, %42, %false : i1
          %56:4 = scf.if %55 -> (i64, i64, i64, i64) {
            %60 = arith.remsi %12, %c2_i64 : i64
            %61 = arith.cmpi ne, %60, %c0_i64 : i64
            %62 = arith.cmpi slt, %60, %c0_i64 : i64
            %63 = arith.andi %61, %62 : i1
            %64 = arith.divsi %12, %c2_i64 : i64
            %65 = arith.extui %63 : i1 to i64
            %66 = arith.subi %64, %65 : i64
            %67 = arith.addi %51, %66 : i64
            %68 = arith.select %63, %c2_i64, %c0_i64 : i64
            %69 = arith.addi %60, %68 overflow<nsw> : i64
            %70 = arith.maxsi %3, %c1_i64 : i64
            %71 = arith.remsi %67, %70 : i64
            %72 = arith.cmpi slt, %71, %c0_i64 : i64
            %73 = arith.divsi %67, %70 : i64
            %74 = arith.extui %72 : i1 to i64
            %75 = arith.subi %73, %74 : i64
            %76 = arith.maxsi %2, %c1_i64 : i64
            %77 = arith.remsi %75, %76 : i64
            %78 = arith.cmpi slt, %77, %c0_i64 : i64
            %79 = arith.select %78, %76, %c0_i64 : i64
            %80 = arith.addi %77, %79 : i64
            %81 = arith.select %72, %70, %c0_i64 : i64
            %82 = arith.addi %71, %81 : i64
            %83 = arith.index_cast %80 : i64 to index
            %84 = memref.load %alloc_3[%83] : memref<1xi32>
            %85 = arith.extsi %84 : i32 to i64
            %86 = arith.cmpi sgt, %85, %c0_i64 : i64
            %87:4 = scf.if %86 -> (i64, i64, i64, i64) {
              %88 = cannir.extract_physical_layout %arg0 : !cannir.tensor<<gm, f32>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, identity>
              %89 = cannir.extract_stride %88 : <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>> -> <(-1, 16384, 128, 1)>
              %90 = cannir.extract_dim %89[0] : !cannir.stride<(-1, 16384, 128, 1)>
              %91 = arith.muli %80, %90 : i64
              %92 = arith.muli %91, %c4_i64 : i64
              %93 = arith.muli %82, %c65536_i64 : i64
              %94 = arith.addi %92, %93 : i64
              %95 = arith.muli %69, %c32768_i64 overflow<nsw> : i64
              %96 = arith.addi %94, %95 : i64
              ascvec.get_buf mte2, %c0_i64, 0 {cannir.buffer.generated_sync}
              "ascvec.copy_gm2ub_plan"(%alloc_4, %alloc, %c0_i32, %c0_i32, %c0_i64, %96, %c1_i64, %c32768_i64, %c0_i64, %c32768_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 0 : i64, ub_capacity = 65536 : i64}> : (memref<16384xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
              ascvec.release_buf mte2, %c0_i64, 0 {cannir.buffer.generated_sync}
              %97 = arith.cmpi eq, %48#0, %c0_i64 : i64
              %98 = ascscalar.memref_address %alloc_6 : memref<128xf16, 1>
              %99 = ascscalar.memref_address %alloc_7 : memref<128xf16, 1>
              %100 = arith.select %97, %98, %99 : i64
              %101 = arith.select %97, %c2_i64, %c3_i64 : i64
              %102 = arith.addi %48#0, %c1_i64 : i64
              %103 = arith.remui %102, %c2_i64 : i64
              %104 = cannir.extract_dim %10[2] : !cannir.shape<(-1, -1, -1, 128)>
              %105 = arith.minsi %82, %104 : i64
              %106 = arith.subi %104, %105 : i64
              %107 = arith.minsi %106, %c1_i64 : i64
              %108 = arith.cmpi sle, %82, %104 : i64
              %109 = arith.select %108, %107, %c0_i64 : i64
              %110 = cannir.extract_physical_layout %arg1 : !cannir.tensor<<gm, f16>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
              %111 = cannir.extract_stride %110 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, 128, 1)>
              %112 = cannir.extract_dim %111[0] : !cannir.stride<(-1, -1, 128, 1)>
              %113 = arith.muli %80, %112 : i64
              %114 = arith.muli %113, %c2_i64 : i64
              %115 = arith.muli %82, %c256_i64 : i64
              %116 = arith.addi %114, %115 : i64
              %117 = arith.muli %69, %c128_i64 overflow<nsw> : i64
              %118 = arith.addi %116, %117 : i64
              %119 = ascscalar.bind_local_ptr %100 : memref<1xf16, 1>
              ascvec.get_buf mte2, %101, 0 {cannir.buffer.generated_sync}
              "ascvec.copy_gm2ub_plan"(%119, %alloc_0, %c0_i32, %c0_i32, %c0_i64, %118, %109, %c128_i64, %c256_i64, %c128_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xf16, 1>, memref<1xf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
              ascvec.release_buf mte2, %101, 0 {cannir.buffer.generated_sync}
              %120 = ascscalar.memref_address %alloc_8 : memref<128xf16, 1>
              %121 = ascscalar.memref_address %alloc_9 : memref<128xf16, 1>
              %122 = arith.select %97, %120, %121 : i64
              %123 = arith.select %97, %c4_i64, %c5_i64 : i64
              %124 = cannir.extract_layout %arg2 : !cannir.tensor<<gm, f16>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
              %125 = cannir.extract_shape %124 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, -1, 128)>
              %126 = cannir.extract_dim %125[2] : !cannir.shape<(-1, -1, -1, 128)>
              %127 = arith.minsi %82, %126 : i64
              %128 = arith.subi %126, %127 : i64
              %129 = arith.minsi %128, %c1_i64 : i64
              %130 = arith.cmpi sle, %82, %126 : i64
              %131 = arith.select %130, %129, %c0_i64 : i64
              %132 = cannir.extract_physical_layout %arg2 : !cannir.tensor<<gm, f16>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
              %133 = cannir.extract_stride %132 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, 128, 1)>
              %134 = cannir.extract_dim %133[0] : !cannir.stride<(-1, -1, 128, 1)>
              %135 = arith.muli %80, %134 : i64
              %136 = arith.muli %135, %c2_i64 : i64
              %137 = arith.addi %136, %115 : i64
              %138 = arith.muli %131, %c256_i64 : i64
              %139 = arith.cmpi eq, %138, %c0_i64 : i64
              %140 = arith.cmpi ne, %138, %c0_i64 : i64
              %141 = arith.extui %140 : i1 to i64
              %142 = arith.select %139, %c2_i64, %138 : i64
              %143 = ascscalar.bind_local_ptr %122 : memref<1xf16, 1>
              ascvec.get_buf mte2, %123, 0 {cannir.buffer.generated_sync}
              "ascvec.copy_gm2ub_plan"(%143, %alloc_1, %c0_i32, %c0_i32, %c0_i64, %137, %141, %142, %c0_i64, %142, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xf16, 1>, memref<1xf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
              ascvec.release_buf mte2, %123, 0 {cannir.buffer.generated_sync}
              %144 = ascscalar.memref_address %alloc_10 : memref<128xf32, 1>
              %145 = ascscalar.memref_address %alloc_11 : memref<128xf32, 1>
              %146 = arith.select %97, %144, %145 : i64
              %147 = arith.select %97, %c6_i64, %c7_i64 : i64
              %148 = cannir.extract_layout %arg3 : !cannir.tensor<<gm, f32>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
              %149 = cannir.extract_shape %148 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, -1, 128)>
              %150 = cannir.extract_dim %149[2] : !cannir.shape<(-1, -1, -1, 128)>
              %151 = arith.minsi %82, %150 : i64
              %152 = arith.subi %150, %151 : i64
              %153 = arith.minsi %152, %c1_i64 : i64
              %154 = arith.cmpi sle, %82, %150 : i64
              %155 = arith.select %154, %153, %c0_i64 : i64
              %156 = cannir.extract_physical_layout %arg3 : !cannir.tensor<<gm, f32>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
              %157 = cannir.extract_stride %156 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, 128, 1)>
              %158 = cannir.extract_dim %157[0] : !cannir.stride<(-1, -1, 128, 1)>
              %159 = arith.muli %80, %158 : i64
              %160 = arith.muli %159, %c4_i64 : i64
              %161 = arith.muli %82, %c512_i64 : i64
              %162 = arith.addi %160, %161 : i64
              %163 = arith.muli %155, %c512_i64 : i64
              %164 = arith.cmpi eq, %163, %c0_i64 : i64
              %165 = arith.cmpi ne, %163, %c0_i64 : i64
              %166 = arith.extui %165 : i1 to i64
              %167 = arith.select %164, %c4_i64, %163 : i64
              %168 = ascscalar.bind_local_ptr %146 : memref<1xf32, 1>
              ascvec.get_buf mte2, %147, 0 {cannir.buffer.generated_sync}
              "ascvec.copy_gm2ub_plan"(%168, %alloc_2, %c0_i32, %c0_i32, %c0_i64, %162, %166, %167, %c0_i64, %167, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = -1 : i64, ub_capacity = 4 : i64}> : (memref<1xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
              ascvec.release_buf mte2, %147, 0 {cannir.buffer.generated_sync}
              %169:2 = scf.for %arg5 = %c0_i64 to %11 step %c1_i64 iter_args(%arg6 = %103, %arg7 = %48#1) -> (i64, i64)  : i64 {
                %180 = arith.cmpi slt, %arg5, %85 : i64
                %181:2 = scf.if %180 -> (i64, i64) {
                  %182 = arith.addi %arg5, %c1_i64 : i64
                  %183 = arith.cmpi slt, %182, %85 : i64
                  %184 = scf.if %183 -> (i64) {
                    %197 = arith.cmpi eq, %arg6, %c0_i64 : i64
                    %198 = arith.select %197, %98, %99 : i64
                    %199 = arith.select %197, %c2_i64, %c3_i64 : i64
                    %200 = arith.addi %arg6, %c1_i64 overflow<nsw> : i64
                    %201 = arith.remui %200, %c2_i64 : i64
                    %202 = cannir.extract_dim %111[1] : !cannir.stride<(-1, -1, 128, 1)>
                    %203 = arith.muli %182, %202 : i64
                    %204 = arith.muli %203, %c2_i64 : i64
                    %205 = arith.addi %114, %204 : i64
                    %206 = arith.addi %205, %115 : i64
                    %207 = arith.addi %206, %117 : i64
                    %208 = ascscalar.bind_local_ptr %198 : memref<1xf16, 1>
                    ascvec.get_buf mte2, %199, 0 {cannir.buffer.generated_sync}
                    "ascvec.copy_gm2ub_plan"(%208, %alloc_0, %c0_i32, %c0_i32, %c0_i64, %207, %109, %c128_i64, %c256_i64, %c128_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xf16, 1>, memref<1xf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                    ascvec.release_buf mte2, %199, 0 {cannir.buffer.generated_sync}
                    %209 = arith.select %197, %120, %121 : i64
                    %210 = arith.select %197, %c4_i64, %c5_i64 : i64
                    %211 = cannir.extract_dim %133[1] : !cannir.stride<(-1, -1, 128, 1)>
                    %212 = arith.muli %182, %211 : i64
                    %213 = arith.muli %212, %c2_i64 : i64
                    %214 = arith.addi %136, %213 : i64
                    %215 = arith.addi %214, %115 : i64
                    %216 = ascscalar.bind_local_ptr %209 : memref<1xf16, 1>
                    ascvec.get_buf mte2, %210, 0 {cannir.buffer.generated_sync}
                    "ascvec.copy_gm2ub_plan"(%216, %alloc_1, %c0_i32, %c0_i32, %c0_i64, %215, %141, %142, %c0_i64, %142, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xf16, 1>, memref<1xf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                    ascvec.release_buf mte2, %210, 0 {cannir.buffer.generated_sync}
                    %217 = arith.select %197, %144, %145 : i64
                    %218 = arith.select %197, %c6_i64, %c7_i64 : i64
                    %219 = cannir.extract_dim %157[1] : !cannir.stride<(-1, -1, 128, 1)>
                    %220 = arith.muli %182, %219 : i64
                    %221 = arith.muli %220, %c4_i64 : i64
                    %222 = arith.addi %160, %221 : i64
                    %223 = arith.addi %222, %161 : i64
                    %224 = ascscalar.bind_local_ptr %217 : memref<1xf32, 1>
                    ascvec.get_buf mte2, %218, 0 {cannir.buffer.generated_sync}
                    "ascvec.copy_gm2ub_plan"(%224, %alloc_2, %c0_i32, %c0_i32, %c0_i64, %223, %166, %167, %c0_i64, %167, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = -1 : i64, ub_capacity = 4 : i64}> : (memref<1xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                    ascvec.release_buf mte2, %218, 0 {cannir.buffer.generated_sync}
                    scf.yield %201 : i64
                  } else {
                    scf.yield %arg6 : i64
                  }
                  %185 = arith.cmpi eq, %arg7, %c0_i64 : i64
                  %186 = arith.select %185, %98, %99 : i64
                  %187 = ascscalar.bind_local_ptr %186 : memref<128xf16, 1>
                  %188 = arith.select %185, %c2_i64, %c3_i64 : i64
                  %189 = arith.addi %arg7, %c1_i64 : i64
                  %190 = arith.remui %189, %c2_i64 : i64
                  %191 = arith.select %185, %120, %121 : i64
                  %192 = ascscalar.bind_local_ptr %191 : memref<128xf16, 1>
                  %193 = arith.select %185, %c4_i64, %c5_i64 : i64
                  %194 = arith.select %185, %144, %145 : i64
                  %195 = ascscalar.bind_local_ptr %194 : memref<128xf32, 1>
                  %196 = arith.select %185, %c6_i64, %c7_i64 : i64
                  ascvec.get_buf v, %188, 0 {cannir.buffer.generated_sync}
                  ascvec.vf(%187, %alloc_12 : memref<128xf16, 1>, memref<128xf32, 1>) {
                  ^bb0(%arg8: memref<128xf16, 1>, %arg9: memref<128xf32, 1>):
                    %c0_13 = arith.constant 0 : index
                    %c64_i32 = arith.constant 64 : i32
                    %c0_14 = arith.constant 0 : index
                    %c1 = arith.constant 1 : index
                    scf.for %arg10 = %c0_14 to %c1 step %c1 {
                      %mask, %remaining_out = ascvec.update_mask %c64_i32 : !ascvec.mask
                      %197 = ascvec.loadalign_unpack %arg8[%c0_13] {mode = "b16_to_b32"} : memref<128xf16, 1>, index -> <f16>
                      %198 = ascvec.cast %197, %mask {rounding = 6 : i32} : <f16> -> <f32>
                      ascvec.storealign %arg9[%c0_13], %198, %mask : memref<128xf32, 1>, index, <f32>
                    }
                  } {cannir.vf_mode = "simd"}
                  ascvec.release_buf v, %188, 0 {cannir.buffer.generated_sync}
                  ascvec.get_buf v, %193, 0 {cannir.buffer.generated_sync}
                  ascvec.get_buf v, %196, 0 {cannir.buffer.generated_sync}
                  ascvec.get_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
                  ascvec.vf(%192, %195, %alloc_12, %alloc_4 : memref<128xf16, 1>, memref<128xf32, 1>, memref<128xf32, 1>, memref<16384xf32, 1>) {
                  ^bb0(%arg8: memref<128xf16, 1>, %arg9: memref<128xf32, 1>, %arg10: memref<128xf32, 1>, %arg11: memref<16384xf32, 1>):
                    %c64 = arith.constant 64 : index
                    %c1_i32_13 = arith.constant 1 : i32
                    %c0_i32_14 = arith.constant 0 : i32
                    %c128_i32 = arith.constant 128 : i32
                    %c0_15 = arith.constant 0 : index
                    %c64_i32 = arith.constant 64 : i32
                    %mask, %remaining_out = ascvec.update_mask %c64_i32 : !ascvec.mask
                    %197 = ascvec.loadalign_unpack %arg8[%c0_15] {mode = "b16_to_b32"} : memref<128xf16, 1>, index -> <f16>
                    %198 = ascvec.loadalign_unpack %arg8[%c64] {mode = "b16_to_b32"} : memref<128xf16, 1>, index -> <f16>
                    %199 = ascvec.cast %197, %mask {rounding = 6 : i32} : <f16> -> <f32>
                    %200 = ascvec.cast %198, %mask {rounding = 6 : i32} : <f16> -> <f32>
                    %201 = ascvec.loadalign %arg9[%c0_15] : memref<128xf32, 1>, index -> <f32>
                    %202 = ascvec.loadalign %arg9[%c64] : memref<128xf32, 1>, index -> <f32>
                    scf.for %arg12 = %c0_i32_14 to %c64_i32 step %c1_i32_13  : i32 {
                      %203 = arith.muli %arg12, %c128_i32 overflow<nsw> : i32
                      %204 = arith.index_cast %arg12 : i32 to index
                      %205 = ascvec.loadalign_brc %arg10[%204] : memref<128xf32, 1>, index -> <f32>
                      %206 = arith.index_cast %203 : i32 to index
                      %207 = ascvec.loadalign %arg11[%206] : memref<16384xf32, 1>, index -> <f32>
                      %208 = arith.addi %203, %c64_i32 overflow<nsw> : i32
                      %209 = arith.index_cast %208 : i32 to index
                      %210 = ascvec.loadalign %arg11[%209] : memref<16384xf32, 1>, index -> <f32>
                      %211 = ascvec.mul %207, %201, %mask : <f32>
                      %212 = ascvec.mul %210, %202, %mask : <f32>
                      %213 = ascvec.copy %205, %mask : <f32>
                      %214 = ascvec.madd_two_address %213, %199, %211, %mask : <f32>
                      %215 = ascvec.madd_two_address %205, %200, %212, %mask : <f32>
                      ascvec.storealign %arg11[%206], %214, %mask : memref<16384xf32, 1>, index, <f32>
                      ascvec.storealign %arg11[%209], %215, %mask : memref<16384xf32, 1>, index, <f32>
                    } {cannir.loop_unroll = 4 : i64}
                  } {cannir.vf_mode = "simd"}
                  ascvec.release_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
                  ascvec.release_buf v, %196, 0 {cannir.buffer.generated_sync}
                  ascvec.release_buf v, %193, 0 {cannir.buffer.generated_sync}
                  scf.yield %184, %190 : i64, i64
                } else {
                  scf.yield %arg6, %arg7 : i64, i64
                }
                scf.yield %181#0, %181#1 : i64, i64
              }
              %170 = arith.muli %82, %c16384_i64 : i64
              %171 = arith.addi %91, %170 : i64
              %172 = arith.muli %69, %c8192_i64 overflow<nsw> : i64
              %173 = arith.addi %171, %172 : i64
              %174 = arith.muli %173, %c32_i64 : i64
              %175 = ascscalar.memref_address %alloc : memref<1xf32>
              %176 = arith.divui %174, %c8_i64 : i64
              %177 = arith.addi %175, %176 : i64
              %178 = ascscalar.bind_gm_ptr %177 : memref<1xf32>
              %179 = ascscalar.bitcast_memref %alloc_4 byte_offset = %c0_i64 : memref<16384xf32, 1> -> memref<?xf32, 1>
              ascvec.get_buf mte3, %c0_i64, 0 {cannir.buffer.generated_sync}
              ascvec.copy_ub2gm %178[%c0], %179[%c0], %c1_i32, %c32768_i32, %c0_i64, %c32768_i32 l2_cache_ctl %c0_i32 : memref<1xf32>, memref<?xf32, 1>
              ascvec.release_buf mte3, %c0_i64, 0 {cannir.buffer.generated_sync}
              scf.yield %169#0, %169#1, %169#0, %169#0 : i64, i64, i64, i64
            } else {
              scf.yield %48#0, %48#1, %48#0, %48#0 : i64, i64, i64, i64
            }
            scf.yield %87#0, %87#1, %87#2, %87#3 : i64, i64, i64, i64
          } else {
            scf.yield %48#0, %48#1, %48#0, %48#0 : i64, i64, i64, i64
          }
          %57 = arith.cmpi eq, %53, %c1_i64 : i64
          %58 = arith.cmpi sge, %12, %25 : i64
          %59 = arith.select %57, %58, %false : i1
          scf.if %59 {
            %60 = arith.subi %12, %25 : i64
            %61 = arith.remsi %60, %c2_i64 : i64
            %62 = arith.cmpi ne, %61, %c0_i64 : i64
            %63 = arith.cmpi slt, %61, %c0_i64 : i64
            %64 = arith.andi %62, %63 : i1
            %65 = arith.divsi %60, %c2_i64 : i64
            %66 = arith.extui %64 : i1 to i64
            %67 = arith.subi %65, %66 : i64
            %68 = arith.addi %51, %67 : i64
            %69 = arith.select %64, %c2_i64, %c0_i64 : i64
            %70 = arith.addi %61, %69 overflow<nsw> : i64
            %71 = arith.maxsi %3, %c1_i64 : i64
            %72 = arith.remsi %68, %71 : i64
            %73 = arith.cmpi slt, %72, %c0_i64 : i64
            %74 = arith.divsi %68, %71 : i64
            %75 = arith.extui %73 : i1 to i64
            %76 = arith.subi %74, %75 : i64
            %77 = arith.maxsi %2, %c1_i64 : i64
            %78 = arith.remsi %76, %77 : i64
            %79 = arith.cmpi slt, %78, %c0_i64 : i64
            %80 = arith.select %79, %77, %c0_i64 : i64
            %81 = arith.addi %78, %80 : i64
            %82 = arith.select %73, %71, %c0_i64 : i64
            %83 = arith.addi %72, %82 : i64
            %84 = arith.index_cast %81 : i64 to index
            %85 = memref.load %alloc_3[%84] : memref<1xi32>
            %86 = arith.extsi %85 : i32 to i64
            %87 = arith.cmpi sgt, %86, %c0_i64 : i64
            scf.if %87 {
              %88 = cannir.extract_physical_layout %arg0 : !cannir.tensor<<gm, f32>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, identity>
              %89 = cannir.extract_stride %88 : <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>> -> <(-1, 16384, 128, 1)>
              %90 = cannir.extract_dim %89[0] : !cannir.stride<(-1, 16384, 128, 1)>
              %91 = arith.muli %81, %90 : i64
              %92 = arith.muli %91, %c4_i64 : i64
              %93 = arith.muli %83, %c65536_i64 : i64
              %94 = arith.addi %92, %93 : i64
              %95 = arith.muli %70, %c32768_i64 overflow<nsw> : i64
              %96 = arith.addi %94, %95 : i64
              ascvec.get_buf mte2, %c0_i64, 0 {cannir.buffer.generated_sync}
              "ascvec.copy_gm2ub_plan"(%alloc_4, %alloc, %c0_i32, %c0_i32, %c0_i64, %96, %c1_i64, %c32768_i64, %c0_i64, %c32768_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 0 : i64, ub_capacity = 65536 : i64}> : (memref<16384xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
              ascvec.release_buf mte2, %c0_i64, 0 {cannir.buffer.generated_sync}
              %97 = arith.cmpi eq, %56#0, %c0_i64 : i64
              %98 = ascscalar.memref_address %alloc_6 : memref<128xf16, 1>
              %99 = ascscalar.memref_address %alloc_7 : memref<128xf16, 1>
              %100 = arith.select %97, %98, %99 : i64
              %101 = arith.select %97, %c2_i64, %c3_i64 : i64
              %102 = arith.addi %56#0, %c1_i64 : i64
              %103 = arith.remui %102, %c2_i64 : i64
              %104 = cannir.extract_dim %10[2] : !cannir.shape<(-1, -1, -1, 128)>
              %105 = arith.minsi %83, %104 : i64
              %106 = arith.subi %104, %105 : i64
              %107 = arith.minsi %106, %c1_i64 : i64
              %108 = arith.cmpi sle, %83, %104 : i64
              %109 = arith.select %108, %107, %c0_i64 : i64
              %110 = cannir.extract_physical_layout %arg1 : !cannir.tensor<<gm, f16>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
              %111 = cannir.extract_stride %110 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, 128, 1)>
              %112 = cannir.extract_dim %111[0] : !cannir.stride<(-1, -1, 128, 1)>
              %113 = arith.muli %81, %112 : i64
              %114 = arith.muli %113, %c2_i64 : i64
              %115 = arith.muli %83, %c256_i64 : i64
              %116 = arith.addi %114, %115 : i64
              %117 = arith.muli %70, %c128_i64 overflow<nsw> : i64
              %118 = arith.addi %116, %117 : i64
              %119 = ascscalar.bind_local_ptr %100 : memref<1xf16, 1>
              ascvec.get_buf mte2, %101, 0 {cannir.buffer.generated_sync}
              "ascvec.copy_gm2ub_plan"(%119, %alloc_0, %c0_i32, %c0_i32, %c0_i64, %118, %109, %c128_i64, %c256_i64, %c128_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xf16, 1>, memref<1xf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
              ascvec.release_buf mte2, %101, 0 {cannir.buffer.generated_sync}
              %120 = arith.cmpi eq, %56#2, %c0_i64 : i64
              %121 = ascscalar.memref_address %alloc_8 : memref<128xf16, 1>
              %122 = ascscalar.memref_address %alloc_9 : memref<128xf16, 1>
              %123 = arith.select %120, %121, %122 : i64
              %124 = arith.select %120, %c4_i64, %c5_i64 : i64
              %125 = cannir.extract_layout %arg2 : !cannir.tensor<<gm, f16>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
              %126 = cannir.extract_shape %125 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, -1, 128)>
              %127 = cannir.extract_dim %126[2] : !cannir.shape<(-1, -1, -1, 128)>
              %128 = arith.minsi %83, %127 : i64
              %129 = arith.subi %127, %128 : i64
              %130 = arith.minsi %129, %c1_i64 : i64
              %131 = arith.cmpi sle, %83, %127 : i64
              %132 = arith.select %131, %130, %c0_i64 : i64
              %133 = cannir.extract_physical_layout %arg2 : !cannir.tensor<<gm, f16>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
              %134 = cannir.extract_stride %133 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, 128, 1)>
              %135 = cannir.extract_dim %134[0] : !cannir.stride<(-1, -1, 128, 1)>
              %136 = arith.muli %81, %135 : i64
              %137 = arith.muli %136, %c2_i64 : i64
              %138 = arith.addi %137, %115 : i64
              %139 = arith.muli %132, %c256_i64 : i64
              %140 = arith.cmpi eq, %139, %c0_i64 : i64
              %141 = arith.cmpi ne, %139, %c0_i64 : i64
              %142 = arith.extui %141 : i1 to i64
              %143 = arith.select %140, %c2_i64, %139 : i64
              %144 = ascscalar.bind_local_ptr %123 : memref<1xf16, 1>
              ascvec.get_buf mte2, %124, 0 {cannir.buffer.generated_sync}
              "ascvec.copy_gm2ub_plan"(%144, %alloc_1, %c0_i32, %c0_i32, %c0_i64, %138, %142, %143, %c0_i64, %143, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xf16, 1>, memref<1xf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
              ascvec.release_buf mte2, %124, 0 {cannir.buffer.generated_sync}
              %145 = arith.cmpi eq, %56#3, %c0_i64 : i64
              %146 = ascscalar.memref_address %alloc_10 : memref<128xf32, 1>
              %147 = ascscalar.memref_address %alloc_11 : memref<128xf32, 1>
              %148 = arith.select %145, %146, %147 : i64
              %149 = arith.select %145, %c6_i64, %c7_i64 : i64
              %150 = cannir.extract_layout %arg3 : !cannir.tensor<<gm, f32>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
              %151 = cannir.extract_shape %150 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, -1, 128)>
              %152 = cannir.extract_dim %151[2] : !cannir.shape<(-1, -1, -1, 128)>
              %153 = arith.minsi %83, %152 : i64
              %154 = arith.subi %152, %153 : i64
              %155 = arith.minsi %154, %c1_i64 : i64
              %156 = arith.cmpi sle, %83, %152 : i64
              %157 = arith.select %156, %155, %c0_i64 : i64
              %158 = cannir.extract_physical_layout %arg3 : !cannir.tensor<<gm, f32>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>>, identity>
              %159 = cannir.extract_stride %158 : <<(-1, -1, -1, 128)>, <(-1, -1, 128, 1)>> -> <(-1, -1, 128, 1)>
              %160 = cannir.extract_dim %159[0] : !cannir.stride<(-1, -1, 128, 1)>
              %161 = arith.muli %81, %160 : i64
              %162 = arith.muli %161, %c4_i64 : i64
              %163 = arith.muli %83, %c512_i64 : i64
              %164 = arith.addi %162, %163 : i64
              %165 = arith.muli %157, %c512_i64 : i64
              %166 = arith.cmpi eq, %165, %c0_i64 : i64
              %167 = arith.cmpi ne, %165, %c0_i64 : i64
              %168 = arith.extui %167 : i1 to i64
              %169 = arith.select %166, %c4_i64, %165 : i64
              %170 = ascscalar.bind_local_ptr %148 : memref<1xf32, 1>
              ascvec.get_buf mte2, %149, 0 {cannir.buffer.generated_sync}
              "ascvec.copy_gm2ub_plan"(%170, %alloc_2, %c0_i32, %c0_i32, %c0_i64, %164, %168, %169, %c0_i64, %169, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = -1 : i64, ub_capacity = 4 : i64}> : (memref<1xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
              ascvec.release_buf mte2, %149, 0 {cannir.buffer.generated_sync}
              %171:2 = scf.for %arg5 = %c0_i64 to %11 step %c1_i64 iter_args(%arg6 = %103, %arg7 = %56#1) -> (i64, i64)  : i64 {
                %182 = arith.cmpi slt, %arg5, %86 : i64
                %183:2 = scf.if %182 -> (i64, i64) {
                  %184 = arith.addi %arg5, %c1_i64 : i64
                  %185 = arith.cmpi slt, %184, %86 : i64
                  %186 = scf.if %185 -> (i64) {
                    %199 = arith.cmpi eq, %arg6, %c0_i64 : i64
                    %200 = arith.select %199, %98, %99 : i64
                    %201 = arith.select %199, %c2_i64, %c3_i64 : i64
                    %202 = arith.addi %arg6, %c1_i64 overflow<nsw> : i64
                    %203 = arith.remui %202, %c2_i64 : i64
                    %204 = cannir.extract_dim %111[1] : !cannir.stride<(-1, -1, 128, 1)>
                    %205 = arith.muli %184, %204 : i64
                    %206 = arith.muli %205, %c2_i64 : i64
                    %207 = arith.addi %114, %206 : i64
                    %208 = arith.addi %207, %115 : i64
                    %209 = arith.addi %208, %117 : i64
                    %210 = ascscalar.bind_local_ptr %200 : memref<1xf16, 1>
                    ascvec.get_buf mte2, %201, 0 {cannir.buffer.generated_sync}
                    "ascvec.copy_gm2ub_plan"(%210, %alloc_0, %c0_i32, %c0_i32, %c0_i64, %209, %109, %c128_i64, %c256_i64, %c128_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xf16, 1>, memref<1xf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                    ascvec.release_buf mte2, %201, 0 {cannir.buffer.generated_sync}
                    %211 = arith.select %199, %121, %122 : i64
                    %212 = arith.select %199, %c4_i64, %c5_i64 : i64
                    %213 = cannir.extract_dim %134[1] : !cannir.stride<(-1, -1, 128, 1)>
                    %214 = arith.muli %184, %213 : i64
                    %215 = arith.muli %214, %c2_i64 : i64
                    %216 = arith.addi %137, %215 : i64
                    %217 = arith.addi %216, %115 : i64
                    %218 = ascscalar.bind_local_ptr %211 : memref<1xf16, 1>
                    ascvec.get_buf mte2, %212, 0 {cannir.buffer.generated_sync}
                    "ascvec.copy_gm2ub_plan"(%218, %alloc_1, %c0_i32, %c0_i32, %c0_i64, %217, %142, %143, %c0_i64, %143, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xf16, 1>, memref<1xf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                    ascvec.release_buf mte2, %212, 0 {cannir.buffer.generated_sync}
                    %219 = arith.select %199, %146, %147 : i64
                    %220 = arith.select %199, %c6_i64, %c7_i64 : i64
                    %221 = cannir.extract_dim %159[1] : !cannir.stride<(-1, -1, 128, 1)>
                    %222 = arith.muli %184, %221 : i64
                    %223 = arith.muli %222, %c4_i64 : i64
                    %224 = arith.addi %162, %223 : i64
                    %225 = arith.addi %224, %163 : i64
                    %226 = ascscalar.bind_local_ptr %219 : memref<1xf32, 1>
                    ascvec.get_buf mte2, %220, 0 {cannir.buffer.generated_sync}
                    "ascvec.copy_gm2ub_plan"(%226, %alloc_2, %c0_i32, %c0_i32, %c0_i64, %225, %168, %169, %c0_i64, %169, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = -1 : i64, ub_capacity = 4 : i64}> : (memref<1xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                    ascvec.release_buf mte2, %220, 0 {cannir.buffer.generated_sync}
                    scf.yield %203 : i64
                  } else {
                    scf.yield %arg6 : i64
                  }
                  %187 = arith.cmpi eq, %arg7, %c0_i64 : i64
                  %188 = arith.select %187, %98, %99 : i64
                  %189 = ascscalar.bind_local_ptr %188 : memref<128xf16, 1>
                  %190 = arith.select %187, %c2_i64, %c3_i64 : i64
                  %191 = arith.addi %arg7, %c1_i64 : i64
                  %192 = arith.remui %191, %c2_i64 : i64
                  %193 = arith.select %187, %121, %122 : i64
                  %194 = ascscalar.bind_local_ptr %193 : memref<128xf16, 1>
                  %195 = arith.select %187, %c4_i64, %c5_i64 : i64
                  %196 = arith.select %187, %146, %147 : i64
                  %197 = ascscalar.bind_local_ptr %196 : memref<128xf32, 1>
                  %198 = arith.select %187, %c6_i64, %c7_i64 : i64
                  ascvec.get_buf v, %190, 0 {cannir.buffer.generated_sync}
                  ascvec.vf(%189, %alloc_12 : memref<128xf16, 1>, memref<128xf32, 1>) {
                  ^bb0(%arg8: memref<128xf16, 1>, %arg9: memref<128xf32, 1>):
                    %c0_13 = arith.constant 0 : index
                    %c64_i32 = arith.constant 64 : i32
                    %c0_14 = arith.constant 0 : index
                    %c1 = arith.constant 1 : index
                    scf.for %arg10 = %c0_14 to %c1 step %c1 {
                      %mask, %remaining_out = ascvec.update_mask %c64_i32 : !ascvec.mask
                      %199 = ascvec.loadalign_unpack %arg8[%c0_13] {mode = "b16_to_b32"} : memref<128xf16, 1>, index -> <f16>
                      %200 = ascvec.cast %199, %mask {rounding = 6 : i32} : <f16> -> <f32>
                      ascvec.storealign %arg9[%c0_13], %200, %mask : memref<128xf32, 1>, index, <f32>
                    }
                  } {cannir.vf_mode = "simd"}
                  ascvec.release_buf v, %190, 0 {cannir.buffer.generated_sync}
                  ascvec.get_buf v, %195, 0 {cannir.buffer.generated_sync}
                  ascvec.get_buf v, %198, 0 {cannir.buffer.generated_sync}
                  ascvec.get_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
                  ascvec.vf(%194, %197, %alloc_12, %alloc_4 : memref<128xf16, 1>, memref<128xf32, 1>, memref<128xf32, 1>, memref<16384xf32, 1>) {
                  ^bb0(%arg8: memref<128xf16, 1>, %arg9: memref<128xf32, 1>, %arg10: memref<128xf32, 1>, %arg11: memref<16384xf32, 1>):
                    %c64 = arith.constant 64 : index
                    %c1_i32_13 = arith.constant 1 : i32
                    %c0_i32_14 = arith.constant 0 : i32
                    %c128_i32 = arith.constant 128 : i32
                    %c0_15 = arith.constant 0 : index
                    %c64_i32 = arith.constant 64 : i32
                    %mask, %remaining_out = ascvec.update_mask %c64_i32 : !ascvec.mask
                    %199 = ascvec.loadalign_unpack %arg8[%c0_15] {mode = "b16_to_b32"} : memref<128xf16, 1>, index -> <f16>
                    %200 = ascvec.loadalign_unpack %arg8[%c64] {mode = "b16_to_b32"} : memref<128xf16, 1>, index -> <f16>
                    %201 = ascvec.cast %199, %mask {rounding = 6 : i32} : <f16> -> <f32>
                    %202 = ascvec.cast %200, %mask {rounding = 6 : i32} : <f16> -> <f32>
                    %203 = ascvec.loadalign %arg9[%c0_15] : memref<128xf32, 1>, index -> <f32>
                    %204 = ascvec.loadalign %arg9[%c64] : memref<128xf32, 1>, index -> <f32>
                    scf.for %arg12 = %c0_i32_14 to %c64_i32 step %c1_i32_13  : i32 {
                      %205 = arith.muli %arg12, %c128_i32 overflow<nsw> : i32
                      %206 = arith.index_cast %arg12 : i32 to index
                      %207 = ascvec.loadalign_brc %arg10[%206] : memref<128xf32, 1>, index -> <f32>
                      %208 = arith.index_cast %205 : i32 to index
                      %209 = ascvec.loadalign %arg11[%208] : memref<16384xf32, 1>, index -> <f32>
                      %210 = arith.addi %205, %c64_i32 overflow<nsw> : i32
                      %211 = arith.index_cast %210 : i32 to index
                      %212 = ascvec.loadalign %arg11[%211] : memref<16384xf32, 1>, index -> <f32>
                      %213 = ascvec.mul %209, %203, %mask : <f32>
                      %214 = ascvec.mul %212, %204, %mask : <f32>
                      %215 = ascvec.copy %207, %mask : <f32>
                      %216 = ascvec.madd_two_address %215, %201, %213, %mask : <f32>
                      %217 = ascvec.madd_two_address %207, %202, %214, %mask : <f32>
                      ascvec.storealign %arg11[%208], %216, %mask : memref<16384xf32, 1>, index, <f32>
                      ascvec.storealign %arg11[%211], %217, %mask : memref<16384xf32, 1>, index, <f32>
                    } {cannir.loop_unroll = 4 : i64}
                  } {cannir.vf_mode = "simd"}
                  ascvec.release_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
                  ascvec.release_buf v, %198, 0 {cannir.buffer.generated_sync}
                  ascvec.release_buf v, %195, 0 {cannir.buffer.generated_sync}
                  scf.yield %186, %192 : i64, i64
                } else {
                  scf.yield %arg6, %arg7 : i64, i64
                }
                scf.yield %183#0, %183#1 : i64, i64
              }
              %172 = arith.muli %83, %c16384_i64 : i64
              %173 = arith.addi %91, %172 : i64
              %174 = arith.muli %70, %c8192_i64 overflow<nsw> : i64
              %175 = arith.addi %173, %174 : i64
              %176 = arith.muli %175, %c32_i64 : i64
              %177 = ascscalar.memref_address %alloc : memref<1xf32>
              %178 = arith.divui %176, %c8_i64 : i64
              %179 = arith.addi %177, %178 : i64
              %180 = ascscalar.bind_gm_ptr %179 : memref<1xf32>
              %181 = ascscalar.bitcast_memref %alloc_4 byte_offset = %c0_i64 : memref<16384xf32, 1> -> memref<?xf32, 1>
              ascvec.get_buf mte3, %c0_i64, 0 {cannir.buffer.generated_sync}
              ascvec.copy_ub2gm %180[%c0], %181[%c0], %c1_i32, %c32768_i32, %c0_i64, %c32768_i32 l2_cache_ctl %c0_i32 : memref<1xf32>, memref<?xf32, 1>
              ascvec.release_buf mte3, %c0_i64, 0 {cannir.buffer.generated_sync}
            }
          }
        }
      }
    }
    return
  }
}
