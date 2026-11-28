module {
  func.func @run_non_tail_fused_eager_rng_ccb9781bf111dda8dd0a38e80155da9f(%arg0: !cannir.tensor<<gm, f32>, <<(-1, -1)>, <(-1, 2)>>, <<(-1, -1)>, <(-1, 2)>>, identity>, %arg1: !cannir.tensor<<gm, f32>, <<(-1, -1)>, <(-1, 2)>>, <<(-1, -1)>, <(-1, 2)>>, identity>, %arg2: !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg3: !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64) attributes {cannbotdsl.contract_digest = "ccb9781bf111dda8dd0a38e80155da9f454428d5211a860323f56b1e2e36f77d", cannbotdsl.public_slot_count = 8 : i64, cannbotdsl.slot_dtype_bits = [32, 32, 8, 8, 64, 64, 64, 64], cannbotdsl.slot_dtype_codes = [2, 2, 1, 1, 0, 0, 0, 0], cannbotdsl.slot_dtype_lanes = [1, 1, 1, 1, 1, 1, 1, 1], cannbotdsl.slot_flags = [1, 1, 1, 1, 1, 1, 1, 1], cannbotdsl.slot_kinds = [1, 1, 1, 1, 2, 2, 2, 2], cannbotdsl.slot_type_digests = ["299b8584bfb7b4ecbed3e029654203d20256497197396c46e927fd078ee9206a", "299b8584bfb7b4ecbed3e029654203d20256497197396c46e927fd078ee9206a", "0466425dd1673c3d734b77386a8d2c26ab0afa3f2945f4af539e71f1f55f41c9", "1d2672dce8d9ff4fcf6d37f57043432984990a93dd5ece4b4944846a06c54afa", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515"], cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.host} {
    cannir.kernel_launch @kernel__DynamicMxQuantSrKernel_run_non_tail_fused_0(%arg5) [%arg0, %arg1, %arg2, %arg3, %arg4, %arg6, %arg7] {cannbotdsl.profile = {indices = array<i64: 0, 1, 2, 3>, name = "_DynamicMxQuantSrKernel_run_non_tail_fused", op_type = "_DynamicMxQuantSrKernel_run_non_tail_fused", roles = array<i64: 0, 0, 1, 1>}} : !cannir.tensor<<gm, f32>, <<(-1, -1)>, <(-1, 2)>>, <<(-1, -1)>, <(-1, 2)>>, identity>, !cannir.tensor<<gm, f32>, <<(-1, -1)>, <(-1, 2)>>, <<(-1, -1)>, <(-1, 2)>>, identity>, !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, i64, i64, i64
    return
  }
  func.func @kernel__DynamicMxQuantSrKernel_run_non_tail_fused_0(%arg0: !cannir.tensor<<gm, f32>, <<(-1, -1)>, <(-1, 2)>>, <<(-1, -1)>, <(-1, 2)>>, identity>, %arg1: !cannir.tensor<<gm, f32>, <<(-1, -1)>, <(-1, 2)>>, <<(-1, -1)>, <(-1, 2)>>, identity>, %arg2: !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg3: !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg4: i64, %arg5: i64, %arg6: i64) attributes {cannir.buffer.auto_sync, cannir.buffer.sync_analyzed, cannir.buffer.sync_verified, cannir.buffer.tensor_state, cannir.core = "aiv", cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.device, cannir.explicit_ring_buffer_cursors, cannir.kernel_group = "kernel__DynamicMxQuantSrKernel_run_non_tail_fused_0", cannir.resource.ids_prepared} {
    %c1664_i32 = arith.constant 1664 : i32
    %c26624_i32 = arith.constant 26624 : i32
    %c832_i32 = arith.constant 832 : i32
    %c13312_i32 = arith.constant 13312 : i32
    %c64_i64 = arith.constant 64 : i64
    %c208_i64 = arith.constant 208 : i64
    %c0_i32 = arith.constant 0 : i32
    %c256_i64 = arith.constant 256 : i64
    %c1023_i64 = arith.constant 1023 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %c4294967296_i64 = arith.constant 4294967296 : i64
    %c16_i64 = arith.constant 16 : i64
    %c4_i32 = arith.constant 4 : i32
    %c1073741824_i32 = arith.constant 1073741824 : i32
    %c16384_i64 = arith.constant 16384 : i64
    %c3449720151_i64 = arith.constant 3449720151 : i64
    %c-1640531527_i32 = arith.constant -1640531527 : i32
    %c-1150833019_i32 = arith.constant -1150833019 : i32
    %c3528531795_i64 = arith.constant 3528531795 : i64
    %c1013904242_i32 = arith.constant 1013904242 : i32
    %c1993301258_i32 = arith.constant 1993301258 : i32
    %c1_i32 = arith.constant 1 : i32
    %c64_i32 = arith.constant 64 : i32
    %c0 = arith.constant 0 : index
    %c26624_i64 = arith.constant 26624 : i64
    %c1664_i64 = arith.constant 1664 : i64
    %c32_i64 = arith.constant 32 : i64
    %c13312_i64 = arith.constant 13312 : i64
    %c8_i64 = arith.constant 8 : i64
    %c207_i64 = arith.constant 207 : i64
    %c512_i64 = arith.constant 512 : i64
    %c4_i64 = arith.constant 4 : i64
    %c3_i64 = arith.constant 3 : i64
    %c2_i64 = arith.constant 2 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %alloc = memref.alloc() {cannir.arg_index = 0 : i64} : memref<1xf32>
    %alloc_0 = memref.alloc() {cannir.arg_index = 1 : i64} : memref<1xf32>
    %alloc_1 = memref.alloc() {cannir.arg_index = 2 : i64} : memref<1xui8>
    %alloc_2 = memref.alloc() {cannir.arg_index = 3 : i64} : memref<1xui8>
    %alloc_3 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<13312xf32, 1>
    %alloc_4 = memref.alloc() {cannir.byte_offset = 53248 : i64} : memref<13312xf32, 1>
    %alloc_5 = memref.alloc() {cannir.byte_offset = 106496 : i64} : memref<13312xui8, 1>
    %alloc_6 = memref.alloc() {cannir.byte_offset = 119808 : i64} : memref<13312xui8, 1>
    %alloc_7 = memref.alloc() {cannir.byte_offset = 133120 : i64} : memref<832xui8, 1>
    %alloc_8 = memref.alloc() {cannir.byte_offset = 133952 : i64} : memref<832xui8, 1>
    %alloc_9 = memref.alloc() {cannir.byte_offset = 136576 : i64} : memref<3328xi32, 1>
    %alloc_10 = memref.alloc() {cannir.byte_offset = 163200 : i64} : memref<13312xui8, 1>
    %alloc_11 = memref.alloc() {cannir.byte_offset = 176512 : i64} : memref<832xui8, 1>
    %alloc_12 = memref.alloc() {cannir.byte_offset = 177344 : i64} : memref<26624xui8, 1>
    %alloc_13 = memref.alloc() {cannir.byte_offset = 203968 : i64} : memref<832xui16, 1>
    %0 = cannir.extract_layout %arg0 : !cannir.tensor<<gm, f32>, <<(-1, -1)>, <(-1, 2)>>, <<(-1, -1)>, <(-1, 2)>>, identity>
    %1 = cannir.extract_shape %0 : <<(-1, -1)>, <(-1, 2)>> -> <(-1, -1)>
    %2 = cannir.extract_dim %1[1] : !cannir.shape<(-1, -1)>
    %3 = arith.remsi %2, %c64_i64 : i64
    %4 = arith.cmpi ne, %3, %c0_i64 : i64
    %5 = arith.cmpi slt, %3, %c0_i64 : i64
    %6 = arith.andi %4, %5 : i1
    %7 = arith.divsi %2, %c64_i64 : i64
    %8 = arith.extui %6 : i1 to i64
    %9 = arith.subi %7, %8 : i64
    %10 = cannir.extract_dim %1[0] : !cannir.shape<(-1, -1)>
    %11 = arith.muli %10, %9 : i64
    %12 = ascscalar.get_block_num
    %13 = arith.addi %11, %12 : i64
    %14 = arith.subi %13, %c1_i64 : i64
    %15 = arith.remsi %14, %12 : i64
    %16 = arith.cmpi ne, %15, %c0_i64 : i64
    %17 = arith.cmpi slt, %15, %c0_i64 : i64
    %18 = arith.cmpi slt, %12, %c0_i64 : i64
    %19 = arith.xori %17, %18 : i1
    %20 = arith.andi %16, %19 : i1
    %21 = arith.divsi %14, %12 : i64
    %22 = arith.extui %20 : i1 to i64
    %23 = arith.subi %21, %22 : i64
    %24 = arith.addi %23, %c207_i64 : i64
    %25 = arith.remsi %24, %c208_i64 : i64
    %26 = arith.cmpi ne, %25, %c0_i64 : i64
    %27 = arith.cmpi slt, %25, %c0_i64 : i64
    %28 = arith.andi %26, %27 : i1
    %29 = arith.divsi %24, %c208_i64 : i64
    %30 = arith.extui %28 : i1 to i64
    %31 = arith.subi %29, %30 : i64
    %32 = arith.addi %23, %31 : i64
    %33 = arith.subi %32, %c1_i64 : i64
    %34 = arith.remsi %33, %31 : i64
    %35 = arith.cmpi ne, %34, %c0_i64 : i64
    %36 = arith.cmpi slt, %34, %c0_i64 : i64
    %37 = arith.cmpi slt, %31, %c0_i64 : i64
    %38 = arith.xori %36, %37 : i1
    %39 = arith.andi %35, %38 : i1
    %40 = arith.divsi %33, %31 : i64
    %41 = arith.extui %39 : i1 to i64
    %42 = arith.subi %40, %41 : i64
    %43 = arith.addi %42, %c3_i64 : i64
    %44 = arith.remsi %43, %c4_i64 : i64
    %45 = arith.cmpi ne, %44, %c0_i64 : i64
    %46 = arith.cmpi slt, %44, %c0_i64 : i64
    %47 = arith.andi %45, %46 : i1
    %48 = arith.divsi %43, %c4_i64 : i64
    %49 = arith.extui %47 : i1 to i64
    %50 = arith.subi %48, %49 : i64
    %51 = arith.muli %50, %c4_i64 : i64
    %52 = arith.muli %50, %c256_i64 : i64
    %53 = arith.addi %9, %51 : i64
    %54 = arith.subi %53, %c1_i64 : i64
    %55 = arith.remsi %54, %51 : i64
    %56 = arith.cmpi ne, %55, %c0_i64 : i64
    %57 = arith.cmpi slt, %55, %c0_i64 : i64
    %58 = arith.cmpi slt, %51, %c0_i64 : i64
    %59 = arith.xori %57, %58 : i1
    %60 = arith.andi %56, %59 : i1
    %61 = arith.divsi %54, %51 : i64
    %62 = arith.extui %60 : i1 to i64
    %63 = arith.subi %61, %62 : i64
    %64 = arith.muli %10, %63 : i64
    %65 = ascscalar.get_block_idx
    scf.for %arg7 = %65 to %64 step %12  : i64 {
      %66 = arith.remsi %arg7, %63 : i64
      %67 = arith.cmpi ne, %66, %c0_i64 : i64
      %68 = arith.cmpi slt, %66, %c0_i64 : i64
      %69 = arith.cmpi slt, %63, %c0_i64 : i64
      %70 = arith.xori %68, %69 : i1
      %71 = arith.andi %67, %70 : i1
      %72 = arith.divsi %arg7, %63 : i64
      %73 = arith.extui %71 : i1 to i64
      %74 = arith.subi %72, %73 : i64
      %75 = arith.select %71, %63, %c0_i64 : i64
      %76 = arith.addi %66, %75 : i64
      %77 = arith.muli %76, %52 : i64
      %78 = arith.subi %2, %77 : i64
      %79 = arith.cmpi sgt, %78, %52 : i64
      %80 = arith.select %79, %52, %78 : i64
      %81 = arith.minsi %74, %10 : i64
      %82 = arith.subi %10, %81 : i64
      %83 = arith.minsi %82, %c1_i64 : i64
      %84 = arith.cmpi sle, %74, %10 : i64
      %85 = arith.select %84, %83, %c0_i64 : i64
      %86 = arith.maxsi %52, %c1_i64 : i64
      %87 = arith.divsi %2, %86 : i64
      %88 = arith.minsi %76, %87 : i64
      %89 = arith.muli %88, %52 : i64
      %90 = arith.subi %2, %89 : i64
      %91 = arith.minsi %52, %90 : i64
      %92 = arith.cmpi sle, %76, %87 : i64
      %93 = arith.select %92, %91, %c0_i64 : i64
      %94 = cannir.extract_stride %0 : <<(-1, -1)>, <(-1, 2)>> -> <(-1, 2)>
      %95 = cannir.extract_dim %94[0] : !cannir.stride<(-1, 2)>
      %96 = cannir.extract_physical_layout %arg0 : !cannir.tensor<<gm, f32>, <<(-1, -1)>, <(-1, 2)>>, <<(-1, -1)>, <(-1, 2)>>, identity>
      %97 = cannir.extract_stride %96 : <<(-1, -1)>, <(-1, 2)>> -> <(-1, 2)>
      %98 = cannir.extract_dim %97[0] : !cannir.stride<(-1, 2)>
      %99 = arith.muli %74, %98 : i64
      %100 = arith.muli %77, %c2_i64 : i64
      %101 = arith.addi %99, %100 : i64
      %102 = arith.muli %101, %c32_i64 : i64
      %103 = ascscalar.memref_address %alloc : memref<1xf32>
      %104 = arith.divui %102, %c8_i64 : i64
      %105 = arith.addi %103, %104 : i64
      %106 = ascscalar.bind_gm_ptr %105 : memref<1xf32>
      %107 = ascscalar.bitcast_memref %alloc_3 byte_offset = %c0_i64 : memref<13312xf32, 1> -> memref<?xf32, 1>
      %108 = arith.cmpi ugt, %85, %c0_i64 : i64
      %109 = arith.cmpi ugt, %93, %c0_i64 : i64
      %110 = arith.andi %108, %109 : i1
      scf.if %110 {
        ascvec.get_buf mte2, %c0_i64, 0 {cannir.buffer.generated_sync}
        %317 = arith.trunci %93 : i64 to i32
        %318 = arith.trunci %85 : i64 to i32
        ascvec.copy_gm2ub_nddma %107[%c0], %106[%c0], %317, %318, %c1_i32, %c1_i32, %c1_i32, %c2_i64, %95, %c0_i64, %c0_i64, %c0_i64, %c1_i64, %c13312_i64, %c0_i64, %c0_i64, %c0_i64 l2_cache_ctl %c0_i32 left_padding(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) right_padding(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) pad_value %c0_i32 : i32 : memref<?xf32, 1>, memref<1xf32>
        ascvec.release_buf mte2, %c0_i64, 0 {cannir.buffer.generated_sync}
      }
      %111 = arith.muli %74, %c2_i64 : i64
      %112 = arith.muli %111, %2 : i64
      %113 = arith.addi %112, %77 : i64
      %114 = arith.remsi %80, %c256_i64 : i64
      %115 = arith.cmpi ne, %114, %c0_i64 : i64
      %116 = arith.cmpi slt, %114, %c0_i64 : i64
      %117 = arith.andi %115, %116 : i1
      %118 = arith.divsi %80, %c256_i64 : i64
      %119 = arith.extui %117 : i1 to i64
      %120 = arith.subi %118, %119 : i64
      %121 = arith.addi %80, %c1023_i64 : i64
      %122 = arith.remsi %121, %c1024_i64 : i64
      %123 = arith.cmpi ne, %122, %c0_i64 : i64
      %124 = arith.cmpi slt, %122, %c0_i64 : i64
      %125 = arith.andi %123, %124 : i1
      %126 = arith.divsi %121, %c1024_i64 : i64
      %127 = arith.extui %125 : i1 to i64
      %128 = arith.subi %126, %127 : i64
      %129 = arith.trunci %arg5 : i64 to i32
      %130 = arith.extui %129 : i32 to i64
      %131 = arith.subi %arg5, %130 : i64
      %132 = arith.remsi %131, %c4294967296_i64 : i64
      %133 = arith.cmpi ne, %132, %c0_i64 : i64
      %134 = arith.cmpi slt, %132, %c0_i64 : i64
      %135 = arith.andi %133, %134 : i1
      %136 = arith.divsi %131, %c4294967296_i64 : i64
      %137 = arith.extui %135 : i1 to i64
      %138 = arith.subi %136, %137 : i64
      %139 = arith.trunci %138 : i64 to i32
      %140 = arith.divui %113, %c16_i64 : i64
      %141 = arith.trunci %140 : i64 to i32
      %142 = arith.extui %141 : i32 to i64
      %143 = arith.subi %140, %142 : i64
      %144 = arith.remsi %143, %c4294967296_i64 : i64
      %145 = arith.cmpi ne, %144, %c0_i64 : i64
      %146 = arith.cmpi slt, %144, %c0_i64 : i64
      %147 = arith.andi %145, %146 : i1
      %148 = arith.divsi %143, %c4294967296_i64 : i64
      %149 = arith.extui %147 : i1 to i64
      %150 = arith.subi %148, %149 : i64
      %151 = arith.trunci %150 : i64 to i32
      %152 = arith.trunci %arg6 : i64 to i32
      %153 = arith.extui %152 : i32 to i64
      %154 = arith.subi %arg6, %153 : i64
      %155 = arith.remsi %154, %c4294967296_i64 : i64
      %156 = arith.cmpi ne, %155, %c0_i64 : i64
      %157 = arith.cmpi slt, %155, %c0_i64 : i64
      %158 = arith.andi %156, %157 : i1
      %159 = arith.divsi %154, %c4294967296_i64 : i64
      %160 = arith.extui %158 : i1 to i64
      %161 = arith.subi %159, %160 : i64
      %162 = arith.trunci %161 : i64 to i32
      %163 = arith.divui %152, %c4_i32 : i32
      %164 = arith.remui %162, %c4_i32 : i32
      %165 = arith.muli %164, %c1073741824_i32 : i32
      %166 = arith.addi %163, %165 : i32
      %167 = arith.divui %162, %c4_i32 : i32
      %168 = arith.muli %128, %c64_i64 : i64
      %169 = arith.divui %140, %c16384_i64 : i64
      %170 = arith.addi %140, %168 : i64
      %171 = arith.subi %170, %c1_i64 : i64
      %172 = arith.divui %171, %c16384_i64 : i64
      %173 = arith.cmpi eq, %169, %172 : i64
      scf.if %173 {
        %317 = arith.trunci %169 : i64 to i32
        %318 = arith.extui %317 : i32 to i64
        %319 = arith.subi %169, %318 : i64
        %320 = arith.remsi %319, %c4294967296_i64 : i64
        %321 = arith.cmpi ne, %320, %c0_i64 : i64
        %322 = arith.cmpi slt, %320, %c0_i64 : i64
        %323 = arith.andi %321, %322 : i1
        %324 = arith.divsi %319, %c4294967296_i64 : i64
        %325 = arith.extui %323 : i1 to i64
        %326 = arith.subi %324, %325 : i64
        %327 = arith.trunci %326 : i64 to i32
        %328 = arith.addi %166, %317 : i32
        %329 = arith.addi %167, %327 : i32
        %330 = arith.cmpi ult, %328, %166 : i32
        %331 = arith.extui %330 : i1 to i32
        %332 = arith.addi %329, %331 : i32
        %333 = arith.extui %328 : i32 to i64
        %334 = arith.muli %333, %c3449720151_i64 : i64
        %335 = arith.trunci %334 : i64 to i32
        %336 = arith.extui %335 : i32 to i64
        %337 = arith.subi %334, %336 : i64
        %338 = arith.remsi %337, %c4294967296_i64 : i64
        %339 = arith.cmpi ne, %338, %c0_i64 : i64
        %340 = arith.cmpi slt, %338, %c0_i64 : i64
        %341 = arith.andi %339, %340 : i1
        %342 = arith.divsi %337, %c4294967296_i64 : i64
        %343 = arith.extui %341 : i1 to i64
        %344 = arith.subi %342, %343 : i64
        %345 = arith.trunci %344 : i64 to i32
        %346 = arith.xori %332, %139 : i32
        %347 = arith.extui %346 : i32 to i64
        %348 = arith.muli %347, %c3449720151_i64 : i64
        %349 = arith.trunci %348 : i64 to i32
        %350 = arith.extui %349 : i32 to i64
        %351 = arith.subi %348, %350 : i64
        %352 = arith.remsi %351, %c4294967296_i64 : i64
        %353 = arith.cmpi ne, %352, %c0_i64 : i64
        %354 = arith.cmpi slt, %352, %c0_i64 : i64
        %355 = arith.andi %353, %354 : i1
        %356 = arith.divsi %351, %c4294967296_i64 : i64
        %357 = arith.extui %355 : i1 to i64
        %358 = arith.subi %356, %357 : i64
        %359 = arith.trunci %358 : i64 to i32
        %360 = arith.addi %129, %c-1640531527_i32 : i32
        %361 = arith.addi %139, %c-1150833019_i32 : i32
        %362 = arith.xori %359, %335 : i32
        %363 = arith.xori %362, %360 : i32
        %364 = arith.extui %363 : i32 to i64
        %365 = arith.muli %364, %c3528531795_i64 : i64
        %366 = arith.trunci %365 : i64 to i32
        %367 = arith.extui %366 : i32 to i64
        %368 = arith.subi %365, %367 : i64
        %369 = arith.remsi %368, %c4294967296_i64 : i64
        %370 = arith.cmpi ne, %369, %c0_i64 : i64
        %371 = arith.cmpi slt, %369, %c0_i64 : i64
        %372 = arith.andi %370, %371 : i1
        %373 = arith.divsi %368, %c4294967296_i64 : i64
        %374 = arith.extui %372 : i1 to i64
        %375 = arith.subi %373, %374 : i64
        %376 = arith.trunci %375 : i64 to i32
        %377 = arith.xori %345, %129 : i32
        %378 = arith.addi %129, %c1013904242_i32 : i32
        %379 = arith.xori %349, %378 : i32
        %380 = arith.addi %139, %c1993301258_i32 : i32
        %381 = arith.xori %376, %380 : i32
        ascsimt.vf(%141, %377, %361, %379, %381, %129, %139, %366, %alloc_9, %168 : i32, i32, i32, i32, i32, i32, i32, i32, memref<3328xi32, 1>, i64) {
        ^bb0(%arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: memref<3328xi32, 1>, %arg17: i64):
          %c256_i64_14 = arith.constant 256 : i64
          %c3_i64_15 = arith.constant 3 : i64
          %c2_i64_16 = arith.constant 2 : i64
          %c1_i64_17 = arith.constant 1 : i64
          %c4_i64_18 = arith.constant 4 : i64
          %c1_i32_19 = arith.constant 1 : i32
          %c10_i32 = arith.constant 10 : i32
          %c3_i32 = arith.constant 3 : i32
          %c-1150833019_i32_20 = arith.constant -1150833019 : i32
          %c-1640531527_i32_21 = arith.constant -1640531527 : i32
          %c-845247145_i32 = arith.constant -845247145 : i32
          %c-766435501_i32 = arith.constant -766435501 : i32
          %c16384_i32 = arith.constant 16384 : i32
          %382 = ascsimt.thread_idx {ascsimt.scope = 1 : i64, axis = "x", cannir.vf_threads = 256 : i64} : i32
          %383 = arith.extsi %382 {cannir.vf_threads = 256 : i64} : i32 to i64
          scf.for %arg18 = %383 to %arg17 step %c256_i64_14  : i64 {
            %384 = arith.trunci %arg18 : i64 to i32
            %385 = arith.addi %arg8, %384 : i32
            %386 = arith.remui %385, %c16384_i32 : i32
            %387 = arith.xori %386, %arg9 : i32
            %388 = arith.muli %387, %c-766435501_i32 : i32
            %389 = "ascsimt.bitcast"(%387) : (i32) -> ui32
            %390 = "ascsimt.cast"(%c-766435501_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
            %391 = ascsimt.math_binary %389, %390, "mulhi" : ui32, ui32 -> ui32
            %392 = "ascsimt.bitcast"(%391) : (ui32) -> i32
            %393 = arith.xori %392, %arg10 : i32
            %394 = arith.muli %393, %c-845247145_i32 : i32
            %395 = "ascsimt.bitcast"(%393) : (i32) -> ui32
            %396 = "ascsimt.cast"(%c-845247145_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
            %397 = ascsimt.math_binary %395, %396, "mulhi" : ui32, ui32 -> ui32
            %398 = "ascsimt.bitcast"(%397) : (ui32) -> i32
            %399 = arith.xori %398, %arg11 : i32
            %400 = arith.xori %388, %arg12 : i32
            %401:4 = scf.for %arg19 = %c3_i32 to %c10_i32 step %c1_i32_19 iter_args(%arg20 = %399, %arg21 = %394, %arg22 = %400, %arg23 = %arg15) -> (i32, i32, i32, i32)  : i32 {
              %410 = arith.muli %arg20, %c-766435501_i32 : i32
              %411 = "ascsimt.bitcast"(%arg20) : (i32) -> ui32
              %412 = ascsimt.math_binary %411, %390, "mulhi" : ui32, ui32 -> ui32
              %413 = "ascsimt.bitcast"(%412) : (ui32) -> i32
              %414 = arith.muli %arg22, %c-845247145_i32 : i32
              %415 = "ascsimt.bitcast"(%arg22) : (i32) -> ui32
              %416 = ascsimt.math_binary %415, %396, "mulhi" : ui32, ui32 -> ui32
              %417 = "ascsimt.bitcast"(%416) : (ui32) -> i32
              %418 = arith.muli %arg19, %c-1640531527_i32_21 : i32
              %419 = arith.addi %arg13, %418 : i32
              %420 = arith.muli %arg19, %c-1150833019_i32_20 : i32
              %421 = arith.addi %arg14, %420 : i32
              %422 = arith.xori %417, %arg21 : i32
              %423 = arith.xori %422, %419 : i32
              %424 = arith.xori %413, %arg23 : i32
              %425 = arith.xori %424, %421 : i32
              scf.yield %423, %414, %425, %410 : i32, i32, i32, i32
            }
            %402 = arith.muli %arg18, %c4_i64_18 : i64
            %403 = arith.index_cast %402 : i64 to index
            ascsimt.store %arg16[%403], %401#0 : memref<3328xi32, 1>, i32
            %404 = arith.addi %402, %c1_i64_17 : i64
            %405 = arith.index_cast %404 : i64 to index
            ascsimt.store %arg16[%405], %401#1 : memref<3328xi32, 1>, i32
            %406 = arith.addi %402, %c2_i64_16 : i64
            %407 = arith.index_cast %406 : i64 to index
            ascsimt.store %arg16[%407], %401#2 : memref<3328xi32, 1>, i32
            %408 = arith.addi %402, %c3_i64_15 : i64
            %409 = arith.index_cast %408 : i64 to index
            ascsimt.store %arg16[%409], %401#3 : memref<3328xi32, 1>, i32
          } {cannir.vf_threads = 256 : i64}
        } {threads = 256 : i64}
      } else {
        ascsimt.vf(%141, %151, %166, %167, %129, %139, %alloc_9, %168 : i32, i32, i32, i32, i32, i32, memref<3328xi32, 1>, i64) {
        ^bb0(%arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: memref<3328xi32, 1>, %arg15: i64):
          %c256_i64_14 = arith.constant 256 : i64
          %c3_i64_15 = arith.constant 3 : i64
          %c2_i64_16 = arith.constant 2 : i64
          %c1_i64_17 = arith.constant 1 : i64
          %c4_i64_18 = arith.constant 4 : i64
          %c1_i32_19 = arith.constant 1 : i32
          %c10_i32 = arith.constant 10 : i32
          %c0_i32_20 = arith.constant 0 : i32
          %c-1150833019_i32_21 = arith.constant -1150833019 : i32
          %c-1640531527_i32_22 = arith.constant -1640531527 : i32
          %c-845247145_i32 = arith.constant -845247145 : i32
          %c-766435501_i32 = arith.constant -766435501 : i32
          %c262144_i32 = arith.constant 262144 : i32
          %c16384_i32 = arith.constant 16384 : i32
          %317 = ascsimt.thread_idx {ascsimt.scope = 2 : i64, axis = "x", cannir.vf_threads = 256 : i64} : i32
          %318 = arith.extsi %317 {cannir.vf_threads = 256 : i64} : i32 to i64
          scf.for %arg16 = %318 to %arg15 step %c256_i64_14  : i64 {
            %319 = arith.trunci %arg16 : i64 to i32
            %320 = arith.addi %arg8, %319 : i32
            %321 = arith.cmpi ult, %320, %arg8 : i32
            %322 = arith.extui %321 : i1 to i32
            %323 = arith.addi %arg9, %322 : i32
            %324 = arith.divui %320, %c16384_i32 : i32
            %325 = arith.remui %323, %c16384_i32 : i32
            %326 = arith.muli %325, %c262144_i32 : i32
            %327 = arith.addi %324, %326 : i32
            %328 = arith.divui %323, %c16384_i32 : i32
            %329 = arith.addi %arg10, %327 : i32
            %330 = arith.addi %arg11, %328 : i32
            %331 = arith.cmpi ult, %329, %arg10 : i32
            %332 = arith.extui %331 : i1 to i32
            %333 = arith.addi %330, %332 : i32
            %334 = arith.remui %320, %c16384_i32 : i32
            %335:4 = scf.for %arg17 = %c0_i32_20 to %c10_i32 step %c1_i32_19 iter_args(%arg18 = %c0_i32_20, %arg19 = %334, %arg20 = %329, %arg21 = %333) -> (i32, i32, i32, i32)  : i32 {
              %344 = arith.muli %arg18, %c-766435501_i32 : i32
              %345 = "ascsimt.bitcast"(%arg18) : (i32) -> ui32
              %346 = "ascsimt.cast"(%c-766435501_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
              %347 = ascsimt.math_binary %345, %346, "mulhi" : ui32, ui32 -> ui32
              %348 = "ascsimt.bitcast"(%347) : (ui32) -> i32
              %349 = arith.muli %arg20, %c-845247145_i32 : i32
              %350 = "ascsimt.bitcast"(%arg20) : (i32) -> ui32
              %351 = "ascsimt.cast"(%c-845247145_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
              %352 = ascsimt.math_binary %350, %351, "mulhi" : ui32, ui32 -> ui32
              %353 = "ascsimt.bitcast"(%352) : (ui32) -> i32
              %354 = arith.muli %arg17, %c-1640531527_i32_22 : i32
              %355 = arith.addi %arg12, %354 : i32
              %356 = arith.muli %arg17, %c-1150833019_i32_21 : i32
              %357 = arith.addi %arg13, %356 : i32
              %358 = arith.xori %353, %arg19 : i32
              %359 = arith.xori %358, %355 : i32
              %360 = arith.xori %348, %arg21 : i32
              %361 = arith.xori %360, %357 : i32
              scf.yield %359, %349, %361, %344 : i32, i32, i32, i32
            }
            %336 = arith.muli %arg16, %c4_i64_18 : i64
            %337 = arith.index_cast %336 : i64 to index
            ascsimt.store %arg14[%337], %335#0 : memref<3328xi32, 1>, i32
            %338 = arith.addi %336, %c1_i64_17 : i64
            %339 = arith.index_cast %338 : i64 to index
            ascsimt.store %arg14[%339], %335#1 : memref<3328xi32, 1>, i32
            %340 = arith.addi %336, %c2_i64_16 : i64
            %341 = arith.index_cast %340 : i64 to index
            ascsimt.store %arg14[%341], %335#2 : memref<3328xi32, 1>, i32
            %342 = arith.addi %336, %c3_i64_15 : i64
            %343 = arith.index_cast %342 : i64 to index
            ascsimt.store %arg14[%343], %335#3 : memref<3328xi32, 1>, i32
          } {cannir.vf_threads = 256 : i64}
        } {threads = 256 : i64}
      }
      %174 = arith.trunci %120 : i64 to i32
      ascvec.get_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
      %175 = ascscalar.bitcast_memref %alloc_3 byte_offset = %c0_i64 : memref<13312xf32, 1> -> memref<?xui16, 1>
      ascvec.vf(%alloc_7, %alloc_3, %175, %alloc_9, %alloc_5, %174 : memref<832xui8, 1>, memref<13312xf32, 1>, memref<?xui16, 1>, memref<3328xi32, 1>, memref<13312xui8, 1>, i32) {
      ^bb0(%arg8: memref<832xui8, 1>, %arg9: memref<13312xf32, 1>, %arg10: memref<?xui16, 1>, %arg11: memref<3328xi32, 1>, %arg12: memref<13312xui8, 1>, %arg13: i32):
        %c512_i32 = arith.constant 512 : i32
        %c1_i32_14 = arith.constant 1 : i32
        %c2139062143_i32 = arith.constant 2139062143 : i32
        %c-8388608_i32 = arith.constant -8388608 : i32
        %c126_i8 = arith.constant 126 : i8
        %c-1_i8 = arith.constant -1 : i8
        %c56_i8 = arith.constant 56 : i8
        %cst = arith.constant 0.0156249991 : f32
        %c64_i32_15 = arith.constant 64 : i32
        %c254_i32 = arith.constant 254 : i32
        %c-8_i32 = arith.constant -8 : i32
        %c8_i32 = arith.constant 8 : i32
        %c255_i32 = arith.constant 255 : i32
        %c128_i32 = arith.constant 128 : i32
        %c256_i32 = arith.constant 256 : i32
        %c0_16 = arith.constant 0 : index
        %c1_i8 = arith.constant 1 : i8
        %c32_i8 = arith.constant 32 : i8
        %c65535_i32 = arith.constant 65535 : i32
        %c3_i8 = arith.constant 3 : i8
        %c0_i8 = arith.constant 0 : i8
        %c-128_i8 = arith.constant -128 : i8
        %c0_i32_17 = arith.constant 0 : i32
        %317 = ascvec.create_mask : !ascvec.mask
        %318 = ascvec.create_mask elem_bits 8 : !ascvec.mask
        %319 = ascvec.arange %c0_i32_17 : i32 -> <ui32>
        %320 = ascvec.shr %319, 3, %317 : <ui32> -> <ui32>
        %321 = ascvec.duplicate_scalar %c-128_i8, %318 : i8 -> <ui8>
        %322 = ascvec.arange %c0_i8 : i8 -> <ui8>
        %323 = ascvec.duplicate_scalar %c3_i8, %318 : i8 -> <ui8>
        %324 = ascvec.and %322, %323, %318 : <ui8>, <ui8> -> <ui8>
        %325 = ascvec.shl %324, 6, %318 : <ui8> -> <ui8>
        %326 = ascvec.shr %322, 2, %318 : <ui8> -> <ui8>
        %327 = ascvec.or %325, %326, %318 : <ui8>, <ui8> -> <ui8>
        %328 = ascvec.duplicate_scalar %c65535_i32, %317 : i32 -> <ui32>
        %329 = ascvec.cmp_lt_scalar %322, %c32_i8, %318 : <ui8>, i8 -> !ascvec.mask
        %330 = ascvec.cmp_lt_scalar %324, %c1_i8, %318 : <ui8>, i8 -> !ascvec.mask
        %331 = ascvec.mask_and %329, %330, %318
        ascvec.store_unalign_begin %arg8 : memref<832xui8, 1>
        scf.for %arg14 = %c0_i32_17 to %arg13 step %c1_i32_14  : i32 {
          %332 = arith.muli %arg14, %c256_i32 : i32
          %333 = arith.index_cast %332 : i32 to index
          %result0, %result1 = ascvec.loadalign_deintlv %arg9[%333] {dintlv_width = "b32"} : memref<13312xf32, 1>, index -> <f32>, <f32>
          %334 = arith.addi %332, %c128_i32 : i32
          %335 = arith.index_cast %334 : i32 to index
          %result0_19, %result1_20 = ascvec.loadalign_deintlv %arg9[%335] {dintlv_width = "b32"} : memref<13312xf32, 1>, index -> <f32>, <f32>
          %result0_21, %result1_22 = ascvec.deintlv %result0, %result0_19 : <f32>, <f32> -> <f32>, <f32>
          %result0_23, %result1_24 = ascvec.deintlv %result1, %result1_20 : <f32>, <f32> -> <f32>, <f32>
          %336 = ascvec.abs %result0_21, %317 : <f32>
          %337 = ascvec.reinterpret %336 : <f32> -> <ui32>
          %338 = ascvec.abs %result0_23, %317 : <f32>
          %339 = ascvec.reinterpret %338 : <f32> -> <ui32>
          %340 = ascvec.abs %result1_22, %317 : <f32>
          %341 = ascvec.reinterpret %340 : <f32> -> <ui32>
          %342 = ascvec.abs %result1_24, %317 : <f32>
          %343 = ascvec.reinterpret %342 : <f32> -> <ui32>
          %344 = ascvec.max %337, %339, %317 : <ui32>
          %345 = ascvec.max %341, %343, %317 : <ui32>
          %346 = ascvec.max %344, %345, %317 : <ui32>
          %347 = ascvec.reduce_max_datablock %346, %317 : <ui32>
          %348 = arith.muli %arg14, %c512_i32 : i32
          %349 = arith.index_cast %348 : i32 to index
          %result0_25, %result1_26 = ascvec.loadalign_deintlv %arg10[%349] {dintlv_width = "b16"} : memref<?xui16, 1>, index -> <ui16>, <ui16>
          %350 = arith.addi %348, %c256_i32 : i32
          %351 = arith.index_cast %350 : i32 to index
          %result0_27, %result1_28 = ascvec.loadalign_deintlv %arg10[%351] {dintlv_width = "b16"} : memref<?xui16, 1>, index -> <ui16>, <ui16>
          %352 = ascvec.reinterpret %result1_26 : <ui16> -> <ui8>
          %353 = ascvec.reinterpret %result1_28 : <ui16> -> <ui8>
          %result0_29, %result1_30 = ascvec.deintlv %352, %353 : <ui8>, <ui8> -> <ui8>, <ui8>
          %354 = ascvec.and %result1_30, %321, %318 : <ui8>, <ui8> -> <ui8>
          %355 = ascvec.reinterpret %354 : <ui8> -> <ui32>
          %356 = ascvec.shr %347, 23, %317 : <ui32> -> <ui32>
          %357 = ascvec.cmp_ge_scalar %356, %c255_i32, %317 : <ui32>, i32 -> !ascvec.mask
          %358 = ascvec.max_scalar %356, %c8_i32, %317 : <ui32>, i32
          %359 = ascvec.add_scalar %358, %c-8_i32, %317 : <ui32>, i32
          %360 = ascvec.duplicate_scalar %c255_i32, %357 merge %359 : !ascvec.vreg<ui32> {mode = "merging"} : i32 -> <ui32>
          %361 = ascvec.duplicate_scalar %c254_i32, %317 : i32 -> <ui32>
          %362 = ascvec.sub %361, %360, %317 : <ui32>
          %363 = ascvec.cmp_ne_scalar %360, %c0_i32_17, %317 : <ui32>, i32 -> !ascvec.mask
          %364 = ascvec.shl %362, 23, %363 : <ui32> -> <ui32>
          %365 = ascvec.reinterpret %364 : <ui32> -> <f32>
          %366 = ascvec.gather_reg %365, %320 : <f32>, <ui32> -> <f32>
          %367 = ascvec.reinterpret %337 : <ui32> -> <f32>
          %368 = ascvec.mul %367, %366, %317 : <f32>
          %369 = ascvec.reinterpret %339 : <ui32> -> <f32>
          %370 = ascvec.mul %369, %366, %317 : <f32>
          %371 = ascvec.reinterpret %341 : <ui32> -> <f32>
          %372 = ascvec.mul %371, %366, %317 : <f32>
          %373 = ascvec.reinterpret %343 : <ui32> -> <f32>
          %374 = ascvec.mul %373, %366, %317 : <f32>
          %375 = arith.muli %arg14, %c64_i32_15 : i32
          %376 = arith.index_cast %375 : i32 to index
          %377 = ascvec.loadalign %arg11[%376] : memref<3328xi32, 1>, index -> <i32>
          %378 = ascvec.reinterpret %377 : <i32> -> <ui32>
          %379 = ascvec.shl %378, 8, %317 : <ui32> -> <ui32>
          %380 = ascvec.shr %378, 24, %317 : <ui32> -> <ui32>
          %381 = ascvec.or %379, %380, %317 : <ui32>, <ui32> -> <ui32>
          %382 = ascvec.shr %378, 16, %317 : <ui32> -> <ui32>
          %383 = ascvec.and %381, %328, %317 : <ui32>, <ui32> -> <ui32>
          %384 = ascvec.and %378, %328, %317 : <ui32>, <ui32> -> <ui32>
          %385 = ascvec.shr %381, 16, %317 : <ui32> -> <ui32>
          %386 = ascvec.reinterpret %368 : <f32> -> <ui32>
          %387 = ascvec.reinterpret %386 : <ui32> -> <f32>
          %388 = ascvec.max_scalar %387, %cst, %317 : <f32>, f32
          %389 = ascvec.add %387, %388, %317 : <f32>
          %390 = ascvec.reinterpret %389 : <f32> -> <ui32>
          %391 = ascvec.shr %390, 4, %317 : <ui32> -> <ui32>
          %392 = ascvec.add %391, %382, %317 : <ui32>
          %393 = ascvec.reinterpret %370 : <f32> -> <ui32>
          %394 = ascvec.reinterpret %393 : <ui32> -> <f32>
          %395 = ascvec.max_scalar %394, %cst, %317 : <f32>, f32
          %396 = ascvec.add %394, %395, %317 : <f32>
          %397 = ascvec.reinterpret %396 : <f32> -> <ui32>
          %398 = ascvec.shr %397, 4, %317 : <ui32> -> <ui32>
          %399 = ascvec.add %398, %383, %317 : <ui32>
          %400 = ascvec.reinterpret %372 : <f32> -> <ui32>
          %401 = ascvec.reinterpret %400 : <ui32> -> <f32>
          %402 = ascvec.max_scalar %401, %cst, %317 : <f32>, f32
          %403 = ascvec.add %401, %402, %317 : <f32>
          %404 = ascvec.reinterpret %403 : <f32> -> <ui32>
          %405 = ascvec.shr %404, 4, %317 : <ui32> -> <ui32>
          %406 = ascvec.add %405, %384, %317 : <ui32>
          %407 = ascvec.reinterpret %374 : <f32> -> <ui32>
          %408 = ascvec.reinterpret %407 : <ui32> -> <f32>
          %409 = ascvec.max_scalar %408, %cst, %317 : <f32>, f32
          %410 = ascvec.add %408, %409, %317 : <f32>
          %411 = ascvec.reinterpret %410 : <f32> -> <ui32>
          %412 = ascvec.shr %411, 4, %317 : <ui32> -> <ui32>
          %413 = ascvec.add %412, %385, %317 : <ui32>
          %414 = ascvec.reinterpret %392 : <ui32> -> <ui16>
          %415 = ascvec.reinterpret %399 : <ui32> -> <ui16>
          %result0_31, %result1_32 = ascvec.deintlv %414, %415 : <ui16>, <ui16> -> <ui16>, <ui16>
          %416 = ascvec.reinterpret %406 : <ui32> -> <ui16>
          %417 = ascvec.reinterpret %413 : <ui32> -> <ui16>
          %result0_33, %result1_34 = ascvec.deintlv %416, %417 : <ui16>, <ui16> -> <ui16>, <ui16>
          %418 = ascvec.reinterpret %result1_32 : <ui16> -> <ui8>
          %419 = ascvec.reinterpret %result1_34 : <ui16> -> <ui8>
          %result0_35, %result1_36 = ascvec.deintlv %418, %419 : <ui8>, <ui8> -> <ui8>, <ui8>
          %420 = ascvec.gather_reg %result0_35, %327 : <ui8>, <ui8> -> <ui8>
          %421 = ascvec.reinterpret %420 : <ui8> -> <ui32>
          %422 = ascvec.reinterpret %421 : <ui32> -> <ui8>
          %423 = ascvec.add_scalar %422, %c56_i8, %318 : <ui8>, i8
          %424 = ascvec.cmp_ne_scalar %423, %c-1_i8, %318 : <ui8>, i8 -> !ascvec.mask
          %425 = ascvec.min_scalar %423, %c126_i8, %424 : <ui8>, i8
          %426 = ascvec.reinterpret %425 : <ui8> -> <ui32>
          %427 = ascvec.or %426, %355, %317 : <ui32>, <ui32> -> <ui32>
          %428 = ascvec.reinterpret %366 : <f32> -> <ui32>
          %429 = ascvec.cmp_eq_scalar %428, %c-8388608_i32, %317 : <ui32>, i32 -> !ascvec.mask
          %430 = ascvec.duplicate_scalar %c2139062143_i32, %429 merge %427 : !ascvec.vreg<ui32> {mode = "merging"} : i32 -> <ui32>
          %431 = ascvec.reinterpret %430 : <ui32> -> <ui8>
          ascvec.storealign %arg12[%333], %431, %318 : memref<13312xui8, 1>, index, <ui8>
          %432 = ascvec.reinterpret %360 : <ui32> -> <ui8>
          %433 = ascvec.squeeze_store_reg %432, %331 : <ui8>, !ascvec.mask -> <ui8>
          ascvec.store_unalign %arg8[%c0_16], %433 : memref<832xui8, 1>, <ui8>
        }
        %c0_18 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        scf.for %arg14 = %c0_18 to %c1 step %c1 {
          ascvec.store_unalign_post %arg8[%c0_16] : memref<832xui8, 1>
        }
        ascvec.mem_bar vst_vld
      } {cannir.vf_mode = "simd"}
      ascvec.release_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
      ascvec.copy_ub2ub %alloc_10[%c0], %alloc_5[%c0], %c13312_i32 : memref<13312xui8, 1>, memref<13312xui8, 1>
      ascvec.vf(%alloc_10, %alloc_5 : memref<13312xui8, 1>, memref<13312xui8, 1>) {
      ^bb0(%arg8: memref<13312xui8, 1>, %arg9: memref<13312xui8, 1>):
      } {cannir.vf_mode = "simd"}
      ascvec.copy_ub2ub %alloc_11[%c0], %alloc_7[%c0], %c832_i32 : memref<832xui8, 1>, memref<832xui8, 1>
      ascvec.vf(%alloc_11, %alloc_7 : memref<832xui8, 1>, memref<832xui8, 1>) {
      ^bb0(%arg8: memref<832xui8, 1>, %arg9: memref<832xui8, 1>):
      } {cannir.vf_mode = "simd"}
      %176 = cannir.extract_layout %arg1 : !cannir.tensor<<gm, f32>, <<(-1, -1)>, <(-1, 2)>>, <<(-1, -1)>, <(-1, 2)>>, identity>
      %177 = cannir.extract_shape %176 : <<(-1, -1)>, <(-1, 2)>> -> <(-1, -1)>
      %178 = cannir.extract_dim %177[0] : !cannir.shape<(-1, -1)>
      %179 = arith.minsi %74, %178 : i64
      %180 = arith.subi %178, %179 : i64
      %181 = arith.minsi %180, %c1_i64 : i64
      %182 = arith.cmpi sle, %74, %178 : i64
      %183 = arith.select %182, %181, %c0_i64 : i64
      %184 = cannir.extract_dim %177[1] : !cannir.shape<(-1, -1)>
      %185 = arith.divsi %184, %86 : i64
      %186 = arith.minsi %76, %185 : i64
      %187 = arith.muli %186, %52 : i64
      %188 = arith.subi %184, %187 : i64
      %189 = arith.minsi %52, %188 : i64
      %190 = arith.cmpi sle, %76, %185 : i64
      %191 = arith.select %190, %189, %c0_i64 : i64
      %192 = cannir.extract_stride %176 : <<(-1, -1)>, <(-1, 2)>> -> <(-1, 2)>
      %193 = cannir.extract_dim %192[0] : !cannir.stride<(-1, 2)>
      %194 = cannir.extract_physical_layout %arg1 : !cannir.tensor<<gm, f32>, <<(-1, -1)>, <(-1, 2)>>, <<(-1, -1)>, <(-1, 2)>>, identity>
      %195 = cannir.extract_stride %194 : <<(-1, -1)>, <(-1, 2)>> -> <(-1, 2)>
      %196 = cannir.extract_dim %195[0] : !cannir.stride<(-1, 2)>
      %197 = arith.muli %74, %196 : i64
      %198 = arith.addi %197, %100 : i64
      %199 = arith.muli %198, %c32_i64 : i64
      %200 = ascscalar.memref_address %alloc_0 : memref<1xf32>
      %201 = arith.divui %199, %c8_i64 : i64
      %202 = arith.addi %200, %201 : i64
      %203 = ascscalar.bind_gm_ptr %202 : memref<1xf32>
      %204 = ascscalar.bitcast_memref %alloc_4 byte_offset = %c0_i64 : memref<13312xf32, 1> -> memref<?xf32, 1>
      %205 = arith.cmpi ugt, %183, %c0_i64 : i64
      %206 = arith.cmpi ugt, %191, %c0_i64 : i64
      %207 = arith.andi %205, %206 : i1
      scf.if %207 {
        ascvec.get_buf mte2, %c1_i64, 0 {cannir.buffer.generated_sync}
        %317 = arith.trunci %191 : i64 to i32
        %318 = arith.trunci %183 : i64 to i32
        ascvec.copy_gm2ub_nddma %204[%c0], %203[%c0], %317, %318, %c1_i32, %c1_i32, %c1_i32, %c2_i64, %193, %c0_i64, %c0_i64, %c0_i64, %c1_i64, %c13312_i64, %c0_i64, %c0_i64, %c0_i64 l2_cache_ctl %c0_i32 left_padding(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) right_padding(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) pad_value %c0_i32 : i32 : memref<?xf32, 1>, memref<1xf32>
        ascvec.release_buf mte2, %c1_i64, 0 {cannir.buffer.generated_sync}
      }
      %208 = arith.addi %111, %c1_i64 : i64
      %209 = arith.muli %208, %2 : i64
      %210 = arith.addi %209, %77 : i64
      %211 = arith.divui %210, %c16_i64 : i64
      %212 = arith.trunci %211 : i64 to i32
      %213 = arith.extui %212 : i32 to i64
      %214 = arith.subi %211, %213 : i64
      %215 = arith.remsi %214, %c4294967296_i64 : i64
      %216 = arith.cmpi ne, %215, %c0_i64 : i64
      %217 = arith.cmpi slt, %215, %c0_i64 : i64
      %218 = arith.andi %216, %217 : i1
      %219 = arith.divsi %214, %c4294967296_i64 : i64
      %220 = arith.extui %218 : i1 to i64
      %221 = arith.subi %219, %220 : i64
      %222 = arith.trunci %221 : i64 to i32
      %223 = arith.divui %211, %c16384_i64 : i64
      %224 = arith.addi %211, %168 : i64
      %225 = arith.subi %224, %c1_i64 : i64
      %226 = arith.divui %225, %c16384_i64 : i64
      %227 = arith.cmpi eq, %223, %226 : i64
      scf.if %227 {
        %317 = arith.trunci %223 : i64 to i32
        %318 = arith.extui %317 : i32 to i64
        %319 = arith.subi %223, %318 : i64
        %320 = arith.remsi %319, %c4294967296_i64 : i64
        %321 = arith.cmpi ne, %320, %c0_i64 : i64
        %322 = arith.cmpi slt, %320, %c0_i64 : i64
        %323 = arith.andi %321, %322 : i1
        %324 = arith.divsi %319, %c4294967296_i64 : i64
        %325 = arith.extui %323 : i1 to i64
        %326 = arith.subi %324, %325 : i64
        %327 = arith.trunci %326 : i64 to i32
        %328 = arith.addi %166, %317 : i32
        %329 = arith.addi %167, %327 : i32
        %330 = arith.cmpi ult, %328, %166 : i32
        %331 = arith.extui %330 : i1 to i32
        %332 = arith.addi %329, %331 : i32
        %333 = arith.extui %328 : i32 to i64
        %334 = arith.muli %333, %c3449720151_i64 : i64
        %335 = arith.trunci %334 : i64 to i32
        %336 = arith.extui %335 : i32 to i64
        %337 = arith.subi %334, %336 : i64
        %338 = arith.remsi %337, %c4294967296_i64 : i64
        %339 = arith.cmpi ne, %338, %c0_i64 : i64
        %340 = arith.cmpi slt, %338, %c0_i64 : i64
        %341 = arith.andi %339, %340 : i1
        %342 = arith.divsi %337, %c4294967296_i64 : i64
        %343 = arith.extui %341 : i1 to i64
        %344 = arith.subi %342, %343 : i64
        %345 = arith.trunci %344 : i64 to i32
        %346 = arith.xori %332, %139 : i32
        %347 = arith.extui %346 : i32 to i64
        %348 = arith.muli %347, %c3449720151_i64 : i64
        %349 = arith.trunci %348 : i64 to i32
        %350 = arith.extui %349 : i32 to i64
        %351 = arith.subi %348, %350 : i64
        %352 = arith.remsi %351, %c4294967296_i64 : i64
        %353 = arith.cmpi ne, %352, %c0_i64 : i64
        %354 = arith.cmpi slt, %352, %c0_i64 : i64
        %355 = arith.andi %353, %354 : i1
        %356 = arith.divsi %351, %c4294967296_i64 : i64
        %357 = arith.extui %355 : i1 to i64
        %358 = arith.subi %356, %357 : i64
        %359 = arith.trunci %358 : i64 to i32
        %360 = arith.addi %129, %c-1640531527_i32 : i32
        %361 = arith.addi %139, %c-1150833019_i32 : i32
        %362 = arith.xori %359, %335 : i32
        %363 = arith.xori %362, %360 : i32
        %364 = arith.extui %363 : i32 to i64
        %365 = arith.muli %364, %c3528531795_i64 : i64
        %366 = arith.trunci %365 : i64 to i32
        %367 = arith.extui %366 : i32 to i64
        %368 = arith.subi %365, %367 : i64
        %369 = arith.remsi %368, %c4294967296_i64 : i64
        %370 = arith.cmpi ne, %369, %c0_i64 : i64
        %371 = arith.cmpi slt, %369, %c0_i64 : i64
        %372 = arith.andi %370, %371 : i1
        %373 = arith.divsi %368, %c4294967296_i64 : i64
        %374 = arith.extui %372 : i1 to i64
        %375 = arith.subi %373, %374 : i64
        %376 = arith.trunci %375 : i64 to i32
        %377 = arith.xori %345, %129 : i32
        %378 = arith.addi %129, %c1013904242_i32 : i32
        %379 = arith.xori %349, %378 : i32
        %380 = arith.addi %139, %c1993301258_i32 : i32
        %381 = arith.xori %376, %380 : i32
        ascsimt.vf(%212, %377, %361, %379, %381, %129, %139, %366, %alloc_9, %168 : i32, i32, i32, i32, i32, i32, i32, i32, memref<3328xi32, 1>, i64) {
        ^bb0(%arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: memref<3328xi32, 1>, %arg17: i64):
          %c256_i64_14 = arith.constant 256 : i64
          %c3_i64_15 = arith.constant 3 : i64
          %c2_i64_16 = arith.constant 2 : i64
          %c1_i64_17 = arith.constant 1 : i64
          %c4_i64_18 = arith.constant 4 : i64
          %c1_i32_19 = arith.constant 1 : i32
          %c10_i32 = arith.constant 10 : i32
          %c3_i32 = arith.constant 3 : i32
          %c-1150833019_i32_20 = arith.constant -1150833019 : i32
          %c-1640531527_i32_21 = arith.constant -1640531527 : i32
          %c-845247145_i32 = arith.constant -845247145 : i32
          %c-766435501_i32 = arith.constant -766435501 : i32
          %c16384_i32 = arith.constant 16384 : i32
          %382 = ascsimt.thread_idx {ascsimt.scope = 4 : i64, axis = "x", cannir.vf_threads = 256 : i64} : i32
          %383 = arith.extsi %382 {cannir.vf_threads = 256 : i64} : i32 to i64
          scf.for %arg18 = %383 to %arg17 step %c256_i64_14  : i64 {
            %384 = arith.trunci %arg18 : i64 to i32
            %385 = arith.addi %arg8, %384 : i32
            %386 = arith.remui %385, %c16384_i32 : i32
            %387 = arith.xori %386, %arg9 : i32
            %388 = arith.muli %387, %c-766435501_i32 : i32
            %389 = "ascsimt.bitcast"(%387) : (i32) -> ui32
            %390 = "ascsimt.cast"(%c-766435501_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
            %391 = ascsimt.math_binary %389, %390, "mulhi" : ui32, ui32 -> ui32
            %392 = "ascsimt.bitcast"(%391) : (ui32) -> i32
            %393 = arith.xori %392, %arg10 : i32
            %394 = arith.muli %393, %c-845247145_i32 : i32
            %395 = "ascsimt.bitcast"(%393) : (i32) -> ui32
            %396 = "ascsimt.cast"(%c-845247145_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
            %397 = ascsimt.math_binary %395, %396, "mulhi" : ui32, ui32 -> ui32
            %398 = "ascsimt.bitcast"(%397) : (ui32) -> i32
            %399 = arith.xori %398, %arg11 : i32
            %400 = arith.xori %388, %arg12 : i32
            %401:4 = scf.for %arg19 = %c3_i32 to %c10_i32 step %c1_i32_19 iter_args(%arg20 = %399, %arg21 = %394, %arg22 = %400, %arg23 = %arg15) -> (i32, i32, i32, i32)  : i32 {
              %410 = arith.muli %arg20, %c-766435501_i32 : i32
              %411 = "ascsimt.bitcast"(%arg20) : (i32) -> ui32
              %412 = ascsimt.math_binary %411, %390, "mulhi" : ui32, ui32 -> ui32
              %413 = "ascsimt.bitcast"(%412) : (ui32) -> i32
              %414 = arith.muli %arg22, %c-845247145_i32 : i32
              %415 = "ascsimt.bitcast"(%arg22) : (i32) -> ui32
              %416 = ascsimt.math_binary %415, %396, "mulhi" : ui32, ui32 -> ui32
              %417 = "ascsimt.bitcast"(%416) : (ui32) -> i32
              %418 = arith.muli %arg19, %c-1640531527_i32_21 : i32
              %419 = arith.addi %arg13, %418 : i32
              %420 = arith.muli %arg19, %c-1150833019_i32_20 : i32
              %421 = arith.addi %arg14, %420 : i32
              %422 = arith.xori %417, %arg21 : i32
              %423 = arith.xori %422, %419 : i32
              %424 = arith.xori %413, %arg23 : i32
              %425 = arith.xori %424, %421 : i32
              scf.yield %423, %414, %425, %410 : i32, i32, i32, i32
            }
            %402 = arith.muli %arg18, %c4_i64_18 : i64
            %403 = arith.index_cast %402 : i64 to index
            ascsimt.store %arg16[%403], %401#0 : memref<3328xi32, 1>, i32
            %404 = arith.addi %402, %c1_i64_17 : i64
            %405 = arith.index_cast %404 : i64 to index
            ascsimt.store %arg16[%405], %401#1 : memref<3328xi32, 1>, i32
            %406 = arith.addi %402, %c2_i64_16 : i64
            %407 = arith.index_cast %406 : i64 to index
            ascsimt.store %arg16[%407], %401#2 : memref<3328xi32, 1>, i32
            %408 = arith.addi %402, %c3_i64_15 : i64
            %409 = arith.index_cast %408 : i64 to index
            ascsimt.store %arg16[%409], %401#3 : memref<3328xi32, 1>, i32
          } {cannir.vf_threads = 256 : i64}
        } {threads = 256 : i64}
      } else {
        ascsimt.vf(%212, %222, %166, %167, %129, %139, %alloc_9, %168 : i32, i32, i32, i32, i32, i32, memref<3328xi32, 1>, i64) {
        ^bb0(%arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: memref<3328xi32, 1>, %arg15: i64):
          %c256_i64_14 = arith.constant 256 : i64
          %c3_i64_15 = arith.constant 3 : i64
          %c2_i64_16 = arith.constant 2 : i64
          %c1_i64_17 = arith.constant 1 : i64
          %c4_i64_18 = arith.constant 4 : i64
          %c1_i32_19 = arith.constant 1 : i32
          %c10_i32 = arith.constant 10 : i32
          %c0_i32_20 = arith.constant 0 : i32
          %c-1150833019_i32_21 = arith.constant -1150833019 : i32
          %c-1640531527_i32_22 = arith.constant -1640531527 : i32
          %c-845247145_i32 = arith.constant -845247145 : i32
          %c-766435501_i32 = arith.constant -766435501 : i32
          %c262144_i32 = arith.constant 262144 : i32
          %c16384_i32 = arith.constant 16384 : i32
          %317 = ascsimt.thread_idx {ascsimt.scope = 5 : i64, axis = "x", cannir.vf_threads = 256 : i64} : i32
          %318 = arith.extsi %317 {cannir.vf_threads = 256 : i64} : i32 to i64
          scf.for %arg16 = %318 to %arg15 step %c256_i64_14  : i64 {
            %319 = arith.trunci %arg16 : i64 to i32
            %320 = arith.addi %arg8, %319 : i32
            %321 = arith.cmpi ult, %320, %arg8 : i32
            %322 = arith.extui %321 : i1 to i32
            %323 = arith.addi %arg9, %322 : i32
            %324 = arith.divui %320, %c16384_i32 : i32
            %325 = arith.remui %323, %c16384_i32 : i32
            %326 = arith.muli %325, %c262144_i32 : i32
            %327 = arith.addi %324, %326 : i32
            %328 = arith.divui %323, %c16384_i32 : i32
            %329 = arith.addi %arg10, %327 : i32
            %330 = arith.addi %arg11, %328 : i32
            %331 = arith.cmpi ult, %329, %arg10 : i32
            %332 = arith.extui %331 : i1 to i32
            %333 = arith.addi %330, %332 : i32
            %334 = arith.remui %320, %c16384_i32 : i32
            %335:4 = scf.for %arg17 = %c0_i32_20 to %c10_i32 step %c1_i32_19 iter_args(%arg18 = %c0_i32_20, %arg19 = %334, %arg20 = %329, %arg21 = %333) -> (i32, i32, i32, i32)  : i32 {
              %344 = arith.muli %arg18, %c-766435501_i32 : i32
              %345 = "ascsimt.bitcast"(%arg18) : (i32) -> ui32
              %346 = "ascsimt.cast"(%c-766435501_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
              %347 = ascsimt.math_binary %345, %346, "mulhi" : ui32, ui32 -> ui32
              %348 = "ascsimt.bitcast"(%347) : (ui32) -> i32
              %349 = arith.muli %arg20, %c-845247145_i32 : i32
              %350 = "ascsimt.bitcast"(%arg20) : (i32) -> ui32
              %351 = "ascsimt.cast"(%c-845247145_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
              %352 = ascsimt.math_binary %350, %351, "mulhi" : ui32, ui32 -> ui32
              %353 = "ascsimt.bitcast"(%352) : (ui32) -> i32
              %354 = arith.muli %arg17, %c-1640531527_i32_22 : i32
              %355 = arith.addi %arg12, %354 : i32
              %356 = arith.muli %arg17, %c-1150833019_i32_21 : i32
              %357 = arith.addi %arg13, %356 : i32
              %358 = arith.xori %353, %arg19 : i32
              %359 = arith.xori %358, %355 : i32
              %360 = arith.xori %348, %arg21 : i32
              %361 = arith.xori %360, %357 : i32
              scf.yield %359, %349, %361, %344 : i32, i32, i32, i32
            }
            %336 = arith.muli %arg16, %c4_i64_18 : i64
            %337 = arith.index_cast %336 : i64 to index
            ascsimt.store %arg14[%337], %335#0 : memref<3328xi32, 1>, i32
            %338 = arith.addi %336, %c1_i64_17 : i64
            %339 = arith.index_cast %338 : i64 to index
            ascsimt.store %arg14[%339], %335#1 : memref<3328xi32, 1>, i32
            %340 = arith.addi %336, %c2_i64_16 : i64
            %341 = arith.index_cast %340 : i64 to index
            ascsimt.store %arg14[%341], %335#2 : memref<3328xi32, 1>, i32
            %342 = arith.addi %336, %c3_i64_15 : i64
            %343 = arith.index_cast %342 : i64 to index
            ascsimt.store %arg14[%343], %335#3 : memref<3328xi32, 1>, i32
          } {cannir.vf_threads = 256 : i64}
        } {threads = 256 : i64}
      }
      ascvec.get_buf v, %c1_i64, 0 {cannir.buffer.generated_sync}
      %228 = ascscalar.bitcast_memref %alloc_4 byte_offset = %c0_i64 : memref<13312xf32, 1> -> memref<?xui16, 1>
      ascvec.vf(%alloc_8, %alloc_4, %228, %alloc_9, %alloc_6, %174 : memref<832xui8, 1>, memref<13312xf32, 1>, memref<?xui16, 1>, memref<3328xi32, 1>, memref<13312xui8, 1>, i32) {
      ^bb0(%arg8: memref<832xui8, 1>, %arg9: memref<13312xf32, 1>, %arg10: memref<?xui16, 1>, %arg11: memref<3328xi32, 1>, %arg12: memref<13312xui8, 1>, %arg13: i32):
        %c512_i32 = arith.constant 512 : i32
        %c1_i32_14 = arith.constant 1 : i32
        %c2139062143_i32 = arith.constant 2139062143 : i32
        %c-8388608_i32 = arith.constant -8388608 : i32
        %c126_i8 = arith.constant 126 : i8
        %c-1_i8 = arith.constant -1 : i8
        %c56_i8 = arith.constant 56 : i8
        %cst = arith.constant 0.0156249991 : f32
        %c64_i32_15 = arith.constant 64 : i32
        %c254_i32 = arith.constant 254 : i32
        %c-8_i32 = arith.constant -8 : i32
        %c8_i32 = arith.constant 8 : i32
        %c255_i32 = arith.constant 255 : i32
        %c128_i32 = arith.constant 128 : i32
        %c256_i32 = arith.constant 256 : i32
        %c0_16 = arith.constant 0 : index
        %c1_i8 = arith.constant 1 : i8
        %c32_i8 = arith.constant 32 : i8
        %c65535_i32 = arith.constant 65535 : i32
        %c3_i8 = arith.constant 3 : i8
        %c0_i8 = arith.constant 0 : i8
        %c-128_i8 = arith.constant -128 : i8
        %c0_i32_17 = arith.constant 0 : i32
        %317 = ascvec.create_mask : !ascvec.mask
        %318 = ascvec.create_mask elem_bits 8 : !ascvec.mask
        %319 = ascvec.arange %c0_i32_17 : i32 -> <ui32>
        %320 = ascvec.shr %319, 3, %317 : <ui32> -> <ui32>
        %321 = ascvec.duplicate_scalar %c-128_i8, %318 : i8 -> <ui8>
        %322 = ascvec.arange %c0_i8 : i8 -> <ui8>
        %323 = ascvec.duplicate_scalar %c3_i8, %318 : i8 -> <ui8>
        %324 = ascvec.and %322, %323, %318 : <ui8>, <ui8> -> <ui8>
        %325 = ascvec.shl %324, 6, %318 : <ui8> -> <ui8>
        %326 = ascvec.shr %322, 2, %318 : <ui8> -> <ui8>
        %327 = ascvec.or %325, %326, %318 : <ui8>, <ui8> -> <ui8>
        %328 = ascvec.duplicate_scalar %c65535_i32, %317 : i32 -> <ui32>
        %329 = ascvec.cmp_lt_scalar %322, %c32_i8, %318 : <ui8>, i8 -> !ascvec.mask
        %330 = ascvec.cmp_lt_scalar %324, %c1_i8, %318 : <ui8>, i8 -> !ascvec.mask
        %331 = ascvec.mask_and %329, %330, %318
        ascvec.store_unalign_begin %arg8 : memref<832xui8, 1>
        scf.for %arg14 = %c0_i32_17 to %arg13 step %c1_i32_14  : i32 {
          %332 = arith.muli %arg14, %c256_i32 : i32
          %333 = arith.index_cast %332 : i32 to index
          %result0, %result1 = ascvec.loadalign_deintlv %arg9[%333] {dintlv_width = "b32"} : memref<13312xf32, 1>, index -> <f32>, <f32>
          %334 = arith.addi %332, %c128_i32 : i32
          %335 = arith.index_cast %334 : i32 to index
          %result0_19, %result1_20 = ascvec.loadalign_deintlv %arg9[%335] {dintlv_width = "b32"} : memref<13312xf32, 1>, index -> <f32>, <f32>
          %result0_21, %result1_22 = ascvec.deintlv %result0, %result0_19 : <f32>, <f32> -> <f32>, <f32>
          %result0_23, %result1_24 = ascvec.deintlv %result1, %result1_20 : <f32>, <f32> -> <f32>, <f32>
          %336 = ascvec.abs %result0_21, %317 : <f32>
          %337 = ascvec.reinterpret %336 : <f32> -> <ui32>
          %338 = ascvec.abs %result0_23, %317 : <f32>
          %339 = ascvec.reinterpret %338 : <f32> -> <ui32>
          %340 = ascvec.abs %result1_22, %317 : <f32>
          %341 = ascvec.reinterpret %340 : <f32> -> <ui32>
          %342 = ascvec.abs %result1_24, %317 : <f32>
          %343 = ascvec.reinterpret %342 : <f32> -> <ui32>
          %344 = ascvec.max %337, %339, %317 : <ui32>
          %345 = ascvec.max %341, %343, %317 : <ui32>
          %346 = ascvec.max %344, %345, %317 : <ui32>
          %347 = ascvec.reduce_max_datablock %346, %317 : <ui32>
          %348 = arith.muli %arg14, %c512_i32 : i32
          %349 = arith.index_cast %348 : i32 to index
          %result0_25, %result1_26 = ascvec.loadalign_deintlv %arg10[%349] {dintlv_width = "b16"} : memref<?xui16, 1>, index -> <ui16>, <ui16>
          %350 = arith.addi %348, %c256_i32 : i32
          %351 = arith.index_cast %350 : i32 to index
          %result0_27, %result1_28 = ascvec.loadalign_deintlv %arg10[%351] {dintlv_width = "b16"} : memref<?xui16, 1>, index -> <ui16>, <ui16>
          %352 = ascvec.reinterpret %result1_26 : <ui16> -> <ui8>
          %353 = ascvec.reinterpret %result1_28 : <ui16> -> <ui8>
          %result0_29, %result1_30 = ascvec.deintlv %352, %353 : <ui8>, <ui8> -> <ui8>, <ui8>
          %354 = ascvec.and %result1_30, %321, %318 : <ui8>, <ui8> -> <ui8>
          %355 = ascvec.reinterpret %354 : <ui8> -> <ui32>
          %356 = ascvec.shr %347, 23, %317 : <ui32> -> <ui32>
          %357 = ascvec.cmp_ge_scalar %356, %c255_i32, %317 : <ui32>, i32 -> !ascvec.mask
          %358 = ascvec.max_scalar %356, %c8_i32, %317 : <ui32>, i32
          %359 = ascvec.add_scalar %358, %c-8_i32, %317 : <ui32>, i32
          %360 = ascvec.duplicate_scalar %c255_i32, %357 merge %359 : !ascvec.vreg<ui32> {mode = "merging"} : i32 -> <ui32>
          %361 = ascvec.duplicate_scalar %c254_i32, %317 : i32 -> <ui32>
          %362 = ascvec.sub %361, %360, %317 : <ui32>
          %363 = ascvec.cmp_ne_scalar %360, %c0_i32_17, %317 : <ui32>, i32 -> !ascvec.mask
          %364 = ascvec.shl %362, 23, %363 : <ui32> -> <ui32>
          %365 = ascvec.reinterpret %364 : <ui32> -> <f32>
          %366 = ascvec.gather_reg %365, %320 : <f32>, <ui32> -> <f32>
          %367 = ascvec.reinterpret %337 : <ui32> -> <f32>
          %368 = ascvec.mul %367, %366, %317 : <f32>
          %369 = ascvec.reinterpret %339 : <ui32> -> <f32>
          %370 = ascvec.mul %369, %366, %317 : <f32>
          %371 = ascvec.reinterpret %341 : <ui32> -> <f32>
          %372 = ascvec.mul %371, %366, %317 : <f32>
          %373 = ascvec.reinterpret %343 : <ui32> -> <f32>
          %374 = ascvec.mul %373, %366, %317 : <f32>
          %375 = arith.muli %arg14, %c64_i32_15 : i32
          %376 = arith.index_cast %375 : i32 to index
          %377 = ascvec.loadalign %arg11[%376] : memref<3328xi32, 1>, index -> <i32>
          %378 = ascvec.reinterpret %377 : <i32> -> <ui32>
          %379 = ascvec.shl %378, 8, %317 : <ui32> -> <ui32>
          %380 = ascvec.shr %378, 24, %317 : <ui32> -> <ui32>
          %381 = ascvec.or %379, %380, %317 : <ui32>, <ui32> -> <ui32>
          %382 = ascvec.shr %378, 16, %317 : <ui32> -> <ui32>
          %383 = ascvec.and %381, %328, %317 : <ui32>, <ui32> -> <ui32>
          %384 = ascvec.and %378, %328, %317 : <ui32>, <ui32> -> <ui32>
          %385 = ascvec.shr %381, 16, %317 : <ui32> -> <ui32>
          %386 = ascvec.reinterpret %368 : <f32> -> <ui32>
          %387 = ascvec.reinterpret %386 : <ui32> -> <f32>
          %388 = ascvec.max_scalar %387, %cst, %317 : <f32>, f32
          %389 = ascvec.add %387, %388, %317 : <f32>
          %390 = ascvec.reinterpret %389 : <f32> -> <ui32>
          %391 = ascvec.shr %390, 4, %317 : <ui32> -> <ui32>
          %392 = ascvec.add %391, %382, %317 : <ui32>
          %393 = ascvec.reinterpret %370 : <f32> -> <ui32>
          %394 = ascvec.reinterpret %393 : <ui32> -> <f32>
          %395 = ascvec.max_scalar %394, %cst, %317 : <f32>, f32
          %396 = ascvec.add %394, %395, %317 : <f32>
          %397 = ascvec.reinterpret %396 : <f32> -> <ui32>
          %398 = ascvec.shr %397, 4, %317 : <ui32> -> <ui32>
          %399 = ascvec.add %398, %383, %317 : <ui32>
          %400 = ascvec.reinterpret %372 : <f32> -> <ui32>
          %401 = ascvec.reinterpret %400 : <ui32> -> <f32>
          %402 = ascvec.max_scalar %401, %cst, %317 : <f32>, f32
          %403 = ascvec.add %401, %402, %317 : <f32>
          %404 = ascvec.reinterpret %403 : <f32> -> <ui32>
          %405 = ascvec.shr %404, 4, %317 : <ui32> -> <ui32>
          %406 = ascvec.add %405, %384, %317 : <ui32>
          %407 = ascvec.reinterpret %374 : <f32> -> <ui32>
          %408 = ascvec.reinterpret %407 : <ui32> -> <f32>
          %409 = ascvec.max_scalar %408, %cst, %317 : <f32>, f32
          %410 = ascvec.add %408, %409, %317 : <f32>
          %411 = ascvec.reinterpret %410 : <f32> -> <ui32>
          %412 = ascvec.shr %411, 4, %317 : <ui32> -> <ui32>
          %413 = ascvec.add %412, %385, %317 : <ui32>
          %414 = ascvec.reinterpret %392 : <ui32> -> <ui16>
          %415 = ascvec.reinterpret %399 : <ui32> -> <ui16>
          %result0_31, %result1_32 = ascvec.deintlv %414, %415 : <ui16>, <ui16> -> <ui16>, <ui16>
          %416 = ascvec.reinterpret %406 : <ui32> -> <ui16>
          %417 = ascvec.reinterpret %413 : <ui32> -> <ui16>
          %result0_33, %result1_34 = ascvec.deintlv %416, %417 : <ui16>, <ui16> -> <ui16>, <ui16>
          %418 = ascvec.reinterpret %result1_32 : <ui16> -> <ui8>
          %419 = ascvec.reinterpret %result1_34 : <ui16> -> <ui8>
          %result0_35, %result1_36 = ascvec.deintlv %418, %419 : <ui8>, <ui8> -> <ui8>, <ui8>
          %420 = ascvec.gather_reg %result0_35, %327 : <ui8>, <ui8> -> <ui8>
          %421 = ascvec.reinterpret %420 : <ui8> -> <ui32>
          %422 = ascvec.reinterpret %421 : <ui32> -> <ui8>
          %423 = ascvec.add_scalar %422, %c56_i8, %318 : <ui8>, i8
          %424 = ascvec.cmp_ne_scalar %423, %c-1_i8, %318 : <ui8>, i8 -> !ascvec.mask
          %425 = ascvec.min_scalar %423, %c126_i8, %424 : <ui8>, i8
          %426 = ascvec.reinterpret %425 : <ui8> -> <ui32>
          %427 = ascvec.or %426, %355, %317 : <ui32>, <ui32> -> <ui32>
          %428 = ascvec.reinterpret %366 : <f32> -> <ui32>
          %429 = ascvec.cmp_eq_scalar %428, %c-8388608_i32, %317 : <ui32>, i32 -> !ascvec.mask
          %430 = ascvec.duplicate_scalar %c2139062143_i32, %429 merge %427 : !ascvec.vreg<ui32> {mode = "merging"} : i32 -> <ui32>
          %431 = ascvec.reinterpret %430 : <ui32> -> <ui8>
          ascvec.storealign %arg12[%333], %431, %318 : memref<13312xui8, 1>, index, <ui8>
          %432 = ascvec.reinterpret %360 : <ui32> -> <ui8>
          %433 = ascvec.squeeze_store_reg %432, %331 : <ui8>, !ascvec.mask -> <ui8>
          ascvec.store_unalign %arg8[%c0_16], %433 : memref<832xui8, 1>, <ui8>
        }
        %c0_18 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        scf.for %arg14 = %c0_18 to %c1 step %c1 {
          ascvec.store_unalign_post %arg8[%c0_16] : memref<832xui8, 1>
        }
        ascvec.mem_bar vst_vld
      } {cannir.vf_mode = "simd"}
      ascvec.release_buf v, %c1_i64, 0 {cannir.buffer.generated_sync}
      %229 = arith.trunci %80 : i64 to i32
      %230 = arith.remsi %229, %c64_i32 : i32
      %231 = arith.cmpi ne, %230, %c0_i32 : i32
      %232 = arith.cmpi slt, %230, %c0_i32 : i32
      %233 = arith.andi %231, %232 : i1
      %234 = arith.divsi %229, %c64_i32 : i32
      %235 = arith.extui %233 : i1 to i32
      %236 = arith.subi %234, %235 : i32
      ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
      ascvec.get_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
      %237 = ascscalar.bitcast_memref %alloc_11 byte_offset = %c0_i64 : memref<832xui8, 1> -> memref<?xui16, 1>
      %238 = ascscalar.bitcast_memref %alloc_8 byte_offset = %c0_i64 : memref<832xui8, 1> -> memref<?xui16, 1>
      ascvec.vf(%alloc_10, %alloc_6, %alloc_12, %229, %237, %238, %alloc_13, %236 : memref<13312xui8, 1>, memref<13312xui8, 1>, memref<26624xui8, 1>, i32, memref<?xui16, 1>, memref<?xui16, 1>, memref<832xui16, 1>, i32) {
      ^bb0(%arg8: memref<13312xui8, 1>, %arg9: memref<13312xui8, 1>, %arg10: memref<26624xui8, 1>, %arg11: i32, %arg12: memref<?xui16, 1>, %arg13: memref<?xui16, 1>, %arg14: memref<832xui16, 1>, %arg15: i32):
        %c128_i32 = arith.constant 128 : i32
        %c256_i32 = arith.constant 256 : i32
        %c0_i32_14 = arith.constant 0 : i32
        %c2_i32 = arith.constant 2 : i32
        scf.for %arg16 = %c0_i32_14 to %arg11 step %c256_i32  : i32 {
          %317 = arith.index_cast %arg16 : i32 to index
          %318 = ascvec.loadalign %arg8[%317] : memref<13312xui8, 1>, index -> <ui8>
          %319 = ascvec.loadalign %arg9[%317] : memref<13312xui8, 1>, index -> <ui8>
          %320 = arith.muli %arg16, %c2_i32 : i32
          %321 = arith.index_cast %320 : i32 to index
          ascvec.storealign_intlv %arg10[%321], %318, %319 : memref<26624xui8, 1>, <ui8>, <ui8>
        }
        scf.for %arg16 = %c0_i32_14 to %arg15 step %c128_i32  : i32 {
          %317 = arith.index_cast %arg16 : i32 to index
          %318 = ascvec.loadalign %arg12[%317] : memref<?xui16, 1>, index -> <ui16>
          %319 = ascvec.loadalign %arg13[%317] : memref<?xui16, 1>, index -> <ui16>
          %320 = arith.muli %arg16, %c2_i32 : i32
          %321 = arith.index_cast %320 : i32 to index
          ascvec.storealign_intlv %arg14[%321], %318, %319 : memref<832xui16, 1>, <ui16>, <ui16>
        }
      } {cannir.vf_mode = "simd"}
      ascvec.release_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
      ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
      %239 = arith.muli %50, %c512_i64 : i64
      %240 = cannir.extract_layout %arg2 : !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
      %241 = cannir.extract_shape %240 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, -1)>
      %242 = cannir.extract_dim %241[0] : !cannir.shape<(-1, -1)>
      %243 = arith.minsi %74, %242 : i64
      %244 = arith.subi %242, %243 : i64
      %245 = arith.minsi %244, %c1_i64 : i64
      %246 = arith.cmpi sle, %74, %242 : i64
      %247 = arith.select %246, %245, %c0_i64 : i64
      %248 = arith.cmpi sgt, %247, %c1_i64 : i64
      %249 = arith.select %248, %c1_i64, %247 : i64
      %250 = cannir.extract_dim %241[1] : !cannir.shape<(-1, -1)>
      %251 = arith.maxsi %239, %c1_i64 : i64
      %252 = arith.divsi %250, %251 : i64
      %253 = arith.minsi %76, %252 : i64
      %254 = arith.muli %253, %239 : i64
      %255 = arith.subi %250, %254 : i64
      %256 = arith.minsi %239, %255 : i64
      %257 = arith.cmpi sle, %76, %252 : i64
      %258 = arith.select %257, %256, %c0_i64 : i64
      %259 = arith.cmpi sgt, %258, %c26624_i64 : i64
      %260 = arith.select %259, %c26624_i64, %258 : i64
      %261 = cannir.extract_physical_layout %arg2 : !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
      %262 = cannir.extract_stride %261 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
      %263 = cannir.extract_dim %262[0] : !cannir.stride<(-1, 1)>
      %264 = arith.muli %74, %263 : i64
      %265 = arith.muli %76, %239 : i64
      %266 = arith.addi %264, %265 : i64
      %267 = arith.muli %266, %c8_i64 : i64
      %268 = ascscalar.memref_address %alloc_1 : memref<1xui8>
      %269 = arith.divui %267, %c8_i64 : i64
      %270 = arith.addi %268, %269 : i64
      %271 = ascscalar.bind_gm_ptr %270 : memref<1xui8>
      %272 = ascscalar.bitcast_memref %alloc_12 byte_offset = %c0_i64 : memref<26624xui8, 1> -> memref<?xui8, 1>
      %273 = cannir.extract_stride %240 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
      %274 = cannir.extract_dim %273[0] : !cannir.stride<(-1, 1)>
      %275 = arith.cmpi ugt, %249, %c0_i64 : i64
      %276 = arith.cmpi ugt, %260, %c0_i64 : i64
      %277 = arith.andi %275, %276 : i1
      scf.if %277 {
        ascvec.get_buf mte3, %c2_i64, 0 {cannir.buffer.generated_sync}
        %317 = arith.trunci %249 : i64 to i32
        %318 = arith.trunci %260 : i64 to i32
        ascvec.copy_ub2gm %271[%c0], %272[%c0], %317, %318, %274, %c26624_i32 l2_cache_ctl %c0_i32 : memref<1xui8>, memref<?xui8, 1>
        ascvec.release_buf mte3, %c2_i64, 0 {cannir.buffer.generated_sync}
      }
      %278 = arith.muli %50, %c16_i64 : i64
      %279 = cannir.extract_layout %arg3 : !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
      %280 = cannir.extract_shape %279 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, -1)>
      %281 = cannir.extract_dim %280[0] : !cannir.shape<(-1, -1)>
      %282 = arith.minsi %74, %281 : i64
      %283 = arith.subi %281, %282 : i64
      %284 = arith.minsi %283, %c1_i64 : i64
      %285 = arith.cmpi sle, %74, %281 : i64
      %286 = arith.select %285, %284, %c0_i64 : i64
      %287 = arith.cmpi sgt, %286, %c1_i64 : i64
      %288 = arith.select %287, %c1_i64, %286 : i64
      %289 = cannir.extract_dim %280[1] : !cannir.shape<(-1, -1)>
      %290 = arith.maxsi %278, %c1_i64 : i64
      %291 = arith.divsi %289, %290 : i64
      %292 = arith.minsi %76, %291 : i64
      %293 = arith.muli %292, %278 : i64
      %294 = arith.subi %289, %293 : i64
      %295 = arith.minsi %278, %294 : i64
      %296 = arith.cmpi sle, %76, %291 : i64
      %297 = arith.select %296, %295, %c0_i64 : i64
      %298 = arith.cmpi sgt, %297, %c1664_i64 : i64
      %299 = arith.select %298, %c1664_i64, %297 : i64
      %300 = cannir.extract_physical_layout %arg3 : !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
      %301 = cannir.extract_stride %300 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
      %302 = cannir.extract_dim %301[0] : !cannir.stride<(-1, 1)>
      %303 = arith.muli %74, %302 : i64
      %304 = arith.muli %76, %278 : i64
      %305 = arith.addi %303, %304 : i64
      %306 = arith.muli %305, %c8_i64 : i64
      %307 = ascscalar.memref_address %alloc_2 : memref<1xui8>
      %308 = arith.divui %306, %c8_i64 : i64
      %309 = arith.addi %307, %308 : i64
      %310 = ascscalar.bind_gm_ptr %309 : memref<1xui8>
      %311 = ascscalar.bitcast_memref %alloc_13 byte_offset = %c0_i64 : memref<832xui16, 1> -> memref<?xui8, 1>
      %312 = cannir.extract_stride %279 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
      %313 = cannir.extract_dim %312[0] : !cannir.stride<(-1, 1)>
      %314 = arith.cmpi ugt, %288, %c0_i64 : i64
      %315 = arith.cmpi ugt, %299, %c0_i64 : i64
      %316 = arith.andi %314, %315 : i1
      scf.if %316 {
        ascvec.get_buf mte3, %c3_i64, 0 {cannir.buffer.generated_sync}
        %317 = arith.trunci %288 : i64 to i32
        %318 = arith.trunci %299 : i64 to i32
        ascvec.copy_ub2gm %310[%c0], %311[%c0], %317, %318, %313, %c1664_i32 l2_cache_ctl %c0_i32 : memref<1xui8>, memref<?xui8, 1>
        ascvec.release_buf mte3, %c3_i64, 0 {cannir.buffer.generated_sync}
      }
    }
    return
  }
}
