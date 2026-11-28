module {
  func.func @run_non_tail_fused_graph_rng_63f58c7ed69605bcd895c2416f45e123(%arg0: !cannir.tensor<<gm, f32>, <<(-1, -1)>, <(-1, 2)>>, <<(-1, -1)>, <(-1, 2)>>, identity>, %arg1: !cannir.tensor<<gm, f32>, <<(-1, -1)>, <(-1, 2)>>, <<(-1, -1)>, <(-1, 2)>>, identity>, %arg2: !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg3: !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg4: i64, %arg5: i64, %arg6: !cannir.tensor<<gm, i64>, <<(1)>, <(1)>>, <<(1)>, <(1)>>, identity>, %arg7: !cannir.tensor<<gm, i64>, <<(1)>, <(1)>>, <<(1)>, <(1)>>, identity>, %arg8: i64) attributes {cannbotdsl.contract_digest = "63f58c7ed69605bcd895c2416f45e1233434a36c0af7707d8957e3bca52a49c2", cannbotdsl.public_slot_count = 9 : i64, cannbotdsl.slot_dtype_bits = [32, 32, 8, 8, 64, 64, 64, 64, 64], cannbotdsl.slot_dtype_codes = [2, 2, 1, 1, 0, 0, 0, 0, 0], cannbotdsl.slot_dtype_lanes = [1, 1, 1, 1, 1, 1, 1, 1, 1], cannbotdsl.slot_flags = [1, 1, 1, 1, 1, 1, 1, 1, 1], cannbotdsl.slot_kinds = [1, 1, 1, 1, 2, 2, 1, 1, 2], cannbotdsl.slot_type_digests = ["299b8584bfb7b4ecbed3e029654203d20256497197396c46e927fd078ee9206a", "299b8584bfb7b4ecbed3e029654203d20256497197396c46e927fd078ee9206a", "0466425dd1673c3d734b77386a8d2c26ab0afa3f2945f4af539e71f1f55f41c9", "1d2672dce8d9ff4fcf6d37f57043432984990a93dd5ece4b4944846a06c54afa", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "a9d876affafdf57be3f0e1a45cb0b293a48851dbda8b74101f5184f5130b1b63", "a9d876affafdf57be3f0e1a45cb0b293a48851dbda8b74101f5184f5130b1b63", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515"], cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.host} {
    cannir.kernel_launch @kernel__DynamicMxQuantSrKernel_run_non_tail_fused_0(%arg5) [%arg0, %arg1, %arg2, %arg3, %arg4, %arg6, %arg7, %arg8] {cannbotdsl.profile = {indices = array<i64: 0, 1, 2, 3, 5, 6>, name = "_DynamicMxQuantSrKernel_run_non_tail_fused", op_type = "_DynamicMxQuantSrKernel_run_non_tail_fused", roles = array<i64: 0, 0, 1, 1, 0, 0>}} : !cannir.tensor<<gm, f32>, <<(-1, -1)>, <(-1, 2)>>, <<(-1, -1)>, <(-1, 2)>>, identity>, !cannir.tensor<<gm, f32>, <<(-1, -1)>, <(-1, 2)>>, <<(-1, -1)>, <(-1, 2)>>, identity>, !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, i64, !cannir.tensor<<gm, i64>, <<(1)>, <(1)>>, <<(1)>, <(1)>>, identity>, !cannir.tensor<<gm, i64>, <<(1)>, <(1)>>, <<(1)>, <(1)>>, identity>, i64
    return
  }
  func.func @kernel__DynamicMxQuantSrKernel_run_non_tail_fused_0(%arg0: !cannir.tensor<<gm, f32>, <<(-1, -1)>, <(-1, 2)>>, <<(-1, -1)>, <(-1, 2)>>, identity>, %arg1: !cannir.tensor<<gm, f32>, <<(-1, -1)>, <(-1, 2)>>, <<(-1, -1)>, <(-1, 2)>>, identity>, %arg2: !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg3: !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg4: i64, %arg5: !cannir.tensor<<gm, i64>, <<(1)>, <(1)>>, <<(1)>, <(1)>>, identity>, %arg6: !cannir.tensor<<gm, i64>, <<(1)>, <(1)>>, <<(1)>, <(1)>>, identity>, %arg7: i64) attributes {cannir.buffer.auto_sync, cannir.buffer.sync_analyzed, cannir.buffer.sync_verified, cannir.buffer.tensor_state, cannir.core = "aiv", cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.device, cannir.explicit_ring_buffer_cursors, cannir.kernel_group = "kernel__DynamicMxQuantSrKernel_run_non_tail_fused_0", cannir.resource.ids_prepared} {
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
    %alloc_3 = memref.alloc() {cannir.arg_index = 5 : i64} : memref<1xi64>
    %alloc_4 = memref.alloc() {cannir.arg_index = 6 : i64} : memref<1xi64>
    %alloc_5 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<13312xf32, 1>
    %alloc_6 = memref.alloc() {cannir.byte_offset = 53248 : i64} : memref<13312xf32, 1>
    %alloc_7 = memref.alloc() {cannir.byte_offset = 106496 : i64} : memref<13312xui8, 1>
    %alloc_8 = memref.alloc() {cannir.byte_offset = 119808 : i64} : memref<13312xui8, 1>
    %alloc_9 = memref.alloc() {cannir.byte_offset = 133120 : i64} : memref<832xui8, 1>
    %alloc_10 = memref.alloc() {cannir.byte_offset = 133952 : i64} : memref<832xui8, 1>
    %alloc_11 = memref.alloc() {cannir.byte_offset = 136576 : i64} : memref<3328xi32, 1>
    %alloc_12 = memref.alloc() {cannir.byte_offset = 163200 : i64} : memref<13312xui8, 1>
    %alloc_13 = memref.alloc() {cannir.byte_offset = 176512 : i64} : memref<832xui8, 1>
    %alloc_14 = memref.alloc() {cannir.byte_offset = 177344 : i64} : memref<26624xui8, 1>
    %alloc_15 = memref.alloc() {cannir.byte_offset = 203968 : i64} : memref<832xui16, 1>
    %0 = ascscalar.scalar_ptr %alloc_3[%c0] : memref<1xi64>, index -> <gm, i64>
    %1 = ascscalar.load_dev %0 : !ascscalar.scalar_ptr<gm, i64> -> i64
    %2 = ascscalar.scalar_ptr %alloc_4[%c0] : memref<1xi64>, index -> <gm, i64>
    %3 = ascscalar.load_dev %2 : !ascscalar.scalar_ptr<gm, i64> -> i64
    %4 = arith.addi %3, %arg7 : i64
    %5 = cannir.extract_layout %arg0 : !cannir.tensor<<gm, f32>, <<(-1, -1)>, <(-1, 2)>>, <<(-1, -1)>, <(-1, 2)>>, identity>
    %6 = cannir.extract_shape %5 : <<(-1, -1)>, <(-1, 2)>> -> <(-1, -1)>
    %7 = cannir.extract_dim %6[1] : !cannir.shape<(-1, -1)>
    %8 = arith.remsi %7, %c64_i64 : i64
    %9 = arith.cmpi ne, %8, %c0_i64 : i64
    %10 = arith.cmpi slt, %8, %c0_i64 : i64
    %11 = arith.andi %9, %10 : i1
    %12 = arith.divsi %7, %c64_i64 : i64
    %13 = arith.extui %11 : i1 to i64
    %14 = arith.subi %12, %13 : i64
    %15 = cannir.extract_dim %6[0] : !cannir.shape<(-1, -1)>
    %16 = arith.muli %15, %14 : i64
    %17 = ascscalar.get_block_num
    %18 = arith.addi %16, %17 : i64
    %19 = arith.subi %18, %c1_i64 : i64
    %20 = arith.remsi %19, %17 : i64
    %21 = arith.cmpi ne, %20, %c0_i64 : i64
    %22 = arith.cmpi slt, %20, %c0_i64 : i64
    %23 = arith.cmpi slt, %17, %c0_i64 : i64
    %24 = arith.xori %22, %23 : i1
    %25 = arith.andi %21, %24 : i1
    %26 = arith.divsi %19, %17 : i64
    %27 = arith.extui %25 : i1 to i64
    %28 = arith.subi %26, %27 : i64
    %29 = arith.addi %28, %c207_i64 : i64
    %30 = arith.remsi %29, %c208_i64 : i64
    %31 = arith.cmpi ne, %30, %c0_i64 : i64
    %32 = arith.cmpi slt, %30, %c0_i64 : i64
    %33 = arith.andi %31, %32 : i1
    %34 = arith.divsi %29, %c208_i64 : i64
    %35 = arith.extui %33 : i1 to i64
    %36 = arith.subi %34, %35 : i64
    %37 = arith.addi %28, %36 : i64
    %38 = arith.subi %37, %c1_i64 : i64
    %39 = arith.remsi %38, %36 : i64
    %40 = arith.cmpi ne, %39, %c0_i64 : i64
    %41 = arith.cmpi slt, %39, %c0_i64 : i64
    %42 = arith.cmpi slt, %36, %c0_i64 : i64
    %43 = arith.xori %41, %42 : i1
    %44 = arith.andi %40, %43 : i1
    %45 = arith.divsi %38, %36 : i64
    %46 = arith.extui %44 : i1 to i64
    %47 = arith.subi %45, %46 : i64
    %48 = arith.addi %47, %c3_i64 : i64
    %49 = arith.remsi %48, %c4_i64 : i64
    %50 = arith.cmpi ne, %49, %c0_i64 : i64
    %51 = arith.cmpi slt, %49, %c0_i64 : i64
    %52 = arith.andi %50, %51 : i1
    %53 = arith.divsi %48, %c4_i64 : i64
    %54 = arith.extui %52 : i1 to i64
    %55 = arith.subi %53, %54 : i64
    %56 = arith.muli %55, %c4_i64 : i64
    %57 = arith.muli %55, %c256_i64 : i64
    %58 = arith.addi %14, %56 : i64
    %59 = arith.subi %58, %c1_i64 : i64
    %60 = arith.remsi %59, %56 : i64
    %61 = arith.cmpi ne, %60, %c0_i64 : i64
    %62 = arith.cmpi slt, %60, %c0_i64 : i64
    %63 = arith.cmpi slt, %56, %c0_i64 : i64
    %64 = arith.xori %62, %63 : i1
    %65 = arith.andi %61, %64 : i1
    %66 = arith.divsi %59, %56 : i64
    %67 = arith.extui %65 : i1 to i64
    %68 = arith.subi %66, %67 : i64
    %69 = arith.muli %15, %68 : i64
    %70 = ascscalar.get_block_idx
    scf.for %arg8 = %70 to %69 step %17  : i64 {
      %71 = arith.remsi %arg8, %68 : i64
      %72 = arith.cmpi ne, %71, %c0_i64 : i64
      %73 = arith.cmpi slt, %71, %c0_i64 : i64
      %74 = arith.cmpi slt, %68, %c0_i64 : i64
      %75 = arith.xori %73, %74 : i1
      %76 = arith.andi %72, %75 : i1
      %77 = arith.divsi %arg8, %68 : i64
      %78 = arith.extui %76 : i1 to i64
      %79 = arith.subi %77, %78 : i64
      %80 = arith.select %76, %68, %c0_i64 : i64
      %81 = arith.addi %71, %80 : i64
      %82 = arith.muli %81, %57 : i64
      %83 = arith.subi %7, %82 : i64
      %84 = arith.cmpi sgt, %83, %57 : i64
      %85 = arith.select %84, %57, %83 : i64
      %86 = arith.minsi %79, %15 : i64
      %87 = arith.subi %15, %86 : i64
      %88 = arith.minsi %87, %c1_i64 : i64
      %89 = arith.cmpi sle, %79, %15 : i64
      %90 = arith.select %89, %88, %c0_i64 : i64
      %91 = arith.maxsi %57, %c1_i64 : i64
      %92 = arith.divsi %7, %91 : i64
      %93 = arith.minsi %81, %92 : i64
      %94 = arith.muli %93, %57 : i64
      %95 = arith.subi %7, %94 : i64
      %96 = arith.minsi %57, %95 : i64
      %97 = arith.cmpi sle, %81, %92 : i64
      %98 = arith.select %97, %96, %c0_i64 : i64
      %99 = cannir.extract_stride %5 : <<(-1, -1)>, <(-1, 2)>> -> <(-1, 2)>
      %100 = cannir.extract_dim %99[0] : !cannir.stride<(-1, 2)>
      %101 = cannir.extract_physical_layout %arg0 : !cannir.tensor<<gm, f32>, <<(-1, -1)>, <(-1, 2)>>, <<(-1, -1)>, <(-1, 2)>>, identity>
      %102 = cannir.extract_stride %101 : <<(-1, -1)>, <(-1, 2)>> -> <(-1, 2)>
      %103 = cannir.extract_dim %102[0] : !cannir.stride<(-1, 2)>
      %104 = arith.muli %79, %103 : i64
      %105 = arith.muli %82, %c2_i64 : i64
      %106 = arith.addi %104, %105 : i64
      %107 = arith.muli %106, %c32_i64 : i64
      %108 = ascscalar.memref_address %alloc : memref<1xf32>
      %109 = arith.divui %107, %c8_i64 : i64
      %110 = arith.addi %108, %109 : i64
      %111 = ascscalar.bind_gm_ptr %110 : memref<1xf32>
      %112 = ascscalar.bitcast_memref %alloc_5 byte_offset = %c0_i64 : memref<13312xf32, 1> -> memref<?xf32, 1>
      %113 = arith.cmpi ugt, %90, %c0_i64 : i64
      %114 = arith.cmpi ugt, %98, %c0_i64 : i64
      %115 = arith.andi %113, %114 : i1
      scf.if %115 {
        ascvec.get_buf mte2, %c0_i64, 0 {cannir.buffer.generated_sync}
        %322 = arith.trunci %98 : i64 to i32
        %323 = arith.trunci %90 : i64 to i32
        ascvec.copy_gm2ub_nddma %112[%c0], %111[%c0], %322, %323, %c1_i32, %c1_i32, %c1_i32, %c2_i64, %100, %c0_i64, %c0_i64, %c0_i64, %c1_i64, %c13312_i64, %c0_i64, %c0_i64, %c0_i64 l2_cache_ctl %c0_i32 left_padding(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) right_padding(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) pad_value %c0_i32 : i32 : memref<?xf32, 1>, memref<1xf32>
        ascvec.release_buf mte2, %c0_i64, 0 {cannir.buffer.generated_sync}
      }
      %116 = arith.muli %79, %c2_i64 : i64
      %117 = arith.muli %116, %7 : i64
      %118 = arith.addi %117, %82 : i64
      %119 = arith.remsi %85, %c256_i64 : i64
      %120 = arith.cmpi ne, %119, %c0_i64 : i64
      %121 = arith.cmpi slt, %119, %c0_i64 : i64
      %122 = arith.andi %120, %121 : i1
      %123 = arith.divsi %85, %c256_i64 : i64
      %124 = arith.extui %122 : i1 to i64
      %125 = arith.subi %123, %124 : i64
      %126 = arith.addi %85, %c1023_i64 : i64
      %127 = arith.remsi %126, %c1024_i64 : i64
      %128 = arith.cmpi ne, %127, %c0_i64 : i64
      %129 = arith.cmpi slt, %127, %c0_i64 : i64
      %130 = arith.andi %128, %129 : i1
      %131 = arith.divsi %126, %c1024_i64 : i64
      %132 = arith.extui %130 : i1 to i64
      %133 = arith.subi %131, %132 : i64
      %134 = arith.trunci %1 : i64 to i32
      %135 = arith.extui %134 : i32 to i64
      %136 = arith.subi %1, %135 : i64
      %137 = arith.remsi %136, %c4294967296_i64 : i64
      %138 = arith.cmpi ne, %137, %c0_i64 : i64
      %139 = arith.cmpi slt, %137, %c0_i64 : i64
      %140 = arith.andi %138, %139 : i1
      %141 = arith.divsi %136, %c4294967296_i64 : i64
      %142 = arith.extui %140 : i1 to i64
      %143 = arith.subi %141, %142 : i64
      %144 = arith.trunci %143 : i64 to i32
      %145 = arith.divui %118, %c16_i64 : i64
      %146 = arith.trunci %145 : i64 to i32
      %147 = arith.extui %146 : i32 to i64
      %148 = arith.subi %145, %147 : i64
      %149 = arith.remsi %148, %c4294967296_i64 : i64
      %150 = arith.cmpi ne, %149, %c0_i64 : i64
      %151 = arith.cmpi slt, %149, %c0_i64 : i64
      %152 = arith.andi %150, %151 : i1
      %153 = arith.divsi %148, %c4294967296_i64 : i64
      %154 = arith.extui %152 : i1 to i64
      %155 = arith.subi %153, %154 : i64
      %156 = arith.trunci %155 : i64 to i32
      %157 = arith.trunci %4 : i64 to i32
      %158 = arith.extui %157 : i32 to i64
      %159 = arith.subi %4, %158 : i64
      %160 = arith.remsi %159, %c4294967296_i64 : i64
      %161 = arith.cmpi ne, %160, %c0_i64 : i64
      %162 = arith.cmpi slt, %160, %c0_i64 : i64
      %163 = arith.andi %161, %162 : i1
      %164 = arith.divsi %159, %c4294967296_i64 : i64
      %165 = arith.extui %163 : i1 to i64
      %166 = arith.subi %164, %165 : i64
      %167 = arith.trunci %166 : i64 to i32
      %168 = arith.divui %157, %c4_i32 : i32
      %169 = arith.remui %167, %c4_i32 : i32
      %170 = arith.muli %169, %c1073741824_i32 : i32
      %171 = arith.addi %168, %170 : i32
      %172 = arith.divui %167, %c4_i32 : i32
      %173 = arith.muli %133, %c64_i64 : i64
      %174 = arith.divui %145, %c16384_i64 : i64
      %175 = arith.addi %145, %173 : i64
      %176 = arith.subi %175, %c1_i64 : i64
      %177 = arith.divui %176, %c16384_i64 : i64
      %178 = arith.cmpi eq, %174, %177 : i64
      scf.if %178 {
        %322 = arith.trunci %174 : i64 to i32
        %323 = arith.extui %322 : i32 to i64
        %324 = arith.subi %174, %323 : i64
        %325 = arith.remsi %324, %c4294967296_i64 : i64
        %326 = arith.cmpi ne, %325, %c0_i64 : i64
        %327 = arith.cmpi slt, %325, %c0_i64 : i64
        %328 = arith.andi %326, %327 : i1
        %329 = arith.divsi %324, %c4294967296_i64 : i64
        %330 = arith.extui %328 : i1 to i64
        %331 = arith.subi %329, %330 : i64
        %332 = arith.trunci %331 : i64 to i32
        %333 = arith.addi %171, %322 : i32
        %334 = arith.addi %172, %332 : i32
        %335 = arith.cmpi ult, %333, %171 : i32
        %336 = arith.extui %335 : i1 to i32
        %337 = arith.addi %334, %336 : i32
        %338 = arith.extui %333 : i32 to i64
        %339 = arith.muli %338, %c3449720151_i64 : i64
        %340 = arith.trunci %339 : i64 to i32
        %341 = arith.extui %340 : i32 to i64
        %342 = arith.subi %339, %341 : i64
        %343 = arith.remsi %342, %c4294967296_i64 : i64
        %344 = arith.cmpi ne, %343, %c0_i64 : i64
        %345 = arith.cmpi slt, %343, %c0_i64 : i64
        %346 = arith.andi %344, %345 : i1
        %347 = arith.divsi %342, %c4294967296_i64 : i64
        %348 = arith.extui %346 : i1 to i64
        %349 = arith.subi %347, %348 : i64
        %350 = arith.trunci %349 : i64 to i32
        %351 = arith.xori %337, %144 : i32
        %352 = arith.extui %351 : i32 to i64
        %353 = arith.muli %352, %c3449720151_i64 : i64
        %354 = arith.trunci %353 : i64 to i32
        %355 = arith.extui %354 : i32 to i64
        %356 = arith.subi %353, %355 : i64
        %357 = arith.remsi %356, %c4294967296_i64 : i64
        %358 = arith.cmpi ne, %357, %c0_i64 : i64
        %359 = arith.cmpi slt, %357, %c0_i64 : i64
        %360 = arith.andi %358, %359 : i1
        %361 = arith.divsi %356, %c4294967296_i64 : i64
        %362 = arith.extui %360 : i1 to i64
        %363 = arith.subi %361, %362 : i64
        %364 = arith.trunci %363 : i64 to i32
        %365 = arith.addi %134, %c-1640531527_i32 : i32
        %366 = arith.addi %144, %c-1150833019_i32 : i32
        %367 = arith.xori %364, %340 : i32
        %368 = arith.xori %367, %365 : i32
        %369 = arith.extui %368 : i32 to i64
        %370 = arith.muli %369, %c3528531795_i64 : i64
        %371 = arith.trunci %370 : i64 to i32
        %372 = arith.extui %371 : i32 to i64
        %373 = arith.subi %370, %372 : i64
        %374 = arith.remsi %373, %c4294967296_i64 : i64
        %375 = arith.cmpi ne, %374, %c0_i64 : i64
        %376 = arith.cmpi slt, %374, %c0_i64 : i64
        %377 = arith.andi %375, %376 : i1
        %378 = arith.divsi %373, %c4294967296_i64 : i64
        %379 = arith.extui %377 : i1 to i64
        %380 = arith.subi %378, %379 : i64
        %381 = arith.trunci %380 : i64 to i32
        %382 = arith.xori %350, %134 : i32
        %383 = arith.addi %134, %c1013904242_i32 : i32
        %384 = arith.xori %354, %383 : i32
        %385 = arith.addi %144, %c1993301258_i32 : i32
        %386 = arith.xori %381, %385 : i32
        ascsimt.vf(%146, %382, %366, %384, %386, %134, %144, %371, %alloc_11, %173 : i32, i32, i32, i32, i32, i32, i32, i32, memref<3328xi32, 1>, i64) {
        ^bb0(%arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: memref<3328xi32, 1>, %arg18: i64):
          %c256_i64_16 = arith.constant 256 : i64
          %c3_i64_17 = arith.constant 3 : i64
          %c2_i64_18 = arith.constant 2 : i64
          %c1_i64_19 = arith.constant 1 : i64
          %c4_i64_20 = arith.constant 4 : i64
          %c1_i32_21 = arith.constant 1 : i32
          %c10_i32 = arith.constant 10 : i32
          %c3_i32 = arith.constant 3 : i32
          %c-1150833019_i32_22 = arith.constant -1150833019 : i32
          %c-1640531527_i32_23 = arith.constant -1640531527 : i32
          %c-845247145_i32 = arith.constant -845247145 : i32
          %c-766435501_i32 = arith.constant -766435501 : i32
          %c16384_i32 = arith.constant 16384 : i32
          %387 = ascsimt.thread_idx {ascsimt.scope = 1 : i64, axis = "x", cannir.vf_threads = 256 : i64} : i32
          %388 = arith.extsi %387 {cannir.vf_threads = 256 : i64} : i32 to i64
          scf.for %arg19 = %388 to %arg18 step %c256_i64_16  : i64 {
            %389 = arith.trunci %arg19 : i64 to i32
            %390 = arith.addi %arg9, %389 : i32
            %391 = arith.remui %390, %c16384_i32 : i32
            %392 = arith.xori %391, %arg10 : i32
            %393 = arith.muli %392, %c-766435501_i32 : i32
            %394 = "ascsimt.bitcast"(%392) : (i32) -> ui32
            %395 = "ascsimt.cast"(%c-766435501_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
            %396 = ascsimt.math_binary %394, %395, "mulhi" : ui32, ui32 -> ui32
            %397 = "ascsimt.bitcast"(%396) : (ui32) -> i32
            %398 = arith.xori %397, %arg11 : i32
            %399 = arith.muli %398, %c-845247145_i32 : i32
            %400 = "ascsimt.bitcast"(%398) : (i32) -> ui32
            %401 = "ascsimt.cast"(%c-845247145_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
            %402 = ascsimt.math_binary %400, %401, "mulhi" : ui32, ui32 -> ui32
            %403 = "ascsimt.bitcast"(%402) : (ui32) -> i32
            %404 = arith.xori %403, %arg12 : i32
            %405 = arith.xori %393, %arg13 : i32
            %406:4 = scf.for %arg20 = %c3_i32 to %c10_i32 step %c1_i32_21 iter_args(%arg21 = %404, %arg22 = %399, %arg23 = %405, %arg24 = %arg16) -> (i32, i32, i32, i32)  : i32 {
              %415 = arith.muli %arg21, %c-766435501_i32 : i32
              %416 = "ascsimt.bitcast"(%arg21) : (i32) -> ui32
              %417 = ascsimt.math_binary %416, %395, "mulhi" : ui32, ui32 -> ui32
              %418 = "ascsimt.bitcast"(%417) : (ui32) -> i32
              %419 = arith.muli %arg23, %c-845247145_i32 : i32
              %420 = "ascsimt.bitcast"(%arg23) : (i32) -> ui32
              %421 = ascsimt.math_binary %420, %401, "mulhi" : ui32, ui32 -> ui32
              %422 = "ascsimt.bitcast"(%421) : (ui32) -> i32
              %423 = arith.muli %arg20, %c-1640531527_i32_23 : i32
              %424 = arith.addi %arg14, %423 : i32
              %425 = arith.muli %arg20, %c-1150833019_i32_22 : i32
              %426 = arith.addi %arg15, %425 : i32
              %427 = arith.xori %422, %arg22 : i32
              %428 = arith.xori %427, %424 : i32
              %429 = arith.xori %418, %arg24 : i32
              %430 = arith.xori %429, %426 : i32
              scf.yield %428, %419, %430, %415 : i32, i32, i32, i32
            }
            %407 = arith.muli %arg19, %c4_i64_20 : i64
            %408 = arith.index_cast %407 : i64 to index
            ascsimt.store %arg17[%408], %406#0 : memref<3328xi32, 1>, i32
            %409 = arith.addi %407, %c1_i64_19 : i64
            %410 = arith.index_cast %409 : i64 to index
            ascsimt.store %arg17[%410], %406#1 : memref<3328xi32, 1>, i32
            %411 = arith.addi %407, %c2_i64_18 : i64
            %412 = arith.index_cast %411 : i64 to index
            ascsimt.store %arg17[%412], %406#2 : memref<3328xi32, 1>, i32
            %413 = arith.addi %407, %c3_i64_17 : i64
            %414 = arith.index_cast %413 : i64 to index
            ascsimt.store %arg17[%414], %406#3 : memref<3328xi32, 1>, i32
          } {cannir.vf_threads = 256 : i64}
        } {threads = 256 : i64}
      } else {
        ascsimt.vf(%146, %156, %171, %172, %134, %144, %alloc_11, %173 : i32, i32, i32, i32, i32, i32, memref<3328xi32, 1>, i64) {
        ^bb0(%arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: memref<3328xi32, 1>, %arg16: i64):
          %c256_i64_16 = arith.constant 256 : i64
          %c3_i64_17 = arith.constant 3 : i64
          %c2_i64_18 = arith.constant 2 : i64
          %c1_i64_19 = arith.constant 1 : i64
          %c4_i64_20 = arith.constant 4 : i64
          %c1_i32_21 = arith.constant 1 : i32
          %c10_i32 = arith.constant 10 : i32
          %c0_i32_22 = arith.constant 0 : i32
          %c-1150833019_i32_23 = arith.constant -1150833019 : i32
          %c-1640531527_i32_24 = arith.constant -1640531527 : i32
          %c-845247145_i32 = arith.constant -845247145 : i32
          %c-766435501_i32 = arith.constant -766435501 : i32
          %c262144_i32 = arith.constant 262144 : i32
          %c16384_i32 = arith.constant 16384 : i32
          %322 = ascsimt.thread_idx {ascsimt.scope = 2 : i64, axis = "x", cannir.vf_threads = 256 : i64} : i32
          %323 = arith.extsi %322 {cannir.vf_threads = 256 : i64} : i32 to i64
          scf.for %arg17 = %323 to %arg16 step %c256_i64_16  : i64 {
            %324 = arith.trunci %arg17 : i64 to i32
            %325 = arith.addi %arg9, %324 : i32
            %326 = arith.cmpi ult, %325, %arg9 : i32
            %327 = arith.extui %326 : i1 to i32
            %328 = arith.addi %arg10, %327 : i32
            %329 = arith.divui %325, %c16384_i32 : i32
            %330 = arith.remui %328, %c16384_i32 : i32
            %331 = arith.muli %330, %c262144_i32 : i32
            %332 = arith.addi %329, %331 : i32
            %333 = arith.divui %328, %c16384_i32 : i32
            %334 = arith.addi %arg11, %332 : i32
            %335 = arith.addi %arg12, %333 : i32
            %336 = arith.cmpi ult, %334, %arg11 : i32
            %337 = arith.extui %336 : i1 to i32
            %338 = arith.addi %335, %337 : i32
            %339 = arith.remui %325, %c16384_i32 : i32
            %340:4 = scf.for %arg18 = %c0_i32_22 to %c10_i32 step %c1_i32_21 iter_args(%arg19 = %c0_i32_22, %arg20 = %339, %arg21 = %334, %arg22 = %338) -> (i32, i32, i32, i32)  : i32 {
              %349 = arith.muli %arg19, %c-766435501_i32 : i32
              %350 = "ascsimt.bitcast"(%arg19) : (i32) -> ui32
              %351 = "ascsimt.cast"(%c-766435501_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
              %352 = ascsimt.math_binary %350, %351, "mulhi" : ui32, ui32 -> ui32
              %353 = "ascsimt.bitcast"(%352) : (ui32) -> i32
              %354 = arith.muli %arg21, %c-845247145_i32 : i32
              %355 = "ascsimt.bitcast"(%arg21) : (i32) -> ui32
              %356 = "ascsimt.cast"(%c-845247145_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
              %357 = ascsimt.math_binary %355, %356, "mulhi" : ui32, ui32 -> ui32
              %358 = "ascsimt.bitcast"(%357) : (ui32) -> i32
              %359 = arith.muli %arg18, %c-1640531527_i32_24 : i32
              %360 = arith.addi %arg13, %359 : i32
              %361 = arith.muli %arg18, %c-1150833019_i32_23 : i32
              %362 = arith.addi %arg14, %361 : i32
              %363 = arith.xori %358, %arg20 : i32
              %364 = arith.xori %363, %360 : i32
              %365 = arith.xori %353, %arg22 : i32
              %366 = arith.xori %365, %362 : i32
              scf.yield %364, %354, %366, %349 : i32, i32, i32, i32
            }
            %341 = arith.muli %arg17, %c4_i64_20 : i64
            %342 = arith.index_cast %341 : i64 to index
            ascsimt.store %arg15[%342], %340#0 : memref<3328xi32, 1>, i32
            %343 = arith.addi %341, %c1_i64_19 : i64
            %344 = arith.index_cast %343 : i64 to index
            ascsimt.store %arg15[%344], %340#1 : memref<3328xi32, 1>, i32
            %345 = arith.addi %341, %c2_i64_18 : i64
            %346 = arith.index_cast %345 : i64 to index
            ascsimt.store %arg15[%346], %340#2 : memref<3328xi32, 1>, i32
            %347 = arith.addi %341, %c3_i64_17 : i64
            %348 = arith.index_cast %347 : i64 to index
            ascsimt.store %arg15[%348], %340#3 : memref<3328xi32, 1>, i32
          } {cannir.vf_threads = 256 : i64}
        } {threads = 256 : i64}
      }
      %179 = arith.trunci %125 : i64 to i32
      ascvec.get_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
      %180 = ascscalar.bitcast_memref %alloc_5 byte_offset = %c0_i64 : memref<13312xf32, 1> -> memref<?xui16, 1>
      ascvec.vf(%alloc_9, %alloc_5, %180, %alloc_11, %alloc_7, %179 : memref<832xui8, 1>, memref<13312xf32, 1>, memref<?xui16, 1>, memref<3328xi32, 1>, memref<13312xui8, 1>, i32) {
      ^bb0(%arg9: memref<832xui8, 1>, %arg10: memref<13312xf32, 1>, %arg11: memref<?xui16, 1>, %arg12: memref<3328xi32, 1>, %arg13: memref<13312xui8, 1>, %arg14: i32):
        %c512_i32 = arith.constant 512 : i32
        %c1_i32_16 = arith.constant 1 : i32
        %c2139062143_i32 = arith.constant 2139062143 : i32
        %c-8388608_i32 = arith.constant -8388608 : i32
        %c126_i8 = arith.constant 126 : i8
        %c-1_i8 = arith.constant -1 : i8
        %c56_i8 = arith.constant 56 : i8
        %cst = arith.constant 0.0156249991 : f32
        %c64_i32_17 = arith.constant 64 : i32
        %c254_i32 = arith.constant 254 : i32
        %c-8_i32 = arith.constant -8 : i32
        %c8_i32 = arith.constant 8 : i32
        %c255_i32 = arith.constant 255 : i32
        %c128_i32 = arith.constant 128 : i32
        %c256_i32 = arith.constant 256 : i32
        %c0_18 = arith.constant 0 : index
        %c1_i8 = arith.constant 1 : i8
        %c32_i8 = arith.constant 32 : i8
        %c65535_i32 = arith.constant 65535 : i32
        %c3_i8 = arith.constant 3 : i8
        %c0_i8 = arith.constant 0 : i8
        %c-128_i8 = arith.constant -128 : i8
        %c0_i32_19 = arith.constant 0 : i32
        %322 = ascvec.create_mask : !ascvec.mask
        %323 = ascvec.create_mask elem_bits 8 : !ascvec.mask
        %324 = ascvec.arange %c0_i32_19 : i32 -> <ui32>
        %325 = ascvec.shr %324, 3, %322 : <ui32> -> <ui32>
        %326 = ascvec.duplicate_scalar %c-128_i8, %323 : i8 -> <ui8>
        %327 = ascvec.arange %c0_i8 : i8 -> <ui8>
        %328 = ascvec.duplicate_scalar %c3_i8, %323 : i8 -> <ui8>
        %329 = ascvec.and %327, %328, %323 : <ui8>, <ui8> -> <ui8>
        %330 = ascvec.shl %329, 6, %323 : <ui8> -> <ui8>
        %331 = ascvec.shr %327, 2, %323 : <ui8> -> <ui8>
        %332 = ascvec.or %330, %331, %323 : <ui8>, <ui8> -> <ui8>
        %333 = ascvec.duplicate_scalar %c65535_i32, %322 : i32 -> <ui32>
        %334 = ascvec.cmp_lt_scalar %327, %c32_i8, %323 : <ui8>, i8 -> !ascvec.mask
        %335 = ascvec.cmp_lt_scalar %329, %c1_i8, %323 : <ui8>, i8 -> !ascvec.mask
        %336 = ascvec.mask_and %334, %335, %323
        ascvec.store_unalign_begin %arg9 : memref<832xui8, 1>
        scf.for %arg15 = %c0_i32_19 to %arg14 step %c1_i32_16  : i32 {
          %337 = arith.muli %arg15, %c256_i32 : i32
          %338 = arith.index_cast %337 : i32 to index
          %result0, %result1 = ascvec.loadalign_deintlv %arg10[%338] {dintlv_width = "b32"} : memref<13312xf32, 1>, index -> <f32>, <f32>
          %339 = arith.addi %337, %c128_i32 : i32
          %340 = arith.index_cast %339 : i32 to index
          %result0_21, %result1_22 = ascvec.loadalign_deintlv %arg10[%340] {dintlv_width = "b32"} : memref<13312xf32, 1>, index -> <f32>, <f32>
          %result0_23, %result1_24 = ascvec.deintlv %result0, %result0_21 : <f32>, <f32> -> <f32>, <f32>
          %result0_25, %result1_26 = ascvec.deintlv %result1, %result1_22 : <f32>, <f32> -> <f32>, <f32>
          %341 = ascvec.abs %result0_23, %322 : <f32>
          %342 = ascvec.reinterpret %341 : <f32> -> <ui32>
          %343 = ascvec.abs %result0_25, %322 : <f32>
          %344 = ascvec.reinterpret %343 : <f32> -> <ui32>
          %345 = ascvec.abs %result1_24, %322 : <f32>
          %346 = ascvec.reinterpret %345 : <f32> -> <ui32>
          %347 = ascvec.abs %result1_26, %322 : <f32>
          %348 = ascvec.reinterpret %347 : <f32> -> <ui32>
          %349 = ascvec.max %342, %344, %322 : <ui32>
          %350 = ascvec.max %346, %348, %322 : <ui32>
          %351 = ascvec.max %349, %350, %322 : <ui32>
          %352 = ascvec.reduce_max_datablock %351, %322 : <ui32>
          %353 = arith.muli %arg15, %c512_i32 : i32
          %354 = arith.index_cast %353 : i32 to index
          %result0_27, %result1_28 = ascvec.loadalign_deintlv %arg11[%354] {dintlv_width = "b16"} : memref<?xui16, 1>, index -> <ui16>, <ui16>
          %355 = arith.addi %353, %c256_i32 : i32
          %356 = arith.index_cast %355 : i32 to index
          %result0_29, %result1_30 = ascvec.loadalign_deintlv %arg11[%356] {dintlv_width = "b16"} : memref<?xui16, 1>, index -> <ui16>, <ui16>
          %357 = ascvec.reinterpret %result1_28 : <ui16> -> <ui8>
          %358 = ascvec.reinterpret %result1_30 : <ui16> -> <ui8>
          %result0_31, %result1_32 = ascvec.deintlv %357, %358 : <ui8>, <ui8> -> <ui8>, <ui8>
          %359 = ascvec.and %result1_32, %326, %323 : <ui8>, <ui8> -> <ui8>
          %360 = ascvec.reinterpret %359 : <ui8> -> <ui32>
          %361 = ascvec.shr %352, 23, %322 : <ui32> -> <ui32>
          %362 = ascvec.cmp_ge_scalar %361, %c255_i32, %322 : <ui32>, i32 -> !ascvec.mask
          %363 = ascvec.max_scalar %361, %c8_i32, %322 : <ui32>, i32
          %364 = ascvec.add_scalar %363, %c-8_i32, %322 : <ui32>, i32
          %365 = ascvec.duplicate_scalar %c255_i32, %362 merge %364 : !ascvec.vreg<ui32> {mode = "merging"} : i32 -> <ui32>
          %366 = ascvec.duplicate_scalar %c254_i32, %322 : i32 -> <ui32>
          %367 = ascvec.sub %366, %365, %322 : <ui32>
          %368 = ascvec.cmp_ne_scalar %365, %c0_i32_19, %322 : <ui32>, i32 -> !ascvec.mask
          %369 = ascvec.shl %367, 23, %368 : <ui32> -> <ui32>
          %370 = ascvec.reinterpret %369 : <ui32> -> <f32>
          %371 = ascvec.gather_reg %370, %325 : <f32>, <ui32> -> <f32>
          %372 = ascvec.reinterpret %342 : <ui32> -> <f32>
          %373 = ascvec.mul %372, %371, %322 : <f32>
          %374 = ascvec.reinterpret %344 : <ui32> -> <f32>
          %375 = ascvec.mul %374, %371, %322 : <f32>
          %376 = ascvec.reinterpret %346 : <ui32> -> <f32>
          %377 = ascvec.mul %376, %371, %322 : <f32>
          %378 = ascvec.reinterpret %348 : <ui32> -> <f32>
          %379 = ascvec.mul %378, %371, %322 : <f32>
          %380 = arith.muli %arg15, %c64_i32_17 : i32
          %381 = arith.index_cast %380 : i32 to index
          %382 = ascvec.loadalign %arg12[%381] : memref<3328xi32, 1>, index -> <i32>
          %383 = ascvec.reinterpret %382 : <i32> -> <ui32>
          %384 = ascvec.shl %383, 8, %322 : <ui32> -> <ui32>
          %385 = ascvec.shr %383, 24, %322 : <ui32> -> <ui32>
          %386 = ascvec.or %384, %385, %322 : <ui32>, <ui32> -> <ui32>
          %387 = ascvec.shr %383, 16, %322 : <ui32> -> <ui32>
          %388 = ascvec.and %386, %333, %322 : <ui32>, <ui32> -> <ui32>
          %389 = ascvec.and %383, %333, %322 : <ui32>, <ui32> -> <ui32>
          %390 = ascvec.shr %386, 16, %322 : <ui32> -> <ui32>
          %391 = ascvec.reinterpret %373 : <f32> -> <ui32>
          %392 = ascvec.reinterpret %391 : <ui32> -> <f32>
          %393 = ascvec.max_scalar %392, %cst, %322 : <f32>, f32
          %394 = ascvec.add %392, %393, %322 : <f32>
          %395 = ascvec.reinterpret %394 : <f32> -> <ui32>
          %396 = ascvec.shr %395, 4, %322 : <ui32> -> <ui32>
          %397 = ascvec.add %396, %387, %322 : <ui32>
          %398 = ascvec.reinterpret %375 : <f32> -> <ui32>
          %399 = ascvec.reinterpret %398 : <ui32> -> <f32>
          %400 = ascvec.max_scalar %399, %cst, %322 : <f32>, f32
          %401 = ascvec.add %399, %400, %322 : <f32>
          %402 = ascvec.reinterpret %401 : <f32> -> <ui32>
          %403 = ascvec.shr %402, 4, %322 : <ui32> -> <ui32>
          %404 = ascvec.add %403, %388, %322 : <ui32>
          %405 = ascvec.reinterpret %377 : <f32> -> <ui32>
          %406 = ascvec.reinterpret %405 : <ui32> -> <f32>
          %407 = ascvec.max_scalar %406, %cst, %322 : <f32>, f32
          %408 = ascvec.add %406, %407, %322 : <f32>
          %409 = ascvec.reinterpret %408 : <f32> -> <ui32>
          %410 = ascvec.shr %409, 4, %322 : <ui32> -> <ui32>
          %411 = ascvec.add %410, %389, %322 : <ui32>
          %412 = ascvec.reinterpret %379 : <f32> -> <ui32>
          %413 = ascvec.reinterpret %412 : <ui32> -> <f32>
          %414 = ascvec.max_scalar %413, %cst, %322 : <f32>, f32
          %415 = ascvec.add %413, %414, %322 : <f32>
          %416 = ascvec.reinterpret %415 : <f32> -> <ui32>
          %417 = ascvec.shr %416, 4, %322 : <ui32> -> <ui32>
          %418 = ascvec.add %417, %390, %322 : <ui32>
          %419 = ascvec.reinterpret %397 : <ui32> -> <ui16>
          %420 = ascvec.reinterpret %404 : <ui32> -> <ui16>
          %result0_33, %result1_34 = ascvec.deintlv %419, %420 : <ui16>, <ui16> -> <ui16>, <ui16>
          %421 = ascvec.reinterpret %411 : <ui32> -> <ui16>
          %422 = ascvec.reinterpret %418 : <ui32> -> <ui16>
          %result0_35, %result1_36 = ascvec.deintlv %421, %422 : <ui16>, <ui16> -> <ui16>, <ui16>
          %423 = ascvec.reinterpret %result1_34 : <ui16> -> <ui8>
          %424 = ascvec.reinterpret %result1_36 : <ui16> -> <ui8>
          %result0_37, %result1_38 = ascvec.deintlv %423, %424 : <ui8>, <ui8> -> <ui8>, <ui8>
          %425 = ascvec.gather_reg %result0_37, %332 : <ui8>, <ui8> -> <ui8>
          %426 = ascvec.reinterpret %425 : <ui8> -> <ui32>
          %427 = ascvec.reinterpret %426 : <ui32> -> <ui8>
          %428 = ascvec.add_scalar %427, %c56_i8, %323 : <ui8>, i8
          %429 = ascvec.cmp_ne_scalar %428, %c-1_i8, %323 : <ui8>, i8 -> !ascvec.mask
          %430 = ascvec.min_scalar %428, %c126_i8, %429 : <ui8>, i8
          %431 = ascvec.reinterpret %430 : <ui8> -> <ui32>
          %432 = ascvec.or %431, %360, %322 : <ui32>, <ui32> -> <ui32>
          %433 = ascvec.reinterpret %371 : <f32> -> <ui32>
          %434 = ascvec.cmp_eq_scalar %433, %c-8388608_i32, %322 : <ui32>, i32 -> !ascvec.mask
          %435 = ascvec.duplicate_scalar %c2139062143_i32, %434 merge %432 : !ascvec.vreg<ui32> {mode = "merging"} : i32 -> <ui32>
          %436 = ascvec.reinterpret %435 : <ui32> -> <ui8>
          ascvec.storealign %arg13[%338], %436, %323 : memref<13312xui8, 1>, index, <ui8>
          %437 = ascvec.reinterpret %365 : <ui32> -> <ui8>
          %438 = ascvec.squeeze_store_reg %437, %336 : <ui8>, !ascvec.mask -> <ui8>
          ascvec.store_unalign %arg9[%c0_18], %438 : memref<832xui8, 1>, <ui8>
        }
        %c0_20 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        scf.for %arg15 = %c0_20 to %c1 step %c1 {
          ascvec.store_unalign_post %arg9[%c0_18] : memref<832xui8, 1>
        }
        ascvec.mem_bar vst_vld
      } {cannir.vf_mode = "simd"}
      ascvec.release_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
      ascvec.copy_ub2ub %alloc_12[%c0], %alloc_7[%c0], %c13312_i32 : memref<13312xui8, 1>, memref<13312xui8, 1>
      ascvec.vf(%alloc_12, %alloc_7 : memref<13312xui8, 1>, memref<13312xui8, 1>) {
      ^bb0(%arg9: memref<13312xui8, 1>, %arg10: memref<13312xui8, 1>):
      } {cannir.vf_mode = "simd"}
      ascvec.copy_ub2ub %alloc_13[%c0], %alloc_9[%c0], %c832_i32 : memref<832xui8, 1>, memref<832xui8, 1>
      ascvec.vf(%alloc_13, %alloc_9 : memref<832xui8, 1>, memref<832xui8, 1>) {
      ^bb0(%arg9: memref<832xui8, 1>, %arg10: memref<832xui8, 1>):
      } {cannir.vf_mode = "simd"}
      %181 = cannir.extract_layout %arg1 : !cannir.tensor<<gm, f32>, <<(-1, -1)>, <(-1, 2)>>, <<(-1, -1)>, <(-1, 2)>>, identity>
      %182 = cannir.extract_shape %181 : <<(-1, -1)>, <(-1, 2)>> -> <(-1, -1)>
      %183 = cannir.extract_dim %182[0] : !cannir.shape<(-1, -1)>
      %184 = arith.minsi %79, %183 : i64
      %185 = arith.subi %183, %184 : i64
      %186 = arith.minsi %185, %c1_i64 : i64
      %187 = arith.cmpi sle, %79, %183 : i64
      %188 = arith.select %187, %186, %c0_i64 : i64
      %189 = cannir.extract_dim %182[1] : !cannir.shape<(-1, -1)>
      %190 = arith.divsi %189, %91 : i64
      %191 = arith.minsi %81, %190 : i64
      %192 = arith.muli %191, %57 : i64
      %193 = arith.subi %189, %192 : i64
      %194 = arith.minsi %57, %193 : i64
      %195 = arith.cmpi sle, %81, %190 : i64
      %196 = arith.select %195, %194, %c0_i64 : i64
      %197 = cannir.extract_stride %181 : <<(-1, -1)>, <(-1, 2)>> -> <(-1, 2)>
      %198 = cannir.extract_dim %197[0] : !cannir.stride<(-1, 2)>
      %199 = cannir.extract_physical_layout %arg1 : !cannir.tensor<<gm, f32>, <<(-1, -1)>, <(-1, 2)>>, <<(-1, -1)>, <(-1, 2)>>, identity>
      %200 = cannir.extract_stride %199 : <<(-1, -1)>, <(-1, 2)>> -> <(-1, 2)>
      %201 = cannir.extract_dim %200[0] : !cannir.stride<(-1, 2)>
      %202 = arith.muli %79, %201 : i64
      %203 = arith.addi %202, %105 : i64
      %204 = arith.muli %203, %c32_i64 : i64
      %205 = ascscalar.memref_address %alloc_0 : memref<1xf32>
      %206 = arith.divui %204, %c8_i64 : i64
      %207 = arith.addi %205, %206 : i64
      %208 = ascscalar.bind_gm_ptr %207 : memref<1xf32>
      %209 = ascscalar.bitcast_memref %alloc_6 byte_offset = %c0_i64 : memref<13312xf32, 1> -> memref<?xf32, 1>
      %210 = arith.cmpi ugt, %188, %c0_i64 : i64
      %211 = arith.cmpi ugt, %196, %c0_i64 : i64
      %212 = arith.andi %210, %211 : i1
      scf.if %212 {
        ascvec.get_buf mte2, %c1_i64, 0 {cannir.buffer.generated_sync}
        %322 = arith.trunci %196 : i64 to i32
        %323 = arith.trunci %188 : i64 to i32
        ascvec.copy_gm2ub_nddma %209[%c0], %208[%c0], %322, %323, %c1_i32, %c1_i32, %c1_i32, %c2_i64, %198, %c0_i64, %c0_i64, %c0_i64, %c1_i64, %c13312_i64, %c0_i64, %c0_i64, %c0_i64 l2_cache_ctl %c0_i32 left_padding(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) right_padding(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) pad_value %c0_i32 : i32 : memref<?xf32, 1>, memref<1xf32>
        ascvec.release_buf mte2, %c1_i64, 0 {cannir.buffer.generated_sync}
      }
      %213 = arith.addi %116, %c1_i64 : i64
      %214 = arith.muli %213, %7 : i64
      %215 = arith.addi %214, %82 : i64
      %216 = arith.divui %215, %c16_i64 : i64
      %217 = arith.trunci %216 : i64 to i32
      %218 = arith.extui %217 : i32 to i64
      %219 = arith.subi %216, %218 : i64
      %220 = arith.remsi %219, %c4294967296_i64 : i64
      %221 = arith.cmpi ne, %220, %c0_i64 : i64
      %222 = arith.cmpi slt, %220, %c0_i64 : i64
      %223 = arith.andi %221, %222 : i1
      %224 = arith.divsi %219, %c4294967296_i64 : i64
      %225 = arith.extui %223 : i1 to i64
      %226 = arith.subi %224, %225 : i64
      %227 = arith.trunci %226 : i64 to i32
      %228 = arith.divui %216, %c16384_i64 : i64
      %229 = arith.addi %216, %173 : i64
      %230 = arith.subi %229, %c1_i64 : i64
      %231 = arith.divui %230, %c16384_i64 : i64
      %232 = arith.cmpi eq, %228, %231 : i64
      scf.if %232 {
        %322 = arith.trunci %228 : i64 to i32
        %323 = arith.extui %322 : i32 to i64
        %324 = arith.subi %228, %323 : i64
        %325 = arith.remsi %324, %c4294967296_i64 : i64
        %326 = arith.cmpi ne, %325, %c0_i64 : i64
        %327 = arith.cmpi slt, %325, %c0_i64 : i64
        %328 = arith.andi %326, %327 : i1
        %329 = arith.divsi %324, %c4294967296_i64 : i64
        %330 = arith.extui %328 : i1 to i64
        %331 = arith.subi %329, %330 : i64
        %332 = arith.trunci %331 : i64 to i32
        %333 = arith.addi %171, %322 : i32
        %334 = arith.addi %172, %332 : i32
        %335 = arith.cmpi ult, %333, %171 : i32
        %336 = arith.extui %335 : i1 to i32
        %337 = arith.addi %334, %336 : i32
        %338 = arith.extui %333 : i32 to i64
        %339 = arith.muli %338, %c3449720151_i64 : i64
        %340 = arith.trunci %339 : i64 to i32
        %341 = arith.extui %340 : i32 to i64
        %342 = arith.subi %339, %341 : i64
        %343 = arith.remsi %342, %c4294967296_i64 : i64
        %344 = arith.cmpi ne, %343, %c0_i64 : i64
        %345 = arith.cmpi slt, %343, %c0_i64 : i64
        %346 = arith.andi %344, %345 : i1
        %347 = arith.divsi %342, %c4294967296_i64 : i64
        %348 = arith.extui %346 : i1 to i64
        %349 = arith.subi %347, %348 : i64
        %350 = arith.trunci %349 : i64 to i32
        %351 = arith.xori %337, %144 : i32
        %352 = arith.extui %351 : i32 to i64
        %353 = arith.muli %352, %c3449720151_i64 : i64
        %354 = arith.trunci %353 : i64 to i32
        %355 = arith.extui %354 : i32 to i64
        %356 = arith.subi %353, %355 : i64
        %357 = arith.remsi %356, %c4294967296_i64 : i64
        %358 = arith.cmpi ne, %357, %c0_i64 : i64
        %359 = arith.cmpi slt, %357, %c0_i64 : i64
        %360 = arith.andi %358, %359 : i1
        %361 = arith.divsi %356, %c4294967296_i64 : i64
        %362 = arith.extui %360 : i1 to i64
        %363 = arith.subi %361, %362 : i64
        %364 = arith.trunci %363 : i64 to i32
        %365 = arith.addi %134, %c-1640531527_i32 : i32
        %366 = arith.addi %144, %c-1150833019_i32 : i32
        %367 = arith.xori %364, %340 : i32
        %368 = arith.xori %367, %365 : i32
        %369 = arith.extui %368 : i32 to i64
        %370 = arith.muli %369, %c3528531795_i64 : i64
        %371 = arith.trunci %370 : i64 to i32
        %372 = arith.extui %371 : i32 to i64
        %373 = arith.subi %370, %372 : i64
        %374 = arith.remsi %373, %c4294967296_i64 : i64
        %375 = arith.cmpi ne, %374, %c0_i64 : i64
        %376 = arith.cmpi slt, %374, %c0_i64 : i64
        %377 = arith.andi %375, %376 : i1
        %378 = arith.divsi %373, %c4294967296_i64 : i64
        %379 = arith.extui %377 : i1 to i64
        %380 = arith.subi %378, %379 : i64
        %381 = arith.trunci %380 : i64 to i32
        %382 = arith.xori %350, %134 : i32
        %383 = arith.addi %134, %c1013904242_i32 : i32
        %384 = arith.xori %354, %383 : i32
        %385 = arith.addi %144, %c1993301258_i32 : i32
        %386 = arith.xori %381, %385 : i32
        ascsimt.vf(%217, %382, %366, %384, %386, %134, %144, %371, %alloc_11, %173 : i32, i32, i32, i32, i32, i32, i32, i32, memref<3328xi32, 1>, i64) {
        ^bb0(%arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: memref<3328xi32, 1>, %arg18: i64):
          %c256_i64_16 = arith.constant 256 : i64
          %c3_i64_17 = arith.constant 3 : i64
          %c2_i64_18 = arith.constant 2 : i64
          %c1_i64_19 = arith.constant 1 : i64
          %c4_i64_20 = arith.constant 4 : i64
          %c1_i32_21 = arith.constant 1 : i32
          %c10_i32 = arith.constant 10 : i32
          %c3_i32 = arith.constant 3 : i32
          %c-1150833019_i32_22 = arith.constant -1150833019 : i32
          %c-1640531527_i32_23 = arith.constant -1640531527 : i32
          %c-845247145_i32 = arith.constant -845247145 : i32
          %c-766435501_i32 = arith.constant -766435501 : i32
          %c16384_i32 = arith.constant 16384 : i32
          %387 = ascsimt.thread_idx {ascsimt.scope = 4 : i64, axis = "x", cannir.vf_threads = 256 : i64} : i32
          %388 = arith.extsi %387 {cannir.vf_threads = 256 : i64} : i32 to i64
          scf.for %arg19 = %388 to %arg18 step %c256_i64_16  : i64 {
            %389 = arith.trunci %arg19 : i64 to i32
            %390 = arith.addi %arg9, %389 : i32
            %391 = arith.remui %390, %c16384_i32 : i32
            %392 = arith.xori %391, %arg10 : i32
            %393 = arith.muli %392, %c-766435501_i32 : i32
            %394 = "ascsimt.bitcast"(%392) : (i32) -> ui32
            %395 = "ascsimt.cast"(%c-766435501_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
            %396 = ascsimt.math_binary %394, %395, "mulhi" : ui32, ui32 -> ui32
            %397 = "ascsimt.bitcast"(%396) : (ui32) -> i32
            %398 = arith.xori %397, %arg11 : i32
            %399 = arith.muli %398, %c-845247145_i32 : i32
            %400 = "ascsimt.bitcast"(%398) : (i32) -> ui32
            %401 = "ascsimt.cast"(%c-845247145_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
            %402 = ascsimt.math_binary %400, %401, "mulhi" : ui32, ui32 -> ui32
            %403 = "ascsimt.bitcast"(%402) : (ui32) -> i32
            %404 = arith.xori %403, %arg12 : i32
            %405 = arith.xori %393, %arg13 : i32
            %406:4 = scf.for %arg20 = %c3_i32 to %c10_i32 step %c1_i32_21 iter_args(%arg21 = %404, %arg22 = %399, %arg23 = %405, %arg24 = %arg16) -> (i32, i32, i32, i32)  : i32 {
              %415 = arith.muli %arg21, %c-766435501_i32 : i32
              %416 = "ascsimt.bitcast"(%arg21) : (i32) -> ui32
              %417 = ascsimt.math_binary %416, %395, "mulhi" : ui32, ui32 -> ui32
              %418 = "ascsimt.bitcast"(%417) : (ui32) -> i32
              %419 = arith.muli %arg23, %c-845247145_i32 : i32
              %420 = "ascsimt.bitcast"(%arg23) : (i32) -> ui32
              %421 = ascsimt.math_binary %420, %401, "mulhi" : ui32, ui32 -> ui32
              %422 = "ascsimt.bitcast"(%421) : (ui32) -> i32
              %423 = arith.muli %arg20, %c-1640531527_i32_23 : i32
              %424 = arith.addi %arg14, %423 : i32
              %425 = arith.muli %arg20, %c-1150833019_i32_22 : i32
              %426 = arith.addi %arg15, %425 : i32
              %427 = arith.xori %422, %arg22 : i32
              %428 = arith.xori %427, %424 : i32
              %429 = arith.xori %418, %arg24 : i32
              %430 = arith.xori %429, %426 : i32
              scf.yield %428, %419, %430, %415 : i32, i32, i32, i32
            }
            %407 = arith.muli %arg19, %c4_i64_20 : i64
            %408 = arith.index_cast %407 : i64 to index
            ascsimt.store %arg17[%408], %406#0 : memref<3328xi32, 1>, i32
            %409 = arith.addi %407, %c1_i64_19 : i64
            %410 = arith.index_cast %409 : i64 to index
            ascsimt.store %arg17[%410], %406#1 : memref<3328xi32, 1>, i32
            %411 = arith.addi %407, %c2_i64_18 : i64
            %412 = arith.index_cast %411 : i64 to index
            ascsimt.store %arg17[%412], %406#2 : memref<3328xi32, 1>, i32
            %413 = arith.addi %407, %c3_i64_17 : i64
            %414 = arith.index_cast %413 : i64 to index
            ascsimt.store %arg17[%414], %406#3 : memref<3328xi32, 1>, i32
          } {cannir.vf_threads = 256 : i64}
        } {threads = 256 : i64}
      } else {
        ascsimt.vf(%217, %227, %171, %172, %134, %144, %alloc_11, %173 : i32, i32, i32, i32, i32, i32, memref<3328xi32, 1>, i64) {
        ^bb0(%arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: memref<3328xi32, 1>, %arg16: i64):
          %c256_i64_16 = arith.constant 256 : i64
          %c3_i64_17 = arith.constant 3 : i64
          %c2_i64_18 = arith.constant 2 : i64
          %c1_i64_19 = arith.constant 1 : i64
          %c4_i64_20 = arith.constant 4 : i64
          %c1_i32_21 = arith.constant 1 : i32
          %c10_i32 = arith.constant 10 : i32
          %c0_i32_22 = arith.constant 0 : i32
          %c-1150833019_i32_23 = arith.constant -1150833019 : i32
          %c-1640531527_i32_24 = arith.constant -1640531527 : i32
          %c-845247145_i32 = arith.constant -845247145 : i32
          %c-766435501_i32 = arith.constant -766435501 : i32
          %c262144_i32 = arith.constant 262144 : i32
          %c16384_i32 = arith.constant 16384 : i32
          %322 = ascsimt.thread_idx {ascsimt.scope = 5 : i64, axis = "x", cannir.vf_threads = 256 : i64} : i32
          %323 = arith.extsi %322 {cannir.vf_threads = 256 : i64} : i32 to i64
          scf.for %arg17 = %323 to %arg16 step %c256_i64_16  : i64 {
            %324 = arith.trunci %arg17 : i64 to i32
            %325 = arith.addi %arg9, %324 : i32
            %326 = arith.cmpi ult, %325, %arg9 : i32
            %327 = arith.extui %326 : i1 to i32
            %328 = arith.addi %arg10, %327 : i32
            %329 = arith.divui %325, %c16384_i32 : i32
            %330 = arith.remui %328, %c16384_i32 : i32
            %331 = arith.muli %330, %c262144_i32 : i32
            %332 = arith.addi %329, %331 : i32
            %333 = arith.divui %328, %c16384_i32 : i32
            %334 = arith.addi %arg11, %332 : i32
            %335 = arith.addi %arg12, %333 : i32
            %336 = arith.cmpi ult, %334, %arg11 : i32
            %337 = arith.extui %336 : i1 to i32
            %338 = arith.addi %335, %337 : i32
            %339 = arith.remui %325, %c16384_i32 : i32
            %340:4 = scf.for %arg18 = %c0_i32_22 to %c10_i32 step %c1_i32_21 iter_args(%arg19 = %c0_i32_22, %arg20 = %339, %arg21 = %334, %arg22 = %338) -> (i32, i32, i32, i32)  : i32 {
              %349 = arith.muli %arg19, %c-766435501_i32 : i32
              %350 = "ascsimt.bitcast"(%arg19) : (i32) -> ui32
              %351 = "ascsimt.cast"(%c-766435501_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
              %352 = ascsimt.math_binary %350, %351, "mulhi" : ui32, ui32 -> ui32
              %353 = "ascsimt.bitcast"(%352) : (ui32) -> i32
              %354 = arith.muli %arg21, %c-845247145_i32 : i32
              %355 = "ascsimt.bitcast"(%arg21) : (i32) -> ui32
              %356 = "ascsimt.cast"(%c-845247145_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
              %357 = ascsimt.math_binary %355, %356, "mulhi" : ui32, ui32 -> ui32
              %358 = "ascsimt.bitcast"(%357) : (ui32) -> i32
              %359 = arith.muli %arg18, %c-1640531527_i32_24 : i32
              %360 = arith.addi %arg13, %359 : i32
              %361 = arith.muli %arg18, %c-1150833019_i32_23 : i32
              %362 = arith.addi %arg14, %361 : i32
              %363 = arith.xori %358, %arg20 : i32
              %364 = arith.xori %363, %360 : i32
              %365 = arith.xori %353, %arg22 : i32
              %366 = arith.xori %365, %362 : i32
              scf.yield %364, %354, %366, %349 : i32, i32, i32, i32
            }
            %341 = arith.muli %arg17, %c4_i64_20 : i64
            %342 = arith.index_cast %341 : i64 to index
            ascsimt.store %arg15[%342], %340#0 : memref<3328xi32, 1>, i32
            %343 = arith.addi %341, %c1_i64_19 : i64
            %344 = arith.index_cast %343 : i64 to index
            ascsimt.store %arg15[%344], %340#1 : memref<3328xi32, 1>, i32
            %345 = arith.addi %341, %c2_i64_18 : i64
            %346 = arith.index_cast %345 : i64 to index
            ascsimt.store %arg15[%346], %340#2 : memref<3328xi32, 1>, i32
            %347 = arith.addi %341, %c3_i64_17 : i64
            %348 = arith.index_cast %347 : i64 to index
            ascsimt.store %arg15[%348], %340#3 : memref<3328xi32, 1>, i32
          } {cannir.vf_threads = 256 : i64}
        } {threads = 256 : i64}
      }
      ascvec.get_buf v, %c1_i64, 0 {cannir.buffer.generated_sync}
      %233 = ascscalar.bitcast_memref %alloc_6 byte_offset = %c0_i64 : memref<13312xf32, 1> -> memref<?xui16, 1>
      ascvec.vf(%alloc_10, %alloc_6, %233, %alloc_11, %alloc_8, %179 : memref<832xui8, 1>, memref<13312xf32, 1>, memref<?xui16, 1>, memref<3328xi32, 1>, memref<13312xui8, 1>, i32) {
      ^bb0(%arg9: memref<832xui8, 1>, %arg10: memref<13312xf32, 1>, %arg11: memref<?xui16, 1>, %arg12: memref<3328xi32, 1>, %arg13: memref<13312xui8, 1>, %arg14: i32):
        %c512_i32 = arith.constant 512 : i32
        %c1_i32_16 = arith.constant 1 : i32
        %c2139062143_i32 = arith.constant 2139062143 : i32
        %c-8388608_i32 = arith.constant -8388608 : i32
        %c126_i8 = arith.constant 126 : i8
        %c-1_i8 = arith.constant -1 : i8
        %c56_i8 = arith.constant 56 : i8
        %cst = arith.constant 0.0156249991 : f32
        %c64_i32_17 = arith.constant 64 : i32
        %c254_i32 = arith.constant 254 : i32
        %c-8_i32 = arith.constant -8 : i32
        %c8_i32 = arith.constant 8 : i32
        %c255_i32 = arith.constant 255 : i32
        %c128_i32 = arith.constant 128 : i32
        %c256_i32 = arith.constant 256 : i32
        %c0_18 = arith.constant 0 : index
        %c1_i8 = arith.constant 1 : i8
        %c32_i8 = arith.constant 32 : i8
        %c65535_i32 = arith.constant 65535 : i32
        %c3_i8 = arith.constant 3 : i8
        %c0_i8 = arith.constant 0 : i8
        %c-128_i8 = arith.constant -128 : i8
        %c0_i32_19 = arith.constant 0 : i32
        %322 = ascvec.create_mask : !ascvec.mask
        %323 = ascvec.create_mask elem_bits 8 : !ascvec.mask
        %324 = ascvec.arange %c0_i32_19 : i32 -> <ui32>
        %325 = ascvec.shr %324, 3, %322 : <ui32> -> <ui32>
        %326 = ascvec.duplicate_scalar %c-128_i8, %323 : i8 -> <ui8>
        %327 = ascvec.arange %c0_i8 : i8 -> <ui8>
        %328 = ascvec.duplicate_scalar %c3_i8, %323 : i8 -> <ui8>
        %329 = ascvec.and %327, %328, %323 : <ui8>, <ui8> -> <ui8>
        %330 = ascvec.shl %329, 6, %323 : <ui8> -> <ui8>
        %331 = ascvec.shr %327, 2, %323 : <ui8> -> <ui8>
        %332 = ascvec.or %330, %331, %323 : <ui8>, <ui8> -> <ui8>
        %333 = ascvec.duplicate_scalar %c65535_i32, %322 : i32 -> <ui32>
        %334 = ascvec.cmp_lt_scalar %327, %c32_i8, %323 : <ui8>, i8 -> !ascvec.mask
        %335 = ascvec.cmp_lt_scalar %329, %c1_i8, %323 : <ui8>, i8 -> !ascvec.mask
        %336 = ascvec.mask_and %334, %335, %323
        ascvec.store_unalign_begin %arg9 : memref<832xui8, 1>
        scf.for %arg15 = %c0_i32_19 to %arg14 step %c1_i32_16  : i32 {
          %337 = arith.muli %arg15, %c256_i32 : i32
          %338 = arith.index_cast %337 : i32 to index
          %result0, %result1 = ascvec.loadalign_deintlv %arg10[%338] {dintlv_width = "b32"} : memref<13312xf32, 1>, index -> <f32>, <f32>
          %339 = arith.addi %337, %c128_i32 : i32
          %340 = arith.index_cast %339 : i32 to index
          %result0_21, %result1_22 = ascvec.loadalign_deintlv %arg10[%340] {dintlv_width = "b32"} : memref<13312xf32, 1>, index -> <f32>, <f32>
          %result0_23, %result1_24 = ascvec.deintlv %result0, %result0_21 : <f32>, <f32> -> <f32>, <f32>
          %result0_25, %result1_26 = ascvec.deintlv %result1, %result1_22 : <f32>, <f32> -> <f32>, <f32>
          %341 = ascvec.abs %result0_23, %322 : <f32>
          %342 = ascvec.reinterpret %341 : <f32> -> <ui32>
          %343 = ascvec.abs %result0_25, %322 : <f32>
          %344 = ascvec.reinterpret %343 : <f32> -> <ui32>
          %345 = ascvec.abs %result1_24, %322 : <f32>
          %346 = ascvec.reinterpret %345 : <f32> -> <ui32>
          %347 = ascvec.abs %result1_26, %322 : <f32>
          %348 = ascvec.reinterpret %347 : <f32> -> <ui32>
          %349 = ascvec.max %342, %344, %322 : <ui32>
          %350 = ascvec.max %346, %348, %322 : <ui32>
          %351 = ascvec.max %349, %350, %322 : <ui32>
          %352 = ascvec.reduce_max_datablock %351, %322 : <ui32>
          %353 = arith.muli %arg15, %c512_i32 : i32
          %354 = arith.index_cast %353 : i32 to index
          %result0_27, %result1_28 = ascvec.loadalign_deintlv %arg11[%354] {dintlv_width = "b16"} : memref<?xui16, 1>, index -> <ui16>, <ui16>
          %355 = arith.addi %353, %c256_i32 : i32
          %356 = arith.index_cast %355 : i32 to index
          %result0_29, %result1_30 = ascvec.loadalign_deintlv %arg11[%356] {dintlv_width = "b16"} : memref<?xui16, 1>, index -> <ui16>, <ui16>
          %357 = ascvec.reinterpret %result1_28 : <ui16> -> <ui8>
          %358 = ascvec.reinterpret %result1_30 : <ui16> -> <ui8>
          %result0_31, %result1_32 = ascvec.deintlv %357, %358 : <ui8>, <ui8> -> <ui8>, <ui8>
          %359 = ascvec.and %result1_32, %326, %323 : <ui8>, <ui8> -> <ui8>
          %360 = ascvec.reinterpret %359 : <ui8> -> <ui32>
          %361 = ascvec.shr %352, 23, %322 : <ui32> -> <ui32>
          %362 = ascvec.cmp_ge_scalar %361, %c255_i32, %322 : <ui32>, i32 -> !ascvec.mask
          %363 = ascvec.max_scalar %361, %c8_i32, %322 : <ui32>, i32
          %364 = ascvec.add_scalar %363, %c-8_i32, %322 : <ui32>, i32
          %365 = ascvec.duplicate_scalar %c255_i32, %362 merge %364 : !ascvec.vreg<ui32> {mode = "merging"} : i32 -> <ui32>
          %366 = ascvec.duplicate_scalar %c254_i32, %322 : i32 -> <ui32>
          %367 = ascvec.sub %366, %365, %322 : <ui32>
          %368 = ascvec.cmp_ne_scalar %365, %c0_i32_19, %322 : <ui32>, i32 -> !ascvec.mask
          %369 = ascvec.shl %367, 23, %368 : <ui32> -> <ui32>
          %370 = ascvec.reinterpret %369 : <ui32> -> <f32>
          %371 = ascvec.gather_reg %370, %325 : <f32>, <ui32> -> <f32>
          %372 = ascvec.reinterpret %342 : <ui32> -> <f32>
          %373 = ascvec.mul %372, %371, %322 : <f32>
          %374 = ascvec.reinterpret %344 : <ui32> -> <f32>
          %375 = ascvec.mul %374, %371, %322 : <f32>
          %376 = ascvec.reinterpret %346 : <ui32> -> <f32>
          %377 = ascvec.mul %376, %371, %322 : <f32>
          %378 = ascvec.reinterpret %348 : <ui32> -> <f32>
          %379 = ascvec.mul %378, %371, %322 : <f32>
          %380 = arith.muli %arg15, %c64_i32_17 : i32
          %381 = arith.index_cast %380 : i32 to index
          %382 = ascvec.loadalign %arg12[%381] : memref<3328xi32, 1>, index -> <i32>
          %383 = ascvec.reinterpret %382 : <i32> -> <ui32>
          %384 = ascvec.shl %383, 8, %322 : <ui32> -> <ui32>
          %385 = ascvec.shr %383, 24, %322 : <ui32> -> <ui32>
          %386 = ascvec.or %384, %385, %322 : <ui32>, <ui32> -> <ui32>
          %387 = ascvec.shr %383, 16, %322 : <ui32> -> <ui32>
          %388 = ascvec.and %386, %333, %322 : <ui32>, <ui32> -> <ui32>
          %389 = ascvec.and %383, %333, %322 : <ui32>, <ui32> -> <ui32>
          %390 = ascvec.shr %386, 16, %322 : <ui32> -> <ui32>
          %391 = ascvec.reinterpret %373 : <f32> -> <ui32>
          %392 = ascvec.reinterpret %391 : <ui32> -> <f32>
          %393 = ascvec.max_scalar %392, %cst, %322 : <f32>, f32
          %394 = ascvec.add %392, %393, %322 : <f32>
          %395 = ascvec.reinterpret %394 : <f32> -> <ui32>
          %396 = ascvec.shr %395, 4, %322 : <ui32> -> <ui32>
          %397 = ascvec.add %396, %387, %322 : <ui32>
          %398 = ascvec.reinterpret %375 : <f32> -> <ui32>
          %399 = ascvec.reinterpret %398 : <ui32> -> <f32>
          %400 = ascvec.max_scalar %399, %cst, %322 : <f32>, f32
          %401 = ascvec.add %399, %400, %322 : <f32>
          %402 = ascvec.reinterpret %401 : <f32> -> <ui32>
          %403 = ascvec.shr %402, 4, %322 : <ui32> -> <ui32>
          %404 = ascvec.add %403, %388, %322 : <ui32>
          %405 = ascvec.reinterpret %377 : <f32> -> <ui32>
          %406 = ascvec.reinterpret %405 : <ui32> -> <f32>
          %407 = ascvec.max_scalar %406, %cst, %322 : <f32>, f32
          %408 = ascvec.add %406, %407, %322 : <f32>
          %409 = ascvec.reinterpret %408 : <f32> -> <ui32>
          %410 = ascvec.shr %409, 4, %322 : <ui32> -> <ui32>
          %411 = ascvec.add %410, %389, %322 : <ui32>
          %412 = ascvec.reinterpret %379 : <f32> -> <ui32>
          %413 = ascvec.reinterpret %412 : <ui32> -> <f32>
          %414 = ascvec.max_scalar %413, %cst, %322 : <f32>, f32
          %415 = ascvec.add %413, %414, %322 : <f32>
          %416 = ascvec.reinterpret %415 : <f32> -> <ui32>
          %417 = ascvec.shr %416, 4, %322 : <ui32> -> <ui32>
          %418 = ascvec.add %417, %390, %322 : <ui32>
          %419 = ascvec.reinterpret %397 : <ui32> -> <ui16>
          %420 = ascvec.reinterpret %404 : <ui32> -> <ui16>
          %result0_33, %result1_34 = ascvec.deintlv %419, %420 : <ui16>, <ui16> -> <ui16>, <ui16>
          %421 = ascvec.reinterpret %411 : <ui32> -> <ui16>
          %422 = ascvec.reinterpret %418 : <ui32> -> <ui16>
          %result0_35, %result1_36 = ascvec.deintlv %421, %422 : <ui16>, <ui16> -> <ui16>, <ui16>
          %423 = ascvec.reinterpret %result1_34 : <ui16> -> <ui8>
          %424 = ascvec.reinterpret %result1_36 : <ui16> -> <ui8>
          %result0_37, %result1_38 = ascvec.deintlv %423, %424 : <ui8>, <ui8> -> <ui8>, <ui8>
          %425 = ascvec.gather_reg %result0_37, %332 : <ui8>, <ui8> -> <ui8>
          %426 = ascvec.reinterpret %425 : <ui8> -> <ui32>
          %427 = ascvec.reinterpret %426 : <ui32> -> <ui8>
          %428 = ascvec.add_scalar %427, %c56_i8, %323 : <ui8>, i8
          %429 = ascvec.cmp_ne_scalar %428, %c-1_i8, %323 : <ui8>, i8 -> !ascvec.mask
          %430 = ascvec.min_scalar %428, %c126_i8, %429 : <ui8>, i8
          %431 = ascvec.reinterpret %430 : <ui8> -> <ui32>
          %432 = ascvec.or %431, %360, %322 : <ui32>, <ui32> -> <ui32>
          %433 = ascvec.reinterpret %371 : <f32> -> <ui32>
          %434 = ascvec.cmp_eq_scalar %433, %c-8388608_i32, %322 : <ui32>, i32 -> !ascvec.mask
          %435 = ascvec.duplicate_scalar %c2139062143_i32, %434 merge %432 : !ascvec.vreg<ui32> {mode = "merging"} : i32 -> <ui32>
          %436 = ascvec.reinterpret %435 : <ui32> -> <ui8>
          ascvec.storealign %arg13[%338], %436, %323 : memref<13312xui8, 1>, index, <ui8>
          %437 = ascvec.reinterpret %365 : <ui32> -> <ui8>
          %438 = ascvec.squeeze_store_reg %437, %336 : <ui8>, !ascvec.mask -> <ui8>
          ascvec.store_unalign %arg9[%c0_18], %438 : memref<832xui8, 1>, <ui8>
        }
        %c0_20 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        scf.for %arg15 = %c0_20 to %c1 step %c1 {
          ascvec.store_unalign_post %arg9[%c0_18] : memref<832xui8, 1>
        }
        ascvec.mem_bar vst_vld
      } {cannir.vf_mode = "simd"}
      ascvec.release_buf v, %c1_i64, 0 {cannir.buffer.generated_sync}
      %234 = arith.trunci %85 : i64 to i32
      %235 = arith.remsi %234, %c64_i32 : i32
      %236 = arith.cmpi ne, %235, %c0_i32 : i32
      %237 = arith.cmpi slt, %235, %c0_i32 : i32
      %238 = arith.andi %236, %237 : i1
      %239 = arith.divsi %234, %c64_i32 : i32
      %240 = arith.extui %238 : i1 to i32
      %241 = arith.subi %239, %240 : i32
      ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
      ascvec.get_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
      %242 = ascscalar.bitcast_memref %alloc_13 byte_offset = %c0_i64 : memref<832xui8, 1> -> memref<?xui16, 1>
      %243 = ascscalar.bitcast_memref %alloc_10 byte_offset = %c0_i64 : memref<832xui8, 1> -> memref<?xui16, 1>
      ascvec.vf(%alloc_12, %alloc_8, %alloc_14, %234, %242, %243, %alloc_15, %241 : memref<13312xui8, 1>, memref<13312xui8, 1>, memref<26624xui8, 1>, i32, memref<?xui16, 1>, memref<?xui16, 1>, memref<832xui16, 1>, i32) {
      ^bb0(%arg9: memref<13312xui8, 1>, %arg10: memref<13312xui8, 1>, %arg11: memref<26624xui8, 1>, %arg12: i32, %arg13: memref<?xui16, 1>, %arg14: memref<?xui16, 1>, %arg15: memref<832xui16, 1>, %arg16: i32):
        %c128_i32 = arith.constant 128 : i32
        %c256_i32 = arith.constant 256 : i32
        %c0_i32_16 = arith.constant 0 : i32
        %c2_i32 = arith.constant 2 : i32
        scf.for %arg17 = %c0_i32_16 to %arg12 step %c256_i32  : i32 {
          %322 = arith.index_cast %arg17 : i32 to index
          %323 = ascvec.loadalign %arg9[%322] : memref<13312xui8, 1>, index -> <ui8>
          %324 = ascvec.loadalign %arg10[%322] : memref<13312xui8, 1>, index -> <ui8>
          %325 = arith.muli %arg17, %c2_i32 : i32
          %326 = arith.index_cast %325 : i32 to index
          ascvec.storealign_intlv %arg11[%326], %323, %324 : memref<26624xui8, 1>, <ui8>, <ui8>
        }
        scf.for %arg17 = %c0_i32_16 to %arg16 step %c128_i32  : i32 {
          %322 = arith.index_cast %arg17 : i32 to index
          %323 = ascvec.loadalign %arg13[%322] : memref<?xui16, 1>, index -> <ui16>
          %324 = ascvec.loadalign %arg14[%322] : memref<?xui16, 1>, index -> <ui16>
          %325 = arith.muli %arg17, %c2_i32 : i32
          %326 = arith.index_cast %325 : i32 to index
          ascvec.storealign_intlv %arg15[%326], %323, %324 : memref<832xui16, 1>, <ui16>, <ui16>
        }
      } {cannir.vf_mode = "simd"}
      ascvec.release_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
      ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
      %244 = arith.muli %55, %c512_i64 : i64
      %245 = cannir.extract_layout %arg2 : !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
      %246 = cannir.extract_shape %245 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, -1)>
      %247 = cannir.extract_dim %246[0] : !cannir.shape<(-1, -1)>
      %248 = arith.minsi %79, %247 : i64
      %249 = arith.subi %247, %248 : i64
      %250 = arith.minsi %249, %c1_i64 : i64
      %251 = arith.cmpi sle, %79, %247 : i64
      %252 = arith.select %251, %250, %c0_i64 : i64
      %253 = arith.cmpi sgt, %252, %c1_i64 : i64
      %254 = arith.select %253, %c1_i64, %252 : i64
      %255 = cannir.extract_dim %246[1] : !cannir.shape<(-1, -1)>
      %256 = arith.maxsi %244, %c1_i64 : i64
      %257 = arith.divsi %255, %256 : i64
      %258 = arith.minsi %81, %257 : i64
      %259 = arith.muli %258, %244 : i64
      %260 = arith.subi %255, %259 : i64
      %261 = arith.minsi %244, %260 : i64
      %262 = arith.cmpi sle, %81, %257 : i64
      %263 = arith.select %262, %261, %c0_i64 : i64
      %264 = arith.cmpi sgt, %263, %c26624_i64 : i64
      %265 = arith.select %264, %c26624_i64, %263 : i64
      %266 = cannir.extract_physical_layout %arg2 : !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
      %267 = cannir.extract_stride %266 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
      %268 = cannir.extract_dim %267[0] : !cannir.stride<(-1, 1)>
      %269 = arith.muli %79, %268 : i64
      %270 = arith.muli %81, %244 : i64
      %271 = arith.addi %269, %270 : i64
      %272 = arith.muli %271, %c8_i64 : i64
      %273 = ascscalar.memref_address %alloc_1 : memref<1xui8>
      %274 = arith.divui %272, %c8_i64 : i64
      %275 = arith.addi %273, %274 : i64
      %276 = ascscalar.bind_gm_ptr %275 : memref<1xui8>
      %277 = ascscalar.bitcast_memref %alloc_14 byte_offset = %c0_i64 : memref<26624xui8, 1> -> memref<?xui8, 1>
      %278 = cannir.extract_stride %245 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
      %279 = cannir.extract_dim %278[0] : !cannir.stride<(-1, 1)>
      %280 = arith.cmpi ugt, %254, %c0_i64 : i64
      %281 = arith.cmpi ugt, %265, %c0_i64 : i64
      %282 = arith.andi %280, %281 : i1
      scf.if %282 {
        ascvec.get_buf mte3, %c2_i64, 0 {cannir.buffer.generated_sync}
        %322 = arith.trunci %254 : i64 to i32
        %323 = arith.trunci %265 : i64 to i32
        ascvec.copy_ub2gm %276[%c0], %277[%c0], %322, %323, %279, %c26624_i32 l2_cache_ctl %c0_i32 : memref<1xui8>, memref<?xui8, 1>
        ascvec.release_buf mte3, %c2_i64, 0 {cannir.buffer.generated_sync}
      }
      %283 = arith.muli %55, %c16_i64 : i64
      %284 = cannir.extract_layout %arg3 : !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
      %285 = cannir.extract_shape %284 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, -1)>
      %286 = cannir.extract_dim %285[0] : !cannir.shape<(-1, -1)>
      %287 = arith.minsi %79, %286 : i64
      %288 = arith.subi %286, %287 : i64
      %289 = arith.minsi %288, %c1_i64 : i64
      %290 = arith.cmpi sle, %79, %286 : i64
      %291 = arith.select %290, %289, %c0_i64 : i64
      %292 = arith.cmpi sgt, %291, %c1_i64 : i64
      %293 = arith.select %292, %c1_i64, %291 : i64
      %294 = cannir.extract_dim %285[1] : !cannir.shape<(-1, -1)>
      %295 = arith.maxsi %283, %c1_i64 : i64
      %296 = arith.divsi %294, %295 : i64
      %297 = arith.minsi %81, %296 : i64
      %298 = arith.muli %297, %283 : i64
      %299 = arith.subi %294, %298 : i64
      %300 = arith.minsi %283, %299 : i64
      %301 = arith.cmpi sle, %81, %296 : i64
      %302 = arith.select %301, %300, %c0_i64 : i64
      %303 = arith.cmpi sgt, %302, %c1664_i64 : i64
      %304 = arith.select %303, %c1664_i64, %302 : i64
      %305 = cannir.extract_physical_layout %arg3 : !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
      %306 = cannir.extract_stride %305 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
      %307 = cannir.extract_dim %306[0] : !cannir.stride<(-1, 1)>
      %308 = arith.muli %79, %307 : i64
      %309 = arith.muli %81, %283 : i64
      %310 = arith.addi %308, %309 : i64
      %311 = arith.muli %310, %c8_i64 : i64
      %312 = ascscalar.memref_address %alloc_2 : memref<1xui8>
      %313 = arith.divui %311, %c8_i64 : i64
      %314 = arith.addi %312, %313 : i64
      %315 = ascscalar.bind_gm_ptr %314 : memref<1xui8>
      %316 = ascscalar.bitcast_memref %alloc_15 byte_offset = %c0_i64 : memref<832xui16, 1> -> memref<?xui8, 1>
      %317 = cannir.extract_stride %284 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
      %318 = cannir.extract_dim %317[0] : !cannir.stride<(-1, 1)>
      %319 = arith.cmpi ugt, %293, %c0_i64 : i64
      %320 = arith.cmpi ugt, %304, %c0_i64 : i64
      %321 = arith.andi %319, %320 : i1
      scf.if %321 {
        ascvec.get_buf mte3, %c3_i64, 0 {cannir.buffer.generated_sync}
        %322 = arith.trunci %293 : i64 to i32
        %323 = arith.trunci %304 : i64 to i32
        ascvec.copy_ub2gm %315[%c0], %316[%c0], %322, %323, %318, %c1664_i32 l2_cache_ctl %c0_i32 : memref<1xui8>, memref<?xui8, 1>
        ascvec.release_buf mte3, %c3_i64, 0 {cannir.buffer.generated_sync}
      }
    }
    return
  }
}
